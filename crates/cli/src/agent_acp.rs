//! `greppy agent stdio` — Agent Client Protocol over newline-delimited JSON-RPC.
//!
//! Workjet's ACP client encodes Effect JSON-RPC: one JSON object per line,
//! notifications use an empty-string `id`, and a typed failure is a JSON-RPC
//! error whose `_tag` is `Cause` and whose `data` is a one-element `Fail`
//! cause. Stdout carries only those frames. Diagnostics stay on stderr.
//!
//! The handler drives the existing agent loop. Each session keeps its own
//! transcript, cancel flag, and permission memory, and tool execution uses
//! the working folder the client sent with `session/new` or `session/load`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use clap::Parser;
use greppy_agent::{
    run_agent_loop_with_content, AgentConfig, BuiltinPromptMode, Client, ClientError, ExecutionEnv,
    GreppyEnv, LoopError, LoopEvent, LoopStop, Message, ModelRequest, ModelStream,
    OutputTokenLimit, StreamEvent, ToolOutcome, TurnResult,
};
use serde_json::{json, Value};

#[cfg(test)]
#[path = "agent_acp_tests.rs"]
mod tests;

use crate::agent::{EXIT_OK, EXIT_USAGE};
use crate::agent_tui::{
    messages_from_protocol, new_session_id, protocol_from_persisted, ImportAck, SessionRecord,
    SessionStore,
};

const PROTOCOL_VERSION: u64 = 1;
const AUTH_METHOD_ID: &str = "greppy.env";
const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8317";
const DEFAULT_MAX_TURNS: usize = 0;
const OUTPUT_LIMIT_METHOD: &str = "_greppy/set_output_limit";
/// The request with which greppy has its client run a tool the client offered with a
/// prompt (`_meta.greppy.clientTools`).
const CLIENT_TOOL_METHOD: &str = "_greppy/client_tool";
/// The smallest model window a client may name; below it the window is not taken (the
/// instructions and tools alone take nearly that).
const SMALLEST_WINDOW: u64 = 16_000;
/// At most this many client tools go with one prompt.
const CLIENT_TOOLS_LIMIT: usize = 64;

thread_local! {
    static ACTIVE_TOOL_CALL: RefCell<Option<String>> = const { RefCell::new(None) };
}

type ToolEnvFactory = Arc<dyn Fn(&Path) -> Box<dyn ExecutionEnv + Send> + Send + Sync>;
#[cfg(test)]
type PersistenceFault = Arc<dyn Fn() -> io::Result<()> + Send + Sync>;

/// Process-wide ACP settings. The tool factory is optional so tests can
/// substitute a sandbox that records calls without invoking greppy.
#[derive(Clone)]
pub(crate) struct AcpConfig {
    pub endpoint: String,
    pub model: String,
    /// Verified response limit for the initial model at this endpoint, not a global cap.
    pub max_output_tokens: Option<OutputTokenLimit>,
    pub api_key: Option<String>,
    /// Further headers sent with every model request (`--header NAME:VALUE`).
    pub headers: Vec<(String, String)>,
    /// Confine every tool child (`--sandbox`): it may write only in the session's
    /// folder and its run's own temporary folder, and has no network. Where that
    /// cannot be made, tools are refused rather than run unconfined.
    pub sandbox: bool,
    pub max_turns: usize,
    pub capabilities: greppy_agent::CapabilitySet,
    pub data_root: Option<PathBuf>,
    #[cfg(test)]
    pub after_messages: Option<PersistenceFault>,
    #[cfg(test)]
    pub after_import_rename: Option<PersistenceFault>,
    pub tool_env: Option<ToolEnvFactory>,
}

impl Default for AcpConfig {
    fn default() -> Self {
        Self {
            endpoint: std::env::var("GREPPY_ENDPOINT")
                .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string()),
            model: std::env::var("GREPPY_MODEL").unwrap_or_default(),
            max_output_tokens: None,
            api_key: std::env::var("GREPPY_API_KEY")
                .ok()
                .filter(|key| !key.is_empty()),
            headers: Vec::new(),
            sandbox: false,
            max_turns: DEFAULT_MAX_TURNS,
            capabilities: greppy_agent::CapabilitySet::default(),
            data_root: None,
            #[cfg(test)]
            after_messages: None,
            #[cfg(test)]
            after_import_rename: None,
            tool_env: None,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "greppy agent stdio",
    about = "Speak the Agent Client Protocol on stdin/stdout.",
    disable_version_flag = true
)]
struct StdioArgs {
    /// Initial model id. `session/set_model` can replace it.
    #[arg(long, env = "GREPPY_MODEL")]
    model: Option<String>,
    /// Anthropic-Messages gateway base URL.
    #[arg(long, env = "GREPPY_ENDPOINT", default_value = DEFAULT_ENDPOINT)]
    endpoint: String,
    /// Caller-verified response limit for the initial --model at this endpoint.
    /// Model changes discard it; a new route requires freshly verified metadata.
    #[arg(long, value_name = "N", requires = "model")]
    max_output_tokens: Option<OutputTokenLimit>,
    /// Maximum assistant turns for one prompt.
    #[arg(long, default_value_t = DEFAULT_MAX_TURNS, value_name = "N")]
    max_turns: usize,
    /// Include a capability manual from the first turn (`web`, `desktop`, or both).
    #[arg(long, value_name = "web|desktop")]
    enable: Vec<String>,
    /// Send `NAME:VALUE` with every model request (repeatable), e.g. a router's
    /// provider header. Never a secret: the key comes from GREPPY_API_KEY.
    #[arg(long = "header", value_name = "NAME:VALUE")]
    headers: Vec<String>,
    /// Confine every command and greppy call: writes only in the session's folder
    /// and the run's temporary folder, no network. Fail-closed: where the sandbox
    /// cannot be made (no Landlock ≥ V3, no Seatbelt), tools are refused.
    #[arg(long)]
    sandbox: bool,
}

/// `NAME:VALUE` (spaces around the value trimmed) as a header pair.
fn header_pair(said: &str) -> Result<(String, String), String> {
    let (name, value) = said
        .split_once(':')
        .ok_or_else(|| format!("--header {said:?} is not NAME:VALUE"))?;
    let (name, value) = (name.trim(), value.trim());
    // The same check the client makes, so a bad header fails at start, not per request.
    greppy_agent::Client::new(DEFAULT_ENDPOINT, "")
        .with_header(name, value)
        .map_err(|error| format!("--header: {error}"))?;
    Ok((name.to_owned(), value.to_owned()))
}

/// Parse and serve `greppy agent stdio`.
pub(crate) fn run(rest: &[std::ffi::OsString]) -> u8 {
    if std::env::var_os(greppy_agent::AGENT_RUN_ENV).is_some() {
        eprintln!(
            "greppy agent stdio: refusing a nested agent run — you are already inside an agent"
        );
        return EXIT_USAGE;
    }
    let mut argv = vec![std::ffi::OsString::from("greppy agent stdio")];
    argv.extend(rest.iter().skip(2).cloned());
    let args = match StdioArgs::try_parse_from(argv) {
        Ok(args) => args,
        Err(error) => {
            use clap::error::ErrorKind;
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) {
                let _ = error.print();
                return EXIT_OK;
            }
            let _ = error.print();
            return EXIT_USAGE;
        }
    };
    let capabilities = match greppy_agent::CapabilitySet::parse_list(&args.enable) {
        Ok(capabilities) => capabilities,
        Err(error) => {
            eprintln!("greppy agent stdio: {error}");
            return EXIT_USAGE;
        }
    };
    let headers = match args
        .headers
        .iter()
        .map(|said| header_pair(said))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(headers) => headers,
        Err(error) => {
            eprintln!("greppy agent stdio: {error}");
            return EXIT_USAGE;
        }
    };
    let defaults = AcpConfig::default();
    let config = AcpConfig {
        endpoint: args.endpoint,
        headers,
        sandbox: args.sandbox,
        model: args.model.unwrap_or(defaults.model),
        max_output_tokens: args.max_output_tokens,
        max_turns: args.max_turns,
        capabilities,
        ..defaults
    };
    let stdin = io::stdin();
    serve(stdin.lock(), io::stdout(), config)
}

/// Read ACP frames from `input` and write frames to `output`.
pub(crate) fn serve<R, W>(input: R, output: W, config: AcpConfig) -> u8
where
    R: BufRead,
    W: Write + Send + 'static,
{
    let server = Server::new(output, config);
    let code = server.run(input);
    server.shutdown();
    server.wait_idle();
    code
}

struct Server {
    data_root: PathBuf,
    out: Out,
    state: Arc<Mutex<State>>,
    pending: Arc<Mutex<HashMap<String, PendingPermission>>>,
    next_request_id: Arc<AtomicU64>,
    config: AcpConfig,
}

struct State {
    initialized: bool,
    sessions: HashMap<String, Session>,
    /// What the client said it can do (`clientCapabilities` of `initialize`).
    client: Value,
    /// The session of each `session/prompt` under way, by its request id (for
    /// `$/cancel_request`).
    prompts: HashMap<String, String>,
}

struct Session {
    id: String,
    cwd: PathBuf,
    data_root: PathBuf,
    project: String,
    model: String,
    messages: Vec<Message>,
    max_output_tokens: Option<OutputTokenLimit>,
    import_ack: Vec<ImportAck>,
    cancel: Arc<AtomicBool>,

    busy: bool,
    closed: bool,
    perms: Arc<Mutex<PermMemory>>,
    usage_in: u64,
    usage_out: u64,
    /// The MCP servers the client named for this session, started once.
    mcp: Arc<Mutex<greppy_agent::mcp::McpHub>>,
    /// The summary of what was compacted away, where the conversation was.
    context_summary: Option<String>,
    /// What the client chose for this session (config options and mode).
    choices: Choices,
    /// Further folders the client named (`additionalDirectories`): writable like the
    /// session's folder, and named to the model.
    extra_dirs: Vec<PathBuf>,
}

/// What a client chose for a session: how it asks before tools run, how hard the
/// model is to think, and which of greppy's capabilities are on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Choices {
    /// `ask` (permission before commands and edits), `auto` (no asking), `read-only`.
    mode: &'static str,
    /// Reasoning effort; none: the provider's default.
    effort: Option<String>,
    capabilities: greppy_agent::CapabilitySet,
}

const MODES: [(&str, &str, &str); 3] = [
    (
        "ask",
        "Ask",
        "Ask before running commands and changing files",
    ),
    (
        "auto",
        "Auto",
        "Run commands and change files without asking",
    ),
    (
        "read-only",
        "Read only",
        "Read and search only; no commands, no changes",
    ),
];

struct PermMemory {
    allow: HashSet<String>,
    reject: HashSet<String>,
}

/// One in-flight `session/request_permission`. Dropping `sender` unblocks the
/// tool gate with a denial, which is how cancel clears a pending decision.
struct PendingPermission {
    session_id: String,
    sender: Sender<Value>,
}

struct Out {
    inner: Arc<Mutex<dyn Write + Send>>,
}

impl Out {
    fn send(&self, value: &Value) -> io::Result<()> {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        serde_json::to_writer(&mut *guard, value)?;
        guard.write_all(b"\n")?;
        guard.flush()
    }
}

impl Server {
    fn new<W>(output: W, config: AcpConfig) -> Self
    where
        W: Write + Send + 'static,
    {
        let data_root = config
            .data_root
            .clone()
            .unwrap_or_else(greppy_core::cache::data_root);
        Self {
            data_root,
            out: Out {
                inner: Arc::new(Mutex::new(output)),
            },
            state: Arc::new(Mutex::new(State {
                initialized: false,
                sessions: HashMap::new(),
                client: Value::Null,
                prompts: HashMap::new(),
            })),
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_request_id: Arc::new(AtomicU64::new(1)),
            config,
        }
    }

    fn run<R: BufRead>(&self, mut input: R) -> u8 {
        let mut buf = Vec::new();
        loop {
            buf.clear();
            let read = match input.read_until(b'\n', &mut buf) {
                Ok(n) => n,
                Err(error) => {
                    eprintln!("greppy agent stdio: stdin read failed: {error}");
                    return EXIT_USAGE;
                }
            };
            if read == 0 {
                return EXIT_OK;
            }
            if buf.last() == Some(&b'\n') {
                buf.pop();
            }
            if buf.last() == Some(&b'\r') {
                buf.pop();
            }
            if buf.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let line = match std::str::from_utf8(&buf) {
                Ok(line) => line.to_string(),
                Err(_) => {
                    let _ = self
                        .out
                        .send(&rpc_error(&Value::Null, -32700, "Parse error"));
                    continue;
                }
            };
            self.dispatch_line(&line);
        }
    }

    fn wait_idle(&self) {
        loop {
            let busy = match self.state.lock() {
                Ok(state) => state.sessions.values().any(|session| session.busy),
                Err(poisoned) => poisoned
                    .into_inner()
                    .sessions
                    .values()
                    .any(|session| session.busy),
            };
            if !busy {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn dispatch_line(&self, line: &str) {
        let message: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => {
                let _ = self
                    .out
                    .send(&rpc_error(&Value::Null, -32700, "Parse error"));
                return;
            }
        };
        if !message.is_object() {
            let _ = self
                .out
                .send(&rpc_error(&Value::Null, -32600, "Invalid request"));
            return;
        }
        if message.get("method").is_none() {
            self.complete_response(&message);
            return;
        }
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if method.starts_with("@effect/rpc/") && is_notification(&message) {
            return;
        }
        if is_notification(&message) {
            self.handle_notification(
                &method,
                message.get("params").cloned().unwrap_or(Value::Null),
            );
            return;
        }
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        if let Some(reply) = self.handle_request(&method, &id, &params) {
            let _ = self.out.send(&reply);
            if matches!(
                method.as_str(),
                "session/new" | "session/load" | "session/resume"
            ) && reply.get("error").is_none()
            {
                let session_id = reply
                    .pointer("/result/sessionId")
                    .or_else(|| params.get("sessionId"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let Some(session_id) = session_id {
                    self.after_session_setup(&session_id);
                }
            }
        }
    }

    /// What a client is told once a session is set up: the slash commands it can send,
    /// and, where it takes notices, what went wrong while setting up (an MCP server
    /// left out).
    fn after_session_setup(&self, session_id: &str) {
        let _ = send_update(
            &self.out,
            session_id,
            json!({
                "sessionUpdate": "available_commands_update",
                "availableCommands": SLASH_COMMANDS.iter().map(|(name, description)| json!({
                    "name": name, "description": description
                })).collect::<Vec<_>>()
            }),
        );
        let (notices, hub) = {
            let state = lock_state(&self.state);
            (
                state
                    .client
                    .pointer("/session/notices")
                    .is_some_and(|v| !v.is_null() && v != &json!(false)),
                state
                    .sessions
                    .get(session_id)
                    .map(|session| Arc::clone(&session.mcp)),
            )
        };
        if let (true, Some(hub)) = (notices, hub) {
            let notes = hub.lock().map(|hub| hub.notes.clone()).unwrap_or_default();
            for note in notes {
                let _ = send_update(
                    &self.out,
                    session_id,
                    json!({
                        "sessionUpdate": "notice",
                        "severity": "warning",
                        "title": "An MCP server is left out",
                        "description": note
                    }),
                );
            }
        }
    }

    fn complete_response(&self, message: &Value) {
        let Some(id) = message.get("id") else {
            return;
        };
        let key = id_key(id);
        let sender = match self.pending.lock() {
            Ok(mut pending) => pending.remove(&key),
            Err(poisoned) => poisoned.into_inner().remove(&key),
        };
        if let Some(pending) = sender {
            let _ = pending.sender.send(message.clone());
        }
    }

    fn handle_notification(&self, method: &str, params: Value) {
        if method == "session/cancel" {
            if let Some(session_id) = params.get("sessionId").and_then(Value::as_str) {
                self.cancel_session(session_id);
            }
        }
        // A request the client no longer wants answered: a prompt under way is
        // cancelled as with session/cancel (it ends with stopReason cancelled).
        if method == "$/cancel_request" {
            let session_id = params
                .get("requestId")
                .map(id_key)
                .and_then(|key| lock_state(&self.state).prompts.get(&key).cloned());
            if let Some(session_id) = session_id {
                self.cancel_session(&session_id);
            }
        }
    }

    fn shutdown(&self) {
        let ids: Vec<String> = lock_state(&self.state).sessions.keys().cloned().collect();
        for id in ids {
            self.cancel_session(&id);
        }
    }

    fn cancel_session(&self, session_id: &str) {
        let flag = {
            let state = lock_state(&self.state);
            state
                .sessions
                .get(session_id)
                .map(|session| Arc::clone(&session.cancel))
        };
        if let Some(flag) = flag {
            flag.store(true, Ordering::Relaxed);
        }
        let mut pending = match self.pending.lock() {
            Ok(pending) => pending,
            Err(poisoned) => poisoned.into_inner(),
        };
        pending.retain(|_, permission| permission.session_id != session_id);
    }

    /// Returns a response frame, or `None` when the request stays open
    /// (`session/prompt` answers from its worker).
    fn handle_request(&self, method: &str, id: &Value, params: &Value) -> Option<Value> {
        if method != "initialize" && !lock_state(&self.state).initialized {
            return Some(rpc_error(id, -32600, "ACP connection is not initialized"));
        }
        match method {
            "initialize" => Some(self.initialize(id, params)),
            "authenticate" => Some(self.authenticate(id, params)),
            "session/new" => Some(self.session_new(id, params)),
            "session/load" | "session/resume" => Some(self.session_load(id, params, method)),
            "session/list" => Some(self.session_list(id, params)),
            "session/close" => Some(self.session_close(id, params)),
            "session/set_model" => Some(self.session_set_model(id, params)),
            "session/set_mode" => Some(self.session_set_mode(id, params)),
            "session/delete" => Some(self.session_delete(id, params)),
            "_workjet/import_history" => Some(self.import_history(id, params)),
            OUTPUT_LIMIT_METHOD => Some(self.set_route_output_limit(id, params)),

            "session/set_config_option" => Some(self.session_set_config(id, params)),
            "session/prompt" => {
                self.session_prompt(id, params);
                None
            }
            other => Some(rpc_error(id, -32601, &format!("Method not found: {other}"))),
        }
    }

    fn initialize(&self, id: &Value, params: &Value) -> Value {
        let version = params.get("protocolVersion").and_then(Value::as_u64);
        if version.is_none() {
            return rpc_error(id, -32602, "protocolVersion is required");
        }
        {
            let mut state = lock_state(&self.state);
            state.initialized = true;
            state.client = params
                .get("clientCapabilities")
                .cloned()
                .unwrap_or(Value::Null);
        }
        rpc_ok(
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "agentInfo": {
                    "name": "greppy",
                    "title": "Greppy",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "agentCapabilities": {
                    "_meta": {
                        "workjetImportHistory": {"version": 1},
                        "greppyOutputLimit": {"version": 1, "method": OUTPUT_LIMIT_METHOD},
                        "greppyClientTools": {"version": 1, "method": CLIENT_TOOL_METHOD},
                        "greppyPromptEffort": {"version": 1, "values": ["low", "medium", "high"]}
                    },
                    "loadSession": true,

                    "promptCapabilities": {
                        "image": true,
                        "audio": false,
                        "embeddedContext": true
                    },
                    "mcpCapabilities": {
                        "http": true,
                        "sse": false
                    },
                    "sessionCapabilities": {
                        "list": {},
                        "resume": {},
                        "close": {},
                        "delete": {},
                        "additionalDirectories": {}
                    }
                },
                "authMethods": [{
                    "id": AUTH_METHOD_ID,
                    "name": "Local gateway environment",
                    "description": "Uses GREPPY_ENDPOINT, GREPPY_MODEL, and GREPPY_API_KEY from the agent process. ACP does not accept a secret."
                }]
            }),
        )
    }

    fn authenticate(&self, id: &Value, params: &Value) -> Value {
        match params.get("methodId").and_then(Value::as_str) {
            Some(AUTH_METHOD_ID) => rpc_ok(id, json!({})),
            Some(other) => rpc_error(
                id,
                -32602,
                &format!("unsupported authentication method: {other}"),
            ),
            None => rpc_error(id, -32602, "methodId is required"),
        }
    }

    fn import_history(&self, id: &Value, params: &Value) -> Value {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        let Some(input) = params.get("messages").and_then(Value::as_array) else {
            return rpc_error(id, -32602, "messages must be an array");
        };
        let mut ids = HashSet::new();
        let mut acknowledgements = Vec::with_capacity(input.len());
        let mut imported = Vec::with_capacity(input.len());
        for message in input {
            let (Some(message_id), Some(role), Some(text)) = (
                message.get("id").and_then(Value::as_str),
                message.get("role").and_then(Value::as_str),
                message.get("text").and_then(Value::as_str),
            ) else {
                return rpc_error(
                    id,
                    -32602,
                    "imported messages require id, role and text strings",
                );
            };
            if message_id.trim().is_empty() || message_id.len() > 512 || !ids.insert(message_id) {
                return rpc_error(
                    id,
                    -32602,
                    "imported message ids must be nonempty and unique",
                );
            }
            let native_role = match role {
                "user" => greppy_agent::Role::User,
                "assistant" => greppy_agent::Role::Assistant,
                _ => {
                    return rpc_error(
                        id,
                        -32602,
                        "imported message role must be user or assistant",
                    )
                }
            };
            acknowledgements.push(ImportAck::new(message_id.to_owned(), role.to_owned(), text));
            imported.push(Message {
                role: native_role,
                content: vec![greppy_agent::ContentPart::Text {
                    text: text.to_owned(),
                }],
            });
        }
        let mut state = lock_state(&self.state);
        let Some(session) = state.sessions.get_mut(session_id) else {
            return rpc_error(id, -32002, "session not found");
        };
        if session.closed || session.busy {
            return rpc_error(id, -32600, "session is closed or busy");
        }
        if !acknowledgements.starts_with(&session.import_ack) {
            return rpc_error(
                id,
                -32602,
                "imported history changed or lost its acknowledged prefix",
            );
        }
        let accepted = acknowledgements
            .iter()
            .map(|ack| ack.id.clone())
            .collect::<Vec<_>>();
        let mut history = session.messages.clone();
        history.extend(imported.into_iter().skip(session.import_ack.len()));
        let store = SessionStore::new(&session.data_root, &session.project);
        let saved = store.commit_import_history(
            session_id,
            crate::agent_tui::ImportHistoryCommit {
                expected: &messages_from_protocol(&session.messages),
                expected_ack: &session.import_ack,
                messages: &messages_from_protocol(&history),
                acknowledgements: &acknowledgements,
            },
            || {
                #[cfg(test)]
                if let Some(hook) = &self.config.after_messages {
                    hook()?;
                }
                Ok(())
            },
            || {
                #[cfg(test)]
                if let Some(hook) = &self.config.after_import_rename {
                    hook()?;
                }
                Ok(())
            },
        );
        if let Err(error) = saved {
            if error.renamed {
                // The replacement is visible but durability was not confirmed.
                // Do not prompt with stale live state or acknowledge IDs. Loading
                // the saved session reestablishes a single authoritative history.
                session.closed = true;
                return rpc_error(id, -32603, &format!(
                    "imported history durability was not confirmed; reload the session before continuing: {error}"
                ));
            }
            return rpc_error(
                id,
                -32603,
                &format!("cannot persist imported history: {error}"),
            );
        }
        session.messages = history;
        session.import_ack = acknowledgements;
        rpc_ok(id, json!({"acceptedMessageIds": accepted}))
    }

    fn session_new(&self, id: &Value, params: &Value) -> Value {
        let servers = match mcp_servers(params, true) {
            Ok(servers) => servers,
            Err(message) => return rpc_error(id, -32602, &message),
        };
        let cwd = match require_cwd(params) {
            Ok(cwd) => cwd,
            Err(message) => return rpc_error(id, -32602, &message),
        };
        let extra_dirs = match additional_directories(params) {
            Ok(dirs) => dirs,
            Err(message) => return rpc_error(id, -32602, &message),
        };
        let (data_root, project) = self.store_identity(&cwd);
        let model = self.config.model.clone();
        let session_id = unique_session_id();
        let mut record = SessionRecord::new(
            session_id.clone(),
            project.clone(),
            model.clone(),
            session_id.clone(),
        );
        record.source = "acp".to_string();
        record.worktree = cwd.display().to_string();
        let store = SessionStore::new(data_root.as_path(), project.as_str());
        if let Err(error) = store.create(&record) {
            return rpc_error(id, -32603, &format!("cannot persist session: {error}"));
        }
        let mcp = start_mcp(&servers, &cwd);
        let session = Session {
            id: session_id.clone(),
            cwd,
            data_root,
            project,
            model: model.clone(),
            messages: Vec::new(),
            max_output_tokens: self.config.max_output_tokens,
            import_ack: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),

            busy: false,
            closed: false,
            perms: Arc::new(Mutex::new(PermMemory {
                allow: HashSet::new(),
                reject: HashSet::new(),
            })),
            usage_in: 0,
            usage_out: 0,
            mcp,
            context_summary: None,
            choices: Choices {
                mode: "ask",
                effort: None,
                capabilities: self.config.capabilities,
            },
            extra_dirs: extra_dirs.clone(),
        };
        lock_state(&self.state)
            .sessions
            .insert(session_id.clone(), session);
        let choices = lock_state(&self.state).sessions[&session_id]
            .choices
            .clone();
        rpc_ok(id, session_setup(&session_id, &model, true, &choices))
    }

    fn session_load(&self, id: &Value, params: &Value, method: &str) -> Value {
        let mcp_required = method == "session/load";
        let servers = match mcp_servers(params, mcp_required) {
            Ok(servers) => servers,
            Err(message) => return rpc_error(id, -32602, &message),
        };
        let cwd = match require_cwd(params) {
            Ok(cwd) => cwd,
            Err(message) => return rpc_error(id, -32602, &message),
        };
        let extra_dirs = match additional_directories(params) {
            Ok(dirs) => dirs,
            Err(message) => return rpc_error(id, -32602, &message),
        };
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        if !SessionStore::is_valid_id(session_id) {
            return rpc_error(id, -32602, "invalid session id");
        }
        let (data_root, project) = self.store_identity(&cwd);
        let store = SessionStore::new(data_root.as_path(), project.as_str());
        let record = match store.load(session_id) {
            Ok(record) => record,
            Err(_) => return rpc_error(id, -32002, &format!("session not found: {session_id}")),
        };
        {
            let state = lock_state(&self.state);
            if state
                .sessions
                .get(session_id)
                .is_some_and(|session| session.busy)
            {
                return rpc_error(id, -32600, "session is busy");
            }
        }
        let messages = protocol_from_persisted(&record.messages);
        // `session/load` replays the conversation to the client; `session/resume`
        // restores it without a replay.
        if method == "session/load" {
            self.emit_replay(session_id, &messages);
        }
        let model = if record.model.is_empty() {
            self.config.model.clone()
        } else {
            record.model.clone()
        };
        let mcp = start_mcp(&servers, &cwd);
        let session = Session {
            id: session_id.to_string(),
            cwd,
            data_root,
            project,
            model: model.clone(),
            messages,
            // Never revive launch metadata while loading persisted history.
            // The caller can supply freshly verified metadata via session config.
            max_output_tokens: None,
            import_ack: record.import_ack,
            cancel: Arc::new(AtomicBool::new(false)),

            busy: false,
            closed: false,
            perms: Arc::new(Mutex::new(PermMemory {
                allow: HashSet::new(),
                reject: HashSet::new(),
            })),
            usage_in: record.usage.input_tokens,
            usage_out: record.usage.output_tokens,
            mcp,
            context_summary: record.context_summary.clone(),
            choices: Choices {
                mode: "ask",
                effort: None,
                capabilities: self.config.capabilities,
            },
            extra_dirs: extra_dirs.clone(),
        };
        lock_state(&self.state)
            .sessions
            .insert(session_id.to_string(), session);
        let choices = lock_state(&self.state).sessions[session_id].choices.clone();
        rpc_ok(id, session_setup(session_id, &model, false, &choices))
    }

    fn emit_replay(&self, session_id: &str, messages: &[Message]) {
        for message in messages {
            let user = matches!(message.role, greppy_agent::Role::User);
            for part in &message.content {
                let update = match part {
                    greppy_agent::ContentPart::Text { text } if !text.is_empty() => json!({
                        "sessionUpdate": if user { "user_message_chunk" } else { "agent_message_chunk" },
                        "content": {"type": "text", "text": text}
                    }),
                    greppy_agent::ContentPart::Thinking { text } if !text.is_empty() => json!({
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": text}
                    }),
                    greppy_agent::ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    } => json!({
                        "sessionUpdate": "tool_call",
                        "toolCallId": id,
                        "title": tool_title(name, arguments),
                        "kind": tool_kind(name, arguments),
                        "status": "completed",
                        "rawInput": arguments
                    }),
                    _ => continue,
                };
                let _ = self.notify(
                    "session/update",
                    json!({
                        "sessionId": session_id,
                        "_meta": {"isReplay": true},
                        "update": update
                    }),
                );
            }
        }
    }

    fn session_list(&self, id: &Value, params: &Value) -> Value {
        if params
            .get("cursor")
            .and_then(Value::as_str)
            .is_some_and(|cursor| !cursor.is_empty())
        {
            return rpc_error(id, -32602, "pagination cursor is not supported");
        }
        let cwd_filter = params.get("cwd").and_then(Value::as_str).map(PathBuf::from);
        let (data_root, project) = match &cwd_filter {
            Some(cwd) => self.store_identity(cwd),
            None => {
                let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                self.store_identity(&cwd)
            }
        };
        let store = SessionStore::new(data_root.as_path(), project.as_str());
        let records = match store.list() {
            Ok(records) => records,
            Err(error) => return rpc_error(id, -32603, &format!("cannot list sessions: {error}")),
        };
        let sessions: Vec<Value> = records
            .into_iter()
            .filter(|record| match &cwd_filter {
                Some(cwd) => same_dir(Path::new(&record.worktree), cwd),
                None => true,
            })
            .map(|record| {
                json!({
                    "sessionId": record.id,
                    "cwd": record.worktree,
                    "title": record.title
                })
            })
            .collect();
        rpc_ok(id, json!({ "sessions": sessions }))
    }

    fn session_close(&self, id: &Value, params: &Value) -> Value {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        let mut state = lock_state(&self.state);
        let Some(session) = state.sessions.get_mut(session_id) else {
            return rpc_error(id, -32002, &format!("session not found: {session_id}"));
        };
        if session.busy {
            return rpc_error(id, -32600, "session is busy");
        }
        session.closed = true;
        session.cancel.store(true, Ordering::Relaxed);
        rpc_ok(id, json!({}))
    }

    fn session_set_model(&self, id: &Value, params: &Value) -> Value {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        let Some(model_id) = params.get("modelId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "modelId is required");
        };
        if model_id.trim().is_empty() {
            return rpc_error(id, -32602, "modelId is empty");
        }
        self.write_model(session_id, model_id.trim(), id)
    }

    /// Greppy extension: fresh caller metadata, independent of ACP selectors.
    fn set_route_output_limit(&self, id: &Value, params: &Value) -> Value {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        let Some(limit) = params
            .get("maxOutputTokens")
            .and_then(Value::as_u64)
            .and_then(|n| OutputTokenLimit::try_from(n).ok())
        else {
            return rpc_error(
                id,
                -32602,
                "maxOutputTokens expects a finite positive integer",
            );
        };
        let mut state = lock_state(&self.state);
        let Some(session) = state.sessions.get_mut(session_id) else {
            return rpc_error(id, -32002, &format!("session not found: {session_id}"));
        };
        if session.closed || session.busy {
            return rpc_error(id, -32600, "session is closed or busy");
        }
        if params.get("modelId").and_then(Value::as_str) != Some(session.model.as_str())
            || params.get("endpoint").and_then(Value::as_str) != Some(self.config.endpoint.as_str())
        {
            return rpc_error(
                id,
                -32602,
                "output limit requires the exact current modelId and endpoint route",
            );
        }
        session.max_output_tokens = Some(limit);
        rpc_ok(
            id,
            json!({
                "sessionId": session_id, "modelId": session.model,
                "endpoint": self.config.endpoint, "maxOutputTokens": limit.get()
            }),
        )
    }

    fn session_set_config(&self, id: &Value, params: &Value) -> Value {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        let Some(config_id) = params.get("configId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "configId is required");
        };
        if config_id != "model" {
            let Some(value) = params.get("value").and_then(Value::as_str) else {
                return rpc_error(
                    id,
                    -32602,
                    &format!("config option {config_id} expects a string"),
                );
            };
            return match self.choose(session_id, config_id, value) {
                Ok(()) => {
                    let state = lock_state(&self.state);
                    let session = &state.sessions[session_id];
                    rpc_ok(
                        id,
                        json!({ "configOptions": config_options(&session.model, &session.choices) }),
                    )
                }
                Err((code, message)) => rpc_error(id, code, &message),
            };
        }
        if params.get("type").and_then(Value::as_str) == Some("boolean") {
            return rpc_error(id, -32602, "model config option expects a string");
        }
        let Some(model_id) = params.get("value").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "model config option expects a string");
        };
        if model_id.trim().is_empty() {
            return rpc_error(id, -32602, "model config value is empty");
        }
        match self.write_model(session_id, model_id.trim(), id) {
            value if value.get("error").is_some() => value,
            _ => {
                let model = lock_state(&self.state)
                    .sessions
                    .get(session_id)
                    .map(|session| session.model.clone())
                    .unwrap_or_else(|| model_id.to_string());
                rpc_ok(id, json!({ "configOptions": [model_config(&model)] }))
            }
        }
    }

    /// Change one of a session's choices (`mode`, `effort`, `web`, `desktop`).
    fn choose(&self, session_id: &str, config_id: &str, value: &str) -> Result<(), (i64, String)> {
        let mut state = lock_state(&self.state);
        let Some(session) = state.sessions.get_mut(session_id) else {
            return Err((-32002, format!("session not found: {session_id}")));
        };
        if session.closed {
            return Err((-32600, "session is closed".to_string()));
        }
        let wrong = || {
            (
                -32602,
                format!("{value:?} is no value of config option {config_id}"),
            )
        };
        let switch = |value: &str| match value {
            "on" => Ok(true),
            "off" => Ok(false),
            _ => Err(wrong()),
        };
        match config_id {
            "mode" => {
                session.choices.mode = MODES
                    .iter()
                    .find(|(id, _, _)| *id == value)
                    .map(|(id, _, _)| *id)
                    .ok_or_else(wrong)?;
            }
            "effort" => {
                session.choices.effort = match value {
                    "default" => None,
                    "low" | "medium" | "high" => Some(value.to_owned()),
                    _ => return Err(wrong()),
                };
            }
            "web" => session.choices.capabilities.web = switch(value)?,
            "desktop" => session.choices.capabilities.desktop = switch(value)?,
            _ => {
                return Err((
                    -32602,
                    format!("unsupported session config option: {config_id}"),
                ))
            }
        }
        Ok(())
    }

    /// Delete a session: forgotten here, and its saved conversation removed.
    fn session_delete(&self, id: &Value, params: &Value) -> Value {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        if !SessionStore::is_valid_id(session_id) {
            return rpc_error(id, -32602, "invalid session id");
        }
        let removed = {
            let mut state = lock_state(&self.state);
            if state
                .sessions
                .get(session_id)
                .is_some_and(|session| session.busy)
            {
                return rpc_error(id, -32600, "session is busy");
            }
            state.sessions.remove(session_id)
        };
        let (data_root, project) = match &removed {
            Some(session) => (session.data_root.clone(), session.project.clone()),
            None => match params.get("cwd").and_then(Value::as_str) {
                Some(cwd) => self.store_identity(Path::new(cwd)),
                None => return rpc_error(id, -32002, &format!("session not found: {session_id}")),
            },
        };
        let store = SessionStore::new(data_root.as_path(), project.as_str());
        match store
            .path_for(session_id)
            .and_then(|path| match std::fs::remove_file(&path) {
                Err(error) if error.kind() == io::ErrorKind::NotFound && removed.is_some() => {
                    Ok(())
                }
                other => other,
            }) {
            Ok(()) => rpc_ok(id, json!({})),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                rpc_error(id, -32002, &format!("session not found: {session_id}"))
            }
            Err(error) => rpc_error(id, -32603, &format!("cannot delete session: {error}")),
        }
    }

    fn session_set_mode(&self, id: &Value, params: &Value) -> Value {
        let (Some(session_id), Some(mode)) = (
            params.get("sessionId").and_then(Value::as_str),
            params.get("modeId").and_then(Value::as_str),
        ) else {
            return rpc_error(id, -32602, "sessionId and modeId are required");
        };
        match self.choose(session_id, "mode", mode) {
            Ok(()) => rpc_ok(id, json!({})),
            Err((code, message)) => rpc_error(id, code, &message),
        }
    }

    fn write_model(&self, session_id: &str, model_id: &str, id: &Value) -> Value {
        let mut state = lock_state(&self.state);
        let Some(session) = state.sessions.get_mut(session_id) else {
            return rpc_error(id, -32002, &format!("session not found: {session_id}"));
        };
        if session.closed {
            return rpc_error(id, -32600, "session is closed");
        }
        if session.busy {
            return rpc_error(id, -32600, "session is busy");
        }
        let store = SessionStore::new(session.data_root.clone(), session.project.clone());
        if let Err(error) = store.set_model(session_id, model_id) {
            return rpc_error(id, -32603, &format!("cannot persist model: {error}"));
        }
        if session.model != model_id {
            session.max_output_tokens = None;
        }
        session.model = model_id.to_string();
        rpc_ok(id, json!({}))
    }

    fn session_prompt(&self, id: &Value, params: &Value) {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            let _ = self
                .out
                .send(&rpc_error(id, -32602, "sessionId is required"));
            return;
        };
        let blocks = match params.get("prompt").and_then(Value::as_array) {
            Some(blocks) => blocks,
            None => {
                let _ = self
                    .out
                    .send(&rpc_error(id, -32602, "prompt must be an array"));
                return;
            }
        };
        let prompt = match prompt_text(blocks) {
            Ok(prompt) => prompt,
            Err(message) => {
                let _ = self.out.send(&rpc_error(id, -32602, &message));
                return;
            }
        };
        let images = match prompt_images(blocks) {
            Ok(images) => images,
            Err(message) => {
                let _ = self.out.send(&rpc_error(id, -32602, &message));
                return;
            }
        };
        if prompt.trim().is_empty() && images.is_empty() {
            let _ = self
                .out
                .send(&rpc_error(id, -32602, "prompt text is empty"));
            return;
        }
        let effort = match params.pointer("/_meta/greppy/effort") {
            None | Some(Value::Null) => None,
            Some(Value::String(effort)) if matches!(effort.as_str(), "low" | "medium" | "high") => {
                Some(effort.clone())
            }
            Some(other) => {
                let _ = self.out.send(&rpc_error(
                    id,
                    -32602,
                    &format!("_meta.greppy.effort {other} is none of low, medium, high"),
                ));
                return;
            }
        };
        // The model's window and what its latest request held, as the client knows them:
        // the conversation is compacted at 90 % of a window of at least 16 000 tokens.
        let number = |name: &str| {
            params
                .pointer(&format!("/_meta/greppy/{name}"))
                .and_then(Value::as_u64)
        };
        let context_window = number("contextWindow").filter(|window| *window >= SMALLEST_WINDOW);
        let compact_at = context_window.map(|window| window / 10 * 9);
        let context_tokens = number("contextTokens");
        let client_tools = match client_tools(params) {
            Ok(tools) => tools,
            Err(message) => {
                let _ = self.out.send(&rpc_error(id, -32602, &message));
                return;
            }
        };
        let prepared = {
            let mut state = lock_state(&self.state);
            let Some(session) = state.sessions.get_mut(session_id) else {
                drop(state);
                let _ = self.out.send(&rpc_error(
                    id,
                    -32002,
                    &format!("session not found: {session_id}"),
                ));
                return;
            };
            if session.closed {
                drop(state);
                let _ = self.out.send(&rpc_error(id, -32600, "session is closed"));
                return;
            }
            if session.busy {
                drop(state);
                let _ = self.out.send(&rpc_error(id, -32600, "session is busy"));
                return;
            }
            if session.model.trim().is_empty() {
                drop(state);
                let _ = self.out.send(&rpc_error(
                    id,
                    -32602,
                    "model is not configured; set GREPPY_MODEL or call session/set_model",
                ));
                return;
            }
            session.busy = true;
            session.cancel.store(false, Ordering::Relaxed);
            let prepared = PreparedPrompt {
                session_id: session.id.clone(),
                cwd: session.cwd.clone(),
                data_root: session.data_root.clone(),
                project: session.project.clone(),
                model: session.model.clone(),
                history: session.messages.clone(),
                max_output_tokens: session.max_output_tokens,
                cancel: Arc::clone(&session.cancel),
                perms: Arc::clone(&session.perms),
                client_tools,
                images,
                mcp: Arc::clone(&session.mcp),
                context_summary: session.context_summary.clone(),
                choices: session.choices.clone(),
                effort,
                compact_at,
                context_tokens,
                context_window,
                extra_dirs: session.extra_dirs.clone(),
                client: Value::Null,
            };
            // (Kept so that `$/cancel_request` finds the prompt's session.)
            let mut prepared = prepared;
            prepared.client = state.client.clone();
            state.prompts.insert(id_key(id), session_id.to_owned());
            prepared
        };
        let request_id = id.clone();
        let out = self.out.clone();
        let state = Arc::clone(&self.state);
        let pending = Arc::clone(&self.pending);
        let next_request_id = Arc::clone(&self.next_request_id);
        let config = self.config.clone();
        let message_id = params.get("messageId").cloned();
        thread::spawn(move || {
            let busy = BusyGuard {
                state: Arc::clone(&state),
                session_id: prepared.session_id.clone(),
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_prompt(
                    &prepared,
                    &prompt,
                    &config,
                    &out,
                    &pending,
                    &next_request_id,
                )
            }));
            let reply = match result {
                Ok(Ok(done)) => {
                    if let Err(error) = finish_prompt(&state, &prepared, &config, &done) {
                        drop(busy);
                        let _ = out.send(&rpc_error(&request_id, -32603, &error));
                        return;
                    }
                    let mut response = json!({
                        "stopReason": done.stop_reason,
                        "usage": {
                            "inputTokens": done.usage.input_tokens,
                            "outputTokens": done.usage.output_tokens,
                            "totalTokens": done.usage.input_tokens.saturating_add(done.usage.output_tokens),
                            "cachedReadTokens": done.usage.cache_read_input_tokens,
                            "cachedWriteTokens": done.usage.cache_creation_input_tokens
                        }
                    });
                    if let Some(message_id) = message_id {
                        if !message_id.is_null() {
                            response["userMessageId"] = message_id;
                        }
                    }
                    rpc_ok(&request_id, response)
                }
                Ok(Err(error)) => rpc_error(&request_id, -32603, &error),
                Err(_) => rpc_error(&request_id, -32603, "agent prompt failed unexpectedly"),
            };
            drop(busy);
            lock_state(&state).prompts.remove(&id_key(&request_id));
            let _ = out.send(&reply);
        });
    }

    fn notify(&self, method: &str, params: Value) -> io::Result<()> {
        self.out.send(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": "",
            "headers": []
        }))
    }

    fn store_identity(&self, cwd: &Path) -> (PathBuf, String) {
        (
            self.data_root.clone(),
            greppy_core::workspace::project_identity_from_workspace(cwd),
        )
    }
}

impl Clone for Out {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct PreparedPrompt {
    session_id: String,
    cwd: PathBuf,
    data_root: PathBuf,
    project: String,
    model: String,
    history: Vec<Message>,
    max_output_tokens: Option<OutputTokenLimit>,
    cancel: Arc<AtomicBool>,
    perms: Arc<Mutex<PermMemory>>,
    /// Tools the client offered with this prompt; it runs them (`_greppy/client_tool`).
    client_tools: Vec<greppy_agent::ToolDefinition>,
    /// The prompt's pictures, after its text.
    images: Vec<greppy_agent::ContentPart>,
    /// The session's MCP servers.
    mcp: Arc<Mutex<greppy_agent::mcp::McpHub>>,
    /// The conversation's checkpoint summary before this prompt.
    context_summary: Option<String>,
    /// The session's choices when the prompt began.
    choices: Choices,
    /// The model's window as the client named it (`_meta.greppy.contextWindow`).
    context_window: Option<u64>,
    /// Further folders the client named for the session (`additionalDirectories`).
    extra_dirs: Vec<PathBuf>,
    /// The client's capabilities (`initialize`).
    client: Value,
    /// The reasoning effort chosen for this prompt (`_meta.greppy.effort`).
    effort: Option<String>,
    /// Prompt tokens at which this prompt's conversation is compacted (90 % of
    /// `_meta.greppy.contextWindow`).
    compact_at: Option<u64>,
    /// What the conversation's latest request held (`_meta.greppy.contextTokens`).
    context_tokens: Option<u64>,
}

struct PromptDone {
    messages: Vec<Message>,
    /// The checkpoint summary after this prompt (a new one where it compacted).
    context_summary: Option<String>,
    stop_reason: &'static str,
    usage: greppy_agent::Usage,
}

struct BusyGuard {
    state: Arc<Mutex<State>>,
    session_id: String,
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        let mut state = lock_state(&self.state);
        if let Some(session) = state.sessions.get_mut(&self.session_id) {
            session.busy = false;
        }
    }
}

fn run_prompt(
    prepared: &PreparedPrompt,
    prompt: &str,
    config: &AcpConfig,
    out: &Out,
    pending: &Arc<Mutex<HashMap<String, PendingPermission>>>,
    next_request_id: &Arc<AtomicU64>,
) -> Result<PromptDone, String> {
    let mut client = Client::new(&config.endpoint, &prepared.model);
    if let Some(key) = &config.api_key {
        client = client.with_api_key(key);
    }
    for (name, value) in &config.headers {
        client = client.with_header(name, value)?;
    }
    if let Some(effort) = prepared
        .effort
        .as_ref()
        .or(prepared.choices.effort.as_ref())
    {
        client = client.with_effort(effort)?;
    }
    let mut client = CancelModel {
        client,
        cancel: Arc::clone(&prepared.cancel),
    };

    let live = LiveView::new(out.clone(), prepared.session_id.clone());
    let inner: Box<dyn ExecutionEnv + Send> = if let Some(factory) = &config.tool_env {
        factory(&prepared.cwd)
    } else {
        let greppy = GreppyEnv::new(prepared.cwd.clone())
            .map_err(|error| format!("cannot build the tool environment: {error}"))?
            .with_capabilities(prepared.choices.capabilities)
            .with_live_output(live.sink());
        if config.sandbox {
            sandboxed(greppy, &prepared.cwd, &prepared.extra_dirs)
        } else {
            Box::new(greppy)
        }
    };
    // The tools of the session's MCP servers beside greppy's (permission as for any tool).
    let inner: Box<dyn ExecutionEnv + Send> = Box::new(greppy_agent::mcp::McpEnv {
        inner,
        hub: Arc::clone(&prepared.mcp),
    });
    let gated = GatingEnv {
        mode: prepared.choices.mode,
        inner,
        session_id: prepared.session_id.clone(),
        perms: Arc::clone(&prepared.perms),
        cancel: Arc::clone(&prepared.cancel),
        out: out.clone(),
        pending: Arc::clone(pending),
        next_request_id: Arc::clone(next_request_id),
    };
    // greppy's plan, said to the client as an ACP plan (no permission: it changes nothing).
    let planned = PlanEnv {
        inner: Box::new(gated),
        out: out.clone(),
        session_id: prepared.session_id.clone(),
    };
    // Where the client can ask the person (form elicitation), greppy can too.
    let asking: Box<dyn ExecutionEnv + Send> = if prepared
        .client
        .pointer("/elicitation/form")
        .is_some_and(|form| !form.is_null() && form != &json!(false))
    {
        Box::new(AskEnv {
            inner: Box::new(planned),
            session_id: prepared.session_id.clone(),
            cancel: Arc::clone(&prepared.cancel),
            out: out.clone(),
            pending: Arc::clone(pending),
            next_request_id: Arc::clone(next_request_id),
        })
    } else {
        Box::new(planned)
    };
    let mut env = ClientToolEnv {
        inner: asking,
        tools: prepared.client_tools.clone(),
        session_id: prepared.session_id.clone(),
        cancel: Arc::clone(&prepared.cancel),
        out: out.clone(),
        pending: Arc::clone(pending),
        next_request_id: Arc::clone(next_request_id),
    };
    let agent_config = AgentConfig {
        max_turns: config.max_turns,
        system: Some(greppy_agent::system_prompt_with_capabilities(
            BuiltinPromptMode::Acp,
            prepared.choices.capabilities,
        )),
        model: prepared.model.clone(),
        cancel: Some(Arc::clone(&prepared.cancel)),
        compact_at_tokens: {
            let mut thresholds = crate::agent_tui::AgentSettings::load().compaction_thresholds();
            if let Some(at) = prepared.compact_at {
                thresholds.insert(prepared.model.clone(), at);
            }
            thresholds
        },
        measured_prompt_tokens: prepared.context_tokens,
        context_summary: prepared.context_summary.clone(),
        default_compact_at_tokens: Some(greppy_agent::default_compact_at_tokens()),
        ..AgentConfig::default()
    }
    .with_max_tokens(
        prepared
            .max_output_tokens
            .map_or(u64::MAX, OutputTokenLimit::get),
    );
    let session_id = prepared.session_id.clone();
    // A new conversation is named after its first words.
    if prepared.history.is_empty() && !prompt.trim().is_empty() {
        let _ = send_update(
            out,
            &session_id,
            json!({"sessionUpdate": "session_info_update", "title": truncate_chars(prompt.trim(), 80)}),
        );
    }
    // A slash command is the client's, not a message for the model.
    if let Some(done) = slash_command(prompt, prepared, &agent_config, &mut client, &mut env, out)?
    {
        return Ok(done);
    }
    let window = prepared
        .context_window
        .unwrap_or(greppy_agent::DEFAULT_CONTEXT_WINDOW);
    let compaction_updates = prepared
        .client
        .pointer("/session/compaction")
        .is_some_and(|v| !v.is_null() && v != &json!(false));
    let mut refused = false;
    let mut compactions = 0u32;
    let mut view = ToolView::new(&prepared.cwd);
    let mut on_event = |event: LoopEvent| {
        if let LoopEvent::ToolStart { call_id, .. } = &event {
            let call_id = call_id.clone();
            ACTIVE_TOOL_CALL.with(|slot| *slot.borrow_mut() = Some(call_id.clone()));
            live.begin(call_id);
        }
        if let LoopEvent::TurnComplete { stop_reason, .. } = &event {
            refused = matches!(stop_reason, greppy_agent::StopReason::Other(reason) if reason == "refusal");
        }
        if let (true, LoopEvent::ContextCompacted { summary, .. }) = (compaction_updates, &event) {
            compactions += 1;
            let _ = send_update(
                out,
                &session_id,
                json!({
                    "sessionUpdate": "compaction_update",
                    "compactionId": format!("{session_id}-{}-{compactions}", now_ms()),
                    "status": "completed",
                    "summary": summary
                }),
            );
        }
        if let Some(mut update) = loop_update(&event) {
            view.decorate(&event, &mut update);
            if update["sessionUpdate"] == "usage_update" {
                update["size"] = json!(window);
            }
            let _ = send_update(out, &session_id, update);
        }
    };
    // Further folders of the session, named to the model with its first message.
    let mut text = prompt.to_owned();
    if prepared.history.is_empty() && !prepared.extra_dirs.is_empty() {
        let dirs: Vec<String> = prepared
            .extra_dirs
            .iter()
            .map(|dir| dir.display().to_string())
            .collect();
        text = format!(
            "[Besides the session's folder you may work in: {}]\n{text}",
            dirs.join(", ")
        );
    }
    let content = (!text.is_empty())
        .then(|| greppy_agent::ContentPart::Text { text })
        .into_iter()
        .chain(prepared.images.iter().cloned())
        .collect();
    match run_agent_loop_with_content(
        &mut client,
        &mut env,
        &agent_config,
        &prepared.history,
        content,
        &mut on_event,
    ) {
        Ok(result) => Ok(PromptDone {
            messages: result.messages,
            context_summary: result.context_summary,
            stop_reason: if refused && result.stop == LoopStop::EndTurn {
                "refusal"
            } else {
                stop_reason(&result.stop)
            },
            usage: result.usage,
        }),
        Err(error) => Err(loop_error_message(&error)),
    }
}

/// The slash commands a client is offered (`available_commands_update`).
const SLASH_COMMANDS: [(&str, &str); 2] = [
    (
        "compact",
        "Make the conversation shorter now: a checkpoint of what was said",
    ),
    ("undo", "Undo greppy's last change to a file"),
];

/// A prompt that is one of greppy's slash commands, carried out without the model's
/// turn: `/compact` checkpoints the conversation now, `/undo` takes back greppy's
/// last edit. None for any other prompt.
fn slash_command(
    prompt: &str,
    prepared: &PreparedPrompt,
    config: &AgentConfig,
    client: &mut CancelModel,
    env: &mut dyn ExecutionEnv,
    out: &Out,
) -> Result<Option<PromptDone>, String> {
    let command = prompt.trim();
    let say = |text: &str| {
        let _ = send_update(
            out,
            &prepared.session_id,
            json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}),
        );
    };
    let unchanged = |usage| PromptDone {
        messages: prepared.history.clone(),
        context_summary: prepared.context_summary.clone(),
        stop_reason: "end_turn",
        usage,
    };
    match command {
        "/compact" => {
            let mut messages = prepared.history.clone();
            let mut system = config.system.clone();
            greppy_agent::context::restore_summary(
                &mut system,
                prepared.context_summary.as_deref(),
            );
            let template = ModelRequest {
                model: config.model.clone(),
                system: system.clone(),
                messages: Vec::new(),
                tools: Vec::new(),
                tool_choice: greppy_agent::ToolChoice::None,
                max_tokens: config.max_tokens,
            };
            let before = messages.len();
            let budget = greppy_agent::context::history_bytes(&messages).saturating_sub(1);
            match greppy_agent::context::compact_with_model(
                client,
                &mut messages,
                &mut system,
                &template,
                budget,
            ) {
                Ok(Some(usage)) => {
                    say(&format!(
                        "The conversation was compacted: {before} messages are now {} and a summary.",
                        messages.len()
                    ));
                    Ok(Some(PromptDone {
                        messages,
                        context_summary: greppy_agent::context::saved_summary(system.as_deref())
                            .map(str::to_owned),
                        stop_reason: "end_turn",
                        usage,
                    }))
                }
                Ok(None) => {
                    say("There is nothing to compact yet.");
                    Ok(Some(unchanged(greppy_agent::Usage::default())))
                }
                Err(error) => Err(format!("compaction failed: {error}")),
            }
        }
        "/undo" => {
            let outcome = env.call_tool("greppy", &json!({"args": ["undo"]}));
            say(&outcome.content);
            Ok(Some(unchanged(greppy_agent::Usage::default())))
        }
        _ => Ok(None),
    }
}

/// The question tool greppy offers where the client can ask the person (form
/// elicitation): the questions go to the client as one form; the answers, or that the
/// person declined, are the tool's result.
struct AskEnv {
    inner: Box<dyn ExecutionEnv + Send>,
    session_id: String,
    cancel: Arc<AtomicBool>,
    out: Out,
    pending: Arc<Mutex<HashMap<String, PendingPermission>>>,
    next_request_id: Arc<AtomicU64>,
}

const ASK_TOOL: &str = "request_user_input";

impl ExecutionEnv for AskEnv {
    fn tool_definitions(&self) -> Vec<greppy_agent::ToolDefinition> {
        let mut tools = self.inner.tool_definitions();
        tools.push(greppy_agent::ToolDefinition {
            name: ASK_TOOL.to_owned(),
            description: "Ask the person one to three short questions, each with two to four choices (they may answer in words of their own). Ask only when you cannot go on without their decision.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {"questions": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "header": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "object", "properties": {
                            "label": {"type": "string"}, "description": {"type": "string"}
                        }, "required": ["label"]}}
                    },
                    "required": ["id", "question"]
                }}},
                "required": ["questions"]
            }),
        });
        tools
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        if name != ASK_TOOL {
            return self.inner.call_tool(name, arguments);
        }
        let questions = arguments
            .get("questions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if questions.is_empty() {
            return ToolOutcome::err("request_user_input needs `questions`");
        }
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for (at, question) in questions.iter().enumerate() {
            let id = question
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("q{}", at + 1), str::to_owned);
            let mut property = json!({
                "type": "string",
                "title": question.get("header").and_then(Value::as_str).unwrap_or(&id),
                "description": question.get("question").and_then(Value::as_str).unwrap_or("")
            });
            let labels: Vec<&str> = question
                .get("options")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|option| option.get("label").and_then(Value::as_str))
                .collect();
            if !labels.is_empty() {
                property["enum"] = json!(labels);
            }
            required.push(id.clone());
            properties.insert(id, property);
        }
        let message = questions
            .iter()
            .filter_map(|question| question.get("question").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        let request_number = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let request_id = Value::from(request_number);
        let (tx, rx) = mpsc::channel();
        lock_pending(&self.pending).insert(
            id_key(&request_id),
            PendingPermission {
                session_id: self.session_id.clone(),
                sender: tx,
            },
        );
        let request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "elicitation/create",
            "params": {
                "sessionId": self.session_id,
                "mode": "form",
                "message": message,
                "requestedSchema": {"type": "object", "properties": properties, "required": required}
            },
            "headers": []
        });
        let answer = if self.out.send(&request).is_ok() {
            wait_permission(&rx, &self.cancel)
        } else {
            None
        };
        lock_pending(&self.pending).remove(&id_key(&request_id));
        let Some(answer) = answer else {
            return ToolOutcome::err("The person was not asked: the question was cancelled.");
        };
        let result = answer.get("result").cloned().unwrap_or(Value::Null);
        match result.get("action").and_then(Value::as_str) {
            Some("accept") => {
                let content = result.get("content").cloned().unwrap_or(Value::Null);
                let answers: Vec<String> = content
                    .as_object()
                    .into_iter()
                    .flatten()
                    .map(|(id, value)| {
                        format!(
                            "{id}: {}",
                            value
                                .as_str()
                                .map_or_else(|| value.to_string(), str::to_owned)
                        )
                    })
                    .collect();
                ToolOutcome::ok(format!("The person answered:\n{}", answers.join("\n")))
            }
            Some("decline") => ToolOutcome::ok("The person declined to answer."),
            _ => ToolOutcome::ok("The person did not answer (the question was cancelled)."),
        }
    }
}

#[cfg(test)]
fn emit_loop_event(out: &Out, session_id: &str, event: &LoopEvent) -> io::Result<()> {
    match loop_update(event) {
        Some(update) => send_update(out, session_id, update),
        None => Ok(()),
    }
}

fn send_update(out: &Out, session_id: &str, update: Value) -> io::Result<()> {
    out.send(&json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": update
        },
        "id": "",
        "headers": []
    }))
}

/// The ACP update a loop event is said as, where it is said.
fn loop_update(event: &LoopEvent) -> Option<Value> {
    let update = match event {
        LoopEvent::Stream(StreamEvent::TextDelta { text }) if !text.is_empty() => json!({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text}
        }),
        LoopEvent::Stream(StreamEvent::ThinkingDelta { text }) if !text.is_empty() => json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": text}
        }),
        LoopEvent::ToolStart {
            call_id,
            name,
            arguments,
        } => json!({
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": tool_title(name, arguments),
            "kind": tool_kind(name, arguments),
            "status": "pending",
            "rawInput": arguments,
            "_meta": {"greppy": {"tool": name}}
        }),
        LoopEvent::ToolFinish {
            call_id, outcome, ..
        } => json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": if outcome.is_error { "failed" } else { "completed" },
            "rawOutput": truncate_chars(&outcome.content, 16_384)
        }),
        // What one answered model request took, as the provider said it. `used` is
        // the tokens that request held (prompt and answer); a provider that says
        // nothing gives no update rather than zeros.
        LoopEvent::TurnComplete { usage, .. } => {
            let prompt = usage
                .input_tokens
                .saturating_add(usage.cache_read_input_tokens)
                .saturating_add(usage.cache_creation_input_tokens);
            if prompt == 0 && usage.output_tokens == 0 {
                return None;
            }
            json!({
                "sessionUpdate": "usage_update",
                "used": prompt.saturating_add(usage.output_tokens),
                "_meta": {"greppy": {
                    "inputTokens": usage.input_tokens,
                    "outputTokens": usage.output_tokens,
                    "cachedReadTokens": usage.cache_read_input_tokens,
                    "cachedWriteTokens": usage.cache_creation_input_tokens
                }}
            })
        }
        _ => return None,
    };
    Some(update)
}

/// At most this much of a running command's output is shown live (its end).
const LIVE_OUTPUT_BYTES: usize = 16 * 1024;

/// A running command's output, said to the client as it comes: the tool call's content
/// while it is `in_progress` (the end of what it printed so far).
#[derive(Clone)]
struct LiveView {
    out: Out,
    session_id: String,
    /// The tool call under way, and what it printed so far.
    current: Arc<Mutex<(Option<String>, Vec<u8>)>>,
}

impl LiveView {
    fn new(out: Out, session_id: String) -> Self {
        Self {
            out,
            session_id,
            current: Arc::default(),
        }
    }

    fn begin(&self, call_id: String) {
        if let Ok(mut current) = self.current.lock() {
            *current = (Some(call_id), Vec::new());
        }
    }

    fn sink(&self) -> greppy_agent::greppy_env::LiveOutput {
        let view = self.clone();
        greppy_agent::greppy_env::LiveOutput(Arc::new(move |chunk: &[u8]| view.add(chunk)))
    }

    fn add(&self, chunk: &[u8]) {
        let (call_id, text) = {
            let Ok(mut current) = self.current.lock() else {
                return;
            };
            let Some(call_id) = current.0.clone() else {
                return;
            };
            current.1.extend_from_slice(chunk);
            let excess = current.1.len().saturating_sub(LIVE_OUTPUT_BYTES);
            current.1.drain(..excess);
            (call_id, String::from_utf8_lossy(&current.1).into_owned())
        };
        let _ = send_update(
            &self.out,
            &self.session_id,
            json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "in_progress",
                "content": [{"type": "content", "content": {"type": "text", "text": text}}]
            }),
        );
    }
}

/// greppy's subcommands that change one file named by their first argument.
const FILE_EDITS: [&str; 5] = [
    "write",
    "replace-text",
    "replace-lines",
    "insert-lines",
    "delete-lines",
];
/// A file this large is not shown as a diff.
const DIFF_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// What a client is shown of a tool call beyond its title: the places it reads or
/// changes (`locations`, absolute paths) and, for an edit of a named file, the change
/// as a `diff` (the file as it was before the call and as it is after).
struct ToolView {
    cwd: PathBuf,
    before: HashMap<String, (PathBuf, Option<String>)>,
}

impl ToolView {
    fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            before: HashMap::new(),
        }
    }

    fn absolute(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    fn read(path: &Path) -> Option<String> {
        let metadata = std::fs::metadata(path).ok()?;
        if metadata.len() > DIFF_FILE_BYTES {
            return None;
        }
        std::fs::read_to_string(path).ok()
    }

    fn decorate(&mut self, event: &LoopEvent, update: &mut Value) {
        match event {
            LoopEvent::ToolStart {
                call_id,
                name,
                arguments,
            } if name == "greppy" => {
                let words = greppy_words(arguments);
                let sub = words.first().map(String::as_str).unwrap_or("");
                let Some(target) = words.get(1).filter(|word| !word.starts_with('-')) else {
                    return;
                };
                if FILE_EDITS.contains(&sub) {
                    let path = self.absolute(target);
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    if size <= DIFF_FILE_BYTES {
                        self.before
                            .insert(call_id.clone(), (path.clone(), Self::read(&path)));
                    }
                    update["locations"] = json!([{"path": path}]);
                } else if matches!(sub, "read" | "read-file" | "read-smart") {
                    let path = self.absolute(target);
                    if path.is_file() {
                        let line = words
                            .iter()
                            .position(|word| word == "--lines")
                            .and_then(|at| words.get(at + 1))
                            .and_then(|range| range.split(':').next()?.parse::<u64>().ok());
                        let mut location = json!({"path": path});
                        if let Some(line) = line {
                            location["line"] = json!(line);
                        }
                        update["locations"] = json!([location]);
                    }
                }
            }
            LoopEvent::ToolFinish {
                call_id, outcome, ..
            } => {
                let mut locations = Vec::new();
                // greppy says each change as `applied PATH:LINE …`.
                for line in outcome.content.lines() {
                    let Some(rest) = line.trim_start().strip_prefix("applied ") else {
                        continue;
                    };
                    let place = rest.split_whitespace().next().unwrap_or("");
                    if let Some((path, number)) = place.rsplit_once(':') {
                        if let Ok(number) = number.parse::<u64>() {
                            locations.push(json!({"path": self.absolute(path), "line": number}));
                        }
                    }
                }
                if !locations.is_empty() {
                    update["locations"] = Value::Array(locations);
                }
                if let Some((path, old)) = self.before.remove(call_id) {
                    let new = Self::read(&path);
                    if new.is_some() && new != old {
                        update["content"] = json!([{
                            "type": "diff",
                            "path": path,
                            "oldText": old,
                            "newText": new
                        }]);
                    }
                }
            }
            _ => {}
        }
    }
}

/// The words of a greppy tool call (`{"args":[…]}`, or the first call of a batch).
fn greppy_words(arguments: &Value) -> Vec<String> {
    let args = if arguments.get("args").is_some_and(Value::is_array) {
        &arguments["args"]
    } else {
        &arguments["batch"][0]
    };
    args.as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

struct CancelModel {
    client: Client,
    cancel: Arc<AtomicBool>,
}

impl ModelStream for CancelModel {
    fn stream_turn(
        &mut self,
        request: &ModelRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        self.client
            .stream_turn_interruptible(request, on_event, &self.cancel)
    }
}

fn persist_turn(
    prepared: &PreparedPrompt,
    _config: &AcpConfig,
    done: &PromptDone,
) -> Result<(), String> {
    let store = SessionStore::new(prepared.data_root.clone(), prepared.project.clone());
    let title = prepared.history.is_empty().then(|| {
        done.messages
            .iter()
            .flat_map(|message| message.content.iter())
            .find_map(|part| match part {
                greppy_agent::ContentPart::Text { text } => Some(truncate_chars(text, 80)),
                _ => None,
            })
            .unwrap_or_else(|| "untitled".to_string())
    });
    store
        .commit_turn(
            &prepared.session_id,
            crate::agent_tui::TurnCommit {
                expected: &messages_from_protocol(&prepared.history),
                messages: &messages_from_protocol(&done.messages),
                usage: &done.usage,
                stop: done.stop_reason,
                summary: done
                    .context_summary
                    .as_deref()
                    .filter(|_| done.context_summary != prepared.context_summary),
                title: title.as_deref(),
            },
            || {
                #[cfg(test)]
                if let Some(hook) = &_config.after_messages {
                    hook()?;
                }
                Ok(())
            },
        )
        .map_err(|error| format!("cannot persist session history: {error}"))
}

fn finish_prompt(
    state: &Mutex<State>,
    prepared: &PreparedPrompt,
    config: &AcpConfig,
    done: &PromptDone,
) -> Result<(), String> {
    persist_turn(prepared, config, done)?;
    let mut state = lock_state(state);
    if let Some(session) = state.sessions.get_mut(&prepared.session_id) {
        session.messages = done.messages.clone();
        session.context_summary = done.context_summary.clone();
        session.usage_in = session.usage_in.saturating_add(done.usage.input_tokens);
        session.usage_out = session.usage_out.saturating_add(done.usage.output_tokens);
    }
    Ok(())
}

/// The plan tool greppy offers an ACP client: the model keeps its plan with it, and
/// every change goes to the client as an ACP `plan` (entries with their status).
struct PlanEnv {
    inner: Box<dyn ExecutionEnv + Send>,
    out: Out,
    session_id: String,
}

const PLAN_TOOL: &str = "update_plan";

impl ExecutionEnv for PlanEnv {
    fn tool_definitions(&self) -> Vec<greppy_agent::ToolDefinition> {
        let mut tools = self.inner.tool_definitions();
        tools.push(greppy_agent::ToolDefinition {
            name: PLAN_TOOL.to_owned(),
            description: "Keep your plan for this task: its steps, each pending, in_progress or completed (at most one in_progress). Send the whole plan each time; the person sees it as your progress.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "plan": {"type": "array", "items": {"type": "object", "properties": {
                        "step": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                    }, "required": ["step", "status"]}}
                },
                "required": ["plan"]
            }),
        });
        tools
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        if name != PLAN_TOOL {
            return self.inner.call_tool(name, arguments);
        }
        let Some(steps) = arguments.get("plan").and_then(Value::as_array) else {
            return ToolOutcome::err("update_plan needs `plan`, a list of steps");
        };
        let mut entries = Vec::with_capacity(steps.len());
        for step in steps {
            let (Some(content), Some(status)) = (
                step.get("step").and_then(Value::as_str),
                step.get("status").and_then(Value::as_str),
            ) else {
                return ToolOutcome::err("each step needs `step` and `status`");
            };
            if !matches!(status, "pending" | "in_progress" | "completed") {
                return ToolOutcome::err(format!("{status:?} is no step status"));
            }
            entries.push(json!({"content": content, "priority": "medium", "status": status}));
        }
        let _ = send_update(
            &self.out,
            &self.session_id,
            json!({"sessionUpdate": "plan", "entries": entries}),
        );
        ToolOutcome::ok("The plan is kept.")
    }
}

/// greppy's tools confined to `folder` and the run's own temporary folder, without
/// network; where that sandbox cannot be made, the same tools refusing every call
/// with the reason.
fn sandboxed(greppy: GreppyEnv, folder: &Path, extra: &[PathBuf]) -> Box<dyn ExecutionEnv + Send> {
    let roots: Vec<PathBuf> = [folder.to_path_buf(), greppy.run_scratch().to_path_buf()]
        .into_iter()
        .chain(extra.iter().cloned())
        .collect();
    match greppy_agent::sandbox::resolve_enforce_spec(&roots) {
        Ok(greppy_agent::SandboxMode::Enforce(spec)) => Box::new(
            greppy.with_sandbox(greppy_agent::SandboxMode::Enforce(spec.without_network())),
        ),
        Ok(greppy_agent::SandboxMode::Off) => Box::new(Refusing {
            tools: greppy.tool_definitions(),
            why: "the sandbox cannot be made on this host".to_owned(),
        }),
        Err(error) => Box::new(Refusing {
            tools: greppy.tool_definitions(),
            why: format!("the sandbox cannot be made on this host: {error}"),
        }),
    }
}

/// Tools that are offered but refused, with the reason: a sandbox that was asked for
/// and cannot be made is never replaced by running unconfined.
struct Refusing {
    tools: Vec<greppy_agent::ToolDefinition>,
    why: String,
}

impl ExecutionEnv for Refusing {
    fn tool_definitions(&self) -> Vec<greppy_agent::ToolDefinition> {
        self.tools.clone()
    }

    fn call_tool(&mut self, _: &str, _: &Value) -> ToolOutcome {
        ToolOutcome::err(format!("Not run: {}.", self.why))
    }
}

/// The tools a client offered with a prompt (`_meta.greppy.clientTools`, each
/// `{name, description, inputSchema}`), checked: named plainly and once, described, and
/// not under the name of greppy's own tool.
fn client_tools(params: &Value) -> Result<Vec<greppy_agent::ToolDefinition>, String> {
    let Some(offered) = params.pointer("/_meta/greppy/clientTools") else {
        return Ok(Vec::new());
    };
    let offered = offered
        .as_array()
        .ok_or("_meta.greppy.clientTools must be an array")?;
    if offered.len() > CLIENT_TOOLS_LIMIT {
        return Err(format!(
            "at most {CLIENT_TOOLS_LIMIT} client tools go with one prompt"
        ));
    }
    let mut tools: Vec<greppy_agent::ToolDefinition> = Vec::with_capacity(offered.len());
    for tool in offered {
        let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
        let plain = !name.is_empty()
            && name.len() <= 64
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !plain {
            return Err(format!(
                "client tool name {name:?} must be 1-64 letters, digits, _ or -"
            ));
        }
        if name == "greppy" || tools.iter().any(|known| known.name == name) {
            return Err(format!(
                "client tool {name} is named twice or as greppy's own"
            ));
        }
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        if description.trim().is_empty() {
            return Err(format!("client tool {name} has no description"));
        }
        let schema = tool
            .get("inputSchema")
            .filter(|schema| schema.is_object())
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        tools.push(greppy_agent::ToolDefinition {
            name: name.to_owned(),
            description: description.to_owned(),
            input_schema: schema,
        });
    }
    Ok(tools)
}

/// greppy's tools, and the client's: a call of one of the client's is a request to the
/// client (`_greppy/client_tool`), whose answer is the tool's result. Client tools need
/// no permission: the client offered them.
struct ClientToolEnv {
    inner: Box<dyn ExecutionEnv + Send>,
    tools: Vec<greppy_agent::ToolDefinition>,
    session_id: String,
    cancel: Arc<AtomicBool>,
    out: Out,
    pending: Arc<Mutex<HashMap<String, PendingPermission>>>,
    next_request_id: Arc<AtomicU64>,
}

impl ExecutionEnv for ClientToolEnv {
    fn tool_definitions(&self) -> Vec<greppy_agent::ToolDefinition> {
        let mut tools = self.inner.tool_definitions();
        let own: Vec<String> = tools.iter().map(|tool| tool.name.clone()).collect();
        tools.extend(
            self.tools
                .iter()
                .filter(|tool| !own.contains(&tool.name))
                .cloned(),
        );
        tools
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        let own = self
            .inner
            .tool_definitions()
            .iter()
            .any(|tool| tool.name == name);
        if own || !self.tools.iter().any(|tool| tool.name == name) {
            return self.inner.call_tool(name, arguments);
        }
        if self.cancel.load(Ordering::Relaxed) {
            return ToolOutcome::err("cancelled before execution");
        }
        let request_number = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let request_id = Value::from(request_number);
        let (tx, rx) = mpsc::channel();
        lock_pending(&self.pending).insert(
            id_key(&request_id),
            PendingPermission {
                session_id: self.session_id.clone(),
                sender: tx,
            },
        );
        let request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": CLIENT_TOOL_METHOD,
            "params": {
                "sessionId": self.session_id,
                "toolCallId": active_tool_call_id(request_number),
                "name": name,
                "input": arguments
            },
            "headers": []
        });
        let answer = if self.out.send(&request).is_ok() {
            wait_permission(&rx, &self.cancel)
        } else {
            None
        };
        lock_pending(&self.pending).remove(&id_key(&request_id));
        match answer {
            None if self.cancel.load(Ordering::Relaxed) => {
                ToolOutcome::err("cancelled before the client answered")
            }
            None => ToolOutcome::err(format!("the client did not run the tool {name}")),
            Some(answer) => match (answer.get("result"), answer.get("error")) {
                (Some(result), _) => {
                    let content = result
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    if result.get("isError").and_then(Value::as_bool) == Some(true) {
                        ToolOutcome::err(content)
                    } else {
                        ToolOutcome::ok(content)
                    }
                }
                (None, Some(error)) => ToolOutcome::err(
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("the client refused the tool")
                        .to_owned(),
                ),
                (None, None) => ToolOutcome::err(format!("the client did not run the tool {name}")),
            },
        }
    }
}

fn lock_pending(
    pending: &Mutex<HashMap<String, PendingPermission>>,
) -> std::sync::MutexGuard<'_, HashMap<String, PendingPermission>> {
    match pending.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct GatingEnv {
    /// The session's mode: `ask`, `auto` or `read-only`.
    mode: &'static str,
    inner: Box<dyn ExecutionEnv + Send>,
    session_id: String,
    perms: Arc<Mutex<PermMemory>>,
    cancel: Arc<AtomicBool>,
    out: Out,
    pending: Arc<Mutex<HashMap<String, PendingPermission>>>,
    next_request_id: Arc<AtomicU64>,
}

impl ExecutionEnv for GatingEnv {
    fn tool_definitions(&self) -> Vec<greppy_agent::ToolDefinition> {
        self.inner.tool_definitions()
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        if self.cancel.load(Ordering::Relaxed) {
            return ToolOutcome::err("cancelled before execution");
        }
        let kind = tool_kind(name, arguments);
        if self.mode == "read-only" && matches!(kind, "edit" | "execute" | "other") {
            return ToolOutcome::err(
                "The session is read only: no commands, no changes, no outside tools. Read and search only.",
            );
        }
        let authorized = if self.mode == "auto" {
            Ok(())
        } else {
            self.authorize(name, arguments)
        };
        match authorized {
            Ok(()) => {
                if self.cancel.load(Ordering::Relaxed) {
                    return ToolOutcome::err("cancelled before execution");
                }
                let tool_call_id = active_tool_call_id(0);
                let _ = self.out.send(&json!({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": self.session_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": tool_call_id,
                            "status": "in_progress"
                        }
                    },
                    "id": "",
                    "headers": []
                }));
                self.inner.call_tool(name, arguments)
            }
            Err(message) => ToolOutcome::err(message),
        }
    }
}

impl GatingEnv {
    fn authorize(&mut self, name: &str, arguments: &Value) -> Result<(), String> {
        let key = perm_key(name, arguments);
        {
            let memory = match self.perms.lock() {
                Ok(memory) => memory,
                Err(poisoned) => poisoned.into_inner(),
            };
            if memory.reject.contains(&key) {
                return Err("permission denied for the rest of the session".to_string());
            }
            if memory.allow.contains(&key) {
                return Ok(());
            }
        }
        if self.cancel.load(Ordering::Relaxed) {
            return Err("cancelled before execution".to_string());
        }
        let request_number = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let request_id = Value::from(request_number);
        let (tx, rx) = mpsc::channel();
        {
            let mut pending = match self.pending.lock() {
                Ok(pending) => pending,
                Err(poisoned) => poisoned.into_inner(),
            };
            pending.insert(
                id_key(&request_id),
                PendingPermission {
                    session_id: self.session_id.clone(),
                    sender: tx,
                },
            );
        }
        let tool_call_id = active_tool_call_id(request_number);
        let request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "session/request_permission",
            "params": {
                "sessionId": self.session_id,
                "toolCall": {
                    "toolCallId": tool_call_id,
                    "title": tool_title(name, arguments),
                    "kind": tool_kind(name, arguments),
                    "status": "pending",
                    "rawInput": arguments
                },
                "options": permission_options()
            },
            "headers": []
        });
        if self.out.send(&request).is_err() {
            let mut pending = match self.pending.lock() {
                Ok(pending) => pending,
                Err(poisoned) => poisoned.into_inner(),
            };
            pending.remove(&id_key(&request_id));
            return Err("permission request failed".to_string());
        }
        let response = wait_permission(&rx, &self.cancel);
        {
            let mut pending = match self.pending.lock() {
                Ok(pending) => pending,
                Err(poisoned) => poisoned.into_inner(),
            };
            pending.remove(&id_key(&request_id));
        }
        let decision = response.as_ref().and_then(permission_decision);
        match decision {
            Some("allow-once") => Ok(()),
            Some("allow-always") => {
                let mut memory = match self.perms.lock() {
                    Ok(memory) => memory,
                    Err(poisoned) => poisoned.into_inner(),
                };
                memory.allow.insert(key);
                Ok(())
            }
            Some("reject-always") => {
                let mut memory = match self.perms.lock() {
                    Ok(memory) => memory,
                    Err(poisoned) => poisoned.into_inner(),
                };
                memory.reject.insert(key);
                Err("permission denied".to_string())
            }
            Some("reject-once") | None => Err("permission denied".to_string()),
            Some(_) => Err("permission denied".to_string()),
        }
    }
}

fn wait_permission(rx: &Receiver<Value>, cancel: &AtomicBool) -> Option<Value> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(value) => {
                if cancel.load(Ordering::Relaxed) {
                    return None;
                }
                return Some(value);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return None,
        }
    }
}

fn permission_decision(message: &Value) -> Option<&str> {
    if message.get("error").is_some() {
        return None;
    }
    let outcome = message.pointer("/result/outcome")?;
    match outcome.get("outcome").and_then(Value::as_str) {
        Some("cancelled") => None,
        Some("selected") => outcome.get("optionId").and_then(Value::as_str),
        _ => None,
    }
}

fn permission_options() -> Value {
    json!([
        {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
        {"optionId": "allow-always", "name": "Allow for this session", "kind": "allow_always"},
        {"optionId": "reject-once", "name": "Reject once", "kind": "reject_once"},
        {"optionId": "reject-always", "name": "Reject for this session", "kind": "reject_always"}
    ])
}

fn lock_state(state: &Mutex<State>) -> std::sync::MutexGuard<'_, State> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn rpc_ok(id: &Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn rpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "_tag": "Cause",
            "code": code,
            "message": message,
            "data": [{
                "_tag": "Fail",
                "error": {
                    "code": code,
                    "message": message
                }
            }]
        }
    })
}

fn is_notification(message: &Value) -> bool {
    match message.get("id") {
        None | Some(Value::Null) => true,
        Some(Value::String(id)) => id.is_empty(),
        Some(_) => false,
    }
}

fn id_key(id: &Value) -> String {
    match id {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        other => other.to_string(),
    }
}

/// The further folders a session request names (`additionalDirectories`): absolute
/// paths of existing directories.
fn additional_directories(params: &Value) -> Result<Vec<PathBuf>, String> {
    let Some(list) = params.get("additionalDirectories") else {
        return Ok(Vec::new());
    };
    let list = list
        .as_array()
        .ok_or("additionalDirectories must be an array")?;
    list.iter()
        .map(|dir| {
            let dir = PathBuf::from(dir.as_str().ok_or("additionalDirectories holds paths")?);
            if !dir.is_absolute() || !dir.is_dir() {
                return Err(format!(
                    "additional directory {} is not an absolute path of a directory",
                    dir.display()
                ));
            }
            Ok(dir)
        })
        .collect()
}

fn require_cwd(params: &Value) -> Result<PathBuf, String> {
    let Some(raw) = params.get("cwd").and_then(Value::as_str) else {
        return Err("cwd is required".to_string());
    };
    if raw.trim().is_empty() {
        return Err("cwd is required".to_string());
    }
    let mut path = PathBuf::from(raw);
    if path.is_relative() {
        let current =
            std::env::current_dir().map_err(|error| format!("cannot resolve cwd: {error}"))?;
        path = current.join(path);
    }
    if !path.is_dir() {
        return Err(format!("cwd is not a directory: {}", path.display()));
    }
    Ok(path)
}

/// The MCP servers a session request names (`mcpServers`: stdio, or http).
fn mcp_servers(
    params: &Value,
    required: bool,
) -> Result<Vec<greppy_agent::mcp::McpServerConfig>, String> {
    match params.get("mcpServers") {
        None if required => Err("mcpServers is required".to_string()),
        None => Ok(Vec::new()),
        Some(list) => greppy_agent::mcp::McpServerConfig::parse_list(list),
    }
}

/// Start a session's MCP servers in its folder; a server that cannot start is left out
/// and said on stderr.
fn start_mcp(
    servers: &[greppy_agent::mcp::McpServerConfig],
    cwd: &Path,
) -> Arc<Mutex<greppy_agent::mcp::McpHub>> {
    let hub = greppy_agent::mcp::McpHub::start(servers, cwd);
    for note in &hub.notes {
        eprintln!("greppy agent stdio: {note}");
    }
    Arc::new(Mutex::new(hub))
}

/// Most bytes of base64 one prompt's pictures may hold (about 15 MB of pictures).
const PROMPT_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// A prompt's pictures (`{"type":"image","mimeType","data"}` blocks): png, jpeg, gif or
/// webp, as base64, after its text. A prompt is never only pictures.
fn prompt_images(blocks: &[Value]) -> Result<Vec<greppy_agent::ContentPart>, String> {
    let mut images = Vec::new();
    let mut bytes = 0usize;
    for block in blocks {
        // An embedded resource that is a picture goes to the model as one.
        let embedded = block.get("type").and_then(Value::as_str) == Some("resource")
            && block.pointer("/resource/blob").is_some()
            && is_picture_type(block.pointer("/resource/mimeType").and_then(Value::as_str));
        let converted;
        let block = if embedded {
            converted = json!({
                "type": "image",
                "mimeType": block.pointer("/resource/mimeType").cloned().unwrap_or_default(),
                "data": block.pointer("/resource/blob").cloned().unwrap_or_default(),
            });
            &converted
        } else if block.get("type").and_then(Value::as_str) == Some("image") {
            block
        } else {
            continue;
        };
        let media_type = block.get("mimeType").and_then(Value::as_str).unwrap_or("");
        if !matches!(
            media_type,
            "image/png" | "image/jpeg" | "image/gif" | "image/webp"
        ) {
            return Err(format!("unsupported image type: {media_type:?}"));
        }
        let data = block.get("data").and_then(Value::as_str).unwrap_or("");
        let base64 = !data.is_empty()
            && data.len() % 4 == 0
            && data
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='));
        if !base64 {
            return Err("image data must be base64".to_string());
        }
        bytes = bytes.saturating_add(data.len());
        if bytes > PROMPT_IMAGE_BYTES {
            return Err("the prompt's images are too large".to_string());
        }
        images.push(greppy_agent::ContentPart::Image {
            media_type: media_type.to_owned(),
            data: data.to_owned(),
        });
    }
    Ok(images)
}

/// A prompt's words for the model, in order: text blocks as they are, a linked
/// resource (`resource_link`) as a reference the model can follow (a local file by
/// its path, which greppy reads), an embedded resource (`resource`) with its text, or,
/// for binary content that is no picture, as a note. Pictures go separately
/// ([`prompt_images`]); audio is not taken (the capability says so). Empty where the
/// prompt is only pictures.
fn prompt_text(blocks: &[Value]) -> Result<String, String> {
    if blocks.is_empty() {
        return Err("prompt is empty".to_string());
    }
    let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
    for block in blocks {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        let field = |name: &str| block.get(name).and_then(Value::as_str);
        match kind {
            "image" => {}
            "text" => {
                let Some(text) = field("text") else {
                    return Err("text prompt block is missing text".to_string());
                };
                parts.push(text.to_owned());
            }
            "resource_link" => {
                let Some(uri) = field("uri").filter(|uri| !uri.is_empty()) else {
                    return Err("resource_link prompt block is missing uri".to_string());
                };
                let name = field("title").or(field("name")).unwrap_or(uri);
                let mut line = format!("[Linked resource: {name} <{uri}>");
                if let Some(path) = local_path(uri) {
                    line.push_str(&format!(", a local file: {path}"));
                }
                if let Some(mime) = field("mimeType") {
                    line.push_str(&format!(", {mime}"));
                }
                if let Some(description) = field("description") {
                    line.push_str(&format!(" — {description}"));
                }
                line.push(']');
                parts.push(line);
            }
            "resource" => {
                let resource = block.get("resource").cloned().unwrap_or(Value::Null);
                let Some(uri) = resource.get("uri").and_then(Value::as_str) else {
                    return Err("resource prompt block is missing resource.uri".to_string());
                };
                let mime = resource.get("mimeType").and_then(Value::as_str);
                match (
                    resource.get("text").and_then(Value::as_str),
                    resource.get("blob"),
                ) {
                    (Some(text), _) => parts.push(format!(
                        "<resource uri=\"{uri}\"{}>\n{text}\n</resource>",
                        mime.map(|mime| format!(" mimeType=\"{mime}\""))
                            .unwrap_or_default()
                    )),
                    (None, Some(Value::String(blob))) if !is_picture_type(mime) => {
                        parts.push(format!(
                            "[Embedded binary resource {uri} ({}, {} bytes as base64), not shown]",
                            mime.unwrap_or("unknown type"),
                            blob.len()
                        ))
                    }
                    (None, Some(Value::String(_))) => {}
                    _ => {
                        return Err(
                            "resource prompt block has neither resource.text nor resource.blob"
                                .to_string(),
                        )
                    }
                }
            }
            "" => return Err("unsupported prompt content type: unknown".to_string()),
            other => return Err(format!("unsupported prompt content type: {other}")),
        }
    }
    Ok(parts.join("\n"))
}

/// The local path a `file://` URI names (percent-decoding not attempted beyond `%20`).
fn local_path(uri: &str) -> Option<String> {
    let path = uri.strip_prefix("file://")?;
    let path = path.strip_prefix("localhost").unwrap_or(path);
    path.starts_with('/').then(|| path.replace("%20", " "))
}

fn is_picture_type(mime: Option<&str>) -> bool {
    matches!(
        mime,
        Some("image/png" | "image/jpeg" | "image/gif" | "image/webp")
    )
}

fn session_setup(session_id: &str, model: &str, include_id: bool, choices: &Choices) -> Value {
    let mut value = json!({
        "models": model_state(model),
        "configOptions": config_options(model, choices),
        "modes": {
            "currentModeId": choices.mode,
            "availableModes": MODES.iter().map(|(id, name, description)| json!({
                "id": id, "name": name, "description": description
            })).collect::<Vec<_>>()
        }
    });
    if include_id {
        value["sessionId"] = Value::String(session_id.to_string());
    }
    value
}

fn model_state(model: &str) -> Value {
    if model.is_empty() {
        return json!({
            "currentModelId": "",
            "availableModels": []
        });
    }
    json!({
        "currentModelId": model,
        "availableModels": [{
            "modelId": model,
            "name": model,
            "description": "Model id passed through to the local gateway"
        }]
    })
}

/// All of a session's config options: model, mode, reasoning effort, and greppy's
/// capabilities (each on or off; on puts its manual in the system prompt and unlocks
/// its commands, off takes both away again).
fn config_options(model: &str, choices: &Choices) -> Vec<Value> {
    let select = |id: &str,
                  name: &str,
                  description: &str,
                  category: Option<&str>,
                  current: &str,
                  options: &[(&str, &str)]| {
        let mut option = json!({
            "id": id, "name": name, "description": description, "type": "select",
            "currentValue": current,
            "options": options.iter().map(|(value, name)| json!({"value": value, "name": name})).collect::<Vec<_>>()
        });
        if let Some(category) = category {
            option["category"] = json!(category);
        }
        option
    };
    let on = |yes: bool| if yes { "on" } else { "off" };
    vec![
        model_config(model),
        select(
            "mode",
            "Mode",
            "How greppy asks before tools run",
            Some("mode"),
            choices.mode,
            &MODES.map(|(id, name, _)| (id, name)),
        ),
        select(
            "effort",
            "Reasoning effort",
            "How hard the model is to think",
            Some("thought_level"),
            choices.effort.as_deref().unwrap_or("default"),
            &[
                ("default", "Default"),
                ("low", "Low"),
                ("medium", "Medium"),
                ("high", "High"),
            ],
        ),
        select(
            "web",
            "Web",
            "Browse and read web pages (greppy web); needs network",
            None,
            on(choices.capabilities.web),
            &[("on", "On"), ("off", "Off")],
        ),
        select(
            "desktop",
            "Desktop",
            "See and operate desktop apps (greppy desktop)",
            None,
            on(choices.capabilities.desktop),
            &[("on", "On"), ("off", "Off")],
        ),
    ]
}

fn model_config(model: &str) -> Value {
    let current = if model.is_empty() { "unset" } else { model };
    json!({
        "id": "model",
        "name": "Model",
        "description": "Model id passed through to the local gateway",
        "category": "model",
        "type": "select",
        "currentValue": current,
        "options": [{
            "value": current,
            "name": current
        }]
    })
}

fn unique_session_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{n}", new_session_id())
}

fn same_dir(stored: &Path, requested: &Path) -> bool {
    if stored.as_os_str().is_empty() {
        return false;
    }
    match (stored.canonicalize(), requested.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => stored == requested,
    }
}

fn active_tool_call_id(fallback: u64) -> String {
    ACTIVE_TOOL_CALL
        .with(|slot| slot.borrow().clone())
        .unwrap_or_else(|| format!("perm-{fallback}"))
}

fn perm_key(name: &str, arguments: &Value) -> String {
    format!("{name}:{}", tool_kind(name, arguments))
}

fn tool_kind(name: &str, arguments: &Value) -> &'static str {
    if name != "greppy" {
        return "other";
    }
    let arg0 = arguments
        .get("args")
        .and_then(Value::as_array)
        .and_then(|args| args.first())
        .and_then(Value::as_str)
        .unwrap_or("");
    match arg0 {
        "read" | "read-file" | "read-smart" => "read",
        "replace" | "replace-text" | "replace-lines" | "replace-span" | "write" | "delete"
        | "delete-lines" | "insert-lines" | "patch" | "rename" | "undo" => "edit",
        "search" | "search-symbol" | "search-pattern" | "search-graph" | "plus" | "who-calls"
        | "callees" | "impact" | "path" | "brief" | "fan-in" | "fan-out" => "search",
        "bash-smart" => "execute",
        _ => "other",
    }
}

fn tool_title(name: &str, arguments: &Value) -> String {
    let args = arguments
        .get("args")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    if args.is_empty() {
        name.to_string()
    } else {
        truncate_chars(&format!("{name} {args}"), 160)
    }
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_millis())
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn stop_reason(stop: &LoopStop) -> &'static str {
    match stop {
        LoopStop::EndTurn => "end_turn",
        LoopStop::MaxTokens => "max_tokens",
        LoopStop::MaxTurns | LoopStop::Stuck | LoopStop::Deadline => "max_turn_requests",
        LoopStop::Cancelled => "cancelled",
    }
}

fn loop_error_message(error: &LoopError) -> String {
    match error {
        LoopError::Transport(message) => format!("model transport failed: {message}"),
        LoopError::Http { status, body } => {
            format!(
                "model gateway returned HTTP {status}: {}",
                truncate_chars(body, 400)
            )
        }
        LoopError::Stream(message) => format!("model stream failed: {message}"),
        LoopError::Incomplete(message) => format!("model stream incomplete: {message}"),
        LoopError::Client(message) => format!("model client failed: {message}"),
    }
}
