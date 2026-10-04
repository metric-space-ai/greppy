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
    run_agent_loop_with_history, AgentConfig, Client, ClientError, ExecutionEnv, GreppyEnv,
    LoopError, LoopEvent, LoopStop, Message, ModelRequest, ModelStream, StreamEvent, ToolOutcome,
    TurnResult, SYSTEM_PROMPT,
};
use serde_json::{json, Value};

#[cfg(test)]
#[path = "agent_acp_tests.rs"]
mod tests;

use crate::agent::{EXIT_OK, EXIT_USAGE};
use crate::agent_tui::{
    messages_from_protocol, new_session_id, protocol_from_persisted, SessionRecord, SessionStore,
};

const PROTOCOL_VERSION: u64 = 1;
const AUTH_METHOD_ID: &str = "greppy.env";
const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8317";
const DEFAULT_MAX_TURNS: usize = 40;

thread_local! {
    static ACTIVE_TOOL_CALL: RefCell<Option<String>> = RefCell::new(None);
}

/// Process-wide ACP settings. The tool factory is optional so tests can
/// substitute a sandbox that records calls without invoking greppy.
#[derive(Clone)]
pub(crate) struct AcpConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub max_turns: usize,
    pub data_root: Option<PathBuf>,
    #[cfg(test)]
    pub after_messages: Option<Arc<dyn Fn() -> io::Result<()> + Send + Sync>>,
    pub tool_env: Option<Arc<dyn Fn(&Path) -> Box<dyn ExecutionEnv + Send> + Send + Sync>>,
}

impl Default for AcpConfig {
    fn default() -> Self {
        Self {
            endpoint: std::env::var("GREPPY_ENDPOINT")
                .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string()),
            model: std::env::var("GREPPY_MODEL").unwrap_or_default(),
            api_key: std::env::var("GREPPY_API_KEY")
                .ok()
                .filter(|key| !key.is_empty()),
            max_turns: DEFAULT_MAX_TURNS,
            data_root: None,
            #[cfg(test)]
            after_messages: None,
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
    /// Maximum assistant turns for one prompt.
    #[arg(long, default_value_t = DEFAULT_MAX_TURNS, value_name = "N")]
    max_turns: usize,
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
    let mut config = AcpConfig::default();
    config.endpoint = args.endpoint;
    if let Some(model) = args.model {
        config.model = model;
    }
    config.max_turns = args.max_turns;
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
}

struct Session {
    id: String,
    cwd: PathBuf,
    data_root: PathBuf,
    project: String,
    model: String,
    messages: Vec<Message>,
    cancel: Arc<AtomicBool>,
    busy: bool,
    closed: bool,
    perms: Arc<Mutex<PermMemory>>,
    usage_in: u64,
    usage_out: u64,
}

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
        let data_root = config.data_root.clone().unwrap_or_else(greppy_core::cache::data_root);
        Self {
            data_root,
            out: Out {
                inner: Arc::new(Mutex::new(output)),
            },
            state: Arc::new(Mutex::new(State {
                initialized: false,
                sessions: HashMap::new(),
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
        lock_state(&self.state).initialized = true;
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
                    "loadSession": true,
                    "promptCapabilities": {
                        "image": false,
                        "audio": false,
                        "embeddedContext": false
                    },
                    "mcpCapabilities": {
                        "http": false,
                        "sse": false
                    },
                    "sessionCapabilities": {
                        "list": {},
                        "resume": {},
                        "close": {}
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

    fn session_new(&self, id: &Value, params: &Value) -> Value {
        if let Err(message) = require_empty_mcp(params, true) {
            return rpc_error(id, -32602, &message);
        }
        let cwd = match require_cwd(params) {
            Ok(cwd) => cwd,
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
        let session = Session {
            id: session_id.clone(),
            cwd,
            data_root,
            project,
            model: model.clone(),
            messages: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            busy: false,
            closed: false,
            perms: Arc::new(Mutex::new(PermMemory {
                allow: HashSet::new(),
                reject: HashSet::new(),
            })),
            usage_in: 0,
            usage_out: 0,
        };
        lock_state(&self.state)
            .sessions
            .insert(session_id.clone(), session);
        rpc_ok(id, session_setup(&session_id, &model, true))
    }

    fn session_load(&self, id: &Value, params: &Value, method: &str) -> Value {
        let mcp_required = method == "session/load";
        if let Err(message) = require_empty_mcp(params, mcp_required) {
            return rpc_error(id, -32602, &message);
        }
        let cwd = match require_cwd(params) {
            Ok(cwd) => cwd,
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
        self.emit_replay(session_id, &messages);
        let model = if record.model.is_empty() {
            self.config.model.clone()
        } else {
            record.model.clone()
        };
        let session = Session {
            id: session_id.to_string(),
            cwd,
            data_root,
            project,
            model: model.clone(),
            messages,
            cancel: Arc::new(AtomicBool::new(false)),
            busy: false,
            closed: false,
            perms: Arc::new(Mutex::new(PermMemory {
                allow: HashSet::new(),
                reject: HashSet::new(),
            })),
            usage_in: record.usage.input_tokens,
            usage_out: record.usage.output_tokens,
        };
        lock_state(&self.state)
            .sessions
            .insert(session_id.to_string(), session);
        rpc_ok(id, session_setup(session_id, &model, false))
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

    fn session_set_config(&self, id: &Value, params: &Value) -> Value {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "sessionId is required");
        };
        let Some(config_id) = params.get("configId").and_then(Value::as_str) else {
            return rpc_error(id, -32602, "configId is required");
        };
        if config_id != "model" {
            return rpc_error(
                id,
                -32602,
                &format!("unsupported session config option: {config_id}"),
            );
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
            PreparedPrompt {
                session_id: session.id.clone(),
                cwd: session.cwd.clone(),
                data_root: session.data_root.clone(),
                project: session.project.clone(),
                model: session.model.clone(),
                history: session.messages.clone(),
                cancel: Arc::clone(&session.cancel),
                perms: Arc::clone(&session.perms),
            }
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
    cancel: Arc<AtomicBool>,
    perms: Arc<Mutex<PermMemory>>,
}

struct PromptDone {
    messages: Vec<Message>,
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
    let mut client = CancelModel {
        client,
        cancel: Arc::clone(&prepared.cancel),
    };

    let inner: Box<dyn ExecutionEnv + Send> = if let Some(factory) = &config.tool_env {
        factory(&prepared.cwd)
    } else {
        Box::new(
            GreppyEnv::new(prepared.cwd.clone())
                .map_err(|error| format!("cannot build the tool environment: {error}"))?,
        )
    };
    let mut env = GatingEnv {
        inner,
        session_id: prepared.session_id.clone(),
        perms: Arc::clone(&prepared.perms),
        cancel: Arc::clone(&prepared.cancel),
        out: out.clone(),
        pending: Arc::clone(pending),
        next_request_id: Arc::clone(next_request_id),
    };
    let agent_config = AgentConfig {
        max_turns: config.max_turns,
        system: Some(SYSTEM_PROMPT.to_string()),
        model: prepared.model.clone(),
        cancel: Some(Arc::clone(&prepared.cancel)),
        ..AgentConfig::default()
    };
    let session_id = prepared.session_id.clone();
    let mut on_event = |event: LoopEvent| {
        if let LoopEvent::ToolStart { call_id, .. } = &event {
            let call_id = call_id.clone();
            ACTIVE_TOOL_CALL.with(|slot| *slot.borrow_mut() = Some(call_id));
        }
        let _ = emit_loop_event(out, &session_id, &event);
    };
    match run_agent_loop_with_history(
        &mut client,
        &mut env,
        &agent_config,
        &prepared.history,
        prompt,
        &mut on_event,
    ) {
        Ok(result) => Ok(PromptDone {
            messages: result.messages,
            stop_reason: stop_reason(&result.stop),
            usage: result.usage,
        }),
        Err(error) => Err(loop_error_message(&error)),
    }
}

fn emit_loop_event(out: &Out, session_id: &str, event: &LoopEvent) -> io::Result<()> {
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
            "rawInput": arguments
        }),
        LoopEvent::ToolFinish {
            call_id, outcome, ..
        } => json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": if outcome.is_error { "failed" } else { "completed" },
            "rawOutput": truncate_chars(&outcome.content, 16_384)
        }),
        _ => return Ok(()),
    };
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
        done.messages.iter().flat_map(|message| message.content.iter()).find_map(|part| match part {
            greppy_agent::ContentPart::Text { text } => Some(truncate_chars(text, 80)),
            _ => None,
        }).unwrap_or_else(|| "untitled".to_string())
    });
    store.commit_turn(
        &prepared.session_id,
        &messages_from_protocol(&prepared.history),
        &messages_from_protocol(&done.messages),
        &done.usage,
        done.stop_reason,
        title.as_deref(),
        || {
            #[cfg(test)]
            if let Some(hook) = &_config.after_messages {
                hook()?;
            }
            Ok(())
        },
    ).map_err(|error| format!("cannot persist session history: {error}"))
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
        session.usage_in = session.usage_in.saturating_add(done.usage.input_tokens);
        session.usage_out = session.usage_out.saturating_add(done.usage.output_tokens);
    }
    Ok(())
}

struct GatingEnv {
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
        match self.authorize(name, arguments) {
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

fn require_empty_mcp(params: &Value, required: bool) -> Result<(), String> {
    match params.get("mcpServers") {
        None if required => Err("mcpServers is required".to_string()),
        None => Ok(()),
        Some(Value::Array(items)) if items.is_empty() => Ok(()),
        Some(Value::Array(_)) => {
            Err("MCP servers are not supported; send an empty mcpServers array".to_string())
        }
        Some(_) => Err("mcpServers must be an array".to_string()),
    }
}

fn prompt_text(blocks: &[Value]) -> Result<String, String> {
    if blocks.is_empty() {
        return Err("prompt is empty".to_string());
    }
    let mut parts = Vec::with_capacity(blocks.len());
    for block in blocks {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "text" {
            let label = if kind.is_empty() { "unknown" } else { kind };
            return Err(format!("unsupported prompt content type: {label}"));
        }
        let Some(text) = block.get("text").and_then(Value::as_str) else {
            return Err("text prompt block is missing text".to_string());
        };
        parts.push(text);
    }
    let joined = parts.join("\n");
    if joined.trim().is_empty() {
        return Err("prompt text is empty".to_string());
    }
    Ok(joined)
}

fn session_setup(session_id: &str, model: &str, include_id: bool) -> Value {
    let mut value = json!({
        "models": model_state(model),
        "configOptions": [model_config(model)]
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
