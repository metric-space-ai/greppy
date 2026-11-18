//! MCP client: the tools of Model Context Protocol servers a caller names (ACP
//! `session/new` `mcpServers`), offered to the model beside greppy's own.
//!
//! Transports:
//! - **stdio** — the server is a child process; JSON-RPC messages are single lines on
//!   its stdin/stdout (MCP stdio transport). Its stderr is passed through.
//! - **http** — MCP "streamable HTTP": every request is a POST of one JSON-RPC message;
//!   the answer is either JSON or an SSE stream that carries it. A session id the server
//!   sets (`Mcp-Session-Id`) goes along with every later request. Plain HTTP only:
//!   greppy's HTTP client has no TLS stack.
//!
//! The model sees each tool as `mcp__<server>__<tool>` (letters, digits, `_` and `-`,
//! at most 64 characters). A server that fails to start or to list its tools is left
//! out with a note; a failing call is the tool's error, never the run's.
//!
//! Requests a server makes of its client: `roots/list` is answered with the session's
//! folder, `ping` with `{}`, anything else with "method not found". Notifications are
//! ignored.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::env::{ExecutionEnv, ToolOutcome};
use crate::protocol::ToolDefinition;

/// The MCP protocol version greppy speaks first (a server may answer with an older one).
pub const PROTOCOL_VERSION: &str = "2025-06-18";
/// How long a server may take to start and list its tools.
const SETUP_LIMIT: Duration = Duration::from_secs(60);
/// How long one tool call may take.
const CALL_LIMIT: Duration = Duration::from_secs(600);
/// Most tools greppy takes from one server.
const TOOLS_PER_SERVER: usize = 128;

/// One server as the caller names it (ACP `McpServer`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerConfig {
    Stdio {
        name: String,
        command: PathBuf,
        args: Vec<String>,
        env: Vec<(String, String)>,
    },
    Http {
        name: String,
        url: String,
        headers: Vec<(String, String)>,
    },
}

impl McpServerConfig {
    pub fn name(&self) -> &str {
        match self {
            Self::Stdio { name, .. } | Self::Http { name, .. } => name,
        }
    }

    /// The servers an ACP `mcpServers` list names. `sse` (deprecated in MCP) is refused.
    pub fn parse_list(list: &Value) -> Result<Vec<Self>, String> {
        let Some(items) = list.as_array() else {
            return Err("mcpServers must be an array".to_string());
        };
        let pairs = |value: &Value, what: &str| -> Result<Vec<(String, String)>, String> {
            value
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .map(|item| {
                            match (
                                item.get("name").and_then(Value::as_str),
                                item.get("value").and_then(Value::as_str),
                            ) {
                                (Some(name), Some(value)) => {
                                    Ok((name.to_owned(), value.to_owned()))
                                }
                                _ => Err(format!("{what} entries need name and value")),
                            }
                        })
                        .collect()
                })
                .unwrap_or_else(|| Ok(Vec::new()))
        };
        let mut servers: Vec<Self> = Vec::with_capacity(items.len());
        for item in items {
            let Some(name) = item
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.trim().is_empty())
            else {
                return Err("an MCP server needs a name".to_string());
            };
            if servers.iter().any(|known| known.name() == name) {
                return Err(format!("MCP server {name} is named twice"));
            }
            let server = match item.get("type").and_then(Value::as_str) {
                None | Some("stdio") => {
                    let Some(command) = item.get("command").and_then(Value::as_str) else {
                        return Err(format!("MCP server {name} needs a command"));
                    };
                    let command = PathBuf::from(command);
                    if !command.is_absolute() {
                        return Err(format!(
                            "MCP server {name}: the command must be an absolute path"
                        ));
                    }
                    let args = item
                        .get("args")
                        .and_then(Value::as_array)
                        .map(|args| {
                            args.iter()
                                .filter_map(Value::as_str)
                                .map(str::to_owned)
                                .collect()
                        })
                        .unwrap_or_default();
                    let env = pairs(item.get("env").unwrap_or(&Value::Null), "env")?;
                    Self::Stdio {
                        name: name.to_owned(),
                        command,
                        args,
                        env,
                    }
                }
                Some("http") => {
                    let Some(url) = item.get("url").and_then(Value::as_str) else {
                        return Err(format!("MCP server {name} needs a url"));
                    };
                    if !url.starts_with("http://") {
                        return Err(format!(
                            "MCP server {name}: only plain http:// URLs are supported (greppy has no TLS stack)"
                        ));
                    }
                    let headers = pairs(item.get("headers").unwrap_or(&Value::Null), "headers")?;
                    Self::Http {
                        name: name.to_owned(),
                        url: url.to_owned(),
                        headers,
                    }
                }
                Some(other) => {
                    return Err(format!(
                        "MCP server {name}: transport {other} is not supported"
                    ))
                }
            };
            servers.push(server);
        }
        Ok(servers)
    }
}

/// The model's name for a server's tool: `mcp__<server>__<tool>`, reduced to letters,
/// digits, `_` and `-`, at most 64 characters (a longer one keeps its head and a hash).
pub fn tool_name(server: &str, tool: &str) -> String {
    let clean = |text: &str| {
        text.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
    };
    let name = format!("mcp__{}__{}", clean(server), clean(tool));
    if name.len() <= 64 {
        return name;
    }
    let hash = blake3::hash(name.as_bytes()).to_hex();
    format!("{}_{}", &name[..55], &hash[..8])
}

/// How a server is spoken to.
enum Transport {
    Stdio {
        child: Child,
        input: ChildStdin,
        lines: Receiver<String>,
    },
    Http {
        url: String,
        headers: Vec<(String, String)>,
        session: Option<String>,
    },
}

/// One running server and its tools.
struct Server {
    name: String,
    transport: Transport,
    next_id: u64,
    root: PathBuf,
    /// (model's name, server's name, definition).
    tools: Vec<(String, String, ToolDefinition)>,
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Transport::Stdio { child, .. } = &mut self.transport {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Server {
    fn start(config: &McpServerConfig, root: &Path) -> Result<Self, String> {
        let transport = match config {
            McpServerConfig::Stdio {
                command, args, env, ..
            } => {
                let mut child = Command::new(command)
                    .args(args)
                    .envs(
                        env.iter()
                            .map(|(name, value)| (name.as_str(), value.as_str())),
                    )
                    .current_dir(root)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .map_err(|error| format!("start {}: {error}", command.display()))?;
                let output = BufReader::new(child.stdout.take().ok_or("no stdout")?);
                let input = child.stdin.take().ok_or("no stdin")?;
                let (tell, lines) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    for line in output.lines() {
                        let Ok(line) = line else { break };
                        if tell.send(line).is_err() {
                            break;
                        }
                    }
                });
                Transport::Stdio {
                    child,
                    input,
                    lines,
                }
            }
            McpServerConfig::Http { url, headers, .. } => Transport::Http {
                url: url.clone(),
                headers: headers.clone(),
                session: None,
            },
        };
        let mut server = Self {
            name: config.name().to_owned(),
            transport,
            next_id: 1,
            root: root.to_path_buf(),
            tools: Vec::new(),
        };
        server.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"roots": {"listChanged": false}},
                "clientInfo": {"name": "greppy", "version": env!("CARGO_PKG_VERSION")}
            }),
            SETUP_LIMIT,
        )?;
        server.notify("notifications/initialized", json!({}))?;
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(cursor) => json!({"cursor": cursor}),
                None => json!({}),
            };
            let page = server.request("tools/list", params, SETUP_LIMIT)?;
            for tool in page
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(name) = tool.get("name").and_then(Value::as_str) else {
                    continue;
                };
                if server.tools.len() >= TOOLS_PER_SERVER {
                    break;
                }
                let schema = tool
                    .get("inputSchema")
                    .filter(|schema| schema.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
                let description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .filter(|d| !d.trim().is_empty())
                    .map_or_else(
                        || format!("Tool {name} of the MCP server {}.", server.name),
                        str::to_owned,
                    );
                let exposed = tool_name(&server.name, name);
                server.tools.push((
                    exposed.clone(),
                    name.to_owned(),
                    ToolDefinition {
                        name: exposed,
                        description,
                        input_schema: schema,
                    },
                ));
            }
            cursor = page
                .get("nextCursor")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if cursor.is_none() || server.tools.len() >= TOOLS_PER_SERVER {
                break;
            }
        }
        Ok(server)
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        let message = json!({"jsonrpc": "2.0", "method": method, "params": params});
        match &mut self.transport {
            Transport::Stdio { input, .. } => writeln!(input, "{message}")
                .and_then(|()| input.flush())
                .map_err(|error| format!("write to MCP server {}: {error}", self.name)),
            Transport::Http { .. } => self.post(&message, Duration::from_secs(30)).map(|_| ()),
        }
    }

    /// One request; its result, or the server's error as text.
    fn request(&mut self, method: &str, params: Value, limit: Duration) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let answer = match &mut self.transport {
            Transport::Stdio { input, .. } => {
                writeln!(input, "{message}")
                    .and_then(|()| input.flush())
                    .map_err(|error| format!("write to MCP server {}: {error}", self.name))?;
                self.stdio_answer(id, limit)?
            }
            Transport::Http { .. } => self
                .post(&message, limit)?
                .into_iter()
                .find(|frame| frame.get("id") == Some(&json!(id)))
                .ok_or_else(|| format!("MCP server {} did not answer {method}", self.name))?,
        };
        if let Some(error) = answer.get("error") {
            return Err(format!(
                "MCP server {}: {}",
                self.name,
                error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("error")
            ));
        }
        Ok(answer.get("result").cloned().unwrap_or(Value::Null))
    }

    /// The answer to `id` on stdio, answering the server's own requests meanwhile.
    fn stdio_answer(&mut self, id: u64, limit: Duration) -> Result<Value, String> {
        let deadline = Instant::now() + limit;
        loop {
            let Transport::Stdio { lines, .. } = &self.transport else {
                unreachable!()
            };
            let line = match lines.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(format!(
                        "MCP server {} did not answer within {limit:?}",
                        self.name
                    ))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(format!("MCP server {} ended", self.name))
                }
            };
            let Ok(frame) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if frame.get("method").is_some() {
                if let Some(reply) = self.answer_server_request(&frame) {
                    let Transport::Stdio { input, .. } = &mut self.transport else {
                        unreachable!()
                    };
                    let _ = writeln!(input, "{reply}").and_then(|()| input.flush());
                }
                continue;
            }
            if frame.get("id") == Some(&json!(id)) {
                return Ok(frame);
            }
        }
    }

    /// The reply to a request a server makes of its client (none for a notification).
    fn answer_server_request(&self, frame: &Value) -> Option<Value> {
        let id = frame.get("id")?.clone();
        let result = match frame.get("method").and_then(Value::as_str) {
            Some("roots/list") => json!({"roots": [{
                "uri": format!("file://{}", self.root.display()),
                "name": self.root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            }]}),
            Some("ping") => json!({}),
            _ => {
                return Some(json!({"jsonrpc": "2.0", "id": id,
                    "error": {"code": -32601, "message": "not offered by greppy"}}))
            }
        };
        Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    /// POST one message (streamable HTTP); the JSON-RPC messages the answer carries.
    fn post(&mut self, message: &Value, limit: Duration) -> Result<Vec<Value>, String> {
        let Transport::Http {
            url,
            headers,
            session,
        } = &mut self.transport
        else {
            unreachable!()
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| format!("MCP runtime: {error}"))?;
        let name = self.name.clone();
        let result = runtime.block_on(async {
            let client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(limit)
                .build()
                .map_err(|error| format!("MCP client: {error}"))?;
            let mut request = client
                .post(url.as_str())
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .header("MCP-Protocol-Version", PROTOCOL_VERSION)
                .body(message.to_string());
            for (header, value) in headers.iter() {
                request = request.header(header.as_str(), value.as_str());
            }
            if let Some(session) = session.as_deref() {
                request = request.header("Mcp-Session-Id", session);
            }
            let response = request
                .send()
                .await
                .map_err(|error| format!("MCP server {name}: {error}"))?;
            let status = response.status();
            let new_session = response
                .headers()
                .get("mcp-session-id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            let event_stream = response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"));
            let body = response
                .text()
                .await
                .map_err(|error| format!("MCP server {name}: {error}"))?;
            if !status.is_success() {
                return Err(format!(
                    "MCP server {name}: HTTP {}: {}",
                    status.as_u16(),
                    body.chars().take(300).collect::<String>()
                ));
            }
            Ok((new_session, event_stream, body))
        });
        let (new_session, event_stream, body) = result?;
        if new_session.is_some() {
            *session = new_session;
        }
        if body.trim().is_empty() {
            return Ok(Vec::new());
        }
        let parse = |text: &str| -> Vec<Value> {
            match serde_json::from_str::<Value>(text) {
                Ok(Value::Array(items)) => items,
                Ok(value) => vec![value],
                Err(_) => Vec::new(),
            }
        };
        if !event_stream {
            return Ok(parse(&body));
        }
        // SSE: the data lines of each event, joined.
        let mut frames = Vec::new();
        for event in body.split("\n\n") {
            let data: Vec<&str> = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.strip_prefix(' ').unwrap_or(line))
                .collect();
            if !data.is_empty() {
                frames.extend(parse(&data.join("\n")));
            }
        }
        Ok(frames)
    }

    fn call(&mut self, tool: &str, arguments: &Value) -> ToolOutcome {
        let arguments = if arguments.is_object() {
            arguments.clone()
        } else {
            json!({})
        };
        match self.request(
            "tools/call",
            json!({"name": tool, "arguments": arguments}),
            CALL_LIMIT,
        ) {
            Ok(result) => {
                let mut text = Vec::new();
                let mut image = None;
                for part in result
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") => text.push(
                            part.get("text")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_owned(),
                        ),
                        Some("image")
                            if image.is_none()
                                && part.get("mimeType").and_then(Value::as_str)
                                    == Some("image/png") =>
                        {
                            image = part.get("data").and_then(Value::as_str).map(str::to_owned);
                        }
                        Some("resource") => {
                            let resource = part.get("resource").cloned().unwrap_or_default();
                            match resource.get("text").and_then(Value::as_str) {
                                Some(body) => text.push(format!(
                                    "<resource uri=\"{}\">\n{body}\n</resource>",
                                    resource.get("uri").and_then(Value::as_str).unwrap_or("")
                                )),
                                None => text.push(format!(
                                    "[binary resource {}]",
                                    resource.get("uri").and_then(Value::as_str).unwrap_or("")
                                )),
                            }
                        }
                        Some("resource_link") => text.push(format!(
                            "[Linked resource: {} <{}>]",
                            part.get("name").and_then(Value::as_str).unwrap_or(""),
                            part.get("uri").and_then(Value::as_str).unwrap_or("")
                        )),
                        Some(other) => text.push(format!("[{other} content not shown]")),
                        None => {}
                    }
                }
                if text.is_empty() {
                    if let Some(structured) = result.get("structuredContent") {
                        text.push(structured.to_string());
                    }
                }
                let mut outcome = if result.get("isError") == Some(&json!(true)) {
                    ToolOutcome::err(text.join("\n"))
                } else {
                    ToolOutcome::ok(text.join("\n"))
                };
                outcome.image_png_base64 = image;
                outcome
            }
            Err(error) => ToolOutcome::err(error),
        }
    }
}

/// The servers of a session, started once; their tools by the model's names.
#[derive(Default)]
pub struct McpHub {
    servers: Vec<Server>,
    /// Model's tool name → (server index, server's tool name).
    routes: HashMap<String, (usize, String)>,
    /// Why a server is left out, per server.
    pub notes: Vec<String>,
}

impl std::fmt::Debug for McpHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpHub")
            .field(
                "servers",
                &self.servers.iter().map(|s| &s.name).collect::<Vec<_>>(),
            )
            .field("tools", &self.routes.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl McpHub {
    /// Start every server in `root`; one that fails is left out (see `notes`).
    pub fn start(configs: &[McpServerConfig], root: &Path) -> Self {
        let mut hub = Self::default();
        for config in configs {
            match Server::start(config, root) {
                Ok(server) => {
                    let index = hub.servers.len();
                    for (exposed, original, _) in &server.tools {
                        if hub.routes.contains_key(exposed) {
                            hub.notes.push(format!(
                                "MCP tool {exposed} is named twice; the second is left out"
                            ));
                            continue;
                        }
                        hub.routes
                            .insert(exposed.clone(), (index, original.clone()));
                    }
                    hub.servers.push(server);
                }
                Err(error) => hub
                    .notes
                    .push(format!("MCP server {} is left out: {error}", config.name())),
            }
        }
        hub
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.servers
            .iter()
            .flat_map(|server| server.tools.iter())
            .filter(|(exposed, _, _)| self.routes.contains_key(exposed))
            .map(|(_, _, definition)| definition.clone())
            .collect()
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.routes.contains_key(name)
    }

    pub fn call(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        match self.routes.get(name).cloned() {
            Some((index, original)) => self.servers[index].call(&original, arguments),
            None => ToolOutcome::unknown_tool(name),
        }
    }
}

/// An execution environment with the tools of MCP servers beside `inner`'s. A server
/// tool never takes the place of one of `inner`'s.
pub struct McpEnv {
    pub inner: Box<dyn ExecutionEnv + Send>,
    pub hub: std::sync::Arc<std::sync::Mutex<McpHub>>,
}

impl std::fmt::Debug for McpEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpEnv").field("hub", &self.hub).finish()
    }
}

impl ExecutionEnv for McpEnv {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut tools = self.inner.tool_definitions();
        let own: Vec<String> = tools.iter().map(|tool| tool.name.clone()).collect();
        if let Ok(hub) = self.hub.lock() {
            tools.extend(
                hub.tool_definitions()
                    .into_iter()
                    .filter(|tool| !own.contains(&tool.name)),
            );
        }
        tools
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        let own = self
            .inner
            .tool_definitions()
            .iter()
            .any(|tool| tool.name == name);
        if !own {
            if let Ok(mut hub) = self.hub.lock() {
                if hub.has_tool(name) {
                    return hub.call(name, arguments);
                }
            }
        }
        self.inner.call_tool(name, arguments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn servers_are_parsed_as_acp_names_them_and_bad_ones_refused() {
        let servers = McpServerConfig::parse_list(&json!([
            {"name": "notes", "command": "/usr/bin/made-up", "args": ["--x"], "env": [{"name": "A", "value": "1"}]},
            {"type": "http", "name": "remote", "url": "http://127.0.0.1:9/mcp", "headers": [{"name": "X-Key", "value": "k"}]}
        ]))
        .unwrap();
        assert_eq!(
            servers,
            [
                McpServerConfig::Stdio {
                    name: "notes".into(),
                    command: "/usr/bin/made-up".into(),
                    args: vec!["--x".into()],
                    env: vec![("A".into(), "1".into())]
                },
                McpServerConfig::Http {
                    name: "remote".into(),
                    url: "http://127.0.0.1:9/mcp".into(),
                    headers: vec![("X-Key".into(), "k".into())]
                }
            ]
        );
        for broken in [
            json!([{"name": "a", "command": "relative"}]),
            json!([{"name": "a", "command": "/x"}, {"name": "a", "command": "/y"}]),
            json!([{"type": "sse", "name": "a", "url": "http://x"}]),
            json!([{"type": "http", "name": "a", "url": "https://x"}]),
            json!([{"command": "/x"}]),
            json!({"name": "a"}),
        ] {
            assert!(McpServerConfig::parse_list(&broken).is_err(), "{broken}");
        }
    }

    #[test]
    fn tool_names_are_plain_and_at_most_64_characters() {
        assert_eq!(tool_name("notes", "look_up"), "mcp__notes__look_up");
        assert_eq!(tool_name("my server", "a.b/c"), "mcp__my_server__a_b_c");
        let long = tool_name("server", &"x".repeat(100));
        assert_eq!(long.len(), 64);
        assert!(long
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
        assert_ne!(long, tool_name("server", &"x".repeat(101)));
    }
}
