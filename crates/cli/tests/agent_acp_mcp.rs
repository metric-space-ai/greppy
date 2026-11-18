//! `greppy agent stdio` as an ACP agent with MCP servers, end to end: the real binary,
//! a stand-in Messages model on loopback, one stdio MCP server (a small Python script)
//! and one streamable-HTTP MCP server (in this test). The model is offered both
//! servers' tools, calls each, and is given what they answered; a linked resource in
//! the prompt reaches it as a reference. Everything here is made up.
#![cfg(unix)]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the model was asked, request by request.
type Asked = Arc<Mutex<Vec<Value>>>;

/// The steering word of the latest user message: "CALL: <tool>" has the model call that
/// tool (with `{"query": "made-up"}`), unless a tool answered since.
fn wanted(request: &Value) -> Option<String> {
    let last = request["messages"].as_array()?.last()?;
    let blocks = last["content"].as_array()?;
    if blocks.iter().any(|block| block["type"] == "tool_result") {
        return None;
    }
    blocks
        .iter()
        .filter_map(|block| block["text"].as_str())
        .find_map(|text| {
            Some(
                text.split_once("CALL: ")?
                    .1
                    .lines()
                    .next()?
                    .trim()
                    .to_owned(),
            )
        })
}

fn read_http(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let (mut head, mut length) = (String::new(), 0usize);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap();
        }
        head.push_str(&line);
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    (head, body)
}

/// A stand-in Messages model; its port.
fn model(asked: &Asked) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let asked = asked.clone();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let (head, body) = read_http(&mut stream);
            if !head.starts_with("POST") {
                continue;
            }
            let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let call = wanted(&request);
            asked.lock().unwrap().push(request);
            let mut frames = vec![
                json!({"type": "message_start", "message": {"id": "m", "type": "message",
                "role": "assistant", "model": "made-up", "content": [], "stop_reason": null,
                "usage": {"input_tokens": 10, "output_tokens": 1}}}),
            ];
            let stop = match call {
                Some(tool) => {
                    frames.push(
                        json!({"type": "content_block_start", "index": 0, "content_block":
                        {"type": "tool_use", "id": "toolu_1", "name": tool, "input": {}}}),
                    );
                    frames.push(json!({"type": "content_block_delta", "index": 0, "delta":
                        {"type": "input_json_delta", "partial_json": "{\"query\":\"made-up\"}"}}));
                    "tool_use"
                }
                None => {
                    frames.push(
                        json!({"type": "content_block_start", "index": 0, "content_block":
                        {"type": "text", "text": ""}}),
                    );
                    frames.push(json!({"type": "content_block_delta", "index": 0, "delta":
                        {"type": "text_delta", "text": "done"}}));
                    "end_turn"
                }
            };
            frames.push(json!({"type": "content_block_stop", "index": 0}));
            frames.push(json!({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}}));
            frames.push(json!({"type": "message_stop"}));
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n"
            );
            for frame in frames {
                let _ = write!(
                    stream,
                    "event: {}\ndata: {frame}\n\n",
                    frame["type"].as_str().unwrap()
                );
            }
        }
    });
    port
}

/// A streamable-HTTP MCP server with one tool, `release_date`; its URL. It sets a
/// session id at `initialize` and refuses any later request without it.
fn http_mcp(seen: &Arc<Mutex<Vec<String>>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let seen = seen.clone();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let (head, body) = read_http(&mut stream);
            let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let method = request["method"].as_str().unwrap_or("").to_owned();
            seen.lock().unwrap().push(method.clone());
            let has_session = head
                .to_ascii_lowercase()
                .contains("mcp-session-id: made-up-session");
            if method != "initialize" && !has_session {
                let _ = write!(
                    stream,
                    "HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                );
                continue;
            }
            let result = match method.as_str() {
                "initialize" => {
                    json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "made-up-http", "version": "0"}})
                }
                "tools/list" => {
                    json!({"tools": [{"name": "release_date", "description": "The made-up release date.",
                    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}]})
                }
                "tools/call" => {
                    json!({"content": [{"type": "text", "text": "The release is on a made-up Tuesday."}]})
                }
                _ => {
                    // A notification: accepted without a body.
                    let _ = write!(
                        stream,
                        "HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                    continue;
                }
            };
            // The answer comes as an SSE stream, as streamable HTTP allows.
            let event = format!(
                "event: message\ndata: {}\n\n",
                json!({"jsonrpc": "2.0", "id": request["id"], "result": result})
            );
            let _ = write!(stream, "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nmcp-session-id: made-up-session\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{event}", event.len());
        }
    });
    url
}

/// A stdio MCP server: a Python script with one tool, `look_up`, that answers with the
/// root the client named (it asks `roots/list` before answering).
const STDIO_MCP: &str = r#"
import json, sys
def say(message):
    sys.stdout.write(json.dumps(message) + "\n"); sys.stdout.flush()
def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None
while True:
    message = read()
    if message is None:
        break
    method, ident = message.get("method"), message.get("id")
    if method == "initialize":
        say({"jsonrpc": "2.0", "id": ident, "result": {"protocolVersion": "2025-06-18",
             "capabilities": {"tools": {}}, "serverInfo": {"name": "made-up-notes", "version": "0"}}})
    elif method == "tools/list":
        say({"jsonrpc": "2.0", "id": ident, "result": {"tools": [{"name": "look_up",
             "description": "Look up the made-up notes.",
             "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}]}})
    elif method == "tools/call":
        say({"jsonrpc": "2.0", "id": "roots-1", "method": "roots/list", "params": {}})
        roots = read()
        root = roots["result"]["roots"][0]["uri"]
        query = message["params"]["arguments"].get("query")
        say({"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text",
             "text": "notes for " + str(query) + " in " + root}]}})
"#;

fn python() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|dir| dir.join("python3"))
        .find(|candidate| candidate.is_file())
        .expect("python3 on PATH for the stand-in stdio MCP server")
}

struct Acp {
    child: std::process::Child,
    lines: Receiver<Value>,
    next: u64,
}

impl Acp {
    fn start(port: u16) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .args(["agent", "stdio", "--model", "made-up"])
            .env("GREPPY_ENDPOINT", format!("http://127.0.0.1:{port}"))
            .env("GREPPY_API_KEY", "made-up-key")
            .env_remove("GREPPY_AGENT_RUN")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        let (tell, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in output.lines() {
                let Ok(line) = line else { break };
                if let Ok(value) = serde_json::from_str::<Value>(&line) {
                    if tell.send(value).is_err() {
                        break;
                    }
                }
            }
        });
        Self {
            child,
            lines,
            next: 1,
        }
    }

    fn send(&mut self, value: &Value) {
        let input = self.child.stdin.as_mut().unwrap();
        writeln!(input, "{value}").unwrap();
        input.flush().unwrap();
    }

    /// Ask; allow every permission meanwhile; the answer, and the permissions asked.
    fn ask(&mut self, method: &str, params: Value) -> (Value, Vec<Value>) {
        let id = self.next;
        self.next += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let (deadline, mut permissions) = (Instant::now() + Duration::from_secs(120), Vec::new());
        loop {
            let frame = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|_| panic!("greppy did not answer {method}"));
            if frame["method"] == "session/request_permission" {
                permissions.push(frame["params"]["toolCall"].clone());
                let reply = json!({"jsonrpc": "2.0", "id": frame["id"], "result":
                    {"outcome": {"outcome": "selected", "optionId": "allow-once"}}});
                self.send(&reply);
            } else if frame.get("method").is_none() && frame["id"] == json!(id) {
                assert!(frame.get("error").is_none(), "{method}: {frame}");
                return (frame["result"].clone(), permissions);
            }
        }
    }
}

impl Drop for Acp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn folder(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("greppy-acp-mcp-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root.canonicalize().unwrap()
}

fn prompt(acp: &mut Acp, session: &str, blocks: Value) -> Vec<Value> {
    let (ended, permissions) = acp.ask(
        "session/prompt",
        json!({"sessionId": session, "prompt": blocks}),
    );
    assert_eq!(ended["stopReason"], "end_turn", "{ended}");
    permissions
}

#[test]
fn the_tools_of_stdio_and_http_mcp_servers_are_offered_and_answer_the_model() {
    let root = folder("servers");
    let script = root.join("notes_server.py");
    std::fs::write(&script, STDIO_MCP).unwrap();
    let asked = Asked::default();
    let port = model(&asked);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let url = http_mcp(&seen);
    let mut acp = Acp::start(port);
    let (init, _) = acp.ask("initialize", json!({"protocolVersion": 1}));
    assert_eq!(
        init["agentCapabilities"]["mcpCapabilities"]["http"], true,
        "{init}"
    );
    assert_eq!(
        init["agentCapabilities"]["promptCapabilities"]["embeddedContext"], true,
        "{init}"
    );
    let servers = json!([
        {"name": "notes", "command": python(), "args": [script], "env": []},
        {"type": "http", "name": "dates", "url": url, "headers": [{"name": "X-Made-Up", "value": "1"}]}
    ]);
    let (began, _) = acp.ask("session/new", json!({"cwd": root, "mcpServers": servers}));
    let session = began["sessionId"].as_str().unwrap().to_owned();
    acp.ask(
        "_greppy/set_output_limit",
        json!({"sessionId": session, "modelId": "made-up",
            "endpoint": format!("http://127.0.0.1:{port}"), "maxOutputTokens": 1024}),
    );

    // Both servers' tools are offered beside greppy's own.
    let permissions = prompt(
        &mut acp,
        &session,
        json!([{"type": "text", "text": "CALL: mcp__notes__look_up"}]),
    );
    let first = asked.lock().unwrap()[0].clone();
    let offered: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert!(
        offered.contains(&"greppy")
            && offered.contains(&"mcp__notes__look_up")
            && offered.contains(&"mcp__dates__release_date"),
        "{offered:?}"
    );
    // The stdio tool ran (after asking the client for permission) and answered with the
    // session's folder as its root.
    assert_eq!(permissions.len(), 1, "{permissions:?}");
    let told = asked.lock().unwrap().last().unwrap()["messages"].to_string();
    assert!(
        told.contains(&format!("notes for made-up in file://{}", root.display())),
        "{told}"
    );

    // The http tool answers through its session.
    prompt(
        &mut acp,
        &session,
        json!([{"type": "text", "text": "CALL: mcp__dates__release_date"}]),
    );
    let told = asked.lock().unwrap().last().unwrap()["messages"].to_string();
    assert!(
        told.contains("The release is on a made-up Tuesday."),
        "{told}"
    );
    let seen = seen.lock().unwrap().clone();
    assert_eq!(
        seen.first().map(String::as_str),
        Some("initialize"),
        "{seen:?}"
    );
    assert!(seen.iter().any(|m| m == "tools/call"), "{seen:?}");

    // A linked resource reaches the model as a reference it can follow.
    prompt(
        &mut acp,
        &session,
        json!([
            {"type": "text", "text": "Look at this."},
            {"type": "resource_link", "uri": format!("file://{}/notes_server.py", root.display()), "name": "notes_server.py"}
        ]),
    );
    let told = asked.lock().unwrap().last().unwrap()["messages"].to_string();
    assert!(
        told.contains(&format!("a local file: {}/notes_server.py", root.display())),
        "{told}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_mcp_server_that_cannot_start_is_left_out_and_the_session_goes_on() {
    let root = folder("broken");
    let asked = Asked::default();
    let port = model(&asked);
    let mut acp = Acp::start(port);
    acp.ask("initialize", json!({"protocolVersion": 1}));
    let missing: &Path = Path::new("/nonexistent/made-up-mcp-server");
    let (began, _) = acp.ask(
        "session/new",
        json!({"cwd": root, "mcpServers": [{"name": "gone", "command": missing, "args": [], "env": []}]}),
    );
    assert!(began["sessionId"].is_string(), "{began}");
    let _ = std::fs::remove_dir_all(&root);
}
