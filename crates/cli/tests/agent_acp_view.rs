//! What an ACP client is shown of greppy's tool calls, end to end with the real binary
//! and a stand-in Messages model: a running command's output while it runs, an edit
//! of a file as a diff, and the places a call reads or changes. Everything is made up.
#![cfg(unix)]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

/// The greppy call the latest user message asks for: "CALL: <arg> <arg> …" (arguments
/// separated by " | "), unless a tool answered since.
fn wanted(request: &Value) -> Option<Vec<String>> {
    let last = request["messages"].as_array()?.last()?;
    let blocks = last["content"].as_array()?;
    if blocks.iter().any(|block| block["type"] == "tool_result") {
        return None;
    }
    blocks
        .iter()
        .filter_map(|block| block["text"].as_str())
        .find_map(|text| {
            let call = text.split_once("CALL: ")?.1.lines().next()?;
            Some(call.split(" | ").map(str::to_owned).collect())
        })
}

fn answer(mut stream: TcpStream) {
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
    if !head.starts_with("POST") {
        return;
    }
    let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let mut frames = vec![
        json!({"type": "message_start", "message": {"id": "m", "type": "message",
        "role": "assistant", "model": "made-up", "content": [], "stop_reason": null,
        "usage": {"input_tokens": 10, "output_tokens": 1}}}),
    ];
    let stop = match wanted(&request) {
        Some(args) => {
            frames.push(
                json!({"type": "content_block_start", "index": 0, "content_block":
                {"type": "tool_use", "id": "toolu_1", "name": "greppy", "input": {}}}),
            );
            frames.push(json!({"type": "content_block_delta", "index": 0, "delta":
                {"type": "input_json_delta", "partial_json": json!({"args": args}).to_string()}}));
            "tool_use"
        }
        None => {
            frames.push(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}));
            frames.push(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}}));
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

fn model() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || answer(stream));
        }
    });
    port
}

struct Acp {
    child: std::process::Child,
    lines: Receiver<(Instant, Value)>,
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
                    if tell.send((Instant::now(), value)).is_err() {
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

    /// Ask; allow every permission; the answer and the tool-call updates meanwhile.
    fn ask(&mut self, method: &str, params: Value) -> (Value, Vec<(Instant, Value)>) {
        let id = self.next;
        self.next += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let (deadline, mut updates) = (Instant::now() + Duration::from_secs(120), Vec::new());
        loop {
            let (at, frame) = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|_| panic!("greppy did not answer {method}: {updates:#?}"));
            if frame["method"] == "session/request_permission" {
                let reply = json!({"jsonrpc": "2.0", "id": frame["id"], "result":
                    {"outcome": {"outcome": "selected", "optionId": "allow-always"}}});
                self.send(&reply);
            } else if frame["method"] == "session/update" {
                updates.push((at, frame["params"]["update"].clone()));
            } else if frame.get("method").is_none() && frame["id"] == json!(id) {
                assert!(frame.get("error").is_none(), "{method}: {frame}");
                return (frame["result"].clone(), updates);
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

#[test]
fn the_client_sees_live_output_diffs_and_locations() {
    let root = std::env::temp_dir().join(format!("greppy-acp-view-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    std::fs::write(root.join("tide.txt"), "low tide\n").unwrap();
    let port = model();
    let mut acp = Acp::start(port);
    acp.ask("initialize", json!({"protocolVersion": 1}));
    let (began, _) = acp.ask("session/new", json!({"cwd": root, "mcpServers": []}));
    let session = began["sessionId"].as_str().unwrap().to_owned();
    acp.ask(
        "_greppy/set_output_limit",
        json!({"sessionId": session, "modelId": "made-up",
            "endpoint": format!("http://127.0.0.1:{port}"), "maxOutputTokens": 1024}),
    );
    let mut prompt = |text: &str| {
        let (ended, updates) = acp.ask(
            "session/prompt",
            json!({"sessionId": session, "prompt": [{"type": "text", "text": text}]}),
        );
        assert_eq!(ended["stopReason"], "end_turn", "{ended}");
        updates
    };

    // A command that prints, then keeps running: what it printed is shown while it runs.
    let updates = prompt("CALL: bash-smart | -- | bash | -c | echo made-up-tide-line; sleep 1.5");
    let live: Vec<&(Instant, Value)> = updates
        .iter()
        .filter(|(_, update)| {
            update["status"] == "in_progress"
                && update["content"].to_string().contains("made-up-tide-line")
        })
        .collect();
    let completed = updates
        .iter()
        .find(|(_, update)| update["status"] == "completed")
        .unwrap_or_else(|| panic!("no completed update: {updates:#?}"));
    assert!(
        live.first()
            .is_some_and(|(at, _)| completed.0.duration_since(*at) >= Duration::from_millis(500)),
        "the output was not shown while the command ran: {updates:#?}"
    );

    // Writing a new file: a diff without old text, and the file as its location.
    let updates = prompt("CALL: write | made-up-new.txt | fresh tide\n");
    let start = &updates
        .iter()
        .find(|(_, u)| u["sessionUpdate"] == "tool_call")
        .unwrap()
        .1;
    assert_eq!(
        start["locations"][0]["path"],
        json!(root.join("made-up-new.txt")),
        "{start}"
    );
    let done = &updates
        .iter()
        .find(|(_, u)| u["status"] == "completed")
        .unwrap()
        .1;
    assert_eq!(done["content"][0]["type"], "diff", "{done}");
    assert_eq!(done["content"][0]["oldText"], Value::Null, "{done}");
    assert!(
        done["content"][0]["newText"]
            .as_str()
            .unwrap()
            .contains("fresh tide"),
        "{done}"
    );

    // Changing a file: its old and new text.
    let updates = prompt("CALL: replace-text | tide.txt | low tide | high tide");
    let done = &updates
        .iter()
        .find(|(_, u)| u["status"] == "completed")
        .unwrap()
        .1;
    assert_eq!(
        done["content"][0]["path"],
        json!(root.join("tide.txt")),
        "{done}"
    );
    assert_eq!(
        (
            &done["content"][0]["oldText"],
            &done["content"][0]["newText"]
        ),
        (&json!("low tide\n"), &json!("high tide\n")),
        "{done}"
    );

    // Reading a file names it as the place the call looks at.
    let updates = prompt("CALL: read | tide.txt | --lines | 1:1");
    let start = &updates
        .iter()
        .find(|(_, u)| u["sessionUpdate"] == "tool_call")
        .unwrap()
        .1;
    assert_eq!(
        start["locations"],
        json!([{"path": root.join("tide.txt"), "line": 1}]),
        "{start}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_client_sees_the_title_the_window_and_takes_back_an_edit_with_undo() {
    let root = std::env::temp_dir().join(format!("greppy-acp-undo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    std::fs::write(root.join("buoy.txt"), "red buoy\n").unwrap();
    let port = model();
    let mut acp = Acp::start(port);
    acp.ask("initialize", json!({"protocolVersion": 1}));
    let (began, _) = acp.ask("session/new", json!({"cwd": root, "mcpServers": []}));
    let session = began["sessionId"].as_str().unwrap().to_owned();
    acp.ask(
        "_greppy/set_output_limit",
        json!({"sessionId": session, "modelId": "made-up",
            "endpoint": format!("http://127.0.0.1:{port}"), "maxOutputTokens": 1024}),
    );
    let mut prompt = |text: &str| {
        let (ended, updates) = acp.ask(
            "session/prompt",
            json!({"sessionId": session, "prompt": [{"type": "text", "text": text}]}),
        );
        assert_eq!(ended["stopReason"], "end_turn", "{ended}");
        updates
    };
    // The first prompt names the conversation; what a request took comes with the window.
    let updates = prompt("Say something about buoys.");
    let title = updates
        .iter()
        .find(|(_, u)| u["sessionUpdate"] == "session_info_update")
        .map(|(_, u)| u.clone());
    assert_eq!(
        title.map(|u| u["title"].clone()),
        Some(json!("Say something about buoys.")),
        "{updates:#?}"
    );
    let usage = updates
        .iter()
        .find(|(_, u)| u["sessionUpdate"] == "usage_update")
        .map(|(_, u)| u.clone())
        .unwrap();
    assert_eq!(usage["size"], 256000, "{usage}");

    // An edit, then /undo: the file is as it was, and the model was not asked.
    prompt("CALL: replace-text | buoy.txt | red buoy | green buoy");
    assert_eq!(
        std::fs::read_to_string(root.join("buoy.txt")).unwrap(),
        "green buoy\n"
    );
    let updates = prompt("/undo");
    assert_eq!(
        std::fs::read_to_string(root.join("buoy.txt")).unwrap(),
        "red buoy\n",
        "{updates:#?}"
    );
    assert!(
        updates
            .iter()
            .any(|(_, u)| u["sessionUpdate"] == "agent_message_chunk"),
        "{updates:#?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
