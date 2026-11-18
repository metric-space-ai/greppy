//! `greppy agent stdio --sandbox`, end to end with the real binary and a stand-in
//! Messages model on loopback: a command the model has greppy run may write in the
//! session's folder and nowhere else, and it has no network; without `--sandbox` the
//! same command reaches the network. Everything here is made up.
#![cfg(unix)]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

/// The command the latest user message asks for (`RUN: <command>`), where no tool
/// answered since.
fn wanted(request: &Value) -> Option<String> {
    let last = request["messages"].as_array()?.last()?;
    let blocks = last["content"].as_array()?;
    if blocks.iter().any(|block| block["type"] == "tool_result") {
        return None;
    }
    blocks
        .iter()
        .filter_map(|block| block["text"].as_str())
        .find_map(|text| Some(text.split_once("RUN: ")?.1.lines().next()?.to_owned()))
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
        // Also the network probe of the sandboxed command lands here: say nothing.
        return;
    }
    let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let mut frames = vec![
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message",
        "role": "assistant", "model": "made-up", "content": [], "stop_reason": null,
        "usage": {"input_tokens": 10, "output_tokens": 1}}}),
    ];
    let stop = match wanted(&request) {
        Some(command) => {
            let input = json!({"args": ["bash-smart", "--", "bash", "-c", command]}).to_string();
            frames.push(
                json!({"type": "content_block_start", "index": 0, "content_block":
                {"type": "tool_use", "id": "toolu_1", "name": "greppy", "input": {}}}),
            );
            frames.push(json!({"type": "content_block_delta", "index": 0, "delta":
                {"type": "input_json_delta", "partial_json": input}}));
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

/// A stand-in model; its port.
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

/// greppy over ACP, line by line.
struct Acp {
    child: std::process::Child,
    lines: Receiver<Value>,
    next: u64,
}

impl Acp {
    fn start(port: u16, sandbox: bool) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_greppy"));
        command.args(["agent", "stdio", "--model", "made-up"]);
        if sandbox {
            command.arg("--sandbox");
        }
        let mut child = command
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

    /// Ask, allow every permission meanwhile, and keep the tool results said.
    fn ask(&mut self, method: &str, params: Value, outputs: &mut Vec<String>) -> Value {
        let id = self.next;
        self.next += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let deadline = Instant::now() + Duration::from_secs(120);
        let began = Instant::now();
        let mut heard: Vec<String> = Vec::new();
        loop {
            let frame = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|_| {
                    panic!(
                        "greppy did not answer {method} within {:?}; it said meanwhile: {heard:#?}",
                        began.elapsed()
                    )
                });
            heard.push(format!(
                "{:?} {}",
                began.elapsed(),
                frame.to_string().chars().take(300).collect::<String>()
            ));
            if frame["method"] == "session/request_permission" {
                let reply = json!({"jsonrpc": "2.0", "id": frame["id"], "result":
                    {"outcome": {"outcome": "selected", "optionId": "allow-always"}}});
                self.send(&reply);
            } else if frame["params"]["update"]["sessionUpdate"] == "tool_call_update" {
                if let Some(output) = frame["params"]["update"]["rawOutput"].as_str() {
                    outputs.push(output.to_owned());
                }
            } else if frame.get("method").is_none() && frame["id"] == json!(id) {
                assert!(frame.get("error").is_none(), "{method}: {frame}");
                return frame["result"].clone();
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

/// Run one command through greppy in a session in `folder`; what the tool said.
fn run(port: u16, sandbox: bool, folder: &Path, command: &str) -> String {
    let mut acp = Acp::start(port, sandbox);
    let mut outputs = Vec::new();
    acp.ask("initialize", json!({"protocolVersion": 1}), &mut outputs);
    let session = acp.ask(
        "session/new",
        json!({"cwd": folder, "mcpServers": []}),
        &mut outputs,
    )["sessionId"]
        .as_str()
        .unwrap()
        .to_owned();
    acp.ask(
        "_greppy/set_output_limit",
        json!({"sessionId": session, "modelId": "made-up",
            "endpoint": format!("http://127.0.0.1:{port}"), "maxOutputTokens": 1024}),
        &mut outputs,
    );
    let ended = acp.ask(
        "session/prompt",
        json!({"sessionId": session, "prompt": [{"type": "text", "text": format!("RUN: {command}")}]}),
        &mut outputs,
    );
    assert_eq!(ended["stopReason"], "end_turn", "{ended}");
    assert_eq!(outputs.len(), 1, "{outputs:?}");
    outputs.remove(0)
}

fn root(tag: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("greppy-acp-sandbox-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("project")).unwrap();
    root.canonicalize().unwrap()
}

#[test]
fn a_sandboxed_command_writes_only_in_its_folder_and_has_no_network() {
    let root = root("confined");
    let folder = root.join("project");
    let port = model();

    let said = run(
        port,
        true,
        &folder,
        "printf made > inside.txt && cat inside.txt",
    );
    if said.contains("sandbox cannot be made") {
        // A host that cannot confine refuses the command; it never runs unconfined.
        assert!(!folder.join("inside.txt").exists(), "{said}");
        println!("THIS HOST CANNOT MAKE THE SANDBOX; confinement is NOT shown by this run: {said}");
        return;
    }
    assert_eq!(
        std::fs::read_to_string(folder.join("inside.txt"))
            .ok()
            .as_deref(),
        Some("made"),
        "{said}"
    );

    let outside = root.join("outside.txt");
    let said = run(
        port,
        true,
        &folder,
        &format!("printf no > {}", outside.display()),
    );
    assert!(
        !outside.exists(),
        "the command wrote outside its folder: {said}"
    );

    let probe = format!("exec 3<>/dev/tcp/127.0.0.1/{port} && echo connected");
    let said = run(port, true, &folder, &probe);
    assert!(
        !said.contains("connected"),
        "a sandboxed command reached the network: {said}"
    );

    // The same probe, unconfined, does reach it: the probe can tell.
    let said = run(port, false, &folder, &probe);
    assert!(
        said.contains("connected"),
        "the network probe never connects: {said}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
