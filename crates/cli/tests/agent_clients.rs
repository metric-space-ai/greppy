//! Integration proofs for `greppy agent status|send|interrupt|quit`.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

#[path = "support/portable_provider.rs"]
mod portable_provider;
use portable_provider::spawn_fake_provider;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique_temp(tag: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let temp_root = std::env::temp_dir();
    let short_root = temp_root.parent().unwrap_or(&temp_root);
    let parent = short_root.join(format!(
        "gc-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let leaf = if tag.contains("repo") {
        "r"
    } else if tag.contains("store") {
        "s"
    } else {
        "p"
    };
    let path = parent.join(leaf);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn git(cwd: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_repo(root: &Path) {
    git(root, &["init"]);
    git(root, &["checkout", "-b", "main"]);
    git(root, &["config", "user.name", "fixture"]);
    git(root, &["config", "user.email", "fixture@test.local"]);
    git(root, &["config", "commit.gpgsign", "false"]);
    std::fs::write(root.join("hello.txt"), b"hello\n").unwrap();
    git(root, &["add", "hello.txt"]);
    git(root, &["commit", "-m", "initial"]);
}

fn spawn_gateway(delay: Duration) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    let handle = thread::spawn(move || {
        let sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"test\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi from serve stub\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        while !stop_thread.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    thread::spawn(move || {
                        stream.set_nonblocking(false).ok();
                        stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
                        let mut request = Vec::new();
                        let mut chunk = [0; 4096];
                        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                            let count = stream.read(&mut chunk).unwrap_or(0);
                            if count == 0 {
                                return;
                            }
                            request.extend_from_slice(&chunk[..count]);
                        }
                        let header_end = request
                            .windows(4)
                            .position(|window| window == b"\r\n\r\n")
                            .map(|index| index + 4)
                            .unwrap_or(request.len());
                        let head = String::from_utf8_lossy(&request[..header_end]).to_string();
                        let content_length = head
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.trim()
                                    .eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())?
                            })
                            .unwrap_or(0);
                        while request.len() < header_end + content_length {
                            let count = stream.read(&mut chunk).unwrap_or(0);
                            if count == 0 {
                                break;
                            }
                            request.extend_from_slice(&chunk[..count]);
                        }
                        let first = head.lines().next().unwrap_or("");
                        if first.starts_with("GET /v1/models") {
                            let body = r#"{"data":[{"id":"test"}]}"#;
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                body.len()
                            );
                            stream.write_all(response.as_bytes()).ok();
                        } else if first.starts_with("POST /v1/messages") {
                            thread::sleep(delay);
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                                sse.len()
                            );
                            stream.write_all(response.as_bytes()).ok();
                        }
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });
    let mut ready = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    ready
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    ready
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut response = String::new();
    ready.read_to_string(&mut response).unwrap();
    assert!(response.contains("200 OK"));
    (format!("http://127.0.0.1:{port}"), stop, handle)
}

struct Hosted {
    child: Child,
    stdout: BufReader<ChildStdout>,
    session: Value,
}

fn spawn_serve(
    repo: &Path,
    store: &Path,
    provider: &Path,
    endpoint: &str,
    extra: &[&str],
) -> Hosted {
    let mut args = vec![
        "agent",
        "serve",
        "--model",
        "test",
        "--endpoint",
        endpoint,
        "--max-turns",
        "2",
        "--private-store",
        "--skip-selfcheck",
    ];
    args.extend_from_slice(extra);
    let mut child = Command::new(binary_path())
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_WORKSPACE_DIR", provider)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut first = String::new();
    stdout.read_line(&mut first).unwrap();
    assert!(!first.is_empty(), "serve exited before session event");
    let session: Value = serde_json::from_str(&first)
        .unwrap_or_else(|error| panic!("first stdout line is not JSON: {error}: {first:?}"));
    if session["mode"] != "serve" {
        let status = child.wait().unwrap();
        let mut stderr = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        let mut rest = String::new();
        stdout.read_to_string(&mut rest).unwrap();
        panic!("serve startup failed: status={status}; session={session}; stdout={rest}; stderr={stderr}");
    }
    Hosted {
        child,
        stdout,
        session,
    }
}

fn stop_gateway(stop: Arc<AtomicBool>, handle: thread::JoinHandle<()>) {
    stop.store(true, Ordering::SeqCst);
    handle.join().unwrap();
}

fn greppy(repo: &Path, store: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary_path())
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .env_remove("GREPPY_PROJECT_IDENTITY")
        .args(args)
        .output()
        .unwrap()
}

fn utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn wait_status_idle(repo: &Path, store: &Path, id: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let output = greppy(repo, store, &["agent", "status", id, "--json"]);
        let stdout = utf8(&output.stdout);
        let stderr = utf8(&output.stderr);
        if output.status.success() {
            let description: Value = serde_json::from_str(&stdout)
                .unwrap_or_else(|error| panic!("status json: {error}: {stdout} stderr={stderr}"));
            if description["phase"] == "idle" {
                return description;
            }
        }
        assert!(
            Instant::now() < deadline,
            "status did not become idle: code={:?} stdout={stdout} stderr={stderr}",
            output.status.code()
        );
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn unknown_id_exits_2() {
    let repo = unique_temp("unknown-repo");
    init_repo(&repo);
    let store = unique_temp("unknown-store");
    let output = greppy(&repo, &store, &["agent", "status", "nosuch", "--json"]);
    assert_eq!(
        output.status.code(),
        Some(2),
        "stdout={} stderr={}",
        utf8(&output.stdout),
        utf8(&output.stderr)
    );
}

#[test]
fn status_send_interrupt_quit_drive_a_live_session() {
    let repo = unique_temp("clients-repo");
    init_repo(&repo);
    let store = unique_temp("clients-store");
    let provider_root = unique_temp("clients-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, gateway) = spawn_gateway(Duration::from_millis(50));
    let mut hosted = spawn_serve(&repo, &store, &provider.data, &endpoint, &[]);
    let id = hosted.session["session_id"].as_str().unwrap().to_string();

    let description = wait_status_idle(&repo, &store, &id);
    assert_eq!(description["phase"], "idle");
    assert_eq!(description["session_id"], id);

    let interrupted = greppy(&repo, &store, &["agent", "interrupt", &id, "--json"]);
    assert_eq!(
        interrupted.status.code(),
        Some(0),
        "stdout={} stderr={}",
        utf8(&interrupted.stdout),
        utf8(&interrupted.stderr)
    );
    let interrupt_json: Value = serde_json::from_slice(&interrupted.stdout).unwrap();
    assert_eq!(interrupt_json["accepted"], true);

    let sent = greppy(
        &repo,
        &store,
        &["agent", "send", &id, "hello", "--wait", "--json"],
    );
    let stdout = utf8(&sent.stdout);
    let stderr = utf8(&sent.stderr);
    assert_eq!(
        sent.status.code(),
        Some(0),
        "stdout={stdout} stderr={stderr}"
    );
    let events: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let start = events
        .iter()
        .find(|event| event["type"] == "turn_start")
        .unwrap_or_else(|| panic!("missing turn_start in {events:?}"));
    let prompt_id = start["prompt_id"].as_str().unwrap();
    assert!(
        prompt_id.starts_with("p-"),
        "prompt_id={prompt_id} events={events:?}"
    );
    assert!(
        events.iter().any(|event| event["type"] == "text"),
        "missing text in {events:?}"
    );
    assert_eq!(
        events.last().map(|event| &event["type"]),
        Some(&Value::from("turn_complete"))
    );

    let queued = greppy(&repo, &store, &["agent", "send", &id, "second"]);
    let queued_out = utf8(&queued.stdout);
    let queued_err = utf8(&queued.stderr);
    assert_eq!(
        queued.status.code(),
        Some(0),
        "stdout={queued_out} stderr={queued_err}"
    );
    assert!(
        queued_out.contains("queued p-") && queued_out.contains("position "),
        "queued output={queued_out}"
    );
    wait_status_idle(&repo, &store, &id);

    let quit = greppy(&repo, &store, &["agent", "quit", &id]);
    assert_eq!(
        quit.status.code(),
        Some(0),
        "stdout={} stderr={}",
        utf8(&quit.stdout),
        utf8(&quit.stderr)
    );
    assert_eq!(utf8(&quit.stdout).trim(), "quit requested");
    let status = hosted.child.wait().unwrap();
    let mut rest = String::new();
    hosted.stdout.read_to_string(&mut rest).unwrap();
    assert_eq!(status.code(), Some(0), "stdout remainder={rest}");

    let after = greppy(&repo, &store, &["agent", "status", &id, "--json"]);
    let after_out = utf8(&after.stdout);
    let after_err = utf8(&after.stderr);
    assert_eq!(
        after.status.code(),
        Some(3),
        "stdout={after_out} stderr={after_err}"
    );
    assert!(
        after_err.contains("is not live (no control socket)")
            || after_out.contains("is not live (no control socket)"),
        "stdout={after_out} stderr={after_err}"
    );
    assert!(
        after_err.contains(&format!("greppy agent serve --resume {id}"))
            || after_out.contains(&format!("greppy agent serve --resume {id}")),
        "stdout={after_out} stderr={after_err}"
    );

    stop_gateway(stop, gateway);
    drop(provider);
}
