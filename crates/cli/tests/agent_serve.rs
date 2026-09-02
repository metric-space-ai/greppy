//! Integration proofs for `greppy agent serve` and its local control socket.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use greppy::agent_control::ControlClient;
use serde_json::{json, Value};

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
        "gs-{}-{}",
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

fn spawn_gateway(
    delay: Duration,
    fail_messages: bool,
) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
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
                        // Accepted sockets inherit O_NONBLOCK from the listener on
                        // macOS; a blocking read is required for the request loop.
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
                        // Drain the request body: closing with unread bytes resets the
                        // connection and the client loses the response.
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
                            if fail_messages {
                                let body = r#"{"error":{"message":"serve stub failure"}}"#;
                                let response = format!(
                                    "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                    body.len()
                                );
                                stream.write_all(response.as_bytes()).ok();
                            } else {
                                let response = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                                    sse.len()
                                );
                                stream.write_all(response.as_bytes()).ok();
                            }
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

fn wait_idle(client: &mut ControlClient) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let description = client.call("session/describe", json!({})).unwrap();
        if description["phase"] == "idle" {
            return description;
        }
        assert!(
            Instant::now() < deadline,
            "serve did not become idle: {description}"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn next_type(client: &mut ControlClient, wanted: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(event) = client.next_event(Duration::from_millis(500)).unwrap() {
            if event["type"] == wanted {
                return event;
            }
        }
        assert!(Instant::now() < deadline, "did not receive {wanted}");
    }
}

fn stop_gateway(stop: Arc<AtomicBool>, handle: thread::JoinHandle<()>) {
    stop.store(true, Ordering::SeqCst);
    handle.join().unwrap();
}

#[test]
fn control_socket_hosts_queued_turns_and_quits_cleanly() {
    let repo = unique_temp("control-repo");
    init_repo(&repo);
    let store = unique_temp("control-store");
    let provider_root = unique_temp("control-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, gateway) = spawn_gateway(Duration::from_millis(250), false);
    let mut hosted = spawn_serve(&repo, &store, &provider.data, &endpoint, &[]);

    assert_eq!(hosted.session["type"], "session");
    assert_eq!(
        hosted.session["mode"], "serve",
        "unexpected first session event: {}",
        hosted.session
    );
    let socket = PathBuf::from(hosted.session["socket"].as_str().unwrap());
    assert!(socket.exists());
    assert_eq!(
        std::fs::metadata(&socket).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let listed = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .args(["agent", "sessions", "list", "--json"])
        .output()
        .unwrap();
    assert!(listed.status.success());
    let rows: Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(
        rows[0]["live"],
        true,
        "rows={rows}; socket_exists={}; direct_live={}",
        socket.exists(),
        greppy::agent_control::is_live(&socket)
    );
    assert_eq!(rows[0]["socket"], socket.display().to_string());

    let mut client = ControlClient::connect(&socket).unwrap();
    let description = wait_idle(&mut client);
    assert_eq!(description["session_id"], hosted.session["session_id"]);
    assert_eq!(description["phase"], "idle");
    client.subscribe().unwrap();
    assert_eq!(
        client.call("turn/interrupt", json!({})).unwrap(),
        json!({"accepted":true})
    );

    let accepted = client.call("turn/start", json!({"text":"hello"})).unwrap();
    assert_eq!(accepted["accepted"], true);
    let start = next_type(&mut client, "turn_start");
    assert_eq!(start["source"], "remote");
    let queued_one = client.call("turn/start", json!({"text":"second"})).unwrap();
    let queued_two = client.call("turn/start", json!({"text":"third"})).unwrap();
    assert_eq!(queued_one["position"], 1);
    assert_eq!(queued_two["position"], 2);

    let mut text_events = 0;
    let mut completions = 0;
    let deadline = Instant::now() + Duration::from_secs(15);
    while completions < 3 {
        if let Some(event) = client.next_event(Duration::from_millis(500)).unwrap() {
            if event["type"] == "text" {
                text_events += 1;
            } else if event["type"] == "turn_complete" {
                completions += 1;
            } else if event["type"] == "error" {
                panic!("serve emitted error after {completions} turns: {event}");
            }
        }
        assert!(
            Instant::now() < deadline,
            "only {completions} turns completed"
        );
    }
    assert!(text_events >= 1);

    let jsonl = PathBuf::from(description["jsonl"].as_str().unwrap());
    let log = std::fs::read_to_string(jsonl).unwrap();
    assert!(log.contains(r#""event":"start""#));
    assert!(log.contains(r#""source":"remote""#));

    let mut raw = UnixStream::connect(&socket).unwrap();
    raw.write_all(b"not-json\n").unwrap();
    let mut raw_reader = BufReader::new(raw.try_clone().unwrap());
    raw_reader
        .get_ref()
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut line = String::new();
    raw_reader.read_line(&mut line).unwrap();
    let parse_error: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(parse_error["error"]["code"], -32700);
    raw.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"bogus\",\"params\":{}}\n")
        .unwrap();
    line.clear();
    raw_reader.read_line(&mut line).unwrap();
    let unknown: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(unknown["error"]["code"], -32601);

    assert_eq!(
        client.call("session/quit", json!({})).unwrap(),
        json!({"accepted":true})
    );
    let status = hosted.child.wait().unwrap();
    let mut rest = String::new();
    hosted.stdout.read_to_string(&mut rest).unwrap();
    assert_eq!(status.code(), Some(0), "stdout remainder={rest}");
    assert!(rest.lines().any(|line| {
        serde_json::from_str::<Value>(line).is_ok_and(|event| event["type"] == "result")
    }));
    assert!(!socket.exists());

    stop_gateway(stop, gateway);
    drop(provider);
}

#[test]
fn worker_error_exits_with_error_result_and_unlinks_socket() {
    let repo = unique_temp("error-repo");
    init_repo(&repo);
    let store = unique_temp("error-store");
    let provider_root = unique_temp("error-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, gateway) = spawn_gateway(Duration::ZERO, true);
    let mut hosted = spawn_serve(&repo, &store, &provider.data, &endpoint, &[]);
    let socket = PathBuf::from(hosted.session["socket"].as_str().unwrap());
    let mut client = ControlClient::connect(&socket).unwrap();
    wait_idle(&mut client);
    assert_eq!(
        client.call("turn/start", json!({"text":"fail"})).unwrap()["accepted"],
        true
    );

    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = hosted.child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = hosted.child.kill();
            let _ = hosted.child.wait();
            panic!("serve did not exit after worker error");
        }
        thread::sleep(Duration::from_millis(50));
    };
    let mut rest = String::new();
    hosted.stdout.read_to_string(&mut rest).unwrap();
    assert!(!status.success(), "stdout remainder={rest}");
    let result: Value = serde_json::from_str(rest.lines().last().expect("final result line"))
        .unwrap_or_else(|error| panic!("final stdout line is not JSON: {error}: {rest:?}"));
    assert_eq!(result["type"], "result");
    assert_eq!(result["status"], "error");
    assert!(!socket.exists());

    stop_gateway(stop, gateway);
    drop(provider);
}

#[test]
fn sigterm_during_idle_emits_result_and_unlinks_socket() {
    let repo = unique_temp("signal-repo");
    init_repo(&repo);
    let store = unique_temp("signal-store");
    let provider_root = unique_temp("signal-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, gateway) = spawn_gateway(Duration::ZERO, false);
    let mut hosted = spawn_serve(&repo, &store, &provider.data, &endpoint, &[]);
    let socket = PathBuf::from(hosted.session["socket"].as_str().unwrap());
    let mut client = ControlClient::connect(&socket).unwrap();
    wait_idle(&mut client);
    unsafe { libc::kill(hosted.child.id() as i32, libc::SIGTERM) };
    let status = hosted.child.wait().unwrap();
    let mut rest = String::new();
    hosted.stdout.read_to_string(&mut rest).unwrap();
    assert_eq!(status.code(), Some(0), "stdout remainder={rest}");
    assert!(rest.contains(r#""type":"result""#));
    assert!(!socket.exists());
    stop_gateway(stop, gateway);
    drop(provider);
}

#[test]
fn idle_timeout_exits_without_a_client() {
    let repo = unique_temp("timeout-repo");
    init_repo(&repo);
    let store = unique_temp("timeout-store");
    let provider_root = unique_temp("timeout-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, gateway) = spawn_gateway(Duration::ZERO, false);
    let mut hosted = spawn_serve(
        &repo,
        &store,
        &provider.data,
        &endpoint,
        &["--idle-timeout-secs", "1"],
    );
    let socket = PathBuf::from(
        hosted.session["socket"]
            .as_str()
            .unwrap_or_else(|| panic!("unexpected session event: {}", hosted.session)),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = hosted.child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "idle timeout did not exit");
        thread::sleep(Duration::from_millis(50));
    };
    let mut rest = String::new();
    hosted.stdout.read_to_string(&mut rest).unwrap();
    assert_eq!(status.code(), Some(0), "stdout remainder={rest}");
    assert!(rest.contains(r#""type":"result""#));
    assert!(!socket.exists());
    stop_gateway(stop, gateway);
    drop(provider);
}
