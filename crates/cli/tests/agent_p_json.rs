//! Integration: `greppy -p --json` against a loopback Anthropic-Messages stub.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[path = "support/portable_provider.rs"]
mod portable_provider;
use portable_provider::spawn_fake_provider;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique_temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "greppy-agent-p-json-{tag}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn git(cwd: &std::path::Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .expect("spawn git");
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn init_repo(root: &std::path::Path) {
    git(root, &["init"]);
    git(root, &["checkout", "-b", "main"]);
    git(root, &["config", "user.name", "fixture"]);
    git(root, &["config", "user.email", "fixture@test.local"]);
    git(root, &["config", "commit.gpgsign", "false"]);
    std::fs::write(root.join("hello.txt"), b"hello\n").unwrap();
    git(root, &["add", "hello.txt"]);
    git(root, &["commit", "-m", "initial"]);
}

fn spawn_stub_gateway() -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let port = listener.local_addr().unwrap().port();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);

    let handle = thread::spawn(move || {
        let sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"test\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n",
            "\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi from stub\"}}\n",
            "\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        );

        while !stop_flag.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let mut buf = [0u8; 16384];
                    let n = stream.read(&mut buf).unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first_line = req.lines().next().unwrap_or("");
                    if first_line.starts_with("GET /v1/models") {
                        let body = r#"{"data":[{"id":"test"}]}"#;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(resp.as_bytes());
                    } else if first_line.starts_with("POST /v1/messages") {
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            sse.len(),
                            sse
                        );
                        let _ = stream.write_all(resp.as_bytes());
                    } else {
                        let body = "not found";
                        let resp = format!(
                            "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(resp.as_bytes());
                    }
                    let _ = stream.flush();
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    });

    let endpoint = format!("http://127.0.0.1:{port}");
    for _ in 0..50 {
        if std::net::TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }

    (endpoint, stop, handle)
}

fn parse_json_lines(stdout: &[u8]) -> Vec<serde_json::Value> {
    let text = std::str::from_utf8(stdout).expect("stdout must be utf-8 JSON");
    let mut values = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|error| {
            panic!("stdout line {index} is not JSON ({error}): {line:?}\nstdout={text}")
        });
        values.push(value);
    }
    assert!(
        !values.is_empty(),
        "expected at least one JSON event; stdout={text}"
    );
    values
}

fn session_jsonl(store: &std::path::Path) -> PathBuf {
    let root = store.join("agent-sessions");
    let mut found = Vec::new();
    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries {
            let path = entry.expect("read directory entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                out.push(path);
            }
        }
    }
    walk(&root, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected one session jsonl under {}, got {found:?}",
        root.display()
    );
    found.remove(0)
}

#[test]
fn greppy_p_json_streams_session_text_and_result() {
    let repo = unique_temp("repo");
    init_repo(&repo);
    let store = unique_temp("store");
    let provider_root = unique_temp("provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, handle) = spawn_stub_gateway();

    let output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_WORKSPACE_DIR", &provider.data)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "say hi",
            "--json",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--max-turns",
            "2",
            "--private-store",
            "--skip-selfcheck",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -p --json");

    stop.store(true, Ordering::SeqCst);
    let _ = handle.join();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        !stdout.contains("no changes proposed."),
        "human outcome must not leak onto --json stdout: {stdout}"
    );
    let events = parse_json_lines(&output.stdout);
    assert_eq!(
        events.first().and_then(|value| value["type"].as_str()),
        Some("session"),
        "first event must be session; stdout={stdout}"
    );
    let last = events.last().unwrap();
    assert_eq!(last["type"].as_str(), Some("result"));
    let status = last["status"].as_str().unwrap_or("");
    assert!(
        status == "clean" || status == "proposal",
        "result status must be clean or proposal, got {status}; stdout={stdout}"
    );
    assert_eq!(last["exit_code"].as_u64(), Some(0));
    assert!(
        events
            .iter()
            .any(|value| value["type"].as_str() == Some("text")),
        "expected at least one text event; stdout={stdout}"
    );

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
    drop(provider);
    let _ = std::fs::remove_dir_all(&provider_root);
}

#[test]
fn greppy_p_json_continue_reuses_session_and_appends_usage() {
    let repo = unique_temp("continue-repo");
    init_repo(&repo);
    let store = unique_temp("continue-store");
    let provider_root = unique_temp("continue-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, handle) = spawn_stub_gateway();

    let first = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_WORKSPACE_DIR", &provider.data)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "say hi",
            "--json",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--max-turns",
            "2",
            "--private-store",
            "--skip-selfcheck",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -p --json");
    let first_stdout = String::from_utf8_lossy(&first.stdout);
    let first_stderr = String::from_utf8_lossy(&first.stderr);
    assert_eq!(
        first.status.code(),
        Some(0),
        "stdout={first_stdout}\nstderr={first_stderr}"
    );
    let first_events = parse_json_lines(&first.stdout);
    let session_id = first_events[0]["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();
    assert_eq!(first_events[0]["resumed"].as_bool(), Some(false));

    let second = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_WORKSPACE_DIR", &provider.data)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "keep going",
            "--json",
            "--continue",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--max-turns",
            "2",
            "--private-store",
            "--skip-selfcheck",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -p --json --continue");
    let second_stdout = String::from_utf8_lossy(&second.stdout);
    let second_stderr = String::from_utf8_lossy(&second.stderr);
    assert_eq!(
        second.status.code(),
        Some(0),
        "stdout={second_stdout}\nstderr={second_stderr}"
    );
    let second_events = parse_json_lines(&second.stdout);
    assert_eq!(
        second_events[0]["session_id"].as_str(),
        Some(session_id.as_str())
    );
    assert_eq!(second_events[0]["resumed"].as_bool(), Some(true));

    let jsonl = std::fs::read_to_string(session_jsonl(&store)).unwrap_or_default();
    let usage_lines = jsonl
        .lines()
        .filter(|line| line.contains("\"type\":\"usage\""))
        .count();
    let turn_lines = jsonl
        .lines()
        .filter(|line| line.contains("\"type\":\"turn\""))
        .count();
    assert_eq!(
        usage_lines, 2,
        "expected two usage lines after --continue; jsonl={jsonl}"
    );
    assert!(
        turn_lines >= 2,
        "expected at least two turn lines after --continue; jsonl={jsonl}"
    );

    stop.store(true, Ordering::SeqCst);
    let _ = handle.join();
    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
    drop(provider);
    let _ = std::fs::remove_dir_all(&provider_root);
}

#[test]
fn greppy_p_json_resume_missing_exits_2_with_error_result() {
    let dir = unique_temp("resume-missing");
    let store = unique_temp("resume-missing-store");
    let output = Command::new(binary_path())
        .current_dir(&dir)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "say hi",
            "--json",
            "--resume",
            "does-not-exist",
            "--model",
            "test",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -p --json --resume");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(2),
        "stdout={stdout}\nstderr={stderr}"
    );
    let events = parse_json_lines(&output.stdout);
    let last = events.last().unwrap();
    assert_eq!(last["type"].as_str(), Some("result"));
    assert_eq!(last["status"].as_str(), Some("error"));
    assert_eq!(last["exit_code"].as_u64(), Some(2));

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&store);
}

#[test]
fn greppy_agent_json_is_usage_error() {
    let dir = unique_temp("agent-json");
    let output = Command::new(binary_path())
        .current_dir(&dir)
        .args(["agent", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy agent --json");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(2),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("error: --json is only valid with `greppy -p`"),
        "stderr={stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
