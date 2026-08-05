//! WP20: startup self-check for `greppy -p`.
//!
//! A -p run must never silently degrade into a shell-only agent. Before the
//! first model turn, greppy verifies index-backed navigation and a worktree
//! write through the production sandboxed tool path. Failures abort (exit 3).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use greppy_agent::{run_startup_self_check, ExecutionEnv, GreppyEnv};
use serde_json::json;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique(tag: &str) -> PathBuf {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!(
        "greppy-wp20-{tag}-{}-{}-{}",
        std::process::id(),
        seq,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn git(cwd: &Path, args: &[&str]) {
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

fn init_repo(root: &Path) {
    git(root, &["init"]);
    git(root, &["checkout", "-b", "main"]);
    git(root, &["config", "user.name", "fixture"]);
    git(root, &["config", "user.email", "fixture@test.local"]);
    git(root, &["config", "commit.gpgsign", "false"]);
    // A tiny Rust source so `where-am-i` reports files > 0 after index.
    std::fs::write(root.join("main.rs"), b"fn main() { println!(\"hi\"); }\n").unwrap();
    git(root, &["add", "main.rs"]);
    git(root, &["commit", "-m", "initial"]);
}

/// Minimal Anthropic Messages gateway: GET /v1/models → 200; POST /v1/messages
/// → canned SSE text-only end_turn stream.
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

#[cfg(unix)]
fn write_stub(body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = unique("stub").join("greppy-stub");
    let script = format!("#!/bin/sh\n{body}\n");
    std::fs::write(&path, script).expect("write stub");
    let mut perms = std::fs::metadata(&path).expect("meta").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).expect("chmod");
    path
}

/// Healthy fixture: real greppy binary + real index → self-check ok line, model runs.
#[test]
fn selfcheck_passes_on_healthy_fixture() {
    let repo = unique("healthy-repo");
    init_repo(&repo);
    let store = unique("healthy-store");

    // Pre-build the index in the repo so the agent worktree clone has something
    // to warm from, and so prewarm is quick.
    let index = Command::new(binary_path())
        .args(["index"])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .output()
        .expect("index");
    // Index may warn about embeddings; structural must succeed for where-am-i.
    assert!(
        index.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&index.stderr)
    );

    let (endpoint, stop, handle) = spawn_stub_gateway();

    let output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .env_remove("GREPPY_SKIP_SELFCHECK")
        .args([
            "-p",
            "say hi",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--max-turns",
            "2",
            "--no-sandbox",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -p");

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
        stderr.contains("self-check ok — index answers, worktree writable")
            || stderr.contains(
                "self-check ok — index answers (census shape unrecognized), worktree writable"
            ),
        "expected self-check success line; stderr={stderr}"
    );
    assert!(
        stdout.contains("hi from stub"),
        "model loop must still run after self-check; stdout={stdout}\nstderr={stderr}"
    );

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
}

/// `--skip-selfcheck` / GREPPY_SKIP_SELFCHECK=1 bypasses the probes entirely.
#[test]
fn skip_selfcheck_bypasses_probes() {
    let repo = unique("skip-repo");
    init_repo(&repo);
    let store = unique("skip-store");
    let (endpoint, stop, handle) = spawn_stub_gateway();

    let output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "say hi",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--max-turns",
            "2",
            "--skip-selfcheck",
            "--no-sandbox",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -p");

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
        !stderr.contains("self-check ok"),
        "skip must not print success line; stderr={stderr}"
    );
    assert!(
        !stderr.contains("self-check failed"),
        "skip must not run probes; stderr={stderr}"
    );
    assert!(
        stdout.contains("hi from stub"),
        "model loop must run; stdout={stdout}\nstderr={stderr}"
    );

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
}

/// Self-check failure aborts before the model loop (stub greppy that errors).
#[cfg(unix)]
#[test]
fn selfcheck_failure_aborts_before_loop() {
    // Direct GreppyEnv path with a stub that always errors — same production
    // call_tool path the CLI uses. Proves the failure surface that CLI aborts on.
    let root = unique("fail-root");
    let stub = write_stub(
        r#"
printf 'Operation not permitted: lifecycle lease\n' >&2
exit 1
"#,
    );
    let mut env = GreppyEnv::with_binary(stub, root.clone()).expect("env");
    let err = run_startup_self_check(&mut env).expect_err("must fail");
    assert_eq!(err.probe, "where-am-i");
    let diag = err.diagnostic();
    assert!(diag.contains("self-check failed"), "diag={diag}");
    assert!(
        diag.contains("Operation not permitted") || diag.contains("--no-sandbox"),
        "diag={diag}"
    );

    // Also prove: after a failure outcome, no model traffic is required — the
    // CLI path returns EXIT_AGENT before run_agent_loop. Covered structurally
    // by agent.rs match on Err; here we assert the env surface is enough to
    // decide abort without any further tool call succeeding.
    let _ = std::fs::remove_dir_all(&root);
}

/// Empty-index census (`0 files`) is a failure even when the tool exits 0.
#[cfg(unix)]
#[test]
fn selfcheck_empty_index_is_failure() {
    let root = unique("empty-root");
    let stub = write_stub(
        r#"
if [ "$1" = "where-am-i" ]; then
  printf '/tmp/fixture — 0 files, 0 definitions\n'
  exit 0
fi
printf 'ok — exit 0\n'
exit 0
"#,
    );
    let mut env = GreppyEnv::with_binary(stub, root.clone()).expect("env");
    let err = run_startup_self_check(&mut env).expect_err("empty index must fail");
    assert_eq!(err.probe, "where-am-i");
    assert!(err.output.contains("0 files"), "output={}", err.output);
    let _ = std::fs::remove_dir_all(&root);
}

/// Unrecognized where-am-i shape is a pass (never fail on formatting drift).
#[cfg(unix)]
#[test]
fn selfcheck_unrecognized_shape_passes() {
    let root = unique("shape-root");
    let stub = write_stub(
        r#"
if [ "$1" = "where-am-i" ]; then
  printf 'orientation complete (shape drifted)\n'
  exit 0
fi
if [ "$1" = "bash-smart" ]; then
  printf 'ok — exit 0\n'
  exit 0
fi
exit 0
"#,
    );
    let mut env = GreppyEnv::with_binary(stub, root.clone()).expect("env");
    let ok = run_startup_self_check(&mut env).expect("unrecognized shape must pass");
    assert!(ok.unrecognized_census_shape);
    let _ = std::fs::remove_dir_all(&root);
}

/// Direct healthy path through GreppyEnv with a cooperating stub.
#[cfg(unix)]
#[test]
fn selfcheck_healthy_stub_passes() {
    let root = unique("ok-root");
    let stub = write_stub(
        r#"
if [ "$1" = "where-am-i" ]; then
  printf '/tmp/fixture — rust, 3 files, 7 definitions\n'
  exit 0
fi
if [ "$1" = "bash-smart" ]; then
  printf 'ok — exit 0\n'
  exit 0
fi
printf 'unexpected: %s\n' "$*" >&2
exit 2
"#,
    );
    let mut env = GreppyEnv::with_binary(stub, root.clone()).expect("env");
    let ok = run_startup_self_check(&mut env).expect("must pass");
    assert!(!ok.unrecognized_census_shape);
    // Sanity: env still works after self-check (loop would call tools next).
    let again = env.call_tool("greppy", &json!({"args": ["where-am-i"]}));
    assert!(!again.is_error, "content={}", again.content);
    let _ = std::fs::remove_dir_all(&root);
}
