//! Integration: `greppy -p` against a loopback Anthropic-Messages stub.
//!
//! No external network — a local `TcpListener` answers `/v1/models` and a
//! canned `/v1/messages` SSE stream (text-only, end_turn).

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
use portable_provider::{spawn_fake_provider, spawn_fake_provider_with_edits};

#[cfg(unix)]
#[path = "../../web-runtime/runtime/tests/support/recorded_http_fixture.rs"]
mod recorded_http_fixture;

fn binary_path() -> PathBuf {
    std::env::var_os("GREPPY_ACCEPTANCE_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_greppy")))
}

fn unique_temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "greppy-agent-p-{tag}-{}-{}",
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

/// Minimal Anthropic Messages gateway: GET /v1/models → 200; POST /v1/messages
/// → canned SSE text-only end_turn stream.
fn spawn_gateway(sse: &'static str) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    spawn_gateway_sequence(vec![sse])
}

fn spawn_gateway_sequence(
    responses: Vec<&'static str>,
) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    spawn_gateway_contract(
        responses,
        r#"{"data":[{"id":"test","max_completion_tokens":1024}]}"#,
        None,
    )
}

type GatewayMessages = Arc<std::sync::Mutex<Vec<serde_json::Value>>>;

fn spawn_gateway_contract<T: AsRef<str> + Send + 'static>(
    responses: Vec<T>,
    catalog: &'static str,
    caller_bound: Option<(&'static str, u64)>,
) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    spawn_gateway_observed(responses, catalog, caller_bound, None)
}

fn spawn_gateway_contract_with_error(
    responses: Vec<&'static str>,
    catalog: &'static str,
    caller_bound: Option<(&'static str, u64)>,
    error_after_edit: Option<u16>,
) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    spawn_gateway_observed_with_error(responses, catalog, caller_bound, None, error_after_edit)
}

fn spawn_gateway_observed<T: AsRef<str> + Send + 'static>(
    responses: Vec<T>,
    catalog: &'static str,
    caller_bound: Option<(&'static str, u64)>,
    messages: Option<GatewayMessages>,
) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    spawn_gateway_observed_with_error(responses, catalog, caller_bound, messages, None)
}

fn spawn_gateway_observed_with_error<T: AsRef<str> + Send + 'static>(
    responses: Vec<T>,
    catalog: &'static str,
    caller_bound: Option<(&'static str, u64)>,
    messages: Option<GatewayMessages>,
    error_after_edit: Option<u16>,
) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    assert!(!responses.is_empty());
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let port = listener.local_addr().unwrap().port();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);

    let handle = thread::spawn(move || {
        let mut message_index: usize = 0;
        while !stop_flag.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let Ok(request) = read_gateway_request(&mut stream) else {
                        continue;
                    };
                    let req = String::from_utf8_lossy(&request);
                    let first_line = req.lines().next().unwrap_or("");
                    if first_line.starts_with("GET /v1/models") {
                        let body = catalog;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(resp.as_bytes());
                    } else if first_line.starts_with("POST /v1/messages") {
                        let body_start = request
                            .windows(4)
                            .position(|part| part == b"\r\n\r\n")
                            .expect("complete HTTP headers")
                            + 4;
                        let body: serde_json::Value =
                            serde_json::from_slice(&request[body_start..]).expect("Messages JSON");
                        if let Some(messages) = messages.as_ref() {
                            let mut messages = messages.lock().unwrap();
                            assert!(
                                messages.len() < 4,
                                "scripted web gateway request bound exceeded"
                            );
                            messages.push(body.clone());
                        }
                        if let Some((model, limit)) = caller_bound {
                            assert_eq!(body["model"], model, "wire model must remain unchanged");
                            assert_eq!(body["max_tokens"].as_u64(), Some(limit));
                        } else {
                            assert!(
                                body["max_tokens"].as_u64().is_some_and(|n| (1..=1024).contains(&n)),
                                "Messages requires a positive numeric bound within the advertised fixture cap"
                            );
                        }
                        if let Some(status) = error_after_edit.filter(|_| message_index > 0) {
                            let edited = body["messages"].as_array().is_some_and(|messages| {
                                messages.iter().any(|message| {
                                    message["content"].as_array().is_some_and(|parts| {
                                        parts.iter().any(|part| {
                                            part["type"] == "tool_result"
                                                && part["is_error"] != true
                                        })
                                    })
                                })
                            });
                            assert!(
                                edited,
                                "fixture must observe a successful real edit before HTTP error"
                            );
                            // Valid close-delimited HTTP isolates lifecycle handling from
                            // the separately tracked buffered-response transport failure.
                            let resp = format!(
                                "HTTP/1.1 {status} Fixture failure\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{{\"error\":\"controlled lifecycle error\"}}"
                            );
                            let _ = stream.write_all(resp.as_bytes());
                            let _ = stream.flush();
                            continue;
                        }
                        let sse = responses[message_index.min(responses.len() - 1)].as_ref();
                        message_index += 1;
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

    // Wait until the listener accepts a connection.
    let endpoint = format!("http://127.0.0.1:{port}");
    for _ in 0..50 {
        if std::net::TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }

    (endpoint, stop, handle)
}

fn read_gateway_request(stream: &mut impl Read) -> std::io::Result<Vec<u8>> {
    // Drain the complete HTTP request before replying. A partial read can leave
    // unread request bytes and reset the connection, consuming a scripted turn
    // that the agent never received.
    let mut request = Vec::new();
    let mut buf = [0u8; 4096];
    let mut expected = None;
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        request.extend_from_slice(&buf[..n]);
        if request.len() > 1024 * 1024 {
            return Err(std::io::Error::from(std::io::ErrorKind::InvalidData));
        }
        if expected.is_none() {
            if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]);
                let length = headers
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .map(|(_, value)| value.trim().parse::<usize>())
                    .transpose()
                    .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidData))?
                    .unwrap_or(0);
                expected = Some(end + 4 + length);
            }
        }
        if expected.is_some_and(|length| request.len() >= length) {
            return Ok(request);
        }
    }
}

#[test]
fn gateway_reads_fragmented_request_before_consuming_scripted_response() {
    struct Fragments(Vec<std::io::Cursor<Vec<u8>>>);
    impl Read for Fragments {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            loop {
                let Some(first) = self.0.first_mut() else {
                    return Ok(0);
                };
                let count = first.read(bytes)?;
                if count != 0 {
                    return Ok(count);
                }
                self.0.remove(0);
            }
        }
    }
    let body = vec![b'x'; 32768];
    let headers = format!(
        "POST /v1/messages HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    let mut fragments = Fragments(vec![
        std::io::Cursor::new(headers.as_bytes()[..7].to_vec()),
        std::io::Cursor::new(headers.as_bytes()[7..].to_vec()),
        std::io::Cursor::new(body.clone()),
    ]);
    let request = read_gateway_request(&mut fragments).unwrap();
    assert_eq!(request.len(), headers.len() + body.len());
    assert_eq!(&request[headers.len()..], body);
    assert!(read_gateway_request(&mut &headers.as_bytes()[..]).is_err());
}

fn spawn_stub_gateway() -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    spawn_gateway(stub_gateway_response())
}

fn stub_gateway_response() -> &'static str {
    concat!(
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
    )
}

fn spawn_edit_gateway() -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    spawn_gateway(edit_gateway_response())
}

fn edit_gateway_response() -> &'static str {
    concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_edit\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"test\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n",
        "\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_edit\",\"name\":\"greppy\",\"input\":{}}}\n",
        "\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"args\\\":[\\\"write\\\",\\\"hello.txt\\\",\\\"partial\\\\n\\\"]}\"}}\n",
        "\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
        "\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":8}}\n",
        "\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n",
        "\n",
    )
}

#[cfg(unix)]
#[test]
fn coding_task_does_not_start_a_browser() {
    use std::os::unix::fs::PermissionsExt;
    let root = unique_temp("no-browser");
    let repo = root.join("repo");
    std::fs::create_dir(&repo).unwrap();
    init_repo(&repo);
    let marker = root.join("browser-was-started");
    let runtime = root.join("unexpected-runtime");
    std::fs::write(
        &runtime,
        "#!/bin/sh\nprintf started > \"$GREPPY_BROWSER_START_MARKER\"\nexit 91\n",
    )
    .unwrap();
    std::fs::set_permissions(&runtime, std::fs::Permissions::from_mode(0o700)).unwrap();
    let (endpoint, stop, handle) = spawn_gateway(stub_gateway_response());
    let output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", root.join("store"))
        .env("GREPPY_WORKSPACE_DIR", root.join("no-provider"))
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_WEB_RUNTIME", &runtime)
        .env("GREPPY_BROWSER_START_MARKER", &marker)
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .args([
            "-p",
            "Answer briefly without tools",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--private-store",
            "--skip-selfcheck",
            "--deadline-secs",
            "20",
            "--json",
        ])
        .output()
        .unwrap();
    stop.store(true, Ordering::SeqCst);
    handle.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !marker.exists(),
        "coding-only task started a browser runtime"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
fn web_gateway_response(args: serde_json::Value, id: &str) -> String {
    let events = [
        (
            "message_start",
            serde_json::json!({"type":"message_start","message":{
            "id":format!("msg_{id}"),"type":"message","role":"assistant","model":"test","content":[],
            "stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}),
        ),
        (
            "content_block_start",
            serde_json::json!({"type":"content_block_start","index":0,
            "content_block":{"type":"tool_use","id":id,"name":"greppy","input":{}}}),
        ),
        (
            "content_block_delta",
            serde_json::json!({"type":"content_block_delta","index":0,
            "delta":{"type":"input_json_delta","partial_json":serde_json::to_string(&args).unwrap()}}),
        ),
        (
            "content_block_stop",
            serde_json::json!({"type":"content_block_stop","index":0}),
        ),
        (
            "message_delta",
            serde_json::json!({"type":"message_delta",
            "delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":8}}),
        ),
        ("message_stop", serde_json::json!({"type":"message_stop"})),
    ];
    events
        .into_iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
        .collect()
}

#[cfg(unix)]
fn collect_runtime_scopes(
    value: &serde_json::Value,
    scopes: &mut std::collections::BTreeSet<(String, String)>,
) {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(result) = object.get("result") {
                if object.get("schema").and_then(serde_json::Value::as_str)
                    == Some("greppy.web-runtime.v1")
                    && object.get("status").and_then(serde_json::Value::as_str) == Some("ok")
                    && result["running"] == true
                    && result["owned"] == true
                {
                    if let (Some(socket), Some(run_id)) =
                        (result["socket"].as_str(), result["run_id"].as_str())
                    {
                        scopes.insert((socket.to_owned(), run_id.to_owned()));
                    }
                }
            }
            for child in object.values() {
                collect_runtime_scopes(child, scopes);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                collect_runtime_scopes(child, scopes);
            }
        }
        serde_json::Value::String(text) => {
            // Fenced web-do text contains one JSON document per operation.
            // Consume each complete document so a preceding session result
            // cannot hide the later authenticated runtime-status receipt.
            let mut remaining = text.as_str();
            while let Some(offset) = remaining.find('{') {
                let candidate = &remaining[offset..];
                let mut documents =
                    serde_json::Deserializer::from_str(candidate).into_iter::<serde_json::Value>();
                match documents.next() {
                    Some(Ok(document)) => {
                        collect_runtime_scopes(&document, scopes);
                        remaining = &candidate[documents.byte_offset()..];
                    }
                    _ => remaining = &candidate[1..],
                }
            }
        }
        _ => {}
    }
}

/// Run only as an actual child process of the confinement contract below.
#[cfg(unix)]
#[test]
#[ignore = "requires a task-owned confinement canary"]
fn sandbox_canary_child_probe() {
    use std::io::Write;
    let path = std::env::var_os("GREPPY_TEST_SANDBOX_CANARY")
        .expect("task-owned canary path");
    let result = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(b"canary-write-attempt\n"));
    match result {
        Ok(()) => println!("sandbox-canary-write-allowed"),
        Err(error) => {
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::PermissionDenied,
                "canary failed for a reason other than OS write denial: {error}"
            );
            println!("sandbox-canary-write-denied");
        }
    }
}

/// Real native startup and DOM read through the confined agent tool. A scripted
/// gateway selects the tool; the HTTP fixture's opaque title is the oracle.
/// The compatibility job builds and supplies the actual native runtime.
#[cfg(unix)]
#[test]
#[ignore = "requires the native runtime built by the compatibility job"]
fn confined_web_uses_parent_runtime_and_reads_real_page() {
    use recorded_http_fixture::{http_response, HttpFixture};
    let runtime =
        std::env::var_os("GREPPY_WEB_RUNTIME").expect("actual native runtime is required");
    assert!(std::path::Path::new(&runtime).is_file());
    let root = unique_temp("confined-web");
    let repo = root.join("repo");
    std::fs::create_dir(&repo).unwrap();
    init_repo(&repo);
    // A separate sibling belongs to this test, but to none of the agent's
    // worktree, Git, private data, per-run scratch or runtime write roots.
    // The actual denied write also catches any future overly broad root grant.
    let canary_dir = unique_temp("outside-confined-web");
    std::fs::create_dir_all(&canary_dir).unwrap();
    struct CanaryDirectory(std::path::PathBuf);
    impl Drop for CanaryDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _canary_owner = CanaryDirectory(canary_dir.clone());
    let canary = canary_dir.join("outside-write-canary");
    let original_canary: &[u8] = b"task-owned-canary-must-not-change\n";
    std::fs::write(&canary, original_canary).unwrap();
    let canary = std::fs::canonicalize(canary).unwrap();
    assert!(!canary.starts_with(std::fs::canonicalize(&root).unwrap()));
    let probe_binary = std::env::current_exe().unwrap();
    let probe_args = [
        "--exact", "sandbox_canary_child_probe",
        "--ignored", "--nocapture", "--test-threads=1",
    ];
    // The same executable and file must really be writable before confinement;
    // a read-only file, nonexistent path or broken probe cannot pass as denial.
    let control = Command::new(&probe_binary)
        .env("GREPPY_TEST_SANDBOX_CANARY", &canary)
        .args(probe_args)
        .output()
        .unwrap();
    assert!(
        control.status.success()
            && String::from_utf8_lossy(&control.stdout).contains("sandbox-canary-write-allowed")
            && std::fs::read(&canary).unwrap() != original_canary,
        "unconfined canary control did not perform the real write: {control:?}"
    );
    std::fs::write(&canary, original_canary).unwrap();
    let title = format!(
        "native-proof-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let html = format!("<!doctype html><title>{title}</title><h1>{title}</h1>");
    let fixture = HttpFixture::start(move |_| http_response(html.as_bytes(), "text/html", ""));
    let responses = vec![
        web_gateway_response(
            serde_json::json!({"args":[
                "bash-smart", "--", "/usr/bin/env",
                format!("GREPPY_TEST_SANDBOX_CANARY={}", canary.display()),
                probe_binary.to_str().unwrap(),
                "--exact", "sandbox_canary_child_probe",
                "--ignored", "--nocapture", "--test-threads=1"
            ]}),
            "toolu_sandbox_canary",
        ),
        web_gateway_response(
            serde_json::json!({"args":[
                "web","do","--json","session","new","--profile","project","::",
                "open",format!("{}proof", fixture.origin),"::","js","document.title",
                "::","runtime","status","--json"
            ]}),
            "toolu_web_first",
        ),
        web_gateway_response(
            serde_json::json!({"args":[
                "web","do","--json","js","document.title","::","runtime","status","--json"
            ]}),
            "toolu_web_reuse",
        ),
        stub_gateway_response().to_owned(),
    ];
    let messages: GatewayMessages = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (endpoint, stop, handle) = spawn_gateway_observed(
        responses,
        r#"{"data":[{"id":"test","max_completion_tokens":1024}]}"#,
        None,
        Some(Arc::clone(&messages)),
    );
    // A test-owned in-memory capability can clean up the deliberate no-shutdown
    // mutant through the official FD provider; never put it in argv/env/evidence.
    let cleanup_token = greppy::generate_attach_token().expect("test cleanup capability");
    let mut agent_command = Command::new(binary_path());
    agent_command
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", root.join("store"))
        .env("GREPPY_WORKSPACE_DIR", root.join("no-provider"))
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_WEB_RUNTIME", &runtime)
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .env_remove("GREPPY_NO_SANDBOX")
        .args([
            "-p",
            "Read the real page title with the browser",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--private-store",
            "--skip-selfcheck",
            "--enable",
            "web",
            "--max-turns",
            "4",
            "--deadline-secs",
            "30",
            "--json",
        ]);
    let attach = greppy::give_child_attach_token(&mut agent_command, &cleanup_token)
        .expect("official agent control FD");
    let output = agent_command.output().unwrap();
    drop(attach);
    stop.store(true, Ordering::SeqCst);
    handle.join().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        std::fs::read(&canary).unwrap(),
        original_canary,
        "confined tool changed outside-write canary"
    );
    let events: Vec<_> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .collect();
    let session = events.iter().find(|event| event["type"] == "session")
        .expect("actual agent session event");
    assert_eq!(
        session["sandbox"], "confined",
        "native agent test did not select confined mode: {stdout}\n{stderr}"
    );
    let finishes: Vec<_> = events.iter()
        .filter(|event| event["type"] == "tool_finish")
        .collect();
    let wire = messages.lock().unwrap();
    let mut genuine = std::collections::BTreeSet::new();
    let mut scopes = std::collections::BTreeSet::new();
    let mut canary_denied = false;
    for request in wire.iter() {
        for message in request["messages"].as_array().into_iter().flatten() {
            for block in message["content"].as_array().into_iter().flatten() {
                if block["type"] == "tool_result" && block["is_error"] != true {
                    if block["tool_use_id"] == "toolu_sandbox_canary"
                        && block.to_string().contains("sandbox-canary-write-denied")
                    {
                        canary_denied = true;
                    }
                    if block.to_string().contains(&title) {
                        genuine.insert(block["tool_use_id"].as_str().unwrap().to_owned());
                    }
                    collect_runtime_scopes(block, &mut scopes);
                }
            }
        }
    }
    assert!(
        canary_denied,
        "confined tool did not report real OS canary write denial: {stdout}\n{stderr}"
    );
    assert!(
        output.status.success()
            && genuine.contains("toolu_web_first")
            && genuine.contains("toolu_web_reuse")
            && finishes.len() == 3
            && finishes.iter().all(|event| event["failed"] == false),
        "confined web tool did not read the genuine native page: {stdout}\n{stderr}"
    );
    assert_eq!(
        scopes.len(),
        1,
        "web tool calls did not reuse one authenticated runtime: {scopes:?}"
    );
    let (socket, run_id) = scopes.into_iter().next().unwrap();
    drop(wire);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match std::os::unix::net::UnixStream::connect(&socket) {
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) =>
            {
                break
            }
            Err(error) => panic!("cannot verify agent runtime socket closure: {error}"),
            Ok(_) if std::time::Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10))
            }
            Ok(_) => {
                // A cleanup mutant must still release its own authenticated test
                // runtime through the public stop operation before failing.
                let mut stop_command = Command::new(binary_path());
                stop_command
                    .current_dir(&repo)
                    .env("GREPPY_STORE_DIR", root.join("store"))
                    .env("GREPPY_WEB_RUNTIME", &runtime)
                    .env("GREPPY_RUN_ID", &run_id)
                    .env(
                        "GREPPY_WEB_RUNTIME_DIR",
                        greppy_agent::greppy_env::agent_runtime_dir(&run_id),
                    )
                    .env_remove("GREPPY_WEB_RUNTIME_DIST")
                    .args(["web", "runtime", "stop", "--json"]);
                let attach = greppy::give_child_attach_token(&mut stop_command, &cleanup_token)
                    .expect("official cleanup control FD");
                let stopped = stop_command.output().unwrap();
                drop(attach);
                assert!(
                    stopped.status.success(),
                    "test could not stop its retained runtime"
                );
                let stop_result: serde_json::Value =
                    serde_json::from_slice(&stopped.stdout).expect("public stop returned JSON");
                assert_eq!(
                    stop_result["result"]["socket"].as_str(),
                    Some(socket.as_str()),
                    "test recovery selected a different runtime namespace: {stop_result}"
                );
                assert_eq!(
                    stop_result["result"]["run_id"].as_str(),
                    Some(run_id.as_str()),
                    "test recovery selected a different run: {stop_result}"
                );
                let cleanup_deadline = std::time::Instant::now() + Duration::from_secs(5);
                while std::os::unix::net::UnixStream::connect(&socket).is_ok() {
                    assert!(
                        std::time::Instant::now() < cleanup_deadline,
                        "test runtime stop did not close its socket: {stop_result}"
                    );
                    thread::sleep(Duration::from_millis(10));
                }
                panic!("agent exit retained its authenticated runtime socket");
            }
        }
    }
    assert_eq!(
        fixture
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|path| path.as_str() == "/proof")
            .count(),
        1,
        "browser startup must issue exactly one genuine page request"
    );
    assert!(
        !repo.join(".greppy/web").exists(),
        "parent route escaped into the source checkout"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn greppy_p_without_provider_preserves_dirty_source_and_publishes_proposal() {
    assert_provider_optional_proposal(false);
}

#[test]
fn greppy_p_with_inactive_provider_preserves_dirty_source_and_publishes_proposal() {
    assert_provider_optional_proposal(true);
}

fn assert_provider_optional_proposal(installed_inactive: bool) {
    let root = unique_temp("no-provider-agent");
    let repo = root.join("repo");
    std::fs::create_dir(&repo).unwrap();
    init_repo(&repo);
    std::fs::write(repo.join("hello.txt"), "staged\n").unwrap();
    git(&repo, &["add", "hello.txt"]);
    std::fs::write(repo.join("hello.txt"), "local\n").unwrap();
    std::fs::write(repo.join("untracked.txt"), "keep me\n").unwrap();
    let index_before = std::fs::read(repo.join(".git/index")).unwrap();
    let no_provider = root.join("workspace-without-provider");
    let manifest_bytes = installed_inactive.then(|| {
        std::fs::create_dir(&no_provider).unwrap();
        let manifest = serde_json::json!({
            "protocol_version": greppy_workspace_core::PROVIDER_PROTOCOL_VERSION,
            "adapter_version": "0.4.1", "adapter_kind": greppy_workspace_core::AdapterKind::FsKit,
            "state": "ready", "instance_id": "installed-inactive",
            "data_root": no_provider, "mount_root": root.join("unmounted"),
            "heartbeat_unix_ms": 1,
            "capabilities": {"hard_links":true,"symbolic_links":true,"byte_range_locks":true,
                "memory_maps":true,"atomic_rename":true,"case_preserving":true}
        });
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(no_provider.join("provider.json"), &bytes).unwrap();
        bytes
    });
    let (endpoint, stop, handle) =
        spawn_gateway_sequence(vec![edit_gateway_response(), stub_gateway_response()]);
    let mut command = Command::new(binary_path());
    command
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", root.join("store"))
        .env("GREPPY_WORKSPACE_DIR", &no_provider)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "change hello.txt",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--private-store",
            "--skip-selfcheck",
            "--json",
        ]);
    if std::env::var_os("GREPPY_ACCEPTANCE_BINARY").is_some() {
        command
            .env_remove("GREPPY_TEST_SKIP_INFERENCE")
            .env_remove("CI");
    }
    let output = command.output().expect("spawn no-provider agent");
    stop.store(true, Ordering::SeqCst);
    handle.join().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stdout}\n{stderr}");
    let result = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|event| event["type"] == "result")
        .expect("completed agent result");
    assert_eq!(result["turns"], 2, "{result}");
    assert_eq!(result["applied"], false, "{result}");
    let proposal = result["proposal_ref"].as_str().expect("proposal ref");
    assert!(proposal.starts_with("refs/greppy/agent/"), "{proposal}");
    assert_eq!(result["final_answer"], "hi from stub");
    let message = Command::new("git")
        .current_dir(&repo)
        .args(["show", "-s", "--format=%s", proposal])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&message.stdout).trim(),
        "hi from stub"
    );
    let paths = Command::new("git")
        .current_dir(&repo)
        .args(["diff-tree", "--no-commit-id", "--name-only", "-r", proposal])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&paths.stdout).trim(), "hello.txt");
    let session = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|event| event["type"] == "session")
        .unwrap();
    assert!(!PathBuf::from(session["worktree"].as_str().unwrap()).exists());
    let shown = Command::new("git")
        .current_dir(&repo)
        .args(["show", &format!("{proposal}:hello.txt")])
        .output()
        .unwrap();
    assert!(shown.status.success());
    assert_eq!(shown.stdout, b"partial\n");
    assert_eq!(std::fs::read(repo.join("hello.txt")).unwrap(), b"local\n");
    assert_eq!(
        std::fs::read(repo.join("untracked.txt")).unwrap(),
        b"keep me\n"
    );
    assert_eq!(
        std::fs::read(repo.join(".git/index")).unwrap(),
        index_before
    );
    match manifest_bytes {
        Some(bytes) => assert_eq!(
            std::fs::read(no_provider.join("provider.json")).unwrap(),
            bytes
        ),
        None => assert!(!no_provider.join("provider.json").exists()),
    }
    assert!(
        !root.join("unmounted").exists(),
        "must not activate a provider"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn greppy_p_http_error_saves_partial_proposal_without_applying() {
    for status in [502_u16, 503] {
        let root = unique_temp("http-error-proposal");
        let repo = root.join("repo");
        std::fs::create_dir(&repo).unwrap();
        init_repo(&repo);
        std::fs::write(repo.join("hello.txt"), "staged\n").unwrap();
        git(&repo, &["add", "hello.txt"]);
        std::fs::write(repo.join("hello.txt"), "local\n").unwrap();
        std::fs::write(repo.join("untracked.txt"), "keep me\n").unwrap();
        let index_before = std::fs::read(repo.join(".git/index")).unwrap();
        let (endpoint, stop, handle) = spawn_gateway_contract_with_error(
            vec![edit_gateway_response()],
            r#"{"data":[{"id":"test","max_completion_tokens":1024}]}"#,
            None,
            Some(status),
        );
        let mut command = Command::new(binary_path());
        command
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", root.join("store"))
            .env("GREPPY_WORKSPACE_DIR", root.join("ordinary-workspace"))
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env_remove("GREPPY_MODEL")
            .env_remove("GREPPY_ENDPOINT")
            .env_remove("GREPPY_API_KEY")
            .args([
                "-p",
                "change hello.txt",
                "--model",
                "test",
                "--endpoint",
                &endpoint,
                "--deadline-secs",
                "120",
                "--private-store",
                "--skip-selfcheck",
                "--apply",
                "--json",
            ]);
        if std::env::var_os("GREPPY_ACCEPTANCE_BINARY").is_some() {
            command
                .env_remove("GREPPY_TEST_SKIP_INFERENCE")
                .env_remove("CI");
        }
        let output = command.output().expect("real agent edit then HTTP error");
        stop.store(true, Ordering::SeqCst);
        handle.join().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(3), "{stdout}\n{stderr}");
        let events: Vec<serde_json::Value> = stdout
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        let result = events
            .iter()
            .find(|event| event["type"] == "result")
            .expect("failed agent result");
        assert_eq!(result["status"], "error", "{result}");
        assert_eq!(result["applied"], false, "{result}");
        assert!(
            stderr.contains("agent error:") && stderr.contains(&status.to_string()),
            "HTTP error must remain visible: {stderr}"
        );
        let proposal = result["proposal_ref"]
            .as_str()
            .filter(|value| !value.is_empty())
            .expect("HTTP error must publish the partial proposal");
        let shown = Command::new("git")
            .current_dir(&repo)
            .args(["show", &format!("{proposal}:hello.txt")])
            .output()
            .unwrap();
        assert!(shown.status.success(), "{shown:?}");
        assert_eq!(
            shown.stdout, b"partial\n",
            "real edit must be preserved exactly"
        );
        let changed = Command::new("git")
            .current_dir(&repo)
            .args(["diff-tree", "--no-commit-id", "--name-only", "-r", proposal])
            .output()
            .unwrap();
        assert!(changed.status.success(), "{changed:?}");
        assert_eq!(
            changed.stdout, b"hello.txt\n",
            "user changes are not the agent delta"
        );
        assert_eq!(std::fs::read(repo.join("hello.txt")).unwrap(), b"local\n");
        assert_eq!(
            std::fs::read(repo.join("untracked.txt")).unwrap(),
            b"keep me\n"
        );
        assert_eq!(
            std::fs::read(repo.join(".git/index")).unwrap(),
            index_before
        );
        let worktree = events
            .iter()
            .find(|event| event["type"] == "session")
            .unwrap()["worktree"]
            .as_str()
            .unwrap();
        assert!(
            !PathBuf::from(worktree).exists(),
            "saved partial proposal must release its temporary worktree"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn greppy_p_text_only_end_turn_proposes_nothing() {
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
        // Avoid pulling a real model path from the parent environment.
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
            "--private-store",
            // Self-check is covered by agent_selfcheck.rs; keep this test on the
            // gateway/loop contract without depending on a warm index.
            "--skip-selfcheck",
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
        stdout.contains("hi from stub"),
        "expected streamed text in stdout: {stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("no changes proposed."),
        "expected clean outcome in stdout: {stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("store mode: private (--private-store)"),
        "expected explicit full-private fallback mode: {stdout}\nstderr={stderr}"
    );
    let session_id = session_id_from_stderr(&stderr);
    assert!(
        !session_id.is_empty(),
        "expected a non-empty session id; stderr={stderr}"
    );

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
    drop(provider);
    let _ = std::fs::remove_dir_all(&provider_root);
}

#[test]
fn caller_output_limit_reaches_wire_and_continues() {
    let repo = unique_temp("caller-limit-repo");
    init_repo(&repo);
    let store = unique_temp("caller-limit-store");
    let provider_root = unique_temp("caller-limit-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let truncated: &'static str = Box::leak(
        stub_gateway_response()
            .replace("end_turn", "max_tokens")
            .replace("hi from stub", "partial from stub")
            .into_boxed_str(),
    );
    // The route exposes no cap. Caller metadata must reach both actual POSTs.
    let (endpoint, stop, handle) = spawn_gateway_contract(
        vec![truncated, stub_gateway_response()],
        r#"{"data":[{"id":"glm-5.3"}]}"#,
        Some(("glm-5.3", 65536)),
    );
    let invoke = |bound: Option<&str>| {
        let mut command = Command::new(binary_path());
        command
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", &store)
            .env("GREPPY_WORKSPACE_DIR", &provider.data)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env_remove("GREPPY_MODEL")
            .env_remove("GREPPY_ENDPOINT")
            .args([
                "-p",
                "say hi",
                "--model",
                "glm-5.3",
                "--endpoint",
                &endpoint,
                "--private-store",
                "--skip-selfcheck",
            ]);
        if let Some(bound) = bound {
            command.args(["--max-output-tokens", bound]);
        }
        command.output().expect("spawn caller-bound CLI")
    };
    let unknown = invoke(None);
    assert_eq!(unknown.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&unknown.stderr)
        .contains("no advertised or registered output limit"));
    let bounded = invoke(Some("65536"));
    stop.store(true, Ordering::SeqCst);
    handle.join().expect("actual request body contract");
    assert_eq!(
        bounded.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&bounded.stderr)
    );
    let text = String::from_utf8_lossy(&bounded.stdout);
    assert!(
        text.contains("partial from stub") && text.contains("hi from stub"),
        "truncated output must continue: {text}"
    );
    drop(provider);
    for path in [repo, store, provider_root] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn greppy_p_aborts_before_model_turn_when_base_preparation_fails() {
    let repo = unique_temp("base-fail-repo");
    init_repo(&repo);
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), b"pub fn value() -> usize { 7 }\n").unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    git(&repo, &["commit", "-m", "code"]);
    let store = unique_temp("base-fail-store");
    let provider_root = unique_temp("base-fail-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);
    let (endpoint, stop, handle) = spawn_stub_gateway();

    let output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_WORKSPACE_DIR", &provider.data)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_TEST_BASE_SUMMARY_FAIL", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "must not reach the model",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--max-turns",
            "1",
            "--skip-selfcheck",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -p");

    stop.store(true, Ordering::SeqCst);
    let _ = handle.join();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_ne!(
        output.status.code(),
        Some(0),
        "Base failure must abort the agent; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("shared Base unavailable")
            && stderr.contains("agent start aborted before the first model call"),
        "agent failure must identify the fail-closed Base boundary; stderr={stderr}"
    );
    let _ = session_id_from_stderr(&stderr);
    assert!(
        !stdout.contains("hi from stub") && !stdout.contains("no changes proposed"),
        "no model turn or proposal outcome may run after Base failure; stdout={stdout}"
    );
    assert!(
        try_find_file_named(&store, "COMPLETE").is_none(),
        "failed agent Base preparation must not publish COMPLETE"
    );

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
    drop(provider);
    let _ = std::fs::remove_dir_all(&provider_root);
}

/// F9: leading `-p` is reserved for the agent, but `greppy -e -p X` must reach
/// the grep passthrough (not the agent interceptor).
#[test]
fn greppy_e_dash_p_is_not_intercepted_as_agent() {
    let dir = unique_temp("e-dash-p");
    std::fs::write(dir.join("hay.txt"), b"alpha\n-p needle line\nbeta\n").unwrap();

    let output = Command::new(binary_path())
        .current_dir(&dir)
        .env("GREPPY_STORE_DIR", unique_temp("e-dash-p-store"))
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args(["-e", "-p", "hay.txt"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn greppy -e -p");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Must NOT be the agent path (which would complain about missing TASK/model/gateway).
    assert!(
        !stderr.contains("greppy -p needs a local model gateway")
            && !stderr.contains("missing TASK")
            && !stderr.contains("--model is required"),
        "was intercepted as agent:\nstdout={stdout}\nstderr={stderr}"
    );
    // Grep for the literal pattern `-p` should match the haystack line.
    assert!(
        stdout.contains("-p needle line"),
        "expected grep hit for literal -p; stdout={stdout}\nstderr={stderr}"
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={stdout}\nstderr={stderr}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn greppy_p_limits_report_incomplete_and_deliver_outcome() {
    // Both limit stops must still deliver a usable result.
    for (limit_args, expected_stop) in [
        (
            vec!["--max-turns", "4", "--deadline-secs", "0"],
            "deadline reached",
        ),
        (vec!["--max-turns", "1"], "turn limit reached"),
    ] {
        let repo = unique_temp("limited-repo");
        init_repo(&repo);
        let store = unique_temp("limited-store");
        let provider_root = unique_temp("limited-provider");
        let provider = spawn_fake_provider(&provider_root, &repo);
        let (endpoint, stop, handle) = if expected_stop == "turn limit reached" {
            spawn_edit_gateway()
        } else {
            spawn_stub_gateway()
        };
        let output = Command::new(binary_path())
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", &store)
            .env("GREPPY_WORKSPACE_DIR", &provider.data)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env_remove("GREPPY_MODEL")
            .env_remove("GREPPY_ENDPOINT")
            .env_remove("GREPPY_DEADLINE_SECS")
            .args([
                "-p",
                "say hi",
                "--model",
                "test",
                "--endpoint",
                &endpoint,
                "--private-store",
                "--skip-selfcheck",
                "--json",
            ])
            .args(limit_args)
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
            Some(5),
            "stdout={stdout}\nstderr={stderr}"
        );
        let result = stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|event| event["type"] == "result")
            .expect("final result must be delivered when a limit is reached");
        assert_eq!(result["status"], "incomplete");
        assert_eq!(result["exit_code"], 5);
        assert_eq!(result["stop"], expected_stop);
        // `turns` counts model turns: the one action turn plus the tool-free
        // report turn that always follows the action-turn cap.
        assert_eq!(
            result["turns"],
            if expected_stop == "turn limit reached" {
                2
            } else {
                0
            }
        );
        assert!(!result["session_id"].as_str().unwrap().is_empty());
        assert_eq!(result["applied"], false);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&store);
        drop(provider);
        let _ = std::fs::remove_dir_all(&provider_root);
    }
}

#[test]
fn greppy_p_incomplete_proposal_is_not_applied_and_cleans_worktree() {
    let repo = unique_temp("partial-proposal-repo");
    init_repo(&repo);
    let store = unique_temp("partial-proposal-store");
    let provider_root = unique_temp("partial-proposal-provider");
    let provider = spawn_fake_provider_with_edits(
        &provider_root,
        &repo,
        vec![(PathBuf::from("hello.txt"), b"partial\n".to_vec())],
    );
    let (endpoint, stop, handle) = spawn_edit_gateway();

    let output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_WORKSPACE_DIR", &provider.data)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .args([
            "-p",
            "change hello.txt",
            "--model",
            "test",
            "--endpoint",
            &endpoint,
            "--max-turns",
            "1",
            "--private-store",
            "--skip-selfcheck",
            "--apply",
            "--json",
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
        Some(5),
        "stdout={stdout}\nstderr={stderr}"
    );
    let result = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|event| event["type"] == "result")
        .expect("incomplete proposal result");
    assert_eq!(result["status"], "incomplete");
    assert_eq!(result["stop"], "turn limit reached");
    // One action turn plus the tool-free report turn after the cap.
    assert_eq!(result["turns"], 2);
    assert_eq!(result["applied"], false);
    let proposal = result["proposal_ref"].as_str().unwrap_or_else(|| {
        panic!("partial proposal must remain inspectable; stdout={stdout}\nstderr={stderr}")
    });
    assert!(!proposal.is_empty(), "{result}");
    assert_eq!(
        std::fs::read_to_string(repo.join("hello.txt")).unwrap(),
        "hello\n",
        "--apply must not stage an incomplete proposal"
    );
    let shown = Command::new("git")
        .args(["show", "--format=", proposal, "--", "hello.txt"])
        .current_dir(&repo)
        .output()
        .expect("inspect proposal ref");
    assert!(
        shown.status.success(),
        "proposal={proposal} stderr={}",
        String::from_utf8_lossy(&shown.stderr)
    );
    assert!(
        String::from_utf8_lossy(&shown.stdout).contains("partial"),
        "proposal={proposal} stdout={}",
        String::from_utf8_lossy(&shown.stdout)
    );
    let worktree = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|event| event["type"] == "session")
        .unwrap()["worktree"]
        .as_str()
        .map(PathBuf::from)
        .unwrap();
    // The fixture provider mirrors workspaces into its mount on its own
    // thread and drops the view asynchronously once the core forgets the
    // workspace; under load that lags the agent's exit.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while worktree.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        !worktree.exists(),
        "saved incomplete proposal must clean temporary worktree"
    );
    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
    drop(provider);
    let _ = std::fs::remove_dir_all(&provider_root);
}

#[test]
fn greppy_p_persists_session_and_continue_reuses_id() {
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
        .expect("spawn greppy -p");
    let first_stdout = String::from_utf8_lossy(&first.stdout);
    let first_stderr = String::from_utf8_lossy(&first.stderr);
    assert_eq!(
        first.status.code(),
        Some(0),
        "stdout={first_stdout}\nstderr={first_stderr}"
    );
    let session_id = session_id_from_stderr(&first_stderr);
    let jsonl = session_jsonl(&store);
    let first_log = std::fs::read_to_string(&jsonl).unwrap_or_default();
    assert_eq!(
        usage_line_count(&first_log),
        1,
        "first run should append one usage line; jsonl={first_log}"
    );

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
        .expect("spawn greppy -p --continue");
    let second_stdout = String::from_utf8_lossy(&second.stdout);
    let second_stderr = String::from_utf8_lossy(&second.stderr);
    assert_eq!(
        second.status.code(),
        Some(0),
        "stdout={second_stdout}\nstderr={second_stderr}"
    );
    assert_eq!(
        session_id_from_stderr(&second_stderr),
        session_id,
        "continue must reuse the same session id; stderr={second_stderr}"
    );
    let second_log = std::fs::read_to_string(&jsonl).unwrap_or_default();
    assert_eq!(
        usage_line_count(&second_log),
        2,
        "continue should append a second usage line; jsonl={second_log}"
    );

    stop.store(true, Ordering::SeqCst);
    let _ = handle.join();
    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&store);
    drop(provider);
    let _ = std::fs::remove_dir_all(&provider_root);
}

#[test]
fn nested_agent_run_is_refused() {
    // GREPPY_AGENT_RUN is set in every tool subprocess of a running agent;
    // a second `greppy -p` must refuse before doing any work. Command-scoped
    // env only — parallel-test safe.
    let dir = unique_temp("nested");
    let output = Command::new(binary_path())
        .args(["-p", "do something", "--model", "m"])
        .env("GREPPY_AGENT_RUN", "1")
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .current_dir(&dir)
        .stdin(Stdio::null())
        .output()
        .expect("spawn greppy");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("refusing a nested agent run"),
        "stderr={stderr}"
    );
    // Refusal must happen before gateway probing or workspace creation.
    assert!(
        !stderr.contains("needs a local model gateway"),
        "stderr={stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn workspace_doctor_json_is_machine_readable_when_provider_is_missing() {
    let dir = unique_temp("doctor-missing");
    let missing = dir.join("not-created");
    let output = Command::new(binary_path())
        .args(["workspace", "doctor", "--json"])
        .env("GREPPY_WORKSPACE_DIR", &missing)
        .output()
        .expect("spawn workspace doctor");
    assert_eq!(output.status.code(), Some(73));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["healthy"], false);
    assert!(value["error"].as_str().unwrap().contains("provider.json"));
    assert_eq!(
        value["diagnostics"]["checks"]["control_manifest"]["status"],
        "failed"
    );
    assert_eq!(
        value["diagnostics"]["checks"]["heartbeat"]["status"],
        "not_checked"
    );
    assert_eq!(value["smoke_status"], "not_run");
    assert!(!value["next_actions"].as_array().unwrap().is_empty());
    assert!(
        !missing.exists(),
        "diagnosis must not initialize a missing store"
    );
    let plain = Command::new(binary_path())
        .args(["workspace", "doctor"])
        .env("GREPPY_WORKSPACE_DIR", &missing)
        .output()
        .expect("spawn plain workspace doctor");
    assert_eq!(plain.status.code(), Some(73));
    let stderr = String::from_utf8_lossy(&plain.stderr);
    assert!(stderr.contains("control_manifest: failed"), "{stderr}");
    assert!(stderr.contains("heartbeat: not_checked"), "{stderr}");
    assert!(stderr.contains("next:"), "{stderr}");
    assert!(!missing.exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn workspace_doctor_explains_stale_heartbeat_without_hiding_mount_identity() {
    let dir = unique_temp("doctor-stale");
    let data = dir.join("data");
    let mount = dir.join("mount");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&mount).unwrap();
    let manifest = serde_json::json!({
        "protocol_version": 1, "adapter_version": "0.4.0", "adapter_kind": "fuse3",
        "state": "ready", "instance_id": "doctor-stale", "data_root": data,
        "mount_root": mount, "heartbeat_unix_ms": 1,
        "capabilities": {"hard_links":true,"symbolic_links":true,"byte_range_locks":true,
            "memory_maps":true,"atomic_rename":true,"case_preserving":true}
    });
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(data.join("provider.json"), &bytes).unwrap();
    std::fs::write(mount.join(".greppy-provider.json"), &bytes).unwrap();
    let output = Command::new(binary_path())
        .args(["workspace", "doctor", "--json"])
        .env("GREPPY_WORKSPACE_DIR", &data)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(73));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["healthy"], false);
    assert_eq!(
        value["diagnostics"]["checks"]["heartbeat"]["status"],
        "failed"
    );
    assert_eq!(
        value["diagnostics"]["checks"]["mounted_identity"]["status"],
        "passed"
    );
    assert_eq!(value["smoke_status"], "not_run");
    assert!(!data.join("core").exists());
    assert!(!mount.join("doctor").exists());
    assert_eq!(std::fs::read(data.join("provider.json")).unwrap(), bytes);
    let _ = std::fs::remove_dir_all(dir);
}

/// `index --agent-worktree` warms the portable namespace `-p` will use.
///
/// The warm-up must exercise a provider namespace while publishing a reusable
/// immutable index Base. The temporary namespace is removed afterwards and is
/// never registered as a native Git worktree.
#[test]
#[cfg(not(feature = "ci-test-assets"))]
fn index_agent_worktree_warms_the_tree_the_agent_will_use() {
    let dir = unique_temp("index-agent-worktree");
    let repo = dir.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    init_repo(&repo);
    std::fs::write(
        repo.join("src/lib.rs"),
        b"pub fn parse_path(p: &str) -> usize { p.len() }\n",
    )
    .unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    git(&repo, &["commit", "-m", "code"]);

    let store = dir.join("store");
    let provider = spawn_fake_provider(&dir, &repo);
    let out = std::process::Command::new(binary_path())
        .args(["index", "--agent-worktree"])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_WORKSPACE_DIR", &provider.data)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("spawn greppy");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(
        out.status.code(),
        Some(0),
        "warm-up must succeed; stdout={stdout}\nstderr={stderr}"
    );

    // It has to SAY which tree it warmed: the caller cannot poll a path it
    // was never told.
    let worktree = stdout
        .lines()
        .find_map(|line| line.strip_prefix("agent worktree: "))
        .map(str::to_string)
        .unwrap_or_else(|| panic!("warm-up must name the worktree; stdout={stdout}"));
    assert!(
        worktree.contains("workspaces"),
        "warmed path must be in the provider namespace; got {worktree}"
    );
    for _ in 0..50 {
        if !std::path::Path::new(&worktree).exists() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !std::path::Path::new(&worktree).exists(),
        "temporary provider namespace must be cleaned after warm-up"
    );

    // Portable workspaces are private namespaces, never native Git worktrees.
    let listed = std::process::Command::new("git")
        .args(["worktree", "list"])
        .current_dir(&repo)
        .output()
        .expect("git worktree list");
    assert!(
        !String::from_utf8_lossy(&listed.stdout).contains(&worktree),
        "portable namespace must not be registered as a native Git worktree"
    );

    let base_root = store.join("agent-base-stores");
    let complete = find_file_named(&base_root, "COMPLETE");
    let base_dir = complete
        .parent()
        .unwrap_or_else(|| panic!("COMPLETE must have a Base directory parent"));
    assert!(base_dir.join("graph.db").is_file(), "Base graph is missing");
    assert!(
        base_dir.join("summary_cache.db").is_file(),
        "Base summary cache is missing"
    );
    assert!(
        base_dir.join("manifest.json").is_file(),
        "Base manifest is missing"
    );

    drop(provider);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn index_agent_worktree_fails_closed_when_base_summary_cache_publication_fails() {
    let dir = unique_temp("index-agent-worktree-summary-failure");
    let repo = dir.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    init_repo(&repo);
    std::fs::write(repo.join("src/lib.rs"), b"pub fn value() -> usize { 7 }\n").unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    git(&repo, &["commit", "-m", "code"]);

    let store = dir.join("store");
    let provider = spawn_fake_provider(&dir, &repo);
    let out = std::process::Command::new(binary_path())
        .args(["index", "--agent-worktree"])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_WORKSPACE_DIR", &provider.data)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_TEST_BASE_SUMMARY_FAIL", "1")
        .output()
        .expect("spawn greppy");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert_ne!(
        out.status.code(),
        Some(0),
        "failed Base publication must not report success; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("agent Base prewarm failed closed"),
        "failure must name the fail-closed Base boundary; stderr={stderr}"
    );
    assert!(
        stderr.contains("injected immutable Base summary cache publication failure"),
        "test must reach the post-graph summary publication boundary; stderr={stderr}"
    );
    assert!(
        !stderr.contains("warming a full private Store"),
        "Base failure must not fall back to a private Store; stderr={stderr}"
    );
    assert!(
        try_find_file_named(&store, "COMPLETE").is_none(),
        "failed Base publication must not leave a COMPLETE marker"
    );

    drop(provider);
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(not(feature = "ci-test-assets"))]
fn find_file_named(root: &std::path::Path, name: &str) -> PathBuf {
    try_find_file_named(root, name)
        .unwrap_or_else(|| panic!("{name} not found under {}", root.display()))
}

fn session_id_from_stderr(stderr: &str) -> String {
    let line = stderr.lines().next().unwrap_or("");
    assert!(
        line.starts_with("session: "),
        "first stderr line must be session: <id>; stderr={stderr}"
    );
    line["session: ".len()..].trim().to_string()
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

fn usage_line_count(jsonl: &str) -> usize {
    jsonl
        .lines()
        .filter(|line| line.contains("\"type\":\"usage\""))
        .count()
}

fn try_find_file_named(root: &std::path::Path, name: &str) -> Option<PathBuf> {
    let entries =
        std::fs::read_dir(root).unwrap_or_else(|error| panic!("read {}: {error}", root.display()));
    for entry in entries {
        let path = entry.expect("read directory entry").path();
        if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = try_find_file_named(&path, name) {
                return Some(found);
            }
        }
    }
    None
}
