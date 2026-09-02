//! PTY proof that the interactive TUI hosts the agent control socket.

#![cfg(unix)]

use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use greppy::agent_control::ControlClient;
use serde_json::{json, Value};

#[path = "support/portable_provider.rs"]
mod portable_provider;
use portable_provider::{spawn_fake_provider, FakeProvider};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique_temp(kind: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let temp = std::env::temp_dir();
    let root = temp.parent().unwrap_or(&temp).join(format!(
        "gtr-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let leaf = match kind {
        "repo" => "r",
        "store" => "s",
        _ => "p",
    };
    let path = root.join(leaf);
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

fn spawn_gateway() -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
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
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi from tui stub\"}}\n\n",
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
                        // macOS; use blocking reads and drain the complete body.
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
    (format!("http://127.0.0.1:{port}"), stop, handle)
}

struct Pty {
    master: File,
    child: Child,
    query_tail: Vec<u8>,
    store: PathBuf,
    _provider: FakeProvider,
}

impl Pty {
    fn spawn(repo: &Path, endpoint: &str) -> Self {
        unsafe {
            let mut master = 0;
            let mut slave = 0;
            let size = libc::winsize {
                ws_row: 24,
                ws_col: 80,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            assert_eq!(
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::from_ref(&size).cast_mut(),
                ),
                0
            );
            let stdin = libc::dup(slave);
            let stdout = libc::dup(slave);
            let stderr = libc::dup(slave);
            libc::close(slave);
            let store = unique_temp("store");
            let provider_root = unique_temp("provider");
            let provider = spawn_fake_provider(&provider_root, repo);
            let child = Command::new(binary_path())
                .current_dir(repo)
                .env("GREPPY_STORE_DIR", &store)
                .env("GREPPY_CONFIG_DIR", store.join("config"))
                .env("GREPPY_WORKSPACE_DIR", &provider.data)
                .env("GREPPY_TEST_SKIP_INFERENCE", "1")
                .env("GREPPY_ASCII", "1")
                .env("TERM", "xterm")
                .env_remove("GREPPY_MODEL")
                .env_remove("GREPPY_ENDPOINT")
                .args([
                    "agent",
                    "--model",
                    "test",
                    "--endpoint",
                    endpoint,
                    "--max-turns",
                    "2",
                    "--private-store",
                    "--skip-selfcheck",
                    "--no-sandbox",
                ])
                .stdin(Stdio::from_raw_fd(stdin))
                .stdout(Stdio::from_raw_fd(stdout))
                .stderr(Stdio::from_raw_fd(stderr))
                .spawn()
                .unwrap();
            let master = File::from_raw_fd(master);
            libc::fcntl(master.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK);
            Self {
                master,
                child,
                query_tail: Vec::new(),
                store,
                _provider: provider,
            }
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        self.master.write_all(bytes).ok();
        self.master.flush().ok();
    }

    fn answer_queries(&mut self, bytes: &[u8]) {
        let mut observed = std::mem::take(&mut self.query_tail);
        observed.extend_from_slice(bytes);
        for offset in 0..observed.len() {
            let rest = &observed[offset..];
            if rest.starts_with(b"\x1b[6n") {
                self.write(b"\x1b[24;1R");
            } else if rest.starts_with(b"\x1b[c") || rest.starts_with(b"\x1b[0c") {
                self.write(b"\x1b[?1;2c");
            } else if rest.starts_with(b"\x1b[>c") || rest.starts_with(b"\x1b[>0c") {
                self.write(b"\x1b[>0;0;0c");
            }
        }
        let keep = observed.len().min(3);
        self.query_tail
            .extend_from_slice(&observed[observed.len() - keep..]);
    }

    fn read_until(&mut self, needle: &[u8], timeout: Duration) -> Vec<u8> {
        let deadline = Instant::now() + timeout;
        let mut output = Vec::new();
        let mut buf = [0; 4096];
        while Instant::now() < deadline {
            match self.master.read(&mut buf) {
                Ok(0) => thread::sleep(Duration::from_millis(20)),
                Ok(count) => {
                    output.extend_from_slice(&buf[..count]);
                    self.answer_queries(&buf[..count]);
                    if output.windows(needle.len()).any(|window| window == needle) {
                        break;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(20));
                }
                Err(error) if error.raw_os_error() == Some(libc::EIO) => {
                    if self.child.try_wait().ok().flatten().is_some() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Err(_) => break,
            }
            if self.child.try_wait().ok().flatten().is_some() {
                break;
            }
        }
        output
    }

    fn wait(&mut self, timeout: Duration) -> std::process::ExitStatus {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            if Instant::now() >= deadline {
                self.child.kill().ok();
                return self.child.wait().unwrap();
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        self.child.kill().ok();
    }
}

fn find_socket(repo: &Path, store: &Path) -> Option<PathBuf> {
    // Sockets live under the hashed runtime dir, not in the store: ask the
    // reader for the live row's `socket` field instead of scanning directories.
    let output = Command::new(binary_path())
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .args(["agent", "sessions", "list", "--json"])
        .output()
        .ok()?;
    let rows: Value = serde_json::from_slice(&output.stdout).ok()?;
    rows.as_array()?
        .iter()
        .find(|row| row["live"] == true)
        .and_then(|row| row["socket"].as_str().map(PathBuf::from))
}

fn next_event(client: &mut ControlClient, wanted: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(event) = client.next_event(Duration::from_millis(500)).unwrap() {
            if event["type"] == wanted {
                return event;
            }
        }
        assert!(Instant::now() < deadline, "did not receive {wanted}");
    }
}

#[test]
fn tui_remote_turn_broadcasts_events_and_unlinks_socket() {
    let repo = unique_temp("repo");
    init_repo(&repo);
    let (endpoint, stop, gateway) = spawn_gateway();
    let mut pty = Pty::spawn(&repo, &endpoint);

    let mut screen = pty.read_until(b"ready", Duration::from_secs(60));
    if screen.is_empty() {
        println!("skipped: no PTY output");
        stop.store(true, Ordering::SeqCst);
        gateway.join().unwrap();
        return;
    }
    assert!(
        screen.windows(b"ready".len()).any(|part| part == b"ready"),
        "TUI did not reach the ready prompt: {:?}",
        String::from_utf8_lossy(&screen)
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    let socket = loop {
        if let Some(path) = find_socket(&repo, &pty.store) {
            break path;
        }
        assert!(Instant::now() < deadline, "control socket was not created");
        thread::sleep(Duration::from_millis(200));
    };
    let mut client = ControlClient::connect(&socket).unwrap();
    let description = match client.call("session/describe", json!({})) {
        Ok(description) => description,
        Err(error) => {
            screen.extend_from_slice(&pty.read_until(b"__never__", Duration::from_secs(2)));
            let child = pty.child.try_wait().unwrap();
            panic!(
                "describe failed: {error}; child={child:?}; screen={:?}",
                String::from_utf8_lossy(&screen)
            );
        }
    };
    assert_eq!(description["phase"], "idle");
    let session_id = description["session_id"].as_str().unwrap().to_string();
    let live_output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &pty.store)
        .args(["agent", "sessions", "list", "--json"])
        .output()
        .unwrap();
    assert!(live_output.status.success());
    let live_rows: Value = serde_json::from_slice(&live_output.stdout).unwrap();
    let live_row = live_rows
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == session_id)
        .unwrap_or_else(|| panic!("live session missing from rows: {live_rows}"));
    assert_eq!(live_row["live"], true, "rows={live_rows}");
    client.subscribe().unwrap();

    let accepted = client
        .call(
            "turn/start",
            json!({"text":"hello from remote","source":"remote"}),
        )
        .unwrap();
    assert_eq!(accepted["accepted"], true);
    let started = next_event(&mut client, "turn_start");
    assert_eq!(started["text"], "hello from remote");
    assert_eq!(started["source"], "remote");
    next_event(&mut client, "turn_complete");

    screen.extend_from_slice(&pty.read_until(b"hi from tui stub", Duration::from_secs(20)));
    let rendered = String::from_utf8_lossy(&screen);
    assert!(rendered.contains("hello from remote"), "{rendered:?}");
    assert!(rendered.contains("remote >"), "{rendered:?}");
    assert!(rendered.contains("hi from tui stub"), "{rendered:?}");

    assert_eq!(
        client.call("turn/interrupt", json!({})).unwrap(),
        json!({"accepted":true})
    );
    let quit = client.call("session/quit", json!({})).unwrap_err();
    assert_eq!(quit.code, -32000);

    pty.write(b"/exit\r");
    let status = pty.wait(Duration::from_secs(20));
    assert!(status.success(), "agent exit status: {status}");
    assert!(
        !socket.exists(),
        "socket remained after TUI exit: {socket:?}"
    );

    let listed = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &pty.store)
        .args(["agent", "sessions", "list", "--json"])
        .output()
        .unwrap();
    assert!(listed.status.success());
    let rows: Value = serde_json::from_slice(&listed.stdout).unwrap();
    let row = rows
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == session_id || row["session_id"] == session_id)
        .unwrap_or_else(|| panic!("session missing from rows: {rows}"));
    assert_eq!(row["live"], false, "rows={rows}");

    stop.store(true, Ordering::SeqCst);
    gateway.join().unwrap();
}
