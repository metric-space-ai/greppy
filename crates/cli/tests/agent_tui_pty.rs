//! PTY and routing tests for the interactive agent TUI.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique_temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "greppy-agent-tui-{tag}-{}-{}",
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

fn spawn_stub_gateway(delay_ms: u64) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.set_nonblocking(true).expect("nonblocking");
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
                        let body = r#"{"data":[{"id":"test"},{"id":"other-model"}]}"#;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(resp.as_bytes());
                    } else if first_line.starts_with("POST /v1/messages") {
                        if delay_ms > 0 {
                            thread::sleep(Duration::from_millis(delay_ms));
                        }
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
    (format!("http://127.0.0.1:{port}"), stop, handle)
}

#[test]
fn greppy_agent_help_documents_commands() {
    let output = Command::new(binary_path())
        .args(["agent", "--help"])
        .env_remove("GREPPY_MODEL")
        .output()
        .expect("help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("/model"), "{stdout}");
    assert!(stdout.contains("--continue"), "{stdout}");
    assert!(stdout.contains("Ctrl+C"), "{stdout}");
}

#[test]
fn greppy_agent_refuses_nontty() {
    let output = Command::new(binary_path())
        .args(["agent", "--model", "test"])
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(2),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("TTY") || stderr.contains("tty"),
        "stderr={stderr}"
    );
    assert!(
        !stdout.contains("\x1b["),
        "must not emit control sequences to redirected stdout: {stdout:?}"
    );
}

#[cfg(unix)]
mod pty {
    use super::*;
    use std::fs::File;
    use std::os::fd::FromRawFd;
    use std::os::unix::io::AsRawFd;
    use std::process::Child;

    // Each case launches a real agent, indexer and PTY. Running those startup
    // paths concurrently makes timing assertions measure machine contention
    // instead of the UI contract.
    static PTY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Pty {
        master: File,
        child: Child,
    }

    impl Pty {
        fn spawn(repo: &std::path::Path, endpoint: &str, extra: &[&str], plain: bool) -> Self {
            unsafe {
                let mut master = 0;
                let mut slave = 0;
                let mut ws = libc::winsize {
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
                        &mut ws
                    ),
                    0
                );
                let slave_in = libc::dup(slave);
                let slave_out = libc::dup(slave);
                let slave_err = libc::dup(slave);
                libc::close(slave);
                let store = super::unique_temp("pty-store");
                let mut cmd = Command::new(super::binary_path());
                cmd.current_dir(repo)
                    .env("GREPPY_STORE_DIR", &store)
                    .env("GREPPY_CONFIG_DIR", store.join("config"))
                    .env("GREPPY_TEST_SKIP_INFERENCE", "1")
                    .env("GREPPY_ASCII", "1")
                    .env("TERM", "xterm")
                    .env_remove("GREPPY_MODEL")
                    .env("GREPPY_ENDPOINT", endpoint);
                if plain {
                    cmd.arg("agent");
                } else {
                    cmd.args([
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
                    ]);
                }
                cmd.args(extra)
                    .stdin(Stdio::from_raw_fd(slave_in))
                    .stdout(Stdio::from_raw_fd(slave_out))
                    .stderr(Stdio::from_raw_fd(slave_err));
                let child = cmd.spawn().expect("spawn greppy agent");
                let master = File::from_raw_fd(master);
                let _ = libc::fcntl(master.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK);
                Self { master, child }
            }
        }

        fn write_all(&mut self, bytes: &[u8]) {
            let _ = self.master.write_all(bytes);
            let _ = self.master.flush();
        }

        fn read_for(&mut self, dur: Duration) -> Vec<u8> {
            let deadline = std::time::Instant::now() + dur;
            let mut out = Vec::new();
            let mut buf = [0u8; 4096];
            while std::time::Instant::now() < deadline {
                match self.master.read(&mut buf) {
                    Ok(0) => {
                        if self.child.try_wait().ok().flatten().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    Ok(n) => out.extend_from_slice(&buf[..n]),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(e) if e.raw_os_error() == Some(libc::EIO) => {
                        if self.child.try_wait().ok().flatten().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => {
                        if self.child.try_wait().ok().flatten().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                }
                if let Ok(Some(_)) = self.child.try_wait() {
                    let _ = self
                        .master
                        .read(&mut buf)
                        .map(|n| out.extend_from_slice(&buf[..n]));
                    break;
                }
            }
            out
        }

        fn read_until(&mut self, needle: &[u8], dur: Duration) -> Vec<u8> {
            let deadline = std::time::Instant::now() + dur;
            let mut out = Vec::new();
            let mut buf = [0u8; 4096];
            while std::time::Instant::now() < deadline {
                match self.master.read(&mut buf) {
                    Ok(0) => {
                        if self.child.try_wait().ok().flatten().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    Ok(n) => {
                        out.extend_from_slice(&buf[..n]);
                        if out.windows(needle.len()).any(|window| window == needle) {
                            break;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(e) if e.raw_os_error() == Some(libc::EIO) => {
                        if self.child.try_wait().ok().flatten().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => {
                        if self.child.try_wait().ok().flatten().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                }
                if let Ok(Some(_)) = self.child.try_wait() {
                    break;
                }
            }
            out
        }

        fn wait(mut self, dur: Duration) -> (std::process::ExitStatus, Vec<u8>) {
            let bytes = self.read_for(dur);
            let status = match self.child.try_wait() {
                Ok(Some(status)) => status,
                _ => {
                    let _ = self.child.kill();
                    self.child.wait().expect("wait")
                }
            };
            (status, bytes)
        }

        fn resize(&mut self, cols: u16, rows: u16) {
            let ws = libc::winsize {
                ws_row: rows,
                ws_col: cols,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            unsafe {
                libc::ioctl(self.master.as_raw_fd(), libc::TIOCSWINSZ, &ws);
                if let Ok(id) = self.child.id().try_into() {
                    libc::kill(id, libc::SIGWINCH);
                }
            }
        }
    }

    impl Drop for Pty {
        fn drop(&mut self) {
            let _ = self.child.kill();
        }
    }

    #[test]
    fn pty_idle_exit_restores_terminal_and_publishes_proposal() {
        let _serial = PTY_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = super::unique_temp("pty-repo");
        super::init_repo(&repo);
        let (endpoint, stop, handle) = super::spawn_stub_gateway(0);
        let mut pty = Pty::spawn(&repo, &endpoint, &[], false);
        let mut bytes = pty.read_until(b"\x1b[?1049h", Duration::from_secs(15));
        bytes.extend_from_slice(&pty.read_until(b"prompt", Duration::from_secs(60)));
        assert!(
            bytes
                .windows(b"prompt".len())
                .any(|window| window == b"prompt"),
            "agent never reached the ready prompt: {:?}",
            String::from_utf8_lossy(&bytes)
        );
        pty.write_all(b"/setup\r");
        let setup = pty.read_until(b"Acceleration", Duration::from_secs(5));
        assert!(
            setup
                .windows(b"Gateway".len())
                .any(|window| window == b"Gateway")
                && setup
                    .windows(b"Language".len())
                    .any(|window| window == b"Language")
                && setup
                    .windows(b"Acceleration".len())
                    .any(|window| window == b"Acceleration"),
            "setup menu did not open: {:?}",
            String::from_utf8_lossy(&setup)
        );
        bytes.extend_from_slice(&setup);
        for _ in 0..8 {
            pty.write_all(b"\x1b[B");
        }
        pty.write_all(b"\r");
        thread::sleep(Duration::from_millis(300));
        pty.write_all(b"/exit\r");
        let (status, tail) = pty.wait(Duration::from_secs(20));
        bytes.extend_from_slice(&tail);
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        let text = String::from_utf8_lossy(&bytes);
        assert!(status.success(), "status={status:?} output={text:?}");
        assert!(
            text.contains("\x1b[?1049h") || text.contains("greppy"),
            "expected alt-screen or UI: {text:?}"
        );
        assert!(
            text.contains("\x1b[?1049l") || text.contains("no changes proposed"),
            "expected restore or proposal: {text:?}"
        );
        assert!(
            text.contains("no changes proposed") || text.contains("proposal saved"),
            "expected proposal outcome in scrollback: {text:?}"
        );
        let _ = std::fs::remove_dir_all(repo);
    }

    #[test]
    fn continue_without_session_restores_bootstrap_terminal() {
        let _serial = PTY_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = super::unique_temp("pty-continue-empty");
        super::init_repo(&repo);
        let (endpoint, stop, handle) = super::spawn_stub_gateway(0);
        let pty = Pty::spawn(&repo, &endpoint, &["--continue"], false);
        let (status, bytes) = pty.wait(Duration::from_secs(20));
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        let text = String::from_utf8_lossy(&bytes);
        assert!(!status.success(), "unexpected success: {text:?}");
        assert!(text.contains("no previous interactive session"), "{text:?}");
        assert!(
            text.contains("\x1b[?1049l"),
            "bootstrap terminal was not restored: {text:?}"
        );
        let _ = std::fs::remove_dir_all(repo);
    }

    #[test]
    fn pty_streaming_follow_up_resize_and_cancel() {
        let _serial = PTY_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = super::unique_temp("pty-stream");
        super::init_repo(&repo);
        let (endpoint, stop, handle) = super::spawn_stub_gateway(200);
        let mut pty = Pty::spawn(&repo, &endpoint, &["say hi"], false);
        let mut bytes = pty.read_until(b"\x1b[?1049h", Duration::from_secs(15));
        bytes.extend_from_slice(&pty.read_until(b"stub", Duration::from_secs(60)));
        assert!(
            bytes.windows(b"stub".len()).any(|window| window == b"stub"),
            "initial task was not submitted after startup: {:?}",
            String::from_utf8_lossy(&bytes)
        );
        pty.resize(60, 18);
        thread::sleep(Duration::from_millis(200));
        pty.resize(80, 24);
        pty.write_all(b"follow up\r");
        thread::sleep(Duration::from_millis(300));
        pty.write_all(&[0x03]); // Ctrl+C
        thread::sleep(Duration::from_millis(200));
        pty.write_all(b"/exit\r");
        let (status, tail) = pty.wait(Duration::from_secs(25));
        bytes.extend_from_slice(&tail);
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            status.success() || status.code() == Some(0) || status.code() == Some(3),
            "status={status:?} output={text:?}"
        );
        assert!(
            text.contains("\x1b[?1049l")
                || text.contains("no changes proposed")
                || text.contains("proposal"),
            "terminal should restore: {text:?}"
        );
        let _ = std::fs::remove_dir_all(repo);
    }

    #[test]
    fn plain_agent_command_shows_startup_ui_immediately() {
        let _serial = PTY_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = super::unique_temp("pty-plain-start");
        super::init_repo(&repo);
        let (endpoint, stop, handle) = super::spawn_stub_gateway(0);
        let mut pty = Pty::spawn(&repo, &endpoint, &[], true);
        let mut bytes = pty.read_until(b"\x1b[?1049h", Duration::from_secs(30));
        if !bytes
            .windows(b"Creating agent workspace".len())
            .any(|window| window == b"Creating agent workspace")
        {
            bytes.extend_from_slice(
                &pty.read_until(b"Creating agent workspace", Duration::from_secs(5)),
            );
        }
        let text = String::from_utf8_lossy(&bytes);
        let expected_header = format!("greppy agent {}", env!("CARGO_PKG_VERSION"));
        assert!(
            text.contains("\x1b[?1049h")
                && text.contains(expected_header.as_str())
                && text.contains("Creating agent workspace")
                && text.contains("prompt")
                && !text.contains("[ OK ]"),
            "plain `greppy agent` must enter the harness immediately: {text:?}"
        );
        pty.write_all(b"queued during startup\r");
        let queued = pty.read_until(b"one-time code analysis", Duration::from_secs(2));
        assert!(
            queued
                .windows(b"one-time code analysis".len())
                .any(|window| window == b"one-time code analysis"),
            "startup harness did not accept and queue input: {:?}",
            String::from_utf8_lossy(&queued)
        );
        pty.write_all(&[0x03]);
        let (status, tail) = pty.wait(Duration::from_secs(20));
        assert_eq!(status.code(), Some(130), "status={status:?}");
        assert!(
            tail.windows(b"\x1b[?1049l".len())
                .any(|window| window == b"\x1b[?1049l"),
            "Ctrl-C during startup must restore the terminal: {:?}",
            String::from_utf8_lossy(&tail)
        );
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        let _ = std::fs::remove_dir_all(repo);
    }

    #[test]
    fn cold_plain_agent_reaches_ready_while_index_runs_in_background() {
        let _serial = PTY_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = super::unique_temp("pty-cold-start");
        super::init_repo(&repo);
        let (endpoint, stop, handle) = super::spawn_stub_gateway(0);
        let mut pty = Pty::spawn(&repo, &endpoint, &[], true);
        let mut bytes = pty.read_until(b"\x1b[?1049h", Duration::from_secs(35));
        bytes.extend_from_slice(&pty.read_until(b"ready", Duration::from_secs(20)));
        let text = String::from_utf8_lossy(&bytes);
        let child_state = pty.child.try_wait().expect("inspect agent process");
        assert!(
            text.contains("ready") && text.contains("greppy agent"),
            "cold flagless agent never reached ready (child={child_state:?}): {text:?}"
        );
        pty.write_all(b"/exit\r");
        let (status, tail) = pty.wait(Duration::from_secs(20));
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert!(status.success(), "status={status:?} tail={tail:?}");
        assert!(
            tail.windows(b"\x1b[?1049l".len())
                .any(|window| window == b"\x1b[?1049l"),
            "terminal was not restored: {:?}",
            String::from_utf8_lossy(&tail)
        );
        let _ = std::fs::remove_dir_all(repo);
    }

    #[test]
    fn pasted_gateway_url_and_enter_are_processed_without_input_lag() {
        let _serial = PTY_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = super::unique_temp("pty-input-latency");
        super::init_repo(&repo);
        let (endpoint, stop, handle) = super::spawn_stub_gateway(0);
        let mut pty = Pty::spawn(&repo, &endpoint, &[], false);
        let mut ready = pty.read_until(b"\x1b[?1049h", Duration::from_secs(35));
        ready.extend_from_slice(&pty.read_until(b"ready", Duration::from_secs(20)));
        assert!(
            ready
                .windows(b"ready".len())
                .any(|window| window == b"ready"),
            "agent never reached ready: {:?}",
            String::from_utf8_lossy(&ready)
        );

        pty.write_all(b"/setup\r");
        let setup = pty.read_until(b"Acceleration", Duration::from_secs(5));
        assert!(
            setup
                .windows(b"Acceleration".len())
                .any(|window| window == b"Acceleration"),
            "setup did not open: {:?}",
            String::from_utf8_lossy(&setup)
        );
        pty.write_all(b"\r");
        let gateway = pty.read_until(b"Gateway URL", Duration::from_secs(2));
        assert!(
            gateway
                .windows(b"Gateway URL".len())
                .any(|window| window == b"Gateway URL"),
            "gateway editor did not open: {:?}",
            String::from_utf8_lossy(&gateway)
        );

        pty.write_all(&[0x15]); // Ctrl-U clears the prefilled endpoint.
        let pasted = b"http://localhost:1";
        let paste_started = std::time::Instant::now();
        pty.write_all(pasted);
        let rendered = pty.read_until(b"localhost", Duration::from_secs(2));
        assert!(
            rendered
                .windows(b"localhost".len())
                .any(|window| window == b"localhost"),
            "pasted URL was not rendered: {:?}",
            String::from_utf8_lossy(&rendered)
        );
        assert!(
            paste_started.elapsed() < Duration::from_secs(1),
            "pasted URL took {:?} to render",
            paste_started.elapsed()
        );

        let enter_started = std::time::Instant::now();
        pty.write_all(b"\r");
        let rejected = pty.read_until(b"unreachable", Duration::from_secs(3));
        assert!(
            rejected
                .windows(b"unreachable".len())
                .any(|window| window == b"unreachable"),
            "Enter did not submit the URL: {:?}",
            String::from_utf8_lossy(&rejected)
        );
        assert!(
            enter_started.elapsed() < Duration::from_secs(2),
            "Enter took {:?} to submit",
            enter_started.elapsed()
        );

        pty.write_all(&[0x03]);
        let (_status, tail) = pty.wait(Duration::from_secs(10));
        assert!(
            tail.windows(b"\x1b[?1049l".len())
                .any(|window| window == b"\x1b[?1049l"),
            "terminal was not restored: {:?}",
            String::from_utf8_lossy(&tail)
        );
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        let _ = std::fs::remove_dir_all(repo);
    }
}

#[test]
fn greppy_p_still_streams_to_stdout() {
    // Regression: interactive TUI must not change `greppy -p` routing.
    let repo = unique_temp("p-regression");
    init_repo(&repo);
    let store = unique_temp("p-regression-store");
    let (endpoint, stop, handle) = spawn_stub_gateway(0);
    let output = Command::new(binary_path())
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
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
        .output()
        .expect("spawn -p");
    stop.store(true, Ordering::SeqCst);
    let _ = handle.join();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("hi from stub"), "{stdout}");
    assert!(stdout.contains("no changes proposed"), "{stdout}");
    let _ = std::fs::remove_dir_all(repo);
    let _ = std::fs::remove_dir_all(store);
}
