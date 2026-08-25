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

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
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

struct FakeProvider {
    data: PathBuf,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for FakeProvider {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Integration-test adapter for the CLI process. It deliberately has no
/// product fallback path: tests publish the same provider identity contract
/// as a real adapter, while this helper mirrors the fixture repository only
/// after WorkspaceCore has created a namespace.
fn spawn_fake_provider(root: &std::path::Path, repo: &std::path::Path) -> FakeProvider {
    use greppy_workspace_core::{
        AdapterKind, ProviderCapabilities, ProviderManifest, ProviderState, WorkspaceCore,
        PROVIDER_PROTOCOL_VERSION,
    };

    let data = root.join("provider-data");
    let mount = root.join("provider-mount");
    std::fs::create_dir_all(mount.join("doctor")).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let manifest = ProviderManifest {
        protocol_version: PROVIDER_PROTOCOL_VERSION,
        adapter_version: "0.3.4-cli-test".into(),
        adapter_kind: AdapterKind::FsKit,
        state: ProviderState::Ready,
        instance_id: "cli-test-provider".into(),
        data_root: data.clone(),
        mount_root: mount.clone(),
        heartbeat_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        capabilities: ProviderCapabilities {
            hard_links: true,
            symbolic_links: true,
            byte_range_locks: true,
            memory_maps: true,
            atomic_rename: true,
            case_preserving: true,
        },
    };
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(data.join("provider.json"), &bytes).unwrap();
    std::fs::write(mount.join(".greppy-provider.json"), bytes).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    let data_thread = data.clone();
    let mount_thread = mount.clone();
    let repo_thread = repo.to_path_buf();
    let handle = thread::spawn(move || {
        let mut core = None;
        let mut mirrored = std::collections::HashSet::new();
        while !stop_thread.load(Ordering::SeqCst) {
            if core.is_none() {
                core = WorkspaceCore::open(data_thread.join("core")).ok();
            }
            if let Some(core) = &core {
                if let Ok(workspaces) = core.list_workspaces() {
                    let active = workspaces
                        .iter()
                        .map(|workspace| workspace.id.clone())
                        .collect::<std::collections::HashSet<_>>();
                    for workspace in &workspaces {
                        let destination = mount_thread.join("workspaces").join(&workspace.id);
                        if !destination.exists() {
                            std::fs::create_dir_all(&destination).unwrap();
                            copy_fixture_tree(&repo_thread, &destination);
                            mirrored.insert(workspace.id.clone());
                        }
                    }
                    let removed = mirrored.difference(&active).cloned().collect::<Vec<_>>();
                    for workspace in removed {
                        let _ = std::fs::remove_dir_all(
                            mount_thread.join("workspaces").join(&workspace),
                        );
                        mirrored.remove(&workspace);
                    }
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
    FakeProvider {
        data,
        stop,
        handle: Some(handle),
    }
}

fn copy_fixture_tree(source: &std::path::Path, destination: &std::path::Path) {
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() == ".git" {
            continue;
        }
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
            copy_fixture_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
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
fn greppy_p_deadline_zero_stops_cleanly_and_delivers_outcome() {
    // --deadline-secs 0 expires at loop start (Instant computed after self-check).
    // The loop must stop with LoopStop::Deadline, print the stopped: line, and
    // still produce the normal clean outcome (exit 0) — never discard work.
    let repo = unique_temp("deadline-repo");
    init_repo(&repo);
    let store = unique_temp("deadline-store");
    let provider_root = unique_temp("deadline-provider");
    let provider = spawn_fake_provider(&provider_root, &repo);

    let (endpoint, stop, handle) = spawn_stub_gateway();

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
            "--max-turns",
            "4",
            "--deadline-secs",
            "0",
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
        stderr.contains("stopped: wall-clock deadline reached (0s) — the result may be incomplete"),
        "expected deadline stop line; stderr={stderr}"
    );
    assert!(
        stdout.contains("no changes proposed."),
        "deadline stop must still deliver the clean/proposal outcome: stdout={stdout}\nstderr={stderr}"
    );

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

/// `index --agent-worktree` warms the portable namespace `-p` will use.
///
/// The warm-up must exercise a provider namespace while publishing a reusable
/// immutable index Base. The temporary namespace is removed afterwards and is
/// never registered as a native Git worktree.
#[test]
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

    drop(provider);
    let _ = std::fs::remove_dir_all(&dir);
}
