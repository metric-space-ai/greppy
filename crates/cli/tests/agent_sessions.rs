//! Integration tests for `greppy agent sessions` readers.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique_temp(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "greppy-agent-sessions-{tag}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&path).unwrap();
    path
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
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

struct Fixture {
    repo: PathBuf,
    store: PathBuf,
    project: String,
    sess1: PathBuf,
    sess2: PathBuf,
    tool_result: String,
}

fn write_session(path: &Path, lines: &[&str]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut body = String::new();
    for line in lines {
        body.push_str(line);
        body.push('\n');
    }
    fs::write(path, body).unwrap();
}

fn setup_fixture() -> Fixture {
    let root = unique_temp("fx");
    let repo = root.join("repo");
    let store = root.join("store");
    fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init"]);
    git(&repo, &["config", "user.email", "sessions@example.com"]);
    git(&repo, &["config", "user.name", "Sessions"]);
    fs::write(repo.join("README.md"), "session fixture\n").unwrap();
    git(&repo, &["add", "README.md"]);
    git(&repo, &["commit", "-m", "init"]);

    let (_, project) = greppy::agent_session_store_identity(&repo);
    let tool_result = "R".repeat(1000);
    let dir = store.join("agent-sessions").join(&project);
    let sess1 = dir.join("sess-1.jsonl");
    let sess2 = dir.join("sess-2.jsonl");
    let tool_json = serde_json::json!({
        "v": 1,
        "type": "message",
        "role": "user",
        "parts": [{
            "kind": "tool_result",
            "text": tool_result,
            "id": "t1",
            "name": "",
            "is_error": false
        }]
    })
    .to_string();

    write_session(
        &sess1,
        &[
            &format!(
                r#"{{"v":1,"type":"meta","id":"sess-1","project":"{project}","title":"first","model":"m1","created_ms":1000,"run_id":"run-1","worktree":"/tmp/wt1","branch":"main","proposal_ref":"refs/greppy/agent/run-1","source":"interactive"}}"#
            ),
            r#"{"v":1,"type":"turn","event":"start","ts_ms":1001,"source":"interactive","prompt":"hello"}"#,
            r#"{"v":1,"type":"message","role":"user","parts":[{"kind":"text","text":"hello","id":"","name":"","is_error":false}]}"#,
            r#"{"v":1,"type":"tool","event":"start","ts_ms":1002,"id":"t1","name":"greppy","summary":"search foo"}"#,
            r#"{"v":1,"type":"message","role":"assistant","parts":[{"kind":"thinking","text":"secret-thought","id":"","name":"","is_error":false},{"kind":"text","text":"working","id":"","name":"","is_error":false}]}"#,
            r#"{"v":1,"type":"tool","event":"finish","ts_ms":1100,"id":"t1","failed":false,"elapsed_ms":98,"preview":"ok"}"#,
            &tool_json,
            r#"{"v":1,"type":"message","role":"assistant","parts":[{"kind":"text","text":"done","id":"","name":"","is_error":false}]}"#,
            r#"{"v":1,"type":"usage","input":10,"output":5,"cache_read":0,"cache_write":0,"turns":1,"stop":"end_turn"}"#,
            r#"{"v":1,"type":"worktree","path":"/tmp/wt1","proposal_ref":"refs/greppy/agent/run-1"}"#,
            r#"{"v":1,"type":"turn","event":"done","ts_ms":1200,"stop":"end_turn","turns":1,"usage":{"input":10,"output":5,"cache_read":0,"cache_write":0}}"#,
            r#"{"v":1,"type":"future","x":1}"#,
        ],
    );
    write_session(
        &sess2,
        &[
            &format!(
                r#"{{"v":1,"type":"meta","id":"sess-2","project":"{project}","title":"second","model":"m2","created_ms":2000,"run_id":"run-2","worktree":"/tmp/wt2","branch":"dev","proposal_ref":"","source":"headless"}}"#
            ),
            r#"{"v":1,"type":"message","role":"user","parts":[{"kind":"text","text":"later","id":"","name":"","is_error":false}]}"#,
            r#"{"v":1,"type":"usage","input":1,"output":1,"cache_read":0,"cache_write":0,"turns":2,"stop":"end_turn"}"#,
            r#"{"v":1,"type":"worktree","path":"/tmp/wt2","proposal_ref":""}"#,
            r#"{"v":1,"type":"title","title":"sess-2-title"}"#,
            r#"{"v":1,"type":"model","model":"m2-final"}"#,
            r#"{"v":1,"type":"future","x":1}"#,
        ],
    );

    Fixture {
        repo,
        store,
        project,
        sess1,
        sess2,
        tool_result,
    }
}

fn run(fx: &Fixture, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(binary_path())
        .args(args)
        .current_dir(&fx.repo)
        .env("GREPPY_STORE_DIR", &fx.store)
        .env_remove("GREPPY_PROJECT_IDENTITY")
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn sessions_list_show_path_and_dispatch() {
    let fx = setup_fixture();

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "list", "--json"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let rows: Vec<serde_json::Value> = serde_json::from_str(&stdout).unwrap();
    assert_eq!(rows.len(), 2, "{stdout}");
    assert_eq!(rows[0]["id"], "sess-2");
    assert_eq!(rows[1]["id"], "sess-1");
    assert_eq!(rows[0]["source"], "headless");
    assert_eq!(rows[1]["source"], "interactive");

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "list"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let ids: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("sess-"))
        .map(|line| line.split_whitespace().next().unwrap())
        .collect();
    assert_eq!(ids, vec!["sess-2", "sess-1"], "{stdout}");

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "show", "sess-1"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("user: hello"), "{stdout}");
    assert!(stdout.contains("assistant: working"), "{stdout}");
    assert!(stdout.contains("tool ▶ greppy search foo"), "{stdout}");
    assert!(stdout.contains("tool ✓ 98 ms"), "{stdout}");
    assert!(!stdout.contains("secret-thought"), "{stdout}");
    assert!(stdout.contains(&"R".repeat(400)), "{stdout}");
    assert!(!stdout.contains(&fx.tool_result), "{stdout}");

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "show", "sess-1", "--full"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains(&fx.tool_result), "{stdout}");

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "path", "sess-1"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let printed = stdout.trim();
    assert_eq!(printed, fx.sess1.canonicalize().unwrap().to_str().unwrap());

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "show", "missing"]);
    assert_eq!(code, 2, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stderr.contains(&format!("no session missing in project {}", fx.project)),
        "stderr={stderr}"
    );

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "path", "sess-"]);
    assert_eq!(code, 2, "stdout={stdout}\nstderr={stderr}");
    assert!(stderr.contains("sess-1"), "stderr={stderr}");
    assert!(stderr.contains("sess-2"), "stderr={stderr}");

    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "--help"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("list"), "{stdout}");
    assert!(stdout.contains("show"), "{stdout}");
    assert!(stdout.contains("tail"), "{stdout}");
    assert!(stdout.contains("path"), "{stdout}");

    let output = Command::new(binary_path())
        .args(["agent"])
        .current_dir(&fx.repo)
        .env("GREPPY_STORE_DIR", &fx.store)
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn agent");
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

    let output = Command::new(binary_path())
        .args(["agent", "some task"])
        .current_dir(&fx.repo)
        .env("GREPPY_STORE_DIR", &fx.store)
        .env_remove("GREPPY_MODEL")
        .env_remove("GREPPY_ENDPOINT")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn agent task");
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
}

#[test]
fn sessions_tail_json_and_follow() {
    let fx = setup_fixture();
    let (code, stdout, stderr) = run(
        &fx,
        &[
            "agent", "sessions", "tail", "sess-2", "--json", "--lines", "3",
        ],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let expected = fs::read_to_string(&fx.sess2).unwrap();
    let last_three: String = expected
        .lines()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|line| format!("{line}\n"))
        .collect();
    assert_eq!(stdout, last_three);

    let mut child = Command::new(binary_path())
        .args([
            "agent", "sessions", "tail", "sess-2", "--json", "--follow", "--lines", "0",
        ])
        .current_dir(&fx.repo)
        .env("GREPPY_STORE_DIR", &fx.store)
        .env_remove("GREPPY_PROJECT_IDENTITY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn tail --follow");
    let stdout = child.stdout.take().expect("stdout");
    let reader = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut handle = stdout;
        handle.read_to_string(&mut buf).ok();
        buf
    });
    std::thread::sleep(Duration::from_millis(400));
    let marker = r#"{"v":1,"type":"title","title":"follow-marker-xyz"}"#;
    let mut file = fs::OpenOptions::new().append(true).open(&fx.sess2).unwrap();
    writeln!(file, "{marker}").unwrap();
    file.flush().unwrap();
    std::thread::sleep(Duration::from_millis(800));
    let _ = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status();
    let status = child.wait().expect("wait tail");
    let out = reader.join().unwrap();
    assert_eq!(status.code(), Some(0), "stdout={out}");
    assert!(
        out.lines().any(|line| line == marker),
        "expected appended line in {out:?}"
    );
}

#[test]
fn sessions_empty_store_is_ok() {
    let root = unique_temp("empty");
    let repo = root.join("repo");
    let store = root.join("store");
    fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init"]);
    let (_, project) = greppy::agent_session_store_identity(&repo);
    let fx = Fixture {
        repo,
        store,
        project: project.clone(),
        sess1: PathBuf::new(),
        sess2: PathBuf::new(),
        tool_result: String::new(),
    };
    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "list"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stderr.contains(&format!("no sessions for project {project}")),
        "stderr={stderr}"
    );
    let (code, stdout, stderr) = run(&fx, &["agent", "sessions", "list", "--json"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(stdout.trim(), "[]");
}
