//! End-to-end smoke coverage for the agent-facing edit verbs.

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

static COUNTER: AtomicU32 = AtomicU32::new(0);

const RUST_FIXTURE: &str = r#"pub const ANSWER: i32 = 42;

pub fn greet(name: &str) -> String {
    format!("hello {}", name)
}

pub fn combine(a: i32, b: i32) -> i32 {
    a + b
}

pub fn caller() -> i32 {
    combine(1, 2)
}
"#;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

struct Fixture {
    base: PathBuf,
    repo: PathBuf,
    store: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let base = std::env::temp_dir().join(format!(
            "greppy-cli-edit-smoke-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let store = base.join("store");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(repo.join("src/lib.rs"), RUST_FIXTURE).unwrap();
        let fixture = Self { base, repo, store };
        let output = fixture.run(&["index", "."]);
        assert_success("index fixture", &output);
        fixture
    }

    fn command(&self) -> Command {
        let mut command = Command::new(bin());
        command
            .current_dir(&self.repo)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1");
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command()
            .args(args)
            .stdin(Stdio::null())
            .output()
            .expect("run greppy")
    }

    fn scratch(&self, name: &str, content: &str) -> PathBuf {
        let path = self.base.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn assert_success(action: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{action} failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("\"status\": \"applied\"")
            || action == "index fixture",
        "{action} did not report applied:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn replace_body_accepts_natural_inner_body() {
    let fixture = Fixture::new("replace-body-inner");
    let body = fixture.scratch("body.rs", r#"format!("hey {}", name)"#);

    let output = fixture.run(&[
        "edit",
        "replace-body",
        "--symbol",
        "greet",
        "--content-file",
        body.to_str().unwrap(),
    ]);

    assert_success("replace-body with inner body", &output);
    let changed = std::fs::read_to_string(fixture.repo.join("src/lib.rs")).unwrap();
    assert!(
        changed.contains(r#"{format!("hey {}", name)}"#),
        "{changed}"
    );
}

#[test]
fn change_signature_accepts_inline_json_spec() {
    let fixture = Fixture::new("change-signature-inline");
    let spec = r#"{"old_parameters":"(a: i32, b: i32)","new_parameters":"(b: i32, a: i32)","expect_call_sites":1}"#;

    let output = fixture.run(&[
        "edit",
        "change-signature",
        "--symbol",
        "combine",
        "--spec",
        spec,
    ]);

    assert_success("change-signature with inline JSON", &output);
    let changed = std::fs::read_to_string(fixture.repo.join("src/lib.rs")).unwrap();
    assert!(changed.contains("combine(b: i32, a: i32)"), "{changed}");
    assert!(changed.contains("combine(2, 1)"), "{changed}");
}

#[test]
fn replace_span_symbol_error_teaches_handle_workflow() {
    let fixture = Fixture::new("replace-span-symbol-error");
    let source = fixture.scratch("replacement.rs", "pub fn greet() {}\n");

    let output = fixture.run(&[
        "edit",
        "replace-span",
        "--symbol",
        "greet",
        "--source-file",
        source.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(64));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("greppy read SYM --handle"), "{stdout}");
    assert!(stdout.contains("--target <HANDLE>"), "{stdout}");
}

#[test]
fn replace_body_with_content_file_does_not_wait_for_open_stdin_pipe() {
    let fixture = Fixture::new("replace-body-open-stdin");
    let body = fixture.scratch("body.rs", r#"{ format!("hey {}", name) }"#);

    let mut feeder = Command::new("sh")
        .args(["-c", "sleep 30"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("start open stdin feeder");
    let open_pipe = feeder.stdout.take().expect("feeder stdout");
    let mut child = fixture
        .command()
        .args([
            "edit",
            "replace-body",
            "--symbol",
            "greet",
            "--content-file",
            body.to_str().unwrap(),
        ])
        .stdin(Stdio::from(open_pipe))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run replace-body with open stdin");

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().expect("poll replace-body").is_some() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = feeder.kill();
            panic!("replace-body waited for EOF on an ignored stdin pipe");
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    let output = child
        .wait_with_output()
        .expect("collect replace-body output");
    let _ = feeder.kill();
    let _ = feeder.wait();

    assert_success("replace-body with open stdin", &output);
}
