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
fn replace_span_uses_read_handle() {
    let fixture = Fixture::new("replace-span");
    let read = fixture.run(&["read", "greet", "--handle", "--json"]);
    assert!(
        read.status.success(),
        "read failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&read.stdout),
        String::from_utf8_lossy(&read.stderr)
    );
    let read_json: serde_json::Value = serde_json::from_slice(&read.stdout).unwrap();
    let handle = read_json["handle"].as_str().expect("read handle");
    let source = fixture.scratch(
        "replacement.rs",
        r#"pub fn greet(name: &str) -> String {
    format!("welcome {}", name)
}
"#,
    );

    let output = fixture.run(&[
        "edit",
        "replace-span",
        "--target",
        handle,
        "--source-file",
        source.to_str().unwrap(),
    ]);

    assert_success("replace-span", &output);
    let changed = std::fs::read_to_string(fixture.repo.join("src/lib.rs")).unwrap();
    assert!(
        changed.contains(r#"format!("welcome {}", name)"#),
        "{changed}"
    );
}

#[test]
fn text_cas_replaces_exact_text() {
    let fixture = Fixture::new("text-cas");

    let output = fixture.run(&[
        "edit",
        "text-cas",
        "--file",
        "src/lib.rs",
        "--old",
        "42",
        "--new",
        "43",
    ]);

    assert_success("text-cas", &output);
    let changed = std::fs::read_to_string(fixture.repo.join("src/lib.rs")).unwrap();
    assert!(changed.contains("pub const ANSWER: i32 = 43;"), "{changed}");
}

#[test]
fn text_cas_reports_ambiguous_cardinality_then_accepts_explicit_expect() {
    let fixture = Fixture::new("text-cas-ambiguous");
    let file = fixture.repo.join("repeated.txt");
    let original = "OLD\nOLD\nOLD\n";
    std::fs::write(&file, original).unwrap();

    let ambiguous = fixture.run(&[
        "edit",
        "text-cas",
        "--file",
        "repeated.txt",
        "--old",
        "OLD",
        "--new",
        "NEW",
    ]);

    assert_eq!(ambiguous.status.code(), Some(11));
    let stdout = String::from_utf8_lossy(&ambiguous.stdout);
    assert!(stdout.contains("\"status\": \"ambiguous\""), "{stdout}");
    assert!(stdout.contains("`OLD` occurs 3 times"), "{stdout}");
    assert!(stdout.contains("expected 1"), "{stdout}");
    assert!(stdout.contains("`--expect 3`"), "{stdout}");
    assert!(stdout.contains("rename-symbol"), "{stdout}");
    assert!(stdout.contains("rename-call"), "{stdout}");
    assert_eq!(std::fs::read_to_string(&file).unwrap(), original);

    let applied = fixture.run(&[
        "edit",
        "text-cas",
        "--file",
        "repeated.txt",
        "--old",
        "OLD",
        "--new",
        "NEW",
        "--expect",
        "3",
    ]);

    assert_success("text-cas --expect 3", &applied);
    assert_eq!(std::fs::read_to_string(file).unwrap(), "NEW\nNEW\nNEW\n");
}

#[test]
fn insert_after_adds_top_level_definition() {
    let fixture = Fixture::new("insert-after");
    let content = fixture.scratch(
        "insert.rs",
        "pub fn inserted() -> i32 {\n    ANSWER + 1\n}\n",
    );

    let output = fixture.run(&[
        "edit",
        "insert-after",
        "--symbol",
        "greet",
        "--content-file",
        content.to_str().unwrap(),
    ]);

    assert_success("insert-after", &output);
    let changed = std::fs::read_to_string(fixture.repo.join("src/lib.rs")).unwrap();
    assert!(changed.contains("pub fn inserted() -> i32"), "{changed}");
}

#[test]
fn rename_symbol_updates_definition_and_call() {
    let fixture = Fixture::new("rename-symbol");

    let output = fixture.run(&[
        "edit",
        "rename-symbol",
        "--symbol",
        "combine",
        "--new-name",
        "merge_numbers",
    ]);

    assert_success("rename-symbol", &output);
    let changed = std::fs::read_to_string(fixture.repo.join("src/lib.rs")).unwrap();
    assert!(
        changed.contains("pub fn merge_numbers(a: i32, b: i32)"),
        "{changed}"
    );
    assert!(changed.contains("merge_numbers(1, 2)"), "{changed}");
}

#[test]
fn ensure_import_adds_rust_use() {
    let fixture = Fixture::new("ensure-import");

    let output = fixture.run(&[
        "edit",
        "ensure-import",
        "--file",
        "src/lib.rs",
        "--module",
        "std::collections",
        "--name",
        "HashMap",
    ]);

    assert_success("ensure-import", &output);
    let changed = std::fs::read_to_string(fixture.repo.join("src/lib.rs")).unwrap();
    assert!(
        changed.starts_with("use std::collections::HashMap;\n"),
        "{changed}"
    );
}

#[test]
fn ensure_import_adds_go_module_inside_existing_group() {
    let fixture = Fixture::new("ensure-import-go-group");
    let go_file = fixture.repo.join("main.go");
    std::fs::write(
        &go_file,
        "package main\n\nimport (\n\t\"fmt\"\n)\n\nfunc main() { fmt.Println(\"ok\") }\n",
    )
    .unwrap();

    let output = fixture.run(&[
        "edit",
        "ensure-import",
        "--file",
        "main.go",
        "--module",
        "time",
        "--name",
        "time",
    ]);

    assert_success("ensure-import Go group", &output);
    let changed = std::fs::read_to_string(go_file).unwrap();
    assert!(
        changed.contains("import (\n\t\"fmt\"\n\t\"time\"\n)"),
        "{changed}"
    );
    assert!(!changed.contains(")\nimport \"time\""), "{changed}");
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
