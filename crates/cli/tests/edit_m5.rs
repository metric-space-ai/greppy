//! End-to-end smoke coverage for the M5 multi-file plan and recover CLI paths.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-edit-m5-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    (repo, base.join("store"))
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn run_with_stdin(repo: &Path, store: &Path, args: &[&str], stdin: &[u8]) -> (i32, String, String) {
    let mut child = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run greppy with stdin");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin)
        .expect("write greppy stdin");
    let output = child.wait_with_output().expect("wait for greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn index_repo(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(repo, store, &["index", "."]);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn plan_json(repo: &Path, operations: &str) -> String {
    format!(
        r#"{{
  "schema_version": "greppy.edit-plan.v1",
  "workspace": {{ "root": "{}" }},
  "operations": [{operations}],
  "publish": {{ "mode": "journal" }}
}}"#,
        repo.to_string_lossy().replace('\\', "\\\\")
    )
}

#[test]
fn plan_edits_two_files_in_one_transaction() {
    let (repo, store) = fresh_workspace("two-files");
    std::fs::write(repo.join("a.txt"), "alpha one\n").unwrap();
    std::fs::write(repo.join("b.txt"), "beta two\n").unwrap();
    let hash_a = sha256_hex(b"alpha one\n");
    let hash_b = sha256_hex(b"beta two\n");
    let ops = format!(
        r#"
    {{ "id": "op-a", "file": "a.txt",
      "selector": {{ "engine": "text", "old_text": "one", "expect": 1 }},
      "action": {{ "type": "replace", "content": "ONE" }},
      "preconditions": {{ "file_sha256": "{hash_a}" }} }},
    {{ "id": "op-b", "file": "b.txt",
      "selector": {{ "engine": "text", "old_text": "two", "expect": 1 }},
      "action": {{ "type": "replace", "content": "TWO" }},
      "preconditions": {{ "file_sha256": "{hash_b}" }} }}
"#
    );
    std::fs::write(repo.join("plan.json"), plan_json(&repo, &ops)).unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["edit", "apply", "--plan", "plan.json"]);
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert_eq!(
        std::fs::read_to_string(repo.join("a.txt")).unwrap(),
        "alpha ONE\n"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("b.txt")).unwrap(),
        "beta TWO\n"
    );
    // Compact certificate on stdout: one report per operation, in plan order.
    assert!(stdout.contains("op-a"), "{stdout}");
    assert!(stdout.contains("op-b"), "{stdout}");
}

#[test]
fn text_cas_stdout_includes_the_resulting_span() {
    let (repo, store) = fresh_workspace("text-cas-result-span");
    std::fs::write(repo.join("note.txt"), "before text\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "edit",
            "text-cas",
            "--file",
            "note.txt",
            "--old",
            "before text",
            "--new",
            "after text",
        ],
    );

    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    let certificate: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(certificate["operations"][0]["result_span"], "after text");
    assert_eq!(
        std::fs::read_to_string(repo.join("note.txt")).unwrap(),
        "after text\n"
    );
}

#[test]
fn aliased_plan_paths_share_overlap_checks_and_exit_13() {
    let (repo, store) = fresh_workspace("aliased-overlap");
    std::fs::write(repo.join("a.txt"), "alpha beta\n").unwrap();
    let hash = sha256_hex(b"alpha beta\n");
    let ops = format!(
        r#"
    {{ "id": "plain-path", "file": "a.txt",
      "selector": {{ "engine": "text", "old_text": "alpha", "expect": 1 }},
      "action": {{ "type": "replace", "content": "ALPHA" }},
      "preconditions": {{ "file_sha256": "{hash}" }} }},
    {{ "id": "dot-path", "file": "./a.txt",
      "selector": {{ "engine": "text", "old_text": "alpha", "expect": 1 }},
      "action": {{ "type": "replace", "content": "OMEGA" }},
      "preconditions": {{ "file_sha256": "{hash}" }} }}
"#
    );
    std::fs::write(repo.join("plan.json"), plan_json(&repo, &ops)).unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["edit", "apply", "--plan", "plan.json"]);

    assert_eq!(code, 13, "stdout:\n{stdout}\nstderr:\n{stderr}");
    let certificate: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(certificate["status"], "invalid-result");
    assert_eq!(certificate["operations"][0]["file"], "a.txt");
    assert_eq!(certificate["operations"][1]["file"], "a.txt");
    assert_eq!(
        std::fs::read_to_string(repo.join("a.txt")).unwrap(),
        "alpha beta\n"
    );
}

#[test]
fn plan_with_stale_precondition_changes_nothing_and_exits_12() {
    let (repo, store) = fresh_workspace("stale");
    std::fs::write(repo.join("a.txt"), "alpha one\n").unwrap();
    std::fs::write(repo.join("b.txt"), "beta two\n").unwrap();
    let stale = "0".repeat(64);
    let hash_a = sha256_hex(b"alpha one\n");
    let ops = format!(
        r#"
    {{ "id": "op-a", "file": "a.txt",
      "selector": {{ "engine": "text", "old_text": "one", "expect": 1 }},
      "action": {{ "type": "replace", "content": "ONE" }},
      "preconditions": {{ "file_sha256": "{hash_a}" }} }},
    {{ "id": "op-b", "file": "b.txt",
      "selector": {{ "engine": "text", "old_text": "two", "expect": 1 }},
      "action": {{ "type": "replace", "content": "TWO" }},
      "preconditions": {{ "file_sha256": "{stale}" }} }}
"#
    );
    std::fs::write(repo.join("plan.json"), plan_json(&repo, &ops)).unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["edit", "apply", "--plan", "plan.json"]);
    assert_eq!(code, 12, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert_eq!(
        std::fs::read_to_string(repo.join("a.txt")).unwrap(),
        "alpha one\n",
        "no file may change when any operation is stale"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("b.txt")).unwrap(),
        "beta two\n"
    );
}

#[test]
fn recover_reports_clean_workspace_and_writes_report_file() {
    let (repo, store) = fresh_workspace("recover");
    std::fs::write(repo.join("a.txt"), "alpha\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["edit", "recover", "--report", "recovery.json"],
    );
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("nothing to recover"), "{stdout}");
    let report = std::fs::read_to_string(repo.join("recovery.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&report).unwrap();
    assert_eq!(parsed["found_journal"], false, "{report}");
    assert_eq!(parsed["action"], "nothing-to-recover", "{report}");
}

#[test]
fn replace_body_ignores_stdin_when_content_file_given() {
    // A real --content-file path is used verbatim; stdin is never read, so an
    // open or non-empty stdin can neither block (the earlier hang regression)
    // nor spuriously reject. The content file wins; the stdin bytes are ignored.
    let (repo, store) = fresh_workspace("ignored-stdin");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn target() -> i32 { 1 }\n").unwrap();
    index_repo(&repo, &store);
    std::fs::write(repo.join("body.rs"), "2\n").unwrap();

    let (code, stdout, stderr) = run_with_stdin(
        &repo,
        &store,
        &[
            "edit",
            "replace-body",
            "--symbol",
            "target",
            "--content-file",
            "body.rs",
        ],
        b"{ 3 }\n",
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let src = std::fs::read_to_string(repo.join("src/lib.rs")).unwrap();
    // content file "2" became the body; the stdin body "{ 3 }" was never applied.
    assert!(src.contains("{2") || src.contains("{ 2"), "content file must be used: {src}");
    assert!(!src.contains("{ 3 }") && !src.contains("{3"), "stdin must be ignored: {src}");
}

#[test]
fn replace_body_rejects_target_file_as_content_before_planning() {
    let (repo, store) = fresh_workspace("same-content-target");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    let original = "pub fn target() -> i32 { 1 }\n";
    std::fs::write(repo.join("src/lib.rs"), original).unwrap();
    index_repo(&repo, &store);

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-body",
            "--symbol",
            "target",
            "--content-file",
            "src/lib.rs",
        ],
    );

    assert_eq!(code, 20, "stdout={stdout}\nstderr={stderr}");
    let output = format!("{stdout}\n{stderr}");
    assert!(output.contains("status: invalid-request"), "{output}");
    assert!(
        output.contains("must contain only the new content"),
        "{output}"
    );
    assert!(output.contains("--content-file -"), "{output}");
    assert_eq!(
        std::fs::read_to_string(repo.join("src/lib.rs")).unwrap(),
        original
    );
}

#[test]
fn replace_body_reads_content_file_dash_from_stdin() {
    let (repo, store) = fresh_workspace("stdin-content");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn target() -> i32 { 1 }\n").unwrap();
    index_repo(&repo, &store);

    let (code, stdout, stderr) = run_with_stdin(
        &repo,
        &store,
        &[
            "edit",
            "replace-body",
            "--symbol",
            "target",
            "--content-file",
            "-",
        ],
        b"{ 2 }",
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("\"status\": \"applied\""), "{stdout}");
    assert_eq!(
        std::fs::read_to_string(repo.join("src/lib.rs")).unwrap(),
        "pub fn target() -> i32 { 2 }\n"
    );
}
