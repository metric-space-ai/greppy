//! Agent-facing edit-plan shorthand, templates, and complete validation errors.

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
        "greppy-edit-plan-usability-{tag}-{}-{n}",
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
        .stdin(Stdio::null())
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn apply_plan(repo: &Path, store: &Path, plan: &str) -> (i32, String, String) {
    std::fs::write(repo.join("plan.json"), plan).unwrap();
    run(repo, store, &["edit", "apply", "--plan", "plan.json"])
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[test]
fn minimal_text_plan_defaults_schema_workspace_id_selector_and_publish() {
    let (repo, store) = fresh_workspace("minimal");
    std::fs::write(repo.join("a.txt"), "before foo after\n").unwrap();

    let (code, stdout, stderr) = apply_plan(
        &repo,
        &store,
        r#"{"operations":[{"file":"a.txt","old":"foo","new":"bar"}]}"#,
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        std::fs::read_to_string(repo.join("a.txt")).unwrap(),
        "before bar after\n"
    );
    let certificate: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(certificate["operations"][0]["id"], "op-1");
}

#[test]
fn top_level_ops_alias_is_accepted() {
    let (repo, store) = fresh_workspace("ops-alias");
    std::fs::write(repo.join("a.txt"), "alpha\n").unwrap();

    let (code, stdout, stderr) = apply_plan(
        &repo,
        &store,
        r#"{"ops":[{"file":"a.txt","old":"alpha","new":"omega"}]}"#,
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        std::fs::read_to_string(repo.join("a.txt")).unwrap(),
        "omega\n"
    );
}

#[test]
fn text_cas_plan_template_round_trips_without_edits() {
    let (repo, store) = fresh_workspace("template");
    let (code, template, stderr) = run(
        &repo,
        &store,
        &["edit", "plan-template", "--op", "text-cas"],
    );
    assert_eq!(code, 0, "template stderr={stderr}");

    let parsed: serde_json::Value = serde_json::from_str(&template).unwrap();
    let operation = &parsed["operations"][0];
    let file = operation["file"].as_str().unwrap();
    let old = operation["old"].as_str().unwrap();
    let target = repo.join(file);
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    std::fs::write(&target, format!("{old}\n")).unwrap();

    let (code, stdout, stderr) = apply_plan(&repo, &store, &template);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        std::fs::read_to_string(target).unwrap(),
        format!("{}\n", operation["new"].as_str().unwrap())
    );
}

#[test]
fn broken_plan_reports_all_missing_fields_and_a_complete_example() {
    let (repo, store) = fresh_workspace("broken");

    let (code, stdout, stderr) = apply_plan(&repo, &store, r#"{"operations":[{}]}"#);

    assert_eq!(code, 20, "stdout={stdout}\nstderr={stderr}");
    for field in ["schema_version", "publish", "file", "selector", "action"] {
        assert!(stderr.contains(field), "missing {field} in:\n{stderr}");
    }
    assert!(stderr.contains("operations"), "{stderr}");
    assert!(stderr.contains("minimal complete example:"), "{stderr}");
    assert!(stderr.contains(r#""old": "OLD""#), "{stderr}");
    assert!(stderr.contains("allowed optional fields:"), "{stderr}");
}

#[test]
fn canonical_long_form_remains_valid() {
    let (repo, store) = fresh_workspace("canonical");
    let original = b"alpha one\n";
    std::fs::write(repo.join("a.txt"), original).unwrap();
    let plan = serde_json::json!({
        "schema_version": "greppy.edit-plan.v1",
        "operations": [{
            "id": "canonical-op",
            "file": "a.txt",
            "selector": {"engine": "text", "old_text": "one", "expect": 1},
            "action": {"type": "replace", "content": "two"},
            "preconditions": {"file_sha256": sha256_hex(original)}
        }],
        "publish": {"mode": "journal"}
    });

    let (code, stdout, stderr) = apply_plan(&repo, &store, &plan.to_string());

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        std::fs::read_to_string(repo.join("a.txt")).unwrap(),
        "alpha two\n"
    );
}

#[test]
fn replace_body_shorthand_resolves_symbol_and_inline_body() {
    let (repo, store) = fresh_workspace("replace-body");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn target() -> i32 { 1 }\n").unwrap();
    let (code, stdout, stderr) = run(&repo, &store, &["index", "."]);
    assert_eq!(code, 0, "index stdout={stdout}\nindex stderr={stderr}");

    let plan = serde_json::json!({
        "operations": [{
            "file": "src/lib.rs",
            "symbol": "target",
            "new_body": "{ 2 }"
        }]
    });
    let (code, stdout, stderr) = apply_plan(&repo, &store, &plan.to_string());

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        std::fs::read_to_string(repo.join("src/lib.rs")).unwrap(),
        "pub fn target() -> i32 { 2 }\n"
    );
}
