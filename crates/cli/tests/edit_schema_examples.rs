//! Invalid edit schemas must print complete, embedded retry examples.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

const PLAN_EXAMPLE: &str = include_str!("../../../docs/contracts/edit-plan.minimal.json");
const SIGNATURE_EXAMPLE: &str =
    include_str!("../../../docs/contracts/change-signature-spec.minimal.json");

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-edit-schema-{tag}-{}-{n}",
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

#[test]
fn apply_schema_error_prints_the_complete_embedded_plan_example() {
    let (repo, store) = fresh_workspace("plan");
    std::fs::write(repo.join("plan.json"), "{}\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["edit", "apply", "--plan", "plan.json"],
    );

    assert_eq!(code, 20, "stdout={stdout}\nstderr={stderr}");
    assert!(stderr.contains("plan invalid:"), "{stderr}");
    assert!(stderr.contains("minimal complete example:"), "{stderr}");
    assert!(stderr.contains(PLAN_EXAMPLE.trim()), "{stderr}");
}

#[test]
fn change_signature_schema_error_prints_the_complete_embedded_spec_example() {
    let (repo, store) = fresh_workspace("signature");
    std::fs::write(repo.join("sig.json"), "{}\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "edit",
            "change-signature",
            "--symbol",
            "target",
            "--spec",
            "sig.json",
        ],
    );

    assert_eq!(code, 20, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stderr.contains("change-signature --spec sig.json is invalid:"),
        "{stderr}"
    );
    assert!(stderr.contains("minimal complete example:"), "{stderr}");
    assert!(stderr.contains(SIGNATURE_EXAMPLE.trim()), "{stderr}");
}
