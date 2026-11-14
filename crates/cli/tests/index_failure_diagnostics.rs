//! An index child must persist the real error, including before snapshot setup.
use std::process::{Command, Stdio};

#[test]
fn background_index_preserves_early_store_error() {
    let scratch = tempfile::tempdir().unwrap();
    let repo = scratch.path().join("repo");
    std::fs::create_dir(&repo).unwrap();
    std::fs::write(repo.join("fixture.c"), "int fixture(void) { return 1; }\n").unwrap();
    let blocked_store = scratch.path().join("cache-blocker");
    std::fs::write(&blocked_store, "a file cannot contain a workspace cache").unwrap();
    let job_path = scratch.path().join("background-job.json");
    let mut child = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["index", "."])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &blocked_store)
        .env("GREPPY_BACKGROUND_JOB", &job_path)
        .env_remove("GREPPY_BACKGROUND_DEMAND_LOCK")
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let job = serde_json::json!({
        "schema_version": "greppy.background-job.v2",
        "pid": child.id(),
        "started_at_unix_secs": 1,
        "target_generation": 1,
        "kind": "index",
        "cause": "fixture-store-error",
        "state": "refreshing"
    });
    if let Err(error) = std::fs::write(&job_path, serde_json::to_vec(&job).unwrap()) {
        let _ = child.kill();
        let _ = child.wait();
        panic!("write child ownership record: {error}");
    }
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "invalid store was accepted: {stderr}"
    );
    let job: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&job_path).unwrap()).unwrap();
    assert_eq!(job["state"], "failed", "{job}");
    let reason = job["last_error"].as_str().unwrap();
    assert!(
        reason.contains("cache-blocker"),
        "early store error was replaced by a generic publication failure: {job}; stderr: {stderr}"
    );
    assert!(
        stderr.contains(reason),
        "persisted cause differs from the actual CLI error: {job}; stderr: {stderr}"
    );
}
