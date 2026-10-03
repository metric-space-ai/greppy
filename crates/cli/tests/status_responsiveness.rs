#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

fn fixture(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "greppy-status-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn status_budget_reports_unknown_and_reaps_only_its_git_child() {
    let root = fixture("bounded-git");
    let bin = root.join("bin");
    std::fs::create_dir(&bin).unwrap();
    let git = bin.join("git");
    std::fs::write(
        &git,
        "#!/bin/sh\necho $$ > \"$STATUS_TEST_PID\"\nexec /bin/sleep 30\n",
    )
    .unwrap();
    std::fs::set_permissions(&git, std::fs::Permissions::from_mode(0o700)).unwrap();
    let pid_path = root.join("git.pid");
    let start = std::time::Instant::now();
    let result = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["index", "status", "--json"])
        .current_dir(&root)
        .env("PATH", &bin)
        .env("STATUS_TEST_PID", &pid_path)
        .env("GREPPY_STORE_DIR", root.join("store"))
        .env_remove("GREPPY_INTERNAL_STATUS_WORKER")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(75), "{:?}", result);
    assert!(start.elapsed() < std::time::Duration::from_secs(15));
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["status"], "unknown");
    assert_eq!(value["diagnostic_phase"], "git_status");
    assert_eq!(value["diagnostics_complete"], false);
    for field in [
        "healthy",
        "fresh",
        "integrity_ok",
        "embedding_complete",
        "store_bytes",
    ] {
        assert!(value[field].is_null(), "{field}: {value}");
    }
    assert!(!String::from_utf8_lossy(&result.stderr).contains("greppy-status-phase:"));
    let pid: i32 = std::fs::read_to_string(&pid_path)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    // SIGKILL delivery/reaping of a grandchild can finish just after its parent.
    for _ in 0..100 {
        if unsafe { libc::kill(pid, 0) } == -1 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        -1,
        "owned Git child survived status timeout"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn prompt_status_keeps_existing_no_index_result() {
    let root = fixture("prompt");
    let result = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["index", "status", "--json"])
        .current_dir(&root)
        .env("GREPPY_STORE_DIR", root.join("store"))
        .env_remove("GREPPY_INTERNAL_STATUS_WORKER")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1), "{:?}", result);
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["status"], "no_index");
    assert_eq!(value["healthy"], false);
    assert_eq!(value["fresh"], false);
    assert!(!String::from_utf8_lossy(&result.stderr).contains("greppy-status-phase:"));
    std::fs::remove_dir_all(root).unwrap();
}
