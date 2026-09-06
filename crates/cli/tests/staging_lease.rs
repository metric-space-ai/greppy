//! Actual CLI ownership handoff, without models, FUSE or a real user store.
#![cfg(all(unix, feature = "bash-smart"))]

use greppy_core::cache::{create_base_build_staging_lease, ENV_BASE_BUILD_STAGING_LEASES};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

fn command(root: &Path) -> Command {
    // An explicitly captured, hash-verified copy on a local disk may be used
    // when the build target is on slow removable storage. This is test-only;
    // the ordinary Cargo test path always uses its own actual built CLI.
    let executable = std::env::var_os("GREPPY_TEST_CLI_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_greppy")));
    assert!(executable.is_absolute(), "test CLI path must be absolute");
    let mut command = Command::new(executable);
    command
        .current_dir(root.join("workspace"))
        .env("GREPPY_STORE_DIR", root.join("store"))
        .env("GREPPY_SHARED_INFERENCE_ROOT", root.join("store"))
        .env("GREPPY_RUNTIME_DIR", root.join("runtime"))
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove(ENV_BASE_BUILD_STAGING_LEASES);
    command
}

fn staging(root: &Path, name: &str) -> PathBuf {
    let path = root.join("store").join(name);
    std::fs::create_dir_all(path.join("data")).unwrap();
    std::fs::write(path.join("data/payload"), b"synthetic build output").unwrap();
    path
}

fn age(path: &Path) {
    let old = SystemTime::now() - Duration::from_secs(7 * 60 * 60);
    std::fs::File::open(path)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new()
                .set_accessed(old)
                .set_modified(old),
        )
        .unwrap();
}

struct LiveCli(Option<Child>);
impl LiveCli {
    fn finish(mut self) {
        self.0
            .as_mut()
            .unwrap()
            .stdin
            .take()
            .unwrap()
            .write_all(b"done\n")
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.0.as_mut().unwrap().try_wait().unwrap().is_none() {
            assert!(
                Instant::now() < deadline,
                "owned CLI did not finish after releasing its input"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = self.0.take().unwrap().wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
impl Drop for LiveCli {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            // Closing our sole writer releases the fixture's read, including
            // assertion failures. Only this directly owned child may be killed.
            child.stdin.take();
            let deadline = Instant::now() + Duration::from_secs(5);
            while matches!(child.try_wait(), Ok(None)) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if matches!(child.try_wait(), Ok(None)) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

fn gc(root: &Path, expected_exit: i32) -> serde_json::Value {
    let output = command(root)
        .args(["cache", "gc", "--json"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(expected_exit),
        "GC failed: status={} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn cli_retains_both_staging_leases_after_parent_release_then_gc_reclaims_them() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    std::fs::create_dir(root.join("workspace")).unwrap();
    let roots = [
        staging(root, "greppy-base-build-cli-parent"),
        staging(root, "greppy-linked-base-checkout-cli-parent"),
    ];
    let parent_leases: Vec<_> = roots
        .iter()
        .map(|path| create_base_build_staging_lease(path).unwrap())
        .collect();
    let ready = root.join("ready");
    let child = command(root)
        .env(
            ENV_BASE_BUILD_STAGING_LEASES,
            std::env::join_paths(&roots).unwrap(),
        )
        .args([
            "bash-smart",
            "--",
            "sh",
            "-c",
            "set -eu; printf ready > \"$1.tmp\"; mv \"$1.tmp\" \"$1\"; IFS= read -r release",
            "greppy-staging-test",
        ])
        .arg(&ready)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut live = LiveCli(Some(child));
    let deadline = Instant::now() + Duration::from_secs(30);
    while std::fs::read(&ready).ok().as_deref() != Some(b"ready") {
        assert!(
            live.0.as_mut().unwrap().try_wait().unwrap().is_none(),
            "CLI exited before handoff"
        );
        assert!(
            Instant::now() < deadline,
            "CLI handoff did not reach command execution"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(parent_leases); // Only the actual CLI may now retain these leases.
    for path in &roots {
        age(path);
    }
    let abandoned = staging(root, "greppy-base-build-abandoned-control");
    drop(create_base_build_staging_lease(&abandoned).unwrap());
    age(&abandoned);
    // GC uses TEMPFAIL for a partial collection with protected, live bytes.
    // That documented status is the success condition for this half of the test.
    let during = gc(root, 75);
    assert!(during["locked_bytes"].as_u64().unwrap() > 0, "{during}");
    assert!(!abandoned.exists(), "GC must actually run: {during}");
    for path in &roots {
        assert!(
            path.join("data/payload").exists(),
            "live CLI staging reclaimed: {during}"
        );
    }
    assert!(live.0.as_mut().unwrap().try_wait().unwrap().is_none());
    live.finish();
    let after = gc(root, 0);
    for path in &roots {
        assert!(!path.exists(), "released staging not reclaimed: {after}");
        assert!(
            after["removed"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry.as_str() == path.to_str()),
            "removal missing from report: {after}"
        );
    }
}

#[test]
fn cli_refuses_missing_staging_lease_before_running_command_or_recreating_marker() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    std::fs::create_dir(root.join("workspace")).unwrap();
    let missing = staging(root, "greppy-base-build-unowned");
    let marker = root.join("command-ran");
    let output = command(root)
        .env(ENV_BASE_BUILD_STAGING_LEASES, &missing)
        .args([
            "bash-smart",
            "--",
            "sh",
            "-c",
            "touch \"$1\"",
            "greppy-staging-test",
        ])
        .arg(&marker)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(73));
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot retain Base build staging"));
    assert!(!marker.exists(), "command ran without ownership");
    assert!(
        !missing.join("locks").exists(),
        "missing ownership was silently recreated"
    );
}
