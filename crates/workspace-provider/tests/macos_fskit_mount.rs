#![cfg(target_os = "macos")]

#[path = "common/mounted_contract.rs"]
mod mounted_contract;

use greppy_workspace_core::{capture_repository, ProviderInstallation, WorkspaceCore};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn git(path: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().into()
}

#[test]
fn activated_fskit_provider_satisfies_shared_mounted_contract() {
    if std::env::var_os("GREPPY_RUN_FSKIT_TEST").is_none() {
        return;
    }
    let data_root = std::env::var_os("GREPPY_FSKIT_DATA_ROOT")
        .map(std::path::PathBuf::from)
        .expect("GREPPY_FSKIT_DATA_ROOT must bind the activated provider data root");
    assert!(
        data_root.is_absolute(),
        "GREPPY_FSKIT_DATA_ROOT must be absolute"
    );

    let provider = ProviderInstallation::require_healthy(&data_root)
        .expect("the signed FSKit provider must already be activated and mounted");
    provider.doctor_io("macos-fskit-contract").unwrap();
    let core = WorkspaceCore::open(data_root.join("core")).unwrap();

    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    fs::create_dir_all(repo.join("nested/deeper")).unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "test@example.test"]);
    git(&repo, &["config", "user.name", "Test"]);
    fs::write(repo.join("tracked.txt"), b"base\n").unwrap();
    fs::write(repo.join("nested/deeper/base.txt"), b"nested base\n").unwrap();
    git(&repo, &["add", "tracked.txt", "nested/deeper/base.txt"]);
    git(&repo, &["commit", "-qm", "base"]);
    fs::write(repo.join("tracked.txt"), b"dirty\n").unwrap();
    fs::write(repo.join("untracked.txt"), b"user\n").unwrap();

    let baseline = capture_repository(&repo, core.chunks()).unwrap();
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let workspace = core
        .create_workspace(
            &format!("macos-fskit-contract-{}-{suffix}", std::process::id()),
            baseline,
        )
        .unwrap();
    let root = provider.workspace_path(workspace.id()).unwrap();
    let started = Instant::now();
    while !root.is_dir() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "FSKit did not publish workspace {}",
            workspace.id()
        );
        thread::sleep(Duration::from_millis(25));
    }

    let immutable_inode = core
        .metadata(&workspace, "nested/deeper/base.txt")
        .unwrap()
        .unwrap()
        .inode;
    assert_eq!(
        fs::read(root.join("nested/deeper/base.txt")).unwrap(),
        b"nested base\n"
    );
    assert_eq!(
        core.metadata(&workspace, "nested/deeper/base.txt")
            .unwrap()
            .unwrap()
            .inode,
        immutable_inode,
        "a read-only FSKit open must not copy an immutable Base file into the private namespace"
    );

    mounted_contract::exercise_mounted_contract(&root, &core);
    assert_eq!(fs::read(root.join("tracked.txt")).unwrap(), b"dirty\n");
    assert_eq!(fs::read(root.join("untracked.txt")).unwrap(), b"user\n");

    core.remove_workspace(workspace).unwrap();
}
