#![cfg(target_os = "windows")]

#[path = "common/mounted_contract.rs"]
mod mounted_contract;

use greppy_workspace_core::{capture_repository, ProviderInstallation, WorkspaceCore, CHUNK_SIZE};
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::os::windows::fs::symlink_file;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

struct MountGuard(Child);

impl Drop for MountGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

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
fn mounted_winfsp_provider_satisfies_workspace_and_private_git_contract() {
    if std::env::var_os("GREPPY_RUN_WINFSP_TEST").is_none() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let data = temp.path().join("data");
    let mount = temp.path().join("mount");
    fs::create_dir_all(&repo).unwrap();
    fs::create_dir_all(&mount).unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "test@example.test"]);
    git(&repo, &["config", "user.name", "Test"]);
    fs::write(repo.join("tracked.txt"), b"base\n").unwrap();
    git(&repo, &["add", "tracked.txt"]);
    git(&repo, &["commit", "-qm", "base"]);
    fs::write(repo.join("tracked.txt"), b"dirty\n").unwrap();
    fs::write(repo.join("untracked.txt"), b"user\n").unwrap();

    let core = WorkspaceCore::open(data.join("core")).unwrap();
    let baseline = capture_repository(&repo, core.chunks()).unwrap();
    let workspace = core.create_workspace("mount-test", baseline).unwrap();

    let child = Command::new(env!("CARGO_BIN_EXE_greppy-workspace-provider"))
        .args(["--data-root", data.to_str().unwrap()])
        .args(["--mount-root", mount.to_str().unwrap()])
        .env("GREPPY_WINFSP_DEBUG", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let _guard = MountGuard(child);
    let started = Instant::now();
    let provider = loop {
        match ProviderInstallation::require_healthy(&data) {
            Ok(provider) => break provider,
            Err(error) => {
                assert!(
                    started.elapsed() < Duration::from_secs(20),
                    "WinFsp provider did not become healthy: {error}"
                );
            }
        }
        thread::sleep(Duration::from_millis(100));
    };
    provider.doctor_io("windows-mount").unwrap();

    let root = provider.workspace_path(workspace.id()).unwrap();
    mounted_contract::exercise_mounted_contract(&root, &core);
    assert_eq!(fs::read(root.join("tracked.txt")).unwrap(), b"dirty\n");
    assert_eq!(fs::read(root.join("untracked.txt")).unwrap(), b"user\n");
    fs::write(root.join("unicode-ä.txt"), b"unicode").unwrap();
    fs::rename(root.join("unicode-ä.txt"), root.join("renamed.txt")).unwrap();
    fs::remove_file(root.join("renamed.txt")).unwrap();
    fs::write(root.join("atomic-target.txt"), b"old").unwrap();
    fs::write(root.join("atomic-temporary.txt"), b"new").unwrap();
    fs::rename(
        root.join("atomic-temporary.txt"),
        root.join("atomic-target.txt"),
    )
    .unwrap();
    assert_eq!(fs::read(root.join("atomic-target.txt")).unwrap(), b"new");
    symlink_file("tracked.txt", root.join("tracked.link")).unwrap();
    assert_eq!(
        fs::read_link(root.join("tracked.link")).unwrap(),
        Path::new("tracked.txt")
    );
    fs::hard_link(root.join("tracked.txt"), root.join("tracked.hard")).unwrap();
    assert_eq!(fs::read(root.join("tracked.hard")).unwrap(), b"dirty\n");

    fs::write(root.join("large.bin"), vec![7_u8; CHUNK_SIZE * 2]).unwrap();
    let before = core.chunks().stats().unwrap();
    let mut large = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join("large.bin"))
        .unwrap();
    large.seek(SeekFrom::Start(CHUNK_SIZE as u64 + 3)).unwrap();
    large.write_all(b"X").unwrap();
    large.sync_all().unwrap();
    let after = core.chunks().stats().unwrap();
    assert_eq!(after.chunk_count, before.chunk_count + 1);

    git(&root, &["init", "-q"]);
    git(&root, &["config", "user.email", "test@example.test"]);
    git(&root, &["config", "user.name", "Test"]);
    git(&root, &["add", "-A"]);
    assert!(!git(&root, &["status", "--porcelain"]).is_empty());
    assert!(!git(&root, &["write-tree"]).is_empty());
}
