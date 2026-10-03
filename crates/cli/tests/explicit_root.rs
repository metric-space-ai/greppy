//! Diagnose an absent explicit root before edit containment or any write.
use std::process::Command;

#[test]
fn absent_explicit_root_write_identifies_missing_directory_without_writing() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("not-created");
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args([
            "--root",
            root.to_str().unwrap(),
            "write",
            "child/probe.rs",
            "fn probe() {}",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("--root directory does not exist:"),
        "{stderr}"
    );
    assert!(stderr.contains(&root.display().to_string()), "{stderr}");
    assert!(
        stderr.contains("create that directory before retrying"),
        "{stderr}"
    );
    assert!(!stderr.contains("is outside"), "{stderr}");
    assert!(!root.exists());
}

#[test]
fn explicit_root_file_is_diagnosed_without_overwriting_it() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("file");
    std::fs::write(&root, "protected").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args([
            "--root",
            root.to_str().unwrap(),
            "write",
            "probe.rs",
            "fn probe() {}",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("--root must name a directory, not a file:"),
        "{stderr}"
    );
    assert_eq!(std::fs::read_to_string(root).unwrap(), "protected");
    assert!(!dir.path().join("probe.rs").exists());
}
