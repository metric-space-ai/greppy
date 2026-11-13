//! Byte-preserving public C++ edits and refusal atomicity for allocation calls.
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

const SOURCE: &str = "#include <stddef.h>\nvoid *operator new(size_t);\nvoid *operator new[](size_t n) {return operator new(n);}\n";

fn run(repo: &Path, store: &Path, args: &[&str], source: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_AGENT_STORE_MODE")
        .env_remove("GREPPY_AGENT_BASE_STORE")
        .env_remove("GREPPY_AGENT_BASE_COMMIT")
        .env_remove("GREPPY_VERIFY_TEST_COMMAND")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn cpp_operator_call_write_preserves_bytes_and_refuses_invalid_transactions() {
    let scratch = tempfile::tempdir().unwrap();
    let repo = scratch.path().join("repo");
    let store = scratch.path().join("store");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    for source in [SOURCE.to_string(), SOURCE.replace('\n', "\r\n")] {
        let output = run(&repo, &store, &["write", "calls.cpp", "--json"], &source);
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            std::fs::read(repo.join("calls.cpp")).unwrap(),
            source.as_bytes()
        );
    }
    let retained = std::fs::read(repo.join("calls.cpp")).unwrap();
    let invalid = SOURCE.replace("operator new(n);", "operator new(n)");
    for file in ["calls.cpp", "absent.cpp"] {
        let output = run(&repo, &store, &["write", file, "--json"], &invalid);
        assert_eq!(
            output.status.code(),
            Some(13),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let answer: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(answer["published"], false);
        let guidance = answer["error"]["next"].as_str().unwrap();
        assert!(guidance.contains("C++") && guidance.contains("tree-sitter"));
        assert!(!guidance.contains("intermediate state"));
    }
    assert_eq!(std::fs::read(repo.join("calls.cpp")).unwrap(), retained);
    assert!(!repo.join("absent.cpp").exists());
    let patch = format!("*** Begin Patch\n*** Add File: witness.txt\n+must stay absent\n*** Add File: invalid.cpp\n{}*** End Patch\n", invalid.lines().map(|line| format!("+{line}\n")).collect::<String>());
    let output = run(&repo, &store, &["patch", "--json"], &patch);
    assert_eq!(
        output.status.code(),
        Some(13),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!repo.join("witness.txt").exists());
    assert!(!repo.join("invalid.cpp").exists());
    assert_eq!(std::fs::read(repo.join("calls.cpp")).unwrap(), retained);
}
