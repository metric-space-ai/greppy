//! Exact recovery for FILE A:B and FILE:LINE; no index or model is needed.
use std::process::Command;

#[test]
fn positional_file_locations_give_exact_recovery_before_opening_a_graph() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir(repo.join("tests")).unwrap();
    std::fs::write(repo.join("tests/conftest.py"), "def fixture(): pass\n").unwrap();
    std::fs::write(repo.join("test data.py"), "def fixture(): pass\n").unwrap();
    let store = temp.path().join("not-a-store");
    std::fs::write(&store, "must not open").unwrap();
    for (operands, expected) in [
        (
            vec!["tests/conftest.py", "281:380"],
            "greppy read-file tests/conftest.py --lines 281:380",
        ),
        (
            vec!["tests/conftest.py:281"],
            "greppy read-file tests/conftest.py --lines 281:281",
        ),
        (
            vec!["test data.py", "2:3"],
            "greppy read-file 'test data.py' --lines 2:3",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .arg("read")
            .args(operands)
            .args(["--json", "--handle", "--root"])
            .arg(&repo)
            .current_dir(temp.path())
            .env("GREPPY_STORE_DIR", &store)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(64));
        let err = String::from_utf8(output.stderr).unwrap();
        assert!(err.contains(expected), "{err}");
        assert!(err.contains("--handle --json --root"), "{err}");
        assert!(output.stdout.is_empty());
        assert_eq!(std::fs::read_to_string(&store).unwrap(), "must not open");
    }
}

#[cfg(unix)]
#[test]
fn colon_number_in_existing_literal_filename_keeps_file_read_behavior() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join(".git")).unwrap();
    let mut content = "literal contents\n".repeat(401).into_bytes();
    content.push(0xff);
    std::fs::write(temp.path().join("note.txt:7"), content).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["read", "note.txt:7"])
        .current_dir(temp.path())
        .env("GREPPY_STORE_DIR", temp.path().join("store"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("literal contents"));
}

#[test]
fn write_omitted_new_preserves_shell_metacharacters_in_new_python_test() {
    use std::io::Write as _;
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join(".git")).unwrap();
    let content = b"def test_literal():\n    \"\"\"Keep `backticks`, $(printf BAD), $HOME and 'quotes'.\"\"\"\n    assert True\n";
    let mut child = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["write", "test_new.py"])
        .current_dir(temp.path())
        .env("GREPPY_STORE_DIR", temp.path().join("store"))
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(content).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read(temp.path().join("test_new.py")).unwrap(), content);
}
