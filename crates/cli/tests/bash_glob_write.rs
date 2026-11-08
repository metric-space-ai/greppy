//! Actual Bash parser and atomic public write contract; scripts are never executed.
#![cfg(unix)]
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

const VALID: &str = "#!/usr/bin/env bash\nfor acceptance_candidate in /synthetic-node; do\n  [[ \"$acceptance_candidate\" = /* && -x \"$acceptance_candidate\" ]] || continue\ndone\n";

fn input(mut command: Command, content: &str) -> Output {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(content.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
fn bash_n(repo: &Path, content: &str) -> Output {
    let mut command = Command::new("/bin/bash");
    command
        .args(["--noprofile", "--norc", "-n"])
        .env_clear()
        .current_dir(repo);
    input(command, content)
}
fn write(repo: &Path, store: &Path, file: &str, content: &str) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_greppy"));
    command
        .args(["write", file])
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_AGENT_STORE_MODE")
        .env_remove("GREPPY_AGENT_BASE_STORE")
        .env_remove("GREPPY_AGENT_BASE_COMMIT")
        .env_remove("GREPPY_VERIFY_TEST_COMMAND");
    input(command, content)
}
#[test]
fn complete_bash_absolute_glob_write_matches_native_syntax() {
    let scratch = tempfile::tempdir().unwrap();
    let repo = scratch.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let native = bash_n(&repo, VALID);
    assert!(
        native.status.success(),
        "bash -n rejected fixture: {}",
        String::from_utf8_lossy(&native.stderr)
    );
    eprintln!("BASH_GLOB_SYNTAX_VALID");
    let tree = greppy_parser::parse(greppy_parser::Language::Bash, VALID.as_bytes()).unwrap();
    eprintln!("ORIGINAL_BASH_TREE={}", tree.root_node().to_sexp());
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if matches!(node.kind(), "test_command" | "binary_expression" | "regex")
            || node.is_missing()
        {
            eprintln!(
                "BASH_NODE kind={} range={:?} bytes={:?}",
                node.kind(),
                node.byte_range(),
                &VALID.as_bytes()[node.byte_range()]
            );
        }
        let mut cursor = node.walk();
        stack.extend(node.children(&mut cursor));
    }
    let result = write(
        &repo,
        &scratch.path().join("store"),
        "greppy-glob-repro.sh",
        VALID,
    );
    assert!(
        result.status.success(),
        "valid complete Bash glob write rejected: exit={:?}; {}\n{}",
        result.status.code(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("greppy-glob-repro.sh")).unwrap(),
        VALID
    );
}
#[test]
fn broken_complete_bash_remains_atomic() {
    let scratch = tempfile::tempdir().unwrap();
    let repo = scratch.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let store = scratch.path().join("store");
    let retained = "#!/usr/bin/env bash\nprintf '%s\\n' retained\n";
    for bad in [
        VALID.replace(" ]]", " ]"),
        VALID.replace("done\n", ""),
        VALID.replace(" && -x", " && && -x"),
        VALID.replace("\" ]]", " ]]"),
        format!("{VALID}if then\n"),
    ] {
        assert!(
            !bash_n(&repo, &bad).status.success(),
            "invalid native control accepted: {bad}"
        );
        for (name, existing) in [("existing.sh", true), ("new.sh", false)] {
            let file = repo.join(name);
            if existing {
                std::fs::write(&file, retained).unwrap();
            }
            let result = write(&repo, &store, name, &bad);
            assert_eq!(
                result.status.code(),
                Some(13),
                "broken Bash accepted: {bad}; {}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            if existing {
                assert_eq!(std::fs::read_to_string(&file).unwrap(), retained);
            } else {
                assert!(!file.exists());
            }
        }
    }
}
