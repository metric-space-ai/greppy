use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn web_parse_errors_never_rewrite_or_dispatch_another_operation() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let cases = [
        vec![
            "web",
            "session",
            "close",
            "selected-session",
            "--unrecognized-selector=other",
            "--json",
        ],
        vec![
            "web",
            "session",
            "close",
            "--session",
            "selected-session",
            "--json",
        ],
        vec![
            "web",
            "runtime",
            "stop",
            "--unrecognized-selector=other",
            "--json",
        ],
        vec!["web", "runtime", "stop", "unexpected-owner", "--json"],
    ];
    for (index, args) in cases.iter().enumerate() {
        // A private namespace and deliberately absent runtime also keep the
        // controlled pre-fix mutant from reaching any installed browser.
        let private = temp.path().join(index.to_string());
        std::fs::create_dir_all(&private).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .current_dir(&repo)
            .env("HOME", &private)
            .env("XDG_RUNTIME_DIR", private.join("runtime"))
            .env("GREPPY_STORE_DIR", private.join("store"))
            .env(
                "GREPPY_RUN_ID",
                format!("invalid-web-{}-{index}", std::process::id()),
            )
            .env("GREPPY_WEB_RUNTIME", private.join("no-such-runtime"))
            .env_remove("GREPPY_WEB_RUNTIME_DIST")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .args(args)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(8);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("invalid Web arguments must refuse before Web dispatch: {args:?}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            output.status.code(), Some(64),
            "invalid Web arguments must refuse before Web dispatch: {args:?}; stdout={stdout}; stderr={stderr}"
        );
        assert!(
            stdout.contains("no web operation was performed"),
            "{stdout}"
        );
        assert!(stdout.contains("Usage: greppy web"), "{stdout}");
        assert!(
            !stdout.contains("ignoring") && !stdout.contains("using it as"),
            "{stdout}"
        );
    }
}
