//! Invalid waits must fail before session discovery or browser polling.
use std::process::Command;

fn run(verb: &str, query: &str) -> (i32, String) {
    let dir = tempfile::tempdir().expect("isolated workspace");
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["web", verb, query])
        .current_dir(dir.path())
        .env_remove("GREPPY_WEB_SESSION")
        .env_remove("GREPPY_WEB_TAB")
        .env_remove("GREPPY_AGENT_ID")
        .env_remove("GREPPY_WEB_AGENT")
        .env_remove("GREPPY_RUN_ID")
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .env("GREPPY_WEB_RUNTIME", dir.path().join("absent-runtime"))
        .env("GREPPY_WEB_RUNTIME_DIR", dir.path().join("web-runtime"))
        .env("GREPPY_RUNTIME_DIR", dir.path().join("runtime"))
        .output()
        .expect("run condition command");
    (
        output.status.code().unwrap_or(-1),
        format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

#[test]
fn unknown_query_kind_fails_before_wait_or_assert_resolves_a_session() {
    for verb in ["wait", "assert"] {
        let (code, output) = run(verb, "time=500ms");
        assert_eq!(code, 30, "{output}");
        assert!(output.contains("unknown query kind"), "{output}");
        assert!(output.contains("time"), "{output}");
        assert!(!output.contains("no current web session"), "{output}");
        assert!(!output.contains("waited_ms"), "{output}");
    }
}

#[test]
fn invalid_query_operator_fails_before_browser_polling() {
    let (code, output) = run("wait", "css~div");
    assert_eq!(code, 30, "{output}");
    assert!(output.contains("needs a text query"), "{output}");
    assert!(!output.contains("no current web session"), "{output}");
    assert!(!output.contains("waited_ms"), "{output}");
}

#[test]
fn valid_css_and_javascript_regex_reach_session_resolution() {
    for query in [
        "input[name=quantity]",
        "[data-x=y]",
        "input[class~=quantity]",
        "div~span",
        "css=#absent",
    ] {
        let (code, output) = run("wait", query);
        assert_eq!(code, 30, "query={query}: {output}");
        assert!(output.contains("no current web session"), "query={query}: {output}");
        assert!(!output.contains("unknown query kind"), "{output}");
    }
}
