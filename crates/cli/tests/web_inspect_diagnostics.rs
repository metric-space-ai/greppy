//! Invalid identities must never reach CSS evaluation or runtime startup.
use std::process::Command;

#[test]
fn malformed_inspect_refs_fail_before_session_discovery() {
    let workspace = tempfile::tempdir().unwrap();
    for (query, diagnostic) in [
        ("@0", "ref numbering starts at @1"),
        ("@abc", "ref must be @ followed by digits"),
        (
            "@18446744073709551616",
            "outside the supported integer range",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .args(["web", "inspect", query, "--tab", "selected-tab", "--json"])
            .current_dir(workspace.path())
            .env_remove("GREPPY_WEB_SESSION")
            .env_remove("GREPPY_WEB_TAB")
            .env_remove("GREPPY_AGENT_ID")
            .env_remove("GREPPY_WEB_AGENT")
            .env_remove("GREPPY_RUN_ID")
            .env_remove("GREPPY_WEB_RUNTIME_DIST")
            .env(
                "GREPPY_WEB_RUNTIME",
                workspace.path().join("missing-runtime"),
            )
            .env("GREPPY_RUNTIME_DIR", workspace.path().join("runtime"))
            .env(
                "GREPPY_WEB_RUNTIME_DIR",
                workspace.path().join("web-runtime"),
            )
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            output.status.code(),
            Some(30),
            "{query}: {stdout}\n{stderr}"
        );
        let response: serde_json::Value = serde_json::from_str(&stdout).expect("one JSON error");
        assert!(response.to_string().contains(diagnostic), "{response}");
        assert!(!stdout.contains("NO_SESSION"), "{stdout}");
        assert!(!stdout.contains("EvaluationFailure"), "{stdout}");
        assert!(!stderr.contains("ignoring"), "{stderr}");
    }
}
