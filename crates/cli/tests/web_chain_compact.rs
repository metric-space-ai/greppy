//! Compact browser-chain output must preserve step attribution and failure control.
use std::process::Command;

fn run(args: &[&str]) -> (i32, String, String) {
    run_with_view(args, true)
}

fn run_with_view(args: &[&str], compact: bool) -> (i32, String, String) {
    let dir = tempfile::tempdir().expect("isolated workspace");
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(args)
        .current_dir(dir.path())
        .env_remove("GREPPY_WEB_RUNTIME")
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .env_remove("GREPPY_WEB_VIEW")
        .env(
            "GREPPY_WEB_CHAIN_VIEW",
            if compact { "compact" } else { "" },
        )
        .env("GREPPY_RUNTIME_DIR", dir.path().join("runtime"))
        .output()
        .expect("run web chain");
    (
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn compact_chain_is_an_opt_in_experiment() {
    let (code, stdout, stderr) = run_with_view(
        &["web", "do", "script", "list", "::", "script", "list"],
        false,
    );
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert_eq!(stdout.matches("web.do.step").count(), 2, "{stdout}");
    assert!(stdout.contains("\"steps_ran\":2"), "{stdout}");
    assert!(!stdout.contains("chain:"), "{stdout}");
}

#[test]
fn human_chain_attributes_each_payload_without_repeating_protocol_records() {
    let (code, stdout, stderr) = run(&["web", "do", "script", "list", "::", "script", "list"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert_eq!(stdout.matches("step 1/2 script: ok").count(), 1);
    assert_eq!(stdout.matches("step 2/2 script: ok").count(), 1);
    assert_eq!(stdout.matches("\"scripts\":[]").count(), 2);
    assert!(!stdout.contains("web.do.step"), "{stdout}");
}

#[test]
fn json_chain_retains_complete_machine_readable_step_records() {
    let (code, stdout, stderr) = run(&[
        "web", "do", "--json", "script", "list", "::", "script", "list",
    ]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let records: Vec<serde_json::Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("each JSON line is parseable"))
        .filter(|value: &serde_json::Value| value["kind"] == "step")
        .collect();
    assert_eq!(records.len(), 2, "{stdout}");
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["schema"], "greppy.web-runtime.v1");
        assert_eq!(record["operation"], "web.do.step");
        assert_eq!(record["step"], index + 1);
        assert_eq!(record["steps_total"], 2);
        assert_eq!(record["argv"], serde_json::json!(["script", "list"]));
        assert_eq!(record["exit_code"], 0);
        assert_eq!(record["status"], "ok");
    }
}

#[test]
fn compact_chain_stops_at_failure_and_preserves_exit_status() {
    let (code, stdout, stderr) = run(&[
        "web", "do", "script", "list", "::", "script", "show", "missing", "::", "script", "list",
    ]);
    assert_eq!(code, 30, "{stdout}\n{stderr}");
    assert!(
        stdout.contains("step 2/3 script: FAILED (exit 30)"),
        "{stdout}"
    );
    assert!(!stdout.contains("step 3/3"), "{stdout}");
    assert!(
        stdout.contains(
            "chain: 2/3 steps executed, 1 failed; stopped at 2; no rollback attempted"
        ),
        "{stdout}"
    );
}

#[test]
fn compact_chain_continue_on_error_still_returns_failure() {
    let (code, stdout, stderr) = run(&[
        "web",
        "do",
        "--continue-on-error",
        "script",
        "list",
        "::",
        "script",
        "show",
        "missing",
        "::",
        "script",
        "list",
    ]);
    assert_eq!(code, 30, "{stdout}\n{stderr}");
    assert!(
        stdout.contains("step 2/3 script: FAILED (exit 30)"),
        "{stdout}"
    );
    assert!(stdout.contains("step 3/3 script: ok"), "{stdout}");
}
