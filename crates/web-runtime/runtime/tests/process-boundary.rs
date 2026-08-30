use std::path::PathBuf;
use std::process::Command;

#[test]
fn supervisor_runs_both_runtime_workers_across_process_boundaries() {
    let output = Command::new(env!("CARGO_BIN_EXE_web-runtime"))
        .output()
        .expect("supervisor must launch");

    assert!(
        output.status.success(),
        "supervisor failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("supervisor stdout must be UTF-8");
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [
            "web_runtime.controller=ready",
            "web_runtime.content=ready",
            "web_runtime.content=stopped",
            "web_runtime.controller=stopped",
            "web_runtime.supervisor=stopped",
        ]
    );
}

fn binary_contains(path: &str, needle: &[u8]) -> bool {
    let bytes = std::fs::read(path).expect("read binary");
    bytes.windows(needle.len()).any(|window| window == needle)
}

fn worker_role_rejects_capability_secret_in_argv_for(role: &str) {
    let output = Command::new(env!("CARGO_BIN_EXE_web-runtime"))
        .args(["--internal-role", role, "--capability", "secret-token"])
        .output()
        .expect("spawn argv-capability worker");
    assert!(
        !output.status.success(),
        "{role} accepted argv capability: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let err = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        err.contains("argv") || err.contains("capability"),
        "{role} unexpected rejection text: {err}"
    );
}

#[test]
fn worker_role_rejects_capability_secret_in_argv() {
    worker_role_rejects_capability_secret_in_argv_for("controller");
    worker_role_rejects_capability_secret_in_argv_for("content");
}

#[test]
fn worker_role_rejects_missing_inherited_capability_fd() {
    for role in ["controller", "content"] {
        let output = Command::new(env!("CARGO_BIN_EXE_web-runtime"))
            .args(["--internal-role", role])
            .output()
            .expect("spawn worker without capability fd");
        assert!(
            !output.status.success(),
            "{role} started without inherited capability FD: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let err = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            err.contains("missing inherited capability FD") || err.contains("capability"),
            "{role} unexpected rejection text: {err}"
        );
    }
}

#[test]
fn packaged_or_cargo_web_runtime_is_the_only_runtime_bin_name() {
    let runtime = PathBuf::from(env!("CARGO_BIN_EXE_web-runtime"));
    assert_eq!(
        runtime.file_name().and_then(|name| name.to_str()),
        Some("web-runtime")
    );
}

#[test]
fn one_runtime_image_contains_supervisor_controller_and_content() {
    let runtime = env!("CARGO_BIN_EXE_web-runtime");
    let servo_marker = b"SoftwareRenderingContext";
    let v8_marker = b"op_engine_call";
    assert!(
        binary_contains(runtime, v8_marker),
        "web-runtime must contain the V8 controller op"
    );
    assert!(
        binary_contains(runtime, servo_marker),
        "web-runtime must contain Servo software renderer types"
    );
}

#[test]
fn internal_role_is_hidden_from_ordinary_invocation() {
    let runtime = env!("CARGO_BIN_EXE_web-runtime");
    let help = Command::new(runtime).arg("--help").output().expect("help");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&help.stdout),
        String::from_utf8_lossy(&help.stderr)
    );
    assert!(
        !text.contains("--internal-role controller")
            && !text.contains("--internal-role content")
            && !text.to_ascii_lowercase().contains("usage:"),
        "--internal-role must stay hidden from ordinary CLI help: {text}"
    );
}

#[test]
fn greppy_cli_crate_does_not_depend_on_javascript_engines() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cli/Cargo.toml");
    let text = std::fs::read_to_string(&manifest).expect("cli Cargo.toml");
    assert!(
        !text.contains("servo") && !text.contains("deno_core"),
        "crates/cli must not depend on Servo or deno_core: {manifest:?}"
    );
    let client = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web-client/Cargo.toml");
    let text = std::fs::read_to_string(&client).expect("web-client Cargo.toml");
    assert!(
        !text.contains("servo") && !text.contains("deno_core"),
        "web-client must not depend on Servo or deno_core"
    );
}
