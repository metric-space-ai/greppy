use std::path::PathBuf;
use std::process::Command;

#[test]
fn supervisor_runs_both_runtime_workers_across_process_boundaries() {
    let output = Command::new(env!("CARGO_BIN_EXE_web-runtime-supervisor"))
        .arg("--controller-worker")
        .arg(env!("CARGO_BIN_EXE_web-controller-worker"))
        .arg("--content-worker")
        .arg(env!("CARGO_BIN_EXE_web-content-worker"))
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

#[test]
fn three_runtime_images_do_not_colink_engines() {
    let supervisor = env!("CARGO_BIN_EXE_web-runtime-supervisor");
    let controller = env!("CARGO_BIN_EXE_web-controller-worker");
    let content = env!("CARGO_BIN_EXE_web-content-worker");
    assert_ne!(supervisor, controller);
    assert_ne!(supervisor, content);
    assert_ne!(controller, content);

    let servo_marker = b"SoftwareRenderingContext";
    let v8_marker = b"op_engine_call";
    assert!(
        !binary_contains(supervisor, servo_marker),
        "supervisor must not contain Servo software renderer types"
    );
    assert!(
        !binary_contains(supervisor, v8_marker),
        "supervisor must not contain the V8 controller op"
    );
    assert!(
        binary_contains(controller, v8_marker),
        "controller worker must contain op_engine_call"
    );
    assert!(
        !binary_contains(controller, servo_marker),
        "controller worker must not contain Servo software renderer types"
    );
    assert!(
        binary_contains(content, servo_marker),
        "content worker must contain SoftwareRenderingContext"
    );
    assert!(
        !binary_contains(content, v8_marker),
        "content worker must not contain the V8 controller op"
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
