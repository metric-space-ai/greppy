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
