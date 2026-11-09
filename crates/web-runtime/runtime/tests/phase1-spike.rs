use std::path::PathBuf;
use std::process::Command;

#[path = "support/owned_html_fixture.rs"]
mod owned_html_fixture;

fn serve_fixture(
    html: impl Into<std::borrow::Cow<'static, str>>,
) -> (owned_html_fixture::HtmlFixture, String) {
    owned_html_fixture::HtmlFixture::start(html)
}

#[test]
fn unchanged_playwright_script_controls_servo_across_process_boundary() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/spike.mjs");
    let (_fixture_owner, fixture) = serve_fixture(include_str!("../fixtures/spike.html"));
    let output = Command::new(env!("CARGO_BIN_EXE_web-runtime"))
        .arg("--script")
        .arg(&script)
        .arg("--fixture-url")
        .arg(&fixture)
        .output()
        .expect("supervisor must launch the Phase 1 spike");

    assert!(
        output.status.success(),
        "Phase 1 spike failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("supervisor stdout must be UTF-8");
    assert!(
        stdout.contains("web_runtime.script=ok"),
        "missing script completion marker in {stdout}"
    );
    assert!(stdout.contains("web_runtime.supervisor=stopped"));
}

#[test]
fn supervisor_reuses_workers_for_two_scripts() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/spike.mjs");
    let (_fixture_owner, fixture) = serve_fixture(include_str!("../fixtures/spike.html"));
    let output = Command::new(env!("CARGO_BIN_EXE_web-runtime"))
        .arg("--script")
        .arg(&script)
        .arg("--script")
        .arg(&script)
        .arg("--fixture-url")
        .arg(&fixture)
        .output()
        .expect("supervisor must launch two scripts");

    assert!(
        output.status.success(),
        "two-script session failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout utf-8");
    assert_eq!(stdout.matches("web_runtime.script=ok").count(), 2);
}
