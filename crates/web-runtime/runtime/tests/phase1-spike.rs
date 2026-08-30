use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::thread;

fn serve_fixture(html: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
    let address = listener.local_addr().expect("fixture server address");
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else {
                continue;
            };
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            let body = html.as_bytes();
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
        }
    });
    format!("http://{address}/")
}

#[test]
fn unchanged_playwright_script_controls_servo_across_process_boundary() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/spike.mjs");
    let fixture = serve_fixture(include_str!("../fixtures/spike.html"));
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
    let fixture = serve_fixture(include_str!("../fixtures/spike.html"));
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
