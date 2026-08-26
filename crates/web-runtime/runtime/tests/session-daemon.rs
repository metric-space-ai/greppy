#![cfg(unix)]

use greppy_web_client::{unix_request, Request, SCHEMA};
use serde_json::json;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn wait_for_socket(path: &std::path::Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("supervisor socket {} was not created", path.display());
}

struct Supervisor(Child);

impl Drop for Supervisor {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn serve_fixture(html: &'static str) -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture");
    let address = listener.local_addr().expect("addr");
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
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
fn session_create_run_close_over_unix_socket() {
    let socket =
        std::env::temp_dir().join(format!("greppy-web-session-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);
    let fixture = serve_fixture(include_str!("../fixtures/spike.html"));
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/spike.mjs");
    let child = Command::new(env!("CARGO_BIN_EXE_web-runtime-supervisor"))
        .arg("--controller-worker")
        .arg(env!("CARGO_BIN_EXE_web-controller-worker"))
        .arg("--content-worker")
        .arg(env!("CARGO_BIN_EXE_web-content-worker"))
        .arg("--socket")
        .arg(&socket)
        .arg("--run-id")
        .arg("run_test")
        .arg("--fixture-url")
        .arg(&fixture)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn supervisor daemon");
    let _guard = Supervisor(child);
    wait_for_socket(&socket, Duration::from_secs(30));

    let handshake = unix_request(
        &socket,
        &Request::new("run_test", "handshake", json!({})),
        Duration::from_secs(5),
    )
    .expect("handshake");
    assert_eq!(handshake.status, "ok");
    assert_eq!(handshake.schema, SCHEMA);
    assert!(handshake.handshake.is_some());

    let created = unix_request(
        &socket,
        &Request::new(
            "run_test",
            "web.session.create",
            json!({ "profile": "research" }),
        ),
        Duration::from_secs(5),
    )
    .expect("session create");
    assert_eq!(created.status, "ok");
    let session_id = created
        .result
        .as_ref()
        .and_then(|value| value.get("session_id"))
        .and_then(|value| value.as_str())
        .expect("session_id")
        .to_owned();
    assert!(session_id.starts_with("wrs_"));

    let source = std::fs::read_to_string(&script).unwrap();
    let mut run = Request::new(
        "run_test",
        "web.run",
        json!({
            "session_id": session_id,
            "script_source": "file",
            "script_file": script.display().to_string(),
            "script_text": source,
        }),
    );
    run.deadline_ms = 120_000;
    let ran = unix_request(&socket, &run, Duration::from_secs(120)).expect("web.run");
    assert_eq!(ran.status, "ok", "{ran:?}");

    let closed = unix_request(
        &socket,
        &Request::new(
            "run_test",
            "web.session.close",
            json!({ "session_id": session_id }),
        ),
        Duration::from_secs(5),
    )
    .expect("session close");
    assert_eq!(closed.status, "ok");
}
