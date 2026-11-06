//! Real local HTTP regressions for buffered error bodies and model discovery.
use super::*;
use crate::protocol::ToolChoice;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Instant;

fn request() -> ModelRequest {
    ModelRequest {
        model: "fixture-model".into(),
        system: None,
        messages: vec![],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: 8,
    }
}

fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    // Darwin accept can inherit the listener's nonblocking flag. A read timeout
    // does not clear it; normalize before the bounded request read.
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut headers = String::new();
    let mut length = 0;
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        if line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap();
        }
        headers.push_str(&line);
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    (headers, body)
}

fn fixture(responses: Vec<(u16, String, usize)>, auth: bool) -> (Client, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let server = thread::spawn(move || {
        for (index, (status, body, missing_bytes)) in responses.into_iter().enumerate() {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "missing HTTP request {index}");
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("accept fixture request: {error}"),
                }
            };
            let (headers, request_body) = read_request(&mut stream);
            if auth {
                let headers = headers.to_ascii_lowercase();
                assert!(headers.contains("x-api-key: fixture-not-a-credential\r\n"));
                assert!(headers.contains("authorization: bearer fixture-not-a-credential\r\n"));
                if index == 0 {
                    assert!(headers.starts_with("get /v1/models "));
                } else {
                    assert!(headers.starts_with("post /v1/messages "));
                    let value: Value = serde_json::from_slice(&request_body).unwrap();
                    assert_eq!(value["max_tokens"], 16);
                }
            }
            // One buffered write reproduces the old content-length conversion
            // path; the Linux shim makes its timeout-reset syscall fail EINVAL.
            let response = format!(
                "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
                body.len() + missing_bytes
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
    });
    let mut client = Client::new(&format!("http://{address}"), "fixture-model");
    if auth {
        client = client.with_api_key("fixture-not-a-credential");
    }
    (client, server)
}

fn buffered_http_error() {
    let body = "model request limit reached";
    let (client, server) = fixture(vec![(502, body.into(), 0)], false);
    let result = client.stream_turn(&request(), &mut |_| {});
    server.join().unwrap();
    assert_eq!(
        result,
        Err(ClientError::Http {
            status: 502,
            body: body.into()
        })
    );
}

#[test]
fn buffered_http_error_survives_socket_timeout_reset_failure() {
    #[cfg(target_os = "linux")]
    if std::env::var_os("GREPPY_TEST_SOCKET_RESET_FAULT").is_none() {
        let temporary = tempfile::tempdir().unwrap();
        let shim = temporary.path().join("fail-timeout-reset.so");
        let compiler = std::process::Command::new("cc")
            .args(["-shared", "-fPIC", "-Wall", "-Werror"])
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/test_fail_timeout_reset.c"
            ))
            .arg("-o")
            .arg(&shim)
            .arg("-ldl")
            .output()
            .expect("compile Linux socket fault fixture");
        assert!(
            compiler.status.success(),
            "socket fault fixture: {}",
            String::from_utf8_lossy(&compiler.stderr)
        );
        let mut preload = shim.into_os_string();
        if let Some(existing) = std::env::var_os("LD_PRELOAD") {
            preload.push(":");
            preload.push(existing);
        }
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "client::http_error_tests::buffered_http_error_survives_socket_timeout_reset_failure", "--nocapture", "--test-threads=1"])
            .env("LD_PRELOAD", preload)
            .env("GREPPY_TEST_SOCKET_RESET_FAULT", "1")
            .output().unwrap();
        assert!(
            child.status.success(),
            "buffered HTTP error must be returned, not panic:\n{}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        return;
    }
    buffered_http_error();
}

#[test]
fn http_body_read_errors_remain_explicit() {
    let (client, server) = fixture(vec![(502, "short".into(), 100)], false);
    let result = client.stream_turn(&request(), &mut |_| {});
    server.join().unwrap();
    assert!(
        matches!(result, Err(ClientError::Http { status: 502, body }) if body.contains("response body unavailable")),
        "HTTP status and unreadable body must be reported"
    );

    let (client, server) = fixture(vec![(200, "short".into(), 100)], false);
    let result = client.list_models();
    server.join().unwrap();
    assert!(
        matches!(result, Err(ProbeError::BadResponse(ref body)) if body.contains("read HTTP 200 models response")),
        "unreadable model list is not an empty successful list: {result:?}"
    );

    let (client, server) = fixture(vec![(200, "short".into(), 100)], false);
    let result = client.probe();
    server.join().unwrap();
    assert!(
        matches!(result, Err(ProbeError::BadResponse(_))),
        "unreadable probe is not success: {result:?}"
    );
}

#[test]
fn model_discovery_preserves_auth_and_does_not_nest_runtimes() {
    for interruptible in [false, true] {
        let (client, server) = fixture(
            vec![
                (
                    200,
                    r#"{"data":[{"id":"fixture-model","max_output_tokens":16}]}"#.into(),
                    0,
                ),
                (503, "gateway unavailable".into(), 0),
            ],
            true,
        );
        let mut request = request();
        request.max_tokens = u64::MAX;
        let result = if interruptible {
            client.stream_turn_interruptible(&request, &mut |_| {}, &AtomicBool::new(false))
        } else {
            client.stream_turn(&request, &mut |_| {})
        };
        server.join().unwrap();
        assert_eq!(
            result,
            Err(ClientError::Http {
                status: 503,
                body: "gateway unavailable".into()
            })
        );
    }
}
