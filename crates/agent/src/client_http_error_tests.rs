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

#[test]
fn extra_headers_go_with_every_model_request_and_broken_ones_are_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let (headers, _) = read_request(&mut stream);
        let body = "no";
        let response = format!(
            "HTTP/1.1 503 Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        headers.to_ascii_lowercase()
    });
    let client = Client::new(&format!("http://{address}"), "fixture-model")
        .with_header("x-workjet-provider", "made-up")
        .unwrap();
    let result = client.stream_turn(&request(), &mut |_| {});
    let headers = server.join().unwrap();
    assert!(matches!(result, Err(ClientError::Http { status: 503, .. })));
    assert!(headers.starts_with("post /v1/messages "), "{headers}");
    assert!(
        headers.contains("x-workjet-provider: made-up\r\n"),
        "{headers}"
    );

    let base = Client::new("http://127.0.0.1:1", "fixture-model");
    for (name, value) in [
        ("x-a:b", "v"),
        ("x-a", "v\r\nx-b: w"),
        ("", "v"),
        ("x-a", ""),
    ] {
        assert!(
            base.clone().with_header(name, value).is_err(),
            "{name:?}: {value:?}"
        );
    }
}

#[test]
fn a_chosen_effort_goes_with_the_request_and_none_without() {
    let client = Client::new("http://127.0.0.1:1", "fixture-model");
    let request = request();
    assert!(client
        .request_body(&request)
        .unwrap()
        .get("output_config")
        .is_none());
    let body = client
        .clone()
        .with_effort("high")
        .unwrap()
        .request_body(&request)
        .unwrap();
    assert_eq!(body["output_config"]["effort"], "high");
    assert!(client.with_effort("max").is_err());
}

#[test]
fn the_output_limit_a_refusal_allows_is_read_from_it() {
    // As Anthropic and OpenAI-style routes word it.
    let anthropic = r#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: 64000 > 32000, which is the maximum allowed number of output tokens for made-up-model"}}"#;
    assert_eq!(allowed_output_limit(anthropic, 64_000), Some(32_000));
    let openai =
        "max_tokens is too large: 64000. This model supports at most 16384 completion tokens.";
    assert_eq!(allowed_output_limit(openai, 64_000), Some(16_384));
    // About the output limit, without a number: half of what was asked.
    assert_eq!(
        allowed_output_limit("max_tokens is too large", 64_000),
        Some(32_000)
    );
    assert_eq!(allowed_output_limit("max_tokens is too large", 6_000), None);
    // About something else: not taken for the output limit.
    assert_eq!(
        allowed_output_limit("prompt is too long: 250000 tokens > 200000 maximum", 64_000),
        None
    );
}

#[test]
fn a_refused_output_limit_is_asked_again_with_the_allowed_one_and_kept() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let refusal = r#"{"error":{"message":"max_tokens: 64000 > 32000, which is the maximum allowed number of output tokens"}}"#;
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"made-up\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    let server = thread::spawn(move || {
        let mut asked = Vec::new();
        // The catalog says nothing of a cap; then the refusal, then two answers.
        for answer in [
            None,
            Some((400, refusal)),
            Some((200, sse)),
            Some((200, sse)),
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            let (head, body) = read_request(&mut stream);
            let (status, text) = match answer {
                None => {
                    assert!(head.starts_with("GET /v1/models"), "{head}");
                    (200, r#"{"data":[{"id":"made-up"}]}"#)
                }
                Some(answer) => {
                    asked.push(
                        serde_json::from_slice::<Value>(&body).unwrap()["max_tokens"].clone(),
                    );
                    answer
                }
            };
            let response = format!(
                "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
        asked
    });
    let client = Client::new(&format!("http://{address}"), "made-up");
    let mut request = request();
    request.model = "made-up".into();
    request.max_tokens = u64::MAX;
    let first = client.stream_turn(&request, &mut |_| {}).unwrap();
    assert_eq!(first.stop_reason, crate::protocol::StopReason::EndTurn);
    // The next request of this model asks with the allowed limit straight away.
    client.stream_turn(&request, &mut |_| {}).unwrap();
    let asked = server.join().unwrap();
    assert_eq!(
        asked,
        [json!(DEFAULT_OUTPUT_LIMIT), json!(32_000), json!(32_000)]
    );
}
