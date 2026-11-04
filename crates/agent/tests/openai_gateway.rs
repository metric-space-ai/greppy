//! In-process OpenAI Responses gateway. No external network.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use greppy_agent::{
    ApiFormat, ApiFormatSetting, Client, ClientError, ContentPart, Message, ModelRequest, Role,
    StopReason, StreamEvent, ToolChoice, ToolDefinition,
};
use serde_json::{json, Value};

fn request() -> ModelRequest {
    ModelRequest {
        model: "fixture-model".into(),
        system: Some("be brief".into()),
        messages: vec![
            Message {
                role: Role::User,
                content: vec![ContentPart::Text {
                    text: "look".into(),
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall {
                    id: "call_prev".into(),
                    name: "greppy".into(),
                    arguments: json!({"args": ["search", "clamp"]}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentPart::Image {
                        media_type: "image/png".into(),
                        data: "AAAA".into(),
                    },
                    ContentPart::ToolResult {
                        call_id: "call_prev".into(),
                        content: "screenshot attached".into(),
                        is_error: false,
                    },
                ],
            },
        ],
        tools: vec![ToolDefinition {
            name: "greppy".into(),
            description: "navigate the repository".into(),
            input_schema: json!({"type": "object", "properties": {"args": {"type": "array"}}}),
        }],
        tool_choice: ToolChoice::Auto,
        max_tokens: 32,
    }
}

struct CapturedRequest {
    path: String,
    authorization: String,
    x_api_key: String,
    body: Value,
}

fn read_http(stream: &mut TcpStream) -> CapturedRequest {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(3)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).unwrap();
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_string();
    let mut authorization = String::new();
    let mut x_api_key = String::new();
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        assert!(
            reader.read_line(&mut line).unwrap() > 0,
            "truncated headers"
        );
        if line == "\r\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            length = value.trim().parse().unwrap();
        } else if let Some(value) = line.strip_prefix("Authorization:") {
            authorization = value.trim().to_string();
        } else if let Some(value) = lower.strip_prefix("x-api-key:") {
            x_api_key = value.trim().to_string();
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    CapturedRequest {
        path,
        authorization,
        x_api_key,
        body,
    }
}

fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}

fn spawn(
    responses: Vec<(&'static str, &'static str, &'static str)>,
) -> (String, thread::JoinHandle<Vec<CapturedRequest>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let mut captured = Vec::new();
        for (status, content_type, body) in responses {
            let (mut stream, _) = listener.accept().unwrap();
            captured.push(read_http(&mut stream));
            write_response(&mut stream, status, content_type, body);
        }
        captured
    });
    (format!("http://{address}"), server)
}

fn openai_client(base: &str) -> Client {
    Client::new(base, "fixture-model")
        .with_api_format(ApiFormatSetting::Explicit(ApiFormat::OpenAi))
        .with_api_key("sekrit-key")
}

const TOOL_SSE: &str = concat!(
    "event: response.created\n",
    "data: {\"type\":\"response.created\",\"response\":{\"model\":\"fixture-model\"}}\n\n",
    "event: response.output_text.delta\n",
    "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hi\"}\n\n",
    "event: response.output_item.added\n",
    "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"greppy\",\"arguments\":\"\"}}\n\n",
    "event: response.function_call_arguments.delta\n",
    "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"delta\":\"{\\\"args\\\":[\\\"who-calls\\\",\\\"clamp\\\"]}\"}\n\n",
    "event: response.output_item.added\n",
    "data: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_b\",\"name\":\"greppy\",\"arguments\":\"\"}}\n\n",
    "event: response.function_call_arguments.delta\n",
    "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":2,\"delta\":\"{\\\"args\\\":[\\\"brief\\\",\\\"clamp\\\"]}\"}\n\n",
    "event: response.output_item.done\n",
    "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\"}}\n\n",
    "event: response.output_item.done\n",
    "data: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"type\":\"function_call\"}}\n\n",
    "event: response.completed\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":11,\"output_tokens\":7,\"input_tokens_details\":{\"cached_tokens\":3}}}}\n\n",
);

#[test]
fn responses_round_trip_builds_tools_and_parses_parallel_calls() {
    let (base, server) = spawn(vec![("200 OK", "text/event-stream", TOOL_SSE)]);
    let client = openai_client(&base);
    let mut events = Vec::new();
    let turn = client
        .stream_turn(&request(), &mut |event| events.push(event))
        .unwrap_or_else(|error| panic!("stream failed: {error}\nevents={events:#?}"));
    let captured = server.join().unwrap();
    assert_eq!(captured.len(), 1);
    let got = &captured[0];
    assert_eq!(got.path, "/v1/responses");
    assert_eq!(got.authorization, "Bearer sekrit-key");
    assert!(got.x_api_key.is_empty(), "openai must not send x-api-key");
    let rendered = got.body.to_string();
    assert!(!rendered.contains("sekrit-key"));
    assert_eq!(got.body["model"], "fixture-model");
    assert_eq!(got.body["instructions"], "be brief");
    assert_eq!(got.body["stream"], true);
    assert_eq!(got.body["store"], false);
    assert_eq!(got.body["max_output_tokens"], 32);
    assert!(got.body.get("max_tokens").is_none());
    assert!(got.body.get("messages").is_none());
    assert_eq!(got.body["tool_choice"], "auto");
    assert_eq!(got.body["tools"][0]["type"], "function");
    assert_eq!(got.body["tools"][0]["name"], "greppy");
    let input = got.body["input"].as_array().unwrap();
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"][0]["type"], "input_text");
    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["call_id"], "call_prev");
    assert_eq!(input[2]["type"], "function_call_output");
    assert_eq!(input[2]["call_id"], "call_prev");
    assert_eq!(input[2]["output"], "screenshot attached");
    assert_eq!(input[3]["content"][1]["type"], "input_image");
    assert_eq!(
        input[3]["content"][1]["image_url"],
        "data:image/png;base64,AAAA"
    );

    assert_eq!(turn.stop_reason, StopReason::ToolUse);
    assert_eq!(turn.usage.input_tokens, 11);
    assert_eq!(turn.usage.output_tokens, 7);
    assert_eq!(turn.usage.cache_read_input_tokens, 3);
    assert_eq!(
        turn.message.content,
        vec![
            ContentPart::Text { text: "Hi".into() },
            ContentPart::ToolCall {
                id: "call_a".into(),
                name: "greppy".into(),
                arguments: json!({"args": ["who-calls", "clamp"]}),
            },
            ContentPart::ToolCall {
                id: "call_b".into(),
                name: "greppy".into(),
                arguments: json!({"args": ["brief", "clamp"]}),
            },
        ]
    );
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ToolCallStarted { name, .. } if name == "greppy"
    )));
}

#[test]
fn error_json_is_an_http_failure_and_vision_rejection_retries_without_images() {
    let error = r#"{"error":{"message":"model does not support image input","type":"invalid_request_error"}}"#;
    let ok = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"model\":\"fixture-model\",\"delta\":\"text only\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n",
    );
    let (base, server) = spawn(vec![
        ("400 Bad Request", "application/json", error),
        ("200 OK", "text/event-stream", ok),
    ]);
    let client = openai_client(&base);
    let turn = client.stream_turn(&request(), &mut |_| {}).unwrap();
    let captured = server.join().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(captured[0].body.to_string().contains("input_image"));
    let retry = captured[1].body.to_string();
    assert!(!retry.contains("input_image"), "{retry}");
    assert!(!retry.contains("AAAA"), "{retry}");
    assert!(retry.contains("model does not accept image input"));
    assert_eq!(
        turn.message.content,
        vec![ContentPart::Text {
            text: "text only".into()
        }]
    );
    assert_eq!(turn.stop_reason, StopReason::EndTurn);
}

#[test]
fn non_vision_error_is_returned_and_not_retried() {
    let (base, server) = spawn(vec![(
        "401 Unauthorized",
        "application/json",
        r#"{"error":{"message":"invalid api key"}}"#,
    )]);
    let client = openai_client(&base);
    let error = client.stream_turn(&request(), &mut |_| {}).unwrap_err();
    match error {
        ClientError::Http { status: 401, body } => assert!(body.contains("invalid api key")),
        other => panic!("{other:?}"),
    }
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn path_auto_detects_responses() {
    let (base, server) = spawn(vec![(
        "200 OK",
        "text/event-stream",
        concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"model\":\"fixture-model\",\"delta\":\"ok\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
        ),
    )]);
    let client = Client::new(&format!("{base}/v1/responses"), "fixture-model");
    assert_eq!(client.resolved_format(), ApiFormat::OpenAi);
    let turn = client
        .stream_turn(
            &ModelRequest {
                model: "fixture-model".into(),
                system: None,
                messages: vec![],
                tools: vec![],
                tool_choice: ToolChoice::None,
                max_tokens: 8,
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(server.join().unwrap()[0].path, "/v1/responses");
    assert_eq!(
        turn.message.content,
        vec![ContentPart::Text { text: "ok".into() }]
    );
}
