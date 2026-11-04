//! In-process OpenAI Chat Completions gateway. No external network.

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
    "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"greppy\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"args\\\":[\\\"who-calls\\\",\\\"clamp\\\"]}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"greppy\",\"arguments\":\"{\\\"args\\\":[\\\"brief\\\",\\\"clamp\\\"]}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
    "data: [DONE]\n\n",
);

#[test]
fn chat_completions_round_trip_builds_tools_and_parses_parallel_calls() {
    let (base, server) = spawn(vec![("200 OK", "text/event-stream", TOOL_SSE)]);
    let client = openai_client(&base);
    let mut events = Vec::new();
    let turn = client
        .stream_turn(&request(), &mut |event| events.push(event))
        .unwrap();
    let captured = server.join().unwrap();
    assert_eq!(captured.len(), 1);
    let got = &captured[0];
    assert_eq!(got.path, "/v1/chat/completions");
    assert_eq!(got.authorization, "Bearer sekrit-key");
    assert!(got.x_api_key.is_empty(), "openai must not send x-api-key");
    let rendered = got.body.to_string();
    assert!(!rendered.contains("sekrit-key"));
    assert_eq!(got.body["model"], "fixture-model");
    assert_eq!(got.body["stream"], true);
    assert_eq!(got.body["max_tokens"], 32);
    assert_eq!(got.body["tool_choice"], "auto");
    assert_eq!(got.body["tools"][0]["function"]["name"], "greppy");
    let messages = got.body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call_prev");
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call_prev");
    assert_eq!(messages[3]["content"], "screenshot attached");
    assert_eq!(
        messages[4]["content"][1]["image_url"]["url"],
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
        "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":\"text only\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    let (base, server) = spawn(vec![
        ("400 Bad Request", "application/json", error),
        ("200 OK", "text/event-stream", ok),
    ]);
    let client = openai_client(&base);
    let turn = client.stream_turn(&request(), &mut |_| {}).unwrap();
    let captured = server.join().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(captured[0].body.to_string().contains("image_url"));
    let retry = captured[1].body.to_string();
    assert!(!retry.contains("image_url"), "{retry}");
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
fn path_auto_detects_chat_completions() {
    let (base, server) = spawn(vec![(
        "200 OK",
        "text/event-stream",
        concat!(
            "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ),
    )]);
    let client = Client::new(&format!("{base}/v1/chat/completions"), "fixture-model");
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
    assert_eq!(server.join().unwrap()[0].path, "/v1/chat/completions");
    assert_eq!(
        turn.message.content,
        vec![ContentPart::Text { text: "ok".into() }]
    );
}
