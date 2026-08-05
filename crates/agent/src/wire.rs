//! Anthropic Messages API wire adapter.
//!
//! Builds the `POST /v1/messages` JSON body and incrementally parses the
//! server-sent event stream into provider-neutral [`StreamEvent`]s.

use serde_json::{json, Value};

use crate::protocol::{
    ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, ToolChoice, Usage,
};

/// Serialize a [`ModelRequest`] into the Anthropic Messages request body.
///
/// Always sets `stream: true`. System prompt is a plain string when present.
/// `tool_choice` maps as `auto` / `none` / `any` (Anthropic's "required").
pub fn to_messages_request_body(req: &ModelRequest) -> Value {
    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": map_messages(&req.messages),
        "stream": true,
    });

    if let Some(system) = &req.system {
        body["system"] = Value::String(system.clone());
    }

    if !req.tools.is_empty() {
        body["tools"] = Value::Array(
            req.tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect(),
        );
    }

    body["tool_choice"] = match req.tool_choice {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Required => json!({"type": "any"}),
    };

    body
}

fn map_messages(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            json!({
                "role": role,
                "content": map_content(&m.content),
            })
        })
        .collect()
}

fn map_content(parts: &[ContentPart]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => json!({
                "type": "text",
                "text": text,
            }),
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": arguments,
            }),
            ContentPart::ToolResult {
                call_id,
                content,
                is_error,
            } => json!({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content,
                "is_error": is_error,
            }),
            ContentPart::Thinking { text } => json!({
                "type": "thinking",
                "thinking": text,
            }),
        })
        .collect()
}

/// One item produced by the SSE parser: a stream event, or a hard parse error
/// for a *recognized* event type whose payload is malformed.
#[derive(Debug, Clone, PartialEq)]
pub enum SseItem {
    Event(StreamEvent),
    /// `message_stop` — terminal marker with no stop_reason payload.
    MessageStop,
    /// Recognized event type with unparsable / malformed JSON payload.
    Malformed {
        event_type: String,
        detail: String,
    },
}

/// Incremental SSE parser for Anthropic Messages streams.
///
/// Call [`SseParser::feed_line`] with each complete UTF-8 line (trailing
/// newline already stripped). Call [`SseParser::finish`] after the last byte
/// to flush a trailing event that lacks a final blank line.
///
/// Recognized event types with malformed JSON become [`SseItem::Malformed`]
/// (caller must treat as stream error). Unknown event types are ignored.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Accumulated `event:` field for the current SSE event.
    event_name: Option<String>,
    /// Accumulated `data:` fields (joined with `\n` per the SSE spec).
    data_lines: Vec<String>,
    /// `usage.input_tokens` (and cache fields) from `message_start`, merged
    /// into the eventual [`StreamEvent::Finished`].
    pending_input_usage: Usage,
}

impl SseParser {
    /// Create an empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one complete line of the SSE body (newline already stripped; a
    /// trailing `\r` is accepted and dropped).
    ///
    /// Returns every complete item that became available from this line
    /// (usually zero or one).
    pub fn feed_line(&mut self, line: &str) -> Vec<SseItem> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let mut out = Vec::new();

        if line.is_empty() {
            if let Some(item) = self.dispatch_event() {
                out.push(item);
            }
            return out;
        }

        // SSE comments start with ':' — ignore.
        if line.starts_with(':') {
            return out;
        }

        if let Some(rest) = line.strip_prefix("event:") {
            self.event_name = Some(rest.trim_start().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            // One optional leading space after the colon is conventional.
            let data = rest.strip_prefix(' ').unwrap_or(rest);
            self.data_lines.push(data.to_string());
        }
        // Other fields (id:, retry:) are ignored.

        out
    }

    /// Flush any pending event after the body ends.
    pub fn finish(&mut self) -> Vec<SseItem> {
        let mut out = Vec::new();
        if let Some(item) = self.dispatch_event() {
            out.push(item);
        }
        out
    }

    fn dispatch_event(&mut self) -> Option<SseItem> {
        let name = self.event_name.take().unwrap_or_default();
        let data = self.data_lines.join("\n");
        self.data_lines.clear();

        if name.is_empty() && data.is_empty() {
            return None;
        }

        // `[DONE]` terminator used by some OpenAI-compatible proxies — ignore.
        if data.trim() == "[DONE]" {
            return None;
        }

        match name.as_str() {
            "message_start" => Some(parse_message_start(&data, &mut self.pending_input_usage)),
            "content_block_start" => parse_content_block_start(&data),
            "content_block_delta" => Some(parse_content_block_delta(&data)),
            "content_block_stop" => Some(parse_content_block_stop(&data)),
            "message_delta" => Some(parse_message_delta(&data, &self.pending_input_usage)),
            "message_stop" => Some(SseItem::MessageStop),
            "ping" => None,
            "error" => Some(parse_error_event(&data)),
            // Unknown event TYPES stay ignored.
            _ => None,
        }
    }
}

fn parse_json(data: &str) -> Result<Value, String> {
    serde_json::from_str(data).map_err(|e| e.to_string())
}

fn malformed(event_type: &str, detail: impl Into<String>) -> SseItem {
    SseItem::Malformed {
        event_type: event_type.to_string(),
        detail: detail.into(),
    }
}

fn parse_message_start(data: &str, pending: &mut Usage) -> SseItem {
    let v = match parse_json(data) {
        Ok(v) => v,
        Err(e) => return malformed("message_start", format!("invalid JSON: {e}")),
    };
    // Require a message object with a model field so garbage does not slip by.
    let message = match v.get("message") {
        Some(m) if m.is_object() => m,
        _ => {
            return malformed(
                "message_start",
                "missing or non-object message field".to_string(),
            )
        }
    };
    let model = message
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    *pending = Usage {
        input_tokens: message
            .pointer("/usage/input_tokens")
            .and_then(|n| n.as_u64())
            .unwrap_or(0),
        output_tokens: message
            .pointer("/usage/output_tokens")
            .and_then(|n| n.as_u64())
            .unwrap_or(0),
        cache_read_input_tokens: message
            .pointer("/usage/cache_read_input_tokens")
            .and_then(|n| n.as_u64())
            .unwrap_or(0),
        cache_creation_input_tokens: message
            .pointer("/usage/cache_creation_input_tokens")
            .and_then(|n| n.as_u64())
            .unwrap_or(0),
    };
    SseItem::Event(StreamEvent::Started { model })
}

/// Parse content_block_start. `None` = recognized & silent (text/thinking start).
fn parse_content_block_start(data: &str) -> Option<SseItem> {
    let v = match parse_json(data) {
        Ok(v) => v,
        Err(e) => {
            return Some(malformed(
                "content_block_start",
                format!("invalid JSON: {e}"),
            ))
        }
    };
    let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
    let Some(block) = v.get("content_block") else {
        return Some(malformed("content_block_start", "missing content_block"));
    };
    let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

    match block_type {
        // Text / thinking starts are silent; deltas carry the payload.
        "text" | "thinking" => None,
        "tool_use" => {
            let id = block
                .get("id")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let name = block
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            Some(SseItem::Event(StreamEvent::ToolCallStarted {
                index,
                id,
                name,
            }))
        }
        "" => Some(malformed(
            "content_block_start",
            "missing content_block.type",
        )),
        // Unknown block type inside a recognized event type: ignore silently.
        _ => None,
    }
}

fn parse_content_block_delta(data: &str) -> SseItem {
    let v = match parse_json(data) {
        Ok(v) => v,
        Err(e) => return malformed("content_block_delta", format!("invalid JSON: {e}")),
    };
    let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
    let Some(delta) = v.get("delta") else {
        return malformed("content_block_delta", "missing delta");
    };
    let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");

    match delta_type {
        "text_delta" => {
            let text = delta
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            SseItem::Event(StreamEvent::TextDelta { text })
        }
        "thinking_delta" => {
            let text = delta
                .get("thinking")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            SseItem::Event(StreamEvent::ThinkingDelta { text })
        }
        "input_json_delta" => {
            let json_fragment = delta
                .get("partial_json")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                index,
                json_fragment,
            })
        }
        "" => malformed("content_block_delta", "missing delta.type"),
        _ => {
            // Unknown delta type on a recognized event: ignore.
            SseItem::Event(StreamEvent::TextDelta {
                text: String::new(),
            })
        }
    }
}

fn parse_content_block_stop(data: &str) -> SseItem {
    let v = match parse_json(data) {
        Ok(v) => v,
        Err(e) => return malformed("content_block_stop", format!("invalid JSON: {e}")),
    };
    let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
    SseItem::Event(StreamEvent::BlockFinished { index })
}

fn parse_message_delta(data: &str, pending_input: &Usage) -> SseItem {
    let v = match parse_json(data) {
        Ok(v) => v,
        Err(e) => return malformed("message_delta", format!("invalid JSON: {e}")),
    };
    let stop_reason = v
        .pointer("/delta/stop_reason")
        .and_then(|s| s.as_str())
        .map(map_stop_reason)
        .unwrap_or(StopReason::Other(String::new()));

    let mut usage = *pending_input;
    if let Some(n) = v.pointer("/usage/input_tokens").and_then(|n| n.as_u64()) {
        if n > 0 {
            usage.input_tokens = n;
        }
    }
    if let Some(n) = v.pointer("/usage/output_tokens").and_then(|n| n.as_u64()) {
        usage.output_tokens = n;
    }
    if let Some(n) = v
        .pointer("/usage/cache_read_input_tokens")
        .and_then(|n| n.as_u64())
    {
        usage.cache_read_input_tokens = n;
    }
    if let Some(n) = v
        .pointer("/usage/cache_creation_input_tokens")
        .and_then(|n| n.as_u64())
    {
        usage.cache_creation_input_tokens = n;
    }

    SseItem::Event(StreamEvent::Finished { stop_reason, usage })
}

fn parse_error_event(data: &str) -> SseItem {
    match parse_json(data) {
        Ok(v) => {
            let message = v
                .pointer("/error/message")
                .or_else(|| v.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("error event")
                .to_string();
            SseItem::Event(StreamEvent::Error { message })
        }
        Err(e) => malformed("error", format!("invalid JSON: {e}")),
    }
}

pub(crate) fn map_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        other => StopReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ToolDefinition;

    fn sample_request() -> ModelRequest {
        ModelRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            system: Some("You are a coding agent.".to_string()),
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![ContentPart::Text {
                        text: "list files".to_string(),
                    }],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::ToolCall {
                        id: "toolu_01".to_string(),
                        name: "bash".to_string(),
                        arguments: json!({"command": "ls"}),
                    }],
                },
                Message {
                    role: Role::User,
                    content: vec![ContentPart::ToolResult {
                        call_id: "toolu_01".to_string(),
                        content: "Cargo.toml\nsrc\n".to_string(),
                        is_error: false,
                    }],
                },
            ],
            tools: vec![ToolDefinition {
                name: "bash".to_string(),
                description: "Run a shell command".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }),
            }],
            tool_choice: ToolChoice::Auto,
            max_tokens: 1024,
        }
    }

    #[test]
    fn request_body_golden() {
        let body = to_messages_request_body(&sample_request());
        let expected = json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are a coding agent.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "list files"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "bash",
                            "input": {"command": "ls"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "Cargo.toml\nsrc\n",
                            "is_error": false
                        }
                    ]
                }
            ],
            "tools": [
                {
                    "name": "bash",
                    "description": "Run a shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    }
                }
            ],
            "tool_choice": {"type": "auto"},
            "stream": true
        });
        assert_eq!(body, expected);
    }

    /// Full happy-path Anthropic SSE fixture: text then tool_use.
    const HAPPY_PATH_SSE: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-20250514\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":100,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"I'll list \"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"the files.\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"bash\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\": \\\"ls\\\"}\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":1}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":42}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

    fn expected_happy_path_events() -> Vec<StreamEvent> {
        vec![
            StreamEvent::Started {
                model: "claude-sonnet-4-20250514".to_string(),
            },
            StreamEvent::TextDelta {
                text: "I'll list ".to_string(),
            },
            StreamEvent::TextDelta {
                text: "the files.".to_string(),
            },
            StreamEvent::BlockFinished { index: 0 },
            StreamEvent::ToolCallStarted {
                index: 1,
                id: "toolu_01".to_string(),
                name: "bash".to_string(),
            },
            StreamEvent::ToolCallArgumentsDelta {
                index: 1,
                json_fragment: "{\"command\"".to_string(),
            },
            StreamEvent::ToolCallArgumentsDelta {
                index: 1,
                json_fragment: ": \"ls\"}".to_string(),
            },
            StreamEvent::BlockFinished { index: 1 },
            StreamEvent::Finished {
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 42,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            },
        ]
    }

    fn parse_all(fixture: &str) -> Result<Vec<StreamEvent>, String> {
        let mut p = SseParser::new();
        let mut events = Vec::new();
        for line in fixture.split_inclusive('\n') {
            let line = line.strip_suffix('\n').unwrap_or(line);
            for item in p.feed_line(line) {
                match item {
                    SseItem::Event(ev) => events.push(ev),
                    SseItem::MessageStop => {}
                    SseItem::Malformed { event_type, detail } => {
                        return Err(format!("malformed {event_type}: {detail}"));
                    }
                }
            }
        }
        for item in p.finish() {
            match item {
                SseItem::Event(ev) => events.push(ev),
                SseItem::MessageStop => {}
                SseItem::Malformed { event_type, detail } => {
                    return Err(format!("malformed {event_type}: {detail}"));
                }
            }
        }
        Ok(events)
    }

    #[test]
    fn sse_happy_path_one_chunk() {
        let events = parse_all(HAPPY_PATH_SSE).expect("parse");
        assert_eq!(events, expected_happy_path_events());
    }

    #[test]
    fn sse_happy_path_line_by_line() {
        let mut p = SseParser::new();
        let mut events = Vec::new();
        for line in HAPPY_PATH_SSE.lines() {
            for item in p.feed_line(line) {
                if let SseItem::Event(ev) = item {
                    events.push(ev);
                }
            }
        }
        for item in p.finish() {
            if let SseItem::Event(ev) = item {
                events.push(ev);
            }
        }
        assert_eq!(events, expected_happy_path_events());
    }

    #[test]
    fn sse_input_json_delta_fragments_assemble() {
        let events = parse_all(HAPPY_PATH_SSE).expect("parse");
        let mut acc = String::new();
        for ev in &events {
            if let StreamEvent::ToolCallArgumentsDelta {
                index,
                json_fragment,
            } = ev
            {
                assert_eq!(*index, 1);
                acc.push_str(json_fragment);
            }
        }
        let parsed: Value = serde_json::from_str(&acc).expect("assembled JSON");
        assert_eq!(parsed, json!({"command": "ls"}));
    }

    #[test]
    fn sse_unknown_event_type_ignored() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: something_novel
data: {\"foo\":1}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}

";
        let events = parse_all(fixture).expect("parse");
        assert_eq!(
            events,
            vec![
                StreamEvent::Started {
                    model: "m".to_string()
                },
                StreamEvent::TextDelta {
                    text: "hi".to_string()
                },
            ]
        );
    }

    #[test]
    fn sse_error_event_surfaces() {
        let fixture = "\
event: error
data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}

";
        let events = parse_all(fixture).expect("parse");
        assert_eq!(
            events,
            vec![StreamEvent::Error {
                message: "Overloaded".to_string()
            }]
        );
    }

    #[test]
    fn sse_recognized_event_with_garbage_json_is_malformed() {
        let fixture = "\
event: message_start
data: {not-json

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_start"), "err={err}");
        assert!(
            err.contains("invalid JSON") || err.contains("malformed"),
            "err={err}"
        );
    }

    #[test]
    fn sse_content_block_delta_garbage_is_malformed() {
        let fixture = "\
event: content_block_delta
data: not-json-at-all

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
    }

    #[test]
    fn tool_choice_required_maps_to_any() {
        let mut req = sample_request();
        req.tool_choice = ToolChoice::Required;
        let body = to_messages_request_body(&req);
        assert_eq!(body["tool_choice"], json!({"type": "any"}));
    }

    #[test]
    fn tool_choice_none() {
        let mut req = sample_request();
        req.tool_choice = ToolChoice::None;
        let body = to_messages_request_body(&req);
        assert_eq!(body["tool_choice"], json!({"type": "none"}));
    }
}
