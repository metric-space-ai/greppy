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

/// One item produced by the SSE parser: a stream event, a non-emitting but
/// counted record, or a hard parse / protocol error.
#[derive(Debug, Clone, PartialEq)]
pub enum SseItem {
    Event(StreamEvent),
    /// `message_stop` — terminal marker with no stop_reason payload.
    MessageStop,
    /// Completed SSE record that intentionally produces no model event
    /// (ping, silent block starts, unknown types). Still counted toward the
    /// stream event cap.
    Ignored,
    /// Recognized event type with unparsable / malformed JSON payload, or a
    /// protocol state-machine violation.
    Malformed {
        event_type: String,
        detail: String,
    },
}

/// Declared content-block kind while a block is open.
///
/// Used to reject cross-kind deltas (`text_delta` into a thinking block, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse,
    /// Forward-compat unknown `content_block.type` — stop may still close it.
    Unknown,
}

/// Ordered protocol states for an Anthropic Messages SSE stream.
///
/// `AwaitingStart → InMessage → (InBlock ↔ InMessage) → Terminal → Stopped`.
///
/// `message_delta` with a stop_reason moves to `Terminal`. `message_stop` is
/// legal exactly once (from `InMessage` or `Terminal`) and moves to `Stopped`.
/// Any recognized event after `Stopped` is a protocol error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ProtocolState {
    #[default]
    AwaitingStart,
    InMessage,
    InBlock {
        index: usize,
        kind: BlockKind,
    },
    /// After `message_delta` with stop_reason; trailing `message_stop` still ok.
    Terminal,
    /// After `message_stop`; stream is fully closed.
    Stopped,
}

/// Incremental SSE parser for Anthropic Messages streams.
///
/// Call [`SseParser::feed_line`] with each complete UTF-8 line (trailing
/// newline already stripped). Call [`SseParser::finish`] after the last byte
/// to flush a trailing event that lacks a final blank line.
///
/// Enforces an explicit ordered state machine. Recognized event types with
/// malformed JSON or illegal ordering become [`SseItem::Malformed`] (caller
/// must treat as stream error). Unknown event types are ignored (forward
/// compatibility) only while the stream is still open.
///
/// Every completed SSE record (whether or not it emits a [`StreamEvent`])
/// yields an [`SseItem`] so callers can apply an honest event-count cap.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Accumulated `event:` field for the current SSE event.
    event_name: Option<String>,
    /// Accumulated `data:` fields (joined with `\n` per the SSE spec).
    data_lines: Vec<String>,
    /// `usage.input_tokens` (and cache fields) from `message_start`, merged
    /// into the eventual [`StreamEvent::Finished`].
    pending_input_usage: Usage,
    /// Protocol state machine.
    state: ProtocolState,
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
    /// (usually zero or one). Every completed SSE record produces an item
    /// (including [`SseItem::Ignored`]).
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
            return Some(SseItem::Ignored);
        }

        // After Stopped the stream is fully closed: any recognized event (including
        // a second message_stop) is a protocol error. ping/unknown keep-alives
        // remain ignored for forward compatibility.
        if matches!(self.state, ProtocolState::Stopped) {
            return Some(match name.as_str() {
                "ping" => SseItem::Ignored,
                "message_start"
                | "content_block_start"
                | "content_block_delta"
                | "content_block_stop"
                | "message_delta"
                | "message_stop"
                | "error" => malformed(&name, "event after stream stopped".to_string()),
                _ => SseItem::Ignored,
            });
        }

        // After Terminal, only a trailing message_stop (Anthropic's normal trailer
        // after message_delta) and unknown/ping keep-alives are legal. Any other
        // recognized event is a stream error.
        if matches!(self.state, ProtocolState::Terminal) {
            return Some(match name.as_str() {
                "message_stop" => self.handle_message_stop(&data),
                "ping" => SseItem::Ignored,
                "message_start"
                | "content_block_start"
                | "content_block_delta"
                | "content_block_stop"
                | "message_delta"
                | "error" => malformed(&name, "event after terminal stop".to_string()),
                // Unknown types remain ignored (forward compatibility).
                _ => SseItem::Ignored,
            });
        }

        match name.as_str() {
            "message_start" => Some(self.handle_message_start(&data)),
            "content_block_start" => Some(self.handle_content_block_start(&data)),
            "content_block_delta" => Some(self.handle_content_block_delta(&data)),
            "content_block_stop" => Some(self.handle_content_block_stop(&data)),
            "message_delta" => Some(self.handle_message_delta(&data)),
            "message_stop" => Some(self.handle_message_stop(&data)),
            "ping" => Some(SseItem::Ignored),
            "error" => Some(parse_error_event(&data)),
            // Unknown event TYPES stay ignored (still counted via Ignored).
            _ => Some(SseItem::Ignored),
        }
    }

    fn handle_message_start(&mut self, data: &str) -> SseItem {
        if !matches!(self.state, ProtocolState::AwaitingStart) {
            return malformed(
                "message_start",
                "duplicate or out-of-order message_start".to_string(),
            );
        }
        let v = match parse_json(data) {
            Ok(v) => v,
            Err(e) => return malformed("message_start", format!("invalid JSON: {e}")),
        };
        if let Err(e) = expect_type(&v, "message_start") {
            return malformed("message_start", e);
        }
        let message = match v.get("message") {
            Some(m) if m.is_object() => m,
            _ => {
                return malformed(
                    "message_start",
                    "missing or non-object message field".to_string(),
                )
            }
        };
        // Require a real string model — no "" substitution for missing/non-string.
        let model = match message.get("model").and_then(|m| m.as_str()) {
            Some(m) => m.to_string(),
            None => {
                return malformed(
                    "message_start",
                    "missing or non-string message.model".to_string(),
                )
            }
        };
        self.pending_input_usage = Usage {
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
        self.state = ProtocolState::InMessage;
        SseItem::Event(StreamEvent::Started { model })
    }

    fn handle_content_block_start(&mut self, data: &str) -> SseItem {
        match self.state {
            ProtocolState::InMessage => {}
            ProtocolState::InBlock { .. } => {
                return malformed(
                    "content_block_start",
                    "content_block_start while a block is open".to_string(),
                )
            }
            ProtocolState::AwaitingStart => {
                return malformed(
                    "content_block_start",
                    "content_block_start before message_start".to_string(),
                )
            }
            ProtocolState::Terminal => {
                return malformed(
                    "content_block_start",
                    "event after terminal stop".to_string(),
                )
            }
            ProtocolState::Stopped => {
                return malformed(
                    "content_block_start",
                    "event after stream stopped".to_string(),
                )
            }
        }
        let v = match parse_json(data) {
            Ok(v) => v,
            Err(e) => return malformed("content_block_start", format!("invalid JSON: {e}")),
        };
        if let Err(e) = expect_type(&v, "content_block_start") {
            return malformed("content_block_start", e);
        }
        let index = match v.get("index").and_then(|i| i.as_u64()) {
            Some(i) => i as usize,
            None => return malformed("content_block_start", "missing content_block_start.index"),
        };
        let Some(block) = v.get("content_block") else {
            return malformed("content_block_start", "missing content_block");
        };
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match block_type {
            // Text / thinking starts are silent; deltas carry the payload.
            "text" => {
                self.state = ProtocolState::InBlock {
                    index,
                    kind: BlockKind::Text,
                };
                SseItem::Ignored
            }
            "thinking" => {
                self.state = ProtocolState::InBlock {
                    index,
                    kind: BlockKind::Thinking,
                };
                SseItem::Ignored
            }
            "tool_use" => {
                // Require real string id/name — no "" substitution.
                let id = match block.get("id").and_then(|s| s.as_str()) {
                    Some(s) => s.to_string(),
                    None => {
                        return malformed(
                            "content_block_start",
                            "missing or non-string content_block.id".to_string(),
                        )
                    }
                };
                let name = match block.get("name").and_then(|s| s.as_str()) {
                    Some(s) => s.to_string(),
                    None => {
                        return malformed(
                            "content_block_start",
                            "missing or non-string content_block.name".to_string(),
                        )
                    }
                };
                self.state = ProtocolState::InBlock {
                    index,
                    kind: BlockKind::ToolUse,
                };
                SseItem::Event(StreamEvent::ToolCallStarted { index, id, name })
            }
            "" => malformed("content_block_start", "missing content_block.type"),
            // Unknown block type: keep the block open so stop can close it.
            _ => {
                self.state = ProtocolState::InBlock {
                    index,
                    kind: BlockKind::Unknown,
                };
                SseItem::Ignored
            }
        }
    }

    fn handle_content_block_delta(&mut self, data: &str) -> SseItem {
        let (open_index, open_kind) = match self.state {
            ProtocolState::InBlock { index, kind } => (index, kind),
            ProtocolState::InMessage | ProtocolState::AwaitingStart => {
                return malformed(
                    "content_block_delta",
                    "content_block_delta without an open block".to_string(),
                )
            }
            ProtocolState::Terminal => {
                return malformed(
                    "content_block_delta",
                    "event after terminal stop".to_string(),
                )
            }
            ProtocolState::Stopped => {
                return malformed(
                    "content_block_delta",
                    "event after stream stopped".to_string(),
                )
            }
        };
        let v = match parse_json(data) {
            Ok(v) => v,
            Err(e) => return malformed("content_block_delta", format!("invalid JSON: {e}")),
        };
        if let Err(e) = expect_type(&v, "content_block_delta") {
            return malformed("content_block_delta", e);
        }
        let index = match v.get("index").and_then(|i| i.as_u64()) {
            Some(i) => i as usize,
            None => return malformed("content_block_delta", "missing content_block_delta.index"),
        };
        if index != open_index {
            return malformed(
                "content_block_delta",
                format!("index mismatch: open={open_index}, delta={index}"),
            );
        }
        let Some(delta) = v.get("delta") else {
            return malformed("content_block_delta", "missing delta");
        };
        let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match delta_type {
            "text_delta" => {
                if open_kind != BlockKind::Text {
                    return malformed(
                        "content_block_delta",
                        format!("text_delta into non-text block ({open_kind:?})"),
                    );
                }
                // Require a present string field; empty string is legal.
                let text = match delta.get("text").and_then(|t| t.as_str()) {
                    Some(t) => t.to_string(),
                    None => {
                        return malformed(
                            "content_block_delta",
                            "missing or non-string delta.text".to_string(),
                        )
                    }
                };
                SseItem::Event(StreamEvent::TextDelta { text })
            }
            "thinking_delta" => {
                if open_kind != BlockKind::Thinking {
                    return malformed(
                        "content_block_delta",
                        format!("thinking_delta into non-thinking block ({open_kind:?})"),
                    );
                }
                let text = match delta.get("thinking").and_then(|t| t.as_str()) {
                    Some(t) => t.to_string(),
                    None => {
                        return malformed(
                            "content_block_delta",
                            "missing or non-string delta.thinking".to_string(),
                        )
                    }
                };
                SseItem::Event(StreamEvent::ThinkingDelta { text })
            }
            "input_json_delta" => {
                if open_kind != BlockKind::ToolUse {
                    return malformed(
                        "content_block_delta",
                        format!("input_json_delta into non-tool_use block ({open_kind:?})"),
                    );
                }
                let json_fragment = match delta.get("partial_json").and_then(|t| t.as_str()) {
                    Some(t) => t.to_string(),
                    None => {
                        return malformed(
                            "content_block_delta",
                            "missing or non-string delta.partial_json".to_string(),
                        )
                    }
                };
                SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                    index,
                    json_fragment,
                })
            }
            "" => malformed("content_block_delta", "missing delta.type"),
            // Unknown delta type on a recognized event: ignore (counted).
            _ => SseItem::Ignored,
        }
    }

    fn handle_content_block_stop(&mut self, data: &str) -> SseItem {
        let open_index = match self.state {
            ProtocolState::InBlock { index, .. } => index,
            ProtocolState::InMessage | ProtocolState::AwaitingStart => {
                return malformed(
                    "content_block_stop",
                    "content_block_stop without an open block".to_string(),
                )
            }
            ProtocolState::Terminal => {
                return malformed(
                    "content_block_stop",
                    "event after terminal stop".to_string(),
                )
            }
            ProtocolState::Stopped => {
                return malformed(
                    "content_block_stop",
                    "event after stream stopped".to_string(),
                )
            }
        };
        let v = match parse_json(data) {
            Ok(v) => v,
            Err(e) => return malformed("content_block_stop", format!("invalid JSON: {e}")),
        };
        if let Err(e) = expect_type(&v, "content_block_stop") {
            return malformed("content_block_stop", e);
        }
        let index = match v.get("index").and_then(|i| i.as_u64()) {
            Some(i) => i as usize,
            None => return malformed("content_block_stop", "missing content_block_stop.index"),
        };
        if index != open_index {
            return malformed(
                "content_block_stop",
                format!("index mismatch: open={open_index}, stop={index}"),
            );
        }
        self.state = ProtocolState::InMessage;
        SseItem::Event(StreamEvent::BlockFinished { index })
    }

    fn handle_message_delta(&mut self, data: &str) -> SseItem {
        match self.state {
            ProtocolState::InMessage => {}
            ProtocolState::InBlock { .. } => {
                return malformed(
                    "message_delta",
                    "terminal event while a content block is open".to_string(),
                )
            }
            ProtocolState::AwaitingStart => {
                return malformed(
                    "message_delta",
                    "terminal event before message_start".to_string(),
                )
            }
            ProtocolState::Terminal => {
                return malformed("message_delta", "event after terminal stop".to_string())
            }
            ProtocolState::Stopped => {
                return malformed("message_delta", "event after stream stopped".to_string())
            }
        }
        let v = match parse_json(data) {
            Ok(v) => v,
            Err(e) => return malformed("message_delta", format!("invalid JSON: {e}")),
        };
        if let Err(e) = expect_type(&v, "message_delta") {
            return malformed("message_delta", e);
        }
        // Require a non-null string stop_reason — `{}` / missing is a stream error,
        // not a synthetic terminal.
        let stop_reason = match v.pointer("/delta/stop_reason") {
            Some(s) if s.is_null() => {
                return malformed("message_delta", "delta.stop_reason is null".to_string())
            }
            Some(s) => match s.as_str() {
                Some(r) => map_stop_reason(r),
                None => {
                    return malformed(
                        "message_delta",
                        "delta.stop_reason is not a string".to_string(),
                    )
                }
            },
            None => return malformed("message_delta", "missing delta.stop_reason".to_string()),
        };

        let mut usage = self.pending_input_usage;
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

        self.state = ProtocolState::Terminal;
        SseItem::Event(StreamEvent::Finished { stop_reason, usage })
    }

    fn handle_message_stop(&mut self, data: &str) -> SseItem {
        match self.state {
            // Legal exactly once: from InMessage (stop without message_delta)
            // or Terminal (Anthropic trailer after message_delta).
            ProtocolState::InMessage | ProtocolState::Terminal => {}
            ProtocolState::InBlock { .. } => {
                return malformed(
                    "message_stop",
                    "terminal event while a content block is open".to_string(),
                )
            }
            ProtocolState::AwaitingStart => {
                return malformed(
                    "message_stop",
                    "terminal event before message_start".to_string(),
                )
            }
            ProtocolState::Stopped => {
                return malformed("message_stop", "event after stream stopped".to_string())
            }
        }
        let v = match parse_json(data) {
            Ok(v) => v,
            Err(e) => return malformed("message_stop", format!("invalid JSON: {e}")),
        };
        if !v.is_object() {
            return malformed("message_stop", "payload must be an object".to_string());
        }
        if let Err(e) = expect_type(&v, "message_stop") {
            return malformed("message_stop", e);
        }
        self.state = ProtocolState::Stopped;
        SseItem::MessageStop
    }
}

fn parse_json(data: &str) -> Result<Value, String> {
    serde_json::from_str(data).map_err(|e| e.to_string())
}

fn expect_type(v: &Value, expected: &str) -> Result<(), String> {
    match v.get("type").and_then(|t| t.as_str()) {
        Some(t) if t == expected => Ok(()),
        Some(t) => Err(format!("expected type {expected:?}, got {t:?}")),
        None => Err(format!("missing type field (expected {expected:?})")),
    }
}

fn malformed(event_type: &str, detail: impl Into<String>) -> SseItem {
    SseItem::Malformed {
        event_type: event_type.to_string(),
        detail: detail.into(),
    }
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
                    SseItem::MessageStop | SseItem::Ignored => {}
                    SseItem::Malformed { event_type, detail } => {
                        return Err(format!("malformed {event_type}: {detail}"));
                    }
                }
            }
        }
        for item in p.finish() {
            match item {
                SseItem::Event(ev) => events.push(ev),
                SseItem::MessageStop | SseItem::Ignored => {}
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

    /// Happy path covering thinking + text + two tool_use blocks.
    const HAPPY_PATH_THINKING_TEXT_TWO_TOOLS: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-20250514\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":50,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan it\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":1}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"bash\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"ls\\\"}\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":2}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_b\",\"name\":\"greppy\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"args\\\":[]}\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":3}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":20}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

    #[test]
    fn sse_happy_path_thinking_text_two_tools() {
        let events = parse_all(HAPPY_PATH_THINKING_TEXT_TWO_TOOLS).expect("parse");
        assert_eq!(
            events,
            vec![
                StreamEvent::Started {
                    model: "claude-sonnet-4-20250514".to_string(),
                },
                StreamEvent::ThinkingDelta {
                    text: "plan it".to_string(),
                },
                StreamEvent::BlockFinished { index: 0 },
                StreamEvent::TextDelta {
                    text: "ok".to_string(),
                },
                StreamEvent::BlockFinished { index: 1 },
                StreamEvent::ToolCallStarted {
                    index: 2,
                    id: "toolu_a".to_string(),
                    name: "bash".to_string(),
                },
                StreamEvent::ToolCallArgumentsDelta {
                    index: 2,
                    json_fragment: "{\"command\":\"ls\"}".to_string(),
                },
                StreamEvent::BlockFinished { index: 2 },
                StreamEvent::ToolCallStarted {
                    index: 3,
                    id: "toolu_b".to_string(),
                    name: "greppy".to_string(),
                },
                StreamEvent::ToolCallArgumentsDelta {
                    index: 3,
                    json_fragment: "{\"args\":[]}".to_string(),
                },
                StreamEvent::BlockFinished { index: 3 },
                StreamEvent::Finished {
                    stop_reason: StopReason::ToolUse,
                    usage: Usage {
                        input_tokens: 50,
                        output_tokens: 20,
                        cache_read_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                    },
                },
            ]
        );
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

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}

event: message_stop
data: {\"type\":\"message_stop\"}

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
                StreamEvent::BlockFinished { index: 0 },
                StreamEvent::Finished {
                    stop_reason: StopReason::EndTurn,
                    usage: Usage {
                        input_tokens: 0,
                        output_tokens: 1,
                        cache_read_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                    },
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
        // Garbage after a valid open block → invalid JSON, not "no open block".
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: not-json-at-all

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(err.contains("invalid JSON"), "err={err}");
    }

    #[test]
    fn sse_message_delta_empty_object_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: message_delta
data: {}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_delta"), "err={err}");
        assert!(
            err.contains("stop_reason") || err.contains("type"),
            "err={err}"
        );
    }

    #[test]
    fn sse_garbage_message_stop_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: message_stop
data: not-an-object

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_stop"), "err={err}");
    }

    #[test]
    fn sse_terminal_before_start_is_malformed() {
        let fixture = "\
event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_delta"), "err={err}");
        assert!(err.contains("before message_start"), "err={err}");
    }

    #[test]
    fn sse_duplicate_message_start_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m2\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_start"), "err={err}");
        assert!(
            err.contains("duplicate") || err.contains("out-of-order"),
            "err={err}"
        );
    }

    #[test]
    fn sse_truncated_text_block_then_message_stop_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"half\"}}

event: message_stop
data: {\"type\":\"message_stop\"}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_stop"), "err={err}");
        assert!(err.contains("open"), "err={err}");
    }

    #[test]
    fn sse_truncated_tool_use_block_then_message_stop_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"bash\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"c\"}}

event: message_stop
data: {\"type\":\"message_stop\"}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_stop"), "err={err}");
        assert!(err.contains("open"), "err={err}");
    }

    #[test]
    fn sse_content_block_delta_without_start_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(err.contains("without an open block"), "err={err}");
    }

    #[test]
    fn sse_message_start_missing_model_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_start"), "err={err}");
        assert!(err.contains("model"), "err={err}");
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

    // --- WP10 residual strictness: block kinds, required fields, Stopped ---

    #[test]
    fn sse_thinking_delta_into_text_block_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"nope\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(
            err.contains("thinking_delta") || err.contains("non-thinking"),
            "err={err}"
        );
    }

    #[test]
    fn sse_text_delta_into_thinking_block_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"nope\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(
            err.contains("text_delta") || err.contains("non-text"),
            "err={err}"
        );
    }

    #[test]
    fn sse_input_json_delta_into_text_block_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(
            err.contains("input_json_delta") || err.contains("non-tool_use"),
            "err={err}"
        );
    }

    #[test]
    fn sse_text_delta_into_tool_use_block_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"bash\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"nope\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(
            err.contains("text_delta") || err.contains("non-text"),
            "err={err}"
        );
    }

    #[test]
    fn sse_tool_use_start_missing_id_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"name\":\"bash\",\"input\":{}}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_start"), "err={err}");
        assert!(err.contains("id"), "err={err}");
    }

    #[test]
    fn sse_tool_use_start_missing_name_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"input\":{}}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_start"), "err={err}");
        assert!(err.contains("name"), "err={err}");
    }

    #[test]
    fn sse_tool_use_start_non_string_id_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":123,\"name\":\"bash\",\"input\":{}}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_start"), "err={err}");
        assert!(err.contains("id"), "err={err}");
    }

    #[test]
    fn sse_text_delta_missing_text_field_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(err.contains("text"), "err={err}");
    }

    #[test]
    fn sse_thinking_delta_missing_thinking_field_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(err.contains("thinking"), "err={err}");
    }

    #[test]
    fn sse_input_json_delta_missing_partial_json_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"bash\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_delta"), "err={err}");
        assert!(err.contains("partial_json"), "err={err}");
    }

    #[test]
    fn sse_empty_string_delta_fields_remain_legal() {
        // Present empty strings are legal (distinct from missing/non-string).
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":1}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"bash\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":2}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}

event: message_stop
data: {\"type\":\"message_stop\"}

";
        let events = parse_all(fixture).expect("parse");
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { text } if text.is_empty())));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::ThinkingDelta { text } if text.is_empty())));
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::ToolCallArgumentsDelta {
                json_fragment,
                ..
            } if json_fragment.is_empty()
        )));
    }

    #[test]
    fn sse_double_message_stop_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}

event: message_stop
data: {\"type\":\"message_stop\"}

event: message_stop
data: {\"type\":\"message_stop\"}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_stop"), "err={err}");
        assert!(
            err.contains("stopped") || err.contains("after"),
            "err={err}"
        );
    }

    #[test]
    fn sse_event_after_stopped_is_malformed() {
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: message_stop
data: {\"type\":\"message_stop\"}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("content_block_start"), "err={err}");
        assert!(
            err.contains("stopped") || err.contains("after"),
            "err={err}"
        );
    }

    #[test]
    fn sse_message_delta_after_stopped_is_malformed() {
        // message_stop alone moves to Stopped; a late message_delta is illegal.
        let fixture = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}

event: message_stop
data: {\"type\":\"message_stop\"}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}

";
        let err = parse_all(fixture).expect_err("must fail");
        assert!(err.contains("message_delta"), "err={err}");
        assert!(
            err.contains("stopped") || err.contains("after"),
            "err={err}"
        );
    }
}
