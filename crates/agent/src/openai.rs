//! OpenAI Responses API wire adapter.
//!
//! Maps a [`ModelRequest`] onto `POST /v1/responses` and parses the named-event
//! SSE stream into the same [`StreamEvent`]s the Anthropic parser produces.
//! The agent loop stays protocol-neutral. Each turn sends the full history
//! (`store: false`); nothing is resumed with `previous_response_id`.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::protocol::{
    ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, ToolChoice, Usage,
};
use crate::wire::SseItem;

const IMAGE_OMITTED: &str = "[image omitted: model does not accept image input]";

/// Serialize `req` as a streaming Responses request.
///
/// `include_images` is false after this gateway has rejected image input.
/// Image parts then become a text note on the matching `function_call_output`
/// (or a user text item when the image was not attached to a tool).
pub(crate) fn to_responses_body(req: &ModelRequest, include_images: bool) -> Value {
    let mut input = Vec::new();
    for message in &req.messages {
        push_message(&mut input, message, include_images);
    }

    let mut body = json!({
        "model": req.model,
        "input": input,
        "stream": true,
        "store": false,
    });
    if let Some(system) = &req.system {
        body["instructions"] = json!(system);
    }
    if req.max_tokens != u64::MAX {
        body["max_output_tokens"] = json!(req.max_tokens);
    }
    if !req.tools.is_empty() {
        body["tools"] = Value::Array(req.tools.iter().map(map_tool).collect());
        body["tool_choice"] = match req.tool_choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"),
        };
    }
    body
}

/// True when a non-success response is specifically about image input, so the
/// client can retry the same turn without images. Other 400s are not retried.
pub(crate) fn vision_unsupported(status: u16, body: &str) -> bool {
    if status != 400 && status != 422 {
        return false;
    }
    let body = body.to_ascii_lowercase();
    let mentions_image =
        body.contains("image") || body.contains("vision") || body.contains("multimodal");
    if !mentions_image {
        return false;
    }
    const REJECTS: &[&str] = &[
        "not support",
        "unsupported",
        "unrecognized",
        "unknown content",
        "invalid content",
        "invalid image",
        "expected a string",
        "must be a string",
        "only text",
        "input_image",
        "image_url",
    ];
    REJECTS.iter().any(|needle| body.contains(needle))
}

pub(crate) fn request_has_image(req: &ModelRequest) -> bool {
    req.messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::Image { .. }))
    })
}

fn map_tool(tool: &crate::protocol::ToolDefinition) -> Value {
    let parameters = if tool.input_schema.is_null() {
        json!({"type": "object", "properties": {}})
    } else {
        tool.input_schema.clone()
    };
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
    })
}

fn push_message(out: &mut Vec<Value>, message: &Message, include_images: bool) {
    match message.role {
        Role::Assistant => push_assistant(out, message),
        Role::User => push_user(out, message, include_images),
    }
}

fn push_assistant(out: &mut Vec<Value>, message: &Message) {
    let mut text = Vec::new();
    let mut calls = Vec::new();
    for part in &message.content {
        match part {
            ContentPart::Text { text: chunk } if !chunk.is_empty() => {
                text.push(json!({"type": "output_text", "text": chunk}));
            }
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => calls.push(json!({
                "type": "function_call",
                "call_id": id,
                "name": name,
                "arguments": arguments_string(arguments),
            })),
            // Reasoning text is streamed for this turn. It is not replayed:
            // a Responses reasoning item needs the provider's original id,
            // which the neutral history does not keep.
            ContentPart::Text { .. }
            | ContentPart::Thinking { .. }
            | ContentPart::SignedThinking { .. }
            | ContentPart::ToolResult { .. }
            | ContentPart::Image { .. } => {}
        }
    }
    if !text.is_empty() {
        out.push(json!({"role": "assistant", "content": text}));
    }
    out.extend(calls);
}

fn push_user(out: &mut Vec<Value>, message: &Message, include_images: bool) {
    let mut texts: Vec<Value> = Vec::new();
    let mut images: Vec<Value> = Vec::new();
    let mut pending_omitted = 0usize;

    for part in &message.content {
        match part {
            ContentPart::ToolResult {
                call_id,
                content,
                is_error: _,
            } => {
                let mut body = content.clone();
                if pending_omitted > 0 {
                    append_note(&mut body, IMAGE_OMITTED);
                    pending_omitted = 0;
                }
                out.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": body,
                }));
            }
            ContentPart::Text { text } if !text.is_empty() => {
                texts.push(json!({"type": "input_text", "text": text}));
            }
            ContentPart::Image { media_type, data } => {
                if include_images {
                    images.push(image_part(media_type, data));
                } else {
                    pending_omitted += 1;
                }
            }
            ContentPart::Text { .. }
            | ContentPart::ToolCall { .. }
            | ContentPart::Thinking { .. }
            | ContentPart::SignedThinking { .. } => {}
        }
    }

    if pending_omitted > 0 {
        texts.push(json!({"type": "input_text", "text": IMAGE_OMITTED}));
    }
    if texts.is_empty() && images.is_empty() {
        return;
    }
    if images.is_empty() {
        out.push(json!({"role": "user", "content": texts}));
        return;
    }
    let mut content = Vec::new();
    if texts.is_empty() {
        content.push(json!({
            "type": "input_text",
            "text": "Image attached from the previous tool result."
        }));
    } else {
        content.extend(texts);
    }
    content.extend(images);
    out.push(json!({"role": "user", "content": content}));
}

fn append_note(body: &mut String, note: &str) {
    if !body.is_empty() && !body.ends_with('\n') {
        body.push('\n');
    }
    body.push_str(note);
}

fn image_part(media_type: &str, data: &str) -> Value {
    let url = if data.starts_with("data:") {
        data.to_string()
    } else {
        let media_type = if media_type.is_empty() {
            "image/png"
        } else {
            media_type
        };
        format!("data:{media_type};base64,{data}")
    };
    json!({
        "type": "input_image",
        "image_url": url,
    })
}

fn arguments_string(arguments: &Value) -> String {
    match arguments {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SoftBlock {
    Text,
    Thinking,
}

#[derive(Debug)]
struct PendingTool {
    assembler_index: usize,
    id: String,
    name: String,
    started: bool,
    saw_argument_delta: bool,
    buffered_arguments: String,
}

/// Incremental parser for Responses API SSE.
///
/// Accepts `event:` plus `data:` records. When the event name is missing, the
/// JSON `type` field is the event name. Unknown events are ignored. Reasoning
/// deltas become thinking text; other reasoning records are ignored. A turn
/// completes on `response.completed` or `response.incomplete`.
#[derive(Debug, Default)]
pub struct OpenAiSseParser {
    event_name: Option<String>,
    data_lines: Vec<String>,
    started: bool,
    cursor: usize,
    soft: Option<(usize, SoftBlock)>,
    tools: BTreeMap<usize, PendingTool>,
    tool_order: Vec<usize>,
    saw_tools: bool,
    closed: bool,
}

impl OpenAiSseParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed one SSE line with the trailing newline already removed.
    pub fn feed_line(&mut self, line: &str) -> Vec<SseItem> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return Vec::new();
        }
        if let Some(rest) = line.strip_prefix("event:") {
            self.event_name = Some(rest.trim_start().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            let data = rest.strip_prefix(' ').unwrap_or(rest);
            self.data_lines.push(data.to_string());
        }
        Vec::new()
    }

    /// Flush a trailing event. End of stream without `response.completed` or
    /// `response.incomplete` is a protocol error.
    pub fn finish(&mut self) -> Vec<SseItem> {
        let mut out = self.dispatch();
        if !self.closed {
            self.closed = true;
            out.push(malformed(
                "response.completed",
                if self.started {
                    "stream ended before response.completed"
                } else {
                    "stream ended before the first event"
                },
            ));
        }
        out
    }

    fn dispatch(&mut self) -> Vec<SseItem> {
        let name = self.event_name.take().unwrap_or_default();
        let data = std::mem::take(&mut self.data_lines).join("\n");
        if data.is_empty() && name.is_empty() {
            return Vec::new();
        }
        if data.trim() == "[DONE]" {
            return vec![SseItem::Ignored];
        }
        if self.closed {
            return vec![malformed(&name, "event after stream stopped")];
        }
        let value = match serde_json::from_str::<Value>(&data) {
            Ok(value) => value,
            Err(error) => {
                self.closed = true;
                return vec![malformed(
                    if name.is_empty() { "response" } else { &name },
                    format!("invalid JSON: {error}"),
                )];
            }
        };
        let event_type = if !name.is_empty() {
            name
        } else {
            value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        self.handle(&event_type, &value)
    }

    fn handle(&mut self, event_type: &str, value: &Value) -> Vec<SseItem> {
        if event_type == "error" || event_type == "response.failed" {
            self.closed = true;
            return vec![SseItem::Event(StreamEvent::Error {
                message: error_message(value),
            })];
        }
        if is_reasoning_delta(event_type) {
            let mut out = self.ensure_started(value);
            if let Some(text) = value.get("delta").and_then(Value::as_str) {
                if !text.is_empty() {
                    out.extend(self.begin_soft(SoftBlock::Thinking));
                    out.push(SseItem::Event(StreamEvent::ThinkingDelta {
                        text: text.to_string(),
                    }));
                }
            }
            return out;
        }
        match event_type {
            "response.created" | "response.in_progress" => self.ensure_started(value),
            "response.output_text.delta" => {
                let mut out = self.ensure_started(value);
                if let Some(text) = value.get("delta").and_then(Value::as_str) {
                    if !text.is_empty() {
                        out.extend(self.begin_soft(SoftBlock::Text));
                        out.push(SseItem::Event(StreamEvent::TextDelta {
                            text: text.to_string(),
                        }));
                    }
                }
                out
            }
            "response.output_item.added" => {
                let mut out = self.ensure_started(value);
                out.extend(self.note_added(value));
                out
            }
            "response.function_call_arguments.delta" => {
                let mut out = self.ensure_started(value);
                out.extend(self.note_arguments_delta(value));
                out
            }
            "response.function_call_arguments.done" => {
                let mut out = self.ensure_started(value);
                out.extend(self.note_arguments_done(value));
                out
            }
            "response.output_item.done" => {
                let mut out = self.ensure_started(value);
                out.extend(self.note_item_done(value));
                out
            }
            "response.completed" => self.finish_turn(value, completed_stop(self.saw_tools)),
            "response.incomplete" => self.finish_turn(value, incomplete_stop(value)),
            _ => vec![SseItem::Ignored],
        }
    }

    fn ensure_started(&mut self, value: &Value) -> Vec<SseItem> {
        if self.started {
            return Vec::new();
        }
        let model = value
            .pointer("/response/model")
            .or_else(|| value.get("model"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        self.started = true;
        vec![SseItem::Event(StreamEvent::Started { model })]
    }

    fn note_added(&mut self, value: &Value) -> Vec<SseItem> {
        let Some(item) = value.get("item") else {
            return Vec::new();
        };
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "function_call" {
            return Vec::new();
        }
        let Some(output_index) = output_index(value) else {
            return vec![malformed(
                "response.output_item.added",
                "missing output_index",
            )];
        };
        self.touch_tool(output_index, item)
    }

    fn touch_tool(&mut self, output_index: usize, item: &Value) -> Vec<SseItem> {
        let mut out = Vec::new();
        if !self.tools.contains_key(&output_index) {
            out.extend(self.close_soft());
            let assembler_index = self.cursor;
            self.cursor = self.cursor.saturating_add(1);
            self.tools.insert(
                output_index,
                PendingTool {
                    assembler_index,
                    id: String::new(),
                    name: String::new(),
                    started: false,
                    saw_argument_delta: false,
                    buffered_arguments: String::new(),
                },
            );
            self.tool_order.push(output_index);
            self.saw_tools = true;
        }
        let Some(tool) = self.tools.get_mut(&output_index) else {
            return vec![malformed("function_call", "tool index disappeared")];
        };
        if tool.id.is_empty() {
            if let Some(id) = item.get("call_id").and_then(Value::as_str) {
                tool.id = id.to_string();
            }
        }
        if tool.name.is_empty() {
            if let Some(name) = item.get("name").and_then(Value::as_str) {
                tool.name = name.to_string();
            }
        }
        if !tool.started && !tool.saw_argument_delta {
            if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                if !arguments.is_empty() && tool.buffered_arguments.is_empty() {
                    tool.buffered_arguments = arguments.to_string();
                }
            }
        }
        if !tool.started && !tool.id.is_empty() && !tool.name.is_empty() {
            tool.started = true;
            let index = tool.assembler_index;
            let id = tool.id.clone();
            let name = tool.name.clone();
            let buffered = std::mem::take(&mut tool.buffered_arguments);
            out.push(SseItem::Event(StreamEvent::ToolCallStarted {
                index,
                id,
                name,
            }));
            if !buffered.is_empty() {
                out.push(SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                    index,
                    json_fragment: buffered,
                }));
            }
        }
        out
    }

    fn note_arguments_delta(&mut self, value: &Value) -> Vec<SseItem> {
        let Some(output_index) = output_index(value) else {
            return vec![malformed(
                "response.function_call_arguments.delta",
                "missing output_index",
            )];
        };
        let delta = value.get("delta").and_then(Value::as_str).unwrap_or("");
        if !self.tools.contains_key(&output_index) {
            let mut out = self.touch_tool(output_index, &json!({}));
            if let Some(tool) = self.tools.get_mut(&output_index) {
                tool.saw_argument_delta = true;
                if tool.started && !delta.is_empty() {
                    out.push(SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                        index: tool.assembler_index,
                        json_fragment: delta.to_string(),
                    }));
                } else if !delta.is_empty() {
                    tool.buffered_arguments.push_str(delta);
                }
            }
            return out;
        }
        let Some(tool) = self.tools.get_mut(&output_index) else {
            return Vec::new();
        };
        tool.saw_argument_delta = true;
        if !delta.is_empty() {
            if tool.started {
                return vec![SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                    index: tool.assembler_index,
                    json_fragment: delta.to_string(),
                })];
            }
            tool.buffered_arguments.push_str(delta);
        }
        Vec::new()
    }

    fn note_arguments_done(&mut self, value: &Value) -> Vec<SseItem> {
        let Some(output_index) = output_index(value) else {
            return vec![malformed(
                "response.function_call_arguments.done",
                "missing output_index",
            )];
        };
        let arguments = value.get("arguments").and_then(Value::as_str).unwrap_or("");
        if !self.tools.contains_key(&output_index) {
            return self.touch_tool(output_index, &json!({"arguments": arguments}));
        }
        let Some(tool) = self.tools.get_mut(&output_index) else {
            return Vec::new();
        };
        if tool.saw_argument_delta || arguments.is_empty() {
            return Vec::new();
        }
        tool.saw_argument_delta = true;
        if tool.started {
            vec![SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                index: tool.assembler_index,
                json_fragment: arguments.to_string(),
            })]
        } else {
            tool.buffered_arguments = arguments.to_string();
            Vec::new()
        }
    }

    fn note_item_done(&mut self, value: &Value) -> Vec<SseItem> {
        let kind = value
            .pointer("/item/type")
            .and_then(Value::as_str)
            .unwrap_or("");
        match kind {
            "function_call" => {
                let Some(output_index) = output_index(value) else {
                    return vec![malformed(
                        "response.output_item.done",
                        "missing output_index",
                    )];
                };
                let item = value.get("item").cloned().unwrap_or(Value::Null);
                let mut out = self.touch_tool(output_index, &item);
                out.extend(self.finish_tool(output_index));
                out
            }
            "message" | "reasoning" => self.close_soft(),
            _ => Vec::new(),
        }
    }

    fn finish_tool(&mut self, output_index: usize) -> Vec<SseItem> {
        let Some(tool) = self.tools.remove(&output_index) else {
            return Vec::new();
        };
        self.tool_order.retain(|index| *index != output_index);
        if !tool.started {
            return vec![malformed(
                "function_call",
                format!(
                    "function call {} ended before call_id and name were sent",
                    tool.assembler_index
                ),
            )];
        }
        vec![SseItem::Event(StreamEvent::BlockFinished {
            index: tool.assembler_index,
        })]
    }

    fn finish_turn(&mut self, value: &Value, stop_reason: StopReason) -> Vec<SseItem> {
        if self.tools.values().any(|tool| !tool.started) {
            self.closed = true;
            return vec![malformed(
                "function_call",
                "function call ended before call_id and name were sent",
            )];
        }
        let mut out = self.ensure_started(value);
        self.closed = true;
        if self.tools.is_empty() {
            out.extend(self.close_soft());
        }
        out.extend(self.close_tools());
        out.push(SseItem::Event(StreamEvent::Finished {
            stop_reason,
            usage: usage_from(value),
        }));
        out
    }

    fn begin_soft(&mut self, kind: SoftBlock) -> Vec<SseItem> {
        if matches!(self.soft, Some((_, open)) if open == kind) {
            return Vec::new();
        }
        let mut out = self.close_soft();
        if !self.tools.is_empty() {
            out.extend(self.close_tools());
        }
        let index = self.cursor;
        self.soft = Some((index, kind));
        out
    }

    fn close_soft(&mut self) -> Vec<SseItem> {
        let Some((index, _)) = self.soft.take() else {
            return Vec::new();
        };
        self.cursor = self.cursor.max(index.saturating_add(1));
        vec![SseItem::Event(StreamEvent::BlockFinished { index })]
    }

    fn close_tools(&mut self) -> Vec<SseItem> {
        let mut out = Vec::new();
        for output_index in std::mem::take(&mut self.tool_order) {
            if let Some(tool) = self.tools.remove(&output_index) {
                out.push(SseItem::Event(StreamEvent::BlockFinished {
                    index: tool.assembler_index,
                }));
            }
        }
        self.tools.clear();
        out
    }
}

fn output_index(value: &Value) -> Option<usize> {
    value
        .get("output_index")
        .and_then(Value::as_u64)
        .map(|index| index as usize)
}

fn is_reasoning_delta(event_type: &str) -> bool {
    event_type.contains("reasoning") && event_type.ends_with(".delta")
}

fn completed_stop(saw_tools: bool) -> StopReason {
    if saw_tools {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    }
}

fn incomplete_stop(value: &Value) -> StopReason {
    match value
        .pointer("/response/incomplete_details/reason")
        .and_then(Value::as_str)
    {
        Some("max_output_tokens") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_string()),
        None => StopReason::Other("incomplete".to_string()),
    }
}

fn error_message(value: &Value) -> String {
    value
        .pointer("/response/error/message")
        .or_else(|| value.pointer("/error/message"))
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("error event")
        .to_string()
}

fn usage_from(value: &Value) -> Usage {
    let Some(usage) = value
        .pointer("/response/usage")
        .or_else(|| value.get("usage"))
        .filter(|usage| usage.is_object())
    else {
        return Usage::default();
    };
    let number = |paths: &[&str]| {
        paths
            .iter()
            .find_map(|path| usage.pointer(path).and_then(Value::as_u64))
    };
    Usage {
        input_tokens: number(&["/input_tokens", "/prompt_tokens"]).unwrap_or(0),
        output_tokens: number(&["/output_tokens", "/completion_tokens"]).unwrap_or(0),
        cache_read_input_tokens: number(&[
            "/input_tokens_details/cached_tokens",
            "/prompt_tokens_details/cached_tokens",
        ])
        .unwrap_or(0),
        cache_creation_input_tokens: 0,
    }
}

fn malformed(event_type: &str, detail: impl Into<String>) -> SseItem {
    SseItem::Malformed {
        event_type: event_type.to_string(),
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ToolChoice, ToolDefinition};

    fn sample_request() -> ModelRequest {
        ModelRequest {
            model: "gpt-4.1".into(),
            system: Some("be brief".into()),
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![ContentPart::Text { text: "hi".into() }],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentPart::Text {
                            text: "looking".into(),
                        },
                        ContentPart::Thinking {
                            text: "REASONING_NOT_REPLAYED".into(),
                        },
                        ContentPart::ToolCall {
                            id: "call_prev".into(),
                            name: "greppy".into(),
                            arguments: json!({"args": ["search", "clamp"]}),
                        },
                    ],
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
                description: "navigate".into(),
                input_schema: json!({"type": "object", "properties": {"args": {"type": "array"}}}),
            }],
            tool_choice: ToolChoice::Auto,
            max_tokens: 64,
        }
    }

    #[test]
    fn request_uses_responses_items_and_image_data_url() {
        let body = to_responses_body(&sample_request(), true);
        assert_eq!(body["model"], "gpt-4.1");
        assert_eq!(body["instructions"], "be brief");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["max_output_tokens"], 64);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("messages").is_none());
        assert!(body.get("previous_response_id").is_none());
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "greppy");
        assert!(body["tools"][0].get("function").is_none());
        assert_eq!(body["tools"][0]["parameters"]["type"], "object");

        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hi");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["content"][0]["text"], "looking");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_prev");
        assert_eq!(input[2]["name"], "greppy");
        assert_eq!(input[2]["arguments"], "{\"args\":[\"search\",\"clamp\"]}");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_prev");
        assert_eq!(input[3]["output"], "screenshot attached");
        assert_eq!(input[4]["role"], "user");
        assert_eq!(input[4]["content"][1]["type"], "input_image");
        assert_eq!(
            input[4]["content"][1]["image_url"],
            "data:image/png;base64,AAAA"
        );
        let rendered = body.to_string();
        assert!(!rendered.contains("REASONING_NOT_REPLAYED"));
        assert!(!rendered.contains("sekrit"));
    }

    #[test]
    fn images_degrade_to_a_text_note() {
        let body = to_responses_body(&sample_request(), false);
        let rendered = body.to_string();
        assert!(!rendered.contains("input_image"));
        assert!(!rendered.contains("AAAA"));
        assert!(rendered.contains(IMAGE_OMITTED));
        assert_eq!(
            body["input"][3]["output"],
            format!("screenshot attached\n{IMAGE_OMITTED}")
        );
    }

    #[test]
    fn unlimited_budget_and_empty_tools_omit_provider_fields() {
        let mut req = sample_request();
        req.max_tokens = u64::MAX;
        req.tools.clear();
        req.tool_choice = ToolChoice::None;
        let body = to_responses_body(&req, true);
        assert!(body.get("max_output_tokens").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert_eq!(body["store"], false);
    }

    #[test]
    fn required_tool_choice_and_null_schema() {
        let mut req = sample_request();
        req.tool_choice = ToolChoice::Required;
        req.tools[0].input_schema = Value::Null;
        let body = to_responses_body(&req, true);
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["tools"][0]["parameters"]["type"], "object");
    }

    fn feed(parser: &mut OpenAiSseParser, body: &str) -> Vec<SseItem> {
        let mut out = Vec::new();
        for line in body.split('\n') {
            out.extend(parser.feed_line(line));
        }
        out.extend(parser.finish());
        out
    }

    fn events(items: &[SseItem]) -> Vec<StreamEvent> {
        items
            .iter()
            .map(|item| match item {
                SseItem::Event(event) => event.clone(),
                SseItem::Ignored => panic!("ignored record in event list"),
                other => panic!("expected events, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn parses_parallel_function_calls_and_usage() {
        let mut parser = OpenAiSseParser::new();
        let body = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"model\":\"local\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hi\"}\n\n",
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"greppy\",\"arguments\":\"\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"delta\":\"{\\\"cmd\\\":\\\"a\\\"}\"}\n\n",
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_b\",\"name\":\"greppy\",\"arguments\":\"\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":2,\"delta\":\"{\\\"cmd\\\":\\\"b\\\"}\"}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"greppy\"}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"type\":\"function_call\"}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":11,\"output_tokens\":7,\"input_tokens_details\":{\"cached_tokens\":3}}}}\n\n",
        );
        let got = events(&feed(&mut parser, body));
        assert!(matches!(got[0], StreamEvent::Started { ref model } if model == "local"));
        assert!(matches!(got[1], StreamEvent::TextDelta { ref text } if text == "Hi"));
        assert!(matches!(got[2], StreamEvent::BlockFinished { index: 0 }));
        assert!(matches!(
            &got[3],
            StreamEvent::ToolCallStarted { index: 1, id, name }
            if id == "call_a" && name == "greppy"
        ));
        assert!(matches!(
            &got[4],
            StreamEvent::ToolCallArgumentsDelta { index: 1, json_fragment }
            if json_fragment == "{\"cmd\":\"a\"}"
        ));
        assert!(matches!(
            &got[5],
            StreamEvent::ToolCallStarted { index: 2, id, name }
            if id == "call_b" && name == "greppy"
        ));
        let finished: Vec<usize> = got
            .iter()
            .filter_map(|event| match event {
                StreamEvent::BlockFinished { index } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(finished, vec![0, 1, 2]);
        assert!(matches!(
            got.last(),
            Some(StreamEvent::Finished { stop_reason: StopReason::ToolUse, usage })
            if usage.input_tokens == 11
                && usage.output_tokens == 7
                && usage.cache_read_input_tokens == 3
        ));
    }

    #[test]
    fn data_only_events_reasoning_and_incomplete() {
        let mut parser = OpenAiSseParser::new();
        let body = concat!(
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"think\",\"model\":\"m\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ans\"}\n\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
        );
        let got = events(&feed(&mut parser, body));
        assert!(matches!(got[0], StreamEvent::Started { ref model } if model == "m"));
        assert!(matches!(got[1], StreamEvent::ThinkingDelta { ref text } if text == "think"));
        assert!(got
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "ans")));
        assert!(matches!(
            got.last(),
            Some(StreamEvent::Finished { stop_reason: StopReason::MaxTokens, usage })
            if usage.input_tokens == 1 && usage.output_tokens == 2
        ));
    }

    #[test]
    fn failed_and_error_events_and_event_after_completed() {
        let mut parser = OpenAiSseParser::new();
        let failed = feed(
            &mut parser,
            "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"overloaded\"}}}\n\n",
        );
        assert!(matches!(
            &failed[0],
            SseItem::Event(StreamEvent::Error { message }) if message == "overloaded"
        ));

        let mut parser = OpenAiSseParser::new();
        let err = feed(
            &mut parser,
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"bad request\"}}\n\n",
        );
        assert!(matches!(
            &err[0],
            SseItem::Event(StreamEvent::Error { message }) if message == "bad request"
        ));

        let mut parser = OpenAiSseParser::new();
        let mut items = Vec::new();
        for line in "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"model\":\"m\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n".split('\n') {
            items.extend(parser.feed_line(line));
        }
        items.extend(parser.feed_line("event: response.output_text.delta"));
        items.extend(
            parser.feed_line("data: {\"type\":\"response.output_text.delta\",\"delta\":\"more\"}"),
        );
        items.extend(parser.feed_line(""));
        assert!(items
            .iter()
            .any(|item| matches!(item, SseItem::Malformed { .. })));
    }

    #[test]
    fn malformed_json_is_a_protocol_error() {
        let mut parser = OpenAiSseParser::new();
        let bad = feed(&mut parser, "data: {not json\n\n");
        assert!(matches!(bad[0], SseItem::Malformed { .. }));
    }

    #[test]
    fn reasoning_item_without_delta_does_not_break_the_turn() {
        let mut parser = OpenAiSseParser::new();
        let body = concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"model\":\"m\",\"status\":\"completed\"}}\n\n",
        );
        let got = events(&feed(&mut parser, body));
        assert!(got
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "ok")));
        assert!(matches!(
            got.last(),
            Some(StreamEvent::Finished {
                stop_reason: StopReason::EndTurn,
                ..
            })
        ));
    }

    #[test]
    fn vision_heuristic_is_specific() {
        assert!(vision_unsupported(
            400,
            r#"{"error":{"message":"model does not support image input"}}"#
        ));
        assert!(vision_unsupported(422, "unknown content type input_image"));
        assert!(!vision_unsupported(400, "invalid api key"));
        assert!(!vision_unsupported(
            500,
            "model does not support image input"
        ));
        assert!(!vision_unsupported(400, "image"));
    }
}
