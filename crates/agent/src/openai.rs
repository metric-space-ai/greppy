//! OpenAI Chat Completions wire adapter.
//!
//! Maps the provider-neutral [`ModelRequest`] onto `POST /v1/chat/completions`
//! and parses the SSE stream (`data: {json}`, terminated by `data: [DONE]`)
//! into the same [`StreamEvent`]s the Anthropic parser produces. The agent
//! loop, tool execution, cancellation, and retry stay on that shared stream.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::protocol::{
    ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, ToolChoice, Usage,
};
use crate::wire::SseItem;

const IMAGE_OMITTED: &str = "[image omitted: model does not accept image input]";

/// Serialize `req` as a streaming Chat Completions body.
///
/// `include_images` is false after this gateway has rejected vision input.
/// Image parts then become a text note on the following tool result (or a
/// user text part when the image was not attached to a tool) so the turn
/// still runs.
pub(crate) fn to_chat_completions_body(req: &ModelRequest, include_images: bool) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = &req.system {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }
    for message in &req.messages {
        push_message(&mut messages, message, include_images);
    }

    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if req.max_tokens != u64::MAX {
        body["max_tokens"] = json!(req.max_tokens);
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
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }
    })
}

fn push_message(out: &mut Vec<Value>, message: &Message, include_images: bool) {
    match message.role {
        Role::Assistant => push_assistant(out, message),
        Role::User => push_user(out, message, include_images),
    }
}

fn push_assistant(out: &mut Vec<Value>, message: &Message) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for part in &message.content {
        match part {
            ContentPart::Text { text: chunk } => {
                if !text.is_empty() && !chunk.is_empty() {
                    text.push('\n');
                }
                text.push_str(chunk);
            }
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_string(arguments),
                }
            })),
            // Thinking is streamed for the current turn. It is not replayed:
            // official Chat Completions rejects `reasoning_content`, and local
            // servers disagree on the field name.
            ContentPart::Thinking { .. }
            | ContentPart::SignedThinking { .. }
            | ContentPart::ToolResult { .. }
            | ContentPart::Image { .. } => {}
        }
    }
    if tool_calls.is_empty() {
        out.push(json!({"role": "assistant", "content": text}));
    } else if text.is_empty() {
        out.push(json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": tool_calls,
        }));
    } else {
        out.push(json!({
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        }));
    }
}

fn push_user(out: &mut Vec<Value>, message: &Message, include_images: bool) {
    let mut texts: Vec<String> = Vec::new();
    let mut images: Vec<Value> = Vec::new();
    let mut pending_omitted = 0usize;
    let mut saw_tool = false;

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
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": body,
                }));
                saw_tool = true;
            }
            ContentPart::Text { text } => {
                if !text.is_empty() {
                    texts.push(text.clone());
                }
            }
            ContentPart::Image { media_type, data } => {
                if include_images {
                    images.push(image_part(media_type, data));
                } else {
                    pending_omitted += 1;
                }
            }
            ContentPart::ToolCall { .. }
            | ContentPart::Thinking { .. }
            | ContentPart::SignedThinking { .. } => {}
        }
    }

    if pending_omitted > 0 {
        texts.push(IMAGE_OMITTED.to_string());
    }
    if texts.is_empty() && images.is_empty() {
        return;
    }
    // Tool messages have to sit directly under the assistant tool_calls.
    // Images and leftover prose follow them as one user message.
    if images.is_empty() && texts.len() == 1 && !saw_tool {
        out.push(json!({"role": "user", "content": texts[0]}));
        return;
    }
    if images.is_empty() {
        out.push(json!({"role": "user", "content": texts.join("\n")}));
        return;
    }
    let mut content = Vec::new();
    let prose = if texts.is_empty() {
        "Image attached from the previous tool result.".to_string()
    } else {
        texts.join("\n")
    };
    content.push(json!({"type": "text", "text": prose}));
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
        "type": "image_url",
        "image_url": {"url": url}
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
    buffered_arguments: String,
}

/// Incremental parser for Chat Completions SSE.
///
/// Emits the provider-neutral events [`crate::client`]'s turn assembler
/// already understands: one `Started`, text and thinking deltas, overlapping
/// tool-call deltas addressed by assembler index, `BlockFinished` for every
/// open block, then one `Finished`. `[DONE]` and end-of-stream both close the
/// turn. A chunk that arrives after `[DONE]` is a protocol error.
#[derive(Debug, Default)]
pub(crate) struct OpenAiSseParser {
    data_lines: Vec<String>,
    started: bool,
    model: String,
    /// Mirrors the assembler's next soft index so text and tools do not share one.
    cursor: usize,
    soft: Option<(usize, SoftBlock)>,
    tools: BTreeMap<usize, PendingTool>,
    tool_order: Vec<usize>,
    saw_tools: bool,
    pending_stop: Option<StopReason>,
    usage: Usage,
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
        if let Some(rest) = line.strip_prefix("data:") {
            let data = rest.strip_prefix(' ').unwrap_or(rest);
            self.data_lines.push(data.to_string());
        }
        Vec::new()
    }

    /// Flush a trailing event and close the turn when the body ends without `[DONE]`.
    pub fn finish(&mut self) -> Vec<SseItem> {
        let mut out = self.dispatch();
        if !self.closed {
            out.extend(self.finalize());
        }
        out
    }

    fn dispatch(&mut self) -> Vec<SseItem> {
        let data = std::mem::take(&mut self.data_lines).join("\n");
        if data.is_empty() {
            return Vec::new();
        }
        if self.closed {
            return vec![malformed(
                "chat.completion.chunk",
                "event after stream stopped",
            )];
        }
        if data.trim() == "[DONE]" {
            return self.finalize();
        }
        let value = match serde_json::from_str::<Value>(&data) {
            Ok(value) => value,
            Err(error) => {
                return vec![malformed(
                    "chat.completion.chunk",
                    format!("invalid JSON: {error}"),
                )]
            }
        };
        if let Some(message) = error_message(&value) {
            self.closed = true;
            return vec![SseItem::Event(StreamEvent::Error { message })];
        }
        self.consume_chunk(&value)
    }

    fn consume_chunk(&mut self, value: &Value) -> Vec<SseItem> {
        let mut out = Vec::new();
        if !self.started {
            let model = value
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            self.model = model.clone();
            self.started = true;
            out.push(SseItem::Event(StreamEvent::Started { model }));
        } else if self.model.is_empty() {
            if let Some(model) = value.get("model").and_then(Value::as_str) {
                self.model = model.to_string();
            }
        }
        if let Some(usage) = usage_from(value.get("usage")) {
            self.usage = usage;
        }
        let Some(choice) = value
            .pointer("/choices/0")
            .filter(|choice| !choice.is_null())
        else {
            return out;
        };
        if let Some(delta) = choice.get("delta") {
            out.extend(self.consume_delta(delta));
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.pending_stop = Some(map_finish_reason(reason, self.saw_tools));
        }
        out
    }

    fn consume_delta(&mut self, delta: &Value) -> Vec<SseItem> {
        let mut out = Vec::new();
        if let Some(text) = string_field(delta, &["reasoning_content", "reasoning"]) {
            if !text.is_empty() {
                out.extend(self.begin_soft(SoftBlock::Thinking));
                out.push(SseItem::Event(StreamEvent::ThinkingDelta { text }));
            }
        }
        if let Some(text) = content_text(delta.get("content")) {
            if !text.is_empty() {
                out.extend(self.begin_soft(SoftBlock::Text));
                out.push(SseItem::Event(StreamEvent::TextDelta { text }));
            }
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                out.extend(self.consume_tool_call(call));
            }
        }
        out
    }

    fn begin_soft(&mut self, kind: SoftBlock) -> Vec<SseItem> {
        if matches!(self.soft, Some((_, open)) if open == kind) {
            return Vec::new();
        }
        let mut out = self.close_soft();
        // Text cannot overlap an open tool in the shared assembler.
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
        for openai_index in std::mem::take(&mut self.tool_order) {
            if let Some(tool) = self.tools.remove(&openai_index) {
                out.push(SseItem::Event(StreamEvent::BlockFinished {
                    index: tool.assembler_index,
                }));
            }
        }
        self.tools.clear();
        out
    }

    fn consume_tool_call(&mut self, call: &Value) -> Vec<SseItem> {
        let Some(openai_index) = call.get("index").and_then(Value::as_u64) else {
            return vec![malformed("tool_call", "missing tool_calls index")];
        };
        let openai_index = openai_index as usize;
        // Close text/thinking before taking an assembler index so the two
        // never share the index the turn assembler already gave the soft block.
        let mut out = self.close_soft();
        if !self.tools.contains_key(&openai_index) {
            let assembler_index = self.cursor;
            self.cursor = self.cursor.saturating_add(1);
            self.tools.insert(
                openai_index,
                PendingTool {
                    assembler_index,
                    id: String::new(),
                    name: String::new(),
                    started: false,
                    buffered_arguments: String::new(),
                },
            );
            self.tool_order.push(openai_index);
            self.saw_tools = true;
        }
        {
            let Some(tool) = self.tools.get_mut(&openai_index) else {
                return vec![malformed("tool_call", "tool index disappeared")];
            };
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                if tool.id.is_empty() {
                    tool.id = id.to_string();
                }
            }
            if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                if tool.name.is_empty() {
                    tool.name = name.to_string();
                }
            }
            if let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str) {
                if tool.started {
                    if !arguments.is_empty() {
                        out.push(SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                            index: tool.assembler_index,
                            json_fragment: arguments.to_string(),
                        }));
                    }
                } else {
                    tool.buffered_arguments.push_str(arguments);
                }
            }
            if !tool.started && !tool.id.is_empty() && !tool.name.is_empty() {
                tool.started = true;
                let index = tool.assembler_index;
                out.push(SseItem::Event(StreamEvent::ToolCallStarted {
                    index,
                    id: tool.id.clone(),
                    name: tool.name.clone(),
                }));
                if !tool.buffered_arguments.is_empty() {
                    out.push(SseItem::Event(StreamEvent::ToolCallArgumentsDelta {
                        index,
                        json_fragment: std::mem::take(&mut tool.buffered_arguments),
                    }));
                }
            }
        }
        out
    }

    fn finalize(&mut self) -> Vec<SseItem> {
        if self.closed {
            return Vec::new();
        }
        self.closed = true;
        if !self.started {
            return vec![malformed(
                "chat.completion.chunk",
                "stream ended before the first chunk",
            )];
        }
        for tool in self.tools.values() {
            if !tool.started {
                return vec![malformed(
                    "tool_call",
                    format!(
                        "tool call {} ended before id and name were sent",
                        tool.assembler_index
                    ),
                )];
            }
        }
        let mut out = self.close_soft();
        out.extend(self.close_tools());
        let stop_reason = self.pending_stop.take().unwrap_or(if self.saw_tools {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        });
        out.push(SseItem::Event(StreamEvent::Finished {
            stop_reason,
            usage: self.usage,
        }));
        out
    }
}

fn error_message(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    if error.is_null() {
        return None;
    }
    Some(
        error
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| error.as_str())
            .unwrap_or("error event")
            .to_string(),
    )
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        value
            .get(*name)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

fn content_text(content: Option<&Value>) -> Option<String> {
    let content = content?;
    match content {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if let Some(chunk) = part.get("text").and_then(Value::as_str) {
                    text.push_str(chunk);
                } else if let Some(chunk) = part.as_str() {
                    text.push_str(chunk);
                }
            }
            Some(text)
        }
        _ => None,
    }
}

fn usage_from(value: Option<&Value>) -> Option<Usage> {
    let value = value.filter(|usage| usage.is_object())?;
    let number = |paths: &[&str]| {
        paths
            .iter()
            .find_map(|path| value.pointer(path).and_then(Value::as_u64))
    };
    Some(Usage {
        input_tokens: number(&["/prompt_tokens", "/input_tokens"]).unwrap_or(0),
        output_tokens: number(&["/completion_tokens", "/output_tokens"]).unwrap_or(0),
        cache_read_input_tokens: number(&[
            "/prompt_tokens_details/cached_tokens",
            "/cache_read_input_tokens",
            "/prompt_cache_hit_tokens",
        ])
        .unwrap_or(0),
        cache_creation_input_tokens: number(&["/cache_creation_input_tokens"]).unwrap_or(0),
    })
}

fn map_finish_reason(reason: &str, saw_tools: bool) -> StopReason {
    match reason {
        "stop" if saw_tools => StopReason::ToolUse,
        "stop" => StopReason::EndTurn,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => StopReason::Other(other.to_string()),
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

    fn user_text(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentPart::Text {
                text: text.to_string(),
            }],
        }
    }

    fn sample_request() -> ModelRequest {
        ModelRequest {
            model: "gpt-4.1".into(),
            system: Some("be brief".into()),
            messages: vec![
                user_text("hi"),
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentPart::Text {
                            text: "looking".into(),
                        },
                        ContentPart::Thinking {
                            text: "hidden".into(),
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
    fn request_uses_chat_tools_and_image_data_url() {
        let body = to_chat_completions_body(&sample_request(), true);
        assert_eq!(body["model"], "gpt-4.1");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "greppy");
        assert_eq!(body["tools"][0]["function"]["parameters"]["type"], "object");

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be brief");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hi");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "looking");
        assert!(messages[2].get("reasoning_content").is_none());
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call_prev");
        assert_eq!(messages[2]["tool_calls"][0]["function"]["name"], "greppy");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            "{\"args\":[\"search\",\"clamp\"]}"
        );
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_prev");
        assert_eq!(messages[3]["content"], "screenshot attached");
        assert_eq!(messages[4]["role"], "user");
        assert_eq!(messages[4]["content"][0]["type"], "text");
        assert_eq!(messages[4]["content"][1]["type"], "image_url");
        assert_eq!(
            messages[4]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
        let rendered = body.to_string();
        assert!(!rendered.contains("sekrit"));
    }

    #[test]
    fn images_degrade_to_a_text_note() {
        let body = to_chat_completions_body(&sample_request(), false);
        let rendered = body.to_string();
        assert!(!rendered.contains("image_url"));
        assert!(!rendered.contains("AAAA"));
        assert!(rendered.contains(IMAGE_OMITTED));
        assert_eq!(
            body["messages"][3]["content"],
            format!("screenshot attached\n{IMAGE_OMITTED}")
        );
    }

    #[test]
    fn unlimited_budget_and_empty_tools_omit_provider_fields() {
        let mut req = sample_request();
        req.max_tokens = u64::MAX;
        req.tools.clear();
        req.tool_choice = ToolChoice::None;
        let body = to_chat_completions_body(&req, true);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn required_tool_choice_and_null_schema() {
        let mut req = sample_request();
        req.tool_choice = ToolChoice::Required;
        req.tools[0].input_schema = Value::Null;
        let body = to_chat_completions_body(&req, true);
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["tools"][0]["function"]["parameters"]["type"], "object");
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
                other => panic!("expected events, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn parses_parallel_tool_deltas_usage_and_done() {
        let mut parser = OpenAiSseParser::new();
        let body = concat!(
            "data: {\"id\":\"c\",\"model\":\"local\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
            ": keep-alive\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"greppy\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"a\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"greppy\",\"arguments\":\"{\\\"cmd\\\":\\\"b\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
            "data: [DONE]\n\n",
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
        assert!(got.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallArgumentsDelta { index: 2, json_fragment }
            if json_fragment == "{\"cmd\":\"b\"}"
        )));
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
    fn done_without_finish_reason_still_ends_the_turn() {
        let mut parser = OpenAiSseParser::new();
        let body = concat!(
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
            "data: [DONE]\n",
        );
        let got = events(&feed(&mut parser, body));
        assert!(matches!(
            got.last(),
            Some(StreamEvent::Finished {
                stop_reason: StopReason::EndTurn,
                ..
            })
        ));
    }

    #[test]
    fn length_finish_and_reasoning_delta() {
        let mut parser = OpenAiSseParser::new();
        let body = concat!(
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ans\"},\"finish_reason\":\"length\"}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}\n\n",
        );
        let got = events(&feed(&mut parser, body));
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
    fn error_payload_and_done_and_malformed_json() {
        let mut parser = OpenAiSseParser::new();
        let err = parser
            .feed_line("data: {\"error\":{\"message\":\"overloaded\",\"type\":\"server_error\"}}");
        assert!(err.is_empty());
        let err = parser.feed_line("");
        assert!(matches!(
            &err[0],
            SseItem::Event(StreamEvent::Error { message }) if message == "overloaded"
        ));

        let mut parser = OpenAiSseParser::new();
        let items = feed(
            &mut parser,
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"z\"}}]}\n\ndata: [DONE]\n\n",
        );
        assert!(events(&items).iter().any(|event| matches!(
            event,
            StreamEvent::Finished {
                stop_reason: StopReason::EndTurn,
                ..
            }
        )));

        let mut parser = OpenAiSseParser::new();
        let bad = feed(&mut parser, "data: {not json\n\n");
        assert!(matches!(bad[0], SseItem::Malformed { .. }));
    }

    #[test]
    fn event_after_done_is_malformed() {
        let mut parser = OpenAiSseParser::new();
        let mut items = Vec::new();
        for line in "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"z\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".split('\n')
        {
            items.extend(parser.feed_line(line));
        }
        items.extend(parser.feed_line("data: {\"choices\":[{\"delta\":{\"content\":\"more\"}}]}"));
        items.extend(parser.feed_line(""));
        assert!(items
            .iter()
            .any(|item| matches!(item, SseItem::Malformed { .. })));
    }

    #[test]
    fn vision_heuristic_is_specific() {
        assert!(vision_unsupported(
            400,
            r#"{"error":{"message":"model does not support image input"}}"#
        ));
        assert!(vision_unsupported(422, "unknown content type image_url"));
        assert!(!vision_unsupported(400, "invalid api key"));
        assert!(!vision_unsupported(
            500,
            "model does not support image input"
        ));
        assert!(!vision_unsupported(400, "image"));
    }
}
