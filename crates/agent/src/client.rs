//! Blocking Anthropic Messages client for a localhost gateway.
//!
//! Speaks plain HTTP (no TLS) against a CLIProxyAPI-style proxy, default
//! base `http://127.0.0.1:8317`. Streaming is assembled into a single
//! assistant [`Message`] plus stop reason and usage.

use std::io::Read;
use std::time::Duration;

use serde_json::Value;

use crate::protocol::{ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, Usage};
use crate::wire::{to_messages_request_body, SseItem, SseParser};

/// Default hard cap on total SSE body bytes (64 MiB).
pub const DEFAULT_STREAM_BYTE_CAP: usize = 64 * 1024 * 1024;
/// Default hard cap on completed SSE records (every parsed `event:`/`data:`
/// record, whether or not it emits a model event — pings, silent starts, and
/// unknown types count).
pub const DEFAULT_STREAM_EVENT_CAP: usize = 100_000;

/// Result of a completed streaming turn.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnResult {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// Errors from [`Client::stream_turn`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientError {
    /// Transport / connect / IO failure.
    Transport(String),
    /// Non-success HTTP status from the gateway.
    Http { status: u16, body: String },
    /// Stream ended with an error event or malformed payload.
    Stream(String),
    /// Could not assemble a coherent assistant message from the stream.
    Incomplete(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Transport(m) => write!(f, "transport error: {m}"),
            ClientError::Http { status, body } => {
                write!(f, "HTTP {status}: {body}")
            }
            ClientError::Stream(m) => write!(f, "stream error: {m}"),
            ClientError::Incomplete(m) => write!(f, "incomplete turn: {m}"),
        }
    }
}

impl std::error::Error for ClientError {}

/// Errors from [`Client::probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    /// Could not connect to the gateway at all.
    Unreachable(String),
    /// Connected but response was non-2xx or unreadable.
    BadResponse(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbeError::Unreachable(m) => write!(f, "gateway unreachable: {m}"),
            ProbeError::BadResponse(m) => write!(f, "bad probe response: {m}"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// Blocking client for a single model against a localhost Messages gateway.
#[derive(Clone)]
pub struct Client {
    base_url: String,
    model: String,
    api_key: Option<String>,
    /// Injectable stream byte cap (tests use a small value).
    stream_byte_cap: usize,
    /// Injectable stream event cap (tests use a small value).
    stream_event_cap: usize,
}

// Manual impl: the api key must never reach logs or error output.
impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("stream_byte_cap", &self.stream_byte_cap)
            .field("stream_event_cap", &self.stream_event_cap)
            .finish()
    }
}

impl Client {
    /// Create a client.
    ///
    /// `base_url` trailing slashes are stripped. The model name is stored for
    /// convenience; callers may still override it via [`ModelRequest::model`].
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: normalize_base_url(base_url),
            model: model.to_string(),
            api_key: None,
            stream_byte_cap: DEFAULT_STREAM_BYTE_CAP,
            stream_event_cap: DEFAULT_STREAM_EVENT_CAP,
        }
    }

    /// Attach a gateway API key, sent as both `x-api-key` and
    /// `Authorization: Bearer` on every request (gateways accept either).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        self.api_key = if key.is_empty() { None } else { Some(key) };
        self
    }

    /// Override the SSE stream byte / event caps (tests).
    pub fn with_stream_caps(mut self, byte_cap: usize, event_cap: usize) -> Self {
        self.stream_byte_cap = byte_cap;
        self.stream_event_cap = event_cap;
        self
    }

    /// Apply the configured auth headers, if any, to a request.
    fn authed(&self, req: ureq::Request) -> ureq::Request {
        match &self.api_key {
            Some(key) => req
                .set("x-api-key", key)
                .set("Authorization", &format!("Bearer {key}")),
            None => req,
        }
    }

    /// Base URL after trailing-slash normalization.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Default model name configured at construction.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// URL used by [`Client::probe`] (`GET {base}/v1/models`).
    pub fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url)
    }

    /// URL used by [`Client::stream_turn`] (`POST {base}/v1/messages`).
    pub fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// List model ids advertised by `GET {base}/v1/models`.
    ///
    /// Unknown JSON shapes return an empty list rather than failing; the
    /// interactive UI then falls back to the currently selected model.
    pub fn list_models(&self) -> Result<Vec<String>, ProbeError> {
        let url = self.models_url();
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(2))
            .timeout(Duration::from_secs(2))
            .build();

        match self.authed(agent.get(&url)).call() {
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                Ok(parse_model_ids(&body))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(ProbeError::BadResponse(format!(
                    "HTTP {code}: {}",
                    truncate(&body, 200)
                )))
            }
            Err(ureq::Error::Transport(t)) => Err(ProbeError::Unreachable(t.to_string())),
        }
    }

    /// Probe the gateway: `GET {base}/v1/models` with a 2 s timeout.
    ///
    /// Distinguishes connect failures ([`ProbeError::Unreachable`]) from
    /// non-2xx / garbage ([`ProbeError::BadResponse`]).
    pub fn probe(&self) -> Result<(), ProbeError> {
        let url = self.models_url();
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(2))
            .timeout(Duration::from_secs(2))
            .build();

        match self.authed(agent.get(&url)).call() {
            Ok(resp) => {
                // Drain body so the connection is not left half-open; ignore content.
                let _ = resp.into_string();
                Ok(())
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(ProbeError::BadResponse(format!(
                    "HTTP {code}: {}",
                    truncate(&body, 200)
                )))
            }
            Err(ureq::Error::Transport(t)) => Err(ProbeError::Unreachable(t.to_string())),
        }
    }

    /// Stream one model turn, invoking `on_event` for every parsed event.
    ///
    /// Assembles text blocks (concatenated), tool-use arguments (JSON
    /// fragments joined and parsed at block end), thinking blocks, stop
    /// reason, and usage into a [`TurnResult`].
    pub fn stream_turn(
        &self,
        req: &ModelRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        let url = self.messages_url();
        let body = to_messages_request_body(req);
        let body_str =
            serde_json::to_string(&body).map_err(|e| ClientError::Transport(e.to_string()))?;

        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            // Streaming turns can run long; no short overall read timeout.
            .timeout_read(Duration::from_secs(600))
            .build();

        let response = match self
            .authed(agent.post(&url))
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream")
            .send_string(&body_str)
        {
            Ok(r) => r,
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                return Err(ClientError::Http { status: code, body });
            }
            Err(ureq::Error::Transport(t)) => {
                return Err(ClientError::Transport(t.to_string()));
            }
        };

        let mut reader = response.into_reader();
        self.consume_sse(&mut reader, on_event)
    }

    /// Parse an SSE body from any `Read` (production HTTP body or test fixture).
    ///
    /// Buffers raw bytes, splits on `\n`, UTF-8-decodes only complete lines
    /// (so multi-byte characters that straddle a read boundary are safe),
    /// and enforces stream byte / event caps.
    pub(crate) fn consume_sse(
        &self,
        reader: &mut dyn Read,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        let mut parser = SseParser::new();
        let mut assembler = TurnAssembler::new();
        let mut byte_buf: Vec<u8> = Vec::new();
        let mut total_bytes: usize = 0;
        let mut total_events: usize = 0;
        let mut read_buf = [0u8; 4096];

        loop {
            let n = reader
                .read(&mut read_buf)
                .map_err(|e| ClientError::Transport(e.to_string()))?;
            if n == 0 {
                break;
            }
            total_bytes = total_bytes.saturating_add(n);
            if total_bytes > self.stream_byte_cap {
                return Err(ClientError::Stream(format!(
                    "stream cap exceeded: more than {} bytes",
                    self.stream_byte_cap
                )));
            }
            byte_buf.extend_from_slice(&read_buf[..n]);

            // Split on '\n'; leave a trailing partial line in byte_buf.
            while let Some(nl) = byte_buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = byte_buf.drain(..=nl).collect();
                // Drop the trailing '\n'.
                let line_bytes = &line_bytes[..line_bytes.len() - 1];
                let line = std::str::from_utf8(line_bytes)
                    .map_err(|e| ClientError::Stream(format!("invalid utf-8 in SSE line: {e}")))?;
                for item in parser.feed_line(line) {
                    total_events = total_events.saturating_add(1);
                    if total_events > self.stream_event_cap {
                        return Err(ClientError::Stream(format!(
                            "stream cap exceeded: more than {} events",
                            self.stream_event_cap
                        )));
                    }
                    handle_sse_item(item, &mut assembler, on_event)?;
                }
            }
        }

        // Trailing partial line (no final newline).
        if !byte_buf.is_empty() {
            let line = std::str::from_utf8(&byte_buf)
                .map_err(|e| ClientError::Stream(format!("invalid utf-8 in SSE line: {e}")))?;
            for item in parser.feed_line(line) {
                total_events = total_events.saturating_add(1);
                if total_events > self.stream_event_cap {
                    return Err(ClientError::Stream(format!(
                        "stream cap exceeded: more than {} events",
                        self.stream_event_cap
                    )));
                }
                handle_sse_item(item, &mut assembler, on_event)?;
            }
            byte_buf.clear();
        }

        for item in parser.finish() {
            total_events = total_events.saturating_add(1);
            if total_events > self.stream_event_cap {
                return Err(ClientError::Stream(format!(
                    "stream cap exceeded: more than {} events",
                    self.stream_event_cap
                )));
            }
            handle_sse_item(item, &mut assembler, on_event)?;
        }

        assembler.finish()
    }
}

fn handle_sse_item(
    item: SseItem,
    assembler: &mut TurnAssembler,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<(), ClientError> {
    match item {
        SseItem::Event(ev) => {
            assembler.observe(&ev)?;
            on_event(ev);
            Ok(())
        }
        SseItem::MessageStop => assembler.observe_message_stop(),
        // Counted toward the event cap by the caller; nothing to assemble.
        SseItem::Ignored => Ok(()),
        SseItem::Malformed { event_type, detail } => Err(ClientError::Stream(format!(
            "malformed {event_type} event: {detail}"
        ))),
    }
}

/// Strip trailing `/` characters from a base URL.
pub(crate) fn normalize_base_url(base: &str) -> String {
    base.trim().trim_end_matches('/').to_string()
}

pub(crate) fn parse_model_ids(body: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    let push_id = |ids: &mut Vec<String>, raw: &Value| {
        if let Some(id) = raw.as_str() {
            if !id.is_empty() && !ids.iter().any(|existing| existing == id) {
                ids.push(id.to_string());
            }
        } else if let Some(id) = raw.get("id").and_then(Value::as_str) {
            if !id.is_empty() && !ids.iter().any(|existing| existing == id) {
                ids.push(id.to_string());
            }
        }
    };
    if let Some(arr) = value.get("data").and_then(Value::as_array) {
        for item in arr {
            push_id(&mut ids, item);
        }
    } else if let Some(arr) = value.get("models").and_then(Value::as_array) {
        for item in arr {
            push_id(&mut ids, item);
        }
    } else if let Some(arr) = value.as_array() {
        for item in arr {
            push_id(&mut ids, item);
        }
    }
    ids
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    // Back off to a char boundary so we never panic on multibyte UTF-8.
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Currently-open content block while streaming.
#[derive(Debug)]
enum OpenBlock {
    None,
    Text {
        index: usize,
        text: String,
    },
    Thinking {
        index: usize,
        text: String,
    },
    Tool {
        index: usize,
        id: String,
        name: String,
        args_json: String,
    },
}

/// Accumulates stream events into a final assistant message.
///
/// Anthropic streams content blocks sequentially, so at most one block is
/// open at a time. The wire parser enforces ordered protocol states; this
/// assembler refuses to silently flush half-finished blocks. Requires a seen
/// `message_start` and a terminal stop (`message_delta` with stop_reason, or
/// `message_stop`).
#[derive(Debug)]
struct TurnAssembler {
    open: OpenBlock,
    /// Completed parts as `(index, part)`.
    parts: Vec<(usize, ContentPart)>,
    stop_reason: Option<StopReason>,
    usage: Usage,
    seen_message_start: bool,
    seen_terminal: bool,
    /// Index of the next text/thinking block opened by a delta (silent starts
    /// do not carry an index into the assembler; we track via BlockFinished).
    next_soft_index: usize,
}

impl TurnAssembler {
    fn new() -> Self {
        Self {
            open: OpenBlock::None,
            parts: Vec::new(),
            stop_reason: None,
            usage: Usage::default(),
            seen_message_start: false,
            seen_terminal: false,
            next_soft_index: 0,
        }
    }

    fn observe(&mut self, ev: &StreamEvent) -> Result<(), ClientError> {
        if self.seen_terminal {
            return Err(ClientError::Stream("event after terminal stop".to_string()));
        }
        match ev {
            StreamEvent::Started { .. } => {
                if self.seen_message_start {
                    return Err(ClientError::Stream("duplicate message_start".to_string()));
                }
                self.seen_message_start = true;
                Ok(())
            }
            StreamEvent::TextDelta { text } => {
                if text.is_empty() {
                    return Ok(());
                }
                self.append_text(text)
            }
            StreamEvent::ThinkingDelta { text } => {
                if text.is_empty() {
                    return Ok(());
                }
                self.append_thinking(text)
            }
            StreamEvent::ToolCallStarted { index, id, name } => {
                if !matches!(self.open, OpenBlock::None) {
                    return Err(ClientError::Stream(
                        "tool_use start while a content block is open".to_string(),
                    ));
                }
                self.open = OpenBlock::Tool {
                    index: *index,
                    id: id.clone(),
                    name: name.clone(),
                    args_json: String::new(),
                };
                // Soft indices track the highest seen block index + 1.
                self.next_soft_index = self.next_soft_index.max(index.saturating_add(1));
                Ok(())
            }
            StreamEvent::ToolCallArgumentsDelta {
                index,
                json_fragment,
            } => match &mut self.open {
                OpenBlock::Tool {
                    index: open_idx,
                    args_json,
                    ..
                } => {
                    if *open_idx != *index {
                        return Err(ClientError::Stream(format!(
                            "tool argument index mismatch: open={open_idx}, delta={index}"
                        )));
                    }
                    args_json.push_str(json_fragment);
                    Ok(())
                }
                _ => Err(ClientError::Stream(
                    "tool argument delta without an open tool block".to_string(),
                )),
            },
            StreamEvent::BlockFinished { index } => self.flush_open(*index),
            StreamEvent::Finished { stop_reason, usage } => {
                if !matches!(self.open, OpenBlock::None) {
                    return Err(ClientError::Stream(
                        "terminal event while a content block is open".to_string(),
                    ));
                }
                self.stop_reason = Some(stop_reason.clone());
                self.usage = *usage;
                self.seen_terminal = true;
                Ok(())
            }
            StreamEvent::Error { message } => Err(ClientError::Stream(message.clone())),
        }
    }

    fn observe_message_stop(&mut self) -> Result<(), ClientError> {
        if !matches!(self.open, OpenBlock::None) {
            return Err(ClientError::Stream(
                "terminal event while a content block is open".to_string(),
            ));
        }
        self.seen_terminal = true;
        Ok(())
    }

    fn append_text(&mut self, text: &str) -> Result<(), ClientError> {
        match &mut self.open {
            OpenBlock::Text { text: acc, .. } => {
                acc.push_str(text);
                Ok(())
            }
            OpenBlock::None => {
                let index = self.next_soft_index;
                self.open = OpenBlock::Text {
                    index,
                    text: text.to_string(),
                };
                Ok(())
            }
            _ => Err(ClientError::Stream(
                "text delta while a non-text content block is open".to_string(),
            )),
        }
    }

    fn append_thinking(&mut self, text: &str) -> Result<(), ClientError> {
        match &mut self.open {
            OpenBlock::Thinking { text: acc, .. } => {
                acc.push_str(text);
                Ok(())
            }
            OpenBlock::None => {
                let index = self.next_soft_index;
                self.open = OpenBlock::Thinking {
                    index,
                    text: text.to_string(),
                };
                Ok(())
            }
            _ => Err(ClientError::Stream(
                "thinking delta while a non-thinking content block is open".to_string(),
            )),
        }
    }

    fn flush_open(&mut self, index: usize) -> Result<(), ClientError> {
        match std::mem::replace(&mut self.open, OpenBlock::None) {
            OpenBlock::None => {
                // Silent text/thinking start may produce a BlockFinished with
                // no deltas; treat as an empty text part so indices stay coherent.
                self.parts.push((
                    index,
                    ContentPart::Text {
                        text: String::new(),
                    },
                ));
                self.next_soft_index = self.next_soft_index.max(index.saturating_add(1));
                Ok(())
            }
            OpenBlock::Text {
                index: open_idx,
                text,
            } => {
                if open_idx != index && !text.is_empty() {
                    // Soft-opened text used next_soft_index; accept the stop's index.
                }
                self.parts.push((index, ContentPart::Text { text }));
                self.next_soft_index = self.next_soft_index.max(index.saturating_add(1));
                Ok(())
            }
            OpenBlock::Thinking {
                index: open_idx,
                text,
            } => {
                let _ = open_idx;
                self.parts.push((index, ContentPart::Thinking { text }));
                self.next_soft_index = self.next_soft_index.max(index.saturating_add(1));
                Ok(())
            }
            OpenBlock::Tool {
                index: tool_index,
                id,
                name,
                args_json,
            } => {
                if tool_index != index {
                    return Err(ClientError::Stream(format!(
                        "block stop index mismatch: open={tool_index}, stop={index}"
                    )));
                }
                let arguments = parse_arguments(&args_json);
                self.parts.push((
                    tool_index,
                    ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    },
                ));
                self.next_soft_index = self.next_soft_index.max(index.saturating_add(1));
                Ok(())
            }
        }
    }

    fn finish(mut self) -> Result<TurnResult, ClientError> {
        // Never silently flush a half-finished block (especially tool_use): an
        // unfinished tool call must not finalize a turn.
        if !matches!(self.open, OpenBlock::None) {
            return Err(ClientError::Stream(
                "stream ended with an open content block".to_string(),
            ));
        }

        if !self.seen_message_start {
            return Err(ClientError::Incomplete("missing message_start".to_string()));
        }
        if !self.seen_terminal {
            return Err(ClientError::Incomplete(
                "missing terminal stop (message_delta/message_stop)".to_string(),
            ));
        }

        self.parts.sort_by_key(|(i, _)| *i);
        let content: Vec<ContentPart> = self.parts.into_iter().map(|(_, p)| p).collect();

        // message_delta supplies the real stop_reason; message_stop alone is still
        // a valid terminal (no fabrication of "missing_stop_reason").
        let stop_reason = self
            .stop_reason
            .unwrap_or(StopReason::Other("message_stop".to_string()));

        Ok(TurnResult {
            message: Message {
                role: Role::Assistant,
                content,
            },
            stop_reason,
            usage: self.usage,
        })
    }
}

fn parse_arguments(raw: &str) -> Value {
    if raw.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Map a probe/transport classification for tests without opening sockets.
pub fn classify_probe_transport_error(msg: &str) -> ProbeError {
    ProbeError::Unreachable(msg.to_string())
}

/// Map a non-2xx probe status into [`ProbeError::BadResponse`].
pub fn classify_probe_http_status(status: u16, body: &str) -> ProbeError {
    ProbeError::BadResponse(format!("HTTP {status}: {}", truncate(body, 200)))
}

/// Drive the SSE consumer over an in-memory body (tests).
#[cfg(test)]
pub(crate) fn consume_sse_for_test(
    body: &[u8],
    byte_cap: usize,
    event_cap: usize,
) -> Result<(TurnResult, Vec<StreamEvent>), ClientError> {
    let client = Client::new("http://127.0.0.1:9", "test").with_stream_caps(byte_cap, event_cap);
    let mut events = Vec::new();
    let mut cursor = std::io::Cursor::new(body);
    let turn = client.consume_sse(&mut cursor, &mut |ev| events.push(ev))?;
    Ok((turn, events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::map_stop_reason;

    #[test]
    fn normalize_base_url_strips_trailing_slashes() {
        assert_eq!(
            normalize_base_url("http://127.0.0.1:8317/"),
            "http://127.0.0.1:8317"
        );
        assert_eq!(
            normalize_base_url("http://127.0.0.1:8317///"),
            "http://127.0.0.1:8317"
        );
        assert_eq!(
            normalize_base_url("http://127.0.0.1:8317"),
            "http://127.0.0.1:8317"
        );
    }

    #[test]
    fn client_urls() {
        let c = Client::new("http://127.0.0.1:8317/", "claude-sonnet-4-20250514");
        assert_eq!(c.base_url(), "http://127.0.0.1:8317");
        assert_eq!(c.models_url(), "http://127.0.0.1:8317/v1/models");
        assert_eq!(c.messages_url(), "http://127.0.0.1:8317/v1/messages");
        assert_eq!(c.model(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn parse_model_ids_accepts_openai_and_anthropic_shapes() {
        assert_eq!(
            parse_model_ids(r#"{"data":[{"id":"test"},{"id":"other"}]}"#),
            vec!["test".to_string(), "other".to_string()]
        );
        assert_eq!(
            parse_model_ids(r#"{"models":["alpha",{"id":"beta"}]}"#),
            vec!["alpha".to_string(), "beta".to_string()]
        );
        assert_eq!(parse_model_ids("not json"), Vec::<String>::new());
        assert_eq!(parse_model_ids("{}"), Vec::<String>::new());
    }

    #[test]
    fn probe_error_mapping() {
        let u = classify_probe_transport_error("Connection refused");
        assert!(matches!(u, ProbeError::Unreachable(_)));
        assert!(u.to_string().contains("unreachable"));

        let b = classify_probe_http_status(503, "nope");
        assert!(matches!(b, ProbeError::BadResponse(_)));
        assert!(b.to_string().contains("503"));
    }

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

    #[test]
    fn assemble_turn_from_happy_path_events() {
        let (turn, _) = consume_sse_for_test(
            HAPPY_PATH_SSE.as_bytes(),
            DEFAULT_STREAM_BYTE_CAP,
            DEFAULT_STREAM_EVENT_CAP,
        )
        .expect("assemble");
        assert_eq!(turn.stop_reason, StopReason::ToolUse);
        assert_eq!(turn.usage.output_tokens, 42);
        assert_eq!(turn.usage.input_tokens, 100);
        assert_eq!(turn.message.role, Role::Assistant);
        assert_eq!(turn.message.content.len(), 2);
        match &turn.message.content[0] {
            ContentPart::Text { text } => assert_eq!(text, "I'll list the files."),
            other => panic!("expected text, got {other:?}"),
        }
        match &turn.message.content[1] {
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "bash");
                assert_eq!(arguments, &serde_json::json!({"command": "ls"}));
            }
            other => panic!("expected tool call, got {other:?}"),
        }
    }

    #[test]
    fn garbage_json_on_recognized_event_is_stream_error() {
        let body = b"event: message_start\ndata: {not-json\n\n";
        let err = consume_sse_for_test(body, DEFAULT_STREAM_BYTE_CAP, DEFAULT_STREAM_EVENT_CAP)
            .expect_err("must fail");
        assert!(matches!(err, ClientError::Stream(_)), "got {err:?}");
        assert!(err.to_string().contains("malformed") || err.to_string().contains("message_start"));
    }

    #[test]
    fn truncated_stream_is_incomplete() {
        // message_start only — no terminal stop.
        let body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n",
            "\n",
        );
        let err = consume_sse_for_test(
            body.as_bytes(),
            DEFAULT_STREAM_BYTE_CAP,
            DEFAULT_STREAM_EVENT_CAP,
        )
        .expect_err("must be incomplete");
        assert!(matches!(err, ClientError::Incomplete(_)), "got {err:?}");
        assert!(err.to_string().contains("terminal") || err.to_string().contains("missing"));
    }

    #[test]
    fn missing_message_start_is_stream_error() {
        // State machine rejects content before message_start as a stream error
        // (not a successful Incomplete after the fact).
        let body = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n",
            "\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n",
            "\n",
        );
        let err = consume_sse_for_test(
            body.as_bytes(),
            DEFAULT_STREAM_BYTE_CAP,
            DEFAULT_STREAM_EVENT_CAP,
        )
        .expect_err("must be stream error");
        assert!(matches!(err, ClientError::Stream(_)), "got {err:?}");
    }

    #[test]
    fn empty_stream_is_incomplete_missing_start() {
        let err = consume_sse_for_test(b"", DEFAULT_STREAM_BYTE_CAP, DEFAULT_STREAM_EVENT_CAP)
            .expect_err("must be incomplete");
        assert!(matches!(err, ClientError::Incomplete(_)), "got {err:?}");
        assert!(err.to_string().contains("message_start"));
    }

    #[test]
    fn stream_byte_cap_exceeded() {
        // Build a stream larger than a tiny cap.
        let mut body = String::from(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}\n\n",
        );
        for _ in 0..50 {
            body.push_str(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"xxxxxxxx\"}}\n\n",
            );
        }
        let err = consume_sse_for_test(body.as_bytes(), 200, DEFAULT_STREAM_EVENT_CAP)
            .expect_err("must hit byte cap");
        match err {
            ClientError::Stream(m) => {
                assert!(m.contains("stream cap exceeded"), "msg={m}");
                assert!(m.contains("bytes"), "msg={m}");
            }
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn stream_event_cap_exceeded() {
        let mut body = String::from(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}\n\n",
        );
        // Open a block so content_block_delta is legal under the state machine.
        body.push_str(
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        );
        for _ in 0..20 {
            body.push_str(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n",
            );
        }
        let err = consume_sse_for_test(body.as_bytes(), DEFAULT_STREAM_BYTE_CAP, 5)
            .expect_err("must hit event cap");
        match err {
            ClientError::Stream(m) => {
                assert!(m.contains("stream cap exceeded"), "msg={m}");
                assert!(m.contains("events"), "msg={m}");
            }
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn stream_event_cap_counts_ping_flood() {
        // Pings do not emit StreamEvents but must still count toward the cap.
        let mut body = String::from(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}\n\n",
        );
        for _ in 0..20 {
            body.push_str("event: ping\ndata: {}\n\n");
        }
        let err = consume_sse_for_test(body.as_bytes(), DEFAULT_STREAM_BYTE_CAP, 5)
            .expect_err("must hit event cap on pings");
        match err {
            ClientError::Stream(m) => {
                assert!(m.contains("stream cap exceeded"), "msg={m}");
                assert!(m.contains("events"), "msg={m}");
            }
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn stream_event_cap_counts_unknown_event_flood() {
        // Unknown event types are ignored for protocol but still count.
        let mut body = String::from(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}\n\n",
        );
        for _ in 0..20 {
            body.push_str("event: something_novel\ndata: {\"foo\":1}\n\n");
        }
        let err = consume_sse_for_test(body.as_bytes(), DEFAULT_STREAM_BYTE_CAP, 5)
            .expect_err("must hit event cap on unknown events");
        match err {
            ClientError::Stream(m) => {
                assert!(m.contains("stream cap exceeded"), "msg={m}");
                assert!(m.contains("events"), "msg={m}");
            }
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn truncated_open_block_is_stream_error_not_silent_flush() {
        let body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}\n",
            "\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"bash\",\"input\":{}}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"c\"}}\n",
            "\n",
            // Stream ends mid-block — no stop, no terminal.
        );
        let err = consume_sse_for_test(
            body.as_bytes(),
            DEFAULT_STREAM_BYTE_CAP,
            DEFAULT_STREAM_EVENT_CAP,
        )
        .expect_err("must not silently finalize open tool block");
        // Either Incomplete (no terminal) or Stream (open block) is correct;
        // silent success is not.
        assert!(
            matches!(err, ClientError::Stream(_) | ClientError::Incomplete(_)),
            "got {err:?}"
        );
        assert!(!matches!(err, ClientError::Transport(_)), "got {err:?}");
    }

    #[test]
    fn multibyte_utf8_at_every_byte_split_is_stable() {
        // Multibyte text that straddles many read boundaries.
        let text = "héllo wörld → ✓";
        let escaped = text; // already valid JSON string content
        let body = format!(
            "event: message_start\n\
data: {{\"type\":\"message_start\",\"message\":{{\"model\":\"m\",\"usage\":{{\"input_tokens\":1,\"output_tokens\":0}}}}}}\n\
\n\
event: content_block_start\n\
data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\
\n\
event: content_block_delta\n\
data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{escaped}\"}}}}\n\
\n\
event: content_block_stop\n\
data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\
\n\
event: message_delta\n\
data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":3}}}}\n\
\n\
event: message_stop\n\
data: {{\"type\":\"message_stop\"}}\n\
\n"
        );
        let bytes = body.as_bytes();

        let (baseline_turn, baseline_events) =
            consume_sse_for_test(bytes, DEFAULT_STREAM_BYTE_CAP, DEFAULT_STREAM_EVENT_CAP)
                .expect("baseline");
        match &baseline_turn.message.content[0] {
            ContentPart::Text { text: t } => assert_eq!(t, text),
            other => panic!("expected text, got {other:?}"),
        }

        // Chunking reader: deliver the body in fixed-size chunks of every
        // size 1..16, and also at every mid-body split for size 1.
        for chunk_size in 1..=16 {
            struct Chunked<'a> {
                data: &'a [u8],
                chunk: usize,
            }
            impl Read for Chunked<'_> {
                fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                    if self.data.is_empty() {
                        return Ok(0);
                    }
                    let n = self.chunk.min(self.data.len()).min(buf.len());
                    buf[..n].copy_from_slice(&self.data[..n]);
                    self.data = &self.data[n..];
                    Ok(n)
                }
            }
            let client = Client::new("http://127.0.0.1:9", "test")
                .with_stream_caps(DEFAULT_STREAM_BYTE_CAP, DEFAULT_STREAM_EVENT_CAP);
            let mut events = Vec::new();
            let mut reader = Chunked {
                data: bytes,
                chunk: chunk_size,
            };
            let turn = client
                .consume_sse(&mut reader, &mut |ev| events.push(ev))
                .unwrap_or_else(|e| panic!("chunk_size={chunk_size}: {e}"));
            assert_eq!(turn, baseline_turn, "chunk_size={chunk_size}");
            assert_eq!(events, baseline_events, "chunk_size={chunk_size}");
        }
    }

    #[test]
    fn truncate_backs_off_char_boundary() {
        // "é" is two bytes (0xC3 0xA9). Cutting at 1 would panic without backoff.
        let s = "héllo";
        // max=2 lands inside 'é' (bytes: h=1, é=2..3).
        // "h" is byte 0; "é" is bytes 1..2 (0-indexed: h at 0, é at 1-2).
        assert_eq!(s.as_bytes()[0], b'h');
        // Cut at 2: inside é → back off to 1 → "h".
        assert_eq!(truncate(s, 2), "h");
        // Cut at 3: on boundary after é → "hé".
        assert_eq!(truncate(s, 3), "hé");
        // Cut past end.
        assert_eq!(truncate(s, 100), "héllo");
    }

    #[test]
    fn map_stop_reasons() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::MaxTokens);
        assert_eq!(
            map_stop_reason("refusal"),
            StopReason::Other("refusal".to_string())
        );
    }

    #[test]
    fn invalid_utf8_complete_line_is_stream_error() {
        // Build a complete line with invalid UTF-8 bytes.
        let mut body = Vec::new();
        body.extend_from_slice(b"event: message_start\n");
        body.extend_from_slice(b"data: ");
        body.extend_from_slice(&[0xff, 0xff, 0xff]);
        body.extend_from_slice(b"\n\n");
        let err = consume_sse_for_test(&body, DEFAULT_STREAM_BYTE_CAP, DEFAULT_STREAM_EVENT_CAP)
            .expect_err("must fail");
        assert!(matches!(err, ClientError::Stream(_)), "got {err:?}");
        assert!(err.to_string().contains("utf-8") || err.to_string().contains("utf8"));
    }
}
