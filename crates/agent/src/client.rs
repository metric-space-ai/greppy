//! Blocking Anthropic Messages client for a localhost gateway.
//!
//! Speaks plain HTTP (no TLS) against a CLIProxyAPI-style proxy, default
//! base `http://127.0.0.1:8317`. Streaming is assembled into a single
//! assistant [`Message`] plus stop reason and usage.

use std::io::Read;
use std::time::Duration;

use serde_json::Value;

use crate::protocol::{ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, Usage};
use crate::wire::{to_messages_request_body, SseParser};

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
#[derive(Debug, Clone)]
pub struct Client {
    base_url: String,
    model: String,
    api_key: Option<String>,
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
        }
    }

    /// Attach a gateway API key, sent as both `x-api-key` and
    /// `Authorization: Bearer` on every request (gateways accept either).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        self.api_key = if key.is_empty() { None } else { Some(key) };
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
        let mut parser = SseParser::new();
        let mut assembler = TurnAssembler::new();
        let mut buf = [0u8; 4096];

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| ClientError::Transport(e.to_string()))?;
            if n == 0 {
                break;
            }
            let chunk = std::str::from_utf8(&buf[..n])
                .map_err(|e| ClientError::Stream(format!("invalid utf-8 in SSE body: {e}")))?;
            for ev in parser.feed(chunk) {
                assembler.observe(&ev);
                on_event(ev);
            }
            if let Some(err) = assembler.take_stream_error() {
                return Err(ClientError::Stream(err));
            }
        }

        for ev in parser.finish() {
            assembler.observe(&ev);
            on_event(ev);
        }
        if let Some(err) = assembler.take_stream_error() {
            return Err(ClientError::Stream(err));
        }

        assembler.finish()
    }
}

/// Strip trailing `/` characters from a base URL.
pub(crate) fn normalize_base_url(base: &str) -> String {
    base.trim().trim_end_matches('/').to_string()
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

/// Currently-open content block while streaming.
#[derive(Debug)]
enum OpenBlock {
    None,
    Text {
        text: String,
    },
    Thinking {
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
/// open at a time.
#[derive(Debug)]
struct TurnAssembler {
    open: OpenBlock,
    /// Completed parts as `(index, part)`.
    parts: Vec<(usize, ContentPart)>,
    stop_reason: Option<StopReason>,
    usage: Usage,
    stream_error: Option<String>,
}

impl TurnAssembler {
    fn new() -> Self {
        Self {
            open: OpenBlock::None,
            parts: Vec::new(),
            stop_reason: None,
            usage: Usage::default(),
            stream_error: None,
        }
    }

    fn observe(&mut self, ev: &StreamEvent) {
        match ev {
            StreamEvent::Started { .. } => {}
            StreamEvent::TextDelta { text } => {
                self.append_text(text);
            }
            StreamEvent::ThinkingDelta { text } => {
                self.append_thinking(text);
            }
            StreamEvent::ToolCallStarted { index, id, name } => {
                if !matches!(self.open, OpenBlock::None) {
                    let idx = match &self.open {
                        OpenBlock::Tool { index, .. } => *index,
                        _ => index.saturating_sub(1),
                    };
                    self.flush_open(idx);
                }
                self.open = OpenBlock::Tool {
                    index: *index,
                    id: id.clone(),
                    name: name.clone(),
                    args_json: String::new(),
                };
            }
            StreamEvent::ToolCallArgumentsDelta {
                index: _,
                json_fragment,
            } => {
                if let OpenBlock::Tool { args_json, .. } = &mut self.open {
                    args_json.push_str(json_fragment);
                }
            }
            StreamEvent::BlockFinished { index } => {
                self.flush_open(*index);
            }
            StreamEvent::Finished { stop_reason, usage } => {
                self.stop_reason = Some(stop_reason.clone());
                self.usage = *usage;
            }
            StreamEvent::Error { message } => {
                self.stream_error = Some(message.clone());
            }
        }
    }

    fn append_text(&mut self, text: &str) {
        if let OpenBlock::Text { text: acc } = &mut self.open {
            acc.push_str(text);
            return;
        }
        if !matches!(self.open, OpenBlock::None) {
            let idx = self.parts.len();
            self.flush_open(idx);
        }
        self.open = OpenBlock::Text {
            text: text.to_string(),
        };
    }

    fn append_thinking(&mut self, text: &str) {
        if let OpenBlock::Thinking { text: acc } = &mut self.open {
            acc.push_str(text);
            return;
        }
        if !matches!(self.open, OpenBlock::None) {
            let idx = self.parts.len();
            self.flush_open(idx);
        }
        self.open = OpenBlock::Thinking {
            text: text.to_string(),
        };
    }

    fn flush_open(&mut self, index: usize) {
        match std::mem::replace(&mut self.open, OpenBlock::None) {
            OpenBlock::None => {}
            OpenBlock::Text { text } => {
                self.parts.push((index, ContentPart::Text { text }));
            }
            OpenBlock::Thinking { text } => {
                self.parts.push((index, ContentPart::Thinking { text }));
            }
            OpenBlock::Tool {
                index: tool_index,
                id,
                name,
                args_json,
            } => {
                let arguments = parse_arguments(&args_json);
                self.parts.push((
                    tool_index,
                    ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    },
                ));
                let _ = index;
            }
        }
    }

    fn take_stream_error(&mut self) -> Option<String> {
        self.stream_error.take()
    }

    fn finish(mut self) -> Result<TurnResult, ClientError> {
        if !matches!(self.open, OpenBlock::None) {
            let idx = match &self.open {
                OpenBlock::Tool { index, .. } => *index,
                _ => self.parts.len(),
            };
            self.flush_open(idx);
        }

        self.parts.sort_by_key(|(i, _)| *i);
        let content: Vec<ContentPart> = self.parts.into_iter().map(|(_, p)| p).collect();

        let stop_reason = self
            .stop_reason
            .unwrap_or(StopReason::Other("missing_stop_reason".to_string()));

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{map_stop_reason, SseParser};

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
    fn probe_error_mapping() {
        let u = classify_probe_transport_error("Connection refused");
        assert!(matches!(u, ProbeError::Unreachable(_)));
        assert!(u.to_string().contains("unreachable"));

        let b = classify_probe_http_status(503, "nope");
        assert!(matches!(b, ProbeError::BadResponse(_)));
        assert!(b.to_string().contains("503"));
    }

    #[test]
    fn assemble_turn_from_happy_path_events() {
        const SSE: &str = "\
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
        let mut parser = SseParser::new();
        let mut assembler = TurnAssembler::new();
        for ev in parser.feed(SSE) {
            assembler.observe(&ev);
        }
        for ev in parser.finish() {
            assembler.observe(&ev);
        }
        let turn = assembler.finish().expect("assemble");
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
    fn map_stop_reasons() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::MaxTokens);
        assert_eq!(
            map_stop_reason("refusal"),
            StopReason::Other("refusal".to_string())
        );
    }
}
