//! Provider-neutral agent protocol types.
//!
//! These types intentionally do not mirror any single vendor schema. The
//! Anthropic Messages wire shape lives in [`crate::wire`]; other adapters can
//! map the same types later without changing the agent loop.

use serde_json::Value;

/// One model turn request assembled by the agent loop.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub max_tokens: u64,
}

/// A single conversational message (user or assistant).
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

/// Speaker role for a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One content block inside a message.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        call_id: String,
        content: String,
        is_error: bool,
    },
    Thinking {
        text: String,
    },
    /// PNG (or other still image) shown to the model. Keep out of logs/traces.
    Image {
        media_type: String,
        data: String,
    },
}

/// Tool schema advertised to the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// How the model should select tools for this turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

/// Token accounting for a completed turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

/// Incremental events produced while streaming one model turn.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Started {
        model: String,
    },
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },
    ToolCallStarted {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: usize,
        json_fragment: String,
    },
    BlockFinished {
        index: usize,
    },
    Finished {
        stop_reason: StopReason,
        usage: Usage,
    },
    Error {
        message: String,
    },
}
