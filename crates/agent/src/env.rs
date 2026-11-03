//! Execution environment: the agent/host tool boundary.
//!
//! The agent loop *proposes* tool calls; the host *executes* them via
//! [`ExecutionEnv`]. This mirrors the pi/ctox sidecar contract in shape:
//! tools are advertised as schemas, invoked by name with JSON arguments, and
//! return text content plus an error flag. The loop never panics on model
//! misbehavior (unknown tool names become `is_error` results).

use serde_json::Value;

use crate::protocol::ToolDefinition;

/// Outcome of a single tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutcome {
    /// Human-readable (or machine-readable) tool result body.
    pub content: String,
    /// When `true`, the model should treat this as a failed tool call.
    pub is_error: bool,
    /// Optional still image (PNG base64) shown to the model, not logged.
    pub image_png_base64: Option<String>,
}

impl ToolOutcome {
    /// Successful tool result.
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            image_png_base64: None,
        }
    }

    /// Failed tool result.
    pub fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            image_png_base64: None,
        }
    }

    /// Standard unknown-tool error payload.
    pub fn unknown_tool(name: &str) -> Self {
        Self::err(format!("unknown tool: {name}"))
    }
}

/// Host-side tool boundary used by the agent loop.
///
/// # Contract
///
/// - [`tool_definitions`](ExecutionEnv::tool_definitions) is re-read every
///   turn and advertised to the model.
/// - [`call_tool`](ExecutionEnv::call_tool) must not panic on unknown names or
///   bad arguments; return [`ToolOutcome::is_error`] `true` instead.
pub trait ExecutionEnv {
    /// Tool schemas currently available to the model.
    fn tool_definitions(&self) -> Vec<ToolDefinition>;

    /// Execute a tool by name with JSON arguments.
    ///
    /// Unknown tool names must return an error outcome (see
    /// [`ToolOutcome::unknown_tool`]); the loop relies on this instead of
    /// panicking.
    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome;
}
