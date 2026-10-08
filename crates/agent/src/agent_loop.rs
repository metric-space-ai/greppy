//! Agent loop: turn sequencing, tool-call dispatch, message threading.
//!
//! Ported from pi v0.80.2 (github.com/earendil-works/pi, MIT).
//!
//! # Semantics
//!
//! 1. Append the user prompt, then repeatedly:
//!    - Stream one assistant turn with the full history + tool schemas.
//!    - On `stop_reason == ToolUse`, execute every requested tool call **in
//!      order**, then append **one** user message carrying all
//!      `tool_result` blocks (matching call ids) and continue.
//!    - Stop on `EndTurn`, an opt-in `max_turns`, a wall-clock
//!      `deadline` (checked only between turns), or a non-recoverable
//!      transport error.
//! 2. Tool execution errors become `is_error: true` tool_results and do
//!    **not** abort the loop.
//! 3. Transport/connect errors: **retry once**, then surface a typed
//!    [`LoopError`]. (pi's full session-level auto-retry is richer; this
//!    port keeps a single immediate retry at the loop boundary.)
//! 4. Streaming is surfaced via [`LoopEvent`] (wrapping [`StreamEvent`] plus
//!    tool start/finish events).

use crate::client::{ClientError, TurnResult};
use crate::env::{ExecutionEnv, ToolOutcome};
use crate::model::ModelStream;
use crate::protocol::{
    ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, ToolChoice, Usage,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for [`run_agent_loop`].
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Optional cap on assistant action turns, followed by one tools-disabled report.
    /// Zero means unlimited (the default).
    pub max_turns: usize,
    /// Optional system prompt forwarded every turn.
    pub system: Option<String>,
    /// Saved context checkpoint, distinct from the signed mode contract.
    pub context_summary: Option<String>,
    /// Model tag placed on each [`ModelRequest`].
    pub model: String,
    /// Optional response limit; u64::MAX delegates the limit to the model provider.
    pub max_tokens: u64,
    /// Tool-choice policy. Default: [`ToolChoice::Auto`].
    pub tool_choice: ToolChoice,
    /// Legacy compatibility field, ignored: failures do not imply missing capability.
    pub consecutive_failure_advisory: usize,
    /// Legacy compatibility field, ignored: failures never end a run.
    pub consecutive_failure_stop: usize,
    /// Absolute wall-clock deadline. Checked only at the top of each loop
    /// iteration (before a new model turn). An `Instant` (not a duration) so
    /// tests can inject one. Default: `None` (no wall-clock limit).
    pub deadline: Option<Instant>,
    /// Original wall-clock budget used only for the low-time advisory
    /// (when remaining < 20% of this total). When `None`, the advisory is
    /// skipped even if [`Self::deadline`] is set. Default: `None`.
    pub deadline_total: Option<Duration>,
    /// Cooperative cancel flag. Checked only at safe boundaries: between
    /// turns, after a model stream finishes (before any tool starts), and
    /// after a tool returns. A running tool is never interrupted, so an
    /// in-flight edit cannot be left half-applied. Default: `None`.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl PartialEq for AgentConfig {
    fn eq(&self, other: &Self) -> bool {
        self.max_turns == other.max_turns
            && self.system == other.system
            && self.context_summary == other.context_summary
            && self.model == other.model
            && self.max_tokens == other.max_tokens
            && self.tool_choice == other.tool_choice
            && self.consecutive_failure_advisory == other.consecutive_failure_advisory
            && self.consecutive_failure_stop == other.consecutive_failure_stop
            && self.deadline == other.deadline
            && self.deadline_total == other.deadline_total
    }
}

impl Eq for AgentConfig {}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 0,
            system: None,
            context_summary: None,
            model: String::new(),
            max_tokens: u64::MAX,
            tool_choice: ToolChoice::Auto,
            consecutive_failure_advisory: 0,
            consecutive_failure_stop: 0,
            deadline: None,
            deadline_total: None,
            cancel: None,
        }
    }
}

impl AgentConfig {
    /// Builder: set max turns.
    pub fn with_max_turns(mut self, n: usize) -> Self {
        self.max_turns = n;
        self
    }

    /// Builder: set system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Builder: set model tag.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder: set max tokens.
    pub fn with_max_tokens(mut self, n: u64) -> Self {
        self.max_tokens = n;
        self
    }

    /// Legacy builder retained for compatibility; failure advisories are disabled.
    pub fn with_consecutive_failure_advisory(mut self, n: usize) -> Self {
        self.consecutive_failure_advisory = n;
        self
    }

    /// Legacy builder retained for compatibility; failure stops are disabled.
    pub fn with_consecutive_failure_stop(mut self, n: usize) -> Self {
        self.consecutive_failure_stop = n;
        self
    }
}

/// Why the agent loop stopped successfully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopStop {
    /// Model ended its turn with no pending tool calls.
    EndTurn,
    /// Hit [`AgentConfig::max_turns`].
    MaxTurns,
    /// Legacy result variant. Response token exhaustion now continues the run.
    MaxTokens,
    /// Legacy result variant. Repeated failures no longer stop the run.
    Stuck,
    /// Hit [`AgentConfig::deadline`] between turns (never mid-turn / mid-tool).
    Deadline,
    /// Cooperative cancel requested at a safe boundary (never mid-tool).
    Cancelled,
}

/// Successful loop outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopResult {
    /// Active conversation window to continue from: the user prompt and every
    /// assistant/tool-result message since the last context compaction. After
    /// a compaction this is *not* the full history; earlier messages live in
    /// the archive of [`LoopEvent::ContextCompacted`] and are summarized in
    /// [`LoopResult::context_summary`].
    pub messages: Vec<Message>,
    /// Continuation checkpoint kept separately from the signed mode contract.
    pub context_summary: Option<String>,
    /// Concatenated text from the final assistant message (empty if the last
    /// assistant turn had only tool calls / thinking).
    pub final_text: String,
    /// Why the loop stopped.
    pub stop: LoopStop,
    /// Usage summed across every completed model turn.
    pub usage: Usage,
}

/// Fatal loop errors (tool failures are *not* fatal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopError {
    /// Model transport/stream failure after the one allowed retry.
    Transport(String),
    /// Non-success HTTP from the gateway.
    Http { status: u16, body: String },
    /// Stream protocol / assembly failure.
    Stream(String),
    /// Incomplete turn assembly.
    Incomplete(String),
    /// Other client error.
    Client(String),
}

impl std::fmt::Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopError::Transport(m) => write!(f, "transport error: {m}"),
            LoopError::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            LoopError::Stream(m) => write!(f, "stream error: {m}"),
            LoopError::Incomplete(m) => write!(f, "incomplete turn: {m}"),
            LoopError::Client(m) => write!(f, "client error: {m}"),
        }
    }
}

impl std::error::Error for LoopError {}

impl From<ClientError> for LoopError {
    fn from(e: ClientError) -> Self {
        match e {
            ClientError::Transport(m) => LoopError::Transport(m),
            ClientError::Cancelled => LoopError::Transport("turn cancelled".into()),
            ClientError::Http { status, body } => LoopError::Http { status, body },
            ClientError::Stream(m) => LoopError::Stream(m),
            ClientError::Incomplete(m) => LoopError::Incomplete(m),
        }
    }
}

/// Events emitted by the agent loop while it runs.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopEvent {
    /// One complete compaction: archive before replacing active session state.
    ContextCompacted {
        archive: Vec<Message>,
        messages: Vec<Message>,
        summary: String,
    },
    /// A model-stream event for the current assistant turn.
    Stream(StreamEvent),
    /// About to execute a tool call.
    ToolStart {
        call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// Tool call finished (success or error).
    ToolFinish {
        call_id: String,
        name: String,
        outcome: ToolOutcome,
    },
    /// An assistant turn just completed (before any tool execution).
    TurnComplete {
        stop_reason: StopReason,
        usage: Usage,
    },
}

/// Run the agent loop to completion.
///
/// See the module docs for turn / tool / error semantics.
fn cancel_requested(config: &AgentConfig) -> bool {
    config
        .cancel
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}

fn cancelled_tool_result(call_id: String) -> ContentPart {
    ContentPart::ToolResult {
        call_id,
        content: "cancelled before execution".to_string(),
        is_error: true,
    }
}

fn cancelled_tool_results(tool_calls: &[(String, String, serde_json::Value)]) -> Vec<ContentPart> {
    tool_calls
        .iter()
        .map(|(id, _, _)| cancelled_tool_result(id.clone()))
        .collect()
}

pub fn run_agent_loop(
    model: &mut dyn ModelStream,
    env: &mut dyn ExecutionEnv,
    config: &AgentConfig,
    prompt: &str,
    on_event: &mut dyn FnMut(LoopEvent),
) -> Result<LoopResult, LoopError> {
    run_agent_loop_with_history(model, env, config, &[], prompt, on_event)
}

/// Continue an agent conversation from an existing message history.
///
/// The supplied history is never mutated. The returned [`LoopResult`] owns
/// the complete updated history, so interactive clients can feed its
/// `messages` field into the next call.
pub fn run_agent_loop_with_history(
    model: &mut dyn ModelStream,
    env: &mut dyn ExecutionEnv,
    config: &AgentConfig,
    history: &[Message],
    prompt: &str,
    on_event: &mut dyn FnMut(LoopEvent),
) -> Result<LoopResult, LoopError> {
    let mut messages: Vec<Message> = history.to_vec();
    let mut system = config.system.clone();
    crate::context::restore_summary(&mut system, config.context_summary.as_deref());
    messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::Text {
            text: prompt.to_string(),
        }],
    });

    let mut total_usage = Usage::default();
    let mut turns: usize = 0;
    // Every exit from the loop below sets this before `break`.
    let mut last_stop: LoopStop;
    let mut final_text = String::new();
    let mut turn_budget_advised = false;
    let mut deadline_advised = false;

    // An optional action cap reserves a final reporting response without tools.
    loop {
        // Wall-clock deadline is checked only between turns — never mid-turn
        // and never while a tool call is running, so a partial edit cannot be
        // left half-applied.
        if let Some(deadline) = config.deadline {
            if Instant::now() >= deadline {
                last_stop = LoopStop::Deadline;
                break;
            }
        }
        if cancel_requested(config) {
            last_stop = LoopStop::Cancelled;
            break;
        }

        // Snapshot the pre-compaction window only when a checkpoint is due,
        // not on every turn.
        const COMPACTION_BYTES: usize = 256 * 1024;
        if crate::context::compaction_due(&messages, COMPACTION_BYTES) {
            let template = ModelRequest {
                model: config.model.clone(),
                system: system.clone(),
                messages: Vec::new(),
                tools: Vec::new(),
                tool_choice: ToolChoice::None,
                max_tokens: config.max_tokens,
            };
            let archive = messages.clone();
            if let Some(usage) = crate::context::compact_with_model(
                model,
                &mut messages,
                &mut system,
                &template,
                COMPACTION_BYTES,
            )? {
                total_usage = sum_usage(total_usage, usage);
                on_event(LoopEvent::ContextCompacted {
                    archive,
                    messages: messages.clone(),
                    summary: crate::context::saved_summary(system.as_deref())
                        .unwrap()
                        .to_owned(),
                });
            }
        }
        if cancel_requested(config) {
            last_stop = LoopStop::Cancelled;
            break;
        }
        if config
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            last_stop = LoopStop::Deadline;
            break;
        }
        let report_only = config.max_turns > 0 && turns >= config.max_turns;
        if report_only {
            messages.push(Message {
                role: Role::User,
                content: vec![ContentPart::Text {
                    text: "The requested action-turn budget is complete. Provide the final report of changes, verification and remaining work using the evidence already obtained. Tools are disabled; do not perform further actions.".into(),
                }],
            });
        }
        let tools = if report_only {
            Vec::new()
        } else {
            env.tool_definitions()
        };
        let req = ModelRequest {
            model: config.model.clone(),
            system: system.clone(),
            messages: messages.clone(),
            tools,
            tool_choice: if report_only {
                ToolChoice::None
            } else {
                config.tool_choice
            },
            max_tokens: config.max_tokens,
        };

        let turn = match stream_turn_with_retry(model, &req, on_event) {
            Ok(turn) => turn,
            Err(ClientError::Cancelled) => {
                last_stop = LoopStop::Cancelled;
                break;
            }
            Err(error) => return Err(error.into()),
        };
        turns += 1;
        total_usage = sum_usage(total_usage, turn.usage);
        on_event(LoopEvent::TurnComplete {
            stop_reason: turn.stop_reason.clone(),
            usage: turn.usage,
        });

        messages.push(turn.message.clone());
        final_text = extract_text(&turn.message);

        // A model request is not interrupted mid-frame, but its completion is
        // a safe boundary. Without this check an interrupt received while an
        // EndTurn response was in flight was acknowledged and then silently
        // reported as a successful turn.
        if cancel_requested(config) && collect_tool_calls(&turn.message).is_empty() {
            last_stop = LoopStop::Cancelled;
            break;
        }

        if report_only {
            // Enforce the tools-disabled request even if a provider emits calls.
            // Record matching results so saved history remains protocol-valid.
            let rejected_calls = collect_tool_calls(&turn.message);
            if !rejected_calls.is_empty() {
                messages.push(Message {
                    role: Role::User,
                    content: rejected_calls.into_iter().map(|(id, _, _)| ContentPart::ToolResult {
                        call_id: id,
                        content: "The action-turn budget is exhausted; tools are disabled for the final report.".into(),
                        is_error: true,
                    }).collect(),
                });
            }
            last_stop = if cancel_requested(config) {
                LoopStop::Cancelled
            } else if config
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                LoopStop::Deadline
            } else {
                LoopStop::MaxTurns
            };
            break;
        }

        match turn.stop_reason {
            StopReason::ToolUse | StopReason::MaxTokens
                if !collect_tool_calls(&turn.message).is_empty() =>
            {
                let tool_calls = collect_tool_calls(&turn.message);
                if tool_calls.is_empty() {
                    // Model claimed tool_use but produced no calls — treat as end.
                    last_stop = LoopStop::EndTurn;
                    break;
                }

                // Stream finished and no tool has started: safe to honour cancel
                // without leaving an edit half-applied.
                if cancel_requested(config) {
                    messages.push(Message {
                        role: Role::User,
                        content: cancelled_tool_results(&tool_calls),
                    });
                    last_stop = LoopStop::Cancelled;
                    break;
                }

                let mut result_parts: Vec<ContentPart> = Vec::with_capacity(tool_calls.len());
                let mut cancel_remaining = false;
                for (id, name, arguments) in tool_calls {
                    if cancel_remaining {
                        result_parts.push(cancelled_tool_result(id));
                        continue;
                    }
                    on_event(LoopEvent::ToolStart {
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                    let mut outcome = dispatch_tool(env, &name, &arguments);

                    // Turn-budget awareness: once, when ≤25% of max_turns remain
                    // and at least one turn was already used.
                    // Not after the last action turn: the next request is the
                    // tool-free report turn, which carries its own instruction.
                    if !turn_budget_advised
                        && turns > 0
                        && config.max_turns > turns
                        && remaining_turns_at_or_below_quarter(turns, config.max_turns)
                    {
                        let remaining = config.max_turns.saturating_sub(turns);
                        append_tool_advisory(
                            &mut outcome.content,
                            &format!(
                                "{remaining} of {} turns left — wrap up: finish what is verifiable and report \
the rest.",
                                config.max_turns
                            ),
                        );
                        turn_budget_advised = true;
                    }

                    // Wall-clock budget awareness: once, when remaining time is
                    // below 20% of deadline_total. Skipped entirely when
                    // deadline_total is None.
                    if !deadline_advised {
                        if let (Some(deadline), Some(total)) =
                            (config.deadline, config.deadline_total)
                        {
                            if !total.is_zero()
                                && remaining_wall_below_fifth(deadline, total, Instant::now())
                            {
                                let remaining_secs =
                                    deadline.saturating_duration_since(Instant::now()).as_secs();
                                append_tool_advisory(
                                    &mut outcome.content,
                                    &format!(
                                        "{remaining_secs} seconds of wall clock left — finish what is \
verifiable and report the rest."
                                    ),
                                );
                                deadline_advised = true;
                            }
                        }
                    }

                    on_event(LoopEvent::ToolFinish {
                        call_id: id.clone(),
                        name: name.clone(),
                        outcome: outcome.clone(),
                    });
                    if let Some(data) = outcome.image_png_base64.clone() {
                        result_parts.push(ContentPart::Image {
                            media_type: "image/png".to_owned(),
                            data,
                        });
                    }
                    result_parts.push(ContentPart::ToolResult {
                        call_id: id,
                        content: outcome.content,
                        is_error: outcome.is_error,
                    });

                    // After a tool returns is a safe boundary. Remaining calls
                    // are recorded as cancelled results so history stays valid.
                    if cancel_requested(config) {
                        cancel_remaining = true;
                    }
                }

                // One user message carrying every tool_result block, in order.
                messages.push(Message {
                    role: Role::User,
                    content: result_parts,
                });

                if cancel_remaining || cancel_requested(config) {
                    last_stop = LoopStop::Cancelled;
                    break;
                }

                // Continue the outer loop for the next assistant turn.
                // After the last action turn, the next request permits only a final report.
                continue;
            }
            StopReason::EndTurn | StopReason::ToolUse => {
                last_stop = LoopStop::EndTurn;
                break;
            }
            StopReason::MaxTokens => {
                messages.push(Message {
                    role: Role::User,
                    content: vec![ContentPart::Text {
                        text: "The response reached its token limit. Continue from the interruption and complete the task.".into(),
                    }],
                });
                continue;
            }
            StopReason::Other(_) => {
                // Unknown stop: treat like end_turn so the loop does not hang.
                last_stop = LoopStop::EndTurn;
                break;
            }
        }
    }

    // If we never broke on a clean EndTurn/MaxTokens (including max_turns == 0
    // or always-tools exhaustion), the provisional MaxTurns stands. When the
    // final turn *did* end cleanly on the last allowed turn, honor that stop.
    // Stuck / Deadline already set last_stop and must not be overwritten.
    if turns == 0 && !matches!(last_stop, LoopStop::Deadline | LoopStop::Cancelled) {
        last_stop = LoopStop::MaxTurns;
    }

    Ok(LoopResult {
        messages,
        context_summary: crate::context::saved_summary(system.as_deref()).map(str::to_owned),
        final_text,
        stop: last_stop,
        usage: total_usage,
    })
}

/// Append one advisory line to a tool result body.
fn append_tool_advisory(content: &mut String, line: &str) {
    if !content.is_empty() && !content.ends_with('\n') {
        content.push('\n');
    }
    content.push_str(line);
}

/// True when turns used leave ≤25% of `max_turns` remaining.
fn remaining_turns_at_or_below_quarter(turns_used: usize, max_turns: usize) -> bool {
    if max_turns == 0 {
        return false;
    }
    let remaining = max_turns.saturating_sub(turns_used);
    // remaining / max_turns <= 0.25  ⇔  remaining * 4 <= max_turns
    remaining.saturating_mul(4) <= max_turns
}

/// True when remaining wall-clock time is strictly below 20% of `total`.
fn remaining_wall_below_fifth(deadline: Instant, total: Duration, now: Instant) -> bool {
    if total.is_zero() {
        return false;
    }
    let remaining = deadline.saturating_duration_since(now);
    // remaining < 0.2 * total  ⇔  remaining * 5 < total
    remaining
        .checked_mul(5)
        .map(|five| five < total)
        .unwrap_or(true)
}

/// Stream a turn; on a retryable failure, retry exactly once.
///
/// Retryable: pure transport/connect errors, and gateway-side transient
/// HTTP statuses (429 and the 5xx family, incl. Anthropic's 529). Client
/// errors (4xx other than 429) are surfaced immediately — retrying a bad
/// request or auth failure only burns quota.
fn stream_turn_with_retry(
    model: &mut dyn ModelStream,
    req: &ModelRequest,
    on_event: &mut dyn FnMut(LoopEvent),
) -> Result<TurnResult, ClientError> {
    match call_model(model, req, on_event) {
        Ok(t) => Ok(t),
        Err(first) if is_retryable(&first) => {
            // One immediate retry at the loop boundary.
            // Deviation from pi: pi's session-level auto-retry is configurable
            // and delayed; we do a single immediate retry.
            std::thread::sleep(std::time::Duration::from_secs(2));
            call_model(model, req, on_event)
        }
        Err(other) => Err(other),
    }
}

fn is_retryable(err: &ClientError) -> bool {
    match err {
        // 4xx (other than 429) means the request itself is wrong; everything
        // else — transport failures, 429/5xx, mid-stream error events,
        // truncated streams — is worth one retry.
        ClientError::Http { status, .. } => *status == 429 || *status >= 500,
        ClientError::Cancelled => false,
        _ => true,
    }
}

fn call_model(
    model: &mut dyn ModelStream,
    req: &ModelRequest,
    on_event: &mut dyn FnMut(LoopEvent),
) -> Result<TurnResult, ClientError> {
    model.stream_turn(req, &mut |ev| {
        on_event(LoopEvent::Stream(ev));
    })
}

fn dispatch_tool(
    env: &mut dyn ExecutionEnv,
    name: &str,
    arguments: &serde_json::Value,
) -> ToolOutcome {
    // The env is the source of truth. Contract: unknown names return
    // `is_error: true` (see [`ToolOutcome::unknown_tool`]); the loop never
    // panics on model misbehavior.
    env.call_tool(name, arguments)
}

fn collect_tool_calls(message: &Message) -> Vec<(String, String, serde_json::Value)> {
    message
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => Some((id.clone(), name.clone(), arguments.clone())),
            _ => None,
        })
        .collect()
}

fn extract_text(message: &Message) -> String {
    let mut out = String::new();
    for part in &message.content {
        if let ContentPart::Text { text } = part {
            out.push_str(text);
        }
    }
    out
}

fn sum_usage(a: Usage, b: Usage) -> Usage {
    Usage {
        input_tokens: a.input_tokens.saturating_add(b.input_tokens),
        output_tokens: a.output_tokens.saturating_add(b.output_tokens),
        cache_read_input_tokens: a
            .cache_read_input_tokens
            .saturating_add(b.cache_read_input_tokens),
        cache_creation_input_tokens: a
            .cache_creation_input_tokens
            .saturating_add(b.cache_creation_input_tokens),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ToolDefinition;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    // --- fakes ----------------------------------------------------------------

    /// One scripted model turn: either a successful [`TurnResult`] (with the
    /// events that should be emitted first) or a hard [`ClientError`].
    struct ScriptedTurn {
        events: Vec<StreamEvent>,
        result: Result<TurnResult, ClientError>,
    }

    struct FakeModel {
        turns: VecDeque<ScriptedTurn>,
        /// How many times `stream_turn` was invoked (includes retries).
        calls: usize,
        requests: Vec<ModelRequest>,
    }

    impl FakeModel {
        fn new(turns: Vec<ScriptedTurn>) -> Self {
            Self {
                turns: turns.into(),
                calls: 0,
                requests: Vec::new(),
            }
        }
    }

    impl ModelStream for FakeModel {
        fn stream_turn(
            &mut self,
            req: &ModelRequest,
            on_event: &mut dyn FnMut(StreamEvent),
        ) -> Result<TurnResult, ClientError> {
            self.calls += 1;
            self.requests.push(req.clone());
            let scripted = self
                .turns
                .pop_front()
                .expect("FakeModel: no more scripted turns");
            for ev in scripted.events {
                on_event(ev);
            }
            scripted.result
        }
    }

    struct FakeEnv {
        tools: Vec<ToolDefinition>,
        /// Canned outcomes by tool name. Missing → unknown_tool.
        outcomes: std::collections::HashMap<String, ToolOutcome>,
        /// Recorded (name, arguments) pairs in call order.
        calls: Vec<(String, serde_json::Value)>,
    }

    impl FakeEnv {
        fn new(tools: Vec<ToolDefinition>) -> Self {
            Self {
                tools,
                outcomes: std::collections::HashMap::new(),
                calls: Vec::new(),
            }
        }

        fn with_outcome(mut self, name: &str, outcome: ToolOutcome) -> Self {
            self.outcomes.insert(name.to_string(), outcome);
            self
        }
    }

    impl ExecutionEnv for FakeEnv {
        fn tool_definitions(&self) -> Vec<ToolDefinition> {
            self.tools.clone()
        }

        fn call_tool(&mut self, name: &str, arguments: &serde_json::Value) -> ToolOutcome {
            self.calls.push((name.to_string(), arguments.clone()));
            self.outcomes
                .get(name)
                .cloned()
                .unwrap_or_else(|| ToolOutcome::unknown_tool(name))
        }
    }

    fn text_turn(text: &str, usage: Usage) -> ScriptedTurn {
        ScriptedTurn {
            events: vec![
                StreamEvent::TextDelta {
                    text: text.to_string(),
                },
                StreamEvent::Finished {
                    stop_reason: StopReason::EndTurn,
                    usage,
                },
            ],
            result: Ok(TurnResult {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text {
                        text: text.to_string(),
                    }],
                },
                stop_reason: StopReason::EndTurn,
                usage,
            }),
        }
    }

    fn tool_turn(
        text: Option<&str>,
        calls: Vec<(&str, &str, serde_json::Value)>,
        usage: Usage,
    ) -> ScriptedTurn {
        let mut content = Vec::new();
        if let Some(t) = text {
            content.push(ContentPart::Text {
                text: t.to_string(),
            });
        }
        for (id, name, args) in &calls {
            content.push(ContentPart::ToolCall {
                id: (*id).to_string(),
                name: (*name).to_string(),
                arguments: args.clone(),
            });
        }
        ScriptedTurn {
            events: vec![StreamEvent::Finished {
                stop_reason: StopReason::ToolUse,
                usage,
            }],
            result: Ok(TurnResult {
                message: Message {
                    role: Role::Assistant,
                    content,
                },
                stop_reason: StopReason::ToolUse,
                usage,
            }),
        }
    }

    fn bash_tool() -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "run a command".to_string(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn echo_tool() -> ToolDefinition {
        ToolDefinition {
            name: "echo".to_string(),
            description: "echo".to_string(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        }
    }

    fn run(
        model: &mut dyn ModelStream,
        env: &mut dyn ExecutionEnv,
        config: &AgentConfig,
        prompt: &str,
    ) -> Result<(LoopResult, Vec<LoopEvent>), LoopError> {
        let mut events = Vec::new();
        let result = run_agent_loop(model, env, config, prompt, &mut |e| events.push(e))?;
        Ok((result, events))
    }

    // --- tests ----------------------------------------------------------------

    #[test]
    fn happy_path_text_only_one_turn() {
        let mut model = FakeModel::new(vec![text_turn("hello world", usage(10, 5))]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("loop");
        assert_eq!(result.final_text, "hello world");
        assert_eq!(result.stop, LoopStop::EndTurn);
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, Role::User);
        assert_eq!(result.messages[1].role, Role::Assistant);
        assert_eq!(model.calls, 1);
    }

    #[test]
    fn continued_loop_preserves_history_and_appends_prompt() {
        let history = vec![
            Message {
                role: Role::User,
                content: vec![ContentPart::Text {
                    text: "first question".to_string(),
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::Text {
                    text: "first answer".to_string(),
                }],
            },
        ];
        let mut model = FakeModel::new(vec![text_turn("second answer", usage(12, 4))]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let result = run_agent_loop_with_history(
            &mut model,
            &mut env,
            &config,
            &history,
            "second question",
            &mut |_| {},
        )
        .expect("continued loop");

        assert_eq!(model.requests.len(), 1);
        assert_eq!(model.requests[0].messages.len(), 3);
        assert_eq!(model.requests[0].messages[..2], history);
        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.messages[..2], history);
        assert_eq!(result.final_text, "second answer");
    }

    #[test]
    fn tool_loop_two_calls_then_end() {
        // Turn 1: two tool calls (bash, echo). Turn 2: final text.
        let mut model = FakeModel::new(vec![
            tool_turn(
                Some("I'll call two tools."),
                vec![
                    ("call_1", "bash", json!({"command": "ls"})),
                    ("call_2", "echo", json!({"text": "hi"})),
                ],
                usage(100, 40),
            ),
            text_turn("done", usage(120, 8)),
        ]);
        let mut env = FakeEnv::new(vec![bash_tool(), echo_tool()])
            .with_outcome("bash", ToolOutcome::ok("a\nb\n"))
            .with_outcome("echo", ToolOutcome::ok("hi"));
        let config = AgentConfig::default().with_model("mock");

        let (result, events) = run(&mut model, &mut env, &config, "do stuff").expect("loop");

        // Both tools executed in order.
        assert_eq!(
            env.calls
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["bash", "echo"]
        );

        // History shape: user, assistant(tool calls), user(two tool_results), assistant(final)
        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.messages[0].role, Role::User);
        assert_eq!(result.messages[1].role, Role::Assistant);
        assert_eq!(result.messages[2].role, Role::User);
        assert_eq!(result.messages[3].role, Role::Assistant);

        let results = &result.messages[2].content;
        assert_eq!(results.len(), 2);
        match &results[0] {
            ContentPart::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(content, "a\nb\n");
                assert!(!is_error);
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
        match &results[1] {
            ContentPart::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, "call_2");
                assert_eq!(content, "hi");
                assert!(!is_error);
            }
            other => panic!("expected tool_result, got {other:?}"),
        }

        assert_eq!(result.final_text, "done");
        assert_eq!(result.stop, LoopStop::EndTurn);

        // Tool start/finish events fired.
        let starts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, LoopEvent::ToolStart { .. }))
            .collect();
        let finishes: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, LoopEvent::ToolFinish { .. }))
            .collect();
        assert_eq!(starts.len(), 2);
        assert_eq!(finishes.len(), 2);

        // Usage summed.
        assert_eq!(result.usage.input_tokens, 220);
        assert_eq!(result.usage.output_tokens, 48);
    }

    #[test]
    fn long_task_compacts_in_the_real_loop_and_resumes_with_its_checkpoint() {
        struct CheckpointModel {
            actions: usize,
            checkpoints: usize,
        }
        impl ModelStream for CheckpointModel {
            fn stream_turn(
                &mut self,
                req: &ModelRequest,
                _: &mut dyn FnMut(StreamEvent),
            ) -> Result<TurnResult, ClientError> {
                if req
                    .system
                    .as_deref()
                    .is_some_and(|s| s.starts_with("Create an accurate continuation checkpoint"))
                {
                    assert!(req.tools.is_empty());
                    assert_eq!(req.tool_choice, ToolChoice::None);
                    self.checkpoints += 1;
                    return Ok(TurnResult {
                        message: Message { role: Role::Assistant, content: vec![ContentPart::Text {
                            text: json!({"task":"Fix clamp", "constraints":["Preserve operator changes"],
                                "decisions":["Clamp above upper bound"], "changed_files":["mathlib/ranges.py"],
                                "tests_results":["pytest passed"], "open_work":["Report verified change"],
                                "recovery_ids":["agent-output-fixture"]}).to_string()
                        }] }, stop_reason: StopReason::EndTurn, usage: usage(20, 5),
                    });
                }
                if self.actions == 60 {
                    return text_turn("Fix clamp\nStatus: done", usage(1, 1)).result;
                }
                self.actions += 1;
                tool_turn(
                    None,
                    vec![(&format!("call-{}", self.actions), "echo", json!({}))],
                    usage(1, 1),
                )
                .result
            }
        }
        let mut model = CheckpointModel {
            actions: 0,
            checkpoints: 0,
        };
        let mut env = FakeEnv::new(vec![echo_tool()]).with_outcome(
            "echo",
            ToolOutcome::ok(format!(
                "mathlib/ranges.py changed; pytest passed\n{}",
                "source evidence ".repeat(800)
            )),
        );
        let config = AgentConfig::default()
            .with_model("fixture")
            .with_system("Signed one-shot role");
        let (result, events) = run(
            &mut model,
            &mut env,
            &config,
            "Fix clamp; preserve operator changes",
        )
        .unwrap();
        assert_eq!(model.actions, 60);
        assert!(model.checkpoints >= 2);
        assert_eq!(result.stop, LoopStop::EndTurn);
        assert_eq!(result.final_text, "Fix clamp\nStatus: done");
        assert!(result.messages.len() < 122);
        assert!(
            events
                .iter()
                .filter(|event| matches!(event, LoopEvent::ContextCompacted { .. }))
                .count()
                >= 2
        );
        let state: serde_json::Value =
            serde_json::from_str(result.context_summary.as_deref().unwrap()).unwrap();
        assert_eq!(state["changed_files"][0], "mathlib/ranges.py");
        assert_eq!(state["tests_results"][0], "pytest passed");
        let mut followup = FakeModel::new(vec![text_turn("Follow-up done", usage(1, 1))]);
        let config = AgentConfig {
            system: Some("Signed interactive role".into()),
            context_summary: result.context_summary,
            ..AgentConfig::default()
        };
        run_agent_loop_with_history(
            &mut followup,
            &mut env,
            &config,
            &result.messages,
            "Continue with the next test",
            &mut |_| {},
        )
        .unwrap();
        let system = followup.requests[0].system.as_deref().unwrap();
        assert!(system.starts_with("Signed interactive role"));
        assert!(system.contains("mathlib/ranges.py"));
        assert!(system.contains("pytest passed"));
        assert!(system.contains("agent-output-fixture"));
        assert_eq!(system.matches(crate::context::CHECKPOINT_MARKER).count(), 1);
    }

    #[test]
    fn tool_error_continues_loop() {
        let mut model = FakeModel::new(vec![
            tool_turn(
                None,
                vec![("c1", "bash", json!({"command": "nope"}))],
                usage(10, 5),
            ),
            text_turn("recovered", usage(12, 3)),
        ]);
        let mut env =
            FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::err("boom"));
        let config = AgentConfig::default().with_model("mock");

        let (result, _) = run(&mut model, &mut env, &config, "try").expect("loop continues");
        assert_eq!(result.final_text, "recovered");
        assert_eq!(result.stop, LoopStop::EndTurn);

        match &result.messages[2].content[0] {
            ContentPart::ToolResult {
                is_error, content, ..
            } => {
                assert!(*is_error);
                assert_eq!(content, "boom");
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tool_name_is_error_and_continues() {
        let mut model = FakeModel::new(vec![
            tool_turn(None, vec![("c1", "does_not_exist", json!({}))], usage(1, 1)),
            text_turn("ok", usage(1, 1)),
        ]);
        // Env has no tools — unknown path.
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let (result, _) = run(&mut model, &mut env, &config, "x").expect("continues");
        assert_eq!(result.final_text, "ok");
        match &result.messages[2].content[0] {
            ContentPart::ToolResult {
                is_error, content, ..
            } => {
                assert!(*is_error);
                assert!(content.contains("unknown tool"));
                assert!(content.contains("does_not_exist"));
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn last_allowed_edit_is_followed_by_final_report_without_tools() {
        let mut model = FakeModel::new(vec![
            tool_turn(
                None,
                vec![("edit", "bash", json!({"command": "apply edit"}))],
                usage(1, 1),
            ),
            text_turn(
                "Edited clamp and verified the upper-bound regression.",
                usage(1, 1),
            ),
        ]);
        let mut env = FakeEnv::new(vec![bash_tool()])
            .with_outcome("bash", ToolOutcome::ok("edit and test passed"));
        let config = AgentConfig::default().with_max_turns(1);
        let (result, _) = run(&mut model, &mut env, &config, "fix clamp").unwrap();
        assert_eq!(result.stop, LoopStop::MaxTurns);
        assert_eq!(
            result.final_text,
            "Edited clamp and verified the upper-bound regression."
        );
        assert_eq!(env.calls.len(), 1);
        assert_eq!(model.calls, 2);
        let final_request = &model.requests[1];
        assert!(final_request.tools.is_empty());
        assert_eq!(final_request.tool_choice, ToolChoice::None);
        assert!(final_request.messages.iter().any(|message| message.content.iter().any(|part|
            matches!(part, ContentPart::ToolResult { content, .. } if content == "edit and test passed"))));
    }

    #[test]
    fn deadline_prevents_reporting_turn_after_last_action() {
        struct ExpiringEnv {
            deadline: Instant,
            calls: usize,
        }
        impl ExecutionEnv for ExpiringEnv {
            fn tool_definitions(&self) -> Vec<ToolDefinition> {
                vec![bash_tool()]
            }
            fn call_tool(&mut self, _: &str, _: &serde_json::Value) -> ToolOutcome {
                self.calls += 1;
                while Instant::now() < self.deadline {
                    std::thread::yield_now();
                }
                ToolOutcome::ok("last edit completed")
            }
        }
        let deadline = Instant::now() + Duration::from_millis(5);
        let mut env = ExpiringEnv { deadline, calls: 0 };
        let mut model = FakeModel::new(vec![tool_turn(
            None,
            vec![("edit", "bash", json!({}))],
            usage(1, 1),
        )]);
        let config = AgentConfig {
            max_turns: 1,
            deadline: Some(deadline),
            ..AgentConfig::default()
        };
        let (result, _) = run(&mut model, &mut env, &config, "fix").unwrap();
        assert_eq!(result.stop, LoopStop::Deadline);
        assert_eq!(model.calls, 1);
        assert_eq!(env.calls, 1);
    }

    #[test]
    fn max_turns_stops_when_model_always_requests_tools() {
        // Script more turns than the cap so the loop must self-stop.
        let mut turns = Vec::new();
        for i in 0..5 {
            let id = format!("c{i}");
            turns.push(ScriptedTurn {
                events: vec![StreamEvent::Finished {
                    stop_reason: StopReason::ToolUse,
                    usage: usage(1, 1),
                }],
                result: Ok(TurnResult {
                    message: Message {
                        role: Role::Assistant,
                        content: vec![ContentPart::ToolCall {
                            id,
                            name: "bash".to_string(),
                            arguments: json!({}),
                        }],
                    },
                    stop_reason: StopReason::ToolUse,
                    usage: usage(1, 1),
                }),
            });
        }

        let mut model = FakeModel::new(turns);
        let mut env = FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::ok("ok"));
        let config = AgentConfig::default().with_model("mock").with_max_turns(3);

        let (result, _) = run(&mut model, &mut env, &config, "loop forever").expect("ok");
        assert_eq!(result.stop, LoopStop::MaxTurns);
        // Three action turns and one tools-disabled reporting response.
        assert_eq!(model.calls, 4);
        assert!(model.requests[3].tools.is_empty());
        assert_eq!(model.requests[3].tool_choice, ToolChoice::None);
        // 3 tool executions.
        assert_eq!(env.calls.len(), 3);
        // Last disobedient tool call is rejected and recorded without execution.
        assert_eq!(result.messages.len(), 1 + 3 * 2 + 3);
        assert_eq!(result.usage.input_tokens, 4);
        assert_eq!(result.usage.output_tokens, 4);
    }

    #[test]
    fn transport_error_retries_once_then_fails() {
        let mut model = FakeModel::new(vec![
            ScriptedTurn {
                events: vec![],
                result: Err(ClientError::Transport("connection reset".into())),
            },
            ScriptedTurn {
                events: vec![],
                result: Err(ClientError::Transport("still down".into())),
            },
        ]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let err = run(&mut model, &mut env, &config, "hi").expect_err("must fail");
        assert!(matches!(err, LoopError::Transport(_)));
        assert!(err.to_string().contains("still down"));
        // Initial attempt + one retry.
        assert_eq!(model.calls, 2);
    }

    #[test]
    fn transport_error_recovers_on_retry() {
        let mut model = FakeModel::new(vec![
            ScriptedTurn {
                events: vec![],
                result: Err(ClientError::Transport("blip".into())),
            },
            text_turn("recovered", usage(5, 2)),
        ]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("retry works");
        assert_eq!(result.final_text, "recovered");
        assert_eq!(model.calls, 2);
        assert_eq!(result.stop, LoopStop::EndTurn);
    }

    #[test]
    fn http_client_error_does_not_retry() {
        let mut model = FakeModel::new(vec![ScriptedTurn {
            events: vec![],
            result: Err(ClientError::Http {
                status: 400,
                body: "bad request".into(),
            }),
        }]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let err = run(&mut model, &mut env, &config, "hi").expect_err("http fatal");
        assert!(matches!(err, LoopError::Http { status: 400, .. }));
        assert_eq!(model.calls, 1);
    }

    #[test]
    fn http_5xx_retries_once_then_fails() {
        let scripted = || ScriptedTurn {
            events: vec![],
            result: Err(ClientError::Http {
                status: 503,
                body: "overloaded".into(),
            }),
        };
        let mut model = FakeModel::new(vec![scripted(), scripted()]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let err = run(&mut model, &mut env, &config, "hi").expect_err("http fatal");
        assert!(matches!(err, LoopError::Http { status: 503, .. }));
        assert_eq!(model.calls, 2);
    }

    #[test]
    fn stream_error_retries_once_and_recovers() {
        let mut model = FakeModel::new(vec![
            ScriptedTurn {
                events: vec![],
                result: Err(ClientError::Stream(
                    "Service temporarily unavailable.".into(),
                )),
            },
            text_turn("recovered", usage(1, 1)),
        ]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let (result, _events) = run(&mut model, &mut env, &config, "hi").expect("recovers");
        assert_eq!(result.final_text, "recovered");
        assert_eq!(model.calls, 2);
    }

    #[test]
    fn default_turn_budget_does_not_stop_at_forty() {
        let mut turns = failing_tool_turns(45);
        turns.push(text_turn("done", usage(1, 1)));
        let mut model = FakeModel::new(turns);
        let mut env = FakeEnv::new(vec![bash_tool()]);
        let (result, _) = run(&mut model, &mut env, &AgentConfig::default(), "complete").unwrap();
        assert_eq!(result.stop, LoopStop::EndTurn);
        assert_eq!(model.calls, 46);
        assert_eq!(AgentConfig::default().max_tokens, u64::MAX);
    }

    #[test]
    fn max_tokens_continues() {
        let mut model = FakeModel::new(vec![
            ScriptedTurn {
                events: vec![StreamEvent::Finished {
                    stop_reason: StopReason::MaxTokens,
                    usage: usage(1, 99),
                }],
                result: Ok(TurnResult {
                    message: Message {
                        role: Role::Assistant,
                        content: vec![ContentPart::Text {
                            text: "cut off".into(),
                        }],
                    },
                    stop_reason: StopReason::MaxTokens,
                    usage: usage(1, 99),
                }),
            },
            text_turn("completed", usage(1, 1)),
        ]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");
        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("ok");
        assert_eq!(result.stop, LoopStop::EndTurn);
        assert_eq!(result.final_text, "completed");
        assert_eq!(model.calls, 2);
    }

    #[test]
    fn usage_summing_across_turns() {
        let mut model = FakeModel::new(vec![
            tool_turn(
                None,
                vec![("c1", "bash", json!({}))],
                Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                    cache_read_input_tokens: 1,
                    cache_creation_input_tokens: 2,
                },
            ),
            text_turn(
                "end",
                Usage {
                    input_tokens: 30,
                    output_tokens: 40,
                    cache_read_input_tokens: 3,
                    cache_creation_input_tokens: 4,
                },
            ),
        ]);
        let mut env = FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::ok("x"));
        let config = AgentConfig::default().with_model("mock");
        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("ok");
        assert_eq!(result.usage.input_tokens, 40);
        assert_eq!(result.usage.output_tokens, 60);
        assert_eq!(result.usage.cache_read_input_tokens, 4);
        assert_eq!(result.usage.cache_creation_input_tokens, 6);
    }

    #[test]
    fn system_and_tools_forwarded_each_turn() {
        struct CaptureModel {
            last_req: Option<ModelRequest>,
        }
        impl ModelStream for CaptureModel {
            fn stream_turn(
                &mut self,
                req: &ModelRequest,
                _on_event: &mut dyn FnMut(StreamEvent),
            ) -> Result<TurnResult, ClientError> {
                self.last_req = Some(req.clone());
                Ok(TurnResult {
                    message: Message {
                        role: Role::Assistant,
                        content: vec![ContentPart::Text { text: "ok".into() }],
                    },
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
        let mut model = CaptureModel { last_req: None };
        let mut env = FakeEnv::new(vec![bash_tool()]);
        let config = AgentConfig::default()
            .with_model("claude-test")
            .with_system("be good")
            .with_max_tokens(1234);
        let mut events = Vec::new();
        let _ = run_agent_loop(&mut model, &mut env, &config, "hi", &mut |e| {
            events.push(e);
        })
        .expect("ok");
        let req = model.last_req.expect("captured");
        assert_eq!(req.model, "claude-test");
        assert_eq!(req.system.as_deref(), Some("be good"));
        assert_eq!(req.max_tokens, 1234);
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "bash");
        assert_eq!(req.messages.len(), 1);
    }

    /// Sequence helper: N failing tool turns, optional success, then more fails / end.
    fn failing_tool_turns(n: usize) -> Vec<ScriptedTurn> {
        (0..n)
            .map(|i| {
                tool_turn(
                    None,
                    vec![(&format!("c{i}"), "bash", json!({}))],
                    usage(1, 1),
                )
            })
            .collect()
    }

    fn tool_result_contents(result: &LoopResult) -> Vec<(bool, String)> {
        let mut out = Vec::new();
        for msg in &result.messages {
            if msg.role != Role::User {
                continue;
            }
            for part in &msg.content {
                if let ContentPart::ToolResult {
                    content, is_error, ..
                } = part
                {
                    out.push((*is_error, content.clone()));
                }
            }
        }
        out
    }

    #[test]
    fn consecutive_failure_counter_resets_on_success() {
        // 3 fails + 1 success + 3 fails → advisory never fires at threshold 4;
        // counter reset by the success in the middle.
        let mut turns = failing_tool_turns(3);
        turns.push(tool_turn(
            None,
            vec![("ok1", "bash", json!({}))],
            usage(1, 1),
        ));
        turns.extend(failing_tool_turns(3));
        turns.push(text_turn("done", usage(1, 1)));

        // FakeEnv with sequential outcomes via a queue-backed env.
        struct SeqEnv {
            tools: Vec<ToolDefinition>,
            outcomes: VecDeque<ToolOutcome>,
            calls: usize,
        }
        impl ExecutionEnv for SeqEnv {
            fn tool_definitions(&self) -> Vec<ToolDefinition> {
                self.tools.clone()
            }
            fn call_tool(&mut self, name: &str, _arguments: &serde_json::Value) -> ToolOutcome {
                self.calls += 1;
                self.outcomes
                    .pop_front()
                    .unwrap_or_else(|| ToolOutcome::unknown_tool(name))
            }
        }

        let mut outcomes = VecDeque::new();
        for _ in 0..3 {
            outcomes.push_back(ToolOutcome::err("fail"));
        }
        outcomes.push_back(ToolOutcome::ok("ok"));
        for _ in 0..3 {
            outcomes.push_back(ToolOutcome::err("fail"));
        }

        let mut model = FakeModel::new(turns);
        let mut env = SeqEnv {
            tools: vec![bash_tool()],
            outcomes,
            calls: 0,
        };
        let config = AgentConfig::default()
            .with_model("mock")
            .with_max_turns(20)
            .with_consecutive_failure_advisory(4)
            .with_consecutive_failure_stop(8);

        let mut events = Vec::new();
        let result = run_agent_loop(&mut model, &mut env, &config, "x", &mut |e| {
            events.push(e);
        })
        .expect("loop");

        assert_eq!(result.stop, LoopStop::EndTurn);
        assert_eq!(env.calls, 7);
        let advisory = "4 tool calls in a row failed";
        let contents = tool_result_contents(&result);
        assert!(
            contents.iter().all(|(_, c)| !c.contains(advisory)),
            "advisory must not fire when streak resets: {contents:?}"
        );
    }

    #[test]
    fn repeated_failures_do_not_stop_or_advise() {
        // 8 consecutive fails → Stuck; loop returns what it has (no further model turn).
        struct SeqEnv {
            tools: Vec<ToolDefinition>,
            outcomes: VecDeque<ToolOutcome>,
            calls: usize,
        }
        impl ExecutionEnv for SeqEnv {
            fn tool_definitions(&self) -> Vec<ToolDefinition> {
                self.tools.clone()
            }
            fn call_tool(&mut self, name: &str, _arguments: &serde_json::Value) -> ToolOutcome {
                self.calls += 1;
                self.outcomes
                    .pop_front()
                    .unwrap_or_else(|| ToolOutcome::unknown_tool(name))
            }
        }

        // Provide more scripted turns than needed so Stuck is what stops us.
        let mut turns = failing_tool_turns(12);
        turns.push(text_turn("should not reach", usage(1, 1)));

        let mut outcomes = VecDeque::new();
        for _ in 0..12 {
            outcomes.push_back(ToolOutcome::err("fail"));
        }

        let mut model = FakeModel::new(turns);
        let mut env = SeqEnv {
            tools: vec![bash_tool()],
            outcomes,
            calls: 0,
        };
        let config = AgentConfig::default()
            .with_model("mock")
            .with_max_turns(20)
            .with_consecutive_failure_advisory(4)
            .with_consecutive_failure_stop(8);

        let result = run_agent_loop(&mut model, &mut env, &config, "x", &mut |_| {}).expect("loop");
        assert_eq!(result.stop, LoopStop::EndTurn);
        assert_eq!(env.calls, 12, "all tool calls must run");
        // 8 tool results present in history.
        let contents = tool_result_contents(&result);
        assert_eq!(contents.len(), 12);
        // Advisory at 4th is still present.
        assert!(
            !contents[3].1.contains("4 tool calls in a row failed"),
            "advisory on 4th: {:?}",
            contents[3]
        );
    }

    #[test]
    fn turn_budget_advisory_once_when_quarter_or_less_remain() {
        // max_turns=4 → after turn 3, remaining=1 → 1*4=4 <= 4, so advise.
        // Advise once only, even if more tools run in later turns.
        struct AlwaysOkEnv {
            tools: Vec<ToolDefinition>,
        }
        impl ExecutionEnv for AlwaysOkEnv {
            fn tool_definitions(&self) -> Vec<ToolDefinition> {
                self.tools.clone()
            }
            fn call_tool(&mut self, _name: &str, _arguments: &serde_json::Value) -> ToolOutcome {
                ToolOutcome::ok("ok")
            }
        }

        let mut turns = Vec::new();
        for i in 0..3 {
            turns.push(tool_turn(
                None,
                vec![(&format!("c{i}"), "bash", json!({}))],
                usage(1, 1),
            ));
        }
        turns.push(text_turn("wrapped up", usage(1, 1)));

        let mut model = FakeModel::new(turns);
        let mut env = AlwaysOkEnv {
            tools: vec![bash_tool()],
        };
        let config = AgentConfig::default().with_model("mock").with_max_turns(4);

        let result = run_agent_loop(&mut model, &mut env, &config, "x", &mut |_| {}).expect("loop");
        let contents = tool_result_contents(&result);
        assert_eq!(contents.len(), 3);
        let budget_hits: Vec<_> = contents
            .iter()
            .filter(|(_, c)| c.contains("turns left — wrap up"))
            .collect();
        assert_eq!(budget_hits.len(), 1, "budget advisory once: {contents:?}");
        // Fires on the first tool result of the turn that crosses the threshold
        // (turn 3 of 4 → remaining 1).
        assert!(
            contents[2].1.contains("1 of 4 turns left — wrap up"),
            "content={}",
            contents[2].1
        );
        // Earlier results must not have it.
        assert!(!contents[0].1.contains("turns left"));
        assert!(!contents[1].1.contains("turns left"));
    }

    #[test]
    fn remaining_turns_quarter_math() {
        assert!(!remaining_turns_at_or_below_quarter(1, 40)); // 39 left
        assert!(remaining_turns_at_or_below_quarter(30, 40)); // 10 left = 25%
        assert!(remaining_turns_at_or_below_quarter(31, 40)); // 9 left < 25%
        assert!(!remaining_turns_at_or_below_quarter(29, 40)); // 11 left > 25%
        assert!(remaining_turns_at_or_below_quarter(3, 4)); // 1 left = 25%
        assert!(!remaining_turns_at_or_below_quarter(2, 4)); // 2 left = 50%
        assert!(!remaining_turns_at_or_below_quarter(0, 0));
    }

    #[test]
    fn remaining_wall_below_fifth_math() {
        let now = Instant::now();
        let total = Duration::from_secs(100);
        // 10s left = 10% < 20%
        assert!(remaining_wall_below_fifth(
            now + Duration::from_secs(10),
            total,
            now
        ));
        // 20s left = 20% is NOT strictly below 20%
        assert!(!remaining_wall_below_fifth(
            now + Duration::from_secs(20),
            total,
            now
        ));
        // 21s left > 20%
        assert!(!remaining_wall_below_fifth(
            now + Duration::from_secs(21),
            total,
            now
        ));
        // zero total → never advise
        assert!(!remaining_wall_below_fifth(
            now + Duration::from_secs(1),
            Duration::ZERO,
            now
        ));
    }

    #[test]
    fn deadline_already_past_stops_before_first_model_turn() {
        let mut model = FakeModel::new(vec![text_turn("should not run", usage(1, 1))]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig {
            model: "mock".into(),
            deadline: Some(Instant::now() - Duration::from_secs(1)),
            deadline_total: Some(Duration::from_secs(60)),
            ..AgentConfig::default()
        };

        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("loop");
        assert_eq!(result.stop, LoopStop::Deadline);
        assert_eq!(model.calls, 0, "must not start a model turn");
        // Empty-but-valid: user prompt only, no assistant turn, zero usage.
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, Role::User);
        assert!(result.final_text.is_empty());
        assert_eq!(result.usage, Usage::default());
    }

    #[test]
    fn deadline_after_two_tool_turns_stops_without_third() {
        // Both early turns request tools so the loop continues; sleep after the
        // second model turn so the next top-of-loop deadline check fires.
        struct SleepOnCallModel {
            inner: FakeModel,
            sleep_on_call: usize,
            sleep_for: Duration,
            calls: usize,
        }
        impl ModelStream for SleepOnCallModel {
            fn stream_turn(
                &mut self,
                req: &ModelRequest,
                on_event: &mut dyn FnMut(StreamEvent),
            ) -> Result<TurnResult, ClientError> {
                self.calls += 1;
                let n = self.calls;
                let result = self.inner.stream_turn(req, on_event);
                if n == self.sleep_on_call {
                    std::thread::sleep(self.sleep_for);
                }
                result
            }
        }

        let mut model = SleepOnCallModel {
            inner: FakeModel::new(vec![
                tool_turn(
                    Some("turn one"),
                    vec![("c1", "bash", json!({}))],
                    usage(10, 5),
                ),
                tool_turn(
                    Some("turn two"),
                    vec![("c2", "bash", json!({}))],
                    usage(20, 6),
                ),
                text_turn("turn three must not run", usage(1, 1)),
            ]),
            sleep_on_call: 2,
            // Load-robust: outlast the deadline by a wide margin so turn 3 is
            // never requested even under host contention.
            sleep_for: Duration::from_secs(3),
            calls: 0,
        };
        let mut env = FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::ok("ok"));
        let config = AgentConfig {
            model: "mock".into(),
            max_turns: 10,
            // Enough headroom for turns 1–2 under load; the sleep on call 2
            // pushes past it before turn 3 can start.
            deadline: Some(Instant::now() + Duration::from_secs(1)),
            deadline_total: Some(Duration::from_secs(60)),
            ..AgentConfig::default()
        };

        let result =
            run_agent_loop(&mut model, &mut env, &config, "hi", &mut |_| {}).expect("loop");
        assert_eq!(result.stop, LoopStop::Deadline);
        assert_eq!(model.calls, 2, "turn 3 must never be requested");
        // History: user, asst1, tools1, asst2, tools2
        assert_eq!(result.messages.len(), 5);
        assert_eq!(result.messages[0].role, Role::User);
        assert_eq!(result.messages[1].role, Role::Assistant);
        assert_eq!(result.messages[2].role, Role::User);
        assert_eq!(result.messages[3].role, Role::Assistant);
        assert_eq!(result.messages[4].role, Role::User);
        assert_eq!(extract_text(&result.messages[1]), "turn one");
        assert_eq!(extract_text(&result.messages[3]), "turn two");
        assert_eq!(result.usage.input_tokens, 30);
        assert_eq!(result.usage.output_tokens, 11);
        assert_eq!(env.calls.len(), 2);
    }

    #[test]
    fn running_tool_is_not_interrupted_by_deadline() {
        // Deadline passes while a tool is running: the env call must complete
        // and its result must land in history; only the *next* model turn is
        // skipped.
        struct SlowEnv {
            tools: Vec<ToolDefinition>,
            calls: usize,
            sleep_for: Duration,
        }
        impl ExecutionEnv for SlowEnv {
            fn tool_definitions(&self) -> Vec<ToolDefinition> {
                self.tools.clone()
            }
            fn call_tool(&mut self, _name: &str, _arguments: &serde_json::Value) -> ToolOutcome {
                self.calls += 1;
                std::thread::sleep(self.sleep_for);
                ToolOutcome::ok("tool finished fully")
            }
        }

        let mut model = FakeModel::new(vec![
            tool_turn(
                Some("calling tool"),
                vec![("c1", "bash", json!({"command": "slow"}))],
                usage(1, 1),
            ),
            text_turn("must not run", usage(1, 1)),
        ]);
        let mut env = SlowEnv {
            tools: vec![bash_tool()],
            calls: 0,
            // Load-robust: tool outlasts the deadline so the next top-of-loop
            // check fires after the tool fully completes.
            sleep_for: Duration::from_secs(3),
        };
        let config = AgentConfig {
            model: "mock".into(),
            max_turns: 10,
            deadline: Some(Instant::now() + Duration::from_secs(1)),
            deadline_total: Some(Duration::from_secs(60)),
            ..AgentConfig::default()
        };

        let result =
            run_agent_loop(&mut model, &mut env, &config, "hi", &mut |_| {}).expect("loop");
        assert_eq!(result.stop, LoopStop::Deadline);
        assert_eq!(model.calls, 1, "no second model turn");
        assert_eq!(env.calls, 1, "tool must complete");
        let contents = tool_result_contents(&result);
        assert_eq!(contents.len(), 1);
        assert!(!contents[0].0);
        // Tool body is present even if the low-time advisory was also appended
        // (deadline_total is set and remaining may already be <20% after sleep).
        assert!(
            contents[0].1.starts_with("tool finished fully"),
            "tool result must be threaded into history: {}",
            contents[0].1
        );
    }

    #[test]
    fn wall_clock_advisory_once_only_when_deadline_total_set() {
        // deadline far enough that the loop finishes; remaining << 20% of total
        // so the advisory fires on the first tool result, exactly once.
        let mut model = FakeModel::new(vec![
            tool_turn(None, vec![("c1", "bash", json!({}))], usage(1, 1)),
            tool_turn(None, vec![("c2", "bash", json!({}))], usage(1, 1)),
            text_turn("done", usage(1, 1)),
        ]);
        let mut env = FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::ok("ok"));
        let now = Instant::now();
        let config = AgentConfig {
            model: "mock".into(),
            max_turns: 10,
            // Plenty of wall time to finish the fake turns.
            deadline: Some(now + Duration::from_secs(30)),
            // Total 200s → 30 left is 15% < 20% → advisory fires.
            deadline_total: Some(Duration::from_secs(200)),
            ..AgentConfig::default()
        };

        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("loop");
        assert_eq!(result.stop, LoopStop::EndTurn);
        let contents = tool_result_contents(&result);
        assert_eq!(contents.len(), 2);
        let hits: Vec<_> = contents
            .iter()
            .filter(|(_, c)| c.contains("seconds of wall clock left"))
            .collect();
        assert_eq!(hits.len(), 1, "advisory exactly once: {contents:?}");
        assert!(
            contents[0]
                .1
                .contains("seconds of wall clock left — finish what is verifiable"),
            "first tool result should carry advisory: {}",
            contents[0].1
        );
        assert!(
            !contents[1].1.contains("wall clock left"),
            "second must not re-advise: {}",
            contents[1].1
        );
    }

    #[test]
    fn wall_clock_advisory_skipped_without_deadline_total() {
        let mut model = FakeModel::new(vec![
            tool_turn(None, vec![("c1", "bash", json!({}))], usage(1, 1)),
            text_turn("done", usage(1, 1)),
        ]);
        let mut env = FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::ok("ok"));
        let config = AgentConfig {
            model: "mock".into(),
            // Deadline set (and near) but total absent → no advisory.
            deadline: Some(Instant::now() + Duration::from_secs(1)),
            deadline_total: None,
            ..AgentConfig::default()
        };

        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("loop");
        let contents = tool_result_contents(&result);
        assert!(
            contents.iter().all(|(_, c)| !c.contains("wall clock left")),
            "no advisory without deadline_total: {contents:?}"
        );
    }

    #[test]
    fn no_deadline_matches_default_behaviour() {
        // Regression: default config (no deadline fields) is unchanged.
        let mut model = FakeModel::new(vec![
            tool_turn(
                Some("working"),
                vec![("c1", "bash", json!({}))],
                usage(10, 5),
            ),
            text_turn("all done", usage(12, 3)),
        ]);
        let mut env = FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::ok("ok"));
        let config = AgentConfig::default().with_model("mock");
        assert!(config.deadline.is_none());
        assert!(config.deadline_total.is_none());

        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("loop");
        assert_eq!(result.stop, LoopStop::EndTurn);
        assert_eq!(result.final_text, "all done");
        assert_eq!(model.calls, 2);
        assert_eq!(env.calls.len(), 1);
        assert_eq!(result.messages.len(), 4);
    }

    #[test]
    fn cancelled_transport_stops_without_retry_or_tool_execution() {
        let mut model = FakeModel::new(vec![ScriptedTurn {
            events: vec![],
            result: Err(ClientError::Cancelled),
        }]);
        let mut env = FakeEnv::new(vec![bash_tool()]);
        let config = AgentConfig::default().with_model("mock");
        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("cancelled loop");
        assert_eq!(result.stop, LoopStop::Cancelled);
        assert_eq!(model.calls, 1);
        assert!(env.calls.is_empty());
        assert_eq!(result.messages.len(), 1);
    }

    #[test]
    fn cancel_before_first_turn_does_not_call_model() {
        let flag = Arc::new(AtomicBool::new(true));
        let mut model = FakeModel::new(vec![text_turn("must not run", usage(1, 1))]);
        let mut env = FakeEnv::new(vec![bash_tool()]);
        let config = AgentConfig {
            model: "mock".into(),
            cancel: Some(Arc::clone(&flag)),
            ..AgentConfig::default()
        };
        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("loop");
        assert_eq!(result.stop, LoopStop::Cancelled);
        assert_eq!(model.calls, 0);
        assert!(env.calls.is_empty());
    }

    #[test]
    fn cancel_after_stream_skips_tools() {
        let flag = Arc::new(AtomicBool::new(false));
        struct CancelAfterStream {
            inner: FakeModel,
            flag: Arc<AtomicBool>,
        }
        impl ModelStream for CancelAfterStream {
            fn stream_turn(
                &mut self,
                req: &ModelRequest,
                on_event: &mut dyn FnMut(StreamEvent),
            ) -> Result<TurnResult, ClientError> {
                let result = self.inner.stream_turn(req, on_event)?;
                self.flag.store(true, Ordering::Relaxed);
                Ok(result)
            }
        }
        let mut model = CancelAfterStream {
            inner: FakeModel::new(vec![tool_turn(
                Some("calling"),
                vec![("c1", "bash", json!({"command": "rm"}))],
                usage(1, 1),
            )]),
            flag: Arc::clone(&flag),
        };
        let mut env = FakeEnv::new(vec![bash_tool()]).with_outcome("bash", ToolOutcome::ok("ran"));
        let config = AgentConfig {
            model: "mock".into(),
            cancel: Some(Arc::clone(&flag)),
            ..AgentConfig::default()
        };
        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("loop");
        assert_eq!(result.stop, LoopStop::Cancelled);
        assert!(env.calls.is_empty(), "tool must not start after cancel");
        let contents = tool_result_contents(&result);
        assert_eq!(contents.len(), 1);
        assert!(contents[0].0);
        assert!(contents[0].1.contains("cancelled before execution"));
    }

    #[test]
    fn running_tool_is_not_interrupted_by_cancel() {
        let flag = Arc::new(AtomicBool::new(false));
        struct SlowEnv {
            tools: Vec<ToolDefinition>,
            calls: usize,
            sleep_for: Duration,
            flag: Arc<AtomicBool>,
        }
        impl ExecutionEnv for SlowEnv {
            fn tool_definitions(&self) -> Vec<ToolDefinition> {
                self.tools.clone()
            }
            fn call_tool(&mut self, _name: &str, _arguments: &serde_json::Value) -> ToolOutcome {
                self.calls += 1;
                self.flag.store(true, Ordering::Relaxed);
                std::thread::sleep(self.sleep_for);
                ToolOutcome::ok("tool finished fully")
            }
        }
        let mut model = FakeModel::new(vec![
            tool_turn(
                Some("calling tool"),
                vec![("c1", "bash", json!({"command": "slow"}))],
                usage(1, 1),
            ),
            text_turn("must not run", usage(1, 1)),
        ]);
        let mut env = SlowEnv {
            tools: vec![bash_tool()],
            calls: 0,
            sleep_for: Duration::from_millis(50),
            flag: Arc::clone(&flag),
        };
        let config = AgentConfig {
            model: "mock".into(),
            max_turns: 10,
            cancel: Some(Arc::clone(&flag)),
            ..AgentConfig::default()
        };
        let result =
            run_agent_loop(&mut model, &mut env, &config, "hi", &mut |_| {}).expect("loop");
        assert_eq!(result.stop, LoopStop::Cancelled);
        assert_eq!(model.calls, 1, "no second model turn");
        assert_eq!(env.calls, 1, "tool must complete");
        let contents = tool_result_contents(&result);
        assert_eq!(contents.len(), 1);
        assert!(contents[0].1.starts_with("tool finished fully"));
    }
}
