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
//!    - Stop on `EndTurn`, `MaxTokens`, configured `max_turns`, or a
//!      non-recoverable transport error.
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

/// Configuration for [`run_agent_loop`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    /// Hard cap on assistant turns (each model response counts as one).
    /// Default: 40.
    pub max_turns: usize,
    /// Optional system prompt forwarded every turn.
    pub system: Option<String>,
    /// Model tag placed on each [`ModelRequest`].
    pub model: String,
    /// `max_tokens` for each model turn. Default: 8192.
    pub max_tokens: u64,
    /// Tool-choice policy. Default: [`ToolChoice::Auto`].
    pub tool_choice: ToolChoice,
    /// Consecutive failed tool outcomes before an advisory is appended to the
    /// tool result. Default: 4.
    pub consecutive_failure_advisory: usize,
    /// Consecutive failed tool outcomes before the loop stops as
    /// [`LoopStop::Stuck`]. Default: 8.
    pub consecutive_failure_stop: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 40,
            system: None,
            model: String::new(),
            max_tokens: 8192,
            tool_choice: ToolChoice::Auto,
            consecutive_failure_advisory: 4,
            consecutive_failure_stop: 8,
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

    /// Builder: consecutive-failure advisory threshold.
    pub fn with_consecutive_failure_advisory(mut self, n: usize) -> Self {
        self.consecutive_failure_advisory = n;
        self
    }

    /// Builder: consecutive-failure stop threshold ([`LoopStop::Stuck`]).
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
    /// Model hit its generation token budget.
    MaxTokens,
    /// Hit [`AgentConfig::consecutive_failure_stop`] failed tool outcomes in a
    /// row — the agent could not make progress.
    Stuck,
}

/// Successful loop outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopResult {
    /// Full conversation history (user prompt + every assistant/tool-result
    /// message produced during the run).
    pub messages: Vec<Message>,
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
            ClientError::Http { status, body } => LoopError::Http { status, body },
            ClientError::Stream(m) => LoopError::Stream(m),
            ClientError::Incomplete(m) => LoopError::Incomplete(m),
        }
    }
}

/// Events emitted by the agent loop while it runs.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopEvent {
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
pub fn run_agent_loop(
    model: &mut dyn ModelStream,
    env: &mut dyn ExecutionEnv,
    config: &AgentConfig,
    prompt: &str,
    on_event: &mut dyn FnMut(LoopEvent),
) -> Result<LoopResult, LoopError> {
    let mut messages: Vec<Message> = Vec::new();
    messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::Text {
            text: prompt.to_string(),
        }],
    });

    let mut total_usage = Usage::default();
    let mut turns: usize = 0;
    let mut last_stop = LoopStop::EndTurn;
    let mut final_text = String::new();
    let mut consecutive_failures: usize = 0;
    let mut turn_budget_advised = false;

    // Cap the number of model turns. Each successful stream_turn counts as one.
    while turns < config.max_turns {
        let tools = env.tool_definitions();
        let req = ModelRequest {
            model: config.model.clone(),
            system: config.system.clone(),
            messages: messages.clone(),
            tools,
            tool_choice: config.tool_choice,
            max_tokens: config.max_tokens,
        };

        let turn = stream_turn_with_retry(model, &req, on_event)?;
        turns += 1;
        total_usage = sum_usage(total_usage, turn.usage);
        on_event(LoopEvent::TurnComplete {
            stop_reason: turn.stop_reason.clone(),
            usage: turn.usage,
        });

        messages.push(turn.message.clone());
        final_text = extract_text(&turn.message);

        match turn.stop_reason {
            StopReason::ToolUse => {
                let tool_calls = collect_tool_calls(&turn.message);
                if tool_calls.is_empty() {
                    // Model claimed tool_use but produced no calls — treat as end.
                    last_stop = LoopStop::EndTurn;
                    break;
                }

                let mut result_parts: Vec<ContentPart> = Vec::with_capacity(tool_calls.len());
                let mut stuck = false;
                for (id, name, arguments) in tool_calls {
                    on_event(LoopEvent::ToolStart {
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                    let mut outcome = dispatch_tool(env, &name, &arguments);

                    if outcome.is_error {
                        consecutive_failures = consecutive_failures.saturating_add(1);
                        if config.consecutive_failure_advisory > 0
                            && consecutive_failures == config.consecutive_failure_advisory
                        {
                            append_tool_advisory(
                                &mut outcome.content,
                                &format!(
                                    "{} tool calls in a row failed. Change approach: if something the task \
needs is missing from this environment, stop and report it instead of \
retrying.",
                                    config.consecutive_failure_advisory
                                ),
                            );
                        }
                    } else {
                        consecutive_failures = 0;
                    }

                    // Turn-budget awareness: once, when ≤25% of max_turns remain
                    // and at least one turn was already used.
                    if !turn_budget_advised
                        && turns > 0
                        && config.max_turns > 0
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

                    on_event(LoopEvent::ToolFinish {
                        call_id: id.clone(),
                        name: name.clone(),
                        outcome: outcome.clone(),
                    });
                    result_parts.push(ContentPart::ToolResult {
                        call_id: id,
                        content: outcome.content,
                        is_error: outcome.is_error,
                    });

                    if config.consecutive_failure_stop > 0
                        && consecutive_failures >= config.consecutive_failure_stop
                    {
                        stuck = true;
                        break;
                    }
                }

                // One user message carrying every tool_result block, in order.
                messages.push(Message {
                    role: Role::User,
                    content: result_parts,
                });

                if stuck {
                    last_stop = LoopStop::Stuck;
                    break;
                }

                // Continue the outer loop for the next assistant turn.
                // If this was the last allowed turn, the while-guard stops us
                // with MaxTurns *after* tools ran (matching the "model always
                // requests tools → stop after configured count" acceptance).
                last_stop = LoopStop::MaxTurns; // provisional; overwritten on next end
                continue;
            }
            StopReason::EndTurn => {
                last_stop = LoopStop::EndTurn;
                break;
            }
            StopReason::MaxTokens => {
                last_stop = LoopStop::MaxTokens;
                break;
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
    // Stuck already set last_stop and must not be overwritten.
    if turns == 0 {
        last_stop = LoopStop::MaxTurns;
    }

    Ok(LoopResult {
        messages,
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
) -> Result<TurnResult, LoopError> {
    match call_model(model, req, on_event) {
        Ok(t) => Ok(t),
        Err(first) if is_retryable(&first) => {
            // One immediate retry at the loop boundary.
            // Deviation from pi: pi's session-level auto-retry is configurable
            // and delayed; we do a single immediate retry.
            std::thread::sleep(std::time::Duration::from_secs(2));
            call_model(model, req, on_event).map_err(LoopError::from)
        }
        Err(other) => Err(LoopError::from(other)),
    }
}

fn is_retryable(err: &ClientError) -> bool {
    match err {
        // 4xx (other than 429) means the request itself is wrong; everything
        // else — transport failures, 429/5xx, mid-stream error events,
        // truncated streams — is worth one retry.
        ClientError::Http { status, .. } => *status == 429 || *status >= 500,
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
    }

    impl FakeModel {
        fn new(turns: Vec<ScriptedTurn>) -> Self {
            Self {
                turns: turns.into(),
                calls: 0,
            }
        }
    }

    impl ModelStream for FakeModel {
        fn stream_turn(
            &mut self,
            _req: &ModelRequest,
            on_event: &mut dyn FnMut(StreamEvent),
        ) -> Result<TurnResult, ClientError> {
            self.calls += 1;
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
        model: &mut FakeModel,
        env: &mut FakeEnv,
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
        // Exactly 3 model turns.
        assert_eq!(model.calls, 3);
        // 3 tool executions.
        assert_eq!(env.calls.len(), 3);
        // messages: user + (assistant + tool_user)*3
        assert_eq!(result.messages.len(), 1 + 3 * 2);
        assert_eq!(result.usage.input_tokens, 3);
        assert_eq!(result.usage.output_tokens, 3);
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
    fn max_tokens_stop() {
        let mut model = FakeModel::new(vec![ScriptedTurn {
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
        }]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");
        let (result, _) = run(&mut model, &mut env, &config, "hi").expect("ok");
        assert_eq!(result.stop, LoopStop::MaxTokens);
        assert_eq!(result.final_text, "cut off");
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
    fn consecutive_failure_advisory_appears_exactly_once_at_threshold() {
        // 4 consecutive fails → advisory on the 4th; then end.
        // Also: after a success, a new streak of 4 gets the advisory again.
        struct SeqEnv {
            tools: Vec<ToolDefinition>,
            outcomes: VecDeque<ToolOutcome>,
        }
        impl ExecutionEnv for SeqEnv {
            fn tool_definitions(&self) -> Vec<ToolDefinition> {
                self.tools.clone()
            }
            fn call_tool(&mut self, name: &str, _arguments: &serde_json::Value) -> ToolOutcome {
                self.outcomes
                    .pop_front()
                    .unwrap_or_else(|| ToolOutcome::unknown_tool(name))
            }
        }

        // Phase 1: 4 fails → advisory once; model ends.
        {
            let mut turns = failing_tool_turns(4);
            turns.push(text_turn("stopped trying", usage(1, 1)));
            let mut outcomes = VecDeque::new();
            for _ in 0..4 {
                outcomes.push_back(ToolOutcome::err("fail"));
            }
            let mut model = FakeModel::new(turns);
            let mut env = SeqEnv {
                tools: vec![bash_tool()],
                outcomes,
            };
            let config = AgentConfig::default()
                .with_model("mock")
                .with_consecutive_failure_advisory(4)
                .with_consecutive_failure_stop(8);
            let result =
                run_agent_loop(&mut model, &mut env, &config, "x", &mut |_| {}).expect("loop");
            let contents = tool_result_contents(&result);
            let advisory_hits: Vec<_> = contents
                .iter()
                .filter(|(_, c)| c.contains("4 tool calls in a row failed"))
                .collect();
            assert_eq!(
                advisory_hits.len(),
                1,
                "advisory exactly once: {contents:?}"
            );
            assert!(
                contents[3].1.contains("4 tool calls in a row failed"),
                "advisory on 4th: {contents:?}"
            );
            // First three must not have it.
            for (i, (_, c)) in contents.iter().take(3).enumerate() {
                assert!(
                    !c.contains("4 tool calls in a row failed"),
                    "advisory early at {i}: {c}"
                );
            }
        }

        // Phase 2: 4 fails (advisory) + success (reset) + 4 fails (advisory again).
        {
            let mut turns = failing_tool_turns(4);
            turns.push(tool_turn(
                None,
                vec![("ok1", "bash", json!({}))],
                usage(1, 1),
            ));
            turns.extend(failing_tool_turns(4));
            turns.push(text_turn("done", usage(1, 1)));

            let mut outcomes = VecDeque::new();
            for _ in 0..4 {
                outcomes.push_back(ToolOutcome::err("fail"));
            }
            outcomes.push_back(ToolOutcome::ok("ok"));
            for _ in 0..4 {
                outcomes.push_back(ToolOutcome::err("fail"));
            }

            let mut model = FakeModel::new(turns);
            let mut env = SeqEnv {
                tools: vec![bash_tool()],
                outcomes,
            };
            let config = AgentConfig::default()
                .with_model("mock")
                .with_max_turns(20)
                .with_consecutive_failure_advisory(4)
                .with_consecutive_failure_stop(8);
            let result =
                run_agent_loop(&mut model, &mut env, &config, "x", &mut |_| {}).expect("loop");
            let contents = tool_result_contents(&result);
            let advisory_hits: Vec<_> = contents
                .iter()
                .filter(|(_, c)| c.contains("4 tool calls in a row failed"))
                .collect();
            assert_eq!(
                advisory_hits.len(),
                2,
                "advisory once per streak: {contents:?}"
            );
        }
    }

    #[test]
    fn consecutive_failure_stuck_at_stop_threshold() {
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
        assert_eq!(result.stop, LoopStop::Stuck);
        assert_eq!(env.calls, 8, "must stop after 8th failure");
        // 8 tool results present in history.
        let contents = tool_result_contents(&result);
        assert_eq!(contents.len(), 8);
        // Advisory at 4th is still present.
        assert!(
            contents[3].1.contains("4 tool calls in a row failed"),
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
}
