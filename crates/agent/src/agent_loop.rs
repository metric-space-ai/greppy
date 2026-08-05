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
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 40,
            system: None,
            model: String::new(),
            max_tokens: 8192,
            tool_choice: ToolChoice::Auto,
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
                for (id, name, arguments) in tool_calls {
                    on_event(LoopEvent::ToolStart {
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                    let outcome = dispatch_tool(env, &name, &arguments);
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
                }

                // One user message carrying every tool_result block, in order.
                messages.push(Message {
                    role: Role::User,
                    content: result_parts,
                });

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

/// Stream a turn; on a pure transport error, retry exactly once.
fn stream_turn_with_retry(
    model: &mut dyn ModelStream,
    req: &ModelRequest,
    on_event: &mut dyn FnMut(LoopEvent),
) -> Result<TurnResult, LoopError> {
    match call_model(model, req, on_event) {
        Ok(t) => Ok(t),
        Err(ClientError::Transport(_first)) => {
            // One immediate retry on transport/connect failures.
            // Deviation from pi: pi's session-level auto-retry is configurable
            // and delayed; we do a single immediate retry at the loop boundary.
            call_model(model, req, on_event).map_err(LoopError::from)
        }
        Err(other) => Err(LoopError::from(other)),
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
    fn http_error_does_not_retry() {
        let mut model = FakeModel::new(vec![ScriptedTurn {
            events: vec![],
            result: Err(ClientError::Http {
                status: 500,
                body: "nope".into(),
            }),
        }]);
        let mut env = FakeEnv::new(vec![]);
        let config = AgentConfig::default().with_model("mock");

        let err = run(&mut model, &mut env, &config, "hi").expect_err("http fatal");
        assert!(matches!(err, LoopError::Http { status: 500, .. }));
        assert_eq!(model.calls, 1);
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
}
