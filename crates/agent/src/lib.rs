//! `greppy-agent` — provider-neutral coding-agent wire layer and agent loop.
//!
//! This crate hosts the 0.4.0 agent transport and loop:
//! - [`protocol`] — model-agnostic request/response and stream event types.
//! - [`wire`] — Anthropic Messages API request-body adapter and incremental
//!   SSE parser (`/v1/messages`, `stream: true`).
//! - [`client`] — blocking HTTP client aimed at a localhost gateway
//!   (CLIProxyAPI-style, default `http://127.0.0.1:8317`).
//! - [`model`] — [`ModelStream`] trait so the loop is testable without network.
//! - [`env`] — [`ExecutionEnv`] tool boundary (agent proposes, host executes).
//! - [`greppy_env`] — production [`GreppyEnv`]: single `greppy` tool over self-invocation.
//! - [`sandbox`] — write-confinement for tool subprocesses (Seatbelt / Landlock).
//! - [`workspace`] — per-repository agent worktree isolation and review-patch proposals.
//! - [`agent_loop`] — ported multi-turn agent loop (pi v0.80.2 semantics, MIT).
//!
//! Dependencies are deliberately minimal (`serde` / `serde_json` / plain-HTTP
//! `ureq`). No async runtime.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

/// Env marker set in every tool subprocess of a running agent. `greppy -p`
/// refuses to start while it is present, so agents cannot nest — not through
/// a greppy invocation and not through `bash-smart`.
pub const AGENT_RUN_ENV: &str = "GREPPY_AGENT_RUN";

pub mod agent_loop;
pub mod client;
pub mod env;
pub mod greppy_env;
pub mod local_model;
pub mod model;
#[cfg(feature = "local-model-download")]
pub mod model_download;
pub mod prompt;
pub mod protocol;
pub mod sandbox;
pub mod wire;
#[path = "portable_workspace.rs"]
pub mod workspace;

pub use agent_loop::{
    run_agent_loop, run_agent_loop_with_history, AgentConfig, LoopError, LoopEvent, LoopResult,
    LoopStop,
};
pub use client::{Client, ClientError, ProbeError, TurnResult};
pub use env::{ExecutionEnv, ToolOutcome};
pub use greppy_env::{
    parse_where_am_i_file_count, run_startup_self_check, GreppyEnv, SelfCheckError, SelfCheckOk,
};
pub use model::ModelStream;
pub use prompt::{browser_prompt, system_prompt, SYSTEM_PROMPT};
pub use protocol::{
    ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, ToolChoice, ToolDefinition,
    Usage,
};
pub use sandbox::{SandboxError, SandboxMode, SandboxSpec};
pub use wire::{to_messages_request_body, SseItem, SseParser};
pub use workspace::{apply_proposal, AgentWorkspace, RunOutcome, WorkspaceError};
