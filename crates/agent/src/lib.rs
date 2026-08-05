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
//! - [`greppy_env`] — production [`GreppyEnv`]: `greppy` + `bash` tools over self-invocation.
//! - [`workspace`] — per-run git worktree isolation and review-patch proposals.
//! - [`agent_loop`] — ported multi-turn agent loop (pi v0.80.2 semantics, MIT).
//!
//! Dependencies are deliberately minimal (`serde` / `serde_json` / plain-HTTP
//! `ureq`). No async runtime.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod agent_loop;
pub mod client;
pub mod env;
pub mod greppy_env;
pub mod model;
pub mod protocol;
pub mod wire;
pub mod workspace;

pub use agent_loop::{run_agent_loop, AgentConfig, LoopError, LoopEvent, LoopResult, LoopStop};
pub use client::{Client, ClientError, ProbeError, TurnResult};
pub use env::{ExecutionEnv, ToolOutcome};
pub use greppy_env::GreppyEnv;
pub use model::ModelStream;
pub use protocol::{
    ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, ToolChoice, ToolDefinition,
    Usage,
};
pub use wire::{to_messages_request_body, SseParser};
pub use workspace::{AgentWorkspace, RunOutcome, WorkspaceError};
