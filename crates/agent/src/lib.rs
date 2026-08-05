//! `greppy-agent` — provider-neutral coding-agent wire layer.
//!
//! This crate hosts the 0.4.0 agent *transport* only:
//! - [`protocol`] — model-agnostic request/response and stream event types.
//! - [`wire`] — Anthropic Messages API request-body adapter and incremental
//!   SSE parser (`/v1/messages`, `stream: true`).
//! - [`client`] — blocking HTTP client aimed at a localhost gateway
//!   (CLIProxyAPI-style, default `http://127.0.0.1:8317`).
//!
//! What this crate is **not** (later work packages):
//! - No agent loop, no tool execution, no CLI wiring, no multi-provider
//!   fan-out. Those sit on top of this wire surface.
//!
//! Dependencies are deliberately minimal (`serde` / `serde_json` / plain-HTTP
//! `ureq`). No async runtime.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod client;
pub mod protocol;
pub mod wire;

pub use client::{Client, ClientError, ProbeError, TurnResult};
pub use protocol::{
    ContentPart, Message, ModelRequest, Role, StopReason, StreamEvent, ToolChoice, ToolDefinition,
    Usage,
};
pub use wire::{to_messages_request_body, SseParser};
