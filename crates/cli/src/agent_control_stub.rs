//! Non-Unix boundary for the optional live-session control transport.
//!
//! Windows 0.4.0 keeps the local interactive and one-shot agents, but does not
//! claim a Unix-domain-socket control endpoint. The CLI control subcommands
//! already report that scope explicitly. These inert server types let the
//! platform-neutral TUI retain one implementation without inventing a second
//! transport contract during the release freeze.

use std::io;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::agent_tui::SessionStore;

pub type ConnId = u64;

#[derive(Debug)]
pub enum Incoming {
    Connected {
        conn: ConnId,
    },
    Request {
        conn: ConnId,
        id: Value,
        method: String,
        params: Value,
    },
    Disconnected {
        conn: ConnId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for RpcError {}

pub struct ControlServer;

impl ControlServer {
    pub fn bind(_path: &Path) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "live agent control is not available on Windows in 0.4.0",
        ))
    }

    pub fn poll(&mut self) -> Vec<Incoming> {
        Vec::new()
    }

    pub fn reply(&mut self, _conn: ConnId, _id: Value, _result: Result<Value, RpcError>) {}

    pub fn broadcast(&mut self, _event: &Value) {}
}

pub fn socket_path_for(_store: &SessionStore, _session_id: &str) -> PathBuf {
    PathBuf::new()
}
