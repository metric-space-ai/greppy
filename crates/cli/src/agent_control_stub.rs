//! Windows placeholder for the Unix-socket control transport.
//!
//! Remote control rides on Unix domain sockets (`agent_control.rs`). Every
//! caller — the readers, the CLI clients and the TUI — is otherwise
//! platform-independent, so Windows builds get the same API with a transport
//! that reports "unsupported" instead of connecting. `is_live` is always
//! `false`, so sessions list as not live and the clients exit with their
//! regular "not live" message.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};

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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for RpcError {}

const UNSUPPORTED: &str = "agent remote control needs Unix domain sockets";

fn unsupported() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, UNSUPPORTED)
}

pub struct ControlServer {
    path: PathBuf,
}

impl ControlServer {
    pub fn bind(_path: &Path) -> io::Result<Self> {
        Err(unsupported())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn poll(&mut self) -> Vec<Incoming> {
        Vec::new()
    }

    pub fn reply(&mut self, _conn: ConnId, _id: Value, _result: Result<Value, RpcError>) {}

    pub fn broadcast(&mut self, _event: &Value) {}
}

pub struct ControlClient;

impl ControlClient {
    pub fn connect(_path: &Path) -> io::Result<Self> {
        Err(unsupported())
    }

    pub fn call(&mut self, _method: &str, _params: Value) -> Result<Value, RpcError> {
        Err(RpcError::new(-32000, UNSUPPORTED))
    }

    pub fn subscribe(&mut self) -> Result<(), RpcError> {
        Err(RpcError::new(-32000, UNSUPPORTED))
    }

    pub fn next_event(&mut self, _timeout: Duration) -> io::Result<Option<Value>> {
        Err(unsupported())
    }
}

/// Where the socket *would* live: the same hashed shape the Unix transport
/// uses, so `sessions list` reports a stable path. Nothing binds it here.
pub fn socket_path_for(store: &SessionStore, session_id: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(store.project().as_bytes());
    hasher.update([0]);
    hasher.update(session_id.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    std::env::temp_dir().join(format!("greppy-agent-control-{}.sock", &digest[..16]))
}

pub fn is_live(_path: &Path) -> bool {
    false
}

pub fn not_live_message(session_id: &str) -> String {
    format!(
        "session {session_id} is not live (no control socket); start it with greppy agent serve --resume {session_id}"
    )
}
