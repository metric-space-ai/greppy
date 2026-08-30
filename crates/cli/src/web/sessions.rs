//! Session-scope web verbs: status, doctor, session, runtime.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use serde_json::json;

#[derive(Debug, Subcommand)]
pub enum SessionsCommand {
    /// Report web-runtime availability. Does not link engines.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Check supervisor and worker images without constructing engines.
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// Create, list, or close a run-owned web session.
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
    /// Long-lived runtime owner: status, stop, restart.
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum SessionCommand {
    Create {
        #[arg(long, default_value = "research")]
        profile: String,
        #[arg(long)]
        json: bool,
    },
    List {
        #[arg(long)]
        json: bool,
    },
    Close {
        session: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum RuntimeCommand {
    /// Report whether the owner is running. Does not spawn.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Shut down the owner if this process holds the attach token.
    Stop {
        #[arg(long)]
        json: bool,
    },
    /// Restart the owner and leave it running.
    Restart {
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch(command: SessionsCommand, root: Option<&str>) -> Result<i32> {
    match command {
        SessionsCommand::Status { json } => status(json, root),
        SessionsCommand::Doctor { json } => doctor(json, root),
        SessionsCommand::Session { command } => match command {
            SessionCommand::Create { profile, json } => rpc(
                root,
                json,
                "web.session.create",
                json!({ "profile": profile }),
                None,
            ),
            SessionCommand::List { json } => rpc(root, json, "web.session.list", json!({}), None),
            SessionCommand::Close { session, json } => rpc(
                root,
                json,
                "web.session.close",
                json!({ "session_id": session }),
                Some(session),
            ),
        },
        SessionsCommand::Runtime { command } => match command {
            RuntimeCommand::Status { json } => runtime_status(json, root),
            RuntimeCommand::Stop { json } => runtime_stop(json, root),
            RuntimeCommand::Restart { json } => runtime_restart(json, root),
        },
    }
}
