//! Navigation verbs: goto, back, forward, reload, open.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use serde_json::json;

#[derive(Debug, Subcommand)]
pub enum NavCommand {
    /// Navigate the current tab to a URL.
    Goto {
        url: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Ensure a session+tab, then navigate (same engine path as goto).
    Open {
        url: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// History back.
    Back {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// History forward.
    Forward {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Reload the current document.
    Reload {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch(command: NavCommand, root: Option<&str>) -> Result<i32> {
    match command {
        NavCommand::Goto { url, session, json } | NavCommand::Open { url, session, json } => {
            let Some(session) = session else {
                return emit_error(json, invalid("web goto requires --session SESSION"));
            };
            if url.trim().is_empty() {
                return emit_error(json, invalid("web goto requires a URL"));
            }
            rpc(
                root,
                json,
                "web.goto",
                json!({ "session_id": session, "url": url }),
                Some(session),
            )
        }
        NavCommand::Back { session, json } => {
            let Some(session) = session else {
                return emit_error(json, invalid("web back requires --session SESSION"));
            };
            rpc(root, json, "web.back", json!({ "session_id": session }), Some(session))
        }
        NavCommand::Forward { session, json } => {
            let Some(session) = session else {
                return emit_error(json, invalid("web forward requires --session SESSION"));
            };
            rpc(
                root,
                json,
                "web.forward",
                json!({ "session_id": session }),
                Some(session),
            )
        }
        NavCommand::Reload { session, json } => {
            let Some(session) = session else {
                return emit_error(json, invalid("web reload requires --session SESSION"));
            };
            rpc(
                root,
                json,
                "web.reload",
                json!({ "session_id": session }),
                Some(session),
            )
        }
    }
}
