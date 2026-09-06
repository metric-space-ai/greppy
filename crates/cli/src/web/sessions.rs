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
    /// Manage tabs: pages inside the current session.
    Tab {
        #[command(subcommand)]
        command: TabCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum SessionCommand {
    Create {
        /// Network profile: research: public web; project: public web plus loopback
        /// for explicitly requested local development. LAN and cloud metadata remain blocked.
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
        SessionsCommand::Tab { command } => dispatch_tab(command, root),
        SessionsCommand::Status { json } => status(json, root),
        SessionsCommand::Doctor { json } => doctor(json, root),
        SessionsCommand::Session { command } => match command {
            SessionCommand::Create { profile, json } => {
                match rpc_response(
                    root,
                    "web.session.create",
                    json!({ "profile": profile }),
                    None,
                ) {
                    Err(error) => emit_error(json, error),
                    Ok(response) => {
                        if response.status == "ok" {
                            if let Some(session) = response
                                .result
                                .as_ref()
                                .and_then(|value| value.get("session_id"))
                                .and_then(|value| value.as_str())
                            {
                                let _ = write_current_scope(root, session, None);
                            }
                        }
                        emit_response(json, response)
                    }
                }
            }
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

// ---------------------------------------------------------------------------
// Tabs
//
// A tab is a page inside the current session: cookies and storage stay
// shared, the document does not. `session.page_id` in the runtime names the
// active one.
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Subcommand)]
pub enum TabCommand {
    /// Open a new tab in the current session and make it active.
    New {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// List the tabs of the current session.
    List {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Make a tab active.
    Switch {
        /// Tab id from `tab list`.
        tab: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Close a tab. Without an id, closes the active one.
    Close {
        tab: Option<String>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch_tab(command: TabCommand, root: Option<&str>) -> Result<i32> {
    let (operation, tab, session, json_out) = match command {
        TabCommand::New { session, json } => ("web.tab.new", None, session, json),
        TabCommand::List { session, json } => ("web.tab.list", None, session, json),
        TabCommand::Switch { tab, session, json } => ("web.tab.switch", Some(tab), session, json),
        TabCommand::Close { tab, session, json } => ("web.tab.close", tab, session, json),
    };
    let session = match resolve_session(root, session) {
        Ok(session) => session,
        Err(error) => return emit_error(json_out, error),
    };
    let mut payload = serde_json::json!({ "session_id": session });
    if let (Some(tab), Some(object)) = (tab, payload.as_object_mut()) {
        object.insert("tab".into(), serde_json::json!(tab));
    }
    rpc(root, json_out, operation, payload, Some(session))
}
