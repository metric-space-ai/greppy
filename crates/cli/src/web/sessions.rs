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
        #[arg(long, default_value = "research")]
        profile: String,
        /// Explicit goal for task-conditioned observation ranking.
        #[arg(long, value_parser = parse_goal_arg)]
        goal: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Explicitly set or clear the session goal using a version precondition.
    SetGoal {
        #[arg(long, value_parser = parse_goal_arg, conflicts_with = "clear", required_unless_present = "clear")]
        goal: Option<String>,
        #[arg(long)]
        clear: bool,
        #[arg(long)]
        expected_goal_version: u64,
        #[arg(long)]
        session: Option<String>,
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

fn parse_goal_arg(value: &str) -> std::result::Result<String, String> {
    greppy_web_client::observation_context::validate_goal(value).map_err(|_| {
        format!(
            "goal must be nonempty and at most {} UTF-8 bytes",
            greppy_web_client::observation_context::MAX_GOAL_BYTES
        )
    })?;
    Ok(value.to_owned())
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
            SessionCommand::Create {
                profile,
                goal,
                json,
            } => {
                let mut params = json!({ "profile": profile });
                if let Some(goal) = goal {
                    params["goal"] = json!(goal);
                }
                match rpc_response(root, "web.session.create", params, None) {
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
            SessionCommand::SetGoal {
                goal,
                clear: _,
                expected_goal_version,
                session,
                json,
            } => {
                let session = match resolve_session(root, session) {
                    Ok(session) => session,
                    Err(error) => return emit_error(json, error),
                };
                let request = greppy_web_client::SetGoalRequest {
                    session_id: session.clone(),
                    goal,
                    expected_goal_version,
                };
                rpc(
                    root,
                    json,
                    "web.session.set_goal",
                    json!(request),
                    Some(session),
                )
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

#[cfg(test)]
mod goal_argument_tests {
    use super::*;
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct Args {
        #[command(subcommand)]
        command: SessionCommand,
    }

    #[test]
    fn create_accepts_optional_explicit_goal() {
        let args =
            Args::try_parse_from(["session", "create", "--goal", "Choose a product"]).unwrap();
        assert!(
            matches!(args.command,SessionCommand::Create { goal:Some(goal),.. } if goal=="Choose a product")
        );
        let args = Args::try_parse_from(["session", "create"]).unwrap();
        assert!(matches!(
            args.command,
            SessionCommand::Create { goal: None, .. }
        ));
        assert!(Args::try_parse_from(["session", "create", "--goal", " "]).is_err());
    }

    #[test]
    fn update_requires_version_and_exactly_one_set_or_clear() {
        assert!(Args::try_parse_from(["session", "set-goal", "--goal", "Save"]).is_err());
        assert!(
            Args::try_parse_from(["session", "set-goal", "--expected-goal-version", "1"]).is_err()
        );
        assert!(Args::try_parse_from([
            "session",
            "set-goal",
            "--goal",
            "Save",
            "--clear",
            "--expected-goal-version",
            "1"
        ])
        .is_err());
        let args = Args::try_parse_from([
            "session",
            "set-goal",
            "--clear",
            "--expected-goal-version",
            "2",
            "--session",
            "wrs_1",
        ])
        .unwrap();
        assert!(matches!(
            args.command,
            SessionCommand::SetGoal {
                goal: None,
                clear: true,
                expected_goal_version: 2,
                ..
            }
        ));
        let args = Args::try_parse_from([
            "session",
            "set-goal",
            "--goal",
            "Save",
            "--expected-goal-version",
            "2",
        ])
        .unwrap();
        assert!(
            matches!(args.command,SessionCommand::SetGoal { goal:Some(goal),clear:false,expected_goal_version:2,.. } if goal=="Save")
        );
    }
}
