//! Diagnostic verbs. All three read what the page already recorded — console
//! messages and network requests — so they share one runtime handler and
//! differ only in what they ask for.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use serde_json::json;

#[derive(Debug, Subcommand)]
pub enum DiagnoseCommand {
    /// Console output the page produced.
    ///
    ///   greppy web console
    ///   greppy web console --errors
    Console {
        /// Only entries of type `error`.
        #[arg(long)]
        errors: bool,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Network requests the page issued.
    ///
    ///   greppy web network
    ///   greppy web network --failed
    Network {
        /// Only requests that did not complete successfully.
        #[arg(long)]
        failed: bool,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Console and network together, for one look at what the page did.
    Events {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch(command: DiagnoseCommand, root: Option<&str>) -> Result<i32> {
    match command {
        DiagnoseCommand::Console {
            errors,
            session,
            json,
        } => records(root, json, session, "web.console", errors.then_some("error")),
        DiagnoseCommand::Network {
            failed,
            session,
            json,
        } => records(root, json, session, "web.network", failed.then_some("failed")),
        DiagnoseCommand::Events { session, json } => {
            records(root, json, session, "web.events", None)
        }
    }
}

/// One call for all three verbs. `filter` is passed through so the runtime
/// can narrow the list rather than the caller receiving everything and
/// discarding most of it.
fn records(
    root: Option<&str>,
    json_out: bool,
    session: Option<String>,
    operation: &str,
    filter: Option<&str>,
) -> Result<i32> {
    let session = match resolve_session(root, session) {
        Ok(session) => session,
        Err(error) => return emit_error(json_out, error),
    };
    let mut payload = json!({ "session_id": session });
    if let (Some(filter), Some(object)) = (filter, payload.as_object_mut()) {
        object.insert("filter".into(), json!(filter));
    }
    rpc(root, json_out, operation, payload, Some(session))
}
