//! Diagnostic verbs. All three read what the page already recorded — console
//! messages and network requests — so they share one runtime handler and
//! differ only in what they ask for.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use greppy_web_client::ErrorObject;
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
    ///   greppy web network 'status>=400'
    ///   greppy web network --failed
    Network {
        /// Record query, using the same predicate grammar as `web match`.
        query: Option<String>,
        /// Only requests that did not complete successfully.
        #[arg(long)]
        failed: bool,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Record a Playwright trace.
    ///
    /// The resulting action archive uses Playwright trace schema v8.
    /// Snapshot and screenshot capture are not yet supported.
    Trace {
        #[command(subcommand)]
        command: TraceCommand,
    },
    /// Expose a native Playwright endpoint for an external program.
    ///
    /// On a separate release track: `BrowserType.connect`, `launchServer` and
    /// `connectOverCDP` are `unsupported` in the compatibility contract. A
    /// native endpoint is also browser-wide privileged — it would hand a
    /// client every context, not just the current tab — so it needs an
    /// exclusive lease before it can exist at all.
    Endpoint {
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

#[derive(Debug, Subcommand)]
pub enum TraceCommand {
    /// Begin recording.
    Start {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Stop recording and write the archive.
    Stop {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        to: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

/// Refuse a command that the compatibility contract lists as `unsupported`.
///
/// The contract's own vocabulary requires these to fail explicitly: a silent
/// no-op or a half-working stand-in would be worse than the refusal, because
/// a caller could not tell the difference from a working implementation.
fn separate_track(json_out: bool, what: &str, symbols: &str, instead: &str) -> Result<i32> {
    emit_error(
        json_out,
        ErrorObject {
            code: "unsupported_operation".into(),
            message: format!(
                "{what} is on a separate release track: {symbols} are `unsupported` \
                 in the compatibility contract"
            )
            .into_boxed_str(),
            operation_id: String::new(),
            session_id: None,
            retryable: false,
            next_action: instead.to_owned(),
            exit_code: 31,
        },
    )
}

pub(super) fn dispatch(command: DiagnoseCommand, root: Option<&str>) -> Result<i32> {
    match command {
        DiagnoseCommand::Trace { command } => match command {
            TraceCommand::Start { session, json } => trace_start(root, session, json),
            TraceCommand::Stop { session, to, json } => trace_stop(root, session, to, json),
        },
        DiagnoseCommand::Endpoint { json } => separate_track(
            json,
            "web endpoint",
            "BrowserType.connect, launchServer and connectOverCDP",
            "use greppy web pw for Playwright code inside this runtime",
        ),
        DiagnoseCommand::Console {
            errors,
            session,
            json,
        } => records(
            root,
            json,
            session,
            "web.console",
            errors.then_some("error"),
        ),
        DiagnoseCommand::Network {
            query,
            failed,
            session,
            json,
        } => network_records(root, json, session, query, failed),
        DiagnoseCommand::Events { session, json } => {
            records(root, json, session, "web.events", None)
        }
    }
}

fn trace_start(root: Option<&str>, session: Option<String>, json_out: bool) -> Result<i32> {
    let session = match resolve_session(root, session) {
        Ok(value) => value,
        Err(error) => return emit_error(json_out, error),
    };
    rpc(
        root,
        json_out,
        "web.trace.start",
        json!({"session_id":session}),
        Some(session),
    )
}

fn trace_stop(
    root: Option<&str>,
    session: Option<String>,
    to: Option<String>,
    json_out: bool,
) -> Result<i32> {
    if json_out && to.is_some() {
        return emit_error(
            true,
            invalid("web trace stop --to cannot be combined with --json"),
        );
    }
    let session = match resolve_session(root, session) {
        Ok(value) => value,
        Err(error) => return emit_error(json_out, error),
    };
    match rpc_response(
        root,
        "web.trace.stop",
        json!({"session_id":session}),
        Some(session.clone()),
    ) {
        Err(error) => emit_error(json_out, error),
        Ok(response) if response.status != "ok" => emit_response(json_out, response),
        Ok(response) => {
            if let Some(to) = to {
                let id = response
                    .result
                    .as_ref()
                    .and_then(|v| v.get("artifact"))
                    .and_then(|v| v.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                return super::results::artifact_export(root, Some(session), id, Some(to), false);
            }
            emit_response(json_out, response)
        }
    }
}

fn network_records(
    root: Option<&str>,
    json_out: bool,
    session: Option<String>,
    query: Option<String>,
    failed: bool,
) -> Result<i32> {
    let session = match resolve_session(root, session) {
        Ok(session) => session,
        Err(error) => return emit_error(json_out, error),
    };
    let mut payload = json!({ "session_id": session });
    let object = payload
        .as_object_mut()
        .expect("network payload is an object");
    if let Some(query) = query {
        object.insert("query".into(), json!(query));
    }
    if failed {
        object.insert("filter".into(), json!("failed"));
    }
    rpc(root, json_out, "web.network", payload, Some(session))
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
