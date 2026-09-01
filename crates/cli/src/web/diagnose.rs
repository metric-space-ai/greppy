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
    /// Record a Playwright trace.
    ///
    /// On a separate release track: `Tracing.start` and `Tracing.stop` are
    /// `unsupported` in `contracts/web-runtime/compatibility.v1.json`, and the
    /// contract requires such calls to fail explicitly rather than pretend.
    /// Use `greppy web events`, `console` and `network` for what the page did,
    /// and `screenshot` for what it looked like.
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
        json: bool,
    },
    /// Stop recording and write the archive.
    Stop {
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
            ),
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
        DiagnoseCommand::Trace { command } => {
            let json_out = match command {
                TraceCommand::Start { json } => json,
                TraceCommand::Stop { json, .. } => json,
            };
            separate_track(
                json_out,
                "web trace",
                "Tracing.start and Tracing.stop",
                "use greppy web events, console, network and screenshot",
            )
        }
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
            failed,
            session,
            json,
        } => records(
            root,
            json,
            session,
            "web.network",
            failed.then_some("failed"),
        ),
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
