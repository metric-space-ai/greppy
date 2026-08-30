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
        tab: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Ensure a session+tab, then navigate (same engine path as goto).
    Open {
        url: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        tab: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// History back.
    Back {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        tab: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// History forward.
    Forward {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        tab: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Reload the current document.
    Reload {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        tab: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch(command: NavCommand, root: Option<&str>) -> Result<i32> {
    match command {
        NavCommand::Goto {
            url,
            session,
            tab,
            json,
        } => goto_url(root, url, session, tab, json, false),
        NavCommand::Open {
            url,
            session,
            tab,
            json,
        } => goto_url(root, url, session, tab, json, true),
        NavCommand::Back {
            session,
            tab,
            json,
        } => history(root, session, tab, json, "web.back"),
        NavCommand::Forward {
            session,
            tab,
            json,
        } => history(root, session, tab, json, "web.forward"),
        NavCommand::Reload {
            session,
            tab,
            json,
        } => history(root, session, tab, json, "web.reload"),
    }
}

fn goto_url(
    root: Option<&str>,
    url: String,
    session: Option<String>,
    tab: Option<String>,
    json: bool,
    create: bool,
) -> Result<i32> {
    if url.trim().is_empty() {
        return emit_error(json, invalid("web goto requires a URL"));
    }
    let session = match resolve_or_create_session(root, session, json, create) {
        Ok(session) => session,
        Err(code) => return Ok(code),
    };
    let tab = resolve_tab(root, tab);
    let mut payload = json!({ "session_id": session, "url": url });
    if let Some(tab) = &tab {
        payload["tab_id"] = json!(tab);
    }
    let code = rpc(root, json, "web.goto", payload, Some(session.clone()))?;
    if code == 0 {
        let _ = write_current_scope(root, &session, tab.as_deref());
    }
    Ok(code)
}

fn history(
    root: Option<&str>,
    session: Option<String>,
    tab: Option<String>,
    json: bool,
    operation: &str,
) -> Result<i32> {
    let session = match resolve_session(root, session) {
        Ok(session) => session,
        Err(error) => return emit_error(json, error),
    };
    let tab = resolve_tab(root, tab);
    let mut payload = json!({ "session_id": session });
    if let Some(tab) = tab {
        payload["tab_id"] = json!(tab);
    }
    rpc(root, json, operation, payload, Some(session))
}

fn resolve_or_create_session(
    root: Option<&str>,
    session: Option<String>,
    json: bool,
    create: bool,
) -> std::result::Result<String, i32> {
    match resolve_session(root, session) {
        Ok(session) => Ok(session),
        Err(_) if create => create_session(root, json),
        Err(error) => {
            let code = emit_error(json, error).unwrap_or(EXIT_WEB_INVALID);
            Err(code)
        }
    }
}

fn create_session(root: Option<&str>, json: bool) -> std::result::Result<String, i32> {
    match rpc_response(
        root,
        "web.session.create",
        json!({ "profile": "project" }),
        None,
    ) {
        Err(error) => {
            let code = emit_error(json, error).unwrap_or(EXIT_WEB_UNAVAILABLE);
            Err(code)
        }
        Ok(response) if response.status != "ok" => {
            let code = emit_response(json, response).unwrap_or(EXIT_WEB_SESSION);
            Err(code)
        }
        Ok(response) => {
            let Some(session) = response
                .result
                .as_ref()
                .and_then(|value| value.get("session_id"))
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
            else {
                let code = emit_error(json, invalid("web.session.create returned no session_id"))
                    .unwrap_or(EXIT_WEB_SESSION);
                return Err(code);
            };
            if let Err(error) = write_current_scope(root, &session, None) {
                let code = emit_error(json, error).unwrap_or(EXIT_WEB_INVALID);
                return Err(code);
            }
            Ok(session)
        }
    }
}
