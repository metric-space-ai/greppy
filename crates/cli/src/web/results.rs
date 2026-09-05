//! Result and artifact web verbs already shipped.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use greppy_web_client::ErrorObject;
use serde_json::json;
use std::path::Path;

#[derive(Debug, Subcommand)]
pub enum ResultsCommand {
    /// Run an unchanged Playwright script in a session.
    Run {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        script_file: Option<String>,
        #[arg(long)]
        script_stdin: bool,
        #[arg(long)]
        timeout: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Return a compact observation of the current page.
    Observe {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        tab: Option<String>,
        #[arg(long)]
        format: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Capture a screenshot into an artifact path.
    Screenshot {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        output: Option<String>,
        /// Wait for Servo's complete-render readiness before capturing.
        #[arg(long)]
        render_complete: bool,
        #[arg(long)]
        json: bool,
    },
    /// Search the public web through the runtime.
    Search {
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        domain: Option<String>,
        /// Cap search hits. Distinct from the global `--limit` (usize).
        #[arg(long = "result-limit")]
        result_limit: Option<u32>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long = "fixture-url")]
        fixture_url: Option<String>,
        #[arg(long = "search-endpoint")]
        search_endpoint: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Read one URL through the runtime.
    Read {
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long = "fixture-url")]
        fixture_url: Option<String>,
        #[arg(long = "search-endpoint")]
        search_endpoint: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Bounded research over the runtime.
    Research {
        #[arg(long)]
        query: Option<String>,
        #[arg(long = "max-sources")]
        max_sources: Option<u32>,
        #[arg(long)]
        depth: Option<String>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long = "fixture-url")]
        fixture_url: Option<String>,
        #[arg(long = "search-endpoint")]
        search_endpoint: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// List artifacts for a session.
    Artifacts {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Cancel a specific in-flight web.run by request id.
    Cancel {
        #[arg(long)]
        session: String,
        #[arg(long = "target-request-id")]
        target_request_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Heartbeat a busy session so its idle timer stays fresh.
    Heartbeat {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        seq: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Artifacts produced by a session (`artifact list` is the prompt verb).
    Artifact {
        #[command(subcommand)]
        command: ArtifactCommand,
    },
    /// Continue a truncated result.
    Result {
        #[command(subcommand)]
        command: ResultCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum ArtifactCommand {
    /// List artifacts for a session.
    List {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Show one artifact's metadata. ID is the sha256 digest (or a unique prefix).
    Show {
        id: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Print the filesystem path of an artifact object.
    Path {
        id: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Write raw artifact bytes to FILE. Cannot be combined with --json.
    Export {
        id: String,
        #[arg(long = "to")]
        to: Option<String>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum ResultCommand {
    /// Fetch the rest of a truncated result.
    Next {
        cursor: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch(command: ResultsCommand, root: Option<&str>) -> Result<i32> {
    match command {
        ResultsCommand::Run {
            session,
            script_file,
            script_stdin,
            timeout,
            json,
        } => run(root, session, script_file, script_stdin, timeout, json),
        ResultsCommand::Observe {
            session,
            tab,
            format,
            json,
        } => {
            let session = match resolve_session(root, session) {
                Ok(session) => session,
                Err(error) => return emit_error(json, error),
            };
            let tab = resolve_tab(root, tab);
            let mut payload = json!({
                "session_id": session,
                "format": format.unwrap_or_else(|| "agent-tree".into()),
            });
            if let Some(tab) = tab {
                payload["tab_id"] = json!(tab);
            }
            rpc(root, json, "web.observe", payload, Some(session))
        }
        ResultsCommand::Screenshot {
            session,
            output,
            render_complete,
            json,
        } => screenshot(root, session, output, render_complete, json),
        ResultsCommand::Search {
            query,
            domain,
            result_limit,
            session,
            fixture_url,
            search_endpoint,
            json,
        } => {
            let Some(query) = query.filter(|query| !query.is_empty()) else {
                return emit_error(json, invalid("web search requires --query QUERY"));
            };
            let session = match resolve_session(root, session) {
                Ok(session) => session,
                Err(error) => return emit_error(json, error),
            };
            rpc_with_spawn(
                root,
                json,
                "web.search",
                json!({
                    "query": query,
                    "domain": domain,
                    "limit": result_limit,
                    "session_id": session
                }),
                Some(session),
                SupervisorSpawn {
                    fixture_url,
                    search_endpoint,
                },
            )
        }
        ResultsCommand::Read {
            url,
            query,
            session,
            fixture_url,
            search_endpoint,
            json,
        } => {
            let Some(url) = url.filter(|url| !url.is_empty()) else {
                return emit_error(json, invalid("web read requires --url URL"));
            };
            let session = match resolve_session(root, session) {
                Ok(session) => session,
                Err(error) => return emit_error(json, error),
            };
            rpc_with_spawn(
                root,
                json,
                "web.read",
                json!({ "url": url, "query": query, "session_id": session }),
                Some(session),
                SupervisorSpawn {
                    fixture_url,
                    search_endpoint,
                },
            )
        }
        ResultsCommand::Research {
            query,
            max_sources,
            depth,
            session,
            fixture_url,
            search_endpoint,
            json,
        } => {
            let Some(query) = query.filter(|query| !query.is_empty()) else {
                return emit_error(json, invalid("web research requires --query QUERY"));
            };
            let session = match resolve_session(root, session) {
                Ok(session) => session,
                Err(error) => return emit_error(json, error),
            };
            rpc_with_spawn(
                root,
                json,
                "web.research",
                json!({
                    "query": query,
                    "max_sources": max_sources,
                    "depth": depth,
                    "session_id": session
                }),
                Some(session),
                SupervisorSpawn {
                    fixture_url,
                    search_endpoint,
                },
            )
        }
        ResultsCommand::Artifacts { session, json } => {
            let session = match resolve_session(root, session) {
                Ok(session) => session,
                Err(error) => return emit_error(json, error),
            };
            rpc(
                root,
                json,
                "web.artifacts",
                json!({ "session_id": session }),
                Some(session),
            )
        }
        ResultsCommand::Cancel {
            session,
            target_request_id,
            json,
        } => rpc(
            root,
            json,
            "web.cancel",
            json!({
                "session_id": session,
                "target_request_id": target_request_id
            }),
            Some(session),
        ),
        ResultsCommand::Heartbeat { session, seq, json } => {
            let session = match resolve_session(root, session) {
                Ok(session) => session,
                Err(error) => return emit_error(json, error),
            };
            rpc(
                root,
                json,
                "web.heartbeat",
                json!({ "session_id": session, "seq": seq.unwrap_or(1) }),
                Some(session),
            )
        }
        ResultsCommand::Artifact { command } => match command {
            ArtifactCommand::List { session, json } => {
                let session = match resolve_session(root, session) {
                    Ok(session) => session,
                    Err(error) => return emit_error(json, error),
                };
                rpc(
                    root,
                    json,
                    "web.artifacts",
                    json!({ "session_id": session }),
                    Some(session),
                )
            }
            ArtifactCommand::Show { id, session, json } => {
                let session = match resolve_session(root, session) {
                    Ok(session) => session,
                    Err(error) => return emit_error(json, error),
                };
                if id.trim().is_empty() {
                    return emit_error(json, invalid("web artifact show requires an artifact id"));
                }
                rpc(
                    root,
                    json,
                    "web.artifact.show",
                    json!({ "session_id": session, "id": id }),
                    Some(session),
                )
            }
            ArtifactCommand::Path { id, session, json } => {
                let session = match resolve_session(root, session) {
                    Ok(session) => session,
                    Err(error) => return emit_error(json, error),
                };
                if id.trim().is_empty() {
                    return emit_error(json, invalid("web artifact path requires an artifact id"));
                }
                rpc(
                    root,
                    json,
                    "web.artifact.path",
                    json!({ "session_id": session, "id": id }),
                    Some(session),
                )
            }
            ArtifactCommand::Export {
                id,
                to,
                session,
                json,
            } => artifact_export(root, session, id, to, json),
        },
        ResultsCommand::Result { command } => match command {
            ResultCommand::Next {
                cursor,
                session,
                json,
            } => {
                if super::view::is_cursor(&cursor) {
                    return match super::view::resume(
                        &cursor,
                        session.as_deref(),
                        &super::view::cache_dir(),
                        json,
                    ) {
                        Ok(text) => {
                            println!("{text}");
                            Ok(0)
                        }
                        Err(message) => emit_error(json, invalid(&message)),
                    };
                }
                let session = match resolve_session(root, session) {
                    Ok(session) => session,
                    Err(error) => return emit_error(json, error),
                };
                if cursor.trim().is_empty() {
                    return emit_error(json, invalid("web result next requires a cursor"));
                }
                rpc(
                    root,
                    json,
                    "web.result.next",
                    json!({ "session_id": session, "cursor": cursor }),
                    Some(session),
                )
            }
        },
    }
}

fn artifact_export(
    root: Option<&str>,
    session: Option<String>,
    id: String,
    to: Option<String>,
    json: bool,
) -> Result<i32> {
    if json {
        return emit_error(
            true,
            invalid("artifact export writes raw bytes and cannot be combined with --json"),
        );
    }
    let session = match resolve_session(root, session) {
        Ok(session) => session,
        Err(error) => return emit_error(false, error),
    };
    if id.trim().is_empty() {
        return emit_error(
            false,
            invalid("web artifact export requires an artifact id"),
        );
    }
    let Some(to) = to.filter(|path| !path.trim().is_empty()) else {
        return emit_error(false, invalid("web artifact export requires --to FILE"));
    };
    match rpc_response(
        root,
        "web.artifact.path",
        json!({ "session_id": session, "id": id }),
        Some(session),
    ) {
        Err(error) => emit_error(false, error),
        Ok(response) if response.status != "ok" => emit_response(false, response),
        Ok(response) => {
            let path = response
                .result
                .as_ref()
                .and_then(|value| value.get("path"))
                .and_then(|value| value.as_str())
                .unwrap_or("");
            if path.is_empty() {
                return emit_error(false, invalid("artifact path was empty"));
            }
            let bytes = match std::fs::read(path) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return emit_error(
                        false,
                        ErrorObject::new(
                            "ARTIFACT_IO",
                            format!("cannot read artifact {path}: {error}"),
                            response.request_id.clone(),
                            EXIT_WEB_ARTIFACT,
                            "retry greppy web artifact path ID",
                        ),
                    );
                }
            };
            if let Err(error) = export_regular_file(Path::new(&to), &bytes) {
                return emit_error(false, error);
            }
            println!("{to}");
            Ok(0)
        }
    }
}
