//! Result and artifact web verbs already shipped.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use serde_json::json;

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
            format,
            json,
        } => {
            let Some(session) = session else {
                return emit_error(json, invalid("web observe requires --session SESSION"));
            };
            rpc(
                root,
                json,
                "web.observe",
                json!({ "session_id": session, "format": format.unwrap_or_else(|| "agent-tree".into()) }),
                Some(session),
            )
        }
        ResultsCommand::Screenshot {
            session,
            output,
            json,
        } => screenshot(root, session, output, json),
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
            let Some(session) = session else {
                return emit_error(json, invalid("web search requires --session SESSION"));
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
            let Some(session) = session else {
                return emit_error(json, invalid("web read requires --session SESSION"));
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
            let Some(session) = session else {
                return emit_error(json, invalid("web research requires --session SESSION"));
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
        ResultsCommand::Artifacts { session, json } => rpc(
            root,
            json,
            "web.artifacts",
            json!({ "session_id": session }),
            session,
        ),
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
        ResultsCommand::Heartbeat { session, seq, json } => rpc(
            root,
            json,
            "web.heartbeat",
            json!({ "session_id": session, "seq": seq.unwrap_or(1) }),
            session,
        ),
        ResultsCommand::Artifact { command } => match command {
            ArtifactCommand::List { session, json } => {
                let Some(session) = session else {
                    return emit_error(json, invalid("web artifact list requires --session SESSION"));
                };
                rpc(
                    root,
                    json,
                    "web.artifacts",
                    json!({ "session_id": session }),
                    Some(session),
                )
            }
        },
        ResultsCommand::Result { command } => match command {
            ResultCommand::Next {
                cursor,
                session,
                json,
            } => {
                let Some(session) = session else {
                    return emit_error(json, invalid("web result next requires --session SESSION"));
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
