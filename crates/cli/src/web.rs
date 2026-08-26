//! `greppy web` client. Does not link V8 or Servo.

use super::*;
use greppy_web_client::{new_request_id, ErrorObject, Request, Response, SCHEMA};
use serde_json::json;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

pub const EXIT_WEB_INVALID: i32 = 30;
pub const EXIT_WEB_UNAVAILABLE: i32 = 31;
#[allow(dead_code)]
pub const EXIT_WEB_SESSION: i32 = 32;
#[allow(dead_code)]
pub const EXIT_WEB_SCRIPT: i32 = 33;
#[allow(dead_code)]
pub const EXIT_WEB_ENGINE: i32 = 34;
#[allow(dead_code)]
pub const EXIT_WEB_TIMEOUT: i32 = 35;
#[allow(dead_code)]
pub const EXIT_WEB_POLICY: i32 = 36;
#[allow(dead_code)]
pub const EXIT_WEB_LIMIT: i32 = 37;
#[allow(dead_code)]
pub const EXIT_WEB_WORKER: i32 = 38;
#[allow(dead_code)]
pub const EXIT_WEB_ARTIFACT: i32 = 39;

pub fn dispatch(command: WebCommand, root: Option<&str>) -> Result<i32> {
    match command {
        WebCommand::Status { json } => status(json, root),
        WebCommand::Doctor { json } => doctor(json, root),
        WebCommand::Session { command } => match command {
            WebSessionCommand::Create { profile, json } => rpc(
                root,
                json,
                "web.session.create",
                json!({ "profile": profile }),
                None,
            ),
            WebSessionCommand::List { json } => {
                rpc(root, json, "web.session.list", json!({}), None)
            }
            WebSessionCommand::Close { session, json } => rpc(
                root,
                json,
                "web.session.close",
                json!({ "session_id": session }),
                Some(session),
            ),
        },
        WebCommand::Run {
            session,
            script_file,
            script_stdin,
            timeout,
            json,
        } => run(root, session, script_file, script_stdin, timeout, json),
        WebCommand::Observe {
            session,
            format,
            json,
        } => rpc(
            root,
            json,
            "web.observe",
            json!({ "session_id": session, "format": format.unwrap_or_else(|| "agent-tree".into()) }),
            session,
        ),
        WebCommand::Screenshot {
            session,
            output,
            json,
        } => rpc(
            root,
            json,
            "web.screenshot",
            json!({ "session_id": session, "output": output }),
            session,
        ),
        WebCommand::Search {
            query,
            domain,
            limit,
            json,
        } => rpc(
            root,
            json,
            "web.search",
            json!({ "query": query, "domain": domain, "limit": limit }),
            None,
        ),
        WebCommand::Read { url, query, json } => rpc(
            root,
            json,
            "web.read",
            json!({ "url": url, "query": query }),
            None,
        ),
        WebCommand::Research {
            query,
            max_sources,
            depth,
            json,
        } => rpc(
            root,
            json,
            "web.research",
            json!({ "query": query, "max_sources": max_sources, "depth": depth }),
            None,
        ),
        WebCommand::Artifacts { session, json } => rpc(
            root,
            json,
            "web.artifacts",
            json!({ "session_id": session }),
            session,
        ),
    }
}

fn status(json: bool, root: Option<&str>) -> Result<i32> {
    match ensure_supervisor(root) {
        Ok(ctx) => rpc_on(&ctx, json, "web.status", json!({}), None),
        Err(error) => emit_error(json, error),
    }
}

fn doctor(json: bool, root: Option<&str>) -> Result<i32> {
    let supervisor = find_binary("web-runtime-supervisor");
    let controller = find_binary("web-controller-worker");
    let content = find_binary("web-content-worker");
    let bins_ok = [&supervisor, &controller, &content]
        .into_iter()
        .all(|path| path.as_ref().is_some_and(|path| path.exists()));
    if !bins_ok {
        return emit_error(
            json,
            unavailable("web-runtime binaries are not installed next to greppy"),
        );
    }
    match ensure_supervisor(root) {
        Ok(ctx) => rpc_on(&ctx, json, "web.doctor", json!({}), None),
        Err(error) => emit_error(json, error),
    }
}

fn run(
    root: Option<&str>,
    session: Option<String>,
    script_file: Option<String>,
    script_stdin: bool,
    timeout: Option<u64>,
    json: bool,
) -> Result<i32> {
    let Some(session) = session else {
        return emit_error(json, invalid("web run requires --session SESSION"));
    };
    if script_file.is_some() && script_stdin {
        return emit_error(
            json,
            invalid("web run accepts only one of --script-file or --script-stdin"),
        );
    }
    let (script_source, script_file_field, script_text) = if script_stdin {
        let mut text = String::new();
        io::stdin()
            .read_to_string(&mut text)
            .map_err(|error| Error::Io {
                context: "read --script-stdin".into(),
                source: error,
            })?;
        if text.is_empty() {
            return emit_error(json, invalid("--script-stdin was empty"));
        }
        ("stdin", None, Some(text))
    } else if let Some(path) = script_file {
        let text = std::fs::read_to_string(&path).map_err(|error| Error::Io {
            context: format!("read script {path}"),
            source: error,
        })?;
        ("file", Some(path), Some(text))
    } else {
        return emit_error(
            json,
            invalid("web run requires --script-file FILE or --script-stdin"),
        );
    };
    let mut payload = json!({
        "session_id": session,
        "script_source": script_source,
    });
    if let Some(path) = script_file_field {
        payload["script_file"] = json!(path);
    }
    if let Some(text) = script_text {
        payload["script_text"] = json!(text);
    }
    if let Some(timeout) = timeout {
        payload["timeout_seconds"] = json!(timeout);
    }
    rpc(root, json, "web.run", payload, Some(session))
}

fn rpc(
    root: Option<&str>,
    json_out: bool,
    operation: &str,
    payload: serde_json::Value,
    session_id: Option<String>,
) -> Result<i32> {
    match ensure_supervisor(root) {
        Ok(ctx) => rpc_on(&ctx, json_out, operation, payload, session_id),
        Err(error) => emit_error(json_out, error),
    }
}

struct SupervisorCtx {
    socket: PathBuf,
    run_id: String,
}

fn rpc_on(
    ctx: &SupervisorCtx,
    json_out: bool,
    operation: &str,
    payload: serde_json::Value,
    session_id: Option<String>,
) -> Result<i32> {
    #[cfg(not(unix))]
    {
        let _ = (ctx, operation, payload, session_id);
        return emit_error(json_out, unavailable("web runtime sockets require Unix"));
    }
    #[cfg(unix)]
    {
        let mut request = Request::new(&ctx.run_id, operation, payload);
        request.session_id = session_id;
        if operation == "web.run" {
            request.deadline_ms = 120_000;
        }
        match greppy_web_client::unix_request(&ctx.socket, &request, Duration::from_secs(120)) {
            Ok(response) => emit_response(json_out, response),
            Err(error) => emit_error(
                json_out,
                ErrorObject::new(
                    "runtime_unavailable",
                    error.to_string(),
                    request.request_id,
                    EXIT_WEB_UNAVAILABLE,
                    "retry greppy web doctor",
                ),
            ),
        }
    }
}

fn ensure_supervisor(root: Option<&str>) -> std::result::Result<SupervisorCtx, ErrorObject> {
    let supervisor = find_binary("web-runtime-supervisor")
        .ok_or_else(|| unavailable("web-runtime-supervisor is not installed"))?;
    let controller = find_binary("web-controller-worker")
        .ok_or_else(|| unavailable("web-controller-worker is not installed"))?;
    let content = find_binary("web-content-worker")
        .ok_or_else(|| unavailable("web-content-worker is not installed"))?;
    let run_id =
        std::env::var("GREPPY_RUN_ID").unwrap_or_else(|_| format!("run_{}", std::process::id()));
    #[cfg(not(unix))]
    {
        let _ = (root, supervisor, controller, content, run_id);
        return Err(unavailable("web runtime sockets require Unix"));
    }
    #[cfg(unix)]
    {
        let endpoint = crate::inference_daemon::Endpoint::for_identity("web-runtime", &run_id)
            .ok_or_else(|| unavailable("cannot allocate web-runtime socket"))?;
        let socket = PathBuf::from(endpoint.address());
        if socket.exists() {
            return Ok(SupervisorCtx { socket, run_id });
        }
        let identity = root.map(str::to_owned).unwrap_or_else(|| run_id.clone());
        let _ = identity;
        let spawned = crate::inference_daemon::spawn_once(&endpoint, || {
            let mut command = ProcessCommand::new(&supervisor);
            command
                .arg("--socket")
                .arg(endpoint.address())
                .arg("--run-id")
                .arg(&run_id)
                .arg("--controller-worker")
                .arg(&controller)
                .arg("--content-worker")
                .arg(&content)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            if let Ok(fixture) = std::env::var("GREPPY_WEB_FIXTURE_URL") {
                command.arg("--fixture-url").arg(fixture);
            }
            crate::inference_daemon::detach_command(&mut command);
            command.spawn().ok().map(|_| ())
        });
        match spawned {
            crate::inference_daemon::SpawnOutcome::Spawned
            | crate::inference_daemon::SpawnOutcome::Contended => {}
            _ => {
                return Err(unavailable("failed to spawn web-runtime-supervisor"));
            }
        }
        for delay in crate::inference_daemon::retry_delays() {
            if socket.exists() {
                return Ok(SupervisorCtx { socket, run_id });
            }
            std::thread::sleep(delay);
        }
        Err(unavailable(
            "web-runtime-supervisor did not create its socket",
        ))
    }
}

fn emit_response(json_out: bool, response: Response) -> Result<i32> {
    let code = response
        .error
        .as_ref()
        .map(|error| error.exit_code)
        .unwrap_or(0);
    emit_web(
        json_out,
        &serde_json::to_value(&response).unwrap_or(json!({})),
    )?;
    Ok(code)
}

fn emit_error(json_out: bool, error: ErrorObject) -> Result<i32> {
    let code = error.exit_code;
    let payload = json!({
        "schema": SCHEMA,
        "request_id": error.operation_id,
        "status": "error",
        "error": error,
    });
    emit_web(json_out, &payload)?;
    Ok(code)
}

fn emit_web(json_out: bool, payload: &serde_json::Value) -> Result<()> {
    if json_out {
        println!(
            "{}",
            serde_json::to_string(payload)
                .map_err(|error| Error::Invalid(format!("web json encode failed: {error}")))?
        );
    } else if let Some(error) = payload.get("error") {
        println!(
            "{}",
            error
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("web runtime error")
        );
    } else if let Some(result) = payload.get("result") {
        println!("{result}");
    } else {
        println!("{payload}");
    }
    Ok(())
}

fn unavailable(message: &str) -> ErrorObject {
    ErrorObject::new(
        "runtime_unavailable",
        message,
        new_request_id(),
        EXIT_WEB_UNAVAILABLE,
        "install web-runtime-supervisor, web-controller-worker, and web-content-worker",
    )
}

fn invalid(message: &str) -> ErrorObject {
    ErrorObject::new(
        "protocol_violation",
        message,
        new_request_id(),
        EXIT_WEB_INVALID,
        "see greppy web --help",
    )
}

fn find_binary(name: &str) -> Option<PathBuf> {
    let env_name = format!("GREPPY_{}", name.to_uppercase().replace('-', "_"));
    if let Ok(path) = std::env::var(&env_name) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_web_status_json() {
        let cli = Cli::try_parse_from(["greppy", "web", "status", "--json"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Status { json: true }
            })
        ));
    }

    #[test]
    fn parse_web_session_and_run() {
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "session",
            "create",
            "--profile",
            "project",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Session {
                    command: WebSessionCommand::Create { .. }
                }
            })
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "run",
            "--session",
            "wrs_1",
            "--script-file",
            "spec.mjs",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Run { json: true, .. }
            })
        ));
    }

    #[test]
    fn missing_named_binary_is_none() {
        assert!(find_binary("web-runtime-supervisor-missing-name").is_none());
    }
}
