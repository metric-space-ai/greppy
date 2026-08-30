//! Shared web-CLI helpers: RPC, supervisor spawn, envelope/error mapping.

use greppy_core::error::{Error, Result};
use greppy_web_client::{new_request_id, ErrorObject, Handshake, Request, Response, SCHEMA};
use serde_json::json;
use std::cell::RefCell;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, Instant};

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

pub fn web_runtime_socket(run_id: &str) -> Option<PathBuf> {
    crate::inference_daemon::Endpoint::for_identity("web-runtime", run_id)
        .map(|endpoint| PathBuf::from(endpoint.address()))
}

pub(super) fn web_run_id() -> String {
    if let Ok(run_id) = std::env::var("GREPPY_RUN_ID") {
        if !run_id.is_empty() {
            return run_id;
        }
    }
    if let Ok(agent) = std::env::var("GREPPY_WEB_AGENT") {
        let agent = agent.trim();
        if !agent.is_empty() {
            return format!("run_{agent}");
        }
    }
    format!("run_{}", std::process::id())
}

pub(super) fn inject_agent_id(mut payload: serde_json::Value) -> serde_json::Value {
    if let Ok(agent) = std::env::var("GREPPY_WEB_AGENT") {
        let agent = agent.trim();
        if !agent.is_empty() {
            if let Some(object) = payload.as_object_mut() {
                object
                    .entry("agent_id")
                    .or_insert_with(|| json!(agent));
            }
        }
    }
    payload
}

pub(super) fn status(json: bool, root: Option<&str>) -> Result<i32> {
    match ensure_supervisor(root, &SupervisorSpawn::default()) {
        Ok(ctx) => rpc_on(&ctx, json, "web.status", json!({}), None),
        Err(error) => emit_error(json, error),
    }
}

pub(super) fn runtime_run_id(root: Option<&str>) -> (String, String) {
    let run_id =
        web_run_id();
    let identity = match root {
        Some(root) => format!("{run_id}:{root}"),
        None => run_id.clone(),
    };
    (run_id, identity)
}

pub(super) fn runtime_status(json: bool, root: Option<&str>) -> Result<i32> {
    let (run_id, identity) = runtime_run_id(root);
    let Some(socket) = web_runtime_socket(&identity) else {
        return emit_error(json, unavailable("cannot allocate web-runtime socket"));
    };
    let running = socket_connected(&socket);
    let owned = match crate::web_attach::current_token() {
        Some(capability) if running => socket_is_live(&socket, &run_id, &capability),
        _ => false,
    };
    let payload = json!({
        "schema": SCHEMA,
        "status": "ok",
        "result": {
            "running": running,
            "owned": owned,
            "run_id": run_id,
            "socket": socket,
        }
    });
    emit_web(json, &payload)?;
    Ok(0)
}

pub(super) fn runtime_stop(json: bool, root: Option<&str>) -> Result<i32> {
    shutdown_if_running();
    runtime_status(json, root)
}

pub(super) fn runtime_restart(json: bool, root: Option<&str>) -> Result<i32> {
    if let Err(error) = crate::web_attach::claim_persistent_parent() {
        return emit_error(
            json,
            unavailable(&format!("failed to claim runtime owner: {error}")),
        );
    }
    shutdown_if_running();
    match ensure_supervisor(root, &SupervisorSpawn::default()) {
        Ok(ctx) => rpc_on(&ctx, json, "web.status", json!({}), None),
        Err(error) => emit_error(json, error),
    }
}

pub(super) fn doctor(json: bool, _root: Option<&str>) -> Result<i32> {
    crate::startup_trace("web.doctor.enter");
    let runtime = match resolve_runtime() {
        Ok(runtime) => runtime,
        Err(error) => return emit_error(json, error),
    };
    crate::startup_trace("web.doctor.resolved");
    let handshake = Handshake::runtime_facts();
    let stamp = runtime
        .dist
        .as_ref()
        .map(|dist| dist.join(".greppy-web-runtime-dist"));
    let payload = json!({
        "schema": SCHEMA,
        "status": "ok",
        "result": {
            "executable": runtime.executable,
            "dist": runtime.dist,
            "stamp": stamp,
            "protocol_version": handshake.protocol_version,
            "runtime_build_id": handshake.runtime_build_id,
            "playwright_compatibility_version": handshake.playwright_compatibility_version,
            "servo_revision": handshake.servo_revision,
            "v8_revision": handshake.v8_revision,
            "platform": handshake.platform,
            "architecture": handshake.architecture,
            "supported_capabilities": handshake.supported_capabilities,
            "compatibility_coverage_level": handshake.compatibility_coverage_level,
            "max_message_bytes": handshake.max_message_bytes,
            "max_artifact_bytes": handshake.max_artifact_bytes,
        }
    });
    emit_web(json, &payload)?;
    Ok(0)
}

pub(super) fn run(
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

pub(super) fn screenshot(
    root: Option<&str>,
    session: Option<String>,
    output: Option<String>,
    json: bool,
) -> Result<i32> {
    let payload = json!({ "session_id": session });
    match ensure_supervisor(root, &SupervisorSpawn::default()) {
        Err(error) => emit_error(json, error),
        Ok(ctx) => {
            #[cfg(not(unix))]
            {
                let _ = (ctx, payload, session, output);
                emit_error(json, unavailable("web runtime sockets require Unix"))
            }
            #[cfg(unix)]
            {
                let mut request = Request::new(&ctx.run_id, "web.screenshot", payload);
                request.session_id = session;
                request.capability = ctx.capability.clone();
                match greppy_web_client::unix_request(
                    &ctx.socket,
                    &request,
                    Duration::from_secs(120),
                ) {
                    Err(error) => emit_error(
                        json,
                        ErrorObject::new(
                            "runtime_unavailable",
                            error.to_string(),
                            request.request_id,
                            EXIT_WEB_UNAVAILABLE,
                            "retry greppy web doctor",
                        ),
                    ),
                    Ok(response) => {
                        if response.status == "ok" {
                            if let Some(dest) = output.as_deref() {
                                if let Err(error) =
                                    export_screenshot_artifact(&ctx.run_id, &response, dest)
                                {
                                    return emit_error(json, error);
                                }
                            }
                        }
                        emit_response(json, response)
                    }
                }
            }
        }
    }
}

pub(super) fn artifact_store_root(run_id: &str) -> PathBuf {
    let base = std::env::var("GREPPY_STORE_DIR")
        .or_else(|_| std::env::var("GREPPY_RUNTIME_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("greppy-web-runtime"));
    base.join("web-runtime").join(run_id)
}

pub(super) fn export_screenshot_artifact(
    run_id: &str,
    response: &Response,
    dest: &str,
) -> std::result::Result<(), ErrorObject> {
    let object_path = response
        .result
        .as_ref()
        .and_then(|value| value.get("object_path"))
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if object_path.is_empty()
        || Path::new(object_path).is_absolute()
        || Path::new(object_path)
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(invalid(
            "screenshot object_path is not a confined artifact path",
        ));
    }
    let src = artifact_store_root(run_id).join(object_path);
    if is_symlink(&src) {
        return Err(invalid("refusing symlink artifact object"));
    }
    let bytes = std::fs::read(&src).map_err(|error| {
        ErrorObject::new(
            "artifact_unavailable",
            format!("cannot read screenshot artifact: {error}"),
            new_request_id(),
            EXIT_WEB_ARTIFACT,
            "retry greppy web screenshot",
        )
    })?;
    export_regular_file(Path::new(dest), &bytes)
}

pub(super) fn lexical_output_path(dest: &Path) -> std::result::Result<PathBuf, ErrorObject> {
    if dest.as_os_str().is_empty() {
        return Err(invalid("--output path is empty"));
    }
    let mut out = if dest.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().map_err(|error| {
            invalid(&format!(
                "cannot resolve --output against current_dir: {error}"
            ))
        })?
    };
    for component in dest.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            Component::Normal(part) => out.push(part),
        }
    }
    if out.as_os_str().is_empty() {
        return Err(invalid("--output path is empty"));
    }
    Ok(out)
}

pub(super) fn export_regular_file(dest: &Path, bytes: &[u8]) -> std::result::Result<(), ErrorObject> {
    let dest = lexical_output_path(dest)?;
    let mut cursor = dest.clone();
    loop {
        match std::fs::symlink_metadata(&cursor) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(invalid(
                    "refusing --output that is or is reached through a symlink",
                ));
            }
            Ok(meta) if cursor == dest => {
                if meta.is_dir() {
                    return Err(invalid("refusing to overwrite a directory --output"));
                }
                return Err(invalid("refusing to overwrite an existing --output file"));
            }
            Ok(meta) if !meta.is_dir() => {
                return Err(invalid("--output ancestor is not a directory"));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound && cursor == dest => {}
            Err(error) => {
                return Err(invalid(&format!("cannot stat --output path: {error}")));
            }
        }
        match cursor.parent() {
            Some(parent) if parent != cursor.as_path() && !parent.as_os_str().is_empty() => {
                cursor = parent.to_path_buf();
            }
            _ => break,
        }
    }
    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&dest)
            .map_err(|error| invalid(&format!("cannot create --output: {error}")))?;
        file.write_all(bytes)
            .map_err(|error| invalid(&format!("cannot write --output: {error}")))?;
        file.flush()
            .map_err(|error| invalid(&format!("cannot flush --output: {error}")))?;
    }
    #[cfg(not(unix))]
    {
        let _ = bytes;
        return Err(unavailable("web screenshot --output requires Unix"));
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct SupervisorSpawn {
    pub(super) fixture_url: Option<String>,
    pub(super) search_endpoint: Option<String>,
}

pub(super) fn rpc(
    root: Option<&str>,
    json_out: bool,
    operation: &str,
    payload: serde_json::Value,
    session_id: Option<String>,
) -> Result<i32> {
    rpc_with_spawn(
        root,
        json_out,
        operation,
        payload,
        session_id,
        SupervisorSpawn::default(),
    )
}

pub(super) fn rpc_with_spawn(
    root: Option<&str>,
    json_out: bool,
    operation: &str,
    payload: serde_json::Value,
    session_id: Option<String>,
    spawn: SupervisorSpawn,
) -> Result<i32> {
    match ensure_supervisor(root, &spawn) {
        Ok(ctx) => rpc_on(&ctx, json_out, operation, payload, session_id),
        Err(error) => emit_error(json_out, error),
    }
}

pub(super) struct SupervisorCtx {
    socket: PathBuf,
    run_id: String,
    capability: String,
}

pub(super) fn rpc_on(
    ctx: &SupervisorCtx,
    json_out: bool,
    operation: &str,
    payload: serde_json::Value,
    session_id: Option<String>,
) -> Result<i32> {
    match rpc_on_response(ctx, operation, payload, session_id) {
        Ok(response) => emit_response(json_out, response),
        Err(error) => emit_error(json_out, error),
    }
}

pub(super) fn rpc_response(
    root: Option<&str>,
    operation: &str,
    payload: serde_json::Value,
    session_id: Option<String>,
) -> std::result::Result<Response, ErrorObject> {
    match ensure_supervisor(root, &SupervisorSpawn::default()) {
        Ok(ctx) => rpc_on_response(&ctx, operation, payload, session_id),
        Err(error) => Err(error),
    }
}

pub(super) fn rpc_on_response(
    ctx: &SupervisorCtx,
    operation: &str,
    payload: serde_json::Value,
    session_id: Option<String>,
) -> std::result::Result<Response, ErrorObject> {
    #[cfg(not(unix))]
    {
        let _ = (ctx, operation, payload, session_id);
        return Err(unavailable("web runtime sockets require Unix"));
    }
    #[cfg(unix)]
    {
        let payload = inject_agent_id(payload.clone());
        let mut request = Request::new(&ctx.run_id, operation, payload.clone());
        request.session_id = session_id;
        request.capability = ctx.capability.clone();
        let wait = if operation == "web.run" {
            let deadline_ms = payload
                .get("timeout_seconds")
                .and_then(|value| value.as_u64())
                .map(|seconds| seconds.saturating_mul(1000))
                .unwrap_or(120_000)
                .max(1);
            request.deadline_ms = deadline_ms;
            Duration::from_millis(deadline_ms.saturating_add(5_000))
        } else {
            Duration::from_secs(120)
        };
        greppy_web_client::unix_request(&ctx.socket, &request, wait).map_err(|error| {
            ErrorObject::new(
                "runtime_unavailable",
                error.to_string(),
                request.request_id,
                EXIT_WEB_UNAVAILABLE,
                "retry greppy web doctor",
            )
        })
    }
}

pub(super) fn ensure_supervisor(
    root: Option<&str>,
    spawn: &SupervisorSpawn,
) -> std::result::Result<SupervisorCtx, ErrorObject> {
    if std::env::var_os("GREPPY_WEB_FIXTURE_URL").is_some() {
        return Err(invalid(
            "GREPPY_WEB_FIXTURE_URL is not a production path; pass --fixture-url to web-runtime",
        ));
    }
    let ResolvedRuntime { dist, executable } = resolve_runtime()?;
    let run_id =
        web_run_id();
    #[cfg(not(unix))]
    {
        let _ = (root, dist, executable, run_id, spawn);
        return Err(unavailable("web runtime sockets require Unix"));
    }
    #[cfg(unix)]
    {
        let identity = match root {
            Some(root) => format!("{run_id}:{root}"),
            None => run_id.clone(),
        };
        let endpoint = crate::inference_daemon::Endpoint::for_identity("web-runtime", &identity)
            .ok_or_else(|| unavailable("cannot allocate web-runtime socket"))?;
        let socket = PathBuf::from(endpoint.address());
        if crate::web_attach::current_token().is_none() && socket_connected(&socket) {
            return Err(not_owned(
                "web-runtime is running but this process has no inherited attach token",
            ));
        }
        let capability = match crate::web_attach::current_token() {
            Some(token) => token,
            None => crate::web_attach::become_standalone_owner().map_err(|error| {
                unavailable(&format!("failed to generate attach token: {error}"))
            })?,
        };
        if socket_is_live(&socket, &run_id, &capability) {
            return Ok(SupervisorCtx {
                socket,
                run_id,
                capability,
            });
        }
        if socket_connected(&socket) {
            return Err(not_owned(
                "attach capability does not match the live web-runtime supervisor",
            ));
        }
        let _ = std::fs::remove_file(&socket);
        let spawned_child: RefCell<Option<std::process::Child>> = RefCell::new(None);
        let issued_for_child = capability.clone();
        let mut attach_pass = None;
        let mut attach_error = None;
        let spawned = crate::inference_daemon::spawn_once(&endpoint, || {
            let mut command = ProcessCommand::new(&executable);
            command
                .arg("--socket")
                .arg(endpoint.address())
                .arg("--run-id")
                .arg(&run_id);
            if let Some(dist) = &dist {
                command.arg("--dist").arg(dist);
            }
            if let Some(fixture_url) = &spawn.fixture_url {
                command.arg("--fixture-url").arg(fixture_url);
            }
            if let Some(search_endpoint) = &spawn.search_endpoint {
                command.arg("--search-endpoint").arg(search_endpoint);
            }
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            crate::inference_daemon::detach_command(&mut command);
            match crate::web_attach::give_child_attach_token(&mut command, &issued_for_child) {
                Ok(pass) => attach_pass = Some(pass),
                Err(error) => {
                    attach_error = Some(error);
                    return None;
                }
            }
            match command.spawn() {
                Ok(child) => {
                    *spawned_child.borrow_mut() = Some(child);
                    Some(())
                }
                Err(_) => None,
            }
        });
        drop(attach_pass);
        if let Some(error) = attach_error {
            return Err(unavailable(&format!(
                "failed to pass attach token on inherited fd: {error}"
            )));
        }
        match spawned {
            crate::inference_daemon::SpawnOutcome::Spawned
            | crate::inference_daemon::SpawnOutcome::Contended => {}
            crate::inference_daemon::SpawnOutcome::Cooldown => {
                crate::inference_daemon::record_spawn_failure(&endpoint, spawned.attempted());
                return Err(unavailable(
                    "web-runtime recently crashed; wait before retrying",
                ));
            }
            crate::inference_daemon::SpawnOutcome::SpawnFailed => {
                crate::inference_daemon::record_spawn_failure(&endpoint, true);
                return Err(unavailable("failed to spawn web-runtime"));
            }
        }
        let started = Instant::now();
        let budget = Duration::from_secs(60);
        loop {
            if socket_is_live(&socket, &run_id, &capability) {
                return Ok(SupervisorCtx {
                    socket,
                    run_id,
                    capability,
                });
            }
            if started.elapsed() >= budget {
                break;
            }
            if started.elapsed() >= Duration::from_secs(1) {
                if let Some(child) = spawned_child.borrow_mut().as_mut() {
                    match child.try_wait() {
                        Ok(Some(_)) | Err(_) => break,
                        Ok(None) => {}
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        if let Some(mut child) = spawned_child.borrow_mut().take() {
            reap_detached_runtime(&mut child, &socket, &run_id, &capability);
        }
        crate::inference_daemon::record_spawn_failure(&endpoint, spawned.attempted());
        Err(unavailable("web-runtime did not create its socket"))
    }
}

pub(super) fn socket_connected(socket: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        std::os::unix::net::UnixStream::connect(socket).is_ok()
    }
    #[cfg(not(unix))]
    {
        let _ = socket;
        false
    }
}

pub(super) fn not_owned(message: &str) -> ErrorObject {
    ErrorObject::new(
        "session_not_owned",
        message,
        new_request_id(),
        EXIT_WEB_SESSION,
        "inherit the parent-issued attach token on fd 4",
    )
}

pub(super) fn socket_is_live(socket: &std::path::Path, run_id: &str, capability: &str) -> bool {
    if !socket.exists() {
        return false;
    }
    let mut probe = Request::new(run_id, "web.status", serde_json::json!({}));
    probe.capability = capability.to_owned();
    greppy_web_client::unix_request(socket, &probe, Duration::from_millis(400)).is_ok()
}

pub fn shutdown_if_running() {
    #[cfg(unix)]
    {
        let run_id = web_run_id();
        let Some(capability) = crate::web_attach::current_token() else {
            return;
        };
        if let Some(endpoint) =
            crate::inference_daemon::Endpoint::for_identity("web-runtime", &run_id)
        {
            let socket = PathBuf::from(endpoint.address());
            let mut request = Request::new(&run_id, "web.shutdown", json!({}));
            request.capability = capability;
            let _ = greppy_web_client::unix_request(&socket, &request, Duration::from_secs(3));
        }
    }
}

pub(super) fn wait_child_exit(child: &mut std::process::Child, budget: Duration) -> bool {
    let deadline = Instant::now() + budget;
    loop {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return true,
            Ok(None) if Instant::now() >= deadline => return false,
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

pub(super) fn reap_detached_runtime(
    child: &mut std::process::Child,
    socket: &std::path::Path,
    run_id: &str,
    capability: &str,
) {
    // Supervisor owns worker Children/PGIDs. Ask it to shut them down; do not pgrep.
    if socket.exists() {
        let mut request = Request::new(run_id, "web.shutdown", json!({}));
        request.capability = capability.to_owned();
        let _ = greppy_web_client::unix_request(socket, &request, Duration::from_secs(2));
    }
    if wait_child_exit(child, Duration::from_secs(3)) {
        return;
    }
    #[cfg(unix)]
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    if wait_child_exit(child, Duration::from_secs(1)) {
        return;
    }
    let _ = child.kill();
    let _ = child.try_wait();
}

pub(super) fn emit_response(json_out: bool, response: Response) -> Result<i32> {
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

pub(super) fn emit_error(json_out: bool, error: ErrorObject) -> Result<i32> {
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

pub(super) fn emit_web(json_out: bool, payload: &serde_json::Value) -> Result<()> {
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

pub(super) fn unavailable(message: &str) -> ErrorObject {
    ErrorObject::new(
        "runtime_unavailable",
        message,
        new_request_id(),
        EXIT_WEB_UNAVAILABLE,
        "install the web-runtime distributable (one linked executable)",
    )
}

pub(super) fn invalid(message: &str) -> ErrorObject {
    ErrorObject::new(
        "protocol_violation",
        message,
        new_request_id(),
        EXIT_WEB_INVALID,
        "see greppy web --help",
    )
}

pub(super) struct ResolvedRuntime {
    pub(super) dist: Option<PathBuf>,
    pub(super) executable: PathBuf,
}

pub(super) fn is_symlink(path: &std::path::Path) -> bool {
    path.symlink_metadata()
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false)
}

pub(super) fn resolve_runtime() -> std::result::Result<ResolvedRuntime, ErrorObject> {
    if let Ok(dist) = std::env::var("GREPPY_WEB_RUNTIME_DIST") {
        return images_from_dist(std::path::Path::new(&dist));
    }
    if let Ok(path) = std::env::var("GREPPY_WEB_RUNTIME") {
        let path = PathBuf::from(path);
        return runtime_from_file(&path, None)
            .ok_or_else(|| unavailable("GREPPY_WEB_RUNTIME is not a usable executable"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling_dist = dir.join("web-runtime");
            if sibling_dist.join(".greppy-web-runtime-dist").is_file() {
                return images_from_dist(&sibling_dist);
            }
            if let Some(runtime) = runtime_from_file(&dir.join("web-runtime"), None) {
                return Ok(runtime);
            }
            if exe.file_name().is_some_and(|name| name == "web-runtime") {
                if let Some(runtime) = runtime_from_file(&exe, None) {
                    return Ok(runtime);
                }
            }
        }
    }
    images_from_path_env().ok_or_else(|| unavailable("web-runtime distributable is not installed"))
}

pub(super) fn images_from_dist(dist: &std::path::Path) -> std::result::Result<ResolvedRuntime, ErrorObject> {
    if is_symlink(dist) {
        return Err(unavailable("refusing symlink web-runtime dist"));
    }
    let stamp = dist.join(".greppy-web-runtime-dist");
    if is_symlink(&stamp) || !stamp.is_file() {
        return Err(unavailable(
            "web-runtime dist is missing the .greppy-web-runtime-dist stamp",
        ));
    }
    let bin = dist.join("bin");
    if is_symlink(&bin) {
        return Err(unavailable("refusing symlink web-runtime dist/bin"));
    }
    if !bin.is_dir() {
        return Err(unavailable("web-runtime dist is missing bin/web-runtime"));
    }
    let executable = bin.join("web-runtime");
    if is_symlink(&executable) {
        return Err(unavailable("refusing symlink web-runtime dist/bin member"));
    }
    if !executable.is_file() {
        return Err(unavailable("web-runtime dist is missing bin/web-runtime"));
    }
    if let Ok(entries) = std::fs::read_dir(&bin) {
        for entry in entries.flatten() {
            if entry.file_name() != "web-runtime" {
                return Err(unavailable("web-runtime dist/bin has unexpected members"));
            }
        }
    }
    Ok(ResolvedRuntime {
        dist: Some(dist.to_path_buf()),
        executable,
    })
}

pub(super) fn runtime_from_file(path: &std::path::Path, dist: Option<PathBuf>) -> Option<ResolvedRuntime> {
    if is_symlink(path) || !path.is_file() {
        return None;
    }
    Some(ResolvedRuntime {
        dist,
        executable: path.to_path_buf(),
    })
}

pub(super) fn images_from_path_env() -> Option<ResolvedRuntime> {
    Some(ResolvedRuntime {
        dist: None,
        executable: find_binary("web-runtime")?,
    })
}

pub(super) fn find_binary(name: &str) -> Option<PathBuf> {
    let env_name = format!("GREPPY_{}", name.to_uppercase().replace('-', "_"));
    if let Ok(path) = std::env::var(&env_name) {
        let path = PathBuf::from(path);
        if path.is_file() && !is_symlink(&path) {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(name);
            if candidate.is_file() && !is_symlink(&candidate) {
                return Some(candidate);
            }
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(name);
            if candidate.is_file() && !is_symlink(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

pub(super) fn query_syntax(message: &str) -> ErrorObject {
    ErrorObject::new(
        "QUERY_SYNTAX",
        message,
        new_request_id(),
        EXIT_WEB_INVALID,
        "see greppy web --help",
    )
}

/// Engine locator recipe built from a SPEC §3 target query.
#[derive(Debug)]
pub(super) struct ParsedTarget {
    pub selector: serde_json::Value,
}

pub(super) fn parse_target(
    raw: &str,
    first: bool,
    last: bool,
    nth: Option<i64>,
) -> std::result::Result<ParsedTarget, ErrorObject> {
    let nth_count = [first, last, nth.is_some()]
        .into_iter()
        .filter(|flag| *flag)
        .count();
    if nth_count > 1 {
        return Err(query_syntax(
            "use only one of --first, --last, or --nth",
        ));
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(query_syntax("target query is empty"));
    }
    if trimmed.starts_with('@') {
        let rest = &trimmed[1..];
        if rest.bytes().all(|b| b.is_ascii_digit()) && !rest.is_empty() {
            return Err(query_syntax(
                "ref @N requires an observe snapshot that stores locator recipes",
            ));
        }
        return Err(query_syntax("ref must be @ followed by digits"));
    }
    let mut css = None;
    let mut xpath = None;
    let mut text = None;
    let mut role = None;
    let mut name = None;
    let mut rest = trimmed;
    while !rest.is_empty() {
        let (atom, next) = parse_target_atom(rest)?;
        match atom {
            TargetAtom::Css(value) => assign_once(&mut css, value, "css")?,
            TargetAtom::Xpath(value) => assign_once(&mut xpath, value, "xpath")?,
            TargetAtom::Text(value) => assign_once(&mut text, value, "text")?,
            TargetAtom::Role(value) => assign_once(&mut role, value, "role")?,
            TargetAtom::Name(value) => assign_once(&mut name, value, "name")?,
        }
        rest = next.trim_start();
    }
    let kinds = [css.is_some(), xpath.is_some(), text.is_some(), role.is_some()]
        .into_iter()
        .filter(|flag| *flag)
        .count();
    if kinds != 1 {
        return Err(query_syntax(
            "target must be one of css=, xpath=, text=, or role=",
        ));
    }
    if name.is_some() && role.is_none() {
        return Err(query_syntax("name= is only valid with role="));
    }
    let mut selector = if let Some(value) = css {
        json!({ "type": "css", "value": value })
    } else if let Some(value) = xpath {
        json!({ "type": "xpath", "value": value })
    } else if let Some(value) = text {
        json!({ "type": "text", "value": value })
    } else {
        let mut object = serde_json::Map::new();
        object.insert("type".into(), json!("role"));
        object.insert("role".into(), json!(role.expect("role set")));
        if let Some(name) = name {
            object.insert("name".into(), json!(name));
        }
        serde_json::Value::Object(object)
    };
    if let Some(index) = if first {
        Some(0)
    } else if last {
        Some(-1)
    } else {
        nth
    } {
        if let Some(object) = selector.as_object_mut() {
            object.insert("nth".into(), json!(index));
        }
    }
    Ok(ParsedTarget { selector })
}

enum TargetAtom {
    Css(String),
    Xpath(String),
    Text(String),
    Role(String),
    Name(String),
}

fn assign_once(
    slot: &mut Option<String>,
    value: String,
    label: &str,
) -> std::result::Result<(), ErrorObject> {
    if slot.is_some() {
        return Err(query_syntax(&format!("{label}= specified more than once")));
    }
    *slot = Some(value);
    Ok(())
}

fn parse_target_atom(input: &str) -> std::result::Result<(TargetAtom, &str), ErrorObject> {
    let input = input.trim_start();
    for (prefix, ctor) in [
        ("css=", TargetAtom::Css as fn(String) -> TargetAtom),
        ("xpath=", TargetAtom::Xpath),
        ("text=", TargetAtom::Text),
        ("role=", TargetAtom::Role),
        ("name=", TargetAtom::Name),
    ] {
        if let Some(rest) = input.strip_prefix(prefix) {
            let (value, next) = parse_selector_value(rest)?;
            if value.is_empty() {
                return Err(query_syntax(&format!("{prefix} value is empty")));
            }
            return Ok((ctor(value), next));
        }
    }
    if input.starts_with("text~/") || input.starts_with("name~/") {
        return Err(query_syntax(
            "regex targets (text~/…/, name~/…/) land with find/extract",
        ));
    }
    Err(query_syntax(
        "target must start with css=, xpath=, text=, or role=",
    ))
}

fn parse_selector_value(input: &str) -> std::result::Result<(String, &str), ErrorObject> {
    if let Some(rest) = input.strip_prefix('"') {
        let mut out = String::new();
        let mut chars = rest.char_indices();
        while let Some((index, ch)) = chars.next() {
            match ch {
                '\\' => match chars.next() {
                    Some((_, escaped)) => out.push(escaped),
                    None => return Err(query_syntax("unterminated escape in quoted target")),
                },
                '"' => return Ok((out, &rest[index + 1..])),
                other => out.push(other),
            }
        }
        return Err(query_syntax("unterminated quoted target"));
    }
    let mut out = String::new();
    let mut chars = input.char_indices();
    while let Some((index, ch)) = chars.next() {
        match ch {
            '\\' => match chars.next() {
                Some((_, escaped)) => out.push(escaped),
                None => return Err(query_syntax("unterminated escape in target")),
            },
            c if c.is_whitespace() => return Ok((out, &input[index..])),
            other => out.push(other),
        }
    }
    Ok((out, ""))
}

#[cfg(test)]
mod target_tests {
    use super::*;

    #[test]
    fn parse_css_quoted_and_nth() {
        let parsed = parse_target(r#"css="div > a""#, false, false, Some(2)).unwrap();
        assert_eq!(parsed.selector["type"], "css");
        assert_eq!(parsed.selector["value"], "div > a");
        assert_eq!(parsed.selector["nth"], 2);
    }

    #[test]
    fn parse_role_with_name() {
        let parsed = parse_target("role=button name=Continue", false, false, None).unwrap();
        assert_eq!(parsed.selector["type"], "role");
        assert_eq!(parsed.selector["role"], "button");
        assert_eq!(parsed.selector["name"], "Continue");
    }

    #[test]
    fn parse_ref_is_query_syntax_until_observe_stores_recipes() {
        let error = parse_target("@12", false, true, None).unwrap_err();
        assert_eq!(error.code, "QUERY_SYNTAX");
        assert!(error.message.contains("observe"), "{}", error.message);
    }

    #[test]
    fn parse_rejects_combined_nth_flags() {
        let error = parse_target("css=a", true, true, None).unwrap_err();
        assert_eq!(error.code, "QUERY_SYNTAX");
    }
}

