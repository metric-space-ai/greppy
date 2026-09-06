//! Unix-socket client/supervisor daemon (guide §6.3, §9).

use crate::artifacts::ArtifactStore;
use crate::locator_diagnostics::{failure_observation_budget, recovery_for_locator_error, recovery_with_observed_state};
use crate::policy::{decide_url, NetworkProfile, UrlDecision};
use crate::protocol::{Message, WorkerKind};
use crate::session::{LocatorSnapshot, Session, SessionState};
use crate::supervisor::WorkerProcess;
use greppy_web_client::{
    new_session_id, read_frame, write_frame, ErrorObject, Handshake, Request, Response, SCHEMA,
};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[path = "daemon_workflow.rs"]
mod workflow;
fn isolated_id(value: &str) -> Result<&str, String> {
    if value.is_empty()
        || value.len() > 80
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("session/request id is not an isolated path component".to_owned());
    }
    Ok(value)
}

fn script_stage_dir(run_id: &str, session_id: &str, request_id: &str) -> Result<PathBuf, String> {
    Ok(std::env::temp_dir()
        .join("greppy-web-runtime")
        .join(isolated_id(run_id)?)
        .join("sessions")
        .join(isolated_id(session_id)?)
        .join("script-root")
        .join(isolated_id(request_id)?))
}

fn remove_script_stage(run_id: &str, session_id: &str, request_id: Option<&str>) {
    let Ok(run_id) = isolated_id(run_id) else {
        return;
    };
    let Ok(session_id) = isolated_id(session_id) else {
        return;
    };
    let mut path = std::env::temp_dir()
        .join("greppy-web-runtime")
        .join(run_id)
        .join("sessions")
        .join(session_id)
        .join("script-root");
    if let Some(request_id) = request_id {
        let Ok(request_id) = isolated_id(request_id) else {
            return;
        };
        path.push(request_id);
    }
    let _ = std::fs::remove_dir_all(path);
}

struct ScriptStageGuard {
    run_id: String,
    session_id: String,
    request_id: String,
}

impl Drop for ScriptStageGuard {
    fn drop(&mut self) {
        remove_script_stage(&self.run_id, &self.session_id, Some(&self.request_id));
    }
}

fn refuse_unbounded_script_root(root: &Path) -> Result<(), String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize script root: {error}"))?;
    if root == Path::new("/") {
        return Err("script root cannot be filesystem root".to_owned());
    }
    const FORBIDDEN: &[&str] = &[
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/dev",
        "/proc",
        "/sys",
        "/root",
        "/System",
        "/Library",
        "/Users",
        "/home",
        "/var",
        "/private/var",
        "/private/etc",
        "/tmp",
        "/private/tmp",
    ];
    if FORBIDDEN.iter().any(|prefix| root == Path::new(prefix)) {
        return Err(format!(
            "script root cannot be a system directory: {}",
            root.display()
        ));
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() && root == Path::new(&home) {
            return Err("script root cannot be the home directory".to_owned());
        }
    }
    if root.components().count() < 4 {
        return Err("script root is too shallow to grant".to_owned());
    }
    Ok(())
}

fn path_is_within_root(root: &Path, candidate: &Path) -> bool {
    let Ok(root) = root.canonicalize() else {
        return false;
    };
    let Ok(candidate) = candidate.canonicalize() else {
        return false;
    };
    candidate.starts_with(root)
}
fn stage_script_for_controller(
    script_file: &str,
    run_id: &str,
    session_id: &str,
    request_id: &str,
) -> Result<String, String> {
    let src = Path::new(script_file)
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize script file: {error}"))?;
    if !src.is_file() {
        return Err("script_file is not a file".to_owned());
    }
    let root = src
        .parent()
        .ok_or_else(|| "script file has no parent directory".to_owned())?;
    refuse_unbounded_script_root(root)?;
    let dest = script_stage_dir(run_id, session_id, request_id)?;
    if dest.exists() {
        let _ = std::fs::remove_dir_all(&dest);
    }
    let file_name = src
        .file_name()
        .ok_or_else(|| "script file name missing".to_owned())?;
    let staged = dest.join(file_name);
    let staged_result = (|| {
        std::fs::create_dir_all(&dest)
            .map_err(|error| format!("cannot stage script root: {error}"))?;
        copy_granted_modules(root, root, &dest, &mut 0, &mut 0)?;
        if !staged.is_file() {
            std::fs::copy(&src, &staged)
                .map_err(|error| format!("cannot stage script file: {error}"))?;
        }
        let dest_canon = dest
            .canonicalize()
            .map_err(|error| format!("cannot canonicalize staged root: {error}"))?;
        let staged_canon = staged
            .canonicalize()
            .map_err(|error| format!("cannot canonicalize staged script: {error}"))?;
        if !path_is_within_root(&dest_canon, &staged_canon) {
            return Err("staged script escaped isolated temp path".to_owned());
        }
        Ok(staged.to_string_lossy().into_owned())
    })();
    if staged_result.is_err() {
        let _ = std::fs::remove_dir_all(&dest);
    }
    staged_result
}

fn copy_granted_modules(
    root: &Path,
    from: &Path,
    to: &Path,
    files: &mut u32,
    bytes: &mut u64,
) -> Result<(), String> {
    const MAX_FILES: u32 = 128;
    const MAX_BYTES: u64 = 4 * 1024 * 1024;
    let from_canon = from
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize script walk: {error}"))?;
    if !path_is_within_root(root, &from_canon) {
        return Err("script walk escaped granted root".to_owned());
    }
    std::fs::create_dir_all(to).map_err(|error| format!("cannot stage script dir: {error}"))?;
    let entries =
        std::fs::read_dir(from).map_err(|error| format!("cannot read script root: {error}"))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read script root: {error}"))?;
        let name = entry.file_name();
        let name_text = name.to_string_lossy();
        if name_text.starts_with('.') || name_text == ".." || name_text.contains('/') {
            continue;
        }
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat script root: {error}"))?;
        if file_type.is_symlink() {
            continue;
        }
        let dest = to.join(&name);
        if dest.parent() != Some(to) {
            return Err("staged path escaped destination directory".to_owned());
        }
        if file_type.is_dir() {
            copy_granted_modules(root, &entry.path(), &dest, files, bytes)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let canonical = match entry.path().canonicalize() {
            Ok(path) => path,
            Err(_) => {
                return Err("cannot canonicalize granted module".to_owned());
            }
        };
        if !path_is_within_root(root, &canonical) {
            return Err("granted module escaped script root".to_owned());
        }
        let ext = canonical
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        if !matches!(ext, "mjs" | "js" | "cjs" | "json") {
            continue;
        }
        let meta = std::fs::metadata(&canonical)
            .map_err(|error| format!("cannot stat granted module: {error}"))?;
        *files = files.saturating_add(1);
        *bytes = bytes.saturating_add(meta.len());
        if *files > MAX_FILES || *bytes > MAX_BYTES {
            return Err("granted script root exceeds module copy budget".to_owned());
        }
        std::fs::copy(&canonical, &dest)
            .map_err(|error| format!("cannot stage granted module: {error}"))?;
    }
    Ok(())
}

pub struct DaemonConfig {
    pub socket: PathBuf,
    pub run_id: String,
    pub fixture_url: Option<String>,
    pub search_endpoint: Option<String>,
    pub idle_ttl: Duration,
}

pub fn serve(config: DaemonConfig) -> io::Result<()> {
    if let Some(parent) = config.socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[allow(fn_to_numeric_cast)]
    unsafe {
        libc::signal(
            libc::SIGTERM,
            crate::supervisor::handle_supervisor_sigterm as *const () as libc::sighandler_t,
        );
    }
    let started = Instant::now();
    let attach = crate::worker::take_parent_attach_token().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "supervisor requires inherited attach token on fd 4",
        )
    })?;
    let early_control = Arc::new(RunControl::new());
    match crate::supervisor::warmup_parent_image() {
        Ok(hash) => {
            if crate::supervisor::phase_trace_enabled() {
                eprintln!("web-runtime: phase parent-image elapsed_ms={}", hash.as_millis());
            }
        }
        Err(error) => {
            if crate::supervisor::phase_trace_enabled() {
                eprintln!("web-runtime: phase parent-image error={error}");
            }
        }
    }
    if crate::supervisor::phase_trace_enabled() {
        if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase start-workers socket={}",
            config.socket.display()
        ); }
    }
    let mut daemon = Daemon::start(config, attach, Arc::clone(&early_control))?;
    if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase bind-socket socket={}",
        daemon.socket.display()
    ); }
    let listener = bind_socket_healing_stale(&daemon.socket)?;
    let mut permissions = std::fs::metadata(&daemon.socket)?.permissions();
    permissions.set_mode(0o600);
    std::fs::set_permissions(&daemon.socket, permissions)?;
    if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase listening"); } }
    let (tx, rx) = mpsc::channel::<(UnixStream, Request)>();
    let accept_attach = daemon.attach_capability.clone();
    let accept_control = Arc::clone(&early_control);
    thread::Builder::new()
        .name("web-runtime-accept".into())
        .spawn(move || accept_loop(listener, tx, accept_control, accept_attach))
        .map_err(io::Error::other)?;
    if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase request-ready elapsed_ms={}",
        started.elapsed().as_millis()
    ); }
    loop {
        let (mut stream, request) = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(pair) => pair,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if daemon.exiting {
                    break;
                }
                daemon.reap_idle_sessions();
                daemon
                    .run_control
                    .publish_sessions(snapshot_session_rows(&daemon.sessions, &daemon.run_control));
                if daemon.should_idle_exit() {
                    daemon.idle_exit();
                    break;
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let operation = request.operation.clone();
        let response = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            daemon.handle(request.clone())
        })) {
            Ok(response) => response,
            Err(_) => {
                if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase handle-panic operation={operation}"); } }
                Response::error(
                    &request,
                    ErrorObject::new(
                        "engine_error",
                        format!("supervisor panicked while handling {operation}"),
                        request.request_id.clone(),
                        38,
                        "retry greppy web run",
                    ),
                )
            }
        };
        let _ = write_frame(&mut stream, &response);
        if daemon.exiting {
            break;
        }
    }
    Ok(())
}

fn capability_matches(expected: &str, provided: &str) -> bool {
    if expected.is_empty() || provided.len() != expected.len() {
        return false;
    }
    provided
        .as_bytes()
        .iter()
        .zip(expected.as_bytes())
        .fold(0_u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

struct LateEngineResult {
    session_id: String,
    target_request_id: String,
    engine_request_id: u64,
    ok: bool,
    error: Option<String>,
}

struct RunControl {
    busy: std::sync::Mutex<Option<(String, String)>>,
    cancel_target: std::sync::Mutex<Option<(String, String)>>,
    completed: std::sync::Mutex<HashSet<(String, String)>>,
    heartbeats: std::sync::Mutex<Vec<String>>,
    session_rows: std::sync::Mutex<Vec<serde_json::Value>>,
    late_engine_result: std::sync::Mutex<Option<LateEngineResult>>,
    discarded_engine_results: AtomicU64,
    controller_pid: AtomicU32,
    content_pid: AtomicU32,
    controller_generation: AtomicU64,
    content_generation: AtomicU64,
}

impl RunControl {
    fn new() -> Self {
        Self {
            busy: std::sync::Mutex::new(None),
            cancel_target: std::sync::Mutex::new(None),
            completed: std::sync::Mutex::new(HashSet::new()),
            heartbeats: std::sync::Mutex::new(Vec::new()),
            session_rows: std::sync::Mutex::new(Vec::new()),
            late_engine_result: std::sync::Mutex::new(None),
            discarded_engine_results: AtomicU64::new(0),
            controller_pid: AtomicU32::new(0),
            content_pid: AtomicU32::new(0),
            controller_generation: AtomicU64::new(1),
            content_generation: AtomicU64::new(1),
        }
    }

    fn publish_sessions(&self, rows: Vec<serde_json::Value>) {
        *self.session_rows.lock().unwrap_or_else(|e| e.into_inner()) = rows;
    }
}

fn snapshot_session_rows(
    sessions: &HashMap<String, Session>,
    control: &RunControl,
) -> Vec<serde_json::Value> {
    let controller_pid = control.controller_pid.load(Ordering::Relaxed);
    let content_pid = control.content_pid.load(Ordering::Relaxed);
    let controller_generation = control.controller_generation.load(Ordering::Relaxed);
    let content_generation = control.content_generation.load(Ordering::Relaxed);
    sessions
        .values()
        .map(|session| {
            serde_json::json!({
                "session_id": session.id,
                "state": format!("{:?}", session.state).to_lowercase(),
                "run_id": session.run_id,
                "operation_id": session.operation_id,
                "heartbeat_age_ms": session.last_heartbeat.elapsed().as_millis() as u64,
                "inflight_engine_request_id": session.inflight_engine_request_id,
                "inflight_engine_method": session.inflight_engine_method,
                "discarded_engine_results": session.discarded_engine_results,
                "owner": session.owner,
                "controller_pid": controller_pid,
                "content_pid": content_pid,
                "controller_generation": controller_generation,
                "content_generation": content_generation,
            })
        })
        .collect()
}

/// Remove this runtime's control socket and its sibling attach token.
/// Called on orderly shutdown so the next start never meets a stale path.
fn remove_runtime_socket_files(socket: &Path) {
    let _ = std::fs::remove_file(socket);
    let _ = std::fs::remove_file(socket.with_extension("attach"));
}

/// Bind the control socket, healing a stale one left by a previous runtime.
///
/// Nothing removed the socket when a runtime exited (finding 040), so a
/// crash, a kill, or even an ordinary `runtime stop` left the path behind.
/// The next start then failed with EADDRINUSE, which the CLI reported as
/// "did not create its socket" followed by a spawn cooldown -- the caller
/// saw silence, not a cause. A stale path is only removed once a connect
/// proves nobody is listening; a live runtime keeps its socket and the
/// caller gets the real address-in-use error.
fn bind_socket_healing_stale(path: &Path) -> io::Result<UnixListener> {
    match UnixListener::bind(path) {
        Ok(listener) => return Ok(listener),
        Err(error) if error.kind() != io::ErrorKind::AddrInUse => return Err(error),
        Err(error) => {
            if UnixStream::connect(path).is_ok() {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!(
                        "another web-runtime is already listening on {}",
                        path.display()
                    ),
                ));
            }
            eprintln!(
                "web-runtime: removing stale socket {} left by a previous runtime",
                path.display()
            );
            std::fs::remove_file(path).map_err(|remove_error| {
                io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!(
                        "stale socket {} could not be removed ({remove_error}); original bind error: {error}",
                        path.display()
                    ),
                )
            })?;
            UnixListener::bind(path)
        }
    }
}

fn accept_loop(
    listener: UnixListener,
    tx: mpsc::Sender<(UnixStream, Request)>,
    control: Arc<RunControl>,
    attach: String,
) {
    for connection in listener.incoming() {
        let mut stream = match connection {
            Ok(stream) => stream,
            Err(_) => continue,
        };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
        let request: Request = match read_frame(&mut stream) {
            Ok(request) => request,
            Err(_) => continue,
        };
        if !capability_matches(&attach, &request.capability) {
            let error = ErrorObject::new(
                "session_not_owned",
                "attach capability does not match this supervisor",
                request.request_id.clone(),
                32,
                "use the parent-issued attach token from the inherited channel",
            );
            let _ = write_frame(&mut stream, &Response::error(&request, error));
            continue;
        }
        match request.operation.as_str() {
            "cancel" | "web.cancel" => {
                let session_id = request
                    .payload
                    .get("session_id")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
                    .or_else(|| request.session_id.clone())
                    .unwrap_or_default();
                let target = request
                    .payload
                    .get("target_request_id")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_owned();
                if session_id.is_empty() || target.is_empty() {
                    let error = ErrorObject::new(
                        "protocol_violation",
                        "web.cancel requires session_id and target_request_id",
                        request.request_id.clone(),
                        30,
                        "pass session_id and target_request_id",
                    );
                    let _ = write_frame(&mut stream, &Response::error(&request, error));
                    continue;
                }
                let pair = (session_id, target);
                let completed = control.completed.lock().unwrap_or_else(|e| e.into_inner());
                if completed.contains(&pair) {
                    drop(completed);
                    let _ = write_frame(
                        &mut stream,
                        &Response::ok(
                            &request,
                            json!({ "cancelled": false, "reason": "already_completed" }),
                        ),
                    );
                    continue;
                }
                drop(completed);
                let busy = control.busy.lock().unwrap_or_else(|e| e.into_inner());
                if busy.as_ref() != Some(&pair) {
                    drop(busy);
                    let _ = write_frame(
                        &mut stream,
                        &Response::ok(
                            &request,
                            json!({ "cancelled": false, "reason": "no_match" }),
                        ),
                    );
                    continue;
                }
                drop(busy);
                *control
                    .cancel_target
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(pair);
                let _ = write_frame(
                    &mut stream,
                    &Response::ok(&request, json!({ "cancelled": true })),
                );
            }
            "heartbeat" | "web.heartbeat" => {
                if let Some(session_id) = request
                    .payload
                    .get("session_id")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
                    .or_else(|| request.session_id.clone())
                {
                    control
                        .heartbeats
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(session_id);
                }
                let _ = write_frame(&mut stream, &Response::ok(&request, json!({ "ok": true })));
            }
            "web.session.list" | "session.list" => {
                let mut sessions = control
                    .session_rows
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                let controller_pid = control.controller_pid.load(Ordering::Relaxed);
                let content_pid = control.content_pid.load(Ordering::Relaxed);
                let controller_generation = control.controller_generation.load(Ordering::Relaxed);
                let content_generation = control.content_generation.load(Ordering::Relaxed);
                for row in &mut sessions {
                    if let Some(object) = row.as_object_mut() {
                        object.insert("controller_pid".into(), json!(controller_pid));
                        object.insert("content_pid".into(), json!(content_pid));
                        object.insert("controller_generation".into(), json!(controller_generation));
                        object.insert("content_generation".into(), json!(content_generation));
                    }
                }
                if let Some(agent) = request
                    .payload
                    .get("agent_id")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    sessions.retain(|row| {
                        row.get("owner").and_then(|value| value.as_str()) == Some(agent)
                    });
                }
                let _ = write_frame(
                    &mut stream,
                    &Response::ok(&request, json!({ "sessions": sessions })),
                );
            }
            _ => {
                if tx.send((stream, request)).is_err() {
                    return;
                }
            }
        }
    }
}

struct Daemon {
    socket: PathBuf,
    run_id: String,
    fixture_url: String,
    search_endpoint: Option<String>,
    ever_had_session: bool,
    store: ArtifactStore,
    controller: WorkerProcess,
    content: WorkerProcess,
    sessions: HashMap<String, Session>,
    profile_locks: HashMap<String, crate::profile_lock::ProfileLock>,
    next_engine_id: AtomicU64,
    observed_refs: crate::observed_refs::RefAllocator,
    last_crash: Option<String>,
    crash_receipts: Vec<serde_json::Value>,
    last_request: Instant,
    idle_ttl: Duration,
    exiting: bool,
    attach_capability: String,
    run_control: Arc<RunControl>,
    workflow_deadline: Option<Instant>,
    workflow_defer_observation: bool,
}

impl Daemon {
    fn start(
        config: DaemonConfig,
        attach_capability: String,
        run_control: Arc<RunControl>,
    ) -> io::Result<Self> {
        // Controller (V8) and content (Servo) init independently. Sequential handshake
        // of the 400MB image exceeded the 30s socket wait and panicked wait_for_accepting
        // before bind, leaving in-flight process-group leaders reparented to PID 1.
        let controller_token = random_token()?;
        let content_token = random_token()?;
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase spawn-controller"); } }
        let controller_thread = thread::Builder::new()
            .name("web-spawn-controller".into())
            .spawn(move || {
                let mut worker = WorkerProcess::spawn(WorkerKind::Controller, controller_token)?;
                if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase handshake-controller"); } }
                worker.handshake()?;
                if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase controller-ready"); } }
                Ok::<_, io::Error>(worker)
            })
            .map_err(io::Error::other)?;
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase spawn-content"); } }
        let content_thread = thread::Builder::new()
            .name("web-spawn-content".into())
            .spawn(move || {
                let mut worker = WorkerProcess::spawn(WorkerKind::Content, content_token)?;
                if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase handshake-content"); } }
                worker.handshake()?;
                if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase content-ready"); } }
                Ok::<_, io::Error>(worker)
            })
            .map_err(io::Error::other)?;
        let controller = controller_thread
            .join()
            .map_err(|_| io::Error::other("controller spawn thread panicked"))??;
        let content = content_thread
            .join()
            .map_err(|_| io::Error::other("content spawn thread panicked"))??;
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase workers-ready"); } }
        let data_root = data_root(&config.run_id);
        run_control
            .controller_pid
            .store(controller.pid(), Ordering::Relaxed);
        run_control
            .content_pid
            .store(content.pid(), Ordering::Relaxed);
        Ok(Self {
            socket: config.socket,
            run_id: config.run_id.clone(),
            fixture_url: config.fixture_url.unwrap_or_default(),
            search_endpoint: config.search_endpoint,
            store: ArtifactStore::new(data_root)?,
            ever_had_session: false,
            controller,
            content,
            sessions: HashMap::new(),
            profile_locks: HashMap::new(),
            next_engine_id: AtomicU64::new(1),
            observed_refs: crate::observed_refs::RefAllocator::default(),
            last_crash: None,
            crash_receipts: Vec::new(),
            last_request: Instant::now(),
            idle_ttl: config.idle_ttl,
            exiting: false,
            attach_capability,
            run_control,
            workflow_deadline: None,
            workflow_defer_observation: false,
        })
    }

    fn handle(&mut self, request: Request) -> Response {
        if request.operation != "web.workflow" {
            return self.handle_scoped(request);
        }
        let Some(deadline) = Instant::now().checked_add(Duration::from_millis(request.deadline_ms)) else {
            return protocol_error(&request, "workflow deadline exceeds monotonic clock range");
        };
        let previous = self.workflow_deadline.replace(deadline);
        let response = self.handle_scoped(request);
        self.workflow_deadline = previous;
        response
    }

    fn handle_scoped(&mut self, request: Request) -> Response {
        self.last_request = Instant::now();
        if request.schema != SCHEMA {
            return Response::error(
                &request,
                ErrorObject::new(
                    "protocol_violation",
                    format!("unsupported schema {}", request.schema),
                    request.request_id.clone(),
                    30,
                    "send schema greppy.web-runtime.v1",
                ),
            );
        }
        if request.run_id != self.run_id {
            let mut error = ErrorObject::new(
                "session_not_owned",
                "run_id does not match this supervisor",
                request.request_id.clone(),
                32,
                "create a session under this Greppy run",
            );
            error.session_id = request.session_id.clone();
            return Response::error(&request, error);
        }
        let content_died = !self.content.is_running();
        if request.operation != "web.shutdown" {
            self.reap_idle_sessions();
            // session.close after a timed-out web.run must not spawn replacement
            // workers: handshake of the 400MB image overruns the client read timeout
            // and those process-group leaders reparent to PID 1 when Drop kills us.
            // web.doctor is image/handshake only and must not recover engines.
            if request.operation != "web.session.close"
                && request.operation != "web.doctor"
                && request.operation != "handshake"
                && request.operation != "web.workflow"
            {
                self.ensure_workers();
            }
        }
        let touches_page = request.operation.starts_with("web.")
            && request.operation != "web.status"
            && request.operation != "web.doctor"
            && request.operation != "web.session.create"
            && request.operation != "web.session.close"
            && request.operation != "web.session.list"
            && request.operation != "web.shutdown";
        // A content worker that died before this request was recovered above
        // (pages reset, session kept usable). The call that was actually in
        // flight when the worker died gets the typed worker_restarted error
        // from engine_error below; a later call must not be refused just
        // because a recovery happened earlier (web_run_deadline_is_enforced_externally).
        let _ = content_died;
        // Only the web.run handlers filled the metrics; every other operation
        // answered with zeros, so the paths an agent actually uses -- goto,
        // click, observe -- reported nothing about what they cost. Time the
        // dispatch here and fill in whatever the handler left empty.
        let dispatch_started = Instant::now();
        let content_before = sample_cpu_ms(self.content.pid());
        let controller_before = sample_cpu_ms(self.controller.pid());
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|value| value.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        if let Some(session) = session_id
            .as_ref()
            .and_then(|session_id| self.sessions.get_mut(session_id))
        {
            // Start a new session's CPU accounting before any supervisor
            // preflight. `session.networkBytes` below is real controller work
            // performed for this request and must not sit outside the budget.
            let _ = session_cpu_delta_ms(
                &mut session.content_cpu_baseline,
                self.content.pid(),
            );
            let _ = session_cpu_delta_ms(
                &mut session.controller_cpu_baseline,
                self.controller.pid(),
            );
        }
        // The engine keeps a running total of bytes relayed through the policy
        // proxy. Sampling it around the dispatch turns that into the traffic
        // this one operation caused; the session field it used to report was
        // never incremented, which is why a 60 MB page showed 4096 bytes.
        let bytes_before = (touches_page && self.content.is_running())
            .then(|| {
                self.engine_call_timed("session.networkBytes", json!({}), Duration::from_secs(2))
                    .ok()
            })
            .flatten()
            .and_then(|value| value.get("bytes").and_then(|b| b.as_u64()));
        let limit_request = request.clone();
        let mut response = self.dispatch_operation(request);
        if response.metrics.wall_ms == 0 {
            response.metrics.wall_ms = dispatch_started.elapsed().as_millis() as u64;
        }
        if response.metrics.peak_rss_bytes == 0 {
            response.metrics.peak_rss_bytes = sample_rss_bytes(self.content.pid());
        }
        if response.metrics.content_cpu_ms == 0 {
            response.metrics.content_cpu_ms =
                sample_cpu_ms(self.content.pid()).saturating_sub(content_before);
        }
        if response.metrics.controller_cpu_ms == 0 {
            response.metrics.controller_cpu_ms =
                sample_cpu_ms(self.controller.pid()).saturating_sub(controller_before);
        }
        if response.metrics.network_bytes == 0 {
            if let Some(before) = bytes_before {
                if self.content.is_running() {
                    if let Some(after) = self
                        .engine_call_timed("session.networkBytes", json!({}), Duration::from_secs(2))
                        .ok()
                        .and_then(|value| value.get("bytes").and_then(|b| b.as_u64()))
                    {
                        response.metrics.network_bytes = after.saturating_sub(before);
                    }
                }
            }
        }
        // The pre-dispatch check in `with_session_page` protects every later
        // operation from CPU already consumed by this session. The first
        // operation starts the per-worker baseline, though, so its actual CPU
        // can only be judged here. Enforce the same cumulative session budget
        // before returning success; otherwise a one-shot request can exceed a
        // 1 ms limit while reporting the overage only in metrics.
        if response.status == "ok" {
            if let Some(session_id) = session_id {
                let content_pid = self.content.pid();
                let controller_pid = self.controller.pid();
                let measured_content_cpu_ms = response.metrics.content_cpu_ms;
                let measured_controller_cpu_ms = response.metrics.controller_cpu_ms;
                let cpu_limit_error = self.sessions.get_mut(&session_id).and_then(|session| {
                    let content_cpu = Duration::from_millis(
                        session_cpu_delta_ms(&mut session.content_cpu_baseline, content_pid)
                            .max(measured_content_cpu_ms),
                    );
                    let controller_cpu = Duration::from_millis(
                        session_cpu_delta_ms(&mut session.controller_cpu_baseline, controller_pid)
                            .max(measured_controller_cpu_ms),
                    );
                    let error = session
                        .limits
                        .check_cpu_time(content_cpu, session.limits.content_cpu_time, "content")
                        .and_then(|_| {
                            session.limits.check_cpu_time(
                                controller_cpu,
                                session.limits.controller_cpu_time,
                                "controller",
                            )
                        })
                        .err();
                    if error.is_some() {
                        let _ = session.transition(SessionState::Failed);
                    }
                    error
                });
                if let Some(message) = cpu_limit_error {
                    let metrics = response.metrics.clone();
                    response = limit_error(&limit_request, message);
                    response.metrics = metrics;
                }
            }
        }
        response
    }

    fn dispatch_operation(&mut self, request: Request) -> Response {
        match request.operation.as_str() {
            "handshake" => self.handshake(&request),
            "web.status" => self.status(&request),
            "web.doctor" => self.doctor(&request),
            "web.session.create" => self.session_create(&request),
            "web.session.list" => self.session_list(&request),
            "web.session.close" => self.session_close(&request),
            "web.shutdown" => self.shutdown(&request),
            "web.run" => self.web_run(&request),
            "web.observe" => self.web_observe(&request),
            "web.screenshot" => self.web_screenshot(&request),
            "web.read" => self.web_read(&request),
            "web.search" => self.web_search(&request),
            "web.research" => self.web_research(&request),
            "web.artifacts" => self.web_artifacts(&request),
            "web.result.next" => self.web_result_next(&request),
            "web.artifact.show" => self.web_artifact_show(&request),
            "web.artifact.path" => self.web_artifact_path(&request),
            "web.goto" => self.web_goto(&request),
            "web.back" => self.web_history(&request, "page.goBack", "web.back"),
            "web.forward" => self.web_history(&request, "page.goForward", "web.forward"),
            "web.reload" => self.web_history(&request, "page.reload", "web.reload"),
            "web.evaluate" => self.web_evaluate(&request),
            "web.wait" => self.web_wait(&request),
            "web.workflow" => self.web_workflow(&request),
            "web.tab.new" => self.web_tab(&request, "new"),
            "web.tab.list" => self.web_tab(&request, "list"),
            "web.tab.switch" => self.web_tab(&request, "switch"),
            "web.tab.close" => self.web_tab(&request, "close"),
            "web.console" => self.web_records(&request, "console"),
            "web.network" => self.web_records(&request, "network"),
            "web.events" => self.web_records(&request, "all"),
            "web.click" => self.web_locator_method(&request, "locator.click", json!({})),
            "web.inspect" => self.web_locator_method(&request, "locator.inspect", json!({
                "attrs": request.payload.get("attrs").and_then(|v| v.as_bool()).unwrap_or(false),
                "html": request.payload.get("html").and_then(|v| v.as_bool()).unwrap_or(false),
            })),
            "web.fill" => self.web_fill(&request),
            "web.type" => self.web_type(&request),
            "web.check" => self.web_locator_method(&request, "locator.check", json!({})),
            "web.uncheck" => self.web_locator_method(&request, "locator.uncheck", json!({})),
            "web.select" => self.web_select(&request),
            "web.hover" => self.web_locator_method(&request, "locator.hover", json!({})),
            "web.press" => self.web_press(&request),
            "web.scroll" => self.web_scroll(&request),
            "web.upload" => self.web_upload(&request),
            other => Response::error(
                &request,
                ErrorObject::new(
                    "unsupported_playwright_operation",
                    format!("{other} is not implemented in this runtime build"),
                    request.request_id.clone(),
                    31,
                    "use web.status, web.session.*, or web.run",
                ),
            ),
        }
    }

    fn handshake(&self, request: &Request) -> Response {
        let mut response = Response::ok(
            request,
            serde_json::json!({
                "label": "experimental web-runtime spike",
            }),
        );
        response.handshake = Some(Handshake::runtime_facts());
        response
    }

    fn doctor(&self, request: &Request) -> Response {
        let handshake = Handshake::runtime_facts();
        let executable = std::env::current_exe()
            .ok()
            .map(|path| path.display().to_string());
        Response::ok(
            request,
            json!({
                "executable": executable,
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
            }),
        )
    }

    fn status(&mut self, request: &Request) -> Response {
        let idle = self
            .sessions
            .values()
            .filter(|session| session.state == SessionState::Ready)
            .count();
        let busy = self
            .sessions
            .values()
            .filter(|session| session.state == SessionState::Busy)
            .count();
        let failed = self
            .sessions
            .values()
            .filter(|session| session.state == SessionState::Failed)
            .count();
        let controller_alive = self.controller.is_running();
        let content_alive = self.content.is_running();
        Response::ok(
            request,
            serde_json::json!({
                "label": "experimental web-runtime spike",
                "runtime_version": "0.1.0",
                "runtime_build_id": "web-runtime-0.1.0",
                "playwright_compatibility_version": "1.62.1",
                "compatibility_coverage_level": "unverified",
                "process_health": {
                    "controller_alive": controller_alive,
                    "content_alive": content_alive,
                    "healthy": controller_alive && content_alive,
                },
                "sessions": self.sessions.len(),
                "session_counts": {
                    "total": self.sessions.len(),
                    "idle": idle,
                    "active": busy,
                    "failed": failed,
                },
                "ready": idle,
                "busy": busy,
                "failed": failed,
                "workers": 2,
                "controller_alive": controller_alive,
                "content_alive": content_alive,
                "resource_totals": {
                    "sessions": self.sessions.len(),
                    "workers": 2,
                    "crash_receipts": self.crash_receipts.len(),
                },
                "last_crash": self.last_crash.clone(),
                "crash_receipts": self.crash_receipts.clone(),
                "unsupported_capability_count": 500,
                "conformance_receipt_id": "contracts/web-runtime/receipts/oracle-setcontent.json",
                "engines_linked_into_greppy_parent": false,
                "signed_distributable": false,
                "oracle_receipt": "contracts/web-runtime/receipts/oracle-setcontent.json",
                "oracle_receipts": [
                    "contracts/web-runtime/receipts/oracle-setcontent.json",
                    "contracts/web-runtime/receipts/oracle-dialog.json",
                    "contracts/web-runtime/receipts/oracle-fill.json",
                    "contracts/web-runtime/receipts/oracle-console.json",
                    "contracts/web-runtime/receipts/oracle-content.json"
                ],
                "inventory_entries": 1354,
                "discarded_engine_results": self.run_control.discarded_engine_results.load(Ordering::Relaxed),
                "compatibility_coverage_level_note": "schema implemented is not Chromium oracle behavior; oracle receipts are scoped cases only",
            }),
        )
    }

    fn session_create(&mut self, request: &Request) -> Response {
        let profile = request
            .payload
            .get("profile")
            .and_then(|v| v.as_str())
            .unwrap_or("research");
        if profile != "research" && profile != "project" {
            return Response::error(
                request,
                ErrorObject::new(
                    "protocol_violation",
                    "profile must be research or project",
                    request.request_id.clone(),
                    30,
                    "pass --profile research|project",
                ),
            );
        }
        let parsed = NetworkProfile::parse(profile).expect("validated");
        let id = new_session_id();
        let mut session = Session::new(&id, &self.run_id, parsed);
        session.owner = request_agent_id(request);
        if let Some(limits) = request.payload.get("limits") {
            session.limits.apply_payload(limits);
        }
        if session.transition(SessionState::Ready).is_err() {
            return Response::error(
                request,
                ErrorObject::new(
                    "engine_error",
                    "failed to create session",
                    request.request_id.clone(),
                    38,
                    "retry web.session.create",
                ),
            );
        }
        let profile_lock = match request
            .payload
            .get("persistent_profile")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(name) => match self.acquire_persistent_profile(name) {
                Ok(lock) => {
                    session.persistent_profile = Some(name.to_owned());
                    Some(lock)
                }
                Err(error) => {
                    let code = if error.contains("held by live") {
                        "profile_in_use"
                    } else {
                        "protocol_violation"
                    };
                    let exit = if code == "profile_in_use" { 38 } else { 30 };
                    return Response::error(
                        request,
                        ErrorObject::new(
                            code,
                            error,
                            request.request_id.clone(),
                            exit,
                            "close the other session that owns this persistent profile",
                        ),
                    );
                }
            },
            None => None,
        };
        self.sessions.insert(id.clone(), session);
        if let Some(lock) = profile_lock {
            self.profile_locks.insert(id.clone(), lock);
        }
        self.ever_had_session = true;
        self.run_control
            .publish_sessions(snapshot_session_rows(&self.sessions, &self.run_control));
        self.journal(
            &id,
            &request.request_id,
            "session.ready",
            json!({ "profile": profile }),
        );
        if let Some(created) = self.sessions.get(&id) {
            self.persist_session_snapshot(created, json!({ "event": "session.ready" }));
        }
        Response::ok(
            request,
            json!({
                "session_id": id,
                "profile": profile,
                "state": "ready",
            }),
        )
    }

    fn session_list(&self, request: &Request) -> Response {
        let mut sessions = snapshot_session_rows(&self.sessions, &self.run_control);
        if let Some(agent) = request_agent_id(request) {
            sessions.retain(|row| row.get("owner").and_then(|value| value.as_str()) == Some(agent.as_str()));
        }
        self.run_control.publish_sessions(snapshot_session_rows(&self.sessions, &self.run_control));
        Response::ok(request, serde_json::json!({ "sessions": sessions }))
    }

    fn acquire_persistent_profile(
        &self,
        name: &str,
    ) -> Result<crate::profile_lock::ProfileLock, String> {
        if name.is_empty()
            || name.len() > 64
            || !name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
        {
            return Err(
                "persistent_profile must be a short [A-Za-z0-9_-] name under the run store".into(),
            );
        }
        let dir = self.store.root().join("profiles").join(name);
        crate::profile_lock::ProfileLock::acquire(&dir).map_err(|error| error.to_string())
    }
    fn shutdown(&mut self, request: &Request) -> Response {
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase shutdown-begin"); } }
        self.exiting = true;
        self.sessions.clear();
        self.profile_locks.clear();
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase shutdown-controller-eof"); } }
        self.controller.shutdown_or_kill();
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase shutdown-content-reap"); } }
        self.content.shutdown_or_kill();
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase shutdown-accept-break"); } }
        self.journal(
            "runtime",
            &request.request_id,
            "runtime.shutdown",
            json!({}),
        );
        // Leave no socket behind (finding 040): an orphaned path made the
        // next start fail with a silent spawn cooldown. The sibling .attach
        // token is only meaningful for this runtime, so it goes too.
        remove_runtime_socket_files(&self.socket);
        Response::ok(request, json!({ "shutdown": true }))
    }

    fn drain_late_engine_results(&mut self, session_id: &str, operation_id: &str) {
        let late = self
            .run_control
            .late_engine_result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let Some(late) = late else {
            return;
        };
        if late.session_id != session_id || late.target_request_id != operation_id {
            return;
        }
        self.journal(
            session_id,
            operation_id,
            "late.engine_result",
            json!({
                "discarded": true,
                "kind": "EngineResult",
                "engine_request_id": late.engine_request_id,
                "ok": late.ok,
                "session_id": late.session_id,
                "target_request_id": late.target_request_id,
            }),
        );
    }

    fn session_close(&mut self, request: &Request) -> Response {
        let Some(session_id) = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone())
        else {
            return Response::error(
                request,
                ErrorObject::new(
                    "protocol_violation",
                    "session_id is required",
                    request.request_id.clone(),
                    30,
                    "pass the session id",
                ),
            );
        };
        if let Some(owner) = self
            .sessions
            .get(&session_id)
            .and_then(|session| session.owner.clone())
        {
            let steal = request
                .payload
                .get("steal")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if request_agent_id(request).as_ref() != Some(&owner) && !steal {
                return Response::error(
                    request,
                    ErrorObject::new(
                        "policy_denied",
                        "session is leased to another agent",
                        request.request_id.clone(),
                        36,
                        "pass steal=true to take the lease",
                    ),
                );
            }
        }
        match self.sessions.remove(&session_id) {
            Some(mut session) => {
                let ephemeral = session.persistent_profile.is_none();
                let _ = self.profile_locks.remove(&session_id);
                let _ = session.transition(SessionState::Closing);
                if let Some(page) = session.page_id.take() {
                    if self.content.is_running() {
                        let _ = self.engine_call("page.close", json!({ "page": page }));
                    }
                }
                let _ = session.transition(SessionState::Closed);
                self.run_control
                    .publish_sessions(snapshot_session_rows(&self.sessions, &self.run_control));
                self.journal(
                    &session_id,
                    &request.request_id,
                    "session.closed",
                    json!({}),
                );
                if ephemeral {
                    self.remove_ephemeral_session_dir(&session_id);
                }
                Response::ok(
                    request,
                    serde_json::json!({ "session_id": session_id, "state": "closed" }),
                )
            }
            None => {
                let mut error = ErrorObject::new(
                    "session_not_found",
                    format!("session {session_id} was not found"),
                    request.request_id.clone(),
                    32,
                    "create a session first",
                );
                error.session_id = Some(session_id);
                Response::error(request, error)
            }
        }
    }

    fn web_run(&mut self, request: &Request) -> Response {
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return Response::error(
                request,
                ErrorObject::new(
                    "protocol_violation",
                    "web.run requires session_id",
                    request.request_id.clone(),
                    30,
                    "create a session and pass --session",
                ),
            );
        };
        if !self.sessions.contains_key(&session_id) {
            let mut error = ErrorObject::new(
                "session_not_found",
                format!("session {session_id} was not found"),
                request.request_id.clone(),
                32,
                "create a session first",
            );
            error.session_id = Some(session_id);
            return Response::error(request, error);
        }
        if !self.controller.is_running() {
            if let Err(error) = self.recover_controller("controller worker exited before web.run") {
                return engine_error(request, error, 33);
            }
        }
        if !self.content.is_running() {
            if let Err(error) = self.recover_content("content worker exited before web.run") {
                return engine_error(request, error, 33);
            }
        }
        if let Some(session) = self.sessions.get_mut(&session_id) {
            if let Err(message) = session.begin_operation(&request.request_id) {
                return Response::error(
                    request,
                    ErrorObject::new(
                        "engine_error",
                        message,
                        request.request_id.clone(),
                        38,
                        "wait for the session to become ready",
                    ),
                );
            }
        }
        *self
            .run_control
            .busy
            .lock()
            .unwrap_or_else(|e| e.into_inner()) =
            Some((session_id.clone(), request.request_id.clone()));
        self.run_control
            .controller_pid
            .store(self.controller.pid(), Ordering::Relaxed);
        self.run_control
            .content_pid
            .store(self.content.pid(), Ordering::Relaxed);
        self.run_control
            .publish_sessions(snapshot_session_rows(&self.sessions, &self.run_control));
        self.journal(
            &session_id,
            &request.request_id,
            "run.started",
            json!({ "operation": "web.run" }),
        );
        let source = request
            .payload
            .get("script_text")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let file = request
            .payload
            .get("script_file")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let (specifier, source) = match (file, source) {
            (Some(path), maybe_text) => {
                let text = match maybe_text {
                    Some(text) => text,
                    None => match std::fs::read_to_string(&path) {
                        Ok(text) => text,
                        Err(error) => {
                            self.finish_session(&session_id);
                            return Response::error(
                                request,
                                ErrorObject::new(
                                    "protocol_violation",
                                    format!("cannot read script file: {error}"),
                                    request.request_id.clone(),
                                    30,
                                    "pass a readable --script-file",
                                ),
                            );
                        }
                    },
                };
                match stage_script_for_controller(
                    &path,
                    &self.run_id,
                    &session_id,
                    &request.request_id,
                ) {
                    Ok(staged) => (staged, text),
                    Err(error) => {
                        self.finish_session(&session_id);
                        return Response::error(
                            request,
                            ErrorObject::new(
                                "protocol_violation",
                                error,
                                request.request_id.clone(),
                                30,
                                "pass a readable --script-file inside a bounded script root",
                            ),
                        );
                    }
                }
            }
            (None, Some(text)) => ("greppy:stdin".to_owned(), text),
            (None, None) => {
                self.finish_session(&session_id);
                return Response::error(
                    request,
                    ErrorObject::new(
                        "protocol_violation",
                        "web.run requires script_text or script_file",
                        request.request_id.clone(),
                        30,
                        "use --script-file or --script-stdin",
                    ),
                );
            }
        };
        let _stage_guard = if specifier != "greppy:stdin" {
            Some(ScriptStageGuard {
                run_id: self.run_id.clone(),
                session_id: session_id.clone(),
                request_id: request.request_id.clone(),
            })
        } else {
            None
        };
        if !self.controller.is_running() {
            if let Err(error) = self.recover_controller("controller worker exited") {
                self.finish_session(&session_id);
                return engine_error(request, error, 33);
            }
        }
        let profile = self
            .sessions
            .get(&session_id)
            .map(|session| session.profile)
            .unwrap_or(NetworkProfile::Research);
        let started = Instant::now();
        let run_budget = Duration::from_millis(request.deadline_ms.max(1_000));
        let run_deadline = started + run_budget;
        if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase run-wait point=set-profile worker=content session={} deadline_ms={}",
            session_id,
            run_budget.as_millis()
        ); }
        if let Err(error) = self.engine_call_timed(
            "session.setProfile",
            json!({ "profile": profile.as_str() }),
            run_deadline.saturating_duration_since(Instant::now()),
        ) {
            self.finish_session(&session_id);
            if error.contains("timed out") {
                let mut object = ErrorObject::new(
                    "timeout",
                    redact_secrets(&error),
                    request.request_id.clone(),
                    35,
                    "retry with a longer deadline or a smaller script",
                );
                object.session_id = Some(session_id);
                return Response::error(request, object);
            }
            return engine_error(request, error, 34);
        }
        let content_pid = self.content.pid();
        let controller_pid = self.controller.pid();
        let content_cpu_baseline_ns = sample_cpu_ns(content_pid);
        let controller_cpu_baseline_ns = sample_cpu_ns(controller_pid);
        let control = Arc::clone(&self.run_control);
        let operation_id = request.request_id.clone();
        let remaining = run_deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(1));
        if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase run-wait point=send-script worker=controller remaining_ms={}",
            remaining.as_millis()
        ); }
        let outcome = {
            let controller = &mut self.controller;
            let content = &mut self.content;
            let sessions = &mut self.sessions;
            let session_key = session_id.clone();
            if let Err(error) = controller.send_timeout(
                &crate::protocol::Message::run_script(
                    specifier.clone(),
                    source,
                    self.fixture_url.clone(),
                ),
                remaining,
            ) {
                Err(error)
            } else {
                if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase run-wait point=script-complete remaining_ms={}",
                    remaining.as_millis()
                ); }
                crate::supervisor::route_until_script_complete_gated(
                    controller,
                    content,
                    remaining,
                    SessionEngineGate {
                        sessions,
                        session_id: session_key,
                        content_pid,
                        controller_pid,
                        content_cpu_baseline_ns,
                        controller_cpu_baseline_ns,
                        operation_id: operation_id.clone(),
                        control: Arc::clone(&control),
                    },
                )
            }
        };
        let (network_bytes, peak_rss) = self
            .sessions
            .get(&session_id)
            .map(|session| (session.network_bytes, session.peak_rss_bytes))
            .unwrap_or((0, 0));
        // Prefer the proxy's real relay counter over the session's fixed
        // 4096-per-navigation accounting stub. Skip the ask when the run
        // timed out or was cancelled: the content worker is hung or about to
        // be killed, and even the short bound below would stall the reply.
        let run_settled = match &outcome {
            Ok(_) => true,
            Err(error) => {
                let message = error.to_string();
                error.kind() != io::ErrorKind::TimedOut
                    && !message.contains("timed out")
                    && !message.contains("cancelled")
            }
        };
        let network_bytes = if run_settled {
            self.engine_call_timed("session.networkBytes", json!({}), Duration::from_secs(2))
                .ok()
                .and_then(|value| value.get("bytes").and_then(|b| b.as_u64()))
                .unwrap_or(network_bytes)
        } else {
            network_bytes
        };
        let run_event = match &outcome {
            Ok(_) => "run.completed",
            Err(error)
                if error.kind() == io::ErrorKind::TimedOut
                    || error.to_string().contains("timed out") =>
            {
                "run.timeout"
            }
            Err(error) if error.to_string().contains("cancelled") => "run.cancelled",
            Err(_) => "run.failed",
        };
        self.journal(&session_id, &request.request_id, run_event, json!({}));
        remove_script_stage(&self.run_id, &session_id, Some(&request.request_id));
        let had_inflight_engine = self
            .sessions
            .get(&session_id)
            .and_then(|session| session.inflight_engine_request_id);
        self.finish_session(&session_id);
        match outcome {
            Ok(result) => {
                let stdout = result
                    .get("stdout")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let mut response = Response::ok(
                    request,
                    serde_json::json!({
                        "session_id": session_id,
                        "completed": true,
                        "stdout": stdout,
                    }),
                );
                response.metrics.wall_ms = started.elapsed().as_millis() as u64;
                response.metrics.network_bytes = network_bytes;
                response.metrics.peak_rss_bytes = peak_rss.max(sample_rss_bytes(content_pid));
                response.metrics.content_cpu_ms =
                    cpu_ms_since(content_pid, content_cpu_baseline_ns);
                response.metrics.controller_cpu_ms =
                    cpu_ms_since(controller_pid, controller_cpu_baseline_ns);
                response
            }
            Err(error) => {
                let message = error.to_string();
                if message.contains("cancelled") {
                    let pair = (session_id.clone(), operation_id.clone());
                    control
                        .completed
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(pair.clone());
                    {
                        let mut cancel_target = control
                            .cancel_target
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        if cancel_target.as_ref() == Some(&pair) {
                            *cancel_target = None;
                        }
                    }
                    {
                        let mut busy = control.busy.lock().unwrap_or_else(|e| e.into_inner());
                        if busy.as_ref() == Some(&pair) {
                            *busy = None;
                        }
                    }
                    self.drain_late_engine_results(&session_id, &operation_id);
                    if had_inflight_engine.is_some() {
                        match self.respawn_content_after_cancel("cancelled content isolate") {
                            Ok((pid_before, pid_after, generation)) => {
                                self.journal(
                                    &session_id,
                                    &operation_id,
                                    "worker.respawn",
                                    json!({
                                        "worker": "content",
                                        "pid_before": pid_before,
                                        "pid_after": pid_after,
                                        "generation": generation,
                                    }),
                                );
                            }
                            Err(error) => {
                                let mut object = ErrorObject::new(
                                    "worker_unavailable",
                                    error,
                                    request.request_id.clone(),
                                    38,
                                    "retry greppy web run",
                                );
                                object.session_id = Some(session_id.clone());
                                return Response::error(request, object);
                            }
                        }
                    }
                    let controller_pid_before = self.controller.pid();
                    if let Err(error) = self.recover_controller("cancelled controller isolate") {
                        let mut object = ErrorObject::new(
                            "worker_unavailable",
                            error,
                            request.request_id.clone(),
                            38,
                            "retry greppy web run",
                        );
                        object.session_id = Some(session_id.clone());
                        return Response::error(request, object);
                    }
                    self.journal(
                        &session_id,
                        &operation_id,
                        "worker.respawn",
                        json!({
                            "worker": "controller",
                            "pid_before": controller_pid_before,
                            "pid_after": self.controller.pid(),
                            "generation": self
                                .run_control
                                .controller_generation
                                .load(Ordering::Relaxed),
                        }),
                    );
                    self.run_control
                        .publish_sessions(snapshot_session_rows(&self.sessions, &self.run_control));
                    let mut object = ErrorObject::new(
                        "cancelled",
                        "web.run was cancelled",
                        request.request_id.clone(),
                        35,
                        "retry the script or send a new web.run",
                    );
                    object.session_id = Some(session_id.clone());
                    return Response::error(request, object);
                }
                if error.kind() == io::ErrorKind::TimedOut || message.contains("timed out") {
                    let content_cpu_ms = cpu_ms_since(content_pid, content_cpu_baseline_ns);
                    let controller_cpu_ms =
                        cpu_ms_since(controller_pid, controller_cpu_baseline_ns);
                    self.controller.kill_tree_now();
                    self.content.kill_tree_now();
                    // Do not recover here. Spawn+handshake of replacement workers can
                    // exceed the client unix read slack; Drop then SIGKILLs the supervisor
                    // while those process-group leaders reparent to PID 1.
                    let mut object = ErrorObject::new(
                        "timeout",
                        redact_secrets(&message),
                        request.request_id.clone(),
                        35,
                        "retry with a longer deadline or a smaller script",
                    );
                    object.session_id = Some(session_id.clone());
                    let mut response = Response::error(request, object);
                    if let Ok(manifest) = self.store.put(
                        format!(
                            "{{\"partial\":true,\"reason\":\"timeout\",\"session_id\":\"{session_id}\"}}"
                        )
                        .as_bytes(),
                        "application/json",
                        &session_id,
                        &self.run_id,
                        "web.run.timeout",
                        false,
                    ) {
                        if let Ok(value) = serde_json::to_value(&manifest) {
                            response.artifacts.push(value);
                        }
                    }
                    response.metrics.wall_ms = started.elapsed().as_millis() as u64;
                    response.metrics.network_bytes = network_bytes;
                    response.metrics.peak_rss_bytes = peak_rss;
                    response.metrics.content_cpu_ms = content_cpu_ms;
                    response.metrics.controller_cpu_ms = controller_cpu_ms;
                    return response;
                }
                if let Some(limit) = message.strip_prefix("resource_limit: ") {
                    let mut response = limit_error(request, limit);
                    if let Some(error) = response.error.as_mut() {
                        error.session_id = Some(session_id);
                    }
                    response.metrics.wall_ms = started.elapsed().as_millis() as u64;
                    response.metrics.network_bytes = network_bytes;
                    response.metrics.peak_rss_bytes = peak_rss;
                    response.metrics.content_cpu_ms =
                        cpu_ms_since(content_pid, content_cpu_baseline_ns);
                    response.metrics.controller_cpu_ms =
                        cpu_ms_since(controller_pid, controller_cpu_baseline_ns);
                    return response;
                }
                let safe: String = redact_secrets(&message).chars().take(512).collect();
                let mut object = ErrorObject::new(
                    "controller_exception",
                    safe,
                    request.request_id.clone(),
                    33,
                    "inspect the controller script and retry",
                );
                object.session_id = Some(session_id);
                // A script that throws is still a script that ran. Without
                // these the only path a harness can read a result on reports
                // zero CPU and zero bytes, which is how `content_cpu_ms: 0`
                // came to look like a broken counter rather than a missing
                // assignment.
                let mut response = Response::error(request, object);
                response.metrics.wall_ms = started.elapsed().as_millis() as u64;
                response.metrics.network_bytes = network_bytes;
                response.metrics.peak_rss_bytes = peak_rss.max(sample_rss_bytes(content_pid));
                response.metrics.content_cpu_ms =
                    cpu_ms_since(content_pid, content_cpu_baseline_ns);
                response.metrics.controller_cpu_ms =
                    cpu_ms_since(controller_pid, controller_cpu_baseline_ns);
                response
            }
        }
    }

    fn finish_session(&mut self, session_id: &str) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            let _ = session.transition(SessionState::Ready);
        }
    }

    /// Observe without beginning/finishing another operation. Action callers
    /// use this while their original operation is still Busy.
    fn observe_page(&mut self, session_id: &str, page: &str) -> Result<serde_json::Value, String> {
        self.observe_page_bounded(session_id, page, Duration::from_secs(60), true)
    }

    fn observe_page_bounded(
        &mut self,
        session_id: &str,
        page: &str,
        budget: Duration,
        recover_worker: bool,
    ) -> Result<serde_json::Value, String> {
        self.observe_page_scoped(session_id, page, budget, recover_worker, None, false)
    }

    fn observe_page_scoped(
        &mut self, session_id: &str, page: &str, budget: Duration,
        recover_worker: bool, query: Option<&str>, include_html: bool,
    ) -> Result<serde_json::Value, String> {
        let session = self.sessions.get_mut(session_id).ok_or("observation session no longer exists")?;
        let elapsed = session.started.elapsed();
        session.limits.check_wall_time(elapsed)?;
        let content_cpu = Duration::from_millis(session_cpu_delta_ms(
            &mut session.content_cpu_baseline, self.content.pid(),
        ));
        session.limits.check_cpu_time(content_cpu, session.limits.content_cpu_time, "content")?;
        let timeout = session.limits.wall_time.saturating_sub(elapsed).min(budget);
        if timeout.is_zero() {
            return Err("observation has no remaining request/session budget".into());
        }
        let range = self.observed_refs.reserve().map_err(str::to_owned)?;
        let proposed = format!(
            "gref-{}-{}",
            std::process::id(),
            self.next_engine_id.fetch_add(1, Ordering::Relaxed)
        );
        // Keep the last confirmed scope on observation failure. Otherwise a
        // transient page error would strand the worker's still-live registry
        // and make every subsequent observation reject its document token.
        let previous = self.sessions.get(session_id)
            .and_then(|session| session.locator_snapshots.get(page)).cloned();
        let mut tree = self.engine_call_timed_with_recovery("page.observe", json!({
            "page": page, "snapshot": proposed,
            "ref_first": range.first, "ref_last": range.last,
            "query": query, "include_html": include_html,
        }), timeout, recover_worker)?;
        let object = tree.as_object_mut().ok_or("observe returned no page object")?;
        let token = object.remove("ref_snapshot")
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or("observe returned no document scope")?;
        // The worker can retain an unchanged document's token, but page data
        // must never invent a CSS selector scope or another tab's token.
        let retained = previous.as_ref().filter(|previous| previous.token == token);
        if token != proposed && retained.is_none() {
            return Err("observe returned an unrecognized document scope".into());
        }
        let actionables = object.get("actionables").and_then(|value| value.as_array())
            .ok_or("observe returned no actionable list")?;
        if actionables.len() > crate::observed_refs::OBSERVED_REF_LIMIT as usize {
            return Err("observe exceeded the actionable limit".into());
        }
        for actionable in actionables {
            let reference = actionable.get("ref").and_then(|value| value.as_str())
                .and_then(|value| value.strip_prefix('@'))
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or("observe returned an invalid reference")?;
            if !range.contains(reference)
                && !retained.is_some_and(|previous| reference > 0 && reference <= previous.ref_ceiling)
            {
                return Err("observe returned an unallocated reference".into());
            }
        }
        if let Some(url) = object.get("url").and_then(|value| value.as_str()) {
            let redacted = redact_secrets(url);
            object.insert("url".into(), json!(redacted));
        }
        object.insert("untrusted_content_boundary".into(), json!("UNTRUSTED_PAGE_CONTENT"));
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.locator_snapshots.insert(page.to_owned(), LocatorSnapshot {
                token, page_id: page.to_owned(), ref_ceiling: range.last,
            });
        }
        Ok(tree)
    }

    fn finish_action_with_page_state(
        &mut self,
        request: &Request,
        session_id: &str,
        page: &str,
        mut result: serde_json::Value,
    ) -> Response {
        // Observe exactly once, before the original operation becomes idle.
        // Observation failure cannot replay or erase a completed side effect.
        let state = if self.workflow_defer_observation {
            None
        } else {
            Some(page_state_envelope(self.observe_page(session_id, page)))
        };
        if let Some(object) = result.as_object_mut() {
            // Identity comes from the resolved native target, including an
            // implicit active tab, not from an optional CLI request field.
            object.insert("session_id".into(), json!(session_id));
            object.insert("tab_id".into(), json!(page));
            if let Some(state) = state {
                object.insert("page_state".into(), state);
            }
        }
        self.finish_session(session_id);
        Response::ok(request, result)
    }

    fn finish_failed_action_with_page_state(
        &mut self,
        request: &Request,
        session_id: &str,
        page: &str,
        mut response: Response,
        started: Instant,
    ) -> Response {
        // A read-only, best-effort observation is not an action retry. Keep the
        // original error even if observation fails; never restart workers just
        // to decorate an error, or grant a fresh operation/session budget.
        let observable = response.error.as_ref().is_some_and(|error| {
            matches!(error.code.as_str(), "NO_MATCH" | "AMBIGUOUS_TARGET" | "STALE_REF" | "TIMEOUT")
        });
        if observable {
            let budget = failure_observation_budget(request.deadline_ms, started.elapsed());
            let state = page_state_envelope(
                self.observe_page_bounded(session_id, page, budget, false),
            );
            if let Some(error) = response.error.as_mut() {
                if let Some(next) = recovery_with_observed_state(
                    &error.code, state["status"].as_str() == Some("available"),
                ) {
                    error.next_action = next.to_owned();
                }
            }
            response.result = Some(json!({
                "session_id": session_id,
                "tab_id": page,
                "page_state": state,
                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
            }));
        }
        self.finish_session(session_id);
        response
    }

    fn web_observe(&mut self, request: &Request) -> Response {
        let query = match request.payload.get("query") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(value)) if !value.trim().is_empty() => Some(value.as_str()),
            _ => return protocol_error(request, "observe query must be a nonempty string"),
        };
        let format = request.payload.get("format").and_then(|value| value.as_str()).unwrap_or("agent-tree");
        match self.with_session_page(request, "web.observe") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.observe_page_scoped(&session_id, &page, Duration::from_secs(60), true, query, format == "html") {
                    Ok(tree) => {
                        if let Some(query) = query {
                            if greppy_web_client::observation_scope_roots(&tree, query).is_err() {
                                self.finish_session(&session_id);
                                return engine_error(request, "scoped observation returned no matching scope evidence; refusing an unfiltered result", 34);
                            }
                        }
                        if query.is_some() && tree.pointer("/observation_scope/roots_returned").and_then(|value| value.as_u64()) == Some(0) {
                            self.finish_session(&session_id);
                            let mut response = Response::error(request, ErrorObject::new(
                                "NO_MATCH", "observation query matched no visible region", request.request_id.clone(), 32,
                                "inspect the query or open the intended region; use unfiltered observe only when the whole page is intended",
                            ));
                            response.result = Some(tree);
                            return response;
                        }
                        let format = request
                            .payload
                            .get("format")
                            .and_then(|value| value.as_str())
                            .unwrap_or("agent-tree");
                        match format {
                            "text" => {
                                let text = tree
                                    .get("text")
                                    .and_then(|value| value.as_str())
                                    .or_else(|| tree.get("title").and_then(|value| value.as_str()))
                                    .unwrap_or("")
                                    .to_owned();
                                let stored = self.store_bytes(
                                    request,
                                    &session_id,
                                    text.as_bytes(),
                                    "text/plain",
                                    "web.observe",
                                    false,
                                );
                                self.finish_session(&session_id);
                                match stored {
                                    Ok(manifest) => match model_facing_observe_payload(
                                        "text", "text", &text, &manifest,
                                    ) {
                                        Ok(mut payload) => {
                                            if let Some(scope) = tree.get("observation_scope") {
                                                payload["observation_scope"] = scope.clone();
                                            }
                                            Response::ok(request, payload)
                                        }
                                        Err(error) => engine_error(request, error, 39),
                                    },
                                    Err(response) => response,
                                }
                            }
                            "html" => {
                                let content = if query.is_some() {
                                    tree.get("scoped_html").and_then(|value| value.as_str())
                                        .map(|html| json!({"html":html}))
                                        .ok_or_else(|| "scoped observation returned no HTML".to_owned())
                                } else {
                                    self.engine_call("page.content", json!({ "page": page }))
                                };
                                match content {
                                    Ok(value) => match Self::html_from_page_content(&value) {
                                        Ok(html) => {
                                            let stored = self.store_bytes(
                                                request,
                                                &session_id,
                                                html.as_bytes(),
                                                "text/html",
                                                "web.observe",
                                                false,
                                            );
                                            self.finish_session(&session_id);
                                            match stored {
                                                Ok(manifest) => match model_facing_observe_payload(
                                                    "html", "html", &html, &manifest,
                                                ) {
                                                    Ok(mut payload) => {
                                                        if let Some(scope) = tree.get("observation_scope") {
                                                            payload["observation_scope"] = scope.clone();
                                                        }
                                                        Response::ok(request, payload)
                                                    }
                                                    Err(error) => engine_error(request, error, 39),
                                                },
                                                Err(response) => response,
                                            }
                                        }
                                        Err(error) => {
                                            self.finish_session(&session_id);
                                            engine_error(request, error, 34)
                                        }
                                    },
                                    Err(error) => {
                                        self.finish_session(&session_id);
                                        engine_error(request, error, 34)
                                    }
                                }
                            }
                            _ => {
                                self.finish_session(&session_id);
                                Response::ok(request, tree)
                            }
                        }
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        engine_error(request, error, 34)
                    }
                }
            }
        }
    }

    fn web_screenshot(&mut self, request: &Request) -> Response {
        match self.with_session_page(request, "web.screenshot") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.engine_call("page.screenshot", json!({ "page": page })) {
                    Ok(result) => {
                        let bytes = match screenshot_png_bytes(&result) {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                self.finish_session(&session_id);
                                return engine_error(request, error, 34);
                            }
                        };
                        let stored = self.store_bytes(
                            request,
                            &session_id,
                            &bytes,
                            "image/png",
                            "web.screenshot",
                            true,
                        );
                        self.finish_session(&session_id);
                        match stored {
                            Ok(manifest) => {
                                let mut payload = json!({
                                    "session_id": session_id,
                                    "digest": manifest.digest.hex,
                                    "byte_count": manifest.byte_count,
                                    "object_path": manifest.object_path,
                                    "media_type": "image/png",
                                });
                                if let Some(b64) = result.get("png_base64").cloned() {
                                    payload["png_base64"] = b64;
                                } else {
                                    payload["truncated"] = json!(true);
                                    payload["cursor"] =
                                        json!(format!("sha256:{}:0", manifest.digest.hex));
                                }
                                let mut response = Response::ok(request, payload);
                                response
                                    .artifacts
                                    .push(serde_json::to_value(manifest).unwrap_or(json!({})));
                                response
                            }
                            Err(response) => response,
                        }
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        engine_error(request, error, 34)
                    }
                }
            }
        }
    }

    fn web_read(&mut self, request: &Request) -> Response {
        let Some(url) = request.payload.get("url").and_then(|v| v.as_str()) else {
            return protocol_error(request, "web.read requires url");
        };
        match self.with_session_page(request, "web.read") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.navigate_and_extract(&session_id, &page, url, request) {
                    Ok(source) => {
                        let mut response = Response::ok(
                            request,
                            json!({
                                "session_id": session_id,
                                "source": source,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        );
                        self.attach_session_metrics(&session_id, &mut response);
                        self.finish_session(&session_id);
                        response
                    }
                    Err(response) => {
                        self.finish_session(&session_id);
                        response
                    }
                }
            }
        }
    }

    pub(crate) fn host_matches_domain(href: &str, domain: &str) -> bool {
        let domain = domain.trim().trim_start_matches('.').to_ascii_lowercase();
        if domain.is_empty() {
            return true;
        }
        let host = href
            .split("://")
            .nth(1)
            .unwrap_or(href)
            .split('/')
            .next()
            .unwrap_or("")
            .rsplit('@')
            .next()
            .unwrap_or("")
            .split(':')
            .next()
            .unwrap_or("")
            .trim_end_matches('.')
            .to_ascii_lowercase();
        host == domain || host.ends_with(&format!(".{domain}"))
    }

    pub(crate) fn html_from_page_content(value: &serde_json::Value) -> Result<String, String> {
        value
            .get("html")
            .and_then(|html| html.as_str())
            .map(str::to_owned)
            .ok_or_else(|| "page.content missing html".to_owned())
    }
    fn web_search(&mut self, request: &Request) -> Response {
        let Some(query) = request.payload.get("query").and_then(|v| v.as_str()) else {
            return protocol_error(request, "web.search requires query");
        };
        let limit = request
            .payload
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .max(1) as usize;
        match self.with_session_page(request, "web.search") {
            Err(response) => response,
            Ok((session_id, page)) => {
                let search_url = self.search_url(query);
                match self.navigate_and_extract(&session_id, &page, &search_url, request) {
                    Ok(mut source) => {
                        let links = self
                            .engine_call("page.observe", json!({ "page": page }))
                            .ok()
                            .and_then(|tree| tree.get("links").cloned())
                            .and_then(|value| value.as_array().cloned())
                            .unwrap_or_default();
                        let domain = request
                            .payload
                            .get("domain")
                            .and_then(|value| value.as_str())
                            .map(str::trim)
                            .filter(|value| !value.is_empty());
                        let results: Vec<_> = links
                            .into_iter()
                            .filter(|link| match domain {
                                Some(domain) => link
                                    .get("href")
                                    .and_then(|value| value.as_str())
                                    .is_some_and(|href| Self::host_matches_domain(href, domain)),
                                None => true,
                            })
                            .take(limit)
                            .collect();
                        source["classification"] = json!("aggregator");
                        self.finish_session(&session_id);
                        Response::ok(
                            request,
                            json!({
                                "query": query,
                                "results": results,
                                "source": source,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(response) => {
                        self.finish_session(&session_id);
                        response
                    }
                }
            }
        }
    }

    fn web_research(&mut self, request: &Request) -> Response {
        let Some(query) = request.payload.get("query").and_then(|v| v.as_str()) else {
            return protocol_error(request, "web.research requires query");
        };
        let depth = request
            .payload
            .get("depth")
            .and_then(|v| v.as_str())
            .unwrap_or("standard");
        let requested = request
            .payload
            .get("max_sources")
            .and_then(|v| v.as_u64())
            .unwrap_or(3)
            .clamp(1, 8) as usize;
        let max_sources = match depth {
            "shallow" => 1,
            "deep" => requested.max(6).min(8),
            _ => requested,
        };
        let search = self.web_search(request);
        if search.status != "ok" {
            return search;
        }
        let results = search
            .result
            .as_ref()
            .and_then(|value| value.get("results"))
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        let mut admitted = Vec::new();
        let mut omitted = 0u32;
        let mut omitted_reasons = Vec::new();
        for result in results.into_iter().take(max_sources) {
            let Some(href) = result.get("href").and_then(|v| v.as_str()) else {
                omitted += 1;
                omitted_reasons.push(json!({"reason": "missing href"}));
                continue;
            };
            let mut read_req = request.clone();
            read_req.payload =
                json!({ "url": href, "session_id": request.payload.get("session_id") });
            match self.web_read(&read_req) {
                response if response.status == "ok" => {
                    if let Some(source) = response
                        .result
                        .and_then(|value| value.get("source").cloned())
                    {
                        admitted.push(source);
                    } else {
                        omitted += 1;
                        omitted_reasons.push(json!({
                            "url": href,
                            "reason": "read returned no source",
                        }));
                    }
                }
                response => {
                    omitted += 1;
                    omitted_reasons.push(json!({
                        "url": href,
                        "status": response.status,
                        "error": response.error,
                    }));
                }
            }
        }
        let snippets: Vec<_> = admitted
            .iter()
            .map(|source| {
                json!({
                    "url": source.get("final_url"),
                    "title": source.get("title"),
                    "snippet": source.get("text").and_then(|v| v.as_str()).unwrap_or("").chars().take(280).collect::<String>(),
                    "digest": source.get("digest"),
                })
            })
            .collect();
        let continuation = if omitted > 0 {
            json!(format!("offset={}", admitted.len()))
        } else {
            serde_json::Value::Null
        };
        Response::ok(
            request,
            json!({
                "query_summary": query,
                "admitted_sources": admitted.len(),
                "omitted": omitted,
                "omitted_reasons": omitted_reasons,
                "evidence": snippets,
                "sources": admitted,
                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                "continuation_token": continuation,
            }),
        )
    }

    fn web_artifacts(&mut self, request: &Request) -> Response {
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return protocol_error(request, "web.artifacts requires session_id");
        };
        if !self.sessions.contains_key(&session_id) {
            return missing_session(request, &session_id);
        }
        match self.store.list_session(&session_id) {
            Ok(list) => Response::ok(
                request,
                json!({ "session_id": session_id, "artifacts": list }),
            ),
            Err(error) => engine_error(request, error.to_string(), 39),
        }
    }

    fn web_result_next(&mut self, request: &Request) -> Response {
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return protocol_error(request, "web.result.next requires session_id");
        };
        if !self.sessions.contains_key(&session_id) {
            return missing_session(request, &session_id);
        }
        let Some(cursor) = request.payload.get("cursor").and_then(|v| v.as_str()) else {
            return protocol_error(request, "web.result.next requires cursor");
        };
        let (digest, offset) = match parse_result_cursor(cursor) {
            Ok(parsed) => parsed,
            Err(error) => return protocol_error(request, &error),
        };
        let listed = match self.store.list_session(&session_id) {
            Ok(list) => list,
            Err(error) => return engine_error(request, error.to_string(), 39),
        };
        let Some(manifest) = listed
            .iter()
            .find(|row| row.digest.hex.eq_ignore_ascii_case(&digest))
        else {
            return protocol_error(
                request,
                "cursor does not refer to an artifact of this session",
            );
        };
        if manifest.is_restricted() {
            return match manifest.model_facing_ref() {
                Ok(facing) => Response::ok(
                    request,
                    json!({
                        "session_id": session_id,
                        "truncated": false,
                        "cursor": serde_json::Value::Null,
                        "digest": manifest.digest.hex,
                        "byte_count": 0,
                        "offset": offset,
                        "media_type": manifest.media_type,
                        "artifact": facing,
                        "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                    }),
                ),
                Err(error) => engine_error(request, error, 39),
            };
        }
        let bytes = match self.store.read_object(&digest) {
            Ok(bytes) => bytes,
            Err(error) => return engine_error(request, error.to_string(), 39),
        };
        if offset > bytes.len() {
            return protocol_error(request, "cursor offset is past the artifact");
        }
        let chunk = &bytes[offset..bytes.len().min(offset + RESULT_NEXT_CHUNK)];
        let next_offset = offset + chunk.len();
        let truncated = next_offset < bytes.len();
        let next_cursor = if truncated {
            json!(format!("sha256:{digest}:{next_offset}"))
        } else {
            serde_json::Value::Null
        };
        Response::ok(
            request,
            json!({
                "session_id": session_id,
                "truncated": truncated,
                "cursor": next_cursor,
                "digest": manifest.digest.hex,
                "byte_count": chunk.len(),
                "offset": offset,
                "media_type": manifest.media_type,
                "bytes_base64": encode_base64(chunk),
                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
            }),
        )
    }

    fn web_artifact_show(&mut self, request: &Request) -> Response {
        match self.find_session_artifact(request) {
            Err(response) => response,
            Ok((session_id, manifest)) => {
                let path = self
                    .store
                    .root()
                    .join(&manifest.object_path)
                    .display()
                    .to_string();
                if manifest.is_restricted() {
                    return match manifest.model_facing_ref() {
                        Ok(mut facing) => {
                            if let Some(object) = facing.as_object_mut() {
                                object.insert("session_id".into(), json!(session_id));
                                object.insert("absolute_path".into(), json!(path));
                                object.insert("byte_count".into(), json!(manifest.byte_count));
                                object.insert("media_type".into(), json!(manifest.media_type));
                            }
                            Response::ok(request, facing)
                        }
                        Err(error) => engine_error(request, error, 39),
                    };
                }
                Response::ok(
                    request,
                    json!({
                        "session_id": session_id,
                        "digest": manifest.digest.hex,
                        "byte_count": manifest.byte_count,
                        "media_type": manifest.media_type,
                        "producing_operation": manifest.producing_operation,
                        "path": path,
                        "object_path": manifest.object_path,
                        "redaction_status": manifest.redaction_status,
                    }),
                )
            }
        }
    }

    fn web_artifact_path(&mut self, request: &Request) -> Response {
        match self.find_session_artifact(request) {
            Err(response) => response,
            Ok((session_id, manifest)) => {
                let path = self
                    .store
                    .root()
                    .join(&manifest.object_path)
                    .display()
                    .to_string();
                Response::ok(
                    request,
                    json!({
                        "session_id": session_id,
                        "digest": manifest.digest.hex,
                        "path": path,
                        "byte_count": manifest.byte_count,
                        "media_type": manifest.media_type,
                    }),
                )
            }
        }
    }

    fn find_session_artifact(
        &self,
        request: &Request,
    ) -> Result<(String, crate::artifacts::ArtifactManifest), Response> {
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|value| value.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return Err(protocol_error(request, "web.artifact requires session_id"));
        };
        if !self.sessions.contains_key(&session_id) {
            return Err(missing_session(request, &session_id));
        }
        let Some(id) = request
            .payload
            .get("id")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Err(protocol_error(request, "web.artifact requires id"));
        };
        let listed = match self.store.list_session(&session_id) {
            Ok(list) => list,
            Err(error) => return Err(engine_error(request, error.to_string(), 39)),
        };
        let id_lower = id.to_ascii_lowercase();
        let matches: Vec<_> = listed
            .into_iter()
            .filter(|manifest| {
                let hex = manifest.digest.hex.to_ascii_lowercase();
                hex == id_lower || (id_lower.len() >= 8 && hex.starts_with(&id_lower))
            })
            .collect();
        match matches.len() {
            0 => Err(protocol_error(
                request,
                "artifact id does not refer to an artifact of this session",
            )),
            1 => Ok((session_id, matches.into_iter().next().expect("len 1"))),
            _ => Err(protocol_error(
                request,
                "artifact id is ambiguous; pass a longer digest prefix",
            )),
        }
    }

    fn web_goto(&mut self, request: &Request) -> Response {
        let Some(url) = request
            .payload
            .get("url")
            .and_then(|value| value.as_str())
            .filter(|url| !url.is_empty())
        else {
            return protocol_error(request, "web.goto requires url");
        };
        let url = url.to_owned();
        match self.with_session_page(request, "web.goto") {
            Err(response) => response,
            Ok((session_id, page)) => match self.engine_call(
                "page.goto",
                json!({ "page": page, "url": url, "timeout": 30_000 }),
            ) {
                Ok(result) => {
                    self.finish_action_with_page_state(
                        request,
                        &session_id,
                        &page,
                        json!({
                            "session_id": session_id,
                            "url": result.get("url").cloned().unwrap_or(json!(url)),
                            "status": result.get("status").cloned().unwrap_or(json!(0)),
                            "ok": result.get("ok").cloned().unwrap_or(json!(false)),
                            "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                        }),
                    )
                }
                Err(error) => {
                    self.finish_session(&session_id);
                    engine_error(request, error, 34)
                }
            },
        }
    }


    /// Tabs are pages inside one session: a `web.tab` call adds, lists,
    /// switches or closes a page while the session's cookies and storage stay
    /// shared. `session.page_id` names the active one; `session.tabs` keeps
    /// the rest so switching does not lose them.
    fn web_tab(&mut self, request: &Request, action: &str) -> Response {
        match self.with_session_page(request, "web.tab") {
            Err(response) => response,
            Ok((session_id, active)) => {
                let target = request
                    .payload
                    .get("tab")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned);
                // Sessions created before tabs existed carry an active page
                // that is not in the list yet; adopt it rather than losing it.
                if let Some(session) = self.sessions.get_mut(&session_id) {
                    if !session.tabs.contains(&active) {
                        session.tabs.push(active.clone());
                    }
                }
                let result = match action {
                    "new" => match self.engine_call("context.newPage", json!({})) {
                        Ok(value) => match value.get("page").and_then(|v| v.as_str()) {
                            Some(page) => {
                                let page = page.to_owned();
                                if let Some(session) = self.sessions.get_mut(&session_id) {
                                    session.tabs.push(page.clone());
                                    session.page_id = Some(page.clone());
                                    session.pages = session.tabs.len() as u32;
                                }
                                Ok(json!({ "tab": page, "active": true }))
                            }
                            None => Err("engine returned no page id".to_owned()),
                        },
                        Err(error) => Err(error),
                    },
                    "switch" => match target {
                        None => Err("web.tab switch requires a tab id".to_owned()),
                        Some(tab) => {
                            let known = self
                                .sessions
                                .get(&session_id)
                                .map(|session| session.tabs.contains(&tab))
                                .unwrap_or(false);
                            if !known {
                                Err(format!("no tab {tab} in this session"))
                            } else {
                                if let Some(session) = self.sessions.get_mut(&session_id) {
                                    session.page_id = Some(tab.clone());
                                }
                                Ok(json!({ "tab": tab, "active": true }))
                            }
                        }
                    },
                    "close" => {
                        let tab = target.unwrap_or_else(|| active.clone());
                        let remaining = {
                            let session = self.sessions.get_mut(&session_id);
                            match session {
                                None => Vec::new(),
                                Some(session) => {
                                    session.tabs.retain(|id| id != &tab);
                                    session.locator_snapshots.remove(&tab);
                                    session.pages = session.tabs.len() as u32;
                                    if session.page_id.as_deref() == Some(tab.as_str()) {
                                        session.page_id = session.tabs.last().cloned();
                                    }
                                    session.tabs.clone()
                                }
                            }
                        };
                        // Closing the last tab would leave the session without
                        // a page; the next operation recreates one on demand.
                        let _ = self.engine_call("page.close", json!({ "page": tab }));
                        Ok(json!({ "closed": tab, "tabs": remaining }))
                    }
                    _ => {
                        let (tabs, active_id) = self
                            .sessions
                            .get(&session_id)
                            .map(|session| (session.tabs.clone(), session.page_id.clone()))
                            .unwrap_or_default();
                        let rows: Vec<serde_json::Value> = tabs
                            .iter()
                            .map(|id| {
                                let url = self
                                    .engine_call("page.url", json!({ "page": id }))
                                    .ok()
                                    .and_then(|v| v.get("url").cloned())
                                    .unwrap_or(json!(""));
                                json!({
                                    "tab": id,
                                    "active": Some(id.clone()) == active_id,
                                    "url": url,
                                })
                            })
                            .collect();
                        Ok(json!({ "tabs": rows, "count": rows.len() }))
                    }
                };
                self.finish_session(&session_id);
                match result {
                    Ok(mut value) => {
                        if let Some(object) = value.as_object_mut() {
                            object.insert("session_id".into(), json!(session_id));
                        }
                        Response::ok(request, value)
                    }
                    Err(message) => engine_error(request, message, 34),
                }
            }
        }
    }

    /// Return what the page recorded: console messages, network requests, or
    /// both merged into one time-ordered stream.
    ///
    /// The content worker already keeps both lists per page; this only lifts
    /// them to the protocol so a caller can ask without writing a script.
    /// Records come from the page and are therefore untrusted.
    fn web_records(&mut self, request: &Request, kind: &str) -> Response {
        match self.with_session_page(request, "web.records") {
            Err(response) => response,
            Ok((session_id, page)) => {
                let mut console = json!([]);
                let mut requests = json!([]);
                if kind != "network" {
                    match self.engine_call("page.consoleMessages", json!({ "page": page })) {
                        Ok(value) => {
                            console = value.get("messages").cloned().unwrap_or(json!([]));
                        }
                        Err(error) => {
                            self.finish_session(&session_id);
                            return engine_error(request, error, 34);
                        }
                    }
                }
                if kind != "console" {
                    match self.engine_call("page.requests", json!({ "page": page })) {
                        Ok(value) => {
                            requests = value
                                .get("requests")
                                .cloned()
                                .unwrap_or_else(|| value.clone());
                        }
                        Err(error) => {
                            self.finish_session(&session_id);
                            return engine_error(request, error, 34);
                        }
                    }
                }
                self.finish_session(&session_id);
                let counts = json!({
                    "console": console.as_array().map(Vec::len).unwrap_or(0),
                    "requests": requests.as_array().map(Vec::len).unwrap_or(0),
                });
                let mut result = json!({
                    "session_id": session_id,
                    "counts": counts,
                    "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                });
                if let Some(object) = result.as_object_mut() {
                    if kind != "network" {
                        object.insert("console".into(), console);
                    }
                    if kind != "console" {
                        object.insert("requests".into(), requests);
                    }
                }
                Response::ok(request, result)
            }
        }
    }

    /// Turn the engine's tagged value encoding into plain JSON.
    ///
    /// The content worker serializes JavaScript values structurally —
    /// `{"o":[{"k":…,"v":…}]}` for objects, `{"b":…}`, `{"n":…}`, `{"s":…}`,
    /// `{"a":[…]}`, and `{"v":"null"|"undefined"|"NaN"|…}` for the rest. That
    /// shape is right for round-tripping but unusable for a caller who just
    /// wants the value, so `web.evaluate` hands back plain JSON and keeps the
    /// tagged form alongside it for anything that needs the distinction
    /// between `null` and `undefined`.
    fn plain_value(tagged: &serde_json::Value) -> serde_json::Value {
        if let Some(entries) = tagged.get("o").and_then(|value| value.as_array()) {
            let mut object = serde_json::Map::new();
            for entry in entries {
                let Some(key) = entry.get("k").and_then(|key| key.as_str()) else {
                    continue;
                };
                let value = entry.get("v").map(Self::plain_value).unwrap_or(json!(null));
                object.insert(key.to_owned(), value);
            }
            return serde_json::Value::Object(object);
        }
        if let Some(items) = tagged.get("a").and_then(|value| value.as_array()) {
            return serde_json::Value::Array(items.iter().map(Self::plain_value).collect());
        }
        for key in ["b", "n", "s"] {
            if let Some(value) = tagged.get(key) {
                return value.clone();
            }
        }
        match tagged.get("v").and_then(|value| value.as_str()) {
            // `undefined`, `NaN` and the infinities have no JSON spelling.
            // Reporting them as null loses information the caller may need,
            // so they stay as their engine name in string form.
            Some("null") | None => json!(null),
            Some(other) => json!(other),
        }
    }

    /// Wait for an internal Boolean predicate within one request/session budget.
    /// Never replace a shared worker merely because this wait expires. A true
    /// result remains true if the bounded follow-up observation is unavailable.
    fn web_wait(&mut self, request: &Request) -> Response {
        let started = Instant::now();
        let Some(source) = request.payload.get("source").and_then(|v| v.as_str())
            .filter(|source| !source.trim().is_empty()).map(str::to_owned) else {
            return protocol_error(request, "web.wait requires a non-empty internal Boolean source expression");
        };
        let Some(timeout_ms) = request.payload.get("timeout_ms").and_then(|v| v.as_u64()) else {
            return protocol_error(request, "web.wait requires an unsigned integer timeout_ms");
        };
        let request_deadline = started.checked_add(Duration::from_millis(request.deadline_ms.min(timeout_ms)))
            .unwrap_or(started);
        let (session_id, page) = match self.with_session_page_until(request, "web.wait", Some(request_deadline)) {
            Ok(context) => context,
            Err(response) => return response,
        };
        let source = match self.bind_condition_source(request, &session_id, &page, source) {
            Ok(source) => source,
            Err(response) => {
                self.finish_session(&session_id);
                return response;
            }
        };
        let session_remaining = self.sessions.get(&session_id)
            .map(|session| session.limits.wall_time.saturating_sub(session.started.elapsed()))
            .unwrap_or(Duration::ZERO);
        let budget = crate::wait_contract::remaining_wait_budget(
            request.deadline_ms, timeout_ms, started.elapsed(), session_remaining,
        );
        let wait_deadline = Instant::now().checked_add(budget).unwrap_or_else(Instant::now);
        let result = if budget.as_millis() == 0 {
            Err("timeout: no remaining wait budget".to_string())
        } else {
            // One send/receive budget, without restarting a worker and thereby
            // silently replacing the document the caller intended to wait on.
            match crate::wait_contract::monotonic_ns() {
                Ok(now) => self.engine_call_timed_with_recovery("page.waitForBoolean", json!({
                    "page": page, "source": source,
                    "timeout": budget.as_millis().min(u64::MAX as u128) as u64,
                    "deadline_monotonic_ns": now.saturating_add(budget.as_nanos().min(u64::MAX as u128) as u64),
                }), budget, false),
                Err(error) => Err(format!("wait clock unavailable: {error}")),
            }
        };
        match result {
            Ok(value) if value.get("serialized").and_then(|v| v.get("b")) == Some(&json!(true)) => {
                let waited_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                let observation_budget = wait_deadline.saturating_duration_since(Instant::now())
                    .min(Duration::from_secs(2));
                let state = if self.workflow_defer_observation {
                    json!({"status":"deferred","reason":"intermediate workflow step"})
                } else {
                    page_state_envelope(self.observe_page_bounded(
                        &session_id, &page, observation_budget, false,
                    ))
                };
                let document = self.sessions.get(&session_id)
                    .and_then(|session| session.locator_snapshots.get(&page))
                    .map(|snapshot| snapshot.token.clone());
                self.finish_session(&session_id);
                Response::ok(request, json!({
                    "session_id": session_id, "tab_id": page,
                    "document_id": if state["status"] == "available" { document } else { None },
                    "held": true,
                    "waited_ms": waited_ms,
                    "page_state": state,
                    "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                }))
            }
            other => {
                let error = other.err().unwrap_or_else(|| "INVALID_WAIT_PREDICATE".into());
                let (code, message, recovery) = crate::wait_contract::wait_error_detail(&error);
                self.finish_session(&session_id);
                Response::error(request, ErrorObject::new(
                    code, message, request.request_id.clone(), 34, recovery,
                ))
            }
        }
    }

    /// Evaluate a page expression and return its serialized, untrusted value.
    /// Read/inspect commands use this primitive; the bounded Boolean wait has
    /// its own operation so setup, evaluation and observation share a deadline.
    fn web_evaluate(&mut self, request: &Request) -> Response {
        let source = request
            .payload
            .get("source")
            .and_then(|value| value.as_str())
            .map(str::to_owned);
        let Some(source) = source else {
            return Response::error(
                request,
                ErrorObject::new(
                    "protocol_violation",
                    "web.evaluate requires a source expression",
                    request.request_id.clone(),
                    30,
                    "pass params.source",
                ),
            );
        };
        match self.with_session_page(request, "web.evaluate") {
            Err(response) => response,
            Ok((session_id, page)) => {
                let source = match self.bind_condition_source(request, &session_id, &page, source) {
                    Ok(source) => source,
                    Err(response) => {
                        self.finish_session(&session_id);
                        return response;
                    }
                };
                match self.engine_call("page.evaluate", json!({ "page": page, "source": source })) {
                    Ok(value) => {
                        self.finish_session(&session_id);
                        let tagged = value.get("serialized").cloned().unwrap_or(json!(null));
                        Response::ok(
                            request,
                            json!({
                                "session_id": session_id,
                                "value": Self::plain_value(&tagged),
                                "serialized": tagged,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        if request.payload.get("condition_ref").is_some() {
                            locator_error(request, error)
                        } else {
                            engine_error(request, error, 34)
                        }
                    }
                }
            }
        }
    }

    fn bind_condition_source(
        &self,
        request: &Request,
        session_id: &str,
        page: &str,
        source: String,
    ) -> Result<String, Response> {
        let Some(selector) = request.payload.get("condition_ref") else {
            return Ok(source);
        };
        if selector.get("type").and_then(|kind| kind.as_str()) != Some("ref") {
            return Err(protocol_error(request, "condition_ref requires an observed ref selector"));
        }
        let bound = self.bind_observed_selector(request, session_id, page, selector.clone())?;
        Ok(crate::content_worker::observed_ref_condition_source(&source, &bound))
    }

    fn web_history(&mut self, request: &Request, method: &str, operation: &str) -> Response {
        match self.with_session_page(request, operation) {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.engine_call(method, json!({ "page": page, "timeout": 30_000 })) {
                    Ok(result) => {
                        let url = result
                            .get("url")
                            .cloned()
                            .unwrap_or(json!(""));
                        self.finish_action_with_page_state(
                            request,
                            &session_id,
                            &page,
                            json!({
                                "session_id": session_id,
                                "url": url,
                                "ok": result.get("ok").cloned().unwrap_or(json!(true)),
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        engine_error(request, error, 34)
                    }
                }
            }
        }
    }

    // All native target-taking paths must bind refs before the content worker
    // resolves a node. Type/targeted press focus first, but that is not a reason
    // to bypass the same session/page/snapshot validation used by other actions.
    fn bind_observed_selector(
        &self,
        request: &Request,
        session_id: &str,
        page: &str,
        mut selector: serde_json::Value,
    ) -> Result<serde_json::Value, Response> {
        if selector.get("type").and_then(|value| value.as_str()) != Some("ref") {
            return Ok(selector);
        }
        let Some(ref_number) = selector.get("value").and_then(|value| value.as_u64()) else {
            return Err(protocol_error(
                request,
                "ref selector requires a positive integer value",
            ));
        };
        let snapshot = self.sessions.get(session_id).and_then(|session| {
            session.locator_snapshots.get(page).and_then(|snapshot| {
                (snapshot.page_id == page && ref_number > 0 && ref_number <= snapshot.ref_ceiling)
                    .then(|| snapshot.token.clone())
            })
        });
        let Some(snapshot) = snapshot else {
            return Err(locator_error(
                request,
                "STALE_REF: run web.observe and use a ref from its current page snapshot",
            ));
        };
        selector = json!({
            "type": "css",
            "value": format!("[data-greppy-ref=\"{}:{}\"]", snapshot, ref_number),
            "snapshot": snapshot,
            "observed_ref": ref_number,
        });
        Ok(selector)
    }

    fn web_locator_method(
        &mut self,
        request: &Request,
        method: &str,
        extra: serde_json::Value,
    ) -> Response {
        let started = Instant::now();
        let Some(selector) = request.payload.get("selector").cloned() else {
            return protocol_error(request, &format!("{} requires selector", request.operation));
        };
        match self.with_session_page(request, &request.operation) {
            Err(response) => response,
            Ok((session_id, page)) => {
                let selector =
                    match self.bind_observed_selector(request, &session_id, &page, selector) {
                        Ok(selector) => selector,
                        Err(response) => {
                            if method == "locator.inspect" {
                                self.finish_session(&session_id);
                                return response;
                            }
                            return self.finish_failed_action_with_page_state(
                                request, &session_id, &page, response, started,
                            );
                        }
                    };
                let timeout = request
                    .payload
                    .get("timeout")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(30_000);
                let mut params = json!({
                    "page": page,
                    "selector": selector,
                    "timeout": timeout,
                });
                if let Some(object) = extra.as_object() {
                    if let Some(dst) = params.as_object_mut() {
                        for (key, value) in object {
                            dst.insert(key.clone(), value.clone());
                        }
                    }
                }
                match self.engine_call(method, params) {
                    Ok(result) => {
                        if method == "locator.inspect" {
                            self.finish_session(&session_id);
                            let tagged = result.get("serialized").cloned().unwrap_or(json!(null));
                            return Response::ok(
                                request,
                                json!({
                                    "session_id": session_id,
                                    "value": Self::plain_value(&tagged),
                                    "serialized": tagged,
                                    "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                                }),
                            );
                        }
                        let dispatch = result.get("dispatch").cloned();
                        let mut response = json!({
                            "session_id": session_id,
                            "ok": true,
                            "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                        });
                        if let (Some(dispatch), Some(object)) = (dispatch, response.as_object_mut())
                        {
                            object.insert("dispatch".into(), dispatch);
                        }
                        self.finish_action_with_page_state(request, &session_id, &page, response)
                    }
                    Err(error) => {
                        let response = locator_error(request, error);
                        if method == "locator.inspect" {
                            self.finish_session(&session_id);
                            response
                        } else {
                            self.finish_failed_action_with_page_state(
                                request, &session_id, &page, response, started,
                            )
                        }
                    }
                }
            }
        }
    }

    fn web_fill(&mut self, request: &Request) -> Response {
        let Some(value) = request.payload.get("value").and_then(|value| value.as_str()) else {
            return protocol_error(request, "web.fill requires value");
        };
        self.web_locator_method(
            request,
            "locator.fill",
            json!({ "value": value, "editable": true }),
        )
    }

    fn web_select(&mut self, request: &Request) -> Response {
        let Some(value) = request.payload.get("value").and_then(|value| value.as_str()) else {
            return protocol_error(request, "web.select requires value");
        };
        self.web_locator_method(request, "locator.selectOption", json!({ "value": value }))
    }

    fn web_type(&mut self, request: &Request) -> Response {
        let started = Instant::now();
        let Some(text) = request
            .payload
            .get("text")
            .and_then(|value| value.as_str())
            .map(str::to_owned)
        else {
            return protocol_error(request, "web.type requires text");
        };
        let Some(selector) = request.payload.get("selector").cloned() else {
            return protocol_error(request, "web.type requires selector");
        };
        match self.with_session_page(request, "web.type") {
            Err(response) => response,
            Ok((session_id, page)) => {
                let selector =
                    match self.bind_observed_selector(request, &session_id, &page, selector) {
                        Ok(selector) => selector,
                        Err(response) => {
                            return self.finish_failed_action_with_page_state(
                                request, &session_id, &page, response, started,
                            );
                        }
                    };
                let timeout = request
                    .payload
                    .get("timeout")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(30_000);
                let focus = self.engine_call(
                    "locator.focus",
                    json!({ "page": page, "selector": selector, "timeout": timeout }),
                );
                if let Err(error) = focus {
                    return self.finish_failed_action_with_page_state(
                        request, &session_id, &page, locator_error(request, error), started,
                    );
                }
                match self.engine_call("page.keyboard.type", json!({ "page": page, "text": text }))
                {
                    Ok(_) => {
                        self.finish_action_with_page_state(
                            request,
                            &session_id,
                            &page,
                            json!({
                                "session_id": session_id,
                                "ok": true,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(error) => {
                        self.finish_failed_action_with_page_state(
                            request, &session_id, &page, locator_error(request, error), started,
                        )
                    }
                }
            }
        }
    }

    fn web_press(&mut self, request: &Request) -> Response {
        let started = Instant::now();
        let Some(key) = request
            .payload
            .get("key")
            .and_then(|value| value.as_str())
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
        else {
            return protocol_error(request, "web.press requires key");
        };
        match self.with_session_page(request, "web.press") {
            Err(response) => response,
            Ok((session_id, page)) => {
                if let Some(selector) = request.payload.get("selector").cloned() {
                    let selector =
                        match self.bind_observed_selector(request, &session_id, &page, selector) {
                            Ok(selector) => selector,
                            Err(response) => {
                                return self.finish_failed_action_with_page_state(
                                    request, &session_id, &page, response, started,
                                );
                            }
                        };
                    let timeout = request
                        .payload
                        .get("timeout")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(30_000);
                    if let Err(error) = self.engine_call(
                        "locator.focus",
                        json!({ "page": page, "selector": selector, "timeout": timeout }),
                    ) {
                        return self.finish_failed_action_with_page_state(
                            request, &session_id, &page, locator_error(request, error), started,
                        );
                    }
                }
                match self.engine_call("page.keyboard.press", json!({ "page": page, "key": key })) {
                    Ok(_) => {
                        self.finish_action_with_page_state(
                            request,
                            &session_id,
                            &page,
                            json!({
                                "session_id": session_id,
                                "ok": true,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(error) => {
                        self.finish_failed_action_with_page_state(
                            request, &session_id, &page, locator_error(request, error), started,
                        )
                    }
                }
            }
        }
    }

    fn web_scroll(&mut self, request: &Request) -> Response {
        let started = Instant::now();
        if request.payload.get("selector").is_some() {
            return self.web_locator_method(
                request,
                "locator.scrollIntoViewIfNeeded",
                json!({}),
            );
        }
        let delta_y = request
            .payload
            .get("delta_y")
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        match self.with_session_page(request, "web.scroll") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.engine_call(
                    "page.mouse.wheel",
                    json!({ "page": page, "x": 0, "y": 0, "deltaX": 0, "deltaY": delta_y }),
                ) {
                    Ok(_) => {
                        // The wheel event reaches the DOM, so listeners fire,
                        // but in Servo the viewport is moved by the compositor
                        // rather than by the event. Without this the caller
                        // gets `ok` and a page that never moved. Report where
                        // the viewport actually ended up so a scroll that hit
                        // the bottom is visible rather than silently ignored.
                        let scrolled = self
                            .engine_call(
                                "page.evaluate",
                                json!({
                                    "page": page,
                                    "source": format!(
                                        "(function(){{ window.scrollBy(0, {delta_y}); \
                                         return window.scrollY; }})()"
                                    ),
                                }),
                            )
                            .ok()
                            .map(|value| {
                                Self::plain_value(
                                    &value.get("serialized").cloned().unwrap_or(json!(null)),
                                )
                            })
                            .unwrap_or(json!(null));
                        self.finish_action_with_page_state(
                            request,
                            &session_id,
                            &page,
                            json!({
                                "session_id": session_id,
                                "ok": true,
                                "scroll_y": scrolled,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(error) => {
                        self.finish_failed_action_with_page_state(
                            request, &session_id, &page, locator_error(request, error), started,
                        )
                    }
                }
            }
        }
    }

    fn web_upload(&mut self, request: &Request) -> Response {
        let started = Instant::now();
        let Some(selector) = request.payload.get("selector").cloned() else {
            return protocol_error(request, "web.upload requires selector");
        };
        let css = selector
            .get("type")
            .and_then(|value| value.as_str())
            .filter(|kind| *kind == "css")
            .and_then(|_| selector.get("value").and_then(|value| value.as_str()))
            .map(str::to_owned);
        let Some(css) = css else {
            return protocol_error(request, "web.upload requires a css= TARGET");
        };
        let files = request
            .payload
            .get("files")
            .cloned()
            .unwrap_or(json!([]));
        match self.with_session_page(request, "web.upload") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.engine_call(
                    "page.setInputFiles",
                    json!({ "page": page, "selector": css, "files": files }),
                ) {
                    Ok(_) => {
                        self.finish_action_with_page_state(
                            request,
                            &session_id,
                            &page,
                            json!({
                                "session_id": session_id,
                                "ok": true,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(error) => {
                        self.finish_failed_action_with_page_state(
                            request, &session_id, &page, locator_error(request, error), started,
                        )
                    }
                }
            }
        }
    }

    fn with_session_page(
        &mut self,
        request: &Request,
        operation: &str,
    ) -> Result<(String, String), Response> {
        self.with_session_page_until(request, operation, None)
    }

    fn with_session_page_until(
        &mut self,
        request: &Request,
        operation: &str,
        deadline: Option<Instant>,
    ) -> Result<(String, String), Response> {
        let deadline = match (deadline, self.workflow_deadline) {
            (Some(first), Some(second)) => Some(first.min(second)),
            (first, second) => first.or(second),
        };
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return Err(protocol_error(
                request,
                &format!("{operation} requires session_id"),
            ));
        };
        if !self.sessions.contains_key(&session_id) {
            return Err(missing_session(request, &session_id));
        }
        if !self.content.is_running() {
            if deadline.is_some() {
                return Err(engine_error(request, "content worker is unavailable; bounded wait did not restart or replace its document", 33));
            }
            self.recover_content("content worker exited before session operation")
                .map_err(|error| engine_error(request, error, 33))?;
        }
        // An explicit tab is a scoped target, not a request to silently use or
        // switch the session's active page. Validate ownership before any work.
        let requested_page = match request.payload.get("tab_id") {
            None => None,
            Some(value) => {
                let Some(tab) = value.as_str().filter(|tab| !tab.is_empty()) else {
                    return Err(protocol_error(request, "tab_id must be a non-empty string"));
                };
                let known = self.sessions.get(&session_id).is_some_and(|session| {
                    session.page_id.as_deref() == Some(tab)
                        || session.tabs.iter().any(|page| page == tab)
                });
                if !known {
                    let mut error = ErrorObject::new(
                        "TAB_NOT_FOUND",
                        "the requested tab does not belong to this session (or was reset)",
                        request.request_id.clone(),
                        30,
                        "run greppy web tab list --session SID and use a tab from that session",
                    );
                    error.session_id = Some(session_id.clone());
                    return Err(Response::error(request, error));
                }
                Some(tab.to_owned())
            }
        };
        let content_rss = sample_rss_bytes(self.content.pid());
        let controller_rss = sample_rss_bytes(self.controller.pid());
        let content_pid = self.content.pid();
        let controller_pid = self.controller.pid();
        let wall_time_error = self.sessions.get_mut(&session_id).and_then(|session| {
            session.peak_rss_bytes = session.peak_rss_bytes.max(content_rss);
            // Budget the CPU this SESSION used, not the worker lifetime
            // (finding 039); a respawned worker resets the baseline.
            let content_cpu = Duration::from_millis(session_cpu_delta_ms(
                &mut session.content_cpu_baseline,
                content_pid,
            ));
            let controller_cpu = Duration::from_millis(session_cpu_delta_ms(
                &mut session.controller_cpu_baseline,
                controller_pid,
            ));
            if let Err(message) = session.begin_operation(&request.request_id) {
                Some(("engine", message))
            } else if let Err(message) = session.limits.check_wall_time(session.started.elapsed()) {
                let _ = session.transition(SessionState::Failed);
                Some(("wall_limit", message))
            } else if let Err(message) = session.limits.check_content_rss(content_rss) {
                let _ = session.transition(SessionState::Failed);
                Some(("limit", message))
            } else if let Err(message) = session.limits.check_controller_memory(controller_rss) {
                let _ = session.transition(SessionState::Failed);
                Some(("limit", message))
            } else if let Err(message) = session.limits.check_cpu_time(
                content_cpu,
                session.limits.content_cpu_time,
                "content",
            ) {
                let _ = session.transition(SessionState::Failed);
                Some(("limit", message))
            } else if let Err(message) = session.limits.check_cpu_time(
                controller_cpu,
                session.limits.controller_cpu_time,
                "controller",
            ) {
                let _ = session.transition(SessionState::Failed);
                Some(("limit", message))
            } else {
                None
            }
        });
        match wall_time_error {
            Some(("engine", message)) => return Err(engine_error(request, message, 38)),
            Some(("wall_limit", message)) if deadline.is_some() => {
                // Expiring this session's elapsed-time quota says nothing
                // about the health of the shared worker. A bounded wait must
                // refuse without replacing other sessions' live documents.
                // CPU and RSS violations still take the recovery path below.
                return Err(limit_error(request, message));
            }
            Some(("wall_limit", message)) => {
                let _ = self.recover_content(&format!("wall time exceeded: {message}"));
                return Err(limit_error(request, message));
            }
            Some(("limit", message)) => {
                let _ = self.recover_content(&format!("wall time exceeded: {message}"));
                return Err(limit_error(request, message));
            }
            Some(_) => unreachable!(),
            None => {}
        }
        let page = requested_page.or_else(|| {
            self.sessions.get(&session_id).and_then(|session| session.page_id.clone())
        });
        let page = match page {
            Some(page) => page,
            None => {
                if let Some(session) = self.sessions.get(&session_id) {
                    if let Err(message) =
                        session.limits.check_pages(session.pages.saturating_add(1))
                    {
                        return Err(limit_error(request, message));
                    }
                }
                if deadline.is_some() {
                    self.finish_session(&session_id);
                    return Err(protocol_error(request, "web.wait requires an existing page; open or select a tab first"));
                }
                match self.engine_call("session.ensurePage", json!({})) {
                    Ok(result) => {
                        let page = result
                            .get("page")
                            .and_then(|value| value.as_str())
                            .map(str::to_owned);
                        let Some(page) = page else {
                            self.finish_session(&session_id);
                            return Err(engine_error(request, "session has no page", 34));
                        };
                        if let Some(session) = self.sessions.get_mut(&session_id) {
                            session.page_id = Some(page.clone());
                            session.pages = 1;
                        }
                        page
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        return Err(engine_error(request, error, 34));
                    }
                }
            }
        };
        let profile = self
            .sessions
            .get(&session_id)
            .map(|session| session.profile)
            .unwrap_or(NetworkProfile::Research);
        let profile_result = if let Some(end) = deadline {
            let remaining = end.saturating_duration_since(Instant::now());
            let remaining = self.sessions.get(&session_id)
                .map(|session| remaining.min(session.limits.wall_time.saturating_sub(session.started.elapsed())))
                .unwrap_or(Duration::ZERO);
            if remaining < Duration::from_millis(1) {
                Err("timeout: no remaining wait setup budget".into())
            } else {
                self.engine_call_timed_with_recovery("session.setProfile", json!({"profile":profile.as_str()}), remaining, false)
            }
        } else {
            self.engine_call("session.setProfile", json!({ "profile": profile.as_str() }))
        };
        if let Err(error) = profile_result {
            self.finish_session(&session_id);
            if deadline.is_some() {
                let (code, message, recovery) = crate::wait_contract::wait_error_detail(&error);
                return Err(Response::error(request, ErrorObject::new(code, message, request.request_id.clone(), 34, recovery)));
            }
            return Err(engine_error(request, error, 34));
        }
        Ok((session_id, page))
    }

    fn navigate_and_extract(
        &mut self,
        session_id: &str,
        page: &str,
        url: &str,
        request: &Request,
    ) -> Result<serde_json::Value, Response> {
        let profile = self
            .sessions
            .get(session_id)
            .map(|session| session.profile)
            .unwrap_or(NetworkProfile::Research);
        if let UrlDecision::Deny { reason } = decide_url(profile, url) {
            return Err({
                let mut error = ErrorObject::new(
                    "policy_denied",
                    format!("{reason}: {}", redact_secrets(url)),
                    request.request_id.clone(),
                    36,
                    policy_recovery(reason),
                );
                error.session_id = Some(session_id.to_owned());
                Response::error(request, error)
            });
        }
        if let Some(session) = self.sessions.get_mut(session_id) {
            if let Err(message) = session
                .limits
                .check_requests(session.requests.saturating_add(1))
            {
                return Err(limit_error(request, message));
            }
            if let Err(message) = session
                .limits
                .check_network_bytes(session.network_bytes, 4096)
            {
                return Err(limit_error(request, message));
            }
            session.requests = session.requests.saturating_add(1);
            session.network_bytes = session.network_bytes.saturating_add(4096);
        }
        self.engine_call("page.goto", json!({ "page": page, "url": url }))
            .map_err(|error| engine_error(request, error, 34))?;
        self.apply_page_record_limits(session_id, page, request)?;
        let tree = self
            .engine_call("page.observe", json!({ "page": page }))
            .map_err(|error| engine_error(request, error, 34))?;
        let recorded = self
            .engine_call("page.requests", json!({ "page": page }))
            .unwrap_or_else(|_| json!({ "requests": [] }));
        let responses = self
            .engine_call("page.responses", json!({ "page": page }))
            .ok()
            .and_then(|value| value.get("responses").cloned())
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default();
        let http_status = responses
            .iter()
            .rev()
            .find_map(|row| row.get("status").and_then(|value| value.as_u64()));
        let text = tree
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let title = tree
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let stored = self.store_bytes(
            request,
            session_id,
            text.as_bytes(),
            "text/plain",
            &request.operation,
            false,
        )?;
        model_facing_source(
            json!({
                "requested_url": redact_secrets(url),
                "final_url": tree.get("url").and_then(|value| value.as_str()).map(redact_secrets),
                "redirect_chain": redirect_chain(url, tree.get("url"), recorded.get("requests"))
                    .into_iter()
                    .map(|hop| redact_secrets(&hop))
                    .collect::<Vec<_>>(),
                "retrieved_at": stored.timestamp,
                "title": title,
                "media_type": "text/html",
                "text": text,
                "digest": stored.digest.hex,
                "artifact_digest": stored.digest.hex,
                "http_status": http_status,
                "classification": "original",
                "session_id": session_id,
                "operation_id": request.request_id,
                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
            }),
            &stored,
        )
        .map_err(|error| engine_error(request, error, 39))
    }

    fn store_bytes(
        &mut self,
        request: &Request,
        session_id: &str,
        bytes: &[u8],
        media_type: &str,
        operation: &str,
        sensitive: bool,
    ) -> Result<crate::artifacts::ArtifactManifest, Response> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if let Err(message) = session
                .limits
                .check_artifact_bytes(session.artifact_bytes, bytes.len() as u64)
            {
                return Err(limit_error(request, message));
            }
            session.artifact_bytes = session.artifact_bytes.saturating_add(bytes.len() as u64);
        }
        self.store
            .put(
                bytes,
                media_type,
                session_id,
                &self.run_id,
                &format!("{operation}:{}", request.request_id),
                sensitive,
            )
            .map_err(|error| engine_error(request, error.to_string(), 39))
    }

    fn search_url(&self, query: &str) -> String {
        if let Some(endpoint) = &self.search_endpoint {
            if endpoint.contains('?') {
                format!("{endpoint}&q={}", urlencoding(query))
            } else {
                format!("{endpoint}?q={}", urlencoding(query))
            }
        } else {
            format!("https://html.duckduckgo.com/html/?q={}", urlencoding(query))
        }
    }

    fn engine_call(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        self.engine_call_timed(method, params, Duration::from_secs(60))
    }

    fn engine_call_timed(
        &mut self,
        method: &str,
        params: serde_json::Value,
        timeout: Duration,
    ) -> Result<serde_json::Value, String> {
        self.engine_call_timed_with_recovery(method, params, timeout, true)
    }

    fn engine_call_timed_with_recovery(
        &mut self,
        method: &str,
        params: serde_json::Value,
        timeout: Duration,
        recover_worker: bool,
    ) -> Result<serde_json::Value, String> {
        let started = Instant::now();
        let timeout = self.workflow_deadline.map_or(timeout, |deadline| {
            timeout.min(deadline.saturating_duration_since(started))
        });
        if self.workflow_deadline.is_some() && timeout.as_millis() == 0 {
            return Err("timeout: workflow request budget exhausted before engine dispatch".into());
        }
        let recover_worker = recover_worker && self.workflow_deadline.is_none();
        let mut params = params;
        if self.workflow_deadline.is_some() {
            if let Some(object) = params.as_object_mut() {
                let budget_ms = timeout.as_millis().min(u64::MAX as u128) as u64;
                let operation_ms = object.get("timeout").and_then(|value| value.as_u64())
                    .unwrap_or(budget_ms);
                object.insert("timeout".into(), json!(operation_ms.min(budget_ms)));
            }
        }
        if !self.content.is_running() {
            if !recover_worker {
                return Err("content worker is unavailable; observation did not restart it".into());
            }
            self.recover_content("content worker exited")?;
            return Err(
                "content worker crashed and was restarted; session pages were reset".into(),
            );
        }
        let request_id = self.next_engine_id.fetch_add(1, Ordering::Relaxed);
        let stale = self.content.discard_stale_engine_results();
        if stale > 0 {
            self.run_control
                .discarded_engine_results
                .fetch_add(stale, Ordering::Relaxed);
        }
        if let Err(error) = self.content.send_timeout(
            &Message::engine_call(request_id, method.to_owned(), params),
            timeout,
        ) {
            if recover_worker {
                let _ = self.recover_content(&format!("content send failed: {error}"));
            }
            return Err(error.to_string());
        }
        // Diagnostic reads spend one bounded budget across send and receive.
        // Preserve the existing ordinary-call contract in this scoped change.
        let deadline = if recover_worker { Instant::now() + timeout } else { started + timeout };
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(format!("timed out after {timeout:?} waiting for {method}"));
            }
            // A worker can die immediately after the liveness check above.  Do not
            // then hold the request hostage for the full engine timeout: poll the
            // protocol reader in small slices so child death is observed before the
            // client-side request deadline expires.
            let poll = remaining.min(Duration::from_millis(50));
            match self.content.recv(poll) {
                Ok(Message::EngineResult {
                    request_id: got,
                    ok,
                    result,
                    error,
                    ..
                }) if got == request_id => {
                    return if ok {
                        Ok(result)
                    } else {
                        Err(error.unwrap_or_else(|| "engine call failed".to_owned()))
                    };
                }
                Ok(Message::EngineResult {
                    request_id: got,
                    ok,
                    error,
                    ..
                }) => {
                    self.run_control
                        .discarded_engine_results
                        .fetch_add(1, Ordering::Relaxed);
                    if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: discarded unmatched EngineResult id={got} want={request_id} ok={ok} err={error:?}"
                    ); }
                }
                Ok(other) => return Err(format!("unexpected content message {other:?}")),
                Err(error) => {
                    if error.kind() == io::ErrorKind::TimedOut {
                        if !self.content.is_running() {
                            if !recover_worker {
                                return Err("content worker exited during observation; it was not restarted".into());
                            }
                            self.recover_content("content worker exited while handling request")?;
                            return Err(
                                "content worker crashed and was restarted; session pages were reset"
                                    .into(),
                            );
                        }
                        continue;
                    }
                    let message = error.to_string();
                    if recover_worker {
                        let _ = self.recover_content(&format!("content worker: {message}"));
                    }
                    return Err(message);
                }
            }
        }
    }

    fn apply_page_record_limits(
        &mut self,
        session_id: &str,
        page: &str,
        request: &Request,
    ) -> Result<(), Response> {
        let console = self
            .engine_call("page.consoleMessages", json!({ "page": page }))
            .map_err(|error| engine_error(request, error, 34))?;
        let downloads = self
            .engine_call("page.downloads", json!({ "page": page }))
            .map_err(|error| engine_error(request, error, 34))?;
        let recorded_responses = self
            .engine_call("page.responses", json!({ "page": page }))
            .map_err(|error| engine_error(request, error, 34))?;
        let records = json!({
            "messages": console.get("messages").cloned().unwrap_or(json!([])),
            "downloads": downloads.get("downloads").cloned().unwrap_or(json!([])),
            "responses": recorded_responses.get("responses").cloned().unwrap_or(json!([])),
        });
        match apply_record_limits(&mut self.sessions, session_id, &records) {
            Ok(()) => Ok(()),
            Err(message) => Err(limit_error(request, message)),
        }
    }

    fn ensure_workers(&mut self) {
        if !self.content.is_running() {
            let _ = self.recover_content("content worker exited");
        }
        if !self.controller.is_running() {
            let _ = self.recover_controller("controller worker exited");
        }
    }

    fn record_crash(&mut self, worker: &str, reason: &str, recovered: bool) {
        let reason = redact_secrets(reason);
        self.last_crash = Some(reason.clone());
        let recovered_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        self.crash_receipts.push(json!({
            "kind": "worker_crash",
            "worker": worker,
            "reason": reason,
            "recovered": recovered,
            "recovered_at_unix_ms": recovered_at_unix_ms,
        }));
    }

    fn recover_controller(&mut self, reason: &str) -> Result<(), String> {
        let token = match random_token() {
            Ok(token) => token,
            Err(error) => {
                self.record_crash("controller", reason, false);
                return Err(error.to_string());
            }
        };
        let mut controller = match WorkerProcess::spawn(WorkerKind::Controller, token) {
            Ok(controller) => controller,
            Err(error) => {
                self.record_crash("controller", reason, false);
                return Err(error.to_string());
            }
        };
        if let Err(error) = controller.handshake() {
            self.record_crash("controller", reason, false);
            return Err(error.to_string());
        }
        self.controller.kill_tree();
        self.controller = controller;
        self.run_control
            .controller_pid
            .store(self.controller.pid(), Ordering::Relaxed);
        self.run_control
            .controller_generation
            .fetch_add(1, Ordering::Relaxed);
        self.record_crash("controller", reason, true);
        Ok(())
    }

    fn replace_content_worker(&mut self, reason: &str) -> Result<(u32, u32, u64), String> {
        let pid_before = self.content.pid();
        let token = match random_token() {
            Ok(token) => token,
            Err(error) => {
                self.record_crash("content", reason, false);
                return Err(error.to_string());
            }
        };
        let mut content = match WorkerProcess::spawn(WorkerKind::Content, token) {
            Ok(content) => content,
            Err(error) => {
                self.record_crash("content", reason, false);
                return Err(error.to_string());
            }
        };
        if let Err(error) = content.handshake() {
            self.record_crash("content", reason, false);
            return Err(error.to_string());
        }
        self.content.kill_tree();
        self.content = content;
        self.run_control
            .content_pid
            .store(self.content.pid(), Ordering::Relaxed);
        let generation = self
            .run_control
            .content_generation
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        self.record_crash("content", reason, true);
        Ok((pid_before, self.content.pid(), generation))
    }

    fn respawn_content_after_cancel(&mut self, reason: &str) -> Result<(u32, u32, u64), String> {
        let ids = self.replace_content_worker(reason)?;
        for session in self.sessions.values_mut() {
            session.page_id = None;
            session.tabs.clear();
            session.locator_snapshots.clear();
            session.pages = 0;
        }
        Ok(ids)
    }

    fn recover_content(&mut self, reason: &str) -> Result<(), String> {
        self.replace_content_worker(reason)?;
        let mut recovered = Vec::new();
        for session in self.sessions.values_mut() {
            session.page_id = None;
            session.tabs.clear();
            session.locator_snapshots.clear();
            session.pages = 0;
            if session.state == SessionState::Busy {
                let request_id = session.operation_id.clone().unwrap_or_default();
                let _ = session.transition(SessionState::Ready);
                recovered.push((session.id.clone(), request_id));
            } else if session.state == SessionState::Ready {
                recovered.push((session.id.clone(), String::new()));
            }
        }
        for (session_id, request_id) in recovered {
            self.journal(
                &session_id,
                &request_id,
                "session.recovered",
                json!({ "reason": redact_secrets(reason) }),
            );
            if let Some(session) = self.sessions.get(&session_id) {
                self.persist_session_snapshot(
                    session,
                    json!({ "event": "session.recovered", "reason": redact_secrets(reason) }),
                );
            }
        }
        self.journal(
            "runtime",
            "",
            "content.recovered",
            json!({ "reason": redact_secrets(reason) }),
        );
        Ok(())
    }

    fn persist_session_snapshot(&self, session: &Session, extra: serde_json::Value) {
        let path = self
            .store
            .root()
            .join("sessions")
            .join(&session.id)
            .join("session.json");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let body = json!({
            "session_id": session.id,
            "run_id": session.run_id,
            "state": format!("{:?}", session.state).to_lowercase(),
            "profile": session.profile.as_str(),
            "persistent_profile": session.persistent_profile,
            "ephemeral": session.persistent_profile.is_none(),
            "extra": extra,
        });
        if let Ok(bytes) = serde_json::to_vec_pretty(&body) {
            let _ = std::fs::write(path, bytes);
        }
    }

    fn journal(&self, session_id: &str, request_id: &str, event: &str, extra: serde_json::Value) {
        let path = self
            .store
            .root()
            .join("sessions")
            .join(session_id)
            .join("journal.jsonl");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let line = journal_line(event, session_id, &self.run_id, request_id, extra);
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }

    fn remove_ephemeral_session_dir(&self, session_id: &str) {
        remove_script_stage(&self.run_id, session_id, None);
        let path = self.store.root().join("sessions").join(session_id);
        let _ = std::fs::remove_dir_all(path);
    }

    fn should_idle_exit(&self) -> bool {
        self.ever_had_session
            && self.sessions.is_empty()
            && self.last_request.elapsed() >= self.idle_ttl
    }

    fn idle_exit(&mut self) {
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase idle-exit"); } }
        self.exiting = true;
        self.sessions.clear();
        self.profile_locks.clear();
        self.controller.shutdown_or_kill();
        self.content.shutdown_or_kill();
        let _ = std::fs::remove_file(&self.socket);
        self.journal(
            "runtime",
            "",
            "runtime.idle_exit",
            json!({ "idle_ttl_ms": self.idle_ttl.as_millis() as u64 }),
        );
    }

    fn reap_idle_sessions(&mut self) {
        let now = Instant::now();
        let stale: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, session)| {
                session.state != SessionState::Busy
                    && now.duration_since(session.last_heartbeat) > session.limits.idle_ttl
            })
            .map(|(id, _)| id.clone())
            .collect();
        for session_id in stale {
            if let Some(mut session) = self.sessions.remove(&session_id) {
                let ephemeral = session.persistent_profile.is_none();
                let _ = self.profile_locks.remove(&session_id);
                if let Some(page) = session.page_id.take() {
                    let _ = self.engine_call("page.close", json!({ "page": page }));
                }
                if ephemeral {
                    self.remove_ephemeral_session_dir(&session_id);
                }
            }
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        if self.exiting {
            return;
        }
        if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase supervisor-drop"); } }
        self.exiting = true;
        self.controller.shutdown_or_kill();
        self.content.shutdown_or_kill();
    }
}

fn data_root(run_id: &str) -> PathBuf {
    let base = std::env::var("GREPPY_STORE_DIR")
        .or_else(|_| std::env::var("GREPPY_RUNTIME_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("greppy-web-runtime"));
    base.join("web-runtime").join(run_id)
}

fn urlencoding(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

const RESULT_NEXT_CHUNK: usize = 64 * 1024;

fn parse_result_cursor(cursor: &str) -> Result<(String, usize), String> {
    let rest = cursor
        .strip_prefix("sha256:")
        .ok_or_else(|| "cursor must start with sha256:".to_owned())?;
    let (digest, offset) = rest
        .rsplit_once(':')
        .ok_or_else(|| "cursor missing offset".to_owned())?;
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("cursor digest is not sha256 hex".to_owned());
    }
    let offset: usize = offset
        .parse()
        .map_err(|_| "cursor offset is not a number".to_owned())?;
    Ok((digest.to_ascii_lowercase(), offset))
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let a = chunk[0] as u32;
        let b = chunk.get(1).copied().unwrap_or(0) as u32;
        let c = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (a << 16) | (b << 8) | c;
        out.push(TABLE[((triple >> 18) & 63) as usize] as char);
        out.push(TABLE[((triple >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn confine_screenshot_sidecar(path: &str) -> Result<PathBuf, String> {
    let requested = PathBuf::from(path);
    let root = std::env::temp_dir()
        .canonicalize()
        .unwrap_or_else(|_| std::env::temp_dir());
    let canon = requested.canonicalize().map_err(|error| error.to_string())?;
    if !canon.starts_with(&root) {
        return Err(format!("path outside worker temp: {}", canon.display()));
    }
    let name = canon
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if !name.starts_with("greppy-web-shot-") || !name.ends_with(".png") {
        return Err("refusing non-screenshot sidecar".into());
    }
    Ok(canon)
}

fn screenshot_png_bytes(result: &serde_json::Value) -> Result<Vec<u8>, String> {
    if let Some(b64) = result.get("png_base64").and_then(|value| value.as_str()) {
        if !b64.is_empty() {
            return decode_base64(b64);
        }
    }
    if let Some(path) = result.get("png_path").and_then(|value| value.as_str()) {
        let confined = confine_screenshot_sidecar(path)?;
        let bytes = std::fs::read(&confined).map_err(|error| error.to_string())?;
        let _ = std::fs::remove_file(&confined);
        return Ok(bytes);
    }
    Err("screenshot missing png".into())
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err("invalid base64".into()),
        }
    }
    let filtered: Vec<u8> = input
        .bytes()
        .filter(|c| *c != b'=' && !c.is_ascii_whitespace())
        .collect();
    let mut out = Vec::new();
    for chunk in filtered.chunks(4) {
        let a = val(chunk[0])?;
        let b = val(*chunk.get(1).unwrap_or(&b'A'))?;
        let c = val(*chunk.get(2).unwrap_or(&b'A'))?;
        let d = val(*chunk.get(3).unwrap_or(&b'A'))?;
        let triple = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | d as u32;
        out.push(((triple >> 16) & 255) as u8);
        if chunk.len() > 2 {
            out.push(((triple >> 8) & 255) as u8);
        }
        if chunk.len() > 3 {
            out.push((triple & 255) as u8);
        }
    }
    Ok(out)
}

fn protocol_error(request: &Request, message: &str) -> Response {
    Response::error(
        request,
        ErrorObject::new(
            "protocol_violation",
            message,
            request.request_id.clone(),
            30,
            "see greppy web --help",
        ),
    )
}

fn missing_session(request: &Request, session_id: &str) -> Response {
    let mut error = ErrorObject::new(
        "session_not_found",
        format!("session {session_id} was not found"),
        request.request_id.clone(),
        32,
        "create a session first",
    );
    error.session_id = Some(session_id.to_owned());
    Response::error(request, error)
}

fn policy_recovery(reason: &str) -> &'static str {
    if reason.starts_with("research profile denies loopback and private networks") {
        "do not retry this URL in the unchanged research session. For explicitly requested local development only, run `greppy web session create --profile project --json`, then pass the returned session ID with `--session SID` to the local operation. Project permits loopback; LAN and cloud metadata remain blocked"
    } else {
        "choose a permitted public HTTP(S) URL; do not retry the unchanged denied operation or restart the runtime. LAN and cloud metadata remain blocked in both profiles"
    }
}

fn engine_error(request: &Request, message: impl Into<String>, exit_code: i32) -> Response {
    let message = message.into();
    if let Some(reason) = message.strip_prefix("policy_denied:") {
        let mut error = ErrorObject::new(
            "policy_denied",
            redact_secrets(&message),
            request.request_id.clone(),
            exit_code,
            policy_recovery(reason.trim_start()),
        );
        error.session_id = request.session_id.clone();
        error.retryable = false;
        return Response::error(request, error);
    }
    // The call is lost here on purpose: the worker died at an unknown point,
    // so replaying it could repeat a half-applied action. What the caller
    // needs is a signal it can branch on (finding 030): the CLI forgets the
    // session on this text, and a direct protocol client -- an SDK, a
    // foreign harness -- gets the same in a typed field instead of prose.
    if message.starts_with("content worker crashed and was restarted") {
        let mut error = ErrorObject::new(
            "worker_restarted",
            redact_secrets(&message),
            request.request_id.clone(),
            38,
            "open a new session, then repeat this call",
        );
        error.session_id = request.session_id.clone();
        error.retryable = true;
        return Response::error(request, error);
    }
    Response::error(
        request,
        ErrorObject::new(
            "engine_error",
            redact_secrets(&message),
            request.request_id.clone(),
            exit_code,
            "retry the operation or inspect web.doctor",
        ),
    )
}

fn page_state_envelope(observation: Result<serde_json::Value, String>) -> serde_json::Value {
    match observation {
        Ok(snapshot) => json!({
            "schema": "greppy.web.page-state.v1",
            "status": "available", "snapshot": snapshot,
        }),
        Err(error) => json!({
            "schema": "greppy.web.page-state.v1",
            "status": "unavailable",
            "error": {"code": "OBSERVATION_UNAVAILABLE", "message": redact_secrets(&error)},
        }),
    }
}

fn locator_error(request: &Request, message: impl Into<String>) -> Response {
    let message = redact_secrets(&message.into());
    let (code, next_action) = recovery_for_locator_error(&message);
    Response::error(
        request,
        ErrorObject::new(
            code,
            message,
            request.request_id.clone(),
            34,
            next_action,
        ),
    )
}

fn limit_error(request: &Request, message: impl Into<String>) -> Response {
    Response::error(
        request,
        ErrorObject::new(
            "resource_limit",
            redact_secrets(&message.into()),
            request.request_id.clone(),
            37,
            "close the session or raise the documented limit",
        ),
    )
}

const MODEL_TEXT_CHARS: usize = 4096;

fn model_facing_observe_payload(
    format: &str,
    content_key: &str,
    content: &str,
    manifest: &crate::artifacts::ArtifactManifest,
) -> Result<serde_json::Value, String> {
    if manifest.is_restricted() {
        let mut facing = manifest.model_facing_ref()?;
        if let Some(object) = facing.as_object_mut() {
            object.insert("format".into(), json!(format));
            object.insert(
                "untrusted_content_boundary".into(),
                json!("UNTRUSTED_PAGE_CONTENT"),
            );
        }
        return Ok(facing);
    }
    Ok(json!({
        "format": format,
        content_key: content,
        "digest": manifest.digest.hex,
        "path": manifest.object_path,
        "label": manifest.producing_operation,
        "redaction_status": manifest.redaction_status,
        "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
    }))
}

fn model_facing_source(
    mut source: serde_json::Value,
    manifest: &crate::artifacts::ArtifactManifest,
) -> Result<serde_json::Value, String> {
    let restricted = manifest.is_restricted();
    if restricted {
        let facing = manifest.model_facing_ref()?;
        if let Some(object) = source.as_object_mut() {
            object.remove("text");
            object.remove("html");
            object.remove("bytes");
            object.remove("full_text");
            object.remove("png_base64");
            object.insert("digest".into(), facing["digest"].clone());
            object.insert("path".into(), facing["path"].clone());
            object.insert("label".into(), facing["label"].clone());
            object.insert(
                "redaction_status".into(),
                facing["redaction_status"].clone(),
            );
            object.insert("text_truncated".into(), json!(false));
        }
        return Ok(source);
    }
    if let Some(text) = source
        .get("text")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
    {
        let truncated = text.chars().count() > MODEL_TEXT_CHARS;
        let snippet: String = text.chars().take(MODEL_TEXT_CHARS).collect();
        if truncated {
            let digest_ok = source
                .get("digest")
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty())
                .is_some()
                || !manifest.digest.hex.is_empty();
            if !digest_ok {
                return Err("artifact digest missing".to_owned());
            }
        }
        if let Some(object) = source.as_object_mut() {
            object.insert("text".into(), json!(snippet));
            object.insert("text_truncated".into(), json!(truncated));
            if truncated {
                object.insert("full_text".into(), json!("artifact"));
            }
            object.insert("digest".into(), json!(manifest.digest.hex));
            object.insert("path".into(), json!(manifest.object_path));
            object.insert("label".into(), json!(manifest.producing_operation));
            object.insert("redaction_status".into(), json!(manifest.redaction_status));
        }
    }
    Ok(source)
}

fn journal_line(
    event: &str,
    session_id: &str,
    run_id: &str,
    request_id: &str,
    extra: serde_json::Value,
) -> serde_json::Value {
    json!({
        "event": event,
        "session_id": session_id,
        "run_id": run_id,
        "request_id": request_id,
        "extra": extra,
    })
}

pub(crate) fn redact_secrets(input: &str) -> String {
    let mut out = input.to_owned();
    if let Some(scheme) = out.find("://") {
        let rest_at = scheme + 3;
        if let Some(at_rel) = out[rest_at..].find('@') {
            let creds = &out[rest_at..rest_at + at_rel];
            if let Some(colon) = creds.find(':') {
                let user = creds[..colon].to_owned();
                out.replace_range(rest_at..rest_at + at_rel, &format!("{user}:****"));
            }
        }
    }
    for key in [
        "password",
        "token",
        "secret",
        "authorization",
        "cookie",
        "api_key",
        "access_token",
        "refresh_token",
        "id_token",
    ] {
        let needle = format!("{key}=");
        let mut search_from = 0;
        let lower = out.to_ascii_lowercase();
        while let Some(rel) = lower[search_from..].find(&needle) {
            let start = search_from + rel + needle.len();
            let end = out[start..]
                .find(|ch: char| matches!(ch, '&' | ' ' | '"' | '\'' | '\n' | '\r'))
                .map(|idx| start + idx)
                .unwrap_or(out.len());
            out.replace_range(start..end, "****");
            search_from = start + 4;
        }
    }
    redact_labeled_secrets(&mut out);
    out
}

fn redact_labeled_secrets(out: &mut String) {
    let labels = [
        "authorization:",
        "proxy-authorization:",
        "cookie:",
        "set-cookie:",
        "bearer ",
    ];
    loop {
        let lower = out.to_ascii_lowercase();
        let mut hit: Option<(usize, usize)> = None;
        for label in labels {
            let mut search_from = 0;
            while let Some(rel) = lower[search_from..].find(label) {
                let start = search_from + rel + label.len();
                let start = start
                    + out[start..]
                        .chars()
                        .take_while(|ch| ch.is_whitespace())
                        .map(char::len_utf8)
                        .sum::<usize>();
                let rest_of_line = label.contains("cookie") || label.contains("authorization");
                let end = out[start..]
                    .find(|ch: char| {
                        if rest_of_line {
                            matches!(ch, '\n' | '\r' | '"')
                        } else {
                            matches!(ch, ' ' | '"' | '\'' | '\n' | '\r' | '&' | ',')
                        }
                    })
                    .map(|idx| start + idx)
                    .unwrap_or(out.len());
                if end > start && out[start..end] != *"****" {
                    hit = Some((start, end));
                    break;
                }
                search_from = start.max(search_from + label.len());
            }
            if hit.is_some() {
                break;
            }
        }
        match hit {
            Some((start, end)) => out.replace_range(start..end, "****"),
            None => break,
        }
    }
}

fn sensitive_header_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization"
            | "proxy-authorization"
            | "cookie"
            | "set-cookie"
            | "x-api-key"
            | "x-auth-token"
            | "x-csrf-token"
    )
}

fn sensitive_object_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "password"
            | "token"
            | "secret"
            | "authorization"
            | "cookie"
            | "cookies"
            | "credential"
            | "api_key"
            | "access_token"
            | "refresh_token"
            | "id_token"
    )
}

pub(crate) fn redact_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::String(text) => json!(redact_secrets(&text)),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(redact_json).collect())
        }
        serde_json::Value::Object(mut map) => {
            if let (Some(serde_json::Value::String(name)), Some(serde_json::Value::String(value))) =
                (map.get("name").cloned(), map.get("value").cloned())
            {
                let redacted = if sensitive_header_name(&name) {
                    "****".to_owned()
                } else {
                    redact_secrets(&value)
                };
                map.insert("value".into(), json!(redacted));
            }
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                if key == "value" && map.get("name").and_then(|item| item.as_str()).is_some() {
                    continue;
                }
                if let Some(child) = map.remove(&key) {
                    let redacted = if sensitive_object_key(&key) {
                        json!("****")
                    } else {
                        redact_json(child)
                    };
                    map.insert(key, redacted);
                }
            }
            serde_json::Value::Object(map)
        }
        other => other,
    }
}

fn redirect_chain(
    requested: &str,
    final_url: Option<&serde_json::Value>,
    requests: Option<&serde_json::Value>,
) -> Vec<String> {
    let mut chain = vec![requested.to_owned()];
    if let Some(serde_json::Value::Array(rows)) = requests {
        for row in rows {
            let Some(url) = row.get("url").and_then(|value| value.as_str()) else {
                continue;
            };
            let main = row
                .get("main_frame")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !main {
                continue;
            }
            if chain.last().map(String::as_str) != Some(url) {
                chain.push(url.to_owned());
            }
        }
    }
    if let Some(final_url) = final_url.and_then(|value| value.as_str()) {
        if chain.last().map(String::as_str) != Some(final_url) {
            chain.push(final_url.to_owned());
        }
    }
    chain
}

fn sample_rss_bytes(pid: u32) -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "rss="])
        .output();
    let Ok(output) = output else {
        return 0;
    };
    let kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0);
    kb.saturating_mul(1024)
}

/// CPU spent since this session's baseline for the given worker, resetting
/// the baseline when the worker was respawned (pid changed) so a fresh
/// process never inherits or wrongly credits another lifetime (finding 039).
fn session_cpu_delta_ms(baseline: &mut Option<(u32, u64)>, pid: u32) -> u64 {
    let now_ns = sample_cpu_ns(pid);
    match baseline {
        Some((base_pid, base_ns)) if *base_pid == pid => {
            now_ns.saturating_sub(*base_ns) / 1_000_000
        }
        _ => {
            *baseline = Some((pid, now_ns));
            0
        }
    }
}

fn sample_cpu_ms(pid: u32) -> u64 {
    sample_cpu_ns(pid) / 1_000_000
}

fn cpu_ms_since(pid: u32, baseline_ns: u64) -> u64 {
    sample_cpu_ns(pid).saturating_sub(baseline_ns) / 1_000_000
}

fn sample_cpu_ns(pid: u32) -> u64 {
    #[cfg(target_os = "macos")]
    {
        if let Some(ns) = sample_cpu_ns_rusage(pid) {
            if ns > 0 {
                return ns;
            }
        }
    }
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "time="])
        .output();
    let Ok(output) = output else {
        return 0;
    };
    parse_ps_time(String::from_utf8_lossy(&output.stdout).trim()).saturating_mul(1_000_000)
}

#[cfg(target_os = "macos")]
fn mach_timebase() -> (u64, u64) {
    static TIMEBASE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *TIMEBASE.get_or_init(|| {
        #[repr(C)]
        struct MachTimebaseInfo {
            numer: u32,
            denom: u32,
        }
        extern "C" {
            fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
        }
        let mut info = MachTimebaseInfo { numer: 1, denom: 1 };
        let rc = unsafe { mach_timebase_info(&mut info) };
        if rc != 0 || info.denom == 0 {
            (1, 1)
        } else {
            (u64::from(info.numer), u64::from(info.denom))
        }
    })
}

#[cfg(target_os = "macos")]
fn mach_ticks_to_ns(ticks: u64) -> u64 {
    let (numer, denom) = mach_timebase();
    (u128::from(ticks) * u128::from(numer) / u128::from(denom)) as u64
}

#[cfg(target_os = "macos")]
fn sample_cpu_ns_rusage(pid: u32) -> Option<u64> {
    #[repr(C)]
    struct RusageInfoV0 {
        ri_uuid: [u8; 16],
        ri_user_time: u64,
        ri_system_time: u64,
        ri_pkg_idle_wkups: u64,
        ri_interrupt_wkups: u64,
        ri_pageins: u64,
        ri_wired_size: u64,
        ri_resident_size: u64,
        ri_phys_footprint: u64,
        ri_proc_start_abstime: u64,
        ri_proc_exit_abstime: u64,
    }
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
    }
    const RUSAGE_INFO_V0: i32 = 0;
    let mut info = std::mem::MaybeUninit::<RusageInfoV0>::uninit();
    let rc = unsafe { proc_pid_rusage(pid as i32, RUSAGE_INFO_V0, info.as_mut_ptr().cast()) };
    if rc != 0 {
        return None;
    }
    let info = unsafe { info.assume_init() };
    let ticks = info.ri_user_time.saturating_add(info.ri_system_time);
    Some(mach_ticks_to_ns(ticks))
}

fn parse_ps_time(text: &str) -> u64 {
    let text = text.trim();
    if text.is_empty() {
        return 0;
    }
    let mut parts = text.split(':');
    let mut values = [0_f64; 3];
    let mut count = 0;
    for part in parts.by_ref() {
        if count >= 3 {
            break;
        }
        values[count] = part.parse().unwrap_or(0.0);
        count += 1;
    }
    let seconds = match count {
        1 => values[0],
        2 => values[0] * 60.0 + values[1],
        3 => values[0] * 3600.0 + values[1] * 60.0 + values[2],
        _ => 0.0,
    };
    (seconds * 1000.0) as u64
}

fn gate_session_engine(
    sessions: &mut HashMap<String, Session>,
    session_id: &str,
    content_pid: u32,
    controller_pid: u32,
    content_cpu_baseline_ns: u64,
    controller_cpu_baseline_ns: u64,
    method: &str,
    _params: &serde_json::Value,
) -> Result<(), String> {
    let Some(session) = sessions.get_mut(session_id) else {
        return Err("session was closed".to_owned());
    };
    session.limits.check_wall_time(session.started.elapsed())?;
    let content_rss = sample_rss_bytes(content_pid);
    session.peak_rss_bytes = session.peak_rss_bytes.max(content_rss);
    session.limits.check_content_rss(content_rss)?;
    session
        .limits
        .check_controller_memory(sample_rss_bytes(controller_pid))?;
    session.limits.check_cpu_time(
        Duration::from_millis(cpu_ms_since(content_pid, content_cpu_baseline_ns)),
        session.limits.content_cpu_time,
        "content",
    )?;
    session.limits.check_cpu_time(
        Duration::from_millis(cpu_ms_since(controller_pid, controller_cpu_baseline_ns)),
        session.limits.controller_cpu_time,
        "controller",
    )?;
    match method {
        "browser.newContext" => {
            session
                .limits
                .check_contexts(session.contexts.saturating_add(1))?;
            session.contexts = session.contexts.saturating_add(1);
        }
        "context.newPage" | "session.ensurePage" => {
            session
                .limits
                .check_pages(session.pages.saturating_add(1))?;
            session.pages = session.pages.saturating_add(1);
        }
        "page.goto" | "page.reload" | "page.goBack" | "page.goForward" | "page.frameGoto" => {
            session
                .limits
                .check_requests(session.requests.saturating_add(1))?;
            session
                .limits
                .check_network_bytes(session.network_bytes, 4096)?;
            session.requests = session.requests.saturating_add(1);
            session.network_bytes = session.network_bytes.saturating_add(4096);
        }
        _ => {}
    }
    Ok(())
}

struct SessionEngineGate<'a> {
    sessions: &'a mut HashMap<String, Session>,
    session_id: String,
    content_pid: u32,
    controller_pid: u32,
    content_cpu_baseline_ns: u64,
    controller_cpu_baseline_ns: u64,
    operation_id: String,
    control: Arc<RunControl>,
}

impl crate::supervisor::EngineGate for SessionEngineGate<'_> {
    fn is_cancelled(&self) -> bool {
        self.control
            .cancel_target
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|pair| pair.0 == self.session_id && pair.1 == self.operation_id)
    }

    fn poll_control(&mut self) {
        let mut beats = self
            .control
            .heartbeats
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for session_id in beats.drain(..) {
            if let Some(session) = self.sessions.get_mut(&session_id) {
                session.last_heartbeat = Instant::now();
            }
        }
        drop(beats);
        self.control
            .publish_sessions(snapshot_session_rows(self.sessions, &self.control));
    }

    fn note_inflight_engine(&mut self, request_id: u64, method: &str) {
        if let Some(session) = self.sessions.get_mut(&self.session_id) {
            session.inflight_engine_request_id = Some(request_id);
            session.inflight_engine_method = Some(method.to_owned());
        }
        self.control
            .publish_sessions(snapshot_session_rows(self.sessions, &self.control));
    }

    fn note_discarded_engine_result(&mut self, request_id: u64, ok: bool, error: Option<String>) {
        self.control
            .discarded_engine_results
            .fetch_add(1, Ordering::Relaxed);
        *self
            .control
            .late_engine_result
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(LateEngineResult {
            session_id: self.session_id.clone(),
            target_request_id: self.operation_id.clone(),
            engine_request_id: request_id,
            ok,
            error,
        });
        if let Some(session) = self.sessions.get_mut(&self.session_id) {
            session.discarded_engine_results = session.discarded_engine_results.saturating_add(1);
        }
    }

    fn note_discarded_count(&mut self, n: u64) {
        if n == 0 {
            return;
        }
        self.control
            .discarded_engine_results
            .fetch_add(n, Ordering::Relaxed);
        if let Some(session) = self.sessions.get_mut(&self.session_id) {
            session.discarded_engine_results = session.discarded_engine_results.saturating_add(n);
        }
    }

    fn inflight_engine_id(&self) -> Option<u64> {
        self.sessions
            .get(&self.session_id)
            .and_then(|session| session.inflight_engine_request_id)
    }

    fn before_call(&mut self, method: &str, params: &serde_json::Value) -> Result<(), String> {
        self.poll_control();
        if self.is_cancelled() {
            return Err("cancelled".into());
        }
        gate_session_engine(
            self.sessions,
            &self.session_id,
            self.content_pid,
            self.controller_pid,
            self.content_cpu_baseline_ns,
            self.controller_cpu_baseline_ns,
            method,
            params,
        )
    }

    fn after_records(&mut self, records: &serde_json::Value) -> Result<(), String> {
        apply_record_limits(self.sessions, &self.session_id, records)
    }
}

fn apply_record_limits(
    sessions: &mut HashMap<String, Session>,
    session_id: &str,
    records: &serde_json::Value,
) -> Result<(), String> {
    let Some(session) = sessions.get_mut(session_id) else {
        return Err("session was closed".to_owned());
    };
    let console_bytes = records
        .get("messages")
        .and_then(|value| value.as_array())
        .map(|rows| sum_text_bytes(rows))
        .unwrap_or(0);
    let download_bytes = records
        .get("downloads")
        .and_then(|value| value.as_array())
        .map(|rows| sum_byte_lengths(rows))
        .unwrap_or(0);
    session.limits.check_console_bytes(0, console_bytes)?;
    session.limits.check_download_bytes(0, download_bytes)?;
    session.console_bytes = console_bytes;
    session.download_bytes = download_bytes;
    let recorded_network = records
        .get("responses")
        .and_then(|value| value.as_array())
        .map(|rows| sum_byte_lengths(rows))
        .unwrap_or(0);
    if recorded_network > 0 {
        let accounted = session.network_bytes.max(recorded_network);
        session.limits.check_network_bytes(0, accounted)?;
        session.network_bytes = accounted;
    }
    Ok(())
}

fn sum_text_bytes(rows: &[serde_json::Value]) -> u64 {
    rows.iter()
        .map(|row| {
            row.get("text")
                .and_then(|value| value.as_str())
                .map(|text| text.len() as u64)
                .unwrap_or(0)
        })
        .sum()
}

fn sum_byte_lengths(rows: &[serde_json::Value]) -> u64 {
    rows.iter()
        .map(|row| {
            row.get("byteLength")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
        })
        .sum()
}

impl Daemon {
    fn attach_session_metrics(&self, session_id: &str, response: &mut Response) {
        if let Some(session) = self.sessions.get(session_id) {
            response.metrics.network_bytes = session.network_bytes;
            response.metrics.peak_rss_bytes = session.peak_rss_bytes;
        }
        response.metrics.content_cpu_ms = sample_cpu_ms(self.content.pid());
        response.metrics.controller_cpu_ms = sample_cpu_ms(self.controller.pid());
        if response.metrics.peak_rss_bytes == 0 {
            response.metrics.peak_rss_bytes = sample_rss_bytes(self.content.pid());
        }
    }
}

fn random_token() -> io::Result<String> {
    use std::io::Read;
    let mut bytes = [0_u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}


fn request_agent_id(request: &Request) -> Option<String> {
    request
        .payload
        .get("agent_id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub fn socket_exists(path: &Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod script_stage_tests {
    use super::{
        bind_socket_healing_stale, copy_granted_modules, isolated_id, path_is_within_root,
        refuse_unbounded_script_root, remove_script_stage, script_stage_dir,
        stage_script_for_controller,
    };
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};

    fn unique_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join("greppy-stage-unit")
            .join("bounded")
            .join(format!("r{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn parse_result_cursor_reads_digest_and_offset() {
        let digest = "ab".repeat(32);
        let (got, offset) = super::parse_result_cursor(&format!("sha256:{digest}:12")).unwrap();
        assert_eq!(got, digest);
        assert_eq!(offset, 12);
        assert!(super::parse_result_cursor("offset=12").is_err());
        assert!(super::parse_result_cursor("sha256:short:0").is_err());
        assert!(super::parse_result_cursor(&format!("sha256:{digest}:x")).is_err());
    }

    #[test]
    fn isolated_id_rejects_path_escape() {
        assert!(isolated_id("wrs_abc").is_ok());
        assert!(isolated_id("../etc").is_err());
        assert!(isolated_id("a/b").is_err());
        assert!(isolated_id("").is_err());
    }

    #[test]
    fn refuse_unbounded_script_root_rejects_system_and_home() {
        assert!(refuse_unbounded_script_root(Path::new("/")).is_err());
        if Path::new("/etc").exists() {
            assert!(refuse_unbounded_script_root(Path::new("/etc")).is_err());
        }
        if let Ok(home) = std::env::var("HOME") {
            if Path::new(&home).exists() {
                assert!(refuse_unbounded_script_root(Path::new(&home)).is_err());
            }
        }
        let root = unique_root("refuse");
        assert!(
            refuse_unbounded_script_root(&root).is_ok(),
            "{}",
            root.display()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stage_uses_isolated_temp_and_skips_symlink_escape() {
        let root = unique_root("stage");
        fs::write(root.join("entry.mjs"), "export const n = 1;\n").unwrap();
        fs::write(root.join("helper.mjs"), "export const n = 2;\n").unwrap();
        let secret = root.join("secret.mjs");
        let _ = fs::remove_file(&secret);
        symlink("/etc/passwd", &secret).unwrap();

        let staged = stage_script_for_controller(
            root.join("entry.mjs").to_str().unwrap(),
            "run_stage",
            "wrs_stage1",
            "wrq_stage1",
        )
        .expect("stage");
        let staged = PathBuf::from(staged);
        let expected = script_stage_dir("run_stage", "wrs_stage1", "wrq_stage1").unwrap();
        assert!(
            path_is_within_root(&expected, &staged.canonicalize().unwrap()),
            "{}",
            staged.display()
        );
        assert!(expected.join("helper.mjs").is_file());
        assert!(
            !expected.join("secret.mjs").exists(),
            "symlink escape must not be staged"
        );
        remove_script_stage("run_stage", "wrs_stage1", Some("wrq_stage1"));
        assert!(!expected.exists(), "per-request stage must be cleaned up");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn copy_granted_modules_fails_closed_on_walk_escape() {
        let root = unique_root("walk");
        let outside = std::env::temp_dir()
            .join("greppy-stage-unit")
            .join("outside")
            .join(format!("r{}", std::process::id()));
        let _ = fs::remove_dir_all(&outside);
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("escape.mjs"), "export const n = 0;\n").unwrap();
        let dest = unique_root("walk-dest");
        let err = copy_granted_modules(&root, &outside, &dest, &mut 0, &mut 0).unwrap_err();
        assert!(err.contains("escaped granted root"), "{err}");
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(outside);
        let _ = fs::remove_dir_all(dest);
    }

    #[test]
    fn socket_bind_heals_only_a_stale_path() {
        let root = unique_root("socket");
        let path = root.join("runtime.sock");

        let live = UnixListener::bind(&path).unwrap();
        let error = bind_socket_healing_stale(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        assert!(
            UnixStream::connect(&path).is_ok(),
            "live listener was displaced"
        );

        drop(live);
        let healed = bind_socket_healing_stale(&path).expect("stale socket should be replaced");
        assert!(
            UnixStream::connect(&path).is_ok(),
            "healed listener is not live"
        );
        drop(healed);
        let _ = fs::remove_dir_all(root);
    }
}
#[cfg(test)]
mod redirect_chain_tests {
    use super::redirect_chain;
    use serde_json::json;

    #[test]
    fn missing_target_guidance_differs_from_ambiguous_target_guidance() {
        let request = super::Request::new("locator-diagnostic", "web.click", json!({}));
        let missing = super::locator_error(
            &request, "timed out waiting for actionable locator target (failed_check=attached; count=0)",
        );
        assert_eq!(missing.status, "error");
        let error = missing.error.unwrap();
        assert_eq!(error.code, "NO_MATCH");
        assert_eq!(error.exit_code, 34);
        assert!(!error.retryable);
        assert!(error.next_action.contains("no target matched"));
        assert!(!error.next_action.contains("narrow"));
        let ambiguous = super::locator_error(&request, "strict mode: selector matched 2 nodes");
        let error = ambiguous.error.unwrap();
        assert_eq!(error.code, "AMBIGUOUS_TARGET");
        assert_eq!(error.exit_code, 34);
        assert!(error.next_action.contains("narrow"));
    }

    #[test]
    fn failed_action_observation_cannot_extend_its_request_budget() {
        use std::time::Duration;
        assert_eq!(super::failure_observation_budget(30_000, Duration::ZERO), Duration::from_secs(2));
        assert_eq!(super::failure_observation_budget(1_000, Duration::from_millis(750)), Duration::from_millis(250));
        assert_eq!(super::failure_observation_budget(1_000, Duration::from_secs(1)), Duration::ZERO);
        assert_eq!(super::failure_observation_budget(1_000, Duration::from_secs(2)), Duration::ZERO);
        assert_eq!(super::failure_observation_budget(0, Duration::ZERO), Duration::ZERO);
    }

    #[test]
    fn policy_denial_has_nonretrying_profile_guidance() {
        let mut request = super::Request::new("policy-test", "web.open", json!({}));
        request.session_id = Some("wrs_policy_test".into());
        let response = super::engine_error(
            &request,
            "policy_denied: research profile denies loopback and private networks",
            34,
        );
        let error = response.error.unwrap();
        assert_eq!(error.code, "policy_denied");
        assert_eq!(error.exit_code, 34);
        assert!(!error.retryable);
        assert_eq!(error.session_id, request.session_id);
        assert!(error
            .next_action
            .contains("session create --profile project --json"));
        assert!(error.next_action.contains("explicitly requested local"));
        assert!(error
            .next_action
            .contains("LAN and cloud metadata remain blocked"));
        assert!(!error.next_action.contains("retry the operation"));
    }

    #[test]
    fn policy_denial_never_suggests_project_as_metadata_bypass() {
        for reason in [
            "cloud metadata endpoint denied",
            "LAN and non-public endpoints denied",
        ] {
            let request = super::Request::new("policy-test", "web.open", json!({}));
            let response = super::engine_error(&request, format!("policy_denied: {reason}"), 34);
            let error = response.error.unwrap();
            assert_eq!(error.code, "policy_denied");
            assert!(!error.retryable);
            assert!(!error.next_action.contains("session create"));
            assert!(error.next_action.contains("permitted public"));
        }
    }

    #[test]
    fn parse_ps_time_minutes_and_seconds() {
        assert_eq!(super::parse_ps_time("0:01.50"), 1500);
        assert_eq!(super::parse_ps_time("1:00.00"), 60_000);
        assert_eq!(super::parse_ps_time("0:00.15"), 150);
        assert_eq!(super::parse_ps_time("2:26.50"), 146_500);
        assert_eq!(super::parse_ps_time("1:02:03"), 3_723_000);
        assert_eq!(super::parse_ps_time(""), 0);
    }

    #[test]
    fn sample_cpu_ms_sees_this_process() {
        let start = std::time::Instant::now();
        let mut acc = 0_u64;
        while start.elapsed() < std::time::Duration::from_millis(30) {
            acc = acc.wrapping_add((0..10_000).sum());
        }
        std::hint::black_box(acc);
        let ns = super::sample_cpu_ns(std::process::id());
        assert!(
            ns > 1_000_000,
            "this process should have >1ms CPU after a 30ms spin, got {ns} ns"
        );
        let ms = super::sample_cpu_ms(std::process::id());
        assert!(ms > 0, "this process should have nonzero CPU, got {ms}");
    }

    #[test]
    fn session_cpu_delta_starts_at_zero_and_resets_for_a_new_worker() {
        let pid = std::process::id();
        let mut baseline = None;
        assert_eq!(super::session_cpu_delta_ms(&mut baseline, pid), 0);
        let start = std::time::Instant::now();
        while start.elapsed() < std::time::Duration::from_millis(20) {
            std::hint::black_box((0..10_000_u64).sum::<u64>());
        }
        assert!(super::session_cpu_delta_ms(&mut baseline, pid) > 0);
        let replacement_pid = pid.saturating_add(1);
        assert_eq!(
            super::session_cpu_delta_ms(&mut baseline, replacement_pid),
            0,
            "a replacement worker must not inherit its predecessor's CPU"
        );
        assert_eq!(baseline.map(|entry| entry.0), Some(replacement_pid));
    }

    #[test]
    fn recorded_main_frame_hops_are_kept() {
        let requests = json!([
            {"url": "http://example.test/start", "main_frame": true},
            {"url": "http://example.test/asset.css", "main_frame": false},
            {"url": "http://example.test/end", "main_frame": true, "redirect": true}
        ]);
        let chain = redirect_chain(
            "http://example.test/start",
            Some(&json!("http://example.test/end")),
            Some(&requests),
        );
        assert_eq!(
            chain,
            vec![
                "http://example.test/start".to_owned(),
                "http://example.test/end".to_owned()
            ]
        );
    }

    #[test]
    fn final_url_is_appended_when_requests_are_missing() {
        let chain = redirect_chain(
            "http://example.test/start",
            Some(&json!("http://example.test/end")),
            None,
        );
        assert_eq!(
            chain,
            vec![
                "http://example.test/start".to_owned(),
                "http://example.test/end".to_owned()
            ]
        );
    }

    #[test]
    fn redact_secrets_masks_userinfo_and_password_query() {
        assert_eq!(
            super::redact_secrets("https://alice:s3cret@example.test/x"),
            "https://alice:****@example.test/x"
        );
        let masked = super::redact_secrets("http://example.test/?password=s3cret&q=1");
        assert!(!masked.contains("s3cret"), "{masked}");
        assert!(masked.contains("password=****"), "{masked}");
    }

    #[test]
    fn redact_secrets_masks_authorization_cookie_and_bearer() {
        for sample in [
            "Authorization: Bearer s3cret",
            "Cookie: session=s3cret; theme=dark",
            "Set-Cookie: id=s3cret; HttpOnly",
            "proxy-authorization: Basic s3cret",
            "token leaked as Bearer s3cret in log",
            "https://example.test/?access_token=s3cret",
        ] {
            let masked = super::redact_secrets(sample);
            assert!(!masked.contains("s3cret"), "{sample} -> {masked}");
        }
    }

    #[test]
    fn redact_json_masks_header_objects_and_credential_keys() {
        let masked = super::redact_json(json!({
            "requests": [{
                "url": "https://alice:s3cret@example.test/",
                "headers": [
                    {"name": "Authorization", "value": "Bearer s3cret"},
                    {"name": "Accept", "value": "text/html"}
                ]
            }],
            "cookies": [{"value": "s3cret"}]
        }));
        let dumped = masked.to_string();
        assert!(!dumped.contains("s3cret"), "{dumped}");
        assert!(dumped.contains("Accept"), "{dumped}");
        assert!(dumped.contains("text/html"), "{dumped}");
    }

    #[test]
    fn journal_line_includes_request_id_for_correlation() {
        let line = super::journal_line(
            "session.ready",
            "wrs_1",
            "run_x",
            "wrq_9",
            json!({ "profile": "project" }),
        );
        assert_eq!(line["event"], "session.ready");
        assert_eq!(line["session_id"], "wrs_1");
        assert_eq!(line["run_id"], "run_x");
        assert_eq!(line["request_id"], "wrq_9");
        assert_eq!(line["extra"]["profile"], "project");
    }

    fn sample_manifest(sensitive: bool, digest_hex: &str) -> crate::artifacts::ArtifactManifest {
        crate::artifacts::ArtifactManifest {
            contract: "greppy.web-runtime.artifact-manifest.v1".to_owned(),
            digest: crate::artifacts::DigestFields {
                algorithm: "sha256".to_owned(),
                hex: digest_hex.to_owned(),
            },
            byte_count: 12,
            media_type: "text/plain".to_owned(),
            producing_operation: "web.read".to_owned(),
            session_id: "wrs_1".to_owned(),
            run_id: "run".to_owned(),
            timestamp: "0.000Z".to_owned(),
            redaction_status: if sensitive {
                "redacted_for_model".to_owned()
            } else {
                "not_redacted".to_owned()
            },
            sensitive,
            credential_labeled: sensitive,
            object_path: format!("objects/sha256/{digest_hex}"),
        }
    }

    #[test]
    fn model_facing_source_truncates_long_text() {
        let long = "x".repeat(5000);
        let manifest = sample_manifest(false, "abc");
        let compact = super::model_facing_source(
            json!({
                "text": long,
                "digest": "abc"
            }),
            &manifest,
        )
        .unwrap();
        assert_eq!(compact["text_truncated"], true);
        assert_eq!(compact["text"].as_str().unwrap().chars().count(), 4096);
        assert_eq!(compact["full_text"], "artifact");
        assert_eq!(compact["digest"], "abc");
        assert_eq!(compact["path"], "objects/sha256/abc");
        assert_eq!(compact["label"], "web.read");
        assert_eq!(compact["redaction_status"], "not_redacted");
    }

    #[test]
    fn model_facing_source_omits_sensitive_bytes_from_payload() {
        let secret = "SUPER_SECRET_TOKEN_value=hunter2";
        let digest = crate::artifacts::hex_sha256(secret.as_bytes());
        let manifest = sample_manifest(true, &digest);
        let compact = super::model_facing_source(
            json!({
                "text": secret,
                "html": format!("<p>{secret}</p>"),
                "full_text": secret,
                "bytes": secret,
                "digest": digest,
                "title": "ok",
            }),
            &manifest,
        )
        .unwrap();
        let dumped = compact.to_string();
        assert!(
            !dumped.contains("SUPER_SECRET_TOKEN"),
            "sensitive bytes leaked into model-facing payload: {dumped}"
        );
        assert!(
            !dumped.contains("hunter2"),
            "sensitive bytes leaked into model-facing payload: {dumped}"
        );
        assert!(compact.get("text").is_none(), "{compact:?}");
        assert!(compact.get("html").is_none(), "{compact:?}");
        assert!(compact.get("full_text").is_none(), "{compact:?}");
        assert!(compact.get("bytes").is_none(), "{compact:?}");
        assert_eq!(compact["digest"], digest);
        assert_eq!(compact["path"], format!("objects/sha256/{digest}"));
        assert_eq!(compact["label"], "web.read");
        assert_eq!(compact["redaction_status"], "redacted_for_model");
        assert_eq!(compact["title"], "ok");
    }

    #[test]
    fn model_facing_observe_payload_omits_sensitive_html() {
        let secret = "observe-secret-cookie=s3cret";
        let digest = crate::artifacts::hex_sha256(secret.as_bytes());
        let manifest = sample_manifest(true, &digest);
        let payload =
            super::model_facing_observe_payload("html", "html", secret, &manifest).unwrap();
        let dumped = payload.to_string();
        assert!(!dumped.contains("s3cret"), "{dumped}");
        assert!(!dumped.contains("observe-secret-cookie"), "{dumped}");
        assert!(payload.get("html").is_none(), "{payload:?}");
        assert_eq!(payload["format"], "html");
        assert_eq!(payload["digest"], digest);
        assert_eq!(payload["path"], format!("objects/sha256/{digest}"));
        assert_eq!(payload["label"], "web.read");
        assert_eq!(payload["redaction_status"], "redacted_for_model");
    }

    #[test]
    fn model_facing_source_fails_closed_without_digest_when_sensitive() {
        let manifest = sample_manifest(true, "");
        let err = super::model_facing_source(json!({ "text": "secret", "digest": "" }), &manifest)
            .unwrap_err();
        assert!(err.contains("digest"), "{err}");
    }

    #[test]
    fn html_from_page_content_rejects_missing_html() {
        assert_eq!(
            super::Daemon::html_from_page_content(&json!({"html": "<p>ok</p>"})).unwrap(),
            "<p>ok</p>"
        );
        assert_eq!(
            super::Daemon::html_from_page_content(&json!({})).unwrap_err(),
            "page.content missing html"
        );
        assert_eq!(
            super::Daemon::html_from_page_content(&json!({"html": null})).unwrap_err(),
            "page.content missing html"
        );
    }
}
