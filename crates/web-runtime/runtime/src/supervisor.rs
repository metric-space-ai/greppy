use crate::policy::{decide_url, NetworkProfile, UrlDecision};
use crate::protocol::{read_message, Message, WorkerKind, MAX_FRAME_BYTES};
use std::collections::{HashMap, VecDeque};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};

fn emit_line(line: &str) {
    println!("{line}");
    let _ = io::stdout().flush();
}
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};


/// Emit one lifecycle line, but only when someone asked for them.
///
/// These nine lines per session -- spawn, two handshakes, two readies,
/// workers-ready, bind-socket, listening, request-ready -- go to stderr, and
/// stderr is whatever the parent handed over. Under `cargo test` that is a
/// pipe; a pipe nobody drains fills after 64 KB, and the next write blocks the
/// supervisor for good. It then answers nothing, including
/// `web.session.close`, which is how a 1000-cycle run died at cycle 37, 262,
/// 270 or 74 depending on timing.
///
/// Proven by isolation: same driver, stderr to a file 400 cycles clean, stderr
/// to an unread pipe stalls at 170.
///
/// Setting O_NONBLOCK on fd 2 also fixes it and was measured at six times the
/// runtime -- the flag is process-wide and something retries failed writes. So
/// the cure is to not write them: diagnostics nobody reads should not exist in
/// normal operation. `GREPPY_WEB_TRACE_PHASE=1` brings them back.
pub(crate) fn phase_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GREPPY_WEB_TRACE_PHASE").is_some())
}

macro_rules! phase {
    ($($arg:tt)*) => {
        if crate::supervisor::phase_trace_enabled() {
            eprintln!($($arg)*);
        }
    };
}

const STDOUT_LOG_CAP: usize = 256 * 1024;

const OWNED_WORKER_SLOTS: usize = 8;
static OWNED_WORKER_PGIDS: [AtomicU32; OWNED_WORKER_SLOTS] =
    [const { AtomicU32::new(0) }; OWNED_WORKER_SLOTS];

fn register_owned_worker(pid: u32) {
    for slot in &OWNED_WORKER_PGIDS {
        if slot
            .compare_exchange(0, pid, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

fn unregister_owned_worker(pid: u32) {
    for slot in &OWNED_WORKER_PGIDS {
        let _ = slot.compare_exchange(pid, 0, Ordering::Relaxed, Ordering::Relaxed);
    }
}

#[cfg(unix)]
pub(crate) extern "C" fn handle_supervisor_sigterm(_sig: i32) {
    for slot in &OWNED_WORKER_PGIDS {
        let pid = slot.swap(0, Ordering::Relaxed);
        if pid != 0 {
            let raw = pid as i32;
            unsafe {
                libc::kill(-raw, libc::SIGKILL);
                libc::kill(raw, libc::SIGKILL);
            }
        }
    }
    unsafe { libc::_exit(143) };
}

const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const SCRIPT_TIMEOUT: Duration = Duration::from_secs(120);
const REAP_POLL_INTERVAL: Duration = Duration::from_millis(10);
const WORKER_EOF_WAIT: Duration = Duration::from_secs(2);
const WORKER_REAP_WAIT: Duration = Duration::from_secs(2);
const TIMEOUT_REAP_WAIT: Duration = Duration::from_millis(100);
const READER_JOIN_WAIT: Duration = Duration::from_millis(250);

#[derive(Debug, Eq, PartialEq)]
pub struct Config {
    pub scripts: Vec<PathBuf>,
    pub fixture_url: Option<String>,
    pub search_endpoint: Option<String>,
    pub socket: Option<PathBuf>,
    pub run_id: Option<String>,
    pub idle_ttl: Option<Duration>,
}

impl Config {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, String> {
        let mut dist = None;
        let mut scripts = Vec::new();
        let mut fixture_url = None;
        let mut search_endpoint = None;
        let mut socket = None;
        let mut run_id = None;
        let mut idle_ttl = None;
        let mut args = args.into_iter();

        while let Some(argument) = args.next() {
            match argument.to_str() {
                Some("--controller-worker") | Some("--content-worker") => {
                    return Err(
                        "separate worker images are not supported; this binary re-execs itself with --internal-role"
                            .to_owned(),
                    );
                }
                Some("--dist") => {
                    set_path(&mut dist, "--dist", args.next())?;
                }
                Some("--script") => {
                    let mut script = None;
                    set_path(&mut script, "--script", args.next())?;
                    scripts.push(script.expect("script path"));
                }
                Some("--fixture-url") => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value after --fixture-url".to_owned())?;
                    let value = value
                        .into_string()
                        .map_err(|value| format!("invalid --fixture-url {value:?}"))?;
                    if value.is_empty() {
                        return Err("empty value after --fixture-url".to_owned());
                    }
                    if fixture_url.replace(value).is_some() {
                        return Err("duplicate --fixture-url".to_owned());
                    }
                }
                Some("--search-endpoint") => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value after --search-endpoint".to_owned())?;
                    let value = value
                        .into_string()
                        .map_err(|value| format!("invalid --search-endpoint {value:?}"))?;
                    if value.is_empty() {
                        return Err("empty value after --search-endpoint".to_owned());
                    }
                    if search_endpoint.replace(value).is_some() {
                        return Err("duplicate --search-endpoint".to_owned());
                    }
                }
                Some("--socket") => {
                    set_path(&mut socket, "--socket", args.next())?;
                }
                Some("--run-id") => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value after --run-id".to_owned())?;
                    let value = value
                        .into_string()
                        .map_err(|value| format!("invalid --run-id {value:?}"))?;
                    if value.is_empty() {
                        return Err("empty value after --run-id".to_owned());
                    }
                    if run_id.replace(value).is_some() {
                        return Err("duplicate --run-id".to_owned());
                    }
                }
                Some("--idle-ttl-ms") => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value after --idle-ttl-ms".to_owned())?;
                    let value = value
                        .into_string()
                        .map_err(|value| format!("invalid --idle-ttl-ms {value:?}"))?;
                    let parsed = value.parse::<u64>().map_err(|_| {
                        format!("invalid --idle-ttl-ms {value}")
                    })?;
                    let ttl = Duration::from_millis(parsed.clamp(20, 3_600_000));
                    if idle_ttl.replace(ttl).is_some() {
                        return Err("duplicate --idle-ttl-ms".to_owned());
                    }
                }
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }

        if let Some(dist) = dist {
            runtime_from_dist(&dist)?;
        }
        Ok(Self {
            scripts,
            fixture_url,
            search_endpoint,
            socket,
            run_id,
            idle_ttl,
        })
    }
}

fn is_symlink(path: &Path) -> bool {
    path.symlink_metadata()
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false)
}

pub(crate) fn runtime_from_dist(dist: &Path) -> Result<PathBuf, String> {
    if is_symlink(dist) {
        return Err(format!("refusing symlink dist: {}", dist.display()));
    }
    let stamp = dist.join(".greppy-web-runtime-dist");
    if is_symlink(&stamp) || !stamp.is_file() {
        return Err(format!(
            "not a stamped web-runtime dist: {}",
            dist.display()
        ));
    }
    let bin = dist.join("bin");
    if is_symlink(&bin) || !bin.is_dir() {
        return Err(format!(
            "missing real dist bin directory: {}",
            bin.display()
        ));
    }
    let runtime = bin.join("web-runtime");
    if is_symlink(&runtime) {
        return Err(format!(
            "refusing symlink runtime executable: {}",
            runtime.display()
        ));
    }
    if !runtime.is_file() {
        return Err(format!(
            "missing web-runtime executable in dist: {}",
            runtime.display()
        ));
    }
    Ok(runtime)
}

fn set_path(
    destination: &mut Option<PathBuf>,
    option: &str,
    value: Option<OsString>,
) -> Result<(), String> {
    if destination.is_some() {
        return Err(format!("duplicate {option}"));
    }
    let value = value.ok_or_else(|| format!("missing path after {option}"))?;
    if value.is_empty() {
        return Err(format!("empty path after {option}"));
    }
    *destination = Some(value.into());
    Ok(())
}

pub fn run(config: Config) -> io::Result<()> {
    if let Some(socket) = config.socket.clone() {
        let run_id = config
            .run_id
            .clone()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing --run-id"))?;
        #[cfg(unix)]
        {
            return crate::daemon::serve(crate::daemon::DaemonConfig {
                socket,
                run_id,
                fixture_url: config.fixture_url,
                search_endpoint: config.search_endpoint,
                idle_ttl: config
                    .idle_ttl
                    .unwrap_or(Duration::from_secs(5 * 60)),
            });
        }
        #[cfg(not(unix))]
        {
            let _ = (socket, run_id);
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "web-runtime supervisor socket mode requires Unix domain sockets",
            ));
        }
    }

    let mut controller = WorkerProcess::spawn(WorkerKind::Controller, random_capability()?)?;
    controller.handshake()?;
    emit_line("web_runtime.controller=ready");

    let mut content = WorkerProcess::spawn(WorkerKind::Content, random_capability()?)?;
    content.handshake()?;
    emit_line("web_runtime.content=ready");

    for script in &config.scripts {
        let fixture_url = config.fixture_url.clone().unwrap_or_default();
        run_script(&mut controller, &mut content, script, fixture_url)?;
        emit_line("web_runtime.script=ok");
    }

    content.shutdown()?;
    emit_line("web_runtime.content=stopped");
    controller.shutdown()?;
    emit_line("web_runtime.controller=stopped");
    emit_line("web_runtime.supervisor=stopped");
    Ok(())
}

fn random_capability() -> io::Result<String> {
    let mut bytes = [0_u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn run_script(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    script: &Path,
    fixture_url: String,
) -> io::Result<()> {
    let source = std::fs::read_to_string(script).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to read script {}: {error}", script.display()),
        )
    })?;
    let specifier = script
        .canonicalize()
        .unwrap_or_else(|_| script.to_path_buf())
        .to_string_lossy()
        .into_owned();
    if fixture_url_grants_project_loopback(&fixture_url) {
        let mut sidecar = 0;
        let mut parked = VecDeque::new();
        sidecar_engine_call(
            content,
            &mut sidecar,
            "session.setProfile",
            serde_json::json!({ "profile": "project" }),
            PROCESS_TIMEOUT,
            &mut parked,
        )
        .map_err(|error| io::Error::other(error))?;
        emit_line("web_runtime.local_origin_grant=project");
    }
    controller.send(&Message::run_script(specifier, source, fixture_url))?;
    route_until_script_complete(controller, content, SCRIPT_TIMEOUT).map(|_| ())
}

fn fixture_url_grants_project_loopback(fixture_url: &str) -> bool {
    if fixture_url.is_empty() {
        return false;
    }
    matches!(
        decide_url(NetworkProfile::Research, fixture_url),
        UrlDecision::Deny { .. }
    ) && matches!(
        decide_url(NetworkProfile::Project, fixture_url),
        UrlDecision::Allow
    )
}

pub(crate) trait EngineGate {
    fn before_call(&mut self, method: &str, params: &serde_json::Value) -> Result<(), String>;
    fn after_records(&mut self, records: &serde_json::Value) -> Result<(), String> {
        let _ = records;
        Ok(())
    }
    fn is_cancelled(&self) -> bool {
        false
    }
    fn poll_control(&mut self) {}
    fn note_inflight_engine(&mut self, request_id: u64, method: &str) {
        let _ = (request_id, method);
    }
    fn note_discarded_engine_result(&mut self, request_id: u64, ok: bool, error: Option<String>) {
        let _ = (request_id, ok, error);
    }
    fn note_discarded_count(&mut self, n: u64) {
        let _ = n;
    }
    fn inflight_engine_id(&self) -> Option<u64> {
        None
    }
}

struct AllowAllGate;

impl EngineGate for AllowAllGate {
    fn before_call(&mut self, _method: &str, _params: &serde_json::Value) -> Result<(), String> {
        Ok(())
    }
}

pub(crate) fn route_until_script_complete(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    timeout: Duration,
) -> io::Result<serde_json::Value> {
    route_until_script_complete_gated(controller, content, timeout, AllowAllGate)
}

pub(crate) fn route_until_script_complete_gated(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    timeout: Duration,
    mut gate: impl EngineGate,
) -> io::Result<serde_json::Value> {
    let deadline = Instant::now() + timeout;
    let mut sidecar = 1_u64 << 62;
    let mut pending: HashMap<u64, (String, serde_json::Value)> = HashMap::new();
    let mut parked: VecDeque<Message> = VecDeque::new();
    let mut wait_point = "controller:script-complete".to_owned();
    let stale = content.discard_stale_engine_results();
    if stale > 0 {
        phase!("web-runtime: drained {stale} stale content messages before script routing");
        gate.note_discarded_count(stale);
    }
    loop {
        gate.poll_control();
        if gate.is_cancelled() {
            loop {
                match content.messages.try_recv() {
                    Ok(Ok(crate::protocol::Message::EngineResult {
                        request_id,
                        ok,
                        error,
                        ..
                    })) => {
                        if pending.remove(&request_id).is_some() {
                            gate.note_discarded_engine_result(request_id, ok, error);
                        }
                        if pending.is_empty() {
                            break;
                        }
                    }
                    Ok(Ok(_)) => continue,
                    Ok(Err(_)) | Err(_) => break,
                }
            }
            return Err(io::Error::other("cancelled"));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            phase!("web-runtime: phase run-timeout wait={wait_point}");
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "timed out after {timeout:?} running controller script (wait={wait_point})"
                ),
            ));
        }
        let incoming = if let Some(message) = parked.pop_front() {
            Ok(Incoming::Content(message))
        } else {
            let slice = remaining.min(Duration::from_millis(50));
            match recv_any(controller, content, slice) {
                Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                    continue;
                }
                other => other,
            }
        };
        match incoming {
            Err(error) => return Err(error),
            Ok(incoming) => match incoming {
                Incoming::Controller(Message::EngineCall {
                    request_id,
                    method,
                    params,
                    ..
                }) => {
                    if let Err(message) = gate.before_call(&method, &params) {
                        return Err(io::Error::other(format!("resource_limit: {message}")));
                    }
                    wait_point = format!("content:{method}");
                    phase!("web-runtime: phase run-wait point={wait_point}");
                    pending.insert(request_id, (method.clone(), params.clone()));
                    content.send_timeout(
                        &Message::engine_call(request_id, method.clone(), params),
                        remaining,
                    )?;
                    gate.note_inflight_engine(request_id, &method);
                }
                Incoming::Content(Message::EngineResult {
                    request_id,
                    ok,
                    result,
                    error,
                    ..
                }) => {
                    let Some((method, params)) = pending.remove(&request_id) else {
                        gate.note_discarded_engine_result(request_id, ok, error);
                        eprintln!(
                            "web-runtime: discarded unmatched EngineResult id={request_id} wait={wait_point}"
                        );
                        continue;
                    };
                    if ok && tally_after(&method) {
                        if let Err(message) = poll_session_records(
                            content,
                            &params,
                            remaining,
                            &mut sidecar,
                            &mut gate,
                            &mut parked,
                        ) {
                            let _ = controller.send(&Message::engine_result(
                                request_id,
                                false,
                                serde_json::json!({}),
                                Some(format!("resource_limit: {message}")),
                            ));
                            return Err(io::Error::other(format!("resource_limit: {message}")));
                        }
                    }
                    controller.send_timeout(
                        &Message::engine_result(request_id, ok, result, error),
                        remaining,
                    )?;
                    if pending.is_empty() {
                        wait_point = "controller:script-complete".to_owned();
                    }
                }
                Incoming::Controller(Message::ScriptComplete {
                    ok, result, error, ..
                }) => {
                    if !ok {
                        return Err(io::Error::other(format!(
                            "controller script failed: {}",
                            error.unwrap_or_else(|| result.to_string())
                        )));
                    }
                    return Ok(result);
                }
                Incoming::Controller(message) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected controller message during script: {message:?}"),
                    ));
                }
                Incoming::Content(message) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected content message during script: {message:?}"),
                    ));
                }
            },
        }
    }
}

fn tally_after(method: &str) -> bool {
    matches!(
        method,
        "page.evaluate"
            | "page.frameEvaluate"
            | "page.goto"
            | "page.reload"
            | "page.goBack"
            | "page.goForward"
            | "page.frameGoto"
            | "page.setContent"
            | "page.saveDownload"
            | "page.click"
            | "page.fill"
            | "page.type"
            | "page.press"
            | "locator.click"
            | "locator.fill"
            | "locator.type"
            | "locator.press"
    )
}

fn poll_session_records(
    content: &mut WorkerProcess,
    params: &serde_json::Value,
    timeout: Duration,
    sidecar: &mut u64,
    gate: &mut impl EngineGate,
    parked: &mut VecDeque<Message>,
) -> Result<(), String> {
    let Some(page) = params.get("page").and_then(|value| value.as_str()) else {
        return Ok(());
    };
    let console = sidecar_engine_call(
        content,
        sidecar,
        "page.consoleMessages",
        serde_json::json!({ "page": page }),
        timeout,
        parked,
    )?;
    let downloads = sidecar_engine_call(
        content,
        sidecar,
        "page.downloads",
        serde_json::json!({ "page": page }),
        timeout,
        parked,
    )?;
    let responses = sidecar_engine_call(
        content,
        sidecar,
        "page.responses",
        serde_json::json!({ "page": page }),
        timeout,
        parked,
    )?;
    let records = serde_json::json!({
        "messages": console.get("messages").cloned().unwrap_or_else(|| serde_json::json!([])),
        "downloads": downloads.get("downloads").cloned().unwrap_or_else(|| serde_json::json!([])),
        "responses": responses.get("responses").cloned().unwrap_or_else(|| serde_json::json!([])),
    });
    gate.after_records(&records)
}

fn sidecar_engine_call(
    content: &mut WorkerProcess,
    sidecar: &mut u64,
    method: &str,
    params: serde_json::Value,
    timeout: Duration,
    parked: &mut VecDeque<Message>,
) -> Result<serde_json::Value, String> {
    *sidecar = sidecar.saturating_add(1);
    let request_id = *sidecar;
    content
        .send(&Message::engine_call(request_id, method.to_owned(), params))
        .map_err(|error| error.to_string())?;
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(format!("timed out after {timeout:?} waiting for {method} sidecar"));
        }
        match content.recv(remaining) {
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
                    Err(error.unwrap_or_else(|| format!("{method} sidecar failed")))
                };
            }
            Ok(message @ Message::EngineResult { .. }) => {
                parked.push_back(message);
            }
            Ok(other) => return Err(format!("unexpected sidecar message {other:?}")),
            Err(error) => return Err(error.to_string()),
        }
    }
}

enum Incoming {
    Controller(Message),
    Content(Message),
}

fn recv_any(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    timeout: Duration,
) -> io::Result<Incoming> {
    let deadline = Instant::now() + timeout;
    // This loop sits on BOTH directions of every engine call. A fixed
    // REAP_POLL_INTERVAL nap here charged each call ~10ms in and ~10ms out;
    // at ~17 calls per script that fixed tax was most of the ~500ms
    // per-navigation overhead (finding 020). Poll finely at first, back off
    // to 1ms so an idle 50ms slice stays cheap.
    let mut idle_polls: u32 = 0;
    loop {
        match controller.messages.try_recv() {
            Ok(message) => return Ok(Incoming::Controller(message?)),
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "controller worker protocol reader stopped",
                ));
            }
        }
        match content.messages.try_recv() {
            Ok(message) => return Ok(Incoming::Content(message?)),
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "content worker protocol reader stopped",
                ));
            }
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for worker protocol traffic",
            ));
        }
        idle_polls = idle_polls.saturating_add(1);
        let nap = if idle_polls < 20 {
            Duration::from_micros(100)
        } else {
            Duration::from_millis(1)
        };
        thread::sleep(nap);
    }
}

#[cfg(unix)]
fn poll_writable(fd: i32, timeout: Duration) -> io::Result<()> {
    let mut fds = [libc::pollfd {
        fd,
        events: libc::POLLOUT,
        revents: 0,
    }];
    let ms = i32::try_from(timeout.as_millis().min(i32::MAX as u128)).unwrap_or(i32::MAX);
    loop {
        let n = unsafe { libc::poll(fds.as_mut_ptr(), 1, ms) };
        if n > 0 {
            if fds[0].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "worker protocol socket is not writable",
                ));
            }
            return Ok(());
        }
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("timed out after {timeout:?} writing worker protocol"),
            ));
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

fn write_all_timeout(
    writer: &mut BufWriter<File>,
    bytes: &[u8],
    timeout: Duration,
) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        poll_writable(writer.get_mut().as_raw_fd(), timeout)?;
    }
    writer.write_all(bytes)?;
    writer.flush()
}

pub(crate) struct WorkerProcess {
    worker: WorkerKind,
    capability: String,
    child: Child,
    input: Option<BufWriter<File>>,
    messages: Receiver<io::Result<Message>>,
    reader_thread: Option<JoinHandle<()>>,
    #[allow(dead_code)]
    stdout_log: Arc<Mutex<Vec<u8>>>,
    stdout_drain: Option<JoinHandle<()>>,
    reaped: bool,
}

fn inherited_worker_env() -> Vec<(OsString, OsString)> {
    const ALLOW: &[&str] = &[
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "FONTCONFIG_PATH",
        "FONTCONFIG_FILE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "GREPPY_WEB_TEST_IGNORE_CERTS",
        // Opt-in navigation phase tracing (finding 020); read by the content
        // worker, harmless to leak, and useless if scrubbed here.
        "GREPPY_WEB_TRACE_NAV",
    ];
    std::env::vars_os()
        .filter(|(key, _)| key.to_str().is_some_and(|name| ALLOW.contains(&name)))
        .collect()
}

impl WorkerProcess {
    pub(crate) fn spawn(worker: WorkerKind, capability: String) -> io::Result<Self> {
        #[cfg(not(unix))]
        {
            let _ = (worker, capability);
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "worker re-exec requires a Unix inherited capability FD",
            ));
        }
        #[cfg(unix)]
        {
            spawn_unix(worker, capability)
        }
    }
}

#[cfg(unix)]
fn spawn_unix(worker: WorkerKind, capability: String) -> io::Result<WorkerProcess> {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
        use std::os::unix::process::CommandExt;

        let path = std::env::current_exe().map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to resolve current executable for {worker:?} re-exec: {error}"),
            )
        })?;
        let role = match worker {
            WorkerKind::Controller => "controller",
            WorkerKind::Content => "content",
        };
        let mut cap = [0; 2];
        if unsafe { libc::pipe(cap.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let cap_read = unsafe { OwnedFd::from_raw_fd(cap[0]) };
        let cap_write = unsafe { OwnedFd::from_raw_fd(cap[1]) };
        unsafe {
            libc::fcntl(cap_read.as_raw_fd(), libc::F_SETFD, 0);
            libc::fcntl(cap_write.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
        }
        let cap_read_fd = cap_read.into_raw_fd();
        let mut cap_write = std::fs::File::from(cap_write);

        let mut proto = [0; 2];
        if unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, proto.as_mut_ptr()) } != 0
        {
            unsafe {
                libc::close(cap_read_fd);
            }
            return Err(io::Error::last_os_error());
        }
        let proto_child = unsafe { OwnedFd::from_raw_fd(proto[0]) };
        let proto_parent = unsafe { OwnedFd::from_raw_fd(proto[1]) };
        unsafe {
            libc::fcntl(proto_child.as_raw_fd(), libc::F_SETFD, 0);
            libc::fcntl(proto_parent.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
        }
        let proto_child_fd = proto_child.into_raw_fd();
        let proto_parent_fd = proto_parent.into_raw_fd();

        let mut command = Command::new(&path);
        command
            .arg("--internal-role")
            .arg(role)
            .env_clear()
            .envs(inherited_worker_env())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        command.process_group(0);
        let sandbox_exe = path.clone();
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(cap_read_fd, crate::worker::CAPABILITY_FD) < 0 {
                    return Err(io::Error::last_os_error());
                }
                if cap_read_fd != crate::worker::CAPABILITY_FD {
                    libc::close(cap_read_fd);
                }
                if libc::dup2(proto_child_fd, crate::worker::PROTOCOL_FD) < 0 {
                    return Err(io::Error::last_os_error());
                }
                if proto_child_fd != crate::worker::PROTOCOL_FD {
                    libc::close(proto_child_fd);
                }
                let _ = sandbox_exe;
                Ok(())
            });
        }
        let image = crate::worker::apply_same_image_reexec(&mut command)?;
        // Cache parent identity before exec so the first child cannot race the digest.
        let _ = parent_image()?;
        let mut child = command.spawn().map_err(|error| {
            unsafe {
                libc::close(cap_read_fd);
                libc::close(proto_child_fd);
                libc::close(proto_parent_fd);
            }
            io::Error::new(
                error.kind(),
                format!(
                    "failed to spawn {worker:?} worker at {}: {error}",
                    path.display()
                ),
            )
        })?;
        unsafe {
            libc::close(cap_read_fd);
            libc::close(proto_child_fd);
        }
        use std::io::Write as _;
        cap_write.write_all(capability.as_bytes())?;
        cap_write.write_all(b"\n")?;
        drop(cap_write);
        image.prove_child_or_kill(&mut child)?;
        if let Err(error) = prove_same_executable(child.id()) {
            unsafe {
                libc::close(proto_parent_fd);
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }

        let proto_read_fd = unsafe { libc::dup(proto_parent_fd) };
        if proto_read_fd < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(proto_parent_fd);
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        unsafe {
            libc::fcntl(proto_read_fd, libc::F_SETFD, libc::FD_CLOEXEC);
        }
        let protocol_write = File::from(unsafe { OwnedFd::from_raw_fd(proto_parent_fd) });
        let protocol_read = File::from(unsafe { OwnedFd::from_raw_fd(proto_read_fd) });
        let stdout = child.stdout.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "worker stdout was not piped")
        })?;
        let stdout_log = Arc::new(Mutex::new(Vec::new()));
        let drain_buf = Arc::clone(&stdout_log);
        let stdout_drain = match thread::Builder::new()
            .name(format!("web-runtime-{worker:?}-stdout-drain"))
            .spawn(move || drain_worker_stdout(stdout, drain_buf))
        {
            Ok(stdout_drain) => stdout_drain,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::new(
                    error.kind(),
                    format!("failed to spawn {worker:?} stdout drain: {error}"),
                ));
            }
        };
        let (message_sender, messages) = mpsc::channel();
        let reader_thread = match thread::Builder::new()
            .name(format!("web-runtime-{worker:?}-protocol-reader"))
            .spawn(move || {
                let mut output = BufReader::new(protocol_read);
                loop {
                    match read_message(&mut output) {
                        Ok(message) => {
                            if message_sender.send(Ok(message)).is_err() {
                                return;
                            }
                        }
                        Err(error) => {
                            let _ = message_sender.send(Err(error));
                            return;
                        }
                    }
                }
            }) {
            Ok(reader_thread) => reader_thread,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::new(
                    error.kind(),
                    format!("failed to spawn {worker:?} protocol reader: {error}"),
                ));
            }
        };

        register_owned_worker(child.id());
        Ok(WorkerProcess {
            worker,
            capability,
            child,
            input: Some(BufWriter::new(protocol_write)),
            messages,
            reader_thread: Some(reader_thread),
            stdout_log,
            stdout_drain: Some(stdout_drain),
            reaped: false,
        })
}

fn drain_worker_stdout(mut stdout: std::process::ChildStdout, log: Arc<Mutex<Vec<u8>>>) {
    let mut buf = [0_u8; 4096];
    loop {
        match stdout.read(&mut buf) {
            Ok(0) => return,
            Ok(n) => {
                let mut log = log.lock().unwrap_or_else(|error| error.into_inner());
                if log.len() < STDOUT_LOG_CAP {
                    let take = n.min(STDOUT_LOG_CAP - log.len());
                    log.extend_from_slice(&buf[..take]);
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return,
        }
    }
}


#[cfg(target_os = "macos")]
fn sbpl_subpath(path: &Path) -> String {
    let text = path.display().to_string().replace('\\', "\\\\").replace('"', "\\\"");
    format!("(subpath \"{text}\")")
}

#[cfg(target_os = "macos")]
pub(crate) fn apply_worker_sandbox(exe: &Path, tmp: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    extern "C" {
        fn sandbox_init(profile: *const c_char, flags: u64, errorbuf: *mut *mut c_char) -> c_int;
        fn sandbox_free_error(errorbuf: *mut c_char);
    }
    let exe_dir = exe.parent().unwrap_or(exe);
    let profile = macos_sandbox_profile(exe, exe_dir, tmp);
    let profile = CString::new(profile)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let mut errorbuf: *mut c_char = std::ptr::null_mut();
    let rc = unsafe { sandbox_init(profile.as_ptr(), 0, &mut errorbuf) };
    if rc != 0 {
        let message = if errorbuf.is_null() {
            "sandbox_init failed".to_owned()
        } else {
            let text = unsafe { std::ffi::CStr::from_ptr(errorbuf) }
                .to_string_lossy()
                .into_owned();
            unsafe { sandbox_free_error(errorbuf) };
            text
        };
        return Err(io::Error::other(format!("worker sandbox: {message}")));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn macos_sandbox_profile(exe: &Path, exe_dir: &Path, tmp: &Path) -> String {
    format!(
        r#"(version 1)
(deny default)
(allow file-read-metadata)
(allow file-map-executable)
(allow file-read*
  (subpath "/usr")
  (subpath "/System")
  (subpath "/Library")
  (subpath "/opt")
  (subpath "/dev")
  (literal "/dev/urandom")
  (literal "/dev/random")
  (literal "/dev/null")
  (literal "/dev/zero")
  {exe}
  {exe_dir}
  {tmp}
  (subpath "/private/var/folders")
  (subpath "/private/tmp")
  (subpath "/tmp")
  (literal "/etc/resolv.conf")
  (literal "/etc/hosts")
  (literal "/private/etc/resolv.conf")
  (literal "/private/etc/hosts")
)
(allow file-write*
  {tmp}
  (subpath "/private/var/folders")
  (subpath "/private/tmp")
  (subpath "/tmp")
)
(allow sysctl-read)
(allow mach-lookup)
(allow iokit-open)
(allow ipc-posix-shm)
(allow ipc-posix-sem)
(allow process-info*)
(allow signal (target self))
(allow file-ioctl)
(allow process-exec {exe} {exe_dir})
(allow system-socket)
(allow network-bind)
(allow network-outbound)
(allow network-inbound (local ip))
"#,
        exe = sbpl_subpath(exe),
        exe_dir = sbpl_subpath(exe_dir),
        tmp = sbpl_subpath(tmp),
    )
}

#[cfg(target_os = "linux")]
pub(crate) fn apply_worker_sandbox(exe: &Path, tmp: &Path) -> io::Result<()> {
    crate::linux_sandbox::apply(exe, tmp)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn apply_worker_sandbox(_exe: &Path, _tmp: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "worker OS sandbox is not implemented on this platform; refusing to start unsandboxed",
    ))
}

/// On-disk identity of the supervisor image sampled at cache time.
///
/// Path + dev/inode + size + high-resolution mtime/ctime + SHA-256 describe the
/// file as observed then. They do **not** prove the kernel mapped those exact
/// bytes into a later child: an in-place rewrite of the same inode after this
/// snapshot (and before exec completes) is a residual TOCTOU. Closing it needs
/// a pinned executable FD (`fexecve` on Linux) or a platform code-signing proof
/// over the mapped pages. macOS spawn is path-based, so this crate documents
/// rather than claims that stronger guarantee.
#[cfg(unix)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct ImageId {
    path: PathBuf,
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
    sha256: String,
}


#[cfg(unix)]
fn image_digest_cache_path(_path: &Path, meta: &std::fs::Metadata) -> PathBuf {
    use std::os::unix::fs::MetadataExt;
    std::env::temp_dir().join(format!(
        "greppy-web-image-{}-{}-{}-{}-{}.sha256",
        meta.dev(),
        meta.ino(),
        meta.size(),
        meta.mtime(),
        meta.mtime_nsec()
    ))
}

#[cfg(unix)]
fn digest_from_sha256sums(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let hex = parts.next()?;
        let name = parts.next()?.trim_start_matches('*');
        if hex.len() == 64
            && hex.bytes().all(|b| b.is_ascii_hexdigit())
            && (name == "web-runtime"
                || name == "bin/web-runtime"
                || name.ends_with("/web-runtime"))
        {
            return Some(hex.to_ascii_lowercase());
        }
    }
    None
}

#[cfg(unix)]
fn prefill_digest_cache_from_dist(exe: &Path, meta: &std::fs::Metadata) {
    let Some(parent) = exe.parent() else {
        return;
    };
    for sums in [parent.join("SHA256SUMS"), parent.join("..").join("SHA256SUMS")] {
        let Ok(text) = fs::read_to_string(&sums) else {
            continue;
        };
        if let Some(digest) = digest_from_sha256sums(&text) {
            store_image_digest_cache(exe, meta, &digest);
            return;
        }
    }
}

#[cfg(unix)]
fn reap_stale_image_digest_caches(keep: &std::fs::Metadata) {
    use std::os::unix::fs::MetadataExt;
    let needle = format!("-{}-", keep.ino());
    let Ok(entries) = fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("greppy-web-image-") || !name.ends_with(".sha256") {
            continue;
        }
        if name.contains(&needle) {
            continue;
        }
        let _ = fs::remove_file(entry.path());
    }
}

#[cfg(unix)]
fn load_image_digest_cache(path: &Path, meta: &std::fs::Metadata) -> Option<String> {
    let cached = fs::read_to_string(image_digest_cache_path(path, meta)).ok()?;
    let digest = cached.trim().to_owned();
    if digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        Some(digest)
    } else {
        None
    }
}

#[cfg(unix)]
fn store_image_digest_cache(path: &Path, meta: &std::fs::Metadata, digest: &str) {
    let _ = fs::write(image_digest_cache_path(path, meta), digest);
}

#[cfg(unix)]
fn observe_image(path: &Path, digest: bool) -> io::Result<ImageId> {
    use std::os::unix::fs::MetadataExt;
    let path = fs::canonicalize(path)?;
    let meta = fs::metadata(&path)?;
    let sha256 = if digest {
        reap_stale_image_digest_caches(&meta);
        prefill_digest_cache_from_dist(&path, &meta);
        if let Some(cached) = load_image_digest_cache(&path, &meta) {
            phase!("web-runtime: phase parent-image cache-hit");
            cached
        } else {
            let digest = crate::artifacts::hex_sha256_file(&path)?;
            store_image_digest_cache(&path, &meta, &digest);
            digest
        }
    } else {
        String::new()
    };
    Ok(ImageId {
        path,
        dev: meta.dev(),
        ino: meta.ino(),
        size: meta.size(),
        mtime_sec: meta.mtime(),
        mtime_nsec: meta.mtime_nsec(),
        ctime_sec: meta.ctime(),
        ctime_nsec: meta.ctime_nsec(),
        sha256,
    })
}

#[cfg(unix)]
fn identity_mismatch(expected: &ImageId, observed: &ImageId) -> Option<String> {
    if expected.path != observed.path {
        return Some(format!(
            "path {} != {}",
            observed.path.display(),
            expected.path.display()
        ));
    }
    if expected.dev != observed.dev || expected.ino != observed.ino {
        return Some(format!(
            "device/inode {}/{} != parent {}/{}",
            observed.dev, observed.ino, expected.dev, expected.ino
        ));
    }
    if expected.size != observed.size {
        return Some(format!(
            "size {} != parent {}",
            observed.size, expected.size
        ));
    }
    if expected.mtime_sec != observed.mtime_sec || expected.mtime_nsec != observed.mtime_nsec {
        return Some(format!(
            "mtime {}:{:09} != parent {}:{:09}",
            observed.mtime_sec, observed.mtime_nsec, expected.mtime_sec, expected.mtime_nsec
        ));
    }
    if expected.ctime_sec != observed.ctime_sec || expected.ctime_nsec != observed.ctime_nsec {
        return Some(format!(
            "ctime {}:{:09} != parent {}:{:09}",
            observed.ctime_sec, observed.ctime_nsec, expected.ctime_sec, expected.ctime_nsec
        ));
    }
    if !observed.sha256.is_empty() && expected.sha256 != observed.sha256 {
        return Some("SHA-256 digest mismatch".into());
    }
    None
}

#[cfg(unix)]
pub(crate) fn warmup_parent_image() -> io::Result<Duration> {
    let started = Instant::now();
    parent_image()?;
    Ok(started.elapsed())
}

#[cfg(unix)]
fn parent_image() -> io::Result<&'static ImageId> {
    use std::sync::{Mutex, OnceLock};
    static PARENT: OnceLock<ImageId> = OnceLock::new();
    static LOAD: Mutex<()> = Mutex::new(());
    if let Some(parent) = PARENT.get() {
        return Ok(parent);
    }
    let _guard = LOAD.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(parent) = PARENT.get() {
        return Ok(parent);
    }
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let loaded = observe_image(&exe, true)?;
    if loaded.sha256.len() != 64 || !loaded.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("parent image digest is not a SHA-256 hex: {}", loaded.sha256),
        ));
    }
    let _ = PARENT.set(loaded);
    Ok(PARENT.get().expect("parent image id"))
}

#[cfg(unix)]
fn prove_same_executable(child_pid: u32) -> io::Result<()> {
    let parent = parent_image()?;
    let child_path = child_executable_path(child_pid)?;
    let child = observe_image(&child_path, false)?;
    if let Some(reason) = identity_mismatch(parent, &child) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("worker pid {child_pid} {reason}"),
        ));
    }
    Ok(())
}

fn kill_process_tree(pid: u32) {
    #[cfg(unix)]
    {
        // Workers are process-group leaders (`process_group(0)`). The supervisor
        // owns that PGID as the Child pid; SIGKILL the group then the leader.
        let raw = pid as i32;
        unsafe {
            libc::kill(-raw, libc::SIGKILL);
            libc::kill(raw, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
}

#[cfg(unix)]
fn child_executable_path(pid: u32) -> io::Result<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let mut buf = vec![0u8; 4096];
        let n = unsafe {
            libc::proc_pidpath(
                pid as i32,
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len() as u32,
            )
        };
        if n <= 0 {
            return Err(io::Error::other(format!(
                "proc_pidpath({pid}) failed: {}",
                io::Error::last_os_error()
            )));
        }
        buf.truncate(n as usize);
        Ok(PathBuf::from(String::from_utf8(buf).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "worker executable path is not UTF-8")
        })?))
    }
    #[cfg(target_os = "linux")]
    {
        fs::read_link(format!("/proc/{pid}/exe"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = pid;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "same-image re-exec identity proof is unimplemented on this Unix",
        ))
    }
}

impl WorkerProcess {
    pub(crate) fn handshake(&mut self) -> io::Result<()> {
        self.send(&Message::hello(self.worker, self.capability.clone()))?;
        self.expect(Message::ready(self.worker))
    }

    pub(crate) fn shutdown(&mut self) -> io::Result<()> {
        self.send(&Message::shutdown())?;
        self.expect(Message::shutdown_ack(self.worker))?;
        self.input.take();

        let status = self.wait_for_exit()?;
        self.reaped = true;
        self.join_reader();
        if !status.success() {
            return Err(io::Error::other(format!(
                "{:?} worker exited with {status}",
                self.worker
            )));
        }
        Ok(())
    }

    pub(crate) fn recv(&mut self, timeout: Duration) -> io::Result<Message> {
        match self.messages.recv_timeout(timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "timed out after {timeout:?} waiting for {:?} worker message",
                    self.worker
                ),
            )),
            Err(RecvTimeoutError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("{:?} worker protocol reader stopped", self.worker),
            )),
        }
    }

    pub(crate) fn pid(&self) -> u32 {
        self.child.id()
    }

    pub(crate) fn is_running(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) => {
                self.reaped = true;
                false
            }
            Err(_) => false,
        }
    }

    pub(crate) fn kill_tree(&mut self) {
        self.kill_tree_wait(WORKER_REAP_WAIT, READER_JOIN_WAIT);
    }

    pub(crate) fn kill_tree_now(&mut self) {
        self.kill_tree_wait(TIMEOUT_REAP_WAIT, Duration::from_millis(50));
    }

    fn kill_tree_wait(&mut self, reap_wait: Duration, reader_wait: Duration) {
        let pid = self.child.id();
        eprintln!(
            "web-runtime: phase {:?}-reap start pid={pid}",
            self.worker
        );
        unregister_owned_worker(pid);
        kill_process_tree(pid);
        let deadline = Instant::now() + reap_wait;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() >= deadline => {
                    kill_process_tree(pid);
                    let _ = self.child.try_wait();
                    break;
                }
                Ok(None) => thread::sleep(REAP_POLL_INTERVAL),
                Err(_) => break,
            }
        }
        self.reaped = true;
        self.input.take();
        self.join_reader_bounded(reader_wait);
        phase!("web-runtime: phase {:?}-reap done pid={pid}", self.worker);
    }

    pub(crate) fn shutdown_or_kill(&mut self) {
        let pid = self.child.id();
        eprintln!(
            "web-runtime: phase {:?}-eof start pid={pid}",
            self.worker
        );
        if self.send(&Message::shutdown()).is_ok() {
            let deadline = Instant::now() + WORKER_EOF_WAIT;
            while Instant::now() < deadline {
                match self.child.try_wait() {
                    Ok(Some(_)) => {
                        self.reaped = true;
                        self.input.take();
                        self.join_reader_bounded(READER_JOIN_WAIT);
                        eprintln!(
                            "web-runtime: phase {:?}-eof done pid={pid}",
                            self.worker
                        );
                        return;
                    }
                    Ok(None) => thread::sleep(REAP_POLL_INTERVAL),
                    Err(_) => break,
                }
            }
        }
        eprintln!(
            "web-runtime: phase {:?}-eof timeout pid={pid}; escalating to reap",
            self.worker
        );
        self.kill_tree();
    }

    pub(crate) fn send(&mut self, message: &Message) -> io::Result<()> {
        self.send_timeout(message, PROCESS_TIMEOUT)
    }

    pub(crate) fn send_timeout(
        &mut self,
        message: &Message,
        timeout: Duration,
    ) -> io::Result<()> {
        let input = self.input.as_mut().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "worker protocol socket is already closed",
            )
        })?;
        let payload = serde_json::to_vec(message).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cannot encode protocol JSON: {error}"),
            )
        })?;
        if payload.len() > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "encoded frame length {} exceeds {MAX_FRAME_BYTES}-byte limit",
                    payload.len()
                ),
            ));
        }
        let payload_len = u32::try_from(payload.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "encoded frame length does not fit in u32",
            )
        })?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&payload_len.to_be_bytes());
        frame.extend_from_slice(&payload);
        write_all_timeout(input, &frame, timeout)
    }

    fn expect(&mut self, expected: Message) -> io::Result<()> {
        let actual = match self.messages.recv_timeout(PROCESS_TIMEOUT) {
            Ok(result) => result?,
            Err(RecvTimeoutError::Timeout) => {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "timed out after {PROCESS_TIMEOUT:?} waiting for {:?} worker message",
                        self.worker
                    ),
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    format!("{:?} worker protocol reader stopped", self.worker),
                ));
            }
        };
        if actual != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{:?} worker sent {actual:?}, expected {expected:?}",
                    self.worker
                ),
            ));
        }
        Ok(())
    }

    fn wait_for_exit(&mut self) -> io::Result<std::process::ExitStatus> {
        let deadline = Instant::now() + PROCESS_TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "timed out after {PROCESS_TIMEOUT:?} reaping {:?} worker",
                        self.worker
                    ),
                ));
            }
            thread::sleep(REAP_POLL_INTERVAL);
        }
    }

    fn join_reader(&mut self) {
        self.join_reader_bounded(READER_JOIN_WAIT);
    }

    fn join_reader_bounded(&mut self, timeout: Duration) {
        if let Some(drain) = self.stdout_drain.take() {
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || {
                let _ = drain.join();
                let _ = tx.send(());
            });
            let _ = rx.recv_timeout(timeout);
        }
        let Some(reader) = self.reader_thread.take() else {
            return;
        };
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = reader.join();
            let _ = tx.send(());
        });
        if rx.recv_timeout(timeout).is_err() {
            eprintln!(
                "web-runtime: phase {:?}-reader-join timeout",
                self.worker
            );
        }
    }

    pub(crate) fn discard_stale_engine_results(&mut self) -> u64 {
        let mut discarded = 0;
        loop {
            match self.messages.try_recv() {
                Ok(Ok(Message::EngineResult {
                    request_id,
                    ok,
                    error,
                    ..
                })) => {
                    discarded += 1;
                    eprintln!(
                        "web-runtime: discarded stale EngineResult id={request_id} ok={ok} err={error:?}"
                    );
                }
                Ok(Ok(other)) => {
                    discarded += 1;
                    phase!("web-runtime: discarded leftover content message {other:?}");
                }
                Ok(Err(error)) => {
                    phase!("web-runtime: discarded leftover content read error {error}");
                    break;
                }
                Err(_) => break,
            }
        }
        discarded
    }

    #[allow(dead_code)]
    pub(crate) fn stdout_snapshot(&self) -> Vec<u8> {
        self.stdout_log
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        self.kill_tree();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_from_sha256sums_reads_bin_web_runtime() {
        let text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  bin/web-runtime\n";
        assert_eq!(
            digest_from_sha256sums(text).as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        assert_eq!(digest_from_sha256sums("not-a-sum\n"), None);
    }

    #[test]
    fn loopback_fixture_url_grants_project_profile() {
        assert!(fixture_url_grants_project_loopback("http://127.0.0.1:9/x"));
        assert!(!fixture_url_grants_project_loopback("https://example.com/"));
        assert!(!fixture_url_grants_project_loopback(""));
        assert!(!fixture_url_grants_project_loopback(
            "http://169.254.169.254/latest"
        ));
    }

    #[cfg(all(not(target_os = "macos"), not(target_os = "linux")))]
    #[test]
    fn non_macos_worker_sandbox_refuses_unsandboxed_start() {
        let err = apply_worker_sandbox(Path::new("/"), Path::new("/tmp")).unwrap_err();
        assert!(
            err.to_string().contains("refusing to start unsandboxed"),
            "{err}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_worker_sandbox_allows_public_network_outbound() {
        let profile = macos_sandbox_profile(Path::new("/tmp/exe"), Path::new("/tmp"), Path::new("/tmp"));
        assert!(
            profile.contains("(allow network-outbound)\n"),
            "policy proxy must be able to dial non-loopback hosts; seatbelt is not the policy layer: {profile}"
        );
        assert!(
            profile.contains("(allow system-socket)"),
            "socket() itself is a separate seatbelt gate from network-outbound: {profile}"
        );
        assert!(
            !profile.contains("localhost:*"),
            "localhost-only outbound was the 003 connect failure: {profile}"
        );
        assert!(
            profile.contains("/etc/resolv.conf"),
            "DNS needs resolv.conf: {profile}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_worker_sandbox_refuses_filesystem_root() {
        let err = apply_worker_sandbox(Path::new("/"), Path::new("/")).unwrap_err();
        assert_ne!(err.kind(), io::ErrorKind::NotFound, "{err}");
        assert!(
            !err.to_string().is_empty(),
            "linux sandbox must fail closed, got empty error"
        );
    }

    #[cfg(unix)]
    #[test]
    fn current_process_matches_its_own_image_identity() {
        prove_same_executable(std::process::id()).unwrap();
        // Second call must hit the process-lifetime cache, not re-slurp the image.
        prove_same_executable(std::process::id()).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn in_place_rewrite_is_detected_without_claiming_inode_equals_bytes() {
        use std::io::Write as _;
        use std::os::unix::fs::MetadataExt;
        let root = std::env::temp_dir().join(format!("greppy-img-tamper-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("image.bin");
        std::fs::write(&path, b"original-bytes").unwrap();
        let before = observe_image(&path, true).unwrap();
        let ino = std::fs::metadata(&path).unwrap().ino();

        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all(b"original-bytes-TAMPER")
            .unwrap();
        let grown = observe_image(&path, false).unwrap();
        assert_eq!(grown.ino, ino, "rewrite must stay on the same inode");
        let reason = identity_mismatch(&before, &grown).expect("size-changing in-place rewrite");
        assert!(reason.contains("size"), "{reason}");

        std::fs::write(&path, b"original-bytes").unwrap();
        let same_len_before = observe_image(&path, true).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all(b"XXXXXXXXXXXXXX")
            .unwrap();
        let same_len_after = observe_image(&path, false).unwrap();
        assert_eq!(same_len_after.ino, same_len_before.ino);
        assert_eq!(same_len_after.size, same_len_before.size);
        let reason =
            identity_mismatch(&same_len_before, &same_len_after).expect("same-length in-place rewrite");
        assert!(
            reason.contains("mtime") || reason.contains("ctime"),
            "{reason}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn parses_without_separate_worker_images() {
        let config = Config::parse([
            OsString::from("--socket"),
            OsString::from("/tmp/x.sock"),
            OsString::from("--run-id"),
            OsString::from("run_one"),
        ])
        .unwrap();
        assert_eq!(config.scripts, Vec::<PathBuf>::new());
        assert_eq!(config.socket.as_deref(), Some(Path::new("/tmp/x.sock")));
        assert_eq!(config.run_id.as_deref(), Some("run_one"));
        assert_eq!(config.search_endpoint, None);
    }

    #[test]
    fn parses_search_endpoint() {
        let config = Config::parse([
            OsString::from("--socket"),
            OsString::from("/tmp/x.sock"),
            OsString::from("--run-id"),
            OsString::from("run_one"),
            OsString::from("--search-endpoint"),
            OsString::from("http://127.0.0.1:9/search"),
        ])
        .unwrap();
        assert_eq!(
            config.search_endpoint.as_deref(),
            Some("http://127.0.0.1:9/search")
        );
    }

    #[test]
    fn refuses_separate_worker_image_flags() {
        let error = Config::parse([
            OsString::from("--controller-worker"),
            OsString::from("controller"),
        ])
        .unwrap_err();
        assert!(error.contains("--internal-role"), "{error}");
    }

    #[test]
    fn dist_layout_requires_the_single_runtime_executable() {
        let root =
            std::env::temp_dir().join(format!("greppy-web-dist-parse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(
            root.join(".greppy-web-runtime-dist"),
            "greppy.web-runtime.package.v1\n",
        )
        .unwrap();
        std::fs::write(root.join("bin").join("web-runtime"), b"runtime").unwrap();
        let config = Config::parse([
            OsString::from("--dist"),
            OsString::from(root.to_string_lossy().as_ref()),
            OsString::from("--socket"),
            OsString::from("/tmp/x.sock"),
            OsString::from("--run-id"),
            OsString::from("run_dist"),
        ])
        .unwrap();
        assert_eq!(config.socket.as_deref(), Some(Path::new("/tmp/x.sock")));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dist_rejects_legacy_three_image_layout() {
        let root =
            std::env::temp_dir().join(format!("greppy-web-dist-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(
            root.join(".greppy-web-runtime-dist"),
            "greppy.web-runtime.package.v1\n",
        )
        .unwrap();
        std::fs::write(
            root.join("bin").join("web-controller-worker"),
            b"controller",
        )
        .unwrap();
        std::fs::write(root.join("bin").join("web-content-worker"), b"content").unwrap();
        let error = Config::parse([
            OsString::from("--dist"),
            OsString::from(root.to_string_lossy().as_ref()),
        ])
        .unwrap_err();
        assert!(error.contains("web-runtime"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
