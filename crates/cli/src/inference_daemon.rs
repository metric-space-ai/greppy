//! Shared lifecycle and local transport for the two embedded inference daemons.

use std::io::{Read, Write};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

pub(super) const PROTOCOL_VERSION: u32 = 2;
const ACCEPT_QUEUE_LENGTH: usize = 16;
const INFERENCE_QUEUE_LENGTH: usize = 8;
const READER_WORKERS: usize = 4;
const CONNECTION_READ_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECTION_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const LOOP_INTERVAL: Duration = Duration::from_millis(100);
const CRASH_COOLDOWN: Duration = Duration::from_secs(2);
const CAPACITY_RETRY_INITIAL_DELAY: Duration = Duration::from_millis(10);
const CAPACITY_RETRY_MAX_DELAY: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestOutcome<T> {
    Response(T),
    DaemonBusy,
    NoDaemon,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpawnOutcome {
    Spawned,
    SpawnFailed,
    Contended,
    Cooldown,
}

impl SpawnOutcome {
    pub(super) fn attempted(self) -> bool {
        matches!(self, Self::Spawned | Self::SpawnFailed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecycleState {
    Starting,
    Loading,
    Ready,
    Evicted,
    Faulted,
}

impl LifecycleState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Evicted => "evicted",
            Self::Faulted => "faulted",
        }
    }
}

#[derive(Debug)]
struct RuntimeStatus {
    state: LifecycleState,
    state_started: Instant,
    active_request_id: Option<String>,
    active_request_started: Option<Instant>,
    completed_requests: u64,
    rejected_requests: u64,
    last_error: Option<String>,
}

impl Default for RuntimeStatus {
    fn default() -> Self {
        Self {
            state: LifecycleState::Starting,
            state_started: Instant::now(),
            active_request_id: None,
            active_request_started: None,
            completed_requests: 0,
            rejected_requests: 0,
            last_error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct Endpoint {
    kind: &'static str,
    digest: String,
    address: String,
}

impl Endpoint {
    pub(super) fn for_identity(kind: &'static str, identity: &str) -> Option<Self> {
        let mut hash = Sha256::new();
        hash.update(b"greppy-inference-daemon\0");
        hash.update(PROTOCOL_VERSION.to_le_bytes());
        hash.update(kind.as_bytes());
        hash.update(b"\0");
        hash.update(identity.as_bytes());
        let digest = hex_encode(&hash.finalize())[..32].to_string();
        #[cfg(unix)]
        let address = {
            let dir = unix_runtime_dir();
            ensure_private_dir(&dir).ok()?;
            dir.join(format!("{kind}-{digest}.sock"))
                .to_string_lossy()
                .into_owned()
        };
        #[cfg(windows)]
        let address = format!(r"\\.\pipe\greppy-{kind}-{digest}");
        #[cfg(not(any(unix, windows)))]
        return None;
        Some(Self {
            kind,
            digest,
            address,
        })
    }

    pub(super) fn address(&self) -> &str {
        &self.address
    }

    fn lock_name(&self) -> String {
        format!("daemon-{}-{}.owner", self.kind, self.digest)
    }

    fn spawn_lock_name(&self) -> String {
        format!("daemon-{}-{}.spawn", self.kind, self.digest)
    }

    fn cooldown_path(&self) -> std::path::PathBuf {
        greppy_core::cache::data_root()
            .join("runtime")
            .join("daemon-state")
            .join(format!("{}-{}.cooldown", self.kind, self.digest))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ServerPolicy {
    pub model_ttl: Duration,
    pub exit_ttl: Duration,
    pub request_deadline: Duration,
    pub hard_request_timeout: Option<Duration>,
    pub max_request_bytes: usize,
    pub max_response_bytes: usize,
}

struct RequestJob {
    id: String,
    raw: String,
    deadline: Instant,
    stream: TransportStream,
}

pub(super) fn request(
    endpoint: &Endpoint,
    mut value: serde_json::Value,
    timeout: Duration,
    max_request_bytes: usize,
    max_response_bytes: usize,
) -> RequestOutcome<serde_json::Value> {
    let Some(object) = value.as_object_mut() else {
        return RequestOutcome::Failed;
    };
    object.insert("protocol".into(), PROTOCOL_VERSION.into());
    object
        .entry("request_id")
        .or_insert_with(|| request_id().into());

    let encoded = match serialize_json_frame(&value, max_request_bytes) {
        Ok(encoded) => encoded,
        Err(_) => return RequestOutcome::Failed,
    };
    let deadline = Instant::now() + timeout;
    let mut retry_delay = CAPACITY_RETRY_INITIAL_DELAY;
    let mut capacity_seen = false;
    loop {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return if capacity_seen {
                RequestOutcome::DaemonBusy
            } else {
                RequestOutcome::Failed
            };
        };
        if remaining.is_zero() {
            return if capacity_seen {
                RequestOutcome::DaemonBusy
            } else {
                RequestOutcome::Failed
            };
        }
        match request_once(endpoint, &encoded, remaining, max_response_bytes) {
            RequestAttempt::Response(response) => return RequestOutcome::Response(response),
            RequestAttempt::NoDaemon => return RequestOutcome::NoDaemon,
            RequestAttempt::Failed if capacity_seen => return RequestOutcome::DaemonBusy,
            RequestAttempt::Failed => return RequestOutcome::Failed,
            RequestAttempt::RetryableCapacity => {
                capacity_seen = true;
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    return RequestOutcome::DaemonBusy;
                };
                std::thread::sleep(retry_delay.min(remaining));
                retry_delay = retry_delay
                    .checked_mul(2)
                    .unwrap_or(CAPACITY_RETRY_MAX_DELAY)
                    .min(CAPACITY_RETRY_MAX_DELAY);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RequestAttempt<T> {
    Response(T),
    RetryableCapacity,
    NoDaemon,
    Failed,
}

fn request_once(
    endpoint: &Endpoint,
    encoded: &[u8],
    timeout: Duration,
    max_response_bytes: usize,
) -> RequestAttempt<serde_json::Value> {
    let mut stream = match TransportStream::connect(endpoint, timeout) {
        Ok(stream) => stream,
        Err(error) if no_daemon_error(&error) => return RequestAttempt::NoDaemon,
        Err(_) => return RequestAttempt::Failed,
    };
    if stream
        .set_timeouts(CONNECTION_WRITE_TIMEOUT.min(timeout), timeout)
        .is_err()
    {
        return RequestAttempt::Failed;
    }
    if write_frame(&mut stream, encoded, CONNECTION_WRITE_TIMEOUT.min(timeout)).is_err() {
        return RequestAttempt::Failed;
    }
    let response = match read_frame(&mut stream, max_response_bytes, timeout) {
        Ok(response) => response,
        Err(_) => return RequestAttempt::Failed,
    };
    match serde_json::from_str(&response) {
        Ok(response) if retryable_capacity_response(&response) => RequestAttempt::RetryableCapacity,
        Ok(response) => RequestAttempt::Response(response),
        Err(_) => RequestAttempt::Failed,
    }
}

fn retryable_capacity_response(response: &serde_json::Value) -> bool {
    let classified = response
        .get("error_kind")
        .and_then(serde_json::Value::as_str)
        == Some("capacity")
        && response
            .get("retryable")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
    classified
        || matches!(
            response.get("error").and_then(serde_json::Value::as_str),
            Some("inference queue full" | "daemon busy")
        )
}

pub(super) fn diagnostic(endpoint: &Endpoint) -> serde_json::Value {
    let request = serde_json::json!({"op": "status"});
    match self::request(
        endpoint,
        request,
        Duration::from_millis(500),
        16 * 1024,
        16 * 1024,
    ) {
        RequestOutcome::Response(mut status) => {
            if let Some(object) = status.as_object_mut() {
                object.insert("endpoint".into(), endpoint.address().into());
            }
            status
        }
        RequestOutcome::NoDaemon => serde_json::json!({
            "endpoint": endpoint.address(),
            "protocol": PROTOCOL_VERSION,
            "state": "stopped",
        }),
        RequestOutcome::DaemonBusy => serde_json::json!({
            "endpoint": endpoint.address(),
            "protocol": PROTOCOL_VERSION,
            "state": "busy",
        }),
        RequestOutcome::Failed => serde_json::json!({
            "endpoint": endpoint.address(),
            "protocol": PROTOCOL_VERSION,
            "state": "faulted",
            "last_error": "daemon status request failed",
        }),
    }
}

pub(super) fn spawn_once(endpoint: &Endpoint, spawn: impl FnOnce() -> Option<()>) -> SpawnOutcome {
    if cooldown_active(endpoint) {
        return SpawnOutcome::Cooldown;
    }
    let Some(lock) = greppy_core::cache::acquire_named_lock(
        &endpoint.spawn_lock_name(),
        greppy_core::cache::LockMode::Exclusive,
        true,
    )
    .ok()
    .flatten() else {
        return SpawnOutcome::Contended;
    };
    let outcome = if spawn().is_some() {
        SpawnOutcome::Spawned
    } else {
        SpawnOutcome::SpawnFailed
    };
    drop(lock);
    outcome
}

pub(super) fn record_spawn_failure(endpoint: &Endpoint, spawn_attempted: bool) {
    if !spawn_attempted || cooldown_active(endpoint) {
        return;
    }
    let path = endpoint.cooldown_path();
    let Some(parent) = path.parent() else {
        return;
    };
    if ensure_private_dir(parent).is_err() {
        return;
    }
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return,
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    if let Ok(mut file) = options.open(path) {
        let _ = file.write_all(b"greppy-daemon-crash-cooldown\n");
        let _ = file.sync_all();
    }
}

fn cooldown_active(endpoint: &Endpoint) -> bool {
    endpoint
        .cooldown_path()
        .metadata()
        .ok()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|elapsed| elapsed < CRASH_COOLDOWN)
}

pub(super) fn retry_delays() -> impl Iterator<Item = Duration> {
    [50u64, 100, 200, 400, 800, 800, 800]
        .into_iter()
        .map(Duration::from_millis)
}

pub(super) fn detach_command(command: &mut std::process::Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::System::Threading::{
            CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, DETACHED_PROCESS,
        };
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | DETACHED_PROCESS);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn serve<M, Load, Validate, Handle>(
    endpoint: Endpoint,
    supplied_address: &str,
    policy: ServerPolicy,
    prewarm: bool,
    mut load: Load,
    mut validate: Validate,
    mut handle: Handle,
    log_prefix: &'static str,
) -> !
where
    Load: FnMut() -> Result<M, String>,
    Validate: FnMut(&str) -> Result<(), serde_json::Value>,
    Handle: FnMut(&str, &mut Option<M>) -> serde_json::Value,
{
    let code = run_server(
        endpoint,
        supplied_address,
        policy,
        prewarm,
        &mut load,
        &mut validate,
        &mut handle,
        log_prefix,
    );
    std::process::exit(code)
}

#[allow(clippy::too_many_arguments)]
fn run_server<M, Load, Validate, Handle>(
    endpoint: Endpoint,
    supplied_address: &str,
    policy: ServerPolicy,
    prewarm: bool,
    mut load: Load,
    mut validate: Validate,
    mut handle: Handle,
    log_prefix: &'static str,
) -> i32
where
    Load: FnMut() -> Result<M, String>,
    Validate: FnMut(&str) -> Result<(), serde_json::Value>,
    Handle: FnMut(&str, &mut Option<M>) -> serde_json::Value,
{
    if supplied_address != endpoint.address() {
        return 64;
    }
    let owner_lock = match greppy_core::cache::acquire_named_lock(
        &endpoint.lock_name(),
        greppy_core::cache::LockMode::Exclusive,
        true,
    ) {
        Ok(Some(lock)) => lock,
        Ok(None) => return 0,
        Err(_) => return 1,
    };
    let listener = match TransportListener::bind(&endpoint) {
        Ok(listener) => listener,
        Err(_) => return 1,
    };
    let _ = std::fs::remove_file(endpoint.cooldown_path());
    let endpoint_guard = EndpointGuard::new(endpoint.clone());
    let status = Arc::new(Mutex::new(RuntimeStatus::default()));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let pending_requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let last_activity = Arc::new(Mutex::new(Instant::now()));
    let (accepted_tx, accepted_rx) = mpsc::sync_channel(ACCEPT_QUEUE_LENGTH);
    let (job_tx, job_rx) = mpsc::sync_channel(INFERENCE_QUEUE_LENGTH);

    spawn_accept_loop(
        listener,
        accepted_tx,
        Arc::clone(&status),
        Arc::clone(&stop),
        Arc::clone(&pending_requests),
        Arc::clone(&last_activity),
    );
    spawn_reader_workers(
        accepted_rx,
        job_tx,
        policy,
        Arc::clone(&status),
        Arc::clone(&stop),
        Arc::clone(&pending_requests),
        Arc::clone(&last_activity),
    );
    spawn_hung_worker_watchdog(
        endpoint.clone(),
        Arc::clone(&status),
        Arc::clone(&stop),
        policy.hard_request_timeout,
        log_prefix,
    );

    let mut model = None;
    let mut last_model_completed = Instant::now();
    if prewarm {
        set_state(&status, LifecycleState::Loading, None);
        match load() {
            Ok(loaded) => {
                model = Some(loaded);
                set_state(&status, LifecycleState::Ready, None);
            }
            Err(error) => set_state(&status, LifecycleState::Faulted, Some(error)),
        }
        last_model_completed = Instant::now();
    }

    loop {
        match job_rx.recv_timeout(LOOP_INTERVAL) {
            Ok(mut job) => {
                if Instant::now() >= job.deadline {
                    reject(&status);
                    write_response(
                        &mut job.stream,
                        serde_json::json!({"request_id": job.id, "error": "deadline exceeded"}),
                        policy.max_response_bytes,
                    );
                    finish_request(&pending_requests, &last_activity);
                    continue;
                }
                set_active(&status, Some(job.id.clone()));
                if let Err(mut response) = validate(&job.raw) {
                    if let Some(object) = response.as_object_mut() {
                        object
                            .entry("request_id")
                            .or_insert_with(|| job.id.clone().into());
                    }
                    reject(&status);
                    set_active(&status, None);
                    write_response(&mut job.stream, response, policy.max_response_bytes);
                    finish_request(&pending_requests, &last_activity);
                    continue;
                }
                if model.is_none() {
                    set_state(&status, LifecycleState::Loading, None);
                    match load() {
                        Ok(loaded) => model = Some(loaded),
                        Err(error) => {
                            set_state(&status, LifecycleState::Faulted, Some(error.clone()));
                            set_active(&status, None);
                            write_response(
                                &mut job.stream,
                                serde_json::json!({"request_id": job.id, "error": format!("model load: {error}")}),
                                policy.max_response_bytes,
                            );
                            finish_request(&pending_requests, &last_activity);
                            continue;
                        }
                    }
                }
                let mut response = handle(&job.raw, &mut model);
                let response_error = response
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(object) = response.as_object_mut() {
                    object
                        .entry("request_id")
                        .or_insert_with(|| job.id.clone().into());
                }
                write_response(&mut job.stream, response, policy.max_response_bytes);
                last_model_completed = Instant::now();
                complete(&status, model.is_some(), response_error);
                finish_request(&pending_requests, &last_activity);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        let model_idle = last_model_completed.elapsed();
        if model.is_some() && model_idle >= policy.model_ttl {
            model = None;
            set_state(&status, LifecycleState::Evicted, None);
            if log_enabled(log_prefix) {
                eprintln!("{log_prefix}: model evicted after {model_idle:?} idle");
            }
        }
        if activity_idle(&last_activity) >= policy.exit_ttl
            && pending_requests.load(std::sync::atomic::Ordering::Acquire) == 0
        {
            break;
        }
    }

    stop.store(true, std::sync::atomic::Ordering::Release);
    drop(model);
    drop(endpoint_guard);
    drop(owner_lock);
    0
}

fn spawn_accept_loop(
    listener: TransportListener,
    accepted: mpsc::SyncSender<TransportStream>,
    status: Arc<Mutex<RuntimeStatus>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    pending_requests: Arc<std::sync::atomic::AtomicUsize>,
    last_activity: Arc<Mutex<Instant>>,
) {
    std::thread::spawn(move || {
        while !stop.load(std::sync::atomic::Ordering::Acquire) {
            match listener.accept() {
                Ok(mut stream) => {
                    pending_requests.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    match accepted.try_send(stream) {
                        Ok(()) => {}
                        Err(mpsc::TrySendError::Full(returned)) => {
                            stream = returned;
                            reject(&status);
                            write_response(
                                &mut stream,
                                capacity_response(None, "daemon busy"),
                                4096,
                            );
                            finish_request(&pending_requests, &last_activity);
                        }
                        Err(mpsc::TrySendError::Disconnected(_)) => {
                            finish_request(&pending_requests, &last_activity);
                            break;
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(LOOP_INTERVAL);
                }
                Err(_) => std::thread::sleep(LOOP_INTERVAL),
            }
        }
    });
}

fn spawn_reader_workers(
    accepted: mpsc::Receiver<TransportStream>,
    jobs: mpsc::SyncSender<RequestJob>,
    policy: ServerPolicy,
    status: Arc<Mutex<RuntimeStatus>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    pending_requests: Arc<std::sync::atomic::AtomicUsize>,
    last_activity: Arc<Mutex<Instant>>,
) {
    let accepted = Arc::new(Mutex::new(accepted));
    for _ in 0..READER_WORKERS {
        let accepted = Arc::clone(&accepted);
        let jobs = jobs.clone();
        let status = Arc::clone(&status);
        let stop = Arc::clone(&stop);
        let pending_requests = Arc::clone(&pending_requests);
        let last_activity = Arc::clone(&last_activity);
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                let stream = match accepted.lock().ok().and_then(|rx| rx.recv().ok()) {
                    Some(stream) => stream,
                    None => break,
                };
                read_and_queue(
                    stream,
                    &jobs,
                    policy,
                    &status,
                    &pending_requests,
                    &last_activity,
                );
            }
        });
    }
}

fn read_and_queue(
    mut stream: TransportStream,
    jobs: &mpsc::SyncSender<RequestJob>,
    policy: ServerPolicy,
    status: &Arc<Mutex<RuntimeStatus>>,
    pending_requests: &Arc<std::sync::atomic::AtomicUsize>,
    last_activity: &Arc<Mutex<Instant>>,
) {
    if stream
        .set_timeouts(CONNECTION_WRITE_TIMEOUT, CONNECTION_READ_TIMEOUT)
        .is_err()
    {
        finish_request(pending_requests, last_activity);
        return;
    }
    let raw = match read_frame(
        &mut stream,
        policy.max_request_bytes,
        CONNECTION_READ_TIMEOUT,
    ) {
        Ok(raw) => raw,
        Err(_) => {
            reject(status);
            write_response(
                &mut stream,
                serde_json::json!({"error": "request too large or incomplete"}),
                policy.max_response_bytes,
            );
            finish_request(pending_requests, last_activity);
            return;
        }
    };
    let value: serde_json::Value = match serde_json::from_str(raw.trim()) {
        Ok(value) => value,
        Err(_) => {
            reject(status);
            write_response(
                &mut stream,
                serde_json::json!({"error": "malformed request"}),
                policy.max_response_bytes,
            );
            finish_request(pending_requests, last_activity);
            return;
        }
    };
    let id = value
        .get("request_id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty() && id.len() <= 128)
        .map(ToOwned::to_owned)
        .unwrap_or_else(request_id);
    if value.get("protocol").and_then(serde_json::Value::as_u64)
        != Some(u64::from(PROTOCOL_VERSION))
    {
        reject(status);
        write_response(
            &mut stream,
            serde_json::json!({"request_id": id, "error": "protocol-version mismatch"}),
            policy.max_response_bytes,
        );
        finish_request(pending_requests, last_activity);
        return;
    }
    match value.get("op").and_then(serde_json::Value::as_str) {
        Some("ping") => {
            write_response(
                &mut stream,
                serde_json::json!({"request_id": id, "ok": true}),
                policy.max_response_bytes,
            );
            finish_request(pending_requests, last_activity);
            return;
        }
        Some("status") => {
            let response = status_response(
                status,
                &id,
                pending_requests.load(std::sync::atomic::Ordering::Acquire),
            );
            write_response(&mut stream, response, policy.max_response_bytes);
            finish_request(pending_requests, last_activity);
            return;
        }
        _ => {}
    }
    let deadline = Instant::now() + policy.request_deadline;
    match jobs.try_send(RequestJob {
        id: id.clone(),
        raw,
        deadline,
        stream,
    }) {
        Ok(()) => {}
        Err(mpsc::TrySendError::Full(mut job)) => {
            reject(status);
            write_response(
                &mut job.stream,
                capacity_response(Some(&id), "inference queue full"),
                policy.max_response_bytes,
            );
            finish_request(pending_requests, last_activity);
        }
        Err(mpsc::TrySendError::Disconnected(_)) => {
            finish_request(pending_requests, last_activity);
        }
    }
}

fn status_response(
    status: &Arc<Mutex<RuntimeStatus>>,
    request_id: &str,
    pending_requests: usize,
) -> serde_json::Value {
    let Ok(status) = status.lock() else {
        return serde_json::json!({"request_id": request_id, "error": "status unavailable"});
    };
    serde_json::json!({
        "request_id": request_id,
        "protocol": PROTOCOL_VERSION,
        "daemon_pid": std::process::id(),
        "state": status.state.as_str(),
        "state_elapsed_ms": status.state_started.elapsed().as_millis(),
        "active_request_id": status.active_request_id,
        "active_request_elapsed_ms": status
            .active_request_started
            .map(|started| started.elapsed().as_millis()),
        "completed_requests": status.completed_requests,
        "rejected_requests": status.rejected_requests,
        "last_error": status.last_error,
        "queue_capacity": INFERENCE_QUEUE_LENGTH,
        "pending_requests": pending_requests,
    })
}

fn capacity_response(request_id: Option<&str>, error: &'static str) -> serde_json::Value {
    let mut response = serde_json::json!({
        "error": error,
        "error_kind": "capacity",
        "retryable": true,
    });
    if let (Some(request_id), Some(object)) = (request_id, response.as_object_mut()) {
        object.insert("request_id".into(), request_id.into());
    }
    response
}

fn finish_request(
    pending_requests: &std::sync::atomic::AtomicUsize,
    last_activity: &Mutex<Instant>,
) {
    let previous = pending_requests.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    debug_assert!(previous > 0, "pending inference request counter underflow");
    if let Ok(mut activity) = last_activity.lock() {
        *activity = Instant::now();
    }
}

fn activity_idle(last_activity: &Mutex<Instant>) -> Duration {
    last_activity
        .lock()
        .map(|activity| activity.elapsed())
        .unwrap_or_default()
}

fn write_response(
    stream: &mut TransportStream,
    value: serde_json::Value,
    max_response_bytes: usize,
) {
    let bytes = serialize_json_frame(&value, max_response_bytes).or_else(|_| {
        serialize_json_frame(
            &serde_json::json!({"error": "response too large"}),
            max_response_bytes,
        )
    });
    if let Ok(bytes) = bytes {
        let _ = write_frame(stream, &bytes, CONNECTION_WRITE_TIMEOUT);
    }
}

fn serialize_json_frame(value: &serde_json::Value, max_bytes: usize) -> std::io::Result<Vec<u8>> {
    let mut frame = LimitedFrame::new(max_bytes);
    serde_json::to_writer(&mut frame, value)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    frame.write_all(b"\n")?;
    Ok(frame.into_inner())
}

struct LimitedFrame {
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl LimitedFrame {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(max_bytes.min(4096)),
            max_bytes,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for LimitedFrame {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(next_len) = self.bytes.len().checked_add(bytes.len()) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "daemon frame exceeds limit",
            ));
        };
        if next_len > self.max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "daemon frame exceeds limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn write_frame(
    stream: &mut TransportStream,
    bytes: &[u8],
    timeout: Duration,
) -> std::io::Result<()> {
    let deadline = Instant::now() + timeout;
    let mut written = 0usize;
    while written < bytes.len() {
        if Instant::now() >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "daemon frame write timed out",
            ));
        }
        match stream.write(&bytes[written..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "daemon frame write returned zero",
                ));
            }
            Ok(count) => written = written.saturating_add(count),
            Err(error) if retryable_io(&error) => std::thread::sleep(Duration::from_millis(5)),
            Err(error) => return Err(error),
        }
    }
    stream.flush()
}

fn read_frame(
    stream: &mut TransportStream,
    max_bytes: usize,
    timeout: Duration,
) -> std::io::Result<String> {
    let deadline = Instant::now() + timeout;
    let mut bytes = Vec::with_capacity(max_bytes.min(4096));
    let mut buffer = [0u8; 4096];
    loop {
        if Instant::now() >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "daemon frame read timed out",
            ));
        }
        match stream.read(&mut buffer) {
            #[cfg(windows)]
            Ok(0) => {
                // A byte-mode named pipe in PIPE_NOWAIT mode reports an
                // empty read as zero bytes while the peer is still alive.
                std::thread::sleep(Duration::from_millis(5));
            }
            #[cfg(not(windows))]
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "daemon frame ended before newline",
                ));
            }
            Ok(read) => {
                let remaining = max_bytes.saturating_add(1).saturating_sub(bytes.len());
                bytes.extend_from_slice(&buffer[..read.min(remaining)]);
                if bytes.len() > max_bytes {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "daemon frame exceeds limit",
                    ));
                }
                if let Some(newline) = bytes.iter().position(|byte| *byte == b'\n') {
                    bytes.truncate(newline);
                    return String::from_utf8(bytes).map_err(|error| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, error)
                    });
                }
            }
            Err(error) if retryable_io(&error) => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => return Err(error),
        }
    }
}

fn retryable_io(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::WouldBlock
        || matches!(error.raw_os_error(), Some(231 | 232 | 233 | 997))
}

fn set_state(status: &Arc<Mutex<RuntimeStatus>>, state: LifecycleState, error: Option<String>) {
    if let Ok(mut status) = status.lock() {
        if status.state != state {
            status.state_started = Instant::now();
        }
        status.state = state;
        status.last_error = error;
    }
}

fn set_active(status: &Arc<Mutex<RuntimeStatus>>, request_id: Option<String>) {
    if let Ok(mut status) = status.lock() {
        status.active_request_started = request_id.as_ref().map(|_| Instant::now());
        status.active_request_id = request_id;
    }
}

fn complete(status: &Arc<Mutex<RuntimeStatus>>, model_loaded: bool, error: Option<String>) {
    if let Ok(mut status) = status.lock() {
        status.active_request_id = None;
        status.active_request_started = None;
        status.completed_requests = status.completed_requests.saturating_add(1);
        status.last_error = error;
        let next_state = if model_loaded {
            LifecycleState::Ready
        } else {
            LifecycleState::Faulted
        };
        if status.state != next_state {
            status.state_started = Instant::now();
        }
        status.state = next_state;
    }
}

fn spawn_hung_worker_watchdog(
    endpoint: Endpoint,
    status: Arc<Mutex<RuntimeStatus>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    timeout: Option<Duration>,
    log_prefix: &'static str,
) {
    let Some(timeout) = timeout else {
        return;
    };
    std::thread::spawn(move || {
        while !stop.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::sleep(LOOP_INTERVAL);
            let timed_out = status.lock().ok().is_some_and(|status| {
                status
                    .active_request_started
                    .is_some_and(|started| started.elapsed() >= timeout)
                    || (status.state == LifecycleState::Loading
                        && status.state_started.elapsed() >= timeout)
            });
            if timed_out {
                if log_enabled(log_prefix) {
                    eprintln!(
                        "{log_prefix}: inference worker exceeded hard timeout {timeout:?}; exiting"
                    );
                }
                // Inference backends cannot be safely interrupted in-process.
                // Exiting releases the owner lock; the next client repairs the
                // stale endpoint and starts one clean model instance.
                record_spawn_failure(&endpoint, true);
                std::process::exit(70);
            }
        }
    });
}

fn reject(status: &Arc<Mutex<RuntimeStatus>>) {
    if let Ok(mut status) = status.lock() {
        status.rejected_requests = status.rejected_requests.saturating_add(1);
    }
}

fn request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let counter = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{}-{counter}", std::process::id())
}

fn log_enabled(prefix: &str) -> bool {
    let variable = if prefix == "embed-daemon" {
        "GREPPY_EMBED_DAEMON_LOG"
    } else {
        "GREPPY_SUMMARIZE_DAEMON_LOG"
    };
    std::env::var_os(variable).is_some()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(unix)]
fn ensure_private_dir(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    std::fs::create_dir_all(path)?;
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_dir() || metadata.uid() != unsafe { libc::geteuid() } {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "daemon runtime directory is not an owned directory",
        ));
    }
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    let secured = std::fs::symlink_metadata(path)?;
    if secured.permissions().mode() & 0o077 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "daemon runtime directory is accessible by other users",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn unix_runtime_dir() -> std::path::PathBuf {
    if let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") {
        let candidate = std::path::PathBuf::from(runtime).join("greppy");
        if candidate.as_os_str().len() <= 32 {
            return candidate;
        }
    }
    std::path::PathBuf::from("/tmp").join(format!("greppy-daemon-{}", unsafe { libc::geteuid() }))
}

#[cfg(windows)]
fn ensure_private_dir(path: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

#[cfg(unix)]
fn no_daemon_error(error: &std::io::Error) -> bool {
    // ENOTSOCK: a stale REGULAR file occupies the endpoint path. macOS/BSD
    // report ENOTSOCK from connect(2) in that case (Linux reports
    // ECONNREFUSED, already classified below). Treating it as anything but
    // "no daemon" would permanently brick the endpoint: only the no-daemon
    // classification triggers a spawn, and only the spawned owner repairs
    // the stale path (TransportListener::bind removes dead endpoint files).
    if error.raw_os_error() == Some(libc::ENOTSOCK) {
        return true;
    }
    matches!(
        error.kind(),
        std::io::ErrorKind::NotFound
            | std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
    )
}

#[cfg(windows)]
fn no_daemon_error(error: &std::io::Error) -> bool {
    matches!(error.raw_os_error(), Some(2 | 53 | 231 | 233))
}

struct EndpointGuard {
    endpoint: Endpoint,
}

impl EndpointGuard {
    fn new(endpoint: Endpoint) -> Self {
        Self { endpoint }
    }
}

impl Drop for EndpointGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            let _ = std::fs::remove_file(&self.endpoint.address);
        }
        #[cfg(windows)]
        let _ = &self.endpoint;
    }
}

#[cfg(unix)]
struct TransportStream(std::os::unix::net::UnixStream);

#[cfg(unix)]
impl TransportStream {
    fn connect(endpoint: &Endpoint, timeout: Duration) -> std::io::Result<Self> {
        unix_connect_with_timeout(std::path::Path::new(&endpoint.address), timeout).map(Self)
    }

    fn set_timeouts(&self, write: Duration, read: Duration) -> std::io::Result<()> {
        self.0.set_write_timeout(Some(write))?;
        self.0.set_read_timeout(Some(read))
    }
}

#[cfg(unix)]
fn unix_connect_with_timeout(
    path: &std::path::Path,
    timeout: Duration,
) -> std::io::Result<std::os::unix::net::UnixStream> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;

    let path = path.as_os_str().as_bytes();
    let mut address = std::mem::MaybeUninit::<libc::sockaddr_un>::zeroed();
    // SAFETY: A zeroed sockaddr_un is valid to initialize field by field before use.
    let address = unsafe { address.assume_init_mut() };
    if path.len() >= address.sun_path.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "daemon endpoint path is too long",
        ));
    }
    address.sun_family = libc::sa_family_t::try_from(libc::AF_UNIX).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "AF_UNIX does not fit sa_family_t",
        )
    })?;
    #[cfg(any(
        target_os = "aix",
        target_os = "freebsd",
        target_os = "haiku",
        target_os = "ios",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "tvos",
        target_os = "visionos",
        target_os = "watchos"
    ))]
    {
        address.sun_len =
            u8::try_from(std::mem::offset_of!(libc::sockaddr_un, sun_path) + path.len() + 1)
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "daemon endpoint address is too long",
                    )
                })?;
    }
    for (target, source) in address.sun_path.iter_mut().zip(path.iter().copied()) {
        *target = libc::c_char::from_ne_bytes([source]);
    }
    let address_len = libc::socklen_t::try_from(
        std::mem::offset_of!(libc::sockaddr_un, sun_path) + path.len() + 1,
    )
    .map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "daemon endpoint address is too long",
        )
    })?;

    // SAFETY: socket returns a new owned descriptor or -1 without aliasing Rust values.
    let raw = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: raw was just returned by socket and ownership is transferred exactly once.
    let socket = unsafe { OwnedFd::from_raw_fd(raw) };
    let original_flags = unsafe { libc::fcntl(socket.as_raw_fd(), libc::F_GETFL) };
    if original_flags < 0
        || unsafe {
            libc::fcntl(
                socket.as_raw_fd(),
                libc::F_SETFL,
                original_flags | libc::O_NONBLOCK,
            )
        } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    let descriptor_flags = unsafe { libc::fcntl(socket.as_raw_fd(), libc::F_GETFD) };
    if descriptor_flags < 0
        || unsafe {
            libc::fcntl(
                socket.as_raw_fd(),
                libc::F_SETFD,
                descriptor_flags | libc::FD_CLOEXEC,
            )
        } < 0
    {
        return Err(std::io::Error::last_os_error());
    }

    let connected = unsafe {
        libc::connect(
            socket.as_raw_fd(),
            std::ptr::from_ref(address).cast::<libc::sockaddr>(),
            address_len,
        )
    };
    if connected < 0 {
        let error = std::io::Error::last_os_error();
        if !matches!(
            error.raw_os_error(),
            Some(libc::EINPROGRESS | libc::EALREADY | libc::EAGAIN)
        ) {
            return Err(error);
        }
        poll_connected(socket.as_raw_fd(), timeout)?;
    }
    if unsafe { libc::fcntl(socket.as_raw_fd(), libc::F_SETFL, original_flags) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(std::os::unix::net::UnixStream::from(socket))
}

#[cfg(unix)]
fn poll_connected(fd: std::os::fd::RawFd, timeout: Duration) -> std::io::Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let timeout_ms = i32::try_from(remaining.as_millis().saturating_add(u128::from(
            !remaining.subsec_nanos().is_multiple_of(1_000_000),
        )))
        .unwrap_or(i32::MAX);
        let mut descriptor = libc::pollfd {
            fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        // SAFETY: descriptor points to one initialized pollfd for the duration of the call.
        let ready = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
        if ready == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "daemon connect timed out",
            ));
        }
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted && Instant::now() < deadline {
                continue;
            }
            return Err(error);
        }

        let mut socket_error: libc::c_int = 0;
        let mut error_len = libc::socklen_t::try_from(std::mem::size_of_val(&socket_error))
            .expect("socket error length fits socklen_t");
        // SAFETY: socket_error and error_len are valid writable values of the declared sizes.
        if unsafe {
            libc::getsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_ERROR,
                std::ptr::from_mut(&mut socket_error).cast(),
                &mut error_len,
            )
        } < 0
        {
            return Err(std::io::Error::last_os_error());
        }
        return if socket_error == 0 {
            Ok(())
        } else {
            Err(std::io::Error::from_raw_os_error(socket_error))
        };
    }
}

#[cfg(unix)]
impl Read for TransportStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buffer)
    }
}

#[cfg(unix)]
impl Write for TransportStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.0.write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

#[cfg(unix)]
struct TransportListener(std::os::unix::net::UnixListener);

#[cfg(unix)]
impl TransportListener {
    fn bind(endpoint: &Endpoint) -> std::io::Result<Self> {
        let path = std::path::Path::new(&endpoint.address);
        if let Some(parent) = path.parent() {
            ensure_private_dir(parent)?;
        }
        if path.exists() {
            if TransportStream::connect(endpoint, Duration::from_millis(100)).is_ok() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    "daemon endpoint is live",
                ));
            }
            std::fs::remove_file(path)?;
        }
        let listener = std::os::unix::net::UnixListener::bind(path)?;
        use std::os::unix::fs::PermissionsExt;
        if let Err(error) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .and_then(|()| listener.set_nonblocking(true))
        {
            let _ = std::fs::remove_file(path);
            return Err(error);
        }
        Ok(Self(listener))
    }

    fn accept(&self) -> std::io::Result<TransportStream> {
        self.0.accept().map(|(stream, _)| TransportStream(stream))
    }
}

#[cfg(windows)]
struct TransportStream(std::fs::File);

#[cfg(windows)]
impl TransportStream {
    fn connect(endpoint: &Endpoint, timeout: Duration) -> std::io::Result<Self> {
        use std::os::windows::io::FromRawHandle;
        use windows_sys::Win32::Foundation::{
            ERROR_PIPE_BUSY, ERROR_SEM_TIMEOUT, GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE,
        };
        use windows_sys::Win32::Storage::FileSystem::{
            CreateFileW, FILE_ATTRIBUTE_NORMAL, OPEN_EXISTING,
        };
        use windows_sys::Win32::System::Pipes::{
            SetNamedPipeHandleState, WaitNamedPipeW, PIPE_NOWAIT, PIPE_READMODE_BYTE,
        };

        let deadline = Instant::now() + timeout;
        let name = wide_string(&endpoint.address);
        loop {
            unsafe {
                if WaitNamedPipeW(name.as_ptr(), 50) == 0 {
                    let error = std::io::Error::last_os_error();
                    if !windows_pipe_busy(&error, ERROR_PIPE_BUSY, ERROR_SEM_TIMEOUT)
                        || Instant::now() >= deadline
                    {
                        return Err(error);
                    }
                    continue;
                }
                let handle = CreateFileW(
                    name.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    std::ptr::null(),
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL,
                    std::ptr::null_mut(),
                );
                if handle == INVALID_HANDLE_VALUE {
                    let error = std::io::Error::last_os_error();
                    if windows_pipe_busy(&error, ERROR_PIPE_BUSY, ERROR_SEM_TIMEOUT)
                        && Instant::now() < deadline
                    {
                        continue;
                    }
                    return Err(error);
                }
                let mode = PIPE_READMODE_BYTE | PIPE_NOWAIT;
                if SetNamedPipeHandleState(handle, &mode, std::ptr::null(), std::ptr::null()) == 0 {
                    let error = std::io::Error::last_os_error();
                    drop(std::fs::File::from_raw_handle(handle));
                    return Err(error);
                }
                return Ok(Self(std::fs::File::from_raw_handle(handle)));
            }
        }
    }

    fn set_timeouts(&self, _write: Duration, _read: Duration) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(windows)]
fn windows_pipe_busy(error: &std::io::Error, pipe_busy: u32, timeout: u32) -> bool {
    error.raw_os_error().is_some_and(|code| {
        u32::try_from(code).is_ok_and(|code| code == pipe_busy || code == timeout)
    })
}

#[cfg(windows)]
impl Read for TransportStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buffer)
    }
}

#[cfg(windows)]
impl Write for TransportStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.0.write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

#[cfg(windows)]
struct TransportListener {
    endpoint: String,
    pending: Mutex<Option<std::fs::File>>,
}

#[cfg(windows)]
impl TransportListener {
    fn bind(endpoint: &Endpoint) -> std::io::Result<Self> {
        Ok(Self {
            endpoint: endpoint.address.clone(),
            pending: Mutex::new(Some(create_named_pipe(&endpoint.address)?)),
        })
    }

    fn accept(&self) -> std::io::Result<TransportStream> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{ERROR_PIPE_CONNECTED, ERROR_PIPE_LISTENING};
        use windows_sys::Win32::System::Pipes::ConnectNamedPipe;

        let mut pending = self
            .pending
            .lock()
            .map_err(|_| std::io::Error::other("named-pipe listener poisoned"))?;
        if pending.is_none() {
            *pending = Some(create_named_pipe(&self.endpoint)?);
        }
        let handle = pending
            .as_ref()
            .expect("pending pipe created")
            .as_raw_handle();
        if unsafe { ConnectNamedPipe(handle, std::ptr::null_mut()) } != 0 {
            return Ok(TransportStream(
                pending.take().expect("connected pipe must exist"),
            ));
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(code) if code == i32::try_from(ERROR_PIPE_CONNECTED).unwrap_or(535) => Ok(
                TransportStream(pending.take().expect("connected pipe must exist")),
            ),
            Some(code) if code == i32::try_from(ERROR_PIPE_LISTENING).unwrap_or(536) => Err(
                std::io::Error::new(std::io::ErrorKind::WouldBlock, "pipe has no client"),
            ),
            _ => {
                pending.take();
                Err(error)
            }
        }
    }
}

#[cfg(windows)]
fn create_named_pipe(endpoint: &str) -> std::io::Result<std::fs::File> {
    use std::os::windows::io::FromRawHandle;
    use windows_sys::Win32::Foundation::{LocalFree, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES};
    use windows_sys::Win32::Storage::FileSystem::PIPE_ACCESS_DUPLEX;
    use windows_sys::Win32::System::Pipes::{
        CreateNamedPipeW, PIPE_NOWAIT, PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS,
        PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES,
    };

    let descriptor_text = wide_string("D:P(A;;GA;;;OW)(A;;GA;;;SY)");
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            descriptor_text.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    let length = u32::try_from(std::mem::size_of::<SECURITY_ATTRIBUTES>())
        .map_err(|_| std::io::Error::other("security attributes size does not fit u32"))?;
    let attributes = SECURITY_ATTRIBUTES {
        nLength: length,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    };
    let name = wide_string(endpoint);
    let handle = unsafe {
        CreateNamedPipeW(
            name.as_ptr(),
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
            PIPE_UNLIMITED_INSTANCES,
            64 * 1024,
            256 * 1024,
            100,
            &attributes,
        )
    };
    unsafe {
        LocalFree(descriptor);
    }
    if handle == INVALID_HANDLE_VALUE {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(unsafe { std::fs::File::from_raw_handle(handle) })
    }
}

#[cfg(windows)]
fn wide_string(value: &str) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    std::ffi::OsStr::new(value)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_identity_is_stable_and_versioned() {
        let a = Endpoint::for_identity("summary", "model|prompt|cpu").unwrap();
        let b = Endpoint::for_identity("summary", "model|prompt|cpu").unwrap();
        let c = Endpoint::for_identity("summary", "model|prompt|cuda").unwrap();
        assert_eq!(a.address(), b.address());
        assert_ne!(a.address(), c.address());
        assert!(a.address().contains("summary-"));
    }

    #[test]
    fn retry_backoff_is_bounded() {
        let delays = retry_delays().collect::<Vec<_>>();
        assert_eq!(delays.first(), Some(&Duration::from_millis(50)));
        assert!(delays.iter().copied().sum::<Duration>() < Duration::from_secs(4));
    }

    #[test]
    fn crash_cooldown_is_endpoint_scoped_and_not_refreshed() {
        let endpoint = Endpoint::for_identity(
            "cooldown-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        assert!(!cooldown_active(&endpoint));
        record_spawn_failure(&endpoint, false);
        assert!(!cooldown_active(&endpoint));

        record_spawn_failure(&endpoint, true);
        assert!(cooldown_active(&endpoint));
        let first_modified = endpoint
            .cooldown_path()
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        record_spawn_failure(&endpoint, true);
        let second_modified = endpoint
            .cooldown_path()
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap();
        assert_eq!(first_modified, second_modified);
        assert_eq!(
            spawn_once(&endpoint, || panic!("cooldown must suppress spawn")),
            SpawnOutcome::Cooldown
        );
        let _ = std::fs::remove_file(endpoint.cooldown_path());
    }

    #[test]
    fn spawn_lock_contention_is_distinct_from_an_absent_daemon() {
        let endpoint = Endpoint::for_identity(
            "spawn-contention-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let lock = greppy_core::cache::acquire_named_lock(
            &endpoint.spawn_lock_name(),
            greppy_core::cache::LockMode::Exclusive,
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            spawn_once(&endpoint, || panic!("contender must not spawn")),
            SpawnOutcome::Contended
        );
        drop(lock);
    }

    #[test]
    fn json_frame_limit_counts_the_newline_at_the_exact_boundary() {
        let value = serde_json::json!({"result": "ok"});
        let mut expected = serde_json::to_vec(&value).unwrap();
        expected.push(b'\n');

        assert_eq!(
            serialize_json_frame(&value, expected.len()).unwrap(),
            expected
        );
        let error = serialize_json_frame(&value, expected.len() - 1).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn client_request_limit_is_enforced_before_connect() {
        let endpoint = Endpoint::for_identity(
            "request-limit-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let request_value = serde_json::json!({
            "op": "infer",
            "request_id": "fixed-request-id",
            "payload": "bounded",
        });
        let mut framed_value = request_value.clone();
        framed_value
            .as_object_mut()
            .unwrap()
            .insert("protocol".into(), PROTOCOL_VERSION.into());
        let exact_limit = serialize_json_frame(&framed_value, 4096).unwrap().len();

        assert_eq!(
            request(
                &endpoint,
                request_value.clone(),
                Duration::from_millis(50),
                exact_limit - 1,
                4096,
            ),
            RequestOutcome::Failed
        );
        assert_eq!(
            request(
                &endpoint,
                request_value,
                Duration::from_millis(50),
                exact_limit,
                4096,
            ),
            RequestOutcome::NoDaemon
        );
    }

    #[test]
    fn retryable_capacity_responses_are_explicitly_classified() {
        let response = capacity_response(Some("request-1"), "inference queue full");
        assert_eq!(response["error_kind"], "capacity");
        assert_eq!(response["retryable"], true);
        assert!(retryable_capacity_response(&response));
        assert!(retryable_capacity_response(
            &serde_json::json!({"error": "daemon busy"})
        ));
        assert!(!retryable_capacity_response(
            &serde_json::json!({"error": "model load failed"})
        ));
    }

    #[cfg(unix)]
    #[test]
    fn stale_non_socket_endpoint_classifies_as_no_daemon() {
        // A regular file at the endpoint path must classify as "no daemon"
        // so the client attempts a spawn and the new owner repairs the path
        // (macOS connect(2) reports ENOTSOCK there; Linux ECONNREFUSED).
        // Anything else deadlocks the endpoint forever: no spawn, no repair.
        let endpoint = Endpoint::for_identity(
            "stale-file-client-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        std::fs::write(endpoint.address(), b"stale").unwrap();
        let outcome = request(
            &endpoint,
            serde_json::json!({"op": "status"}),
            Duration::from_millis(200),
            4096,
            4096,
        );
        std::fs::remove_file(endpoint.address()).unwrap();
        assert_eq!(outcome, RequestOutcome::NoDaemon);
    }

    #[cfg(unix)]
    #[test]
    fn daemon_owner_repairs_stale_endpoint() {
        use std::os::unix::fs::FileTypeExt;

        let endpoint = Endpoint::for_identity(
            "stale-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        std::fs::write(endpoint.address(), b"stale").unwrap();
        let listener = TransportListener::bind(&endpoint).unwrap();
        assert!(std::fs::symlink_metadata(endpoint.address())
            .unwrap()
            .file_type()
            .is_socket());
        drop(listener);
        std::fs::remove_file(endpoint.address()).unwrap();
    }

    #[test]
    fn local_transport_round_trip() {
        let endpoint = Endpoint::for_identity(
            "transport-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let listener = TransportListener::bind(&endpoint).unwrap();
        let server = std::thread::spawn(move || loop {
            match listener.accept() {
                Ok(mut stream) => {
                    let request = read_frame(&mut stream, 4096, Duration::from_secs(2)).unwrap();
                    let value: serde_json::Value = serde_json::from_str(&request).unwrap();
                    assert_eq!(
                        value.get("protocol").and_then(serde_json::Value::as_u64),
                        Some(u64::from(PROTOCOL_VERSION))
                    );
                    write_frame(
                        &mut stream,
                        b"{\"state\":\"ready\"}\n",
                        Duration::from_secs(2),
                    )
                    .unwrap();
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("transport accept failed: {error}"),
            }
        });
        let response = request(
            &endpoint,
            serde_json::json!({"op": "status"}),
            Duration::from_secs(2),
            4096,
            4096,
        );
        assert!(matches!(
            response,
            RequestOutcome::Response(ref value) if value["state"] == "ready"
        ));
        server.join().unwrap();
        #[cfg(unix)]
        let _ = std::fs::remove_file(endpoint.address());
    }

    #[test]
    fn capacity_retry_deadline_preserves_daemon_busy_outcome() {
        let endpoint = Endpoint::for_identity(
            "capacity-deadline-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let listener = TransportListener::bind(&endpoint).unwrap();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let server_stop = Arc::clone(&stop);
        let server = std::thread::spawn(move || {
            while !server_stop.load(std::sync::atomic::Ordering::Acquire) {
                match listener.accept() {
                    Ok(mut stream) => {
                        read_frame(&mut stream, 4096, Duration::from_millis(100)).unwrap();
                        write_response(
                            &mut stream,
                            capacity_response(None, "inference queue full"),
                            4096,
                        );
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("capacity server accept failed: {error}"),
                }
            }
        });

        let started = Instant::now();
        let outcome = request(
            &endpoint,
            serde_json::json!({"op": "infer"}),
            Duration::from_millis(75),
            4096,
            4096,
        );
        stop.store(true, std::sync::atomic::Ordering::Release);
        server.join().unwrap();
        assert_eq!(outcome, RequestOutcome::DaemonBusy);
        assert!(started.elapsed() < Duration::from_secs(1));
        #[cfg(unix)]
        let _ = std::fs::remove_file(endpoint.address());
    }

    #[test]
    fn thirty_two_spawn_contenders_have_one_owner() {
        let endpoint = Endpoint::for_identity(
            "lock-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(33));
        let acquired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..32 {
            let barrier = Arc::clone(&barrier);
            let acquired = Arc::clone(&acquired);
            let lock_name = endpoint.lock_name();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                let lock = greppy_core::cache::acquire_named_lock(
                    &lock_name,
                    greppy_core::cache::LockMode::Exclusive,
                    true,
                )
                .expect("lock attempt");
                if lock.is_some() {
                    acquired.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                barrier.wait();
                drop(lock);
            }));
        }
        barrier.wait();
        barrier.wait();
        for thread in threads {
            thread.join().expect("spawn contender");
        }
        assert_eq!(acquired.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn live_server_serializes_clients_evicts_and_reloads_one_model() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let endpoint = Endpoint::for_identity(
            "lifecycle-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let server_endpoint = endpoint.clone();
        let server_address = endpoint.address().to_string();
        let loads = Arc::new(AtomicUsize::new(0));
        let handled = Arc::new(AtomicUsize::new(0));
        let server_loads = Arc::clone(&loads);
        let server_handled = Arc::clone(&handled);
        let server = std::thread::spawn(move || {
            run_server(
                server_endpoint,
                &server_address,
                ServerPolicy {
                    model_ttl: Duration::from_millis(150),
                    exit_ttl: Duration::from_millis(700),
                    request_deadline: Duration::from_secs(3),
                    hard_request_timeout: None,
                    max_request_bytes: 4096,
                    max_response_bytes: 4096,
                },
                false,
                move || {
                    let owner = server_loads.fetch_add(1, Ordering::SeqCst) + 1;
                    Ok::<_, String>(owner)
                },
                |raw| {
                    let value: serde_json::Value = serde_json::from_str(raw)
                        .map_err(|_| serde_json::json!({"error": "malformed request"}))?;
                    if value.get("op").and_then(serde_json::Value::as_str) == Some("infer") {
                        Ok(())
                    } else {
                        Err(serde_json::json!({"error": "unsupported operation"}))
                    }
                },
                move |_raw, model| {
                    server_handled.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(15));
                    serde_json::json!({
                        "ok": model.is_some(),
                        "model_owner": model.as_ref().copied(),
                    })
                },
                "lifecycle-test",
            )
        });

        let mut reachable = false;
        for _ in 0..100 {
            if matches!(
                request(
                    &endpoint,
                    serde_json::json!({"op": "status"}),
                    Duration::from_millis(250),
                    4096,
                    4096,
                ),
                RequestOutcome::Response(_)
            ) {
                reachable = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(reachable, "lifecycle daemon became reachable");

        let barrier = Arc::new(std::sync::Barrier::new(33));
        let mut clients = Vec::new();
        for _ in 0..32 {
            let barrier = Arc::clone(&barrier);
            let endpoint = endpoint.clone();
            clients.push(std::thread::spawn(move || {
                barrier.wait();
                request(
                    &endpoint,
                    serde_json::json!({"op": "infer"}),
                    Duration::from_secs(5),
                    4096,
                    4096,
                )
            }));
        }
        barrier.wait();
        let mut model_owners = std::collections::BTreeSet::new();
        for client in clients {
            match client.join().expect("client thread") {
                RequestOutcome::Response(value) if value["ok"] == true => {
                    model_owners.insert(
                        value["model_owner"]
                            .as_u64()
                            .expect("successful response identifies its model owner"),
                    );
                }
                RequestOutcome::Response(value) => {
                    panic!("client received error response: {value}")
                }
                RequestOutcome::DaemonBusy | RequestOutcome::NoDaemon | RequestOutcome::Failed => {
                    panic!("live daemon lost a client request")
                }
            }
        }
        assert_eq!(model_owners, std::collections::BTreeSet::from([1]));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(handled.load(Ordering::SeqCst), 32);
        assert!(
            diagnostic(&endpoint)["rejected_requests"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "test did not exercise retryable capacity responses"
        );

        let evict_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let state = diagnostic(&endpoint)["state"].as_str().map(str::to_string);
            if state.as_deref() == Some("evicted") {
                break;
            }
            assert!(
                Instant::now() < evict_deadline,
                "model did not evict: {state:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(matches!(
            request(
                &endpoint,
                serde_json::json!({"op": "infer"}),
                Duration::from_secs(3),
                4096,
                4096,
            ),
            RequestOutcome::Response(ref value) if value["ok"] == true
        ));
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        assert_eq!(server.join().expect("server thread"), 0);
    }

    #[cfg(unix)]
    #[test]
    fn slow_client_does_not_block_inference_or_prematurely_end_server() {
        let endpoint = Endpoint::for_identity(
            "slow-client-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let server_endpoint = endpoint.clone();
        let server_address = endpoint.address().to_string();
        let server = std::thread::spawn(move || {
            run_server(
                server_endpoint,
                &server_address,
                ServerPolicy {
                    model_ttl: Duration::from_secs(2),
                    exit_ttl: Duration::from_millis(300),
                    request_deadline: Duration::from_secs(1),
                    hard_request_timeout: None,
                    max_request_bytes: 4096,
                    max_response_bytes: 4096,
                },
                false,
                || Ok::<_, String>(()),
                |_| Ok(()),
                |_raw, model| serde_json::json!({"ok": model.is_some()}),
                "slow-client-test",
            )
        });
        wait_for_server(&endpoint);

        let mut slow = TransportStream::connect(&endpoint, Duration::from_secs(1)).unwrap();
        slow.set_timeouts(Duration::from_secs(1), Duration::from_secs(1))
            .unwrap();
        slow.write_all(b"{").unwrap();
        std::thread::sleep(Duration::from_millis(30));

        let started = Instant::now();
        assert!(matches!(
            request(
                &endpoint,
                serde_json::json!({"op": "infer"}),
                Duration::from_secs(1),
                4096,
                4096,
            ),
            RequestOutcome::Response(ref value) if value["ok"] == true
        ));
        assert!(started.elapsed() < Duration::from_millis(500));

        std::thread::sleep(Duration::from_millis(400));
        let status = diagnostic(&endpoint);
        assert!(status["pending_requests"].as_u64().unwrap_or(0) >= 1);
        drop(slow);
        assert_eq!(server.join().expect("slow-client server"), 0);
    }

    #[test]
    fn saturated_queue_rejects_work_and_expires_queued_deadlines() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let endpoint = Endpoint::for_identity(
            "queue-test",
            &format!("{}-{}", std::process::id(), request_id()),
        )
        .unwrap();
        let server_endpoint = endpoint.clone();
        let server_address = endpoint.address().to_string();
        let loads = Arc::new(AtomicUsize::new(0));
        let server_loads = Arc::clone(&loads);
        let server = std::thread::spawn(move || {
            run_server(
                server_endpoint,
                &server_address,
                ServerPolicy {
                    model_ttl: Duration::from_secs(2),
                    exit_ttl: Duration::from_millis(500),
                    request_deadline: Duration::from_millis(50),
                    hard_request_timeout: None,
                    max_request_bytes: 4096,
                    max_response_bytes: 4096,
                },
                false,
                move || {
                    server_loads.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, String>(())
                },
                |_| Ok(()),
                |_raw, model| {
                    std::thread::sleep(Duration::from_millis(150));
                    serde_json::json!({"ok": model.is_some()})
                },
                "queue-test",
            )
        });
        wait_for_server(&endpoint);

        let barrier = Arc::new(std::sync::Barrier::new(21));
        let mut clients = Vec::new();
        for _ in 0..20 {
            let endpoint = endpoint.clone();
            let barrier = Arc::clone(&barrier);
            clients.push(std::thread::spawn(move || {
                barrier.wait();
                request(
                    &endpoint,
                    serde_json::json!({"op": "infer"}),
                    Duration::from_secs(3),
                    4096,
                    4096,
                )
            }));
        }
        barrier.wait();

        let mut completed = 0usize;
        let mut deadline_rejections = 0usize;
        for client in clients {
            let RequestOutcome::Response(value) = client.join().expect("queue client") else {
                panic!("live queue server lost a client");
            };
            match value.get("error").and_then(serde_json::Value::as_str) {
                Some("deadline exceeded") => deadline_rejections += 1,
                Some(other) => panic!("unexpected queue response: {other}"),
                None if value["ok"] == true => completed += 1,
                None => panic!("unexpected queue response: {value}"),
            }
        }
        assert!(completed >= 1);
        assert!(deadline_rejections >= 1);
        assert_eq!(completed + deadline_rejections, 20);
        assert!(
            diagnostic(&endpoint)["rejected_requests"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(server.join().expect("queue server"), 0);
    }

    const CRASH_HELPER_IDENTITY: &str = "GREPPY_TEST_DAEMON_CRASH_IDENTITY";
    const HANG_HELPER: &str = "GREPPY_TEST_DAEMON_HANG";
    const HANG_LOAD_HELPER: &str = "GREPPY_TEST_DAEMON_HANG_LOAD";

    #[test]
    fn daemon_subprocess_helper() {
        let Ok(identity) = std::env::var(CRASH_HELPER_IDENTITY) else {
            return;
        };
        let endpoint = Endpoint::for_identity("crash-test", &identity).unwrap();
        let address = endpoint.address().to_string();
        let hang = std::env::var_os(HANG_HELPER).is_some();
        let hang_load = std::env::var_os(HANG_LOAD_HELPER).is_some();
        let code = run_server(
            endpoint,
            &address,
            ServerPolicy {
                model_ttl: Duration::from_secs(5),
                exit_ttl: Duration::from_secs(10),
                request_deadline: Duration::from_secs(1),
                hard_request_timeout: (hang || hang_load).then_some(Duration::from_millis(100)),
                max_request_bytes: 4096,
                max_response_bytes: 4096,
            },
            hang_load,
            move || {
                if hang_load {
                    std::thread::sleep(Duration::from_secs(5));
                }
                Ok::<_, String>(())
            },
            |_| Ok(()),
            move |_raw, model| {
                if hang {
                    std::thread::sleep(Duration::from_secs(5));
                }
                serde_json::json!({"ok": model.is_some()})
            },
            "crash-test",
        );
        assert_eq!(code, 0);
    }

    #[test]
    fn hung_worker_exits_and_replacement_repairs_endpoint() {
        let identity = format!("{}-{}", std::process::id(), request_id());
        let endpoint = Endpoint::for_identity("crash-test", &identity).unwrap();
        let mut hung = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("inference_daemon::tests::daemon_subprocess_helper")
            .arg("--nocapture")
            .env(CRASH_HELPER_IDENTITY, &identity)
            .env(HANG_HELPER, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn hung daemon test child");
        wait_for_server(&endpoint);

        assert!(matches!(
            request(
                &endpoint,
                serde_json::json!({"op": "infer"}),
                Duration::from_secs(1),
                4096,
                4096,
            ),
            RequestOutcome::Failed | RequestOutcome::NoDaemon
        ));
        assert_eq!(hung.wait().expect("reap watchdog daemon").code(), Some(70));
        assert!(cooldown_active(&endpoint));

        let mut replacement = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("inference_daemon::tests::daemon_subprocess_helper")
            .arg("--nocapture")
            .env(CRASH_HELPER_IDENTITY, &identity)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn replacement daemon test child");
        wait_for_server(&endpoint);
        assert!(matches!(
            request(
                &endpoint,
                serde_json::json!({"op": "infer"}),
                Duration::from_secs(1),
                4096,
                4096,
            ),
            RequestOutcome::Response(ref value) if value["ok"] == true
        ));
        replacement.kill().expect("kill replacement daemon child");
        let _ = replacement.wait();
        #[cfg(unix)]
        let _ = std::fs::remove_file(endpoint.address());
    }

    #[test]
    fn hung_prewarm_load_is_terminated() {
        let identity = format!("{}-{}", std::process::id(), request_id());
        let endpoint = Endpoint::for_identity("crash-test", &identity).unwrap();
        let mut hung = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("inference_daemon::tests::daemon_subprocess_helper")
            .arg("--nocapture")
            .env(CRASH_HELPER_IDENTITY, &identity)
            .env(HANG_LOAD_HELPER, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn hung prewarm daemon test child");
        let deadline = Instant::now() + Duration::from_secs(5);
        let status = loop {
            if let Some(status) = hung.try_wait().expect("poll prewarm watchdog daemon") {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = hung.kill();
                let _ = hung.wait();
                panic!("prewarm watchdog daemon did not terminate");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(status.code(), Some(70));
        assert!(cooldown_active(&endpoint));
        #[cfg(unix)]
        let _ = std::fs::remove_file(endpoint.address());
        let _ = std::fs::remove_file(endpoint.cooldown_path());
    }

    #[test]
    fn killed_daemon_is_replaced_and_stale_endpoint_is_repaired() {
        let identity = format!("{}-{}", std::process::id(), request_id());
        let endpoint = Endpoint::for_identity("crash-test", &identity).unwrap();
        let spawn = || {
            std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("inference_daemon::tests::daemon_subprocess_helper")
                .arg("--nocapture")
                .env(CRASH_HELPER_IDENTITY, &identity)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn daemon test child")
        };

        let mut first = spawn();
        wait_for_server(&endpoint);
        first.kill().expect("kill first daemon child");
        let first_status = first.wait().expect("reap first daemon child");
        assert!(!first_status.success());

        let mut second = spawn();
        wait_for_server(&endpoint);
        assert!(matches!(
            request(
                &endpoint,
                serde_json::json!({"op": "infer"}),
                Duration::from_secs(1),
                4096,
                4096,
            ),
            RequestOutcome::Response(ref value) if value["ok"] == true
        ));
        second.kill().expect("kill replacement daemon child");
        let _ = second.wait();
        #[cfg(unix)]
        let _ = std::fs::remove_file(endpoint.address());
    }

    fn wait_for_server(endpoint: &Endpoint) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if matches!(
                request(
                    endpoint,
                    serde_json::json!({"op": "status"}),
                    Duration::from_millis(250),
                    4096,
                    4096,
                ),
                RequestOutcome::Response(_)
            ) {
                return;
            }
            assert!(Instant::now() < deadline, "daemon did not become reachable");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(unix)]
    #[test]
    fn frame_reader_rejects_oversize_and_slow_clients() {
        let (reader, mut writer) = std::os::unix::net::UnixStream::pair().unwrap();
        writer.write_all(b"1234\n").unwrap();
        assert_eq!(
            read_frame(&mut TransportStream(reader), 5, Duration::from_millis(100)).unwrap(),
            "1234"
        );

        let (reader, mut writer) = std::os::unix::net::UnixStream::pair().unwrap();
        writer.write_all(b"1234\n").unwrap();
        let error =
            read_frame(&mut TransportStream(reader), 4, Duration::from_millis(100)).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

        let (reader, _writer) = std::os::unix::net::UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        let started = Instant::now();
        let error =
            read_frame(&mut TransportStream(reader), 16, Duration::from_millis(30)).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
