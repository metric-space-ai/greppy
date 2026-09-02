use crate::WorkspaceCore;
use notify::{Config, PollWatcher, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender, TrySendError},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const EVENT_QUEUE_CAPACITY: usize = 4_096;
const RECOMMENDED_READY_TIMEOUT: Duration = Duration::from_secs(2);
const POLL_READY_TIMEOUT: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(250);
type TrackerEvent = notify::Result<notify::Event>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WatcherBackend {
    Recommended,
    Poll,
}

impl WatcherBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Recommended => "recommended",
            Self::Poll => "poll-250ms",
        }
    }
}

enum LiveWatcher {
    Recommended(RecommendedWatcher),
    Poll(PollWatcher),
}

impl LiveWatcher {
    fn watch(&mut self, path: &Path, mode: RecursiveMode) -> notify::Result<()> {
        match self {
            Self::Recommended(watcher) => watcher.watch(path, mode),
            Self::Poll(watcher) => watcher.watch(path, mode),
        }
    }
}

struct PreparedWatcher {
    watcher: LiveWatcher,
    backend: WatcherBackend,
    armed: Arc<AtomicBool>,
    readiness_events: Receiver<TrackerEvent>,
    readiness_overflowed: Arc<AtomicBool>,
}

/// Starts the repository mutation tracker for one workspace data root.
///
/// Native provider processes call this on Linux and Windows. macOS calls it
/// from the unsandboxed Greppy agent process because the sandboxed FSKit
/// extension cannot watch arbitrary user repositories.
pub fn spawn_repository_tracker(data_root: PathBuf) -> io::Result<thread::JoinHandle<()>> {
    let core = Arc::new(WorkspaceCore::open(data_root.join("core")).map_err(io::Error::other)?);
    thread::Builder::new()
        .name("greppy-repository-tracker".into())
        .spawn(move || supervise(core))
}

fn supervise(core: Arc<WorkspaceCore>) {
    let mut watchers = HashMap::<PathBuf, LiveWatcher>::new();
    let mut last_heartbeat = Instant::now();
    loop {
        let requests = match core.pending_repository_trackers() {
            Ok(requests) => requests,
            Err(_) => {
                thread::sleep(Duration::from_millis(250));
                continue;
            }
        };
        for repository in requests {
            watchers.remove(&repository);
            let git_dir = match repository_git_dir(&repository) {
                Ok(git_dir) => git_dir,
                Err(error) => {
                    let _ = core.mark_repository_tracker_gap(
                        &repository,
                        &format!("cannot resolve Git directory: {error}"),
                        now_ms(),
                    );
                    continue;
                }
            };
            match prepare_watcher(core.clone(), &repository, &git_dir) {
                Ok(prepared) => {
                    // Publish the callback routing before exposing Active in
                    // SQLite. The client writes its fence as soon as it sees
                    // Active; activating first allowed that event to fall into
                    // the no-longer-consumed readiness queue under load.
                    prepared.armed.store(true, Ordering::Release);
                    match core.activate_repository_tracker(&repository, now_ms()) {
                        Ok(active) => match verify_or_fallback_active_watcher(
                            core.clone(),
                            &repository,
                            &git_dir,
                            &active,
                            prepared,
                        ) {
                            Ok(watcher) => {
                                watchers.insert(repository, watcher);
                            }
                            Err(error) => {
                                let _ = core.mark_repository_tracker_gap(
                                    &repository,
                                    &format!("active watcher fence failed: {error}"),
                                    now_ms(),
                                );
                            }
                        },
                        Err(error) => {
                            prepared.armed.store(false, Ordering::Release);
                            let _ = core.mark_repository_tracker_gap(
                                &repository,
                                &format!("cannot activate watcher: {error}"),
                                now_ms(),
                            );
                        }
                    }
                }
                Err(error) => {
                    let _ = core.mark_repository_tracker_gap(
                        &repository,
                        &format!("cannot create watcher: {error}"),
                        now_ms(),
                    );
                }
            }
        }
        if last_heartbeat.elapsed()
            >= Duration::from_millis(crate::repository_tracker::HEARTBEAT_INTERVAL_MS)
        {
            let heartbeat = now_ms();
            watchers.retain(|repository, _| {
                match core.heartbeat_repository_tracker(repository, heartbeat) {
                    Ok(()) => true,
                    Err(crate::Error::ConcurrentRepositoryMutation) => false,
                    Err(_) => true,
                }
            });
            last_heartbeat = Instant::now();
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn verify_or_fallback_active_watcher(
    core: Arc<WorkspaceCore>,
    repository: &Path,
    git_dir: &Path,
    active: &crate::RepositoryTrackerStatus,
    prepared: PreparedWatcher,
) -> Result<LiveWatcher, String> {
    let timeout = if prepared.backend == WatcherBackend::Recommended {
        RECOMMENDED_READY_TIMEOUT
    } else {
        POLL_READY_TIMEOUT
    };
    match wait_for_active_watcher_probe(
        &core,
        repository,
        git_dir,
        active.epoch,
        active.generation,
        timeout,
    ) {
        Ok(()) => {
            trace_tracker_state(repository, "tracker-active", prepared.backend, None);
            Ok(prepared.watcher)
        }
        Err(primary_error)
            if cfg!(target_os = "macos") && prepared.backend == WatcherBackend::Recommended =>
        {
            prepared.armed.store(false, Ordering::Release);
            drop(prepared.watcher);
            trace_tracker_state(
                repository,
                "tracker-poll-fallback",
                WatcherBackend::Poll,
                Some(&primary_error.to_string()),
            );
            let fallback = install_and_probe_watcher(
                core.clone(),
                repository,
                git_dir,
                WatcherBackend::Poll,
                POLL_READY_TIMEOUT,
            )?;
            fallback.armed.store(true, Ordering::Release);
            let generation = core
                .repository_tracker_status(repository)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "repository tracker disappeared during fallback".to_owned())?
                .generation;
            wait_for_active_watcher_probe(
                &core,
                repository,
                git_dir,
                active.epoch,
                generation,
                POLL_READY_TIMEOUT,
            )
            .map_err(|error| format!("PollWatcher active fence failed: {error}"))?;
            trace_tracker_state(repository, "tracker-active", WatcherBackend::Poll, None);
            Ok(fallback.watcher)
        }
        Err(error) => Err(format!(
            "{} active fence failed: {error}",
            prepared.backend.label()
        )),
    }
}

fn prepare_watcher(
    core: Arc<WorkspaceCore>,
    repository: &Path,
    git_dir: &Path,
) -> Result<PreparedWatcher, String> {
    match install_and_probe_watcher(
        core.clone(),
        repository,
        git_dir,
        WatcherBackend::Recommended,
        RECOMMENDED_READY_TIMEOUT,
    ) {
        Ok(prepared) => Ok(prepared),
        Err(recommended_error) if cfg!(target_os = "macos") => {
            trace_tracker_state(
                repository,
                "tracker-poll-fallback",
                WatcherBackend::Poll,
                Some(&recommended_error),
            );
            install_and_probe_watcher(
                core,
                repository,
                git_dir,
                WatcherBackend::Poll,
                POLL_READY_TIMEOUT,
            )
            .map_err(|poll_error| {
                format!(
                    "recommended watcher failed ({recommended_error}); PollWatcher failed ({poll_error})"
                )
            })
        }
        Err(error) => Err(error),
    }
}

fn install_and_probe_watcher(
    core: Arc<WorkspaceCore>,
    repository: &Path,
    git_dir: &Path,
    backend: WatcherBackend,
    timeout: Duration,
) -> Result<PreparedWatcher, String> {
    let mut prepared = build_watcher(core, repository, git_dir, backend)
        .map_err(|error| format!("cannot create {} watcher: {error}", backend.label()))?;
    prepared
        .watcher
        .watch(repository, RecursiveMode::Recursive)
        .map_err(|error| format!("cannot watch repository with {}: {error}", backend.label()))?;
    if !git_dir.starts_with(repository) {
        prepared
            .watcher
            .watch(git_dir, RecursiveMode::Recursive)
            .map_err(|error| {
                format!(
                    "cannot watch linked Git directory with {}: {error}",
                    backend.label()
                )
            })?;
    }
    wait_for_watcher_probe(
        git_dir,
        &prepared.readiness_events,
        &prepared.readiness_overflowed,
        timeout,
    )
    .map_err(|error| format!("{} readiness probe failed: {error}", backend.label()))?;
    Ok(prepared)
}

fn build_watcher(
    core: Arc<WorkspaceCore>,
    repository: &Path,
    git_dir: &Path,
    backend: WatcherBackend,
) -> notify::Result<PreparedWatcher> {
    let repository = repository.to_path_buf();
    let git_dir = git_dir.to_path_buf();
    let armed = Arc::new(AtomicBool::new(false));
    let callback_armed = armed.clone();
    let (events, pending_events) = mpsc::sync_channel(EVENT_QUEUE_CAPACITY);
    let overflowed = Arc::new(AtomicBool::new(false));
    let (readiness_sender, readiness_events) = mpsc::sync_channel(EVENT_QUEUE_CAPACITY);
    let readiness_overflowed = Arc::new(AtomicBool::new(false));
    let worker_overflowed = overflowed.clone();
    let worker_core = core.clone();
    let worker_repository = repository.clone();
    let worker_git_dir = git_dir.clone();
    thread::spawn(move || {
        drain_watcher_events(
            &worker_core,
            &worker_repository,
            &worker_git_dir,
            pending_events,
            &worker_overflowed,
        )
    });
    let callback_readiness_overflowed = readiness_overflowed.clone();
    let callback_repository = repository.clone();
    let callback_git_dir = git_dir.clone();
    let callback = move |event: TrackerEvent| {
        trace_tracker_event(&callback_repository, &callback_git_dir, backend, &event);
        enqueue_watcher_event(
            &events,
            &overflowed,
            &readiness_sender,
            &callback_readiness_overflowed,
            &callback_armed,
            event,
        )
    };
    let watcher = match backend {
        WatcherBackend::Recommended => {
            LiveWatcher::Recommended(notify::recommended_watcher(callback)?)
        }
        WatcherBackend::Poll => LiveWatcher::Poll(PollWatcher::new(
            callback,
            Config::default().with_poll_interval(POLL_INTERVAL),
        )?),
    };
    Ok(PreparedWatcher {
        watcher,
        backend,
        armed,
        readiness_events,
        readiness_overflowed,
    })
}

fn enqueue_watcher_event(
    events: &SyncSender<TrackerEvent>,
    overflowed: &AtomicBool,
    readiness_events: &SyncSender<TrackerEvent>,
    readiness_overflowed: &AtomicBool,
    armed: &AtomicBool,
    event: TrackerEvent,
) {
    if event.as_ref().is_ok_and(|event| event.kind.is_access()) {
        return;
    }
    if !armed.load(Ordering::Acquire) {
        if matches!(
            readiness_events.try_send(event),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        ) {
            readiness_overflowed.store(true, Ordering::Release);
        }
        return;
    }
    if matches!(
        events.try_send(event),
        Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
    ) {
        overflowed.store(true, Ordering::Release);
    }
}

fn tracker_trace_enabled() -> bool {
    std::env::var("GREPPY_TRACKER_TRACE")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

fn trace_tracker_event(
    repository: &Path,
    git_dir: &Path,
    backend: WatcherBackend,
    event: &TrackerEvent,
) {
    if !tracker_trace_enabled() {
        return;
    }
    let (kind, need_rescan, paths, error) = match event {
        Ok(event) => (
            Some(format!("{:?}", event.kind)),
            Some(event.need_rescan()),
            event
                .paths
                .iter()
                .map(|path| match relative_utf8(repository, git_dir, path) {
                    Ok(relative) => serde_json::json!({
                        "raw": path,
                        "relative_utf8": relative,
                    }),
                    Err(error) => serde_json::json!({
                        "raw": path,
                        "relative_utf8_error": error,
                    }),
                })
                .collect::<Vec<_>>(),
            None,
        ),
        Err(error) => (None, None, Vec::new(), Some(error.to_string())),
    };
    trace_tracker_json(serde_json::json!({
        "phase": "tracker-notify-event",
        "backend": backend.label(),
        "repository": repository,
        "git_dir": git_dir,
        "kind": kind,
        "need_rescan": need_rescan,
        "paths": paths,
        "error": error,
        "timestamp_unix_ms": now_ms(),
    }));
}

fn trace_tracker_state(
    repository: &Path,
    phase: &str,
    backend: WatcherBackend,
    detail: Option<&str>,
) {
    if !tracker_trace_enabled() {
        return;
    }
    trace_tracker_json(serde_json::json!({
        "phase": phase,
        "backend": backend.label(),
        "repository": repository,
        "detail": detail,
        "timestamp_unix_ms": now_ms(),
    }));
}

fn trace_tracker_json(event: serde_json::Value) {
    let encoded = event.to_string();
    eprintln!("greppy tracker trace: {encoded}");
    let Some(root) = std::env::var_os("GREPPY_WORKSPACE_PHASE_TRACE_DIR") else {
        return;
    };
    let root = PathBuf::from(root);
    if !root.is_absolute() || std::fs::create_dir_all(&root).is_err() {
        return;
    }
    if let Ok(mut output) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(format!("tracker-{}.jsonl", std::process::id())))
    {
        let _ = writeln!(output, "{encoded}");
    }
}

fn wait_for_watcher_probe(
    git_dir: &Path,
    events: &Receiver<TrackerEvent>,
    overflowed: &AtomicBool,
    timeout: Duration,
) -> io::Result<()> {
    let name = format!("greppy-tracker-ready-{}-{}", std::process::id(), now_ms());
    let probe = git_dir.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)?;
    if let Err(error) = file
        .write_all(b"greppy.repository-tracker-ready.v1\n")
        .and_then(|()| file.sync_all())
    {
        drop(file);
        let _ = std::fs::remove_file(&probe);
        return Err(error);
    }
    drop(file);

    let deadline = std::time::Instant::now() + timeout;
    let result = loop {
        if overflowed.swap(false, Ordering::AcqRel) {
            break Err(io::Error::other("readiness event queue overflowed"));
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            break Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "no event for {} within {} seconds",
                    probe.display(),
                    timeout.as_secs()
                ),
            ));
        }
        match events.recv_timeout(deadline.saturating_duration_since(now)) {
            Ok(Ok(event)) if event.paths.iter().any(|path| path == &probe) => break Ok(()),
            Ok(Ok(_)) => {}
            Ok(Err(error)) => {
                break Err(io::Error::other(format!(
                    "watcher backend error before activation: {error}"
                )))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                break Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "no event for {} within {} seconds",
                        probe.display(),
                        timeout.as_secs()
                    ),
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                break Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "readiness event channel disconnected",
                ))
            }
        }
    };
    let cleanup = std::fs::remove_file(&probe);
    result?;
    cleanup
}

fn wait_for_active_watcher_probe(
    core: &WorkspaceCore,
    repository: &Path,
    git_dir: &Path,
    epoch: u64,
    after_generation: u64,
    timeout: Duration,
) -> io::Result<()> {
    let name = format!(
        "greppy-tracker-fence-health-{}-{}",
        std::process::id(),
        now_ms()
    );
    let probe = git_dir.join(&name);
    let expected = format!(".git/{name}");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)?;
    if let Err(error) = file
        .write_all(b"greppy.repository-tracker-active.v1\n")
        .and_then(|()| file.sync_all())
    {
        drop(file);
        let _ = std::fs::remove_file(&probe);
        return Err(error);
    }
    drop(file);

    let deadline = std::time::Instant::now() + timeout;
    let result = loop {
        let status = core
            .repository_tracker_status(repository)
            .map_err(io::Error::other)?
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "tracker disappeared"))?;
        if status.state != crate::RepositoryTrackerState::Active || status.epoch != epoch {
            break Err(io::Error::other(format!(
                "tracker lost continuity: state={:?}, epoch={}",
                status.state, status.epoch
            )));
        }
        if status.generation > after_generation {
            let changes = core
                .repository_changes_since(repository, epoch, after_generation)
                .map_err(io::Error::other)?;
            if changes.paths.iter().any(|path| path == &expected) {
                break Ok(());
            }
        }
        if std::time::Instant::now() >= deadline {
            break Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "no journaled event for {expected} within {} seconds",
                    timeout.as_secs()
                ),
            ));
        }
        thread::sleep(Duration::from_millis(10));
    };
    let cleanup = std::fs::remove_file(&probe);
    result?;
    cleanup
}

fn drain_watcher_events(
    core: &WorkspaceCore,
    repository: &Path,
    git_dir: &Path,
    events: Receiver<TrackerEvent>,
    overflowed: &AtomicBool,
) {
    loop {
        if overflowed.swap(false, Ordering::AcqRel) {
            mark_tracker_gap(core, repository, "watcher event queue overflowed");
            return;
        }
        match events.recv_timeout(Duration::from_millis(50)) {
            Ok(event) => handle_armed_event(core, repository, git_dir, event),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn handle_armed_event(
    core: &WorkspaceCore,
    repository: &Path,
    git_dir: &Path,
    event: notify::Result<notify::Event>,
) {
    match event {
        Ok(event) if event.kind.is_access() => {
            // notify defines every Access variant as non-mutating. Linux emits
            // noisy open/close-read events for Git's tree scans, including an
            // event naming the watched repository root itself. These are not
            // repository changes and must not advance or invalidate the
            // generation-bound mutation journal.
        }
        Ok(event) => {
            let emitted_paths = !event.paths.is_empty();
            let paths = event
                .paths
                .iter()
                .map(|path| relative_utf8(repository, git_dir, path))
                .collect::<Result<Vec<_>, _>>();
            match paths {
                Ok(mut paths) => {
                    // Creating or removing a child updates its parent
                    // directory. macOS Kqueue can report that parent as
                    // Modify *or Remove* even though the directory still
                    // exists, in addition to the precise child event. Ignore
                    // only those proven-existing parent paths. A missing root
                    // or a rescan request is retained and forces a fail-closed
                    // full capture.
                    if !event.need_rescan() {
                        paths.retain(|path| match path.as_str() {
                            "." => !repository.is_dir(),
                            ".git" => !git_dir.is_dir(),
                            _ => true,
                        });
                    }
                    let fence_paths = paths
                        .iter()
                        .filter(|path| is_tracker_fence_path(path))
                        .cloned()
                        .collect::<Vec<_>>();
                    paths.retain(|path| !is_tracker_internal_path(path));
                    if paths.is_empty() {
                        if !emitted_paths {
                            mark_tracker_gap(
                                core,
                                repository,
                                &format!(
                                    "watcher emitted an event without paths: kind={:?}, need_rescan={}",
                                    event.kind,
                                    event.need_rescan()
                                ),
                            );
                            return;
                        }
                    } else if let Err(error) =
                        core.record_repository_changes(repository, &paths, now_ms())
                    {
                        mark_tracker_gap(
                            core,
                            repository,
                            &format!("cannot record watcher event: {error}"),
                        );
                        return;
                    }
                    // A fence is an ordering acknowledgement, not a repository
                    // mutation. Publish it only after every ordinary path in
                    // the same callback has advanced the mutation journal.
                    if !fence_paths.is_empty() {
                        if let Err(error) =
                            core.record_repository_fences(repository, &fence_paths, now_ms())
                        {
                            mark_tracker_gap(
                                core,
                                repository,
                                &format!("cannot record tracker fence: {error}"),
                            );
                        }
                    }
                }
                Err(detail) => mark_tracker_gap(core, repository, &detail),
            }
        }
        Err(error) => {
            mark_tracker_gap(core, repository, &format!("watcher backend error: {error}"));
        }
    }
}

fn is_tracker_internal_path(path: &str) -> bool {
    path.starts_with(".git/greppy-tracker-ready-") || is_tracker_fence_path(path)
}

fn is_tracker_fence_path(path: &str) -> bool {
    path.starts_with(".git/greppy-tracker-fence-")
}

fn mark_tracker_gap(core: &WorkspaceCore, repository: &Path, detail: &str) {
    // Backend failure during installation must prevent activation. Once
    // armed, any unrepresentable event breaks continuity. The tracker store
    // preserves the first gap reason.
    let _ = core.mark_repository_tracker_gap(repository, detail, now_ms());
}

fn relative_utf8(repository: &Path, git_dir: &Path, path: &Path) -> Result<String, String> {
    let (prefix, relative) = if let Ok(relative) = path.strip_prefix(git_dir) {
        (".git/", relative)
    } else if let Ok(relative) = path.strip_prefix(repository) {
        ("", relative)
    } else {
        return Err(format!(
            "watcher path escaped repository roots: {}",
            path.display()
        ));
    };
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => parts.push(
                part.to_str()
                    .ok_or_else(|| format!("watcher path is not UTF-8: {}", path.display()))?,
            ),
            _ => {
                return Err(format!(
                    "watcher path contains an invalid component: {}",
                    path.display()
                ))
            }
        }
    }
    if parts.is_empty() && prefix == ".git/" {
        return Ok(".git".into());
    }
    if parts.is_empty() {
        return Ok(".".into());
    }
    Ok(format!("{prefix}{}", parts.join("/")))
}

fn repository_git_dir(repository: &Path) -> io::Result<PathBuf> {
    let dot_git = repository.join(".git");
    if dot_git.is_dir() {
        return std::fs::canonicalize(dot_git);
    }
    let marker = std::fs::read_to_string(&dot_git)?;
    let value = marker
        .trim()
        .strip_prefix("gitdir:")
        .ok_or_else(|| io::Error::other(".git file has no gitdir marker"))?
        .trim();
    let path = PathBuf::from(value);
    std::fs::canonicalize(if path.is_absolute() {
        path
    } else {
        repository.join(path)
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RepositoryTrackerState;
    use std::time::Instant;

    #[test]
    fn spawned_service_activates_a_late_repository_request() {
        let temp = tempfile::tempdir().unwrap();
        let repository = temp.path().join("repo");
        std::fs::create_dir(&repository).unwrap();
        std::fs::create_dir(repository.join(".git")).unwrap();
        let _tracker = spawn_repository_tracker(temp.path().to_path_buf()).unwrap();
        let core = WorkspaceCore::open(temp.path().join("core")).unwrap();
        core.request_repository_tracker(&repository).unwrap();

        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let status = core.repository_tracker_status(&repository).unwrap();
            if status
                .as_ref()
                .is_some_and(|status| status.state == RepositoryTrackerState::Active)
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "repository tracker did not activate: {status:?}"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn active_state_never_precedes_fence_event_routing() {
        let temp = tempfile::tempdir().unwrap();
        let repository_path = temp.path().join("repo");
        std::fs::create_dir(&repository_path).unwrap();
        let repository = std::fs::canonicalize(repository_path).unwrap();
        std::fs::create_dir(repository.join(".git")).unwrap();
        let _tracker = spawn_repository_tracker(temp.path().to_path_buf()).unwrap();
        let core = WorkspaceCore::open(temp.path().join("core")).unwrap();
        core.request_repository_tracker(&repository).unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        let active = loop {
            let status = core.repository_tracker_status(&repository).unwrap();
            if let Some(status) =
                status.filter(|status| status.state == RepositoryTrackerState::Active)
            {
                break status;
            }
            assert!(Instant::now() < deadline, "tracker did not activate");
            thread::sleep(Duration::from_millis(5));
        };

        let name = format!("greppy-tracker-fence-test-{}", now_ms());
        std::fs::write(repository.join(".git").join(&name), b"fence").unwrap();
        let expected = format!(".git/{name}");
        loop {
            if core
                .consume_repository_fence(&repository, active.epoch, &expected)
                .unwrap()
                == Some(active.generation)
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "first post-Active fence was not acknowledged"
            );
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            core.repository_tracker_status(&repository)
                .unwrap()
                .unwrap()
                .generation,
            active.generation
        );
    }

    #[test]
    fn a_new_service_reclaims_a_dead_persisted_owner_and_routes_its_fence() {
        let temp = tempfile::tempdir().unwrap();
        let data_root = temp.path().join("data");
        let repository_path = temp.path().join("repo");
        std::fs::create_dir(&repository_path).unwrap();
        let repository = std::fs::canonicalize(repository_path).unwrap();
        std::fs::create_dir(repository.join(".git")).unwrap();
        let core = WorkspaceCore::open(data_root.join("core")).unwrap();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("repository_tracker_service::tests::repository_tracker_owner_child")
            .arg("--nocapture")
            .env("GREPPY_TRACKER_OWNER_CHILD_DATA", &data_root)
            .env("GREPPY_TRACKER_OWNER_CHILD_REPOSITORY", &repository)
            .status()
            .unwrap();
        assert!(child.success());
        let stale = core
            .repository_tracker_status(&repository)
            .unwrap()
            .unwrap();
        assert_eq!(stale.state, RepositoryTrackerState::Active);
        assert!(!stale.is_live_at(now_ms()));

        let _tracker = spawn_repository_tracker(data_root).unwrap();
        core.request_repository_tracker(&repository).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let active = loop {
            let status = core.repository_tracker_status(&repository).unwrap();
            if status.as_ref().is_some_and(|status| {
                status.state == RepositoryTrackerState::Active
                    && status.epoch > stale.epoch
                    && status.is_live_at(now_ms())
            }) {
                break status.unwrap();
            }
            assert!(
                Instant::now() < deadline,
                "replacement tracker did not become live: {status:?}"
            );
            thread::sleep(Duration::from_millis(5));
        };

        let name = format!("greppy-tracker-fence-second-process-{}", now_ms());
        let fence = repository.join(".git").join(&name);
        std::fs::write(&fence, b"fence").unwrap();
        let expected = format!(".git/{name}");
        loop {
            if core
                .consume_repository_fence(&repository, active.epoch, &expected)
                .unwrap()
                == Some(active.generation)
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "replacement tracker did not acknowledge its fence"
            );
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            core.repository_tracker_status(&repository)
                .unwrap()
                .unwrap()
                .generation,
            active.generation
        );
        std::fs::remove_file(fence).unwrap();
    }

    #[test]
    fn repository_tracker_owner_child() {
        let Some(data_root) = std::env::var_os("GREPPY_TRACKER_OWNER_CHILD_DATA") else {
            return;
        };
        let repository = std::env::var_os("GREPPY_TRACKER_OWNER_CHILD_REPOSITORY")
            .map(PathBuf::from)
            .unwrap();
        let core = WorkspaceCore::open(PathBuf::from(data_root).join("core")).unwrap();
        core.request_repository_tracker(&repository).unwrap();
        let active = core
            .activate_repository_tracker(&repository, now_ms())
            .unwrap();
        assert_eq!(active.owner_pid, std::process::id());
        assert!(active.is_live_at(now_ms()));
    }

    #[test]
    fn path_normalization_is_relative_and_rejects_escape() {
        let root = Path::new("/tmp/repository");
        let git_dir = Path::new("/tmp/repository/.git");
        assert_eq!(
            relative_utf8(root, git_dir, Path::new("/tmp/repository/src/lib.rs")).unwrap(),
            "src/lib.rs"
        );
        assert_eq!(relative_utf8(root, git_dir, git_dir).unwrap(), ".git");
        assert_eq!(relative_utf8(root, git_dir, root).unwrap(), ".");
        assert!(relative_utf8(root, git_dir, Path::new("/tmp/other/file")).is_err());
    }

    #[test]
    fn recommended_watcher_records_a_generation_bound_change() {
        let temp = tempfile::tempdir().unwrap();
        let repository_path = temp.path().join("repo");
        std::fs::create_dir(&repository_path).unwrap();
        let repository = std::fs::canonicalize(repository_path).unwrap();
        let git_dir_path = repository.join(".git");
        std::fs::create_dir(&git_dir_path).unwrap();
        let git_dir = std::fs::canonicalize(git_dir_path).unwrap();
        let core = Arc::new(WorkspaceCore::open(temp.path().join("core")).unwrap());
        core.request_repository_tracker(&repository).unwrap();
        let mut prepared = build_watcher(
            core.clone(),
            &repository,
            &git_dir,
            WatcherBackend::Recommended,
        )
        .unwrap();
        prepared
            .watcher
            .watch(&repository, RecursiveMode::Recursive)
            .unwrap();
        std::fs::write(
            repository.join("before-active.txt"),
            b"covered by full capture",
        )
        .unwrap();
        thread::sleep(Duration::from_millis(200));
        assert_eq!(
            core.repository_tracker_status(&repository)
                .unwrap()
                .unwrap()
                .state,
            RepositoryTrackerState::Requested
        );
        wait_for_watcher_probe(
            &git_dir,
            &prepared.readiness_events,
            &prepared.readiness_overflowed,
            RECOMMENDED_READY_TIMEOUT,
        )
        .unwrap();
        prepared.armed.store(true, Ordering::Release);
        let active = core
            .activate_repository_tracker(&repository, now_ms())
            .unwrap();
        std::fs::write(repository.join("changed.txt"), b"changed").unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let status = core
                .repository_tracker_status(&repository)
                .unwrap()
                .unwrap();
            if status.generation > 0 {
                let changes = core
                    .repository_changes_since(&repository, active.epoch, 0)
                    .unwrap();
                if changes.paths.iter().any(|path| path == "changed.txt") {
                    assert!(!changes
                        .paths
                        .iter()
                        .any(|path| path.starts_with(".git/greppy-tracker-ready-")));
                    break;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "watcher event timed out"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn poll_fallback_probes_and_records_a_generation_bound_change() {
        let temp = tempfile::tempdir().unwrap();
        let repository_path = temp.path().join("repo");
        std::fs::create_dir(&repository_path).unwrap();
        let repository = std::fs::canonicalize(repository_path).unwrap();
        let git_dir_path = repository.join(".git");
        std::fs::create_dir(&git_dir_path).unwrap();
        let git_dir = std::fs::canonicalize(git_dir_path).unwrap();
        let core = Arc::new(WorkspaceCore::open(temp.path().join("core")).unwrap());
        core.request_repository_tracker(&repository).unwrap();

        let prepared = install_and_probe_watcher(
            core.clone(),
            &repository,
            &git_dir,
            WatcherBackend::Poll,
            POLL_READY_TIMEOUT,
        )
        .unwrap();
        assert_eq!(prepared.backend, WatcherBackend::Poll);
        prepared.armed.store(true, Ordering::Release);
        let active = core
            .activate_repository_tracker(&repository, now_ms())
            .unwrap();
        std::fs::write(repository.join("changed-by-poll.txt"), b"changed").unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let changes = core
                .repository_changes_since(&repository, active.epoch, 0)
                .unwrap();
            if changes
                .paths
                .iter()
                .any(|path| path == "changed-by-poll.txt")
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "PollWatcher event timed out: {changes:?}"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn preactivation_events_are_ignored_but_backend_errors_block_activation() {
        let temp = tempfile::tempdir().unwrap();
        let repository = temp.path().join("repo");
        std::fs::create_dir(&repository).unwrap();
        let core = WorkspaceCore::open(temp.path().join("core")).unwrap();
        core.request_repository_tracker(&repository).unwrap();

        core.record_repository_changes(&repository, &["before-active.txt".into()], now_ms())
            .unwrap();
        assert_eq!(
            core.repository_tracker_status(&repository)
                .unwrap()
                .unwrap()
                .state,
            RepositoryTrackerState::Requested
        );

        mark_tracker_gap(&core, &repository, "watcher backend stopped");
        mark_tracker_gap(&core, &repository, "later callback noise");
        let status = core
            .repository_tracker_status(&repository)
            .unwrap()
            .unwrap();
        assert_eq!(status.state, RepositoryTrackerState::Gap);
        assert_eq!(status.detail.as_deref(), Some("watcher backend stopped"));
        assert!(core
            .activate_repository_tracker(&repository, now_ms())
            .is_err());
    }

    #[test]
    fn access_and_parent_modify_events_do_not_invalidate_the_tracker() {
        use notify::event::{AccessKind, AccessMode, ModifyKind, RemoveKind};
        use notify::{Event, EventKind};

        let temp = tempfile::tempdir().unwrap();
        let repository = temp.path().join("repo");
        std::fs::create_dir(&repository).unwrap();
        let git_dir = repository.join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        let core = WorkspaceCore::open(temp.path().join("core")).unwrap();
        core.request_repository_tracker(&repository).unwrap();
        let active = core
            .activate_repository_tracker(&repository, now_ms())
            .unwrap();

        let read = Event::new(EventKind::Access(AccessKind::Close(AccessMode::Read)))
            .add_path(repository.clone());
        handle_armed_event(&core, &repository, &git_dir, Ok(read));
        let after_read = core
            .repository_tracker_status(&repository)
            .unwrap()
            .unwrap();
        assert_eq!(after_read.state, RepositoryTrackerState::Active);
        assert_eq!(after_read.epoch, active.epoch);
        assert_eq!(after_read.generation, active.generation);

        let mutation = Event::new(EventKind::Modify(ModifyKind::Any)).add_path(repository.clone());
        handle_armed_event(&core, &repository, &git_dir, Ok(mutation));
        let after_mutation = core
            .repository_tracker_status(&repository)
            .unwrap()
            .unwrap();
        assert_eq!(after_mutation.state, RepositoryTrackerState::Active);
        assert_eq!(after_mutation.generation, active.generation);

        let parent_remove_noise =
            Event::new(EventKind::Remove(RemoveKind::Any)).add_path(repository.clone());
        handle_armed_event(&core, &repository, &git_dir, Ok(parent_remove_noise));
        let after_parent_noise = core
            .repository_tracker_status(&repository)
            .unwrap()
            .unwrap();
        assert_eq!(after_parent_noise.state, RepositoryTrackerState::Active);
        assert_eq!(after_parent_noise.generation, active.generation);

        std::fs::remove_dir_all(&repository).unwrap();
        let actual_removal =
            Event::new(EventKind::Remove(RemoveKind::Any)).add_path(repository.clone());
        handle_armed_event(&core, &repository, &git_dir, Ok(actual_removal));
        let changes = core
            .repository_changes_since(&repository, active.epoch, active.generation)
            .unwrap();
        assert_eq!(changes.paths, vec!["."]);
    }

    #[test]
    fn delayed_fence_events_acknowledge_without_invalidating_snapshot_generation() {
        use notify::event::{CreateKind, ModifyKind, RemoveKind};
        use notify::{Event, EventKind};

        let temp = tempfile::tempdir().unwrap();
        let repository = temp.path().join("repo");
        std::fs::create_dir(&repository).unwrap();
        let git_dir = repository.join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        let core = WorkspaceCore::open(temp.path().join("core")).unwrap();
        core.request_repository_tracker(&repository).unwrap();
        let active = core
            .activate_repository_tracker(&repository, now_ms())
            .unwrap();
        let fence_name = "greppy-tracker-fence-123-456-0";
        let fence_path = git_dir.join(fence_name);
        let virtual_path = format!(".git/{fence_name}");

        for event in [
            Event::new(EventKind::Create(CreateKind::File)).add_path(fence_path.clone()),
            Event::new(EventKind::Modify(ModifyKind::Any)).add_path(fence_path.clone()),
            Event::new(EventKind::Remove(RemoveKind::File)).add_path(fence_path),
        ] {
            handle_armed_event(&core, &repository, &git_dir, Ok(event));
            assert_eq!(
                core.repository_tracker_status(&repository)
                    .unwrap()
                    .unwrap()
                    .generation,
                active.generation
            );
        }

        assert_eq!(
            core.consume_repository_fence(&repository, active.epoch, &virtual_path)
                .unwrap(),
            Some(active.generation)
        );
        let changes = core
            .repository_changes_since(&repository, active.epoch, active.generation)
            .unwrap();
        assert!(changes.paths.is_empty());
        assert_eq!(changes.generation, active.generation);

        let combined_name = "greppy-tracker-fence-123-789-1";
        let combined_virtual_path = format!(".git/{combined_name}");
        let combined = Event::new(EventKind::Modify(ModifyKind::Any))
            .add_path(repository.join("src/lib.rs"))
            .add_path(git_dir.join(combined_name));
        handle_armed_event(&core, &repository, &git_dir, Ok(combined));
        assert_eq!(
            core.consume_repository_fence(&repository, active.epoch, &combined_virtual_path)
                .unwrap(),
            Some(active.generation + 1)
        );
        let changes = core
            .repository_changes_since(&repository, active.epoch, active.generation)
            .unwrap();
        assert_eq!(changes.generation, active.generation + 1);
        assert_eq!(changes.paths, ["src/lib.rs"]);
    }

    #[test]
    fn watcher_callback_filters_access_and_never_blocks_on_backpressure() {
        use notify::event::{AccessKind, AccessMode, ModifyKind};
        use notify::{Event, EventKind};

        let (events, _pending) = mpsc::sync_channel(1);
        let (readiness_events, _pending_readiness) = mpsc::sync_channel(1);
        let overflowed = AtomicBool::new(false);
        let readiness_overflowed = AtomicBool::new(false);
        let armed = AtomicBool::new(true);
        enqueue_watcher_event(
            &events,
            &overflowed,
            &readiness_events,
            &readiness_overflowed,
            &armed,
            Ok(Event::new(EventKind::Access(AccessKind::Close(
                AccessMode::Read,
            )))),
        );
        enqueue_watcher_event(
            &events,
            &overflowed,
            &readiness_events,
            &readiness_overflowed,
            &armed,
            Ok(Event::new(EventKind::Modify(ModifyKind::Any))),
        );
        enqueue_watcher_event(
            &events,
            &overflowed,
            &readiness_events,
            &readiness_overflowed,
            &armed,
            Ok(Event::new(EventKind::Modify(ModifyKind::Any))),
        );
        assert!(overflowed.load(Ordering::Acquire));
    }
}
