use crate::WorkspaceCore;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const EVENT_QUEUE_CAPACITY: usize = 4_096;
const WATCHER_READY_TIMEOUT: Duration = Duration::from_secs(5);
type TrackerEvent = notify::Result<notify::Event>;

struct PreparedWatcher {
    watcher: RecommendedWatcher,
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
    let mut watchers = HashMap::<PathBuf, RecommendedWatcher>::new();
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
            match build_watcher(core.clone(), &repository, &git_dir) {
                Ok(mut prepared) => {
                    if let Err(error) = prepared
                        .watcher
                        .watch(&repository, RecursiveMode::Recursive)
                    {
                        let _ = core.mark_repository_tracker_gap(
                            &repository,
                            &format!("cannot watch repository: {error}"),
                            now_ms(),
                        );
                        continue;
                    }
                    if !git_dir.starts_with(&repository) {
                        if let Err(error) =
                            prepared.watcher.watch(&git_dir, RecursiveMode::Recursive)
                        {
                            let _ = core.mark_repository_tracker_gap(
                                &repository,
                                &format!("cannot watch linked Git directory: {error}"),
                                now_ms(),
                            );
                            continue;
                        }
                    }
                    if let Err(error) = wait_for_watcher_probe(
                        &git_dir,
                        &prepared.readiness_events,
                        &prepared.readiness_overflowed,
                    ) {
                        let _ = core.mark_repository_tracker_gap(
                            &repository,
                            &format!("watcher readiness probe failed: {error}"),
                            now_ms(),
                        );
                        continue;
                    }
                    if core
                        .activate_repository_tracker(&repository, now_ms())
                        .is_ok()
                    {
                        prepared.armed.store(true, Ordering::Release);
                        watchers.insert(repository, prepared.watcher);
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
        thread::sleep(Duration::from_millis(100));
    }
}

fn build_watcher(
    core: Arc<WorkspaceCore>,
    repository: &Path,
    git_dir: &Path,
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
    let watcher = notify::recommended_watcher(move |event: TrackerEvent| {
        enqueue_watcher_event(
            &events,
            &overflowed,
            &readiness_sender,
            &callback_readiness_overflowed,
            &callback_armed,
            event,
        )
    })?;
    Ok(PreparedWatcher {
        watcher,
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

fn wait_for_watcher_probe(
    git_dir: &Path,
    events: &Receiver<TrackerEvent>,
    overflowed: &AtomicBool,
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

    let deadline = std::time::Instant::now() + WATCHER_READY_TIMEOUT;
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
                    WATCHER_READY_TIMEOUT.as_secs()
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
                        WATCHER_READY_TIMEOUT.as_secs()
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
            let paths = event
                .paths
                .iter()
                .map(|path| relative_utf8(repository, git_dir, path))
                .collect::<Result<Vec<_>, _>>();
            match paths {
                Ok(paths) if !paths.is_empty() => {
                    if let Err(error) = core.record_repository_changes(repository, &paths, now_ms())
                    {
                        mark_tracker_gap(
                            core,
                            repository,
                            &format!("cannot record watcher event: {error}"),
                        );
                    }
                }
                Ok(_) => mark_tracker_gap(
                    core,
                    repository,
                    &format!(
                        "watcher emitted an event without paths: kind={:?}, need_rescan={}",
                        event.kind,
                        event.need_rescan()
                    ),
                ),
                Err(detail) => mark_tracker_gap(core, repository, &detail),
            }
        }
        Err(error) => {
            mark_tracker_gap(core, repository, &format!("watcher backend error: {error}"));
        }
    }
}

fn mark_tracker_gap(core: &WorkspaceCore, repository: &Path, detail: &str) {
    // Backend failure during installation must prevent activation. Once
    // armed, any unrepresentable event breaks continuity. The tracker store
    // preserves the first gap reason.
    let _ = core.mark_repository_tracker_gap(repository, detail, now_ms());
}

fn relative_utf8(repository: &Path, git_dir: &Path, path: &Path) -> Result<String, String> {
    let (prefix, relative) = if let Ok(relative) = path.strip_prefix(repository) {
        ("", relative)
    } else if let Ok(relative) = path.strip_prefix(git_dir) {
        (".git/", relative)
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
    if parts.is_empty() {
        return Err("watcher reported the repository root without a child path".into());
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
    fn path_normalization_is_relative_and_rejects_escape() {
        let root = Path::new("/tmp/repository");
        let git_dir = Path::new("/tmp/repository/.git");
        assert_eq!(
            relative_utf8(root, git_dir, Path::new("/tmp/repository/src/lib.rs")).unwrap(),
            "src/lib.rs"
        );
        assert!(relative_utf8(root, git_dir, Path::new("/tmp/other/file")).is_err());
        assert!(relative_utf8(root, git_dir, root).is_err());
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
        let mut prepared = build_watcher(core.clone(), &repository, &git_dir).unwrap();
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
        )
        .unwrap();
        let active = core
            .activate_repository_tracker(&repository, now_ms())
            .unwrap();
        prepared.armed.store(true, Ordering::Release);
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
                assert_eq!(changes.paths, ["changed.txt"]);
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "watcher event timed out"
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
    fn access_events_do_not_mutate_or_invalidate_the_tracker() {
        use notify::event::{AccessKind, AccessMode, ModifyKind};
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
        assert_eq!(after_mutation.state, RepositoryTrackerState::Gap);
        assert_eq!(
            after_mutation.detail.as_deref(),
            Some("watcher reported the repository root without a child path")
        );
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
