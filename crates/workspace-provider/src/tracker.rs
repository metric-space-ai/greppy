use greppy_workspace_core::WorkspaceCore;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(crate) fn spawn(data_root: PathBuf) -> io::Result<thread::JoinHandle<()>> {
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
                Ok((mut watcher, armed)) => {
                    if let Err(error) = watcher.watch(&repository, RecursiveMode::Recursive) {
                        let _ = core.mark_repository_tracker_gap(
                            &repository,
                            &format!("cannot watch repository: {error}"),
                            now_ms(),
                        );
                        continue;
                    }
                    if !git_dir.starts_with(&repository) {
                        if let Err(error) = watcher.watch(&git_dir, RecursiveMode::Recursive) {
                            let _ = core.mark_repository_tracker_gap(
                                &repository,
                                &format!("cannot watch linked Git directory: {error}"),
                                now_ms(),
                            );
                            continue;
                        }
                    }
                    if core
                        .activate_repository_tracker(&repository, now_ms())
                        .is_ok()
                    {
                        armed.store(true, Ordering::Release);
                        watchers.insert(repository, watcher);
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
) -> notify::Result<(RecommendedWatcher, Arc<AtomicBool>)> {
    let repository = repository.to_path_buf();
    let git_dir = git_dir.to_path_buf();
    let armed = Arc::new(AtomicBool::new(false));
    let callback_armed = armed.clone();
    let watcher =
        notify::recommended_watcher(move |event: notify::Result<notify::Event>| match event {
            Ok(_) if !callback_armed.load(Ordering::Acquire) => {
                // Successful events emitted while watch roots are being installed
                // are covered by the first full double-capture after activation.
            }
            Ok(event) => {
                let paths = event
                    .paths
                    .iter()
                    .map(|path| relative_utf8(&repository, &git_dir, path))
                    .collect::<Result<Vec<_>, _>>();
                match paths {
                    Ok(paths) if !paths.is_empty() => {
                        if let Err(error) =
                            core.record_repository_changes(&repository, &paths, now_ms())
                        {
                            mark_tracker_gap(
                                &core,
                                &repository,
                                &format!("cannot record watcher event: {error}"),
                            );
                        }
                    }
                    Ok(_) => {
                        mark_tracker_gap(
                            &core,
                            &repository,
                            "watcher emitted an event without paths",
                        );
                    }
                    Err(detail) => {
                        mark_tracker_gap(&core, &repository, &detail);
                    }
                }
            }
            Err(error) => {
                mark_tracker_gap(
                    &core,
                    &repository,
                    &format!("watcher backend error: {error}"),
                );
            }
        })?;
    Ok((watcher, armed))
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
    use greppy_workspace_core::RepositoryTrackerState;

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
        let (mut watcher, armed) = build_watcher(core.clone(), &repository, &git_dir).unwrap();
        watcher
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
        let active = core
            .activate_repository_tracker(&repository, now_ms())
            .unwrap();
        armed.store(true, Ordering::Release);
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
}
