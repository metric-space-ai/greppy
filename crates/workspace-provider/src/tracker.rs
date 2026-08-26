use greppy_workspace_core::WorkspaceCore;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
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
            match build_watcher(core.clone(), &repository) {
                Ok(mut watcher) => {
                    if let Err(error) = watcher.watch(&repository, RecursiveMode::Recursive) {
                        let _ = core.mark_repository_tracker_gap(
                            &repository,
                            &format!("cannot watch repository: {error}"),
                            now_ms(),
                        );
                        continue;
                    }
                    if core
                        .activate_repository_tracker(&repository, now_ms())
                        .is_ok()
                    {
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
) -> notify::Result<RecommendedWatcher> {
    let repository = repository.to_path_buf();
    notify::recommended_watcher(move |event: notify::Result<notify::Event>| match event {
        Ok(event) => {
            let paths = event
                .paths
                .iter()
                .map(|path| relative_utf8(&repository, path))
                .collect::<Result<Vec<_>, _>>();
            match paths {
                Ok(paths) if !paths.is_empty() => {
                    if let Err(error) =
                        core.record_repository_changes(&repository, &paths, now_ms())
                    {
                        let _ = core.mark_repository_tracker_gap(
                            &repository,
                            &format!("cannot record watcher event: {error}"),
                            now_ms(),
                        );
                    }
                }
                Ok(_) => {
                    let _ = core.mark_repository_tracker_gap(
                        &repository,
                        "watcher emitted an event without paths",
                        now_ms(),
                    );
                }
                Err(detail) => {
                    let _ = core.mark_repository_tracker_gap(&repository, &detail, now_ms());
                }
            }
        }
        Err(error) => {
            let _ = core.mark_repository_tracker_gap(
                &repository,
                &format!("watcher backend error: {error}"),
                now_ms(),
            );
        }
    })
}

fn relative_utf8(repository: &Path, path: &Path) -> Result<String, String> {
    let relative = path.strip_prefix(repository).map_err(|_| {
        format!(
            "watcher path escaped repository: {} not under {}",
            path.display(),
            repository.display()
        )
    })?;
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
    Ok(parts.join("/"))
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

    #[test]
    fn path_normalization_is_relative_and_rejects_escape() {
        let root = Path::new("/tmp/repository");
        assert_eq!(
            relative_utf8(root, Path::new("/tmp/repository/src/lib.rs")).unwrap(),
            "src/lib.rs"
        );
        assert!(relative_utf8(root, Path::new("/tmp/other/file")).is_err());
        assert!(relative_utf8(root, root).is_err());
    }

    #[test]
    fn recommended_watcher_records_a_generation_bound_change() {
        let temp = tempfile::tempdir().unwrap();
        let repository_path = temp.path().join("repo");
        std::fs::create_dir(&repository_path).unwrap();
        let repository = std::fs::canonicalize(repository_path).unwrap();
        let core = Arc::new(WorkspaceCore::open(temp.path().join("core")).unwrap());
        core.request_repository_tracker(&repository).unwrap();
        let mut watcher = build_watcher(core.clone(), &repository).unwrap();
        watcher
            .watch(&repository, RecursiveMode::Recursive)
            .unwrap();
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
}
