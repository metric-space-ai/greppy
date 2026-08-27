use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use greppy_workspace_core::{
    AdapterKind, ProviderCapabilities, ProviderManifest, ProviderState, WorkspaceCore,
    PROVIDER_PROTOCOL_VERSION,
};

pub struct FakeProvider {
    pub data: PathBuf,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for FakeProvider {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Integration-test adapter for CLI child processes. It publishes the same
/// identity and heartbeat contract as a real provider and only mirrors the
/// fixture repository after WorkspaceCore creates a namespace.
pub fn spawn_fake_provider(root: &Path, repo: &Path) -> FakeProvider {
    let data = root.join("provider-data");
    let mount = root.join("provider-mount");
    std::fs::create_dir_all(mount.join("doctor")).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let manifest = ProviderManifest {
        protocol_version: PROVIDER_PROTOCOL_VERSION,
        adapter_version: "0.3.4-cli-test".into(),
        adapter_kind: fixture_adapter_kind(),
        state: ProviderState::Ready,
        instance_id: "cli-test-provider".into(),
        data_root: data.clone(),
        mount_root: mount.clone(),
        heartbeat_unix_ms: unix_milliseconds(),
        capabilities: ProviderCapabilities {
            hard_links: true,
            symbolic_links: true,
            byte_range_locks: true,
            memory_maps: true,
            atomic_rename: true,
            case_preserving: true,
        },
    };
    publish_manifest(&data, &manifest);
    publish_mount_manifest(&mount, &manifest);

    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    let data_thread = data.clone();
    let mount_thread = mount.clone();
    let repo_thread = repo.to_path_buf();
    let handle = thread::spawn(move || {
        let mut manifest = manifest;
        let mut core = None;
        let tracked_repo = std::fs::canonicalize(&repo_thread).unwrap();
        let git_dir = tracked_repo.join(".git");
        let mut tracker_active = false;
        let mut tracker_generation = 0_u64;
        let mut tracker_fences = HashSet::new();
        let mut mirrored = HashSet::new();
        let mut last_heartbeat = 0;
        while !stop_thread.load(Ordering::SeqCst) {
            let now = unix_milliseconds();
            if now.saturating_sub(last_heartbeat) >= 500 {
                manifest.heartbeat_unix_ms = now;
                publish_manifest(&data_thread, &manifest);
                publish_mount_manifest(&mount_thread, &manifest);
                last_heartbeat = now;
            }
            if core.is_none() {
                core = WorkspaceCore::open(data_thread.join("core")).ok();
            }
            if let Some(core) = &core {
                if !tracker_active {
                    core.request_repository_tracker(&tracked_repo).unwrap();
                    core.activate_repository_tracker(&tracked_repo, unix_milliseconds())
                        .unwrap();
                    tracker_active = true;
                }
                let current_fences = std::fs::read_dir(&git_dir)
                    .unwrap()
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| entry.file_name().into_string().ok())
                    .filter(|name| name.starts_with("greppy-tracker-fence-"))
                    .collect::<HashSet<_>>();
                let fence_changes = current_fences
                    .symmetric_difference(&tracker_fences)
                    .map(|name| format!(".git/{name}"))
                    .collect::<Vec<_>>();
                if !fence_changes.is_empty() {
                    tracker_generation += 1;
                    core.record_repository_changes(
                        &tracked_repo,
                        &fence_changes,
                        tracker_generation,
                    )
                    .unwrap();
                    tracker_fences = current_fences;
                }
                if let Ok(workspaces) = core.list_workspaces() {
                    let active = workspaces
                        .iter()
                        .map(|workspace| workspace.id.clone())
                        .collect::<HashSet<_>>();
                    for workspace in &workspaces {
                        let destination = mount_thread.join("workspaces").join(&workspace.id);
                        if !destination.exists() {
                            let temporary = mount_thread
                                .join("workspaces")
                                .join(format!(".{}.materializing", workspace.id));
                            let _ = std::fs::remove_dir_all(&temporary);
                            std::fs::create_dir_all(&temporary).unwrap();
                            if workspace.id.starts_with("git-") {
                                let payload = git_control_payload(&data_thread).unwrap();
                                copy_directory_tree(&payload, &temporary, false);
                            } else {
                                copy_directory_tree(&repo_thread, &temporary, true);
                            }
                            std::fs::rename(temporary, &destination).unwrap();
                            mirrored.insert(workspace.id.clone());
                        }
                    }
                    let removed = mirrored.difference(&active).cloned().collect::<Vec<_>>();
                    for workspace in removed {
                        let _ = std::fs::remove_dir_all(
                            mount_thread.join("workspaces").join(&workspace),
                        );
                        mirrored.remove(&workspace);
                    }
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
    FakeProvider {
        data,
        stop,
        handle: Some(handle),
    }
}

fn publish_manifest(data: &Path, manifest: &ProviderManifest) {
    let temporary = data.join(format!("provider.json.{}.tmp", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(manifest).unwrap()).unwrap();
    std::fs::rename(temporary, data.join("provider.json")).unwrap();
}

fn publish_mount_manifest(mount: &Path, manifest: &ProviderManifest) {
    let temporary = mount.join(format!(".greppy-provider.json.{}.tmp", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(manifest).unwrap()).unwrap();
    std::fs::rename(temporary, mount.join(".greppy-provider.json")).unwrap();
}

fn git_control_payload(data: &Path) -> Option<PathBuf> {
    std::fs::read_dir(data.join("g").join("ct3"))
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.join("COMPLETE").is_file())
        .map(|path| path.join("payload"))
}

fn copy_directory_tree(source: &Path, destination: &Path, skip_dot_git: bool) {
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if skip_dot_git && entry.file_name() == ".git" {
            continue;
        }
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
            copy_directory_tree(&entry.path(), &target, skip_dot_git);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn unix_milliseconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(target_os = "linux")]
fn fixture_adapter_kind() -> AdapterKind {
    AdapterKind::Fuse3
}

#[cfg(target_os = "macos")]
fn fixture_adapter_kind() -> AdapterKind {
    AdapterKind::FsKit
}

#[cfg(target_os = "windows")]
fn fixture_adapter_kind() -> AdapterKind {
    AdapterKind::WinFsp
}
