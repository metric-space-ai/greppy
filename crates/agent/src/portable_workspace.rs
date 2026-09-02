//! Portable, provider-mounted CoW workspace lifecycle.
//!
//! No native checkout or filesystem snapshot fallback exists here. Creation
//! verifies the installed adapter and its mounted identity before capturing the
//! repository and before any model request can be made.

use greppy_workspace_core::{
    capture_overlay_directory, capture_repository, capture_repository_incremental,
    capture_repository_with_observer, BaselineEntry, BaselineSnapshot, ChunkStore, EntryKind,
    ProposalRecord, ProviderInstallation, RepositoryTrackerState, WorkspaceCore, WorkspaceHandle,
    WorkspacePairLease,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct AgentWorkspace {
    repo_root: PathBuf,
    worktree: PathBuf,
    private_git_dir: PathBuf,
    private_index: PathBuf,
    git_handle: WorkspaceHandle,
    run_id: String,
    base_commit: String,
    baseline_hash: String,
    baseline_tree: String,
    baseline_view_commit: String,
    provider_instance: String,
    data_root: PathBuf,
    core: WorkspaceCore,
    handle: WorkspaceHandle,
    pair_lease: WorkspacePairLease,
}

impl fmt::Debug for AgentWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentWorkspace")
            .field("repo_root", &self.repo_root)
            .field("worktree", &self.worktree)
            .field("private_git_dir", &self.private_git_dir)
            .field("private_index", &self.private_index)
            .field("run_id", &self.run_id)
            .field("base_commit", &self.base_commit)
            .field("baseline_hash", &self.baseline_hash)
            .field("provider_instance", &self.provider_instance)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    Clean,
    Proposal {
        commit: String,
        ref_name: String,
        patch: String,
        stat: String,
    },
}

#[derive(Debug)]
pub enum WorkspaceError {
    NotGitRepo {
        path: PathBuf,
        detail: String,
    },
    GitFailed {
        command: String,
        stderr: String,
        status: Option<i32>,
    },
    DirtyTarget {
        ref_name: String,
        detail: String,
    },
    Conflict {
        ref_name: String,
        detail: String,
    },
    Tampered {
        path: PathBuf,
        detail: String,
    },
    Unsupported(String),
    AdapterUnavailable(String),
    Io(io::Error),
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotGitRepo { path, detail } => {
                write!(f, "not a Git repository ({}): {detail}", path.display())
            }
            Self::GitFailed {
                command,
                stderr,
                status,
            } => {
                write!(f, "Git command failed: {command}")?;
                if let Some(status) = status {
                    write!(f, " (exit {status})")?;
                }
                if !stderr.trim().is_empty() {
                    write!(f, ": {}", stderr.trim())?;
                }
                Ok(())
            }
            Self::DirtyTarget { ref_name, detail } => write!(
                f,
                "target no longer matches the proposal baseline; {ref_name} was not applied: {detail}"
            ),
            Self::Conflict { ref_name, detail } => {
                write!(f, "proposal {ref_name} could not be applied: {detail}")
            }
            Self::Tampered { path, detail } => write!(
                f,
                "portable workspace identity is invalid ({}): {detail}",
                path.display()
            ),
            Self::Unsupported(reason) => {
                write!(f, "repository is not supported by greppy -p: {reason}")
            }
            Self::AdapterUnavailable(reason) => {
                write!(f, "portable workspace adapter is unavailable: {reason}")
            }
            Self::Io(error) => write!(f, "workspace I/O error: {error}"),
        }
    }
}

impl std::error::Error for WorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for WorkspaceError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<greppy_workspace_core::Error> for WorkspaceError {
    fn from(error: greppy_workspace_core::Error) -> Self {
        match error {
            greppy_workspace_core::Error::NotGitRepository { path, detail } => {
                Self::NotGitRepo { path, detail }
            }
            greppy_workspace_core::Error::UnsupportedRepository(reason) => {
                Self::Unsupported(reason)
            }
            greppy_workspace_core::Error::AdapterUnavailable(reason)
            | greppy_workspace_core::Error::AdapterUnhealthy(reason) => {
                Self::AdapterUnavailable(reason)
            }
            other => Self::Io(io::Error::other(other.to_string())),
        }
    }
}

impl AgentWorkspace {
    pub fn create(repo_root: &Path, run_id: &str) -> Result<Self, WorkspaceError> {
        validate_run_id(run_id)?;
        let started = Instant::now();
        trace_workspace_phase(run_id, "start", started);
        let data_root = workspace_data_root()?;
        let provider = ProviderInstallation::require_healthy(&data_root)?;
        trace_workspace_phase(run_id, "provider-healthy", started);
        provider.doctor_io(&format!("startup-{run_id}"))?;
        trace_workspace_phase(run_id, "provider-io-verified", started);
        #[cfg(target_os = "macos")]
        {
            greppy_workspace_core::spawn_repository_tracker(data_root.clone())?;
            trace_workspace_phase(run_id, "repository-tracker-started", started);
        }
        let provider_instance = provider.manifest().instance_id.clone();
        let core = WorkspaceCore::open(data_root.join("core"))?;
        trace_workspace_phase(run_id, "core-open", started);
        recover_proposal_publish_journals(&core)?;
        recover_apply_journals(&core)?;
        trace_workspace_phase(run_id, "recovery-complete", started);
        let (baseline, captured_snapshot_owns_chunks) =
            capture_tracked_repository(repo_root, &core, run_id, started)?;
        trace_workspace_phase(run_id, "snapshot-captured", started);
        let repo_root = baseline.repository.clone();
        let base_commit = baseline.base_commit.clone();
        let baseline_hash = baseline.baseline_hash.clone();
        let baseline_for_git = baseline.clone();
        let git_run_id = git_workspace_id(run_id);
        let pair_lease = core.begin_workspace_pair(run_id, &git_run_id)?;
        trace_workspace_phase(run_id, "pair-started", started);
        let handle = if captured_snapshot_owns_chunks {
            core.create_workspace(run_id, baseline)
        } else {
            core.create_workspace_from_shared_baseline(run_id, &baseline)
        };
        let handle = match handle {
            Ok(handle) => handle,
            Err(error) => {
                let _ = core.abort_workspace_pair(run_id, &git_run_id);
                return Err(error.into());
            }
        };
        trace_workspace_phase(run_id, "content-namespace-created", started);
        let worktree = match provider.workspace_path(run_id) {
            Ok(path) => path,
            Err(error) => {
                let _ = core.abort_workspace_pair(run_id, &git_run_id);
                return Err(error.into());
            }
        };
        if let Err(error) = wait_for_workspace_snapshot(&worktree, &baseline_for_git.entries) {
            let _ = core.abort_workspace_pair(run_id, &git_run_id);
            return Err(error);
        }
        trace_workspace_phase(run_id, "content-visible", started);
        let (git_baseline, git_baseline_owns_chunks, baseline_tree, baseline_view_commit) =
            match prepare_git_control_baseline(
                &repo_root,
                &worktree,
                &data_root,
                &baseline_for_git,
                core.chunks(),
            ) {
                Ok(prepared) => prepared,
                Err(error) => {
                    let _ = core.abort_workspace_pair(run_id, &git_run_id);
                    return Err(error);
                }
            };
        trace_workspace_phase(run_id, "git-baseline-prepared", started);
        let git_handle = if git_baseline_owns_chunks {
            core.create_overlay_workspace(&git_run_id, git_baseline)
        } else {
            core.create_overlay_workspace_from_shared_baseline(&git_run_id, &git_baseline)
        };
        let git_handle = match git_handle {
            Ok(handle) => handle,
            Err(error) => {
                let _ = core.abort_workspace_pair(run_id, &git_run_id);
                return Err(error.into());
            }
        };
        trace_workspace_phase(run_id, "git-namespace-created", started);
        let private_git_dir = match provider.workspace_path(&git_run_id) {
            Ok(path) => path,
            Err(error) => {
                let _ = core.abort_workspace_pair(run_id, &git_run_id);
                return Err(error.into());
            }
        };
        if let Err(error) = wait_for_workspace(&private_git_dir) {
            let _ = core.abort_workspace_pair(run_id, &git_run_id);
            return Err(error);
        }
        trace_workspace_phase(run_id, "git-namespace-visible", started);
        let initialized = initialize_private_git(
            &worktree,
            &private_git_dir,
            &baseline_tree,
            &baseline_view_commit,
        );
        let (baseline_tree, baseline_view_commit, private_index) = match initialized {
            Ok(result) => result,
            Err(error) => {
                let _ = core.abort_workspace_pair(run_id, &git_run_id);
                return Err(error);
            }
        };
        trace_workspace_phase(run_id, "git-initialized", started);
        if let Err(error) = core.complete_workspace_pair(&handle, &git_handle) {
            let _ = core.abort_workspace_pair(run_id, &git_run_id);
            return Err(error.into());
        }
        trace_workspace_phase(run_id, "pair-committed", started);
        Ok(Self {
            repo_root,
            worktree,
            private_git_dir,
            private_index,
            git_handle,
            run_id: run_id.into(),
            base_commit,
            baseline_hash,
            baseline_tree,
            baseline_view_commit,
            provider_instance,
            data_root,
            core,
            handle,
            pair_lease,
        })
    }

    pub fn worktree_path(&self) -> &Path {
        &self.worktree
    }

    /// Original repository captured into this immutable workspace view.
    pub fn repository_path(&self) -> &Path {
        &self.repo_root
    }

    pub fn linked_git_dir(&self) -> &Path {
        &self.private_git_dir
    }

    pub fn git_index_path(&self) -> &Path {
        &self.private_index
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Private Greppy Store for this workspace. This must live under the
    /// provider's writable data root, never beside a mounted workspace: the
    /// mount namespace intentionally exposes only WorkspaceCore-managed IDs.
    pub fn agent_data_root(&self) -> PathBuf {
        self.data_root.join("agent-data").join(&self.run_id)
    }

    /// Per-run tool scratch is provider-private for the same reason as the
    /// Greppy Store: arbitrary siblings are not part of the mounted namespace.
    pub fn agent_scratch_root(&self) -> PathBuf {
        self.data_root.join("agent-scratch").join(&self.run_id)
    }

    pub fn base_commit(&self) -> &str {
        &self.base_commit
    }

    pub fn is_stable(&self) -> bool {
        false
    }

    pub fn ref_name(&self) -> String {
        format!("refs/greppy/agent/{}", self.run_id)
    }

    pub fn keep(&self) -> Result<(), WorkspaceError> {
        self.verify_identity()?;
        self.core
            .keep_workspace_pair(&self.handle, &self.git_handle)?;
        Ok(())
    }

    pub fn finish(&self, message: &str) -> Result<RunOutcome, WorkspaceError> {
        self.verify_identity()?;
        let changed_paths = filter_ignored_paths(
            &self.worktree,
            &self.private_index,
            self.core.changed_paths(&self.handle)?,
        )?;
        let hardlink_groups = self.core.hardlink_groups(&self.handle, &changed_paths)?;
        if !changed_paths.is_empty() {
            let mut arguments = vec!["add", "-A", "--"];
            arguments.extend(changed_paths.iter().map(String::as_str));
            git_with_index(&self.worktree, &self.private_index, &arguments)?;
        }
        stage_hardlink_groups(
            &self.core,
            &self.handle,
            &self.worktree,
            &self.private_index,
            &hardlink_groups,
        )?;
        let final_tree = git_with_index(&self.worktree, &self.private_index, &["write-tree"])?;
        if final_tree == self.baseline_tree {
            return Ok(RunOutcome::Clean);
        }
        let commit_message = proposal_commit_message(message, &hardlink_groups);
        let commit = commit_tree(
            &self.worktree,
            &final_tree,
            &self.base_commit,
            &commit_message,
        )?;
        let export_ref = "refs/greppy/export/proposal";
        git_ok(&self.worktree, &["update-ref", export_ref, &commit])?;
        let baseline_export_ref = "refs/greppy/export/baseline";
        git_ok(
            &self.worktree,
            &[
                "update-ref",
                baseline_export_ref,
                &self.baseline_view_commit,
            ],
        )?;
        git_ok(
            &self.repo_root,
            &[
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                path_text(&self.private_git_dir)?,
                export_ref,
                baseline_export_ref,
            ],
        )?;

        let ref_name = self.ref_name();
        let baseline_ref = format!("refs/greppy/baselines/{}", self.run_id);
        publish_proposal_transaction(
            &self.core,
            &self.handle,
            &self.repo_root,
            &ref_name,
            &baseline_ref,
            &self.baseline_view_commit,
            &self.baseline_tree,
            &final_tree,
            &commit,
            &hardlink_groups,
        )?;
        let patch = git_ok(
            &self.repo_root,
            &["diff", "--binary", &self.baseline_tree, &final_tree],
        )?;
        let stat = git_ok(
            &self.repo_root,
            &["diff", "--stat", &self.baseline_tree, &final_tree],
        )?;
        Ok(RunOutcome::Proposal {
            commit,
            ref_name,
            patch,
            stat,
        })
    }

    /// Apply only the Agent delta to the exact dirty baseline. `git apply`
    /// operates on the working tree without `--index`; an explicit index hash
    /// assertion proves that staged user state was left byte-identical.
    pub fn apply_to(&self, target_checkout: &Path, commit: &str) -> Result<(), WorkspaceError> {
        self.verify_identity()?;
        let ref_name = self.ref_name();
        apply_from_core(&self.core, target_checkout, &ref_name, Some(commit))
    }

    pub fn cleanup(self) -> Result<(), WorkspaceError> {
        self.verify_identity()?;
        for private_root in [self.agent_data_root(), self.agent_scratch_root()] {
            match fs::remove_dir_all(&private_root) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        self.core
            .remove_workspace_pair(self.handle, self.git_handle)?;
        drop(self.pair_lease);
        Ok(())
    }

    fn verify_identity(&self) -> Result<(), WorkspaceError> {
        let provider = ProviderInstallation::require_healthy(&self.data_root)?;
        if provider.manifest().instance_id != self.provider_instance {
            return Err(WorkspaceError::Tampered {
                path: self.worktree.clone(),
                detail: "provider instance changed during the agent run".into(),
            });
        }
        if provider.workspace_path(&self.run_id)? != self.worktree
            || provider.workspace_path(self.git_handle.id())? != self.private_git_dir
            || !self.private_git_dir.is_dir()
        {
            return Err(WorkspaceError::Tampered {
                path: self.worktree.clone(),
                detail: "mounted workspace or private Git directory disappeared".into(),
            });
        }
        let status = self.core.status(&self.handle)?;
        if status.base_commit != self.base_commit || status.baseline_hash != self.baseline_hash {
            return Err(WorkspaceError::Tampered {
                path: self.worktree.clone(),
                detail: "workspace baseline metadata changed".into(),
            });
        }
        let git_status = self.core.status(&self.git_handle)?;
        if !git_status.base_commit.starts_with("virtual-empty:") {
            return Err(WorkspaceError::Tampered {
                path: self.private_git_dir.clone(),
                detail: "private Git namespace lost its virtual empty-base binding".into(),
            });
        }
        Ok(())
    }
}

fn trace_workspace_phase(run_id: &str, phase: &str, started: Instant) {
    let Some(root) = std::env::var_os("GREPPY_WORKSPACE_PHASE_TRACE_DIR") else {
        return;
    };
    let root = PathBuf::from(root);
    if !root.is_absolute() || fs::create_dir_all(&root).is_err() {
        return;
    }
    let event = serde_json::json!({
        "run_id": run_id,
        "phase": phase,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1_000.0,
    });
    let Ok(mut encoded) = serde_json::to_vec(&event) else {
        return;
    };
    encoded.push(b'\n');
    if let Ok(mut output) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(format!("{run_id}.jsonl")))
    {
        let _ = output.write_all(&encoded);
    }
}

fn capture_tracked_repository(
    repository: &Path,
    core: &WorkspaceCore,
    run_id: &str,
    started: Instant,
) -> Result<(BaselineSnapshot, bool), WorkspaceError> {
    let repository = fs::canonicalize(repository)?;
    core.request_repository_tracker(&repository)?;
    trace_workspace_phase(run_id, "tracker-requested", started);
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let active = loop {
        if let Some(status) = core.repository_tracker_status(&repository)? {
            if status.state == RepositoryTrackerState::Active {
                trace_workspace_phase(run_id, "tracker-active", started);
                break status;
            }
            if status.state == RepositoryTrackerState::Gap {
                return Err(WorkspaceError::AdapterUnavailable(format!(
                    "repository tracker has an event gap: {}",
                    status
                        .detail
                        .unwrap_or_else(|| "unknown watcher failure".into())
                )));
            }
        }
        if std::time::Instant::now() >= deadline {
            return Err(WorkspaceError::AdapterUnavailable(
                "repository tracker did not become active within three seconds".into(),
            ));
        }
        thread::sleep(Duration::from_millis(20));
    };

    trace_workspace_phase(run_id, "tracker-fence-start", started);
    repository_tracker_fence(&repository, core, active.epoch)?;
    trace_workspace_phase(run_id, "tracker-fence-complete", started);
    if let Some(mut cached) = core.cached_repository_snapshot(&repository, active.epoch)? {
        trace_workspace_phase(run_id, "cached-snapshot-found", started);
        let cached_generation = cached.tracker_generation.ok_or_else(|| {
            WorkspaceError::AdapterUnavailable("cached baseline has no tracker generation".into())
        })?;
        let changes =
            core.repository_changes_since(&repository, active.epoch, cached_generation)?;
        let changed_paths = changes
            .paths
            .iter()
            .filter(|path| !is_repository_tracker_fence(path))
            .cloned()
            .collect::<Vec<_>>();
        if changed_paths.is_empty() {
            cached.tracker_generation = Some(changes.generation);
            return Ok((cached, false));
        }
        match capture_repository_incremental(
            &repository,
            core.chunks(),
            &cached,
            &changed_paths,
            active.epoch,
            changes.generation,
        ) {
            Ok(Some(mut incremental)) => {
                repository_tracker_fence(&repository, core, active.epoch)?;
                let trailing =
                    core.repository_changes_since(&repository, active.epoch, changes.generation)?;
                if trailing
                    .paths
                    .iter()
                    .all(|path| is_repository_tracker_fence(path))
                {
                    incremental.tracker_generation = Some(trailing.generation);
                    return Ok((incremental, true));
                }
                release_snapshot(core.chunks(), incremental);
            }
            Ok(None) | Err(greppy_workspace_core::Error::ConcurrentRepositoryMutation) => {}
            Err(error) => return Err(error.into()),
        }
    }

    for attempt in 0..2 {
        let before = core
            .repository_tracker_status(&repository)?
            .ok_or_else(|| {
                WorkspaceError::AdapterUnavailable("repository tracker disappeared".into())
            })?;
        if before.state != RepositoryTrackerState::Active || before.epoch != active.epoch {
            return Err(WorkspaceError::AdapterUnavailable(
                format!(
                    "repository tracker invalid before snapshot capture: state={:?}, epoch={}, generation={}, expected_epoch={}, detail={}",
                    before.state,
                    before.epoch,
                    before.generation,
                    active.epoch,
                    before.detail.as_deref().unwrap_or("none")
                ),
            ));
        }
        let mut baseline = capture_repository_with_observer(&repository, core.chunks(), |phase| {
            trace_workspace_phase(run_id, phase, started)
        })?;
        trace_workspace_phase(run_id, "full-snapshot-captured", started);
        repository_tracker_fence(&repository, core, active.epoch)?;
        let after = core
            .repository_tracker_status(&repository)?
            .ok_or_else(|| {
                WorkspaceError::AdapterUnavailable("repository tracker disappeared".into())
            })?;
        if after.state != RepositoryTrackerState::Active || after.epoch != before.epoch {
            trace_workspace_phase(run_id, "tracker-invalidated-after-full-snapshot", started);
            release_snapshot(core.chunks(), baseline);
            return Err(WorkspaceError::AdapterUnavailable(format!(
                "repository tracker invalidated during snapshot capture: state={:?}, epoch={}, generation={}, expected_epoch={}, expected_generation={}, detail={}",
                after.state,
                after.epoch,
                after.generation,
                before.epoch,
                before.generation,
                after.detail.as_deref().unwrap_or("none")
            )));
        }
        let changes =
            core.repository_changes_since(&repository, before.epoch, before.generation)?;
        if changes
            .paths
            .iter()
            .all(|path| is_repository_tracker_fence(path))
        {
            baseline.tracker_epoch = Some(after.epoch);
            baseline.tracker_generation = Some(changes.generation);
            return Ok((baseline, true));
        }
        trace_workspace_phase(
            run_id,
            "tracker-generation-changed-after-full-snapshot",
            started,
        );
        release_snapshot(core.chunks(), baseline);
        if attempt == 1 {
            return Err(WorkspaceError::Unsupported(
                "repository changed during both snapshot attempts".into(),
            ));
        }
    }
    unreachable!("two bounded snapshot attempts")
}

fn is_repository_tracker_fence(path: &str) -> bool {
    path.starts_with(".git/greppy-tracker-fence-")
}

fn repository_tracker_fence(
    repository: &Path,
    core: &WorkspaceCore,
    epoch: u64,
) -> Result<u64, WorkspaceError> {
    let git_dir = PathBuf::from(git_ok(
        repository,
        &["rev-parse", "--path-format=absolute", "--absolute-git-dir"],
    )?);
    let name = format!(
        "greppy-tracker-fence-{}-{}",
        std::process::id(),
        now_unix_ns()
    );
    let path = git_dir.join(&name);
    let virtual_path = format!(".git/{name}");
    let before = core.repository_tracker_status(repository)?.ok_or_else(|| {
        WorkspaceError::AdapterUnavailable("repository tracker disappeared".into())
    })?;
    if before.state != RepositoryTrackerState::Active || before.epoch != epoch {
        return Err(WorkspaceError::AdapterUnavailable(
            "repository tracker changed before fence".into(),
        ));
    }
    fs::write(&path, b"greppy.repository-tracker-fence.v1\n")?;
    let created = wait_for_tracker_path(repository, core, epoch, before.generation, &virtual_path);
    let remove = fs::remove_file(&path);
    let created = created?;
    remove?;
    // The observed create event is the ordering barrier: every repository
    // event that happened before the fence write has reached the journal.
    // Removal is synchronous and fence paths are excluded from every snapshot
    // delta, so waiting for a second watcher round-trip adds latency without
    // strengthening the captured baseline. A delayed removal event remains a
    // harmless ignored journal entry for the next fence.
    Ok(created)
}

fn wait_for_tracker_path(
    repository: &Path,
    core: &WorkspaceCore,
    epoch: u64,
    after_generation: u64,
    expected_path: &str,
) -> Result<u64, WorkspaceError> {
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        let status = core.repository_tracker_status(repository)?.ok_or_else(|| {
            WorkspaceError::AdapterUnavailable("repository tracker disappeared".into())
        })?;
        if status.state != RepositoryTrackerState::Active || status.epoch != epoch {
            return Err(WorkspaceError::AdapterUnavailable(format!(
                "repository tracker lost continuity during fence: {}",
                status
                    .detail
                    .unwrap_or_else(|| "epoch/state changed".into())
            )));
        }
        if status.generation > after_generation {
            let changes = core.repository_changes_since(repository, epoch, after_generation)?;
            if changes.paths.iter().any(|path| path == expected_path) {
                return Ok(status.generation);
            }
        }
        if std::time::Instant::now() >= deadline {
            return Err(WorkspaceError::AdapterUnavailable(format!(
                "repository tracker fence timed out for {expected_path}"
            )));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Apply a persisted proposal after its agent workspace has been cleaned up.
pub fn apply_proposal(target_checkout: &Path, ref_name: &str) -> Result<(), WorkspaceError> {
    let data_root = workspace_data_root()?;
    let core = WorkspaceCore::open(data_root.join("core"))?;
    recover_proposal_publish_journals(&core)?;
    recover_apply_journals(&core)?;
    let _operation_lease = acquire_repository_operation_lease(&core, target_checkout, ref_name)?;
    apply_from_core(&core, target_checkout, ref_name, None)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProposalPublishJournal {
    schema: u32,
    workspace_id: String,
    ref_name: String,
    baseline_ref: String,
    repository: PathBuf,
    baseline_hash: String,
    baseline_view_commit: String,
    baseline_tree: String,
    final_tree: String,
    proposal_commit: String,
    #[serde(default)]
    hardlink_groups: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApplyJournal {
    schema: u32,
    ref_name: String,
    repository: PathBuf,
    baseline_hash: String,
    baseline_tree: String,
    affected_paths: Vec<String>,
    modified_times: Vec<(String, i64)>,
}

#[cfg(test)]
fn test_crash_point(point: &str) {
    if std::env::var_os("GREPPY_AGENT_TEST_CRASH_POINT").as_deref()
        == Some(std::ffi::OsStr::new(point))
    {
        std::process::abort();
    }
}

#[allow(clippy::too_many_arguments)]
fn publish_proposal_transaction(
    core: &WorkspaceCore,
    workspace: &WorkspaceHandle,
    repository: &Path,
    ref_name: &str,
    baseline_ref: &str,
    baseline_view_commit: &str,
    baseline_tree: &str,
    final_tree: &str,
    proposal_commit: &str,
    hardlink_groups: &[Vec<String>],
) -> Result<(), WorkspaceError> {
    let _operation_lease = acquire_repository_operation_lease(core, repository, ref_name)?;
    let baseline = core.workspace_baseline(workspace)?;
    let journal = ProposalPublishJournal {
        schema: 1,
        workspace_id: workspace.id().into(),
        ref_name: ref_name.into(),
        baseline_ref: baseline_ref.into(),
        repository: repository.canonicalize()?,
        baseline_hash: baseline.baseline_hash,
        baseline_view_commit: baseline_view_commit.into(),
        baseline_tree: baseline_tree.into(),
        final_tree: final_tree.into(),
        proposal_commit: proposal_commit.into(),
        hardlink_groups: hardlink_groups.to_vec(),
    };
    validate_proposal_publish_journal(core, &journal, Path::new("proposal publication"))?;
    let journal_path = publish_proposal_journal(core, &journal)?;
    let result = (|| -> Result<(), WorkspaceError> {
        git_ok(
            &journal.repository,
            &[
                "update-ref",
                &journal.baseline_ref,
                &journal.baseline_view_commit,
            ],
        )?;
        core.preserve_proposal(
            workspace,
            &journal.ref_name,
            &journal.baseline_tree,
            &journal.final_tree,
            &journal.proposal_commit,
            &journal.hardlink_groups,
        )?;
        #[cfg(test)]
        test_crash_point("proposal-after-core-record");
        git_ok(
            &journal.repository,
            &["update-ref", &journal.ref_name, &journal.proposal_commit],
        )?;
        Ok(())
    })();
    match result {
        Ok(()) => remove_proposal_publish_journal(&journal_path),
        Err(error) => {
            if recover_proposal_publish_journal_locked(core, &journal_path, &journal)? {
                Ok(())
            } else {
                Err(error)
            }
        }
    }
}

fn proposal_publish_journal_root(core: &WorkspaceCore) -> PathBuf {
    core.root().join("proposal-publish-journals")
}

fn publish_proposal_journal(
    core: &WorkspaceCore,
    journal: &ProposalPublishJournal,
) -> Result<PathBuf, WorkspaceError> {
    let root = proposal_publish_journal_root(core);
    fs::create_dir_all(&root)?;
    let mut hasher = Sha256::new();
    hasher.update(journal.repository.as_os_str().to_string_lossy().as_bytes());
    hasher.update([0]);
    hasher.update(journal.ref_name.as_bytes());
    let id = format!("{:x}", hasher.finalize());
    let path = root.join(format!("{id}.json"));
    let temporary = root.join(format!(".{id}.{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec(journal)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, &path)?;
    sync_directory(&root)?;
    Ok(path)
}

fn recover_proposal_publish_journals(core: &WorkspaceCore) -> Result<(), WorkspaceError> {
    let root = proposal_publish_journal_root(core);
    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let mut paths = entries
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(WorkspaceError::Tampered {
                path,
                detail: "proposal publication journal is not a regular file".into(),
            });
        }
        let bytes = fs::read(&path)?;
        let journal: ProposalPublishJournal =
            serde_json::from_slice(&bytes).map_err(|error| WorkspaceError::Tampered {
                path: path.clone(),
                detail: format!("proposal publication journal is invalid: {error}"),
            })?;
        recover_proposal_publish_journal(core, &path, &journal)?;
    }
    Ok(())
}

fn recover_proposal_publish_journal(
    core: &WorkspaceCore,
    journal_path: &Path,
    journal: &ProposalPublishJournal,
) -> Result<bool, WorkspaceError> {
    let _operation_lease =
        acquire_repository_operation_lease(core, &journal.repository, &journal.ref_name)?;
    recover_proposal_publish_journal_locked(core, journal_path, journal)
}

fn recover_proposal_publish_journal_locked(
    core: &WorkspaceCore,
    journal_path: &Path,
    journal: &ProposalPublishJournal,
) -> Result<bool, WorkspaceError> {
    let _workspace = validate_proposal_publish_journal(core, journal, journal_path)?;
    let proposal = if core.has_proposal(&journal.ref_name)? {
        Some(core.proposal(&journal.ref_name)?)
    } else {
        None
    };
    if let Some(proposal) = proposal.as_ref() {
        if proposal.repository != journal.repository
            || proposal.baseline_hash != journal.baseline_hash
            || proposal.baseline_tree != journal.baseline_tree
            || proposal.final_tree != journal.final_tree
            || proposal.proposal_commit != journal.proposal_commit
            || proposal.hardlink_groups != journal.hardlink_groups
        {
            return Err(WorkspaceError::Tampered {
                path: journal_path.to_path_buf(),
                detail: "proposal publication journal does not match the pinned proposal".into(),
            });
        }
    }
    let proposal_ref = read_optional_commit_ref(&journal.repository, &journal.ref_name)?;
    let baseline_ref = read_optional_commit_ref(&journal.repository, &journal.baseline_ref)?;
    let complete = proposal.is_some()
        && proposal_ref.as_deref() == Some(journal.proposal_commit.as_str())
        && baseline_ref.as_deref() == Some(journal.baseline_view_commit.as_str());
    if complete {
        remove_proposal_publish_journal(journal_path)?;
        return Ok(true);
    }
    delete_ref_if_expected(
        &journal.repository,
        &journal.ref_name,
        proposal_ref,
        &journal.proposal_commit,
        journal_path,
    )?;
    delete_ref_if_expected(
        &journal.repository,
        &journal.baseline_ref,
        baseline_ref,
        &journal.baseline_view_commit,
        journal_path,
    )?;
    if proposal.is_some() {
        core.remove_proposal(&journal.ref_name)?;
    }
    remove_proposal_publish_journal(journal_path)?;
    Ok(false)
}

fn validate_proposal_publish_journal(
    core: &WorkspaceCore,
    journal: &ProposalPublishJournal,
    journal_path: &Path,
) -> Result<WorkspaceHandle, WorkspaceError> {
    let invalid = |detail: &str| WorkspaceError::Tampered {
        path: journal_path.to_path_buf(),
        detail: detail.into(),
    };
    if journal.schema != 1 {
        return Err(invalid("unsupported proposal publication journal schema"));
    }
    validate_run_id(&journal.workspace_id)?;
    if journal.ref_name != format!("refs/greppy/agent/{}", journal.workspace_id)
        || journal.baseline_ref != format!("refs/greppy/baselines/{}", journal.workspace_id)
    {
        return Err(invalid(
            "proposal publication refs do not match the workspace id",
        ));
    }
    for (name, value) in [
        (
            "baseline view commit",
            journal.baseline_view_commit.as_str(),
        ),
        ("baseline tree", journal.baseline_tree.as_str()),
        ("final tree", journal.final_tree.as_str()),
        ("proposal commit", journal.proposal_commit.as_str()),
    ] {
        if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid(&format!("invalid {name}")));
        }
    }
    let mut seen_paths = BTreeSet::new();
    let mut prior_group = None::<&Vec<String>>;
    for group in &journal.hardlink_groups {
        if group.len() < 2 {
            return Err(invalid(
                "proposal hardlink group must contain at least two paths",
            ));
        }
        if prior_group.is_some_and(|prior| prior >= group) {
            return Err(invalid("proposal hardlink groups are not canonical"));
        }
        let mut prior_path = None::<&String>;
        for path in group {
            validate_apply_path(path)?;
            if prior_path.is_some_and(|prior| prior >= path) || !seen_paths.insert(path) {
                return Err(invalid(
                    "proposal hardlink paths are not canonical and disjoint",
                ));
            }
            prior_path = Some(path);
        }
        prior_group = Some(group);
    }
    let repository = journal.repository.canonicalize()?;
    if repository != journal.repository {
        return Err(invalid("proposal publication repository is not canonical"));
    }
    let workspace = core.open_workspace(&journal.workspace_id)?;
    let baseline = core.workspace_baseline(&workspace)?;
    if baseline.repository != repository || baseline.baseline_hash != journal.baseline_hash {
        return Err(invalid(
            "proposal publication journal does not match the workspace baseline",
        ));
    }
    validate_commit_hardlink_binding(
        &repository,
        &journal.proposal_commit,
        &journal.hardlink_groups,
    )?;
    Ok(workspace)
}

fn read_optional_commit_ref(
    repository: &Path,
    ref_name: &str,
) -> Result<Option<String>, WorkspaceError> {
    let expression = format!("{ref_name}^{{commit}}");
    let output = Command::new("git")
        .args(["rev-parse", "--verify", "--quiet", &expression])
        .current_dir(repository)
        .output()?;
    if output.status.success() {
        return Ok(Some(String::from_utf8_lossy(&output.stdout).trim().into()));
    }
    if output.status.code() == Some(1) {
        return Ok(None);
    }
    Err(git_failed("git read proposal publication ref", &output))
}

fn delete_ref_if_expected(
    repository: &Path,
    ref_name: &str,
    observed: Option<String>,
    expected: &str,
    journal_path: &Path,
) -> Result<(), WorkspaceError> {
    let Some(observed) = observed else {
        return Ok(());
    };
    if observed != expected {
        return Err(WorkspaceError::Tampered {
            path: journal_path.to_path_buf(),
            detail: format!("{ref_name} moved during proposal publication recovery"),
        });
    }
    git_ok(repository, &["update-ref", "-d", ref_name, expected])?;
    Ok(())
}

fn remove_proposal_publish_journal(path: &Path) -> Result<(), WorkspaceError> {
    fs::remove_file(path)?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn acquire_repository_operation_lease(
    core: &WorkspaceCore,
    repository: &Path,
    ref_name: &str,
) -> Result<greppy_workspace_core::WorkspaceOperationLease, WorkspaceError> {
    core.try_repository_operation_lease(repository)?
        .ok_or_else(|| WorkspaceError::Conflict {
            ref_name: ref_name.into(),
            detail: "another proposal/apply operation is active for this repository".into(),
        })
}

fn apply_from_core(
    core: &WorkspaceCore,
    target_checkout: &Path,
    ref_name: &str,
    expected_commit: Option<&str>,
) -> Result<(), WorkspaceError> {
    let proposal = core.proposal(ref_name)?;
    if expected_commit.is_some_and(|commit| proposal.proposal_commit != commit) {
        return Err(WorkspaceError::Tampered {
            path: target_checkout.to_path_buf(),
            detail: "requested commit does not match pinned proposal metadata".into(),
        });
    }
    let canonical_target = target_checkout.canonicalize()?;
    if canonical_target != proposal.repository {
        return Err(WorkspaceError::DirtyTarget {
            ref_name: ref_name.into(),
            detail: "proposal belongs to a different repository".into(),
        });
    }
    validate_proposal_git_binding(&canonical_target, &proposal)?;
    let observed = capture_repository(&canonical_target, core.chunks())?;
    let observed_hash = observed.baseline_hash.clone();
    release_snapshot(core.chunks(), observed);
    if observed_hash != proposal.baseline_hash {
        return Err(WorkspaceError::DirtyTarget {
            ref_name: ref_name.into(),
            detail: format!(
                "baseline hash is {observed_hash}, expected {}",
                proposal.baseline_hash
            ),
        });
    }

    let index_path = git_path(&canonical_target, "index")?;
    let index_before = hash_optional_file(&index_path)?;
    let affected_paths = apply_affected_paths(&canonical_target, &proposal)?;
    let journal = ApplyJournal {
        schema: 1,
        ref_name: ref_name.into(),
        repository: canonical_target.clone(),
        baseline_hash: proposal.baseline_hash.clone(),
        baseline_tree: proposal.baseline_tree.clone(),
        affected_paths,
        modified_times: proposal
            .baseline
            .entries
            .iter()
            .filter(|entry| entry.kind != greppy_workspace_core::EntryKind::Tombstone)
            .map(|entry| (entry.path.clone(), entry.modified_unix_ns))
            .collect(),
    };
    let journal_path = publish_apply_journal(core, &journal)?;
    for path in &journal.affected_paths {
        if let Err(error) =
            materialize_git_tree_entry(&canonical_target, &proposal.final_tree, path)
        {
            restore_apply_journal(core, &journal_path, &journal)?;
            return Err(error);
        }
        #[cfg(test)]
        if journal.affected_paths.first() == Some(path) {
            test_crash_point("apply-after-first-path");
        }
    }
    if let Err(error) = materialize_hardlink_groups(&canonical_target, &proposal.hardlink_groups) {
        restore_apply_journal(core, &journal_path, &journal)?;
        return Err(error);
    }
    let index_after = hash_optional_file(&index_path)?;
    if index_before != index_after {
        restore_apply_journal(core, &journal_path, &journal)?;
        return Err(WorkspaceError::Tampered {
            path: index_path,
            detail: "Git index changed during a worktree-only apply".into(),
        });
    }
    let final_index = journal_path.with_extension("final-index");
    let read_final = Command::new("git")
        .args([
            "-C",
            path_text(&canonical_target)?,
            "read-tree",
            &proposal.final_tree,
        ])
        .env("GIT_INDEX_FILE", &final_index)
        .output()?;
    if !read_final.status.success() {
        restore_apply_journal(core, &journal_path, &journal)?;
        return Err(git_failed(
            "git read-tree for final apply check",
            &read_final,
        ));
    }
    let _ = Command::new("git")
        .args([
            "-C",
            path_text(&canonical_target)?,
            "update-index",
            "--refresh",
        ])
        .env("GIT_INDEX_FILE", &final_index)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    let final_check = Command::new("git")
        .args([
            "-C",
            path_text(&canonical_target)?,
            "diff-files",
            "--name-status",
            "--",
        ])
        .env("GIT_INDEX_FILE", &final_index)
        .output()?;
    let _ = fs::remove_file(final_index);
    if !final_check.status.success() || !final_check.stdout.is_empty() {
        restore_apply_journal(core, &journal_path, &journal)?;
        return Err(WorkspaceError::Tampered {
            path: canonical_target,
            detail: format!(
                "working tree does not exactly match the proposal final tree after apply: {}",
                String::from_utf8_lossy(&final_check.stdout).trim()
            ),
        });
    }
    remove_apply_journal(&journal_path)?;
    Ok(())
}

fn validate_proposal_git_binding(
    repository: &Path,
    proposal: &ProposalRecord,
) -> Result<(), WorkspaceError> {
    let read_git = |args: &[&str], subject: &str| -> Result<String, WorkspaceError> {
        let output = Command::new("git")
            .args(args)
            .current_dir(repository)
            .output()?;
        if !output.status.success() {
            return Err(WorkspaceError::Tampered {
                path: repository.to_path_buf(),
                detail: format!(
                    "proposal {subject} is not present in the repository object graph: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().into())
    };

    let ref_expression = format!("{}^{{commit}}", proposal.ref_name);
    let ref_commit = read_git(&["rev-parse", "--verify", &ref_expression], "ref")?;
    if ref_commit != proposal.proposal_commit {
        return Err(WorkspaceError::Tampered {
            path: repository.to_path_buf(),
            detail: "proposal ref does not resolve to its pinned commit".into(),
        });
    }

    let parents = read_git(
        &[
            "rev-list",
            "--parents",
            "-n",
            "1",
            &proposal.proposal_commit,
        ],
        "commit",
    )?;
    let parents = parents.split_ascii_whitespace().collect::<Vec<_>>();
    if parents.as_slice()
        != [
            proposal.proposal_commit.as_str(),
            proposal.base_commit.as_str(),
        ]
    {
        return Err(WorkspaceError::Tampered {
            path: repository.to_path_buf(),
            detail: "proposal commit does not have exactly the pinned base commit as parent".into(),
        });
    }

    let tree_expression = format!("{}^{{tree}}", proposal.proposal_commit);
    let final_tree = read_git(&["rev-parse", "--verify", &tree_expression], "final tree")?;
    if final_tree != proposal.final_tree {
        return Err(WorkspaceError::Tampered {
            path: repository.to_path_buf(),
            detail: "proposal commit tree does not match the pinned final tree".into(),
        });
    }
    let baseline_expression = format!("{}^{{tree}}", proposal.baseline_tree);
    let baseline_tree = read_git(
        &["rev-parse", "--verify", &baseline_expression],
        "baseline tree",
    )?;
    if baseline_tree != proposal.baseline_tree {
        return Err(WorkspaceError::Tampered {
            path: repository.to_path_buf(),
            detail: "proposal baseline tree is not a canonical Git tree".into(),
        });
    }
    validate_commit_hardlink_binding(
        repository,
        &proposal.proposal_commit,
        &proposal.hardlink_groups,
    )?;
    Ok(())
}

const HARDLINK_BINDING_TRAILER: &str = "Greppy-Hardlinks-SHA256: ";

fn proposal_hardlink_digest(groups: &[Vec<String>]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"greppy-proposal-hardlinks-v1\0");
    digest.update((groups.len() as u64).to_le_bytes());
    for group in groups {
        digest.update((group.len() as u64).to_le_bytes());
        for path in group {
            digest.update((path.len() as u64).to_le_bytes());
            digest.update(path.as_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn proposal_commit_message(message: &str, hardlink_groups: &[Vec<String>]) -> String {
    format!(
        "{}\n\n{}{}",
        message.trim_end(),
        HARDLINK_BINDING_TRAILER,
        proposal_hardlink_digest(hardlink_groups)
    )
}

fn validate_commit_hardlink_binding(
    repository: &Path,
    proposal_commit: &str,
    hardlink_groups: &[Vec<String>],
) -> Result<(), WorkspaceError> {
    let output = Command::new("git")
        .args(["show", "-s", "--format=%B", proposal_commit])
        .current_dir(repository)
        .output()?;
    if !output.status.success() {
        return Err(git_failed("git show proposal hardlink binding", &output));
    }
    let message = String::from_utf8_lossy(&output.stdout);
    let bindings = message
        .lines()
        .filter_map(|line| line.strip_prefix(HARDLINK_BINDING_TRAILER))
        .collect::<Vec<_>>();
    let expected = proposal_hardlink_digest(hardlink_groups);
    if bindings.as_slice() != [expected.as_str()] {
        return Err(WorkspaceError::Tampered {
            path: repository.to_path_buf(),
            detail: "proposal commit does not bind the pinned hardlink topology".into(),
        });
    }
    Ok(())
}

fn changed_paths(
    repository: &Path,
    baseline: &str,
    final_tree: &str,
) -> Result<Vec<String>, WorkspaceError> {
    let bytes = git_bytes(
        repository,
        &[
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            baseline,
            final_tree,
        ],
    )?;
    let mut paths = Vec::new();
    for value in bytes
        .split(|byte| *byte == 0)
        .filter(|value| !value.is_empty())
    {
        let path = String::from_utf8(value.to_vec()).map_err(|_| {
            WorkspaceError::Unsupported("proposal contains a non-UTF-8 path".into())
        })?;
        validate_apply_path(&path)?;
        paths.push(path);
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn stage_hardlink_groups(
    core: &WorkspaceCore,
    workspace: &WorkspaceHandle,
    worktree: &Path,
    index: &Path,
    hardlink_groups: &[Vec<String>],
) -> Result<(), WorkspaceError> {
    for group in hardlink_groups {
        let Some(canonical) = group.first() else {
            return Err(WorkspaceError::Tampered {
                path: worktree.to_path_buf(),
                detail: "proposal contains an empty hardlink group".into(),
            });
        };
        let metadata =
            core.metadata(workspace, canonical)?
                .ok_or_else(|| WorkspaceError::Tampered {
                    path: worktree.join(canonical),
                    detail: "hardlink group canonical path disappeared before staging".into(),
                })?;
        let mode = if metadata.mode & 0o111 != 0 {
            "100755"
        } else {
            "100644"
        };
        let object = git_ok(
            worktree,
            &["hash-object", "-w", "--no-filters", "--", canonical],
        )?;
        for path in group {
            validate_apply_path(path)?;
            let cache_info = format!("{mode},{object},{path}");
            git_with_index(
                worktree,
                index,
                &["update-index", "--add", "--cacheinfo", &cache_info],
            )?;
        }
    }
    Ok(())
}

fn apply_affected_paths(
    repository: &Path,
    proposal: &ProposalRecord,
) -> Result<Vec<String>, WorkspaceError> {
    let mut affected_paths =
        changed_paths(repository, &proposal.baseline_tree, &proposal.final_tree)?
            .into_iter()
            .collect::<BTreeSet<_>>();
    for path in proposal.hardlink_groups.iter().flatten() {
        validate_apply_path(path)?;
        affected_paths.insert(path.clone());
    }
    for group in &proposal.baseline.hardlink_groups {
        if group.iter().any(|path| affected_paths.contains(path)) {
            for path in group {
                validate_apply_path(path)?;
                affected_paths.insert(path.clone());
            }
        }
    }
    Ok(affected_paths.into_iter().collect())
}

fn validate_apply_path(path: &str) -> Result<(), WorkspaceError> {
    let value = Path::new(path);
    if path.is_empty()
        || value.is_absolute()
        || value.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        })
    {
        return Err(WorkspaceError::Tampered {
            path: value.into(),
            detail: "proposal contains an unsafe apply path".into(),
        });
    }
    Ok(())
}

fn apply_journal_root(core: &WorkspaceCore) -> PathBuf {
    core.root().join("apply-journals")
}

fn publish_apply_journal(
    core: &WorkspaceCore,
    journal: &ApplyJournal,
) -> Result<PathBuf, WorkspaceError> {
    let root = apply_journal_root(core);
    fs::create_dir_all(&root)?;
    let mut hasher = Sha256::new();
    hasher.update(journal.repository.as_os_str().to_string_lossy().as_bytes());
    hasher.update([0]);
    hasher.update(journal.ref_name.as_bytes());
    let id = format!("{:x}", hasher.finalize());
    let path = root.join(format!("{id}.json"));
    let temporary = root.join(format!(".{id}.{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec(journal)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, &path)?;
    sync_directory(&root)?;
    Ok(path)
}

fn recover_apply_journals(core: &WorkspaceCore) -> Result<(), WorkspaceError> {
    let root = apply_journal_root(core);
    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let mut paths = entries
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(WorkspaceError::Tampered {
                path,
                detail: "apply recovery journal is not a regular file".into(),
            });
        }
        let bytes = fs::read(&path)?;
        let journal: ApplyJournal =
            serde_json::from_slice(&bytes).map_err(|error| WorkspaceError::Tampered {
                path: path.clone(),
                detail: format!("apply recovery journal is invalid: {error}"),
            })?;
        let _operation_lease =
            acquire_repository_operation_lease(core, &journal.repository, &journal.ref_name)?;
        if journal.schema != 1 {
            return Err(WorkspaceError::Tampered {
                path,
                detail: format!("unsupported apply recovery schema {}", journal.schema),
            });
        }
        restore_apply_journal(core, &path, &journal)?;
    }
    Ok(())
}

fn restore_apply_journal(
    core: &WorkspaceCore,
    journal_path: &Path,
    journal: &ApplyJournal,
) -> Result<(), WorkspaceError> {
    let repository = journal.repository.canonicalize()?;
    let proposal = core.proposal(&journal.ref_name)?;
    if proposal.repository != repository
        || proposal.baseline_hash != journal.baseline_hash
        || proposal.baseline_tree != journal.baseline_tree
    {
        return Err(WorkspaceError::Tampered {
            path: journal_path.to_path_buf(),
            detail: "apply recovery journal does not match its pinned proposal metadata".into(),
        });
    }
    for path in &journal.affected_paths {
        validate_apply_path(path)?;
    }
    let index = journal_path.with_extension("index");
    let index_text = path_text(&index)?.to_string();
    let read_tree = Command::new("git")
        .args([
            "-C",
            path_text(&repository)?,
            "read-tree",
            &journal.baseline_tree,
        ])
        .env("GIT_INDEX_FILE", &index)
        .output()?;
    if !read_tree.status.success() {
        return Err(git_failed("git read-tree for apply recovery", &read_tree));
    }
    let mut removal = journal.affected_paths.clone();
    removal.sort_by_key(|path| std::cmp::Reverse(path.matches('/').count()));
    for path in &removal {
        remove_visible_path(&repository.join(path))?;
    }
    for path in &journal.affected_paths {
        let present = Command::new("git")
            .args([
                "-C",
                path_text(&repository)?,
                "ls-files",
                "--error-unmatch",
                "--",
                path,
            ])
            .env("GIT_INDEX_FILE", &index)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !present.success() {
            continue;
        }
        if let Some(parent) = repository.join(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let output = Command::new("git")
            .args([
                "-C",
                path_text(&repository)?,
                "checkout-index",
                "--force",
                "--",
                path,
            ])
            .env("GIT_INDEX_FILE", &index_text)
            .output()?;
        if !output.status.success() {
            return Err(git_failed("git checkout-index for apply recovery", &output));
        }
    }
    // checkout-index is correct for paths that were clean in the original
    // checkout, including the checkout conversion configured by Git. Dirty
    // and untracked paths are different: their exact visible bytes are pinned
    // in the baseline CAS and must not pass through autocrlf or a filter during
    // recovery.
    for entry in &proposal.baseline.entries {
        if journal.affected_paths.contains(&entry.path) {
            restore_pinned_baseline_entry(core, &repository, entry)?;
        }
    }
    let baseline_hardlinks = proposal
        .baseline
        .hardlink_groups
        .iter()
        .filter(|group| {
            group
                .iter()
                .any(|path| journal.affected_paths.contains(path))
        })
        .cloned()
        .collect::<Vec<_>>();
    for group in &baseline_hardlinks {
        if !group
            .iter()
            .all(|path| journal.affected_paths.contains(path))
        {
            return Err(WorkspaceError::Tampered {
                path: journal_path.to_path_buf(),
                detail: "apply journal contains only part of a baseline hardlink group".into(),
            });
        }
    }
    materialize_hardlink_groups(&repository, &baseline_hardlinks)?;
    for (path, modified_unix_ns) in &journal.modified_times {
        validate_apply_path(path)?;
        let target = repository.join(path);
        if fs::symlink_metadata(&target).is_err() {
            continue;
        }
        let seconds = modified_unix_ns.div_euclid(1_000_000_000);
        let nanos = modified_unix_ns.rem_euclid(1_000_000_000) as u32;
        let time = filetime::FileTime::from_unix_time(seconds, nanos);
        if fs::symlink_metadata(&target)?.file_type().is_symlink() {
            filetime::set_symlink_file_times(&target, time, time)?;
        } else {
            filetime::set_file_times(&target, time, time)?;
        }
    }
    let _ = fs::remove_file(&index);
    let observed = capture_repository(&repository, core.chunks())?;
    let observed_hash = observed.baseline_hash.clone();
    release_snapshot(core.chunks(), observed);
    if observed_hash != journal.baseline_hash {
        return Err(WorkspaceError::Tampered {
            path: repository,
            detail: format!(
                "apply rollback could not restore baseline {}; observed {observed_hash}",
                journal.baseline_hash
            ),
        });
    }
    remove_apply_journal(journal_path)
}

fn restore_pinned_baseline_entry(
    core: &WorkspaceCore,
    repository: &Path,
    entry: &BaselineEntry,
) -> Result<(), WorkspaceError> {
    validate_apply_path(&entry.path)?;
    let target = repository.join(&entry.path);
    remove_visible_path(&target)?;
    if entry.kind == EntryKind::Tombstone {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::with_capacity(entry.size.try_into().unwrap_or(0));
    for id in &entry.chunks {
        bytes.extend_from_slice(&core.chunks().read(*id)?);
    }
    bytes.truncate(
        entry
            .size
            .try_into()
            .map_err(|_| WorkspaceError::Tampered {
                path: target.clone(),
                detail: "captured baseline entry is too large for this platform".into(),
            })?,
    );
    if blake3::hash(&bytes).to_hex().as_str() != entry.content_hash {
        return Err(WorkspaceError::Tampered {
            path: target,
            detail: "captured baseline chunks do not match their content hash".into(),
        });
    }
    match entry.kind {
        EntryKind::File => {
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&target)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            set_restored_mode(&target, entry.mode)?;
        }
        EntryKind::Symlink => create_restored_symlink(&bytes, &target)?,
        EntryKind::Tombstone => unreachable!(),
    }
    Ok(())
}

fn materialize_git_tree_entry(
    repository: &Path,
    tree: &str,
    relative: &str,
) -> Result<(), WorkspaceError> {
    validate_apply_path(relative)?;
    let target = repository.join(relative);
    remove_visible_path(&target)?;
    let listing = git_bytes(repository, &["ls-tree", "-z", tree, "--", relative])?;
    if listing.is_empty() {
        return Ok(());
    }
    let record = listing
        .strip_suffix(&[0])
        .ok_or_else(|| WorkspaceError::Tampered {
            path: target.clone(),
            detail: "Git tree entry is not NUL terminated".into(),
        })?;
    let tab = record
        .iter()
        .position(|byte| *byte == b'\t')
        .ok_or_else(|| WorkspaceError::Tampered {
            path: target.clone(),
            detail: "Git tree entry has no path separator".into(),
        })?;
    let header = std::str::from_utf8(&record[..tab]).map_err(|_| WorkspaceError::Tampered {
        path: target.clone(),
        detail: "Git tree entry header is not UTF-8".into(),
    })?;
    let listed_path =
        std::str::from_utf8(&record[tab + 1..]).map_err(|_| WorkspaceError::Tampered {
            path: target.clone(),
            detail: "Git tree entry path is not UTF-8".into(),
        })?;
    if listed_path != relative {
        return Err(WorkspaceError::Tampered {
            path: target,
            detail: "Git tree returned a different path than requested".into(),
        });
    }
    let mut fields = header.split_ascii_whitespace();
    let mode = fields.next().unwrap_or_default();
    let kind = fields.next().unwrap_or_default();
    let oid = fields.next().unwrap_or_default();
    if fields.next().is_some() || kind != "blob" || oid.len() != 40 && oid.len() != 64 {
        return Err(WorkspaceError::Tampered {
            path: target,
            detail: format!("unsupported Git tree entry {header:?}"),
        });
    }
    let bytes = git_bytes(repository, &["cat-file", "blob", oid])?;
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    match mode {
        "100644" | "100755" => {
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&target)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            let restored_mode = if mode == "100755" { 0o755 } else { 0o644 };
            set_restored_mode(&target, restored_mode)?;
        }
        "120000" => create_restored_symlink(&bytes, &target)?,
        _ => {
            return Err(WorkspaceError::Tampered {
                path: target,
                detail: format!("unsupported Git tree mode {mode}"),
            });
        }
    }
    Ok(())
}

fn materialize_hardlink_groups(
    repository: &Path,
    groups: &[Vec<String>],
) -> Result<(), WorkspaceError> {
    let mut seen = BTreeSet::new();
    for group in groups {
        if group.len() < 2 {
            return Err(WorkspaceError::Tampered {
                path: repository.to_path_buf(),
                detail: "proposal hardlink group contains fewer than two paths".into(),
            });
        }
        let source_relative = &group[0];
        validate_apply_path(source_relative)?;
        let source = repository.join(source_relative);
        let source_metadata = fs::symlink_metadata(&source)?;
        if !source_metadata.file_type().is_file() {
            return Err(WorkspaceError::Tampered {
                path: source,
                detail: "proposal hardlink source is not a regular file".into(),
            });
        }
        for relative in group {
            validate_apply_path(relative)?;
            if !seen.insert(relative) {
                return Err(WorkspaceError::Tampered {
                    path: repository.join(relative),
                    detail: "proposal path belongs to more than one hardlink group".into(),
                });
            }
        }
        for relative in &group[1..] {
            let target = repository.join(relative);
            let target_metadata = fs::symlink_metadata(&target)?;
            if !target_metadata.file_type().is_file() {
                return Err(WorkspaceError::Tampered {
                    path: target,
                    detail: "proposal hardlink target is not a regular file".into(),
                });
            }
            remove_visible_path(&target)?;
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::hard_link(&source, &target)?;
            if let Some(parent) = target.parent() {
                sync_directory(parent)?;
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn set_restored_mode(path: &Path, mode: u32) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    fs::set_permissions(path, fs::Permissions::from_mode(mode & 0o777))
}

#[cfg(windows)]
fn set_restored_mode(_path: &Path, _mode: u32) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn create_restored_symlink(target: &[u8], path: &Path) -> io::Result<()> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt as _;
    std::os::unix::fs::symlink(OsStr::from_bytes(target), path)
}

#[cfg(windows)]
fn create_restored_symlink(target: &[u8], path: &Path) -> io::Result<()> {
    let target = String::from_utf8(target.to_vec())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 symlink target"))?;
    std::os::windows::fs::symlink_file(target, path)
}

fn remove_visible_path(path: &Path) -> Result<(), WorkspaceError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            fs::remove_dir_all(path)?
        }
        Ok(_) => fs::remove_file(path)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn remove_apply_journal(path: &Path) -> Result<(), WorkspaceError> {
    fs::remove_file(path)?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), WorkspaceError> {
    #[cfg(unix)]
    fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

pub fn workspace_data_root() -> Result<PathBuf, WorkspaceError> {
    if let Some(path) = std::env::var_os("GREPPY_WORKSPACE_DIR") {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err(WorkspaceError::AdapterUnavailable(
                "GREPPY_WORKSPACE_DIR must be absolute".into(),
            ));
        }
        return Ok(path);
    }
    #[cfg(target_os = "macos")]
    {
        Ok(home_dir()?.join("Library/Group Containers/group.ai.metricspace.greppy/workspace"))
    }
    #[cfg(target_os = "windows")]
    {
        let root = std::env::var_os("LOCALAPPDATA")
            .ok_or_else(|| WorkspaceError::AdapterUnavailable("LOCALAPPDATA is not set".into()))?;
        Ok(PathBuf::from(root).join("greppy/workspace"))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(root) = std::env::var_os("XDG_DATA_HOME") {
            return Ok(PathBuf::from(root).join("greppy/workspace"));
        }
        Ok(home_dir()?.join(".local/share/greppy/workspace"))
    }
}

#[cfg(unix)]
fn home_dir() -> Result<PathBuf, WorkspaceError> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| WorkspaceError::AdapterUnavailable("home directory is unavailable".into()))
}

fn prepare_git_control_baseline(
    repo_root: &Path,
    worktree: &Path,
    data_root: &Path,
    baseline: &BaselineSnapshot,
    chunks: &ChunkStore,
) -> Result<(BaselineSnapshot, bool, String, String), WorkspaceError> {
    let object_format = git_ok(repo_root, &["rev-parse", "--show-object-format"])?;
    let common = git_ok(
        repo_root,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )?;
    let objects = PathBuf::from(common).join("objects");
    let layer = ensure_shared_git_layer(
        data_root,
        worktree,
        baseline,
        chunks,
        &object_format,
        &objects,
    )?;

    let (control, owns_chunks, baseline_view_commit) = ensure_git_control_template(
        data_root,
        baseline,
        chunks,
        &layer,
        &objects,
        &object_format,
    )?;
    Ok((
        control,
        owns_chunks,
        layer.baseline_tree,
        baseline_view_commit,
    ))
}

fn ensure_git_control_template(
    data_root: &Path,
    repository_baseline: &BaselineSnapshot,
    chunks: &ChunkStore,
    layer: &SharedGitLayer,
    source_objects: &Path,
    object_format: &str,
) -> Result<(BaselineSnapshot, bool, String), WorkspaceError> {
    let templates = data_root.join("g").join("ct3");
    fs::create_dir_all(&templates)?;
    let final_root = templates.join(private_git_storage_key(&repository_baseline.baseline_hash)?);
    if final_root.exists() {
        let (snapshot, identity) = load_git_control_template(
            &final_root,
            repository_baseline,
            &layer.baseline_tree,
            object_format,
        )?;
        return Ok((snapshot, false, identity.baseline_view_commit));
    }

    let temporary = private_git_temporary_path(&templates);
    let payload = temporary.join("payload");
    fs::create_dir(&temporary)?;
    let build = (|| -> Result<(BaselineSnapshot, GitControlTemplateIdentity), WorkspaceError> {
        init_bare(&payload, object_format)?;
        let alternates = payload.join("objects/info/alternates");
        fs::create_dir_all(
            alternates
                .parent()
                .ok_or_else(|| io::Error::other("invalid template alternates path"))?,
        )?;
        fs::write(
            &alternates,
            format!(
                "{}\n{}\n",
                layer.objects.display(),
                source_objects.display()
            ),
        )?;
        fs::copy(&layer.index, payload.join("index"))?;
        let shared_name = layer
            .shared_index
            .file_name()
            .ok_or_else(|| io::Error::other("shared Git index has no file name"))?;
        fs::copy(&layer.shared_index, payload.join(shared_name))?;
        configure_git_control_template(&payload)?;
        let baseline_view_commit = commit_tree_in_git_dir(
            &payload,
            &layer.baseline_tree,
            &repository_baseline.base_commit,
            "greppy private baseline view",
        )?;
        fs::create_dir_all(payload.join("refs/heads"))?;
        fs::write(
            payload.join("refs/heads/greppy-baseline"),
            format!("{baseline_view_commit}\n"),
        )?;
        fs::write(payload.join("HEAD"), b"ref: refs/heads/greppy-baseline\n")?;
        let snapshot = capture_overlay_directory(final_root.join("namespace"), &payload, chunks)?;
        Ok((
            snapshot,
            GitControlTemplateIdentity {
                schema: 2,
                baseline_hash: repository_baseline.baseline_hash.clone(),
                base_commit: repository_baseline.base_commit.clone(),
                baseline_tree: layer.baseline_tree.clone(),
                baseline_view_commit,
                object_format: object_format.into(),
            },
        ))
    })();
    let (snapshot, identity) = match build {
        Ok(result) => result,
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
    };
    fs::write(
        temporary.join("baseline.json"),
        serde_json::to_vec(&snapshot).map_err(|error| io::Error::other(error.to_string()))?,
    )?;
    fs::write(
        temporary.join("identity.json"),
        serde_json::to_vec(&identity).map_err(|error| io::Error::other(error.to_string()))?,
    )?;
    fs::write(
        temporary.join("COMPLETE"),
        b"greppy.git-control-template.v2\n",
    )?;
    match fs::rename(&temporary, &final_root) {
        Ok(()) => Ok((snapshot, true, identity.baseline_view_commit)),
        Err(_error) if final_root.exists() => {
            let _ = fs::remove_dir_all(&temporary);
            release_snapshot(chunks, snapshot);
            let (snapshot, identity) = load_git_control_template(
                &final_root,
                repository_baseline,
                &layer.baseline_tree,
                object_format,
            )?;
            Ok((snapshot, false, identity.baseline_view_commit))
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            release_snapshot(chunks, snapshot);
            Err(error.into())
        }
    }
}

fn load_git_control_template(
    root: &Path,
    repository_baseline: &BaselineSnapshot,
    baseline_tree: &str,
    object_format: &str,
) -> Result<(BaselineSnapshot, GitControlTemplateIdentity), WorkspaceError> {
    if fs::read(root.join("COMPLETE")).ok().as_deref() != Some(b"greppy.git-control-template.v2\n")
        || !root.join("payload").is_dir()
    {
        return Err(WorkspaceError::Tampered {
            path: root.into(),
            detail: "Git control template is incomplete".into(),
        });
    }
    let baseline: BaselineSnapshot = serde_json::from_slice(&fs::read(root.join("baseline.json"))?)
        .map_err(|error| WorkspaceError::Tampered {
            path: root.join("baseline.json"),
            detail: error.to_string(),
        })?;
    let identity: GitControlTemplateIdentity =
        serde_json::from_slice(&fs::read(root.join("identity.json"))?).map_err(|error| {
            WorkspaceError::Tampered {
                path: root.join("identity.json"),
                detail: error.to_string(),
            }
        })?;
    if baseline.repository != root.join("namespace")
        || !baseline.base_commit.starts_with("virtual-empty:")
        || identity.schema != 2
        || identity.baseline_hash != repository_baseline.baseline_hash
        || identity.base_commit != repository_baseline.base_commit
        || identity.baseline_tree != baseline_tree
        || identity.object_format != object_format
        || !valid_object_id(&identity.baseline_view_commit, object_format)
    {
        return Err(WorkspaceError::Tampered {
            path: root.into(),
            detail: "Git control template identity is invalid".into(),
        });
    }
    Ok((baseline, identity))
}

fn git_workspace_id(run_id: &str) -> String {
    let digest = blake3::hash(run_id.as_bytes()).to_hex().to_string();
    format!("git-{}", &digest[..32])
}

fn private_git_storage_key(baseline_hash: &str) -> Result<&str, WorkspaceError> {
    const KEY_HEX_LEN: usize = 32;
    if baseline_hash.len() != 64 || !baseline_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(WorkspaceError::Unsupported(
            "private Git cache requires a canonical 256-bit hexadecimal baseline hash".into(),
        ));
    }
    Ok(&baseline_hash[..KEY_HEX_LEN])
}

fn private_git_temporary_path(parent: &Path) -> PathBuf {
    parent.join(format!(".t.{:x}.{:x}", std::process::id(), now_unix_ns()))
}

fn initialize_private_git(
    worktree: &Path,
    private_git_dir: &Path,
    baseline_tree: &str,
    expected_baseline_view_commit: &str,
) -> Result<(String, String, PathBuf), WorkspaceError> {
    if !private_git_dir.is_dir() {
        return Err(WorkspaceError::Tampered {
            path: private_git_dir.into(),
            detail: "provider did not expose the private Git CoW namespace".into(),
        });
    }
    let private_index = private_git_dir.join("index");
    if !private_index.is_file() {
        return Err(WorkspaceError::Tampered {
            path: private_index,
            detail: "private Git CoW namespace has no split index".into(),
        });
    }

    let baseline_view_commit =
        fs::read_to_string(private_git_dir.join("refs/heads/greppy-baseline"))?;
    let baseline_view_commit = baseline_view_commit.trim();
    let head = fs::read(private_git_dir.join("HEAD"))?;
    if baseline_view_commit != expected_baseline_view_commit
        || head != b"ref: refs/heads/greppy-baseline\n"
    {
        return Err(WorkspaceError::Tampered {
            path: private_git_dir.into(),
            detail: "private Git namespace does not match its immutable template".into(),
        });
    }
    fs::write(
        worktree.join(".git"),
        format!("gitdir: {}\n", private_git_dir.display()),
    )?;
    Ok((
        baseline_tree.into(),
        baseline_view_commit.into(),
        private_index,
    ))
}

#[derive(Debug, Serialize, Deserialize)]
struct GitControlTemplateIdentity {
    schema: u32,
    baseline_hash: String,
    base_commit: String,
    baseline_tree: String,
    baseline_view_commit: String,
    object_format: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SharedGitLayerManifest {
    schema: u32,
    baseline_hash: String,
    base_commit: String,
    baseline_tree: String,
    object_format: String,
}

struct SharedGitLayer {
    objects: PathBuf,
    index: PathBuf,
    shared_index: PathBuf,
    baseline_tree: String,
}

fn ensure_shared_git_layer(
    data_root: &Path,
    worktree: &Path,
    baseline: &BaselineSnapshot,
    chunks: &ChunkStore,
    object_format: &str,
    source_objects: &Path,
) -> Result<SharedGitLayer, WorkspaceError> {
    let layers = data_root.join("g").join("sl1");
    fs::create_dir_all(&layers)?;
    let final_root = layers.join(private_git_storage_key(&baseline.baseline_hash)?);
    if final_root.exists() {
        return open_shared_git_layer(&final_root, baseline, object_format);
    }

    let temporary = private_git_temporary_path(&layers);
    fs::create_dir(&temporary)?;
    let build = (|| -> Result<SharedGitLayerManifest, WorkspaceError> {
        let repository = temporary.join("repo");
        init_bare(&repository, object_format)?;
        let alternates = repository.join("objects/info/alternates");
        fs::create_dir_all(
            alternates
                .parent()
                .ok_or_else(|| io::Error::other("invalid shared alternates path"))?,
        )?;
        fs::write(&alternates, format!("{}\n", source_objects.display()))?;
        let indexes = temporary.join("indexes");
        fs::create_dir(&indexes)?;
        let seed_index = indexes.join("seed.index");
        git_private(
            &repository,
            worktree,
            Some(&seed_index),
            &["read-tree", &baseline.base_commit],
        )?;
        for entry in &baseline.entries {
            match entry.kind {
                EntryKind::Tombstone => {
                    git_private(
                        &repository,
                        worktree,
                        Some(&seed_index),
                        &["update-index", "--force-remove", "--", &entry.path],
                    )?;
                }
                EntryKind::File | EntryKind::Symlink => {
                    let bytes = baseline_bytes(chunks, entry)?;
                    let oid = hash_blob(&repository, worktree, &bytes)?;
                    let cache_info = format!("{:o},{oid},{}", entry.mode, entry.path);
                    git_private(
                        &repository,
                        worktree,
                        Some(&seed_index),
                        &["update-index", "--add", "--cacheinfo", &cache_info],
                    )?;
                }
            }
        }
        let baseline_tree = git_private(&repository, worktree, Some(&seed_index), &["write-tree"])?;
        git_private(
            &repository,
            worktree,
            Some(&seed_index),
            &["update-index", "--split-index"],
        )?;
        for entry in fs::read_dir(&repository)? {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with("sharedindex.") {
                fs::rename(entry.path(), indexes.join(name))?;
            }
        }
        let seed_bytes = fs::metadata(&seed_index)?.len();
        if seed_bytes > 512 * 1024 {
            return Err(WorkspaceError::Unsupported(format!(
                "private split-index seed is {seed_bytes} bytes; 0.3.4 requires at most 512 KiB"
            )));
        }
        Ok(SharedGitLayerManifest {
            schema: 1,
            baseline_hash: baseline.baseline_hash.clone(),
            base_commit: baseline.base_commit.clone(),
            baseline_tree,
            object_format: object_format.into(),
        })
    })();
    let manifest = match build {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
    };
    fs::write(
        temporary.join("layer.json"),
        serde_json::to_vec(&manifest).map_err(|error| io::Error::other(error.to_string()))?,
    )?;
    fs::write(temporary.join("COMPLETE"), b"greppy.shared-git-layer.v1\n")?;
    match fs::rename(&temporary, &final_root) {
        Ok(()) => {}
        Err(_error) if final_root.exists() => {
            let _ = fs::remove_dir_all(&temporary);
            open_shared_git_layer(&final_root, baseline, object_format)?;
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error.into());
        }
    }
    open_shared_git_layer(&final_root, baseline, object_format)
}

fn open_shared_git_layer(
    root: &Path,
    baseline: &BaselineSnapshot,
    object_format: &str,
) -> Result<SharedGitLayer, WorkspaceError> {
    if !root.join("COMPLETE").is_file() {
        return Err(WorkspaceError::Tampered {
            path: root.into(),
            detail: "shared Git layer is incomplete".into(),
        });
    }
    let manifest: SharedGitLayerManifest =
        serde_json::from_slice(&fs::read(root.join("layer.json"))?).map_err(|error| {
            WorkspaceError::Tampered {
                path: root.join("layer.json"),
                detail: error.to_string(),
            }
        })?;
    if !shared_git_layer_identity_matches(
        &manifest,
        &baseline.baseline_hash,
        &baseline.base_commit,
        object_format,
    ) {
        return Err(WorkspaceError::Tampered {
            path: root.into(),
            detail: "shared Git layer identity does not match the captured baseline".into(),
        });
    }
    let index = root.join("indexes/seed.index");
    let objects = root.join("repo/objects");
    if !index.is_file() || !objects.is_dir() {
        return Err(WorkspaceError::Tampered {
            path: root.into(),
            detail: "shared Git layer is missing index or objects".into(),
        });
    }
    let mut shared_indexes = fs::read_dir(root.join("indexes"))?
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("sharedindex."))
        })
        .collect::<Vec<_>>();
    shared_indexes.sort();
    if shared_indexes.len() != 1 || !shared_indexes[0].is_file() {
        return Err(WorkspaceError::Tampered {
            path: root.into(),
            detail: "shared Git layer must contain exactly one complete shared index".into(),
        });
    }
    Ok(SharedGitLayer {
        objects,
        index,
        shared_index: shared_indexes.remove(0),
        baseline_tree: manifest.baseline_tree,
    })
}

fn shared_git_layer_identity_matches(
    manifest: &SharedGitLayerManifest,
    baseline_hash: &str,
    base_commit: &str,
    object_format: &str,
) -> bool {
    manifest.schema == 1
        && manifest.baseline_hash == baseline_hash
        && manifest.base_commit == base_commit
        && manifest.object_format == object_format
}

fn init_bare(path: &Path, object_format: &str) -> Result<(), WorkspaceError> {
    let output = private_git_command()
        .args([
            "init",
            "--bare",
            "--quiet",
            "--template=",
            &format!("--object-format={object_format}"),
            path_text(path)?,
        ])
        .output()?;
    output_text("git init --bare private workspace repository", output)?;
    Ok(())
}

fn configure_git_control_template(git_dir: &Path) -> Result<(), WorkspaceError> {
    for (key, value) in [
        ("core.bare", "false"),
        ("core.autocrlf", "false"),
        ("core.symlinks", "true"),
    ] {
        let output = private_git_command()
            .args(["--git-dir", path_text(git_dir)?, "config", key, value])
            .output()?;
        output_text(&format!("git config {key}"), output)?;
    }
    Ok(())
}

fn commit_tree_in_git_dir(
    git_dir: &Path,
    tree: &str,
    parent: &str,
    message: &str,
) -> Result<String, WorkspaceError> {
    let output = private_git_command()
        .args([
            "--git-dir",
            path_text(git_dir)?,
            "commit-tree",
            tree,
            "-p",
            parent,
            "-m",
            message,
        ])
        .env("GIT_AUTHOR_NAME", "greppy agent")
        .env("GIT_AUTHOR_EMAIL", "agent@greppy.local")
        .env("GIT_AUTHOR_DATE", "2000-01-01T00:00:00Z")
        .env("GIT_COMMITTER_NAME", "greppy agent")
        .env("GIT_COMMITTER_EMAIL", "agent@greppy.local")
        .env("GIT_COMMITTER_DATE", "2000-01-01T00:00:00Z")
        .output()?;
    output_text("git commit-tree private baseline template", output)
}

fn valid_object_id(value: &str, object_format: &str) -> bool {
    let expected = match object_format {
        "sha1" => 40,
        "sha256" => 64,
        _ => return false,
    };
    value.len() == expected
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn baseline_bytes(chunks: &ChunkStore, entry: &BaselineEntry) -> Result<Vec<u8>, WorkspaceError> {
    let mut bytes = Vec::with_capacity(entry.size as usize);
    for chunk in &entry.chunks {
        bytes.extend_from_slice(&chunks.read(*chunk)?);
    }
    bytes.truncate(entry.size as usize);
    Ok(bytes)
}

fn hash_blob(
    private_git_dir: &Path,
    worktree: &Path,
    bytes: &[u8],
) -> Result<String, WorkspaceError> {
    let mut child = private_git_command()
        .args([
            "--git-dir",
            path_text(private_git_dir)?,
            "--work-tree",
            path_text(worktree)?,
            "hash-object",
            "-w",
            "--stdin",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("git hash-object stdin unavailable"))?
        .write_all(bytes)?;
    output_text("git hash-object -w --stdin", child.wait_with_output()?)
}

fn git_private(
    private_git_dir: &Path,
    worktree: &Path,
    index: Option<&Path>,
    args: &[&str],
) -> Result<String, WorkspaceError> {
    let mut command = private_git_command();
    command
        .args(["--git-dir", path_text(private_git_dir)?])
        .args(["--work-tree", path_text(worktree)?]);
    if let Some(index) = index {
        command.env("GIT_INDEX_FILE", index);
    }
    let output = command.args(args).output()?;
    output_text(
        &format!(
            "git --git-dir {} {}",
            private_git_dir.display(),
            args.join(" ")
        ),
        output,
    )
}

fn private_git_command() -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("git");
        command.args(["-c", "core.longpaths=true"]);
        command
    }
    #[cfg(not(windows))]
    {
        Command::new("git")
    }
}

fn commit_tree(
    worktree: &Path,
    tree: &str,
    parent: &str,
    message: &str,
) -> Result<String, WorkspaceError> {
    let output = Command::new("git")
        .args([
            "-C",
            path_text(worktree)?,
            "commit-tree",
            tree,
            "-p",
            parent,
            "-m",
            message,
        ])
        .env("GIT_AUTHOR_NAME", "greppy agent")
        .env("GIT_AUTHOR_EMAIL", "agent@greppy.local")
        .env("GIT_COMMITTER_NAME", "greppy agent")
        .env("GIT_COMMITTER_EMAIL", "agent@greppy.local")
        .output()?;
    output_text("git commit-tree", output)
}

fn git_ok(cwd: &Path, args: &[&str]) -> Result<String, WorkspaceError> {
    let output = Command::new("git")
        .args(["-C", path_text(cwd)?])
        .args(args)
        .output()?;
    output_text(
        &format!("git -C {} {}", cwd.display(), args.join(" ")),
        output,
    )
}

fn git_with_index(cwd: &Path, index: &Path, args: &[&str]) -> Result<String, WorkspaceError> {
    let output = Command::new("git")
        .args(["-C", path_text(cwd)?])
        .args(args)
        .env("GIT_INDEX_FILE", index)
        .output()?;
    output_text(
        &format!(
            "GIT_INDEX_FILE={} git -C {} {}",
            index.display(),
            cwd.display(),
            args.join(" ")
        ),
        output,
    )
}

fn filter_ignored_paths(
    worktree: &Path,
    index: &Path,
    paths: Vec<String>,
) -> Result<Vec<String>, WorkspaceError> {
    let paths = paths
        .into_iter()
        .filter(|path| path != ".git" && !path.starts_with(".git/"))
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return Ok(paths);
    }
    let mut child = Command::new("git")
        .args(["-C", path_text(worktree)?, "check-ignore", "-z", "--stdin"])
        .env("GIT_INDEX_FILE", index)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let mut input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("git check-ignore stdin unavailable"))?;
        for path in &paths {
            input.write_all(path.as_bytes())?;
            input.write_all(&[0])?;
        }
    }
    let output = child.wait_with_output()?;
    if !output.status.success() && output.status.code() != Some(1) {
        return Err(git_failed("git check-ignore -z --stdin", &output));
    }
    let ignored = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .collect::<std::collections::HashSet<_>>();
    Ok(paths
        .into_iter()
        .filter(|path| !ignored.contains(path))
        .collect())
}

fn git_bytes(cwd: &Path, args: &[&str]) -> Result<Vec<u8>, WorkspaceError> {
    let output = Command::new("git")
        .args(["-C", path_text(cwd)?])
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(git_failed(
            &format!("git -C {} {}", cwd.display(), args.join(" ")),
            &output,
        ))
    }
}

fn output_text(command: &str, output: Output) -> Result<String, WorkspaceError> {
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim_end().into())
    } else {
        Err(git_failed(command, &output))
    }
}

fn git_failed(command: &str, output: &Output) -> WorkspaceError {
    WorkspaceError::GitFailed {
        command: command.into(),
        stderr: String::from_utf8_lossy(&output.stderr).into(),
        status: output.status.code(),
    }
}

fn git_path(repo: &Path, name: &str) -> Result<PathBuf, WorkspaceError> {
    let path = PathBuf::from(git_ok(repo, &["rev-parse", "--git-path", name])?);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(repo.join(path))
    }
}

fn hash_optional_file(path: &Path) -> Result<Option<[u8; 32]>, WorkspaceError> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(Sha256::digest(bytes).into())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn release_snapshot(store: &greppy_workspace_core::ChunkStore, snapshot: BaselineSnapshot) {
    for chunk in snapshot
        .entries
        .into_iter()
        .flat_map(|entry| entry.chunks)
        .chain(snapshot.index_chunks)
    {
        let _ = store.unpin(chunk);
    }
}

fn wait_for_workspace(path: &Path) -> Result<(), WorkspaceError> {
    for _ in 0..100 {
        if path.is_dir() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(20));
    }
    Err(WorkspaceError::AdapterUnavailable(format!(
        "provider did not expose {} within two seconds",
        path.display()
    )))
}

fn wait_for_workspace_snapshot(
    path: &Path,
    entries: &[BaselineEntry],
) -> Result<(), WorkspaceError> {
    let witness = entries
        .iter()
        .find(|entry| !matches!(entry.kind, EntryKind::Tombstone));
    for _ in 0..100 {
        if workspace_snapshot_visible(path, entries) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(20));
    }
    let witness = witness.map_or_else(
        || "workspace root".to_string(),
        |entry| format!("baseline witness {}", entry.path),
    );
    Err(WorkspaceError::AdapterUnavailable(format!(
        "provider did not expose {witness} under {} within two seconds",
        path.display()
    )))
}

fn workspace_snapshot_visible(path: &Path, entries: &[BaselineEntry]) -> bool {
    if !path.is_dir() {
        return false;
    }
    entries
        .iter()
        .find(|entry| !matches!(entry.kind, EntryKind::Tombstone))
        .is_none_or(|entry| {
            let Ok(metadata) = fs::symlink_metadata(path.join(&entry.path)) else {
                return false;
            };
            match entry.kind {
                EntryKind::File => metadata.is_file() && metadata.len() == entry.size,
                EntryKind::Symlink => metadata.file_type().is_symlink(),
                EntryKind::Tombstone => true,
            }
        })
}

fn validate_run_id(value: &str) -> Result<(), WorkspaceError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(WorkspaceError::Unsupported(format!(
            "invalid agent run id {value:?}"
        )));
    }
    Ok(())
}

fn path_text(path: &Path) -> Result<&str, WorkspaceError> {
    path.to_str().ok_or_else(|| {
        WorkspaceError::Unsupported(format!("path is not UTF-8: {}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_workspace_core::{
        AdapterKind, ProviderCapabilities, ProviderManifest, ProviderState,
        PROVIDER_PROTOCOL_VERSION,
    };
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    static ENV_LOCK: Mutex<()> = Mutex::new(());
    const APPLY_CRASH_CHILD_TEST: &str = "workspace::tests::crash_child_applies_proposal";
    const PROPOSAL_CRASH_CHILD_TEST: &str = "workspace::tests::crash_child_publishes_proposal";

    #[test]
    fn crash_child_applies_proposal() {
        if std::env::var_os("GREPPY_AGENT_TEST_CRASH_POINT").is_none() {
            return;
        }
        let data = std::env::var_os("GREPPY_AGENT_TEST_CRASH_DATA").unwrap();
        let repository = std::env::var_os("GREPPY_AGENT_TEST_CRASH_REPOSITORY").unwrap();
        let ref_name = std::env::var("GREPPY_AGENT_TEST_CRASH_REF").unwrap();
        std::env::set_var("GREPPY_WORKSPACE_DIR", data);
        apply_proposal(Path::new(&repository), &ref_name).unwrap();
        panic!("apply crash point did not abort the child process");
    }

    fn abort_apply_child(data: &Path, repository: &Path, ref_name: &str) {
        let status = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(APPLY_CRASH_CHILD_TEST)
            .arg("--nocapture")
            .env("GREPPY_AGENT_TEST_CRASH_POINT", "apply-after-first-path")
            .env("GREPPY_AGENT_TEST_CRASH_DATA", data)
            .env("GREPPY_AGENT_TEST_CRASH_REPOSITORY", repository)
            .env("GREPPY_AGENT_TEST_CRASH_REF", ref_name)
            .status()
            .unwrap();
        assert!(!status.success(), "apply crash child exited cleanly");
    }

    #[test]
    fn crash_child_publishes_proposal() {
        if std::env::var_os("GREPPY_AGENT_TEST_CRASH_POINT").is_none() {
            return;
        }
        let core_root = std::env::var_os("GREPPY_AGENT_TEST_CRASH_CORE").unwrap();
        let journal: ProposalPublishJournal =
            serde_json::from_str(&std::env::var("GREPPY_AGENT_TEST_CRASH_JOURNAL").unwrap())
                .unwrap();
        let core = WorkspaceCore::open(core_root).unwrap();
        let workspace = core.open_workspace(&journal.workspace_id).unwrap();
        publish_proposal_transaction(
            &core,
            &workspace,
            &journal.repository,
            &journal.ref_name,
            &journal.baseline_ref,
            &journal.baseline_view_commit,
            &journal.baseline_tree,
            &journal.final_tree,
            &journal.proposal_commit,
            &journal.hardlink_groups,
        )
        .unwrap();
        panic!("proposal publication crash point did not abort the child process");
    }

    fn abort_proposal_publish_child(core_root: &Path, journal: &ProposalPublishJournal) {
        let status = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(PROPOSAL_CRASH_CHILD_TEST)
            .arg("--nocapture")
            .env(
                "GREPPY_AGENT_TEST_CRASH_POINT",
                "proposal-after-core-record",
            )
            .env("GREPPY_AGENT_TEST_CRASH_CORE", core_root)
            .env(
                "GREPPY_AGENT_TEST_CRASH_JOURNAL",
                serde_json::to_string(journal).unwrap(),
            )
            .status()
            .unwrap();
        assert!(
            !status.success(),
            "proposal publication crash child exited cleanly"
        );
    }

    fn json_journal_count(root: &Path) -> usize {
        fs::read_dir(root)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension().and_then(|value| value.to_str()) == Some("json")
            })
            .count()
    }

    #[test]
    fn proposal_publication_process_crash_rolls_back_then_commits_atomically() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let repository = temp.path().join("repository");
        fs::create_dir(&repository).unwrap();
        git(&repository, &["init", "-q"]);
        git(&repository, &["config", "user.email", "test@example.test"]);
        git(&repository, &["config", "user.name", "Test"]);
        fs::write(repository.join("tracked.txt"), b"base\n").unwrap();
        git(&repository, &["add", "tracked.txt"]);
        git(&repository, &["commit", "-qm", "base"]);
        let base_commit = git(&repository, &["rev-parse", "HEAD"]);
        let baseline_tree = git(&repository, &["rev-parse", "HEAD^{tree}"]);

        let core_root = temp.path().join("core");
        let core = WorkspaceCore::open(&core_root).unwrap();
        let baseline = capture_repository(&repository, core.chunks()).unwrap();
        let baseline_hash = baseline.baseline_hash.clone();
        let workspace = core.create_workspace("proposal-crash", baseline).unwrap();
        fs::write(repository.join("tracked.txt"), b"proposal\n").unwrap();
        git(&repository, &["add", "tracked.txt"]);
        let final_tree = git(&repository, &["write-tree"]);
        let proposal_commit = commit_tree(
            &repository,
            &final_tree,
            &base_commit,
            &proposal_commit_message("crash-safe proposal", &[]),
        )
        .unwrap();
        git(&repository, &["reset", "--hard", "-q", &base_commit]);

        let journal = ProposalPublishJournal {
            schema: 1,
            workspace_id: workspace.id().into(),
            ref_name: "refs/greppy/agent/proposal-crash".into(),
            baseline_ref: "refs/greppy/baselines/proposal-crash".into(),
            repository: repository.canonicalize().unwrap(),
            baseline_hash,
            baseline_view_commit: base_commit,
            baseline_tree,
            final_tree,
            proposal_commit,
            hardlink_groups: vec![],
        };
        drop(core);

        abort_proposal_publish_child(&core_root, &journal);
        let core = WorkspaceCore::open(&core_root).unwrap();
        assert!(core.has_proposal(&journal.ref_name).unwrap());
        assert_eq!(
            read_optional_commit_ref(&journal.repository, &journal.baseline_ref)
                .unwrap()
                .as_deref(),
            Some(journal.baseline_view_commit.as_str())
        );
        assert!(
            read_optional_commit_ref(&journal.repository, &journal.ref_name)
                .unwrap()
                .is_none()
        );
        assert_eq!(json_journal_count(&proposal_publish_journal_root(&core)), 1);

        let active_publish = core
            .try_repository_operation_lease(&journal.repository)
            .unwrap()
            .unwrap();
        assert!(matches!(
            recover_proposal_publish_journals(&core),
            Err(WorkspaceError::Conflict { .. })
        ));
        assert!(core.has_proposal(&journal.ref_name).unwrap());
        assert_eq!(json_journal_count(&proposal_publish_journal_root(&core)), 1);
        drop(active_publish);
        recover_proposal_publish_journals(&core).unwrap();
        assert!(!core.has_proposal(&journal.ref_name).unwrap());
        assert!(
            read_optional_commit_ref(&journal.repository, &journal.baseline_ref)
                .unwrap()
                .is_none()
        );
        assert_eq!(json_journal_count(&proposal_publish_journal_root(&core)), 0);

        let workspace = core.open_workspace(&journal.workspace_id).unwrap();
        publish_proposal_transaction(
            &core,
            &workspace,
            &journal.repository,
            &journal.ref_name,
            &journal.baseline_ref,
            &journal.baseline_view_commit,
            &journal.baseline_tree,
            &journal.final_tree,
            &journal.proposal_commit,
            &journal.hardlink_groups,
        )
        .unwrap();
        assert!(core.has_proposal(&journal.ref_name).unwrap());
        assert_eq!(
            read_optional_commit_ref(&journal.repository, &journal.ref_name)
                .unwrap()
                .as_deref(),
            Some(journal.proposal_commit.as_str())
        );
        assert_eq!(json_journal_count(&proposal_publish_journal_root(&core)), 0);
    }

    #[test]
    fn tracker_fences_are_internal_but_other_git_paths_are_mutations() {
        assert!(is_repository_tracker_fence(
            ".git/greppy-tracker-fence-123-456"
        ));
        assert!(!is_repository_tracker_fence(".git/index"));
        assert!(!is_repository_tracker_fence("src/lib.rs"));
    }

    #[test]
    fn workspace_snapshot_visibility_requires_a_real_baseline_entry() {
        let root = tempfile::tempdir().unwrap();
        let entries = vec![BaselineEntry {
            path: "src/lib.rs".into(),
            kind: EntryKind::File,
            mode: 0o644,
            size: 4,
            modified_unix_ns: 0,
            content_hash: String::new(),
            chunks: Vec::new(),
        }];
        assert!(!workspace_snapshot_visible(root.path(), &entries));
        fs::create_dir_all(root.path().join("src")).unwrap();
        fs::write(root.path().join("src/lib.rs"), b"rust").unwrap();
        assert!(workspace_snapshot_visible(root.path(), &entries));
    }

    fn git(path: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .args(["-C", path.to_str().unwrap()])
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim_end().into()
    }

    fn publish_provider(data: &Path, mount: &Path) {
        fs::create_dir_all(data).unwrap();
        fs::create_dir_all(mount).unwrap();
        let manifest = ProviderManifest {
            protocol_version: PROVIDER_PROTOCOL_VERSION,
            adapter_version: "0.3.4-test".into(),
            adapter_kind: AdapterKind::FsKit,
            state: ProviderState::Ready,
            instance_id: "test-provider".into(),
            data_root: data.into(),
            mount_root: mount.into(),
            heartbeat_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            capabilities: ProviderCapabilities {
                hard_links: true,
                symbolic_links: true,
                byte_range_locks: true,
                memory_maps: true,
                atomic_rename: true,
                case_preserving: true,
            },
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        fs::write(data.join("provider.json"), &bytes).unwrap();
        fs::write(mount.join(".greppy-provider.json"), bytes).unwrap();
    }

    struct TestProviderHeartbeat {
        stop: Option<mpsc::Sender<()>>,
        worker: Option<std::thread::JoinHandle<()>>,
    }

    impl Drop for TestProviderHeartbeat {
        fn drop(&mut self) {
            if let Some(stop) = self.stop.take() {
                let _ = stop.send(());
            }
            if let Some(worker) = self.worker.take() {
                worker.join().unwrap();
            }
        }
    }

    fn heartbeat_provider(data: &Path, mount: &Path) -> TestProviderHeartbeat {
        publish_provider(data, mount);
        let data = data.to_owned();
        let mount = mount.to_owned();
        let (stop, receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || loop {
            match receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => publish_provider(&data, &mount),
            }
        });
        TestProviderHeartbeat {
            stop: Some(stop),
            worker: Some(worker),
        }
    }

    fn copy_test_tree(source: &Path, destination: &Path) {
        fs::create_dir_all(destination).unwrap();
        for entry in fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let source_path = entry.path();
            let destination_path = destination.join(entry.file_name());
            let metadata = fs::symlink_metadata(&source_path).unwrap();
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                copy_test_tree(&source_path, &destination_path);
            } else if metadata.file_type().is_symlink() {
                #[cfg(unix)]
                std::os::unix::fs::symlink(fs::read_link(source_path).unwrap(), destination_path)
                    .unwrap();
                #[cfg(windows)]
                panic!("test Git control template unexpectedly contains a symbolic link");
            } else {
                fs::copy(source_path, destination_path).unwrap();
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn private_git_layer_supports_windows_long_paths() {
        let temp = tempfile::tempdir().unwrap();
        const TARGET_DATA_ROOT_LEN: usize = 130;
        let padding = TARGET_DATA_ROOT_LEN
            .checked_sub(temp.path().as_os_str().len() + 1)
            .expect("temporary path leaves no room for the Windows path-budget fixture");
        let long_root = temp.path().join("x".repeat(padding));
        assert_eq!(long_root.as_os_str().len(), TARGET_DATA_ROOT_LEN);
        let baseline_hash = "a".repeat(64);
        let layers = long_root.join("g/sl1");
        let temporary = private_git_temporary_path(&layers);
        let repository = temporary.join("repo");
        let worktree = long_root.join("workspace");
        let index = temporary.join("indexes/seed.index");
        let legacy_temporary = long_root.join("git-layers").join(format!(
            ".{}.tmp.{}.{}",
            baseline_hash,
            std::process::id(),
            now_unix_ns()
        ));
        let legacy_repository = legacy_temporary.join("repo");
        assert!(legacy_repository.as_os_str().len() >= 230);
        assert!(repository.as_os_str().len() < 180);
        fs::create_dir_all(repository.parent().unwrap()).unwrap();
        fs::create_dir_all(&worktree).unwrap();
        fs::create_dir_all(index.parent().unwrap()).unwrap();

        init_bare(&repository, "sha1").unwrap();
        configure_git_control_template(&repository).unwrap();
        git_private(
            &repository,
            &worktree,
            Some(&index),
            &["read-tree", "--empty"],
        )
        .unwrap();
        git_private(
            &repository,
            &worktree,
            Some(&index),
            &["update-index", "--split-index"],
        )
        .unwrap();
        assert!(index.is_file());
    }

    #[test]
    fn private_git_storage_keys_are_compact_but_full_identity_remains_authoritative() {
        let prefix = "0123456789abcdef0123456789abcdef";
        let first = format!("{prefix}{}", "0".repeat(32));
        let second = format!("{prefix}{}", "f".repeat(32));
        assert_eq!(private_git_storage_key(&first).unwrap(), prefix);
        assert_eq!(private_git_storage_key(&second).unwrap(), prefix);
        assert_ne!(first, second);
        assert!(private_git_storage_key("not-a-baseline-hash").is_err());

        let expected = SharedGitLayerManifest {
            schema: 1,
            baseline_hash: first.clone(),
            base_commit: "1".repeat(40),
            baseline_tree: "2".repeat(40),
            object_format: "sha1".into(),
        };
        let colliding = SharedGitLayerManifest {
            baseline_hash: second,
            ..expected
        };
        assert!(!shared_git_layer_identity_matches(
            &colliding,
            &first,
            &"1".repeat(40),
            "sha1"
        ));
    }

    #[test]
    fn dirty_baseline_proposal_applies_only_agent_delta_and_preserves_index() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["config", "user.email", "test@example.test"]);
        git(&repo, &["config", "user.name", "Test"]);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        fs::write(repo.join(".gitignore"), "cache/\n").unwrap();
        git(&repo, &["add", "tracked.txt", ".gitignore"]);
        git(&repo, &["commit", "-qm", "base"]);
        // Exercise the Windows/Git-for-Windows checkout conversion explicitly
        // on every host. Recovery must restore the captured dirty bytes, not
        // bytes rewritten by checkout-index through core.autocrlf.
        git(&repo, &["config", "core.autocrlf", "true"]);
        let base = git(&repo, &["rev-parse", "HEAD"]);
        fs::write(repo.join("tracked.txt"), "staged\n").unwrap();
        git(&repo, &["add", "tracked.txt"]);
        fs::write(repo.join("tracked.txt"), "dirty\n").unwrap();
        fs::write(repo.join("untracked.txt"), "user\n").unwrap();
        fs::write(repo.join("baseline-linked-a.txt"), "baseline-linked\n").unwrap();
        fs::hard_link(
            repo.join("baseline-linked-a.txt"),
            repo.join("baseline-linked-b.txt"),
        )
        .unwrap();

        let data = temp.path().join("provider-data");
        let mount = temp.path().join("provider-mount");
        let _provider_heartbeat = heartbeat_provider(&data, &mount);
        let tracker_core = Arc::new(WorkspaceCore::open(data.join("core")).unwrap());
        let tracked_repo = fs::canonicalize(&repo).unwrap();
        tracker_core
            .request_repository_tracker(&tracked_repo)
            .unwrap();
        tracker_core
            .activate_repository_tracker(&tracked_repo, 1)
            .unwrap();
        let tracker_for_fence = tracker_core.clone();
        let repo_for_fence = tracked_repo.clone();
        let fence_emulator = std::thread::spawn(move || {
            let git_dir = repo_for_fence.join(".git");
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            let mut observed = None::<String>;
            let mut completed = 0;
            loop {
                let current = fs::read_dir(&git_dir)
                    .unwrap()
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| entry.file_name().into_string().ok())
                    .find(|name| name.starts_with("greppy-tracker-fence-"));
                if observed.is_none() {
                    if let Some(name) = current {
                        let path = format!(".git/{name}");
                        tracker_for_fence
                            .record_repository_changes(
                                &repo_for_fence,
                                std::slice::from_ref(&path),
                                2,
                            )
                            .unwrap();
                        observed = Some(path);
                    }
                } else if let Some(observed_path) = observed.as_ref().filter(|_| current.is_none())
                {
                    tracker_for_fence
                        .record_repository_changes(
                            &repo_for_fence,
                            std::slice::from_ref(observed_path),
                            3,
                        )
                        .unwrap();
                    completed += 1;
                    if completed == 2 {
                        break;
                    }
                    observed = None;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "fence emulator timed out"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        let worktree = mount.join("workspaces/test-run");
        fs::create_dir_all(&worktree).unwrap();
        fs::write(worktree.join(".gitignore"), "cache/\n").unwrap();
        fs::write(worktree.join("tracked.txt"), "dirty\n").unwrap();
        fs::write(worktree.join("untracked.txt"), "user\n").unwrap();
        fs::write(worktree.join("baseline-linked-a.txt"), "baseline-linked\n").unwrap();
        fs::hard_link(
            worktree.join("baseline-linked-a.txt"),
            worktree.join("baseline-linked-b.txt"),
        )
        .unwrap();

        // The unit test uses an ordinary directory instead of a mounted
        // provider. Mirror the immutable Git-control template exactly once so
        // AgentWorkspace still exercises the paired namespace contract.
        let data_for_git = data.clone();
        let mount_for_git = mount.clone();
        let git_mount_emulator = std::thread::spawn(move || {
            let templates = data_for_git.join("g/ct3");
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Ok(entries) = fs::read_dir(&templates) {
                    if let Some(template) = entries
                        .filter_map(|entry| entry.ok())
                        .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
                        .map(|entry| entry.path())
                        .find(|path| path.join("COMPLETE").is_file())
                    {
                        let workspaces = mount_for_git.join("workspaces");
                        let git_id = git_workspace_id("test-run");
                        let staging = workspaces.join(format!(".{git_id}.publishing"));
                        let destination = workspaces.join(git_id);
                        copy_test_tree(&template.join("payload"), &staging);
                        fs::rename(staging, destination).unwrap();
                        break;
                    }
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "Git control namespace emulator timed out"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        });

        let previous = std::env::var_os("GREPPY_WORKSPACE_DIR");
        std::env::set_var("GREPPY_WORKSPACE_DIR", &data);
        let workspace = AgentWorkspace::create(&repo, "test-run").unwrap();
        fence_emulator.join().unwrap();
        git_mount_emulator.join().unwrap();
        assert_eq!(
            workspace.git_index_path(),
            workspace.linked_git_dir().join("index")
        );
        assert!(workspace.linked_git_dir().join("index").is_file());
        assert!(fs::metadata(workspace.git_index_path()).unwrap().len() <= 512 * 1024);
        assert!(fs::read_dir(workspace.linked_git_dir())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with("sharedindex.")));
        assert!(git(workspace.worktree_path(), &["status", "--porcelain"]).is_empty());
        assert_eq!(
            filter_ignored_paths(
                workspace.worktree_path(),
                workspace.git_index_path(),
                vec![".git".into(), ".git/config".into(), "tracked.txt".into()],
            )
            .unwrap(),
            ["tracked.txt"]
        );
        assert_eq!(
            fs::canonicalize(git(workspace.worktree_path(), &["rev-parse", "--git-dir"])).unwrap(),
            fs::canonicalize(workspace.linked_git_dir()).unwrap()
        );
        fs::write(workspace.worktree_path().join("tracked.txt"), "tool-edit\n").unwrap();
        assert_eq!(
            git(workspace.worktree_path(), &["status", "--porcelain"]),
            " M tracked.txt"
        );
        assert!(
            git(workspace.worktree_path(), &["diff", "--", "tracked.txt"]).contains("+tool-edit")
        );
        git(workspace.worktree_path(), &["add", "--", "tracked.txt"]);
        assert!(git(
            workspace.worktree_path(),
            &["diff", "--cached", "--", "tracked.txt"]
        )
        .contains("+tool-edit"));
        git(
            workspace.worktree_path(),
            &["reset", "--quiet", "HEAD", "--", "tracked.txt"],
        );
        fs::write(workspace.worktree_path().join("tracked.txt"), "dirty\n").unwrap();
        assert!(git(workspace.worktree_path(), &["status", "--porcelain"]).is_empty());
        git(
            workspace.worktree_path(),
            &["branch", "workspace-private-branch"],
        );
        assert!(!git(
            workspace.worktree_path(),
            &[
                "show-ref",
                "--verify",
                "refs/heads/workspace-private-branch"
            ]
        )
        .is_empty());
        git(
            workspace.worktree_path(),
            &["branch", "-D", "workspace-private-branch"],
        );
        git(
            workspace.worktree_path(),
            &["config", "user.email", "workspace@example.test"],
        );
        git(
            workspace.worktree_path(),
            &["config", "user.name", "Workspace Test"],
        );
        fs::write(
            workspace.worktree_path().join("commit.txt"),
            "private commit\n",
        )
        .unwrap();
        git(workspace.worktree_path(), &["add", "--", "commit.txt"]);
        git(
            workspace.worktree_path(),
            &["commit", "--quiet", "-m", "private workspace commit"],
        );
        assert_eq!(
            git(workspace.worktree_path(), &["rev-parse", "HEAD^1"]),
            workspace.baseline_view_commit
        );
        git(
            workspace.worktree_path(),
            &[
                "reset",
                "--hard",
                "--quiet",
                &workspace.baseline_view_commit,
            ],
        );
        assert!(!workspace.worktree_path().join("commit.txt").exists());
        assert!(git(workspace.worktree_path(), &["status", "--porcelain"]).is_empty());
        let agent_data = workspace.agent_data_root();
        assert!(agent_data.starts_with(&data));
        assert!(!agent_data.starts_with(&mount));
        let agent_scratch = workspace.agent_scratch_root();
        assert!(agent_scratch.starts_with(&data));
        assert!(!agent_scratch.starts_with(&mount));
        fs::create_dir_all(&agent_data).unwrap();
        fs::write(agent_data.join("graph.db"), b"private store").unwrap();
        fs::create_dir_all(&agent_scratch).unwrap();
        fs::write(agent_scratch.join("tool.tmp"), b"scratch").unwrap();
        assert!(git_with_index(
            workspace.worktree_path(),
            workspace.git_index_path(),
            &["status", "--porcelain"]
        )
        .unwrap()
        .is_empty());
        let baseline_link_a = workspace
            .core
            .metadata(&workspace.handle, "baseline-linked-a.txt")
            .unwrap()
            .unwrap();
        let baseline_link_b = workspace
            .core
            .metadata(&workspace.handle, "baseline-linked-b.txt")
            .unwrap()
            .unwrap();
        assert_eq!(baseline_link_a.inode, baseline_link_b.inode);
        assert_eq!(baseline_link_a.nlink, 2);
        fs::write(
            workspace.worktree_path().join("baseline-linked-a.txt"),
            "baseline-change\n",
        )
        .unwrap();
        workspace
            .core
            .write(
                &workspace.handle,
                "baseline-linked-a.txt",
                0,
                b"baseline-change\n",
            )
            .unwrap();
        assert_eq!(
            workspace
                .core
                .read(&workspace.handle, "baseline-linked-b.txt", 0, 64)
                .unwrap(),
            b"baseline-change\n"
        );
        let promoted_link_a = workspace
            .core
            .metadata(&workspace.handle, "baseline-linked-a.txt")
            .unwrap()
            .unwrap();
        let promoted_link_b = workspace
            .core
            .metadata(&workspace.handle, "baseline-linked-b.txt")
            .unwrap()
            .unwrap();
        assert_eq!(promoted_link_a.inode, promoted_link_b.inode);
        assert_eq!(promoted_link_a.nlink, 2);
        assert_eq!(promoted_link_b.nlink, 2);
        fs::write(workspace.worktree_path().join("tracked.txt"), "agent\n").unwrap();
        workspace
            .core
            .write(&workspace.handle, "tracked.txt", 0, b"agent\n")
            .unwrap();
        fs::write(workspace.worktree_path().join("linked-a.txt"), b"linked\n").unwrap();
        fs::hard_link(
            workspace.worktree_path().join("linked-a.txt"),
            workspace.worktree_path().join("linked-b.txt"),
        )
        .unwrap();
        workspace
            .core
            .create_file(&workspace.handle, "linked-a.txt", 0o100644)
            .unwrap();
        workspace
            .core
            .write(&workspace.handle, "linked-a.txt", 0, b"linked\n")
            .unwrap();
        workspace
            .core
            .hard_link(&workspace.handle, "linked-a.txt", "linked-b.txt")
            .unwrap();
        fs::create_dir(workspace.worktree_path().join("cache")).unwrap();
        fs::write(
            workspace.worktree_path().join("cache/output.bin"),
            b"ignored",
        )
        .unwrap();
        workspace
            .core
            .mkdir(&workspace.handle, "cache", 0o755)
            .unwrap();
        workspace
            .core
            .create_file(&workspace.handle, "cache/output.bin", 0o100644)
            .unwrap();
        workspace
            .core
            .write(&workspace.handle, "cache/output.bin", 0, b"ignored")
            .unwrap();
        let outcome = workspace.finish("agent result").unwrap();
        let (commit, ref_name, patch) = match outcome {
            RunOutcome::Proposal {
                commit,
                ref_name,
                patch,
                ..
            } => (commit, ref_name, patch),
            RunOutcome::Clean => panic!("expected proposal"),
        };
        assert_eq!(git(&repo, &["rev-parse", &format!("{commit}^1")]), base);
        assert!(patch.contains("-dirty"));
        assert!(patch.contains("+agent"));
        assert!(!patch.lines().any(|line| line == "-base"));
        assert!(git(&repo, &["ls-tree", "-r", &commit, "--", "cache"]).is_empty());

        let index = git_path(&repo, "index").unwrap();
        let index_before = fs::read(&index).unwrap();
        let recovery_core = WorkspaceCore::open(data.join("core")).unwrap();
        workspace.cleanup().unwrap();
        assert!(!agent_data.exists());
        assert!(!agent_scratch.exists());

        let proposal = recovery_core.proposal(&ref_name).unwrap();
        assert_eq!(
            git(
                &repo,
                &["rev-parse", &format!("{commit}:baseline-linked-a.txt")]
            ),
            git(
                &repo,
                &["rev-parse", &format!("{commit}:baseline-linked-b.txt")]
            )
        );
        assert_eq!(
            proposal.hardlink_groups,
            vec![
                vec![
                    String::from("baseline-linked-a.txt"),
                    String::from("baseline-linked-b.txt"),
                ],
                vec![String::from("linked-a.txt"), String::from("linked-b.txt"),],
            ]
        );
        let mut tampered_hardlinks = proposal.clone();
        tampered_hardlinks.hardlink_groups = vec![vec![
            String::from("linked-a.txt"),
            String::from("untracked.txt"),
        ]];
        assert!(matches!(
            validate_proposal_git_binding(&repo, &tampered_hardlinks),
            Err(WorkspaceError::Tampered { .. })
        ));
        let journal = ApplyJournal {
            schema: 1,
            ref_name: ref_name.clone(),
            repository: repo.canonicalize().unwrap(),
            baseline_hash: proposal.baseline_hash.clone(),
            baseline_tree: proposal.baseline_tree.clone(),
            affected_paths: apply_affected_paths(&repo, &proposal).unwrap(),
            modified_times: proposal
                .baseline
                .entries
                .iter()
                .filter(|entry| entry.kind != greppy_workspace_core::EntryKind::Tombstone)
                .map(|entry| (entry.path.clone(), entry.modified_unix_ns))
                .collect(),
        };
        let apply_journals = apply_journal_root(&recovery_core);
        fs::create_dir_all(&apply_journals).unwrap();
        let foreign_type = apply_journals.join("foreign.json");
        fs::create_dir(&foreign_type).unwrap();
        assert!(matches!(
            recover_apply_journals(&recovery_core),
            Err(WorkspaceError::Tampered { .. })
        ));
        fs::remove_dir(&foreign_type).unwrap();

        #[cfg(unix)]
        {
            let outside = temp.path().join("outside-apply-journal.json");
            fs::write(&outside, serde_json::to_vec(&journal).unwrap()).unwrap();
            let escaped = apply_journals.join("escaped.json");
            std::os::unix::fs::symlink(&outside, &escaped).unwrap();
            assert!(matches!(
                recover_apply_journals(&recovery_core),
                Err(WorkspaceError::Tampered { .. })
            ));
            assert_eq!(
                fs::read(&outside).unwrap(),
                serde_json::to_vec(&journal).unwrap()
            );
            fs::remove_file(&escaped).unwrap();
        }

        let unrelated = temp.path().join("unrelated");
        fs::create_dir(&unrelated).unwrap();
        let mut tampered_journal = journal.clone();
        tampered_journal.repository = unrelated.canonicalize().unwrap();
        let tampered_path = publish_apply_journal(&recovery_core, &tampered_journal).unwrap();
        assert!(matches!(
            recover_apply_journals(&recovery_core),
            Err(WorkspaceError::Tampered { .. })
        ));
        assert_eq!(fs::read(repo.join("tracked.txt")).unwrap(), b"dirty\n");
        assert_eq!(fs::read(&index).unwrap(), index_before);
        remove_apply_journal(&tampered_path).unwrap();

        let journal_path = publish_apply_journal(&recovery_core, &journal).unwrap();
        fs::write(repo.join("tracked.txt"), "partially applied\n").unwrap();
        let active_apply = recovery_core
            .try_repository_operation_lease(&repo)
            .unwrap()
            .unwrap();
        assert!(matches!(
            recover_apply_journals(&recovery_core),
            Err(WorkspaceError::Conflict { .. })
        ));
        assert!(journal_path.exists());
        assert_eq!(
            fs::read(repo.join("tracked.txt")).unwrap(),
            b"partially applied\n"
        );
        drop(active_apply);
        recover_apply_journals(&recovery_core).unwrap();
        assert!(!journal_path.exists());
        assert_eq!(fs::read(repo.join("tracked.txt")).unwrap(), b"dirty\n");
        assert_eq!(fs::read(&index).unwrap(), index_before);

        git(&repo, &["update-ref", &ref_name, &base]);
        assert!(matches!(
            apply_proposal(&repo, &ref_name),
            Err(WorkspaceError::Tampered { .. })
        ));
        assert_eq!(fs::read(repo.join("tracked.txt")).unwrap(), b"dirty\n");
        assert_eq!(fs::read(&index).unwrap(), index_before);
        git(&repo, &["update-ref", &ref_name, &commit]);

        let competing = recovery_core
            .try_repository_operation_lease(&repo)
            .unwrap()
            .unwrap();
        assert!(matches!(
            apply_proposal(&repo, &ref_name),
            Err(WorkspaceError::Conflict { .. })
        ));
        assert_eq!(fs::read(repo.join("tracked.txt")).unwrap(), b"dirty\n");
        assert_eq!(fs::read(&index).unwrap(), index_before);
        drop(competing);

        abort_apply_child(&data, &repo, &ref_name);
        assert_eq!(fs::read(&index).unwrap(), index_before);
        assert_eq!(
            fs::read_dir(recovery_core.root().join("apply-journals"))
                .unwrap()
                .filter_map(|entry| entry.ok())
                .filter(
                    |entry| entry.path().extension().and_then(|value| value.to_str())
                        == Some("json")
                )
                .count(),
            1
        );
        apply_proposal(&repo, &ref_name).unwrap();
        assert_eq!(fs::read(repo.join("tracked.txt")).unwrap(), b"agent\n");
        fs::write(repo.join("baseline-linked-b.txt"), b"baseline-same-inode\n").unwrap();
        assert_eq!(
            fs::read(repo.join("baseline-linked-a.txt")).unwrap(),
            b"baseline-same-inode\n"
        );
        fs::write(repo.join("linked-b.txt"), b"same-inode\n").unwrap();
        assert_eq!(
            fs::read(repo.join("linked-a.txt")).unwrap(),
            b"same-inode\n"
        );
        assert_eq!(fs::read(&index).unwrap(), index_before);
        match previous {
            Some(value) => std::env::set_var("GREPPY_WORKSPACE_DIR", value),
            None => std::env::remove_var("GREPPY_WORKSPACE_DIR"),
        }
    }

    #[test]
    fn missing_provider_fails_closed() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("GREPPY_WORKSPACE_DIR");
        std::env::set_var("GREPPY_WORKSPACE_DIR", temp.path());
        let error = AgentWorkspace::create(temp.path(), "missing-provider").unwrap_err();
        assert!(matches!(error, WorkspaceError::AdapterUnavailable(_)));
        match previous {
            Some(value) => std::env::set_var("GREPPY_WORKSPACE_DIR", value),
            None => std::env::remove_var("GREPPY_WORKSPACE_DIR"),
        }
    }
}
