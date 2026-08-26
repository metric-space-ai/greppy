//! Portable, provider-mounted CoW workspace lifecycle.
//!
//! No native checkout or filesystem snapshot fallback exists here. Creation
//! verifies the installed adapter and its mounted identity before capturing the
//! repository and before any model request can be made.

use greppy_workspace_core::{
    capture_repository, BaselineEntry, BaselineSnapshot, ChunkStore, EntryKind,
    ProviderInstallation, WorkspaceCore, WorkspaceHandle,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct AgentWorkspace {
    repo_root: PathBuf,
    worktree: PathBuf,
    private_git_dir: PathBuf,
    private_index: PathBuf,
    run_id: String,
    base_commit: String,
    baseline_hash: String,
    baseline_tree: String,
    baseline_view_commit: String,
    provider_instance: String,
    data_root: PathBuf,
    core: WorkspaceCore,
    handle: WorkspaceHandle,
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
        let data_root = workspace_data_root()?;
        let provider = ProviderInstallation::require_healthy(&data_root)?;
        provider.doctor_io(&format!("startup-{run_id}"))?;
        let provider_instance = provider.manifest().instance_id.clone();
        let core = WorkspaceCore::open(data_root.join("core"))?;
        recover_apply_journals(&core)?;
        let baseline = capture_repository(repo_root, core.chunks())?;
        let repo_root = baseline.repository.clone();
        let base_commit = baseline.base_commit.clone();
        let baseline_hash = baseline.baseline_hash.clone();
        let baseline_for_git = baseline.clone();
        let handle = core.create_workspace(run_id, baseline)?;
        let worktree = provider.workspace_path(run_id)?;
        if let Err(error) = wait_for_workspace(&worktree) {
            let _ = core.remove_workspace(handle);
            return Err(error);
        }
        let private_git_dir = data_root.join("private-git").join(run_id);
        let initialized = initialize_private_git(
            &repo_root,
            &worktree,
            &private_git_dir,
            &data_root,
            &baseline_for_git,
            core.chunks(),
        );
        let (baseline_tree, baseline_view_commit, private_index) = match initialized {
            Ok(result) => result,
            Err(error) => {
                let _ = core.remove_workspace(handle);
                let _ = fs::remove_dir_all(&private_git_dir);
                return Err(error);
            }
        };
        Ok(Self {
            repo_root,
            worktree,
            private_git_dir,
            private_index,
            run_id: run_id.into(),
            base_commit,
            baseline_hash,
            baseline_tree,
            baseline_view_commit,
            provider_instance,
            data_root,
            core,
            handle,
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
        self.core.keep(&self.handle)?;
        Ok(())
    }

    pub fn finish(&self, message: &str) -> Result<RunOutcome, WorkspaceError> {
        self.verify_identity()?;
        let changed_paths = filter_ignored_paths(
            &self.worktree,
            &self.private_index,
            self.core.changed_paths(&self.handle)?,
        )?;
        if !changed_paths.is_empty() {
            let mut arguments = vec!["add", "-A", "--"];
            arguments.extend(changed_paths.iter().map(String::as_str));
            git_with_index(&self.worktree, &self.private_index, &arguments)?;
        }
        let final_tree = git_with_index(&self.worktree, &self.private_index, &["write-tree"])?;
        if final_tree == self.baseline_tree {
            return Ok(RunOutcome::Clean);
        }
        let commit = commit_tree(&self.worktree, &final_tree, &self.base_commit, message)?;
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
        git_ok(
            &self.repo_root,
            &["update-ref", &baseline_ref, &self.baseline_view_commit],
        )?;
        self.core.preserve_proposal(
            &self.handle,
            &ref_name,
            &self.baseline_tree,
            &final_tree,
            &commit,
        )?;
        if let Err(error) = git_ok(&self.repo_root, &["update-ref", &ref_name, &commit]) {
            let _ = self.core.remove_proposal(&ref_name);
            let _ = git_ok(&self.repo_root, &["update-ref", "-d", &baseline_ref]);
            return Err(error);
        }
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
        match fs::remove_dir_all(&self.private_git_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        match fs::remove_file(&self.private_index) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        self.core.remove_workspace(self.handle)?;
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
        if provider.workspace_path(&self.run_id)? != self.worktree || !self.private_git_dir.is_dir()
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
        Ok(())
    }
}

/// Apply a persisted proposal after its agent workspace has been cleaned up.
pub fn apply_proposal(target_checkout: &Path, ref_name: &str) -> Result<(), WorkspaceError> {
    let data_root = workspace_data_root()?;
    let core = WorkspaceCore::open(data_root.join("core"))?;
    recover_apply_journals(&core)?;
    apply_from_core(&core, target_checkout, ref_name, None)
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
    let affected_paths = changed_paths(
        &canonical_target,
        &proposal.baseline_tree,
        &proposal.final_tree,
    )?;
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
    if let Err(error) = journal.affected_paths.iter().try_for_each(|path| {
        materialize_git_tree_entry(&canonical_target, &proposal.final_tree, path)
    }) {
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
        let bytes = fs::read(&path)?;
        let journal: ApplyJournal =
            serde_json::from_slice(&bytes).map_err(|error| WorkspaceError::Tampered {
                path: path.clone(),
                detail: format!("apply recovery journal is invalid: {error}"),
            })?;
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
    if proposal.baseline_hash != journal.baseline_hash
        || proposal.baseline_tree != journal.baseline_tree
    {
        return Err(WorkspaceError::Tampered {
            path: journal_path.to_path_buf(),
            detail: "apply recovery journal does not match its pinned proposal".into(),
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

fn home_dir() -> Result<PathBuf, WorkspaceError> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| WorkspaceError::AdapterUnavailable("home directory is unavailable".into()))
}

fn initialize_private_git(
    repo_root: &Path,
    worktree: &Path,
    private_git_dir: &Path,
    data_root: &Path,
    baseline: &BaselineSnapshot,
    chunks: &ChunkStore,
) -> Result<(String, String, PathBuf), WorkspaceError> {
    if private_git_dir.exists() {
        return Err(WorkspaceError::Tampered {
            path: private_git_dir.into(),
            detail: "private Git state already exists for this run id".into(),
        });
    }
    fs::create_dir_all(
        private_git_dir
            .parent()
            .ok_or_else(|| io::Error::other("invalid private Git path"))?,
    )?;
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

    init_bare(private_git_dir, &object_format)?;
    let alternates = private_git_dir.join("objects/info/alternates");
    fs::create_dir_all(
        alternates
            .parent()
            .ok_or_else(|| io::Error::other("invalid alternates path"))?,
    )?;
    fs::write(
        &alternates,
        format!("{}\n{}\n", layer.objects.display(), objects.display()),
    )?;
    git_private(
        private_git_dir,
        worktree,
        None,
        &["config", "core.bare", "false"],
    )?;
    git_private(
        private_git_dir,
        worktree,
        None,
        &["config", "core.worktree", path_text(worktree)?],
    )?;
    git_private(
        private_git_dir,
        worktree,
        None,
        &["config", "core.autocrlf", "false"],
    )?;
    git_private(
        private_git_dir,
        worktree,
        None,
        &["config", "core.symlinks", "true"],
    )?;
    fs::write(
        worktree.join(".git"),
        format!("gitdir: {}\n", private_git_dir.display()),
    )?;

    let private_index = layer
        .index
        .parent()
        .ok_or_else(|| io::Error::other("shared Git index has no parent"))?
        .join(format!(
            "workspace-{}.index",
            private_git_dir
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| WorkspaceError::Unsupported(
                    "private Git run id is not valid UTF-8".into()
                ))?
        ));
    if private_index.exists() {
        return Err(WorkspaceError::Tampered {
            path: private_index,
            detail: "private split index already exists".into(),
        });
    }
    fs::copy(&layer.index, &private_index)?;
    let baseline_tree = layer.baseline_tree;
    let baseline_view_commit = commit_tree(
        worktree,
        &baseline_tree,
        &baseline.base_commit,
        "greppy private baseline view",
    )?;
    git_ok(
        worktree,
        &[
            "update-ref",
            "refs/heads/greppy-baseline",
            &baseline_view_commit,
        ],
    )?;
    git_ok(
        worktree,
        &["symbolic-ref", "HEAD", "refs/heads/greppy-baseline"],
    )?;
    Ok((baseline_tree, baseline_view_commit, private_index))
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
    let layers = data_root.join("git-layers");
    fs::create_dir_all(&layers)?;
    let final_root = layers.join(&baseline.baseline_hash);
    if final_root.exists() {
        return open_shared_git_layer(&final_root, baseline, object_format);
    }

    let temporary = layers.join(format!(
        ".{}.tmp.{}.{}",
        baseline.baseline_hash,
        std::process::id(),
        now_unix_ns()
    ));
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
    if manifest.schema != 1
        || manifest.baseline_hash != baseline.baseline_hash
        || manifest.base_commit != baseline.base_commit
        || manifest.object_format != object_format
    {
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
    Ok(SharedGitLayer {
        objects,
        index,
        baseline_tree: manifest.baseline_tree,
    })
}

fn init_bare(path: &Path, object_format: &str) -> Result<(), WorkspaceError> {
    let output = Command::new("git")
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
    let mut child = Command::new("git")
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
    let mut command = Command::new("git");
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
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn git(path: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
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

        let data = temp.path().join("provider-data");
        let mount = temp.path().join("provider-mount");
        publish_provider(&data, &mount);
        let worktree = mount.join("workspaces/test-run");
        fs::create_dir_all(&worktree).unwrap();
        fs::write(worktree.join(".gitignore"), "cache/\n").unwrap();
        fs::write(worktree.join("tracked.txt"), "dirty\n").unwrap();
        fs::write(worktree.join("untracked.txt"), "user\n").unwrap();

        let previous = std::env::var_os("GREPPY_WORKSPACE_DIR");
        std::env::set_var("GREPPY_WORKSPACE_DIR", &data);
        let workspace = AgentWorkspace::create(&repo, "test-run").unwrap();
        assert!(!workspace.linked_git_dir().join("index").exists());
        assert!(fs::metadata(workspace.git_index_path()).unwrap().len() <= 512 * 1024);
        assert!(fs::read_dir(workspace.git_index_path().parent().unwrap())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.file_name().to_string_lossy().starts_with("sharedindex.")));
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
        fs::write(workspace.worktree_path().join("tracked.txt"), "agent\n").unwrap();
        workspace
            .core
            .write(&workspace.handle, "tracked.txt", 0, b"agent\n")
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
        assert!(!patch.contains("-base"));
        assert!(git(&repo, &["ls-tree", "-r", &commit, "--", "cache"]).is_empty());

        let index = git_path(&repo, "index").unwrap();
        let index_before = fs::read(&index).unwrap();
        let recovery_core = WorkspaceCore::open(data.join("core")).unwrap();
        workspace.cleanup().unwrap();
        assert!(!agent_data.exists());
        assert!(!agent_scratch.exists());

        let proposal = recovery_core.proposal(&ref_name).unwrap();
        let journal = ApplyJournal {
            schema: 1,
            ref_name: ref_name.clone(),
            repository: repo.canonicalize().unwrap(),
            baseline_hash: proposal.baseline_hash.clone(),
            baseline_tree: proposal.baseline_tree.clone(),
            affected_paths: changed_paths(&repo, &proposal.baseline_tree, &proposal.final_tree)
                .unwrap(),
            modified_times: proposal
                .baseline
                .entries
                .iter()
                .filter(|entry| entry.kind != greppy_workspace_core::EntryKind::Tombstone)
                .map(|entry| (entry.path.clone(), entry.modified_unix_ns))
                .collect(),
        };
        let journal_path = publish_apply_journal(&recovery_core, &journal).unwrap();
        fs::write(repo.join("tracked.txt"), "partially applied\n").unwrap();
        recover_apply_journals(&recovery_core).unwrap();
        assert!(!journal_path.exists());
        assert_eq!(fs::read(repo.join("tracked.txt")).unwrap(), b"dirty\n");
        assert_eq!(fs::read(&index).unwrap(), index_before);

        apply_proposal(&repo, &ref_name).unwrap();
        assert_eq!(fs::read(repo.join("tracked.txt")).unwrap(), b"agent\n");
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
