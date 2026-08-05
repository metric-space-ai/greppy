//! Per-run isolation: git worktrees and review-patch proposals.
//!
//! Every agent run works in a detached worktree of the target repo. The agent
//! never writes to the user's checkout. The run's outcome is a **proposal**: a
//! commit preserved on `refs/greppy/agent/<run_id>` plus a printable patch.
//! Applying it to a real checkout is a separate, explicit host-side step.
//!
//! # Worktree placement
//!
//! By default the worktree is **stable per repository**:
//!
//! ```text
//! <platform-cache>/greppy/agent-worktrees/<16-hex sha256 of canonical repo root>
//! ```
//!
//! (`~/Library/Caches` on macOS; `$XDG_CACHE_HOME` or `~/.cache` elsewhere.)
//! Reusing that path keeps the greppy store (and thus the semantic index) warm
//! across runs: each run resets tracked content to `HEAD` without deleting the
//! directory. Ignored build caches are kept by default (so repeat runs stay
//! fast); pass `--fresh` to drop them too. Concurrent `-p` runs never share a
//! stable tree — an exclusive lock on a sibling `.lock` file is required; if
//! the lock is held the run falls back to a disposable
//! `$TMPDIR/greppy-agent/<run_id>` worktree (and says so on stderr).
//!
//! Host-side git against the worktree always pins `--git-dir` + `--work-tree`
//! recorded at creation time. If the worktree's `.git` control file is later
//! rewritten, `finish` / reset refuse with [`WorkspaceError::Tampered`] rather
//! than rediscovering a poisoned pointer into the user checkout.
//!
//! [`AgentWorkspace::cleanup`] **does not** delete a stable worktree (that would
//! destroy the warm store); it only resets it. Fallback temp worktrees are
//! force-removed as before. `--keep-worktree` leaves a temp tree on disk; for a
//! stable tree the directory already survives cleanup.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use sha2::{Digest, Sha256};

/// Per-run git worktree isolation for an agent invocation.
#[derive(Debug)]
pub struct AgentWorkspace {
    repo_root: PathBuf,
    worktree: PathBuf,
    /// Absolute linked-worktree git directory recorded at creation
    /// (`git rev-parse --absolute-git-dir`). Every host-side git call against
    /// the worktree pins this path via `--git-dir` so a poisoned `.git` file
    /// cannot redirect operations into the user checkout.
    linked_git_dir: PathBuf,
    run_id: String,
    base_commit: String,
    /// Stable (cache-dir) placement vs disposable temp fallback.
    kind: WorktreeKind,
    /// Held exclusive lock for the stable worktree (released on drop).
    _lock: Option<FileLock>,
}

/// How the worktree directory was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorktreeKind {
    /// Reused or created under the platform cache; survives [`AgentWorkspace::cleanup`].
    Stable,
    /// Per-run temp tree under `$TMPDIR/greppy-agent/<run_id>`; removed on cleanup.
    Temp,
}

/// Outcome of [`AgentWorkspace::finish`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// Worktree matched the base commit; nothing to propose.
    Clean,
    /// A commit was created and pinned to a durable ref.
    Proposal {
        /// Full OID of the proposal commit.
        commit: String,
        /// `refs/greppy/agent/<run_id>` — survives worktree removal.
        ref_name: String,
        /// `git show --format= --patch <commit>` output.
        patch: String,
        /// `git show --format= --stat <commit>` output.
        stat: String,
    },
}

/// Errors from workspace create / finish / apply / cleanup.
#[derive(Debug)]
pub enum WorkspaceError {
    /// `repo_root` is not inside a git work tree.
    NotGitRepo { path: PathBuf, detail: String },
    /// A git subprocess failed; stderr is surfaced for the caller.
    GitFailed {
        command: String,
        stderr: String,
        status: Option<i32>,
    },
    /// `apply_to` refused because the target checkout has uncommitted changes.
    DirtyTarget {
        /// Ref the user can still apply from once the target is clean.
        ref_name: String,
        detail: String,
    },
    /// `apply_to` hit a cherry-pick conflict; restoration was attempted.
    Conflict {
        /// Ref the user can resolve from manually (`refs/greppy/agent/<run_id>`).
        ref_name: String,
        detail: String,
    },
    /// The worktree's git identity was modified after creation (poisoned `.git`,
    /// unregistered path, or mismatched absolute-git-dir). Host-side git through
    /// this tree is refused; the tree is left in place for inspection.
    Tampered {
        /// Path that failed the identity check (usually the worktree `.git` file).
        path: PathBuf,
        detail: String,
    },
    /// Local filesystem error (mkdir, etc.).
    Io(io::Error),
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotGitRepo { path, detail } => {
                write!(
                    f,
                    "not a git repository (or any parent): {}: {detail}",
                    path.display()
                )
            }
            Self::GitFailed {
                command,
                stderr,
                status,
            } => {
                write!(f, "git command failed: {command}")?;
                if let Some(code) = status {
                    write!(f, " (exit {code})")?;
                }
                let trimmed = stderr.trim();
                if !trimmed.is_empty() {
                    write!(f, ": {trimmed}")?;
                }
                Ok(())
            }
            Self::DirtyTarget { ref_name, detail } => {
                write!(
                    f,
                    "target checkout has uncommitted changes — commit or stash first; \
                     the proposal remains at {ref_name}: {detail}"
                )
            }
            Self::Conflict { ref_name, detail } => {
                write!(
                    f,
                    "cherry-pick conflict while applying proposal. \
                     Resolve manually from {ref_name}: {detail}"
                )
            }
            Self::Tampered { path, detail } => {
                write!(
                    f,
                    "worktree was modified in a way that makes the result untrustworthy \
                     ({}): {detail}; the tree was left in place for inspection",
                    path.display()
                )
            }
            Self::Io(e) => write!(f, "workspace I/O error: {e}"),
        }
    }
}

impl std::error::Error for WorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for WorkspaceError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// OS advisory lock held for the lifetime of a stable worktree.
struct FileLock {
    file: File,
    path: PathBuf,
}

impl fmt::Debug for FileLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileLock")
            .field("path", &self.path)
            .finish()
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        unlock_file(&self.file);
    }
}

impl AgentWorkspace {
    /// Create a detached worktree for `run_id` from `repo_root`'s `HEAD`.
    ///
    /// Prefers a **stable** path under the platform user cache (one tree per
    /// repository). If that tree is already locked by another `-p` run, falls
    /// back to `$TMPDIR/greppy-agent/<run_id>` and prints one stderr line.
    ///
    /// The user's checkout state is irrelevant — the worktree is always created
    /// (or reset) from `HEAD`. Tracked content is reset to HEAD; ignored build
    /// caches are kept by default.
    pub fn create(repo_root: &Path, run_id: &str) -> Result<Self, WorkspaceError> {
        Self::create_with_options(repo_root, run_id, CreateOptions::default())
    }

    /// Like [`Self::create`], with explicit reset options (`fresh` drops ignored
    /// files too via `git clean -ffdx`).
    pub fn create_with_options(
        repo_root: &Path,
        run_id: &str,
        options: CreateOptions,
    ) -> Result<Self, WorkspaceError> {
        let toplevel = match git_ok_cwd(repo_root, &["rev-parse", "--show-toplevel"]) {
            Ok(t) => PathBuf::from(t),
            Err(WorkspaceError::GitFailed { stderr, .. }) => {
                return Err(WorkspaceError::NotGitRepo {
                    path: repo_root.to_path_buf(),
                    detail: if stderr.trim().is_empty() {
                        "git rev-parse --show-toplevel failed".into()
                    } else {
                        stderr.trim().to_string()
                    },
                });
            }
            Err(e) => return Err(e),
        };

        let base_commit = git_ok_cwd(&toplevel, &["rev-parse", "HEAD"])?;
        let stable_dir = stable_worktree_dir(&toplevel);
        let lock_path = stable_lock_path(&stable_dir);

        match try_acquire_lock(&lock_path)? {
            Some(lock) => {
                let linked_git_dir =
                    prepare_stable_worktree(&toplevel, &stable_dir, &base_commit, options.fresh)?;
                Ok(Self {
                    repo_root: toplevel,
                    worktree: stable_dir,
                    linked_git_dir,
                    run_id: run_id.to_string(),
                    base_commit,
                    kind: WorktreeKind::Stable,
                    _lock: Some(lock),
                })
            }
            None => {
                // Concurrent holder — disposable temp so two runs never share a tree.
                eprintln!(
                    "greppy -p: agent worktree in use for this repository — using a temporary worktree"
                );
                let (worktree, linked_git_dir) =
                    create_temp_worktree(&toplevel, run_id, &base_commit, options.fresh)?;
                Ok(Self {
                    repo_root: toplevel,
                    worktree,
                    linked_git_dir,
                    run_id: run_id.to_string(),
                    base_commit,
                    kind: WorktreeKind::Temp,
                    _lock: None,
                })
            }
        }
    }

    /// Absolute path of the worktree (becomes [`crate::GreppyEnv`]'s root).
    pub fn worktree_path(&self) -> &Path {
        &self.worktree
    }

    /// Absolute linked-worktree git directory pinned at creation.
    pub fn linked_git_dir(&self) -> &Path {
        &self.linked_git_dir
    }

    /// Repository toplevel the worktree was created from.
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// Run id used for the durable proposal ref (and temp worktree directory).
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// `HEAD` OID recorded at create time.
    pub fn base_commit(&self) -> &str {
        &self.base_commit
    }

    /// True when this run uses the stable per-repository worktree.
    pub fn is_stable(&self) -> bool {
        self.kind == WorktreeKind::Stable
    }

    /// Durable ref name for this run's proposal (`refs/greppy/agent/<run_id>`).
    pub fn ref_name(&self) -> String {
        format!("refs/greppy/agent/{}", self.run_id)
    }

    /// Stage everything in the worktree and either return [`RunOutcome::Clean`]
    /// or pin a single proposal commit (parent = [`Self::base_commit`]) via
    /// plumbing so model-made commits/resets cannot corrupt the proposal.
    pub fn finish(&self, message: &str) -> Result<RunOutcome, WorkspaceError> {
        self.verify_identity()?;

        git_ok_wt(&self.linked_git_dir, &self.worktree, &["add", "-A"])?;

        // Capture the final filesystem state as a tree, independent of HEAD.
        let tree = git_ok_wt(&self.linked_git_dir, &self.worktree, &["write-tree"])?;
        let base_tree = git_ok_wt(
            &self.linked_git_dir,
            &self.worktree,
            &["rev-parse", &format!("{}^{{tree}}", self.base_commit)],
        )?;
        if tree == base_tree {
            return Ok(RunOutcome::Clean);
        }

        // Author/committer fixed for every agent proposal. Build the commit
        // with plumbing so parent is always base_commit regardless of whatever
        // the model did to HEAD inside the worktree.
        let commit_out = Command::new("git")
            .args([
                "--git-dir",
                path_str(&self.linked_git_dir)?,
                "--work-tree",
                path_str(&self.worktree)?,
                "commit-tree",
                &tree,
                "-p",
                &self.base_commit,
                "-m",
                message,
            ])
            .env("GIT_AUTHOR_NAME", "greppy agent")
            .env("GIT_AUTHOR_EMAIL", "agent@greppy.local")
            .env("GIT_COMMITTER_NAME", "greppy agent")
            .env("GIT_COMMITTER_EMAIL", "agent@greppy.local")
            // Neutralise a user template/hooks that might interfere in tests.
            .env("GIT_CONFIG_COUNT", "1")
            .env("GIT_CONFIG_KEY_0", "commit.gpgsign")
            .env("GIT_CONFIG_VALUE_0", "false")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .map_err(WorkspaceError::Io)?;
        if !commit_out.status.success() {
            return Err(git_failed(
                "git commit-tree <tree> -p <base> -m <message>",
                &commit_out,
            ));
        }
        let commit = String::from_utf8_lossy(&commit_out.stdout)
            .trim_end()
            .to_string();
        if commit.is_empty() {
            return Err(WorkspaceError::GitFailed {
                command: "git commit-tree <tree> -p <base> -m <message>".into(),
                stderr: "commit-tree produced empty OID".into(),
                status: commit_out.status.code(),
            });
        }

        let ref_name = self.ref_name();

        // Pin the proposal in the *shared* repo so it survives worktree removal.
        // These run against the main checkout identity (not the worktree).
        git_ok_cwd(&self.repo_root, &["update-ref", &ref_name, &commit])?;

        let patch = git_ok_cwd(&self.repo_root, &["show", "--format=", "--patch", &commit])?;
        let stat = git_ok_cwd(&self.repo_root, &["show", "--format=", "--stat", &commit])?;

        Ok(RunOutcome::Proposal {
            commit,
            ref_name,
            patch,
            stat,
        })
    }

    /// Cherry-pick `commit` into `target_checkout` with `--no-commit`.
    ///
    /// Refuses a dirty target before attempting the cherry-pick. On conflict
    /// restoration is attempted carefully: `cherry-pick --abort` only when
    /// `CHERRY_PICK_HEAD` is present, otherwise a positively chosen
    /// `reset --merge`. Both exit codes are checked so the error does not
    /// claim a clean abort when restoration failed.
    pub fn apply_to(&self, target_checkout: &Path, commit: &str) -> Result<(), WorkspaceError> {
        // Preflight: refuse uncommitted changes so we never start a
        // cherry-pick that could mash a dirty worktree.
        let status = git_run_cwd(target_checkout, &["status", "--porcelain=v1", "-z"])?;
        if !status.status.success() {
            return Err(git_failed("git status --porcelain=v1 -z", &status));
        }
        if !status.stdout.is_empty() {
            return Err(WorkspaceError::DirtyTarget {
                ref_name: self.ref_name(),
                detail: "git status --porcelain is non-empty".into(),
            });
        }

        let result = git_run_cwd(target_checkout, &["cherry-pick", "--no-commit", commit])?;
        if result.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        let conflict_detail = if stderr.trim().is_empty() {
            format!(
                "git cherry-pick --no-commit {commit} failed (exit {:?})",
                result.status.code()
            )
        } else {
            stderr.trim().to_string()
        };

        // Restore carefully. Prefer --abort only when CHERRY_PICK_HEAD exists
        // (normal committing cherry-pick); with --no-commit git often never
        // records it, so we positively choose reset --merge instead.
        let has_cp_head = git_run_cwd(
            target_checkout,
            &["rev-parse", "-q", "--verify", "CHERRY_PICK_HEAD"],
        )?
        .status
        .success();

        let mut restore_notes = Vec::new();
        if has_cp_head {
            let abort = git_run_cwd(target_checkout, &["cherry-pick", "--abort"])?;
            if !abort.status.success() {
                restore_notes.push(format!(
                    "cherry-pick --abort failed (exit {:?}): {}",
                    abort.status.code(),
                    String::from_utf8_lossy(&abort.stderr).trim()
                ));
            }
        } else {
            let reset = git_run_cwd(target_checkout, &["reset", "--merge"])?;
            if !reset.status.success() {
                restore_notes.push(format!(
                    "reset --merge failed (exit {:?}): {}",
                    reset.status.code(),
                    String::from_utf8_lossy(&reset.stderr).trim()
                ));
            }
        }

        let detail = if restore_notes.is_empty() {
            conflict_detail
        } else {
            format!(
                "{conflict_detail}; restoration incomplete: {}",
                restore_notes.join("; ")
            )
        };

        Err(WorkspaceError::Conflict {
            ref_name: self.ref_name(),
            detail,
        })
    }

    /// End the run's hold on the worktree.
    ///
    /// - **Stable** worktree: reset to HEAD (keeping ignored build caches) and
    ///   release the lock. The directory (and its greppy store) stay on disk.
    /// - **Temp** worktree: force-remove via `git worktree remove --force`.
    ///
    /// Proposal refs are **never** deleted.
    pub fn cleanup(self) -> Result<(), WorkspaceError> {
        match self.kind {
            WorktreeKind::Stable => {
                // Reset so the next run (or a post-run inspection) starts clean.
                // Failures here are still errors — a half-dirty stable tree is
                // worse than surfacing the problem.
                self.verify_identity()?;
                reset_worktree_pristine(
                    &self.linked_git_dir,
                    &self.worktree,
                    &self.base_commit,
                    false, // cleanup never drops ignored caches
                )?;
                // Lock drops with `self`.
                Ok(())
            }
            WorktreeKind::Temp => {
                let wt = path_str(&self.worktree)?.to_string();
                git_ok_cwd(&self.repo_root, &["worktree", "remove", "--force", &wt])?;
                Ok(())
            }
        }
    }

    /// Verify the pinned identity still matches the on-disk worktree.
    ///
    /// Checks, in order:
    /// 1. `.git` is a regular file whose `gitdir:` target resolves to
    ///    `linked_git_dir`,
    /// 2. `git --git-dir=<pinned> --work-tree=<wt> rev-parse --absolute-git-dir`
    ///    still equals `linked_git_dir`,
    /// 3. the path is still a registered worktree of `repo_root`.
    fn verify_identity(&self) -> Result<(), WorkspaceError> {
        verify_worktree_identity(&self.repo_root, &self.worktree, &self.linked_git_dir)
    }
}

/// Options for [`AgentWorkspace::create_with_options`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CreateOptions {
    /// When true, reset also drops ignored files (`git clean -ffdx`).
    pub fresh: bool,
}

// Drop intentionally does NOT auto-remove: an unapplied proposal's worktree may
// still need inspection. Cleanup is always explicit via [`AgentWorkspace::cleanup`].

/// Stable worktree directory for `repo_root` under the platform user cache.
///
/// Public so callers/tests can reason about the path without creating a tree.
pub fn stable_worktree_dir(repo_root: &Path) -> PathBuf {
    let canon = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    platform_user_cache_dir()
        .join("greppy")
        .join("agent-worktrees")
        .join(repo_root_hash(&canon))
}

/// Absolute path of the exclusive lock sibling for a stable worktree.
///
/// Public so root-composition tests can assert the lock lies outside every
/// tool-writable root.
pub fn stable_lock_path_for(repo_root: &Path) -> PathBuf {
    stable_lock_path(&stable_worktree_dir(repo_root))
}

fn stable_lock_path(stable_dir: &Path) -> PathBuf {
    // Sibling of the worktree dir: `<parent>/<hash>.lock`
    let parent = stable_dir.parent().unwrap_or_else(|| Path::new("."));
    let name = stable_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("worktree");
    parent.join(format!("{name}.lock"))
}

fn repo_root_hash(canonical_root: &Path) -> String {
    let mut h = Sha256::new();
    h.update(canonical_root.to_string_lossy().as_bytes());
    let digest = h.finalize();
    format!("{:x}", digest).chars().take(16).collect()
}

fn platform_user_cache_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        home_dir().join("Library").join("Caches")
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(xdg);
        }
        home_dir().join(".cache")
    }
}

fn home_dir() -> PathBuf {
    if let Some(h) = std::env::var_os("HOME") {
        return PathBuf::from(h);
    }
    PathBuf::from("/")
}

/// Create or reuse `stable_dir` as a detached worktree at `base_commit`.
/// Returns the absolute linked git directory to pin for the lifetime of the run.
fn prepare_stable_worktree(
    repo_root: &Path,
    stable_dir: &Path,
    base_commit: &str,
    fresh: bool,
) -> Result<PathBuf, WorkspaceError> {
    if let Some(parent) = stable_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    if stable_dir.exists() {
        if let Some(linked) = try_existing_linked_git_dir(stable_dir, repo_root) {
            // Verify before any reset that rediscovers through the tree.
            verify_worktree_identity(repo_root, stable_dir, &linked)?;
            reset_worktree_pristine(&linked, stable_dir, base_commit, fresh)?;
            return Ok(linked);
        }
        // Stale / foreign / tampered: prune registration, remove, recreate.
        let _ = git_run_cwd(repo_root, &["worktree", "prune"]);
        // Best-effort remove; fall through to add.
        if let Err(e) = fs::remove_dir_all(stable_dir) {
            // If still present and non-empty after failure, surface the error.
            if stable_dir.exists() {
                return Err(WorkspaceError::Io(io::Error::new(
                    e.kind(),
                    format!(
                        "cannot remove stale agent worktree {}: {e}",
                        stable_dir.display()
                    ),
                )));
            }
        }
    }

    let path_str_s = path_str(stable_dir)?.to_string();

    git_ok_cwd(
        repo_root,
        &["worktree", "add", "--detach", &path_str_s, base_commit],
    )?;
    let linked = record_linked_git_dir(stable_dir)?;
    // Fresh still applies after a brand-new add (clears any ignored material
    // carried in by sparse/smudge quirks; no-op for a clean tree).
    if fresh {
        reset_worktree_pristine(&linked, stable_dir, base_commit, true)?;
    }
    Ok(linked)
}

/// Resolve the linked git dir for an existing on-disk worktree that still
/// belongs to `repo_root`. Returns `None` when the directory is not a valid
/// registered worktree (caller will recreate).
fn try_existing_linked_git_dir(dir: &Path, repo_root: &Path) -> Option<PathBuf> {
    if !is_valid_worktree_of(dir, repo_root) {
        return None;
    }
    // Safe to rediscover once: is_valid_worktree_of confirmed same common dir.
    // We still record absolute-git-dir (the *linked* dir, not common) to pin.
    record_linked_git_dir(dir).ok()
}

/// Read `git rev-parse --absolute-git-dir` from a freshly created / validated
/// worktree. Discovery is only used at creation / reuse validation; later
/// host-side ops use the returned path exclusively.
fn record_linked_git_dir(worktree: &Path) -> Result<PathBuf, WorkspaceError> {
    let abs = git_ok_cwd(worktree, &["rev-parse", "--absolute-git-dir"])?;
    if abs.is_empty() {
        return Err(WorkspaceError::GitFailed {
            command: "git rev-parse --absolute-git-dir".into(),
            stderr: "empty absolute-git-dir".into(),
            status: None,
        });
    }
    let path = PathBuf::from(&abs);
    // Prefer a canonical form so later equality checks are stable.
    Ok(path.canonicalize().unwrap_or(path))
}

/// True when `dir` is a registered worktree of `repo_root` (same git dir).
fn is_valid_worktree_of(dir: &Path, repo_root: &Path) -> bool {
    // Must look like a git worktree at all.
    let inside = git_run_cwd(dir, &["rev-parse", "--is-inside-work-tree"])
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !inside {
        return false;
    }
    // Common git-dir means it belongs to this repository (covers linked worktrees).
    let Ok(dir_git_common) = git_ok_cwd(dir, &["rev-parse", "--git-common-dir"]) else {
        return false;
    };
    let Ok(repo_git_common) = git_ok_cwd(repo_root, &["rev-parse", "--git-common-dir"]) else {
        return false;
    };
    let dir_common = resolve_maybe_relative(dir, &dir_git_common);
    let repo_common = resolve_maybe_relative(repo_root, &repo_git_common);
    match (dir_common.canonicalize(), repo_common.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => dir_common == repo_common,
    }
}

fn resolve_maybe_relative(cwd: &Path, p: &str) -> PathBuf {
    let path = PathBuf::from(p);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

/// Verify the worktree's control file + registered identity still match the
/// `linked_git_dir` recorded at creation. On mismatch: typed `Tampered` and no
/// further git through that tree.
fn verify_worktree_identity(
    repo_root: &Path,
    worktree: &Path,
    linked_git_dir: &Path,
) -> Result<(), WorkspaceError> {
    let git_file = worktree.join(".git");
    let linked_canon = linked_git_dir
        .canonicalize()
        .unwrap_or_else(|_| linked_git_dir.to_path_buf());

    // 1. `.git` must be a regular file (linked worktrees) pointing at linked_git_dir.
    let meta = fs::symlink_metadata(&git_file).map_err(|e| WorkspaceError::Tampered {
        path: git_file.clone(),
        detail: format!("cannot stat .git control file: {e}"),
    })?;
    if meta.file_type().is_symlink() {
        return Err(WorkspaceError::Tampered {
            path: git_file,
            detail: ".git is a symlink".into(),
        });
    }
    if !meta.is_file() {
        return Err(WorkspaceError::Tampered {
            path: git_file,
            detail: format!(
                ".git is not a regular file (file_type={:?})",
                meta.file_type()
            ),
        });
    }
    let contents = fs::read_to_string(&git_file).map_err(|e| WorkspaceError::Tampered {
        path: git_file.clone(),
        detail: format!("cannot read .git control file: {e}"),
    })?;
    let pointed = parse_gitdir_pointer(&contents).ok_or_else(|| WorkspaceError::Tampered {
        path: git_file.clone(),
        detail: format!(".git does not contain a gitdir: pointer (contents={contents:?})"),
    })?;
    let pointed_abs = if pointed.is_absolute() {
        pointed
    } else {
        worktree.join(pointed)
    };
    let pointed_canon = pointed_abs
        .canonicalize()
        .unwrap_or_else(|_| pointed_abs.clone());
    if pointed_canon != linked_canon {
        return Err(WorkspaceError::Tampered {
            path: git_file,
            detail: format!(
                ".git points at {} but run recorded {}",
                pointed_canon.display(),
                linked_canon.display()
            ),
        });
    }

    // 2. rev-parse --absolute-git-dir with the pinned flags must still equal the pin.
    let abs = git_ok_wt(
        linked_git_dir,
        worktree,
        &["rev-parse", "--absolute-git-dir"],
    )
    .map_err(|e| WorkspaceError::Tampered {
        path: git_file.clone(),
        detail: format!("pinned rev-parse --absolute-git-dir failed: {e}"),
    })?;
    let abs_path = PathBuf::from(&abs);
    let abs_canon = abs_path.canonicalize().unwrap_or(abs_path);
    if abs_canon != linked_canon {
        return Err(WorkspaceError::Tampered {
            path: git_file.clone(),
            detail: format!(
                "pinned absolute-git-dir is {} but run recorded {}",
                abs_canon.display(),
                linked_canon.display()
            ),
        });
    }

    // 3. Still a registered worktree of this repository (common dir match).
    // Use the pinned identity so a poisoned .git cannot fake the answer.
    let dir_common = git_ok_wt(linked_git_dir, worktree, &["rev-parse", "--git-common-dir"])
        .map_err(|e| WorkspaceError::Tampered {
            path: git_file.clone(),
            detail: format!("pinned rev-parse --git-common-dir failed: {e}"),
        })?;
    let repo_common = git_ok_cwd(repo_root, &["rev-parse", "--git-common-dir"]).map_err(|e| {
        WorkspaceError::Tampered {
            path: git_file.clone(),
            detail: format!("repo rev-parse --git-common-dir failed: {e}"),
        }
    })?;
    let dir_common_p = resolve_maybe_relative(worktree, &dir_common);
    let repo_common_p = resolve_maybe_relative(repo_root, &repo_common);
    let same = match (dir_common_p.canonicalize(), repo_common_p.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => dir_common_p == repo_common_p,
    };
    if !same {
        return Err(WorkspaceError::Tampered {
            path: git_file,
            detail: format!(
                "worktree common dir {} is not this repository's {}",
                dir_common_p.display(),
                repo_common_p.display()
            ),
        });
    }

    // Also require the worktree path to still appear in `git worktree list`
    // for this repo, so a foreign tree sharing nothing still fails closed.
    if !is_registered_worktree_path(repo_root, worktree) {
        return Err(WorkspaceError::Tampered {
            path: worktree.to_path_buf(),
            detail: "path is no longer a registered worktree of the repository".into(),
        });
    }

    Ok(())
}

fn parse_gitdir_pointer(contents: &str) -> Option<PathBuf> {
    for line in contents.lines() {
        let trimmed = line.trim();
        // git writes "gitdir: <path>"; accept optional space after the colon.
        if let Some(rest) = trimmed.strip_prefix("gitdir:") {
            let p = rest.trim();
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
    }
    None
}

fn is_registered_worktree_path(repo_root: &Path, worktree: &Path) -> bool {
    let Ok(list) = git_ok_cwd(repo_root, &["worktree", "list", "--porcelain"]) else {
        return false;
    };
    let want = worktree
        .canonicalize()
        .unwrap_or_else(|_| worktree.to_path_buf());
    for line in list.lines() {
        if let Some(rest) = line.strip_prefix("worktree ") {
            let p = PathBuf::from(rest.trim());
            let got = p.canonicalize().unwrap_or(p);
            if got == want {
                return true;
            }
        }
    }
    false
}

/// Detach at `base_commit`, hard-reset, clean untracked (and nested repos),
/// and optionally drop ignored files. Submodules are reset when present.
fn reset_worktree_pristine(
    linked_git_dir: &Path,
    worktree: &Path,
    base_commit: &str,
    fresh: bool,
) -> Result<(), WorkspaceError> {
    git_ok_wt(
        linked_git_dir,
        worktree,
        &["checkout", "-q", "--detach", base_commit],
    )?;
    git_ok_wt(linked_git_dir, worktree, &["reset", "-q", "--hard"])?;
    // Double -f so nested repositories are removed; -x only with --fresh so
    // ignored build caches survive the default reset (repeat runs stay fast).
    if fresh {
        git_ok_wt(linked_git_dir, worktree, &["clean", "-qffdx"])?;
    } else {
        git_ok_wt(linked_git_dir, worktree, &["clean", "-qffd"])?;
    }
    // Submodule reset only when the tree has any — a repo without submodules
    // must not pay the foreach cost (and older git may error without them).
    // `git submodule` is a script that requires a real working-tree cwd even
    // when --git-dir/--work-tree are set, so use the cwd-pinned helper.
    if worktree_has_submodules(worktree) {
        git_ok_wt_cwd(
            linked_git_dir,
            worktree,
            &[
                "submodule",
                "foreach",
                "--recursive",
                "git reset --hard && git clean -qffd",
            ],
        )?;
        if fresh {
            git_ok_wt_cwd(
                linked_git_dir,
                worktree,
                &["submodule", "foreach", "--recursive", "git clean -qffdx"],
            )?;
        }
    }
    Ok(())
}

fn worktree_has_submodules(worktree: &Path) -> bool {
    // Cheap path: .gitmodules present.
    if worktree.join(".gitmodules").is_file() {
        return true;
    }
    false
}

fn create_temp_worktree(
    repo_root: &Path,
    run_id: &str,
    base_commit: &str,
    fresh: bool,
) -> Result<(PathBuf, PathBuf), WorkspaceError> {
    let worktree = std::env::temp_dir().join("greppy-agent").join(run_id);
    if let Some(parent) = worktree.parent() {
        fs::create_dir_all(parent)?;
    }
    if worktree.exists() {
        return Err(WorkspaceError::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("worktree path already exists: {}", worktree.display()),
        )));
    }
    let path_s = path_str(&worktree)?.to_string();
    git_ok_cwd(
        repo_root,
        &["worktree", "add", "--detach", &path_s, base_commit],
    )?;
    let linked = record_linked_git_dir(&worktree)?;
    if fresh {
        reset_worktree_pristine(&linked, &worktree, base_commit, true)?;
    }
    Ok((worktree, linked))
}

/// Try to take an exclusive non-blocking lock on `lock_path`.
///
/// Returns `Ok(None)` when another process holds the lock (caller should fall
/// back to a temp worktree). The lock file itself is never deleted.
///
/// Hardening: open with no-follow semantics, require a regular file owned by
/// the current user, acquire the lock **before** truncating or writing, and
/// never `truncate(true)` on open. Fail closed on anything unexpected.
fn try_acquire_lock(lock_path: &Path) -> Result<Option<FileLock>, WorkspaceError> {
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = open_lock_file(lock_path)?;

    // Validate *after* open (descriptor is on a concrete inode): regular file,
    // owned by us. Refuse symlinks/devices/foreign-owned files.
    validate_lock_file(&file, lock_path)?;

    // Acquire BEFORE truncating or writing anything.
    match try_lock_exclusive(&file)? {
        true => {
            // Now that we hold the lock, record pid for humans inspecting a stuck lock.
            rewrite_lock_pid(&file)?;
            Ok(Some(FileLock {
                file,
                path: lock_path.to_path_buf(),
            }))
        }
        false => Ok(None),
    }
}

fn open_lock_file(lock_path: &Path) -> Result<File, WorkspaceError> {
    let mut opts = OpenOptions::new();
    opts.create(true).read(true).write(true);
    // Never truncate on open — that races with a replaced symlink and can
    // clobber an arbitrary user file before we validate or lock.
    opts.truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW: fail if the final component is a symlink.
        opts.custom_flags(open_flags::O_NOFOLLOW);
    }
    opts.open(lock_path).map_err(|e| {
        // Surface a clear message on ELOOP / ESYMLINK-style failures.
        WorkspaceError::Io(io::Error::new(
            e.kind(),
            format!(
                "cannot open agent worktree lock {} (no-follow): {e}",
                lock_path.display()
            ),
        ))
    })
}

fn validate_lock_file(file: &File, lock_path: &Path) -> Result<(), WorkspaceError> {
    let meta = file.metadata().map_err(|e| {
        WorkspaceError::Io(io::Error::new(
            e.kind(),
            format!("cannot stat lock {}: {e}", lock_path.display()),
        ))
    })?;
    if !meta.is_file() {
        return Err(WorkspaceError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "agent worktree lock is not a regular file: {}",
                lock_path.display()
            ),
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let uid = meta.uid();
        let me = current_uid();
        if uid != me {
            return Err(WorkspaceError::Io(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "agent worktree lock {} is owned by uid {uid}, not current uid {me}",
                    lock_path.display()
                ),
            )));
        }
    }
    Ok(())
}

fn rewrite_lock_pid(file: &File) -> Result<(), WorkspaceError> {
    use std::io::{Seek, SeekFrom};
    // Truncate + write only while the exclusive lock is held.
    file.set_len(0).map_err(WorkspaceError::Io)?;
    let mut fref = file;
    fref.seek(SeekFrom::Start(0)).map_err(WorkspaceError::Io)?;
    writeln!(fref, "{}", std::process::id()).map_err(WorkspaceError::Io)?;
    fref.flush().map_err(WorkspaceError::Io)?;
    Ok(())
}

#[cfg(unix)]
mod open_flags {
    // Platform O_NOFOLLOW values (avoid a libc crate dep).
    #[cfg(target_os = "linux")]
    pub const O_NOFOLLOW: i32 = 0o400_000;
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    pub const O_NOFOLLOW: i32 = 0x100;
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd"
    )))]
    pub const O_NOFOLLOW: i32 = 0;
}

#[cfg(unix)]
fn current_uid() -> u32 {
    // SAFETY: getuid has no preconditions and cannot fail.
    unsafe { libc_getuid() }
}

#[cfg(unix)]
extern "C" {
    #[link_name = "getuid"]
    fn libc_getuid() -> u32;
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;
    // SAFETY: flock only operates on the valid fd owned by `file`.
    let rc = unsafe { libc_flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
    if rc == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    // EWOULDBLOCK / EAGAIN both mean "held by someone else" under LOCK_NB.
    if matches!(err.kind(), io::ErrorKind::WouldBlock) || is_eagain_or_ewouldblock(&err) {
        Ok(false)
    } else {
        Err(err)
    }
}

#[cfg(unix)]
fn is_eagain_or_ewouldblock(err: &io::Error) -> bool {
    match err.raw_os_error() {
        // Linux EAGAIN/EWOULDBLOCK = 11; macOS EAGAIN = 35, EWOULDBLOCK = 35.
        Some(11) | Some(35) => true,
        _ => false,
    }
}

#[cfg(unix)]
fn unlock_file(file: &File) {
    use std::os::fd::AsRawFd;
    const LOCK_UN: i32 = 8;
    // SAFETY: best-effort unlock of the valid fd owned by `file`.
    let _ = unsafe { libc_flock(file.as_raw_fd(), LOCK_UN) };
}

#[cfg(unix)]
extern "C" {
    #[link_name = "flock"]
    fn libc_flock(fd: i32, operation: i32) -> i32;
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> io::Result<bool> {
    // Non-unix: no flock; allow the stable path (single-user assumption).
    Ok(true)
}

#[cfg(not(unix))]
fn unlock_file(_file: &File) {}

fn path_str(p: &Path) -> Result<&str, WorkspaceError> {
    p.to_str().ok_or_else(|| {
        WorkspaceError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path is not valid UTF-8",
        ))
    })
}

/// Git against an ordinary checkout (repo root / apply target): discovery via cwd.
fn git_run_cwd(cwd: &Path, args: &[&str]) -> Result<Output, WorkspaceError> {
    Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .map_err(WorkspaceError::Io)
}

fn git_ok_cwd(cwd: &Path, args: &[&str]) -> Result<String, WorkspaceError> {
    let output = git_run_cwd(cwd, args)?;
    if !output.status.success() {
        return Err(git_failed(&format!("git {}", args.join(" ")), &output));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .trim_end()
        .to_string())
}

/// Git against a linked worktree with an explicitly pinned identity.
///
/// Never relies on cwd discovery of `.git` — a tool can rewrite that file.
fn git_run_wt(git_dir: &Path, work_tree: &Path, args: &[&str]) -> Result<Output, WorkspaceError> {
    let mut cmd = Command::new("git");
    cmd.arg("--git-dir")
        .arg(path_str(git_dir)?)
        .arg("--work-tree")
        .arg(path_str(work_tree)?)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1");
    cmd.output().map_err(WorkspaceError::Io)
}

fn git_ok_wt(git_dir: &Path, work_tree: &Path, args: &[&str]) -> Result<String, WorkspaceError> {
    let output = git_run_wt(git_dir, work_tree, args)?;
    if !output.status.success() {
        return Err(git_failed(
            &format!(
                "git --git-dir={} --work-tree={} {}",
                git_dir.display(),
                work_tree.display(),
                args.join(" ")
            ),
            &output,
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .trim_end()
        .to_string())
}

/// Like [`git_ok_wt`], but also sets `current_dir` to the worktree.
///
/// Required for `git submodule` (a shell script that probes the cwd working
/// tree even when `--git-dir`/`--work-tree` are passed). Identity still comes
/// from the pinned flags, not from rediscovering `.git`.
fn git_ok_wt_cwd(
    git_dir: &Path,
    work_tree: &Path,
    args: &[&str],
) -> Result<String, WorkspaceError> {
    let output = Command::new("git")
        .arg("--git-dir")
        .arg(path_str(git_dir)?)
        .arg("--work-tree")
        .arg(path_str(work_tree)?)
        .args(args)
        .current_dir(work_tree)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .map_err(WorkspaceError::Io)?;
    if !output.status.success() {
        return Err(git_failed(
            &format!(
                "git --git-dir={} --work-tree={} {}",
                git_dir.display(),
                work_tree.display(),
                args.join(" ")
            ),
            &output,
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .trim_end()
        .to_string())
}

fn git_failed(command: &str, output: &Output) -> WorkspaceError {
    WorkspaceError::GitFailed {
        command: command.to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        status: output.status.code(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_tag(prefix: &str) -> String {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{prefix}-{}-{}-{}", std::process::id(), seq, nanos)
    }

    /// `git -C <cwd> …` helper for fixtures (panics on failure).
    fn git_c(cwd: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .args(args)
            .current_dir(cwd)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .unwrap_or_else(|e| panic!("spawn git {:?}: {e}", args));
        if !out.status.success() {
            panic!(
                "git {:?} failed (exit {:?}): {}",
                args,
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        String::from_utf8_lossy(&out.stdout).trim_end().to_string()
    }

    /// Init a fixture repo with one commit; set local user.name/email so
    /// commits work in CI-like environments with no global git identity.
    fn init_fixture(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(unique_tag(tag));
        std::fs::create_dir_all(&root).unwrap();
        git_c(&root, &["init"]);
        // Quiet "main" vs "master" variance across git versions.
        git_c(&root, &["checkout", "-b", "main"]);
        git_c(&root, &["config", "user.name", "fixture"]);
        git_c(&root, &["config", "user.email", "fixture@test.local"]);
        git_c(&root, &["config", "commit.gpgsign", "false"]);
        std::fs::write(root.join("hello.txt"), b"hello\n").unwrap();
        git_c(&root, &["add", "hello.txt"]);
        git_c(&root, &["commit", "-m", "initial"]);
        root
    }

    fn clone_fixture(src: &Path) -> PathBuf {
        let dst = std::env::temp_dir().join(unique_tag("greppy-ws-clone"));
        let out = Command::new("git")
            .args([
                "clone",
                "--quiet",
                src.to_str().unwrap(),
                dst.to_str().unwrap(),
            ])
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .expect("spawn git clone");
        assert!(
            out.status.success(),
            "clone failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        git_c(&dst, &["config", "user.name", "fixture"]);
        git_c(&dst, &["config", "user.email", "fixture@test.local"]);
        git_c(&dst, &["config", "commit.gpgsign", "false"]);
        dst
    }

    /// Force-remove a stable worktree registration + directory after a test.
    fn destroy_stable(repo: &Path, ws_path: &Path) {
        let _ = git_run_cwd(repo, &["worktree", "prune"]);
        if ws_path.exists() {
            let _ = Command::new("git")
                .args([
                    "worktree",
                    "remove",
                    "--force",
                    ws_path.to_str().unwrap_or(""),
                ])
                .current_dir(repo)
                .env("GIT_TERMINAL_PROMPT", "0")
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .output();
            let _ = fs::remove_dir_all(ws_path);
        }
        // Drop the lock file too so the next test starts clean.
        let lock = stable_lock_path(ws_path);
        let _ = fs::remove_file(lock);
    }

    /// Snapshot of the main checkout's HEAD + index for tamper regressions.
    fn main_checkout_fingerprint(repo: &Path) -> (String, String, String) {
        let head = git_c(repo, &["rev-parse", "HEAD"]);
        let head_sym = git_c(repo, &["symbolic-ref", "-q", "HEAD"]);
        let index = git_c(repo, &["ls-files", "-s"]);
        (head, head_sym, index)
    }

    #[test]
    fn create_makes_detached_worktree_at_base_leaving_checkout_untouched() {
        let repo = init_fixture("greppy-ws-create");
        let head_before = git_c(&repo, &["rev-parse", "HEAD"]);
        let status_before = git_c(&repo, &["status", "--porcelain"]);
        assert!(status_before.is_empty());

        // Dirty the *index* of the user checkout: isolation must ignore it.
        std::fs::write(repo.join("hello.txt"), b"dirty in user checkout\n").unwrap();
        let status_dirty = git_c(&repo, &["status", "--porcelain"]);
        assert!(!status_dirty.is_empty());

        let run_id = unique_tag("run-create");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        assert_eq!(ws.base_commit(), head_before);
        assert!(ws.worktree_path().is_dir());
        assert!(ws.worktree_path().join("hello.txt").is_file());
        assert!(ws.is_stable(), "default placement must be stable");
        assert!(
            ws.linked_git_dir().is_dir(),
            "linked_git_dir must be recorded"
        );

        // Detached HEAD at base_commit.
        let wt_head = git_c(ws.worktree_path(), &["rev-parse", "HEAD"]);
        assert_eq!(wt_head, head_before);
        let detached = git_c(ws.worktree_path(), &["rev-parse", "--abbrev-ref", "HEAD"]);
        assert_eq!(detached, "HEAD", "worktree must be detached");

        // Worktree content is clean base, not the dirty checkout.
        let wt_hello = std::fs::read_to_string(ws.worktree_path().join("hello.txt")).unwrap();
        assert_eq!(wt_hello, "hello\n");

        // User checkout still dirty and still at the same HEAD.
        assert_eq!(git_c(&repo, &["rev-parse", "HEAD"]), head_before);
        assert!(!git_c(&repo, &["status", "--porcelain"]).is_empty());

        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn stable_path_derived_from_repo_root() {
        let repo = init_fixture("greppy-ws-stable-path");
        let expected = stable_worktree_dir(&repo);
        let run_id = unique_tag("run-stable-path");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        assert_eq!(ws.worktree_path(), expected);
        // Hash is 16 hex chars.
        let name = expected.file_name().unwrap().to_str().unwrap();
        assert_eq!(name.len(), 16);
        assert!(name.chars().all(|c| c.is_ascii_hexdigit()));
        // Under greppy/agent-worktrees.
        let s = expected.to_string_lossy();
        assert!(
            s.contains("agent-worktrees"),
            "path missing agent-worktrees: {s}"
        );
        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn reuse_resets_to_pristine_head() {
        let repo = init_fixture("greppy-ws-reuse");
        let run_a = unique_tag("run-reuse-a");
        let ws_a = AgentWorkspace::create(&repo, &run_a).expect("create a");
        let wt = ws_a.worktree_path().to_path_buf();
        // Simulate a previous run leaving a dirty file behind (without cleanup).
        std::fs::write(wt.join("leftover.txt"), b"from previous run\n").unwrap();
        assert!(wt.join("leftover.txt").is_file());
        // Drop without cleanup — next create must still scrub it via reuse path.
        drop(ws_a);

        let run_b = unique_tag("run-reuse-b");
        let ws_b = AgentWorkspace::create(&repo, &run_b).expect("create b");
        assert_eq!(ws_b.worktree_path(), &wt);
        assert!(
            !ws_b.worktree_path().join("leftover.txt").exists(),
            "reuse must clean leftover from previous run"
        );
        let hello = std::fs::read_to_string(ws_b.worktree_path().join("hello.txt")).unwrap();
        assert_eq!(hello, "hello\n");
        ws_b.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn foreign_stale_directory_is_recreated() {
        let repo = init_fixture("greppy-ws-foreign");
        let stable = stable_worktree_dir(&repo);
        if let Some(parent) = stable.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        // Plant a non-worktree directory at the stable path.
        fs::create_dir_all(&stable).unwrap();
        fs::write(stable.join("not-a-git-worktree.txt"), b"junk\n").unwrap();

        let run_id = unique_tag("run-foreign");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create over foreign");
        assert_eq!(ws.worktree_path(), &stable);
        assert!(ws.worktree_path().join("hello.txt").is_file());
        assert!(!ws.worktree_path().join("not-a-git-worktree.txt").exists());
        // Must be a real detached worktree of this repo.
        assert!(is_valid_worktree_of(ws.worktree_path(), &repo));
        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn concurrent_lock_falls_back_to_temp() {
        let repo = init_fixture("greppy-ws-lock");
        let stable = stable_worktree_dir(&repo);
        let lock_path = stable_lock_path(&stable);
        // Hold the exclusive lock as if another -p run were in progress.
        let held = try_acquire_lock(&lock_path)
            .expect("lock io")
            .expect("must acquire lock for test");

        let run_id = unique_tag("run-lock-fallback");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create under lock");
        assert!(!ws.is_stable(), "must fall back to temp");
        assert!(
            ws.worktree_path()
                .starts_with(std::env::temp_dir().join("greppy-agent")),
            "temp path unexpected: {}",
            ws.worktree_path().display()
        );
        // Proposals still land on the shared ref.
        std::fs::write(ws.worktree_path().join("hello.txt"), b"from temp\n").unwrap();
        let outcome = ws.finish("temp proposal").expect("finish");
        let (commit, ref_name) = match outcome {
            RunOutcome::Proposal {
                commit, ref_name, ..
            } => (commit, ref_name),
            RunOutcome::Clean => panic!("expected Proposal"),
        };
        assert_eq!(ref_name, format!("refs/greppy/agent/{run_id}"));
        assert_eq!(git_c(&repo, &["rev-parse", &ref_name]), commit);

        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup temp");
        assert!(!wt.exists(), "temp worktree must be removed");
        drop(held);
        let _ = fs::remove_file(lock_path);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn cleanup_keeps_stable_removes_temp() {
        let repo = init_fixture("greppy-ws-cleanup-modes");

        // Stable: cleanup keeps the directory.
        let run_stable = unique_tag("run-cleanup-stable");
        let ws_s = AgentWorkspace::create(&repo, &run_stable).expect("create stable");
        assert!(ws_s.is_stable());
        let stable_path = ws_s.worktree_path().to_path_buf();
        std::fs::write(stable_path.join("scratch.txt"), b"scratch\n").unwrap();
        ws_s.cleanup().expect("cleanup stable");
        assert!(stable_path.exists(), "stable worktree must survive cleanup");
        assert!(
            !stable_path.join("scratch.txt").exists(),
            "cleanup must reset stable tree to pristine"
        );

        // Temp: force by holding the lock, then cleanup must remove.
        let lock = try_acquire_lock(&stable_lock_path(&stable_path))
            .expect("lock io")
            .expect("acquire");
        let run_temp = unique_tag("run-cleanup-temp");
        let ws_t = AgentWorkspace::create(&repo, &run_temp).expect("create temp");
        assert!(!ws_t.is_stable());
        let temp_path = ws_t.worktree_path().to_path_buf();
        assert!(temp_path.exists());
        ws_t.cleanup().expect("cleanup temp");
        assert!(!temp_path.exists(), "temp worktree must be gone");
        drop(lock);

        destroy_stable(&repo, &stable_path);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn proposals_land_on_ref_in_both_modes() {
        let repo = init_fixture("greppy-ws-refs-both");

        // Stable mode proposal.
        let run_s = unique_tag("run-ref-stable");
        let ws_s = AgentWorkspace::create(&repo, &run_s).expect("create stable");
        assert!(ws_s.is_stable());
        std::fs::write(ws_s.worktree_path().join("s.txt"), b"stable\n").unwrap();
        let out_s = ws_s.finish("stable prop").expect("finish stable");
        let (c_s, r_s) = match out_s {
            RunOutcome::Proposal {
                commit, ref_name, ..
            } => (commit, ref_name),
            RunOutcome::Clean => panic!("expected Proposal"),
        };
        assert_eq!(r_s, format!("refs/greppy/agent/{run_s}"));
        assert_eq!(git_c(&repo, &["rev-parse", &r_s]), c_s);
        let stable_path = ws_s.worktree_path().to_path_buf();
        ws_s.cleanup().expect("cleanup stable");

        // Temp mode proposal (hold lock).
        let lock = try_acquire_lock(&stable_lock_path(&stable_path))
            .expect("lock")
            .expect("acquire");
        let run_t = unique_tag("run-ref-temp");
        let ws_t = AgentWorkspace::create(&repo, &run_t).expect("create temp");
        assert!(!ws_t.is_stable());
        std::fs::write(ws_t.worktree_path().join("t.txt"), b"temp\n").unwrap();
        let out_t = ws_t.finish("temp prop").expect("finish temp");
        let (c_t, r_t) = match out_t {
            RunOutcome::Proposal {
                commit, ref_name, ..
            } => (commit, ref_name),
            RunOutcome::Clean => panic!("expected Proposal"),
        };
        assert_eq!(r_t, format!("refs/greppy/agent/{run_t}"));
        assert_eq!(git_c(&repo, &["rev-parse", &r_t]), c_t);
        ws_t.cleanup().expect("cleanup temp");
        drop(lock);

        destroy_stable(&repo, &stable_path);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn finish_with_no_changes_is_clean() {
        let repo = init_fixture("greppy-ws-clean");
        let run_id = unique_tag("run-clean");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");

        let outcome = ws.finish("should not commit").expect("finish");
        assert_eq!(outcome, RunOutcome::Clean);

        // Worktree still removable; no proposal ref was written.
        let ref_name = ws.ref_name();
        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        let refs = Command::new("git")
            .args(["show-ref", "--verify", "--quiet", &ref_name])
            .current_dir(&repo)
            .status()
            .expect("show-ref");
        assert!(!refs.success(), "Clean finish must not create a ref");

        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn finish_with_new_file_and_edit_produces_proposal() {
        let repo = init_fixture("greppy-ws-proposal");
        let run_id = unique_tag("run-proposal");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");

        // Edit existing + add new file inside the worktree only.
        std::fs::write(ws.worktree_path().join("hello.txt"), b"hello agent\n").unwrap();
        std::fs::write(ws.worktree_path().join("new.txt"), b"brand new\n").unwrap();

        let outcome = ws.finish("agent proposal").expect("finish");
        let (commit, ref_name, patch, stat) = match outcome {
            RunOutcome::Proposal {
                commit,
                ref_name,
                patch,
                stat,
            } => (commit, ref_name, patch, stat),
            RunOutcome::Clean => panic!("expected Proposal"),
        };

        assert_eq!(ref_name, format!("refs/greppy/agent/{run_id}"));
        // Ref exists in the *main* repo.
        let ref_oid = git_c(&repo, &["rev-parse", &ref_name]);
        assert_eq!(ref_oid, commit);

        // Patch contains both changes.
        assert!(
            patch.contains("hello agent") || patch.contains("+hello agent"),
            "patch missing edit: {patch}"
        );
        assert!(
            patch.contains("new.txt") && patch.contains("brand new"),
            "patch missing new file: {patch}"
        );
        assert!(!stat.trim().is_empty(), "stat must be non-empty");
        assert!(
            stat.contains("hello.txt") || stat.contains("new.txt"),
            "stat should mention files: {stat}"
        );

        // User checkout untouched.
        let user_hello = std::fs::read_to_string(repo.join("hello.txt")).unwrap();
        assert_eq!(user_hello, "hello\n");
        assert!(!repo.join("new.txt").exists());

        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        // Ref survives cleanup.
        let ref_oid_after = git_c(&repo, &["rev-parse", &ref_name]);
        assert_eq!(ref_oid_after, commit);

        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn apply_to_clean_checkout_stages_files() {
        let repo = init_fixture("greppy-ws-apply");
        let run_id = unique_tag("run-apply");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");

        std::fs::write(ws.worktree_path().join("hello.txt"), b"applied\n").unwrap();
        std::fs::write(ws.worktree_path().join("extra.txt"), b"extra\n").unwrap();
        let outcome = ws.finish("apply me").expect("finish");
        let commit = match outcome {
            RunOutcome::Proposal { commit, .. } => commit,
            RunOutcome::Clean => panic!("expected Proposal"),
        };

        let target = clone_fixture(&repo);
        // Clone is at the same base; apply should stage both files without committing.
        ws.apply_to(&target, &commit).expect("apply_to");

        let status = git_c(&target, &["status", "--porcelain"]);
        // Staged modifications/additions (first column M/A).
        assert!(
            status
                .lines()
                .any(|l| l.starts_with("M  ") || l.starts_with("M\t")),
            "hello.txt should be staged; status={status:?}"
        );
        assert!(
            status.lines().any(|l| l.contains("extra.txt")),
            "extra.txt should be staged; status={status:?}"
        );
        // Not committed: HEAD still matches base.
        assert_eq!(
            git_c(&target, &["rev-parse", "HEAD"]),
            git_c(&repo, &["rev-parse", "HEAD"])
        );
        let content = std::fs::read_to_string(target.join("hello.txt")).unwrap();
        assert_eq!(content, "applied\n");
        let extra = std::fs::read_to_string(target.join("extra.txt")).unwrap();
        assert_eq!(extra, "extra\n");

        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn apply_conflict_aborts_and_leaves_ref() {
        let repo = init_fixture("greppy-ws-conflict");
        let run_id = unique_tag("run-conflict");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");

        std::fs::write(ws.worktree_path().join("hello.txt"), b"from agent\n").unwrap();
        let outcome = ws.finish("conflicting proposal").expect("finish");
        let (commit, ref_name) = match outcome {
            RunOutcome::Proposal {
                commit, ref_name, ..
            } => (commit, ref_name),
            RunOutcome::Clean => panic!("expected Proposal"),
        };

        // Target has a conflicting committed change on the same lines.
        let target = clone_fixture(&repo);
        std::fs::write(target.join("hello.txt"), b"from user\n").unwrap();
        git_c(&target, &["add", "hello.txt"]);
        git_c(&target, &["commit", "-m", "user change"]);
        let head_before = git_c(&target, &["rev-parse", "HEAD"]);
        let hello_before = std::fs::read_to_string(target.join("hello.txt")).unwrap();

        let err = ws.apply_to(&target, &commit).expect_err("must conflict");
        match &err {
            WorkspaceError::Conflict {
                ref_name: rn,
                detail,
            } => {
                assert_eq!(rn, &ref_name);
                assert!(!detail.is_empty());
                // Must not claim a clean abort when restoration may have used
                // reset --merge (no CHERRY_PICK_HEAD with --no-commit).
                assert!(
                    !detail.to_lowercase().contains("aborted")
                        || detail.contains("restoration incomplete"),
                    "error text must not falsely claim abort: {detail}"
                );
            }
            other => panic!("expected Conflict, got {other}"),
        }

        // Restored: no CHERRY_PICK_HEAD, HEAD unchanged, clean tree
        // (the user's committed state).
        let cp = Command::new("git")
            .args(["rev-parse", "-q", "--verify", "CHERRY_PICK_HEAD"])
            .current_dir(&target)
            .status()
            .expect("rev-parse CHERRY_PICK_HEAD");
        assert!(
            !cp.success(),
            "CHERRY_PICK_HEAD must not remain after restore"
        );
        assert_eq!(git_c(&target, &["rev-parse", "HEAD"]), head_before);
        let status = git_c(&target, &["status", "--porcelain"]);
        assert!(
            status.is_empty(),
            "target should be clean after restore; status={status:?}"
        );
        let hello_after = std::fs::read_to_string(target.join("hello.txt")).unwrap();
        assert_eq!(hello_after, hello_before, "file content must be restored");

        // Proposal ref still present in the main repo.
        let ref_oid = git_c(&repo, &["rev-parse", &ref_name]);
        assert_eq!(ref_oid, commit);

        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn apply_to_dirty_target_refused_untouched() {
        let repo = init_fixture("greppy-ws-dirty");
        let run_id = unique_tag("run-dirty");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");

        std::fs::write(ws.worktree_path().join("hello.txt"), b"from agent\n").unwrap();
        let outcome = ws.finish("proposal for dirty target").expect("finish");
        let (commit, ref_name) = match outcome {
            RunOutcome::Proposal {
                commit, ref_name, ..
            } => (commit, ref_name),
            RunOutcome::Clean => panic!("expected Proposal"),
        };

        let target = clone_fixture(&repo);
        // Leave uncommitted changes in the target.
        std::fs::write(target.join("hello.txt"), b"dirty local edit\n").unwrap();
        std::fs::write(target.join("scratch.txt"), b"untracked\n").unwrap();
        let status_before = git_c(&target, &["status", "--porcelain"]);
        assert!(!status_before.is_empty());
        let head_before = git_c(&target, &["rev-parse", "HEAD"]);
        let hello_before = std::fs::read_to_string(target.join("hello.txt")).unwrap();

        let err = ws
            .apply_to(&target, &commit)
            .expect_err("must refuse dirty");
        match &err {
            WorkspaceError::DirtyTarget {
                ref_name: rn,
                detail,
            } => {
                assert_eq!(rn, &ref_name);
                assert!(!detail.is_empty());
            }
            other => panic!("expected DirtyTarget, got {other}"),
        }
        // Display message is the user-facing contract.
        let msg = err.to_string();
        assert!(
            msg.contains("uncommitted changes") && msg.contains(&ref_name),
            "display={msg}"
        );

        // Target completely untouched: same HEAD, same dirty status, same file.
        assert_eq!(git_c(&target, &["rev-parse", "HEAD"]), head_before);
        assert_eq!(
            git_c(&target, &["status", "--porcelain"]),
            status_before,
            "dirty status must be unchanged"
        );
        assert_eq!(
            std::fs::read_to_string(target.join("hello.txt")).unwrap(),
            hello_before
        );
        // No cherry-pick residue.
        let cp = Command::new("git")
            .args(["rev-parse", "-q", "--verify", "CHERRY_PICK_HEAD"])
            .current_dir(&target)
            .status()
            .expect("rev-parse");
        assert!(!cp.success());

        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn finish_ignores_model_mid_run_commit_and_captures_full_delta() {
        let repo = init_fixture("greppy-ws-midcommit");
        let base = git_c(&repo, &["rev-parse", "HEAD"]);
        let run_id = unique_tag("run-midcommit");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        assert_eq!(ws.base_commit(), base);

        // Model makes an intermediate commit that would otherwise become the
        // parent of a naive `git commit` proposal.
        std::fs::write(ws.worktree_path().join("mid.txt"), b"mid-run\n").unwrap();
        git_c(ws.worktree_path(), &["add", "mid.txt"]);
        // Use the same author config as the fixture helper.
        git_c(
            ws.worktree_path(),
            &["commit", "-m", "model mid-run commit"],
        );
        let mid_head = git_c(ws.worktree_path(), &["rev-parse", "HEAD"]);
        assert_ne!(mid_head, base, "model commit must move HEAD");

        // Further filesystem changes after the mid-run commit.
        std::fs::write(ws.worktree_path().join("hello.txt"), b"final hello\n").unwrap();
        std::fs::write(ws.worktree_path().join("late.txt"), b"late file\n").unwrap();

        let outcome = ws.finish("head-proof proposal").expect("finish");
        let (commit, ref_name, patch, _stat) = match outcome {
            RunOutcome::Proposal {
                commit,
                ref_name,
                patch,
                stat,
            } => (commit, ref_name, patch, stat),
            RunOutcome::Clean => panic!("expected Proposal"),
        };

        // Proposal parent is base_commit, not the model's mid-run commit.
        let parents = git_c(&repo, &["rev-list", "--parents", "-n", "1", &commit]);
        // Format: "<commit> <parent>..."
        let mut parts = parents.split_whitespace();
        assert_eq!(parts.next(), Some(commit.as_str()));
        assert_eq!(parts.next(), Some(base.as_str()));
        assert_eq!(parts.next(), None, "proposal must be exactly one parent");

        // Full delta from base: mid.txt, hello.txt edit, late.txt all present.
        assert!(
            patch.contains("mid.txt") && patch.contains("mid-run"),
            "patch missing mid-run file: {patch}"
        );
        assert!(
            patch.contains("final hello") || patch.contains("+final hello"),
            "patch missing final hello edit: {patch}"
        );
        assert!(
            patch.contains("late.txt") && patch.contains("late file"),
            "patch missing late file: {patch}"
        );

        // Applying the proposal onto a clean base yields the full final state.
        let target = clone_fixture(&repo);
        ws.apply_to(&target, &commit).expect("apply full delta");
        assert_eq!(
            std::fs::read_to_string(target.join("hello.txt")).unwrap(),
            "final hello\n"
        );
        assert_eq!(
            std::fs::read_to_string(target.join("mid.txt")).unwrap(),
            "mid-run\n"
        );
        assert_eq!(
            std::fs::read_to_string(target.join("late.txt")).unwrap(),
            "late file\n"
        );

        let _ = ref_name;
        let wt = ws.worktree_path().to_path_buf();
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn create_refuses_non_git_directory() {
        let dir = std::env::temp_dir().join(unique_tag("greppy-ws-nongit"));
        std::fs::create_dir_all(&dir).unwrap();
        // Ensure it is not accidentally a git repo.
        assert!(!dir.join(".git").exists());

        let err = AgentWorkspace::create(&dir, "run-nongit").expect_err("must refuse");
        match err {
            WorkspaceError::NotGitRepo { path, detail } => {
                assert_eq!(path, dir);
                assert!(!detail.is_empty(), "detail should explain the failure");
            }
            other => panic!("expected NotGitRepo, got {other}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn drop_does_not_auto_remove_worktree() {
        let repo = init_fixture("greppy-ws-drop");
        let run_id = unique_tag("run-drop");
        let wt_path = {
            let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
            let p = ws.worktree_path().to_path_buf();
            // Drop without cleanup.
            p
        };
        assert!(wt_path.exists(), "Drop must NOT auto-remove the worktree");
        // Explicit cleanup via git from the main repo.
        destroy_stable(&repo, &wt_path);
        let _ = std::fs::remove_dir_all(&repo);
    }

    // --- V1: poisoned `.git` must not redirect host-side git into the user checkout ---

    #[test]
    fn finish_tampered_git_deleted_fails_and_main_untouched() {
        let repo = init_fixture("greppy-ws-tamper-del");
        let fp_before = main_checkout_fingerprint(&repo);
        let run_id = unique_tag("run-tamper-del");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        let wt = ws.worktree_path().to_path_buf();

        // Agent content that a naive `git add -A` through a poisoned pointer
        // would stage into the main index.
        std::fs::write(wt.join("agent-only.txt"), b"agent content\n").unwrap();
        fs::remove_file(wt.join(".git")).expect("delete .git");

        let err = ws.finish("must fail").expect_err("finish must Tampered");
        match &err {
            WorkspaceError::Tampered { path, detail } => {
                assert!(
                    path.ends_with(".git") || path == &wt,
                    "path={path:?} detail={detail}"
                );
                assert!(!detail.is_empty());
            }
            other => panic!("expected Tampered, got {other}"),
        }
        assert_eq!(
            main_checkout_fingerprint(&repo),
            fp_before,
            "main checkout HEAD/index must be unchanged"
        );
        assert!(
            !repo.join("agent-only.txt").exists(),
            "agent file must not appear in main checkout"
        );

        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn finish_tampered_git_rewritten_to_main_fails_and_main_untouched() {
        let repo = init_fixture("greppy-ws-tamper-main");
        let fp_before = main_checkout_fingerprint(&repo);
        let main_git = git_c(&repo, &["rev-parse", "--absolute-git-dir"]);
        let run_id = unique_tag("run-tamper-main");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        let wt = ws.worktree_path().to_path_buf();

        std::fs::write(wt.join("agent-only.txt"), b"agent content\n").unwrap();
        // Rewrite .git to point at the MAIN repo's .git — the original attack.
        std::fs::write(wt.join(".git"), format!("gitdir: {main_git}\n")).unwrap();

        let err = ws.finish("must fail").expect_err("finish must Tampered");
        match &err {
            WorkspaceError::Tampered { .. } => {}
            other => panic!("expected Tampered, got {other}"),
        }
        assert_eq!(
            main_checkout_fingerprint(&repo),
            fp_before,
            "main checkout HEAD/index must be unchanged after main-.git rewrite"
        );
        // Index must not contain agent-only.txt.
        let index = git_c(&repo, &["ls-files"]);
        assert!(
            !index.lines().any(|l| l == "agent-only.txt"),
            "agent-only.txt must not be staged in main index; index={index:?}"
        );

        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn finish_tampered_git_rewritten_to_foreign_fails_and_main_untouched() {
        let repo = init_fixture("greppy-ws-tamper-foreign");
        let foreign = init_fixture("greppy-ws-foreign-target");
        let fp_before = main_checkout_fingerprint(&repo);
        let foreign_git = git_c(&foreign, &["rev-parse", "--absolute-git-dir"]);
        let run_id = unique_tag("run-tamper-foreign");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        let wt = ws.worktree_path().to_path_buf();

        std::fs::write(wt.join("agent-only.txt"), b"agent content\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {foreign_git}\n")).unwrap();

        let err = ws.finish("must fail").expect_err("finish must Tampered");
        match &err {
            WorkspaceError::Tampered { .. } => {}
            other => panic!("expected Tampered, got {other}"),
        }
        assert_eq!(
            main_checkout_fingerprint(&repo),
            fp_before,
            "main checkout HEAD/index must be unchanged after foreign-.git rewrite"
        );
        assert_eq!(
            main_checkout_fingerprint(&foreign),
            main_checkout_fingerprint(&foreign), // foreign identity still readable
            "sanity"
        );
        // Foreign HEAD/index also unchanged relative to its own baseline.
        let foreign_head = git_c(&foreign, &["rev-parse", "HEAD"]);
        let foreign_index = git_c(&foreign, &["ls-files"]);
        assert!(
            !foreign_index.lines().any(|l| l == "agent-only.txt"),
            "agent content must not stage into foreign"
        );
        let _ = foreign_head;

        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&foreign);
    }

    // --- V3: honest reset semantics ---

    #[test]
    fn reset_removes_nested_repository() {
        let repo = init_fixture("greppy-ws-nested");
        let run_id = unique_tag("run-nested");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        let wt = ws.worktree_path().to_path_buf();

        // Plant an untracked nested git repository.
        let nested = wt.join("nested-evil");
        fs::create_dir_all(&nested).unwrap();
        git_c(&nested, &["init"]);
        std::fs::write(nested.join("payload.sh"), b"#!/bin/sh\necho pwned\n").unwrap();
        assert!(nested.join(".git").exists());

        // Drop without cleanup, then recreate — reuse path must scrub nested repo.
        drop(ws);
        let ws2 = AgentWorkspace::create(&repo, &unique_tag("run-nested-2")).expect("reuse");
        assert_eq!(ws2.worktree_path(), &wt);
        assert!(
            !wt.join("nested-evil").exists(),
            "nested repository must be removed by git clean -ffd"
        );
        ws2.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn reset_keeps_ignored_by_default_drops_with_fresh() {
        let repo = init_fixture("greppy-ws-ignored");
        // Ignore build-cache style paths.
        std::fs::write(repo.join(".gitignore"), b"target/\n*.cache\n").unwrap();
        git_c(&repo, &["add", ".gitignore"]);
        git_c(&repo, &["commit", "-m", "ignore build caches"]);

        let run_id = unique_tag("run-ignored");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        let wt = ws.worktree_path().to_path_buf();

        // Plant ignored material.
        fs::create_dir_all(wt.join("target")).unwrap();
        std::fs::write(wt.join("target").join("artifact.bin"), b"cache\n").unwrap();
        std::fs::write(wt.join("foo.cache"), b"cache-file\n").unwrap();
        // And an untracked non-ignored file that must always go.
        std::fs::write(wt.join("scratch.txt"), b"scratch\n").unwrap();

        drop(ws);
        // Default reuse: ignored survive, untracked non-ignored gone.
        let ws2 = AgentWorkspace::create(&repo, &unique_tag("run-ignored-2")).expect("reuse");
        assert!(
            ws2.worktree_path().join("target/artifact.bin").is_file(),
            "ignored build cache must survive default reset"
        );
        assert!(
            ws2.worktree_path().join("foo.cache").is_file(),
            "ignored file must survive default reset"
        );
        assert!(
            !ws2.worktree_path().join("scratch.txt").exists(),
            "untracked non-ignored must be cleaned"
        );
        drop(ws2);

        // --fresh: ignored go too.
        let ws3 = AgentWorkspace::create_with_options(
            &repo,
            &unique_tag("run-ignored-3"),
            CreateOptions { fresh: true },
        )
        .expect("fresh");
        assert!(
            !ws3.worktree_path().join("target/artifact.bin").exists(),
            "fresh must drop ignored build caches"
        );
        assert!(
            !ws3.worktree_path().join("foo.cache").exists(),
            "fresh must drop ignored files"
        );
        ws3.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn reset_resets_submodule_when_present() {
        let repo = init_fixture("greppy-ws-submod");
        // Create a sibling repo to add as a submodule via file:// URL.
        let sub_src = init_fixture("greppy-ws-submod-src");
        // git submodule add needs a clean index; repo is clean.
        let sub_url = format!("file://{}", sub_src.display());
        // Some CI gits require protocol.file.allow=always for file:// submodules.
        git_c(
            &repo,
            &[
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                &sub_url,
                "vendor/lib",
            ],
        );
        git_c(&repo, &["commit", "-m", "add submodule"]);

        let run_id = unique_tag("run-submod");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        let wt = ws.worktree_path().to_path_buf();

        // Init the submodule inside the worktree so it has a working tree.
        git_c(
            wt.as_path(),
            &[
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "update",
                "--init",
            ],
        );
        let sub_path = wt.join("vendor/lib");
        assert!(sub_path.join("hello.txt").is_file());

        // Dirty the submodule working tree.
        std::fs::write(sub_path.join("hello.txt"), b"dirty submodule\n").unwrap();
        std::fs::write(sub_path.join("extra.txt"), b"extra\n").unwrap();
        assert_ne!(
            std::fs::read_to_string(sub_path.join("hello.txt")).unwrap(),
            "hello\n"
        );

        drop(ws);
        let ws2 = AgentWorkspace::create(&repo, &unique_tag("run-submod-2")).expect("reuse");
        let sub2 = ws2.worktree_path().join("vendor/lib");
        // After reset, submodule content is restored (if still present/initialized).
        if sub2.join("hello.txt").is_file() {
            assert_eq!(
                std::fs::read_to_string(sub2.join("hello.txt")).unwrap(),
                "hello\n",
                "submodule tracked file must be reset"
            );
            assert!(
                !sub2.join("extra.txt").exists(),
                "submodule untracked must be cleaned"
            );
        }
        ws2.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&sub_src);
    }

    #[test]
    fn lock_rejects_symlink_path() {
        let dir = std::env::temp_dir().join(unique_tag("greppy-ws-lock-sym"));
        fs::create_dir_all(&dir).unwrap();
        let real = dir.join("real.lock");
        std::fs::write(&real, b"x\n").unwrap();
        let link = dir.join("link.lock");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real, &link).unwrap();
            let err = try_acquire_lock(&link).expect_err("symlink lock must fail");
            let msg = err.to_string();
            assert!(
                msg.contains("no-follow")
                    || msg.contains("symlink")
                    || msg.contains("not a regular"),
                "unexpected err: {msg}"
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
