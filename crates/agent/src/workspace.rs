//! Per-run isolation: native worktrees or private CoW snapshots and proposals.
//!
//! Every agent run works in a private tree pinned to the target repository's
//! current commit. The agent never writes to the user's checkout. The run's
//! outcome is a **proposal**: a commit preserved on
//! `refs/greppy/agent/<run_id>` plus a printable patch. Applying it to a real
//! checkout is a separate, explicit host-side step.
//!
//! # Worktree placement
//!
//! The 0.3.3 CLI defaults to Filesystem-CoW with fail-closed native fallback.
//! A short-lived stable Git worktree is the pristine snapshot template:
//!
//! ```text
//! <platform-cache>/greppy/agent-worktrees/<16-hex sha256 of canonical repo root>
//! ```
//!
//! (`~/Library/Caches` on macOS; `$XDG_CACHE_HOME` or `~/.cache` elsewhere.)
//! The template is reset to the pinned commit under an exclusive lock. A CoW
//! run snapshots it into a private sibling and immediately releases that lock,
//! allowing concurrent agents to share physical extents without sharing
//! writable files. The snapshot receives a private Git directory whose object
//! database uses the main repository only as a read-only alternate. New model
//! objects and refs remain private until `finish` transfers the proposal.
//!
//! `--workspace-backend native` retains the 0.3.2 stable/temp implementation.
//! `auto` falls back there before model startup when CoW is unavailable;
//! `cow` fails explicitly instead. Ignored build caches are retained in the
//! template by default; `--fresh` drops them before snapshotting.
//!
//! Host-side git against the worktree always pins `--git-dir` + `--work-tree`
//! recorded at creation time. If the worktree's `.git` control file is later
//! rewritten, `finish` / reset refuse with [`WorkspaceError::Tampered`] rather
//! than rediscovering a poisoned pointer into the user checkout.
//!
//! [`AgentWorkspace::cleanup`] removes CoW and temporary workspaces but keeps
//! and resets the stable native/template worktree. `--keep-worktree` preserves
//! a CoW or temporary tree for inspection.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// Per-run git worktree isolation for an agent invocation.
#[derive(Debug)]
pub struct AgentWorkspace {
    repo_root: PathBuf,
    worktree: PathBuf,
    /// Absolute Git directory pinned at creation. Native worktrees use the
    /// main repository's linked-worktree registration; CoW workspaces use a
    /// private real Git directory inside the snapshot.
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
    /// Per-run native CoW snapshot with private Git state; removed on cleanup.
    Cow,
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
    /// The repository uses a feature the agent worktree cannot handle safely
    /// (currently: tracked submodules). Create refuses without writing anything.
    Unsupported(String),
    /// A caller explicitly required CoW but the native backend could not be
    /// created safely. Auto mode consumes this error and falls back natively.
    CowUnavailable(String),
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
            Self::Unsupported(reason) => {
                write!(f, "repository is not supported by greppy -p: {reason}")
            }
            Self::CowUnavailable(reason) => {
                write!(f, "Filesystem-CoW is unavailable: {reason}")
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
    /// Create a native detached worktree for `run_id` from `repo_root`'s
    /// `HEAD`, preserving the pre-0.3.3 library default.
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

    /// Like [`Self::create`], with explicit reset and backend selection.
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

        // Resolve base_commit FIRST with pinned main-repo flags, then inspect
        // that exact commit for gitlink entries (mode 160000). Never a second
        // HEAD lookup — detection and recorded base must be the same object.
        let main_git_dir =
            PathBuf::from(git_ok_cwd(&toplevel, &["rev-parse", "--absolute-git-dir"])?);
        let main_common_git_dir = PathBuf::from(git_ok_wt(
            &main_git_dir,
            &toplevel,
            &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        )?);
        let base_commit = git_ok_wt(&main_git_dir, &toplevel, &["rev-parse", "HEAD"])?;
        if main_commit_has_gitlinks(&main_git_dir, &toplevel, &base_commit)? {
            return Err(WorkspaceError::Unsupported(
                "gitlink entries (submodules) present; greppy -p cannot reset submodules safely"
                    .into(),
            ));
        }

        let stable_dir = stable_worktree_dir(&toplevel);
        let lock_path = stable_lock_path(&stable_dir);

        match options.backend {
            WorkspaceBackend::Native => match try_acquire_lock(&lock_path)? {
                Some(lock) => {
                    let linked_git_dir = prepare_stable_worktree(
                        &toplevel,
                        &stable_dir,
                        &base_commit,
                        options.fresh,
                        false,
                    )?;
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
            },
            WorkspaceBackend::Auto => match try_acquire_lock(&lock_path)? {
                Some(lock) => {
                    // Preserve the warm serial path: CoW replaces only the concurrent
                    // temporary-worktree fallback in automatic mode.
                    let linked_git_dir = prepare_stable_worktree(
                        &toplevel,
                        &stable_dir,
                        &base_commit,
                        options.fresh,
                        false,
                    )?;
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
                None => match create_cow_from_template(
                    &toplevel,
                    &main_common_git_dir,
                    run_id,
                    &base_commit,
                    true,
                ) {
                    Ok((worktree, private_git_dir)) => Ok(Self {
                        repo_root: toplevel,
                        worktree,
                        linked_git_dir: private_git_dir,
                        run_id: run_id.to_string(),
                        base_commit,
                        kind: WorktreeKind::Cow,
                        _lock: None,
                    }),
                    Err(error) => {
                        eprintln!(
                            "greppy -p: Filesystem-CoW unavailable ({error}) — using a temporary native worktree"
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
                },
            },
            WorkspaceBackend::Cow => {
                let (worktree, private_git_dir) = create_cow_from_template(
                    &toplevel,
                    &main_common_git_dir,
                    run_id,
                    &base_commit,
                    false,
                )?;
                Ok(Self {
                    repo_root: toplevel,
                    worktree,
                    linked_git_dir: private_git_dir,
                    run_id: run_id.to_string(),
                    base_commit,
                    kind: WorktreeKind::Cow,
                    _lock: None,
                })
            }
        }
    }

    /// Absolute path of the worktree (becomes [`crate::GreppyEnv`]'s root).
    pub fn worktree_path(&self) -> &Path {
        &self.worktree
    }

    /// Absolute pinned Git directory: linked for native, private for CoW.
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

    /// True when this run uses the stable native per-repository worktree.
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
        let mut commit_cmd = git_command();
        commit_cmd
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
            .env("GIT_CONFIG_VALUE_0", "false");
        let commit_out = commit_cmd.output().map_err(WorkspaceError::Io)?;
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

        if self.kind == WorktreeKind::Cow {
            let export_ref = "refs/greppy/export/proposal";
            git_ok_wt(
                &self.linked_git_dir,
                &self.worktree,
                &["update-ref", export_ref, &commit],
            )?;
            git_ok_cwd(
                &self.repo_root,
                &[
                    "fetch",
                    "--no-tags",
                    "--no-write-fetch-head",
                    path_str(&self.linked_git_dir)?,
                    export_ref,
                ],
            )?;
            git_ok_cwd(
                &self.repo_root,
                &["cat-file", "-e", &format!("{commit}^{{commit}}")],
            )?;
        }

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
    /// - **Temp** worktree: verify identity, then force-remove via
    ///   `git worktree remove --force` from the main repo. On
    ///   [`WorkspaceError::Tampered`] the tree is kept for inspection.
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
                // Same identity gate as stable: refuse to act through a tree
                // whose git pointer no longer matches the main registration.
                self.verify_identity()?;
                let wt = path_str(&self.worktree)?.to_string();
                git_ok_cwd(&self.repo_root, &["worktree", "remove", "--force", &wt])?;
                Ok(())
            }
            WorktreeKind::Cow => {
                self.verify_identity()?;
                greppy_rift_core::remove_snapshot(&self.worktree).map_err(|error| {
                    WorkspaceError::Io(io::Error::other(format!(
                        "failed to remove CoW workspace {}: {error}",
                        self.worktree.display()
                    )))
                })?;
                Ok(())
            }
        }
    }

    /// Verify the pinned identity still matches the on-disk worktree.
    ///
    /// Native identity is anchored in the main repository registration. CoW
    /// identity requires a real private `.git` directory contained by the
    /// pinned snapshot path.
    fn verify_identity(&self) -> Result<(), WorkspaceError> {
        match self.kind {
            WorktreeKind::Cow => verify_private_git_identity(&self.worktree, &self.linked_git_dir),
            WorktreeKind::Stable | WorktreeKind::Temp => {
                verify_worktree_identity(&self.repo_root, &self.worktree, &self.linked_git_dir)
            }
        }
    }
}

/// Options for [`AgentWorkspace::create_with_options`].
#[derive(Debug, Clone, Copy)]
pub struct CreateOptions {
    /// When true, reset also drops ignored files (`git clean -ffdx`).
    pub fresh: bool,
    /// Workspace allocation backend. Direct library callers default to native
    /// for API compatibility; the 0.3.3 CLI passes [`WorkspaceBackend::Auto`].
    pub backend: WorkspaceBackend,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            fresh: false,
            backend: WorkspaceBackend::Native,
        }
    }
}

/// Requested workspace allocation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceBackend {
    /// Use CoW when available and fall back before model startup.
    Auto,
    /// Always use the existing Git-worktree implementation.
    Native,
    /// Require CoW and return an explicit error when unavailable.
    Cow,
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

fn cow_template_dir(repo_root: &Path) -> PathBuf {
    let stable = stable_worktree_dir(repo_root);
    let parent = stable.parent().unwrap_or_else(|| Path::new("."));
    let name = stable
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("worktree");
    parent.join(format!("{name}.cow-template"))
}

fn cow_template_ready_path(template: &Path) -> PathBuf {
    let parent = template.parent().unwrap_or_else(|| Path::new("."));
    let name = template
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("cow-template");
    parent.join(format!("{name}.ready"))
}

fn cow_worktree_dir(stable_dir: &Path, run_id: &str) -> PathBuf {
    let parent = stable_dir.parent().unwrap_or_else(|| Path::new("."));
    let template = stable_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("worktree");
    let mut hash = Sha256::new();
    hash.update(run_id.as_bytes());
    let run_hash: String = format!("{:x}", hash.finalize()).chars().take(16).collect();
    parent.join(format!("{template}.cow.{run_hash}"))
}

fn snapshot_cow_workspace(
    template: &Path,
    run_id: &str,
    require_constant_time_metadata: bool,
) -> Result<(PathBuf, bool), WorkspaceError> {
    let destination = cow_worktree_dir(template, run_id);
    if destination.exists() {
        return Err(WorkspaceError::CowUnavailable(format!(
            "CoW workspace path already exists and was preserved for recovery: {}",
            destination.display()
        )));
    }
    let destination_root = destination.parent().ok_or_else(|| {
        WorkspaceError::CowUnavailable(format!(
            "CoW workspace has no destination root: {}",
            destination.display()
        ))
    })?;
    let capability = greppy_rift_core::probe(template, destination_root).map_err(|error| {
        WorkspaceError::CowUnavailable(format!("native CoW capability probe failed: {error}"))
    })?;
    if require_constant_time_metadata && !capability.constant_time_metadata {
        return Err(WorkspaceError::CowUnavailable(format!(
            "{} is exact CoW but traverses the full tree; auto requires constant-time metadata",
            match capability.backend {
                greppy_rift_core::Backend::ApfsClonefile => "APFS clonefile",
                greppy_rift_core::Backend::BtrfsSnapshot => "Btrfs snapshot",
                greppy_rift_core::Backend::LinuxReflinkTree => "Linux reflink tree",
            }
        )));
    }
    greppy_rift_core::snapshot_exact(template, &destination).map_err(|error| {
        WorkspaceError::CowUnavailable(format!("native CoW snapshot failed: {error}"))
    })?;
    Ok((
        destination,
        capability.constant_time_metadata && capability.source_immutable,
    ))
}

fn finish_cow_workspace(
    destination: PathBuf,
    main_common_git_dir: &Path,
    base_commit: &str,
    source_was_immutable: bool,
) -> Result<(PathBuf, PathBuf), WorkspaceError> {
    match setup_private_git(
        &destination,
        main_common_git_dir,
        base_commit,
        !source_was_immutable,
    ) {
        Ok(private_git_dir) => Ok((destination, private_git_dir)),
        Err(setup_error) => match greppy_rift_core::remove_snapshot(&destination) {
            Ok(_) => Err(setup_error),
            Err(cleanup_error) => Err(WorkspaceError::Io(io::Error::other(format!(
                "{setup_error}; cleanup of partial CoW workspace {} also failed: {cleanup_error}",
                destination.display()
            )))),
        },
    }
}

fn create_cow_from_template(
    repo_root: &Path,
    main_common_git_dir: &Path,
    run_id: &str,
    base_commit: &str,
    require_constant_time_metadata: bool,
) -> Result<(PathBuf, PathBuf), WorkspaceError> {
    let template = cow_template_dir(repo_root);
    let template_lock_path = stable_lock_path(&template);
    let lock =
        acquire_template_lock(&template_lock_path, Duration::from_secs(30))?.ok_or_else(|| {
            WorkspaceError::CowUnavailable(
                "Filesystem-CoW template is currently locked by another creator".into(),
            )
        })?;
    let ready_path = cow_template_ready_path(&template);
    let ready = template.is_dir()
        && fs::read_to_string(&ready_path)
            .map(|value| value.trim() == base_commit)
            .unwrap_or(false);
    if !ready {
        // Publish readiness last. A killed or failed preparation leaves no
        // trusted marker, so the next creator rebuilds under the same lock.
        let _ = fs::remove_file(&ready_path);
        if template.exists() {
            greppy_rift_core::set_snapshot_source_immutable(&template, false).map_err(|error| {
                WorkspaceError::CowUnavailable(format!(
                    "failed to unseal stale CoW template {}: {error}",
                    template.display()
                ))
            })?;
            discard_worktree_from_main(repo_root, &template)?;
        }
        prepare_stable_worktree(repo_root, &template, base_commit, true, true)?;
        greppy_rift_core::set_snapshot_source_immutable(&template, true).map_err(|error| {
            WorkspaceError::CowUnavailable(format!(
                "failed to seal CoW template {}: {error}",
                template.display()
            ))
        })?;
        fs::write(&ready_path, format!("{base_commit}\n"))?;
    }
    let (worktree, source_was_immutable) =
        snapshot_cow_workspace(&template, run_id, require_constant_time_metadata)?;
    // The template stays immutable while the native snapshot is taken. Private
    // Git setup mutates only the destination and can proceed concurrently.
    drop(lock);
    let result = finish_cow_workspace(
        worktree,
        main_common_git_dir,
        base_commit,
        source_was_immutable,
    );
    if result.is_err() {
        // A template that cloned but failed the private-Git cleanliness check
        // is no longer trusted. The next creator rebuilds it under the lock.
        let _ = fs::remove_file(ready_path);
    }
    result
}

fn setup_private_git(
    worktree: &Path,
    main_common_git_dir: &Path,
    base_commit: &str,
    verify_snapshot_clean: bool,
) -> Result<PathBuf, WorkspaceError> {
    let control = worktree.join(".git");
    let metadata = fs::symlink_metadata(&control).map_err(WorkspaceError::Io)?;
    if !metadata.file_type().is_file() {
        return Err(WorkspaceError::Tampered {
            path: control,
            detail: "CoW template did not contain a regular linked-worktree .git control file"
                .into(),
        });
    }
    fs::remove_file(&control)?;

    // Refuse user/global init templates: repository-provided hooks must never
    // be copied into the agent's private Git directory.
    let empty_template = worktree.join(".greppy-empty-git-template");
    fs::create_dir(&empty_template)?;
    let template_arg = format!("--template={}", path_str(&empty_template)?);
    let init = git_run_cwd(worktree, &["init", "--quiet", &template_arg]);
    let template_cleanup = fs::remove_dir(&empty_template);
    let init = init?;
    template_cleanup?;
    if !init.status.success() {
        return Err(git_failed("git init --quiet --template=<empty>", &init));
    }

    let private_git_dir = fs::canonicalize(worktree.join(".git"))?;
    let shared_objects = fs::canonicalize(main_common_git_dir.join("objects"))?;
    let shared_objects = path_str(&shared_objects)?;
    if shared_objects.contains(['\n', '\r']) {
        return Err(WorkspaceError::CowUnavailable(
            "Git object path contains a line break and cannot be represented safely as an alternate"
                .into(),
        ));
    }
    let alternates = private_git_dir.join("objects/info/alternates");
    fs::write(&alternates, format!("{shared_objects}\n"))?;
    fs::write(private_git_dir.join("HEAD"), format!("{base_commit}\n"))?;
    git_ok_wt(&private_git_dir, worktree, &["read-tree", base_commit])?;

    if verify_snapshot_clean {
        let status = git_run_wt(
            &private_git_dir,
            worktree,
            &["status", "--porcelain=v1", "-z"],
        )?;
        if !status.status.success() {
            return Err(git_failed("git status --porcelain=v1 -z", &status));
        }
        if !status.stdout.is_empty() {
            return Err(WorkspaceError::CowUnavailable(
                "CoW snapshot and pinned base tree differ before agent startup".into(),
            ));
        }
    }
    Ok(private_git_dir)
}

fn verify_private_git_identity(
    worktree: &Path,
    expected_git_dir: &Path,
) -> Result<(), WorkspaceError> {
    let control = worktree.join(".git");
    let metadata = fs::symlink_metadata(&control).map_err(|error| WorkspaceError::Tampered {
        path: control.clone(),
        detail: format!("private Git directory is missing or unreadable: {error}"),
    })?;
    if !metadata.file_type().is_dir() {
        return Err(WorkspaceError::Tampered {
            path: control,
            detail: "private .git is not a real directory".into(),
        });
    }
    let actual = fs::canonicalize(&control).map_err(|error| WorkspaceError::Tampered {
        path: control.clone(),
        detail: format!("private Git directory cannot be canonicalized: {error}"),
    })?;
    let expected =
        fs::canonicalize(expected_git_dir).map_err(|error| WorkspaceError::Tampered {
            path: expected_git_dir.to_path_buf(),
            detail: format!("pinned private Git directory cannot be canonicalized: {error}"),
        })?;
    let root = fs::canonicalize(worktree).map_err(|error| WorkspaceError::Tampered {
        path: worktree.to_path_buf(),
        detail: format!("CoW workspace cannot be canonicalized: {error}"),
    })?;
    if actual != expected || !actual.starts_with(&root) {
        return Err(WorkspaceError::Tampered {
            path: control,
            detail: format!(
                "private Git identity mismatch: expected {}, found {}",
                expected.display(),
                actual.display()
            ),
        });
    }
    Ok(())
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
    #[cfg(target_os = "windows")]
    {
        if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local_app_data);
        }
        if let Some(user_profile) = std::env::var_os("USERPROFILE") {
            return PathBuf::from(user_profile).join("AppData").join("Local");
        }
        std::env::temp_dir()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(xdg);
        }
        home_dir().join(".cache")
    }
}

fn home_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(profile);
    }
    if let Some(h) = std::env::var_os("HOME") {
        return PathBuf::from(h);
    }
    std::env::temp_dir()
}

/// Create or reuse `stable_dir` as a detached worktree at `base_commit`.
/// Returns the absolute linked git directory to pin for the lifetime of the run.
fn prepare_stable_worktree(
    repo_root: &Path,
    stable_dir: &Path,
    base_commit: &str,
    fresh: bool,
    prepare_snapshot_source: bool,
) -> Result<PathBuf, WorkspaceError> {
    if let Some(parent) = stable_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    if stable_dir.exists() {
        match try_existing_linked_git_dir(stable_dir, repo_root) {
            Ok(Some(linked)) => {
                // Cross-check identity before any reset. Never run git through a
                // worktree whose pointer no longer matches the main registration.
                match verify_worktree_identity(repo_root, stable_dir, &linked) {
                    Ok(()) => {
                        reset_worktree_pristine(&linked, stable_dir, base_commit, fresh)?;
                        return Ok(linked);
                    }
                    Err(WorkspaceError::Tampered { .. }) => {
                        // Pointer diverged since last run: discard from the MAIN
                        // side and recreate. Do NOT run git through the worktree.
                        eprintln!(
                            "greppy -p: agent worktree {} discarded — its git pointer no longer matched the main repository registration",
                            stable_dir.display()
                        );
                        discard_worktree_from_main(repo_root, stable_dir)?;
                    }
                    Err(e) => return Err(e),
                }
            }
            Ok(None) => {
                // Not registered (or registration unreadable) — discard & recreate.
                discard_worktree_from_main(repo_root, stable_dir)?;
            }
            Err(e) => return Err(e),
        }
    } else if linked_git_dir_from_main(repo_root, stable_dir)?.is_some() {
        // Directory gone (cache purge / user delete) but main still has a
        // registration for this exact path. Prune the stale entry from the MAIN
        // side before attempting `worktree add` — never run git through the
        // (absent) worktree.
        git_ok_cwd(repo_root, &["worktree", "prune"])?;
    }

    prepare_snapshot_source_if_needed(stable_dir, prepare_snapshot_source)?;
    add_worktree_with_stale_recovery(
        repo_root,
        stable_dir,
        base_commit,
        fresh,
        prepare_snapshot_source,
    )
}

fn prepare_snapshot_source_if_needed(path: &Path, enabled: bool) -> Result<(), WorkspaceError> {
    if enabled && !path.exists() {
        greppy_rift_core::prepare_snapshot_source(path).map_err(|error| {
            WorkspaceError::CowUnavailable(format!(
                "failed to prepare native CoW template source {}: {error}",
                path.display()
            ))
        })?;
    }
    Ok(())
}

/// Force-remove a (possibly tampered) worktree using only main-repo commands and
/// plain filesystem removal. Never runs git with the worktree as cwd / via its
/// `.git` pointer.
fn discard_worktree_from_main(repo_root: &Path, worktree: &Path) -> Result<(), WorkspaceError> {
    let wt = path_str(worktree)?.to_string();
    // Preferred: git worktree remove --force from the main side.
    let removed = git_run_cwd(repo_root, &["worktree", "remove", "--force", &wt])
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !removed {
        // Registration may be half-broken after a rewritten `.git` (git refuses
        // remove when the reverse pointer does not match). Wipe the path first
        // with plain filesystem ops (directory *or* a plain file left in its
        // place by a human/cache tool), then prune the main registry so a
        // subsequent `worktree add` can re-register the same path.
        remove_path_for_discard(worktree)?;
        // Propagate prune failure: a silent prune leave-behind can hide the
        // original cause of a subsequent `worktree add` failure.
        git_ok_cwd(repo_root, &["worktree", "prune"])?;
    }
    Ok(())
}

/// Remove a leftover worktree path that may be a directory, a plain file, or a
/// symlink. Used only from the main-side discard path.
fn remove_path_for_discard(worktree: &Path) -> Result<(), WorkspaceError> {
    let meta = match fs::symlink_metadata(worktree) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(WorkspaceError::Io(io::Error::new(
                e.kind(),
                format!(
                    "cannot stat stale agent worktree {}: {e}",
                    worktree.display()
                ),
            )));
        }
    };
    let result = if meta.file_type().is_symlink() || meta.is_file() {
        fs::remove_file(worktree)
    } else {
        fs::remove_dir_all(worktree)
    };
    if let Err(e) = result {
        if worktree.exists() {
            return Err(WorkspaceError::Io(io::Error::new(
                e.kind(),
                format!(
                    "cannot remove stale agent worktree {}: {e}",
                    worktree.display()
                ),
            )));
        }
    }
    Ok(())
}

/// `worktree add` for the stable create path, with a single recovery attempt when
/// the path is still a lingering registration.
///
/// Does **not** use `add -f`. On a stale-registration failure: discard via the
/// main-side helper, then retry the add **exactly once**. Unrelated failures
/// propagate immediately; the retry cannot loop.
fn add_worktree_with_stale_recovery(
    repo_root: &Path,
    worktree: &Path,
    base_commit: &str,
    fresh: bool,
    prepare_snapshot_source: bool,
) -> Result<PathBuf, WorkspaceError> {
    match add_worktree_and_record(repo_root, worktree, base_commit, fresh) {
        Ok(linked) => Ok(linked),
        Err(e) if should_recover_stale_worktree_add(repo_root, worktree, &e) => {
            discard_worktree_from_main(repo_root, worktree)?;
            // Exactly one retry — no loop.
            prepare_snapshot_source_if_needed(worktree, prepare_snapshot_source)?;
            add_worktree_and_record(repo_root, worktree, base_commit, fresh)
        }
        Err(e) => Err(e),
    }
}

/// Decide whether a failed `worktree add` should be recovered by discarding the
/// path and retrying once.
///
/// **Primary (state, locale-independent):** the path is registered in the main
/// repository (authoritative `worktree list` lookup) but its directory is absent
/// or not a directory. Recovery triggers on this state regardless of stderr
/// wording.
///
/// **Secondary (hint only):** English stderr fragments such as "already
/// registered". Git diagnostics are also pinned to the C locale (see
/// [`configure_git_command`]), so the text match is stable when present, but it
/// is never required.
fn should_recover_stale_worktree_add(
    repo_root: &Path,
    worktree: &Path,
    err: &WorkspaceError,
) -> bool {
    if is_registered_without_worktree_dir(repo_root, worktree) {
        return true;
    }
    is_already_registered_worktree_error(err)
}

/// Authoritative state check: main-side registration exists for `worktree`, but
/// the path is missing or is not a directory (plain file, symlink, etc.).
fn is_registered_without_worktree_dir(repo_root: &Path, worktree: &Path) -> bool {
    let registered = match linked_git_dir_from_main(repo_root, worktree) {
        Ok(Some(_)) => true,
        Ok(None) | Err(_) => false,
    };
    if !registered {
        return false;
    }
    match fs::symlink_metadata(worktree) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => true,
        Err(_) => false,
        Ok(m) => !m.file_type().is_dir(),
    }
}

/// Secondary hint: English stderr fragments from a `worktree add` failure.
/// Locale-pinned git (see [`configure_git_command`]) keeps these stable; the
/// state-based check above is authoritative.
fn is_already_registered_worktree_error(err: &WorkspaceError) -> bool {
    match err {
        WorkspaceError::GitFailed {
            command, stderr, ..
        } => {
            let cmd = command.to_ascii_lowercase();
            let msg = stderr.to_ascii_lowercase();
            cmd.contains("worktree")
                && cmd.contains("add")
                && (msg.contains("already registered worktree")
                    || msg.contains("missing but already registered")
                    || msg.contains("already registered"))
        }
        _ => false,
    }
}

fn add_worktree_and_record(
    repo_root: &Path,
    worktree: &Path,
    base_commit: &str,
    fresh: bool,
) -> Result<PathBuf, WorkspaceError> {
    let path_str_s = path_str(worktree)?.to_string();
    git_ok_cwd(
        repo_root,
        &["worktree", "add", "--detach", &path_str_s, base_commit],
    )?;
    // Authoritative pin comes from the MAIN repo registration only.
    //
    // Chosen (Z2): do **not** run `git rev-parse --absolute-git-dir` with the new
    // worktree as cwd (discovery through the worktree). The main-side porcelain
    // lookup below is already authoritative. A create-time cross-check would be
    // redundant with `verify_worktree_identity`, which pins
    // `--git-dir=<recorded> --work-tree=<worktree>` before finish/cleanup/reuse
    // and never records a rediscovered value. Removing the call keeps the create
    // path free of any worktree-cwd git and free of a second source of truth.
    let linked = linked_git_dir_from_main(repo_root, worktree)?.ok_or_else(|| {
        WorkspaceError::GitFailed {
            command: "git worktree list --porcelain".into(),
            stderr: format!(
                "newly added worktree {} is not registered in the main repository",
                worktree.display()
            ),
            status: None,
        }
    })?;
    if fresh {
        reset_worktree_pristine(&linked, worktree, base_commit, true)?;
    }
    Ok(linked)
}

/// Resolve the linked git dir for an existing on-disk worktree **from the main
/// repository's registration only**.
///
/// Returns:
/// - `Ok(Some(gitdir))` when the main repo lists this path with a usable gitdir,
/// - `Ok(None)` when the path is not registered (caller should recreate),
/// - `Err(_)` on main-repo git failures that prevent a trustworthy answer.
///
/// Never reads or runs git through `dir`'s `.git` file.
fn try_existing_linked_git_dir(
    dir: &Path,
    repo_root: &Path,
) -> Result<Option<PathBuf>, WorkspaceError> {
    linked_git_dir_from_main(repo_root, dir)
}

/// Authoritative linked-git-dir lookup: parse `git worktree list --porcelain`
/// **in the main repository** and return the `gitdir` belonging to the entry
/// whose `worktree` path equals `worktree`.
///
/// Requires **exactly one** porcelain block matching `worktree`. Zero or many
/// matching blocks return `None` (same uniqueness rule as the common-dir
/// reverse scan). For a single match that omits an explicit `gitdir` line
/// (older git / main checkout), falls through to a reverse scan of
/// `<common>/worktrees/*/gitdir`.
fn linked_git_dir_from_main(
    repo_root: &Path,
    worktree: &Path,
) -> Result<Option<PathBuf>, WorkspaceError> {
    let list = git_ok_cwd(repo_root, &["worktree", "list", "--porcelain"])?;
    let want = worktree
        .canonicalize()
        .unwrap_or_else(|_| worktree.to_path_buf());

    match porcelain_match_gitdir(&list, &want) {
        PorcelainMatch::None | PorcelainMatch::Ambiguous => Ok(None),
        PorcelainMatch::One(Some(gd)) => {
            let canon = gd.canonicalize().unwrap_or(gd);
            Ok(Some(git_cli_compatible_path(canon)))
        }
        PorcelainMatch::One(None) => {
            // No explicit gitdir line for this entry. Two possibilities:
            // 1. This IS the main checkout (porcelain never emits gitdir for it)
            //    — not an agent worktree; treat as unregistered for reuse.
            // 2. Older git omitted the line — recover via common-dir scan.
            if let Some(from_fs) = linked_git_dir_via_common_scan(repo_root, &want)? {
                return Ok(Some(from_fs));
            }
            Ok(None)
        }
    }
}

/// Outcome of matching a worktree path against porcelain `worktree list` output.
#[derive(Debug, PartialEq, Eq)]
enum PorcelainMatch {
    /// Zero matching blocks.
    None,
    /// Exactly one matching block; `Some` when it carried an explicit `gitdir`.
    One(Option<PathBuf>),
    /// Two or more matching blocks — registration is ambiguous.
    Ambiguous,
}

/// Parse `git worktree list --porcelain` text and collect blocks whose
/// `worktree` path equals `want`. Enforces uniqueness: only exactly one match
/// is accepted.
fn porcelain_match_gitdir(list: &str, want: &Path) -> PorcelainMatch {
    // Porcelain entries are blank-line separated blocks:
    //   worktree <path>
    //   HEAD <oid>
    //   branch <ref> | detached
    //   gitdir <path>            # linked worktrees only (git ≥ 2.36 emits this)
    let mut current_worktree: Option<PathBuf> = None;
    let mut current_gitdir: Option<PathBuf> = None;
    let mut matches: Vec<Option<PathBuf>> = Vec::new();

    let flush =
        |cw: &mut Option<PathBuf>, cg: &mut Option<PathBuf>, matches: &mut Vec<Option<PathBuf>>| {
            if let Some(wt) = cw.take() {
                let got = wt.canonicalize().unwrap_or(wt);
                if path_eq(&got, want) {
                    matches.push(cg.take());
                } else {
                    cg.take();
                }
            } else {
                cg.take();
            }
        };

    for line in list.lines() {
        if line.is_empty() {
            flush(&mut current_worktree, &mut current_gitdir, &mut matches);
            continue;
        }
        if let Some(rest) = line.strip_prefix("worktree ") {
            // Starting a new entry; flush any previous.
            flush(&mut current_worktree, &mut current_gitdir, &mut matches);
            current_worktree = Some(PathBuf::from(rest.trim()));
        } else if let Some(rest) = line.strip_prefix("gitdir ") {
            current_gitdir = Some(PathBuf::from(rest.trim()));
        }
    }
    flush(&mut current_worktree, &mut current_gitdir, &mut matches);

    match matches.len() {
        0 => PorcelainMatch::None,
        1 => PorcelainMatch::One(matches.remove(0)),
        _ => PorcelainMatch::Ambiguous,
    }
}

/// Scan `<common-dir>/worktrees/*/gitdir` for a reverse pointer to `worktree`.
///
/// Requires **exactly one** registration whose recorded `gitdir` file points at
/// this worktree's `.git` path. Zero or multiple matches are treated as "no
/// registration found" so the reuse path discards and recreates rather than
/// picking an arbitrary stale entry.
fn linked_git_dir_via_common_scan(
    repo_root: &Path,
    worktree_canon: &Path,
) -> Result<Option<PathBuf>, WorkspaceError> {
    let common = main_common_dir(repo_root)?;
    let worktrees_dir = common.join("worktrees");
    let Ok(entries) = fs::read_dir(&worktrees_dir) else {
        return Ok(None);
    };
    // Expected reverse-pointer target: `<worktree>/.git` (absolute).
    let expected_git_file = worktree_canon.join(".git");
    let expected_git_file_canon = expected_git_file
        .canonicalize()
        .unwrap_or_else(|_| expected_git_file.clone());

    let mut matches: Vec<PathBuf> = Vec::new();
    for ent in entries.flatten() {
        let gitdir_file = ent.path().join("gitdir");
        let Ok(contents) = fs::read_to_string(&gitdir_file) else {
            continue;
        };
        // Contents are "<worktree-path>/.git\n". Require the recorded path to
        // resolve to THIS worktree's .git (not merely a parent-dir match).
        let pointed = PathBuf::from(contents.trim());
        let pointed_abs = if pointed.is_absolute() {
            pointed
        } else {
            // Relative reverse pointers are rare; resolve against the meta dir.
            ent.path().join(pointed)
        };
        let pointed_canon = pointed_abs
            .canonicalize()
            .unwrap_or_else(|_| pointed_abs.clone());
        // Also accept an un-canonicalized equality when canonicalize fails
        // (e.g. the worktree was just wiped but the registration lingers).
        let points_here = path_eq(&pointed_canon, &expected_git_file_canon)
            || path_eq(&pointed_canon, &expected_git_file)
            || pointed_canon == expected_git_file
            || pointed_canon == expected_git_file_canon;
        if points_here {
            let linked = ent.path();
            matches.push(git_cli_compatible_path(
                linked.canonicalize().unwrap_or(linked),
            ));
        }
    }
    if matches.len() == 1 {
        return Ok(Some(matches.remove(0)));
    }
    // Zero or ambiguous: treat as unregistered so reuse discards + recreates.
    Ok(None)
}

fn main_common_dir(repo_root: &Path) -> Result<PathBuf, WorkspaceError> {
    let raw = git_ok_cwd(repo_root, &["rev-parse", "--git-common-dir"])?;
    let p = resolve_maybe_relative(repo_root, &raw);
    Ok(p.canonicalize().unwrap_or(p))
}

fn path_eq(a: &Path, b: &Path) -> bool {
    // Canonical paths should already be comparable; fall back to component-wise
    // equality when canonicalize failed earlier.
    if a == b {
        return true;
    }
    // On macOS /var vs /private/var can still sneak through if only one side
    // was canonicalized. Try both sides again.
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(aa), Ok(bb)) => aa == bb,
        _ => false,
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
/// `linked_git_dir` recorded at creation (which itself came from the main
/// repository registration). On mismatch: typed `Tampered` and no further git
/// through that tree.
///
/// Asserts all three:
/// (a) the main repo's registration for THIS path yields exactly the recorded
///     git dir;
/// (b) the worktree's `.git` control file points at that same git dir;
/// (c) the recorded git dir exists and lives under the main repo's common dir
///     (`<common-dir>/worktrees/…`).
fn verify_worktree_identity(
    repo_root: &Path,
    worktree: &Path,
    linked_git_dir: &Path,
) -> Result<(), WorkspaceError> {
    let git_file = worktree.join(".git");
    let linked_canon = linked_git_dir
        .canonicalize()
        .unwrap_or_else(|_| linked_git_dir.to_path_buf());

    // (a) Main-repo registration for this path must yield exactly linked_git_dir.
    let registered =
        linked_git_dir_from_main(repo_root, worktree).map_err(|e| WorkspaceError::Tampered {
            path: worktree.to_path_buf(),
            detail: format!("cannot read main repository worktree list: {e}"),
        })?;
    match registered {
        Some(reg) => {
            let reg_canon = reg.canonicalize().unwrap_or(reg);
            if !path_eq(&reg_canon, &linked_canon) {
                return Err(WorkspaceError::Tampered {
                    path: worktree.to_path_buf(),
                    detail: format!(
                        "main registration gitdir is {} but run recorded {}",
                        reg_canon.display(),
                        linked_canon.display()
                    ),
                });
            }
        }
        None => {
            return Err(WorkspaceError::Tampered {
                path: worktree.to_path_buf(),
                detail: "path is no longer a registered worktree of the repository".into(),
            });
        }
    }

    // (b) `.git` must be a regular file pointing at linked_git_dir.
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
    if !path_eq(&pointed_canon, &linked_canon) {
        return Err(WorkspaceError::Tampered {
            path: git_file,
            detail: format!(
                ".git points at {} but run recorded {}",
                pointed_canon.display(),
                linked_canon.display()
            ),
        });
    }

    // (c) Recorded git dir must exist and live under <common>/worktrees/.
    if !linked_canon.is_dir() {
        return Err(WorkspaceError::Tampered {
            path: linked_canon.clone(),
            detail: "recorded linked git dir does not exist or is not a directory".into(),
        });
    }
    let common = main_common_dir(repo_root).map_err(|e| WorkspaceError::Tampered {
        path: git_file.clone(),
        detail: format!("cannot resolve main git-common-dir: {e}"),
    })?;
    let expected_parent = common.join("worktrees");
    let expected_parent_canon = expected_parent
        .canonicalize()
        .unwrap_or_else(|_| expected_parent.clone());
    let parent_ok = linked_canon
        .parent()
        .map(|p| {
            let pc = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
            path_eq(&pc, &expected_parent_canon)
        })
        .unwrap_or(false);
    if !parent_ok {
        return Err(WorkspaceError::Tampered {
            path: linked_canon.clone(),
            detail: format!(
                "recorded linked git dir {} is not under {}/worktrees/",
                linked_canon.display(),
                common.display()
            ),
        });
    }

    // Cross-check only: pinned rev-parse --absolute-git-dir must agree.
    // Never used as the source of truth for recording.
    if let Ok(abs) = git_ok_wt(
        linked_git_dir,
        worktree,
        &["rev-parse", "--absolute-git-dir"],
    ) {
        let abs_path = PathBuf::from(&abs);
        let abs_canon = abs_path.canonicalize().unwrap_or(abs_path);
        if !path_eq(&abs_canon, &linked_canon) {
            return Err(WorkspaceError::Tampered {
                path: git_file,
                detail: format!(
                    "pinned absolute-git-dir is {} but run recorded {}",
                    abs_canon.display(),
                    linked_canon.display()
                ),
            });
        }
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

/// Detach at `base_commit`, hard-reset, and clean untracked (and nested repos).
/// Optionally drops ignored files (`fresh`). Git failures are always propagated.
///
/// Submodules are **not** handled: create refuses repositories whose recorded
/// base commit contains gitlink entries (see [`main_commit_has_gitlinks`]).
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
    Ok(())
}

/// True when `base_commit` in the **main** repository contains any gitlink
/// entry (mode `160000` — a submodule / nested-commit tree entry).
///
/// Detection is pinned to the main repo (`--git-dir` / `--work-tree`) and
/// never reads an agent worktree. Uses recursive `ls-tree -r -z` so nested
/// paths are covered. The commit inspected is the same OID create records as
/// `base_commit` — never a separate `HEAD` lookup.
fn main_commit_has_gitlinks(
    main_git_dir: &Path,
    work_tree: &Path,
    base_commit: &str,
) -> Result<bool, WorkspaceError> {
    let output = git_run_wt(
        main_git_dir,
        work_tree,
        &["ls-tree", "-r", "-z", base_commit],
    )?;
    if !output.status.success() {
        return Err(git_failed(
            &format!(
                "git --git-dir={} --work-tree={} ls-tree -r -z {}",
                main_git_dir.display(),
                work_tree.display(),
                base_commit
            ),
            &output,
        ));
    }
    // Each -z entry is: "<mode> <type> <object>\t<file>\0"
    // Gitlink (submodule) entries have mode 160000.
    for entry in output.stdout.split(|&b| b == 0) {
        if entry.is_empty() {
            continue;
        }
        if entry.starts_with(b"160000 ") {
            return Ok(true);
        }
    }
    Ok(false)
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
    let linked = add_worktree_and_record(repo_root, &worktree, base_commit, fresh)?;
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

fn acquire_template_lock(
    lock_path: &Path,
    timeout: Duration,
) -> Result<Option<FileLock>, WorkspaceError> {
    let started = Instant::now();
    loop {
        if let Some(lock) = try_acquire_lock(lock_path)? {
            return Ok(Some(lock));
        }
        if started.elapsed() >= timeout {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(10));
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

#[cfg(windows)]
fn try_lock_exclusive(file: &File) -> io::Result<bool> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{ERROR_LOCK_VIOLATION, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };

    let mut overlapped = unsafe { std::mem::zeroed() };
    // SAFETY: the handle stays valid for the call, OVERLAPPED is initialized,
    // and the one-byte range remains locked by this handle until Drop.
    let rc = unsafe {
        LockFileEx(
            file.as_raw_handle() as HANDLE,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            &mut overlapped,
        )
    };
    if rc != 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    if err.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
        Ok(false)
    } else {
        Err(err)
    }
}

#[cfg(windows)]
fn unlock_file(file: &File) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::UnlockFileEx;

    let mut overlapped = unsafe { std::mem::zeroed() };
    // SAFETY: best-effort unlock of the same one-byte range held by this handle.
    let _ = unsafe { UnlockFileEx(file.as_raw_handle() as HANDLE, 0, 1, 0, &mut overlapped) };
}

#[cfg(all(not(unix), not(windows)))]
fn try_lock_exclusive(_file: &File) -> io::Result<bool> {
    Ok(true)
}

#[cfg(all(not(unix), not(windows)))]
fn unlock_file(_file: &File) {}

fn path_str(p: &Path) -> Result<&str, WorkspaceError> {
    p.to_str().ok_or_else(|| {
        WorkspaceError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path is not valid UTF-8",
        ))
    })
}

/// Git for Windows rejects Rust's extended-length `\\?\` canonical paths when
/// supplied through `--git-dir`, even though Win32 filesystem APIs accept them.
/// Normalize only the subprocess-facing copy; identity checks keep using their
/// canonical `PathBuf`s.
fn git_cli_compatible_path(path: PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let Some(text) = path.to_str() else {
            return path;
        };
        if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{rest}"));
        }
        if let Some(rest) = text.strip_prefix(r"\\?\") {
            return PathBuf::from(rest);
        }
    }
    path
}

/// Build a `git` Command with diagnostics pinned to the C locale.
///
/// Every production git subprocess in this module must go through this helper
/// (via [`git_run_cwd`] / [`git_run_wt`] or a direct call) so stderr fragments
/// used for classification stay English regardless of the host `LANG` /
/// `LC_ALL` / `LANGUAGE`.
fn git_command() -> Command {
    let mut cmd = Command::new("git");
    configure_git_command(&mut cmd);
    cmd
}

/// Apply the shared env for every git subprocess this module spawns.
fn configure_git_command(cmd: &mut Command) {
    cmd.env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        // Pin diagnostics language so classifiers are not locale-dependent.
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .env_remove("LANGUAGE");
}

/// Git against an ordinary checkout (repo root / apply target): discovery via cwd.
fn git_run_cwd(cwd: &Path, args: &[&str]) -> Result<Output, WorkspaceError> {
    git_command()
        .args(args)
        .current_dir(cwd)
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
    let mut cmd = git_command();
    cmd.arg("--git-dir")
        .arg(path_str(git_dir)?)
        .arg("--work-tree")
        .arg(path_str(work_tree)?)
        .args(args);
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

    fn create_cow_fixture(repo: &Path, run_id: &str) -> Option<AgentWorkspace> {
        match AgentWorkspace::create_with_options(
            repo,
            run_id,
            CreateOptions {
                fresh: true,
                backend: WorkspaceBackend::Cow,
            },
        ) {
            Ok(workspace) => Some(workspace),
            Err(WorkspaceError::CowUnavailable(detail))
                if std::env::var_os("GREPPY_REQUIRE_COW_TESTS").is_none() =>
            {
                eprintln!("skipping native CoW integration test: {detail}");
                None
            }
            Err(error) => panic!("create CoW workspace: {error}"),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn forced_cow_on_unsupported_filesystem_fails_closed() {
        if std::env::var_os("GREPPY_REQUIRE_COW_UNAVAILABLE_TEST").is_none() {
            return;
        }
        let repo = init_fixture("greppy-ws-cow-unavailable");
        let fp_before = main_checkout_fingerprint(&repo);
        let result = AgentWorkspace::create_with_options(
            &repo,
            &unique_tag("run-cow-unavailable"),
            CreateOptions {
                fresh: true,
                backend: WorkspaceBackend::Cow,
            },
        );
        assert!(
            matches!(result, Err(WorkspaceError::CowUnavailable(_))),
            "forced CoW on the hosted runner filesystem must fail closed"
        );
        assert_eq!(main_checkout_fingerprint(&repo), fp_before);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn auto_prefers_stable_then_uses_registered_concurrent_backend() {
        let repo = init_fixture("greppy-ws-auto-policy");
        let holder = AgentWorkspace::create_with_options(
            &repo,
            &unique_tag("run-auto-holder"),
            CreateOptions {
                fresh: true,
                backend: WorkspaceBackend::Auto,
            },
        )
        .expect("first auto workspace");
        assert_eq!(holder.kind, WorktreeKind::Stable);

        let concurrent = AgentWorkspace::create_with_options(
            &repo,
            &unique_tag("run-auto-concurrent"),
            CreateOptions {
                fresh: true,
                backend: WorkspaceBackend::Auto,
            },
        )
        .expect("concurrent auto workspace");
        assert_ne!(concurrent.kind, WorktreeKind::Stable);
        if std::env::var_os("GREPPY_REQUIRE_AUTO_COW_TEST").is_some() {
            assert_eq!(concurrent.kind, WorktreeKind::Cow);
        }
        if std::env::var_os("GREPPY_REQUIRE_AUTO_NATIVE_FALLBACK_TEST").is_some() {
            assert_eq!(concurrent.kind, WorktreeKind::Temp);
        }

        let concurrent_path = concurrent.worktree_path().to_path_buf();
        concurrent
            .cleanup()
            .expect("cleanup concurrent auto workspace");
        assert!(!concurrent_path.exists());
        let stable = holder.worktree_path().to_path_buf();
        holder.cleanup().expect("cleanup stable auto workspace");
        destroy_stable(&repo, &stable);
        let template = cow_template_dir(&repo);
        destroy_stable(&repo, &template);
        let _ = fs::remove_file(cow_template_ready_path(&template));
        let _ = std::fs::remove_dir_all(&repo);
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
            let _ = greppy_rift_core::set_snapshot_source_immutable(ws_path, false);
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
    fn cow_workspace_has_private_git_proposal_and_cleanup() {
        let repo = init_fixture("greppy-ws-cow");
        let main_before = main_checkout_fingerprint(&repo);
        let run_id = unique_tag("run-cow");
        let Some(ws) = create_cow_fixture(&repo, &run_id) else {
            let stable = stable_worktree_dir(&repo);
            destroy_stable(&repo, &stable);
            let _ = std::fs::remove_dir_all(&repo);
            return;
        };
        let worktree = ws.worktree_path().to_path_buf();
        let stable = stable_worktree_dir(&repo);

        assert_eq!(ws.kind, WorktreeKind::Cow);
        assert_ne!(worktree, stable);
        assert!(worktree.join(".git").is_dir());
        assert!(git_c(&worktree, &["status", "--porcelain"]).is_empty());
        assert_eq!(main_checkout_fingerprint(&repo), main_before);

        std::fs::write(worktree.join("hello.txt"), b"changed in CoW\n").unwrap();
        std::fs::write(worktree.join("cow-only.txt"), b"private\n").unwrap();
        git_c(&worktree, &["config", "user.name", "model"]);
        git_c(&worktree, &["config", "user.email", "model@test.local"]);
        git_c(&worktree, &["add", "-A"]);
        git_c(&worktree, &["commit", "-m", "model commit"]);
        let model_commit = git_c(&worktree, &["rev-parse", "HEAD"]);
        assert!(
            !git_run_cwd(&repo, &["cat-file", "-e", &model_commit])
                .unwrap()
                .status
                .success(),
            "model-created objects must remain private before finish"
        );

        let proposal = ws.finish("CoW proposal").expect("finish");
        let (commit, ref_name) = match proposal {
            RunOutcome::Proposal {
                commit, ref_name, ..
            } => (commit, ref_name),
            RunOutcome::Clean => panic!("expected proposal"),
        };
        assert_eq!(git_c(&repo, &["rev-parse", &ref_name]), commit);
        assert_eq!(
            git_c(&repo, &["rev-parse", &format!("{commit}^")]),
            ws.base_commit
        );
        assert_eq!(main_checkout_fingerprint(&repo), main_before);

        ws.cleanup().expect("cleanup CoW workspace");
        assert!(!worktree.exists());
        destroy_stable(&repo, &stable);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn cow_invalidates_tampered_ready_template_and_next_run_recovers() {
        let repo = init_fixture("greppy-ws-cow-template-recovery");
        let Some(first) = create_cow_fixture(&repo, &unique_tag("run-cow-template-first")) else {
            let template = cow_template_dir(&repo);
            destroy_stable(&repo, &template);
            let _ = fs::remove_file(cow_template_ready_path(&template));
            let _ = std::fs::remove_dir_all(&repo);
            return;
        };
        first.cleanup().expect("cleanup first CoW workspace");
        let template = cow_template_dir(&repo);
        let ready = cow_template_ready_path(&template);
        assert!(ready.is_file());
        let capability = greppy_rift_core::probe(
            &template,
            template.parent().expect("template destination root"),
        )
        .expect("probe prepared template");
        let tamper = std::fs::write(template.join("hello.txt"), b"tampered template\n");
        if capability.source_immutable {
            assert!(tamper.is_err(), "sealed template accepted a source write");
            let protected = AgentWorkspace::create_with_options(
                &repo,
                &unique_tag("run-cow-template-protected"),
                CreateOptions {
                    fresh: true,
                    backend: WorkspaceBackend::Cow,
                },
            )
            .expect("sealed template remains usable");
            protected.cleanup().expect("cleanup protected workspace");
            assert!(ready.exists(), "rejected tamper must retain readiness");
        } else {
            tamper.expect("tamper unsealed template");
            let failed = AgentWorkspace::create_with_options(
                &repo,
                &unique_tag("run-cow-template-tampered"),
                CreateOptions {
                    fresh: true,
                    backend: WorkspaceBackend::Cow,
                },
            );
            assert!(matches!(failed, Err(WorkspaceError::CowUnavailable(_))));
            assert!(!ready.exists(), "failed validation must revoke readiness");

            let recovered = AgentWorkspace::create_with_options(
                &repo,
                &unique_tag("run-cow-template-recovered"),
                CreateOptions {
                    fresh: true,
                    backend: WorkspaceBackend::Cow,
                },
            )
            .expect("next CoW creation rebuilds the template");
            assert!(git_c(recovered.worktree_path(), &["status", "--porcelain"]).is_empty());
            recovered.cleanup().expect("cleanup recovered workspace");
        }

        destroy_stable(&repo, &template);
        let _ = fs::remove_file(ready);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[cfg(unix)]
    #[test]
    fn cow_private_git_rewrite_is_rejected_and_tree_is_preserved() {
        let repo = init_fixture("greppy-ws-cow-tamper");
        let main_before = main_checkout_fingerprint(&repo);
        let Some(ws) = create_cow_fixture(&repo, &unique_tag("run-cow-tamper")) else {
            let stable = stable_worktree_dir(&repo);
            destroy_stable(&repo, &stable);
            let _ = std::fs::remove_dir_all(&repo);
            return;
        };
        let worktree = ws.worktree_path().to_path_buf();
        let private_backup = worktree.join(".git-private-backup");
        std::fs::rename(worktree.join(".git"), &private_backup).unwrap();
        std::os::unix::fs::symlink(repo.join(".git"), worktree.join(".git")).unwrap();

        let error = ws.finish("must reject").expect_err("tamper rejection");
        assert!(matches!(error, WorkspaceError::Tampered { .. }));
        assert!(
            worktree.exists(),
            "tampered CoW workspace must be preserved"
        );
        assert_eq!(main_checkout_fingerprint(&repo), main_before);

        std::fs::remove_file(worktree.join(".git")).unwrap();
        std::fs::rename(private_backup, worktree.join(".git")).unwrap();
        let stable = stable_worktree_dir(&repo);
        ws.cleanup().expect("cleanup after restoring private Git");
        destroy_stable(&repo, &stable);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn native_and_cow_proposals_have_identical_trees() {
        let repo = init_fixture("greppy-ws-cow-parity");
        let native = AgentWorkspace::create_with_options(
            &repo,
            &unique_tag("run-native-parity"),
            CreateOptions {
                fresh: true,
                backend: WorkspaceBackend::Native,
            },
        )
        .expect("native workspace");
        std::fs::write(native.worktree_path().join("hello.txt"), b"same edit\n").unwrap();
        std::fs::write(native.worktree_path().join("new.txt"), b"same new file\n").unwrap();
        let native_commit = match native.finish("native parity").unwrap() {
            RunOutcome::Proposal { commit, .. } => commit,
            RunOutcome::Clean => panic!("expected native proposal"),
        };
        let native_tree = git_c(&repo, &["rev-parse", &format!("{native_commit}^{{tree}}")]);
        native.cleanup().unwrap();

        let Some(cow) = create_cow_fixture(&repo, &unique_tag("run-cow-parity")) else {
            let stable = stable_worktree_dir(&repo);
            destroy_stable(&repo, &stable);
            let _ = std::fs::remove_dir_all(&repo);
            return;
        };
        std::fs::write(cow.worktree_path().join("hello.txt"), b"same edit\n").unwrap();
        std::fs::write(cow.worktree_path().join("new.txt"), b"same new file\n").unwrap();
        let cow_commit = match cow.finish("CoW parity").unwrap() {
            RunOutcome::Proposal { commit, .. } => commit,
            RunOutcome::Clean => panic!("expected CoW proposal"),
        };
        let cow_tree = git_c(&repo, &["rev-parse", &format!("{cow_commit}^{{tree}}")]);
        assert_eq!(cow_tree, native_tree);
        let stable = stable_worktree_dir(&repo);
        cow.cleanup().unwrap();
        destroy_stable(&repo, &stable);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn ten_concurrent_cow_workspaces_are_private_and_publish_distinct_refs() {
        use std::sync::{Arc, Barrier};

        let repo = Arc::new(init_fixture("greppy-ws-cow-concurrent"));
        let Some(probe_workspace) =
            create_cow_fixture(&repo, &unique_tag("run-cow-capability-probe"))
        else {
            let stable = stable_worktree_dir(&repo);
            destroy_stable(&repo, &stable);
            let _ = std::fs::remove_dir_all(repo.as_path());
            return;
        };
        probe_workspace.cleanup().expect("cleanup capability probe");
        let main_before = main_checkout_fingerprint(&repo);
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = Vec::new();
        for worker in 0..10 {
            let repo = Arc::clone(&repo);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                let run_id = unique_tag(&format!("run-cow-{worker}"));
                barrier.wait();
                let ws = AgentWorkspace::create_with_options(
                    &repo,
                    &run_id,
                    CreateOptions {
                        fresh: true,
                        backend: WorkspaceBackend::Cow,
                    },
                )
                .expect("concurrent CoW create");
                assert_eq!(ws.kind, WorktreeKind::Cow);
                std::fs::write(
                    ws.worktree_path().join(format!("worker-{worker}.txt")),
                    format!("worker {worker}\n"),
                )
                .unwrap();
                let (commit, ref_name) = match ws.finish("concurrent CoW").unwrap() {
                    RunOutcome::Proposal {
                        commit, ref_name, ..
                    } => (commit, ref_name),
                    RunOutcome::Clean => panic!("expected proposal"),
                };
                let path = ws.worktree_path().to_path_buf();
                ws.cleanup().unwrap();
                assert!(!path.exists());
                (commit, ref_name)
            }));
        }

        let mut refs = Vec::new();
        for handle in handles {
            let (commit, ref_name) = handle.join().expect("worker thread");
            assert_eq!(git_c(&repo, &["rev-parse", &ref_name]), commit);
            refs.push(ref_name);
        }
        refs.sort();
        refs.dedup();
        assert_eq!(refs.len(), 10);
        assert_eq!(main_checkout_fingerprint(&repo), main_before);

        let stable = stable_worktree_dir(&repo);
        destroy_stable(&repo, &stable);
        let _ = std::fs::remove_dir_all(repo.as_path());
    }

    #[test]
    fn fifty_concurrent_cow_workspaces_stress() {
        use std::sync::{Arc, Barrier};

        let repo = Arc::new(init_fixture("greppy-ws-cow-stress-50"));
        let Some(probe) = create_cow_fixture(&repo, &unique_tag("run-cow-stress-probe")) else {
            let template = cow_template_dir(&repo);
            destroy_stable(&repo, &template);
            let _ = fs::remove_file(cow_template_ready_path(&template));
            let _ = std::fs::remove_dir_all(repo.as_path());
            return;
        };
        probe.cleanup().expect("cleanup capability probe");
        let main_before = main_checkout_fingerprint(&repo);
        let barrier = Arc::new(Barrier::new(50));
        let mut handles = Vec::new();
        for worker in 0..50 {
            let repo = Arc::clone(&repo);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let workspace = AgentWorkspace::create_with_options(
                    &repo,
                    &unique_tag(&format!("run-cow-stress-{worker}")),
                    CreateOptions {
                        fresh: true,
                        backend: WorkspaceBackend::Cow,
                    },
                )
                .expect("stress CoW create");
                assert_eq!(workspace.kind, WorktreeKind::Cow);
                assert!(git_c(workspace.worktree_path(), &["status", "--porcelain"]).is_empty());
                let path = workspace.worktree_path().to_path_buf();
                workspace.cleanup().expect("stress CoW cleanup");
                assert!(!path.exists());
            }));
        }
        for handle in handles {
            handle.join().expect("stress worker");
        }
        assert_eq!(main_checkout_fingerprint(&repo), main_before);

        let template = cow_template_dir(&repo);
        destroy_stable(&repo, &template);
        let _ = fs::remove_file(cow_template_ready_path(&template));
        let _ = std::fs::remove_dir_all(repo.as_path());
    }

    #[test]
    #[ignore = "registered 300k-file Btrfs release performance gate"]
    fn cow_large_fixture_warm_creation_meets_registered_gate() {
        let Some(file_count) = std::env::var("GREPPY_COW_PERF_FILES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        else {
            return;
        };
        assert!(file_count > 0, "GREPPY_COW_PERF_FILES must be positive");

        let repo = init_fixture("greppy-ws-cow-perf");
        let fixture = repo.join("large-fixture");
        let files_per_directory = 1_000usize;
        for directory in 0..file_count.div_ceil(files_per_directory) {
            let shard = fixture.join(format!("shard-{directory:04}"));
            fs::create_dir_all(&shard).expect("create performance fixture shard");
            let start = directory * files_per_directory;
            let end = (start + files_per_directory).min(file_count);
            for file in start..end {
                fs::write(shard.join(format!("file-{file:06}.txt")), b"x\n")
                    .expect("write performance fixture file");
            }
        }
        git_c(&repo, &["add", "--all"]);
        git_c(&repo, &["commit", "-m", "large fixture"]);

        let warmup = AgentWorkspace::create_with_options(
            &repo,
            &unique_tag("run-cow-perf-warmup"),
            CreateOptions {
                fresh: true,
                backend: WorkspaceBackend::Cow,
            },
        )
        .expect("warm CoW template");
        assert_eq!(warmup.kind, WorktreeKind::Cow);
        warmup.cleanup().expect("cleanup warmup workspace");

        let template = cow_template_dir(&repo);
        let capability = greppy_rift_core::probe(
            &template,
            template.parent().expect("template destination root"),
        )
        .expect("probe warmed template");
        assert_eq!(capability.backend, greppy_rift_core::Backend::BtrfsSnapshot);
        assert!(capability.constant_time_metadata);

        let sample_count = 10usize;
        let mut samples_ms = Vec::with_capacity(sample_count);
        for sample in 0..sample_count {
            let started = std::time::Instant::now();
            let workspace = AgentWorkspace::create_with_options(
                &repo,
                &unique_tag(&format!("run-cow-perf-{sample}")),
                CreateOptions {
                    fresh: true,
                    backend: WorkspaceBackend::Cow,
                },
            )
            .expect("create warm performance workspace");
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(workspace.kind, WorktreeKind::Cow);
            workspace.cleanup().expect("cleanup performance workspace");
            samples_ms.push(elapsed);
        }
        samples_ms.sort_by(f64::total_cmp);
        let median_ms = (samples_ms[4] + samples_ms[5]) / 2.0;
        let p95_ms = samples_ms[9];
        eprintln!(
            "COW_PERF_JSON {{\"backend\":\"btrfs_snapshot\",\"files\":{file_count},\"samples\":{sample_count},\"median_ms\":{median_ms:.3},\"p95_ms\":{p95_ms:.3}}}"
        );
        assert!(
            median_ms <= 500.0,
            "warm CoW median {median_ms:.3} ms exceeds 500 ms"
        );
        assert!(
            p95_ms <= 1_000.0,
            "warm CoW P95 {p95_ms:.3} ms exceeds 1000 ms"
        );

        destroy_stable(&repo, &template);
        let _ = fs::remove_file(cow_template_ready_path(&template));
        let _ = fs::remove_dir_all(&repo);
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
        // Must be a real detached worktree of this repo (main registration).
        assert!(
            linked_git_dir_from_main(&repo, ws.worktree_path())
                .expect("main list")
                .is_some(),
            "recreated path must be registered under the main repository"
        );
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
        let foreign_fp_before = main_checkout_fingerprint(&foreign);
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
            foreign_fp_before,
            "foreign repo HEAD/index must be unchanged after rewrite attack"
        );
        let foreign_index = git_c(&foreign, &["ls-files"]);
        assert!(
            !foreign_index.lines().any(|l| l == "agent-only.txt"),
            "agent content must not stage into foreign"
        );

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
            CreateOptions {
                fresh: true,
                backend: WorkspaceBackend::Native,
            },
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

    /// Helper: main git-dir + HEAD for fixture gitlink detection assertions.
    fn main_git_and_head(repo: &Path) -> (PathBuf, String) {
        let gd = PathBuf::from(git_c(repo, &["rev-parse", "--absolute-git-dir"]));
        let head = git_c(repo, &["rev-parse", "HEAD"]);
        (gd, head)
    }

    /// Plant a bare gitlink (mode 160000) at `rel_path` without creating
    /// `.gitmodules`. Uses a real commit OID as the link target.
    fn plant_gitlink_no_gitmodules(repo: &Path, rel_path: &str, target_oid: &str) {
        // Three-arg --cacheinfo form is portable across git versions.
        // No .gitmodules file is written.
        git_c(
            repo,
            &[
                "update-index",
                "--add",
                "--cacheinfo",
                "160000",
                target_oid,
                rel_path,
            ],
        );
        git_c(repo, &["commit", "-m", "add bare gitlink"]);
    }

    #[test]
    fn create_refuses_repo_with_tracked_gitmodules() {
        // Real `git submodule add` path still refuses (gitlinks + .gitmodules).
        let repo = init_fixture("greppy-ws-submod-refuse");
        let sub_src = init_fixture("greppy-ws-submod-refuse-src");
        let sub_url = format!("file://{}", sub_src.display());
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

        let (gd, head) = main_git_and_head(&repo);
        assert!(
            main_commit_has_gitlinks(&gd, &repo, &head).expect("detect"),
            "fixture must contain a gitlink at HEAD"
        );

        let stable = stable_worktree_dir(&repo);
        assert!(
            !stable.exists(),
            "precondition: no stable worktree yet for this fixture"
        );

        let err = AgentWorkspace::create(&repo, &unique_tag("run-submod-refuse"))
            .expect_err("create must refuse submodule repos");
        match err {
            WorkspaceError::Unsupported(reason) => {
                assert!(
                    reason.contains("gitlink") || reason.contains("submodule"),
                    "reason={reason}"
                );
            }
            other => panic!("expected Unsupported, got {other}"),
        }

        // Nothing created: no stable worktree, no lock.
        assert!(
            !stable.exists(),
            "refused create must not materialise a stable worktree"
        );
        assert!(
            !stable_lock_path(&stable).exists(),
            "refused create must not leave a lock file"
        );

        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&sub_src);
    }

    #[test]
    fn create_refuses_gitlink_without_gitmodules() {
        // A committed mode-160000 entry with NO .gitmodules must still be refused.
        let repo = init_fixture("greppy-ws-gitlink-no-gm");
        let sub_src = init_fixture("greppy-ws-gitlink-no-gm-src");
        let target_oid = git_c(&sub_src, &["rev-parse", "HEAD"]);
        plant_gitlink_no_gitmodules(&repo, "vendor/lib", &target_oid);

        // Precondition: no .gitmodules, but a gitlink is present.
        assert!(
            !repo.join(".gitmodules").exists(),
            "fixture must not have .gitmodules on disk"
        );
        let ls = git_c(&repo, &["ls-tree", "-r", "HEAD"]);
        assert!(
            ls.lines().any(|l| l.starts_with("160000 ")),
            "fixture must contain a gitlink; ls-tree={ls}"
        );
        assert!(
            !ls.contains(".gitmodules"),
            "fixture tree must not track .gitmodules; ls-tree={ls}"
        );

        let (gd, head) = main_git_and_head(&repo);
        assert!(
            main_commit_has_gitlinks(&gd, &repo, &head).expect("detect"),
            "detection must see the bare gitlink"
        );

        let err = AgentWorkspace::create(&repo, &unique_tag("run-gitlink-no-gm"))
            .expect_err("create must refuse bare-gitlink repos");
        match err {
            WorkspaceError::Unsupported(reason) => {
                assert!(
                    reason.contains("gitlink") || reason.contains("submodule"),
                    "reason={reason}"
                );
            }
            other => panic!("expected Unsupported, got {other}"),
        }

        let stable = stable_worktree_dir(&repo);
        assert!(!stable.exists(), "refused create must not create worktree");
        assert!(
            !stable_lock_path(&stable).exists(),
            "refused create must not leave a lock"
        );

        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&sub_src);
    }

    #[test]
    fn create_refuses_gitlink_in_subdirectory() {
        // Nested path (subdir/deep/mod) must be caught by recursive ls-tree -r.
        let repo = init_fixture("greppy-ws-gitlink-nested");
        let sub_src = init_fixture("greppy-ws-gitlink-nested-src");
        let target_oid = git_c(&sub_src, &["rev-parse", "HEAD"]);
        plant_gitlink_no_gitmodules(&repo, "subdir/deep/mod", &target_oid);

        let (gd, head) = main_git_and_head(&repo);
        assert!(
            main_commit_has_gitlinks(&gd, &repo, &head).expect("detect"),
            "nested gitlink must be detected"
        );

        let err = AgentWorkspace::create(&repo, &unique_tag("run-gitlink-nested"))
            .expect_err("create must refuse nested-gitlink repos");
        match err {
            WorkspaceError::Unsupported(reason) => {
                assert!(
                    reason.contains("gitlink") || reason.contains("submodule"),
                    "reason={reason}"
                );
            }
            other => panic!("expected Unsupported, got {other}"),
        }

        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&sub_src);
    }

    #[test]
    fn create_unaffected_without_submodules() {
        let repo = init_fixture("greppy-ws-no-submod");
        let (gd, head) = main_git_and_head(&repo);
        assert!(!main_commit_has_gitlinks(&gd, &repo, &head).expect("detect"));
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-no-submod")).expect("create");
        // Detection commit equals the recorded base_commit.
        assert_eq!(
            ws.base_commit(),
            head,
            "recorded base_commit must equal the HEAD used for detection"
        );
        let wt = ws.worktree_path().to_path_buf();
        assert!(wt.join("hello.txt").is_file());
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn detection_commit_equals_recorded_base_commit() {
        // Even with a gitlink present, the OID used for detection is the same
        // value that would have been recorded (create refuses before writing).
        let repo = init_fixture("greppy-ws-detect-oid");
        let sub_src = init_fixture("greppy-ws-detect-oid-src");
        let target_oid = git_c(&sub_src, &["rev-parse", "HEAD"]);
        plant_gitlink_no_gitmodules(&repo, "ext", &target_oid);

        let head = git_c(&repo, &["rev-parse", "HEAD"]);
        let (gd, detect_oid) = main_git_and_head(&repo);
        assert_eq!(detect_oid, head, "helper HEAD must match rev-parse HEAD");
        assert!(
            main_commit_has_gitlinks(&gd, &repo, &detect_oid).expect("detect"),
            "must detect gitlink at that OID"
        );
        // Create refuses; if it had proceeded, base_commit would be that OID.
        let err =
            AgentWorkspace::create(&repo, &unique_tag("run-detect-oid")).expect_err("must refuse");
        assert!(matches!(err, WorkspaceError::Unsupported(_)));

        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&sub_src);
    }

    #[test]
    fn porcelain_match_enforces_zero_one_many() {
        // Pure parser test: zero / one / many matching porcelain blocks.
        let want = PathBuf::from("/tmp/greppy-agent-wt-unique");
        let want_s = want.to_string_lossy();

        // Zero matches.
        let zero = "\
worktree /tmp/other-wt
HEAD abc
detached
gitdir /tmp/other.git

";
        assert_eq!(
            porcelain_match_gitdir(zero, &want),
            PorcelainMatch::None,
            "zero matching blocks => None"
        );

        // Exactly one match with explicit gitdir.
        let one = format!(
            "\
worktree {want_s}
HEAD abc
detached
gitdir /tmp/linked-meta

worktree /tmp/other-wt
HEAD def
detached
gitdir /tmp/other.git

"
        );
        match porcelain_match_gitdir(&one, &want) {
            PorcelainMatch::One(Some(gd)) => {
                assert_eq!(gd, PathBuf::from("/tmp/linked-meta"));
            }
            other => panic!("expected One(Some), got {other:?}"),
        }

        // Exactly one match without gitdir line.
        let one_no_gd = format!(
            "\
worktree {want_s}
HEAD abc
detached

"
        );
        assert_eq!(
            porcelain_match_gitdir(&one_no_gd, &want),
            PorcelainMatch::One(None),
            "single match without gitdir => One(None)"
        );

        // Many matches: last-wins must NOT apply; treat as Ambiguous.
        let many = format!(
            "\
worktree {want_s}
HEAD abc
detached
gitdir /tmp/first-meta

worktree {want_s}
HEAD def
detached
gitdir /tmp/second-meta

"
        );
        assert_eq!(
            porcelain_match_gitdir(&many, &want),
            PorcelainMatch::Ambiguous,
            "multiple matching blocks => Ambiguous (not last-wins)"
        );
    }

    #[test]
    fn worktree_prune_failure_is_propagated() {
        // discard_worktree_from_main must surface prune errors rather than
        // swallowing them. Point repo_root at a real directory that is not a
        // git repo: worktree remove fails, then prune is attempted and must
        // propagate the failure (previously the Result was ignored).
        let bogus = std::env::temp_dir().join(unique_tag("greppy-ws-prune-fail"));
        fs::create_dir_all(&bogus).unwrap();
        let missing_wt = bogus.join("no-such-worktree");
        let err = discard_worktree_from_main(&bogus, &missing_wt)
            .expect_err("prune against non-repo must fail");
        match err {
            WorkspaceError::GitFailed { command, .. } => {
                assert!(
                    command.contains("prune") || command.contains("worktree"),
                    "expected prune/worktree failure, got command={command}"
                );
            }
            other => panic!("expected GitFailed from prune, got {other}"),
        }
        let _ = fs::remove_dir_all(&bogus);
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

    // --- WP22/WP23: reuse-time identity from main; unique registration; temp cleanup ---

    #[test]
    fn common_scan_rejects_ambiguous_duplicate_registrations() {
        // Two <common>/worktrees/*/gitdir entries both pointing at the same
        // worktree's .git must yield Ok(None) (treat as unregistered) so reuse
        // discards and recreates rather than picking one at random.
        let repo = init_fixture("greppy-ws-ambig-reg");
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-ambig")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        let linked = ws.linked_git_dir().to_path_buf();
        let wt_canon = wt.canonicalize().unwrap_or_else(|_| wt.clone());
        drop(ws);

        // Real registration must currently resolve uniquely.
        let once = linked_git_dir_via_common_scan(&repo, &wt_canon)
            .expect("scan")
            .expect("exactly one registration before ambiguity");
        assert_eq!(
            once.canonicalize().unwrap_or(once),
            linked.canonicalize().unwrap_or(linked.clone())
        );

        // Plant a second metadata directory with the same reverse pointer.
        let common = main_common_dir(&repo).expect("common");
        let spoof = common.join("worktrees").join(unique_tag("spoof"));
        fs::create_dir_all(&spoof).unwrap();
        let real_reverse_pointer = fs::read_to_string(linked.join("gitdir")).unwrap();
        fs::write(spoof.join("gitdir"), real_reverse_pointer).unwrap();

        let after = linked_git_dir_via_common_scan(&repo, &wt_canon).expect("scan");
        assert!(
            after.is_none(),
            "ambiguous reverse-pointer match must be rejected as no registration; got {after:?}"
        );

        // Cleanup spoof + real tree.
        let _ = fs::remove_dir_all(&spoof);
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn reuse_detects_rewritten_git_to_main_and_recreates_without_touching_main() {
        let repo = init_fixture("greppy-ws-reuse-main");
        let fp_before = main_checkout_fingerprint(&repo);
        let main_git = git_c(&repo, &["rev-parse", "--absolute-git-dir"]);
        let run_id = unique_tag("run-reuse-main");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        let linked_before = ws.linked_git_dir().to_path_buf();
        assert!(linked_before.is_dir());
        // Drop without cleanup so the stable tree (and its lock) is free for reuse.
        drop(ws);

        // Rewrite .git to the MAIN checkout's .git — deferred attack on next run.
        std::fs::write(wt.join(".git"), format!("gitdir: {main_git}\n")).unwrap();
        std::fs::write(wt.join("agent-only.txt"), b"should not stage into main\n").unwrap();

        let ws2 = AgentWorkspace::create(&repo, &unique_tag("run-reuse-main-2")).expect("recreate");
        assert_eq!(ws2.worktree_path(), &wt);
        // New registration: linked dir must again live under <common>/worktrees/.
        let linked2 = ws2.linked_git_dir().to_path_buf();
        assert!(
            linked2.is_dir(),
            "recreated linked git dir must exist: {}",
            linked2.display()
        );
        let common = git_c(&repo, &["rev-parse", "--git-common-dir"]);
        let common_p = PathBuf::from(&common);
        let common_abs = if common_p.is_absolute() {
            common_p
        } else {
            repo.join(common_p)
        };
        let common_canon = common_abs.canonicalize().unwrap_or(common_abs);
        let parent = linked2.parent().expect("parent");
        assert_eq!(
            parent
                .canonicalize()
                .unwrap_or_else(|_| parent.to_path_buf()),
            common_canon.join("worktrees"),
            "recreated linked dir must live under common/worktrees"
        );
        // Worktree content recreated from HEAD — agent-only gone, hello present.
        assert!(wt.join("hello.txt").is_file());
        assert!(
            !wt.join("agent-only.txt").exists(),
            "agent-only must not survive discard+recreate"
        );
        // .git pointer must again target the registration under
        // <common>/worktrees/, not the main absolute-git-dir itself.
        let git_contents = std::fs::read_to_string(wt.join(".git")).unwrap();
        let pointed = parse_gitdir_pointer(&git_contents).expect("gitdir pointer");
        let pointed_canon = if pointed.is_absolute() {
            pointed.canonicalize().unwrap_or(pointed)
        } else {
            let p = wt.join(&pointed);
            p.canonicalize().unwrap_or(p)
        };
        let main_git_canon = PathBuf::from(main_git.trim())
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(main_git.trim()));
        assert_ne!(
            pointed_canon, main_git_canon,
            ".git must not point at the main absolute-git-dir; contents={git_contents:?}"
        );
        assert_eq!(
            pointed_canon,
            linked2.canonicalize().unwrap_or(linked2.clone()),
            ".git must point at the recreated linked git dir"
        );

        assert_eq!(
            main_checkout_fingerprint(&repo),
            fp_before,
            "main checkout HEAD/symbolic-HEAD/index must be unchanged after reuse recreate"
        );
        let index = git_c(&repo, &["ls-files"]);
        assert!(
            !index.lines().any(|l| l == "agent-only.txt"),
            "agent-only.txt must not be staged in main index"
        );

        ws2.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn reuse_detects_rewritten_git_to_foreign_and_leaves_foreign_untouched() {
        let repo = init_fixture("greppy-ws-reuse-foreign");
        let foreign = init_fixture("greppy-ws-reuse-foreign-tgt");
        let fp_before = main_checkout_fingerprint(&repo);
        let foreign_fp = main_checkout_fingerprint(&foreign);
        let foreign_git = git_c(&foreign, &["rev-parse", "--absolute-git-dir"]);

        let ws = AgentWorkspace::create(&repo, &unique_tag("run-reuse-foreign")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        drop(ws);

        std::fs::write(wt.join(".git"), format!("gitdir: {foreign_git}\n")).unwrap();
        std::fs::write(wt.join("agent-only.txt"), b"payload\n").unwrap();

        let ws2 =
            AgentWorkspace::create(&repo, &unique_tag("run-reuse-foreign-2")).expect("recreate");
        assert_eq!(ws2.worktree_path(), &wt);
        assert!(wt.join("hello.txt").is_file());
        assert!(!wt.join("agent-only.txt").exists());

        assert_eq!(main_checkout_fingerprint(&repo), fp_before);
        assert_eq!(
            main_checkout_fingerprint(&foreign),
            foreign_fp,
            "foreign repo must be untouched by reuse recovery"
        );
        let foreign_index = git_c(&foreign, &["ls-files"]);
        assert!(!foreign_index.lines().any(|l| l == "agent-only.txt"));

        ws2.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&foreign);
    }

    #[test]
    fn finish_tampered_during_run_keeps_tree_and_surfaces_error() {
        // Same-run tamper is still Tampered (not silent recreate): finish refuses.
        let repo = init_fixture("greppy-ws-finish-keep");
        let fp_before = main_checkout_fingerprint(&repo);
        let main_git = git_c(&repo, &["rev-parse", "--absolute-git-dir"]);
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-finish-keep")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        std::fs::write(wt.join(".git"), format!("gitdir: {main_git}\n")).unwrap();
        let err = ws.finish("x").expect_err("must Tampered");
        match err {
            WorkspaceError::Tampered { path, .. } => {
                assert!(path.ends_with(".git") || path == wt, "path={path:?}");
            }
            other => panic!("expected Tampered, got {other}"),
        }
        assert!(wt.exists(), "Tampered must keep the tree for inspection");
        assert_eq!(main_checkout_fingerprint(&repo), fp_before);
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn cleanup_temp_tampered_keeps_tree_and_errors() {
        let repo = init_fixture("greppy-ws-temp-tamper");
        let fp_before = main_checkout_fingerprint(&repo);
        // Hold the stable lock so create falls back to a temp worktree.
        let stable = stable_worktree_dir(&repo);
        let lock_path = stable_lock_path(&stable);
        let held = try_acquire_lock(&lock_path)
            .expect("lock io")
            .expect("must acquire lock");
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-temp-tamper")).expect("temp");
        assert!(!ws.is_stable(), "must be temp fallback");
        let wt = ws.worktree_path().to_path_buf();
        let main_git = git_c(&repo, &["rev-parse", "--absolute-git-dir"]);
        std::fs::write(wt.join(".git"), format!("gitdir: {main_git}\n")).unwrap();

        let err = ws.cleanup().expect_err("temp cleanup must Tampered");
        match err {
            WorkspaceError::Tampered { .. } => {}
            other => panic!("expected Tampered, got {other}"),
        }
        assert!(
            wt.exists(),
            "tampered temp worktree must be kept for inspection"
        );
        assert_eq!(main_checkout_fingerprint(&repo), fp_before);

        // Manual cleanup of the abandoned temp tree via main-side discard.
        discard_worktree_from_main(&repo, &wt).expect("manual discard");
        drop(held);
        let _ = fs::remove_file(&lock_path);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn cleanup_stable_tampered_keeps_tree_and_errors() {
        let repo = init_fixture("greppy-ws-stable-cleanup-tamper");
        let fp_before = main_checkout_fingerprint(&repo);
        let ws =
            AgentWorkspace::create(&repo, &unique_tag("run-stable-cleanup-tamper")).expect("c");
        let wt = ws.worktree_path().to_path_buf();
        let main_git = git_c(&repo, &["rev-parse", "--absolute-git-dir"]);
        std::fs::write(wt.join(".git"), format!("gitdir: {main_git}\n")).unwrap();
        let err = ws.cleanup().expect_err("stable cleanup must Tampered");
        match err {
            WorkspaceError::Tampered { path, detail } => {
                assert!(!detail.is_empty());
                assert!(path.ends_with(".git") || path == wt, "path={path:?}");
            }
            other => panic!("expected Tampered, got {other}"),
        }
        assert!(wt.exists(), "tampered stable tree kept");
        assert_eq!(main_checkout_fingerprint(&repo), fp_before);
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn linked_git_dir_comes_from_main_registration_not_worktree_rev_parse() {
        let repo = init_fixture("greppy-ws-pin-source");
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-pin-source")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        let linked = ws.linked_git_dir().to_path_buf();
        // Main registration must agree.
        let from_main = linked_git_dir_from_main(&repo, &wt)
            .expect("list")
            .expect("registered");
        assert_eq!(
            linked.canonicalize().unwrap_or(linked.clone()),
            from_main.canonicalize().unwrap_or(from_main),
            "pinned linked_git_dir must equal main registration"
        );
        // And it must live under common/worktrees.
        let common = main_common_dir(&repo).expect("common");
        assert!(
            linked.starts_with(common.join("worktrees"))
                || linked
                    .canonicalize()
                    .unwrap()
                    .starts_with(common.join("worktrees").canonicalize().unwrap()),
            "linked={} common={}",
            linked.display(),
            common.display()
        );
        ws.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    // --- WP25: recover from a stale worktree registration ---

    #[test]
    fn create_recovers_when_registration_lingers_after_directory_deleted() {
        // Production-realistic: stable worktree dir deleted (cache purge / human)
        // while main still has the registration. Create must prune + re-add.
        let repo = init_fixture("greppy-ws-stale-missing");
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-stale-missing")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        assert!(ws.is_stable());
        drop(ws);

        // Confirm registration is present, then wipe only the directory (leave
        // the main-repo registration intact — the bug scenario).
        assert!(
            linked_git_dir_from_main(&repo, &wt)
                .expect("list")
                .is_some(),
            "precondition: registration must still exist before delete"
        );
        fs::remove_dir_all(&wt).expect("delete worktree directory");
        assert!(!wt.exists());
        assert!(
            linked_git_dir_from_main(&repo, &wt)
                .expect("list after delete")
                .is_some(),
            "precondition: registration must linger after directory delete"
        );

        let ws2 =
            AgentWorkspace::create(&repo, &unique_tag("run-stale-missing-2")).expect("recover");
        assert_eq!(ws2.worktree_path(), &wt);
        assert!(ws2.is_stable());
        assert!(
            wt.join("hello.txt").is_file(),
            "recovered worktree must be usable"
        );
        assert!(
            linked_git_dir_from_main(&repo, &wt)
                .expect("list")
                .is_some(),
            "recovered worktree must be registered"
        );
        // Host-side identity must verify (usable for finish/cleanup).
        ws2.verify_identity()
            .expect("recovered worktree identity must verify");
        let outcome = ws2.finish("no changes").expect("finish");
        assert_eq!(outcome, RunOutcome::Clean);
        ws2.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn create_recovers_when_registration_lingers_and_path_is_plain_file() {
        // Directory replaced by a plain file (human / cache tool mishap) while
        // registration still points at that path. Discard must remove the file
        // (not only directories) and recreate.
        let repo = init_fixture("greppy-ws-stale-file");
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-stale-file")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        drop(ws);

        assert!(
            linked_git_dir_from_main(&repo, &wt)
                .expect("list")
                .is_some(),
            "precondition: registration present"
        );
        fs::remove_dir_all(&wt).expect("remove worktree dir");
        // Plant a plain file at the stable path (not a directory).
        fs::write(&wt, b"not a worktree\n").expect("plant plain file");
        assert!(wt.is_file(), "precondition: path is a plain file");

        let ws2 = AgentWorkspace::create(&repo, &unique_tag("run-stale-file-2")).expect("recover");
        assert_eq!(ws2.worktree_path(), &wt);
        assert!(
            wt.is_dir(),
            "recovered path must be a real worktree directory"
        );
        assert!(wt.join("hello.txt").is_file());
        assert!(linked_git_dir_from_main(&repo, &wt)
            .expect("list")
            .is_some());
        ws2.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn already_registered_error_classifier_and_unrelated_add_failure() {
        // Secondary stderr-hint recognises registered-path failure messages;
        // recovery is non-recursive (calls add once more, never itself) so
        // retry cannot loop. Unrelated worktree-add failures still propagate.
        assert!(is_already_registered_worktree_error(&WorkspaceError::GitFailed {
            command: "git worktree add --detach /tmp/x abc".into(),
            stderr: "fatal: '/tmp/x' is a missing but already registered worktree; use 'add -f' to override, or 'prune' or 'remove' to clear\n".into(),
            status: Some(128),
        }));
        assert!(is_already_registered_worktree_error(
            &WorkspaceError::GitFailed {
                command: "git worktree add --detach /tmp/x abc".into(),
                stderr: "fatal: '/tmp/x' is already registered as a worktree\n".into(),
                status: Some(128),
            }
        ));
        // Unrelated failures must NOT be classified as registered-path.
        assert!(!is_already_registered_worktree_error(
            &WorkspaceError::GitFailed {
                command: "git worktree add --detach /tmp/x not-a-commit".into(),
                stderr: "fatal: invalid reference: not-a-commit\n".into(),
                status: Some(128),
            }
        ));
        assert!(!is_already_registered_worktree_error(
            &WorkspaceError::GitFailed {
                command: "git worktree prune".into(),
                stderr: "fatal: not a git repository\n".into(),
                status: Some(128),
            }
        ));
        assert!(!is_already_registered_worktree_error(&WorkspaceError::Io(
            io::Error::other("boom")
        )));

        // Unrelated add failure propagates (invalid base commit OID).
        let repo = init_fixture("greppy-ws-add-fail");
        let wt = stable_worktree_dir(&repo);
        if let Some(parent) = wt.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        // Ensure no leftover registration/dir at the stable path.
        destroy_stable(&repo, &wt);
        let err = add_worktree_with_stale_recovery(
            &repo,
            &wt,
            "0000000000000000000000000000000000000000",
            false,
            false,
        )
        .expect_err("invalid base must fail");
        match &err {
            WorkspaceError::GitFailed {
                command, stderr, ..
            } => {
                assert!(
                    command.contains("worktree") && command.contains("add"),
                    "command={command}"
                );
                assert!(
                    !is_already_registered_worktree_error(&err),
                    "must be unrelated failure; stderr={stderr}"
                );
                assert!(
                    !should_recover_stale_worktree_add(&repo, &wt, &err),
                    "unrelated failure must not trigger recovery; stderr={stderr}"
                );
            }
            other => panic!("expected GitFailed, got {other}"),
        }
        assert!(!wt.exists(), "failed add must not leave a worktree dir");
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn state_based_stale_registration_triggers_recovery_without_english_stderr() {
        // Primary classifier is state, not wording: registration present + path
        // absent/non-directory => recover, even when the GitFailed stderr is
        // empty or non-English (locale-independent).
        let repo = init_fixture("greppy-ws-state-stale");
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-state-stale")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        drop(ws);

        assert!(
            linked_git_dir_from_main(&repo, &wt)
                .expect("list")
                .is_some(),
            "precondition: registration present"
        );
        fs::remove_dir_all(&wt).expect("remove worktree dir");
        assert!(!wt.exists(), "precondition: directory absent");
        assert!(
            is_registered_without_worktree_dir(&repo, &wt),
            "state classifier must see registration without directory"
        );

        // Synthetic non-English / empty stderr: text match must NOT be required.
        let non_english = WorkspaceError::GitFailed {
            command: "git worktree add --detach /tmp/x abc".into(),
            stderr: "fatal: «chemin déjà enregistré»\n".into(),
            status: Some(128),
        };
        assert!(
            !is_already_registered_worktree_error(&non_english),
            "secondary text match must not fire on non-English wording"
        );
        assert!(
            should_recover_stale_worktree_add(&repo, &wt, &non_english),
            "state-based recovery must trigger without English stderr"
        );
        let empty_stderr = WorkspaceError::GitFailed {
            command: "git worktree add --detach /tmp/x abc".into(),
            stderr: String::new(),
            status: Some(128),
        };
        assert!(should_recover_stale_worktree_add(&repo, &wt, &empty_stderr));

        // End-to-end: create recovers via the state path.
        let ws2 = AgentWorkspace::create(&repo, &unique_tag("run-state-stale-2")).expect("recover");
        assert_eq!(ws2.worktree_path(), &wt);
        assert!(wt.is_dir());
        assert!(wt.join("hello.txt").is_file());
        ws2.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn git_command_helper_pins_locale_env() {
        // Every production git Command is built through git_command() /
        // configure_git_command so diagnostics stay English.
        let cmd = git_command();
        let envs: std::collections::HashMap<String, String> = cmd
            .get_envs()
            .filter_map(|(k, v)| {
                let key = k.to_str()?.to_string();
                match v {
                    Some(val) => Some((key, val.to_str()?.to_string())),
                    None => Some((key, String::new())), // env_remove marks as cleared
                }
            })
            .collect();
        assert_eq!(envs.get("LC_ALL").map(String::as_str), Some("C"));
        assert_eq!(envs.get("LANG").map(String::as_str), Some("C"));
        // LANGUAGE must be cleared (present as key with None in Command; we map
        // that to empty string above, or it may be absent depending on platform
        // representation — either way it must not carry a non-empty value).
        match envs.get("LANGUAGE") {
            None => {}
            Some(v) => assert!(v.is_empty(), "LANGUAGE must be cleared, got {v:?}"),
        }
        assert_eq!(
            envs.get("GIT_TERMINAL_PROMPT").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            envs.get("GIT_CONFIG_NOSYSTEM").map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn stale_recovery_retries_add_at_most_once() {
        // Structural guarantee: add_worktree_with_stale_recovery calls
        // add_worktree_and_record at most twice (initial + one retry), never
        // re-enters itself. Simulate a path that stays "registered" after
        // discard by using a non-repo directory as repo_root: discard's prune
        // fails and is propagated *before* a second add — still no loop.
        // Separately, when discard succeeds but the second add fails for a
        // registered-path reason is not reachable in practice (discard clears
        // the registration); the classifier + non-recursive call graph is the
        // contract.
        //
        // Practical check: after a successful recover from a missing dir, a
        // second create reuses cleanly (no repeated discard loop / no hang).
        let repo = init_fixture("greppy-ws-retry-once");
        let ws = AgentWorkspace::create(&repo, &unique_tag("run-retry-once")).expect("create");
        let wt = ws.worktree_path().to_path_buf();
        drop(ws);
        fs::remove_dir_all(&wt).expect("delete");
        // First recover.
        let ws2 = AgentWorkspace::create(&repo, &unique_tag("run-retry-once-2")).expect("recover");
        assert_eq!(ws2.worktree_path(), &wt);
        drop(ws2);
        // Second create reuses the live tree (no stale-registration path).
        let ws3 = AgentWorkspace::create(&repo, &unique_tag("run-retry-once-3")).expect("reuse");
        assert_eq!(ws3.worktree_path(), &wt);
        assert!(wt.join("hello.txt").is_file());
        ws3.cleanup().expect("cleanup");
        destroy_stable(&repo, &wt);
        let _ = std::fs::remove_dir_all(&repo);
    }
}
