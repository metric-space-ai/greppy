//! Per-run isolation: disposable git worktrees and review-patch proposals.
//!
//! Every agent run works in a detached worktree of the target repo. The agent
//! never writes to the user's checkout. The run's outcome is a **proposal**: a
//! commit preserved on `refs/greppy/agent/<run_id>` plus a printable patch.
//! Applying it to a real checkout is a separate, explicit host-side step.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Disposable worktree isolation for a single agent run.
#[derive(Debug, Clone)]
pub struct AgentWorkspace {
    repo_root: PathBuf,
    worktree: PathBuf,
    run_id: String,
    base_commit: String,
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

impl AgentWorkspace {
    /// Create a detached worktree for `run_id` from `repo_root`'s `HEAD`.
    ///
    /// The user's checkout state is irrelevant — the worktree is always created
    /// from `HEAD`. The worktree lives under
    /// `std::env::temp_dir()/greppy-agent/<run_id>`.
    pub fn create(repo_root: &Path, run_id: &str) -> Result<Self, WorkspaceError> {
        // Verify repo_root is inside a git work tree and capture the toplevel.
        let toplevel = match git_ok(repo_root, &["rev-parse", "--show-toplevel"]) {
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

        let base_commit = git_ok(&toplevel, &["rev-parse", "HEAD"])?;

        let worktree = std::env::temp_dir().join("greppy-agent").join(run_id);
        if let Some(parent) = worktree.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Refuse to clobber an existing path — create must be clean.
        if worktree.exists() {
            return Err(WorkspaceError::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("worktree path already exists: {}", worktree.display()),
            )));
        }

        // Detached HEAD at the recorded base commit. Use the OID (not the
        // symbolic HEAD) so the worktree is unambiguously pinned.
        git_ok(
            &toplevel,
            &[
                "worktree",
                "add",
                "--detach",
                worktree.to_str().ok_or_else(|| {
                    WorkspaceError::Io(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "worktree path is not valid UTF-8",
                    ))
                })?,
                &base_commit,
            ],
        )?;

        Ok(Self {
            repo_root: toplevel,
            worktree,
            run_id: run_id.to_string(),
            base_commit,
        })
    }

    /// Absolute path of the disposable worktree (becomes [`crate::GreppyEnv`]'s root).
    pub fn worktree_path(&self) -> &Path {
        &self.worktree
    }

    /// Repository toplevel the worktree was created from.
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// Run id used for the worktree directory and the durable proposal ref.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// `HEAD` OID recorded at create time.
    pub fn base_commit(&self) -> &str {
        &self.base_commit
    }

    /// Durable ref name for this run's proposal (`refs/greppy/agent/<run_id>`).
    pub fn ref_name(&self) -> String {
        format!("refs/greppy/agent/{}", self.run_id)
    }

    /// Stage everything in the worktree and either return [`RunOutcome::Clean`]
    /// or pin a single proposal commit (parent = [`Self::base_commit`]) via
    /// plumbing so model-made commits/resets cannot corrupt the proposal.
    pub fn finish(&self, message: &str) -> Result<RunOutcome, WorkspaceError> {
        git_ok(&self.worktree, &["add", "-A"])?;

        // Capture the final filesystem state as a tree, independent of HEAD.
        let tree = git_ok(&self.worktree, &["write-tree"])?;
        let base_tree = git_ok(
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
            .args(["commit-tree", &tree, "-p", &self.base_commit, "-m", message])
            .current_dir(&self.worktree)
            .env("GIT_AUTHOR_NAME", "greppy agent")
            .env("GIT_AUTHOR_EMAIL", "agent@greppy.local")
            .env("GIT_COMMITTER_NAME", "greppy agent")
            .env("GIT_COMMITTER_EMAIL", "agent@greppy.local")
            // Neutralise a user template/hooks that might interfere in tests.
            .env("GIT_CONFIG_COUNT", "1")
            .env("GIT_CONFIG_KEY_0", "commit.gpgsign")
            .env("GIT_CONFIG_VALUE_0", "false")
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
        git_ok(&self.repo_root, &["update-ref", &ref_name, &commit])?;

        let patch = git_ok(&self.repo_root, &["show", "--format=", "--patch", &commit])?;
        let stat = git_ok(&self.repo_root, &["show", "--format=", "--stat", &commit])?;

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
        let status = git_run(target_checkout, &["status", "--porcelain=v1", "-z"])?;
        if !status.status.success() {
            return Err(git_failed("git status --porcelain=v1 -z", &status));
        }
        if !status.stdout.is_empty() {
            return Err(WorkspaceError::DirtyTarget {
                ref_name: self.ref_name(),
                detail: "git status --porcelain is non-empty".into(),
            });
        }

        let result = git_run(target_checkout, &["cherry-pick", "--no-commit", commit])?;
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
        let has_cp_head = git_run(
            target_checkout,
            &["rev-parse", "-q", "--verify", "CHERRY_PICK_HEAD"],
        )?
        .status
        .success();

        let mut restore_notes = Vec::new();
        if has_cp_head {
            let abort = git_run(target_checkout, &["cherry-pick", "--abort"])?;
            if !abort.status.success() {
                restore_notes.push(format!(
                    "cherry-pick --abort failed (exit {:?}): {}",
                    abort.status.code(),
                    String::from_utf8_lossy(&abort.stderr).trim()
                ));
            }
        } else {
            let reset = git_run(target_checkout, &["reset", "--merge"])?;
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

    /// Force-remove the disposable worktree. Proposal refs are **not** deleted.
    pub fn cleanup(self) -> Result<(), WorkspaceError> {
        let wt = self
            .worktree
            .to_str()
            .ok_or_else(|| {
                WorkspaceError::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "worktree path is not valid UTF-8",
                ))
            })?
            .to_string();
        git_ok(&self.repo_root, &["worktree", "remove", "--force", &wt])?;
        Ok(())
    }
}

// Drop intentionally does NOT auto-remove: an unapplied proposal's worktree may
// still need inspection. Cleanup is always explicit via [`AgentWorkspace::cleanup`].

fn git_run(cwd: &Path, args: &[&str]) -> Result<Output, WorkspaceError> {
    Command::new("git")
        .args(args)
        .current_dir(cwd)
        // Keep subprocesses non-interactive and hermetic for tests/CI.
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .map_err(WorkspaceError::Io)
}

fn git_ok(cwd: &Path, args: &[&str]) -> Result<String, WorkspaceError> {
    let output = git_run(cwd, args)?;
    if !output.status.success() {
        return Err(git_failed(&format!("git {}", args.join(" ")), &output));
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

        ws.cleanup().expect("cleanup");
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
        ws.cleanup().expect("cleanup");
        let refs = Command::new("git")
            .args(["show-ref", "--verify", "--quiet", &ref_name])
            .current_dir(&repo)
            .status()
            .expect("show-ref");
        assert!(!refs.success(), "Clean finish must not create a ref");

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

        ws.cleanup().expect("cleanup");
        // Ref survives cleanup.
        let ref_oid_after = git_c(&repo, &["rev-parse", &ref_name]);
        assert_eq!(ref_oid_after, commit);

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

        ws.cleanup().expect("cleanup");
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

        ws.cleanup().expect("cleanup");
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

        ws.cleanup().expect("cleanup");
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
        ws.cleanup().expect("cleanup");
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn cleanup_removes_worktree_dir_but_ref_survives() {
        let repo = init_fixture("greppy-ws-cleanup");
        let run_id = unique_tag("run-cleanup");
        let ws = AgentWorkspace::create(&repo, &run_id).expect("create");
        std::fs::write(ws.worktree_path().join("x.txt"), b"x\n").unwrap();
        let outcome = ws.finish("keep ref").expect("finish");
        let (commit, ref_name) = match outcome {
            RunOutcome::Proposal {
                commit, ref_name, ..
            } => (commit, ref_name),
            RunOutcome::Clean => panic!("expected Proposal"),
        };

        let wt = ws.worktree_path().to_path_buf();
        assert!(wt.exists());
        ws.cleanup().expect("cleanup");
        assert!(!wt.exists(), "worktree dir must be gone");

        let ref_oid = git_c(&repo, &["rev-parse", &ref_name]);
        assert_eq!(ref_oid, commit);

        let _ = std::fs::remove_dir_all(&repo);
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
        git_c(
            &repo,
            &["worktree", "remove", "--force", wt_path.to_str().unwrap()],
        );
        let _ = std::fs::remove_dir_all(&repo);
    }
}
