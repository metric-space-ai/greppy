//! Headless checkout for `greppy -p --in-place`.
//!
//! The isolated worktree remains the default. This workspace is the caller's
//! repository: tool writes, `/tmp`, and absolute paths inside the checkout are
//! the task's files. Cleanup removes only the per-run agent data directory.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::workspace::{workspace_data_root, RunOutcome, WorkspaceError};

#[derive(Debug)]
pub struct InPlaceWorkspace {
    repo_root: PathBuf,
    git_dir: PathBuf,
    run_id: String,
    data_root: PathBuf,
    base_commit: String,
}

impl InPlaceWorkspace {
    pub fn create(repo_root: &Path, run_id: &str) -> Result<Self, WorkspaceError> {
        if run_id.is_empty() || run_id.contains('/') || run_id.contains('\0') {
            return Err(WorkspaceError::Unsupported(
                "in-place run id must be a single path segment".into(),
            ));
        }
        let repo_root =
            fs::canonicalize(repo_root).map_err(|error| WorkspaceError::NotGitRepo {
                path: repo_root.to_path_buf(),
                detail: error.to_string(),
            })?;
        if !repo_root.is_dir() {
            return Err(WorkspaceError::NotGitRepo {
                path: repo_root,
                detail: "in-place root is not a directory".into(),
            });
        }
        let git_dir = git_dir(&repo_root)?;
        let base_commit = git(&repo_root, &["rev-parse", "HEAD"]).unwrap_or_else(|_| "none".into());
        let data_root = workspace_data_root()?;
        let workspace = Self {
            repo_root,
            git_dir,
            run_id: run_id.to_string(),
            data_root,
            base_commit,
        };
        fs::create_dir_all(workspace.agent_data_root())?;
        Ok(workspace)
    }

    pub fn worktree_path(&self) -> &Path {
        &self.repo_root
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn linked_git_dir(&self) -> &Path {
        &self.git_dir
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn base_commit(&self) -> &str {
        &self.base_commit
    }

    pub fn agent_data_root(&self) -> PathBuf {
        self.data_root.join("agent-data").join(&self.run_id)
    }

    /// Scratch stays available for the agent data layout. In-place runs do not
    /// point `TMPDIR` here; the process temp directory stays in force.
    pub fn agent_scratch_root(&self) -> PathBuf {
        self.data_root.join("agent-scratch").join(&self.run_id)
    }

    pub fn restore_proposal(&self, _ref_name: &str) -> Result<(), WorkspaceError> {
        Err(WorkspaceError::Unsupported(
            "in-place runs do not restore a proposal ref; the checkout is the workspace".into(),
        ))
    }

    /// Leave every edit in the checkout. The outcome carries the working-tree
    /// diff so `--diff` can print it; nothing is committed or reset.
    pub fn finish(&self, _message: &str) -> Result<RunOutcome, WorkspaceError> {
        let porcelain = git(
            &self.repo_root,
            &["status", "--porcelain", "--untracked-files=all"],
        )
        .unwrap_or_default();
        if porcelain.trim().is_empty() {
            return Ok(RunOutcome::Clean);
        }
        let tracked = git(&self.repo_root, &["diff", "HEAD"]).unwrap_or_default();
        let stat = git(&self.repo_root, &["diff", "--stat", "HEAD"]).unwrap_or_default();
        let untracked = untracked_paths(&porcelain);
        let extra = untracked_patch(&self.repo_root, &untracked);
        let mut patch = tracked;
        if !extra.is_empty() {
            if !patch.is_empty() && !patch.ends_with('\n') {
                patch.push('\n');
            }
            patch.push_str(&extra);
        }
        let mut stat = stat;
        if !untracked.is_empty() {
            if !stat.is_empty() && !stat.ends_with('\n') {
                stat.push('\n');
            }
            stat.push_str(&format!("untracked: {}\n", untracked.join(", ")));
        }
        Ok(RunOutcome::Proposal {
            commit: "working-tree".into(),
            ref_name: format!("in-place:{}", self.run_id),
            patch,
            stat,
        })
    }

    /// Files are already in the checkout. Applying a proposal would copy them
    /// onto themselves.
    pub fn apply_to(&self, _target: &Path, _commit: &str) -> Result<(), WorkspaceError> {
        Ok(())
    }

    pub fn keep(&self) -> Result<(), WorkspaceError> {
        Ok(())
    }

    pub fn cleanup(self) -> Result<(), WorkspaceError> {
        for path in [self.agent_data_root(), self.agent_scratch_root()] {
            if path.exists() {
                fs::remove_dir_all(&path)?;
            }
        }
        Ok(())
    }
}

fn git_dir(repo: &Path) -> Result<PathBuf, WorkspaceError> {
    match git(repo, &["rev-parse", "--absolute-git-dir"]) {
        Ok(path) => Ok(PathBuf::from(path.trim())),
        Err(error) => Err(WorkspaceError::NotGitRepo {
            path: repo.to_path_buf(),
            detail: error.to_string(),
        }),
    }
}

fn git(repo: &Path, args: &[&str]) -> Result<String, WorkspaceError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| WorkspaceError::GitFailed {
            command: format!("git {}", args.join(" ")),
            stderr: error.to_string(),
            status: None,
        })?;
    if !output.status.success() {
        return Err(WorkspaceError::GitFailed {
            command: format!("git {}", args.join(" ")),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            status: output.status.code(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn untracked_paths(porcelain: &str) -> Vec<String> {
    porcelain
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("?? ")?;
            let path = rest.trim();
            (!path.is_empty()).then(|| path.to_string())
        })
        .collect()
}

fn untracked_patch(repo: &Path, paths: &[String]) -> String {
    let mut out = String::new();
    for rel in paths {
        let abs = repo.join(rel);
        let Ok(bytes) = fs::read(&abs) else {
            continue;
        };
        out.push_str(&format!("diff --git a/{rel} b/{rel}\n"));
        if bytes.contains(&0) {
            out.push_str("new file mode 100644\nBinary file\n");
            continue;
        }
        out.push_str(&format!("--- /dev/null\n+++ b/{rel}\n"));
        let text = String::from_utf8_lossy(&bytes);
        for line in text.lines() {
            out.push('+');
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn temp_repo(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "greppy-in-place-{tag}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        let git = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(&path)
                .args(args)
                .status()
                .unwrap();
            assert!(status.success(), "git {args:?}");
        };
        git(&["init"]);
        git(&["config", "user.email", "in-place@example.test"]);
        git(&["config", "user.name", "In Place"]);
        fs::write(path.join("tracked.txt"), "base\n").unwrap();
        git(&["add", "tracked.txt"]);
        git(&["commit", "-m", "base"]);
        path
    }

    #[test]
    fn in_place_edits_stay_in_the_checkout_after_cleanup() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let repo = temp_repo("stay");
        let data = repo.join("workspace-data");
        fs::create_dir_all(&data).unwrap();
        std::env::set_var("GREPPY_WORKSPACE_DIR", &data);
        let workspace = InPlaceWorkspace::create(&repo, "run-1").unwrap();
        assert_eq!(
            workspace.worktree_path(),
            fs::canonicalize(&repo).unwrap().as_path()
        );
        fs::write(repo.join("tracked.txt"), "changed\n").unwrap();
        fs::write(repo.join("fresh.txt"), "new\n").unwrap();
        let outcome = workspace.finish("report").unwrap();
        match outcome {
            RunOutcome::Proposal {
                patch,
                stat,
                commit,
                ..
            } => {
                assert_eq!(commit, "working-tree");
                assert!(patch.contains("+changed"), "{patch}");
                assert!(patch.contains("+new"), "{patch}");
                assert!(stat.contains("untracked: fresh.txt"), "{stat}");
            }
            RunOutcome::Clean => panic!("dirty checkout must be reported"),
        }
        let workspace = InPlaceWorkspace::create(&repo, "run-1").unwrap();
        workspace.cleanup().unwrap();
        assert_eq!(
            fs::read_to_string(repo.join("tracked.txt")).unwrap(),
            "changed\n"
        );
        assert_eq!(fs::read_to_string(repo.join("fresh.txt")).unwrap(), "new\n");
        assert!(!data.join("agent-data").join("run-1").exists());
        std::env::remove_var("GREPPY_WORKSPACE_DIR");
        let _ = fs::remove_dir_all(repo);
    }

    #[test]
    fn clean_checkout_finishes_clean() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let repo = temp_repo("clean");
        let data = repo.join("workspace-data");
        fs::create_dir_all(&data).unwrap();
        std::env::set_var("GREPPY_WORKSPACE_DIR", &data);
        let workspace = InPlaceWorkspace::create(&repo, "run-clean").unwrap();
        assert!(matches!(workspace.finish("").unwrap(), RunOutcome::Clean));
        std::env::remove_var("GREPPY_WORKSPACE_DIR");
        let _ = fs::remove_dir_all(repo);
    }
}
