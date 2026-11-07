//! Independent Git coverage evidence for index diagnostics.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GitTrackedFiles {
    pub total: u64,
    pub skip_worktree: u64,
}

impl GitTrackedFiles {
    /// Git excludes skip-worktree entries from checkout requirements.
    /// Other tracked files remain in the oracle, including accidentally missing ones.
    pub fn non_skip_worktree(self) -> u64 {
        self.total - self.skip_worktree
    }
}

/// Count tracked files independently of Greppy's discovery walker. Keep the full
/// Git count for diagnostics, but exclude Git's explicit skip-worktree entries
/// from the discovery denominator. NUL records preserve arbitrary path bytes.
/// `None` means Git evidence is unavailable; do not guess coverage in that case.
pub(crate) fn git_tracked_file_counts(root: &std::path::Path) -> Option<GitTrackedFiles> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-t", "-z"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mut counts = GitTrackedFiles {
        total: 0,
        skip_worktree: 0,
    };
    for record in out.stdout.split(|b| *b == 0).filter(|r| !r.is_empty()) {
        if record.len() < 3 || record[1] != b' ' {
            return None;
        }
        counts.total += 1;
        counts.skip_worktree += u64::from(record[0] == b'S');
    }
    Some(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf, process::Command};

    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn real_git_sparse_checkout_excludes_only_skip_worktree_entries() {
        let root = std::env::temp_dir().join(format!(
            "greppy-sparse-coverage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let fixture = Fixture(root);
        let git = |args: &[&str]| {
            let output = Command::new("git")
                .arg("-C")
                .arg(&fixture.0)
                .args(args)
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env(
                    "GIT_CONFIG_GLOBAL",
                    if cfg!(windows) { "NUL" } else { "/dev/null" },
                )
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        };
        git(&["init", "--quiet"]);
        for (directory, count) in [("visible", 120), ("hidden", 1100)] {
            fs::create_dir(fixture.0.join(directory)).unwrap();
            for index in 0..count {
                fs::write(
                    fixture.0.join(directory).join(format!("{index}.rs")),
                    "fn item() {}\n",
                )
                .unwrap();
            }
        }
        git(&["add", "."]);
        git(&[
            "-c",
            "user.name=Greppy coverage fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ]);
        assert_eq!(
            git_tracked_file_counts(&fixture.0),
            Some(GitTrackedFiles {
                total: 1220,
                skip_worktree: 0
            })
        );
        git(&["sparse-checkout", "init", "--cone"]);
        git(&["sparse-checkout", "set", "visible"]);
        let sparse = git_tracked_file_counts(&fixture.0).unwrap();
        assert_eq!(
            sparse,
            GitTrackedFiles {
                total: 1220,
                skip_worktree: 1100
            }
        );
        assert_eq!(sparse.non_skip_worktree(), 120);
        // A missing non-sparse source is still counted: do not hide real gaps.
        fs::remove_file(fixture.0.join("visible/0.rs")).unwrap();
        assert_eq!(git_tracked_file_counts(&fixture.0), Some(sparse));
    }
}
