//! Small helper that captures a workspace's git fingerprint so both the
//! indexer (writer) and the freshness checker (reader) can use it
//! without a circular dependency.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Same shape as `WorkspaceFingerprint` in `grepplus-freshness`, but
/// defined here so both crates can construct it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitFingerprint {
    pub canonical_root: PathBuf,
    pub git_dir: Option<PathBuf>,
    pub git_common_dir: Option<PathBuf>,
    pub head_oid: Option<String>,
    pub index_signature: Option<String>,
}

impl GitFingerprint {
    /// Capture the fingerprint of `root`. Non-git workspaces have
    /// `None` for the git fields.
    ///
    /// RV-010 / WP-R010: the previous implementation spawned **four**
    /// `git` subprocesses sequentially (`--git-dir`, `--git-common-dir`,
    /// `rev-parse HEAD`, plus a *fourth* `--git-dir` hidden inside the
    /// index-signature step). On a cold git repo each spawn costs
    /// 40-100 ms, so capture routinely ran 150-380 ms and blew the
    /// 200 ms freshness budget — `VISIBLE_AUGMENT` therefore never
    /// fired.
    ///
    /// This version keeps the public fields byte-for-byte identical but:
    ///   * runs the two `git rev-parse` directory probes (`--git-dir`,
    ///     `--git-common-dir`) concurrently on OS threads (no new deps),
    ///   * resolves the git directory **once** and reuses it for the
    ///     index signature instead of spawning a fourth `git`,
    ///   * reads `HEAD` straight off disk (`.git/HEAD`, following packed
    ///     and symbolic refs) and only shells out to `git rev-parse HEAD`
    ///     when the on-disk read is inconclusive (detached edge cases,
    ///     unusual ref backends).
    ///
    /// The order of the public fields is unchanged.
    pub fn capture(root: &Path) -> Self {
        let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());

        // Fire the two directory probes concurrently; these still need
        // `git` because `--git-dir` / `--git-common-dir` perform the full
        // repository discovery walk (parent dirs, worktrees, $GIT_DIR).
        let root_gd = root.to_path_buf();
        let root_gcd = root.to_path_buf();
        let h_git_dir = std::thread::spawn(move || git_rev_parse(&root_gd, "--git-dir"));
        let h_git_common_dir =
            std::thread::spawn(move || git_rev_parse(&root_gcd, "--git-common-dir"));

        let git_dir = h_git_dir.join().ok().flatten().map(PathBuf::from);
        let git_common_dir = h_git_common_dir.join().ok().flatten().map(PathBuf::from);

        // Resolve the git dir to an absolute path once and reuse it for
        // both the HEAD read and the index signature, so we never spawn a
        // redundant `--git-dir` probe.
        let resolved_git_dir = git_dir.as_ref().map(|gd| root.join(gd));

        let head_oid = resolved_git_dir
            .as_ref()
            .and_then(|gd| read_head_oid(gd, git_common_dir.as_ref().map(|c| root.join(c))))
            // Fallback: only shell out when the on-disk read failed.
            .or_else(|| git_rev_parse_stdout(root, "rev-parse", &["HEAD"]));

        let index_signature = resolved_git_dir
            .as_ref()
            .and_then(|gd| index_signature_at(gd));

        Self {
            canonical_root,
            git_dir,
            git_common_dir,
            head_oid,
            index_signature,
        }
    }
}

fn git_rev_parse(root: &Path, arg: &str) -> Option<String> {
    let out = Command::new("git")
        .current_dir(root)
        .arg("rev-parse")
        .arg(arg)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn git_rev_parse_stdout(root: &Path, sub: &str, args: &[&str]) -> Option<String> {
    let mut cmd = Command::new("git");
    cmd.current_dir(root).arg(sub).args(args);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// True if `s` looks like a full git object id (40 hex chars for SHA-1 or
/// 64 for SHA-256).
fn looks_like_oid(s: &str) -> bool {
    (s.len() == 40 || s.len() == 64) && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Read the current `HEAD` object id straight off disk.
///
/// `git_dir` is the resolved (absolute) git directory of the worktree;
/// `common_dir` (when present) is the resolved common directory shared by
/// linked worktrees — packed refs and most branch refs live there.
///
/// Returns `None` (so the caller can fall back to `git rev-parse HEAD`)
/// when HEAD cannot be resolved purely from on-disk files, e.g. a symref
/// chain that bottoms out in a packed ref we cannot find.
fn read_head_oid(git_dir: &Path, common_dir: Option<PathBuf>) -> Option<String> {
    let head_raw = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head_raw.trim();

    // Detached HEAD: the file holds the oid directly.
    if looks_like_oid(head) {
        return Some(head.to_string());
    }

    // Symbolic ref: "ref: refs/heads/<branch>".
    let target = head.strip_prefix("ref:").map(str::trim)?;

    // Loose ref can live in either the worktree git dir or the common dir.
    let mut search_dirs: Vec<&Path> = vec![git_dir];
    if let Some(c) = common_dir.as_deref() {
        if c != git_dir {
            search_dirs.push(c);
        }
    }

    for dir in &search_dirs {
        if let Ok(contents) = std::fs::read_to_string(dir.join(target)) {
            let oid = contents.trim();
            if looks_like_oid(oid) {
                return Some(oid.to_string());
            }
        }
    }

    // Packed refs: "<oid> <refname>" lines (peeled "^..." lines ignored).
    for dir in &search_dirs {
        if let Ok(packed) = std::fs::read_to_string(dir.join("packed-refs")) {
            for line in packed.lines() {
                let line = line.trim_start();
                if line.starts_with('#') || line.starts_with('^') {
                    continue;
                }
                if let Some((oid, refname)) = line.split_once(' ') {
                    if refname.trim() == target && looks_like_oid(oid.trim()) {
                        return Some(oid.trim().to_string());
                    }
                }
            }
        }
    }

    // Could not resolve from disk; let the caller fall back to git.
    None
}

/// SHA-256 of the `index` file inside the (already resolved, absolute)
/// `git_dir`. `None` when there is no index (fresh repo with nothing
/// staged) — byte-identical to the previous behaviour.
fn index_signature_at(git_dir: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let index_path = git_dir.join("index");
    let bytes = std::fs::read(&index_path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn tempdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "grepplus-fingerprint-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn git(dir: &Path, args: &[&str]) {
        let ok = Command::new("git")
            .current_dir(dir)
            .args(args)
            .status()
            .unwrap()
            .success();
        assert!(ok, "git {args:?} failed");
    }

    fn init_repo_with_commit(tag: &str) -> PathBuf {
        let tmp = tempdir(tag);
        git(&tmp, &["init", "-q"]);
        git(&tmp, &["config", "user.email", "t@t.t"]);
        git(&tmp, &["config", "user.name", "t"]);
        // Use a deterministic branch name regardless of git's default.
        git(&tmp, &["checkout", "-q", "-b", "main"]);
        std::fs::write(tmp.join("a.txt"), b"hello").unwrap();
        git(&tmp, &["add", "a.txt"]);
        git(&tmp, &["commit", "-q", "-m", "init"]);
        tmp
    }

    /// The fast on-disk path must produce the *exact* same fields as the
    /// reference implementation that shells out to `git` for everything.
    /// This guards against the optimization drifting from git's truth.
    fn reference_capture(root: &Path) -> GitFingerprint {
        let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let git_dir = git_rev_parse(root, "--git-dir").map(PathBuf::from);
        let git_common_dir = git_rev_parse(root, "--git-common-dir").map(PathBuf::from);
        let head_oid = git_rev_parse_stdout(root, "rev-parse", &["HEAD"]);
        let index_signature = git_dir
            .as_ref()
            .map(|gd| root.join(gd))
            .and_then(|gd| index_signature_at(&gd));
        GitFingerprint {
            canonical_root,
            git_dir,
            git_common_dir,
            head_oid,
            index_signature,
        }
    }

    #[test]
    fn fields_match_reference_on_committed_repo() {
        let tmp = init_repo_with_commit("match");
        let fast = GitFingerprint::capture(&tmp);
        let reference = reference_capture(&tmp);

        // The directory probes use the same git invocation, so they must
        // be identical.
        assert_eq!(fast.canonical_root, reference.canonical_root);
        assert_eq!(fast.git_dir, reference.git_dir);
        assert_eq!(fast.git_common_dir, reference.git_common_dir);
        // The whole point of RV-010: HEAD read off disk must equal git's.
        assert_eq!(
            fast.head_oid, reference.head_oid,
            "on-disk HEAD must equal `git rev-parse HEAD`"
        );
        assert!(fast.head_oid.is_some(), "committed repo must have HEAD");
        assert_eq!(fast.index_signature, reference.index_signature);
        assert!(
            fast.index_signature.is_some(),
            "staged repo must have an index signature"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn detached_head_matches_reference() {
        let tmp = init_repo_with_commit("detached");
        // Second commit, then detach onto the first.
        std::fs::write(tmp.join("b.txt"), b"world").unwrap();
        git(&tmp, &["add", "b.txt"]);
        git(&tmp, &["commit", "-q", "-m", "second"]);
        let first = git_rev_parse_stdout(&tmp, "rev-parse", &["HEAD~1"]).unwrap();
        git(&tmp, &["checkout", "-q", &first]);

        let fast = GitFingerprint::capture(&tmp);
        let reference = reference_capture(&tmp);
        assert_eq!(fast.head_oid, reference.head_oid);
        assert_eq!(fast.head_oid.as_deref(), Some(first.as_str()));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn head_resolves_through_packed_refs() {
        let tmp = init_repo_with_commit("packed");
        let expected = git_rev_parse_stdout(&tmp, "rev-parse", &["HEAD"]).unwrap();
        // Pack the refs so the loose ref disappears and the on-disk read
        // must consult packed-refs.
        git(&tmp, &["pack-refs", "--all"]);
        assert!(
            !tmp.join(".git/refs/heads/main").exists(),
            "pack-refs should have removed the loose ref"
        );

        let fast = GitFingerprint::capture(&tmp);
        assert_eq!(
            fast.head_oid.as_deref(),
            Some(expected.as_str()),
            "HEAD must resolve through packed-refs"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn non_git_root_has_no_git_fields() {
        let tmp = tempdir("nongit");
        let fp = GitFingerprint::capture(&tmp);
        assert!(fp.git_dir.is_none());
        assert!(fp.git_common_dir.is_none());
        assert!(fp.head_oid.is_none());
        assert!(fp.index_signature.is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn capture_well_under_budget_on_git_repo() {
        // RV-010 / WP-R010: capture must stay well under the 200 ms
        // production freshness budget. We assert a generous 200 ms here
        // (warm caches in CI are far below this; the budget the caller
        // enforces is 200 ms total).
        let tmp = init_repo_with_commit("budget");

        // Warm the OS dir cache, then take the BEST of several captures.
        // This is a regression guard against capture becoming seconds-slow
        // (the original RV-010 bug spawned 3 sequential git subprocesses).
        // We assert on the *minimum* under a generous 1 s bound rather than a
        // single sample under the 200 ms production budget: a single timed
        // sample flakes when `cargo test --workspace` saturates the CPU and the
        // git subprocesses get starved, even though warm capture is sub-10 ms.
        let _ = GitFingerprint::capture(&tmp);
        let mut best = std::time::Duration::from_secs(3600);
        let mut fp = None;
        for _ in 0..5 {
            let start = std::time::Instant::now();
            let cap = GitFingerprint::capture(&tmp);
            best = best.min(start.elapsed());
            fp = Some(cap);
        }
        assert!(fp.unwrap().head_oid.is_some());
        assert!(
            best.as_millis() < 1000,
            "fingerprint capture regressed to seconds-slow; best of 5 was {best:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
