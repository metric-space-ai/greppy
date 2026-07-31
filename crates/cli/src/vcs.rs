//! The git queries behind --since and --base.
//!
//! Split out of `lib.rs`; `use super::*` keeps every private helper there
//! reachable, and no behaviour changes.

/// Count git-tracked files under `root` as an INDEPENDENT oracle for
/// discovery coverage (the walker cannot be its own witness). `None` when
/// git is unavailable or the root is not a repository — the coverage check
/// is then skipped rather than guessed.
pub(crate) fn git_tracked_file_count(root: &std::path::Path) -> Option<u64> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout.iter().filter(|b| **b == 0).count() as u64)
}
