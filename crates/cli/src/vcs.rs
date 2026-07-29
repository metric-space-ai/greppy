//! The git queries behind --since and --base.
//!
//! Split out of `lib.rs`; `use super::*` keeps every private helper there
//! reachable, and no behaviour changes.

use super::*;

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

pub(crate) fn git_changed_files(root_path: &std::path::Path) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io("spawn git status for search-code --changed", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "search-code --changed requires a git worktree ({})",
            err.trim()
        )));
    }

    let mut changed = Vec::new();
    let mut records = out.stdout.split(|b| *b == 0).filter(|r| !r.is_empty());
    while let Some(record) = records.next() {
        if record.len() < 4 {
            continue;
        }
        let x = record[0] as char;
        let y = record[1] as char;
        let path = String::from_utf8_lossy(&record[3..]).to_string();
        if matches!(x, 'R' | 'C') || matches!(y, 'R' | 'C') {
            let _ = records.next();
        }
        if path.is_empty() {
            continue;
        }
        if root_path.join(&path).is_file() {
            changed.push(path);
        }
    }
    changed.sort();
    changed.dedup();
    Ok(changed)
}

pub(crate) fn git_staged_files(root_path: &std::path::Path) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .args([
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            "--",
        ])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io("spawn git diff for search-code --staged", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "search-code --staged requires a git worktree ({})",
            err.trim()
        )));
    }

    let mut staged = out
        .stdout
        .split(|b| *b == 0)
        .filter(|r| !r.is_empty())
        .map(|r| String::from_utf8_lossy(r).to_string())
        .collect::<Vec<_>>();
    staged.sort();
    staged.dedup();
    Ok(staged)
}

pub(crate) fn git_diff_search_spec(
    root_path: &std::path::Path,
    scope: DiffSearchScope<'_>,
) -> Result<DiffSearchSpec> {
    match scope {
        DiffSearchScope::Since { rev } => {
            let diff_rev = git_resolve_commitish(root_path, rev, "search-code --since")?;
            let files = git_diff_files(root_path, &diff_rev, "search-code --since")?;
            Ok(DiffSearchSpec {
                scope: "since",
                diff_rev,
                merge_base: None,
                files,
            })
        }
        DiffSearchScope::Base { base } => {
            let base_rev = git_resolve_commitish(root_path, base, "search-code --base")?;
            let merge_base = git_merge_base(root_path, &base_rev)?;
            let files = git_diff_files(root_path, &merge_base, "search-code --base")?;
            Ok(DiffSearchSpec {
                scope: "base",
                diff_rev: base_rev,
                merge_base: Some(merge_base),
                files,
            })
        }
    }
}

pub(crate) fn git_resolve_commitish(root_path: &std::path::Path, rev: &str, context: &str) -> Result<String> {
    let rev = rev.trim();
    if rev.is_empty() {
        return Err(Error::Invalid(format!(
            "{context} requires a non-empty revision"
        )));
    }
    let spec = format!("{rev}^{{commit}}");
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--verify", spec.as_str()])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io(format!("spawn git rev-parse for {context}"), e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "{context} requires a valid git revision ({})",
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub(crate) fn git_merge_base(root_path: &std::path::Path, base_rev: &str) -> Result<String> {
    let out = std::process::Command::new("git")
        .args(["merge-base", base_rev, "HEAD"])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io("spawn git merge-base for search-code --base", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "search-code --base requires a revision with a merge-base against HEAD ({})",
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub(crate) fn git_diff_files(
    root_path: &std::path::Path,
    diff_rev: &str,
    context: &str,
) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .args([
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            diff_rev,
            "--",
        ])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io(format!("spawn git diff for {context}"), e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "{context} requires a valid git diff base ({})",
            err.trim()
        )));
    }

    let mut files = out
        .stdout
        .split(|b| *b == 0)
        .filter(|r| !r.is_empty())
        .map(|r| String::from_utf8_lossy(r).to_string())
        .filter(|path| root_path.join(path).is_file())
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    Ok(files)
}

pub(crate) fn git_diff_changed_lines(
    root_path: &std::path::Path,
    diff_rev: &str,
    context: &str,
) -> Result<std::collections::BTreeMap<String, std::collections::BTreeSet<i64>>> {
    let out = std::process::Command::new("git")
        .args(["diff", "--unified=0", "--diff-filter=ACMR", diff_rev, "--"])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io(format!("spawn git diff hunks for {context}"), e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "{context} requires a valid git diff base ({})",
            err.trim()
        )));
    }

    let mut current_file: Option<String> = None;
    let mut changed: std::collections::BTreeMap<String, std::collections::BTreeSet<i64>> =
        std::collections::BTreeMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            current_file = Some(path.to_string());
            continue;
        }
        if line.starts_with("+++ /dev/null") {
            current_file = None;
            continue;
        }
        if !line.starts_with("@@") {
            continue;
        }
        let Some(file) = current_file.as_ref() else {
            continue;
        };
        let Some((start, count)) = parse_git_diff_new_range(line) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let lines = changed.entry(file.clone()).or_default();
        for offset in 0..count {
            lines.insert(start + offset);
        }
    }
    Ok(changed)
}
