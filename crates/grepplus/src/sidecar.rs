//! Render the `.md` sidecar for a semantic augmentation.

use std::path::Path;

use grepplus_search::SemanticHit;

use crate::heuristic::sidecar_path;

/// Render the markdown content for the sidecar file.
///
/// The content follows the layout documented in
/// `docs/grepplus_rust_projektidee.md` and
/// `docs/grepplus_rust_phasenplan_port.md` §11.4:
/// sentinel comment at the top so real `grep` will treat it as a
/// non-canonical hit, then the structured context.
pub fn render_sidecar(
    _workspace_root: &Path,
    query: &str,
    original_command: &str,
    graph_generation: u64,
    hits: &[SemanticHit],
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "<!-- GREPPLUS_NON_CANONICAL_HIT: {} -->\n\n",
        query
    ));
    out.push_str("# GrepPlus semantic context for `");
    out.push_str(query);
    out.push_str("`\n\n");
    out.push_str("This file is generated helper context.\n");
    out.push_str("It is not part of the repository.\n");
    out.push_str("It is not a canonical source-code match.\n\n");

    out.push_str("## Original grep command\n```\n");
    out.push_str(original_command);
    out.push_str("\n```\n\n");

    out.push_str(&format!(
        "## Graph freshness\nfresh, graph_generation={}\n\n",
        graph_generation
    ));

    if hits.is_empty() {
        out.push_str("(no semantic hits)\n");
        return out;
    }

    out.push_str("## Likely symbols\n");
    for h in hits.iter().take(20) {
        out.push_str(&format!("- {}\n", h.node.qualified_name));
    }
    out.push_str("\n## Definitions\n");
    for h in hits
        .iter()
        .filter(|h| {
            h.node.label == "Function"
                || h.node.label == "Struct"
                || h.node.label == "Trait"
                || h.node.label == "Enum"
        })
        .take(10)
    {
        out.push_str(&format!(
            "- {} {}:{}-{}\n",
            h.node.qualified_name, h.node.file_path, h.node.start_line, h.node.end_line
        ));
    }

    out.push_str("\n## Suggested next reads\n");
    for h in hits.iter().take(10) {
        out.push_str(&format!("- {}\n", h.node.file_path));
    }

    out
}

/// Write the sidecar to disk and return its path. Parent directories
/// are created if they do not exist.
///
/// R-019 / WP-R006 (secure temp files, DoD §14):
/// - the sidecar directory is created mode 0700 (caller responsibility;
///   the public wrapper here documents the requirement);
/// - the sidecar file is created with `O_CREAT | O_EXCL` so a pre-existing
///   file or symlink at the target cannot redirect the write;
/// - the file mode is 0600;
/// - the path includes a per-invocation random component
///   (`sidecar_path` in `heuristic.rs`) so an attacker cannot predict
///   the write target.
pub fn write_sidecar(
    workspace_root: &Path,
    query: &str,
    original_command: &str,
    graph_generation: u64,
    hits: &[SemanticHit],
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;

    let path = sidecar_path(workspace_root, query);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Refuse to follow a symlink at the sidecar location itself. If
    // a co-located attacker has pre-planted a symlink here, we error
    // out rather than silently writing the sidecar through it. The
    // path includes a nonce, so this is a defence-in-depth — the
    // primary defence is the `O_EXCL`-equivalent `create_new(true)`.
    if let Ok(meta) = std::fs::symlink_metadata(&path) {
        if meta.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("refusing to follow symlink at {}", path.display()),
            ));
        }
        if meta.is_file() {
            // If a file already exists at this path, the nonce already
            // collided with an earlier write. Treat it as a no-op error
            // so the agent-visible call does not crash.
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("sidecar already exists at {}", path.display()),
            ));
        }
    }
    let content = render_sidecar(
        workspace_root,
        query,
        original_command,
        graph_generation,
        hits,
    );
    // Stage the content in a sibling `.tmp` first so a concurrent
    // `grep -R <query>` sees either nothing or the full file, never
    // a half-written one.
    let mut tmp_path = path.clone().into_os_string();
    tmp_path.push(".tmp");
    let tmp_path = std::path::PathBuf::from(tmp_path);

    // Atomic-create with strict mode. Unix uses `OpenOptionsExt::mode`;
    // Windows ignores mode (NTFS ACL is the user's responsibility).
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp_path)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
        std::fs::rename(&tmp_path, &path)?;
    }
    #[cfg(not(unix))]
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
        std::fs::rename(&tmp_path, &path)?;
    }
    Ok(path)
}

/// Time-to-live for sidecar files, in seconds. Honoured by
/// [`cleanup_expired`]. Defaults to 24 h. Overridable via the
/// `GREPPLUS_SIDECAR_TTL_SECS` environment variable.
pub fn sidecar_ttl_secs() -> u64 {
    std::env::var("GREPPLUS_SIDECAR_TTL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(86_400)
}

/// Walk the grepplus sidecar directory and remove files whose
/// last-modified timestamp is older than `ttl_secs`. Returns the
/// number of files removed.
///
/// This is the cleanup pass that phasenplan §14 calls for. It is
/// run on demand (today: from the bench scripts and from the
/// parity-run smoke; a future "auto-clean on start" integration
/// is a follow-on). The walk is bounded to the sidecar root
/// derived from `workspace_root` and cannot escape it.
pub fn cleanup_expired(workspace_root: &Path, ttl_secs: u64) -> std::io::Result<usize> {
    use std::time::{Duration, SystemTime};

    let path = sidecar_path(workspace_root, "ignored-for-path-shape");
    let dir = match path.parent() {
        Some(d) => d,
        None => return Ok(0),
    };
    if !dir.is_dir() {
        return Ok(0);
    }
    let now = SystemTime::now();
    let cutoff = Duration::from_secs(ttl_secs);
    let mut removed = 0usize;
    for entry in std::fs::read_dir(dir)? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        // Only consider files that look like sidecars (matching
        // the documented suffix). This prevents accidental
        // deletion of unrelated files in the temp dir.
        let name = match p.file_name().and_then(|s| s.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.ends_with("__GREPPLUS_SEMANTIC_NONCANONICAL.md") {
            continue;
        }
        let modified = match entry.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let age = match now.duration_since(modified) {
            Ok(a) => a,
            Err(_) => continue,
        };
        if age > cutoff && std::fs::remove_file(&p).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use grepplus_search::{graph::SearchGraphRow, SemanticHit, SemanticSignal};

    fn fake_hit(name: &str) -> SemanticHit {
        SemanticHit {
            node: SearchGraphRow {
                id: 1,
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: format!("p::Function::{name}"),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 5,
            },
            score: 0.5,
            signals: SemanticSignal {
                token_overlap: true,
                label_affinity: false,
                file_proximity: false,
                ..Default::default()
            },
            breakdown: Default::default(),
        }
    }

    #[test]
    fn render_starts_with_sentinel() {
        let s = render_sidecar(
            std::path::Path::new("/tmp/repo"),
            "processOrder",
            "grep -R processOrder .",
            42,
            &[fake_hit("processOrder")],
        );
        assert!(s.starts_with("<!-- GREPPLUS_NON_CANONICAL_HIT: processOrder -->"));
        assert!(s.contains("graph_generation=42"));
        assert!(s.contains("p::Function::processOrder"));
    }

    #[test]
    fn render_handles_empty_hits() {
        let s = render_sidecar(
            std::path::Path::new("/tmp/repo"),
            "nothing",
            "grep -R nothing .",
            1,
            &[],
        );
        assert!(s.contains("(no semantic hits)"));
    }

    // Serialise the env-mutating cleanup tests: GREPPLUS_STORE_DIR is a
    // process-global, so running them concurrently lets one test's
    // set_var/remove_var race another's, flaking the sidecar location
    // (observed under `cargo test --workspace` parallelism). The lock is
    // held across each test's whole set→use→remove window.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn cleanup_expired_removes_old_sidecars() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Set up a temp workspace, write a sidecar, backdate its
        // mtime, then run cleanup_expired with a small TTL.
        let tmp = std::env::temp_dir().join(format!(
            "grepplus-sidecar-cleanup-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        std::env::set_var("GREPPLUS_STORE_DIR", &tmp);
        let sidecar = write_sidecar(&tmp, "old_query", "grep -R old_query .", 1, &[])
            .expect("write_sidecar should succeed in test");
        std::env::remove_var("GREPPLUS_STORE_DIR");
        assert!(sidecar.is_file(), "sidecar should exist after write");

        // Backdate the mtime by 2 hours.
        filetime_set_mtime(
            &sidecar,
            std::time::SystemTime::now() - std::time::Duration::from_secs(7_200),
        );

        // Run cleanup with a 1-hour TTL: the backdated file is
        // outside the window and should be removed.
        std::env::set_var("GREPPLUS_STORE_DIR", &tmp);
        let removed = cleanup_expired(&tmp, 3_600).unwrap();
        std::env::remove_var("GREPPLUS_STORE_DIR");
        assert!(
            removed >= 1,
            "expected at least 1 file removed, got {removed}"
        );
        assert!(!sidecar.is_file(), "sidecar should be gone after cleanup");

        // Run again with the same TTL: nothing to do.
        std::env::set_var("GREPPLUS_STORE_DIR", &tmp);
        let removed_again = cleanup_expired(&tmp, 3_600).unwrap();
        std::env::remove_var("GREPPLUS_STORE_DIR");
        assert_eq!(removed_again, 0);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cleanup_expired_keeps_fresh_sidecars() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = std::env::temp_dir().join(format!(
            "grepplus-sidecar-fresh-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        std::env::set_var("GREPPLUS_STORE_DIR", &tmp);
        let sidecar = write_sidecar(&tmp, "fresh_query", "grep -R fresh_query .", 1, &[])
            .expect("write_sidecar should succeed in test");
        std::env::remove_var("GREPPLUS_STORE_DIR");
        assert!(sidecar.is_file());
        std::env::set_var("GREPPLUS_STORE_DIR", &tmp);
        let removed = cleanup_expired(&tmp, 86_400).unwrap();
        std::env::remove_var("GREPPLUS_STORE_DIR");
        assert_eq!(removed, 0, "fresh sidecar should not be removed");
        assert!(sidecar.is_file());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Set the mtime of `path` to `t`. Uses the `filetime` crate
    /// would be cleaner, but adding a dep just for tests is
    /// overkill — we shell out to `touch -t` instead.
    fn filetime_set_mtime(path: &Path, t: std::time::SystemTime) {
        let secs = t
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let day_secs = secs.rem_euclid(86_400);
        let hour = day_secs / 3600;
        let minute = (day_secs % 3600) / 60;
        let second = day_secs % 60;
        let mut days = secs.div_euclid(86_400);
        let mut year: i64 = 1970;
        loop {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            let year_days = if leap { 366 } else { 365 };
            if days >= year_days {
                days -= year_days;
                year += 1;
            } else {
                break;
            }
        }
        let month_days = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };
        let mut month = 1;
        for &md in &month_days {
            if days >= md {
                days -= md;
                month += 1;
            } else {
                break;
            }
        }
        let day = days + 1;
        let stamp = format!("{year:04}{month:02}{day:02}{hour:02}{minute:02}.{second:02}",);
        let _ = std::process::Command::new("touch")
            .arg("-t")
            .arg(&stamp)
            .arg(path)
            .status();
    }
}
