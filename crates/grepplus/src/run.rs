//! Drop-in grep runner with optional semantic augmentation.
//!
//! This module is shared between:
//! - `grepplus-grep` (the dedicated drop-in binary at `crates/grepplus/src/main.rs`)
//! - `grepplus` (the unified CLI dispatcher at `crates/cli/src/lib.rs::dispatch_grep`)
//!
//! Both call [`run_with_optional_augment`] so the bare-flag form
//! (`grepplus -R foo .`) and the explicit-subcommand form
//! (`grepplus grep -R foo .`) get the same heuristic + freshness
//! behaviour.

use std::ffi::OsString;
use std::path::Path;

use grepplus_core::error::Error;
use grepplus_core::workspace as workspace_locator;
use grepplus_freshness::FreshnessOutcome;
use grepplus_store::OpenOptions;

use crate::heuristic::{classify, FreshnessGate, GrepArgs, Mode};
use crate::sidecar;

/// Run real grep, then if the heuristic + freshness gate allow, write
/// a sidecar (and optionally append one labelled line to stdout).
///
/// Returns the real-grep exit code (modulo signal handling).
///
/// Drop-in contract (phasenplan §11.5, R-002, R-005): when real `grep`
/// returned a non-zero exit code (no matches, or an error), no synthetic
/// semantic content is produced: no sidecar, no synthetic stdout line.
/// The exit code and stdout/stderr are returned byte-exactly as real grep
/// produced them.
pub fn run_with_optional_augment(
    real_grep: &Path,
    argv: &[String],
    args: &GrepArgs,
) -> Result<i32, Error> {
    let real_exit = crate::run_grep(real_grep, argv)?;

    // R-002 / phasenplan §11.5: real-grep miss/error must not trigger
    // any visible semantic output. A non-zero exit code skips the
    // augment entirely (no sidecar, no synthetic line); the gate is
    // consulted only on a real-grep match.
    if real_exit != 0 {
        return Ok(real_exit);
    }

    let gate = freshness_gate(args);
    let mode = classify(args, gate);

    if matches!(mode, Mode::Sidecar | Mode::VisibleAugment) {
        if let Ok(workspace_root) = std::env::current_dir() {
            // Augment errors are non-fatal; we never let them bubble
            // up as a real-grep exit-code change.
            let _ = run_augment(args, mode, &workspace_root, real_grep, argv);
        }
    }

    Ok(real_exit)
}

/// `OsString` argv variant of [`run_with_optional_augment`].
///
/// P0 (R-014 re-review): the drop-in `grepplus-grep` entrypoint forwards
/// the original `OsString` argv to real grep byte-for-byte so it can
/// never panic on argv it cannot UTF-8-decode. The `GrepArgs`
/// classifier still operates on a best-effort lossy view for the
/// augmentation decision ONLY — the bytes that reach real grep are the
/// untouched `OsString`s.
pub fn run_with_optional_augment_os(
    real_grep: &Path,
    argv: &[OsString],
    args: &GrepArgs,
) -> Result<i32, Error> {
    let real_exit = crate::run_grep_os(real_grep, argv)?;

    // R-002 / phasenplan §11.5: real-grep miss/error must not trigger
    // any visible semantic output.
    if real_exit != 0 {
        return Ok(real_exit);
    }

    let gate = freshness_gate(args);
    let mode = classify(args, gate);

    if matches!(mode, Mode::Sidecar | Mode::VisibleAugment) {
        if let Ok(workspace_root) = std::env::current_dir() {
            // The original command string for the sidecar header is a
            // best-effort lossy rendering of the OsString argv; only the
            // forwarded argv (above) must be byte-exact.
            let argv_lossy: Vec<String> = argv
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            let _ = run_augment(args, mode, &workspace_root, real_grep, &argv_lossy);
        }
    }

    Ok(real_exit)
}

/// Compute the freshness gate for the current invocation. Used by
/// both the drop-in binary and the CLI dispatcher. Returns `Strict`
/// if the graph is stale, the store is unreadable, or the budget is
/// exceeded. Only `FreshnessOutcome::Fresh` yields `FreshnessGate::Fresh`.
pub fn freshness_gate(args: &GrepArgs) -> FreshnessGate {
    if args.is_stdin_only() {
        return FreshnessGate::Strict;
    }
    let cwd = match std::env::current_dir() {
        Ok(c) => c,
        Err(_) => return FreshnessGate::Strict,
    };
    // R-005: read the graph DB from the platform-locator's path,
    // never from `<cwd>/.grepplus/graph.db`.
    let store_path = workspace_locator::store_path(&cwd);
    let store = match grepplus_store::Store::open_with(&store_path, OpenOptions::read_only()) {
        Ok(s) => s,
        Err(_) => return FreshnessGate::Strict,
    };
    let project = cwd
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("default");
    let res = match grepplus_freshness::check_files(
        &store,
        &cwd,
        project,
        std::time::Duration::from_millis(200),
    ) {
        Ok(r) => r,
        Err(_) => return FreshnessGate::Strict,
    };
    match res.outcome {
        FreshnessOutcome::Fresh => FreshnessGate::Fresh,
        _ => FreshnessGate::Strict,
    }
}

fn latest_workspace_generation(store: &grepplus_store::Store) -> Option<u64> {
    let conn = store.conn();
    conn.query_row(
        "SELECT graph_generation FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
        [],
        |row| row.get::<_, i64>(0),
    )
    .ok()
    .map(|g| g as u64)
}

fn run_augment(
    args: &GrepArgs,
    mode: Mode,
    workspace_root: &Path,
    real_grep_path: &Path,
    argv: &[String],
) -> std::io::Result<()> {
    let Some(query) = args.pattern.as_deref() else {
        return Ok(());
    };

    // R-005: read the same locator'd store the freshness gate used.
    let store_path = workspace_locator::store_path(workspace_root);
    let store = match grepplus_store::Store::open_with(&store_path, OpenOptions::read_only()) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let near = workspace_root.to_string_lossy().to_string();
    let hits = match grepplus_search::semantic_query(&store, query, Some(&near), None, 10) {
        Ok(h) => h,
        Err(_) => return Ok(()),
    };
    if hits.is_empty() {
        return Ok(());
    }

    let generation = latest_workspace_generation(&store).unwrap_or(0);

    let original_cmd = format!("{} {}", real_grep_path.display(), argv[1..].join(" "));

    match mode {
        Mode::Strict => Ok(()),
        Mode::Sidecar => {
            sidecar::write_sidecar(workspace_root, query, &original_cmd, generation, &hits)?;
            Ok(())
        }
        Mode::VisibleAugment => {
            let sidecar_path =
                sidecar::write_sidecar(workspace_root, query, &original_cmd, generation, &hits)?;
            println!(
                "{}:1:<!-- GREPPLUS_NON_CANONICAL_HIT: {} -->",
                sidecar_path.display(),
                query
            );
            Ok(())
        }
    }
}
