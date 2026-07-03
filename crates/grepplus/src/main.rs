//! `grepplus-grep` binary entry point.
//!
//! Phase 6 wiring:
//! 1. Parse argv.
//! 2. Discover real grep.
//! 3. Run the freshness check (workspace fingerprint vs persisted
//!    `workspace_state`). If stale or budget-exceeded, the freshness
//!    gate is `Strict` regardless of argv.
//! 4. Classify the invocation (Strict / Sidecar / VisibleAugment).
//! 5. Run real grep.
//! 6. If classification was `Sidecar` or `VisibleAugment`, run a
//!    semantic query against the on-disk store and either write a
//!    sidecar (Sidecar) or print a single non-canonical line
//!    (VisibleAugment).
//!
//! Real-grep stdout/stderr/exit are always preserved. Synthetic
//! content is appended after real-grep's own output and is labelled
//! so agents can ignore it.

use std::process::ExitCode;

use grepplus_grep::heuristic::GrepArgs;
use grepplus_grep::run;
use grepplus_grep::sidecar;

fn main() -> ExitCode {
    let _ = grepplus_core::logging::init();

    // R-014 / WP-R014: probabilistically clean expired sidecars on
    // start. Once every ~10 minutes per process, walk each known
    // workspace's sidecar dir and remove files older than the
    // configured TTL. Errors are non-fatal: cleanup is a
    // best-effort.
    maybe_run_sidecar_cleanup();

    // P0 (R-014 re-review): use `args_os` throughout so the wrapper can
    // never panic on argv it cannot UTF-8-decode. The original
    // `OsString` argv is forwarded to real grep byte-for-byte; the
    // `GrepArgs` classifier operates on a best-effort lossy view for the
    // augmentation decision ONLY.
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let args = GrepArgs::parse_os(&argv[1..]);

    let real = match grepplus_grep::discover_grep() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("grepplus-grep: {e}");
            return ExitCode::from(3);
        }
    };

    let exit = match run::run_with_optional_augment_os(&real, &argv, &args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("grepplus-grep: {e}");
            return ExitCode::from(2);
        }
    };

    let normalized = if exit < 0 { 1 } else { exit as u8 };
    ExitCode::from(normalized)
}

/// R-014 / WP-R014 — probabilistically run `cleanup_expired` for the
/// current working directory's sidecar dir. Throttled to ~ once per
/// 10 minutes per process so a tight `grep` loop doesn't keep
/// walking the dir.
///
/// Files written by the wrapper are mode 0600 and named with a
/// nonce, so a competing process on the same workspace cannot race
/// us. The TTL is honoured with a small grace period so that
/// multiple chained invocations in the same minute do not race.
fn maybe_run_sidecar_cleanup() {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static LAST_RUN: Mutex<Option<Instant>> = Mutex::new(None);
    const MIN_GAP: Duration = Duration::from_secs(10 * 60);

    let should_run = {
        let mut guard = LAST_RUN.lock().unwrap();
        match *guard {
            Some(t) if t.elapsed() < MIN_GAP => false,
            _ => {
                *guard = Some(Instant::now());
                true
            }
        }
    };
    if !should_run {
        return;
    }
    let cwd = match std::env::current_dir() {
        Ok(c) => c,
        Err(_) => return,
    };
    let ttl = sidecar::sidecar_ttl_secs();
    let _ = sidecar::cleanup_expired(&cwd, ttl);
}
