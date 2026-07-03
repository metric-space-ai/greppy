// Phase 7 — Freshness probe example binary.
//
// Usage:
//   cargo run --bin freshness-probe -- <repo-root>
//
// Prints a one-line JSON document with the freshness outcome against
// the persisted workspace_state in `<repo-root>/.grepplus/graph.db`:
//
//   {"outcome":"Fresh","elapsed_ms":3,"reasons":[]}
//
// Exit codes:
//   0   probe ran, JSON line on stdout
//   2   no .grepplus/graph.db at the given root
//   3   store open failed
//   4   probe error (logged to stderr)

use std::time::Instant;

use grepplus_core::Result;
use grepplus_store::{OpenOptions, Store};

fn main() {
    let _ = grepplus_core::logging::init();
    let root = std::env::args()
        .nth(1)
        .expect("usage: freshness-probe <repo-root>");
    let project = std::env::args().nth(2).unwrap_or_else(|| {
        std::path::Path::new(&root)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string()
    });
    let code = match run(&root, &project) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("freshness-probe: {e}");
            4
        }
    };
    std::process::exit(code);
}

fn run(root: &str, project: &str) -> Result<()> {
    // R-005 / WP-R005: read the graph DB from the platform locator
    // rather than the legacy `<root>/.grepplus/graph.db`. The
    // legacy path is no longer where `grepplus index` writes.
    let workspace_root = std::path::Path::new(root);
    let path = grepplus_core::workspace::store_path(workspace_root);
    // Cold-start path: no store yet → emit Cold instead of failing.
    // The bench's Scenario 1 expects this; agents running the probe
    // in a fresh tree get a clean JSON answer rather than an exit
    // code they have to special-case.
    if !path.is_file() {
        println!("{{\"outcome\":\"Cold\",\"elapsed_ms\":0,\"reasons\":[]}}");
        return Ok(());
    }
    let store = match Store::open_with(&path, OpenOptions::read_only()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("store open failed: {e}");
            std::process::exit(3);
        }
    };
    let start = Instant::now();
    // Phase 7 freshness probe uses the per-file check so that
    // unstaged file edits are detected as Stale. Budget is generous
    // (30 s) because the per-file walk hashes every file in the
    // search paths; the production `grepplus-grep` gate uses 200 ms
    // because it only walks the search paths, not the whole repo.
    let res = grepplus_freshness::check_files(
        &store,
        std::path::Path::new(root),
        project,
        std::time::Duration::from_millis(30_000),
    )?;
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let (label, reasons) = match res.outcome {
        grepplus_freshness::FreshnessOutcome::Cold => ("Cold".to_string(), vec![]),
        grepplus_freshness::FreshnessOutcome::Fresh => ("Fresh".to_string(), vec![]),
        grepplus_freshness::FreshnessOutcome::Stale { reasons } => {
            ("Stale".to_string(), reasons.clone())
        }
        grepplus_freshness::FreshnessOutcome::RootMismatch => ("RootMismatch".to_string(), vec![]),
    };
    let reasons_json = serde_json::to_string(&reasons).unwrap_or_else(|_| "[]".into());
    println!("{{\"outcome\":\"{label}\",\"elapsed_ms\":{elapsed_ms},\"reasons\":{reasons_json}}}");
    Ok(())
}
