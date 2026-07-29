//! `greppy-indexer` — multi-pass indexer.
//!
//! The pipeline is a **two-phase, parallel-extract / serial-write**
//! engine:
//! 1. walk the repository (via `greppy-discover`),
//! 2. filter to files whose language is supported,
//! 3. **parse + extract every supported file in PARALLEL** (CPU-bound,
//!    pure over the file bytes) using a bounded `rayon` pool — see
//!    [`Concurrency`] below,
//! 4. **apply store writes SERIALLY** in a deterministic order (SQLite
//!    is a single-writer): per-file delete-then-insert, file-content,
//!    `file_state` with the real graph generation,
//! 5. resolve and persist edges in a second project-wide phase
//!    (`CALLS` and `IMPORTS` cross-file resolution),
//! 6. bump the workspace generation counter.
//!
//! ## Concurrency & memory budget
//!
//! Parsing is the CPU-bound hot path; the store write is serial because
//! SQLite has one writer. We therefore extract in parallel and apply in
//! order, which keeps the resulting graph **byte-for-byte identical** to
//! a fully sequential run (the determinism test enforces this).
//!
//! - Worker threads are capped to
//!   [`greppy_core::default_worker_count`] (cgroup-aware, honours the
//!   `GREPPY_WORKERS` env override).
//! - Before the parallel phase the memory budget is initialised
//!   ([`greppy_core::mem_budget_init`]). If
//!   [`greppy_core::mem_over_budget`] trips mid-run we **throttle**:
//!   the remaining files are extracted sequentially (one buffered file
//!   at a time) rather than fanned out, so a low-RAM container degrades
//!   to the serial path instead of OOMing.
//! - The existing 50 MiB per-file cap is preserved: oversized
//!   files are detected by `stat` and never read into memory.
//!
//! Hardening:
//! - Files larger than `MAX_FILE_SIZE_BYTES` are skipped with a count
//!   in the report; they are NOT read into memory (avoids OOM on
//!   multi-GB inputs).
//! - `greppy_indexer::index` is wrapped by an advisory `fs2`/`fd`
//!   lock so two parallel runs do not corrupt the SQLite file.
//!
//! ## Incremental indexing (Track A)
//!
//! `index()` is incremental from the second run onward. It diffs the
//! on-disk inventory against the persisted `file_state`
//! ([`greppy_freshness::compute_file_diff`]): Added / Modified files are
//! re-parsed and rewritten, Deleted files have their nodes / content /
//! state removed, and **Unchanged files are skipped** (counted in
//! [`IndexReport::files_skipped`]). Because a cross-file edge from an
//! unchanged file can target a changed file's symbol, the indexer persists
//! every file's *raw* extracted edges in the store-owned `raw_edges` table
//! (via [`Store::insert_raw_edges`] / [`Store::list_raw_edges`]) so it can
//! re-resolve without re-parsing unchanged
//! files — yet still produce a graph byte-for-byte identical to a full
//! reindex (enforced by
//! `incremental_matches_full_reindex_across_a_sequence_of_edits`).
//!
//! ### Incremental edge re-resolution
//!
//! Rather than re-resolving the **whole** project's raw edges after every
//! run — O(total edges) even for a no-op — edge re-resolution is scoped to
//! only the edges a run could have affected. After the extract/write phase,
//! SQLite's FK-cascade has already
//! removed every resolved edge with an endpoint in a changed/deleted file, so
//! the surviving edges all connect two *unchanged* files. Edge resolution is
//! a pure function of the project's **definition fingerprint** —
//! `(qualified_name, name, label, file_path)` over every node (node ids are
//! excluded; they are autoincrement and never change *which* def a name
//! resolves to). The chosen invariant ([`resolve_edges_incremental`]):
//!
//! - **No file changed** → re-resolve nothing (the no-op headline win).
//! - **Files changed but the def fingerprint did NOT** (pure body edits) →
//!   re-resolve only the cascaded edges: raw edges whose source is a changed
//!   file, plus raw edges (from any file) that name a definition living in a
//!   changed file (the only way a target could have landed in one).
//! - **The def fingerprint changed** → fall back to the full insert-only
//!   re-resolution (byte-identical to a first run), because an unchanged
//!   file's edge may now resolve, unresolve, or become ambiguous and the
//!   insert-only path cannot prove which survivors are stale.
//!
//! All three branches are byte-identical to a full re-resolution
//! (`incremental_matches_full_reindex…`, `noop_reindex_reresolves_zero_edges`,
//! `body_only_edit_takes_cheap_path_and_matches_full`,
//! `cross_file_caller_unchanged_when_callee_body_edited`).
//!
//! ## Scale (the O(n²) edge hotspot)
//!
//! A naive edge resolution would issue per-edge SQLite queries (a name
//! lookup, plus an extra round-trip for ambiguous names) and one
//! transaction per inserted edge, making indexing super-linear. Instead
//! we build an in-memory [`GraphIndex`] once per run (a single node
//! query → `qname →
//! node` map + `name → [nodes]` multimap) and resolves every edge in
//! memory, then inserts all edges in one transaction. Measured on a
//! synthetic Rust corpus (debug build, in-memory store; reproduce with
//! `cargo run -p greppy-indexer --example profile_index -- <repo>`):
//!
//! | corpus | before (full) | after (full) | after (incremental no-op) |
//! |--------|---------------|--------------|---------------------------|
//! | 500 files / 2k edges  | 45 s  | ~17 s | — |
//! | 1000 files / 4k edges | 169 s | ~28 s | ~1.3 s |
//!
//! The edge-resolution phase alone dropped from ~35 s to ~0.2 s on the
//! 500-file corpus (≈165×), turning the previously super-linear phase into
//! an O(nodes + edges) one. `edge_resolution_scales_linearly_not_quadratically`
//! guards the asymptotics in CI.

#![deny(rust_2018_idioms)]

pub mod embedding;
mod structural;

use std::path::Path;

use greppy_core::Result;
use greppy_discover::{read_stable_file, stable_metadata, InventoryEntry, StableFileMetadata};
use greppy_parser::{
    self, extract as parser_extract, manifest_for_language, ExtractedEdge, ExtractedNode, Language,
    ProviderManifest, ProviderOutput, ProviderStatus,
};
use greppy_store::{
    self,
    file_state::{self, FileState},
    workspace_state as ws, ContentRow, FileIdentity, IndexSkip, NewEdge, NewNode, NewRawEdge,
    Project, ProviderState, RawEdge, Store, WorkspaceState,
};
use rayon::prelude::*;

pub use embedding::{
    count_code_embedding_documents_for_project, count_embedding_candidate_nodes,
    index_code_embeddings_for_project, index_code_embeddings_for_project_with_progress,
    CodeEmbeddingProvider, EmbeddingGemmaCodeProvider, EmbeddingIndexOptions,
    EmbeddingIndexProgress, EmbeddingIndexReport,
};

/// Fraction of the process RAM budget the indexer initialises
/// [`greppy_core::membudget`] with on first run. `membudget::init` is
/// idempotent, so
/// if the CLI already initialised it with another fraction that value
/// wins; this is purely a safety floor for library callers (and tests).
const INDEX_RAM_FRACTION: f64 = 0.5;

/// Files above this size are skipped during indexing
/// and recorded as `files_oversize` in the report. ~50 MiB is the
/// indexer-side default; CLI users can override via
/// `GREPPY_MAX_FILE_SIZE` (in bytes).
pub const MAX_FILE_SIZE_BYTES: u64 = 50 * 1024 * 1024;

/// One indexer run, captured for tests and diagnostics.
#[derive(Debug, Clone, Default)]
pub struct IndexReport {
    pub project: String,
    pub root: std::path::PathBuf,
    pub files_considered: usize,
    pub files_indexed: usize,
    pub files_unsupported_language: usize,
    pub files_unreadable: usize,
    /// Number of files skipped because their size exceeds
    /// `MAX_FILE_SIZE_BYTES`.
    pub files_oversize: usize,
    pub nodes_extracted: usize,
    pub edges_extracted: usize,
    pub graph_generation: u64,
    /// Number of worker threads the parallel extract phase was bounded
    /// to (capped to [`greppy_core::default_worker_count`]). `1` means
    /// the run was effectively sequential (single core, or a
    /// `GREPPY_WORKERS=1` override). Additive field — older callers
    /// that only read the original fields are unaffected.
    pub worker_count: usize,
    /// `true` if the memory budget tripped during the parallel extract
    /// phase and the indexer throttled the remaining files onto the
    /// sequential path. Stays `false` on a normally-provisioned host.
    pub throttled_for_memory: bool,
    /// Number of files the **incremental** path left untouched because
    /// their content hash matched the persisted `file_state` (Added /
    /// Modified / Deleted files are re-processed; Unchanged ones are
    /// skipped — their nodes, content, and persisted raw edges are kept).
    /// `0` on a first/full run (every file is processed). Additive field —
    /// older callers that only read the original counters are unaffected.
    pub files_skipped: usize,
    /// Files skipped because `GREPPY_MAX_FILES` limited the index scope.
    pub files_skipped_by_file_limit: usize,
    /// Files skipped because `GREPPY_INDEX_TIME_BUDGET_MS` was exhausted
    /// before they could be scheduled for extraction.
    pub files_skipped_by_time_budget: usize,
}

impl IndexReport {
    pub fn is_clean(&self) -> bool {
        self.files_unreadable == 0
    }
}

/// Optional index-time discovery controls.
///
/// This is deliberately separate from the public CLI until the selected
/// include/exclude scope is persisted and consumed by freshness checks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexOptions {
    pub discover_overrides: greppy_discover::WalkOverrides,
}

/// Run the indexer against `root`. The store is mutated in-place; nodes
/// and file_state rows are upserted, edges are inserted, and the
/// workspace generation counter is bumped.
///
/// Callers should hold `greppy_freshness::with_lock`
/// around this call to serialise concurrent indexers on the same
/// store. This function does NOT acquire the lock itself because
/// the caller's `Store` borrow cannot be passed through the
/// lock-helper's closure; we expect the public CLI dispatcher to
/// wrap the call.
pub fn index(store: &mut Store, root: &Path, project_name: &str) -> Result<IndexReport> {
    index_with_options(store, root, project_name, &IndexOptions::default())
}

/// Run the indexer with explicit discovery options.
pub fn index_with_options(
    store: &mut Store,
    root: &Path,
    project_name: &str,
    options: &IndexOptions,
) -> Result<IndexReport> {
    let abs_root = greppy_discover::detect_repo_root(root)?;
    let all_entries = greppy_discover::walk_with_policy_and_overrides(
        &abs_root,
        &greppy_discover::SkipPolicy::walk_default(),
        &options.discover_overrides,
    )?;

    let mut report = IndexReport {
        project: project_name.to_string(),
        root: abs_root.clone(),
        files_considered: all_entries.len(),
        ..Default::default()
    };

    // Project row.
    store.upsert_project(&Project {
        name: project_name.to_string(),
        indexed_at: ws::now_iso8601(),
        root_path: abs_root.to_string_lossy().into_owned(),
    })?;
    if !content_indexing_enabled() {
        // v4 no longer persists full source bodies. Purge rows left by an
        // older store before deciding between full and incremental indexing,
        // otherwise unchanged files could keep stale searchable source.
        store.delete_project_file_content(project_name)?;
    }

    // Workspace state (fresh insert on first run). Capture the git
    // fingerprint so the freshness check has something to compare
    // against on the next greppy invocation.
    let fp = greppy_core::GitFingerprint::capture(&abs_root);
    store.upsert_workspace_state(&WorkspaceState {
        root_path: abs_root.to_string_lossy().into_owned(),
        git_dir: fp
            .git_dir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        git_common_dir: fp
            .git_common_dir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        head_oid: fp.head_oid.clone(),
        index_signature: fp.index_signature.clone(),
        schema_version: store.schema_version()?,
        indexer_version: indexer_version_for_options(options),
        graph_generation: 0,
        updated_at: ws::now_iso8601(),
    })?;

    // Bump the workspace generation BEFORE we walk the entries so
    // every file_state row we write in this run carries the
    // generation that will be current after the run completes.
    let current_gen = store.bump_generation(&abs_root.to_string_lossy())?;

    // Generation stamp for every file_state row written this run. It is
    // the value bumped at the start of this invocation and is constant
    // across the whole run, so we read it once (the old per-file read
    // returned the same number N times).
    let generation = current_gen;

    let controls = IndexControls::from_env();
    let controlled_entries = apply_large_repo_controls(&all_entries, &controls, &mut report);
    let entries = controlled_entries.active;

    // Did a prior run of THIS (migrated) indexer materialize raw edges for
    // this project? We must NOT treat a run as incremental unless the store's
    // `raw_edges` table reflects a previous extraction, or re-resolution would
    // run over an empty raw-edge set and silently drop every edge.
    //
    // The store-owned `raw_edges` table is created by migration 0007 on every
    // open, so its mere existence is no longer a usable signal. We combine two
    // facts:
    //   * `raw_edges` holds rows for this project — a prior migrated run
    //     extracted at least one edge; OR
    //   * the legacy `indexer_raw_edges` sidecar is ABSENT — no pre-migration
    //     binary ever indexed this store, so an empty `raw_edges` means a prior
    //     migrated run simply produced zero edges (a legitimate edgeless repo),
    //     not a stale pre-migration graph we would wrongly inherit.
    // A store last indexed by the pre-migration binary keeps its
    // `indexer_raw_edges` sidecar but has an empty `raw_edges`; the second
    // clause is false for it, so we correctly fall back to a full reindex
    // (safe — it repopulates `raw_edges`).
    let raw_edges_present =
        store.count_raw_edges(project_name)? > 0 || !legacy_raw_edge_sidecar_exists(store)?;

    let worker_count = greppy_core::default_worker_count(true).max(1);
    report.worker_count = worker_count;
    // Initialise the RAM budget once (idempotent). On a host where total
    // RAM could not be read the budget is 0 and `over_budget()` is
    // always false, so this is a no-op guard there.
    let _ = greppy_core::mem_budget_init(INDEX_RAM_FRACTION);

    // ── Incremental vs full ─────────────────────────────────────────
    // A run is **incremental** when the project already has persisted
    // `file_state` rows (a prior `index()` populated them) AND the store's
    // `raw_edges` table holds edges for this project (so we can re-resolve
    // unchanged files' edges without re-parsing them). Otherwise it is a
    // full first run.
    //
    // The incremental path re-extracts + rewrites ONLY changed files
    // (Added / Modified), deletes nodes/content/state/raw-edges for
    // Deleted files, and KEEPS unchanged files' nodes + raw edges. Both
    // paths then re-resolve over the *whole* project's raw edges, so the
    // resulting graph is byte-for-byte identical to a full reindex (the
    // `incremental_matches_full_reindex` test enforces this).
    let prior_state = store.list_file_states(project_name)?;
    let incremental = !prior_state.is_empty() && raw_edges_present;

    if incremental {
        // Capture the project's **definition fingerprint** before we touch
        // any node (PHASE A deletes/re-inserts changed files' nodes). The
        // fingerprint is the exact set of node identity tuples that
        // cross-file edge resolution consults — `(qname, name, label,
        // file_path)`. Comparing it before vs after PHASE A tells us whether
        // any changed file altered the *resolvable* definition set, which is
        // the sole reason an edge from an UNCHANGED file could change its
        // resolution. See `resolve_edges_incremental` for the invariant.
        let def_fp_before = def_fingerprint(store, project_name)?;

        let changed_files = run_incremental(
            store,
            project_name,
            &entries,
            generation,
            worker_count,
            &mut report,
        )?;

        // PHASE B (incremental). Re-resolve only the edges that PHASE A's
        // FK-cascade removed (or whose resolution could have flipped),
        // instead of the whole project's raw edges.
        report.edges_extracted =
            resolve_edges_incremental(store, project_name, &changed_files, &def_fp_before)?;
    } else {
        let profile = std::env::var("GREPPY_PROFILE").is_ok();
        let t = std::time::Instant::now();
        run_full(
            store,
            project_name,
            &entries,
            generation,
            worker_count,
            &mut report,
        )?;
        if profile {
            eprintln!(
                "[profile] run_full (parse+extract+write graph) {:?}",
                t.elapsed()
            );
        }

        // PHASE B (full). Resolve over the WHOLE project's freshly-persisted
        // raw edges — the first run has no prior graph to preserve.
        let t = std::time::Instant::now();
        let raw_edges = load_all_raw_edges(store, project_name)?;
        if profile {
            eprintln!("[profile] load_all_raw_edges {:?}", t.elapsed());
        }
        let t = std::time::Instant::now();
        report.edges_extracted = resolve_and_persist_edges(store, project_name, &raw_edges)?;
        if profile {
            eprintln!("[profile] resolve_and_persist_edges {:?}", t.elapsed());
        }
    }

    // Structural spine (Project / Folder / File nodes + CONTAINS_FILE /
    // CONTAINS_FOLDER / DEFINES edges) — builds the structural pass plus the
    // File→DEFINES edges. Runs AFTER
    // all per-file nodes exist (both paths above have written them) so the
    // File→DEFINES targets are resolvable. Node/edge upserts make it
    // idempotent across incremental re-indexes; a deleted file's File node is
    // removed by the per-file node cascade in `run_incremental`.
    structural::build_structural(store, project_name, &entries)?;

    // `require`/`import`→File IMPORTS. A path-style import
    // (Ruby `require 'record'`, Clojure `(:require ..)`, Elm/Erlang/Zig/Dart
    // module imports) resolves to the imported FILE node. The edge-resolution
    // pass (above) only targets symbol definitions and runs BEFORE the File
    // nodes exist, so those imports drop. This post-structural pass adds them:
    // for each raw IMPORTS edge whose name does NOT resolve to a symbol, if it
    // maps to exactly one File basename stem, link the importer's Module node
    // to that File. Symbol-resolving imports (rust/python/java — already at
    // parity) are re-checked and skipped, so nothing is double-counted.
    resolve_file_imports(store, project_name)?;

    record_control_skips(store, project_name, &controlled_entries.skipped, generation)?;

    // R3.5 diagnostics: record the provider completeness state reflected by
    // this index generation. The provider table is store-owned so query-time
    // diagnostics can expose partial language providers without depending on
    // parser internals.
    sync_provider_states(store, project_name, &all_entries, generation)?;

    report.graph_generation = generation;
    Ok(report)
}

fn indexer_version_for_options(options: &IndexOptions) -> String {
    let scope = options.discover_overrides.scope_key();
    if scope == "default" {
        greppy_core::INDEXER_VERSION_BASE.into()
    } else {
        format!(
            "{};discover_scope={scope}",
            greppy_core::INDEXER_VERSION_BASE
        )
    }
}

/// Full (first) index: classify every file, extract every supported file
/// in parallel, write all nodes/content/file_state, and persist every
/// file's raw edges. This is the original PHASE A behaviour with raw-edge
/// persistence added so the *next* run can go incremental.
fn run_full(
    store: &mut Store,
    project_name: &str,
    entries: &[InventoryEntry],
    generation: u64,
    worker_count: usize,
    report: &mut IndexReport,
) -> Result<()> {
    let max_size = max_file_size_bytes();

    // ── Classification (serial, cheap stat only) ────────────────────
    let mut supported: Vec<(usize, &InventoryEntry, Language)> = Vec::new();
    for (idx, entry) in entries.iter().enumerate() {
        let lang = greppy_parser::language_for_path(&entry.abs_path);
        if !lang.is_supported() {
            report.files_unsupported_language += 1;
            // Even for unsupported files we record file_state so the
            // freshness check can detect when the file changes:
            // file_state covers every indexed file, not just supported
            // ones.
            record_unsupported_file_state(store, project_name, entry, generation);
            record_index_skip(
                store,
                project_name,
                entry,
                lang.name(),
                "unsupported_language",
                "language provider is unsupported",
                generation,
            )?;
            continue;
        }
        // Skip oversized files before reading them.
        if let Ok(md) = std::fs::metadata(&entry.abs_path) {
            if md.len() > max_size {
                report.files_oversize += 1;
                record_index_skip(
                    store,
                    project_name,
                    entry,
                    lang.name(),
                    "oversize",
                    &format!("file size {} exceeds cap {}", md.len(), max_size),
                    generation,
                )?;
                continue;
            }
        }
        supported.push((idx, entry, lang));
    }

    // ── PHASE A1 — parallel extract (CPU-bound, no store) ───────────
    let profile = std::env::var("GREPPY_PROFILE").is_ok();
    let t_a1 = std::time::Instant::now();
    let (extractions, throttled) = parallel_extract(&supported, worker_count);
    report.throttled_for_memory = throttled;
    if profile {
        eprintln!(
            "[profile]   A1 parallel_extract (parse+extract) {:?}",
            t_a1.elapsed()
        );
    }

    // ── PHASE A2 — serial store writes (single-writer) ──────────────
    let t_a2 = std::time::Instant::now();
    // Content rows are collected here and written in ONE batched transaction
    // after the loop (see insert_file_content_batch) — content-FTS was the
    // dominant cold-index cost and one-commit-per-file was much of it.
    let mut content_batch: Vec<(String, Vec<greppy_store::ContentRow>)> = Vec::new();
    for outcome in extractions {
        match outcome {
            FileOutcome::Extracted {
                rel_path,
                abs_path,
                bytes,
                metadata,
                nodes,
                edges,
            } => {
                match apply_file_nodes(
                    store,
                    project_name,
                    &rel_path,
                    &abs_path,
                    &bytes,
                    metadata,
                    &nodes,
                    generation,
                    false, // content batched below
                ) {
                    Ok(()) => {
                        store.delete_index_skip(project_name, &rel_path)?;
                        report.files_indexed += 1;
                        report.nodes_extracted += nodes.len();
                        // Full source text is not duplicated into SQLite by
                        // default. Exact code search reads the authoritative
                        // worktree through real grep; graph spans and vectors
                        // remain indexed. A private opt-in exists only for
                        // store/FTS regression and comparison runs.
                        if content_indexing_enabled() {
                            let rows = content_rows_from_bytes(&bytes);
                            if !rows.is_empty() {
                                content_batch.push((rel_path.clone(), rows));
                            }
                        }
                        // Persist this file's raw edges for incremental
                        // re-resolution on the next run.
                        persist_raw_edges_for_file(store, project_name, &rel_path, &edges)?;
                    }
                    Err(_) => report.files_unreadable += 1,
                }
            }
            FileOutcome::Unreadable {
                entry,
                language,
                reason,
                detail,
            } => {
                report.files_unreadable += 1;
                record_index_skip(
                    store,
                    project_name,
                    &entry,
                    language.name(),
                    reason,
                    &detail,
                    generation,
                )?;
            }
        }
    }
    // One transaction for ALL files' content (vs one per file before).
    if !content_batch.is_empty() {
        store.insert_file_content_batch(project_name, &content_batch)?;
    }
    if profile {
        eprintln!(
            "[profile]   A2 serial store writes (nodes+optional-content+raw_edges) {:?}",
            t_a2.elapsed()
        );
    }
    Ok(())
}

/// Incremental index: re-extract + rewrite ONLY changed files, drop
/// deleted files, keep unchanged files' nodes + raw edges. The targeted
/// edge re-resolve in [`index`] then rebuilds only the affected edges.
///
/// Returns the set of rel_paths that this run changed (Added | Modified |
/// Deleted). PHASE B uses it to scope edge re-resolution: PHASE A's
/// FK-cascade already removed every edge with an endpoint in one of these
/// files, so only those (plus edges that named a now-changed definition)
/// need re-resolving — not the whole project.
fn run_incremental(
    store: &mut Store,
    project_name: &str,
    entries: &[InventoryEntry],
    generation: u64,
    worker_count: usize,
    report: &mut IndexReport,
) -> Result<std::collections::HashSet<String>> {
    let max_size = max_file_size_bytes();
    let mut changed_files: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Diff the on-disk inventory against the persisted file_state.
    let diffs = greppy_freshness::compute_file_diff(store, project_name, entries)?;

    // Map rel_path → inventory entry so a diff can recover the abs_path /
    // language for re-extraction.
    let by_rel: std::collections::HashMap<&str, &InventoryEntry> =
        entries.iter().map(|e| (e.rel_path.as_str(), e)).collect();

    // Collect the changed (Added | Modified) supported files to re-extract
    // in parallel; handle Deleted + Unchanged inline.
    let mut changed: Vec<(usize, &InventoryEntry, Language)> = Vec::new();
    for diff in &diffs {
        match diff {
            greppy_freshness::FileDiff::Unchanged => {
                report.files_skipped += 1;
            }
            greppy_freshness::FileDiff::Deleted(rel) => {
                // Remove the file's nodes (FK-cascades its edges), content,
                // file_state, and its persisted raw edges.
                let _ = store.delete_nodes_for_file(project_name, rel)?;
                let _ = store.delete_file_content(project_name, rel)?;
                store.delete_file_state(project_name, rel)?;
                store.delete_index_skip(project_name, rel)?;
                delete_raw_edges_for_file(store, project_name, rel)?;
                changed_files.insert(rel.clone());
            }
            greppy_freshness::FileDiff::Added(entry)
            | greppy_freshness::FileDiff::Modified { entry, .. } => {
                // Any Added/Modified file changes the graph for its own
                // file; record it so PHASE B re-resolves the cascaded edges.
                changed_files.insert(entry.rel_path.clone());
                let Some(&full_entry) = by_rel.get(entry.rel_path.as_str()) else {
                    continue;
                };
                let lang = greppy_parser::language_for_path(&full_entry.abs_path);
                if !lang.is_supported() {
                    report.files_unsupported_language += 1;
                    record_unsupported_file_state(store, project_name, full_entry, generation);
                    record_index_skip(
                        store,
                        project_name,
                        full_entry,
                        lang.name(),
                        "unsupported_language",
                        "language provider is unsupported",
                        generation,
                    )?;
                    // A previously-supported file could have become
                    // unsupported (rename); drop its stale graph + edges.
                    let _ = store.delete_nodes_for_file(project_name, &entry.rel_path)?;
                    let _ = store.delete_file_content(project_name, &entry.rel_path)?;
                    delete_raw_edges_for_file(store, project_name, &entry.rel_path)?;
                    continue;
                }
                if let Ok(md) = std::fs::metadata(&full_entry.abs_path) {
                    if md.len() > max_size {
                        report.files_oversize += 1;
                        record_index_skip(
                            store,
                            project_name,
                            full_entry,
                            lang.name(),
                            "oversize",
                            &format!("file size {} exceeds cap {}", md.len(), max_size),
                            generation,
                        )?;
                        // Oversized now: drop any stale graph for it.
                        let _ = store.delete_nodes_for_file(project_name, &entry.rel_path)?;
                        let _ = store.delete_file_content(project_name, &entry.rel_path)?;
                        delete_raw_edges_for_file(store, project_name, &entry.rel_path)?;
                        continue;
                    }
                }
                // Preserve the inventory index so the parallel extract keeps
                // its deterministic ordering contract.
                let idx = entries
                    .iter()
                    .position(|e| e.rel_path == full_entry.rel_path)
                    .unwrap_or(0);
                changed.push((idx, full_entry, lang));
            }
        }
    }

    // Re-extract the changed files in parallel, then apply writes serially
    // in inventory order (same determinism contract as the full path).
    let (extractions, throttled) = parallel_extract(&changed, worker_count);
    report.throttled_for_memory = throttled;
    for outcome in extractions {
        match outcome {
            FileOutcome::Extracted {
                rel_path,
                abs_path,
                bytes,
                metadata,
                nodes,
                edges,
            } => {
                match apply_file_nodes(
                    store,
                    project_name,
                    &rel_path,
                    &abs_path,
                    &bytes,
                    metadata,
                    &nodes,
                    generation,
                    content_indexing_enabled(), // incremental per-file content
                ) {
                    Ok(()) => {
                        store.delete_index_skip(project_name, &rel_path)?;
                        report.files_indexed += 1;
                        report.nodes_extracted += nodes.len();
                        persist_raw_edges_for_file(store, project_name, &rel_path, &edges)?;
                    }
                    Err(_) => report.files_unreadable += 1,
                }
            }
            FileOutcome::Unreadable {
                entry,
                language,
                reason,
                detail,
            } => {
                report.files_unreadable += 1;
                record_index_skip(
                    store,
                    project_name,
                    &entry,
                    language.name(),
                    reason,
                    &detail,
                    generation,
                )?;
            }
        }
    }

    // Every persisted file or skip row that survived this run reflects the
    // run that confirmed it, even when we did not rewrite its content. We
    // skip re-hashing unchanged files, but stamp the current generation onto
    // all remaining rows in one transaction so `last_indexed_generation`
    // advances exactly as it does on a full reindex (the
    // `file_state_records_real_generation_stamp` contract). Deleted files'
    // rows are already gone; changed files were just written with this
    // generation, so this is idempotent for them.
    bump_all_persisted_generations(store, project_name, generation)?;
    Ok(changed_files)
}

/// Bulk-stamp `generation` onto every `file_state` and `index_skips` row for
/// `project`. Two statements share one transaction, so the update remains
/// atomic and uses O(1) round-trips regardless of file count.
fn bump_all_persisted_generations(store: &mut Store, project: &str, generation: u64) -> Result<()> {
    store.bump_file_and_skip_generations(project, generation)?;
    Ok(())
}

/// Result of extracting one file in the parallel phase. The bytes are
/// retained so the serial phase can hash them for `file_state` and split
/// them into content rows without re-reading the file (which would also
/// be racy under concurrent edits).
enum FileOutcome {
    Extracted {
        rel_path: String,
        abs_path: std::path::PathBuf,
        bytes: Vec<u8>,
        metadata: StableFileMetadata,
        nodes: Vec<ExtractedNode>,
        edges: Vec<ExtractedEdge>,
    },
    /// The file could not be read or parsed; counted as `files_unreadable`.
    Unreadable {
        entry: InventoryEntry,
        language: Language,
        reason: &'static str,
        detail: String,
    },
}

/// Read + parse + extract one file. Pure with respect to the store, so
/// it is safe to call from many rayon worker threads at once.
fn extract_one(entry: &InventoryEntry, lang: Language) -> FileOutcome {
    let (bytes, metadata) = match read_stable_file(&entry.abs_path) {
        Ok(value) => value,
        Err(e) => {
            return FileOutcome::Unreadable {
                entry: entry.clone(),
                language: lang,
                reason: "unreadable",
                detail: format!("read failed: {e}"),
            };
        }
    };
    match parser_extract(lang, &bytes, &entry.rel_path) {
        Ok(extraction) => {
            let provider_output = ProviderOutput::from_extraction(
                manifest_for_language(lang),
                lang,
                &entry.rel_path,
                extraction.clone(),
            );
            if provider_output.validate().is_err() {
                return FileOutcome::Unreadable {
                    entry: entry.clone(),
                    language: lang,
                    reason: "provider_invalid",
                    detail: "provider output failed contract validation".into(),
                };
            }
            FileOutcome::Extracted {
                rel_path: entry.rel_path.clone(),
                abs_path: entry.abs_path.clone(),
                bytes,
                metadata,
                nodes: extraction.nodes,
                edges: extraction.edges,
            }
        }
        Err(e) => FileOutcome::Unreadable {
            entry: entry.clone(),
            language: lang,
            reason: "parse_failed",
            detail: e.to_string(),
        },
    }
}

/// Extract every supported file. Returns the per-file outcomes **in
/// inventory order** plus a flag indicating whether the memory budget
/// forced a throttle to the sequential path.
///
/// Concurrency model:
/// - A private `rayon::ThreadPool` bounds the worker threads to
///   `worker_count` (cgroup-aware via [`greppy_core::default_worker_count`])
///   regardless of the size of the global pool.
/// - The file list is processed in bounded chunks. Before each chunk we
///   check [`greppy_core::mem_over_budget`]; once it trips we stop
///   fanning out and drain the remaining files one at a time (still
///   in order), so a memory-constrained run degrades to sequential
///   instead of allocating every file's bytes + tree at once.
/// - Output order is independent of completion order: results are
///   written back into a pre-sized vector by inventory index, so the
///   graph is deterministic.
fn parallel_extract(
    supported: &[(usize, &InventoryEntry, Language)],
    worker_count: usize,
) -> (Vec<FileOutcome>, bool) {
    let n = supported.len();
    if n == 0 {
        return (Vec::new(), false);
    }

    // Single core / single worker → just go sequential; the parallel
    // machinery would only add overhead and the result is identical.
    if worker_count <= 1 || n == 1 {
        let out = supported
            .iter()
            .map(|(_, entry, lang)| extract_one(entry, *lang))
            .collect();
        return (out, false);
    }

    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .thread_name(|i| format!("greppy-index-{i}"))
        .build()
    {
        Ok(p) => p,
        // If we cannot build a pool, never fail the index — fall back to
        // a correct sequential extract.
        Err(_) => {
            let out = supported
                .iter()
                .map(|(_, entry, lang)| extract_one(entry, *lang))
                .collect();
            return (out, false);
        }
    };

    // Pre-size the output so we can place results by position. `None`
    // marks a not-yet-filled slot; every slot is filled before return.
    let mut slots: Vec<Option<FileOutcome>> = (0..n).map(|_| None).collect();
    let mut throttled = false;

    // Use a wider work-stealing window than the worker count. A four-file
    // barrier on a four-P-core Mac serialized repositories containing one
    // very large source file beside three tiny files: three workers went idle
    // until the large parse completed before the next wave could start. Rayon
    // still executes at most `worker_count` parses simultaneously; the wider
    // window only gives idle workers enough queued files to steal. We retain
    // periodic memory-budget checks between bounded windows.
    const WORK_STEALING_WINDOW_MULTIPLIER: usize = 16;
    let chunk = worker_count
        .saturating_mul(WORK_STEALING_WINDOW_MULTIPLIER)
        .max(worker_count)
        .max(1);
    let mut pos = 0usize;
    while pos < n {
        // Memory throttle: once over budget, finish the remaining
        // files sequentially so we never hold the whole repo's bytes +
        // parse trees in memory at once.
        if greppy_core::mem_over_budget() {
            throttled = true;
            for (slot, (_, entry, lang)) in slots[pos..].iter_mut().zip(&supported[pos..]) {
                *slot = Some(extract_one(entry, *lang));
            }
            break;
        }

        let end = (pos + chunk).min(n);
        let window = &supported[pos..end];
        let results: Vec<FileOutcome> = pool.install(|| {
            window
                .par_iter()
                .map(|(_, entry, lang)| extract_one(entry, *lang))
                .collect()
        });
        for (slot, res) in slots[pos..end].iter_mut().zip(results) {
            *slot = Some(res);
        }
        pos = end;
    }

    let out = slots
        .into_iter()
        .map(|s| s.expect("every slot is filled before return"))
        .collect();
    (out, throttled)
}

/// The serial write for one file: delete any prior rows for that file,
/// then insert the current nodes + file-content + file_state.
/// This is the **serial** half of the pipeline — the bytes/nodes were
/// already produced in parallel by [`extract_one`]; here we only touch
/// the (single-writer) store. Edges are resolved later in the second
/// phase.
///
/// The body takes pre-extracted inputs and a precomputed `generation`,
/// so the resulting rows are byte-for-byte what a fully sequential
/// indexer would write. See the top-of-file doc comment for context.
/// Whether to eagerly index full file content into the content-FTS table.
/// Eager content FTS is a private comparison mode, not the product default.
/// Duplicating every source line made cold indexing and the SQLite store grow
/// dramatically while exact search can read the fresher worktree through the
/// byte-compatible real-grep backend.
fn content_indexing_enabled() -> bool {
    std::env::var("GREPPY_CONTENT_FTS")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
}

#[allow(clippy::too_many_arguments)]
fn apply_file_nodes(
    store: &mut Store,
    project: &str,
    rel_path: &str,
    abs_path: &Path,
    bytes: &[u8],
    metadata: StableFileMetadata,
    nodes: &[ExtractedNode],
    generation: u64,
    // When false, this file's content rows are NOT written here — the caller
    // (the full-index path) batches every file's content into ONE transaction
    // afterwards (content-FTS was ~64% of cold-index cost, dominated by one
    // commit per file). The incremental path passes true: it re-indexes few
    // files and needs the per-file delete-then-insert for correctness.
    insert_content: bool,
) -> Result<()> {
    // Delete any prior nodes for this file BEFORE inserting
    // the new ones. The FK cascade on `edges.source_id` /
    // `edges.target_id` removes orphaned edges in the same
    // transaction. This per-file deletion is preserved exactly under
    // the two-phase split — each file is still individually cleaned
    // before its fresh nodes land.
    let _removed = store.delete_nodes_for_file(project, rel_path)?;

    // Also delete prior file-content rows before the new
    // content is inserted, so a renamed/removed symbol's content
    // does not linger as "this line still says X". Only on the per-file
    // content path; the batched full-index path has no prior content.
    if insert_content {
        let _removed_content = store.delete_file_content(project, rel_path)?;
    }

    // Persist a real per-file **Module** node so `IMPORTS`
    // edges (whose parser source endpoint is the synthetic
    // `<file>::__file__` qname) have a genuine, resolvable source node
    // instead of a dangling synthetic. The qname is kept identical to
    // the parser's file qname so the existing source-qname lookup in
    // `resolve_and_persist_edges` finds it with no parser change. It is
    // re-created every run (the delete above removes the prior one), so
    // it carries no stale state. This is deliberately minimal: one
    // Module node per file, keyed by the file path.
    //
    // The Module node + every extracted node for this file are inserted
    // in ONE batched transaction via `Store::insert_nodes` instead of
    // one self-committing transaction per node — avoiding an fsync
    // amplification DoS. A file with N symbols now costs a single fsync
    // rather than N+1. The row contents, the contentless-FTS token
    // writes, and the insertion order are identical to a straightforward
    // per-node `insert_node` loop, so determinism (delete-then-insert is
    // still done above) and the generation stamp are unchanged.
    let module_node = ExtractedNode {
        label: "Module".into(),
        name: module_name_for(rel_path),
        qualified_name: file_module_qname(rel_path),
        file_path: rel_path.into(),
        start_line: 1,
        end_line: 1,
        properties: serde_json::json!({ "kind": "module", "synthetic": true }),
    };
    let mut new_nodes: Vec<NewNode> = Vec::with_capacity(nodes.len() + 1);
    new_nodes.push(new_node_for(project, rel_path, module_node));
    for n in nodes {
        new_nodes.push(new_node_for(project, rel_path, n.clone()));
    }
    store.insert_nodes(&new_nodes)?;

    // Feed indexed file-content rows for `search-code`. We
    // split the source on newlines, take each line as one snippet,
    // and let the store's contentless FTS5 mirror index it. The
    // full-index path passes insert_content=false and batches this
    // afterwards (one transaction for the whole repo).
    if insert_content {
        let content_rows = content_rows_from_bytes(bytes);
        if !content_rows.is_empty() {
            store.insert_file_content_rows(project, rel_path, &content_rows)?;
        }
    }

    // File state with the real generation stamp. The generation
    // is the one bumped at the start of this index() invocation — it
    // represents the run that wrote this row.
    let fs = FileState {
        project: project.to_string(),
        rel_path: rel_path.to_string(),
        language: greppy_parser::language_for_path(abs_path)
            .name()
            .to_string(),
        sha256: file_state::sha256_hex(bytes),
        mtime_ns: metadata.mtime_ns.unwrap_or(0),
        size: bytes.len() as i64,
        parser_version: format!("tree-sitter-{}", tree_sitter_version()),
        extractor_version: "greppy-extractor-v1".into(),
        last_indexed_generation: generation,
    };
    store.upsert_file_state(&fs)?;
    store.upsert_file_identity(
        project,
        rel_path,
        FileIdentity {
            ctime_ns: metadata.ctime_ns,
            file_id: metadata.file_id,
        },
    )?;
    Ok(())
}

/// PHASE B. Resolve and persist every buffered edge now that all
/// nodes for all files exist. Returns the number of edges actually
/// inserted.
///
/// Resolution is per edge-type so each new kind mirrors the CALLS
/// name-based path while pointing at the right definition kind:
///
/// - **CALLS** — direct same-file qname target if it exists, else a
///   name-based cross-file resolve of `callee_name` to a unique
///   `Function`/`Method`; if no callable resolves, fall back to a unique
///   constructable class/type (`greppy_resolver::resolve_call`).
/// - **TYPE_REF** — direct same-file qname target if it exists, else a
///   name-based cross-file resolve of `type_name` to a unique
///   `Struct`/`Enum`/`Trait`/`TypeAlias` (`resolve_type_ref`).
/// - **USES** — name-based resolve of `ref_name` to a unique definition
///   of any resolvable kind (`resolve_use`). The parser's `__ref__`
///   guess qname is never a real node, so there is no direct path.
/// - **IMPORTS** — name-based resolve of `imported_name` to the unique
///   *defined* node anywhere in the project (`unique_def_named` over
///   `IMPORTABLE_LABELS`). The source endpoint is the per-file `Module`
///   node (qname `<file>::__file__`, persisted in `apply_file_nodes`),
///   so an IMPORTS edge now has BOTH endpoints real. We deliberately do
///   NOT fall back to the synthetic `Import` node target — the point of
///   this pass is to link the import to its declaration.
///
/// Labels a `CALLS` edge may resolve to. Kept in lock-step with
/// `greppy_resolver::resolve_call`'s candidate set (which is private to
/// that crate); the determinism + cross-file tests guard the agreement.
const CALLABLE_LABELS: [&str; 2] = ["Function", "Method"];

/// Labels a `CALLS` edge may resolve to after callable resolution fails.
/// Kept in lock-step with `greppy_resolver::resolve_call`'s constructable
/// fallback.
const CONSTRUCTABLE_LABELS: [&str; 4] = ["Class", "Struct", "Type", "Enum"];

/// Labels a `TYPE_REF` edge may resolve to (the resolver's `TYPE_LABELS`).
/// Rust type defs use the canonical graph labels (struct/union → `Class`,
/// trait → `Interface`, enum → `Enum`, type alias → `Type`); the alternate
/// `Struct`/`Trait`/`TypeAlias` labels are retained for backward
/// compatibility.
const TYPE_LABELS: [&str; 7] = [
    "Class",
    "Interface",
    "Type",
    "Enum",
    "Struct",
    "Trait",
    "TypeAlias",
];

/// Labels a `USES` edge may resolve to (mirrors the resolver's
/// `DEF_LABELS`), including named values and fields.
const DEF_LABELS: [&str; 11] = [
    "Function",
    "Method",
    "Class",
    "Interface",
    "Type",
    "Enum",
    "Struct",
    "Trait",
    "TypeAlias",
    "Variable",
    "Field",
];

/// Labels a `USAGE` edge may resolve to. The usage pass resolves a
/// reference name against every symbol the definitions pass registered —
/// Function/Method/Class/Interface plus Variable/Field. We take the union
/// of the resolvable def labels and the member labels so an identifier
/// reference can land on a value (`Variable`/`Field`) as well as a type
/// or callable.
const USAGE_LABELS: [&str; 11] = [
    "Function",
    "Method",
    "Class",
    "Interface",
    "Type",
    "Enum",
    "Struct",
    "Trait",
    "TypeAlias",
    "Variable",
    "Field",
];

/// Every name-based resolve obeys the resolver's uniqueness rule: a hit
/// only when the name maps to exactly one definition project-wide; zero
/// or ambiguous → skipped (never guessed). An edge whose source qname
/// does not resolve is skipped.
///
/// ## Scale
///
/// A per-edge approach would issue **per-edge SQLite queries**: one
/// `get_node_by_qname` for the source, then a name-based resolver call
/// that runs `list_nodes_by_name` (and, for ambiguous names, an extra
/// `outgoing_edges` round-trip) *for every edge*. With `E` edges and the
/// per-query fixed cost, the edge-resolution phase would dominate indexing
/// and scale super-linearly on a large corpus (measured on such an
/// approach: 500 files → 45 s, 1000 files → 168 s in the debug build —
/// ~3.7× for 2× input).
///
/// Instead we build an in-memory [`GraphIndex`] **once** per run by loading
/// every node for the project in a single query into a `qname → node` map
/// and a `name → [nodes]` multimap, then resolve all edges against those
/// maps with zero further SQLite reads. The IMPORTS pass records each
/// file's resolved import targets into the same in-memory index so the
/// reference resolver can read them back for disambiguation — the
/// "persist IMPORTS first, read back per file" contract, but without any
/// database round-trips. Finally every resolved
/// edge is inserted inside a **single batched transaction** instead of one
/// transaction per edge.
///
/// The resolution *semantics* are byte-for-byte identical to the
/// `greppy-resolver` path (same-file preference, project-wide
/// uniqueness, import disambiguation, use-path module disambiguation, the
/// CALLS-keeps-self-loops rule). The determinism test
/// (`parallel_and_sequential_indexers_produce_identical_graph`) and the
/// cross-file resolution tests enforce that.
fn resolve_and_persist_edges(
    store: &mut Store,
    project: &str,
    edges: &[ExtractedEdge],
) -> Result<usize> {
    // Build the in-memory index ONCE (single query over the project's
    // nodes) instead of querying the store per edge.
    let mut index = GraphIndex::load(store, project)?;

    // Resolve `IMPORTS` edges BEFORE the reference edges (CALLS / TYPE_REF
    // / USES). The import-based disambiguation reads a file's resolved
    // `IMPORTS` targets to break ties between same-named definitions, so
    // those must be recorded first. Within each pass we preserve the
    // original (parser-emission) order, so the graph stays deterministic;
    // only the relative order of the two edge-type groups changes, and an
    // IMPORTS edge and a reference edge never share endpoints, so no
    // edge's resolution is affected by the reordering.
    let mut resolved: Vec<NewEdge> = Vec::with_capacity(edges.len());

    // PASS 1 — IMPORTS. Record each resolved target into the index so the
    // reference pass can read a file's imports back.
    for edge in edges.iter().filter(|e| e.edge_type == "IMPORTS") {
        let Some(src) = index.by_qname(&edge.source_qualified_name) else {
            continue;
        };
        let src_id = src.id;
        let src_file = src.file_path.clone();
        let target_id = match edge
            .properties
            .get("imported_name")
            .and_then(|v| v.as_str())
        {
            Some(name) if !name.is_empty() => {
                let path = edge
                    .properties
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                index.unique_def_named_with_path(&greppy_resolver::IMPORTABLE_LABELS, name, path)
            }
            // Brace groups / globs / renames leave imported_name empty —
            // a future expansion pass owns those.
            _ => None,
        };
        let Some(target_id) = target_id else { continue };
        // IMPORTS drops self-loops (only CALLS keeps them).
        if target_id == src_id {
            continue;
        }
        // Record the import so the reference resolver can disambiguate.
        index.record_import(&src_file, target_id);
        resolved.push(new_edge(project, src_id, target_id, edge));
    }

    // PASS 2 — reference edges (CALLS / TYPE_REF / USES / other).
    for edge in edges.iter().filter(|e| e.edge_type != "IMPORTS") {
        let Some(src) = index.by_qname(&edge.source_qualified_name) else {
            continue;
        };
        let src_id = src.id;

        let target_id = match edge.edge_type.as_str() {
            "CALLS" => index.resolve_call_target(edge),
            "TYPE_REF" => index.resolve_direct_or_name(edge, "type_name", &TYPE_LABELS),
            "USES" => match edge.properties.get("ref_name").and_then(|v| v.as_str()) {
                // No real direct-target node exists for the `__ref__`
                // guess qname; go straight to the name-based resolver.
                Some(name) if !name.is_empty() => {
                    index.resolve_unique_with_imports(&DEF_LABELS, name, src_id)
                }
                _ => None,
            },
            // Any other edge type: direct qname target only.
            // USAGE — a per-language usages pass emits a reference by name;
            // resolve it to any registered symbol (callable, type, or value)
            // via the symbol registry. No direct target qname exists, so this
            // is name-based only.
            "USAGE" => match edge.properties.get("ref_name").and_then(|v| v.as_str()) {
                Some(name) if !name.is_empty() => {
                    index.resolve_unique_with_imports(&USAGE_LABELS, name, src_id)
                }
                _ => None,
            },
            _ => index.by_qname(&edge.target_qualified_name).map(|n| n.id),
        };

        let Some(target_id) = target_id else { continue };

        // An edge must connect two DISTINCT nodes; a self-loop here is
        // almost always a same-file guess qname accidentally matching the
        // source (e.g. a USES of the enclosing symbol's own name). CALLS
        // keeps self-loops (direct recursion is legitimate); the others
        // drop them.
        if target_id == src_id && edge.edge_type != "CALLS" {
            continue;
        }
        resolved.push(new_edge(project, src_id, target_id, edge));
    }

    // Persist every resolved edge in a SINGLE transaction (was: one
    // transaction per edge). Determinism is unchanged — the edge order is
    // the same IMPORTS-then-references order resolved above.
    insert_edges_batched(store, &resolved)?;
    Ok(resolved.len())
}

/// Map a raw `rusqlite::Error` into the indexer's `greppy_core::Error`.
/// The indexer normally goes through typed `Store` methods (which own this
/// conversion); the few remaining raw-connection paths here (the one-shot
/// node load, the batched edge insert, the def-fingerprint scan and the
/// `file_state` generation bump) need it explicitly.
fn sqlite_err(e: rusqlite::Error) -> greppy_core::Error {
    greppy_core::Error::Store(format!("sqlite: {e}"))
}

/// Whether the pre-migration `indexer_raw_edges` sidecar table is present in
/// the store. The store API has no notion of this legacy table (it owns
/// `raw_edges` instead), so this is a deliberately local `sqlite_master`
/// probe — NOT a raw-edge CRUD shape. It exists only to tell a store last
/// touched by the OLD indexer binary (sidecar present, `raw_edges` empty)
/// apart from a legitimately edgeless repo last indexed by THIS binary
/// (sidecar absent, `raw_edges` empty), so the incremental-vs-full decision
/// stays correct across an upgrade. A future cleanup wave can drop the
/// sidecar and delete this probe.
fn legacy_raw_edge_sidecar_exists(store: &Store) -> Result<bool> {
    let n: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'indexer_raw_edges'",
            [],
            |r| r.get(0),
        )
        .map_err(sqlite_err)?;
    Ok(n > 0)
}

// ── Raw-edge persistence (store-owned `raw_edges` table) ───────────────
//
// The store crate owns the **raw, unresolved** edges the parser extracted
// (migration 0007: the `raw_edges` table + `NewRawEdge` / `RawEdge` and the
// `insert_raw_edges` / `list_raw_edges` / `delete_raw_edges_for_file` /
// `count_raw_edges` API). The indexer drives this typed store API rather
// than an ad-hoc `indexer_raw_edges` sidecar via `conn()`/`execute_batch`,
// which keeps the row layout and behaviour explicit:
//
// - Rows are keyed by `(project, file_path)` where `file_path` is the
//   **owner file** — the file whose extraction produced the edge. The parser
//   stamps every edge's `file_path` with the same value (the extracted
//   file's rel_path), so the owner key equals `ExtractedEdge::file_path` and
//   `source_file_of` still recovers it after a round-trip.
// - Per-file delete-then-insert: a file's contribution is replaced
//   wholesale before its fresh edges land.
// - `list_raw_edges` returns rows ordered by `(file_path, id)`, the same
//   deterministic order the old `ORDER BY file_path, rowid` produced, so the
//   project-wide raw-edge set the resolver runs over is unchanged.
//
// The store models a raw edge as the five resolution-relevant columns
// `(source_qname, target_qname, edge_type, properties)` plus `file_path`.
// `ExtractedEdge` additionally carries a `line`, which no resolution or
// insertion path consults (see `new_edge` / `resolve_*` — only edge_type,
// the two qnames, file_path and properties are read). It is therefore
// dropped on persist and reconstructed as `0` on read-back; the graph the
// resolver produces is identical.

/// Convert a parser [`ExtractedEdge`] into a store [`NewRawEdge`], keyed by
/// `owner_file` (the file whose extraction produced it — equal to the edge's
/// own `file_path`). The extraction `line` (the CALL SITE / reference site)
/// is folded into the properties JSON: the nav commands print it grep-shaped
/// (`file:line: code`) so one who-calls answer carries the evidence an agent
/// would otherwise re-read files for (problem dossier P4).
fn new_raw_edge_for(project: &str, owner_file: &str, e: &ExtractedEdge) -> NewRawEdge {
    let mut properties = e.properties.clone();
    if e.line > 0 {
        if let Some(map) = properties.as_object_mut() {
            map.insert("line".into(), serde_json::json!(e.line));
        }
    }
    NewRawEdge {
        project: project.to_string(),
        file_path: owner_file.to_string(),
        source_qname: e.source_qualified_name.clone(),
        target_qname: e.target_qualified_name.clone(),
        edge_type: e.edge_type.clone(),
        properties,
    }
}

/// Reconstruct an [`ExtractedEdge`] from a persisted store [`RawEdge`]. The
/// `line` is set to `0` — it is never consulted by edge resolution or
/// insertion, and was only ever round-tripped through the old JSON sidecar.
fn extracted_edge_from_raw(r: RawEdge) -> ExtractedEdge {
    ExtractedEdge {
        edge_type: r.edge_type,
        source_qualified_name: r.source_qname,
        target_qualified_name: r.target_qname,
        file_path: r.file_path,
        line: r
            .properties
            .get("line")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        properties: r.properties,
    }
}

/// Delete the persisted raw edges for `(project, file_path)`. Called before
/// re-inserting a re-extracted file's edges, and for deleted files. Thin
/// wrapper over [`Store::delete_raw_edges_for_file`].
fn delete_raw_edges_for_file(store: &mut Store, project: &str, file_path: &str) -> Result<()> {
    store.delete_raw_edges_for_file(project, file_path)?;
    Ok(())
}

/// Persist the raw edges a file produced (replacing any prior rows for that
/// file). The edges' own `file_path` is the parser's per-edge file; we key
/// the rows by `owner_file` (the file whose extraction produced them) so a
/// re-extract of that file replaces exactly its contribution. Drives the
/// store's per-file delete-then-insert.
fn persist_raw_edges_for_file(
    store: &mut Store,
    project: &str,
    owner_file: &str,
    edges: &[ExtractedEdge],
) -> Result<()> {
    delete_raw_edges_for_file(store, project, owner_file)?;
    if edges.is_empty() {
        return Ok(());
    }
    let new_edges: Vec<NewRawEdge> = edges
        .iter()
        .map(|e| new_raw_edge_for(project, owner_file, e))
        .collect();
    store.insert_raw_edges(&new_edges)?;
    Ok(())
}

/// Load every persisted raw edge for `project`, in a deterministic order
/// (`file_path`, then insert id so a file's edges keep their emission
/// order — exactly the order [`Store::list_raw_edges`] returns). This is the
/// project-wide raw-edge set the resolver runs over on the incremental path;
/// on a full run we resolve the freshly-extracted edges directly and only use
/// this table for the *next* incremental run.
fn load_all_raw_edges(store: &Store, project: &str) -> Result<Vec<ExtractedEdge>> {
    let rows = store.list_raw_edges(project)?;
    Ok(rows.into_iter().map(extracted_edge_from_raw).collect())
}

// Test-only instrumentation: the number of raw edges PHASE B actually fed
// through the resolver on the most recent incremental run. A no-op reindex
// must leave this at 0; a pure body edit must leave it far below the
// project's total edge count. Behind `cfg(test)` so it adds nothing to the
// shipped binary and does not touch any public API.
//
// It is **thread-local** so that the many tests that call `index()` in
// parallel each observe only their own resolution counts — a global atomic
// would be raced by every concurrent `index()`.
#[cfg(test)]
thread_local! {
    static LAST_EDGES_RERESOLVED_TLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LAST_EDGE_RESOLUTION_WORK_TLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_reresolve_counter() {
    LAST_EDGES_RERESOLVED_TLS.with(|c| c.set(0));
}

#[cfg(test)]
fn reresolve_count() -> usize {
    LAST_EDGES_RERESOLVED_TLS.with(|c| c.get())
}

#[cfg(test)]
fn note_reresolved(n: usize) {
    LAST_EDGES_RERESOLVED_TLS.with(|c| c.set(c.get() + n));
}

#[cfg(test)]
fn reset_edge_resolution_work_counter() {
    LAST_EDGE_RESOLUTION_WORK_TLS.with(|c| c.set(0));
}

#[cfg(test)]
fn edge_resolution_work_count() -> usize {
    LAST_EDGE_RESOLUTION_WORK_TLS.with(|c| c.get())
}

#[cfg(test)]
fn note_edge_resolution_work(n: usize) {
    LAST_EDGE_RESOLUTION_WORK_TLS.with(|c| c.set(c.get() + n));
}

#[cfg(not(test))]
#[inline]
fn note_reresolved(_n: usize) {}

#[cfg(not(test))]
#[inline]
fn note_edge_resolution_work(_n: usize) {}

/// The project's **definition fingerprint**: the sorted set of node
/// identity tuples that cross-file edge resolution actually consults —
/// `qualified_name`, `name`, `label`, `file_path`. Edge resolution is a
/// pure function of this set (plus the per-edge raw data): `by_qname`
/// targets, `by_name` candidate sets, the same-file preference, and IMPORTS
/// disambiguation all read only these four columns. Node `id`s are
/// deliberately excluded — they are autoincrement and change on
/// re-extraction even for byte-identical content, but a changed id never
/// changes *which* definition a name resolves to.
///
/// Comparing this fingerprint before vs after PHASE A tells us whether any
/// changed file altered the resolvable definition set. If it did NOT (a pure
/// body edit — same symbols, same qnames, same files), then no edge from an
/// *unchanged* file can change its resolution, so PHASE B only has to rebuild
/// the edges PHASE A's FK-cascade removed. If it DID, we fall back to a full
/// re-resolution (byte-identical to a first run) because an unchanged file's
/// edge may now resolve, unresolve, or become ambiguous.
fn def_fingerprint(store: &Store, project: &str) -> Result<std::collections::BTreeSet<String>> {
    let conn = store.conn();
    // Exclude the structural spine (Project / Folder / File). Those nodes are
    // materialized by `structural::build_structural` AFTER edge resolution and
    // are re-created (with fresh ids) whenever their owning file is
    // re-extracted, so a changed file would churn them in and out of the
    // fingerprint. They are never edge-resolution targets (CALLS / IMPORTS /
    // TYPE_REF / USES never resolve to a File/Folder/Project), so their
    // presence or absence cannot change how any edge resolves — including them
    // would only defeat the cheap body-edit path without affecting
    // correctness.
    let mut stmt = conn
        .prepare_cached(
            "SELECT qualified_name, name, label, file_path
             FROM nodes WHERE project = ?1
               AND label NOT IN ('Project', 'Folder', 'File')",
        )
        .map_err(sqlite_err)?;
    let rows = stmt
        .query_map(rusqlite::params![project], |r| {
            Ok(format!(
                "{}\u{1f}{}\u{1f}{}\u{1f}{}",
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .map_err(sqlite_err)?;
    let mut set = std::collections::BTreeSet::new();
    for row in rows {
        set.insert(row.map_err(sqlite_err)?);
    }
    Ok(set)
}

/// Count the resolved edges currently persisted for `project`. Used by the
/// incremental no-op path to report `edges_extracted` without re-resolving.
fn count_edges(store: &Store, project: &str) -> Result<usize> {
    let n: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM edges WHERE project = ?1",
            rusqlite::params![project],
            |r| r.get(0),
        )
        .map_err(sqlite_err)?;
    Ok(n as usize)
}

/// PHASE B for the **incremental** path: re-resolve only the edges that this
/// run's changes could have affected, instead of the whole project's raw
/// edges (the O(total edges) hotspot).
///
/// ## Why this is byte-identical to a full re-resolution
///
/// After PHASE A, SQLite's FK-cascade has already removed **every** resolved
/// edge that had either endpoint in a changed/deleted file (the node was
/// dropped). The edges that survive in `edges` are exactly those whose source
/// AND target are in *unchanged* files. The resolution of such a surviving
/// edge is a pure function of the [`def_fingerprint`]; so:
///
/// - **No file changed** (`changed_files` empty) → nothing was cascaded, the
///   def set is identical, and every surviving edge is already correct.
///   We re-resolve **nothing** (the headline no-op win).
/// - **Files changed but the def set did NOT**
///   (`def_fp_before == def_fp_after`) → no surviving (unchanged↔unchanged)
///   edge can change resolution. The only edges missing are those cascaded by
///   a changed-file endpoint, so we re-resolve exactly the raw edges whose
///   **source** is a changed file plus those (from any file) that **name a
///   definition in a changed file** (the only way a target could land in one).
/// - **The def set changed** → an unchanged file's edge may now resolve
///   differently (gain/lose a candidate, gain/lose ambiguity), and the
///   insert-only resolver cannot prove which survivors are stale. We fall
///   back to the **full** re-resolution — byte-for-byte the first-run path —
///   which is correct by construction.
///
/// The IMPORTS-disambiguation contract is preserved: for every file that owns
/// a candidate reference edge we resolve that file's IMPORTS into the index
/// first (exactly as PASS 1 of the full resolver does), so a CALLS/USES/
/// TYPE_REF edge sees the same imported-target set it would on a full run.
fn resolve_edges_incremental(
    store: &mut Store,
    project: &str,
    changed_files: &std::collections::HashSet<String>,
    def_fp_before: &std::collections::BTreeSet<String>,
) -> Result<usize> {
    // No file changed → nothing cascaded, graph already complete. O(1).
    if changed_files.is_empty() {
        return count_edges(store, project);
    }

    // Did a changed file alter the resolvable definition set? If so, an
    // UNCHANGED file's edge could now resolve differently — fall back to the
    // full, insert-only re-resolution (identical to a first run).
    let def_fp_after = def_fingerprint(store, project)?;
    if &def_fp_after != def_fp_before {
        // The resolvable def set changed: a name referenced by an UNCHANGED
        // file may now be ambiguous (or newly resolvable), so its SURVIVING
        // resolved edge could be stale. `resolve_and_persist_edges` is
        // insert-only and would leave that stale edge in place, diverging from
        // a full reindex. Clear ALL of the project's resolved edges first, then
        // re-resolve from scratch so the result is byte-identical to a full
        // first run (the incremental == full correctness invariant).
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE project = ?1",
                rusqlite::params![project],
            )
            .map_err(sqlite_err)?;
        let raw_edges = load_all_raw_edges(store, project)?;
        note_reresolved(raw_edges.len());
        return resolve_and_persist_edges(store, project, &raw_edges);
    }

    // ── Pure body edit(s): def set unchanged. Re-resolve only the cascaded
    //    edges. Build the index once, then resolve the candidate subset. ──
    let mut index = GraphIndex::load(store, project)?;

    // Names defined in a changed file. An unchanged-file edge can only have
    // been cascaded (via its *target*) if it resolved into a changed file,
    // which means it named one of these. (Source-in-changed edges are caught
    // by the owner-file filter below regardless of the name they reference.)
    let changed_def_names = index.names_in_files(changed_files);

    // Load the whole raw-edge set once (a single indexed read; no per-edge
    // resolution). We then resolve ONLY the candidate subset.
    let all_raw = load_all_raw_edges(store, project)?;

    let owned_by_changed = |e: &ExtractedEdge| changed_files.contains(source_file_of(e));
    let names_changed_def = |e: &ExtractedEdge| edge_references_name(e, &changed_def_names);
    let is_candidate = |e: &ExtractedEdge| owned_by_changed(e) || names_changed_def(e);

    // Instrumentation: how many raw edges this cheap path actually resolves.
    note_reresolved(all_raw.iter().filter(|e| is_candidate(e)).count());

    // PASS 1 — IMPORTS. Resolve the imports of every file that owns a
    // candidate reference edge (and every candidate IMPORTS edge itself) so
    // PASS 2's disambiguation matches the full resolver. Because the def set
    // is unchanged, these imports resolve exactly as they did before.
    let mut candidate_owner_files: std::collections::HashSet<&str> =
        std::collections::HashSet::new();
    for e in &all_raw {
        if e.edge_type != "IMPORTS" && is_candidate(e) {
            candidate_owner_files.insert(source_file_of(e));
        }
    }
    let mut resolved: Vec<NewEdge> = Vec::new();
    for edge in all_raw.iter().filter(|e| e.edge_type == "IMPORTS") {
        let Some(src) = index.by_qname(&edge.source_qualified_name) else {
            continue;
        };
        let src_id = src.id;
        let src_file = src.file_path.clone();
        // Only resolve this import if its file owns a candidate reference
        // edge OR the import edge itself is a candidate (so it lands in the
        // index for disambiguation and, when itself cascaded, is re-inserted).
        let is_cand_import = is_candidate(edge);
        if !is_cand_import && !candidate_owner_files.contains(src_file.as_str()) {
            continue;
        }
        let target_id = match edge
            .properties
            .get("imported_name")
            .and_then(|v| v.as_str())
        {
            Some(name) if !name.is_empty() => {
                let path = edge
                    .properties
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                index.unique_def_named_with_path(&greppy_resolver::IMPORTABLE_LABELS, name, path)
            }
            _ => None,
        };
        let Some(target_id) = target_id else { continue };
        if target_id == src_id {
            continue;
        }
        index.record_import(&src_file, target_id);
        // Only persist the import edge if it was actually cascaded (it is a
        // candidate). Non-candidate imports were recorded purely to feed
        // PASS 2 disambiguation; their resolved row already survives in the DB.
        if is_cand_import {
            resolved.push(new_edge(project, src_id, target_id, edge));
        }
    }

    // PASS 2 — reference edges. Resolve only candidates.
    for edge in all_raw.iter().filter(|e| e.edge_type != "IMPORTS") {
        if !is_candidate(edge) {
            continue;
        }
        let Some(src) = index.by_qname(&edge.source_qualified_name) else {
            continue;
        };
        let src_id = src.id;
        let target_id = match edge.edge_type.as_str() {
            "CALLS" => index.resolve_call_target(edge),
            "TYPE_REF" => index.resolve_direct_or_name(edge, "type_name", &TYPE_LABELS),
            "USES" => match edge.properties.get("ref_name").and_then(|v| v.as_str()) {
                Some(name) if !name.is_empty() => {
                    index.resolve_unique_with_imports(&DEF_LABELS, name, src_id)
                }
                _ => None,
            },
            // USAGE — a per-language usages pass emits a reference by name;
            // resolve it to any registered symbol (callable, type, or value)
            // via the symbol registry. No direct target qname exists, so this
            // is name-based only.
            "USAGE" => match edge.properties.get("ref_name").and_then(|v| v.as_str()) {
                Some(name) if !name.is_empty() => {
                    index.resolve_unique_with_imports(&USAGE_LABELS, name, src_id)
                }
                _ => None,
            },
            _ => index.by_qname(&edge.target_qualified_name).map(|n| n.id),
        };
        let Some(target_id) = target_id else { continue };
        if target_id == src_id && edge.edge_type != "CALLS" {
            continue;
        }
        resolved.push(new_edge(project, src_id, target_id, edge));
    }

    insert_edges_batched(store, &resolved)?;
    // The total live edge count = the survivors PHASE A kept + what we just
    // re-inserted (the `ON CONFLICT` upsert means a re-inserted row that
    // happened to survive is not double-counted by the COUNT).
    count_edges(store, project)
}

/// The owner file of a raw edge — the file whose extraction produced it.
/// The parser stamps this on the edge's `file_path`; it equals the edge's
/// source node's file (the `raw_edges` rows are keyed by it too).
fn source_file_of(edge: &ExtractedEdge) -> &str {
    edge.file_path.as_str()
}

/// Whether a raw edge references one of `names` via the property the
/// resolver keys on for its type (`callee_name` / `type_name` / `ref_name` /
/// `imported_name`). Mirrors exactly the property each PASS reads, so the
/// candidate filter is a precise superset of "could resolve into a changed
/// file".
fn edge_references_name(edge: &ExtractedEdge, names: &std::collections::HashSet<String>) -> bool {
    let prop = match edge.edge_type.as_str() {
        "CALLS" => "callee_name",
        "TYPE_REF" => "type_name",
        "USES" => "ref_name",
        "IMPORTS" => "imported_name",
        _ => return false,
    };
    edge.properties
        .get(prop)
        .and_then(|v| v.as_str())
        .map(|n| names.contains(n))
        .unwrap_or(false)
}

/// Build a [`NewEdge`] from a resolved source/target pair, cloning the
/// parser's edge properties.
fn new_edge(project: &str, source_id: i64, target_id: i64, edge: &ExtractedEdge) -> NewEdge {
    NewEdge {
        project: project.to_string(),
        source_id,
        target_id,
        // The compatibility graph schema folds reference kinds into `USAGE`.
        // Providers with certified logical classification may opt into keeping
        // `TYPE_REF` / `USES` distinct via `preserve_reference_kind`.
        edge_type: persisted_edge_label(edge).to_string(),
        properties: {
            // Fold the reference-site line into the resolved edge too (P4):
            // nav commands print it grep-shaped so one answer carries the
            // call-site evidence. Raw edges round-trip it via properties, so
            // `edge.line` is populated on both extract and re-resolve paths.
            let mut props = edge.properties.clone();
            if edge.line > 0 {
                if let Some(map) = props.as_object_mut() {
                    map.insert("line".into(), serde_json::json!(edge.line));
                }
            }
            props
        },
    }
}

/// Map an extraction-time edge to its persisted graph label. Most providers
/// retain the compatibility rule `TYPE_REF`/`USES`/`USAGE` → `USAGE`. A provider
/// that has certified distinct logical reference classes can set
/// `preserve_reference_kind=true` to persist `TYPE_REF` / `USES` verbatim.
fn persisted_edge_label(edge: &ExtractedEdge) -> &str {
    if edge
        .properties
        .get("preserve_reference_kind")
        .and_then(|value| value.as_bool())
        == Some(true)
    {
        edge.edge_type.as_str()
    } else {
        usage_persist_label(&edge.edge_type)
    }
}

fn usage_persist_label(edge_type: &str) -> &str {
    match edge_type {
        "TYPE_REF" | "USES" | "USAGE" => "USAGE",
        other => other,
    }
}

/// Insert every resolved edge inside ONE transaction. The per-edge
/// `Store::insert_edge` opens its own transaction; doing that `E` times is
/// the bulk of the old edge phase's fixed cost. Here we open the
/// transaction once and reuse a single prepared statement, preserving the
/// exact upsert semantics (`ON CONFLICT(source_id, target_id, edge_type)`)
/// and insertion order.
/// `require`/`import`→File IMPORTS. Runs AFTER `build_structural` (File nodes
/// exist) — see the call in `index`. For each raw IMPORTS edge whose name does
/// NOT resolve to a symbol but maps to exactly one File basename stem, link the
/// importer's Module node to that File (the `require`/module-import→File
/// model). Symbol-resolving imports are re-checked and skipped, so nothing is
/// double-counted; `insert_edge` upserts on the unique triple, so re-indexing
/// is idempotent. rust/python/java imports all name symbols → never reach the
/// File branch, so their IMPORTS resolution is unaffected.
fn resolve_file_imports(store: &mut Store, project: &str) -> Result<()> {
    let index = GraphIndex::load(store, project)?;
    let raw = load_all_raw_edges(store, project)?;
    let mut resolved: Vec<NewEdge> = Vec::new();
    for edge in raw.iter().filter(|e| e.edge_type == "IMPORTS") {
        let name = match edge
            .properties
            .get("imported_name")
            .and_then(|v| v.as_str())
        {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let path = edge
            .properties
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if index
            .unique_def_named_with_path(&greppy_resolver::IMPORTABLE_LABELS, name, path)
            .is_some()
        {
            continue; // already resolved to a symbol by the reference pass
        }
        // Only a FILESYSTEM-style import names a file: a bare stem
        // (Ruby `require 'record'`, Erlang `-module(a)`) or a filename with a
        // source extension (Zig `@import("token.zig")`, Bash `source util.sh`).
        // A DOTTED MODULE namespace (PureScript `Data.List`, Clojure
        // `myapp.util`) names a symbol/module, NOT a file — resolving it to a
        // File over-emits, so we skip it. Take the last path segment, strip a
        // trailing SHORT lowercase extension (`.zig`/`.sh`, never `.List`), and
        // require the result to have no interior dot before matching a File.
        let last_path = name.rsplit(['/', '\\']).next().unwrap_or(name);
        let stem = match last_path.rsplit_once('.') {
            Some((base, ext))
                if !ext.is_empty()
                    && ext.len() <= 4
                    && ext
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()) =>
            {
                base
            }
            _ => last_path,
        };
        if stem.contains('.') {
            continue; // dotted module namespace → not a file import
        }
        let target_id = if edge
            .properties
            .get("filesystem_module_import")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            index.resolve_filesystem_module_import(edge, stem)
        } else {
            match index.files_by_stem.get(stem).map(Vec::as_slice) {
                Some([file_id]) => Some(*file_id),
                _ => None,
            }
        };
        let Some(target_id) = target_id else {
            continue; // no target, or an ambiguous stem — never guess
        };
        let Some(src) = index.by_qname(&edge.source_qualified_name) else {
            continue;
        };
        let mut target_ids = vec![target_id];
        if edge
            .properties
            .get("dart_relative_import")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            // A Dart library import exposes every top-level definition
            // from the imported file. Persist the file-Module edge and symbol
            // edges so `find-usages <symbol>` can surface the import directly.
            if let Some(target_file) = index.file_of(target_id) {
                target_ids.extend(
                    index
                        .by_qname
                        .values()
                        .filter(|node| {
                            node.file_path == target_file
                                && greppy_resolver::IMPORTABLE_LABELS
                                    .contains(&node.label.as_str())
                        })
                        .map(|node| node.id),
                );
            }
        }
        target_ids.sort_unstable();
        target_ids.dedup();
        for target_id in target_ids {
            if target_id != src.id {
                resolved.push(new_edge(project, src.id, target_id, edge));
            }
        }
    }
    insert_edges_batched(store, &resolved)
}

fn insert_edges_batched(store: &mut Store, edges: &[NewEdge]) -> Result<()> {
    if edges.is_empty() {
        return Ok(());
    }
    // The store's `Transaction` does not expose its raw connection to other
    // crates, so we drive one explicit transaction on the public
    // `conn()` borrow instead: BEGIN, reuse a single cached prepared
    // statement for every insert, then COMMIT (rolling back on any error).
    // The `ON CONFLICT` clause and column order are byte-for-byte those of
    // `Store::insert_edge`, so the persisted rows are identical — only the
    // transaction boundary moves from per-edge to once-per-run.
    let conn = store.conn();
    conn.execute_batch("BEGIN").map_err(sqlite_err)?;
    let result = (|| -> Result<()> {
        let mut stmt = conn
            .prepare_cached(
                "INSERT INTO edges (project, source_id, target_id, edge_type, properties)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                   properties = excluded.properties",
            )
            .map_err(sqlite_err)?;
        for e in edges {
            let props_str = serde_json::to_string(&e.properties)
                .map_err(|err| greppy_core::Error::Store(format!("json: {err}")))?;
            stmt.execute(rusqlite::params![
                e.project,
                e.source_id,
                e.target_id,
                e.edge_type,
                props_str,
            ])
            .map_err(sqlite_err)?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            conn.execute_batch("COMMIT").map_err(sqlite_err)?;
            Ok(())
        }
        Err(e) => {
            // Best-effort rollback; surface the original error.
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// A node, reduced to the fields cross-file resolution actually consults.
/// Loading these once (instead of full `Node` rows per edge) keeps the
/// in-memory index compact.
#[derive(Clone)]
struct NodeLite {
    id: i64,
    label: String,
    file_path: String,
}

/// Rank graph labels exactly like CLI symbol navigation. When one source symbol
/// has multiple persisted facets (for example Scala Method + Function twins),
/// CALLS resolution and single-node navigation must choose the same facet or a
/// real edge can appear unreachable to `path`.
fn navigation_label_rank(label: &str) -> u8 {
    match label {
        "Class" | "Interface" | "Type" | "Struct" | "Enum" | "Trait" | "Function"
        | "Method" | "TypeAlias" => 0,
        "Impl" | "EnumVariant" | "AssocConst" | "AssocType" | "Module" => 1,
        "Call" | "Import" => 3,
        _ => 2,
    }
}

/// In-memory mirror of the project's node graph, built **once** per index
/// run so edge resolution issues no per-edge SQLite queries. It replicates
/// the `greppy-resolver` semantics exactly:
///
/// - [`by_qname`](GraphIndex::by_qname) — `(qname) → node`, the source /
///   direct-target lookup (was `Store::get_node_by_qname`).
/// - [`defs_named`](GraphIndex::defs_named) — `(name) → [nodes]` filtered
///   by label, in `qualified_name` order (was `Store::list_nodes_by_name`
///   + label filter in `greppy_resolver::defs_named`).
/// - [`imports_by_file`] — each file's resolved IMPORTS target ids,
///   populated during the IMPORTS pass and read back for ambiguity
///   disambiguation (was the persisted IMPORTS edges read via
///   `Store::outgoing_edges`).
struct GraphIndex {
    by_qname: std::collections::HashMap<String, NodeLite>,
    /// `name → nodes sharing that name`, each inner vec sorted by
    /// `qualified_name` so the candidate order matches the old
    /// `list_nodes_by_name` ordering (resolution depends only on the
    /// set + same-file count, but we keep order stable for determinism).
    by_name: std::collections::HashMap<String, Vec<NodeLite>>,
    /// `file_path → resolved IMPORTS target ids` for this file. Filled by
    /// [`record_import`](GraphIndex::record_import) during the IMPORTS
    /// pass; consulted by [`resolve_unique_with_imports`].
    imports_by_file: std::collections::HashMap<String, std::collections::HashSet<i64>>,
    /// `node id → file_path`, so a referrer's file (needed for the
    /// same-file preference) is an O(1) lookup from its id.
    id_to_file: std::collections::HashMap<i64, String>,
    /// `file basename stem → File node ids`. Backs the `require`/`import`→File
    /// IMPORTS pass (Ruby `require 'record'`, Clojure `(:require ..)`, Elm/
    /// Erlang/Zig/Dart module imports). Populated at load; only usable AFTER
    /// the structural pass has created the File nodes.
    files_by_stem: std::collections::HashMap<String, Vec<i64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UniqueResolution {
    Unique(i64),
    Unresolved,
    Ambiguous,
}

impl UniqueResolution {
    fn unique_id(self) -> Option<i64> {
        match self {
            Self::Unique(id) => Some(id),
            Self::Unresolved | Self::Ambiguous => None,
        }
    }
}

impl GraphIndex {
    /// Load every node for `project` in a single query and build the
    /// lookup maps. `qualified_name` order from SQL gives a deterministic
    /// per-name candidate order.
    fn load(store: &Store, project: &str) -> Result<Self> {
        let mut by_qname: std::collections::HashMap<String, NodeLite> =
            std::collections::HashMap::new();
        let mut by_name: std::collections::HashMap<String, Vec<NodeLite>> =
            std::collections::HashMap::new();
        let mut id_to_file: std::collections::HashMap<i64, String> =
            std::collections::HashMap::new();
        let mut files_by_stem: std::collections::HashMap<String, Vec<i64>> =
            std::collections::HashMap::new();
        {
            let conn = store.conn();
            let mut stmt = conn
                .prepare_cached(
                    "SELECT id, name, qualified_name, label, file_path
                     FROM nodes WHERE project = ?1 ORDER BY qualified_name",
                )
                .map_err(sqlite_err)?;
            let rows = stmt
                .query_map(rusqlite::params![project], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, String>(4)?,
                    ))
                })
                .map_err(sqlite_err)?;
            for row in rows {
                let (id, name, qname, label, file_path) = row.map_err(sqlite_err)?;
                note_edge_resolution_work(1);
                if label == "File" {
                    let base = file_path.rsplit('/').next().unwrap_or(&file_path);
                    let stem = base.rsplit_once('.').map_or(base, |(s, _)| s);
                    files_by_stem.entry(stem.to_string()).or_default().push(id);
                }
                let node = NodeLite {
                    id,
                    label,
                    file_path: file_path.clone(),
                };
                id_to_file.insert(id, file_path);
                by_name.entry(name).or_default().push(node.clone());
                by_qname.insert(qname, node);
            }
        }
        Ok(GraphIndex {
            by_qname,
            by_name,
            imports_by_file: std::collections::HashMap::new(),
            id_to_file,
            files_by_stem,
        })
    }

    /// `(qname) → node` lookup (mirrors `Store::get_node_by_qname`).
    fn by_qname(&self, qname: &str) -> Option<&NodeLite> {
        note_edge_resolution_work(1);
        self.by_qname.get(qname)
    }

    /// Record a resolved IMPORTS target for `file` so the reference
    /// resolver can read the file's imports back for disambiguation.
    fn record_import(&mut self, file: &str, target_id: i64) {
        self.imports_by_file
            .entry(file.to_string())
            .or_default()
            .insert(target_id);
    }

    /// The set of node `name`s defined in any of `files`. Used by the
    /// incremental edge re-resolution to find which raw edges could have
    /// resolved into a changed file (and were therefore FK-cascaded). A name
    /// is included if at least one node bearing it lives in a changed file.
    fn names_in_files(
        &self,
        files: &std::collections::HashSet<String>,
    ) -> std::collections::HashSet<String> {
        let mut out = std::collections::HashSet::new();
        for (name, nodes) in &self.by_name {
            if nodes.iter().any(|n| files.contains(&n.file_path)) {
                out.insert(name.clone());
            }
        }
        out
    }

    /// Every node whose `name` equals `name` and whose `label` is in
    /// `labels`. Mirrors `greppy_resolver::defs_named`: the by-name
    /// multimap is the in-memory equivalent of the `idx_nodes_name`
    /// lookup, and the label filter is applied after.
    fn defs_named(&self, labels: &[&str], name: &str) -> Vec<&NodeLite> {
        match self.by_name.get(name) {
            Some(nodes) => {
                note_edge_resolution_work(nodes.len());
                nodes
                    .iter()
                    .filter(|n| labels.contains(&n.label.as_str()))
                    .collect()
            }
            None => {
                note_edge_resolution_work(1);
                Vec::new()
            }
        }
    }

    /// Same-file preference, then project-wide uniqueness. Byte-for-byte
    /// the logic of `greppy_resolver::resolve_unique`, returning the
    /// resolved id only on a unique hit (`None` otherwise — zero or
    /// ambiguous, never guessed). `src_id` identifies the referrer so we
    /// can find its file via `by_qname` indirectly; we pass the referrer's
    /// file directly instead.
    fn resolve_unique(candidates: &[&NodeLite], referrer_file: &str) -> Option<i64> {
        if candidates.is_empty() {
            return None;
        }
        let same_file: Vec<&&NodeLite> = candidates
            .iter()
            .filter(|n| n.file_path == referrer_file)
            .collect();
        if same_file.len() == 1 {
            return Some(same_file[0].id);
        }
        if candidates.len() == 1 {
            return Some(candidates[0].id);
        }
        // Ambiguous → caller decides (import disambiguation) — never guess.
        None
    }

    /// [`resolve_unique`](GraphIndex::resolve_unique) plus import
    /// disambiguation: when project-wide resolution is ambiguous, prefer
    /// the single candidate the referrer's file imports. Mirrors
    /// `greppy_resolver::resolve_unique_with_imports` exactly, including
    /// "zero or several imported → still ambiguous".
    fn resolve_unique_with_imports(
        &self,
        labels: &[&str],
        name: &str,
        referrer_id: i64,
    ) -> Option<i64> {
        self.resolve_unique_status_with_imports(labels, name, referrer_id)
            .unique_id()
    }

    fn resolve_unique_status_with_imports(
        &self,
        labels: &[&str],
        name: &str,
        referrer_id: i64,
    ) -> UniqueResolution {
        note_edge_resolution_work(1);
        let Some(referrer_file) = self.file_of(referrer_id) else {
            return UniqueResolution::Unresolved;
        };
        let candidates = self.defs_named(labels, name);
        if let Some(id) = Self::resolve_unique(&candidates, referrer_file) {
            return UniqueResolution::Unique(id);
        }
        // Only ambiguous results reach here (resolve_unique returned None
        // on either empty *or* ambiguous). An empty candidate set must NOT
        // be "narrowed" by imports, so bail when there are no candidates.
        if candidates.is_empty() {
            return UniqueResolution::Unresolved;
        }
        // Resolution for SAME-FILE ambiguity takes precedence over import
        // disambiguation. An import can resolve to only one compatibility facet
        // (Scala imports the free Function twin, since Method is not importable)
        // even though navigation selects another facet of the same symbol.
        // Resolve the symbol's facets together before consulting that signal.
        //
        // The node model can emit multiple nodes for ONE source symbol — a
        // Function AND a Method twin per Ruby/PHP/Scala method, a Field AND a
        // Variable per Java member — so a reference to that symbol maps to >1
        // candidate that all live in the SAME file. Those candidates are the
        // same source entity. Choose them with the exact label-rank + node-id
        // ordering used by CLI single-symbol navigation; otherwise CALLS can
        // target one twin while `path --to <name>` selects the other and falsely
        // reports no path. Languages with no twins never reach here.
        //
        // Genuinely CROSS-FILE ambiguity — the same name defined in DIFFERENT
        // files, i.e. distinct symbols — is still NOT guessed: picking one
        // would be a real (possibly wrong) edge. We keep the honesty guard
        // there (tests: ambiguous_cross_file_callee_is_not_guessed,
        // no_import_keeps_same_named_cross_file_call_unresolved), leaving the
        // reference unresolved rather than guessing.
        let first_file = &candidates[0].file_path;
        if candidates
            .iter()
            .all(|n| n.file_path.as_str() == first_file.as_str())
        {
            return candidates
                .iter()
                .min_by_key(|node| (navigation_label_rank(&node.label), node.id))
                .map(|node| UniqueResolution::Unique(node.id))
                .unwrap_or(UniqueResolution::Unresolved);
        }

        // Import-disambiguation (preference): if the genuinely cross-file
        // candidates include exactly one target imported by the referrer's
        // file, that is the intended definition.
        if let Some(set) = self.imports_by_file.get(referrer_file) {
            if !set.is_empty() {
                let preferred: Vec<&&NodeLite> =
                    candidates.iter().filter(|n| set.contains(&n.id)).collect();
                if preferred.len() == 1 {
                    return UniqueResolution::Unique(preferred[0].id);
                }
            }
        }
        UniqueResolution::Ambiguous
    }

    /// The file path of the node with id `id`, if known. Resolution needs
    /// the referrer's file for the same-file preference; this is an O(1)
    /// lookup in the `id → file` map built at load time.
    fn file_of(&self, id: i64) -> Option<&str> {
        self.id_to_file.get(&id).map(|s| s.as_str())
    }

    /// Resolve a CALLS edge. Receiver dispatch is deliberately method-only:
    /// resolving `value.as_bytes()` to an unrelated free `as_bytes` function
    /// is worse than leaving the edge unresolved. Other calls retain the
    /// direct-qname, callable-name, then constructable fallback sequence.
    fn resolve_call_target(&self, edge: &ExtractedEdge) -> Option<i64> {
        let src = self.by_qname(&edge.source_qualified_name)?;
        let src_id = src.id;
        let name = edge
            .properties
            .get("callee_name")
            .and_then(|v| v.as_str())?;
        if name.is_empty() {
            return None;
        }
        if edge
            .properties
            .get("callee_form")
            .and_then(|value| value.as_str())
            == Some("receiver")
        {
            let owner = edge
                .properties
                .get("receiver_owner")
                .and_then(|value| value.as_str())?;
            return self.resolve_receiver_method(&edge.file_path, owner, name);
        }
        if let Some(tgt) = self.by_qname(&edge.target_qualified_name) {
            return Some(tgt.id);
        }
        match self.resolve_unique_status_with_imports(&CALLABLE_LABELS, name, src_id) {
            UniqueResolution::Unique(id) => Some(id),
            UniqueResolution::Unresolved => {
                self.resolve_unique_with_imports(&CONSTRUCTABLE_LABELS, name, src_id)
            }
            UniqueResolution::Ambiguous => None,
        }
    }

    /// Resolve a receiver call only when its statically observed owner and
    /// method name identify one Method node. Prefer the exact same-file qname;
    /// cross-file resolution requires a globally unique owner/name suffix.
    fn resolve_receiver_method(&self, file_path: &str, owner: &str, name: &str) -> Option<i64> {
        if owner.is_empty() || name.is_empty() {
            return None;
        }
        let local_qname = format!("{file_path}::{owner}::{name}");
        if let Some(target) = self.by_qname(&local_qname) {
            return (target.label == "Method").then_some(target.id);
        }

        let suffix = format!("::{owner}::{name}");
        note_edge_resolution_work(self.by_qname.len());
        let mut matches = self
            .by_qname
            .iter()
            .filter(|(qname, node)| node.label == "Method" && qname.ends_with(&suffix))
            .map(|(_, node)| node.id);
        let target = matches.next()?;
        matches.next().is_none().then_some(target)
    }

    /// Resolve a reference edge: try the parser's direct same-file guess
    /// qname first, then fall back to a name-based resolve keyed on
    /// `name_prop`. Mirrors the old `resolve_direct_or_name` + the
    /// resolver's `resolve_call` / `resolve_type_ref` (which are
    /// `resolve_unique_with_imports` under the hood).
    fn resolve_direct_or_name(
        &self,
        edge: &ExtractedEdge,
        name_prop: &str,
        labels: &[&str],
    ) -> Option<i64> {
        if let Some(tgt) = self.by_qname(&edge.target_qualified_name) {
            return Some(tgt.id);
        }
        let src = self.by_qname(&edge.source_qualified_name)?;
        let src_id = src.id;
        match edge.properties.get(name_prop).and_then(|v| v.as_str()) {
            Some(name) if !name.is_empty() => {
                self.resolve_unique_with_imports(labels, name, src_id)
            }
            _ => None,
        }
    }

    /// Resolve a filesystem import to the required file's per-file Module
    /// node. Ruby `require_relative` first gets an exact lexical path lookup;
    /// bare `require` falls back to a unique file stem, preserving never-guess
    /// behavior when multiple files share the stem.
    fn resolve_filesystem_module_import(&self, edge: &ExtractedEdge, stem: &str) -> Option<i64> {
        let relative_extension = if edge
            .properties
            .get("ruby_require_relative")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            Some("rb")
        } else if edge
            .properties
            .get("dart_relative_import")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            Some("dart")
        } else {
            None
        };
        if let Some(extension) = relative_extension {
            let path = edge
                .properties
                .get("path")
                .and_then(|value| value.as_str())?;
            if let Some(qname) = relative_module_qname(&edge.file_path, path, extension) {
                if let Some(module) = self.by_qname(&qname) {
                    if module.label == "Module" {
                        return Some(module.id);
                    }
                }
            }
        }

        note_edge_resolution_work(self.by_qname.len());
        let mut matches = self
            .by_qname
            .values()
            .filter(|node| node.label == "Module" && file_stem_matches(&node.file_path, stem))
            .map(|node| node.id);
        let target = matches.next()?;
        matches.next().is_none().then_some(target)
    }

    /// Resolve an imported symbol to a unique definition, using the use-
    /// `path`'s module segment to break a name tie. Mirrors
    /// `greppy_resolver::unique_def_named_with_path` exactly.
    fn unique_def_named_with_path(&self, labels: &[&str], name: &str, path: &str) -> Option<i64> {
        note_edge_resolution_work(1);
        let candidates = self.defs_named(labels, name);
        match candidates.len() {
            0 => return None,
            1 => return Some(candidates[0].id),
            _ => {}
        }
        let module_seg = path_module_segment(path, name)?;
        let matched: Vec<&&NodeLite> = candidates
            .iter()
            .filter(|n| file_stem_matches(&n.file_path, module_seg))
            .collect();
        if matched.len() == 1 {
            Some(matched[0].id)
        } else {
            None
        }
    }
}

/// Build the exact per-file Module qname targeted by Ruby
/// `require_relative`. The path is resolved lexically against the importing
/// file, without touching the filesystem, and an omitted extension means `.rb`.
fn relative_module_qname(
    source_file: &str,
    import_path: &str,
    default_extension: &str,
) -> Option<String> {
    let parent = Path::new(source_file)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let mut candidate = parent.join(import_path);
    if candidate.extension().is_none() {
        candidate.set_extension(default_extension);
    }

    let mut normalized = std::path::PathBuf::new();
    for component in candidate.components() {
        match component {
            std::path::Component::Normal(segment) => normalized.push(segment),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return None;
                }
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => return None,
        }
    }
    let relative = normalized.to_str()?.replace('\\', "/");
    Some(file_module_qname(&relative))
}

/// The qualified name of the per-file `Module` node. Kept byte-identical
/// to the parser's file-level synthetic qname (`<file>::__file__`) so the
/// `IMPORTS` edges the parser emits — whose `source_qualified_name` is
/// exactly this — resolve to a real, persisted node with no parser
/// change. See the Module-node insert in `apply_file_nodes`.
fn file_module_qname(rel_path: &str) -> String {
    format!("{rel_path}::__file__")
}

/// A human-readable name for the per-file `Module` node: the file's base
/// name without extension (`src/foo/bar.rs` → `bar`), falling back to the
/// full relative path when there is no stem.
fn module_name_for(rel_path: &str) -> String {
    Path::new(rel_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(rel_path)
        .to_string()
}

/// The module segment of a Rust use-`path` for a given final `name`: the
/// path segment immediately before `name`. Mirrors
/// `greppy_resolver::path_module_segment` (private to that crate) so the
/// in-memory IMPORTS resolution matches the store-backed path exactly.
/// `b::dup` → `Some("b")`, `crate::b::dup` → `Some("b")`, `dup` → `None`,
/// `self::dup` / `crate::dup` → `None`.
fn path_module_segment<'a>(path: &'a str, name: &str) -> Option<&'a str> {
    let segs: Vec<&str> = path
        .split("::")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let last = segs.last()?;
    if *last != name || segs.len() < 2 {
        return None;
    }
    let module = segs[segs.len() - 2];
    if matches!(module, "crate" | "self" | "super") {
        return None;
    }
    Some(module)
}

/// Whether a node's `file_path` belongs to a module named `module`:
/// `src/b.rs` or `src/b/mod.rs` both match module `b`. Mirrors
/// `greppy_resolver::file_stem_matches`.
fn file_stem_matches(file_path: &str, module: &str) -> bool {
    let p = Path::new(file_path);
    if p.file_name().and_then(|s| s.to_str()) == Some("mod.rs") {
        if let Some(parent) = p
            .parent()
            .and_then(|d| d.file_name())
            .and_then(|s| s.to_str())
        {
            return parent == module;
        }
    }
    p.file_stem().and_then(|s| s.to_str()) == Some(module)
}

/// Build a [`NewNode`] from a parser [`ExtractedNode`], stamping the
/// owning file path. Pure (no store access) so a whole file's nodes can
/// be collected and handed to the batched `Store::insert_nodes` in one
/// transaction (P1 fsync fix).
fn new_node_for(project: &str, rel_path: &str, n: ExtractedNode) -> NewNode {
    NewNode {
        project: project.into(),
        label: n.label,
        name: n.name,
        qualified_name: n.qualified_name,
        file_path: rel_path.into(),
        start_line: n.start_line as i64,
        end_line: n.end_line as i64,
        properties: n.properties,
    }
}

/// Split `bytes` into per-line `ContentRow` values. Non-UTF-8 bytes
/// are filtered out (lossy). Empty lines are dropped (they would
/// only add FTS noise). For non-text files we still emit one row per
/// line — file_content rows are not language-gated; they're a
/// grep-like fallback.
fn content_rows_from_bytes(bytes: &[u8]) -> Vec<ContentRow> {
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    text.lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                None
            } else {
                Some(ContentRow {
                    line: (i as u32) + 1,
                    snippet: trimmed.to_string(),
                })
            }
        })
        .collect()
}

fn tree_sitter_version() -> &'static str {
    "0.25"
}

/// Sentinel sha256 stamped for oversized files whose body we
/// deliberately never read. The freshness check on the
/// hotpath diffs these by `(size, mtime_ns)`, so the content hash is
/// never consulted; the sentinel only marks "this row was recorded
/// without hashing the body". It is intentionally not a valid 64-hex
/// digest so it can never collide with a real content hash.
const OVERSIZE_SENTINEL_SHA: &str = "<oversize>";

/// Resolve the effective max-file-size cap, honouring
/// `GREPPY_MAX_FILE_SIZE` (bytes). Mirrors the resolution used in
/// [`index`] so the supported-file cap, the unsupported-file guard and
/// the freshness hotpath all agree on which files are "oversize".
fn max_file_size_bytes() -> u64 {
    std::env::var("GREPPY_MAX_FILE_SIZE")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(MAX_FILE_SIZE_BYTES)
}

#[derive(Debug, Clone)]
struct IndexControls {
    max_files: Option<usize>,
    time_budget: Option<std::time::Duration>,
    started_at: std::time::Instant,
}

impl IndexControls {
    fn from_env() -> Self {
        Self {
            max_files: parse_positive_usize_env("GREPPY_MAX_FILES"),
            time_budget: parse_duration_ms_env("GREPPY_INDEX_TIME_BUDGET_MS"),
            started_at: std::time::Instant::now(),
        }
    }

    fn time_budget_exhausted(&self) -> bool {
        self.time_budget
            .is_some_and(|budget| self.started_at.elapsed() >= budget)
    }
}

#[derive(Debug, Clone)]
struct ControlledEntries {
    active: Vec<InventoryEntry>,
    skipped: Vec<ControlSkip>,
}

#[derive(Debug, Clone)]
struct ControlSkip {
    entry: InventoryEntry,
    reason: &'static str,
    detail: String,
}

fn parse_positive_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
}

fn parse_duration_ms_env(name: &str) -> Option<std::time::Duration> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(std::time::Duration::from_millis)
}

fn apply_large_repo_controls(
    entries: &[InventoryEntry],
    controls: &IndexControls,
    report: &mut IndexReport,
) -> ControlledEntries {
    if controls.max_files.is_none() && controls.time_budget.is_none() {
        return ControlledEntries {
            active: entries.to_vec(),
            skipped: Vec::new(),
        };
    }

    let mut active = Vec::with_capacity(entries.len());
    let mut skipped = Vec::new();
    let mut time_budget_closed = false;
    for entry in entries {
        if let Some(max_files) = controls.max_files {
            if active.len() >= max_files {
                report.files_skipped_by_file_limit += 1;
                skipped.push(ControlSkip {
                    entry: entry.clone(),
                    reason: "file_limit",
                    detail: format!("GREPPY_MAX_FILES={max_files} limited this index run"),
                });
                continue;
            }
        }

        if time_budget_closed || controls.time_budget_exhausted() {
            time_budget_closed = true;
            let detail = controls
                .time_budget
                .map(|d| format!("GREPPY_INDEX_TIME_BUDGET_MS={} exhausted", d.as_millis()))
                .unwrap_or_else(|| "index time budget exhausted".into());
            report.files_skipped_by_time_budget += 1;
            skipped.push(ControlSkip {
                entry: entry.clone(),
                reason: "time_budget",
                detail,
            });
            continue;
        }

        active.push(entry.clone());
    }
    ControlledEntries { active, skipped }
}

fn record_control_skips(
    store: &mut Store,
    project: &str,
    skipped: &[ControlSkip],
    generation: u64,
) -> Result<()> {
    for skip in skipped {
        drop_indexed_rows_for_skip(store, project, &skip.entry.rel_path)?;
        record_index_skip(
            store,
            project,
            &skip.entry,
            greppy_parser::language_for_path(&skip.entry.abs_path).name(),
            skip.reason,
            &skip.detail,
            generation,
        )?;
    }
    Ok(())
}

fn drop_indexed_rows_for_skip(store: &mut Store, project: &str, rel_path: &str) -> Result<()> {
    let _ = store.delete_nodes_for_file(project, rel_path)?;
    let _ = store.delete_file_content(project, rel_path)?;
    store.delete_file_state(project, rel_path)?;
    delete_raw_edges_for_file(store, project, rel_path)?;
    Ok(())
}

fn record_index_skip(
    store: &mut Store,
    project: &str,
    entry: &InventoryEntry,
    language: &str,
    reason: &str,
    detail: &str,
    generation: u64,
) -> Result<()> {
    let metadata = std::fs::metadata(&entry.abs_path)
        .map(|md| stable_metadata(&md))
        .unwrap_or(StableFileMetadata {
            size: 0,
            mtime_ns: None,
            ctime_ns: None,
            file_id: None,
        });
    store.upsert_index_skip(&IndexSkip {
        project: project.to_string(),
        rel_path: entry.rel_path.clone(),
        language: language.to_string(),
        reason: reason.to_string(),
        detail: detail.to_string(),
        size: metadata.size as i64,
        mtime_ns: metadata.mtime_ns.unwrap_or(0),
        ctime_ns: metadata.ctime_ns,
        file_id: metadata.file_id,
        last_indexed_generation: generation,
        updated_at: ws::now_iso8601(),
    })?;
    Ok(())
}

#[derive(Debug, Clone)]
struct ProviderRunSummary {
    manifest: ProviderManifest,
    files_seen: i64,
    files_indexed: i64,
}

/// Persist the provider-state rows that describe the current active index.
///
/// `files_seen` comes from the current discovered inventory. `files_indexed`
/// comes from persisted `file_state` rows, so unreadable/oversized files are
/// visible as `files_failed` instead of being silently erased from diagnostics.
fn sync_provider_states(
    store: &mut Store,
    project: &str,
    entries: &[InventoryEntry],
    generation: u64,
) -> Result<()> {
    let mut by_language: std::collections::BTreeMap<String, ProviderRunSummary> =
        std::collections::BTreeMap::new();
    for entry in entries {
        let language = greppy_parser::language_for_path(&entry.abs_path);
        let name = language.name().to_string();
        let summary = by_language
            .entry(name)
            .or_insert_with(|| ProviderRunSummary {
                manifest: manifest_for_language(language),
                files_seen: 0,
                files_indexed: 0,
            });
        summary.files_seen += 1;
    }

    for state in store.list_file_states(project)? {
        let language = if state.language.trim().is_empty() {
            greppy_parser::language_for_path(Path::new(&state.rel_path))
                .name()
                .to_string()
        } else {
            state.language
        };
        let Some(summary) = by_language.get_mut(&language) else {
            continue;
        };
        if !matches!(summary.manifest.status, ProviderStatus::Unsupported) {
            summary.files_indexed += 1;
        }
    }

    let updated_at = ws::now_iso8601();
    let states: Vec<ProviderState> = by_language
        .into_values()
        .map(|summary| provider_state_from_summary(project, summary, generation, &updated_at))
        .collect();
    store.replace_provider_states(project, &states)?;
    Ok(())
}

fn provider_state_from_summary(
    project: &str,
    summary: ProviderRunSummary,
    generation: u64,
    updated_at: &str,
) -> ProviderState {
    let manifest = summary.manifest;
    let unsupported_edges: Vec<String> = manifest
        .unsupported_edge_classes
        .iter()
        .map(|class| class.as_str().to_string())
        .collect();
    let supported_edges: Vec<String> = manifest
        .supported_edge_classes
        .iter()
        .map(|class| class.as_str().to_string())
        .collect();
    let files_indexed = summary.files_indexed.min(summary.files_seen).max(0);
    let files_failed = (summary.files_seen - files_indexed).max(0);
    let mut diagnostics = manifest.notes.clone();
    match manifest.status {
        ProviderStatus::Unsupported => {
            diagnostics.push("language detected but provider is unsupported".into());
        }
        ProviderStatus::Partial => {
            diagnostics.push(format!(
                "provider status partial; {} unsupported edge class(es)",
                unsupported_edges.len()
            ));
        }
        ProviderStatus::ParityCandidate => {
            diagnostics.push("provider is a parity candidate, not accepted".into());
        }
        ProviderStatus::Accepted => {}
    }
    if files_failed > 0 {
        diagnostics.push(format!(
            "{files_failed} of {} seen file(s) were not indexed in the latest generation",
            summary.files_seen
        ));
    }

    ProviderState {
        project: project.to_string(),
        language: manifest.language,
        provider_version: manifest.provider_version,
        status: provider_status_str(manifest.status).into(),
        supported_edge_classes: supported_edges,
        unsupported_edge_classes: unsupported_edges,
        files_seen: summary.files_seen,
        files_indexed,
        files_failed,
        diagnostics,
        last_indexed_generation: generation,
        updated_at: updated_at.to_string(),
    }
}

fn provider_status_str(status: ProviderStatus) -> &'static str {
    match status {
        ProviderStatus::Unsupported => "unsupported",
        ProviderStatus::Partial => "partial",
        ProviderStatus::ParityCandidate => "parity_candidate",
        ProviderStatus::Accepted => "accepted",
    }
}

/// Record file_state for an unsupported-language file so the
/// freshness check has a complete view. Errors are swallowed because
/// the indexer must keep going on partial failures.
///
/// An untrusted repo can contain a multi-GB binary.
/// We MUST NOT `fs::read` it here — that would OOM the indexer just as
/// surely as the freshness hotpath. So we stat first: oversized files
/// get a `(size, mtime)`-only `file_state` row with a sentinel hash
/// (no body read), and only within-cap files are hashed. Recording the
/// size/mtime row (rather than skipping entirely) lets the freshness
/// check report the file as `Unchanged` when it has not moved, instead
/// of forcing a reindex on every `greppy grep`.
fn record_unsupported_file_state(
    store: &mut Store,
    project: &str,
    entry: &InventoryEntry,
    generation: u64,
) {
    let Ok(md) = std::fs::metadata(&entry.abs_path) else {
        return;
    };
    if md.len() > max_file_size_bytes() {
        let metadata = stable_metadata(&md);
        // Oversized: record stat only, never read the body.
        let fs = FileState {
            project: project.to_string(),
            rel_path: entry.rel_path.clone(),
            language: greppy_parser::language_for_path(&entry.abs_path)
                .name()
                .to_string(),
            sha256: OVERSIZE_SENTINEL_SHA.to_string(),
            mtime_ns: metadata.mtime_ns.unwrap_or(0),
            size: md.len() as i64,
            parser_version: format!("tree-sitter-{}", tree_sitter_version()),
            extractor_version: "greppy-extractor-v1".into(),
            last_indexed_generation: generation,
        };
        let _ = store.upsert_file_state(&fs);
        let _ = store.upsert_file_identity(
            project,
            &entry.rel_path,
            FileIdentity {
                ctime_ns: metadata.ctime_ns,
                file_id: metadata.file_id,
            },
        );
        return;
    }
    let Ok((bytes, metadata)) = read_stable_file(&entry.abs_path) else {
        return;
    };
    let fs = FileState {
        project: project.to_string(),
        rel_path: entry.rel_path.clone(),
        language: greppy_parser::language_for_path(&entry.abs_path)
            .name()
            .to_string(),
        sha256: file_state::sha256_hex(&bytes),
        mtime_ns: metadata.mtime_ns.unwrap_or(0),
        size: bytes.len() as i64,
        parser_version: format!("tree-sitter-{}", tree_sitter_version()),
        extractor_version: "greppy-extractor-v1".into(),
        last_indexed_generation: generation,
    };
    let _ = store.upsert_file_state(&fs);
    let _ = store.upsert_file_identity(
        project,
        &entry.rel_path,
        FileIdentity {
            ctime_ns: metadata.ctime_ns,
            file_id: metadata.file_id,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const RUST_SAMPLE: &str = r#"
        use std::collections::HashMap;

        pub fn hello() -> String {
            "hi".to_string()
        }

        pub struct Greeter {
            name: String,
        }

        impl Greeter {
            pub fn greet(&self) -> String {
                format!("hi {}", self.name)
            }
        }
    "#;

    const CALLS_SAMPLE: &str = r#"
        fn a() {
            b();
        }
        fn b() {}
    "#;

    const TWO_NEWS: &str = r#"
        struct Foo;
        struct Bar;
        impl Foo {
            fn new() -> Foo { Foo }
        }
        impl Bar {
            fn new() -> Bar { Bar }
        }
    "#;

    fn setup_repo(label: &str, source: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&tmp).unwrap();
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/lib.rs"), source).unwrap();
        fs::write(tmp.join("src/empty.txt"), "").unwrap();
        tmp
    }

    #[test]
    fn ruby_relative_module_qname_normalizes_path_and_extension() {
        assert_eq!(
            relative_module_qname("src/app.rb", "../lib/helper", "rb"),
            Some("lib/helper.rb::__file__".into())
        );
        assert_eq!(
            relative_module_qname("app.rb", "./helper.rb", "rb"),
            Some("helper.rb::__file__".into())
        );
        assert_eq!(relative_module_qname("app.rb", "../helper", "rb"), None);
    }

    #[test]
    fn ruby_require_relative_targets_exact_file_module() {
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-ruby-relative-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("nested")).unwrap();
        fs::write(repo.join("app.rb"), "require_relative './nested/helper'\n").unwrap();
        fs::write(repo.join("helper.rb"), "module RootHelper\nend\n").unwrap();
        fs::write(repo.join("nested/helper.rb"), "module NestedHelper\nend\n").unwrap();

        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").expect("index Ruby require_relative fixture");
        let nested = store
            .list_nodes_by_name("test", "helper", 10)
            .unwrap()
            .into_iter()
            .find(|node| node.label == "Module" && node.file_path == "nested/helper.rb")
            .expect("nested helper Module node");
        let incoming = store
            .incoming_edges(nested.id, Some("IMPORTS"), 10)
            .unwrap();
        assert_eq!(incoming.len(), 1, "exact module import edge: {incoming:?}");
        let source = store
            .get_node(incoming[0].source_id)
            .unwrap()
            .expect("import source Module");
        assert_eq!(source.qualified_name, "app.rb::__file__");

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn dart_relative_import_targets_exact_module_and_public_symbols() {
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-dart-relative-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("lib/nested")).unwrap();
        fs::write(
            repo.join("lib/main.dart"),
            "import 'nested/helper.dart';\nint caller() => do_it() + HELPER_VALUE;\n",
        )
        .unwrap();
        fs::write(
            repo.join("lib/helper.dart"),
            "int do_it() => 1;\n",
        )
        .unwrap();
        fs::write(
            repo.join("lib/nested/helper.dart"),
            "const int HELPER_VALUE = 7;\nint do_it() => 2;\n",
        )
        .unwrap();

        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").expect("index Dart relative import fixture");
        let nested_module = store
            .get_node_by_qname("test", "lib/nested/helper.dart::__file__")
            .unwrap()
            .expect("nested helper Module");
        let nested_function = store
            .get_node_by_qname("test", "lib/nested/helper.dart::Function::do_it")
            .unwrap()
            .expect("nested do_it Function");
        let source = store
            .get_node_by_qname("test", "lib/main.dart::__file__")
            .unwrap()
            .expect("main Module");
        let imports = store.outgoing_edges(source.id, Some("IMPORTS"), 10).unwrap();
        assert!(
            imports.iter().any(|edge| edge.target_id == nested_module.id),
            "relative import must target exact nested Module: {imports:?}"
        );
        assert!(
            imports
                .iter()
                .any(|edge| edge.target_id == nested_function.id),
            "Dart import must expose imported top-level function: {imports:?}"
        );
        let helper_value = store
            .get_node_by_qname("test", "lib/nested/helper.dart::Variable::HELPER_VALUE")
            .unwrap()
            .expect("nested HELPER_VALUE Variable");
        let usages = store
            .incoming_edges(helper_value.id, Some("USAGE"), 10)
            .unwrap();
        assert_eq!(usages.len(), 1, "cross-file constant usage: {usages:?}");
        let usage_source = store
            .get_node(usages[0].source_id)
            .unwrap()
            .expect("usage source");
        assert_eq!(usage_source.name, "caller");

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn certified_reference_kind_can_bypass_usage_label_folding() {
        let preserved = ExtractedEdge {
            edge_type: "TYPE_REF".into(),
            source_qualified_name: "src/main.kt::Function::caller".into(),
            target_qualified_name: "src/types.kt::Class::Payload".into(),
            file_path: "src/main.kt".into(),
            line: 1,
            properties: serde_json::json!({ "preserve_reference_kind": true }),
        };
        assert_eq!(persisted_edge_label(&preserved), "TYPE_REF");

        let mut folded = preserved;
        folded.properties = serde_json::json!({});
        assert_eq!(persisted_edge_label(&folded), "USAGE");
    }

    #[test]
    fn index_small_rust_repo_extracts_known_symbols() {
        let repo = setup_repo("symbols", RUST_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let report = index(&mut store, &repo, "test").expect("indexer run");
        assert!(
            report.files_indexed >= 1,
            "expected ≥1 file indexed: {report:?}"
        );
        assert_eq!(
            report.files_unsupported_language, 1,
            "empty.txt should be unsupported"
        );
        assert_eq!(report.files_unreadable, 0);

        let all = store.list_nodes_by_label("test", "Function", 100).unwrap();
        let names: Vec<&str> = all.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"hello"));
        // `greet` is a Method (its qname is `src/lib.rs::Greeter::greet`).
        // Functions-only is incomplete; check Methods too.
        let methods = store.list_nodes_by_label("test", "Method", 100).unwrap();
        let mnames: Vec<&str> = methods.iter().map(|n| n.name.as_str()).collect();
        assert!(
            mnames.contains(&"greet"),
            "Method greet must exist; got fn={names:?} methods={mnames:?}"
        );
    }

    #[test]
    fn default_index_keeps_source_in_worktree_instead_of_sqlite() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var("GREPPY_CONTENT_FTS").ok();
        // SAFETY: this test serializes the only mutation of this private
        // comparison variable and restores it before returning.
        unsafe {
            std::env::remove_var("GREPPY_CONTENT_FTS");
        }

        let repo = setup_repo("no-content-duplication", RUST_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").expect("indexer run");
        let rows: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM file_content", [], |row| row.get(0))
            .unwrap();

        // SAFETY: serialized by ENV_LOCK and restored before return.
        unsafe {
            match previous {
                Some(value) => std::env::set_var("GREPPY_CONTENT_FTS", value),
                None => std::env::remove_var("GREPPY_CONTENT_FTS"),
            }
        }
        assert_eq!(rows, 0);
    }

    #[test]
    fn index_with_options_honors_discovery_overrides() {
        let repo = setup_repo("discover-overrides", "pub fn keep_me() {}\n");
        fs::write(repo.join("src/generated.rs"), "pub fn drop_me() {}\n").unwrap();
        fs::create_dir_all(repo.join("tests")).unwrap();
        fs::write(
            repo.join("tests/integration.rs"),
            "pub fn outside_scope() {}\n",
        )
        .unwrap();

        let mut store = Store::open_memory().unwrap();
        let options = IndexOptions {
            discover_overrides: greppy_discover::WalkOverrides::empty()
                .include("src/*.rs")
                .exclude("src/generated.rs"),
        };
        let report = index_with_options(&mut store, &repo, "test", &options).unwrap();

        assert_eq!(report.files_considered, 1, "override inventory is scoped");
        assert_eq!(report.files_indexed, 1);
        assert_eq!(report.files_unsupported_language, 0);
        let ws_rows = store.list_workspace_states().unwrap();
        assert_eq!(ws_rows.len(), 1, "exactly one workspace row expected");
        let ws = &ws_rows[0];
        assert!(
            ws.indexer_version
                .contains(";discover_scope=v1;I8:src/*.rs;E16:src/generated.rs"),
            "override scope must be persisted in indexer_version, got {}",
            ws.indexer_version
        );
        assert!(store
            .get_file_state("test", "src/lib.rs")
            .unwrap()
            .is_some());
        assert!(store
            .get_file_state("test", "src/generated.rs")
            .unwrap()
            .is_none());
        assert!(store
            .get_file_state("test", "tests/integration.rs")
            .unwrap()
            .is_none());

        let fns = store.list_nodes_by_label("test", "Function", 100).unwrap();
        let names: Vec<_> = fns.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"keep_me"));
        assert!(!names.contains(&"drop_me"));
        assert!(!names.contains(&"outside_scope"));
    }

    #[test]
    fn index_records_file_state_with_sha256() {
        let repo = setup_repo("fsstate", RUST_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();
        let fs = store.get_file_state("test", "src/lib.rs").unwrap().unwrap();
        assert_eq!(fs.language, "rust");
        assert_eq!(fs.size as usize, RUST_SAMPLE.len());
        assert!(!fs.sha256.is_empty());
        assert_eq!(fs.sha256.len(), 64);
    }

    #[test]
    fn index_records_provider_state_for_diagnostics() {
        let repo = setup_repo("provider-state", RUST_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let rust = store
            .get_provider_state("test", "rust")
            .unwrap()
            .expect("rust provider state must exist");
        assert_eq!(rust.status, "partial");
        assert_eq!(rust.files_seen, 1);
        assert_eq!(rust.files_indexed, 1);
        assert_eq!(rust.files_failed, 0);
        assert!(rust.supported_edge_classes.contains(&"definitions".into()));
        assert!(
            rust.unsupported_edge_classes.contains(&"tests".into()),
            "partial providers must expose missing edge classes: {rust:?}"
        );
        assert!(rust.is_incomplete());

        let txt = store
            .get_provider_state("test", "file extension .txt")
            .unwrap()
            .expect("unsupported txt provider state must exist");
        assert_eq!(txt.status, "unsupported");
        assert_eq!(txt.files_seen, 1);
        assert_eq!(txt.files_indexed, 0);
        assert_eq!(txt.files_failed, 1);
    }

    #[test]
    fn index_bumps_generation_after_run() {
        let repo = setup_repo("gen", RUST_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let r1 = index(&mut store, &repo, "test").unwrap();
        let r2 = index(&mut store, &repo, "test").unwrap();
        assert!(r2.graph_generation > r1.graph_generation);
    }

    #[test]
    fn index_writes_calls_edge_for_caller_callee_pair() {
        // A CALLS edge from `a` to `b` is persisted.
        let repo = setup_repo("calls", CALLS_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();
        let a = store
            .get_node_by_qname("test", "src/lib.rs::Function::a")
            .unwrap()
            .expect("node a must exist");
        let b = store
            .get_node_by_qname("test", "src/lib.rs::Function::b")
            .unwrap()
            .expect("node b must exist");
        let outs: Vec<_> = store
            .outgoing_edges(a.id, None, 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.target_id == b.id && e.edge_type == "CALLS")
            .collect();
        assert_eq!(outs.len(), 1, "expected one CALLS edge a→b, got {outs:?}");
    }

    #[test]
    fn scala_call_targets_same_twin_as_single_symbol_navigation() {
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-scala-path-twin-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/main.scala"),
            "package grid.main\nimport grid.helper.Helper.doIt\nobject MainFlow { def caller(): Int = doIt(2) }\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/helper.scala"),
            "package grid.helper\nobject Helper { def doIt(x: Int): Int = x }\n",
        )
        .unwrap();

        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").expect("index Scala twin fixture");

        let caller = store
            .list_nodes_by_name("test", "caller", 10)
            .unwrap()
            .into_iter()
            .min_by_key(|node| (navigation_label_rank(&node.label), node.id))
            .expect("caller definition");
        let twins: Vec<_> = store
            .list_nodes_by_name("test", "doIt", 10)
            .unwrap()
            .into_iter()
            .filter(|node| matches!(node.label.as_str(), "Function" | "Method"))
            .collect();
        assert_eq!(twins.len(), 2, "Scala method must expose both facets");
        let navigated = twins
            .iter()
            .min_by_key(|node| (navigation_label_rank(&node.label), node.id))
            .expect("navigation target");
        let calls = store.outgoing_edges(caller.id, Some("CALLS"), 10).unwrap();
        assert!(
            calls.iter().any(|edge| edge.target_id == navigated.id),
            "CALLS must target the same twin selected by path navigation: twins={twins:?}, calls={calls:?}"
        );

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn rust_receiver_call_does_not_resolve_to_same_named_free_function() {
        const SOURCE: &str = r#"
fn as_bytes<T>(_value: T) -> usize { 0 }

struct Unrelated;

impl Unrelated {
    fn as_bytes(&self) -> &[u8] { &[] }
}

fn caller(value: &str) -> &[u8] {
    value.as_bytes()
}
"#;
        let repo = setup_repo("receiver-not-free", SOURCE);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .expect("caller must exist");
        let free_function = store
            .get_node_by_qname("test", "src/lib.rs::Function::as_bytes")
            .unwrap()
            .expect("free as_bytes must exist");
        let unrelated_method = store
            .get_node_by_qname("test", "src/lib.rs::Unrelated::as_bytes")
            .unwrap()
            .expect("Unrelated::as_bytes must exist");
        let calls = store.outgoing_edges(caller.id, Some("CALLS"), 256).unwrap();

        assert!(
            calls.iter().all(|edge| {
                edge.target_id != free_function.id && edge.target_id != unrelated_method.id
            }),
            "receiver call must remain unresolved instead of targeting a same-named free function or unrelated method: {calls:?}"
        );
    }

    #[test]
    fn rust_receiver_call_resolves_to_unique_method() {
        const SOURCE: &str = r#"
struct Buffer;

impl Buffer {
    fn as_bytes(&self) -> &[u8] { &[] }
}

fn caller(value: Buffer) -> &'static [u8] {
    value.as_bytes()
}
"#;
        let repo = setup_repo("receiver-method", SOURCE);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .expect("caller must exist");
        let method = store
            .get_node_by_qname("test", "src/lib.rs::Buffer::as_bytes")
            .unwrap()
            .expect("Buffer::as_bytes must exist");
        let calls = store.outgoing_edges(caller.id, Some("CALLS"), 256).unwrap();

        assert!(
            calls.iter().any(|edge| edge.target_id == method.id),
            "receiver call must resolve to the unique Method node: {calls:?}"
        );
    }

    #[test]
    fn class_construction_persists_calls_edge_to_class() {
        const APP_PY: &str = r#"
class RunnerFilter:
    pass

def build():
    return RunnerFilter()
"#;
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-class-call-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/app.py"), APP_PY).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let build = store
            .list_nodes_by_name("test", "build", 100)
            .unwrap()
            .into_iter()
            .find(|n| n.label == "Function")
            .expect("build function must exist");
        let runner_filter = store
            .list_nodes_by_name("test", "RunnerFilter", 100)
            .unwrap()
            .into_iter()
            .find(|n| n.label == "Class")
            .expect("RunnerFilter class must exist");
        let calls: Vec<_> = store
            .outgoing_edges(build.id, Some("CALLS"), 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.target_id == runner_filter.id)
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "expected build() to CALLS RunnerFilter class, got {calls:?}"
        );
    }

    #[test]
    fn ambiguous_callable_does_not_fall_back_to_constructable_class() {
        const APP_PY: &str = r#"
class Widget:
    pass

def build():
    return Widget()
"#;
        const A_PY: &str = r#"
def Widget():
    return 1
"#;
        const B_PY: &str = r#"
def Widget():
    return 2
"#;
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-ambig-callable-class-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/app.py"), APP_PY).unwrap();
        fs::write(repo.join("src/a.py"), A_PY).unwrap();
        fs::write(repo.join("src/b.py"), B_PY).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let build = store
            .list_nodes_by_name("test", "build", 100)
            .unwrap()
            .into_iter()
            .find(|n| n.label == "Function")
            .expect("build function must exist");
        let widget_class = store
            .get_node_by_qname("test", "src/app.py::Class::Widget")
            .unwrap()
            .expect("Widget class must exist");

        let calls: Vec<_> = store.outgoing_edges(build.id, Some("CALLS"), 256).unwrap();
        assert!(
            calls.iter().all(|edge| edge.target_id != widget_class.id),
            "ambiguous callable `Widget` must not guess the class constructor target, got {calls:?}"
        );
    }

    /// Write a repo with two source files: `src/lib.rs` and
    /// `src/helper.rs`. Returns the repo root.
    fn setup_multifile_repo(label: &str, lib_rs: &str, helper_rs: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/lib.rs"), lib_rs).unwrap();
        fs::write(tmp.join("src/helper.rs"), helper_rs).unwrap();
        tmp
    }

    #[test]
    fn cross_file_calls_edge_is_persisted_and_traceable() {
        // The core Track-A capability: `caller` in src/lib.rs calls
        // `do_it` defined in src/helper.rs. Before the two-phase split
        // + final-callee capture + name-based resolver, this produced
        // ZERO edges. It must now produce exactly one cross-file CALLS
        // edge, and the edge must be reachable from the caller node
        // (a one-hop trace).
        const LIB_RS: &str = r#"
            mod helper;
            fn caller() {
                helper::do_it();
            }
        "#;
        const HELPER_RS: &str = r#"
            pub fn do_it() -> u32 { 42 }
        "#;
        let repo = setup_multifile_repo("xfile", LIB_RS, HELPER_RS);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .expect("caller node must exist");
        let target = store
            .get_node_by_qname("test", "src/helper.rs::Function::do_it")
            .unwrap()
            .expect("cross-file target do_it must exist");

        // The CALLS edge is persisted from caller → do_it, crossing
        // the file boundary.
        let outs: Vec<_> = store
            .outgoing_edges(caller.id, None, 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == "CALLS" && e.target_id == target.id)
            .collect();
        assert_eq!(
            outs.len(),
            1,
            "expected one cross-file CALLS edge caller→do_it, got {outs:?}"
        );

        // Trace reaches it: a one-hop walk from `caller` lands on a
        // node in a DIFFERENT file.
        let hop = store.get_node(outs[0].target_id).unwrap().unwrap();
        assert_eq!(hop.file_path, "src/helper.rs");
        assert_eq!(hop.name, "do_it");
        assert_ne!(
            hop.file_path, caller.file_path,
            "trace must cross the file boundary"
        );
    }

    #[test]
    fn cross_file_type_ref_edge_is_persisted() {
        // Track-A TYPE_REF: a function in src/lib.rs takes a parameter
        // whose type `Widget` is a struct defined in src/types.rs. The
        // TYPE_REF edge must resolve cross-file to the Struct node.
        const LIB_RS: &str = r#"
            mod types;
            fn render(w: types::Widget) -> u32 { 0 }
        "#;
        const TYPES_RS: &str = r#"
            pub struct Widget { pub w: u32 }
        "#;
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-xtype-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/lib.rs"), LIB_RS).unwrap();
        fs::write(repo.join("src/types.rs"), TYPES_RS).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let render = store
            .get_node_by_qname("test", "src/lib.rs::Function::render")
            .unwrap()
            .expect("render fn must exist");
        let widget = store
            .get_node_by_qname("test", "src/types.rs::Class::Widget")
            .unwrap()
            .expect("Widget struct must exist cross-file");

        let type_refs: Vec<_> = store
            .outgoing_edges(render.id, Some("USAGE"), 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.target_id == widget.id)
            .collect();
        assert_eq!(
            type_refs.len(),
            1,
            "expected one cross-file TYPE_REF render→Widget, got {type_refs:?}"
        );
        // It genuinely crosses the file boundary.
        let hop = store.get_node(type_refs[0].target_id).unwrap().unwrap();
        assert_ne!(hop.file_path, render.file_path);
    }

    #[test]
    fn cross_file_uses_edge_is_persisted() {
        // Track-A USES: a function in src/lib.rs references the bare
        // identifier `CONSTVALUE`-like symbol `helper_struct` (a Struct
        // defined cross-file). The USES edge must resolve to it.
        //
        // We use a Struct reference (not a call) so the parser classifies
        // it as a USES, not a CALLS. `Marker` is defined in other.rs and
        // mentioned (constructed via path) from lib.rs.
        const LIB_RS: &str = r#"
            mod other;
            fn build() {
                let _m = make(Marker);
            }
            fn make(_x: u8) {}
        "#;
        const OTHER_RS: &str = r#"
            pub struct Marker;
        "#;
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-xuses-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/lib.rs"), LIB_RS).unwrap();
        fs::write(tmp.join("src/other.rs"), OTHER_RS).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &tmp, "test").unwrap();

        let build = store
            .get_node_by_qname("test", "src/lib.rs::Function::build")
            .unwrap()
            .expect("build fn must exist");
        let marker = store
            .get_node_by_qname("test", "src/other.rs::Class::Marker")
            .unwrap()
            .expect("Marker struct must exist cross-file");

        let uses: Vec<_> = store
            .outgoing_edges(build.id, Some("USAGE"), 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.target_id == marker.id)
            .collect();
        assert_eq!(
            uses.len(),
            1,
            "expected one cross-file USES build→Marker, got {uses:?}"
        );
        let hop = store.get_node(uses[0].target_id).unwrap().unwrap();
        assert_ne!(
            hop.file_path, build.file_path,
            "USES must cross the file boundary"
        );
    }

    #[test]
    fn intra_crate_imports_edge_resolves_to_definition() {
        // Track-A IMPORTS: `use other::Thing;` in src/lib.rs must produce
        // an IMPORTS edge from the per-file Module node to the real
        // `Thing` definition in src/other.rs (NOT to the synthetic Import
        // node). Both endpoints must be real, persisted nodes.
        const LIB_RS: &str = r#"
            mod other;
            use other::Thing;
            fn f(_t: Thing) {}
        "#;
        const OTHER_RS: &str = r#"
            pub struct Thing { pub n: u32 }
        "#;
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-ximports-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/lib.rs"), LIB_RS).unwrap();
        fs::write(tmp.join("src/other.rs"), OTHER_RS).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &tmp, "test").unwrap();

        // Source endpoint: the per-file Module node now exists and is real.
        let module = store
            .get_node_by_qname("test", "src/lib.rs::__file__")
            .unwrap()
            .expect("per-file Module node must be persisted for IMPORTS source");
        assert_eq!(module.label, "Module");

        let thing = store
            .get_node_by_qname("test", "src/other.rs::Class::Thing")
            .unwrap()
            .expect("Thing struct must exist cross-file");

        let imports: Vec<_> = store
            .outgoing_edges(module.id, Some("IMPORTS"), 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.target_id == thing.id)
            .collect();
        assert_eq!(
            imports.len(),
            1,
            "expected one IMPORTS module→Thing resolving to the real def, got {imports:?}"
        );

        // The edge must NOT point at a synthetic Import node.
        let tgt = store.get_node(imports[0].target_id).unwrap().unwrap();
        assert_eq!(
            tgt.label, "Class",
            "IMPORTS must resolve to the definition, not an Import node"
        );
    }

    #[test]
    fn same_file_calls_still_resolve_under_two_phase() {
        // Regression guard: the two-phase split must not break the
        // same-file case. `a` calls `b`, both in src/lib.rs.
        let repo = setup_repo("samefile-2phase", CALLS_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();
        let a = store
            .get_node_by_qname("test", "src/lib.rs::Function::a")
            .unwrap()
            .expect("a must exist");
        let b = store
            .get_node_by_qname("test", "src/lib.rs::Function::b")
            .unwrap()
            .expect("b must exist");
        let outs: Vec<_> = store
            .outgoing_edges(a.id, None, 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == "CALLS" && e.target_id == b.id)
            .collect();
        assert_eq!(outs.len(), 1, "same-file CALLS a→b must still resolve");
    }

    #[test]
    fn ambiguous_cross_file_callee_is_not_guessed() {
        // Honesty guard: if two files both define `dup`, a call to
        // `dup()` from a third file must NOT be resolved (the resolver
        // refuses to guess). No CALLS edge from the caller should be
        // created for `dup`.
        const LIB_RS: &str = r#"
            mod a;
            mod b;
            fn caller() { dup(); }
        "#;
        const A_RS: &str = r#"pub fn dup() {}"#;
        const B_RS: &str = r#"pub fn dup() {}"#;
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-ambig-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/lib.rs"), LIB_RS).unwrap();
        fs::write(tmp.join("src/a.rs"), A_RS).unwrap();
        fs::write(tmp.join("src/b.rs"), B_RS).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &tmp, "test").unwrap();
        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .expect("caller must exist");
        let calls: Vec<_> = store
            .outgoing_edges(caller.id, None, 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == "CALLS")
            .collect();
        assert!(
            calls.is_empty(),
            "ambiguous callee `dup` must not be resolved, got {calls:?}"
        );
    }

    #[test]
    fn import_disambiguates_same_named_cross_file_call() {
        // Two files each define `dup`. The caller's file `use`s exactly
        // one of them (`use b::dup;`). The CALLS edge must resolve to the
        // imported `dup` (src/b.rs) — NOT stay unresolved as it would
        // under bare project-wide uniqueness, and NOT pick the other one.
        const LIB_RS: &str = r#"
            mod a;
            mod b;
            use b::dup;
            fn caller() { dup(); }
        "#;
        const A_RS: &str = r#"pub fn dup() -> u32 { 1 }"#;
        const B_RS: &str = r#"pub fn dup() -> u32 { 2 }"#;
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-import-disambig-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/lib.rs"), LIB_RS).unwrap();
        fs::write(tmp.join("src/a.rs"), A_RS).unwrap();
        fs::write(tmp.join("src/b.rs"), B_RS).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &tmp, "test").unwrap();

        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .expect("caller must exist");
        let dup_b = store
            .get_node_by_qname("test", "src/b.rs::Function::dup")
            .unwrap()
            .expect("dup in b.rs must exist");
        let dup_a = store
            .get_node_by_qname("test", "src/a.rs::Function::dup")
            .unwrap()
            .expect("dup in a.rs must exist");

        let calls: Vec<_> = store
            .outgoing_edges(caller.id, Some("CALLS"), 256)
            .unwrap()
            .into_iter()
            .collect();
        // Exactly one CALLS edge, and it points at the imported dup (b.rs).
        assert_eq!(
            calls.len(),
            1,
            "expected exactly one resolved CALLS edge, got {calls:?}"
        );
        assert_eq!(
            calls[0].target_id, dup_b.id,
            "the imported dup (src/b.rs) must win"
        );
        assert_ne!(
            calls[0].target_id, dup_a.id,
            "must not resolve to the non-imported dup (src/a.rs)"
        );
    }

    #[test]
    fn no_import_keeps_same_named_cross_file_call_unresolved() {
        // Same two-`dup` setup but the caller's file imports NEITHER.
        // The call stays unresolved (no CALLS edge) — we never guess.
        const LIB_RS: &str = r#"
            mod a;
            mod b;
            fn caller() { dup(); }
        "#;
        const A_RS: &str = r#"pub fn dup() -> u32 { 1 }"#;
        const B_RS: &str = r#"pub fn dup() -> u32 { 2 }"#;
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-no-import-ambig-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/lib.rs"), LIB_RS).unwrap();
        fs::write(tmp.join("src/a.rs"), A_RS).unwrap();
        fs::write(tmp.join("src/b.rs"), B_RS).unwrap();

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &tmp, "test").unwrap();

        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .expect("caller must exist");
        let calls: Vec<_> = store.outgoing_edges(caller.id, Some("CALLS"), 256).unwrap();
        assert!(
            calls.is_empty(),
            "ambiguous `dup` with no disambiguating import must stay unresolved, got {calls:?}"
        );
    }

    #[test]
    fn reindex_after_symbol_rename_removes_stale_node() {
        // Rename a symbol; re-index; the old node must no longer
        // be in the graph.
        let repo = setup_repo("rename", RUST_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let _r1 = index(&mut store, &repo, "test").unwrap();
        assert!(
            store
                .get_node_by_qname("test", "src/lib.rs::Function::hello")
                .unwrap()
                .is_some(),
            "hello must exist before rename"
        );

        // Rename `hello` → `world`.
        fs::write(
            repo.join("src/lib.rs"),
            r#"pub fn world() -> String { "hi".to_string() }"#,
        )
        .unwrap();
        let _r2 = index(&mut store, &repo, "test").unwrap();
        assert!(
            store
                .get_node_by_qname("test", "src/lib.rs::Function::hello")
                .unwrap()
                .is_none(),
            "stale `hello` node must be deleted on re-index"
        );
        assert!(
            store
                .get_node_by_qname("test", "src/lib.rs::Function::world")
                .unwrap()
                .is_some(),
            "fresh `world` node must exist after re-index"
        );
    }

    /// End-to-end through the real indexer: renaming a
    /// symbol across 6 reindex cycles must keep `nodes_fts` orphan-free so
    /// `search-symbols` (backed by `greppy_store::fts::search_fts`) keeps
    /// returning exactly the live symbol — instead of corrupting the index
    /// ("database disk image is malformed", exit 73) while
    /// `integrity_check` still reports `ok`.
    #[test]
    fn search_symbols_survives_rename_cycles_without_fts_corruption() {
        with_index_control_env_cleared(|| {
            let repo = setup_repo("fts-rename", RUST_SAMPLE);
            let mut store = Store::open_memory().unwrap();

            let mut live = String::new();
            for cycle in 0..6 {
                live = format!("processOrderV{cycle}");
                fs::write(
                    repo.join("src/lib.rs"),
                    format!(r#"pub fn {live}() -> String {{ "hi".to_string() }}"#),
                )
                .unwrap();
                let _ = index(&mut store, &repo, "test").unwrap();

                // After every reindex the integrity check passes AND a symbol
                // search for the live name returns only live nodes (no orphan
                // rowid that has no backing `nodes` row).
                store.integrity_check().expect("integrity must hold");
                let hits = greppy_store::fts::search_fts(&store, &live, 10).unwrap();
                assert!(
                    !hits.is_empty(),
                    "live symbol {live} must be searchable after cycle {cycle}"
                );
                for h in &hits {
                    assert!(
                        store.get_node(h.node_id).unwrap().is_some(),
                        "search-symbols returned orphan rowid {} after cycle {cycle}",
                        h.node_id
                    );
                }
            }

            // The final live function resolves to exactly one Function node,
            // and the old names are gone from the graph.
            let final_node = store
                .get_node_by_qname("test", &format!("src/lib.rs::Function::{live}"))
                .unwrap();
            assert!(final_node.is_some(), "final live function must exist");
            assert!(
                store
                    .get_node_by_qname("test", "src/lib.rs::Function::processOrderV0")
                    .unwrap()
                    .is_none(),
                "the first cycle's symbol must be gone"
            );

            // A prefix MATCH (the form search_fts issues internally) succeeds.
            let prefix = greppy_store::fts::search_fts(&store, "processOrder", 10).unwrap();
            for h in &prefix {
                assert!(
                    store.get_node(h.node_id).unwrap().is_some(),
                    "prefix search must never surface an orphan rowid"
                );
            }
        });
    }

    #[test]
    fn impl_method_qnames_do_not_collide_on_same_file() {
        // Two impls with `fn new` produce two distinct qnames
        // and both nodes are persisted.
        let repo = setup_repo("two-new", TWO_NEWS);
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();
        let foo_new = store
            .get_node_by_qname("test", "src/lib.rs::Foo::new")
            .unwrap();
        let bar_new = store
            .get_node_by_qname("test", "src/lib.rs::Bar::new")
            .unwrap();
        assert!(foo_new.is_some(), "Foo::new must exist");
        assert!(bar_new.is_some(), "Bar::new must exist");
        assert_ne!(
            foo_new.unwrap().id,
            bar_new.unwrap().id,
            "Foo::new and Bar::new must be distinct node ids"
        );
    }

    /// Create a *sparse* file: `metadata().len()` reports `len` but no
    /// real disk/memory is consumed. Used to simulate a multi-GB binary
    /// in an untrusted repo without allocating one — if the guard ever
    /// regressed to read-before-stat, this would slurp `len` bytes.
    fn write_sparse(path: &std::path::Path, len: u64) {
        let f = fs::File::create(path).unwrap();
        f.set_len(len).unwrap();
    }

    #[test]
    fn oversized_unsupported_binary_is_recorded_by_stat_not_read() {
        // An untrusted repo with a huge unsupported binary must
        // not OOM the indexer. The oversized file gets a (size, mtime)
        // -only file_state row with a sentinel hash — its body is never
        // read.
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-oversize-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        // A normal supported file so the run does real work too.
        fs::write(tmp.join("src/lib.rs"), "pub fn ok() {}").unwrap();
        // Unsupported extension, sparse, well above the cap.
        let huge = MAX_FILE_SIZE_BYTES + 8 * 1024 * 1024;
        write_sparse(&tmp.join("blob.bin"), huge);

        let mut store = Store::open_memory().unwrap();
        let report = index(&mut store, &tmp, "test").expect("indexer run must not OOM");
        assert!(report.files_indexed >= 1);

        // The oversized binary has a file_state row recorded by stat.
        let fs_row = store
            .get_file_state("test", "blob.bin")
            .unwrap()
            .expect("oversized unsupported file must still get a stat-only file_state row");
        assert_eq!(
            fs_row.size as u64, huge,
            "recorded size must equal the on-disk (apparent) size"
        );
        assert_eq!(
            fs_row.sha256, OVERSIZE_SENTINEL_SHA,
            "oversized file must carry the sentinel hash (body never read)"
        );
        let skip = store
            .get_index_skip("test", "blob.bin")
            .unwrap()
            .expect("unsupported binary must have skip metadata");
        assert_eq!(skip.reason, "unsupported_language");
        assert_eq!(skip.language, "file extension .bin");
        assert_eq!(skip.size as u64, huge);
    }

    #[test]
    fn oversized_supported_source_is_skipped_not_slurped() {
        // Even a *supported* file (e.g. a multi-GB generated
        // .rs) must be skipped before reading. Sparse .rs above the cap.
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-oversize-rs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        let huge = MAX_FILE_SIZE_BYTES + 1;
        write_sparse(&tmp.join("src/generated.rs"), huge);

        let mut store = Store::open_memory().unwrap();
        let report = index(&mut store, &tmp, "test").expect("indexer run must not OOM");
        assert_eq!(
            report.files_oversize, 1,
            "the oversized .rs must be counted as oversize and skipped: {report:?}"
        );
        assert_eq!(
            report.files_indexed, 0,
            "no oversized file should be indexed"
        );
        // No nodes were extracted (body never parsed).
        assert_eq!(report.nodes_extracted, 0);
        let skip = store
            .get_index_skip("test", "src/generated.rs")
            .unwrap()
            .expect("oversized supported source must have skip metadata");
        assert_eq!(skip.reason, "oversize");
        assert_eq!(skip.language, "rust");
        assert_eq!(skip.size as u64, huge);
    }

    #[test]
    fn max_file_size_env_override_is_honoured() {
        // Unit-test the cap resolver directly so we do not have to
        // mutate the env around a full index() run (which would race
        // other tests' small fixtures in this binary).
        // Use a value far ABOVE the default so that, even if this
        // mutation transiently leaks to a concurrent index() in this
        // binary, no small fixture file is reclassified as oversized.
        let override_val: u64 = MAX_FILE_SIZE_BYTES * 4;
        let prev = std::env::var("GREPPY_MAX_FILE_SIZE").ok();
        // SAFETY: restored immediately below; no diff/index runs here.
        unsafe {
            std::env::set_var("GREPPY_MAX_FILE_SIZE", override_val.to_string());
        }
        let got = max_file_size_bytes();
        unsafe {
            match prev {
                Some(v) => std::env::set_var("GREPPY_MAX_FILE_SIZE", v),
                None => std::env::remove_var("GREPPY_MAX_FILE_SIZE"),
            }
        }
        assert_eq!(
            got, override_val,
            "GREPPY_MAX_FILE_SIZE must override the default cap"
        );
    }

    fn setup_three_rust_files(label: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-large-controls-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        fs::write(tmp.join("src/a.rs"), "pub fn alpha_limit() {}\n").unwrap();
        fs::write(tmp.join("src/b.rs"), "pub fn beta_limit() {}\n").unwrap();
        fs::write(tmp.join("src/c.rs"), "pub fn gamma_limit() {}\n").unwrap();
        tmp
    }

    #[test]
    fn max_files_limit_persists_skips_and_removes_old_graph_rows() {
        let repo = setup_three_rust_files("max-files");
        let mut store = Store::open_memory().unwrap();
        let full = index(&mut store, &repo, "p").unwrap();
        assert_eq!(full.files_indexed, 3, "baseline must index every file");
        assert!(store.count_nodes("p", "", "src/c.rs").unwrap() > 0);

        let limited = with_env_var("GREPPY_MAX_FILES", "1", || {
            index(&mut store, &repo, "p").unwrap()
        });
        assert_eq!(limited.files_considered, 3);
        assert_eq!(limited.files_skipped_by_file_limit, 2);
        let skips = store.list_index_skips("p").unwrap();
        assert_eq!(skips.len(), 2, "two files must carry file_limit metadata");
        assert!(skips.iter().all(|s| s.reason == "file_limit"));
        for skip in &skips {
            assert_eq!(
                store.count_nodes("p", "", &skip.rel_path).unwrap(),
                0,
                "skipped file {} must have no stale graph rows",
                skip.rel_path
            );
        }
    }

    #[test]
    fn zero_ms_time_budget_persists_time_budget_skips() {
        let repo = setup_three_rust_files("time-budget");
        let mut store = Store::open_memory().unwrap();
        let report = with_env_var("GREPPY_INDEX_TIME_BUDGET_MS", "0", || {
            index(&mut store, &repo, "p").unwrap()
        });

        assert_eq!(report.files_considered, 3);
        assert_eq!(report.files_indexed, 0);
        assert_eq!(report.files_skipped_by_time_budget, 3);
        assert_eq!(report.nodes_extracted, 0);
        let skips = store.list_index_skips("p").unwrap();
        assert_eq!(skips.len(), 3);
        assert!(skips.iter().all(|s| s.reason == "time_budget"));
        for skip in &skips {
            assert_eq!(
                store.count_nodes("p", "", &skip.rel_path).unwrap(),
                0,
                "budget-skipped file {} must have no file graph rows",
                skip.rel_path
            );
        }
    }

    /// Build a multi-file repo whose graph exercises both same-file and
    /// cross-file edges plus several files indexed concurrently. Returns
    /// the repo root. The file count is deliberately > worker_count so
    /// the parallel path actually fans out across waves.
    fn setup_many_file_repo(label: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(tmp.join("src")).unwrap();
        // helper.rs defines do_it; lib.rs calls it cross-file; the rest
        // are independent modules with same-file calls and structs so the
        // node + edge set is non-trivial.
        fs::write(
            tmp.join("src/lib.rs"),
            "mod helper;\nfn caller() { helper::do_it(); }\nfn local() { local2(); }\nfn local2() {}\n",
        )
        .unwrap();
        fs::write(tmp.join("src/helper.rs"), "pub fn do_it() -> u32 { 42 }\n").unwrap();
        for i in 0..12 {
            let body = format!(
                "pub struct S{i};\nimpl S{i} {{ pub fn new() -> S{i} {{ S{i} }} pub fn run(&self) {{ self.step(); }} fn step(&self) {{}} }}\npub fn free{i}() {{ free{i}b(); }}\nfn free{i}b() {{}}\n"
            );
            fs::write(tmp.join(format!("src/m{i}.rs")), body).unwrap();
        }
        tmp
    }

    /// A canonical, order-independent snapshot of the whole graph: every
    /// node's (qname,label,file,name,start,end) and every edge as
    /// (src_qname,tgt_qname,type), each set sorted. Two indexer runs that
    /// produce the same graph must produce identical snapshots.
    fn graph_snapshot(store: &mut Store, project: &str) -> (Vec<String>, Vec<String>) {
        let conn = store.conn();
        let mut node_rows: Vec<String> = conn
            .prepare(
                "SELECT qualified_name, label, file_path, name, start_line, end_line \
                 FROM nodes WHERE project = ?1",
            )
            .unwrap()
            .query_map([project], |r| {
                Ok(format!(
                    "{}|{}|{}|{}|{}|{}",
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            })
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        node_rows.sort();

        // Resolve edges to endpoint qnames so the snapshot is independent
        // of autoincrement node ids (which differ by insert order).
        let mut edge_rows: Vec<String> = conn
            .prepare(
                "SELECT s.qualified_name, t.qualified_name, e.edge_type \
                 FROM edges e \
                 JOIN nodes s ON s.id = e.source_id \
                 JOIN nodes t ON t.id = e.target_id \
                 WHERE e.project = ?1",
            )
            .unwrap()
            .query_map([project], |r| {
                Ok(format!(
                    "{}->{}|{}",
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        edge_rows.sort();
        (node_rows, edge_rows)
    }

    #[test]
    fn parallel_and_sequential_indexers_produce_identical_graph() {
        // Determinism contract: indexing the SAME repo with the parallel
        // pool (many workers) and with a forced single worker must yield
        // byte-for-byte the same node set AND edge set. This is the core
        // guarantee that makes parallelising the extract phase safe.
        let repo = setup_many_file_repo("determinism");

        // Run 1: forced sequential (GREPPY_WORKERS=1).
        let seq = {
            let _g = ENV_LOCK.lock().unwrap();
            let prev = std::env::var("GREPPY_WORKERS").ok();
            unsafe { std::env::set_var("GREPPY_WORKERS", "1") };
            let mut store = Store::open_memory().unwrap();
            let report = index(&mut store, &repo, "p").unwrap();
            let snap = graph_snapshot(&mut store, "p");
            unsafe {
                match prev {
                    Some(v) => std::env::set_var("GREPPY_WORKERS", v),
                    None => std::env::remove_var("GREPPY_WORKERS"),
                }
            }
            assert_eq!(
                report.worker_count, 1,
                "forced-sequential run must report 1 worker"
            );
            snap
        };

        // Run 2: forced parallel (GREPPY_WORKERS=8) on a fresh store.
        let par = {
            let _g = ENV_LOCK.lock().unwrap();
            let prev = std::env::var("GREPPY_WORKERS").ok();
            unsafe { std::env::set_var("GREPPY_WORKERS", "8") };
            let mut store = Store::open_memory().unwrap();
            let report = index(&mut store, &repo, "p").unwrap();
            let snap = graph_snapshot(&mut store, "p");
            unsafe {
                match prev {
                    Some(v) => std::env::set_var("GREPPY_WORKERS", v),
                    None => std::env::remove_var("GREPPY_WORKERS"),
                }
            }
            assert_eq!(
                report.worker_count, 8,
                "forced-parallel run must report 8 workers"
            );
            snap
        };

        assert_eq!(
            seq.0, par.0,
            "node sets must be identical across worker counts"
        );
        assert_eq!(
            seq.1, par.1,
            "edge sets must be identical across worker counts"
        );
        // Non-vacuous: the repo really has a meaningful graph.
        assert!(
            seq.0.len() >= 12,
            "expected a substantial node set, got {}",
            seq.0.len()
        );
        assert!(!seq.1.is_empty(), "expected at least one resolved edge");
        // Cross-file edge survived in both: caller -> do_it.
        let xfile = "src/lib.rs::Function::caller->src/helper.rs::Function::do_it|CALLS";
        assert!(
            seq.1.iter().any(|e| e == xfile),
            "cross-file CALLS edge must be present in sequential graph"
        );
        assert!(
            par.1.iter().any(|e| e == xfile),
            "cross-file CALLS edge must be present in parallel graph"
        );
    }

    #[test]
    fn reindex_is_idempotent_under_parallelism() {
        // Re-running the parallel indexer over an unchanged repo must not
        // duplicate nodes or edges (delete-then-insert still holds
        // under the two-phase parallel split).
        let repo = setup_many_file_repo("idempotent");
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("GREPPY_WORKERS").ok();
        unsafe { std::env::set_var("GREPPY_WORKERS", "8") };

        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "p").unwrap();
        let first = graph_snapshot(&mut store, "p");
        let _ = index(&mut store, &repo, "p").unwrap();
        let second = graph_snapshot(&mut store, "p");

        unsafe {
            match prev {
                Some(v) => std::env::set_var("GREPPY_WORKERS", v),
                None => std::env::remove_var("GREPPY_WORKERS"),
            }
        }
        assert_eq!(first.0, second.0, "re-index must not change the node set");
        assert_eq!(first.1, second.1, "re-index must not change the edge set");
    }

    #[test]
    fn worker_count_respects_env_override() {
        // The report's worker_count must follow the GREPPY_WORKERS
        // override (the same budget knob the parallel pool is sized to).
        let repo = setup_repo("workers-env", RUST_SAMPLE);
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("GREPPY_WORKERS").ok();

        unsafe { std::env::set_var("GREPPY_WORKERS", "3") };
        let mut store = Store::open_memory().unwrap();
        let report = index(&mut store, &repo, "p").unwrap();
        assert_eq!(
            report.worker_count, 3,
            "GREPPY_WORKERS=3 must cap the indexer to 3 workers, got {}",
            report.worker_count
        );

        unsafe { std::env::set_var("GREPPY_WORKERS", "1") };
        let mut store2 = Store::open_memory().unwrap();
        let report2 = index(&mut store2, &repo, "p").unwrap();
        assert_eq!(
            report2.worker_count, 1,
            "GREPPY_WORKERS=1 forces sequential"
        );

        unsafe {
            match prev {
                Some(v) => std::env::set_var("GREPPY_WORKERS", v),
                None => std::env::remove_var("GREPPY_WORKERS"),
            }
        }
    }

    #[test]
    fn parallel_extract_preserves_inventory_order_and_unreadable_count() {
        // White-box: parallel_extract must return outcomes in inventory
        // order regardless of which worker finishes first. We assert that
        // each result slot lines up with its source entry's rel_path.
        let repo = setup_many_file_repo("order");
        let entries =
            greppy_discover::walk(&greppy_discover::detect_repo_root(&repo).unwrap()).unwrap();
        let supported: Vec<(usize, &InventoryEntry, Language)> = entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let l = greppy_parser::language_for_path(&e.abs_path);
                l.is_supported().then_some((i, e, l))
            })
            .collect();
        assert!(
            supported.len() > 4,
            "need several files to exercise ordering"
        );

        let (out, _throttled) = parallel_extract(&supported, 8);
        assert_eq!(out.len(), supported.len());
        // Each extracted outcome's rel_path must equal the rel_path of the
        // supported entry at the SAME position (order preserved).
        for (outcome, (_, entry, _)) in out.iter().zip(&supported) {
            match outcome {
                FileOutcome::Extracted { rel_path, .. } => {
                    assert_eq!(
                        rel_path, &entry.rel_path,
                        "parallel_extract must preserve inventory order"
                    );
                }
                FileOutcome::Unreadable { .. } => {
                    panic!("known-good fixture file should not be Unreadable")
                }
            }
        }
    }

    // Serialise env-mutating indexer tests; GREPPY_WORKERS is process
    // global and several tests below toggle it.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env_var<T>(name: &str, value: &str, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var(name).ok();
        // SAFETY: serialized by ENV_LOCK and restored before return.
        unsafe {
            std::env::set_var(name, value);
        }
        let out = f();
        // SAFETY: serialized by ENV_LOCK and restored before return.
        unsafe {
            match prev {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
        out
    }

    fn with_index_control_env_cleared<T>(f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_max_files = std::env::var("GREPPY_MAX_FILES").ok();
        let prev_time_budget = std::env::var("GREPPY_INDEX_TIME_BUDGET_MS").ok();
        // SAFETY: serialized by ENV_LOCK and restored before return.
        unsafe {
            std::env::remove_var("GREPPY_MAX_FILES");
            std::env::remove_var("GREPPY_INDEX_TIME_BUDGET_MS");
        }
        let out = f();
        // SAFETY: serialized by ENV_LOCK and restored before return.
        unsafe {
            match prev_max_files {
                Some(v) => std::env::set_var("GREPPY_MAX_FILES", v),
                None => std::env::remove_var("GREPPY_MAX_FILES"),
            }
            match prev_time_budget {
                Some(v) => std::env::set_var("GREPPY_INDEX_TIME_BUDGET_MS", v),
                None => std::env::remove_var("GREPPY_INDEX_TIME_BUDGET_MS"),
            }
        }
        out
    }

    #[test]
    fn file_state_records_real_generation_stamp() {
        // last_indexed_generation must reflect the run that
        // wrote the row, not 0.
        let repo = setup_repo("gen-stamp", RUST_SAMPLE);
        let mut store = Store::open_memory().unwrap();
        let r1 = index(&mut store, &repo, "test").unwrap();
        let fs1 = store.get_file_state("test", "src/lib.rs").unwrap().unwrap();
        assert!(
            fs1.last_indexed_generation >= 1,
            "first run must record a non-zero generation; got {}",
            fs1.last_indexed_generation
        );

        // Re-index to confirm the stamp advances.
        let _r2 = index(&mut store, &repo, "test").unwrap();
        let fs2 = store.get_file_state("test", "src/lib.rs").unwrap().unwrap();
        assert!(
            fs2.last_indexed_generation > fs1.last_indexed_generation,
            "generation must advance across re-indexes (was {}, now {})",
            fs1.last_indexed_generation,
            fs2.last_indexed_generation
        );
        let _ = r1;
    }

    /// Write a single source file into a fresh repo and return the root.
    fn setup_one_file(label: &str, rel: &str, body: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "greppy-indexer-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let abs = tmp.join(rel);
        fs::create_dir_all(abs.parent().unwrap()).unwrap();
        fs::write(&abs, body).unwrap();
        tmp
    }

    /// The canonical graph snapshot (nodes + edges) of a fresh FULL index
    /// of `repo` into a brand-new store. Used as the reference that the
    /// incremental path must match.
    fn full_reindex_snapshot(repo: &std::path::Path) -> (Vec<String>, Vec<String>) {
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, repo, "p").unwrap();
        graph_snapshot(&mut store, "p")
    }

    #[test]
    fn incremental_matches_full_reindex_across_a_sequence_of_edits() {
        // Hold ENV_LOCK for the whole test (via the env-clearing wrapper):
        // sibling tests mutate GREPPY_MAX_FILES / GREPPY_INDEX_TIME_BUDGET_MS
        // through with_env_var, and any index() run that reads them mid-window is
        // silently truncated (release-flaky: the full-reindex reference snapshot
        // collapsed to a single file under a leaked GREPPY_MAX_FILES=1).
        with_index_control_env_cleared(|| {
            // THE Track-A incremental contract: indexing a repo, then applying
            // a sequence of edits (add file, modify a cross-file callee, delete
            // a file) and re-indexing the SAME store incrementally each time,
            // must produce — at every step — a graph byte-for-byte identical to
            // a from-scratch FULL reindex of the on-disk tree at that step.
            let repo = std::env::temp_dir().join(format!(
                "greppy-indexer-test-incr-eq-full-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(repo.join("src")).unwrap();

            // Step 0 — initial tree: lib.rs calls helper::do_it cross-file.
            fs::write(
                repo.join("src/lib.rs"),
                "mod helper;\nmod util;\nfn caller() { helper::do_it(); }\n",
            )
            .unwrap();
            fs::write(repo.join("src/helper.rs"), "pub fn do_it() -> u32 { 1 }\n").unwrap();
            fs::write(repo.join("src/util.rs"), "pub fn util_fn() {}\n").unwrap();

            let mut store = Store::open_memory().unwrap();
            let r0 = index(&mut store, &repo, "p").unwrap();
            // First run is full: nothing skipped.
            assert_eq!(
                r0.files_skipped, 0,
                "first run must be full, not incremental"
            );
            let incr0 = graph_snapshot(&mut store, "p");
            assert_eq!(incr0, full_reindex_snapshot(&repo), "step 0: incr == full");

            // Step 1 — ADD a new file that the existing caller will (after the
            // next edit) reference; for now it just exists.
            fs::write(repo.join("src/extra.rs"), "pub fn extra() {}\n").unwrap();
            fs::write(
                repo.join("src/lib.rs"),
                "mod helper;\nmod util;\nmod extra;\nfn caller() { helper::do_it(); extra(); }\n",
            )
            .unwrap();
            let r1 = index(&mut store, &repo, "p").unwrap();
            // helper.rs + util.rs are unchanged → skipped; lib.rs modified and
            // extra.rs added are reprocessed.
            assert!(
                r1.files_skipped >= 1,
                "unchanged files must be skipped on the incremental run, got {r1:?}"
            );
            let incr1 = graph_snapshot(&mut store, "p");
            assert_eq!(
                incr1,
                full_reindex_snapshot(&repo),
                "step 1 (add+modify): incr == full"
            );

            // Step 2 — MODIFY a cross-file callee target's file (rename do_it →
            // do_it2) AND update the caller, exercising stale-node removal +
            // cross-file re-resolution from an unchanged-then-changed file.
            fs::write(repo.join("src/helper.rs"), "pub fn do_it2() -> u32 { 2 }\n").unwrap();
            fs::write(
                repo.join("src/lib.rs"),
                "mod helper;\nmod util;\nmod extra;\nfn caller() { helper::do_it2(); extra(); }\n",
            )
            .unwrap();
            let _r2 = index(&mut store, &repo, "p").unwrap();
            let incr2 = graph_snapshot(&mut store, "p");
            assert_eq!(
                incr2,
                full_reindex_snapshot(&repo),
                "step 2 (rename callee): incr == full"
            );

            // Step 3 — DELETE a file (util.rs). Its nodes/edges must vanish.
            fs::remove_file(repo.join("src/util.rs")).unwrap();
            fs::write(
                repo.join("src/lib.rs"),
                "mod helper;\nmod extra;\nfn caller() { helper::do_it2(); extra(); }\n",
            )
            .unwrap();
            let _r3 = index(&mut store, &repo, "p").unwrap();
            let incr3 = graph_snapshot(&mut store, "p");
            assert_eq!(
                incr3,
                full_reindex_snapshot(&repo),
                "step 3 (delete): incr == full"
            );

            // Sanity: the deleted file's node is really gone.
            assert!(
                store
                    .get_node_by_qname("p", "src/util.rs::Function::util_fn")
                    .unwrap()
                    .is_none(),
                "deleted file's node must be removed on incremental reindex"
            );
            // And the cross-file CALLS edge tracks the renamed callee.
            let caller = store
                .get_node_by_qname("p", "src/lib.rs::Function::caller")
                .unwrap()
                .unwrap();
            let do_it2 = store
                .get_node_by_qname("p", "src/helper.rs::Function::do_it2")
                .unwrap()
                .unwrap();
            let calls: Vec<_> = store
                .outgoing_edges(caller.id, Some("CALLS"), 256)
                .unwrap()
                .into_iter()
                .filter(|e| e.target_id == do_it2.id)
                .collect();
            assert_eq!(
                calls.len(),
                1,
                "cross-file CALLS must re-resolve to the renamed callee"
            );
        });
    }

    #[test]
    fn unchanged_reindex_skips_all_files_and_keeps_graph() {
        // Hold ENV_LOCK for the whole test (via the env-clearing wrapper):
        // sibling tests mutate GREPPY_MAX_FILES / GREPPY_INDEX_TIME_BUDGET_MS
        // through with_env_var, and any index() run that reads them mid-window is
        // silently truncated (release-flaky: the full-reindex reference snapshot
        // collapsed to a single file under a leaked GREPPY_MAX_FILES=1).
        with_index_control_env_cleared(|| {
            // Re-indexing an untouched repo must skip every supported file and
            // leave the graph identical (idempotent incremental path).
            let repo = setup_many_file_repo("incr-idempotent");
            let mut store = Store::open_memory().unwrap();
            let r0 = index(&mut store, &repo, "p").unwrap();
            assert_eq!(r0.files_skipped, 0, "first run is full");
            let before = graph_snapshot(&mut store, "p");

            let r1 = index(&mut store, &repo, "p").unwrap();
            // Every supported file is unchanged → skipped; none re-indexed.
            assert!(
                r1.files_skipped >= 13,
                "all unchanged supported files must be skipped, got {}",
                r1.files_skipped
            );
            assert_eq!(
                r1.files_indexed, 0,
                "no file should be re-extracted when nothing changed"
            );
            let after = graph_snapshot(&mut store, "p");
            assert_eq!(before.0, after.0, "node set unchanged on no-op reindex");
            assert_eq!(before.1, after.1, "edge set unchanged on no-op reindex");
        });
    }

    #[test]
    fn incremental_modify_only_reprocesses_the_changed_file() {
        // Hold ENV_LOCK for the whole test (via the env-clearing wrapper):
        // sibling tests mutate GREPPY_MAX_FILES / GREPPY_INDEX_TIME_BUDGET_MS
        // through with_env_var, and any index() run that reads them mid-window is
        // silently truncated (release-flaky: the full-reindex reference snapshot
        // collapsed to a single file under a leaked GREPPY_MAX_FILES=1).
        with_index_control_env_cleared(|| {
            // A single-file edit must re-extract ONLY that file (files_indexed
            // == 1) and skip the rest, while still re-resolving the project.
            let repo = setup_one_file("incr-one", "src/a.rs", "pub fn a() {}\n");
            fs::write(repo.join("src/b.rs"), "pub fn b() {}\n").unwrap();
            let mut store = Store::open_memory().unwrap();
            let _ = index(&mut store, &repo, "p").unwrap();

            // Edit only a.rs.
            fs::write(repo.join("src/a.rs"), "pub fn a() {}\npub fn a2() {}\n").unwrap();
            let r = index(&mut store, &repo, "p").unwrap();
            assert_eq!(
                r.files_indexed, 1,
                "only the edited file is re-extracted, got {r:?}"
            );
            assert!(
                r.files_skipped >= 1,
                "the untouched file must be skipped, got {r:?}"
            );
            // The new symbol is present and matches a full reindex.
            assert!(
                store
                    .get_node_by_qname("p", "src/a.rs::Function::a2")
                    .unwrap()
                    .is_some(),
                "newly-added symbol must be indexed on the incremental path"
            );
            assert_eq!(
                graph_snapshot(&mut store, "p"),
                full_reindex_snapshot(&repo),
                "incremental single-file edit must equal a full reindex"
            );
        });
    }

    #[test]
    fn noop_reindex_reresolves_zero_edges() {
        // Re-review P2: a no-op reindex of a MANY-file repo must not
        // re-resolve the whole project's edges. After the first (full) run,
        // a second run over the untouched tree must feed ZERO raw edges
        // through the resolver — yet leave the graph byte-for-byte identical.
        with_index_control_env_cleared(|| {
            let repo = setup_many_file_repo("noop-zero-reresolve");
            let mut store = Store::open_memory().unwrap();
            let r0 = index(&mut store, &repo, "p").unwrap();
            assert_eq!(r0.files_skipped, 0, "first run is full");
            let before = graph_snapshot(&mut store, "p");
            // The project genuinely has many edges, so "0 re-resolved" is a real
            // saving, not a vacuous one.
            assert!(
                !before.1.is_empty(),
                "fixture must have edges to make the no-op saving meaningful"
            );

            reset_reresolve_counter();
            let r1 = index(&mut store, &repo, "p").unwrap();
            let reresolved = reresolve_count();
            assert_eq!(
                reresolved, 0,
                "a no-op reindex must re-resolve ZERO edges (was O(total edges))"
            );
            // Reported edge count and the graph are unchanged.
            assert_eq!(
                r1.edges_extracted,
                before.1.len(),
                "edges_extracted on a no-op must equal the live edge count"
            );
            let after = graph_snapshot(&mut store, "p");
            assert_eq!(before.0, after.0, "no-op must not change the node set");
            assert_eq!(before.1, after.1, "no-op must not change the edge set");
        });
    }

    #[test]
    fn body_only_edit_takes_cheap_path_and_matches_full() {
        // A pure body edit (the def set — qname/name/label/file — is
        // unchanged) must take the cheap incremental path: it re-resolves
        // FAR fewer than the project's total raw edges, yet produces a graph
        // byte-for-byte identical to a full reindex.
        with_index_control_env_cleared(|| {
            let repo = setup_many_file_repo("body-only-cheap");
            let mut store = Store::open_memory().unwrap();
            let _ = index(&mut store, &repo, "p").unwrap();

            // Count the project's total raw edges (the full path's workload).
            let total_raw = load_all_raw_edges(&store, "p").unwrap().len();
            assert!(total_raw >= 12, "fixture must have many raw edges");

            // Edit ONLY the body of free0() in src/m0.rs — same symbols, same
            // qnames, same labels, same file. `step` body changed too; no symbol
            // added or removed. This keeps the definition fingerprint identical.
            fs::write(
                repo.join("src/m0.rs"),
                "pub struct S0;\nimpl S0 { pub fn new() -> S0 { S0 } pub fn run(&self) { self.step(); self.step(); } fn step(&self) { let _x = 1 + 1; } }\npub fn free0() { free0b(); free0b(); }\nfn free0b() {}\n",
            )
            .unwrap();

            reset_reresolve_counter();
            let r = index(&mut store, &repo, "p").unwrap();
            let reresolved = reresolve_count();
            assert_eq!(r.files_indexed, 1, "only m0.rs re-extracted, got {r:?}");
            // The cheap path was taken: strictly fewer than every raw edge.
            assert!(
                reresolved < total_raw,
                "body-only edit must re-resolve < all {total_raw} raw edges, re-resolved {reresolved}"
            );
            // And it equals a full reindex of the same on-disk tree.
            assert_eq!(
                graph_snapshot(&mut store, "p"),
                full_reindex_snapshot(&repo),
                "body-only incremental edit must equal a full reindex"
            );
        });
    }

    #[test]
    fn cross_file_caller_unchanged_when_callee_body_edited() {
        // The hard case for the cheap path: file A's caller() calls B's
        // do_it() cross-file. We edit ONLY do_it's BODY (not its signature/
        // name). do_it's node is deleted+reinserted with a NEW id, so the
        // cross-file CALLS edge A->do_it is FK-cascaded away. The cheap path
        // must re-resolve it (caller names a changed-file def) and re-point it
        // at the new id — matching a full reindex exactly.
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-xfile-bodyedit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/lib.rs"),
            "mod helper;\nfn caller() { helper::do_it(); }\n",
        )
        .unwrap();
        fs::write(repo.join("src/helper.rs"), "pub fn do_it() -> u32 { 1 }\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "p").unwrap();

        let old_target = store
            .get_node_by_qname("p", "src/helper.rs::Function::do_it")
            .unwrap()
            .unwrap()
            .id;

        // Body-only edit of do_it (same name/qname/label).
        fs::write(repo.join("src/helper.rs"), "pub fn do_it() -> u32 { 2 }\n").unwrap();
        let _ = index(&mut store, &repo, "p").unwrap();

        // The callee was re-extracted → new node id.
        let new_target = store
            .get_node_by_qname("p", "src/helper.rs::Function::do_it")
            .unwrap()
            .unwrap()
            .id;
        assert_ne!(
            old_target, new_target,
            "body edit must re-extract the callee"
        );

        // The cross-file CALLS edge must now point at the NEW id, exactly one,
        // matching a full reindex.
        let caller = store
            .get_node_by_qname("p", "src/lib.rs::Function::caller")
            .unwrap()
            .unwrap();
        let calls: Vec<_> = store
            .outgoing_edges(caller.id, Some("CALLS"), 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.target_id == new_target)
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "cross-file CALLS from an UNCHANGED caller must re-point at the re-extracted callee"
        );
        assert_eq!(
            graph_snapshot(&mut store, "p"),
            full_reindex_snapshot(&repo),
            "callee-body-edit incremental must equal a full reindex"
        );
    }

    #[test]
    fn new_file_introducing_ambiguity_unresolves_unchanged_caller_edge() {
        // The def-fingerprint-changed fallback (the case that exercises the
        // store-backed raw-edge read-back): an UNCHANGED file's resolved edge
        // must be DROPPED when a newly-added file makes its callee ambiguous.
        //
        // Step 0: lib.rs::caller() calls dup(); a.rs defines the only dup() →
        // the cross-file CALLS edge resolves uniquely.
        // Step 1: ADD b.rs that ALSO defines dup(). lib.rs is UNCHANGED, but
        // dup() is now ambiguous project-wide, so its surviving CALLS edge is
        // stale. The def fingerprint changed (a new node appeared), so the
        // incremental path must fall back to the full, from-scratch
        // re-resolution — reading EVERY file's raw edges back from the store's
        // `raw_edges` table (including unchanged lib.rs's) — and produce a
        // graph byte-for-byte identical to a full reindex (edge gone).
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-newambig-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/lib.rs"), "mod a;\nfn caller() { dup(); }\n").unwrap();
        fs::write(repo.join("src/a.rs"), "pub fn dup() {}\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "p").unwrap();

        // The CALLS edge resolves while dup() is unique.
        let caller = store
            .get_node_by_qname("p", "src/lib.rs::Function::caller")
            .unwrap()
            .unwrap();
        let dup_a = store
            .get_node_by_qname("p", "src/a.rs::Function::dup")
            .unwrap()
            .unwrap();
        let calls0: Vec<_> = store
            .outgoing_edges(caller.id, Some("CALLS"), 256)
            .unwrap()
            .into_iter()
            .filter(|e| e.target_id == dup_a.id)
            .collect();
        assert_eq!(
            calls0.len(),
            1,
            "while dup() is unique the cross-file CALLS must resolve"
        );

        // Step 1: ADD a second dup() in a new file. lib.rs is untouched.
        fs::write(
            repo.join("src/lib.rs"),
            "mod a;\nmod b;\nfn caller() { dup(); }\n",
        )
        .unwrap();
        fs::write(repo.join("src/b.rs"), "pub fn dup() {}\n").unwrap();
        let r1 = index(&mut store, &repo, "p").unwrap();

        // lib.rs's `mod b;` line changed it, but a.rs is genuinely unchanged
        // and skipped — yet its (and lib.rs's) edges are re-resolved from the
        // store's raw_edges. The ambiguous CALLS must now be gone.
        let caller = store
            .get_node_by_qname("p", "src/lib.rs::Function::caller")
            .unwrap()
            .unwrap();
        let calls1 = store.outgoing_edges(caller.id, Some("CALLS"), 256).unwrap();
        assert!(
            calls1.is_empty(),
            "ambiguous dup() must leave the caller's CALLS unresolved, got {calls1:?}"
        );

        // And the whole graph equals a from-scratch full reindex.
        assert_eq!(
            graph_snapshot(&mut store, "p"),
            full_reindex_snapshot(&repo),
            "new-ambiguity incremental must equal a full reindex"
        );
        // Sanity: this run really took the incremental path (a.rs skipped).
        assert!(
            r1.files_skipped >= 1,
            "the unchanged a.rs must be skipped on the incremental run, got {r1:?}"
        );
    }

    /// Build a store with `n` files (each a uniquely-named function) and a
    /// cross-file CALLS edge from every file to the previous one, then
    /// return deterministic resolver work units for
    /// `resolve_and_persist_edges` over those edges. The setup (per-node
    /// inserts) is OUTSIDE the measured region; the counter covers the
    /// edge-resolution phase without using wall-clock time, which can flake
    /// under machine oversubscription.
    fn edge_resolution_work(n: usize) -> usize {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&greppy_store::Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        // The store-owned `raw_edges` table is created on open (migration
        // 0007); `resolve_and_persist_edges` does not touch it, so no extra
        // setup is needed here.
        let mut edges: Vec<ExtractedEdge> = Vec::new();
        for i in 0..n {
            let file = format!("src/m{i}.rs");
            let qn = format!("{file}::Function::f{i}");
            store
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: format!("f{i}"),
                    qualified_name: qn.clone(),
                    file_path: file.clone(),
                    start_line: 1,
                    end_line: 2,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            if i > 0 {
                edges.push(ExtractedEdge {
                    edge_type: "CALLS".into(),
                    source_qualified_name: qn,
                    target_qualified_name: format!("src/m{}.rs::Function::__guess__", i - 1),
                    file_path: file,
                    line: 1,
                    properties: serde_json::json!({ "callee_name": format!("f{}", i - 1) }),
                });
            }
        }
        reset_edge_resolution_work_counter();
        let persisted = resolve_and_persist_edges(&mut store, "p", &edges).unwrap();
        let work = edge_resolution_work_count();
        assert_eq!(
            persisted,
            n - 1,
            "every cross-file call must resolve uniquely"
        );
        work
    }

    #[test]
    fn edge_resolution_scales_linearly_not_quadratically() {
        // Scale guard. A per-edge resolver would issue a
        // name-lookup query (and, for ambiguous names, an extra
        // `outgoing_edges` round-trip) PER edge, so doubling the corpus
        // would more than double the resolve time. The in-memory
        // `GraphIndex` loads
        // the project's nodes once and resolves every edge in memory, so
        // the phase is O(nodes + edges).
        //
        // We assert near-linear growth with deterministic work units rather
        // than wall-clock time. The counter tracks node/index visits and
        // resolver candidate checks, so machine load cannot turn the test
        // red while an O(n²) candidate walk would still push the 4x corpus
        // toward ~16x work.
        let base = 1000;
        let w1 = edge_resolution_work(base);
        let w4 = edge_resolution_work(base * 4);
        let ratio = w4 as f64 / w1.max(1) as f64;
        assert!(
            ratio < 5.0,
            "edge resolution must scale ~linearly; 4x input took {ratio:.2}x work \
             (quadratic would be ~16x). w1={w1}, w4={w4}"
        );
    }
}
