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
    workspace_state as ws, ContentRow, FileIdentity, IndexSkip, NewEdge, NewNode, NewOverlayEdge,
    NewRawEdge, Project, ProviderState, RawEdge, Store, WorkspaceState,
};
use rayon::prelude::*;

pub use embedding::{
    count_code_embedding_documents_for_project, count_code_embedding_documents_for_scope,
    count_code_embedding_work_for_scope, count_embedding_candidate_nodes, embedding_path_matches,
    index_code_embeddings_for_project, index_code_embeddings_for_project_with_progress,
    index_code_embeddings_for_scope_with_progress, CodeEmbeddingProvider,
    EmbeddingGemmaCodeProvider, EmbeddingIndexOptions, EmbeddingIndexProgress,
    EmbeddingIndexProgressContext, EmbeddingIndexReport, EmbeddingProviderCacheStats,
    EmbeddingWorkload,
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

/// Observable progress for the non-embedding part of an index build.
///
/// The CLI persists these events in the background-job record so a large
/// repository cannot look stalled while files are actively being parsed or
/// written into the graph. Counts are phase-local: a new phase starts at zero
/// and completes at `total_files`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexBuildProgress {
    pub phase: &'static str,
    pub completed_files: usize,
    pub total_files: usize,
}

impl IndexBuildProgress {
    fn new(phase: &'static str, completed_files: usize, total_files: usize) -> Self {
        Self {
            phase,
            completed_files,
            total_files,
        }
    }
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
    /// Internal Store-CoW extraction boundary. Discovery still walks the
    /// complete repository so policy and freshness stay authoritative, while
    /// extraction and structural contributions are restricted to paths owned
    /// by the private Delta.
    pub only_paths: Option<std::collections::BTreeSet<String>>,
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
    index_with_options_and_progress(store, root, project_name, options, &mut |_| {})
}

/// Run the indexer with explicit discovery options and report real work as it
/// completes. The callback is invoked serially, including when extraction is
/// parallel, so callers do not need synchronization around their progress
/// sink.
pub fn index_with_options_and_progress(
    store: &mut Store,
    root: &Path,
    project_name: &str,
    options: &IndexOptions,
    progress: &mut dyn FnMut(IndexBuildProgress),
) -> Result<IndexReport> {
    progress(IndexBuildProgress::new("discovering_files", 0, 0));
    let abs_root = greppy_discover::detect_repo_root(root)?;
    let indexer_version = indexer_version_for_options(options);
    let prior_state = store.list_private_file_states(project_name)?;
    // Compatibility must be read before publishing the new version or filtering
    // the inventory. Migrate every retained file in this store layer, even when
    // this invocation would ordinarily refresh only a few Delta paths.
    let prior_indexer_version = store
        .list_private_workspace_states()?
        .iter()
        .find(|state| state.root_path == abs_root.to_string_lossy())
        .map(|state| state.indexer_version.clone());
    let rust_reexport_migration = prior_indexer_version
        .as_deref()
        .is_some_and(|prior| is_rust_reexport_migration(prior, &indexer_version));
    let incompatible_index = prior_indexer_version
        .as_deref()
        .map_or(!prior_state.is_empty(), |prior| prior != indexer_version)
        && !rust_reexport_migration;
    let mut only_paths = options.only_paths.clone();
    if rust_reexport_migration {
        if let Some(paths) = only_paths.as_mut() {
            // `compute_file_diff` treats retained rows omitted from the
            // inventory as deleted. Include every retained path for the
            // migration, while the raw-edge refresh below still extracts
            // only unchanged Rust files.
            paths.extend(prior_state.iter().map(|state| state.rel_path.clone()));
        }
    } else if incompatible_index {
        if let Some(paths) = only_paths.as_mut() {
            paths.extend(prior_state.iter().map(|state| state.rel_path.clone()));
        }
    }
    let discovered_entries = greppy_discover::walk_with_policy_and_overrides(
        &abs_root,
        &greppy_discover::SkipPolicy::walk_default(),
        &options.discover_overrides,
    )?;
    let (all_entries, discovery_filtered_entries) = if let Some(only_paths) = only_paths.as_ref() {
        let discovered_paths = discovered_entries
            .iter()
            .map(|entry| entry.rel_path.as_str())
            .collect::<std::collections::HashSet<_>>();
        let filtered = only_paths
            .iter()
            .filter(|rel_path| !discovered_paths.contains(rel_path.as_str()))
            .filter_map(|rel_path| explicit_filtered_inventory_entry(&abs_root, rel_path))
            .collect::<Vec<_>>();
        (
            discovered_entries
                .into_iter()
                .filter(|entry| only_paths.contains(&entry.rel_path))
                .collect::<Vec<_>>(),
            filtered,
        )
    } else {
        (discovered_entries, Vec::new())
    };
    // Store inventory preparation is not file classification. Incompatible
    // generations can spend substantial time here before any file is parsed.
    progress(IndexBuildProgress::new("preparing_graph_inventory", 0, 0));

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
        // Do not advertise v7 until its scoped Rust raw-edge refresh and
        // full re-resolution have both completed. A failed migration must
        // remain retryable on the next invocation.
        indexer_version: if rust_reexport_migration {
            prior_indexer_version
                .clone()
                .unwrap_or_else(|| indexer_version.clone())
        } else {
            indexer_version.clone()
        },
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
    if incompatible_index {
        // Reuse the ordinary per-file cleanup, including files removed or
        // excluded since the old snapshot, before full extraction. This stays
        // inside the unpublished snapshot used by CLI indexing.
        progress(IndexBuildProgress::new(
            "removing_previous_graph",
            0,
            prior_state.len(),
        ));
        for (position, state) in prior_state.iter().enumerate() {
            drop_indexed_rows_for_skip(store, project_name, &state.rel_path)?;
            store.delete_index_skip(project_name, &state.rel_path)?;
            progress(IndexBuildProgress::new(
                "removing_previous_graph",
                position + 1,
                prior_state.len(),
            ));
        }
    }
    let incremental = !incompatible_index && !prior_state.is_empty() && raw_edges_present;

    if rust_reexport_migration && incremental {
        let validated_unchanged = refresh_unchanged_rust_raw_edges(
            store,
            project_name,
            &entries,
            worker_count,
            prior_indexer_version
                .as_deref()
                .is_some_and(|version| version.starts_with("greppy-indexer-v6")),
            &mut report,
            progress,
        )?;
        let _changed_files = run_incremental(
            store,
            project_name,
            &entries,
            only_paths.as_ref(),
            generation,
            worker_count,
            &mut report,
            progress,
        )?;
        // Changed/deleted files lost stale vectors through the normal node
        // rewrite. Every row left here still belongs to a retained node whose
        // bytes are unchanged, so make that cached vector visible in v7.
        for rel_path in validated_unchanged {
            store
                .conn()
                .execute(
                    "UPDATE main.vector_embeddings SET graph_generation = ?3
                     WHERE project = ?1 AND file_path = ?2",
                    rusqlite::params![project_name, rel_path, generation as i64],
                )
                .map_err(sqlite_err)?;
        }
        // Resolver semantics changed. Replace the complete non-structural
        // graph so an edge rejected by v7 cannot survive from v6.
        if !store.is_overlay() {
            store
                .conn()
                .execute(
                    "DELETE FROM main.edges WHERE project = ?1",
                    rusqlite::params![project_name],
                )
                .map_err(sqlite_err)?;
        }
        let raw_edges = load_all_raw_edges(store, project_name)?;
        report.edges_extracted =
            resolve_and_persist_edges_with_progress(store, project_name, &raw_edges, progress)?;
    } else if incremental {
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
            only_paths.as_ref(),
            generation,
            worker_count,
            &mut report,
            progress,
        )?;

        // PHASE B (incremental). Re-resolve only the edges that PHASE A's
        // FK-cascade removed (or whose resolution could have flipped),
        // instead of the whole project's raw edges.
        progress(IndexBuildProgress::new("resolving_edges", 0, 1));
        report.edges_extracted = resolve_edges_incremental(
            store,
            project_name,
            &changed_files,
            &def_fp_before,
            progress,
        )?;
        progress(IndexBuildProgress::new("resolving_edges", 1, 1));
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
            progress,
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
        report.edges_extracted =
            resolve_and_persist_edges_with_progress(store, project_name, &raw_edges, progress)?;
        if profile {
            eprintln!("[profile] resolve_and_persist_edges {:?}", t.elapsed());
        }
    }

    if !store.is_overlay() {
        if incremental
            && (!rust_caller_edges_repaired(store)?
                || !anyhow_factory_edges_repaired(store)?
                || !direct_self_field_edges_repaired(store)?)
        {
            report.edges_extracted += rebuild_single_store_rust_edges(store, project_name)?;
        }
        mark_rust_caller_edges_repaired(store)?;
    }

    // Structural spine (Project / Folder / File nodes + CONTAINS_FILE /
    // CONTAINS_FOLDER / DEFINES edges) — builds the structural pass plus the
    // File→DEFINES edges. Runs AFTER
    // all per-file nodes exist (both paths above have written them) so the
    // File→DEFINES targets are resolvable. Node/edge upserts make it
    // idempotent across incremental re-indexes; a deleted file's File node is
    // removed by the per-file node cascade in `run_incremental`.
    structural::build_structural(
        store,
        project_name,
        &entries,
        &mut |phase, completed, total| {
            progress(IndexBuildProgress::new(phase, completed, total));
        },
    )?;

    // `require`/`import`→File IMPORTS. A path-style import
    // (Ruby `require 'record'`, Clojure `(:require ..)`, Elm/Erlang/Zig/Dart
    // module imports) resolves to the imported FILE node. The edge-resolution
    // pass (above) only targets symbol definitions and runs BEFORE the File
    // nodes exist, so those imports drop. This post-structural pass adds them:
    // for each raw IMPORTS edge whose name does NOT resolve to a symbol, if it
    // maps to exactly one File basename stem, link the importer's Module node
    // to that File. Symbol-resolving imports (rust/python/java — already at
    // parity) are re-checked and skipped, so nothing is double-counted.
    progress(IndexBuildProgress::new("resolving_file_imports", 0, 1));
    resolve_file_imports(store, project_name)?;
    progress(IndexBuildProgress::new("resolving_file_imports", 1, 1));

    record_control_skips(store, project_name, &controlled_entries.skipped, generation)?;
    for entry in &discovery_filtered_entries {
        drop_indexed_rows_for_skip(store, project_name, &entry.rel_path)?;
        record_index_skip(
            store,
            project_name,
            entry,
            greppy_parser::language_for_path(&entry.abs_path).name(),
            "discovery_filtered",
            "tracked Store-CoW Delta path is intentionally excluded by discovery policy",
            generation,
        )?;
        // A discovery-filtered Delta path is still part of the visible
        // workspace. Persist its content identity as well as the diagnostic
        // skip row. The skip row's stat tuple is a fast path, but metadata can
        // legitimately change while Git content remains byte-identical (for
        // example after a checkout, chmod, or backup restore). Without the
        // hash-backed file_state fallback, the next query falsely declared a
        // clean Store-CoW snapshot stale and entered a refresh loop.
        record_unsupported_file_state(store, project_name, entry, generation);
    }

    // R3.5 diagnostics: record the provider completeness state reflected by
    // this index generation. The provider table is store-owned so query-time
    // diagnostics can expose partial language providers without depending on
    // parser internals.
    sync_provider_states(store, project_name, &all_entries, generation)?;

    if rust_reexport_migration {
        let mut migrated_state = store
            .get_workspace_state(abs_root.to_string_lossy().as_ref())?
            .ok_or_else(|| {
                greppy_core::Error::Store("workspace state disappeared during v7 migration".into())
            })?;
        migrated_state.indexer_version = indexer_version;
        store.upsert_workspace_state(&migrated_state)?;
    }

    report.graph_generation = generation;
    if store.is_overlay() {
        recover_visible_effect_fn_bindings(store, project_name, &abs_root)?;
    }
    recover_persisted_js_ts_usages(store, project_name, &abs_root)?;
    progress(IndexBuildProgress::new("finalizing_graph", 1, 1));
    Ok(report)
}

fn explicit_filtered_inventory_entry(root: &Path, rel_path: &str) -> Option<InventoryEntry> {
    let relative = Path::new(rel_path);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return None;
    }
    let abs_path = root.join(relative);
    let metadata = std::fs::symlink_metadata(&abs_path).ok()?;
    // Tracked Delta symlinks are not source files, but still need an identity
    // so unchanged queries remain fresh without following their targets.
    if !metadata.file_type().is_symlink() && !metadata.is_file() {
        return None;
    }
    let stable = stable_metadata(&metadata);
    Some(InventoryEntry {
        rel_path: rel_path.replace('\\', "/"),
        abs_path,
        size: Some(stable.size),
        mtime_ns: stable.mtime_ns,
        ctime_ns: stable.ctime_ns,
        file_id: stable.file_id,
    })
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

fn is_rust_reexport_migration(prior: &str, current: &str) -> bool {
    let (prior_base, prior_scope) = prior.split_once(';').unwrap_or((prior, ""));
    let (current_base, current_scope) = current.split_once(';').unwrap_or((current, ""));
    ((prior_base == "greppy-indexer-v6" && current_base == "greppy-indexer-v7")
        || (matches!(prior_base, "greppy-indexer-v6" | "greppy-indexer-v7")
            && current_base == "greppy-indexer-v8"))
        && prior_scope == current_scope
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
    progress: &mut dyn FnMut(IndexBuildProgress),
) -> Result<()> {
    let max_size = max_file_size_bytes();

    // ── Classification (serial, cheap stat only) ────────────────────
    let mut supported: Vec<(usize, &InventoryEntry, Language)> = Vec::new();
    for (idx, entry) in entries.iter().enumerate() {
        progress(IndexBuildProgress::new(
            "classifying_files",
            idx,
            entries.len(),
        ));
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
    progress(IndexBuildProgress::new(
        "classifying_files",
        entries.len(),
        entries.len(),
    ));

    // ── PHASE A1 — parallel extract (CPU-bound, no store) ───────────
    let profile = std::env::var("GREPPY_PROFILE").is_ok();
    let t_a1 = std::time::Instant::now();
    let mut extraction_progress = |completed, total| {
        progress(IndexBuildProgress::new(
            "extracting_files",
            completed,
            total,
        ));
    };
    let (extractions, throttled) =
        parallel_extract(&supported, worker_count, &mut extraction_progress);
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
    let extraction_count = extractions.len();
    progress(IndexBuildProgress::new(
        "writing_graph",
        0,
        extraction_count,
    ));
    for (position, outcome) in extractions.into_iter().enumerate() {
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
        progress(IndexBuildProgress::new(
            "writing_graph",
            position + 1,
            extraction_count,
        ));
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
    only_paths: Option<&std::collections::BTreeSet<String>>,
    generation: u64,
    worker_count: usize,
    report: &mut IndexReport,
    progress: &mut dyn FnMut(IndexBuildProgress),
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
                // A filtered inventory says nothing about unselected files.
                // In an overlay these may own certified derived relations:
                // treating them as deleted invalidates their retained proof.
                if only_paths.is_some_and(|paths| !paths.contains(rel)) {
                    continue;
                }
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
    let mut extraction_progress = |completed, total| {
        progress(IndexBuildProgress::new(
            "extracting_files",
            completed,
            total,
        ));
    };
    let (extractions, throttled) =
        parallel_extract(&changed, worker_count, &mut extraction_progress);
    report.throttled_for_memory = throttled;
    let extraction_count = extractions.len();
    progress(IndexBuildProgress::new(
        "writing_graph",
        0,
        extraction_count,
    ));
    for (position, outcome) in extractions.into_iter().enumerate() {
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
        progress(IndexBuildProgress::new(
            "writing_graph",
            position + 1,
            extraction_count,
        ));
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

/// Refresh byte-identical files affected by structural migrations: Rust raw
/// edges for v6 upgrades, JS/TS raw edges and proven Effect.fn binding kinds
/// for v8. Unaffected nodes, content and vectors stay put; changed definition
/// identities lose their own vectors so embedding hashes remain authoritative.
fn refresh_unchanged_rust_raw_edges(
    store: &mut Store,
    project_name: &str,
    entries: &[InventoryEntry],
    worker_count: usize,
    refresh_rust: bool,
    report: &mut IndexReport,
    progress: &mut dyn FnMut(IndexBuildProgress),
) -> Result<std::collections::HashSet<String>> {
    let diffs = greppy_freshness::compute_file_diff(store, project_name, entries)?;
    // `compute_file_diff` sorts its result by path, while discovery order is
    // independently deterministic and need not be the same. `Unchanged`
    // intentionally carries no path, so zipping the two sequences can assign
    // another file's classification. Build the explicit changed-path set and
    // classify inventory entries by exclusion instead.
    let changed_paths = diffs
        .iter()
        .filter_map(|diff| match diff {
            greppy_freshness::FileDiff::Added(entry)
            | greppy_freshness::FileDiff::Modified { entry, .. } => Some(entry.rel_path.as_str()),
            greppy_freshness::FileDiff::Deleted(rel) => Some(rel.as_str()),
            greppy_freshness::FileDiff::Unchanged => None,
        })
        .collect::<std::collections::HashSet<_>>();
    let validated_unchanged = entries
        .iter()
        .filter(|entry| !changed_paths.contains(entry.rel_path.as_str()))
        .map(|entry| entry.rel_path.clone())
        .collect::<std::collections::HashSet<_>>();
    let unchanged_rust = entries
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| {
            let language = greppy_parser::language_for_path(&entry.abs_path);
            (!changed_paths.contains(entry.rel_path.as_str())
                && ((refresh_rust && language == Language::Rust)
                    || matches!(language, Language::JavaScript | Language::TypeScript { .. })))
            .then_some((idx, entry, language))
        })
        .collect::<Vec<_>>();
    let mut extraction_progress = |completed, total| {
        progress(IndexBuildProgress::new(
            "extracting_files",
            completed,
            total,
        ));
    };
    let (extractions, throttled) =
        parallel_extract(&unchanged_rust, worker_count, &mut extraction_progress);
    report.throttled_for_memory |= throttled;
    for outcome in extractions {
        match outcome {
            FileOutcome::Extracted {
                rel_path,
                nodes,
                edges,
                ..
            } => {
                // v8 changes only proven Effect.fn Variables into Functions.
                // Keep unaffected node IDs/content/vectors; retire embeddings
                // for the changed identity rather than reuse an unverified hash.
                for node in nodes.iter().filter(|node| node.label == "Function") {
                    let old_qname = node.qualified_name.replace("::Function::", "::Variable::");
                    if let Some(old) = store.get_node_by_qname(project_name, &old_qname)? {
                        if old.label == "Variable"
                            && old.name == node.name
                            && old.start_line == i64::from(node.start_line)
                            && old.end_line == i64::from(node.end_line)
                        {
                            store.update_node_identity(
                                old.id,
                                &node.label,
                                &node.qualified_name,
                            )?;
                        }
                    }
                }
                persist_raw_edges_for_file(store, project_name, &rel_path, &edges)?;
                report.files_indexed += 1;
            }
            FileOutcome::Unreadable { entry, detail, .. } => {
                report.files_unreadable += 1;
                return Err(greppy_core::Error::Invalid(format!(
                    "v7 Rust migration could not re-extract {}: {detail}",
                    entry.rel_path
                )));
            }
        }
    }
    Ok(validated_unchanged)
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
            let (extraction, dropped, contract_error) =
                validate_or_degrade(lang, &entry.rel_path, extraction);
            // A C-parsed `.h` whose output needed record drops (or failed
            // outright) is the signature of C++ syntax in a C header —
            // ClickHouse and Poco write C++ under `.h` throughout. Re-extract
            // with the C++ grammar and keep whichever result carries more of
            // the file.
            let c_header_needs_cpp = lang == Language::C
                && entry.rel_path.ends_with(".h")
                && (dropped > 0 || contract_error.is_some());
            if c_header_needs_cpp {
                if let Ok(retry) = parser_extract(Language::Cpp, &bytes, &entry.rel_path) {
                    let (cpp_extraction, _, cpp_error) =
                        validate_or_degrade(Language::Cpp, &entry.rel_path, retry);
                    if cpp_error.is_none() {
                        let cpp_wins = match &contract_error {
                            Some(_) => true,
                            None => cpp_extraction.nodes.len() >= extraction.nodes.len(),
                        };
                        if cpp_wins {
                            return FileOutcome::Extracted {
                                rel_path: entry.rel_path.clone(),
                                abs_path: entry.abs_path.clone(),
                                bytes,
                                metadata,
                                nodes: cpp_extraction.nodes,
                                edges: cpp_extraction.edges,
                            };
                        }
                    }
                }
            }
            if let Some(contract_error) = contract_error {
                // Name the violated rule: "failed contract validation" alone
                // sent a whole ClickHouse triage hunting through 47 headers
                // for a cause the error already knew.
                return FileOutcome::Unreadable {
                    entry: entry.clone(),
                    language: lang,
                    reason: "provider_invalid",
                    detail: format!("provider output failed contract validation: {contract_error}"),
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

/// Validate the extraction under the provider contract; when validation fails,
/// degrade by dropping the records that violate it instead of refusing the
/// whole file. One anonymous node produced by grammar error-recovery must not
/// discard the hundreds of real definitions around it — 47 template-heavy
/// ClickHouse headers went dark exactly this way. The contract itself is
/// unchanged: the filtered output must still validate, and a file whose
/// failure filtering cannot cure is refused as before.
///
/// Returns the (possibly filtered) extraction, how many records were dropped,
/// and the original contract error when even the filtered output is invalid.
fn validate_or_degrade(
    lang: Language,
    rel_path: &str,
    extraction: greppy_parser::ExtractionResult,
) -> (
    greppy_parser::ExtractionResult,
    usize,
    Option<greppy_parser::ProviderContractError>,
) {
    let provider_output = ProviderOutput::from_extraction(
        manifest_for_language(lang),
        lang,
        rel_path,
        extraction.clone(),
    );
    let Err(contract_error) = provider_output.validate() else {
        return (extraction, 0, None);
    };
    let mut filtered = extraction;
    let before = filtered.nodes.len() + filtered.edges.len();
    filtered.nodes.retain(|node| {
        node.start_line >= 1
            && node.end_line >= node.start_line
            && !node.name.trim().is_empty()
            && !node.qualified_name.trim().is_empty()
    });
    filtered.edges.retain(|edge| {
        edge.line >= 1
            && !edge.edge_type.trim().is_empty()
            && !edge.source_qualified_name.trim().is_empty()
            && !edge.target_qualified_name.trim().is_empty()
    });
    let dropped = before - (filtered.nodes.len() + filtered.edges.len());
    let recheck = ProviderOutput::from_extraction(
        manifest_for_language(lang),
        lang,
        rel_path,
        filtered.clone(),
    );
    if recheck.validate().is_err() {
        return (filtered, dropped, Some(contract_error));
    }
    (filtered, dropped, None)
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
    progress: &mut dyn FnMut(usize, usize),
) -> (Vec<FileOutcome>, bool) {
    let n = supported.len();
    progress(0, n);
    if n == 0 {
        return (Vec::new(), false);
    }

    // Single core / single worker → just go sequential; the parallel
    // machinery would only add overhead and the result is identical.
    if worker_count <= 1 || n == 1 {
        let out = supported
            .iter()
            .enumerate()
            .map(|(position, (_, entry, lang))| {
                let outcome = extract_one(entry, *lang);
                progress(position + 1, n);
                outcome
            })
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
                .enumerate()
                .map(|(position, (_, entry, lang))| {
                    let outcome = extract_one(entry, *lang);
                    progress(position + 1, n);
                    outcome
                })
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
            for (offset, (slot, (_, entry, lang))) in
                slots[pos..].iter_mut().zip(&supported[pos..]).enumerate()
            {
                *slot = Some(extract_one(entry, *lang));
                progress(pos + offset + 1, n);
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
        progress(pos, n);
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
    resolve_and_persist_edges_with_progress_and_preserved(store, project, edges, &mut |_| {}, &[])
}

fn resolve_and_persist_edges_with_progress(
    store: &mut Store,
    project: &str,
    edges: &[ExtractedEdge],
    progress: &mut dyn FnMut(IndexBuildProgress),
) -> Result<usize> {
    resolve_and_persist_edges_with_progress_and_preserved(store, project, edges, progress, &[])
}

fn resolve_and_persist_edges_with_progress_and_preserved(
    store: &mut Store,
    project: &str,
    edges: &[ExtractedEdge],
    progress: &mut dyn FnMut(IndexBuildProgress),
    preserved_overlay_edges: &[NewOverlayEdge],
) -> Result<usize> {
    resolve_edges_with_replacement(
        store,
        project,
        edges,
        progress,
        preserved_overlay_edges,
        false,
    )
}

fn resolve_edges_with_replacement(
    store: &mut Store,
    project: &str,
    edges: &[ExtractedEdge],
    progress: &mut dyn FnMut(IndexBuildProgress),
    preserved_overlay_edges: &[NewOverlayEdge],
    replace_single_rust_edges: bool,
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
    let mut examined = 0usize;
    progress(IndexBuildProgress::new("resolving_edges", 0, edges.len()));

    // PASS 1 — IMPORTS. Record each resolved target into the index so the
    // reference pass can read a file's imports back.
    for edge in edges.iter().filter(|e| e.edge_type == "IMPORTS") {
        examined += 1;
        progress(IndexBuildProgress::new(
            "resolving_edges",
            examined,
            edges.len(),
        ));
        let Some(src) = index.by_qname(&edge.source_qualified_name) else {
            continue;
        };
        let src_id = src.id;
        let src_file = src.file_path.clone();
        index.record_import_items(edge, &src_file);
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
                index.resolve_import_target(
                    &src_file,
                    name,
                    path,
                    edge.properties.get("imported_items"),
                )
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

    // A Store-CoW Delta resolves only its own raw edges, but a qualified Rust
    // reference can depend on reexports in the immutable Base. Hydrate import
    // context only for module files reached from the Delta's imports. This
    // preserves the O(Delta + reached modules) bound instead of scanning every
    // Base raw edge, while allowing `channels::target` to follow a Base
    // `pub use command::{..., target}`.
    if store.is_overlay() {
        let mut pending = index
            .rust_namespaces_by_file
            .values()
            .flat_map(|aliases| aliases.values().flatten().cloned())
            .collect::<Vec<_>>();
        pending.extend(
            index
                .import_module_files_by_file
                .values()
                .flat_map(|files| files.iter().cloned()),
        );
        for (source_file, globs) in &index.import_globs_by_file {
            for glob in globs {
                pending.extend(
                    index
                        .rust_module_files_for_module_path(source_file, glob)
                        .into_iter()
                        .filter(|file| index.known_files.contains(file)),
                );
            }
        }
        let mut seen = std::collections::HashSet::new();
        while let Some(module_file) = pending.pop() {
            if !seen.insert(module_file.clone()) {
                continue;
            }
            for raw in store.list_raw_import_edges_for_file(project, &module_file)? {
                let edge = extracted_edge_from_raw(raw);
                if edge.edge_type != "IMPORTS" {
                    continue;
                }
                let Some(src) = index.by_qname(&edge.source_qualified_name) else {
                    continue;
                };
                let src_id = src.id;
                let src_file = src.file_path.clone();
                index.record_import_items(&edge, &src_file);
                let Some(name) = edge
                    .properties
                    .get("imported_name")
                    .and_then(|value| value.as_str())
                    .filter(|name| !name.is_empty())
                else {
                    continue;
                };
                let path = edge
                    .properties
                    .get("path")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                if let Some(target_id) = index
                    .resolve_import_target(
                        &src_file,
                        name,
                        path,
                        edge.properties.get("imported_items"),
                    )
                    .filter(|target_id| *target_id != src_id)
                {
                    index.record_import(&src_file, target_id);
                }
            }
            pending.extend(
                index
                    .rust_namespaces_by_file
                    .get(&module_file)
                    .into_iter()
                    .flat_map(|aliases| aliases.values().flatten().cloned()),
            );
            pending.extend(
                index
                    .import_module_files_by_file
                    .get(&module_file)
                    .into_iter()
                    .flat_map(|files| files.iter().cloned()),
            );
            if let Some(globs) = index.import_globs_by_file.get(&module_file) {
                for glob in globs {
                    pending.extend(
                        index
                            .rust_module_files_for_module_path(&module_file, glob)
                            .into_iter()
                            .filter(|file| index.known_files.contains(file)),
                    );
                }
            }
        }
    }

    // PASS 2 — reference edges (CALLS / TYPE_REF / USES / other).
    for edge in edges.iter().filter(|e| e.edge_type != "IMPORTS") {
        examined += 1;
        progress(IndexBuildProgress::new(
            "resolving_edges",
            examined,
            edges.len(),
        ));
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
            "USAGE" => index.resolve_usage_target(edge, src_id),
            _ => index.by_qname(&edge.target_qualified_name).map(|n| n.id),
        };

        let Some(target_id) = target_id else {
            clear_option_field_unresolved();
            continue;
        };

        // An edge must connect two DISTINCT nodes; a self-loop here is
        // almost always a same-file guess qname accidentally matching the
        // source (e.g. a USES of the enclosing symbol's own name). CALLS
        // keeps self-loops (direct recursion is legitimate); the others
        // drop them.
        if target_id == src_id && edge.edge_type != "CALLS" {
            clear_option_field_unresolved();
            continue;
        }
        resolved.push(new_edge(project, src_id, target_id, edge));
        push_rust_enum_owner_usage(&index, project, src_id, target_id, edge, &mut resolved);
    }

    // Persist every resolved edge in a SINGLE transaction (was: one
    // transaction per edge). Determinism is unchanged — the edge order is
    // the same IMPORTS-then-references order resolved above.
    progress(IndexBuildProgress::new("writing_resolved_edges", 0, 1));
    if rust_anyhow_context(store, Some(project))?.1 != index.anyhow_dependency_binding {
        return Err(greppy_core::Error::Invalid(
            "Cargo dependency identity changed during caller resolution".into(),
        ));
    }
    if store.is_overlay() {
        let resolved_logical = resolved
            .iter()
            .map(|edge| {
                let source_qualified_name =
                    index.qname_for_id(edge.source_id).ok_or_else(|| {
                        greppy_core::Error::Store(format!(
                            "overlay edge source id {} has no logical identity",
                            edge.source_id
                        ))
                    })?;
                let target_qualified_name =
                    index.qname_for_id(edge.target_id).ok_or_else(|| {
                        greppy_core::Error::Store(format!(
                            "overlay edge target id {} has no logical identity",
                            edge.target_id
                        ))
                    })?;
                Ok(NewOverlayEdge {
                    project: edge.project.clone(),
                    source_qualified_name: source_qualified_name.to_owned(),
                    target_qualified_name: target_qualified_name.to_owned(),
                    edge_type: edge.edge_type.clone(),
                    properties: edge.properties.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut logical = preserved_overlay_edges.to_vec();
        logical.extend(resolved_logical);
        store.replace_overlay_edges(project, &logical)?;
    } else {
        persist_edges_batched(
            store,
            &resolved,
            replace_single_rust_edges.then_some(project),
        )?;
    }
    store.conn().execute("INSERT INTO main.schema_meta(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", rusqlite::params![anyhow_factory_repair_key(project),index.anyhow_dependency_binding]).map_err(sqlite_err)?;
    store.conn().execute("INSERT INTO main.schema_meta(key,value) VALUES(?1,'complete') ON CONFLICT(key) DO UPDATE SET value=excluded.value", [format!("{DIRECT_SELF_FIELD_REPAIR_KEY}.{project}")]).map_err(sqlite_err)?;
    progress(IndexBuildProgress::new("writing_resolved_edges", 1, 1));
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
    let rows = if store.is_overlay() {
        store.list_delta_raw_edges(project)?
    } else {
        store.list_raw_edges(project)?
    };
    Ok(rows.into_iter().map(extracted_edge_from_raw).collect())
}

pub const RUST_CALLER_EDGES_REPAIR_META_KEY: &str = "greppy.rust_caller_edges_repair.v15";
const ANYHOW_FACTORY_REPAIR_KEY: &str = "greppy.rust_anyhow_factory_repair.v1";

/// Bind known anyhow semantics to authored Cargo dependency identity, never to
/// an arbitrary external Result spelling. The returned digest also invalidates
/// completed caller repair when manifests or lock identity change.
fn cargo_item_mentions_anyhow(item: &toml_edit::Item) -> bool {
    item.as_table_like().is_some_and(|table| {
        table.iter().any(|(key, value)| {
            key == "anyhow"
                || key.starts_with("anyhow:")
                || (key == "package" && value.as_str() == Some("anyhow"))
                || cargo_item_mentions_anyhow(value)
        })
    })
}

fn rust_anyhow_context(
    store: &Store,
    only_project: Option<&str>,
) -> Result<(std::collections::HashSet<(String, String)>, String)> {
    let projects = {
        let mut stmt = store
            .conn()
            .prepare("SELECT name,root_path FROM projects")
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(sqlite_err)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sqlite_err)?
    };
    let mut documents = std::collections::BTreeMap::<String, String>::new();
    let mut allowed = std::collections::HashSet::new();
    for (project, root) in projects
        .into_iter()
        .filter(|(project, _)| only_project.is_none_or(|selected| selected == project.as_str()))
    {
        let root = Path::new(&root);
        let root_manifest_path = root.join("Cargo.toml");
        let root_text = std::fs::read_to_string(&root_manifest_path).unwrap_or_default();
        documents.insert(
            root_manifest_path.to_string_lossy().into_owned(),
            root_text.clone(),
        );
        let root_doc = root_text.parse::<toml_edit::DocumentMut>().ok();
        let lock_path = root.join("Cargo.lock");
        let lock_text = std::fs::read_to_string(&lock_path).unwrap_or_default();
        documents.insert(lock_path.to_string_lossy().into_owned(), lock_text.clone());
        let lock_doc = lock_text.parse::<toml_edit::DocumentMut>().ok();
        let identity = lock_doc
            .as_ref()
            .and_then(|doc| doc.get("package"))
            .and_then(|item| item.as_array_of_tables())
            .map(|packages| {
                let packages = packages
                    .iter()
                    .filter(|package| {
                        package.get("name").and_then(|item| item.as_str()) == Some("anyhow")
                    })
                    .collect::<Vec<_>>();
                packages.len() == 1
                    && packages[0]
                        .get("version")
                        .and_then(|item| item.as_str())
                        .is_some_and(|version| {
                            let parts = version.split('.').collect::<Vec<_>>();
                            parts.len() == 3
                                && parts[0] == "1"
                                && parts.iter().all(|part| part.parse::<u64>().is_ok())
                        })
                    && packages[0].get("source").and_then(|item| item.as_str())
                        == Some("registry+https://github.com/rust-lang/crates.io-index")
                    && packages[0]
                        .get("checksum")
                        .and_then(|item| item.as_str())
                        .is_some_and(|hash| {
                            hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                        })
            })
            .unwrap_or(false);
        let patched = root_doc.as_ref().is_some_and(|doc| {
            doc.get("patch").is_some_and(cargo_item_mentions_anyhow)
                || doc.get("replace").is_some_and(cargo_item_mentions_anyhow)
        });
        let mut manifest_decisions = std::collections::HashMap::new();
        let mut nearest_manifests = std::collections::HashMap::new();
        for state in store
            .list_file_states(&project)?
            .into_iter()
            .filter(|state| state.rel_path.ends_with(".rs"))
        {
            let relative = Path::new(&state.rel_path);
            if relative
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                continue;
            }
            let file_path = root.join(relative);
            let parent_directory = file_path.parent().unwrap_or(root).to_path_buf();
            let mut directory = file_path.parent();
            let mut selected = nearest_manifests.get(&parent_directory).cloned().flatten();
            let cached_directory = nearest_manifests.contains_key(&parent_directory);
            while let Some(dir) = directory.filter(|dir| !cached_directory && dir.starts_with(root))
            {
                let manifest = dir.join("Cargo.toml");
                if manifest.is_file() {
                    selected = Some(manifest);
                    break;
                }
                directory = dir.parent();
            }
            nearest_manifests.insert(parent_directory, selected.clone());
            let Some(manifest) = selected else {
                continue;
            };
            if let Some(proven) = manifest_decisions.get(&manifest).copied() {
                if proven {
                    allowed.insert((project.clone(), state.rel_path));
                }
                continue;
            }
            // Record a conservative failed decision before fallible checks.
            manifest_decisions.insert(manifest.clone(), false);
            let text = documents
                .entry(manifest.to_string_lossy().into_owned())
                .or_insert_with(|| std::fs::read_to_string(&manifest).unwrap_or_default());
            let Ok(doc) = text.parse::<toml_edit::DocumentMut>() else {
                continue;
            };
            if !identity
                || patched
                || doc.get("package").is_none()
                || doc
                    .get("package")
                    .and_then(|item| item.as_table_like())
                    .and_then(|table| table.get("name"))
                    .and_then(|item| item.as_str())
                    == Some("anyhow")
                || doc.get("patch").is_some_and(cargo_item_mentions_anyhow)
                || doc.get("replace").is_some_and(cargo_item_mentions_anyhow)
                || doc.get("target").is_some_and(cargo_item_mentions_anyhow)
            {
                continue;
            }
            let mut dependency = doc
                .get("dependencies")
                .and_then(|item| item.as_table_like())
                .and_then(|table| table.get("anyhow"));
            if dependency
                .and_then(|item| item.as_table_like())
                .and_then(|table| table.get("optional"))
                .and_then(|item| item.as_bool())
                == Some(true)
            {
                continue;
            }
            if dependency
                .and_then(|item| item.as_table_like())
                .and_then(|table| table.get("package"))
                .and_then(|item| item.as_str())
                .is_some_and(|name| name != "anyhow")
            {
                continue;
            }
            if dependency
                .and_then(|item| item.as_table_like())
                .and_then(|table| table.get("workspace"))
                .and_then(|item| item.as_bool())
                == Some(true)
            {
                dependency = root_doc
                    .as_ref()
                    .and_then(|root| root.get("workspace"))
                    .and_then(|item| item.as_table_like())
                    .and_then(|table| table.get("dependencies"))
                    .and_then(|item| item.as_table_like())
                    .and_then(|table| table.get("anyhow"));
            }
            let proven = dependency.is_some_and(|item| {
                if let Some(version) = item.as_str() {
                    return matches!(version, "1" | "^1");
                }
                let Some(table) = item.as_table_like() else {
                    return false;
                };
                table
                    .get("version")
                    .and_then(|item| item.as_str())
                    .is_some_and(|version| matches!(version, "1" | "^1"))
                    && table.get("path").is_none()
                    && table.get("git").is_none()
                    && table.get("registry").is_none()
                    && table.get("optional").and_then(|item| item.as_bool()) != Some(true)
                    && table
                        .get("package")
                        .and_then(|item| item.as_str())
                        .is_none_or(|name| name == "anyhow")
            });
            manifest_decisions.insert(manifest, proven);
            if proven {
                allowed.insert((project.clone(), state.rel_path));
            }
        }
    }
    // Authored glob exports participate in cache certification too. A normal
    // source refresh that changes these file fingerprints must replay callers.
    let globs = {
        let mut stmt = store.conn().prepare("SELECT e.project,e.file_path,g.value FROM raw_edges e, json_each(CASE WHEN json_type(e.properties,'$.receiver_anyhow_glob_files')='array' THEN json_extract(e.properties,'$.receiver_anyhow_glob_files') ELSE '[]' END) g WHERE json_extract(e.properties,'$.receiver_anyhow_factory_owner') IS NOT NULL AND (?1 IS NULL OR e.project=?1)").map_err(sqlite_err)?;
        let rows = stmt
            .query_map(rusqlite::params![only_project], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(sqlite_err)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sqlite_err)?
    };
    for (project, source, target) in globs {
        let target = Path::new(&source)
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(target)
            .to_string_lossy()
            .replace('\\', "/");
        let state = store
            .list_file_states(&project)?
            .into_iter()
            .find(|state| state.rel_path == target);
        documents.insert(
            format!("glob-source:{project}:{target}"),
            state.map(|state| state.sha256).unwrap_or_default(),
        );
    }
    let encoded = serde_json::to_vec(&documents)
        .map_err(|error| greppy_core::Error::Store(format!("Cargo identity encoding: {error}")))?;
    Ok((allowed, file_state::sha256_hex(&encoded)))
}

fn anyhow_factory_repair_key(project: &str) -> String {
    format!("{ANYHOW_FACTORY_REPAIR_KEY}.{project}")
}

pub fn anyhow_factory_edges_repaired(store: &Store) -> Result<bool> {
    let projects = {
        let mut stmt = store
            .conn()
            .prepare("SELECT name FROM projects")
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(sqlite_err)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sqlite_err)?
    };
    for project in projects {
        let (_, expected) = rust_anyhow_context(store, Some(&project))?;
        let present: bool = store
            .conn()
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM main.schema_meta WHERE key=?1 AND value=?2)",
                rusqlite::params![anyhow_factory_repair_key(&project), expected],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        if !present {
            return Ok(false);
        }
    }
    Ok(true)
}

const DIRECT_SELF_FIELD_REPAIR_KEY: &str = "greppy.rust_direct_self_field_repair.v1";

pub fn direct_self_field_edges_repaired(store: &Store) -> Result<bool> {
    let mut stmt = store
        .conn()
        .prepare("SELECT name FROM projects")
        .map_err(sqlite_err)?;
    let projects = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(sqlite_err)?;
    for project in projects {
        let key = format!(
            "{DIRECT_SELF_FIELD_REPAIR_KEY}.{}",
            project.map_err(sqlite_err)?
        );
        let present: bool = store
            .conn()
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM main.schema_meta WHERE key=?1 AND value='complete')",
                [key],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        if !present {
            return Ok(false);
        }
    }
    Ok(true)
}

pub const RUST_CALLER_EDGES_REPAIR_COMPLETE: &str = "complete";

pub fn rust_caller_edges_repaired(store: &Store) -> Result<bool> {
    let marker = store.conn().query_row(
        "SELECT value FROM main.schema_meta WHERE key = ?1",
        [RUST_CALLER_EDGES_REPAIR_META_KEY],
        |row| row.get::<_, String>(0),
    );
    match marker {
        Ok(value) => Ok(value == RUST_CALLER_EDGES_REPAIR_COMPLETE),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false),
        Err(error) => Err(sqlite_err(error)),
    }
}

pub fn mark_rust_caller_edges_repaired(store: &Store) -> Result<()> {
    store
        .conn()
        .execute(
            "INSERT INTO main.schema_meta(key, value) VALUES(?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (
                RUST_CALLER_EDGES_REPAIR_META_KEY,
                RUST_CALLER_EDGES_REPAIR_COMPLETE,
            ),
        )
        .map_err(sqlite_err)?;
    Ok(())
}

fn discovery_filtered_recovery_identities(
    store: &Store,
    project: &str,
) -> Result<std::collections::BTreeMap<String, IndexSkip>> {
    Ok(store
        .list_index_skips(project)?
        .into_iter()
        .filter(|skip| skip.reason == "discovery_filtered")
        .map(|skip| (skip.rel_path.clone(), skip))
        .collect())
}

fn current_discovery_filtered_recovery_identity(
    state: &FileState,
    skips: &std::collections::BTreeMap<String, IndexSkip>,
    indexed_paths: &std::collections::BTreeSet<&str>,
) -> bool {
    !indexed_paths.contains(state.rel_path.as_str())
        && skips.get(&state.rel_path).is_some_and(|skip| {
            skip.last_indexed_generation == state.last_indexed_generation
                && skip.size == state.size
                && skip.mtime_ns == state.mtime_ns
        })
}

/// Repair legacy identifier-only enum ranges on the exact indexed source.
/// Only span columns change: IDs, references, source ownership and vectors stay
/// intact. Rust caller repair v14 invokes this for already indexed snapshots.
pub fn recover_persisted_rust_enum_variant_spans(
    store: &mut Store,
    project: &str,
    root: &Path,
) -> Result<usize> {
    let variants = store.list_nodes_by_label(project, "EnumVariant", usize::MAX)?;
    if variants.is_empty() {
        return Ok(0);
    }
    let root = std::fs::canonicalize(root).map_err(|e| {
        greppy_core::Error::Invalid(format!("Rust variant span repair root unavailable: {e}"))
    })?;
    let states = store.list_file_states(project)?;
    let files: std::collections::BTreeSet<_> =
        variants.iter().map(|n| n.file_path.as_str()).collect();
    let mut prepared = Vec::new();
    let mut fingerprints = Vec::new();
    for file in files {
        let state = states.iter().find(|s| s.rel_path == file).ok_or_else(|| {
            greppy_core::Error::Invalid(format!(
                "Rust variant span repair requires indexed fingerprint: {file}"
            ))
        })?;
        let relative = Path::new(file);
        if !file.ends_with(".rs")
            || relative
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return Err(greppy_core::Error::Invalid(
                "unsafe Rust variant span repair path".into(),
            ));
        }
        let path = std::fs::canonicalize(root.join(relative)).map_err(|e| {
            greppy_core::Error::Invalid(format!(
                "Rust variant span repair source {file} unavailable: {e}"
            ))
        })?;
        if !path.starts_with(&root) {
            return Err(greppy_core::Error::Invalid(
                "Rust variant span repair source escapes root".into(),
            ));
        }
        let (bytes, _) = read_stable_file(&path).map_err(|e| {
            greppy_core::Error::Invalid(format!(
                "Rust variant span repair source {file} unreadable: {e}"
            ))
        })?;
        if file_state::sha256_hex(&bytes) != state.sha256 {
            return Err(greppy_core::Error::Invalid(format!(
                "Rust variant span repair source {file} changed since indexing"
            )));
        }
        let extraction = parser_extract(Language::Rust, &bytes, file)?;
        let (extraction, dropped, error) = validate_or_degrade(Language::Rust, file, extraction);
        if dropped != 0 || error.is_some() {
            return Err(greppy_core::Error::Invalid(format!(
                "Rust variant span repair extraction incomplete: {file}"
            )));
        }
        let definitions: std::collections::BTreeMap<_, _> = extraction
            .nodes
            .iter()
            .map(|n| (n.qualified_name.as_str(), n))
            .collect();
        for cached in variants.iter().filter(|n| n.file_path == file) {
            let node = definitions
                .get(cached.qualified_name.as_str())
                .ok_or_else(|| {
                    greppy_core::Error::Invalid(format!(
                        "Rust variant span repair definition unavailable: {}",
                        cached.qualified_name
                    ))
                })?;
            if node.label != "EnumVariant"
                || node.name != cached.name
                || node.properties != cached.properties
                || cached.start_line != i64::from(node.start_line)
                || (cached.end_line != cached.start_line
                    && cached.end_line != i64::from(node.end_line))
            {
                return Err(greppy_core::Error::Invalid(format!(
                    "Rust variant span repair definition mismatch: {}",
                    cached.qualified_name
                )));
            }
            if cached.end_line != i64::from(node.end_line) {
                prepared.push((
                    cached.id,
                    i64::from(node.start_line),
                    i64::from(node.end_line),
                ));
            }
        }
        for node in definitions.values().filter(|n| n.label == "EnumVariant") {
            if !variants.iter().any(|cached| {
                cached.qualified_name == node.qualified_name && cached.file_path == file
            }) {
                return Err(greppy_core::Error::Invalid(format!(
                    "Rust variant span repair stored definition missing: {}",
                    node.qualified_name
                )));
            }
        }
        fingerprints.push((path, state.sha256.clone()));
    }
    // Recheck source just before publication. A failed validation publishes no
    // partially repaired spans and never certifies caller recovery.
    for (path, sha) in fingerprints {
        let (bytes, _) = read_stable_file(&path).map_err(|e| {
            greppy_core::Error::Invalid(format!("Rust variant span repair source changed: {e}"))
        })?;
        if file_state::sha256_hex(&bytes) != sha {
            return Err(greppy_core::Error::Invalid(
                "Rust variant span repair source changed before publication".into(),
            ));
        }
    }
    store
        .conn()
        .execute_batch("SAVEPOINT greppy_rust_variant_spans")
        .map_err(sqlite_err)?;
    let result = (|| -> Result<()> {
        store.update_node_spans(&prepared)?;
        Ok(())
    })();
    match result {
        Ok(()) => store
            .conn()
            .execute_batch("RELEASE greppy_rust_variant_spans")
            .map_err(sqlite_err)?,
        Err(error) => {
            store
                .conn()
                .execute_batch(
                    "ROLLBACK TO greppy_rust_variant_spans; RELEASE greppy_rust_variant_spans",
                )
                .map_err(sqlite_err)?;
            return Err(error);
        }
    }
    Ok(prepared.len())
}

/// Recover references and caller provenance without rebuilding nodes or embeddings.
/// Validate every visible Rust source before writing anything.
/// Private usage overrides also work for immutable Base files: no file-state ownership
/// is copied into Delta, and ordinary sparse publication remains Delta-bounded.
pub fn recover_persisted_rust_usages(
    store: &mut Store,
    project: &str,
    root: &Path,
) -> Result<usize> {
    recover_persisted_rust_enum_variant_spans(store, project, root)?;
    let states = store.list_file_states(project)?;
    let indexed_paths = {
        let mut statement = store
            .conn()
            .prepare(
                "SELECT DISTINCT file_path FROM nodes WHERE project=?1 AND file_path LIKE '%.rs'",
            )
            .map_err(sqlite_err)?;
        let rows = statement
            .query_map([project], |row| row.get::<_, String>(0))
            .map_err(sqlite_err)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sqlite_err)?
    };
    if indexed_paths
        .iter()
        .any(|path| !states.iter().any(|state| &state.rel_path == path))
    {
        return Err(greppy_core::Error::Invalid("Rust reference repair requires indexed source fingerprints for every visible Rust file".into()));
    }
    let discovery_filtered = discovery_filtered_recovery_identities(store, project)?;
    let indexed_paths = indexed_paths
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    let root = std::fs::canonicalize(root).map_err(|error| {
        greppy_core::Error::Invalid(format!("Rust reference repair cannot read root: {error}"))
    })?;
    let mut replacements = Vec::new();
    let mut files = Vec::new();
    for state in states
        .iter()
        .filter(|state| state.rel_path.ends_with(".rs"))
    {
        // CoW retains identities for deliberately excluded files without
        // graph definitions. Apply the same narrow eligibility rule as JS/TS;
        // stale skips or any visible nodes still require full validation.
        if current_discovery_filtered_recovery_identity(state, &discovery_filtered, &indexed_paths)
        {
            continue;
        }
        let relative = Path::new(&state.rel_path);
        if relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err(greppy_core::Error::Invalid(
                "unsafe Rust repair source path".into(),
            ));
        }
        let path = std::fs::canonicalize(root.join(relative)).map_err(|error| {
            greppy_core::Error::Invalid(format!(
                "Rust reference repair source {} unavailable: {error}",
                state.rel_path
            ))
        })?;
        if !path.starts_with(&root) {
            return Err(greppy_core::Error::Invalid(
                "Rust repair source escapes root".into(),
            ));
        }
        let (bytes, _) = read_stable_file(&path).map_err(|error| {
            greppy_core::Error::Invalid(format!(
                "Rust reference repair source {} unreadable: {error}",
                state.rel_path
            ))
        })?;
        if file_state::sha256_hex(&bytes) != state.sha256 {
            return Err(greppy_core::Error::Invalid(format!(
                "Rust reference repair source {} changed since indexing",
                state.rel_path
            )));
        }
        let extraction = parser_extract(Language::Rust, &bytes, &state.rel_path)?;
        let (extraction, dropped, error) =
            validate_or_degrade(Language::Rust, &state.rel_path, extraction);
        if dropped != 0 || error.is_some() {
            return Err(greppy_core::Error::Invalid(format!(
                "Rust reference repair source {} failed extraction validation",
                state.rel_path
            )));
        }
        // Receiver provenance depends on persisted Field declarations as well
        // as raw calls. Never certify an older Base whose field facts are
        // absent or differ from the fingerprint-validated source extraction.
        // Enum variants (and repeated cfg declarations) can share a qualified
        // name. Persistence upserts in extraction order, so only the final
        // definition can describe the visible stored facts. Earlier overwritten
        // declarations must not make a freshly indexed file fail recovery.
        let definitions: std::collections::BTreeMap<_, _> = extraction
            .nodes
            .iter()
            .map(|node| (node.qualified_name.as_str(), node))
            .collect();
        for field in definitions.values().filter(|node| node.label == "Field") {
            let cached = store.get_node_by_qname(project, &field.qualified_name)?;
            if cached.as_ref().is_none_or(|node| {
                node.label != "Field"
                    || node.properties.get("return_type") != field.properties.get("return_type")
                    || node.properties.get("generic_payload")
                        != field.properties.get("generic_payload")
            }) {
                return Err(greppy_core::Error::Invalid(format!(
                    "Rust reference repair source {} has unavailable or stale declared field facts for {}; re-extract declared field nodes from source before caller repair",
                    state.rel_path, field.qualified_name
                )));
            }
        }
        for trait_node in definitions
            .values()
            .filter(|node| node.label == "Interface")
        {
            let cached = store.get_node_by_qname(project, &trait_node.qualified_name)?;
            if cached.as_ref().is_none_or(|node| {
                node.label != "Interface"
                    || node.properties.get("has_bounds") != trait_node.properties.get("has_bounds")
                    || node.properties.get("as_ref_receiver")
                        != trait_node.properties.get("as_ref_receiver")
            }) {
                return Err(greppy_core::Error::Invalid(format!(
                    "Rust reference repair source {} has unavailable or stale trait receiver facts for {}; re-extract trait nodes from source before caller repair",
                    state.rel_path, trait_node.qualified_name
                )));
            }
        }
        files.push(state.rel_path.clone());
        replacements.extend(
            extraction
                .edges
                .iter()
                .filter(|edge| matches!(edge.edge_type.as_str(), "USAGE" | "CALLS"))
                .map(|edge| new_raw_edge_for(project, &state.rel_path, edge)),
        );
    }
    // All source fingerprints validate before any persisted write. Replace
    // USAGE and CALLS contributions, retaining imports, nodes and cached data.
    let usages = replacements
        .iter()
        .filter(|edge| edge.edge_type == "USAGE")
        .cloned()
        .collect::<Vec<_>>();
    let calls = replacements
        .into_iter()
        .filter(|edge| edge.edge_type == "CALLS")
        .collect::<Vec<_>>();
    let changed = store.replace_validated_rust_usages(project, &files, &usages)?;
    Ok(changed + store.replace_validated_rust_calls(project, &files, &calls)?)
}

pub const JS_TS_USAGE_REPAIR_KEY: &str = "greppy.js_ts_usage_repair_v4";

/// JSON `{"count":N,"paths":[...]}` for files skipped by
/// [`recover_persisted_js_ts_usages`] because extraction still violated the
/// provider contract after degradation. `paths` is sorted and capped at 20;
/// `count` is the full skip count. Written in the same savepoint as the
/// completion marker, including `{"count":0,"paths":[]}` so a later repair
/// clears a stale diagnostic.
pub const JS_TS_USAGE_REPAIR_SKIPS_KEY: &str = "greppy.js_ts_usage_repair_skips_v1";

// Test-only stand-in for a residual contract failure. Real JS/TS extracts
// rewrite file identity and confidence, and `validate_or_degrade` drops the
// invalid spans and blank identities the grammar emits, so a source file
// cannot currently force the post-degrade error. The repair test arms this
// for `contract-invalid.js` only.
#[cfg(test)]
thread_local! {
    static FORCE_JS_TS_CONTRACT_SKIP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn js_ts_usages_repaired(store: &Store) -> Result<bool> {
    store
        .conn()
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM main.schema_meta WHERE key=?1 AND value='complete')",
            [JS_TS_USAGE_REPAIR_KEY],
            |row| row.get(0),
        )
        .map_err(sqlite_err)
}

fn js_ts_usage_repair_skip_json(paths: &[String]) -> String {
    let mut ordered = paths.to_vec();
    ordered.sort();
    let count = ordered.len();
    ordered.truncate(20);
    serde_json::json!({
        "count": count,
        "paths": ordered,
    })
    .to_string()
}

fn warn_js_ts_usage_repair_skips(paths: &[String]) {
    let mut ordered = paths.to_vec();
    ordered.sort();
    let count = ordered.len();
    ordered.truncate(20);
    tracing::warn!(
        count,
        paths = %ordered.join(", "),
        "JS/TS usage repair skipped files that still violate the provider contract"
    );
}

/// Fresh indexing and recovery must use the same validated extraction. Grammar
/// recovery can emit an invalid anonymous record among valid definitions; the
/// provider contract already filters that record at first use. Reject only an
/// extraction that still violates the contract, not a successfully cured one.
fn validated_js_ts_repair_extraction(
    language: Language,
    relative: &str,
    extraction: greppy_parser::ExtractionResult,
) -> Result<greppy_parser::ExtractionResult> {
    #[cfg(test)]
    if FORCE_JS_TS_CONTRACT_SKIP.with(|flag| flag.get())
        && relative.rsplit(['/', '\\']).next() == Some("contract-invalid.js")
    {
        return Err(greppy_core::Error::Invalid(format!(
            "JS/TS usage repair extraction incomplete for {relative}: synthetic residual contract violation (0 invalid records removed)"
        )));
    }
    let (validated, dropped, error) = validate_or_degrade(language, relative, extraction);
    if let Some(error) = error {
        return Err(greppy_core::Error::Invalid(format!(
            "JS/TS usage repair extraction incomplete for {relative}: {error} ({dropped} invalid records removed)"
        )));
    }
    Ok(validated)
}

/// One-shot source-validated usage recovery, separately partitioned from Rust.
/// Validate all visible source and definition identities before atomic publication.
pub fn recover_persisted_js_ts_usages(
    store: &mut Store,
    project: &str,
    root: &Path,
) -> Result<bool> {
    if js_ts_usages_repaired(store)? {
        return Ok(false);
    }
    let root = std::fs::canonicalize(root).map_err(|e| {
        greppy_core::Error::Invalid(format!("JS/TS usage repair root unavailable: {e}"))
    })?;
    let states = store.list_file_states(project)?;
    let indexed = store.list_nodes(project, "", "", 0, i64::MAX as usize)?;
    let relevant = |path: &str| {
        matches!(
            greppy_parser::language_for_path(Path::new(path)),
            Language::JavaScript | Language::TypeScript { .. }
        )
    };
    if indexed.iter().any(|node| {
        relevant(&node.file_path) && !states.iter().any(|state| state.rel_path == node.file_path)
    }) {
        return Err(greppy_core::Error::Invalid(
            "JS/TS usage repair requires indexed source fingerprints".into(),
        ));
    }
    // A Store-CoW Delta retains fingerprints for discovery-filtered files,
    // although those files deliberately have no graph definitions. Do not
    // re-extract them as if their absent definitions were cache corruption.
    let discovery_filtered = discovery_filtered_recovery_identities(store, project)?;
    let indexed_paths: std::collections::BTreeSet<_> =
        indexed.iter().map(|node| node.file_path.as_str()).collect();
    let mut files = Vec::new();
    let mut extracted = Vec::new();
    let mut skipped_contract_files = Vec::new();
    for state in states.iter().filter(|state| relevant(&state.rel_path)) {
        if current_discovery_filtered_recovery_identity(state, &discovery_filtered, &indexed_paths)
        {
            continue;
        }
        let relative = Path::new(&state.rel_path);
        if relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err(greppy_core::Error::Invalid(
                "unsafe JS/TS usage repair path".into(),
            ));
        }
        let path = std::fs::canonicalize(root.join(relative)).map_err(|e| {
            greppy_core::Error::Invalid(format!("JS/TS usage repair source unavailable: {e}"))
        })?;
        if !path.starts_with(&root) {
            return Err(greppy_core::Error::Invalid(
                "JS/TS usage repair source escapes root".into(),
            ));
        }
        let (bytes, _) = read_stable_file(&path).map_err(|e| {
            greppy_core::Error::Invalid(format!("JS/TS usage repair source unreadable: {e}"))
        })?;
        if file_state::sha256_hex(&bytes) != state.sha256 {
            return Err(greppy_core::Error::Invalid(format!(
                "JS/TS usage repair source {} changed since indexing",
                state.rel_path
            )));
        }
        let language = greppy_parser::language_for_path(relative);
        let extraction = parser_extract(language, &bytes, &state.rel_path)?;
        // One file that still violates the provider contract must not refuse
        // the repository. Previously persisted edges for that file stay as
        // they are; fingerprint and definition-identity failures still abort.
        // Marking the repair complete despite the skip is safe: the bytes were
        // just verified against their recorded sha256, so the same extractor
        // fails the same way on every retry. A changed file is re-extracted by
        // ordinary indexing, and an extractor change bumps INDEXER_VERSION_BASE,
        // which forces a full re-extraction.
        let extraction =
            match validated_js_ts_repair_extraction(language, &state.rel_path, extraction) {
                Ok(extraction) => extraction,
                Err(_) => {
                    skipped_contract_files.push(state.rel_path.clone());
                    continue;
                }
            };
        // Persistence upserts in extraction order by (project, qualified_name).
        // Object-literal methods can share a qualified name: validate the final
        // stored definition, rather than rejecting the overwritten earlier span.
        let definitions: std::collections::BTreeMap<_, _> = extraction
            .nodes
            .iter()
            .map(|node| (node.qualified_name.as_str(), node))
            .collect();
        for node in definitions.values() {
            let cached = store.get_node_by_qname(project, &node.qualified_name)?;
            if cached.as_ref().is_none_or(|cached| {
                cached.label != node.label
                    || cached.file_path != state.rel_path
                    || cached.start_line != i64::from(node.start_line)
                    || cached.end_line != i64::from(node.end_line)
            }) {
                return Err(greppy_core::Error::Invalid(format!(
                    "JS/TS usage repair definition unavailable: {}",
                    node.qualified_name
                )));
            }
        }
        files.push(state.rel_path.clone());
        extracted.extend(extraction.edges);
    }
    let mut index = GraphIndex::load(store, project)?;
    for edge in extracted.iter().filter(|edge| edge.edge_type == "IMPORTS") {
        if let Some(source) = index.by_qname(&edge.source_qualified_name) {
            let file = source.file_path.clone();
            let source_id = source.id;
            index.record_import_items(edge, &file);
            if let Some(name) = edge
                .properties
                .get("imported_name")
                .and_then(|v| v.as_str())
            {
                let path = edge
                    .properties
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let Some(target) = index
                    .resolve_import_target(&file, name, path, edge.properties.get("imported_items"))
                    .filter(|target| *target != source_id)
                {
                    index.record_import(&file, target);
                }
            }
        }
    }
    let mut resolved = Vec::new();

    for edge in extracted
        .iter()
        .filter(|edge| matches!(edge.edge_type.as_str(), "USAGE" | "CALLS"))
    {
        let Some(source) = index.by_qname(&edge.source_qualified_name) else {
            return Err(greppy_core::Error::Invalid(format!(
                "JS/TS usage repair source identity unavailable: {}:{} source={} reference={}",
                edge.file_path,
                edge.line,
                edge.source_qualified_name,
                edge.properties
                    .get(if edge.edge_type == "CALLS" {
                        "callee_name"
                    } else {
                        "ref_name"
                    })
                    .and_then(|value| value.as_str())
                    .unwrap_or("<unknown>")
            )));
        };
        let target = if edge.edge_type == "CALLS" {
            index.resolve_call_target(edge)
        } else {
            index.resolve_usage_target(edge, source.id)
        };
        if let Some(target) =
            target.filter(|target| edge.edge_type == "CALLS" || *target != source.id)
        {
            resolved.push((
                source.id,
                target,
                edge.source_qualified_name.clone(),
                index.qname_for_id(target).unwrap().to_owned(),
                // Extraction stores the reference address separately from
                // properties. Recovery must publish it just as fresh indexing
                // does, or navigation falls back to the owner's definition.
                new_raw_edge_for(project, &edge.file_path, edge).properties,
                edge.edge_type.clone(),
            ));
        }
    }
    let calls = extracted
        .iter()
        .filter(|edge| edge.edge_type == "CALLS")
        .map(|edge| new_raw_edge_for(project, &edge.file_path, edge))
        .collect::<Vec<_>>();
    let raw = extracted
        .iter()
        .filter(|edge| edge.edge_type == "USAGE")
        .map(|edge| new_raw_edge_for(project, &edge.file_path, edge))
        .collect::<Vec<_>>();
    let skip_json = js_ts_usage_repair_skip_json(&skipped_contract_files);
    store
        .conn()
        .execute_batch("SAVEPOINT greppy_js_ts_usage_repair")
        .map_err(sqlite_err)?;
    let result = (|| -> Result<()> {
        store.replace_validated_js_ts_usages(project, &files, &raw)?;
        store.replace_validated_js_ts_calls(project, &files, &calls)?;
        for file in &files {
            if store.is_overlay() {
                store.conn().execute("DELETE FROM main.overlay_edges WHERE project=?1 AND edge_type IN ('USAGE','CALLS') AND source_qualified_name IN (SELECT qualified_name FROM nodes WHERE project=?1 AND file_path=?2)", rusqlite::params![project,file]).map_err(sqlite_err)?;
            } else {
                store.conn().execute("DELETE FROM main.edges WHERE project=?1 AND edge_type IN ('USAGE','CALLS') AND source_id IN (SELECT id FROM nodes WHERE project=?1 AND file_path=?2)", rusqlite::params![project,file]).map_err(sqlite_err)?;
            }
        }
        for (source, target, source_name, target_name, mut properties, kind) in resolved {
            if store.is_overlay() {
                properties["greppy_base_repair_v2"] = serde_json::json!(1);
                store.insert_overlay_edges(&[NewOverlayEdge {
                    project: project.into(),
                    source_qualified_name: source_name,
                    target_qualified_name: target_name,
                    edge_type: kind.clone(),
                    properties,
                }])?;
            } else {
                store.conn().execute("INSERT INTO main.edges(project,source_id,target_id,edge_type,properties) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(source_id,target_id,edge_type) DO UPDATE SET properties=excluded.properties", rusqlite::params![project,source,target,kind,properties.to_string()]).map_err(sqlite_err)?;
            }
        }
        store.conn().execute("INSERT INTO main.schema_meta(key,value) VALUES(?1,'complete') ON CONFLICT(key) DO UPDATE SET value=excluded.value", [JS_TS_USAGE_REPAIR_KEY]).map_err(sqlite_err)?;
        store.conn().execute("INSERT INTO main.schema_meta(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", rusqlite::params![JS_TS_USAGE_REPAIR_SKIPS_KEY, skip_json]).map_err(sqlite_err)?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            store
                .conn()
                .execute_batch("RELEASE greppy_js_ts_usage_repair")
                .map_err(sqlite_err)?;
            if !skipped_contract_files.is_empty() {
                warn_js_ts_usage_repair_skips(&skipped_contract_files);
            }
            Ok(true)
        }
        Err(error) => {
            store
                .conn()
                .execute_batch(
                    "ROLLBACK TO greppy_js_ts_usage_repair; RELEASE greppy_js_ts_usage_repair",
                )
                .map_err(sqlite_err)?;
            Err(error)
        }
    }
}

/// Repair Effect.fn identities in a private overlay without mutating its Base
/// or copying Base file-state/content/vector ownership into Delta.
pub fn recover_visible_effect_fn_bindings(
    store: &mut Store,
    project: &str,
    root: &Path,
) -> Result<bool> {
    store
        .conn()
        .execute_batch("SAVEPOINT greppy_effect_fn_repair")
        .map_err(sqlite_err)?;
    let result = recover_visible_effect_fn_bindings_inner(store, project, root);
    match result {
        Ok(repaired) => {
            store
                .conn()
                .execute_batch("RELEASE greppy_effect_fn_repair")
                .map_err(sqlite_err)?;
            Ok(repaired)
        }
        Err(error) => {
            store
                .conn()
                .execute_batch(
                    "ROLLBACK TO greppy_effect_fn_repair; RELEASE greppy_effect_fn_repair",
                )
                .map_err(sqlite_err)?;
            Err(error)
        }
    }
}

fn recover_visible_effect_fn_bindings_inner(
    store: &mut Store,
    project: &str,
    root: &Path,
) -> Result<bool> {
    if !store.is_overlay() {
        return Ok(false);
    }
    let marker = format!("greppy.effect_fn_repair_v9.{project}");
    let completed: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM main.schema_meta WHERE key=?1 AND value='complete'",
            [&marker],
            |r| r.get(0),
        )
        .map_err(sqlite_err)?;
    if completed != 0 {
        return Ok(false);
    }
    let root = std::fs::canonicalize(root).map_err(|e| {
        greppy_core::Error::Invalid(format!("Effect.fn repair root unavailable: {e}"))
    })?;
    let mut prepared = Vec::new();
    for state in store.list_file_states(project)? {
        let language = greppy_parser::language_for_path(Path::new(&state.rel_path));
        if !matches!(language, Language::JavaScript | Language::TypeScript { .. }) {
            continue;
        }
        let relative = Path::new(&state.rel_path);
        if relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return Err(greppy_core::Error::Invalid(
                "unsafe Effect.fn repair path".into(),
            ));
        }
        let path = std::fs::canonicalize(root.join(relative)).map_err(|e| {
            greppy_core::Error::Invalid(format!("Effect.fn source unavailable: {e}"))
        })?;
        if !path.starts_with(&root) {
            return Err(greppy_core::Error::Invalid(
                "Effect.fn source escapes root".into(),
            ));
        }
        let (bytes, _) = read_stable_file(&path).map_err(|e| {
            greppy_core::Error::Invalid(format!("Effect.fn source unreadable: {e}"))
        })?;
        if file_state::sha256_hex(&bytes) != state.sha256 {
            return Err(greppy_core::Error::Invalid(format!(
                "Effect.fn source {} changed since indexing",
                state.rel_path
            )));
        }
        let extraction = parser_extract(language, &bytes, &state.rel_path)?;
        let (extraction, dropped, error) =
            validate_or_degrade(language, &state.rel_path, extraction);
        if dropped != 0 || error.is_some() {
            return Err(greppy_core::Error::Invalid(
                "Effect.fn extraction validation failed".into(),
            ));
        }
        prepared.push((state.rel_path, extraction, state.sha256));
    }
    // Compare complete contributions, including duplicate counts, before
    // publishing overrides. Identical clean Base relations need no Delta copy.
    let signature = |source: &str, target: &str, kind: &str, properties: &serde_json::Value| {
        (
            source.to_owned(),
            target.to_owned(),
            kind.to_owned(),
            properties.to_string(),
        )
    };
    let mut previous = std::collections::HashMap::new();
    for edge in store.list_raw_edges(project)? {
        let counts = previous
            .entry(edge.file_path)
            .or_insert_with(std::collections::BTreeMap::new);
        *counts
            .entry(signature(
                &edge.source_qname,
                &edge.target_qname,
                &edge.edge_type,
                &edge.properties,
            ))
            .or_insert(0usize) += 1;
    }
    let mut changed_paths = Vec::new();
    let mut changed_identity = false;
    // Validate every visible source fingerprint before changing any identity.
    for (path, extraction, source_sha256) in &prepared {
        for node in extraction
            .nodes
            .iter()
            .filter(|node| node.label == "Function")
        {
            let old_qname = node.qualified_name.replace("::Function::", "::Variable::");
            if let Some(old) = store.get_node_by_qname(project, &old_qname)? {
                if old.label == "Variable"
                    && old.name == node.name
                    && old.start_line == i64::from(node.start_line)
                    && old.end_line == i64::from(node.end_line)
                {
                    store.update_node_identity(old.id, &node.label, &node.qualified_name)?;
                    changed_identity = true;
                }
            }
        }
        let mut current = std::collections::BTreeMap::new();
        for edge in &extraction.edges {
            let edge = new_raw_edge_for(project, path, edge);
            *current
                .entry(signature(
                    &edge.source_qname,
                    &edge.target_qname,
                    &edge.edge_type,
                    &edge.properties,
                ))
                .or_insert(0usize) += 1;
        }
        if previous.get(path).cloned().unwrap_or_default() != current {
            persist_raw_edges_for_file(store, project, path, &extraction.edges)?;
            changed_paths.push((path, source_sha256));
        } else if store.conn().query_row(
            "SELECT EXISTS(SELECT 1 FROM main.js_ts_reference_override_files WHERE project=?1 AND file_path=?2)",
            rusqlite::params![project, path], |row| row.get::<_, bool>(0),
        ).map_err(sqlite_err)? {
            // An already-current legacy v8 contribution still needs a v9
            // certificate. Do not create ownership/overrides for clean Base.
            store.certify_js_ts_reference_repair(project, path, source_sha256)?;
        }
    }
    let changed_relations = !changed_paths.is_empty();
    for (path, source_sha256) in changed_paths {
        store.conn().execute("INSERT OR IGNORE INTO main.js_ts_reference_override_files(project,file_path) VALUES(?1,?2)", rusqlite::params![project,path]).map_err(sqlite_err)?;
        store.certify_js_ts_reference_repair(project, path, source_sha256)?;
    }
    // Current extraction already resolved clean Base relations. Rebuilding
    // them needlessly materializes and pins duplicate CALLS/USAGE rows in the
    // Delta, even after an exact source revert. Older or unidentified Bases
    // still require the compatibility repair when their raw facts are equal.
    let current_base: bool = store
        .conn()
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM greppy_base.workspace_state)
         AND NOT EXISTS(SELECT 1 FROM greppy_base.workspace_state WHERE indexer_version <> ?1)",
            [greppy_core::INDEXER_VERSION_BASE],
            |row| row.get(0),
        )
        .map_err(sqlite_err)?;
    if changed_identity || changed_relations || !current_base {
        rebuild_visible_overlay_edges(store, project)?;
    }
    store.conn().execute("INSERT INTO main.schema_meta(key,value) VALUES(?1,'complete') ON CONFLICT(key) DO UPDATE SET value=excluded.value", [marker]).map_err(sqlite_err)?;
    Ok(true)
}

/// One-shot single-store compatibility repair. Replace only Rust-owned
/// non-structural relations from persisted raw edges; nodes, file identity,
/// graph generation, content, embeddings and non-Rust edges remain untouched.
/// Edge replacement and completion marker commit in the same transaction.
pub fn rebuild_single_store_rust_edges(store: &mut Store, project: &str) -> Result<usize> {
    if store.is_overlay() {
        return Err(greppy_core::Error::Invalid(
            "single-store Rust repair requires a private Store".into(),
        ));
    }
    let root = store
        .get_project(project)?
        .ok_or_else(|| greppy_core::Error::Invalid("Rust repair project is missing".into()))?
        .root_path;
    recover_persisted_rust_usages(store, project, Path::new(&root))?;
    let raw = load_all_raw_edges(store, project)?;
    let rust_edges = raw
        .into_iter()
        .filter(|edge| edge.file_path.ends_with(".rs"))
        .collect::<Vec<_>>();
    if rust_edges.is_empty() {
        let existing: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM main.edges WHERE project = ?1
             AND edge_type NOT IN ('CONTAINS_FOLDER', 'CONTAINS_FILE', 'DEFINES')
             AND source_id IN (SELECT id FROM nodes WHERE project = ?1 AND file_path LIKE '%.rs')",
                [project],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        if existing != 0 {
            return Err(greppy_core::Error::Invalid(
                "single-store Rust graph has relations but no persisted Rust raw edges to repair"
                    .into(),
            ));
        }
    }
    note_reresolved(rust_edges.len());
    resolve_edges_with_replacement(store, project, &rust_edges, &mut |_| {}, &[], true)
}

/// Rebuild the complete bounded logical edge contribution of a private
/// Store-CoW Delta. This is intentionally O(Delta raw edges), never O(Base):
/// it closes no-op/prune generations where the incremental parser correctly
/// skips unchanged dirty files but publication still needs a self-contained
/// next Delta snapshot. A small set of compatibility edges published by the
/// one-shot Base repair is carried forward from existing overlay rows; it is
/// filtered by current visibility and does not rescan Base raw edges.
pub fn rebuild_overlay_edges(store: &mut Store, project: &str) -> Result<usize> {
    if !store.is_overlay() {
        return Err(greppy_core::Error::Invalid(
            "rebuild_overlay_edges requires an overlay Store".into(),
        ));
    }
    let raw_edges = load_all_raw_edges(store, project)?;
    let preserved = repaired_base_overlay_edges(store, project)?;
    resolve_and_persist_edges_with_progress_and_preserved(
        store,
        project,
        &raw_edges,
        &mut |_| {},
        &preserved,
    )
}

/// Rebuild every visible logical edge in a private Store-CoW Delta from the
/// composed raw-edge view. This is reserved for one-shot compatibility repair
/// of an immutable Base whose raw edges were extracted by an older resolver;
/// ordinary Delta indexing must keep using [`rebuild_overlay_edges`] so its
/// work remains bounded by Delta-owned files.
pub fn rebuild_visible_overlay_edges(store: &mut Store, project: &str) -> Result<usize> {
    if !store.is_overlay() {
        return Err(greppy_core::Error::Invalid(
            "rebuild_visible_overlay_edges requires an overlay Store".into(),
        ));
    }
    let raw_edges = store.list_raw_edges(project)?;
    let edges = raw_edges
        .into_iter()
        .map(extracted_edge_from_raw)
        .collect::<Vec<_>>();
    let resolved = resolve_and_persist_edges(store, project, &edges)?;
    mark_missing_base_repair_edges(store, project)?;
    Ok(resolved)
}

fn mark_missing_base_repair_edges(store: &mut Store, project: &str) -> Result<()> {
    store
        .conn()
        .execute(
            "UPDATE main.overlay_edges AS d
             SET properties = json_set(
                 CASE WHEN json_type(d.properties) = 'object' THEN d.properties ELSE '{}' END,
                 '$.greppy_base_repair_v2', 1)
             WHERE d.project = ?1
               AND EXISTS (
                   SELECT 1 FROM nodes s
                   WHERE s.project = d.project
                     AND s.qualified_name = d.source_qualified_name
                     AND NOT EXISTS (
                         SELECT 1 FROM temp.greppy_hidden_paths h
                         WHERE h.path = s.file_path
                     )
               )
               AND (d.edge_type IN ('USAGE', 'CALLS') OR NOT EXISTS (
                   SELECT 1
                   FROM greppy_base.nodes bs
                   JOIN greppy_base.edges e
                     ON e.project = bs.project AND e.source_id = bs.id
                   JOIN greppy_base.nodes bt
                     ON bt.project = e.project AND bt.id = e.target_id
                   WHERE bs.project = d.project
                     AND bs.qualified_name = d.source_qualified_name
                     AND bt.project = d.project
                     AND bt.qualified_name = d.target_qualified_name
                     AND e.project = d.project
                     AND e.edge_type = d.edge_type
               ))
               AND json_extract(d.properties, '$.greppy_base_repair_v2') IS NULL",
            rusqlite::params![project],
        )
        .map_err(sqlite_err)?;
    Ok(())
}

fn repaired_base_overlay_edges(store: &Store, project: &str) -> Result<Vec<NewOverlayEdge>> {
    let rows = {
        let mut stmt = store
            .conn()
            .prepare_cached(
                "SELECT source_qualified_name, target_qualified_name, edge_type, properties
                 FROM main.overlay_edges
                 WHERE project = ?1
                   AND json_extract(properties, '$.greppy_base_repair_v2') = 1",
            )
            .map_err(sqlite_err)?;
        let collected = stmt
            .query_map(rusqlite::params![project], |row| {
                let properties: String = row.get(3)?;
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    properties,
                ))
            })
            .map_err(sqlite_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(sqlite_err)?;
        collected
    };
    let mut preserved = Vec::with_capacity(rows.len());
    for (source, target, edge_type, properties) in rows {
        let Some(source_node) = store.get_node_by_qname(project, &source)? else {
            continue;
        };
        let source_hidden: i64 = store
            .conn()
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM temp.greppy_hidden_paths WHERE path = ?1)",
                rusqlite::params![source_node.file_path],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        if source_hidden != 0 || store.get_node_by_qname(project, &target)?.is_none() {
            continue;
        }
        preserved.push(NewOverlayEdge {
            project: project.to_string(),
            source_qualified_name: source,
            target_qualified_name: target,
            edge_type,
            properties: serde_json::from_str(&properties).map_err(|error| {
                greppy_core::Error::Store(format!("overlay edge JSON: {error}"))
            })?,
        });
    }
    Ok(preserved)
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
/// `qualified_name`, `name`, `label`, `file_path`, Field declared types and
/// `generic_payload`, and Interface `has_bounds` / `as_ref_receiver`. Edge resolution is a
/// pure function of this set (plus the per-edge raw data): `by_qname`
/// targets, `by_name` candidate sets, the same-file preference, and IMPORTS
/// disambiguation and typed field receivers consult these facts. Node `id`s are
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
            "SELECT qualified_name, name, label, file_path,
                    CASE
                        WHEN label = 'Field' THEN
                            COALESCE(json_extract(properties, '$.return_type'), '')
                            || char(31)
                            || COALESCE(json_extract(properties, '$.generic_payload'), '')
                        WHEN label = 'Interface' THEN
                            COALESCE(json_extract(properties, '$.has_bounds'), '')
                            || char(31)
                            || COALESCE(json_extract(properties, '$.as_ref_receiver'), '')
                        ELSE ''
                    END
             FROM nodes WHERE project = ?1
               AND label NOT IN ('Project', 'Folder', 'File')",
        )
        .map_err(sqlite_err)?;
    let rows = stmt
        .query_map(rusqlite::params![project], |r| {
            Ok(format!(
                "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?.unwrap_or_default(),
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
    progress: &mut dyn FnMut(IndexBuildProgress),
) -> Result<usize> {
    // No file changed → nothing cascaded, graph already complete. O(1).
    if changed_files.is_empty() {
        return count_edges(store, project);
    }

    // A private Delta owns only edges extracted from dirty files. Rebuild
    // that bounded logical edge set from its own raw rows: unchanged Base
    // edges are already supplied by the composed view and must never be
    // copied into the Delta merely because a definition changed.
    if store.is_overlay() {
        let raw_edges = load_all_raw_edges(store, project)?;
        // A persisted Delta may also contain the bounded Base-edge repairs
        // published by `rebuild_visible_overlay_edges`. Replacing the
        // Delta's resolved edges from its own raw rows must carry those
        // explicitly marked rows forward; otherwise the structural index
        // pass erases them before the caller can rebuild the composed view.
        let repaired_base_edges = repaired_base_overlay_edges(store, project)?;
        note_reresolved(raw_edges.len());
        return resolve_and_persist_edges_with_progress_and_preserved(
            store,
            project,
            &raw_edges,
            progress,
            &repaired_base_edges,
        );
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
                "DELETE FROM main.edges WHERE project = ?1",
                rusqlite::params![project],
            )
            .map_err(sqlite_err)?;
        let raw_edges = load_all_raw_edges(store, project)?;
        note_reresolved(raw_edges.len());
        return resolve_and_persist_edges_with_progress(store, project, &raw_edges, progress);
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
    let mut examined = 0usize;
    progress(IndexBuildProgress::new("resolving_edges", 0, all_raw.len()));

    let owned_by_changed = |e: &ExtractedEdge| changed_files.contains(source_file_of(e));
    let names_changed_def = |e: &ExtractedEdge| edge_references_name(e, &changed_def_names);
    let is_candidate = |e: &ExtractedEdge| owned_by_changed(e) || names_changed_def(e);

    // Instrumentation: how many raw edges this cheap path actually resolves.
    note_reresolved(all_raw.iter().filter(|e| is_candidate(e)).count());

    // PASS 1 — IMPORTS. Resolve all import signals so grouped reexports and
    // parent-module globs can disambiguate candidate references exactly as in
    // the full resolver. Only candidate IMPORTS edges are persisted again.
    let mut resolved: Vec<NewEdge> = Vec::new();
    for edge in all_raw.iter().filter(|e| e.edge_type == "IMPORTS") {
        examined += 1;
        progress(IndexBuildProgress::new(
            "resolving_edges",
            examined,
            all_raw.len(),
        ));
        let Some(src) = index.by_qname(&edge.source_qualified_name) else {
            continue;
        };
        let src_id = src.id;
        let src_file = src.file_path.clone();
        let is_cand_import = is_candidate(edge);
        index.record_import_items(edge, &src_file);
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
                index.resolve_import_target(
                    &src_file,
                    name,
                    path,
                    edge.properties.get("imported_items"),
                )
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
        examined += 1;
        progress(IndexBuildProgress::new(
            "resolving_edges",
            examined,
            all_raw.len(),
        ));
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
            "USAGE" => index.resolve_usage_target(edge, src_id),
            _ => index.by_qname(&edge.target_qualified_name).map(|n| n.id),
        };
        let Some(target_id) = target_id else {
            clear_option_field_unresolved();
            continue;
        };
        if target_id == src_id && edge.edge_type != "CALLS" {
            clear_option_field_unresolved();
            continue;
        }
        resolved.push(new_edge(project, src_id, target_id, edge));
        push_rust_enum_owner_usage(&index, project, src_id, target_id, edge, &mut resolved);
    }

    progress(IndexBuildProgress::new("writing_resolved_edges", 0, 1));
    insert_edges_batched(store, &resolved)?;
    progress(IndexBuildProgress::new("writing_resolved_edges", 1, 1));
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

/// Reasons for one Option-field `CALLS` edge. The slot is keyed by that edge
/// and cleared when resolution is skipped, fails, or persists a different
/// edge. A later `CALLS` row must not inherit it.
struct OptionFieldUnresolved {
    source_qualified_name: String,
    file_path: String,
    line: u32,
    callee_name: String,
    reasons: Vec<String>,
}

fn option_edge_callee_name(edge: &ExtractedEdge) -> &str {
    edge.properties
        .get("callee_name")
        .and_then(|value| value.as_str())
        .unwrap_or("")
}

fn option_unresolved_matches(pending: &OptionFieldUnresolved, edge: &ExtractedEdge) -> bool {
    pending.source_qualified_name == edge.source_qualified_name
        && pending.file_path == edge.file_path
        && pending.line == edge.line
        && pending.callee_name == option_edge_callee_name(edge)
}

std::thread_local! {
    static OPTION_FIELD_UNRESOLVED_REASONS: std::cell::RefCell<Option<OptionFieldUnresolved>> =
        const { std::cell::RefCell::new(None) };
}

fn clear_option_field_unresolved() {
    OPTION_FIELD_UNRESOLVED_REASONS.with(|slot| *slot.borrow_mut() = None);
}

fn set_option_field_unresolved(edge: &ExtractedEdge, reasons: Vec<String>) {
    let pending = OptionFieldUnresolved {
        source_qualified_name: edge.source_qualified_name.clone(),
        file_path: edge.file_path.clone(),
        line: edge.line,
        callee_name: option_edge_callee_name(edge).to_string(),
        reasons,
    };
    OPTION_FIELD_UNRESOLVED_REASONS.with(|slot| *slot.borrow_mut() = Some(pending));
}

fn take_option_field_unresolved_for(edge: &ExtractedEdge) -> Option<Vec<String>> {
    OPTION_FIELD_UNRESOLVED_REASONS.with(|slot| {
        let mut pending = slot.borrow_mut();
        if pending
            .as_ref()
            .is_some_and(|item| option_unresolved_matches(item, edge))
        {
            pending.take().map(|item| item.reasons)
        } else {
            *pending = None;
            None
        }
    })
}

/// An exact resolved variant also proves a dependency on its owning enum.
fn push_rust_enum_owner_usage(
    index: &GraphIndex,
    project: &str,
    source_id: i64,
    target_id: i64,
    edge: &ExtractedEdge,
    resolved: &mut Vec<NewEdge>,
) {
    if !edge.file_path.ends_with(".rs") || !matches!(edge.edge_type.as_str(), "USAGE" | "CALLS") {
        return;
    }
    // Only an already-resolved exact variant proves an enum dependency. Do
    // not guess a type from an arbitrary module/associated-member prefix.
    let Some(qname) = index.qname_for_id(target_id) else {
        return;
    };
    let Some(variant) = index
        .by_qname(qname)
        .filter(|node| node.label == "EnumVariant")
    else {
        return;
    };
    let Some((prefix, _)) = qname.rsplit_once("::") else {
        return;
    };
    let Some((file, owner_name)) = prefix.rsplit_once("::") else {
        return;
    };
    if file != variant.file_path {
        return;
    }
    let Some(owner) = index
        .by_qname(&format!("{file}::Enum::{owner_name}"))
        .filter(|node| node.label == "Enum")
    else {
        return;
    };
    if owner.id == source_id {
        return;
    }
    resolved.push(NewEdge {
        project: project.to_owned(), source_id, target_id: owner.id,
        edge_type: "USAGE".into(),
        properties: serde_json::json!({ "line": edge.line, "rust_enum_variant_owner": true, "variant": qname }),
    });
}

/// Build a [`NewEdge`] from a resolved source/target pair, cloning the
/// parser's edge properties. A typed Option field whose `as_ref`/`Some`
/// identity was not proven is stored as `UNRESOLVED_CALLS` instead of a
/// confirmed caller. Ambiguous receivers never reach this override.
fn new_edge(project: &str, source_id: i64, target_id: i64, edge: &ExtractedEdge) -> NewEdge {
    let mut edge_type = persisted_edge_label(edge).to_string();
    let mut props = edge.properties.clone();
    if edge_type == "CALLS" {
        if let Some(reasons) = take_option_field_unresolved_for(edge) {
            edge_type = "UNRESOLVED_CALLS".to_string();
            if let Some(map) = props.as_object_mut() {
                map.insert("unresolved_reasons".into(), serde_json::json!(reasons));
            }
        }
    } else {
        // An import or other edge persisted out of order must not leave a
        // pending Option reason for the next CALLS row.
        clear_option_field_unresolved();
    }
    // Fold the reference-site line into the resolved edge too (P4):
    // nav commands print it grep-shaped so one answer carries the
    // call-site evidence. Raw edges round-trip it via properties, so
    // `edge.line` is populated on both extract and re-resolve paths.
    if edge.line > 0 {
        if let Some(map) = props.as_object_mut() {
            map.insert("line".into(), serde_json::json!(edge.line));
        }
    }
    NewEdge {
        project: project.to_string(),
        source_id,
        target_id,
        edge_type,
        properties: props,
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
            .resolve_import_target(
                &edge.file_path,
                name,
                path,
                edge.properties.get("imported_items"),
            )
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
                                && greppy_resolver::IMPORTABLE_LABELS.contains(&node.label.as_str())
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
    persist_edges_batched(store, edges, None)
}

fn persist_edges_batched(
    store: &mut Store,
    edges: &[NewEdge],
    replace_rust_project: Option<&str>,
) -> Result<()> {
    if edges.is_empty() && replace_rust_project.is_none() {
        return Ok(());
    }
    if store.is_overlay() {
        let mut qualified_names: std::collections::HashMap<(String, i64), String> =
            std::collections::HashMap::new();
        let mut overlay_edges = Vec::with_capacity(edges.len());
        {
            let conn = store.conn();
            let mut lookup = conn
                .prepare_cached("SELECT qualified_name FROM nodes WHERE id = ?1 AND project = ?2")
                .map_err(sqlite_err)?;
            for edge in edges {
                let source_key = (edge.project.clone(), edge.source_id);
                let source_qualified_name = match qualified_names.get(&source_key) {
                    Some(name) => name.clone(),
                    None => {
                        let name: String = lookup
                            .query_row(rusqlite::params![edge.source_id, edge.project], |row| {
                                row.get(0)
                            })
                            .map_err(sqlite_err)?;
                        qualified_names.insert(source_key, name.clone());
                        name
                    }
                };
                let target_key = (edge.project.clone(), edge.target_id);
                let target_qualified_name = match qualified_names.get(&target_key) {
                    Some(name) => name.clone(),
                    None => {
                        let name: String = lookup
                            .query_row(rusqlite::params![edge.target_id, edge.project], |row| {
                                row.get(0)
                            })
                            .map_err(sqlite_err)?;
                        qualified_names.insert(target_key, name.clone());
                        name
                    }
                };
                overlay_edges.push(NewOverlayEdge {
                    project: edge.project.clone(),
                    source_qualified_name,
                    target_qualified_name,
                    edge_type: edge.edge_type.clone(),
                    properties: edge.properties.clone(),
                });
            }
        }
        store.insert_overlay_edges(&overlay_edges)?;
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
        if let Some(project) = replace_rust_project {
            conn.execute(
                "DELETE FROM main.edges WHERE project = ?1
                 AND edge_type NOT IN ('CONTAINS_FOLDER', 'CONTAINS_FILE', 'DEFINES')
                 AND source_id IN (SELECT id FROM nodes WHERE project = ?1 AND file_path LIKE '%.rs')",
                [project],
            ).map_err(sqlite_err)?;
        }
        let mut stmt = conn
            .prepare_cached(
                "INSERT INTO main.edges (project, source_id, target_id, edge_type, properties)
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
        if replace_rust_project.is_some() {
            mark_rust_caller_edges_repaired(store)?;
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
    declared_type: Option<String>,
}

/// Rank graph labels exactly like CLI symbol navigation. When one source symbol
/// has multiple persisted facets (for example Scala Method + Function twins),
/// CALLS resolution and single-node navigation must choose the same facet or a
/// real edge can appear unreachable to `path`.
fn navigation_label_rank(label: &str) -> u8 {
    match label {
        "Class" | "Interface" | "Type" | "Struct" | "Enum" | "Trait" | "Function" | "Method"
        | "TypeAlias" => 0,
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
    by_id: std::collections::HashMap<i64, NodeLite>,
    /// `name → nodes sharing that name`, each inner vec sorted by
    /// `qualified_name` so the candidate order matches the old
    /// `list_nodes_by_name` ordering (resolution depends only on the
    /// set + same-file count, but we keep order stable for determinism).
    by_name: std::collections::HashMap<String, Vec<NodeLite>>,
    /// `file_path → resolved IMPORTS target ids` for this file. Filled by
    /// [`record_import`](GraphIndex::record_import) during the IMPORTS
    /// pass; consulted by [`resolve_unique_with_imports`].
    imports_by_file: std::collections::HashMap<String, std::collections::HashSet<i64>>,
    import_aliases_by_file: std::collections::HashMap<
        String,
        std::collections::HashMap<String, std::collections::HashSet<i64>>,
    >,
    import_alias_sources_by_file:
        std::collections::HashMap<String, std::collections::HashMap<String, Vec<(String, String)>>>,
    import_module_files_by_file:
        std::collections::HashMap<String, std::collections::HashSet<String>>,
    import_globs_by_file: std::collections::HashMap<String, Vec<String>>,
    /// Rust namespace aliases imported into a file (`channels` in
    /// `use crate::core::mission::channels`). Values are exact candidate
    /// module files, never global basename matches.
    rust_namespaces_by_file:
        std::collections::HashMap<String, std::collections::HashMap<String, Vec<String>>>,
    /// Declared Cargo target files for this project. `None` means no readable
    /// Cargo manifest was available. Uncovered files retain the conventional
    /// source-layout fallback because target discovery is intentionally bounded.
    rust_crate_roots: Option<std::collections::HashSet<String>>,
    anyhow_factory_files: std::collections::HashSet<String>,
    anyhow_dependency_binding: String,
    anyhow_glob_proofs: std::collections::HashSet<String>,
    direct_field_trait_owners: std::collections::HashSet<String>,
    rust_libraries: Vec<RustPackage>,
    /// `node id → file_path`, so a referrer's file (needed for the
    /// same-file preference) is an O(1) lookup from its id.
    id_to_file: std::collections::HashMap<i64, String>,
    /// `node id → qualified_name`, used to persist cross-database Delta
    /// edges by logical identity rather than connection-local ids.
    id_to_qname: std::collections::HashMap<i64, String>,
    /// `file basename stem → File node ids`. Backs the `require`/`import`→File
    /// IMPORTS pass (Ruby `require 'record'`, Clojure `(:require ..)`, Elm/
    /// Erlang/Zig/Dart module imports). Populated at load; only usable AFTER
    /// the structural pass has created the File nodes.
    files_by_stem: std::collections::HashMap<String, Vec<i64>>,
    known_files: std::collections::HashSet<String>,
    /// Rust defs that live in an inline `mod` (no `mod.rs` / `name.rs`).
    /// Keyed by node id. Absent for file-module items.
    rust_inline_scopes: std::collections::HashMap<i64, RustInlineScope>,
    /// `file::Trait::as_ref` → the method's self parameter text. Used only to
    /// reject a by-value adapter that can hide Option::as_ref.
    as_ref_receivers: std::collections::HashMap<String, String>,
    open_traits: std::collections::HashSet<i64>,
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

/// Languages that share an import and call namespace. A same-named definition
/// in any other language must not make an in-language match ambiguous.
fn resolution_language_family(path: &str) -> &'static str {
    match greppy_parser::language_for_path(Path::new(path)) {
        Language::JavaScript | Language::TypeScript { .. } => "javascript",
        Language::C | Language::Cpp => "c",
        other => other.name(),
    }
}

fn rust_module_files_for_module_path_with_crate_roots(
    referrer_file: &str,
    module_path: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> Vec<String> {
    // Cargo permits a binary/library target outside the conventional
    // `src/{main,lib}.rs` location. When Cargo target metadata is available,
    // choose the nearest ancestor of one of those declared target files for
    // `crate::` paths. This keeps module resolution lexical without treating
    // an arbitrary nested `main.rs`/`lib.rs` as a crate root.
    let referrer = Path::new(referrer_file);
    let parent = referrer.parent().unwrap_or_else(|| Path::new(""));
    let is_module_root = referrer
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "lib.rs" | "main.rs" | "mod.rs"))
        || crate_roots.is_some_and(|roots| roots.contains(referrer_file));
    let mut base = if is_module_root {
        parent.to_path_buf()
    } else {
        referrer.with_extension("")
    };
    for (position, module) in module_path
        .split("::")
        .filter(|part| !part.is_empty())
        .enumerate()
    {
        match module {
            "crate" if position == 0 => {
                base = crate_roots
                    .and_then(|roots| rust_crate_root_for_file(referrer_file, roots))
                    .or_else(|| {
                        // Manifest discovery is intentionally bounded. Preserve
                        // conventional crate resolution for uncovered members.
                        let root_end = referrer_file
                            .rfind("/src/")
                            .map(|offset| offset + "/src".len())
                            .or_else(|| referrer_file.starts_with("src/").then_some("src".len()));
                        root_end.map(|end| std::path::PathBuf::from(&referrer_file[..end]))
                    })
                    .unwrap_or_else(|| {
                        referrer
                            .parent()
                            .unwrap_or_else(|| Path::new(""))
                            .to_path_buf()
                    });
            }
            "self" if position == 0 => {}
            "super" => {
                base.pop();
            }
            part => base.push(part),
        }
    }
    let flat = base
        .with_extension("rs")
        .to_string_lossy()
        .replace('\\', "/");
    let nested = base.join("mod.rs").to_string_lossy().replace('\\', "/");
    let lib = base.join("lib.rs").to_string_lossy().replace('\\', "/");
    let main = base.join("main.rs").to_string_lossy().replace('\\', "/");
    vec![flat, nested, lib, main]
}

/// A Rust definition inside an inline `mod` item.
///
/// `module` is the `mod` chain (`trace`, `a::b`). `module_vis` is parallel to
/// those segments; an empty string means the module is private. `visibility`
/// is the item's own visibility text (`None` when the item is private).
struct RustInlineScope {
    module: String,
    visibility: Option<String>,
    module_vis: Vec<String>,
}

struct RustModuleSite {
    dir: std::path::PathBuf,
    file: String,
    inline: String,
}

fn slash_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn rust_file_module_dir(
    file: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> std::path::PathBuf {
    let path = std::path::Path::new(file);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if matches!(name, "lib.rs" | "main.rs" | "mod.rs")
        || crate_roots.is_some_and(|roots| roots.contains(file))
    {
        path.parent()
            .unwrap_or_else(|| std::path::Path::new(""))
            .to_path_buf()
    } else {
        path.with_extension("")
    }
}

fn rust_module_site_for_file(
    file: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> RustModuleSite {
    RustModuleSite {
        dir: rust_file_module_dir(file, crate_roots),
        file: file.to_string(),
        inline: String::new(),
    }
}

fn rust_crate_directory(
    referrer_file: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> std::path::PathBuf {
    crate_roots
        .and_then(|roots| rust_crate_root_for_file(referrer_file, roots))
        .or_else(|| {
            let root_end = referrer_file
                .rfind("/src/")
                .map(|offset| offset + "/src".len())
                .or_else(|| referrer_file.starts_with("src/").then_some("src".len()));
            root_end.map(|end| std::path::PathBuf::from(&referrer_file[..end]))
        })
        .unwrap_or_else(|| {
            std::path::Path::new(referrer_file)
                .parent()
                .unwrap_or_else(|| std::path::Path::new(""))
                .to_path_buf()
        })
}

fn rust_crate_root_files(
    dir: &std::path::Path,
    known_files: &std::collections::HashSet<String>,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> Vec<String> {
    let mut hits = Vec::new();
    for name in ["lib.rs", "main.rs", "mod.rs"] {
        let candidate = slash_path(&dir.join(name));
        if known_files.contains(&candidate) {
            hits.push(candidate);
        }
    }
    if let Some(roots) = crate_roots {
        for root in roots {
            if std::path::Path::new(root).parent() == Some(dir)
                && known_files.contains(root)
                && !hits.contains(root)
            {
                hits.push(root.clone());
            }
        }
    }
    hits
}

fn rust_child_module_site(
    dir: &std::path::Path,
    segment: &str,
    known_files: &std::collections::HashSet<String>,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> Option<RustModuleSite> {
    let child_dir = dir.join(segment);
    let candidates = [
        slash_path(&dir.join(format!("{segment}.rs"))),
        slash_path(&child_dir.join("mod.rs")),
        slash_path(&child_dir.join("lib.rs")),
        slash_path(&child_dir.join("main.rs")),
    ];
    let hits = candidates
        .into_iter()
        .filter(|candidate| known_files.contains(candidate))
        .collect::<Vec<_>>();
    match hits.as_slice() {
        [file] => Some(rust_module_site_for_file(file, crate_roots)),
        _ => None,
    }
}

fn rust_parent_module_site(
    file: &str,
    known_files: &std::collections::HashSet<String>,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> Option<RustModuleSite> {
    let path = std::path::Path::new(file);
    let parent_dir = path.parent()?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let container = if matches!(name, "lib.rs" | "main.rs" | "mod.rs")
        || crate_roots.is_some_and(|roots| roots.contains(file))
    {
        parent_dir.parent()?.to_path_buf()
    } else {
        parent_dir.to_path_buf()
    };
    let stem = container
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if stem.is_empty() {
        return None;
    }
    let parent_of_container = container
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""));
    let mut hits = Vec::new();
    for candidate in [
        slash_path(&container.join("mod.rs")),
        slash_path(&parent_of_container.join(format!("{stem}.rs"))),
        slash_path(&container.join("lib.rs")),
        slash_path(&container.join("main.rs")),
    ] {
        if candidate != file && known_files.contains(&candidate) && !hits.contains(&candidate) {
            hits.push(candidate);
        }
    }
    match hits.as_slice() {
        [one] => Some(rust_module_site_for_file(one, crate_roots)),
        _ => None,
    }
}

/// Where an inline module path lives when no file module covers the whole path.
///
/// `crate::trace` in a crate whose only root is `src/lib.rs` yields
/// `("src/lib.rs", "trace")`. A path that lands entirely on files yields nothing.
fn rust_inline_module_sites(
    referrer_file: &str,
    module_path: &str,
    known_files: &std::collections::HashSet<String>,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> Vec<(String, String)> {
    let segments = module_path
        .split("::")
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return Vec::new();
    }
    let mut index = 0;
    let mut states = Vec::new();
    match segments[0] {
        "crate" => {
            let dir = rust_crate_directory(referrer_file, crate_roots);
            for file in rust_crate_root_files(&dir, known_files, crate_roots) {
                states.push(rust_module_site_for_file(&file, crate_roots));
            }
            index = 1;
        }
        "self" => {
            states.push(rust_module_site_for_file(referrer_file, crate_roots));
            index = 1;
        }
        "super" => {
            if let Some(parent) = rust_parent_module_site(referrer_file, known_files, crate_roots) {
                states.push(parent);
            }
            index = 1;
        }
        _ => states.push(rust_module_site_for_file(referrer_file, crate_roots)),
    }
    while index < segments.len() {
        let segment = segments[index];
        index += 1;
        if states.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for state in states {
            if segment == "super" && state.inline.is_empty() {
                if let Some(parent) = rust_parent_module_site(&state.file, known_files, crate_roots)
                {
                    next.push(parent);
                }
                continue;
            }
            if state.inline.is_empty() {
                if let Some(child) =
                    rust_child_module_site(&state.dir, segment, known_files, crate_roots)
                {
                    next.push(child);
                    continue;
                }
            }
            let inline = if state.inline.is_empty() {
                segment.to_string()
            } else {
                format!("{}::{segment}", state.inline)
            };
            next.push(RustModuleSite { inline, ..state });
        }
        states = next;
    }
    states
        .into_iter()
        .filter(|state| !state.inline.is_empty())
        .map(|state| (state.file, state.inline))
        .collect()
}

fn rust_inline_sites_below_alias(
    alias_files: &[String],
    ref_path: &str,
    name: &str,
    known_files: &std::collections::HashSet<String>,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> Vec<(String, String)> {
    let segments = ref_path
        .split("::")
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if segments.last().copied() != Some(name) || segments.len() < 2 {
        return Vec::new();
    }
    let mut states = alias_files
        .iter()
        .map(|file| rust_module_site_for_file(file, crate_roots))
        .collect::<Vec<_>>();
    for segment in &segments[1..segments.len() - 1] {
        let mut next = Vec::new();
        for state in states {
            if state.inline.is_empty() {
                if let Some(child) =
                    rust_child_module_site(&state.dir, segment, known_files, crate_roots)
                {
                    next.push(child);
                    continue;
                }
            }
            let inline = if state.inline.is_empty() {
                (*segment).to_string()
            } else {
                format!("{}::{segment}", state.inline)
            };
            next.push(RustModuleSite { inline, ..state });
        }
        states = next;
    }
    states
        .into_iter()
        .filter(|state| !state.inline.is_empty())
        .map(|state| (state.file, state.inline))
        .collect()
}

fn compact_visibility(visibility: &str) -> String {
    visibility
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect()
}

fn caller_in_rust_module(
    module: &str,
    owner_file: &str,
    caller_module: &str,
    caller_file: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> bool {
    if module.is_empty() {
        if caller_file == owner_file {
            return true;
        }
        let dir = slash_path(&rust_file_module_dir(owner_file, crate_roots));
        return !dir.is_empty() && caller_file.starts_with(&format!("{dir}/"));
    }
    if caller_file == owner_file {
        return caller_module == module || caller_module.starts_with(&format!("{module}::"));
    }
    let mut dir = rust_file_module_dir(owner_file, crate_roots);
    for segment in module.split("::") {
        dir.push(segment);
    }
    let prefix = format!("{}/", slash_path(&dir));
    caller_file.starts_with(&prefix)
}

fn rust_module_item_visible(
    visibility: &str,
    module_path: &str,
    owner_file: &str,
    caller_module: &str,
    caller_file: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> bool {
    let vis = compact_visibility(visibility);
    if vis == "pub"
        || vis == "pub(crate)"
        || vis.starts_with("pub(crate::")
        || vis.starts_with("pub(in crate")
        || vis.starts_with("pub(incrate")
    {
        return true;
    }
    if vis.is_empty() || vis == "pub(self)" || vis == "pub(in self)" || vis == "pub(inself)" {
        let parent = module_path
            .rsplit_once("::")
            .map(|(parent, _)| parent)
            .unwrap_or("");
        return caller_in_rust_module(parent, owner_file, caller_module, caller_file, crate_roots);
    }
    if vis == "pub(super)" || vis == "pub(in super)" || vis == "pub(insuper)" {
        let Some((containing, _)) = module_path.rsplit_once("::") else {
            return false;
        };
        let grandparent = containing
            .rsplit_once("::")
            .map(|(parent, _)| parent)
            .unwrap_or("");
        return caller_in_rust_module(
            grandparent,
            owner_file,
            caller_module,
            caller_file,
            crate_roots,
        );
    }
    false
}

fn rust_path_modules_visible(
    module: &str,
    module_vis: &[String],
    owner_file: &str,
    caller_module: &str,
    caller_file: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> bool {
    let segments = module.split("::").collect::<Vec<_>>();
    for (index, _) in segments.iter().enumerate() {
        let visibility = module_vis.get(index).map(String::as_str).unwrap_or("");
        let module_path = segments[..=index].join("::");
        if !rust_module_item_visible(
            visibility,
            &module_path,
            owner_file,
            caller_module,
            caller_file,
            crate_roots,
        ) {
            return false;
        }
    }
    true
}

fn rust_item_visible(
    visibility: Option<&str>,
    module: &str,
    owner_file: &str,
    caller_module: &str,
    caller_file: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> bool {
    let vis = compact_visibility(visibility.unwrap_or(""));
    if vis == "pub"
        || vis == "pub(crate)"
        || vis.starts_with("pub(crate::")
        || vis.starts_with("pub(in crate")
        || vis.starts_with("pub(incrate")
    {
        return true;
    }
    if vis.is_empty() || vis == "pub(self)" || vis == "pub(in self)" || vis == "pub(inself)" {
        return caller_in_rust_module(module, owner_file, caller_module, caller_file, crate_roots);
    }
    if vis == "pub(super)" || vis == "pub(in super)" || vis == "pub(insuper)" {
        let parent = module
            .rsplit_once("::")
            .map(|(parent, _)| parent)
            .unwrap_or("");
        return caller_in_rust_module(parent, owner_file, caller_module, caller_file, crate_roots);
    }
    false
}

fn rust_crate_root_for_file(
    referrer_file: &str,
    crate_roots: &std::collections::HashSet<String>,
) -> Option<std::path::PathBuf> {
    let mut directory = Path::new(referrer_file).parent()?.to_path_buf();
    loop {
        if crate_roots
            .iter()
            .any(|root| Path::new(root).parent() == Some(directory.as_path()))
        {
            return Some(directory);
        }
        if !directory.pop() {
            return None;
        }
    }
}

/// The owning package's library is an implicit extern crate. Another workspace
/// member is an extern crate only when this package's manifest names it with a
/// path dependency or a `workspace = true` dependency whose workspace entry has
/// a path. A matching name elsewhere, including a crates.io version requirement,
/// is not evidence of a dependency.
#[derive(Debug)]
struct RustPackage {
    package_dir: String,
    library: Option<RustLibrary>,
    /// `(extern crate name, that dependency's library root file)`.
    extern_crates: Vec<(String, String)>,
}

#[derive(Debug)]
struct RustLibrary {
    name: String,
    root_file: String,
}

fn rust_crate_roots_for_project(
    store: &Store,
    project: &str,
    known_files: &std::collections::HashSet<String>,
) -> Option<(std::collections::HashSet<String>, Vec<RustPackage>)> {
    let project = store.get_project(project).ok().flatten()?;
    let root = std::fs::canonicalize(project.root_path).ok()?;
    let mut roots = std::collections::HashSet::new();
    let mut libraries = Vec::new();
    let mut visited = std::collections::HashSet::new();
    if !rust_crate_roots_from_manifest(
        &root.join("Cargo.toml"),
        &root,
        known_files,
        &mut roots,
        &mut libraries,
        &mut visited,
    ) {
        return None;
    }
    attach_rust_extern_crates(&root, &mut libraries);
    Some((roots, libraries))
}

fn rust_crate_roots_from_manifest(
    manifest_path: &std::path::Path,
    repository_root: &std::path::Path,
    known_files: &std::collections::HashSet<String>,
    roots: &mut std::collections::HashSet<String>,
    libraries: &mut Vec<RustPackage>,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
) -> bool {
    let manifest_path = std::fs::canonicalize(manifest_path)
        .ok()
        .unwrap_or_else(|| manifest_path.to_path_buf());
    if !visited.insert(manifest_path.clone()) {
        return true;
    }
    let Ok(text) = std::fs::read_to_string(&manifest_path) else {
        return false;
    };
    let Ok(document) = text.parse::<toml_edit::DocumentMut>() else {
        return false;
    };
    let manifest_root = manifest_path.parent().unwrap_or(repository_root);
    if let Some(members) = document
        .get("workspace")
        .and_then(|item| item.as_table())
        .and_then(|workspace| workspace.get("members"))
        .and_then(|item| item.as_value())
        .and_then(|value| value.as_array())
    {
        for member in members.iter().filter_map(|value| value.as_str()) {
            for member_manifest in workspace_member_manifests(manifest_root, member) {
                let _ = rust_crate_roots_from_manifest(
                    &member_manifest,
                    repository_root,
                    known_files,
                    roots,
                    libraries,
                    visited,
                );
            }
        }
    }
    let Some(package) = document.get("package").and_then(|item| item.as_table()) else {
        return true;
    };

    let relative_path = |path: &std::path::Path| -> Option<String> {
        // Cargo accepts `./` and parent components in explicit target paths.
        // Canonicalize existing targets before comparing with indexed files;
        // targets escaping the repository still fail the prefix check.
        let canonical = std::fs::canonicalize(path).ok();
        let relative = canonical
            .as_deref()
            .unwrap_or(path)
            .strip_prefix(repository_root)
            .ok()?;
        Some(relative.to_string_lossy().replace('\\', "/"))
    };
    let add_target = |target: &str, roots: &mut std::collections::HashSet<String>| {
        let path = manifest_root.join(target);
        if let Some(relative) = relative_path(&path) {
            roots.insert(relative);
        }
    };
    let known_target = |target: &str| {
        relative_path(&manifest_root.join(target))
            .is_some_and(|relative| known_files.contains(&relative))
    };
    let autolib = package
        .get("autolib")
        .and_then(|item| item.as_value())
        .and_then(|value| value.as_bool())
        != Some(false);
    let autobins = package
        .get("autobins")
        .and_then(|item| item.as_value())
        .and_then(|value| value.as_bool())
        != Some(false);

    let lib = document.get("lib").and_then(|item| item.as_table());
    if let Some(path) = lib
        .and_then(|table| table.get("path"))
        .and_then(|item| item.as_value())
        .and_then(|value| value.as_str())
    {
        add_target(path, roots);
    } else if (autolib || lib.is_some()) && known_target("src/lib.rs") {
        add_target("src/lib.rs", roots);
    }

    let library_path = lib
        .and_then(|table| table.get("path"))
        .and_then(|item| item.as_str())
        .or_else(|| {
            ((autolib || lib.is_some()) && known_target("src/lib.rs")).then_some("src/lib.rs")
        });
    let library_name = lib
        .and_then(|table| table.get("name"))
        .and_then(|item| item.as_str())
        .or_else(|| package.get("name").and_then(|item| item.as_str()));
    let mut library = None;
    if let (Some(path), Some(name)) = (library_path, library_name) {
        if let Some(root_file) = relative_path(&manifest_root.join(path)) {
            if known_files.contains(&root_file) {
                library = Some(RustLibrary {
                    name: name.replace('-', "_"),
                    root_file,
                });
            }
        }
    }

    if let Some(package_dir) = relative_path(manifest_root) {
        libraries.push(RustPackage {
            package_dir,
            library,
            extern_crates: Vec::new(),
        });
    }

    if let Some(bins) = document
        .get("bin")
        .and_then(|item| item.as_array_of_tables())
    {
        for bin in bins {
            if let Some(path) = bin
                .get("path")
                .and_then(|item| item.as_value())
                .and_then(|value| value.as_str())
            {
                add_target(path, roots);
            } else if let Some(name) = bin
                .get("name")
                .and_then(|item| item.as_value())
                .and_then(|value| value.as_str())
            {
                let inferred = format!("src/bin/{name}.rs");
                if known_target(&inferred) {
                    add_target(&inferred, roots);
                }
            }
        }
    }
    if autobins {
        if known_target("src/main.rs") {
            add_target("src/main.rs", roots);
        }
        let manifest_prefix = relative_path(manifest_root).unwrap_or_default();
        let bin_prefix = if manifest_prefix.is_empty() {
            "src/bin/".to_string()
        } else {
            format!("{manifest_prefix}/src/bin/")
        };
        for path in known_files {
            let Some(rest) = path.strip_prefix(&bin_prefix) else {
                continue;
            };
            let direct = !rest.contains('/') && rest.ends_with(".rs");
            let nested = rest.matches('/').count() == 1 && rest.ends_with("/main.rs");
            if direct || nested {
                let target = format!("src/bin/{rest}");
                add_target(&target, roots);
            }
        }
    }
    true
}

fn workspace_member_manifests(
    workspace_root: &std::path::Path,
    member: &str,
) -> Vec<std::path::PathBuf> {
    // This is deliberately a narrow manifest reader: explicit members and a
    // single standalone `*` component are enough for the source layouts that
    // need crate-root resolution here. Cargo's full glob semantics and
    // `workspace.exclude` are not reproduced by this helper.
    let pattern = std::path::Path::new(member);
    let Some((wildcard_index, _)) = pattern
        .components()
        .enumerate()
        .find(|(_, component)| component.as_os_str().to_string_lossy() == "*")
    else {
        return vec![workspace_root.join(pattern).join("Cargo.toml")];
    };
    let components = pattern.components().collect::<Vec<_>>();
    let mut prefix = std::path::PathBuf::new();
    for component in &components[..wildcard_index] {
        prefix.push(component.as_os_str());
    }
    let mut suffix = std::path::PathBuf::new();
    for component in &components[wildcard_index + 1..] {
        suffix.push(component.as_os_str());
    }
    let Ok(entries) = std::fs::read_dir(workspace_root.join(prefix)) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|kind| kind.is_dir())
                .map(|_| entry.path())
        })
        .map(|path| path.join(&suffix).join("Cargo.toml"))
        .collect()
}

fn relative_repo_dir(repository_root: &std::path::Path, path: &std::path::Path) -> Option<String> {
    let canonical = std::fs::canonicalize(path).ok()?;
    let relative = canonical.strip_prefix(repository_root).ok()?;
    Some(relative.to_string_lossy().replace('\\', "/"))
}

fn toml_dep_path(item: &toml_edit::Item) -> Option<&str> {
    item.as_table_like()
        .and_then(|table| table.get("path"))
        .and_then(|value| value.as_str())
        .filter(|path| !path.is_empty())
}

fn toml_dep_is_optional(item: &toml_edit::Item) -> bool {
    item.as_table_like()
        .and_then(|table| table.get("optional"))
        .and_then(|value| value.as_bool())
        == Some(true)
}

fn toml_dep_uses_workspace(item: &toml_edit::Item) -> bool {
    item.as_table_like()
        .and_then(|table| table.get("workspace"))
        .and_then(|value| value.as_bool())
        == Some(true)
}

/// The Rust 2018 extern name. A `package = "..."` rename uses the dependency
/// key; otherwise the dependency's library name already includes `[lib] name`.
fn rust_extern_crate_name(key: &str, item: &toml_edit::Item, library_name: &str) -> String {
    let key_name = key.replace('-', "_");
    let renamed = item
        .as_table_like()
        .and_then(|table| table.get("package"))
        .and_then(|value| value.as_str())
        .is_some_and(|package| package.replace('-', "_") != key_name);
    if renamed {
        key_name
    } else {
        library_name.to_string()
    }
}

fn workspace_path_dep_dirs(
    repository_root: &std::path::Path,
) -> std::collections::HashMap<String, String> {
    let mut dirs = std::collections::HashMap::new();
    let Ok(text) = std::fs::read_to_string(repository_root.join("Cargo.toml")) else {
        return dirs;
    };
    let Ok(document) = text.parse::<toml_edit::DocumentMut>() else {
        return dirs;
    };
    let Some(dependencies) = document
        .get("workspace")
        .and_then(|item| item.as_table())
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(|item| item.as_table_like())
    else {
        return dirs;
    };
    for (key, item) in dependencies.iter() {
        let Some(path) = toml_dep_path(item) else {
            continue;
        };
        let Some(relative) = relative_repo_dir(repository_root, &repository_root.join(path)) else {
            continue;
        };
        dirs.insert(key.to_string(), relative);
    }
    dirs
}

/// Link each package to library roots it actually depends on. Target-specific
/// and optional dependencies are skipped: cfg and feature selection are not
/// known here, and a wrong crate is worse than a missing edge.
fn attach_rust_extern_crates(repository_root: &std::path::Path, libraries: &mut [RustPackage]) {
    let workspace_deps = workspace_path_dep_dirs(repository_root);
    let mut attached = Vec::with_capacity(libraries.len());
    for package in libraries.iter() {
        let mut extern_crates = Vec::new();
        let manifest = repository_root
            .join(&package.package_dir)
            .join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            attached.push(extern_crates);
            continue;
        };
        let Ok(document) = text.parse::<toml_edit::DocumentMut>() else {
            attached.push(extern_crates);
            continue;
        };
        let manifest_dir = manifest.parent().unwrap_or(repository_root);
        for table_name in ["dependencies", "dev-dependencies"] {
            let Some(table) = document
                .get(table_name)
                .and_then(|item| item.as_table_like())
            else {
                continue;
            };
            for (key, item) in table.iter() {
                if toml_dep_is_optional(item) {
                    continue;
                }
                let dep_dir = if let Some(path) = toml_dep_path(item) {
                    relative_repo_dir(repository_root, &manifest_dir.join(path))
                } else if toml_dep_uses_workspace(item) {
                    workspace_deps.get(key).cloned()
                } else {
                    None
                };
                let Some(dep_dir) = dep_dir else {
                    continue;
                };
                if dep_dir == package.package_dir {
                    continue;
                }
                let mut roots = libraries
                    .iter()
                    .filter(|candidate| candidate.package_dir == dep_dir)
                    .filter_map(|candidate| candidate.library.as_ref())
                    .collect::<Vec<_>>();
                roots.sort_by_key(|library| library.root_file.as_str());
                roots.dedup_by(|left, right| left.root_file == right.root_file);
                let [library] = roots.as_slice() else {
                    continue;
                };
                let extern_name = rust_extern_crate_name(key, item, &library.name);
                if extern_name.is_empty() {
                    continue;
                }
                extern_crates.push((extern_name, library.root_file.clone()));
            }
        }
        attached.push(extern_crates);
    }
    for (package, extern_crates) in libraries.iter_mut().zip(attached) {
        package.extern_crates = extern_crates;
    }
}

enum ExternModuleFiles {
    Miss,
    Ambiguous,
    Hit(Vec<String>),
}

fn rust_library_module_files(root_file: &str, rest: &[&str]) -> Vec<String> {
    if rest.is_empty() {
        return vec![root_file.to_string()];
    }
    let mut base = Path::new(root_file)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();
    for segment in rest {
        base.push(segment);
    }
    vec![
        base.with_extension("rs")
            .to_string_lossy()
            .replace('\\', "/"),
        base.join("mod.rs").to_string_lossy().replace('\\', "/"),
    ]
}

/// Exactly one declared extern crate may supply `first`. Zero falls through to
/// lexical lookup; two different library roots are not guessed.
fn rust_extern_module_files(
    packages: &[&RustPackage],
    first: &str,
    rest: &[&str],
) -> ExternModuleFiles {
    let mut roots = Vec::new();
    for package in packages {
        for (name, root_file) in &package.extern_crates {
            if name == first {
                roots.push(root_file.clone());
            }
        }
    }
    roots.sort();
    roots.dedup();
    match roots.as_slice() {
        [] => ExternModuleFiles::Miss,
        [root_file] => ExternModuleFiles::Hit(rust_library_module_files(root_file, rest)),
        _ => ExternModuleFiles::Ambiguous,
    }
}

fn rust_module_files_below_alias(
    alias_files: &[String],
    ref_path: &str,
    name: &str,
    crate_roots: Option<&std::collections::HashSet<String>>,
) -> Vec<String> {
    let segments = ref_path.split("::").collect::<Vec<_>>();
    let nested = segments
        .get(1..segments.len().saturating_sub(1))
        .unwrap_or(&[]);
    if segments.last().copied() != Some(name) {
        return Vec::new();
    }
    alias_files
        .iter()
        .flat_map(|module_file| {
            let path = Path::new(module_file);
            let mut base = if path.file_name().and_then(|part| part.to_str()) == Some("mod.rs")
                || crate_roots.is_some_and(|roots| roots.contains(module_file))
            {
                path.parent().unwrap_or_else(|| Path::new("")).to_path_buf()
            } else {
                path.with_extension("")
            };
            for segment in nested {
                base.push(segment);
            }
            [
                base.with_extension("rs")
                    .to_string_lossy()
                    .replace('\\', "/"),
                base.join("mod.rs").to_string_lossy().replace('\\', "/"),
            ]
        })
        .collect()
}

impl GraphIndex {
    fn resolve_import_target(
        &self,
        file: &str,
        name: &str,
        path: &str,
        imported_items: Option<&serde_json::Value>,
    ) -> Option<i64> {
        if matches!(
            greppy_parser::language_for_path(Path::new(file)),
            Language::JavaScript | Language::TypeScript { .. }
        ) && path.starts_with('.')
        {
            return self.resolve_relative_js_import(file, name, path);
        }
        if !file.ends_with(".rs") {
            return self.unique_def_named_with_path(
                &greppy_resolver::IMPORTABLE_LABELS,
                name,
                path,
            );
        }
        if path.is_empty() {
            // Older cached Rust imports may preserve the exact provenance
            // only in imported_items. Recover that path rather than guessing
            // a project-wide namesake or requiring source re-extraction.
            let mut targets = Vec::new();
            for item in imported_items
                .and_then(|value| value.as_array())
                .into_iter()
                .flatten()
            {
                let Some(original) = item.get("original_name").and_then(|value| value.as_str())
                else {
                    continue;
                };
                if original != name
                    && item.get("imported_name").and_then(|value| value.as_str()) != Some(name)
                {
                    continue;
                }
                let Some(item_path) = item.get("path").and_then(|value| value.as_str()) else {
                    continue;
                };
                let files = self.rust_module_files_for_path(file, item_path, original);
                targets.extend(self.rust_module_export_targets(
                    &files,
                    original,
                    &greppy_resolver::IMPORTABLE_LABELS,
                ));
            }
            targets.sort_unstable();
            targets.dedup();
            return match targets.as_slice() {
                [target] => Some(*target),
                _ => None,
            };
        }
        let files = self.rust_module_files_for_path(file, path, name);
        let targets =
            self.rust_module_export_targets(&files, name, &greppy_resolver::IMPORTABLE_LABELS);
        match targets.as_slice() {
            [target] => Some(*target),
            _ => None,
        }
    }

    fn resolve_relative_js_import(&self, file: &str, name: &str, path: &str) -> Option<i64> {
        if !path.starts_with("./") && !path.starts_with("../") {
            return None;
        }
        let parent = Path::new(file).parent().unwrap_or_else(|| Path::new(""));
        let mut base = std::path::PathBuf::new();
        for part in parent.join(path).components() {
            match part {
                std::path::Component::Normal(part) => base.push(part),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    if !base.pop() {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        let mut files = Vec::new();
        if base.extension().is_some() {
            files.push(base.to_string_lossy().replace('\\', "/"));
        } else {
            for extension in ["ts", "tsx", "js", "jsx", "mts", "cts", "mjs", "cjs"] {
                files.push(
                    base.with_extension(extension)
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
                files.push(
                    base.join(format!("index.{extension}"))
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
        let visible = files
            .iter()
            .filter(|file| self.known_files.contains(*file))
            .collect::<Vec<_>>();
        let [file] = visible.as_slice() else {
            return None;
        };
        // ES modules import value bindings as well as types and functions.
        // Keep this language-specific and fenced to the exact visible module.
        let labels = greppy_resolver::IMPORTABLE_LABELS
            .into_iter()
            .chain(["Variable"])
            .collect::<Vec<_>>();
        let targets = self
            .defs_named(&labels, name)
            .into_iter()
            .filter(|node| &node.file_path == *file)
            .map(|node| node.id)
            .collect::<Vec<_>>();
        match targets.as_slice() {
            [target] => Some(*target),
            _ => None,
        }
    }

    fn rust_module_files_for_path(&self, file: &str, path: &str, name: &str) -> Vec<String> {
        let Some(module) = path
            .strip_suffix(name)
            .and_then(|path| path.strip_suffix("::"))
        else {
            return Vec::new();
        };
        self.rust_module_files_for_module_path(file, module)
    }

    fn rust_module_files_for_module_path(&self, file: &str, module: &str) -> Vec<String> {
        let normalized = module.trim_start_matches("::");
        let mut segments = normalized.split("::");
        let first = segments.next().unwrap_or("");
        let rest = segments.collect::<Vec<_>>();
        if !module.starts_with("::") && !matches!(first, "crate" | "self" | "super") {
            let lexical = rust_module_files_for_module_path_with_crate_roots(
                file,
                module,
                self.rust_crate_roots.as_ref(),
            );
            if lexical
                .iter()
                .any(|candidate| self.known_files.contains(candidate))
            {
                return lexical;
            }
        }
        let owning = self
            .rust_libraries
            .iter()
            .filter(|library| Path::new(file).starts_with(&library.package_dir))
            .collect::<Vec<_>>();
        let nearest = owning.iter().map(|library| library.package_dir.len()).max();
        let nearest_packages = owning
            .into_iter()
            .filter(|package| Some(package.package_dir.len()) == nearest)
            .collect::<Vec<_>>();
        let libraries = nearest_packages
            .iter()
            .filter_map(|package| package.library.as_ref())
            .filter(|library| library.name == first)
            .collect::<Vec<_>>();
        if let [library] = libraries.as_slice() {
            return rust_library_module_files(&library.root_file, &rest);
        }
        if libraries.is_empty()
            && !module.starts_with("::")
            && !matches!(first, "" | "crate" | "self" | "super")
        {
            match rust_extern_module_files(&nearest_packages, first, &rest) {
                ExternModuleFiles::Hit(files) => return files,
                ExternModuleFiles::Ambiguous => return Vec::new(),
                ExternModuleFiles::Miss => {}
            }
        }
        if !libraries.is_empty() || module.starts_with("::") {
            return Vec::new();
        }
        rust_module_files_for_module_path_with_crate_roots(
            file,
            module,
            self.rust_crate_roots.as_ref(),
        )
    }

    /// Load every node for `project` in a single query and build the
    /// lookup maps. `qualified_name` order from SQL gives a deterministic
    /// per-name candidate order.
    fn load(store: &Store, project: &str) -> Result<Self> {
        let mut by_qname: std::collections::HashMap<String, NodeLite> =
            std::collections::HashMap::new();
        let mut by_name: std::collections::HashMap<String, Vec<NodeLite>> =
            std::collections::HashMap::new();
        let mut by_id: std::collections::HashMap<i64, NodeLite> = std::collections::HashMap::new();
        let mut id_to_file: std::collections::HashMap<i64, String> =
            std::collections::HashMap::new();
        let mut id_to_qname: std::collections::HashMap<i64, String> =
            std::collections::HashMap::new();
        let mut files_by_stem: std::collections::HashMap<String, Vec<i64>> =
            std::collections::HashMap::new();
        let mut known_files = std::collections::HashSet::new();
        let mut rust_inline_scopes = std::collections::HashMap::new();
        let mut as_ref_receivers = std::collections::HashMap::new();
        let mut open_traits = std::collections::HashSet::new();
        {
            let conn = store.conn();
            let mut stmt = conn
                .prepare_cached(
                    "SELECT id, name, qualified_name, label, file_path,
                            CASE
                                WHEN label = 'Field' AND json_extract(properties, '$.generic_payload') = 1 THEN NULL
                                WHEN label = 'Field' THEN json_extract(properties, '$.return_type')
                            END,
                            CASE WHEN label = 'Method' AND name = 'as_ref'
                                THEN json_extract(properties, '$.params[0].type') END,
                            CASE WHEN label = 'Interface'
                                THEN json_extract(properties, '$.has_bounds') END,
                            CASE WHEN label = 'Interface'
                                THEN json_extract(properties, '$.as_ref_receiver') END,
                            CASE WHEN file_path LIKE '%.rs'
                                THEN json_extract(properties, '$.rust_inline_module') END,
                            CASE WHEN file_path LIKE '%.rs'
                                THEN json_extract(properties, '$.visibility') END,
                            CASE WHEN file_path LIKE '%.rs'
                                THEN json_extract(properties, '$.rust_inline_module_vis') END
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
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, Option<String>>(6)?,
                        r.get::<_, Option<i64>>(7)?,
                        r.get::<_, Option<String>>(8)?,
                        r.get::<_, Option<String>>(9)?,
                        r.get::<_, Option<String>>(10)?,
                        r.get::<_, Option<String>>(11)?,
                    ))
                })
                .map_err(sqlite_err)?;
            for row in rows {
                let (
                    id,
                    name,
                    qname,
                    label,
                    file_path,
                    declared_type,
                    as_ref_receiver,
                    trait_bounds,
                    trait_as_ref,
                    rust_inline_module,
                    visibility,
                    rust_inline_module_vis,
                ) = row.map_err(sqlite_err)?;
                note_edge_resolution_work(1);
                if label == "File" {
                    let base = file_path.rsplit('/').next().unwrap_or(&file_path);
                    let stem = base.rsplit_once('.').map_or(base, |(s, _)| s);
                    files_by_stem.entry(stem.to_string()).or_default().push(id);
                }
                if label == "Method" && name == "as_ref" {
                    if let Some(receiver) = as_ref_receiver {
                        as_ref_receivers.insert(qname.clone(), receiver);
                    }
                }
                if label == "Interface" {
                    if let Some(receiver) = trait_as_ref {
                        let trait_name = qname.rsplit("::").next().unwrap_or("");
                        if !trait_name.is_empty() {
                            as_ref_receivers
                                .entry(format!("{file_path}::{trait_name}::as_ref"))
                                .or_insert(receiver);
                        }
                    }
                }
                if trait_bounds == Some(1) {
                    open_traits.insert(id);
                }
                if let Some(module) = rust_inline_module.filter(|module| !module.is_empty()) {
                    let module_vis = rust_inline_module_vis
                        .unwrap_or_default()
                        .split('\u{1f}')
                        .map(str::to_string)
                        .collect::<Vec<_>>();
                    rust_inline_scopes.insert(
                        id,
                        RustInlineScope {
                            module,
                            visibility,
                            module_vis,
                        },
                    );
                }
                let node = NodeLite {
                    id,
                    label,
                    file_path: file_path.clone(),
                    declared_type,
                };
                known_files.insert(file_path.clone());
                id_to_file.insert(id, file_path);
                id_to_qname.insert(id, qname.clone());
                by_name.entry(name).or_default().push(node.clone());
                by_id.insert(id, node.clone());
                by_qname.insert(qname, node);
            }
        }
        let metadata = rust_crate_roots_for_project(store, project, &known_files);
        let (rust_crate_roots, rust_libraries) = match metadata {
            Some((roots, libraries)) => (Some(roots), libraries),
            None => (None, Vec::new()),
        };
        let (anyhow_context, anyhow_dependency_binding) =
            rust_anyhow_context(store, Some(project))?;
        let namespace_shadowed = known_files.iter().any(|file| {
            Path::new(file)
                .file_name()
                .is_some_and(|name| name == "anyhow.rs")
                || file.ends_with("anyhow/mod.rs")
        });
        let anyhow_factory_files: std::collections::HashSet<String> = anyhow_context
            .into_iter()
            .filter(|(scope_project, _)| scope_project == project && !namespace_shadowed)
            .map(|(_, file)| file)
            .collect();
        let mut anyhow_glob_proofs = std::collections::HashSet::new();
        let mut observed_globs = std::collections::HashSet::new();
        if let Some(project_info) = store
            .get_project(project)?
            .filter(|_| !anyhow_factory_files.is_empty())
        {
            let root = Path::new(&project_info.root_path);
            let states = store.list_file_states(project)?;
            let facts = {
                let mut stmt = store.conn().prepare("SELECT file_path, json_extract(properties,'$.receiver_anyhow_glob_files') FROM raw_edges WHERE project=?1 AND json_type(properties,'$.receiver_anyhow_factory_owner')='text' AND json_type(properties,'$.receiver_anyhow_glob_files')='array'").map_err(sqlite_err)?;
                let rows = stmt
                    .query_map(rusqlite::params![project], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })
                    .map_err(sqlite_err)?;
                rows.collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(sqlite_err)?
            };
            for (file_path, encoded_globs) in facts {
                let Ok(globs) = serde_json::from_str::<Vec<serde_json::Value>>(&encoded_globs)
                else {
                    continue;
                };
                let proof_key =
                    format!("{}:{}", file_path, serde_json::Value::Array(globs.clone()));
                if !observed_globs.insert(proof_key.clone()) {
                    continue;
                }
                let proof = globs.iter().all(|value| {
                    let Some(path) = value.as_str() else {
                        return false;
                    };
                    let relative = Path::new(&file_path)
                        .parent()
                        .unwrap_or_else(|| Path::new(""))
                        .join(path);
                    if relative
                        .components()
                        .any(|part| !matches!(part, std::path::Component::Normal(_)))
                    {
                        return false;
                    }
                    let relative_text = relative.to_string_lossy().replace('\\', "/");
                    let Some(state) = states.iter().find(|state| state.rel_path == relative_text)
                    else {
                        return false;
                    };
                    let Ok(canonical_root) = std::fs::canonicalize(root) else {
                        return false;
                    };
                    let Ok(canonical_path) = std::fs::canonicalize(root.join(&relative)) else {
                        return false;
                    };
                    if !canonical_path.starts_with(&canonical_root) {
                        return false;
                    }
                    let Ok((bytes, _)) = read_stable_file(&canonical_path) else {
                        return false;
                    };
                    if file_state::sha256_hex(&bytes) != state.sha256 {
                        return false;
                    }
                    let Ok(tree) = greppy_parser::parse(Language::Rust, &bytes) else {
                        return false;
                    };
                    if tree.root_node().has_error() {
                        return false;
                    }
                    let mut cursor = tree.root_node().walk();
                    let clear = tree.root_node().named_children(&mut cursor).all(|item| {
                        if item.kind() == "macro_invocation"
                            || (item.kind() == "expression_statement"
                                && (0..item.named_child_count())
                                    .filter_map(|i| item.named_child(i))
                                    .any(|child| child.kind() == "macro_invocation"))
                        {
                            return false;
                        }
                        if item.kind() == "use_declaration"
                            && (0..item.named_child_count())
                                .filter_map(|i| item.named_child(i))
                                .any(|child| child.kind() == "visibility_modifier")
                        {
                            return false;
                        }
                        !item.child_by_field_name("name").is_some_and(|name| {
                            matches!(
                                std::str::from_utf8(&bytes[name.byte_range()]).unwrap_or(""),
                                "anyhow" | "Option" | "Some"
                            )
                        })
                    });
                    clear
                });
                if proof {
                    anyhow_glob_proofs.insert(proof_key);
                }
            }
        }
        let direct_field_trait_owners = {
            let mut stmt = store.conn().prepare("SELECT DISTINCT json_extract(properties,'$.type_name') FROM raw_edges WHERE project=?1 AND edge_type='IMPLEMENTS' AND json_type(properties,'$.type_name')='text'").map_err(sqlite_err)?;
            let rows = stmt
                .query_map(rusqlite::params![project], |row| row.get::<_, String>(0))
                .map_err(sqlite_err)?;
            rows.collect::<std::result::Result<std::collections::HashSet<_>, _>>()
                .map_err(sqlite_err)?
        };
        Ok(GraphIndex {
            by_qname,
            by_id,
            by_name,
            imports_by_file: std::collections::HashMap::new(),
            import_aliases_by_file: std::collections::HashMap::new(),
            import_alias_sources_by_file: std::collections::HashMap::new(),
            import_module_files_by_file: std::collections::HashMap::new(),
            import_globs_by_file: std::collections::HashMap::new(),
            rust_namespaces_by_file: std::collections::HashMap::new(),
            rust_crate_roots,
            anyhow_factory_files,
            anyhow_dependency_binding,
            anyhow_glob_proofs,
            direct_field_trait_owners,
            rust_libraries,
            id_to_file,
            id_to_qname,
            files_by_stem,
            known_files,
            rust_inline_scopes,
            as_ref_receivers,
            open_traits,
        })
    }

    /// `(qname) → node` lookup (mirrors `Store::get_node_by_qname`).
    fn by_qname(&self, qname: &str) -> Option<&NodeLite> {
        note_edge_resolution_work(1);
        self.by_qname.get(qname)
    }

    fn qname_for_id(&self, id: i64) -> Option<&str> {
        self.id_to_qname.get(&id).map(String::as_str)
    }

    /// Record a resolved IMPORTS target for `file` so the reference
    /// resolver can read the file's imports back for disambiguation.
    fn record_import(&mut self, file: &str, target_id: i64) {
        self.imports_by_file
            .entry(file.to_string())
            .or_default()
            .insert(target_id);
    }

    fn record_import_items(&mut self, edge: &ExtractedEdge, file: &str) {
        if matches!(
            greppy_parser::language_for_path(Path::new(file)),
            Language::JavaScript | Language::TypeScript { .. }
        ) {
            let name = edge
                .properties
                .get("original_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let alias = edge
                .properties
                .get("imported_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let path = edge
                .properties
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !name.is_empty() && !alias.is_empty() && path.starts_with('.') {
                // Preserve the explicit binding even if its module or export
                // is unresolved: a project-wide namesake is not that import.
                self.import_alias_sources_by_file
                    .entry(file.to_string())
                    .or_default()
                    .entry(alias.to_string())
                    .or_default()
                    .push((path.to_string(), name.to_string()));
                if let Some(target) = self.resolve_relative_js_import(file, name, path) {
                    self.record_import(file, target);
                    self.import_aliases_by_file
                        .entry(file.to_string())
                        .or_default()
                        .entry(alias.to_string())
                        .or_default()
                        .insert(target);
                }
            }
            return;
        }
        let Some(items) = edge
            .properties
            .get("imported_items")
            .and_then(|value| value.as_array())
        else {
            return;
        };
        for item in items {
            if item.get("glob").and_then(|value| value.as_bool()) == Some(true) {
                if let Some(path) = item.get("path").and_then(|value| value.as_str()) {
                    self.import_globs_by_file
                        .entry(file.to_string())
                        .or_default()
                        .push(path.to_string());
                }
                continue;
            }
            let Some(name) = item.get("original_name").and_then(|value| value.as_str()) else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let path = item
                .get("path")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let module_files = self.rust_module_files_for_path(file, path, name);
            let visible_module_files = module_files
                .iter()
                .filter(|module_file| self.known_files.contains(*module_file))
                .cloned()
                .collect::<Vec<_>>();
            self.import_module_files_by_file
                .entry(file.to_string())
                .or_default()
                .extend(visible_module_files);
            let alias = item
                .get("imported_name")
                .and_then(|value| value.as_str())
                .filter(|alias| !alias.is_empty())
                .unwrap_or(name);
            self.import_alias_sources_by_file
                .entry(file.to_string())
                .or_default()
                .entry(alias.to_string())
                .or_default()
                .push((path.to_string(), name.to_string()));
            let candidates = self.defs_named(&greppy_resolver::IMPORTABLE_LABELS, name);
            let exact = candidates
                .iter()
                .filter(|candidate| module_files.contains(&candidate.file_path))
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>();
            if let [target] = exact.as_slice() {
                self.record_import(file, *target);
                self.import_aliases_by_file
                    .entry(file.to_string())
                    .or_default()
                    .entry(alias.to_string())
                    .or_default()
                    .insert(*target);
            }
            let namespace_path = format!("{path}::__namespace__");
            let existing_modules = self
                .rust_module_files_for_path(file, &namespace_path, "__namespace__")
                .into_iter()
                .filter(|module_file| self.known_files.contains(module_file))
                .collect::<Vec<_>>();
            if !existing_modules.is_empty() {
                self.rust_namespaces_by_file
                    .entry(file.to_string())
                    .or_default()
                    .insert(alias.to_string(), existing_modules);
            }
        }
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

    fn rust_module_export_targets(
        &self,
        module_files: &[String],
        name: &str,
        labels: &[&str],
    ) -> Vec<i64> {
        let mut seen = std::collections::HashSet::new();
        let mut targets = Vec::new();
        for module_file in module_files {
            self.rust_module_export_targets_from_file(
                module_file,
                name,
                labels,
                &mut seen,
                &mut targets,
            );
        }
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    fn rust_module_export_targets_from_file(
        &self,
        module_file: &str,
        name: &str,
        labels: &[&str],
        seen: &mut std::collections::HashSet<(String, String)>,
        targets: &mut Vec<i64>,
    ) {
        if !seen.insert((module_file.to_string(), name.to_string())) {
            return;
        }
        let initial_count = targets.len();
        targets.extend(
            self.defs_named(labels, name)
                .into_iter()
                .filter(|node| node.file_path == module_file)
                .map(|node| node.id),
        );
        if targets.len() > initial_count {
            return;
        }
        if let Some(aliases) = self
            .import_aliases_by_file
            .get(module_file)
            .and_then(|aliases| aliases.get(name))
        {
            targets.extend(aliases.iter().copied().filter(|target| {
                self.by_id
                    .get(target)
                    .is_some_and(|node| labels.contains(&node.label.as_str()))
            }));
        }
        if let Some(sources) = self
            .import_alias_sources_by_file
            .get(module_file)
            .and_then(|aliases| aliases.get(name))
        {
            for (path, original_name) in sources {
                let mut source_files = self
                    .rust_module_files_for_path(module_file, path, original_name)
                    .into_iter()
                    .filter(|file| self.known_files.contains(file))
                    .collect::<Vec<_>>();
                if source_files.is_empty() && path == original_name {
                    source_files.push(module_file.to_string());
                }
                for source_file in source_files {
                    self.rust_module_export_targets_from_file(
                        &source_file,
                        original_name,
                        labels,
                        seen,
                        targets,
                    );
                }
            }
        }
        // An explicit binding shadows glob imports even when its target is
        // unresolved; do not invent a different binding through a glob.
        if self
            .import_alias_sources_by_file
            .get(module_file)
            .is_some_and(|aliases| aliases.contains_key(name))
        {
            return;
        }
        if let Some(globs) = self.import_globs_by_file.get(module_file) {
            for glob in globs {
                for exported_file in self.rust_module_files_for_module_path(module_file, glob) {
                    if self.known_files.contains(&exported_file) {
                        self.rust_module_export_targets_from_file(
                            &exported_file,
                            name,
                            labels,
                            seen,
                            targets,
                        );
                    }
                }
            }
        }
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
        let referrer_family = resolution_language_family(referrer_file);
        if let Some(alias_targets) = self
            .import_aliases_by_file
            .get(referrer_file)
            .and_then(|aliases| aliases.get(name))
        {
            let matching = alias_targets
                .iter()
                .filter(|target| {
                    self.by_id
                        .get(target)
                        .is_some_and(|node| labels.contains(&node.label.as_str()))
                })
                .copied()
                .collect::<Vec<_>>();
            if let [target] = matching.as_slice() {
                return UniqueResolution::Unique(*target);
            }
        }
        // A local Rust item shadows a glob import. Do not let Base export
        // hydration redirect a same-file reference to an imported namesake.
        let candidates = self
            .defs_named(labels, name)
            .into_iter()
            .filter(|node| resolution_language_family(&node.file_path) == referrer_family)
            .collect::<Vec<_>>();
        let local = candidates
            .iter()
            .filter(|node| node.file_path == referrer_file)
            .collect::<Vec<_>>();
        if let Some(target) = local
            .iter()
            .min_by_key(|node| (navigation_label_rank(&node.label), node.id))
        {
            return UniqueResolution::Unique(target.id);
        }
        if self
            .import_alias_sources_by_file
            .get(referrer_file)
            .is_some_and(|aliases| aliases.contains_key(name))
        {
            let exported =
                self.rust_module_export_targets(&[referrer_file.to_string()], name, labels);
            return match exported.as_slice() {
                [target] => UniqueResolution::Unique(*target),
                [] => UniqueResolution::Unresolved,
                _ => UniqueResolution::Ambiguous,
            };
        }
        if let Some(globs) = self.import_globs_by_file.get(referrer_file) {
            let module_files = globs
                .iter()
                .flat_map(|glob| self.rust_module_files_for_module_path(referrer_file, glob))
                .filter(|file| self.known_files.contains(file))
                .collect::<Vec<_>>();
            let exported = self.rust_module_export_targets(&module_files, name, labels);
            if let [target] = exported.as_slice() {
                return UniqueResolution::Unique(*target);
            }
        }
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

    /// Resolve `crate::trace::name` when `trace` is an inline module in the crate
    /// root file rather than `trace.rs` / `trace/mod.rs`. Exactly one visible
    /// callable wins. A miss falls through to associated-item resolution.
    fn resolve_rust_inline_module_call(
        &self,
        src_id: i64,
        name: &str,
        sites: &[(String, String)],
    ) -> Option<i64> {
        if sites.is_empty() {
            return None;
        }
        let caller_file = self.file_of(src_id)?;
        let caller_module = self
            .rust_inline_scopes
            .get(&src_id)
            .map(|scope| scope.module.as_str())
            .unwrap_or("");
        let nodes = self.by_name.get(name)?;
        let mut functions = Vec::new();
        let mut other = Vec::new();
        for (parent_file, inline_path) in sites {
            for node in nodes {
                if node.file_path != *parent_file || !CALLABLE_LABELS.contains(&node.label.as_str())
                {
                    continue;
                }
                let Some(scope) = self.rust_inline_scopes.get(&node.id) else {
                    continue;
                };
                if scope.module != *inline_path {
                    continue;
                }
                if !rust_path_modules_visible(
                    &scope.module,
                    &scope.module_vis,
                    parent_file,
                    caller_module,
                    caller_file,
                    self.rust_crate_roots.as_ref(),
                ) || !rust_item_visible(
                    scope.visibility.as_deref(),
                    &scope.module,
                    parent_file,
                    caller_module,
                    caller_file,
                    self.rust_crate_roots.as_ref(),
                ) {
                    continue;
                }
                if node.label == "Function" {
                    functions.push(node.id);
                } else {
                    other.push(node.id);
                }
            }
        }
        functions.sort_unstable();
        functions.dedup();
        other.sort_unstable();
        other.dedup();
        match functions.as_slice() {
            [id] => Some(*id),
            [] => match other.as_slice() {
                [id] => Some(*id),
                _ => None,
            },
            _ => None,
        }
    }

    /// Resolve the initializer of a local alias (`let test = crate_b::f`) to one
    /// callable. An unresolved or ambiguous path stays unresolved; it must not
    /// fall through to a function that happens to share the local's name.
    fn resolve_rust_local_alias_path(&self, src_id: i64, path: &str) -> Option<i64> {
        let path = path.trim();
        let name = path.rsplit("::").next().unwrap_or("");
        if name.is_empty() || matches!(path, "Ok" | "Err" | "Some") {
            return None;
        }
        if !path.contains("::") {
            return self
                .resolve_unique_status_with_imports(&CALLABLE_LABELS, name, src_id)
                .unique_id();
        }
        let referrer_file = self.file_of(src_id)?;
        let first_segment = path.split("::").next().unwrap_or("");
        let module_files = self
            .rust_namespaces_by_file
            .get(referrer_file)
            .and_then(|aliases| aliases.get(first_segment))
            .map(|alias_files| {
                rust_module_files_below_alias(
                    alias_files,
                    path,
                    name,
                    self.rust_crate_roots.as_ref(),
                )
            })
            .unwrap_or_else(|| self.rust_module_files_for_path(referrer_file, path, name));
        let targets = self.rust_module_export_targets(&module_files, name, &CALLABLE_LABELS);
        match targets.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }

    /// Resolve a CALLS edge. Receiver dispatch is deliberately method-only:
    /// resolving `value.as_bytes()` to an unrelated free `as_bytes` function
    /// is worse than leaving the edge unresolved. Other calls retain the
    /// direct-qname, callable-name, then constructable fallback sequence.
    fn resolve_call_target(&self, edge: &ExtractedEdge) -> Option<i64> {
        clear_option_field_unresolved();
        if let Some(name) = edge
            .properties
            .get("rust_expression_macro")
            .and_then(|v| v.as_str())
        {
            self.rust_expression_macro_identity(
                &edge.file_path,
                name,
                true,
                &mut std::collections::HashSet::new(),
            )?;
        }
        if edge
            .properties
            .get("rust_local_type_owner")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            return None;
        }
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
            if let Some(fact) = edge.properties.get("receiver_provenance") {
                let kind = fact.get("kind").and_then(|value| value.as_str());
                if kind == Some("direct_self_field") || kind == Some("typed_field_chain") {
                    return self.resolve_direct_self_field_receiver(src_id, fact, name);
                }
                return self.resolve_option_field_receiver(src_id, edge, fact, name);
            }
            let owner = edge
                .properties
                .get("receiver_owner")
                .and_then(|value| value.as_str())
                .or_else(|| {
                    let globs = edge
                        .properties
                        .get("receiver_anyhow_glob_files")?
                        .as_array()?;
                    let proof_key = format!(
                        "{}:{}",
                        edge.file_path,
                        serde_json::Value::Array(globs.clone())
                    );
                    (self.anyhow_factory_files.contains(&edge.file_path)
                        && self.anyhow_glob_proofs.contains(&proof_key))
                    .then(|| {
                        edge.properties
                            .get("receiver_anyhow_factory_owner")
                            .and_then(|value| value.as_str())
                    })
                    .flatten()
                })?;
            if edge
                .properties
                .get("receiver_option_pattern")
                .and_then(|value| value.as_bool())
                == Some(true)
            {
                let id = self.resolve_rust_type_path(&edge.file_path, owner)?;
                let owner_file = self.file_of(id)?;
                let owner_name = self.qname_for_id(id)?.rsplit("::").next()?;
                let method = self.by_qname(&format!("{owner_file}::{owner_name}::{name}"))?;
                return (method.label == "Method").then_some(method.id);
            }
            return self.resolve_receiver_method(&edge.file_path, owner, name);
        }
        // A module-qualified Rust call names its complete module path. Resolve
        // it against the referrer's namespace and that module's imports before
        // any same-file or global-name fallback. This distinguishes paths such
        // as `left::channel::target` and `right::channel::target` even though
        // both defining files are named `implementation.rs`.
        let rust_qualified_path = edge
            .properties
            .get("callee_path")
            .and_then(|value| value.as_str())
            .filter(|_| edge.file_path.ends_with(".rs"));
        if let Some(ref_path) = rust_qualified_path {
            let referrer_file = self.file_of(src_id)?;
            let first_segment = ref_path.split("::").next().unwrap_or("");
            let module_files = self
                .rust_namespaces_by_file
                .get(referrer_file)
                .and_then(|aliases| aliases.get(first_segment))
                .map(|alias_files| {
                    rust_module_files_below_alias(
                        alias_files,
                        ref_path,
                        name,
                        self.rust_crate_roots.as_ref(),
                    )
                })
                .unwrap_or_else(|| self.rust_module_files_for_path(referrer_file, ref_path, name));
            let module_exists = module_files
                .iter()
                .any(|module_file| self.known_files.contains(module_file));
            if !module_exists {
                let sites = if let Some(alias_files) = self
                    .rust_namespaces_by_file
                    .get(referrer_file)
                    .and_then(|aliases| aliases.get(first_segment))
                {
                    rust_inline_sites_below_alias(
                        alias_files,
                        ref_path,
                        name,
                        &self.known_files,
                        self.rust_crate_roots.as_ref(),
                    )
                } else {
                    let module = ref_path
                        .strip_suffix(name)
                        .and_then(|path| path.strip_suffix("::"))
                        .unwrap_or("");
                    rust_inline_module_sites(
                        referrer_file,
                        module,
                        &self.known_files,
                        self.rust_crate_roots.as_ref(),
                    )
                };
                if let Some(id) = self.resolve_rust_inline_module_call(src_id, name, &sites) {
                    return Some(id);
                }
                return self.resolve_associated_member(
                    src_id,
                    ref_path,
                    name,
                    &["Method", "EnumVariant"],
                );
            }
            let in_module = self.rust_module_export_targets(&module_files, name, &CALLABLE_LABELS);
            match in_module.as_slice() {
                [id] => return Some(*id),
                _ => return None,
            }
        }
        // Preserve the existing basename-based qualified-call behavior for
        // non-Rust extractors, whose path syntax is language-specific.
        if edge.file_path.ends_with(".rs")
            && edge
                .properties
                .get("ref_local_binding")
                .and_then(|value| value.as_bool())
                == Some(true)
        {
            // A `let test = real` / parameter named `test` is not a call to an
            // unrelated `fn test`. Only a plain initializer path may retarget it.
            return edge
                .properties
                .get("rust_local_callee_path")
                .and_then(|value| value.as_str())
                .and_then(|path| self.resolve_rust_local_alias_path(src_id, path));
        }
        // Unqualified prelude constructors are not user functions. A same-file
        // `fn Err` is shadowed by the prelude in this position; `MyEnum::Err`
        // stays on the qualified path above.
        if edge.file_path.ends_with(".rs")
            && matches!(name, "Ok" | "Err" | "Some")
            && edge
                .properties
                .get("callee_path")
                .and_then(|value| value.as_str())
                .is_none()
        {
            return None;
        }
        if rust_qualified_path.is_none() {
            if let Some(module) = edge
                .properties
                .get("callee_path")
                .and_then(|value| value.as_str())
                .and_then(|path| greppy_resolver::path_module_segment(path, name))
            {
                let in_module: Vec<i64> = self
                    .defs_named(&CALLABLE_LABELS, name)
                    .into_iter()
                    .filter(|node| greppy_resolver::file_stem_matches(&node.file_path, module))
                    .map(|node| node.id)
                    .collect();
                match in_module.as_slice() {
                    [id] => return Some(*id),
                    [] => {}
                    _ => return None,
                }
            }
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

    /// Resolve a qualified method or enum member only when its owner resolves in the
    /// referrer's scope. This preserves imported and lowercase Rust type names
    /// without treating a missing qualified module as an unqualified call.
    fn resolve_associated_member(
        &self,
        src_id: i64,
        ref_path: &str,
        name: &str,
        member_labels: &[&str],
    ) -> Option<i64> {
        let owner = ref_path.rsplit("::").nth(1).unwrap_or("");
        if owner.is_empty() || name.is_empty() {
            return None;
        }
        let referrer_file = self.file_of(src_id)?;
        let owner_path = ref_path.strip_suffix(name)?.strip_suffix("::")?;
        let owner_id = if owner_path == "Self" {
            let source_qname = self.qname_for_id(src_id)?;
            if self.by_qname(source_qname)?.label != "Method" {
                return None;
            }
            let context_owner = source_qname.rsplit("::").nth(1)?;
            self.resolve_unique_status_with_imports(&CONSTRUCTABLE_LABELS, context_owner, src_id)
                .unique_id()?
        } else if owner_path == owner {
            self.resolve_unique_status_with_imports(&CONSTRUCTABLE_LABELS, owner, src_id)
                .unique_id()?
        } else {
            let first_segment = owner_path.split("::").next().unwrap_or("");
            let module_files = self
                .rust_namespaces_by_file
                .get(referrer_file)
                .and_then(|aliases| aliases.get(first_segment))
                .map(|alias_files| {
                    rust_module_files_below_alias(
                        alias_files,
                        owner_path,
                        owner,
                        self.rust_crate_roots.as_ref(),
                    )
                })
                .unwrap_or_else(|| {
                    self.rust_module_files_for_path(referrer_file, owner_path, owner)
                });
            let owners = self
                .defs_named(&CONSTRUCTABLE_LABELS, owner)
                .into_iter()
                .filter(|candidate| {
                    module_files.iter().any(|file| {
                        candidate.file_path == *file
                            || self
                                .imports_by_file
                                .get(file)
                                .is_some_and(|targets| targets.contains(&candidate.id))
                    })
                })
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>();
            match owners.as_slice() {
                [id] => *id,
                _ => return None,
            }
        };
        let owner_file = self.file_of(owner_id)?;
        if self.by_id.get(&owner_id)?.label == "Enum"
            && (owner_path == owner || owner_path == "Self")
        {
            // Enum owners must be bound in the current file. Project-wide
            // uniqueness alone does not make another module's type visible.
            let bound_name = if owner_path == "Self" {
                self.qname_for_id(src_id)?.rsplit("::").nth(1)?
            } else {
                owner
            };
            let scoped = self.rust_module_export_targets(
                &[referrer_file.to_string()],
                bound_name,
                &["Enum"],
            );
            if scoped.as_slice() != [owner_id] {
                return None;
            }
        }
        let resolved_owner = self.qname_for_id(owner_id)?.rsplit("::").next()?;
        let suffix = format!("::{resolved_owner}::{name}");
        let matches = self
            .defs_named(member_labels, name)
            .into_iter()
            .filter(|node| {
                node.file_path == owner_file
                    && self
                        .qname_for_id(node.id)
                        .is_some_and(|qname| qname.ends_with(&suffix))
            })
            .map(|node| node.id)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }

    /// Resolve a type only in its lexical module/import scope. Project-wide
    /// uniqueness does not establish that a type is bound in this scope.
    fn resolve_rust_type_path(&self, file: &str, path: &str) -> Option<i64> {
        let path = path.trim();
        if path.is_empty()
            || !path.split("::").all(|part| {
                !part.is_empty() && part.chars().all(|c| c.is_alphanumeric() || c == '_')
            })
        {
            return None;
        }
        let name = path.rsplit("::").next()?;
        if !path.contains("::") {
            return self.resolve_bare_rust_type_name(file, name);
        }
        let files = {
            let first = path.split("::").next()?;
            self.rust_namespaces_by_file
                .get(file)
                .and_then(|aliases| aliases.get(first))
                .map(|files| {
                    rust_module_files_below_alias(files, path, name, self.rust_crate_roots.as_ref())
                })
                .unwrap_or_else(|| self.rust_module_files_for_path(file, path, name))
        };
        let owners = self.rust_module_export_targets(&files, name, &["Class", "Struct", "Enum"]);
        match owners.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }

    fn resolve_direct_self_field_receiver(
        &self,
        src_id: i64,
        fact: &serde_json::Value,
        name: &str,
    ) -> Option<i64> {
        let file = self.file_of(src_id)?;
        let base = self.resolve_rust_type_path(file, fact.get("base_type")?.as_str()?)?;
        let owner = self.qname_for_id(base)?;
        let field = self.by_qname(&format!("{owner}::{}", fact.get("field")?.as_str()?))?;
        if field.label != "Field" {
            return None;
        }
        let declared = field.declared_type.as_deref()?.trim();
        // No wrapper, reference, generic substitution or opaque inference.
        if declared.is_empty()
            || declared.split("::").any(|segment| {
                segment.is_empty() || !segment.chars().all(|ch| ch.is_alphanumeric() || ch == '_')
            })
        {
            return None;
        }
        let payload = self.resolve_rust_type_path(&field.file_path, declared)?;
        let payload_file = self.file_of(payload)?;
        let payload_name = self.qname_for_id(payload)?.rsplit("::").next()?;
        // Method nodes do not distinguish inherent and trait implementations.
        // Conservatively refuse represented trait owners rather than assigning
        // a trait method by its coincidentally matching qname.
        if self.direct_field_trait_owners.contains(payload_name) {
            return None;
        }
        let method = self.by_qname(&format!("{payload_file}::{payload_name}::{name}"))?;
        (method.label == "Method").then_some(method.id)
    }

    fn resolve_option_field_receiver(
        &self,
        src_id: i64,
        edge: &ExtractedEdge,
        fact: &serde_json::Value,
        name: &str,
    ) -> Option<i64> {
        if fact.get("adapter")?.as_str()? != "as_ref" || fact.get("pattern")?.as_str()? != "Some" {
            return None;
        }
        let file = self.file_of(src_id)?;
        let base = self.resolve_rust_type_path(file, fact.get("base_type")?.as_str()?)?;
        let owner_qname = self.qname_for_id(base)?;
        let field_name = fact.get("field")?.as_str()?;
        let field = self.by_qname(&format!("{owner_qname}::{field_name}"))?;
        if field.label != "Field" {
            return None;
        }
        let declared = field.declared_type.as_deref()?.trim();
        let (option, payload) = declared.split_once('<')?;
        let payload = payload.strip_suffix('>')?.trim();
        // Only a simple explicit Option payload is represented. References,
        // generics, aliases and arbitrary adapters remain unresolved.
        match option.trim() {
            "std::option::Option" | "core::option::Option" => {
                let files =
                    self.rust_module_files_for_path(&field.file_path, option.trim(), "Option");
                let first = option.trim().split("::").next()?;
                if !self
                    .rust_module_export_targets(&files, "Option", &CONSTRUCTABLE_LABELS)
                    .is_empty()
                    || self
                        .rust_namespaces_by_file
                        .get(&field.file_path)
                        .is_some_and(|aliases| aliases.contains_key(first))
                {
                    return None;
                }
            }
            "Option" => {
                if !self
                    .rust_module_export_targets(
                        std::slice::from_ref(&field.file_path),
                        "Option",
                        &CONSTRUCTABLE_LABELS,
                    )
                    .is_empty()
                    || self
                        .import_globs_by_file
                        .get(&field.file_path)
                        .is_some_and(|globs| !globs.is_empty())
                {
                    return None;
                }
                if let Some(sources) = self
                    .import_alias_sources_by_file
                    .get(&field.file_path)
                    .and_then(|aliases| aliases.get("Option"))
                {
                    if sources.iter().any(|(path, original)| {
                        original != "Option"
                            || !matches!(
                                path.as_str(),
                                "std::option::Option" | "core::option::Option"
                            )
                    }) {
                        return None;
                    }
                }
            }
            _ => return None,
        }
        let payload_id = self.resolve_rust_type_path(&field.file_path, payload)?;
        let payload_file = self.file_of(payload_id)?;
        let payload_name = self.qname_for_id(payload_id)?.rsplit("::").next()?;
        let method = self.by_qname(&format!("{payload_file}::{payload_name}::{name}"))?;
        let reasons = self.option_receiver_limits(file, fact)?;
        let id = (method.label == "Method").then_some(method.id)?;
        if reasons.is_empty() {
            clear_option_field_unresolved();
        } else {
            // `new_edge` applies these reasons only when the persisted row is
            // this same CALLS edge. A skipped resolution clears the slot.
            set_option_field_unresolved(edge, reasons);
        }
        Some(id)
    }

    /// A bare type name is the same-file definition, or one imported type alias.
    /// A local item and an import, or two imports, are ambiguous and resolve
    /// to nothing.
    fn resolve_bare_rust_type_name(&self, file: &str, name: &str) -> Option<i64> {
        let local = self.rust_module_export_targets(
            &[file.to_string()],
            name,
            &["Class", "Struct", "Enum"],
        );
        let Some(ids) = self
            .import_aliases_by_file
            .get(file)
            .and_then(|aliases| aliases.get(name))
        else {
            return match local.as_slice() {
                [id] => Some(*id),
                _ => None,
            };
        };
        let typed = ids
            .iter()
            .copied()
            .filter(|id| {
                self.by_id
                    .get(id)
                    .is_some_and(|node| matches!(node.label.as_str(), "Class" | "Struct" | "Enum"))
            })
            .collect::<Vec<_>>();
        if ids.len() != typed.len() || typed.len() > 1 {
            return None;
        }
        if let [imported] = typed.as_slice() {
            if local.is_empty() || local.as_slice() == [*imported].as_slice() {
                return Some(*imported);
            }
            return None;
        }
        match local.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }

    /// `None` rejects the receiver (no edge). An empty vec is a proven
    /// `CALLS` edge. Any other vec is an unresolved candidate for the one
    /// method the declared field type already identified.
    fn option_receiver_limits(&self, file: &str, fact: &serde_json::Value) -> Option<Vec<String>> {
        let Some(limits) = fact.get("limits") else {
            // A cached fact with no scope record is not proof that the
            // receiver is Option::as_ref. Drop the candidate. Fresh
            // extraction writes `limits` and can prove or leave it unresolved.
            return None;
        };
        if !limits.is_object() {
            return Some(vec!["unparsed import".to_string()]);
        }
        let mut reasons = Vec::new();
        let mut globs = Vec::new();
        if let Some(values) = limits.get("globs").and_then(|value| value.as_array()) {
            for value in values {
                if let Some(path) = value.as_str() {
                    push_unique_glob(&mut globs, path);
                }
            }
        }
        if let Some(stored) = self.import_globs_by_file.get(file) {
            for path in stored {
                push_unique_glob(&mut globs, path);
            }
        }
        for path in globs {
            reasons.push(format!("wildcard import {path}"));
        }
        if let Some(values) = limits.get("macros").and_then(|value| value.as_array()) {
            for value in values {
                let Some(name) = value.as_str().filter(|name| !name.is_empty()) else {
                    continue;
                };
                if name == "unparsed import" {
                    reasons.push("unparsed import".to_string());
                } else {
                    reasons.push(format!("macro {name}"));
                }
            }
        }
        if let Some(values) = limits.get("attributes").and_then(|value| value.as_array()) {
            for value in values {
                if let Some(name) = value.as_str().filter(|name| !name.is_empty()) {
                    reasons.push(format!("attribute {name}"));
                }
            }
        }
        if let Some(values) = limits.get("imports").and_then(|value| value.as_array()) {
            for item in values {
                let path = item
                    .get("path")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let name = item
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                if path.starts_with("extern crate") || name.is_empty() {
                    reasons.push(format!("extern crate {path}"));
                    continue;
                }
                if matches!(name, "Some" | "Option") {
                    return None;
                }
                let first = path.split("::").next().unwrap_or("");
                let scope_shadowed = !path.starts_with("::")
                    && limits
                        .get("standard_namespace_bindings")
                        .and_then(|value| value.as_array())
                        .is_some_and(|bindings| {
                            bindings
                                .iter()
                                .any(|binding| binding.as_str() == Some(first))
                        });
                if scope_shadowed {
                    reasons.push(format!("shadowed standard import {path}"));
                    continue;
                }
                if self.rust_standard_import(file, path) {
                    continue;
                }
                match self.classify_rust_import(file, path, name) {
                    RustImportClass::Safe => {}
                    RustImportClass::Reject => return None,
                    RustImportClass::Unresolved(reason) => reasons.push(reason),
                }
            }
        }
        reasons.sort();
        reasons.dedup();
        Some(reasons)
    }

    /// Raw macro-argument calls are provisional until the actual import
    /// binding proves an expression macro. Follow explicit bindings before
    /// globs; an unresolved/custom/cyclic parent is not anyhow::ensure.
    fn rust_expression_macro_identity(
        &self,
        file: &str,
        path: &str,
        prelude: bool,
        seen: &mut std::collections::HashSet<(String, String)>,
    ) -> Option<String> {
        if seen.len() >= 32 || !seen.insert((file.to_string(), path.to_string())) {
            return None;
        }
        if let Some((owner, name)) = path.rsplit_once("::") {
            if matches!(owner, "std" | "core") && self.rust_standard_import(file, path) {
                return Some(format!("std::{name}"));
            }
            if owner == "anyhow"
                && name == "ensure"
                && self.anyhow_factory_files.contains(file)
                && !self.standard_namespace_is_shadowed(file, owner, false)
            {
                return Some("anyhow::ensure".to_string());
            }
            return None;
        }
        if let Some(bindings) = self
            .import_alias_sources_by_file
            .get(file)
            .and_then(|aliases| aliases.get(path))
        {
            let [(import_path, original)] = bindings.as_slice() else {
                return None;
            };
            if let Some(identity) =
                self.rust_expression_macro_identity(file, import_path, false, &mut seen.clone())
            {
                return Some(identity);
            }
            let modules = self
                .rust_module_files_for_path(file, import_path, original)
                .into_iter()
                .filter(|m| self.known_files.contains(m))
                .collect::<Vec<_>>();
            let [module] = modules.as_slice() else {
                return None;
            };
            return self.rust_expression_macro_identity(module, original, false, seen);
        }
        if let Some(globs) = self
            .import_globs_by_file
            .get(file)
            .filter(|g| !g.is_empty())
        {
            let mut identities = std::collections::HashSet::new();
            for glob in globs {
                let modules = self
                    .rust_module_files_for_module_path(file, glob)
                    .into_iter()
                    .filter(|m| self.known_files.contains(m))
                    .collect::<Vec<_>>();
                let [module] = modules.as_slice() else {
                    return None;
                };
                identities.insert(self.rust_expression_macro_identity(
                    module,
                    path,
                    false,
                    &mut seen.clone(),
                )?);
            }
            if identities.len() != 1 {
                return None;
            }
            return identities.into_iter().next();
        }
        // ensure is not in the Rust prelude. Only the standard expression
        // macros can use an unshadowed, import-free prelude binding.
        (prelude && path != "ensure").then(|| format!("std::{path}"))
    }

    fn classify_rust_import(&self, file: &str, path: &str, name: &str) -> RustImportClass {
        if matches!(name, "Some" | "Option") {
            return RustImportClass::Reject;
        }
        let Some(ids) = self
            .import_aliases_by_file
            .get(file)
            .and_then(|aliases| aliases.get(name))
        else {
            let rooted = matches!(
                path.trim().trim_start_matches("::").split("::").next(),
                Some("crate" | "self" | "super")
            );
            if rooted {
                return RustImportClass::Unresolved(format!("unresolved import {name}"));
            }
            return RustImportClass::Unresolved(format!("external import {path}"));
        };
        if ids.len() != 1 {
            return RustImportClass::Unresolved(format!("ambiguous import {name}"));
        }
        let Some(id) = ids.iter().next().copied() else {
            return RustImportClass::Unresolved(format!("unresolved import {name}"));
        };
        let Some(node) = self.by_id.get(&id) else {
            return RustImportClass::Unresolved(format!("unresolved import {name}"));
        };
        match node.label.as_str() {
            "Class" | "Enum" | "Function" | "Variable" | "Module" | "Type" | "AssocConst"
            | "AssocType" => RustImportClass::Safe,
            "Interface" => self.classify_trait_import(node, name),
            other => RustImportClass::Unresolved(format!("import {name} is {other}")),
        }
    }

    fn classify_trait_import(&self, node: &NodeLite, name: &str) -> RustImportClass {
        let trait_name = self
            .qname_for_id(node.id)
            .and_then(|qname| qname.rsplit("::").next())
            .unwrap_or(name);
        let method_qname = format!("{}::{trait_name}::as_ref", node.file_path);
        if let Some(receiver) = self.as_ref_receivers.get(&method_qname) {
            let compact: String = receiver.chars().filter(|ch| !ch.is_whitespace()).collect();
            if compact == "self"
                || compact == "mutself"
                || compact.starts_with("self:")
                || compact.starts_with("mutself:")
            {
                // A by-value adapter is probed before autoref reaches
                // Option::as_ref, so the declared payload is not the receiver.
                return RustImportClass::Reject;
            }
            if !compact.starts_with('&') {
                return RustImportClass::Unresolved(format!(
                    "trait import {name} has an unusual as_ref receiver"
                ));
            }
        }
        if self.open_traits.contains(&node.id) {
            return RustImportClass::Unresolved(format!("trait import {name} has supertraits"));
        }
        RustImportClass::Safe
    }

    fn resolve_usage_target(&self, edge: &ExtractedEdge, src_id: i64) -> Option<i64> {
        if edge
            .properties
            .get("ref_local_binding")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            return None;
        }
        let labels = if edge
            .properties
            .get("rust_type_reference")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            &TYPE_LABELS[..]
        } else {
            &USAGE_LABELS[..]
        };
        let name = edge
            .properties
            .get("ref_name")
            .and_then(|value| value.as_str())?;
        if name.is_empty() {
            return None;
        }
        if let Some(ref_path) = edge
            .properties
            .get("ref_path")
            .and_then(|value| value.as_str())
        {
            let referrer_file = self.file_of(src_id)?;
            let first_segment = ref_path.split("::").next().unwrap_or("");
            let module_files = self
                .rust_namespaces_by_file
                .get(referrer_file)
                .and_then(|aliases| aliases.get(first_segment))
                .map(|alias_files| {
                    rust_module_files_below_alias(
                        alias_files,
                        ref_path,
                        name,
                        self.rust_crate_roots.as_ref(),
                    )
                })
                .unwrap_or_else(|| self.rust_module_files_for_path(referrer_file, ref_path, name));
            let imported = self.rust_module_export_targets(&module_files, name, labels);
            if let [id] = imported.as_slice() {
                return Some(*id);
            }
            if !module_files
                .iter()
                .any(|file| self.known_files.contains(file))
            {
                return (labels == &USAGE_LABELS[..])
                    .then(|| {
                        self.resolve_associated_member(src_id, ref_path, name, &["EnumVariant"])
                    })
                    .flatten();
            }
            // Never discard syntactic qualification and retry this as an
            // unqualified same-file/import lookup.
            return None;
        }
        self.resolve_unique_with_imports(labels, name, src_id)
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
        let family = resolution_language_family(file_path);
        // The name index already bounds candidates to this method name.
        // Scanning every project node here made cross-file receiver calls
        // proportional to the whole graph for every individual edge.
        let candidates = self.defs_named(&["Method"], name);
        let mut matches = candidates
            .into_iter()
            .filter(|node| {
                resolution_language_family(&node.file_path) == family
                    && self
                        .qname_for_id(node.id)
                        .is_some_and(|qname| qname.ends_with(&suffix))
            })
            .map(|node| node.id);
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

    /// Named `std`/`core`/`alloc` imports are safe only when that path is the
    /// standard library. A local module, namespace alias, or import that binds
    /// the same name is not that crate, so classification continues.
    fn rust_standard_import(&self, file: &str, path: &str) -> bool {
        let trimmed = path.trim();
        let absolute = trimmed.starts_with("::");
        let first = trimmed
            .trim_start_matches("::")
            .split("::")
            .find(|part| !part.is_empty())
            .unwrap_or("");
        if !matches!(first, "std" | "core" | "alloc") {
            return false;
        }
        !self.standard_namespace_is_shadowed(file, first, absolute)
    }

    fn standard_namespace_is_shadowed(&self, file: &str, name: &str, absolute: bool) -> bool {
        if self
            .rust_namespaces_by_file
            .get(file)
            .is_some_and(|aliases| aliases.contains_key(name))
        {
            return true;
        }
        if self
            .import_alias_sources_by_file
            .get(file)
            .is_some_and(|aliases| aliases.get(name).is_some_and(|sources| !sources.is_empty()))
        {
            return true;
        }
        if absolute {
            return false;
        }
        let classified = self.rust_module_files_for_module_path(file, name);
        if classified
            .iter()
            .any(|candidate| self.known_files.contains(candidate))
        {
            return true;
        }
        self.by_name
            .get(name)
            .is_some_and(|nodes| nodes.iter().any(|node| node.file_path == file))
    }
}

enum RustImportClass {
    Safe,
    Reject,
    Unresolved(String),
}

fn push_unique_glob(out: &mut Vec<String>, path: &str) {
    let key = path
        .trim()
        .trim_end_matches('*')
        .trim_end_matches(':')
        .to_string();
    if !key.is_empty() && !out.iter().any(|existing| existing == &key) {
        out.push(key);
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
    let metadata = std::fs::symlink_metadata(&entry.abs_path)
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
    let Ok(md) = std::fs::symlink_metadata(&entry.abs_path) else {
        return;
    };
    // The skip row owns a link's identity. Never read its target into a
    // file_state: that mismatches lstat freshness and may leave the root.
    if !md.is_file() {
        return;
    }
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

    #[test]
    fn store_cow_only_paths_persist_discovery_filtered_file_identity() {
        let repo = setup_repo("filtered-hidden", RUST_SAMPLE);
        fs::write(repo.join(".gitattributes"), "*.bin binary\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        let options = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                ".gitattributes".to_string()
            ])),
            ..IndexOptions::default()
        };
        index_with_options(&mut store, &repo, "p", &options).unwrap();
        let skip = store
            .get_index_skip("p", ".gitattributes")
            .unwrap()
            .expect("filtered Delta path must have a persisted identity");
        assert_eq!(skip.reason, "discovery_filtered");
        assert_eq!(skip.size, 13);
        assert!(skip.file_id.is_some());
        let state = store
            .get_file_state("p", ".gitattributes")
            .unwrap()
            .expect("filtered Delta path must retain a hash-backed file identity");
        assert_eq!(state.size, 13);
        assert_eq!(
            state.sha256,
            file_state::sha256_hex(b"*.bin binary\n"),
            "metadata-only drift must be recoverable through content identity"
        );
        let _ = fs::remove_dir_all(repo);
    }

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
        fs::write(repo.join("lib/helper.dart"), "int do_it() => 1;\n").unwrap();
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
        let imports = store
            .outgoing_edges(source.id, Some("IMPORTS"), 10)
            .unwrap();
        assert!(
            imports
                .iter()
                .any(|edge| edge.target_id == nested_module.id),
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
    fn index_version_upgrade_does_not_copy_base_into_delta() {
        let repo = setup_repo("version-upgrade-overlay", "pub fn dirty_file() {}\n");
        fs::write(repo.join("src/clean.rs"), "pub fn clean_base_file() {}\n").unwrap();
        let stores = repo.with_extension("stores");
        fs::create_dir_all(&stores).unwrap();
        let base_path = stores.join("base.db");
        let delta_path = stores.join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, &repo, "test").unwrap();
        }
        let options = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from(["src/lib.rs".to_string()])),
            ..IndexOptions::default()
        };
        {
            let mut delta = Store::open(&delta_path).unwrap();
            index_with_options(&mut delta, &repo, "test", &options).unwrap();
            for mut state in delta.list_workspace_states().unwrap() {
                state.indexer_version = "greppy-indexer-v5".into();
                delta.upsert_workspace_state(&state).unwrap();
            }
        }
        let visibility =
            greppy_store::VisibilityIndex::new(["src/lib.rs".to_string()], Vec::<String>::new())
                .unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        assert!(overlay
            .list_file_states("test")
            .unwrap()
            .iter()
            .any(|state| state.rel_path == "src/clean.rs"));
        assert_eq!(overlay.list_private_file_states("test").unwrap().len(), 1);
        let report = index_with_options(&mut overlay, &repo, "test", &options).unwrap();
        assert_eq!(
            report.files_indexed, 1,
            "only the old Delta may be migrated"
        );
        let private = overlay.list_private_file_states("test").unwrap();
        assert_eq!(private.len(), 1);
        assert_eq!(private[0].rel_path, "src/lib.rs");
        let nodes = overlay
            .list_nodes_by_label("test", "Function", 100)
            .unwrap();
        assert!(nodes.iter().any(|node| node.name == "dirty_file"));
        assert!(nodes.iter().any(|node| node.name == "clean_base_file"));
        // A current Base row must not hide lost compatibility metadata in an
        // existing private layer and wrongly permit incremental reuse.
        overlay
            .conn()
            .execute("DELETE FROM main.workspace_state", [])
            .unwrap();
        assert!(!overlay.list_workspace_states().unwrap().is_empty());
        assert!(overlay.list_private_workspace_states().unwrap().is_empty());
        let repaired = index_with_options(&mut overlay, &repo, "test", &options).unwrap();
        assert_eq!(
            repaired.files_indexed, 1,
            "missing Delta metadata requires migration"
        );
        assert_eq!(overlay.list_private_file_states("test").unwrap().len(), 1);
        drop(overlay);
        let _ = fs::remove_dir_all(repo);
        let _ = fs::remove_dir_all(stores);
    }

    #[test]
    fn overlay_delta_resolves_rust_usage_through_base_reexports() {
        let repo = setup_multifile_repo(
            "overlay-base-reexport",
            "mod alias_chain; mod bare_glob; mod business_os; mod channels; mod decoys; mod glob_channels; mod parent; mod renamed_channels; mod super_exports;\n",
            "// fixture placeholder\n",
        );
        fs::create_dir_all(repo.join("src/alias_chain")).unwrap();
        fs::create_dir_all(repo.join("src/bare_glob")).unwrap();
        fs::create_dir_all(repo.join("src/business_os")).unwrap();
        fs::create_dir_all(repo.join("src/channels")).unwrap();
        fs::create_dir_all(repo.join("src/glob_channels")).unwrap();
        fs::create_dir_all(repo.join("src/parent/child")).unwrap();
        fs::create_dir_all(repo.join("src/renamed_channels")).unwrap();
        fs::create_dir_all(repo.join("src/super_exports")).unwrap();
        fs::write(
            repo.join("src/business_os/mod.rs"),
            "pub mod store; pub use crate::super_exports::*;\n",
        )
        .unwrap();
        let caller_source =
            "use crate::{alias_chain, channels, glob_channels, parent::child, renamed_channels};\n\
use crate::bare_glob::*;\n\
use crate::alias_chain::outer;\n\
use super::*;\n\
pub fn grouped_caller() { let selected = channels::target; selected(); }\n\
pub fn renamed_caller() { let selected = renamed_channels::renamed; selected(); }\n\
pub fn glob_caller() { let selected = glob_channels::target; selected(); }\n\
pub fn super_glob_caller() { let selected = child::target; selected(); }\n\
pub fn alias_chain_caller() { let selected = alias_chain::outer; selected(); }\n\
pub fn delta_crate_glob_caller() { let selected = bare_target; selected(); }\n\
pub fn delta_super_glob_caller() { let selected = super_target; selected(); }\n\
pub fn shadow_target() {}\n\
pub fn local_shadow_caller() { let selected = shadow_target; selected(); }\n\
pub struct Shadow; impl Shadow { pub fn shadow_target() {} }\n\
pub fn imported_alias_caller() { let selected = outer; selected(); }\n";
        fs::write(repo.join("src/business_os/store.rs"), caller_source).unwrap();
        fs::write(
            repo.join("src/alias_chain/mod.rs"),
            "mod sub; pub use sub::target as middle; pub use middle as outer;\n",
        )
        .unwrap();
        fs::write(repo.join("src/alias_chain/sub.rs"), "pub fn target() {}\n").unwrap();
        fs::write(
            repo.join("src/bare_glob/mod.rs"),
            "mod command; pub use command::*;\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/bare_glob/command.rs"),
            "pub fn bare_target() {}\npub fn shadow_target() {}\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/channels/mod.rs"),
            "mod command; pub use command::{first, target};\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/channels/command.rs"),
            "pub fn first() {}\npub fn target() {}\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/glob_channels/mod.rs"),
            "mod command; pub use command::*;\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/glob_channels/command.rs"),
            "pub fn target() {}\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/parent/mod.rs"),
            "pub fn target() {}\npub mod child;\n",
        )
        .unwrap();
        fs::write(repo.join("src/parent/child/mod.rs"), "pub use super::*;\n").unwrap();
        fs::write(
            repo.join("src/renamed_channels/mod.rs"),
            "mod command; pub use command::target as renamed;\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/renamed_channels/command.rs"),
            "pub fn target() {}\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/super_exports/mod.rs"),
            "pub fn super_target() {}\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/decoys.rs"),
            "pub fn bare_target() {}\npub fn super_target() {}\n",
        )
        .unwrap();

        let stores = repo.with_extension("overlay-base-reexport-stores");
        fs::create_dir_all(&stores).unwrap();
        let base_path = stores.join("base.db");
        let delta_path = stores.join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, &repo, "test").unwrap();
        }

        fs::write(
            repo.join("src/business_os/store.rs"),
            format!("{caller_source}// dirty worktree comment\n"),
        )
        .unwrap();
        let dirty_path = "src/business_os/store.rs".to_string();
        let visibility =
            greppy_store::VisibilityIndex::new([dirty_path.clone()], Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        index_with_options(
            &mut overlay,
            &repo,
            "test",
            &IndexOptions {
                only_paths: Some(std::collections::BTreeSet::from([dirty_path])),
                ..IndexOptions::default()
            },
        )
        .unwrap();
        rebuild_overlay_edges(&mut overlay, "test").unwrap();

        for (target_qname, caller_qname, reexport_kind) in [
            (
                "src/channels/command.rs::Function::target",
                "src/business_os/store.rs::Function::grouped_caller",
                "grouped",
            ),
            (
                "src/renamed_channels/command.rs::Function::target",
                "src/business_os/store.rs::Function::renamed_caller",
                "renamed",
            ),
            (
                "src/glob_channels/command.rs::Function::target",
                "src/business_os/store.rs::Function::glob_caller",
                "glob",
            ),
            (
                "src/parent/mod.rs::Function::target",
                "src/business_os/store.rs::Function::super_glob_caller",
                "transitive super glob",
            ),
            (
                "src/alias_chain/sub.rs::Function::target",
                "src/business_os/store.rs::Function::alias_chain_caller",
                "explicit alias chain",
            ),
            (
                "src/bare_glob/command.rs::Function::bare_target",
                "src/business_os/store.rs::Function::delta_crate_glob_caller",
                "Delta crate glob",
            ),
            (
                "src/super_exports/mod.rs::Function::super_target",
                "src/business_os/store.rs::Function::delta_super_glob_caller",
                "Delta super glob",
            ),
            (
                "src/business_os/store.rs::Function::shadow_target",
                "src/business_os/store.rs::Function::local_shadow_caller",
                "local item shadows Base glob",
            ),
            (
                "src/alias_chain/sub.rs::Function::target",
                "src/business_os/store.rs::Function::imported_alias_caller",
                "explicit import of chained reexport",
            ),
        ] {
            let target = overlay
                .get_node_by_qname("test", target_qname)
                .unwrap()
                .expect("Base target remains visible");
            let caller = overlay
                .get_node_by_qname("test", caller_qname)
                .unwrap()
                .expect("dirty Delta caller");
            let incoming = overlay
                .incoming_edges(target.id, Some("USAGE"), 10)
                .unwrap();
            assert!(
                incoming.iter().any(|edge| edge.source_id == caller.id),
                "dirty Delta usage must resolve through the Base {reexport_kind} reexport: {incoming:?}"
            );
        }

        drop(overlay);
        let _ = fs::remove_dir_all(repo);
        let _ = fs::remove_dir_all(stores);
    }

    #[test]
    fn index_version_upgrade_preserves_sparse_layer_and_then_reuses_it() {
        let repo = setup_repo("version-upgrade-sparse", "pub fn changed_path() {}\n");
        fs::write(repo.join("src/retained.rs"), "pub fn retained_path() {}\n").unwrap();
        fs::write(repo.join("src/outside.rs"), "pub fn outside_layer() {}\n").unwrap();
        fs::write(repo.join(".gitattributes"), "*.bin binary\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        let initial = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                "src/lib.rs".to_string(),
                "src/retained.rs".to_string(),
                ".gitattributes".to_string(),
            ])),
            ..IndexOptions::default()
        };
        index_with_options(&mut store, &repo, "test", &initial).unwrap();
        assert!(store
            .get_index_skip("test", ".gitattributes")
            .unwrap()
            .is_some());
        fs::remove_file(repo.join(".gitattributes")).unwrap();
        for mut state in store.list_workspace_states().unwrap() {
            state.indexer_version = "greppy-indexer-v5".into();
            store.upsert_workspace_state(&state).unwrap();
        }
        let narrow = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from(["src/lib.rs".to_string()])),
            ..IndexOptions::default()
        };
        let upgraded = index_with_options(&mut store, &repo, "test", &narrow).unwrap();
        assert_eq!(
            upgraded.files_indexed, 2,
            "all retained layer files need migration"
        );
        assert_eq!(upgraded.files_skipped, 0);
        assert!(store
            .get_index_skip("test", ".gitattributes")
            .unwrap()
            .is_none());
        assert!(store
            .get_file_state("test", ".gitattributes")
            .unwrap()
            .is_none());
        let nodes = store.list_nodes_by_label("test", "Function", 100).unwrap();
        assert!(nodes.iter().any(|node| node.name == "retained_path"));
        assert!(!nodes.iter().any(|node| node.name == "outside_layer"));
        let unchanged = index_with_options(&mut store, &repo, "test", &initial).unwrap();
        assert_eq!(unchanged.files_indexed, 0, "migration must run only once");
        assert_eq!(unchanged.files_skipped, 2);
        let _ = fs::remove_dir_all(repo);
    }
    #[test]
    fn js_ts_method_identity_upgrade_repairs_unchanged_sparse_graph_once() {
        let repo = setup_repo("computed-method-upgrade", "pub fn keep() {}\n");
        fs::write(repo.join("stream.ts"), "function tick() {}\nclass Stream { async *[Symbol.asyncIterator]() { await new Promise(resolve => resolve()); tick(); } }\n").unwrap();
        fs::write(repo.join("outside.ts"), "function outside() {}\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        let initial = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                "src/lib.rs".into(),
                "stream.ts".into(),
            ])),
            ..IndexOptions::default()
        };
        index_with_options(&mut store, &repo, "test", &initial).unwrap();
        let qname = "stream.ts::Stream::[Symbol.asyncIterator]";
        let method = store
            .list_nodes_by_label("test", "Method", 100)
            .unwrap()
            .into_iter()
            .find(|node| node.qualified_name == qname)
            .expect("cold index must persist computed method");
        // v9 retained source fingerprints, but never emitted this definition.
        // Keep the file state unchanged so only the version upgrade can repair it.
        store.delete_node(method.id).unwrap();
        for mut state in store.list_workspace_states().unwrap() {
            state.indexer_version = state.indexer_version.replacen(
                greppy_core::INDEXER_VERSION_BASE,
                "greppy-indexer-v9",
                1,
            );
            store.upsert_workspace_state(&state).unwrap();
        }
        let narrow = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from(["src/lib.rs".into()])),
            ..IndexOptions::default()
        };
        let upgraded = index_with_options(&mut store, &repo, "test", &narrow).unwrap();
        assert_eq!(
            upgraded.files_indexed, 2,
            "retained TS file must migrate despite narrow request and identical bytes"
        );
        assert!(store
            .list_nodes_by_label("test", "Method", 100)
            .unwrap()
            .iter()
            .any(|node| node.qualified_name == qname));
        assert!(store
            .get_file_state("test", "outside.ts")
            .unwrap()
            .is_none());
        let unchanged = index_with_options(&mut store, &repo, "test", &initial).unwrap();
        assert_eq!(unchanged.files_indexed, 0, "migration must run only once");
        assert_eq!(unchanged.files_skipped, 2);
        let _ = fs::remove_dir_all(repo);
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
            only_paths: None,
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
    fn index_progress_reports_real_file_work_before_embeddings() {
        let repo = setup_repo("index-progress", "pub fn keep_me() {}\n");
        fs::write(repo.join("src/other.rs"), "pub fn other() {}\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        let mut events = Vec::new();

        let report = index_with_options_and_progress(
            &mut store,
            &repo,
            "test",
            &IndexOptions::default(),
            &mut |event| events.push(event),
        )
        .unwrap();

        assert_eq!(report.files_indexed, 2);
        assert_eq!(events.first().unwrap().phase, "discovering_files");
        assert!(events.iter().any(|event| {
            event.phase == "extracting_files"
                && event.total_files == 2
                && event.completed_files == 2
        }));
        assert!(events.iter().any(|event| {
            event.phase == "writing_graph" && event.total_files == 2 && event.completed_files == 2
        }));
        assert_eq!(events.last().unwrap().phase, "finalizing_graph");

        let inventory_count = store.list_private_file_states("test").unwrap().len();
        assert!(
            inventory_count >= 2,
            "both supported source files must be retained"
        );
        let mut previous = store.list_private_workspace_states().unwrap().remove(0);
        previous.indexer_version = "incompatible-progress-test".into();
        store.upsert_workspace_state(&previous).unwrap();
        events.clear();
        index_with_options_and_progress(
            &mut store,
            &repo,
            "test",
            &IndexOptions::default(),
            &mut |event| events.push(event),
        )
        .unwrap();
        let cleanup = events
            .iter()
            .filter(|event| event.phase == "removing_previous_graph")
            .map(|event| (event.completed_files, event.total_files))
            .collect::<Vec<_>>();
        assert_eq!(
            cleanup,
            (0..=inventory_count)
                .map(|done| (done, inventory_count))
                .collect::<Vec<_>>()
        );
        let preparation = events
            .iter()
            .position(|event| event.phase == "preparing_graph_inventory")
            .unwrap();
        let cleanup_done = events
            .iter()
            .rposition(|event| event.phase == "removing_previous_graph")
            .unwrap();
        let classification = events
            .iter()
            .position(|event| event.phase == "classifying_files")
            .unwrap();
        assert!(preparation < cleanup_done && cleanup_done < classification);
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
    fn rust_reexport_migration_preserves_matching_discovery_scope() {
        assert!(is_rust_reexport_migration(
            "greppy-indexer-v6;discover_scope=tracked",
            "greppy-indexer-v7;discover_scope=tracked"
        ));
        assert!(!is_rust_reexport_migration(
            "greppy-indexer-v6;discover_scope=tracked",
            "greppy-indexer-v7;discover_scope=all"
        ));
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
    fn effect_fn_generator_call_persists_incoming_edge_from_exported_binding() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("routing.ts"),
            r#"
import { Effect } from "effect";
const resolveGatewayProviderForModel = Effect.fn("resolveGatewayProviderForModel")(
    function* (input: { model: string }) { return input.model; },
);
export const resolveGatewayRoutedEnvironment = Effect.fn("resolveGatewayRoutedEnvironment")(
    function* (input: { model: string }) {
        const gatewayProvider = input.model.length > 0
            ? yield* resolveGatewayProviderForModel({ model: input.model })
            : undefined;
        return gatewayProvider;
    },
);
const plainValue = 42;
const effectValue = Effect.gen(function* () { return 42; });
export function invalidCalls() { plainValue(); effectValue(); }
"#,
        )
        .unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        let target = store
            .get_node_by_qname(
                "test",
                "routing.ts::Function::resolveGatewayProviderForModel",
            )
            .unwrap()
            .expect("private Effect.fn binding must exist");
        let caller = store
            .get_node_by_qname(
                "test",
                "routing.ts::Function::resolveGatewayRoutedEnvironment",
            )
            .unwrap()
            .expect("exported Effect.fn binding must exist");
        let incoming = store.incoming_edges(target.id, Some("CALLS"), 10).unwrap();
        assert_eq!(
            incoming.len(),
            1,
            "expected one persisted direct caller: {incoming:?}"
        );
        assert_eq!(incoming[0].source_id, caller.id);
        fs::write(repo.path().join("unrelated.py"), "def retained(): pass\n").unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        let retained = store
            .get_node_by_qname("test", "unrelated.py::Function::retained")
            .unwrap()
            .unwrap();
        // Recreate the v7 binding identities while keeping identical file bytes.
        store
            .conn()
            .execute("DELETE FROM main.edges WHERE source_id = ?1", [caller.id])
            .unwrap();
        for node in [&target, &caller] {
            store
                .update_node_identity(
                    node.id,
                    "Variable",
                    &node.qualified_name.replace("::Function::", "::Variable::"),
                )
                .unwrap();
        }
        for mut state in store.list_workspace_states().unwrap() {
            state.indexer_version = "greppy-indexer-v7".into();
            store.upsert_workspace_state(&state).unwrap();
        }
        let repaired = index(&mut store, repo.path(), "test").unwrap();
        assert_eq!(
            repaired.files_indexed, 2,
            "v9 re-extracts every retained source in the incompatible cache"
        );
        let restored = store
            .get_node_by_qname("test", &target.qualified_name)
            .unwrap()
            .unwrap();
        assert_ne!(restored.id, target.id, "v9 replaces old declaration nodes");
        assert!(
            store
                .get_node_by_qname("test", &retained.qualified_name)
                .unwrap()
                .is_some(),
            "cross-language definitions survive full cache refresh"
        );
        let incoming = store
            .incoming_edges(restored.id, Some("CALLS"), 10)
            .unwrap();
        assert_eq!(incoming.len(), 1);
        let restored_caller = store
            .get_node_by_qname("test", &caller.qualified_name)
            .unwrap()
            .unwrap();
        assert_eq!(incoming[0].source_id, restored_caller.id);
        assert_eq!(
            index(&mut store, repo.path(), "test")
                .unwrap()
                .files_indexed,
            0,
            "migration must run once, without manual full reindex"
        );

        for name in ["plainValue", "effectValue"] {
            let value = store
                .get_node_by_qname("test", &format!("routing.ts::Variable::{name}"))
                .unwrap()
                .expect("ordinary value must remain a Variable");
            assert!(
                store
                    .incoming_edges(value.id, Some("CALLS"), 10)
                    .unwrap()
                    .is_empty(),
                "noncallable values must not resolve as call targets"
            );
        }
    }

    fn rust_discovery_filtered_fixture() -> (tempfile::TempDir, IndexOptions) {
        let repo = tempfile::tempdir().unwrap();
        fs::create_dir_all(repo.path().join("node_modules/package")).unwrap();
        fs::write(
            repo.path().join("node_modules/package/vendor.rs"),
            "pub struct Vendor { pub load_data: String }\n",
        )
        .unwrap();
        fs::write(
            repo.path().join("lib.rs"),
            "pub fn target() {}\npub fn caller() { target(); }\n",
        )
        .unwrap();
        let options = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                "lib.rs".to_string(),
                "node_modules/package/vendor.rs".to_string(),
            ])),
            ..IndexOptions::default()
        };
        (repo, options)
    }

    #[test]
    fn rust_usage_recovery_respects_current_discovery_filtered_identity() {
        let (repo, options) = rust_discovery_filtered_fixture();
        let mut store = Store::open_memory().unwrap();
        assert_eq!(
            index_with_options(&mut store, repo.path(), "test", &options)
                .unwrap()
                .files_indexed,
            1
        );
        let skip = store
            .get_index_skip("test", "node_modules/package/vendor.rs")
            .unwrap()
            .unwrap();
        assert_eq!(skip.reason, "discovery_filtered");
        let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 100).unwrap());
        let states = store.list_file_states("test").unwrap();
        let raw = format!("{:?}", store.list_raw_edges("test").unwrap());
        for case in 0..4 {
            let mut stale = skip.clone();
            match case {
                0 => stale.reason = "parse_failed".into(),
                1 => {
                    stale.last_indexed_generation = stale.last_indexed_generation.saturating_sub(1)
                }
                2 => stale.size += 1,
                _ => stale.mtime_ns += 1,
            }
            store.upsert_index_skip(&stale).unwrap();
            let error = recover_persisted_rust_usages(&mut store, "test", repo.path()).unwrap_err();
            assert!(
                error.to_string().contains("declared field facts"),
                "{error}"
            );
            assert_eq!(
                nodes,
                format!("{:?}", store.list_nodes("test", "", "", 0, 100).unwrap())
            );
            assert_eq!(states, store.list_file_states("test").unwrap());
            assert_eq!(raw, format!("{:?}", store.list_raw_edges("test").unwrap()));
        }
        store.upsert_index_skip(&skip).unwrap();
        let qname = "node_modules/package/vendor.rs::Class::Vendor::load_data";
        store.conn().execute(
            "INSERT INTO main.nodes(project,label,name,qualified_name,file_path,start_line,end_line,properties) VALUES('test','Field','load_data',?1,?2,1,1,'{\"return_type\":\"Wrong\"}')",
            [qname, skip.rel_path.as_str()],
        ).unwrap();
        assert!(
            recover_persisted_rust_usages(&mut store, "test", repo.path())
                .unwrap_err()
                .to_string()
                .contains("declared field facts")
        );
        store
            .conn()
            .execute("DELETE FROM main.nodes WHERE qualified_name=?1", [qname])
            .unwrap();
        recover_persisted_rust_usages(&mut store, "test", repo.path()).unwrap();
        assert_eq!(
            recover_persisted_rust_usages(&mut store, "test", repo.path()).unwrap(),
            0
        );
        assert_eq!(
            nodes,
            format!("{:?}", store.list_nodes("test", "", "", 0, 100).unwrap())
        );
        assert_eq!(states, store.list_file_states("test").unwrap());
    }

    #[test]
    fn rust_usage_recovery_respects_filtered_delta_and_preserves_base() {
        let (repo, _) = rust_discovery_filtered_fixture();
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, repo.path(), "test").unwrap();
            assert!(base
                .get_file_state("test", "node_modules/package/vendor.rs")
                .unwrap()
                .is_none());
            base.conn().execute_batch("DELETE FROM raw_edges WHERE edge_type='CALLS'; DELETE FROM edges WHERE edge_type='CALLS';").unwrap();
            base.conn()
                .execute(
                    "DELETE FROM main.schema_meta WHERE key=?1",
                    [RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
        }
        let bytes = fs::read(&base_path).unwrap();
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        let options = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                "node_modules/package/vendor.rs".to_string(),
            ])),
            ..IndexOptions::default()
        };
        assert_eq!(
            index_with_options(&mut overlay, repo.path(), "test", &options)
                .unwrap()
                .files_indexed,
            0
        );
        assert_eq!(
            overlay
                .get_index_skip("test", "node_modules/package/vendor.rs")
                .unwrap()
                .unwrap()
                .reason,
            "discovery_filtered"
        );
        recover_persisted_rust_usages(&mut overlay, "test", repo.path()).unwrap();
        rebuild_visible_overlay_edges(&mut overlay, "test").unwrap();
        let target = overlay
            .get_node_by_qname("test", "lib.rs::Function::target")
            .unwrap()
            .unwrap();
        let caller = overlay
            .get_node_by_qname("test", "lib.rs::Function::caller")
            .unwrap()
            .unwrap();
        assert!(overlay
            .incoming_edges(target.id, Some("CALLS"), 10)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id));
        assert_eq!(
            recover_persisted_rust_usages(&mut overlay, "test", repo.path()).unwrap(),
            0
        );
        assert!(overlay.list_delta_raw_edges("test").unwrap().is_empty());
        assert_eq!(
            overlay.list_private_file_states("test").unwrap().len(),
            1,
            "only filtered identity belongs to Delta"
        );
        drop(overlay);
        assert_eq!(bytes, fs::read(&base_path).unwrap());
    }

    fn jsx_discovery_filtered_fixture() -> (tempfile::TempDir, IndexOptions) {
        let repo = tempfile::tempdir().unwrap();
        fs::create_dir_all(repo.path().join("node_modules/package")).unwrap();
        fs::write(
            repo.path().join("node_modules/package/media-controls.js"),
            "(() => { class MediaControls { render() { return 1; } } new MediaControls(); })();\n",
        )
        .unwrap();
        fs::write(
            repo.path().join("boundary.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(repo.path().join("view.tsx"), "import { Boundary } from './boundary';\nexport function Render() { return <Boundary />; }\n").unwrap();
        let options = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                "boundary.tsx".to_string(),
                "view.tsx".to_string(),
                "node_modules/package/media-controls.js".to_string(),
            ])),
            ..IndexOptions::default()
        };
        (repo, options)
    }

    #[test]
    fn jsx_usage_recovery_respects_current_discovery_filtered_identity() {
        let (repo, options) = jsx_discovery_filtered_fixture();
        let mut store = Store::open_memory().unwrap();
        assert_eq!(
            index_with_options(&mut store, repo.path(), "test", &options)
                .unwrap()
                .files_indexed,
            2
        );
        let skip = store
            .get_index_skip("test", "node_modules/package/media-controls.js")
            .unwrap()
            .unwrap();
        assert_eq!(skip.reason, "discovery_filtered");
        assert!(store
            .get_node_by_qname(
                "test",
                "node_modules/package/media-controls.js::Class::MediaControls"
            )
            .unwrap()
            .is_none());
        store.conn().execute_batch(
            "DELETE FROM main.raw_edges WHERE edge_type='USAGE'; DELETE FROM main.edges WHERE edge_type='USAGE';"
        ).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_KEY],
            )
            .unwrap();
        let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
        let states = format!("{:?}", store.list_file_states("test").unwrap());
        // An unrelated failure or stale skip row cannot excuse missing definitions.
        for case in 0..4 {
            let mut stale = skip.clone();
            match case {
                0 => stale.reason = "parse_failed".into(),
                1 => {
                    stale.last_indexed_generation = stale.last_indexed_generation.saturating_sub(1)
                }
                2 => stale.size += 1,
                _ => stale.mtime_ns += 1,
            }
            store.upsert_index_skip(&stale).unwrap();
            let error =
                recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap_err();
            assert!(error.to_string().contains("MediaControls"), "{error}");
            assert!(!js_ts_usages_repaired(&store).unwrap());
            assert_eq!(
                nodes,
                format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
            );
            assert_eq!(
                states,
                format!("{:?}", store.list_file_states("test").unwrap())
            );
        }
        store.upsert_index_skip(&skip).unwrap();
        let filtered_qname = "node_modules/package/media-controls.js::Class::MediaControls";
        store.conn().execute(
            "INSERT INTO main.nodes(project,label,name,qualified_name,file_path,start_line,end_line,properties) VALUES('test','Class','MediaControls',?1,?2,999,999,'{}')",
            [filtered_qname, skip.rel_path.as_str()],
        ).unwrap();
        let error = recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap_err();
        assert!(error.to_string().contains("MediaControls"), "{error}");
        assert!(!js_ts_usages_repaired(&store).unwrap());
        assert_eq!(
            store
                .get_node_by_qname("test", filtered_qname)
                .unwrap()
                .unwrap()
                .start_line,
            999
        );
        store
            .conn()
            .execute(
                "DELETE FROM main.nodes WHERE project='test' AND qualified_name=?1",
                [filtered_qname],
            )
            .unwrap();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());

        assert!(!recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());
        let boundary = store
            .get_node_by_qname("test", "boundary.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .incoming_edges(boundary.id, Some("USAGE"), 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            nodes,
            format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        assert_eq!(
            states,
            format!("{:?}", store.list_file_states("test").unwrap())
        );
    }

    #[test]
    fn jsx_usage_recovery_respects_filtered_delta_and_preserves_base() {
        let (repo, _) = jsx_discovery_filtered_fixture();
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, repo.path(), "test").unwrap();
            assert!(base
                .get_file_state("test", "node_modules/package/media-controls.js")
                .unwrap()
                .is_none());
            base.conn().execute_batch("DELETE FROM raw_edges WHERE edge_type='USAGE'; DELETE FROM edges WHERE edge_type='USAGE';").unwrap();
            base.conn()
                .execute(
                    "DELETE FROM main.schema_meta WHERE key=?1",
                    [JS_TS_USAGE_REPAIR_KEY],
                )
                .unwrap();
        }
        let bytes = fs::read(&base_path).unwrap();
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        let options = IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                "node_modules/package/media-controls.js".to_string(),
            ])),
            ..IndexOptions::default()
        };
        assert_eq!(
            index_with_options(&mut overlay, repo.path(), "test", &options)
                .unwrap()
                .files_indexed,
            0
        );
        let skip = overlay
            .get_index_skip("test", "node_modules/package/media-controls.js")
            .unwrap()
            .unwrap();
        assert_eq!(skip.reason, "discovery_filtered");
        assert!(js_ts_usages_repaired(&overlay).unwrap());
        assert!(!recover_persisted_js_ts_usages(&mut overlay, "test", repo.path()).unwrap());
        let boundary = overlay
            .get_node_by_qname("test", "boundary.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        assert_eq!(
            overlay
                .incoming_edges(boundary.id, Some("USAGE"), 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(bytes, fs::read(&base_path).unwrap());
    }

    #[test]
    fn jsx_usage_recovery_indexes_original_observe_fixture_without_manual_repair() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("observe_choices_test.cjs"),
            include_str!("../../../bench/web_study/basic_fixture/observe_choices_test.cjs"),
        )
        .unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let mut store = Store::open(&scratch.path().join("graph.db")).unwrap();
        assert_eq!(
            index(&mut store, repo.path(), "test")
                .unwrap()
                .files_indexed,
            1
        );
        assert!(js_ts_usages_repaired(&store).unwrap());
        assert_eq!(
            index(&mut store, repo.path(), "test")
                .unwrap()
                .files_indexed,
            0
        );
        let method = store
            .get_node_by_qname("test", "observe_choices_test.cjs::Function::getAttribute")
            .unwrap()
            .unwrap();
        assert_eq!(method.start_line, 65);
    }

    #[test]
    fn js_ts_usage_owner_recovery_migrates_completed_named_callback_cache() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("boundary.ts"),
            "export const Boundary = 42; export function helper() { return 42; }\n",
        )
        .unwrap();
        fs::write(
            repo.path().join("app.ts"),
            r#"
import { Effect } from "effect";
import { Boundary, helper } from "./boundary";
export const make = Effect.gen(function* PreviewManagerMake() { return helper() + Boundary; });
export const exposed = function Internal() { return helper() + Boundary; };
function* UnsupportedDeclaration() { yield Boundary; }
module.exports = function ExportedInternal() { return helper() + Boundary; };
"#,
        )
        .unwrap();
        fs::write(
            repo.path().join("other.ts"),
            "export const Boundary = 7; export function helper() { return 7; }\n",
        )
        .unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, repo.path(), "test").unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [JS_TS_USAGE_REPAIR_KEY],
                )
                .unwrap();
            base.conn().execute_batch("DELETE FROM raw_edges WHERE edge_type IN ('USAGE','CALLS'); DELETE FROM edges WHERE edge_type IN ('USAGE','CALLS'); INSERT OR REPLACE INTO schema_meta VALUES('greppy.js_ts_usage_repair_v2','complete'); INSERT OR REPLACE INTO schema_meta VALUES('greppy.js_ts_usage_repair_v3','complete');").unwrap();
        }
        let base_bytes = fs::read(&base_path).unwrap();
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        overlay.conn().execute_batch("INSERT OR REPLACE INTO main.schema_meta VALUES('greppy.js_ts_usage_repair_v2','complete'); INSERT OR REPLACE INTO main.schema_meta VALUES('greppy.js_ts_usage_repair_v3','complete');").unwrap();
        let private_path = scratch.path().join("private.db");
        let mut private = Store::open(&private_path).unwrap();
        index(&mut private, repo.path(), "test").unwrap();
        private
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_KEY],
            )
            .unwrap();
        private.conn().execute_batch("DELETE FROM raw_edges WHERE edge_type IN ('USAGE','CALLS'); DELETE FROM edges WHERE edge_type IN ('USAGE','CALLS'); INSERT OR REPLACE INTO schema_meta VALUES('greppy.js_ts_usage_repair_v2','complete'); INSERT OR REPLACE INTO schema_meta VALUES('greppy.js_ts_usage_repair_v3','complete');").unwrap();
        for store in [&mut overlay, &mut private] {
            let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
            let states = format!("{:?}", store.list_file_states("test").unwrap());
            assert!(!js_ts_usages_repaired(store).unwrap());
            assert!(recover_persisted_js_ts_usages(store, "test", repo.path()).unwrap());
            assert!(!recover_persisted_js_ts_usages(store, "test", repo.path()).unwrap());
            assert_eq!(
                nodes,
                format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
            );
            assert_eq!(
                states,
                format!("{:?}", store.list_file_states("test").unwrap())
            );
            let boundary = store
                .get_node_by_qname("test", "boundary.ts::Variable::Boundary")
                .unwrap()
                .unwrap();
            let incoming = store
                .incoming_edges(boundary.id, Some("USAGE"), 100)
                .unwrap();
            // The generator declaration and CommonJS callback both use the
            // file owner. Graph edges are unique by owner, target and kind.
            assert_eq!(incoming.len(), 3, "{incoming:?}");
            let unrelated = store
                .get_node_by_qname("test", "other.ts::Variable::Boundary")
                .unwrap()
                .unwrap();
            assert!(store
                .incoming_edges(unrelated.id, Some("USAGE"), 100)
                .unwrap()
                .is_empty());
            for suffix in ["Variable::make", "Function::exposed", "__file__"] {
                let owner = store
                    .get_node_by_qname("test", &format!("app.ts::{suffix}"))
                    .unwrap()
                    .unwrap();
                assert!(
                    incoming.iter().any(|edge| edge.source_id == owner.id),
                    "{suffix}: {incoming:?}"
                );
            }
            let helper = store
                .get_node_by_qname("test", "boundary.ts::Function::helper")
                .unwrap()
                .unwrap();
            let callers = store.incoming_edges(helper.id, Some("CALLS"), 100).unwrap();
            assert_eq!(callers.len(), 3, "{callers:?}");
            for (suffix, line) in [
                ("Variable::make", 4),
                ("Function::exposed", 5),
                ("__file__", 7),
            ] {
                let owner = store
                    .get_node_by_qname("test", &format!("app.ts::{suffix}"))
                    .unwrap()
                    .unwrap();
                let caller = callers
                    .iter()
                    .find(|edge| edge.source_id == owner.id)
                    .unwrap();
                assert_eq!(caller.properties["line"], line, "{caller:?}");
                assert!(store
                    .outgoing_edges(owner.id, Some("CALLS"), 100)
                    .unwrap()
                    .iter()
                    .any(|edge| edge.target_id == helper.id));
            }
        }
        drop(overlay);
        drop(private);
        let mut reopened = Store::open(&private_path).unwrap();
        assert!(!recover_persisted_js_ts_usages(&mut reopened, "test", repo.path()).unwrap());
        let mut reopened_overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        assert!(
            !recover_persisted_js_ts_usages(&mut reopened_overlay, "test", repo.path()).unwrap()
        );
        for store in [&reopened, &reopened_overlay] {
            let helper = store
                .get_node_by_qname("test", "boundary.ts::Function::helper")
                .unwrap()
                .unwrap();
            assert_eq!(
                store
                    .incoming_edges(helper.id, Some("CALLS"), 100)
                    .unwrap()
                    .len(),
                3
            );
        }
        assert_eq!(base_bytes, fs::read(&base_path).unwrap());
    }

    #[test]
    fn jsx_usage_recovery_validates_persisted_last_definition_for_colliding_methods() {
        let repo = tempfile::tempdir().unwrap();
        let source = "function first() { return { getAttribute(name) { return name; } }; }\n\
                      function second() { return { getAttribute() { return null; } }; }\n\
                      function third() { return { getAttribute(name) { return name === 'x'; } }; }\n";
        fs::write(repo.path().join("fixture.cjs"), source).unwrap();
        fs::write(
            repo.path().join("boundary.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(repo.path().join("view.tsx"), "import { Boundary } from './boundary';\nexport function Render() { return <Boundary />; }\n").unwrap();
        let extraction =
            parser_extract(Language::JavaScript, source.as_bytes(), "fixture.cjs").unwrap();
        let methods: Vec<_> = extraction
            .nodes
            .iter()
            .filter(|node| node.qualified_name == "fixture.cjs::Function::getAttribute")
            .collect();
        assert!(
            methods.len() >= 2,
            "fixture must exercise the actual parser collision"
        );
        assert_ne!(methods[0].start_line, methods.last().unwrap().start_line);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("graph.db");
        let mut store = Store::open(&path).unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        let qname = "fixture.cjs::Function::getAttribute";
        let cached = store.get_node_by_qname("test", qname).unwrap().unwrap();
        assert_eq!(
            cached.start_line,
            i64::from(methods.last().unwrap().start_line)
        );
        let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
        let states = format!("{:?}", store.list_file_states("test").unwrap());
        store
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_KEY],
            )
            .unwrap();
        store.conn().execute_batch("DELETE FROM raw_edges WHERE edge_type='USAGE'; DELETE FROM edges WHERE edge_type='USAGE';").unwrap();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());
        assert!(!recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());
        assert_eq!(
            nodes,
            format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        assert_eq!(
            states,
            format!("{:?}", store.list_file_states("test").unwrap())
        );
        let boundary = store
            .get_node_by_qname("test", "boundary.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        let usages = store
            .incoming_edges(boundary.id, Some("USAGE"), 100)
            .unwrap();
        assert_eq!(usages.len(), 1);
        store
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_KEY],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "UPDATE main.nodes SET start_line=999 WHERE project='test' AND qualified_name=?1",
                [qname],
            )
            .unwrap();
        let raw = store.list_raw_edges("test").unwrap();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).is_err());
        assert_eq!(raw, store.list_raw_edges("test").unwrap());
        assert_eq!(
            usages,
            store
                .incoming_edges(boundary.id, Some("USAGE"), 100)
                .unwrap()
        );
        assert!(!js_ts_usages_repaired(&store).unwrap());
    }

    #[test]
    fn jsx_usage_recovery_colliding_methods_preserves_immutable_base() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(repo.path().join("fixture.cjs"), "const a = { getAttribute() { return 'a'; } };\nconst b = { getAttribute() { return 'b'; } };\n").unwrap();
        fs::write(
            repo.path().join("boundary.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(repo.path().join("view.tsx"), "import { Boundary } from './boundary';\nexport function Render() { return <Boundary />; }\n").unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, repo.path(), "test").unwrap();
            base.conn()
                .execute(
                    "DELETE FROM main.schema_meta WHERE key=?1",
                    [JS_TS_USAGE_REPAIR_KEY],
                )
                .unwrap();
            base.conn().execute_batch("DELETE FROM raw_edges WHERE edge_type='USAGE'; DELETE FROM edges WHERE edge_type='USAGE';").unwrap();
        }
        let bytes = fs::read(&base_path).unwrap();
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        let nodes = format!("{:?}", overlay.list_nodes("test", "", "", 0, 1000).unwrap());
        assert!(recover_persisted_js_ts_usages(&mut overlay, "test", repo.path()).unwrap());
        assert!(!recover_persisted_js_ts_usages(&mut overlay, "test", repo.path()).unwrap());
        assert_eq!(
            nodes,
            format!("{:?}", overlay.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        let boundary = overlay
            .get_node_by_qname("test", "boundary.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        assert_eq!(
            overlay
                .incoming_edges(boundary.id, Some("USAGE"), 100)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(bytes, fs::read(&base_path).unwrap());
    }

    #[test]
    fn jsx_usage_migrates_completed_private_and_overlay_cache_without_identity_rewrite() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("boundary.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(repo.path().join("view.tsx"), "import { Boundary } from './boundary';\nexport function Render() { return <Boundary />; }\n").unwrap();
        fs::write(repo.path().join("retained.rs"), "pub fn retained() {}\n").unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, repo.path(), "test").unwrap();
            base.conn().execute_batch("DELETE FROM raw_edges WHERE json_extract(properties,'$.jsx_component')=1; DELETE FROM edges WHERE edge_type='USAGE'; INSERT OR REPLACE INTO schema_meta VALUES('greppy.effect_fn_repair_v8.test','complete');").unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [JS_TS_USAGE_REPAIR_KEY],
                )
                .unwrap();
            mark_rust_caller_edges_repaired(&base).unwrap();
        }
        let base_bytes = fs::read(&base_path).unwrap();
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        overlay.conn().execute_batch("INSERT OR REPLACE INTO main.schema_meta VALUES('greppy.effect_fn_repair_v8.test','complete'); INSERT OR REPLACE INTO main.schema_meta VALUES('greppy.rust_usage_override_files.test','[\"retained.rs\"]'); INSERT OR REPLACE INTO main.schema_meta VALUES('greppy.rust_usage_override_rows.test','[]');").unwrap();
        let rust_metadata: String = overlay.conn().query_row("SELECT value FROM main.schema_meta WHERE key='greppy.rust_usage_override_files.test'", [], |r| r.get(0)).unwrap();
        {
            let store = &mut overlay;
            let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
            let states = format!("{:?}", store.list_file_states("test").unwrap());
            let workspace = format!(
                "{:?}",
                store
                    .get_workspace_state(repo.path().to_str().unwrap())
                    .unwrap()
            );
            assert!(recover_persisted_js_ts_usages(store, "test", repo.path()).unwrap());
            assert!(!recover_persisted_js_ts_usages(store, "test", repo.path()).unwrap());
            assert_eq!(
                nodes,
                format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
            );
            assert_eq!(
                states,
                format!("{:?}", store.list_file_states("test").unwrap())
            );
            assert_eq!(
                workspace,
                format!(
                    "{:?}",
                    store
                        .get_workspace_state(repo.path().to_str().unwrap())
                        .unwrap()
                )
            );
            let target = store
                .get_node_by_qname("test", "boundary.tsx::Function::Boundary")
                .unwrap()
                .unwrap();
            assert_eq!(
                store
                    .incoming_edges(target.id, Some("USAGE"), 100)
                    .unwrap()
                    .len(),
                1
            );
            assert!(store.list_delta_raw_edges("test").unwrap().is_empty());
        }
        assert_eq!(rust_metadata, overlay.conn().query_row::<String,_,_>("SELECT value FROM main.schema_meta WHERE key='greppy.rust_usage_override_files.test'", [], |r| r.get(0)).unwrap());
        assert_eq!(base_bytes, fs::read(&base_path).unwrap());
        let mut private = Store::open(&base_path).unwrap();
        let nodes = format!("{:?}", private.list_nodes("test", "", "", 0, 1000).unwrap());
        assert!(recover_persisted_js_ts_usages(&mut private, "test", repo.path()).unwrap());
        assert_eq!(
            nodes,
            format!("{:?}", private.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        let target = private
            .get_node_by_qname("test", "boundary.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        assert_eq!(
            private
                .incoming_edges(target.id, Some("USAGE"), 100)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn jsx_usage_recovery_refuses_changed_and_deleted_source_before_writing() {
        let repo = tempfile::tempdir().unwrap();
        let source = "export function Render() { return <Boundary />; }\n";
        fs::write(repo.path().join("view.tsx"), source).unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let mut store = Store::open(&scratch.path().join("graph.db")).unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_KEY],
            )
            .unwrap();
        let raw = store.list_raw_edges("test").unwrap();
        fs::write(
            repo.path().join("view.tsx"),
            "export function Render() { return <Changed />; }\n",
        )
        .unwrap();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).is_err());
        assert_eq!(raw, store.list_raw_edges("test").unwrap());
        assert!(!js_ts_usages_repaired(&store).unwrap());
        fs::remove_file(repo.path().join("view.tsx")).unwrap();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).is_err());
        assert_eq!(raw, store.list_raw_edges("test").unwrap());
        assert!(!js_ts_usages_repaired(&store).unwrap());
        fs::write(repo.path().join("view.tsx"), source).unwrap();
        store.conn().execute_batch(&format!("CREATE TRIGGER reject_js_marker BEFORE INSERT ON main.schema_meta WHEN NEW.key='{JS_TS_USAGE_REPAIR_KEY}' BEGIN SELECT RAISE(ABORT,'fixture marker failure'); END;")).unwrap();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).is_err());
        assert_eq!(raw, store.list_raw_edges("test").unwrap());
        assert!(!js_ts_usages_repaired(&store).unwrap());
    }

    #[test]
    fn jsx_imported_component_usage_resolves_to_exact_definition() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("boundary.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(
            repo.path().join("other.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(repo.path().join("view.tsx"), "import { Boundary } from './boundary';\nexport function Render() { return (<Boundary>\n<Boundary />\n</Boundary>); }\nexport function Shadow(Boundary: unknown) { return <Boundary />; }\n").unwrap();
        let stores = tempfile::tempdir().unwrap();
        let mut store = Store::open(&stores.path().join("graph.db")).unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        let target = store
            .get_node_by_qname("test", "boundary.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        let other = store
            .get_node_by_qname("test", "other.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        let render = store
            .get_node_by_qname("test", "view.tsx::Function::Render")
            .unwrap()
            .unwrap();
        let incoming = store.incoming_edges(target.id, Some("USAGE"), 100).unwrap();
        assert_eq!(
            incoming.len(),
            1,
            "distinct sites share one resolved symbol edge"
        );
        assert!(
            incoming.iter().all(|edge| edge.source_id == render.id),
            "incoming={incoming:?}; render={render:?}; raw={:?}",
            store.list_raw_edges("test").unwrap()
        );
        assert!(store
            .incoming_edges(other.id, Some("USAGE"), 100)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn jsx_relative_imports_keep_alias_identity_and_reject_missing_or_ambiguous_modules() {
        let repo = tempfile::tempdir().unwrap();
        fs::create_dir(repo.path().join("components")).unwrap();
        fs::create_dir(repo.path().join("views")).unwrap();
        fs::write(
            repo.path().join("components/boundary.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(
            repo.path().join("other.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(repo.path().join("views/good.tsx"), "import { Boundary as Guard } from '../components/boundary';\nexport function Render() { return <Guard />; }\n").unwrap();
        fs::write(repo.path().join("views/missing.tsx"), "import { Boundary } from './missing';\nexport function Missing() { return <Boundary />; }\n").unwrap();
        fs::write(repo.path().join("views/ambiguous.tsx"), "import { Boundary } from '../components/ambiguous';\nexport function Ambiguous() { return <Boundary />; }\n").unwrap();
        fs::write(
            repo.path().join("components/ambiguous.tsx"),
            "export function Boundary() { return <div />; }\n",
        )
        .unwrap();
        fs::write(
            repo.path().join("components/ambiguous.ts"),
            "export function Boundary() { return null; }\n",
        )
        .unwrap();
        let stores = tempfile::tempdir().unwrap();
        let mut store = Store::open(&stores.path().join("graph.db")).unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        let target = store
            .get_node_by_qname("test", "components/boundary.tsx::Function::Boundary")
            .unwrap()
            .unwrap();
        let render = store
            .get_node_by_qname("test", "views/good.tsx::Function::Render")
            .unwrap()
            .unwrap();
        let incoming = store.incoming_edges(target.id, Some("USAGE"), 100).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].source_id, render.id);
        for (file, name) in [
            ("views/missing.tsx", "Missing"),
            ("views/ambiguous.tsx", "Ambiguous"),
        ] {
            let source = store
                .get_node_by_qname("test", &format!("{file}::Function::{name}"))
                .unwrap()
                .unwrap();
            assert!(store
                .outgoing_edges(source.id, Some("USAGE"), 100)
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn effect_fn_repair_keeps_current_base_sparse_but_repairs_old_resolver_edges() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(repo.path().join("routing.ts"),
            "export function target() { return 1; }\nexport function caller() { return target(); }\n").unwrap();
        let stores = tempfile::tempdir().unwrap();
        let base_path = stores.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, repo.path(), "test").unwrap();
        }
        let visibility = greppy_store::VisibilityIndex::default();
        {
            let mut overlay =
                Store::open_overlay(&base_path, &stores.path().join("current.db"), &visibility)
                    .unwrap();
            assert!(recover_visible_effect_fn_bindings(&mut overlay, "test", repo.path()).unwrap());
            let rows: i64 = overlay
                .conn()
                .query_row("SELECT COUNT(*) FROM main.overlay_edges", [], |r| r.get(0))
                .unwrap();
            assert_eq!(rows, 0, "unchanged current Base edges must stay shared");
            let target = overlay
                .get_node_by_qname("test", "routing.ts::Function::target")
                .unwrap()
                .unwrap();
            assert!(!overlay
                .incoming_edges(target.id, Some("CALLS"), 10)
                .unwrap()
                .is_empty());
        }
        // An older resolver can have identical raw facts and missing resolved
        // edges. That case must still repair the immutable Base via the Delta.
        {
            let base = Store::open(&base_path).unwrap();
            base.conn()
                .execute(
                    "UPDATE main.workspace_state SET indexer_version='greppy-indexer-v8'",
                    [],
                )
                .unwrap();
            base.conn()
                .execute("DELETE FROM main.edges WHERE edge_type='CALLS'", [])
                .unwrap();
        }
        let mut overlay =
            Store::open_overlay(&base_path, &stores.path().join("old.db"), &visibility).unwrap();
        assert!(recover_visible_effect_fn_bindings(&mut overlay, "test", repo.path()).unwrap());
        let target = overlay
            .get_node_by_qname("test", "routing.ts::Function::target")
            .unwrap()
            .unwrap();
        assert!(!overlay
            .incoming_edges(target.id, Some("CALLS"), 10)
            .unwrap()
            .is_empty());
        let repairs: i64 = overlay.conn().query_row("SELECT COUNT(*) FROM main.overlay_edges WHERE json_extract(properties,'$.greppy_base_repair_v2')=1", [], |r| r.get(0)).unwrap();
        assert!(repairs > 0, "old resolver recovery remains active");
    }

    #[test]
    fn effect_fn_overlay_upgrade_keeps_immutable_base_and_unaffected_vector() {
        let repo = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("routing.ts"),
            r#"
import { Effect } from 'effect';
const target = Effect.fn('target')(function* () { return 1; });
export const caller = Effect.fn('caller')(function* () { return yield* target(); });
"#,
        )
        .unwrap();
        fs::write(repo.path().join("retained.py"), "def retained(): pass\n").unwrap();
        let stores = tempfile::tempdir().unwrap();
        let base_path = stores.path().join("base.db");
        let delta_path = stores.path().join("delta.db");
        let retained_id;
        {
            let mut base = Store::open(&base_path).unwrap();
            let report = index(&mut base, repo.path(), "test").unwrap();
            for name in ["target", "caller"] {
                let node = base
                    .get_node_by_qname("test", &format!("routing.ts::Function::{name}"))
                    .unwrap()
                    .unwrap();
                base.update_node_identity(
                    node.id,
                    "Variable",
                    &format!("routing.ts::Variable::{name}"),
                )
                .unwrap();
            }
            base.conn().execute("UPDATE main.raw_edges SET source_qname=replace(source_qname,'::Function::','::Variable::') WHERE file_path='routing.ts'", []).unwrap();
            base.conn()
                .execute("DELETE FROM main.edges WHERE edge_type='CALLS'", [])
                .unwrap();
            let retained = base
                .get_node_by_qname("test", "retained.py::Function::retained")
                .unwrap()
                .unwrap();
            retained_id = retained.id;
            base.upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
                project: "test".into(),
                model_id: "test-model".into(),
                prompt_version: "v1".into(),
                task: "definition".into(),
                node_id: Some(retained.id),
                chunk_idx: 0,
                qualified_name: retained.qualified_name,
                file_path: retained.file_path,
                start_line: retained.start_line,
                end_line: retained.end_line,
                content_sha256: file_state::sha256_hex(b"def retained(): pass\n"),
                graph_generation: report.graph_generation,
                vector: vec![1.0, 0.0],
            })
            .unwrap();
        }
        let base_hash = file_state::sha256_hex(&fs::read(&base_path).unwrap());
        let visibility = greppy_store::VisibilityIndex::default();
        {
            let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
            let routing = repo.path().join("routing.ts");
            let original = fs::read(&routing).unwrap();
            let mut changed = original.clone();
            changed.extend_from_slice(b"\n// changed since indexing\n");
            fs::write(&routing, changed).unwrap();
            assert!(recover_visible_effect_fn_bindings(&mut overlay, "test", repo.path()).is_err());
            let writes: i64 = overlay
                .conn()
                .query_row("SELECT COUNT(*) FROM main.nodes", [], |r| r.get(0))
                .unwrap();
            assert_eq!(
                writes, 0,
                "source mismatch must reject before identity writes"
            );
            fs::write(&routing, original).unwrap();
            overlay.conn().execute_batch("CREATE TEMP TRIGGER fail_effect_repair BEFORE INSERT ON main.raw_edges BEGIN SELECT RAISE(ABORT,'injected repair failure'); END;").unwrap();
            let injected =
                recover_visible_effect_fn_bindings(&mut overlay, "test", repo.path()).unwrap_err();
            assert!(
                injected.to_string().contains("injected repair failure"),
                "must reach the injected raw-edge failure after node promotion: {injected}"
            );
            for table in [
                "main.projects",
                "main.nodes",
                "main.definition_identity_overrides",
                "main.js_ts_reference_override_files",
                "main.raw_edges",
            ] {
                let rows: i64 = overlay
                    .conn()
                    .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                    .unwrap();
                assert_eq!(rows, 0, "failed repair must roll back {table}");
            }
            let markers: i64 = overlay.conn().query_row("SELECT COUNT(*) FROM main.schema_meta WHERE key LIKE 'greppy.effect_fn_repair%'", [], |r| r.get(0)).unwrap();
            assert_eq!(markers, 0);
            overlay
                .conn()
                .execute_batch("DROP TRIGGER fail_effect_repair;")
                .unwrap();
            assert!(recover_visible_effect_fn_bindings(&mut overlay, "test", repo.path()).unwrap());
            let target = overlay
                .get_node_by_qname("test", "routing.ts::Function::target")
                .unwrap()
                .unwrap();
            let caller = overlay
                .get_node_by_qname("test", "routing.ts::Function::caller")
                .unwrap()
                .unwrap();
            assert!(overlay
                .get_node_by_qname("test", "routing.ts::Variable::target")
                .unwrap()
                .is_none());
            assert!(overlay
                .incoming_edges(target.id, Some("CALLS"), 10)
                .unwrap()
                .iter()
                .any(|e| e.source_id == caller.id));
            assert_eq!(
                overlay
                    .get_node_by_qname("test", "retained.py::Function::retained")
                    .unwrap()
                    .unwrap()
                    .id,
                -retained_id
            );
            let vector: (String, i64) = overlay.conn().query_row("SELECT content_sha256,node_id FROM vector_embeddings WHERE file_path='retained.py'", [], |r| Ok((r.get(0)?,r.get(1)?))).unwrap();
            assert_eq!(
                vector,
                (
                    file_state::sha256_hex(b"def retained(): pass\n"),
                    -retained_id
                )
            );
            assert!(overlay.list_private_file_states("test").unwrap().is_empty());
            assert!(
                !recover_visible_effect_fn_bindings(&mut overlay, "test", repo.path()).unwrap()
            );
        }
        assert_eq!(
            file_state::sha256_hex(&fs::read(&base_path).unwrap()),
            base_hash
        );
        let reopened = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        assert!(reopened
            .get_node_by_qname("test", "routing.ts::Variable::target")
            .unwrap()
            .is_none());
        assert!(reopened
            .get_node_by_qname("test", "routing.ts::Function::target")
            .unwrap()
            .is_some());
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

    #[test]
    fn rust_type_arguments_and_variant_owners_preserve_exact_import_and_shadow_resolution() {
        let repo = setup_multifile_repo(
            "type-inputs",
            r#"
mod helper; mod other;
use helper::{Marker, Response as Imported};
enum Response { Ready, Tuple(u8), Struct { count: u8 } }
fn Response() {}
fn parse<T>() {}
fn generic_local() { parse::<Response>(); }
fn generic_imported() { parse::<Imported>(); }
fn generic_qualified() { parse::<other::Response>(); }
fn generic_missing() { parse::<missing::Response>(); }
fn generic_shadow<Response>() { parse::<Response>(); }
fn value_shadow(Response: fn()) { let _ = Response; }
fn variant_local() { let _ = Response::Ready; }
fn variant_tuple() { let _ = Response::Tuple(1); }
fn variant_struct() { let value = Response::Struct { count: 1 }; match value { Response::Struct { count } => count, _ => 0 }; }
fn variant_imported() { let _ = Imported::Ready; }
fn variant_qualified() { let _ = other::Response::Ready; }
fn variant_missing() { let _ = missing::Response::Ready; }
fn variant_wrong() { let _ = Response::Missing; }
fn variant_shadow<Response>() { let _ = Response::Ready; let _ = Response::Tuple(1); }
"#,
            "pub struct Marker; pub enum Response { Ready }\n",
        );
        fs::write(repo.join("src/other.rs"), "pub enum Response { Ready }\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let check = |store: &Store| {
            for (file, callers) in [
                (
                    "src/lib.rs",
                    vec![
                        "generic_local",
                        "variant_local",
                        "variant_tuple",
                        "variant_struct",
                    ],
                ),
                (
                    "src/helper.rs",
                    vec!["generic_imported", "variant_imported"],
                ),
                (
                    "src/other.rs",
                    vec!["generic_qualified", "variant_qualified"],
                ),
            ] {
                let target = store
                    .get_node_by_qname("test", &format!("{file}::Enum::Response"))
                    .unwrap()
                    .unwrap();
                let incoming = store.incoming_edges(target.id, Some("USAGE"), 100).unwrap();
                let sources = incoming
                    .iter()
                    .map(|edge| store.get_node(edge.source_id).unwrap().unwrap().name)
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(
                    sources,
                    callers
                        .into_iter()
                        .map(str::to_owned)
                        .collect::<std::collections::BTreeSet<_>>(),
                    "{file}: {incoming:?}"
                );
            }
            let callable = store
                .get_node_by_qname("test", "src/lib.rs::Function::Response")
                .unwrap()
                .unwrap();
            assert!(store
                .incoming_edges(callable.id, Some("USAGE"), 100)
                .unwrap()
                .is_empty());
            for caller in [
                "generic_missing",
                "generic_shadow",
                "value_shadow",
                "variant_missing",
                "variant_wrong",
                "variant_shadow",
            ] {
                let caller = store
                    .get_node_by_qname("test", &format!("src/lib.rs::Function::{caller}"))
                    .unwrap()
                    .unwrap();
                assert!(
                    store
                        .outgoing_edges(caller.id, Some("USAGE"), 100)
                        .unwrap()
                        .is_empty(),
                    "{}",
                    caller.name
                );
            }
        };
        check(&store);
        let states = store.list_file_states("test").unwrap();
        let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
        store
            .conn()
            .execute("DELETE FROM edges WHERE edge_type='USAGE'", [])
            .unwrap();
        store
            .conn()
            .execute("DELETE FROM raw_edges WHERE edge_type='USAGE'", [])
            .unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        store.conn().execute("INSERT INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v12','complete')", []).unwrap();
        assert!(!rust_caller_edges_repaired(&store).unwrap());
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(rust_caller_edges_repaired(&store).unwrap());
        check(&store);
        assert_eq!(store.list_file_states("test").unwrap(), states);
        assert_eq!(
            format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap()),
            nodes
        );
        fs::remove_dir_all(repo).unwrap();
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
    fn rust_expression_macro_parent_custom_binding_does_not_invent_calls() {
        for parent in [
            "pub use custom::ensure;",
            "pub use custom::dsl as ensure;",
            "macro_rules! ensure { ($($tokens:tt)*) => {} }",
            "pub use self::tests::ensure;",
        ] {
            let repo = setup_multifile_repo(
                "rust-custom-expression-macro",
                "mod channel;\n",
                "// placeholder\n",
            );
            fs::create_dir_all(repo.join("src/channel")).unwrap();
            fs::write(
                repo.join("src/channel/mod.rs"),
                format!("{parent}\nmod tests; pub fn decoy() {{}}\n"),
            )
            .unwrap();
            fs::write(
                repo.join("src/channel/tests.rs"),
                "use super::*; fn caller() { ensure!(decoy()); }\n",
            )
            .unwrap();
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            let target = store
                .get_node_by_qname("test", "src/channel/mod.rs::Function::decoy")
                .unwrap()
                .unwrap();
            assert!(
                store
                    .incoming_edges(target.id, Some("CALLS"), 10)
                    .unwrap()
                    .is_empty(),
                "unproved parent macro binding invented a caller: {parent}"
            );
            fs::remove_dir_all(repo).unwrap();
        }
    }

    #[test]
    fn rust_expression_macro_generic_cfg_call_is_resolved_and_repaired() {
        let repo = setup_multifile_repo(
            "rust-expression-macro-call",
            "mod channel;\n",
            "// placeholder\n",
        );
        fs::write(repo.join("Cargo.toml"), "[package]\nname='macro_fixture'\nversion='0.1.0'\nedition='2021'\n[dependencies]\nanyhow='1'\n").unwrap();
        fs::write(repo.join("Cargo.lock"), format!("version=3\n[[package]]\nname='anyhow'\nversion='1.0.102'\nsource='registry+https://github.com/rust-lang/crates.io-index'\nchecksum='{}'\n", "a".repeat(64))).unwrap();
        fs::create_dir_all(repo.join("src/channel")).unwrap();
        fs::write(
            repo.join("src/channel/mod.rs"),
            "use anyhow::ensure; mod implementation; mod tests; pub use implementation::run_guest_desktop_effects;\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/channel/implementation.rs"),
            "pub async fn run_guest_desktop_effects<T>(driver: &T) -> bool { true }\n",
        )
        .unwrap();
        fs::write(repo.join("src/channel/tests.rs"), "use super::*;\n#[cfg(target_os = \"linux\")]\nasync fn caller() { ensure!(run_guest_desktop_effects(&driver).await == Err(Unavailable), \"no endpoint\"); }\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let target = store
            .get_node_by_qname(
                "test",
                "src/channel/implementation.rs::Function::run_guest_desktop_effects",
            )
            .unwrap()
            .unwrap();
        let caller = store
            .get_node_by_qname("test", "src/channel/tests.rs::Function::caller")
            .unwrap()
            .unwrap();
        assert!(store
            .incoming_edges(target.id, Some("CALLS"), 10)
            .unwrap()
            .iter()
            .any(|e| e.source_id == caller.id));
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE edge_type='CALLS' AND target_id=?1",
                [target.id],
            )
            .unwrap();
        store.conn().execute("DELETE FROM raw_edges WHERE edge_type='CALLS' AND json_extract(properties,'$.callee_name')='run_guest_desktop_effects'", []).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        store.conn().execute("INSERT INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v14','complete')", []).unwrap();
        assert!(!rust_caller_edges_repaired(&store).unwrap());
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(rust_caller_edges_repaired(&store).unwrap());
        assert!(store
            .incoming_edges(target.id, Some("CALLS"), 10)
            .unwrap()
            .iter()
            .any(|e| e.source_id == caller.id));
        for dependency in [
            "anyhow={package='custom_dsl',version='1'}",
            "anyhow={path='custom_dsl'}",
            "anyhow='1'\n[patch.crates-io]\nanyhow={path='custom_dsl'}",
        ] {
            fs::write(repo.join("Cargo.toml"), format!("[package]\nname='macro_fixture'\nversion='0.1.0'\nedition='2021'\n[dependencies]\n{dependency}\n")).unwrap();
            rebuild_single_store_rust_edges(&mut store, "test").unwrap();
            assert!(
                store
                    .incoming_edges(target.id, Some("CALLS"), 10)
                    .unwrap()
                    .is_empty(),
                "custom/overridden anyhow package invented macro calls: {dependency}"
            );
        }
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn rust_qualified_function_item_resolves_through_grouped_reexport() {
        let repo = setup_multifile_repo(
            "rust-reexport-function-item",
            "mod business_os; mod channels; mod core; mod flat; mod left; mod right; mod other; mod types;\nuse other::imported_worker;\nfn target() {}\nfn root_target() {}\nstruct worker; impl worker { fn run() {} }\nfn caller() { let selected = channels::target; selected(); }\nfn left_caller() { let selected = left::channel::target; selected(); }\nfn left_direct_caller() { left::channel::target(); }\nfn local_associated_caller() { worker::run(); }\nfn imported_associated_caller() { imported_worker::run(); }\nfn qualified_associated_caller() { types::worker::run(); }\nfn missing_qualified_associated_caller() { missing::worker::run(); }\nfn missing_caller() { let _selected = missing::target; }\nfn missing_direct_caller() { missing::target(); }\n",
            "// fixture placeholder\n",
        );
        fs::create_dir_all(repo.join("src/channels")).unwrap();
        fs::write(
            repo.join("src/channels/mod.rs"),
            "mod command; mod tests; pub use command::{first, target};\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/channels/command.rs"),
            "pub fn first() {}\npub fn target() {}\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/channels/tests.rs"),
            "use super::*;\nfn direct_test() { target(); }\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/other.rs"),
            "pub fn target() {}\npub struct imported_worker; impl imported_worker { pub fn run() {} }\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/types.rs"),
            "pub struct worker; impl worker { pub fn run() {} }\n",
        )
        .unwrap();
        for side in ["left", "right"] {
            fs::create_dir_all(repo.join(format!("src/{side}/channel"))).unwrap();
            fs::write(
                repo.join(format!("src/{side}/mod.rs")),
                "pub mod channel;\n",
            )
            .unwrap();
            fs::write(
                repo.join(format!("src/{side}/channel/mod.rs")),
                "mod implementation; pub use implementation::target;\n",
            )
            .unwrap();
            fs::write(
                repo.join(format!("src/{side}/channel/implementation.rs")),
                "pub fn target() {}\n",
            )
            .unwrap();
        }
        fs::create_dir_all(repo.join("src/core/mission/channels")).unwrap();
        fs::create_dir_all(repo.join("src/business_os")).unwrap();
        fs::write(repo.join("src/core/mod.rs"), "pub mod mission;\n").unwrap();
        fs::write(repo.join("src/core/mission/mod.rs"), "pub mod channels;\n").unwrap();
        fs::write(
            repo.join("src/core/mission/channels/mod.rs"),
            "mod command; pub use command::target;\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/core/mission/channels/command.rs"),
            "pub fn target() {}\n",
        )
        .unwrap();
        fs::write(repo.join("src/business_os/mod.rs"), "pub mod store;\n").unwrap();
        fs::write(
            repo.join("src/business_os/store.rs"),
            "use crate::core::mission::channels;\nfn alias_caller() { let selected = channels::target; selected(); }\nfn nested_alias_missing() { let _selected = channels::missing::target; }\n",
        )
        .unwrap();
        fs::create_dir_all(repo.join("src/flat")).unwrap();
        fs::write(
            repo.join("src/flat.rs"),
            "pub mod child;\nfn self_caller() { let selected = self::child::target; selected(); }\nfn super_caller() { let selected = super::root_target; selected(); }\n",
        )
        .unwrap();
        fs::write(repo.join("src/flat/child.rs"), "pub fn target() {}\n").unwrap();
        fs::write(
            repo.join("src/unrelated.py"),
            "def untouched():\n    return 1\n",
        )
        .unwrap();

        let mut store = Store::open_memory().unwrap();
        let initial = index(&mut store, &repo, "test").unwrap();
        let target = store
            .get_node_by_qname("test", "src/channels/command.rs::Function::target")
            .unwrap()
            .expect("reexported target function");
        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .expect("caller function");
        let incoming = store.incoming_edges(target.id, Some("USAGE"), 10).unwrap();
        assert!(
            incoming.iter().any(|edge| edge.source_id == caller.id),
            "qualified function item must resolve through the module's grouped reexport: {incoming:?}"
        );
        let direct_test = store
            .get_node_by_qname("test", "src/channels/tests.rs::Function::direct_test")
            .unwrap()
            .expect("direct test caller");
        let calls = store.incoming_edges(target.id, Some("CALLS"), 10).unwrap();
        assert!(
            calls.iter().any(|edge| edge.source_id == direct_test.id),
            "super glob must disambiguate the reexported direct call: {calls:?}"
        );
        let homonym = store
            .get_node_by_qname("test", "src/other.rs::Function::target")
            .unwrap()
            .expect("unrelated homonym");
        assert!(
            store
                .incoming_edges(homonym.id, None, 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != caller.id),
            "module and glob evidence must not create cross-module homonym callers"
        );
        let left_target = store
            .get_node_by_qname(
                "test",
                "src/left/channel/implementation.rs::Function::target",
            )
            .unwrap()
            .expect("left channel target");
        let right_target = store
            .get_node_by_qname(
                "test",
                "src/right/channel/implementation.rs::Function::target",
            )
            .unwrap()
            .expect("right channel target");
        let left_caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::left_caller")
            .unwrap()
            .expect("qualified left caller");
        assert!(
            store
                .incoming_edges(left_target.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == left_caller.id),
            "the full module path must select the left channel export"
        );
        assert!(
            store
                .incoming_edges(right_target.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != left_caller.id),
            "a duplicate channel/mod.rs basename must not steal the qualified usage"
        );
        let left_direct_caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::left_direct_caller")
            .unwrap()
            .expect("qualified left direct caller");
        assert!(
            store
                .incoming_edges(left_target.id, Some("CALLS"), 10)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == left_direct_caller.id),
            "the complete module path must select the left channel direct-call target"
        );
        assert!(
            store
                .incoming_edges(right_target.id, Some("CALLS"), 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != left_direct_caller.id),
            "the duplicate right channel must not steal the qualified direct call"
        );
        let local_target = store
            .get_node_by_qname("test", "src/lib.rs::Function::target")
            .unwrap()
            .expect("same-file homonym");
        let missing_caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::missing_caller")
            .unwrap()
            .expect("missing-module caller");
        assert!(
            store
                .incoming_edges(local_target.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != missing_caller.id),
            "missing::target must not degrade to the same-file target"
        );
        let missing_direct_caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::missing_direct_caller")
            .unwrap()
            .expect("missing-module direct caller");
        assert!(
            [left_target.id, right_target.id, local_target.id]
                .into_iter()
                .all(|target_id| store
                    .incoming_edges(target_id, Some("CALLS"), 10)
                    .unwrap()
                    .iter()
                    .all(|edge| edge.source_id != missing_direct_caller.id)),
            "missing::target must remain unresolved instead of guessing any homonym"
        );
        for (method_qname, caller_qname) in [
            (
                "src/lib.rs::worker::run",
                "src/lib.rs::Function::local_associated_caller",
            ),
            (
                "src/other.rs::imported_worker::run",
                "src/lib.rs::Function::imported_associated_caller",
            ),
            (
                "src/types.rs::worker::run",
                "src/lib.rs::Function::qualified_associated_caller",
            ),
        ] {
            let method = store
                .get_node_by_qname("test", method_qname)
                .unwrap()
                .expect("lowercase associated method");
            let associated_caller = store
                .get_node_by_qname("test", caller_qname)
                .unwrap()
                .expect("lowercase associated caller");
            assert!(
                store
                    .incoming_edges(method.id, Some("CALLS"), 10)
                    .unwrap()
                    .iter()
                    .any(|edge| edge.source_id == associated_caller.id),
                "a real lowercase type owner must preserve its associated call: {method_qname}"
            );
        }
        let missing_qualified_associated_caller = store
            .get_node_by_qname(
                "test",
                "src/lib.rs::Function::missing_qualified_associated_caller",
            )
            .unwrap()
            .expect("missing qualified associated caller");
        let local_worker_method = store
            .get_node_by_qname("test", "src/lib.rs::worker::run")
            .unwrap()
            .expect("local lowercase associated method");
        assert!(
            store
                .incoming_edges(local_worker_method.id, Some("CALLS"), 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != missing_qualified_associated_caller.id),
            "missing::worker::run must not discard its prefix and bind to local worker::run"
        );
        let namespaced_target = store
            .get_node_by_qname(
                "test",
                "src/core/mission/channels/command.rs::Function::target",
            )
            .unwrap()
            .expect("mission channel target");
        let alias_caller = store
            .get_node_by_qname("test", "src/business_os/store.rs::Function::alias_caller")
            .unwrap()
            .expect("CTOX-shaped namespace caller");
        assert!(
            store
                .incoming_edges(namespaced_target.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == alias_caller.id),
            "a namespace imported with `use crate::...::channels` must resolve channels::target"
        );
        let nested_alias_missing = store
            .get_node_by_qname(
                "test",
                "src/business_os/store.rs::Function::nested_alias_missing",
            )
            .unwrap()
            .expect("nested namespace negative caller");
        assert!(
            store
                .incoming_edges(namespaced_target.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != nested_alias_missing.id),
            "channels::missing::target must not collapse to channels::target"
        );
        let flat_child_target = store
            .get_node_by_qname("test", "src/flat/child.rs::Function::target")
            .unwrap()
            .expect("flat module child target");
        let self_caller = store
            .get_node_by_qname("test", "src/flat.rs::Function::self_caller")
            .unwrap()
            .expect("flat-module self caller");
        assert!(
            store
                .incoming_edges(flat_child_target.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == self_caller.id),
            "self::child from flat.rs must resolve beneath the flat module namespace"
        );
        let root_target = store
            .get_node_by_qname("test", "src/lib.rs::Function::root_target")
            .unwrap()
            .expect("crate-root target");
        let super_caller = store
            .get_node_by_qname("test", "src/flat.rs::Function::super_caller")
            .unwrap()
            .expect("flat-module super caller");
        assert!(
            store
                .incoming_edges(root_target.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == super_caller.id),
            "super:: from flat.rs must resolve in the parent module"
        );

        let untouched_before = store
            .get_node_by_qname("test", "src/unrelated.py::Function::untouched")
            .unwrap()
            .expect("unrelated Python definition");
        store
            .upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
                project: "test".into(),
                model_id: "fixture-model".into(),
                prompt_version: "v1".into(),
                task: "code".into(),
                node_id: Some(untouched_before.id),
                chunk_idx: 0,
                qualified_name: "src/unrelated.py::Function::untouched".into(),
                file_path: "src/unrelated.py".into(),
                start_line: 1,
                end_line: 2,
                content_sha256: "79e7f0faa5c096d71e2144fed19041c227465b02667a95b613c0ecd4648e1a03"
                    .into(),
                graph_generation: initial.graph_generation,
                vector: vec![1.0, 0.0],
            })
            .unwrap();
        store
            .upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
                project: "test".into(),
                model_id: "fixture-model".into(),
                prompt_version: "v1".into(),
                task: "code".into(),
                node_id: Some(target.id),
                chunk_idx: 0,
                qualified_name: target.qualified_name.clone(),
                file_path: target.file_path.clone(),
                start_line: target.start_line,
                end_line: target.end_line,
                content_sha256: "0126ac6c598444305c31117e8a38a15cb496335cbd34fb503dfd331926e93fb7"
                    .into(),
                graph_generation: initial.graph_generation,
                vector: vec![0.0, 1.0],
            })
            .unwrap();
        store
            .insert_edge(&NewEdge {
                project: "test".into(),
                source_id: missing_caller.id,
                target_id: local_target.id,
                edge_type: "USAGE".into(),
                properties: serde_json::json!({
                    "ref_name": "target",
                    "ref_path": "missing::target"
                }),
            })
            .unwrap();
        let root = greppy_discover::detect_repo_root(&repo).unwrap();
        let mut state = store
            .get_workspace_state(root.to_string_lossy().as_ref())
            .unwrap()
            .expect("workspace state");
        state.indexer_version = "greppy-indexer-v6".into();
        store.upsert_workspace_state(&state).unwrap();
        fs::write(repo.join("src/other.rs"), "pub fn replacement() {}\n").unwrap();

        let migration = index_with_options(
            &mut store,
            &repo,
            "test",
            &IndexOptions {
                only_paths: Some(["src/lib.rs".to_string()].into_iter().collect()),
                ..IndexOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            migration.files_indexed, initial.files_indexed,
            "v9 re-extracts every retained source despite sparse scope"
        );
        let untouched_after = store
            .get_node_by_qname("test", "src/unrelated.py::Function::untouched")
            .unwrap()
            .expect("unrelated Python definition survives migration");
        assert_ne!(
            untouched_after.id, untouched_before.id,
            "v9 replaces old cache nodes"
        );
        let target_after = store
            .get_node_by_qname("test", "src/channels/command.rs::Function::target")
            .unwrap()
            .expect("Rust target survives migration");
        assert_ne!(
            target_after.id, target.id,
            "v9 re-extracts unchanged Rust declarations"
        );
        assert!(
            store
                .get_node_by_qname("test", "src/other.rs::Function::target")
                .unwrap()
                .is_none(),
            "a changed Rust file is also freshly extracted during the full refresh"
        );
        assert!(store
            .get_node_by_qname("test", "src/other.rs::Function::replacement")
            .unwrap()
            .is_some());
        let local_target_after = store
            .get_node_by_qname("test", &local_target.qualified_name)
            .unwrap()
            .unwrap();
        let missing_caller_after = store
            .get_node_by_qname("test", &missing_caller.qualified_name)
            .unwrap()
            .unwrap();
        assert!(
            store
                .incoming_edges(local_target_after.id, Some("USAGE"), 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != missing_caller_after.id),
            "full refresh removes a stale v6 false-positive edge"
        );
        let preserved_vectors: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM vector_embeddings WHERE project='test' AND graph_generation=?1 AND file_path IN ('src/unrelated.py', 'src/channels/command.rs')",
                rusqlite::params![migration.graph_generation as i64],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            preserved_vectors, 0,
            "old node-bound vectors are retired with the incompatible declaration cache"
        );
        for (qualified_name, content_sha256) in [
            (
                "src/unrelated.py::Function::untouched",
                "79e7f0faa5c096d71e2144fed19041c227465b02667a95b613c0ecd4648e1a03",
            ),
            (
                "src/channels/command.rs::Function::target",
                "0126ac6c598444305c31117e8a38a15cb496335cbd34fb503dfd331926e93fb7",
            ),
        ] {
            let reusable = store
                .find_reusable_vector_embedding(&greppy_store::ReusableVectorEmbeddingKey {
                    project: "test",
                    model_id: "fixture-model",
                    prompt_version: "v1",
                    task: "code",
                    qualified_name,
                    chunk_idx: 0,
                    content_sha256,
                })
                .unwrap();
            assert!(
                reusable.is_none(),
                "old node-bound vectors must not certify fresh declarations"
            );
        }
        let clean = index_with_options(
            &mut store,
            &repo,
            "test",
            &IndexOptions {
                only_paths: Some(["src/lib.rs".to_string()].into_iter().collect()),
                ..IndexOptions::default()
            },
        )
        .unwrap();
        assert_eq!(clean.files_indexed, 0, "the next sparse run is clean");
    }

    #[test]
    fn rust_crate_root_under_src_core_resolves_calls_and_function_items() {
        let repo = setup_multifile_repo(
            "rust-crate-root-under-src-core",
            "// ordinary source root remains in the fixture\n",
            "// ordinary source root remains in the fixture\n",
        );
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname = \"src-core-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[[bin]]\nname = \"fixture\"\npath = \"src/core/main.rs\"\n",
        )
        .unwrap();
        fs::create_dir_all(repo.join("src/core/mission/channels")).unwrap();
        fs::create_dir_all(repo.join("src/core/business_os")).unwrap();
        fs::write(
            repo.join("src/core/main.rs"),
            "mod mission; mod business_os;\n",
        )
        .unwrap();
        fs::write(repo.join("src/core/mission/mod.rs"), "pub mod channels;\n").unwrap();
        fs::write(
            repo.join("src/core/mission/channels/mod.rs"),
            "pub fn direct_target() {}\npub fn alternate_target() {}\n",
        )
        .unwrap();
        fs::write(repo.join("src/core/business_os/mod.rs"), "pub mod store;\n").unwrap();
        fs::write(
            repo.join("src/core/business_os/store.rs"),
            "use crate::mission::channels;\n\
             pub fn direct_caller() { channels::direct_target(); }\n\
             pub fn function_item_caller(flag: bool) {\n\
                 let selected = if flag { channels::direct_target } else { channels::alternate_target };\n\
                 selected();\n\
             }\n",
        )
        .unwrap();

        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();

        let direct_target = store
            .get_node_by_qname(
                "test",
                "src/core/mission/channels/mod.rs::Function::direct_target",
            )
            .unwrap()
            .expect("direct target");
        let alternate_target = store
            .get_node_by_qname(
                "test",
                "src/core/mission/channels/mod.rs::Function::alternate_target",
            )
            .unwrap()
            .expect("alternate target");
        let direct_caller = store
            .get_node_by_qname(
                "test",
                "src/core/business_os/store.rs::Function::direct_caller",
            )
            .unwrap()
            .expect("direct caller");
        let function_item_caller = store
            .get_node_by_qname(
                "test",
                "src/core/business_os/store.rs::Function::function_item_caller",
            )
            .unwrap()
            .expect("function-item caller");

        assert!(
            store
                .incoming_edges(direct_target.id, Some("CALLS"), 10)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == direct_caller.id),
            "crate-root direct call under src/core must resolve"
        );
        for target in [direct_target, alternate_target] {
            assert!(
                store
                    .incoming_edges(target.id, Some("USAGE"), 10)
                    .unwrap()
                    .iter()
                    .any(|edge| edge.source_id == function_item_caller.id),
                "crate-root function item under src/core must resolve: {}",
                target.qualified_name
            );
        }
    }

    #[test]
    fn rust_inline_module_site_stops_at_the_missing_file_module() {
        let known = std::collections::HashSet::from([
            "src/lib.rs".to_string(),
            "src/sync/mod.rs".to_string(),
            "src/sync/barrier.rs".to_string(),
        ]);
        assert_eq!(
            rust_inline_module_sites("src/sync/barrier.rs", "crate::trace", &known, None),
            vec![("src/lib.rs".to_string(), "trace".to_string())]
        );
        assert_eq!(
            rust_inline_module_sites("src/sync/barrier.rs", "crate::a::b", &known, None),
            vec![("src/lib.rs".to_string(), "a::b".to_string())]
        );
        assert_eq!(
            rust_inline_module_sites("src/sync/barrier.rs", "crate::sync::helper", &known, None),
            vec![("src/sync/mod.rs".to_string(), "helper".to_string())]
        );
        assert!(
            rust_inline_module_sites("src/lib.rs", "crate::sync::barrier", &known, None).is_empty(),
            "a path that lands on a real file module is not inline"
        );
    }

    #[test]
    fn rust_inline_module_calls_resolve_without_a_module_file() {
        let repo = setup_multifile_repo(
            "rust-inline-module",
            "mod trace {\n\
                 pub(crate) async fn async_trace_leaf() {}\n\
                 fn private_leaf() {}\n\
                 pub(crate) fn calls_private() { crate::trace::private_leaf(); }\n\
             }\n\
             mod a {\n\
                 pub mod b { pub fn nested() {} }\n\
                 mod hidden { pub fn secret() {} }\n\
             }\n\
             mod sync;\n",
            "// placeholder\n",
        );
        fs::create_dir_all(repo.join("src/sync")).unwrap();
        fs::write(repo.join("src/sync/mod.rs"), "pub mod barrier;\n").unwrap();
        fs::write(
            repo.join("src/sync/barrier.rs"),
            "pub fn changed_impl() {\n\
                 crate::trace::async_trace_leaf();\n\
                 crate::trace::private_leaf();\n\
                 crate::a::b::nested();\n\
                 crate::a::hidden::secret();\n\
             }\n",
        )
        .unwrap();

        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();

        let target = |qname: &str| {
            store
                .get_node_by_qname("test", qname)
                .unwrap()
                .unwrap_or_else(|| panic!("missing {qname}"))
        };
        let leaf = target("src/lib.rs::Function::async_trace_leaf");
        let private_leaf = target("src/lib.rs::Function::private_leaf");
        let calls_private = target("src/lib.rs::Function::calls_private");
        let nested = target("src/lib.rs::Function::nested");
        let secret = target("src/lib.rs::Function::secret");
        let caller = target("src/sync/barrier.rs::Function::changed_impl");
        let calls_from = |id: i64, source: i64| {
            store
                .incoming_edges(id, Some("CALLS"), 20)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == source)
        };

        assert!(
            calls_from(leaf.id, caller.id),
            "pub(crate) fn in an inline module must resolve across files"
        );
        assert!(
            calls_from(nested.id, caller.id),
            "nested inline modules must resolve crate::a::b::nested"
        );
        assert!(
            !calls_from(private_leaf.id, caller.id),
            "a private inline fn must not be linked from another file"
        );
        assert!(
            calls_from(private_leaf.id, calls_private.id),
            "a private inline fn must still resolve from inside its module"
        );
        assert!(
            !calls_from(secret.id, caller.id),
            "a public fn inside a private nested inline module is not visible outside it"
        );
    }

    #[test]
    fn rust_crate_root_uncovered_manifest_member_preserves_conventional_layout() {
        let roots = std::collections::HashSet::from(["other/src/lib.rs".to_string()]);
        let files = rust_module_files_for_module_path_with_crate_roots(
            "crates/widget/src/nested/caller.rs",
            "crate::helpers",
            Some(&roots),
        );
        assert_eq!(
            files,
            vec![
                "crates/widget/src/helpers.rs".to_string(),
                "crates/widget/src/helpers/mod.rs".to_string(),
                "crates/widget/src/helpers/lib.rs".to_string(),
                "crates/widget/src/helpers/main.rs".to_string(),
            ]
        );
    }

    fn calls_between(store: &Store, target: &str, source: &str) -> bool {
        let Some(target) = store.get_node_by_qname("test", target).unwrap() else {
            panic!("missing target {target}");
        };
        let Some(source) = store.get_node_by_qname("test", source).unwrap() else {
            panic!("missing source {source}");
        };
        store
            .incoming_edges(target.id, Some("CALLS"), 30)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == source.id)
    }

    #[test]
    fn rust_workspace_path_deps_resolve_qualified_and_imported_calls() {
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-workspace-extern-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("crates/crate_b/src")).unwrap();
        fs::create_dir_all(repo.join("crates/app/src")).unwrap();
        fs::create_dir_all(repo.join("crates/workspace_user/src")).unwrap();
        fs::create_dir_all(repo.join("crates/decoy/src")).unwrap();
        fs::create_dir_all(repo.join("crates/shadow/src")).unwrap();
        fs::write(
            repo.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/crate_b\", \"crates/app\", \"crates/workspace_user\", \"crates/decoy\", \"crates/shadow\"]\nresolver = \"2\"\n\n[workspace.dependencies]\ncrate_b = { path = \"crates/crate_b\" }\n",
        )
        .unwrap();
        fs::write(
            repo.join("crates/crate_b/Cargo.toml"),
            "[package]\nname = \"crate_b\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        fs::write(
            repo.join("crates/crate_b/src/lib.rs"),
            "pub fn f() {}\npub mod nested;\n",
        )
        .unwrap();
        fs::write(repo.join("crates/crate_b/src/nested.rs"), "pub fn g() {}\n").unwrap();
        fs::write(
            repo.join("crates/app/Cargo.toml"),
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\ncrate_b = { path = \"../crate_b\" }\ndecoy = \"1.0\"\n",
        )
        .unwrap();
        fs::write(
            repo.join("crates/app/src/lib.rs"),
            "use crate_b::f;\npub fn qualified() { crate_b::f(); crate_b::nested::g(); }\npub fn imported() { f(); }\npub fn not_a_dep() { decoy::f(); }\n",
        )
        .unwrap();
        fs::write(
            repo.join("crates/shadow/Cargo.toml"),
            "[package]\nname = \"shadow\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\ncrate_b = { path = \"../crate_b\" }\n",
        )
        .unwrap();
        fs::write(
            repo.join("crates/shadow/src/lib.rs"),
            "pub fn shadowed() { crate_b::f(); }\n",
        )
        .unwrap();
        fs::write(repo.join("crates/shadow/src/crate_b.rs"), "pub fn f() {}\n").unwrap();
        fs::write(
            repo.join("crates/workspace_user/Cargo.toml"),
            "[package]\nname = \"workspace_user\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\ncrate_b = { workspace = true }\n",
        )
        .unwrap();
        fs::write(
            repo.join("crates/workspace_user/src/lib.rs"),
            "pub fn via_workspace() { crate_b::f(); }\n",
        )
        .unwrap();
        fs::write(
            repo.join("crates/decoy/Cargo.toml"),
            "[package]\nname = \"decoy\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        fs::write(repo.join("crates/decoy/src/lib.rs"), "pub fn f() {}\n").unwrap();

        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();

        assert!(
            calls_between(
                &store,
                "crates/crate_b/src/lib.rs::Function::f",
                "crates/workspace_user/src/lib.rs::Function::via_workspace",
            ),
            "workspace = true path dependency must resolve crate_b::f"
        );
        assert!(
            calls_between(
                &store,
                "crates/crate_b/src/lib.rs::Function::f",
                "crates/app/src/lib.rs::Function::qualified",
            ),
            "path dependency must resolve crate_b::f"
        );
        assert!(
            calls_between(
                &store,
                "crates/crate_b/src/nested.rs::Function::g",
                "crates/app/src/lib.rs::Function::qualified",
            ),
            "path dependency must resolve crate_b::nested::g"
        );
        assert!(
            calls_between(
                &store,
                "crates/shadow/src/crate_b.rs::Function::f",
                "crates/shadow/src/lib.rs::Function::shadowed",
            ),
            "a lexical module file must keep shadowing the extern crate"
        );
        assert!(
            !calls_between(
                &store,
                "crates/crate_b/src/lib.rs::Function::f",
                "crates/shadow/src/lib.rs::Function::shadowed",
            ),
            "shadowed extern crate must not also receive the call"
        );
        assert!(
            calls_between(
                &store,
                "crates/crate_b/src/lib.rs::Function::f",
                "crates/app/src/lib.rs::Function::imported",
            ),
            "use crate_b::f must resolve the imported call"
        );
        assert!(
            !calls_between(
                &store,
                "crates/decoy/src/lib.rs::Function::f",
                "crates/app/src/lib.rs::Function::not_a_dep",
            ),
            "a version-only dependency must not bind a same-named workspace member"
        );
        let _ = fs::remove_dir_all(&repo);
    }

    #[test]
    fn mixed_language_same_name_does_not_suppress_in_language_call() {
        let repo = std::env::temp_dir().join(format!(
            "greppy-indexer-test-lang-family-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/Pricing.java"),
            "class Pricing { static int parsePrice(String raw) { return 1; } }\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/Cart.java"),
            "class Cart { int cartTotal() { return Pricing.parsePrice(\"1\"); } }\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/price.ts"),
            "export function parsePrice(raw: string): number { return 1; }\nexport function cartTotal(): number { return parsePrice(\"1\"); }\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/util.c"),
            "int parsePrice(const char *raw) { return 1; }\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/app.cpp"),
            "int run() { return parsePrice(\"1\"); }\n",
        )
        .unwrap();

        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert!(
            calls_between(
                &store,
                "src/Pricing.java::Pricing::parsePrice",
                "src/Cart.java::Cart::cartTotal",
            ),
            "a TypeScript namesake must not suppress the Java call"
        );
        assert!(
            !calls_between(
                &store,
                "src/price.ts::Function::parsePrice",
                "src/Cart.java::Cart::cartTotal",
            ),
            "a Java call must not bind to a TypeScript function"
        );
        assert!(
            calls_between(
                &store,
                "src/price.ts::Function::parsePrice",
                "src/price.ts::Function::cartTotal",
            ),
            "the TypeScript call still resolves in its own language"
        );
        assert!(
            calls_between(
                &store,
                "src/util.c::Function::parsePrice",
                "src/app.cpp::Function::run",
            ),
            "C and C++ stay one family, so a Java namesake must not suppress them"
        );

        fs::write(
            repo.join("src/Other.java"),
            "class Other { static int parsePrice(String raw) { return 2; } }\n",
        )
        .unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert!(
            !calls_between(
                &store,
                "src/Pricing.java::Pricing::parsePrice",
                "src/Cart.java::Cart::cartTotal",
            ),
            "two Java definitions stay ambiguous"
        );
        assert!(
            calls_between(
                &store,
                "src/price.ts::Function::parsePrice",
                "src/price.ts::Function::cartTotal",
            ),
            "Java ambiguity must not suppress the unique TypeScript call"
        );
        let _ = fs::remove_dir_all(&repo);
    }

    #[test]
    fn rust_local_alias_calls_do_not_bind_unrelated_fn_test() {
        let repo = setup_multifile_repo(
            "rust-local-alias",
            "pub fn real() {}\npub fn test() {}\npub fn assert_de() {}\n",
            "use crate::real as test;\npub fn renamed() { test(); }\npub fn local_path() { let test = assert_de; test(); }\npub fn opaque() { let test = assert_de::<i8>; test(); }\npub fn param(test: fn()) { test(); }\n",
        );
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert!(
            calls_between(
                &store,
                "src/lib.rs::Function::real",
                "src/helper.rs::Function::renamed",
            ),
            "use … as test must call the alias target"
        );
        assert!(
            !calls_between(
                &store,
                "src/lib.rs::Function::test",
                "src/helper.rs::Function::renamed",
            ),
            "a renamed import must not call an unrelated fn test"
        );
        assert!(
            calls_between(
                &store,
                "src/lib.rs::Function::assert_de",
                "src/helper.rs::Function::local_path",
            ),
            "let test = assert_de must call assert_de"
        );
        assert!(
            !calls_between(
                &store,
                "src/lib.rs::Function::test",
                "src/helper.rs::Function::local_path",
            ),
            "a local let binding must not call fn test"
        );
        assert!(
            !calls_between(
                &store,
                "src/lib.rs::Function::test",
                "src/helper.rs::Function::opaque",
            ),
            "a turbofish local must not fall through to fn test"
        );
        assert!(
            !calls_between(
                &store,
                "src/lib.rs::Function::test",
                "src/helper.rs::Function::param",
            ),
            "a parameter named test must not call fn test"
        );
        let _ = fs::remove_dir_all(&repo);
    }

    #[test]
    fn rust_field_chain_and_ufcs_resolve_known_receiver_types() {
        let repo = setup_multifile_repo(
            "rust-field-chain",
            "mod helper;\nmod other;\nuse helper::RenameRule;\nstruct RenameAllRules { serialize: RenameRule }\nimpl RenameAllRules {\n    fn apply(&self, value: &str) -> String { self.serialize.apply_to_field(value) }\n}\nfn apply_param(rules: &RenameAllRules, value: &str) -> String { rules.serialize.apply_to_field(value) }\nfn ufcs(rule: &RenameRule) { RenameRule::apply_to_field(rule, \"x\"); RenameRule::apply_to_variant(rule, \"x\"); }\nfn unknown(rules: RenameAllRules) { rules.missing.apply_to_field(\"x\"); }\nfn opaque() { let rules = opaque_rules(); rules.serialize.apply_to_field(\"x\"); }\nfn opaque_rules() -> RenameAllRules { unimplemented!() }\n",
            "pub enum RenameRule { None, PascalCase }\nimpl RenameRule {\n    pub fn apply_to_field(&self, value: &str) -> String { value.to_string() }\n    pub fn apply_to_variant(&self, value: &str) -> String { value.to_string() }\n}\n",
        );
        fs::write(
            repo.join("src/other.rs"),
            "pub enum RenameRule { None }\nimpl RenameRule {\n    pub fn apply_to_field(&self, value: &str) -> String { value.to_string() }\n    pub fn apply_to_variant(&self, value: &str) -> String { value.to_string() }\n}\n",
        )
        .unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let method = "src/helper.rs::RenameRule::apply_to_field";
        let variant = "src/helper.rs::RenameRule::apply_to_variant";
        let foreign = "src/other.rs::RenameRule::apply_to_field";
        assert!(calls_between(
            &store,
            method,
            "src/lib.rs::RenameAllRules::apply"
        ));
        assert!(calls_between(
            &store,
            method,
            "src/lib.rs::Function::apply_param"
        ));
        assert!(calls_between(&store, method, "src/lib.rs::Function::ufcs"));
        assert!(calls_between(&store, variant, "src/lib.rs::Function::ufcs"));
        assert!(!calls_between(
            &store,
            foreign,
            "src/lib.rs::RenameAllRules::apply"
        ));
        assert!(!calls_between(
            &store,
            foreign,
            "src/lib.rs::Function::apply_param"
        ));
        assert!(!calls_between(
            &store,
            method,
            "src/lib.rs::Function::unknown"
        ));
        assert!(!calls_between(
            &store,
            method,
            "src/lib.rs::Function::opaque"
        ));
        assert!(!calls_between(
            &store,
            foreign,
            "src/lib.rs::Function::ufcs"
        ));
        let _ = fs::remove_dir_all(&repo);
    }

    #[test]
    fn rust_prelude_enum_constructors_do_not_bind_user_functions() {
        let repo = setup_repo(
            "rust-prelude-ctors",
            "pub fn Err() {}\npub fn Ok() {}\npub fn Some() {}\npub enum Outcome { Err(u8), Ok, Some }\npub fn expand_derive_deserialize() {\n    let _ = Err(\"no\");\n    let _ = Ok(1);\n    let _ = Some(1);\n    let _ = Outcome::Err(1);\n}\n",
        );
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let caller = "src/lib.rs::Function::expand_derive_deserialize";
        assert!(!calls_between(&store, "src/lib.rs::Function::Err", caller));
        assert!(!calls_between(&store, "src/lib.rs::Function::Ok", caller));
        assert!(!calls_between(&store, "src/lib.rs::Function::Some", caller));
        assert!(calls_between(&store, "src/lib.rs::Outcome::Err", caller));
        let _ = fs::remove_dir_all(&repo);
    }

    #[test]
    fn rust_enum_variants_resolve_constructors_patterns_and_exact_owners() {
        let repo = setup_repo(
            "enum-variant-references",
            r#"
mod caller;
mod foreign;
use crate::foreign::Remote as Renamed;
pub fn unimported() { let _ = Remote::Halt; }
pub fn remote_alias() { let _ = Renamed::Halt; }
pub enum Instruction { AddImmediateByte { amount: u8 }, Tuple(u8), Halt }
pub enum Other { AddImmediateByte { amount: u8 } }
pub fn decode() -> Instruction { Instruction::AddImmediateByte { amount: 1 } }
pub fn state(value: Instruction) -> u8 {
    match value { Instruction::AddImmediateByte { amount } => amount, _ => 0 }
}
pub fn tuple() -> Instruction { Instruction::Tuple(1) }
pub fn unit() -> Instruction { Instruction::Halt }
pub fn function_item() { let _ = Instruction::Tuple; }
pub fn other() -> Other { Other::AddImmediateByte { amount: 1 } }
pub fn missing() { let _ = missing::Instruction::AddImmediateByte { amount: 1 }; }
impl Instruction {
    pub fn self_pattern(&self) -> u8 {
        match self { Self::AddImmediateByte { amount } => *amount, _ => 0 }
    }
}
"#,
        );
        fs::write(
            repo.join("src/caller.rs"),
            r#"
use crate::Instruction as Opcode;
pub fn aliased() -> Opcode { Opcode::AddImmediateByte { amount: 2 } }
"#,
        )
        .unwrap();
        fs::write(repo.join("src/foreign.rs"), "pub enum Remote { Halt }\n").unwrap();
        let mut store = Store::open(std::path::Path::new(":memory:")).unwrap();
        index(&mut store, &repo, "test").unwrap();
        for (target, callers) in [
            (
                "src/lib.rs::Instruction::AddImmediateByte",
                vec![
                    ("src/lib.rs::Function::decode", "USAGE"),
                    ("src/lib.rs::Function::state", "USAGE"),
                    ("src/lib.rs::Instruction::self_pattern", "USAGE"),
                    ("src/caller.rs::Function::aliased", "USAGE"),
                ],
            ),
            (
                "src/lib.rs::Instruction::Tuple",
                vec![
                    ("src/lib.rs::Function::tuple", "CALLS"),
                    ("src/lib.rs::Function::function_item", "USAGE"),
                ],
            ),
            (
                "src/lib.rs::Instruction::Halt",
                vec![("src/lib.rs::Function::unit", "USAGE")],
            ),
            (
                "src/lib.rs::Other::AddImmediateByte",
                vec![("src/lib.rs::Function::other", "USAGE")],
            ),
        ] {
            let target = store
                .get_node_by_qname("test", target)
                .unwrap()
                .expect("enum variant");
            for (source, edge_type) in callers {
                let source = store
                    .get_node_by_qname("test", source)
                    .unwrap()
                    .expect("caller");
                assert!(
                    store
                        .incoming_edges(target.id, Some(edge_type), 20)
                        .unwrap()
                        .iter()
                        .any(|edge| edge.source_id == source.id),
                    "missing {edge_type} {} -> {}; direct member={:?}; raw={:?}",
                    source.qualified_name,
                    target.qualified_name,
                    GraphIndex::load(&store, "test")
                        .unwrap()
                        .resolve_associated_member(
                            source.id,
                            "Instruction::AddImmediateByte",
                            "AddImmediateByte",
                            &["EnumVariant"]
                        ),
                    load_all_raw_edges(&store, "test")
                        .unwrap()
                        .into_iter()
                        .filter(|edge| edge.source_qualified_name == source.qualified_name)
                        .collect::<Vec<_>>()
                );
            }
        }
        let remote = store
            .get_node_by_qname("test", "src/foreign.rs::Remote::Halt")
            .unwrap()
            .unwrap();
        let imported = store
            .get_node_by_qname("test", "src/lib.rs::Function::remote_alias")
            .unwrap()
            .unwrap();
        let unimported = store
            .get_node_by_qname("test", "src/lib.rs::Function::unimported")
            .unwrap()
            .unwrap();
        let edges = store.incoming_edges(remote.id, Some("USAGE"), 20).unwrap();
        assert!(edges.iter().any(|edge| edge.source_id == imported.id));
        assert!(
            edges.iter().all(|edge| edge.source_id != unimported.id),
            "a unique cross-file enum is not an in-scope binding"
        );
        let missing = store
            .get_node_by_qname("test", "src/lib.rs::Function::missing")
            .unwrap()
            .unwrap();
        assert!(
            store
                .outgoing_edges(missing.id, Some("USAGE"), 20)
                .unwrap()
                .is_empty(),
            "an unknown qualified owner must not bind to a same-name local enum; resolved={:?}; raw={:?}",
            store.outgoing_edges(missing.id, Some("USAGE"), 20).unwrap(),
            load_all_raw_edges(&store, "test")
                .unwrap()
                .into_iter()
                .filter(|edge| edge.source_qualified_name == missing.qualified_name)
                .collect::<Vec<_>>()
        );
        let other = store
            .get_node_by_qname("test", "src/lib.rs::Function::other")
            .unwrap()
            .unwrap();
        let instruction = store
            .get_node_by_qname("test", "src/lib.rs::Instruction::AddImmediateByte")
            .unwrap()
            .unwrap();
        assert!(
            store
                .incoming_edges(instruction.id, Some("USAGE"), 20)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != other.id),
            "same-name variants retain their enum ownership"
        );
        assert!(rust_caller_edges_repaired(&store).unwrap());
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE target_id = ?1 AND edge_type IN ('CALLS', 'USAGE')",
                [instruction.id],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key = ?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        store.conn().execute("INSERT INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v3','complete')", []).unwrap();
        store.conn().execute("DELETE FROM raw_edges WHERE target_qname LIKE '%AddImmediateByte%' AND edge_type='USAGE'", []).unwrap();
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(rust_caller_edges_repaired(&store).unwrap());
        assert!(
            !store
                .incoming_edges(instruction.id, Some("USAGE"), 20)
                .unwrap()
                .is_empty(),
            "an unchanged older store recovers omitted constructor raw references"
        );
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn persisted_rust_recovery_validates_winning_enum_fields_in_sparse_overlay() {
        let repo = setup_repo(
            "enum-field-recovery",
            "pub enum Command { Index { path: Option<String> }, Write { path: String } }\npub fn invoke(command: Command) { let _ = command; }\n",
        );
        let base_path = repo.join("base.db");
        let delta_path = repo.join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, &repo, "test").unwrap();
            let field = base
                .get_node_by_qname("test", "src/lib.rs::Enum::Command::path")
                .unwrap()
                .unwrap();
            assert_eq!(field.properties["return_type"], "String");
            base.conn()
                .execute("DELETE FROM raw_edges WHERE edge_type='USAGE'", [])
                .unwrap();
        }
        let base_before = fs::read(&base_path).unwrap();
        let visibility =
            greppy_store::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        let nodes_before = format!("{:?}", overlay.list_nodes("test", "", "", 0, 100).unwrap());
        let states_before = overlay.list_file_states("test").unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).unwrap() > 0);
        assert_eq!(
            recover_persisted_rust_usages(&mut overlay, "test", &repo).unwrap(),
            0
        );
        assert_eq!(overlay.list_file_states("test").unwrap(), states_before);
        assert!(overlay.list_private_file_states("test").unwrap().is_empty());
        assert!(overlay.list_delta_raw_edges("test").unwrap().is_empty());
        assert_eq!(
            format!("{:?}", overlay.list_nodes("test", "", "", 0, 100).unwrap()),
            nodes_before
        );
        drop(overlay);
        assert_eq!(fs::read(&base_path).unwrap(), base_before);
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn persisted_rust_recovery_rejects_missing_or_corrupt_winning_enum_fields() {
        let repo = setup_repo(
            "enum-field-recovery-negative",
            "pub enum Command { Index { path: Option<String> }, Write { path: String } }\npub fn invoke(command: Command) { let _ = command; }\n",
        );
        for sql in [
            "UPDATE nodes SET properties=json_set(properties, '$.return_type', 'Option<String>') WHERE label='Field' AND name='path'",
            "DELETE FROM nodes WHERE label='Field' AND name='path'",
        ] {
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            store.conn().execute(sql, []).unwrap();
            let raw_before = format!("{:?}", store.list_raw_edges("test").unwrap());
            let error = recover_persisted_rust_usages(&mut store, "test", &repo)
                .unwrap_err()
                .to_string();
            assert!(error.contains("declared field facts"), "{error}");
            assert_eq!(
                format!("{:?}", store.list_raw_edges("test").unwrap()),
                raw_before
            );
        }
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn persisted_rust_recovery_validates_winning_cfg_trait_facts() {
        let repo = setup_repo(
            "cfg-trait-recovery",
            "#[cfg(feature=\"first\")] pub trait View { fn as_ref(&self); }\n#[cfg(not(feature=\"first\"))] pub trait View: core::fmt::Debug { fn as_ref(self); }\n",
        );
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let winner = store
            .get_node_by_qname("test", "src/lib.rs::Interface::View")
            .unwrap()
            .unwrap();
        assert_eq!(winner.properties["has_bounds"], 1);
        recover_persisted_rust_usages(&mut store, "test", &repo).unwrap();
        store
            .conn()
            .execute(
                "UPDATE nodes SET properties=json_remove(properties, '$.has_bounds') WHERE id=?1",
                [winner.id],
            )
            .unwrap();
        let error = recover_persisted_rust_usages(&mut store, "test", &repo)
            .unwrap_err()
            .to_string();
        assert!(error.contains("trait receiver facts"), "{error}");
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn persisted_rust_usage_recovery_validates_all_sources_and_preserves_sparse_base() {
        let repo = setup_repo(
            "constructor-recovery",
            "pub enum Instruction { AddImmediateByte { amount: u8 } }\npub fn decode() -> Instruction { Instruction::AddImmediateByte { amount: 1 } }\npub fn amount() {}\npub fn valid() { let _ = amount; }\n",
        );
        let base_path = repo.join("base.db");
        let delta_path = repo.join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, &repo, "test").unwrap();
            base.conn().execute("DELETE FROM raw_edges WHERE target_qname LIKE '%AddImmediateByte%' AND edge_type='USAGE'", []).unwrap();
            let decode = base
                .get_node_by_qname("test", "src/lib.rs::Function::decode")
                .unwrap()
                .unwrap();
            let amount = base
                .get_node_by_qname("test", "src/lib.rs::Function::amount")
                .unwrap()
                .unwrap();
            base.insert_raw_edges(&[NewRawEdge {
                project: "test".into(),
                file_path: "src/lib.rs".into(),
                source_qname: decode.qualified_name.clone(),
                target_qname: amount.qualified_name.clone(),
                edge_type: "USAGE".into(),
                properties: serde_json::json!({"ref_name": "amount", "line": 2}),
            }])
            .unwrap();
            base.insert_edge(&NewEdge {
                project: "test".into(),
                source_id: decode.id,
                target_id: amount.id,
                edge_type: "USAGE".into(),
                properties: serde_json::json!({"ref_name": "amount"}),
            })
            .unwrap();
            base.conn().execute("DELETE FROM edges WHERE target_id IN (SELECT id FROM nodes WHERE name='AddImmediateByte')", []).unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
        }
        let visibility =
            greppy_store::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        let states_before = overlay.list_file_states("test").unwrap();
        let nodes_before = format!("{:?}", overlay.list_nodes("test", "", "", 0, 100).unwrap());
        let base_raw_before = format!(
            "{:?}",
            Store::open(&base_path)
                .unwrap()
                .list_raw_edges("test")
                .unwrap()
        );
        let original = fs::read(repo.join("src/lib.rs")).unwrap();
        fs::write(repo.join("src/lib.rs"), "pub fn changed() {}\n").unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).is_err());
        assert!(overlay.list_delta_raw_edges("test").unwrap().is_empty());
        fs::remove_file(repo.join("src/lib.rs")).unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).is_err());
        assert!(overlay.list_delta_raw_edges("test").unwrap().is_empty());
        fs::write(repo.join("src/lib.rs"), original).unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).unwrap() > 0);
        rebuild_visible_overlay_edges(&mut overlay, "test").unwrap();
        let target = overlay
            .get_node_by_qname("test", "src/lib.rs::Instruction::AddImmediateByte")
            .unwrap()
            .unwrap();
        let caller = overlay
            .get_node_by_qname("test", "src/lib.rs::Function::decode")
            .unwrap()
            .unwrap();
        assert!(overlay
            .incoming_edges(target.id, Some("USAGE"), 20)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id));
        let amount = overlay
            .get_node_by_qname("test", "src/lib.rs::Function::amount")
            .unwrap()
            .unwrap();
        let valid = overlay
            .get_node_by_qname("test", "src/lib.rs::Function::valid")
            .unwrap()
            .unwrap();
        let usages = overlay
            .incoming_edges(amount.id, Some("USAGE"), 20)
            .unwrap();
        assert!(
            usages.iter().all(|edge| edge.source_id != caller.id),
            "obsolete struct field label must not resolve to a free function; usages={usages:?}; raw={:?}; mask={:?}",
            overlay.list_raw_edges("test").unwrap(),
            overlay.conn().query_row("SELECT value FROM main.schema_meta WHERE key='greppy.rust_usage_override_files.test'", [], |row| row.get::<_, String>(0))
        );
        assert!(
            usages.iter().any(|edge| edge.source_id == valid.id),
            "legitimate raw usage remains visible"
        );
        assert!(
            overlay
                .outgoing_edges(caller.id, Some("USAGE"), 20)
                .unwrap()
                .iter()
                .all(|edge| edge.target_id != amount.id),
            "typed outgoing queries must honor the same repair mask as incoming queries"
        );
        assert!(overlay
            .outgoing_edges(valid.id, Some("USAGE"), 20)
            .unwrap()
            .iter()
            .any(|edge| edge.target_id == amount.id));
        assert!(
            overlay.list_delta_raw_edges("test").unwrap().is_empty(),
            "Base compatibility repair must not enter sparse raw re-resolution"
        );
        assert_eq!(
            recover_persisted_rust_usages(&mut overlay, "test", &repo).unwrap(),
            0
        );
        assert!(overlay.list_private_file_states("test").unwrap().is_empty());
        assert_eq!(overlay.list_file_states("test").unwrap(), states_before);
        assert_eq!(
            format!("{:?}", overlay.list_nodes("test", "", "", 0, 100).unwrap()),
            nodes_before
        );
        rebuild_overlay_edges(&mut overlay, "test").unwrap();
        assert!(overlay
            .incoming_edges(target.id, Some("USAGE"), 20)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id));
        let usages = overlay
            .incoming_edges(amount.id, Some("USAGE"), 20)
            .unwrap();
        assert!(usages.iter().all(|edge| edge.source_id != caller.id));
        assert!(usages.iter().any(|edge| edge.source_id == valid.id));
        drop(overlay);
        // Persisted override works after a normal overlay reopen too.
        let reopened = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        assert!(reopened
            .incoming_edges(amount.id, Some("USAGE"), 20)
            .unwrap()
            .iter()
            .all(|edge| edge.source_id != caller.id));
        drop(reopened);
        let base = Store::open(&base_path).unwrap();
        assert_eq!(
            format!("{:?}", base.list_raw_edges("test").unwrap()),
            base_raw_before,
            "all immutable Base raw edges are preserved exactly"
        );
        drop(base);
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn cargo_library_paths_resolve_grouped_alias_and_example_callers() {
        for (manifest_lib, library_file, library_name) in [
            ("", "src/lib.rs", "vcop2_tools"),
            (
                "[lib]\nname = \"tool_api\"\npath = \"./engine/entry.rs\"\n",
                "engine/entry.rs",
                "tool_api",
            ),
        ] {
            let repo = setup_repo("cargo-library-callers", "// conventional source\n");
            fs::write(repo.join("Cargo.toml"), format!("[package]\nname = \"vcop2-tools\"\nversion = \"0.1.0\"\nedition = \"2021\"\n{manifest_lib}")).unwrap();
            let library_parent = Path::new(library_file).parent().unwrap();
            fs::create_dir_all(repo.join(library_parent)).unwrap();
            fs::write(
                repo.join(library_file),
                "pub mod m68000_aot; pub fn root_target() {}\n",
            )
            .unwrap();
            let target_file = library_parent.join("m68000_aot.rs");
            fs::write(
                repo.join(&target_file),
                "pub fn compile() {}\npub fn compile_reachable() {}\n",
            )
            .unwrap();
            fs::create_dir_all(repo.join("tests")).unwrap();
            fs::create_dir_all(repo.join("examples")).unwrap();
            fs::write(repo.join("tests/grouped.rs"), format!("use {library_name}::m68000_aot::{{compile, compile_reachable as reach}};\nfn grouped() {{ compile(); reach(); }}\n")).unwrap();
            fs::write(repo.join("tests/module.rs"), format!("use {library_name}::m68000_aot as engine;\nfn module_alias() {{ engine::compile(); }}\n")).unwrap();
            fs::write(repo.join("tests/library.rs"), format!("use {library_name} as api;\nfn library_alias() {{ api::m68000_aot::compile(); }}\n")).unwrap();
            fs::write(repo.join("examples/qualified.rs"), format!("fn example() {{ {library_name}::m68000_aot::compile(); {library_name}::root_target(); }}\nfn function_item() {{ let _ = {library_name}::m68000_aot::compile_reachable; }}\n")).unwrap();
            fs::write(repo.join("tests/external.rs"), "use external_crate::m68000_aot::compile;\nfn external() { compile(); external_crate::m68000_aot::compile(); }\n").unwrap();
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            let target = store
                .get_node_by_qname(
                    "test",
                    &format!("{}::Function::compile", target_file.to_string_lossy()),
                )
                .unwrap()
                .unwrap();
            for (file, caller) in [
                ("tests/grouped.rs", "grouped"),
                ("tests/module.rs", "module_alias"),
                ("tests/library.rs", "library_alias"),
                ("examples/qualified.rs", "example"),
            ] {
                let caller = store
                    .get_node_by_qname("test", &format!("{file}::Function::{caller}"))
                    .unwrap()
                    .unwrap();
                assert!(
                    store
                        .outgoing_edges(caller.id, Some("CALLS"), 20)
                        .unwrap()
                        .iter()
                        .any(|edge| edge.target_id == target.id),
                    "missing Cargo library caller in {file}"
                );
            }
            let external = store
                .get_node_by_qname("test", "tests/external.rs::Function::external")
                .unwrap()
                .unwrap();
            assert!(
                store
                    .outgoing_edges(external.id, Some("CALLS"), 20)
                    .unwrap()
                    .is_empty(),
                "external crate must not resolve by basename"
            );
            let reachable = store
                .get_node_by_qname(
                    "test",
                    &format!(
                        "{}::Function::compile_reachable",
                        target_file.to_string_lossy()
                    ),
                )
                .unwrap()
                .unwrap();
            let item = store
                .get_node_by_qname("test", "examples/qualified.rs::Function::function_item")
                .unwrap()
                .unwrap();
            assert!(store
                .outgoing_edges(item.id, Some("USAGE"), 20)
                .unwrap()
                .iter()
                .any(|edge| edge.target_id == reachable.id));
            let grouped = store
                .get_node_by_qname("test", "tests/grouped.rs::Function::grouped")
                .unwrap()
                .unwrap();
            assert!(store
                .outgoing_edges(grouped.id, Some("CALLS"), 20)
                .unwrap()
                .iter()
                .any(|edge| edge.target_id == reachable.id));
            let root = store
                .get_node_by_qname("test", &format!("{library_file}::Function::root_target"))
                .unwrap()
                .unwrap();
            assert!(!store
                .incoming_edges(root.id, Some("CALLS"), 20)
                .unwrap()
                .is_empty());
            assert_eq!(
                index(&mut store, &repo, "test").unwrap().files_indexed,
                0,
                "unchanged Cargo metadata must preserve incremental extraction"
            );
            let stores = tempfile::tempdir().unwrap();
            let base_path = stores.path().join("base.db");
            let delta_path = stores.path().join("delta.db");
            {
                let mut base = Store::open(&base_path).unwrap();
                index(&mut base, &repo, "test").unwrap();
                // Simulate an old resolver: retain raw edges and all nodes,
                // but remove the previously missing Cargo library relations.
                base.conn()
                    .execute(
                        "DELETE FROM edges WHERE edge_type IN ('CALLS', 'USAGE', 'IMPORTS')",
                        [],
                    )
                    .unwrap();
            }
            let visibility =
                greppy_store::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new())
                    .unwrap();
            let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
            let before = overlay
                .list_nodes_by_label("test", "Function", 100)
                .unwrap();
            rebuild_visible_overlay_edges(&mut overlay, "test").unwrap();
            let repaired_target = overlay
                .get_node_by_qname("test", &target.qualified_name)
                .unwrap()
                .unwrap();
            assert_eq!(
                overlay
                    .incoming_edges(repaired_target.id, Some("CALLS"), 20)
                    .unwrap()
                    .len(),
                4
            );
            assert!(
                overlay.list_private_file_states("test").unwrap().is_empty(),
                "repair must not copy Base file state into Delta"
            );
            assert_eq!(
                overlay
                    .list_nodes_by_label("test", "Function", 100)
                    .unwrap()
                    .iter()
                    .map(|node| node.id)
                    .collect::<Vec<_>>(),
                before.iter().map(|node| node.id).collect::<Vec<_>>(),
                "repair must preserve cached node identities"
            );
            rebuild_overlay_edges(&mut overlay, "test").unwrap();
            assert_eq!(
                overlay
                    .incoming_edges(repaired_target.id, Some("CALLS"), 20)
                    .unwrap()
                    .len(),
                4,
                "ordinary sparse rebuild preserves the one-shot repair"
            );
            drop(overlay);
            fs::remove_dir_all(repo).unwrap();
        }
    }

    #[test]
    fn cargo_library_name_does_not_override_a_local_binary_module() {
        let repo = setup_repo("cargo-library-shadowing", "pub fn compile() {}\n");
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname = \"tools\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/main.rs"),
            "mod tools; fn main() { tools::compile(); }\n",
        )
        .unwrap();
        fs::write(repo.join("src/tools.rs"), "pub fn compile() {}\n").unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let caller = store
            .get_node_by_qname("test", "src/main.rs::Function::main")
            .unwrap()
            .unwrap();
        let local = store
            .get_node_by_qname("test", "src/tools.rs::Function::compile")
            .unwrap()
            .unwrap();
        let library = store
            .get_node_by_qname("test", "src/lib.rs::Function::compile")
            .unwrap()
            .unwrap();
        let edges = store.outgoing_edges(caller.id, Some("CALLS"), 20).unwrap();
        assert!(edges.iter().any(|edge| edge.target_id == local.id));
        assert!(edges.iter().all(|edge| edge.target_id != library.id));
        assert!(
            rust_caller_edges_repaired(&store).unwrap(),
            "fresh index already uses current resolver"
        );
        reset_reresolve_counter();
        let unchanged = index(&mut store, &repo, "test").unwrap();
        assert_eq!(unchanged.files_indexed, 0);
        assert_eq!(
            reresolve_count(),
            0,
            "current unchanged index must not repeat repair"
        );
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn cargo_library_resolution_does_not_guess_ambiguous_modules_or_other_packages() {
        let repo = setup_repo("cargo-library-negative", "pub mod engine;\n");
        fs::write(repo.join("Cargo.toml"), "[package]\nname = \"tools\"\nversion = \"0.1.0\"\n[workspace]\nmembers = [\"binary\"]\n").unwrap();
        fs::create_dir_all(repo.join("src/engine")).unwrap();
        fs::create_dir_all(repo.join("tests")).unwrap();
        fs::write(repo.join("src/engine.rs"), "pub fn compile() {}\n").unwrap();
        fs::write(repo.join("src/engine/mod.rs"), "pub fn compile() {}\n").unwrap();
        fs::write(
            repo.join("tests/ambiguous.rs"),
            "fn ambiguous() { tools::engine::compile(); }\n",
        )
        .unwrap();
        fs::create_dir_all(repo.join("binary/src")).unwrap();
        fs::write(
            repo.join("binary/Cargo.toml"),
            "[package]\nname = \"binary\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(
            repo.join("binary/src/main.rs"),
            "fn unrelated() { tools::engine::compile(); }\n",
        )
        .unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        for qname in [
            "tests/ambiguous.rs::Function::ambiguous",
            "binary/src/main.rs::Function::unrelated",
        ] {
            let caller = store.get_node_by_qname("test", qname).unwrap().unwrap();
            assert!(store
                .outgoing_edges(caller.id, Some("CALLS"), 20)
                .unwrap()
                .is_empty());
        }
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn cargo_manifest_targets_cover_workspace_custom_and_implicit_bins() {
        let repo = tempfile::tempdir().unwrap();
        let custom = repo.path().join("custom");
        let implicit = repo.path().join("implicit");
        fs::create_dir_all(custom.join("src/core")).unwrap();
        fs::create_dir_all(custom.join("src/core/decoy")).unwrap();
        fs::create_dir_all(implicit.join("src/bin/nested")).unwrap();
        fs::write(
            repo.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"*\"]\n",
        )
        .unwrap();
        fs::write(
            custom.join("Cargo.toml"),
            "[package]\nname = \"custom\"\nversion = \"0.1.0\"\n\n[[bin]]\nname = \"custom\"\npath = \"src/core/main.rs\"\n",
        )
        .unwrap();
        fs::write(
            implicit.join("Cargo.toml"),
            "[package]\nname = \"implicit\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        for path in [
            "custom/src/core/main.rs",
            "custom/src/core/decoy/main.rs",
            "implicit/src/main.rs",
            "implicit/src/bin/tool.rs",
            "implicit/src/bin/nested/main.rs",
            "implicit/src/bin/nested/helper.rs",
        ] {
            fs::write(repo.path().join(path), "fn item() {}\n").unwrap();
        }
        let known_files = [
            "custom/src/core/main.rs",
            "custom/src/core/decoy/main.rs",
            "implicit/src/main.rs",
            "implicit/src/bin/tool.rs",
            "implicit/src/bin/nested/main.rs",
            "implicit/src/bin/nested/helper.rs",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let mut roots = std::collections::HashSet::new();
        // Match the production caller, which canonicalizes the repository
        // before comparing canonical Cargo targets (TMPDIR may be an alias).
        let repository_root = std::fs::canonicalize(repo.path()).unwrap();
        assert!(rust_crate_roots_from_manifest(
            &repository_root.join("Cargo.toml"),
            &repository_root,
            &known_files,
            &mut roots,
            &mut Vec::new(),
            &mut std::collections::HashSet::new(),
        ));
        assert!(roots.contains("custom/src/core/main.rs"));
        assert!(!roots.contains("custom/src/core/decoy/main.rs"));
        assert!(roots.contains("implicit/src/main.rs"));
        assert!(roots.contains("implicit/src/bin/tool.rs"));
        assert!(roots.contains("implicit/src/bin/nested/main.rs"));
        assert!(!roots.contains("implicit/src/bin/nested/helper.rs"));
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
    fn module_qualified_call_targets_the_named_module_not_the_same_file_twin() {
        // `app.rs` defines its own `resolve_root` AND calls
        // `store::resolve_root()`. The same-file preference used to attribute
        // the qualified call to the local twin and leave the store function
        // without callers (readiness ledger, item 5). The explicit module
        // qualifier decides; the unqualified call still resolves same-file.
        const LIB_RS: &str = r#"
            pub mod store;
            pub mod app;
        "#;
        const STORE_RS: &str = r#"
            pub fn resolve_root() -> u32 { 1 }
        "#;
        const APP_RS: &str = r#"
            use crate::store;
            pub fn resolve_root() -> u32 { 2 }
            pub fn use_store_root() -> u32 { store::resolve_root() }
            pub fn use_local_root() -> u32 { resolve_root() }
        "#;
        let repo = setup_multifile_repo("qualified", LIB_RS, STORE_RS);
        std::fs::write(repo.join("src/app.rs"), APP_RS).unwrap();
        std::fs::rename(repo.join("src/helper.rs"), repo.join("src/store.rs")).unwrap();
        let mut store = Store::open_memory().unwrap();
        let _ = index(&mut store, &repo, "test").unwrap();

        let store_fn = store
            .get_node_by_qname("test", "src/store.rs::Function::resolve_root")
            .unwrap()
            .expect("store::resolve_root must exist");
        let app_fn = store
            .get_node_by_qname("test", "src/app.rs::Function::resolve_root")
            .unwrap()
            .expect("app::resolve_root must exist");
        let calls_of = |caller: &str| -> Vec<i64> {
            let node = store
                .get_node_by_qname("test", caller)
                .unwrap()
                .unwrap_or_else(|| panic!("{caller} must exist"));
            store
                .outgoing_edges(node.id, None, 256)
                .unwrap()
                .into_iter()
                .filter(|e| e.edge_type == "CALLS")
                .map(|e| e.target_id)
                .collect()
        };
        assert_eq!(
            calls_of("src/app.rs::Function::use_store_root"),
            vec![store_fn.id],
            "store::resolve_root() must target the store module, not the same-file twin"
        );
        assert_eq!(
            calls_of("src/app.rs::Function::use_local_root"),
            vec![app_fn.id],
            "an unqualified call keeps the same-file resolution"
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

        let (out, _throttled) = parallel_extract(&supported, 8, &mut |_, _| {});
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

    fn receiver_resolution_work(n: usize) -> usize {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&greppy_store::Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let mut ids = Vec::new();
        for i in 0..n {
            ids.push(
                store
                    .insert_node(&NewNode {
                        project: "p".into(),
                        label: "Method".into(),
                        name: format!("method{i}"),
                        qualified_name: format!("src/m{i}.rs::Owner{i}::method{i}"),
                        file_path: format!("src/m{i}.rs"),
                        start_line: 1,
                        end_line: 2,
                        properties: serde_json::json!({}),
                    })
                    .unwrap(),
            );
        }
        let index = GraphIndex::load(&store, "p").unwrap();
        reset_edge_resolution_work_counter();
        for (i, id) in ids.into_iter().enumerate() {
            assert_eq!(
                index.resolve_receiver_method(
                    "src/caller.rs",
                    &format!("Owner{i}"),
                    &format!("method{i}")
                ),
                Some(id)
            );
        }
        edge_resolution_work_count()
    }

    #[test]
    fn cross_file_receiver_resolution_avoids_project_wide_scans() {
        let small = receiver_resolution_work(1000);
        let large = receiver_resolution_work(4000);
        assert!(
            large < small * 5,
            "4x receiver calls/graph grew from {small} to {large} work; global scans grow quadratically"
        );
    }

    #[test]
    fn direct_self_field_receiver_uses_declared_type_and_repairs_completed_cache() {
        let source = "mod recovery; mod other; struct GameRuntime { recovery: recovery::Recovery } impl GameRuntime { fn frame(&mut self, reason: String) { self.recovery.device_lost(reason); } }";
        let repo = setup_repo("direct-self-field", source);
        fs::write(repo.join("src/recovery.rs"), "pub struct Recovery; impl Recovery { pub fn device_lost(&mut self, reason: String) {} }").unwrap();
        fs::write(repo.join("src/other.rs"), "pub struct Recovery; impl Recovery { pub fn device_lost(&mut self, reason: String) {} }").unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let method = store
            .get_node_by_qname("test", "src/recovery.rs::Recovery::device_lost")
            .unwrap()
            .unwrap();
        let namesake = store
            .get_node_by_qname("test", "src/other.rs::Recovery::device_lost")
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1
        );
        assert!(store
            .incoming_edges(namesake.id, Some("CALLS"), 20)
            .unwrap()
            .is_empty());
        let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
        let states = store.list_file_states("test").unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE edge_type='CALLS' AND target_id=?1",
                [method.id],
            )
            .unwrap();
        store.conn().execute("UPDATE raw_edges SET properties=json_remove(properties,'$.receiver_provenance') WHERE edge_type='CALLS'", []).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [format!("{DIRECT_SELF_FIELD_REPAIR_KEY}.test")],
            )
            .unwrap();
        assert!(rust_caller_edges_repaired(&store).unwrap());
        assert!(!direct_self_field_edges_repaired(&store).unwrap());
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(direct_self_field_edges_repaired(&store).unwrap());
        assert_eq!(
            store
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            nodes,
            format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        assert_eq!(states, store.list_file_states("test").unwrap());
        for (label, changed, recovery) in [
            (
                "opaque",
                source.replace("recovery::Recovery", "Unknown"),
                "pub struct Recovery; impl Recovery { pub fn device_lost(&mut self, reason: String) {} }",
            ),
            (
                "ambiguous",
                source
                    .replace(
                        "mod other;",
                        "mod other; use recovery::Recovery; use other::Recovery;",
                    )
                    .replace("recovery: recovery::Recovery", "recovery: Recovery"),
                "pub struct Recovery; impl Recovery { pub fn device_lost(&mut self, reason: String) {} }",
            ),
            (
                "trait-method",
                source.to_string(),
                "pub struct Recovery; pub trait Lost { fn device_lost(&mut self, reason: String); } impl Lost for Recovery { fn device_lost(&mut self, reason: String) {} }",
            ),
            (
                "opaque-expression",
                source.replace("self.recovery.device_lost", "opaque().device_lost"),
                "pub struct Recovery; impl Recovery { pub fn device_lost(&mut self, reason: String) {} }",
            ),
        ] {
            fs::write(repo.join("src/lib.rs"), changed).unwrap();
            fs::write(repo.join("src/recovery.rs"), recovery).unwrap();
            index(&mut store, &repo, "test").unwrap();
            let method = store
                .get_node_by_qname("test", "src/recovery.rs::Recovery::device_lost")
                .unwrap()
                .unwrap();
            assert!(
                store
                    .incoming_edges(method.id, Some("CALLS"), 20)
                    .unwrap()
                    .is_empty(),
                "{label}"
            );
        }
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn direct_self_field_overlay_recovery_preserves_base_and_refuses_stale_source() {
        let source = "mod recovery; struct GameRuntime { recovery: recovery::Recovery } impl GameRuntime { fn frame(&mut self, reason: String) { self.recovery.device_lost(reason); } }";
        let repo = setup_repo("direct-self-field-overlay", source);
        fs::write(repo.join("src/recovery.rs"), "pub struct Recovery; impl Recovery { pub fn device_lost(&mut self, reason: String) {} }").unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, &repo, "test").unwrap();
            base.conn().execute("DELETE FROM edges WHERE edge_type='CALLS' AND target_id IN (SELECT id FROM nodes WHERE name='device_lost')", []).unwrap();
            base.conn().execute("UPDATE raw_edges SET properties=json_remove(properties,'$.receiver_provenance') WHERE edge_type='CALLS'", []).unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [format!("{DIRECT_SELF_FIELD_REPAIR_KEY}.test")],
                )
                .unwrap();
        }
        let bytes = fs::read(&base_path).unwrap();
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        let nodes = format!("{:?}", overlay.list_nodes("test", "", "", 0, 1000).unwrap());
        let states = format!("{:?}", overlay.list_file_states("test").unwrap());
        let workspace = format!(
            "{:?}",
            overlay.get_workspace_state(repo.to_str().unwrap()).unwrap()
        );
        assert!(!direct_self_field_edges_repaired(&overlay).unwrap());
        recover_persisted_rust_usages(&mut overlay, "test", &repo).unwrap();
        rebuild_visible_overlay_edges(&mut overlay, "test").unwrap();
        assert!(direct_self_field_edges_repaired(&overlay).unwrap());
        let method = overlay
            .get_node_by_qname("test", "src/recovery.rs::Recovery::device_lost")
            .unwrap()
            .unwrap();
        assert_eq!(
            overlay
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            nodes,
            format!("{:?}", overlay.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        assert_eq!(
            states,
            format!("{:?}", overlay.list_file_states("test").unwrap())
        );
        assert_eq!(
            workspace,
            format!(
                "{:?}",
                overlay.get_workspace_state(repo.to_str().unwrap()).unwrap()
            )
        );
        assert_eq!(bytes, fs::read(&base_path).unwrap());
        overlay
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [format!("{DIRECT_SELF_FIELD_REPAIR_KEY}.test")],
            )
            .unwrap();
        fs::write(
            repo.join("src/lib.rs"),
            source.replace("recovery::Recovery", "Unknown"),
        )
        .unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).is_err());
        assert!(!direct_self_field_edges_repaired(&overlay).unwrap());
        assert_eq!(bytes, fs::read(&base_path).unwrap());
        fs::remove_dir_all(repo).unwrap();
    }

    const OPTION_FIELD_CALLER: &str = r#"
mod scene;
pub fn load_scene() {
    let manifest: crate::scene::Manifest = opaque();
    let gi_matrix: Option<[f32;16]> = None;
    match (manifest.remaster_irradiance.as_ref(), gi_matrix) {
        (Some(field), Some(matrix)) => { field.storage(); field.uniform(matrix); },
        _ => (),
    }
}
"#;
    const OPTION_FIELD_SCENE: &str = r#"
pub struct Manifest { pub remaster_irradiance: Option<crate::scene::IrradianceField> }
pub struct IrradianceField;
impl IrradianceField {
    pub fn storage(&self) {}
    pub fn uniform(&self, matrix: [f32;16]) {}
}
pub struct Other;
impl Other { pub fn uniform(&self, matrix: [f32;16]) {} }
"#;

    fn assert_option_field_caller(store: &Store, target: &str, present: bool) {
        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::load_scene")
            .unwrap()
            .unwrap();
        let method = store.get_node_by_qname("test", target).unwrap().unwrap();
        assert_eq!(
            store
                .incoming_edges(method.id, Some("CALLS"), 100)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == caller.id),
            present,
            "{target}"
        );
    }

    #[test]
    fn option_field_tuple_receiver_resolves_across_files_and_tracks_declared_type_changes() {
        let repo = setup_repo("option-field-tuple", OPTION_FIELD_CALLER);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::storage", true);
        assert_option_field_caller(&store, "src/scene.rs::Other::uniform", false);
        // Same Field identity but different payload: unchanged caller facts must
        // be re-resolved, not incorrectly retained by the body-edit fast path.
        fs::write(
            repo.join("src/scene.rs"),
            OPTION_FIELD_SCENE.replace(
                "Option<crate::scene::IrradianceField>",
                "Option<crate::scene::Other>",
            ),
        )
        .unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", false);
        assert_option_field_caller(&store, "src/scene.rs::Other::uniform", true);
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn option_field_receiver_preserves_owner_ambiguity_and_shadowing() {
        for (label, caller, scene) in [
            ("opaque-wildcard", OPTION_FIELD_CALLER.replace(
                "mod scene;", "mod scene; use custom::*;"), OPTION_FIELD_SCENE.to_string()),
            ("opaque-consuming-trait", OPTION_FIELD_CALLER.replace(
                "mod scene;", "mod scene; trait Consume { fn as_ref(self) -> Option<crate::scene::Other>; } impl Consume for Option<crate::scene::IrradianceField> { fn as_ref(self) -> Option<crate::scene::Other> { None } }"), OPTION_FIELD_SCENE.to_string()),
            ("late-value-item", OPTION_FIELD_CALLER.replace(
                "{ field.storage(); field.uniform(matrix); }", "{ field.storage(); field.uniform(matrix); const field: crate::scene::Other = crate::scene::Other; }"), OPTION_FIELD_SCENE.to_string()),
            ("shadowed-base", OPTION_FIELD_CALLER.replace(
                "let gi_matrix:", "let manifest = opaque(); let gi_matrix:"), OPTION_FIELD_SCENE.to_string()),
            ("generic-base", OPTION_FIELD_CALLER.replace(
                "let manifest: crate::scene::Manifest", "let manifest: Manifest").replace("pub fn load_scene()", "pub fn load_scene<Manifest>()"), OPTION_FIELD_SCENE.to_string()),
            ("ambiguous-owner", OPTION_FIELD_CALLER.replace(
                "mod scene;", "mod scene; use crate::scene::Manifest; use crate::other::Manifest; mod other;").replace("let manifest: crate::scene::Manifest", "let manifest: Manifest"), OPTION_FIELD_SCENE.to_string()),
            ("ambiguous-payload", OPTION_FIELD_CALLER.to_string(), OPTION_FIELD_SCENE.replace(
                "IrradianceField", "LocalField").replace(
                "pub struct Manifest", "use crate::other::IrradianceField; use crate::foreign::IrradianceField; pub struct Manifest").replace(
                "Option<crate::scene::LocalField>", "Option<IrradianceField>")),
            ("custom-option", OPTION_FIELD_CALLER.to_string(), OPTION_FIELD_SCENE.replace(
                "pub struct Manifest", "pub enum Option<T> { Some(T), None } pub struct Manifest")),
        ] {
            let repo = setup_repo(label, &caller);
            fs::write(repo.join("src/scene.rs"), scene).unwrap();
            fs::write(repo.join("src/other.rs"), "pub struct Manifest { pub remaster_irradiance: Option<IrradianceField> } pub struct IrradianceField; impl IrradianceField { pub fn uniform(&self) {} }").unwrap();
            fs::write(repo.join("src/foreign.rs"), "pub struct IrradianceField; impl IrradianceField { pub fn uniform(&self) {} }").unwrap();
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            let caller = store.get_node_by_qname("test", "src/lib.rs::Function::load_scene").unwrap().unwrap();
            assert!(store.outgoing_edges(caller.id, Some("CALLS"), 100).unwrap().iter().all(|edge| {
                store.get_node(edge.target_id).unwrap().unwrap().name != "uniform"
            }), "{label}: ownership must remain unresolved");
            let unresolved_uniform = store.outgoing_edges(caller.id, Some("UNRESOLVED_CALLS"), 100).unwrap().into_iter().any(|edge| {
                store.get_node(edge.target_id).unwrap().unwrap().name == "uniform"
            });
            // A wildcard leaves one declared candidate unproven. A consuming
            // adapter, shadow, or ambiguous owner is not a candidate at all.
            assert_eq!(unresolved_uniform, label == "opaque-wildcard", "{label}");
            fs::remove_dir_all(repo).unwrap();
        }
    }

    #[test]
    fn legacy_upgrade_reextracts_unchanged_trait_receiver_facts() {
        for prior_version in ["greppy-indexer-v7", "greppy-indexer-v8"] {
            let repo = setup_repo(
                &format!("{prior_version}-trait-receiver-upgrade"),
                "pub trait HttpTransport: Send + Sync { fn execute(&self); }\n",
            );
            fs::write(repo.join("src/changed.rs"), "pub fn changed() {}\n").unwrap();
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            let source_before = store.list_file_states("test").unwrap();
            store.conn().execute(
            "UPDATE nodes SET properties=json_remove(properties, '$.has_bounds', '$.as_ref_receiver') WHERE label='Interface'",
            [],
        ).unwrap();
            store
                .conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
            store.conn().execute(
            "INSERT INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v6','complete')",
            [],
        ).unwrap();
            let root = greppy_discover::detect_repo_root(&repo).unwrap();
            let mut state = store
                .get_workspace_state(root.to_string_lossy().as_ref())
                .unwrap()
                .unwrap();
            state.indexer_version = prior_version.into();
            store.upsert_workspace_state(&state).unwrap();
            assert!(recover_persisted_rust_usages(&mut store, "test", &repo)
                .unwrap_err()
                .to_string()
                .contains("trait receiver facts"));

            fs::write(
                repo.join("src/changed.rs"),
                "pub fn changed() { let value = 1; }\n",
            )
            .unwrap();
            let options = IndexOptions {
                only_paths: Some(["src/changed.rs".to_string()].into_iter().collect()),
                ..IndexOptions::default()
            };
            let upgrade = index_with_options(&mut store, &repo, "test", &options).unwrap();
            assert_eq!(
                upgrade.files_indexed, 2,
                "unchanged source requires fresh declaration nodes"
            );
            let node = store
                .get_node_by_qname("test", "src/lib.rs::Interface::HttpTransport")
                .unwrap()
                .unwrap();
            assert_eq!(
                node.properties.get("has_bounds"),
                Some(&serde_json::json!(1))
            );
            assert_eq!(
                store
                    .list_file_states("test")
                    .unwrap()
                    .into_iter()
                    .find(|state| state.rel_path == "src/lib.rs")
                    .unwrap()
                    .sha256,
                source_before
                    .into_iter()
                    .find(|state| state.rel_path == "src/lib.rs")
                    .unwrap()
                    .sha256
            );
            assert!(rust_caller_edges_repaired(&store).unwrap());
            assert_eq!(index(&mut store, &repo, "test").unwrap().files_indexed, 0);
            fs::remove_dir_all(repo).unwrap();
        }
    }

    #[test]
    fn option_field_recovery_rejects_stale_declared_facts_without_certifying_base() {
        let repo = setup_repo("option-field-stale-facts", OPTION_FIELD_CALLER);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        store.conn().execute("UPDATE nodes SET properties=json_remove(properties,'$.return_type') WHERE label='Field'", []).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        let before = store.list_raw_edges("test").unwrap();
        let error = recover_persisted_rust_usages(&mut store, "test", &repo)
            .unwrap_err()
            .to_string();
        assert!(error.contains("declared field facts"), "{error}");
        assert_eq!(store.list_raw_edges("test").unwrap(), before);
        assert!(!rust_caller_edges_repaired(&store).unwrap());
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn option_field_caller_recovery_refreshes_old_raw_facts_without_base_ownership() {
        let repo = setup_repo("option-field-cache", OPTION_FIELD_CALLER);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        let base_path = repo.join("base.db");
        let delta_path = repo.join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, &repo, "test").unwrap();
            base.conn().execute("UPDATE raw_edges SET properties=json_remove(properties,'$.receiver_provenance') WHERE edge_type='CALLS'", []).unwrap();
            base.conn()
                .execute("DELETE FROM edges WHERE edge_type='CALLS'", [])
                .unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
            base.conn().execute("INSERT INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v6','complete')", []).unwrap();
            // The single-store upgrade also re-extracts, not merely re-resolves
            // the stale receiver facts, and does not accept a v6 completion.
            assert!(!rust_caller_edges_repaired(&base).unwrap());
            rebuild_single_store_rust_edges(&mut base, "test").unwrap();
            assert_option_field_caller(&base, "src/scene.rs::IrradianceField::uniform", true);
            base.conn().execute("UPDATE raw_edges SET properties=json_remove(properties,'$.receiver_provenance') WHERE edge_type='CALLS'", []).unwrap();
            base.conn()
                .execute("DELETE FROM edges WHERE edge_type='CALLS'", [])
                .unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
            let caller = base
                .get_node_by_qname("test", "src/lib.rs::Function::load_scene")
                .unwrap()
                .unwrap();
            let wrong = base
                .get_node_by_qname("test", "src/scene.rs::Other::uniform")
                .unwrap()
                .unwrap();
            base.insert_edge(&NewEdge {
                project: "test".into(),
                source_id: caller.id,
                target_id: wrong.id,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        let base_before = fs::read(&base_path).unwrap();
        let visibility =
            greppy_store::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        let original = fs::read(repo.join("src/scene.rs")).unwrap();
        fs::write(repo.join("src/scene.rs"), "// drift\n").unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).is_err());
        assert!(overlay.list_delta_raw_edges("test").unwrap().is_empty());
        fs::write(repo.join("src/scene.rs"), original).unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).unwrap() > 0);
        rebuild_visible_overlay_edges(&mut overlay, "test").unwrap();
        assert_option_field_caller(&overlay, "src/scene.rs::IrradianceField::uniform", true);
        assert_option_field_caller(&overlay, "src/scene.rs::Other::uniform", false);
        assert!(overlay.list_private_file_states("test").unwrap().is_empty());
        assert!(overlay.list_delta_raw_edges("test").unwrap().is_empty());
        drop(overlay);
        assert_eq!(fs::read(&base_path).unwrap(), base_before);
        let mut reopened = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        assert_option_field_caller(&reopened, "src/scene.rs::IrradianceField::uniform", true);
        assert_option_field_caller(&reopened, "src/scene.rs::Other::uniform", false);
        rebuild_overlay_edges(&mut reopened, "test").unwrap();
        assert_option_field_caller(&reopened, "src/scene.rs::IrradianceField::uniform", true);
        assert_option_field_caller(&reopened, "src/scene.rs::Other::uniform", false);
        assert!(recover_persisted_rust_usages(&mut reopened, "test", &repo).unwrap() == 0);
        drop(reopened);
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn receiver_method_name_index_preserves_owner_and_ambiguity() {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&greppy_store::Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let mut ids = Vec::new();
        for (file, owner, label) in [
            ("src/a.rs", "Buffer", "Method"),
            ("src/b.rs", "Buffer", "Method"),
            ("src/c.rs", "Other", "Method"),
            ("src/d.rs", "Buffer", "Function"),
        ] {
            ids.push(
                store
                    .insert_node(&NewNode {
                        project: "p".into(),
                        label: label.into(),
                        name: "as_bytes".into(),
                        qualified_name: format!("{file}::{owner}::as_bytes"),
                        file_path: file.into(),
                        start_line: 1,
                        end_line: 2,
                        properties: serde_json::json!({}),
                    })
                    .unwrap(),
            );
        }
        let index = GraphIndex::load(&store, "p").unwrap();
        assert_eq!(
            index.resolve_receiver_method("src/caller.rs", "Buffer", "as_bytes"),
            None
        );
        assert_eq!(
            index.resolve_receiver_method("src/a.rs", "Buffer", "as_bytes"),
            Some(ids[0])
        );
        assert_eq!(
            index.resolve_receiver_method("src/caller.rs", "Other", "as_bytes"),
            Some(ids[2])
        );
        assert_eq!(
            index.resolve_receiver_method("src/caller.rs", "Missing", "as_bytes"),
            None
        );
    }

    fn option_edge_names(store: &Store, caller: &str, edge_type: &str) -> Vec<String> {
        let caller = store.get_node_by_qname("test", caller).unwrap().unwrap();
        let mut names = store
            .outgoing_edges(caller.id, Some(edge_type), 100)
            .unwrap()
            .into_iter()
            .map(|edge| store.get_node(edge.target_id).unwrap().unwrap().name)
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    fn option_unresolved_reasons(store: &Store) -> String {
        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::load_scene")
            .unwrap()
            .unwrap();
        store
            .outgoing_edges(caller.id, Some("UNRESOLVED_CALLS"), 100)
            .unwrap()
            .into_iter()
            .map(|edge| {
                edge.properties
                    .get("unresolved_reasons")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null)
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn option_field_named_imports_resolve_without_treating_opaque_scopes_as_callers() {
        let proven = r#"
mod scene;
use crate::scene::Manifest;
use std::collections::BTreeMap;
#[allow(unused)]
pub fn load_scene() {
    println!("load");
    let manifest: Manifest = opaque();
    let _table: BTreeMap<String, u8> = BTreeMap::new();
    let gi_matrix: Option<[f32; 16]> = None;
    match (manifest.remaster_irradiance.as_ref(), gi_matrix) {
        (Some(field), Some(matrix)) => { field.storage(); field.uniform(matrix); }
        _ => (),
    }
}
"#;
        let repo = setup_repo("option-field-named-import", proven);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::storage", true);
        assert_option_field_caller(&store, "src/scene.rs::Other::uniform", false);
        assert!(option_edge_names(
            &store,
            "src/lib.rs::Function::load_scene",
            "UNRESOLVED_CALLS"
        )
        .is_empty());
        fs::remove_dir_all(repo).unwrap();

        let alias = proven
            .replace(
                "use crate::scene::Manifest;",
                "use crate::scene::Manifest as SceneManifest;",
            )
            .replace("let manifest: Manifest", "let manifest: SceneManifest");
        let repo = setup_repo("option-field-alias", &alias);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::storage", true);
        assert!(option_edge_names(
            &store,
            "src/lib.rs::Function::load_scene",
            "UNRESOLVED_CALLS"
        )
        .is_empty());
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn option_field_original_import_shape_is_unresolved_not_a_caller() {
        let caller = r#"
mod scene;
use crate::scene::{Manifest, VERTEX_STRIDE};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use wasm_bindgen::{JsCast, prelude::*};
use wgpu::util::DeviceExt;

#[wasm_bindgen]
pub async fn load_scene() {
    let _ = (VERTEX_STRIDE, BTreeMap::<String, u8>::new());
    let manifest: Manifest = opaque();
    let gi_matrix: Option<[f32; 16]> = None;
    match (manifest.remaster_irradiance.as_ref(), gi_matrix) {
        (Some(field), Some(matrix)) => { field.storage(); field.uniform(matrix); }
        _ => (),
    }
}
"#;
        let scene = OPTION_FIELD_SCENE.replace(
            "pub struct Manifest",
            "pub const VERTEX_STRIDE: usize = 32;\npub struct Manifest",
        );
        let repo = setup_repo("option-field-original-shape", caller);
        fs::write(repo.join("src/scene.rs"), scene).unwrap();
        let db = repo.join("graph.db");
        {
            let mut store = Store::open(&db).unwrap();
            index(&mut store, &repo, "test").unwrap();
            assert_eq!(
                option_edge_names(&store, "src/lib.rs::Function::load_scene", "CALLS")
                    .into_iter()
                    .filter(|name| name == "uniform" || name == "storage")
                    .count(),
                0
            );
            assert_eq!(
                option_edge_names(
                    &store,
                    "src/lib.rs::Function::load_scene",
                    "UNRESOLVED_CALLS"
                ),
                vec!["storage".to_string(), "uniform".to_string()]
            );
            let reasons = option_unresolved_reasons(&store);
            assert!(
                reasons.contains("wildcard import wasm_bindgen::prelude"),
                "{reasons}"
            );
            assert!(reasons.contains("attribute wasm_bindgen"), "{reasons}");
            assert!(
                reasons.contains("external import sha2::Digest"),
                "{reasons}"
            );
            assert!(
                reasons.contains("external import wgpu::util::DeviceExt"),
                "{reasons}"
            );
            assert_option_field_caller(&store, "src/scene.rs::Other::uniform", false);
        }
        let store = Store::open(&db).unwrap();
        assert_eq!(
            option_edge_names(
                &store,
                "src/lib.rs::Function::load_scene",
                "UNRESOLVED_CALLS"
            ),
            vec!["storage".to_string(), "uniform".to_string()]
        );
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", false);
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn option_field_adversarial_imports_do_not_invent_or_confirm_callers() {
        let base = r#"
mod scene;
mod adapter;
use crate::scene::Manifest;
pub fn load_scene() {
    let manifest: Manifest = opaque();
    match manifest.remaster_irradiance.as_ref() {
        Some(field) => field.uniform(),
        _ => (),
    }
}
"#;
        let cases = [
            (
                "consuming-import",
                "pub trait Consume { fn as_ref(self) -> Option<crate::scene::Other>; }\n",
                "use crate::adapter::Consume;\n",
                false,
            ),
            (
                "ref-trait-import",
                "pub trait View { fn as_ref(&self); }\n",
                "use crate::adapter::View;\n",
                true,
            ),
            (
                "open-trait-import",
                "pub trait Open: core::fmt::Debug {}\n",
                "use crate::adapter::Open;\n",
                false,
            ),
            ("generic-payload", "", "", false),
        ];
        for (label, adapter, import, proven) in cases {
            let mut caller = base.replace("mod adapter;\n", &format!("mod adapter;\n{import}"));
            let mut scene = OPTION_FIELD_SCENE.to_string();
            if label == "generic-payload" {
                caller = OPTION_FIELD_CALLER.to_string();
                scene = OPTION_FIELD_SCENE.replace(
                    "pub struct Manifest { pub remaster_irradiance: Option<crate::scene::IrradianceField> }",
                    "pub struct Manifest<T> { pub remaster_irradiance: Option<T> }",
                );
            }
            let repo = setup_repo(&format!("option-field-{label}"), &caller);
            fs::write(repo.join("src/scene.rs"), scene).unwrap();
            fs::write(repo.join("src/adapter.rs"), adapter).unwrap();
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            assert_eq!(
                option_edge_names(&store, "src/lib.rs::Function::load_scene", "CALLS")
                    .contains(&"uniform".to_string()),
                proven,
                "{label}"
            );
            let unresolved = option_edge_names(
                &store,
                "src/lib.rs::Function::load_scene",
                "UNRESOLVED_CALLS",
            )
            .contains(&"uniform".to_string());
            if label == "open-trait-import" {
                assert!(unresolved, "{label}");
                assert!(
                    option_unresolved_reasons(&store).contains("trait import Open has supertraits"),
                    "{}",
                    option_unresolved_reasons(&store)
                );
            } else {
                assert!(!unresolved, "{label}");
            }
            fs::remove_dir_all(repo).unwrap();
        }

        let nested = r#"
mod scene;
mod nested { use custom::*; }
use crate::scene::Manifest;
pub fn load_scene() {
    let manifest: Manifest = opaque();
    match manifest.remaster_irradiance.as_ref() {
        Some(field) => field.uniform(),
        _ => (),
    }
}
"#;
        let repo = setup_repo("option-field-nested-glob", nested);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert!(
            !option_edge_names(&store, "src/lib.rs::Function::load_scene", "CALLS")
                .contains(&"uniform".to_string())
        );
        assert!(option_edge_names(
            &store,
            "src/lib.rs::Function::load_scene",
            "UNRESOLVED_CALLS"
        )
        .contains(&"uniform".to_string()));
        assert!(
            option_unresolved_reasons(&store).contains("wildcard import custom"),
            "{}",
            option_unresolved_reasons(&store)
        );
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn option_unresolved_reasons_do_not_attach_to_a_different_edge() {
        let option_edge = greppy_parser::ExtractedEdge {
            edge_type: "CALLS".into(),
            source_qualified_name: "src/lib.rs::Function::load_scene".into(),
            target_qualified_name: "src/lib.rs::Function::uniform".into(),
            file_path: "src/lib.rs".into(),
            line: 10,
            properties: serde_json::json!({"callee_name": "uniform"}),
        };
        let other = greppy_parser::ExtractedEdge {
            edge_type: "CALLS".into(),
            source_qualified_name: "src/lib.rs::Function::load_scene".into(),
            target_qualified_name: "src/lib.rs::Function::helper".into(),
            file_path: "src/lib.rs".into(),
            line: 11,
            properties: serde_json::json!({"callee_name": "helper"}),
        };
        set_option_field_unresolved(&option_edge, vec!["wildcard import custom".into()]);
        let leaked = new_edge("p", 1, 2, &other);
        assert_eq!(leaked.edge_type, "CALLS");
        assert!(leaked.properties.get("unresolved_reasons").is_none());
        let dropped = new_edge("p", 1, 3, &option_edge);
        assert_eq!(dropped.edge_type, "CALLS");
        set_option_field_unresolved(&option_edge, vec!["wildcard import custom".into()]);
        let matched = new_edge("p", 1, 3, &option_edge);
        assert_eq!(matched.edge_type, "UNRESOLVED_CALLS");
        assert!(matched.properties["unresolved_reasons"]
            .to_string()
            .contains("wildcard import custom"));
        clear_option_field_unresolved();
        let cleared = new_edge("p", 1, 3, &option_edge);
        assert_eq!(cleared.edge_type, "CALLS");
        let import = greppy_parser::ExtractedEdge {
            edge_type: "IMPORTS".into(),
            source_qualified_name: "src/lib.rs::__file__".into(),
            target_qualified_name: "src/lib.rs::Function::Manifest".into(),
            file_path: "src/lib.rs".into(),
            line: 1,
            properties: serde_json::json!({}),
        };
        set_option_field_unresolved(&option_edge, vec!["wildcard import custom".into()]);
        let imported = new_edge("p", 1, 4, &import);
        assert_eq!(imported.edge_type, "IMPORTS");
        let after_import = new_edge("p", 1, 3, &option_edge);
        assert_eq!(after_import.edge_type, "CALLS");
    }

    #[test]
    fn option_field_name_resemblance_and_shadowed_std_stay_unresolved() {
        let proven = r#"
mod scene;
use crate::scene::Manifest;
use std::collections::BTreeMap;
#[allow(unused)]
pub fn load_scene() {
    println!("load");
    let manifest: Manifest = opaque();
    let _table: BTreeMap<String, u8> = BTreeMap::new();
    match manifest.remaster_irradiance.as_ref() {
        Some(field) => field.uniform(),
        _ => (),
    }
}
"#;
        let cases = [
            (
                "qualified-macro",
                proven.replace("#[allow(unused)]", "helper::assert!();\n#[custom::allow]"),
                "pub struct LocalStd;\n",
                false,
            ),
            (
                "user-println",
                proven.replace(
                    "pub fn load_scene()",
                    "macro_rules! println { () => {}; }\npub fn load_scene()",
                ),
                "pub struct LocalStd;\n",
                false,
            ),
            (
                "shadowed-std",
                "mod std;\n".to_string() + proven,
                "pub struct LocalStd;\n",
                false,
            ),
        ];
        for (label, caller, std_src, _proven) in cases {
            let repo = setup_repo(&format!("option-field-{label}"), &caller);
            fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
            if label == "shadowed-std" {
                fs::write(repo.join("src/std.rs"), std_src).unwrap();
            }
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            assert!(
                !option_edge_names(&store, "src/lib.rs::Function::load_scene", "CALLS")
                    .contains(&"uniform".to_string()),
                "{label}",
            );
            assert!(
                option_edge_names(
                    &store,
                    "src/lib.rs::Function::load_scene",
                    "UNRESOLVED_CALLS"
                )
                .contains(&"uniform".to_string()),
                "{label}",
            );
            let reasons = option_unresolved_reasons(&store);
            match label {
                "qualified-macro" => {
                    assert!(reasons.contains("macro helper::assert"), "{reasons}");
                    assert!(reasons.contains("attribute custom::allow"), "{reasons}");
                }
                "user-println" => assert!(reasons.contains("macro println"), "{reasons}"),
                "shadowed-std" => assert!(
                    reasons.contains("std::collections::BTreeMap") || reasons.contains("import"),
                    "{reasons}",
                ),
                _ => {}
            }
            fs::remove_dir_all(repo).unwrap();
        }
    }

    #[test]
    fn option_field_missing_limits_and_trait_facts_do_not_keep_stale_callers() {
        let caller = r#"
mod scene;
use crate::scene::Manifest;
use std::collections::BTreeMap;
#[allow(unused)]
pub fn load_scene() {
    println!("load");
    let manifest: Manifest = opaque();
    match manifest.remaster_irradiance.as_ref() {
        Some(field) => field.uniform(),
        _ => (),
    }
}
"#;
        let repo = setup_repo("option-field-missing-limits", caller);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        store.conn().execute(
            "UPDATE raw_edges SET properties=json_remove(properties, '$.receiver_provenance.limits') WHERE edge_type='CALLS'",
            [],
        ).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE edge_type IN ('CALLS', 'UNRESOLVED_CALLS')",
                [],
            )
            .unwrap();
        let raw = load_all_raw_edges(&store, "test").unwrap();
        resolve_edges_with_replacement(&mut store, "test", &raw, &mut |_| {}, &[], true).unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", false);
        assert!(!option_edge_names(
            &store,
            "src/lib.rs::Function::load_scene",
            "UNRESOLVED_CALLS"
        )
        .contains(&"uniform".to_string()));
        store.conn().execute(
            "UPDATE nodes SET properties=json_set(properties, '$.generic_payload', 1) WHERE label='Field' AND name='remaster_irradiance'",
            [],
        ).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        let error = recover_persisted_rust_usages(&mut store, "test", &repo)
            .unwrap_err()
            .to_string();
        assert!(error.contains("declared field facts"), "{error}");
        fs::remove_dir_all(repo).unwrap();

        let trait_caller = r#"
mod scene;
mod adapter;
use crate::scene::Manifest;
use crate::adapter::View;
pub fn load_scene() {
    let manifest: Manifest = opaque();
    match manifest.remaster_irradiance.as_ref() {
        Some(field) => field.uniform(),
        _ => (),
    }
}
"#;
        let repo = setup_repo("option-field-trait-facts", trait_caller);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        fs::write(
            repo.join("src/adapter.rs"),
            "pub trait View { fn as_ref(&self); }\n",
        )
        .unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        fs::write(
            repo.join("src/adapter.rs"),
            "pub trait View: core::fmt::Debug { fn as_ref(&self); }\n",
        )
        .unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", false);
        assert!(
            option_unresolved_reasons(&store).contains("trait import View has supertraits"),
            "{}",
            option_unresolved_reasons(&store)
        );
        fs::write(
            repo.join("src/adapter.rs"),
            "pub trait View { fn as_ref(self); }\n",
        )
        .unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", false);
        assert!(!option_edge_names(
            &store,
            "src/lib.rs::Function::load_scene",
            "UNRESOLVED_CALLS"
        )
        .contains(&"uniform".to_string()));
        fs::write(
            repo.join("src/adapter.rs"),
            "pub trait View { fn as_ref(&self); }\n",
        )
        .unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        store.conn().execute(
            "UPDATE nodes SET properties=json_set(properties, '$.has_bounds', 1) WHERE label='Interface' AND name='View'",
            [],
        ).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        let error = recover_persisted_rust_usages(&mut store, "test", &repo)
            .unwrap_err()
            .to_string();
        assert!(error.contains("trait receiver facts"), "{error}");
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        fs::remove_dir_all(repo).unwrap();

        let repo = setup_repo("option-field-generic-edit", trait_caller);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE).unwrap();
        fs::write(
            repo.join("src/adapter.rs"),
            "pub trait View { fn as_ref(&self); }\n",
        )
        .unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", true);
        fs::write(repo.join("src/scene.rs"), OPTION_FIELD_SCENE.replace(
            "pub struct Manifest { pub remaster_irradiance: Option<crate::scene::IrradianceField> }",
            "pub struct Manifest<T> { pub remaster_irradiance: Option<T> }",
        )).unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_option_field_caller(&store, "src/scene.rs::IrradianceField::uniform", false);
        assert!(!option_edge_names(
            &store,
            "src/lib.rs::Function::load_scene",
            "UNRESOLVED_CALLS"
        )
        .contains(&"uniform".to_string()));
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn callback_v10_store_recovers_omitted_usages_without_shadow_callers() {
        let repo = setup_repo(
            "callback-v10-repair",
            r#"
pub fn predicate(value: i32) -> bool { value > 0 }
pub fn chained(value: Option<i32>) -> bool {
    value.map(predicate).unwrap_or(false)
}
pub fn shadowed(value: Option<i32>, predicate: fn(i32) -> bool) -> bool {
    value.map(predicate).unwrap_or(false)
}
"#,
        );
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let predicate = store
            .get_node_by_qname("test", "src/lib.rs::Function::predicate")
            .unwrap()
            .unwrap();
        let chained = store
            .get_node_by_qname("test", "src/lib.rs::Function::chained")
            .unwrap()
            .unwrap();
        let shadowed = store
            .get_node_by_qname("test", "src/lib.rs::Function::shadowed")
            .unwrap()
            .unwrap();
        store
            .conn()
            .execute("DELETE FROM edges WHERE edge_type='USAGE'", [])
            .unwrap();
        store
            .conn()
            .execute("DELETE FROM raw_edges WHERE edge_type='USAGE'", [])
            .unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        store.conn().execute("INSERT INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v10','complete')", []).unwrap();
        assert!(!rust_caller_edges_repaired(&store).unwrap());
        assert!(store
            .incoming_edges(predicate.id, Some("USAGE"), 20)
            .unwrap()
            .is_empty());
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(rust_caller_edges_repaired(&store).unwrap());
        let callers = store
            .incoming_edges(predicate.id, Some("USAGE"), 20)
            .unwrap();
        assert_eq!(callers.len(), 1, "{callers:?}");
        assert_eq!(callers[0].source_id, chained.id);
        assert!(store
            .outgoing_edges(shadowed.id, Some("USAGE"), 20)
            .unwrap()
            .is_empty());
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn rust_anyhow_factory_dependency_identity_and_completed_cache_replay() {
        let source = "#[path=\"projection.rs\"] mod projection; use projection::*; use std::path::Path; struct Writer; impl Writer { fn open() -> anyhow::Result<Option<Self>> { loop {} } fn upsert(&mut self) {} } fn caller() { if let Some(mut writer) = Writer::open()? { writer.upsert(); } }";
        let repo = setup_repo("anyhow-factory-repair", source);
        fs::write(repo.join("src/projection.rs"), "pub fn project() {}\n").unwrap();
        fs::write(repo.join("Cargo.toml"), "[package]\nname='factory_fixture'\nversion='0.1.0'\nedition='2021'\n[dependencies]\nanyhow='1'\n[target.'cfg(windows)'.dependencies]\nother='1'\n[patch.crates-io]\nother={path='other'}\n").unwrap();
        let lock = format!(
            "version = 3\n[[package]]\nname='anyhow'\nversion='1.0.102'\nsource='registry+https://github.com/rust-lang/crates.io-index'\nchecksum='{}'\n",
            "a".repeat(64)
        );
        fs::write(repo.join("Cargo.lock"), &lock).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let method = store
            .get_node_by_qname("test", "src/lib.rs::Writer::upsert")
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1
        );
        let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
        let states = format!("{:?}", store.list_file_states("test").unwrap());
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE edge_type='CALLS' AND target_id=?1",
                [method.id],
            )
            .unwrap();
        store.conn().execute("UPDATE raw_edges SET properties=json_remove(properties,'$.receiver_anyhow_factory_owner') WHERE json_extract(properties,'$.callee_name')='upsert'", []).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [anyhow_factory_repair_key("test")],
            )
            .unwrap();
        assert!(
            !anyhow_factory_edges_repaired(&store).unwrap(),
            "completed v13 alone cannot certify new external-wrapper provenance"
        );
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(anyhow_factory_edges_repaired(&store).unwrap());
        assert_eq!(
            store
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            nodes,
            format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        assert_eq!(
            states,
            format!("{:?}", store.list_file_states("test").unwrap())
        );
        for exports in [
            "pub use crate::hidden::Option;\n",
            "pub struct Some;\n",
            "generate_exports!();\n",
        ] {
            fs::write(repo.join("src/projection.rs"), exports).unwrap();
            index(&mut store, &repo, "test").unwrap();
            assert!(
                store
                    .incoming_edges(method.id, Some("CALLS"), 20)
                    .unwrap()
                    .is_empty(),
                "unproven or shadowing glob exports: {exports}"
            );
        }
        fs::write(repo.join("src/projection.rs"), "pub fn project() {}\n").unwrap();
        index(&mut store, &repo, "test").unwrap();
        assert_eq!(
            store
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1
        );
        fs::write(
            repo.join("Cargo.lock"),
            format!("{lock}{}", lock.strip_prefix("version = 3\n").unwrap()),
        )
        .unwrap();
        assert!(!anyhow_factory_edges_repaired(&store).unwrap());
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(
            store
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .is_empty(),
            "ambiguous dependency identity must remove inferred caller"
        );
        fs::write(repo.join("Cargo.lock"), &lock).unwrap();
        for dependency in [
            "anyhow={path='local'}",
            "anyhow={version='1',package='different'}",
            "anyhow='2'",
            "alias={version='1',package='anyhow'}",
        ] {
            fs::write(repo.join("Cargo.toml"), format!("[package]\nname='factory_fixture'\nversion='0.1.0'\n[dependencies]\n{dependency}\n")).unwrap();
            rebuild_single_store_rust_edges(&mut store, "test").unwrap();
            assert!(
                store
                    .incoming_edges(method.id, Some("CALLS"), 20)
                    .unwrap()
                    .is_empty(),
                "{dependency}"
            );
        }
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn rust_enum_variant_span_repair_preserves_private_and_base_identity() {
        for overlay_mode in [false, true] {
            let source =
                "pub enum Kind {\n    Branch {\n        condition: bool,\n    },\n    Tail,\n}\n";
            let repo = setup_repo("variant-span-repair", source);
            let scratch = tempfile::tempdir().unwrap();
            let base_path = scratch.path().join("base.db");
            {
                let mut base = Store::open(&base_path).unwrap();
                let report = index(&mut base, &repo, "test").unwrap();
                base.conn()
                    .execute(
                        "UPDATE nodes SET end_line=start_line WHERE label='EnumVariant'",
                        [],
                    )
                    .unwrap();
                let node = base
                    .get_node_by_qname("test", "src/lib.rs::Kind::Branch")
                    .unwrap()
                    .unwrap();
                base.upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
                    project: "test".into(),
                    model_id: "test-model".into(),
                    prompt_version: "v1".into(),
                    task: "definition".into(),
                    node_id: Some(node.id),
                    chunk_idx: 0,
                    qualified_name: node.qualified_name,
                    file_path: node.file_path,
                    start_line: node.start_line,
                    end_line: node.end_line,
                    content_sha256: file_state::sha256_hex(source.as_bytes()),
                    graph_generation: report.graph_generation,
                    vector: vec![1.0, 0.0],
                })
                .unwrap();
            }
            let original_base = fs::read(&base_path).unwrap();
            let delta_path = scratch.path().join("delta.db");
            let mut store = if overlay_mode {
                Store::open_overlay(
                    &base_path,
                    &delta_path,
                    &greppy_store::VisibilityIndex::default(),
                )
                .unwrap()
            } else {
                Store::open(&base_path).unwrap()
            };
            let before = store
                .get_node_by_qname("test", "src/lib.rs::Kind::Branch")
                .unwrap()
                .unwrap();
            let states = store.list_file_states("test").unwrap();
            let edges = format!("{:?}", store.outgoing_edges(before.id, None, 100).unwrap());
            let vectors: i64 = store
                .conn()
                .query_row("SELECT COUNT(*) FROM vector_embeddings", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(
                recover_persisted_rust_enum_variant_spans(&mut store, "test", &repo).unwrap(),
                1
            );
            let after = store
                .get_node_by_qname("test", "src/lib.rs::Kind::Branch")
                .unwrap()
                .unwrap();
            assert_eq!(after.id, before.id);
            assert_eq!((after.start_line, after.end_line), (2, 4));
            assert_eq!(after.properties, before.properties);
            assert_eq!(store.list_file_states("test").unwrap(), states);
            assert_eq!(
                format!("{:?}", store.outgoing_edges(after.id, None, 100).unwrap()),
                edges
            );
            assert_eq!(
                store
                    .conn()
                    .query_row("SELECT COUNT(*) FROM vector_embeddings", [], |row| row
                        .get::<_, i64>(0))
                    .unwrap(),
                vectors
            );
            assert_eq!(
                recover_persisted_rust_enum_variant_spans(&mut store, "test", &repo).unwrap(),
                0
            );
            if overlay_mode {
                assert_eq!(fs::read(&base_path).unwrap(), original_base);
                assert_eq!(
                    store
                        .conn()
                        .query_row("SELECT COUNT(*) FROM main.nodes", [], |row| row
                            .get::<_, i64>(0))
                        .unwrap(),
                    0
                );
                drop(store);
                let reopened = Store::open_overlay_read_only(
                    &base_path,
                    &delta_path,
                    &greppy_store::VisibilityIndex::default(),
                )
                .unwrap();
                assert_eq!(reopened.get_node(before.id).unwrap().unwrap().end_line, 4);
                drop(reopened);
                let hidden = Store::open_overlay_read_only(
                    &base_path,
                    &delta_path,
                    &greppy_store::VisibilityIndex::new(
                        vec!["src/lib.rs".into()],
                        Vec::<String>::new(),
                    )
                    .unwrap(),
                )
                .unwrap();
                assert!(hidden.get_node(before.id).unwrap().is_none());
            }
            fs::remove_dir_all(repo).unwrap();
        }
    }

    #[test]
    fn rust_enum_variant_span_repair_refuses_stale_or_corrupt_source_atomically() {
        for changed_source in [false, true] {
            let repo = setup_repo(
                "variant-span-refusal",
                "pub enum Kind {\n    Branch {\n        condition: bool,\n    },\n}\n",
            );
            fs::write(
                repo.join("src/second.rs"),
                "pub enum Second {\n    Other {\n        value: u32,\n    },\n}\n",
            )
            .unwrap();
            let mut store = Store::open_memory().unwrap();
            index(&mut store, &repo, "test").unwrap();
            store
                .conn()
                .execute(
                    "UPDATE nodes SET end_line=start_line WHERE label='EnumVariant'",
                    [],
                )
                .unwrap();
            if changed_source {
                fs::write(repo.join("src/second.rs"), "pub enum Second { Other }\n").unwrap();
            } else {
                store
                    .conn()
                    .execute(
                        "UPDATE nodes SET end_line=999 WHERE name='Other' AND label='EnumVariant'",
                        [],
                    )
                    .unwrap();
            }
            let before = format!(
                "{:?}",
                store
                    .list_nodes_by_label("test", "EnumVariant", 100)
                    .unwrap()
            );
            assert!(recover_persisted_rust_enum_variant_spans(&mut store, "test", &repo).is_err());
            assert_eq!(
                format!(
                    "{:?}",
                    store
                        .list_nodes_by_label("test", "EnumVariant", 100)
                        .unwrap()
                ),
                before
            );
            fs::remove_dir_all(repo).unwrap();
        }
    }

    #[test]
    fn rust_anyhow_factory_overlay_replay_preserves_base_and_refuses_changed_source() {
        let source = "#[path=\"projection.rs\"] mod projection; use projection::*; use std::path::Path; struct Writer; impl Writer { fn open() -> anyhow::Result<Option<Self>> { loop {} } fn upsert(&mut self) {} } fn caller() { if let Some(mut writer) = Writer::open()? { writer.upsert(); } }";
        let repo = setup_repo("anyhow-overlay-repair", source);
        fs::write(repo.join("src/projection.rs"), "pub fn project() {}\n").unwrap();
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname='factory_fixture'\nversion='0.1.0'\n[dependencies]\nanyhow='1'\n",
        )
        .unwrap();
        fs::write(repo.join("Cargo.lock"), format!("version=3\n[[package]]\nname='anyhow'\nversion='1.0.102'\nsource='registry+https://github.com/rust-lang/crates.io-index'\nchecksum='{}'\n", "a".repeat(64))).unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            index(&mut base, &repo, "test").unwrap();
            base.conn().execute("DELETE FROM edges WHERE edge_type='CALLS' AND target_id IN (SELECT id FROM nodes WHERE name='upsert')", []).unwrap();
            base.conn().execute("UPDATE raw_edges SET properties=json_remove(properties,'$.receiver_anyhow_factory_owner') WHERE json_extract(properties,'$.callee_name')='upsert'", []).unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [anyhow_factory_repair_key("test")],
                )
                .unwrap();
        }
        let bytes = fs::read(&base_path).unwrap();
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        mark_rust_caller_edges_repaired(&overlay).unwrap();
        assert!(!anyhow_factory_edges_repaired(&overlay).unwrap());
        let nodes = format!("{:?}", overlay.list_nodes("test", "", "", 0, 1000).unwrap());
        let states = format!("{:?}", overlay.list_file_states("test").unwrap());
        let workspace = format!(
            "{:?}",
            overlay.get_workspace_state(repo.to_str().unwrap()).unwrap()
        );
        recover_persisted_rust_usages(&mut overlay, "test", &repo).unwrap();
        rebuild_visible_overlay_edges(&mut overlay, "test").unwrap();
        mark_rust_caller_edges_repaired(&overlay).unwrap();
        assert!(anyhow_factory_edges_repaired(&overlay).unwrap());
        let method = overlay
            .get_node_by_qname("test", "src/lib.rs::Writer::upsert")
            .unwrap()
            .unwrap();
        assert_eq!(
            overlay
                .incoming_edges(method.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            nodes,
            format!("{:?}", overlay.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        assert_eq!(
            states,
            format!("{:?}", overlay.list_file_states("test").unwrap())
        );
        assert_eq!(
            workspace,
            format!(
                "{:?}",
                overlay.get_workspace_state(repo.to_str().unwrap()).unwrap()
            )
        );
        assert_eq!(bytes, fs::read(&base_path).unwrap());
        fs::write(
            repo.join("src/lib.rs"),
            source.replace("Writer::open()?", "unknown()?"),
        )
        .unwrap();
        overlay
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [anyhow_factory_repair_key("test")],
            )
            .unwrap();
        assert!(recover_persisted_rust_usages(&mut overlay, "test", &repo).is_err());
        assert!(!anyhow_factory_edges_repaired(&overlay).unwrap());
        assert_eq!(bytes, fs::read(&base_path).unwrap());
        drop(overlay);
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn rust_factory_option_call_repair_rejects_v11_and_recovers_same_source() {
        let repo = setup_repo(
            "factory-option-repair",
            "struct Writer; impl Writer { fn open() -> Result<Option<Self>, ()> { loop {} } fn upsert(&mut self) {} } fn caller() { if let Some(mut writer) = Writer::open()? { writer.upsert(); } }",
        );
        let mut store = Store::open_memory().unwrap();
        index(&mut store, &repo, "test").unwrap();
        let method = store
            .get_node_by_qname("test", "src/lib.rs::Writer::upsert")
            .unwrap()
            .unwrap();
        let caller = store
            .get_node_by_qname("test", "src/lib.rs::Function::caller")
            .unwrap()
            .unwrap();
        assert!(store
            .incoming_edges(method.id, Some("CALLS"), 20)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id));
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE edge_type='CALLS' AND target_id=?1",
                [method.id],
            )
            .unwrap();
        store.conn().execute("UPDATE raw_edges SET properties=json_remove(properties, '$.receiver_owner') WHERE edge_type='CALLS' AND json_extract(properties,'$.callee_name')='upsert'", []).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        store.conn().execute("INSERT INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v11','complete')", []).unwrap();
        assert!(!rust_caller_edges_repaired(&store).unwrap());
        rebuild_single_store_rust_edges(&mut store, "test").unwrap();
        assert!(rust_caller_edges_repaired(&store).unwrap());
        assert!(store
            .incoming_edges(method.id, Some("CALLS"), 20)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id));
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn rust_caller_repair_v7_marker_is_not_current() {
        let store = Store::open_memory().unwrap();
        store.conn().execute(
            "INSERT INTO schema_meta(key, value) VALUES('greppy.rust_caller_edges_repair.v7', 'complete')",
            [],
        ).unwrap();
        store.conn().execute(
            "INSERT INTO schema_meta(key, value) VALUES('greppy.rust_caller_edges_repair.v8', 'complete')",
            [],
        ).unwrap();
        assert!(!rust_caller_edges_repaired(&store).unwrap());
    }

    #[test]
    fn js_ts_usage_recovery_accepts_validated_degraded_records() {
        for (language, file) in [
            (Language::JavaScript, "fixture.js"),
            (Language::TypeScript { tsx: false }, "fixture.ts"),
        ] {
            let source =
                b"function target() { return 1; }\nfunction valid() { return target(); }\n";
            let original = parser_extract(language, source, file).unwrap();
            let mut poisoned = original.clone();
            poisoned.nodes.push(ExtractedNode {
                label: "Function".into(),
                name: String::new(),
                qualified_name: String::new(),
                file_path: file.into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::Value::Null,
            });
            let validated = validated_js_ts_repair_extraction(language, file, poisoned).unwrap();
            assert_eq!(
                format!("{:?}", validated.nodes),
                format!("{:?}", original.nodes)
            );
            assert_eq!(
                format!("{:?}", validated.edges),
                format!("{:?}", original.edges)
            );
            assert!(validated.edges.iter().any(|e| e.edge_type == "CALLS"));
        }
    }

    #[test]
    fn js_ts_usage_recovery_indexes_malformed_fixture_without_losing_valid_calls() {
        let repo = tempfile::tempdir().unwrap();
        let file = "fixture.ts";
        let language = Language::TypeScript { tsx: false };
        let source = "class C { () {} }\nfunction target() { return 1; }\nfunction valid() { return target(); }\n";
        let extraction = parser_extract(language, source.as_bytes(), file).unwrap();
        let (_, dropped, error) = validate_or_degrade(language, file, extraction);
        assert!(
            dropped > 0,
            "the malformed method must exercise grammar recovery"
        );
        assert!(error.is_none());
        eprintln!("malformed_fixture_source={source:?}");
        fs::write(repo.path().join(file), source).unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        assert!(js_ts_usages_repaired(&store).unwrap());
        let target = store
            .get_node_by_qname("test", "fixture.ts::Function::target")
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .incoming_edges(target.id, Some("CALLS"), 10)
                .unwrap()
                .len(),
            1
        );
        let nodes = format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap());
        store.conn().execute_batch("DELETE FROM main.raw_edges WHERE edge_type IN ('USAGE','CALLS'); DELETE FROM main.edges WHERE edge_type IN ('USAGE','CALLS');").unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_KEY],
            )
            .unwrap();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());
        assert_eq!(
            store
                .incoming_edges(target.id, Some("CALLS"), 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            nodes,
            format!("{:?}", store.list_nodes("test", "", "", 0, 1000).unwrap())
        );
        assert!(!recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());
    }

    #[test]
    fn js_ts_usage_repair_skip_json_caps_paths_and_keeps_full_count() {
        let paths: Vec<String> = (0..25).map(|i| format!("f{i:02}.js")).collect();
        let value: serde_json::Value =
            serde_json::from_str(&js_ts_usage_repair_skip_json(&paths)).unwrap();
        assert_eq!(value["count"], 25);
        assert_eq!(value["paths"].as_array().unwrap().len(), 20);
        assert_eq!(value["paths"][0], "f00.js");
        assert_eq!(value["paths"][19], "f19.js");
    }

    /// One residual contract failure is skipped. Other files are still repaired,
    /// and edges already stored for the skipped file are left untouched.
    ///
    /// `contract-invalid.js` is the stand-in fixture (see
    /// `tests/fixtures/contract-invalid.js`). The current provider filter cures
    /// every real JS/TS extract, so the test arms the residual-contract path
    /// for that filename only.
    #[test]
    fn js_ts_usage_repair_skips_contract_invalid_file_and_repairs_the_rest() {
        let repo = tempfile::tempdir().unwrap();
        let good = "function target() { return 1; }\nfunction valid() { return target(); }\n";
        fs::write(repo.path().join("good.js"), good).unwrap();
        fs::write(
            repo.path().join("contract-invalid.js"),
            include_str!("../tests/fixtures/contract-invalid.js"),
        )
        .unwrap();
        let mut store = Store::open_memory().unwrap();
        index(&mut store, repo.path(), "test").unwrap();
        assert!(js_ts_usages_repaired(&store).unwrap());
        store
            .conn()
            .execute_batch(
                "DELETE FROM main.raw_edges WHERE edge_type IN ('USAGE','CALLS'); \
                 DELETE FROM main.edges WHERE edge_type IN ('USAGE','CALLS');",
            )
            .unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM main.schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_KEY],
            )
            .unwrap();
        let skipped = store
            .get_node_by_qname("test", "contract-invalid.js::Function::skipped")
            .unwrap()
            .unwrap();
        store
            .insert_raw_edges(&[NewRawEdge {
                project: "test".into(),
                file_path: "contract-invalid.js".into(),
                source_qname: skipped.qualified_name.clone(),
                target_qname: "preserved-skip-edge".into(),
                edge_type: "USAGE".into(),
                properties: serde_json::json!({"marker": "preserve"}),
            }])
            .unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO main.edges(project,source_id,target_id,edge_type,properties) \
                 VALUES('test',?1,?1,'USAGE','{\"marker\":\"preserve\"}')",
                [skipped.id],
            )
            .unwrap();
        let _guard = ForceJsContractSkip::arm();
        assert!(recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());
        assert!(js_ts_usages_repaired(&store).unwrap());
        let target = store
            .get_node_by_qname("test", "good.js::Function::target")
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .incoming_edges(target.id, Some("CALLS"), 10)
                .unwrap()
                .len(),
            1,
            "the other file's usages are repaired"
        );
        let preserved_raw: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM main.raw_edges WHERE project='test' \
                 AND file_path='contract-invalid.js' AND target_qname='preserved-skip-edge'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(preserved_raw, 1, "skipped file raw edges stay");
        let preserved_edge: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM main.edges WHERE source_id=?1 AND properties LIKE '%preserve%'",
                [skipped.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(preserved_edge, 1, "skipped file resolved edges stay");
        let raw: String = store
            .conn()
            .query_row(
                "SELECT value FROM main.schema_meta WHERE key=?1",
                [JS_TS_USAGE_REPAIR_SKIPS_KEY],
                |row| row.get(0),
            )
            .unwrap();
        let diagnostic: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(diagnostic["count"], 1);
        assert_eq!(
            diagnostic["paths"],
            serde_json::json!(["contract-invalid.js"])
        );
        assert!(!recover_persisted_js_ts_usages(&mut store, "test", repo.path()).unwrap());
    }

    struct ForceJsContractSkip;

    impl ForceJsContractSkip {
        fn arm() -> Self {
            FORCE_JS_TS_CONTRACT_SKIP.with(|flag| flag.set(true));
            Self
        }
    }

    impl Drop for ForceJsContractSkip {
        fn drop(&mut self) {
            FORCE_JS_TS_CONTRACT_SKIP.with(|flag| flag.set(false));
        }
    }

    /// ClickHouse regression: one anonymous node from grammar error-recovery
    /// must degrade to a dropped record, never to a dark file.
    #[test]
    fn invalid_record_is_dropped_not_the_file() {
        let src = "int real_fn(void) { return 1; }\n";
        let extraction =
            greppy_parser::extract(Language::C, src.as_bytes(), "a.c").expect("extract");
        let mut poisoned = extraction;
        poisoned.nodes.push(ExtractedNode {
            label: "Function".into(),
            name: String::new(),
            qualified_name: String::new(),
            file_path: "a.c".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::Value::Null,
        });
        let (filtered, dropped, error) = validate_or_degrade(Language::C, "a.c", poisoned);
        assert_eq!(dropped, 1, "exactly the anonymous node goes");
        assert!(error.is_none(), "the cured file is valid: {error:?}");
        assert!(
            filtered.nodes.iter().any(|n| n.name == "real_fn"),
            "the real definition survives the cure"
        );
    }

    /// ClickHouse regression: C++ syntax under a `.h` name is parsed with the
    /// C grammar; wide_integer_impl.h yielded one anonymous function node and
    /// the file went dark. The cure must not depend on the C++ retry: even
    /// under C, the anonymous node is dropped and the file stays indexable.
    #[test]
    fn cpp_syntax_in_dot_h_degrades_under_c() {
        let src =
            "constexpr const auto & toBitInt256(const wide::integer<Bits, Signed> & n)\n{\n}\n";
        let extraction =
            greppy_parser::extract(Language::C, src.as_bytes(), "a.h").expect("extract");
        assert!(
            extraction.nodes.iter().any(|n| n.name.trim().is_empty()),
            "precondition: the C grammar still emits the anonymous node; if this \
             starts failing the grammar improved and this pin can move to Cpp"
        );
        let (_, dropped, error) = validate_or_degrade(Language::C, "a.h", extraction);
        assert!(dropped >= 1, "the anonymous node is dropped");
        assert!(error.is_none(), "the file is not refused: {error:?}");
    }
}
