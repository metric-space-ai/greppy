//! Index freshness: the gates that refuse to answer from a stale graph.
//!
//! Split out of `lib.rs`; `use super::*` keeps every private helper there
//! reachable, and no behaviour changes.

use super::*;

pub(crate) fn provider_policy_from_env() -> Result<ProviderPolicy> {
    let raw = match std::env::var(ENV_PROVIDER_POLICY) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(ProviderPolicy::Metadata),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::Config(format!(
                "{ENV_PROVIDER_POLICY} must be valid UTF-8"
            )));
        }
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "metadata" | "warn" | "permissive" => Ok(ProviderPolicy::Metadata),
        "require_complete" | "require-complete" | "strict" | "1" | "true" => {
            Ok(ProviderPolicy::RequireComplete)
        }
        _ => Err(Error::Config(format!(
            "{ENV_PROVIDER_POLICY} must be one of metadata or require_complete"
        ))),
    }
}

pub(crate) fn provider_policy_blocks_query(
    incomplete_providers: &[serde_json::Value],
) -> Result<bool> {
    Ok(
        provider_policy_from_env()? == ProviderPolicy::RequireComplete
            && !incomplete_providers.is_empty(),
    )
}

pub(crate) fn graph_stale_skip_json(
    store: &greppy_store::Store,
    _root: Option<&str>,
    project: &str,
    command: &str,
    freshness: serde_json::Value,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let mut obj = serde_json::Map::new();
    obj.insert("command".into(), serde_json::json!(command));
    obj.insert("status".into(), serde_json::json!("skipped_stale_index"));
    obj.insert("project".into(), serde_json::json!(project));
    obj.insert("fresh".into(), serde_json::json!(false));
    obj.insert("freshness".into(), freshness);
    obj.insert(
        "provider_complete".into(),
        serde_json::json!(incomplete_providers.is_empty()),
    );
    obj.insert(
        "incomplete_provider_count".into(),
        serde_json::json!(incomplete_providers.len()),
    );
    obj.insert(
        "incomplete_providers".into(),
        serde_json::json!(incomplete_providers),
    );
    obj.insert("total_exact".into(), serde_json::json!(0));
    obj.insert("shown".into(), serde_json::json!(0));
    obj.insert("omitted".into(), serde_json::json!(0));
    obj.insert("truncated".into(), serde_json::json!(false));
    if let serde_json::Value::Object(extra) = extra {
        for (key, value) in extra {
            obj.insert(key, value);
        }
    }
    obj.insert(empty_collection_field.into(), serde_json::json!([]));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(obj)).map_err(|e| {
            Error::Invalid(format!("serialize stale graph JSON for {command}: {e}"))
        })?
    );
    Ok(())
}

/// Fresh-or-fallback gate for graph navigation. Indexed graph data is only
/// visible when freshness was proven; drift/unknown states trigger refresh
/// and return EX_TEMPFAIL instead of exposing stale rows.
pub(crate) fn graph_stale_gate(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    json: bool,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<Option<i32>> {
    match freshness_serve_decision(store, root, project) {
        FreshnessServe::Fresh(_) => Ok(None),
        FreshnessServe::Refuse(freshness) => {
            if json {
                graph_stale_skip_json(
                    store,
                    root,
                    project,
                    command,
                    freshness.clone(),
                    extra,
                    empty_collection_field,
                )?;
            } else {
                println!("{}", indexed_stale_skip_message(command, &freshness));
            }
            Ok(Some(freshness_refusal_exit(&freshness)))
        }
    }
}

pub(crate) fn provider_policy_graph_gate(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    json: bool,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<Option<i32>> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    if !provider_policy_blocks_query(&incomplete_providers)? {
        return Ok(None);
    }
    if json {
        provider_incomplete_skip_json(
            store,
            root,
            project,
            command,
            &incomplete_providers,
            extra,
            empty_collection_field,
        )?;
    } else {
        println!(
            "{}",
            provider_incomplete_skip_message(command, incomplete_providers.len())
        );
    }
    Ok(Some(1))
}

pub(crate) fn freshness_state_can_trigger_reindex(state: &str) -> bool {
    !matches!(
        state,
        "cold" | "config_error" | "failed" | "unknown" | "refreshing"
    )
}

pub(crate) fn freshness_serve_decision(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
) -> FreshnessServe {
    freshness_serve_decision_with_policy(store, root, project, true, true)
}

/// Heal a reindexable-stale store in-band: rebuild the graph AND (when the
/// store carried them) the embeddings + summaries at a fresh generation, then
/// re-open so the caller serves the current codebase. The edit loop mutates
/// files constantly — through greppy's own edits AND external means (git apply,
/// bash, another tool) — and every query command must reflect those changes or
/// the agent gets stale/empty answers and abandons greppy (forensics
/// 2026-07-18). Genuinely un-reindexable states (cold/failed) are left for the
/// stale gate to refuse. Best-effort: a failed reindex leaves the old store,
/// and the gate then decides.
pub(crate) fn maybe_reindex_stale(
    store: &mut greppy_store::Store,
    root: Option<&str>,
) -> Result<()> {
    // An explicit auto-reindex opt-out must fall through to the fail-closed
    // stale gate. In particular, do not wait on an active writer that the
    // caller has said must not be joined for automatic healing.
    if !auto_reindex_enabled() {
        return Ok(());
    }
    let project = project_for(root)?;
    if freshness_is_reindexable_stale(store, root, &project) {
        let rebuilt = try_auto_reindex_inline(root);
        if !rebuilt {
            let writer_active = workspace_writer_active(root);
            let started = if writer_active {
                false
            } else {
                spawn_background_index(root, "workspace-drift")
            };
            if started || writer_active || workspace_writer_active(root) {
                wait_for_active_index_refresh(root);
            }
        }
        if let Ok(fresh) = open_default_store_query_writer(root) {
            *store = fresh;
        }
    }
    Ok(())
}

/// An edit-owned or background indexer may already be building the exact fresh
/// snapshot this query needs. Give a short refresh a chance to publish, but
/// never strand a noninteractive caller behind a large repository build.
pub(crate) fn wait_for_active_index_refresh(root: Option<&str>) {
    let Ok(effective_root) = resolve_root(root) else {
        return;
    };
    let store_path = workspace_locator::store_path(&effective_root);
    let wait = std::time::Duration::from_secs(2);
    eprintln!("greppy: graph refresh already running; waiting up to 2s for a fresh snapshot");
    let deadline = std::time::Instant::now() + wait;
    loop {
        match greppy_freshness::try_acquire(&store_path) {
            Ok(lock) => {
                drop(lock);
                if store_path.exists() {
                    eprintln!("greppy: graph refresh published; resuming query");
                    return;
                }
                // A newly spawned first-use child publishes its job record
                // before it acquires the writer lock. Do not confuse that
                // short launch window with successful publication.
                if read_background_job(&background_job_path(&effective_root)).is_some()
                    && std::time::Instant::now() < deadline
                {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                eprintln!(
                    "greppy: graph refresh has not published a snapshot yet; inspect `greppy index status --json`"
                );
                return;
            }
            Err(greppy_freshness::LockError::Held { .. })
                if std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(greppy_freshness::LockError::Held { .. }) => {
                let job = read_background_job(&background_job_path(&effective_root));
                let phase = job
                    .as_ref()
                    .and_then(|value| value.get("state"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                let completed = job
                    .as_ref()
                    .and_then(|value| value.get("completed_spans"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let total = job
                    .as_ref()
                    .and_then(|value| value.get("total_spans"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let eta = job
                    .as_ref()
                    .and_then(|value| value.get("eta_seconds"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unknown".into());
                eprintln!(
                    "greppy: graph refresh still active — phase={phase}, completed={completed}/{total}, eta_seconds={eta}; returning temporary failure instead of waiting indefinitely; retry after `greppy index status --json` reports healthy=true"
                );
                return;
            }
            Err(greppy_freshness::LockError::Io { context, source }) => {
                eprintln!(
                    "greppy: cannot observe graph refresh ({context}: {source}); returning temporary failure; inspect `greppy index status --json`"
                );
                return;
            }
        }
    }
}

/// The index is stale AND the drift is one an inline reindex can heal
/// (workspace/content drift or a scope-stable version bump), not a cold or
/// broken store. Used by `read` to reindex in-band before serving rather than
/// refuse and leave the edit-loop agent empty-handed.
pub(crate) fn freshness_is_reindexable_stale(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
) -> bool {
    let freshness = nav_freshness_json(store, root, project);
    if freshness_json_is_fresh(&freshness) {
        return false;
    }
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    if !freshness_state_can_trigger_reindex(state) {
        return false;
    }
    let scope_or_version_drift = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|reasons| {
            reasons
                .iter()
                .filter_map(serde_json::Value::as_str)
                .any(|reason| reason.contains("indexer version/scope"))
        });
    if scope_or_version_drift {
        return version_drift_is_scope_stable(&freshness);
    }
    if metadata_only_fingerprint_drift(&freshness) {
        return false;
    }
    freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|count| {
            count as usize <= AUTO_REINDEX_MAX_FILES
                && freshness_changed_bytes(root, &freshness)
                    .is_some_and(|bytes| bytes <= 8 * 1024 * 1024)
        })
}

pub(crate) fn try_refresh_metadata_only_fingerprint(
    root: Option<&str>,
    freshness: &serde_json::Value,
) -> Option<serde_json::Value> {
    if !metadata_only_fingerprint_drift(freshness) {
        return None;
    }
    let effective_root = resolve_root(root).ok()?;
    let overrides = discover_overrides_from_env().ok()?;
    let store_path = workspace_locator::store_path(&effective_root);
    let _writer = greppy_freshness::try_acquire(&store_path).ok()?;
    let mut store =
        greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::query_writer())
            .ok()?;
    let fingerprint = greppy_core::GitFingerprint::capture(&effective_root);
    if !greppy_freshness::refresh_fingerprint_metadata(
        &mut store,
        &fingerprint,
        NAV_FRESHNESS_BUDGET,
        &overrides,
    )
    .ok()?
    {
        return None;
    }

    let mut refreshed = freshness.clone();
    let object = refreshed.as_object_mut()?;
    object.insert("fresh".into(), serde_json::Value::Bool(true));
    object.insert("state".into(), serde_json::Value::String("fresh".into()));
    object.insert("reasons".into(), serde_json::Value::Array(Vec::new()));
    Some(refreshed)
}

pub(crate) fn freshness_serve_decision_with_policy(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    allow_auto_reindex: bool,
    _warn_on_stale: bool,
) -> FreshnessServe {
    let freshness = nav_freshness_json(store, root, project);
    if freshness_json_is_fresh(&freshness) {
        return FreshnessServe::Fresh(freshness);
    }
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    // Unknown is not evidence of drift. In particular, a budget-exhausted
    // inventory walk must not launch a full reindex that can replace the DB
    // containing expand packs created by the preceding query.
    if !freshness_state_can_trigger_reindex(state) {
        return FreshnessServe::Refuse(freshness);
    }
    let scope_or_version_drift = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|rs| {
            rs.iter()
                .filter_map(serde_json::Value::as_str)
                .any(|r| r.contains("indexer version/scope"))
        });
    if scope_or_version_drift {
        if allow_auto_reindex && auto_reindex_enabled() && version_drift_is_scope_stable(&freshness)
        {
            let started = spawn_background_index(root, "indexer-version-drift");
            return FreshnessServe::Refuse(refresh_state(
                freshness,
                started || workspace_writer_active(root),
            ));
        }
        return FreshnessServe::Refuse(freshness);
    }

    // A commit can change only HEAD after the exact source contents were
    // already indexed. The inventory diff above proves there are zero stale
    // files, so refresh just the fingerprint instead of rebuilding the graph
    // and every embedding at a new generation.
    if allow_auto_reindex && auto_reindex_enabled() {
        if let Some(refreshed) = try_refresh_metadata_only_fingerprint(root, &freshness) {
            return FreshnessServe::Fresh(refreshed);
        }
    }

    let stale_file_count = freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);
    let small_enough = stale_file_count.is_some_and(|count| {
        count <= AUTO_REINDEX_MAX_FILES
            && freshness_changed_bytes(root, &freshness)
                .is_some_and(|bytes| bytes <= 8 * 1024 * 1024)
    });
    if allow_auto_reindex && auto_reindex_enabled() && small_enough {
        let rebuilt = try_auto_reindex_inline(root);
        let writer_active = workspace_writer_active(root);
        let started = if rebuilt || writer_active {
            false
        } else {
            spawn_background_index(root, "workspace-drift")
        };
        return FreshnessServe::Refuse(refresh_state(
            freshness,
            rebuilt || started || writer_active || workspace_writer_active(root),
        ));
    }
    if allow_auto_reindex && auto_reindex_enabled() {
        let started = spawn_background_index(root, "workspace-drift");
        return FreshnessServe::Refuse(refresh_state(
            freshness,
            started || workspace_writer_active(root),
        ));
    }
    FreshnessServe::Refuse(freshness)
}

pub(crate) fn freshness_changed_bytes(
    root: Option<&str>,
    freshness: &serde_json::Value,
) -> Option<u64> {
    let root = resolve_root(root).ok()?;
    let paths = freshness.get("changed_paths")?.as_array()?;
    let mut bytes = 0u64;
    for path in paths {
        let path = path.as_str()?;
        match std::fs::metadata(root.join(path)) {
            Ok(metadata) => bytes = bytes.saturating_add(metadata.len()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return None,
        }
    }
    Some(bytes)
}

pub(crate) fn freshness_refusal_exit(freshness: &serde_json::Value) -> i32 {
    match freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
    {
        "refreshing" | "drift" | "unknown" => EXIT_TEMPFAIL as i32,
        _ => 1,
    }
}

pub(crate) fn workspace_writer_active(root: Option<&str>) -> bool {
    let Ok(root) = resolve_root(root) else {
        return false;
    };
    let hash = greppy_core::workspace::workspace_hash(&root);
    matches!(
        greppy_core::cache::acquire_named_lock(
            &format!("workspace-{hash}.writer"),
            greppy_core::cache::LockMode::Exclusive,
            true,
        ),
        Ok(None)
    )
}

pub(crate) fn auto_reindex_inline_allowed(
    had_vectors: bool,
    indexed_files: i64,
    overlay: bool,
) -> bool {
    !had_vectors
        && (overlay || (0..=AUTO_REINDEX_INLINE_MAX_INDEXED_FILES).contains(&indexed_files))
}

/// Build a genuinely bounded small-drift refresh through the same
/// temp-snapshot publication boundary as an explicit `index`. Vector-backed
/// or large full stores return false so the caller starts one observable
/// background refresh instead of hiding model loading or a full repository
/// rebuild inside a navigation command.
pub(crate) fn try_auto_reindex_inline(root: Option<&str>) -> bool {
    let Ok(effective_root) = resolve_root(root) else {
        return false;
    };
    let Ok(project) = project_for(root) else {
        return false;
    };
    let Ok(overrides) = discover_overrides_from_env() else {
        return false;
    };
    let store_path = workspace_locator::store_path(&effective_root);
    let Ok(Some(_lifecycle)) = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    ) else {
        return false;
    };
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => lock,
        _ => return false, // another writer is active: refuse this snapshot
    };
    let overlay = crate::store_cow::overlay_spec_live(&effective_root)
        .ok()
        .flatten();
    let store = match overlay.as_ref() {
        Some(overlay) => greppy_store::Store::open_overlay_read_only(
            &overlay.base_path,
            &store_path,
            &overlay.visibility,
        ),
        None => greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::read_only()),
    };
    let Ok(store) = store else {
        return false;
    };
    // Remember whether this store served code-span vectors BEFORE the
    // reindex bumps the generation: an inline graph-only reindex would
    // otherwise strand every existing vector row on the old generation and
    // silently degrade `context`/`semantic-search` until a manual
    // `grep index` run (the owner's "gains" path dying quietly).
    let had_vectors = !store
        .vector_model_ids(&project)
        .unwrap_or_default()
        .is_empty();
    let indexed_files = store.file_count(&project).unwrap_or(i64::MAX);
    if !auto_reindex_inline_allowed(had_vectors, indexed_files, overlay.is_some()) {
        return false;
    }
    drop(store);
    let options = greppy_indexer::IndexOptions {
        discover_overrides: overrides,
        only_paths: None,
    };
    if let Some(overlay) = overlay.as_ref() {
        crate::indexing::index_overlay_snapshot(
            &store_path,
            &effective_root,
            &project,
            overlay,
            None,
            &options,
            false,
            None,
        )
        .map(|code| code == 0)
        .unwrap_or(false)
    } else {
        index_atomic_snapshot(
            &store_path,
            &effective_root,
            &project,
            None,
            &options,
            false,
            None,
        )
        .map(|snapshot| snapshot.index.is_clean())
        .unwrap_or(false)
    }
}

pub(crate) fn freshness_json_is_fresh(freshness: &serde_json::Value) -> bool {
    freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn open_default_store(root: Option<&str>) -> Result<greppy_store::Store> {
    // The graph DB lives under the platform locator, never at
    // `<cwd>/.greppy/graph.db`. When no
    // `--root` is given we detect the repo root by walking up for a
    // marker, so a query from a subdirectory targets the same store the
    // indexer wrote from the repo root (instead of opening an empty
    // store under the subdir's hash and exiting 73).
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    // RV-007: tighten the store dir + DB file permissions on every open.
    // This is a no-op when the store doesn't exist yet (read paths before
    // any `greppy index` would have failed to open the store anyway).
    if let Some(parent) = path.parent() {
        let _ = workspace_locator::ensure_store_dir(parent);
    }
    if let Some((base_path, base_commit)) = crate::store_cow::overlay_environment(&effective_root)?
    {
        greppy_core::cache::ensure_workspace_store(&effective_root).map_err(|error| {
            Error::io(
                format!(
                    "create private Delta Store for {}",
                    effective_root.display()
                ),
                error,
            )
        })?;
        if !path.exists() {
            drop(greppy_store::Store::open(&path)?);
        }
        let delta = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::read_only())?;
        let visibility = crate::store_cow::visibility_for_open_connection(
            &effective_root,
            &base_commit,
            delta.conn(),
        )?;
        let store = delta.attach_overlay(&base_path, &visibility)?;
        let _ = workspace_locator::ensure_db_mode(&path);
        if let Some(store_dir) = path.parent() {
            workspace_locator::touch_lastused(store_dir);
        }
        return Ok(store);
    }
    // Forensics F4: a query against a repo that was never indexed used to
    // open a non-existent DB, fail deep in SQLite, and exit 73 (EXIT_IO)
    // with NOTHING on stdout/stderr — the agent just saw an empty result and
    // a bare non-zero code, with no hint that the fix is `greppy index`.
    //
    // Auto-index on first use, but never hide an unbounded repository walk
    // inside a navigation command. Start the ordinary detached indexer and
    // join it for at most the same two-second window used by stale queries.
    // Small repositories still feel immediate; large repositories return a
    // retryable result with a stable status surface instead of appearing hung.
    // Gated behind GREPPY_AUTO_REINDEX so explicit opt-out keeps the old error.
    // (Query commands only — the grep passthrough path never reaches here,
    // so the byte-exact passthrough contract is untouched.)
    if !path.exists() {
        let shown_root = root.unwrap_or(".");
        if auto_reindex_enabled() {
            let started = spawn_background_index(root, "first-use");
            if started || workspace_writer_active(root) {
                wait_for_active_index_refresh(root);
            }
            if !path.exists() {
                return Err(Error::Lock(format!(
                    "first-use index {} for {}; no snapshot is ready yet; retry after `greppy index status --json` reports healthy=true (or run `greppy index {}` in the foreground)",
                    if started { "started" } else { "is already running" },
                    effective_root.display(),
                    shown_root
                )));
            }
        } else {
            eprintln!(
                "greppy: no index for {} — run `greppy index {}` first",
                effective_root.display(),
                shown_root
            );
            return Err(Error::Invalid(format!(
                "no index for {}; run `greppy index {}` first",
                effective_root.display(),
                shown_root
            )));
        }
    }
    // Query commands are READ-ONLY: open read-only so they skip both
    // `migrate()` and the O(db-size) `integrity_check` that a read-write open
    // runs. Those belong on the writer (`greppy index`); paying them on every
    // query open made who-calls/search take seconds on a real repo
    // (the token-efficiency benchmark's latency culprit). Readers tolerate
    // whatever schema the DB has.
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::read_only())?;
    let _ = workspace_locator::ensure_db_mode(&path);
    // Feature B: record that this store was just used to serve a query.
    // A read-only open never bumps graph.db's mtime, so a dedicated
    // `.lastused` marker is what keeps a frequently-queried store from
    // being evicted by `cleanup_stale_stores`. Best-effort — a failed
    // touch never fails the query.
    if let Some(store_dir) = path.parent() {
        workspace_locator::touch_lastused(store_dir);
    }
    // O5 session prewarm: the first graph command of an agent session nudges
    // the embed daemon (with an async model load) so a following `context`
    // query hits a warm model instead of paying the cold start. Guarded to
    // fire only when semantic search is actually in play — env model
    // configured AND this store holds vectors — because prewarming a model
    // nobody will query would hold GPU memory for a TTL for nothing.
    #[cfg(any(unix, windows))]
    {
        let no_args = EmbeddingCliArgs {
            device: None,
            no_gpu: false,
        };
        if let Ok(Some(cfg)) = embedding_config_optional(no_args) {
            let has_vectors = project_for(root)
                .ok()
                .and_then(|p| store.vector_model_ids(&p).ok())
                .is_some_and(|m| !m.is_empty());
            if has_vectors {
                let key = embedding_query_cache_key(&cfg);
                embed_daemon::prewarm_from_env(&cfg, &key);
            }
        }
    }
    Ok(store)
}

pub(crate) fn open_default_store_query_writer(root: Option<&str>) -> Result<greppy_store::Store> {
    open_default_store_writer(root, true)
}

/// Open the small writable evidence/continuation surface without forcing a
/// graph build. Exact filesystem reads must remain available before the first
/// index; their pagination records are not graph-query evidence.
pub(crate) fn open_default_store_pack_writer(root: Option<&str>) -> Result<greppy_store::Store> {
    open_default_store_writer(root, false)
}

fn open_default_store_writer(
    root: Option<&str>,
    require_existing_index: bool,
) -> Result<greppy_store::Store> {
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    if let Some(overlay) = crate::store_cow::overlay_spec(&effective_root)? {
        greppy_core::cache::ensure_workspace_store(&effective_root).map_err(|error| {
            Error::io(
                format!(
                    "create private Delta Store for {}",
                    effective_root.display()
                ),
                error,
            )
        })?;
        if !path.exists() {
            drop(greppy_store::Store::open(&path)?);
        }
        return greppy_store::Store::open_overlay(&overlay.base_path, &path, &overlay.visibility)
            .map_err(Into::into);
    }
    if require_existing_index && !path.exists() {
        // Reuse the normal query open to trigger the existing first-use
        // auto-index/error path, then reopen writable for the evidence write.
        drop(open_default_store(root)?);
    }
    if let Some(parent) = path.parent() {
        let _ = workspace_locator::ensure_store_dir(parent);
    }
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())?;
    let _ = workspace_locator::ensure_db_mode(&path);
    if let Some(store_dir) = path.parent() {
        workspace_locator::touch_lastused(store_dir);
    }
    Ok(store)
}

pub(crate) fn cleanup_expired_legacy_entries(
    current: Option<&std::path::Path>,
    ttl: std::time::Duration,
) {
    if ttl.is_zero() {
        return;
    }
    let now = unix_now_secs_cli();
    for entry in verified_legacy_cache_entries() {
        if current == Some(entry.root.as_path()) || entry.locked {
            continue;
        }
        if now.saturating_sub(entry.last_used_unix_secs) > ttl.as_secs() {
            let _ = remove_verified_legacy_entry(&entry);
        }
    }
}

/// Resume only legacy trash entries whose name, SQLite header, schema and
/// workspace hash all prove that Greppy created them. Unknown trash is left
/// untouched and remains visible as unmanaged cache data.
pub(crate) fn cleanup_verified_legacy_trash() {
    let Ok(entries) = std::fs::read_dir(greppy_core::cache::trash_root()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(rest) = name.strip_prefix("legacy-") else {
            continue;
        };
        let Some(hash) = rest.get(..16) else {
            continue;
        };
        if !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            || rest.as_bytes().get(16) != Some(&b'-')
        {
            continue;
        }
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            continue;
        }
        let graph = path.join("graph.db");
        if !sqlite_header_is_valid(&graph) {
            continue;
        }
        let Ok(connection) = rusqlite::Connection::open_with_flags(
            &graph,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) else {
            continue;
        };
        let schema_valid = connection
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .is_some();
        let workspace_valid = connection
            .query_row(
                "SELECT root_path FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .is_some_and(|root| {
                greppy_core::workspace::workspace_hash(std::path::Path::new(&root))
                    .eq_ignore_ascii_case(hash)
            });
        drop(connection);
        if schema_valid && workspace_valid {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

pub(crate) fn cleanup_stale_snapshot_artifacts(
    active_path: &std::path::Path,
    include_quarantine: bool,
) -> Result<usize> {
    let Some(parent) = active_path.parent() else {
        return Ok(0);
    };
    let Some(file_name) = active_path.file_name().and_then(|s| s.to_str()) else {
        return Ok(0);
    };
    let next_prefix = format!("{file_name}.next.");
    let corrupt_prefix = format!("{file_name}.corrupt.");
    let previous = format!("{file_name}.prev");
    let previous_sidecar_prefix = format!("{previous}-");
    let mut removed = 0usize;
    let entries = match std::fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(Error::io(format!("scan {}", parent.display()), e)),
    };
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(format!("scan {}", parent.display()), e))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let managed = name.starts_with(&next_prefix)
            || name == previous
            || name.starts_with(&previous_sidecar_prefix)
            || (name.starts_with(".index.job.") && name.ends_with(".tmp"))
            || (include_quarantine && name.starts_with(&corrupt_prefix));
        if !managed {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::io(
                    format!("remove stale temp {}", path.display()),
                    e,
                ))
            }
        }
    }
    if removed > 0 {
        sync_parent_dir(active_path)?;
    }
    Ok(removed)
}

pub(crate) fn cleanup_sqlite_family(path: &std::path::Path) -> Result<()> {
    remove_file_if_exists(path)?;
    cleanup_sqlite_sidecars(path)
}

pub(crate) fn cleanup_sqlite_sidecars(path: &std::path::Path) -> Result<()> {
    remove_file_if_exists(&sqlite_sidecar(path, "-wal"))?;
    remove_file_if_exists(&sqlite_sidecar(path, "-shm"))
}
