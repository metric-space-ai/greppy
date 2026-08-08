//! Building and refreshing the index.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

pub(crate) fn dispatch_index_status(json: bool, root: Option<&str>) -> Result<i32> {
    dispatch_index_health("index-status", json, root)
}

pub(crate) fn dispatch_index_health(command: &str, json: bool, root: Option<&str>) -> Result<i32> {
    let effective_root = resolve_root(root)?;
    let project = workspace_locator::project_identity(&effective_root);
    let store_path = workspace_locator::store_path(&effective_root);
    let store_format = store_path
        .parent()
        .and_then(|parent| greppy_core::cache::read_store_manifest(parent).ok())
        .map(|manifest| manifest.format_version);
    let store_bytes = store_path
        .parent()
        .map(cache_path_bytes)
        .unwrap_or_default();
    let background_job = read_background_job(&background_job_path(&effective_root));
    let effective_root_string = effective_root.to_string_lossy().into_owned();
    let writer_active = workspace_writer_active(Some(&effective_root_string));
    let job_state = background_job.as_ref().map(|job| {
        if job
            .get("pid")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|pid| process_is_alive(pid as u32))
        {
            "refreshing"
        } else {
            "failed"
        }
    });
    let background_state = if writer_active {
        Some("refreshing")
    } else {
        job_state
    };
    let dirty_overlay = dirty_overlay(&effective_root)?;
    let inference = (command == "doctor")
        .then(inference_registry_status)
        .transpose()?;
    let inference_daemons = (command == "doctor").then(inference_daemon_status);
    let inference_diagnostics = inference.as_ref().map(|registry| {
        serde_json::json!({
            "registry": registry,
            "daemons": inference_daemons,
            "models": inference_model_status(),
        })
    });

    if !store_path.exists() {
        let status = serde_json::json!({
            "command": command,
            "status": "no_index",
            "healthy": false,
            "store_exists": false,
            "root_path": effective_root,
            "store_path": store_path,
            "store_format": store_format,
            "store_bytes": store_bytes,
            "background_job": background_job,
            "background_state": background_state,
            "embedding_complete": false,
            "project": project,
            "fresh": false,
            "freshness": null,
            "schema_current": false,
            "integrity_ok": false,
            "project_present": false,
            "incomplete_provider_count": null,
            "skip_counts_by_reason": [],
            "dirty_overlay": dirty_overlay.to_json(),
            "inference": inference_diagnostics,
            "message": "no active index; run greppy index first",
        });
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&status)
                    .map_err(|e| Error::Invalid(format!("serialize {command} JSON: {e}")))?
            );
        } else {
            println!("status: no_index");
            println!("root: {}", effective_root.display());
            println!("store: {}", store_path.display());
            println!("message: run `greppy index {}` first", root.unwrap_or("."));
            if let Some(inference) = &inference {
                print_inference_registry(inference);
            }
            if let Some(daemons) = &inference_daemons {
                print_inference_daemons(daemons);
            }
            if dirty_overlay.git_available && !dirty_overlay.clean {
                println!(
                    "dirty_overlay: total={} staged={} unstaged={} untracked={} deleted={} renamed={} ignored={}",
                    dirty_overlay.total,
                    dirty_overlay.staged_count,
                    dirty_overlay.unstaged_count,
                    dirty_overlay.untracked_count,
                    dirty_overlay.deleted_count,
                    dirty_overlay.renamed_count,
                    dirty_overlay.ignored_count
                );
            }
        }
        return Ok(1);
    }

    let store =
        greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::read_only())?;
    let diag = store.diagnostics()?;
    let freshness = nav_freshness_json(&store, root, &project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let project_diag = diag.projects.iter().find(|p| p.project.name == project);
    let workspace = diag
        .workspace_states
        .iter()
        .find(|w| w.root_path == effective_root.to_string_lossy());
    let project_present = project_diag.is_some();
    let incomplete_provider_count = project_diag
        .map(|p| p.incomplete_provider_count)
        .unwrap_or(0);
    let provider_states = project_diag
        .map(|p| p.provider_states.clone())
        .unwrap_or_default();
    let provider_failure_count = provider_states
        .iter()
        .filter(|provider| provider.status != "unsupported")
        .map(|provider| provider.files_failed.max(0) as u64)
        .sum::<u64>();
    let skip_counts = project_diag
        .map(|p| {
            p.skip_counts_by_reason
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "reason": s.reason,
                        "count": s.count,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let stats = project_diag.map(|p| {
        serde_json::json!({
            "files": p.stats.file_count,
            "nodes": p.stats.total_nodes,
            "edges": p.stats.total_edges,
        })
    });
    let graph_generation = workspace.map(|w| w.graph_generation);
    let current_embedding_rows = graph_generation
        .and_then(|generation| {
            store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM vector_embeddings WHERE project = ?1 AND graph_generation = ?2",
                    (&project, generation as i64),
                    |row| row.get::<_, i64>(0),
                )
                .ok()
        })
        .unwrap_or(0);
    let configured_embedding_model = embedding_config_optional(EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    })
    .ok()
    .flatten();
    let embedding_complete = graph_generation.is_some_and(|generation| {
        let Some(model) = configured_embedding_model.as_ref() else {
            return false;
        };
        let key = embedding_complete_key(&project);
        store
            .conn()
            .query_row(
                "SELECT value FROM schema_meta WHERE key = ?1",
                [&key],
                |row| row.get::<_, String>(0),
            )
            .ok()
            == Some(format!("{generation}|{}", model.model_id))
    });
    // Robustness (problem dossier, systemic lesson 1&2): silent
    // under-indexing must be VISIBLE. Two independent-oracle checks:
    //   * coverage: compare the store's indexed file count against
    //     `git ls-files` — a discovery bug (out-of-root gitignore leak,
    //     O9-class) shows up as a tiny fraction of the tracked files.
    //   * vectors: a configured embedding model with zero stored vectors
    //     means every semantic query silently degrades to lexical.
    let indexed_files = project_diag.map(|p| p.stats.file_count).unwrap_or(0);
    let git_tracked = git_tracked_file_count(&effective_root);
    let coverage_warning = match git_tracked {
        Some(tracked) if tracked >= 100 && (indexed_files as u64) * 5 < tracked => Some(format!(
            "store indexed {indexed_files} files but git tracks {tracked} — \
             discovery may be dropping files (nested-repo ignore rules?); \
             re-run `greppy index` with the current binary"
        )),
        _ => None,
    };
    let vectors_missing_with_model = configured_embedding_model.is_some()
        && store
            .vector_model_ids(&project)
            .map(|v| v.is_empty())
            .unwrap_or(false);
    let inference_healthy = inference
        .as_ref()
        .is_none_or(greppy_embed_native::InferenceBackendRegistry::is_satisfied);
    let embedding_healthy = embedding_complete || test_inference_skipped();
    let healthy = diag.schema_current
        && diag.integrity_ok
        && project_present
        && fresh
        && embedding_healthy
        && provider_failure_count == 0
        && coverage_warning.is_none()
        && inference_healthy
        && background_state != Some("refreshing");
    let status_label = if healthy { "ok" } else { "unhealthy" };

    if json {
        let value = serde_json::json!({
            "command": command,
            "status": status_label,
            "healthy": healthy,
            "store_exists": true,
            "root_path": effective_root,
            "store_path": store_path,
            "store_format": store_format,
            "store_bytes": store_bytes,
            "background_job": background_job,
            "background_state": background_state,
            "embedding_complete": embedding_complete,
            "current_embedding_rows": current_embedding_rows,
            "project": project,
            "fresh": fresh,
            "freshness": freshness,
            "schema_version": diag.schema_version,
            "expected_schema_version": diag.expected_schema_version,
            "schema_current": diag.schema_current,
            "integrity_ok": diag.integrity_ok,
            "integrity_messages": diag.integrity_messages,
            "project_present": project_present,
            "graph_generation": graph_generation,
            "stats": stats,
            "incomplete_provider_count": incomplete_provider_count,
            "provider_failure_count": provider_failure_count,
            "providers": provider_states,
            "skip_counts_by_reason": skip_counts,
            "git_tracked_files": git_tracked,
            "coverage_warning": coverage_warning,
            "vectors_missing_with_model": vectors_missing_with_model,
            "dirty_overlay": dirty_overlay.to_json(),
            "inference": inference_diagnostics,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize {command} JSON: {e}")))?
        );
    } else {
        println!("status: {status_label}");
        if let Some(w) = &coverage_warning {
            println!("coverage_warning: {w}");
        }
        if vectors_missing_with_model {
            println!(
                "vectors: none stored though an embedding model is configured \
                 — `semantic-search` will build them on first use, or run \
                 `grep index` now"
            );
        }
        println!("root: {}", effective_root.display());
        println!("store: {}", store_path.display());
        println!("store_format: {}", store_format.unwrap_or(0));
        println!("store_bytes: {store_bytes}");
        println!("embedding_complete: {embedding_complete}");
        if let Some(inference) = &inference {
            print_inference_registry(inference);
        }
        if let Some(daemons) = &inference_daemons {
            print_inference_daemons(daemons);
        }
        if let Some(state) = background_state {
            println!("background_job: {state}");
        }
        println!("project: {project}");
        println!(
            "schema: {}/{} {}",
            diag.schema_version,
            diag.expected_schema_version,
            if diag.schema_current {
                "current"
            } else {
                "stale"
            }
        );
        println!(
            "integrity: {}",
            if diag.integrity_ok { "ok" } else { "failed" }
        );
        println!(
            "freshness: {}",
            if fresh {
                "fresh".to_string()
            } else {
                stale_freshness_reason(&freshness)
            }
        );
        if let Some(generation) = graph_generation {
            println!("generation: {generation}");
        }
        if let Some(project_diag) = project_diag {
            println!(
                "stats: files={} nodes={} edges={}",
                project_diag.stats.file_count,
                project_diag.stats.total_nodes,
                project_diag.stats.total_edges
            );
            println!("incomplete_providers: {incomplete_provider_count}");
            println!("provider_file_failures: {provider_failure_count}");
            for skip in &project_diag.skip_counts_by_reason {
                println!("skipped {} {}", skip.reason, skip.count);
            }
        } else {
            println!("project_present: false");
        }
        if dirty_overlay.git_available && !dirty_overlay.clean {
            println!(
                "dirty_overlay: total={} staged={} unstaged={} untracked={} deleted={} renamed={} ignored={}",
                dirty_overlay.total,
                dirty_overlay.staged_count,
                dirty_overlay.unstaged_count,
                dirty_overlay.untracked_count,
                dirty_overlay.deleted_count,
                dirty_overlay.renamed_count,
                dirty_overlay.ignored_count
            );
        }
    }

    Ok(if healthy { 0 } else { EXIT_IO as i32 })
}

/// Run the indexer against `path` (default: current directory).
/// Warm the worktree `greppy -p` will use, instead of this checkout.
///
/// The built-in agent works in a per-repository worktree whose workspace
/// identity is its own path, so indexing the checkout leaves the agent cold and
/// it pays for the first index -- graph plus embeddings -- inside its own run.
/// Preparing the tree here is the same operation the agent performs, so the
/// agent's later run finds it registered and reuses it.
pub(crate) fn dispatch_index_agent_worktree(
    path: Option<&str>,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
    let repo = resolve_root(root.or(path))?;
    let workspace = greppy_agent::workspace::AgentWorkspace::create(&repo, "index-warm")
        .map_err(|error| {
            Error::Invalid(format!(
                "no agent worktree for {}: {error}",
                repo.display()
            ))
        })?;
    // A disposable fallback tree is warmed for nothing: the agent's next run
    // takes the stable path and finds it cold. Say so instead of reporting a
    // warm index that the measured run will never see.
    if !workspace.is_stable() {
        return Err(Error::Invalid(format!(
            "another `greppy -p` run holds the agent worktree for {}; \
             warming the disposable fallback would leave the real one cold — \
             retry once that run finishes",
            repo.display()
        )));
    }
    let worktree = workspace.worktree_path().to_string_lossy().into_owned();
    if !cli_json_output() {
        println!("agent worktree: {worktree}");
    }
    // Walk AND key the store by the worktree, so the identity matches the one
    // the agent resolves; keying by the checkout would warm a third workspace.
    dispatch_index(Some(&worktree), Some(&worktree), embedding_args)
}

pub(crate) fn dispatch_index(
    path: Option<&str>,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
    let mut background_job = BackgroundJobGuard::from_env();
    // RV-006: `--root` overrides the indexed target. When both are
    // given we still walk `path` (the user's workspace) but key the
    // store under the canonical `root` so the indexer and the
    // query commands share one project identity (RV-011).
    // Defect D9: normalize BOTH paths to canonical absolute form up
    // front. `greppy index .` used to record whatever the walker
    // derived from the relative target (falling back to `.` in a
    // marker-less directory), while later queries looked the workspace
    // up under an absolute root — the index existed but every lookup
    // failed. Canonical-absolute at the boundary keeps one spelling
    // everywhere.
    let target = match path {
        Some(p) => absolutize_path(std::path::Path::new(p)),
        None => std::env::current_dir()
            .map_err(|e| Error::io("read current_dir for `grep index` default", e))?,
    };
    // RV-006 / RV-011: the store path and project identity are keyed on
    // the *resolved* repo root, not on the (possibly sub-directory) index
    // target. When `--root` is given we honour it; otherwise we walk up
    // from `target` to the repo marker. This guarantees `greppy index
    // <subdir>` and a later `greppy search-code` from anywhere in the
    // same repo open the same store and use the same project name.
    let effective_root = match root {
        Some(r) => {
            let explicit = absolutize_path(std::path::Path::new(r));
            workspace_locator::resolve_workspace_root(&explicit)
        }
        None => find_repo_root(&target),
    };
    let project = workspace_locator::project_identity(&effective_root);
    let index_options = greppy_indexer::IndexOptions {
        discover_overrides: discover_overrides_from_env()?,
    };
    let embedding_config = embedding_config_for_index(embedding_args)?;

    // Open the on-disk store under the workspace locator's path
    // never at `<root>/.greppy/graph.db` (which would
    // pollute `grep -R .`). The versioned platform data directory is used on
    // Linux/macOS and can be overridden via `GREPPY_STORE_DIR`.
    let store_path = workspace_locator::store_path(&effective_root);
    greppy_core::cache::ensure_workspace_store(&effective_root).map_err(|e| {
        Error::io(
            format!("create workspace store for {}", effective_root.display()),
            e,
        )
    })?;
    let _lifecycle = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    )
    .map_err(|error| Error::io("acquire index lifecycle lease", error))?
    .ok_or_else(|| Error::Lock("blocking lifecycle lease returned no guard".into()))?;
    // Acquire the crash-safe
    // advisory lock BEFORE opening/migrating the store. Opening first lets a
    // concurrent indexer hit a SQLite busy error inside Store::open and exit
    // EXIT_IO (73) silently, instead of the documented EX_TEMPFAIL (75) with a
    // diagnostic on contention. Concurrent indexers on the same path get
    // `LockError::Held`; a crashed prior holder is released by the OS. The
    // guard must outlive the complete snapshot build + publish operation.
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => Some(lock),
        Err(greppy_freshness::LockError::Held { .. }) => {
            // Contention is a status, not a dead end: another process is
            // already building the very index this call wanted. Saying only
            // that it is "running" left the caller with nothing to do next --
            // and this fires exactly when a stale-index answer has just told
            // them to run `greppy index`, so the two messages together used to
            // form a loop with no exit.
            eprintln!(
                "grep: another indexer is already building the index for {} — \
                 wait for it to finish, then retry; `greppy index status --json` \
                 reports its progress",
                store_path.display()
            );
            return Ok(EXIT_TEMPFAIL as i32);
        }
        Err(greppy_freshness::LockError::Io { context, source }) => {
            return Err(Error::io(context, source));
        }
    };
    // Holding the writer lock, build a fresh snapshot in a temp DB, validate
    // it, then publish it with one filesystem rename. The indexer crate still
    // supports in-place incremental updates for library tests; the CLI path is
    // the production publication boundary, so it must never expose a half-built
    // graph.db to query commands.
    let is_background = background_job.is_background();
    let snapshot = match index_atomic_snapshot(
        &store_path,
        &target,
        &project,
        embedding_config.as_ref(),
        &index_options,
        !is_background,
        if is_background {
            Some(&mut background_job)
        } else {
            None
        },
    ) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            background_job.fail(&error);
            return Err(error);
        }
    };
    let report = &snapshot.index;

    println!(
        "indexed {} files ({} unsupported, {} unreadable, {} oversize, {} file-limit, {} time-budget); {} nodes extracted; generation {} (project: {project})",
        report.files_indexed,
        report.files_unsupported_language,
        report.files_unreadable,
        report.files_oversize,
        report.files_skipped_by_file_limit,
        report.files_skipped_by_time_budget,
        report.nodes_extracted,
        report.graph_generation
    );
    if !report.is_clean()
        || report.files_skipped_by_file_limit > 0
        || report.files_skipped_by_time_budget > 0
    {
        return Ok(EXIT_IO as i32);
    }
    if let Some(embedding_report) = &snapshot.embeddings {
        println!(
            "embedded {} code spans ({} reused, {} considered, {} non-definition skipped, {} missing-file, {} invalid-span, {} oversize, {} failed, {} stale pruned)",
            embedding_report.nodes_embedded,
            embedding_report.nodes_reused,
            embedding_report.nodes_considered,
            embedding_report.nodes_skipped_non_definition,
            embedding_report.nodes_skipped_missing_file,
            embedding_report.nodes_skipped_invalid_span,
            embedding_report.nodes_skipped_oversize,
            embedding_report.nodes_failed,
            embedding_report.stale_rows_pruned
        );
    }
    let discover_scope = index_options.discover_overrides.scope_key();
    if discover_scope != "default" {
        println!(
            "discover scope: {discover_scope} ({} / {})",
            ENV_DISCOVER_INCLUDE, ENV_DISCOVER_EXCLUDE
        );
    }
    retire_verified_legacy_store(&effective_root);
    match snapshot.embedding_degraded.as_deref() {
        // Degraded embeddings never cost the caller the published graph
        // snapshot: record the reason (background job record / stderr) and
        // let the background embed path finish the remaining vectors.
        Some(reason) => background_job.degraded(reason),
        None => background_job.complete(),
    }
    let embedding_deferred = snapshot.embedding_deferred;
    drop(_lock);
    drop(_lifecycle);
    if let Some(reason) = snapshot.embedding_degraded.as_deref() {
        // No immediate respawn: a broken backend would fail the same way
        // again. The next semantic query re-attempts through the existing
        // background-embed path and reuses every vector that DID embed.
        eprintln!(
            "greppy index: embedding generation degraded ({reason}); the graph index is published and complete; the next semantic query retries the remaining embeddings."
        );
    }
    if embedding_deferred {
        if let Some(cfg) = embedding_config.as_ref() {
            let effective_root_string = effective_root.to_string_lossy().into_owned();
            if spawn_background_embed(Some(&effective_root_string), cfg) {
                let progress =
                    embedding_progress_value(&effective_root, cfg, report.graph_generation);
                println!("{}", embedding_progress_text(&progress));
            } else {
                println!(
                    "semantic-search: semantic index is pending; the next semantic query will retry the background job."
                );
            }
        }
    }
    Ok(0)
}

pub(crate) fn index_atomic_snapshot(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
    allow_deferred_embeddings: bool,
    mut background_job: Option<&mut BackgroundJobGuard>,
) -> Result<IndexSnapshotReport> {
    for attempt in 0..2 {
        if let Some(report) = index_atomic_snapshot_attempt(
            active_path,
            target,
            project,
            embedding_config,
            index_options,
            allow_deferred_embeddings,
            background_job.as_deref_mut(),
        )? {
            return Ok(report);
        }
        if attempt == 0 {
            eprintln!("greppy: workspace changed during indexing; rebuilding snapshot once");
        }
    }
    Err(Error::Store(
        "workspace kept changing during indexing; snapshot was not published".into(),
    ))
}

pub(crate) fn index_atomic_snapshot_attempt(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
    allow_deferred_embeddings: bool,
    background_job: Option<&mut BackgroundJobGuard>,
) -> Result<Option<IndexSnapshotReport>> {
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    let temp_path = unique_store_sibling(active_path, "next");
    cleanup_sqlite_family(&temp_path)?;
    seed_temp_store_from_active_if_usable(active_path, &temp_path)?;

    let mut temp_store = match greppy_store::Store::open(&temp_path) {
        Ok(store) => store,
        Err(e) => {
            let _ = cleanup_sqlite_family(&temp_path);
            return Err(e.into());
        }
    };

    let report =
        match greppy_indexer::index_with_options(&mut temp_store, target, project, index_options) {
            Ok(report) => report,
            Err(e) => {
                drop(temp_store);
                let _ = cleanup_sqlite_family(&temp_path);
                return Err(e);
            }
        };

    if !report.is_clean()
        || report.files_skipped_by_file_limit > 0
        || report.files_skipped_by_time_budget > 0
    {
        drop(temp_store);
        cleanup_sqlite_family(&temp_path)?;
        return Ok(Some(IndexSnapshotReport {
            index: report,
            embeddings: None,
            embedding_deferred: false,
            embedding_degraded: None,
        }));
    }

    let embedding_deferred = embedding_config.is_some_and(|cfg| {
        allow_deferred_embeddings
            && greppy_indexer::count_embedding_candidate_nodes(&temp_store, project)
                .is_ok_and(|count| should_defer_embedding(cfg, count))
    });
    let (embedding_report, embedding_degraded) =
        if let Some(cfg) = embedding_config.filter(|_| !embedding_deferred) {
            match index_embeddings_into_temp_store(
                &mut temp_store,
                target,
                project,
                cfg,
                &report,
                active_path.parent().map(std::path::Path::to_path_buf),
                background_job,
            ) {
                Ok(EmbeddingBuildOutcome::Complete(report)) => (Some(report), None),
                Ok(EmbeddingBuildOutcome::Degraded { report, reason }) => (report, Some(reason)),
                Err(e) => {
                    drop(temp_store);
                    let _ = cleanup_sqlite_family(&temp_path);
                    return Err(e);
                }
            }
        } else {
            (None, None)
        };

    checkpoint_store(&temp_store, &temp_path)?;
    temp_store.integrity_check().map_err(|e| {
        Error::Store(format!(
            "temp index integrity_check failed for {}: {e}",
            temp_path.display()
        ))
    })?;
    drop(temp_store);
    cleanup_sqlite_sidecars(&temp_path)?;
    sync_file(&temp_path)?;
    sync_parent_dir(&temp_path)?;
    maybe_index_test_failpoint("after-temp-before-publish", &temp_path)?;

    let verify_store =
        greppy_store::Store::open_with(&temp_path, greppy_store::OpenOptions::read_only())?;
    let verification = greppy_freshness::check_files_report_with_ttl(
        &verify_store,
        target,
        project,
        std::time::Duration::from_secs(300),
        &index_options.discover_overrides,
        std::time::Duration::ZERO,
    )?;
    drop(verify_store);
    if !matches!(
        verification.state.outcome,
        greppy_freshness::FreshnessOutcome::Fresh
    ) {
        cleanup_sqlite_family(&temp_path)?;
        return Ok(None);
    }

    publish_store_snapshot(&temp_path, active_path)?;
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    Ok(Some(IndexSnapshotReport {
        index: report,
        embeddings: embedding_report,
        embedding_deferred,
        embedding_degraded,
    }))
}

pub(crate) fn index_embeddings_into_temp_store(
    store: &mut greppy_store::Store,
    target: &std::path::Path,
    project: &str,
    cfg: &EmbeddingModelConfig,
    report: &greppy_indexer::IndexReport,
    tokenizer_cache_dir: Option<std::path::PathBuf>,
    mut background_job: Option<&mut BackgroundJobGuard>,
) -> Result<EmbeddingBuildOutcome> {
    #[cfg(debug_assertions)]
    if std::env::var_os(ENV_TEST_EMBED_UNAVAILABLE).is_some() {
        return Ok(EmbeddingBuildOutcome::Degraded {
            report: None,
            reason: "test failpoint: embedding backend unavailable".into(),
        });
    }
    if let Some(job) = background_job.as_deref_mut() {
        job.embedding_loading();
    }
    let model = match load_embedding_model(cfg, tokenizer_cache_dir) {
        Ok(model) => model,
        Err(e) => {
            log_embedding_skip_once("index --embeddings", &e);
            return Ok(EmbeddingBuildOutcome::Degraded {
                report: None,
                reason: format!("embedding model load failed: {e}"),
            });
        }
    };
    let mut provider = greppy_indexer::EmbeddingGemmaCodeProvider::new(&cfg.model_id, &model);
    let options = greppy_indexer::EmbeddingIndexOptions::for_generation(report.graph_generation);
    let embedding_report = if let Some(job) = background_job {
        let total_documents = greppy_indexer::count_code_embedding_documents_for_project(
            store, target, project, &provider, options,
        )?;
        job.embedding_started(model.backend_name(), total_documents);
        let mut progress = |value| job.embedding_progress(value);
        greppy_indexer::index_code_embeddings_for_project_with_progress(
            store,
            target,
            project,
            &mut provider,
            options,
            total_documents,
            &mut progress,
        )?
    } else {
        greppy_indexer::index_code_embeddings_for_project(
            store,
            target,
            project,
            &mut provider,
            options,
        )?
    };
    if !embedding_report.is_complete() {
        // The completeness stamp is deliberately withheld: the next
        // semantic query (or the spawned background job) re-runs the
        // embedding pass, reusing every vector that DID embed by content
        // hash and retrying only the failed documents.
        let reason = format!(
            "{} of {} embedding documents failed inference",
            embedding_report.nodes_failed,
            embedding_report
                .nodes_failed
                .saturating_add(embedding_report.nodes_embedded)
        );
        return Ok(EmbeddingBuildOutcome::Degraded {
            report: Some(embedding_report),
            reason,
        });
    }
    let key = embedding_complete_key(project);
    store
        .conn()
        .execute(
            "INSERT INTO schema_meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rusqlite::params![key, format!("{}|{}", report.graph_generation, cfg.model_id)],
        )
        .map_err(|error| Error::Store(format!("record embedding completeness: {error}")))?;
    Ok(EmbeddingBuildOutcome::Complete(embedding_report))
}
