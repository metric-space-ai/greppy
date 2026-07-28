//! Text, symbol and meaning search.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

pub(crate) fn dispatch_search_graph(
    q: greppy_search::GraphQuery,
    name_filter: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let q = if q.project.is_none() {
        q.with_project(project.clone())
    } else {
        q
    };
    let limit = q.limit;
    let graph_gate_extra = serde_json::json!({
        "filters": {
            "name": name_filter,
        },
        "scope": "node_search",
        "limit": limit,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "search-graph",
        json,
        graph_gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "search-graph",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let rows = greppy_search::search_graph(&store, &q)?;
    if json {
        let total_exact = greppy_search::count_search_graph(&store, &q)?;
        search_graph_counts_json(
            &store,
            root,
            &project,
            name_filter,
            limit,
            total_exact,
            &rows,
        )?;
        return Ok(0);
    }
    if rows.is_empty() {
        println!("(no matches)");
    } else {
        for r in &rows {
            println!(
                "{}  {}  {}:{}  {}",
                r.label,
                display_row_name(r),
                r.file_path,
                r.start_line,
                r.name
            );
        }
    }
    Ok(0)
}

pub(crate) fn search_graph_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    name_filter: Option<&str>,
    limit: usize,
    total_exact: usize,
    rows: &[greppy_search::graph::SearchGraphRow],
) -> Result<()> {
    let shown = rows.len();
    let omitted = total_exact.saturating_sub(shown);
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let hits: Vec<_> = rows.iter().map(graph_row_json).collect();
    let v = serde_json::json!({
        "command": "search-graph",
        "project": project,
        "filters": {
            "name": name_filter,
        },
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": "node_search",
        "limit": limit,
        "total_exact": total_exact,
        "shown": shown,
        "omitted": omitted,
        "truncated": omitted > 0,
        "hits": hits,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize search-graph JSON: {e}")))?
    );
    Ok(())
}

pub(crate) fn semantic_embedding_indexing_json(
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    freshness: &serde_json::Value,
    progress: &serde_json::Value,
    fallback: SemanticFallbackContext<'_>,
) -> Result<()> {
    let eta_seconds = progress
        .get("eta_seconds")
        .and_then(serde_json::Value::as_u64);
    let retry_after_seconds = eta_seconds.map(|eta| eta.clamp(5, 30)).unwrap_or(10);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
            "command": "semantic-search",
            "mode": "vector",
            "status": "indexing",
            "project": project,
            "model_id": cfg.model_id,
            "prompt_version": greppy_embed_native::PROMPT_VERSION,
            "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
            "graph_generation": graph_generation,
            "fresh": freshness_json_is_fresh(freshness),
            "freshness": freshness,
            "retryable": true,
            "retry_after_seconds": retry_after_seconds,
            "embedding_index": progress,
            "query_tokens": semantic_fallback_tokens(fallback.query),
            "next": semantic_fallback_commands(fallback.query, fallback.paths, fallback.root),
            "total_exact": 0,
            "shown": 0,
            "omitted": 0,
            "truncated": false,
            "hits": [],
        }))
        .map_err(|error| Error::Invalid(format!("serialize semantic indexing JSON: {error}")))?
    );
    Ok(())
}

/// Semantic queries are plain English, so the symbol-shape rules do not apply;
/// only `-` is expanded.
pub(crate) fn semantic_queries(raw: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for value in raw {
        if value == "-" {
            out.extend(targets_from_stdin()?);
            continue;
        }
        if value.trim().is_empty() {
            return Err(Error::Invalid(
                "empty query: semantic-search needs a plain-English description".into(),
            ));
        }
        out.push(value.clone());
    }
    Ok(out)
}

/// `search-symbols A B` — one lookup per name, each result attributed.
pub(crate) fn dispatch_search_symbols_multi(
    targets: &[String],
    paths: &[String],
    kind: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    if targets.len() <= 1 {
        return dispatch_search_symbols(
            targets.first().map(String::as_str),
            paths,
            kind,
            json,
            root,
        );
    }
    let path_filters = prepare_query_path_filters(root, "search-symbols", "", paths)?;
    let mut store = open_default_store(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let mut entries = Vec::with_capacity(targets.len());
    let mut hits = Vec::new();
    for query in targets {
        let mut found = greppy_search::search_symbols_in_project(&store, &project, query, 10_000)?;
        if let Some(want) = kind.map(|k| k.to_ascii_lowercase()) {
            found.retain(|hit| {
                store
                    .get_node(hit.node_id)
                    .ok()
                    .flatten()
                    .is_some_and(|node| node.label.to_ascii_lowercase() == want)
            });
        }
        let mut rows = Vec::new();
        for hit in &found {
            let Some(node) = store.get_node(hit.node_id)? else {
                continue;
            };
            if !path_filters.matches(&node.file_path) {
                continue;
            }
            let mut value = node_hit_json(&node);
            value["target"] = serde_json::json!(query);
            value["label"] = serde_json::json!(&node.label);
            value["name"] = serde_json::json!(&node.name);
            rows.push(value);
        }
        entries.push(serde_json::json!({
            "symbol": query,
            "symbol_found": !rows.is_empty(),
            "total_exact": rows.len(),
        }));
        hits.extend(rows);
    }
    let total = hits.len();
    let end = cli_result_limit_raw().unwrap_or(usize::MAX).min(total);
    let window = hits[..end].to_vec();
    if json {
        let value = serde_json::json!({
            "command": "search-symbols",
            "status": "ok",
            "project": project,
            "targets": entries,
            "total_exact": total,
            "shown": window.len(),
            "hits": window,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize search-symbols JSON: {e}")))?
        );
        return Ok(0);
    }
    for row in &window {
        println!(
            "{} {} {}:{}",
            row["label"].as_str().unwrap_or(""),
            row["qualified_name"].as_str().unwrap_or(""),
            row["file"].as_str().unwrap_or(""),
            row["line"].as_i64().unwrap_or(0)
        );
    }
    Ok(0)
}

pub(crate) fn dispatch_search_symbols(
    query: Option<&str>,
    paths: &[String],
    kind: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("search-symbols requires a query".into()));
    }
    let path_filters = prepare_query_path_filters(root, "search-symbols", q, paths)?;
    let mut store = open_default_store(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    // Symbol rows are visible only from a freshness-proven snapshot.
    let decision = freshness_serve_decision(&store, root, &project);
    if let FreshnessServe::Refuse(freshness) = &decision {
        if json {
            search_symbols_json(
                &store,
                q,
                &project,
                "skipped_stale_index",
                Some(freshness),
                &[],
                &path_filters,
                Some(0),
            )?;
        } else {
            println!(
                "{}",
                indexed_stale_skip_message("search-symbols", freshness)
            );
        }
        return Ok(freshness_refusal_exit(freshness));
    }
    let freshness = decision.freshness().clone();
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            search_symbols_json(
                &store,
                q,
                &project,
                "skipped_incomplete_provider",
                Some(&freshness),
                &[],
                &path_filters,
                Some(0),
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("search-symbols", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    // Path/kind filters are post-query result filters: fetch broadly, then
    // narrow on node metadata without changing symbol ranking/resolution.
    let fetch = if kind.is_some() || !path_filters.is_empty() {
        10_000
    } else {
        cli_result_limit(20)
    };
    let mut hits = greppy_search::search_symbols_in_project(&store, &project, q, fetch)?;
    if let Some(k) = kind {
        let want = k.to_ascii_lowercase();
        hits.retain(|h| {
            store
                .get_node(h.node_id)
                .ok()
                .flatten()
                .map(|n| n.label.to_ascii_lowercase() == want)
                .unwrap_or(false)
        });
    }
    hits.retain(|hit| {
        store
            .get_node(hit.node_id)
            .ok()
            .flatten()
            .is_some_and(|node| path_filters.matches(&node.file_path))
    });
    let total_filtered = hits.len() as i64;
    hits.truncate(cli_result_limit(20));
    if json {
        search_symbols_json(
            &store,
            q,
            &project,
            "ok",
            Some(&freshness),
            &hits,
            &path_filters,
            (!path_filters.is_empty() || kind.is_some()).then_some(total_filtered),
        )?;
        return Ok(if hits.is_empty() { 1 } else { 0 });
    }
    if hits.is_empty() {
        // A name that matches no definition is an empty answer, not a failure:
        // `AGENTS.md` says such a question prints nothing and exits 0, and a
        // caller that pipes `search-symbols` into `read -` must not receive a
        // line of prose where it expects results. The path filter is the one
        // thing worth saying, because it can be the reason the answer is empty.
        if !path_filters.is_empty() {
            println!("(no matches under path filter: {})", path_filters.shown());
        }
    } else {
        for h in &hits {
            // Resolve each FTS hit to its node so we can print the
            // actionable label + qualified_name + file:line instead of
            // a bare node id (matches the other query commands' output).
            match store.get_node(h.node_id)? {
                Some(n) => println!(
                    "{} {} {}:{}",
                    n.label,
                    display_node_name(&n),
                    n.file_path,
                    n.start_line
                ),
                None => println!("node={}", h.node_id),
            }
        }
    }
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search_symbols_json(
    store: &greppy_store::Store,
    query: &str,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    hits: &[greppy_search::SymbolHit],
    path_filters: &QueryPathFilters,
    total_override: Option<i64>,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let total_exact = if status == "ok" {
        match total_override {
            Some(total) => total,
            None => greppy_search::count_symbols_in_project(store, project, query)?,
        }
    } else {
        0
    };
    let mut rows = Vec::new();
    for h in hits {
        match store.get_node(h.node_id)? {
            Some(n) => rows.push(serde_json::json!({
                "node_id": h.node_id,
                "rank": h.rank,
                "target": query,
                "label": n.label,
                "name": n.name,
                "qualified_name": n.qualified_name,
                "file": n.file_path,
                "line": n.start_line,
                "file_path": n.file_path,
                "start_line": n.start_line,
                "end_line": n.end_line,
            })),
            None => rows.push(serde_json::json!({
                "node_id": h.node_id,
                "rank": h.rank,
                "source_available": false,
            })),
        }
    }
    let shown = rows.len() as i64;
    let omitted = total_exact.saturating_sub(shown);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-symbols",
            "status": status,
            "query": query,
            "project": project,
            "path_filters": path_filters.json_value(),
            "fresh": freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": incomplete_providers.is_empty(),
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-symbols JSON: {e}")))?
    );
    Ok(())
}

pub(crate) fn search_code_definition_entry(
    root_path: &std::path::Path,
    row: &greppy_search::graph::SearchGraphRow,
) -> Result<Option<SearchCodeDefinitionEntry>> {
    if row.start_line < 1 || is_synthetic_file_anchor(&row.label, &row.name, &row.qualified_name) {
        return Ok(None);
    }
    let absolute = root_path.join(&row.file_path);
    let content = match std::fs::read(&absolute) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(Error::Io {
                context: format!("read {}", absolute.display()),
                source,
            });
        }
    };
    let Some(span) = read_span_with_meta(
        root_path,
        &row.file_path,
        row.start_line,
        row.end_line,
        usize::MAX,
        false,
    ) else {
        return Ok(None);
    };
    let (byte_start, byte_end) =
        line_range_to_bytes(&content, row.start_line as usize, span.end_line as usize);
    let mut handle = greppy_edit::EditHandle::for_range(
        root_path,
        std::path::Path::new(&row.file_path),
        &content,
        byte_start,
        byte_end,
    )?;
    let language = greppy_edit::language_for_path(std::path::Path::new(&row.file_path));
    handle.signature_fingerprint =
        greppy_edit::verbs::signature_fingerprint(language, &content, (byte_start, byte_end));
    handle.grammar_id = Some(format!("{language:?}"));
    handle.grammar_version = Some(env!("CARGO_PKG_VERSION").to_string());
    Ok(Some(SearchCodeDefinitionEntry {
        node_id: row.id,
        qualified_name: row.qualified_name.clone(),
        file: row.file_path.clone(),
        start_line: row.start_line,
        end_line: span.end_line,
        source: span.text,
        handle: handle.encode(),
        matches: Vec::new(),
    }))
}

pub(crate) fn search_code_entries(
    store: &greppy_store::Store,
    project: &str,
    root_path: &std::path::Path,
    hits: &[greppy_search::CodeHit],
    resolve_definitions: bool,
) -> Result<Vec<SearchCodeEntry>> {
    let mut entries = Vec::new();
    let mut definition_entries = std::collections::HashMap::<i64, usize>::new();
    for hit in hits {
        let Some(match_line) = parse_search_code_match(hit) else {
            continue;
        };
        let row = if resolve_definitions {
            greppy_search::definition_at(store, Some(project), &match_line.file, match_line.line)?
        } else {
            None
        };
        let Some(row) = row else {
            entries.push(SearchCodeEntry::Unenclosed(match_line));
            continue;
        };
        if let Some(index) = definition_entries.get(&row.id).copied() {
            if let SearchCodeEntry::Definition(definition) = &mut entries[index] {
                definition.matches.push(match_line);
            }
            continue;
        }
        let Some(mut definition) = search_code_definition_entry(root_path, &row)? else {
            entries.push(SearchCodeEntry::Unenclosed(match_line));
            continue;
        };
        definition.matches.push(match_line);
        let index = entries.len();
        definition_entries.insert(definition.node_id, index);
        entries.push(SearchCodeEntry::Definition(definition));
    }
    Ok(entries)
}

pub(crate) fn search_code_entry_json(entry: &SearchCodeEntry) -> serde_json::Value {
    match entry {
        SearchCodeEntry::Definition(definition) => serde_json::json!({
            "qualified_name": &definition.qualified_name,
            "file": &definition.file,
            "span": {
                "start_line": definition.start_line,
                "end_line": definition.end_line,
            },
            "source": &definition.source,
            "handle": &definition.handle,
            "matches": definition.matches.iter().map(|hit| serde_json::json!({
                "location": &hit.location,
                "line": hit.line,
                "text": &hit.text,
            })).collect::<Vec<_>>(),
        }),
        SearchCodeEntry::Unenclosed(hit) => serde_json::json!({
            "qualified_name": serde_json::Value::Null,
            "file": &hit.file,
            "span": {
                "start_line": hit.line,
                "end_line": hit.line,
            },
            "source": serde_json::Value::Null,
            "handle": serde_json::Value::Null,
            "matches": [{
                "location": &hit.location,
                "line": hit.line,
                "text": &hit.text,
            }],
        }),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_search_code(
    query: Option<&str>,
    paths: &[String],
    changed: bool,
    staged: bool,
    since: Option<&str>,
    base: Option<&str>,
    json: bool,
    no_code: bool,
    fixed: bool,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("search-code requires a query".into()));
    }
    let path_filters = prepare_query_path_filters(root, "search-code", q, paths)?;
    let git_scope_count = usize::from(changed)
        + usize::from(staged)
        + usize::from(since.is_some())
        + usize::from(base.is_some());
    if git_scope_count > 1 {
        return Err(Error::Invalid(
            "search-code accepts only one git scope flag at a time".into(),
        ));
    }
    if changed {
        return dispatch_search_code_changed(q, json, fixed, root, &path_filters);
    }
    if staged {
        return dispatch_search_code_staged(q, json, fixed, root, &path_filters);
    }
    if let Some(rev) = since {
        return dispatch_search_code_since(q, rev, json, fixed, root, &path_filters);
    }
    if let Some(rev) = base {
        return dispatch_search_code_base(q, rev, json, fixed, root, &path_filters);
    }

    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let root_path = resolve_root(root)?;
    let decision = freshness_serve_decision(&store, root, &project);
    let resolve_definitions = matches!(decision, FreshnessServe::Fresh(_));
    let status = if resolve_definitions {
        "ok"
    } else {
        "live-fallback"
    };
    if let FreshnessServe::Refuse(freshness) = &decision {
        if !json {
            eprintln!(
                "{}; falling back to live grep",
                indexed_stale_skip_message("search-code", freshness)
            );
        }
    }
    let all_hits = if path_filters.is_empty() {
        live_grep_code_hits_pattern(q, &root_path, fixed)?
    } else {
        live_grep_code_hits_filtered_pattern(q, &root_path, &path_filters, fixed)?
    };
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();
    emit_search_code_results_with_format(
        &store,
        q,
        &project,
        status,
        Some(decision.freshness()),
        all_hits.len(),
        &shown_hits,
        &path_filters,
        &root_path,
        json,
        no_code,
        fixed,
        resolve_definitions,
    )?;
    Ok(if all_hits.is_empty() { 1 } else { 0 })
}

pub(crate) fn dispatch_search_code_changed(
    query: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let mut changed_files = git_changed_files(&root_path)?;
    changed_files.retain(|path| path_filters.matches(path));
    let all_hits = live_grep_search_code_paths_pattern(query, &root_path, &changed_files, fixed)?;
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_changed_json(
            query,
            &project,
            changed_files.len(),
            all_hits.len(),
            &shown_hits,
            path_filters,
        )?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        print_search_code_no_matches(query, fixed, path_filters);
        return Ok(0);
    }
    for h in &shown_hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

pub(crate) fn search_code_changed_json(
    query: &str,
    project: &str,
    changed_files_total: usize,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
    path_filters: &QueryPathFilters,
) -> Result<()> {
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "location": h.location,
                "rank": h.rank,
                "snippet": clamp_snippet(&h.snippet).as_ref(),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": if total_exact == 0 { "no_matches" } else { "ok" },
            "query": query,
            "project": project,
            "scope": "changed",
            "path_filters": path_filters.json_value(),
            "backend": "live_grep",
            "fresh": true,
            "freshness": serde_json::Value::Null,
            "changed_files_total": changed_files_total,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-code changed JSON: {e}")))?
    );
    Ok(())
}

pub(crate) fn dispatch_search_code_staged(
    query: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let mut staged_files = git_staged_files(&root_path)?;
    staged_files.retain(|path| path_filters.matches(path));
    let all_hits = grep_staged_git_blobs_pattern(query, &root_path, &staged_files, fixed)?;
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_staged_json(
            query,
            &project,
            staged_files.len(),
            all_hits.len(),
            &shown_hits,
        )?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        print_search_code_no_matches(query, fixed, path_filters);
        return Ok(0);
    }
    for h in &shown_hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

pub(crate) fn search_code_staged_json(
    query: &str,
    project: &str,
    staged_files_total: usize,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
) -> Result<()> {
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "location": h.location,
                "rank": h.rank,
                "snippet": clamp_snippet(&h.snippet).as_ref(),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": if total_exact == 0 { "no_matches" } else { "ok" },
            "query": query,
            "project": project,
            "scope": "staged",
            "backend": "git_blob_grep",
            "fresh": true,
            "freshness": serde_json::Value::Null,
            "staged_files_total": staged_files_total,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-code staged JSON: {e}")))?
    );
    Ok(())
}

pub(crate) fn dispatch_search_code_since(
    query: &str,
    rev: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    dispatch_search_code_diff_scope(
        query,
        DiffSearchScope::Since { rev },
        json,
        fixed,
        root,
        path_filters,
    )
}

pub(crate) fn dispatch_search_code_base(
    query: &str,
    base: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    dispatch_search_code_diff_scope(
        query,
        DiffSearchScope::Base { base },
        json,
        fixed,
        root,
        path_filters,
    )
}

pub(crate) fn dispatch_search_code_diff_scope(
    query: &str,
    scope: DiffSearchScope<'_>,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let mut spec = git_diff_search_spec(&root_path, scope)?;
    spec.files.retain(|path| path_filters.matches(path));
    let all_hits = live_grep_search_code_paths_pattern(query, &root_path, &spec.files, fixed)?;
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_diff_scope_json(query, &project, &spec, all_hits.len(), &shown_hits)?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        print_search_code_no_matches(query, fixed, path_filters);
        return Ok(0);
    }
    for h in &shown_hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

pub(crate) fn search_code_diff_scope_json(
    query: &str,
    project: &str,
    spec: &DiffSearchSpec,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
) -> Result<()> {
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "location": h.location,
                "rank": h.rank,
                "snippet": clamp_snippet(&h.snippet).as_ref(),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": if total_exact == 0 { "no_matches" } else { "ok" },
            "query": query,
            "project": project,
            "scope": spec.scope,
            "backend": "git_diff_live_grep",
            "fresh": true,
            "freshness": serde_json::Value::Null,
            "diff_rev": &spec.diff_rev,
            "merge_base": spec.merge_base.as_deref(),
            "diff_files_total": spec.files.len(),
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-code diff JSON: {e}")))?
    );
    Ok(())
}

pub(crate) fn dispatch_semantic(
    query: Option<&str>,
    paths: &[String],
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("semantic-search requires a query".into()));
    }
    let path_filters = prepare_query_path_filters(root, "semantic-search", q, paths)?;

    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    // Stale/unknown snapshots are never served. Semantic search is always
    // vector-backed on current main, so auto-refresh is allowed only when the
    // embedding model can be rebuilt in the same atomic snapshot.
    let allow_reindex = vector_auto_reindex_can_rebuild(embedding_args);
    let decision =
        freshness_serve_decision_with_policy(&store, root, &project, allow_reindex, false);
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    let freshness = decision.freshness().clone();

    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            semantic_provider_incomplete_json(
                &project,
                "vector",
                Some(&freshness),
                &incomplete_providers,
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("semantic-search", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    let cfg = match embedding_config_for_required_use(embedding_args) {
        Ok(cfg) => cfg,
        Err(error) if embedding_asset_missing_error(&error) => {
            emit_semantic_backend_unavailable(
                &project,
                q,
                paths,
                root,
                json,
                "EmbeddingGemma assets could not be resolved; use one of the exact non-semantic fallbacks below.",
            )?;
            return Ok(i32::from(EXIT_NOT_IMPLEMENTED));
        }
        Err(error) => return Err(error),
    };
    {
        let generation = current_graph_generation(&store, root)?;
        let candidate_limit = vector_exact_candidate_limit()?;
        if !freshness_json_is_fresh(&freshness) {
            let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
                &project,
                &cfg.model_id,
                Some(generation),
                SEMANTIC_VECTOR_CANDIDATE_LIMIT,
            );
            scope.limit = SEMANTIC_VECTOR_CANDIDATE_LIMIT;
            let total = greppy_search::count_vector_search_scope(&store, &scope)?;
            if json {
                semantic_vector_json(
                    &store,
                    &project,
                    &cfg,
                    generation,
                    total,
                    candidate_limit,
                    Some(&freshness),
                    "skipped_stale_index",
                    &[],
                )?;
            } else {
                println!(
                    "{}",
                    vector_stale_skip_message("semantic-search", &freshness)
                );
            }
            return Ok(freshness_refusal_exit(&freshness));
        }
        if !embedding_generation_complete(&store, &project, generation, &cfg.model_id) {
            let root_path = resolve_root(root)?;
            let _ = spawn_background_embed(root, &cfg);
            let progress = embedding_progress_value(&root_path, &cfg, generation);
            if json {
                semantic_embedding_indexing_json(
                    &project,
                    &cfg,
                    generation,
                    &freshness,
                    &progress,
                    SemanticFallbackContext {
                        query: q,
                        paths,
                        root,
                    },
                )?;
            } else {
                println!("{}", embedding_progress_text(&progress));
                print_semantic_fallback_commands(q, paths, root);
            }
            return Ok(i32::from(EXIT_TEMPFAIL));
        }
        let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
            &project,
            &cfg.model_id,
            Some(generation),
            SEMANTIC_VECTOR_CANDIDATE_LIMIT,
        );
        let total = greppy_search::count_vector_search_scope(&store, &scope)?;
        if total == 0 {
            if json {
                semantic_vector_json(
                    &store,
                    &project,
                    &cfg,
                    generation,
                    total,
                    candidate_limit,
                    Some(&freshness),
                    "no_indexed_vectors",
                    &[],
                )?;
            } else {
                println!(
                    "semantic index unavailable — 0 indexed spans for model {}",
                    cfg.model_id
                );
                print_semantic_fallback_commands(q, paths, root);
            }
            return Ok(freshness_refusal_exit(&freshness));
        }
        if let Some(limit) = vector_exact_scan_exceeds_limit(total, candidate_limit) {
            if json {
                semantic_vector_json(
                    &store,
                    &project,
                    &cfg,
                    generation,
                    total,
                    candidate_limit,
                    Some(&freshness),
                    "skipped_exact_scan_candidate_limit",
                    &[],
                )?;
            } else {
                println!(
                    "{}",
                    vector_exact_scan_skip_message("semantic-search", total, limit)
                );
            }
            return Ok(1);
        }

        match embed_query_cached(&cfg, root, q) {
            Ok(query_vector) => {
                scope.limit = SEMANTIC_VECTOR_CANDIDATE_LIMIT;
                let mut candidates =
                    greppy_search::vector_search_exact(&store, &query_vector, &scope)?;
                candidates.retain(|hit| path_filters.matches(&hit.embedding.file_path));
                let hits = dedupe_semantic_vector_hits(
                    candidates,
                    cli_result_limit(SEMANTIC_VECTOR_RESULT_LIMIT),
                );
                let shown = hits
                    .len()
                    .min(cli_result_limit(SEMANTIC_VECTOR_DISPLAY_LIMIT));
                let display_hits = hits[..shown].to_vec();
                let further_hits = &hits[shown..];
                let purposes = semantic_vector_purposes(&store, root, &display_hits, true)?;
                let expand = insert_semantic_vector_expand_pack(
                    &store,
                    root,
                    &project,
                    q,
                    generation,
                    further_hits,
                );
                if json {
                    semantic_vector_json_with_expand(
                        &store,
                        &project,
                        &cfg,
                        generation,
                        total,
                        hits.len(),
                        candidate_limit,
                        Some(&freshness),
                        "ok",
                        &display_hits,
                        purposes.as_deref(),
                        expand.as_ref(),
                    )?;
                } else if hits.is_empty() {
                    println!("(no vector matches)");
                    return Ok(1);
                } else {
                    for h in &display_hits {
                        print_semantic_vector_hit(h, purposes.as_deref());
                    }
                    if let Some(expand) = &expand {
                        println!("{}", expand.semantic_text_line());
                    }
                }
                Ok(if hits.is_empty() { 1 } else { 0 })
            }
            Err(e) => Err(e),
        }
    }
}

pub(crate) fn semantic_vector_purposes(
    store: &greppy_store::Store,
    root: Option<&str>,
    hits: &[greppy_store::VectorSearchHit],
    summarize: bool,
) -> Result<Option<Vec<SemanticVectorPurpose>>> {
    if hits.is_empty() {
        return Ok(None);
    }
    let root_path = match resolve_root(root) {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    #[cfg(any(unix, windows))]
    let summary_runtime = if summarize {
        qwen_summary_config_optional().ok().flatten()
    } else {
        None
    }
    .map(|cfg| {
        let model_key = qwen_summary_model_key(&cfg);
        (cfg, model_key)
    });
    let mut purposes = Vec::new();
    for hit in hits {
        let node = hit
            .embedding
            .node_id
            .and_then(|id| store.get_node(id).ok().flatten());
        let file_path = node
            .as_ref()
            .map(|n| n.file_path.as_str())
            .unwrap_or(&hit.embedding.file_path);
        let start_line = node
            .as_ref()
            .map(|n| n.start_line)
            .unwrap_or(hit.embedding.start_line);
        let stored_end_line = node
            .as_ref()
            .map(|n| n.end_line)
            .unwrap_or(hit.embedding.end_line);
        let Some(span) = read_span_with_meta(
            &root_path,
            file_path,
            start_line,
            stored_end_line,
            SEMANTIC_PURPOSE_SPAN_CAP_LINES,
            false,
        ) else {
            continue;
        };
        let signature = node
            .as_ref()
            .and_then(|node| node.properties.get("source_signature"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| semantic_signature_from_span(&span.text));
        let Some(signature) = signature else {
            continue;
        };
        let mut bullets = Vec::new();
        #[cfg(any(unix, windows))]
        if semantic_signature_is_function_like(&signature, node.as_ref().map(|n| n.label.as_str()))
        {
            if let Some((cfg, model_key)) = summary_runtime.as_ref() {
                let code = cap_semantic_purpose_span(&span.text);
                bullets =
                    summarize_daemon::summarize_source_via_daemon(cfg, model_key, file_path, &code)
                        .unwrap_or_default();
            }
        }
        purposes.push(SemanticVectorPurpose {
            embedding_id: hit.embedding.id,
            file_path: file_path.to_string(),
            start_line,
            end_line: span.end_line,
            display_loc: line_span(file_path, start_line, span.end_line),
            signature,
            bullets,
        });
    }
    if purposes.is_empty() {
        Ok(None)
    } else {
        Ok(Some(purposes))
    }
}

pub(crate) fn semantic_signature_from_span(code: &str) -> Option<String> {
    let mut leading_offset = 0usize;
    for line in code.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.trim().is_empty() || trimmed.starts_with("//") {
            leading_offset += line.len();
        } else {
            break;
        }
    }
    let start = code[leading_offset..]
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(leading_offset + idx))?;
    let declaration = &code[start..];
    let python_declaration =
        declaration.starts_with("def ") || declaration.starts_with("async def ");
    let bytes = code.as_bytes();
    let mut round_depth = 0usize;
    let mut square_depth = 0usize;
    let mut angle_depth = 0usize;
    let mut string_delimiter = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut end = code.len();
    let mut idx = start;
    while idx < bytes.len() {
        let byte = bytes[idx];
        let next = bytes.get(idx + 1).copied();
        if line_comment {
            if byte == b'\n' {
                line_comment = false;
            }
            idx += 1;
            continue;
        }
        if block_comment_depth > 0 {
            if byte == b'/' && next == Some(b'*') {
                block_comment_depth += 1;
                idx += 2;
            } else if byte == b'*' && next == Some(b'/') {
                block_comment_depth -= 1;
                idx += 2;
            } else {
                idx += 1;
            }
            continue;
        }
        if let Some(delimiter) = string_delimiter {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == delimiter {
                string_delimiter = None;
            }
            idx += 1;
            continue;
        }
        if byte == b'/' && next == Some(b'/') {
            line_comment = true;
            idx += 2;
            continue;
        }
        if byte == b'/' && next == Some(b'*') {
            block_comment_depth = 1;
            idx += 2;
            continue;
        }
        match byte {
            b'"' => string_delimiter = Some(byte),
            b'\'' if python_declaration => string_delimiter = Some(byte),
            b'(' => round_depth += 1,
            b')' => round_depth = round_depth.saturating_sub(1),
            b'[' => square_depth += 1,
            b']' => square_depth = square_depth.saturating_sub(1),
            b'<' => angle_depth += 1,
            b'>' if angle_depth > 0 => angle_depth -= 1,
            b'{' if round_depth == 0 && square_depth == 0 && angle_depth == 0 => {
                end = idx;
                break;
            }
            b':' if python_declaration
                && round_depth == 0
                && square_depth == 0
                && angle_depth == 0 =>
            {
                end = idx;
                break;
            }
            b';' if round_depth == 0 && square_depth == 0 && angle_depth == 0 => {
                end = idx;
                break;
            }
            _ => {}
        }
        idx += 1;
    }
    let signature = code[start..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!signature.is_empty()).then_some(signature)
}

pub(crate) fn semantic_signature_is_function_like(signature: &str, label: Option<&str>) -> bool {
    if let Some(label) = label {
        let lower = label.to_ascii_lowercase();
        if lower.contains("function") || lower.contains("method") {
            return true;
        }
        if lower.contains("struct")
            || lower.contains("class")
            || lower.contains("enum")
            || lower.contains("trait")
            || lower.contains("interface")
            || lower.contains("module")
        {
            return false;
        }
    }
    let s = signature.trim_start();
    s.starts_with("fn ")
        || s.starts_with("pub fn ")
        || s.starts_with("async fn ")
        || s.starts_with("pub async fn ")
        || s.starts_with("def ")
        || s.starts_with("async def ")
        || s.starts_with("function ")
        || s.contains(" function ")
}

pub(crate) fn semantic_vector_json_row(
    hit: &greppy_store::VectorSearchHit,
    purpose: Option<&SemanticVectorPurpose>,
    expand: Option<&ExpandHandle>,
) -> serde_json::Value {
    let mut row = serde_json::json!({
        "score": hit.score,
        "qualified_name": hit.embedding.qualified_name,
        "file_path": hit.embedding.file_path,
        "start_line": hit.embedding.start_line,
        "end_line": hit.embedding.end_line,
        "content_sha256": hit.embedding.content_sha256,
        "graph_generation": hit.embedding.graph_generation,
        "summary": [],
    });
    if let Some(purpose) = purpose {
        row["file_path"] = serde_json::json!(&purpose.file_path);
        row["start_line"] = serde_json::json!(purpose.start_line);
        row["end_line"] = serde_json::json!(purpose.end_line);
        row["signature"] = serde_json::json!(&purpose.signature);
        row["summary_loc"] = serde_json::json!(&purpose.display_loc);
        row["summary"] = serde_json::json!(&purpose.bullets);
        if !purpose.bullets.is_empty() {
            row["summary_prompt_version"] = serde_json::json!(greppy_qwen35_native::PROMPT_VERSION);
        }
    }
    if let Some(expand) = expand {
        row["expand_id"] = serde_json::json!(&expand.id);
    }
    row
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn semantic_vector_json(
    store: &greppy_store::Store,
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    total: i64,
    candidate_limit: Option<i64>,
    freshness: Option<&serde_json::Value>,
    status: &str,
    hits: &[greppy_store::VectorSearchHit],
) -> Result<()> {
    let retrieved = hits.len();
    semantic_vector_json_with_expand(
        store,
        project,
        cfg,
        graph_generation,
        total,
        retrieved,
        candidate_limit,
        freshness,
        status,
        hits,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn semantic_vector_json_with_expand(
    store: &greppy_store::Store,
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    total: i64,
    retrieved: usize,
    candidate_limit: Option<i64>,
    freshness: Option<&serde_json::Value>,
    status: &str,
    hits: &[greppy_store::VectorSearchHit],
    purposes: Option<&[SemanticVectorPurpose]>,
    expand: Option<&ExpandHandle>,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let rows = hits
        .iter()
        .map(|hit| semantic_vector_json_row(hit, vector_purpose_for_hit(purposes, hit), expand))
        .collect::<Vec<_>>();
    let shown = rows.len() as i64;
    let (retrieved, omitted, unranked_candidates, truncated) =
        semantic_vector_count_values(total, retrieved, rows.len());
    let mut v = serde_json::json!({
        "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
        "command": "semantic-search",
        "mode": "vector",
        "status": status,
        "project": project,
        "backend": "exact_cosine",
        "scope": "embeddinggemma_code_retrieval_current_generation",
        "model_id": cfg.model_id,
        "prompt_version": greppy_embed_native::PROMPT_VERSION,
        "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
        "graph_generation": graph_generation,
        "fresh": freshness
            .and_then(|v| v.get("fresh"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "candidate_limit": candidate_limit,
        "candidate_limit_env": ENV_VECTOR_EXACT_CANDIDATE_LIMIT,
        "candidate_total": total,
        "total_exact": total,
        "retrieved": retrieved,
        "shown": shown,
        "omitted": omitted,
        "unranked_candidates": unranked_candidates,
        "truncated": truncated,
        "hits": rows,
    });
    if let Some(expand) = expand {
        v["expand"] = expand.json_value();
        v["expand_id"] = serde_json::json!(&expand.id);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize vector semantic JSON: {e}")))?
    );
    Ok(())
}

pub(crate) fn semantic_vector_count_values(
    candidate_total: i64,
    retrieved: usize,
    shown: usize,
) -> (i64, i64, i64, bool) {
    let retrieved = i64::try_from(retrieved).unwrap_or(i64::MAX);
    let shown = i64::try_from(shown).unwrap_or(i64::MAX);
    let omitted = retrieved.saturating_sub(shown);
    let unranked_candidates = candidate_total.saturating_sub(retrieved);
    (
        retrieved,
        omitted,
        unranked_candidates,
        omitted > 0 || unranked_candidates > 0,
    )
}

pub(crate) fn semantic_provider_incomplete_json(
    project: &str,
    mode: &str,
    freshness: Option<&serde_json::Value>,
    incomplete_providers: &[serde_json::Value],
) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
            "command": "semantic-search",
            "mode": mode,
            "status": "skipped_incomplete_provider",
            "project": project,
            "fresh": freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": false,
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "total_exact": 0,
            "shown": 0,
            "omitted": 0,
            "truncated": false,
            "hits": [],
        }))
        .map_err(|e| Error::Invalid(format!("serialize semantic provider policy JSON: {e}")))?
    );
    Ok(())
}

/// Keep fallback searches tied to the user's actual words. Natural-language
/// glue is dropped so the grep pattern remains selective; when every word is
/// glue, preserve the original tokens rather than inventing a pattern.
pub(crate) fn semantic_fallback_tokens(query: &str) -> Vec<String> {
    fn is_search_glue(token: &str) -> bool {
        matches!(
            token.to_ascii_lowercase().as_str(),
            "a" | "an"
                | "and"
                | "are"
                | "as"
                | "at"
                | "be"
                | "by"
                | "code"
                | "find"
                | "for"
                | "from"
                | "how"
                | "in"
                | "is"
                | "it"
                | "of"
                | "on"
                | "or"
                | "search"
                | "that"
                | "the"
                | "this"
                | "to"
                | "what"
                | "where"
                | "which"
                | "with"
        )
    }

    let all = query
        .split(|character: char| !(character.is_alphanumeric() || character == '_'))
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut selected = all
        .iter()
        .filter(|token| token.len() >= 2 && !is_search_glue(token))
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        selected = all;
    }
    let mut seen = std::collections::HashSet::new();
    selected.retain(|token| seen.insert(token.to_ascii_lowercase()));
    selected.truncate(8);
    selected
}

pub(crate) fn semantic_fallback_commands(query: &str, paths: &[String], root: Option<&str>) -> Vec<String> {
    let tokens = semantic_fallback_tokens(query);
    let mut commands = Vec::new();
    if let Some(symbol_token) = tokens.last() {
        let mut command = format!("greppy search-symbols {}", shell_example_arg(symbol_token));
        for path in paths {
            command.push(' ');
            command.push_str(&shell_example_arg(path));
        }
        if let Some(root) = root {
            command.push_str(" --root ");
            command.push_str(&shell_example_arg(root));
        }
        commands.push(command);
    }

    let pattern = if tokens.is_empty() {
        query.to_string()
    } else {
        tokens.join("|")
    };
    let grep_targets = if paths.is_empty() {
        vec![root.unwrap_or(".").to_string()]
    } else if let Some(root) = root {
        paths
            .iter()
            .map(|path| {
                std::path::Path::new(root)
                    .join(path)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    } else {
        paths.to_vec()
    };
    let mut grep = format!("greppy grep -rnE {}", shell_example_arg(&pattern));
    for target in grep_targets {
        grep.push(' ');
        grep.push_str(&shell_example_arg(&target));
    }
    commands.push(grep);
    commands
}
