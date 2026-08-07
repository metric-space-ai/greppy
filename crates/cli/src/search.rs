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
            "command": "search",
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

fn search_all_nodes(store: &greppy_store::Store, project: &str) -> Result<Vec<greppy_store::Node>> {
    const PAGE: usize = 4096;
    let mut nodes = Vec::new();
    loop {
        let page = store.list_nodes(project, "", "", nodes.len(), PAGE)?;
        let done = page.len() < PAGE;
        nodes.extend(page);
        if done {
            break;
        }
    }
    nodes.retain(|node| !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name));
    Ok(nodes)
}

fn search_kind_matches(
    root_path: &std::path::Path,
    node: &greppy_store::Node,
    kind: Option<&str>,
) -> bool {
    let Some(kind) = kind else {
        return true;
    };
    let lines = nav_file_lines(root_path, &node.file_path);
    nav_kind_word(lines.as_ref(), node).eq_ignore_ascii_case(kind)
}

fn search_name_normalized(name: &str) -> String {
    name.chars()
        .filter(|character| *character != '_' && *character != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

fn search_identifier_words(name: &str) -> std::collections::BTreeSet<String> {
    greppy_store::fts::camel_split(name)
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect()
}

fn search_sort_name_rows(query: &str, nodes: &mut [greppy_store::Node]) {
    nodes.sort_by(|left, right| {
        (left.name != query)
            .cmp(&(right.name != query))
            .then_with(|| left.name.chars().count().cmp(&right.name.chars().count()))
            .then_with(|| left.file_path.cmp(&right.file_path))
            .then_with(|| left.start_line.cmp(&right.start_line))
            .then_with(|| left.name.cmp(&right.name))
    });
}

fn search_symbol_no_match_status(
    query: &str,
    path_filters: &QueryPathFilters,
    matches_outside_filter: usize,
) {
    println!("status: no_matches");
    if path_filters.is_empty() {
        println!("scope: indexed definitions in the repository");
        println!("message: no definition named `{query}`");
        println!(
            "next: search source text: greppy search-pattern {} --fixed",
            shell_example_arg(query)
        );
    } else {
        println!("scope: path filter {}", path_filters.shown());
        println!(
            "message: no definition named `{query}` under path filter: {}",
            path_filters.shown()
        );
        if matches_outside_filter > 0 {
            println!(
                "reason: {matches_outside_filter} matching definition(s) exist outside the path filter"
            );
        } else {
            println!("reason: no indexed definition matched inside this scope");
        }
        println!(
            "next: retry without the path filter: greppy search-symbol {}",
            shell_example_arg(query)
        );
    }
    println!("next: refresh definitions after source changes: greppy index .");
}

fn search_pattern_no_match_status(
    query: &str,
    fixed: bool,
    path_filters: &QueryPathFilters,
    matches_outside_filter: usize,
) {
    println!("status: no_matches");
    if path_filters.is_empty() {
        println!("scope: live source files in the repository");
        println!("message: no matches");
    } else {
        println!("scope: path filter {}", path_filters.shown());
        println!(
            "message: no matches under path filter: {}",
            path_filters.shown()
        );
        if matches_outside_filter > 0 {
            println!(
                "reason: {matches_outside_filter} source match(es) exist outside the path filter"
            );
        } else {
            println!("reason: no source match exists inside this scope");
        }
        let mode = if fixed { " --fixed" } else { "" };
        println!(
            "next: retry without the path filter: greppy search-pattern {}{mode}",
            shell_example_arg(query)
        );
    }
    println!(
        "next: search definition names: greppy search-symbol {}",
        shell_example_arg(query)
    );
    println!("next: refresh graph-backed definition filters after source changes: greppy index .");
}

fn semantic_no_match_status(query: &str, path_filters: &QueryPathFilters) {
    println!("status: no_matches");
    if path_filters.is_empty() {
        println!("scope: semantic definitions in the repository");
        println!("message: no matches");
    } else {
        println!("scope: path filter {}", path_filters.shown());
        println!(
            "message: no matches under path filter: {}",
            path_filters.shown()
        );
        println!(
            "next: retry without the path filter: greppy search {}",
            shell_example_arg(query)
        );
    }
    println!(
        "next: search source text: greppy search-pattern {} --fixed",
        shell_example_arg(query)
    );
    println!("next: refresh semantic definitions after source changes: greppy index .");
}

fn search_print_node_source(root_path: &std::path::Path, node: &greppy_store::Node) {
    if let Some(span) = read_span_with_meta(
        root_path,
        &node.file_path,
        node.start_line,
        node.end_line,
        usize::MAX,
        false,
    ) {
        print_search_source(&span.text);
    }
}

fn search_print_symbol_rows(root_path: &std::path::Path, nodes: &[greppy_store::Node], code: bool) {
    let mut source_cache: std::collections::HashMap<String, Option<Vec<String>>> =
        Default::default();
    for (index, node) in nodes.iter().enumerate() {
        let lines = source_cache
            .entry(node.file_path.clone())
            .or_insert_with(|| nav_file_lines(root_path, &node.file_path));
        let kind = nav_kind_word(lines.as_ref(), node);
        let test = nav_is_test(lines.as_ref(), node);
        print_search_row(
            &node.file_path,
            node.start_line,
            &nav_short_name(node),
            Some(SearchRowDetail::Kind(&kind)),
            test,
        );
        if code {
            search_print_node_source(root_path, node);
            if index + 1 < nodes.len() {
                println!();
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps search inputs explicit across the embedding fallback boundary"
)]
fn search_symbol_meaning_hits(
    store: &greppy_store::Store,
    project: &str,
    root: Option<&str>,
    root_path: &std::path::Path,
    query: &str,
    kind: Option<&str>,
    path_filters: &QueryPathFilters,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Vec<greppy_store::VectorSearchHit> {
    const MEANING_FLOOR: f32 = 0.45;
    let Ok(cfg) = embedding_config_for_required_use(embedding_args) else {
        return Vec::new();
    };
    let Ok(generation) = current_graph_generation(store, root) else {
        return Vec::new();
    };
    if !embedding_generation_complete(store, project, generation, &cfg.model_id) {
        return Vec::new();
    }
    let Ok(query_vector) = embed_query_cached(&cfg, root, query) else {
        return Vec::new();
    };
    let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
        project,
        &cfg.model_id,
        Some(generation),
        SEMANTIC_VECTOR_CANDIDATE_LIMIT,
    );
    scope.limit = SEMANTIC_VECTOR_CANDIDATE_LIMIT;
    let Ok(candidates) = greppy_search::vector_search_exact(store, &query_vector, &scope) else {
        return Vec::new();
    };
    dedupe_semantic_vector_hits(candidates, 8)
        .into_iter()
        .filter(|hit| hit.score >= MEANING_FLOOR)
        .filter(|hit| path_filters.matches(&hit.embedding.file_path))
        .filter(|hit| {
            hit.embedding
                .node_id
                .and_then(|id| store.get_node(id).ok().flatten())
                .is_some_and(|node| search_kind_matches(root_path, &node, kind))
        })
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the stable search-symbol CLI surface on the release branch"
)]
pub(crate) fn dispatch_search_symbols(
    query: Option<&str>,
    kind: Option<&str>,
    code: bool,
    all: bool,
    json: bool,
    paths: &[String],
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("search-symbol requires a name".into()));
    }
    let path_filters = prepare_query_path_filters(root, "search-symbol", q, paths)?;
    let mut store = open_default_store(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let root_path = resolve_root(root)?;
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
            println!("{}", indexed_stale_skip_message("search-symbol", freshness));
        }
        // A refreshing/drifting index is a TEMPORARY refusal — the same
        // situation search, plus and context report as retryable. Returning a
        // flat 1 told an agent "permanently not there" for a state that
        // resolves by itself, and a wrong answer is what follows.
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
                provider_incomplete_skip_message("search-symbol", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    if json {
        let fetch = if kind.is_some() {
            10_000
        } else {
            cli_result_limit(20)
        };
        let mut hits = greppy_search::search_symbols_in_project(&store, &project, q, fetch)?;
        hits.retain(|hit| {
            store
                .get_node(hit.node_id)
                .ok()
                .flatten()
                .is_some_and(|node| {
                    node.name.contains(q)
                        && path_filters.matches(&node.file_path)
                        && search_kind_matches(&root_path, &node, kind)
                })
        });
        let total_filtered = hits.len() as i64;
        hits.truncate(cli_result_limit_unless_all(20, all));
        let status = if hits.is_empty() { "no_matches" } else { "ok" };
        search_symbols_json(
            &store,
            q,
            &project,
            status,
            Some(&freshness),
            &hits,
            &path_filters,
            // The containment filter shapes `hits` on every run, so the total
            // must count the same set on every run — a count from before the
            // filter is a false number (measured: total_exact 2 beside 1 hit).
            Some(total_filtered),
        )?;
        return Ok(0);
    }

    let mut all_nodes = search_all_nodes(&store, &project)?;
    all_nodes.retain(|node| search_kind_matches(&root_path, node, kind));
    let matches_outside_filter = all_nodes
        .iter()
        .filter(|node| node.name.contains(q) && !path_filters.matches(&node.file_path))
        .count();
    let nodes = all_nodes
        .into_iter()
        .filter(|node| path_filters.matches(&node.file_path))
        .collect::<Vec<_>>();
    let mut contained = nodes
        .iter()
        .filter(|node| node.name.contains(q))
        .cloned()
        .collect::<Vec<_>>();
    search_sort_name_rows(q, &mut contained);
    if !contained.is_empty() {
        contained.truncate(cli_result_limit_unless_all(20, all));
        search_print_symbol_rows(&root_path, &contained, code);
        return Ok(0);
    }

    search_symbol_no_match_status(q, &path_filters, matches_outside_filter);

    let wanted_normalized = search_name_normalized(q);
    let mut similar = nodes
        .iter()
        .filter(|node| search_name_normalized(&node.name) == wanted_normalized)
        .cloned()
        .collect::<Vec<_>>();
    if similar.is_empty() {
        similar = nodes
            .iter()
            .filter(|node| {
                levenshtein(&node.name.to_ascii_lowercase(), &q.to_ascii_lowercase()) <= 2
            })
            .cloned()
            .collect::<Vec<_>>();
    }
    if similar.is_empty() {
        let wanted_words = search_identifier_words(q);
        let best = nodes
            .iter()
            .map(|node| {
                search_identifier_words(&node.name)
                    .intersection(&wanted_words)
                    .count()
            })
            .max()
            .unwrap_or(0);
        if best > 0 {
            similar = nodes
                .iter()
                .filter(|node| {
                    search_identifier_words(&node.name)
                        .intersection(&wanted_words)
                        .count()
                        == best
                })
                .cloned()
                .collect();
        }
    }
    if !similar.is_empty() {
        search_sort_name_rows(q, &mut similar);
        similar.truncate(cli_result_limit_unless_all(20, all));
        println!();
        println!("similar names:");
        search_print_symbol_rows(&root_path, &similar, code);
        return Ok(0);
    }

    let meaning = search_symbol_meaning_hits(
        &store,
        &project,
        root,
        &root_path,
        q,
        kind,
        &path_filters,
        embedding_args,
    );
    if !meaning.is_empty() {
        println!();
        println!("closest by meaning:");
        let purposes = semantic_vector_purposes(&store, root, &meaning, true)?;
        print_search_meaning_rows(&store, &root_path, &meaning, purposes.as_deref(), code)?;
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
            "command": "search-symbol",
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
            "next": if status == "no_matches" {
                vec![
                    format!("greppy search-pattern {} --fixed", shell_example_arg(query)),
                    "greppy index .".to_string(),
                ]
            } else {
                Vec::new()
            },
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

#[derive(Debug)]
struct SearchPatternRow {
    hit: SearchCodeMatchLine,
    node: Option<greppy_store::Node>,
    test: bool,
}

fn search_pattern_case_insensitive_hits(
    query: &str,
    root_path: &std::path::Path,
    fixed: bool,
) -> Result<Vec<greppy_search::CodeHit>> {
    let overrides = discover_overrides_from_env()?;
    let entries = greppy_discover::walk_with_policy_and_overrides(
        root_path,
        &greppy_discover::SkipPolicy::walk_default(),
        &overrides,
    )?;
    let paths = entries
        .into_iter()
        .map(|entry| entry.rel_path)
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let mut hits = Vec::new();
    for chunk in paths.chunks(128) {
        let mut command = std::process::Command::new("grep");
        command.args(["-H", "-n", "-I", "-i"]);
        command.arg(if fixed { "-F" } else { "-E" });
        command
            .arg("--")
            .arg(query)
            .args(chunk)
            .current_dir(root_path);
        let output = command
            .output()
            .map_err(|error| Error::io("spawn grep for case-insensitive search-pattern", error))?;
        if !output.status.success() && output.status.code() != Some(1) {
            return Err(Error::Invalid(format!(
                "grep source scan failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        hits.extend(
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(parse_grep_code_hit),
        );
    }
    Ok(hits)
}

fn search_pattern_rows(
    store: &greppy_store::Store,
    project: &str,
    root_path: &std::path::Path,
    hits: &[greppy_search::CodeHit],
    resolve_definitions: bool,
    kind: Option<&str>,
) -> Result<Vec<SearchPatternRow>> {
    let mut source_cache: std::collections::HashMap<String, Option<Vec<String>>> =
        Default::default();
    let mut rows = Vec::new();
    for hit in hits {
        let Some(match_line) = parse_search_code_match(hit) else {
            continue;
        };
        let node = if resolve_definitions {
            greppy_search::definition_at(store, Some(project), &match_line.file, match_line.line)?
                .and_then(|row| store.get_node(row.id).ok().flatten())
        } else {
            None
        };
        if kind.is_some()
            && node
                .as_ref()
                .is_none_or(|node| !search_kind_matches(root_path, node, kind))
        {
            continue;
        }
        let test = node.as_ref().is_some_and(|node| {
            let lines = source_cache
                .entry(node.file_path.clone())
                .or_insert_with(|| nav_file_lines(root_path, &node.file_path));
            nav_is_test(lines.as_ref(), node)
        });
        rows.push(SearchPatternRow {
            hit: match_line,
            node,
            test,
        });
    }
    let mut per_file: std::collections::BTreeMap<String, usize> = Default::default();
    for row in &rows {
        *per_file.entry(row.hit.file.clone()).or_insert(0) += 1;
    }
    rows.sort_by(|left, right| {
        per_file[&left.hit.file]
            .cmp(&per_file[&right.hit.file])
            .then_with(|| left.hit.file.cmp(&right.hit.file))
            .then_with(|| left.hit.line.cmp(&right.hit.line))
    });
    Ok(rows)
}

fn print_search_pattern_rows(rows: &[SearchPatternRow], code: bool, all: bool) {
    const FULL_LIMIT: usize = 25;
    const SUMMARY_ROWS: usize = 5;
    let mut per_file: std::collections::BTreeMap<&str, usize> = Default::default();
    for row in rows {
        *per_file.entry(&row.hit.file).or_insert(0) += 1;
    }
    let summarize = !all && rows.len() > FULL_LIMIT;
    if summarize {
        let mut spread = per_file.into_iter().collect::<Vec<_>>();
        spread.sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(right.0)));
        println!(
            "{} matches: {}",
            rows.len(),
            spread
                .into_iter()
                .map(|(file, count)| format!("{file} {count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!();
    }
    let default_shown = if summarize { SUMMARY_ROWS } else { rows.len() };
    let shown = default_shown.min(cli_result_limit_unless_all(default_shown, all));
    for (index, row) in rows.iter().take(shown).enumerate() {
        if let Some(node) = &row.node {
            print_search_row(
                &row.hit.file,
                row.hit.line,
                &nav_short_name(node),
                None,
                row.test,
            );
        } else {
            println!("{}:{}", row.hit.file, row.hit.line.max(1));
        }
        if code {
            println!("{}", row.hit.text);
            if index + 1 < shown {
                println!();
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the stable search-code CLI surface on the release branch"
)]
pub(crate) fn dispatch_search_code(
    query: Option<&str>,
    kind: Option<&str>,
    code: bool,
    all: bool,
    json: bool,
    fixed: bool,
    paths: &[String],
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid(
            "search-pattern requires a regular expression".into(),
        ));
    }
    let path_filters = prepare_query_path_filters(root, "search-pattern", q, paths)?;
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
    let mut all_hits = live_grep_code_hits_pattern(q, &root_path, fixed)?;
    let matches_outside_filter = all_hits
        .iter()
        .filter(|hit| {
            hit.location
                .rsplit_once(':')
                .is_some_and(|(file, _)| !path_filters.matches(file))
        })
        .count();
    // The path filter shapes the hit set BEFORE any count is taken — a count
    // from before the filter is a false number (the --kind discipline).
    all_hits.retain(|hit| {
        hit.location
            .rsplit_once(':')
            .is_some_and(|(file, _)| path_filters.matches(file))
    });

    if json {
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
            true,
            false,
            fixed,
            resolve_definitions,
        )?;
        return Ok(0);
    }

    let rows = search_pattern_rows(
        &store,
        &project,
        &root_path,
        &all_hits,
        resolve_definitions,
        kind,
    )?;
    if rows.is_empty() {
        search_pattern_no_match_status(q, fixed, &path_filters, matches_outside_filter);
        let mut insensitive = search_pattern_case_insensitive_hits(q, &root_path, fixed)?;
        insensitive.retain(|hit| {
            hit.location
                .rsplit_once(':')
                .is_some_and(|(file, _)| path_filters.matches(file))
        });
        if !insensitive.is_empty() {
            println!("case-insensitive: {} matches", insensitive.len());
        }
        return Ok(0);
    }
    print_search_pattern_rows(&rows, code, all);
    Ok(0)
}

fn search_sentence(raw: &str) -> Option<String> {
    let sentence = raw.trim().trim_end_matches('.').trim();
    if sentence.is_empty() {
        return None;
    }
    let mut chars = sentence.chars();
    let first = chars.next()?.to_lowercase().to_string();
    Some(format!("{first}{}", chars.as_str()))
}

fn print_search_meaning_rows(
    store: &greppy_store::Store,
    root_path: &std::path::Path,
    hits: &[greppy_store::VectorSearchHit],
    purposes: Option<&[SemanticVectorPurpose]>,
    code: bool,
) -> Result<()> {
    let mut source_cache: std::collections::HashMap<String, Option<Vec<String>>> =
        Default::default();
    for (index, hit) in hits.iter().enumerate() {
        let node = hit
            .embedding
            .node_id
            .and_then(|id| store.get_node(id).ok().flatten());
        let purpose = vector_purpose_for_hit(purposes, hit);
        let sentence = purpose
            .and_then(|purpose| purpose.bullets.first())
            .and_then(|sentence| search_sentence(sentence));
        if let Some(node) = node.as_ref() {
            let lines = source_cache
                .entry(node.file_path.clone())
                .or_insert_with(|| nav_file_lines(root_path, &node.file_path));
            let test = nav_is_test(lines.as_ref(), node);
            print_search_row(
                &node.file_path,
                node.start_line,
                &nav_short_name(node),
                sentence.as_deref().map(SearchRowDetail::Sentence),
                test,
            );
            if code {
                search_print_node_source(root_path, node);
            }
        } else {
            let name = hit
                .embedding
                .qualified_name
                .rsplit("::")
                .next()
                .unwrap_or(&hit.embedding.qualified_name);
            print_search_row(
                &hit.embedding.file_path,
                hit.embedding.start_line,
                name,
                sentence.as_deref().map(SearchRowDetail::Sentence),
                false,
            );
            if code {
                if let Some(span) = read_span_with_meta(
                    root_path,
                    &hit.embedding.file_path,
                    hit.embedding.start_line,
                    hit.embedding.end_line,
                    usize::MAX,
                    false,
                ) {
                    print_search_source(&span.text);
                }
            }
        }
        if code && index + 1 < hits.len() {
            println!();
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the stable semantic-search CLI surface on the release branch"
)]
pub(crate) fn dispatch_semantic(
    query: Option<&str>,
    paths: &[String],
    kind: Option<&str>,
    code: bool,
    _all: bool,
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid(
            "search requires a plain-English query".into(),
        ));
    }
    // Result purposes reach the Qwen daemon; overlap its model load with the
    // embedding query and vector search.
    prewarm_summary_daemon();
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
            if json {
                emit_semantic_backend_unavailable(
                    &project,
                    q,
                    paths,
                    root,
                    true,
                    "EmbeddingGemma assets could not be resolved; use one of the exact non-semantic fallbacks below.",
                )?;
            } else {
                println!("semantic index unavailable — embedding assets could not be resolved");
            }
            return Ok(1);
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
            }
            return Ok(1);
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
                let root_path = resolve_root(root)?;
                candidates.retain(|hit| {
                    path_filters.matches(&hit.embedding.file_path)
                        && kind.is_none_or(|kind| {
                            hit.embedding
                                .node_id
                                .and_then(|id| store.get_node(id).ok().flatten())
                                .is_some_and(|node| {
                                    search_kind_matches(&root_path, &node, Some(kind))
                                })
                        })
                });
                let result_limit = if json {
                    cli_result_limit(SEMANTIC_VECTOR_RESULT_LIMIT)
                } else {
                    cli_result_limit(8).min(8)
                };
                let hits = dedupe_semantic_vector_hits(candidates, result_limit);
                if json {
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
                    semantic_vector_json_with_expand(
                        &store,
                        &project,
                        &cfg,
                        generation,
                        total,
                        hits.len(),
                        candidate_limit,
                        Some(&freshness),
                        if hits.is_empty() { "no_matches" } else { "ok" },
                        &display_hits,
                        purposes.as_deref(),
                        expand.as_ref(),
                    )?;
                } else if hits.is_empty() {
                    semantic_no_match_status(q, &path_filters);
                    return Ok(0);
                } else {
                    let purposes = semantic_vector_purposes(&store, root, &hits, true)?;
                    print_search_meaning_rows(
                        &store,
                        &root_path,
                        &hits,
                        purposes.as_deref(),
                        code,
                    )?;
                }
                Ok(0)
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
    // One cache connection for every purpose span of this command.
    #[cfg(any(unix, windows))]
    let summary_cache = summary_runtime.as_ref().and_then(|_| {
        greppy_store::SummaryCache::open(&workspace_locator::store_dir(&root_path)).ok()
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
                bullets = summarize_source_cached(
                    cfg,
                    model_key,
                    summary_cache.as_ref(),
                    file_path,
                    &code,
                )
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
        "command": "search",
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
        "next": if status == "no_matches" {
            vec!["retry without --path/--kind", "greppy index ."]
        } else {
            Vec::new()
        },
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
            "command": "search",
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

pub(crate) fn semantic_fallback_commands(
    query: &str,
    paths: &[String],
    root: Option<&str>,
) -> Vec<String> {
    let tokens = semantic_fallback_tokens(query);
    let mut commands = Vec::new();
    if let Some(symbol_token) = tokens.last() {
        let mut command = format!("greppy search-symbol {}", shell_example_arg(symbol_token));
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
