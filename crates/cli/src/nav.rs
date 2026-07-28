//! The five navigation commands and the pieces only they use.
//!
//! `lib.rs` had grown to 26,400 lines and 528 top-level functions, with these
//! five scattered between line 7,473 and line 16,736 — so every change to one
//! of them meant working by line number in a file nobody can hold in view. The
//! move is mechanical: no behaviour changes, and the child module still reaches
//! every private helper in `lib.rs` through `use super::*`.

use super::*;

pub(crate) fn dispatch_impact(
    symbol: Option<&str>,
    paths: &[String],
    direction: &str,
    edge: Option<&str>,
    depth: usize,
    since: Option<&str>,
    base: Option<&str>,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    // The tree's per-node hints reach the Qwen daemon; start its async model
    // load now so the walk below overlaps the cold start.
    prewarm_summary_daemon();
    let path_filters = prepare_query_path_filters(root, "impact", symbol.unwrap_or(""), paths)?;
    let dir = match direction.to_ascii_lowercase().as_str() {
        "incoming" | "in" | "callers" => greppy_search::ReachDirection::Incoming,
        "outgoing" | "out" | "callees" => greppy_search::ReachDirection::Outgoing,
        other => {
            return Err(Error::Invalid(format!(
                "impact --direction must be 'incoming' or 'outgoing', got '{other}'"
            )));
        }
    };
    let direction_label = match dir {
        greppy_search::ReachDirection::Incoming => "incoming",
        greppy_search::ReachDirection::Outgoing => "outgoing",
    };
    let edge_upper = edge.map(|edge| edge.trim().to_ascii_uppercase());
    if since.is_some() && base.is_some() {
        return Err(Error::Invalid(
            "impact accepts only one git diff scope at a time".into(),
        ));
    }
    let edge_spec = impact_edge_spec(dir, edge_upper.as_deref());
    if since.is_some() || base.is_some() {
        if symbol.map(str::trim).is_some_and(|s| !s.is_empty()) {
            return Err(Error::Invalid(
                "impact accepts either a symbol or a git diff scope, not both".into(),
            ));
        }
        let scope = match (since, base) {
            (Some(rev), None) => DiffSearchScope::Since { rev },
            (None, Some(base)) => DiffSearchScope::Base { base },
            _ => {
                return Err(Error::Invalid(
                    "impact accepts exactly one git diff scope".into(),
                ));
            }
        };
        return dispatch_impact_diff_scope(
            scope,
            dir,
            direction_label,
            &edge_spec,
            depth,
            json,
            root,
        );
    }
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let query_symbol = symbol.unwrap_or("");
    let mut graph_gate_extra = serde_json::json!({
        "symbol": query_symbol,
        "symbol_found": false,
        "scope": "transitive",
        "direction": direction_label,
        "max_hops": depth,
        "all": false,
    });
    insert_impact_edge_meta(&mut graph_gate_extra, &edge_spec);
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "impact",
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
        "impact",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let Some(start) = resolve_symbol_id(&store, symbol)? else {
        if json {
            impact_counts_json(
                &store,
                root,
                query_symbol,
                &project,
                false,
                0,
                0,
                false,
                ImpactJsonMeta {
                    direction: direction_label,
                    edge_type: edge_spec.mode,
                    edge_types: &edge_spec.edge_types,
                    max_hops: depth,
                    scope: "transitive",
                },
                Vec::new(),
            )?;
            return Ok(1);
        }
        return content_fallback(
            &store,
            root,
            symbol.unwrap_or(""),
            "impact",
            &QueryPathFilters::default(),
        );
    };
    // Aggregate over every same-name start node (e.g. a Class and its Impl) and
    // union the reach, keeping the minimum hop count. For default incoming
    // impact, each traversal follows all reference edge types at every BFS
    // layer so mixed paths are preserved. Generous internal limit; the PRINTED
    // rows are capped separately so the footer can report the true transitive
    // total.
    let starts = resolve_symbol_nodes(&store, symbol)?;
    let starts = if starts.is_empty() {
        vec![start]
    } else {
        starts
    };
    let start_ids: std::collections::HashSet<i64> = starts.iter().copied().collect();
    let mut by_id: std::collections::HashMap<i64, greppy_search::ImpactNode> =
        std::collections::HashMap::new();
    for &sid in &starts {
        for n in greppy_search::impact_radius_any_edge_type(
            &store,
            sid,
            dir,
            &edge_spec.edge_types,
            depth,
            4096,
        )? {
            if start_ids.contains(&n.node.id) {
                continue; // a start node is not its own dependent
            }
            by_id
                .entry(n.node.id)
                .and_modify(|e| {
                    if n.hops < e.hops {
                        e.hops = n.hops;
                    }
                })
                .or_insert(n);
        }
    }
    let mut reached: Vec<greppy_search::ImpactNode> = by_id.into_values().collect();
    reached.sort_by(|a, b| a.hops.cmp(&b.hops).then_with(|| a.node.id.cmp(&b.node.id)));
    reached.retain(|n| path_filters.matches(&n.node.file_path));
    // "and the tests among it" — the reason to run impact before a change is
    // to learn which tests will speak up.
    let mut impact_tests = Vec::new();
    for step in &reached {
        if let Some(node) = store.get_node(step.node.id)? {
            if is_test_node(&node) {
                impact_tests.push(node_hit_json(&node));
            }
        }
    }
    if reached.is_empty() {
        if json {
            impact_counts_json(
                &store,
                root,
                query_symbol,
                &project,
                true,
                0,
                0,
                false,
                ImpactJsonMeta {
                    direction: direction_label,
                    edge_type: edge_spec.mode,
                    edge_types: &edge_spec.edge_types,
                    max_hops: depth,
                    scope: "transitive",
                },
                Vec::new(),
            )?;
            return Ok(0);
        }
        let what = match dir {
            greppy_search::ReachDirection::Incoming => "(nothing depends on it transitively)",
            greppy_search::ReachDirection::Outgoing => "(it reaches nothing transitively)",
        };
        println!("{what}");
        return Ok(0);
    }
    let total = reached.len();
    // `--all` lifts the print cap so the full transitive set is inspectable
    // in one call (the footer's "T total" was previously unreachable — clap
    // rejected --all — forcing a 28-round reconcile, r061).
    let shown = total.min(cli_result_limit_unless_all(NAV_LIMIT, all));
    // Informative sampling (r071/r074/r075 forensics): when the print cap
    // truncates, the FIRST `shown` rows are the answer most agents run with —
    // so rank named definitions before `__file__` file anchors and product
    // code before tests, THEN by hop, instead of letting a wall of hop-2
    // test-file anchors crowd named callers out of the sample. Rank beats
    // hop deliberately: every printed row still carries its `hop N` label,
    // but a named hop-3 caller answers a blast-radius question while a
    // test-file anchor rarely does. Stable sort: ordering within a rank
    // class, the true total, the footer, and `--all` output are unchanged.
    if shown < total {
        reached.sort_by(|a, b| {
            (nav_sample_rank(&a.node.file_path, &a.node.name), a.hops)
                .cmp(&(nav_sample_rank(&b.node.file_path, &b.node.name), b.hops))
        });
    }
    let expand = if !all {
        let mut nodes = Vec::new();
        for n in &reached {
            if let Some(node) = store.get_node(n.node.id)? {
                nodes.push((n.hops, display_row_name(&n.node), node));
            }
        }
        let rows = nodes
            .iter()
            .map(|(hops, name, node)| ExpandEvidenceNode {
                title: format!("hop {hops} {name}"),
                node,
                site_lines: Vec::new(),
                extra_json: serde_json::json!({"hops": hops}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(&store, root, &project, "impact", query_symbol, total, &rows)
    } else {
        None
    };
    if json {
        let hits = reached[..shown]
            .iter()
            .map(|n| {
                serde_json::json!({
                    "hops": n.hops,
                    "qualified_name": &n.node.qualified_name,
                    "label": &n.node.label,
                    "file": &n.node.file_path,
                    "line": n.node.start_line,
                    "file_path": &n.node.file_path,
                    "start_line": n.node.start_line,
                    "end_line": n.node.end_line,
                })
            })
            .collect();
        impact_counts_json_with_expand(
            &store,
            root,
            query_symbol,
            &project,
            true,
            total,
            shown,
            all,
            ImpactJsonMeta {
                direction: direction_label,
                edge_type: edge_spec.mode,
                edge_types: &edge_spec.edge_types,
                max_hops: depth,
                scope: "transitive",
            },
            hits,
            impact_tests,
            expand.as_ref(),
        )?;
        return Ok(0);
    }
    // A flat list of hops throws away the one thing `impact` has that six
    // `who-calls` calls do not: which route leads where. At `hop 3 dispatch_edit`
    // nobody can tell whether the path runs through `data_set` or `data_delete`.
    // The walk below keeps the edge a node was discovered on, so the answer is
    // the tree it actually is.
    let repo_root = resolve_root(root)?;
    let mut children: std::collections::HashMap<i64, Vec<i64>> = Default::default();
    let mut nodes: std::collections::HashMap<i64, greppy_store::Node> = Default::default();
    let mut seen: std::collections::HashSet<i64> = starts.iter().copied().collect();
    // Breadth first, so a node attaches to the shallowest parent that reaches
    // it. Depth first would hand every shared node to whichever branch happened
    // to run first, and the tree would claim a route the code does not take.
    let mut frontier: std::collections::VecDeque<(i64, usize)> =
        starts.iter().map(|id| (*id, 0)).collect();
    while let Some((id, hops)) = frontier.pop_front() {
        if hops >= depth {
            continue;
        }
        for edge_type in &edge_spec.edge_types {
            let steps = match dir {
                greppy_search::ReachDirection::Incoming => {
                    store.incoming_edges(id, Some(edge_type), 1024)?
                }
                greppy_search::ReachDirection::Outgoing => {
                    store.outgoing_edges(id, Some(edge_type), 1024)?
                }
            };
            for edge in steps {
                let next = match dir {
                    greppy_search::ReachDirection::Incoming => edge.source_id,
                    greppy_search::ReachDirection::Outgoing => edge.target_id,
                };
                let Some(node) = store.get_node(next)? else {
                    continue;
                };
                if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name)
                    || !path_filters.matches(&node.file_path)
                {
                    continue;
                }
                // The edge is recorded even when the node was already reached on
                // another branch: that second route is real and the tree says so
                // with `(above)`. Only the expansion happens once.
                children.entry(id).or_default().push(next);
                nodes.insert(next, node);
                if seen.insert(next) {
                    frontier.push_back((next, hops + 1));
                }
            }
        }
    }
    if nodes.is_empty() {
        println!("nothing reached");
        return Ok(0);
    }
    // The queried symbol is not part of its own impact: the tree starts at what
    // it reaches, so the roots are the children of the start nodes.
    let roots: Vec<i64> = starts
        .iter()
        .filter_map(|id| children.get(id))
        .flatten()
        .copied()
        .collect();
    let mut sources: std::collections::HashMap<String, Option<Vec<String>>> = Default::default();
    print_impact_tree(
        &repo_root,
        &children,
        &nodes,
        &roots,
        &mut sources,
        &mut std::collections::HashSet::new(),
        0,
    );
    Ok(0)
}

/// One node per line, its children indented by the step. A node already printed
/// is repeated as a bare `(above)` row: cycles terminate without a depth limit,
/// and the CLI spine that every symbol in a repository reaches is written out
/// once instead of once per branch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn print_impact_tree(
    repo_root: &std::path::Path,
    children: &std::collections::HashMap<i64, Vec<i64>>,
    nodes: &std::collections::HashMap<i64, greppy_store::Node>,
    ids: &[i64],
    sources: &mut std::collections::HashMap<String, Option<Vec<String>>>,
    printed: &mut std::collections::HashSet<i64>,
    depth: usize,
) {
    let mut ids: Vec<i64> = ids.to_vec();
    ids.sort_by_key(|id| {
        nodes
            .get(id)
            .map(|n| (n.file_path.clone(), n.start_line))
            .unwrap_or_default()
    });
    for id in ids {
        let Some(node) = nodes.get(&id) else {
            continue;
        };
        let indent = "  ".repeat(depth);
        let name = nav_short_name(node);
        let address = format!("{}:{}", node.file_path, node.start_line.max(1));
        if !printed.insert(id) {
            println!("{indent}{address}  {name}  (above)");
            continue;
        }
        let lines = sources
            .entry(node.file_path.clone())
            .or_insert_with(|| nav_file_lines(repo_root, &node.file_path))
            .clone();
        if nav_is_test(lines.as_ref(), node) {
            // A test's name states what it checks; a sentence would restate it.
            println!("{indent}{address}  {name}  test");
        } else {
            match impact_node_sentence(repo_root, node) {
                Some(sentence) => println!("{indent}{address}  {name} — {sentence}"),
                None => println!("{indent}{address}  {name}"),
            }
        }
        if let Some(next) = children.get(&id) {
            print_impact_tree(
                repo_root,
                children,
                nodes,
                next,
                sources,
                printed,
                depth + 1,
            );
        }
    }
}

/// The node's navigation hint, lowercased into the line. Without it the tree is
/// a map with no labels: the agent would have to ask what every name means.
pub(crate) fn impact_node_sentence(
    repo_root: &std::path::Path,
    node: &greppy_store::Node,
) -> Option<String> {
    let span = read_span_with_meta(
        repo_root,
        &node.file_path,
        node.start_line,
        node.end_line,
        CONTEXT_SPAN_CAP,
        false,
    )?;
    let sentence = summarize_definition_span(repo_root, &node.file_path, &span.text)?
        .into_iter()
        .next()?;
    let sentence = sentence.trim().to_string();
    if sentence.is_empty() {
        return None;
    }
    let sentence = sentence.trim_end_matches('.');
    let mut chars = sentence.chars();
    let first = chars.next()?.to_lowercase().to_string();
    Some(format!("{first}{}", chars.as_str()))
}

pub(crate) fn dispatch_impact_diff_scope(
    scope: DiffSearchScope<'_>,
    direction: greppy_search::ReachDirection,
    direction_label: &str,
    edge_spec: &ImpactEdgeSpec<'_>,
    depth: usize,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let project = project_for(root)?;
    // Never compute impact from a stale graph; trigger the bounded refresh
    // policy and return a stable refusal payload.
    if let FreshnessServe::Refuse(freshness) = freshness_serve_decision(&store, root, &project) {
        let refusal_exit = freshness_refusal_exit(&freshness);
        if json {
            // Impact reports only real code providers (excludes .stderr/.snap).
            let incomplete_providers = code_incomplete_provider_json(&store, &project)?;
            let mut v = serde_json::json!({
                "command": "impact",
                "status": "skipped_stale_index",
                "project": project,
                "fresh": false,
                "freshness": freshness,
                "provider_complete": incomplete_providers.is_empty(),
                "incomplete_provider_count": incomplete_providers.len(),
                "incomplete_providers": incomplete_providers,
                "scope": "diff",
                "direction": direction_label,
                "max_hops": depth,
                "source_total": 0,
                "total_exact": 0,
                "shown": 0,
                "hits": [],
            });
            insert_impact_edge_meta(&mut v, edge_spec);
            println!(
                "{}",
                serde_json::to_string_pretty(&v)
                    .map_err(|e| Error::Invalid(format!("serialize impact stale JSON: {e}")))?
            );
        } else {
            println!("{}", indexed_stale_skip_message("impact diff", &freshness));
        }
        return Ok(refusal_exit);
    }
    let mut gate_extra = serde_json::json!({
        "scope": "diff",
        "direction": direction_label,
        "max_hops": depth,
        "source_total": 0,
        "source_shown": 0,
        "source_omitted": 0,
        "all": false,
    });
    insert_impact_edge_meta(&mut gate_extra, edge_spec);
    if let Some(code) =
        provider_policy_graph_gate(&store, root, &project, "impact", json, gate_extra, "hits")?
    {
        return Ok(code);
    }

    let root_path = resolve_root(root)?;
    let spec = git_diff_search_spec(&root_path, scope)?;
    let diff_base = spec.merge_base.as_deref().unwrap_or(&spec.diff_rev);
    let changed_lines = git_diff_changed_lines(&root_path, diff_base, "impact")?;
    let sources = diff_impact_sources(&store, &project, &changed_lines)?;
    let source_shown = sources.len().min(NAV_LIMIT);
    let hits = diff_impact_hits(&store, &sources, direction, &edge_spec.edge_types, depth)?;
    let shown = hits.len().min(NAV_LIMIT);

    if json {
        let source_rows = sources[..source_shown]
            .iter()
            .map(|source| {
                serde_json::json!({
                    "qualified_name": &source.row.qualified_name,
                    "label": &source.row.label,
                    "file_path": &source.row.file_path,
                    "start_line": source.row.start_line,
                    "end_line": source.row.end_line,
                })
            })
            .collect::<Vec<_>>();
        let hit_rows = hits[..shown]
            .iter()
            .map(|hit| {
                let sources = hit
                    .sources
                    .iter()
                    .take(8)
                    .map(|source| {
                        serde_json::json!({
                            "qualified_name": &source.qualified_name,
                            "file_path": &source.file_path,
                            "start_line": source.start_line,
                            "end_line": source.end_line,
                        })
                    })
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "hops": hit.hops,
                    "qualified_name": &hit.node.qualified_name,
                    "label": &hit.node.label,
                    "file_path": &hit.node.file_path,
                    "start_line": hit.node.start_line,
                    "end_line": hit.node.end_line,
                    "source_count": hit.sources.len(),
                    "sources": sources,
                })
            })
            .collect::<Vec<_>>();
        impact_diff_counts_json(
            &store,
            root,
            &project,
            &spec,
            direction_label,
            edge_spec.mode,
            &edge_spec.edge_types,
            depth,
            sources.len(),
            source_shown,
            hits.len(),
            shown,
            hit_rows,
            source_rows,
        )?;
        return Ok(if sources.is_empty() { 1 } else { 0 });
    }

    if sources.is_empty() {
        println!("(no indexed definitions touched by diff)");
        return Ok(1);
    }
    println!(
        "diff sources: {} shown of {} total",
        source_shown,
        sources.len()
    );
    for source in &sources[..source_shown] {
        println!(
            "source {} {}",
            display_row_name(&source.row),
            line_span(
                &source.row.file_path,
                source.row.start_line,
                source.row.end_line
            )
        );
    }
    if hits.is_empty() {
        println!("(no transitive impact from diff sources)");
        return Ok(0);
    }
    for hit in &hits[..shown] {
        let source_names = hit
            .sources
            .iter()
            .take(3)
            .map(display_row_name)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "hop {} {} {} sources={}",
            hit.hops,
            display_row_name(&hit.node),
            line_span(&hit.node.file_path, hit.node.start_line, hit.node.end_line),
            source_names
        );
    }
    print_nav_more_footer(hits.len(), shown);
    Ok(0)
}

pub(crate) fn dispatch_brief(
    symbol: Option<&str>,
    paths: &[String],
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    // brief always summarizes its definition span; overlap the model load
    // with resolution and store open.
    prewarm_summary_daemon();
    let query_symbol = symbol.unwrap_or("");
    let path_filters = prepare_query_path_filters(root, "brief", query_symbol, paths)?;
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "brief",
        json,
        serde_json::json!({"schema_version": BRIEF_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "brief",
        json,
        serde_json::json!({"schema_version": BRIEF_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    // `brief` is graph-first and must remain useful when EmbeddingGemma cannot
    // be resolved. Probe configuration only (never load the model), surface the
    // degradation, then continue with definition/callers/callees unchanged.
    let semantic_backend_unavailable = embedding_config_for_required_use(EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    })
    .err()
    .filter(embedding_asset_missing_error)
    .map(|error| error.to_string());
    if !json && semantic_backend_unavailable.is_some() {
        println!(
            "semantic backend unavailable (asset missing) — brief continuing with graph-only definition/callers/callees"
        );
    }
    let targets = resolve_symbol_nodes(&store, symbol)?;
    if targets.is_empty() {
        if json {
            let miss = symbol_miss_json(&store, &project, query_symbol);
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema_version": BRIEF_JSON_SCHEMA_VERSION,
                    "command": "brief",
                    "status": "not_found",
                    "project": project,
                    "query": query_symbol,
                    "semantic_backend": brief_semantic_backend_json(
                        semantic_backend_unavailable.as_deref()
                    ),
                    "suggestions": miss["suggestions"].clone(),
                    "next": miss["next"].clone(),
                    "definitions": [],
                    "callers": [],
                    "references": [],
                    "calls": [],
                    "expand_id": serde_json::Value::Null,
                }))
                .map_err(|e| Error::Invalid(format!("serialize brief JSON: {e}")))?
            );
            return Ok(1);
        }
        return content_fallback(&store, root, symbol.unwrap_or(""), "brief", &path_filters);
    }
    let root_path = resolve_root(root)?;
    if json {
        return dispatch_brief_json(
            &store,
            &project,
            query_symbol,
            &targets,
            &root_path,
            BriefJsonContext {
                root,
                path_filters: &path_filters,
                semantic_backend_unavailable: semantic_backend_unavailable.as_deref(),
            },
        );
    }
    let mut evidence_nodes: Vec<(String, greppy_store::Node, serde_json::Value)> = Vec::new();

    // Definition(s) + source span.
    let mut seen_def = std::collections::BTreeSet::new();
    for id in &targets {
        if let Some(n) = store.get_node(*id)? {
            if path_filters.matches(&n.file_path) && seen_def.insert(n.id) {
                evidence_nodes.push((
                    format!("definition {}", display_node_name(&n)),
                    n.clone(),
                    serde_json::json!({"section": "definition"}),
                ));
                let span = read_span_with_meta(
                    &root_path,
                    &n.file_path,
                    n.start_line,
                    n.end_line,
                    CONTEXT_SPAN_CAP,
                    false,
                );
                let header_end_line = span
                    .as_ref()
                    .map(|span| span.end_line)
                    .unwrap_or(n.end_line);
                println!(
                    "== {} ({}:{}-{}) ==",
                    display_node_name(&n),
                    n.file_path,
                    n.start_line,
                    header_end_line
                );
                if let Some(span) = span {
                    if let Some(summary) =
                        summarize_definition_span(&root_path, &n.file_path, &span.text)
                    {
                        for bullet in summary {
                            println!("  - {bullet}");
                        }
                        if !span.text.is_empty() {
                            println!();
                        }
                    }
                    print_code_span_text(&span.text);
                }
            }
        }
    }

    let mut callers = incoming_call_nodes_for_targets(&store, &targets)?;
    callers.retain(|node| path_filters.matches(&node.file_path));
    let cshown = callers.len().min(cli_result_limit(BRIEF_LIMIT));
    println!("\n-- CALLERS ({}) --", callers.len());
    for n in &callers[..cshown] {
        evidence_nodes.push((
            format!("caller {}", display_node_name(n)),
            n.clone(),
            serde_json::json!({"section": "callers"}),
        ));
        println!("  {} {}", display_node_name(n), node_line_span(n));
    }
    print_nav_more_footer(callers.len(), cshown);

    if targets_include_non_callable(&store, &targets)? {
        let mut refs = greppy_search::find_references_to_any(
            &store,
            &targets,
            greppy_search::MAX_REACH_RESULTS,
        )?;
        refs.retain(|reference| path_filters.matches(&reference.node.file_path));
        let total = refs.len();
        refs.truncate(cli_result_limit(BRIEF_LIMIT));
        println!("\n-- REFERENCES ({}) --", total);
        for r in &refs {
            if let Some(node) = store.get_node(r.node.id)? {
                evidence_nodes.push((
                    format!("reference {} {}", r.edge_type, display_node_name(&node)),
                    node,
                    serde_json::json!({"section": "references", "edge_type": r.edge_type}),
                ));
            }
            println!(
                "  {} {} {}",
                r.edge_type,
                display_row_name(&r.node),
                line_span(&r.node.file_path, r.node.start_line, r.node.end_line)
            );
        }
        print_nav_more_footer(total, refs.len());
    }

    // Direct callees (outgoing CALLS).
    let mut callees: std::collections::BTreeMap<i64, greppy_store::Node> =
        std::collections::BTreeMap::new();
    let callee_sources = callee_source_ids_for_symbols(&store, &project, &targets)?;
    for id in &callee_sources {
        for step in greppy_search::callees_of(&store, *id)? {
            if let Some(n) = step.node {
                callees.entry(step.node_id).or_insert(n);
            }
        }
    }
    callees.retain(|_, node| path_filters.matches(&node.file_path));
    let eshown = callees.len().min(cli_result_limit(BRIEF_LIMIT));
    println!("\n-- CALLS ({}) --", callees.len());
    for n in callees.values().take(eshown) {
        evidence_nodes.push((
            format!("callee {}", display_node_name(n)),
            n.clone(),
            serde_json::json!({"section": "calls"}),
        ));
        println!("  {} {}", display_node_name(n), node_line_span(n));
    }
    print_nav_more_footer(callees.len(), eshown);
    if evidence_nodes.is_empty() && !path_filters.is_empty() {
        println!(
            "\n(no brief results under path filter: {})",
            path_filters.shown()
        );
    }
    let evidence_rows = evidence_nodes
        .iter()
        .map(|(title, node, extra_json)| ExpandEvidenceNode {
            title: title.clone(),
            node,
            site_lines: Vec::new(),
            extra_json: extra_json.clone(),
        })
        .collect::<Vec<_>>();
    if let Some(expand) = insert_nav_expand_pack(
        &store,
        root,
        &project,
        "brief",
        query_symbol,
        evidence_rows.len(),
        &evidence_rows,
    ) {
        println!("{}", expand.text_line());
    }
    Ok(0)
}

pub(crate) fn dispatch_brief_json(
    store: &greppy_store::Store,
    project: &str,
    query_symbol: &str,
    targets: &[i64],
    root_path: &std::path::Path,
    context: BriefJsonContext<'_>,
) -> Result<i32> {
    let root = context.root;
    let path_filters = context.path_filters;
    let semantic_backend_unavailable = context.semantic_backend_unavailable;
    let mut evidence_nodes: Vec<(String, greppy_store::Node, serde_json::Value)> = Vec::new();
    let mut definitions = Vec::new();
    let mut seen_def = std::collections::BTreeSet::new();
    for id in targets {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if !path_filters.matches(&node.file_path) || !seen_def.insert(node.id) {
            continue;
        }
        let span = read_span_with_meta(
            root_path,
            &node.file_path,
            node.start_line,
            node.end_line,
            CONTEXT_SPAN_CAP,
            false,
        );
        let source = span.as_ref().map(|span| span.text.as_str()).unwrap_or("");
        let end_line = span
            .as_ref()
            .map(|span| span.end_line)
            .unwrap_or(node.end_line);
        let signature = node
            .properties
            .get("source_signature")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| semantic_signature_from_span(source));
        let summary =
            summarize_definition_span(root_path, &node.file_path, source).unwrap_or_default();
        let summary_prompt_version = if summary.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::json!(greppy_qwen35_native::PROMPT_VERSION)
        };
        definitions.push(serde_json::json!({
            "qualified_name": &node.qualified_name,
            "name": display_node_name(&node),
            "label": &node.label,
            "file_path": &node.file_path,
            "start_line": node.start_line,
            "end_line": end_line,
            "signature": signature,
            "summary": summary,
            "summary_prompt_version": summary_prompt_version,
            "source": source,
        }));
        evidence_nodes.push((
            format!("definition {}", display_node_name(&node)),
            node,
            serde_json::json!({"section": "definition"}),
        ));
    }

    let mut callers = incoming_call_nodes_for_targets(store, targets)?;
    callers.retain(|node| path_filters.matches(&node.file_path));
    let callers_json = callers.iter().map(node_hit_json).collect::<Vec<_>>();
    for node in &callers {
        evidence_nodes.push((
            format!("caller {}", display_node_name(node)),
            node.clone(),
            serde_json::json!({"section": "callers"}),
        ));
    }

    let mut references_json = Vec::new();
    if targets_include_non_callable(store, targets)? {
        for reference in
            greppy_search::find_references_to_any(store, targets, greppy_search::MAX_REACH_RESULTS)?
        {
            if !path_filters.matches(&reference.node.file_path) {
                continue;
            }
            if references_json.len() == cli_result_limit(BRIEF_LIMIT) {
                break;
            }
            references_json.push(serde_json::json!({
                "edge_type": &reference.edge_type,
                "qualified_name": &reference.node.qualified_name,
                "file_path": &reference.node.file_path,
                "start_line": reference.node.start_line,
                "end_line": reference.node.end_line,
            }));
            if let Some(node) = store.get_node(reference.node.id)? {
                evidence_nodes.push((
                    format!(
                        "reference {} {}",
                        reference.edge_type,
                        display_node_name(&node)
                    ),
                    node,
                    serde_json::json!({
                        "section": "references",
                        "edge_type": reference.edge_type,
                    }),
                ));
            }
        }
    }

    let mut callees = std::collections::BTreeMap::<i64, greppy_store::Node>::new();
    for id in callee_source_ids_for_symbols(store, project, targets)? {
        for step in greppy_search::callees_of(store, id)? {
            if let Some(node) = step.node {
                callees.entry(step.node_id).or_insert(node);
            }
        }
    }
    callees.retain(|_, node| path_filters.matches(&node.file_path));
    let calls_json = callees.values().map(node_hit_json).collect::<Vec<_>>();
    for node in callees.values() {
        evidence_nodes.push((
            format!("callee {}", display_node_name(node)),
            node.clone(),
            serde_json::json!({"section": "calls"}),
        ));
    }

    let evidence_rows = evidence_nodes
        .iter()
        .map(|(title, node, extra_json)| ExpandEvidenceNode {
            title: title.clone(),
            node,
            site_lines: Vec::new(),
            extra_json: extra_json.clone(),
        })
        .collect::<Vec<_>>();
    let expand = insert_nav_expand_pack(
        store,
        root,
        project,
        "brief",
        query_symbol,
        evidence_rows.len(),
        &evidence_rows,
    );
    let freshness = nav_freshness_json(store, root, project);
    let mut output = serde_json::json!({
        "schema_version": BRIEF_JSON_SCHEMA_VERSION,
        "command": "brief",
        "status": "ok",
        "project": project,
        "query": query_symbol,
        "path_filters": path_filters.json_value(),
        "freshness": freshness,
        "semantic_backend": brief_semantic_backend_json(semantic_backend_unavailable),
        "definitions": definitions,
        "callers": callers_json,
        "references": references_json,
        "calls": calls_json,
        "expand_id": serde_json::Value::Null,
    });
    if let Some(expand) = expand {
        output["expand_id"] = serde_json::json!(&expand.id);
        output["expand"] = expand.json_value();
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|e| Error::Invalid(format!("serialize brief JSON: {e}")))?
    );
    Ok(0)
}

/// One answer line of `who-calls` / `callees`: an address, the symbol, and
/// whether it lives in a test. The path appears once, inside the address — the
/// name never repeats it, and no kind prefix is printed, because the command
/// has already said which relation this is.
pub(crate) struct NavAnswerRow {
    file: String,
    /// What the agent acts on: the call site for `who-calls`, the definition
    /// for `callees`.
    line: u32,
    /// What `--code` prints. Collapses to `line` when it is a single line.
    span: (u32, u32),
    name: String,
    test: bool,
}

/// The symbol without the file path in front of it. A qualified name carries
/// the file and usually a kind segment — `edit-src/data.rs::Function::data_set`
/// — and both are noise beside an address that already names the file.
pub(crate) fn nav_short_name(node: &greppy_store::Node) -> String {
    let qualified = node.qualified_name.as_str();
    let tail = qualified
        .split_once("::")
        .map(|(_, rest)| rest)
        .unwrap_or(qualified);
    let name = tail
        .split_once("::")
        .and_then(|(head, rest)| {
            matches!(
                head,
                "Function"
                    | "Method"
                    | "Class"
                    | "Struct"
                    | "Enum"
                    | "Trait"
                    | "Interface"
                    | "EnumVariant"
                    | "Field"
                    | "Variable"
                    | "Constant"
                    | "Module"
            )
            .then_some(rest)
        })
        .unwrap_or(tail);
    if name.is_empty() {
        node.name.clone()
    } else {
        name.to_string()
    }
}

/// Net `(`/`[` nesting a line leaves open. Braces are deliberately not counted:
/// a call inside `if let Some(x) = f( … ) {` balances its parentheses on the
/// very line that opens the block, and counting the brace would drag the whole
/// block in. Text inside double quotes and after `//` is ignored; a lone `'`
/// is left alone so Rust lifetimes do not read as an unterminated string.
pub(crate) fn nav_bracket_delta(text: &str) -> i32 {
    let chars: Vec<char> = text.chars().collect();
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (index, c) in chars.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if *c == '\\' {
                escaped = true;
            } else if *c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '/' if chars.get(index + 1) == Some(&'/') => break,
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// The statement containing `line`, as it stands in the file. Follows the
/// line's own bracket nesting downwards and keeps a trailing `.`/`?`
/// continuation with it, so a call a formatter broke across ten lines arrives
/// whole. It never reformats and never rewrites: the caller prints exactly the
/// range returned here, so the address above the source is always true.
pub(crate) fn nav_statement_span(lines: &[String], line: u32) -> (u32, u32) {
    const MAX_LINES: usize = 60;
    let start = (line as usize).saturating_sub(1);
    if start >= lines.len() {
        return (line, line);
    }
    let mut depth = 0;
    let mut end = start;
    for (offset, text) in lines[start..].iter().enumerate().take(MAX_LINES) {
        depth += nav_bracket_delta(text);
        end = start + offset;
        if depth > 0 {
            continue;
        }
        match lines.get(end + 1).map(|next| next.trim_start()) {
            Some(next) if next.starts_with('.') || next.starts_with('?') => depth = 0,
            _ => break,
        }
    }
    ((start + 1) as u32, (end + 1) as u32)
}

/// Whether a node is a test. `is_test_node` decides on the path and the name,
/// which misses the most common Rust shape by far: a `#[test]` function with an
/// ordinary name inside an inline `mod tests`. Six of `data_set`'s seven callers
/// are exactly that, and calling them production code overstates the blast
/// radius of a change. So the attribute directly above the definition is
/// consulted too — the same line a human reads to decide.
pub(crate) fn nav_is_test(lines: Option<&Vec<String>>, node: &greppy_store::Node) -> bool {
    if is_test_node(node) {
        return true;
    }
    let Some(lines) = lines else {
        return false;
    };
    let definition = node.start_line.max(1) as usize;
    let first = definition.saturating_sub(4);
    lines
        .get(first..definition.saturating_sub(1))
        .into_iter()
        .flatten()
        .map(|line| line.trim())
        .any(|line| {
            line.starts_with("#[test]")
                || line.starts_with("#[tokio::test")
                || line.starts_with("#[rstest")
                || line.starts_with("#[test_case")
                || line.starts_with("@Test")
                || line.starts_with("@pytest.mark")
        })
}

pub(crate) fn nav_file_lines(root: &std::path::Path, file: &str) -> Option<Vec<String>> {
    std::fs::read_to_string(root.join(file))
        .ok()
        .map(|text| text.lines().map(str::to_string).collect())
}

/// What a reader would call this definition. The index label is a poor source:
/// it says `Class` for a Rust struct, so the refusal would name a thing the
/// language does not have. The declaring keyword in the source says it exactly,
/// in the words of the language the agent is reading, and the label is only the
/// fallback for a file that cannot be read.
pub(crate) fn nav_kind_word(lines: Option<&Vec<String>>, node: &greppy_store::Node) -> String {
    const KEYWORDS: [&str; 14] = [
        "struct",
        "enum",
        "trait",
        "class",
        "interface",
        "record",
        "union",
        "type",
        "const",
        "static",
        "protocol",
        "object",
        "module",
        "package",
    ];
    if let Some(lines) = lines {
        if let Some(line) = lines.get((node.start_line.max(1) as usize).saturating_sub(1)) {
            for word in line.split_whitespace() {
                let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                if KEYWORDS.contains(&word) {
                    return word.to_string();
                }
            }
        }
    }
    match node.label.as_str() {
        "EnumVariant" => "enum variant".to_string(),
        other => other.to_ascii_lowercase(),
    }
}

/// Which kinds a direction can answer for. Incoming navigation follows both
/// calls and usages, so every real definition can be referenced. Outgoing
/// navigation still requires a definition that can hold code. Synthetic file
/// anchors are handled separately by `nav_refuse_non_callable`.
#[derive(Clone, Copy)]
pub(crate) enum NavDirection {
    Incoming,
    Outgoing,
}

impl NavDirection {
    fn answerable(self, label: &str) -> bool {
        match self {
            NavDirection::Incoming => true,
            NavDirection::Outgoing => !matches!(
                label,
                "Field" | "Variable" | "Constant" | "EnumVariant" | "Property"
            ),
        }
    }
}

/// A definition the direction cannot answer for. Incoming references apply to
/// every real definition; outgoing calls still refuse definitions that cannot
/// hold code. Synthetic file anchors are not referenceable definitions.
pub(crate) fn nav_refuse_non_callable(
    store: &greppy_store::Store,
    repo_root: &std::path::Path,
    target: &str,
    ids: &[i64],
    direction: NavDirection,
) -> Result<Option<i32>> {
    let mut nodes = Vec::new();
    let mut anchors = Vec::new();
    for id in ids {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name) {
            // A file anchor is the module, not a referenceable definition
            // inside it, so neither navigation direction can answer for it.
            anchors.push(node);
            continue;
        }
        if direction.answerable(&node.label) {
            return Ok(None);
        }
        nodes.push(node);
    }
    let Some(node) = nodes.first().or_else(|| anchors.first()) else {
        return Ok(None);
    };
    let lines = nav_file_lines(repo_root, &node.file_path);
    println!(
        "`{target}` is a {}, not a function  {}:{}",
        nav_kind_word(lines.as_ref(), node),
        node.file_path,
        node.start_line.max(1)
    );
    Ok(Some(1))
}

/// A bare name that names several callable definitions selects none of them.
/// Answering for one, or merging them, is the reinterpretation the resolver
/// exists to prevent. The candidates carry no name because it is the same for
/// all of them — what separates them is the file, and the address holds it.
pub(crate) fn nav_refuse_ambiguous(
    store: &greppy_store::Store,
    target: &str,
    ids: &[i64],
) -> Result<Option<i32>> {
    if ids.len() < 2 || split_path_qualified(target).is_some() {
        return Ok(None);
    }
    let mut sites: Vec<(String, i64)> = Vec::new();
    for id in ids {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name) {
            continue;
        }
        if !sites.iter().any(|(file, _)| file == &node.file_path) {
            sites.push((node.file_path.clone(), node.start_line.max(1)));
        }
    }
    if sites.len() < 2 {
        return Ok(None);
    }
    sites.sort();
    println!("`{target}` is {} definitions", sites.len());
    for (file, line) in &sites {
        println!("{file}:{line}");
    }
    Ok(Some(1))
}

/// The name does not resolve. Say so, and add only what the agent could not get
/// with one command of its own: names close enough to be the one it meant, with
/// where they are. Nothing close enough means no block at all — the absence is
/// the statement, and it says the repository does not know this name.
pub(crate) fn nav_report_missing(store: &greppy_store::Store, project: &str, query: &str) {
    println!("no symbol `{query}`");
    let normalise = |name: &str| name.to_ascii_lowercase().replace('_', "").replace('-', "");
    let wanted = normalise(query);
    let mut near: Vec<(String, String, i64)> = Vec::new();
    for name in symbol_miss_suggestions(store, project, query) {
        let close = normalise(&name) == wanted
            || symbol_name_distance(&name, &suggestion_needles(query)) <= 2;
        if !close {
            continue;
        }
        let Ok(ids) = resolve_symbol_nodes(store, Some(&name)) else {
            continue;
        };
        let Some(node) = ids
            .first()
            .and_then(|id| store.get_node(*id).ok().flatten())
        else {
            continue;
        };
        near.push((name, node.file_path.clone(), node.start_line.max(1)));
        if near.len() == 3 {
            break;
        }
    }
    if near.is_empty() {
        return;
    }
    println!();
    println!("similar names:");
    for (name, file, line) in near {
        println!("{file}:{line}  {name}");
    }
}

/// Prints the answer of `who-calls` / `callees`.
///
/// A short result is printed whole — a count beneath lines the reader can count
/// is packaging. Past `NAV_FULL_LIMIT` the shape leads instead: how many, and
/// across which files. On a skewed result that distribution *is* the answer,
/// and the rows below it are an example. Files with the fewest rows come first,
/// so the outliers are not buried under a hub file.
pub(crate) fn print_nav_rows(
    repo_root: &std::path::Path,
    noun: &str,
    rows: &mut [NavAnswerRow],
    code: bool,
    all: bool,
) {
    const NAV_FULL_LIMIT: usize = 25;
    const NAV_SUMMARY_ROWS: usize = 5;

    let mut per_file: std::collections::BTreeMap<String, usize> = Default::default();
    for row in rows.iter() {
        *per_file.entry(row.file.clone()).or_insert(0) += 1;
    }
    rows.sort_by(|a, b| {
        let left = per_file.get(&a.file).copied().unwrap_or(0);
        let right = per_file.get(&b.file).copied().unwrap_or(0);
        left.cmp(&right)
            .then_with(|| a.file.cmp(&b.file))
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.name.cmp(&b.name))
    });

    let total = rows.len();
    let summarize = !all && total > NAV_FULL_LIMIT;
    if summarize {
        let mut spread: Vec<(&String, &usize)> = per_file.iter().collect();
        spread.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
        let spread = spread
            .iter()
            .map(|(file, count)| format!("{file} {count}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("{total} {noun}: {spread}");
        println!();
    }
    let shown = if summarize {
        NAV_SUMMARY_ROWS.min(total)
    } else {
        total
    };
    let mut cache: std::collections::HashMap<String, Option<Vec<String>>> = Default::default();
    for (index, row) in rows.iter().take(shown).enumerate() {
        let marker = if row.test { "  test" } else { "" };
        if !code {
            println!("{}:{}  {}{}", row.file, row.line, row.name, marker);
            continue;
        }
        if index > 0 {
            println!();
        }
        let source = cache
            .entry(row.file.clone())
            .or_insert_with(|| nav_file_lines(repo_root, &row.file));
        let (start, mut end) = row.span;
        if let Some(lines) = source.as_ref() {
            end = end.min(lines.len() as u32).max(start);
        }
        if start == end {
            println!("{}:{}  {}{}", row.file, start, row.name, marker);
        } else {
            println!("{}:{}-{}  {}{}", row.file, start, end, row.name, marker);
        }
        if let Some(lines) = source.as_ref() {
            for line in lines
                .iter()
                .take(end as usize)
                .skip(start.saturating_sub(1) as usize)
            {
                println!("{line}");
            }
        }
    }
}

pub(crate) fn dispatch_who_calls(
    symbol: Option<&str>,
    paths: &[String],
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let query_symbol = symbol.unwrap_or("");
    let path_filters = prepare_query_path_filters(root, "who-calls", query_symbol, paths)?;
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let graph_gate_extra = serde_json::json!({
        "symbol": query_symbol,
        "symbol_found": false,
        "all": all,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "who-calls",
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
        "who-calls",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    // Aggregate incoming CALLS and USAGE across ALL nodes sharing the name + a
    // primary label (e.g. a Struct and its Impl), so references are not lost
    // to a name resolving to the wrong single node.
    let targets = resolve_symbol_nodes(&store, symbol)?;
    // Synthetic file anchors are rejected before ambiguity; every real
    // definition can have incoming CALLS or USAGE references.
    if let Some(symbol) = symbol {
        if json {
            ensure_unambiguous_target(&store, symbol, &targets)?;
        } else {
            let repo_root = resolve_root(root)?;
            if let Some(code) = nav_refuse_non_callable(
                &store,
                &repo_root,
                symbol,
                &targets,
                NavDirection::Incoming,
            )? {
                return Ok(code);
            }
            if let Some(code) = nav_refuse_ambiguous(&store, symbol, &targets)? {
                return Ok(code);
            }
        }
    }
    if targets.is_empty() {
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "who-calls",
                query_symbol,
                &project,
                false,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(1);
        }
        nav_report_missing(&store, &project, query_symbol);
        return Ok(1);
    }
    let mut edges = Vec::new();
    for target in &targets {
        for edge_type in ["CALLS", "USAGE"] {
            edges.extend(store.incoming_edges(*target, Some(edge_type), 1024)?);
        }
    }
    if edges.is_empty() {
        // The symbol IS a defined graph node but has no callers — that is a
        // valid, useful answer, not a failure. Do not fall back to content
        // search (it would just echo the definition as noise).
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "who-calls",
                query_symbol,
                &project,
                true,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(0);
        }
        // Nobody calls it. That is an answer, and it needs no packaging and no
        // textual consolation prize.
        if path_filters.is_empty() {
            println!("no callers");
        } else {
            println!("no callers under path filter: {}", path_filters.shown());
        }
        return Ok(0);
    }
    // Deterministic, de-duplicated output across the aggregated targets.
    // First collect the unique caller nodes so we know the true total, then
    // print at most NAV_LIMIT (F1: cap the token-bomb) unless `--all`.
    let mut seen = std::collections::BTreeSet::new();
    let mut nodes = Vec::new();
    // Collect each reference-site line persisted in the edge properties. The
    // line locates the statement that `--code` prints for the answer row.
    let mut sites: std::collections::HashMap<i64, Vec<u32>> = std::collections::HashMap::new();
    for e in &edges {
        if let Some(l) = e.properties.get("line").and_then(|v| v.as_u64()) {
            sites.entry(e.source_id).or_default().push(l as u32);
        }
        if !seen.insert(e.source_id) {
            continue;
        }
        if let Some(n) = store.get_node(e.source_id)? {
            nodes.push(n);
        }
    }
    nodes.retain(|node| path_filters.matches(&node.file_path));
    if nodes.is_empty() && !path_filters.is_empty() && !json {
        println!("no callers under path filter: {}", path_filters.shown());
        return Ok(0);
    }
    let total = nodes.len();
    let cap = cli_result_limit_unless_all(if code { CODE_NAV_LIMIT } else { NAV_LIMIT }, all);
    let shown = total.min(cap);
    // The expand pack exists only for the JSON consumer now: in text mode
    // everything the pack could carry is one flag away, so offering it would be
    // a second spelling of `--all` plus a handle to remember.
    let expand = if json && !all && !code {
        let rows = nodes
            .iter()
            .map(|n| ExpandEvidenceNode {
                title: display_node_name(n),
                node: n,
                site_lines: sorted_site_lines(sites.get(&n.id)),
                extra_json: serde_json::json!({"role": "caller"}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(
            &store,
            root,
            &project,
            "who-calls",
            query_symbol,
            total,
            &rows,
        )
    } else {
        None
    };
    if json {
        let project = project_for(root)?;
        let hits = nodes[..shown].iter().map(node_hit_json).collect();
        nav_counts_json_with_expand(
            &store,
            root,
            "who-calls",
            query_symbol,
            &project,
            true,
            total,
            shown,
            all,
            hits,
            expand.as_ref(),
        )?;
        return Ok(0);
    }
    let repo_root = resolve_root(root)?;
    // The caller's name answers "who"; the call site answers "where the
    // dependency sits". Both fit one line, and nothing else belongs on it.
    let mut sources: std::collections::HashMap<String, Option<Vec<String>>> = Default::default();
    let mut rows = Vec::with_capacity(nodes.len());
    for n in &nodes {
        // A file anchor is greppy's own bookkeeping, not a symbol. `__file__`
        // in a result list is a name the agent cannot carry anywhere.
        if is_synthetic_file_anchor(&n.label, &n.name, &n.qualified_name) {
            continue;
        }
        let site = sorted_site_lines(sites.get(&n.id))
            .first()
            .copied()
            .unwrap_or_else(|| n.start_line.max(1) as u32);
        let lines = sources
            .entry(n.file_path.clone())
            .or_insert_with(|| nav_file_lines(&repo_root, &n.file_path));
        let span = match lines.as_ref() {
            Some(lines) if code => nav_statement_span(lines, site),
            _ => (site, site),
        };
        rows.push(NavAnswerRow {
            file: n.file_path.clone(),
            line: site,
            span,
            name: nav_short_name(n),
            test: nav_is_test(lines.as_ref(), n),
        });
    }
    print_nav_rows(&repo_root, "callers", &mut rows, code, all);
    Ok(0)
}
/// `greppy callees S` — what `S` calls: every node reached by a direct
/// outgoing CALLS edge from `S`. Printed as `qualified_name file:line` so
/// an agent can jump straight to each callee's definition. Backed by the
/// search `callees_of` helper.
///
/// Like who-calls, this aggregates across ALL nodes sharing the name + a
/// primary label (e.g. a Struct and its Impl) so callees are not lost to
/// a name resolving to the wrong single node. Output is de-duplicated and
/// deterministically ordered by node id.
pub(crate) fn dispatch_callees(
    symbol: Option<&str>,
    paths: &[String],
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let query_symbol = symbol.unwrap_or("");
    let path_filters = prepare_query_path_filters(root, "callees", query_symbol, paths)?;
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let graph_gate_extra = serde_json::json!({
        "symbol": query_symbol,
        "symbol_found": false,
        "all": all,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "callees",
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
        "callees",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let sources = resolve_symbol_nodes(&store, symbol)?;
    if let Some(symbol) = symbol {
        if json {
            ensure_unambiguous_target(&store, symbol, &sources)?;
        } else {
            let repo_root = resolve_root(root)?;
            if let Some(code) = nav_refuse_non_callable(
                &store,
                &repo_root,
                symbol,
                &sources,
                NavDirection::Outgoing,
            )? {
                return Ok(code);
            }
            if let Some(code) = nav_refuse_ambiguous(&store, symbol, &sources)? {
                return Ok(code);
            }
        }
    }
    if sources.is_empty() {
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "callees",
                query_symbol,
                &project,
                false,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(1);
        }
        nav_report_missing(&store, &project, query_symbol);
        return Ok(1);
    }
    // Aggregate direct callees across the resolved source nodes, keyed on
    // the callee node id so a callee reached from both a Struct and its
    // Impl is printed once. BTreeMap keeps the output id-ordered. We keep
    // the full node so `--code` can read its source span.
    let mut callees: std::collections::BTreeMap<i64, greppy_store::Node> =
        std::collections::BTreeMap::new();
    let callee_sources = callee_source_ids_for_symbols(&store, &project, &sources)?;
    for src in &callee_sources {
        for step in greppy_search::callees_of(&store, *src)? {
            if let Some(n) = step.node {
                callees.entry(step.node_id).or_insert(n);
            }
        }
    }
    callees.retain(|_, node| path_filters.matches(&node.file_path));
    if callees.is_empty() {
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "callees",
                query_symbol,
                &project,
                true,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(0);
        }
        if path_filters.is_empty() {
            println!("no callees");
        } else {
            println!("no callees under path filter: {}", path_filters.shown());
        }
        return Ok(0);
    }
    let total = callees.len();
    let cap = cli_result_limit_unless_all(if code { CODE_NAV_LIMIT } else { NAV_LIMIT }, all);
    let shown = total.min(cap);
    let expand = if json && !all && !code {
        let rows = callees
            .values()
            .map(|n| ExpandEvidenceNode {
                title: display_node_name(n),
                node: n,
                site_lines: Vec::new(),
                extra_json: serde_json::json!({"role": "callee"}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(
            &store,
            root,
            &project,
            "callees",
            query_symbol,
            total,
            &rows,
        )
    } else {
        None
    };
    if json {
        let project = project_for(root)?;
        let hits = callees.values().take(shown).map(node_hit_json).collect();
        nav_counts_json_with_expand(
            &store,
            root,
            "callees",
            query_symbol,
            &project,
            true,
            total,
            shown,
            all,
            hits,
            expand.as_ref(),
        )?;
        return Ok(0);
    }
    // Mirror image of `who-calls`: the caller is known, so the new information
    // is where the callee lives — its definition, which is also what `--code`
    // prints.
    let repo_root = resolve_root(root)?;
    let mut sources: std::collections::HashMap<String, Option<Vec<String>>> = Default::default();
    let mut rows = Vec::with_capacity(callees.len());
    for n in callees.values() {
        if is_synthetic_file_anchor(&n.label, &n.name, &n.qualified_name) {
            continue;
        }
        let lines = sources
            .entry(n.file_path.clone())
            .or_insert_with(|| nav_file_lines(&repo_root, &n.file_path));
        rows.push(NavAnswerRow {
            file: n.file_path.clone(),
            line: n.start_line.max(1) as u32,
            span: (
                n.start_line.max(1) as u32,
                n.end_line.max(n.start_line).max(1) as u32,
            ),
            name: nav_short_name(n),
            test: nav_is_test(lines.as_ref(), n),
        });
    }
    print_nav_rows(&repo_root, "callees", &mut rows, code, all);
    Ok(0)
}

/// One edge in a concrete `path` answer. The source definition supplies the
/// file and the edge supplies the line: together they name the editable call
/// site. The target supplies the short name printed beside that address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NavPathStep {
    source_id: i64,
    target_id: i64,
    file: String,
    line: u32,
    name: String,
}

/// A prefix tree made from concrete simple paths. Converged suffixes remain in
/// each branch because only prefixes are merged.
#[derive(Debug)]
pub(crate) struct NavPathBranch {
    step: NavPathStep,
    children: Vec<NavPathBranch>,
}

/// Minimum outgoing distance from every discovered predecessor to `target`.
/// The text walk uses this only to discard branches that cannot end at the
/// requested target; it never changes or filters graph edges that do reach it.
pub(crate) fn nav_path_distances_to(
    store: &greppy_store::Store,
    target: i64,
    edge_type: &str,
    max_hops: usize,
) -> Result<std::collections::HashMap<i64, usize>> {
    let mut distances = std::collections::HashMap::new();
    distances.insert(target, 0);
    let mut frontier = std::collections::VecDeque::from([(target, 0usize)]);
    while let Some((node, hops)) = frontier.pop_front() {
        if hops >= max_hops {
            continue;
        }
        for edge in store.incoming_edges(node, Some(edge_type), 1024)? {
            if let std::collections::hash_map::Entry::Vacant(slot) = distances.entry(edge.source_id)
            {
                slot.insert(hops + 1);
                frontier.push_back((edge.source_id, hops + 1));
            }
        }
    }
    Ok(distances)
}

/// Collect at most eight simple paths in deterministic call-site order. The
/// endpoint is accepted even when it is already on the current stack, which is
/// what makes `--from A --to A` ask for a real cycle instead of inventing the
/// zero-edge path returned by the lower-level shortest-path query.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nav_collect_paths(
    store: &greppy_store::Store,
    current: i64,
    target: i64,
    edge_type: &str,
    max_hops: usize,
    distances: &std::collections::HashMap<i64, usize>,
    nodes: &mut std::collections::HashMap<i64, greppy_store::Node>,
    stack: &mut std::collections::HashSet<i64>,
    path: &mut Vec<NavPathStep>,
    paths: &mut Vec<Vec<NavPathStep>>,
    capped: &mut bool,
) -> Result<()> {
    const PATH_LIMIT: usize = 8;
    if path.len() >= max_hops || paths.len() >= PATH_LIMIT {
        *capped |= paths.len() >= PATH_LIMIT;
        return Ok(());
    }
    let source = match nodes.get(&current) {
        Some(source) => source.clone(),
        None => {
            let Some(source) = store.get_node(current)? else {
                return Ok(());
            };
            nodes.insert(current, source.clone());
            source
        }
    };

    let mut next_steps = Vec::new();
    for edge in store.outgoing_edges(current, Some(edge_type), 1024)? {
        let Some(target_node) = store.get_node(edge.target_id)? else {
            continue;
        };
        let line = edge
            .properties
            .get("line")
            .and_then(serde_json::Value::as_u64)
            .map(|line| line as u32)
            .unwrap_or_else(|| source.start_line.max(1) as u32);
        nodes
            .entry(edge.target_id)
            .or_insert_with(|| target_node.clone());
        next_steps.push(NavPathStep {
            source_id: edge.source_id,
            target_id: edge.target_id,
            file: source.file_path.clone(),
            line,
            name: nav_short_name(&target_node),
        });
    }
    next_steps.sort_by(|left, right| {
        left.line
            .cmp(&right.line)
            .then_with(|| left.file.cmp(&right.file))
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.target_id.cmp(&right.target_id))
    });

    for step in next_steps {
        if paths.len() >= PATH_LIMIT {
            *capped = true;
            break;
        }
        let next_hops = path.len() + 1;
        if step.target_id != target {
            if stack.contains(&step.target_id) {
                continue;
            }
            let Some(remaining) = distances.get(&step.target_id) else {
                continue;
            };
            if next_hops + remaining > max_hops {
                continue;
            }
        }
        path.push(step.clone());
        if step.target_id == target {
            paths.push(path.clone());
            if paths.len() >= PATH_LIMIT {
                *capped = true;
            }
        } else {
            stack.insert(step.target_id);
            nav_collect_paths(
                store,
                step.target_id,
                target,
                edge_type,
                max_hops,
                distances,
                nodes,
                stack,
                path,
                paths,
                capped,
            )?;
            stack.remove(&step.target_id);
        }
        path.pop();
    }
    Ok(())
}

pub(crate) fn nav_insert_path(tree: &mut Vec<NavPathBranch>, path: &[NavPathStep]) {
    let Some((step, rest)) = path.split_first() else {
        return;
    };
    let index = match tree.iter().position(|branch| branch.step == *step) {
        Some(index) => index,
        None => {
            tree.push(NavPathBranch {
                step: step.clone(),
                children: Vec::new(),
            });
            tree.len() - 1
        }
    };
    nav_insert_path(&mut tree[index].children, rest);
}

pub(crate) fn nav_print_path_tree(branches: &[NavPathBranch], depth: usize) {
    for branch in branches {
        println!(
            "{}{}:{}  {}",
            "  ".repeat(depth),
            branch.step.file,
            branch.step.line,
            branch.step.name
        );
        nav_print_path_tree(&branch.children, depth + 1);
    }
}

/// `greppy path --from A --to B [--edge CALLS]` — print every simple path
/// between two unambiguous symbols as one prefix tree. The root is `A` at its
/// definition. Every other row is an edge: the call site inside its parent and
/// the short name of what is called there.
pub(crate) fn dispatch_path(
    from: Option<&str>,
    to: Option<&str>,
    edge: &str,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let from = from.map(str::trim).filter(|s| !s.is_empty());
    let to = to.map(str::trim).filter(|s| !s.is_empty());
    let (Some(from), Some(to)) = (from, to) else {
        return Err(Error::Invalid(
            "path requires both --from <SYMBOL> and --to <SYMBOL>".into(),
        ));
    };
    let edge_upper = edge.trim().to_ascii_uppercase();
    if edge_upper.is_empty() {
        return Err(Error::Invalid("path --edge must not be empty".into()));
    }

    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let max_hops = greppy_search::MAX_REACH_HOPS;
    let graph_gate_extra = serde_json::json!({
        "from": from,
        "to": to,
        "from_found": false,
        "to_found": false,
        "path_found": false,
        "reason": "skipped_stale_index",
        "scope": "shortest_path",
        "direction": "outgoing",
        "edge_type": &edge_upper,
        "max_hops": max_hops,
        "hops": serde_json::Value::Null,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "path",
        json,
        graph_gate_extra.clone(),
        "steps",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "path",
        json,
        serde_json::json!({
            "from": from,
            "to": to,
            "from_found": false,
            "to_found": false,
            "path_found": false,
            "reason": "skipped_incomplete_provider",
            "scope": "shortest_path",
            "direction": "outgoing",
            "edge_type": &edge_upper,
            "max_hops": max_hops,
            "hops": serde_json::Value::Null,
        }),
        "steps",
    )? {
        return Ok(code);
    }

    // JSON remains the existing single-shortest-path contract. The text surface
    // below is deliberately separate so changing its edge-oriented shape cannot
    // alter any field in path's stable machine-readable response.
    if json {
        let from_id = resolve_symbol_id(&store, Some(from))?;
        let to_id = resolve_symbol_id(&store, Some(to))?;
        let Some(from_id) = from_id else {
            path_counts_json(
                &store,
                root,
                from,
                to,
                &project,
                false,
                to_id.is_some(),
                None,
                PathJsonMeta {
                    edge_type: &edge_upper,
                    max_hops,
                    reason: Some("missing_from"),
                },
            )?;
            return Ok(1);
        };
        let Some(to_id) = to_id else {
            path_counts_json(
                &store,
                root,
                from,
                to,
                &project,
                true,
                false,
                None,
                PathJsonMeta {
                    edge_type: &edge_upper,
                    max_hops,
                    reason: Some("missing_to"),
                },
            )?;
            return Ok(1);
        };
        let path = greppy_search::path_query(
            &store,
            from_id,
            to_id,
            greppy_search::ReachDirection::Outgoing,
            &edge_upper,
            max_hops,
        )?;
        let reason = if path.is_some() {
            None
        } else {
            Some("no_path")
        };
        path_counts_json(
            &store,
            root,
            from,
            to,
            &project,
            true,
            true,
            path.as_ref(),
            PathJsonMeta {
                edge_type: &edge_upper,
                max_hops,
                reason,
            },
        )?;
        return Ok(if path.is_some() { 0 } else { 1 });
    }

    let repo_root = resolve_root(root)?;
    let from_nodes = resolve_symbol_nodes(&store, Some(from))?;
    if let Some(code) = nav_refuse_non_callable(
        &store,
        &repo_root,
        from,
        &from_nodes,
        NavDirection::Incoming,
    )? {
        return Ok(code);
    }
    if let Some(code) = nav_refuse_ambiguous(&store, from, &from_nodes)? {
        return Ok(code);
    }
    if from_nodes.is_empty() {
        nav_report_missing(&store, &project, from);
        return Ok(1);
    }

    let to_nodes = resolve_symbol_nodes(&store, Some(to))?;
    if let Some(code) =
        nav_refuse_non_callable(&store, &repo_root, to, &to_nodes, NavDirection::Incoming)?
    {
        return Ok(code);
    }
    if let Some(code) = nav_refuse_ambiguous(&store, to, &to_nodes)? {
        return Ok(code);
    }
    if to_nodes.is_empty() {
        nav_report_missing(&store, &project, to);
        return Ok(1);
    }

    let Some(from_id) = resolve_symbol_id(&store, Some(from))? else {
        nav_report_missing(&store, &project, from);
        return Ok(1);
    };
    let Some(to_id) = resolve_symbol_id(&store, Some(to))? else {
        nav_report_missing(&store, &project, to);
        return Ok(1);
    };
    let Some(start) = store.get_node(from_id)? else {
        nav_report_missing(&store, &project, from);
        return Ok(1);
    };

    let distances = nav_path_distances_to(&store, to_id, &edge_upper, max_hops)?;
    let mut nodes = std::collections::HashMap::from([(from_id, start.clone())]);
    let mut stack = std::collections::HashSet::from([from_id]);
    let mut current_path = Vec::new();
    let mut paths = Vec::new();
    let mut capped = false;
    nav_collect_paths(
        &store,
        from_id,
        to_id,
        &edge_upper,
        max_hops,
        &distances,
        &mut nodes,
        &mut stack,
        &mut current_path,
        &mut paths,
        &mut capped,
    )?;
    if paths.is_empty() {
        println!("no path from {from} to {to}");
        return Ok(0);
    }

    println!(
        "{}:{}  {}",
        start.file_path,
        start.start_line.max(1),
        nav_short_name(&start)
    );
    let mut tree = Vec::new();
    for path in &paths {
        nav_insert_path(&mut tree, path);
    }
    nav_print_path_tree(&tree, 1);
    if capped {
        println!("at least 8 paths shown");
    }
    Ok(0)
}
