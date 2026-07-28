//! The `plus` surface.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

pub(crate) fn plus_relevance_from_ranks(ranks: &[f64], rank: f64) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let mut best = ranks[0];
    let mut worst = ranks[0];
    for r in ranks {
        if *r < best {
            best = *r;
        }
        if *r > worst {
            worst = *r;
        }
    }
    let span = worst - best;
    if span > 0.0 {
        (worst - rank) / span
    } else {
        1.0
    }
}

pub(crate) fn plus_query_tokens(query: &str) -> Vec<String> {
    greppy_store::fts::camel_split(query)
        .split_whitespace()
        .map(plus_canonical_token)
        .filter(|tok| tok.len() >= 3)
        .collect()
}

pub(crate) fn plus_canonical_token(token: &str) -> String {
    let t = token.to_ascii_lowercase();
    for suffix in ["isation", "ization"] {
        if let Some(base) = t.strip_suffix(suffix) {
            return format!("{base}ize");
        }
    }
    for suffix in ["ising", "izing", "ised", "ized"] {
        if let Some(base) = t.strip_suffix(suffix) {
            return format!("{base}ize");
        }
    }
    if let Some(base) = t.strip_suffix("ise") {
        return format!("{base}ize");
    }
    t
}

pub(crate) fn plus_symbol_tokens(node: &greppy_store::Node) -> Vec<String> {
    greppy_store::fts::camel_split(&format!(
        "{} {} {}",
        node.name, node.qualified_name, node.file_path
    ))
    .split_whitespace()
    .map(plus_canonical_token)
    .filter(|tok| tok.len() >= 3)
    .collect()
}

pub(crate) fn plus_is_pseudo_node(node: &greppy_store::Node) -> bool {
    matches!(node.label.as_str(), "Module" | "Import" | "Call")
        || node.qualified_name.ends_with("::__file__")
}

pub(crate) fn plus_is_constructor_like(node: &greppy_store::Node) -> bool {
    if !matches!(node.label.as_str(), "Method" | "Function") {
        return false;
    }
    let parts: Vec<&str> = node
        .qualified_name
        .split("::")
        .filter(|part| !part.is_empty())
        .collect();
    parts.len() >= 2 && parts[parts.len() - 1] == parts[parts.len() - 2]
}

pub(crate) fn plus_is_executable_node(node: &greppy_store::Node) -> bool {
    matches!(node.label.as_str(), "Function" | "Method") && !plus_is_constructor_like(node)
}

pub(crate) fn plus_is_code_intent(tokens: &[String]) -> bool {
    tokens.iter().any(|tok| {
        matches!(
            tok.as_str(),
            "code"
                | "show"
                | "where"
                | "return"
                | "loop"
                | "fold"
                | "function"
                | "method"
                | "implement"
                | "implemented"
                | "implementation"
        )
    })
}

pub(crate) fn plus_is_literal_intent(query: &str, tokens: &[String]) -> bool {
    let trimmed = query.trim();
    tokens.len() == 1
        || trimmed.contains('_')
        || trimmed.contains("::")
        || trimmed.chars().any(|c| c.is_ascii_digit())
}

pub(crate) fn plus_has_camel_identifier(query: &str) -> bool {
    query
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
        .any(|part| {
            let mut chars = part.chars();
            let Some(first) = chars.next() else {
                return false;
            };
            first.is_ascii_alphabetic()
                && part.chars().any(|c| c.is_ascii_lowercase())
                && chars.any(|c| c.is_ascii_uppercase())
        })
}

pub(crate) fn plus_is_graph_control_token(token: &str) -> bool {
    matches!(
        token,
        "affected"
            | "blast"
            | "break"
            | "call"
            | "called"
            | "caller"
            | "callers"
            | "calls"
            | "callee"
            | "callees"
            | "change"
            | "changed"
            | "dependency"
            | "dependents"
            | "depends"
            | "direct"
            | "find"
            | "from"
            | "impact"
            | "path"
            | "radius"
            | "reference"
            | "referenced"
            | "references"
            | "trace"
            | "usage"
            | "usages"
            | "what"
            | "where"
            | "would"
    )
}

pub(crate) fn plus_is_graph_control_intent(query: &str, tokens: &[String]) -> bool {
    let q = query.to_ascii_lowercase();
    let graph_phrase = q.contains("who calls")
        || q.contains("what calls")
        || q.contains("called by")
        || q.contains("direct caller")
        || q.contains("direct callee")
        || q.contains("call path")
        || q.contains("trace from")
        || q.contains("trace path")
        || q.starts_with("trace ")
        || q.contains("path from")
        || q.contains("depends on")
        || q.contains("dependency path")
        || q.contains("referenced by")
        || q.contains("references to")
        || q.contains("find usages")
        || q.contains("usages of")
        || q.contains("what would break")
        || q.contains("break if")
        || q.contains("affected by")
        || q.contains("blast radius")
        || (q.contains("impact") && (q.contains("change") || q.contains("changed")));
    if !graph_phrase {
        return false;
    }

    query.contains('_')
        || query.contains("::")
        || plus_has_camel_identifier(query)
        || tokens
            .iter()
            .any(|tok| tok.len() >= 5 && !plus_is_graph_control_token(tok))
}

pub(crate) fn plus_vector_control_intent(
    query: &str,
    tokens: &[String],
    has_exact_text_hit: bool,
) -> Option<PlusVectorControlIntent> {
    if plus_is_graph_control_intent(query, tokens) {
        Some(PlusVectorControlIntent::Graph)
    } else if has_exact_text_hit || plus_is_literal_intent(query, tokens) {
        Some(PlusVectorControlIntent::Literal)
    } else {
        None
    }
}

pub(crate) fn plus_allows_ranked_node(node: &greppy_store::Node, code_intent: bool) -> bool {
    if plus_is_pseudo_node(node) {
        return false;
    }
    !code_intent || plus_is_executable_node(node)
}

pub(crate) fn plus_precision_floor(best_score: f64) -> f64 {
    if best_score >= 0.80 {
        best_score - 0.10
    } else if best_score >= 0.70 {
        best_score - 0.15
    } else {
        0.0
    }
}

pub(crate) fn plus_token_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    let distance = plus_levenshtein(a, b) as f64;
    let width = a.chars().count().max(b.chars().count()) as f64;
    (1.0 - (distance / width)).clamp(0.0, 1.0)
}

pub(crate) fn plus_levenshtein(a: &str, b: &str) -> usize {
    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0; b_chars.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_chars.len()]
}

pub(crate) fn plus_key(file_path: &str, line: i64) -> String {
    format!("{file_path}:{line}")
}

pub(crate) fn plus_first_line(root: &std::path::Path, node: &greppy_store::Node) -> String {
    read_span(
        root,
        &node.file_path,
        node.start_line,
        node.end_line,
        1,
        false,
    )
    .and_then(|span| span.lines().next().map(|line| line.trim().to_string()))
    .filter(|line| !line.is_empty())
    .unwrap_or_else(|| node.qualified_name.clone())
}

pub(crate) fn plus_store_node_from_row(
    store: &greppy_store::Store,
    row: &greppy_search::graph::SearchGraphRow,
) -> Result<Option<greppy_store::Node>> {
    Ok(store.get_node(row.id)?)
}

pub(crate) fn plus_enclosing_node(
    store: &greppy_store::Store,
    project: &str,
    location: &str,
) -> Result<Option<greppy_store::Node>> {
    let Some((file, line_str)) = location.rsplit_once(':') else {
        return Ok(None);
    };
    let Ok(line) = line_str.parse::<i64>() else {
        return Ok(None);
    };
    match greppy_search::definition_at(store, Some(project), file, line)? {
        Some(row) => plus_store_node_from_row(store, &row),
        None => Ok(None),
    }
}

pub(crate) fn plus_put_hit(
    hits: &mut std::collections::BTreeMap<String, PlusHit>,
    file_path: &str,
    line: i64,
    snippet: String,
    node: Option<greppy_store::Node>,
    signal: impl Into<String>,
    confidence: f64,
) {
    let key = plus_key(file_path, line);
    let entry = hits.entry(key).or_insert_with(|| PlusHit {
        location: format!("{file_path}:{line}"),
        file_path: file_path.to_string(),
        line,
        symbol: node.as_ref().map(|n| n.qualified_name.clone()),
        node: node.clone(),
        score: 0.0,
        signals: std::collections::BTreeSet::new(),
        snippet: snippet.clone(),
    });
    if entry.symbol.is_none() {
        entry.symbol = node.as_ref().map(|n| n.qualified_name.clone());
    }
    if entry.node.is_none() {
        entry.node = node;
    }
    if entry.snippet.trim().is_empty() && !snippet.trim().is_empty() {
        entry.snippet = snippet;
    }
    entry.add_signal(signal, confidence);
}

pub(crate) fn plus_vector_confidence(score: f32) -> f64 {
    let min = f64::from(PLUS_VECTOR_MIN_SCORE);
    let s = f64::from(score);
    if s < min {
        return 0.0;
    }
    (((s - min) / (1.0 - min)) * PLUS_VECTOR_MAX_CONFIDENCE).clamp(0.0, PLUS_VECTOR_MAX_CONFIDENCE)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plus_add_vector_hits_from_query_vector(
    store: &greppy_store::Store,
    project: &str,
    root_path: &std::path::Path,
    code_intent: bool,
    hits: &mut std::collections::BTreeMap<String, PlusHit>,
    model_id: &str,
    graph_generation: u64,
    query_vector: &[f32],
    limit: usize,
) -> Result<usize> {
    let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
        project,
        model_id,
        Some(graph_generation),
        limit,
    );
    scope.min_score = Some(PLUS_VECTOR_MIN_SCORE);

    let mut added = 0usize;
    for h in greppy_search::vector_search_exact(store, query_vector, &scope)? {
        let node = match h.embedding.node_id {
            Some(id) => store.get_node(id)?,
            None => None,
        };
        if let Some(n) = &node {
            if !plus_allows_ranked_node(n, code_intent) {
                continue;
            }
        }
        let (file_path, line, snippet) = if let Some(n) = &node {
            (
                n.file_path.clone(),
                n.start_line,
                plus_first_line(root_path, n),
            )
        } else {
            (
                h.embedding.file_path.clone(),
                h.embedding.start_line,
                h.embedding.qualified_name.clone(),
            )
        };
        plus_put_hit(
            hits,
            &file_path,
            line,
            snippet,
            node,
            "vector",
            plus_vector_confidence(h.score),
        );
        added += 1;
    }
    Ok(added)
}

pub(crate) fn plus_add_graph_signals(store: &greppy_store::Store, hit: &mut PlusHit) -> Result<()> {
    let Some(node) = &hit.node else {
        return Ok(());
    };
    let is_executable = plus_is_executable_node(node);
    let incoming = store.incoming_edges(node.id, Some("CALLS"), 1024)?.len();
    let outgoing = store.outgoing_edges(node.id, Some("CALLS"), 1024)?.len();
    if incoming > 0 {
        let boost = ((incoming as f64 + 1.0).log10() * 0.10).min(0.22);
        hit.add_signal(format!("graph-in={incoming}"), boost);
    }
    if outgoing > 0 {
        let boost = ((outgoing as f64 + 1.0).log10() * 0.10).min(0.18);
        hit.add_signal(format!("graph-out={outgoing}"), boost);
    }
    if is_executable {
        hit.add_signal("kind=code", 0.10);
    }
    Ok(())
}

pub(crate) fn plus_json(
    meta: PlusJsonMeta<'_>,
    ranked: &[PlusHit],
    root_path: &std::path::Path,
) -> Result<()> {
    let eligible = ranked
        .iter()
        .filter(|hit| hit.score >= meta.precision_floor)
        .collect::<Vec<_>>();
    let shown_hits = eligible
        .iter()
        .copied()
        .take(meta.limit)
        .collect::<Vec<_>>();
    let omitted = eligible.len().saturating_sub(shown_hits.len());
    let mut source_unavailable_count = 0usize;
    let mut source_truncated_count = 0usize;
    let mut rows = Vec::new();

    for hit in shown_hits {
        let signals = hit.signals.iter().cloned().collect::<Vec<_>>();
        let snippet = clamp_snippet(&hit.snippet).into_owned();
        let mut source_available = false;
        let mut source_included = false;
        let mut source = serde_json::Value::Null;
        let mut source_total_lines = serde_json::Value::Null;
        let mut source_shown_lines = serde_json::Value::Null;
        let mut source_omitted_lines = serde_json::Value::Null;
        let mut source_truncated = false;

        if meta.code {
            if let Some(node) = &hit.node {
                match read_span_with_meta(
                    root_path,
                    &node.file_path,
                    node.start_line,
                    node.end_line,
                    CODE_SPAN_CAP,
                    false,
                ) {
                    Some(span) => {
                        source_available = true;
                        source_included = true;
                        source_truncated = span.truncated;
                        if source_truncated {
                            source_truncated_count += 1;
                        }
                        source = serde_json::Value::String(span.text);
                        source_total_lines = serde_json::json!(span.total_lines);
                        source_shown_lines = serde_json::json!(span.shown_lines);
                        source_omitted_lines = serde_json::json!(span.omitted_lines);
                    }
                    None => {
                        source_unavailable_count += 1;
                    }
                }
            } else {
                source_unavailable_count += 1;
            }
        }

        rows.push(serde_json::json!({
            "location": hit.location,
            "file_path": hit.file_path,
            "line": hit.line,
            "snippet": snippet,
            "score": hit.score,
            "signals": signals,
            "symbol": hit.symbol,
            "source_available": source_available,
            "source_included": source_included,
            "source": source,
            "source_total_lines": source_total_lines,
            "source_shown_lines": source_shown_lines,
            "source_omitted_lines": source_omitted_lines,
            "source_truncated": source_truncated,
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "plus",
            "status": meta.status,
            "project": meta.project,
            "query": meta.query,
            "fresh": meta.freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": meta.freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": meta.provider_complete,
            "incomplete_provider_count": meta.incomplete_providers.len(),
            "incomplete_providers": meta.incomplete_providers,
            "limit": meta.limit,
            "code": meta.code,
            "explain": meta.explain,
            "vectors": meta.vectors,
            "fetch_limit_per_signal": meta.fetch_limit_per_signal,
            "candidate_total_kind": "bounded_fetch_union",
            "ranked_total": ranked.len(),
            "eligible_total": eligible.len(),
            "shown": rows.len(),
            "omitted": omitted,
            "truncated": omitted > 0 || source_truncated_count > 0,
            "precision_floor": meta.precision_floor,
            "source_cap_lines": CODE_SPAN_CAP,
            "source_unavailable_count": source_unavailable_count,
            "source_truncated_count": source_truncated_count,
            "vector_status": meta.vector_status,
            "vector_candidate_total": meta.vector_candidate_total,
            "vector_candidate_limit": meta.vector_candidate_limit,
            "vector_hits_added": meta.vector_hits_added,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize JSON: {e}")))?
    );
    Ok(())
}

/// `greppy plus <query>` — a grep-like fused search path.
///
/// This deliberately stays a SEARCH command: it does not summarize, does not
/// answer the user's question, and does not invent context. It emits ranked
/// hits with stable locations and signal labels, combining the "plus" parts
/// grep lacks: symbol matching, fuzzy semantic matching, and graph-neighbour
/// hints. EmbeddingGemma code-retrieval hits are always available as another
/// signal, scoped to the current graph generation. Exact literal/graph control
/// queries still short-circuit before loading the model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_plus(
    query: Option<&str>,
    k: usize,
    code: bool,
    explain: bool,
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let vectors = true;
    let store = open_default_store(root)?;
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("a query is required".into()));
    }
    let k = cli_result_limit(k).max(1);
    let project = project_for(root)?;
    let root_path = resolve_root(root)?;
    // Combined search also refuses stale indexed lexical/graph/vector rows.
    let decision = freshness_serve_decision(&store, root, &project);
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    let provider_complete = incomplete_providers.is_empty();
    let fetch = (k * 4).max(20);
    if let FreshnessServe::Refuse(freshness) = &decision {
        if json {
            plus_json(
                PlusJsonMeta {
                    status: "skipped_stale_index",
                    project: &project,
                    query: q,
                    freshness: Some(freshness),
                    provider_complete,
                    incomplete_providers: &incomplete_providers,
                    limit: k,
                    code,
                    explain,
                    vectors,
                    fetch_limit_per_signal: fetch,
                    precision_floor: 0.0,
                    vector_status: if vectors {
                        Some("skipped_stale_index")
                    } else {
                        None
                    },
                    vector_candidate_total: None,
                    vector_candidate_limit: None,
                    vector_hits_added: None,
                },
                &[],
                &root_path,
            )?;
        } else {
            eprintln!("{}", plus_stale_skip_message(freshness));
            println!(
                "(no usable index; run `greppy index {}` first)",
                root.unwrap_or(".")
            );
        }
        return Ok(freshness_refusal_exit(freshness));
    }
    let freshness = decision.freshness().clone();
    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            plus_json(
                PlusJsonMeta {
                    status: "skipped_incomplete_provider",
                    project: &project,
                    query: q,
                    freshness: Some(&freshness),
                    provider_complete,
                    incomplete_providers: &incomplete_providers,
                    limit: k,
                    code,
                    explain,
                    vectors,
                    fetch_limit_per_signal: fetch,
                    precision_floor: 0.0,
                    vector_status: if vectors {
                        Some("skipped_incomplete_provider")
                    } else {
                        None
                    },
                    vector_candidate_total: None,
                    vector_candidate_limit: None,
                    vector_hits_added: None,
                },
                &[],
                &root_path,
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("grep", incomplete_providers.len())
            );
        }
        return Ok(1);
    }
    let q_tokens = plus_query_tokens(q);
    let code_intent = plus_is_code_intent(&q_tokens);
    let mut hits: std::collections::BTreeMap<String, PlusHit> = std::collections::BTreeMap::new();
    let mut vector_status = if vectors { Some("requested") } else { None };
    let mut vector_candidate_total = None;
    let mut vector_candidate_limit = None;
    let mut vector_hits_added = None;

    // Literal/full-text signal: exact current-worktree lines remain
    // first-class grep-like results even though source bodies are not copied
    // into SQLite by default.
    let code_hits = source_code_hits_ranked(&store, &project, q, &root_path, fetch)?;
    let exact_literal_text = !code && !code_hits.is_empty() && plus_is_literal_intent(q, &q_tokens);
    let vector_control_intent = if vectors {
        plus_vector_control_intent(q, &q_tokens, exact_literal_text)
    } else {
        None
    };
    if let Some(control) = vector_control_intent {
        vector_status = Some(control.status());
        if !json {
            eprintln!("{}", control.message());
        }
    }
    let vector_config = if vectors && vector_control_intent.is_none() {
        Some(embedding_config_for_required_use(embedding_args)?)
    } else {
        None
    };
    for h in &code_hits {
        let Some((file, line_str)) = h.location.rsplit_once(':') else {
            continue;
        };
        let Ok(line) = line_str.parse::<i64>() else {
            continue;
        };
        let node = plus_enclosing_node(&store, &project, &h.location)?;
        plus_put_hit(
            &mut hits,
            file,
            line,
            h.snippet.clone(),
            node,
            "text",
            h.relevance,
        );
    }
    if code_hits.is_empty() {
        let mut seen_tokens = std::collections::BTreeSet::new();
        for tok in plus_query_tokens(q) {
            if !seen_tokens.insert(tok.clone()) {
                continue;
            }
            for h in source_code_hits_ranked(&store, &project, &tok, &root_path, fetch / 2)? {
                let Some((file, line_str)) = h.location.rsplit_once(':') else {
                    continue;
                };
                let Ok(line) = line_str.parse::<i64>() else {
                    continue;
                };
                let node = plus_enclosing_node(&store, &project, &h.location)?;
                plus_put_hit(
                    &mut hits,
                    file,
                    line,
                    h.snippet,
                    node,
                    format!("text-token={tok}"),
                    h.relevance * 0.72,
                );
            }
        }
    }

    if !exact_literal_text {
        // Symbol FTS signal: identifier/camel-case aware, still output as a
        // location + snippet, not as prose.
        let symbol_hits = greppy_search::search_symbols_in_project(&store, &project, q, fetch)?;
        let symbol_ranks: Vec<f64> = symbol_hits.iter().map(|h| h.rank).collect();
        for h in symbol_hits {
            if let Some(n) = store.get_node(h.node_id)? {
                if !plus_allows_ranked_node(&n, code_intent) {
                    continue;
                }
                let rel = plus_relevance_from_ranks(&symbol_ranks, h.rank);
                let snippet = plus_first_line(&root_path, &n);
                let file_path = n.file_path.clone();
                let start_line = n.start_line;
                plus_put_hit(
                    &mut hits,
                    &file_path,
                    start_line,
                    snippet,
                    Some(n),
                    "symbol",
                    rel,
                );
            }
        }

        // Fuzzy token signal over symbols: catches spelling/convention variants
        // such as normalisation/normalize without turning the command into prose.
        if !q_tokens.is_empty() {
            for n in store.list_nodes(&project, "", "", 0, 100_000)? {
                if !plus_allows_ranked_node(&n, code_intent) {
                    continue;
                }
                let node_tokens = plus_symbol_tokens(&n);
                let best = q_tokens
                    .iter()
                    .flat_map(|qt| {
                        node_tokens
                            .iter()
                            .map(move |nt| plus_token_similarity(qt, nt))
                    })
                    .fold(0.0_f64, f64::max);
                if best >= 0.86 {
                    let snippet = plus_first_line(&root_path, &n);
                    let file_path = n.file_path.clone();
                    let start_line = n.start_line;
                    plus_put_hit(
                        &mut hits,
                        &file_path,
                        start_line,
                        snippet,
                        Some(n),
                        "fuzzy-token",
                        (best * 0.78).min(0.78),
                    );
                }
            }
        }

        // Fuzzy semantic signal: algorithmic semantic scorer over indexed symbol
        // metadata. This is the "plus" part, still represented as a search hit.
        for h in greppy_search::semantic_query(&store, q, None, Some(&project), fetch)? {
            if let Some(n) = plus_store_node_from_row(&store, &h.node)? {
                if !plus_allows_ranked_node(&n, code_intent) {
                    continue;
                }
                let confidence = (h.score / greppy_search::MAX_SEMANTIC_SCORE).clamp(0.0, 1.0);
                let snippet = plus_first_line(&root_path, &n);
                let file_path = n.file_path.clone();
                let start_line = n.start_line;
                plus_put_hit(
                    &mut hits,
                    &file_path,
                    start_line,
                    snippet,
                    Some(n),
                    "fuzzy",
                    confidence,
                );
            }
        }

        if let Some(cfg) = &vector_config {
            let freshness = nav_freshness_json(&store, root, &project);
            if !freshness_json_is_fresh(&freshness) {
                vector_status = Some("skipped_stale_index");
                if !json {
                    eprintln!("{}", vector_stale_skip_message("grep", &freshness));
                }
            } else {
                let generation = current_graph_generation(&store, root)?;
                let scope = greppy_search::embeddinggemma_code_retrieval_scope(
                    &project,
                    &cfg.model_id,
                    Some(generation),
                    fetch,
                );
                let total = greppy_search::count_vector_search_scope(&store, &scope)?;
                let candidate_limit = vector_exact_candidate_limit()?;
                vector_candidate_total = Some(total);
                vector_candidate_limit = candidate_limit;
                if total == 0 {
                    vector_status = Some("no_current_vectors");
                    // No vector rows for this model/profile/generation; keep
                    // the normal plus path intact.
                } else if let Some(limit) = vector_exact_scan_exceeds_limit(total, candidate_limit)
                {
                    vector_status = Some("skipped_over_budget");
                    if !json {
                        eprintln!("{}", vector_exact_scan_skip_message("grep", total, limit));
                    }
                } else {
                    match embed_query_cached(cfg, root, q) {
                        Ok(query_vector) => {
                            let added = plus_add_vector_hits_from_query_vector(
                                &store,
                                &project,
                                &root_path,
                                code_intent,
                                &mut hits,
                                &cfg.model_id,
                                generation,
                                &query_vector,
                                fetch,
                            )?;
                            vector_status = Some("searched");
                            vector_hits_added = Some(added);
                        }
                        Err(e) => {
                            vector_status = Some("skipped_embedding_error");
                            log_embedding_skip_once("grep", &e);
                        }
                    }
                }
            }
        }
    }

    for hit in hits.values_mut() {
        plus_add_graph_signals(&store, hit)?;
    }

    let mut ranked: Vec<PlusHit> = hits.into_values().collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.file_path.cmp(&b.file_path))
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.symbol.cmp(&b.symbol))
    });

    if ranked.is_empty() {
        if json {
            plus_json(
                PlusJsonMeta {
                    status: "ok",
                    project: &project,
                    query: q,
                    freshness: Some(&freshness),
                    provider_complete,
                    incomplete_providers: &incomplete_providers,
                    limit: k,
                    code,
                    explain,
                    vectors,
                    fetch_limit_per_signal: fetch,
                    precision_floor: 0.0,
                    vector_status,
                    vector_candidate_total,
                    vector_candidate_limit,
                    vector_hits_added,
                },
                &ranked,
                &root_path,
            )?;
        } else {
            println!("(no matches)");
        }
        return Ok(1);
    }

    let precision_floor = if explain || ranked.is_empty() {
        0.0
    } else {
        plus_precision_floor(ranked[0].score)
    };
    if json {
        plus_json(
            PlusJsonMeta {
                status: "ok",
                project: &project,
                query: q,
                freshness: Some(&freshness),
                provider_complete,
                incomplete_providers: &incomplete_providers,
                limit: k,
                code,
                explain,
                vectors,
                fetch_limit_per_signal: fetch,
                precision_floor,
                vector_status,
                vector_candidate_total,
                vector_candidate_limit,
                vector_hits_added,
            },
            &ranked,
            &root_path,
        )?;
        return Ok(0);
    }
    let mut printed = 0usize;
    for hit in ranked
        .iter()
        .filter(|hit| hit.score >= precision_floor)
        .take(k)
    {
        print!("{}:{}", hit.location, clamp_snippet(&hit.snippet));
        if explain {
            let signals = hit
                .signals
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(",");
            let symbol = hit
                .node
                .as_ref()
                .map(display_node_name)
                .or_else(|| hit.symbol.clone())
                .unwrap_or_else(|| "-".to_string());
            print!(
                "\t# score={:.3} signals={} symbol={}",
                hit.score, signals, symbol
            );
        }
        println!();
        printed += 1;
        if code {
            if let Some(node) = &hit.node {
                print_code_span(&root_path, node, CODE_SPAN_CAP);
            }
        }
    }
    if printed == 0 {
        let hit = &ranked[0];
        println!("{}:{}", hit.location, clamp_snippet(&hit.snippet));
    }
    Ok(0)
}

pub(crate) fn plus_stale_skip_message(freshness: &serde_json::Value) -> String {
    format!(
        "grep: indexed search skipped because {}; no stale indexed hits emitted",
        stale_freshness_reason(freshness)
    )
}
