//! The `context` command.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

/// `greppy context <query> [--k N] [--lines]` — the token-saving lever.
///
/// Instead of returning `file:line` POINTERS (which force the agent to
/// READ whole files), this resolves the most relevant DEFINITIONS for
/// `<query>` and prints their ACTUAL SOURCE SPANS, so the agent reads the
/// relevant function/struct bodies directly from greppy output.
///
/// Resolution unions four signals, in priority order, deduplicating on
/// node id while preserving first-seen order:
/// 1. `search_symbols` — exact/FTS symbol-name matches (most precise).
/// 2. `semantic_query` — algorithmic similarity (catches paraphrases).
/// 3. `search_code` → `definition_at` — content matches resolved to the
///    enclosing definition (catches symbols only the body mentions).
///
/// The top-K (default 6) definitions are emitted with a compact
/// `== qualified_name (file:start-end) ==` header followed by the source
/// span read from disk (capped at [`CONTEXT_SPAN_CAP`] lines, with a
/// truncation note). The command refuses a stale index before emitting spans;
/// missing files / out-of-range lines are still skipped gracefully as a final
/// guard against races.
pub(crate) fn dispatch_context(
    query: Option<&str>,
    k: usize,
    lines: bool,
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("context requires a query".into()));
    }
    let k = cli_result_limit(k).max(1);
    let project = project_for(root)?;
    let span_root = resolve_root(root)?;
    // Context also refuses stale/unknown graph locations. It never combines
    // an old indexed line number with current source text.
    let decision = freshness_serve_decision(&store, root, &project);
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    if let FreshnessServe::Refuse(freshness) = &decision {
        if json {
            context_json(
                &store,
                &project,
                "skipped_stale_index",
                Some(freshness),
                k,
                lines,
                &[],
                &span_root,
            )?;
        } else {
            println!("{}", context_stale_skip_message(freshness));
        }
        return Ok(freshness_refusal_exit(freshness));
    }
    let freshness = decision.freshness().clone();
    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            context_json(
                &store,
                &project,
                "skipped_incomplete_provider",
                Some(&freshness),
                k,
                lines,
                &[],
                &span_root,
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("context", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    // Ordered, de-duplicated candidate definitions keyed on node id.
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut defs: Vec<ContextDef> = Vec::new();

    // Exact-name / show-definition fast path (Z3): when the query is a
    // single bare identifier that resolves to real primary definition(s)
    // by EXACT name, this is a "find the definition of X" lookup — the
    // domain where plain grep is optimal. Return ONLY those exact-name
    // definitions (grep-shaped: file:line + the def's own span) and skip
    // the semantic / code-search padding that would otherwise pull in
    // callers and paraphrase matches. This keeps a literal lookup
    // grep-competitive instead of ingesting several unrelated full spans.
    // Natural-language research queries (which contain spaces) never take
    // this path, so `context` stays rich for research.
    if is_bare_identifier(q) {
        let exact = resolve_symbol_nodes(&store, Some(q))?;
        if !exact.is_empty() {
            for id in exact {
                if defs.len() >= k {
                    break;
                }
                if let Some(n) = store.get_node(id)? {
                    if seen.insert(n.id) {
                        defs.push(ContextDef {
                            qualified_name: n.qualified_name,
                            file_path: n.file_path,
                            start_line: n.start_line,
                            end_line: n.end_line,
                            node_id: Some(n.id),
                        });
                    }
                }
            }
            return emit_context_locators(
                &store, &project, &freshness, k, lines, json, &defs, &span_root,
            );
        }
    }

    // Over-fetch from each source so the union has enough candidates to
    // fill K distinct definitions after dedup. A small multiple of K is
    // plenty and keeps each query fast.
    let fetch = (k * 4).max(20);

    // Decide the vector fallback up front (D6 / fuzzy-lever fix). It fires for
    // a multi-word (non-bare) natural-language query that names NO real symbol
    // by EXACT name. Two facts make this the right gate — and make it the
    // PRIMARY signal, not a trailing append:
    //
    //  * The lexical semantic step (`semantic_query` -> `score_one`) returns a
    //    hit for ANY node sharing even ONE token with the query. A conceptual
    //    phrase ("restrict a numeric value to stay within bounds") therefore
    //    fills all K slots with token-overlap NOISE (`Identifier::Field`, …)
    //    before any embedding runs. Appending vectors AFTER that union is a
    //    no-op: the `defs.len() >= k` guard drops every vector hit. So when we
    //    know the query names no symbol (lexical hits are noise), the vectors
    //    must LEAD, and the lexical union only backfills leftover slots.
    //  * If the phrase DID resolve to a real symbol (`Owner.method`, an exact
    //    name) vectors stay off and the lexical/exact union answers, so a
    //    genuine exact/FTS match is never displaced by a paraphrase.
    //
    // Router safety (task_classes_v2 `avoid_embedding`): a bare name is handled
    // on the Z3 exact fast path far above and never reaches here; the
    // `!is_bare_identifier` guard keeps any bare name that slipped through
    // (resolved nothing) off vectors. Degrades gracefully (a labeled stderr
    // note, not a crash) when no model is configured or no vectors exist.
    let use_vectors = !is_bare_identifier(q) && resolve_symbol_nodes(&store, Some(q))?.is_empty();
    if use_vectors {
        if let Some((hits, low_confidence)) =
            context_vector_fallback(&store, &project, &freshness, q, fetch, embedding_args, root)?
        {
            // The vectors ARE the answer for this conceptual query (it named no
            // symbol by exact name, so any lexical hit is token-overlap noise —
            // see the `use_vectors` rationale above). Emit the top semantic
            // matches as LEAN grep-shaped locators and STOP: return the location
            // + signature, not K full function bodies. The old behaviour pushed
            // vector hits into `defs` and fell through to the full-body union,
            // which turned the vectors' quality win into a token LOSS and left
            // the agent iterating because it could not tell which body answered.
            let mut vec_defs: Vec<ContextDef> = Vec::new();
            for h in hits {
                if vec_defs.len() >= CONTEXT_VECTOR_LEAN_TOP_N {
                    break;
                }
                // Dedup by node id when present; span-only rows (no node id)
                // cannot collide, so take them.
                if h.node_id.map(|id| seen.insert(id)).unwrap_or(true) {
                    vec_defs.push(ContextDef {
                        qualified_name: h.qualified_name,
                        file_path: h.file_path,
                        start_line: h.start_line,
                        end_line: h.end_line,
                        node_id: h.node_id,
                    });
                }
            }
            if !vec_defs.is_empty() {
                // A multi-word conceptual query wants the mechanism, not just a
                // location: give the #1 hit a bounded body so the agent answers
                // in one call instead of rephrasing and re-searching.
                let conceptual = q.split_whitespace().count() >= CONTEXT_CONCEPTUAL_MIN_WORDS;
                return emit_context_vector_locators(
                    &store,
                    &project,
                    &freshness,
                    k,
                    lines,
                    json,
                    &vec_defs,
                    &span_root,
                    low_confidence,
                    conceptual,
                );
            }
        }
    }

    // 1. Symbol-name FTS hits (most precise). Resolve each to its node.
    for h in greppy_search::search_symbols_in_project(&store, &project, q, fetch)? {
        if defs.len() >= k {
            break;
        }
        if let Some(n) = store.get_node(h.node_id)? {
            if seen.insert(n.id) {
                defs.push(ContextDef {
                    qualified_name: n.qualified_name,
                    file_path: n.file_path,
                    start_line: n.start_line,
                    end_line: n.end_line,
                    node_id: Some(n.id),
                });
            }
        }
    }

    // 2. Semantic hits (paraphrase / related symbols).
    if defs.len() < k {
        for h in greppy_search::semantic_query(&store, q, None, Some(&project), fetch)? {
            if defs.len() >= k {
                break;
            }
            if seen.insert(h.node.id) {
                defs.push(ContextDef {
                    qualified_name: h.node.qualified_name,
                    file_path: h.node.file_path,
                    start_line: h.node.start_line,
                    end_line: h.node.end_line,
                    node_id: Some(h.node.id),
                });
            }
        }
    }

    // 3. Code-search hits resolved to their enclosing definition. This
    //    catches symbols a query only matches inside a body (where neither
    //    the symbol-name FTS nor the semantic signals fired).
    if defs.len() < k {
        let mut code_hits = greppy_search::search_code(&store, &project, q, fetch)?;
        if code_hits.is_empty() {
            code_hits = live_grep_code_hits(q, &span_root)?
                .into_iter()
                .take(fetch)
                .collect();
        }
        for h in code_hits {
            if defs.len() >= k {
                break;
            }
            // `location` is `file:line`; split on the LAST colon so a
            // path containing a colon is still parsed correctly.
            let Some((file, line_str)) = h.location.rsplit_once(':') else {
                continue;
            };
            let Ok(line) = line_str.parse::<i64>() else {
                continue;
            };
            if let Some(row) = greppy_search::definition_at(&store, Some(&project), file, line)? {
                if seen.insert(row.id) {
                    defs.push(ContextDef {
                        qualified_name: row.qualified_name,
                        file_path: row.file_path,
                        start_line: row.start_line,
                        end_line: row.end_line,
                        node_id: Some(row.id),
                    });
                }
            }
        }
    }

    emit_context_defs(
        &store, &project, &freshness, k, lines, json, &defs, &span_root,
    )
}

/// Native EmbeddingGemma vector fallback for `context` (D6). Returns
/// `Ok(Some(defs))` with the top vector hits when an embedding model is
/// configured, the index has current-generation vectors, and it is fresh;
/// returns `Ok(None)` (with a labeled stderr note) when the fallback cannot
/// run — no model configured, no indexed vectors, stale index, or the exact
/// scan would exceed its candidate guard. It NEVER errors on a missing model:
/// a research question just degrades to the current (lexical-only) behaviour.
///
/// Only reached for multi-word natural-language queries whose lexical union
/// was empty, so it never runs on `avoid_embedding` exact-name / graph
/// queries (see the call site).
pub(crate) fn context_vector_fallback(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    query: &str,
    fetch: usize,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<Option<(Vec<ContextVectorDef>, bool)>> {
    let cfg = embedding_config_for_required_use(embedding_args)?;

    let generation = current_graph_generation(store, root)?;
    if !embedding_generation_complete(store, project, generation, &cfg.model_id) {
        let root_path = resolve_root(root)?;
        let _ = spawn_background_embed(root, &cfg);
        let progress = embedding_progress_value(&root_path, &cfg, generation);
        eprintln!("{}", embedding_progress_text(&progress));
        return Ok(None);
    }
    let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
        project,
        &cfg.model_id,
        Some(generation),
        fetch,
    );
    let total = greppy_search::count_vector_search_scope(store, &scope)?;
    if total == 0 {
        eprintln!(
            "context: the completed semantic index contains no embeddable code spans for this project."
        );
        return Ok(None);
    }
    if !freshness_json_is_fresh(freshness) {
        eprintln!("{}", vector_stale_skip_message("context", freshness));
        return Ok(None);
    }
    let candidate_limit = vector_exact_candidate_limit()?;
    if let Some(limit) = vector_exact_scan_exceeds_limit(total, candidate_limit) {
        eprintln!(
            "{}",
            vector_exact_scan_skip_message("context", total, limit)
        );
        return Ok(None);
    }

    let query_vector = match embed_query_cached(&cfg, root, query) {
        Ok(query_vector) => query_vector,
        Err(e) => {
            log_embedding_skip_once("context", &e);
            return Ok(None);
        }
    };
    // P2b: over-fetch before the class prior below — re-ranking a set the
    // auxiliary stubs already saturated cannot surface the real code.
    scope.limit = fetch.saturating_mul(4).max(64);
    let mut hits = greppy_search::vector_search_exact(store, &query_vector, &scope)?;
    // P2b (spot forensics): tiny bench/test stubs and vendored/lock files
    // embed to near-uniform vectors and crowd out the real definitions on
    // vocabulary queries (zod: `packages/bench/*.ts zod3(){}` outranked
    // coerce.ts). Apply a deterministic class prior — a mild multiplicative
    // penalty for auxiliary paths and one-to-two-line spans — then re-rank.
    // Production code with a genuinely better score still wins; the prior
    // only breaks the near-ties the low-confidence header flags anyway.
    for h in &mut hits {
        let p = h.embedding.file_path.to_ascii_lowercase();
        let auxiliary = p.split('/').any(|seg| {
            matches!(
                seg,
                "test"
                    | "tests"
                    | "__tests__"
                    | "testing"
                    | "spec"
                    | "specs"
                    | "bench"
                    | "benches"
                    | "benchmark"
                    | "benchmarks"
                    | "example"
                    | "examples"
                    | "fixtures"
                    | "docs"
                    | "doc"
                    | "node_modules"
                    | "vendor"
                    | "third_party"
            )
        }) || p.ends_with(".lock")
            || p.ends_with("lock.yaml")
            || p.ends_with("lock.json")
            || p.ends_with(".md");
        if auxiliary {
            h.score *= 0.85;
        }
        if h.embedding.end_line.saturating_sub(h.embedding.start_line) < 2 {
            h.score *= 0.92;
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(fetch);
    // Forensics visibility (F2): env-driven vector use carries no
    // `--embedding-*` flag in the agent's command and the success path is
    // otherwise silent, so a control-class task could run vectors and
    // `forensics.py --enforce` would miss the hard-negative violation. Emit
    // ONE stderr line whose text contains a `candidate_uses_vector`
    // VECTOR_TRIGGER substring ("embeddinggemma" and "vector search" both
    // match forensics.py verbatim) so env-driven use is detectable.
    eprintln!(
        "context: vector search fallback used (embeddinggemma, {} hits)",
        hits.len()
    );
    // Confidence from the score MARGIN, not the absolute score (r042
    // forensics): a genuine hit separates clearly from the runner-up
    // (control-case margin ≈ 0.27) while a vocabulary-mismatch query returns
    // a near-tie of equally-plausible wrong candidates (margin ≈ 0.02) —
    // exactly the shape that sent an agent into a 39-call verify spiral
    // while the header still claimed "#1 is the most likely answer".
    let low_confidence = hits.len() >= 2 && (hits[0].score - hits[1].score) < 0.05;
    Ok(Some((
        hits.into_iter()
            .map(|h| ContextVectorDef {
                node_id: h.embedding.node_id,
                qualified_name: h.embedding.qualified_name,
                file_path: h.embedding.file_path,
                start_line: h.embedding.start_line,
                end_line: h.embedding.end_line,
            })
            .collect(),
        low_confidence,
    )))
}

/// Build a compact, graph-linked structural digest of the top semantic hit:
/// its signature (header), the key functions it calls (with their signatures,
/// from the graph's CALLS edges) and its return type — the body elided with a
/// `…` marker. This fuses the semantic hit with graph discovery so a conceptual
/// "how does X work" query is answered in ONE call, instead of a raw-body dump
/// (which drove the rephrase-and-re-search cost spiral) or a bare signature
/// (which made the agent re-query for the mechanism). Returns `None` when the
/// node carries no detail beyond its header, so the caller falls back.
pub(crate) fn context_top_digest(
    store: &greppy_store::Store,
    node: &greppy_store::Node,
    span_root: &std::path::Path,
) -> Option<String> {
    fn prop_trimmed<'a>(node: &'a greppy_store::Node, key: &str) -> Option<&'a str> {
        node.properties
            .get(key)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }
    fn cap(s: &str, n: usize) -> String {
        if s.chars().count() > n {
            format!("{}…", s.chars().take(n).collect::<String>())
        } else {
            s.to_string()
        }
    }

    let header = prop_trimmed(node, "signature")
        .map(str::to_string)
        .or_else(|| {
            read_source_line(span_root, &node.file_path, node.start_line as u32)
                .map(|s| s.trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| node.name.clone());

    // Key callees (from the graph's CALLS edges), each with its signature.
    let mut callees: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    if let Ok(steps) = greppy_search::callees_of(store, node.id) {
        for step in steps {
            let Some(n) = step.node else { continue };
            if n.id == node.id || !seen.insert(n.id) {
                continue;
            }
            let short = n
                .qualified_name
                .rsplit("::")
                .next()
                .unwrap_or(&n.qualified_name);
            let label = match n
                .properties
                .get("signature")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(sig) => cap(sig, 72),
                None => short.to_string(),
            };
            // The callee's own location AND line span, so the agent can open
            // and read exactly that function directly when it needs the detail
            // — a navigable map, not a dead-end name list.
            let loc = if n.file_path.is_empty() {
                String::new()
            } else if n.start_line > 0 && n.end_line >= n.start_line {
                format!("  [{}:{}-{}]", n.file_path, n.start_line, n.end_line)
            } else if n.start_line > 0 {
                format!("  [{}:{}]", n.file_path, n.start_line)
            } else {
                format!("  [{}]", n.file_path)
            };
            callees.push(format!("{label}{loc}"));
            if callees.len() >= CONTEXT_DIGEST_MAX_CALLEES {
                break;
            }
        }
    }

    let returns = prop_trimmed(node, "return_type").map(str::to_string);
    let doc = prop_trimmed(node, "doc")
        .map(|d| d.split('\n').next().unwrap_or(d).trim().to_string())
        .filter(|s| !s.is_empty());

    // Not worth a digest if there is nothing beyond the header.
    if callees.is_empty() && returns.is_none() && doc.is_none() {
        return None;
    }

    let mut out = String::new();
    out.push_str(&header);
    if let Some(d) = doc {
        out.push_str("\n    doc: ");
        out.push_str(&cap(&d, 120));
    }
    if !callees.is_empty() {
        // One callee per line, each with its own `file:line`, so the agent can
        // scan the mechanism and open any building-block function directly.
        out.push_str("\n    calls:");
        for c in &callees {
            out.push_str("\n      ");
            out.push_str(c);
        }
    }
    if let Some(rt) = returns {
        out.push_str("\n    returns: ");
        out.push_str(&rt);
    }
    let span = node.end_line - node.start_line;
    if span > 1 {
        out.push_str(&format!("\n    … [{span} lines elided]"));
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_json(
    store: &greppy_store::Store,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    limit: usize,
    line_numbers: bool,
    defs: &[ContextDef],
    root: &std::path::Path,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let mut span_truncated_count = 0usize;
    let mut source_unavailable_count = 0usize;
    let mut spans = Vec::new();
    for def in defs.iter().take(limit) {
        let span = read_span_with_meta(
            root,
            &def.file_path,
            def.start_line,
            def.end_line,
            CONTEXT_SPAN_CAP,
            line_numbers,
        );
        match span {
            Some(span) => {
                if span.truncated {
                    span_truncated_count += 1;
                }
                spans.push(serde_json::json!({
                    "qualified_name": &def.qualified_name,
                    "file_path": &def.file_path,
                    "start_line": def.start_line,
                    "end_line": def.end_line,
                    "source_available": true,
                    "source": span.text,
                    "total_lines": span.total_lines,
                    "shown_lines": span.shown_lines,
                    "omitted_lines": span.omitted_lines,
                    "truncated": span.truncated,
                }));
            }
            None => {
                source_unavailable_count += 1;
                spans.push(serde_json::json!({
                    "qualified_name": &def.qualified_name,
                    "file_path": &def.file_path,
                    "start_line": def.start_line,
                    "end_line": def.end_line,
                    "source_available": false,
                    "source": null,
                    "total_lines": null,
                    "shown_lines": 0,
                    "omitted_lines": null,
                    "truncated": false,
                }));
            }
        }
    }
    let truncated = span_truncated_count > 0;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "context",
            "status": status,
            "project": project,
            "fresh": freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": incomplete_providers.is_empty(),
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "limit": limit,
            "line_numbers": line_numbers,
            "span_cap_lines": CONTEXT_SPAN_CAP,
            "candidate_total_kind": "top_k_only",
            "shown": spans.len(),
            "source_unavailable_count": source_unavailable_count,
            "span_truncated_count": span_truncated_count,
            "truncated": truncated,
            "spans": spans,
        }))
        .map_err(|e| Error::Invalid(format!("serialize context JSON: {e}")))?
    );
    Ok(())
}

pub(crate) fn context_stale_skip_message(freshness: &serde_json::Value) -> String {
    format!(
        "context: {} — source-span lookup skipped, no stale indexed spans emitted ({})",
        crate::STALE_REMEDIATION,
        stale_freshness_reason(freshness)
    )
}
