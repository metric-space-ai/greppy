//! Shared emitters: the lines and JSON envelopes every command shares.
//!
//! Split out of `lib.rs`; `use super::*` keeps every private helper there
//! reachable, and no behaviour changes.

use super::*;

pub(crate) fn print_gc_report(report: &greppy_core::cache::GcReport, json: bool) -> Result<()> {
    let value = serde_json::json!({
        "dry_run": report.dry_run,
        "throttled": report.throttled,
        "scanned_bytes": report.scanned_bytes,
        "removed_bytes": report.removed_bytes,
        "locked_bytes": report.locked_bytes,
        "removed": report.removed,
        "skipped_locked": report.skipped_locked,
    });
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|error| Error::Invalid(format!("serialize cache GC: {error}")))?
        );
    } else {
        println!(
            "cache GC: scanned={} removed={} locked={} dry_run={}",
            report.scanned_bytes, report.removed_bytes, report.locked_bytes, report.dry_run
        );
        for path in &report.removed {
            println!("removed {}", path.display());
        }
        for path in &report.skipped_locked {
            println!("locked {}", path.display());
        }
    }
    Ok(())
}

pub(crate) fn print_symbol_miss_guidance(store: &greppy_store::Store, project: &str, query: &str) {
    println!("symbol not found: `{query}`");
    for suggestion in symbol_miss_suggestions(store, project, query) {
        println!("suggestion: `{suggestion}`");
    }
    println!("try: greppy search-symbols {}", shell_example_arg(query));
    println!("try: greppy semantic-search {}", shell_example_arg(query));
}

/// Emit a truncation footer for the navigation commands when more results
/// exist than were printed. Centralised so navigation commands word it identically.
pub(crate) fn print_nav_more_footer(total: usize, shown: usize) {
    if total > shown {
        // Report the TRUE total so the agent can answer "how many" from this
        // line alone (e.g. "called by 72 functions"). Deliberately frame
        // `--all` as rarely needed: the F1 forensics showed agents reflexively
        // re-running with `--all` and flooding their own context when the
        // count + sample already answered the question.
        // D2: when serving from a stale index, say so in the count itself
        // so the total is never mistaken for the current state of the tree.
        let stale_note = if serving_stale() {
            " (as of last index)"
        } else {
            ""
        };
        println!(
            "… and {} more ({} shown of {} total{stale_note} — this sample usually answers the question; pass --all only if you truly need every site)",
            total - shown,
            shown,
            total
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_expand_pack_best_effort(
    store: &greppy_store::Store,
    project: &str,
    command: &str,
    query: &str,
    graph_generation: u64,
    summary: serde_json::Value,
    payload_text: String,
    payload_json: Option<serde_json::Value>,
) -> Option<ExpandHandle> {
    if payload_text.trim().is_empty() {
        return None;
    }
    let summary_text = expand_summary_text(&summary);
    let pack = greppy_store::NewExpandPack {
        project: project.to_string(),
        command: command.to_string(),
        query: query.to_string(),
        graph_generation,
        summary_json: summary,
        payload_text,
        payload_json,
        ttl_secs: expand_ttl_secs(),
    };
    store.insert_expand_pack(&pack).ok().map(|id| ExpandHandle {
        id,
        summary: summary_text,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_nav_expand_pack(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    query: &str,
    total: usize,
    rows: &[ExpandEvidenceNode<'_>],
) -> Option<ExpandHandle> {
    if rows.is_empty() {
        return None;
    }
    let root_path = resolve_root(root).ok()?;
    let limit = rows.len().min(EXPAND_NAV_EVIDENCE_LIMIT);
    let mut text = String::new();
    text.push_str(&format!("# evidence pack: {command} {query}\n"));
    text.push_str(&format!("# rows: {} shown of {} total\n\n", limit, total));
    let mut callsite_count = 0usize;
    let mut json_rows = Vec::new();
    for row in rows.iter().take(limit) {
        callsite_count += row.site_lines.len();
        append_node_evidence(&mut text, &root_path, row.node, &row.title, &row.site_lines);
        json_rows.push(serde_json::json!({
            "title": row.title,
            "qualified_name": &row.node.qualified_name,
            "label": &row.node.label,
            "file_path": &row.node.file_path,
            "start_line": row.node.start_line,
            "end_line": row.node.end_line,
            "site_lines": &row.site_lines,
            "extra": &row.extra_json,
        }));
    }
    let summary_text = if callsite_count == 0 {
        format!("{limit} spans")
    } else {
        format!("{limit} spans, {callsite_count} callsites")
    };
    let summary = serde_json::json!({
        "text": summary_text,
        "spans": limit,
        "callsites": callsite_count,
        "total": total,
    });
    let payload_json = serde_json::json!({
        "command": command,
        "query": query,
        "total": total,
        "shown": limit,
        "hits": json_rows,
    });
    insert_expand_pack_best_effort(
        store,
        project,
        command,
        query,
        current_graph_generation_or_zero(store, root),
        summary,
        text,
        Some(payload_json),
    )
}

pub(crate) fn insert_semantic_vector_expand_pack(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    query: &str,
    graph_generation: u64,
    hits: &[greppy_store::VectorSearchHit],
) -> Option<ExpandHandle> {
    if hits.is_empty() {
        return None;
    }
    let root_path = resolve_root(root).ok()?;
    let purposes = semantic_vector_purposes(store, root, hits, false)
        .ok()
        .flatten()?;
    let limit = purposes.len();
    if limit == 0 {
        return None;
    }
    let mut text = String::new();
    text.push_str(&format!("# evidence pack: semantic-search {query}\n"));
    text.push_str(&format!(
        "# spans: {limit} further of {} retrieved hits\n\n",
        hits.len()
    ));
    let mut json_rows = Vec::new();
    for (idx, purpose) in purposes.iter().enumerate() {
        let hit = hits
            .iter()
            .find(|hit| hit.embedding.id == purpose.embedding_id)?;
        let title = format!("{:.3} {}", hit.score, purpose.signature);
        append_span_evidence(
            &mut text,
            &root_path,
            &title,
            &purpose.file_path,
            purpose.start_line,
            purpose.end_line,
            if idx == 0 {
                CONTEXT_SPAN_CAP
            } else {
                CODE_SPAN_CAP
            },
        );
        json_rows.push(serde_json::json!({
            "score": hit.score,
            "qualified_name": &hit.embedding.qualified_name,
            "file_path": &purpose.file_path,
            "start_line": purpose.start_line,
            "end_line": purpose.end_line,
            "signature": &purpose.signature,
            "content_sha256": &hit.embedding.content_sha256,
            "graph_generation": hit.embedding.graph_generation,
        }));
    }
    let summary = serde_json::json!({
        "text": format!("{limit} further hits"),
        "spans": limit,
        "callsites": 0,
        "total": hits.len(),
    });
    let payload_json = serde_json::json!({
        "command": "semantic-search",
        "mode": "vector",
        "query": query,
        "further_hits": limit,
        "hits": json_rows,
    });
    insert_expand_pack_best_effort(
        store,
        project,
        "semantic-search",
        query,
        graph_generation,
        summary,
        text,
        Some(payload_json),
    )
}

pub(crate) fn insert_impact_edge_meta(obj: &mut serde_json::Value, spec: &ImpactEdgeSpec<'_>) {
    if let Some(map) = obj.as_object_mut() {
        map.insert("edge_type".into(), serde_json::json!(spec.mode));
        map.insert("edge_types".into(), serde_json::json!(&spec.edge_types));
    }
}

pub(crate) fn emit_semantic_backend_unavailable(
    project: &str,
    query: &str,
    paths: &[String],
    root: Option<&str>,
    json: bool,
    detail: &str,
) -> Result<()> {
    let next = semantic_fallback_commands(query, paths, root);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
                "command": "semantic-search",
                "mode": "vector",
                "status": "backend_unavailable",
                "project": project,
                "query": query,
                "reason": "asset_missing",
                "detail": detail,
                "retryable": false,
                "query_tokens": semantic_fallback_tokens(query),
                "next": next,
                "total_exact": 0,
                "shown": 0,
                "omitted": 0,
                "truncated": false,
                "hits": [],
            }))
            .map_err(|error| {
                Error::Invalid(format!("serialize semantic unavailable JSON: {error}"))
            })?
        );
    } else {
        println!("semantic backend unavailable (asset missing) — {detail}");
        print_semantic_fallback_commands(query, paths, root);
    }
    Ok(())
}

/// Print the `--code` source span for a single resolved node, using the
/// shared cap and the standard skip-on-failure semantics. Emitted
/// indented under the node's `file:line` line so the structure stays
/// readable when many nodes are printed.
pub(crate) fn print_code_span(root: &std::path::Path, node: &greppy_store::Node, cap: usize) {
    if let Some(span) = read_span(
        root,
        &node.file_path,
        node.start_line,
        node.end_line,
        cap,
        false,
    ) {
        print_code_span_text(&span);
    }
}

pub(crate) fn print_code_span_text(span: &str) {
    for line in span.lines() {
        println!("    {line}");
    }
}

pub(crate) fn emit_edit_outcome(
    outcome: EditResult<EditRecord>,
    json: bool,
    report: Option<String>,
) -> Result<i32> {
    let report_path = report.as_deref();
    match outcome {
        Ok(record) => {
            if let Some(path) = report_path {
                write_edit_report(path, &edit_record_json(&record, true, report_path))?;
            }
            if json {
                let compact = edit_record_json(&record, false, report_path);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&compact)
                        .map_err(|error| Error::Invalid(format!("serialize edit record: {error}")))?
                );
            } else {
                // Success is one bare line: verb and address. The text that
                // landed is the text the caller sent (CAS); handles appear only
                // where the spec orders them. A dry run must not say "applied"
                // — it wrote nothing, and a receipt that overstates is the
                // worst output this tool can produce.
                let word = if record.published { "applied" } else { "would apply" };
                if let Some(headline) = &record.headline {
                    println!("{headline}");
                } else {
                    for (index, file) in record.files.iter().enumerate() {
                        match record.span {
                            Some((first, last)) if index == 0 && first == last => {
                                println!("{word} {file}:{first}");
                            }
                            Some((first, last)) if index == 0 => {
                                println!("{word} {file}:{first}-{last}");
                            }
                            _ => println!("{word} {file}"),
                        }
                    }
                }
                for note in &record.notes {
                    println!("{note}");
                }
                if let Some(diagnostics) = &record.diagnostics {
                    for diagnostic in diagnostics {
                        println!("{diagnostic}");
                    }
                }
            }
            Ok(0)
        }
        Err(refusal) => {
            let value = edit_refusal_json(&refusal, report_path);
            if let Some(path) = report_path {
                write_edit_report(path, &value)?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())
                );
            } else {
                // stdout carries answers; a refusal is not one. A caller that
                // pipes an edit into the next command must receive nothing
                // rather than a line of prose it would then act on.
                eprintln!("{}", refusal.message);
            }
            Ok(refusal.exit)
        }
    }
}

pub(crate) fn finish_edit(
    certificate: greppy_edit::Certificate,
    report_path: Option<String>,
    json: bool,
    root: Option<&str>,
    root_path: &std::path::Path,
) -> Result<i32> {
    let mut certificate = certificate;
    if certificate.published {
        // close the read->edit->read loop: refresh the store so the next
        // read/graph query addresses the edited file without a manual
        // reindex. index() is incremental from the second run, so this
        // touches only the changed file. A refresh failure downgrades the
        // flag, never the edit (the workspace write already happened).
        let refreshed = std::env::current_exe()
            .ok()
            .and_then(|exe| {
                std::process::Command::new(exe)
                    .arg("--root")
                    .arg(root_path)
                    .arg("index")
                    .arg(root_path)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .ok()
            })
            .map(|status| status.success())
            .unwrap_or(false);
        for op in &mut certificate.operations {
            op.store_refreshed = refreshed;
        }
    }
    let handles = certificate_operation_handles(&certificate, root_path);
    let mut full_value = serde_json::to_value(&certificate)
        .map_err(|e| Error::Invalid(format!("serialize full certificate: {e}")))?;
    certificate_operation_extras(&mut full_value, &certificate.transaction_id, &handles);
    let full = serde_json::to_string_pretty(&full_value)
        .map_err(|e| Error::Invalid(format!("serialize full certificate: {e}")))?;
    let mut report_written = None;
    if let Some(path) = report_path {
        std::fs::write(&path, format!("{full}\n")).map_err(|source| Error::Io {
            context: format!("write report {path}"),
            source,
        })?;
        report_written = Some(path);
    }

    let expand = if let (Ok(store), Ok(project)) =
        (open_default_store_query_writer(root), project_for(root))
    {
        let summary = serde_json::json!({
            "text": format!(
                "edit certificate: {} operation(s), status {}",
                certificate.operations.len(),
                edit_status_name(certificate.status)
            ),
            "transaction_id": &certificate.transaction_id,
            "status": edit_status_name(certificate.status),
            "operations": certificate.operations.len(),
        });
        insert_expand_pack_best_effort(
            &store,
            &project,
            "edit",
            &certificate.transaction_id,
            current_graph_generation_or_zero(&store, root),
            summary,
            format!("{full}\n"),
            serde_json::to_value(&certificate).ok(),
        )
    } else {
        None
    };

    if let Some(expand) = &expand {
        insert_expand_alias_best_effort(root, &certificate.transaction_id, &expand.id);
    }
    if json {
        // The stdout form, not the archival one: it carries `exit_code` and
        // drops evidence that only `--report` and `expand` need. Printing the
        // full Serialize form here silently dropped `exit_code` from the
        // documented certificate contract.
        let stdout_form = certificate
            .to_compact_json_pretty()
            .map_err(|e| Error::Invalid(format!("serialize certificate: {e}")))?;
        // One caller-side branch for every edit verb: a certificate that did
        // not apply carries the same `error.code` the grammar verbs emit.
        let mut value: serde_json::Value = serde_json::from_str(&stdout_form)
            .map_err(|e| Error::Invalid(format!("re-read certificate: {e}")))?;
        certificate_operation_extras(&mut value, &certificate.transaction_id, &handles);
        if let Some(root) = value.as_object_mut() {
            // The evidence stdout omits is omitted, not replaced by a stand-in
            // string that reads like a diff to anything parsing the field.
            if let Some(operations) = root.get_mut("operations").and_then(|v| v.as_array_mut()) {
                for operation in operations {
                    if let Some(operation) = operation.as_object_mut() {
                        operation.remove("unified_diff");
                    }
                }
            }
            if let Some(path) = &report_written {
                root.insert("report_path".into(), serde_json::json!(path));
            }
            if certificate.exit_code() != 0 {
                let message = certificate_refusal_message(&certificate)
                    .or_else(|| certificate.compact_failure_diagnosis())
                    .unwrap_or_else(|| format!("edit {}", edit_status_name(certificate.status)));
                root.insert(
                    "error".into(),
                    serde_json::json!({
                        "code": certificate_refusal_code(&certificate),
                        "message": message,
                    }),
                );
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize certificate: {e}")))?
        );
    } else {
        render_compact_edit_certificate(&certificate, root_path);
    }
    Ok(certificate.exit_code())
}

pub(crate) fn insert_expand_alias_best_effort(root: Option<&str>, alias: &str, id: &str) {
    let Some(path) = expand_alias_path(root, alias) else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_ok() {
        let _ = std::fs::write(path, id);
    }
}

pub(crate) fn print_inference_registry(registry: &greppy_embed_native::InferenceBackendRegistry) {
    println!(
        "inference: preference={} explicit={} selected={} device={}",
        registry.preference,
        registry.explicit,
        registry
            .selected_backend
            .map(greppy_embed_native::BackendKind::as_str)
            .unwrap_or("none"),
        registry.selected_device_id.as_deref().unwrap_or("none")
    );
    for probe in &registry.probes {
        println!(
            "  backend {} compiled={} available={} score={} abi={}{}",
            probe.backend.as_str(),
            probe.compiled,
            probe.available,
            probe.score,
            probe.abi_version,
            probe
                .reason
                .as_deref()
                .map(|reason| format!(" reason={reason}"))
                .unwrap_or_default()
        );
        for device in &probe.devices {
            let memory = match (device.memory_free, device.memory_total) {
                (Some(free), Some(total)) => format!(" memory_free={free} memory_total={total}"),
                (None, Some(total)) => format!(" memory_total={total}"),
                _ => String::new(),
            };
            println!(
                "    device {} {}{}{}",
                device.id,
                device.name,
                memory,
                device
                    .rejection_reason
                    .as_deref()
                    .map(|reason| format!(" rejected={reason}"))
                    .unwrap_or_default()
            );
        }
    }
}

pub(crate) fn print_inference_daemons(daemons: &serde_json::Value) {
    let Some(daemons) = daemons.as_object() else {
        return;
    };
    for (name, status) in daemons {
        let state = status
            .get("state")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let endpoint = status
            .get("endpoint")
            .and_then(serde_json::Value::as_str)
            .map(|endpoint| format!(" endpoint={endpoint}"))
            .unwrap_or_default();
        let error = status
            .get("last_error")
            .and_then(serde_json::Value::as_str)
            .map(|error| format!(" error={error}"))
            .unwrap_or_default();
        println!("  daemon {name} state={state}{endpoint}{error}");
    }
}

pub(crate) fn print_search_code_no_matches(query: &str, fixed: bool, path_filters: &QueryPathFilters) {
    println!("(no matches)");
    println!(
        "query_interpreted_as: {}",
        if fixed { "literal" } else { "regex" }
    );
    if path_filters.is_empty() {
        println!("path_filters: <none>");
    } else {
        println!("path_filters: {}", path_filters.shown());
    }
    if fixed
        && query
            .chars()
            .any(|character| ".^$*+?()[]{}|\\".contains(character))
    {
        println!("hint: regex metacharacters are literal because --fixed was supplied");
        let mut retry = format!("greppy search-code {}", shell_example_arg(query));
        for filter in &path_filters.filters {
            retry.push(' ');
            retry.push_str(&shell_example_arg(&filter.shown));
        }
        println!("try without --fixed: {retry}");
    }
}

pub(crate) fn print_search_code_entries(entries: &[SearchCodeEntry]) {
    for entry in entries {
        match entry {
            SearchCodeEntry::Unenclosed(hit) => {
                println!("{}  {}", hit.location, clamp_snippet(&hit.text));
            }
            SearchCodeEntry::Definition(definition) => {
                println!(
                    "{} {}:{}-{}",
                    definition.qualified_name,
                    definition.file,
                    definition.start_line,
                    definition.end_line
                );
                let width = definition.end_line.to_string().len();
                for (offset, line) in definition.source.lines().enumerate() {
                    println!(
                        "  {:>width$} | {line}",
                        definition.start_line + offset as i64
                    );
                }
                println!("  handle: {}", definition.handle);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_search_code_results_with_format(
    store: &greppy_store::Store,
    query: &str,
    project: &str,
    status: &str,
    index_freshness: Option<&serde_json::Value>,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
    path_filters: &QueryPathFilters,
    root_path: &std::path::Path,
    json: bool,
    no_code: bool,
    fixed: bool,
    resolve_definitions: bool,
) -> Result<()> {
    if !json {
        if hits.is_empty() {
            print_search_code_no_matches(query, fixed, path_filters);
        } else if no_code {
            for hit in hits {
                println!("{}  {}", hit.location, clamp_snippet(&hit.snippet));
            }
        } else {
            let entries =
                search_code_entries(store, project, root_path, hits, resolve_definitions)?;
            print_search_code_entries(&entries);
        }
        return Ok(());
    }

    let incomplete_providers = incomplete_provider_json(store, project)?;
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = if no_code {
        hits.iter()
            .map(|hit| {
                serde_json::json!({
                    "location": hit.location,
                    "rank": hit.rank,
                    "snippet": clamp_snippet(&hit.snippet).as_ref(),
                })
            })
            .collect::<Vec<_>>()
    } else {
        search_code_entries(store, project, root_path, hits, resolve_definitions)?
            .iter()
            .map(search_code_entry_json)
            .collect::<Vec<_>>()
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": status,
            "query": query,
            "pattern_mode": if fixed { "fixed" } else { "regex" },
            "project": project,
            "path_filters": path_filters.json_value(),
            "backend": "live-filesystem",
            "fresh": true,
            "freshness": if resolve_definitions {
                index_freshness.cloned().unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            },
            "index_freshness": if resolve_definitions {
                serde_json::Value::Null
            } else {
                index_freshness.cloned().unwrap_or(serde_json::Value::Null)
            },
            "provider_complete": incomplete_providers.is_empty(),
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|error| Error::Invalid(format!("serialize search-code JSON: {error}")))?
    );
    Ok(())
}

pub(crate) fn print_semantic_vector_hit(
    hit: &greppy_store::VectorSearchHit,
    purposes: Option<&[SemanticVectorPurpose]>,
) {
    let loc = vector_hit_loc(hit);
    if let Some(purpose) = vector_purpose_for_hit(purposes, hit) {
        println!("{}", purpose.display_loc);
        println!("    {}", purpose.signature);
        for bullet in &purpose.bullets {
            println!("        {bullet}");
        }
    } else {
        println!("{loc}");
    }
    println!();
}

/// Render the resolved `context` definitions — shared by the exact-name
/// fast path and the general resolution path so both emit identical
/// JSON / span output. Returns exit 0 when at least one definition was
/// resolved, 1 when the set is empty.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_context_defs(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    k: usize,
    lines: bool,
    json: bool,
    defs: &[ContextDef],
    span_root: &std::path::Path,
) -> Result<i32> {
    if defs.is_empty() {
        if json {
            context_json(
                store,
                project,
                "ok",
                Some(freshness),
                k,
                lines,
                &[],
                span_root,
            )?;
        } else {
            println!("(no matches)");
        }
        return Ok(1);
    }

    if json {
        context_json(
            store,
            project,
            "ok",
            Some(freshness),
            k,
            lines,
            defs,
            span_root,
        )?;
        return Ok(0);
    }

    let mut printed = 0usize;
    for def in defs.iter().take(k) {
        match read_span(
            span_root,
            &def.file_path,
            def.start_line,
            def.end_line,
            CONTEXT_SPAN_CAP,
            lines,
        ) {
            Some(span) => {
                let display_name = display_context_def_name(store, def);
                println!(
                    "== {} ({}:{}-{}) ==",
                    display_name, def.file_path, def.start_line, def.end_line
                );
                print!("{span}");
                println!();
                printed += 1;
            }
            // Span unreadable (missing file / stale lines) — skip the body
            // but still surface the pointer so the agent is not left blind.
            None => {
                let display_name = display_context_def_name(store, def);
                println!(
                    "== {} ({}:{}-{}) == (source unavailable)",
                    display_name, def.file_path, def.start_line, def.end_line
                );
                println!();
            }
        }
    }

    // Exit 0 as long as we resolved at least one definition; the
    // per-span unavailability is reported inline above.
    let _ = printed;
    Ok(0)
}

/// Emit the exact-name / show-definition result as a LEAN locator (Z3):
/// for each resolved definition print the compact
/// `== qname (file:start-end) ==` header followed by ONLY the definition's
/// first line — its signature / def line — not the whole body. This is the
/// grep-shaped answer to a "find the definition site of X" lookup: it
/// gives the file:line and the signature, matching a single `grep -rn`
/// def line in byte cost, so greppy stays grep-competitive on literal
/// find-definition tasks (contract Z3). Only exact-name bare-identifier
/// queries reach this path; natural-language / multi-word research queries
/// still take the rich, full-body union path, and `greppy brief <X>`
/// still prints the full body plus callers/callees for a deeper look.
///
/// JSON mode keeps the existing structured def-span metadata (a separate,
/// machine-readable consumer) — only the human/text output is leaned out,
/// which is what the token-cost comparison measures.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_context_locators(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    k: usize,
    lines: bool,
    json: bool,
    defs: &[ContextDef],
    span_root: &std::path::Path,
) -> Result<i32> {
    if json {
        // Structured consumers still get the full def-span metadata.
        return emit_context_defs(store, project, freshness, k, lines, json, defs, span_root);
    }

    if defs.is_empty() {
        println!("(no matches)");
        return Ok(1);
    }

    for def in defs.iter().take(k) {
        let display_name = display_context_def_name(store, def);
        println!(
            "== {} ({}:{}-{}) ==",
            display_name, def.file_path, def.start_line, def.end_line
        );
        // Only the FIRST line of the span — the signature / def line.
        // `read_span` (via `read_span_with_meta`) computes `total_lines` from
        // `definition_end_idx()` (the whole body) even though we ask for a
        // 1-line cap, so its text is `<sig>\n… (truncated, N more line(s))\n`.
        // That truncation note is dead weight for a lean Z3 locator (~25-40
        // extra bytes/hit). Take ONLY the first line (mirroring
        // `plus_first_line`'s `.lines().next()` guard) rather than modifying
        // the shared `read_span_with_meta`, whose full-body consumers rely on
        // the note.
        if let Some(sig) = read_span(
            span_root,
            &def.file_path,
            def.start_line,
            def.start_line,
            1,
            lines,
        )
        .and_then(|span| span.lines().next().map(str::to_string))
        {
            println!("{sig}");
        }
    }
    Ok(0)
}

/// Emit the vector-fallback result as LEAN, TRUST-BUILDING semantic locators.
///
/// The context vector fallback fires for a conceptual natural-language query
/// that names no symbol by exact name ("which routine converts X into Y") — the
/// answer is a *location*, so this prints the top-N (`CONTEXT_VECTOR_LEAN_TOP_N`)
/// semantic matches as grep-shaped locators — `== qname (file:start-end) ==`
/// plus the def's own signature line — exactly like the Z3 `emit_context_locators`
/// lean form, NOT the old k=6 full-body union that made the vectors' quality win
/// a token LOSS (r041: 5-6 KB, agent iterates because it can't tell which of six
/// bodies is the answer).
///
/// A single SHORT header precedes the locators, telling the agent these are
/// ranked semantic matches (most-relevant first). The header is deliberately
/// terse — the H2 slim lesson is that a verbose hedge backfires (a 22-token
/// hedge doubled outputs and was reverted), so this is one line, no per-hit
/// caveats.
///
/// JSON mode keeps the structured def-span metadata (a separate machine
/// consumer) via `emit_context_locators`; only the human/text output is leaned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_context_vector_locators(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    k: usize,
    lines: bool,
    json: bool,
    defs: &[ContextDef],
    span_root: &std::path::Path,
    low_confidence: bool,
    conceptual: bool,
) -> Result<i32> {
    if json {
        // Structured consumers still get the full locator metadata; the JSON
        // shape must not diverge between the exact and vector lean paths.
        return emit_context_locators(store, project, freshness, k, lines, json, defs, span_root);
    }

    if defs.is_empty() {
        println!("(no matches)");
        return Ok(1);
    }

    // One short line, most-relevant-first.
    // EXCEPT when the top scores nearly tie (vocabulary-mismatch queries):
    // claiming "#1 is the most likely answer" over a near-tie of plausible
    // wrong candidates sent an agent into a 39-call verify spiral (r042).
    // The low-confidence line is a TRUE signal (the margin really is ~0), so
    // it does not violate the no-false-hedges rule — it saves the agent from
    // serially disproving candidates the ranking itself cannot separate.
    // Show the #1 structural digest for any conceptual query — confident OR
    // near-tie. The near-tie case is exactly where agents used to rephrase
    // and re-search (the dominant cost-loss spiral). The digest exposes the
    // evidence without giving procedural instructions.
    // A short/locate query (< min words) stays sig-only, protecting "where is X".
    let show_top_body = conceptual;
    if low_confidence {
        println!(
            "# semantic candidates (top scores are close). The #1 call map is shown below; #2/#3 locators follow."
        );
    } else if show_top_body {
        println!(
            "# top semantic matches (most relevant first). The #1 call map is shown below; additional locators follow."
        );
    } else {
        println!("# top semantic matches (most relevant first).");
    }
    for (idx, def) in defs.iter().take(CONTEXT_VECTOR_LEAN_TOP_N).enumerate() {
        let display_name = display_context_def_name(store, def);
        println!(
            "== {} ({}:{}-{}) ==",
            display_name, def.file_path, def.start_line, def.end_line
        );
        if show_top_body && idx == 0 {
            // #1 of a conceptual query: a graph-linked structural digest —
            // signature (header) + the functions it calls (with their
            // signatures, from the graph's CALLS edges) + return type, body
            // elided — so the agent gets the mechanism in ONE call. Falls back
            // to a bounded raw body when the node carries no graph detail.
            let digest = def
                .node_id
                .and_then(|id| store.get_node(id).ok().flatten())
                .and_then(|node| context_top_digest(store, &node, span_root));
            if let Some(d) = digest {
                println!("{d}");
            } else if let Some(body) = read_span(
                span_root,
                &def.file_path,
                def.start_line,
                def.end_line,
                CONTEXT_TOP1_BODY_LINES,
                lines,
            ) {
                println!("{body}");
            }
        } else if let Some(sig) = read_span(
            span_root,
            &def.file_path,
            def.start_line,
            def.start_line,
            1,
            lines,
        )
        .and_then(|span| span.lines().next().map(str::to_string))
        {
            // Only the FIRST line of the span — the signature / def line (mirrors
            // the Z3 lean form: drop the "N more line(s)" truncation note, keep
            // the bare signature).
            println!("{sig}");
        }
    }
    Ok(0)
}

pub(crate) fn print_semantic_fallback_commands(query: &str, paths: &[String], root: Option<&str>) {
    for command in semantic_fallback_commands(query, paths, root) {
        println!("try: {command}");
    }
}

pub(crate) fn finish_output_capture(spec: &OutputBudgetSpec, exit_code: u8) {
    use std::io::Write as _;

    let captured = OUTPUT_CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default());
    let rendered = if spec.json {
        budget_json_output(&captured, spec).unwrap_or(captured)
    } else {
        budget_text_output(&captured, spec, exit_code)
    };
    let _ = std::io::stdout().lock().write_all(&rendered);
}
