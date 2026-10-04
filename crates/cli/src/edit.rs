//! The edit grammar and the verbs that carry it out.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

pub(crate) fn dispatch_edit(command: EditCommand, json: bool, root: Option<&str>) -> Result<i32> {
    match dispatch_edit_inner(command, json, root) {
        Err(error @ Error::Invalid(_)) => {
            if json {
                let refusal = EditRefusal::new("INVALID_REQUEST", error.to_string(), 20);
                println!("{}", edit_refusal_json(&refusal, None));
            } else {
                eprintln!("greppy: {error}");
            }
            Ok(20)
        }
        result => result,
    }
}

pub(crate) fn dispatch_edit_inner(
    command: EditCommand,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let file_base = resolve_file_operand_base(root, &root_path);
    // Symbol selectors depend on the structural graph. Heal workspace drift
    // before taking the edit transaction lock: structural publication owns its
    // own workspace-store writer locks, and waiting for it while holding the
    // edit journal lock would invert the transaction order. The resolver still
    // re-reads the selected file and publishes with its existing CAS checks.
    if matches!(
        &command,
        EditCommand::Replace { .. } | EditCommand::Delete { .. } | EditCommand::Rename { .. }
    ) {
        let mut store = open_default_store_query_writer(root)?;
        maybe_reindex_stale(&mut store, root)?;
        let project = project_for(root)?;
        if let FreshnessServe::Refuse(freshness) =
            freshness_serve_decision_with_policy(&store, root, &project, true, false, true)
        {
            return Err(Error::Index(indexed_stale_skip_message(
                "symbol edit",
                &freshness,
            )));
        }
    }
    // All grammar verbs share pending.json and the undo stack. Hold one
    // workspace-store lock across planning, publication, rollback and close;
    // file-level CAS alone cannot protect those shared transaction records.
    // A dry run must remain free of journal/lock side effects.
    let dry_run = match &command {
        EditCommand::Replace { dry_run, .. }
        | EditCommand::ReplaceText { dry_run, .. }
        | EditCommand::ReplaceLines { dry_run, .. }
        | EditCommand::ReplaceSpan { dry_run, .. }
        | EditCommand::Write { dry_run, .. }
        | EditCommand::Delete { dry_run, .. }
        | EditCommand::DeleteLines { dry_run, .. }
        | EditCommand::InsertLines { dry_run, .. }
        | EditCommand::Rename { dry_run, .. }
        | EditCommand::Undo { dry_run, .. }
        | EditCommand::Patch { dry_run, .. } => *dry_run,
    };
    let _transaction_lock = if dry_run {
        None
    } else {
        let journal = ensured_workspace_store_path(&root_path)?.with_file_name(EDIT_JOURNAL_DIR);
        Some(acquire_edit_transaction_lock(&journal)?)
    };
    Ok(dispatch_edit_grammar(command, json, root, &root_path, &file_base)?.0)
}

fn acquire_edit_transaction_lock(
    journal: &std::path::Path,
) -> Result<greppy_core::cache::FileLock> {
    greppy_core::cache::acquire_named_lock_in(
        journal,
        "transaction",
        greppy_core::cache::LockMode::Exclusive,
        true,
    )
    .map_err(|error| Error::io("acquire edit transaction lock", error))?
    .ok_or_else(|| Error::Lock(
        "another Greppy edit is active for this workspace; nothing written. Wait for that edit to finish, re-read the affected files, then retry".into(),
    ))
}

pub(crate) fn edit_sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// A minimal line-based unified diff for the full record. The byte ranges are
/// the precise answer; this is the readable view of the same change.
pub(crate) fn edit_unified_diff(path: &str, before: &[u8], after: &[u8]) -> String {
    let before_text = String::from_utf8_lossy(before);
    let after_text = String::from_utf8_lossy(after);
    let old_lines: Vec<&str> = before_text.lines().collect();
    let new_lines: Vec<&str> = after_text.lines().collect();
    let mut head = 0usize;
    while head < old_lines.len() && head < new_lines.len() && old_lines[head] == new_lines[head] {
        head += 1;
    }
    let mut tail = 0usize;
    while tail < old_lines.len() - head
        && tail < new_lines.len() - head
        && old_lines[old_lines.len() - 1 - tail] == new_lines[new_lines.len() - 1 - tail]
    {
        tail += 1;
    }
    let old_span = &old_lines[head..old_lines.len() - tail];
    let new_span = &new_lines[head..new_lines.len() - tail];
    let mut diff = format!("--- a/{path}\n+++ b/{path}\n");
    diff.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        head + 1,
        old_span.len(),
        head + 1,
        new_span.len()
    ));
    for line in old_span {
        diff.push('-');
        diff.push_str(line);
        diff.push('\n');
    }
    for line in new_span {
        diff.push('+');
        diff.push_str(line);
        diff.push('\n');
    }
    diff
}

pub(crate) fn edit_line_count(content: &[u8]) -> usize {
    if content.is_empty() {
        return 0;
    }
    let newlines = content.iter().filter(|byte| **byte == b'\n').count();
    if content.ends_with(b"\n") {
        newlines
    } else {
        newlines + 1
    }
}

pub(crate) fn edit_line_of_offset(content: &[u8], offset: usize) -> usize {
    content[..offset.min(content.len())]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}

/// The 1-based inclusive line span a written region occupies.
pub(crate) fn edit_span_lines(content: &[u8], start: usize, length: usize) -> (usize, usize) {
    let first = edit_line_of_offset(content, start);
    if length == 0 {
        return (first, first);
    }
    let last = first
        + content[start..start + length - 1]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count();
    (first, last)
}

pub(crate) fn edit_strip_trailing_newline(content: &[u8], range: (usize, usize)) -> (usize, usize) {
    let (start, mut end) = range;
    if end > start && content[end - 1] == b'\n' {
        end -= 1;
        if end > start && content[end - 1] == b'\r' {
            end -= 1;
        }
    }
    (start, end)
}

/// Extend a line-oriented span over the newline that ends it, so a deletion
/// removes the line rather than leaving an empty one behind.
pub(crate) fn edit_extend_over_newline(content: &[u8], range: (usize, usize)) -> (usize, usize) {
    let (start, mut end) = range;
    if end < content.len() && content[end] == b'\r' {
        end += 1;
    }
    if end < content.len() && content[end] == b'\n' {
        end += 1;
    }
    (start, end)
}

pub(crate) fn edit_find_all(haystack: &[u8], needle: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if needle.is_empty() || needle.len() > haystack.len() {
        return out;
    }
    let mut index = 0usize;
    while index + needle.len() <= haystack.len() {
        if &haystack[index..index + needle.len()] == needle {
            out.push((index, index + needle.len()));
            index += needle.len();
        } else {
            index += 1;
        }
    }
    out
}

/// Splice every edit in one pass and report, for each of them, the byte range
/// it occupies in the RESULT. `--expect N` writes N spans and the report has to
/// name all of them, so the offsets are collected while they are still exact
/// rather than recomputed from the old content afterwards.
pub(crate) fn edit_splice(
    content: &[u8],
    edits: &mut [(usize, usize, Vec<u8>)],
) -> (Vec<u8>, Vec<(usize, usize)>) {
    edits.sort_by_key(|(start, _, _)| *start);
    let mut out = Vec::with_capacity(content.len());
    let mut written = Vec::with_capacity(edits.len());
    let mut cursor = 0usize;
    for (start, end, replacement) in edits.iter() {
        out.extend_from_slice(&content[cursor..*start]);
        let at = out.len();
        out.extend_from_slice(replacement);
        written.push((at, out.len()));
        cursor = *end;
    }
    out.extend_from_slice(&content[cursor..]);
    (out, written)
}

/// Turn result byte ranges into the exact line runs a compact receipt names.
/// Two writes on adjacent lines are one contiguous run; disjoint writes retain
/// the comma that proves the edit did not touch the lines between them.
pub(crate) fn edit_line_span_runs(
    content: &[u8],
    ranges: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let spans: Vec<(usize, usize)> = ranges
        .iter()
        .map(|(start, end)| {
            let start = (*start).min(content.len());
            let end = (*end).min(content.len()).max(start);
            edit_span_lines(content, start, end - start)
        })
        .collect();
    edit_merge_line_spans(spans)
}

pub(crate) fn edit_merge_line_spans(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    spans.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (first, last) in spans {
        if let Some(run) = runs.last_mut() {
            if first <= run.1.saturating_add(1) {
                run.1 = run.1.max(last);
                continue;
            }
        }
        runs.push((first, last));
    }
    runs
}

pub(crate) fn edit_format_line_address(file: &str, spans: &[(usize, usize)]) -> String {
    let suffix = spans
        .iter()
        .map(|(first, last)| {
            if first == last {
                first.to_string()
            } else {
                format!("{first}-{last}")
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{file}:{suffix}")
}

pub(crate) fn edit_exact_address(file: &str, content: &[u8], ranges: &[(usize, usize)]) -> String {
    edit_format_line_address(file, &edit_line_span_runs(content, ranges))
}

/// The shared emitter has a single contiguous `span` slot. When an operation
/// has several sites (or several files), supply its exact compact lines as the
/// headline instead, preserving the same status words and transaction suffix.
pub(crate) fn edit_set_exact_receipt(
    record: &mut EditRecord,
    addresses: Vec<String>,
    exact_required: bool,
) {
    if !exact_required || addresses.is_empty() {
        return;
    }
    let short_id = record
        .transaction_id
        .as_deref()
        .map(|id| &id[..id.len().min(6)]);
    let lines = addresses
        .into_iter()
        .map(|address| {
            if record.already_as_sent {
                format!("applied, already as sent  {address}")
            } else {
                let word = if record.published {
                    "applied"
                } else {
                    "would apply"
                };
                if let Some(id) = short_id.filter(|_| record.published) {
                    format!("{word} {address}  {id}")
                } else {
                    format!("{word} {address}")
                }
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    record.headline = Some(lines);
}

/// Refuse a path that leaves the repository, whether by climbing out of it or
/// through a symlink that points out.
pub(crate) fn edit_guard_path(
    root_path: &std::path::Path,
    abs: &std::path::Path,
) -> EditResult<()> {
    match greppy_edit::publish::require_inside_workspace(root_path, abs) {
        Ok(_) => Ok(()),
        Err(Error::Io { .. }) => Err(EditRefusal::new(
            "file_not_found",
            format!("no file {}", abs.display()),
            10,
        )),
        Err(error) => Err(EditRefusal::new("path_outside_repo", error.to_string(), 17)),
    }
}

pub(crate) fn edit_read_file(
    root_path: &std::path::Path,
    file_base: &std::path::Path,
    file: &str,
) -> EditResult<(String, std::path::PathBuf, Vec<u8>)> {
    let abs = file_operand_path(file_base, file);
    if std::fs::symlink_metadata(&abs).is_err() {
        return Err(EditRefusal::new(
            "file_not_found",
            format!("no file `{file}`"),
            10,
        ));
    }
    edit_guard_path(root_path, &abs)?;
    let content = std::fs::read(&abs).map_err(|error| {
        EditRefusal::new("file_unreadable", format!("read {file}: {error}"), 10)
    })?;
    let workspace = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.to_path_buf());
    let canonical_abs = abs.canonicalize().unwrap_or_else(|_| abs.clone());
    let rel = canonical_abs
        .strip_prefix(&workspace)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| file.to_string());
    Ok((rel, abs, content))
}

/// Resolve `--symbol S` against the graph for the file, then against the bytes
/// on disk for the span: the graph is a cache and the file is the truth, so a
/// definition that moved since indexing is still addressed correctly.
pub(crate) fn edit_resolve_symbol(
    root_path: &std::path::Path,
    root: Option<&str>,
    name: &str,
    path_filter: Option<&str>,
    want_body: bool,
) -> EditResult<ResolvedSpan> {
    let store = open_default_store_query_writer(root).map_err(|error| {
        EditRefusal::new(
            "symbol_not_found",
            format!("no symbol `{name}`: {error}"),
            10,
        )
    })?;
    let ids = resolve_symbol_nodes(&store, Some(name)).map_err(|error| {
        EditRefusal::new(
            "symbol_not_found",
            format!("no symbol `{name}`: {error}"),
            10,
        )
    })?;
    let mut nodes = Vec::new();
    for id in &ids {
        if let Ok(Some(node)) = store.get_node(*id) {
            if node.file_path.is_empty() || node.start_line < 1 {
                continue;
            }
            if let Some(filter) = path_filter {
                let filter = filter.trim_start_matches("./");
                if !node.file_path.starts_with(filter) {
                    continue;
                }
            }
            nodes.push(node);
        }
    }
    // `--symbol S` names a definition. The graph also carries one synthetic
    // anchor per file, and its name is the file stem — so `pkg/greet.go`
    // answers to `greet` and turns an unambiguous edit into "2 definitions".
    // A file is addressed by `--file`, never by `--symbol`, so the anchor is
    // dropped whenever a real definition answered as well (rule 1: an argument
    // is never reinterpreted into a different question).
    if nodes
        .iter()
        .any(|node| !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name))
    {
        nodes.retain(|node| {
            !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name)
        });
    }
    if nodes.is_empty() {
        return Err(EditRefusal::new(
            "symbol_not_found",
            format!("no symbol `{name}`"),
            10,
        ));
    }
    let mut sites: Vec<(String, i64)> = nodes
        .iter()
        .map(|node| (node.file_path.clone(), node.start_line))
        .collect();
    sites.sort();
    sites.dedup();
    if sites.len() > 1 {
        let candidates: Vec<serde_json::Value> = nodes
            .iter()
            .map(|node| {
                serde_json::json!({
                    "qualified_name": node.qualified_name,
                    "path": node.file_path,
                    "line": node.start_line,
                })
            })
            .collect();
        let listed = nodes
            .iter()
            .map(|node| {
                format!(
                    "  {} {}:{}",
                    node.qualified_name, node.file_path, node.start_line
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Err(EditRefusal::new(
            "ambiguous_symbol",
            format!("`{name}` resolves to {} definitions\n{listed}", sites.len()),
            11,
        )
        .with("candidates", serde_json::json!(candidates)));
    }
    let node = &nodes[0];
    let abs = root_path.join(&node.file_path);
    edit_guard_path(root_path, &abs)?;
    let content = std::fs::read(&abs).map_err(|error| {
        EditRefusal::new(
            "file_unreadable",
            format!("read {}: {error}", node.file_path),
            10,
        )
    })?;
    let language = greppy_edit::language_for_path(std::path::Path::new(&node.file_path));
    let (start_line, end_line) = edit_live_definition_lines(language, &content, node)
        .unwrap_or((node.start_line as usize, node.end_line as usize));
    let range = line_range_to_bytes(&content, start_line, end_line);
    let mut range = edit_strip_trailing_newline(&content, range);
    if want_body {
        let Some(body) = greppy_edit::verbs::body_range_within(language, &content, range) else {
            return Err(
                EditRefusal::new("no_body", format!("`{name}` has no body"), 13)
                    .with("symbol", serde_json::json!(name)),
            );
        };
        range = edit_strip_trailing_newline(&content, body);
    }
    Ok((node.file_path.clone(), abs, content, range))
}

/// Re-extract the definitions of one file and return the live line span of the
/// node the graph named. Falls back to the cached span when the language has no
/// extraction pass.
pub(crate) fn edit_live_definition_lines(
    language: greppy_edit::Language,
    content: &[u8],
    node: &greppy_store::Node,
) -> Option<(usize, usize)> {
    let extracted = greppy_parser::extract::extract(language, content, &node.file_path).ok()?;
    let exact = extracted
        .nodes
        .iter()
        .find(|candidate| candidate.qualified_name == node.qualified_name);
    let chosen = exact.or_else(|| {
        let mut by_name = extracted
            .nodes
            .iter()
            .filter(|candidate| candidate.name == node.name);
        let first = by_name.next()?;
        if by_name.next().is_some() {
            None
        } else {
            Some(first)
        }
    })?;
    Some((chosen.start_line as usize, chosen.end_line as usize))
}

pub(crate) fn edit_parse_line_range(spec: &str) -> EditResult<(usize, usize)> {
    let bad = || {
        EditRefusal::new(
            "invalid_selector",
            format!("--lines takes A:B, 1-based and both ends included; got `{spec}`"),
            20,
        )
    };
    let (first, last) = spec.split_once(':').unwrap_or((spec, spec));
    let first: usize = first.trim().parse().map_err(|_| bad())?;
    let last: usize = last.trim().parse().map_err(|_| bad())?;
    if first == 0 || last < first {
        return Err(bad());
    }
    Ok((first, last))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn edit_locate(
    spec: &WhereSpec,
    kind: SelectorKind,
    root: Option<&str>,
    root_path: &std::path::Path,
    file_base: &std::path::Path,
) -> EditResult<Located> {
    match kind {
        SelectorKind::Symbol => {
            let name = spec.symbol.as_deref().unwrap_or_default();
            let (rel, abs, content, range) =
                edit_resolve_symbol(root_path, root, name, spec.path.as_deref(), spec.body)?;
            Ok(Located {
                rel,
                abs,
                content,
                ranges: vec![range],
                kind,
                regex: None,
                needle: None,
            })
        }
        SelectorKind::Target => {
            let token = spec.target.as_deref().unwrap_or_default();
            let handle = greppy_edit::EditHandle::decode(token).map_err(|error| {
                EditRefusal::new(
                    "invalid_handle",
                    format!("not a usable handle: {error}"),
                    20,
                )
            })?;
            let handle_root = std::path::Path::new(&handle.workspace_root)
                .canonicalize()
                .unwrap_or_else(|_| std::path::PathBuf::from(&handle.workspace_root));
            let here = root_path
                .canonicalize()
                .unwrap_or_else(|_| root_path.to_path_buf());
            if handle_root != here {
                return Err(EditRefusal::new(
                    "foreign_handle",
                    format!(
                        "that handle was taken in {}, not in {}",
                        handle.workspace_root,
                        here.display()
                    ),
                    20,
                ));
            }
            let abs = if std::path::Path::new(&handle.path).is_absolute() {
                std::path::PathBuf::from(&handle.path)
            } else {
                root_path.join(&handle.path)
            };
            edit_guard_path(root_path, &abs)?;
            let content = std::fs::read(&abs).map_err(|error| {
                EditRefusal::new(
                    "file_unreadable",
                    format!("read {}: {error}", handle.path),
                    10,
                )
            })?;
            let range = handle.verify(&content).map_err(|_| {
                EditRefusal::new(
                    "stale_handle",
                    format!("{} changed since that handle was taken", handle.path),
                    12,
                )
            })?;
            let range = edit_strip_trailing_newline(&content, range);
            Ok(Located {
                rel: handle.path.clone(),
                abs,
                content,
                ranges: vec![range],
                kind,
                regex: None,
                needle: None,
            })
        }
        _ => {
            let file = spec.file.as_deref().unwrap_or_default();
            let (rel, abs, content, ranges, regex, needle) = match kind {
                SelectorKind::Lines => {
                    let (first, last) =
                        edit_parse_line_range(spec.lines.as_deref().unwrap_or_default())?;
                    let (rel, abs, content) = edit_read_file(root_path, file_base, file)?;
                    let total = edit_line_count(&content);
                    if last > total || first > total {
                        return Err(EditRefusal::new(
                            "range_out_of_bounds",
                            format!("{rel} has {total} line(s); {first}:{last} runs past its end"),
                            13,
                        ));
                    }
                    let range = line_range_to_bytes(&content, first, last);
                    let range = edit_strip_trailing_newline(&content, range);
                    (rel, abs, content, vec![range], None, None)
                }
                SelectorKind::Text => {
                    let needle = match (&spec.old, &spec.old_file) {
                        (Some(text), None) => text.as_bytes().to_vec(),
                        (None, Some(path)) => read_source_arg(path).map_err(|error| {
                            EditRefusal::new(
                                "invalid_selector",
                                format!("--old-file {path}: {error}"),
                                20,
                            )
                        })?,
                        _ => Vec::new(),
                    };
                    if needle.is_empty() {
                        return Err(EditRefusal::new(
                            "invalid_selector",
                            "--old is empty; empty text matches between every pair of characters",
                            20,
                        ));
                    }
                    let (rel, abs, content) = edit_read_file(root_path, file_base, file)?;
                    let ranges = edit_find_all(&content, &needle);
                    let shown = String::from_utf8_lossy(&needle).into_owned();
                    (rel, abs, content, ranges, None, Some(shown))
                }
                SelectorKind::Pattern => {
                    let pattern = spec.pattern.as_deref().unwrap_or_default();
                    let regex = regex::bytes::Regex::new(pattern).map_err(|error| {
                        EditRefusal::new(
                            "invalid_pattern",
                            format!("--pattern is not a regular expression: {error}"),
                            20,
                        )
                    })?;
                    let (rel, abs, content) = edit_read_file(root_path, file_base, file)?;
                    let ranges = regex
                        .find_iter(&content)
                        .map(|found| (found.start(), found.end()))
                        .collect();
                    (
                        rel,
                        abs,
                        content,
                        ranges,
                        Some(regex),
                        Some(pattern.to_string()),
                    )
                }
                SelectorKind::Symbol | SelectorKind::Target => unreachable!(),
            };
            Ok(Located {
                rel,
                abs,
                content,
                ranges,
                kind,
                regex,
                needle,
            })
        }
    }
}

/// The number of matches a selector is allowed to have. `--old` and
/// `--pattern` search, so they can find none or many; every other selector
/// addresses exactly one span by construction.
pub(crate) fn edit_check_cardinality(located: &Located, expect: Option<usize>) -> EditResult<()> {
    if !matches!(located.kind, SelectorKind::Text | SelectorKind::Pattern) {
        return Ok(());
    }
    let expect = expect.unwrap_or(1);
    if located.ranges.len() != expect {
        // The count alone does not let a caller decide between "pass --expect N"
        // and "I anchored on the wrong text", so the refusal names what was
        // searched for and where every match sits.
        let _subject = located.needle.as_deref().map_or_else(
            || located.kind.name().to_string(),
            |text| format!("`{text}`"),
        );
        let sites: Vec<String> = located
            .ranges
            .iter()
            .take(20)
            .map(|(start, _)| {
                format!(
                    "{}:{}:{}: {}",
                    located.rel,
                    edit_line_of_offset(&located.content, *start),
                    {
                        let ls = located.content[..*start]
                            .iter()
                            .rposition(|&b| b == b'\n')
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        *start - ls + 1
                    },
                    {
                        let ls = located.content[..*start]
                            .iter()
                            .rposition(|&b| b == b'\n')
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        let le = located.content[*start..]
                            .iter()
                            .position(|&b| b == b'\n')
                            .map(|i| *start + i)
                            .unwrap_or(located.content.len());
                        one_line_truncated(&String::from_utf8_lossy(&located.content[ls..le]), 200)
                    }
                )
            })
            .collect();
        // The needle is not echoed — the caller has it in context (law 5).
        let mut message = match located.kind {
            SelectorKind::Text => format!(
                "OLD occurs {} times — nothing written",
                located.ranges.len()
            ),
            SelectorKind::Pattern => format!(
                "the pattern occurs {} times, expected {expect} — nothing written",
                located.ranges.len()
            ),
            _ => unreachable!(),
        };
        for site in &sites {
            message.push_str("\n  ");
            message.push_str(site);
        }
        return Err(EditRefusal::new("match_count", message, 13)
            .with("expected", serde_json::json!(expect))
            .with("found", serde_json::json!(located.ranges.len()))
            .with("matches", serde_json::json!(sites)));
    }
    Ok(())
}

pub(crate) fn edit_expect_positive(expect: Option<usize>) -> EditResult<()> {
    if expect == Some(0) {
        return Err(EditRefusal::new(
            "invalid_expect",
            "--expect 0 asks for an edit that writes nothing",
            20,
        ));
    }
    Ok(())
}

/// There is exactly one stdin, so two arguments asking for it leave the
/// caller's intent unrecoverable — one of them would get nothing, or both would
/// get half. The collision is refused before any of them reads a byte.
pub(crate) fn edit_positional_payload(
    payload: Option<String>,
    name: &'static str,
) -> EditResult<Vec<u8>> {
    if let Some(payload) = payload {
        return Ok(payload.into_bytes());
    }
    use std::io::{IsTerminal, Read};
    if std::io::stdin().is_terminal() {
        return Err(EditRefusal::new(
            "content_missing",
            format!("no {name}: pass it as the final positional or pipe it on stdin"),
            20,
        ));
    }
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).map_err(|error| {
        EditRefusal::new(
            "content_unreadable",
            format!("read {name} from stdin: {error}"),
            20,
        )
    })?;
    if bytes.is_empty() {
        return Err(EditRefusal::new(
            "content_missing",
            format!("no {name}: stdin was empty"),
            20,
        ));
    }
    Ok(bytes)
}

/// Validate a candidate without writing it. Parser locations refer to the
/// proposed content, which may have different line numbers from the live file.
fn edit_validate_syntax(path: &str, before: &[u8], after: &[u8]) -> EditResult<()> {
    let language = greppy_edit::txn::syntax_language_for_path(std::path::Path::new(path), before);
    if !language.is_supported() {
        return Ok(());
    }
    if let (Some(before), Some(counts)) = (
        greppy_edit::txn::syntax_counts(language, before),
        greppy_edit::txn::syntax_counts(language, after),
    ) {
        if counts.errors > before.errors || counts.missing > before.missing {
            let location = greppy_edit::txn::first_syntax_diagnostic(language, after)
                .map(|diagnostic| format!("{path}:{diagnostic}"))
                .unwrap_or_else(|| path.to_string());
            return Err(EditRefusal::new(
                "invalid_result",
                format!(
                    "refused: syntax validation failed in proposed {location}; \
                     errors {} -> {}, missing nodes {} -> {} — nothing written. \
                     Location refers to the proposed result, not the unchanged file",
                    before.errors, counts.errors, before.missing, counts.missing
                ),
                13,
            ));
        }
    }
    Ok(())
}

/// Publish one file and answer with the record the contract promises: the
/// file, every span it wrote, the resulting text, and a handle for the new
/// span so the next edit needs no `read` in between.
pub(crate) fn edit_publish(
    root_path: &std::path::Path,
    located: &Located,
    new_content: Vec<u8>,
    changed: Vec<(usize, usize)>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let (first_start, first_end) = changed.first().copied().unwrap_or((0, 0));
    let first_end = first_end.min(new_content.len());
    let text = String::from_utf8_lossy(&new_content[first_start..first_end]).into_owned();
    let span = edit_span_lines(&new_content, first_start, first_end - first_start);
    let exact_required = changed.len() > 1;
    let exact_address = edit_exact_address(&located.rel, &new_content, &changed);
    let mut operation = EditOperation {
        file: located.rel.clone(),
        ranges: changed,
        result_span: Some(text.clone()),
        sha_before: Some(edit_sha256_hex(&located.content)),
        sha_after: Some(edit_sha256_hex(&new_content)),
        diff: Some(edit_unified_diff(
            &located.rel,
            &located.content,
            &new_content,
        )),
        ..EditOperation::default()
    };
    let mut record = EditRecord {
        files: vec![located.rel.clone()],
        span: Some(span),
        text: Some(text),
        published: !dry_run,
        ..EditRecord::default()
    };
    if new_content == located.content {
        // A dry run never claims an application, even when the requested bytes
        // are already present. Its receipt remains `would apply`; the stronger
        // `applied, already as sent` wording is reserved for a non-dry call.
        record.already_as_sent = !dry_run;
        record.operations = vec![operation];
        edit_set_exact_receipt(&mut record, vec![exact_address], exact_required);
        return Ok(record);
    }
    edit_validate_syntax(&located.rel, &located.content, &new_content)?;
    if dry_run {
        // A handle addresses bytes on disk. A dry run wrote none, so handing
        // one back would hand back an address that is already stale.
        record.operations = vec![operation];
        edit_set_exact_receipt(&mut record, vec![exact_address], exact_required);
        return Ok(record);
    }
    let before_sha = edit_sha256_hex(&located.content);
    // The journal goes down before the write, so an edit that dies in between
    // leaves the evidence `recover` needs instead of a half-written file.
    let transaction = edit_journal_open(
        root_path,
        &[UndoBefore {
            rel: located.rel.clone(),
            content: Some(located.content.clone()),
        }],
    );
    edit_journal_crash_hook()?;
    if let Err(error) =
        greppy_edit::publish::publish_atomic(root_path, &located.abs, &new_content, &before_sha)
    {
        edit_journal_abort(root_path);
        return Err(EditRefusal::new("publish_failed", error.to_string(), 16));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
        record.transaction_id = Some(id);
    }
    let handle = greppy_edit::EditHandle::for_range(
        root_path,
        std::path::Path::new(&located.rel),
        &new_content,
        first_start,
        first_end,
    )
    .ok()
    .map(|handle| handle.encode());
    operation.handle = handle.clone();
    record.handle = handle;
    record.operations = vec![operation];
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    edit_set_exact_receipt(&mut record, vec![exact_address], exact_required);
    Ok(record)
}

fn edit_check_regex_replacement(regex: &regex::bytes::Regex, replacement: &[u8]) -> EditResult<()> {
    let mut at = 0;
    while at < replacement.len() {
        if replacement[at] != b'$' {
            at += 1;
            continue;
        }
        at += 1;
        if replacement.get(at) == Some(&b'$') {
            at += 1;
            continue;
        }
        let start;
        let end;
        if replacement.get(at) == Some(&b'{') {
            start = at + 1;
            let Some(close) = replacement[start..].iter().position(|b| *b == b'}') else {
                continue;
            };
            end = start + close;
            at = end + 1;
        } else {
            start = at;
            while replacement
                .get(at)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
            {
                at += 1;
            }
            end = at;
            if start == end {
                continue;
            }
        }
        let Ok(name) = std::str::from_utf8(&replacement[start..end]) else {
            // The byte-regex engine treats invalid UTF-8 in ${...} literally.
            continue;
        };
        let known = if let Ok(index) = name.parse::<usize>() {
            index < regex.captures_len()
        } else {
            regex
                .capture_names()
                .flatten()
                .any(|capture| capture == name)
        };
        if !known {
            return Err(EditRefusal::new(
                "unknown_replacement_capture",
                format!("--regex expands captures in NEW, but capture '{name}' does not exist in OLD; nothing written. Use $$ for a literal dollar sign, or omit --regex for a literal OLD pattern."),
                17,
            ));
        }
    }
    Ok(())
}

pub(crate) fn edit_op_replace(located: &Located, new_bytes: &[u8]) -> EditResult<EditedContent> {
    if let Some(regex) = &located.regex {
        edit_check_regex_replacement(regex, new_bytes)?;
    }
    // A line-oriented span stops before the newline that ends its last line,
    // because that newline belongs to the file (see `SelectorKind::line_oriented`).
    // New text that carries one of its own would therefore add a blank line the
    // caller never wrote, so the span's own ending is the one that survives.
    let new_bytes = if located.kind.line_oriented() {
        let trimmed = new_bytes
            .strip_suffix(b"\n")
            .map(|text| text.strip_suffix(b"\r").unwrap_or(text));
        trimmed.unwrap_or(new_bytes)
    } else {
        new_bytes
    };
    let mut edits: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    for (start, end) in &located.ranges {
        let replacement = match &located.regex {
            Some(regex) => {
                let mut expanded = Vec::new();
                if let Some(captures) = regex.captures_at(&located.content, *start) {
                    captures.expand(new_bytes, &mut expanded);
                } else {
                    expanded.extend_from_slice(new_bytes);
                }
                expanded
            }
            None => new_bytes.to_vec(),
        };
        edits.push((*start, *end, replacement));
    }
    Ok(edit_splice(&located.content, &mut edits))
}

pub(crate) fn edit_op_delete(located: &Located) -> EditedContent {
    let mut edits: Vec<(usize, usize, Vec<u8>)> = located
        .ranges
        .iter()
        .map(|range| {
            let range = if located.kind.line_oriented() {
                edit_extend_over_newline(&located.content, *range)
            } else {
                *range
            };
            (range.0, range.1, Vec::new())
        })
        .collect();
    edit_splice(&located.content, &mut edits)
}

#[derive(Debug)]
struct EditVerifier {
    label: String,
    program: std::path::PathBuf,
    args: Vec<std::ffi::OsString>,
    cwd: std::path::PathBuf,
}

fn edit_nearest_package_root(
    root_path: &std::path::Path,
    file: &str,
) -> Option<std::path::PathBuf> {
    let root = root_path.canonicalize().ok()?;
    let mut directory = root_path.join(file).parent()?.to_path_buf();
    loop {
        if directory.join("package.json").is_file() {
            return Some(directory);
        }
        if directory == root || !directory.pop() || !directory.starts_with(&root) {
            break;
        }
    }
    root_path
        .join("package.json")
        .is_file()
        .then(|| root_path.to_path_buf())
}

fn edit_local_typescript_compiler(
    root_path: &std::path::Path,
    package_root: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let workspace_root = root_path.canonicalize().ok()?;
    let mut directory = package_root.canonicalize().ok()?;
    if !directory.starts_with(&workspace_root) {
        return None;
    }
    loop {
        let bin = directory.join("node_modules").join(".bin");
        if let Some(compiler) = ["tsc", "tsc.cmd", "tsgo", "tsgo.cmd"]
            .into_iter()
            .map(|name| bin.join(name))
            .find(|path| path.is_file())
        {
            return Some(compiler);
        }
        if directory == workspace_root {
            return None;
        }
        directory = directory.parent()?.to_path_buf();
    }
}

fn edit_verifiers(
    root_path: &std::path::Path,
    files: &[String],
) -> (Vec<EditVerifier>, Option<String>) {
    let extensions = files
        .iter()
        .filter_map(|file| std::path::Path::new(file).extension()?.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let has_typescript = extensions
        .iter()
        .any(|extension| matches!(extension.as_str(), "ts" | "tsx" | "mts" | "cts"));
    if has_typescript {
        let Some(package_root) = files
            .iter()
            .find_map(|file| edit_nearest_package_root(root_path, file))
        else {
            return (
                Vec::new(),
                Some("verify: skipped — no package.json owns the touched TypeScript file".into()),
            );
        };
        let Some(tsc) = edit_local_typescript_compiler(root_path, &package_root) else {
            return (
                Vec::new(),
                Some(format!(
                    "verify: skipped — no local TypeScript compiler from {} through workspace root {}; expected node_modules/.bin/tsc or tsgo (network downloads are never started by --verify)",
                    package_root.display(),
                    root_path.display()
                )),
            );
        };
        return (
            vec![EditVerifier {
                label: "local TypeScript check".into(),
                program: tsc,
                args: ["--noEmit", "--pretty", "false", "--incremental", "false"]
                    .into_iter()
                    .map(std::ffi::OsString::from)
                    .collect(),
                cwd: package_root,
            }],
            None,
        );
    }

    let javascript = files
        .iter()
        .filter(|file| {
            std::path::Path::new(file)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| matches!(extension, "js" | "mjs" | "cjs"))
        })
        .map(|file| EditVerifier {
            label: format!("JavaScript syntax check for {file}"),
            program: std::path::PathBuf::from("node"),
            args: vec!["--check".into(), file.into()],
            cwd: root_path.to_path_buf(),
        })
        .collect::<Vec<_>>();
    if !javascript.is_empty() {
        return (javascript, None);
    }

    if (extensions.iter().any(|extension| extension == "rs")
        || files
            .iter()
            .any(|file| file == "Cargo.toml" || file == "Cargo.lock"))
        && root_path.join("Cargo.toml").is_file()
    {
        return (
            vec![EditVerifier {
                label: "Rust workspace check".into(),
                program: "cargo".into(),
                args: ["check", "--message-format", "short", "--quiet"]
                    .into_iter()
                    .map(std::ffi::OsString::from)
                    .collect(),
                cwd: root_path.to_path_buf(),
            }],
            None,
        );
    }
    if extensions.iter().any(|extension| extension == "go") && root_path.join("go.mod").is_file() {
        return (
            vec![EditVerifier {
                label: "Go workspace build".into(),
                program: "go".into(),
                args: vec!["build".into(), "./...".into()],
                cwd: root_path.to_path_buf(),
            }],
            None,
        );
    }
    let python_files = files
        .iter()
        .filter(|file| file.ends_with(".py"))
        .map(std::ffi::OsString::from)
        .collect::<Vec<_>>();
    if !python_files.is_empty() {
        let mut args = vec!["-m".into(), "py_compile".into()];
        args.extend(python_files);
        return (
            vec![EditVerifier {
                label: "Python syntax check".into(),
                program: "python3".into(),
                args,
                cwd: root_path.to_path_buf(),
            }],
            None,
        );
    }
    (
        Vec::new(),
        Some("verify: skipped — no bounded verifier is declared for the touched file type".into()),
    )
}

fn edit_verify_timeout() -> std::time::Duration {
    let seconds = std::env::var("GREPPY_EDIT_VERIFY_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(60);
    std::time::Duration::from_secs(seconds)
}

fn edit_kill_verifier_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // The verifier owns a process group (configured below). Killing only
        // its shell leaves `sleep`, compilers, or package-manager children
        // alive with inherited descriptors, so callers using captured output
        // still hang until those descendants exit.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    let _ = child.kill();
}

fn edit_run_verifier(verifier: &EditVerifier, timeout: std::time::Duration) -> Vec<String> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    let prefix = std::env::temp_dir().join(format!("greppy-verify-{}-{nonce}", std::process::id()));
    let stdout_path = prefix.with_extension("stdout");
    let stderr_path = prefix.with_extension("stderr");
    let stdout = match std::fs::File::create(&stdout_path) {
        Ok(file) => file,
        Err(error) => {
            return vec![format!(
                "verify: unavailable — cannot capture stdout: {error}"
            )]
        }
    };
    let stderr = match std::fs::File::create(&stderr_path) {
        Ok(file) => file,
        Err(error) => {
            let _ = std::fs::remove_file(&stdout_path);
            return vec![format!(
                "verify: unavailable — cannot capture stderr: {error}"
            )];
        }
    };
    let command = std::iter::once(verifier.program.as_os_str())
        .chain(verifier.args.iter().map(std::ffi::OsString::as_os_str))
        .map(|argument| argument.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "verify: running {} — {} (timeout {}s)",
        verifier.label,
        command,
        timeout.as_secs()
    );
    let mut process = std::process::Command::new(&verifier.program);
    process
        .args(&verifier.args)
        .current_dir(&verifier.cwd)
        .stdout(std::process::Stdio::from(stdout))
        .stderr(std::process::Stdio::from(stderr));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        process.process_group(0);
    }
    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => {
            let _ = std::fs::remove_file(&stdout_path);
            let _ = std::fs::remove_file(&stderr_path);
            let message = format!("verify: unavailable — cannot start {command}: {error}");
            eprintln!("{message}");
            return vec![message];
        }
    };
    let started = std::time::Instant::now();
    let mut next_progress = std::time::Duration::from_secs(10);
    let (status, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (Some(status), false),
            Ok(None) if started.elapsed() >= timeout => {
                edit_kill_verifier_tree(&mut child);
                break (child.wait().ok(), true);
            }
            Ok(None) => {
                if started.elapsed() >= next_progress {
                    eprintln!(
                        "verify: still running {} ({}s elapsed)",
                        verifier.label,
                        started.elapsed().as_secs()
                    );
                    next_progress += std::time::Duration::from_secs(10);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(error) => {
                edit_kill_verifier_tree(&mut child);
                let _ = child.wait();
                let message = format!("verify: failed to observe {command}: {error}");
                eprintln!("{message}");
                break (None, false);
            }
        }
    };
    let stdout = std::fs::read_to_string(&stdout_path).unwrap_or_default();
    let stderr = std::fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = std::fs::remove_file(&stdout_path);
    let _ = std::fs::remove_file(&stderr_path);
    if timed_out {
        let message = format!(
            "verify: timed out after {}s — edit remains applied; run `{command}` directly to continue",
            timeout.as_secs()
        );
        eprintln!("{message}");
        return vec![message];
    }
    let Some(status) = status else {
        return vec![format!("verify: failed — no exit status from {command}")];
    };
    if status.success() {
        let message = format!("verify: passed — {}", verifier.label);
        eprintln!("{message}");
        return vec![message];
    }
    let mut diagnostics = vec![format!(
        "verify: failed (exit {}) — {}",
        status
            .code()
            .map_or_else(|| "signal".into(), |code| code.to_string()),
        verifier.label
    )];
    diagnostics.extend(
        stderr
            .lines()
            .chain(stdout.lines())
            .filter(|line| {
                let lowered = line.to_ascii_lowercase();
                lowered.contains("error") || lowered.contains("warning")
            })
            .take(50)
            .map(str::to_string),
    );
    eprintln!("{}", diagnostics[0]);
    diagnostics
}

/// The compiler or linter for the touched file type, when the workspace has a
/// local one. Verification is observable and bounded; it never downloads a
/// tool and never silently switches to an unrelated language's workspace.
pub(crate) fn edit_verify_diagnostics(
    root_path: &std::path::Path,
    files: &[String],
) -> Vec<String> {
    let (verifiers, skipped) = edit_verifiers(root_path, files);
    if let Some(message) = skipped {
        eprintln!("{message}");
        return vec![message];
    }
    let timeout = edit_verify_timeout();
    verifiers
        .iter()
        .flat_map(|verifier| edit_run_verifier(verifier, timeout))
        .collect()
}

pub(crate) fn edit_journal_dir(root_path: &std::path::Path) -> std::path::PathBuf {
    greppy_core::cache::workspace_store_dir(root_path).join(EDIT_JOURNAL_DIR)
}

pub(crate) fn edit_journal_read(path: &std::path::Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

pub(crate) fn edit_journal_write(path: &std::path::Path, value: &serde_json::Value) {
    if let Ok(text) = serde_json::to_string_pretty(value) {
        let _ = std::fs::write(path, text);
    }
}

/// Record the pre-images and open a transaction. Anything that dies between
/// here and [`edit_journal_close`] leaves `pending.json` behind — which is
/// exactly what `recover` looks for.
pub(crate) fn edit_journal_open(
    root_path: &std::path::Path,
    before: &[UndoBefore],
) -> Option<String> {
    if before.is_empty() {
        return None;
    }
    let dir = ensured_workspace_store_path(root_path)
        .ok()?
        .with_file_name(EDIT_JOURNAL_DIR);
    std::fs::create_dir_all(dir.join(EDIT_JOURNAL_BLOBS)).ok()?;
    let seed = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default()
    );
    let id = edit_sha256_hex(seed.as_bytes());
    let mut entries = Vec::new();
    for (index, item) in before.iter().enumerate() {
        let blob = match &item.content {
            Some(bytes) => {
                let name = format!("{id}-{index}.bin");
                std::fs::write(dir.join(EDIT_JOURNAL_BLOBS).join(&name), bytes).ok()?;
                Some(name)
            }
            None => None,
        };
        entries.push(serde_json::json!({ "path": item.rel, "blob": blob }));
    }
    edit_journal_write(
        &dir.join(EDIT_JOURNAL_PENDING),
        &serde_json::json!({ "id": id, "entries": entries }),
    );
    Some(id)
}

/// Die after the journal is on disk and before anything is published, so the
/// interrupted-edit path can be exercised without killing the process from the
/// outside. Only ever reached when the environment variable is set.
pub(crate) fn edit_journal_crash_hook() -> EditResult<()> {
    if std::env::var_os("GREPPY_TEST_CRASH_AFTER_JOURNAL").is_some() {
        return Err(EditRefusal::new(
            "interrupted",
            "interrupted after the journal was written and before anything was published",
            16,
        ));
    }
    Ok(())
}

/// Close the transaction: record what the files look like now, and push it onto
/// the stack. The after-image is what `undo` checks against, so an edit that
/// somebody else overwrote in the meantime cannot be reversed blindly (D3).
pub(crate) fn edit_journal_close(root_path: &std::path::Path, id: &str) {
    let dir = edit_journal_dir(root_path);
    let Some(mut record) = edit_journal_read(&dir.join(EDIT_JOURNAL_PENDING)) else {
        return;
    };
    if record["id"].as_str() != Some(id) {
        return;
    }
    let closed: Vec<serde_json::Value> = record["entries"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|mut entry| {
            let rel = entry["path"].as_str().unwrap_or_default().to_string();
            match std::fs::read(root_path.join(&rel)) {
                Ok(bytes) => {
                    entry["existed_after"] = serde_json::json!(true);
                    entry["after_sha256"] = serde_json::json!(edit_sha256_hex(&bytes));
                }
                Err(_) => {
                    entry["existed_after"] = serde_json::json!(false);
                    entry["after_sha256"] = serde_json::Value::Null;
                }
            }
            entry
        })
        .collect();
    record["entries"] = serde_json::json!(closed);
    let mut stack = edit_journal_read(&dir.join(EDIT_JOURNAL_STACK))
        .and_then(|value| value["transactions"].as_array().cloned())
        .unwrap_or_default();
    stack.push(record);
    if stack.len() > EDIT_JOURNAL_DEPTH {
        let excess = stack.len() - EDIT_JOURNAL_DEPTH;
        stack.drain(..excess);
    }
    edit_journal_write(
        &dir.join(EDIT_JOURNAL_STACK),
        &serde_json::json!({ "transactions": stack }),
    );
    let _ = std::fs::remove_file(dir.join(EDIT_JOURNAL_PENDING));
}

/// Abandon an open transaction without recording it. Used when the work it was
/// opened for turned out to write nothing after all.
pub(crate) fn edit_journal_abort(root_path: &std::path::Path) {
    let _ = std::fs::remove_file(edit_journal_dir(root_path).join(EDIT_JOURNAL_PENDING));
}

/// Put a transaction's files back the way they were. `guarded` is the D3 rule:
/// `undo` refuses if a file no longer looks the way that edit left it, because
/// it would otherwise overwrite bytes the caller has never seen. `recover`
/// restores unguarded — an interrupted edit has no after-image to compare with.
pub(crate) fn edit_journal_restore(
    root_path: &std::path::Path,
    record: &serde_json::Value,
    guarded: bool,
) -> EditResult<Vec<String>> {
    let dir = edit_journal_dir(root_path);
    let entries = record["entries"].as_array().cloned().unwrap_or_default();
    if guarded {
        for entry in &entries {
            let rel = entry["path"].as_str().unwrap_or_default();
            let live = std::fs::read(root_path.join(rel));
            let expected_after = entry["existed_after"].as_bool().unwrap_or(true);
            let unchanged = match (&live, expected_after) {
                (Ok(bytes), true) => {
                    edit_sha256_hex(bytes) == entry["after_sha256"].as_str().unwrap_or_default()
                }
                (Err(_), false) => true,
                _ => false,
            };
            if !unchanged {
                return Err(EditRefusal::new(
                    "changed_since_edit",
                    format!("{rel} no longer looks the way that edit left it"),
                    12,
                ));
            }
        }
    }
    let mut restored = Vec::new();
    for entry in &entries {
        let rel = entry["path"].as_str().unwrap_or_default().to_string();
        let abs = root_path.join(&rel);
        match entry["blob"].as_str() {
            Some(blob) => {
                let bytes =
                    std::fs::read(dir.join(EDIT_JOURNAL_BLOBS).join(blob)).map_err(|error| {
                        EditRefusal::new(
                            "nothing_to_undo",
                            format!("{rel}: the pre-image is gone ({error})"),
                            10,
                        )
                    })?;
                if let Some(parent) = abs.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                std::fs::write(&abs, &bytes).map_err(|error| {
                    EditRefusal::new("publish_failed", format!("{rel}: {error}"), 16)
                })?;
                restored.push(rel);
            }
            // The file was created by that edit, so putting it back means
            // taking it away again.
            None => {
                let _ = std::fs::remove_file(&abs);
            }
        }
    }
    restored.sort();
    Ok(restored)
}

pub(crate) fn run_edit_undo(
    root_path: &std::path::Path,
    requested: Option<&str>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let dir = edit_journal_dir(root_path);
    let mut stack = edit_journal_read(&dir.join(EDIT_JOURNAL_STACK))
        .and_then(|value| value["transactions"].as_array().cloned())
        .unwrap_or_default();
    if stack.is_empty() {
        return Err(EditRefusal::new(
            "nothing_to_undo",
            "nothing to undo in this workspace",
            10,
        ));
    }
    let index = if let Some(requested) = requested {
        let matches: Vec<usize> = stack
            .iter()
            .enumerate()
            .filter(|(_, transaction)| {
                transaction["id"]
                    .as_str()
                    .is_some_and(|id| id == requested || id.starts_with(requested))
            })
            .map(|(index, _)| index)
            .collect();
        match matches.as_slice() {
            [index] => *index,
            [] => {
                return Err(EditRefusal::new(
                    "nothing_to_undo",
                    format!("no edit transaction begins with {requested}"),
                    10,
                ))
            }
            _ => {
                return Err(EditRefusal::new(
                    "ambiguous_transaction",
                    format!("{requested} names more than one edit transaction"),
                    11,
                ))
            }
        }
    } else {
        stack.len() - 1
    };
    let selected = stack[index].clone();
    let id = selected["id"].as_str().unwrap_or_default().to_string();
    let short_id = &id[..id.len().min(6)];
    let files: Vec<String> = selected["entries"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry["path"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let subject = if files.len() == 1 {
        files[0].clone()
    } else {
        format!("{} files", files.len())
    };
    let mut record = EditRecord {
        headline: Some(if dry_run {
            format!("would reverse {short_id} {subject}")
        } else {
            format!("reversed {short_id} {subject}")
        }),
        files,
        published: !dry_run,
        ..EditRecord::default()
    };
    if dry_run {
        return Ok(record);
    }
    let restored = edit_journal_restore(root_path, &selected, true)?;
    stack.remove(index);
    edit_journal_write(
        &dir.join(EDIT_JOURNAL_STACK),
        &serde_json::json!({ "transactions": stack }),
    );
    record.extra.push(("restored", serde_json::json!(restored)));
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

/// Finish or roll back an edit that was interrupted between the journal and the
/// publish. Returns `None` when there is nothing pending, so the caller can fall
/// through to the transaction journal of the certificate verbs.
pub(crate) fn edit_resolve_new_path(
    root_path: &std::path::Path,
    file_base: &std::path::Path,
    file: &str,
) -> EditResult<(String, std::path::PathBuf)> {
    let workspace = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.to_path_buf());
    let base = file_base
        .canonicalize()
        .unwrap_or_else(|_| file_base.to_path_buf());
    let candidate = std::path::Path::new(file);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base.join(candidate)
    };
    let mut normalized = std::path::PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(EditRefusal::new(
                        "path_outside_repo",
                        format!("{file} is outside {}; nothing written. To edit another workspace, pass --root DIR and a path relative to DIR", workspace.display()),
                        17,
                    ));
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    let Ok(relative) = normalized.strip_prefix(&workspace) else {
        return Err(EditRefusal::new(
            "path_outside_repo",
            format!("{file} is outside {}; nothing written. To edit another workspace, pass --root DIR and a path relative to DIR", workspace.display()),
            17,
        ));
    };
    let rel = relative.to_string_lossy().replace('\\', "/");
    if rel.is_empty() {
        return Err(EditRefusal::new(
            "path_outside_repo",
            format!("{file} is the repository root, not a file in it"),
            17,
        ));
    }
    let abs = root_path.join(&rel);
    Ok((rel, abs))
}

/// The record a whole-file verb answers with once the work is done.
pub(crate) fn edit_whole_file_record(
    root_path: &std::path::Path,
    rel: &str,
    bytes: &[u8],
    before: &[u8],
    published: bool,
) -> EditRecord {
    let text = String::from_utf8_lossy(bytes).into_owned();
    let mut operation = EditOperation {
        file: rel.to_string(),
        ranges: vec![(0, bytes.len())],
        result_span: Some(text.clone()),
        sha_before: Some(edit_sha256_hex(before)),
        sha_after: Some(edit_sha256_hex(bytes)),
        diff: Some(edit_unified_diff(rel, before, bytes)),
        ..EditOperation::default()
    };
    let mut record = EditRecord {
        files: vec![rel.to_string()],
        span: Some(edit_span_lines(bytes, 0, bytes.len())),
        text: Some(text),
        published,
        ..EditRecord::default()
    };
    if published {
        let handle = greppy_edit::EditHandle::for_range(
            root_path,
            std::path::Path::new(rel),
            bytes,
            0,
            bytes.len(),
        )
        .ok()
        .map(|handle| handle.encode());
        operation.handle = handle.clone();
        record.handle = handle;
    }
    record.operations = vec![operation];
    record
}

/// The record as data. `full` is the archival form `--report` writes: it adds
/// the diff and the resulting text, which stdout deliberately leaves out
/// because they cost the context window the compact form exists to protect.
pub(crate) fn edit_record_json(
    record: &EditRecord,
    full: bool,
    report_path: Option<&str>,
) -> serde_json::Value {
    let mut value = serde_json::Map::new();
    value.insert(
        "schema_version".into(),
        serde_json::json!(EDIT_RECORD_SCHEMA),
    );
    value.insert(
        "status".into(),
        serde_json::json!(if record.published {
            "applied"
        } else {
            // A dry run that reports "applied" is read as a completed edit by
            // every caller that trusts `status` over `published`.
            "would_apply"
        }),
    );
    value.insert("published".into(), serde_json::json!(record.published));
    value.insert("exit_code".into(), serde_json::json!(0));
    if let Some(first) = record.files.first() {
        value.insert("file".into(), serde_json::json!(first));
    }
    value.insert("files".into(), serde_json::json!(record.files));
    for (key, extra) in &record.extra {
        value.insert((*key).into(), extra.clone());
    }
    if let Some((first, last)) = record.span {
        value.insert("span".into(), serde_json::json!(format!("{first}:{last}")));
    }
    if let Some(text) = &record.text {
        value.insert("text".into(), serde_json::json!(text));
    }
    if let Some(handle) = &record.handle {
        value.insert("handle".into(), serde_json::json!(handle));
    }
    let operations: Vec<serde_json::Value> = record
        .operations
        .iter()
        .map(|operation| {
            let mut entry = serde_json::Map::new();
            entry.insert("file".into(), serde_json::json!(operation.file));
            entry.insert(
                "changed_byte_ranges".into(),
                serde_json::json!(operation.ranges),
            );
            if let Some(text) = &operation.result_span {
                entry.insert("result_span".into(), serde_json::json!(text));
                if full {
                    entry.insert("node_after".into(), serde_json::json!(text));
                }
            }
            if let Some(handle) = &operation.handle {
                entry.insert("handle".into(), serde_json::json!(handle));
            }
            if let Some(sha) = &operation.sha_before {
                entry.insert("file_sha256_before".into(), serde_json::json!(sha));
            }
            if let Some(sha) = &operation.sha_after {
                entry.insert("file_sha256_after".into(), serde_json::json!(sha));
            }
            if full {
                if let Some(diff) = &operation.diff {
                    entry.insert("unified_diff".into(), serde_json::json!(diff));
                }
            }
            serde_json::Value::Object(entry)
        })
        .collect();
    value.insert("operations".into(), serde_json::json!(operations));
    if let Some(diagnostics) = &record.diagnostics {
        value.insert("diagnostics".into(), serde_json::json!(diagnostics));
        value.insert(
            "verify".into(),
            serde_json::json!({ "diagnostics": diagnostics }),
        );
    }
    if !record.notes.is_empty() {
        value.insert("references".into(), serde_json::json!(record.notes));
    }
    if let Some(path) = report_path {
        value.insert("report_path".into(), serde_json::json!(path));
    }
    serde_json::Value::Object(value)
}

/// A refusal is an answer too: the same shape, with the cause named. Without
/// `published` and `exit_code` a caller that asked for `--json` cannot tell a
/// refusal from a success without re-reading the process exit code.
pub(crate) fn edit_refusal_json(
    refusal: &EditRefusal,
    report_path: Option<&str>,
) -> serde_json::Value {
    let mut error = serde_json::Map::new();
    error.insert("code".into(), serde_json::json!(refusal.code));
    error.insert("message".into(), serde_json::json!(refusal.message));
    for (key, value) in &refusal.extra {
        error.insert((*key).into(), value.clone());
    }
    let mut value = serde_json::Map::new();
    value.insert(
        "schema_version".into(),
        serde_json::json!(EDIT_RECORD_SCHEMA),
    );
    value.insert("status".into(), serde_json::json!("refused"));
    value.insert("published".into(), serde_json::json!(false));
    value.insert("exit_code".into(), serde_json::json!(refusal.exit));
    value.insert("operations".into(), serde_json::json!([]));
    value.insert("error".into(), serde_json::Value::Object(error));
    if let Some(path) = report_path {
        value.insert("report_path".into(), serde_json::json!(path));
    }
    serde_json::Value::Object(value)
}

pub(crate) fn run_trained_write(
    root_path: &std::path::Path,
    file_base: &std::path::Path,
    path: &str,
    bytes: Vec<u8>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    match std::fs::metadata(root_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(EditRefusal::new(
                "workspace_root_missing",
                format!(
                    "workspace root {} does not exist; nothing written. Create this directory, then retry with --root and a path relative to it",
                    root_path.display()
                ),
                17,
            ));
        }
        Err(error) => {
            return Err(EditRefusal::new(
                "workspace_root_unavailable",
                format!(
                    "cannot inspect workspace root {}: {error}; nothing written. Restore access to this directory before retrying",
                    root_path.display()
                ),
                17,
            ));
        }
        Ok(_) => {}
    }
    let (rel, abs) = edit_resolve_new_path(root_path, file_base, path)?;

    if abs.is_dir() {
        return Err(EditRefusal::new(
            "file_exists",
            format!("{path} is a directory, not a file"),
            13,
        ));
    }
    let before = std::fs::read(&abs).ok();
    if abs.exists() {
        edit_guard_path(root_path, &abs)?;
    } else if let Some(parent) = abs.parent() {
        let root = root_path
            .canonicalize()
            .unwrap_or_else(|_| root_path.to_path_buf());
        let mut existing = parent;
        while !existing.exists() {
            let Some(next) = existing.parent() else { break };
            existing = next;
        }
        let canonical = existing
            .canonicalize()
            .unwrap_or_else(|_| existing.to_path_buf());
        if !canonical.starts_with(&root) {
            return Err(EditRefusal::new(
                "path_outside_repo",
                format!("{path} is outside {}; nothing written. To edit another workspace, pass --root DIR and a path relative to DIR", root.display()),
                17,
            ));
        }
    }
    let old = before.as_deref().unwrap_or_default();
    edit_validate_syntax(&rel, old, &bytes)?;
    let mut record = edit_whole_file_record(root_path, &rel, &bytes, old, !dry_run);
    if before.as_deref() == Some(bytes.as_slice()) {
        record.already_as_sent = !dry_run;
        return Ok(record);
    }
    if dry_run {
        return Ok(record);
    }
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            EditRefusal::new(
                "publish_failed",
                format!("create {}: {error}", parent.display()),
                16,
            )
        })?;
    }
    let transaction = edit_journal_open(
        root_path,
        &[UndoBefore {
            rel: rel.clone(),
            content: before.clone(),
        }],
    );
    edit_journal_crash_hook()?;
    let publish = if let Some(old) = &before {
        greppy_edit::publish::publish_atomic(root_path, &abs, &bytes, &edit_sha256_hex(old))
            .map(|_| ())
            .map_err(|error| error.to_string())
    } else {
        use std::io::Write as _;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&abs)
            .and_then(|mut file| file.write_all(&bytes))
            .map_err(|error| error.to_string())
    };
    if let Err(error) = publish {
        edit_journal_abort(root_path);
        return Err(EditRefusal::new(
            "publish_failed",
            format!("{path}: {error}"),
            16,
        ));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
        record.transaction_id = Some(id);
    }
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

#[derive(Debug)]
struct TrainedPatchHunk {
    input_hunk_number: usize,
    input_line: usize,
    declared_old_line: usize,
    old_lines: Vec<String>,
    new_lines: Vec<String>,
}

#[derive(Debug)]
struct TrainedPatchFile {
    path: String,
    hunks: Vec<TrainedPatchHunk>,
}

fn trained_patch_path(header: &str) -> Option<String> {
    let raw = header.split_whitespace().next()?;
    if raw == "/dev/null" {
        return None;
    }
    Some(
        raw.strip_prefix("a/")
            .or_else(|| raw.strip_prefix("b/"))
            .unwrap_or(raw)
            .to_string(),
    )
}

fn parse_trained_patch(diff: &[u8]) -> EditResult<Vec<TrainedPatchFile>> {
    let text = std::str::from_utf8(diff)
        .map_err(|_| EditRefusal::new("invalid_patch", "the unified diff is not UTF-8", 20))?;
    let lines: Vec<&str> = text.lines().collect();
    let mut files = Vec::new();
    let mut index = 0usize;
    let mut input_hunk_number = 0usize;
    while index < lines.len() {
        if lines[index].starts_with("diff --git ") {
            // Git's next-file envelope is outside the preceding hunk. Accept
            // the optional object-ID line, but never silently omit a binary,
            // rename or mode-only section from an otherwise valid transaction.
            let section = lines[index];
            index += 1;
            if lines
                .get(index)
                .is_some_and(|line| line.starts_with("index "))
            {
                index += 1;
            }
            if !lines
                .get(index)
                .is_some_and(|line| line.starts_with("--- "))
            {
                let found = lines.get(index).copied().unwrap_or("end of input");
                return Err(EditRefusal::new(
                    "invalid_patch",
                    format!(
                        "Git patch section `{section}` requires textual ---/+++ headers after its optional index line; found `{found}`. Patch only edits existing file contents, not binary data, modes, renames or Git file creation/deletion metadata. Supply a text-only existing-file patch and handle unsupported operations separately; nothing written"
                    ),
                    20,
                ));
            }
        }
        if !lines[index].starts_with("--- ") {
            index += 1;
            continue;
        }
        let creates_file = lines[index][4..].split_whitespace().next() == Some("/dev/null");
        index += 1;
        let Some(next) = lines.get(index).filter(|line| line.starts_with("+++ ")) else {
            return Err(EditRefusal::new(
                "invalid_patch",
                "a --- file header is not followed by +++",
                20,
            ));
        };
        let Some(path) = trained_patch_path(&next[4..]) else {
            return Err(EditRefusal::new(
                "invalid_patch",
                "patch only edits existing files; file deletion is not supported — nothing written",
                20,
            ));
        };
        if creates_file {
            return Err(EditRefusal::new(
                "invalid_patch",
                format!(
                    "{path}: patch only edits existing files; to create a file, use `greppy write PATH` with its content on stdin. This is a separate transaction, not atomic with edits in this patch — nothing written"
                ),
                20,
            ));
        }
        index += 1;
        let mut hunks = Vec::new();
        while index < lines.len()
            && !lines[index].starts_with("--- ")
            && !lines[index].starts_with("diff --git ")
        {
            if !lines[index].starts_with("@@") {
                index += 1;
                continue;
            }
            input_hunk_number += 1;
            let input_line = index + 1;
            // Positions remain advisory; counts disambiguate actual file
            // headers from removed/added content beginning with ---/+++.
            let header_fields: Vec<&str> = lines[index].split_whitespace().collect();
            let declared_counts = (|| {
                let count = |field: &str, prefix| {
                    let range = field.strip_prefix(prefix)?;
                    let (start, count) = range.split_once(',').unwrap_or((range, "1"));
                    start.parse::<usize>().ok()?;
                    count.parse::<usize>().ok()
                };
                Some((
                    count(header_fields.get(1)?, '-')?,
                    count(header_fields.get(2)?, '+')?,
                ))
            })();
            let counted_header = header_fields
                .iter()
                .skip(1)
                .take(2)
                .any(|field| field.starts_with('-') || field.starts_with('+'));
            if counted_header && (declared_counts.is_none() || header_fields.get(3) != Some(&"@@"))
            {
                return Err(EditRefusal::new(
                    "invalid_patch",
                    format!("{path}: hunk {input_hunk_number} at patch input line {input_line} has invalid unified-diff ranges; use @@ -OLD,COUNT +NEW,COUNT @@ with non-negative integers — nothing written"),
                    20,
                ));
            }
            let declared_old_line = lines[index]
                .split_whitespace()
                .find(|field| field.starts_with('-'))
                .and_then(|field| field[1..].split(',').next())
                .and_then(|number| number.parse::<usize>().ok())
                .unwrap_or(1);
            index += 1;
            let mut old_lines = Vec::new();
            let mut new_lines = Vec::new();
            while index < lines.len()
                && !lines[index].starts_with("@@")
                && !lines[index].starts_with("diff --git ")
            {
                let line = lines[index];
                let header_pair = line.starts_with("--- ")
                    && lines
                        .get(index + 1)
                        .is_some_and(|next| next.starts_with("+++ "));
                if header_pair {
                    match declared_counts {
                        Some((old, new)) if old_lines.len() == old && new_lines.len() == new => {
                            break;
                        }
                        Some(_) => {} // Still inside the declared hunk: these are payload lines.
                        None if old_lines.is_empty() && new_lines.is_empty() => {}
                        None => {
                            return Err(EditRefusal::new(
                                "invalid_patch",
                                format!(
                                    "{path}: ---/+++ at patch input line {} is ambiguous in a count-free hunk; supply an explicit @@ -OLD,COUNT +NEW,COUNT @@ header to distinguish file headers from content — nothing written",
                                    index + 1
                                ),
                                20,
                            ));
                        }
                    }
                }
                match line.as_bytes().first() {
                    Some(b' ') => {
                        old_lines.push(line[1..].to_string());
                        new_lines.push(line[1..].to_string());
                    }
                    Some(b'-') => old_lines.push(line[1..].to_string()),
                    Some(b'+') => new_lines.push(line[1..].to_string()),
                    Some(b'\\') => {}
                    _ => {
                        return Err(EditRefusal::new(
                            "invalid_patch",
                            format!(
                                "{path}: patch input line {} has no unified-diff prefix; each hunk line must start with a space (context), '-' (removal), or '+' (addition). Regenerate the patch with `git diff --no-color -- PATH`; preserve prefixes on empty lines too — nothing written",
                                index + 1
                            ),
                            20,
                        ));
                    }
                }
                index += 1;
            }
            if old_lines.is_empty() {
                return Err(EditRefusal::new(
                    "invalid_patch",
                    if new_lines.is_empty() {
                        format!(
                            "{path}: hunk {input_hunk_number} at patch input line {input_line} is empty; remove its @@ header or add hunk content with context — nothing written"
                        )
                    } else {
                        format!(
                            "{path}: hunk {input_hunk_number} at patch input line {input_line} contains only additions and has no existing line to anchor on; include an unchanged context line — nothing written"
                        )
                    },
                    20,
                ));
            }
            if let Some((old, new)) = declared_counts {
                if old_lines.len() != old || new_lines.len() != new {
                    return Err(EditRefusal::new(
                        "invalid_patch",
                        format!(
                            "{path}: hunk {input_hunk_number} at patch input line {input_line} declares {old} old and {new} new lines, but contains {} old and {} new lines; regenerate the unified diff with correct counts — nothing written",
                            old_lines.len(),
                            new_lines.len()
                        ),
                        20,
                    ));
                }
            }
            hunks.push(TrainedPatchHunk {
                input_hunk_number,
                input_line,
                declared_old_line,
                old_lines,
                new_lines,
            });
        }
        if hunks.is_empty() {
            return Err(EditRefusal::new(
                "invalid_patch",
                format!("{path}: the diff carries no hunk"),
                20,
            ));
        }
        files.push(TrainedPatchFile { path, hunks });
    }
    if files.is_empty() {
        return Err(EditRefusal::new(
            "invalid_patch",
            "the diff carries no file header",
            20,
        ));
    }
    Ok(files)
}

fn apply_trained_patch_file(
    path: &str,
    content: &[u8],
    hunks: &[TrainedPatchHunk],
) -> EditResult<EditedContent> {
    let text = std::str::from_utf8(content)
        .map_err(|_| EditRefusal::new("invalid_patch", format!("{path} is not UTF-8"), 20))?;
    let line_texts: Vec<&str> = text.lines().collect();
    let mut line_ranges = Vec::new();
    let mut cursor = 0usize;
    while cursor < content.len() {
        let end = content[cursor..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|offset| cursor + offset + 1)
            .unwrap_or(content.len());
        line_ranges.push((cursor, end));
        cursor = end;
    }
    let ending = if content.windows(2).any(|pair| pair == b"\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let mut edits = Vec::new();
    for hunk in hunks {
        let candidates: Vec<usize> = line_texts
            .windows(hunk.old_lines.len())
            .enumerate()
            .filter(|(_, window)| {
                window
                    .iter()
                    .zip(&hunk.old_lines)
                    .all(|(actual, expected)| actual.trim_end_matches('\r') == expected)
            })
            .map(|(index, _)| index)
            .collect();
        let first = match candidates.as_slice() {
            [only] => *only,
            [] => {
                return Err(EditRefusal::new(
                    "patch_context",
                    format!(
                        "{path}: hunk context did not match (the @@ line {} is advisory) — nothing written",
                        hunk.declared_old_line
                    ),
                    13,
                ))
            }
            many => {
                let declared = hunk.declared_old_line.saturating_sub(1);
                if many.contains(&declared) {
                    declared
                } else {
                    const MAX_REPORTED_CANDIDATES: usize = 5;
                    let candidate_ranges = many
                        .iter()
                        .take(MAX_REPORTED_CANDIDATES)
                        .map(|start| {
                            let first_line = start + 1;
                            let last_line = start + hunk.old_lines.len();
                            if first_line == last_line {
                                first_line.to_string()
                            } else {
                                format!("{first_line}-{last_line}")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let omitted = many.len().saturating_sub(MAX_REPORTED_CANDIDATES);
                    let omitted_suffix = if omitted == 0 {
                        String::new()
                    } else {
                        format!(", and {omitted} more")
                    };
                    return Err(EditRefusal::new(
                        "patch_context",
                        format!(
                            "{path}: input hunk {} at patch line {} matches more than once (candidate source lines {candidate_ranges}{omitted_suffix}) — nothing written",
                            hunk.input_hunk_number, hunk.input_line
                        ),
                        13,
                    ));
                }
            }
        };
        let start = line_ranges[first].0;
        let end = line_ranges[first + hunk.old_lines.len() - 1].1;
        let had_final_ending = content[start..end].ends_with(b"\n");
        let mut replacement = hunk.new_lines.join(ending).into_bytes();
        if had_final_ending {
            replacement.extend_from_slice(ending.as_bytes());
        }
        edits.push((start, end, replacement));
    }
    edits.sort_by_key(|edit| edit.0);
    if edits.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(EditRefusal::new(
            "invalid_patch",
            format!("{path}: patch hunks overlap"),
            20,
        ));
    }
    Ok(edit_splice(content, &mut edits))
}

/// Undo only writes this invocation actually published. A failed CAS target
/// belongs to another writer and must never be restored from our pre-image.
fn rollback_patch_file(
    root: &std::path::Path,
    path: &std::path::Path,
    before: &[u8],
    published: &[u8],
) -> std::result::Result<(), String> {
    greppy_edit::publish::publish_atomic(root, path, before, &edit_sha256_hex(published))
        .map(|_| ())
        .map_err(|error| error.to_string())
}

pub(crate) fn run_trained_patch(
    root_path: &std::path::Path,
    file_base: &std::path::Path,
    diff: Vec<u8>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    run_trained_patch_with_publish_hook(root_path, file_base, diff, dry_run, verify, |_| {})
}

fn run_trained_patch_with_publish_hook(
    root_path: &std::path::Path,
    file_base: &std::path::Path,
    diff: Vec<u8>,
    dry_run: bool,
    verify: bool,
    mut before_publish: impl FnMut(usize),
) -> EditResult<EditRecord> {
    let parsed = parse_trained_patch(&diff)?;
    let mut targets = std::collections::HashSet::new();
    let mut planned = Vec::new();
    for file in parsed {
        let (rel, abs, content) = edit_read_file(root_path, file_base, &file.path)?;
        let target = std::fs::canonicalize(&abs).map_err(|error| {
            EditRefusal::new(
                "file_unreadable",
                format!("resolve {}: {error}", file.path),
                10,
            )
        })?;
        if !targets.insert(target) {
            return Err(EditRefusal::new(
                "invalid_patch",
                format!(
                    "{}: duplicate patch target; combine all hunks for this file under one ---/+++ header pair before retrying — nothing written",
                    file.path
                ),
                20,
            ));
        }
        let (after, changed) = apply_trained_patch_file(&rel, &content, &file.hunks)?;
        edit_validate_syntax(&rel, &content, &after)?;
        planned.push((rel, abs, content, after, changed));
    }
    let already = planned
        .iter()
        .all(|(_, _, before, after, _)| before == after);
    let exact_required = planned.len() > 1
        || planned
            .iter()
            .any(|(_, _, _, _, changed)| changed.len() > 1);
    let exact_addresses = planned
        .iter()
        .map(|(rel, _, _, after, changed)| edit_exact_address(rel, after, changed))
        .collect::<Vec<_>>();
    let mut record = EditRecord {
        files: planned
            .iter()
            .map(|(rel, _, _, _, _)| rel.clone())
            .collect(),
        span: planned.first().and_then(|(_, _, _, after, changed)| {
            let (start, end) = changed.first().copied()?;
            Some(edit_span_lines(after, start, end.saturating_sub(start)))
        }),
        published: !dry_run,
        already_as_sent: already && !dry_run,
        ..EditRecord::default()
    };
    for (rel, _, before, after, changed) in &planned {
        record.operations.push(EditOperation {
            file: rel.clone(),
            ranges: changed.clone(),
            sha_before: Some(edit_sha256_hex(before)),
            sha_after: Some(edit_sha256_hex(after)),
            diff: Some(edit_unified_diff(rel, before, after)),
            ..EditOperation::default()
        });
    }
    if already || dry_run {
        edit_set_exact_receipt(&mut record, exact_addresses, exact_required);
        return Ok(record);
    }
    let before: Vec<UndoBefore> = planned
        .iter()
        .map(|(rel, _, content, _, _)| UndoBefore {
            rel: rel.clone(),
            content: Some(content.clone()),
        })
        .collect();
    let transaction = edit_journal_open(root_path, &before);
    edit_journal_crash_hook()?;
    for (published_count, (_, abs, content, after, _)) in planned.iter().enumerate() {
        before_publish(published_count);
        if let Err(error) =
            greppy_edit::publish::publish_atomic(root_path, abs, after, &edit_sha256_hex(content))
        {
            let mut conflicts = Vec::new();
            for (rel, path, before, published, _) in planned[..published_count].iter().rev() {
                if let Err(reason) = rollback_patch_file(root_path, path, before, published) {
                    conflicts.push(format!("{rel}: {reason}"));
                }
            }
            let journal = edit_journal_dir(root_path);
            if let Some(pending) = edit_journal_read(&journal.join(EDIT_JOURNAL_PENDING)) {
                // Another invocation can replace pending.json. Never remove or
                // restore its journal on behalf of this failed transaction.
                if transaction
                    .as_deref()
                    .is_some_and(|id| pending["id"].as_str() == Some(id))
                {
                    if !conflicts.is_empty() {
                        // Keep evidence, but do not leave an unsafe automatic
                        // recovery candidate that could overwrite the conflict.
                        let id = transaction.as_deref().unwrap_or_default();
                        edit_journal_write(
                            &journal.join(format!("rollback-conflict-{id}.json")),
                            &serde_json::json!({
                                "transaction": pending,
                                "published_count": published_count,
                                "conflicts": conflicts,
                            }),
                        );
                    }
                    edit_journal_abort(root_path);
                }
            }
            let recovery = if conflicts.is_empty() {
                "earlier files written by this patch were rolled back; the failed target was left unchanged. Re-read the affected files before retrying".to_string()
            } else {
                format!("rollback refused to overwrite changed files: {}. Some patch writes may remain; inspect the affected files and rollback-conflict evidence in {} before retrying", conflicts.join("; "), journal.display())
            };
            return Err(EditRefusal::new(
                "publish_failed",
                format!("patch transaction failed: {error}; {recovery}"),
                16,
            ));
        }
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
        record.transaction_id = Some(id);
    }
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    edit_set_exact_receipt(&mut record, exact_addresses, exact_required);
    Ok(record)
}

pub(crate) fn edit_rename_receipt_addresses(
    root_path: &std::path::Path,
    certificate: &greppy_edit::certificate::Certificate,
    before: &[UndoBefore],
    old_name: &str,
    new_name: &str,
) -> Vec<String> {
    let mut by_file: std::collections::BTreeMap<String, Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();
    let length_delta = new_name.len() as i128 - old_name.len() as i128;
    for operation in &certificate.operations {
        let file = edit_operation_path(operation, root_path);
        let Some(content) = before
            .iter()
            .find(|entry| entry.rel == file)
            .and_then(|entry| entry.content.as_deref())
        else {
            by_file
                .entry(file)
                .or_default()
                .push(edit_operation_line_span(operation, root_path));
            continue;
        };
        let mut ranges = operation.changed_byte_ranges.clone();
        ranges.sort_unstable();
        for (index, (after_start, _)) in ranges.into_iter().enumerate() {
            // Rename replacements cannot add newlines. Translate each result
            // offset back through the preceding identifier-length shifts, then
            // read its unchanged line number from the before image. This also
            // keeps dry-run receipts exact, when the result is not on disk.
            let before_start = (after_start as i128 - length_delta * index as i128)
                .clamp(0, content.len() as i128) as usize;
            by_file
                .entry(file.clone())
                .or_default()
                .push(edit_span_lines(
                    content,
                    before_start,
                    old_name
                        .len()
                        .min(content.len().saturating_sub(before_start)),
                ));
        }
    }
    by_file
        .into_iter()
        .map(|(file, spans)| edit_format_line_address(&file, &edit_merge_line_spans(spans)))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenameEdgeIdentity {
    Related,
    Unrelated,
    Unknown,
}

fn rename_edge_identity(
    edge_type: &str,
    properties: &serde_json::Value,
    owner: Option<&str>,
    name: &str,
) -> RenameEdgeIdentity {
    let known_path_owner = |property: &str| {
        properties
            .get(property)
            .and_then(|value| value.as_str())
            .filter(|path| !path.is_empty())
            .and_then(|path| path.strip_suffix(name))
            .and_then(|path| path.strip_suffix("::"))
            .filter(|owner| !owner.is_empty())
    };
    let owner_relation = |candidate: &str| match owner {
        Some(selected) if candidate == selected => RenameEdgeIdentity::Related,
        Some(selected)
            if candidate.ends_with(&format!("::{selected}"))
                || selected.ends_with(&format!("::{candidate}")) =>
        {
            RenameEdgeIdentity::Unknown
        }
        Some(_) => RenameEdgeIdentity::Unrelated,
        None => RenameEdgeIdentity::Unknown,
    };
    match edge_type {
        "CALLS"
            if properties
                .get("callee_form")
                .and_then(|value| value.as_str())
                == Some("receiver") =>
        {
            match properties
                .get("receiver_owner")
                .and_then(|value| value.as_str())
            {
                Some(candidate) if !candidate.is_empty() => owner_relation(candidate),
                Some(_) => RenameEdgeIdentity::Unknown,
                None => RenameEdgeIdentity::Unknown,
            }
        }
        "CALLS" if owner == Some("Function") => RenameEdgeIdentity::Related,
        "CALLS" if known_path_owner("callee_path").is_some() => {
            owner_relation(known_path_owner("callee_path").unwrap())
        }
        "CALLS" => RenameEdgeIdentity::Unknown,
        "USAGE" | "USES" if owner == Some("Function") => RenameEdgeIdentity::Related,
        "USAGE" | "USES" if known_path_owner("ref_path").is_some() => {
            owner_relation(known_path_owner("ref_path").unwrap())
        }
        "USAGE" | "USES" => RenameEdgeIdentity::Unknown,
        "TYPE_REF" | "IMPORTS" => RenameEdgeIdentity::Related,
        _ => RenameEdgeIdentity::Unrelated,
    }
}

fn rust_rename_reference_inventory(
    root_path: &std::path::Path,
    scopes: &std::collections::BTreeMap<String, Vec<(usize, usize)>>,
    owner: &str,
    short_name: &str,
    symbol: &str,
) -> EditResult<()> {
    let files = greppy_discover::walk(root_path).map_err(|error| {
        EditRefusal::new(
            "unresolved_reference",
            format!("cannot inventory Rust references for `{symbol}`: {error} — nothing written"),
            12,
        )
    })?;
    for entry in files {
        if std::path::Path::new(&entry.rel_path)
            .extension()
            .and_then(|value| value.to_str())
            != Some("rs")
        {
            continue;
        }
        let rel = entry.rel_path;
        let content = greppy_discover::read_stable_file(&entry.abs_path)
            .map(|(content, _)| content)
            .map_err(|error| {
                EditRefusal::new(
                    "unresolved_reference",
                    format!(
                        "cannot read {rel} while inventorying `{symbol}`: {error} — nothing written"
                    ),
                    12,
                )
            })?;
        let extraction = greppy_parser::extract(greppy_parser::Language::Rust, &content, &rel)
            .map_err(|error| {
                EditRefusal::new(
                    "unresolved_reference",
                    format!("cannot parse {rel} while inventorying `{symbol}`: {error} — nothing written"),
                    12,
                )
            })?;
        for edge in extraction.edges {
            let property = match edge.edge_type.as_str() {
                "CALLS" => "callee_name",
                "USAGE" | "USES" => "ref_name",
                _ => continue,
            };
            if edge
                .properties
                .get(property)
                .and_then(|value| value.as_str())
                .and_then(|value| value.rsplit("::").next())
                != Some(short_name)
            {
                continue;
            }
            let line_range = line_range_to_bytes(&content, edge.line as usize, edge.line as usize);
            let sites = greppy_edit::verbs::rename_identifier_sites(
                std::path::Path::new(&rel),
                &content,
                &[line_range],
                short_name,
            )
            .ok_or_else(|| {
                EditRefusal::new(
                    "unresolved_reference",
                    format!("cannot parse live Rust reference in {rel} for `{symbol}` — nothing written"),
                    12,
                )
            })?;
            if sites.is_empty() {
                continue;
            }
            match rename_edge_identity(&edge.edge_type, &edge.properties, Some(owner), short_name) {
                RenameEdgeIdentity::Unrelated => continue,
                RenameEdgeIdentity::Unknown => {
                    return Err(EditRefusal::new(
                        "unresolved_reference_identity",
                        format!("live Rust reference in {rel}:{} lacks identity proving whether it targets `{symbol}` — nothing written", edge.line),
                        12,
                    ));
                }
                RenameEdgeIdentity::Related => {}
            }
            let planned = scopes.get(&rel).map(Vec::as_slice).unwrap_or_default();
            if sites.iter().any(|site| !planned.contains(site)) {
                return Err(EditRefusal::new(
                    "unresolved_reference",
                    format!("live Rust reference in {rel}:{} targets `{symbol}` but is absent from the graph rename plan — refresh the index; nothing written", edge.line),
                    12,
                ));
            }
        }
    }
    Ok(())
}

fn rust_free_function_reference_inventory(
    root_path: &std::path::Path,
    scopes: &std::collections::BTreeMap<String, Vec<(usize, usize)>>,
    selected_files: &std::collections::BTreeSet<String>,
    short_name: &str,
    symbol: &str,
) -> EditResult<()> {
    let files = greppy_discover::walk(root_path).map_err(|error| {
        EditRefusal::new(
            "unresolved_reference",
            format!("cannot inventory Rust references for `{symbol}`: {error} — nothing written"),
            12,
        )
    })?;
    for entry in files {
        if std::path::Path::new(&entry.rel_path)
            .extension()
            .and_then(|value| value.to_str())
            != Some("rs")
        {
            continue;
        }
        let rel = entry.rel_path;
        let content = greppy_discover::read_stable_file(&entry.abs_path)
            .map(|(content, _)| content)
            .map_err(|error| {
                EditRefusal::new(
                    "unresolved_reference",
                    format!("cannot read {rel} while inventorying `{symbol}`: {error} — nothing written"),
                    12,
                )
            })?;
        let extraction = greppy_parser::extract(greppy_parser::Language::Rust, &content, &rel)
            .map_err(|error| {
                EditRefusal::new(
                    "unresolved_reference",
                    format!("cannot parse {rel} while inventorying `{symbol}`: {error} — nothing written"),
                    12,
                )
        })?;
        for edge in extraction.edges {
            let import_items = edge
                .properties
                .get("imported_items")
                .and_then(serde_json::Value::as_array);
            let (name_property, path_property) = match edge.edge_type.as_str() {
                "CALLS" => ("callee_name", "callee_path"),
                "USAGE" | "USES" => ("ref_name", "ref_path"),
                "IMPORTS" => ("imported_name", "path"),
                _ => continue,
            };
            let named_reference = edge
                .properties
                .get(name_property)
                .and_then(|value| value.as_str())
                .and_then(|value| value.rsplit("::").next())
                == Some(short_name);
            let grouped_import_reference = edge.edge_type == "IMPORTS"
                && import_items.is_some_and(|items| {
                    items.iter().any(|item| {
                        ["imported_name", "original_name"]
                            .into_iter()
                            .any(|property| {
                                item.get(property)
                                    .and_then(serde_json::Value::as_str)
                                    .and_then(|value| value.rsplit("::").next())
                                    == Some(short_name)
                            })
                    })
                });
            if !named_reference && !grouped_import_reference {
                continue;
            }
            let line_range = line_range_to_bytes(&content, edge.line as usize, edge.line as usize);
            let sites = greppy_edit::verbs::rename_identifier_sites(
                std::path::Path::new(&rel),
                &content,
                &[line_range],
                short_name,
            )
            .ok_or_else(|| {
                EditRefusal::new(
                    "unresolved_reference",
                    format!("cannot parse live Rust reference in {rel} for `{symbol}` — nothing written"),
                    12,
                )
            })?;
            if sites.is_empty() {
                continue;
            }
            let planned = scopes.get(&rel).map(Vec::as_slice).unwrap_or_default();
            if sites.iter().all(|site| planned.contains(site)) {
                continue;
            }
            let reference_path = edge
                .properties
                .get(path_property)
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty());
            let unqualified = reference_path.is_none() || reference_path == Some(short_name);
            let unplanned = sites
                .iter()
                .copied()
                .filter(|site| !planned.contains(site))
                .collect::<Vec<_>>();
            if !selected_files.contains(&rel)
                && unqualified
                && unplanned.iter().all(|site| {
                    rust_local_free_function_owns_site(&content, short_name, *site, false)
                })
            {
                continue;
            }
            return Err(EditRefusal::new(
                "unresolved_reference_identity",
                format!("live Rust free-function reference in {rel}:{} is not proven to target `{symbol}` or a distinct local definition — refresh the index; nothing written", edge.line),
                12,
            ));
        }
    }
    Ok(())
}

fn rust_local_free_function_owns_site(
    content: &[u8],
    short_name: &str,
    site: (usize, usize),
    glob_import_shadows: bool,
) -> bool {
    let Ok(tree) = greppy_parser::parse(greppy_parser::Language::Rust, content) else {
        return false;
    };
    let Some(reference) = tree
        .root_node()
        .descendant_for_byte_range(site.0, site.1.saturating_sub(1).max(site.0))
    else {
        return false;
    };
    let mut reference_blocks = std::collections::BTreeSet::new();
    let mut reference_module = None;
    let mut ancestor = Some(reference);
    while let Some(node) = ancestor {
        if node.kind() == "block" {
            reference_blocks.insert((node.start_byte(), node.end_byte()));
        } else if reference_module.is_none() && matches!(node.kind(), "source_file" | "mod_item") {
            reference_module = Some((node.start_byte(), node.end_byte()));
        }
        ancestor = node.parent();
    }

    // A same-name import or local binding in an active lexical scope may
    // shadow an otherwise visible module function. Without name resolution it
    // is not positive evidence that the unqualified reference is local.
    let mut shadow_stack = vec![tree.root_node()];
    while let Some(node) = shadow_stack.pop() {
        if matches!(
            node.kind(),
            "use_declaration" | "let_declaration" | "parameter"
        ) && (node.kind() == "use_declaration" || node.start_byte() <= reference.start_byte())
        {
            let (active_scope, active_module_scope) = if node.kind() == "parameter" {
                let mut owner = node.parent();
                let mut active = false;
                while let Some(scope) = owner {
                    if scope.kind() == "function_item" {
                        active = scope.child_by_field_name("body").is_some_and(|body| {
                            reference_blocks.contains(&(body.start_byte(), body.end_byte()))
                        });
                        break;
                    }
                    owner = scope.parent();
                }
                (active, false)
            } else {
                let mut owner = node.parent();
                let mut active = false;
                let mut module_scope = false;
                while let Some(scope) = owner {
                    if matches!(scope.kind(), "source_file" | "mod_item" | "block") {
                        let key = (scope.start_byte(), scope.end_byte());
                        active = if scope.kind() == "block" {
                            reference_blocks.contains(&key)
                        } else {
                            reference_module == Some(key)
                        };
                        module_scope = active && scope.kind() != "block";
                        break;
                    }
                    owner = scope.parent();
                }
                (active, module_scope)
            };
            if active_scope {
                // Only the pattern binds a name. Calls in a let initializer
                // and names in a parameter type cannot shadow the function.
                let binding = if matches!(node.kind(), "let_declaration" | "parameter") {
                    node.child_by_field_name("pattern").unwrap_or(node)
                } else {
                    node
                };
                let mut declaration_stack = vec![binding];
                while let Some(part) = declaration_stack.pop() {
                    if node.kind() == "use_declaration"
                        && matches!(part.kind(), "use_wildcard" | "wildcard_import")
                        && (glob_import_shadows || !active_module_scope)
                    {
                        return false;
                    }
                    if matches!(part.kind(), "identifier" | "field_identifier")
                        && content.get(part.byte_range()) == Some(short_name.as_bytes())
                    {
                        return false;
                    }
                    let mut cursor = part.walk();
                    declaration_stack.extend(part.named_children(&mut cursor));
                }
            }
        }
        let mut cursor = node.walk();
        shadow_stack.extend(node.named_children(&mut cursor));
    }

    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if node.kind() == "function_item"
            && node
                .child_by_field_name("name")
                .and_then(|name| content.get(name.byte_range()))
                == Some(short_name.as_bytes())
        {
            let mut owner = node.parent();
            let mut associated = false;
            while let Some(scope) = owner {
                if matches!(scope.kind(), "impl_item" | "trait_item") {
                    associated = true;
                }
                if matches!(scope.kind(), "source_file" | "mod_item" | "block") {
                    if associated {
                        break;
                    }
                    let key = (scope.start_byte(), scope.end_byte());
                    let owns = if scope.kind() == "block" {
                        reference_blocks.contains(&key)
                    } else {
                        reference_module == Some(key)
                    };
                    if owns {
                        return true;
                    }
                    break;
                }
                owner = scope.parent();
            }
        }
        let mut cursor = node.walk();
        stack.extend(node.named_children(&mut cursor));
    }
    false
}

fn require_rename_edge_identity(
    edge_type: &str,
    properties: &serde_json::Value,
    owner: Option<&str>,
    name: &str,
    symbol: &str,
    source_id: i64,
) -> EditResult<bool> {
    match rename_edge_identity(edge_type, properties, owner, name) {
        RenameEdgeIdentity::Related => Ok(true),
        RenameEdgeIdentity::Unrelated => Ok(false),
        RenameEdgeIdentity::Unknown => Err(EditRefusal::new(
            "unresolved_reference_identity",
            format!(
                "graph reference to `{symbol}` lacks owner identity at source node {source_id}; refresh the index or select a more specific symbol — nothing written"
            ),
            12,
        )),
    }
}

fn select_rename_reference_site(
    symbol: &str,
    short_name: &str,
    file_path: &str,
    sites: &[(usize, usize)],
) -> EditResult<Option<(usize, usize)>> {
    match sites {
        // A structurally valid source span with no live old identifier is a
        // stale graph candidate, not an unresolved source reference.
        [] => Ok(None),
        [site] => Ok(Some(*site)),
        _ => Err(EditRefusal::new(
            "ambiguous_reference",
            format!(
                "graph reference scope {file_path} contains {} live `{short_name}` identifiers for `{symbol}`; select a narrower symbol or refresh the index — nothing written",
                sites.len()
            ),
            12,
        )),
    }
}

pub(crate) fn run_trained_rename(
    root_path: &std::path::Path,
    root: Option<&str>,
    symbol: &str,
    new_name: &str,
    dry_run: bool,
    verify: bool,
) -> Result<EditResult<EditRecord>> {
    let store = open_default_store_query_writer(root)?;
    let ids = match resolve_symbol_nodes(&store, Some(symbol)) {
        Ok(ids) => ids,
        Err(_) => {
            return Ok(Err(EditRefusal::new(
                "symbol_not_found",
                format!("no symbol `{symbol}`"),
                10,
            )))
        }
    };
    let mut def_nodes = Vec::new();
    for id in &ids {
        if let Some(node) = store.get_node(*id)? {
            if !node.file_path.is_empty() && node.start_line >= 1 {
                def_nodes.push(node);
            }
        }
    }
    if def_nodes.is_empty() {
        return Ok(Err(EditRefusal::new(
            "symbol_not_found",
            format!("no symbol `{symbol}`"),
            10,
        )));
    }
    let short_name = def_nodes[0].name.clone();
    use std::collections::BTreeMap;
    let mut scopes: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    let first_owner = def_nodes[0].qualified_name.rsplit("::").nth(1);
    let rust_method_inventory_eligible = first_owner.is_some()
        && def_nodes.iter().all(|def| {
            def.label == "Method"
                && def.file_path.ends_with(".rs")
                && def.qualified_name.rsplit("::").nth(1) == first_owner
        });
    let rust_free_function_inventory_eligible = def_nodes.iter().all(|def| {
        def.label == "Function"
            && def.file_path.ends_with(".rs")
            && symbol.starts_with(&format!("{}::", def.file_path))
    });
    let rust_inventory_eligible =
        rust_method_inventory_eligible || rust_free_function_inventory_eligible;
    let rust_method_owner = rust_method_inventory_eligible.then(|| first_owner.unwrap().to_owned());
    let rust_selected_files = def_nodes
        .iter()
        .map(|def| def.file_path.clone())
        .collect::<std::collections::BTreeSet<_>>();
    for def in &def_nodes {
        let owner = def.qualified_name.rsplit("::").nth(1);
        if !rust_inventory_eligible {
            scopes
                .entry(def.file_path.clone())
                .or_default()
                .push((0, usize::MAX));
            for edge in store.incoming_edges(def.id, None, 100_000)? {
                let Some(source) = store.get_node(edge.source_id)? else {
                    continue;
                };
                if source.file_path.is_empty() || source.start_line < 1 {
                    continue;
                }
                let Ok(content) = std::fs::read(root_path.join(&source.file_path)) else {
                    continue;
                };
                let Some(span) = read_span_with_meta(
                    root_path,
                    &source.file_path,
                    source.start_line,
                    source.end_line,
                    usize::MAX,
                    false,
                ) else {
                    continue;
                };
                scopes
                    .entry(source.file_path.clone())
                    .or_default()
                    .push(line_range_to_bytes(
                        &content,
                        source.start_line as usize,
                        span.end_line as usize,
                    ));
            }
            continue;
        }
        let content = std::fs::read(root_path.join(&def.file_path))
            .map_err(|error| Error::io(format!("read {} for rename", def.file_path), error))?;
        let Some(span) = read_span_with_meta(
            root_path,
            &def.file_path,
            def.start_line,
            def.end_line,
            usize::MAX,
            false,
        ) else {
            return Ok(Err(EditRefusal::new(
                "symbol_not_found",
                format!("selected definition `{symbol}` no longer has a readable source span"),
                10,
            )));
        };
        let definition_range =
            line_range_to_bytes(&content, def.start_line as usize, span.end_line as usize);
        let definition_sites = greppy_edit::verbs::rename_definition_sites(
            std::path::Path::new(&def.file_path),
            &content,
            definition_range,
            &short_name,
        );
        let Some([definition_scope]) = definition_sites.as_deref() else {
            return Ok(Err(EditRefusal::new(
                "ambiguous_symbol",
                format!(
                    "selected definition `{symbol}` does not have one unique live `{short_name}` identifier in its indexed span"
                ),
                12,
            )));
        };
        scopes
            .entry(def.file_path.clone())
            .or_default()
            .push(*definition_scope);
        for edge in store.incoming_edges(def.id, None, 100_000)? {
            let property = match edge.edge_type.as_str() {
                "CALLS" => "callee_name",
                "USAGE" | "USES" => "ref_name",
                "TYPE_REF" => "type_name",
                "IMPORTS" => "imported_name",
                _ => continue,
            };
            let Some(source) = store.get_node(edge.source_id)? else {
                return Ok(Err(EditRefusal::new(
                    "unresolved_reference",
                    format!("graph reference to `{symbol}` has no source node — nothing written"),
                    12,
                )));
            };
            if source.file_path.is_empty() || source.start_line < 1 {
                return Ok(Err(EditRefusal::new(
                    "unresolved_reference",
                    format!("graph reference to `{symbol}` has no readable source location — nothing written"),
                    12,
                )));
            }
            let Ok(content) = std::fs::read(root_path.join(&source.file_path)) else {
                return Ok(Err(EditRefusal::new(
                    "unresolved_reference",
                    format!(
                        "cannot read graph reference source {} for `{symbol}` — nothing written",
                        source.file_path
                    ),
                    12,
                )));
            };
            let Some(span) = read_span_with_meta(
                root_path,
                &source.file_path,
                source.start_line,
                source.end_line,
                usize::MAX,
                false,
            ) else {
                return Ok(Err(EditRefusal::new(
                    "unresolved_reference",
                    format!("cannot resolve graph reference source span {} for `{symbol}` — nothing written", source.file_path),
                    12,
                )));
            };
            let selected_local_call_scope = rust_free_function_inventory_eligible
                && edge.edge_type == "CALLS"
                && rust_selected_files.contains(&source.file_path)
                && matches!(source.label.as_str(), "Function" | "Method");
            let range = if selected_local_call_scope {
                line_range_to_bytes(&content, source.start_line as usize, span.end_line as usize)
            } else if rust_free_function_inventory_eligible {
                edge.properties
                    .get("line")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|line| usize::try_from(line).ok())
                    .filter(|line| *line >= 1)
                    .map(|line| line_range_to_bytes(&content, line, line))
                    .unwrap_or_else(|| {
                        line_range_to_bytes(
                            &content,
                            source.start_line as usize,
                            span.end_line as usize,
                        )
                    })
            } else {
                line_range_to_bytes(&content, source.start_line as usize, span.end_line as usize)
            };
            let Some(sites) = greppy_edit::verbs::rename_identifier_sites(
                std::path::Path::new(&source.file_path),
                &content,
                &[range],
                &short_name,
            ) else {
                return Ok(Err(EditRefusal::new(
                    "unresolved_reference",
                    format!(
                        "cannot parse graph reference scope {} for `{symbol}`; nothing written",
                        source.file_path
                    ),
                    12,
                )));
            };
            if sites.is_empty() {
                continue;
            }
            let Some(reference_name) = edge
                .properties
                .get(property)
                .and_then(|value| value.as_str())
            else {
                return Ok(Err(EditRefusal::new(
                    "unresolved_reference_identity",
                    format!(
                        "live graph reference to `{symbol}` at source node {} lacks `{property}` identity; refresh the index — nothing written",
                        edge.source_id
                    ),
                    12,
                )));
            };
            if reference_name.rsplit("::").next() != Some(short_name.as_str()) {
                continue;
            }
            match require_rename_edge_identity(
                &edge.edge_type,
                &edge.properties,
                owner,
                &short_name,
                symbol,
                edge.source_id,
            ) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(refusal) => return Ok(Err(refusal)),
            }
            if selected_local_call_scope
                && sites.iter().all(|site| {
                    rust_selected_local_free_function_owns_site(
                        &content,
                        &short_name,
                        *definition_scope,
                        *site,
                    )
                })
            {
                scopes
                    .entry(source.file_path.clone())
                    .or_default()
                    .extend(sites);
                continue;
            }
            match select_rename_reference_site(symbol, &short_name, &source.file_path, &sites) {
                Ok(None) => continue,
                Ok(Some(site)) => scopes
                    .entry(source.file_path.clone())
                    .or_default()
                    .push(site),
                Err(refusal) => return Ok(Err(refusal)),
            }
        }
    }
    if rust_method_inventory_eligible {
        if let Err(refusal) = rust_rename_reference_inventory(
            root_path,
            &scopes,
            rust_method_owner.as_deref().expect("eligible method owner"),
            &short_name,
            symbol,
        ) {
            return Ok(Err(refusal));
        }
    } else if rust_free_function_inventory_eligible {
        if let Err(refusal) = rust_free_function_reference_inventory(
            root_path,
            &scopes,
            &rust_selected_files,
            &short_name,
            symbol,
        ) {
            return Ok(Err(refusal));
        }
    }
    let scope_vec: Vec<greppy_edit::verbs::RenameFileScope> = scopes
        .into_iter()
        .map(|(rel_path, mut spans)| {
            spans.sort_unstable();
            spans.dedup();
            greppy_edit::verbs::RenameFileScope { rel_path, spans }
        })
        .collect();
    let before: Vec<UndoBefore> = scope_vec
        .iter()
        .map(|scope| UndoBefore {
            rel: scope.rel_path.clone(),
            content: std::fs::read(root_path.join(&scope.rel_path)).ok(),
        })
        .collect();
    let options = greppy_edit::verbs::VerbOptions {
        dry_run,
        with_diff: true,
        expect_residual: Some(0),
        ..Default::default()
    };
    let certificate = if rust_inventory_eligible {
        greppy_edit::verbs::rename_symbol_files_scoped(
            root_path,
            &scope_vec,
            &short_name,
            new_name,
            &options,
        )?
    } else {
        greppy_edit::verbs::rename_symbol_files(
            root_path,
            &scope_vec,
            &short_name,
            new_name,
            &options,
        )?
    };
    if certificate.exit_code() != 0 {
        let message = certificate.compact_failure_diagnosis().unwrap_or_else(|| {
            format!(
                "rename {} — nothing written",
                edit_status_name(certificate.status)
            )
        });
        return Ok(Err(EditRefusal::new(
            certificate_refusal_code(&certificate),
            message,
            certificate.exit_code(),
        )));
    }
    let exact_required = certificate.operations.len() > 1
        || certificate
            .operations
            .iter()
            .any(|operation| operation.changed_byte_ranges.len() > 1);
    let exact_addresses =
        edit_rename_receipt_addresses(root_path, &certificate, &before, &short_name, new_name);
    let mut files: Vec<String> = certificate
        .operations
        .iter()
        .map(|operation| edit_operation_path(operation, root_path))
        .collect();
    files.sort();
    files.dedup();
    let span = certificate
        .operations
        .first()
        .map(|operation| edit_operation_line_span(operation, root_path));
    let already = certificate.status == greppy_edit::Status::AlreadySatisfied;
    // Preserve the planner's exact change witness for both previews and writes.
    // Rename receipts need ranges/checksums, not another copy of function bodies.
    let operations = certificate
        .operations
        .iter()
        .map(|operation| EditOperation {
            file: edit_operation_path(operation, root_path),
            ranges: operation.changed_byte_ranges.clone(),
            sha_before: Some(operation.file_sha256_before.clone()),
            sha_after: operation.file_sha256_after.clone(),
            diff: operation.unified_diff.clone(),
            ..EditOperation::default()
        })
        .collect();
    let mut record = EditRecord {
        files,
        span,
        operations,
        published: !dry_run,
        already_as_sent: already && !dry_run,
        ..EditRecord::default()
    };
    if certificate.published {
        if let Some(id) = edit_journal_open(root_path, &before) {
            edit_journal_close(root_path, &id);
            record.transaction_id = Some(id);
        }
    }
    if verify && certificate.published {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    edit_set_exact_receipt(&mut record, exact_addresses, exact_required);
    Ok(Ok(record))
}

fn rust_selected_local_free_function_owns_site(
    content: &[u8],
    short_name: &str,
    definition_site: (usize, usize),
    reference_site: (usize, usize),
) -> bool {
    // A same-module item is resolved ahead of glob imports. Keep glob imports
    // conservative when proving an unrelated local definition, but do not let
    // them hide calls to the selected item in its own module.
    if !rust_local_free_function_owns_site(content, short_name, reference_site, false) {
        return false;
    }
    let Ok(tree) = greppy_parser::parse(greppy_parser::Language::Rust, content) else {
        return false;
    };
    let node_at = |site: (usize, usize)| {
        tree.root_node()
            .descendant_for_byte_range(site.0, site.1.saturating_sub(1).max(site.0))
    };
    let (Some(definition), Some(reference)) = (node_at(definition_site), node_at(reference_site))
    else {
        return false;
    };
    let mut call = Some(reference);
    let mut unqualified_call = false;
    while let Some(node) = call {
        if node.kind() == "call_expression" {
            unqualified_call = node
                .child_by_field_name("function")
                .is_some_and(|function| function.byte_range() == reference.byte_range());
            break;
        }
        if matches!(node.kind(), "scoped_identifier" | "field_expression") {
            break;
        }
        call = node.parent();
    }
    if !unqualified_call {
        return false;
    }
    let mut definition_module = None;
    let mut ancestor = Some(definition);
    while let Some(node) = ancestor {
        if matches!(node.kind(), "source_file" | "mod_item") {
            definition_module = Some((node.start_byte(), node.end_byte()));
            break;
        }
        ancestor = node.parent();
    }
    let mut reference_module = None;
    let mut ancestor = Some(reference);
    while let Some(node) = ancestor {
        if matches!(node.kind(), "source_file" | "mod_item") {
            reference_module = Some((node.start_byte(), node.end_byte()));
            break;
        }
        ancestor = node.parent();
    }
    if definition_module.is_none() || definition_module != reference_module {
        return false;
    }

    // Cover binding forms that are not ordinary function parameters or `let`
    // declarations: for/match/if-let/while-let patterns and closure parameters.
    // Any earlier same-name binding whose lexical owner contains this call
    // makes the call's target ambiguous without name resolution.
    let mut binding_stack = vec![tree.root_node()];
    while let Some(candidate) = binding_stack.pop() {
        if matches!(candidate.kind(), "identifier" | "field_identifier")
            && candidate.start_byte() < reference.start_byte()
            && content.get(candidate.byte_range()) == Some(short_name.as_bytes())
        {
            let mut ancestor = candidate.parent();
            let mut binding_owner = None;
            while let Some(node) = ancestor {
                let binds_candidate = match node.kind() {
                    "parameter" | "closure_parameters" => true,
                    "let_declaration" | "for_expression" | "match_arm" | "let_condition" => {
                        node.child_by_field_name("pattern").is_some_and(|pattern| {
                            pattern.byte_range().contains(&candidate.start_byte())
                        })
                    }
                    _ => false,
                };
                if binds_candidate {
                    binding_owner = Some(node);
                    break;
                }
                if matches!(node.kind(), "function_item" | "mod_item" | "source_file") {
                    break;
                }
                ancestor = node.parent();
            }
            if let Some(binding) = binding_owner {
                let mut scope = Some(binding);
                while let Some(node) = scope {
                    if matches!(
                        node.kind(),
                        "block"
                            | "for_expression"
                            | "match_arm"
                            | "if_expression"
                            | "while_expression"
                            | "closure_expression"
                            | "function_item"
                    ) && node.byte_range().contains(&reference.start_byte())
                    {
                        return false;
                    }
                    scope = node.parent();
                }
            }
        }
        let mut cursor = candidate.walk();
        binding_stack.extend(candidate.named_children(&mut cursor));
    }

    // A nested function item can shadow the selected module function for only
    // part of the caller. Refuse expansion for any site inside such a block;
    // the persisted edge's exact line remains the only proven site.
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if node.kind() == "function_item"
            && node
                .child_by_field_name("name")
                .and_then(|name| content.get(name.byte_range()))
                == Some(short_name.as_bytes())
            && !node.byte_range().contains(&definition_site.0)
        {
            let mut owner = node.parent();
            while let Some(scope) = owner {
                if scope.kind() == "block" {
                    if scope.byte_range().contains(&reference_site.0) {
                        return false;
                    }
                    break;
                }
                if matches!(scope.kind(), "source_file" | "mod_item") {
                    break;
                }
                owner = scope.parent();
            }
        }
        let mut cursor = node.walk();
        stack.extend(node.named_children(&mut cursor));
    }
    true
}

/// A full Rust replacement that supplies outer attributes owns those attributes
/// too. Without supplied attributes, preserve the existing prefix as before;
/// body-only edits never call this helper.
fn edit_rust_attribute_replacement_range(
    content: &[u8],
    range: (usize, usize),
    replacement: &[u8],
) -> (usize, usize) {
    let Ok(requested) = greppy_parser::parse(greppy_edit::Language::Rust, replacement) else {
        return range;
    };
    let mut cursor = requested.root_node().walk();
    let supplies_attributes = requested
        .root_node()
        .named_children(&mut cursor)
        .find_map(|node| match node.kind() {
            "attribute_item" => Some(true),
            "line_comment" | "block_comment" => None,
            _ => Some(false),
        })
        .unwrap_or(false);
    if !supplies_attributes {
        return range;
    }
    let Ok(tree) = greppy_parser::parse(greppy_edit::Language::Rust, content) else {
        return range;
    };
    let Some(offset) = content
        .get(range.0..range.1)
        .and_then(|bytes| bytes.iter().position(|byte| !byte.is_ascii_whitespace()))
    else {
        return range;
    };
    let Some(mut node) = tree
        .root_node()
        .descendant_for_byte_range(range.0 + offset, range.1.saturating_sub(1))
    else {
        return range;
    };
    while !matches!(
        node.kind(),
        "function_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "type_item"
            | "impl_item"
            | "mod_item"
            | "const_item"
            | "static_item"
    ) {
        let Some(parent) = node.parent() else {
            return range;
        };
        node = parent;
    }
    if node.end_byte() > range.1 {
        return range;
    }
    let mut start = node.start_byte();
    let mut previous = node.prev_named_sibling();
    while let Some(prefix) = previous {
        match prefix.kind() {
            "attribute_item" => start = prefix.start_byte(),
            "line_comment" | "block_comment" => {}
            _ => break,
        }
        previous = prefix.prev_named_sibling();
    }
    let line_start = content[..start]
        .iter()
        .rposition(|&byte| byte == b'\n')
        .map_or(0, |offset| offset + 1);
    if content[line_start..start]
        .iter()
        .all(|byte| byte.is_ascii_whitespace())
    {
        start = line_start;
    }
    (start.min(range.0), range.1)
}

pub(crate) fn dispatch_edit_grammar(
    command: EditCommand,
    json: bool,
    root: Option<&str>,
    root_path: &std::path::Path,
    file_base: &std::path::Path,
) -> Result<GrammarDispatch> {
    let code = match command {
        EditCommand::Replace {
            symbol,
            new,
            body,
            dry_run,
            verify,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let mut new_bytes = edit_positional_payload(new, "NEW")?;
                let spec = WhereSpec {
                    file: None,
                    old: None,
                    old_file: None,
                    pattern: None,
                    lines: None,
                    symbol: Some(symbol),
                    body,
                    target: None,
                    path: None,
                };
                let mut located =
                    edit_locate(&spec, SelectorKind::Symbol, root, root_path, file_base)?;
                if !body
                    && greppy_edit::language_for_path(std::path::Path::new(&located.rel))
                        == greppy_edit::Language::Rust
                {
                    edit_check_cardinality(&located, Some(1))?;
                    located.ranges[0] = edit_rust_attribute_replacement_range(
                        &located.content,
                        located.ranges[0],
                        &new_bytes,
                    );
                }
                if body {
                    edit_check_cardinality(&located, Some(1))?;
                    let (start, end) = located.ranges[0];
                    new_bytes = greppy_edit::verbs::replacement_body_preserving_delimiters(
                        &located.content[start..end],
                        &new_bytes,
                    );
                }
                let (new_content, changed) = edit_op_replace(&located, &new_bytes)?;
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::ReplaceText {
            file,
            old,
            new,
            expect,
            regex,
            dry_run,
            verify,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let new_bytes = edit_positional_payload(new, "NEW")?;
                edit_expect_positive(expect)?;
                let spec = WhereSpec {
                    file: Some(file),
                    old: (!regex).then_some(old.clone()),
                    old_file: None,
                    pattern: regex.then_some(old),
                    lines: None,
                    symbol: None,
                    body: false,
                    target: None,
                    path: None,
                };
                let kind = if regex {
                    SelectorKind::Pattern
                } else {
                    SelectorKind::Text
                };
                let located = edit_locate(&spec, kind, root, root_path, file_base)?;
                edit_check_cardinality(&located, expect)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes)?;
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::ReplaceLines {
            file,
            lines,
            new,
            dry_run,
            verify,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let new_bytes = edit_positional_payload(new, "NEW")?;
                let spec = WhereSpec {
                    file: Some(file),
                    old: None,
                    old_file: None,
                    pattern: None,
                    lines: Some(lines),
                    symbol: None,
                    body: false,
                    target: None,
                    path: None,
                };
                let located = edit_locate(&spec, SelectorKind::Lines, root, root_path, file_base)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes)?;
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::ReplaceSpan {
            handle,
            new,
            dry_run,
            verify,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let new_bytes = edit_positional_payload(new, "NEW")?;
                let spec = WhereSpec {
                    file: None,
                    old: None,
                    old_file: None,
                    pattern: None,
                    lines: None,
                    symbol: None,
                    body: false,
                    target: Some(handle),
                    path: None,
                };
                let located = edit_locate(&spec, SelectorKind::Target, root, root_path, file_base)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes)?;
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::Write {
            path,
            new,
            dry_run,
            verify,
        } => {
            let outcome = edit_positional_payload(new, "NEW").and_then(|bytes| {
                run_trained_write(root_path, file_base, &path, bytes, dry_run, verify)
            });
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::Delete {
            symbol,
            dry_run,
            verify,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let spec = WhereSpec {
                    file: None,
                    old: None,
                    old_file: None,
                    pattern: None,
                    lines: None,
                    symbol: Some(symbol),
                    body: false,
                    target: None,
                    path: None,
                };
                let located = edit_locate(&spec, SelectorKind::Symbol, root, root_path, file_base)?;
                let (new_content, changed) = edit_op_delete(&located);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::DeleteLines {
            file,
            lines,
            dry_run,
            verify,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let spec = WhereSpec {
                    file: Some(file),
                    old: None,
                    old_file: None,
                    pattern: None,
                    lines: Some(lines),
                    symbol: None,
                    body: false,
                    target: None,
                    path: None,
                };
                let located = edit_locate(&spec, SelectorKind::Lines, root, root_path, file_base)?;
                let (new_content, changed) = edit_op_delete(&located);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::InsertLines {
            file,
            line,
            new,
            dry_run,
            verify,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let mut inserted = edit_positional_payload(new, "NEW")?;
                let (rel, abs, content) = edit_read_file(root_path, file_base, &file)?;
                let total = edit_line_count(&content);
                if line > total {
                    return Err(EditRefusal::new(
                        "range_out_of_bounds",
                        format!("{rel} has {total} line(s); cannot insert after line {line}"),
                        13,
                    ));
                }
                let ending: &[u8] = if content.windows(2).any(|pair| pair == b"\r\n") {
                    b"\r\n"
                } else {
                    b"\n"
                };
                if !inserted.ends_with(b"\n") {
                    inserted.extend_from_slice(ending);
                }
                let at = if line == 0 {
                    0
                } else {
                    line_range_to_bytes(&content, line, line).1
                };
                if line > 0 && at == content.len() && !content.ends_with(b"\n") {
                    let mut separated = ending.to_vec();
                    separated.extend_from_slice(&inserted);
                    inserted = separated;
                }
                let located = Located {
                    rel,
                    abs,
                    content,
                    ranges: vec![(at, at)],
                    kind: SelectorKind::Lines,
                    regex: None,
                    needle: None,
                };
                let mut edits = vec![(at, at, inserted)];
                let (new_content, changed) = edit_splice(&located.content, &mut edits);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::Rename {
            symbol,
            name,
            dry_run,
            verify,
        } => {
            let outcome = run_trained_rename(root_path, root, &symbol, &name, dry_run, verify)?;
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::Undo {
            id,
            dry_run,
            verify,
        } => {
            let outcome = run_edit_undo(root_path, id.as_deref(), dry_run, verify);
            emit_edit_outcome(outcome, json, None, root_path)?
        }
        EditCommand::Patch {
            diff,
            dry_run,
            verify,
        } => {
            let outcome = edit_positional_payload(diff, "DIFF")
                .and_then(|bytes| run_trained_patch(root_path, file_base, bytes, dry_run, verify));
            emit_edit_outcome(outcome, json, None, root_path)?
        }
    };
    Ok(GrammarDispatch(code))
}

pub(crate) fn edit_status_name(status: greppy_edit::Status) -> &'static str {
    match status {
        greppy_edit::Status::Applied => "applied",
        greppy_edit::Status::AlreadySatisfied => "already-satisfied",
        greppy_edit::Status::NotFound => "not-found",
        greppy_edit::Status::Ambiguous => "ambiguous",
        greppy_edit::Status::Stale => "stale",
        greppy_edit::Status::InvalidResult => "invalid-result",
        greppy_edit::Status::ValidationFailed => "validation-failed",
        greppy_edit::Status::PublishFailed => "publish-failed",
    }
}

pub(crate) fn edit_operation_path(
    operation: &greppy_edit::certificate::OperationReport,
    root_path: &std::path::Path,
) -> String {
    let path = std::path::Path::new(&operation.file);
    let relative = if path.is_absolute() {
        path.strip_prefix(root_path).unwrap_or(path)
    } else {
        path
    };
    relative.to_string_lossy().replace('\\', "/")
}

pub(crate) fn edit_operation_line_span(
    operation: &greppy_edit::certificate::OperationReport,
    root_path: &std::path::Path,
) -> (usize, usize) {
    if let Some(span) = operation
        .unified_diff
        .as_deref()
        .and_then(diff_after_line_span)
    {
        return span;
    }
    let path = if std::path::Path::new(&operation.file).is_absolute() {
        std::path::PathBuf::from(&operation.file)
    } else {
        root_path.join(&operation.file)
    };
    let content = std::fs::read(path).unwrap_or_default();
    if let Some(start_byte) = operation
        .changed_byte_ranges
        .iter()
        .map(|range| range.0)
        .min()
    {
        let start = line_for_byte(&content, start_byte);
        let end = operation.node_after.as_deref().map_or_else(
            || {
                operation
                    .changed_byte_ranges
                    .iter()
                    .map(|range| line_for_byte(&content, range.1))
                    .max()
                    .unwrap_or(start)
            },
            |span| start.saturating_add(span.lines().count().max(1) - 1),
        );
        return (start, end.max(start));
    }
    let line_count = content.iter().filter(|byte| **byte == b'\n').count()
        + usize::from(!content.is_empty() && !content.ends_with(b"\n"));
    (1, line_count.max(1))
}

#[cfg(test)]
mod patch_rollback_tests {
    use super::*;

    #[test]
    fn rename_edge_requires_selected_method_identity_evidence() {
        let selected = serde_json::json!({
            "callee_name": "next",
            "callee_form": "receiver",
            "receiver_owner": "Scheduler"
        });
        let unrelated = serde_json::json!({
            "callee_name": "next",
            "callee_form": "receiver",
            "receiver_owner": "Iterator"
        });
        let misleading = serde_json::json!({
            "callee_name": "next",
            "callee_form": "receiver"
        });
        let empty_receiver = serde_json::json!({
            "callee_name": "next",
            "callee_form": "receiver",
            "receiver_owner": ""
        });
        let associated = serde_json::json!({
            "callee_name": "next",
            "callee_form": "direct",
            "callee_path": "Scheduler::next"
        });
        let unqualified_path = serde_json::json!({
            "callee_name": "next",
            "callee_form": "direct",
            "callee_path": "next"
        });
        let null_path = serde_json::json!({
            "callee_name": "next",
            "callee_form": "direct",
            "callee_path": null
        });
        let empty_path = serde_json::json!({
            "callee_name": "next",
            "callee_form": "direct",
            "callee_path": ""
        });
        let non_string_path = serde_json::json!({
            "callee_name": "next",
            "callee_form": "direct",
            "callee_path": 7
        });
        let qualified_same_tail = serde_json::json!({
            "callee_name": "next",
            "callee_form": "direct",
            "callee_path": "a::Scheduler::next"
        });

        assert_eq!(
            rename_edge_identity("CALLS", &selected, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Related
        );
        assert_eq!(
            rename_edge_identity("CALLS", &associated, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Related
        );
        assert_eq!(
            rename_edge_identity("CALLS", &unrelated, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unrelated
        );
        assert_eq!(
            rename_edge_identity("CALLS", &misleading, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unknown
        );
        assert_eq!(
            rename_edge_identity("CALLS", &empty_receiver, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unknown
        );
        assert_eq!(
            rename_edge_identity("CALLS", &unqualified_path, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unknown
        );
        assert_eq!(
            rename_edge_identity("CALLS", &null_path, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unknown
        );
        assert_eq!(
            rename_edge_identity("CALLS", &empty_path, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unknown
        );
        assert_eq!(
            rename_edge_identity("CALLS", &non_string_path, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unknown
        );
        assert_eq!(
            rename_edge_identity("CALLS", &qualified_same_tail, Some("Scheduler"), "next"),
            RenameEdgeIdentity::Unknown
        );
        let refusal = require_rename_edge_identity(
            "CALLS",
            &misleading,
            Some("Scheduler"),
            "next",
            "Scheduler::next",
            42,
        )
        .unwrap_err();
        assert_eq!(refusal.code, "unresolved_reference_identity");
        assert!(refusal.message.contains("source node 42"));

        let dir = tempfile::tempdir().unwrap();
        let selected_path = dir.path().join("selected.rs");
        let caller_path = dir.path().join("caller.rs");
        std::fs::write(&selected_path, b"impl Scheduler { fn next(&self) {} }\n").unwrap();
        std::fs::write(&caller_path, b"fn call(s: &Scheduler) { s.next(); }\n").unwrap();
        let before_selected = std::fs::read(&selected_path).unwrap();
        let before_caller = std::fs::read(&caller_path).unwrap();
        assert!(require_rename_edge_identity(
            "CALLS",
            &misleading,
            Some("Scheduler"),
            "next",
            "Scheduler::next",
            42,
        )
        .is_err());
        assert_eq!(std::fs::read(&selected_path).unwrap(), before_selected);
        assert_eq!(std::fs::read(&caller_path).unwrap(), before_caller);

        assert_eq!(
            select_rename_reference_site("Scheduler::next", "next", "stale.rs", &[])
                .unwrap_or_else(|refusal| panic!("{}", refusal.message)),
            None
        );
        assert_eq!(
            select_rename_reference_site("Scheduler::next", "next", "caller.rs", &[(30, 34)])
                .unwrap_or_else(|refusal| panic!("{}", refusal.message)),
            Some((30, 34))
        );
        assert!(select_rename_reference_site(
            "Scheduler::next",
            "next",
            "ambiguous.rs",
            &[(10, 14), (30, 34)]
        )
        .is_err());
    }

    #[test]
    fn trailing_empty_hunk_reports_exact_header_and_recovery() {
        let diff = b"--- a/patch-repro.txt\n+++ b/patch-repro.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n@@\n";
        let refusal = match parse_trained_patch(diff) {
            Err(refusal) => refusal,
            Ok(_) => panic!("empty hunk must be refused"),
        };
        assert_eq!(refusal.code, "invalid_patch");
        assert!(refusal
            .message
            .contains("hunk 2 at patch input line 7 is empty"));
        assert!(refusal.message.contains("remove its @@ header"));
        assert!(refusal.message.contains("nothing written"));
        let recovered = &diff[..diff.len() - 3];
        let files =
            parse_trained_patch(recovered).unwrap_or_else(|refusal| panic!("{}", refusal.message));
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].hunks.len(), 1);
    }

    #[test]
    fn qualified_rust_free_function_rename_preserves_distinct_copy() {
        let dir = tempfile::tempdir().unwrap();
        let selected =
            b"fn get_lit_str() {}\nfn selected_caller() { get_lit_str(); get_lit_str(); }\n";
        // A module-level wildcard cannot shadow an explicit local function.
        // Serde's two independent attr.rs copies both import symbol::*.
        let unrelated = b"mod symbols { pub const TAG: u8 = 0; }\nuse symbols::*;\nfn get_lit_str() {}\nfn unrelated_caller() { let Some(value) = Some(get_lit_str()) else { return; }; let _ = value; }\n";
        std::fs::write(dir.path().join("selected.rs"), selected).unwrap();
        std::fs::write(dir.path().join("unrelated.rs"), unrelated).unwrap();
        let sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("selected.rs"),
            selected,
            &[(0, selected.len())],
            "get_lit_str",
        )
        .unwrap();
        let scopes = std::collections::BTreeMap::from([("selected.rs".to_string(), sites.clone())]);
        rust_free_function_reference_inventory(
            dir.path(),
            &scopes,
            &std::collections::BTreeSet::from(["selected.rs".to_string()]),
            "get_lit_str",
            "selected.rs::get_lit_str",
        )
        .unwrap_or_else(|refusal| panic!("{}", refusal.message));
        let certificate = greppy_edit::verbs::rename_symbol_files_scoped(
            dir.path(),
            &[greppy_edit::verbs::RenameFileScope {
                rel_path: "selected.rs".into(),
                spans: sites,
            }],
            "get_lit_str",
            "get_str_literal",
            &greppy_edit::verbs::VerbOptions {
                expect_residual: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(certificate.status, greppy_edit::Status::Applied);
        let changed = std::fs::read(dir.path().join("selected.rs")).unwrap();
        assert_eq!(
            changed,
            b"fn get_str_literal() {}\nfn selected_caller() { get_str_literal(); get_str_literal(); }\n"
        );
        assert_eq!(
            std::fs::read(dir.path().join("unrelated.rs")).unwrap(),
            unrelated
        );
        assert_eq!(
            greppy_edit::txn::syntax_counts(greppy_parser::Language::Rust, &changed),
            Some(greppy_edit::txn::SyntaxCounts {
                errors: 0,
                missing: 0
            })
        );
    }

    #[test]
    fn qualified_rust_free_function_unknown_caller_refuses_before_publish() {
        let dir = tempfile::tempdir().unwrap();
        let selected = b"fn get_lit_str() {}\n";
        let unknown = b"mod selected;\nmod other { fn get_lit_str() {} }\nuse crate::selected::get_lit_str;\nfn caller() { get_lit_str(); }\n";
        std::fs::write(dir.path().join("selected.rs"), selected).unwrap();
        std::fs::write(dir.path().join("lib.rs"), unknown).unwrap();
        let sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("selected.rs"),
            selected,
            &[(0, selected.len())],
            "get_lit_str",
        )
        .unwrap();
        let scopes = std::collections::BTreeMap::from([("selected.rs".to_string(), sites)]);
        let refusal = rust_free_function_reference_inventory(
            dir.path(),
            &scopes,
            &std::collections::BTreeSet::from(["selected.rs".to_string()]),
            "get_lit_str",
            "selected.rs::get_lit_str",
        )
        .unwrap_err();
        assert_eq!(refusal.code, "unresolved_reference_identity");
        assert!(refusal.message.contains("lib.rs"));
        assert_eq!(
            std::fs::read(dir.path().join("selected.rs")).unwrap(),
            selected
        );
        assert_eq!(std::fs::read(dir.path().join("lib.rs")).unwrap(), unknown);
    }

    #[test]
    fn rust_free_function_local_proof_is_lexically_scoped() {
        let source = b"fn get_lit_str() {}\nfn top() { get_lit_str(); }\nmod left { fn get_lit_str() {} fn local() { get_lit_str(); } }\nmod right { fn caller() { get_lit_str(); } }\n";
        let sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("scope.rs"),
            source,
            &[(0, source.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(sites.len(), 5);
        assert!(rust_local_free_function_owns_site(
            source,
            "get_lit_str",
            sites[1],
            true
        ));
        assert!(rust_local_free_function_owns_site(
            source,
            "get_lit_str",
            sites[3],
            true
        ));
        assert!(!rust_local_free_function_owns_site(
            source,
            "get_lit_str",
            sites[4],
            true
        ));
    }

    #[test]
    fn rust_local_binding_proof_ignores_initializer_references_and_parameter_types() {
        let cases: &[(&[u8], bool)] = &[
            (b"fn get_lit_str() {}\nfn caller() { let value = get_lit_str(); }\n", true),
            (b"fn get_lit_str() {}\nfn caller() { let Some(value) = Some(get_lit_str()) else { return; }; }\n", true),
            (b"fn get_lit_str() {}\nfn caller() { let get_lit_str = || {}; get_lit_str(); }\n", false),
            (b"type get_lit_str = ();\nfn get_lit_str() {}\nfn caller(arg: get_lit_str) { get_lit_str(); }\n", true),
        ];
        for (source, expected) in cases {
            let sites = greppy_edit::verbs::rename_identifier_sites(
                std::path::Path::new("bindings.rs"),
                source,
                &[(0, source.len())],
                "get_lit_str",
            )
            .unwrap();
            assert_eq!(
                rust_local_free_function_owns_site(
                    source,
                    "get_lit_str",
                    *sites.last().unwrap(),
                    false
                ),
                *expected,
                "{}",
                String::from_utf8_lossy(source),
            );
        }
    }

    #[test]
    fn selected_local_free_function_proof_rejects_nested_identity_changes() {
        let block_shadow = b"fn get_lit_str() {}\nfn caller() { get_lit_str(); { fn get_lit_str() {} get_lit_str(); } }\n";
        let block_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("block.rs"),
            block_shadow,
            &[(0, block_shadow.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(block_sites.len(), 4);
        assert!(rust_selected_local_free_function_owns_site(
            block_shadow,
            "get_lit_str",
            block_sites[0],
            block_sites[1],
        ));
        assert!(!rust_selected_local_free_function_owns_site(
            block_shadow,
            "get_lit_str",
            block_sites[0],
            block_sites[3],
        ));

        let nested_module = b"fn get_lit_str() {}\nmod other { fn get_lit_str() {} fn caller() { get_lit_str(); } }\n";
        let module_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("module.rs"),
            nested_module,
            &[(0, nested_module.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(module_sites.len(), 3);
        assert!(!rust_selected_local_free_function_owns_site(
            nested_module,
            "get_lit_str",
            module_sites[0],
            module_sites[2],
        ));

        let block_glob = b"fn get_lit_str() {}\nfn caller() { use other::*; get_lit_str(); }\n";
        let glob_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("block-glob.rs"),
            block_glob,
            &[(0, block_glob.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(glob_sites.len(), 2);
        assert!(!rust_selected_local_free_function_owns_site(
            block_glob,
            "get_lit_str",
            glob_sites[0],
            glob_sites[1],
        ));

        let mixed_form =
            b"fn get_lit_str() {}\nfn caller() { get_lit_str(); other::get_lit_str(); }\n";
        let mixed_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("mixed.rs"),
            mixed_form,
            &[(0, mixed_form.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(mixed_sites.len(), 3);
        assert!(rust_selected_local_free_function_owns_site(
            mixed_form,
            "get_lit_str",
            mixed_sites[0],
            mixed_sites[1],
        ));
        assert!(!rust_selected_local_free_function_owns_site(
            mixed_form,
            "get_lit_str",
            mixed_sites[0],
            mixed_sites[2],
        ));

        let closure_shadow = b"fn get_lit_str() {}\nfn caller() { let invoke = |get_lit_str| get_lit_str(); invoke(|| {}); }\n";
        let closure_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("closure.rs"),
            closure_shadow,
            &[(0, closure_shadow.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(closure_sites.len(), 3);
        assert!(!rust_selected_local_free_function_owns_site(
            closure_shadow,
            "get_lit_str",
            closure_sites[0],
            closure_sites[2],
        ));

        let match_shadow = b"fn get_lit_str() {}\nfn caller(value: Option<fn()>) { match value { Some(get_lit_str) => get_lit_str(), None => {} } }\n";
        let match_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("match.rs"),
            match_shadow,
            &[(0, match_shadow.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(match_sites.len(), 3);
        assert!(!rust_selected_local_free_function_owns_site(
            match_shadow,
            "get_lit_str",
            match_sites[0],
            match_sites[2],
        ));

        for (path, source) in [
            (
                "for.rs",
                b"fn get_lit_str() {}\nfn caller(items: Vec<fn()>) { for get_lit_str in items { get_lit_str(); } }\n".as_slice(),
            ),
            (
                "if-let.rs",
                b"fn get_lit_str() {}\nfn caller(value: Option<fn()>) { if let Some(get_lit_str) = value { get_lit_str(); } }\n".as_slice(),
            ),
            (
                "while-let.rs",
                b"fn get_lit_str() {}\nfn caller(mut value: Option<fn()>) { while let Some(get_lit_str) = value.take() { get_lit_str(); } }\n".as_slice(),
            ),
        ] {
            let sites = greppy_edit::verbs::rename_identifier_sites(
                std::path::Path::new(path),
                source,
                &[(0, source.len())],
                "get_lit_str",
            )
            .unwrap();
            assert_eq!(sites.len(), 3, "{path}");
            assert!(!rust_selected_local_free_function_owns_site(
                source,
                "get_lit_str",
                sites[0],
                sites[2],
            ));
        }

        let serde_shape = b"fn get_lit_str() {}\nfn caller() { if let Some(_) = get_lit_str() {} if let Some(_) = get_lit_str() {} if let Some(_) = get_lit_str() {} }\n";
        let serde_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("serde-shape.rs"),
            serde_shape,
            &[(0, serde_shape.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(serde_sites.len(), 4);
        for site in &serde_sites[1..] {
            assert!(rust_selected_local_free_function_owns_site(
                serde_shape,
                "get_lit_str",
                serde_sites[0],
                *site,
            ));
        }
    }

    #[test]
    fn associated_method_cannot_prove_free_function_ownership() {
        let source = b"struct Helper;\nimpl Helper { fn get_lit_str() {} }\nuse crate::selected::get_lit_str;\nfn caller() { get_lit_str(); }\n";
        let sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("method.rs"),
            source,
            &[(0, source.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(sites.len(), 3);
        assert!(!rust_local_free_function_owns_site(
            source,
            "get_lit_str",
            sites[2],
            true
        ));
    }

    #[test]
    fn associated_method_alone_does_not_own_free_function_call() {
        let source = b"struct Helper;\nimpl Helper { fn get_lit_str() {} }\nfn caller() { get_lit_str(); }\n";
        let sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("method-only.rs"),
            source,
            &[(0, source.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(sites.len(), 2);
        assert!(!rust_local_free_function_owns_site(
            source,
            "get_lit_str",
            sites[1],
            true
        ));
    }

    #[test]
    fn block_import_prevents_module_function_ownership_proof() {
        let source = b"fn get_lit_str() {}\nfn caller() { use crate::selected::get_lit_str; get_lit_str(); }\n";
        let sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("shadow.rs"),
            source,
            &[(0, source.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(sites.len(), 3);
        assert!(!rust_local_free_function_owns_site(
            source,
            "get_lit_str",
            sites[2],
            true
        ));
    }

    #[test]
    fn block_glob_import_prevents_module_function_ownership_proof() {
        let source =
            b"fn get_lit_str() {}\nfn caller() { use crate::selected::*; get_lit_str(); }\n";
        let sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("glob.rs"),
            source,
            &[(0, source.len())],
            "get_lit_str",
        )
        .unwrap();
        assert_eq!(sites.len(), 2);
        assert!(!rust_local_free_function_owns_site(
            source,
            "get_lit_str",
            sites[1],
            false
        ));
    }

    #[test]
    fn glob_without_old_name_needs_no_edit_but_live_glob_call_must_be_planned() {
        let dir = tempfile::tempdir().unwrap();
        let selected = b"pub fn get_lit_str() {}\n";
        std::fs::write(dir.path().join("selected.rs"), selected).unwrap();
        std::fs::write(
            dir.path().join("glob_only.rs"),
            b"pub use crate::selected::*;\n",
        )
        .unwrap();
        let selected_sites = greppy_edit::verbs::rename_identifier_sites(
            std::path::Path::new("selected.rs"),
            selected,
            &[(0, selected.len())],
            "get_lit_str",
        )
        .unwrap();
        let scopes =
            std::collections::BTreeMap::from([("selected.rs".to_string(), selected_sites)]);
        let selected_files = std::collections::BTreeSet::from(["selected.rs".to_string()]);
        rust_free_function_reference_inventory(
            dir.path(),
            &scopes,
            &selected_files,
            "get_lit_str",
            "selected.rs::get_lit_str",
        )
        .unwrap_or_else(|refusal| panic!("{}", refusal.message));

        std::fs::write(
            dir.path().join("glob_call.rs"),
            b"use crate::selected::*;\nfn caller() { get_lit_str(); }\n",
        )
        .unwrap();
        let refusal = rust_free_function_reference_inventory(
            dir.path(),
            &scopes,
            &selected_files,
            "get_lit_str",
            "selected.rs::get_lit_str",
        )
        .expect_err("live call omitted from graph plan must refuse");
        assert_eq!(refusal.code, "unresolved_reference_identity");
        assert!(refusal.message.contains("glob_call.rs"));
    }

    #[test]
    fn guarded_c_header_write_and_invalid_replacement_are_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let valid = include_bytes!("../../edit/tests/fixtures/guarded-protocol.h").to_vec();
        run_trained_write(
            dir.path(),
            dir.path(),
            "protocol.h",
            valid.clone(),
            false,
            false,
        )
        .unwrap_or_else(|error| panic!("{}", error.message));
        assert_eq!(std::fs::read(dir.path().join("protocol.h")).unwrap(), valid);
        let malformed = String::from_utf8(valid.clone()).unwrap().replacen(
            "fma_codec_name(uint32_t codec);",
            "fma_codec_name(uint32_t codec;",
            1,
        );
        let refusal = match run_trained_write(
            dir.path(),
            dir.path(),
            "protocol.h",
            malformed.into_bytes(),
            false,
            false,
        ) {
            Err(error) => error,
            Ok(_) => panic!("malformed header accepted"),
        };
        assert_eq!(refusal.code, "invalid_result");
        assert_eq!(std::fs::read(dir.path().join("protocol.h")).unwrap(), valid);
    }

    #[test]
    fn duplicate_patch_targets_are_refused_before_any_publish() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("example.txt");
        std::fs::write(&path, b"one\nkeep\ntwo\n").unwrap();
        for second in ["example.txt", "./example.txt"] {
            for dry_run in [false, true] {
                let diff = format!(
                    "--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-one\n+ONE\n--- a/{second}\n+++ b/{second}\n@@ -3 +3 @@\n-two\n+TWO\n"
                );
                let result = run_trained_patch_with_publish_hook(
                    dir.path(),
                    dir.path(),
                    diff.into_bytes(),
                    dry_run,
                    false,
                    |_| panic!("duplicate target must be rejected during planning"),
                );
                let refusal = match result {
                    Err(refusal) => refusal,
                    Ok(_) => panic!("duplicate target was accepted"),
                };
                assert_eq!(refusal.code, "invalid_patch");
                assert_eq!(refusal.exit, 20);
                assert!(refusal.message.contains("duplicate patch target"));
                assert!(refusal.message.contains("one ---/+++ header pair"));
                assert!(!refusal.message.contains("stale plan"));
                assert_eq!(std::fs::read(&path).unwrap(), b"one\nkeep\ntwo\n");
            }
        }
        let grouped = parse_trained_patch(
            b"--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-one\n+ONE\n@@ -3 +3 @@\n-two\n+TWO\n",
        ).unwrap_or_else(|refusal| panic!("{}", refusal.message));
        assert_eq!(grouped.len(), 1);
        let (after, _) =
            apply_trained_patch_file("example.txt", b"one\nkeep\ntwo\n", &grouped[0].hunks)
                .unwrap_or_else(|refusal| panic!("{}", refusal.message));
        assert_eq!(after, b"ONE\nkeep\nTWO\n");
    }

    #[test]
    fn failed_patch_never_rolls_back_the_unpublished_conflict_target() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.txt");
        let last = dir.path().join("last.txt");
        std::fs::write(&first, b"before\n").unwrap();
        std::fs::write(&last, b"original\n").unwrap();
        let diff = b"--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-before\n+after\n--- a/last.txt\n+++ b/last.txt\n@@ -1 +1 @@\n-original\n+patched\n";
        let result = run_trained_patch_with_publish_hook(
            dir.path(),
            dir.path(),
            diff.to_vec(),
            false,
            false,
            |index| {
                if index == 1 {
                    std::fs::write(&last, b"concurrent-success\n").unwrap();
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(std::fs::read(&first).unwrap(), b"before\n");
        assert_eq!(std::fs::read(&last).unwrap(), b"concurrent-success\n");
        assert!(!edit_journal_dir(dir.path())
            .join(EDIT_JOURNAL_PENDING)
            .exists());
        // This test alone created the journal under its unique temporary root hash.
        std::fs::remove_dir_all(edit_journal_dir(dir.path())).unwrap();
    }

    #[test]
    fn transaction_lock_excludes_another_writer_and_releases_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let first = acquire_edit_transaction_lock(dir.path()).unwrap();
        assert!(matches!(
            acquire_edit_transaction_lock(dir.path()),
            Err(Error::Lock(_))
        ));
        drop(first);
        assert!(acquire_edit_transaction_lock(dir.path()).is_ok());
    }

    #[test]
    fn rollback_restores_our_published_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, b"our patch").unwrap();
        rollback_patch_file(dir.path(), &path, b"before", b"our patch").unwrap();
        assert_eq!(std::fs::read(path).unwrap(), b"before");
    }

    #[test]
    fn rollback_preserves_a_later_writers_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, b"someone else").unwrap();
        assert!(rollback_patch_file(dir.path(), &path, b"before", b"our patch").is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"someone else");
    }

    #[test]
    fn rollback_does_not_recreate_a_concurrently_deleted_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        assert!(rollback_patch_file(dir.path(), &path, b"before", b"our patch").is_err());
        assert!(!path.exists());
    }
}
