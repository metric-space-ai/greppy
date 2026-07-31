//! The edit grammar and the verbs that carry it out.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

pub(crate) fn dispatch_edit(command: EditCommand, json: bool, root: Option<&str>) -> Result<i32> {
    match dispatch_edit_inner(command, json, root) {
        Err(error @ Error::Invalid(_)) => {
            eprintln!("greppy: {error}");
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
    Ok(dispatch_edit_grammar(command, json, root, &root_path)?.0)
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
    file: &str,
) -> EditResult<(String, std::path::PathBuf, Vec<u8>)> {
    let candidate = std::path::Path::new(file);
    let abs = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root_path.join(candidate)
    };
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
    let rel = abs
        .strip_prefix(root_path)
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
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
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
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
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
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
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
    let language = greppy_edit::language_for_path(std::path::Path::new(&located.rel));
    if language.is_supported() {
        if let (Some(before), Some(after)) = (
            greppy_edit::txn::syntax_counts(language, &located.content),
            greppy_edit::txn::syntax_counts(language, &new_content),
        ) {
            if after.errors > before.errors || after.missing > before.missing {
                return Err(EditRefusal::new(
                    "invalid_result",
                    "refused: the edit would break the file's syntax — nothing written",
                    13,
                ));
            }
        }
    }
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

pub(crate) fn edit_op_replace(located: &Located, new_bytes: &[u8]) -> EditedContent {
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
    edit_splice(&located.content, &mut edits)
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

/// The compiler or linter for the touched files, when the workspace declares
/// one. No tests: a test run is too long for a single edit call.
pub(crate) fn edit_verify_diagnostics(
    root_path: &std::path::Path,
    files: &[String],
) -> Vec<String> {
    let mut argv: Option<Vec<&str>> = None;
    if root_path.join("Cargo.toml").is_file() {
        argv = Some(vec![
            "cargo",
            "check",
            "--message-format",
            "short",
            "--quiet",
        ]);
    } else if root_path.join("go.mod").is_file() {
        argv = Some(vec!["go", "build", "./..."]);
    } else if files.iter().any(|file| file.ends_with(".py"))
        && root_path.join("pyproject.toml").is_file()
    {
        argv = Some(vec!["python3", "-m", "compileall", "-q", "."]);
    }
    let Some(argv) = argv else {
        return Vec::new();
    };
    let Ok(output) = std::process::Command::new(argv[0])
        .args(&argv[1..])
        .current_dir(root_path)
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .filter(|line| line.contains("error") || line.contains("warning"))
        .take(50)
        .map(str::to_string)
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
    let dir = edit_journal_dir(root_path);
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
    file: &str,
) -> EditResult<(String, std::path::PathBuf)> {
    let base = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.to_path_buf());
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
                        format!("{file} is outside {}", base.display()),
                        17,
                    ));
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    let Ok(relative) = normalized.strip_prefix(&base) else {
        return Err(EditRefusal::new(
            "path_outside_repo",
            format!("{file} is outside {}", base.display()),
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
    path: &str,
    bytes: Vec<u8>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let (rel, abs) = edit_resolve_new_path(root_path, path)?;
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
                format!("{path} is outside {}", root.display()),
                17,
            ));
        }
    }
    let old = before.as_deref().unwrap_or_default();
    let language = greppy_edit::language_for_path(std::path::Path::new(&rel));
    if language.is_supported() {
        if let (Some(before_counts), Some(after_counts)) = (
            greppy_edit::txn::syntax_counts(language, old),
            greppy_edit::txn::syntax_counts(language, &bytes),
        ) {
            if after_counts.errors > before_counts.errors
                || after_counts.missing > before_counts.missing
            {
                return Err(EditRefusal::new(
                    "invalid_result",
                    "refused: the edit would break the file's syntax — nothing written",
                    13,
                ));
            }
        }
    }
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
    while index < lines.len() {
        if !lines[index].starts_with("--- ") {
            index += 1;
            continue;
        }
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
                "file creation and deletion are not supported by patch",
                20,
            ));
        };
        index += 1;
        let mut hunks = Vec::new();
        while index < lines.len() && !lines[index].starts_with("--- ") {
            if !lines[index].starts_with("@@") {
                index += 1;
                continue;
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
                && !lines[index].starts_with("--- ")
            {
                let line = lines[index];
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
                            "a hunk contains a line without a unified-diff prefix",
                            20,
                        ))
                    }
                }
                index += 1;
            }
            if old_lines.is_empty() {
                return Err(EditRefusal::new(
                    "invalid_patch",
                    format!("{path}: a hunk has no context line to anchor on"),
                    20,
                ));
            }
            hunks.push(TrainedPatchHunk {
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
                    return Err(EditRefusal::new(
                        "patch_context",
                        format!("{path}: hunk context matches more than once — nothing written"),
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

pub(crate) fn run_trained_patch(
    root_path: &std::path::Path,
    diff: Vec<u8>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let parsed = parse_trained_patch(&diff)?;
    let mut planned = Vec::new();
    for file in parsed {
        let (rel, abs, content) = edit_read_file(root_path, &file.path)?;
        let (after, changed) = apply_trained_patch_file(&rel, &content, &file.hunks)?;
        let language = greppy_edit::language_for_path(std::path::Path::new(&rel));
        if language.is_supported() {
            if let (Some(before_counts), Some(after_counts)) = (
                greppy_edit::txn::syntax_counts(language, &content),
                greppy_edit::txn::syntax_counts(language, &after),
            ) {
                if after_counts.errors > before_counts.errors
                    || after_counts.missing > before_counts.missing
                {
                    return Err(EditRefusal::new(
                        "invalid_result",
                        "refused: the edit would break the file's syntax — nothing written",
                        13,
                    ));
                }
            }
        }
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
    for (_, abs, content, after, _) in &planned {
        if let Err(error) =
            greppy_edit::publish::publish_atomic(root_path, abs, after, &edit_sha256_hex(content))
        {
            if let Some(pending) =
                edit_journal_read(&edit_journal_dir(root_path).join(EDIT_JOURNAL_PENDING))
            {
                let _ = edit_journal_restore(root_path, &pending, false);
            }
            edit_journal_abort(root_path);
            return Err(EditRefusal::new(
                "publish_failed",
                format!("patch transaction failed: {error} — nothing written"),
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
    for def in &def_nodes {
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
    let certificate = greppy_edit::verbs::rename_symbol_files(
        root_path,
        &scope_vec,
        &short_name,
        new_name,
        &options,
    )?;
    if certificate.exit_code() != 0 {
        let message = if certificate.status == greppy_edit::Status::InvalidResult {
            "refused: the edit would break the file's syntax — nothing written".to_string()
        } else {
            certificate.compact_failure_diagnosis().unwrap_or_else(|| {
                format!(
                    "rename {} — nothing written",
                    edit_status_name(certificate.status)
                )
            })
        };
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
    let mut record = EditRecord {
        files,
        span,
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

pub(crate) fn dispatch_edit_grammar(
    command: EditCommand,
    json: bool,
    root: Option<&str>,
    root_path: &std::path::Path,
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
                let new_bytes = edit_positional_payload(new, "NEW")?;
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
                let located = edit_locate(&spec, SelectorKind::Symbol, root, root_path)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None)?
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
                let located = edit_locate(&spec, kind, root, root_path)?;
                edit_check_cardinality(&located, expect)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None)?
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
                let located = edit_locate(&spec, SelectorKind::Lines, root, root_path)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None)?
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
                let located = edit_locate(&spec, SelectorKind::Target, root, root_path)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None)?
        }
        EditCommand::Write {
            path,
            new,
            dry_run,
            verify,
        } => {
            let outcome = edit_positional_payload(new, "NEW")
                .and_then(|bytes| run_trained_write(root_path, &path, bytes, dry_run, verify));
            emit_edit_outcome(outcome, json, None)?
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
                let located = edit_locate(&spec, SelectorKind::Symbol, root, root_path)?;
                let (new_content, changed) = edit_op_delete(&located);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None)?
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
                let located = edit_locate(&spec, SelectorKind::Lines, root, root_path)?;
                let (new_content, changed) = edit_op_delete(&located);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, None)?
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
                let (rel, abs, content) = edit_read_file(root_path, &file)?;
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
            emit_edit_outcome(outcome, json, None)?
        }
        EditCommand::Rename {
            symbol,
            name,
            dry_run,
            verify,
        } => {
            let outcome = run_trained_rename(root_path, root, &symbol, &name, dry_run, verify)?;
            emit_edit_outcome(outcome, json, None)?
        }
        EditCommand::Undo {
            id,
            dry_run,
            verify,
        } => {
            let outcome = run_edit_undo(root_path, id.as_deref(), dry_run, verify);
            emit_edit_outcome(outcome, json, None)?
        }
        EditCommand::Patch {
            diff,
            dry_run,
            verify,
        } => {
            let outcome = edit_positional_payload(diff, "DIFF")
                .and_then(|bytes| run_trained_patch(root_path, bytes, dry_run, verify));
            emit_edit_outcome(outcome, json, None)?
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
