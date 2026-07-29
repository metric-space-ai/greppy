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
    match dispatch_edit_grammar(command, json, root, &root_path)? {
        GrammarDispatch::Handled(code) => Ok(code),
        GrammarDispatch::Passthrough(_) => unreachable!("every trained edit verb is handled"),
    }
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

/// Refuse a path that leaves the repository, whether by climbing out of it or
/// through a symlink that points out.
pub(crate) fn edit_guard_path(root_path: &std::path::Path, abs: &std::path::Path) -> EditResult<()> {
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
        EditRefusal::new("symbol_not_found", format!("no symbol `{name}`: {error}"), 10)
    })?;
    let ids = resolve_symbol_nodes(&store, Some(name)).map_err(|error| {
        EditRefusal::new("symbol_not_found", format!("no symbol `{name}`: {error}"), 10)
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
            .map(|node| format!("  {} {}:{}", node.qualified_name, node.file_path, node.start_line))
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
            return Err(EditRefusal::new(
                "no_body",
                format!("`{name}` has no body"),
                13,
            )
            .with("symbol", serde_json::json!(name)));
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
                EditRefusal::new("invalid_handle", format!("not a usable handle: {error}"), 20)
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
                SelectorKind::WholeFile => {
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
                    let length = content.len();
                    (rel, abs, content, vec![(0, length)], None, None)
                }
                SelectorKind::Lines => {
                    let (first, last) = edit_parse_line_range(
                        spec.lines.as_deref().unwrap_or_default(),
                    )?;
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
                    (rel, abs, content, ranges, Some(regex), Some(pattern.to_string()))
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
        let _subject = located
            .needle
            .as_deref()
            .map_or_else(|| located.kind.name().to_string(), |text| format!("`{text}`"));
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
pub(crate) fn edit_guard_single_stdin(flags: &[(&'static str, Option<&str>)]) -> EditResult<()> {
    let asking: Vec<&'static str> = flags
        .iter()
        .filter(|(_, value)| *value == Some("-"))
        .map(|(name, _)| *name)
        .collect();
    if asking.len() > 1 {
        return Err(EditRefusal::new(
            "stdin_conflict",
            format!(
                "{} both read stdin; exactly one argument may take `-`",
                asking.join(" and ")
            ),
            20,
        )
        .with("flags", serde_json::json!(asking)));
    }
    Ok(())
}

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
        EditRefusal::new("content_unreadable", format!("read {name} from stdin: {error}"), 20)
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

pub(crate) fn edit_content_bytes(
    content: Option<String>,
    content_file: Option<String>,
) -> EditResult<Vec<u8>> {
    match (content, content_file) {
        (Some(text), None) => {
            if text.is_empty() {
                return Err(EditRefusal::new(
                    "content_empty",
                    format!("--content is empty; {EMPTY_CONTENT_HINT}"),
                    20,
                ));
            }
            Ok(text.into_bytes())
        }
        (None, Some(path)) => {
            let bytes = read_source_arg(&path).map_err(|error| {
                EditRefusal::new(
                    "content_unreadable",
                    format!("--content-file {path}: {error}"),
                    20,
                )
            })?;
            if bytes.is_empty() {
                let message = if path == "-" {
                    format!("--content-file -: stdin was empty; {EMPTY_CONTENT_HINT}")
                } else {
                    format!("--content-file {path} is empty; {EMPTY_CONTENT_HINT}")
                };
                return Err(EditRefusal::new("content_empty", message, 20));
            }
            Ok(bytes)
        }
        (Some(_), Some(_)) => Err(EditRefusal::new(
            "content_conflict",
            "--content and --content-file both name the new text; pass exactly one of them",
            20,
        )),
        (None, None) => Err(EditRefusal::new(
            "content_missing",
            "no new text: pass --content TEXT or --content-file FILE",
            20,
        )),
    }
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
    let mut operation = EditOperation {
        file: located.rel.clone(),
        ranges: changed,
        result_span: Some(text.clone()),
        sha_before: Some(edit_sha256_hex(&located.content)),
        sha_after: Some(edit_sha256_hex(&new_content)),
        diff: Some(edit_unified_diff(&located.rel, &located.content, &new_content)),
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
        record.already_as_sent = true;
        record.operations = vec![operation];
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

pub(crate) fn edit_op_insert(located: &Located, text: &[u8], before: bool) -> EditedContent {
    let (start, end) = located.ranges[0];
    let at = if before {
        start
    } else {
        edit_extend_over_newline(&located.content, (start, end)).1
    };
    let mut edits = vec![(at, at, text.to_vec())];
    edit_splice(&located.content, &mut edits)
}

/// Apply a unified diff inside the located span. The hunk headers are read
/// twice — once as file line numbers, once as offsets from the start of WHERE —
/// and whichever reading the context lines confirm is the one that applies.
pub(crate) fn edit_op_patch(located: &Located, patch: &[u8]) -> EditResult<EditedContent> {
    let patch_text = String::from_utf8_lossy(patch);
    let mut hunks: Vec<(usize, Vec<String>, Vec<String>)> = Vec::new();
    let mut current: Option<(usize, Vec<String>, Vec<String>)> = None;
    for line in patch_text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.starts_with("@@") {
            if let Some(hunk) = current.take() {
                hunks.push(hunk);
            }
            let start = line
                .split_whitespace()
                .nth(1)
                .and_then(|field| field.strip_prefix('-'))
                .map(|field| field.split(',').next().unwrap_or(field))
                .and_then(|number| number.parse::<usize>().ok())
                .unwrap_or(1);
            current = Some((start, Vec::new(), Vec::new()));
            continue;
        }
        let Some((_, old_lines, new_lines)) = current.as_mut() else {
            continue;
        };
        match line.as_bytes().first() {
            Some(b' ') => {
                old_lines.push(line[1..].to_string());
                new_lines.push(line[1..].to_string());
            }
            Some(b'-') => old_lines.push(line[1..].to_string()),
            Some(b'+') => new_lines.push(line[1..].to_string()),
            Some(b'\\') => {}
            None => {}
            _ => {}
        }
    }
    if let Some(hunk) = current.take() {
        hunks.push(hunk);
    }
    if hunks.is_empty() {
        return Err(EditRefusal::new(
            "invalid_patch",
            "the patch carries no hunk to apply",
            20,
        ));
    }

    // Physical lines of the file, each with the byte range it occupies.
    let mut lines: Vec<(usize, usize)> = Vec::new();
    let mut cursor = 0usize;
    while cursor < located.content.len() {
        let end = located.content[cursor..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|offset| cursor + offset + 1)
            .unwrap_or(located.content.len());
        lines.push((cursor, end));
        cursor = end;
    }
    let line_text = |index: usize| -> Option<String> {
        let (start, end) = *lines.get(index.checked_sub(1)?)?;
        let raw = &located.content[start..end];
        let raw = raw.strip_suffix(b"\n").unwrap_or(raw);
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        Some(String::from_utf8_lossy(raw).into_owned())
    };
    let (span_start, span_end) = located.ranges[0];
    let span_first_line = edit_line_of_offset(&located.content, span_start);
    let span_last_line = edit_line_of_offset(&located.content, span_end.saturating_sub(1).max(span_start));

    let mut edits: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    for (declared, old_lines, new_lines) in &hunks {
        let matches_at = |first: usize| -> bool {
            if first < span_first_line {
                return false;
            }
            if old_lines.is_empty() {
                return first <= span_last_line + 1;
            }
            if first + old_lines.len() - 1 > span_last_line {
                return false;
            }
            old_lines
                .iter()
                .enumerate()
                .all(|(offset, expected)| line_text(first + offset).as_ref() == Some(expected))
        };
        let relative = span_first_line + declared.saturating_sub(1);
        let first = if matches_at(*declared) {
            *declared
        } else if matches_at(relative) {
            relative
        } else {
            // Which hunk failed is not enough to act on: the caller needs the
            // context lines that were expected and not found, or it cannot tell
            // a wrong WHERE from a stale diff.
            let expected = old_lines
                .iter()
                .take(6)
                .map(|line| format!("\n  {line}"))
                .collect::<String>();
            return Err(EditRefusal::new(
                "patch_context",
                format!(
                    "the hunk at line {declared} matches neither {} nor the selected span; \
                     it expects:{expected}",
                    located.rel
                ),
                13,
            ));
        };
        let (start, end) = if old_lines.is_empty() {
            let at = lines
                .get(first.saturating_sub(1))
                .map(|(start, _)| *start)
                .unwrap_or(located.content.len());
            (at, at)
        } else {
            (
                lines[first - 1].0,
                lines[first + old_lines.len() - 2].1,
            )
        };
        let ending = if located.content[start..end].ends_with(b"\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        let mut replacement = String::new();
        for line in new_lines {
            replacement.push_str(line);
            replacement.push_str(ending);
        }
        // A hunk that reached the end of a file without a final newline must
        // not grow one.
        if !located.content[start..end].ends_with(b"\n") && replacement.ends_with(ending) {
            replacement.truncate(replacement.len() - ending.len());
        }
        edits.push((start, end, replacement.into_bytes()));
    }
    let first_start = edits.iter().map(|(start, _, _)| *start).min().unwrap_or(0);
    let last_end = edits.iter().map(|(_, end, _)| *end).max().unwrap_or(0);
    let delta: isize = edits
        .iter()
        .map(|(start, end, replacement)| replacement.len() as isize - (*end - *start) as isize)
        .sum();
    let (new_content, _) = edit_splice(&located.content, &mut edits);
    // A diff is one change even when it carries several hunks, so the span the
    // report names is the whole patched region, not each hunk on its own.
    let length = ((last_end - first_start) as isize + delta).max(0) as usize;
    Ok((new_content, vec![(first_start, first_start + length)]))
}

/// The compiler or linter for the touched files, when the workspace declares
/// one. No tests: a test run is too long for a single edit call.
pub(crate) fn edit_verify_diagnostics(root_path: &std::path::Path, files: &[String]) -> Vec<String> {
    let mut argv: Option<Vec<&str>> = None;
    if root_path.join("Cargo.toml").is_file() {
        argv = Some(vec!["cargo", "check", "--message-format", "short", "--quiet"]);
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
pub(crate) fn edit_journal_open(root_path: &std::path::Path, before: &[UndoBefore]) -> Option<String> {
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
                let bytes = std::fs::read(dir.join(EDIT_JOURNAL_BLOBS).join(blob)).map_err(
                    |error| {
                        EditRefusal::new(
                            "nothing_to_undo",
                            format!("{rel}: the pre-image is gone ({error})"),
                            10,
                        )
                    },
                )?;
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
pub(crate) fn run_edit_recover(root_path: &std::path::Path) -> EditResult<Option<usize>> {
    let dir = edit_journal_dir(root_path);
    let pending = dir.join(EDIT_JOURNAL_PENDING);
    let Some(record) = edit_journal_read(&pending) else {
        return Ok(None);
    };
    let restored = edit_journal_restore(root_path, &record, false)?;
    let _ = std::fs::remove_file(&pending);
    Ok(Some(restored.len()))
}

/// Which bytes of a file are code, as opposed to a comment or a string.
///
/// The distinction is the whole point of the rewrite: a module path inside a
/// comment or a string literal is prose, and rewriting it edits documentation
/// nobody asked about — while a rewrite that stops at the first occurrence
/// leaves a call pointing at a module that is gone.
pub(crate) fn edit_code_mask(text: &str, hash_comments: bool) -> Vec<bool> {
    let bytes = text.as_bytes();
    let mut mask = vec![true; bytes.len()];
    let mut index = 0usize;
    while index < bytes.len() {
        let rest = &bytes[index..];
        let line_comment = if hash_comments {
            rest.first() == Some(&b'#')
        } else {
            rest.starts_with(b"//")
        };
        if line_comment {
            while index < bytes.len() && bytes[index] != b'\n' {
                mask[index] = false;
                index += 1;
            }
            continue;
        }
        if !hash_comments && rest.starts_with(b"/*") {
            let end = text[index..]
                .find("*/")
                .map(|offset| index + offset + 2)
                .unwrap_or(bytes.len());
            for flag in mask.iter_mut().take(end).skip(index) {
                *flag = false;
            }
            index = end;
            continue;
        }
        if hash_comments && (rest.starts_with(b"\"\"\"") || rest.starts_with(b"'''")) {
            let quote = &text[index..index + 3];
            let end = text[index + 3..]
                .find(quote)
                .map(|offset| index + 3 + offset + 3)
                .unwrap_or(bytes.len());
            for flag in mask.iter_mut().take(end).skip(index) {
                *flag = false;
            }
            index = end;
            continue;
        }
        if rest[0] == b'"' || rest[0] == b'\'' {
            let quote = rest[0];
            let mut cursor = index + 1;
            mask[index] = false;
            while cursor < bytes.len() {
                mask[cursor] = false;
                if bytes[cursor] == b'\\' {
                    cursor += 2;
                    continue;
                }
                if bytes[cursor] == quote || bytes[cursor] == b'\n' {
                    cursor += 1;
                    break;
                }
                cursor += 1;
            }
            index = cursor.min(bytes.len());
            continue;
        }
        index += 1;
    }
    mask
}

pub(crate) fn edit_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Replace every code occurrence of `from` with `to`, where an occurrence is
/// bounded by something that is not part of a name — so `crate::helper` never
/// matches inside `crate::helper_answer`, and `pkg.helper` never matches inside
/// `pkg.helpers`.
pub(crate) fn edit_replace_in_code(text: &str, mask: &[bool], from: &str, to: &str) -> (String, usize) {
    let bytes = text.as_bytes();
    let boundary = |offset: usize, delta: isize| -> bool {
        let probe = if delta < 0 {
            match offset.checked_sub(1) {
                Some(index) => index,
                None => return true,
            }
        } else if offset >= bytes.len() {
            return true;
        } else {
            offset
        };
        !edit_ident_byte(bytes[probe])
    };
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    let mut hits = 0usize;
    while let Some(found) = text[cursor..].find(from) {
        let start = cursor + found;
        let end = start + from.len();
        out.push_str(&text[cursor..start]);
        let leading = text[..start]
            .as_bytes()
            .last()
            .is_none_or(|byte| !edit_ident_byte(*byte) && *byte != b'.' && *byte != b':');
        if mask.get(start).copied().unwrap_or(false)
            && leading
            && boundary(start, -1)
            && boundary(end, 1)
        {
            out.push_str(to);
            hits += 1;
        } else {
            out.push_str(&text[start..end]);
        }
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    (out, hits)
}

pub(crate) fn edit_module_identity(rel: &str) -> Option<ModuleIdentity> {
    let path = std::path::Path::new(rel);
    let extension = path.extension().and_then(|value| value.to_str())?;
    match extension {
        "rs" => {
            let mut segments: Vec<String> = rel
                .trim_end_matches(".rs")
                .split('/')
                .map(str::to_string)
                .collect();
            if segments.first().is_some_and(|first| first == "src") {
                segments.remove(0);
            }
            if segments.last().is_some_and(|last| last == "mod") {
                segments.pop();
            }
            if segments
                .last()
                .is_some_and(|last| last == "lib" || last == "main")
            {
                segments.pop();
            }
            let ident = segments.last()?.clone();
            Some(ModuleIdentity::Rust {
                path: format!("crate::{}", segments.join("::")),
                ident,
            })
        }
        "py" => {
            let mut segments: Vec<String> = rel
                .trim_end_matches(".py")
                .split('/')
                .map(str::to_string)
                .collect();
            if segments.last().is_some_and(|last| last == "__init__") {
                segments.pop();
            }
            if segments.is_empty() {
                return None;
            }
            Some(ModuleIdentity::Python {
                path: segments.join("."),
            })
        }
        _ => None,
    }
}

/// The files that could declare a Rust module: the parent module's own file.
/// `src/a/helper.rs` is declared in `src/a/mod.rs`, `src/helper.rs` in
/// `src/lib.rs` or `src/main.rs`. Only those files may have their bare
/// `mod helper;` and `helper::…` rewritten — every other file has to say
/// `crate::…`, and rewriting a bare identifier there would hit an unrelated name.
pub(crate) fn edit_rust_declaring_files(rel: &str) -> Vec<String> {
    let path = std::path::Path::new(rel);
    let mut dir = path.parent().map(std::path::Path::to_path_buf);
    if path.file_stem().and_then(|stem| stem.to_str()) == Some("mod") {
        dir = dir.and_then(|inner| inner.parent().map(std::path::Path::to_path_buf));
    }
    let Some(dir) = dir else {
        return Vec::new();
    };
    ["mod.rs", "lib.rs", "main.rs"]
        .iter()
        .map(|name| {
            let joined = dir.join(name);
            joined.to_string_lossy().replace('\\', "/")
        })
        .filter(|candidate| candidate != rel)
        .collect()
}

/// Every source file of the same language, so "which files name this module" is
/// answered from what is on disk now — the index is a cache, and a cache is
/// wrong the moment somebody edits a file outside greppy.
pub(crate) fn edit_sibling_sources(root_path: &std::path::Path, extension: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![root_path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) != Some(extension) {
                continue;
            }
            if let Ok(relative) = path.strip_prefix(root_path) {
                found.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    found.sort();
    found
}

/// Which files name the module `rel` is, and how they read once it becomes
/// `to`. This is what makes `move` more than `mv` and `remove` more than `rm`.
pub(crate) fn edit_module_references(
    root_path: &std::path::Path,
    rel: &str,
    to: Option<&str>,
) -> Vec<ModuleReference> {
    let Some(identity) = edit_module_identity(rel) else {
        return Vec::new();
    };
    let target = to.and_then(edit_module_identity);
    let (extension, hash_comments) = match identity {
        ModuleIdentity::Rust { .. } => ("rs", false),
        ModuleIdentity::Python { .. } => ("py", true),
    };
    let declaring = edit_rust_declaring_files(rel);
    let mut found = Vec::new();
    for file in edit_sibling_sources(root_path, extension) {
        if file == rel || Some(file.as_str()) == to {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(root_path.join(&file)) else {
            continue;
        };
        let mut current = text.clone();
        let mut hits = 0usize;
        let apply = |current: &mut String, hits: &mut usize, from: &str, to: &str| {
            let mask = edit_code_mask(current, hash_comments);
            let (next, count) = edit_replace_in_code(current, &mask, from, to);
            *hits += count;
            *current = next;
        };
        match (&identity, &target) {
            (ModuleIdentity::Rust { path, ident }, target) => {
                let (new_path, new_ident) = match target {
                    Some(ModuleIdentity::Rust {
                        path: new_path,
                        ident: new_ident,
                    }) => (new_path.clone(), new_ident.clone()),
                    _ => (path.clone(), ident.clone()),
                };
                // The fully qualified path first: after it is rewritten there is
                // no bare segment left to confuse the second pass.
                apply(&mut current, &mut hits, path, &new_path);
                if declaring.contains(&file) {
                    apply(
                        &mut current,
                        &mut hits,
                        &format!("mod {ident};"),
                        &format!("mod {new_ident};"),
                    );
                    apply(
                        &mut current,
                        &mut hits,
                        &format!("{ident}::"),
                        &format!("{new_ident}::"),
                    );
                }
            }
            (ModuleIdentity::Python { path }, target) => {
                let new_path = match target {
                    Some(ModuleIdentity::Python { path: new_path }) => new_path.clone(),
                    _ => path.clone(),
                };
                apply(&mut current, &mut hits, path, &new_path);
            }
        }
        if hits == 0 {
            continue;
        }
        found.push(ModuleReference {
            file,
            rewritten: (to.is_some() && current != text).then_some(current),
        });
    }
    found
}

/// A path that may not exist yet, resolved against the root and checked for
/// escapes lexically — `canonicalize` cannot answer for a file that is about to
/// be created, and a `..` that leaves the repository must never reach the disk.
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

pub(crate) fn run_edit_write(
    root_path: &std::path::Path,
    file: &str,
    content: Option<String>,
    content_file: Option<String>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    // The content is resolved first on purpose: "there is no new text" is the
    // caller's mistake whether or not the target happens to exist, and naming
    // the target instead would send it looking in the wrong place.
    let bytes = edit_content_bytes(content, content_file)?;
    let (rel, abs) = edit_resolve_new_path(root_path, file)?;
    if abs.is_dir() {
        return Err(EditRefusal::new(
            "file_exists",
            format!("{file} is a directory, not a file"),
            13,
        ));
    }
    if abs.exists() {
        // D5: two ways to do the same thing without a named difference breaks
        // "one name per thing". `write` creates; rewriting is `replace --file`.
        return Err(EditRefusal::new(
            "file_exists",
            format!("{file} already exists; `replace --file {file}` rewrites it"),
            13,
        ));
    }
    if dry_run {
        return Ok(edit_whole_file_record(root_path, &rel, &bytes, b"", false));
    }
    // There is no `greppy mkdir`, so a new module in a new directory has to be
    // possible here or it is not possible through greppy at all.
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
            content: None,
        }],
    );
    edit_journal_crash_hook()?;
    if let Err(error) = std::fs::write(&abs, &bytes) {
        edit_journal_abort(root_path);
        return Err(EditRefusal::new(
            "publish_failed",
            format!("{file}: {error}"),
            16,
        ));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    let mut record = edit_whole_file_record(root_path, &rel, &bytes, b"", true);
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

pub(crate) fn run_edit_move(
    root_path: &std::path::Path,
    file: &str,
    to: &str,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let (rel, abs, content) = edit_read_file(root_path, file)?;
    if abs.is_dir() {
        return Err(EditRefusal::new(
            "file_not_found",
            format!("{file} is a directory, not a file"),
            10,
        ));
    }
    let (new_rel, destination) = edit_resolve_new_path(root_path, to)?;
    // A rename that only changes the case of the name is a rename, and on a
    // case-insensitive file system it is the one an `exists()` check refuses by
    // mistake.
    let case_only = new_rel != rel && new_rel.to_lowercase() == rel.to_lowercase();
    if new_rel == rel {
        // Moving a file onto itself is a no-op, not a way to lose it.
        return Ok(EditRecord {
            files: vec![rel.clone()],
            published: !dry_run,
            extra: vec![
                ("from", serde_json::json!(rel)),
                ("to", serde_json::json!(new_rel)),
                ("rewrote", serde_json::json!(Vec::<String>::new())),
            ],
            ..EditRecord::default()
        });
    }
    if destination.exists() && !case_only {
        return Err(EditRefusal::new(
            "file_exists",
            format!("{to} already exists; nothing was moved"),
            13,
        ));
    }
    let references = edit_module_references(root_path, &rel, Some(&new_rel));
    let rewrote: Vec<String> = references
        .iter()
        .filter(|reference| reference.rewritten.is_some())
        .map(|reference| reference.file.clone())
        .collect();
    let mut record = EditRecord {
        headline: Some(if dry_run {
            format!("would move {rel} -> {new_rel}")
        } else {
            format!("moved {rel} -> {new_rel}")
        }),
        files: std::iter::once(new_rel.clone())
            .chain(rewrote.iter().cloned())
            .collect(),
        published: !dry_run,
        extra: vec![
            ("from", serde_json::json!(rel)),
            ("to", serde_json::json!(new_rel)),
            ("rewrote", serde_json::json!(rewrote)),
        ],
        ..EditRecord::default()
    };
    if dry_run {
        return Ok(record);
    }
    let mut before = vec![
        UndoBefore {
            rel: rel.clone(),
            content: Some(content.clone()),
        },
        UndoBefore {
            rel: new_rel.clone(),
            content: None,
        },
    ];
    for reference in &references {
        if reference.rewritten.is_some() {
            before.push(UndoBefore::read(root_path, &reference.file));
        }
    }
    let transaction = edit_journal_open(root_path, &before);
    edit_journal_crash_hook()?;
    let publish = (|| -> std::io::Result<()> {
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if case_only {
            // Writing the destination and unlinking the source is the same file
            // twice on a case-insensitive file system, which destroys it.
            let interim = abs.with_extension("greppy-rename");
            std::fs::rename(&abs, &interim)?;
            std::fs::rename(&interim, &destination)?;
        } else {
            std::fs::write(&destination, &content)?;
            std::fs::remove_file(&abs)?;
        }
        for reference in &references {
            if let Some(text) = &reference.rewritten {
                std::fs::write(root_path.join(&reference.file), text)?;
            }
        }
        Ok(())
    })();
    if let Err(error) = publish {
        if let Some(record) = edit_journal_read(
            &edit_journal_dir(root_path).join(EDIT_JOURNAL_PENDING),
        ) {
            let _ = edit_journal_restore(root_path, &record, false);
        }
        edit_journal_abort(root_path);
        return Err(EditRefusal::new(
            "publish_failed",
            format!("{file} -> {to}: {error}"),
            16,
        ));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

pub(crate) fn run_edit_remove(
    root_path: &std::path::Path,
    file: &str,
    force: bool,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let (rel, abs, content) = edit_read_file(root_path, file)?;
    if abs.is_dir() {
        return Err(EditRefusal::new(
            "file_not_found",
            format!("{file} is a directory, not a file"),
            10,
        ));
    }
    let references: Vec<String> = edit_module_references(root_path, &rel, None)
        .into_iter()
        .map(|reference| reference.file)
        .collect();
    if !references.is_empty() && !force {
        // Owner decision 4: a delete that leaves dangling imports turns a
        // one-command mistake into a broken build, and it cannot be undone by
        // reading, because the content is gone.
        let listed = references
            .iter()
            .map(|file| format!("\n  {file}"))
            .collect::<String>();
        return Err(
            EditRefusal::new(
                "still_referenced",
                format!("{rel} is still referenced by:{listed}"),
                13,
            )
            .with("references", serde_json::json!(references)),
        );
    }
    let mut record = EditRecord {
        headline: Some(if dry_run {
            format!("would remove {rel}")
        } else {
            format!("removed {rel}")
        }),
        files: vec![rel.clone()],
        published: !dry_run,
        extra: vec![("references", serde_json::json!(references))],
        ..EditRecord::default()
    };
    for reference in record
        .extra
        .first()
        .and_then(|(_, value)| value.as_array())
        .cloned()
        .unwrap_or_default()
    {
        if let Some(file) = reference.as_str() {
            record.notes.push(format!("{file} still references {rel}"));
        }
    }
    if dry_run {
        return Ok(record);
    }
    let transaction = edit_journal_open(
        root_path,
        &[UndoBefore {
            rel: rel.clone(),
            content: Some(content),
        }],
    );
    edit_journal_crash_hook()?;
    if let Err(error) = std::fs::remove_file(&abs) {
        edit_journal_abort(root_path);
        return Err(EditRefusal::new(
            "publish_failed",
            format!("{rel}: {error}"),
            16,
        ));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

/// `apply --plan` over `write` / `move` / `remove`: all of it or none of it, and
/// one entry in the journal, so one `undo` takes the whole batch back.
pub(crate) fn run_edit_plan_whole_file(
    root_path: &std::path::Path,
    operations: &[serde_json::Value],
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let mut before: Vec<UndoBefore> = Vec::new();
    let remember = |root_path: &std::path::Path, rel: &str, before: &mut Vec<UndoBefore>| {
        if !before.iter().any(|item| item.rel == rel) {
            before.push(UndoBefore::read(root_path, rel));
        }
    };
    for operation in operations {
        for key in ["file", "to"] {
            if let Some(file) = operation[key].as_str() {
                if let Ok((rel, _)) = edit_resolve_new_path(root_path, file) {
                    remember(root_path, &rel, &mut before);
                }
            }
        }
        // A move rewrites the files that name the module, and they belong to the
        // same transaction — otherwise one `undo` puts the file back and leaves
        // the imports pointing at a module that is gone again.
        if operation["verb"].as_str() == Some("move") {
            if let (Some(file), Some(to)) = (operation["file"].as_str(), operation["to"].as_str()) {
                if let (Ok((rel, _)), Ok((new_rel, _))) = (
                    edit_resolve_new_path(root_path, file),
                    edit_resolve_new_path(root_path, to),
                ) {
                    for reference in edit_module_references(root_path, &rel, Some(&new_rel)) {
                        remember(root_path, &reference.file, &mut before);
                    }
                }
            }
        }
    }
    let transaction = (!dry_run)
        .then(|| edit_journal_open(root_path, &before))
        .flatten();
    if !dry_run {
        edit_journal_crash_hook()?;
    }
    let mut files = Vec::new();
    let mut failure = None;
    for operation in operations {
        let verb = operation["verb"].as_str().unwrap_or_default();
        let file = operation["file"].as_str().unwrap_or_default().to_string();
        let outcome = match verb {
            "write" => run_edit_write(
                root_path,
                &file,
                operation["content"].as_str().map(str::to_string),
                operation["content_file"].as_str().map(str::to_string),
                dry_run,
                false,
            ),
            "move" => run_edit_move(
                root_path,
                &file,
                operation["to"].as_str().unwrap_or_default(),
                dry_run,
                false,
            ),
            "remove" => run_edit_remove(
                root_path,
                &file,
                operation["force"].as_bool().unwrap_or(false),
                dry_run,
                false,
            ),
            other => Err(EditRefusal::new(
                "invalid_plan",
                format!("`{other}` is not one of the whole-file verbs write, move and remove"),
                20,
            )),
        };
        match outcome {
            Ok(record) => files.extend(record.files),
            Err(refusal) => {
                failure = Some(refusal);
                break;
            }
        }
    }
    if let Some(refusal) = failure {
        // "Many edits as one single change" is the only reason to send a plan:
        // a batch whose first two operations are already on disk is worse than
        // no plan at all, because the caller believes nothing happened.
        if !dry_run {
            if let Some(record) =
                edit_journal_read(&edit_journal_dir(root_path).join(EDIT_JOURNAL_PENDING))
            {
                let _ = edit_journal_restore(root_path, &record, false);
            }
            edit_journal_abort(root_path);
        }
        return Err(refusal);
    }
    files.sort();
    files.dedup();
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    let mut record = EditRecord {
        files,
        published: !dry_run,
        ..EditRecord::default()
    };
    if verify && !dry_run {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
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
    value.insert("schema_version".into(), serde_json::json!(EDIT_RECORD_SCHEMA));
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
pub(crate) fn edit_refusal_json(refusal: &EditRefusal, report_path: Option<&str>) -> serde_json::Value {
    let mut error = serde_json::Map::new();
    error.insert("code".into(), serde_json::json!(refusal.code));
    error.insert("message".into(), serde_json::json!(refusal.message));
    for (key, value) in &refusal.extra {
        error.insert((*key).into(), value.clone());
    }
    let mut value = serde_json::Map::new();
    value.insert("schema_version".into(), serde_json::json!(EDIT_RECORD_SCHEMA));
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
        let root = root_path.canonicalize().unwrap_or_else(|_| root_path.to_path_buf());
        let mut existing = parent;
        while !existing.exists() {
            let Some(next) = existing.parent() else { break };
            existing = next;
        }
        let canonical = existing.canonicalize().unwrap_or_else(|_| existing.to_path_buf());
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
        record.already_as_sent = true;
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
        return Err(EditRefusal::new("publish_failed", format!("{path}: {error}"), 16));
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
            scopes.entry(source.file_path.clone()).or_default().push(
                line_range_to_bytes(
                    &content,
                    source.start_line as usize,
                    span.end_line as usize,
                ),
            );
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
            certificate
                .compact_failure_diagnosis()
                .unwrap_or_else(|| format!("rename {} — nothing written", edit_status_name(certificate.status)))
        };
        return Ok(Err(EditRefusal::new(
            certificate_refusal_code(&certificate),
            message,
            certificate.exit_code(),
        )));
    }
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
        already_as_sent: already,
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
                let kind = if regex { SelectorKind::Pattern } else { SelectorKind::Text };
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
        EditCommand::Delete { symbol, dry_run, verify } => {
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
        EditCommand::DeleteLines { file, lines, dry_run, verify } => {
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
        EditCommand::InsertLines { file, line, new, dry_run, verify } => {
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
        EditCommand::Rename { symbol, name, dry_run, verify } => {
            let outcome = run_trained_rename(root_path, root, &symbol, &name, dry_run, verify)?;
            emit_edit_outcome(outcome, json, None)?
        }
        EditCommand::Undo { id, dry_run, verify } => {
            let outcome = run_edit_undo(root_path, id.as_deref(), dry_run, verify);
            emit_edit_outcome(outcome, json, None)?
        }
    };
    Ok(GrammarDispatch::Handled(code))
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

pub(crate) fn edit_symbol_miss_candidates(
    store: &greppy_store::Store,
    project: &str,
    symbol: &str,
) -> Vec<greppy_edit::certificate::Candidate> {
    let mut candidates = Vec::new();
    for name in symbol_miss_suggestions(store, project, symbol) {
        let Ok(ids) = resolve_symbol_nodes(store, Some(&name)) else {
            continue;
        };
        for id in ids {
            let Ok(Some(node)) = store.get_node(id) else {
                continue;
            };
            if node.file_path.is_empty() || node.start_line < 1 {
                continue;
            }
            let duplicate =
                candidates
                    .iter()
                    .any(|candidate: &greppy_edit::certificate::Candidate| {
                        candidate.qualified_name == node.qualified_name
                            && candidate.path == node.file_path
                    });
            if !duplicate {
                candidates.push(greppy_edit::certificate::Candidate {
                    qualified_name: node.qualified_name,
                    path: node.file_path,
                    line: node.start_line as usize,
                });
            }
            if candidates.len() == 5 {
                return candidates;
            }
        }
    }
    candidates
}
