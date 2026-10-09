//! Reading spans, symbols and whole files.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

#[cfg(test)]
#[path = "read_tests.rs"]
mod tests;

pub(crate) fn read_background_job(path: &std::path::Path) -> Option<serde_json::Value> {
    let raw = std::fs::read(path).ok()?;
    serde_json::from_slice(&raw).ok()
}

/// Read one source line (1-based) for the grep-shaped call-site rows the
/// nav commands print (P4). Missing/unreadable files or out-of-range lines
/// return None — the row is skipped, never an error. Trimmed and capped so
/// a pathological line cannot flood the agent's context.
pub(crate) fn read_source_line(
    root: &std::path::Path,
    file_path: &str,
    line: u32,
) -> Option<String> {
    if line == 0 {
        return None;
    }
    let text = std::fs::read_to_string(root.join(file_path)).ok()?;
    let raw = text.lines().nth(line as usize - 1)?.trim();
    if raw.is_empty() {
        return None;
    }
    let mut s = raw.to_string();
    if s.len() > 160 {
        let mut cut = 160;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    Some(s)
}

/// Read the source span for a node from disk and return it as a string,
/// capped at `cap` lines. `file_path` is the node's stored path (relative
/// to the repo root); `root` is the resolved repo root. `start_line` and
/// `end_line` are 1-based inclusive line numbers as stored on the node.
///
/// Robustness (per the task contract): a missing file, an unreadable
/// file, or out-of-range line numbers yield `Ok(None)` so the caller can
/// skip the span gracefully rather than failing the whole command. Only
/// the root-resolution step (which never touches the node's file) can
/// surface a hard error.
///
/// When `with_line_numbers` is set, each emitted line is prefixed with
/// its 1-based line number so an agent can cite exact lines. When the
/// span exceeds `cap` lines it is truncated and a
/// `… (truncated, N more lines)` marker is appended.
///
/// Current indexes store the full tree-sitter definition range. Older indexes
/// may contain only the declaration line (`end_line == start_line`); only for
/// those legacy rows do we recover a body end with [`definition_end_idx`]. A
/// multi-line parser span is authoritative. Extending it heuristically can
/// cross into the next Python method or another adjacent definition.
pub(crate) fn read_span(
    root: &std::path::Path,
    file_path: &str,
    start_line: i64,
    end_line: i64,
    cap: usize,
    with_line_numbers: bool,
) -> Option<String> {
    read_span_with_meta(
        root,
        file_path,
        start_line,
        end_line,
        cap,
        with_line_numbers,
    )
    .map(|span| span.text)
}

pub(crate) fn read_span_with_meta(
    root: &std::path::Path,
    file_path: &str,
    start_line: i64,
    end_line: i64,
    cap: usize,
    with_line_numbers: bool,
) -> Option<SpanRead> {
    // Reject obviously invalid line ranges (the store uses 1-based,
    // inclusive lines; 0 or negative means "unknown").
    if start_line < 1 || end_line < start_line {
        return None;
    }
    let abs = root.join(file_path);
    let content = std::fs::read_to_string(&abs).ok()?;
    let all: Vec<&str> = content.lines().collect();
    // Convert to 0-based indices into the line vector.
    let start_idx = (start_line - 1) as usize;
    if start_idx >= all.len() {
        // start_line is past the end of the file (stale index / edit) —
        // skip gracefully rather than emit nothing useful.
        return None;
    }
    // Stored parser end, clamped to the file.
    let stored_end_idx = std::cmp::min(end_line as usize, all.len()) - 1;
    let end_idx_inclusive = if stored_end_idx == start_idx {
        definition_end_idx(&all, start_idx)
    } else {
        stored_end_idx
    };
    let total_lines = end_idx_inclusive - start_idx + 1;
    let actual_end_line = start_line + total_lines as i64 - 1;
    let shown = std::cmp::min(total_lines, cap);
    let mut out = String::new();
    for (offset, line) in all[start_idx..start_idx + shown].iter().enumerate() {
        if with_line_numbers {
            let lineno = start_line as usize + offset;
            out.push_str(&format!("{lineno:>6}  {line}\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if total_lines > shown {
        out.push_str(&format!(
            "… (truncated, {} more line(s))\n",
            total_lines - shown
        ));
    }
    let omitted_lines = total_lines - shown;
    Some(SpanRead {
        text: out,
        end_line: actual_end_line,
        total_lines,
        shown_lines: shown,
        omitted_lines,
        truncated: omitted_lines > 0,
    })
}

const READ_FILE_PAGE_LINES: usize = 400;
const READ_FILE_PAGE_BYTES: usize = 64 * 1024;

// Probe only one byte beyond the preview budget. UTF-8 errors inside the
// shown prefix are errors; an incomplete character at its edge is omitted.
// The unobserved tail is neither allocated nor validated.
fn read_file_preview(path: &std::path::Path) -> std::io::Result<(String, Option<u64>)> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let mut bytes = Vec::with_capacity(READ_FILE_PAGE_BYTES + 1);
    file.take((READ_FILE_PAGE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    let truncated = bytes.len() > READ_FILE_PAGE_BYTES;
    if truncated {
        bytes.truncate(READ_FILE_PAGE_BYTES);
    }
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text.to_owned(),
        Err(error) if truncated && error.error_len().is_none() => {
            String::from_utf8(bytes[..error.valid_up_to()].to_vec()).expect("validated prefix")
        }
        Err(error) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
    };
    Ok((text, truncated.then_some(size)))
}
const READ_PACK_TTL_SECS: u64 = 365 * 24 * 60 * 60;
const READ_SMART_PACK_KIND: &str = "greppy.read-smart.span.v1";
const READ_FILE_PACK_KIND: &str = "greppy.read-file.page.v1";
const READ_HANDLE_PACK_KIND: &str = "greppy.read.handle.v2";
const COMPACT_HANDLE_PREFIX: &str = "geh2:";

#[derive(Clone)]
struct DefinitionRead {
    node: greppy_store::Node,
    content: String,
    start_line: usize,
    end_line: usize,
}

#[derive(Clone)]
struct FoldGap {
    start_line: usize,
    end_line: usize,
    indent: String,
    sentence: String,
    expand_id: String,
}

fn read_line_count(content: &str) -> usize {
    content.lines().count()
}

fn read_line_slice(content: &str, start_line: usize, end_line: usize) -> &str {
    if end_line < start_line {
        return "";
    }
    let (start, end) = line_range_to_bytes(content.as_bytes(), start_line, end_line);
    &content[start..end]
}

fn read_attribute_group_start(lines: &[&str], end: usize) -> Option<usize> {
    if end == 0 {
        return None;
    }
    let immediate = lines[end - 1].trim();
    if immediate.starts_with("#[") || immediate.starts_with("@") {
        return Some(end - 1);
    }
    if !(immediate.ends_with(']') || immediate.ends_with(')')) {
        return None;
    }
    let mut square = 0i32;
    let mut paren = 0i32;
    for index in (end.saturating_sub(32)..end).rev() {
        let trimmed = lines[index].trim();
        if trimmed.is_empty() {
            return None;
        }
        square += trimmed.matches(']').count() as i32 - trimmed.matches('[').count() as i32;
        paren += trimmed.matches(')').count() as i32 - trimmed.matches('(').count() as i32;
        if (trimmed.starts_with("#[") || trimmed.starts_with('@')) && square <= 0 && paren <= 0 {
            return Some(index);
        }
    }
    None
}

/// Documentation and attributes are part of the definition's read span. The
/// parser/index address remains the definition head; this live-byte scan extends
/// only across contiguous authored interface lines immediately above it.
pub(crate) fn read_definition_start(content: &str, definition_start: usize) -> usize {
    let lines = content.lines().collect::<Vec<_>>();
    let mut cursor = definition_start.saturating_sub(1).min(lines.len());
    loop {
        if cursor == 0 {
            break;
        }
        let trimmed = lines[cursor - 1].trim();
        if trimmed.starts_with("///") {
            cursor -= 1;
            continue;
        }
        if let Some(attribute_start) = read_attribute_group_start(&lines, cursor) {
            cursor = attribute_start;
            continue;
        }
        break;
    }
    cursor + 1
}

fn read_definition(
    store: &greppy_store::Store,
    root_path: &std::path::Path,
    node: greppy_store::Node,
) -> Result<Option<DefinitionRead>> {
    let absolute = root_path.join(&node.file_path);
    let content = match std::fs::read_to_string(&absolute) {
        Ok(content) => content,
        Err(_) => return Ok(None),
    };
    // The freshness gate precedes I/O. Bind the bytes we actually return to
    // this graph snapshot as well, so an edit after that gate cannot pair
    // fresh source with an old node span.
    let indexed = store.get_file_state(&node.project, &node.file_path)?;
    if indexed.is_none_or(|state| state.sha256 != read_sha256(content.as_bytes())) {
        return Err(Error::Workspace(format!(
            "read: source for {} no longer matches the indexed definition {}; no stale span emitted. Run `greppy index .` or wait for the active refresh, then retry; `greppy read-file` reads current bytes without indexed spans",
            node.file_path, node.qualified_name
        )));
    }
    let line_count = read_line_count(&content);
    if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name) {
        // A canonical `__file__` node identifies the current indexed file, not
        // a one-line callable definition. Older graphs persist its bookkeeping
        // span as 1:1, but JSON caller pipelines legitimately carry this node
        // into `read` / `read-smart`. Derive the readable span from the
        // SHA-verified, confined file bytes above so the canonical identity
        // yields real source without inventing a callable owner. Text
        // --head/--tail then slice this full-file DefinitionRead normally.
        return Ok(Some(DefinitionRead {
            node,
            content,
            start_line: 1,
            end_line: line_count.max(1),
        }));
    }
    let node_start = usize::try_from(node.start_line.max(1)).unwrap_or(1);
    if node_start > line_count.max(1) {
        return Ok(None);
    }
    let start_line = read_definition_start(&content, node_start);
    let end_line = usize::try_from(node.end_line.max(node.start_line).max(1))
        .unwrap_or(line_count)
        .min(line_count);
    if end_line < start_line {
        return Ok(None);
    }
    Ok(Some(DefinitionRead {
        node,
        content,
        start_line,
        end_line,
    }))
}

fn read_real_nodes(store: &greppy_store::Store, ids: &[i64]) -> Result<Vec<greppy_store::Node>> {
    let mut nodes = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for id in ids {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if node.file_path.is_empty()
            || node.start_line < 1
            || !seen.insert((
                node.file_path.clone(),
                node.start_line,
                node.end_line,
                node.qualified_name.clone(),
                node.label.clone(),
            ))
        {
            continue;
        }
        // A persisted module caller is represented by the canonical `__file__`
        // graph node. `who-calls --json | greppy read -` must be able to carry
        // that real identity back into the source-reading surface; read-file
        // already exposes the same source, so accepting the anchor here adds no
        // new content or resolver interpretation.
        nodes.push(node);
    }
    Ok(nodes)
}

fn read_is_ambiguous(nodes: &[greppy_store::Node]) -> bool {
    nodes.len() > 1
}

fn read_report_ambiguous(target: &str, nodes: &[greppy_store::Node]) {
    println!("`{target}` is {} definitions", nodes.len());
    for node in nodes {
        println!(
            "{}:{}  {} — greppy read {}",
            node.file_path,
            node.start_line.max(1),
            node.qualified_name,
            node.qualified_name
        );
    }
}

fn read_begin_group(printed: &mut bool, previous_ended_with_newline: &mut bool) {
    if *printed {
        if *previous_ended_with_newline {
            print!("\n");
        } else {
            print!("\n\n");
        }
    }
    *printed = true;
}

fn read_full_handle(
    root_path: &std::path::Path,
    file_path: &str,
    content: &[u8],
    start_line: usize,
    end_line: usize,
) -> Result<String> {
    let (byte_start, byte_end) = line_range_to_bytes(content, start_line, end_line);
    Ok(greppy_edit::EditHandle::for_range(
        root_path,
        std::path::Path::new(file_path),
        content,
        byte_start,
        byte_end,
    )?
    .encode())
}

fn read_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn read_sha256_128(bytes: &[u8]) -> [u8; 16] {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut short = [0u8; 16];
    short.copy_from_slice(&digest[..16]);
    short
}

fn read_hex_bytes(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).ok())
        .collect()
}

fn read_base64url_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut index = 0;
    while index < data.len() {
        let a = data[index] as u32;
        let b = data.get(index + 1).copied().unwrap_or(0) as u32;
        let c = data.get(index + 2).copied().unwrap_or(0) as u32;
        let value = (a << 16) | (b << 8) | c;
        out.push(TABLE[((value >> 18) & 63) as usize] as char);
        out.push(TABLE[((value >> 12) & 63) as usize] as char);
        if index + 1 < data.len() {
            out.push(TABLE[((value >> 6) & 63) as usize] as char);
        }
        if index + 2 < data.len() {
            out.push(TABLE[(value & 63) as usize] as char);
        }
        index += 3;
    }
    out
}

fn read_base64url_decode(text: &str) -> Option<Vec<u8>> {
    fn value(byte: u8) -> Option<u8> {
        match byte {
            b'A'..=b'Z' => Some(byte - b'A'),
            b'a'..=b'z' => Some(byte - b'a' + 26),
            b'0'..=b'9' => Some(byte - b'0' + 52),
            b'-' => Some(62),
            b'_' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut chunk = [0u8; 4];
    let mut used = 0usize;
    for byte in text.bytes() {
        chunk[used] = value(byte)?;
        used += 1;
        if used == 4 {
            out.push((chunk[0] << 2) | (chunk[1] >> 4));
            out.push((chunk[1] << 4) | (chunk[2] >> 2));
            out.push((chunk[2] << 6) | chunk[3]);
            used = 0;
        }
    }
    match used {
        0 => {}
        2 => out.push((chunk[0] << 2) | (chunk[1] >> 4)),
        3 => {
            out.push((chunk[0] << 2) | (chunk[1] >> 4));
            out.push((chunk[1] << 4) | (chunk[2] >> 2));
        }
        _ => return None,
    }
    Some(out)
}

/// Compact format C: one version byte, the pack's 64-bit address, and a
/// 128-bit digest of the full edit handle. The short token is self-checking;
/// the store retains the existing fully qualified handle consumed by edit.
fn read_compact_handle(
    store: &greppy_store::Store,
    project: &str,
    full_handle: String,
) -> Result<String> {
    let digest = read_sha256_128(full_handle.as_bytes());
    let digest_hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let id = store.insert_expand_pack(&greppy_store::NewExpandPack {
        project: project.to_string(),
        command: "read-handle".into(),
        query: digest_hex.clone(),
        graph_generation: 0,
        summary_json: serde_json::json!({
            "kind": READ_HANDLE_PACK_KIND,
            "digest128": digest_hex,
        }),
        payload_text: full_handle,
        payload_json: None,
        ttl_secs: READ_PACK_TTL_SECS,
    })?;
    let id_bytes = read_hex_bytes(&id)
        .filter(|bytes| bytes.len() == 8)
        .ok_or_else(|| Error::Invalid("read handle store returned an invalid address".into()))?;
    let mut binary = Vec::with_capacity(25);
    binary.push(2);
    binary.extend_from_slice(&id_bytes);
    binary.extend_from_slice(&digest);
    Ok(format!(
        "{COMPACT_HANDLE_PREFIX}{}",
        read_base64url_encode(&binary)
    ))
}

pub(crate) fn resolve_compact_read_handle(
    token: &str,
    root: Option<&str>,
) -> Result<Option<String>> {
    let Some(body) = token.strip_prefix(COMPACT_HANDLE_PREFIX) else {
        return Ok(None);
    };
    let Some(binary) = read_base64url_decode(body) else {
        return Ok(None);
    };
    if binary.len() != 25 || binary[0] != 2 {
        return Ok(None);
    }
    let id = binary[1..9]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut expected = [0u8; 16];
    expected.copy_from_slice(&binary[9..25]);
    // Compact handles are continuation metadata, not graph evidence. Resolve
    // them from the workspace-local pack store so a stale linked-worktree Base
    // binding cannot make a filesystem edit handle require reindexing.
    let store = open_default_store_pack_writer(root)?;
    let Some(pack) = store.get_expand_pack(&id)? else {
        return Ok(None);
    };
    if pack.command != "read-handle"
        || pack
            .summary_json
            .get("kind")
            .and_then(serde_json::Value::as_str)
            != Some(READ_HANDLE_PACK_KIND)
        || read_sha256_128(pack.payload_text.as_bytes()) != expected
    {
        return Ok(None);
    }
    Ok(Some(pack.payload_text))
}

fn read_render_block(
    store: &greppy_store::Store,
    project: &str,
    root_path: &std::path::Path,
    definition: &DefinitionRead,
    start_line: usize,
    end_line: usize,
    with_handle: bool,
) -> Result<String> {
    let mut out = format!(
        "{}:{}-{}  {}\n",
        definition.node.file_path,
        start_line,
        end_line,
        nav_short_name(&definition.node)
    );
    let source = read_line_slice(&definition.content, start_line, end_line);
    out.push_str(source);
    if with_handle {
        if !out.ends_with('\n') {
            out.push('\n');
        }
        let full = read_full_handle(
            root_path,
            &definition.node.file_path,
            definition.content.as_bytes(),
            start_line,
            end_line,
        )?;
        let compact = read_compact_handle(store, project, full)?;
        out.push_str("handle: ");
        out.push_str(&compact);
        out.push('\n');
    }
    Ok(out)
}

fn read_text_segments(
    definition: &DefinitionRead,
    head: Option<usize>,
    tail: Option<usize>,
) -> Vec<(usize, usize)> {
    let total = definition.end_line - definition.start_line + 1;
    match (head, tail) {
        (None, None) => vec![(definition.start_line, definition.end_line)],
        (Some(head), None) => vec![(
            definition.start_line,
            definition.start_line + head.min(total) - 1,
        )],
        (None, Some(tail)) => vec![(
            definition.end_line + 1 - tail.min(total),
            definition.end_line,
        )],
        (Some(head), Some(tail)) => vec![
            (
                definition.start_line,
                definition.start_line + head.min(total) - 1,
            ),
            (
                definition.end_line + 1 - tail.min(total),
                definition.end_line,
            ),
        ],
    }
}

// A graph miss cannot rule out definitions in a discovered language without
// a definition provider. Consult existing metadata only: no source scan or index.
fn read_unsupported_definition_coverage(
    store: &greppy_store::Store,
    project: &str,
    path_filters: &QueryPathFilters,
) -> Result<Vec<serde_json::Value>> {
    let mut providers = store
        .list_provider_states(project)?
        .into_iter()
        .filter(|provider| {
            provider.status == "unsupported"
                && provider.files_seen > 0
                // Unlike call-graph completeness, a read miss must retain
                // recognized unsupported source. WGSL is currently registered
                // only as an extension; generic snapshot/log noise stays out.
                && (provider.language == "file extension .wgsl"
                    || (!provider.language.starts_with("file extension .")
                        && provider.language != "no file extension"))
        })
        .collect::<Vec<_>>();
    if !path_filters.is_empty() {
        // An exact indexed file can establish coverage for a filtered lookup.
        // Never infer a missing file or directory's language from its suffix.
        let mut languages = std::collections::BTreeSet::new();
        for path in path_filters.repo_prefixes() {
            let language = greppy_parser::language_for_path(std::path::Path::new(&path));
            if !language.is_supported() {
                if let Some(state) = store.get_file_state(project, &path)? {
                    if state.language == language.name() {
                        languages.insert(state.language);
                    }
                }
            }
        }
        providers.retain(|provider| languages.contains(&provider.language));
    }
    providers.sort_by(|left, right| left.language.cmp(&right.language));
    Ok(providers
        .into_iter()
        .map(|provider| {
            serde_json::json!({
                "language": provider.language,
                "status": provider.status,
                "files_seen": provider.files_seen,
            })
        })
        .collect())
}

fn read_report_missing(
    store: &greppy_store::Store,
    project: &str,
    query: &str,
    root_path: &std::path::Path,
    path_filters: &QueryPathFilters,
) -> Result<()> {
    let coverage = read_unsupported_definition_coverage(store, project, path_filters)?;
    if coverage.is_empty() {
        nav_report_missing(store, project, query);
        return Ok(());
    }
    println!("no indexed symbol `{query}`");
    for provider in coverage.iter().take(3) {
        println!(
            "coverage: definition extraction is unsupported for {} ({} discovered files)",
            provider["language"].as_str().unwrap_or("unknown"),
            provider["files_seen"]
        );
    }
    if coverage.len() > 3 {
        println!(
            "coverage: {} more unsupported languages",
            coverage.len() - 3
        );
    }
    println!(
        "message: this graph lookup cannot rule out a definition in unsupported source; reindexing does not add symbol coverage"
    );
    println!(
        "next: locate it in source: greppy search-pattern {} --fixed --root {}",
        shell_example_arg(query),
        shell_example_arg(&root_path.to_string_lossy())
    );
    println!(
        "next: read a matching file with greppy read-file PATH --root {}",
        shell_example_arg(&root_path.to_string_lossy())
    );
    Ok(())
}

fn read_json_miss(
    store: &greppy_store::Store,
    project: &str,
    query: &str,
    root_path: &std::path::Path,
    path_filters: &QueryPathFilters,
) -> Result<serde_json::Value> {
    let candidates = symbol_miss_suggestions(store, project, query)
        .into_iter()
        .filter_map(|name| {
            let id = resolve_symbol_nodes(store, Some(&name))
                .ok()?
                .first()
                .copied()?;
            let node = store.get_node(id).ok().flatten()?;
            Some(serde_json::json!({
                "qualified_name": node.qualified_name,
                "path": node.file_path,
                "line": node.start_line,
                "kind": node.label,
            }))
        })
        .take(5)
        .collect::<Vec<_>>();
    let coverage = read_unsupported_definition_coverage(store, project, path_filters)?;
    Ok(serde_json::json!({
        "schema_version": "greppy.read.v1",
        "command": "read",
        "status": "not-found",
        "query": query,
        "candidates": candidates,
        "lookup_scope": "indexed-definitions",
        "unsupported_definition_coverage": coverage,
        "source_recovery": format!(
            "greppy search-pattern {} --fixed --root {}",
            shell_example_arg(query),
            shell_example_arg(&root_path.to_string_lossy())
        ),
    }))
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps the read compatibility decisions at one dispatch boundary"
)]
/// Files up to this many lines are printed whole by `read-file` and by
/// `read PATH`; longer indexed sources answer with an outline.
const READ_SMALL_FILE_LINES: usize = 60;

pub(crate) fn dispatch_read(
    subjects: &[String],
    lines: Option<&str>,
    head: Option<usize>,
    tail: Option<usize>,
    with_handle: bool,
    code: bool,
    json: bool,
    path_filters: &[String],
    root: Option<&str>,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let canonical_root = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.clone());
    let file_base = resolve_file_operand_base(root, &root_path);
    if let Some((path, range)) = read_positional_file_range(subjects, &file_base, &canonical_root) {
        let mut retry = format!(
            "greppy read-file {} --lines {}",
            shell_example_arg(path),
            shell_example_arg(&range)
        );
        if with_handle {
            retry.push_str(" --handle");
        }
        if json {
            retry.push_str(" --json");
        }
        for filter in path_filters {
            retry.push_str(&format!(" --path {}", shell_example_arg(filter)));
        }
        if let Some(root) = root {
            retry.push_str(&format!(" --root {}", shell_example_arg(root)));
        }
        return Err(Error::Invalid(format!(
            "read expects symbols; a file location or positional line range uses read-file.\nretry: {retry}"
        )));
    }
    let file_intents = subjects
        .iter()
        .map(|subject| {
            looks_like_path(subject)
                || read_resolve_file(&file_base, &canonical_root, subject).is_some()
        })
        .collect::<Vec<_>>();

    if code {
        let note = "note: `--code` is ignored because `greppy read` already prints source";
        if json {
            eprintln!("{note}");
        } else {
            println!("{note}");
        }
    }

    // `--lines A:B` is the read-file range, addressed by a symbol or a path.
    // It is not a definition slice (`--head` / `--tail` remain that).
    if let Some(raw) = lines {
        return dispatch_read_line_range(subjects, raw, with_handle, json, path_filters, root);
    }

    if !file_intents.iter().any(|is_file| *is_file) {
        return dispatch_read_symbols(subjects, head, tail, with_handle, json, path_filters, root);
    }

    // `read FILE --head N` / `--tail N` asks for lines: serve them through the
    // read-file range instead of answering with a note and an outline.
    if !json && file_intents.iter().all(|is_file| *is_file) {
        if let (Some(n), None) = (head, tail) {
            let range = format!("1:{}", n.max(1));
            return dispatch_read_line_range(
                subjects,
                &range,
                with_handle,
                json,
                path_filters,
                root,
            );
        }
        if let (None, Some(n), [subject]) = (head, tail, subjects) {
            if let Some((_, canonical)) = read_resolve_file(&file_base, &canonical_root, subject) {
                if let Ok(text) = std::fs::read_to_string(&canonical) {
                    let total = text.lines().count().max(1);
                    let range = format!("{}:{total}", total.saturating_sub(n.max(1)) + 1);
                    return dispatch_read_line_range(
                        subjects,
                        &range,
                        with_handle,
                        json,
                        path_filters,
                        root,
                    );
                }
            }
        }
    }

    // `read PATH` on one small file: the file is the answer. An outline of a
    // short file only cost agents a second call (`read-file`) in the v3 bench.
    if !json && head.is_none() && tail.is_none() {
        if let [subject] = subjects {
            if file_intents.first().copied().unwrap_or(false) {
                if let Some((_, canonical)) =
                    read_resolve_file(&file_base, &canonical_root, subject)
                {
                    if let Ok(text) = std::fs::read_to_string(&canonical) {
                        let total = text.lines().count();
                        if (1..=READ_SMALL_FILE_LINES).contains(&total) {
                            let range = format!("1:{total}");
                            return dispatch_read_line_range(
                                subjects,
                                &range,
                                with_handle,
                                json,
                                path_filters,
                                root,
                            );
                        }
                    }
                }
            }
        }
    }

    if head.is_some() || tail.is_some() || json {
        let note = "note: a positional file uses `read-file` paging; --head, --tail, and --json apply only to symbol reads";
        if json {
            eprintln!("{note}");
        } else {
            println!("{note}");
        }
    }

    let mut failed = false;
    for (index, (subject, is_file)) in subjects.iter().zip(file_intents).enumerate() {
        if index > 0 {
            println!();
        }
        let code = if is_file {
            if let Some((shown, canonical)) =
                read_resolve_file(&file_base, &canonical_root, subject)
            {
                let content = std::fs::read_to_string(&canonical).ok();
                if let Some(outline) = content
                    .as_deref()
                    .and_then(|text| read_file_outline(&root_path, &shown, text, true))
                {
                    let filters = prepare_query_path_filters(root, "read", "", path_filters)?;
                    if filters.matches(&shown) {
                        print!("{outline}");
                        continue;
                    }
                }
            }
            println!("note: `{subject}` is a path; reading it as a file");
            dispatch_read_files(
                std::slice::from_ref(subject),
                None,
                false,
                false,
                with_handle,
                false,
                path_filters,
                root,
            )?
        } else {
            dispatch_read_symbols(
                std::slice::from_ref(subject),
                head,
                tail,
                with_handle,
                json,
                path_filters,
                root,
            )?
        };
        failed |= code != 0;
    }
    Ok(i32::from(failed))
}

/// `read SYMBOL|FILE --lines A:B` prints that inclusive file range.
///
/// A path is read directly. A symbol selects the file of each resolved
/// definition (deduped), using the same 1-based coordinates as `read-file
/// --lines`. Ambiguous symbols therefore still show source instead of a
/// usage error: the range does not choose one definition.
fn dispatch_read_line_range(
    subjects: &[String],
    raw: &str,
    with_handle: bool,
    json: bool,
    path_filters: &[String],
    root: Option<&str>,
) -> Result<i32> {
    // Reject a malformed range before resolving symbols, so the failure is
    // the range and not a missing name. EOF clamping happens while reading.
    read_parse_file_range(raw, usize::MAX)?;
    let root_path = resolve_root(root)?;
    let canonical_root = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.clone());
    let file_base = resolve_file_operand_base(root, &root_path);
    let mut files = Vec::new();
    let mut failed = false;
    let mut opened: Option<(greppy_store::Store, String, QueryPathFilters)> = None;
    for subject in subjects {
        let file_intent = looks_like_path(subject)
            || read_resolve_file(&file_base, &canonical_root, subject).is_some();
        if file_intent {
            if !files.iter().any(|existing: &String| existing == subject) {
                files.push(subject.clone());
            }
            continue;
        }
        if opened.is_none() {
            let mut store = open_default_store_query_writer(root)?;
            maybe_reindex_stale(&mut store, root)?;
            let project = project_for(root)?;
            if let Some(code) = graph_stale_gate(
                &store,
                root,
                &project,
                "read",
                json,
                serde_json::json!({ "targets": subjects, "lines": raw }),
                "hits",
            )? {
                return Ok(code);
            }
            let filters = prepare_query_path_filters(root, "read", "", path_filters)?;
            opened = Some((store, project, filters));
        }
        let Some((store, project, filters)) = opened.as_ref() else {
            continue;
        };
        let ids = resolve_symbol_nodes(store, Some(subject))?;
        let mut nodes = read_real_nodes(store, &ids)?;
        nodes.retain(|node| filters.matches(&node.file_path));
        if nodes.is_empty() {
            read_report_missing(store, project, subject, &root_path, filters)?;
            failed = true;
            continue;
        }
        for node in nodes {
            if !files.iter().any(|existing| existing == &node.file_path) {
                files.push(node.file_path);
            }
        }
    }
    if files.is_empty() {
        return Ok(1);
    }
    let code = dispatch_read_files(
        &files,
        Some(raw),
        false,
        false,
        with_handle,
        json,
        path_filters,
        root,
    )?;
    Ok(if failed || code != 0 { 1 } else { 0 })
}

/// Recognize location-shaped misuse before opening a graph or treating the
/// range as another symbol. Literal filenames and qualified symbols keep their
/// existing meaning; this only returns a precise, bounded read-file recovery.
fn read_positional_file_range<'a>(
    subjects: &'a [String],
    file_base: &std::path::Path,
    canonical_root: &std::path::Path,
) -> Option<(&'a str, String)> {
    let file_intent = |path: &str| {
        !path.contains("::")
            && (looks_like_path(path)
                || read_resolve_file(file_base, canonical_root, path).is_some())
    };
    if let [path, range] = subjects {
        if file_intent(path) && read_parse_file_range(range, usize::MAX).is_ok() {
            return Some((path, range.clone()));
        }
    }
    if let [location] = subjects {
        let (path, line) = location.rsplit_once(':')?;
        let line = line.parse::<usize>().ok().filter(|line| *line > 0)?;
        // Existence, not UTF-8 decoding, preserves literal colon filenames.
        // A readable first page may have binary bytes later in the same file.
        if read_resolve_file(file_base, canonical_root, location).is_some() {
            return None;
        }
        if file_intent(path) {
            return Some((path, format!("{line}:{line}")));
        }
    }
    None
}

pub(crate) fn dispatch_read_symbols(
    symbols: &[String],
    head: Option<usize>,
    tail: Option<usize>,
    with_handle: bool,
    json: bool,
    paths: &[String],
    root: Option<&str>,
) -> Result<i32> {
    if head == Some(0) || tail == Some(0) {
        return Err(Error::Invalid(
            "read --head/--tail values must be positive".into(),
        ));
    }
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "read",
        json,
        serde_json::json!({ "targets": symbols }),
        "hits",
    )? {
        return Ok(code);
    }
    let root_path = resolve_root(root)?;
    let path_filters = prepare_query_path_filters(root, "read", "", paths)?;

    if json && (head.is_some() || tail.is_some()) {
        return Err(Error::Invalid(
            "read --json is a whole-symbol shape; --head/--tail are text reads".into(),
        ));
    }

    if json && symbols.len() > 1 {
        let mut hits = Vec::with_capacity(symbols.len());
        for query in symbols {
            let ids = resolve_symbol_nodes(&store, Some(query))?;
            let mut nodes = read_real_nodes(&store, &ids)?;
            nodes.retain(|node| path_filters.matches(&node.file_path));
            let Some(node) = nodes.first().cloned() else {
                return Err(Error::Invalid(format!(
                    "read: `{query}` is not a definition in this repository"
                )));
            };
            let Some(definition) = read_definition(&store, &root_path, node)? else {
                return Err(Error::Invalid(format!(
                    "read: definition span for `{query}` is stale"
                )));
            };
            let source = read_line_slice(
                &definition.content,
                definition.start_line,
                definition.end_line,
            );
            let handle = if with_handle {
                let full = read_full_handle(
                    &root_path,
                    &definition.node.file_path,
                    definition.content.as_bytes(),
                    definition.start_line,
                    definition.end_line,
                )?;
                Some(read_compact_handle(&store, &project, full)?)
            } else {
                None
            };
            hits.push(serde_json::json!({
                "target": query,
                "qualified_name": definition.node.qualified_name,
                "file": definition.node.file_path,
                "line": definition.start_line,
                "path": definition.node.file_path,
                "file_path": definition.node.file_path,
                "start_line": definition.start_line,
                "end_line": definition.end_line,
                "lines": format!("{}:{}", definition.start_line, definition.end_line),
                "source": source,
                "handle": handle,
            }));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": "greppy.read.v1",
                "command": "read",
                "status": "ok",
                "total_exact": hits.len(),
                "shown": hits.len(),
                "hits": hits,
            }))
            .map_err(|error| Error::Invalid(format!("serialize read JSON: {error}")))?
        );
        return Ok(0);
    }

    if json {
        let query = symbols.first().map(String::as_str).unwrap_or("");
        let ids = resolve_symbol_nodes(&store, Some(query))?;
        let mut nodes = read_real_nodes(&store, &ids)?;
        nodes.retain(|node| path_filters.matches(&node.file_path));
        if nodes.is_empty() {
            println!(
                "{}",
                serde_json::to_string_pretty(&read_json_miss(
                    &store,
                    &project,
                    query,
                    &root_path,
                    &path_filters
                )?)
                .map_err(|error| Error::Invalid(format!("serialize read JSON: {error}")))?
            );
            return Ok(1);
        }
        if read_is_ambiguous(&nodes) {
            let candidates = nodes
                .iter()
                .map(|node| {
                    serde_json::json!({
                        "qualified_name": node.qualified_name.clone(),
                        "selector": node.qualified_name.clone(),
                        "path": node.file_path,
                        "line": node.start_line,
                    })
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema_version": "greppy.read.v1",
                    "command": "read",
                    "status": "ambiguous",
                    "query": query,
                    "candidates": candidates,
                }))
                .map_err(|error| Error::Invalid(format!("serialize read JSON: {error}")))?
            );
            return Ok(1);
        }
        let Some(definition) = read_definition(&store, &root_path, nodes[0].clone())? else {
            println!(
                "{}",
                serde_json::to_string_pretty(&read_json_miss(
                    &store,
                    &project,
                    query,
                    &root_path,
                    &path_filters
                )?)
                .unwrap()
            );
            return Ok(1);
        };
        let source = read_line_slice(
            &definition.content,
            definition.start_line,
            definition.end_line,
        );
        let (byte_start, byte_end) = line_range_to_bytes(
            definition.content.as_bytes(),
            definition.start_line,
            definition.end_line,
        );
        let handle = if with_handle {
            let full = read_full_handle(
                &root_path,
                &definition.node.file_path,
                definition.content.as_bytes(),
                definition.start_line,
                definition.end_line,
            )?;
            Some(read_compact_handle(&store, &project, full)?)
        } else {
            None
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": "greppy.read.v1",
                "command": "read",
                "status": "ok",
                "qualified_name": definition.node.qualified_name,
                "path": definition.node.file_path,
                "start_line": definition.start_line,
                "end_line": definition.end_line,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "source": source,
                "handle": handle,
            }))
            .map_err(|error| Error::Invalid(format!("serialize read JSON: {error}")))?
        );
        return Ok(0);
    }

    let mut failed = false;
    let mut printed = false;
    let mut previous_ended_with_newline = true;
    for query in symbols {
        read_begin_group(&mut printed, &mut previous_ended_with_newline);
        let resolved_ids = resolve_symbol_nodes(&store, Some(query))?;
        let mut ids = Vec::with_capacity(resolved_ids.len());
        for id in resolved_ids {
            let Some(node) = store.get_node(id)? else {
                continue;
            };
            if path_filters.matches(&node.file_path) {
                ids.push(id);
            }
        }
        let nodes = read_real_nodes(&store, &ids)?;
        if read_is_ambiguous(&nodes) {
            read_report_ambiguous(query, &nodes);
            previous_ended_with_newline = true;
            failed = true;
            continue;
        }
        let Some(node) = nodes.first().cloned() else {
            read_report_missing(&store, &project, query, &root_path, &path_filters)?;
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        let Some(definition) = read_definition(&store, &root_path, node)? else {
            read_report_missing(&store, &project, query, &root_path, &path_filters)?;
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        let mut group = String::new();
        for (start_line, end_line) in read_text_segments(&definition, head, tail) {
            if !group.is_empty() && !group.ends_with('\n') {
                group.push('\n');
            }
            group.push_str(&read_render_block(
                &store,
                &project,
                &root_path,
                &definition,
                start_line,
                end_line,
                with_handle,
            )?);
        }
        print!("{group}");
        previous_ended_with_newline = group.ends_with('\n');
    }
    Ok(if failed { 1 } else { 0 })
}

fn read_structural_kind(kind: &str) -> bool {
    matches!(
        kind,
        "if_expression"
            | "if_statement"
            | "else_clause"
            | "for_expression"
            | "for_statement"
            | "for_in_statement"
            | "while_expression"
            | "while_statement"
            | "loop_expression"
            | "match_expression"
            | "match_statement"
            | "switch_expression"
            | "switch_statement"
            | "try_statement"
            | "catch_clause"
            | "finally_clause"
            | "with_statement"
            | "do_statement"
            | "synchronized_statement"
            | "async_block"
            | "unsafe_block"
            | "block"
    )
}

fn read_class_member_kind(kind: &str) -> bool {
    matches!(
        kind,
        "function_definition"
            | "method_definition"
            | "method_declaration"
            | "constructor_declaration"
    )
}

fn read_node_end_line(row: usize, column: usize) -> usize {
    row + usize::from(column > 0)
}

fn read_summary_sentence(root_path: &std::path::Path, file_path: &str, source: &str) -> String {
    summarize_definition_span(root_path, file_path, source)
        .into_iter()
        .flatten()
        .map(|sentence| sentence.split_whitespace().collect::<Vec<_>>().join(" "))
        .find(|sentence| !sentence.is_empty())
        .unwrap_or_else(|| "folded source block".to_string())
}

fn read_insert_smart_pack(
    store: &greppy_store::Store,
    project: &str,
    path: &str,
    start_line: usize,
    end_line: usize,
    source: &str,
    sentence: &str,
) -> Result<String> {
    let content_sha256 = read_sha256(source.as_bytes());
    let metadata = serde_json::json!({
        "kind": READ_SMART_PACK_KIND,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "content_sha256": content_sha256,
    });
    store
        .insert_expand_pack(&greppy_store::NewExpandPack {
            project: project.to_string(),
            command: "read-smart".into(),
            query: format!("{path}:{start_line}-{end_line}"),
            graph_generation: 0,
            summary_json: serde_json::json!({
                "text": sentence,
                "content_sha256": content_sha256,
            }),
            payload_text: source.to_string(),
            payload_json: Some(metadata),
            ttl_secs: READ_PACK_TTL_SECS,
        })
        .map_err(Error::from)
}

/// Parse once, count structural blocks from the supplied root, and replace each
/// first block at `depth` with one mechanically identifiable gap line.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps source and structural ranges explicit during rendering"
)]
fn read_render_smart_source(
    store: &greppy_store::Store,
    project: &str,
    root_path: &std::path::Path,
    path: &str,
    content: &str,
    shown_start: usize,
    shown_end: usize,
    structural_start: usize,
    structural_end: usize,
    definition_root: bool,
    depth: usize,
) -> Result<String> {
    let language = greppy_parser::language_for_path(std::path::Path::new(path));
    if !language.is_supported() {
        return Ok(read_line_slice(content, shown_start, shown_end).to_string());
    }
    let Ok(tree) = greppy_parser::parse(language, content.as_bytes()) else {
        return Ok(read_line_slice(content, shown_start, shown_end).to_string());
    };
    let root_node = tree.root_node();
    let mut selected = None;
    let mut selected_width = usize::MAX;
    let mut stack = vec![root_node];
    while let Some(node) = stack.pop() {
        let start = node.start_position().row + 1;
        let end = read_node_end_line(node.end_position().row, node.end_position().column);
        let suitable = if definition_root {
            start == structural_start
                && end >= structural_end
                && node.child_by_field_name("body").is_some()
        } else {
            start == structural_start && end == structural_end && read_structural_kind(node.kind())
        };
        if suitable {
            let width = node.end_byte().saturating_sub(node.start_byte());
            if width < selected_width {
                selected = Some(node);
                selected_width = width;
            }
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.start_position().row < structural_end
                && read_node_end_line(child.end_position().row, child.end_position().column)
                    >= structural_start
            {
                stack.push(child);
            }
        }
    }
    let Some(selected) = selected else {
        return Ok(read_line_slice(content, shown_start, shown_end).to_string());
    };
    let traversal_root = if definition_root {
        let Some(body) = selected.child_by_field_name("body") else {
            return Ok(read_line_slice(content, shown_start, shown_end).to_string());
        };
        body
    } else {
        selected
            .child_by_field_name("body")
            .or_else(|| selected.child_by_field_name("consequence"))
            .unwrap_or(selected)
    };

    let class_root = definition_root
        && matches!(
            selected.kind(),
            "class_definition" | "class_declaration" | "class_specifier"
        );
    let mut candidates = Vec::<(usize, usize, bool)>::new();
    let mut children = traversal_root.walk();
    let mut stack = traversal_root
        .named_children(&mut children)
        .map(|node| (node, 0usize))
        .collect::<Vec<_>>();
    while let Some((node, parent_depth)) = stack.pop() {
        let member_body = class_root
            && node.parent().is_some_and(|parent| {
                read_class_member_kind(parent.kind())
                    && parent
                        .child_by_field_name("body")
                        .is_some_and(|body| body.id() == node.id())
            });
        let candidate = read_structural_kind(node.kind()) || member_body;
        let node_depth = parent_depth + usize::from(candidate);
        let mut start = node.start_position().row + 1;
        let mut end = read_node_end_line(node.end_position().row, node.end_position().column);
        if member_body {
            let body = &content[node.start_byte()..node.end_byte()];
            if body.starts_with('{') {
                // Keep the member signature/opening brace and closing brace.
                // A one-line member has no separate body lines to hide.
                start += 1;
                end = end.saturating_sub(1);
            } else if node
                .parent()
                .is_some_and(|parent| parent.start_position().row == node.start_position().row)
                || node
                    .prev_sibling()
                    .is_some_and(|header| header.end_position().row == node.start_position().row)
            {
                // Python permits an inline suite on the final header line,
                // including a multiline signature. Preserve that header/suite
                // instead of hiding any part of the signature.
                start = end.saturating_add(1);
            }
            if start > end {
                // A nested branch on an inline member's signature line
                // must not hide that signature either.
                continue;
            }
        }
        if candidate
            && node_depth >= depth
            && start >= shown_start
            && end <= shown_end
            && end >= start
        {
            candidates.push((start, end, member_body));
            continue;
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            stack.push((child, node_depth));
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    let mut non_overlapping = Vec::new();
    for range in candidates {
        if non_overlapping
            .last()
            .is_none_or(|(_, end, _)| range.0 > *end)
        {
            non_overlapping.push(range);
        }
    }

    let mut gaps = Vec::with_capacity(non_overlapping.len());
    for (start_line, end_line, member_body) in non_overlapping {
        let source = read_line_slice(content, start_line, end_line);
        // Class overview keeps each member's name/signature. A mechanical gap
        // avoids one extra model inference per method just to fold its body.
        let sentence = if member_body {
            "method body".to_string()
        } else {
            read_summary_sentence(root_path, path, source)
        };
        let expand_id = read_insert_smart_pack(
            store, project, path, start_line, end_line, source, &sentence,
        )?;
        let opening = content.lines().nth(start_line - 1).unwrap_or("");
        let indent = opening
            .chars()
            .take_while(|character| character.is_whitespace())
            .collect::<String>();
        gaps.push(FoldGap {
            start_line,
            end_line,
            indent,
            sentence,
            expand_id,
        });
    }

    let mut out = String::new();
    let mut line = shown_start;
    for gap in gaps {
        if line < gap.start_line {
            out.push_str(read_line_slice(content, line, gap.start_line - 1));
        }
        out.push_str(&format!(
            "{}… {}-{} {} — greppy expand {}\n",
            gap.indent, gap.start_line, gap.end_line, gap.sentence, gap.expand_id
        ));
        line = gap.end_line + 1;
    }
    if line <= shown_end {
        out.push_str(read_line_slice(content, line, shown_end));
    }
    Ok(out)
}

pub(crate) fn dispatch_read_smart(
    symbols: &[String],
    depth: usize,
    with_handle: bool,
    paths: &[String],
    root: Option<&str>,
) -> Result<i32> {
    if depth == 0 {
        return Err(Error::Invalid("read-smart --depth must be positive".into()));
    }
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "read-smart",
        false,
        serde_json::json!({ "targets": symbols }),
        "hits",
    )? {
        return Ok(code);
    }
    prewarm_summary_daemon();
    let root_path = resolve_root(root)?;
    let path_filters = prepare_query_path_filters(root, "read-smart", "", paths)?;
    let mut failed = false;
    let mut printed = false;
    let mut previous_ended_with_newline = true;
    for query in symbols {
        read_begin_group(&mut printed, &mut previous_ended_with_newline);
        let ids = resolve_symbol_nodes(&store, Some(query))?;
        let mut nodes = read_real_nodes(&store, &ids)?;
        nodes.retain(|node| path_filters.matches(&node.file_path));
        if read_is_ambiguous(&nodes) {
            read_report_ambiguous(query, &nodes);
            previous_ended_with_newline = true;
            failed = true;
            continue;
        }
        let Some(node) = nodes.first().cloned() else {
            read_report_missing(&store, &project, query, &root_path, &path_filters)?;
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        let Some(definition) = read_definition(&store, &root_path, node)? else {
            read_report_missing(&store, &project, query, &root_path, &path_filters)?;
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        let mut group = format!(
            "{}:{}-{}  {}\n",
            definition.node.file_path,
            definition.start_line,
            definition.end_line,
            nav_short_name(&definition.node)
        );
        let foldable = matches!(
            definition.node.label.as_str(),
            "Function" | "Method" | "Class"
        );
        let exact_source = read_line_slice(
            &definition.content,
            definition.start_line,
            definition.end_line,
        );
        let rendered = if foldable {
            read_render_smart_source(
                &store,
                &project,
                &root_path,
                &definition.node.file_path,
                &definition.content,
                definition.start_line,
                definition.end_line,
                definition.node.start_line.max(1) as usize,
                definition.end_line,
                true,
                depth,
            )?
        } else {
            exact_source.to_string()
        };
        let folded = rendered != exact_source;
        group.push_str(&rendered);
        if with_handle {
            if !group.ends_with('\n') {
                group.push('\n');
            }
            if folded {
                group.push_str(
                    "note: no edit handle for folded source; request greppy read SYMBOL --handle or read-file PATH --lines A:B --handle\n",
                );
            } else {
                let full = read_full_handle(
                    &root_path,
                    &definition.node.file_path,
                    definition.content.as_bytes(),
                    definition.start_line,
                    definition.end_line,
                )?;
                group.push_str("handle: ");
                group.push_str(&read_compact_handle(&store, &project, full)?);
                group.push('\n');
            }
        }
        print!("{group}");
        previous_ended_with_newline = group.ends_with('\n');
    }
    Ok(if failed { 1 } else { 0 })
}

fn read_file_candidate(root_path: &std::path::Path, subject: &str) -> std::path::PathBuf {
    file_operand_path(root_path, subject)
}

fn read_resolve_file(
    root_path: &std::path::Path,
    canonical_root: &std::path::Path,
    subject: &str,
) -> Option<(String, std::path::PathBuf)> {
    let candidate = read_file_candidate(root_path, subject);
    let canonical = candidate.canonicalize().ok()?;
    if !canonical.is_file() {
        return None;
    }
    let shown = if let Ok(relative) = canonical.strip_prefix(canonical_root) {
        relative.to_string_lossy().replace('\\', "/")
    } else {
        let subject_path = std::path::Path::new(subject);
        if subject_path.is_absolute() {
            // Reading is allowed for an explicitly absolute diagnostic or
            // artifact path.
            canonical.to_string_lossy().replace('\\', "/")
        } else {
            // A dependency directory may be symlinked outside the workspace.
            // Permit that ordinary read while keeping parent traversal from
            // using a symlink plus `..` to escape the workspace implicitly.
            if subject_path
                .components()
                .any(|component| component == std::path::Component::ParentDir)
            {
                return None;
            }
            candidate
                .strip_prefix(canonical_root)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/")
        }
    };
    Some((shown, canonical))
}

fn read_parse_file_range(raw: &str, line_count: usize) -> Result<(usize, usize)> {
    let Some((start, end)) = raw.split_once(':') else {
        return Err(Error::Invalid(format!(
            "read-file --lines expects A:B, got `{raw}`"
        )));
    };
    let start = start
        .parse::<usize>()
        .map_err(|_| Error::Invalid(format!("read-file --lines expects A:B, got `{raw}`")))?;
    let end = end
        .parse::<usize>()
        .map_err(|_| Error::Invalid(format!("read-file --lines expects A:B, got `{raw}`")))?;
    if start == 0 || end < start {
        return Err(Error::Invalid(format!(
            "read-file --lines expects 1 <= A <= B, got `{raw}`"
        )));
    }
    if start > line_count {
        return Err(Error::Invalid(format!(
            "read-file --lines starts at {start}, but the file has {line_count} lines"
        )));
    }
    if end > line_count {
        eprintln!("note: read-file --lines {raw} ends past EOF; clamped to {start}:{line_count}");
    }
    Ok((start, end.min(line_count)))
}

/// A plain explicit span needs neither the whole file nor a graph/handle.
/// Skip preceding bytes without decoding and stop after the requested lines;
/// invalid data elsewhere must not hide a readable diagnostic prefix.
fn read_bounded_file_range(
    reader: &mut impl std::io::BufRead,
    raw: &str,
    path: &str,
) -> Result<(String, usize, usize)> {
    let (start, mut end) = read_parse_file_range(raw, usize::MAX)?;
    let mut selected = Vec::new();
    let mut line_count = 0usize;
    while line_count < end {
        let count = if line_count + 1 < start {
            reader.skip_until(b'\n')
        } else {
            reader.read_until(b'\n', &mut selected)
        }
        .map_err(|error| Error::io(format!("read-file requested lines in {path}"), error))?;
        if count == 0 {
            if line_count < start {
                return Err(Error::Invalid(format!(
                    "read-file --lines starts at {start}, but the file has {line_count} lines"
                )));
            }
            eprintln!(
                "note: read-file --lines {raw} ends past EOF; clamped to {start}:{line_count}"
            );
            end = line_count;
            break;
        }
        line_count += 1;
    }
    let text = String::from_utf8(selected).map_err(|error| {
        Error::io(
            format!("read-file cannot decode requested lines {start}:{end} in {path} as UTF-8"),
            std::io::Error::new(std::io::ErrorKind::InvalidData, error),
        )
    })?;
    Ok((text, start, end))
}

fn read_count(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(byte as char);
    }
    out
}

fn read_insert_file_pack(
    store: &greppy_store::Store,
    project: &str,
    path: &str,
    content: &str,
    start_line: usize,
) -> Result<String> {
    let content_sha256 = read_sha256(content.as_bytes());
    store
        .insert_expand_pack(&greppy_store::NewExpandPack {
            project: project.to_string(),
            command: "read-file".into(),
            query: format!("{path}:{start_line}"),
            graph_generation: 0,
            summary_json: serde_json::json!({
                "text": format!("{path} continues at {start_line}"),
                "content_sha256": content_sha256,
            }),
            payload_text: format!("{path}:{start_line}\n"),
            payload_json: Some(serde_json::json!({
                "kind": READ_FILE_PACK_KIND,
                "path": path,
                "start_line": start_line,
                "content_sha256": content_sha256,
            })),
            ttl_secs: READ_PACK_TTL_SECS,
        })
        .map_err(Error::from)
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps page metadata explicit at the rendering boundary"
)]
fn read_render_file_page(
    store: Option<&greppy_store::Store>,
    project: &str,
    path: &str,
    content: &str,
    start_line: usize,
    end_line: usize,
    with_handle: bool,
    root_path: &std::path::Path,
) -> Result<(String, Option<String>)> {
    let mut out = format!("{path}:{start_line}-{end_line}\n");
    out.push_str(read_line_slice(content, start_line, end_line));
    let mut handle = None;
    if with_handle {
        let store = store.ok_or_else(|| {
            Error::Store("read-file handle requested without an available read store".into())
        })?;
        if !out.ends_with('\n') {
            out.push('\n');
        }
        let full = read_full_handle(root_path, path, content.as_bytes(), start_line, end_line)?;
        out.push_str("handle: ");
        let compact = read_compact_handle(store, project, full)?;
        out.push_str(&compact);
        handle = Some(compact);
        out.push('\n');
    }
    Ok((out, handle))
}

/// An unscoped file read may offer indexed definitions, but must not create an
/// index, prewarm models, repair a graph or acquire a writer just to do so.
/// Exact spans, explicit whole-file reads and cold/unindexed files remain plain
/// filesystem operations even while another task is publishing the graph.
fn read_file_outline(
    root: &std::path::Path,
    shown: &str,
    content: &str,
    symbol_read: bool,
) -> Option<String> {
    read_file_outline_result(root, shown, content, symbol_read).ok()
}

fn read_file_outline_result(
    root: &std::path::Path,
    shown: &str,
    content: &str,
    symbol_read: bool,
) -> std::result::Result<String, &'static str> {
    let path = greppy_core::cache::workspace_store_path(root);
    if !path.is_file() {
        return Err("no index exists for this workspace");
    }
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::read_only())
        .map_err(|_| "the index cannot be opened read-only")?;
    let store = if let Some((base, commit)) = crate::store_cow::overlay_environment(root)
        .map_err(|_| "the linked-worktree index identity is unavailable")?
    {
        let visibility =
            crate::store_cow::visibility_for_open_connection(root, &commit, store.conn())
                .map_err(|_| "the linked-worktree visibility is unavailable")?;
        store
            .attach_overlay(&base, &visibility)
            .map_err(|_| "the linked-worktree Base cannot be opened read-only")?
    } else {
        store
    };
    let project = workspace_locator::project_identity(root);
    // Same-length edits can leave obsolete indexed spans in bounds. Unknown
    // or changed fingerprints fall back without index or repair work.
    let indexed = store
        .get_file_state(&project, shown)
        .map_err(|_| "the indexed file state is unavailable")?
        .ok_or("the file is not indexed")?;
    use sha2::Digest;
    let content_hash = format!("{:x}", sha2::Sha256::digest(content.as_bytes()));
    if indexed.sha256 != content_hash {
        return Err("the indexed fingerprint is stale for the current file");
    }
    let mut nodes = store
        .list_nodes_for_file(&project, shown)
        .map_err(|_| "indexed definitions cannot be read")?;
    let line_count = read_line_count(content) as i64;
    nodes.retain(|node| {
        matches!(
            node.label.as_str(),
            "Function" | "Method" | "Class" | "Struct" | "Enum" | "Trait"
        ) && node.start_line > 0
            && node.start_line <= line_count
            && !node.qualified_name.is_empty()
    });
    nodes.sort_by_key(|node| (node.start_line, std::cmp::Reverse(node.end_line)));
    let mut top_level = Vec::<greppy_store::Node>::new();
    for node in nodes {
        if !top_level
            .iter()
            .any(|parent| parent.start_line <= node.start_line && parent.end_line >= node.end_line)
        {
            top_level.push(node);
        }
    }
    if top_level.is_empty() {
        return Err("the current index has no eligible definitions for this file");
    }
    let mut outline = if symbol_read {
        format!("`{shown}` is a file — read a symbol:\n")
    } else {
        format!("Source outline for `{shown}` — fingerprint-verified indexed definitions:\n")
    };
    for node in &top_level {
        outline.push_str(&format!(
            "{}:{}  {}  {}\n",
            shown,
            node.start_line,
            node.qualified_name,
            node.label.to_ascii_lowercase()
        ));
    }
    outline.push_str(&format!(
        "read one: greppy read {} · lines: greppy read-file {shown} --lines A:B\n",
        top_level[0].qualified_name
    ));
    Ok(outline)
}

pub(crate) fn dispatch_read_files(
    paths: &[String],
    lines: Option<&str>,
    all: bool,
    outline: bool,
    with_handle: bool,
    json_output: bool,
    path_filter_args: &[String],
    root: Option<&str>,
) -> Result<i32> {
    // An exact file/range read is a filesystem operation, not a graph query.
    // Keep it usable while a first index is building and avoid opening a
    // query-writer connection that can collide with the indexer's schema
    // publication. The store is needed only for continuation/handle records.
    let root_path = resolve_root(root)?;
    let file_base = resolve_file_operand_base(root, &root_path);
    let path_filters = if path_filter_args.is_empty() {
        QueryPathFilters::default()
    } else {
        prepare_query_path_filters(root, "read-file", "", path_filter_args)?
    };
    // `resolve_root` already returns a canonical path. Exact range/all reads
    // need neither a project identity nor a graph Store; derive both lazily
    // only for continuation packs or explicit handles. Re-resolving a linked
    // worktree several times made a plain file read crawl under filesystem
    // pressure and could leave callers waiting with no output.
    // File operands join against `file_base` (the explicit --root, if any);
    // shown paths, handles and continuation packs stay workspace-relative.
    let canonical_root = root_path.clone();
    let mut project = None::<String>;
    let mut store = None;
    let mut failed = false;
    let mut printed = false;
    let mut previous_ended_with_newline = true;
    let mut json_files = Vec::new();
    for path in paths {
        let Some((shown, canonical)) = read_resolve_file(&file_base, &canonical_root, path) else {
            if json_output {
                json_files.push(serde_json::json!({"path": path, "error": "no such file"}));
                failed = true;
                continue;
            }
            read_begin_group(&mut printed, &mut previous_ended_with_newline);
            println!("no such file: {path}");
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        if !path_filters.matches(&shown) {
            if json_output {
                json_files.push(serde_json::json!({"path": path, "error": "outside path filter"}));
                failed = true;
                continue;
            }
            read_begin_group(&mut printed, &mut previous_ended_with_newline);
            println!("outside path filter: {path}");
            previous_ended_with_newline = true;
            failed = true;
            continue;
        }
        if let Some(raw) = lines.filter(|_| !with_handle) {
            let span = std::fs::File::open(&canonical)
                .map_err(|error| Error::io(format!("open read-file {path}"), error))
                .and_then(|file| {
                    read_bounded_file_range(&mut std::io::BufReader::new(file), raw, path)
                });
            let (text, start, end) = match span {
                Ok(span) => span,
                Err(error @ Error::Io { .. }) => {
                    if json_output {
                        json_files.push(serde_json::json!({
                            "path": path,
                            "error": error.to_string(),
                        }));
                        failed = true;
                        continue;
                    }
                    read_begin_group(&mut printed, &mut previous_ended_with_newline);
                    println!("cannot read file {path}: {error}");
                    previous_ended_with_newline = true;
                    failed = true;
                    continue;
                }
                Err(error) => return Err(error),
            };
            if json_output {
                json_files.push(serde_json::json!({
                    "path": shown,
                    "start_line": start,
                    "end_line": end,
                    "content": text,
                }));
                continue;
            }
            let group = format!("{shown}:{start}-{end}\n{text}");
            read_begin_group(&mut printed, &mut previous_ended_with_newline);
            let stdout = std::io::stdout();
            let mut output = stdout.lock();
            std::io::Write::write_all(&mut output, group.as_bytes())
                .map_err(|error| Error::Store(format!("write read-file output: {error}")))?;
            std::io::Write::flush(&mut output)
                .map_err(|error| Error::Store(format!("flush read-file output: {error}")))?;
            previous_ended_with_newline = group.ends_with('\n');
            continue;
        }
        let preview = if lines.is_none() && !all && !outline {
            read_file_preview(&canonical)
        } else {
            std::fs::read_to_string(&canonical).map(|text| (text, None))
        };
        let (content, truncated_size) = match preview {
            Ok(preview) => preview,
            Err(error) => {
                if json_output {
                    json_files.push(serde_json::json!({
                        "path": path,
                        "error": error.to_string(),
                    }));
                    failed = true;
                    continue;
                }
                read_begin_group(&mut printed, &mut previous_ended_with_newline);
                if error.kind() == std::io::ErrorKind::NotFound {
                    println!("no such file: {path}");
                } else {
                    println!("cannot read file {path}: {error}");
                }
                previous_ended_with_newline = true;
                failed = true;
                continue;
            }
        };
        let line_count = read_line_count(&content);
        if outline {
            match read_file_outline_result(&root_path, &shown, &content, false) {
                Ok(result) => {
                    if json_output {
                        json_files.push(serde_json::json!({
                            "path": shown,
                            "kind": "outline",
                            "content": result,
                            "fingerprint_verified": true,
                            "handle": serde_json::Value::Null,
                            "handle_unavailable": with_handle.then_some(
                                "outlines cannot produce an edit handle; request an explicit --lines A:B span"
                            ),
                            "lines_command": format!("greppy read-file {shown} --lines A:B"),
                        }));
                    } else {
                        read_begin_group(&mut printed, &mut previous_ended_with_newline);
                        print!("{result}");
                        if with_handle {
                            println!(
                                "note: no handle for an outline; request an explicit --lines A:B span"
                            );
                        }
                        previous_ended_with_newline = true;
                    }
                }
                Err(reason) => {
                    failed = true;
                    if json_output {
                        json_files.push(serde_json::json!({
                            "path": shown,
                            "kind": "outline_unavailable",
                            "error": reason,
                            "fingerprint_verified": false,
                            "handle": serde_json::Value::Null,
                            "handle_unavailable": with_handle.then_some(
                                "no verified outline exists; request an explicit --lines A:B span"
                            ),
                            "lines_command": format!("greppy read-file {shown} --lines A:B"),
                        }));
                    } else {
                        read_begin_group(&mut printed, &mut previous_ended_with_newline);
                        println!("outline unavailable for `{shown}`: {reason}");
                        println!("read lines: greppy read-file {shown} --lines A:B");
                        previous_ended_with_newline = true;
                    }
                }
            }
            continue;
        }
        // A line page is not a byte budget: generated JSON/NDJSON can put
        // megabytes on one line. Bound only implicit reads; explicit ranges
        // and --all remain exact. Do this before outlines and pack creation
        // so a partial line never acquires a misleading editable handle.
        if let Some(size_at_open) = truncated_size {
            let page = read_line_slice(&content, 1, line_count.min(READ_FILE_PAGE_LINES));
            let prefix = page;
            let end = prefix.len();
            let complete_lines = prefix.bytes().filter(|byte| *byte == b'\n').count();
            let partial = !prefix.ends_with('\n');
            let shown_end = complete_lines + usize::from(partial);
            let resume = complete_lines + 1;
            // Absolute operands keep this recovery command exact even
            // when --root selected a nested directory or an external file.
            let operand = format!("'{}'", canonical.to_string_lossy().replace('\'', "'\\''"));
            if json_output {
                json_files.push(serde_json::json!({
                    "path": shown,
                    "start_line": 1,
                    "end_line": shown_end,
                    "content": prefix,
                    "truncated": true,
                    "last_line_partial": partial,
                    "source_bytes_read": end,
                    "source_bytes_at_open": size_at_open,
                    "next_line": resume,
                    "handle": serde_json::Value::Null,
                    "handle_unavailable": with_handle.then_some(
                        "byte-truncated previews cannot produce an edit handle; request an explicit --lines A:B span"
                    ),
                }));
                continue;
            }
            read_begin_group(&mut printed, &mut previous_ended_with_newline);
            println!(
                "{shown}:1-{shown_end}{}",
                if partial { " (last line partial)" } else { "" }
            );
            print!("{prefix}");
            if partial {
                println!();
            }
            println!(
                "truncated at {end} source bytes (default limit {READ_FILE_PAGE_BYTES}); total line count unknown"
            );
            if let Some(omitted) = size_at_open.checked_sub(end as u64) {
                println!(
                    "{omitted} source bytes omitted according to file size at open ({size_at_open} bytes)"
                );
            }
            println!("next line: greppy read-file {operand} --lines {resume}:{resume}");
            if partial {
                println!("note: the next command rereads the partial line in full");
            }
            println!("full file: greppy read-file {operand} --all");
            if with_handle {
                println!(
                    "note: no handle for a byte-truncated page; request an explicit --lines A:B span"
                );
            }
            previous_ended_with_newline = true;
            continue;
        }
        if lines.is_none() && !all && line_count > READ_SMALL_FILE_LINES {
            if let Some(outline) = read_file_outline(&root_path, &shown, &content, false) {
                if json_output {
                    json_files.push(serde_json::json!({
                        "path": shown,
                        "kind": "outline",
                        "content": outline,
                        "handle": serde_json::Value::Null,
                        "handle_unavailable": with_handle.then_some(
                            "outlines cannot produce an edit handle; request an explicit --lines A:B span"
                        ),
                    }));
                    continue;
                }
                read_begin_group(&mut printed, &mut previous_ended_with_newline);
                print!("{outline}");
                if with_handle {
                    println!(
                        "note: for a handle, choose a symbol or an explicit --lines A:B span first"
                    );
                }
                previous_ended_with_newline = true;
                continue;
            }
        }
        let (start_line, end_line, continuation) = if let Some(raw) = lines {
            let (start, end) = read_parse_file_range(raw, line_count)?;
            (start, end, None)
        } else if all || line_count <= READ_FILE_PAGE_LINES {
            (1, line_count, None)
        } else {
            let end = READ_FILE_PAGE_LINES;
            if project.is_none() {
                project = Some(workspace_locator::project_identity(&root_path));
            }
            if store.is_none() {
                store = Some(open_default_store_pack_writer(root)?);
            }
            let id = read_insert_file_pack(
                store.as_ref().expect("read-file store initialized"),
                project.as_deref().expect("read-file project initialized"),
                &shown,
                &content,
                end + 1,
            )?;
            (1, end, Some(id))
        };
        if with_handle && store.is_none() {
            if project.is_none() {
                project = Some(workspace_locator::project_identity(&root_path));
            }
            store = Some(open_default_store_pack_writer(root)?);
        }
        let (mut group, handle) = read_render_file_page(
            store.as_ref(),
            project.as_deref().unwrap_or(""),
            &shown,
            &content,
            start_line,
            end_line,
            with_handle,
            &root_path,
        )?;
        if let Some(id) = continuation.as_ref() {
            if !group.ends_with('\n') {
                group.push('\n');
            }
            group.push_str(&format!(
                "{} more lines — greppy expand {} continues at {}\n",
                read_count(line_count - end_line),
                id,
                end_line + 1
            ));
        }
        if group.is_empty() {
            return Err(Error::Store(format!(
                "read-file produced no output for existing file `{shown}` lines {start_line}:{end_line}; retry the command and report this invariant failure"
            )));
        }
        if json_output {
            let continuation = continuation.as_ref().map(|id| {
                serde_json::json!({
                    "id": id,
                    "next_line": end_line + 1,
                    "remaining_lines": line_count - end_line,
                })
            });
            json_files.push(serde_json::json!({
                "path": shown,
                "start_line": start_line,
                "end_line": end_line,
                "content": read_line_slice(&content, start_line, end_line),
                "handle": handle,
                "continuation": continuation,
            }));
            continue;
        }
        read_begin_group(&mut printed, &mut previous_ended_with_newline);
        {
            let stdout = std::io::stdout();
            let mut output = stdout.lock();
            std::io::Write::write_all(&mut output, group.as_bytes())
                .map_err(|error| Error::Store(format!("write read-file output: {error}")))?;
            std::io::Write::flush(&mut output)
                .map_err(|error| Error::Store(format!("flush read-file output: {error}")))?;
        }
        previous_ended_with_newline = group.ends_with('\n');
    }
    if json_output {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "command": "read-file",
                "files": json_files,
            }))
            .map_err(|error| Error::Store(format!("serialize read-file JSON: {error}")))?
        );
    }
    Ok(if failed { 1 } else { 0 })
}

fn read_locate_file_pack(
    store: &greppy_store::Store,
    root_path: &std::path::Path,
    project: &str,
    path: &str,
    expected_hash: &str,
) -> Result<Option<(String, String)>> {
    if let Ok(content) = std::fs::read_to_string(root_path.join(path)) {
        if read_sha256(content.as_bytes()) == expected_hash {
            return Ok(Some((path.to_string(), content)));
        }
    }
    let mut matches = Vec::new();
    for state in store.list_file_states(project)? {
        let Ok(content) = std::fs::read_to_string(root_path.join(&state.rel_path)) else {
            continue;
        };
        if read_sha256(content.as_bytes()) == expected_hash {
            matches.push((state.rel_path, content));
            if matches.len() > 1 {
                return Ok(None);
            }
        }
    }
    Ok(matches.pop())
}

fn read_payload_line_count(payload: &str) -> usize {
    let newlines = payload
        .as_bytes()
        .iter()
        .filter(|byte| **byte == b'\n')
        .count();
    newlines + usize::from(!payload.is_empty() && !payload.ends_with('\n'))
}

fn read_find_payload(content: &str, payload: &str) -> Vec<(usize, usize)> {
    if payload.is_empty() {
        return Vec::new();
    }
    let mut matches = Vec::new();
    let mut offset = 0usize;
    while let Some(relative) = content[offset..].find(payload) {
        let byte = offset + relative;
        let start_line = content.as_bytes()[..byte]
            .iter()
            .filter(|value| **value == b'\n')
            .count()
            + 1;
        let end_line = start_line + read_payload_line_count(payload).saturating_sub(1);
        matches.push((start_line, end_line));
        offset = byte + 1;
    }
    matches
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps stored and live span identity explicit during relocation"
)]
fn read_locate_smart_pack(
    store: &greppy_store::Store,
    root_path: &std::path::Path,
    project: &str,
    path: &str,
    start_line: usize,
    end_line: usize,
    expected_hash: &str,
    payload: &str,
) -> Result<Option<(String, String, usize, usize)>> {
    if let Ok(content) = std::fs::read_to_string(root_path.join(path)) {
        let current = read_line_slice(&content, start_line, end_line);
        if read_sha256(current.as_bytes()) == expected_hash && current == payload {
            return Ok(Some((path.to_string(), content, start_line, end_line)));
        }
    }
    let mut found = Vec::new();
    for state in store.list_file_states(project)? {
        let Ok(content) = std::fs::read_to_string(root_path.join(&state.rel_path)) else {
            continue;
        };
        for (start, end) in read_find_payload(&content, payload) {
            found.push((state.rel_path.clone(), content.clone(), start, end));
            if found.len() > 1 {
                return Ok(None);
            }
        }
    }
    Ok(found.pop())
}

pub(crate) fn dispatch_read_expand(
    store: &greppy_store::Store,
    pack: &greppy_store::ExpandPack,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let Some(metadata) = pack.payload_json.as_ref() else {
        println!("expand: invalid {} pack", pack.command);
        return Ok(1);
    };
    let root_path = resolve_root(root)?;
    let project = &pack.project;
    let result = match pack.command.as_str() {
        "read-file" => {
            if metadata.get("kind").and_then(serde_json::Value::as_str) != Some(READ_FILE_PACK_KIND)
            {
                None
            } else {
                let path = metadata
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let start = metadata
                    .get("start_line")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0);
                let hash = metadata
                    .get("content_sha256")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let Some((path, content)) =
                    read_locate_file_pack(store, &root_path, project, path, hash)?
                else {
                    println!("expand: read-file changed since this pack was created");
                    return Ok(1);
                };
                let line_count = read_line_count(&content);
                if start == 0 || start > line_count {
                    println!("expand: read-file changed since this pack was created");
                    return Ok(1);
                }
                let end = (start + READ_FILE_PAGE_LINES - 1).min(line_count);
                let (mut text, _) = read_render_file_page(
                    Some(store),
                    project,
                    &path,
                    &content,
                    start,
                    end,
                    false,
                    &root_path,
                )?;
                let mut next = serde_json::Value::Null;
                if end < line_count {
                    let id = read_insert_file_pack(store, project, &path, &content, end + 1)?;
                    text.push_str(&format!(
                        "{} more lines — greppy expand {} continues at {}\n",
                        read_count(line_count - end),
                        id,
                        end + 1
                    ));
                    next = serde_json::json!(id);
                }
                Some((
                    text,
                    serde_json::json!({
                        "kind": READ_FILE_PACK_KIND,
                        "path": path,
                        "start_line": start,
                        "end_line": end,
                        "next_expand_id": next,
                    }),
                ))
            }
        }
        "read-smart" => {
            if metadata.get("kind").and_then(serde_json::Value::as_str)
                != Some(READ_SMART_PACK_KIND)
            {
                None
            } else {
                let path = metadata
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let start = metadata
                    .get("start_line")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0);
                let end = metadata
                    .get("end_line")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0);
                let hash = metadata
                    .get("content_sha256")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                if read_sha256(pack.payload_text.as_bytes()) != hash {
                    println!("expand: read-smart pack hash drift; refusing unverified source");
                    return Ok(1);
                }
                let Some((path, content, start, end)) = read_locate_smart_pack(
                    store,
                    &root_path,
                    project,
                    path,
                    start,
                    end,
                    hash,
                    &pack.payload_text,
                )?
                else {
                    println!("expand: read-smart span changed since this pack was created");
                    return Ok(1);
                };
                let text = read_render_smart_source(
                    store, project, &root_path, &path, &content, start, end, start, end, false, 1,
                )?;
                Some((
                    text,
                    serde_json::json!({
                        "kind": READ_SMART_PACK_KIND,
                        "path": path,
                        "start_line": start,
                        "end_line": end,
                    }),
                ))
            }
        }
        _ => None,
    };
    let Some((text, value)) = result else {
        println!("expand: invalid {} pack", pack.command);
        return Ok(1);
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "id": pack.id,
                "project": pack.project,
                "command": pack.command,
                "query": pack.query,
                "graph_generation": pack.graph_generation,
                "created_at": pack.created_at,
                "expires_at": pack.expires_at,
                "summary": pack.summary_json,
                "payload_text": text,
                "payload_json": value,
            }))
            .map_err(|error| Error::Invalid(format!("serialize expand JSON: {error}")))?
        );
    } else {
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
    }
    Ok(0)
}

/// Read an edit source argument: a file path, or `-` for stdin.
pub(crate) fn read_source_arg(source_file: &str) -> Result<Vec<u8>> {
    if source_file == "-" {
        use std::io::Read;
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|source| Error::Io {
                context: "read edit source from stdin".into(),
                source,
            })?;
        return Ok(buf);
    }
    std::fs::read(source_file).map_err(|source| Error::Io {
        context: format!("read {source_file}"),
        source,
    })
}

pub(crate) fn read_last_used_unix_secs(dir: &std::path::Path) -> u64 {
    let marker = dir.join(".lastused");
    std::fs::read_to_string(&marker)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .or_else(|| {
            std::fs::metadata(&marker)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|age| age.as_secs())
        })
        .or_else(|| {
            std::fs::metadata(dir)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|age| age.as_secs())
        })
        .unwrap_or(0)
}
