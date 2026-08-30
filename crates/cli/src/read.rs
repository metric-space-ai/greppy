//! Reading spans, symbols and whole files.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

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
fn read_definition_start(content: &str, definition_start: usize) -> usize {
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
    root_path: &std::path::Path,
    node: greppy_store::Node,
) -> Result<Option<DefinitionRead>> {
    let absolute = root_path.join(&node.file_path);
    let content = match std::fs::read_to_string(&absolute) {
        Ok(content) => content,
        Err(_) => return Ok(None),
    };
    let line_count = read_line_count(&content);
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
            || is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name)
            || !seen.insert((node.file_path.clone(), node.start_line, node.end_line))
        {
            continue;
        }
        nodes.push(node);
    }
    Ok(nodes)
}

fn read_is_ambiguous(target: &str, nodes: &[greppy_store::Node]) -> bool {
    if split_path_qualified(target).is_some() {
        return false;
    }
    let sites = nodes
        .iter()
        .map(|node| node.file_path.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    sites.len() > 1
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
    let store = open_default_store_query_writer(root)?;
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

fn read_json_miss(store: &greppy_store::Store, project: &str, query: &str) -> serde_json::Value {
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
    serde_json::json!({
        "schema_version": "greppy.read.v1",
        "command": "read",
        "status": "not-found",
        "query": query,
        "candidates": candidates,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps the read compatibility decisions at one dispatch boundary"
)]
pub(crate) fn dispatch_read(
    subjects: &[String],
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
    let file_intents = subjects
        .iter()
        .map(|subject| {
            looks_like_path(subject)
                || read_open_file(&root_path, &canonical_root, subject).is_some()
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

    if !file_intents.iter().any(|is_file| *is_file) {
        return dispatch_read_symbols(subjects, head, tail, with_handle, json, path_filters, root);
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
            println!("note: `{subject}` is a path; reading it as a file");
            dispatch_read_files(
                std::slice::from_ref(subject),
                None,
                false,
                with_handle,
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
            let Some(definition) = read_definition(&root_path, node)? else {
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
                serde_json::to_string_pretty(&read_json_miss(&store, &project, query))
                    .map_err(|error| Error::Invalid(format!("serialize read JSON: {error}")))?
            );
            return Ok(1);
        }
        if read_is_ambiguous(query, &nodes) {
            let candidates = nodes
                .iter()
                .map(|node| {
                    serde_json::json!({
                        "qualified_name": node.qualified_name,
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
        let Some(definition) = read_definition(&root_path, nodes[0].clone())? else {
            println!(
                "{}",
                serde_json::to_string_pretty(&read_json_miss(&store, &project, query)).unwrap()
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
        let ids = resolve_symbol_nodes(&store, Some(query))?;
        if nav_refuse_ambiguous(&store, query, &ids)?.is_some() {
            previous_ended_with_newline = true;
            failed = true;
            continue;
        }
        let nodes = read_real_nodes(&store, &ids)?;
        let Some(node) = nodes.first().cloned() else {
            nav_report_missing(&store, &project, query);
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        let Some(definition) = read_definition(&root_path, node)? else {
            nav_report_missing(&store, &project, query);
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

    let mut candidates = Vec::<(usize, usize)>::new();
    let mut children = traversal_root.walk();
    let mut stack = traversal_root
        .named_children(&mut children)
        .map(|node| (node, 0usize))
        .collect::<Vec<_>>();
    while let Some((node, parent_depth)) = stack.pop() {
        let candidate = read_structural_kind(node.kind());
        let node_depth = parent_depth + usize::from(candidate);
        let start = node.start_position().row + 1;
        let end = read_node_end_line(node.end_position().row, node.end_position().column);
        if candidate
            && node_depth >= depth
            && start >= shown_start
            && end <= shown_end
            && end >= start
        {
            candidates.push((start, end));
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
        if non_overlapping.last().is_none_or(|(_, end)| range.0 > *end) {
            non_overlapping.push(range);
        }
    }

    let mut gaps = Vec::with_capacity(non_overlapping.len());
    for (start_line, end_line) in non_overlapping {
        let source = read_line_slice(content, start_line, end_line);
        let sentence = read_summary_sentence(root_path, path, source);
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
    prewarm_summary_daemon();
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
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
        let filtered_ids = nodes.iter().map(|node| node.id).collect::<Vec<_>>();
        if nav_refuse_ambiguous(&store, query, &filtered_ids)?.is_some() {
            previous_ended_with_newline = true;
            failed = true;
            continue;
        }
        let Some(node) = nodes.first().cloned() else {
            nav_report_missing(&store, &project, query);
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        let Some(definition) = read_definition(&root_path, node)? else {
            nav_report_missing(&store, &project, query);
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
        let foldable = matches!(definition.node.label.as_str(), "Function" | "Method");
        if foldable {
            group.push_str(&read_render_smart_source(
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
            )?);
        } else {
            group.push_str(read_line_slice(
                &definition.content,
                definition.start_line,
                definition.end_line,
            ));
        }
        if with_handle {
            if !group.ends_with('\n') {
                group.push('\n');
            }
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
        print!("{group}");
        previous_ended_with_newline = group.ends_with('\n');
    }
    Ok(if failed { 1 } else { 0 })
}

fn read_file_candidate(root_path: &std::path::Path, subject: &str) -> std::path::PathBuf {
    let supplied = std::path::Path::new(subject);
    if supplied.is_absolute() {
        supplied.to_path_buf()
    } else {
        root_path.join(supplied)
    }
}

fn read_open_file(
    root_path: &std::path::Path,
    canonical_root: &std::path::Path,
    subject: &str,
) -> Option<(String, std::path::PathBuf, String)> {
    let candidate = read_file_candidate(root_path, subject);
    let canonical = candidate.canonicalize().ok()?;
    if !canonical.is_file() {
        return None;
    }
    let shown = if let Ok(relative) = canonical.strip_prefix(canonical_root) {
        relative.to_string_lossy().replace('\\', "/")
    } else {
        // Reading is allowed for an explicitly absolute diagnostic/artifact
        // path. Keep relative `../` traversal confined to the repository.
        if !std::path::Path::new(subject).is_absolute() {
            return None;
        }
        canonical.to_string_lossy().replace('\\', "/")
    };
    let content = std::fs::read_to_string(&canonical).ok()?;
    Some((shown, canonical, content))
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
    if end > line_count {
        return Err(Error::Invalid(format!(
            "read-file --lines ends at {end}, but the file has {line_count} lines"
        )));
    }
    Ok((start, end))
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
) -> Result<String> {
    let mut out = format!("{path}:{start_line}-{end_line}\n");
    out.push_str(read_line_slice(content, start_line, end_line));
    if with_handle {
        let store = store.ok_or_else(|| {
            Error::Store("read-file handle requested without an available read store".into())
        })?;
        if !out.ends_with('\n') {
            out.push('\n');
        }
        let full = read_full_handle(root_path, path, content.as_bytes(), start_line, end_line)?;
        out.push_str("handle: ");
        out.push_str(&read_compact_handle(store, project, full)?);
        out.push('\n');
    }
    Ok(out)
}

pub(crate) fn dispatch_read_files(
    paths: &[String],
    lines: Option<&str>,
    all: bool,
    with_handle: bool,
    path_filter_args: &[String],
    root: Option<&str>,
) -> Result<i32> {
    // An exact file/range read is a filesystem operation, not a graph query.
    // Keep it usable while a first index is building and avoid opening a
    // query-writer connection that can collide with the indexer's schema
    // publication. The store is needed only for continuation/handle records.
    let mut store = None;
    let project = project_for(root)?;
    let root_path = resolve_root(root)?;
    let path_filters = prepare_query_path_filters(root, "read-file", "", path_filter_args)?;
    let canonical_root = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.clone());
    let mut failed = false;
    let mut printed = false;
    let mut previous_ended_with_newline = true;
    for path in paths {
        if !path_filters.matches(path) {
            read_begin_group(&mut printed, &mut previous_ended_with_newline);
            println!("outside path filter: {path}");
            previous_ended_with_newline = true;
            failed = true;
            continue;
        }
        let Some((shown, _, content)) = read_open_file(&root_path, &canonical_root, path) else {
            read_begin_group(&mut printed, &mut previous_ended_with_newline);
            println!("no such file: {path}");
            previous_ended_with_newline = true;
            failed = true;
            continue;
        };
        let line_count = read_line_count(&content);
        let (start_line, end_line, continuation) = if let Some(raw) = lines {
            let (start, end) = read_parse_file_range(raw, line_count)?;
            (start, end, None)
        } else if all || line_count <= READ_FILE_PAGE_LINES {
            (1, line_count, None)
        } else {
            let end = READ_FILE_PAGE_LINES;
            if store.is_none() {
                store = Some(open_default_store_query_writer(root)?);
            }
            let id = read_insert_file_pack(
                store.as_ref().expect("read-file store initialized"),
                &project,
                &shown,
                &content,
                end + 1,
            )?;
            (1, end, Some(id))
        };
        if with_handle && store.is_none() {
            store = Some(open_default_store_query_writer(root)?);
        }
        let mut group = read_render_file_page(
            store.as_ref(),
            &project,
            &shown,
            &content,
            start_line,
            end_line,
            with_handle,
            &root_path,
        )?;
        if let Some(id) = continuation {
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
        read_begin_group(&mut printed, &mut previous_ended_with_newline);
        print!("{group}");
        previous_ended_with_newline = group.ends_with('\n');
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
                let mut text = read_render_file_page(
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
