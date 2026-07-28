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

pub(crate) fn read_file_candidate(
    root_path: &std::path::Path,
    subject: &str,
) -> std::path::PathBuf {
    let supplied = std::path::Path::new(subject);
    if supplied.is_absolute() {
        return supplied.to_path_buf();
    }
    if let Ok(cwd) = std::env::current_dir() {
        let from_cwd = cwd.join(supplied);
        if from_cwd.exists() {
            return from_cwd;
        }
    }
    root_path.join(supplied)
}

pub(crate) fn read_subject_is_path(subject: &str, root: Option<&str>) -> Result<bool> {
    let root_path = resolve_root(root)?;
    if read_file_candidate(&root_path, subject).exists() {
        return Ok(true);
    }
    let supplied = std::path::Path::new(subject);
    if supplied.is_absolute()
        || subject.starts_with('.')
        || subject.contains('/')
        || subject.contains('\\')
    {
        return Ok(true);
    }
    let path_extension = supplied
        .extension()
        .and_then(|extension| extension.to_str());
    Ok(matches!(
        path_extension,
        Some(
            "rs" | "py"
                | "js"
                | "jsx"
                | "mjs"
                | "cjs"
                | "ts"
                | "tsx"
                | "go"
                | "rb"
                | "java"
                | "c"
                | "h"
                | "cpp"
                | "cc"
                | "cxx"
                | "hpp"
                | "hh"
                | "cs"
                | "php"
                | "sh"
                | "bash"
                | "lua"
                | "kt"
                | "kts"
                | "scala"
                | "sc"
                | "swift"
                | "zig"
                | "r"
                | "json"
                | "toml"
                | "yaml"
                | "yml"
                | "md"
                | "txt"
        )
    ))
}

pub(crate) fn dispatch_read_file(
    subject: &str,
    lines: Option<&str>,
    with_handle: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    const READ_FILE_JSON_SCHEMA_VERSION: &str = "greppy.read-file.v1";
    let root_path = resolve_root(root)?;
    let canonical_root = root_path.canonicalize().map_err(|source| Error::Io {
        context: format!("canonicalize {}", root_path.display()),
        source,
    })?;
    let candidate = read_file_candidate(&root_path, subject);
    let canonical = candidate.canonicalize().ok();
    let regular_file = canonical
        .as_deref()
        .is_some_and(|path| path.starts_with(&canonical_root) && path.is_file());
    if !regular_file {
        let suggestions = closest_read_paths(&canonical_root, subject)?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema_version": READ_FILE_JSON_SCHEMA_VERSION,
                    "command": "read",
                    "status": "not-found",
                    "path": subject,
                    "path_candidates": suggestions,
                }))
                .map_err(|error| Error::Invalid(format!("serialize read file JSON: {error}")))?
            );
        } else if suggestions.is_empty() {
            println!("read: file `{subject}` not found");
        } else {
            println!("read: file `{subject}` not found; closest paths:");
            for suggestion in &suggestions {
                println!("  {suggestion}");
            }
            println!("try: greppy read {}", shell_example_arg(&suggestions[0]));
        }
        return Ok(10);
    }
    let canonical = canonical.expect("regular file check requires a canonical path");
    let relative = canonical.strip_prefix(&canonical_root).map_err(|_| {
        Error::Invalid(format!(
            "read path `{subject}` resolves outside workspace {}",
            canonical_root.display()
        ))
    })?;
    let shown_path = relative.to_string_lossy().replace('\\', "/");
    let content = std::fs::read_to_string(&canonical).map_err(|source| Error::Io {
        context: format!("read {}", canonical.display()),
        source,
    })?;
    let file_lines = content.lines().collect::<Vec<_>>();
    let (start, end) = parse_read_line_range(lines, file_lines.len())?;
    let selected = if end < start {
        &file_lines[0..0]
    } else {
        &file_lines[start.saturating_sub(1)..end]
    };
    let (byte_start, byte_end) = if end < start {
        (0, 0)
    } else {
        line_range_to_bytes(content.as_bytes(), start, end)
    };
    let handle_token = if with_handle {
        Some(
            greppy_edit::EditHandle::for_range(
                &canonical_root,
                std::path::Path::new(&shown_path),
                content.as_bytes(),
                byte_start,
                byte_end,
            )?
            .encode(),
        )
    } else {
        None
    };
    if json {
        let rows = selected
            .iter()
            .enumerate()
            .map(|(offset, text)| serde_json::json!({"line": start + offset, "text": text}))
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": READ_FILE_JSON_SCHEMA_VERSION,
                "command": "read",
                "status": "ok",
                "path": shown_path,
                "start_line": start,
                "end_line": end,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "lines": rows,
                "handle": handle_token,
            }))
            .map_err(|error| Error::Invalid(format!("serialize read file JSON: {error}")))?
        );
    } else {
        // The header already states the span, so a per-line number repeats it
        // once per line. Measured over a 41-task benchmark that repetition was
        // 16.6% of everything `read` returned -- about 12,600 characters per
        // task, re-billed on every subsequent turn -- while edits are addressed
        // by content or by handle, not by line number.
        println!("{shown_path}:{start}-{end}");
        for text in selected {
            println!("{text}");
        }
        if let Some(token) = &handle_token {
            println!("handle: {token}");
        }
    }
    Ok(0)
}

/// `greppy read`: a symbol's exact definition span, optionally with an edit
/// handle. Resolution mirrors `brief`; the returned bytes come from the LIVE
/// file (the store addresses, the live file decides), so the handle's hashes
/// always describe what the agent actually saw.
pub(crate) fn dispatch_read(
    symbol: Option<&str>,
    with_handle: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    const READ_JSON_SCHEMA_VERSION: &str = "greppy.read.v1";
    let mut store = open_default_store_query_writer(root)?;
    let project = project_for(root)?;
    // Read is the workhorse of the edit loop, and the loop mutates files
    // constantly (test setup, the agent's own edits). Refusing on a stale
    // index — returning empty until a background reindex catches up — left
    // the agent with nothing and it degraded to bash (forensics 2026-07-18:
    // a real flask task took 123 turns, greppy all but unused). Instead,
    // heal in-band: a reindexable stale index is rebuilt BLOCKING and served
    // fresh on this same call. `read` verifies every span against the live
    // file anyway, so a brief blocking reindex is strictly better than an
    // empty answer. Only genuinely un-reindexable states (cold/failed) still
    // refuse.
    maybe_reindex_stale(&mut store, root)?;
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "read",
        json,
        serde_json::json!({"schema_version": READ_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    let ids = resolve_symbol_nodes(&store, symbol)?;
    let root_path = resolve_root(root)?;
    let mut nodes = Vec::new();
    for id in &ids {
        if let Some(node) = store.get_node(*id)? {
            if !node.file_path.is_empty() && node.start_line >= 1 {
                nodes.push(node);
            }
        }
    }
    // The per-file synthetic anchor answers to the file stem, so `greet` also
    // hits `pkg/greet.go` and a unique definition reads as ambiguous. A file is
    // read by path; the anchor only stands in when nothing else answered.
    if nodes
        .iter()
        .any(|node| !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name))
    {
        nodes.retain(|node| {
            !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name)
        });
    }
    if nodes.is_empty() {
        // Exact/graph resolution missed. greppy has an EMBEDDING engine
        // precisely so a reasonable-but-inexact reference still resolves —
        // "_startsWith", "impl Serialize for Bound", "the fn that validates
        // prefixes". Hard-failing here (as the old bare-name suggestion did)
        // wastes the second engine and forces the agent to guess formats
        // (trace forensics 2026-07-17: 12 read not-founds, 4-5 turns each).
        let query = symbol.unwrap_or("");
        let hits = greppy_search::semantic_query(&store, query, None, Some(&project), 6)
            .unwrap_or_default();
        // A clearly dominant hit is safe to read directly (read mutates
        // nothing); otherwise offer addressable candidates and let the agent
        // pick. Dominance = single hit, or top score >= 1.4x the runner-up.
        let dominant = match hits.as_slice() {
            [only] => Some(only.node.id),
            [top, second, ..] if top.score >= second.score * 1.4 => Some(top.node.id),
            _ => None,
        };
        if let Some(id) = dominant {
            if let Some(node) = store.get_node(id)? {
                if !node.file_path.is_empty() && node.start_line >= 1 {
                    if !json {
                        println!(
                            "read: `{query}` resolved semantically to `{}`",
                            node.qualified_name
                        );
                    }
                    nodes.push(node);
                }
            }
        }
        if nodes.is_empty() {
            // No single confident match: hand back addressable candidates —
            // the exact qualified name read accepts, plus location and kind —
            // so the retry is copy-paste, not another guess.
            let candidates: Vec<serde_json::Value> = hits
                .iter()
                .take(5)
                .map(|h| {
                    serde_json::json!({
                        "qualified_name": h.node.qualified_name,
                        "path": h.node.file_path,
                        "line": h.node.start_line,
                        "kind": h.node.label,
                    })
                })
                .collect();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "schema_version": READ_JSON_SCHEMA_VERSION,
                        "command": "read",
                        "status": "not-found",
                        "query": query,
                        "candidates": candidates,
                    }))
                    .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
                );
            } else if candidates.is_empty() {
                println!("read: no definition found for `{query}`");
            } else {
                println!("read: no exact match for `{query}`; closest definitions:");
                for h in hits.iter().take(5) {
                    println!(
                        "  {}  ({}:{}, {})",
                        h.node.qualified_name, h.node.file_path, h.node.start_line, h.node.label
                    );
                }
            }
            return Ok(10);
        }
    }
    if nodes.len() > 1 {
        // distinct definition sites -> ambiguous, list candidates (exit 11);
        // multiple store nodes on ONE site (Struct + Impl) are not ambiguity
        let mut sites: Vec<(String, i64)> = nodes
            .iter()
            .map(|n| (n.file_path.clone(), n.start_line))
            .collect();
        sites.sort();
        sites.dedup();
        if sites.len() > 1 {
            let candidates: Vec<serde_json::Value> = nodes
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "qualified_name": n.qualified_name,
                        "path": n.file_path,
                        "line": n.start_line,
                    })
                })
                .collect();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "schema_version": READ_JSON_SCHEMA_VERSION,
                        "command": "read",
                        "status": "ambiguous",
                        "query": symbol.unwrap_or(""),
                        "candidates": candidates,
                    }))
                    .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
                );
            } else {
                println!(
                    "read: `{}` is ambiguous; qualify it (Owner.method) or use one of:",
                    symbol.unwrap_or("")
                );
                for n in &nodes {
                    println!("  {} {}:{}", n.qualified_name, n.file_path, n.start_line);
                }
            }
            return Ok(11);
        }
    }
    let node = &nodes[0];
    let abs = root_path.join(&node.file_path);
    let content = std::fs::read(&abs).map_err(|source| Error::Io {
        context: format!("read {}", abs.display()),
        source,
    })?;
    // `--context N` widens the span upwards only: the doc comment sits above the
    // signature, and a definition's own end is where it ends.
    let start_line = (node.start_line - cli_read_context()).max(1);
    let Some(span) = read_span_with_meta(
        &root_path,
        &node.file_path,
        start_line,
        node.end_line,
        usize::MAX,
        false,
    ) else {
        println!(
            "read: definition span for `{}` is stale; re-index and retry",
            node.qualified_name
        );
        return Ok(12);
    };
    // line range -> byte range against the SAME live bytes
    let (byte_start, byte_end) =
        line_range_to_bytes(&content, start_line as usize, span.end_line as usize);
    let handle_token = if with_handle {
        let mut handle = greppy_edit::EditHandle::for_range(
            &root_path,
            std::path::Path::new(&node.file_path),
            &content,
            byte_start,
            byte_end,
        )?;
        let language = greppy_edit::language_for_path(std::path::Path::new(&node.file_path));
        handle.signature_fingerprint =
            greppy_edit::verbs::signature_fingerprint(language, &content, (byte_start, byte_end));
        handle.grammar_id = Some(format!("{language:?}"));
        handle.grammar_version = Some(env!("CARGO_PKG_VERSION").to_string());
        Some(handle.encode())
    } else {
        None
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": READ_JSON_SCHEMA_VERSION,
                "command": "read",
                "status": "ok",
                "qualified_name": node.qualified_name,
                "path": node.file_path,
                "start_line": node.start_line,
                "end_line": span.end_line,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "source": span.text,
                "handle": handle_token,
            }))
            .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
        );
    } else {
        println!(
            "{} {}:{}-{}",
            node.qualified_name, node.file_path, node.start_line, span.end_line
        );
        println!("{}", span.text);
        if let Some(token) = &handle_token {
            println!("handle: {token}");
        }
    }
    Ok(0)
}

/// `greppy edit`: dispatch to the transactional verbs; print the
/// certificate; map its status to the registered exit code.
/// Read an edit source argument: a file path, or `-` for stdin (agents
/// naturally try heredocs; K3 reasoning trace 2026-07-17: "Need pass new
/// source via stdin?").
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

pub(crate) fn read_plan(
    targets: &[String],
    symbol_opts: &[String],
    path_opts: &[String],
    lines: Option<&str>,
) -> Result<ReadPlan> {
    let mut positional = Vec::new();
    for value in targets {
        if value == "-" {
            positional.extend(targets_from_stdin()?);
            continue;
        }
        positional.push(value.clone());
    }
    let forced_symbol = !symbol_opts.is_empty();
    let mut subjects: Vec<String> = Vec::new();
    if forced_symbol {
        subjects.extend(symbol_opts.iter().cloned());
        subjects.extend(positional);
        // A single `--path` next to a single symbol is the historic
        // disambiguator (`read open --path FILE`), not another subject.
        if subjects.len() == 1 && path_opts.len() == 1 {
            if let Some(folded) = qualify_symbol_with_path(
                subjects.first().map(String::as_str),
                Some(path_opts[0].as_str()),
            ) {
                subjects[0] = folded;
            }
        } else {
            subjects.extend(path_opts.iter().cloned());
        }
    } else if positional.len() == 1 && path_opts.len() == 1 {
        if let Some(folded) = qualify_symbol_with_path(
            positional.first().map(String::as_str),
            Some(path_opts[0].as_str()),
        ) {
            subjects.push(folded);
        } else {
            subjects.extend(positional);
            subjects.extend(path_opts.iter().cloned());
        }
    } else {
        subjects.extend(positional);
        subjects.extend(path_opts.iter().cloned());
    }
    for subject in &subjects {
        if subject.trim().is_empty() {
            return Err(Error::Invalid(
                "empty read target: a symbol name or a path was expected (an unexpanded shell \
                 variable produces this). Nothing was read."
                    .into(),
            ));
        }
    }
    if lines.is_some() && subjects.len() > 1 {
        return Err(Error::Invalid(format!(
            "--lines names one range in one file, but {} targets were given; read them \
             separately, or drop --lines",
            subjects.len()
        )));
    }
    Ok(ReadPlan {
        subjects,
        forced_symbol,
        lines: lines.map(str::to_string),
    })
}

/// `read S [S …]` / `read PATH [PATH …]` — every target is read in one call,
/// and one target that cannot be read refuses the whole call: three files that
/// worked plus a silently dropped fourth looks like success.
pub(crate) fn dispatch_read_multi(
    plan: &ReadPlan,
    with_handle: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let canonical_root = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.clone());
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;

    enum Subject {
        File(String),
        Symbol(greppy_store::Node),
    }
    let mut subjects = Vec::with_capacity(plan.subjects.len());
    let mut missing: Vec<String> = Vec::new();
    for raw in &plan.subjects {
        // `file.rs::Symbol` is a qualified SYMBOL, not a slash-containing path.
        let path_qualified = split_path_qualified(raw).is_some();
        if !plan.forced_symbol && !path_qualified && read_subject_is_path(raw, root)? {
            subjects.push(Subject::File(raw.clone()));
            continue;
        }
        if !plan.forced_symbol && !path_qualified && looks_like_path(raw) {
            missing.push(format!("`{raw}` is not a file in this repository"));
            continue;
        }
        let ids = resolve_symbol_nodes(&store, Some(raw.as_str()))?;
        let mut node = None;
        for id in &ids {
            if let Some(candidate) = store.get_node(*id)? {
                if !candidate.file_path.is_empty() && candidate.start_line >= 1 {
                    node = Some(candidate);
                    break;
                }
            }
        }
        match node {
            Some(node) => subjects.push(Subject::Symbol(node)),
            None => missing.push(format!("`{raw}` is not a definition in this repository")),
        }
    }
    if !missing.is_empty() {
        return Err(Error::Invalid(format!(
            "read: {} — nothing was read for the other targets either.",
            missing.join("; ")
        )));
    }

    let mut hits = Vec::with_capacity(subjects.len());
    let mut text = String::new();
    for (index, subject) in subjects.iter().enumerate() {
        let (shown_path, start, end, source, content) = match subject {
            Subject::File(raw) => {
                let candidate = read_file_candidate(&root_path, raw);
                let canonical = candidate.canonicalize().map_err(|source| Error::Io {
                    context: format!("canonicalize {}", candidate.display()),
                    source,
                })?;
                let relative = canonical.strip_prefix(&canonical_root).map_err(|_| {
                    Error::Invalid(format!("read path `{raw}` resolves outside the workspace"))
                })?;
                let shown = relative.to_string_lossy().replace('\\', "/");
                let content = std::fs::read(&canonical).map_err(|source| Error::Io {
                    context: format!("read {}", canonical.display()),
                    source,
                })?;
                let body = String::from_utf8_lossy(&content).into_owned();
                let lines = body.lines().count().max(1);
                (shown, 1i64, lines as i64, body, content)
            }
            Subject::Symbol(node) => {
                let absolute = root_path.join(&node.file_path);
                let content = std::fs::read(&absolute).map_err(|source| Error::Io {
                    context: format!("read {}", absolute.display()),
                    source,
                })?;
                let Some(span) = read_span_with_meta(
                    &root_path,
                    &node.file_path,
                    node.start_line,
                    node.end_line,
                    usize::MAX,
                    false,
                ) else {
                    return Err(Error::Invalid(format!(
                        "read: definition span for `{}` is stale; re-index and retry",
                        node.qualified_name
                    )));
                };
                (
                    node.file_path.clone(),
                    node.start_line,
                    span.end_line,
                    span.text,
                    content,
                )
            }
        };
        let (byte_start, byte_end) =
            line_range_to_bytes(&content, start.max(1) as usize, end.max(start) as usize);
        let handle_token = if with_handle {
            Some(
                greppy_edit::EditHandle::for_range(
                    &root_path,
                    std::path::Path::new(&shown_path),
                    &content,
                    byte_start,
                    byte_end,
                )?
                .encode(),
            )
        } else {
            None
        };
        let target = &plan.subjects[index.min(plan.subjects.len() - 1)];
        hits.push(serde_json::json!({
            "target": target,
            "qualified_name": match subject {
                Subject::Symbol(node) => node.qualified_name.clone(),
                Subject::File(_) => shown_path.clone(),
            },
            "file": &shown_path,
            "line": start,
            "path": &shown_path,
            "file_path": &shown_path,
            "start_line": start,
            "end_line": end,
            "lines": format!("{start}:{end}"),
            "source": &source,
            "handle": handle_token,
        }));
        text.push_str(&format!("{shown_path}:{start}-{end}\n"));
        text.push_str(&source);
        if !source.ends_with('\n') {
            text.push('\n');
        }
        if let Some(token) = &handle_token {
            text.push_str(&format!("handle: {token}\n"));
        }
    }
    if json {
        let value = serde_json::json!({
            "schema_version": "greppy.read.v1",
            "command": "read",
            "status": "ok",
            "total_exact": hits.len(),
            "shown": hits.len(),
            "hits": hits,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
        );
    } else {
        print!("{text}");
    }
    Ok(0)
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

pub(crate) fn read_uses_default_file_budget(cli: &Cli) -> bool {
    let Some(Command::Read {
        targets,
        symbol_opts,
        path_opts,
        lines,
        json,
        all,
        ..
    }) = cli.command.as_ref()
    else {
        return false;
    };
    if *json || *all || lines.is_some() || !symbol_opts.is_empty() {
        return false;
    }
    if targets.len() + path_opts.len() != 1 {
        return false;
    }
    if targets.is_empty() {
        return true;
    }
    targets.first().is_some_and(|subject| {
        split_path_qualified(subject).is_none()
            && read_subject_is_path(subject, cli.root.as_deref()).unwrap_or(false)
    })
}
