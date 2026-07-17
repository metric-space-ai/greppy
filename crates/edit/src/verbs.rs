//! Edit verbs. Every verb compiles to the same transaction pipeline and
//! emits a `greppy.edit-certificate.v1` document; the certificate is the
//! only success signal an agent needs.

use std::path::Path;

use crate::certificate::{
    Candidate, Certificate, Guarantee, Guarantees, OperationReport, PublishMode, SelectorClass,
    SelectorEngine, Status, SyntaxDelta, WorkspaceReport,
};
use crate::handle::EditHandle;
use crate::hash::sha256_hex;
use crate::publish::publish_atomic;
use crate::txn::{apply_in_memory, outside_ranges_unchanged, syntax_counts, PlannedOp, Snapshot};
use greppy_core::Result;
use greppy_parser::Language;

/// Common options for single-operation verbs.
#[derive(Debug, Clone)]
pub struct VerbOptions {
    pub dry_run: bool,
    /// Emit the unified diff inline in the certificate.
    pub with_diff: bool,
}

impl Default for VerbOptions {
    fn default() -> Self {
        Self {
            dry_run: false,
            with_diff: true,
        }
    }
}

/// `greppy edit text-cas`: exact-once (or `--expect N`) text replacement,
/// hash-gated, no regex, no fuzz.
pub fn text_cas(
    workspace_root: &Path,
    file: &Path,
    old_text: &[u8],
    new_text: &[u8],
    expect: usize,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let occurrences = find_all(&snapshot.content, old_text);

    if old_text == new_text
        || (occurrences.is_empty() && find_all(&snapshot.content, new_text).len() == expect)
    {
        // the requested end state already holds
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Text,
            SelectorClass::ExactText,
            Status::AlreadySatisfied,
            occurrences.len(),
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    if occurrences.is_empty() {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Text,
            SelectorClass::ExactText,
            Status::NotFound,
            0,
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    if occurrences.len() != expect {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Text,
            SelectorClass::ExactText,
            Status::Ambiguous,
            occurrences.len(),
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }

    let ops: Vec<PlannedOp> = occurrences
        .iter()
        .enumerate()
        .map(|(i, &start)| PlannedOp {
            id: format!("text-cas-{i}"),
            range: (start, start + old_text.len()),
            replacement: new_text.to_vec(),
        })
        .collect();
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Text,
        SelectorClass::ExactText,
        None,
        options,
    )
}

/// `greppy edit replace-span --target HANDLE`: replace exactly the span the
/// agent previously read, CAS-guarded by the handle.
pub fn replace_span(
    workspace_root: &Path,
    handle: &EditHandle,
    new_content: &[u8],
    language: Option<Language>,
    options: &VerbOptions,
) -> Result<Certificate> {
    let file = workspace_root.join(&handle.path);
    let file = if Path::new(&handle.path).is_absolute() {
        Path::new(&handle.path).to_path_buf()
    } else {
        file
    };
    let snapshot = Snapshot::read(&file)?;

    let (start, end) = match handle.verify(&snapshot.content) {
        Ok(range) => range,
        Err(_) => {
            return Ok(single_op_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::Symbol,
                SelectorClass::Resolved,
                Status::Stale,
                0,
                &[],
                None,
                None,
                options,
                PublishMode::Atomic,
            ));
        }
    };
    if &snapshot.content[start..end] == new_content {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::AlreadySatisfied,
            1,
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    let ops = vec![PlannedOp {
        id: "replace-span".into(),
        range: (start, end),
        replacement: new_content.to_vec(),
    }];
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
        language,
        options,
    )
}

/// `greppy edit patch-span --target HANDLE --patch-file F`: apply a unified
/// diff to exactly the span the agent previously read. fuzz 0: every hunk's
/// context must match byte-for-byte inside the target; anything else refuses
/// without writing.
pub fn patch_span(
    workspace_root: &Path,
    handle: &crate::handle::EditHandle,
    patch_text: &[u8],
    language: Option<Language>,
    options: &VerbOptions,
) -> Result<Certificate> {
    let file = if Path::new(&handle.path).is_absolute() {
        Path::new(&handle.path).to_path_buf()
    } else {
        workspace_root.join(&handle.path)
    };
    let snapshot = Snapshot::read(&file)?;
    let (start, end) = match handle.verify(&snapshot.content) {
        Ok(range) => range,
        Err(_) => {
            return Ok(single_refusal_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::Symbol,
                SelectorClass::Resolved,
                Status::Stale,
                options,
            ));
        }
    };
    let target = &snapshot.content[start..end];
    let Some(new_target) = apply_unified_patch_exact(target, patch_text) else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::InvalidResult,
            options,
        ));
    };
    if new_target == target {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::AlreadySatisfied,
            options,
        ));
    }
    let ops = vec![PlannedOp {
        id: "patch-span".into(),
        range: (start, end),
        replacement: new_target,
    }];
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
        language,
        options,
    )
}

/// Apply a unified diff to `base` with fuzz 0. Line-based; every hunk's
/// context and removals must match exactly at the stated positions. Returns
/// None on any mismatch (including malformed hunks).
fn apply_unified_patch_exact(base: &[u8], patch: &[u8]) -> Option<Vec<u8>> {
    let base_str = std::str::from_utf8(base).ok()?;
    let patch_str = std::str::from_utf8(patch).ok()?;
    let base_lines: Vec<&str> = base_str.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize; // next unconsumed base line
    let mut lines = patch_str.lines().peekable();
    let mut saw_hunk = false;
    while let Some(line) = lines.next() {
        if line.starts_with("--- ") || line.starts_with("+++ ") {
            continue;
        }
        if let Some(header) = line.strip_prefix("@@") {
            saw_hunk = true;
            // "@@ -l,c +l,c @@"
            let minus = header.split_whitespace().find(|t| t.starts_with('-'))?;
            let old_start: usize = minus[1..].split(',').next()?.parse().ok()?;
            let old_start = old_start.saturating_sub(1); // 0-based
            if old_start < cursor || old_start > base_lines.len() {
                return None;
            }
            out.extend(base_lines[cursor..old_start].iter().map(|l| l.to_string()));
            cursor = old_start;
            while let Some(&next) = lines.peek() {
                if next.starts_with("@@") || next.starts_with("--- ") {
                    break;
                }
                let next = lines.next()?;
                match next.chars().next() {
                    Some(' ') => {
                        if base_lines.get(cursor) != Some(&&next[1..]) {
                            return None;
                        }
                        out.push(next[1..].to_string());
                        cursor += 1;
                    }
                    Some('-') => {
                        if base_lines.get(cursor) != Some(&&next[1..]) {
                            return None;
                        }
                        cursor += 1;
                    }
                    Some('+') => out.push(next[1..].to_string()),
                    Some('\\') => {} // "\ No newline at end of file"
                    _ => return None,
                }
            }
        }
    }
    if !saw_hunk {
        return None;
    }
    out.extend(base_lines[cursor..].iter().map(|l| l.to_string()));
    let mut bytes = out.join("\n").into_bytes();
    if base.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    Some(bytes)
}

/// `greppy edit regex-cas`: regex replacement with exact expected match
/// count. Accepted, but reported as the weakest selector class.
pub fn regex_cas(
    workspace_root: &Path,
    file: &Path,
    pattern: &str,
    replacement: &str,
    expect: usize,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let re = regex::Regex::new(pattern)
        .map_err(|e| greppy_core::Error::Invalid(format!("invalid regex: {e}")))?;
    let text = String::from_utf8_lossy(&snapshot.content).into_owned();
    let matches: Vec<(usize, usize)> = re.find_iter(&text).map(|m| (m.start(), m.end())).collect();
    if matches.is_empty() {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Regex,
            SelectorClass::RegexWeak,
            Status::NotFound,
            options,
        ));
    }
    if matches.len() != expect {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Regex,
            SelectorClass::RegexWeak,
            Status::Ambiguous,
            options,
        ));
    }
    let ops: Vec<PlannedOp> = matches
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let expanded = re
                .replace(&text[start..end], replacement)
                .into_owned()
                .into_bytes();
            PlannedOp {
                id: format!("regex-cas-{i}"),
                range: (start, end),
                replacement: expanded,
            }
        })
        .collect();
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Regex,
        SelectorClass::RegexWeak,
        None,
        options,
    )
}

/// Where to place inserted text relative to a definition span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertPosition {
    Before,
    After,
}

/// `greppy edit replace-body --symbol SYM`: replace only the BODY of the
/// definition at `def_range`, located via tree-sitter (`body` field of the
/// smallest definition node covering the span). The signature stays
/// byte-identical.
pub fn replace_body(
    workspace_root: &Path,
    file: &Path,
    def_range: (usize, usize),
    new_body: &[u8],
    language: Language,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let Some(body_range) = body_range_within(language, &snapshot.content, def_range) else {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::NotFound,
            0,
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    };
    if &snapshot.content[body_range.0..body_range.1] == new_body {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::AlreadySatisfied,
            1,
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    let ops = vec![PlannedOp {
        id: "replace-body".into(),
        range: body_range,
        replacement: new_body.to_vec(),
    }];
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
        Some(language),
        options,
    )
}

/// `greppy edit insert-after/-before --symbol SYM`: insert a new top-level
/// block adjacent to the definition span, separated by a blank line.
pub fn insert_adjacent(
    workspace_root: &Path,
    file: &Path,
    def_range: (usize, usize),
    text: &[u8],
    position: InsertPosition,
    language: Option<Language>,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let mut block = Vec::new();
    let at = match position {
        InsertPosition::Before => {
            block.extend_from_slice(text);
            if !text.ends_with(b"\n") {
                block.push(b'\n');
            }
            block.push(b'\n');
            def_range.0
        }
        InsertPosition::After => {
            // insert after the trailing newline of the definition when present
            let mut at = def_range.1;
            if snapshot.content.get(at) == Some(&b'\n') {
                at += 1;
            }
            block.push(b'\n');
            block.extend_from_slice(text);
            if !text.ends_with(b"\n") {
                block.push(b'\n');
            }
            at
        }
    };
    let ops = vec![PlannedOp {
        id: format!(
            "insert-{}",
            if position == InsertPosition::Before {
                "before"
            } else {
                "after"
            }
        ),
        range: (at, at),
        replacement: block,
    }];
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
        language,
        options,
    )
}

/// `greppy edit delete --symbol SYM`: remove the definition span including
/// one trailing newline.
pub fn delete_span(
    workspace_root: &Path,
    file: &Path,
    def_range: (usize, usize),
    language: Option<Language>,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let mut end = def_range.1;
    if snapshot.content.get(end) == Some(&b'\n') {
        end += 1;
    }
    // also swallow ONE preceding blank line so deletions do not accumulate
    // double blank lines between the neighbours
    let mut start = def_range.0;
    if start >= 1 && snapshot.content.get(start - 1) == Some(&b'\n') {
        if start >= 2 && snapshot.content.get(start - 2) == Some(&b'\n') {
            start -= 1;
        }
    }
    let ops = vec![PlannedOp {
        id: "delete".into(),
        range: (start, end),
        replacement: Vec::new(),
    }];
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
        language,
        options,
    )
}

/// `greppy edit rename-call --in SYM --from A --to B`: retarget identifier
/// occurrences of `from` inside one definition span. AST-based: only
/// identifier-kind nodes are renamed, so strings and comments are never
/// touched. `expect`: None = all occurrences (at least one), Some(n) =
/// exactly n or refuse.
pub fn rename_in_span(
    workspace_root: &Path,
    file: &Path,
    def_range: (usize, usize),
    from: &str,
    to: &str,
    expect: Option<usize>,
    language: Language,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let sites = identifier_sites(language, &snapshot.content, def_range, from.as_bytes());
    let Some(sites) = sites else {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            0,
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    };
    if from == to {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::AlreadySatisfied,
            sites.len(),
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    if sites.is_empty() {
        // idempotency: if `to` already appears where `from` is gone, report
        // already-satisfied instead of not-found
        let to_sites = identifier_sites(language, &snapshot.content, def_range, to.as_bytes())
            .unwrap_or_default();
        let status = if to_sites.is_empty() {
            Status::NotFound
        } else {
            Status::AlreadySatisfied
        };
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            status,
            0,
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    if let Some(n) = expect {
        if sites.len() != n {
            return Ok(single_op_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::TreeSitter,
                SelectorClass::Structural,
                Status::Ambiguous,
                sites.len(),
                &[],
                None,
                None,
                options,
                PublishMode::Atomic,
            ));
        }
    }
    let ops: Vec<PlannedOp> = sites
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| PlannedOp {
            id: format!("rename-{i}"),
            range: (start, end),
            replacement: to.as_bytes().to_vec(),
        })
        .collect();
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::TreeSitter,
        SelectorClass::Structural,
        Some(language),
        options,
    )
}

/// Byte ranges of identifier-kind leaf nodes whose text equals `name`,
/// restricted to `def_range`. None when the language cannot be parsed.
fn identifier_sites(
    language: Language,
    content: &[u8],
    def_range: (usize, usize),
    name: &[u8],
) -> Option<Vec<(usize, usize)>> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let mut out = Vec::new();
    let mut cursor = tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = cursor.node();
        if node.start_byte() >= def_range.0
            && node.end_byte() <= def_range.1
            && node.child_count() == 0
            && node.kind().contains("identifier")
            && &content[node.start_byte()..node.end_byte()] == name
        {
            out.push((node.start_byte(), node.end_byte()));
        }
        if node.end_byte() >= def_range.0 && node.start_byte() <= def_range.1 {
            if cursor.goto_first_child() {
                continue;
            }
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                reached_root = true;
                break;
            }
        }
    }
    Some(out)
}

/// Locate the byte range of the `body` field of the smallest named node
/// covering `def_range`. None when the language has no tree-sitter grammar
/// or the node has no body field (e.g. a struct without one).
fn body_range_within(
    language: Language,
    content: &[u8],
    def_range: (usize, usize),
) -> Option<(usize, usize)> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let mut node = tree
        .root_node()
        .descendant_for_byte_range(def_range.0, def_range.1.saturating_sub(1))?;
    loop {
        if let Some(body) = node.child_by_field_name("body") {
            // only accept a body that lies inside the addressed definition
            if body.start_byte() >= def_range.0 && body.end_byte() <= def_range.1 {
                // extend back to the start of the line when only indentation
                // precedes the body: agents supply fully indented bodies, and
                // replacing from mid-line would double the first line's indent
                let mut start = body.start_byte();
                let line_start = content[..start]
                    .iter()
                    .rposition(|&b| b == b'\n')
                    .map(|i| i + 1)
                    .unwrap_or(0);
                if content[line_start..start]
                    .iter()
                    .all(|b| b.is_ascii_whitespace())
                {
                    start = line_start;
                }
                return Some((start, body.end_byte()));
            }
        }
        node = node.parent()?;
        if node.start_byte() < def_range.0.saturating_sub(1) {
            return None;
        }
    }
}

/// Public pipeline entry for sibling modules (`ensure`, future engines).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pipeline_public(
    workspace_root: &Path,
    snapshot: Snapshot,
    ops: Vec<PlannedOp>,
    engine: SelectorEngine,
    class: SelectorClass,
    language: Option<Language>,
    options: &VerbOptions,
) -> Result<Certificate> {
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        engine,
        class,
        language,
        options,
    )
}

/// Public refusal certificate for sibling modules: a status-only report
/// with no candidates and no node text.
pub(crate) fn single_refusal_certificate(
    workspace_root: &Path,
    snapshot: &Snapshot,
    engine: SelectorEngine,
    class: SelectorClass,
    status: Status,
    options: &VerbOptions,
) -> Certificate {
    single_op_certificate(
        workspace_root,
        snapshot,
        engine,
        class,
        status,
        0,
        &[],
        None,
        None,
        options,
        PublishMode::Atomic,
    )
}

/// The shared transaction pipeline for single-file verbs.
fn run_pipeline(
    workspace_root: &Path,
    snapshot: Snapshot,
    ops: Vec<PlannedOp>,
    engine: SelectorEngine,
    class: SelectorClass,
    language: Option<Language>,
    options: &VerbOptions,
) -> Result<Certificate> {
    let syntax_before = language.and_then(|l| syntax_counts(l, &snapshot.content));
    let applied = apply_in_memory(&snapshot, &ops)?;
    let syntax_after = language.and_then(|l| syntax_counts(l, &applied.content));

    let syntax = match (syntax_before, syntax_after) {
        (Some(b), Some(a)) => SyntaxDelta {
            errors_before: b.errors,
            errors_after: a.errors,
            new_errors: a.errors.saturating_sub(b.errors),
            new_missing_nodes: a.missing.saturating_sub(b.missing),
        },
        _ => SyntaxDelta {
            errors_before: 0,
            errors_after: 0,
            new_errors: 0,
            new_missing_nodes: 0,
        },
    };
    let syntax_applicable = syntax_before.is_some() && syntax_after.is_some();
    let syntax_ok = !syntax_applicable || (syntax.new_errors == 0 && syntax.new_missing_nodes == 0);
    let isolation_ok = outside_ranges_unchanged(&snapshot.content, &applied.content, &ops);

    let target_before = ops
        .first()
        .map(|op| sha256_hex(&snapshot.content[op.range.0..op.range.1]))
        .unwrap_or_default();
    let node_before = ops
        .first()
        .and_then(|op| String::from_utf8(snapshot.content[op.range.0..op.range.1].to_vec()).ok());
    let node_after = ops
        .first()
        .and_then(|op| String::from_utf8(op.replacement.clone()).ok());

    let mut status = if !syntax_ok || !isolation_ok {
        Status::InvalidResult
    } else {
        Status::Applied
    };

    let mut published = false;
    let mut file_sha_after = None;
    if status == Status::Applied && !options.dry_run {
        match publish_atomic(
            workspace_root,
            &snapshot.path,
            &applied.content,
            &snapshot.file_sha256,
        ) {
            Ok(sha) => {
                published = true;
                file_sha_after = Some(sha);
            }
            Err(e) => {
                status = if format!("{e}").contains("stale") {
                    Status::Stale
                } else {
                    Status::PublishFailed
                };
            }
        }
    } else if status == Status::Applied {
        file_sha_after = Some(applied.file_sha256.clone());
    }

    let diff = if options.with_diff {
        Some(unified_diff(
            &snapshot.path.to_string_lossy(),
            &snapshot.content,
            &applied.content,
        ))
    } else {
        None
    };

    let op_report = OperationReport {
        id: ops.first().map(|o| o.id.clone()).unwrap_or_default(),
        file: snapshot.path.to_string_lossy().into_owned(),
        selector_engine: engine,
        selector_class: class,
        scope_matches: 1,
        target_matches: ops.len(),
        file_sha256_before: snapshot.file_sha256.clone(),
        file_sha256_after: file_sha_after,
        target_sha256_before: target_before,
        target_sha256_after: ops.first().map(|op| sha256_hex(&op.replacement)),
        outside_declared_ranges_unchanged: isolation_ok,
        changed_byte_ranges: applied.changed_ranges.clone(),
        node_before,
        node_after,
        unified_diff: diff,
        syntax,
        postconditions_passed: syntax_ok && isolation_ok,
        postconditions: vec![],
        guarantees: Guarantees {
            addressed_range: Guarantee::Proved,
            no_clobber: if published || options.dry_run {
                Guarantee::Proved
            } else {
                Guarantee::Failed
            },
            byte_isolation: if isolation_ok {
                Guarantee::Proved
            } else {
                Guarantee::Failed
            },
            syntax: if !syntax_applicable {
                Guarantee::NotApplicable
            } else if syntax_ok {
                Guarantee::Proved
            } else {
                Guarantee::Failed
            },
            validators: Guarantee::NotApplicable,
        },
        formatter_expanded_change_scope: false,
        store_refreshed: false,
        candidates: vec![],
    };

    Ok(Certificate {
        schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
        status,
        transaction_id: transaction_id(&snapshot.file_sha256, &applied.file_sha256),
        workspace: WorkspaceReport {
            root: workspace_root.to_string_lossy().into_owned(),
            git_head_before: None,
            git_head_after: None,
        },
        operations: vec![op_report],
        validators: vec![],
        published,
        publish_mode: if options.dry_run {
            PublishMode::DryRun
        } else {
            PublishMode::Atomic
        },
    })
}

#[allow(clippy::too_many_arguments)] // report assembly: splitting into a builder hides which fields a status requires
fn single_op_certificate(
    workspace_root: &Path,
    snapshot: &Snapshot,
    engine: SelectorEngine,
    class: SelectorClass,
    status: Status,
    matches: usize,
    candidates: &[Candidate],
    node_before: Option<String>,
    node_after: Option<String>,
    options: &VerbOptions,
    mode: PublishMode,
) -> Certificate {
    Certificate {
        schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
        status,
        transaction_id: transaction_id(&snapshot.file_sha256, "none"),
        workspace: WorkspaceReport {
            root: workspace_root.to_string_lossy().into_owned(),
            git_head_before: None,
            git_head_after: None,
        },
        operations: vec![OperationReport {
            id: "op-0".into(),
            file: snapshot.path.to_string_lossy().into_owned(),
            selector_engine: engine,
            selector_class: class,
            scope_matches: 1,
            target_matches: matches,
            file_sha256_before: snapshot.file_sha256.clone(),
            file_sha256_after: None,
            target_sha256_before: String::new(),
            target_sha256_after: None,
            outside_declared_ranges_unchanged: true,
            changed_byte_ranges: vec![],
            node_before,
            node_after,
            unified_diff: None,
            syntax: SyntaxDelta {
                errors_before: 0,
                errors_after: 0,
                new_errors: 0,
                new_missing_nodes: 0,
            },
            postconditions_passed: status == Status::AlreadySatisfied,
            postconditions: vec![],
            guarantees: Guarantees {
                addressed_range: Guarantee::NotApplicable,
                no_clobber: Guarantee::Proved,
                byte_isolation: Guarantee::Proved,
                syntax: Guarantee::NotApplicable,
                validators: Guarantee::NotApplicable,
            },
            formatter_expanded_change_scope: false,
            store_refreshed: false,
            candidates: candidates.to_vec(),
        }],
        validators: vec![],
        published: false,
        publish_mode: if options.dry_run {
            PublishMode::DryRun
        } else {
            mode
        },
    }
}

fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() {
        return vec![];
    }
    let mut out = vec![];
    let mut from = 0usize;
    while from + needle.len() <= haystack.len() {
        match haystack[from..]
            .windows(needle.len())
            .position(|w| w == needle)
        {
            Some(rel) => {
                out.push(from + rel);
                from = from + rel + needle.len();
            }
            None => break,
        }
    }
    out
}

fn transaction_id(before: &str, after: &str) -> String {
    format!(
        "ge-{}",
        &sha256_hex(format!("{before}:{after}").as_bytes())[..16]
    )
}

/// Minimal unified diff (line-based) for the certificate. Precise byte
/// ranges are reported separately; this is the human/agent-readable view.
fn unified_diff(path: &str, before: &[u8], after: &[u8]) -> String {
    let before = String::from_utf8_lossy(before);
    let after = String::from_utf8_lossy(after);
    let b: Vec<&str> = before.lines().collect();
    let a: Vec<&str> = after.lines().collect();
    // trim common prefix/suffix, emit one hunk
    let mut start = 0usize;
    while start < b.len() && start < a.len() && b[start] == a[start] {
        start += 1;
    }
    let mut bend = b.len();
    let mut aend = a.len();
    while bend > start && aend > start && b[bend - 1] == a[aend - 1] {
        bend -= 1;
        aend -= 1;
    }
    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        start + 1,
        bend - start,
        start + 1,
        aend - start
    ));
    for line in &b[start..bend] {
        out.push_str(&format!("-{line}\n"));
    }
    for line in &a[start..aend] {
        out.push_str(&format!("+{line}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn text_cas_applies_exactly_once() {
        let dir = ws();
        let f = dir.path().join("conf.ini");
        std::fs::write(&f, b"port = 9000\nhost = x\n").unwrap();
        let cert = text_cas(
            dir.path(),
            &f,
            b"port = 9000",
            b"port = 8080",
            1,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(cert.published);
        assert!(cert.operations[0].outside_declared_ranges_unchanged);
        assert_eq!(std::fs::read(&f).unwrap(), b"port = 8080\nhost = x\n");
        assert_eq!(cert.exit_code(), 0);
    }

    #[test]
    fn text_cas_ambiguous_changes_nothing() {
        let dir = ws();
        let f = dir.path().join("conf.ini");
        std::fs::write(&f, b"x = 1\nx = 1\n").unwrap();
        let cert = text_cas(
            dir.path(),
            &f,
            b"x = 1",
            b"x = 2",
            1,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Ambiguous);
        assert_eq!(cert.exit_code(), 11);
        assert_eq!(std::fs::read(&f).unwrap(), b"x = 1\nx = 1\n");
    }

    #[test]
    fn text_cas_not_found() {
        let dir = ws();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, b"hello\n").unwrap();
        let cert = text_cas(dir.path(), &f, b"missing", b"y", 1, &VerbOptions::default()).unwrap();
        assert_eq!(cert.status, Status::NotFound);
        assert_eq!(cert.exit_code(), 10);
    }

    #[test]
    fn text_cas_idempotent_second_run() {
        let dir = ws();
        let f = dir.path().join("conf.ini");
        std::fs::write(&f, b"port = 8080\n").unwrap();
        let cert = text_cas(
            dir.path(),
            &f,
            b"port = 9000",
            b"port = 8080",
            1,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
        assert_eq!(cert.exit_code(), 0);
    }

    #[test]
    fn replace_span_roundtrip_with_handle() {
        let dir = ws();
        let f = dir.path().join("m.rs");
        let content = b"fn a() {}\nfn b() { old(); }\n";
        std::fs::write(&f, content).unwrap();
        let start = 10usize; // "fn b() { old(); }"
        let end = content.len() - 1;
        let handle =
            EditHandle::for_range(dir.path(), Path::new("m.rs"), content, start, end).unwrap();
        let cert = replace_span(
            dir.path(),
            &handle,
            b"fn b() { new(); }",
            Some(Language::Rust),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert_eq!(cert.operations[0].syntax.new_errors, 0);
        assert_eq!(
            std::fs::read(&f).unwrap(),
            b"fn a() {}\nfn b() { new(); }\n"
        );
    }

    #[test]
    fn replace_span_stale_handle_is_exit_12() {
        let dir = ws();
        let f = dir.path().join("m.rs");
        std::fs::write(&f, b"fn a() {}\n").unwrap();
        let handle =
            EditHandle::for_range(dir.path(), Path::new("m.rs"), b"fn a() {}\n", 0, 9).unwrap();
        std::fs::write(&f, b"fn a() { changed(); }\n").unwrap();
        let cert = replace_span(
            dir.path(),
            &handle,
            b"fn a() { x(); }",
            Some(Language::Rust),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Stale);
        assert_eq!(cert.exit_code(), 12);
        assert_eq!(std::fs::read(&f).unwrap(), b"fn a() { changed(); }\n");
    }

    #[test]
    fn patch_span_exact_apply_and_refusal() {
        let dir = ws();
        let f = dir.path().join("m.py");
        let content = b"def f():\n    a = 1\n    b = 2\n    return a + b\n";
        std::fs::write(&f, content).unwrap();
        let h = EditHandle::for_range(dir.path(), Path::new("m.py"), content, 0, content.len())
            .unwrap();
        let patch = b"@@ -2,2 +2,2 @@\n     a = 1\n-    b = 2\n+    b = 3\n";
        let cert = patch_span(
            dir.path(),
            &h,
            patch,
            Some(Language::Python),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(std::fs::read_to_string(&f).unwrap().contains("b = 3"));
        // ein Patch mit falschem Kontext: Refusal ohne Schreiben
        std::fs::write(&f, content).unwrap();
        let h2 = EditHandle::for_range(dir.path(), Path::new("m.py"), content, 0, content.len())
            .unwrap();
        let bad = b"@@ -2,2 +2,2 @@\n     a = 999\n-    b = 2\n+    b = 3\n";
        let cert = patch_span(
            dir.path(),
            &h2,
            bad,
            Some(Language::Python),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::InvalidResult);
        assert!(std::fs::read_to_string(&f).unwrap().contains("b = 2"));
    }

    #[test]
    fn regex_cas_expect_gates_count() {
        let dir = ws();
        let f = dir.path().join("conf.ini");
        std::fs::write(
            &f,
            b"port = 9000
timeout = 30
",
        )
        .unwrap();
        let cert = regex_cas(
            dir.path(),
            &f,
            r"^port\s*=\s*\d+",
            "port = 8080",
            1,
            &VerbOptions::default(),
        )
        .unwrap();
        // ^ ohne multiline matcht nur zeile 1 -> genau 1 treffer
        assert_eq!(cert.status, Status::Applied);
        assert!(std::fs::read_to_string(&f)
            .unwrap()
            .starts_with("port = 8080"));
        assert_eq!(
            cert.operations[0].selector_class,
            crate::certificate::SelectorClass::RegexWeak
        );
    }

    #[test]
    fn rename_call_renames_only_identifiers() {
        let dir = ws();
        let f = dir.path().join("m.py");
        let content =
            b"def run():\n    legacy_auth()\n    print(\"legacy_auth\")\n    x = legacy_auth\n";
        std::fs::write(&f, content).unwrap();
        let cert = rename_in_span(
            dir.path(),
            &f,
            (0, content.len()),
            "legacy_auth",
            "validate",
            None,
            Language::Python,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("validate()"), "{out}");
        // string literal untouched
        assert!(out.contains("print(\"legacy_auth\")"), "{out}");
        assert!(out.contains("x = validate"), "{out}");
    }

    #[test]
    fn rename_call_expect_mismatch_refuses() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"def run():\n    a()\n    a()\n").unwrap();
        let cert = rename_in_span(
            dir.path(),
            &f,
            (0, 24),
            "a",
            "b",
            Some(1),
            Language::Python,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Ambiguous);
        assert!(std::fs::read_to_string(&f).unwrap().contains("a()"));
    }

    #[test]
    fn rename_call_idempotent_second_run() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"def run():\n    b()\n").unwrap();
        let cert = rename_in_span(
            dir.path(),
            &f,
            (0, 19),
            "a",
            "b",
            None,
            Language::Python,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
    }

    #[test]
    fn replace_body_keeps_signature() {
        let dir = ws();
        let f = dir.path().join("m.py");
        let content = b"def add(a, b):\n    return a + b\n";
        std::fs::write(&f, content).unwrap();
        let cert = replace_body(
            dir.path(),
            &f,
            (0, content.len() - 1),
            b"    return b + a",
            Language::Python,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.starts_with("def add(a, b):"), "{out}");
        assert!(out.contains("return b + a"), "{out}");
    }

    #[test]
    fn insert_after_appends_block() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"def a():\n    return 1\n").unwrap();
        let cert = insert_adjacent(
            dir.path(),
            &f,
            (0, 21),
            b"def b():\n    return 2",
            InsertPosition::After,
            Some(Language::Python),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("def a():"), "{out}");
        assert!(out.contains("\n\ndef b():"), "{out}");
        assert_eq!(cert.operations[0].syntax.new_errors, 0);
    }

    #[test]
    fn delete_removes_definition_and_blank_line() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"def a():\n    return 1\n\ndef b():\n    return 2\n").unwrap();
        let cert = delete_span(
            dir.path(),
            &f,
            (23, 45),
            Some(Language::Python),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(!out.contains("def b"), "{out}");
        assert!(!out.contains("\n\n\n"), "{out}");
    }

    #[test]
    fn syntax_breaking_edit_is_rejected_and_not_published() {
        let dir = ws();
        let f = dir.path().join("m.rs");
        let content = b"fn a() {}\n";
        std::fs::write(&f, content).unwrap();
        let handle = EditHandle::for_range(dir.path(), Path::new("m.rs"), content, 0, 9).unwrap();
        let cert = replace_span(
            dir.path(),
            &handle,
            b"fn a( {", // broken
            Some(Language::Rust),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::InvalidResult);
        assert_eq!(cert.exit_code(), 13);
        assert!(!cert.published);
        assert_eq!(std::fs::read(&f).unwrap(), content);
    }
}
