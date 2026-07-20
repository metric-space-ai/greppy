//! Edit verbs. Every verb compiles to the same transaction pipeline and
//! emits a `greppy.edit-certificate.v1` document; the certificate is the
//! only success signal an agent needs.

use std::path::Path;

use crate::certificate::{
    Candidate, Certificate, Guarantee, Guarantees, OperationReport, PostconditionResult,
    PublishMode, SelectorClass, SelectorEngine, Status, SyntaxDelta, WorkspaceReport,
};
use crate::handle::EditHandle;
use crate::hash::sha256_hex;
use crate::publish::publish_atomic;
use crate::txn::{apply_in_memory, outside_ranges_unchanged, syntax_counts, PlannedOp, Snapshot};
use greppy_core::Result;
use greppy_parser::Language;

/// Formatter policy for an edit. `SelectedRange` pipes only the replaced
/// span through the formatter; `File` formats the whole result and must be
/// explicitly permitted to change bytes outside the declared ranges (the
/// certificate flags the widened scope either way).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatPolicy {
    None,
    SelectedRange {
        argv: Vec<String>,
    },
    File {
        argv: Vec<String>,
        permit_outside: bool,
    },
}

/// Common options for single-operation verbs.
#[derive(Debug, Clone)]
pub struct VerbOptions {
    pub dry_run: bool,
    /// Emit the unified diff inline in the certificate.
    pub with_diff: bool,
    pub format: FormatPolicy,
    /// File hash captured when a symbol or handle was resolved.
    pub planned_file_sha256: Option<String>,
    /// Hash of the resolved target span captured at the same time.
    pub planned_target_sha256: Option<String>,
    /// Original coordinates of the resolved target span.
    pub planned_target_range: Option<(usize, usize)>,
    /// Accepted old-name occurrences after a workspace rename.
    pub expect_residual: Option<usize>,
}

impl Default for VerbOptions {
    fn default() -> Self {
        Self {
            dry_run: false,
            with_diff: true,
            format: FormatPolicy::None,
            planned_file_sha256: None,
            planned_target_sha256: None,
            planned_target_range: None,
            expect_residual: None,
        }
    }
}

/// Check the resolution-time file and target hashes against one immutable
/// snapshot. Supplying a target hash without its original range fails closed.
pub(crate) fn planned_preconditions_hold(snapshot: &Snapshot, options: &VerbOptions) -> bool {
    if options
        .planned_file_sha256
        .as_ref()
        .is_some_and(|expected| expected != &snapshot.file_sha256)
    {
        return false;
    }
    match (
        options.planned_target_sha256.as_ref(),
        options.planned_target_range,
    ) {
        (Some(expected), Some((start, end))) => snapshot
            .content
            .get(start..end)
            .is_some_and(|target| sha256_hex(target) == expected.as_str()),
        (Some(_), None) => false,
        _ => true,
    }
}

pub(crate) fn planned_precondition_refusal_for(
    workspace_root: &Path,
    snapshot: &Snapshot,
    options: &VerbOptions,
    engine: SelectorEngine,
    class: SelectorClass,
) -> Option<Certificate> {
    (!planned_preconditions_hold(snapshot, options)).then(|| {
        single_refusal_certificate(
            workspace_root,
            snapshot,
            engine,
            class,
            Status::Stale,
            options,
        )
    })
}

pub(crate) fn planned_precondition_refusal(
    workspace_root: &Path,
    snapshot: &Snapshot,
    options: &VerbOptions,
) -> Option<Certificate> {
    planned_precondition_refusal_for(
        workspace_root,
        snapshot,
        options,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
    )
}

/// Run `argv` on a temp file containing `content`; return the formatted
/// bytes. The literal `{}` in argv is replaced by the temp path; without a
/// placeholder the path is appended.
fn run_formatter(argv: &[String], content: &[u8], extension: &str) -> Result<Vec<u8>> {
    let tmp = tempfile::Builder::new()
        .suffix(&format!(".{extension}"))
        .tempfile()
        .map_err(|source| greppy_core::Error::Io {
            context: "formatter tempfile".into(),
            source,
        })?;
    std::fs::write(tmp.path(), content).map_err(|source| greppy_core::Error::Io {
        context: "write formatter input".into(),
        source,
    })?;
    let mut cmd_args: Vec<String> = Vec::new();
    let mut placed = false;
    for a in &argv[1..] {
        if a == "{}" {
            cmd_args.push(tmp.path().to_string_lossy().into_owned());
            placed = true;
        } else {
            cmd_args.push(a.clone());
        }
    }
    if !placed {
        cmd_args.push(tmp.path().to_string_lossy().into_owned());
    }
    let status = std::process::Command::new(&argv[0])
        .args(&cmd_args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|source| greppy_core::Error::Io {
            context: format!("spawn formatter {}", argv[0]),
            source,
        })?;
    if !status.success() {
        return Err(greppy_core::Error::Invalid(format!(
            "formatter {} exited with {}",
            argv[0],
            status.code().unwrap_or(-1)
        )));
    }
    std::fs::read(tmp.path()).map_err(|source| greppy_core::Error::Io {
        context: "read formatter output".into(),
        source,
    })
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
        false,
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
        true,
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
        true,
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
    if let Some(certificate) = planned_precondition_refusal_for(
        workspace_root,
        &snapshot,
        options,
        SelectorEngine::Regex,
        SelectorClass::RegexWeak,
    ) {
        return Ok(certificate);
    }
    let text = std::str::from_utf8(&snapshot.content)
        .map_err(|e| greppy_core::Error::Invalid(format!("regex-cas requires UTF-8: {e}")))?;
    let matches: Vec<(usize, usize)> = re.find_iter(text).map(|m| (m.start(), m.end())).collect();
    if matches.is_empty() {
        let already_satisfied = expect == 0
            || (!replacement.contains('$')
                && find_all(&snapshot.content, replacement.as_bytes()).len() == expect);
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Regex,
            SelectorClass::RegexWeak,
            if already_satisfied {
                Status::AlreadySatisfied
            } else {
                Status::NotFound
            },
            0,
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    if matches.len() != expect {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Regex,
            SelectorClass::RegexWeak,
            Status::Ambiguous,
            matches.len(),
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
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
    if ops
        .iter()
        .all(|op| snapshot.content[op.range.0..op.range.1] == op.replacement)
    {
        return Ok(single_op_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Regex,
            SelectorClass::RegexWeak,
            Status::AlreadySatisfied,
            matches.len(),
            &[],
            None,
            None,
            options,
            PublishMode::Atomic,
        ));
    }
    run_pipeline(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Regex,
        SelectorClass::RegexWeak,
        None,
        options,
        false,
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
    if let Some(certificate) = planned_precondition_refusal(workspace_root, &snapshot, options) {
        return Ok(certificate);
    }
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
        true,
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
    if let Some(certificate) = planned_precondition_refusal(workspace_root, &snapshot, options) {
        return Ok(certificate);
    }
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
        false,
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
    if let Some(certificate) = planned_precondition_refusal(workspace_root, &snapshot, options) {
        return Ok(certificate);
    }
    let mut end = def_range.1;
    if snapshot.content.get(end) == Some(&b'\n') {
        end += 1;
    }
    // also swallow ONE preceding blank line so deletions do not accumulate
    // double blank lines between the neighbours
    let mut start = def_range.0;
    if start >= 2
        && snapshot.content.get(start - 1) == Some(&b'\n')
        && snapshot.content.get(start - 2) == Some(&b'\n')
    {
        start -= 1;
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
        false,
    )
}

/// `greppy edit rename-call --in SYM --from A --to B`: retarget identifier
/// occurrences of `from` inside one definition span. AST-based: only
/// identifier-kind nodes are renamed, so strings and comments are never
/// touched. `expect`: None = all occurrences (at least one), Some(n) =
/// exactly n or refuse.
#[allow(clippy::too_many_arguments)] // symbol verb surface mirrors the CLI contract
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
    if let Some(certificate) = planned_precondition_refusal_for(
        workspace_root,
        &snapshot,
        options,
        SelectorEngine::TreeSitter,
        SelectorClass::Structural,
    ) {
        return Ok(certificate);
    }
    if def_range.0 >= def_range.1 || def_range.1 > snapshot.content.len() {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            options,
        ));
    }
    let sites = call_callee_sites(language, &snapshot.content, def_range, from.as_bytes());
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
        // Idempotency: when the old callee is gone, validate the replacement
        // call cardinality before declaring the requested end state satisfied.
        let to_sites = call_callee_sites(language, &snapshot.content, def_range, to.as_bytes())
            .unwrap_or_default();
        let status = match expect {
            Some(0) if to_sites.is_empty() => Status::AlreadySatisfied,
            Some(n) if to_sites.len() == n => Status::AlreadySatisfied,
            Some(_) if !to_sites.is_empty() => Status::Ambiguous,
            _ if !to_sites.is_empty() => Status::AlreadySatisfied,
            _ => Status::NotFound,
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
        false,
    )
}

/// Byte ranges of terminal callee identifiers for calls wholly contained in
/// `def_range`. Receiver/object identifiers and non-call uses are excluded.
fn call_callee_sites(
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
        if node.start_byte() >= def_range.0 && node.end_byte() <= def_range.1 {
            let target = node
                .child_by_field_name("function")
                .or_else(|| node.child_by_field_name("name"));
            if node.child_by_field_name("arguments").is_some() {
                if let Some(target) = target {
                    let mut target_cursor = target.walk();
                    let mut terminal_identifier = None;
                    loop {
                        let target_node = target_cursor.node();
                        if target_node.child_count() == 0
                            && target_node.kind().contains("identifier")
                        {
                            terminal_identifier =
                                Some((target_node.start_byte(), target_node.end_byte()));
                        }
                        if target_cursor.goto_first_child() {
                            continue;
                        }
                        loop {
                            if target_cursor.goto_next_sibling() {
                                break;
                            }
                            if !target_cursor.goto_parent() {
                                if let Some((start, end)) = terminal_identifier {
                                    if &content[start..end] == name {
                                        out.push((start, end));
                                    }
                                }
                                break;
                            }
                        }
                        if target_cursor.node() == target {
                            break;
                        }
                    }
                }
            }
        }
        if node.end_byte() >= def_range.0
            && node.start_byte() <= def_range.1
            && cursor.goto_first_child()
        {
            continue;
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
    out.sort_unstable();
    out.dedup();
    Some(out)
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
        if node.end_byte() >= def_range.0
            && node.start_byte() <= def_range.1
            && cursor.goto_first_child()
        {
            continue;
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

/// Compatibility entry point for the pre-M4 parameters-only API. Its inputs
/// cannot express argument add/remove/reorder, call-site cardinality, or
/// multi-file CAS, so it now fails closed instead of publishing a partial
/// definition-only edit. Use [`change_signature_files`] with a JSON spec.
pub fn change_signature(
    _workspace_root: &Path,
    _file: &Path,
    _def_range: (usize, usize),
    _new_parameters: &str,
    _call_sites: Vec<crate::certificate::Candidate>,
    _language: Language,
    _options: &VerbOptions,
) -> Result<Certificate> {
    Err(greppy_core::Error::Invalid(
        "parameters-only change-signature cannot rewrite call sites safely; use --spec with the graph backend"
            .into(),
    ))
}

/// JSON specification consumed by the graph-backed change-signature engine.
/// Parameter lists include their delimiters. Added parameters need an explicit
/// call-site expression so application never guesses a value.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChangeSignatureSpec {
    #[serde(alias = "oldParameters")]
    pub old_parameters: String,
    #[serde(alias = "newParameters")]
    pub new_parameters: String,
    #[serde(default, alias = "addedArguments")]
    pub added_arguments: std::collections::BTreeMap<String, String>,
    #[serde(alias = "expectCallSites")]
    pub expect_call_sites: usize,
}

#[derive(Debug, Clone)]
pub struct SignatureDefinition {
    pub rel_path: String,
    pub range: (usize, usize),
}

/// Validate semantic backend selection inside the edit engine. CLI wiring can
/// map the returned invalid specification to contract exit 20.
pub fn require_semantic_backend(backend: &str) -> Result<()> {
    match backend {
        "graph" => Ok(()),
        "lsp" => Err(greppy_core::Error::Invalid(
            "--backend lsp is unavailable in this build; use --backend graph".into(),
        )),
        other => Err(greppy_core::Error::Invalid(format!(
            "unknown semantic backend `{other}`; expected graph or lsp"
        ))),
    }
}

/// Graph-backed change-signature: update one definition and every graph scope
/// in one journal transaction. The spec supports removal (omit a former name),
/// reorder (change name order), and addition (new name plus `added_arguments`).
#[allow(clippy::too_many_arguments)]
pub fn change_signature_files(
    workspace_root: &Path,
    definition: &SignatureDefinition,
    call_scopes: &[RenameFileScope],
    symbol_name: &str,
    spec: &ChangeSignatureSpec,
    language: Language,
    options: &VerbOptions,
) -> Result<Certificate> {
    use std::collections::{BTreeMap, BTreeSet};

    require_semantic_backend("graph")?;
    let old_parameters = parse_parameter_declarations(&spec.old_parameters)?;
    let new_parameters = parse_parameter_declarations(&spec.new_parameters)?;
    let old_names: Vec<String> = old_parameters
        .iter()
        .filter(|parameter| !is_receiver_parameter(&parameter.name))
        .map(|parameter| parameter.name.clone())
        .collect();
    let new_names: Vec<String> = new_parameters
        .iter()
        .filter(|parameter| !is_receiver_parameter(&parameter.name))
        .map(|parameter| parameter.name.clone())
        .collect();
    if old_names.iter().collect::<BTreeSet<_>>().len() != old_names.len()
        || new_names.iter().collect::<BTreeSet<_>>().len() != new_names.len()
    {
        return Err(greppy_core::Error::Invalid(
            "change-signature parameter names must be unique".into(),
        ));
    }
    for added in new_names.iter().filter(|name| !old_names.contains(name)) {
        if !spec.added_arguments.contains_key(added) {
            return Err(greppy_core::Error::Invalid(format!(
                "added parameter `{added}` needs an `added_arguments` expression"
            )));
        }
    }

    let mut snapshots: BTreeMap<String, Snapshot> = BTreeMap::new();
    let definition_path = workspace_root.join(&definition.rel_path);
    let definition_snapshot = Snapshot::read(&definition_path)?;
    if let Some(certificate) =
        planned_precondition_refusal(workspace_root, &definition_snapshot, options)
    {
        return Ok(certificate);
    }
    let Some(parameter_range) =
        parameters_range_within(language, &definition_snapshot.content, definition.range)
    else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &definition_snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Semantic,
            Status::NotFound,
            options,
        ));
    };
    let live_parameters = &definition_snapshot.content[parameter_range.0..parameter_range.1];
    if live_parameters != spec.old_parameters.as_bytes() {
        return Ok(single_status_certificate(
            workspace_root,
            &definition_snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Semantic,
            if live_parameters == spec.new_parameters.as_bytes() {
                Status::AlreadySatisfied
            } else {
                Status::Stale
            },
            1,
            options,
        ));
    }
    snapshots.insert(definition.rel_path.clone(), definition_snapshot);

    let duplicate_scopes = call_scopes
        .iter()
        .map(|scope| (&scope.rel_path, &scope.spans))
        .collect::<BTreeSet<_>>()
        .len()
        != call_scopes.len();
    if duplicate_scopes {
        let snapshot = snapshots.get(&definition.rel_path).expect("inserted above");
        return Ok(single_status_certificate(
            workspace_root,
            snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Semantic,
            Status::Ambiguous,
            call_scopes.len(),
            options,
        ));
    }

    let mut call_ranges: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    let mut missing_scope = false;
    for scope in call_scopes {
        if !snapshots.contains_key(&scope.rel_path) {
            snapshots.insert(
                scope.rel_path.clone(),
                Snapshot::read(&workspace_root.join(&scope.rel_path))?,
            );
        }
        let snapshot = snapshots.get(&scope.rel_path).expect("inserted above");
        let mut sites = call_argument_sites(
            language,
            &snapshot.content,
            &scope.spans,
            symbol_name.as_bytes(),
        )
        .unwrap_or_default();
        sites.sort_unstable();
        sites.dedup();
        missing_scope |= sites.is_empty();
        call_ranges
            .entry(scope.rel_path.clone())
            .or_default()
            .extend(sites);
    }
    for ranges in call_ranges.values_mut() {
        ranges.sort_unstable();
        ranges.dedup();
    }
    let actual_call_sites: usize = call_ranges.values().map(Vec::len).sum();
    if missing_scope || actual_call_sites != spec.expect_call_sites {
        let snapshot = snapshots.get(&definition.rel_path).expect("inserted above");
        return Ok(single_status_certificate(
            workspace_root,
            snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Semantic,
            if actual_call_sites == 0 {
                Status::NotFound
            } else {
                Status::Ambiguous
            },
            actual_call_sites,
            options,
        ));
    }

    let expected_residual = options.expect_residual.unwrap_or(0);
    let mut ops_by_file: BTreeMap<String, Vec<PlannedOp>> = BTreeMap::new();
    ops_by_file
        .entry(definition.rel_path.clone())
        .or_default()
        .push(PlannedOp {
            id: "change-signature-definition".into(),
            range: parameter_range,
            replacement: spec.new_parameters.as_bytes().to_vec(),
        });
    for (rel_path, ranges) in &call_ranges {
        let snapshot = snapshots.get(rel_path).expect("loaded above");
        for (index, &range) in ranges.iter().enumerate() {
            let current =
                std::str::from_utf8(&snapshot.content[range.0..range.1]).map_err(|e| {
                    greppy_core::Error::Invalid(format!("call arguments are not UTF-8: {e}"))
                })?;
            let replacement =
                rewrite_call_arguments(current, &old_names, &new_names, &spec.added_arguments)?;
            ops_by_file
                .entry(rel_path.clone())
                .or_default()
                .push(PlannedOp {
                    id: format!("change-signature-call-{index}"),
                    range,
                    replacement: replacement.into_bytes(),
                });
        }
    }

    let mut reports = Vec::new();
    let mut publications = Vec::new();
    let mut valid = true;
    for (rel_path, snapshot) in snapshots {
        let ops = ops_by_file.remove(&rel_path).unwrap_or_default();
        let plan = plan_semantic_file(
            &rel_path,
            snapshot,
            ops,
            language,
            &format!("change-signature-{rel_path}"),
            1,
            options,
        )?;
        valid &= plan.valid;
        if let Some(publication) = plan.publication {
            publications.push(publication);
        }
        reports.push(plan.report);
    }
    let tx = format!(
        "ge-signature-{}",
        &sha256_hex(
            format!(
                "{symbol_name}:{}->{}",
                spec.old_parameters, spec.new_parameters
            )
            .as_bytes()
        )[..12]
    );
    let mut status = if valid {
        Status::Applied
    } else {
        Status::InvalidResult
    };
    if status == Status::Applied {
        // Evaluate the binding postcondition against the complete projected
        // workspace before publication so a mismatch refuses the transaction.
        let workspace_calls = count_workspace_calls(
            workspace_root,
            language,
            symbol_name,
            Some(publications.as_slice()),
        )?;
        let residual_occurrences = workspace_calls.saturating_sub(actual_call_sites);
        let passed = residual_occurrences == expected_residual;
        for report in &mut reports {
            report.residual_occurrences = Some(residual_occurrences);
            report.postconditions_passed &= passed;
            report.postconditions.push(PostconditionResult {
                name: "residual-call-sites".into(),
                passed,
                detail: Some(format!(
                    "expected {expected_residual} uncovered call site(s) of `{symbol_name}`, found {residual_occurrences}"
                )),
            });
            if !passed && !options.dry_run {
                report.file_sha256_after = None;
            }
        }
        if !passed {
            status = Status::InvalidResult;
        }
    }

    let mut published = false;
    if status == Status::Applied && !options.dry_run {
        match crate::journal::publish_journal(workspace_root, &tx, &publications) {
            Ok(()) => published = true,
            Err(error) => {
                status = crate::certificate::publish_error_status(&error);
                for report in &mut reports {
                    report.file_sha256_after = None;
                    report.guarantees.no_clobber = Guarantee::Failed;
                    report.postconditions_passed = false;
                }
            }
        }
    }

    Ok(Certificate {
        schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
        status,
        transaction_id: tx,
        workspace: WorkspaceReport {
            root: workspace_root.to_string_lossy().into_owned(),
            git_head_before: None,
            git_head_after: None,
        },
        operations: reports,
        validators: vec![],
        published,
        publish_mode: if options.dry_run {
            PublishMode::DryRun
        } else {
            PublishMode::Journal
        },
    })
}

#[derive(Debug)]
struct ParameterDeclaration {
    name: String,
}

fn parse_parameter_declarations(parameters: &str) -> Result<Vec<ParameterDeclaration>> {
    let parts = split_delimited_list(parameters)?;
    parts
        .into_iter()
        .filter(|part| !matches!(part.as_str(), "*" | "/"))
        .map(|part| {
            let name = parameter_name(&part).ok_or_else(|| {
                greppy_core::Error::Invalid(format!(
                    "cannot determine parameter name from `{part}`"
                ))
            })?;
            Ok(ParameterDeclaration { name })
        })
        .collect()
}

fn parameter_name(parameter: &str) -> Option<String> {
    let declaration = parameter
        .split_once('=')
        .map(|(left, _)| left)
        .unwrap_or(parameter)
        .trim();
    if declaration == "self" || declaration.ends_with(" self") || declaration.contains("&self") {
        return Some("self".into());
    }
    let before_type = declaration
        .split_once(':')
        .map(|(name, _)| name)
        .unwrap_or(declaration)
        .trim();
    before_type
        .trim_start_matches(['*', '&'])
        .strip_prefix("mut ")
        .unwrap_or_else(|| before_type.trim_start_matches(['*', '&']))
        .split_whitespace()
        .next()
        .map(|name| name.trim_end_matches('?').to_string())
        .filter(|name| !name.is_empty())
}

fn is_receiver_parameter(name: &str) -> bool {
    name == "self" || name == "this"
}

fn split_delimited_list(list: &str) -> Result<Vec<String>> {
    let trimmed = list.trim();
    let (open, close) = match (trimmed.as_bytes().first(), trimmed.as_bytes().last()) {
        (Some(b'('), Some(b')')) => (b'(', b')'),
        _ => {
            return Err(greppy_core::Error::Invalid(format!(
                "expected a parenthesized list, got `{list}`"
            )))
        }
    };
    let _ = (open, close);
    split_top_level(&trimmed[1..trimmed.len() - 1])
}

fn split_top_level(body: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in body.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote.is_some() && ch == '\\' {
            escaped = true;
            continue;
        }
        if let Some(current) = quote {
            if current == ch {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            '(' | '[' | '{' | '<' => stack.push(ch),
            ')' | ']' | '}' | '>' if stack.pop().is_none() => {
                return Err(greppy_core::Error::Invalid(
                    "unbalanced list delimiter".into(),
                ));
            }
            ')' | ']' | '}' | '>' => {}
            ',' if stack.is_empty() => {
                let part = body[start..index].trim();
                if !part.is_empty() {
                    parts.push(part.to_string());
                }
                start = index + 1;
            }
            _ => {}
        }
    }
    if quote.is_some() || !stack.is_empty() {
        return Err(greppy_core::Error::Invalid(
            "unbalanced parameter or argument list".into(),
        ));
    }
    let tail = body[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    Ok(parts)
}

fn named_argument(argument: &str) -> Option<(&str, &str)> {
    let (name, value) = argument.split_once('=')?;
    let name = name.trim();
    (!name.is_empty() && name.chars().all(|ch| ch.is_alphanumeric() || ch == '_'))
        .then_some((name, value.trim()))
}

fn rewrite_call_arguments(
    arguments: &str,
    old_names: &[String],
    new_names: &[String],
    added_arguments: &std::collections::BTreeMap<String, String>,
) -> Result<String> {
    let current = split_delimited_list(arguments)?;
    let mut positional = current.iter().filter(|arg| named_argument(arg).is_none());
    let named: std::collections::BTreeMap<&str, &str> = current
        .iter()
        .filter_map(|arg| named_argument(arg))
        .collect();
    let mut bound: std::collections::BTreeMap<&str, String> = std::collections::BTreeMap::new();
    for name in old_names {
        if let Some(value) = named.get(name.as_str()) {
            bound.insert(name, format!("{name}={value}"));
        } else if let Some(value) = positional.next() {
            bound.insert(name, value.clone());
        }
    }
    if positional.next().is_some()
        || named
            .keys()
            .any(|name| !old_names.iter().any(|old| old == name))
    {
        return Err(greppy_core::Error::Invalid(format!(
            "call `{arguments}` does not match the declared old signature"
        )));
    }
    let rewritten: Vec<String> = new_names
        .iter()
        .map(|name| {
            bound
                .get(name.as_str())
                .cloned()
                .or_else(|| added_arguments.get(name).cloned())
                .ok_or_else(|| {
                    greppy_core::Error::Invalid(format!(
                        "no call-site value available for parameter `{name}`"
                    ))
                })
        })
        .collect::<Result<_>>()?;
    Ok(format!("({})", rewritten.join(", ")))
}

fn call_argument_sites(
    language: Language,
    content: &[u8],
    scopes: &[(usize, usize)],
    name: &[u8],
) -> Option<Vec<(usize, usize)>> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let mut out = Vec::new();
    let mut cursor = tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = cursor.node();
        if let Some(arguments) = node.child_by_field_name("arguments") {
            let inside_scope = scopes.iter().any(|&(start, end)| {
                node.start_byte() >= start && node.end_byte() <= end.min(content.len())
            });
            let target = node
                .child_by_field_name("function")
                .or_else(|| node.child_by_field_name("name"));
            let target_matches = target.is_some_and(|target| {
                let mut target_cursor = target.walk();
                let mut terminal = None;
                loop {
                    let current = target_cursor.node();
                    if current.child_count() == 0 && current.kind().contains("identifier") {
                        terminal = content.get(current.start_byte()..current.end_byte());
                    }
                    if target_cursor.goto_first_child() {
                        continue;
                    }
                    loop {
                        if target_cursor.goto_next_sibling() {
                            break;
                        }
                        if !target_cursor.goto_parent() {
                            return terminal == Some(name);
                        }
                    }
                    if target_cursor.node() == target {
                        return terminal == Some(name);
                    }
                }
            });
            if inside_scope && target_matches {
                out.push((arguments.start_byte(), arguments.end_byte()));
            }
        }
        if cursor.goto_first_child() {
            continue;
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

fn count_workspace_calls(
    workspace_root: &Path,
    language: Language,
    name: &str,
    projected: Option<&[crate::journal::FilePublication]>,
) -> Result<usize> {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    let projected: BTreeMap<PathBuf, &[u8]> = projected
        .unwrap_or_default()
        .iter()
        .map(|publication| {
            (
                PathBuf::from(&publication.rel_path),
                publication.content.as_slice(),
            )
        })
        .collect();

    fn visit(
        root: &Path,
        dir: &Path,
        language: Language,
        name: &[u8],
        projected: &BTreeMap<PathBuf, &[u8]>,
    ) -> Result<usize> {
        let mut count = 0usize;
        for entry in std::fs::read_dir(dir).map_err(|source| greppy_core::Error::Io {
            context: format!("read directory {}", dir.display()),
            source,
        })? {
            let entry = entry.map_err(|source| greppy_core::Error::Io {
                context: format!("read directory entry in {}", dir.display()),
                source,
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|source| greppy_core::Error::Io {
                context: format!("stat {}", path.display()),
                source,
            })?;
            if file_type.is_dir() {
                let dir_name = entry.file_name();
                let dir_name = dir_name.to_string_lossy();
                if matches!(
                    dir_name.as_ref(),
                    ".git" | ".greppy-edit-journal" | "target" | "node_modules" | ".venv" | "venv"
                ) {
                    continue;
                }
                count += visit(root, &path, language, name, projected)?;
            } else if file_type.is_file() && greppy_parser::language_for_path(&path) == language {
                let rel_path = path.strip_prefix(root).unwrap_or(&path);
                let content = if let Some(content) = projected.get(rel_path) {
                    (*content).to_vec()
                } else {
                    std::fs::read(&path).map_err(|source| greppy_core::Error::Io {
                        context: format!("read {}", path.display()),
                        source,
                    })?
                };
                count += call_argument_sites(language, &content, &[(0, usize::MAX)], name)
                    .unwrap_or_default()
                    .len();
            }
        }
        Ok(count)
    }
    visit(
        workspace_root,
        workspace_root,
        language,
        name.as_bytes(),
        &projected,
    )
}

/// Smallest node covering `def_range` with the range's leading whitespace
/// skipped. Store spans are line-addressed and start at column 0, but the
/// definition node starts after the indentation; querying tree-sitter with
/// the padding included resolves to the PARENT node (for a method inside an
/// impl: the declaration list), whose "body"/"parameters" fields then fail
/// the containment check - an indented definition reported not-found while
/// a top-level one worked (trace forensics 2026-07-17).
fn padded_range_query_start(content: &[u8], def_range: (usize, usize)) -> usize {
    content
        .get(def_range.0..def_range.1)
        .and_then(|span| span.iter().position(|b| !b.is_ascii_whitespace()))
        .map(|off| def_range.0 + off)
        .unwrap_or(def_range.0)
}

/// Byte range of the `parameters` field of the definition at `def_range`.
fn parameters_range_within(
    language: Language,
    content: &[u8],
    def_range: (usize, usize),
) -> Option<(usize, usize)> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let mut node = tree.root_node().descendant_for_byte_range(
        padded_range_query_start(content, def_range),
        def_range.1.saturating_sub(1),
    )?;
    loop {
        if let Some(params) = node.child_by_field_name("parameters") {
            if params.start_byte() >= def_range.0 && params.end_byte() <= def_range.1 {
                return Some((params.start_byte(), params.end_byte()));
            }
        }
        node = node.parent()?;
    }
}

/// Structural signature fingerprint of the definition at `def_range`:
/// node kind, name, and parameter list text - the body is excluded, so the
/// fingerprint is stable across body edits and changes when the signature
/// changes. None when the language cannot be parsed.
pub fn signature_fingerprint(
    language: Language,
    content: &[u8],
    def_range: (usize, usize),
) -> Option<String> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let mut node = tree.root_node().descendant_for_byte_range(
        padded_range_query_start(content, def_range),
        def_range.1.saturating_sub(1),
    )?;
    // climb to the smallest node with a name field covering the range
    loop {
        if node.child_by_field_name("name").is_some() {
            break;
        }
        node = node.parent()?;
    }
    let field = |name: &str| {
        node.child_by_field_name(name)
            .map(|n| String::from_utf8_lossy(&content[n.start_byte()..n.end_byte()]).into_owned())
            .unwrap_or_default()
    };
    let material = format!(
        "{}|{}|{}|{}",
        node.kind(),
        field("name"),
        field("parameters"),
        field("return_type"),
    );
    Some(crate::hash::sha256_hex(material.as_bytes())[..16].to_string())
}

/// Cross-file rename input: per file, the byte spans in which identifier
/// occurrences of the old name are AST-verified and renamed. Spans of
/// `(0, usize::MAX)` mean "the whole file" (definition file, import lines).
#[derive(Debug, Clone)]
pub struct RenameFileScope {
    pub rel_path: String,
    pub spans: Vec<(usize, usize)>,
}

struct SemanticFilePlan {
    report: OperationReport,
    publication: Option<crate::journal::FilePublication>,
    valid: bool,
}

fn plan_semantic_file(
    rel_path: &str,
    snapshot: Snapshot,
    ops: Vec<PlannedOp>,
    language: Language,
    id: &str,
    scope_matches: usize,
    options: &VerbOptions,
) -> Result<SemanticFilePlan> {
    let applied = apply_in_memory(&snapshot, &ops)?;
    let syntax_before = syntax_counts(language, &snapshot.content);
    let syntax_after = syntax_counts(language, &applied.content);
    let (syntax, applicable) = match (syntax_before, syntax_after) {
        (Some(before), Some(after)) => (
            SyntaxDelta {
                errors_before: before.errors,
                errors_after: after.errors,
                new_errors: after.errors.saturating_sub(before.errors),
                new_missing_nodes: after.missing.saturating_sub(before.missing),
            },
            true,
        ),
        _ => (
            SyntaxDelta {
                errors_before: 0,
                errors_after: 0,
                new_errors: 0,
                new_missing_nodes: 0,
            },
            false,
        ),
    };
    let syntax_ok = !applicable || (syntax.new_errors == 0 && syntax.new_missing_nodes == 0);
    let isolation_ok = outside_ranges_unchanged(&snapshot.content, &applied.content, &ops);
    let valid = syntax_ok && isolation_ok;
    let target_before: Vec<u8> = ops
        .iter()
        .flat_map(|op| snapshot.content[op.range.0..op.range.1].iter().copied())
        .collect();
    let target_after: Vec<u8> = ops
        .iter()
        .flat_map(|op| op.replacement.iter().copied())
        .collect();
    let changed = !ops.is_empty() && applied.content != snapshot.content;
    let report = OperationReport {
        id: id.into(),
        file: rel_path.into(),
        selector_engine: SelectorEngine::Symbol,
        selector_class: SelectorClass::Semantic,
        scope_matches,
        target_matches: ops.len(),
        file_sha256_before: snapshot.file_sha256.clone(),
        file_sha256_after: Some(applied.file_sha256.clone()),
        target_sha256_before: sha256_hex(&target_before),
        target_sha256_after: Some(sha256_hex(&target_after)),
        outside_declared_ranges_unchanged: isolation_ok,
        changed_byte_ranges: applied.changed_ranges.clone(),
        node_before: None,
        node_after: None,
        unified_diff: (options.with_diff && changed)
            .then(|| unified_diff(rel_path, &snapshot.content, &applied.content)),
        syntax,
        postconditions_passed: valid,
        postconditions: vec![],
        residual_occurrences: None,
        guarantees: Guarantees {
            addressed_range: if ops.is_empty() {
                Guarantee::Failed
            } else {
                Guarantee::Proved
            },
            no_clobber: Guarantee::Proved,
            byte_isolation: if isolation_ok {
                Guarantee::Proved
            } else {
                Guarantee::Failed
            },
            syntax: if !applicable {
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
    let publication = changed.then(|| crate::journal::FilePublication {
        rel_path: rel_path.into(),
        expected_live_sha256: snapshot.file_sha256,
        content: applied.content,
    });
    Ok(SemanticFilePlan {
        report,
        publication,
        valid,
    })
}

fn identifier_name_occurrences(content: &[u8], name: &[u8]) -> usize {
    if name.is_empty() {
        return 0;
    }
    content
        .windows(name.len())
        .enumerate()
        .filter(|(start, window)| {
            if *window != name {
                return false;
            }
            let before = start
                .checked_sub(1)
                .and_then(|index| content.get(index))
                .copied();
            let after = content.get(start + name.len()).copied();
            let is_identifier = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
            before.is_none_or(|byte| !is_identifier(byte))
                && after.is_none_or(|byte| !is_identifier(byte))
        })
        .count()
}

fn count_workspace_residuals(
    workspace_root: &Path,
    language: Language,
    name: &str,
    projected: Option<&[crate::journal::FilePublication]>,
) -> Result<usize> {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    let projected: BTreeMap<PathBuf, &[u8]> = projected
        .unwrap_or_default()
        .iter()
        .map(|publication| {
            (
                PathBuf::from(&publication.rel_path),
                publication.content.as_slice(),
            )
        })
        .collect();

    fn visit(
        root: &Path,
        dir: &Path,
        language: Language,
        name: &[u8],
        projected: &BTreeMap<PathBuf, &[u8]>,
    ) -> Result<usize> {
        let entries = std::fs::read_dir(dir).map_err(|source| greppy_core::Error::Io {
            context: format!("read directory {}", dir.display()),
            source,
        })?;
        let mut count = 0usize;
        for entry in entries {
            let entry = entry.map_err(|source| greppy_core::Error::Io {
                context: format!("read directory entry in {}", dir.display()),
                source,
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|source| greppy_core::Error::Io {
                context: format!("stat {}", path.display()),
                source,
            })?;
            if file_type.is_dir() {
                let dir_name = entry.file_name();
                let dir_name = dir_name.to_string_lossy();
                if matches!(
                    dir_name.as_ref(),
                    ".git" | ".greppy-edit-journal" | "target" | "node_modules" | ".venv" | "venv"
                ) {
                    continue;
                }
                count += visit(root, &path, language, name, projected)?;
            } else if file_type.is_file() && greppy_parser::language_for_path(&path) == language {
                let rel_path = path.strip_prefix(root).unwrap_or(&path);
                let content = if let Some(content) = projected.get(rel_path) {
                    (*content).to_vec()
                } else {
                    std::fs::read(&path).map_err(|source| greppy_core::Error::Io {
                        context: format!("read {}", path.display()),
                        source,
                    })?
                };
                count += identifier_sites(language, &content, (0, content.len()), name)
                    .map(|sites| sites.len())
                    .unwrap_or_else(|| {
                        // Unsupported or otherwise unparseable files cannot be
                        // classified by syntax, so retain the conservative
                        // identifier-boundary text fallback for those files only.
                        identifier_name_occurrences(&content, name)
                    });
            }
        }
        Ok(count)
    }

    visit(
        workspace_root,
        workspace_root,
        language,
        name.as_bytes(),
        &projected,
    )
}

/// Graph-backed `rename-symbol`: rename identifier occurrences of `from`
/// inside the given per-file scopes, publish all files as one journal
/// transaction. Every site is AST-verified (identifier-kind leaf nodes
/// only). Returns the certificate with one operation report per file.
pub fn rename_symbol_files(
    workspace_root: &Path,
    scopes: &[RenameFileScope],
    from: &str,
    to: &str,
    options: &VerbOptions,
) -> Result<Certificate> {
    use std::collections::BTreeSet;

    let tx = format!(
        "ge-rename-{}",
        &sha256_hex(format!("{from}->{to}").as_bytes())[..12]
    );
    let residual_language = scopes
        .iter()
        .find(|scope| scope.spans.contains(&(0, usize::MAX)))
        .or_else(|| scopes.first())
        .map(|scope| greppy_parser::language_for_path(Path::new(&scope.rel_path)));
    let duplicate_paths = scopes
        .iter()
        .map(|scope| scope.rel_path.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        != scopes.len();
    if scopes.is_empty() || duplicate_paths {
        let dummy = Snapshot {
            path: workspace_root.to_path_buf(),
            content: Vec::new(),
            file_sha256: String::new(),
        };
        return Ok(single_status_certificate(
            workspace_root,
            &dummy,
            SelectorEngine::Symbol,
            SelectorClass::Semantic,
            if duplicate_paths {
                Status::Ambiguous
            } else {
                Status::NotFound
            },
            scopes.len(),
            options,
        ));
    }

    let mut reports = Vec::with_capacity(scopes.len());
    let mut publications = Vec::new();
    let mut missing_scope = false;
    let mut already_renamed = true;
    let mut all_valid = true;
    for scope in scopes {
        let abs = workspace_root.join(&scope.rel_path);
        let snapshot = Snapshot::read(&abs)?;
        let language = greppy_parser::language_for_path(&abs);
        let mut sites = Vec::new();
        let mut replacement_sites = Vec::new();
        for &(start, end) in &scope.spans {
            let range = (start, end.min(snapshot.content.len()));
            if range.0 > range.1 {
                continue;
            }
            if let Some(mut found) =
                identifier_sites(language, &snapshot.content, range, from.as_bytes())
            {
                sites.append(&mut found);
            }
            if let Some(mut found) =
                identifier_sites(language, &snapshot.content, range, to.as_bytes())
            {
                replacement_sites.append(&mut found);
            }
        }
        sites.sort_unstable();
        sites.dedup();
        replacement_sites.sort_unstable();
        replacement_sites.dedup();
        missing_scope |= sites.is_empty();
        already_renamed &= sites.is_empty() && !replacement_sites.is_empty();
        let ops: Vec<PlannedOp> = sites
            .iter()
            .enumerate()
            .map(|(index, &(start, end))| PlannedOp {
                id: format!("rename-{}-{index}", scope.rel_path),
                range: (start, end),
                replacement: to.as_bytes().to_vec(),
            })
            .collect();
        let plan = plan_semantic_file(
            &scope.rel_path,
            snapshot,
            ops,
            language,
            &format!("rename-{}", scope.rel_path),
            scope.spans.len(),
            options,
        )?;
        all_valid &= plan.valid;
        if let Some(publication) = plan.publication {
            publications.push(publication);
        }
        reports.push(plan.report);
    }

    let mut status = if from == to || already_renamed {
        Status::AlreadySatisfied
    } else if missing_scope {
        Status::NotFound
    } else if !all_valid {
        Status::InvalidResult
    } else {
        Status::Applied
    };
    if status == Status::Applied {
        let expected = options.expect_residual.unwrap_or(0);
        if let Some(language) = residual_language {
            let residual_occurrences = count_workspace_residuals(
                workspace_root,
                language,
                from,
                Some(publications.as_slice()),
            )?;
            let passed = residual_occurrences == expected;
            for report in &mut reports {
                report.residual_occurrences = Some(residual_occurrences);
                report.postconditions_passed &= passed;
                report.postconditions.push(PostconditionResult {
                    name: "residual-occurrences".into(),
                    passed,
                    detail: Some(format!(
                        "expected {expected} occurrence(s) of `{from}`, found {residual_occurrences}"
                    )),
                });
                if !passed {
                    report.file_sha256_after = None;
                }
            }
            if !passed {
                status = Status::InvalidResult;
            }
        }
    }

    let mut published = false;
    if status == Status::Applied && !options.dry_run {
        match crate::journal::publish_journal(workspace_root, &tx, &publications) {
            Ok(()) => published = true,
            Err(error) => {
                status = crate::certificate::publish_error_status(&error);
                for report in &mut reports {
                    report.file_sha256_after = None;
                    report.guarantees.no_clobber = Guarantee::Failed;
                    report.postconditions_passed = false;
                }
            }
        }
    }

    Ok(Certificate {
        schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
        status,
        transaction_id: tx,
        workspace: WorkspaceReport {
            root: workspace_root.to_string_lossy().into_owned(),
            git_head_before: None,
            git_head_after: None,
        },
        operations: reports,
        validators: vec![],
        published,
        publish_mode: if options.dry_run {
            PublishMode::DryRun
        } else {
            PublishMode::Journal
        },
    })
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
    let mut node = tree.root_node().descendant_for_byte_range(
        padded_range_query_start(content, def_range),
        def_range.1.saturating_sub(1),
    )?;
    loop {
        let body = node.child_by_field_name("body").or_else(|| {
            let mut cursor = node.walk();
            if matches!(language, Language::TypeScript { .. }) && node.kind() == "export_statement"
            {
                // TypeScript wraps `export function ...` in an export node;
                // the declaration (and its body field) is the direct child.
                node.named_children(&mut cursor)
                    .find_map(|child| child.child_by_field_name("body"))
            } else if language == Language::Kotlin && node.kind() == "function_declaration" {
                // tree-sitter-kotlin-ng does not field-label function bodies.
                node.named_children(&mut cursor)
                    .find(|child| matches!(child.kind(), "function_body" | "block"))
            } else {
                None
            }
        });
        if let Some(body) = body {
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
        false,
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
    single_status_certificate(workspace_root, snapshot, engine, class, status, 0, options)
}

pub(crate) fn single_status_certificate(
    workspace_root: &Path,
    snapshot: &Snapshot,
    engine: SelectorEngine,
    class: SelectorClass,
    status: Status,
    matches: usize,
    options: &VerbOptions,
) -> Certificate {
    single_op_certificate(
        workspace_root,
        snapshot,
        engine,
        class,
        status,
        matches,
        &[],
        None,
        None,
        options,
        PublishMode::Atomic,
    )
}

/// The shared transaction pipeline for single-file verbs.
#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    workspace_root: &Path,
    snapshot: Snapshot,
    ops: Vec<PlannedOp>,
    engine: SelectorEngine,
    class: SelectorClass,
    language: Option<Language>,
    options: &VerbOptions,
    enforce_structure: bool,
) -> Result<Certificate> {
    if let Some(certificate) =
        planned_precondition_refusal_for(workspace_root, &snapshot, options, engine, class)
    {
        return Ok(certificate);
    }
    let syntax_before = language.and_then(|l| syntax_counts(l, &snapshot.content));
    let mut applied = apply_in_memory(&snapshot, &ops)?;
    let mut formatter_expanded = false;
    let ext = snapshot
        .path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("txt")
        .to_string();
    match &options.format {
        FormatPolicy::None => {}
        FormatPolicy::SelectedRange { argv } => {
            // format each replacement in isolation, re-apply
            let mut new_ops = Vec::with_capacity(ops.len());
            for op in &ops {
                let formatted = run_formatter(argv, &op.replacement, &ext)?;
                new_ops.push(PlannedOp {
                    id: op.id.clone(),
                    range: op.range,
                    replacement: formatted,
                });
            }
            applied = apply_in_memory(&snapshot, &new_ops)?;
        }
        FormatPolicy::File {
            argv,
            permit_outside,
        } => {
            let formatted = run_formatter(argv, &applied.content, &ext)?;
            if formatted != applied.content {
                let expanded = !outside_ranges_unchanged(&snapshot.content, &formatted, &ops);
                if expanded && !permit_outside {
                    return Err(greppy_core::Error::Invalid(
                        "formatter changed bytes outside the declared ranges; pass permit-outside to allow".into(),
                    ));
                }
                formatter_expanded = expanded;
                applied.file_sha256 = crate::hash::sha256_hex(&formatted);
                applied.content = formatted;
            }
        }
    }
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
    // Structural context preservation: tree-sitter recovers past many
    // malformations without ERROR nodes, so the count-based check alone is
    // unsound (a body replaced with a whole file parsed clean — see
    // structural_context_preserved). Require the edited region to still sit
    // in the same ancestor-kind chain it did before. Applied only to the
    // single-op verbs (replace-body/-span, change-signature, …) where the
    // post-edit span is exactly [op.range.0, op.range.0 + replacement.len());
    // multi-op edits fall back to the count/isolation checks.
    let structure_ok = match (enforce_structure, language, ops.as_slice()) {
        (true, Some(l), [op]) => {
            let after_range = (op.range.0, op.range.0 + op.replacement.len());
            crate::txn::structural_context_preserved(
                l,
                &snapshot.content,
                op.range,
                &applied.content,
                after_range,
            )
        }
        _ => true,
    };
    let syntax_ok = (!syntax_applicable
        || (syntax.new_errors == 0 && syntax.new_missing_nodes == 0))
        && structure_ok;
    let isolation_ok =
        formatter_expanded || outside_ranges_unchanged(&snapshot.content, &applied.content, &ops);

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
        // Re-check the resolution-time hashes against the live file directly
        // before the publication CAS. The atomic publisher closes the smaller
        // race between this read and rename.
        let planned_still_live =
            if options.planned_file_sha256.is_some() || options.planned_target_sha256.is_some() {
                Snapshot::read(&snapshot.path)
                    .map(|live| planned_preconditions_hold(&live, options))
                    .unwrap_or(true)
            } else {
                true
            };
        if !planned_still_live {
            status = Status::Stale;
        } else {
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
                    status = crate::certificate::publish_error_status(&e);
                }
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
        residual_occurrences: None,
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
        formatter_expanded_change_scope: formatter_expanded,
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
            residual_occurrences: None,
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
pub(crate) fn unified_diff_public(path: &str, before: &[u8], after: &[u8]) -> String {
    unified_diff(path, before, after)
}

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

    #[cfg(unix)]
    #[test]
    fn unsafe_publish_is_publish_failed_not_stale() {
        let dir = ws();
        let f = dir.path().join("conf.ini");
        let alias = dir.path().join("alias.ini");
        std::fs::write(&f, b"port = 9000\n").unwrap();
        std::fs::hard_link(&f, &alias).unwrap();

        let cert = text_cas(
            dir.path(),
            &f,
            b"port = 9000",
            b"port = 8080",
            1,
            &VerbOptions::default(),
        )
        .unwrap();

        assert_eq!(cert.status, Status::PublishFailed);
        assert_eq!(cert.exit_code(), 16);
        assert!(!cert.published);
        assert_eq!(std::fs::read(&f).unwrap(), b"port = 9000\n");
        let json = serde_json::to_value(&cert).unwrap();
        let _: Certificate = serde_json::from_value(json).unwrap();
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

    /// A portable in-place "formatter" (GNU/BSD sed -i syntax differs):
    /// normalizes ` = ` spacing across the whole file via python3.
    fn portable_normalizer_argv() -> Vec<String> {
        vec![
            "python3".into(),
            "-c".into(),
            "import sys,re;p=sys.argv[1];s=open(p).read();open(p,'w').write(re.sub(r' *= *',' = ',s))".into(),
            "{}".into(),
        ]
    }

    #[test]
    fn file_formatter_without_permit_refuses_scope_expansion() {
        let dir = ws();
        let f = dir.path().join("conf.txt");
        std::fs::write(&f, b"a = 1\nb =    2\n").unwrap();
        // "formatter" der die ganze Datei normalisiert: sed als argv-tool
        let opts = VerbOptions {
            dry_run: false,
            with_diff: false,
            format: FormatPolicy::File {
                argv: vec!["sed".into(), "-i".into(), "".into(), "s/ *= */ = /".into()],
                permit_outside: false,
            },
            ..Default::default()
        };
        let err = text_cas(dir.path(), &f, b"a = 1", b"a = 9", 1, &opts);
        // der normalizer aendert auch zeile b -> scope-expansion ohne permit: Fehler,
        // Datei unveraendert
        assert!(err.is_err());
        assert_eq!(std::fs::read(&f).unwrap(), b"a = 1\nb =    2\n");
        // mit permit: durchgelassen und geflaggt
        let opts = VerbOptions {
            dry_run: false,
            with_diff: false,
            format: FormatPolicy::File {
                argv: portable_normalizer_argv(),
                permit_outside: true,
            },
            ..Default::default()
        };
        let cert = text_cas(dir.path(), &f, b"a = 1", b"a = 9", 1, &opts).unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(cert.operations[0].formatter_expanded_change_scope);
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
        // String literals and non-call identifier uses are outside rename-call's
        // structural target class and remain untouched.
        assert!(out.contains("print(\"legacy_auth\")"), "{out}");
        assert!(out.contains("x = legacy_auth"), "{out}");
    }

    #[test]
    fn rename_call_expect_mismatch_refuses() {
        let dir = ws();
        let f = dir.path().join("m.py");
        let content = b"def run():\n    a()\n    a()\n";
        std::fs::write(&f, content).unwrap();
        let cert = rename_in_span(
            dir.path(),
            &f,
            (0, content.len()),
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
    fn replace_body_resolves_indented_method_from_line_start_range() {
        // Store spans are line-addressed: the range starts at column 0,
        // BEFORE the method's indentation. The padded bytes used to make
        // the node lookup land on the impl's declaration list, whose body
        // starts before the range - reported not-found for every indented
        // method while top-level functions worked (2026-07-17 forensics).
        let dir = ws();
        let f = dir.path().join("impls.rs");
        let content: &[u8] = b"pub struct R { pub s: u32 }\nimpl R {\n    pub fn serialize(&self) -> String {\n        format!(\"begin={}\", self.s)\n    }\n}\n";
        std::fs::write(&f, content).unwrap();
        let text = std::str::from_utf8(content).unwrap();
        let start = text.find("    pub fn serialize").unwrap();
        let end = text.rfind("    }\n").unwrap() + "    }".len();
        let cert = replace_body(
            dir.path(),
            &f,
            (start, end),
            b"{\n        format!(\"start={}\", self.s)\n    }",
            Language::Rust,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied, "{cert:?}");
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("start={}"), "{out}");
        assert!(out.contains("pub fn serialize(&self) -> String"), "{out}");
    }

    #[test]
    fn replace_body_rejects_whole_file_as_replacement() {
        // The --source-file footgun (agent points it at the real file, not a
        // snippet): replacing a method body with the whole file's text
        // parses ERROR-node-free under tree-sitter-go's recovery, so the
        // count-based syntax check passed it as `applied`/`proved` while
        // gofmt rejects the result (forensics 2026-07-17). The structural
        // context check must catch it.
        let dir = ws();
        let f = dir.path().join("strings.go");
        let content: &[u8] = b"package hstrings\n\nimport \"strings\"\n\nfunc Fold(a, b string) bool {\n\treturn strings.EqualFold(a, b)\n}\n";
        std::fs::write(&f, content).unwrap();
        let text = std::str::from_utf8(content).unwrap();
        let fs = text.find("func Fold").unwrap();
        let fe = text.rfind('}').unwrap() + 1;
        let cert = replace_body(
            dir.path(),
            &f,
            (fs, fe),
            content, // the whole file as the "new body" — the footgun
            Language::Go,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::InvalidResult, "{cert:?}");
        assert_eq!(cert.operations[0].guarantees.syntax, Guarantee::Failed);
        assert!(!cert.published);
    }

    #[test]
    fn signature_fingerprint_works_for_indented_method() {
        let content: &[u8] = b"impl R {\n    pub fn serialize(&self, n: u32) -> String {\n        format!(\"x\")\n    }\n}\n";
        let text = std::str::from_utf8(content).unwrap();
        let start = text.find("    pub fn").unwrap();
        let end = text.rfind("    }").unwrap() + 5;
        let fp = signature_fingerprint(Language::Rust, content, (start, end));
        assert!(fp.is_some(), "indented method must fingerprint");
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

    fn rename_options(expect_residual: usize) -> VerbOptions {
        VerbOptions {
            expect_residual: Some(expect_residual),
            ..Default::default()
        }
    }

    #[test]
    fn rename_symbol_clean_workspace_satisfies_zero_residuals() {
        let dir = ws();
        std::fs::write(
            dir.path().join("a.rs"),
            b"fn old_name() {}\nfn caller() { old_name(); }\n",
        )
        .unwrap();
        let scopes = vec![RenameFileScope {
            rel_path: "a.rs".into(),
            spans: vec![(0, usize::MAX)],
        }];

        let certificate = rename_symbol_files(
            dir.path(),
            &scopes,
            "old_name",
            "new_name",
            &rename_options(0),
        )
        .unwrap();

        assert_eq!(certificate.status, Status::Applied);
        assert_eq!(certificate.exit_code(), 0);
        assert!(certificate.published);
        assert_eq!(certificate.operations[0].residual_occurrences, Some(0));
    }

    #[test]
    fn rename_symbol_unplanned_leftover_fails_residual_postcondition() {
        let dir = ws();
        std::fs::write(dir.path().join("a.rs"), b"fn old_name() {}\n").unwrap();
        std::fs::write(
            dir.path().join("leftover.rs"),
            b"fn caller() { old_name(); }\n",
        )
        .unwrap();
        let scopes = vec![RenameFileScope {
            rel_path: "a.rs".into(),
            spans: vec![(0, usize::MAX)],
        }];

        let certificate = rename_symbol_files(
            dir.path(),
            &scopes,
            "old_name",
            "new_name",
            &rename_options(0),
        )
        .unwrap();

        assert_eq!(certificate.status, Status::InvalidResult);
        assert_eq!(certificate.exit_code(), 13);
        assert!(!certificate.published);
        assert_eq!(certificate.operations[0].file_sha256_after, None);
        assert_eq!(certificate.operations[0].residual_occurrences, Some(1));
        assert!(!certificate.operations[0].postconditions_passed);
        assert_eq!(
            std::fs::read(dir.path().join("a.rs")).unwrap(),
            b"fn old_name() {}\n"
        );
    }

    #[test]
    fn rename_symbol_expected_one_residual_is_accepted() {
        let dir = ws();
        std::fs::write(dir.path().join("a.rs"), b"fn old_name() {}\n").unwrap();
        std::fs::write(
            dir.path().join("leftover.rs"),
            b"fn caller() { old_name(); }\n",
        )
        .unwrap();
        let scopes = vec![RenameFileScope {
            rel_path: "a.rs".into(),
            spans: vec![(0, usize::MAX)],
        }];

        let certificate = rename_symbol_files(
            dir.path(),
            &scopes,
            "old_name",
            "new_name",
            &rename_options(1),
        )
        .unwrap();

        assert_eq!(certificate.status, Status::Applied);
        assert_eq!(certificate.exit_code(), 0);
        assert!(certificate.published);
        assert_eq!(certificate.operations[0].residual_occurrences, Some(1));
        assert!(certificate.operations[0].postconditions_passed);
    }
}
