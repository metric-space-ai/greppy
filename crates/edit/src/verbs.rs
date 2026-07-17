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
