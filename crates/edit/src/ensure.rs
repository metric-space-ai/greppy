//! Idempotent `ensure-*` operations.
//!
//! `ensure-import`: make sure a module/name import exists. Present →
//! `already-satisfied` (no write, no error); absent → inserted at the
//! canonical position; conflicting (same name bound from a different
//! module) → `invalid-result`, nothing written.

use std::path::Path;

use crate::certificate::{Certificate, SelectorClass, SelectorEngine, Status};
use crate::txn::{PlannedOp, Snapshot};
use crate::verbs::{
    planned_precondition_refusal, planned_precondition_refusal_for, run_pipeline_public,
    single_refusal_certificate, single_status_certificate, VerbOptions,
};
use greppy_core::Result;
use greppy_parser::Language;

/// The import line we would write, per language.
fn import_line(language: Language, module: &str, name: Option<&str>) -> Option<String> {
    Some(match (language, name) {
        (Language::Python, Some(n)) => format!("from {module} import {n}"),
        (Language::Python, None) => format!("import {module}"),
        (Language::Rust, Some(n)) => format!("use {module}::{n};"),
        (Language::Rust, None) => format!("use {module};"),
        (Language::Go, _) => format!("import \"{module}\""),
        (Language::TypeScript { .. } | Language::JavaScript, Some(n)) => {
            format!("import {{ {n} }} from \"{module}\";")
        }
        (Language::TypeScript { .. } | Language::JavaScript, None) => {
            format!("import \"{module}\";")
        }
        _ => return None,
    })
}

/// Node kinds that represent import statements per grammar.
fn import_kinds(language: Language) -> &'static [&'static str] {
    match language {
        Language::Python => &["import_statement", "import_from_statement"],
        Language::Rust => &["use_declaration"],
        Language::Go => &["import_declaration"],
        Language::TypeScript { .. } | Language::JavaScript => &["import_statement"],
        _ => &[],
    }
}

struct ImportScan {
    /// byte offset AFTER the last top-level import (insertion point), or the
    /// canonical start-of-file position when no import exists
    insert_at: usize,
    /// an import binding `name` (or bare `module`) already exists
    satisfied: bool,
    /// `name` is already bound from a DIFFERENT module
    conflict: Option<String>,
}

fn scan_imports(
    language: Language,
    content: &[u8],
    module: &str,
    name: Option<&str>,
) -> Option<ImportScan> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let kinds = import_kinds(language);
    let mut insert_at = 0usize;
    let mut satisfied = false;
    let mut conflict = None;
    let root = tree.root_node();
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if !kinds.contains(&node.kind()) {
            continue;
        }
        let text = String::from_utf8_lossy(&content[node.start_byte()..node.end_byte()]);
        let mut end = node.end_byte();
        if content.get(end) == Some(&b'\n') {
            end += 1;
        }
        insert_at = end;
        let mentions_module = text.contains(module);
        match name {
            Some(n) => {
                let mentions_name = text
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .any(|tok| tok == n);
                if mentions_module && mentions_name {
                    satisfied = true;
                } else if mentions_name && !mentions_module {
                    conflict = Some(text.trim().to_string());
                }
            }
            None => {
                if mentions_module {
                    satisfied = true;
                }
            }
        }
    }
    Some(ImportScan {
        insert_at,
        satisfied,
        conflict,
    })
}

/// `greppy edit ensure-import --file F --module M [--name N]`.
pub fn ensure_import(
    workspace_root: &Path,
    file: &Path,
    module: &str,
    name: Option<&str>,
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
    let language = greppy_parser::language_for_path(file);
    let Some(line) = import_line(language, module, name) else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            options,
        ));
    };
    let Some(scan) = scan_imports(language, &snapshot.content, module, name) else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            options,
        ));
    };
    if scan.satisfied {
        return Ok(single_status_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::AlreadySatisfied,
            1,
            options,
        ));
    }
    if scan.conflict.is_some() {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::InvalidResult,
            options,
        ));
    }
    let mut block = line.into_bytes();
    block.push(b'\n');
    let ops = vec![PlannedOp {
        id: "ensure-import".into(),
        range: (scan.insert_at, scan.insert_at),
        replacement: block,
    }];
    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::TreeSitter,
        SelectorClass::Structural,
        Some(language),
        options,
    )
}

/// `greppy edit remove-if-present --symbol SYM`: delete when present,
/// already-satisfied when absent - never an error for a missing target.
pub fn remove_if_present(
    workspace_root: &Path,
    resolved: Option<(std::path::PathBuf, (usize, usize))>,
    options: &VerbOptions,
) -> Result<Certificate> {
    match resolved {
        None => {
            let dummy = Snapshot {
                path: workspace_root.to_path_buf(),
                content: Vec::new(),
                file_sha256: String::new(),
            };
            Ok(single_refusal_certificate(
                workspace_root,
                &dummy,
                SelectorEngine::Symbol,
                SelectorClass::Resolved,
                Status::AlreadySatisfied,
                options,
            ))
        }
        Some((file, range)) => {
            let language = greppy_parser::language_for_path(&file);
            crate::verbs::delete_span(workspace_root, &file, range, Some(language), options)
        }
    }
}

/// `greppy edit ensure-annotation --symbol SYM --annotation A`: idempotent
/// decorator/attribute line directly above a definition.
pub fn ensure_annotation(
    workspace_root: &Path,
    file: &Path,
    def_range: (usize, usize),
    annotation: &str,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    if let Some(certificate) = planned_precondition_refusal(workspace_root, &snapshot, options) {
        return Ok(certificate);
    }
    let language = greppy_parser::language_for_path(file);
    if def_range.0 >= def_range.1 || def_range.1 > snapshot.content.len() {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::NotFound,
            options,
        ));
    }
    let line = annotation.trim();
    // indentation of the definition line
    let def_line_start = snapshot.content[..def_range.0]
        .iter()
        .rposition(|&b| b == b'\n')
        .map(|i| i + 1)
        .unwrap_or(0);
    let indent: Vec<u8> = snapshot.content[def_line_start..def_range.0]
        .iter()
        .copied()
        .take_while(|b| b.is_ascii_whitespace())
        .collect();
    // already present? scan the contiguous annotation block above
    let mut scan_end = def_line_start;
    loop {
        if scan_end == 0 {
            break;
        }
        let prev_start = snapshot.content[..scan_end - 1]
            .iter()
            .rposition(|&b| b == b'\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prev_line = String::from_utf8_lossy(&snapshot.content[prev_start..scan_end - 1])
            .trim()
            .to_string();
        if prev_line.starts_with('@') || prev_line.starts_with("#[") {
            if prev_line == line {
                return Ok(single_status_certificate(
                    workspace_root,
                    &snapshot,
                    SelectorEngine::Symbol,
                    SelectorClass::Resolved,
                    Status::AlreadySatisfied,
                    1,
                    options,
                ));
            }
            scan_end = prev_start;
        } else {
            break;
        }
    }
    let mut block = indent.clone();
    block.extend_from_slice(line.as_bytes());
    block.push(b'\n');
    let ops = vec![PlannedOp {
        id: "ensure-annotation".into(),
        range: (def_line_start, def_line_start),
        replacement: block,
    }];
    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
        Some(language),
        options,
    )
}

fn method_state_and_insertion(
    language: Language,
    content: &[u8],
    class_range: (usize, usize),
    method_name: &str,
) -> Option<(bool, usize)> {
    if class_range.0 >= class_range.1 || class_range.1 > content.len() {
        return None;
    }
    let tree = greppy_parser::parse(language, content).ok()?;
    let class_bytes = &content[class_range.0..class_range.1];
    let query_start = class_bytes
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .map(|offset| class_range.0 + offset)?;
    let query_end = class_bytes
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|offset| class_range.0 + offset + 1)?;
    let mut scope = tree
        .root_node()
        .descendant_for_byte_range(query_start, query_end.saturating_sub(1))?;
    let body = loop {
        if let Some(body) = scope.child_by_field_name("body") {
            if body.start_byte() >= class_range.0 && body.end_byte() <= class_range.1 {
                break body;
            }
        }
        scope = scope.parent()?;
    };

    let mut found = false;
    let mut cursor = body.walk();
    let mut reached_body = false;
    while !reached_body {
        let node = cursor.node();
        let is_method = node != body
            && (node.kind().contains("function") || node.kind().contains("method"));
        if is_method
            && node.child_by_field_name("name").is_some_and(|name| {
                &content[name.start_byte()..name.end_byte()] == method_name.as_bytes()
            })
        {
            found = true;
            break;
        }
        // A nested definition is not a class method. Once a direct method node
        // is reached, skip its subtree before looking for the next sibling.
        if !is_method && cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                reached_body = true;
                break;
            }
        }
    }

    let body_bytes = &content[body.start_byte()..body.end_byte()];
    let insert_at = body_bytes
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .filter(|offset| body_bytes[*offset] == b'}')
        .map(|offset| body.start_byte() + offset)
        .unwrap_or(body.end_byte());
    Some((found, insert_at))
}

/// `greppy edit ensure-method --symbol CLASS`: append a method to a class
/// body when no method of that name exists; present -> already-satisfied.
pub fn ensure_method(
    workspace_root: &Path,
    file: &Path,
    class_range: (usize, usize),
    method_name: &str,
    method_source: &str,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    if let Some(certificate) = planned_precondition_refusal(workspace_root, &snapshot, options) {
        return Ok(certificate);
    }
    let language = greppy_parser::language_for_path(file);
    let Some((found, insert_at)) =
        method_state_and_insertion(language, &snapshot.content, class_range, method_name)
    else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::NotFound,
            options,
        ));
    };
    if found {
        return Ok(single_status_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::Symbol,
            SelectorClass::Resolved,
            Status::AlreadySatisfied,
            1,
            options,
        ));
    }
    let mut block = Vec::new();
    block.push(b'\n');
    block.extend_from_slice(method_source.as_bytes());
    if !method_source.ends_with('\n') {
        block.push(b'\n');
    }
    let ops = vec![PlannedOp {
        id: "ensure-method".into(),
        range: (insert_at, insert_at),
        replacement: block,
    }];
    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::Symbol,
        SelectorClass::Resolved,
        Some(language),
        options,
    )
}

/// `greppy edit ensure-argument --symbol SYM --call NAME --arg TEXT`:
/// within one resolved definition, append `arg_text` to every call of
/// `callee` that does not already contain it (token-level check).
/// All calls already carrying the argument -> already-satisfied.
pub fn ensure_argument(
    workspace_root: &Path,
    file: &Path,
    def_range: (usize, usize),
    callee: &str,
    arg_text: &str,
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
    let language = greppy_parser::language_for_path(file);
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
    let Some(call_args) = call_argument_spans(language, &snapshot.content, def_range, callee)
    else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            options,
        ));
    };
    if call_args.is_empty() {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            options,
        ));
    }
    let mut ops = Vec::new();
    for (i, (args_start, args_end)) in call_args.iter().enumerate() {
        let args_text = String::from_utf8_lossy(&snapshot.content[*args_start..*args_end]);
        let inner = args_text.trim_start_matches('(').trim_end_matches(')');
        let already = inner.split(',').any(|a| a.trim() == arg_text.trim());
        if already {
            continue;
        }
        // vor der schliessenden klammer einfuegen
        let insert_at = args_end - 1;
        let sep = if inner.trim().is_empty() { "" } else { ", " };
        ops.push(PlannedOp {
            id: format!("ensure-argument-{i}"),
            range: (insert_at, insert_at),
            replacement: format!("{sep}{arg_text}").into_bytes(),
        });
    }
    if ops.is_empty() {
        return Ok(single_status_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::AlreadySatisfied,
            call_args.len(),
            options,
        ));
    }
    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::TreeSitter,
        SelectorClass::Structural,
        Some(language),
        options,
    )
}

/// Argument-list spans `(...)` of every call to `callee` within `def_range`.
fn call_argument_spans(
    language: Language,
    content: &[u8],
    def_range: (usize, usize),
    callee: &str,
) -> Option<Vec<(usize, usize)>> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let mut out = Vec::new();
    let mut cursor = tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = cursor.node();
        if node.start_byte() >= def_range.0
            && node.end_byte() <= def_range.1
            && node.kind().contains("call")
        {
            let fn_node = node
                .child_by_field_name("function")
                .or_else(|| node.child(0));
            let args_node = node.child_by_field_name("arguments");
            if let (Some(f), Some(a)) = (fn_node, args_node) {
                let name = &content[f.start_byte()..f.end_byte()];
                // qualifizierte callees: letztes segment vergleichen
                let short = name
                    .rsplit(|&b| b == b'.' || b == b':')
                    .next()
                    .unwrap_or(name);
                if short == callee.as_bytes() {
                    out.push((a.start_byte(), a.end_byte()));
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
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn ensure_argument_appends_once() {
        let dir = ws();
        let f = dir.path().join("m.py");
        let content = b"def run():\n    fetch(url)\n    fetch(url, timeout=30)\n";
        std::fs::write(&f, content).unwrap();
        let cert = ensure_argument(
            dir.path(),
            &f,
            (0, content.len()),
            "fetch",
            "timeout=30",
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert_eq!(out.matches("timeout=30").count(), 2, "{out}");
        // zweiter lauf: alles versorgt
        let content2 = std::fs::read(&f).unwrap();
        let cert = ensure_argument(
            dir.path(),
            &f,
            (0, content2.len()),
            "fetch",
            "timeout=30",
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
    }

    #[test]
    fn ensure_annotation_idempotent() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"def run():\n    pass\n").unwrap();
        let cert =
            ensure_annotation(dir.path(), &f, (0, 19), "@retry", &VerbOptions::default()).unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.starts_with("@retry\ndef run():"), "{out}");
        // zweiter lauf: bereits vorhanden (def_range hat sich verschoben)
        let content = std::fs::read(&f).unwrap();
        let def_start = out.find("def run").unwrap();
        let cert = ensure_annotation(
            dir.path(),
            &f,
            (def_start, content.len() - 1),
            "@retry",
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
    }

    #[test]
    fn remove_if_present_absent_is_satisfied() {
        let dir = ws();
        let cert = remove_if_present(dir.path(), None, &VerbOptions::default()).unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
        assert_eq!(cert.exit_code(), 0);
    }

    #[test]
    fn python_insert_after_last_import() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"import os\n\ndef run():\n    pass\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "auth.validators",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(
            out.starts_with("import os\nfrom auth.validators import validate\n"),
            "{out}"
        );
    }

    #[test]
    fn second_run_already_satisfied() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"from auth.validators import validate\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "auth.validators",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
        assert_eq!(cert.exit_code(), 0);
    }

    #[test]
    fn conflicting_binding_refuses() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"from other.module import validate\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "auth.validators",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::InvalidResult);
        assert!(std::fs::read_to_string(&f)
            .unwrap()
            .starts_with("from other.module"));
    }

    #[test]
    fn rust_use_insertion() {
        let dir = ws();
        let f = dir.path().join("m.rs");
        std::fs::write(&f, b"use std::io::Write;\n\nfn main() {}\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "crate::auth",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("use crate::auth::validate;\n"), "{out}");
        assert_eq!(cert.operations[0].syntax.new_errors, 0);
    }

    #[test]
    fn file_without_imports_inserts_at_top() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"def run():\n    pass\n").unwrap();
        let cert = ensure_import(dir.path(), &f, "os", None, &VerbOptions::default()).unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(std::fs::read_to_string(&f)
            .unwrap()
            .starts_with("import os\n"));
    }
}
