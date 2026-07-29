//! Cross-verb × cross-scenario grid test for the four edit verbs that
//! operate on resolved byte ranges in Go source files:
//! `replace-body`, `insert-after`, `insert-before`, and `delete`.
//!
//! Layout: 4 verbs × 4 scenarios = 16 tests, one per cell, named
//! `grid_go_<verb>_<scenario>`.
//!
//!   - **unique**: Applied, exit 0, certificate published, file changed.
//!   - **ambiguous**: refusal with the closest contract exit code, file
//!     unchanged. `replace-body` maps body-resolution failure to
//!     NotFound (10) — see NOTES-grid-go.md for the 10-vs-11 mapping.
//!     The byte-splice verbs return `Err` from `apply_in_memory` when
//!     the resolved span is out of range.
//!   - **stale**: file mutated between resolution and apply. Resolution-time
//!     file and target hashes bind the call; every verb refuses with exit 12.
//!   - **syntax-breaking**: InvalidResult, exit 13, file byte-identical
//!     to the pre-call content.
//!
//! Exit codes are the binding values from
//! `docs/contracts/EDIT_CONTRACT.md`.

#![cfg(unix)]

use greppy_edit::verbs::{
    delete_span, insert_adjacent, replace_body, InsertPosition, VerbOptions,
};
use greppy_edit::{Certificate, EditHandle, Language, Status};

// ------------------------------------------------------------------ helpers

fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn write(ws: &std::path::Path, name: &str, content: &[u8]) -> std::path::PathBuf {
    let p = ws.join(name);
    std::fs::write(&p, content).unwrap();
    p
}

fn planned_options(
    workspace: &std::path::Path,
    content: &[u8],
    range: (usize, usize),
) -> VerbOptions {
    let handle = EditHandle::for_range(
        workspace,
        std::path::Path::new("m.go"),
        content,
        range.0,
        range.1,
    )
    .unwrap();
    VerbOptions {
        planned_file_sha256: Some(handle.file_sha256),
        planned_target_sha256: Some(handle.target_sha256),
        planned_target_range: Some(range),
        ..Default::default()
    }
}

/// Byte range covering `func NAME(` through the matching closing brace
/// (end-exclusive). Used as the `def_range` for verbs that operate on
/// one resolved Go definition.
fn go_def_range(content: &[u8], name: &str) -> (usize, usize) {
    let text = std::str::from_utf8(content).expect("utf-8 source");
    let needle = format!("func {name}(");
    let start = text
        .find(&needle)
        .unwrap_or_else(|| panic!("missing `{needle}`"));
    let body_open = text[start..]
        .find('{')
        .map(|i| start + i)
        .unwrap_or_else(|| panic!("missing open brace for `{needle}`"));
    let mut depth = 0usize;
    let mut end = body_open;
    for (i, b) in content[body_open..].iter().enumerate() {
        match *b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = body_open + i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(depth == 0, "unbalanced braces in `{needle}`");
    (start, end)
}

// =======================================================================
// UNIQUE
// =======================================================================

#[test]
fn grid_go_replace_body_unique() {
    let ws = workspace();
    let content = b"func add(a, b int) int {\n\treturn a + b\n}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = go_def_range(content, "add");

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"{\n\treturn a - b\n}",
        Language::Go,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);
    assert_eq!(cert.operations.len(), 1);

    let out = std::fs::read_to_string(&file).unwrap();
    let signature_end = content.iter().position(|&b| b == b'{').unwrap() + 1;
    assert_eq!(
        &out.as_bytes()[..signature_end],
        &content[..signature_end],
        "signature must be preserved"
    );
    assert!(out.starts_with("func add(a, b int) int {"));
    assert!(out.contains("a - b"));
    assert!(!out.contains("a + b"));

    let json = serde_json::to_value(&cert).unwrap();
    assert_eq!(json["schema_version"].as_str(), Some("greppy.edit-certificate.v1"));
    assert_eq!(json["status"].as_str(), Some("applied"));
    let _: Certificate = serde_json::from_value(json).unwrap();

    let op = &cert.operations[0];
    assert!(op.outside_declared_ranges_unchanged);
    assert!(op.unified_diff.is_some());
}

#[test]
fn grid_go_insert_after_unique() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, 13usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"func bar() {}",
        InsertPosition::After,
        Some(Language::Go),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(out.contains("func foo() {}"));
    assert!(out.contains("func bar() {}"));
    let foo_pos = out.find("func foo() {}").unwrap();
    let bar_pos = out.find("func bar() {}").unwrap();
    assert!(bar_pos > foo_pos);
    assert_eq!(&out[foo_pos + 13..bar_pos], "\n\n");
}

#[test]
fn grid_go_insert_before_unique() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, 13usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"func bar() {}",
        InsertPosition::Before,
        Some(Language::Go),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(out.contains("func foo() {}"));
    assert!(out.contains("func bar() {}"));
    let bar_pos = out.find("func bar() {}").unwrap();
    let foo_pos = out.find("func foo() {}").unwrap();
    assert!(foo_pos > bar_pos);
    assert_eq!(&out[bar_pos + 13..foo_pos], "\n\n");
}

#[test]
fn grid_go_delete_unique() {
    let ws = workspace();
    let content = b"func foo() {}\nfunc bar() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, 13usize);

    let cert = delete_span(
        ws.path(),
        &file,
        def_range,
        Some(Language::Go),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(!out.contains("func foo("));
    assert!(out.contains("func bar() {}"));
    assert!(out.starts_with("func bar() {}"));
}

// =======================================================================
// AMBIGUOUS
// =======================================================================

#[test]
fn grid_go_replace_body_ambiguous() {
    let ws = workspace();
    let content = b"// this is a leading comment, not a function body\nfunc foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, 53usize);

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"new body",
        Language::Go,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::NotFound);
    assert_eq!(cert.exit_code(), 10);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), content);

    let op = &cert.operations[0];
    assert_eq!(op.target_matches, 0);
    assert!(op.candidates.is_empty());
    let json = serde_json::to_value(&cert).unwrap();
    assert_eq!(json["status"].as_str(), Some("not-found"));
}

#[test]
fn grid_go_insert_after_ambiguous() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, content.len() + 50);

    let result = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"func bar() {}",
        InsertPosition::After,
        Some(Language::Go),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_go_insert_before_ambiguous() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (content.len() + 50, content.len() + 50);

    let result = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"func bar() {}",
        InsertPosition::Before,
        Some(Language::Go),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_go_delete_ambiguous() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, content.len() + 50);

    let result = delete_span(
        ws.path(),
        &file,
        def_range,
        Some(Language::Go),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

// =======================================================================
// STALE — resolution hashes captured before the concurrent mutation
// =======================================================================

#[test]
fn grid_go_replace_body_stale() {
    let ws = workspace();
    let content = b"func foo() int {\n\treturn 1\n}\n";
    let file = write(ws.path(), "m.go", content);
    let planned_range = go_def_range(content, "foo");
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"func foo() int {\n\treturn 999\n}\nfunc other() {}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = replace_body(
        ws.path(),
        &file,
        planned_range,
        b"{\n\treturn 99\n}",
        Language::Go,
        &options,
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
fn grid_go_insert_after_stale() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let planned_range = (0usize, 13usize);
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"func foo() { // user edit between plan and apply\n}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = insert_adjacent(
        ws.path(),
        &file,
        planned_range,
        b"func bar() {}",
        InsertPosition::After,
        Some(Language::Go),
        &options,
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
fn grid_go_insert_before_stale() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let planned_range = (0usize, 13usize);
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"func foo() { // user edit between plan and apply\n}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = insert_adjacent(
        ws.path(),
        &file,
        planned_range,
        b"func bar() {}",
        InsertPosition::Before,
        Some(Language::Go),
        &options,
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
fn grid_go_delete_stale() {
    let ws = workspace();
    let content = b"func foo() {}\nfunc bar() {}\n";
    let file = write(ws.path(), "m.go", content);
    let planned_range = (0usize, 13usize);
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"func foo() { // user edit between plan and apply\n}\nfunc bar() {}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = delete_span(
        ws.path(),
        &file,
        planned_range,
        Some(Language::Go),
        &options,
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

// =======================================================================
// SYNTAX-BREAKING
// =======================================================================

#[test]
fn grid_go_replace_body_syntax_breaking() {
    let ws = workspace();
    let content = b"func foo() int {\n\treturn 42\n}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = go_def_range(content, "foo");

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"func a( {\n\treturn =",
        Language::Go,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::InvalidResult);
    assert_eq!(cert.exit_code(), 13);
    assert!(!cert.published);
    assert!(!cert.operations[0].postconditions_passed);
    assert_eq!(std::fs::read(&file).unwrap(), content);

    let json = serde_json::to_value(&cert).unwrap();
    assert_eq!(json["status"].as_str(), Some("invalid-result"));
}

#[test]
fn grid_go_insert_after_syntax_breaking() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, 13usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"func bar( {\n\treturn =",
        InsertPosition::After,
        Some(Language::Go),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::InvalidResult);
    assert_eq!(cert.exit_code(), 13);
    assert!(!cert.published);
    assert!(!cert.operations[0].postconditions_passed);
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_go_insert_before_syntax_breaking() {
    let ws = workspace();
    let content = b"func foo() {}\n";
    let file = write(ws.path(), "m.go", content);
    let def_range = (0usize, 13usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"func bar( {\n\treturn =",
        InsertPosition::Before,
        Some(Language::Go),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::InvalidResult);
    assert_eq!(cert.exit_code(), 13);
    assert!(!cert.published);
    assert!(!cert.operations[0].postconditions_passed);
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_go_delete_syntax_breaking() {
    let ws = workspace();
    let content = b"func foo() {\n\treturn 42\n}\n";
    let file = write(ws.path(), "m.go", content);

    let close = content.iter().position(|&b| b == b'}').unwrap();
    let cert = delete_span(
        ws.path(),
        &file,
        (close, close + 1),
        Some(Language::Go),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::InvalidResult);
    assert_eq!(cert.exit_code(), 13);
    assert!(!cert.published);
    assert!(!cert.operations[0].postconditions_passed);
    assert_eq!(std::fs::read(&file).unwrap(), content);
}
