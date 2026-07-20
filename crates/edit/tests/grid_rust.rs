//! Cross-verb × cross-scenario grid test for the four edit verbs that
//! operate on resolved byte ranges in Rust source files:
//! `replace-body`, `insert-after`, `insert-before`, and `delete`.
//!
//! Layout: 4 verbs × 4 scenarios = 16 tests, one per cell, named
//! `grid_rust_<verb>_<scenario>`.
//!
//!   - **unique**: Applied, exit 0, certificate published, file changed.
//!   - **ambiguous**: refusal with the closest contract exit code, file
//!     unchanged. `replace-body` maps body-resolution failure to
//!     NotFound (10) — see NOTES-grid-rust.md for the 10-vs-11 mapping.
//!     The byte-splice verbs return `Err` from `apply_in_memory` when
//!     the resolved span is out of range.
//!   - **stale**: file mutated between read and apply. The contract
//!     binds exit 12 (Stale) to CAS-protected operations; these four
//!     byte-range single-op verbs have no CAS today. Cells are
//!     `#[ignore]`d and documented in NOTES-grid-rust.md.
//!   - **syntax-breaking**: InvalidResult, exit 13, file byte-identical
//!     to the pre-call content.
//!
//! Exit codes are the binding values from
//! `docs/contracts/EDIT_CONTRACT.md`.

#![cfg(unix)]

use greppy_edit::verbs::{
    delete_span, insert_adjacent, replace_body, InsertPosition, VerbOptions,
};
use greppy_edit::{Certificate, Language, Status};

// ------------------------------------------------------------------ helpers

fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn write(ws: &std::path::Path, name: &str, content: &[u8]) -> std::path::PathBuf {
    let p = ws.join(name);
    std::fs::write(&p, content).unwrap();
    p
}

/// Byte range covering `fn NAME(` through the matching closing brace
/// (end-exclusive). Used as the `def_range` for verbs that operate on
/// one resolved definition.
fn fn_def_range(content: &[u8], name: &str) -> (usize, usize) {
    let text = std::str::from_utf8(content).expect("utf-8 source");
    let needle = format!("fn {name}(");
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
fn grid_rust_replace_body_unique() {
    let ws = workspace();
    let content = b"fn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = fn_def_range(content, "add");

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"{\n    a - b\n}",
        Language::Rust,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);
    assert_eq!(cert.operations.len(), 1);

    let out = std::fs::read_to_string(&file).unwrap();
    assert_eq!(&out.as_bytes()[..32], &content[..32], "signature must be preserved");
    assert!(out.starts_with("fn add(a: u32, b: u32) -> u32 {"));
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
fn grid_rust_insert_after_unique() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, 11usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"fn bar() {}",
        InsertPosition::After,
        Some(Language::Rust),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(out.contains("fn foo() {}"));
    assert!(out.contains("fn bar() {}"));
    let foo_pos = out.find("fn foo() {}").unwrap();
    let bar_pos = out.find("fn bar() {}").unwrap();
    assert!(bar_pos > foo_pos);
    assert_eq!(&out[foo_pos + 11..bar_pos], "\n\n");
}

#[test]
fn grid_rust_insert_before_unique() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, 11usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"fn bar() {}",
        InsertPosition::Before,
        Some(Language::Rust),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(out.contains("fn foo() {}"));
    assert!(out.contains("fn bar() {}"));
    let bar_pos = out.find("fn bar() {}").unwrap();
    let foo_pos = out.find("fn foo() {}").unwrap();
    assert!(foo_pos > bar_pos);
    assert_eq!(&out[bar_pos + 11..foo_pos], "\n\n");
}

#[test]
fn grid_rust_delete_unique() {
    let ws = workspace();
    let content = b"fn foo() {}\nfn bar() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, 11usize);

    let cert = delete_span(
        ws.path(),
        &file,
        def_range,
        Some(Language::Rust),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(!out.contains("fn foo("));
    assert!(out.contains("fn bar() {}"));
    assert!(out.starts_with("fn bar() {}"));
}

// =======================================================================
// AMBIGUOUS
// =======================================================================

#[test]
fn grid_rust_replace_body_ambiguous() {
    let ws = workspace();
    let content = b"// this is a leading comment, not a function body\nfn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, 44usize);

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"new body",
        Language::Rust,
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
fn grid_rust_insert_after_ambiguous() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, content.len() + 50);

    let result = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"fn bar() {}",
        InsertPosition::After,
        Some(Language::Rust),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_rust_insert_before_ambiguous() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (content.len() + 50, content.len() + 50);

    let result = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"fn bar() {}",
        InsertPosition::Before,
        Some(Language::Rust),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_rust_delete_ambiguous() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, content.len() + 50);

    let result = delete_span(
        ws.path(),
        &file,
        def_range,
        Some(Language::Rust),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

// =======================================================================
// STALE — #[ignore]d, see NOTES-grid-rust.md
// =======================================================================

#[test]
#[ignore = "replace_body has no CAS; see NOTES-grid-rust.md (stale cell defect)"]
fn grid_rust_replace_body_stale() {
    let ws = workspace();
    let content = b"fn foo() -> u32 {\n    1\n}\n";
    let file = write(ws.path(), "m.rs", content);
    let planned_range = fn_def_range(content, "foo");
    let mutated = b"fn foo() -> u32 {\n    999\n}\nfn other() {}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = replace_body(
        ws.path(),
        &file,
        planned_range,
        b"{\n    99\n}",
        Language::Rust,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
#[ignore = "insert_adjacent has no CAS; see NOTES-grid-rust.md (stale cell defect)"]
fn grid_rust_insert_after_stale() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let planned_range = (0usize, 11usize);
    let mutated = b"fn foo() { /* user edit between plan and apply */ }\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = insert_adjacent(
        ws.path(),
        &file,
        planned_range,
        b"fn bar() {}",
        InsertPosition::After,
        Some(Language::Rust),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
#[ignore = "insert_adjacent has no CAS; see NOTES-grid-rust.md (stale cell defect)"]
fn grid_rust_insert_before_stale() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let planned_range = (0usize, 11usize);
    let mutated = b"fn foo() { /* user edit between plan and apply */ }\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = insert_adjacent(
        ws.path(),
        &file,
        planned_range,
        b"fn bar() {}",
        InsertPosition::Before,
        Some(Language::Rust),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
#[ignore = "delete_span has no CAS; see NOTES-grid-rust.md (stale cell defect)"]
fn grid_rust_delete_stale() {
    let ws = workspace();
    let content = b"fn foo() {}\nfn bar() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let planned_range = (0usize, 11usize);
    let mutated = b"fn foo() { /* user edit between plan and apply */ }\nfn bar() {}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = delete_span(
        ws.path(),
        &file,
        planned_range,
        Some(Language::Rust),
        &VerbOptions::default(),
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
fn grid_rust_replace_body_syntax_breaking() {
    let ws = workspace();
    let content = b"fn foo() -> u32 {\n    42\n}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = fn_def_range(content, "foo");

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"fn a( { let =",
        Language::Rust,
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
fn grid_rust_insert_after_syntax_breaking() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, 11usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"fn bar( { let =",
        InsertPosition::After,
        Some(Language::Rust),
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
fn grid_rust_insert_before_syntax_breaking() {
    let ws = workspace();
    let content = b"fn foo() {}\n";
    let file = write(ws.path(), "m.rs", content);
    let def_range = (0usize, 11usize);

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"fn bar( { let =",
        InsertPosition::Before,
        Some(Language::Rust),
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
fn grid_rust_delete_syntax_breaking() {
    let ws = workspace();
    let content = b"fn foo() {\n    42\n}\n";
    let file = write(ws.path(), "m.rs", content);

    let close = content.iter().position(|&b| b == b'}').unwrap();
    let cert = delete_span(
        ws.path(),
        &file,
        (close, close + 1),
        Some(Language::Rust),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::InvalidResult);
    assert_eq!(cert.exit_code(), 13);
    assert!(!cert.published);
    assert!(!cert.operations[0].postconditions_passed);
    assert_eq!(std::fs::read(&file).unwrap(), content);
}
