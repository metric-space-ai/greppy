//! Cross-verb × cross-scenario grid test for the four edit verbs that
//! operate on resolved byte ranges in Java source files:
//! `replace-body`, `insert-after`, `insert-before`, and `delete`.
//!
//! Layout: 4 verbs × 4 scenarios = 16 tests, one per cell, named
//! `grid_java_<verb>_<scenario>`.
//!
//!   - **unique**: Applied, exit 0, certificate published, file changed.
//!   - **ambiguous**: refusal with the closest contract exit code, file
//!     unchanged. `replace-body` maps body-resolution failure to
//!     NotFound (10) — see NOTES-grid-java.md for the 10-vs-11 mapping.
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
        std::path::Path::new("m.java"),
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

/// Byte range covering a Java `class` or method definition. A class range
/// extends through the separator before the next top-level class (or to the
/// end of content); a method range ends at its matching closing brace. Used
/// as the `def_range` for verbs that operate on one resolved Java definition.
fn java_def_range(content: &[u8], name: &str) -> (usize, usize) {
    let text = std::str::from_utf8(content).expect("utf-8 source");
    let class_needle = format!("class {name}");
    let void_needle = format!("void {name}(");
    let int_needle = format!("int {name}(");
    let start = text
        .find(&class_needle)
        .or_else(|| text.find(&void_needle))
        .or_else(|| text.find(&int_needle))
        .unwrap_or_else(|| panic!("missing Java definition `{name}`"));
    let body_open = text[start..]
        .find('{')
        .map(|i| start + i)
        .unwrap_or_else(|| panic!("missing open brace for `{name}`"));
    if text[start..].starts_with("class ") {
        let end = text[start + 1..]
            .find("\nclass ")
            .map(|offset| start + 1 + offset)
            .unwrap_or(content.len());
        return (start, end);
    }
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
    assert!(depth == 0, "unbalanced braces in Java definition `{name}`");
    (start, end)
}

// =======================================================================
// UNIQUE
// =======================================================================

#[test]
fn grid_java_replace_body_unique() {
    let ws = workspace();
    let content = b"class MathOps {\n    int add(int a, int b) {\n        return a + b;\n    }\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = java_def_range(content, "add");

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"{\n        return a - b;\n    }",
        Language::Java,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);
    assert_eq!(cert.operations.len(), 1);

    let out = std::fs::read_to_string(&file).unwrap();
    let signature_end = content[def_range.0..]
        .iter()
        .position(|&b| b == b'{')
        .map(|offset| def_range.0 + offset + 1)
        .unwrap();
    assert_eq!(
        &out.as_bytes()[..signature_end],
        &content[..signature_end],
        "signature must be preserved"
    );
    assert!(out.contains("int add(int a, int b) {"));
    assert!(out.contains("return a - b;"));
    assert!(!out.contains("return a + b;"));

    let json = serde_json::to_value(&cert).unwrap();
    assert_eq!(json["schema_version"].as_str(), Some("greppy.edit-certificate.v1"));
    assert_eq!(json["status"].as_str(), Some("applied"));
    let _: Certificate = serde_json::from_value(json).unwrap();

    let op = &cert.operations[0];
    assert!(op.outside_declared_ranges_unchanged);
    assert!(op.unified_diff.is_some());
}

#[test]
fn grid_java_insert_after_unique() {
    let ws = workspace();
    let content = b"import java.util.List;\n\nclass Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = java_def_range(content, "Foo");

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"class Bar {\n}",
        InsertPosition::After,
        Some(Language::Java),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(out.contains("class Foo {\n}"));
    assert!(out.contains("class Bar {\n}"));
    let foo_pos = out.find("class Foo {").unwrap();
    let bar_pos = out.find("class Bar {").unwrap();
    assert!(bar_pos > foo_pos);
    let foo_marker = "class Foo {\n}";
    let foo_end = out.find(foo_marker).unwrap() + foo_marker.len();
    assert_eq!(&out[foo_end..bar_pos], "\n\n");
}

#[test]
fn grid_java_insert_before_unique() {
    let ws = workspace();
    let content = b"import java.nio.file.Path;\n\nclass Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = java_def_range(content, "Foo");

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"class Bar {\n}",
        InsertPosition::Before,
        Some(Language::Java),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(out.contains("class Foo {\n}"));
    assert!(out.contains("class Bar {\n}"));
    let bar_marker = "class Bar {\n}";
    let bar_end = out.find(bar_marker).unwrap() + bar_marker.len();
    let foo_pos = out.find("class Foo {").unwrap();
    assert!(foo_pos > bar_end);
    assert_eq!(&out[bar_end..foo_pos], "\n\n");
}

#[test]
fn grid_java_delete_unique() {
    let ws = workspace();
    let content = b"import java.util.List;\n\nclass Foo {\n}\n\nclass Bar {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = java_def_range(content, "Foo");

    let cert = delete_span(
        ws.path(),
        &file,
        def_range,
        Some(Language::Java),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::Applied);
    assert_eq!(cert.exit_code(), 0);
    assert!(cert.published);

    let out = std::fs::read_to_string(&file).unwrap();
    assert!(!out.contains("class Foo"));
    assert!(out.contains("class Bar {\n}"));
    assert!(out.starts_with("import java.util.List;\nclass Bar {"));
}

// =======================================================================
// AMBIGUOUS
// =======================================================================

#[test]
fn grid_java_replace_body_ambiguous() {
    let ws = workspace();
    let content = b"// this is a leading comment, not a Java definition\n\nclass Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_start = std::str::from_utf8(content).unwrap().find("class Foo").unwrap();
    let def_range = (0usize, def_start);

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"{\n        int newBody = 1;\n    }",
        Language::Java,
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
fn grid_java_insert_after_ambiguous() {
    let ws = workspace();
    let content = b"class Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = (0usize, content.len() + 50);

    let result = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"class Bar {\n}",
        InsertPosition::After,
        Some(Language::Java),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_java_insert_before_ambiguous() {
    let ws = workspace();
    let content = b"class Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = (content.len() + 50, content.len() + 50);

    let result = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"class Bar {\n}",
        InsertPosition::Before,
        Some(Language::Java),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

#[test]
fn grid_java_delete_ambiguous() {
    let ws = workspace();
    let content = b"class Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = (0usize, content.len() + 50);

    let result = delete_span(
        ws.path(),
        &file,
        def_range,
        Some(Language::Java),
        &VerbOptions::default(),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), content);
}

// =======================================================================
// STALE — resolution hashes captured before the concurrent mutation
// =======================================================================

#[test]
fn grid_java_replace_body_stale() {
    let ws = workspace();
    let content = b"class Foo {\n    int foo() {\n        return 1;\n    }\n}\n";
    let file = write(ws.path(), "m.java", content);
    let planned_range = java_def_range(content, "foo");
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"class Foo {\n    int foo() {\n        return 999;\n    }\n    void other() {}\n}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = replace_body(
        ws.path(),
        &file,
        planned_range,
        b"{\n        return 99;\n    }",
        Language::Java,
        &options,
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
fn grid_java_insert_after_stale() {
    let ws = workspace();
    let content = b"class Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let planned_range = java_def_range(content, "Foo");
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"class Foo {\n    // user edit between plan and apply\n}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = insert_adjacent(
        ws.path(),
        &file,
        planned_range,
        b"class Bar {\n}",
        InsertPosition::After,
        Some(Language::Java),
        &options,
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
fn grid_java_insert_before_stale() {
    let ws = workspace();
    let content = b"class Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let planned_range = java_def_range(content, "Foo");
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"class Foo {\n    // user edit between plan and apply\n}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = insert_adjacent(
        ws.path(),
        &file,
        planned_range,
        b"class Bar {\n}",
        InsertPosition::Before,
        Some(Language::Java),
        &options,
    )
    .unwrap();

    assert_eq!(cert.status, Status::Stale);
    assert_eq!(cert.exit_code(), 12);
    assert!(!cert.published);
    assert_eq!(std::fs::read(&file).unwrap(), mutated);
}

#[test]
fn grid_java_delete_stale() {
    let ws = workspace();
    let content = b"class Foo {\n}\nclass Bar {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let planned_range = java_def_range(content, "Foo");
    let options = planned_options(ws.path(), content, planned_range);
    let mutated = b"class Foo {\n    // user edit between plan and apply\n}\nclass Bar {\n}\n";
    std::fs::write(&file, mutated).unwrap();

    let cert = delete_span(
        ws.path(),
        &file,
        planned_range,
        Some(Language::Java),
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
fn grid_java_replace_body_syntax_breaking() {
    let ws = workspace();
    let content = b"class Foo {\n    int foo() {\n        return 42;\n    }\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = java_def_range(content, "foo");

    let cert = replace_body(
        ws.path(),
        &file,
        def_range,
        b"{\n        return =;\n    }",
        Language::Java,
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
fn grid_java_insert_after_syntax_breaking() {
    let ws = workspace();
    let content = b"class Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = java_def_range(content, "Foo");

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"class Broken( {\n}",
        InsertPosition::After,
        Some(Language::Java),
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
fn grid_java_insert_before_syntax_breaking() {
    let ws = workspace();
    let content = b"class Foo {\n}\n";
    let file = write(ws.path(), "m.java", content);
    let def_range = java_def_range(content, "Foo");

    let cert = insert_adjacent(
        ws.path(),
        &file,
        def_range,
        b"class Broken( {\n}",
        InsertPosition::Before,
        Some(Language::Java),
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
fn grid_java_delete_syntax_breaking() {
    let ws = workspace();
    let content = b"class Foo {\n    int foo() {\n        return 42;\n    }\n}\n";
    let file = write(ws.path(), "m.java", content);

    let close = content.iter().position(|&b| b == b'}').unwrap();
    let cert = delete_span(
        ws.path(),
        &file,
        (close, close + 1),
        Some(Language::Java),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(cert.status, Status::InvalidResult);
    assert_eq!(cert.exit_code(), 13);
    assert!(!cert.published);
    assert!(!cert.operations[0].postconditions_passed);
    assert_eq!(std::fs::read(&file).unwrap(), content);
}
