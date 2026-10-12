//! Public CLI body replacement must preserve its syntax context, not just parse successfully.
use std::process::Command;

fn replace(source: &str, file: &str, symbol: &str, body: &str) -> (i32, String, String) {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    assert!(Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(root)
        .status()
        .unwrap()
        .success());
    let path = root.join(file);
    std::fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .current_dir(root)
        .args(["replace", symbol, "--body", body])
        .env("GREPPY_STORE_DIR", root.join("store"))
        .env("GREPPY_RUNTIME_DIR", root.join("runtime"))
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_WORKERS", "1")
        .output()
        .unwrap();
    let diagnostics = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (
        output.status.code().unwrap_or(-1),
        std::fs::read_to_string(path).unwrap(),
        diagnostics,
    )
}

#[test]
fn go_inner_body_preserves_signature_comment_and_braces() {
    let source = "package sample\n\n// Value returns one.\nfunc Value() int {\n return 1\n}\n";
    let (code, after, diagnostics) = replace(source, "sample.go", "sample.go::Value", "return 2");
    assert_eq!(code, 0, "{diagnostics}");
    assert_eq!(
        after,
        "package sample\n\n// Value returns one.\nfunc Value() int {return 2}\n"
    );
}

#[test]
fn go_method_inner_body_preserves_receiver() {
    let source = "package sample\ntype Item struct{}\nfunc (i Item) Value() int { return 1 }\n";
    let (code, after, diagnostics) = replace(source, "sample.go", "sample.go::Value", "return 2");
    assert_eq!(code, 0, "{diagnostics}");
    assert_eq!(
        after,
        "package sample\ntype Item struct{}\nfunc (i Item) Value() int {return 2}\n"
    );
}

#[test]
fn whole_go_declaration_is_refused_without_changing_source() {
    let source = "package sample\n\n// Value returns one.\nfunc Value() int {\n return 1\n}\n";
    let (code, after, diagnostics) = replace(
        source,
        "sample.go",
        "sample.go::Value",
        "// Value returns two.\nfunc Value() int { return 2 }",
    );
    assert_eq!(code, 13, "{diagnostics}");
    assert_eq!(after, source, "refusal must leave the exact original bytes");
    assert!(diagnostics.contains("syntax"), "{diagnostics}");
}

#[test]
fn rust_inner_and_complete_blocks_preserve_function_signature() {
    let source = "fn value() -> i32 {\n 1\n}\n";
    for (body, expected) in [
        ("2", "fn value() -> i32 {2}\n"),
        ("{ 2 }", "fn value() -> i32 { 2 }\n"),
    ] {
        let (code, after, diagnostics) = replace(source, "sample.rs", "sample.rs::value", body);
        assert_eq!(code, 0, "{diagnostics}");
        assert_eq!(after, expected);
    }
}
