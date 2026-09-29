//! Ordinary web output is compact and losslessly continuable by default.

use std::process::{Command, Output};

fn run(
    root: &std::path::Path,
    runtime: &std::path::Path,
    args: &[&str],
    view: Option<&str>,
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_greppy"));
    command
        .args(args)
        .current_dir(root)
        .env("GREPPY_WEB_RUNTIME_DIR", runtime)
        .env("GREPPY_RUNTIME_DIR", runtime);
    match view {
        Some(value) => {
            command.env("GREPPY_WEB_VIEW", value);
        }
        None => {
            command.env_remove("GREPPY_WEB_VIEW");
        }
    }
    command.output().expect("run greppy web command")
}

fn page_content(output: &str) -> &str {
    output
        .split_once("UNTRUSTED_PAGE_CONTENT\n")
        .and_then(|(_, tail)| tail.split_once("\nEND_UNTRUSTED_PAGE_CONTENT\n"))
        .map(|(content, _)| content)
        .expect("bounded untrusted content")
}

fn continuation(output: &str) -> Option<String> {
    let marker = "next: greppy web result next '";
    let tail = output.split_once(marker)?.1;
    Some(tail.split_once('\'')?.0.to_owned())
}

#[test]
fn default_human_web_view_bounds_large_reply_and_continues_losslessly() {
    let root = tempfile::tempdir().unwrap();
    let runtime = root.path().join("runtime");
    let source = format!("const payload = '{}TAIL';\n", "x".repeat(60_000));
    let source_path = root.path().join("large.mjs");
    std::fs::write(&source_path, &source).unwrap();
    let source_arg = source_path.to_str().unwrap();

    let saved = run(
        root.path(),
        &runtime,
        &["web", "script", "save", "large", "--file", source_arg],
        None,
    );
    assert!(
        saved.status.success(),
        "{}",
        String::from_utf8_lossy(&saved.stderr)
    );

    let shown = run(
        root.path(),
        &runtime,
        &["web", "script", "show", "large"],
        None,
    );
    assert!(
        shown.status.success(),
        "{}",
        String::from_utf8_lossy(&shown.stderr)
    );
    let shown = String::from_utf8(shown.stdout).unwrap();
    assert!(
        shown.len() <= 8192,
        "default output was {} bytes",
        shown.len()
    );
    assert!(
        shown.starts_with("returned — task outcome not verified\n"),
        "{shown}"
    );
    assert!(shown.contains("operation=\"web.script.show\"\n"), "{shown}");
    assert!(shown.contains("UNTRUSTED_PAGE_CONTENT\n"), "{shown}");
    assert!(shown.contains("END_UNTRUSTED_PAGE_CONTENT\n"), "{shown}");
    assert!(
        !shown.contains("TAIL"),
        "large reply unexpectedly fit first page"
    );

    let json_default = run(
        root.path(),
        &runtime,
        &["web", "script", "show", "large", "--json"],
        None,
    );
    let json_raw = run(
        root.path(),
        &runtime,
        &["web", "script", "show", "large", "--json"],
        Some("raw"),
    );
    assert_eq!(json_default.status.code(), Some(0));
    assert_eq!(
        json_default.stdout, json_raw.stdout,
        "--json bytes changed with view mode"
    );
    let envelope: serde_json::Value = serde_json::from_slice(&json_default.stdout).unwrap();
    assert_eq!(envelope["result"]["source"], source);

    let raw = run(
        root.path(),
        &runtime,
        &["web", "script", "show", "large"],
        Some("raw"),
    );
    assert!(raw.status.success());
    assert!(raw.stdout.len() > 50_000);
    assert!(!String::from_utf8_lossy(&raw.stdout).contains("UNTRUSTED_PAGE_CONTENT"));

    let expected = format!("{}\n", envelope["result"]);
    let mut restored = page_content(&shown).to_owned();
    let mut cursor = continuation(&shown);
    while let Some(next) = cursor {
        let page = run(
            root.path(),
            &runtime,
            &["web", "result", "next", &next],
            None,
        );
        assert!(
            page.status.success(),
            "{}",
            String::from_utf8_lossy(&page.stderr)
        );
        let page = String::from_utf8(page.stdout).unwrap();
        assert!(page.len() <= 8192);
        assert!(page.starts_with("returned — task outcome not verified\n"));
        restored.push_str(page_content(&page));
        cursor = continuation(&page);
    }
    assert_eq!(restored, expected);
}
