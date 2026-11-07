//! Actual CLI source previews: bounded display, unchanged matching and recovery.
#![cfg(unix)]
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

struct Fixture {
    root: PathBuf,
    repo: PathBuf,
    store: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "greppy-source-preview-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&root).unwrap();
        let repo = root.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let fixture = Self {
            store: root.join("store"),
            root,
            repo,
        };
        std::fs::write(fixture.repo.join("lib.rs"), "pub fn fixture_anchor() {}\n").unwrap();
        let indexed = fixture.run(&["index", "."]);
        assert_eq!(
            indexed.status.code(),
            Some(0),
            "fixture index: {}",
            String::from_utf8_lossy(&indexed.stderr)
        );
        fixture
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(bin())
            .args(args)
            .current_dir(&self.repo)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_AUTO_REINDEX", "0")
            .output()
            .expect("run greppy")
    }
}

#[test]
fn long_json_match_has_bounded_preview_and_executable_full_line_recovery() {
    let fixture = Fixture::new();
    let file = fixture.repo.join("payload 'quoted.json");
    let source = format!(
        "{{\"payload\":\"{}parse_selector{}\"}}\n",
        "x".repeat(100_000),
        "🙂".repeat(25_000)
    );
    std::fs::write(&file, &source).unwrap();
    for (query, extra) in [
        ("parse_selector", None),
        ("parse_selector|validate.*selector", None),
        ("parse_selector", Some("--fixed")),
        ("parse_selector", Some("--all")),
    ] {
        let mut args = vec!["search-pattern", query, "--code", "--limit", "1"];
        if let Some(extra) = extra {
            args.push(extra);
        }
        let output = fixture.run(&args);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.stdout.len() < 6000,
            "unbounded preview: {} bytes",
            output.stdout.len()
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("source preview:"));
        assert!(
            text.contains("parse_selector"),
            "middle match must remain visible"
        );
        let recovery = text
            .lines()
            .find_map(|line| line.strip_prefix("full source line (current file): "))
            .unwrap();
        let private_bin = fixture.root.join("bin");
        if !private_bin.exists() {
            std::fs::create_dir(&private_bin).unwrap();
            std::os::unix::fs::symlink(bin(), private_bin.join("greppy")).unwrap();
        }
        let recovered = Command::new("sh")
            .args(["-c", recovery])
            .current_dir(&fixture.repo)
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    private_bin.display(),
                    std::env::var("PATH").unwrap_or_default()
                ),
            )
            .env("GREPPY_STORE_DIR", &fixture.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_AUTO_REINDEX", "0")
            .output()
            .unwrap();
        assert_eq!(
            recovered.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&recovered.stderr)
        );
        assert!(String::from_utf8(recovered.stdout)
            .unwrap()
            .contains(source.trim_end()));
        assert_eq!(std::fs::read(&file).unwrap(), source.as_bytes());
    }
}

#[test]
fn short_fixed_match_is_unchanged_and_no_match_stays_nonzero() {
    let fixture = Fixture::new();
    std::fs::write(fixture.repo.join("short.txt"), "literal[marker]\n").unwrap();
    let output = fixture.run(&[
        "search-pattern",
        "literal[marker]",
        "--fixed",
        "--code",
        "--limit",
        "1",
    ]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "short.txt:1\nliteral[marker]\n"
    );
    let missing = fixture.run(&["search-pattern", "not_present", "--code"]);
    assert_eq!(missing.status.code(), Some(1));
}

fn page_headers(output: &Output) -> Vec<String> {
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.starts_with("page") && line.contains(".rs:"))
        .map(|line| line.split_whitespace().next().unwrap().to_owned())
        .collect()
}

fn page_fixture() -> Fixture {
    let fixture = Fixture::new();
    for file in 0..6 {
        let source = (0..2)
            .map(|line| {
                format!(
                    "pub fn FMA_CODEC_VP9_{file}_{line}() -> i32 {{ {} }}\n",
                    file * 2 + line
                )
            })
            .collect::<String>();
        std::fs::write(fixture.repo.join(format!("page{file}.rs")), source).unwrap();
    }
    fixture
}

#[test]
fn text_pattern_offset_pages_complete_matches_without_repeats() {
    let fixture = page_fixture();
    for code in [false, true] {
        let mut first = vec!["search-pattern", "FMA_CODEC_VP9", "--limit", "6"];
        if code {
            first.push("--code");
        }
        let initial = fixture.run(&first);
        let first_headers = page_headers(&initial);
        assert_eq!(
            first_headers,
            [
                "page0.rs:1",
                "page0.rs:2",
                "page1.rs:1",
                "page1.rs:2",
                "page2.rs:1",
                "page2.rs:2"
            ]
        );
        let mut second = first.clone();
        second.extend(["--offset", "6"]);
        let next = fixture.run(&second);
        let second_headers = page_headers(&next);
        assert_eq!(
            second_headers,
            [
                "page3.rs:1",
                "page3.rs:2",
                "page4.rs:1",
                "page4.rs:2",
                "page5.rs:1",
                "page5.rs:2"
            ]
        );
        let text = String::from_utf8(next.stdout).unwrap();
        assert!(
            text.contains("shown: 6\ntotal: 12\noffset: 6\ntruncated: false"),
            "{text}"
        );
        let mut beyond = first.clone();
        beyond.extend(["--offset", "20"]);
        let empty = fixture.run(&beyond);
        assert!(page_headers(&empty).is_empty());
        assert!(String::from_utf8(empty.stdout)
            .unwrap()
            .contains("shown: 0\ntotal: 12\noffset: 20\ntruncated: false"));
    }
}

#[test]
fn text_pattern_byte_budget_advances_by_complete_matches() {
    let fixture = page_fixture();
    let first = fixture.run(&[
        "search-pattern",
        "FMA_CODEC_VP9",
        "--code",
        "--limit",
        "6",
        "--max-bytes",
        "500",
    ]);
    let headers = page_headers(&first);
    assert!(!headers.is_empty() && headers.len() < 6);
    let text = String::from_utf8(first.stdout).unwrap();
    assert!(
        text.contains(&format!(
            "shown: {}\ntotal: 12\noffset: 0\ntruncated: true",
            headers.len()
        )),
        "{text}"
    );
    assert!(
        text.contains(&format!("--offset {}", headers.len())),
        "{text}"
    );
    assert!(
        !text.contains("showing 6"),
        "pre-budget count must not survive: {text}"
    );
    let next = fixture.run(&[
        "search-pattern",
        "FMA_CODEC_VP9",
        "--code",
        "--limit",
        "6",
        "--max-bytes",
        "500",
        "--offset",
        &headers.len().to_string(),
    ]);
    let next_headers = page_headers(&next);
    assert!(!next_headers.is_empty());
    assert!(next_headers.iter().all(|header| !headers.contains(header)));
}

#[test]
fn text_pattern_all_preserves_explicit_limit_and_offset() {
    let fixture = page_fixture();
    let output = fixture.run(&[
        "search-pattern",
        "FMA_CODEC_VP9",
        "--all",
        "--code",
        "--limit",
        "2",
        "--offset",
        "2",
    ]);
    assert_eq!(page_headers(&output), ["page1.rs:1", "page1.rs:2"]);
}

#[test]
fn code_reads_rust_python_definitions_without_repeating_bodies() {
    let fixture = Fixture::new();
    let rust_first = "pub fn decode_token(value: usize) -> usize {\n    // repeated_source_marker\n    value + 7 // repeated_source_marker\n}";
    let rust_second = "pub fn next_token() -> usize {\n    91\n}";
    std::fs::write(
        fixture.repo.join("source.rs"),
        format!("{rust_first}\n{rust_second}\n"),
    )
    .unwrap();
    let rust = fixture.run(&[
        "search-pattern",
        "^pub fn (decode_token|next_token)",
        "--code",
        "--path",
        "source.rs",
    ]);
    assert_eq!(
        rust.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&rust.stderr)
    );
    let text = String::from_utf8(rust.stdout).unwrap();
    assert!(
        text.contains(rust_first) && text.contains(rust_second),
        "definition body must be present, not only the matching signature: {text}"
    );

    let python_source = "def decode_items(value):\n    python_body_marker = list(value)\n    return python_body_marker\n";
    std::fs::write(fixture.repo.join("source.py"), python_source).unwrap();
    let python = fixture.run(&[
        "search-pattern",
        "^def decode_items",
        "--code",
        "--path",
        "source.py",
    ]);
    assert_eq!(
        python.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&python.stderr)
    );
    let text = String::from_utf8(python.stdout).unwrap();
    assert!(
        text.contains(python_source),
        "Python definition body must be present: {text}"
    );

    let repeated = fixture.run(&[
        "search-pattern",
        "repeated_source_marker",
        "--code",
        "--path",
        "source.rs",
    ]);
    assert_eq!(repeated.status.code(), Some(0));
    let text = String::from_utf8(repeated.stdout).unwrap();
    assert_eq!(
        text.matches("    value + 7 // repeated_source_marker")
            .count(),
        1,
        "definition body must not be repeated for multiple literal hits: {text}"
    );
    assert!(
        text.contains("definition source already shown"),
        "later matches must explain source reuse: {text}"
    );
    let compact = fixture.run(&[
        "search-pattern",
        "^pub fn decode_token",
        "--path",
        "source.rs",
    ]);
    assert_eq!(compact.status.code(), Some(0));
    assert!(!String::from_utf8(compact.stdout)
        .unwrap()
        .contains("value + 7"));

    let wide_source = format!(
        "pub fn wide() -> &'static str {{\n    \"{}middle_marker{}\"\n}}\n",
        "x".repeat(100_000),
        "y".repeat(100_000)
    );
    std::fs::write(fixture.repo.join("wide.rs"), &wide_source).unwrap();
    let wide = fixture.run(&[
        "search-pattern",
        "middle_marker",
        "--code",
        "--path",
        "wide.rs",
    ]);
    assert_eq!(wide.status.code(), Some(0));
    let text = String::from_utf8(wide.stdout).unwrap();
    assert!(
        text.len() < 6000 && text.contains("middle_marker") && text.contains("source preview:"),
        "wide definition must remain bounded and explicitly partial"
    );
    assert!(
        text.contains("--lines 2:2"),
        "wide body recovery must point to the clipped body line: {text}"
    );
    assert_eq!(
        std::fs::read_to_string(fixture.repo.join("wide.rs")).unwrap(),
        wide_source
    );
}
