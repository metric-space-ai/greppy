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
        assert!(
            String::from_utf8(recovered.stdout)
                .unwrap()
                .contains(source.trim_end())
        );
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
        assert!(
            String::from_utf8(empty.stdout)
                .unwrap()
                .contains("shown: 0\ntotal: 12\noffset: 20\ntruncated: false")
        );
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
fn explicit_text_pattern_limit_overrides_large_result_preview() {
    let fixture = Fixture::new();
    let source = (0..32)
        .map(|index| format!("pub fn LIMIT_TOKEN_{index}() {{}}\n"))
        .collect::<String>();
    std::fs::write(fixture.repo.join("page.rs"), source).unwrap();
    for code in [false, true] {
        let mut args = vec!["search-pattern", "LIMIT_TOKEN", "--limit", "28"];
        if code {
            args.push("--code");
        }
        let first = fixture.run(&args);
        let headers = page_headers(&first);
        assert_eq!(headers.len(), 28);
        assert_eq!(headers.first().unwrap(), "page.rs:1");
        assert_eq!(headers.last().unwrap(), "page.rs:28");
        let text = String::from_utf8(first.stdout).unwrap();
        assert!(
            text.contains("shown: 28\ntotal: 32\noffset: 0\ntruncated: true"),
            "{text}"
        );
        args.extend(["--offset", "28"]);
        let next = fixture.run(&args);
        assert_eq!(
            page_headers(&next),
            ["page.rs:29", "page.rs:30", "page.rs:31", "page.rs:32"]
        );
        let text = String::from_utf8(next.stdout).unwrap();
        assert!(
            text.contains("shown: 4\ntotal: 32\noffset: 28\ntruncated: false"),
            "{text}"
        );
    }
    let default = fixture.run(&["search-pattern", "LIMIT_TOKEN"]);
    assert_eq!(page_headers(&default).len(), 5);
}
