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
