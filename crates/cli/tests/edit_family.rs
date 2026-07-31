//! Contract coverage for the trained top-level EDIT family.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

struct Fixture {
    base: PathBuf,
    repo: PathBuf,
    store: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let base = std::env::temp_dir().join(format!(
            "greppy-edit-family-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let store = base.join("store");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        Self { base, repo, store }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(bin());
        command
            .current_dir(&self.repo)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1");
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command()
            .args(args)
            .stdin(Stdio::null())
            .output()
            .expect("run greppy")
    }

    fn run_with_stdin(&self, args: &[&str], stdin: &[u8]) -> Output {
        use std::io::Write as _;

        let mut child = self
            .command()
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn greppy");
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(stdin)
            .expect("write greppy stdin");
        child.wait_with_output().expect("wait for greppy")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn combined(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_file(path: &Path, expected: &str) {
    assert_eq!(std::fs::read_to_string(path).unwrap(), expected);
}

#[test]
fn dry_run_noop_never_claims_applied() {
    let fixture = Fixture::new("dry-run-noop");
    std::fs::write(fixture.repo.join("neu.txt"), "x\n").unwrap();

    let output = fixture.run(&["replace-lines", "neu.txt", "1:1", "x", "--dry-run"]);
    let text = combined(&output);

    assert!(output.status.success(), "{text}");
    assert_eq!(text, "would apply neu.txt:1\n");
    assert!(!text.contains("applied"), "{text}");
    assert_file(&fixture.repo.join("neu.txt"), "x\n");
}

#[test]
fn multi_site_apply_receipt_names_every_touched_line() {
    let fixture = Fixture::new("multi-site-apply");
    std::fs::write(
        fixture.repo.join("repeated.txt"),
        "alpha\nbeta\nalpha\ngamma\n",
    )
    .unwrap();

    let output = fixture.run(&[
        "replace-text",
        "repeated.txt",
        "alpha",
        "ALPHA",
        "--expect",
        "2",
    ]);
    let text = combined(&output);

    assert!(output.status.success(), "{text}");
    let transaction = text
        .strip_prefix("applied repeated.txt:1,3  ")
        .and_then(|tail| tail.strip_suffix('\n'))
        .expect("receipt must name both disjoint sites and only then the transaction");
    assert_eq!(transaction.len(), 6, "{text}");
    assert!(
        transaction.chars().all(|ch| ch.is_ascii_hexdigit()),
        "{text}"
    );
    assert_file(
        &fixture.repo.join("repeated.txt"),
        "ALPHA\nbeta\nALPHA\ngamma\n",
    );
}

#[test]
fn multi_site_dry_run_receipt_names_every_site_without_writing() {
    let fixture = Fixture::new("multi-site-dry-run");
    std::fs::write(
        fixture.repo.join("repeated.txt"),
        "alpha\nbeta\nalpha\ngamma\n",
    )
    .unwrap();

    let output = fixture.run(&[
        "replace-text",
        "repeated.txt",
        "alpha",
        "ALPHA",
        "--expect",
        "2",
        "--dry-run",
    ]);
    let text = combined(&output);

    assert!(output.status.success(), "{text}");
    assert_eq!(text, "would apply repeated.txt:1,3\n");
    assert_file(
        &fixture.repo.join("repeated.txt"),
        "alpha\nbeta\nalpha\ngamma\n",
    );
}

#[test]
fn absent_new_payload_is_read_byte_exactly_from_stdin() {
    let fixture = Fixture::new("piped-stdin");

    let output = fixture.run_with_stdin(&["write", "piped.txt"], b"from stdin\n");
    let text = combined(&output);

    assert!(output.status.success(), "{text}");
    assert!(text.starts_with("applied piped.txt:1  "), "{text}");
    assert_eq!(
        std::fs::read(fixture.repo.join("piped.txt")).unwrap(),
        b"from stdin\n"
    );
}

#[test]
fn absent_payload_with_empty_stdin_is_a_usage_refusal() {
    let fixture = Fixture::new("empty-stdin");

    let output = fixture.run(&["write", "forgotten.txt"]);
    let text = combined(&output);

    assert_eq!(output.status.code(), Some(20), "{text}");
    assert!(text.contains("no NEW: stdin was empty"), "{text}");
    assert!(!fixture.repo.join("forgotten.txt").exists());
}

#[test]
fn patch_refusal_leaves_every_file_untouched() {
    let fixture = Fixture::new("patch-atomic");
    std::fs::write(fixture.repo.join("one.txt"), "one\n").unwrap();
    std::fs::write(fixture.repo.join("two.txt"), "two\n").unwrap();
    let diff = "--- a/one.txt\n+++ b/one.txt\n@@ -40,1 +40,1 @@\n-one\n+ONE\n--- a/two.txt\n+++ b/two.txt\n@@ -90,1 +90,1 @@\n-missing\n+TWO\n";

    let output = fixture.run(&["patch", diff]);
    let text = combined(&output);

    assert_eq!(output.status.code(), Some(13), "{text}");
    assert!(text.contains("nothing written"), "{text}");
    assert_file(&fixture.repo.join("one.txt"), "one\n");
    assert_file(&fixture.repo.join("two.txt"), "two\n");
}

#[test]
fn dead_edit_prefix_is_refused_and_double_dash_preserves_hyphen_payloads() {
    let fixture = Fixture::new("dead-prefix");

    let refused = fixture.run(&["edit", "replace", "--file", "x", "--old", "a"]);
    let refusal = combined(&refused);
    assert_eq!(refused.status.code(), Some(64), "{refusal}");
    assert!(
        refusal.contains("unrecognized subcommand 'edit'"),
        "{refusal}"
    );

    let written = fixture.run(&["write", "--", "-name.txt", "-payload"]);
    let receipt = combined(&written);
    assert!(written.status.success(), "{receipt}");
    assert!(receipt.starts_with("applied -name.txt:1  "), "{receipt}");
    assert_file(&fixture.repo.join("-name.txt"), "-payload");
}
