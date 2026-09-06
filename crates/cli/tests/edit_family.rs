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
fn malformed_patch_reports_input_line_and_preserves_the_file() {
    let fixture = Fixture::new("patch-prefix-diagnostic");
    let original = "fn before() {}\n";
    std::fs::write(fixture.repo.join("item.rs"), original).unwrap();
    let output = fixture.run_with_stdin(
        &["patch"],
        b"--- a/item.rs\n+++ b/item.rs\n@@ -1 +1 @@\n-fn before() {}\nfn after() {}\n",
    );
    assert_eq!(output.status.code(), Some(20), "{}", combined(&output));
    let diagnostic = combined(&output);
    assert!(
        diagnostic.contains("item.rs: patch input line 5"),
        "{diagnostic}"
    );
    assert!(diagnostic.contains("git diff --no-color"), "{diagnostic}");
    assert!(diagnostic.contains("nothing written"), "{diagnostic}");
    assert_file(&fixture.repo.join("item.rs"), original);
}

#[cfg(unix)]
fn install_fake_tsc(fixture: &Fixture, script: &str) {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::write(fixture.repo.join("package.json"), "{}\n").unwrap();
    let bin = fixture.repo.join("node_modules/.bin");
    std::fs::create_dir_all(&bin).unwrap();
    let tsc = bin.join("tsc");
    std::fs::write(&tsc, script).unwrap();
    let mut permissions = std::fs::metadata(&tsc).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(tsc, permissions).unwrap();
}

#[cfg(unix)]
#[test]
fn verify_selects_the_touched_typescript_project_and_reports_live_status() {
    let fixture = Fixture::new("verify-typescript-selection");
    std::fs::write(
        fixture.repo.join("Cargo.toml"),
        "this would make cargo check the wrong verifier\n",
    )
    .unwrap();
    std::fs::write(fixture.repo.join("ui.ts"), "const oldValue = 1;\n").unwrap();
    install_fake_tsc(
        &fixture,
        "#!/bin/sh\nprintf 'typescript verifier ran' > tsc-ran\nexit 0\n",
    );

    let output = fixture.run(&["replace-text", "ui.ts", "oldValue", "newValue", "--verify"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "stdout={stdout}\nstderr={stderr}");
    assert!(
        stderr.contains("verify: running local TypeScript check"),
        "verification start must be visible immediately: {stderr}"
    );
    assert!(stderr.contains("verify: passed"), "stderr={stderr}");
    assert!(stdout.contains("verify: passed — local TypeScript check"));
    assert_file(&fixture.repo.join("tsc-ran"), "typescript verifier ran");
    assert_file(&fixture.repo.join("ui.ts"), "const newValue = 1;\n");
}

#[cfg(unix)]
#[test]
fn verify_timeout_is_bounded_actionable_and_keeps_the_applied_edit() {
    let fixture = Fixture::new("verify-timeout");
    std::fs::write(fixture.repo.join("ui.ts"), "const oldValue = 1;\n").unwrap();
    install_fake_tsc(
        &fixture,
        "#!/bin/sh\nprintf started > verify-started\nsleep 30\n",
    );

    let mut child = fixture
        .command()
        .env("GREPPY_EDIT_VERIFY_TIMEOUT_SECS", "1")
        .env("GREPPY_AUTO_REINDEX", "0")
        .args(["replace-text", "ui.ts", "oldValue", "newValue", "--verify"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bounded verify");
    let marker = fixture.repo.join("verify-started");
    let marker_deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !marker.exists() {
        if let Some(status) = child.try_wait().expect("poll bounded verify") {
            panic!("verify exited before starting its checker: {status}");
        }
        assert!(
            std::time::Instant::now() < marker_deadline,
            "timeout waiting for verifier startup"
        );
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    let started = std::time::Instant::now();
    let output = child.wait_with_output().expect("wait for bounded verify");
    let elapsed = started.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "stdout={stdout}\nstderr={stderr}");
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "elapsed={elapsed:?}; stdout={stdout}; stderr={stderr}"
    );
    assert!(stderr.contains("verify: running local TypeScript check"));
    assert!(
        stderr.contains("verify: timed out after 1s"),
        "stderr={stderr}"
    );
    assert!(
        stdout.contains("edit remains applied; run"),
        "the receipt must provide recovery: {stdout}"
    );
    assert_file(&fixture.repo.join("ui.ts"), "const newValue = 1;\n");
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
fn patch_accepts_git_metadata_between_files_without_changing_payload_lines() {
    let fixture = Fixture::new("patch-git-metadata");
    let one = "diff --git is file content\nindex is file content\none\n";
    std::fs::write(fixture.repo.join("one.txt"), one).unwrap();
    std::fs::write(fixture.repo.join("two.txt"), "two\n").unwrap();
    let diff = "diff --git a/one.txt b/one.txt\nindex 1111111..2222222 100644\n--- a/one.txt\n+++ b/one.txt\n@@ -1,3 +1,3 @@\n diff --git is file content\n index is file content\n-one\n+ONE\ndiff --git a/two.txt b/two.txt\nindex 3333333..4444444 100644\n--- a/two.txt\n+++ b/two.txt\n@@ -1 +1 @@\n-two\n+TWO\n";

    let dry_run = fixture.run_with_stdin(&["patch", "--dry-run"], diff.as_bytes());
    assert!(dry_run.status.success(), "{}", combined(&dry_run));
    assert_file(&fixture.repo.join("one.txt"), one);
    assert_file(&fixture.repo.join("two.txt"), "two\n");
    let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(
        &fixture.repo.join("one.txt"),
        "diff --git is file content\nindex is file content\nONE\n",
    );
    assert_file(&fixture.repo.join("two.txt"), "TWO\n");
}

#[test]
fn patch_git_metadata_keeps_late_context_refusal_atomic() {
    let fixture = Fixture::new("patch-git-atomic");
    std::fs::write(fixture.repo.join("one.txt"), "one\n").unwrap();
    std::fs::write(fixture.repo.join("two.txt"), "two\n").unwrap();
    let diff = "diff --git a/one.txt b/one.txt\nindex 1111111..2222222 100644\n--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-one\n+ONE\ndiff --git a/two.txt b/two.txt\nindex 3333333..4444444 100644\n--- a/two.txt\n+++ b/two.txt\n@@ -1 +1 @@\n-missing\n+TWO\n";
    let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    assert_eq!(output.status.code(), Some(13), "{}", combined(&output));
    assert!(combined(&output).contains("nothing written"));
    assert_file(&fixture.repo.join("one.txt"), "one\n");
    assert_file(&fixture.repo.join("two.txt"), "two\n");
}

#[test]
fn patch_git_unsupported_sections_are_not_silently_dropped() {
    let fixture = Fixture::new("patch-git-unsupported");
    std::fs::write(fixture.repo.join("one.txt"), "one\n").unwrap();
    let edit = "diff --git a/one.txt b/one.txt\nindex 1111111..2222222 100644\n--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-one\n+ONE\n";
    for suffix in [
        "diff --git a/two.txt b/two.txt\nold mode 100644\nnew mode 100755\n",
        "diff --git a/two.bin b/two.bin\nindex 3333333..4444444 100644\nBinary files a/two.bin and b/two.bin differ\n",
        "diff --git a/old.txt b/new.txt\nsimilarity index 100%\nrename from old.txt\nrename to new.txt\n",
        "diff --git a/two.txt b/two.txt\nindex 3333333..4444444 100644\n",
    ] {
        let diff = format!("{edit}{suffix}");
        let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
        let text = combined(&output);
        assert_eq!(output.status.code(), Some(20), "{text}");
        assert!(text.contains("Git patch section"), "{text}");
        assert!(text.contains("nothing written"), "{text}");
        assert_file(&fixture.repo.join("one.txt"), "one\n");
    }
}

#[test]
fn replace_text_accepts_raw_borrows_and_preserves_syntax_refusal_atomicity() {
    let fixture = Fixture::new("replace-text-rust-raw");
    let source = "fn before() {}\n";
    let replacement = "fn inserted() { let raw = 1; let _ = &raw; }\nfn before()";
    let expected = format!("{replacement} {{}}\n");
    std::fs::write(fixture.repo.join("valid.rs"), source).unwrap();

    let dry_run = fixture.run(&[
        "replace-text",
        "valid.rs",
        "fn before()",
        replacement,
        "--dry-run",
    ]);
    assert!(dry_run.status.success(), "{}", combined(&dry_run));
    assert_file(&fixture.repo.join("valid.rs"), source);

    let written = fixture.run(&["replace-text", "valid.rs", "fn before()", replacement]);
    assert!(written.status.success(), "{}", combined(&written));
    assert_file(&fixture.repo.join("valid.rs"), &expected);

    let refused = fixture.run(&["replace-text", "valid.rs", "let _ = &raw;", "let _ = ;"]);
    assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
    assert!(combined(&refused).contains("nothing written"));
    assert_file(&fixture.repo.join("valid.rs"), &expected);
}

#[test]
fn write_accepts_borrow_of_raw_identifier_and_still_rejects_broken_rust() {
    let fixture = Fixture::new("write-rust-raw");
    let source = b"fn main() { let raw = 1; let _ = &raw; }\n";
    let dry_run = fixture.run_with_stdin(&["write", "--dry-run", "valid.rs"], source);
    assert!(dry_run.status.success(), "{}", combined(&dry_run));
    assert!(!fixture.repo.join("valid.rs").exists());
    let written = fixture.run_with_stdin(&["write", "valid.rs"], source);
    assert!(written.status.success(), "{}", combined(&written));
    assert_eq!(
        std::fs::read(fixture.repo.join("valid.rs")).unwrap(),
        source
    );
    let refused = fixture.run_with_stdin(&["write", "valid.rs"], b"fn main( {}\n");
    assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
    assert_eq!(
        std::fs::read(fixture.repo.join("valid.rs")).unwrap(),
        source
    );
}

#[test]
fn patch_creation_refusal_explains_recovery_and_preserves_transaction() {
    let fixture = Fixture::new("patch-create");
    std::fs::write(fixture.repo.join("existing.txt"), "before\n").unwrap();
    let creation = "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,1 @@\n+new\n";
    let mixed = format!(
        "--- a/existing.txt\n+++ b/existing.txt\n@@ -1,1 +1,1 @@\n-before\n+after\n{creation}"
    );
    for diff in [creation, mixed.as_str()] {
        for args in [vec!["patch", "--dry-run"], vec!["patch"]] {
            let output = fixture.run_with_stdin(&args, diff.as_bytes());
            let text = combined(&output);
            assert_eq!(output.status.code(), Some(20), "{text}");
            assert!(text.contains("patch only edits existing files"), "{text}");
            assert!(text.contains("greppy write"), "{text}");
            assert!(text.contains("separate transaction"), "{text}");
            assert!(text.contains("nothing written"), "{text}");
            assert!(!fixture.repo.join("new.txt").exists());
            assert_file(&fixture.repo.join("existing.txt"), "before\n");
        }
    }
}

#[test]
fn patch_deletion_and_contextless_edit_remain_explicit_refusals() {
    let fixture = Fixture::new("patch-delete");
    std::fs::write(fixture.repo.join("existing.txt"), "before\n").unwrap();
    let deletion = "--- a/existing.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-before\n";
    let output = fixture.run_with_stdin(&["patch"], deletion.as_bytes());
    let text = combined(&output);
    assert_eq!(output.status.code(), Some(20), "{text}");
    assert!(text.contains("file deletion is not supported"), "{text}");
    assert!(text.contains("nothing written"), "{text}");
    assert_file(&fixture.repo.join("existing.txt"), "before\n");

    let insertion = "--- a/existing.txt\n+++ b/existing.txt\n@@ -0,0 +1,1 @@\n+new\n";
    let output = fixture.run_with_stdin(&["patch"], insertion.as_bytes());
    let text = combined(&output);
    assert_eq!(output.status.code(), Some(20), "{text}");
    assert!(text.contains("no context line to anchor on"), "{text}");
    assert_file(&fixture.repo.join("existing.txt"), "before\n");
}

#[test]
fn patch_help_discloses_existing_file_only_contract() {
    let fixture = Fixture::new("patch-help");
    let output = fixture.run(&["patch", "--help"]);
    let text = combined(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("existing files"), "{text}");
    assert!(text.contains("greppy write"), "{text}");
    assert!(text.contains("separate transaction"), "{text}");
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
