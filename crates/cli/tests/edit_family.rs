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
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env_remove("GREPPY_VERIFY_TEST_COMMAND");
        command
    }

    fn command_in(&self, cwd: &Path) -> Command {
        let mut command = Command::new(bin());
        command
            .current_dir(cwd)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env_remove("GREPPY_VERIFY_TEST_COMMAND");
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command()
            .args(args)
            .stdin(Stdio::null())
            .output()
            .expect("run greppy")
    }

    fn run_in(&self, cwd: &Path, args: &[&str]) -> Output {
        self.command_in(cwd)
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

fn concurrent_journal(fixture: &Fixture) -> PathBuf {
    fixture
        .store
        .join("workspaces")
        .join(format!("v{}", greppy_core::cache::STORE_FORMAT_VERSION))
        .join(greppy_core::workspace::workspace_hash(&fixture.repo))
        .join("edit-journal")
}

struct OwnedEditChild(Option<std::process::Child>);
impl std::ops::Deref for OwnedEditChild {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}
impl std::ops::DerefMut for OwnedEditChild {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}
impl Drop for OwnedEditChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

fn bounded_output(mut child: OwnedEditChild) -> Output {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.0.take().unwrap().wait_with_output().unwrap();
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.0.take().unwrap().wait_with_output().unwrap();
            panic!(
                "owned concurrent edit exceeded test deadline: {}",
                combined(&output)
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn spawn_edit(fixture: &Fixture, args: &[&str]) -> OwnedEditChild {
    OwnedEditChild(Some(
        fixture
            .command()
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ))
}

fn wait_for_file_owner(fixture: &Fixture, file: &str) {
    use sha2::{Digest, Sha256};
    let identity = fixture.repo.join(file).canonicalize().unwrap();
    let name = format!(
        "file-{:x}",
        Sha256::digest(identity.as_os_str().as_encoded_bytes())
    );
    let journal = concurrent_journal(fixture);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if journal.join("locks").join(&name).exists()
            && greppy_core::cache::acquire_named_lock_in(
                &journal,
                &name,
                greppy_core::cache::LockMode::Exclusive,
                true,
            )
            .unwrap()
            .is_none()
        {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "first edit did not acquire its file lock"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn release_stdin(mut child: OwnedEditChild, replacement: &[u8]) -> Output {
    use std::io::Write as _;
    child.stdin.take().unwrap().write_all(replacement).unwrap();
    bounded_output(child)
}

#[test]
fn concurrent_six_call_report_and_dozen_agents_preserve_every_edit_and_undo() {
    for count in [6, 32] {
        let fixture = Fixture::new("parallel-edit-round");
        use std::io::Write as _;
        for i in 0..count {
            std::fs::write(fixture.repo.join(format!("f{i}.txt")), "A B\n").unwrap();
        }
        let mut children = Vec::new();
        for i in 0..count {
            let (file, old, new) = if count == 6 && i == 5 {
                ("f0.txt".to_string(), "B", "Y")
            } else {
                (format!("f{i}.txt"), "A", "X")
            };
            let child = spawn_edit(&fixture, &["replace-text", &file, old]);
            if !(count == 6 && i == 5) {
                wait_for_file_owner(&fixture, &file);
            }
            children.push((child, new));
        }
        // All disjoint targets demonstrably own their locks simultaneously.
        // Release payloads together, then wait with bounded kill/reap guards.
        for (child, new) in &mut children {
            child
                .stdin
                .take()
                .unwrap()
                .write_all(new.as_bytes())
                .unwrap();
        }
        for (child, _) in children {
            let output = bounded_output(child);
            assert!(output.status.success(), "{}", combined(&output));
        }
        for i in 0..count {
            let expected = if count == 6 && i == 5 {
                "A B\n"
            } else if count == 6 && i == 0 {
                "X Y\n"
            } else {
                "X B\n"
            };
            assert_file(&fixture.repo.join(format!("f{i}.txt")), expected);
        }
        let stack: serde_json::Value = serde_json::from_slice(
            &std::fs::read(concurrent_journal(&fixture).join("stack.json")).unwrap(),
        )
        .unwrap();
        let transactions = stack["transactions"].as_array().unwrap();
        assert_eq!(transactions.len(), count);
        let ids: std::collections::HashSet<_> = transactions
            .iter()
            .map(|v| v["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids.len(), count);
        for _ in 0..count {
            let undo = fixture.run(&["undo"]);
            assert!(undo.status.success(), "{}", combined(&undo));
        }
        for i in 0..count {
            assert_file(&fixture.repo.join(format!("f{i}.txt")), "A B\n");
        }
    }
}

#[test]
fn blocked_stdin_does_not_block_disjoint_edit_and_same_file_waiter_rereads() {
    let fixture = Fixture::new("file-lock-wait");
    std::fs::write(fixture.repo.join("a.txt"), "A B\n").unwrap();
    std::fs::write(fixture.repo.join("b.txt"), "A\n").unwrap();
    let first = spawn_edit(&fixture, &["replace-text", "a.txt", "A"]);
    wait_for_file_owner(&fixture, "a.txt");
    let unrelated = spawn_edit(&fixture, &["replace-text", "b.txt", "A", "X"]);
    let output = bounded_output(unrelated);
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(&fixture.repo.join("b.txt"), "X\n");
    let mut same = spawn_edit(&fixture, &["replace-text", "a.txt", "B", "Y"]);
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(
        same.try_wait().unwrap().is_none(),
        "same-file call refused instead of waiting"
    );
    let first_output = release_stdin(first, b"X");
    assert!(first_output.status.success(), "{}", combined(&first_output));
    let same_output = bounded_output(same);
    assert!(same_output.status.success(), "{}", combined(&same_output));
    assert_file(&fixture.repo.join("a.txt"), "X Y\n");
}

#[cfg(unix)]
#[test]
fn symlink_parent_alias_serializes_with_the_real_file_and_preserves_both_edits() {
    let fixture = Fixture::new("symlink-parent-lock");
    std::fs::create_dir_all(fixture.repo.join("real/child")).unwrap();
    std::fs::write(fixture.repo.join("real/a.txt"), "A B\n").unwrap();
    std::fs::write(fixture.repo.join("a.txt"), "SENTINEL\n").unwrap();
    std::os::unix::fs::symlink(fixture.repo.join("real/child"), fixture.repo.join("link")).unwrap();
    let first = spawn_edit(&fixture, &["replace-text", "link/../a.txt", "A"]);
    wait_for_file_owner(&fixture, "real/a.txt");
    let mut same = spawn_edit(&fixture, &["replace-text", "real/a.txt", "B", "Y"]);
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(same.try_wait().unwrap().is_none());
    let first = release_stdin(first, b"X");
    assert!(first.status.success(), "{}", combined(&first));
    let same = bounded_output(same);
    assert!(same.status.success(), "{}", combined(&same));
    assert_file(&fixture.repo.join("real/a.txt"), "X Y\n");
    assert_file(&fixture.repo.join("a.txt"), "SENTINEL\n");
}

#[test]
fn interrupted_edit_keeps_its_own_pending_evidence_when_another_edit_finishes() {
    let fixture = Fixture::new("pending-isolation");
    for file in ["a.txt", "b.txt"] {
        std::fs::write(fixture.repo.join(file), "A\n").unwrap();
    }
    let interrupted = fixture
        .command()
        .env("GREPPY_TEST_CRASH_AFTER_JOURNAL", "1")
        .args(["replace-text", "a.txt", "A", "X"])
        .output()
        .unwrap();
    assert_eq!(
        interrupted.status.code(),
        Some(16),
        "{}",
        combined(&interrupted)
    );
    let journal = concurrent_journal(&fixture);
    let pending: Vec<_> = std::fs::read_dir(&journal)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("pending-")
        })
        .collect();
    assert_eq!(pending.len(), 1);
    let evidence = std::fs::read(&pending[0]).unwrap();
    let good = fixture.run(&["replace-text", "b.txt", "A", "X"]);
    assert!(good.status.success(), "{}", combined(&good));
    assert_eq!(std::fs::read(&pending[0]).unwrap(), evidence);
    assert_file(&fixture.repo.join("a.txt"), "A\n");
    let undo = fixture.run(&["undo"]);
    assert!(undo.status.success(), "{}", combined(&undo));
    assert_file(&fixture.repo.join("b.txt"), "A\n");
}

#[test]
fn journal_failure_after_publication_reports_that_source_was_written() {
    let fixture = Fixture::new("journal-truth");
    std::fs::write(fixture.repo.join("a.txt"), "A\n").unwrap();
    let first = spawn_edit(&fixture, &["--json", "replace-text", "a.txt", "A"]);
    wait_for_file_owner(&fixture, "a.txt");
    std::fs::write(
        concurrent_journal(&fixture).join("stack.json"),
        b"broken journal",
    )
    .unwrap();
    let output = release_stdin(first, b"X");
    assert_eq!(output.status.code(), Some(16), "{}", combined(&output));
    let record: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(record["published"], true);
    assert_eq!(record["status"], "published_with_error");
    assert!(record["error"]["message"]
        .as_str()
        .unwrap()
        .contains("pending evidence retained"));
    assert_file(&fixture.repo.join("a.txt"), "X\n");
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
fn write_names_an_absent_workspace_root_and_preserves_the_calling_workspace() {
    let fixture = Fixture::new("write-missing-root");
    let missing = fixture.base.join("not-yet-created");
    let root = missing.to_str().unwrap();
    for json in [false, true] {
        let mut args = vec!["--root", root];
        if json {
            args.push("--json");
        }
        args.extend(["write", "client/run.py", "# harmless fixture\n"]);
        let output = fixture.run(&args);
        assert_eq!(output.status.code(), Some(20), "{}", combined(&output));
        let body = combined(&output);
        assert!(body.contains("does not exist"), "{body}");
        assert!(
            body.contains("create that directory before retrying"),
            "{body}"
        );
        assert!(!body.contains("is outside"), "{body}");
        if json {
            let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(value["schema_version"], "greppy.edit-record.v1");
            assert_eq!(value["status"], "refused");
            assert_eq!(value["published"], false);
            assert_eq!(value["exit_code"], 20);
            assert_eq!(value["error"]["code"], "INVALID_REQUEST");
            assert_eq!(value["operations"], serde_json::json!([]));
            assert!(output.stderr.is_empty(), "{}", combined(&output));
        }
        assert!(!missing.exists());
        assert!(!fixture.repo.join("client/run.py").exists());
    }
    std::fs::create_dir_all(missing.join("client")).unwrap();
    let retry = fixture.run(&[
        "--root",
        root,
        "write",
        "client/run.py",
        "# harmless fixture\n",
    ]);
    assert!(retry.status.success(), "{}", combined(&retry));
    assert_file(&missing.join("client/run.py"), "# harmless fixture\n");
    assert!(!fixture.repo.join("client/run.py").exists());
}

#[test]
fn regex_replacement_refuses_unknown_captures_and_preserves_literal_routes() {
    let fixture = Fixture::new("regex-route");
    let path = fixture.repo.join("route.ts");
    let before = "export const route = \"before\";\n";
    std::fs::write(&path, before).unwrap();
    for replacement in ["/$environmentId/$threadId", "/${missing}/", "$1", "$1x"] {
        let out = fixture.run(&["replace-text", "route.ts", "before", replacement, "--regex"]);
        assert_eq!(out.status.code(), Some(17), "{}", combined(&out));
        assert!(combined(&out).contains("capture"));
        assert!(combined(&out).contains("$$"));
        assert_file(&path, before);
    }
    let out = fixture.run(&[
        "replace-text",
        "route.ts",
        "before",
        "/$$environmentId/$$threadId",
        "--regex",
    ]);
    assert!(out.status.success(), "{}", combined(&out));
    assert_file(
        &path,
        "export const route = \"/$environmentId/$threadId\";\n",
    );
    std::fs::write(&path, before).unwrap();
    let out = fixture.run(&[
        "replace-text",
        "route.ts",
        "(?P<part>before)",
        "${part}-$1-$0",
        "--regex",
    ]);
    assert!(out.status.success(), "{}", combined(&out));
    assert_file(&path, "export const route = \"before-before-before\";\n");
}

#[test]
fn regex_replacement_preserves_non_utf8_braced_literal_bytes() {
    let fixture = Fixture::new("regex-bytes");
    let path = fixture.repo.join("value.txt");
    std::fs::write(&path, b"before").unwrap();
    let out = fixture.run_with_stdin(
        &["replace-text", "value.txt", "before", "--regex"],
        b"${\xff}",
    );
    assert!(out.status.success(), "{}", combined(&out));
    assert_eq!(std::fs::read(&path).unwrap(), b"${\xff}");
}

#[test]
fn write_bash_readwrite_redirect_accepts_valid_shell_and_refuses_invalid_changes_atomically() {
    let fixture = Fixture::new("bash-readwrite-redirect");
    let valid = "#!/bin/bash\nexec 9<>/mnt/nvme1/.greppy-heavy.lock\n";
    let out = fixture.run_with_stdin(&["write", "lock.sh"], valid.as_bytes());
    assert!(out.status.success(), "{}", combined(&out));
    let path = fixture.repo.join("lock.sh");
    assert_file(&path, valid);
    for invalid in [
        "#!/bin/bash\nexec 9< >file\n",
        "#!/bin/bash\nexec 9<>\n",
        "#!/bin/bash\nexec 9<>file\nif then\n",
    ] {
        let out = fixture.run_with_stdin(&["write", "lock.sh"], invalid.as_bytes());
        assert_eq!(out.status.code(), Some(13), "{}", combined(&out));
        assert_file(&path, valid);
    }
}

#[test]
fn write_typed_template_accepts_valid_typescript_and_refuses_malformed_changes_atomically() {
    let fixture = Fixture::new("typed-template");
    let valid = "function* run() { const rows = yield* sql<{ readonly workspace_root: string | null }>`SELECT workspace_root`; return rows; }\n";
    let out = fixture.run_with_stdin(&["write", "query.ts"], valid.as_bytes());
    assert!(out.status.success(), "{}", combined(&out));
    let path = fixture.repo.join("query.ts");
    assert_file(&path, valid);
    for bad_type in ["string |", "string|", "string&"] {
        let invalid = valid.replace("string | null", bad_type);
        let out = fixture.run_with_stdin(&["write", "query.ts"], invalid.as_bytes());
        assert_eq!(out.status.code(), Some(13), "{}", combined(&out));
        assert_file(&path, valid);
    }
}

#[test]
fn write_outside_workspace_refuses_nonzero_and_names_root_recovery() {
    let fixture = Fixture::new("write-outside");
    let other = fixture.base.join("other");
    std::fs::create_dir_all(other.join(".git")).unwrap();
    let file = other.join("note.md");
    std::fs::write(&file, "preserve\n").unwrap();
    let path = file.to_str().unwrap();
    for flags in [vec![], vec!["--dry-run"], vec!["--json"]] {
        let mut args = vec!["write", path];
        args.extend(flags);
        let output = fixture.run_with_stdin(&args, b"replacement\n");
        assert_eq!(output.status.code(), Some(17), "{}", combined(&output));
        assert_file(&file, "preserve\n");
        let message = if args.contains(&"--json") {
            let record: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(record["status"], "refused");
            assert_eq!(record["exit_code"], 17);
            assert_eq!(record["published"], false);
            record["error"]["message"].as_str().unwrap().to_owned()
        } else {
            assert!(output.stdout.is_empty(), "{}", combined(&output));
            String::from_utf8_lossy(&output.stderr).into_owned()
        };
        assert!(
            message.contains("nothing written") && message.contains("--root DIR"),
            "{message}"
        );
    }
    let recovery = fixture.run_with_stdin(
        &["write", "note.md", "--root", other.to_str().unwrap()],
        b"replacement\n",
    );
    assert_eq!(recovery.status.code(), Some(0), "{}", combined(&recovery));
    assert_file(&file, "replacement\n");
}

#[test]
fn symbol_edit_repairs_metadata_only_drift_without_rebuilding_graph() {
    let fixture = Fixture::new("metadata-symbol-refresh");
    let source = fixture.repo.join("lib.rs");
    std::fs::write(&source, "fn indexed_definition() {}\n").unwrap();
    let git = |args: &[&str]| {
        let result = Command::new("git")
            .args(args)
            .current_dir(&fixture.repo)
            .output()
            .unwrap();
        assert!(result.status.success(), "{}", combined(&result));
    };
    git(&["init", "-q"]);
    git(&["config", "user.email", "fixture@example.invalid"]);
    git(&["config", "user.name", "Fixture"]);
    git(&["config", "commit.gpgsign", "false"]);
    git(&["add", "lib.rs"]);
    git(&["commit", "-qm", "initial"]);
    let indexed = fixture.run(&["index", "."]);
    assert!(indexed.status.success(), "{}", combined(&indexed));
    let db = fixture
        .store
        .join("workspaces")
        .join("v2")
        .join(greppy_core::workspace::workspace_hash(&fixture.repo))
        .join("graph.db");
    let state = || {
        greppy_store::Store::open_with(&db, greppy_store::OpenOptions::read_only())
            .unwrap()
            .list_workspace_states()
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    };
    let before = state();
    git(&["commit", "--allow-empty", "-qm", "metadata only"]);
    let head = greppy_core::GitFingerprint::capture(&fixture.repo).head_oid;
    assert_ne!(before.head_oid, head);
    let planned = fixture.run(&["delete", "indexed_definition", "--dry-run"]);
    assert!(planned.status.success(), "{}", combined(&planned));
    assert_file(&source, "fn indexed_definition() {}\n");
    let after = state();
    assert_eq!(after.head_oid, head);
    assert_eq!(after.graph_generation, before.graph_generation);
    let deleted = fixture.run(&["delete", "indexed_definition"]);
    assert!(deleted.status.success(), "{}", combined(&deleted));
    assert_file(&source, "");
}

#[test]
fn symbol_edit_refreshes_source_added_after_index_and_absent_stays_absent() {
    let fixture = Fixture::new("stale-symbol-refresh");
    let source = fixture.repo.join("lib.rs");
    std::fs::write(&source, "fn indexed_definition() {}\n").unwrap();

    let indexed = fixture.run(&["index", "."]);
    assert!(
        indexed.status.success(),
        "initial index failed: {}",
        combined(&indexed)
    );

    let drifted = "fn indexed_definition() {}\nfn added_after_index() { println!(\"fresh\"); }\n";
    std::fs::write(&source, drifted).unwrap();
    let deleted = fixture.run(&["delete", "added_after_index"]);
    assert!(
        deleted.status.success(),
        "stale symbol edit did not refresh: {}",
        combined(&deleted)
    );
    assert_file(&source, "fn indexed_definition() {}\n");

    let before_absent = std::fs::read(&source).unwrap();
    let absent = fixture.run(&["delete", "genuinely_absent"]);
    assert!(
        !absent.status.success(),
        "absent symbol unexpectedly edited"
    );
    assert!(
        combined(&absent).contains("no symbol `genuinely_absent`"),
        "unexpected absent-symbol diagnostic: {}",
        combined(&absent)
    );
    assert_eq!(std::fs::read(&source).unwrap(), before_absent);
}

fn nested_collision(fixture: &Fixture) -> (PathBuf, PathBuf) {
    std::fs::write(fixture.base.join("probe.conf"), "CWD_SENTINEL\n").unwrap();
    std::fs::write(fixture.repo.join("probe.conf"), "REPO_SENTINEL\n").unwrap();
    let nested = fixture.repo.join("etc");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("probe.conf"), "SUBDIR_SENTINEL\n").unwrap();
    (fixture.base.clone(), nested)
}

fn assert_collision_untouched(fixture: &Fixture, nested: &Path, nested_expected: &str) {
    assert_file(&fixture.base.join("probe.conf"), "CWD_SENTINEL\n");
    assert_file(&fixture.repo.join("probe.conf"), "REPO_SENTINEL\n");
    assert_file(&nested.join("probe.conf"), nested_expected);
}

#[test]
fn replace_lines_accepts_complete_async_method_from_stdin() {
    let fixture = Fixture::new("replace-lines-method");
    let before = "struct Worker;\nimpl Worker {\n    fn start_subscription_task(&self) {\n        old();\n    }\n}\n";
    let replacement = "    fn start_subscription_task(&self) -> tokio::task::JoinHandle<()> {\n        tokio::spawn(async move {\n            let Some(value) = Some(1) else { return; };\n            println!(\"{value}\");\n        })\n    }\n";
    std::fs::write(fixture.repo.join("worker.rs"), before).unwrap();
    let output = fixture.run_with_stdin(
        &["replace-lines", "worker.rs", "3:5"],
        replacement.as_bytes(),
    );
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(
        &fixture.repo.join("worker.rs"),
        &format!("struct Worker;\nimpl Worker {{\n{replacement}}}\n"),
    );
}

#[test]
fn invalid_edit_reports_candidate_parser_location_without_writing() {
    let fixture = Fixture::new("edit-parser-diagnostic");
    let before = "fn before() {}\n";
    std::fs::write(fixture.repo.join("item.rs"), before).unwrap();
    for args in [
        vec!["replace-lines", "item.rs", "1:1", "fn after( {}"],
        vec!["write", "item.rs", "fn after( {}"],
        vec![
            "patch",
            "--- a/item.rs\n+++ b/item.rs\n@@ -1 +1 @@\n-fn before() {}\n+fn after( {}\n",
        ],
    ] {
        for json in [false, true] {
            let mut command = args.clone();
            if json {
                command.push("--json");
            }
            let output = fixture.run(&command);
            assert_eq!(output.status.code(), Some(13), "{}", combined(&output));
            let message = if json {
                let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(value["error"]["code"], "invalid_result");
                assert_eq!(value["published"], false);
                value["error"]["message"].as_str().unwrap().to_owned()
            } else {
                combined(&output)
            };
            assert!(message.contains("proposed item.rs:1:"), "{message}");
            assert!(message.contains("tree-sitter:"), "{message}");
            assert!(message.contains("nothing written"), "{message}");
            assert!(message.contains("proposed result"), "{message}");
            assert_file(&fixture.repo.join("item.rs"), before);
        }
    }
}

#[test]
fn edit_preview_reads_the_explicit_root_instead_of_cwd() {
    let fixture = Fixture::new("preview-explicit-root");
    let target = fixture.repo.join("target-repo");
    std::fs::create_dir_all(target.join(".git")).unwrap();
    std::fs::write(fixture.repo.join("same.txt"), "CWD_SENTINEL\n").unwrap();
    for relative in [false, true] {
        let root = if relative {
            "target-repo"
        } else {
            target.to_str().unwrap()
        };
        let output = fixture.run(&["--root", root, "write", "same.txt", "TARGET_WRITE\n"]);
        assert_eq!(output.status.code(), Some(0), "{}", combined(&output));
        let text = combined(&output);
        assert!(text.contains("TARGET_WRITE"), "{text}");
        assert!(!text.contains("CWD_SENTINEL"), "{text}");
        assert_file(&target.join("same.txt"), "TARGET_WRITE\n");
        assert_file(&fixture.repo.join("same.txt"), "CWD_SENTINEL\n");

        let output = fixture.run(&[
            "--root",
            root,
            "replace-text",
            "same.txt",
            "TARGET_WRITE",
            "TARGET_REPLACED",
        ]);
        assert_eq!(output.status.code(), Some(0), "{}", combined(&output));
        let text = combined(&output);
        assert!(text.contains("TARGET_REPLACED"), "{text}");
        assert!(!text.contains("CWD_SENTINEL"), "{text}");
        assert_file(&target.join("same.txt"), "TARGET_REPLACED\n");
        assert_file(&fixture.repo.join("same.txt"), "CWD_SENTINEL\n");
    }
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
fn install_fake_typescript_compiler(directory: &Path, name: &str, script: &str) {
    use std::os::unix::fs::PermissionsExt as _;

    let bin = directory.join("node_modules/.bin");
    std::fs::create_dir_all(&bin).unwrap();
    let compiler = bin.join(name);
    std::fs::write(&compiler, script).unwrap();
    let mut permissions = std::fs::metadata(&compiler).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(compiler, permissions).unwrap();
}

#[cfg(unix)]
fn install_fake_tsc(fixture: &Fixture, script: &str) {
    std::fs::write(fixture.repo.join("package.json"), "{}\n").unwrap();
    install_fake_typescript_compiler(&fixture.repo, "tsc", script);
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
fn verify_discovers_workspace_tsgo_and_runs_it_from_owning_package() {
    let fixture = Fixture::new("verify-workspace-tsgo");
    std::fs::write(fixture.repo.join("package.json"), "{}\n").unwrap();
    let package = fixture.repo.join("apps/server");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join("package.json"), "{}\n").unwrap();
    std::fs::write(package.join("ui.ts"), "const oldValue = 1;\n").unwrap();
    install_fake_typescript_compiler(
        &fixture.repo,
        "tsgo",
        "#!/bin/sh\nprintf workspace-tsgo > workspace-tsgo-ran\nexit 0\n",
    );

    let output = fixture.run(&[
        "replace-text",
        "apps/server/ui.ts",
        "oldValue",
        "newValue",
        "--verify",
    ]);
    assert!(output.status.success(), "{}", combined(&output));
    assert!(combined(&output).contains("verify: passed — local TypeScript check"));
    assert_file(&package.join("workspace-tsgo-ran"), "workspace-tsgo");
}

#[cfg(unix)]
#[test]
fn verify_prefers_package_compiler_and_does_not_climb_above_workspace() {
    let fixture = Fixture::new("verify-typescript-compiler-bounds");
    std::fs::write(fixture.repo.join("package.json"), "{}\n").unwrap();
    let package = fixture.repo.join("apps/server");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join("package.json"), "{}\n").unwrap();
    std::fs::write(package.join("ui.ts"), "const oldValue = 1;\n").unwrap();
    install_fake_typescript_compiler(
        &fixture.repo,
        "tsgo",
        "#!/bin/sh\nprintf root > root-compiler-ran\nexit 0\n",
    );
    install_fake_typescript_compiler(
        &package,
        "tsc",
        "#!/bin/sh\nprintf package > package-compiler-ran\nexit 0\n",
    );

    let output = fixture.run(&[
        "replace-text",
        "apps/server/ui.ts",
        "oldValue",
        "newValue",
        "--verify",
    ]);
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(&package.join("package-compiler-ran"), "package");
    assert!(!package.join("root-compiler-ran").exists());

    std::fs::remove_dir_all(package.join("node_modules")).unwrap();
    std::fs::remove_dir_all(fixture.repo.join("node_modules")).unwrap();
    install_fake_typescript_compiler(
        &fixture.base,
        "tsgo",
        "#!/bin/sh\nprintf escaped > escaped-compiler-ran\nexit 0\n",
    );
    let skipped = fixture.run(&[
        "replace-text",
        "apps/server/ui.ts",
        "newValue",
        "finalValue",
        "--verify",
    ]);
    assert!(skipped.status.success(), "{}", combined(&skipped));
    let diagnostic = combined(&skipped);
    assert!(
        diagnostic.contains("no local TypeScript compiler from"),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("through workspace root"),
        "{diagnostic}"
    );
    assert!(!package.join("escaped-compiler-ran").exists());
    assert!(!fixture.base.join("escaped-compiler-ran").exists());
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

    assert_eq!(
        output.status.code(),
        Some(17),
        "stdout={stdout}\nstderr={stderr}"
    );
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
fn patch_marker_updates_and_advisory_counts_roundtrip() {
    let fixture = Fixture::new("patch-marker-counts");
    std::fs::write(fixture.repo.join("one.txt"), "one\nkeep\n").unwrap();
    std::fs::write(fixture.repo.join("two.txt"), "two\n").unwrap();
    let marker = "*** Begin Patch\n*** Update File: one.txt\n@@ label\n-one\n+ONE\n keep\n*** Update File: two.txt\n@@\n-two\n+TWO\n*** End Patch\n";
    let preview = fixture.run_with_stdin(&["patch", "--dry-run"], marker.as_bytes());
    assert!(preview.status.success(), "{}", combined(&preview));
    assert_file(&fixture.repo.join("one.txt"), "one\nkeep\n");
    let output = fixture.run_with_stdin(&["patch"], marker.as_bytes());
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(&fixture.repo.join("one.txt"), "ONE\nkeep\n");
    assert_file(&fixture.repo.join("two.txt"), "TWO\n");
    assert!(fixture.run(&["undo"]).status.success());
    let diff = "--- a/one.txt\n+++ b/one.txt\n@@ -99,50 +99,0 @@\n-one\n+ONE\n keep\n--- a/two.txt\n+++ b/two.txt\n@@ -88,0 +88,200 @@\n-two\n+TWO\n";
    let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(&fixture.repo.join("one.txt"), "ONE\nkeep\n");
    assert_file(&fixture.repo.join("two.txt"), "TWO\n");
}

#[test]
fn patch_marker_mixed_operations_dry_run_apply_and_undo() {
    let fixture = Fixture::new("patch-marker-mixed");
    std::fs::write(fixture.repo.join("update.txt"), "old\n").unwrap();
    std::fs::write(fixture.repo.join("delete.txt"), "remove me\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            fixture.repo.join("delete.txt"),
            std::fs::Permissions::from_mode(0o751),
        )
        .unwrap();
    }
    let diff = "*** Begin Patch\n*** Update File: update.txt\n@@\n-old\n+new\n*** Add File: nested/add.txt\n+created\n+\n*** Add File: empty.txt\n*** Delete File: delete.txt\n*** End Patch\n";
    let preview = fixture.run_with_stdin(&["patch", "--dry-run"], diff.as_bytes());
    assert!(preview.status.success(), "{}", combined(&preview));
    assert_file(&fixture.repo.join("update.txt"), "old\n");
    assert_file(&fixture.repo.join("delete.txt"), "remove me\n");
    assert!(!fixture.repo.join("nested").exists());
    assert!(!fixture.repo.join("empty.txt").exists());
    let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(&fixture.repo.join("update.txt"), "new\n");
    assert_file(&fixture.repo.join("nested/add.txt"), "created\n\n");
    assert_file(&fixture.repo.join("empty.txt"), "");
    assert!(!fixture.repo.join("delete.txt").exists());
    let undone = fixture.run(&["undo"]);
    assert!(undone.status.success(), "{}", combined(&undone));
    assert_file(&fixture.repo.join("update.txt"), "old\n");
    assert_file(&fixture.repo.join("delete.txt"), "remove me\n");
    assert!(!fixture.repo.join("nested/add.txt").exists());
    assert!(!fixture.repo.join("empty.txt").exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(fixture.repo.join("delete.txt"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o751
        );
    }
}

#[test]
fn patch_marker_undo_refuses_a_recreated_deletion_target() {
    let fixture = Fixture::new("patch-marker-undo-conflict");
    std::fs::write(fixture.repo.join("delete.txt"), "original\n").unwrap();
    let diff = "*** Begin Patch\n*** Add File: add.txt\n+created\n*** Delete File: delete.txt\n*** End Patch\n";
    let applied = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    assert!(applied.status.success(), "{}", combined(&applied));
    std::fs::create_dir(fixture.repo.join("delete.txt")).unwrap();
    let undone = fixture.run(&["undo"]);
    assert_eq!(undone.status.code(), Some(12), "{}", combined(&undone));
    assert_file(&fixture.repo.join("add.txt"), "created\n");
    assert!(fixture.repo.join("delete.txt").is_dir());
}

#[test]
fn patch_marker_late_planning_refusals_leave_existence_unchanged() {
    let fixture = Fixture::new("patch-marker-plan-refusal");
    std::fs::write(fixture.repo.join("update.txt"), "old\n").unwrap();
    std::fs::write(fixture.repo.join("delete.txt"), "keep me\n").unwrap();
    let first = "*** Begin Patch\n*** Update File: update.txt\n@@\n-old\n+new\n*** Add File: nested/add.txt\n+created\n*** Delete File: delete.txt\n";
    for suffix in [
        "*** Delete File: missing.txt\n*** End Patch\n",
        "*** Add File: update.txt\n+collision\n*** End Patch\n",
        "*** Add File: invalid.rs\n+fn broken(\n*** End Patch\n",
        "*** Add File: ../outside.txt\n+escape\n*** End Patch\n",
        "*** Add File: nested/./add.txt\n+duplicate\n*** End Patch\n",
    ] {
        for dry_run in [false, true] {
            let args = if dry_run {
                vec!["patch", "--dry-run"]
            } else {
                vec!["patch"]
            };
            let output = fixture.run_with_stdin(&args, format!("{first}{suffix}").as_bytes());
            assert!(!output.status.success(), "{}", combined(&output));
            assert_file(&fixture.repo.join("update.txt"), "old\n");
            assert_file(&fixture.repo.join("delete.txt"), "keep me\n");
            assert!(!fixture.repo.join("nested").exists());
            assert!(!fixture.repo.join("invalid.rs").exists());
            assert!(!fixture.base.join("outside.txt").exists());
        }
    }
}

#[test]
fn patch_marker_refusals_preserve_the_entire_transaction() {
    let fixture = Fixture::new("patch-marker-refusal");
    std::fs::write(fixture.repo.join("one.txt"), "one\n").unwrap();
    std::fs::write(fixture.repo.join("repeat.txt"), "repeat\nrepeat\n").unwrap();
    let first = "*** Begin Patch\n*** Update File: one.txt\n@@\n-one\n+ONE\n";
    for (suffix, status) in [
        (
            "*** Add File: new.txt\nnew without prefix\n*** End Patch\n",
            20,
        ),
        (
            "*** Delete File: repeat.txt\n-unexpected payload\n*** End Patch\n",
            20,
        ),
        (
            "*** Update File: repeat.txt\n@@\n-repeat\n+REPEAT\n*** End Patch\n",
            13,
        ),
        (
            "*** Update File: one.txt\n@@\n-one\n+AGAIN\n*** End Patch\n",
            20,
        ),
        ("*** Update File: repeat.txt\ninvalid\n*** End Patch\n", 20),
        ("*** Unknown: repeat.txt\n*** End Patch\n", 20),
        ("", 20),
    ] {
        for dry_run in [false, true] {
            let args = if dry_run {
                vec!["patch", "--dry-run"]
            } else {
                vec!["patch"]
            };
            let output = fixture.run_with_stdin(&args, format!("{first}{suffix}").as_bytes());
            assert_eq!(output.status.code(), Some(status), "{}", combined(&output));
            assert_file(&fixture.repo.join("one.txt"), "one\n");
            assert_file(&fixture.repo.join("repeat.txt"), "repeat\nrepeat\n");
            assert!(!fixture.repo.join("new.txt").exists());
        }
    }
}

#[test]
fn patch_deleted_lua_comment_is_payload_and_roundtrips() {
    let fixture = Fixture::new("patch-lua-comment");
    let path = fixture.repo.join("comment.lua");
    let before = "-- original comment\nlocal value = 1\n";
    std::fs::write(&path, before).unwrap();
    let diff = "--- a/comment.lua\n+++ b/comment.lua\n@@ -80,2 +80,2 @@\n--- original comment\n+-- revised comment\n local value = 1\n";
    let preview = fixture.run_with_stdin(&["patch", "--dry-run"], diff.as_bytes());
    assert!(preview.status.success(), "{}", combined(&preview));
    assert_file(&path, before);
    let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(&path, "-- revised comment\nlocal value = 1\n");
    let undone = fixture.run(&["undo"]);
    assert!(undone.status.success(), "{}", combined(&undone));
    assert_file(&path, before);
}

#[test]
fn patch_header_shaped_payload_does_not_create_a_phantom_file() {
    let fixture = Fixture::new("patch-header-payload");
    std::fs::write(
        fixture.repo.join("one.txt"),
        "anchor\n-- a/phantom.txt\ntail\n",
    )
    .unwrap();
    std::fs::write(fixture.repo.join("two.txt"), "two\n").unwrap();
    let diff = "--- a/one.txt\n+++ b/one.txt\n@@ -1,3 +1,3 @@\n anchor\n--- a/phantom.txt\n+++ b/phantom.txt\n tail\n--- a/two.txt\n+++ b/two.txt\n@@ -1 +1 @@\n-two\n+TWO\n";
    let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    assert!(output.status.success(), "{}", combined(&output));
    assert_file(
        &fixture.repo.join("one.txt"),
        "anchor\n++ b/phantom.txt\ntail\n",
    );
    assert_file(&fixture.repo.join("two.txt"), "TWO\n");
    assert!(!fixture.repo.join("phantom.txt").exists());
}

#[test]
fn patch_malformed_ranges_and_count_free_header_ambiguity_are_atomic() {
    let fixture = Fixture::new("patch-header-atomic");
    std::fs::write(fixture.repo.join("one.txt"), "one\n").unwrap();
    std::fs::write(fixture.repo.join("two.txt"), "two\n").unwrap();
    let first = "--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-one\n+ONE\n";
    for (suffix, diagnostic) in [

        (
            "--- a/two.txt\n+++ b/two.txt\n@@\n-two\n+TWO\n--- a/phantom.txt\n+++ b/phantom.txt\n@@\n-missing\n+new\n",
            "ambiguous in a count-free hunk",
        ),
        ("--- a/two.txt\n+++ b/two.txt\n@@ -1,x +1,1 @@\n-two\n+TWO\n", "has invalid unified-diff ranges"),
        ("--- a/two.txt\n+++ b/two.txt\n@@ -184467440737095516160 +1,1 @@\n-two\n+TWO\n", "has invalid unified-diff ranges"),
        ("--- a/two.txt\n+++ b/two.txt\n@@ +1,1 -1,1 @@\n-two\n+TWO\n", "has invalid unified-diff ranges"),
        ("--- a/two.txt\n+++ b/two.txt\n@@ -1,1 +1,1\n-two\n+TWO\n", "has invalid unified-diff ranges"),
    ] {
        let diff = format!("{first}{suffix}");
        let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
        assert_eq!(output.status.code(), Some(20), "{}", combined(&output));
        assert!(
            combined(&output).contains(diagnostic),
            "{}",
            combined(&output)
        );
        assert_file(&fixture.repo.join("one.txt"), "one\n");
        assert_file(&fixture.repo.join("two.txt"), "two\n");
        assert!(!fixture.repo.join("phantom.txt").exists());
    }
}

#[test]
fn patch_ambiguity_identifies_late_input_hunk_and_bounded_source_candidates() {
    let fixture = Fixture::new("patch-late-ambiguity");
    let original = "first\nsecond\nrepeat\nrepeat\nrepeat\nrepeat\nrepeat\nrepeat\nrepeat\nlast\n";
    std::fs::write(fixture.repo.join("many.txt"), original).unwrap();
    let diff = "--- a/many.txt\n+++ b/many.txt\n@@ -1 +1,2 @@\n-first\n+FIRST\n+inserted\n@@ -2 +3 @@\n-second\n+SECOND\n@@ -10 +11 @@\n-last\n+LAST\n@@ -20 +21 @@\n-repeat\n+REPEAT\n";

    let output = fixture.run_with_stdin(&["patch"], diff.as_bytes());
    let text = combined(&output);

    assert_eq!(output.status.code(), Some(13), "{text}");
    assert!(text.contains("input hunk 4 at patch line 13"), "{text}");
    assert!(
        text.contains("candidate source lines 3, 4, 5, 6, 7, and 2 more"),
        "{text}"
    );
    assert!(text.contains("nothing written"), "{text}");
    assert_file(&fixture.repo.join("many.txt"), original);
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
fn replace_text_refuses_js_string_newlines_from_stdin_atomically() {
    let fixture = Fixture::new("js-linebreak");
    let source = "const x = document.querySelectorAll('ABC');\n";
    for path in [
        "example.js",
        "example.mjs",
        "example.cjs",
        "example.ts",
        "example.tsx",
    ] {
        std::fs::write(fixture.repo.join(path), source).unwrap();
        for replacement in [b"XYZ\n".as_slice(), b"XYZ\r\n", b"XYZ\r"] {
            for dry_run in [false, true] {
                let mut args = vec!["replace-text", path, "ABC"];
                if dry_run {
                    args.push("--dry-run");
                }
                let refused = fixture.run_with_stdin(&args, replacement);
                assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
                let output = combined(&refused);
                assert!(output.contains("unescaped line break"), "{output}");
                assert!(output.contains(&format!("{path}:1:")), "{output}");
                assert!(output.contains("nothing written"), "{output}");
                assert_file(&fixture.repo.join(path), source);
            }
        }
        let accepted = fixture.run_with_stdin(&["replace-text", path, "ABC"], b"XYZ\\n");
        assert!(accepted.status.success(), "{}", combined(&accepted));
        assert_file(
            &fixture.repo.join(path),
            "const x = document.querySelectorAll('XYZ\\n');\n",
        );
    }
}

#[test]
fn replace_text_preserves_multiline_jsx_attribute_strings() {
    let fixture = Fixture::new("jsx-attribute");
    for path in ["example.jsx", "example.tsx"] {
        std::fs::write(
            fixture.repo.join(path),
            "const view = <div title=\"ABC\" />;\n",
        )
        .unwrap();
        let accepted = fixture.run_with_stdin(&["replace-text", path, "ABC"], b"XYZ\n");
        assert!(accepted.status.success(), "{}", combined(&accepted));
        assert_file(
            &fixture.repo.join(path),
            "const view = <div title=\"XYZ\n\" />;\n",
        );
    }
}

#[test]
fn replace_text_accepts_typescript_import_type_and_preserves_atomicity() {
    let fixture = Fixture::new("replace-text-typescript-import-type");
    let source = "import { vi } from \"vitest\";\nconst marker = 1;\n";
    let replacement = r#"vi.mock("node:child_process", async (importOriginal) => {
  const original = await importOriginal<typeof import("node:child_process")>();
  return { ...original, spawn: vi.fn(original.spawn) };
});"#;
    std::fs::write(fixture.repo.join("example.test.ts"), source).unwrap();

    let dry_run = fixture.run(&[
        "replace-text",
        "example.test.ts",
        "const marker = 1;",
        replacement,
        "--dry-run",
    ]);
    assert!(dry_run.status.success(), "{}", combined(&dry_run));
    assert_file(&fixture.repo.join("example.test.ts"), source);

    let written = fixture.run(&[
        "replace-text",
        "example.test.ts",
        "const marker = 1;",
        replacement,
    ]);
    assert!(written.status.success(), "{}", combined(&written));
    let expected = format!("import {{ vi }} from \"vitest\";\n{replacement}\n");
    assert_file(&fixture.repo.join("example.test.ts"), &expected);

    let refused = fixture.run(&[
        "replace-text",
        "example.test.ts",
        "return { ...original, spawn: vi.fn(original.spawn) };",
        "return { ...original, spawn: ;",
    ]);
    assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
    assert!(combined(&refused).contains("nothing written"));
    assert_file(&fixture.repo.join("example.test.ts"), &expected);
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
    assert!(text.contains("no existing line to anchor on"), "{text}");
    assert!(text.contains("include an unchanged context line"), "{text}");
    assert!(text.contains("nothing written"), "{text}");
    assert_file(&fixture.repo.join("existing.txt"), "before\n");
}

#[test]
fn cpp_header_patch_accepts_valid_member_and_refuses_malformed_edit_atomically() {
    let fixture = Fixture::new("cpp-header");
    let path = fixture.repo.join("layer.h");
    let before = "#include <memory>\nnamespace KWin {\nclass GLFramebuffer;\nclass Layer {\n    std::unique_ptr<GLFramebuffer> buffer;\n};\n}\n";
    std::fs::write(&path, before).unwrap();
    let valid = "--- a/layer.h\n+++ b/layer.h\n@@\n class GLFramebuffer;\n+class GLRenderTimeQuery;\n@@\n     std::unique_ptr<GLFramebuffer> buffer;\n+    std::unique_ptr<GLRenderTimeQuery> query;\n";
    let preview = fixture.run_with_stdin(&["patch", "--dry-run"], valid.as_bytes());
    assert!(preview.status.success(), "{}", combined(&preview));
    assert_file(&path, before);
    let applied = fixture.run_with_stdin(&["patch"], valid.as_bytes());
    assert!(applied.status.success(), "{}", combined(&applied));
    let after = before
        .replace("class GLFramebuffer;", "class GLFramebuffer;\nclass GLRenderTimeQuery;")
        .replace("    std::unique_ptr<GLFramebuffer> buffer;", "    std::unique_ptr<GLFramebuffer> buffer;\n    std::unique_ptr<GLRenderTimeQuery> query;");
    assert_file(&path, &after);
    let invalid = "--- a/layer.h\n+++ b/layer.h\n@@\n-    std::unique_ptr<GLRenderTimeQuery> query;\n+    std::unique_ptr<GLRenderTimeQuery> query( ;\n";
    let refused = fixture.run_with_stdin(&["patch"], invalid.as_bytes());
    assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
    assert!(combined(&refused).contains("nothing written"));
    assert_file(&path, &after);
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

#[test]
fn nested_root_file_edits_select_the_subdir_not_cwd_or_repo_sentinels() {
    let fixture = Fixture::new("nested-root-edits");
    let (cwd, nested) = nested_collision(&fixture);
    let absolute = nested.to_str().unwrap();
    let trailing = format!("{absolute}/");
    let mut roots = vec![
        ("absolute", absolute.to_string()),
        ("relative", "repo/etc".to_string()),
        ("trailing-slash", trailing),
    ];
    #[cfg(unix)]
    {
        let link = cwd.join("etc-link");
        std::os::unix::fs::symlink(&nested, &link).unwrap();
        roots.push(("symlink", "etc-link".to_string()));
    }

    let first_root = roots[0].1.clone();
    let dry_once = fixture.run_in(
        &cwd,
        &[
            "--root",
            &first_root,
            "replace-text",
            "probe.conf",
            "SUBDIR_SENTINEL",
            "DRY_RUN",
            "--dry-run",
        ],
    );
    assert_eq!(dry_once.status.code(), Some(0), "{}", combined(&dry_once));
    assert_collision_untouched(&fixture, &nested, "SUBDIR_SENTINEL\n");
    let undo_after_dry = fixture.run(&["undo"]);
    let undo_dry_text = combined(&undo_after_dry);
    assert!(
        undo_dry_text.contains("nothing to undo"),
        "dry-run must not journal: {undo_dry_text}"
    );

    for (label, root) in &roots {
        std::fs::write(nested.join("probe.conf"), "SUBDIR_SENTINEL\n").unwrap();
        let dry = fixture.run_in(
            &cwd,
            &[
                "--root",
                root,
                "replace-text",
                "probe.conf",
                "SUBDIR_SENTINEL",
                "DRY_RUN",
                "--dry-run",
            ],
        );
        let dry_text = combined(&dry);
        assert_eq!(dry.status.code(), Some(0), "{label}: {dry_text}");
        assert!(
            dry_text.contains("would apply etc/probe.conf"),
            "{label}: {dry_text}"
        );
        assert!(!dry_text.contains("CWD_SENTINEL"), "{label}: {dry_text}");
        assert!(!dry_text.contains("REPO_SENTINEL"), "{label}: {dry_text}");
        assert_collision_untouched(&fixture, &nested, "SUBDIR_SENTINEL\n");

        let applied = fixture.run_in(
            &cwd,
            &[
                "--root",
                root,
                "replace-text",
                "probe.conf",
                "SUBDIR_SENTINEL",
                "SUBDIR_REPLACED",
            ],
        );
        let applied_text = combined(&applied);
        assert_eq!(applied.status.code(), Some(0), "{label}: {applied_text}");
        assert!(
            applied_text.contains("etc/probe.conf"),
            "{label}: {applied_text}"
        );
        assert!(
            applied_text.contains("SUBDIR_REPLACED"),
            "{label}: {applied_text}"
        );
        assert!(
            !applied_text.contains("CWD_SENTINEL"),
            "{label}: {applied_text}"
        );
        assert!(
            !applied_text.contains("REPO_SENTINEL"),
            "{label}: {applied_text}"
        );
        assert_collision_untouched(&fixture, &nested, "SUBDIR_REPLACED\n");

        std::fs::write(nested.join("probe.conf"), "line-one\nline-two\n").unwrap();
        let lines = fixture.run_in(
            &cwd,
            &[
                "--root",
                root,
                "replace-lines",
                "probe.conf",
                "1:1",
                "LINE-ONE",
            ],
        );
        assert_eq!(
            lines.status.code(),
            Some(0),
            "{label}: {}",
            combined(&lines)
        );
        assert!(
            combined(&lines).contains("etc/probe.conf"),
            "{}",
            combined(&lines)
        );
        assert_file(&nested.join("probe.conf"), "LINE-ONE\nline-two\n");
        assert_file(&fixture.repo.join("probe.conf"), "REPO_SENTINEL\n");

        let inserted = fixture.run_in(
            &cwd,
            &[
                "--root",
                root,
                "insert-lines",
                "probe.conf",
                "0",
                "INSERTED",
            ],
        );
        assert_eq!(
            inserted.status.code(),
            Some(0),
            "{label}: {}",
            combined(&inserted)
        );
        assert_file(&nested.join("probe.conf"), "INSERTED\nLINE-ONE\nline-two\n");

        let deleted = fixture.run_in(&cwd, &["--root", root, "delete-lines", "probe.conf", "1:1"]);
        assert_eq!(
            deleted.status.code(),
            Some(0),
            "{label}: {}",
            combined(&deleted)
        );
        assert_file(&nested.join("probe.conf"), "LINE-ONE\nline-two\n");

        let written = fixture.run_in(
            &cwd,
            &["--root", root, "write", "created.conf", "CREATED\n"],
        );
        let written_text = combined(&written);
        assert_eq!(written.status.code(), Some(0), "{label}: {written_text}");
        assert!(
            written_text.contains("etc/created.conf"),
            "{label}: {written_text}"
        );
        assert_file(&nested.join("created.conf"), "CREATED\n");
        assert!(!cwd.join("created.conf").exists());
        assert!(!fixture.repo.join("created.conf").exists());
        std::fs::remove_file(nested.join("created.conf")).unwrap();
    }

    std::fs::remove_file(nested.join("probe.conf")).unwrap();
    let missing = fixture.run_in(
        &cwd,
        &[
            "--root",
            nested.to_str().unwrap(),
            "replace-text",
            "probe.conf",
            "REPO_SENTINEL",
            "SHOULD_NOT",
        ],
    );
    let missing_text = combined(&missing);
    assert_ne!(missing.status.code(), Some(0), "{missing_text}");
    assert!(
        missing_text.contains("no file `probe.conf`"),
        "{missing_text}"
    );
    assert_file(&fixture.repo.join("probe.conf"), "REPO_SENTINEL\n");
    assert_file(&cwd.join("probe.conf"), "CWD_SENTINEL\n");
}

#[test]
fn nested_root_patch_stays_atomic_and_workspace_relative() {
    let fixture = Fixture::new("nested-root-patch");
    let (cwd, nested) = nested_collision(&fixture);
    std::fs::write(nested.join("first.txt"), "one\n").unwrap();
    std::fs::write(nested.join("second.txt"), "two\n").unwrap();
    std::fs::write(fixture.repo.join("first.txt"), "REPO_FIRST\n").unwrap();
    std::fs::write(fixture.repo.join("second.txt"), "REPO_SECOND\n").unwrap();
    let root = nested.to_str().unwrap();

    let ok = fixture.run_in(
        &cwd,
        &[
            "--root",
            root,
            "patch",
            "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-one\n+ONE\n--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n",
        ],
    );
    let ok_text = combined(&ok);
    assert_eq!(ok.status.code(), Some(0), "{ok_text}");
    assert!(ok_text.contains("etc/first.txt"), "{ok_text}");
    assert!(ok_text.contains("etc/second.txt"), "{ok_text}");
    assert_file(&nested.join("first.txt"), "ONE\n");
    assert_file(&nested.join("second.txt"), "TWO\n");
    assert_file(&fixture.repo.join("first.txt"), "REPO_FIRST\n");
    assert_file(&fixture.repo.join("second.txt"), "REPO_SECOND\n");

    std::fs::write(nested.join("first.txt"), "one\n").unwrap();
    std::fs::write(nested.join("second.txt"), "two\n").unwrap();
    let failed = fixture.run_in(
        &cwd,
        &[
            "--root",
            root,
            "patch",
            "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-one\n+ONE\n--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-missing\n+TWO\n",
        ],
    );
    let failed_text = combined(&failed);
    assert_eq!(failed.status.code(), Some(13), "{failed_text}");
    assert!(failed_text.contains("nothing written"), "{failed_text}");
    assert_file(&nested.join("first.txt"), "one\n");
    assert_file(&nested.join("second.txt"), "two\n");
    assert_file(&fixture.repo.join("first.txt"), "REPO_FIRST\n");
}

#[test]
fn undo_from_repo_root_reverses_a_nested_root_edit() {
    let fixture = Fixture::new("nested-root-undo");
    let (cwd, nested) = nested_collision(&fixture);
    let output = fixture.run_in(
        &cwd,
        &[
            "--root",
            nested.to_str().unwrap(),
            "replace-text",
            "probe.conf",
            "SUBDIR_SENTINEL",
            "SUBDIR_EDITED",
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", combined(&output));
    assert_collision_untouched(&fixture, &nested, "SUBDIR_EDITED\n");

    let undone = fixture.run(&["undo"]);
    let undone_text = combined(&undone);
    assert_eq!(undone.status.code(), Some(0), "{undone_text}");
    assert!(undone_text.contains("etc/probe.conf"), "{undone_text}");
    assert_collision_untouched(&fixture, &nested, "SUBDIR_SENTINEL\n");
}

#[test]
fn nested_root_read_handle_drives_replace_span() {
    let fixture = Fixture::new("nested-root-handle");
    let (cwd, nested) = nested_collision(&fixture);
    let read = fixture.run_in(
        &cwd,
        &[
            "--root",
            nested.to_str().unwrap(),
            "read-file",
            "probe.conf",
            "--handle",
            "--all",
        ],
    );
    let read_text = combined(&read);
    assert_eq!(read.status.code(), Some(0), "{read_text}");
    assert!(read_text.contains("etc/probe.conf"), "{read_text}");
    assert!(read_text.contains("SUBDIR_SENTINEL"), "{read_text}");
    assert!(!read_text.contains("CWD_SENTINEL"), "{read_text}");
    assert!(!read_text.contains("REPO_SENTINEL"), "{read_text}");
    let handle = read_text
        .lines()
        .find_map(|line| line.strip_prefix("handle: "))
        .expect("read-file handle");

    let replaced = fixture.run(&["replace-span", handle, "FROM_HANDLE\n"]);
    let replaced_text = combined(&replaced);
    assert_eq!(replaced.status.code(), Some(0), "{replaced_text}");
    assert_collision_untouched(&fixture, &nested, "FROM_HANDLE\n");
}

#[test]
fn omitted_root_keeps_repo_relative_file_operands() {
    let fixture = Fixture::new("omitted-root-legacy");
    let (_cwd, nested) = nested_collision(&fixture);
    let output = fixture.run(&["replace-text", "probe.conf", "REPO_SENTINEL", "REPO_EDITED"]);
    let text = combined(&output);
    assert_eq!(output.status.code(), Some(0), "{text}");
    assert!(text.contains("applied probe.conf"), "{text}");
    assert!(!text.contains("etc/probe.conf"), "{text}");
    assert_file(&fixture.repo.join("probe.conf"), "REPO_EDITED\n");
    assert_file(&nested.join("probe.conf"), "SUBDIR_SENTINEL\n");
    assert_file(&fixture.base.join("probe.conf"), "CWD_SENTINEL\n");
}

#[test]
fn replace_rust_attributes_roundtrip_and_refusal_are_atomic() {
    let fixture = Fixture::new("rust-outer-attributes");
    let file = fixture.repo.join("probe.rs");
    let suffix = "\n\nfn neighbor() { let _ = 99; }\n";
    let original = format!("#[inline]\n#[allow(dead_code)]\nfn probe() {{ let _ = 1; }}{suffix}");
    std::fs::write(&file, &original).unwrap();
    let read = fixture.run(&["read", "probe.rs::Function::probe"]);
    assert_eq!(read.status.code(), Some(0), "{}", combined(&read));
    let replacement = "#[inline]\n#[allow(dead_code)]\nfn probe() { let _ = 2; }";
    let replaced = fixture.run(&["replace", "probe.rs::Function::probe", replacement]);
    assert_eq!(replaced.status.code(), Some(0), "{}", combined(&replaced));
    assert_file(&file, &format!("{replacement}{suffix}"));

    let plain = "fn probe() { let _ = 3; }";
    let replaced = fixture.run(&["replace", "probe.rs::Function::probe", plain]);
    assert_eq!(replaced.status.code(), Some(0), "{}", combined(&replaced));
    assert_file(
        &file,
        &format!("#[inline]\n#[allow(dead_code)]\n{plain}{suffix}"),
    );

    let changed_attributes = "#[cold]\nfn probe() { let _ = 4; }";
    let replaced = fixture.run(&["replace", "probe.rs::Function::probe", changed_attributes]);
    assert_eq!(replaced.status.code(), Some(0), "{}", combined(&replaced));
    let expected = format!("{changed_attributes}{suffix}");
    assert_file(&file, &expected);

    let refused = fixture.run(&[
        "replace",
        "probe.rs::Function::probe",
        "#[cold]\nfn probe( {",
    ]);
    assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
    assert_file(&file, &expected);
}

#[test]
fn replace_rust_attributed_method_keeps_indentation_and_body_edits() {
    let fixture = Fixture::new("rust-method-attributes");
    let file = fixture.repo.join("probe.rs");
    let before =
        "struct Counter;\nimpl Counter {\n    #[inline]\n    fn probe(&self) { let _ = 1; }\n}\n";
    std::fs::write(&file, before).unwrap();
    let replacement = "    #[cold]\n    fn probe(&self) { let _ = 2; }";
    let replaced = fixture.run(&["replace", "probe.rs::Function::probe", replacement]);
    assert_eq!(replaced.status.code(), Some(0), "{}", combined(&replaced));
    assert_file(
        &file,
        &format!("struct Counter;\nimpl Counter {{\n{replacement}\n}}\n"),
    );
    let body = fixture.run(&[
        "replace",
        "probe.rs::Function::probe",
        "{ let _ = 3; }",
        "--body",
    ]);
    assert_eq!(body.status.code(), Some(0), "{}", combined(&body));
    let text = std::fs::read_to_string(&file).unwrap();
    assert_eq!(text.matches("#[cold]").count(), 1, "{text}");
    assert!(!text.contains("#[inline]"), "{text}");
    assert!(text.contains("let _ = 3;"), "{text}");
}

#[test]
fn rust_source_views_and_handles_own_outer_attributes_but_not_the_next_definition() {
    let fixture = Fixture::new("rust-attribute-source-handle");
    let file = fixture.repo.join("probe.rs");
    let original = "struct Runtime;\nimpl Runtime {\n    #[cfg(feature = \"audio\")]\n    #[wasm_bindgen(\n        js_name = takeAudio\n    )]\n    pub fn take_audio(&mut self) {\n        self.flush();\n    }\n    #[inline]\n    fn neighbor(&self) {}\n}\n";
    std::fs::write(&file, original).unwrap();

    let search = fixture.run(&[
        "search-symbol",
        "take_audio",
        "--path",
        "probe.rs",
        "--code",
    ]);
    let search_text = combined(&search);
    assert_eq!(search.status.code(), Some(0), "{search_text}");
    assert!(
        search_text.contains("#[cfg(feature = \"audio\")]"),
        "{search_text}"
    );
    assert!(search_text.contains("js_name = takeAudio"), "{search_text}");
    assert!(!search_text.contains("fn neighbor"), "{search_text}");

    let structured = fixture.run(&[
        "search-pattern",
        "take_audio",
        "--fixed",
        "--path",
        "probe.rs",
        "--code",
        "--json",
    ]);
    assert_eq!(
        structured.status.code(),
        Some(0),
        "{}",
        combined(&structured)
    );
    let value: serde_json::Value = serde_json::from_slice(&structured.stdout).unwrap();
    let definition = &value["hits"][0];
    assert_eq!(definition["span"]["start_line"], 3);
    assert_eq!(definition["span"]["end_line"], 9);
    assert!(definition["source"]
        .as_str()
        .unwrap()
        .contains("js_name = takeAudio"));
    assert!(!definition["source"]
        .as_str()
        .unwrap()
        .contains("fn neighbor"));
    let handle = definition["handle"].as_str().expect("definition handle");
    let replacement = "    #[cold]\n    fn take_audio(&mut self) { self.flush(); }\n";
    let replaced = fixture.run(&["replace-span", handle, replacement]);
    assert_eq!(replaced.status.code(), Some(0), "{}", combined(&replaced));
    let expected = "struct Runtime;\nimpl Runtime {\n    #[cold]\n    fn take_audio(&mut self) { self.flush(); }\n    #[inline]\n    fn neighbor(&self) {}\n}\n";
    assert_file(&file, expected);

    let stale = fixture.run(&["replace-span", handle, "    fn take_audio(&mut self) {}\n"]);
    assert!(!stale.status.success(), "{}", combined(&stale));
    assert_file(&file, expected);
}

#[cfg(unix)]
#[test]
fn verify_failure_and_unavailable_are_nonzero_in_cli_and_json() {
    for (tag, compiler, expected) in [
        (
            "failed",
            "#!/bin/sh\nprintf 'error: intended verifier failure\\n' >&2\nexit 7\n",
            "failed",
        ),
        (
            "unavailable",
            "#!/greppy-test-missing-interpreter\n",
            "unavailable",
        ),
    ] {
        let fixture = Fixture::new(tag);
        std::fs::write(fixture.repo.join("ui.ts"), "const oldValue = 1;\n").unwrap();
        install_fake_tsc(&fixture, compiler);
        let output = fixture.run(&[
            "replace-text",
            "ui.ts",
            "oldValue",
            "newValue",
            "--verify",
            "--json",
        ]);
        assert_eq!(output.status.code(), Some(17), "{}", combined(&output));
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["published"], true);
        assert_eq!(value["exit_code"], 17);
        assert_eq!(value["verify"]["status"], expected);
        assert_eq!(value["verify"]["exit_code"], 17);
        assert_file(&fixture.repo.join("ui.ts"), "const newValue = 1;\n");
        assert!(combined(&output).contains("edit remains applied"));
        // Explicit verification must also run when the requested bytes already exist.
        for args in [
            vec![
                "replace-text",
                "ui.ts",
                "newValue",
                "newValue",
                "--verify",
                "--json",
            ],
            vec![
                "write",
                "ui.ts",
                "const newValue = 1;\n",
                "--verify",
                "--json",
            ],
        ] {
            let repeated = fixture.run(&args);
            assert_eq!(repeated.status.code(), Some(17), "{}", combined(&repeated));
            let value: serde_json::Value = serde_json::from_slice(&repeated.stdout).unwrap();
            assert_eq!(value["verify"]["status"], expected);
            assert_eq!(value["exit_code"], 17);
            assert_file(&fixture.repo.join("ui.ts"), "const newValue = 1;\n");
        }
    }
}

#[test]
fn python_body_indentation_is_checked_before_publication() {
    let fixture = Fixture::new("python-body-indentation");
    let before = "def clamp(value, lower, upper):\n    return min(lower, max(upper, value))\n";
    std::fs::write(fixture.repo.join("limits.py"), before).unwrap();
    let refused = fixture.run(&[
        "replace",
        "clamp",
        "--body",
        "return max(lower, min(upper, value))",
    ]);
    assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
    assert!(combined(&refused).contains("indentation"));
    assert!(combined(&refused).contains("nothing written"));
    assert_file(&fixture.repo.join("limits.py"), before);
    let applied = fixture.run(&[
        "replace",
        "clamp",
        "--body",
        "    return max(lower, min(upper, value))",
    ]);
    assert!(applied.status.success(), "{}", combined(&applied));
    assert_file(
        &fixture.repo.join("limits.py"),
        "def clamp(value, lower, upper):\n    return max(lower, min(upper, value))\n",
    );
}

#[test]
fn verify_reports_syntax_success_without_claiming_tests_passed() {
    let fixture = Fixture::new("verify-syntax-coverage");
    std::fs::write(fixture.repo.join("a.py"), "VALUE = 1\n").unwrap();
    let output = fixture.run(&["replace-text", "a.py", "1", "2", "--verify", "--json"]);
    assert!(output.status.success(), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "not_run");
    assert_eq!(value["verify"]["checks"][0]["scope"], "syntax");
    assert_eq!(value["verify"]["checks"][0]["status"], "passed");
    assert!(combined(&output).contains("tests not run"));
}

#[test]
fn verify_selected_python_tests_catch_import_cycle_despite_valid_syntax() {
    let fixture = Fixture::new("verify-import-cycle");
    std::fs::create_dir(fixture.repo.join("tests")).unwrap();
    std::fs::write(fixture.repo.join("a.py"), "VALUE = 1\n").unwrap();
    std::fs::write(fixture.repo.join("b.py"), "from a import VALUE\n").unwrap();
    std::fs::write(fixture.repo.join("tests/test_imports.py"),
        "import unittest\nimport a\nclass TestImports(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(a.VALUE, 1)\n").unwrap();
    let run = |old, new| {
        fixture
            .command()
            .env(
                "GREPPY_VERIFY_TEST_COMMAND",
                "python3 -m unittest discover -s tests",
            )
            .args(["replace-text", "a.py", old, new, "--verify", "--json"])
            .output()
            .unwrap()
    };
    let broken = run("VALUE = 1", "from b import VALUE\nVALUE = 1");
    assert_eq!(broken.status.code(), Some(17), "{}", combined(&broken));
    let value: serde_json::Value = serde_json::from_slice(&broken.stdout).unwrap();
    assert_eq!(value["verify"]["checks"][0]["scope"], "syntax");
    assert_eq!(value["verify"]["checks"][0]["status"], "passed");
    assert_eq!(value["verify"]["tests_status"], "failed");
    assert_eq!(value["verify"]["checks"][1]["scope"], "tests");
    assert!(
        combined(&broken).contains("ImportError"),
        "{}",
        combined(&broken)
    );
    assert!(combined(&broken).contains("edit remains applied"));
    assert_file(
        &fixture.repo.join("a.py"),
        "from b import VALUE\nVALUE = 1\n",
    );
    let repaired = run("from b import VALUE\nVALUE = 1", "VALUE = 1");
    assert!(repaired.status.success(), "{}", combined(&repaired));
    let value: serde_json::Value = serde_json::from_slice(&repaired.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "passed");
    assert_eq!(value["verify"]["checks"][1]["status"], "passed");
}

#[cfg(unix)]
#[test]
fn verify_selected_test_pipeline_preserves_failure() {
    let fixture = Fixture::new("verify-test-pipeline");
    std::fs::write(fixture.repo.join("a.py"), "VALUE = 1\n").unwrap();
    let output = fixture
        .command()
        .env(
            "GREPPY_VERIFY_TEST_COMMAND",
            "python3 -c 'raise RuntimeError(\"pipeline failure\")' | head -1",
        )
        .args(["replace-text", "a.py", "1", "2", "--verify", "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(17), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "failed");
}

#[test]
fn verify_rejects_empty_selected_tests_instead_of_claiming_success() {
    let fixture = Fixture::new("verify-empty-tests");
    std::fs::write(fixture.repo.join("a.py"), "VALUE = 1\n").unwrap();
    let output = fixture
        .command()
        .env("GREPPY_VERIFY_TEST_COMMAND", "  ")
        .args(["replace-text", "a.py", "1", "2", "--verify", "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(17), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "unavailable");
}

#[cfg(unix)]
#[test]
fn verify_selected_tests_timeout_without_undoing_edit() {
    let fixture = Fixture::new("verify-test-timeout");
    std::fs::write(fixture.repo.join("a.txt"), "old\n").unwrap();
    let output = fixture
        .command()
         .env("GREPPY_VERIFY_TEST_COMMAND", "python3 -c \"import os,time;open('owned-test.pid','w').write(str(os.getpid()));time.sleep(30)\"")
        .env("GREPPY_EDIT_VERIFY_TIMEOUT_SECS", "1")
        .args(["replace-text", "a.txt", "old", "new", "--verify", "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(17), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "timed_out");
    assert_file(&fixture.repo.join("a.txt"), "new\n");
    let pid: i32 = std::fs::read_to_string(fixture.repo.join("owned-test.pid"))
        .expect("selected test actually started")
        .parse()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while unsafe { libc::kill(pid, 0) } == 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_ne!(
        unsafe { libc::kill(pid, 0) },
        0,
        "selected test process survived timeout"
    );
}

#[test]
fn verify_does_not_call_immediate_exit_124_a_timeout() {
    let fixture = Fixture::new("verify-exit124");
    std::fs::write(fixture.repo.join("a.py"), "VALUE = 1\n").unwrap();
    let output = fixture
        .command()
        .env(
            "GREPPY_VERIFY_TEST_COMMAND",
            "python3 -c 'import sys;sys.exit(124)'",
        )
        .args(["replace-text", "a.py", "1", "2", "--verify", "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(17), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "failed");
    assert!(!combined(&output).contains("timed out"));
}

#[test]
fn verify_does_not_leak_private_completion_path_to_test_command() {
    let fixture = Fixture::new("verify-private-channel");
    std::fs::write(fixture.repo.join("a.py"), "VALUE = 1\n").unwrap();
    let output = fixture.command()
        .env("GREPPY_VERIFY_TEST_COMMAND", "python3 -c 'import os;assert \"GREPPY_INTERNAL_VERIFY_STATUS_PATH\" not in os.environ'")
        .args(["replace-text", "a.py", "1", "2", "--verify", "--json"])
        .output().unwrap();
    assert!(output.status.success(), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "passed");
}

#[cfg(unix)]
#[test]
fn verify_reaps_pipe_holding_descendant_after_test_leader_exits() {
    let fixture = Fixture::new("verify-exited-leader");
    std::fs::write(fixture.repo.join("a.txt"), "old\n").unwrap();
    let output = fixture.command()
        .env("GREPPY_VERIFY_TEST_COMMAND", "python3 -c \"import subprocess;p=subprocess.Popen(['sleep','30']);open('owned-test.pid','w').write(str(p.pid))\"")
        .env("GREPPY_EDIT_VERIFY_TIMEOUT_SECS", "1")
        .args(["replace-text", "a.txt", "old", "new", "--verify", "--json"])
        .output().unwrap();
    assert_eq!(output.status.code(), Some(17), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["verify"]["tests_status"], "timed_out");
    let pid: i32 = std::fs::read_to_string(fixture.repo.join("owned-test.pid"))
        .unwrap()
        .parse()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while unsafe { libc::kill(pid, 0) } == 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_ne!(
        unsafe { libc::kill(pid, 0) },
        0,
        "pipe-holding test descendant survived timeout"
    );
}

#[cfg(unix)]
#[test]
fn verify_completion_keeps_shared_temp_permissions_and_cleans_private_capture() {
    use std::os::unix::fs::PermissionsExt as _;
    for code in [0, 3] {
        let fixture = Fixture::new("verify-shared-temp");
        let shared = fixture.base.join("shared-temp");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o1777)).unwrap();
        std::fs::write(fixture.repo.join("a.txt"), "old\n").unwrap();
        let command = format!(
            "python3 -c 'import os, pathlib, stat, sys; roots=list(pathlib.Path(os.environ[\"TMPDIR\"]).glob(\"greppy-verify-*\")); assert len(roots)==1; assert stat.S_IMODE(roots[0].stat().st_mode)==0o700; sys.exit({code})'"
        );
        let output = fixture
            .command()
            .env("TMPDIR", &shared)
            .env("GREPPY_VERIFY_TEST_COMMAND", command)
            .args(["replace-text", "a.txt", "old", "new", "--verify", "--json"])
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(if code == 0 { 0 } else { 17 }),
            "{}",
            combined(&output)
        );
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            value["verify"]["tests_status"],
            if code == 0 { "passed" } else { "failed" }
        );
        assert_eq!(
            std::fs::metadata(&shared).unwrap().permissions().mode() & 0o7777,
            0o1777
        );
        assert!(
            !std::fs::read_dir(&shared).unwrap().any(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("greppy-verify-")
            }),
            "private capture was not removed"
        );
    }
}

#[test]
fn old_that_matches_nowhere_points_at_the_whitespace_variant() {
    let fixture = Fixture::new("nearest-old");
    std::fs::write(
        fixture.repo.join("a.txt"),
        "header line\n    if value > HIGH:\n        return value\nfooter\n",
    )
    .unwrap();
    let out = fixture.run(&[
        "replace-text",
        "a.txt",
        "if value > HIGH:\n    return value",
        "if value > HIGH:\n    return HIGH",
    ]);
    assert_eq!(out.status.code(), Some(13), "{}", combined(&out));
    let text = combined(&out);
    assert!(text.contains("OLD occurs 0 times"), "{text}");
    assert!(
        text.contains("nearest match differs only in whitespace at a.txt:2-3"),
        "{text}"
    );
    assert!(text.contains("greppy replace-lines a.txt 2:3"), "{text}");
    assert!(
        text.contains("\n    if value > HIGH:\n        return value"),
        "{text}"
    );
    assert_file(
        &fixture.repo.join("a.txt"),
        "header line\n    if value > HIGH:\n        return value\nfooter\n",
    );
}

#[test]
fn rename_receipt_lines_stay_exact_when_the_new_name_is_longer() {
    // changed_byte_ranges are in the original file's coordinates; shifting
    // them by the length delta moved every later site onto an earlier line.
    let fixture = Fixture::new("rename-receipt-lines");
    std::fs::write(
        fixture.repo.join("lib.rs"),
        "pub fn a1() {}\n// pad\n// pad\npub fn b() { a1(); }\n// pad\n// pad\npub fn c() { a1(); }\n// pad\n// pad\npub fn d() { a1(); }\n",
    )
    .unwrap();
    let index = fixture.run(&["index", "."]);
    assert!(
        index.status.success(),
        "{}",
        String::from_utf8_lossy(&index.stderr)
    );
    let out = fixture.run(&["rename", "a1", "a_much_longer_replacement_name"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stdout={stdout} stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("lib.rs:1,4,7,10"), "{stdout}");
    let content = std::fs::read_to_string(fixture.repo.join("lib.rs")).unwrap();
    assert_eq!(content.matches("a_much_longer_replacement_name").count(), 4);
}

#[test]
fn replace_lines_syntax_error_names_allow_syntax_errors() {
    let fixture = Fixture::new("syntax-refuse-next");
    let before = "fn before() {}\n";
    std::fs::write(fixture.repo.join("item.rs"), before).unwrap();
    let refused = fixture.run(&["replace-lines", "item.rs", "1:1", "fn after( {}"]);
    assert_eq!(refused.status.code(), Some(13), "{}", combined(&refused));
    let text = combined(&refused);
    assert!(text.contains("--allow-syntax-errors"), "{text}");
    assert!(
        text.contains(
            "next: make the whole change in one `greppy patch` so the file is valid at the end, or re-run with --allow-syntax-errors to write this intermediate state"
        ),
        "{text}"
    );
    assert_file(&fixture.repo.join("item.rs"), before);

    let output = fixture.run(&["replace-lines", "item.rs", "1:1", "fn after( {}", "--json"]);
    assert_eq!(output.status.code(), Some(13), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["published"], false);
    assert_eq!(value["error"]["code"], "invalid_result");
    let next = value["error"]["next"].as_str().unwrap();
    assert!(next.contains("--allow-syntax-errors"), "{next}");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--allow-syntax-errors"),
        "{}",
        combined(&output)
    );
    assert_file(&fixture.repo.join("item.rs"), before);
}

#[test]
fn allow_syntax_errors_writes_and_reports_new_diagnostics() {
    let fixture = Fixture::new("syntax-allow");
    for (file, json) in [("item.rs", false), ("item-json.rs", true)] {
        let path = fixture.repo.join(file);
        std::fs::write(&path, "fn before() {}\n").unwrap();
        let mut args = vec!["replace-lines", file, "1:1", "fn after( {}"];
        args.push("--allow-syntax-errors");
        if json {
            args.push("--json");
        }
        let output = fixture.run(&args);
        assert_eq!(output.status.code(), Some(0), "{}", combined(&output));
        if json {
            let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(value["published"], true);
            let message = value["message"].as_str().unwrap();
            assert!(message.contains("new syntax errors"), "{message}");
            assert!(
                message.contains("allowed by --allow-syntax-errors"),
                "{message}"
            );
        } else {
            let text = combined(&output);
            assert!(text.contains("new syntax errors"), "{text}");
            assert!(text.contains("allowed by --allow-syntax-errors"), "{text}");
        }
        assert_file(&path, "fn after( {}\n");
    }
}

#[test]
fn repeated_old_names_both_lines_and_the_expect_flag() {
    let fixture = Fixture::new("old-cardinality");
    let before = "alpha\nbeta\nalpha\ngamma\n";
    std::fs::write(fixture.repo.join("repeated.txt"), before).unwrap();
    let output = fixture.run(&["replace-text", "repeated.txt", "alpha", "ALPHA"]);
    assert_eq!(output.status.code(), Some(13), "{}", combined(&output));
    let text = combined(&output);
    assert!(
        text.contains("OLD occurs 2 times, expected 1 — nothing written"),
        "{text}"
    );
    assert!(text.contains("repeated.txt:1:1: alpha"), "{text}");
    assert!(text.contains("repeated.txt:3:1: alpha"), "{text}");
    assert!(text.contains("pass --expect 2"), "{text}");
    assert_file(&fixture.repo.join("repeated.txt"), before);

    let output = fixture.run(&["replace-text", "repeated.txt", "alpha", "ALPHA", "--json"]);
    assert_eq!(output.status.code(), Some(13), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["error"]["count"], 2);
    assert_eq!(value["error"]["expected"], 1);
    assert_eq!(value["error"]["match_lines"][0]["line"], 1);
    assert_eq!(value["error"]["match_lines"][1]["line"], 3);
    assert_file(&fixture.repo.join("repeated.txt"), before);
}

#[test]
fn regex_with_no_matches_names_search_pattern() {
    let fixture = Fixture::new("regex-zero");
    std::fs::write(fixture.repo.join("a.txt"), "alpha\n").unwrap();
    let output = fixture.run(&["replace-text", "a.txt", "absent_token", "unused", "--regex"]);
    assert_eq!(output.status.code(), Some(13), "{}", combined(&output));
    let text = combined(&output);
    assert!(
        text.contains("greppy search-pattern absent_token a.txt"),
        "{text}"
    );
    assert_file(&fixture.repo.join("a.txt"), "alpha\n");

    let output = fixture.run(&[
        "replace-text",
        "a.txt",
        "absent_token",
        "unused",
        "--regex",
        "--json",
    ]);
    assert_eq!(output.status.code(), Some(13), "{}", combined(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let next = value["error"]["next"].as_str().unwrap();
    assert!(
        next.contains("greppy search-pattern absent_token a.txt"),
        "{next}"
    );
}

fn receipt_source_lines(stdout: &str) -> usize {
    stdout
        .lines()
        .filter(|line| {
            let Some(rest) = line.strip_prefix('│') else {
                return false;
            };
            rest.starts_with('>') || rest.starts_with(' ')
        })
        .count()
}

#[test]
fn heredoc_body_reindents_rust_and_javascript_from_stdin() {
    let rust_before =
        "fn score(value: i32) -> i32 {\n    let doubled = value * 2;\n    doubled + 1\n}\n";
    let rust_after =
        "fn score(value: i32) -> i32 {\n    let doubled = value * 3;\n    doubled + 4\n}\n";
    let heredoc = b"        let doubled = value * 3;\n        doubled + 4\n";
    let preindented = b"    let doubled = value * 3;\n    doubled + 4\n";
    for (label, bytes) in [
        ("heredoc", heredoc.as_slice()),
        ("preindented", preindented.as_slice()),
    ] {
        let fixture = Fixture::new(&format!("heredoc-rust-{label}"));
        let file = fixture.repo.join("score.rs");
        std::fs::write(&file, rust_before).unwrap();
        let applied =
            fixture.run_with_stdin(&["replace", "score.rs::Function::score", "--body"], bytes);
        assert!(applied.status.success(), "{label}: {}", combined(&applied));
        assert_file(&file, rust_after);
    }

    let js_before =
        "function score(value) {\n    const doubled = value * 2;\n    return doubled + 1;\n}\n";
    let js_after =
        "function score(value) {\n    const doubled = value * 3;\n    return doubled + 4;\n}\n";
    let js_heredoc = b"        const doubled = value * 3;\n        return doubled + 4;\n";
    let js_pre = b"    const doubled = value * 3;\n    return doubled + 4;\n";
    for (label, bytes) in [
        ("heredoc", js_heredoc.as_slice()),
        ("preindented", js_pre.as_slice()),
    ] {
        let fixture = Fixture::new(&format!("heredoc-js-{label}"));
        let file = fixture.repo.join("score.js");
        std::fs::write(&file, js_before).unwrap();
        let applied =
            fixture.run_with_stdin(&["replace", "score.js::Function::score", "--body"], bytes);
        assert!(applied.status.success(), "{label}: {}", combined(&applied));
        assert_file(&file, js_after);
    }
}

#[test]
fn body_replace_receipt_shows_only_changed_lines() {
    let mut inner = Vec::new();
    for i in 0..38 {
        inner.push(format!("    let kept_{i} = {i};"));
    }
    let original = format!("fn probe() {{\n{}\n}}\n", inner.join("\n"));
    inner[18] = "    let after_change = 1;".to_string();
    inner[19] = "    let after_next = 2;".to_string();
    let replacement = format!("{{\n{}\n}}", inner.join("\n"));
    let expected = format!("fn probe() {replacement}\n");

    let fixture = Fixture::new("body-receipt-lines");
    let file = fixture.repo.join("probe.rs");
    std::fs::write(&file, &original).unwrap();
    let applied = fixture.run(&[
        "replace",
        "probe.rs::Function::probe",
        &replacement,
        "--body",
    ]);
    let stdout = String::from_utf8_lossy(&applied.stdout);
    assert!(applied.status.success(), "{}", combined(&applied));
    let first = stdout.lines().next().unwrap_or("");
    assert!(
        first.starts_with("replaced body of probe.rs::Function::probe (lines "),
        "{stdout}"
    );
    assert!(first.contains("changed lines "), "{stdout}");
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("applied probe.rs:1-40")),
        "address line must keep the full span\n{stdout}"
    );
    assert!(
        receipt_source_lines(&stdout) <= 6,
        "changed-line echo too long ({} lines)\n{stdout}",
        receipt_source_lines(&stdout)
    );
    assert!(stdout.contains("after_change"), "{stdout}");
    assert!(!stdout.contains("kept_0"), "{stdout}");
    assert!(!stdout.contains("kept_37"), "{stdout}");
    assert_file(&file, &expected);

    std::fs::write(&file, &original).unwrap();
    let json_run = fixture.run(&[
        "replace",
        "probe.rs::Function::probe",
        &replacement,
        "--body",
        "--json",
    ]);
    assert!(json_run.status.success(), "{}", combined(&json_run));
    let value: serde_json::Value = serde_json::from_slice(&json_run.stdout).unwrap();
    assert_eq!(value["span"], "1:40", "{value}");
    assert_eq!(value["changed_span"], "20:21", "{value}");
    assert_file(&file, &expected);
}
