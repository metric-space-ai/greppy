//! A query never waits without limit for index work.
//!
//! After `GREPPY_QUERY_WAIT_SECS` it answers from the published index and
//! says that its refresh is still running, or, with no index at all, says so
//! in a form a program can read. The work it started goes on and is
//! published, so the next query finds it. A read builds or refreshes only a
//! small index by itself, and `greppy index` builds the graph and nothing else.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

struct Fixture {
    root: PathBuf,
    repo: PathBuf,
    store: PathBuf,
    /// Created by the test to let a held index job publish.
    release: PathBuf,
}

impl Fixture {
    /// `caller` calls `helper`; nothing is indexed yet.
    fn new(tag: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("greppy-bounded-wait-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo = root.join("repo");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::write(
            repo.join("src/lib.rs"),
            "mod util;\npub fn caller() { util::helper(); }\n",
        )
        .unwrap();
        std::fs::write(repo.join("src/util.rs"), "pub fn helper() {}\n").unwrap();
        let lane = root.join("lane-tmp");
        std::fs::create_dir_all(&lane).unwrap();
        Self {
            store: root.join("store"),
            release: root.join("release-the-index-job"),
            repo,
            root,
        }
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(bin());
        command
            .args(args)
            .current_dir(&self.repo)
            .stdin(std::process::Stdio::null())
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("TMPDIR", self.root.join("lane-tmp"));
        command
    }

    fn run(&self, args: &[&str]) -> (i32, String, String) {
        output(self.command(args))
    }

    /// The query waits one second; the index job it starts stops before
    /// publishing until the test releases it.
    fn run_while_the_index_job_is_held(&self, args: &[&str]) -> (i32, String, String) {
        let mut command = self.command(args);
        command
            .env("GREPPY_QUERY_WAIT_SECS", "1")
            .env("GREPPY_TEST_INDEX_FAILPOINT", "after-temp-before-publish")
            .env("GREPPY_TEST_INDEX_FAILPOINT_RELEASE", &self.release)
            .env("GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS", "120000");
        output(command)
    }

    /// Let the held job publish and return what `who-calls helper` answers
    /// once `expected` is in it.
    fn answer_after_release(&self, expected: &str) -> String {
        std::fs::write(&self.release, "").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
        loop {
            let (code, out, err) = self.run(&["who-calls", "helper"]);
            if code == 0 && out.contains(expected) {
                return out;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the index job the query left running never published: code={code}\nstdout={out}\nstderr={err}"
            );
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::write(&self.release, "");
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn output(mut command: Command) -> (i32, String, String) {
    let out = command.output().expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn a_query_answers_from_the_published_index_while_its_refresh_still_runs() {
    let fixture = Fixture::new("stale");
    let (code, out, err) = fixture.run(&["index", "."]);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    std::fs::write(
        fixture.repo.join("src/util.rs"),
        "pub fn helper() {}\npub fn added() { helper(); }\n",
    )
    .unwrap();

    let started = std::time::Instant::now();
    let (code, out, err) = fixture.run_while_the_index_job_is_held(&["who-calls", "helper"]);
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(60),
        "the query waited for the held refresh: {:?}",
        started.elapsed()
    );
    assert!(
        out.contains("src/lib.rs:2  caller") && !out.contains("added"),
        "the answer comes from the published index: {out}"
    );
    assert!(
        err.contains("answering from the existing index")
            && err.contains("its refresh is still running"),
        "the answer must say that it predates the change and why: {err}"
    );

    // The refresh the query left behind is finished, not cancelled.
    fixture.answer_after_release("src/util.rs:2  added");
}

#[test]
fn a_query_without_an_index_says_so_and_the_index_is_still_built() {
    let fixture = Fixture::new("none");
    let started = std::time::Instant::now();
    let (code, out, err) =
        fixture.run_while_the_index_job_is_held(&["who-calls", "helper", "--json"]);
    assert_eq!(code, 75, "stderr={err}\nstdout={out}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(60),
        "the query waited for the held first index: {:?}",
        started.elapsed()
    );
    let said: serde_json::Value =
        serde_json::from_str(out.trim()).unwrap_or_else(|error| panic!("{error}: {out}"));
    assert_eq!(said["status"], "index_building", "{said}");
    assert_eq!(said["index"]["published"], false, "{said}");
    assert_eq!(said["next"]["action"], "retry_original_command", "{said}");

    fixture.answer_after_release("src/lib.rs:2  caller");
}

#[test]
fn a_query_without_an_index_under_a_refusing_gate_starts_no_index() {
    let fixture = Fixture::new("gate");
    let gate = refusing_gate(&fixture);
    let mut command = fixture.command(&["graph-locate", "src/lib.rs:2", "--json"]);
    command.env("GREPPY_HEAVY_GATE", &gate);
    let started = std::time::Instant::now();
    let (code, out, err) = output(command);
    assert_eq!(code, 75, "stderr={err}\nstdout={out}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(15),
        "a refused first index is answered at once: {:?}",
        started.elapsed()
    );
    let said: serde_json::Value =
        serde_json::from_str(out.trim()).unwrap_or_else(|error| panic!("{error}: {out}"));
    assert_eq!(said["status"], "preparation_deferred", "{said}");
    assert_eq!(said["index"]["published"], false, "{said}");
    assert!(
        !fixture.store.join("workspaces").exists() || graph_databases(&fixture.store).is_empty(),
        "no index may be written while the gate refuses"
    );
}

fn graph_databases(store: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![store.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("graph.db"))
            {
                found.push(path);
            }
        }
    }
    found
}

#[test]
fn a_symbol_edit_waits_for_the_refresh_instead_of_planning_on_the_old_graph() {
    let fixture = Fixture::new("edit");
    let (code, out, err) = fixture.run(&["index", "."]);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    std::fs::write(
        fixture.repo.join("src/util.rs"),
        "pub fn helper() {}\npub fn added() { helper(); }\n",
    )
    .unwrap();

    // The refresh is held for three seconds; a query would leave after one.
    let mut command = fixture.command(&["rename", "helper", "renamed_helper"]);
    command
        .env("GREPPY_QUERY_WAIT_SECS", "1")
        .env("GREPPY_TEST_INDEX_FAILPOINT", "after-temp-before-publish")
        .env("GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS", "3000");
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "rename: stderr={err}\nstdout={out}");
    let util = std::fs::read_to_string(fixture.repo.join("src/util.rs")).unwrap();
    assert!(
        util.contains("pub fn added() { renamed_helper(); }"),
        "the call site that only the refreshed graph knows is renamed too: {util}"
    );
}

#[test]
fn the_index_command_builds_the_graph_and_starts_no_model_work() {
    let fixture = Fixture::new("graph-only");
    let mut command = fixture.command(&["index", "."]);
    // Inference is available to this command; the guard turns any attempt
    // to configure it into an error.
    command
        .env_remove("GREPPY_TEST_SKIP_INFERENCE")
        .env("GREPPY_TEST_FORBID_INDEX_INFERENCE", "1");
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    assert!(!out.contains("embedded"), "{out}");

    let (code, out, err) = fixture.run(&["who-calls", "helper"]);
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(out.contains("src/lib.rs:2  caller"), "{out}");

    // Asked for, the embeddings are built by the same command.
    let mut command = fixture.command(&["index", ".", "--with-embeddings"]);
    command
        .env_remove("GREPPY_TEST_SKIP_INFERENCE")
        .env("GREPPY_TEST_FORBID_INDEX_INFERENCE", "1");
    let (code, out, err) = output(command);
    assert_ne!(
        code, 0,
        "the guard must have seen inference being configured: {out}{err}"
    );
}

/// The fixture has two files; with this limit it counts as a large
/// repository, where index work is minutes for the whole host.
const LARGE: (&str, &str) = ("GREPPY_AUTO_INDEX_MAX_FILES", "1");

#[test]
fn a_read_does_not_build_the_index_of_a_large_repository() {
    let fixture = Fixture::new("large-none");
    let mut command = fixture.command(&["who-calls", "helper", "--json"]);
    command.env(LARGE.0, LARGE.1);
    let started = std::time::Instant::now();
    let (code, out, err) = output(command);
    assert_eq!(code, 75, "stderr={err}\nstdout={out}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "the answer does not wait for anything: {:?}",
        started.elapsed()
    );
    let said: serde_json::Value =
        serde_json::from_str(out.trim()).unwrap_or_else(|error| panic!("{error}: {out}"));
    assert_eq!(said["status"], "no_index", "{said}");
    assert_eq!(said["index"]["published"], false, "{said}");
    assert!(
        said["freshness"]["preparation_error"]
            .as_str()
            .is_some_and(|detail| detail.contains("greppy index .")),
        "the answer names the command that helps: {said}"
    );
    assert!(
        graph_databases(&fixture.store).is_empty(),
        "no index work may have started"
    );

    // The text form says the same, and the index command builds it.
    let mut command = fixture.command(&["who-calls", "helper"]);
    command.env(LARGE.0, LARGE.1);
    let (code, _, err) = output(command);
    assert_eq!(code, 75, "{err}");
    assert!(
        err.contains("no index for") && err.contains("greppy index ."),
        "{err}"
    );
    let mut command = fixture.command(&["index", "."]);
    command.env(LARGE.0, LARGE.1);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    let mut command = fixture.command(&["who-calls", "helper"]);
    command.env(LARGE.0, LARGE.1);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(out.contains("src/lib.rs:2  caller"), "{out}");
}

#[test]
fn a_read_answers_from_the_stale_index_of_a_large_repository_at_once() {
    let fixture = Fixture::new("large-stale");
    let (code, out, err) = fixture.run(&["index", "."]);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    std::fs::write(
        fixture.repo.join("src/util.rs"),
        "pub fn helper() {}\npub fn added() { helper(); }\n",
    )
    .unwrap();

    let mut command = fixture.command(&["who-calls", "helper"]);
    command.env(LARGE.0, LARGE.1);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(
        out.contains("src/lib.rs:2  caller") && !out.contains("added"),
        "the answer comes from the published index, it is not empty: {out}"
    );
    assert!(
        err.contains("answering from the existing index")
            && err.contains("refresh it with `greppy index .`"),
        "the answer must say that it predates the change and what refreshes it: {err}"
    );
    let mut command = fixture.command(&["who-calls", "helper", "--json"]);
    command.env(LARGE.0, LARGE.1);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "who-calls --json: stderr={err}\nstdout={out}");
    let said: serde_json::Value =
        serde_json::from_str(out.trim()).unwrap_or_else(|error| panic!("{error}: {out}"));
    assert_eq!(said["stale_served"], true, "{said}");
    assert!(
        said["hits"].as_array().is_some_and(|hits| !hits.is_empty()),
        "{said}"
    );
    let job = graph_databases(&fixture.store)
        .into_iter()
        .any(|path| path.to_string_lossy().contains(".next."));
    assert!(!job, "the read must not have started a refresh");

    // The index command refreshes; after it the read is current again.
    let mut command = fixture.command(&["index", "."]);
    command.env(LARGE.0, LARGE.1);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    let mut command = fixture.command(&["who-calls", "helper"]);
    command.env(LARGE.0, LARGE.1);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(out.contains("src/util.rs:2  added"), "{out}");
    assert!(!err.contains("answering from the existing index"), "{err}");
}

#[test]
fn the_size_limit_can_be_lifted_for_one_command() {
    let fixture = Fixture::new("large-lifted");
    let mut command = fixture.command(&["who-calls", "helper"]);
    command.env(LARGE.0, LARGE.1).env("GREPPY_AUTO_INDEX", "1");
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(out.contains("src/lib.rs:2  caller"), "{out}");
}

fn refusing_gate(fixture: &Fixture) -> PathBuf {
    let gate = fixture.root.join("refusing-gate.py");
    std::fs::write(
        &gate,
        "import sys\nsys.stderr.write('another owner holds the heavy-job lease')\nsys.exit(75)\n",
    )
    .unwrap();
    gate
}

#[test]
fn the_first_index_of_a_large_repository_waits_for_the_host_gate() {
    let fixture = Fixture::new("large-gate");
    let gate = refusing_gate(&fixture);
    let mut command = fixture.command(&["index", "."]);
    command
        .env(LARGE.0, LARGE.1)
        .env("GREPPY_HEAVY_GATE", &gate);
    let (code, out, err) = output(command);
    assert_eq!(code, 75, "stderr={err}\nstdout={out}");
    assert!(err.contains("waits for the host's build gate"), "{err}");
    assert!(
        graph_databases(&fixture.store).is_empty(),
        "no index work may have started"
    );

    // A gate that admits runs the index; refreshing it later needs no gate.
    let admitting = fixture.root.join("admitting-gate.py");
    std::fs::write(
        &admitting,
        "import os, sys\nargs = sys.argv[sys.argv.index('--') + 1:]\nos.execv(args[0], args)\n",
    )
    .unwrap();
    let mut command = fixture.command(&["index", "."]);
    command
        .env(LARGE.0, LARGE.1)
        .env("GREPPY_HEAVY_GATE", &admitting);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "admitted index: stderr={err}\nstdout={out}");
    std::fs::write(
        fixture.repo.join("src/util.rs"),
        "pub fn helper() {}\npub fn added() { helper(); }\n",
    )
    .unwrap();
    let mut command = fixture.command(&["index", "."]);
    command
        .env(LARGE.0, LARGE.1)
        .env("GREPPY_HEAVY_GATE", &gate);
    let (code, out, err) = output(command);
    assert_eq!(
        code, 0,
        "a refresh does not wait for the gate: stderr={err}\nstdout={out}"
    );
    let (code, out, err) = fixture.run(&["who-calls", "helper"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("src/util.rs:2  added"), "{out}");
}

#[test]
fn a_small_repository_is_indexed_without_the_gate() {
    let fixture = Fixture::new("small-gate");
    let gate = refusing_gate(&fixture);
    let mut command = fixture.command(&["index", "."]);
    command.env("GREPPY_HEAVY_GATE", &gate);
    let (code, out, err) = output(command);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
}

#[test]
fn every_read_answers_a_large_stale_repository_from_its_index_and_marks_it() {
    let fixture = Fixture::new("large-stale-all");
    let (code, out, err) = fixture.run(&["index", "."]);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    std::fs::write(
        fixture.repo.join("src/util.rs"),
        "pub fn helper() {}\npub fn added() { helper(); }\n",
    )
    .unwrap();
    for args in [
        &["callees", "src/lib.rs::caller", "--json"][..],
        &["impact", "src/util.rs::helper", "--json"][..],
        &["graph-locate", "src/lib.rs:2", "--json"][..],
        &["search-symbol", "helper", "--json"][..],
        &["isolate", "src/util.rs::helper", "--json"][..],
        &["where-am-i", "--json"][..],
    ] {
        let mut command = fixture.command(args);
        command.env(LARGE.0, LARGE.1);
        let (code, out, err) = output(command);
        assert_eq!(code, 0, "{args:?}: stderr={err}\nstdout={out}");
        assert!(
            err.contains("answering from the existing index"),
            "{args:?} must say that it predates the change: {err}"
        );
        if args[0] != "where-am-i" {
            let said: serde_json::Value = serde_json::from_str(out.trim())
                .unwrap_or_else(|error| panic!("{args:?}: {error}: {out}"));
            assert_eq!(said["stale_served"], true, "{args:?}: {said}");
        }
    }
    let job = graph_databases(&fixture.store)
        .into_iter()
        .any(|path| path.to_string_lossy().contains(".next."));
    assert!(!job, "no read may have started a refresh");
}

#[test]
fn where_am_i_and_isolate_answer_no_index_in_json() {
    let fixture = Fixture::new("large-none-json");
    for args in [
        &["where-am-i", "--json"][..],
        &["isolate", "src/lib.rs:2", "--json"][..],
    ] {
        let mut command = fixture.command(args);
        command.env(LARGE.0, LARGE.1);
        let (code, out, err) = output(command);
        assert_eq!(code, 75, "{args:?}: stderr={err}\nstdout={out}");
        let said: serde_json::Value = serde_json::from_str(out.trim())
            .unwrap_or_else(|error| panic!("{args:?}: {error}: {out}"));
        assert_eq!(said["status"], "no_index", "{args:?}: {said}");
    }
}

#[test]
fn a_store_with_only_command_output_is_no_index_when_automatic_indexing_is_off() {
    let fixture = Fixture::new("packs-only");
    // `read-file` keeps a continuation pack and so creates the store.
    let (code, _, err) = fixture.run(&["read-file", "src/lib.rs", "--json"]);
    assert_eq!(code, 0, "{err}");
    let mut command = fixture.command(&["who-calls", "helper", "--json"]);
    command.env("GREPPY_AUTO_REINDEX", "0");
    let (code, out, err) = output(command);
    assert_eq!(code, 75, "stderr={err}\nstdout={out}");
    let said: serde_json::Value =
        serde_json::from_str(out.trim()).unwrap_or_else(|error| panic!("{error}: {out}"));
    assert_eq!(said["status"], "no_index", "{said}");
}
