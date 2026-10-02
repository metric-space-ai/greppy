//! Public CLI repair acceptance, run under the real host admission gate.
//! The fixture gate owns, cancels and reaps the actual index child.
#![cfg(all(unix, debug_assertions))]
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

struct OwnedChild(Child);
fn bounded_exit(child: &mut OwnedChild) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "owned CLI process did not terminate within 30s"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        // Child::kill uses this still-owned unreaped process, never a recorded PID.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn bounded_output(mut command: Command) -> Output {
    // Files avoid pipe backpressure while polling and keep all process waits bounded.
    let mut stdout = tempfile::tempfile().unwrap();
    let mut stderr = tempfile::tempfile().unwrap();
    let mut child = OwnedChild(
        command
            .stdout(stdout.try_clone().unwrap())
            .stderr(stderr.try_clone().unwrap())
            .spawn()
            .unwrap(),
    );
    let status = bounded_exit(&mut child);
    stdout.seek(SeekFrom::Start(0)).unwrap();
    stderr.seek(SeekFrom::Start(0)).unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    stdout.read_to_end(&mut out).unwrap();
    stderr.read_to_end(&mut err).unwrap();
    Output {
        status,
        stdout: out,
        stderr: err,
    }
}
fn git(root: &Path, args: &[&str]) -> String {
    let mut command = Command::new("git");
    command.args(args).current_dir(root);
    let out = bounded_output(command);
    assert!(
        out.status.success(),
        "fixture git: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().into()
}
struct Fixture {
    _scratch: tempfile::TempDir,
    root: PathBuf,
    store: PathBuf,
    base: PathBuf,
    delta: PathBuf,
    commit: String,
    gate: PathBuf,
    record: PathBuf,
    cancel: PathBuf,
    terminal: PathBuf,
}
impl Fixture {
    fn command(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_greppy"));
        cmd.args(args)
            .current_dir(&self.root)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_PROJECT_IDENTITY", "p")
            .env("GREPPY_AGENT_STORE_MODE", "overlay")
            .env("GREPPY_AGENT_BASE_STORE", &self.base)
            .env("GREPPY_AGENT_BASE_COMMIT", &self.commit)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_AUTO_REINDEX", "1")
            .env("GREPPY_HEAVY_GATE", &self.gate)
            .env("REPAIR_GATE_RECORD", &self.record)
            .env("REPAIR_GATE_CANCEL", &self.cancel)
            .env("REPAIR_GATE_TERMINAL", &self.terminal)
            .env_remove("GREPPY_DELEGATED_BACKGROUND_JOB")
            .env_remove("GREPPY_DISCOVER_INCLUDE")
            .env_remove("GREPPY_DISCOVER_EXCLUDE");
        cmd
    }
    fn query(&self) -> String {
        let out = bounded_output(self.command(&["who-calls", "target", "--json"]));
        assert!(
            out.status.success(),
            "public query failed: {}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        let _: serde_json::Value = serde_json::from_str(&text).expect("public query JSON");
        assert!(
            text.contains("caller"),
            "recovered Base reference absent: {text}"
        );
        text
    }
    fn new() -> Self {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().join("repo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn target() {}\npub fn caller() { let _ = target; }\n",
        )
        .unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "repair@test.invalid"]);
        git(&root, &["config", "user.name", "Repair acceptance"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-qm", "base"]);
        let commit = git(&root, &["rev-parse", "HEAD"]);
        let source = scratch.path().join("source.db");
        {
            let mut base = greppy_store::Store::open(&source).unwrap();
            greppy_indexer::index(&mut base, &root, "p").unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [greppy_indexer::RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
            base.conn()
                .execute("DELETE FROM edges WHERE edge_type='USAGE'", [])
                .unwrap();
        }
        let identity = greppy_store::BaseStoreIdentity {
            format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
            canonical_repository_identity: format!("fixture:{}", root.display()),
            git_object_format: "sha1".into(),
            base_tree_oid: git(&root, &["rev-parse", "HEAD^{tree}"]),
            store_schema_version: greppy_store::migrate::CURRENT_VERSION,
            indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
            parser_and_extractor_versions: format!(
                "greppy-parser/extractor-{}",
                env!("CARGO_PKG_VERSION")
            ),
            summary_model_and_prompt_version: "fixture-summary-v1".into(),
            embedding_model: "fixture-embedding-v1".into(),
            embedding_prompt_version: "fixture-prompt-v1".into(),
            embedding_dimensions: 768,
            embedding_encoding: "f32+i8-v1".into(),
        };
        let layout =
            greppy_store::BaseStoreLayout::new(&scratch.path().join("base"), &identity).unwrap();
        let builder = layout.acquire_builder(false).unwrap().unwrap();
        let summaries = scratch.path().join("summaries");
        drop(greppy_store::SummaryCache::open(&summaries).unwrap());
        layout
            .publish_graph_with_summary(
                identity,
                &source,
                &summaries.join(greppy_store::SUMMARY_CACHE_DB_FILE),
            )
            .unwrap();
        drop(builder);
        let gate = scratch.path().join("gate.py");
        let record = scratch.path().join("gate.jsonl");
        std::fs::write(
            &gate,
            r#"import json, os, subprocess, sys, time
args=sys.argv[sys.argv.index('--')+1:]
child=subprocess.Popen(args)
with open(os.environ['REPAIR_GATE_RECORD'], 'a') as f:
    f.write(json.dumps({'gate_pid':os.getpid(),'child_pid':child.pid,'argv':args})+'\n')
deadline=time.monotonic()+45
try:
    while child.poll() is None:
        if os.path.exists(os.environ['REPAIR_GATE_CANCEL']) or time.monotonic() >= deadline:
            # The unreaped Popen child cannot have its PID reused here.
            child.kill()
            break
        time.sleep(0.02)
finally:
    status=child.wait(timeout=5)
    with open(os.environ['REPAIR_GATE_TERMINAL'], 'w') as f:
        json.dump({'status':status,'reaped':True},f)
sys.exit(status)
"#,
        )
        .unwrap();
        let mut fixture = Self {
            root,
            store: scratch.path().join("store"),
            base: layout.graph,
            delta: PathBuf::new(),
            commit,
            gate,
            record,
            cancel: scratch.path().join("gate-cancel"),
            terminal: scratch.path().join("gate-terminal"),
            _scratch: scratch,
        };
        let out = bounded_output(fixture.command(&["index", "."]));
        assert!(
            out.status.success(),
            "fixture initial Delta: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        fn locate(path: &Path) -> Option<PathBuf> {
            for entry in std::fs::read_dir(path).ok()?.flatten() {
                let p = entry.path();
                if p.file_name().is_some_and(|s| s == "graph.db") {
                    return Some(p);
                }
                if p.is_dir() {
                    if let Some(found) = locate(&p) {
                        return Some(found);
                    }
                }
            }
            None
        }
        fixture.delta = locate(&fixture.store).expect("fixture published Delta");
        let delta = greppy_store::Store::open(&fixture.delta).unwrap();
        delta
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [greppy_indexer::RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        let contribution_count: i64 = delta
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM overlay_edges WHERE project='p' AND edge_type='USAGE'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            contribution_count > 0,
            "fixture initial repair did not create a resolved Base reference"
        );
        delta
            .conn()
            .execute(
                "DELETE FROM overlay_edges WHERE project='p' AND edge_type='USAGE'",
                [],
            )
            .unwrap();
        delta
            .conn()
            .execute("DELETE FROM edges WHERE edge_type='USAGE'", [])
            .unwrap();
        drop(delta);
        let visibility =
            greppy_store::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let visible =
            greppy_store::Store::open_with(&fixture.delta, greppy_store::OpenOptions::read_only())
                .unwrap()
                .attach_overlay(&fixture.base, &visibility)
                .unwrap();
        let target = visible
            .get_node_by_qname("p", "src/lib.rs::Function::target")
            .unwrap()
            .unwrap();
        assert!(
            visible
                .incoming_edges(target.id, Some("USAGE"), 10)
                .unwrap()
                .is_empty(),
            "fixture still exposes a caller before repair"
        );
        drop(visible);
        fixture
    }
    fn job(&self) -> serde_json::Value {
        serde_json::from_slice(
            &std::fs::read(self.delta.parent().unwrap().join("index.job")).unwrap(),
        )
        .unwrap()
    }
    fn assert_certified(&self) {
        let delta =
            greppy_store::Store::open_with(&self.delta, greppy_store::OpenOptions::read_only())
                .unwrap();
        assert!(greppy_indexer::rust_caller_edges_repaired(&delta).unwrap());
    }
}
impl Fixture {
    fn cancel_and_reap(&self) -> bool {
        if !self.record.exists() {
            return true;
        }
        if std::fs::write(&self.cancel, "cancel owned Popen child").is_err() {
            return false;
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while !self.terminal.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        self.terminal.exists()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // The gate alone owns its Popen; cancellation never signals a stored PID.
        if !self.cancel_and_reap() {
            eprintln!("fixture gate cleanup receipt missing; gate has its own 45s deadline");
        }
    }
}
fn wait_ready(ready: &Path, child: &mut OwnedChild) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready.exists() && Instant::now() < deadline {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "query exited before repair synchronization"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(ready.exists(), "fixture synchronization timed out");
}

#[test]
fn public_query_admits_repairs_publishes_and_reopens_uncertified_base() {
    let f = Fixture::new();
    let base_before = std::fs::read(&f.base).unwrap();
    let ready = f._scratch.path().join("repair-ready");
    let release = f._scratch.path().join("release");
    let output = std::fs::File::create(f._scratch.path().join("query.out")).unwrap();
    let mut child = OwnedChild(
        f.command(&["who-calls", "target", "--json"])
            .env("GREPPY_TEST_INDEX_FAILPOINT", "before-rust-repair")
            .env("GREPPY_TEST_INDEX_FAILPOINT_READY", &ready)
            .env("GREPPY_TEST_INDEX_FAILPOINT_RELEASE", &release)
            .env("GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS", "30000")
            .stdout(output)
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_ready(&ready, &mut child);
    let record = std::fs::read_to_string(&f.record).unwrap();
    let admission: serde_json::Value =
        serde_json::from_str(record.lines().next().unwrap()).unwrap();

    assert!(
        admission["argv"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "index"),
        "gate must run actual index child: {admission}"
    );
    let job = f.job();
    assert_eq!(job["cause"], "rust-graph-repair");
    assert_eq!(job["state"], "repairing_graph");
    assert_eq!(
        job["pid"], admission["gate_pid"],
        "journal must retain launcher admission ownership"
    );
    assert_ne!(
        job["pid"],
        child.0.id(),
        "query must not silently repair inline"
    );
    let candidate = PathBuf::from(std::fs::read_to_string(&ready).unwrap());
    assert!(candidate.exists());
    assert_eq!(std::fs::read(&f.base).unwrap(), base_before);
    std::fs::write(&release, "release").unwrap();
    assert!(
        bounded_exit(&mut child).success(),
        "first public query failed after repair"
    );

    let first = std::fs::read_to_string(f._scratch.path().join("query.out")).unwrap();
    assert!(
        first.contains("caller"),
        "first answer lacks recovered Base reference: {first}"
    );
    assert!(
        !candidate.exists(),
        "completed snapshot must publish atomically"
    );
    f.assert_certified();
    let journal = std::fs::read(f.delta.parent().unwrap().join("index.job")).ok();
    f.query();
    assert_eq!(
        std::fs::read_to_string(&f.record).unwrap(),
        record,
        "second query must not launch another repair"
    );
    assert_eq!(
        std::fs::read(f.delta.parent().unwrap().join("index.job")).ok(),
        journal,
        "second query must not mutate repair ownership"
    );
    assert_eq!(
        std::fs::read(&f.base).unwrap(),
        base_before,
        "immutable Base bytes changed"
    );
}

/// Overlay staging is deliberately outside the standalone recovery contract.
/// Cancellation must preserve active bytes; a later query safely rebuilds.
#[test]
fn public_query_cancelled_overlay_preserves_active_and_retries_under_admission() {
    let f = Fixture::new();
    let active_before = std::fs::read(&f.delta).unwrap();
    let base_before = std::fs::read(&f.base).unwrap();
    let ready = f._scratch.path().join("candidate-ready");
    let mut query = OwnedChild(
        f.command(&["who-calls", "target", "--json"])
            .env("GREPPY_TEST_INDEX_FAILPOINT", "after-temp-before-publish")
            .env("GREPPY_TEST_INDEX_FAILPOINT_READY", &ready)
            .env("GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS", "30000")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_ready(&ready, &mut query);
    let candidate = PathBuf::from(std::fs::read_to_string(&ready).unwrap());
    assert!(
        candidate
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("graph.db.delta-building."),
        "unexpected overlay staging name: {}",
        candidate.display()
    );
    assert_eq!(std::fs::read(&f.delta).unwrap(), active_before);
    let live = bounded_output(f.command(&["index", "recover", ".", "--json"]));
    assert!(!live.status.success(), "live writer recovery must fail");
    assert!(
        String::from_utf8_lossy(&live.stderr).contains("index writer is still active"),
        "{}",
        String::from_utf8_lossy(&live.stderr)
    );
    assert!(
        f.cancel_and_reap(),
        "gate must cancel and reap its own actual child"
    );
    assert!(
        !bounded_exit(&mut query).success(),
        "cancelled query claimed success"
    );
    let terminal: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&f.terminal).unwrap()).unwrap();
    assert_eq!(terminal["reaped"], true);
    assert!(candidate.exists());
    assert_eq!(std::fs::read(&f.delta).unwrap(), active_before);
    let recovery = bounded_output(f.command(&["index", "recover", ".", "--json"]));
    assert!(
        recovery.status.success(),
        "{}",
        String::from_utf8_lossy(&recovery.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&recovery.stdout).unwrap();
    assert_eq!(
        report["status"], "no-candidate",
        "standalone recovery must not silently adopt an overlay: {report}"
    );
    assert!(
        candidate.exists(),
        "recovery must not delete an unrecognized overlay candidate"
    );
    assert_eq!(std::fs::read(&f.delta).unwrap(), active_before);
    // Start a fresh admitted index; do not rename staging or bypass recovery guards.
    std::fs::remove_file(&f.cancel).unwrap();
    std::fs::remove_file(&f.terminal).unwrap();
    f.query();
    f.assert_certified();
    assert_eq!(
        std::fs::read_to_string(&f.record).unwrap().lines().count(),
        2,
        "retry must acquire admission anew"
    );
    assert_eq!(std::fs::read(&f.base).unwrap(), base_before);
}
