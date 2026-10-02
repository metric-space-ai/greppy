//! Public CLI repair acceptance. Run under the real host admission gate; the
//! fixture gate records and waits for the actual index child, never replaces it.
#![cfg(all(unix, debug_assertions))]
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct OwnedChild(Child);
struct OwnedIndexer(Option<i32>);
impl OwnedIndexer {
    fn from_record(record: &serde_json::Value) -> Self {
        Self(Some(
            record["child_pid"].as_i64().expect("owned index pid") as i32
        ))
    }
    fn disarm(&mut self) {
        self.0 = None;
    }
}
impl Drop for OwnedIndexer {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
}
fn bounded_exit(child: &mut OwnedChild) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "public query did not reach terminal state within 30s"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn git(root: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
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
            .env_remove("GREPPY_DELEGATED_BACKGROUND_JOB")
            .env_remove("GREPPY_DISCOVER_INCLUDE")
            .env_remove("GREPPY_DISCOVER_EXCLUDE");
        cmd
    }
    fn query(&self) -> String {
        let out = self
            .command(&["who-calls", "target", "--json"])
            .output()
            .unwrap();
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
            r#"import json, os, subprocess, sys
args=sys.argv[sys.argv.index('--')+1:]
child=subprocess.Popen(args)
with open(os.environ['REPAIR_GATE_RECORD'], 'a') as f:
    f.write(json.dumps({'gate_pid':os.getpid(),'child_pid':child.pid,'argv':args})+'\n')
sys.exit(child.wait())
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
            _scratch: scratch,
        };
        let out = fixture.command(&["index", "."]).output().unwrap();
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
        delta
            .conn()
            .execute("DELETE FROM edges WHERE edge_type='USAGE'", [])
            .unwrap();
        drop(delta);
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
    let mut indexer = OwnedIndexer::from_record(&admission);
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
    indexer.disarm();
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

/// Killing only the real index child lets its gate reap it and release ownership.
/// The query retains demand until that failure is observed; it cannot publish.
#[test]
fn public_query_cancelled_before_publication_recovers_only_certified_candidate() {
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
    let record = std::fs::read_to_string(&f.record).unwrap();
    let admission: serde_json::Value =
        serde_json::from_str(record.lines().next().unwrap()).unwrap();
    let mut indexer = OwnedIndexer::from_record(&admission);
    let child_pid = admission["child_pid"].as_i64().unwrap() as i32;
    assert_eq!(
        std::fs::read(&f.delta).unwrap(),
        active_before,
        "unpublished repair changed active Delta"
    );
    let out = f
        .command(&["index", "recover", ".", "--json"])
        .output()
        .unwrap();
    assert!(!out.status.success(), "live writer recovery must fail");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("index writer is still active"),
        "unexpected recovery refusal: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&f.delta).unwrap(), active_before);
    // Only a test-owned child from the admission record can be cancelled.
    assert_eq!(unsafe { libc::kill(child_pid, libc::SIGKILL) }, 0);
    assert!(
        !bounded_exit(&mut query).success(),
        "cancelled query claimed success"
    );
    indexer.disarm();
    assert!(
        candidate.exists(),
        "crash candidate must survive for explicit recovery"
    );
    assert_eq!(std::fs::read(&f.delta).unwrap(), active_before);
    // A recovered candidate cannot acquire certification merely by reopening.
    let marker = {
        let db = greppy_store::Store::open(&candidate).unwrap();
        let marker: String = db
            .conn()
            .query_row(
                "SELECT value FROM schema_meta WHERE key=?1",
                [greppy_indexer::RUST_CALLER_EDGES_REPAIR_META_KEY],
                |row| row.get(0),
            )
            .unwrap();
        db.conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [greppy_indexer::RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        marker
    };
    let rejected = f
        .command(&["index", "recover", ".", "--json"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let refusal: serde_json::Value = serde_json::from_slice(&rejected.stdout).unwrap();
    assert_eq!(refusal["status"], "rejected");
    assert!(
        refusal["reason"]
            .as_str()
            .unwrap()
            .contains("compatibility preparation"),
        "{refusal}"
    );
    assert!(candidate.exists());
    assert_eq!(std::fs::read(&f.delta).unwrap(), active_before);
    {
        let db = greppy_store::Store::open(&candidate).unwrap();
        db.conn()
            .execute(
                "INSERT INTO schema_meta(key,value) VALUES(?1,?2)",
                [
                    greppy_indexer::RUST_CALLER_EDGES_REPAIR_META_KEY,
                    marker.as_str(),
                ],
            )
            .unwrap();
    }
    let out = f
        .command(&["index", "recover", ".", "--json"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "completed candidate recovery failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let recovery: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(recovery["status"], "published", "{recovery}");
    assert_eq!(recovery["candidate"], candidate.to_string_lossy().as_ref());
    assert!(!candidate.exists());
    f.assert_certified();
    f.query();
    assert_eq!(
        std::fs::read_to_string(&f.record).unwrap(),
        record,
        "recovered snapshot must not need a second repair"
    );
    assert_eq!(std::fs::read(&f.base).unwrap(), base_before);
}
