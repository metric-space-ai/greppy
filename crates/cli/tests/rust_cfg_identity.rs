//! Real Rust extraction -> persisted graph -> public read, without target evaluation.
use greppy_indexer::embedding::{
    index_code_embeddings_for_project, CodeEmbeddingProvider, EmbeddingIndexOptions,
};
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = "#[cfg(unix)]\npub fn platform_token() -> u8 { unix_leaf() }\n#[cfg(not(unix))]\npub fn platform_token() -> u8 { other_leaf() }\nfn unix_leaf() -> u8 { 1 }\nfn other_leaf() -> u8 { 2 }\nfn caller() -> u8 { platform_token() }\n";

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    run_binary(
        Path::new(env!("CARGO_BIN_EXE_greppy")),
        repo,
        store,
        args,
        None,
    )
}

fn run_binary(
    binary: &Path,
    repo: &Path,
    store: &Path,
    args: &[&str],
    overlay: Option<(&Path, &str)>,
) -> (i32, String, String) {
    let mut command = Command::new(binary);
    if let Some((base, commit)) = overlay {
        command
            .env("GREPPY_AGENT_STORE_MODE", "overlay")
            .env("GREPPY_AGENT_BASE_STORE", base)
            .env("GREPPY_AGENT_BASE_COMMIT", commit);
    } else {
        command
            .env_remove("GREPPY_AGENT_STORE_MODE")
            .env_remove("GREPPY_AGENT_BASE_STORE")
            .env_remove("GREPPY_AGENT_BASE_COMMIT");
    }
    let out = command
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_FRESHNESS_TTL_MS", "60000")
        .env_remove("GREPPY_AUTO_REINDEX")
        .env("GREPPY_PROJECT_IDENTITY", "rust-cfg-contract")
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

fn graph(root: &Path) -> PathBuf {
    let mut graphs = Vec::new();
    fn visit(root: &Path, graphs: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, graphs);
            } else if path.file_name().unwrap() == "graph.db" {
                graphs.push(path);
            }
        }
    }
    visit(root, &mut graphs);
    assert_eq!(graphs.len(), 1);
    graphs.remove(0)
}

fn fixture(root: &Path, source: &str) -> (PathBuf, PathBuf) {
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(repo.join("lib.rs"), source).unwrap();
    (repo, root.join("store"))
}

#[test]
fn rust_cfg_siblings_survive_parser_store_and_public_read() {
    // Source counterprobe: changing only the second name removes the collision.
    let control = tempfile::tempdir().unwrap();
    let source = SOURCE.replacen("pub fn platform_token", "pub fn renamed_token", 1);
    let (repo, store) = fixture(control.path(), &source);
    assert_eq!(run(&repo, &store, &["index", "."]).0, 0);
    let (code, out, err) = run(&repo, &store, &["read", "renamed_token", "--json"]);
    assert_eq!(code, 0, "{out}\n{err}");
    assert!(out.contains("unix_leaf()"), "{out}");

    let scratch = tempfile::tempdir().unwrap();
    let (repo, store) = fixture(scratch.path(), SOURCE);
    let parsed =
        greppy_parser::extract(greppy_parser::Language::Rust, SOURCE.as_bytes(), "lib.rs").unwrap();
    let parsed_count = parsed
        .nodes
        .iter()
        .filter(|n| n.name == "platform_token" && n.label == "Function")
        .count();
    assert_eq!(
        parsed_count, 2,
        "fixture must exercise both real parser definitions"
    );
    let (code, out, err) = run(&repo, &store, &["index", "."]);
    assert_eq!(code, 0, "{out}\n{err}");
    // Invoke the public read before asserting the persisted cardinality, so the
    // original failure records every layer rather than stopping at the parser.
    let (code, out, err) = run(&repo, &store, &["read", "platform_token", "--json"]);
    let conn = rusqlite::Connection::open(graph(&store)).unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT count(*) FROM nodes WHERE name='platform_token' AND label='Function'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 2, "cfg sibling definitions collapsed: parser={parsed_count}, store={count}, read_exit={code}, read={out}, stderr={err}");
    let value: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        value["status"], "ambiguous",
        "read silently selected a cfg body: {out}\n{err}"
    );
    assert_ne!(
        code, 0,
        "ambiguous read must not report a successful unique definition"
    );
    assert_warm_repaired(&repo, &store, None);
    let outgoing: Vec<(i64, String)> = conn.prepare("SELECT s.start_line,t.name FROM edges e JOIN nodes s ON s.id=e.source_id JOIN nodes t ON t.id=e.target_id WHERE e.edge_type='CALLS' AND s.name='platform_token' ORDER BY s.start_line,t.name").unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().collect::<Result<_, _>>().unwrap();
    assert_eq!(
        outgoing,
        vec![(2, "unix_leaf".into()), (4, "other_leaf".into())],
        "cfg sibling outgoing ownership merged"
    );
    let guessed: i64 = conn.query_row("SELECT count(*) FROM edges e JOIN nodes s ON s.id=e.source_id JOIN nodes t ON t.id=e.target_id WHERE e.edge_type='CALLS' AND s.name='caller' AND t.name='platform_token'", [], |r| r.get(0)).unwrap();
    assert_eq!(guessed, 0, "incoming ambiguous call guessed a platform");
    let same_line =
        "#[cfg(unix)] fn sibling() { first(); } #[cfg(not(unix))] fn sibling() { second(); }";
    let parsed = greppy_parser::extract(
        greppy_parser::Language::Rust,
        same_line.as_bytes(),
        "same.rs",
    )
    .unwrap();
    let identities: std::collections::HashSet<_> = parsed
        .nodes
        .iter()
        .filter(|n| n.name == "sibling")
        .map(|n| n.qualified_name.as_str())
        .collect();
    assert_eq!(identities.len(), 2, "same-line cfg siblings share identity");
    let owners: std::collections::HashSet<_> = parsed
        .edges
        .iter()
        .filter(|e| e.edge_type == "CALLS")
        .map(|e| e.source_qualified_name.as_str())
        .collect();
    assert_eq!(
        owners, identities,
        "same-line call ownership does not match definition identity"
    );
}

fn git(repo: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().into()
}

#[derive(Default)]
struct CacheProvider {
    inputs: Vec<String>,
}
impl CodeEmbeddingProvider for CacheProvider {
    fn model_id(&self) -> &str {
        "fixture-rust-cfg-cache"
    }
    fn prompt_version(&self) -> &str {
        "fixture-v1"
    }
    fn task_profile(&self) -> &str {
        "embeddinggemma_code_retrieval"
    }
    fn embed_code_document(
        &mut self,
        _title: Option<&str>,
        content: &str,
    ) -> greppy_core::Result<Vec<f32>> {
        self.inputs.push(content.into());
        Ok(vec![1.0, 0.0])
    }
}

// Executes the real candidate/chunk/content-key cache pipeline with fixture
// vectors. This proves reuse behavior, never model or GPU acceptance.
fn cache_pass(store: &mut greppy_store::Store, repo: &Path) -> Vec<String> {
    let generation: u64 = store
        .conn()
        .query_row(
            "SELECT MAX(graph_generation) FROM workspace_state",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let mut provider = CacheProvider::default();
    index_code_embeddings_for_project(
        store,
        repo,
        "rust-cfg-contract",
        &mut provider,
        EmbeddingIndexOptions::for_generation(generation),
    )
    .unwrap();
    store
        .conn()
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    provider.inputs
}

fn stable_vectors(conn: &rusqlite::Connection) -> Vec<(i64, i64, String, Vec<u8>)> {
    conn.prepare("SELECT v.id,v.node_id,v.content_sha256,v.vector FROM vector_embeddings v JOIN nodes n ON n.id=v.node_id WHERE n.name IN ('unix_leaf','other_leaf','caller','Retained') ORDER BY n.name,v.chunk_idx").unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).unwrap().collect::<Result<_,_>>().unwrap()
}

fn assert_cache_inputs(inputs: &[String]) {
    assert!(
        !inputs.iter().any(|s| s.contains("fn unix_leaf(")
            || s.contains("fn other_leaf(")
            || s.contains("fn caller(")
            || s.contains("func Retained(")),
        "unchanged definition reached fixture backend instead of actual reuse: {inputs:?}"
    );
    assert_eq!(
        inputs
            .iter()
            .filter(|s| s.contains("pub fn platform_token("))
            .count(),
        2,
        "new disambiguated identities must not reuse the collapsed legacy vector: {inputs:?}"
    );
}

fn assert_warm_repaired(repo: &Path, store: &Path, overlay: Option<(&Path, &str)>) {
    let binary = Path::new(env!("CARGO_BIN_EXE_greppy"));
    let (code, out, err) = run_binary(
        binary,
        repo,
        store,
        &["read", "platform_token", "--json"],
        overlay,
    );
    let value: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        value["status"], "ambiguous",
        "warm cfg revision did not repair collapsed identity: exit={code}, {out}\n{err}"
    );
    assert_eq!(code, 1, "{out}\n{err}");
    let candidates = value["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 2, "{out}");
    for target in ["lib.rs::platform_token", "lib.rs::Function::platform_token"] {
        let (code, out, err) =
            run_binary(binary, repo, store, &["read", target, "--json"], overlay);
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(code, 1, "{out}\n{err}");
        assert_eq!(
            value["status"], "ambiguous",
            "legacy file-qualified selector lost candidates: {out}\n{err}"
        );
        assert_eq!(value["candidates"].as_array().unwrap().len(), 2, "{out}");
    }
    let mut bodies = Vec::new();
    for candidate in candidates {
        let selector = candidate["selector"].as_str().unwrap();
        let (code, out, err) = run_binary(
            binary,
            repo,
            store,
            &["read", selector, "--json", "--handle"],
            overlay,
        );
        assert_eq!(code, 0, "{out}\n{err}");
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(value["handle"].as_str().unwrap().starts_with("geh2:"));
        bodies.push(value["source"].as_str().unwrap().to_owned());
    }
    assert_eq!(
        bodies
            .iter()
            .filter(|body| body.contains("unix_leaf()") && !body.contains("other_leaf()"))
            .count(),
        1,
        "{bodies:?}"
    );
    assert_eq!(
        bodies
            .iter()
            .filter(|body| body.contains("other_leaf()") && !body.contains("unix_leaf()"))
            .count(),
        1,
        "{bodies:?}"
    );
}

#[test]
#[ignore = "requires actual immutable legacy CLI built by check_rust_cfg_identity_mutation.py"]
fn rust_cfg_warm_private_repair_same_source_through_normal_read() {
    warm_case(false);
}

#[test]
#[ignore = "requires actual immutable legacy CLI built by check_rust_cfg_identity_mutation.py"]
fn rust_cfg_warm_base_repair_same_source_through_normal_read() {
    warm_case(true);
}

fn warm_case(use_base: bool) {
    let legacy = Path::new(env!("CARGO_TARGET_TMPDIR")).join("greppy-pre-rust-cfg");
    assert!(legacy.is_file(), "actual pre-fix CLI required");
    {
        let scratch = tempfile::tempdir().unwrap();
        let repo = scratch.path().join("repo");
        std::fs::create_dir(&repo).unwrap();
        std::fs::write(repo.join("lib.rs"), SOURCE).unwrap();
        std::fs::write(
            repo.join("control.go"),
            "package control\nfunc Retained() int { return 3 }\n",
        )
        .unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["add", "."]);
        git(
            &repo,
            &[
                "-c",
                "user.email=fixture@invalid",
                "-c",
                "user.name=Fixture",
                "commit",
                "-qm",
                "cfg siblings",
            ],
        );
        let commit = git(&repo, &["rev-parse", "HEAD"]);
        let store = scratch.path().join("legacy-store");
        let (code, out, err) = run_binary(&legacy, &repo, &store, &["index", "."], None);
        assert_eq!(code, 0, "{out}\n{err}");
        let (code, out, err) = run_binary(
            &legacy,
            &repo,
            &store,
            &["read", "platform_token", "--json"],
            None,
        );
        assert_eq!(
            code, 0,
            "legacy must reproduce successful unique wrong body: {out}\n{err}"
        );
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(value["source"].as_str().unwrap().contains("other_leaf()"));
        let old_graph = graph(&store);
        let mut seeded = greppy_store::Store::open(&old_graph).unwrap();
        let seed_inputs = cache_pass(&mut seeded, &repo);
        assert!(seed_inputs.iter().any(|s| s.contains("fn unix_leaf(")));
        let old_vectors = stable_vectors(seeded.conn());
        assert_eq!(
            old_vectors.len(),
            4,
            "real cache seeding missed stable Rust/Go definitions"
        );
        drop(seeded);
        let before_conn = rusqlite::Connection::open(&old_graph).unwrap();
        // Successful incremental publication advances every surviving file's
        // generation. Hash/revision plus node/vector identity prove retention;
        // that shared publication stamp does not prove re-extraction.
        let control_state: String = before_conn
            .query_row(
                "SELECT sha256||':'||extractor_version FROM file_state WHERE rel_path='control.go'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let stable_id: i64 = before_conn
            .query_row(
                "SELECT id FROM nodes WHERE name='unix_leaf' AND label='Function'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        drop(before_conn);
        if !use_base {
            assert_warm_repaired(&repo, &store, None);
            let conn = rusqlite::Connection::open(graph(&store)).unwrap();
            assert_eq!(
                stable_vectors(&conn),
                old_vectors,
                "unchanged vector identities/payloads were discarded by revision refresh"
            );
            assert_eq!(
                conn.query_row(
                    "SELECT id FROM nodes WHERE name='unix_leaf' AND label='Function'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                stable_id,
                "unchanged Rust definition identity was discarded"
            );
            assert_eq!(conn.query_row("SELECT sha256||':'||extractor_version FROM file_state WHERE rel_path='control.go'", [], |r| r.get::<_, String>(0)).unwrap(), control_state, "unaffected language re-extracted");
            drop(conn);
            let mut refreshed = greppy_store::Store::open(&old_graph).unwrap();
            assert_cache_inputs(&cache_pass(&mut refreshed, &repo));
        } else {
            let identity = greppy_store::BaseStoreIdentity {
                format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
                canonical_repository_identity: "fixture:rust-cfg".into(),
                git_object_format: "sha1".into(),
                base_tree_oid: git(&repo, &["rev-parse", "HEAD^{tree}"]),
                store_schema_version: greppy_store::migrate::CURRENT_VERSION,
                indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
                parser_and_extractor_versions: format!(
                    "greppy-parser/extractor-{}",
                    env!("CARGO_PKG_VERSION")
                ),
                summary_model_and_prompt_version: "fixture-summary".into(),
                embedding_model: "fixture-rust-cfg-cache".into(),
                embedding_prompt_version: "fixture-v1".into(),
                embedding_dimensions: 2,
                embedding_encoding: "f32+i8-v1".into(),
            };
            let layout =
                greppy_store::BaseStoreLayout::new(&scratch.path().join("shared"), &identity)
                    .unwrap();
            let _builder = layout.acquire_builder(false).unwrap().unwrap();
            let summary = scratch.path().join("summary");
            drop(greppy_store::SummaryCache::open(&summary).unwrap());
            layout
                .publish_graph_with_summary(
                    identity,
                    &old_graph,
                    &summary.join(greppy_store::SUMMARY_CACHE_DB_FILE),
                )
                .unwrap();
            let base_bytes = std::fs::read(&layout.graph).unwrap();
            let task = scratch.path().join("task");
            git(
                &repo,
                &["worktree", "add", "-qb", "task", task.to_str().unwrap()],
            );
            let delta = scratch.path().join("delta");
            let overlay = Some((layout.graph.as_path(), commit.as_str()));
            let (code, out, err) = run_binary(&legacy, &task, &delta, &["index", "."], overlay);
            assert_eq!(code, 0, "{out}\n{err}");
            let (code, out, err) = run_binary(
                &legacy,
                &task,
                &delta,
                &["read", "platform_token", "--json"],
                overlay,
            );
            assert_eq!(code, 0, "{out}\n{err}");
            assert_warm_repaired(&task, &delta, overlay);
            let conn = rusqlite::Connection::open(graph(&delta)).unwrap();
            let paths: Vec<String> = conn
                .prepare("SELECT rel_path FROM file_state ORDER BY rel_path")
                .unwrap()
                .query_map([], |r| r.get(0))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            assert_eq!(
                paths,
                vec!["lib.rs"],
                "Base upgrade copied unaffected language"
            );
            drop(conn);
            let visibility = greppy_store::VisibilityIndex::new(["lib.rs".into()], []).unwrap();
            let mut composed =
                greppy_store::Store::open_overlay(&layout.graph, &graph(&delta), &visibility)
                    .unwrap();
            assert_cache_inputs(&cache_pass(&mut composed, &task));
            drop(composed);
            assert_eq!(
                std::fs::read(&layout.graph).unwrap(),
                base_bytes,
                "warm read mutated immutable Base"
            );
            layout.read_verified_manifest().unwrap();
            assert_eq!(git(&task, &["status", "--porcelain"]), "");
        }
        assert_eq!(git(&repo, &["rev-parse", "HEAD"]), commit);
        assert_eq!(
            std::fs::read_to_string(repo.join("lib.rs")).unwrap(),
            SOURCE
        );
        assert_eq!(git(&repo, &["status", "--porcelain"]), "");
    }
}
