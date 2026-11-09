//! Actual legacy CLI -> unchanged warm store -> normal current read.
//! The CI helper builds the immutable pre-fix production sources and supplies
//! target/tmp/greppy-pre-typedef. These tests are never a cold-fixture substitute.
use greppy_indexer::embedding::{
    index_code_embeddings_for_project, CodeEmbeddingProvider, EmbeddingIndexOptions,
};
use std::path::{Path, PathBuf};
use std::process::Command;

const PROJECT: &str = "extraction-revision-fixture";
const DEFINITION: &str =
    "typedef struct {\n    int frames;\n    float gain;\n} SyntheticWorkspace;\n";

fn legacy() -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("greppy-pre-typedef");
    assert!(
        path.is_file(),
        "actual legacy CLI missing; run the revision mutation helper"
    );
    path
}

fn git(repo: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
}

fn fixture(root: &Path, extension: &str) -> PathBuf {
    let repo = root.join("repo");
    std::fs::create_dir(&repo).unwrap();
    std::fs::write(
        repo.join(format!("types.{extension}")),
        format!("{DEFINITION}int unrelated(void);\n"),
    )
    .unwrap();
    std::fs::write(repo.join("keep.c"), "int retained_c(void) { return 7; }\n").unwrap();
    std::fs::write(
        repo.join("keep.go"),
        "package fixture\nfunc retained_go() int { return 11 }\n",
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
            "synthetic source",
        ],
    );
    repo
}

fn run(
    binary: &Path,
    repo: &Path,
    store: &Path,
    args: &[&str],
    overlay: Option<(&Path, &str)>,
) -> (i32, String, String) {
    let mut cmd = Command::new(binary);
    cmd.args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_PROJECT_IDENTITY", PROJECT)
        .env("GREPPY_FRESHNESS_TTL_MS", "60000")
        .env_remove("GREPPY_AUTO_REINDEX")
        .env_remove("GREPPY_DISCOVER_INCLUDE")
        .env_remove("GREPPY_DISCOVER_EXCLUDE")
        .env_remove("GREPPY_AGENT_STORE_MODE")
        .env_remove("GREPPY_AGENT_BASE_STORE")
        .env_remove("GREPPY_AGENT_BASE_COMMIT");
    if let Some((base, commit)) = overlay {
        cmd.env("GREPPY_AGENT_STORE_MODE", "overlay")
            .env("GREPPY_AGENT_BASE_STORE", base)
            .env("GREPPY_AGENT_BASE_COMMIT", commit);
    }
    let out = cmd.output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn graph_below(root: &Path) -> PathBuf {
    fn visit(root: &Path, result: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, result);
            } else if path.file_name().unwrap() == "graph.db" {
                result.push(path);
            }
        }
    }
    let mut found = Vec::new();
    visit(root, &mut found);
    assert_eq!(
        found.len(),
        1,
        "exactly the fixture's graph is required: {found:?}"
    );
    found.remove(0)
}

fn indexed_legacy(repo: &Path, store: &Path, overlay: Option<(&Path, &str)>) -> PathBuf {
    let (code, out, err) = run(&legacy(), repo, store, &["index", "."], overlay);
    assert_eq!(code, 0, "legacy real index failed: {out}\n{err}");
    let (code, out, err) = run(
        &legacy(),
        repo,
        store,
        &["read", "SyntheticWorkspace", "--head", "150"],
        overlay,
    );
    assert_eq!(
        code, 1,
        "legacy read must exhibit the original omission: {out}\n{err}"
    );
    assert!(
        out.contains("no symbol") || err.contains("no symbol"),
        "wrong legacy failure: {out}\n{err}"
    );
    graph_below(store)
}

#[derive(Default)]
struct CacheContractProvider {
    inputs: Vec<String>,
}
impl CodeEmbeddingProvider for CacheContractProvider {
    fn model_id(&self) -> &str {
        "fixture-cache-contract"
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

// This is a content-key reuse contract with fixture vectors, never a GPU/model
// acceptance claim. Production candidate/chunk/hash/reuse code executes unchanged.
fn cache_pass(graph: &Path, repo: &Path) -> CacheContractProvider {
    let mut store = greppy_store::Store::open(graph).unwrap();
    let generation: u64 = store
        .conn()
        .query_row(
            "SELECT MAX(graph_generation) FROM workspace_state",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let mut provider = CacheContractProvider::default();
    index_code_embeddings_for_project(
        &mut store,
        repo,
        PROJECT,
        &mut provider,
        EmbeddingIndexOptions::for_generation(generation),
    )
    .unwrap();
    store
        .conn()
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    provider
}

fn identities(graph: &Path) -> Vec<(String, i64)> {
    let conn =
        rusqlite::Connection::open_with_flags(graph, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT name,id FROM nodes WHERE name IN ('retained_c','retained_go') ORDER BY name",
        )
        .unwrap();
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    rows.collect::<Result<_, _>>().unwrap()
}

fn vectors(graph: &Path) -> Vec<(String, String, Vec<u8>)> {
    let conn =
        rusqlite::Connection::open_with_flags(graph, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    let mut stmt = conn.prepare("SELECT qualified_name,content_sha256,vector FROM vector_embeddings ORDER BY qualified_name,chunk_idx").unwrap();
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap();
    rows.collect::<Result<_, _>>().unwrap()
}

fn read_current(repo: &Path, store: &Path, overlay: Option<(&Path, &str)>) {
    let (code, out, err) = run(
        Path::new(env!("CARGO_BIN_EXE_greppy")),
        repo,
        store,
        &["read", "SyntheticWorkspace", "--head", "150"],
        overlay,
    );
    assert_eq!(
        code, 0,
        "warm revision read did not heal the legacy omission: {out}\n{err}"
    );
    for expected in [
        "typedef struct {",
        "int frames;",
        "float gain;",
        "} SyntheticWorkspace;",
    ] {
        assert!(
            out.contains(expected),
            "warm read omitted typedef content: {out}"
        );
    }
    let (code, out, err) = run(
        Path::new(env!("CARGO_BIN_EXE_greppy")),
        repo,
        store,
        &["read", "SyntheticWorkspace", "--json"],
        overlay,
    );
    assert_eq!(code, 0, "{out}\n{err}");
    let value: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(value["source"].as_str().unwrap().trim(), DEFINITION.trim());
}

#[test]
#[ignore = "requires immutable legacy CLI built by tools/check_anonymous_compound_typedef_mutation.py"]
fn extraction_revision_warm_private_read_preserves_other_languages_and_reuse() {
    for extension in ["c", "h", "hpp"] {
        let scratch = tempfile::tempdir().unwrap();
        let repo = fixture(scratch.path(), extension);
        let store = scratch.path().join("store");
        let graph = indexed_legacy(&repo, &store, None);
        let seed = cache_pass(&graph, &repo);
        assert!(seed.inputs.iter().any(|s| s.contains("retained_c")));
        assert!(seed.inputs.iter().any(|s| s.contains("retained_go")));
        let old_ids = identities(&graph);
        let old_vectors = vectors(&graph);
        let fingerprint = greppy_core::GitFingerprint::capture(&repo);
        let hashes = [
            format!("types.{extension}"),
            "keep.c".into(),
            "keep.go".into(),
        ]
        .map(|p| git(&repo, &["hash-object", &p]));
        read_current(&repo, &store, None);
        assert_eq!(
            identities(&graph),
            old_ids,
            "revision-only refresh invalidated unchanged node identities"
        );
        assert_eq!(
            vectors(&graph),
            old_vectors,
            "revision-only refresh discarded valid cached vectors"
        );
        let now = greppy_core::GitFingerprint::capture(&repo);
        assert_eq!(now.head_oid, fingerprint.head_oid);
        assert_eq!(now.index_signature, fingerprint.index_signature);
        let after = [
            format!("types.{extension}"),
            "keep.c".into(),
            "keep.go".into(),
        ]
        .map(|p| git(&repo, &["hash-object", &p]));
        assert_eq!(after, hashes, "warm upgrade changed source bytes");
        let current =
            greppy_store::Store::open_with(&graph, greppy_store::OpenOptions::read_only()).unwrap();
        for state in current.list_file_states(PROJECT).unwrap() {
            let expected = if state.rel_path == "keep.go" {
                greppy_parser::LEGACY_EXTRACTION_REVISION
            } else {
                greppy_parser::extraction_revision(greppy_parser::Language::C)
            };
            assert_eq!(
                state.extractor_version, expected,
                "wrong file-scoped revision: {}",
                state.rel_path
            );
        }
        drop(current);
        let reused = cache_pass(&graph, &repo);
        assert!(
            !reused
                .inputs
                .iter()
                .any(|s| s.contains("retained_c") || s.contains("retained_go")),
            "unchanged embedding input reached backend instead of actual cache reuse"
        );
    }
}

#[test]
#[ignore = "requires immutable legacy CLI built by tools/check_anonymous_compound_typedef_mutation.py"]
fn extraction_revision_warm_immutable_base_read_uses_only_private_delta() {
    for extension in ["h", "hpp"] {
        let scratch = tempfile::tempdir().unwrap();
        let repo = fixture(scratch.path(), extension);
        let legacy_store = scratch.path().join("legacy-store");
        let old_graph = indexed_legacy(&repo, &legacy_store, None);
        cache_pass(&old_graph, &repo);
        let commit = git(&repo, &["rev-parse", "HEAD"]);
        let identity = greppy_store::BaseStoreIdentity {
            format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
            canonical_repository_identity: "fixture:extraction-revision".into(),
            git_object_format: "sha1".into(),
            base_tree_oid: git(&repo, &["rev-parse", "HEAD^{tree}"]),
            store_schema_version: greppy_store::migrate::CURRENT_VERSION,
            indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
            parser_and_extractor_versions: format!(
                "greppy-parser/extractor-{}",
                env!("CARGO_PKG_VERSION")
            ),
            summary_model_and_prompt_version: "fixture-summary".into(),
            embedding_model: "fixture-cache-contract".into(),
            embedding_prompt_version: "fixture-v1".into(),
            embedding_dimensions: 2,
            embedding_encoding: "f32+i8-v1".into(),
        };
        let layout =
            greppy_store::BaseStoreLayout::new(&scratch.path().join("shared"), &identity).unwrap();
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
        let before = std::fs::read(&layout.graph).unwrap();
        let task = scratch.path().join("task");
        git(
            &repo,
            &["worktree", "add", "-qb", "task", task.to_str().unwrap()],
        );
        let delta = scratch.path().join("delta");
        let overlay = Some((layout.graph.as_path(), commit.as_str()));
        let delta_graph = indexed_legacy(&task, &delta, overlay);
        let fingerprint = greppy_core::GitFingerprint::capture(&task);
        let hashes = [
            format!("types.{extension}"),
            "keep.c".into(),
            "keep.go".into(),
        ]
        .map(|path| git(&task, &["hash-object", &path]));
        read_current(&task, &delta, overlay);
        assert_eq!(
            std::fs::read(&layout.graph).unwrap(),
            before,
            "immutable Base was modified by revision refresh"
        );
        layout.read_verified_manifest().unwrap();
        assert_eq!(
            greppy_core::GitFingerprint::capture(&task).head_oid,
            fingerprint.head_oid
        );
        let hashes_after = [
            format!("types.{extension}"),
            "keep.c".into(),
            "keep.go".into(),
        ]
        .map(|path| git(&task, &["hash-object", &path]));
        assert_eq!(
            hashes_after, hashes,
            "warm Base upgrade changed source bytes"
        );
        assert_eq!(git(&task, &["status", "--porcelain"]), "");
        let conn = rusqlite::Connection::open_with_flags(
            &delta_graph,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        let private: Vec<String> = conn
            .prepare("SELECT rel_path FROM file_state ORDER BY rel_path")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let raw: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key='store_cow.visibility.v1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let proof: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            proof["extraction_compatibility"]["policy"],
            greppy_parser::extraction_revision_policy_key()
        );
        assert_eq!(
            proof["extraction_compatibility"]["base_manifest_sha256"],
            greppy_store::file_state::sha256_hex(&std::fs::read(&layout.manifest).unwrap())
        );
        let revision_paths: Vec<String> =
            serde_json::from_value(proof["extraction_compatibility"]["paths"].clone()).unwrap();
        assert_eq!(
            revision_paths,
            vec!["keep.c".to_owned(), format!("types.{extension}")]
        );
        assert_eq!(
            private,
            vec!["keep.c".to_owned(), format!("types.{extension}")],
            "Delta re-extracted unrelated languages or missed revised C/C++ paths"
        );
        let rust: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM nodes WHERE file_path='keep.go' AND label <> 'Folder'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            rust, 0,
            "unchanged Base language was copied into private graph"
        );
        drop(conn);
        let visibility = greppy_store::VisibilityIndex::new(private, []).unwrap();
        let mut composed =
            greppy_store::Store::open_overlay(&layout.graph, &delta_graph, &visibility).unwrap();
        let generation: u64 = composed
            .conn()
            .query_row(
                "SELECT MAX(graph_generation) FROM workspace_state",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let mut provider = CacheContractProvider::default();
        let mut ignore = |_| {};
        greppy_indexer::embedding::index_code_embeddings_for_scope_with_progress(
            &mut composed,
            &task,
            PROJECT,
            &mut provider,
            EmbeddingIndexOptions::for_generation(generation),
            greppy_indexer::embedding::EmbeddingIndexProgressContext {
                total_documents: 0,
                callback: &mut ignore,
                validation_callback: None,
            },
            &["keep.c".into(), format!("types.{extension}")],
        )
        .unwrap();
        assert!(
            !provider.inputs.iter().any(|s| s.contains("retained_c")),
            "exact hidden Base input reached backend instead of content reuse"
        );
        assert!(
            provider
                .inputs
                .iter()
                .any(|s| s.contains("SyntheticWorkspace")),
            "new alias input incorrectly counted as a Base cache hit"
        );
        assert_eq!(
            std::fs::read(&layout.graph).unwrap(),
            before,
            "content reuse mutated immutable Base"
        );
        drop(composed);
        // A cached revision boundary must survive deletion and exact restore:
        // the underlying immutable Base still has the legacy extraction.
        let header = task.join(format!("types.{extension}"));
        let original_header = std::fs::read(&header).unwrap();
        std::fs::remove_file(&header).unwrap();
        let (code, out, err) = run(
            Path::new(env!("CARGO_BIN_EXE_greppy")),
            &task,
            &delta,
            &["read", "SyntheticWorkspace", "--head", "150"],
            overlay,
        );
        assert_eq!(code, 1, "deleted typedef remained visible: {out}\n{err}");
        std::fs::write(&header, original_header).unwrap();
        read_current(&task, &delta, overlay);
        assert_eq!(std::fs::read(&layout.graph).unwrap(), before);
    }
}
