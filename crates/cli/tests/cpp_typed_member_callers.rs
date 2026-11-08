//! Actual generic C++ parser -> stored graph -> public caller query.
use std::path::{Path, PathBuf};
use std::process::Command;

const HEADER: &str = "struct PrimaryJob { void begin(); };\nstruct OtherJob { void begin(); };\nstruct Unknown {};\nstruct Pipeline {\n PrimaryJob job_;\n OtherJob other_;\n void run(); void run_other(); void shadow(Unknown& job_);\n void local_shadow(); void unknown(Unknown& value); void range_shadow();\n};\n";
const CALLER: &str = "#include \"jobs.hpp\"\nvoid Pipeline::run() { job_.begin(); }\nvoid Pipeline::run_other() { other_.begin(); }\nvoid Pipeline::shadow(Unknown& job_) { job_.begin(); }\nvoid Pipeline::local_shadow() { Unknown job_; job_.begin(); }\nvoid Pipeline::unknown(Unknown& value) { value.begin(); }\nvoid Pipeline::range_shadow() { Unknown jobs[1]; for (auto& job_ : jobs) { job_.begin(); } }\n";

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
fn fixture(root: &Path) -> (PathBuf, PathBuf) {
    let repo = root.join("repo");
    std::fs::create_dir(&repo).unwrap();
    for (name, source) in [
        ("jobs.hpp", HEADER),
        ("caller.cpp", CALLER),
        (
            "primary.cpp",
            "#include \"jobs.hpp\"\nvoid PrimaryJob::begin() {}\n",
        ),
        (
            "other.cpp",
            "#include \"jobs.hpp\"\nvoid OtherJob::begin() {}\n",
        ),
    ] {
        std::fs::write(repo.join(name), source).unwrap();
    }
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
            "generic typed C++ member",
        ],
    );
    (repo, root.join("store"))
}
fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_PROJECT_IDENTITY", "cpp-typed-member-contract")
        .env_remove("GREPPY_AGENT_STORE_MODE")
        .env_remove("GREPPY_AGENT_BASE_STORE")
        .env_remove("GREPPY_AGENT_BASE_COMMIT")
        .env_remove("GREPPY_AUTO_REINDEX")
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

#[test]
fn cpp_typed_members_keep_owner_and_reject_shadowed_or_unknown_receivers() {
    let scratch = tempfile::tempdir().unwrap();
    let (repo, store) = fixture(scratch.path());
    let parsed =
        greppy_parser::extract(greppy_parser::Language::Cpp, HEADER.as_bytes(), "jobs.hpp")
            .unwrap();
    assert!(
        parsed.nodes.iter().any(|n| n.label == "Field"
            && n.name == "job_"
            && n.properties["return_type"] == "PrimaryJob"),
        "real parser did not retain declared member type"
    );
    let calls = greppy_parser::extract(
        greppy_parser::Language::Cpp,
        CALLER.as_bytes(),
        "caller.cpp",
    )
    .unwrap();
    assert_eq!(
        calls
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS" && e.properties["callee_name"] == "begin")
            .count(),
        6,
        "real C++ call syntax must be covered"
    );
    let (code, out, err) = run(&repo, &store, &["index", "."]);
    assert_eq!(code, 0, "{out}\n{err}");
    for (target, good, wrong) in [
        (
            "PrimaryJob::begin",
            "::Pipeline::run",
            "::Pipeline::run_other",
        ),
        (
            "OtherJob::begin",
            "::Pipeline::run_other",
            "::Pipeline::run",
        ),
    ] {
        let (code, out, err) = run(
            &repo,
            &store,
            &["who-calls", target, "--all", "--json", "--diagnostics"],
        );
        assert_eq!(code, 0, "{out}\n{err}");
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value["symbol_found"], true, "real target missing: {value}");
        assert_eq!(value["fresh"], true, "fixture graph not fresh: {value}");
        let hits = value["hits"].as_array().unwrap();
        assert_eq!(
            hits.len(),
            1,
            "typed C++ member caller missing or falsely assigned: target={target}, {value}"
        );
        let hit = hits[0].to_string();
        assert!(hit.contains(good), "{value}");
        // Exact QName matching avoids treating run_other as a run occurrence.
        assert!(
            !hit.contains(&format!("{wrong}\"")),
            "wrong C++ owner: {value}"
        );
        for rejected in ["::shadow", "::local_shadow", "::range_shadow", "::unknown"] {
            assert!(
                !hit.contains(rejected),
                "unproven/shadowed receiver gained a CALLS edge: {value}"
            );
        }
    }
}

#[test]
fn cpp_ambiguous_field_or_payload_owner_is_not_guessed() {
    let scratch = tempfile::tempdir().unwrap();
    let (repo, store) = fixture(scratch.path());
    std::fs::write(repo.join("jobs.hpp"), format!("{HEADER}\nstruct AmbiguousPipeline {{ PrimaryJob job_; void ambiguous(); }};\nstruct DuplicateJob {{ void begin(); }};\nstruct PayloadPipeline {{ DuplicateJob job_; void ambiguous_payload(); }};\n")).unwrap();
    std::fs::write(
        repo.join("duplicates.hpp"),
        "struct AmbiguousPipeline { OtherJob job_; };\nstruct DuplicateJob { void begin(); };\n",
    )
    .unwrap();
    std::fs::write(repo.join("caller.cpp"), format!("{CALLER}\nvoid AmbiguousPipeline::ambiguous() {{ job_.begin(); }}\nvoid PayloadPipeline::ambiguous_payload() {{ job_.begin(); }}\n")).unwrap();
    std::fs::write(
        repo.join("duplicate.cpp"),
        "#include \"jobs.hpp\"\nvoid DuplicateJob::begin() {}\n",
    )
    .unwrap();
    let (code, out, err) = run(&repo, &store, &["index", "."]);
    assert_eq!(code, 0, "{out}\n{err}");
    assert_callers(Path::new(env!("CARGO_BIN_EXE_greppy")), &repo, &store, None);
    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "who-calls",
            "DuplicateJob::begin",
            "--all",
            "--json",
            "--diagnostics",
        ],
    );
    assert_eq!(code, 0, "{out}\n{err}");
    let value: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(value["symbol_found"], true, "{value}");
    assert_eq!(
        value["hits"].as_array().unwrap().len(),
        0,
        "ambiguous payload owner was guessed: {value}"
    );
}

fn run_overlay(
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
        .env("GREPPY_PROJECT_IDENTITY", "cpp-typed-member-contract")
        .env_remove("GREPPY_AUTO_REINDEX")
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

fn assert_callers(binary: &Path, repo: &Path, store: &Path, overlay: Option<(&Path, &str)>) {
    for (target, expected) in [
        ("PrimaryJob::begin", "::Pipeline::run"),
        ("OtherJob::begin", "::Pipeline::run_other"),
    ] {
        let (code, out, err) = run_overlay(
            binary,
            repo,
            store,
            &["who-calls", target, "--all", "--json", "--diagnostics"],
            overlay,
        );
        assert_eq!(code, 0, "{out}\n{err}");
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value["symbol_found"], true, "{value}");
        assert_eq!(value["fresh"], true, "{value}");
        let hits = value["hits"].as_array().unwrap();
        assert_eq!(
            hits.len(),
            1,
            "typed C++ member caller missing or falsely assigned: target={target}, {value}"
        );
        assert!(
            hits[0].to_string().contains(&format!("{expected}\"")),
            "wrong typed C++ owner: {value}"
        );
    }
}

fn graph(root: &Path) -> PathBuf {
    fn visit(root: &Path, found: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, found);
            } else if path.file_name().unwrap() == "graph.db" {
                found.push(path);
            }
        }
    }
    let mut found = Vec::new();
    visit(root, &mut found);
    assert_eq!(found.len(), 1);
    found.remove(0)
}

#[derive(Default)]
struct CacheProvider {
    inputs: Vec<String>,
}
impl greppy_indexer::embedding::CodeEmbeddingProvider for CacheProvider {
    fn model_id(&self) -> &str {
        "fixture-cpp-member-cache"
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

// Real chunk/content-key pipeline; synthetic vectors prove reuse, not inference.
fn cache_pass(store: &mut greppy_store::Store, repo: &Path) -> Vec<String> {
    use greppy_indexer::embedding::{index_code_embeddings_for_project, EmbeddingIndexOptions};
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
        "cpp-typed-member-contract",
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
    conn.prepare("SELECT v.id,v.node_id,v.content_sha256,v.vector FROM vector_embeddings v JOIN nodes n ON n.id=v.node_id ORDER BY n.qualified_name,v.chunk_idx").unwrap().query_map([], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).unwrap().collect::<Result<_,_>>().unwrap()
}

#[test]
#[ignore = "requires actual immutable legacy CLI built by check_cpp_typed_member_mutation.py"]
fn cpp_member_warm_private_repair_preserves_vectors() {
    warm_case(false);
}

#[test]
#[ignore = "requires actual immutable legacy CLI built by check_cpp_typed_member_mutation.py"]
fn cpp_member_warm_base_repair_preserves_content_reuse() {
    warm_case(true);
}

fn warm_case(use_base: bool) {
    let legacy = Path::new(env!("CARGO_TARGET_TMPDIR")).join("greppy-pre-cpp-member");
    assert!(legacy.is_file(), "actual pre-fix CLI required");
    let scratch = tempfile::tempdir().unwrap();
    let (repo, store) = fixture(scratch.path());
    std::fs::write(
        repo.join("control.go"),
        "package control\nfunc Retained() int { return 3 }\n",
    )
    .unwrap();
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
            "unaffected language control",
        ],
    );
    let commit = git(&repo, &["rev-parse", "HEAD"]);
    let (code, out, err) = run_overlay(&legacy, &repo, &store, &["index", "."], None);
    assert_eq!(code, 0, "{out}\n{err}");
    let (code, out, err) = run_overlay(
        &legacy,
        &repo,
        &store,
        &[
            "who-calls",
            "PrimaryJob::begin",
            "--all",
            "--json",
            "--diagnostics",
        ],
        None,
    );
    assert_eq!(code, 0, "{out}\n{err}");
    let value: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(value["fresh"], true, "{value}");
    assert_eq!(value["symbol_found"], true, "{value}");
    assert!(
        value["hits"].as_array().unwrap().is_empty(),
        "actual legacy did not reproduce lost caller: {value}"
    );
    let old_graph = graph(&store);
    let mut seeded = greppy_store::Store::open(&old_graph).unwrap();
    let seed_inputs = cache_pass(&mut seeded, &repo);
    assert!(
        seed_inputs.iter().any(|s| s.contains("Pipeline::run()")),
        "real C++ cache not seeded: {seed_inputs:?}"
    );
    assert!(
        seed_inputs.iter().any(|s| s.contains("func Retained(")),
        "control cache not seeded"
    );
    let old_vectors = stable_vectors(seeded.conn());
    assert!(!old_vectors.is_empty());
    let old_nodes: Vec<(i64, String)> = seeded
        .conn()
        .prepare("SELECT id,qualified_name FROM nodes WHERE label IN ('Function','Method','Class','Field') ORDER BY qualified_name")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let control_state: String = seeded
        .conn()
        .query_row(
            "SELECT sha256||':'||extractor_version FROM file_state WHERE rel_path='control.go'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    drop(seeded);
    if !use_base {
        assert_callers(Path::new(env!("CARGO_BIN_EXE_greppy")), &repo, &store, None);
        let mut refreshed = greppy_store::Store::open(&old_graph).unwrap();
        assert_eq!(
            stable_vectors(refreshed.conn()),
            old_vectors,
            "revision refresh discarded unchanged vector IDs/keys/payloads"
        );
        let new_nodes: Vec<(i64, String)> = refreshed
            .conn()
            .prepare("SELECT id,qualified_name FROM nodes WHERE label IN ('Function','Method','Class','Field') ORDER BY qualified_name")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            new_nodes, old_nodes,
            "revision-only refresh discarded unchanged source-definition IDs"
        );
        assert_eq!(refreshed.conn().query_row("SELECT sha256||':'||extractor_version FROM file_state WHERE rel_path='control.go'", [], |r| r.get::<_,String>(0)).unwrap(), control_state, "unaffected language was re-extracted");
        assert!(
            cache_pass(&mut refreshed, &repo).is_empty(),
            "unchanged source reached fixture embedding backend"
        );
    } else {
        let identity = greppy_store::BaseStoreIdentity {
            format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
            canonical_repository_identity: "fixture:cpp-member".into(),
            git_object_format: "sha1".into(),
            base_tree_oid: git(&repo, &["rev-parse", "HEAD^{tree}"]),
            store_schema_version: greppy_store::migrate::CURRENT_VERSION,
            indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
            parser_and_extractor_versions: format!(
                "greppy-parser/extractor-{}",
                env!("CARGO_PKG_VERSION")
            ),
            summary_model_and_prompt_version: "fixture-summary".into(),
            embedding_model: "fixture-cpp-member-cache".into(),
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
        let base_bytes = std::fs::read(&layout.graph).unwrap();
        let task = scratch.path().join("task");
        git(
            &repo,
            &["worktree", "add", "-qb", "task", task.to_str().unwrap()],
        );
        let delta = scratch.path().join("delta");
        let overlay = Some((layout.graph.as_path(), commit.as_str()));
        let (code, out, err) = run_overlay(&legacy, &task, &delta, &["index", "."], overlay);
        assert_eq!(code, 0, "{out}\n{err}");
        assert_callers(
            Path::new(env!("CARGO_BIN_EXE_greppy")),
            &task,
            &delta,
            overlay,
        );
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
            vec!["caller.cpp", "jobs.hpp", "other.cpp", "primary.cpp"],
            "Base repair copied an unaffected language or missed stale C++ extraction"
        );
        drop(conn);
        let visibility = greppy_store::VisibilityIndex::new(paths, []).unwrap();
        let mut composed =
            greppy_store::Store::open_overlay(&layout.graph, &graph(&delta), &visibility).unwrap();
        assert!(
            cache_pass(&mut composed, &task).is_empty(),
            "unchanged content was re-embedded instead of reused from immutable Base"
        );
        drop(composed);
        assert_eq!(
            std::fs::read(&layout.graph).unwrap(),
            base_bytes,
            "normal warm repair mutated immutable Base"
        );
        layout.read_verified_manifest().unwrap();
        assert_eq!(git(&task, &["status", "--porcelain"]), "");
    }
    assert_eq!(git(&repo, &["rev-parse", "HEAD"]), commit);
    assert_eq!(
        std::fs::read_to_string(repo.join("caller.cpp")).unwrap(),
        CALLER
    );
    assert_eq!(git(&repo, &["status", "--porcelain"]), "");
}
