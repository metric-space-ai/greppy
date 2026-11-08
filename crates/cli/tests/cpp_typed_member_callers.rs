//! Actual generic C++ parser -> stored graph -> public caller query.
use std::path::{Path, PathBuf};
use std::process::Command;

const HEADER: &str = "struct PrimaryJob { void begin(); };\nstruct OtherJob { void begin(); };\nstruct Unknown {};\nstruct Pipeline {\n PrimaryJob job_;\n OtherJob other_;\n void run(); void run_other(); void shadow(Unknown& job_);\n void local_shadow(); void unknown(Unknown& value);\n};\n";
const CALLER: &str = "#include \"jobs.hpp\"\nvoid Pipeline::run() { job_.begin(); }\nvoid Pipeline::run_other() { other_.begin(); }\nvoid Pipeline::shadow(Unknown& job_) { job_.begin(); }\nvoid Pipeline::local_shadow() { Unknown job_; job_.begin(); }\nvoid Pipeline::unknown(Unknown& value) { value.begin(); }\n";

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
        5,
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
        for rejected in ["::shadow", "::local_shadow", "::unknown"] {
            assert!(
                !hit.contains(rejected),
                "unproven/shadowed receiver gained a CALLS edge: {value}"
            );
        }
    }
}
