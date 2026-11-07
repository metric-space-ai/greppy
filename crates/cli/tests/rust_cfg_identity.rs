//! Real Rust extraction -> persisted graph -> public read, without target evaluation.
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = "#[cfg(unix)]\npub fn platform_token() -> u8 { unix_leaf() }\n#[cfg(not(unix))]\npub fn platform_token() -> u8 { other_leaf() }\nfn unix_leaf() -> u8 { 1 }\nfn other_leaf() -> u8 { 2 }\nfn caller() -> u8 { platform_token() }\n";

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_FRESHNESS_TTL_MS", "60000")
        .env_remove("GREPPY_AUTO_REINDEX")
        .env_remove("GREPPY_AGENT_STORE_MODE")
        .env_remove("GREPPY_AGENT_BASE_STORE")
        .env_remove("GREPPY_AGENT_BASE_COMMIT")
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
}
