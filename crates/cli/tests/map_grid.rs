//! End-to-end contract for `greppy where-am-i` and its fractal census packs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_repo(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base =
        std::env::temp_dir().join(format!("greppy-cli-where-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join("edit-src/sub")).unwrap();
    std::fs::create_dir_all(repo.join("cli-src")).unwrap();
    std::fs::create_dir_all(repo.join("tests")).unwrap();
    std::fs::create_dir_all(repo.join("src/main/java/com/acme/api")).unwrap();
    std::fs::create_dir_all(repo.join("src/main/java/com/acme/model")).unwrap();

    std::fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"where-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();

    std::fs::write(repo.join("README.md"), "# Where fixture\n\n## Layout\n").unwrap();
    std::fs::write(repo.join("empty.txt"), "orientation prose only\n").unwrap();

    let mut hub = "pub fn hub() -> u32 { 1 }\n".to_string();
    for index in 0..15 {
        hub.push_str(&format!("pub fn caller_{index:02}() -> u32 {{ hub() }}\n"));
    }
    std::fs::write(repo.join("edit-src/a.rs"), hub).unwrap();

    let mut sub = "#[test]\npub fn inline_smoke() {}\n".to_string();
    for index in 0..13 {
        sub.push_str(&format!("pub fn helper_{index:02}() {{}}\n"));
    }
    std::fs::write(repo.join("edit-src/sub/b.rs"), sub).unwrap();
    std::fs::write(
        repo.join("cli-src/main.rs"),
        "fn main() { dispatch(); }\nfn dispatch() {}\n",
    )
    .unwrap();
    std::fs::write(repo.join("tests/smoke.rs"), "#[test]\nfn smoke() {}\n").unwrap();
    std::fs::write(
        repo.join("src/main/java/com/acme/api/Api.java"),
        "class Api { void serve() {} }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/main/java/com/acme/model/Model.java"),
        "class Model {}\n",
    )
    .unwrap();

    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "where@example.invalid"]);
    git(&repo, &["config", "user.name", "Where Fixture"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-qm", "fixture"]);
    (repo, base.join("store"))
}

fn git(repo: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn index(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(repo, store, &["index", "."]);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
}

fn expand_id(line: &str) -> &str {
    line.rsplit_once("greppy expand ")
        .map(|(_, id)| id.trim())
        .filter(|id| id.len() == 16 && id.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .unwrap_or_else(|| panic!("line has no expand id: {line:?}"))
}

#[test]
fn hub_is_one_screen_of_indexed_facts() {
    let (repo, store) = fresh_repo("hub");
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["where-am-i"]);
    assert_eq!(
        code, 0,
        "where-am-i failed\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.lines().count() <= 60,
        "hub exceeded one screen:\n{stdout}"
    );

    let root = stdout.lines().next().expect("root line");
    assert!(root.contains(&repo.display().to_string()), "{root}");
    assert!(root.contains("rust") && root.contains("java"), "{root}");
    assert!(
        root.contains("files") && root.contains("definitions"),
        "{root}"
    );
    assert!(!stdout.contains("(none detected)") && !stdout.contains("try:"));

    let edit = stdout
        .lines()
        .find(|line| line.starts_with("edit-src/"))
        .expect("edit-src hub line");
    assert!(
        edit.contains("2 files") && edit.contains("30 defs"),
        "{edit}"
    );
    assert!(
        edit.contains("— hub — greppy expand "),
        "only referenced non-test code symbols belong in the trio: {edit}"
    );
    expand_id(edit);

    let cli = stdout
        .lines()
        .find(|line| line.starts_with("cli-src/"))
        .expect("cli-src hub line");
    expand_id(cli);

    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("src/main/java/com/acme/")),
        "single-child directory chains must collapse to the first branch: {stdout}"
    );
    assert!(
        !stdout.lines().any(|line| line.contains("  0 defs")),
        "empty census entries must not get rows: {stdout}"
    );
    for path in ["Cargo.toml", "README.md", "empty.txt"] {
        assert!(
            !stdout.lines().any(|line| line.starts_with(path)),
            "non-code entry got a row for {path}: {stdout}"
        );
    }
    assert!(
        stdout.contains("3 further files hold no definitions"),
        "{stdout}"
    );
    assert!(stdout.contains("docs: 1 file, 2 sections"), "{stdout}");
    assert!(stdout.contains("config: 1 file, 4 keys"), "{stdout}");
    assert!(stdout.contains("entry points: cli-src/main.rs"), "{stdout}");
    assert!(
        stdout.contains("tests:")
            && stdout.contains("tests/")
            && stdout.contains("inline #[test] modules"),
        "{stdout}"
    );
}

#[test]
fn json_carries_the_same_code_census_without_empty_expand_rows() {
    let (repo, store) = fresh_repo("json");
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["where-am-i", "--json"]);
    assert_eq!(
        code, 0,
        "where-am-i JSON failed\nstdout={stdout}\nstderr={stderr}"
    );
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(value["schema_version"], "greppy.where-am-i.v1");
    assert_eq!(value["census"]["files"], 9);
    assert_eq!(value["census"]["further_files_without_definitions"], 3);
    assert_eq!(value["census"]["documentation"]["files"], 1);
    assert_eq!(value["census"]["documentation"]["sections"], 2);
    assert_eq!(value["census"]["config"]["files"], 1);
    assert_eq!(value["census"]["config"]["keys"], 4);

    let inventory = value["inventory"].as_array().expect("inventory array");
    assert!(
        inventory
            .iter()
            .all(|entry| entry["definitions"].as_u64().unwrap_or(0) > 0),
        "empty entry in JSON: {stdout}"
    );
    assert!(
        inventory
            .iter()
            .all(|entry| entry["expand_id"].as_str().is_some_and(|id| id.len() == 16)),
        "every visible answer row has an expand id: {stdout}"
    );
    let edit = inventory
        .iter()
        .find(|entry| entry["path"] == "edit-src/")
        .expect("edit-src inventory row");
    assert_eq!(edit["files"], 2);
    assert_eq!(edit["definitions"], 30);
    assert_eq!(edit["most_used"], serde_json::json!(["hub"]));
    assert!(
        value["census"]["definitions"].as_u64().unwrap_or(0)
            > inventory
                .iter()
                .map(|entry| entry["definitions"].as_u64().unwrap_or(0))
                .max()
                .unwrap_or(0),
        "headline must carry the whole code census: {stdout}"
    );
}

#[test]
fn fractal_pack_descends_from_children_to_full_rows() {
    let (repo, store) = fresh_repo("fractal");
    index(&repo, &store);
    let (_, hub, _) = run(&repo, &store, &["where-am-i"]);
    let edit_id = expand_id(
        hub.lines()
            .find(|line| line.starts_with("edit-src/"))
            .expect("edit line"),
    )
    .to_string();

    let (code, children, stderr) = run(&repo, &store, &["expand", &edit_id]);
    assert_eq!(code, 0, "expand failed: {stderr}\n{children}");
    let file_line = children
        .lines()
        .find(|line| line.starts_with("edit-src/a.rs"))
        .expect("file child line");
    assert!(file_line.contains("16 defs"), "{file_line}");
    expand_id(file_line);
    let sub_line = children
        .lines()
        .find(|line| line.starts_with("edit-src/sub/"))
        .expect("directory child line");
    assert!(sub_line.contains("14 defs"), "{sub_line}");
    let sub_id = expand_id(sub_line).to_string();

    let (code, rows, stderr) = run(&repo, &store, &["expand", &sub_id]);
    assert_eq!(code, 0, "nested expand failed: {stderr}\n{rows}");
    let result_rows = rows
        .lines()
        .filter(|line| line.starts_with("edit-src/sub/b.rs:"))
        .collect::<Vec<_>>();
    assert_eq!(result_rows.len(), 14, "full census expected:\n{rows}");
    assert!(
        result_rows
            .iter()
            .all(|line| line.split_whitespace().count() >= 3),
        "rows must be file:line name kind: {rows}"
    );
    assert!(
        result_rows
            .iter()
            .any(|line| line.contains("inline_smoke") && line.ends_with("test")),
        "test definitions carry the trailing marker: {rows}"
    );
}

#[test]
fn large_file_inventory_pages_in_twenty_five_row_packs() {
    let (repo, store) = fresh_repo("paging");
    let mut source = String::new();
    for index in 0..31 {
        source.push_str(&format!("pub fn large_{index:02}() {{}}\n"));
    }
    std::fs::write(repo.join("large.rs"), source).unwrap();
    index(&repo, &store);

    let (_, hub, _) = run(&repo, &store, &["where-am-i"]);
    let id = expand_id(
        hub.lines()
            .find(|line| line.starts_with("large.rs"))
            .expect("large file line"),
    )
    .to_string();
    let (code, first, stderr) = run(&repo, &store, &["expand", &id]);
    assert_eq!(code, 0, "first page failed: {stderr}\n{first}");
    assert_eq!(
        first
            .lines()
            .filter(|line| line.starts_with("large.rs:"))
            .count(),
        25,
        "first page:\n{first}"
    );
    let next = first
        .lines()
        .find(|line| line.contains("6 defs — greppy expand "))
        .expect("next-page offer");
    let next_id = expand_id(next).to_string();
    let (code, second, stderr) = run(&repo, &store, &["expand", &next_id]);
    assert_eq!(code, 0, "second page failed: {stderr}\n{second}");
    assert_eq!(
        second
            .lines()
            .filter(|line| line.starts_with("large.rs:"))
            .count(),
        6,
        "second page:\n{second}"
    );
}

#[test]
fn inventory_pack_relocates_by_hash_or_refuses_drift() {
    let (repo, store) = fresh_repo("drift");
    std::fs::create_dir_all(repo.join("module")).unwrap();
    std::fs::write(repo.join("module/item.rs"), "pub fn movable() {}\n").unwrap();
    index(&repo, &store);

    let (_, hub, _) = run(&repo, &store, &["where-am-i"]);
    let id = expand_id(
        hub.lines()
            .find(|line| line.starts_with("module/"))
            .expect("module line"),
    )
    .to_string();

    std::fs::rename(repo.join("module"), repo.join("renamed")).unwrap();
    let (code, relocated, stderr) = run(&repo, &store, &["expand", &id]);
    assert_eq!(
        code, 0,
        "hash-preserving rename must relocate: {stderr}\n{relocated}"
    );
    assert!(
        relocated.contains("renamed/item.rs:1  movable"),
        "{relocated}"
    );

    std::fs::write(repo.join("renamed/item.rs"), "pub fn changed() {}\n").unwrap();
    let (code, refused, stderr) = run(&repo, &store, &["expand", &id]);
    assert_eq!(code, 1, "content drift must refuse: {stderr}\n{refused}");
    assert!(
        refused.contains("inventory changed since this pack was created"),
        "{refused}"
    );
}

#[test]
fn dead_orientation_verbs_are_refused_before_grep() {
    let (repo, store) = fresh_repo("dead-verbs");
    std::fs::write(
        repo.join("grep-bait.txt"),
        "map\noutline\nchanges\nverify\n",
    )
    .unwrap();
    for args in [
        vec!["map"],
        vec!["outline", "x.rs"],
        vec!["changes"],
        vec!["verify", "--", "true"],
    ] {
        let verb = args[0];
        let (code, stdout, stderr) = run(&repo, &store, &args);
        assert_eq!(code, 64, "{args:?}\nstdout={stdout}\nstderr={stderr}");
        assert!(
            stdout.contains(&format!("unrecognized subcommand '{verb}'")),
            "{args:?}: {stdout}"
        );
        assert!(
            !stdout.contains("grep-bait.txt"),
            "dead verb became grep: {stdout}"
        );
    }
}
