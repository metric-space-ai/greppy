//! Regression coverage for Postel-style navigation inputs and miss guidance.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "greppy-cli-nav-postel-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn make_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("src/inside")).unwrap();
    std::fs::create_dir_all(repo.join("tests")).unwrap();

    std::fs::write(
        repo.join("src/api.rs"),
        r#"
pub fn target() {}

#[allow(non_snake_case)]
pub fn startsWith() {}

pub trait Encode {
    fn serialize(&self) -> u32;
}

pub struct Option;

impl Encode for Option {
    fn serialize(&self) -> u32 {
        7
    }
}
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("src/inside/caller.rs"),
        "pub fn caller_inside() { crate::api::target(); }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("tests/outside.rs"),
        "pub fn caller_outside() { crate::api::target(); }\n",
    )
    .unwrap();

    (repo, root.join("store"))
}

fn run(args: &[&str], cwd: &Path, store: &Path) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn indexed_repo(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_repo(tag);
    let (code, stdout, stderr) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
    (repo, store)
}

#[test]
fn who_calls_path_filtering_is_a_flag_and_empty_scope_is_an_answer() {
    let (repo, store) = indexed_repo("path-filter");

    // A positional PATH is refused: positional arguments are symbols now
    // (`who-calls A B C` answers for several symbols at once), so a path in
    // that position is a usage error that teaches the flag spelling.
    let (code, stdout, stderr) = run(&["who-calls", "target", "src/inside"], &repo, &store);
    assert_eq!(code, 64, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stderr.contains("`src/inside` is a path, but a positional argument is always a symbol"),
        "stderr={stderr}"
    );
    assert!(stderr.contains("--path src/inside"), "stderr={stderr}");

    // The flag filters to the callers under it.
    let (code, stdout, stderr) = run(
        &["who-calls", "target", "--path", "src/inside"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert_eq!(
        stdout, "src/inside/caller.rs:1  caller_inside\n",
        "stdout={stdout}"
    );

    // An existing scope with no callers in it is an answer, exit 0.
    let (code, stdout, stderr) = run(
        &["who-calls", "target", "--path", "src/api.rs"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert_eq!(
        stdout, "no callers under path filter: src/api.rs\n",
        "stdout={stdout}"
    );

    // A filter path that does not exist is a usage error, not an empty answer.
    let (code, stdout, stderr) = run(
        &["who-calls", "target", "--path", "does/not/exist"],
        &repo,
        &store,
    );
    assert_eq!(code, 64, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stderr.contains("--path `does/not/exist` does not exist under"),
        "stderr={stderr}"
    );
}

#[test]
fn root_file_and_subdirectory_misuse_teaches_the_real_root_and_corrected_command() {
    let (repo, store) = indexed_repo("root-guidance");
    let real_root = repo.canonicalize().unwrap().to_string_lossy().into_owned();

    for wrong_root in ["src/api.rs", "src/inside"] {
        let (code, stdout, stderr) = run(
            &["who-calls", "target", "--root", wrong_root],
            &repo,
            &store,
        );
        assert_ne!(code, 0, "stdout={stdout}\nstderr={stderr}");
        let combined = format!("{stdout}\n{stderr}");
        assert!(
            combined.contains("--root selects the indexed repository root"),
            "combined={combined}"
        );
        assert!(combined.contains(&real_root), "combined={combined}");
        assert!(
            combined.contains(&format!(
                "greppy who-calls target {wrong_root} --root {real_root}"
            )),
            "combined={combined}"
        );
    }
}

#[test]
fn symbol_miss_lists_similar_names_and_nothing_else() {
    let (repo, store) = indexed_repo("miss-guidance");
    let (code, stdout, stderr) = run(&["who-calls", "startswith"], &repo, &store);
    assert_eq!(code, 1, "stderr={stderr}\nstdout={stdout}");
    // The 0.3.0 miss is the bare statement plus the similar-names block —
    // the case/underscore normalization tier finds `startsWith`. Law 1
    // deletes the `try: greppy …` discovery instructions: the commands are
    // documented once in the prompt, never repeated under a miss.
    assert_eq!(
        stdout, "no symbol `startswith`\n\nsimilar names:\nsrc/api.rs:5  startsWith\n",
        "stdout={stdout}"
    );
    assert!(!stdout.contains("try:"), "stdout={stdout}");
}

#[test]
fn type_method_query_resolves_rust_trait_impl_method() {
    let (repo, store) = indexed_repo("trait-impl-method");

    let (code, stdout, stderr) = run(&["brief", "Option"], &repo, &store);
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert!(stdout.contains("Option"), "stdout={stdout}");

    let (code, stdout, stderr) = run(&["read", "Option::serialize"], &repo, &store);
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert!(stdout.contains("fn serialize(&self)"), "stdout={stdout}");
    assert!(stdout.contains("src/api.rs"), "stdout={stdout}");
}

#[test]
fn limit_max_aliases_hold_and_paths_are_flag_only() {
    let (repo, store) = indexed_repo("aliases");

    let limit = run(&["search-pattern", "pub", "--limit", "1"], &repo, &store);
    let max = run(&["search-pattern", "pub", "--max", "1"], &repo, &store);
    assert_eq!(limit.0, 0, "search-pattern must answer: {limit:?}");
    assert_eq!(limit, max, "--limit and --max must be exact aliases");

    // The positional-PATH filter is gone: on the navigation verbs a
    // positional argument is always a symbol, so a path there is refused and
    // the refusal teaches `--path`; the flag performs the filter.
    for (command, query, path, want_row) in [
        (
            "who-calls",
            "target",
            "src/inside",
            "src/inside/caller.rs:1  caller_inside\n",
        ),
        (
            "callees",
            "caller_inside",
            "src/api.rs",
            "src/api.rs:2  target\n",
        ),
    ] {
        let (code, _stdout, stderr) = run(&[command, query, path], &repo, &store);
        assert_eq!(
            code, 64,
            "{command}: a positional path must be a usage error; stderr={stderr}"
        );
        assert!(
            stderr.contains(&format!(
                "`{path}` is a path, but a positional argument is always a symbol"
            )) && stderr.contains(&format!("--path {path}")),
            "{command}: the refusal must teach the flag spelling; stderr={stderr}"
        );
        let (code, stdout, stderr) = run(&[command, query, "--path", path], &repo, &store);
        assert_eq!(code, 0, "{command}: --path must filter; stderr={stderr}");
        assert_eq!(stdout, want_row, "{command}: --path row differs");
    }

    let positional = run(&["read", "target"], &repo, &store);
    assert_eq!(
        positional.0, 0,
        "stdout={}\nstderr={}",
        positional.1, positional.2
    );
    let retired = run(&["read", "--symbol", "target"], &repo, &store);
    // read IS a subcommand; a retired flag on it is a usage error, not a
    // silent grep passthrough.
    assert_eq!(retired.0, 64, "retired --symbol must be a usage error");
}

#[test]
fn global_output_flags_work_before_and_after_subcommand() {
    let (repo, store) = indexed_repo("global-flags");

    for flag in ["--json", "--diagnostics", "--code", "--all"] {
        let before = run(&[flag, "who-calls", "target"], &repo, &store);
        let after = run(&["who-calls", "target", flag], &repo, &store);
        if flag == "--json" {
            assert_eq!(before.0, after.0);
            assert_eq!(before.2, after.2);
            let before: serde_json::Value = serde_json::from_str(&before.1).unwrap();
            let after: serde_json::Value = serde_json::from_str(&after.1).unwrap();
            assert_eq!(before["command"], after["command"]);
            assert_eq!(before["all"], after["all"]);
            assert_eq!(before["hits"], after["hits"]);
            assert_eq!(before["shown"], after["shown"]);
        } else {
            assert_eq!(before, after, "global flag ordering differs for {flag}");
        }
    }

    let before = run(
        &["--root", ".", "search-pattern", "target", "--limit", "1"],
        &repo,
        &store,
    );
    let after = run(
        &["search-pattern", "target", "--limit", "1", "--root", "."],
        &repo,
        &store,
    );
    assert_eq!(before.0, 0, "search-pattern must answer: {before:?}");
    assert_eq!(before, after, "--root ordering differs");
}

#[test]
fn unknown_flag_suggests_a_complete_corrected_invocation() {
    let (repo, store) = indexed_repo("unknown-flag");
    let (code, stdout, stderr) = run(
        &["search-pattern", "target", "--jsoon", "--limit", "1"],
        &repo,
        &store,
    );
    assert_eq!(code, 64, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains(" search-pattern target --json --limit 1"),
        "stdout={stdout}"
    );
}
