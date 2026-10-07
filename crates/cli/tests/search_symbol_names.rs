//! `search-symbol` answers several names in one call instead of failing with a
//! usage error (0.4.1 benchmark traces: `greppy search-symbol GetRunStateRequest
//! RunState` cost the agent a turn).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(tag: &str) -> (PathBuf, PathBuf, Scratch) {
    let scratch = std::env::temp_dir()
        .canonicalize()
        .unwrap()
        .join(format!("greppy-search-symbol-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let repo = scratch.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("lib.rs"),
        "pub struct AlphaRecord { pub id: u32 }\npub fn beta_value() -> u32 { 2 }\n",
    )
    .unwrap();
    (repo, scratch.join("store"), Scratch(scratch))
}

fn run(args: &[&str], repo: &Path, store: &Path) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn several_names_are_answered_in_one_call() {
    let (repo, store, _scratch) = fixture("several");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index failed: {out}\n{err}");

    let (code, out, err) = run(
        &["search-symbol", "AlphaRecord", "beta_value"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    assert!(!err.contains("does not fit"), "{err}");
    assert!(out.contains("== search-symbol AlphaRecord =="), "{out}");
    assert!(out.contains("== search-symbol beta_value =="), "{out}");
    assert!(out.contains("AlphaRecord"), "{out}");
    assert!(out.contains("beta_value"), "{out}");
}

#[test]
fn one_name_keeps_the_single_answer_shape() {
    let (repo, store, _scratch) = fixture("single");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index failed: {out}\n{err}");

    let (code, out, err) = run(&["search-symbol", "beta_value"], &repo, &store);
    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    assert!(!out.contains("== search-symbol"), "{out}");
    assert!(out.contains("beta_value"), "{out}");
}
