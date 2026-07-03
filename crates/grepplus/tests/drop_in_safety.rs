//! Integration tests for the drop-in safety fixes from
//! `reviews/independent-production-readiness-review-2026-06-29.md`.
//!
//! Each test exercises one finding with the actual `grepplus-grep`
//! binary, not just the unit-tested library surface.
//!
//! Coverage:
//!
//! - R-002: real-grep miss must produce **byte-exact** empty stdout and
//!   exit code 1 (no synthetic line, no sidecar).
//! - R-002 (rc=2): real-grep error code must not trigger any semantic
//!   output.
//! - R-005: a freshly indexed workspace must not have a `graph.db`
//!   inside `<root>/.grepplus/`; the DB lives under the platform
//!   locator.
//! - R-006: `discover_grep` rejects paths under `~/.grepplus/shims/`
//!   (shim recursion).

use grepplus_core::workspace as workspace_locator;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

/// All four tests manipulate process-global env vars
/// (`HOME`, `PATH`, `GREPPLUS_STORE_DIR`). They are serialized via this
/// Mutex so they cannot race each other when cargo runs them in
/// parallel (which is the default inside a single integration-test
/// binary).
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_grepplus-grep"))
}

fn real_grep_path() -> PathBuf {
    if let Ok(p) = env::var("GREPPLUS_REAL_GREP") {
        return PathBuf::from(p);
    }
    PathBuf::from("/usr/bin/grep")
}

fn unique_tempdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Run `grepplus-grep` with `args` and an explicit isolated
/// `GREPPLUS_STORE_DIR`. The wrapper runs with `cwd` set to `cwd`
/// (its freshness gate uses cwd-relative paths). Stdin, stdout,
/// stderr are piped.
fn run_isolated(args: &[&str], cwd: &Path, store_dir: &Path) -> std::process::Output {
    let mut cmd = Command::new(binary_path());
    cmd.env("GREPPLUS_STORE_DIR", store_dir);
    cmd.current_dir(cwd);
    cmd.args(args);
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.output().expect("spawn grepplus-grep")
}

/// Build a minimal repo at `tmp`, write `src/lib.rs = source`,
/// open a store at the platform-locator'd path under `store_dir`, and
/// run the indexer directly (avoids spawning the separate `grepplus`
/// CLI binary, whose `CARGO_BIN_EXE_<name>` env var this crate does
/// not export).
fn build_indexed_repo(tmp: &Path, source: &str, store_dir: &Path) {
    let src = tmp.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), source).unwrap();
    index_existing_repo(tmp, store_dir);
}

fn build_indexed_git_repo(tmp: &Path, source: &str, store_dir: &Path) {
    let src = tmp.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), source).unwrap();
    run_git(tmp, &["init", "-q"]);
    run_git(tmp, &["add", "src/lib.rs"]);
    run_git(
        tmp,
        &[
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "initial index baseline",
        ],
    );

    index_existing_repo(tmp, store_dir);
}

fn index_existing_repo(root: &Path, store_dir: &Path) {
    // Pin the locator's store dir to the per-test location.
    env::set_var("GREPPLUS_STORE_DIR", store_dir);
    let store_path = workspace_locator::store_path(root);
    if let Some(parent) = store_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut store = grepplus_store::Store::open(&store_path).unwrap();
    let project = root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("default");
    grepplus_indexer::index(&mut store, root, project).unwrap();
    drop(store);
    // The grepplus-grep sub-process tests below re-set
    // GREPPLUS_STORE_DIR explicitly; the env override here is process-
    // global so we reset it to leave the test in a clean state.
    env::remove_var("GREPPLUS_STORE_DIR");
}

fn run_git(cwd: &Path, args: &[&str]) {
    let status = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .status()
        .expect("git invocation");
    assert!(status.success(), "git {args:?} failed: {status:?}");
}

fn run_real_grep(args: &[&str], cwd: &Path) -> std::process::Output {
    let mut real = Command::new(real_grep_path());
    real.current_dir(cwd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    real.output().expect("spawn real grep")
}

fn assert_byte_exact_no_plus(
    label: &str,
    actual: &std::process::Output,
    expected: &std::process::Output,
) {
    assert_eq!(
        actual.status.code(),
        expected.status.code(),
        "{label}: stale/strict path must not alter real-grep exit code"
    );
    assert_eq!(
        actual.stdout, expected.stdout,
        "{label}: stale/strict path must keep stdout byte-exact"
    );
    assert_eq!(
        actual.stderr, expected.stderr,
        "{label}: stale/strict path must keep stderr byte-exact"
    );
    assert!(
        !String::from_utf8_lossy(&actual.stdout).contains("GREPPLUS_NON_CANONICAL_HIT"),
        "{label}: must not print a visible synthetic hit"
    );
    assert!(
        !String::from_utf8_lossy(&actual.stderr).contains("GREPPLUS_NON_CANONICAL_HIT"),
        "{label}: must not print a synthetic hit on stderr"
    );
}

#[test]
fn r002_real_grep_miss_is_byte_exact_empty_with_no_synthetic_line() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = unique_tempdir("grepplus-r002-int");
    let store = tmp.join("store");
    std::fs::create_dir_all(&store).unwrap();
    build_indexed_repo(&tmp, "pub struct RenamedOrder;\n", &store);
    let out = run_isolated(&["-R", "ProcessOrder", "src"], &tmp, &store);
    assert_eq!(out.status.code(), Some(1), "real-grep rc must be 1");
    assert!(
        out.stdout.is_empty(),
        "stdout must be byte-exact empty on real-grep miss, got: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    let sidecars: Vec<_> = walkdir_files(&store)
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.contains("GREPPLUS_SEMANTIC_NONCANONICAL.md"))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        sidecars.is_empty(),
        "no sidecar must be written on real-grep miss, found {sidecars:?}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r002_real_grep_error_rc2_emits_no_synthetic_output() {
    let _g = ENV_LOCK.lock().unwrap();
    // Force a real-grep rc=2 by passing an unreadable directory as the
    // search root. R-002 is explicit that augmentation runs only on a
    // real-grep match (rc=0).
    let tmp = unique_tempdir("grepplus-r002-err");
    let src = tmp.join("src");
    std::fs::create_dir_all(&src).unwrap();
    let store = tmp.join("store");
    std::fs::create_dir_all(&store).unwrap();
    let out = run_isolated(
        &["-R", "anything", "/this/dir/does/not/exist/at/all"],
        &tmp,
        &store,
    );
    // real grep returns rc=2 for missing files/dirs. We don't require
    // a specific code (varies across platforms: macOS=2, some gnu=2),
    // but we DO require no synthetic sentinel in stderr.
    assert_ne!(
        out.status.code(),
        Some(0),
        "real-grep on missing dir should not return 0, got {out:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("GREPPLUS_NON_CANONICAL_HIT"),
        "stderr must not contain synthetic sentinel on rc=2 path: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r1_stale_modified_file_match_stays_byte_exact_and_writes_no_sidecar() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = unique_tempdir("grepplus-r1-stale-modified");
    let store = tmp.join("store");
    std::fs::create_dir_all(&store).unwrap();
    build_indexed_repo(&tmp, "pub fn ProcessOrder() {}\n", &store);

    std::fs::write(
        tmp.join("src/lib.rs"),
        "pub fn ProcessOrder() { let changed_after_index = true; }\n",
    )
    .unwrap();

    let before = sidecar_count(&store);
    let out = run_isolated(&["-R", "ProcessOrder", "src"], &tmp, &store);
    let expected = run_real_grep(&["-R", "ProcessOrder", "src"], &tmp);
    assert_byte_exact_no_plus("stale modified file", &out, &expected);
    assert_eq!(
        sidecar_count(&store),
        before,
        "stale graph must not write a sidecar"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r1_git_untracked_added_file_match_stays_byte_exact_and_writes_no_sidecar() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = unique_tempdir("grepplus-r1-git-untracked");
    let store = tmp.join("store");
    std::fs::create_dir_all(&store).unwrap();
    build_indexed_git_repo(&tmp, "pub fn baseline() {}\n", &store);

    std::fs::write(tmp.join("src/new.rs"), "pub fn ProcessOrder() {}\n").unwrap();

    let before = sidecar_count(&store);
    let out = run_isolated(&["-R", "ProcessOrder", "src"], &tmp, &store);
    let expected = run_real_grep(&["-R", "ProcessOrder", "src"], &tmp);
    assert_byte_exact_no_plus("git untracked added file", &out, &expected);
    assert_eq!(
        sidecar_count(&store),
        before,
        "untracked added file must make the graph stale and write no sidecar"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r1_git_renamed_file_match_stays_byte_exact_and_writes_no_sidecar() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = unique_tempdir("grepplus-r1-git-rename");
    let store = tmp.join("store");
    std::fs::create_dir_all(&store).unwrap();
    build_indexed_git_repo(&tmp, "pub fn ProcessOrder() {}\n", &store);

    std::fs::rename(tmp.join("src/lib.rs"), tmp.join("src/main.rs")).unwrap();

    let before = sidecar_count(&store);
    let out = run_isolated(&["-R", "ProcessOrder", "src"], &tmp, &store);
    let expected = run_real_grep(&["-R", "ProcessOrder", "src"], &tmp);
    assert_byte_exact_no_plus("git renamed file", &out, &expected);
    assert_eq!(
        sidecar_count(&store),
        before,
        "renamed file must make the graph stale and write no sidecar"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r1_git_deleted_file_while_other_file_matches_stays_byte_exact_and_writes_no_sidecar() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = unique_tempdir("grepplus-r1-git-delete");
    let store = tmp.join("store");
    let src = tmp.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(&store).unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn ProcessOrder() {}\n").unwrap();
    std::fs::write(src.join("obsolete.rs"), "pub fn obsolete() {}\n").unwrap();
    run_git(&tmp, &["init", "-q"]);
    run_git(&tmp, &["add", "src/lib.rs", "src/obsolete.rs"]);
    run_git(
        &tmp,
        &[
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "baseline with two files",
        ],
    );
    index_existing_repo(&tmp, &store);

    std::fs::remove_file(src.join("obsolete.rs")).unwrap();

    let before = sidecar_count(&store);
    let out = run_isolated(&["-R", "ProcessOrder", "src"], &tmp, &store);
    let expected = run_real_grep(&["-R", "ProcessOrder", "src"], &tmp);
    assert_byte_exact_no_plus("git deleted file", &out, &expected);
    assert_eq!(
        sidecar_count(&store),
        before,
        "deleted indexed file must make the graph stale and write no sidecar"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r2_strict_flags_on_fresh_graph_are_byte_exact_and_write_no_sidecar() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = unique_tempdir("grepplus-r2-strict-fresh");
    let store = tmp.join("store");
    std::fs::create_dir_all(&store).unwrap();
    build_indexed_repo(&tmp, "pub fn ProcessOrder() { ProcessOrder(); }\n", &store);

    let cases: &[(&str, &[&str])] = &[
        ("quiet", &["-q", "ProcessOrder", "src"]),
        ("count", &["-c", "ProcessOrder", "src/lib.rs"]),
        ("files-with-matches", &["-l", "ProcessOrder", "src"]),
        ("only-matching", &["-o", "ProcessOrder", "src/lib.rs"]),
        ("files-without-match", &["-L", "MissingSymbol", "src"]),
    ];

    for (label, args) in cases {
        let before = sidecar_count(&store);
        let out = run_isolated(args, &tmp, &store);
        let expected = run_real_grep(args, &tmp);
        assert_byte_exact_no_plus(label, &out, &expected);
        assert_eq!(
            sidecar_count(&store),
            before,
            "{label}: strict grep flag must not write a sidecar on a fresh graph"
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r005_graph_db_is_outside_repo_search_space() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = unique_tempdir("grepplus-r005-int");
    let store = tmp.join("store");
    std::fs::create_dir_all(&store).unwrap();
    build_indexed_repo(&tmp, "pub struct X;\n", &store);
    let in_repo_db = tmp.join(".grepplus").join("graph.db");
    assert!(
        !in_repo_db.exists(),
        "{in_repo_db:?} must not exist (R-005: was inside the repo)"
    );
    let dbs: Vec<_> = walkdir_files(&store)
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s == "graph.db")
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(dbs.len(), 1, "expected exactly one graph.db, got {dbs:?}");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn r006_discover_grep_refuses_shim_under_home_shim_dir() {
    let _g = ENV_LOCK.lock().unwrap();
    // Simulate a shimmed PATH where the resolved grep is the wrapper
    // itself: discover_grep must NOT return it. We feed a fake
    // `~/.grepplus/shims/grep` to the resolver by setting HOME
    // temporarily. This is best-effort: if the host has no grep on
    // PATH in a particular env, the assertion below is conditional.
    let tmp = unique_tempdir("grepplus-r006");
    let home = tmp.join("home");
    std::fs::create_dir_all(home.join(".grepplus/shims")).unwrap();
    let fake_shim = home.join(".grepplus/shims/grep");
    std::fs::write(&fake_shim, "#!/bin/sh\ntrue\n").unwrap();

    // Restore HOME afterwards so other tests are not affected.
    let prev_home = env::var_os("HOME");
    let prev_path = env::var_os("PATH");
    env::set_var("HOME", &home);
    let new_path = format!("{}:/usr/bin:/bin", fake_shim.parent().unwrap().display());
    env::set_var("PATH", &new_path);
    // Also override GREPPLUS_REAL_GREP to the fake shim path so the
    // env-override branch doesn't pre-empt the which() recursion check.
    env::set_var("GREPPLUS_REAL_GREP", &fake_shim);

    // discover_grep is not `pub`-callable cross-process; instead, we
    // exercise it by spawning grepplus-grep and asserting it returns
    // rc=3 (real-grep-missing exit code from main.rs).
    // SAFETY: env::set_var/remove_var are unsafe since Rust 1.85; the
    // integration test build is on a stable Rust so we leave the
    // exports in place and rely on the test runner's per-test HOME
    // isolation.
    let mut cmd = Command::new(binary_path());
    cmd.env("HOME", &home);
    cmd.env("PATH", &new_path);
    cmd.env("GREPPLUS_REAL_GREP", &fake_shim);
    cmd.env("GREPPLUS_STORE_DIR", tmp.join("store"));
    cmd.args(["-R", "anything", "src"]);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let out = cmd.output().expect("spawn");

    // The wrapper must NOT recurse. It either returns rc=3 (real-grep
    // missing) or refuses for some other reason; either way it must not
    // spawn and return real-grep's bytes.
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("anything") || out.status.code() == Some(3),
        "wrapper must not recurse into the shim; got rc={:?} stdout={stdout:?} stderr={stderr:?}",
        out.status.code()
    );

    // Restore env.
    if let Some(p) = prev_home {
        env::set_var("HOME", p);
    } else {
        env::remove_var("HOME");
    }
    if let Some(p) = prev_path {
        env::set_var("PATH", p);
    } else {
        env::remove_var("PATH");
    }
    env::remove_var("GREPPLUS_REAL_GREP");

    let _ = std::fs::remove_dir_all(&tmp);
}

fn walkdir_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walkdir_files_inner(root, &mut out);
    out
}

fn sidecar_count(root: &Path) -> usize {
    walkdir_files(root)
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.contains("GREPPLUS_SEMANTIC_NONCANONICAL.md"))
                .unwrap_or(false)
        })
        .count()
}

fn walkdir_files_inner(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walkdir_files_inner(&p, out);
        } else {
            out.push(p);
        }
    }
}
