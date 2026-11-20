//! Queries keep answering when host admission defers index work.
//!
//! A structural refresh of a published index does not wait for the host's
//! build gate at all: it runs in greppy's own refresh lane. When that lane is
//! full, or turned off and the gate defers, the query answers from the
//! published index and says so. A missing index is still not invented.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "greppy-deferred-refresh-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str], repo: &Path, store: &Path, gate: Option<&Path>) -> (i32, String, String) {
    run_with(args, repo, store, gate, &[])
}

fn run_with(
    args: &[&str],
    repo: &Path,
    store: &Path,
    gate: Option<&Path>,
    env: &[(&str, &std::ffi::OsStr)],
) -> (i32, String, String) {
    let mut command = Command::new(bin());
    command
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .envs(env.iter().copied());
    if let Some(gate) = gate {
        command.env("GREPPY_HEAVY_GATE", gate);
    }
    let out = command.output().expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// An admission gate that refuses every job the way the host gate does when
/// the system disk has less than 20 GiB free.
fn refusing_gate(dir: &Path) -> PathBuf {
    let gate = dir.join("refusing-gate.py");
    std::fs::write(
        &gate,
        "import sys\nsys.stderr.write('System disk has less than 20 GiB free')\nsys.exit(75)\n",
    )
    .unwrap();
    gate
}

/// An indexed two-file repository whose `util.rs` then gains a second caller
/// of `helper`.
fn indexed_then_drifted(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = scratch(tag);
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "mod util;\npub fn caller() { util::helper(); }\n",
    )
    .unwrap();
    std::fs::write(repo.join("src/util.rs"), "pub fn helper() {}\n").unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store, None);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    std::fs::write(
        repo.join("src/util.rs"),
        "pub fn helper() {}\npub fn added() { helper(); }\n",
    )
    .unwrap();
    (root, repo, store)
}

#[test]
fn a_structural_refresh_does_not_wait_for_the_build_gate() {
    let (root, repo, store) = indexed_then_drifted("lane");
    // The host's build gate refuses every job; the refresh never asks it.
    let gate = refusing_gate(&root);
    let lane = root.join("lane-tmp");
    std::fs::create_dir_all(&lane).unwrap();
    let (code, out, err) = run_with(
        &["who-calls", "helper"],
        &repo,
        &store,
        Some(&gate),
        &[("TMPDIR", lane.as_os_str())],
    );
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(
        out.contains("src/util.rs:2  added"),
        "the new caller needs the refreshed index: {out}"
    );
    assert!(
        !err.contains("answering from the existing index"),
        "the refresh ran, nothing is stale: {err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(unix)]
#[test]
fn a_full_refresh_lane_answers_from_the_existing_index() {
    let (root, repo, store) = indexed_then_drifted("lane-full");
    // Both refresh slots of this (private) lane are held by someone else.
    let lane = root.join("lane-tmp");
    let slots = lane.join(format!("greppy-refresh-lane-{}", unsafe {
        libc::geteuid()
    }));
    let _held = (0..2)
        .map(|slot| {
            greppy_core::cache::acquire_named_lock_in(
                &slots,
                &format!("refresh-slot-{slot}"),
                greppy_core::cache::LockMode::Exclusive,
                true,
            )
            .unwrap()
            .expect("free slot")
        })
        .collect::<Vec<_>>();
    let (code, out, err) = run_with(
        &["who-calls", "helper"],
        &repo,
        &store,
        None,
        &[("TMPDIR", lane.as_os_str())],
    );
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(out.contains("src/lib.rs:2  caller"), "{out}");
    assert!(
        err.contains("answering from the existing index") && err.contains("refresh slots"),
        "the answer must say it predates the change and why: {err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_deferred_refresh_answers_from_the_existing_index() {
    let (root, repo, store) = indexed_then_drifted("existing");
    // Without a refresh lane every refresh is a heavy job; the gate defers it.
    let gate = refusing_gate(&root);
    let no_lane = [("GREPPY_REFRESH_SLOTS", std::ffi::OsStr::new("0"))];

    let (code, out, err) = run_with(
        &["who-calls", "helper"],
        &repo,
        &store,
        Some(&gate),
        &no_lane,
    );
    assert_eq!(code, 0, "who-calls: stderr={err}\nstdout={out}");
    assert!(out.contains("src/lib.rs:2  caller"), "{out}");
    assert!(
        err.contains("answering from the existing index") && err.contains("20 GiB"),
        "the answer must say it predates the change and why: {err}"
    );

    for args in [&["stats"][..], &["where-am-i"][..]] {
        let (code, out, err) = run_with(args, &repo, &store, Some(&gate), &no_lane);
        assert_eq!(code, 0, "{args:?}: stderr={err}\nstdout={out}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_admission_gate_learns_where_the_index_job_writes() {
    let root = scratch("writes");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn caller() {}\n").unwrap();
    let store = root.join("store");
    // A gate that records what the job declares and then defers it, as the
    // host gate does for a full system disk.
    let record = root.join("declared-writes.txt");
    let gate = root.join("recording-gate.py");
    std::fs::write(
        &gate,
        format!(
            "import os, sys\nopen({:?}, 'w').write(os.environ.get('DEV_HEAVY_WRITES', ''))\nsys.exit(75)\n",
            record.display().to_string()
        ),
    )
    .unwrap();
    let (code, out, err) = run(&["search-symbol", "caller"], &repo, &store, Some(&gate));
    assert_ne!(code, 0, "no index exists and none may be built: {out}{err}");
    let declared = std::fs::read_to_string(&record).expect("the gate ran");
    // macOS reports a temp directory under /private/var once canonicalized.
    let canonical_store = std::fs::canonicalize(&store).unwrap_or_else(|_| store.clone());
    assert!(
        std::env::split_paths(&declared)
            .any(|path| path.starts_with(&store) || path.starts_with(&canonical_store)),
        "the store directory must be declared: {declared:?} (store {})",
        store.display()
    );
    let _ = std::fs::remove_dir_all(&root);
}
