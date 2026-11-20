//! A fresh clone of an indexed repository answers its first query without a
//! first index from the sources: it starts from the other clone's index and
//! the ordinary refresh brings that to its own files. That is refresh-sized
//! work, so it does not wait for the host's build gate either.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn scratch(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("greppy-sibling-seed-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Checkout roots are compared as recorded: use the resolved spelling.
    dir.canonicalize().unwrap()
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.invalid"])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .expect("spawn git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

struct Host {
    root: PathBuf,
    store: PathBuf,
    /// Refuses every job, as the host gate does while a build holds it.
    gate: PathBuf,
    lane: PathBuf,
}

impl Host {
    fn new(tag: &str) -> Self {
        let root = scratch(tag);
        let gate = root.join("refusing-gate.py");
        std::fs::write(
            &gate,
            "import sys\nsys.stderr.write('another owner holds the heavy-job lease')\nsys.exit(75)\n",
        )
        .unwrap();
        let lane = root.join("lane-tmp");
        std::fs::create_dir_all(&lane).unwrap();
        Self {
            store: root.join("store"),
            root,
            gate,
            lane,
        }
    }

    fn run(&self, args: &[&str], repo: &Path, env: &[(&str, &str)]) -> (i32, String, String) {
        let out = Command::new(bin())
            .args(args)
            .current_dir(repo)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_HEAVY_GATE", &self.gate)
            .env("TMPDIR", &self.lane)
            .envs(env.iter().copied())
            .output()
            .expect("spawn greppy");
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// A committed repository `name`, indexed without any gate.
    fn indexed_repository(&self, name: &str) -> PathBuf {
        let repo = self.root.join(name);
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(
            repo.join("src/lib.rs"),
            "mod util;\npub fn caller() { util::helper(); }\n",
        )
        .unwrap();
        std::fs::write(repo.join("src/util.rs"), "pub fn helper() {}\n").unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-q", "-m", "first"]);
        let out = Command::new(bin())
            .args(["index", "."])
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .output()
            .expect("spawn greppy");
        assert!(
            out.status.success(),
            "index {name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        repo
    }

    /// A clone of `origin` under another directory name, one commit ahead:
    /// a second caller of `helper`, and `caller` no longer calls it.
    fn clone_ahead(&self, origin: &Path, name: &str) -> PathBuf {
        let clone = self.root.join(name);
        git(
            &self.root,
            &[
                "clone",
                "-q",
                origin.to_str().unwrap(),
                clone.to_str().unwrap(),
            ],
        );
        std::fs::write(
            clone.join("src/lib.rs"),
            "mod util;\npub fn caller() {}\npub fn second() { util::helper(); }\n",
        )
        .unwrap();
        git(&clone, &["commit", "-q", "-am", "second"]);
        clone
    }
}

#[test]
fn a_fresh_clone_answers_from_a_sibling_clones_index_without_the_build_gate() {
    let host = Host::new("clone");
    let alpha = host.indexed_repository("alpha");
    let beta = host.clone_ahead(&alpha, "beta");

    // The build gate refuses everything: a first index from the sources
    // could not start.
    let (code, out, err) = host.run(&["who-calls", "helper"], &beta, &[]);
    assert_eq!(
        code, 0,
        "who-calls in the clone: stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("src/lib.rs:3  second") && !out.contains("caller"),
        "the answer describes the clone's own files: {out}"
    );

    // The same answer as an index of the clone built from its sources.
    let apart = Host::new("clone-apart");
    let (code, sources, err) = Command::new(bin())
        .args(["who-calls", "helper"])
        .current_dir(&beta)
        .env("GREPPY_STORE_DIR", &apart.store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .map(|out| {
            (
                out.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&out.stdout).into_owned(),
                String::from_utf8_lossy(&out.stderr).into_owned(),
            )
        })
        .expect("spawn greppy");
    assert_eq!(code, 0, "index from the sources: {err}");
    assert_eq!(out, sources);
    for query in [&["callees", "second"][..], &["search-symbol", "caller"][..]] {
        let (code, seeded, err) = host.run(query, &beta, &[]);
        assert_eq!(code, 0, "{query:?}: {err}");
        let (_, sources, _) = apart.run(query, &beta, &[]);
        assert_eq!(seeded, sources, "{query:?}");
    }

    // The sibling's own index is untouched.
    let (code, out, err) = host.run(&["who-calls", "helper"], &alpha, &[]);
    assert_eq!(code, 0, "who-calls in the sibling: {err}");
    assert!(
        out.contains("src/lib.rs:2  caller") && !out.contains("second"),
        "{out}"
    );
    let _ = std::fs::remove_dir_all(&host.root);
    let _ = std::fs::remove_dir_all(&apart.root);
}

#[test]
fn command_output_kept_before_the_first_query_survives_the_start_from_a_sibling() {
    let host = Host::new("packs");
    let alpha = host.indexed_repository("alpha");
    let beta = host.clone_ahead(&alpha, "beta");

    // Long output is folded behind an id and kept in the clone's store,
    // which therefore exists before any index does.
    let (code, out, err) = host.run(&["bash-smart", "--", "seq", "1", "20000"], &beta, &[]);
    assert_eq!(code, 0, "bash-smart: {err}");
    let shown = format!("{out}{err}");
    let id = shown
        .split("greppy expand ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .unwrap_or_else(|| panic!("no expand id in: {shown}"))
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_string();

    let (code, out, err) = host.run(&["who-calls", "helper"], &beta, &[]);
    assert_eq!(
        code, 0,
        "who-calls in the clone: stderr={err}\nstdout={out}"
    );
    assert!(out.contains("src/lib.rs:3  second"), "{out}");

    let (code, out, err) = host.run(&["expand", &id], &beta, &[]);
    assert_eq!(code, 0, "expand {id}: {err}");
    assert!(out.lines().any(|line| line == "21"), "{out}");
    let _ = std::fs::remove_dir_all(&host.root);
}

#[test]
fn a_sibling_index_with_commits_still_in_its_log_is_copied_whole() {
    let host = Host::new("wal");
    let alpha = host.indexed_repository("alpha");
    let beta = host.clone_ahead(&alpha, "beta");

    // Another process has written to the sibling's store and still has it
    // open: the commit is in the write-ahead log, not in the database file.
    let stores = host.store.join("workspaces");
    let graph = walk(&stores)
        .into_iter()
        .find(|path| path.file_name().is_some_and(|name| name == "graph.db"))
        .expect("the sibling's store");
    let writer = rusqlite::Connection::open(&graph).unwrap();
    writer
        .execute("UPDATE projects SET indexed_at = indexed_at || ' '", [])
        .unwrap();
    let log = std::fs::metadata(graph.with_file_name("graph.db-wal")).unwrap();
    assert!(log.len() > 0, "the commit must still be in the log");

    let (code, out, err) = host.run(&["who-calls", "helper"], &beta, &[]);
    assert_eq!(
        code, 0,
        "who-calls in the clone: stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("src/lib.rs:3  second") && !out.contains("caller"),
        "{out}"
    );
    drop(writer);
    let _ = std::fs::remove_dir_all(&host.root);
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else {
            found.push(path);
        }
    }
    found
}

#[test]
fn an_unrelated_repository_is_no_start() {
    let host = Host::new("unrelated");
    host.indexed_repository("alpha");
    // Same files, its own history.
    let gamma = host.root.join("gamma");
    std::fs::create_dir_all(gamma.join("src")).unwrap();
    std::fs::write(gamma.join("src/lib.rs"), "pub fn caller() {}\n").unwrap();
    git(&gamma, &["init", "-q"]);
    git(&gamma, &["add", "."]);
    git(&gamma, &["commit", "-q", "-m", "other"]);

    let (code, out, err) = host.run(&["search-symbol", "caller"], &gamma, &[]);
    assert_ne!(code, 0, "its first index needs the gate: {out}{err}");
    assert!(err.contains("heavy-job lease"), "{err}");
    let _ = std::fs::remove_dir_all(&host.root);
}

#[test]
fn the_switch_keeps_every_first_index_under_the_gate() {
    let host = Host::new("switch");
    let alpha = host.indexed_repository("alpha");
    let beta = host.clone_ahead(&alpha, "beta");
    let (code, out, err) = host.run(
        &["who-calls", "helper"],
        &beta,
        &[("GREPPY_SIBLING_SEED", "0")],
    );
    assert_ne!(code, 0, "{out}{err}");
    assert!(err.contains("heavy-job lease"), "{err}");
    let _ = std::fs::remove_dir_all(&host.root);
}
