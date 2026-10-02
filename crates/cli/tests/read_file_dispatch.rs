//! Contract coverage for read, read-smart, and read-file.

use std::path::{Path, PathBuf};
#[test]
fn read_file_unknown_options_refuse_before_opening_any_file() {
    let (repo, store) = fresh_workspace("unknown-option-no-read");
    let path = repo.join("long.jsonl");
    std::fs::write(&path, format!("SECRET_PAYLOAD_{}\n", "x".repeat(2_000_000))).unwrap();
    for option in ["--head", "--tail", "--invented-read-window"] {
        let (code, out, err) = run(
            &repo,
            &store,
            &["read-file", path.to_str().unwrap(), option, "5"],
        );
        assert_eq!(code, 64, "{out}\n{err}");
        assert!(
            out.contains("no files were read") && out.contains("--lines A:B"),
            "{out}"
        );
        assert!(
            !out.contains("SECRET_PAYLOAD_") && !out.contains("no such file: 5"),
            "{out}"
        );
        assert!(out.len() + err.len() < 2048);
    }
    for leading in ["--diagnostics", "--max-bytes=128", "--limit=1"] {
        for option in ["--head=5", "--invented-read-window=5"] {
            let (code, out, err) = run(
                &repo,
                &store,
                &[leading, "read-file", path.to_str().unwrap(), option],
            );
            assert_eq!(code, 64, "{out}\n{err}");
            assert!(out.contains("no files were read"), "{out}\n{err}");
            assert!(!out.contains("SECRET_PAYLOAD_"));
            assert!(out.len() + err.len() < 2048);
        }
    }
    let absent = repo.join("missing.txt");
    let (code, out, err) = run(
        &repo,
        &store,
        &["read-file", absent.to_str().unwrap(), "--head", "5"],
    );
    assert_eq!(code, 64, "{out}\n{err}");
    assert!(
        !out.contains("no such file"),
        "argument validation must precede file IO: {out}"
    );
    assert!(
        !store.exists(),
        "invalid file arguments must not open a graph store"
    );
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-read-family-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    (repo, base.join("store"))
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .arg("--root")
        .arg(repo)
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
    let (code, stdout, stderr) = run(repo, store, &["index", repo.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
}

fn only_graph_db_below(root: &Path) -> PathBuf {
    fn visit(path: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                visit(&child, found);
            } else if child.file_name().is_some_and(|name| name == "graph.db") {
                found.push(child);
            }
        }
    }

    let mut found = Vec::new();
    visit(root, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected one graph.db below {root:?}: {found:?}"
    );
    found.pop().unwrap()
}

#[test]
fn large_sparse_file_prefix_is_bounded_and_independent_of_invalid_tail() {
    use std::io::Write;
    let (repo, store) = fresh_workspace("large-bounded-prefix");
    let path = repo.join("dump.txt");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"alpha\nbeta\ngamma\n\xff").unwrap();
    file.set_len(635_837_957).unwrap();
    drop(file);
    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "read-file",
            path.to_str().unwrap(),
            "--lines",
            "1:3",
            "--max-bytes",
            "1200",
        ],
    );
    assert_eq!(code, 0, "{out} {err}");
    assert!(out.contains("dump.txt:1-3\nalpha\nbeta\ngamma\n"), "{out}");
    assert!(
        !store.join("workspaces").exists() && !store.join("graph.db").exists(),
        "plain bounded reads must not create a graph"
    );
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn invalid_selected_or_whole_file_text_is_never_reported_missing() {
    let (repo, store) = fresh_workspace("invalid-file-text");
    let path = repo.join("dump.txt");
    std::fs::write(&path, b"alpha\n\xff\n").unwrap();
    let (code, out, err) = run(&repo, &store, &["read-file", "dump.txt", "--lines", "1:2"]);
    assert_eq!(code, 1);
    assert!(
        out.contains("requested lines 1:2") && out.contains("UTF-8"),
        "{out} {err}"
    );
    assert!(!out.contains("no such file") && !err.contains("no such file"));
    let (code, out, err) = run(&repo, &store, &["read-file", "dump.txt", "--all"]);
    assert_eq!(code, 1, "{out} {err}");
    assert!(out.contains("cannot read file dump.txt:"), "{out} {err}");
    assert!(!out.contains("no such file"));
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn invalid_selected_text_does_not_hide_later_file_spans() {
    let (repo, store) = fresh_workspace("invalid-first-file");
    std::fs::write(repo.join("invalid.txt"), b"alpha\n\xff\n").unwrap();
    std::fs::write(repo.join("valid.txt"), b"beta\r\ngamma\nignored\n").unwrap();
    let (code, out, err) = run(
        &repo,
        &store,
        &["read-file", "invalid.txt", "valid.txt", "--lines", "1:2"],
    );
    assert_eq!(code, 1, "{out} {err}");
    assert!(out.contains("cannot read file invalid.txt:"), "{out}");
    assert!(out.contains("valid.txt:1-2\nbeta\r\ngamma\n"), "{out}");
    assert!(!out.contains("no such file") && !out.contains("ignored"));
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn cold_symbol_reads_bootstrap_after_file_handle_and_preserve_it() {
    for command in ["read", "read-smart", "who-calls", "impact"] {
        let (repo, store) = fresh_workspace(command);
        let source = "pub fn schema_marker() -> i32 { 41 }\npub fn use_marker() -> i32 { schema_marker() }\n";
        std::fs::write(repo.join("lib.rs"), source).unwrap();
        let (code, stdout, stderr) = run(&repo, &store, &["read-file", "lib.rs", "--handle"]);
        assert_eq!(code, 0, "{stdout}\n{stderr}");
        let handle = stdout
            .lines()
            .find_map(|line| line.strip_prefix("handle: "))
            .expect("file handle before graph publication");
        let graph_db = only_graph_db_below(&store);
        assert!(!graph_db.parent().unwrap().join("index.job").exists());

        let (code, stdout, stderr) = run(&repo, &store, &[command, "lib.rs::schema_marker"]);
        assert_eq!(code, 0, "{command}: {stdout}\n{stderr}");
        let expected = if matches!(command, "read" | "read-smart") {
            "41"
        } else {
            "use_marker"
        };
        assert!(stdout.contains(expected), "{stdout}");

        let (code, stdout, stderr) = run(
            &repo,
            &store,
            &[
                "replace-span",
                handle,
                "pub fn schema_marker() -> i32 { 42 }\n",
                "--dry-run",
            ],
        );
        assert_eq!(code, 0, "original handle lost: {stdout}\n{stderr}");
        assert_eq!(
            std::fs::read_to_string(repo.join("lib.rs")).unwrap(),
            source
        );
    }
}

#[test]
fn cold_linked_symbol_read_after_file_handle_attaches_published_base() {
    let (repo, store) = fresh_workspace("cold-linked-handle");
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn schema_marker() -> i32 { 41 }\n",
    )
    .unwrap();
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .args(args)
            .current_dir(&repo)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {args:?}: {output:?}");
    };
    git(&["init", "-q"]);
    git(&["add", "lib.rs"]);
    git(&[
        "-c",
        "user.name=Cold Read Test",
        "-c",
        "user.email=cold-read@test.invalid",
        "commit",
        "-qm",
        "base",
    ]);
    // Structural first use reuses a published immutable Base; it intentionally
    // avoids building a new one inside a query. Establish that Base first.
    let first = repo.parent().unwrap().join("first");
    git(&["worktree", "add", "-qb", "first", first.to_str().unwrap()]);
    index(&first, &store);
    let linked = repo.parent().unwrap().join("linked");
    git(&["worktree", "add", "-qb", "linked", linked.to_str().unwrap()]);
    let (code, stdout, stderr) = run(&linked, &store, &["read-file", "lib.rs", "--handle"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let (code, stdout, stderr) = run(&linked, &store, &["read", "lib.rs::schema_marker"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains("pub fn schema_marker"), "{stdout}");
    let (code, stdout, stderr) = run(&linked, &store, &["index", "status", "--json"]);
    assert!([0, 73].contains(&code), "{stdout}\n{stderr}");
    let status: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(status["store_cow"]["mode"], "overlay", "{status}");
    assert_eq!(status["graph_generation"], 1, "{status}");
}

#[test]
fn cold_file_handle_symbol_read_respects_auto_index_opt_out() {
    let (repo, store) = fresh_workspace("cold-handle-opt-out");
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn schema_marker() -> i32 { 41 }\n",
    )
    .unwrap();
    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "lib.rs", "--handle"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let output = Command::new(bin())
        .args(["read", "lib.rs::schema_marker"])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_AUTO_REINDEX", "0")
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(format!("{stdout}{stderr}").contains("cold"));
    assert!(!only_graph_db_below(&store)
        .parent()
        .unwrap()
        .join("index.job")
        .exists());
}

#[test]
fn symbol_reads_never_mix_stale_spans_with_shifted_source() {
    let (repo, store) = fresh_workspace("shifted-source");
    let original = "fn predecessor() {\n    let _ = 1;\n}\nfn target() {\n    let _ = 42;\n}\n";
    std::fs::write(repo.join("lib.rs"), original).unwrap();
    index(&repo, &store);
    std::fs::write(
        repo.join("lib.rs"),
        format!("// newly inserted line\n// another inserted line\n{original}"),
    )
    .unwrap();
    for args in [
        vec!["read", "target"],
        vec!["read", "target", "--json"],
        vec!["read", "target", "--head", "1", "--handle"],
        vec!["read", "predecessor", "target", "--json"],
        vec!["read-smart", "target"],
    ] {
        let output = Command::new(bin())
            .args(&args)
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", &store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_AUTO_REINDEX", "0")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            output.status.code(),
            Some(75),
            "{args:?}: {stdout}\n{stderr}"
        );
        assert!(!stdout.contains("fn target()"), "{stdout}");
        assert!(!stdout.contains("let _ ="), "{stdout}");
        assert!(
            stdout.contains("freshness") || stdout.contains("stale"),
            "{stdout}"
        );
    }
    index(&repo, &store);
    let (code, stdout, stderr) = run(&repo, &store, &["read", "target"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert_eq!(
        stdout,
        "lib.rs:6-8  target\nfn target() {\n    let _ = 42;\n}\n"
    );
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn read_is_symbols_only_whole_and_doc_extended() {
    let (repo, store) = fresh_workspace("whole");
    std::fs::write(
        repo.join("lib.rs"),
        "/// Authored docs.\n#[inline]\npub fn target() {\n    let x = 1;\n    println!(\"{x}\");\n}\n",
    )
    .unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read", "target"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        stdout,
        "lib.rs:1-6  target\n/// Authored docs.\n#[inline]\npub fn target() {\n    let x = 1;\n    println!(\"{x}\");\n}\n"
    );

    let (path_code, path_out, path_err) = run(&repo, &store, &["read", "lib.rs"]);
    assert_eq!(path_code, 0, "stdout={path_out}\nstderr={path_err}");
    assert!(
        path_out.starts_with("`lib.rs` is a file — read a symbol:\n"),
        "{path_out}"
    );
    assert!(path_out.contains("target"), "{path_out}");
    assert!(!path_out.contains("pub fn target()"), "{path_out}");
}

#[test]
fn indexed_large_source_outline_keeps_explicit_spans_all_and_handles_available() {
    let (repo, store) = fresh_workspace("outline-large");
    let source = format!(
        "pub fn target() -> i32 {{ 731 }}\n{}",
        "// private source content, not an outline\n".repeat(60)
    );
    std::fs::write(repo.join("lib.rs"), &source).unwrap();
    index(&repo, &store);
    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "lib.rs", "--handle"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(
        stdout.starts_with(
            "Source outline for `lib.rs` — large indexed source; text: --lines A:B or --all:"
        ),
        "{stdout}"
    );
    assert!(!stdout.contains("731"), "{stdout}");
    assert!(
        stdout.contains("full text: greppy read-file lib.rs --all"),
        "{stdout}"
    );
    assert!(!stdout.contains("private source content"), "{stdout}");
    assert!(
        !stdout.lines().any(|line| line.starts_with("handle: ")),
        "{stdout}"
    );
    let selector = stdout
        .lines()
        .nth(1)
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap();
    let (code, stdout, stderr) = run(&repo, &store, &["read", selector]);
    assert_eq!(
        code, 0,
        "outline selector does not resolve: {stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("pub fn target() -> i32 { 731 }"),
        "{stdout}"
    );

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read-file", "lib.rs", "--lines", "1:1", "--handle"],
    );
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains("731"), "{stdout}");
    let handle = stdout
        .lines()
        .find_map(|line| line.strip_prefix("handle: "))
        .unwrap();
    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "replace-span",
            handle,
            "pub fn target() -> i32 { 732 }\n",
            "--dry-run",
        ],
    );
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "lib.rs", "--all"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains(&source), "{stdout}");
    assert_eq!(
        std::fs::read_to_string(repo.join("lib.rs")).unwrap(),
        source
    );
}

#[test]
fn changed_source_never_offers_stale_outline_selectors_or_starts_refresh() {
    let (repo, store) = fresh_workspace("outline-stale");
    let before = format!(
        "pub fn old_name() {{}}\n{}",
        "// same-length source\n".repeat(65)
    );
    std::fs::write(repo.join("lib.rs"), &before).unwrap();
    index(&repo, &store);
    let database = only_graph_db_below(&store);
    let job = database.parent().unwrap().join("index.job");
    let old_job = std::fs::read(&job).ok();
    let after = before.replace("old_name", "new_name");
    assert_eq!(before.len(), after.len());
    std::fs::write(repo.join("lib.rs"), &after).unwrap();
    for command in ["read", "read-file"] {
        let (code, stdout, stderr) = run(&repo, &store, &[command, "lib.rs"]);
        assert_eq!(code, 0, "{stdout}\n{stderr}");
        assert!(stdout.contains(&after), "{stdout}");
        assert!(!stdout.contains("is a file — read a symbol:"), "{stdout}");
        assert!(!stdout.contains("old_name"), "{stdout}");
    }
    assert_eq!(
        std::fs::read(&job).ok(),
        old_job,
        "passive outline started a refresh"
    );
}

#[test]
fn source_read_outline_threshold_and_unindexed_fallback_do_not_start_indexing() {
    let (repo, store) = fresh_workspace("outline-threshold");
    let source = format!("pub fn target() {{}}\n{}", "// line\n".repeat(59));
    std::fs::write(repo.join("lib.rs"), &source).unwrap();
    index(&repo, &store);
    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "lib.rs"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains(&source), "{stdout}");

    let (cold_repo, cold_store) = fresh_workspace("outline-no-index");
    let source = format!("pub fn cold_target() {{}}\n{}", "// cold line\n".repeat(79));
    std::fs::write(cold_repo.join("lib.rs"), &source).unwrap();
    for command in ["read-file", "read"] {
        let (code, stdout, stderr) = run(&cold_repo, &cold_store, &[command, "lib.rs"]);
        assert_eq!(code, 0, "{stdout}\n{stderr}");
        assert!(stdout.contains(&source), "{stdout}");
    }
    fn has_graph_or_index_job(path: &Path) -> bool {
        if !path.exists() {
            return false;
        }
        std::fs::read_dir(path).unwrap().any(|entry| {
            let child = entry.unwrap().path();
            if child.is_dir() {
                has_graph_or_index_job(&child)
            } else {
                child
                    .file_name()
                    .is_some_and(|name| name == "graph.db" || name == "index.job")
            }
        })
    }
    // CLI startup may create gc.state/global.gc without graph preparation.
    // The contract is no graph publication or index job, not no cache folder.
    assert!(
        !has_graph_or_index_job(&cold_store),
        "plain reads prepared a graph"
    );
}

#[test]
fn read_head_and_tail_have_truthful_adjacent_headers() {
    let (repo, store) = fresh_workspace("head-tail");
    std::fs::write(
        repo.join("lib.rs"),
        "fn target() {\n    one();\n    two();\n    three();\n}\n",
    )
    .unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read", "target", "--head", "2", "--tail", "2"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        stdout,
        "lib.rs:1-2  target\nfn target() {\n    one();\nlib.rs:4-5  target\n    three();\n}\n"
    );
}

#[test]
fn read_multi_delivers_successes_and_nav_failures() {
    let (repo, store) = fresh_workspace("partial");
    std::fs::write(repo.join("lib.rs"), "fn first() {}\nfn second() {}\n").unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read", "first", "missing", "second"]);
    assert_eq!(code, 1, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains("lib.rs:1-1  first\nfn first() {}"),
        "{stdout}"
    );
    assert!(stdout.contains("\n\nno symbol `missing`\n"), "{stdout}");
    assert!(
        stdout.contains("\n\nlib.rs:2-2  second\nfn second() {}"),
        "{stdout}"
    );
    assert!(!stdout.contains("read:"), "{stdout}");
}

#[test]
fn read_smart_folds_by_structure_and_expand_chains() {
    let (repo, store) = fresh_workspace("smart");
    std::fs::write(
        repo.join("lib.rs"),
        "fn target(xs: &[i32]) {\n    let mut n = 0;\n    for x in xs {\n        if *x > 0 {\n            n += x;\n        }\n    }\n    println!(\"{n}\");\n}\n",
    )
    .unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read-smart", "target"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("    let mut n = 0;\n"), "{stdout}");
    assert!(
        stdout.contains("    … 3-7 folded source block — greppy expand "),
        "{stdout}"
    );
    assert!(!stdout.contains("        if *x > 0"), "{stdout}");
    let id = stdout
        .lines()
        .find_map(|line| line.split("greppy expand ").nth(1))
        .expect("gap id");

    let (expand_code, expanded, expand_stderr) = run(&repo, &store, &["expand", id]);
    assert_eq!(expand_code, 0, "stdout={expanded}\nstderr={expand_stderr}");
    assert!(expanded.starts_with("    for x in xs {\n"), "{expanded}");
    assert!(
        expanded.contains("        … 4-6 folded source block — greppy expand "),
        "{expanded}"
    );
    assert!(expanded.ends_with("    }\n"), "{expanded}");
}

#[test]
fn read_smart_applies_path_filters_before_ambiguity_resolution() {
    let (repo, store) = fresh_workspace("smart-path");
    std::fs::create_dir_all(repo.join("a")).unwrap();
    std::fs::create_dir_all(repo.join("b")).unwrap();
    std::fs::write(repo.join("a/lib.rs"), "fn target() {\n    a();\n}\n").unwrap();
    std::fs::write(repo.join("b/lib.rs"), "fn target() {\n    b();\n}\n").unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read-smart", "target", "--path", "a"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("a/lib.rs:1-3  target\n"), "{stdout}");
    assert!(!stdout.contains("b/lib.rs"), "{stdout}");
}

#[test]
fn read_applies_path_filters_before_ambiguity_resolution() {
    let (repo, store) = fresh_workspace("read-path");
    std::fs::create_dir_all(repo.join("a")).unwrap();
    std::fs::create_dir_all(repo.join("b")).unwrap();
    std::fs::write(repo.join("a/lib.rs"), "fn target() { a(); }\n").unwrap();
    std::fs::write(repo.join("b/lib.rs"), "fn target() { b(); }\n").unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read", "target", "--path", "a"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("a/lib.rs:1-1  target\n"), "{stdout}");
    assert!(!stdout.contains("b/lib.rs"), "{stdout}");
    assert!(!stdout.contains("is 2 definitions"), "{stdout}");
}

#[test]
fn read_accepts_emitted_and_simplified_path_qualified_rust_variables() {
    let (repo, store) = fresh_workspace("path-variable");
    std::fs::create_dir_all(repo.join("src/core")).unwrap();
    std::fs::write(
        repo.join("src/core/gateway.rs"),
        "pub const EMAIL_RUNTIME_ENV_KEYS: &[&str] = &[\n    \"CTO_EMAIL_PASSWORD\",\n    \"CTO_EMAIL_USERNAME\",\n];\n",
    )
    .unwrap();
    index(&repo, &store);

    for symbol in [
        "src/core/gateway.rs::Variable::EMAIL_RUNTIME_ENV_KEYS",
        "src/core/gateway.rs::EMAIL_RUNTIME_ENV_KEYS",
    ] {
        let (code, stdout, stderr) = run(&repo, &store, &["read", symbol]);
        assert_eq!(code, 0, "symbol={symbol}\nstdout={stdout}\nstderr={stderr}");
        assert!(
            stdout.starts_with("src/core/gateway.rs:1-4  EMAIL_RUNTIME_ENV_KEYS\n"),
            "symbol={symbol}\n{stdout}"
        );
        assert!(
            stdout.contains("CTO_EMAIL_PASSWORD"),
            "symbol={symbol}\n{stdout}"
        );
    }

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read", "missing/gateway.rs::EMAIL_RUNTIME_ENV_KEYS"],
    );
    assert_ne!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(
        !stdout.contains("pub const EMAIL_RUNTIME_ENV_KEYS"),
        "a missing path must not fall back to the real declaration: {stdout}"
    );
}

#[test]
fn read_refuses_same_file_field_function_collision_until_exactly_qualified() {
    let (repo, store) = fresh_workspace("same-file-field-function");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(
        repo.join("src/i960_timed.rs"),
        "struct InstructionTiming {\n    cycles: u32,\n}\n\nfn cycles(mnemonic: &str) -> Result<u32, ()> {\n    Ok(mnemonic.len() as u32)\n}\n",
    )
    .unwrap();
    index(&repo, &store);

    let simplified = "src/i960_timed.rs::cycles";
    let field = "src/i960_timed.rs::Class::InstructionTiming::cycles";
    let function = "src/i960_timed.rs::Function::cycles";

    for command in ["read", "read-smart"] {
        let (code, stdout, stderr) = run(&repo, &store, &[command, simplified]);
        assert_ne!(
            code, 0,
            "command={command}\nstdout={stdout}\nstderr={stderr}"
        );
        assert!(stdout.contains("is 2 definitions"), "{stdout}");
        assert!(stdout.contains(field), "{stdout}");
        assert!(stdout.contains(function), "{stdout}");
        assert!(
            stdout.contains(&format!("greppy read {field}"))
                && stdout.contains(&format!("greppy read {function}")),
            "{stdout}"
        );
    }

    let (code, stdout, stderr) = run(&repo, &store, &["read", simplified, "--json"]);
    assert_ne!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["status"], "ambiguous", "{value}");
    let selectors = value["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|candidate| candidate["selector"].as_str())
        .collect::<Vec<_>>();
    assert!(selectors.contains(&field), "{value}");
    assert!(selectors.contains(&function), "{value}");

    let (code, stdout, stderr) = run(&repo, &store, &["read", function]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("fn cycles(mnemonic: &str)"), "{stdout}");
    assert!(!stdout.contains("cycles: u32"), "{stdout}");
}

#[test]
fn read_file_default_bounds_one_long_utf8_line_without_a_false_handle() {
    let (repo, store) = fresh_workspace("byte-budget-utf8");
    let content = "€".repeat(100_000);
    std::fs::write(repo.join("huge.ndjson"), &content).unwrap();
    // Implicit byte previews need neither handles nor a continuation store.
    std::fs::write(&store, "not a store directory").unwrap();
    let (code, out, err) = run(&repo, &store, &["read-file", "huge.ndjson", "--handle"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.len() < 66_500, "unbounded output: {} bytes", out.len());
    assert!(out.starts_with("huge.ndjson:1-1 (last line partial)\n"));
    assert!(out.contains(&"€".repeat(21_845)));
    assert!(!out.contains('\u{fffd}'));
    assert!(out.contains(
        "truncated at 65535 source bytes (default limit 65536); total line count unknown"
    ));
    assert!(
        out.contains("234465 source bytes omitted according to file size at open (300000 bytes)")
    );
    assert!(out.contains("--lines 1:1\n"));
    assert!(out.contains("rereads the partial line in full"));
    assert!(!out.contains("handle: geh"));

    for tail in [&["--all"][..], &["--lines", "1:1"][..]] {
        let mut args = vec!["read-file", "huge.ndjson"];
        args.extend_from_slice(tail);
        let (code, out, err) = run(&repo, &store, &args);
        assert_eq!(code, 0, "{err}");
        assert_eq!(out, format!("huge.ndjson:1-1\n{content}"));
    }
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn read_file_byte_budget_continues_after_a_complete_line() {
    let (repo, store) = fresh_workspace("byte-budget-boundary");
    let content = format!("{}\ntail\n", "x".repeat(65_535));
    std::fs::write(repo.join("large.txt"), &content).unwrap();
    let (code, out, err) = run(&repo, &store, &["read-file", "large.txt"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.starts_with("large.txt:1-1\n"));
    assert!(!out.contains("last line partial"));
    assert!(out.contains("5 source bytes omitted according to file size at open"));
    assert!(out.contains("--lines 2:2\n"));
    let (code, out, err) = run(&repo, &store, &["read-file", "large.txt", "--lines", "2:2"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(out, "large.txt:2-2\ntail\n");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn read_file_default_sparse_preview_does_not_validate_or_allocate_unseen_tail() {
    use std::io::{Seek, SeekFrom, Write};
    let (repo, store) = fresh_workspace("byte-budget-sparse");
    let path = repo.join("huge.txt");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&vec![b'x'; 65_537]).unwrap();
    file.seek(SeekFrom::Start(8 * 1024 * 1024 * 1024)).unwrap();
    file.write_all(b"\xff").unwrap();
    drop(file);
    let (code, out, err) = run(&repo, &store, &["read-file", "huge.txt"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.len() < 66_500);
    assert!(out.contains("total line count unknown"));
    assert!(out.contains("8589869057 source bytes omitted according to file size at open"));
    assert!(out.contains("--lines 1:1\n"));
    assert!(!store.exists(), "preview must not initialize a store");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn read_file_default_rejects_invalid_utf8_inside_observed_prefix() {
    let (repo, store) = fresh_workspace("byte-budget-invalid-prefix");
    let mut bytes = vec![b'x'; 100_000];
    bytes[12] = 0xff;
    std::fs::write(repo.join("invalid.txt"), bytes).unwrap();
    let (code, out, err) = run(&repo, &store, &["read-file", "invalid.txt"]);
    assert_eq!(code, 1, "{out}\n{err}");
    assert!(out.contains("cannot read file invalid.txt"));
    assert!(!out.contains("truncated at"));
    // EOF is not a truncation boundary: an incomplete final character is invalid.
    std::fs::write(repo.join("invalid.txt"), b"abc\xe2\x82").unwrap();
    let (code, out, err) = run(&repo, &store, &["read-file", "invalid.txt"]);
    assert_eq!(code, 1, "{out}\n{err}");
    assert!(out.contains("cannot read file invalid.txt"));
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn read_file_pages_and_expand_continues_at_the_named_line() {
    let (repo, store) = fresh_workspace("pages");
    let content = (1..=805)
        .map(|line| format!("line {line}\n"))
        .collect::<String>();
    std::fs::write(repo.join("long.txt"), &content).unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "long.txt"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("long.txt:1-400\nline 1\n"), "{stdout}");
    assert!(
        stdout.contains("line 400\n405 more lines — greppy expand "),
        "{stdout}"
    );
    assert!(stdout.ends_with(" continues at 401\n"), "{stdout}");
    let id = stdout
        .lines()
        .last()
        .and_then(|line| line.split("greppy expand ").nth(1))
        .and_then(|tail| tail.split_whitespace().next())
        .expect("continuation id");

    let (expand_code, expanded, expand_stderr) = run(&repo, &store, &["expand", id]);
    assert_eq!(expand_code, 0, "stdout={expanded}\nstderr={expand_stderr}");
    assert!(
        expanded.starts_with("long.txt:401-800\nline 401\n"),
        "{expanded}"
    );
    assert!(
        expanded.contains("5 more lines — greppy expand "),
        "{expanded}"
    );
    assert!(expanded.ends_with(" continues at 801\n"), "{expanded}");
}

#[test]
fn read_file_ignores_missing_linked_base_for_pages_handles_and_ranges() {
    let (repo, store) = fresh_workspace("missing-linked-base");
    let content = (1..=805)
        .map(|line| format!("line {line}\n"))
        .collect::<String>();
    std::fs::write(repo.join("long.txt"), &content).unwrap();

    // Initialize only the small continuation store. No graph index is built.
    let (init_code, init_out, init_err) = run(
        &repo,
        &store,
        &["read-file", "long.txt", "--lines", "1:1", "--handle"],
    );
    assert_eq!(init_code, 0, "{init_out}\n{init_err}");
    assert!(init_out.contains("handle: geh2:"), "{init_out}");

    let graph_db = only_graph_db_below(&store);
    let missing_base = store.join("cleaned-base").join("graph.db");
    let binding = serde_json::json!({
        "version": 1,
        "base_path": missing_base,
        "base_commit": "0123456789abcdef0123456789abcdef01234567",
        "project": "missing-linked-base",
    });
    rusqlite::Connection::open(&graph_db)
        .unwrap()
        .execute(
            "INSERT INTO schema_meta(key, value) VALUES(?1, ?2)\n             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("store_cow.binding.v1", binding.to_string()),
        )
        .unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "long.txt", "--handle"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.starts_with("long.txt:1-400\nline 1\n"), "{stdout}");
    assert!(stdout.contains("line 400\nhandle: geh2:"), "{stdout}");
    assert!(!stdout.contains("line 401\n"), "{stdout}");
    let id = stdout
        .lines()
        .find_map(|line| line.split("greppy expand ").nth(1))
        .and_then(|tail| tail.split_whitespace().next())
        .expect("continuation id");

    let (expand_code, expanded, expand_err) = run(&repo, &store, &["expand", id]);
    assert_eq!(expand_code, 0, "{expanded}\n{expand_err}");
    assert!(
        expanded.starts_with("long.txt:401-800\nline 401\n"),
        "{expanded}"
    );
    assert!(!expanded.contains("line 400\n"), "{expanded}");
    assert!(!expanded.contains("line 801\n"), "{expanded}");

    let (range_code, range_out, range_err) = run(
        &repo,
        &store,
        &["read-file", "long.txt", "--lines", "800:805", "--handle"],
    );
    assert_eq!(range_code, 0, "{range_out}\n{range_err}");
    assert!(
        range_out.starts_with("long.txt:800-805\nline 800\n"),
        "{range_out}"
    );
    assert!(range_out.contains("line 805\nhandle: geh2:"), "{range_out}");
    let handle = range_out
        .lines()
        .find_map(|line| line.strip_prefix("handle: "))
        .expect("compact read-file handle");
    let (edit_code, edit_out, edit_err) = run(
        &repo,
        &store,
        &[
            "replace-span",
            handle,
            "replacement remains dry-run only\n",
            "--dry-run",
        ],
    );
    assert_eq!(edit_code, 0, "{edit_out}\n{edit_err}");
    assert_eq!(
        std::fs::read_to_string(repo.join("long.txt")).unwrap(),
        content,
        "compact handle resolution must not publish a dry-run replacement"
    );
    assert!(
        !graph_db.parent().unwrap().join("index.job").exists(),
        "read-file metadata operations must not launch an index job"
    );
}

#[test]
fn read_file_range_and_all_bypass_pagination() {
    let (repo, store) = fresh_workspace("range-all");
    std::fs::write(repo.join("config.json"), "a\nb\nc\nd\n").unwrap();
    // Exact reads must not touch the graph store. A concurrent first index
    // can be creating/migrating it, and read-file must remain deterministic.
    std::fs::write(&store, "not a store directory").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read-file", "config.json", "--lines", "2:3"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(stdout, "config.json:2-3\nb\nc\n");

    let (all_code, all_out, all_err) = run(&repo, &store, &["read-file", "config.json", "--all"]);
    assert_eq!(all_code, 0, "stdout={all_out}\nstderr={all_err}");
    assert_eq!(all_out, "config.json:1-4\na\nb\nc\nd\n");
}

#[test]
fn concurrent_read_file_ranges_never_report_empty_success() {
    let (repo, store) = fresh_workspace("parallel-range");
    let content = (1..=128)
        .map(|line| format!("line {line}\n"))
        .collect::<String>();
    std::fs::write(repo.join("source.txt"), content).unwrap();

    let workers = (0..12)
        .map(|_| {
            let repo = repo.clone();
            let store = store.clone();
            std::thread::spawn(move || {
                run(
                    &repo,
                    &store,
                    &["read-file", "source.txt", "--lines", "32:96"],
                )
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        let (code, stdout, stderr) = worker.join().expect("read worker");
        assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
        assert!(
            stdout.starts_with("source.txt:32-96\nline 32\n"),
            "{stdout}"
        );
        assert!(stdout.ends_with("line 96\n"), "{stdout}");
        assert!(
            !stdout.is_empty(),
            "successful read-file output must not be empty"
        );
    }
}

#[test]
fn read_file_accepts_explicit_absolute_path_outside_repo_only() {
    let (repo, store) = fresh_workspace("absolute-external");
    let external = repo.parent().unwrap().join("diagnostic.json");
    std::fs::write(&external, "{\n  \"passed\": false\n}\n").unwrap();

    let external_arg = external.to_str().unwrap();
    let (code, stdout, stderr) = run(&repo, &store, &["read-file", external_arg, "--all"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let canonical_external = external.canonicalize().unwrap();
    assert_eq!(
        stdout,
        format!(
            "{}:1-3\n{{\n  \"passed\": false\n}}\n",
            canonical_external.display()
        )
    );

    let (escape_code, escape_out, escape_err) =
        run(&repo, &store, &["read-file", "../diagnostic.json", "--all"]);
    assert_eq!(escape_code, 1, "stdout={escape_out}\nstderr={escape_err}");
    assert_eq!(escape_out, "no such file: ../diagnostic.json\n");
}

#[cfg(unix)]
#[test]
fn read_file_follows_relative_symlink_to_external_dependency() {
    use std::os::unix::fs::symlink;

    let (repo, store) = fresh_workspace("relative-external-symlink");
    let dependency = repo.parent().unwrap().join("dependency-store");
    std::fs::create_dir_all(dependency.join(".bin")).unwrap();
    std::fs::write(dependency.join(".bin/vp"), "#!/bin/sh\necho linked\n").unwrap();
    symlink(&dependency, repo.join("node_modules")).unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read-file", "node_modules/.bin/vp", "--all"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(stdout, "node_modules/.bin/vp:1-2\n#!/bin/sh\necho linked\n");
}

#[test]
fn read_handle_is_compact_and_existing_json_shape_survives() {
    let (repo, store) = fresh_workspace("handle");
    std::fs::write(repo.join("lib.rs"), "pub fn target() {}\n").unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read", "target", "--handle", "--json", "--diagnostics"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["schema_version"], "greppy.read.v1");
    assert_eq!(value["status"], "ok");
    let handle = value["handle"].as_str().expect("handle");
    assert!(handle.starts_with("geh2:"), "{handle}");
    assert!(handle.len() <= 70, "{}: {handle}", handle.len());
}

#[test]
fn compact_read_handle_still_drives_replace_span() {
    let (repo, store) = fresh_workspace("handle-replace");
    let original = "pub fn target() {}\n";
    std::fs::write(repo.join("lib.rs"), original).unwrap();
    index(&repo, &store);

    let (read_code, read_out, read_err) = run(&repo, &store, &["read", "target", "--handle"]);
    assert_eq!(read_code, 0, "stdout={read_out}\nstderr={read_err}");
    let handle = read_out
        .lines()
        .find_map(|line| line.strip_prefix("handle: "))
        .expect("compact read handle");

    let replacement = "pub fn target() { println!(\"changed\"); }\n";
    let (edit_code, edit_out, edit_err) = run(
        &repo,
        &store,
        &["replace-span", handle, replacement, "--dry-run"],
    );
    assert_eq!(edit_code, 0, "stdout={edit_out}\nstderr={edit_err}");
    assert_eq!(
        std::fs::read_to_string(repo.join("lib.rs")).unwrap(),
        original,
        "dry-run must not publish the replacement"
    );
}

fn run_from(cwd: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(cwd)
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

fn nested_read_fixture(tag: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let (repo, store) = fresh_workspace(tag);
    let cwd = repo.parent().unwrap().to_path_buf();
    std::fs::write(cwd.join("probe.conf"), "CWD_SENTINEL\n").unwrap();
    std::fs::write(repo.join("probe.conf"), "REPO_SENTINEL\n").unwrap();
    let nested = repo.join("etc");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("probe.conf"), "SUBDIR_SENTINEL\n").unwrap();
    (cwd, repo, store, nested)
}

#[test]
fn nested_root_reads_select_the_subdir_not_cwd_or_repo_sentinels() {
    let (cwd, repo, store, nested) = nested_read_fixture("nested-root-read");
    std::fs::create_dir(repo.join("other")).unwrap();
    let absolute = nested.to_str().unwrap();
    let trailing = format!("{absolute}/");
    let mut roots = vec![absolute.to_string(), "repo/etc".to_string(), trailing];
    #[cfg(unix)]
    {
        let link = cwd.join("etc-link");
        std::os::unix::fs::symlink(&nested, &link).unwrap();
        roots.push("etc-link".to_string());
    }
    for root in &roots {
        for args in [
            vec!["--root", root.as_str(), "read-file", "probe.conf", "--all"],
            vec![
                "--root",
                root.as_str(),
                "read-file",
                "probe.conf",
                "--path",
                "etc",
            ],
            vec!["--root", root.as_str(), "read", "probe.conf"],
        ] {
            let (code, stdout, stderr) = run_from(&cwd, &store, &args);
            assert_eq!(code, 0, "{args:?}: {stdout}\n{stderr}");
            assert!(stdout.contains("etc/probe.conf"), "{args:?}: {stdout}");
            assert!(stdout.contains("SUBDIR_SENTINEL"), "{args:?}: {stdout}");
            assert!(!stdout.contains("CWD_SENTINEL"), "{args:?}: {stdout}");
            assert!(!stdout.contains("REPO_SENTINEL"), "{args:?}: {stdout}");
        }
        let (code, stdout, stderr) = run_from(
            &cwd,
            &store,
            &[
                "--root",
                root.as_str(),
                "read-file",
                "probe.conf",
                "--path",
                "other",
            ],
        );
        assert_ne!(code, 0, "{stdout}\n{stderr}");
        assert!(stdout.contains("outside path filter"), "{stdout}\n{stderr}");
        assert!(!stdout.contains("SUBDIR_SENTINEL"), "{stdout}");
    }

    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "probe.conf", "--all"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains("probe.conf"), "{stdout}");
    assert!(stdout.contains("REPO_SENTINEL"), "{stdout}");
    assert!(!stdout.contains("SUBDIR_SENTINEL"), "{stdout}");
    assert!(!stdout.contains("etc/probe.conf"), "{stdout}");

    let (code, stdout, stderr) = run_from(&repo, &store, &["read-file", "probe.conf", "--all"]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains("REPO_SENTINEL"), "{stdout}");
    assert!(!stdout.contains("SUBDIR_SENTINEL"), "{stdout}");
    assert!(!stdout.contains("etc/probe.conf"), "{stdout}");

    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn nested_root_missing_file_does_not_select_the_ancestor_sentinel() {
    let (cwd, repo, store, nested) = nested_read_fixture("nested-root-missing");
    std::fs::remove_file(nested.join("probe.conf")).unwrap();
    let (code, stdout, stderr) = run_from(
        &cwd,
        &store,
        &[
            "--root",
            nested.to_str().unwrap(),
            "read-file",
            "probe.conf",
            "--all",
        ],
    );
    assert_eq!(code, 1, "{stdout}\n{stderr}");
    assert_eq!(stdout, "no such file: probe.conf\n");
    assert!(!stdout.contains("REPO_SENTINEL"), "{stdout}");
    assert!(!stdout.contains("CWD_SENTINEL"), "{stdout}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn nested_root_read_file_continuation_stays_on_the_subdir_file() {
    let (cwd, repo, store, nested) = nested_read_fixture("nested-root-page");
    let mut long = String::new();
    for line in 1..=805 {
        long.push_str(&format!("subdir line {line}\n"));
    }
    std::fs::write(nested.join("long.txt"), &long).unwrap();
    let mut repo_long = String::new();
    for line in 1..=805 {
        repo_long.push_str(&format!("repo line {line}\n"));
    }
    std::fs::write(repo.join("long.txt"), &repo_long).unwrap();

    let (code, stdout, stderr) = run_from(
        &cwd,
        &store,
        &["--root", nested.to_str().unwrap(), "read-file", "long.txt"],
    );
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(
        stdout.starts_with("etc/long.txt:1-400\nsubdir line 1\n"),
        "{stdout}"
    );
    assert!(stdout.contains("subdir line 400\n"), "{stdout}");
    assert!(!stdout.contains("repo line"), "{stdout}");
    let id = stdout
        .lines()
        .find_map(|line| line.split("greppy expand ").nth(1))
        .and_then(|rest| rest.split_whitespace().next())
        .expect("continuation id");

    let (expand_code, expanded, expand_stderr) = run(&repo, &store, &["expand", id]);
    assert_eq!(expand_code, 0, "{expanded}\n{expand_stderr}");
    assert!(
        expanded.starts_with("etc/long.txt:401-800\nsubdir line 401\n"),
        "{expanded}"
    );
    assert!(!expanded.contains("repo line"), "{expanded}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn nested_root_relative_escape_is_still_refused() {
    let (cwd, repo, store, nested) = nested_read_fixture("nested-root-escape");
    let (code, stdout, stderr) = run_from(
        &cwd,
        &store,
        &[
            "--root",
            nested.to_str().unwrap(),
            "read-file",
            "../probe.conf",
            "--all",
        ],
    );
    // ../probe.conf from etc lands on the repo sentinel, which is still inside
    // the workspace, so the relative operand is allowed and names the repo file.
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains("probe.conf"), "{stdout}");
    assert!(stdout.contains("REPO_SENTINEL"), "{stdout}");
    assert!(!stdout.contains("SUBDIR_SENTINEL"), "{stdout}");

    let (escape_code, escape_out, escape_err) = run_from(
        &cwd,
        &store,
        &[
            "--root",
            nested.to_str().unwrap(),
            "read-file",
            "../../probe.conf",
            "--all",
        ],
    );
    assert_eq!(escape_code, 1, "{escape_out}\n{escape_err}");
    assert_eq!(escape_out, "no such file: ../../probe.conf\n");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}
