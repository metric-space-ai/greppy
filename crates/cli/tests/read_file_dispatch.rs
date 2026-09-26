//! Contract coverage for read, read-smart, and read-file.

use std::path::{Path, PathBuf};
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
        path_out.starts_with("note: `lib.rs` is a path; reading it as a file\nlib.rs:1-6\n"),
        "{path_out}"
    );
    assert!(path_out.contains("pub fn target()"), "{path_out}");
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
            vec!["--root", root.as_str(), "read", "probe.conf"],
        ] {
            let (code, stdout, stderr) = run_from(&cwd, &store, &args);
            assert_eq!(code, 0, "{args:?}: {stdout}\n{stderr}");
            assert!(stdout.contains("etc/probe.conf"), "{args:?}: {stdout}");
            assert!(stdout.contains("SUBDIR_SENTINEL"), "{args:?}: {stdout}");
            assert!(!stdout.contains("CWD_SENTINEL"), "{args:?}: {stdout}");
            assert!(!stdout.contains("REPO_SENTINEL"), "{args:?}: {stdout}");
        }
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
