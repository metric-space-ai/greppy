//! WP-U: multi-symbol call sites, shared ambiguity counts, text --limit,
//! `read --lines`, and a no-op re-index.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "greppy-cli-wp-u-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
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

fn find_graph_db(store_dir: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path, found: &mut Option<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.file_name().and_then(|name| name.to_str()) == Some("graph.db") {
                *found = Some(path);
                return;
            }
            if found.is_some() {
                return;
            }
        }
    }
    let mut found = None;
    walk(store_dir, &mut found);
    found
}

fn index(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(&["index", "."], repo, store);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
}

fn row_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|line| {
            let Some((file, rest)) = line.split_once(':') else {
                return false;
            };
            // A row names a file (`src/lib.rs:4`); paging metadata such as
            // `total: 3` / `offset: 1` does not.
            file.contains('.')
                && !file.contains(' ')
                && rest
                    .split_whitespace()
                    .next()
                    .is_some_and(|line_no| line_no.chars().all(|c| c.is_ascii_digit()))
        })
        .collect()
}

fn definition_count(stdout: &str) -> Option<usize> {
    let line = stdout
        .lines()
        .find(|line| line.contains(" is ") && line.contains(" definitions"))?;
    line.split_whitespace()
        .find(|word| word.chars().all(|c| c.is_ascii_digit()))?
        .parse()
        .ok()
}

#[test]
fn multi_who_calls_uses_call_lines_and_drops_file_anchors() {
    let root = fresh_dir("multi-calls");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/target.ts"),
        "export function target() {}\nexport function other() {}\n",
    )
    .unwrap();
    // The call is not on the caller's definition line. The import is a
    // bookkeeping use that must not come back as a `__file__` row.
    std::fs::write(
        repo.join("src/caller.ts"),
        "import { target, other } from \"./target.ts\";\n\
         \n\
         export function caller() {\n\
         \x20   const ready = true;\n\
         \x20   target();\n\
         \x20   other();\n\
         }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/mention.ts"),
        "import { target } from \"./target.ts\";\nexport const label = \"target\";\n",
    )
    .unwrap();
    let store = root.join("store");
    index(&repo, &store);

    let (code, single, err) = run(&["who-calls", "target"], &repo, &store);
    assert_eq!(
        code, 0,
        "single who-calls failed\nstdout={single}\nstderr={err}"
    );
    let single_rows = row_lines(&single);
    assert!(
        single_rows
            .iter()
            .any(|row| row.contains("caller.ts:") && row.contains("caller")),
        "single who-calls should name the caller at its call site\n{single}"
    );
    assert!(
        !single.contains("__file__"),
        "single who-calls must not print a synthetic anchor\n{single}"
    );

    let (code, multi, err) = run(&["who-calls", "target", "other"], &repo, &store);
    assert_eq!(
        code, 0,
        "multi who-calls failed\nstdout={multi}\nstderr={err}"
    );
    assert!(
        !multi.contains("__file__"),
        "multi who-calls must drop synthetic file anchors\n{multi}"
    );
    for row in &single_rows {
        assert!(
            multi.contains(row),
            "multi who-calls missing single call-site row `{row}`\nmulti:\n{multi}"
        );
    }
    assert!(
        multi.contains("src/caller.ts:5"),
        "call site is line 5, not the function's definition line\n{multi}"
    );
    assert!(
        !multi.contains("src/caller.ts:3  caller") && !multi.contains("src/caller.ts:3  <module>"),
        "definition line must not stand in for the call\n{multi}"
    );

    let (code, json, err) = run(&["who-calls", "target", "other", "--json"], &repo, &store);
    assert_eq!(code, 0, "multi json failed\nstdout={json}\nstderr={err}");
    let value: serde_json::Value = serde_json::from_str(&json).expect("json");
    let hits = value["hits"].as_array().expect("hits");
    assert!(
        hits.iter().any(|hit| {
            hit["file"]
                .as_str()
                .is_some_and(|file| file.ends_with("caller.ts"))
                && hit["line"].as_u64() == Some(5)
                && hit["name"]
                    .as_str()
                    .is_some_and(|name| name.contains("caller"))
        }),
        "json row must use the call-site line\n{json}"
    );
    assert!(
        hits.iter().all(|hit| {
            hit["name"]
                .as_str()
                .is_none_or(|name| !name.contains("__file__"))
                && hit["qualified_name"]
                    .as_str()
                    .is_none_or(|name| !name.ends_with("__file__"))
        }),
        "json must not emit a synthetic file-anchor row\n{json}"
    );
}

#[test]
fn who_calls_and_read_report_the_same_ambiguity_count() {
    let root = fresh_dir("ambiguous");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub struct Left;\n\
         impl Left {\n\
         \x20   pub fn dup(&self) {}\n\
         }\n\
         pub struct Right;\n\
         impl Right {\n\
         \x20   pub fn dup(&self) {}\n\
         }\n",
    )
    .unwrap();
    let store = root.join("store");
    index(&repo, &store);

    let (nav_code, nav, nav_err) = run(&["who-calls", "dup"], &repo, &store);
    let (read_code, read_out, read_err) = run(&["read", "dup"], &repo, &store);
    assert_ne!(
        nav_code, 0,
        "ambiguous who-calls should refuse\n{nav}\n{nav_err}"
    );
    assert_ne!(
        read_code, 0,
        "ambiguous read should refuse\n{read_out}\n{read_err}"
    );
    let nav_count =
        definition_count(&nav).unwrap_or_else(|| panic!("nav count missing\n{nav}\n{nav_err}"));
    let read_count = definition_count(&read_out)
        .unwrap_or_else(|| panic!("read count missing\n{read_out}\n{read_err}"));
    assert!(
        nav_count >= 2,
        "expected multiple definitions, nav={nav_count}\n{nav}"
    );
    assert_eq!(
        nav_count, read_count,
        "who-calls and read disagreed\nnav:\n{nav}\nread:\n{read_out}"
    );
}

#[test]
fn plain_who_calls_limit_truncates_and_names_the_remainder() {
    let root = fresh_dir("limit");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn budgeted() {}\n\
         pub fn extra() {}\n\
         pub fn c1() { budgeted(); extra(); }\n\
         pub fn c2() { budgeted(); }\n\
         pub fn c3() { budgeted(); }\n",
    )
    .unwrap();
    let store = root.join("store");
    index(&repo, &store);

    let (code, full, err) = run(&["who-calls", "budgeted"], &repo, &store);
    assert_eq!(code, 0, "who-calls failed\n{full}\n{err}");
    let full_rows = row_lines(&full);
    assert!(
        full_rows.len() >= 3,
        "need at least 3 callers to prove truncation\n{full}"
    );

    let (code, limited, err) = run(&["who-calls", "budgeted", "--limit", "2"], &repo, &store);
    assert_eq!(code, 0, "limited who-calls failed\n{limited}\n{err}");
    let shown = row_lines(&limited);
    assert_eq!(
        shown.len(),
        2,
        "text --limit 2 should print two rows\n{limited}"
    );
    let omitted = full_rows.len() - 2;
    assert!(
        limited.contains(&format!("… {omitted} more (next page: --offset 2)")),
        "missing truncation footer\n{limited}"
    );

    // --offset windows the text rows exactly like JSON: rows 2.. of the list.
    let (code, paged, err) = run(
        &["who-calls", "budgeted", "--limit", "1", "--offset", "1"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "paged who-calls failed\n{paged}\n{err}");
    assert_eq!(row_lines(&paged), vec![shown[1]], "{paged}");

    let (code, callees, err) = run(&["callees", "c1", "--limit", "1"], &repo, &store);
    assert_eq!(code, 0, "callees failed\n{callees}\n{err}");
    let callee_rows = row_lines(&callees);
    assert!(
        callee_rows.len() <= 1,
        "callees --limit 1 should keep at most one row\n{callees}"
    );
    if row_lines(&run(&["callees", "c1"], &repo, &store).1).len() > 1 {
        assert!(
            callees.contains("more (next page: --offset 1)"),
            "callees text should name the omitted rows\n{callees}"
        );
    }
}

#[test]
fn read_lines_is_a_file_range_for_a_symbol() {
    let root = fresh_dir("read-lines");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "alpha\nbeta\npub fn target() {}\n").unwrap();
    let store = root.join("store");
    index(&repo, &store);

    let (code, help, err) = run(&["read", "--help"], &repo, &store);
    assert_eq!(code, 0, "read --help failed\n{help}\n{err}");
    assert!(
        help.contains("--lines") && help.contains("1-based"),
        "read help must document --lines as a 1-based file range\n{help}"
    );

    let (code, out, err) = run(&["read", "target", "--lines", "1:2"], &repo, &store);
    assert_eq!(code, 0, "read --lines failed\nstdout={out}\nstderr={err}");
    assert!(out.contains("alpha"), "missing first file line\n{out}");
    assert!(out.contains("beta"), "missing second file line\n{out}");
    assert!(
        !out.contains("pub fn target"),
        "range must stop at line 2\n{out}"
    );
    assert!(
        !err.to_ascii_lowercase().contains("usage:"),
        "read --lines must not be a usage error\n{err}"
    );
}

#[test]
fn second_index_of_an_unchanged_repo_is_already_current() {
    let root = fresh_dir("reindex");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn target() {}\n").unwrap();
    let store = root.join("store");
    index(&repo, &store);

    let db = find_graph_db(&store).expect("graph.db after index");
    let before = std::fs::metadata(&db).unwrap().modified().unwrap();
    // A rewrite must not hide inside a coarse timestamp quantum.
    std::thread::sleep(Duration::from_millis(1100));

    let started = Instant::now();
    let (code, stdout, stderr) = run(&["index", "."], &repo, &store);
    let elapsed = started.elapsed();
    assert_eq!(
        code, 0,
        "second index failed\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("index already current"),
        "second index should not rebuild\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        !stdout.contains("indexed "),
        "already-current index must not report a fresh build\n{stdout}"
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "second index took {elapsed:?}, want < 1s\nstdout={stdout}\nstderr={stderr}"
    );
    let after = std::fs::metadata(&db).unwrap().modified().unwrap();
    assert_eq!(before, after, "graph.db mtime changed on a no-op re-index");
}
