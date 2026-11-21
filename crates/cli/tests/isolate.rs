//! `greppy isolate` answers the five questions around a place — the symbol
//! there, its callers, what reaches it, what it calls, what it reaches — in
//! one process that compares the checkout with the index once.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

/// `entry` calls `middle`, `middle` calls `leaf`; all indexed.
fn indexed(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("greppy-isolate-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "mod util;\npub fn entry() { middle(); }\npub fn middle() { util::leaf(); }\n",
    )
    .unwrap();
    std::fs::write(repo.join("src/util.rs"), "pub fn leaf() {}\n").unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store, &[]);
    assert_eq!(code, 0, "index: stderr={err}\nstdout={out}");
    (root, repo, store)
}

fn run(args: &[&str], repo: &Path, store: &Path, env: &[(&str, &str)]) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .stdin(std::process::Stdio::null())
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .envs(env.iter().copied())
        .output()
        .expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn names(hits: &serde_json::Value) -> Vec<String> {
    hits.as_array()
        .map(|hits| {
            hits.iter()
                .filter_map(|hit| hit["qualified_name"].as_str().or(hit["name"].as_str()))
                .map(|name| name.rsplit("::").next().unwrap_or(name).to_string())
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn one_call_answers_what_five_commands_answer() {
    let (root, repo, store) = indexed("five");
    let (code, out, err) = run(
        &["isolate", "src/lib.rs:3", "--json"],
        &repo,
        &store,
        &[("GREPPY_PROFILE", "1")],
    );
    assert_eq!(code, 0, "isolate: stderr={err}\nstdout={out}");
    let said: serde_json::Value =
        serde_json::from_str(out.trim()).unwrap_or_else(|error| panic!("{error}: {out}"));
    assert_eq!(said["status"], "ok", "{said}");
    let place = &said["places"][0];
    assert!(
        place["seed"]["qualified_name"]
            .as_str()
            .is_some_and(|name| name.ends_with("middle")),
        "the symbol at the line: {place}"
    );
    assert_eq!(names(&place["callers"]), ["entry"], "{place}");
    assert_eq!(names(&place["reach"]), ["entry"], "{place}");
    assert_eq!(names(&place["callees"]), ["leaf"], "{place}");
    assert_eq!(names(&place["forward"]), ["leaf"], "{place}");
    assert_eq!(place["gaps"], serde_json::json!([]), "{place}");

    // The same lists as the single commands give.
    for (answer, args) in [
        (
            "callers",
            &["who-calls", "src/lib.rs::middle", "--all", "--json"][..],
        ),
        (
            "callees",
            &["callees", "src/lib.rs::middle", "--all", "--json"][..],
        ),
    ] {
        let (code, single, err) = run(args, &repo, &store, &[]);
        assert_eq!(code, 0, "{args:?}: {err}");
        let single: serde_json::Value = serde_json::from_str(single.trim()).unwrap();
        assert_eq!(names(&place[answer]), names(&single["hits"]), "{answer}");
    }

    // One comparison of the checkout with the index for all five answers.
    let comparisons = err
        .lines()
        .filter(|line| line.contains("freshness: compare the checkout with the index"))
        .count();
    assert_eq!(comparisons, 1, "{err}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_symbol_is_a_place_too_and_the_text_form_lists_the_same() {
    let (root, repo, store) = indexed("symbol");
    let (code, out, err) = run(&["isolate", "src/util.rs::leaf"], &repo, &store, &[]);
    assert_eq!(code, 0, "isolate: stderr={err}\nstdout={out}");
    assert!(out.contains("direct callers and usages: 1"), "{out}");
    assert!(
        out.contains("src/lib.rs:3") && out.contains("middle"),
        "{out}"
    );
    assert!(out.contains("everything that reaches it: 2"), "{out}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_line_without_a_symbol_is_reported_as_a_gap() {
    let (root, repo, store) = indexed("gap");
    let (code, out, err) = run(
        &["isolate", "src/missing.rs:7", "--json"],
        &repo,
        &store,
        &[],
    );
    assert_eq!(code, 1, "isolate: stderr={err}\nstdout={out}");
    let said: serde_json::Value =
        serde_json::from_str(out.trim()).unwrap_or_else(|error| panic!("{error}: {out}"));
    assert_eq!(said["status"], "incomplete", "{said}");
    assert!(
        said["places"][0]["gaps"][0]
            .as_str()
            .is_some_and(|gap| gap.contains("no symbol at src/missing.rs:7")),
        "{said}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_ambiguous_or_unknown_symbol_is_a_gap_not_an_abort() {
    let (root, repo, store) = indexed("ambiguous");
    std::fs::write(repo.join("src/more.rs"), "pub fn leaf() {}\n").unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "mod util;\nmod more;\npub fn entry() { middle(); }\npub fn middle() { util::leaf(); }\n",
    )
    .unwrap();
    let (code, out, err) = run(&["index", "."], &repo, &store, &[]);
    assert_eq!(code, 0, "{err}");
    let (code, out2, err2) = run(
        &[
            "isolate",
            "leaf",
            "src/lib.rs::nothing_here",
            "src/lib.rs:4",
            "--json",
        ],
        &repo,
        &store,
        &[],
    );
    assert_eq!(code, 1, "isolate: stderr={err2}\nstdout={out2} ({out})");
    let said: serde_json::Value =
        serde_json::from_str(out2.trim()).unwrap_or_else(|error| panic!("{error}: {out2}"));
    let places = said["places"].as_array().expect("places");
    assert_eq!(places.len(), 3, "{said}");
    assert!(
        places[0]["gaps"].to_string().contains("leaf"),
        "the ambiguity is reported: {said}"
    );
    let unknown = places[1]["gaps"].to_string();
    assert!(
        unknown.contains("no such symbol") || unknown.contains("not found"),
        "{said}"
    );
    assert_eq!(names(&places[2]["callers"]), ["entry"], "{said}");
    let _ = std::fs::remove_dir_all(&root);
}
