//! Track-A integration tests for the token-saving lever:
//!
//! * `grepplus context <query>` returns the ACTUAL source span of the
//!   most relevant definitions (not just a `file:line` pointer), so an
//!   agent reads the code directly from grepplus output.
//! * The `--code` flag on `who-calls` / `callees` / `find-usages` /
//!   `trace` appends each result node's source body to the usual
//!   `file:line` line.
//!
//! These spawn the real `grepplus` binary against a multi-file fixture
//! indexed end-to-end, so the spans are read from the same files the
//! indexer recorded. Each test gets an isolated `GREPPLUS_STORE_DIR` so
//! parallel runs never collide.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_grepplus")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("grepplus-cli-ctx-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a git-rooted repo whose `src/lib.rs` calls / references symbols
/// defined in sibling modules, so both the navigation edges and the
/// readable source spans are exercised. The bodies carry distinctive
/// marker strings the assertions look for.
///
/// Returns (repo_root, store_dir).
fn make_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("lib.rs"),
        r#"
mod helper;
mod types;

fn caller() {
    // CALLER_BODY_MARKER: this line proves who-calls --code read the body.
    helper::do_it();
}

fn render(w: types::Widget) -> u32 { w.w }

fn build() {
    let _m = make(types::Marker);
}

fn make(_x: types::Marker) {}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("helper.rs"),
        "pub fn do_it() -> u32 {\n    // DO_IT_BODY_MARKER\n    42\n}\n",
    )
    .unwrap();

    std::fs::write(
        src.join("types.rs"),
        "pub struct Widget {\n    // WIDGET_FIELD_MARKER\n    pub w: u32,\n}\npub struct Marker;\n",
    )
    .unwrap();

    let store = root.join("store");
    (repo, store)
}

fn run(args: &[&str], cwd: &Path, store_dir: &Path) -> (i32, String, String) {
    run_with_env(args, cwd, store_dir, &[])
}

fn run_with_env(
    args: &[&str],
    cwd: &Path,
    store_dir: &Path,
    envs: &[(&str, &str)],
) -> (i32, String, String) {
    let mut cmd = Command::new(bin());
    cmd.args(args)
        .current_dir(cwd)
        .env("GREPPLUS_STORE_DIR", store_dir);
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("spawn grepplus");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

// ---------------------------------------------------------------------------
// context — returns the real source body of a known symbol.
// ---------------------------------------------------------------------------

#[test]
fn context_returns_locator_for_a_known_symbol() {
    let (repo, store) = index_fixture("ctx-known");

    // An exact-name lookup ("find the definition site of do_it") returns a
    // lean, grep-shaped locator: the qualified-name + file:line header and
    // the signature line — NOT the whole body (contract Z3). The full body
    // is available via `brief` or a natural-language `context` query.
    let (code, out, err) = run(&["context", "do_it"], &repo, &store);
    assert_eq!(code, 0, "context should exit 0; stderr={err}\nstdout={out}");

    // The compact header must carry the qualified name + file:span.
    assert!(
        out.contains("do_it") && out.contains("src/helper.rs:"),
        "context must print a `== qname (file:start-end) ==` header for do_it; got: {out:?}"
    );
    assert!(
        out.contains("== "),
        "context must use the `== ` span header format; got: {out:?}"
    );
    // The signature / def line is present ...
    assert!(
        out.contains("pub fn do_it"),
        "context must print the definition's signature line; got: {out:?}"
    );
    // ... but the body interior (marker on body line 2) is NOT — lean.
    assert!(
        !out.contains("DO_IT_BODY_MARKER"),
        "exact-name context must be a lean locator, not the full body; got: {out:?}"
    );
}

// Z3 guard: an exact-name / show-definition query (a single bare
// identifier that resolves to a real definition) must return MINIMAL,
// grep-shaped output — a lean locator (header with file:line + the
// signature line) for ONLY the target definition, NOT the full body and
// NOT the caller / related spans the general union path would pad with.
// This is the property that keeps grepplus grep-competitive on
// literal_control find-definition tasks (contract Z3): if it regresses,
// `context <exact_name>` bloats back to full/multiple spans and the
// literal-lookup token factor collapses (was 0.43x, i.e. ~2x worse).
#[test]
fn context_exact_name_returns_lean_locator_only() {
    let (repo, store) = index_fixture("ctx-exact-lean");

    // `do_it` is defined in helper.rs and CALLED from `caller` in lib.rs.
    // The pre-fix union path resolved the caller's body too (it mentions
    // `do_it`), emitting 2 full spans. The exact-name fast path must emit
    // ONLY do_it's locator (one header + its signature line).
    let (code, out, err) = run(&["context", "do_it"], &repo, &store);
    assert_eq!(code, 0, "context do_it should exit 0; stderr={err}");

    let headers = out.matches("== ").count();
    assert_eq!(
        headers, 1,
        "exact-name context must emit exactly ONE locator (the definition), \
         not the caller's span; got {headers}: {out:?}"
    );
    // Locator carries the file:line and the signature line ...
    assert!(
        out.contains("src/helper.rs:") && out.contains("pub fn do_it"),
        "exact-name context must print the locator (file:line + signature); got: {out:?}"
    );
    // ... but NOT the target body interior (lean) ...
    assert!(
        !out.contains("DO_IT_BODY_MARKER"),
        "exact-name context must be lean (signature only, not the full body); got: {out:?}"
    );
    // ... and NOT the caller's span.
    assert!(
        !out.contains("CALLER_BODY_MARKER"),
        "exact-name context must NOT include the caller's span; got: {out:?}"
    );
}

// Complement to the Z3 guard: a NATURAL-LANGUAGE (multi-word) query is a
// research query, not a find-definition lookup, so it must still take the
// rich union path and return FULL source bodies. Guards against
// over-tightening the exact-name fast path so it swallows research queries
// and starves the agent of the code it needs.
#[test]
fn context_multiword_query_returns_full_bodies() {
    let (repo, store) = index_fixture("ctx-rich-research");

    // "do_it helper" is two tokens → never a bare identifier, so it
    // bypasses the exact-name locator fast path and uses the full
    // FTS/semantic/code union, which reads the matched definition's whole
    // body from disk (the DO_IT_BODY_MARKER lives inside do_it's body).
    let (code, out, err) = run(&["context", "do_it helper"], &repo, &store);
    assert_eq!(
        code, 0,
        "multi-word context should resolve and exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("DO_IT_BODY_MARKER"),
        "multi-word (research) context must return the full source body; got: {out:?}"
    );
}

#[test]
fn context_emits_locator_for_type_query() {
    let (repo, store) = index_fixture("ctx-struct");

    // Exact type-name lookup → lean locator (header + declaration line),
    // matching the Z3 find-definition contract. The struct's field body is
    // not emitted for the exact-name path.
    let (code, out, err) = run(&["context", "Widget"], &repo, &store);
    assert_eq!(code, 0, "context Widget should exit 0; stderr={err}");
    assert!(
        out.contains("Widget") && out.contains("src/types.rs:"),
        "context Widget must print the locator header with file:line; got: {out:?}"
    );
    assert!(
        out.contains("pub struct Widget"),
        "context Widget must print the type's declaration line; got: {out:?}"
    );
    assert!(
        !out.contains("WIDGET_FIELD_MARKER"),
        "exact-name context must be a lean locator, not the full struct body; got: {out:?}"
    );
}

#[test]
fn context_k_limits_number_of_spans() {
    let (repo, store) = index_fixture("ctx-k");

    // --k 1 must emit exactly one `== ` header.
    let (code, out, err) = run(&["context", "make", "--k", "1"], &repo, &store);
    assert_eq!(code, 0, "context --k 1 should exit 0; stderr={err}");
    let headers = out.matches("== ").count();
    assert_eq!(
        headers, 1,
        "context --k 1 must emit exactly one span header; got {headers}: {out:?}"
    );
}

#[test]
fn context_lines_flag_prefixes_line_numbers() {
    let (repo, store) = index_fixture("ctx-lines");

    let (code, out, err) = run(&["context", "do_it", "--lines"], &repo, &store);
    assert_eq!(code, 0, "context --lines should exit 0; stderr={err}");
    // The exact-name locator prints the signature line only; with --lines
    // that line must carry its 1-based line number prefix.
    let sig_line = out
        .lines()
        .find(|l| l.contains("pub fn do_it"))
        .expect("signature line present");
    let first_tok = sig_line.split_whitespace().next().unwrap_or("");
    assert!(
        !first_tok.is_empty() && first_tok.chars().all(|c| c.is_ascii_digit()),
        "context --lines must prefix the source line with its line number; got line: {sig_line:?}"
    );
}

#[test]
fn context_json_reports_source_and_budget_metadata() {
    let (repo, store) = index_fixture("ctx-json");

    let (code, out, err) = run(&["context", "do_it", "--json", "--k", "1"], &repo, &store);
    assert_eq!(
        code, 0,
        "context --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "context");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], false);
    assert!(
        v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
        "context JSON must expose provider incompleteness: {v:?}"
    );
    assert!(
        v["incomplete_providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["language"] == "rust"),
        "rust provider incompleteness must be visible: {v:?}"
    );
    assert_eq!(v["limit"], 1);
    assert_eq!(v["candidate_total_kind"], "top_k_only");
    assert_eq!(v["shown"], 1);
    assert_eq!(v["truncated"], false);
    let spans = v["spans"].as_array().expect("spans array");
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0]["source_available"], true);
    assert!(
        spans[0]["source"]
            .as_str()
            .unwrap_or("")
            .contains("DO_IT_BODY_MARKER"),
        "context JSON must include the actual source span; got: {v:?}"
    );
    assert_eq!(spans[0]["truncated"], false);
}

#[test]
fn context_json_reports_span_truncation_metadata() {
    let root = fresh_dir("ctx-json-trunc");
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let body = (0..80)
        .map(|i| format!("    let _line_{i} = {i};"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(
        src.join("lib.rs"),
        format!("pub fn long_context_target() -> u32 {{\n{body}\n    1\n}}\n"),
    )
    .unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run(
        &["context", "long_context_target", "--json", "--k", "1"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "context --json long function should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["span_cap_lines"], 60);
    assert_eq!(v["span_truncated_count"], 1);
    assert_eq!(v["truncated"], true);
    let span = &v["spans"].as_array().unwrap()[0];
    assert_eq!(span["truncated"], true);
    assert_eq!(span["shown_lines"], 60);
    assert!(span["omitted_lines"].as_u64().unwrap_or(0) > 0);
    assert!(
        span["source"].as_str().unwrap_or("").contains("truncated"),
        "truncated source must carry inline marker; got: {span:?}"
    );
}

#[test]
fn context_unknown_query_reports_no_matches() {
    let (repo, store) = index_fixture("ctx-none");

    let (code, out, _err) = run(&["context", "no_such_symbol_zzz_qq"], &repo, &store);
    assert_eq!(code, 1, "unknown context query must exit 1; got: {out:?}");
    assert!(
        out.contains("(no matches)"),
        "unknown context query must report no matches; got: {out:?}"
    );
}

#[test]
fn context_requires_a_query() {
    let (repo, store) = index_fixture("ctx-empty");

    let (code, _out, err) = run(&["context", ""], &repo, &store);
    assert_eq!(
        code, 64,
        "empty context query is a usage error (64); err={err}"
    );
}

/// D2 fail-open: stale context serves spans (from the CURRENT files on
/// disk — the store only supplies locations), labeled with a stderr
/// warning, instead of refusing. The kill switch pins the labeled
/// path; the default policy would auto-heal this one-file drift.
#[test]
fn context_serves_labeled_stale_spans_when_auto_reindex_disabled() {
    let (repo, store) = index_fixture("ctx-stale");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    45\n}\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["context", "do_it"],
        &repo,
        &store,
        &[("GREPPLUS_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 0,
        "labeled-stale context must serve spans; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("index may be stale") && err.contains("run 'grepplus index'"),
        "labeled-stale context must warn on stderr; stderr={err:?}"
    );
    assert!(
        out.contains("do_it"),
        "labeled-stale context must serve the indexed definition; got: {out:?}"
    );
    // Span bodies come from the CURRENT file, never from a stale cache:
    // the old body marker was removed on disk, so it must not appear.
    assert!(
        !out.contains("DO_IT_BODY_MARKER"),
        "context bodies must reflect the current file content; got: {out:?}"
    );
}

/// D2: the same drift with the default policy is auto-healed; context
/// then reflects the current tree.
#[test]
fn context_auto_reindexes_small_stale_drift() {
    let (repo, store) = index_fixture("ctx-heal");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    45\n}\n",
    )
    .unwrap();

    let (code, out, err) = run(&["context", "do_it_changed"], &repo, &store);
    assert_eq!(
        code, 0,
        "healed context must find the CURRENT symbol; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("do_it_changed") && out.contains("src/helper.rs:"),
        "healed context must serve the new definition; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// plus — fused grep-like hits, not answer generation.
// ---------------------------------------------------------------------------

#[test]
fn plus_outputs_grep_like_signal_rows() {
    let (repo, store) = index_fixture("plus-format");

    let (code, out, err) = run(&["plus", "do_it", "--k", "3"], &repo, &store);
    assert_eq!(code, 0, "plus should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("src/") && out.contains(":") && out.contains("do_it"),
        "plus must print grep-like location rows; got: {out:?}"
    );
    assert!(
        !out.contains("score=") && !out.contains("signals="),
        "plus default output must not add diagnostics that can disrupt grep-trained agents; got: {out:?}"
    );
    assert!(
        !out.contains("== "),
        "plus default output must not switch into context/summary formatting; got: {out:?}"
    );

    let (code, out, err) = run(&["plus", "do_it", "--k", "3", "--explain"], &repo, &store);
    assert_eq!(
        code, 0,
        "plus --explain should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("score=") && out.contains("signals=") && out.contains("symbol="),
        "plus --explain must expose diagnostics on request; got: {out:?}"
    );
}

#[test]
fn plus_json_reports_budget_and_ranked_hits_without_changing_text_default() {
    let (repo, store) = index_fixture("plus-json");

    let (code, out, err) = run(&["plus", "do_it", "--json", "--k", "1"], &repo, &store);
    assert_eq!(
        code, 0,
        "plus --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "plus");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], false);
    assert!(
        v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
        "plus JSON must expose provider incompleteness: {v:?}"
    );
    assert!(
        v["incomplete_providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["language"] == "rust"),
        "rust provider incompleteness must be visible: {v:?}"
    );
    assert_eq!(v["limit"], 1);
    assert_eq!(v["candidate_total_kind"], "bounded_fetch_union");
    assert_eq!(v["shown"], 1);
    assert!(v["ranked_total"].as_u64().unwrap_or(0) >= 1);
    assert!(v["eligible_total"].as_u64().unwrap_or(0) >= 1);
    let hits = v["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0]["location"]
            .as_str()
            .unwrap_or("")
            .starts_with("src/")
            && hits[0]["location"].as_str().unwrap_or("").contains(':'),
        "plus JSON must preserve grep-like location; got: {v:?}"
    );
    assert!(
        !hits[0]["signals"].as_array().unwrap().is_empty(),
        "plus JSON must expose ranking signals; got: {v:?}"
    );
    assert!(
        !out.contains("== "),
        "plus --json must not mix context text headers into JSON; got: {out:?}"
    );
}

#[test]
fn plus_vectors_json_skips_embedding_on_literal_control_before_model_load() {
    let (repo, store) = index_fixture("plus-vector-literal-control");

    let (code, out, err) = run(
        &[
            "plus",
            "do_it",
            "--vectors",
            "--json",
            "--embedding-gguf",
            "/missing/embeddinggemma.gguf",
            "--embedding-tokenizer",
            "/missing/tokenizer.json",
            "--k",
            "1",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "literal plus --vectors should skip vector model load and still return exact plus hits; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "JSON literal-control vector skip should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "plus");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["vectors"], true);
    assert_eq!(v["vector_status"], "skipped_literal_control");
    assert_eq!(v["vector_candidate_total"], serde_json::Value::Null);
    assert_eq!(v["vector_hits_added"], serde_json::Value::Null);
    assert_eq!(v["shown"], 1);
    let hits = v["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1);
    let signals = hits[0]["signals"].as_array().expect("signals array");
    assert!(
        !signals.iter().any(|s| s.as_str() == Some("vector")),
        "literal-control plus output must not include vector signal; got: {v:?}"
    );
}

#[test]
fn plus_vectors_json_skips_embedding_on_graph_control_before_model_load() {
    let (repo, store) = index_fixture("plus-vector-graph-control");

    let (code, out, err) = run(
        &[
            "plus",
            "Who calls DoIt",
            "--vectors",
            "--json",
            "--embedding-gguf",
            "/missing/embeddinggemma.gguf",
            "--embedding-tokenizer",
            "/missing/tokenizer.json",
            "--k",
            "1",
        ],
        &repo,
        &store,
    );
    assert!(
        code == 0 || code == 1,
        "graph-control plus --vectors should skip vector model load and return normal plus JSON; code={code}; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "JSON graph-control vector skip should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "plus");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["vectors"], true);
    assert_eq!(v["vector_status"], "skipped_graph_control");
    assert_eq!(v["vector_candidate_total"], serde_json::Value::Null);
    assert_eq!(v["vector_hits_added"], serde_json::Value::Null);
    for hit in v["hits"].as_array().expect("hits array") {
        let signals = hit["signals"].as_array().expect("signals array");
        assert!(
            !signals.iter().any(|s| s.as_str() == Some("vector")),
            "graph-control plus output must not include vector signal; got: {v:?}"
        );
    }
}

#[test]
fn plus_code_flag_prints_source_span_under_hit() {
    let (repo, store) = index_fixture("plus-code");

    let (code, out, err) = run(&["plus", "do_it", "--code", "--k", "1"], &repo, &store);
    assert_eq!(
        code, 0,
        "plus --code should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("src/helper.rs:") && out.contains("do_it"),
        "plus --code must retain the grep-like search-hit row; got: {out:?}"
    );
    assert!(
        out.contains("DO_IT_BODY_MARKER"),
        "plus --code must print the matched symbol's source span; got: {out:?}"
    );
}

/// D2 fail-open, small drift: plus auto-heals the one-file edit and
/// answers FRESH about the current tree (the old contract refused with
/// `skipped_stale_index` + exit 1 and served nothing).
#[test]
fn plus_json_auto_reindexes_small_stale_drift() {
    let (repo, store) = index_fixture("plus-json-stale");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    44\n}\n",
    )
    .unwrap();

    let (code, out, err) = run(&["plus", "do_it", "--json", "--k", "3"], &repo, &store);
    assert_eq!(
        code, 0,
        "healed plus must answer from the current tree; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "plus");
    assert_eq!(v["status"], "ok");
    assert_eq!(
        v["fresh"], true,
        "auto-reindex must yield a fresh answer: {v:?}"
    );
    // `do_it` is still referenced by the (unchanged) caller in lib.rs,
    // so the healed index serves current hits.
    assert!(
        !v["hits"].as_array().unwrap().is_empty(),
        "healed plus must serve current hits: {v:?}"
    );
    assert!(
        !out.contains("DO_IT_BODY_MARKER"),
        "healed plus must not emit bodies that no longer exist on disk; got: {out:?}"
    );
}

/// D2 fail-open: with the auto-reindex kill switch set, stale plus
/// serves the OLD indexed hits, labeled via stderr, instead of
/// suppressing all output.
#[test]
fn plus_serves_labeled_stale_hits_when_auto_reindex_disabled() {
    let (repo, store) = index_fixture("plus-stale");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    44\n}\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["plus", "do_it", "--k", "3"],
        &repo,
        &store,
        &[("GREPPLUS_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 0,
        "labeled-stale plus must serve indexed hits; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("index may be stale") && err.contains("run 'grepplus index'"),
        "labeled-stale plus must warn on stderr; stderr={err:?}"
    );
    assert!(
        out.contains("src/") && out.contains("do_it"),
        "labeled-stale plus must serve rows from the existing index; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// --code flag on the navigation commands.
// ---------------------------------------------------------------------------

#[test]
fn who_calls_code_includes_callers_body() {
    let (repo, store) = index_fixture("whocalls-code");

    // `do_it` is called by `caller` in lib.rs. who-calls --code must
    // include the caller's body (CALLER_BODY_MARKER), not just file:line.
    let (code, out, err) = run(&["who-calls", "do_it", "--code"], &repo, &store);
    assert_eq!(code, 0, "who-calls --code should exit 0; stderr={err}");
    assert!(
        out.contains("caller") && out.contains("src/lib.rs:"),
        "who-calls --code must still print the caller's file:line; got: {out:?}"
    );
    assert!(
        out.contains("CALLER_BODY_MARKER"),
        "who-calls --code must include the caller's source body; got: {out:?}"
    );
}

#[test]
fn who_calls_without_code_omits_body() {
    let (repo, store) = index_fixture("whocalls-nocode");

    // Without --code, the body marker must NOT appear (pointer-only).
    let (code, out, _err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("caller") && !out.contains("CALLER_BODY_MARKER"),
        "who-calls (no --code) must be pointer-only; got: {out:?}"
    );
}

#[test]
fn callees_code_includes_callee_body() {
    let (repo, store) = index_fixture("callees-code");

    // `caller` calls `do_it`; callees --code must include do_it's body.
    let (code, out, err) = run(&["callees", "caller", "--code"], &repo, &store);
    assert_eq!(code, 0, "callees --code should exit 0; stderr={err}");
    assert!(
        out.contains("do_it"),
        "callees --code must list the callee do_it; got: {out:?}"
    );
    assert!(
        out.contains("DO_IT_BODY_MARKER"),
        "callees --code must include the callee's source body; got: {out:?}"
    );
}

#[test]
fn find_usages_code_includes_referrer_body() {
    let (repo, store) = index_fixture("usages-code");

    // `Widget`'s type is referenced by `render`. find-usages --code must
    // include the referrer's body.
    let (code, out, err) = run(&["find-usages", "Widget", "--code"], &repo, &store);
    assert_eq!(code, 0, "find-usages --code should exit 0; stderr={err}");
    assert!(
        out.contains("render") && out.contains("src/lib.rs:"),
        "find-usages --code must still print the referrer file:line; got: {out:?}"
    );
    // `render`'s body contains the `w.w` access; assert the source line is
    // present (the body, not just the pointer).
    assert!(
        out.contains("fn render"),
        "find-usages --code must include the referrer's source body; got: {out:?}"
    );
}

#[test]
fn trace_code_includes_node_body() {
    let (repo, store) = index_fixture("trace-code");

    // Outgoing trace from `caller` reaches `do_it`; --code must include
    // do_it's body.
    let (code, out, err) = run(&["trace", "--symbol", "caller", "--code"], &repo, &store);
    assert_eq!(code, 0, "trace --code should exit 0; stderr={err}");
    assert!(
        out.contains("do_it"),
        "trace --code must reach do_it; got: {out:?}"
    );
    assert!(
        out.contains("DO_IT_BODY_MARKER"),
        "trace --code must include the traced node's source body; got: {out:?}"
    );
}
