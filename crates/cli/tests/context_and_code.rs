//! Track-A integration tests for the token-saving lever:
//!
//! * `greppy context <query>` returns the ACTUAL source span of the
//!   most relevant definitions (not just a `file:line` pointer), so an
//!   agent reads the code directly from greppy output.
//! * The `--code` flag on `who-calls` / `callees` /
//!   `trace` appends verbatim source to the usual `file:line` line: the
//!   call-site statement for `who-calls`, the definition span for
//!   `callees` / `trace` (dev/NAV-OUTPUT-SPEC.md).
//!
//! These spawn the real `greppy` binary against a multi-file fixture
//! indexed end-to-end, so the spans are read from the same files the
//! indexer recorded. Each test gets an isolated `GREPPY_STORE_DIR` so
//! parallel runs never collide.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

struct FixtureCleanup {
    root: PathBuf,
}

impl FixtureCleanup {
    fn for_repo(repo: &Path) -> Self {
        Self {
            root: repo.parent().expect("fixture repo parent").to_path_buf(),
        }
    }

    fn for_root(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }
}

impl Drop for FixtureCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("greppy-cli-ctx-{tag}-{pid}-{n}"));
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
    run_command(args, cwd, store_dir, envs, true)
}

#[cfg(not(feature = "ci-test-assets"))]
fn run_with_inference(args: &[&str], cwd: &Path, store_dir: &Path) -> (i32, String, String) {
    run_command(
        args,
        cwd,
        store_dir,
        &[
            ("GREPPY_EMBED_DAEMON_MODEL_TTL_S", "5"),
            ("GREPPY_EMBED_DAEMON_EXIT_TTL_S", "15"),
            ("GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S", "5"),
            ("GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S", "15"),
        ],
        false,
    )
}

fn run_command(
    args: &[&str],
    cwd: &Path,
    store_dir: &Path,
    envs: &[(&str, &str)],
    skip_inference: bool,
) -> (i32, String, String) {
    let mut cmd = Command::new(bin());
    cmd.args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store_dir);
    if skip_inference {
        // Most cases verify deterministic navigation and freshness contracts.
        // The dedicated semantic case below deliberately leaves this unset.
        cmd.env("GREPPY_TEST_SKIP_INFERENCE", "1");
    }
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn index_fixture(tag: &str) -> (PathBuf, PathBuf, FixtureCleanup) {
    let (repo, store) = make_repo(tag);
    let cleanup = FixtureCleanup::for_repo(&repo);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store, cleanup)
}

#[cfg(not(feature = "ci-test-assets"))]
fn index_fixture_with_inference(tag: &str) -> (PathBuf, PathBuf, FixtureCleanup) {
    let (repo, store) = make_repo(tag);
    let cleanup = FixtureCleanup::for_repo(&repo);
    let (code, out, err) = run_with_inference(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "inference index should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store, cleanup)
}

// ---------------------------------------------------------------------------
// context — returns the real source body of a known symbol.
// ---------------------------------------------------------------------------

#[test]
fn context_returns_locator_for_a_known_symbol() {
    let (repo, store, _cleanup) = index_fixture("ctx-known");

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
// This is the property that keeps greppy grep-competitive on
// literal_control find-definition tasks (contract Z3): if it regresses,
// `context <exact_name>` bloats back to full/multiple spans and the
// literal-lookup token factor collapses (was 0.43x, i.e. ~2x worse).
#[test]
fn context_exact_name_returns_lean_locator_only() {
    let (repo, store, _cleanup) = index_fixture("ctx-exact-lean");

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

#[test]
fn context_exact_module_header_uses_display_name() {
    let (repo, store, _cleanup) = index_fixture("ctx-module-display");

    let (code, out, err) = run(&["context", "lib"], &repo, &store);
    assert_eq!(code, 0, "context lib should exit 0; stderr={err}");
    assert!(
        out.contains("== Module src/lib.rs ("),
        "context must display the synthetic module as Module <path>; got: {out:?}"
    );
    assert!(
        !out.contains("__file__"),
        "context must not leak synthetic module qnames; got: {out:?}"
    );
}

// Complement to the Z3 guard under the embedded-vector default: a
// NATURAL-LANGUAGE (multi-word) query is not a find-definition lookup, so it
// must bypass the exact-name fast path. With vectors available, that now
// returns lean semantic locators/digest output rather than the old full-body
// lexical union.
#[cfg(not(feature = "ci-test-assets"))]
#[test]
fn context_multiword_query_returns_semantic_locators() {
    let (repo, store, _cleanup) = index_fixture_with_inference("ctx-rich-research");

    // "do_it helper" is two tokens -> never a bare identifier. It should not
    // take the exact-name locator fast path; with the embedded model default it
    // takes the semantic locator/digest path.
    let (code, out, err) = run_with_inference(&["context", "do_it helper"], &repo, &store);
    assert_eq!(
        code, 0,
        "multi-word context should resolve and exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("# semantic") && out.contains("== ") && out.contains("pub fn do_it"),
        "multi-word context must return semantic locators/digest output; got: {out:?}"
    );
    assert!(
        !out.contains("DO_IT_BODY_MARKER"),
        "semantic locator context should stay lean instead of dumping full bodies; got: {out:?}"
    );
}

#[test]
fn context_emits_locator_for_type_query() {
    let (repo, store, _cleanup) = index_fixture("ctx-struct");

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
    let (repo, store, _cleanup) = index_fixture("ctx-k");

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
    let (repo, store, _cleanup) = index_fixture("ctx-lines");

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
    let (repo, store, _cleanup) = index_fixture("ctx-json");

    let (code, out, err) = run(
        &["context", "do_it", "--json", "--diagnostics", "--k", "1"],
        &repo,
        &store,
    );
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
    let _cleanup = FixtureCleanup::for_root(&root);
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
        &[
            "context",
            "long_context_target",
            "--json",
            "--diagnostics",
            "--k",
            "1",
        ],
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
    let (repo, store, _cleanup) = index_fixture("ctx-none");

    let (code, out, _err) = run(&["context", "no_such_symbol_zzz_qq"], &repo, &store);
    assert_eq!(code, 1, "unknown context query must exit 1; got: {out:?}");
    assert!(
        out.contains("(no matches)"),
        "unknown context query must report no matches; got: {out:?}"
    );
}

#[test]
fn context_requires_a_query() {
    let (repo, store, _cleanup) = index_fixture("ctx-empty");

    let (code, _out, err) = run(&["context", ""], &repo, &store);
    assert_eq!(
        code, 64,
        "empty context query is a usage error (64); err={err}"
    );
}

/// A stale graph location must never be combined with current source text.
/// With automatic refresh disabled, context refuses the request and emits no
/// indexed span.
#[test]
fn context_refuses_stale_spans_when_auto_reindex_disabled() {
    let (repo, store, _cleanup) = index_fixture("ctx-stale");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    45\n}\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["context", "do_it"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 75,
        "stale context must return EX_TEMPFAIL; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("graph freshness is drift") && out.contains("no stale indexed spans emitted"),
        "stale context must explain the refusal; stdout={out:?}"
    );
    assert!(
        !out.contains("pub fn do_it") && !out.contains("DO_IT_BODY_MARKER"),
        "stale context must not emit indexed source; got: {out:?}"
    );
}

/// The default policy may start an inline/background refresh, but the current
/// request still refuses the old generation. A later retry may use the newly
/// published generation.
#[test]
fn context_refuses_current_request_while_small_drift_refreshes() {
    let (repo, store, _cleanup) = index_fixture("ctx-heal");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    45\n}\n",
    )
    .unwrap();

    let (code, out, err) = run(&["context", "do_it_changed"], &repo, &store);
    assert_eq!(
        code, 75,
        "refreshing context must return EX_TEMPFAIL; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("graph freshness is refreshing")
            && out.contains("no stale indexed spans emitted"),
        "context must refuse while the replacement generation is publishing; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// plus — fused grep-like hits, not answer generation.
// ---------------------------------------------------------------------------

#[test]
fn plus_outputs_grep_like_signal_rows() {
    let (repo, store, _cleanup) = index_fixture("plus-format");

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
    let (repo, store, _cleanup) = index_fixture("plus-json");

    let (code, out, err) = run(
        &["plus", "do_it", "--json", "--diagnostics", "--k", "1"],
        &repo,
        &store,
    );
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
    let (repo, store, _cleanup) = index_fixture("plus-vector-literal-control");

    let (code, out, err) = run(
        &["plus", "do_it", "--json", "--diagnostics", "--k", "1"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "literal plus should skip vector model load and still return exact plus hits; stderr={err}\nstdout={out}"
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
    let (repo, store, _cleanup) = index_fixture("plus-vector-graph-control");

    let (code, out, err) = run(
        &[
            "plus",
            "Who calls DoIt",
            "--json",
            "--diagnostics",
            "--k",
            "1",
        ],
        &repo,
        &store,
    );
    assert!(
        code == 0 || code == 1,
        "graph-control plus should skip vector model load and return normal plus JSON; code={code}; stderr={err}\nstdout={out}"
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
    let (repo, store, _cleanup) = index_fixture("plus-code");

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

/// A refresh can be started for small drift, but plus must not expose rows
/// from the old generation in the triggering request.
#[test]
fn plus_json_refuses_current_request_while_small_drift_refreshes() {
    let (repo, store, _cleanup) = index_fixture("plus-json-stale");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    44\n}\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["plus", "do_it", "--json", "--diagnostics", "--k", "3"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 75,
        "refreshing plus must return EX_TEMPFAIL; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "plus");
    assert_eq!(v["status"], "skipped_stale_index");
    assert_eq!(v["fresh"], false);
    assert_eq!(v["freshness"]["state"], "refreshing");
    assert!(v["hits"].as_array().is_some_and(Vec::is_empty));
}

/// With automatic refresh disabled, stale plus refuses all indexed signals.
#[test]
fn plus_refuses_stale_hits_when_auto_reindex_disabled() {
    let (repo, store, _cleanup) = index_fixture("plus-stale");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_changed() -> u32 {\n    44\n}\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["plus", "do_it", "--k", "3"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 75,
        "stale plus must return EX_TEMPFAIL; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("graph freshness is drift") && err.contains("no stale indexed hits emitted"),
        "stale plus must explain the refusal; stderr={err:?}"
    );
    // The instruction has to precede the digests. A caller reads the front of
    // the line, and the reason string is hundreds of characters of hash
    // comparison -- put it first and the remediation is never reached.
    let remediation = err
        .find("run `greppy index")
        .expect("stale answer must say what to do; stderr={err:?}");
    let digests = err
        .find("graph freshness is drift")
        .expect("reason must still be present as evidence");
    assert!(
        remediation < digests,
        "remediation must come before the freshness digests; stderr={err:?}"
    );
    assert!(
        out.contains("no usable index") && !out.contains("src/helper.rs"),
        "stale plus must not serve indexed rows; got: {out:?}"
    );
}

#[test]
fn plus_explain_line_one_module_hit_uses_display_name() {
    let root = fresh_dir("plus-module-explain");
    let _cleanup = FixtureCleanup::for_root(&root);
    let repo = root.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("main.py"),
        "# module_marker\n\ndef f():\n    return 1\n",
    )
    .unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["plus", "module_marker", "--explain"], &repo, &store);
    assert_eq!(
        code, 0,
        "plus --explain should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("symbol=Module main.py"),
        "plus --explain must display the synthetic module as Module <path>; got: {out:?}"
    );
    assert!(
        !out.contains("__file__"),
        "plus --explain must not leak synthetic module qnames; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// --code flag on the navigation commands.
// ---------------------------------------------------------------------------

// 0.3.0 contract (dev/NAV-OUTPUT-SPEC.md, `--code`): for who-calls the span
// is the STATEMENT ENCLOSING THE CALL SITE inside the caller — NOT the
// caller's body — printed as `<file>:<line>  <name>` followed by the
// byte-exact verbatim file lines (their own indentation kept, no dedent).
#[test]
fn who_calls_code_prints_the_byte_exact_call_site_statement() {
    let (repo, store, _cleanup) = index_fixture("whocalls-code");

    // `do_it` is called by `caller` in lib.rs from one statement on line 7.
    // who-calls --code must print the call-site address and exactly that
    // statement, verbatim — not the caller's body (CALLER_BODY_MARKER).
    let (code, out, err) = run(&["who-calls", "do_it", "--code"], &repo, &store);
    assert_eq!(code, 0, "who-calls --code should exit 0; stderr={err}");
    assert_eq!(
        out, "src/lib.rs:7  caller\n    helper::do_it();\n",
        "who-calls --code must print the call-site address, then exactly the \
         verbatim statement enclosing it (indentation kept, nothing else)"
    );
    assert!(
        !out.contains("CALLER_BODY_MARKER"),
        "who-calls --code must NOT print the caller's body; got: {out:?}"
    );
}

#[test]
fn who_calls_without_code_omits_body() {
    let (repo, store, _cleanup) = index_fixture("whocalls-nocode");

    // Without --code, the body marker must NOT appear (pointer-only).
    let (code, out, _err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("caller") && !out.contains("CALLER_BODY_MARKER"),
        "who-calls (no --code) must be pointer-only; got: {out:?}"
    );
}

#[test]
fn brief_json_preserves_definition_and_expand_contract() {
    let (repo, store, _cleanup) = index_fixture("brief-json-contract");

    let (code, out, err) = run(
        &["brief", "caller", "--json", "--diagnostics"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "brief --json should exit 0; stderr={err}");
    let value: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("brief JSON invalid: {e}: {out}"));
    assert_eq!(value["schema_version"], "greppy.brief.v1");
    assert_eq!(value["command"], "brief");
    assert_eq!(value["status"], "ok");
    let definition = &value["definitions"][0];
    assert_eq!(definition["file_path"], "src/lib.rs");
    assert_eq!(definition["start_line"], 5);
    assert_eq!(definition["end_line"], 8);
    assert_eq!(definition["signature"], "fn caller()");
    assert!(
        definition["source"]
            .as_str()
            .is_some_and(|source| source.contains("CALLER_BODY_MARKER")),
        "brief JSON must contain the unchanged full source span: {definition}"
    );
    assert!(definition["summary"].is_array());

    let expand_id = value["expand_id"]
        .as_str()
        .expect("brief JSON must return a stored expand id");
    let (expand_code, expand_out, expand_err) = run(
        &["expand", expand_id, "--json", "--diagnostics"],
        &repo,
        &store,
    );
    assert_eq!(
        expand_code, 0,
        "brief expand handle must be immediately readable; stderr={expand_err}"
    );
    let expanded: serde_json::Value = serde_json::from_str(&expand_out).unwrap();
    assert_eq!(expanded["id"], expand_id);
    assert!(expanded["payload_text"]
        .as_str()
        .is_some_and(|source| source.contains("CALLER_BODY_MARKER")));
}

#[test]
fn callees_code_includes_callee_body() {
    let (repo, store, _cleanup) = index_fixture("callees-code");

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
fn trace_code_includes_node_body() {
    let (repo, store, _cleanup) = index_fixture("trace-code");

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
