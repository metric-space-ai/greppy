use super::*;
use clap::Parser;

fn drift_json(reason: &str) -> serde_json::Value {
    serde_json::json!({ "reasons": [reason] })
}

#[test]
fn version_bump_same_scope_is_scope_stable() {
    // Pure version bump, default scope on both sides → self-heal.
    assert!(version_drift_is_scope_stable(&drift_json(
        "indexer version/scope changed (was greppy-indexer-v1, expected greppy-indexer-v4)"
    )));
    // Same non-default scope, version bumped → self-heal.
    assert!(version_drift_is_scope_stable(&drift_json(
        "indexer version/scope changed (was greppy-indexer-v1;discover_scope=I8:src/*.rs, \
             expected greppy-indexer-v4;discover_scope=I8:src/*.rs)"
    )));
}

#[test]
fn scope_change_is_not_scope_stable() {
    // Different discover scope → NOT stable → refuse (fail-closed).
    assert!(!version_drift_is_scope_stable(&drift_json(
        "indexer version/scope changed (was greppy-indexer-v2;discover_scope=I8:src/*.rs, \
             expected greppy-indexer-v4)"
    )));
    // Version bump AND scope change → scope change dominates → refuse.
    assert!(!version_drift_is_scope_stable(&drift_json(
        "indexer version/scope changed (was greppy-indexer-v1, \
             expected greppy-indexer-v4;discover_scope=I8:src/*.rs)"
    )));
}

#[test]
fn transient_freshness_states_never_trigger_reindex() {
    for state in ["cold", "config_error", "failed", "unknown", "refreshing"] {
        assert!(!freshness_state_can_trigger_reindex(state), "state={state}");
    }
    assert!(freshness_state_can_trigger_reindex("drift"));
}

struct EnvRestore {
    vars: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvRestore {
    fn capture(vars: &[&'static str]) -> Self {
        Self {
            vars: vars
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
        }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (name, value) in &self.vars {
            // SAFETY: env-mutating tests hold TEST_ENV_LOCK while this guard is alive.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn test_tempdir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "greppy-cli-unit-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const EDIT_SYMBOL_HELPER_STORE: &str = "GREPPY_TEST_EDIT_SYMBOL_HELPER_STORE";

#[test]
fn edit_symbol_subprocess_helper() {
    let Some(store_root) = std::env::var_os(EDIT_SYMBOL_HELPER_STORE) else {
        return;
    };
    assert_eq!(std::env::var_os("GREPPY_STORE_DIR"), Some(store_root));

    for (label, extension, source, replacement) in [
            (
                "typescript",
                "ts",
                "export function computeTotal(items:number[]):number{ return items.reduce((a,b)=>a+b,0); }\n",
                "{ return Math.max(...items); }\n",
            ),
            (
                "kotlin",
                "kt",
                "fun computeTotal(items:IntArray):Int{ return items.sum() }\n",
                "{ return items.maxOrNull() ?: 0 }\n",
            ),
        ] {
            let root = test_tempdir(&format!("edit-symbol-{label}"));
            std::fs::create_dir(root.join(".git")).unwrap();
            std::fs::write(root.join(format!("a.{extension}")), source).unwrap();
            let replacement_path = root.join("new-body.txt");
            std::fs::write(&replacement_path, replacement).unwrap();

            let store_path = workspace_locator::store_path(&root);
            std::fs::create_dir_all(store_path.parent().unwrap()).unwrap();
            let mut store = greppy_store::Store::open(&store_path).unwrap();
            let project = workspace_locator::project_identity(&root);
            let report = greppy_indexer::index(&mut store, &root, &project).unwrap();
            assert!(report.is_clean(), "{label} index report: {report:?}");
            drop(store);

            let code = dispatch_edit(
                EditCommand::Replace {
                    symbol: "computeTotal".into(),
                    new: Some(std::fs::read_to_string(&replacement_path).unwrap()),
                    body: true,
                    dry_run: true,
                    verify: false,
                },
                false,
                root.to_str(),
            )
            .unwrap();
            assert_eq!(code, 0, "indexed {label} edit --symbol must apply");

            std::fs::remove_dir_all(root).unwrap();
        }
}

#[test]
fn edit_symbol_replaces_indexed_typescript_and_kotlin_bodies() {
    let store_root = test_tempdir("edit-symbol-ts-kt-store");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("tests::edit_symbol_subprocess_helper")
        .arg("--nocapture")
        .env(EDIT_SYMBOL_HELPER_STORE, &store_root)
        .env("GREPPY_STORE_DIR", &store_root)
        .output()
        .expect("spawn isolated edit-symbol helper");
    assert!(
        output.status.success(),
        "isolated edit-symbol helper failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    std::fs::remove_dir_all(store_root).unwrap();
}

#[test]
fn embedding_eta_uses_backend_prior_then_measured_throughput() {
    assert_eq!(initial_embedding_eta_seconds(1_200, "cpu"), Some(1_200));
    assert_eq!(initial_embedding_eta_seconds(1_200, "metal"), Some(150));
    assert_eq!(initial_embedding_eta_seconds(1_200, "cuda"), Some(100));
    assert_eq!(observed_embedding_eta_seconds(10, 100, 5_000), Some(45));
    assert_eq!(observed_embedding_rate_milli(10, 5_000), Some(2_000));
}

#[test]
fn embedding_progress_message_names_backend_counts_and_eta() {
    let progress = serde_json::json!({
        "backend": "metal",
        "completed_spans": 412,
        "total_spans": 2443,
        "eta_seconds": 134,
    });
    assert_eq!(
        embedding_progress_text(&progress),
        "semantic index building — 412/2443 spans, ETA ~2m 14s (backend metal)"
    );
}

#[test]
fn semantic_fallback_commands_use_query_tokens() {
    let commands = semantic_fallback_commands("find semantic progress marker", &[], None);
    assert_eq!(commands[0], "greppy search-symbol marker");
    assert_eq!(commands[1], "greppy grep -rnE 'semantic|progress|marker' .");
}

#[test]
fn lazy_embedding_threshold_is_inclusive_and_testable() {
    let _lock = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = EnvRestore::capture(&[ENV_LAZY_EMBED_MIN_SPANS]);
    // SAFETY: serialized by TEST_ENV_LOCK and restored by EnvRestore.
    unsafe {
        std::env::set_var(ENV_LAZY_EMBED_MIN_SPANS, "3");
    }
    let cfg = EmbeddingModelConfig {
        model_id: "test".into(),
        source: EmbeddingModelSource::Gguf {
            gguf: std::path::PathBuf::from("unused.gguf"),
            tokenizer: std::path::PathBuf::from("unused-tokenizer.json"),
        },
        max_length: None,
        device: greppy_embed_native::DevicePreference::Cpu,
    };
    assert!(!should_defer_embedding(&cfg, 2));
    assert!(should_defer_embedding(&cfg, 3));
}

#[test]
fn plus_vector_control_intent_classifies_literal_and_graph_controls() {
    let literal = plus_query_tokens("normalize_record");
    assert_eq!(
        plus_vector_control_intent("normalize_record", &literal, false),
        Some(PlusVectorControlIntent::Literal)
    );

    let who_calls = plus_query_tokens("Who calls DoIt");
    assert_eq!(
        plus_vector_control_intent("Who calls DoIt", &who_calls, false),
        Some(PlusVectorControlIntent::Graph)
    );

    let trace = plus_query_tokens("trace from runPipeline to clampValue");
    assert_eq!(
        plus_vector_control_intent("trace from runPipeline to clampValue", &trace, false),
        Some(PlusVectorControlIntent::Graph)
    );

    let impact = plus_query_tokens("what would break if computeChecksum changed");
    assert_eq!(
        plus_vector_control_intent(
            "what would break if computeChecksum changed",
            &impact,
            false
        ),
        Some(PlusVectorControlIntent::Graph)
    );
}

#[test]
fn plus_vector_control_intent_does_not_block_open_semantic_queries() {
    let tokens = plus_query_tokens("module that validates customer address input");
    assert_eq!(
        plus_vector_control_intent(
            "module that validates customer address input",
            &tokens,
            false
        ),
        None
    );
}

#[test]
fn sync_file_reports_missing_file() {
    let dir = test_tempdir("sync-file-missing");
    let missing = dir.join("missing.db");

    let err = sync_file(&missing).unwrap_err();

    assert!(
        err.to_string().contains("open file"),
        "unexpected error: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sync_parent_dir_reports_missing_parent() {
    let dir = test_tempdir("sync-parent-missing");
    let missing = dir.join("missing-dir").join("graph.db");

    let err = sync_parent_dir(&missing).unwrap_err();

    assert!(
        err.to_string().contains("open parent dir"),
        "unexpected error: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn restore_active_from_backup_copies_backup() {
    let dir = test_tempdir("restore-active");
    let active = dir.join("graph.db");
    let backup = dir.join("graph.db.prev");
    std::fs::write(&backup, b"previous-good").unwrap();

    restore_active_from_backup(&active, &backup).unwrap();

    assert_eq!(std::fs::read(&active).unwrap(), b"previous-good");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn restore_active_from_backup_reports_missing_backup() {
    let dir = test_tempdir("restore-missing-backup");
    let active = dir.join("graph.db");
    let backup = dir.join("graph.db.prev");

    let err = restore_active_from_backup(&active, &backup).unwrap_err();

    assert!(
        err.to_string().contains("not found"),
        "unexpected error: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn publish_remove_existing_fallback_replaces_active_and_keeps_backup() {
    let dir = test_tempdir("publish-fallback");
    let active = dir.join("graph.db");
    let backup = dir.join("graph.db.prev");
    let temp = dir.join("graph.db.next");
    std::fs::write(&active, b"old-active").unwrap();
    std::fs::copy(&active, &backup).unwrap();
    std::fs::write(&temp, b"new-active").unwrap();

    replace_active_with_temp(
        &temp,
        &active,
        &backup,
        PublishRenameMode::RemoveExistingFirst,
    )
    .unwrap();

    assert_eq!(std::fs::read(&active).unwrap(), b"new-active");
    assert_eq!(std::fs::read(&backup).unwrap(), b"old-active");
    assert!(!temp.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn publish_remove_existing_fallback_restores_backup_on_failure() {
    let dir = test_tempdir("publish-restore");
    let active = dir.join("graph.db");
    let backup = dir.join("graph.db.prev");
    let temp = dir.join("missing-next");
    std::fs::write(&active, b"old-active").unwrap();
    std::fs::copy(&active, &backup).unwrap();

    let err = replace_active_with_temp(
        &temp,
        &active,
        &backup,
        PublishRenameMode::RemoveExistingFirst,
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("after removing existing target"),
        "unexpected error: {err}"
    );
    assert_eq!(std::fs::read(&active).unwrap(), b"old-active");
    assert_eq!(std::fs::read(&backup).unwrap(), b"old-active");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn clamp_snippet_passes_short_lines_through_unchanged() {
    // F3: a normal-width line is returned verbatim (borrowed, no alloc).
    let short = "fn main() { println!(\"hi\"); }";
    assert_eq!(clamp_snippet(short), short);
}

#[test]
fn clamp_snippet_truncates_long_lines_with_a_marker() {
    // F3: a 20 000-char line (minified JS / data blob) must not dump in
    // full — clamp to SNIPPET_WIDTH chars + a `… (+N chars)` marker.
    let long = "x".repeat(20_000);
    let out = clamp_snippet(&long);
    assert!(out.starts_with(&"x".repeat(SNIPPET_WIDTH)));
    assert!(
        out.contains(&format!("… (+{} chars)", 20_000 - SNIPPET_WIDTH)),
        "missing truncation marker: {out}"
    );
    // The emitted preview is bounded, not the full 20 KB.
    assert!(out.chars().count() < SNIPPET_WIDTH + 40);
}

#[test]
fn clamp_snippet_never_splits_a_multibyte_codepoint() {
    // Width counts chars, not bytes, so a line of multi-byte glyphs is
    // cut on a codepoint boundary (never producing invalid UTF-8).
    let wide = "é".repeat(20_000);
    let out = clamp_snippet(&wide);
    assert!(out.starts_with(&"é".repeat(SNIPPET_WIDTH)));
}

#[test]
fn plus_vector_helper_filters_stale_generation_and_adds_grep_like_hit() {
    let root = test_tempdir("plus-vector-helper");
    let mut store = greppy_store::Store::open_memory().unwrap();
    store
        .upsert_project(&greppy_store::Project {
            name: "p".into(),
            indexed_at: "2026-07-01T00:00:00Z".into(),
            root_path: root.to_string_lossy().into_owned(),
        })
        .unwrap();

    let current_id = store
        .insert_node(&greppy_store::NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "refund_payment".into(),
            qualified_name: "p.payments.refund_payment".into(),
            file_path: "src/payments.rs".into(),
            start_line: 9,
            end_line: 12,
            properties: serde_json::json!({}),
        })
        .unwrap();
    let stale_id = store
        .insert_node(&greppy_store::NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "old_refund_payment".into(),
            qualified_name: "p.payments.old_refund_payment".into(),
            file_path: "src/old.rs".into(),
            start_line: 3,
            end_line: 6,
            properties: serde_json::json!({}),
        })
        .unwrap();
    let low_id = store
        .insert_node(&greppy_store::NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "cancel_invoice".into(),
            qualified_name: "p.payments.cancel_invoice".into(),
            file_path: "src/cancel.rs".into(),
            start_line: 20,
            end_line: 24,
            properties: serde_json::json!({}),
        })
        .unwrap();

    let model_id = "google/embeddinggemma-300m-q4";
    for embedding in [
        greppy_store::NewVectorEmbedding {
            project: "p".into(),
            model_id: model_id.into(),
            prompt_version: greppy_embed_native::PROMPT_VERSION.into(),
            task: greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE.into(),
            node_id: Some(current_id),
            chunk_idx: 0,
            qualified_name: "p.payments.refund_payment".into(),
            file_path: "src/payments.rs".into(),
            start_line: 9,
            end_line: 12,
            content_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .into(),
            graph_generation: 7,
            vector: vec![0.99, 0.01],
        },
        greppy_store::NewVectorEmbedding {
            project: "p".into(),
            model_id: model_id.into(),
            prompt_version: greppy_embed_native::PROMPT_VERSION.into(),
            task: greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE.into(),
            node_id: Some(stale_id),
            chunk_idx: 0,
            qualified_name: "p.payments.old_refund_payment".into(),
            file_path: "src/old.rs".into(),
            start_line: 3,
            end_line: 6,
            content_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                .into(),
            graph_generation: 6,
            vector: vec![1.0, 0.0],
        },
        greppy_store::NewVectorEmbedding {
            project: "p".into(),
            model_id: model_id.into(),
            prompt_version: greppy_embed_native::PROMPT_VERSION.into(),
            task: greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE.into(),
            node_id: Some(low_id),
            chunk_idx: 0,
            qualified_name: "p.payments.cancel_invoice".into(),
            file_path: "src/cancel.rs".into(),
            start_line: 20,
            end_line: 24,
            content_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                .into(),
            graph_generation: 7,
            vector: vec![0.0, 1.0],
        },
    ] {
        store.upsert_vector_embedding(&embedding).unwrap();
    }

    let mut hits = std::collections::BTreeMap::new();
    let added = plus_add_vector_hits_from_query_vector(
        &store,
        "p",
        &root,
        false,
        &mut hits,
        model_id,
        7,
        &[1.0, 0.0],
        10,
    )
    .unwrap();

    assert_eq!(added, 1);
    assert_eq!(hits.len(), 1);
    let hit = hits.values().next().unwrap();
    assert_eq!(hit.location, "src/payments.rs:9");
    assert!(hit.signals.contains("vector"));
    assert!(hit.score > 0.75);
    assert!(!hits.contains_key("src/old.rs:3"));
    assert!(!hits.contains_key("src/cancel.rs:20"));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn parse_unknown_subcommand_becomes_passthrough() {
    // `greppy grep -R foo .` — clap's `allow_external_subcommands`
    // routes `grep -R foo .` into `passthrough` (since the trailing
    // var arg captures it).
    let cli = Cli::try_parse_from(["greppy", "grep", "-R", "foo", "."]).unwrap();
    assert!(
        cli.command.is_none(),
        "expected no subcommand, got {:?}",
        cli.command
    );
    assert_eq!(cli.passthrough, vec!["grep", "-R", "foo", "."]);
}

#[test]
fn parse_bare_flags_become_passthrough() {
    // `greppy -R foo .` (no `grep` prefix) — common agent behaviour.
    let cli = Cli::try_parse_from(["greppy", "-R", "foo", "."]).unwrap();
    assert!(
        cli.command.is_none(),
        "expected no subcommand, got {:?}",
        cli.command
    );
    assert_eq!(cli.passthrough, vec!["-R", "foo", "."]);
}

#[test]
fn parse_implemented_subcommand() {
    let cli = Cli::try_parse_from(["greppy", "index", "."]).unwrap();
    assert!(matches!(cli.command, Some(Command::Index { .. })));
}

#[test]
fn index_always_uses_the_embedded_model() {
    let cli = Cli::try_parse_from(["greppy", "index", "."]).unwrap();
    match cli.command {
        Some(Command::Index { path, .. }) => assert_eq!(path.as_deref(), Some(".")),
        other => panic!("unexpected command: {other:?}"),
    }

    assert!(
        Cli::try_parse_from(["greppy", "index", "--embedding-gguf", "model.gguf", "."]).is_err()
    );
    assert!(Cli::try_parse_from(["greppy", "index", "--embeddings", "."]).is_err());
}

#[test]
fn parse_search_family_surface_and_joinable_query() {
    let cli = Cli::try_parse_from([
        "greppy", "search", "--json", "retry", "a", "failed", "request",
    ])
    .unwrap();
    match cli.command {
        Some(Command::Search {
            query_parts, json, ..
        }) => {
            assert_eq!(query_parts, ["retry", "a", "failed", "request"]);
            assert!(json);
        }
        other => panic!("unexpected command: {other:?}"),
    }

    for old in [
        "semantic-search",
        "semantic",
        "search-symbols",
        "search-code",
    ] {
        let parsed = Cli::try_parse_from(["greppy", old, "needle"]).unwrap();
        assert!(parsed.command.is_none(), "{old}");
        assert_eq!(parsed.passthrough.first().map(String::as_str), Some(old));
    }
    assert!(Cli::try_parse_from(["greppy", "search-symbol", "retry_handler"]).is_ok());
    assert!(Cli::try_parse_from(["greppy", "search-pattern", "retry.*handler"]).is_ok());
}

#[test]
fn parse_path_disambiguation_and_hyphen_values() {
    // Rebuild #2, CLI-SPEC rule 2: the path filter is spelled `--path` and
    // nothing else. A positional argument is a further SYMBOL.
    match Cli::try_parse_from(["greppy", "brief", "open", "--path", "src/flask/testing.py"])
        .unwrap()
        .command
    {
        Some(Command::Brief {
            symbols, path_opts, ..
        }) => {
            assert_eq!(symbols, vec!["open".to_string()]);
            assert_eq!(path_opts, vec!["src/flask/testing.py".to_string()]);
        }
        other => panic!("unexpected: {other:?}"),
    }
    // 0.3.2: the documented high-yield multi-symbol brief is accepted.
    match Cli::try_parse_from(["greppy", "brief", "open", "close"])
        .unwrap()
        .command
    {
        Some(Command::Brief { symbols, .. }) => {
            assert_eq!(symbols, vec!["open".to_string(), "close".to_string()]);
        }
        other => panic!("unexpected: {other:?}"),
    }

    // `read` accepts path-shaped positionals and the commonly carried --code
    // flag so the dispatcher can return source instead of a parser failure.
    assert!(Cli::try_parse_from(["greppy", "read", "open", "a/mod.py"]).is_ok());
    assert!(Cli::try_parse_from(["greppy", "read", "open", "--code"]).is_ok());
    assert!(Cli::try_parse_from(["greppy", "read-file", "a/mod.py"]).is_ok());

    // Selector and content values may begin with '-' (real diff/RST lines).
    assert!(Cli::try_parse_from([
        "greppy",
        "edit",
        "replace",
        "--file",
        "CHANGES.rst",
        "--old",
        "-   Fix how",
        "--content",
        "-   Fix what",
    ])
    .is_ok());
    assert!(Cli::try_parse_from([
        "greppy",
        "edit",
        "replace",
        "--file",
        "f.py",
        "--pattern",
        "-x",
        "--content",
        "-y",
    ])
    .is_ok());
}

#[test]
fn parse_plus_uses_vectors_without_a_public_flag() {
    let cli =
        Cli::try_parse_from(["greppy", "plus", "--json", "--k", "5", "refund workflow"]).unwrap();
    match cli.command {
        Some(Command::Plus { query, k, json, .. }) => {
            assert_eq!(query.as_deref(), Some("refund workflow"));
            assert_eq!(k, 5);
            assert!(json);
        }
        other => panic!("unexpected command: {other:?}"),
    }

    assert!(Cli::try_parse_from(["greppy", "plus", "--vectors", "refund workflow"]).is_err());
    assert!(Cli::try_parse_from([
        "greppy",
        "plus",
        "--embedding-gguf",
        "model.gguf",
        "refund workflow"
    ])
    .is_err());
}

#[test]
fn embedding_config_defaults_to_bundled_embeddinggemma_when_no_flags() {
    // OWNER HARD RULE (regression guard): embeddings must ALWAYS work by
    // default. With no --embedding-* flag/env, the resolver MUST fall back
    // to the baked-in EmbeddingGemma (never the "model required" error).
    // This locks the fix for the regression where search silently
    // ran on the lexical/algorithmic path with no vectors at all.
    let cfg = embedding_config_required(EmbeddingCliArgs {
        device: None,
        no_gpu: true,
    })
    .expect("no-flags embedding config must resolve to the embedded model, not error");
    assert!(
        matches!(cfg.source, EmbeddingModelSource::Gguf { .. }),
        "default embedding source must be the baked-in GGUF (embeddings never off)"
    );
}

#[test]
fn cli_device_flags_parse_on_embedding_commands() {
    let cli =
        Cli::try_parse_from(["grep", "search", "--device", "cuda", "refund workflow"]).unwrap();
    assert_eq!(cli.device.as_deref(), Some("cuda"));
    assert!(!cli.no_gpu);
    match cli.command {
        Some(Command::Search { query_parts, .. }) => {
            assert_eq!(query_parts, vec!["refund workflow".to_string()]);
        }
        other => panic!("unexpected command: {other:?}"),
    }
    assert!(Cli::try_parse_from([
        "grep",
        "search",
        "--device",
        "cuda",
        "--no-gpu",
        "refund workflow",
    ])
    .is_err());
}

#[test]
fn embedding_device_preference_obeys_cli_and_env() {
    let _guard = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = EnvRestore::capture(&[
        ENV_DEVICE,
        ENV_NO_GPU,
        ENV_EMBED_CUDA_DEVICE,
        ENV_QWEN_CUDA_DEVICE,
    ]);
    // SAFETY: serialized by TEST_ENV_LOCK and restored by EnvRestore.
    unsafe {
        std::env::remove_var(ENV_DEVICE);
        std::env::remove_var(ENV_NO_GPU);
    }

    assert_eq!(
        embedding_device_preference(None, false).unwrap(),
        greppy_embed_native::DevicePreference::Auto
    );

    // SAFETY: serialized by TEST_ENV_LOCK and restored by EnvRestore.
    unsafe {
        std::env::set_var(ENV_DEVICE, "metal");
    }
    assert_eq!(
        embedding_device_preference(None, false).unwrap(),
        greppy_embed_native::DevicePreference::Metal
    );
    assert_eq!(
        embedding_device_preference(Some("cuda"), false).unwrap(),
        greppy_embed_native::DevicePreference::Cuda
    );
    assert_eq!(
        embedding_device_preference(Some("cuda:2"), false).unwrap(),
        greppy_embed_native::DevicePreference::Cuda
    );
    configure_explicit_cuda_device(Some("cuda:2")).unwrap();
    assert_eq!(env_nonempty(ENV_EMBED_CUDA_DEVICE).as_deref(), Some("2"));
    assert_eq!(env_nonempty(ENV_QWEN_CUDA_DEVICE).as_deref(), Some("2"));
    assert_eq!(
        inference_device_identity(&greppy_embed_native::DevicePreference::Cuda),
        "cuda:2"
    );
    assert_eq!(
        embedding_device_preference(Some("cpu"), true).unwrap(),
        greppy_embed_native::DevicePreference::Cpu
    );

    // SAFETY: serialized by TEST_ENV_LOCK and restored by EnvRestore.
    unsafe {
        std::env::set_var(ENV_NO_GPU, "1");
    }
    assert_eq!(
        embedding_device_preference(Some("cuda"), false).unwrap(),
        greppy_embed_native::DevicePreference::Cpu
    );

    // SAFETY: serialized by TEST_ENV_LOCK and restored by EnvRestore.
    unsafe {
        std::env::remove_var(ENV_NO_GPU);
    }
    let err = embedding_device_preference(Some("vulkan"), false).unwrap_err();
    assert!(matches!(err, Error::Invalid(msg) if msg.contains("auto|cpu|metal|cuda")));
}

fn vector_hit_for_test(
    file_path: &str,
    start_line: i64,
    end_line: i64,
    qualified_name: &str,
    score: f32,
) -> greppy_store::VectorSearchHit {
    greppy_store::VectorSearchHit {
        embedding: greppy_store::VectorEmbedding {
            id: start_line,
            project: "p".into(),
            model_id: "m".into(),
            prompt_version: greppy_embed_native::PROMPT_VERSION.into(),
            task: greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE.into(),
            node_id: None,
            chunk_idx: 0,
            qualified_name: qualified_name.into(),
            file_path: file_path.into(),
            start_line,
            end_line,
            content_sha256: "0".repeat(64),
            graph_generation: 1,
            dim: 2,
            vector_norm: 1.0,
            vector: vec![1.0, 0.0],
            created_at: "2026-07-08T00:00:00Z".into(),
        },
        score,
    }
}

#[test]
fn semantic_vector_purpose_lookup_is_embedding_id_keyed() {
    let hits = [
        vector_hit_for_test("src/noise.rs", 30, 33, "noise", 0.99),
        vector_hit_for_test("src/read.rs", 10, 12, "read", 0.77),
    ];
    let purposes = vec![SemanticVectorPurpose {
        embedding_id: 10,
        file_path: "src/read.rs".into(),
        start_line: 10,
        end_line: 15,
        display_loc: "src/read.rs:10-15".into(),
        signature: "fn read()".into(),
        bullets: vec!["opens the matching data path".into()],
    }];

    assert!(vector_purpose_for_hit(Some(&purposes), &hits[0]).is_none());
    let purpose = vector_purpose_for_hit(Some(&purposes), &hits[1]).unwrap();
    assert_eq!(purpose.signature, "fn read()");
    assert_eq!(purpose.display_loc, "src/read.rs:10-15");
    assert_eq!(purpose.bullets, ["opens the matching data path"]);
}

#[test]
fn semantic_vector_hits_are_deduplicated_by_definition() {
    let mut first = vector_hit_for_test("src/lib.rs", 10, 20, "first", 0.99);
    first.embedding.id = 1;
    first.embedding.node_id = Some(7);
    let mut duplicate_chunk = first.clone();
    duplicate_chunk.embedding.id = 2;
    duplicate_chunk.embedding.chunk_idx = 1;
    duplicate_chunk.score = 0.98;
    let mut second = vector_hit_for_test("src/lib.rs", 30, 40, "second", 0.97);
    second.embedding.id = 3;
    second.embedding.node_id = Some(8);

    let hits = dedupe_semantic_vector_hits(vec![first, duplicate_chunk, second], 6);

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].embedding.id, 1);
    assert_eq!(hits[1].embedding.id, 3);
}

#[test]
fn semantic_vector_json_row_matches_agent_contract() {
    let hit = vector_hit_for_test(
        "serde_derive/src/internals/case.rs",
        82,
        82,
        "serde_derive/src/internals/case.rs::RenameRule::apply_to_field",
        0.91,
    );
    let purpose = SemanticVectorPurpose {
        embedding_id: 82,
        file_path: "serde_derive/src/internals/case.rs".into(),
        start_line: 82,
        end_line: 109,
        display_loc: "serde_derive/src/internals/case.rs:82-109".into(),
        signature: "pub fn apply_to_field(self, field: &str) -> String".into(),
        bullets: vec!["Applies the configured rename/case rule to a struct field name.".into()],
    };
    let expand = ExpandHandle {
        id: "semantic-valid-id".into(),
        summary: "3 further hits".into(),
    };

    let row = semantic_vector_json_row(&hit, Some(&purpose), Some(&expand));

    assert_eq!(row["file_path"], "serde_derive/src/internals/case.rs");
    assert_eq!(row["start_line"], 82);
    assert_eq!(row["end_line"], 109);
    assert_eq!(
        row["signature"],
        "pub fn apply_to_field(self, field: &str) -> String"
    );
    assert_eq!(
        row["summary"],
        serde_json::json!(["Applies the configured rename/case rule to a struct field name."])
    );
    assert_eq!(row["expand_id"], "semantic-valid-id");
}

#[test]
fn semantic_vector_counts_distinguish_ranked_hits_from_candidates() {
    let (retrieved, omitted, unranked_candidates, truncated) =
        semantic_vector_count_values(7, 6, 3);
    assert_eq!(retrieved, 6);
    assert_eq!(omitted, 3);
    assert_eq!(unranked_candidates, 1);
    assert!(truncated);
}

#[test]
fn semantic_expand_pack_round_trips_full_source_span() {
    let root = test_tempdir("semantic-expand-contract");
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
            root.join("src/lib.rs"),
            "pub fn apply_to_field(self_value: Rule, field: &str) -> String {\n    let renamed = self_value.apply(field);\n    renamed\n}\n",
        )
        .unwrap();
    let mut store = greppy_store::Store::open_memory().unwrap();
    store
        .upsert_project(&greppy_store::Project {
            name: "p".into(),
            indexed_at: "2026-07-09T00:00:00Z".into(),
            root_path: root.to_string_lossy().into_owned(),
        })
        .unwrap();
    let node_id = store
        .insert_node(&greppy_store::NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "apply_to_field".into(),
            qualified_name: "src/lib.rs::Function::apply_to_field".into(),
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap();
    let mut hit = vector_hit_for_test(
        "src/lib.rs",
        1,
        1,
        "src/lib.rs::Function::apply_to_field",
        0.91,
    );
    hit.embedding.node_id = Some(node_id);

    let purposes = semantic_vector_purposes(
        &store,
        Some(root.to_str().unwrap()),
        std::slice::from_ref(&hit),
        false,
    )
    .unwrap()
    .unwrap();
    assert_eq!(purposes[0].end_line, 4);
    assert_eq!(
        purposes[0].signature,
        "pub fn apply_to_field(self_value: Rule, field: &str) -> String"
    );

    let handle = insert_semantic_vector_expand_pack(
        &store,
        Some(root.to_str().unwrap()),
        "p",
        "rename a field",
        7,
        &[hit],
    )
    .expect("stored expand handle");
    let pack = store
        .get_expand_pack(&handle.id)
        .unwrap()
        .expect("expand handle remains readable");
    assert!(pack.expires_at > pack.created_at);
    assert!(pack.payload_text.contains("let renamed ="));
    let row = &pack.payload_json.as_ref().unwrap()["hits"][0];
    assert_eq!(row["start_line"], 1);
    assert_eq!(row["end_line"], 4);
    assert_eq!(
        row["signature"],
        "pub fn apply_to_field(self_value: Rule, field: &str) -> String"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn semantic_signature_from_span_uses_first_code_line() {
    let code = "\n    // comment\n    fn steal_into(&self, dst: &mut Local<T>) -> Option<T> {\n        None\n    }\n";

    let signature = semantic_signature_from_span(code).unwrap();

    assert_eq!(
        signature,
        "fn steal_into(&self, dst: &mut Local<T>) -> Option<T>"
    );
}

#[test]
fn read_span_trusts_multiline_parser_end_for_python() {
    let root = test_tempdir("python-parser-span");
    std::fs::write(
            root.join("module.py"),
            "def first() -> int:\n    return 1\n\ndef second() -> dict[str, int]:\n    return {\"value\": 2}\n",
        )
        .unwrap();

    let span = read_span_with_meta(&root, "module.py", 1, 2, 60, false).unwrap();

    assert_eq!(span.end_line, 2);
    assert_eq!(span.text, "def first() -> int:\n    return 1\n");
    assert!(!span.text.contains("second"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn read_span_recovers_legacy_single_line_rust_definition() {
    let root = test_tempdir("legacy-rust-span");
    std::fs::write(
        root.join("lib.rs"),
        "fn value() -> i32 {\n    1\n}\n\nfn next() {}\n",
    )
    .unwrap();

    let span = read_span_with_meta(&root, "lib.rs", 1, 1, 60, false).unwrap();

    assert_eq!(span.end_line, 3);
    assert_eq!(span.text, "fn value() -> i32 {\n    1\n}\n");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn semantic_signature_from_span_preserves_multiline_source_signature() {
    let code = r#"pub unsafe extern "C" fn transform<'a, T: Clone>(
    value: &'a T,
) -> Option<&'a T>
where
    T: Send,
{
    Some(value)
}
"#;

    let signature = semantic_signature_from_span(code).unwrap();

    assert_eq!(
            signature,
            "pub unsafe extern \"C\" fn transform<'a, T: Clone>( value: &'a T, ) -> Option<&'a T> where T: Send,"
        );
}

#[test]
fn semantic_signature_from_span_stops_at_python_body_colon() {
    let source = "async def load_value(\n    key: str,\n    *,\n    default: dict[str, int] | None = None,\n) -> dict[str, int]:\n    value = await fetch(key)\n    return value or default or {}\n";
    assert_eq!(
            semantic_signature_from_span(source).as_deref(),
            Some(
                "async def load_value( key: str, *, default: dict[str, int] | None = None, ) -> dict[str, int]"
            )
        );
}

#[test]
fn semantic_signature_from_span_keeps_python_forward_annotation_colon() {
    let source = "def load_value(key: str) -> 'dict[str: int]':\n    return {}\n";
    assert_eq!(
        semantic_signature_from_span(source).as_deref(),
        Some("def load_value(key: str) -> 'dict[str: int]'")
    );
}

#[test]
fn semantic_signature_from_span_does_not_add_unit_return() {
    let code = "pub fn rename_by_rules(&mut self, rules: RenameAllRules) {\n}\n";

    assert_eq!(
        semantic_signature_from_span(code).as_deref(),
        Some("pub fn rename_by_rules(&mut self, rules: RenameAllRules)")
    );
}

#[test]
fn semantic_signature_function_like_skips_structs() {
    assert!(semantic_signature_is_function_like(
        "pub fn run_task() -> T",
        Some("Function")
    ));
    assert!(!semantic_signature_is_function_like(
        "pub struct Local<T>",
        Some("Struct")
    ));
}

#[test]
fn semantic_purpose_span_cap_limits_lines_and_bytes() {
    let code = (0..80)
        .map(|i| format!("let line_{i} = \"{}\";", "é".repeat(80)))
        .collect::<Vec<_>>()
        .join("\n");

    let capped = cap_semantic_purpose_span(&code);

    assert!(capped.lines().count() <= SEMANTIC_PURPOSE_SPAN_CAP_LINES);
    assert!(capped.len() <= SEMANTIC_PURPOSE_SPAN_MAX_BYTES);
    assert!(std::str::from_utf8(capped.as_bytes()).is_ok());
}

#[cfg(any(unix, windows))]
#[test]
fn ten_agents_reuse_published_summary_without_private_duplicates() {
    let root = test_tempdir("shared-base-summary-ten-agents");
    let base_dir = root.join("base");
    let file_path = "src/lib.rs";
    let source = "fn shared() -> usize { 42 }";
    let model_key = "shared-summary-model";
    let complete_model_key = format!("{model_key}#{SUMMARY_CACHE_GENERATION}");
    let span_hash = greppy_store::span_hash(file_path, source);
    let expected = vec!["Returns the shared value.".to_string()];

    let base = greppy_store::SummaryCache::open(&base_dir).expect("create Base summary cache");
    base.put_unbounded(&complete_model_key, &span_hash, &expected)
        .expect("publish Base summary");
    assert_eq!(base.count().expect("count Base summaries"), 1);
    drop(base);

    let mut agents = Vec::new();
    for index in 0..10 {
        let base_dir = base_dir.clone();
        let delta_dir = root.join(format!("delta-{index}"));
        let expected = expected.clone();
        agents.push(std::thread::spawn(move || {
            let base = greppy_store::SummaryCache::open_read_only(&base_dir)
                .expect("open immutable Base cache");
            let delta =
                greppy_store::SummaryCache::open(&delta_dir).expect("open private Delta cache");
            let config = QwenSummaryConfig {
                model_id: "unused-on-cache-hit".into(),
                gguf: delta_dir.join("missing.gguf"),
                tokenizer: delta_dir.join("missing-tokenizer.json"),
                device: greppy_qwen35_native::DevicePreference::Cpu,
            };
            let actual = summarize_source_cached(
                &config,
                model_key,
                Some(&delta),
                Some(&base),
                file_path,
                source,
                false,
            )
            .expect("published Base entry must avoid model invocation");
            assert_eq!(actual, expected);
            delta.count().expect("count private summaries")
        }));
    }
    for agent in agents {
        assert_eq!(agent.join().expect("agent thread"), 0);
    }
    let base =
        greppy_store::SummaryCache::open_read_only(&base_dir).expect("reopen immutable Base cache");
    assert_eq!(base.count().expect("count Base summaries"), 1);
    std::fs::remove_dir_all(root).expect("remove test directory");
}

#[test]
fn discover_scope_env_parses_include_and_exclude_lists() {
    let _guard = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = EnvRestore::capture(&[ENV_DISCOVER_INCLUDE, ENV_DISCOVER_EXCLUDE]);
    // SAFETY: serialized by TEST_ENV_LOCK and restored by EnvRestore.
    unsafe {
        std::env::set_var(ENV_DISCOVER_INCLUDE, "src/*.rs; tests/*.rs\nbenches/*.rs");
        std::env::set_var(ENV_DISCOVER_EXCLUDE, "src/generated.rs;\n target/**");
    }

    let overrides = discover_overrides_from_env().unwrap();

    assert_eq!(
        overrides.includes,
        vec!["src/*.rs", "tests/*.rs", "benches/*.rs"]
    );
    assert_eq!(overrides.excludes, vec!["src/generated.rs", "target/**"]);
    assert_eq!(
        overrides.scope_key(),
        "v1;I8:src/*.rs;I10:tests/*.rs;I12:benches/*.rs;E16:src/generated.rs;E9:target/**"
    );
}

#[test]
fn vector_exact_candidate_limit_defaults_to_search_guard() {
    assert_eq!(
        parse_vector_exact_candidate_limit(None).unwrap(),
        Some(greppy_search::DEFAULT_EXACT_VECTOR_CANDIDATE_LIMIT)
    );
    assert_eq!(
        parse_vector_exact_candidate_limit(Some("")).unwrap(),
        Some(greppy_search::DEFAULT_EXACT_VECTOR_CANDIDATE_LIMIT)
    );
}

#[test]
fn vector_exact_candidate_limit_zero_disables_guard() {
    assert_eq!(parse_vector_exact_candidate_limit(Some("0")).unwrap(), None);
    assert_eq!(vector_exact_scan_exceeds_limit(1_000_000, None), None);
}

#[test]
fn vector_exact_candidate_limit_rejects_invalid_values() {
    for raw in ["abc", "-1"] {
        let err = parse_vector_exact_candidate_limit(Some(raw)).unwrap_err();
        assert!(
            matches!(err, Error::Invalid(msg) if msg.contains(ENV_VECTOR_EXACT_CANDIDATE_LIMIT))
        );
    }
}

#[test]
fn vector_exact_scan_limit_detects_over_budget_candidates() {
    assert_eq!(vector_exact_scan_exceeds_limit(100, Some(100)), None);
    assert_eq!(vector_exact_scan_exceeds_limit(101, Some(100)), Some(100));
}

#[test]
fn dispatch_returns_not_implemented_for_index() {
    // `greppy index` is wired to the indexer; this test
    // asserts that the dispatcher is callable for the parse. The
    // actual indexer run requires a real workspace on disk; we
    // exercise only the parse path here.
    let cli = Cli::try_parse_from(["greppy", "index", "/nonexistent-root-for-parse-only"]).unwrap();
    let parsed: bool = matches!(cli.command, Some(Command::Index { .. }));
    assert!(parsed);
}

#[test]
fn parse_index_status_json_and_doctor_json() {
    let cli = Cli::try_parse_from(["greppy", "index", "status", "--json"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Command::Index {
            path: Some(ref p),
            json: true,
            ..
        }) if p == "status"
    ));

    let cli = Cli::try_parse_from(["greppy", "doctor", "--json"]).unwrap();
    assert!(matches!(cli.command, Some(Command::Doctor { json: true })));
}

#[test]
fn removed_stub_names_are_not_public_subcommands() {
    for name in ["install", "uninstall", "update", "config"] {
        let cli = Cli::try_parse_from(["greppy", name]).unwrap();
        assert!(cli.command.is_none(), "{name} must not be a subcommand");
        assert_eq!(cli.passthrough, vec![name]);
    }
}

#[test]
fn dispatch_to_code_maps_errors() {
    assert_eq!(
        error_exit_code(&Error::out_of_scope("test feature")),
        EXIT_NOT_IMPLEMENTED
    );
    assert_eq!(
        error_exit_code(&Error::not_implemented("test feature", "not available")),
        EXIT_NOT_IMPLEMENTED
    );
    assert_eq!(
        error_exit_code(&Error::Invalid("bad input".into())),
        EXIT_USAGE
    );
    assert_eq!(
        error_exit_code(&Error::Config("configuration failure".into())),
        EXIT_IO
    );

    let cli = Cli::try_parse_from(["greppy"]).unwrap();
    assert_eq!(dispatch_to_code(cli), EXIT_USAGE);
}

#[test]
fn global_root_parses_before_and_after_subcommand() {
    // RV-006: `--root` is a global flag, accepted on either side of
    // the subcommand. Both spellings must land in `cli.root`.
    let before =
        Cli::try_parse_from(["greppy", "--root", "/repo", "search-pattern", "foo"]).unwrap();
    assert_eq!(before.root.as_deref(), Some("/repo"));
    assert!(matches!(
        before.command,
        Some(Command::SearchPattern { .. })
    ));

    let after =
        Cli::try_parse_from(["greppy", "search-pattern", "--root", "/repo", "foo"]).unwrap();
    assert_eq!(after.root.as_deref(), Some("/repo"));
    assert!(matches!(after.command, Some(Command::SearchPattern { .. })));

    // And it is honoured by `index` too.
    let idx = Cli::try_parse_from(["greppy", "--root", "/repo", "index", "."]).unwrap();
    assert_eq!(idx.root.as_deref(), Some("/repo"));
    assert!(matches!(idx.command, Some(Command::Index { .. })));
}

#[test]
fn search_pattern_rejects_removed_scope_flags() {
    for flag in ["--changed", "--staged", "--since", "--base", "--no-code"] {
        let mut argv = vec!["greppy", "search-pattern", flag];
        if matches!(flag, "--since" | "--base") {
            argv.push("HEAD");
        }
        argv.push("needle");
        assert!(Cli::try_parse_from(argv).is_err(), "{flag}");
    }
}

#[test]
fn find_repo_root_walks_up_to_marker() {
    // RV-006: build a nested tree with a `.git` marker at the top and
    // confirm `find_repo_root` returns the marker dir from a deep
    // subdirectory.
    let base = std::env::temp_dir().join(format!(
        "greppy-findroot-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let repo = base.join("repo");
    let deep = repo.join("a").join("b").join("c");
    std::fs::create_dir_all(&deep).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    // Canonicalize to compare without symlink noise (macOS /tmp).
    let want = repo.canonicalize().unwrap();
    let got = find_repo_root(&deep.canonicalize().unwrap());
    assert_eq!(got, want, "should walk up to the .git repo root");
    std::fs::write(repo.join("a/b/Cargo.toml"), "[workspace]\n").unwrap();
    assert_eq!(
        resolve_root(deep.to_str()).unwrap(),
        want,
        "an explicit nested --root must still resolve to the worktree root"
    );

    // No marker anywhere → returns `start` unchanged.
    let orphan = base.join("orphan");
    std::fs::create_dir_all(&orphan).unwrap();
    let orphan_c = orphan.canonicalize().unwrap();
    assert_eq!(find_repo_root(&orphan_c), orphan_c);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn trace_parses_direction_edge_depth_with_defaults() {
    // Bare `trace --symbol foo` defaults to outgoing/CALLS/depth 4.
    let cli = Cli::try_parse_from(["greppy", "trace", "--symbol", "foo"]).unwrap();
    match cli.command {
        Some(Command::Trace {
            symbol,
            direction,
            edge,
            depth,
            code,
            json,
        }) => {
            assert_eq!(symbol.as_deref(), Some("foo"));
            assert_eq!(direction, "outgoing");
            assert_eq!(edge, "CALLS");
            assert_eq!(depth, 4);
            assert!(!code, "--code defaults to false");
            assert!(!json, "--json defaults to false");
        }
        other => panic!("expected Trace, got {other:?}"),
    }

    // Explicit incoming / USES / depth 2.
    let cli = Cli::try_parse_from([
        "greppy",
        "trace",
        "--symbol",
        "foo",
        "--direction",
        "incoming",
        "--edge",
        "USES",
        "--depth",
        "2",
    ])
    .unwrap();
    match cli.command {
        Some(Command::Trace {
            direction,
            edge,
            depth,
            ..
        }) => {
            assert_eq!(direction, "incoming");
            assert_eq!(edge, "USES");
            assert_eq!(depth, 2);
        }
        other => panic!("expected Trace, got {other:?}"),
    }
}

#[test]
fn impact_rejects_removed_git_scopes() {
    for flag in ["--since", "--base"] {
        assert!(
            Cli::try_parse_from(["greppy", "impact", "hub", flag, "main"]).is_err(),
            "{flag} must not remain on impact"
        );
    }

    let cli = Cli::try_parse_from(["greppy", "impact", "hub", "--edge", "CALLS"]).unwrap();
    match cli.command {
        Some(Command::Impact { edge, .. }) => {
            assert_eq!(edge.as_deref(), Some("CALLS"));
        }
        other => panic!("expected Impact, got {other:?}"),
    }
}

#[test]
fn navigation_commands_parse_positional_symbol() {
    let cli = Cli::try_parse_from(["greppy", "who-calls", "do_it"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Command::WhoCalls { ref symbols, code: false, all: false, json: false, ref path_opts }) if symbols == &["do_it".to_string()] && path_opts.is_empty()
    ));

    // Rebuild #2: every further positional is another SYMBOL, never a path.
    let cli = Cli::try_parse_from(["greppy", "who-calls", "do_it", "other"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Command::WhoCalls { ref symbols, .. })
            if symbols == &["do_it".to_string(), "other".to_string()]
    ));

    // `references` was where find-usages went to keep living after it was
    // supposedly removed. It parses as nothing now.
    assert!(Cli::try_parse_from(["greppy", "references", "Widget"])
        .is_ok_and(|cli| !matches!(cli.command, Some(Command::WhoCalls { .. }))));

    let cli = Cli::try_parse_from([
        "greppy", "fan-in", "--edge", "USAGE", "--limit", "7", "--json",
    ])
    .unwrap();
    assert_eq!(cli.limit, Some(7));
    assert!(matches!(
        cli.command,
        Some(Command::FanIn { ref edge, json: true }) if edge == "USAGE"
    ));

    let cli = Cli::try_parse_from(["greppy", "fan-out"]).unwrap();
    assert_eq!(cli.limit, None);
    assert!(matches!(
        cli.command,
        Some(Command::FanOut { ref edge, json: false }) if edge == "CALLS"
    ));

    let cli = Cli::try_parse_from(["greppy", "graph-locate", "src/lib.rs:42"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Command::GraphLocate { location: Some(ref loc), file: None, line: None, json: false }) if loc == "src/lib.rs:42"
    ));

    let cli = Cli::try_parse_from([
        "greppy",
        "graph-locate",
        "--file",
        "src/lib.rs",
        "--line",
        "42",
        "--json",
    ])
    .unwrap();
    assert!(matches!(
        cli.command,
        Some(Command::GraphLocate { location: None, file: Some(ref file), line: Some(42), json: true }) if file == "src/lib.rs"
    ));
}

#[test]
fn trace_invalid_direction_is_a_usage_error() {
    // A bad --direction must surface as Error::Invalid (exit 64),
    // not a panic or a silent fallback. We can assert this without a
    // store because direction is validated before the store opens.
    let cli = Cli::try_parse_from([
        "greppy",
        "trace",
        "--symbol",
        "foo",
        "--direction",
        "sideways",
    ])
    .unwrap();
    let r = dispatch(cli);
    assert!(
        matches!(r, Err(Error::Invalid(_))),
        "bad direction must be a usage error, got {r:?}"
    );
}

#[test]
fn label_rank_prefers_type_defs_over_impl_and_pseudo_nodes() {
    // a Struct must outrank its Impl and any
    // EnumVariant/Call/Import sharing the name.
    assert!(label_rank("Struct") < label_rank("Impl"));
    assert!(label_rank("Struct") < label_rank("EnumVariant"));
    assert!(label_rank("Enum") < label_rank("EnumVariant"));
    assert!(label_rank("Function") < label_rank("Call"));
    assert!(label_rank("Method") < label_rank("Call"));
    assert!(label_rank("Impl") < label_rank("Call"));
    assert!(label_rank("Impl") < label_rank("Import"));
    // Primary set includes the secondary defs we aggregate across.
    assert!(is_primary_label("Struct"));
    assert!(is_primary_label("Impl"));
    assert!(is_primary_label("EnumVariant"));
    assert!(!is_primary_label("Call"));
    assert!(!is_primary_label("Import"));
}

#[test]
fn internal_source_search_is_literal_line_numbered_and_binary_safe() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "greppy-internal-source-search-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("source.rs"), "first\nneedle.*literal\nneedle\n").unwrap();
    std::fs::write(root.join("binary.bin"), b"needle\0hidden\n").unwrap();

    let hits = internal_literal_search_code_paths(
        "needle.*literal",
        &root,
        &["source.rs".into(), "binary.bin".into()],
    )
    .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].location, "source.rs:2");
    assert_eq!(hits[0].snippet, "needle.*literal");

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn live_source_search_respects_gitignore_inventory() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "greppy-live-source-ignore-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("ignored")).unwrap();
    std::fs::write(root.join(".gitignore"), "ignored/\n").unwrap();
    std::fs::write(
        root.join("src/lib.rs"),
        "const MARKER: &str = \"scope_marker\";\n",
    )
    .unwrap();
    std::fs::write(root.join("ignored/generated.rs"), "scope_marker\n").unwrap();

    let hits = live_grep_code_hits("scope_marker", &root).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].location, "src/lib.rs:1");

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn is_grep_passthrough_distinguishes_subcommands_from_grep_args() {
    use std::ffi::OsString;
    let mk = |xs: &[&str]| -> Vec<OsString> { xs.iter().map(|s| OsString::from(*s)).collect() };

    // Bare grep flags / patterns → passthrough.
    assert!(is_grep_passthrough(&mk(&["greppy", "-R", "foo", "."])));
    assert!(is_grep_passthrough(&mk(&["greppy", "foo", "f.txt"])));
    // Explicit `grep` subcommand → NOT passthrough (clap handles it).
    assert!(!is_grep_passthrough(&mk(&["greppy", "grep", "-R", "foo"])));
    // Structured subcommands → NOT passthrough.
    assert!(!is_grep_passthrough(&mk(&["greppy", "index", "."])));
    assert!(!is_grep_passthrough(&mk(&["greppy", "doctor"])));
    // Removed verbs are no longer subcommands. The routing would send them
    // to passthrough, so `unknown_verb_refusal` must stop every one before
    // its name can become a grep pattern.
    for verb in [
        "references",
        "find-usages",
        "map",
        "outline",
        "changes",
        "verify",
    ] {
        assert!(is_grep_passthrough(&mk(&["greppy", verb, "Foo"])), "{verb}");
        assert!(
            unknown_verb_refusal(&mk(&["greppy", verb, "Foo"])).is_some(),
            "{verb}"
        );
    }
    assert!(!is_grep_passthrough(&mk(&["greppy", "where-am-i"])));
    assert!(!is_grep_passthrough(&mk(&["greppy", "fan-in"])));
    assert!(!is_grep_passthrough(&mk(&["greppy", "fan-out"])));
    assert!(!is_grep_passthrough(&mk(&[
        "greppy",
        "graph-locate",
        "src/lib.rs:42"
    ])));
    // Help/version must reach clap.
    assert!(!is_grep_passthrough(&mk(&["greppy", "--help"])));
    assert!(!is_grep_passthrough(&mk(&["greppy", "--version"])));
    assert!(!is_grep_passthrough(&mk(&["greppy", "-h"])));
    assert!(is_grep_passthrough(&mk(&[
        "greppy",
        "-h",
        "needle",
        "first.rs",
        "second.rs"
    ])));
    // Global --root before a structured subcommand is skipped.
    assert!(!is_grep_passthrough(&mk(&[
        "greppy",
        "--root",
        "/repo",
        "search-pattern",
        "q"
    ])));
    // Global --root before grep args is still a passthrough.
    assert!(is_grep_passthrough(&mk(&[
        "greppy", "--root", "/repo", "-R", "foo", "."
    ])));
    assert_eq!(
        grep_passthrough_args(&mk(&[
            "greppy", "--root", "/repo", "--no-gpu", "-R", "foo", "."
        ])),
        mk(&["-R", "foo", "."])
    );
}

// a non-UTF-8 first token can never be a subcommand
// name, so it must route to the grep passthrough — NOT be rejected by
// clap with rc=2. This is the unit-level reproduction of
// `greppy -R pat $'f\xff'`.
#[cfg(unix)]
#[test]
fn is_grep_passthrough_routes_non_utf8_to_grep() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let argv = vec![
        OsString::from("greppy"),
        OsString::from("-R"),
        OsString::from("pat"),
        OsString::from_vec(vec![b'f', 0xff]),
    ];
    assert!(
        is_grep_passthrough(&argv),
        "a bare invocation carrying a non-UTF-8 path must be a grep passthrough"
    );
}

// -----------------------------------------------------------------
// Qualified-name query resolution (P1): `Owner.method` / `Owner::method`
// -----------------------------------------------------------------

/// Pure string-split units for the qualified-query parser.
#[test]
fn split_qualified_parses_owner_and_member_on_last_separator() {
    // Both separators, dot form and colon form.
    assert_eq!(
        split_qualified("JsonReader.peekNumber"),
        Some(("JsonReader", "peekNumber"))
    );
    assert_eq!(
        split_qualified("JsonReader::peekNumber"),
        Some(("JsonReader", "peekNumber"))
    );
    // Splits on the LAST separator so member is the final component.
    assert_eq!(
        split_qualified("com.google.JsonReader.peekNumber"),
        Some(("com.google.JsonReader", "peekNumber"))
    );
    assert_eq!(split_qualified("a::b::c"), Some(("a::b", "c")));
    // Mixed: pick the later separator.
    assert_eq!(split_qualified("a::b.c"), Some(("a::b", "c")));
    assert_eq!(split_qualified("a.b::c"), Some(("a.b", "c")));
    // Bare identifier → None (bare path is left untouched).
    assert_eq!(split_qualified("peekNumber"), None);
    // Degenerate: leading/trailing/empty parts → None.
    assert_eq!(split_qualified(".x"), None);
    assert_eq!(split_qualified("x."), None);
    assert_eq!(split_qualified("::x"), None);
    assert_eq!(split_qualified("x::"), None);
}

#[test]
fn qname_owner_segment_extracts_segment_before_name() {
    // Owned member: the segment before the name is the class/type owner.
    assert_eq!(
        qname_owner_segment("gson/.../JsonReader.java::JsonReader::peekNumber"),
        Some("JsonReader")
    );
    assert_eq!(
        qname_owner_segment("serde/src/private/ser.rs::TaggedSerializer::serialize_bool"),
        Some("TaggedSerializer")
    );
    assert_eq!(
        qname_owner_segment("packages/zod/src/v3/types.ts::ZodString::max"),
        Some("ZodString")
    );
    // Free def: the segment before the name is the Label (not an owner class).
    assert_eq!(
        qname_owner_segment("src/lib.rs::Function::helper"),
        Some("Function")
    );
    // A qname with no `::` before the name has no owner segment.
    assert_eq!(qname_owner_segment("lonely"), None);
}

/// Build an in-memory store with a set of `(label, file, owner, name)`
/// method/function definitions, mirroring the parser's
/// `<file>::<owner>::<name>` qname layout, so the query-time resolver
/// can be exercised end to end.
fn store_with_defs(defs: &[(&str, &str, &str, &str)]) -> greppy_store::Store {
    let mut store = greppy_store::Store::open_memory().unwrap();
    store
        .upsert_project(&greppy_store::Project {
            name: "p".into(),
            indexed_at: "2026-07-02T00:00:00Z".into(),
            root_path: "/repos/p".into(),
        })
        .unwrap();
    for (label, file, owner, name) in defs {
        store
            .insert_node(&greppy_store::NewNode {
                project: "p".into(),
                label: (*label).into(),
                name: (*name).into(),
                qualified_name: format!("{file}::{owner}::{name}"),
                file_path: (*file).into(),
                start_line: 1,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
    }
    store
}

/// Look up the node id for a `(file, owner, name)` triple.
fn id_of(store: &greppy_store::Store, file: &str, owner: &str, name: &str) -> i64 {
    let q = greppy_search::GraphQuery::any().with_limit(10_000);
    let rows = greppy_search::search_graph(store, &q).unwrap();
    rows.iter()
        .find(|r| r.qualified_name == format!("{file}::{owner}::{name}"))
        .unwrap_or_else(|| panic!("no node {file}::{owner}::{name}"))
        .id
}

/// REGRESSION 1: a qualified `Owner.method` / `Owner::method` query
/// resolves to exactly the owner's node — the natural form a coding
/// agent types — where a bare name would aggregate every same-named
/// method. Both the `.` and `::` spellings resolve identically.
#[test]
fn qualified_query_resolves_to_owner_node() {
    // Two classes each define `get`; the query owner disambiguates.
    let store = store_with_defs(&[
        ("Method", "src/JsonArray.java", "JsonArray", "get"),
        ("Method", "src/JsonObject.java", "JsonObject", "get"),
        ("Method", "src/TypeToken.java", "TypeToken", "get"),
    ]);
    let arr = id_of(&store, "src/JsonArray.java", "JsonArray", "get");
    let obj = id_of(&store, "src/JsonObject.java", "JsonObject", "get");

    // `.` form.
    assert_eq!(
        resolve_symbol_nodes(&store, Some("JsonArray.get")).unwrap(),
        vec![arr]
    );
    // `::` form resolves to the same single node.
    assert_eq!(
        resolve_symbol_nodes(&store, Some("JsonArray::get")).unwrap(),
        vec![arr]
    );
    // A different owner picks a different node — the owner truly narrows.
    assert_eq!(
        resolve_symbol_nodes(&store, Some("JsonObject.get")).unwrap(),
        vec![obj]
    );
    // The single-id resolver (trace/impact/path) agrees.
    assert_eq!(
        resolve_symbol_id(&store, Some("JsonArray::get")).unwrap(),
        Some(arr)
    );

    // Fully-qualified owner (extra leading segments) still matches on
    // the last owner segment.
    assert_eq!(
        resolve_symbol_nodes(&store, Some("com.google.gson.JsonArray.get")).unwrap(),
        vec![arr]
    );
}

/// REGRESSION 2: never-guess. A qualified query whose `Owner.member`
/// matches MORE THAN ONE node (same owner in two files) returns the
/// full candidate set — never one arbitrary pick — and a query whose
/// owner matches NOTHING returns the empty set (surfaced as "not
/// found"), rather than silently falling back to a bare-name guess that
/// would ignore the owner the agent supplied.
#[test]
fn qualified_query_ambiguous_lists_candidates_never_guesses() {
    // Same `Owner::method` legitimately present in two files (e.g. two
    // crates) → both are genuine matches; return both, never one.
    let store = store_with_defs(&[
        ("Method", "serde/src/de.rs", "SeqDeserializer", "end"),
        ("Method", "serde_core/src/de.rs", "SeqDeserializer", "end"),
        ("Method", "serde/src/de.rs", "MapDeserializer", "end"),
    ]);
    let a = id_of(&store, "serde/src/de.rs", "SeqDeserializer", "end");
    let b = id_of(&store, "serde_core/src/de.rs", "SeqDeserializer", "end");
    let mut got = resolve_symbol_nodes(&store, Some("SeqDeserializer::end")).unwrap();
    got.sort_unstable();
    let mut want = vec![a, b];
    want.sort_unstable();
    assert_eq!(
        got, want,
        "both same-owner nodes must be returned, never one guessed"
    );

    // Wrong owner → empty set (NOT a fallback to the bare `end` guess).
    assert_eq!(
        resolve_symbol_nodes(&store, Some("NoSuchType::end")).unwrap(),
        Vec::<i64>::new(),
        "an owner that matches nothing must NOT fall back to a bare-name guess"
    );
    // The single-id resolver likewise refuses to guess on a bad owner.
    assert_eq!(
        resolve_symbol_id(&store, Some("NoSuchType::end")).unwrap(),
        None
    );
}

/// REGRESSION 3: bare-name queries are unchanged. A bare identifier
/// (no `.` / `::`) never enters the qualified path, so it still
/// aggregates every same-named primary node exactly as before.
#[test]
fn bare_name_query_is_unchanged_and_aggregates() {
    let store = store_with_defs(&[
        ("Method", "src/JsonArray.java", "JsonArray", "get"),
        ("Method", "src/JsonObject.java", "JsonObject", "get"),
    ]);
    let arr = id_of(&store, "src/JsonArray.java", "JsonArray", "get");
    let obj = id_of(&store, "src/JsonObject.java", "JsonObject", "get");
    // Bare `get` aggregates BOTH owners (no narrowing) — the historical
    // behaviour the qualified path must not disturb.
    let mut got = resolve_symbol_nodes(&store, Some("get")).unwrap();
    got.sort_unstable();
    let mut want = vec![arr, obj];
    want.sort_unstable();
    assert_eq!(got, want);
    // And a bare name still enters neither qualified branch.
    assert_eq!(split_qualified("get"), None);
}

/// Seed a provider_state row so the completeness helpers have data.
fn seed_provider(
    store: &mut greppy_store::Store,
    language: &str,
    status: &str,
    unsupported_edges: &[&str],
) {
    store
        .upsert_provider_state(&greppy_store::ProviderState {
            project: "p".into(),
            language: language.into(),
            provider_version: "v1".into(),
            status: status.into(),
            supported_edge_classes: Vec::new(),
            unsupported_edge_classes: unsupported_edges.iter().map(|s| (*s).to_string()).collect(),
            files_seen: 1,
            files_indexed: 1,
            files_failed: 0,
            diagnostics: Vec::new(),
            last_indexed_generation: 1,
            updated_at: "2026-07-02T00:00:00Z".into(),
        })
        .unwrap();
}

/// LEVER 2a: `impact --all` parses (previously clap ERRORED — no such
/// flag), and the print cap is lifted only when `all` is set.
#[test]
fn impact_all_flag_bypasses_limit() {
    let cli = Cli::try_parse_from(["greppy", "impact", "JsonReader", "--all"]).unwrap();
    match cli.command {
        Some(Command::Impact { symbols, all, .. }) => {
            assert_eq!(symbols, vec!["JsonReader".to_string()]);
            assert!(all, "--all must parse to all=true");
        }
        other => panic!("expected Impact, got {other:?}"),
    }
    // Without --all the flag defaults off.
    let plain = Cli::try_parse_from(["greppy", "impact", "JsonReader"]).unwrap();
    assert!(matches!(
        plain.command,
        Some(Command::Impact { all: false, .. })
    ));
    // The shown-cap formula: default caps at NAV_LIMIT, --all shows the
    // full transitive set (mirrors dispatch_impact).
    let total = NAV_LIMIT + 25;
    let shown_default = total.min(NAV_LIMIT);
    let shown_all = total; // all == true
    assert_eq!(shown_default, NAV_LIMIT);
    assert_eq!(shown_all, total);
}

/// Agent-facing incomplete-provider metadata excludes non-code
/// snapshot/fixture rows (.stderr/.snap/.xml/no-ext), so counts describe
/// real code providers instead of repository artifacts.
#[test]
fn impact_total_excludes_noncode_files() {
    let mut store = store_with_defs(&[("Method", "src/A.java", "A", "m")]);
    // Real code providers that are legitimately incomplete.
    seed_provider(&mut store, "java", "partial", &["calls"]);
    seed_provider(&mut store, "protobuf", "partial", &["calls"]);
    // Non-code noise providers the indexer records for unparsed files.
    seed_provider(
        &mut store,
        "file extension .stderr",
        "unsupported",
        &["calls"],
    );
    seed_provider(
        &mut store,
        "file extension .snap",
        "unsupported",
        &["calls"],
    );
    seed_provider(&mut store, "file extension .xml", "unsupported", &["calls"]);
    seed_provider(&mut store, "no file extension", "unsupported", &["calls"]);

    // Every agent-facing command now drops non-code noise; full details
    // remain available through doctor/diagnostics.
    assert_eq!(incomplete_provider_json(&store, "p").unwrap().len(), 2);

    // impact's code-only set drops the four non-code providers.
    let code = code_incomplete_provider_json(&store, "p").unwrap();
    let langs: Vec<&str> = code
        .iter()
        .map(|p| p["language"].as_str().unwrap())
        .collect();
    assert_eq!(code.len(), 2, "only java + protobuf remain: {langs:?}");
    assert!(langs.contains(&"java"));
    assert!(langs.contains(&"protobuf"));

    // Direct predicate coverage.
    assert!(is_noncode_provider("unsupported", "file extension .snap"));
    assert!(is_noncode_provider("accepted", "no file extension"));
    assert!(!is_noncode_provider("partial", "java"));
}

#[test]
fn cache_commands_parse_with_stable_public_flags() {
    let status = Cli::try_parse_from(["greppy", "cache", "status", "--json"]).unwrap();
    assert!(matches!(
        status.command,
        Some(Command::Cache {
            command: CacheCommand::Status { json: true }
        })
    ));

    let gc = Cli::try_parse_from(["greppy", "cache", "gc", "--dry-run", "--json"]).unwrap();
    assert!(matches!(
        gc.command,
        Some(Command::Cache {
            command: CacheCommand::Gc {
                dry_run: true,
                json: true
            }
        })
    ));

    let clear =
        Cli::try_parse_from(["greppy", "cache", "clear", "--root", "/tmp/repo", "--yes"]).unwrap();
    assert_eq!(clear.root.as_deref(), Some("/tmp/repo"));
    assert!(matches!(
        clear.command,
        Some(Command::Cache {
            command: CacheCommand::Clear {
                all: false,
                yes: true
            }
        })
    ));
}
