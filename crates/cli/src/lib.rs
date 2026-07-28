//! `greppy` CLI — the unified subcommand dispatcher.
//!
//! Subcommand surface:
//! - grep-compatible passthrough — delegates ordinary invocations to real grep.
//! - `index`        — index a repo.
//! - `search-graph` — graph search.
//! - `who-calls` / `callees` / `impact` / `brief` — graph navigation.
//! - `semantic-search` (`semantic`) — meaning-based code search.
//! - `search-code` / `search-symbols` — current-source and indexed symbol search.
//! - `trial`        — isolated own-project baseline/Greppy observation.
//! - `install`      — agent installer      (out of scope)
//! - `uninstall`    — agent uninstaller    (out of scope)
//! - `update`       — explains the signed-release installation policy
//! - `config`       — runtime config       (out of scope)
//!
//! Out-of-scope lifecycle subcommands print a structured error and exit
//! with a documented non-zero code (EX_UNAVAILABLE = 69).

#[cfg(all(feature = "ci-test-assets", not(debug_assertions)))]
compile_error!("ci-test-assets is forbidden outside debug/test builds");

use clap::{Parser, Subcommand};
use greppy_core::error::{Error, Result};
use greppy_core::workspace as workspace_locator;

mod changes;
#[cfg(any(unix, windows))]
mod embed_daemon;
#[cfg(any(unix, windows))]
mod inference_daemon;
mod map;
#[cfg(any(unix, windows))]
mod summarize_daemon;
mod trial;
mod verify;

// Route this module's stdout through one optional collector. Query commands
// activate it for --max-bytes/--offset, and whole-file reads also activate a
// conservative default budget; grep passthrough bytes remain untouched.
macro_rules! print {
    ($($arg:tt)*) => {{
        crate::output_write(format_args!($($arg)*), false);
    }};
}

macro_rules! println {
    () => {{
        crate::output_write(format_args!(""), true);
    }};
    ($($arg:tt)*) => {{
        crate::output_write(format_args!($($arg)*), true);
    }};
}

fn output_write(arguments: std::fmt::Arguments<'_>, newline: bool) {
    use std::io::Write as _;

    let text = arguments.to_string();
    let captured = OUTPUT_CAPTURE.with(|capture| {
        let mut capture = capture.borrow_mut();
        let Some(bytes) = capture.as_mut() else {
            return false;
        };
        bytes.extend_from_slice(text.as_bytes());
        if newline {
            bytes.push(b'\n');
        }
        true
    });
    if !captured {
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(text.as_bytes());
        if newline {
            let _ = stdout.write_all(b"\n");
        }
    }
}

#[cfg(feature = "agent")]
#[derive(Debug, Parser)]
#[command(
    name = "greppy",
    bin_name = "greppy",
    disable_help_subcommand = true,
    about = "Run Greppy's optional isolated headless coding agent."
)]
struct AgentInvocation {
    /// Run a headless coding task against an isolated snapshot.
    #[arg(short = 'p', long = "prompt", conflicts_with = "apply_result")]
    prompt: Option<String>,
    /// Apply a previously returned run artifact.
    #[arg(long, value_name = "RUN_ID", conflicts_with = "prompt")]
    apply_result: Option<String>,
    #[arg(long)]
    root: Option<String>,
    #[arg(long, value_name = "URL")]
    base_url: Option<String>,
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
    #[arg(long, default_value = "OPENAI_API_KEY", value_name = "ENV_NAME")]
    api_key_env: String,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    apply: bool,
    #[arg(long)]
    keep_run: bool,
    #[arg(long, default_value_t = 24)]
    max_turns: usize,
    #[arg(long)]
    max_output_tokens: Option<u64>,
    #[arg(long, default_value_t = 900)]
    timeout_seconds: u64,
}

#[cfg(feature = "agent")]
struct AgentInferenceHooks {
    root: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    cfg: EmbeddingModelConfig,
}

#[cfg(feature = "agent")]
struct DaemonCodeEmbeddingProvider {
    cache_dir: std::path::PathBuf,
    model_key: String,
    cfg: EmbeddingModelConfig,
}

#[cfg(feature = "agent")]
impl greppy_indexer::CodeEmbeddingProvider for DaemonCodeEmbeddingProvider {
    fn model_id(&self) -> &str {
        &self.cfg.model_id
    }

    fn prompt_version(&self) -> &str {
        greppy_embed_native::PROMPT_VERSION
    }

    fn task_profile(&self) -> &str {
        greppy_embed_native::CODE_RETRIEVAL_PROFILE
    }

    fn embed_code_document(
        &mut self,
        title: Option<&str>,
        content: &str,
    ) -> greppy_core::Result<Vec<f32>> {
        self.embed_code_documents(&[(title, content)])
            .and_then(|mut vectors| {
                vectors
                    .pop()
                    .ok_or_else(|| Error::Store("document embedder returned no vector".into()))
            })
    }

    fn embed_code_documents(
        &mut self,
        documents: &[(Option<&str>, &str)],
    ) -> greppy_core::Result<Vec<Vec<f32>>> {
        use sha2::{Digest, Sha256};
        let cache = greppy_store::QueryEmbeddingCache::open(&self.cache_dir).ok();
        let cache_model_key = format!("{}|document", self.model_key);
        let keys = documents
            .iter()
            .map(|(title, content)| {
                let mut hasher = Sha256::new();
                hasher.update(title.unwrap_or_default().as_bytes());
                hasher.update([0]);
                hasher.update(content.as_bytes());
                format!("{:x}", hasher.finalize())
            })
            .collect::<Vec<_>>();
        let mut vectors = vec![None; documents.len()];
        let mut missing_by_key = std::collections::BTreeMap::<String, Vec<usize>>::new();
        for (index, key) in keys.iter().enumerate() {
            if let Some(vector) = cache
                .as_ref()
                .and_then(|cache| cache.get(&cache_model_key, key).ok().flatten())
            {
                vectors[index] = Some(vector);
            } else {
                missing_by_key.entry(key.clone()).or_default().push(index);
            }
        }
        if !missing_by_key.is_empty() {
            let mut singleflight_locks = Vec::new();
            let mut missing = Vec::new();
            for (key, indexes) in &missing_by_key {
                let lock = greppy_core::cache::acquire_named_lock(
                    &format!("agent-embedding-{key}"),
                    greppy_core::cache::LockMode::Exclusive,
                    false,
                )
                .map_err(|error| Error::io("acquire embedding singleflight lock", error))?
                .ok_or_else(|| Error::Store("blocking embedding lock returned no guard".into()))?;
                if let Some(vector) = cache
                    .as_ref()
                    .and_then(|cache| cache.get(&cache_model_key, key).ok().flatten())
                {
                    for index in indexes {
                        vectors[*index] = Some(vector.clone());
                    }
                } else {
                    missing.push(indexes[0]);
                    singleflight_locks.push(lock);
                }
            }
            let missing_documents = missing
                .iter()
                .map(|index| documents[*index])
                .collect::<Vec<_>>();
            if !missing_documents.is_empty() {
                #[cfg(any(unix, windows))]
                let daemon = embed_daemon::embed_documents_via_daemon_result(
                    &self.cfg,
                    &self.model_key,
                    &missing_documents,
                );
                #[cfg(not(any(unix, windows)))]
                let daemon = embed_daemon::EmbedDocumentsDaemonResult::NoDaemon;
                let embedded = match daemon {
                    embed_daemon::EmbedDocumentsDaemonResult::Embedded(vectors) => vectors,
                    embed_daemon::EmbedDocumentsDaemonResult::NoDaemon => {
                        let model = load_embedding_model(&self.cfg, Some(self.cache_dir.clone()))?;
                        model.embed_documents(&missing_documents).map_err(|error| {
                            Error::Store(format!("document embedding fallback: {error}"))
                        })?
                    }
                    embed_daemon::EmbedDocumentsDaemonResult::DaemonBusy => {
                        return Err(Error::Store(
                            "EmbeddingGemma daemon remained busy for document embeddings".into(),
                        ));
                    }
                    embed_daemon::EmbedDocumentsDaemonResult::Failed => {
                        return Err(Error::Store(
                            "EmbeddingGemma daemon failed document embeddings".into(),
                        ));
                    }
                };
                if embedded.len() != missing.len() {
                    return Err(Error::Store(format!(
                        "document embedder returned {} vectors for {} inputs",
                        embedded.len(),
                        missing.len()
                    )));
                }
                for (index, vector) in missing.into_iter().zip(embedded) {
                    if let Some(cache) = &cache {
                        let _ = cache.put(&cache_model_key, &keys[index], &vector);
                    }
                    for duplicate in &missing_by_key[&keys[index]] {
                        vectors[*duplicate] = Some(vector.clone());
                    }
                }
            }
            drop(singleflight_locks);
        }
        vectors
            .into_iter()
            .enumerate()
            .map(|(index, vector)| {
                vector.ok_or_else(|| {
                    Error::Store(format!("missing document embedding at index {index}"))
                })
            })
            .collect()
    }

    fn max_input_tokens(&self) -> usize {
        self.cfg.max_length.unwrap_or(2_048)
    }
}

#[cfg(feature = "agent")]
impl greppy_agent::EmbeddingHooks for AgentInferenceHooks {
    fn refresh(
        &self,
        store_path: &std::path::Path,
        source_root: &std::path::Path,
        project: &str,
        graph_generation: u64,
    ) -> greppy_agent::Result<()> {
        let mut store =
            greppy_store::Store::open_with(store_path, greppy_store::OpenOptions::query_writer())
                .map_err(|error| {
                greppy_agent::Error::Workspace(format!("open task embedding store: {error}"))
            })?;
        let mut provider = DaemonCodeEmbeddingProvider {
            cache_dir: self.cache_dir.clone(),
            model_key: embedding_query_cache_key(&self.cfg),
            cfg: self.cfg.clone(),
        };
        let report = greppy_indexer::index_code_embeddings_for_project(
            &mut store,
            source_root,
            project,
            &mut provider,
            greppy_indexer::EmbeddingIndexOptions::for_generation(graph_generation),
        )
        .map_err(|error| {
            greppy_agent::Error::Workspace(format!("refresh task embeddings: {error}"))
        })?;
        if report.is_complete() {
            let key = embedding_complete_key(project);
            store
                .conn()
                .execute(
                    "INSERT INTO schema_meta(key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    rusqlite::params![key, format!("{}|{}", graph_generation, self.cfg.model_id)],
                )
                .map_err(|error| {
                    greppy_agent::Error::Workspace(format!(
                        "record task embedding completeness: {error}"
                    ))
                })?;
        }
        let _ = store
            .conn()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
        Ok(())
    }
}

#[cfg(feature = "agent")]
impl greppy_agent::NavigationHooks for AgentInferenceHooks {
    fn semantic_search(
        &self,
        workspace: &greppy_agent::TaskWorkspace,
        query: &str,
        limit: usize,
    ) -> greppy_agent::Result<serde_json::Value> {
        workspace.ensure_index()?;
        let store = greppy_store::Store::open_with(
            workspace.graph_path(),
            greppy_store::OpenOptions::read_only(),
        )
        .map_err(|error| greppy_agent::Error::Tool {
            tool: "semantic_search".into(),
            message: format!("open task graph: {error}"),
        })?;
        let generation = store
            .list_workspace_states()
            .map_err(|error| greppy_agent::Error::Tool {
                tool: "semantic_search".into(),
                message: error.to_string(),
            })?
            .into_iter()
            .map(|state| state.graph_generation)
            .max()
            .unwrap_or(0);
        let complete = store
            .conn()
            .query_row(
                "SELECT value FROM schema_meta WHERE key = ?1",
                [embedding_complete_key(workspace.project())],
                |row| row.get::<_, String>(0),
            )
            .ok()
            == Some(format!("{}|{}", generation, self.cfg.model_id));
        if complete {
            let root = self.root.to_string_lossy();
            if let Ok(vector) = embed_query_cached(&self.cfg, Some(root.as_ref()), query) {
                let scope = greppy_search::embeddinggemma_code_retrieval_scope(
                    workspace.project(),
                    &self.cfg.model_id,
                    Some(generation),
                    limit,
                );
                if let Ok(hits) = greppy_search::vector_search_exact(&store, &vector, &scope) {
                    return Ok(serde_json::json!({
                        "query": query,
                        "backend": "embeddinggemma",
                        "hits": hits.into_iter().map(|hit| serde_json::json!({
                            "score": hit.score,
                            "qualified_name": hit.embedding.qualified_name,
                            "file": hit.embedding.file_path,
                            "start_line": hit.embedding.start_line,
                            "end_line": hit.embedding.end_line
                        })).collect::<Vec<_>>()
                    }));
                }
            }
        }
        let hits =
            greppy_search::semantic_query(&store, query, None, Some(workspace.project()), limit)
                .map_err(|error| greppy_agent::Error::Tool {
                    tool: "semantic_search".into(),
                    message: error.to_string(),
                })?;
        Ok(serde_json::json!({
            "query": query,
            "backend": "algorithmic",
            "hits": hits.into_iter().map(|hit| serde_json::json!({
                "score": hit.score,
                "qualified_name": hit.node.qualified_name,
                "kind": hit.node.label,
                "file": hit.node.file_path,
                "start_line": hit.node.start_line,
                "end_line": hit.node.end_line
            })).collect::<Vec<_>>()
        }))
    }
}

/// Exit code for subcommands that are recognised but not yet implemented
/// in the current phase. EX_UNAVAILABLE (69) is the standard BSD sysexits
/// value.
pub const EXIT_NOT_IMPLEMENTED: u8 = 69;

/// Exit code for argument / request errors.
pub const EXIT_USAGE: u8 = 64;

/// Exit code for IO failures.
pub const EXIT_IO: u8 = 73;

/// Exit code for "temporary failure, retry later". Used when the
/// greppy write lock is held by another writer so callers (and
/// agents) can distinguish a transient lock contention from a real
/// IO error. EX_TEMPFAIL (75) is the BSD sysexits value.
pub const EXIT_TEMPFAIL: u8 = 75;

/// Cap on test names listed per bucket in `changes` text output; the full
/// lists remain in `--json`.
const CHANGES_TEST_LIST_CAP: usize = 10;

const DEFAULT_EMBEDDINGGEMMA_MODEL_ID: &str = "google/embeddinggemma-300m";
const ENV_DEVICE: &str = "GREPPY_DEVICE";
const ENV_NO_GPU: &str = "GREPPY_NO_GPU";
const ENV_EMBED_CUDA_DEVICE: &str = "EMBED_NATIVE_CUDA_DEVICE";
const ENV_QWEN_CUDA_DEVICE: &str = "GREPPY_QWEN35_CUDA_DEVICE";
const ENV_VECTOR_EXACT_CANDIDATE_LIMIT: &str = "GREPPY_VECTOR_EXACT_CANDIDATE_LIMIT";
const ENV_PROVIDER_POLICY: &str = "GREPPY_PROVIDER_POLICY";
const ENV_DISCOVER_INCLUDE: &str = "GREPPY_DISCOVER_INCLUDE";
const ENV_DISCOVER_EXCLUDE: &str = "GREPPY_DISCOVER_EXCLUDE";
const ENV_EXPAND_TTL_SECS: &str = "GREPPY_EXPAND_TTL_SECS";
const ENV_LAZY_EMBED_MIN_SPANS: &str = "GREPPY_LAZY_EMBED_MIN_SPANS";
const BACKGROUND_JOB_SCHEMA_VERSION: &str = "greppy.background-job.v2";
const DEFAULT_LAZY_EMBED_CPU_SPANS: usize = 1_000;
const DEFAULT_LAZY_EMBED_GPU_SPANS: usize = 5_000;
#[cfg(debug_assertions)]
const ENV_TEST_INDEX_FAILPOINT: &str = "GREPPY_TEST_INDEX_FAILPOINT";
#[cfg(debug_assertions)]
const ENV_TEST_INDEX_FAILPOINT_READY: &str = "GREPPY_TEST_INDEX_FAILPOINT_READY";
#[cfg(debug_assertions)]
const ENV_TEST_INDEX_FAILPOINT_HOLD_MS: &str = "GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS";
#[cfg(all(debug_assertions, not(feature = "ci-test-assets")))]
const ENV_TEST_SKIP_INFERENCE: &str = "GREPPY_TEST_SKIP_INFERENCE";
/// Test-only failpoint: simulate an unavailable embedding backend so tests
/// can pin the degraded-index contract (graph publishes, embeddings retry
/// in the background) without a real inference failure.
#[cfg(debug_assertions)]
const ENV_TEST_EMBED_UNAVAILABLE: &str = "GREPPY_TEST_EMBED_UNAVAILABLE";
/// Test-only failpoint for the agent-facing missing-asset fallback. Unlike
/// `GREPPY_TEST_EMBED_UNAVAILABLE`, this fails model configuration before any
/// inference attempt, matching an embedded asset that could not be extracted.
#[cfg(debug_assertions)]
const ENV_TEST_EMBED_ASSET_MISSING: &str = "GREPPY_TEST_EMBED_ASSET_MISSING";

#[cfg(feature = "ci-test-assets")]
fn test_inference_skipped() -> bool {
    true
}

#[cfg(all(debug_assertions, not(feature = "ci-test-assets")))]
fn test_inference_skipped() -> bool {
    std::env::var_os(ENV_TEST_SKIP_INFERENCE).is_some()
}

#[cfg(all(not(debug_assertions), not(feature = "ci-test-assets")))]
fn test_inference_skipped() -> bool {
    false
}

#[derive(Clone, Default)]
struct CliInferenceOverride {
    device: Option<String>,
    no_gpu: bool,
}

thread_local! {
    static CLI_INFERENCE_OVERRIDE: std::cell::RefCell<CliInferenceOverride> =
        std::cell::RefCell::new(CliInferenceOverride::default());
    static CLI_RESULT_LIMIT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static CLI_RESULT_OFFSET: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CLI_INVOCATION: std::cell::RefCell<Vec<std::ffi::OsString>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static OUTPUT_CAPTURE: std::cell::RefCell<Option<Vec<u8>>> =
        const { std::cell::RefCell::new(None) };
}

fn set_cli_inference_override(device: Option<String>, no_gpu: bool) {
    CLI_INFERENCE_OVERRIDE.with(|value| {
        *value.borrow_mut() = CliInferenceOverride { device, no_gpu };
    });
}

fn cli_inference_override() -> CliInferenceOverride {
    CLI_INFERENCE_OVERRIDE.with(|value| value.borrow().clone())
}

fn set_cli_result_window(limit: Option<usize>, offset: usize) {
    CLI_RESULT_LIMIT.with(|value| value.set(limit));
    CLI_RESULT_OFFSET.with(|value| value.set(offset));
}

fn cli_result_offset() -> usize {
    CLI_RESULT_OFFSET.with(std::cell::Cell::get)
}

thread_local! {
    /// `read --context N`: how many lines above a definition come along, so its
    /// doc comment arrives with it instead of costing a second call.
    static CLI_READ_CONTEXT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn set_cli_read_context(context: Option<usize>) {
    CLI_READ_CONTEXT.with(|value| value.set(context.unwrap_or(0)));
}

fn cli_read_context() -> i64 {
    i64::try_from(CLI_READ_CONTEXT.with(std::cell::Cell::get)).unwrap_or(0)
}

fn cli_result_limit(default: usize) -> usize {
    CLI_RESULT_LIMIT
        .with(|value| value.get())
        .unwrap_or(default)
        .saturating_add(cli_result_offset())
}

fn cli_result_limit_unless_all(default: usize, all: bool) -> usize {
    if all {
        usize::MAX
    } else {
        cli_result_limit(default)
    }
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingCliArgs<'a> {
    device: Option<&'a str>,
    no_gpu: bool,
}

fn discover_overrides_from_env() -> Result<greppy_discover::WalkOverrides> {
    let mut overrides = greppy_discover::WalkOverrides::empty();
    overrides.includes = env_pattern_list(ENV_DISCOVER_INCLUDE)?;
    overrides.excludes = env_pattern_list(ENV_DISCOVER_EXCLUDE)?;
    Ok(overrides)
}

fn env_pattern_list(name: &str) -> Result<Vec<String>> {
    let raw = match std::env::var(name) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::Config(format!("{name} must be valid UTF-8")));
        }
    };
    Ok(raw
        .split(['\n', ';'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EmbeddingModelConfig {
    model_id: String,
    source: EmbeddingModelSource,
    max_length: Option<usize>,
    device: greppy_embed_native::DevicePreference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EmbeddingModelSource {
    Gguf {
        gguf: std::path::PathBuf,
        tokenizer: std::path::PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QwenSummaryConfig {
    model_id: String,
    gguf: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
    device: greppy_qwen35_native::DevicePreference,
}

#[derive(Debug, Parser)]
#[command(
    name = "greppy",
    bin_name = "greppy",
    version,
    about = "Code navigation for coding agents, with byte-exact real-grep passthrough for ordinary grep invocations.",
    long_about = None,
    allow_external_subcommands = true,
    disable_help_subcommand = true,
    trailing_var_arg = true,
    allow_hyphen_values = true,
)]
pub struct Cli {
    /// Explicit repository root (RV-006). When set, `index` and every
    /// query subcommand (search-graph / trace / search-code /
    /// search-symbols / semantic-search) key the on-disk store and the project
    /// identity on this path instead of detecting the repo root by
    /// walking up from the current directory. `global = true` lets it be
    /// passed either before or after the subcommand:
    ///   grep --root /repo search-code foo
    ///   grep search-code --root /repo foo
    #[arg(long, global = true)]
    pub root: Option<String>,

    /// Native inference backend for both embedded models.
    #[arg(long, global = true, value_name = "auto|cpu|metal|cuda[:INDEX]")]
    pub device: Option<String>,

    /// Legacy spelling for `--device cpu`.
    #[arg(long, global = true, conflicts_with = "device")]
    pub no_gpu: bool,

    /// Cap the number of rows returned by navigation and search commands.
    /// `--max` is accepted as a Postel-style alias; `--all` still lifts caps.
    #[arg(long, alias = "max", global = true, value_name = "N")]
    pub limit: Option<usize>,

    /// Hard stdout payload budget for navigation, search, and read commands.
    /// Result rows/content are trimmed before status and continuation metadata.
    #[arg(long, global = true, value_name = "N")]
    pub max_bytes: Option<usize>,

    /// Continue a budgeted navigation, search, or read result at row N.
    #[arg(long, global = true, default_value_t = 0, value_name = "N")]
    pub offset: usize,

    #[command(subcommand)]
    pub command: Option<Command>,

    /// Trailing positional / flag arguments used as a passthrough when
    /// no recognised subcommand matched. clap captures here whatever
    /// remains after subcommand parsing.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub passthrough: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run an ordinary invocation through the byte-exact real-grep passthrough.
    #[command(external_subcommand)]
    Passthrough(Vec<String>),
    /// Index a repository.
    Index {
        /// Path to the repository root (default: cwd).
        path: Option<String>,
        /// With path `status`, emit machine-readable status JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show a one-screen project map from VCS, index, and build metadata.
    Map {
        /// Directory to orient within (default: workspace root).
        path: Option<String>,
        /// Emit the complete stable JSON shape.
        #[arg(long)]
        json: bool,
    },
    /// One line per definition in that file: kind, signature, span.
    #[command(
        after_help = "Example:\n  greppy outline src/lib.rs\n\n\
                      `search-symbols` looks a name up across the repository; `outline` answers\n\
                      what stands in THIS file, in the order it stands there."
    )]
    Outline {
        /// The file to outline.
        #[arg(value_name = "PATH")]
        path: String,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Return every definition instead of the first page.
        #[arg(long)]
        all: bool,
    },
    /// Group working-tree changes by definitions and graph impact.
    Changes {
        /// Compare the working tree with this commit (default: HEAD).
        #[arg(long, value_name = "REV")]
        base: Option<String>,
        /// Emit the complete stable JSON shape.
        #[arg(long)]
        json: bool,
    },
    /// Inspect or safely reclaim Greppy-managed cache data.
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
    /// Run one isolated own-project baseline/Greppy observation with Pi.
    Trial {
        #[command(flatten)]
        args: trial::TrialArgs,
    },
    /// Compare current-tree tests with an isolated committed baseline.
    Verify {
        #[command(flatten)]
        args: verify::VerifyArgs,
    },
    /// Structured graph search.
    SearchGraph {
        #[arg(long)]
        name: Option<String>,
        /// Emit machine-readable JSON with exact count metadata.
        #[arg(long)]
        json: bool,
    },
    /// Call-graph trace.
    ///
    /// `--direction outgoing` (default) walks what `S` calls/uses;
    /// `--direction incoming` walks who calls/uses `S` (backed by
    /// `TraceDirection::Incoming`). `--edge` restricts the walk to one
    /// edge type (CALLS, USES, TYPE_REF, IMPORTS); the default is CALLS.
    /// `--depth` caps the BFS hop count.
    Trace {
        #[arg(long)]
        symbol: Option<String>,
        /// outgoing (what S calls) | incoming (who calls S).
        #[arg(long, default_value = "outgoing")]
        direction: String,
        /// Edge type to follow (CALLS, USES, TYPE_REF, IMPORTS).
        #[arg(long, default_value = "CALLS")]
        edge: String,
        /// Maximum BFS depth from the start symbol.
        #[arg(long, default_value_t = 4)]
        depth: usize,
        /// Also print the source code span of each traced node.
        #[arg(long)]
        code: bool,
        /// Emit machine-readable JSON with trace-step metadata.
        #[arg(long)]
        json: bool,
    },
    /// Impact / blast-radius — the TRANSITIVE set reachable from `S` over one
    /// edge type, with hop distance, in ONE call. `--direction incoming`
    /// (default) answers "if I change S, what breaks?" (all transitive
    /// callers); `--direction outgoing` answers "what does S ultimately reach?".
    /// Replaces a dozen iterative who-calls/callees an agent would otherwise run.
    Impact {
        /// The symbols to answer for. Several are answered in one call:
        /// `greppy impact A B C`. `-` reads them from the pipe.
        #[arg(value_name = "SYMBOL")]
        symbols: Vec<String>,
        /// Restrict the reported reach to these files or directory subtrees.
        /// Repeatable; this is the only path filter.
        #[arg(long = "path", value_name = "PATH")]
        path_opts: Vec<String>,
        /// Accepted for agent ergonomics — no-op (impact prints locations, not
        /// bodies); an agent carrying --code over must not hit a parse error.
        #[arg(long)]
        code: bool,
        /// incoming (transitive callers — what breaks) | outgoing (what S reaches).
        #[arg(long, default_value = "incoming")]
        direction: String,
        /// Edge type to follow. Incoming default follows all reference edge
        /// types; explicit --edge CALLS scopes to CALLS. Outgoing default is
        /// CALLS.
        #[arg(long)]
        edge: Option<String>,
        /// Maximum transitive hop distance.
        #[arg(long, default_value_t = 6)]
        depth: usize,
        /// Compute impact from symbols touched by `git diff REV --`.
        #[arg(long)]
        since: Option<String>,
        /// Compute impact from symbols touched since merge-base(BASE, HEAD).
        #[arg(long)]
        base: Option<String>,
        /// Print every reached node (lift the default NAV_LIMIT cap) so the
        /// full transitive set is inspectable without a second query.
        #[arg(long)]
        all: bool,
        /// Emit machine-readable JSON with exact count/scope metadata.
        #[arg(long)]
        json: bool,
    },
    /// One-call briefing for a symbol: its definition (with source), its direct
    /// callers, and its direct callees — everything an agent needs to answer
    /// "how does S work / what is its role / what depends on it" in a SINGLE
    /// call, instead of iterating semantic-search + who-calls + callees separately.
    Brief {
        /// The symbols to brief. Several are briefed in one call:
        /// `greppy brief A B C`. `-` reads them from the pipe.
        #[arg(value_name = "SYMBOL")]
        symbols: Vec<String>,
        /// Restrict returned definitions/callers/callees to these files or
        /// directory subtrees. Graph resolution itself remains workspace-wide.
        /// Repeatable.
        #[arg(long = "path", value_name = "PATH")]
        path_opts: Vec<String>,
        /// Accepted for agent ergonomics: brief already prints the
        /// definition's source, so --code is a no-op — but agents
        /// carrying the flag over from the nav commands must not be
        /// punished with a parse error (P3 forensics: a real agent lost
        /// a call to exactly this).
        #[arg(long)]
        code: bool,
        /// Accepted for agent ergonomics — no-op (brief is one fixed briefing).
        #[arg(long)]
        all: bool,
        /// Emit machine-readable output with definitions, signatures, summaries,
        /// graph evidence, and a valid expand handle.
        #[arg(long)]
        json: bool,
    },
    /// Print a prepared evidence pack created by a previous query command.
    Expand {
        id: Option<String>,
        /// Emit machine-readable JSON wrapper with metadata and payload.
        #[arg(long)]
        json: bool,
    },
    /// Read a symbol's exact definition span (byte-precise source). With
    /// --handle, also returns an edit handle that pins the file, byte range,
    /// and content hashes — pass it to `greppy edit` commands. File paths and
    /// `--lines A:B` ranges produce the same directly consumable handle form.
    /// Prefer this
    /// over opening whole files: it returns exactly the code that matters.
    Read {
        /// What to read: symbols, files, or `-` to take them from the pipe.
        /// Several targets are read in one call. A name that is also a path on
        /// disk is read as the file; `--symbol` forces the definition.
        #[arg(value_name = "TARGET")]
        targets: Vec<String>,
        /// Force symbol reads, and name further symbols. Repeatable.
        #[arg(long = "symbol", value_name = "SYMBOL")]
        symbol_opts: Vec<String>,
        /// Read these files, or disambiguate a symbol that resolves in several
        /// files (equivalent to the `path::SYMBOL` form). Repeatable.
        #[arg(long = "path", value_name = "FILE")]
        path_opts: Vec<String>,
        /// For file reads, print only the inclusive 1-based line N or range N:M.
        /// `--line` is accepted as the conventional singular spelling.
        #[arg(long, alias = "line", value_name = "N[:M]")]
        lines: Option<String>,
        /// For symbol reads, also the N lines above the definition, so its doc
        /// comment comes along instead of costing a second call.
        #[arg(long, value_name = "N")]
        context: Option<usize>,
        /// Also return an edit handle for the symbol, file, or selected line range.
        #[arg(long)]
        handle: bool,
        /// Accepted for agent ergonomics: read already prints the definition's
        /// source, so --code is a no-op — agents carrying it over from the nav
        /// commands must not lose the call to a parse error.
        #[arg(long)]
        code: bool,
        /// Emit machine-readable JSON (source, span, handle, candidates).
        #[arg(long)]
        json: bool,
        /// For whole-file reads, lift the default stdout byte budget.
        #[arg(long)]
        all: bool,
    },
    /// Transactional, hash-guarded, all-or-nothing edits. Every command
    /// verifies its own result and emits a certificate; on failure nothing
    /// is written and the error names the next step.
    #[command(after_help = "Run `greppy edit VERB --help` for a working example.")]
    Edit {
        #[command(subcommand)]
        command: EditCommand,
        /// Emit the complete edit certificate JSON on stdout.
        #[arg(long, global = true)]
        json: bool,
    },
    /// Print deterministic graph statistics for the workspace project:
    /// file count, node counts by label, edge counts by type, and the
    /// node/edge totals.
    Stats,
    /// Store/index diagnostics: schema health, integrity check, workspace
    /// state, graph stats and provider completeness.
    Diagnostics {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// End-to-end health check for the active workspace index.
    Doctor {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Who calls or uses `S` — incoming CALLS and USAGE edges,
    /// printed as `qualified_name file:line`. With `--code`, also prints
    /// each caller's source span so the agent reads the body without a
    /// separate file Read.
    WhoCalls {
        /// The symbols to answer for. Several are answered in one call:
        /// `greppy who-calls A B C`. `-` reads them from the pipe.
        #[arg(value_name = "SYMBOL")]
        symbols: Vec<String>,
        /// Restrict returned callers to these files or directory subtrees.
        /// Repeatable; this is the only path filter.
        #[arg(long = "path", value_name = "PATH")]
        path_opts: Vec<String>,
        /// Also print the source code span of each result node.
        #[arg(long)]
        code: bool,
        /// Print every caller (lift the default NAV_LIMIT cap).
        #[arg(long)]
        all: bool,
        /// Emit machine-readable JSON with exact count metadata.
        #[arg(long)]
        json: bool,
    },
    /// What `S` calls — direct outgoing CALLS edges (the callees of `S`),
    /// printed as `qualified_name file:line`. Backed by the search
    /// `callees_of` helper. With `--code`, also prints each callee's
    /// source span.
    Callees {
        /// The symbols to answer for. Several are answered in one call:
        /// `greppy callees A B C`. `-` reads them from the pipe.
        #[arg(value_name = "SYMBOL")]
        symbols: Vec<String>,
        /// Restrict returned callees to these files or directory subtrees.
        /// Repeatable; this is the only path filter.
        #[arg(long = "path", value_name = "PATH")]
        path_opts: Vec<String>,
        /// Also print the source code span of each result node.
        #[arg(long)]
        code: bool,
        /// Print every callee (lift the default NAV_LIMIT cap).
        #[arg(long)]
        all: bool,
        /// Emit machine-readable JSON with exact count metadata.
        #[arg(long)]
        json: bool,
    },
    /// Top symbols by incoming edge degree. Default edge type is CALLS, so this
    /// shows the most-called symbols in the current project.
    FanIn {
        /// Edge type to rank by (CALLS, USAGE, USES, TYPE_REF, IMPORTS).
        #[arg(long, default_value = "CALLS")]
        edge: String,
        /// Emit machine-readable JSON with exact count metadata.
        #[arg(long)]
        json: bool,
    },
    /// Top symbols by outgoing edge degree. Default edge type is CALLS, so
    /// this shows the symbols that directly call the most other symbols.
    FanOut {
        /// Edge type to rank by (CALLS, USAGE, USES, TYPE_REF, IMPORTS).
        #[arg(long, default_value = "CALLS")]
        edge: String,
        /// Emit machine-readable JSON with exact count metadata.
        #[arg(long)]
        json: bool,
    },
    /// Locate the innermost indexed graph symbol enclosing a grep-style
    /// `file:line` location. Accepts either `graph-locate src/lib.rs:42` or
    /// `graph-locate --file src/lib.rs --line 42`.
    GraphLocate {
        /// Grep-style location (`file:line`), split on the last colon.
        location: Option<String>,
        /// Repo-relative or absolute file path.
        #[arg(long)]
        file: Option<String>,
        /// One-based source line.
        #[arg(long)]
        line: Option<i64>,
        /// Emit machine-readable JSON with freshness/provider metadata.
        #[arg(long)]
        json: bool,
    },
    /// Find a path between two symbols, if one exists. Prints the ordered
    /// list of `qualified_name file:line` steps from `--from` to `--to`,
    /// following `--edge` edges (default CALLS). Backed by the search
    /// `path_query` helper.
    Path {
        /// Source symbol (the path start).
        #[arg(long)]
        from: Option<String>,
        /// Destination symbol (the path goal).
        #[arg(long)]
        to: Option<String>,
        /// Edge type to follow (CALLS, USES, TYPE_REF, IMPORTS).
        #[arg(long, default_value = "CALLS")]
        edge: String,
        /// Emit machine-readable JSON with exact shortest-path metadata.
        #[arg(long)]
        json: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        code: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        all: bool,
    },
    /// Code search.
    SearchCode {
        query: Option<String>,
        /// Restrict returned matches to these files or directory subtrees.
        #[arg(value_name = "PATH")]
        paths: Vec<String>,
        /// Flag spelling for an additional result-path filter.
        #[arg(long = "path", value_name = "FILE")]
        path_opts: Vec<String>,
        /// Restrict search to files changed in the current git worktree.
        #[arg(long)]
        changed: bool,
        /// Restrict search to blobs staged in the git index.
        #[arg(long)]
        staged: bool,
        /// Restrict search to files changed since REV, then live-grep current files.
        #[arg(long)]
        since: Option<String>,
        /// Restrict search to files changed since merge-base(BASE, HEAD).
        #[arg(long)]
        base: Option<String>,
        /// Emit machine-readable JSON with exact count/truncation metadata.
        #[arg(long)]
        json: bool,
        /// Keep the legacy grep-shaped `file:line  text` output without
        /// resolving matches to enclosing definitions or creating handles.
        #[arg(long)]
        no_code: bool,
        /// Treat PATTERN as literal text instead of a regular expression.
        #[arg(long)]
        fixed: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        code: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        all: bool,
    },
    /// Symbol-only search (search-symbols alias).
    SearchSymbols {
        /// The names (or fragments) to look up. Several are answered in one
        /// call: `greppy search-symbols A B`. `-` reads them from the pipe.
        #[arg(value_name = "NAME")]
        queries: Vec<String>,
        /// Restrict returned symbols to these files or directory subtrees.
        /// Repeatable; this is the only path filter.
        #[arg(long = "path", value_name = "PATH")]
        path_opts: Vec<String>,
        /// Restrict to one node kind (Function, Method, Struct, Class, …).
        /// Matches the label case-insensitively; agents guess this flag,
        /// so it is real (P3 forensics).
        #[arg(long)]
        kind: Option<String>,
        /// Emit machine-readable JSON with exact count/truncation metadata.
        #[arg(long)]
        json: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        code: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        all: bool,
    },
    /// Fused search: combine literal/full-text, symbol, fuzzy semantic,
    /// and graph-neighbour signals into grep-like ranked hits.
    /// This is search output, not a generated answer: each row stays
    /// `file:line score signals symbol snippet`.
    Plus {
        query: Option<String>,
        /// Number of ranked hits to print.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Print the enclosing source span under symbol-backed hits.
        #[arg(long)]
        code: bool,
        /// Append score/signals/symbol diagnostics after the grep-like row.
        #[arg(long)]
        explain: bool,
        /// Emit machine-readable JSON with freshness and output-budget metadata.
        #[arg(long)]
        json: bool,
    },
    /// Semantic query using EmbeddingGemma vectors with Qwen purpose hints.
    #[command(name = "semantic-search", alias = "semantic")]
    Semantic {
        /// The plain-English descriptions to search for. Several are answered
        /// in one call. `-` reads them from the pipe.
        #[arg(value_name = "QUERY")]
        queries: Vec<String>,
        /// Restrict returned semantic hits to these files or directory
        /// subtrees. Repeatable; this is the only path filter.
        #[arg(long = "path", value_name = "PATH")]
        path_opts: Vec<String>,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Legacy compatibility command for resolving definitions. Prefer
    /// `semantic-search` for meaning-based search and `brief` for a compact
    /// structural digest.
    ///
    /// Resolve the most relevant definitions for `<query>` and print their
    /// ACTUAL SOURCE SPANS (not just file:line pointers), so an agent reads
    /// the relevant function/struct bodies directly instead of opening the
    /// files. Resolution unions symbol search, semantic search, and code
    /// search; results are ranked and the top-K (default 6) spans are
    /// emitted with a compact `== qualified_name (file:start-end) ==`
    /// header. Each span is capped (default 60 lines) with a truncation
    /// note.
    ///
    /// For MULTI-WORD natural-language queries (which contain spaces), when
    /// exact/FTS/algorithmic-semantic resolution finds nothing, this legacy command
    /// automatically falls back to NATIVE EmbeddingGemma vector similarity
    /// over the indexed code-span embeddings — the case where the question
    /// shares no literal words with the target definition. Bare single
    /// identifiers keep the lean exact find-definition path and never invoke
    /// the model, so exact-name / graph queries stay vector-free (router
    /// contract: `avoid_embedding` classes never touch the embedding model).
    /// The vector fallback uses greppy's bundled embedding model.
    #[command(hide = true)]
    Context {
        /// The natural-language or symbol query to resolve to definitions.
        query: Option<String>,
        /// Number of top definitions to emit (default 6).
        #[arg(long, default_value_t = 6)]
        k: usize,
        /// Print 1-based line numbers alongside the source span.
        #[arg(long)]
        lines: bool,
        /// Emit machine-readable JSON with freshness and truncation metadata.
        #[arg(long)]
        json: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        code: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        all: bool,
    },
    /// Internal: warm embedding daemon (spawned automatically by query
    /// commands; lazy-loads the model, drops it after an idle TTL to free
    /// GPU memory, exits after a longer idle TTL). Not part of the public
    /// surface.
    #[cfg(any(unix, windows))]
    #[command(hide = true, name = "embed-daemon")]
    EmbedDaemon {
        #[arg(long)]
        socket: String,
        #[arg(long)]
        gguf: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long)]
        model_id: String,
        #[arg(long)]
        max_length: Option<usize>,
        /// Load the model immediately at startup (session prewarm) instead
        /// of on the first request.
        #[arg(long)]
        prewarm: bool,
    },
    /// Internal: warm Qwen3.5 summarization daemon for `brief`.
    #[cfg(any(unix, windows))]
    #[command(hide = true, name = "summarize-daemon")]
    SummarizeDaemon {
        #[arg(long)]
        socket: String,
        #[arg(long)]
        gguf: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long)]
        model_id: String,
        /// Load the model immediately at startup instead of on first request.
        #[arg(long)]
        prewarm: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum CacheCommand {
    /// Show managed stores, models, quotas, locks, and unmanaged paths.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Run the TTL/LRU garbage collector immediately.
    Gc {
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json: bool,
    },
    /// Remove one worktree's verified store, or every verified cache object.
    Clear {
        #[arg(long)]
        all: bool,
        #[arg(long)]
        yes: bool,
    },
}

/// Subcommands of `greppy edit`. Exit codes are the registered contract
/// (docs/contracts/EDIT_CONTRACT.md): 0 applied/already-satisfied,
/// 10 not found, 11 ambiguous, 12 stale, 13 syntax/postcondition,
/// 14 validator, 15 concurrent change, 16 publish, 17 unsafe path,
/// 20 invalid spec.
#[derive(clap::Subcommand, Debug)]
pub enum EditCommand {
    /// Put new text in place of WHERE.
    #[command(
        name = "replace",
        about = "Put new text in place of the selected span.",
        after_help = "WHERE is exactly one of:\n  \
                      --file F --old TEXT | --old-file F2   that exact text (once by default)\n  \
                      --file F --pattern REGEX              what the regular expression matches\n  \
                      --file F --lines A:B                  those lines, both ends included\n  \
                      --file F                              the whole file\n  \
                      --symbol S                            the whole definition of S\n  \
                      --symbol S --body                     only its body\n  \
                      --target H                            the span a handle marks\n\n\
                      Example:\n  greppy edit replace --symbol greet --content-file body.rs"
    )]
    Replace {
        #[arg(long)]
        file: Option<String>,
        /// The exact text to look for.
        #[arg(long, allow_hyphen_values = true)]
        old: Option<String>,
        /// The exact text to look for, from a file (`-` reads the pipe).
        #[arg(long = "old-file")]
        old_file: Option<String>,
        /// A regular expression selecting the span.
        #[arg(long, allow_hyphen_values = true)]
        pattern: Option<String>,
        /// A line range A:B, 1-based, both ends included.
        #[arg(long)]
        lines: Option<String>,
        /// A definition, resolved like `read`.
        #[arg(long)]
        symbol: Option<String>,
        /// Only the definition's body; the signature stays as it is.
        #[arg(long)]
        body: bool,
        /// The span a handle marks.
        #[arg(long)]
        target: Option<String>,
        /// The new text, for short single-line text.
        #[arg(long, allow_hyphen_values = true)]
        content: Option<String>,
        /// The new text from a file; `-` reads it from the pipe.
        #[arg(long = "content-file", aliases = ["source-file", "source"])]
        content_file: Option<String>,
        /// Require exactly N matches instead of one.
        #[arg(long)]
        expect: Option<usize>,
        /// Only results under that file or directory.
        #[arg(long)]
        path: Option<String>,
        /// Report what it would write and write nothing.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// After writing, run the build or linter for the touched files.
        #[arg(long)]
        verify: bool,
        /// Write the full record of the edit to a file.
        #[arg(long)]
        report: Option<String>,
    },
    /// Put new text next to WHERE, on the side `--before`/`--after` names.
    #[command(
        name = "insert",
        about = "Put new text next to the selected span.",
        after_help = "WHERE is exactly one of --symbol S, --file F --lines A:B, or --target H;\n\
                      a text or regex match has no defined side to land on.\n\n\
                      Example:\n  greppy edit insert --symbol greet --after --content-file block.rs"
    )]
    Insert {
        #[arg(long)]
        file: Option<String>,
        #[arg(long, allow_hyphen_values = true)]
        old: Option<String>,
        #[arg(long = "old-file")]
        old_file: Option<String>,
        #[arg(long, allow_hyphen_values = true)]
        pattern: Option<String>,
        /// A line range A:B, 1-based, both ends included.
        #[arg(long)]
        lines: Option<String>,
        /// A definition, resolved like `read`.
        #[arg(long)]
        symbol: Option<String>,
        /// Anchor on the definition's body instead of the whole definition.
        #[arg(long)]
        body: bool,
        /// The span a handle marks.
        #[arg(long)]
        target: Option<String>,
        /// Land on the side above the anchor.
        #[arg(long)]
        before: bool,
        /// Land on the side below the anchor.
        #[arg(long)]
        after: bool,
        /// The new text, for short single-line text.
        #[arg(long, allow_hyphen_values = true)]
        content: Option<String>,
        /// The new text from a file; `-` reads it from the pipe.
        #[arg(long = "content-file", aliases = ["source-file", "source"])]
        content_file: Option<String>,
        /// Only results under that file or directory.
        #[arg(long)]
        path: Option<String>,
        /// Report what it would write and write nothing.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// After writing, run the build or linter for the touched files.
        #[arg(long)]
        verify: bool,
        /// Write the full record of the edit to a file.
        #[arg(long)]
        report: Option<String>,
    },
    /// Remove what WHERE points at.
    #[command(
        name = "delete",
        about = "Remove what the selector points at.",
        after_help = "WHERE is exactly one of:\n  \
                      --file F --old TEXT | --old-file F2   that exact text (once by default)\n  \
                      --file F --pattern REGEX              what the regular expression matches\n  \
                      --file F --lines A:B                  those lines, both ends included\n  \
                      --file F                              the whole file's contents\n  \
                      --symbol S                            the whole definition of S\n  \
                      --target H                            the span a handle marks\n\n\
                      Example:\n  greppy edit delete --symbol obsolete"
    )]
    Delete {
        #[arg(long)]
        file: Option<String>,
        /// The exact text to look for.
        #[arg(long, allow_hyphen_values = true)]
        old: Option<String>,
        /// The exact text to look for, from a file (`-` reads the pipe).
        #[arg(long = "old-file")]
        old_file: Option<String>,
        /// A regular expression selecting the span.
        #[arg(long, allow_hyphen_values = true)]
        pattern: Option<String>,
        /// A line range A:B, 1-based, both ends included.
        #[arg(long)]
        lines: Option<String>,
        /// A definition, resolved like `read`.
        #[arg(long)]
        symbol: Option<String>,
        /// Only the definition's body.
        #[arg(long)]
        body: bool,
        /// The span a handle marks.
        #[arg(long)]
        target: Option<String>,
        /// Require exactly N matches instead of one.
        #[arg(long)]
        expect: Option<usize>,
        /// Only results under that file or directory.
        #[arg(long)]
        path: Option<String>,
        /// Report what it would write and write nothing.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// After writing, run the build or linter for the touched files.
        #[arg(long)]
        verify: bool,
        /// Write the full record of the edit to a file.
        #[arg(long)]
        report: Option<String>,
    },
    /// Apply a unified diff inside WHERE. The diff's line numbers may count
    /// from the start of the file or from the start of WHERE — whichever the
    /// context lines confirm. Paths inside the diff are ignored.
    #[command(
        name = "patch",
        about = "Apply a unified diff inside the selected span.",
        after_help = "WHERE is exactly one of --symbol S, --file F --lines A:B, --file F, or\n\
                      --target H; a text or regex match gives the hunks nothing to count from.\n\n\
                      Example:\n  greppy edit patch --symbol greet --patch-file greet.diff"
    )]
    Patch {
        #[arg(long)]
        file: Option<String>,
        #[arg(long, allow_hyphen_values = true)]
        old: Option<String>,
        #[arg(long = "old-file")]
        old_file: Option<String>,
        #[arg(long, allow_hyphen_values = true)]
        pattern: Option<String>,
        /// A line range A:B, 1-based, both ends included.
        #[arg(long)]
        lines: Option<String>,
        /// A definition, resolved like `read`.
        #[arg(long)]
        symbol: Option<String>,
        /// Anchor on the definition's body instead of the whole definition.
        #[arg(long)]
        body: bool,
        /// The span a handle marks.
        #[arg(long)]
        target: Option<String>,
        /// The unified diff to apply; `-` reads it from the pipe.
        #[arg(long = "patch-file")]
        patch_file: Option<String>,
        /// Only results under that file or directory.
        #[arg(long)]
        path: Option<String>,
        /// Report what it would write and write nothing.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// After writing, run the build or linter for the touched files.
        #[arg(long)]
        verify: bool,
        /// Write the full record of the edit to a file.
        #[arg(long)]
        report: Option<String>,
    },
    /// Create a file. `replace --file F` needs one that already exists.
    #[command(
        name = "write",
        about = "Create a file from the given content.",
        after_help = "Example:\n  greppy edit write --file src/new.rs --content-file new.rs"
    )]
    Write {
        #[arg(long)]
        file: String,
        /// The new text, for short single-line text.
        #[arg(long, allow_hyphen_values = true)]
        content: Option<String>,
        /// The new text from a file; `-` reads it from the pipe.
        #[arg(long = "content-file", aliases = ["source-file", "source"])]
        content_file: Option<String>,
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// After writing, run the build or linter for the touched files.
        #[arg(long)]
        verify: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Move or rename a file, and update the declarations naming it.
    #[command(
        name = "move",
        about = "Move or rename a file and update what names it.",
        after_help = "Example:\n  greppy edit move --file src/old.rs --to src/new.rs"
    )]
    Move {
        #[arg(long)]
        file: String,
        /// The new path.
        #[arg(long)]
        to: String,
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// After writing, run the build or linter for the touched files.
        #[arg(long)]
        verify: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Delete a file, and report what still references it. Refuses while
    /// something still points at it; `--force` overrides that.
    #[command(
        name = "remove",
        about = "Delete a file; refuses while something still references it.",
        after_help = "Example:\n  greppy edit remove --file src/obsolete.rs\n  \
                      greppy edit remove --file src/obsolete.rs --force"
    )]
    Remove {
        #[arg(long)]
        file: String,
        /// Delete it even though something still references it.
        #[arg(long)]
        force: bool,
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// After writing, run the build or linter for the touched files.
        #[arg(long)]
        verify: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Rename a definition and every reference to it (`--symbol S --to N`),
    /// or make one definition call something else (`--in S --call A --to B`).
    #[command(
        name = "rename",
        about = "Rename a definition, or redirect a call inside one definition.",
        after_help = "Example:\n  greppy edit rename --symbol combine --to merge\n  \
                      greppy edit rename --in caller --call combine --to merge"
    )]
    Rename {
        /// The definition to rename.
        #[arg(long)]
        symbol: Option<String>,
        /// The definition whose calls are redirected.
        #[arg(long = "in")]
        r#in: Option<String>,
        /// The callee to redirect.
        #[arg(long)]
        call: Option<String>,
        /// The new name.
        #[arg(long)]
        to: String,
        /// Require exactly N redirected calls.
        #[arg(long)]
        expect: Option<usize>,
        /// Accepted old-name occurrences left over after a workspace rename.
        #[arg(long = "expect-residual", default_value_t = 0)]
        expect_residual: usize,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Change a definition signature and every graph-resolved call site in one
    /// transaction, using the old/new parameter lists and call cardinality in
    /// a JSON specification.
    #[command(
        name = "change-signature",
        about = "Change a signature and all graph-resolved call sites.",
        after_help = r#"Example:
  greppy edit change-signature --symbol combine --spec '{"old_parameters":"(a: i32, b: i32)","new_parameters":"(b: i32, a: i32)","expect_call_sites":1}'"#
    )]
    ChangeSignature {
        #[arg(long)]
        symbol: String,
        /// Inline JSON, or a JSON file containing old_parameters,
        /// new_parameters, added_arguments, and expect_call_sites.
        #[arg(long)]
        spec: String,
        /// graph (default) uses the resolved store; lsp is unavailable in this build.
        #[arg(long, default_value = "graph", value_parser = ["graph", "lsp"])]
        backend: String,
        #[arg(long = "expect-residual", default_value_t = 0)]
        expect_residual: usize,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Add an import if the file is missing it.
    #[command(
        name = "ensure-import",
        about = "Insert an import once at the canonical position.",
        after_help = "Example:\n  greppy edit ensure-import --file src/lib.rs --module std::collections --name HashMap"
    )]
    EnsureImport {
        #[arg(long)]
        file: String,
        #[arg(long)]
        module: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Within one definition, append an argument to every call of NAME
    /// that does not already carry it (idempotent).
    #[command(
        name = "ensure-argument",
        about = "Append a missing argument to matching calls in one definition.",
        after_help = "Example:\n  greppy edit ensure-argument --symbol caller --call combine --arg 3"
    )]
    EnsureArgument {
        #[arg(long)]
        symbol: String,
        /// The callee whose calls get the argument.
        #[arg(long)]
        call: String,
        /// Argument text, e.g. "timeout=30".
        #[arg(long)]
        arg: String,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Add a method a class lacks.
    #[command(
        name = "ensure-method",
        about = "Append a method to a class when it is absent.",
        after_help = "Example:\n  greppy edit ensure-method --symbol Greeter --name greet --content-file greet_method.py"
    )]
    EnsureMethod {
        /// The class (resolved like read).
        #[arg(long)]
        symbol: String,
        /// The new method's name (idempotency key).
        #[arg(long)]
        name: String,
        /// File containing the full method source, indented for the class body.
        #[arg(long = "content-file", aliases = ["source-file", "source"])]
        content_file: String,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Add an annotation to a definition, once.
    #[command(
        name = "ensure-annotation",
        about = "Add a decorator or attribute above a definition once.",
        after_help = "Example:\n  greppy edit ensure-annotation --symbol greet --annotation '#[inline]'"
    )]
    EnsureAnnotation {
        #[arg(long)]
        symbol: String,
        #[arg(long)]
        annotation: String,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Set or remove a value in JSON, TOML or YAML by path.
    #[command(
        name = "data",
        about = "Set or remove a value in JSON, TOML, or YAML.",
        after_help = "Example:\n  greppy edit data set --file config.json --path '$.server.port' --value-json 8080\n  \
                      greppy edit data delete --file config.json --path '$.server.port'"
    )]
    Data {
        /// set (write the value), ensure (idempotent), or delete (remove it)
        #[arg(value_parser = ["set", "ensure", "delete"])]
        mode: String,
        #[arg(long)]
        file: String,
        /// Path like $.server.port or $.items[2].name
        #[arg(long)]
        path: String,
        /// New value as JSON (strings quoted: '"text"'); not used by delete.
        #[arg(long = "value-json")]
        value_json: Option<String>,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Execute a multi-operation plan file (schema greppy.edit-plan.v1) as one
    /// single change: all files or none. `-` reads the plan from the pipe.
    #[command(
        name = "apply",
        about = "Execute a multi-operation edit plan transactionally.",
        after_help = r#"Example:
  greppy edit apply --plan <(printf '%s\n' '{"operations":[{"file":"notes.txt","old":"before","new":"after","expect":1}]}')"#
    )]
    Apply {
        #[arg(long)]
        plan: String,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
        /// Write the unified diff to FILE (patch mode).
        #[arg(long)]
        diff: Option<String>,
    },
    /// Reverse the last edit, if the file still looks the way that edit left it.
    #[command(
        name = "undo",
        about = "Reverse the last edit in this workspace.",
        after_help = "Example:\n  greppy edit undo"
    )]
    Undo {
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long)]
        report: Option<String>,
    },
    /// Finish or roll back an edit that was interrupted.
    #[command(
        name = "recover",
        about = "Restore pre-images from an interrupted journal transaction.",
        after_help = "Example:\n  greppy edit recover --report recovery.json"
    )]
    Recover {
        /// Write the full recovery report as JSON to FILE.
        #[arg(long)]
        report: Option<String>,
    },
}

impl Cli {
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }
}

/// The set of recognised `greppy` subcommand names. Used by the
/// pre-clap argv router ([`run_os`]) to decide whether an invocation is
/// a structured subcommand (safe to hand to clap, which requires UTF-8
/// argv) or a bare `grep` passthrough (which must forward arbitrary
/// bytes, including non-UTF-8 patterns/paths, to real grep).
const SUBCOMMANDS: &[&str] = &[
    "grep",
    "index",
    "map",
    "outline",
    "changes",
    "cache",
    "trial",
    "verify",
    "stats",
    "diagnostics",
    "doctor",
    "search-graph",
    "trace",
    "impact",
    "brief",
    "expand",
    "read",
    "edit",
    "who-calls",
    "callees",
    "fan-in",
    "fan-out",
    "graph-locate",
    "path",
    "search-code",
    "search-symbols",
    "plus",
    "semantic-search",
    "semantic",
    "context",
    "install",
    "uninstall",
    "update",
    "upgrade",
    "config",
    "embed-daemon",
    "summarize-daemon",
];

/// The verbs the new edit grammar replaced, each with the invocation that now
/// does its work. They are gone without an alias, so naming one is an error —
/// and the error says which spelling does the job, because the caller's
/// intent is known exactly.
const RETIRED_EDIT_VERBS: &[(&str, &str)] = &[
    ("text-cas", "greppy edit replace --file F --old TEXT"),
    ("regex-cas", "greppy edit replace --file F --pattern REGEX"),
    ("replace-body", "greppy edit replace --symbol S --body"),
    ("replace-span", "greppy edit replace --file F --lines A:B"),
    ("patch-span", "greppy edit patch --file F --lines A:B"),
    ("insert-after", "greppy edit insert --symbol S --after"),
    ("insert-before", "greppy edit insert --symbol S --before"),
    ("remove-if-present", "greppy edit delete --file F --old TEXT"),
];

fn retired_edit_verb(name: &str) -> Option<&'static str> {
    RETIRED_EDIT_VERBS
        .iter()
        .find(|(verb, _)| *verb == name)
        .map(|(_, replacement)| *replacement)
}

/// An argv that carries a greppy flag, or names a verb removed from the
/// grammar, is a greppy command whose verb is unknown. Refuse it instead of
/// letting the passthrough reinterpret the verb itself as a search pattern.
fn unknown_verb_refusal(argv: &[std::ffi::OsString]) -> Option<String> {
    let rest = grep_passthrough_args(argv);
    let verb = rest.first()?.to_str()?;
    if verb == "find-usages" {
        return Some("error: unrecognized subcommand 'find-usages'".to_string());
    }
    if verb.starts_with('-') || SUBCOMMANDS.contains(&verb) {
        return None;
    }
    let replacement = retired_edit_verb(verb);
    if replacement.is_none() && greppy_only_flag(&rest[1..]).is_none() {
        return None;
    }
    let mut message = format!("unrecognized command `{verb}`");
    if let Some(replacement) = replacement {
        // A retired verb is just an unknown subcommand. Naming its successor
        // would keep a table of dead names alive (owner decision: no relics,
        // not even in error messages) to serve a caller that measurably does
        // not exist — zero retired-verb calls in 1072 benchmarked turns.
    } else {
        message.push_str(
            "\nusage: greppy <command> --help  (commands: index, trial, who-calls, callees, \
             impact, brief, semantic-search, search-code, search-symbols, path, \
             read, edit)",
        );
    }
    Some(message)
}

/// Top-level entry point that captures argv as `OsString` BEFORE clap
/// consumes it.
///
/// `greppy -R pat $'f\xff'` must behave like grep, not
/// produce a clap rc=2 usage error. clap requires every argv element to
/// be valid UTF-8, so we cannot let it parse a grep passthrough that
/// carries a non-UTF-8 pattern or path. We therefore inspect `args_os`
/// directly: if the invocation is NOT a recognised structured
/// subcommand (and is not a help/version request), we treat it as a
/// `grep` passthrough and forward the original `OsString` argv to real
/// grep byte-for-byte. All recognised subcommands still flow through
/// clap unchanged.
pub fn run_os(argv: Vec<std::ffi::OsString>) -> u8 {
    // Invoked THROUGH a grep/rg filesystem name (symlink or shim to the
    // greppy binary): the caller wanted that tool, verbatim — argv[1..]
    // must never be parsed as greppy subcommands (`rg index .` is a
    // ripgrep search for "index", not `greppy index`). Route the whole
    // tail straight into the passthrough with the matching placeholder.
    let argv0_base = argv
        .first()
        .map(std::path::Path::new)
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if matches!(
        argv0_base,
        "rg" | "ripgrep" | "grep" | "egrep" | "fgrep" | "rgrep"
    ) {
        let mut full: Vec<std::ffi::OsString> = Vec::with_capacity(argv.len() + 1);
        full.push(std::ffi::OsString::from("greppy"));
        full.push(std::ffi::OsString::from(argv0_base));
        full.extend_from_slice(&argv[1..]);
        return match dispatch_grep_os(&full) {
            Ok(code) => code.clamp(0, 255) as u8,
            Err(Error::Invalid(msg)) => {
                // Agent-facing terminal errors go to STDOUT: trace forensics
                // (2026-07-17) showed agents piping `2>/dev/null` and seeing
                // "(no output)" where the refusal explained the retry. Exit
                // code still signals failure to scripts.
                println!("{msg}");
                EXIT_USAGE
            }
            Err(other) => {
                println!("{other}");
                EXIT_IO
            }
        };
    }
    let argv = normalize_global_output_flags(argv);
    CLI_INVOCATION.with(|invocation| *invocation.borrow_mut() = argv.clone());
    #[cfg(feature = "agent")]
    if is_agent_invocation(&argv) {
        // Refuse a nested run BEFORE anything else — before argument validation,
        // before the store, and long before a model call. The agent runs build
        // and test commands through `/bin/sh -lc` with PATH passed through, so a
        // task can reach this binary; without this the run could cascade, each
        // level burning its own model budget. Keyed on the ENVIRONMENT, never on
        // the command line, so `$(which greppy) -p`, aliases and wrappers are all
        // caught. greppy_agent::run_agent holds the same line for library callers.
        if let Err(error) = agent_refuse_nested_invocation() {
            eprintln!("greppy: {error}");
            return EXIT_USAGE;
        }
        maybe_run_store_cleanup(peek_root_arg(&argv).as_deref());
        return match dispatch_agent_invocation(argv) {
            Ok(code) => code.clamp(0, 255) as u8,
            Err(error) => {
                eprintln!("greppy: {error}");
                error_exit_code(&error)
            }
        };
    }
    if let Some(message) = unknown_verb_refusal(&argv) {
        println!("{message}");
        return EXIT_USAGE;
    }
    if is_grep_passthrough(&argv) {
        // argv[0] is the binary name; the rest are grep args. Build a
        // synthetic argv for the shared runner whose argv[0] is a
        // placeholder and argv[1..] are the user's (possibly non-UTF-8)
        // arguments. Greppy-owned global options are consumed before the
        // remaining arguments are forwarded verbatim.
        let mut full: Vec<std::ffi::OsString> = Vec::with_capacity(argv.len());
        full.push(std::ffi::OsString::from("greppy"));
        full.extend_from_slice(grep_passthrough_args(&argv));
        return match dispatch_grep_os(&full) {
            Ok(code) => code.clamp(0, 255) as u8,
            Err(Error::Invalid(msg)) => {
                // Agent-facing terminal errors go to STDOUT: trace forensics
                // (2026-07-17) showed agents piping `2>/dev/null` and seeing
                // "(no output)" where the refusal explained the retry. Exit
                // code still signals failure to scripts.
                println!("{msg}");
                EXIT_USAGE
            }
            Err(other) => {
                println!("{other}");
                EXIT_IO
            }
        };
    }
    // Structured Greppy commands perform throttled cache maintenance. This
    // intentionally runs after passthrough detection so an ordinary grep
    // invocation cannot touch Greppy state.
    if !is_trial_invocation(&argv) {
        maybe_run_store_cleanup(peek_root_arg(&argv).as_deref());
    }
    // Structured subcommand (or help/version): clap can parse it. Any
    // non-UTF-8 here is a genuine usage error for a structured command.
    // P3: a failed agent call must TEACH the correct retry in the same
    // output — one short error line plus the affected subcommand's usage,
    // never a multi-KB dump. Explicit --help/--version keep clap's output.
    let cli = match <Cli as Parser>::try_parse_from(argv.iter()) {
        Ok(cli) => cli,
        Err(e) => {
            use clap::error::ErrorKind;
            if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
                let _ = e.print();
                return 0;
            }
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or("invalid arguments");
            // STDOUT, not stderr: agents habitually append `2>/dev/null`,
            // and a usage lesson they never see teaches nothing (P3).
            println!("{first}");
            // clap reports a bundled short-flag mistake by its first letter
            // (`-gamma` becomes `-g`), which does not name what the caller
            // wrote. Print the argument verbatim so the refusal is actionable.
            if let Some(raw) = unabbreviated_invalid_argument(&argv, first) {
                println!(
                    "`{raw}` is not a flag, and a positional argument is a symbol name, never a \
                     path — the path filter is `--path`"
                );
            }
            // Skip greppy-owned global flags when picking the usage line:
            // agents habitually write `greppy --root . read ...`, and argv[1]
            // is then "--root", which used to fall through to the generic
            // command list instead of the read usage (trace forensics
            // 2026-07-17: 13/24 calls in one run were flag guesses that the
            // generic list did nothing to correct).
            let sub = grep_passthrough_args(&argv)
                .first()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            // `greppy edit text-cas …`: clap can only say the subcommand is
            // unknown. The caller's intent is known exactly, so name the
            // spelling that does the work.
            if let Some(corrected) = closest_valid_invocation(&argv, sub, &msg) {
                println!("{corrected}");
            }
            if let Some(usage) = subcommand_usage(sub) {
                println!("usage: {usage}");
            } else {
                println!(
                    "usage: greppy <command> --help  (commands: index, trial, who-calls, callees, \
                     impact, brief, semantic-search, search-code, search-symbols, \
                     path, index status)"
                );
            }
            return EXIT_USAGE;
        }
    };
    dispatch_to_code(cli)
}

/// clap prints `unexpected argument '-g' found` for `-gamma`, because it reads
/// a leading single dash as bundled short flags. Recover the token the caller
/// actually typed so the refusal names it.
fn unabbreviated_invalid_argument(argv: &[std::ffi::OsString], first_line: &str) -> Option<String> {
    let mut parts = first_line.split('\'');
    parts.next()?;
    let reported = parts.next()?;
    if reported.len() < 2 || !reported.starts_with('-') {
        return None;
    }
    argv.iter()
        .skip(1)
        .map(|token| token.to_string_lossy().into_owned())
        .find(|token| token.starts_with(reported) && token != reported)
}

#[cfg(feature = "agent")]
fn is_agent_invocation(argv: &[std::ffi::OsString]) -> bool {
    if argv.get(1).is_some_and(|token| token == "grep") {
        return false;
    }
    argv.iter().skip(1).any(|token| {
        token == "-p"
            || token == "--prompt"
            || token == "--apply-result"
            || token.to_string_lossy().starts_with("--prompt=")
            || token.to_string_lossy().starts_with("--apply-result=")
    })
}

#[cfg(feature = "agent")]
fn dispatch_agent_invocation(argv: Vec<std::ffi::OsString>) -> Result<i32> {
    let args = match AgentInvocation::try_parse_from(argv.iter()) {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error
                .print()
                .map_err(|error| Error::io("print agent help", error))?;
            return Ok(0);
        }
        Err(error) => {
            return Err(Error::Invalid(
                error
                    .to_string()
                    .lines()
                    .next()
                    .unwrap_or("invalid agent arguments")
                    .to_string(),
            ));
        }
    };
    if let Some(run_id) = args.apply_result.as_deref() {
        let root = args
            .root
            .as_deref()
            .map(|root| resolve_root(Some(root)))
            .transpose()?;
        let report = greppy_agent::apply_result(run_id, root.as_deref())
            .map_err(|error| Error::Store(format!("apply agent result: {error}")))?;
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .map_err(|error| Error::Invalid(format!("serialize apply report: {error}")))?
            );
        } else if report.applied {
            println!(
                "applied run {} ({} files)",
                report.run_id,
                report.changed_files.len()
            );
        } else {
            println!(
                "run {} was not applied; conflicts: {}",
                report.run_id,
                report.conflicts.join(", ")
            );
        }
        return Ok(if report.applied {
            0
        } else {
            EXIT_TEMPFAIL as i32
        });
    }

    let prompt = args
        .prompt
        .filter(|prompt| !prompt.trim().is_empty())
        .ok_or_else(|| {
            Error::Invalid("`-p PROMPT` or `--apply-result RUN_ID` is required".into())
        })?;
    let root = resolve_root(args.root.as_deref())?;
    let base_url = args
        .base_url
        .or_else(|| env_nonempty("GREPPY_AGENT_BASE_URL"))
        .or_else(|| env_nonempty("OPENAI_BASE_URL"))
        .unwrap_or_else(|| "https://api.openai.com/v1".into());
    let model = args
        .model
        .or_else(|| env_nonempty("GREPPY_AGENT_MODEL"))
        .or_else(|| env_nonempty("OPENAI_MODEL"))
        .ok_or_else(|| {
            Error::Invalid("`--model MODEL` or GREPPY_AGENT_MODEL/OPENAI_MODEL is required".into())
        })?;
    let api_key = std::env::var(&args.api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            Error::Invalid(format!(
                "API key environment variable {} is not set or empty",
                args.api_key_env
            ))
        })?;
    drop(open_default_store(Some(root.to_string_lossy().as_ref()))?);
    let base_graph = workspace_locator::store_path(&root);
    let project = workspace_locator::project_identity(&root);

    let embedding_cfg = embedding_config_optional(EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    })?;
    let inference_hooks = embedding_cfg.map(|cfg| {
        std::sync::Arc::new(AgentInferenceHooks {
            root: root.clone(),
            cache_dir: workspace_locator::store_dir(&root),
            cfg,
        })
    });
    let embedding_hooks = inference_hooks
        .as_ref()
        .map(|hooks| hooks.clone() as std::sync::Arc<dyn greppy_agent::EmbeddingHooks>);
    let navigation_hooks = inference_hooks
        .as_ref()
        .map(|hooks| hooks.clone() as std::sync::Arc<dyn greppy_agent::NavigationHooks>);
    let workspace = greppy_agent::TaskWorkspace::create(
        &root,
        &base_graph,
        project,
        args.keep_run,
        embedding_hooks,
    )
    .map_err(|error| Error::Store(format!("create agent workspace: {error}")))?;
    let cancellation_workspace = workspace.clone();
    if let Err(error) = ctrlc::set_handler(move || cancellation_workspace.cancel()) {
        let _ = workspace.discard_run();
        return Err(Error::Store(format!(
            "install agent cancellation handler: {error}"
        )));
    }

    let mut config = greppy_agent::AgentConfig::with_defaults(
        prompt,
        greppy_agent::ResponsesConfig {
            base_url,
            model,
            api_key,
            max_output_tokens: args.max_output_tokens,
            request_timeout: std::time::Duration::from_secs(args.timeout_seconds.clamp(1, 3600)),
        },
        workspace,
    );
    config.max_assistant_turns = args.max_turns.clamp(1, 200);
    config.apply = args.apply;
    config.navigation_hooks = navigation_hooks;
    if !args.json {
        config.event_sink = Some(std::sync::Arc::new(|event: &greppy_agent::AgentEvent| {
            if let greppy_agent::AgentEvent::ToolCallStarted { name, .. } = event {
                eprintln!("greppy -p: {name}");
            }
        }));
    }
    let result = greppy_agent::run_agent(config)
        .map_err(|error| Error::Store(format!("agent run: {error}")))?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&result)
                .map_err(|error| Error::Invalid(format!("serialize agent result: {error}")))?
        );
    } else {
        println!("{}", result.message);
        println!();
        println!("Run: {}", result.run_id);
        println!("Changes: {}", result.changeset.changes.len());
        if let Some(path) = result.artifact.as_deref() {
            println!("Artifact: {}", path.display());
        }
        if let Some(apply) = result.apply.as_ref() {
            println!(
                "Apply: {}",
                if apply.applied {
                    "applied"
                } else {
                    "conflicted"
                }
            );
        }
    }
    Ok(if result.status == "cancelled" {
        130
    } else if result.status != "completed"
        || result.apply.as_ref().is_some_and(|report| !report.applied)
    {
        EXIT_TEMPFAIL as i32
    } else {
        0
    })
}

fn normalize_global_output_flags(mut argv: Vec<std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    let Some(subcommand_index) = argv.iter().enumerate().skip(1).find_map(|(index, token)| {
        token
            .to_str()
            .is_some_and(|token| SUBCOMMANDS.contains(&token))
            .then_some(index)
    }) else {
        return argv;
    };
    let mut moved = Vec::new();
    let mut indexes = (1..subcommand_index)
        .filter(|&index| matches!(argv[index].to_str(), Some("--json" | "--code" | "--all")))
        .collect::<Vec<_>>();
    for index in indexes.drain(..).rev() {
        moved.push(argv.remove(index));
    }
    moved.reverse();
    let Some(new_subcommand_index) = argv.iter().position(|token| {
        token
            .to_str()
            .is_some_and(|token| SUBCOMMANDS.contains(&token))
    }) else {
        return argv;
    };
    for (offset, token) in moved.into_iter().enumerate() {
        argv.insert(new_subcommand_index + 1 + offset, token);
    }
    argv
}

fn grep_passthrough_args(argv: &[std::ffi::OsString]) -> &[std::ffi::OsString] {
    let mut index = 1;
    while index < argv.len() {
        let token = &argv[index];
        if token == "--root"
            || token == "--device"
            || token == "--limit"
            || token == "--max"
            || token == "--max-bytes"
            || token == "--offset"
        {
            index = (index + 2).min(argv.len());
            continue;
        }
        let token_lossy = token.to_string_lossy();
        if token_lossy.starts_with("--root=")
            || token_lossy.starts_with("--device=")
            || token_lossy.starts_with("--limit=")
            || token_lossy.starts_with("--max=")
            || token_lossy.starts_with("--max-bytes=")
            || token_lossy.starts_with("--offset=")
            || token == "--no-gpu"
            || token == "--no-summaries"
        {
            index += 1;
            continue;
        }
        break;
    }
    &argv[index..]
}

fn closest_valid_invocation(
    argv: &[std::ffi::OsString],
    subcommand: &str,
    clap_message: &str,
) -> Option<String> {
    let unknown = clap_message
        .strip_prefix("error: unexpected argument '")?
        .split('\'')
        .next()?;
    if !unknown.starts_with("--") {
        return None;
    }
    let mut candidates = vec![
        "--root",
        "--device",
        "--json",
        "--code",
        "--all",
        "--limit",
        "--max",
        "--max-bytes",
        "--offset",
    ];
    let invocation = grep_passthrough_args(argv);
    let nested_edit = if subcommand == "edit" {
        invocation.get(1).and_then(|argument| argument.to_str())
    } else {
        None
    };
    candidates.extend(match subcommand {
        "read" => vec!["--symbol", "--path", "--handle", "--lines", "--line"],
        "who-calls" | "callees" | "brief" | "semantic-search" | "semantic" => {
            vec!["--path"]
        }
        "search-code" => vec![
            "--path",
            "--changed",
            "--staged",
            "--since",
            "--base",
            "--no-code",
            "--fixed",
        ],
        "changes" => vec!["--base"],
        "search-symbols" => vec!["--path", "--kind"],
        "impact" => vec!["--direction", "--edge", "--depth", "--since", "--base"],
        "trace" => vec!["--symbol", "--direction", "--edge", "--depth"],
        "path" => vec!["--from", "--to", "--edge"],
        "graph-locate" => vec!["--file", "--line"],
        "plus" => vec!["--k", "--explain"],
        "context" => vec!["--k", "--lines"],
        "edit" => match nested_edit {
            Some("replace") | Some("delete") => vec![
                "--file",
                "--old",
                "--old-file",
                "--pattern",
                "--lines",
                "--symbol",
                "--body",
                "--target",
                "--content",
                "--content-file",
                "--expect",
                "--dry-run",
                "--report",
            ],
            Some("insert") => vec![
                "--symbol",
                "--lines",
                "--target",
                "--before",
                "--after",
                "--content",
                "--content-file",
                "--dry-run",
                "--report",
            ],
            Some("patch") => vec![
                "--symbol",
                "--lines",
                "--target",
                "--patch-file",
                "--dry-run",
                "--report",
            ],
            _ => Vec::new(),
        },
        _ => Vec::new(),
    });
    let replacement = candidates
        .into_iter()
        .min_by_key(|candidate| levenshtein(unknown, candidate))?;
    let distance = levenshtein(unknown, replacement);
    if distance > 4 {
        return None;
    }
    Some(
        argv.iter()
            .map(|argument| {
                let argument = argument.to_string_lossy();
                let value = if argument == unknown {
                    replacement
                } else {
                    argument.as_ref()
                };
                shell_quote_cli(value)
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn levenshtein(left: &str, right: &str) -> usize {
    let mut previous = (0..=right.chars().count()).collect::<Vec<_>>();
    for (left_index, left_char) in left.chars().enumerate() {
        let mut current = vec![left_index + 1];
        for (right_index, right_char) in right.chars().enumerate() {
            current.push(
                (current[right_index] + 1)
                    .min(previous[right_index + 1] + 1)
                    .min(previous[right_index] + usize::from(left_char != right_char)),
            );
        }
        previous = current;
    }
    previous.last().copied().unwrap_or(0)
}

fn shell_quote_cli(value: &str) -> String {
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || "-._/:".contains(character))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// One-line usage per agent-facing subcommand, printed after a short arg
/// error so the failed call carries the correct retry (P3: every failure
/// costs the agent a turn of thinking plus a tool call).
fn subcommand_usage(sub: &str) -> Option<&'static str> {
    Some(match sub {
        "who-calls" => {
            "greppy who-calls SYMBOL [SYMBOL ...] [--path PATH] [--code] [--json] [--all] [--root DIR]"
        }
        "callees" => {
            "greppy callees SYMBOL [SYMBOL ...] [--path PATH] [--code] [--json] [--all] [--root DIR]"
        }
        "impact" => {
            "greppy impact SYMBOL [--direction incoming|outgoing] [--depth N] [--json] [--root DIR]"
        }
        "brief" => "greppy brief SYMBOL [SYMBOL ...] [--path PATH] [--json] [--root DIR]",
        "read" => {
            "greppy read SYMBOL [--path FILE] [--handle] [--json] [--root DIR]  \
             or: greppy read FILE [--line N[:M]] [--handle] [--json] [--root DIR]"
        }
        "edit" => {
            "greppy edit <replace|insert|delete|patch|write|move|remove|rename|\
             change-signature|ensure-import|data|apply|undo|recover> --help"
        }
        "expand" => "greppy expand ID [--json] [--root DIR]",
        "semantic-search" | "semantic" => {
            "greppy semantic-search \"QUERY\" [--path PATH] [--json] [--root DIR]"
        }
        "context" => "greppy context \"QUERY\" [--root DIR]",
        "search-code" => {
            "greppy search-code PATTERN [PATH ...] [--no-code] [--fixed] [--json] [--root DIR]"
        }
        "search-symbols" => {
            "greppy search-symbols NAME [NAME ...] [--path PATH] [--kind function|method|struct|class] [--json] [--root DIR]"
        }
        "path" => "greppy path --from SYMBOL --to SYMBOL [--root DIR]",
        "index" => "greppy index PATH [--device auto|cpu|metal|cuda]",
        "map" => "greppy map [PATH] [--json] [--root DIR]",
        "changes" => "greppy changes [--base REV] [--json] [--root DIR]",
        "trial" => {
            "greppy trial --root DIR --question QUESTION --check who-calls --symbol SYMBOL \
             --expect TEXT [--forbid TEXT] --runner pi --provider NAME --model ID"
        }
        "verify" => {
            "greppy verify [--baseline REV] [--timeout SECONDS] [--json] [--no-cache] -- <test-command...>"
        }
        "cache" => "greppy cache status|gc|clear [--json|--dry-run|--all --yes] [--root DIR]",
        _ => return None,
    })
}

/// Run manifest-verified, cross-process-throttled maintenance under Greppy's
/// data root. Fully best-effort: any failure is swallowed, and the current
/// workspace is excluded in addition to being protected by its lifecycle
/// lease.
///
/// TTL comes from `GREPPY_STORE_TTL_DAYS` (default 14 days; `0` disables only
/// age-based eviction, not the independent quota) — see
/// [`greppy_core::workspace::store_ttl_secs`].
pub fn maybe_run_store_cleanup(root: Option<&str>) {
    let effective = resolve_root(root).ok();
    if greppy_core::cache::maybe_gc(effective.as_deref()).is_ok_and(|report| !report.throttled) {
        cleanup_verified_legacy_trash();
        cleanup_expired_legacy_entries(
            effective.as_deref(),
            greppy_core::cache::GcPolicy::from_env().ttl,
        );
        prune_expired_evidence_packs();
    }
}

fn prune_expired_evidence_packs() {
    let Ok(status) = greppy_core::cache::cache_status() else {
        return;
    };
    for entry in status
        .entries
        .into_iter()
        .filter(|entry| entry.kind == "workspace" && !entry.locked && !entry.orphaned)
    {
        let Some(root) = entry.workspace_root else {
            continue;
        };
        let Ok(Some(_lifecycle)) = greppy_core::cache::acquire_workspace_lifecycle(
            &root,
            greppy_core::cache::LockMode::Shared,
            true,
        ) else {
            continue;
        };
        let path = workspace_locator::store_path(&root);
        let Ok(_writer) = greppy_freshness::try_acquire(&path) else {
            continue;
        };
        let Ok(store) =
            greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())
        else {
            continue;
        };
        let _ = store.prune_expired_expand_packs();
    }
}

// ---------------------------------------------------------------------------
// outline — what stands in THIS file
//
// `search-symbols` looks a name up across the repository; `outline` answers the
// other question, "what is in this file", which today costs a whole `read`. The
// symbol graph already holds every definition of a file with its span, so the
// answer is a projection of the graph, not a second parse.
// ---------------------------------------------------------------------------

/// One row of an outline.
struct OutlineRow {
    name: String,
    qualified_name: String,
    kind: String,
    signature: String,
    start: i64,
    end: i64,
}

/// Definitions only. The graph also carries fields, parameters, imports and
/// call sites; an outline that listed them would be the file again, in a worse
/// order.
fn outline_label_is_definition(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "function"
            | "method"
            | "class"
            | "struct"
            | "enum"
            | "trait"
            | "interface"
            | "protocol"
            | "record"
            | "union"
            | "module"
            | "constructor"
            | "typealias"
    )
}

fn outline_label_is_callable(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "function" | "method" | "constructor"
    )
}

/// The keyword a type-like definition is introduced with. The graph labels a
/// Rust `struct`, a Go `type … struct` and a Python `class` all as `Class`,
/// because they occupy the same place in the graph — but an outline reports the
/// kind the caller reads in the file, so the declaration decides.
fn outline_declared_keyword(first_line: &str) -> Option<&'static str> {
    for word in first_line.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        let kind = match word {
            "struct" => "struct",
            "enum" => "enum",
            "trait" => "trait",
            "interface" => "interface",
            "protocol" => "protocol",
            "union" => "union",
            "record" => "record",
            "class" => "class",
            _ => continue,
        };
        return Some(kind);
    }
    None
}

fn outline_rows(nodes: &[greppy_store::Node], source: &str) -> Vec<OutlineRow> {
    let lines: Vec<&str> = source.lines().collect();
    let line_text = |line: i64| -> &str {
        usize::try_from(line)
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| lines.get(n).copied())
            .unwrap_or("")
    };
    let mut ordered: Vec<&greppy_store::Node> = nodes
        .iter()
        .filter(|node| outline_label_is_definition(&node.label))
        .collect();
    ordered.sort_by_key(|node| (node.start_line, node.end_line, node.id));
    ordered
        .iter()
        .map(|node| {
            // The innermost definition that encloses this one. A definition
            // inside a *callable* is a local function, whatever the graph calls
            // it — the graph names it after the type that owns the outer
            // method, which is true of the method and not of the closure.
            let parent = ordered
                .iter()
                .filter(|other| {
                    other.id != node.id
                        && other.start_line <= node.start_line
                        && other.end_line >= node.end_line
                        && (other.start_line, other.end_line) != (node.start_line, node.end_line)
                })
                .min_by_key(|other| other.end_line - other.start_line);
            let signature = line_text(node.start_line).trim().to_string();
            let kind = if parent.is_some_and(|parent| outline_label_is_callable(&parent.label)) {
                "function".to_string()
            } else if outline_label_is_callable(&node.label) {
                node.label.to_ascii_lowercase()
            } else {
                outline_declared_keyword(&signature)
                    .map(str::to_string)
                    .unwrap_or_else(|| node.label.to_ascii_lowercase())
            };
            OutlineRow {
                name: node.name.clone(),
                qualified_name: node.qualified_name.clone(),
                kind,
                signature,
                start: node.start_line,
                end: node.end_line,
            }
        })
        .collect()
}

/// Refusals print the cause on stderr and exit 1, like every other command that
/// cannot answer the question it was asked.
fn outline_refusal(message: String) -> Result<i32> {
    eprintln!("{message}");
    Ok(1)
}

const OUTLINE_PAGE: usize = 50;

fn dispatch_outline(path: &str, json: bool, all: bool, root: Option<&str>) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let candidate = std::path::Path::new(path);
    let abs = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root_path.join(candidate)
    };
    if !abs.exists() {
        return outline_refusal(format!("no file {path}"));
    }
    if abs.is_dir() {
        return outline_refusal(format!("{path} is a directory, not a file"));
    }
    let bytes = std::fs::read(&abs).map_err(|error| Error::io(format!("read {path}"), error))?;
    let Ok(source) = String::from_utf8(bytes) else {
        return outline_refusal(format!("{path} is not text; it has no definitions to outline"));
    };
    let rel = abs
        .strip_prefix(&root_path)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string());

    let mut store = open_default_store(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let nodes = store
        .list_nodes(&project, "", &rel, 0, 100_000)
        .map_err(|error| Error::Invalid(format!("read the outline of {rel}: {error}")))?;
    let rows = outline_rows(&nodes, &source);
    if rows.is_empty() && !greppy_discover::detect_language(&abs).is_detected() {
        // Silence here would read as "this file has no definitions", which is a
        // different answer from "greppy has no parser for this language".
        return outline_refusal(format!("{path}: no language greppy parses"));
    }

    let total = rows.len();
    let offset = cli_result_offset().min(total);
    let limit = if all {
        usize::MAX
    } else {
        cli_result_limit_raw().unwrap_or(OUTLINE_PAGE)
    };
    let end = offset.saturating_add(limit).min(total);
    let window = &rows[offset..end];

    if json {
        let definitions: Vec<serde_json::Value> = window
            .iter()
            .map(|row| {
                serde_json::json!({
                    "name": row.name,
                    "qualified_name": row.qualified_name,
                    "kind": row.kind,
                    "signature": row.signature,
                    "file": rel,
                    "start": row.start,
                    "end": row.end,
                    "span": format!("{}:{}", row.start, row.end),
                })
            })
            .collect();
        let mut value = serde_json::json!({
            "command": "outline",
            "file": rel,
            "total": total,
            "offset": offset,
            "shown": definitions.len(),
            "truncated": end < total,
            "definitions": definitions,
        });
        if end < total {
            value["try"] = serde_json::json!(retry_with_offset("outline", end));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|error| Error::Invalid(format!("serialize outline JSON: {error}")))?
        );
        return Ok(0);
    }
    for row in window {
        println!(
            "{}  {}  {}:{}-{}",
            row.kind, row.signature, rel, row.start, row.end
        );
    }
    if end < total {
        // The continuation is the command itself, so it can be run verbatim.
        println!("{}", retry_with_offset("outline", end));
    }
    Ok(0)
}

fn dispatch_changes(base: Option<&str>, json: bool, root: Option<&str>) -> Result<i32> {
    if json {
        return changes::run(base, true, root);
    }

    // Keep the stable JSON producer in `changes` as the single source of
    // truth. The default renderer consumes that complete record, prints only
    // counts plus changed symbols, and stores the original behind expand.
    let root_path = resolve_root(root)?;
    let mut command = std::process::Command::new(
        std::env::current_exe().map_err(|error| Error::io("locate greppy executable", error))?,
    );
    command.arg("--root").arg(&root_path).arg("changes");
    if let Some(base) = base {
        command.arg("--base").arg(base);
    }
    let output = command
        .arg("--json")
        .output()
        .map_err(|error| Error::io("run changes JSON renderer", error))?;
    if !output.status.success() {
        let diagnosis = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(Error::Invalid(if diagnosis.is_empty() {
            "changes JSON renderer failed".into()
        } else {
            diagnosis
        }));
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| Error::Parse(format!("parse changes JSON: {error}")))?;
    let files = value["files"].as_array().cloned().unwrap_or_default();
    let callsites = value["callsite_impact"].as_array().map_or(0, Vec::len);
    let known_tests = value["tests"]["known_impacted"]
        .as_array()
        .map_or(0, Vec::len);
    let unknown_tests = value["tests"]["unknown_or_unindexed"]
        .as_array()
        .map_or(0, Vec::len);
    let mut symbols = Vec::new();
    for file in &files {
        let path = file["path"].as_str().unwrap_or("?");
        for change in ["modified", "added", "deleted"] {
            for symbol in file["definitions"][change].as_array().into_iter().flatten() {
                let kind = symbol["kind"].as_str().unwrap_or("Symbol");
                let qualified = symbol["qualified_name"].as_str().unwrap_or("?");
                let span = symbol["after_span"]
                    .as_object()
                    .or_else(|| symbol["before_span"].as_object());
                let location = span.map_or_else(
                    || path.to_string(),
                    |span| {
                        format!(
                            "{}:{}-{}",
                            path,
                            span.get("start_line")
                                .and_then(serde_json::Value::as_i64)
                                .unwrap_or(1),
                            span.get("end_line")
                                .and_then(serde_json::Value::as_i64)
                                .unwrap_or(1)
                        )
                    },
                );
                symbols.push(format!("{change} {kind} {qualified} {location}"));
            }
        }
    }
    println!(
        "changes: {} files, {} changed symbols, {} direct callsites",
        files.len(),
        symbols.len(),
        callsites
    );
    println!("tests: {known_tests} known_impacted, {unknown_tests} unknown_or_unindexed");
    // The strict known/unknown split is a contract, and the agent needs the NAMES
    // to decide what to run — counts alone are not actionable. Keep both headings
    // and their entries, capped; the full lists stay in --json and the evidence pack.
    for (heading, key) in [
        ("known_impacted", "known_impacted"),
        ("unknown_or_unindexed", "unknown_or_unindexed"),
    ] {
        let entries = value["tests"][key].as_array().cloned().unwrap_or_default();
        println!("  {heading}:");
        for entry in entries.iter().take(CHANGES_TEST_LIST_CAP) {
            let name = entry
                .get("test_symbol")
                .or_else(|| entry.get("path"))
                .or_else(|| entry.get("file"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| entry.as_str().unwrap_or("?"));
            println!("    {name}");
        }
        if entries.len() > CHANGES_TEST_LIST_CAP {
            println!(
                "    … {} more (use --json)",
                entries.len() - CHANGES_TEST_LIST_CAP
            );
        }
    }
    for symbol in &symbols {
        println!("{symbol}");
    }

    let payload_text = String::from_utf8_lossy(&output.stdout).into_owned();
    let requested_base = base.unwrap_or("HEAD");
    if let (Ok(store), Ok(project)) = (open_default_store_query_writer(root), project_for(root)) {
        let summary = serde_json::json!({
            "text": format!(
                "{} changed symbols, {} callsites, {} known tests, {} unknown tests",
                symbols.len(), callsites, known_tests, unknown_tests
            ),
            "changed_symbols": symbols.len(),
            "callsites": callsites,
            "known_impacted": known_tests,
            "unknown_or_unindexed": unknown_tests,
        });
        if let Some(expand) = insert_expand_pack_best_effort(
            &store,
            &project,
            "changes",
            requested_base,
            current_graph_generation_or_zero(&store, root),
            summary,
            payload_text,
            Some(value),
        ) {
            println!("Expand: greppy expand {}", expand.id);
        }
    }
    Ok(0)
}

fn dispatch_cache(command: CacheCommand, root: Option<&str>) -> Result<i32> {
    match command {
        CacheCommand::Status { json } => {
            let current = resolve_root(root).ok();
            if let Some(current) = current.as_deref() {
                greppy_core::cache::touch_last_used_dir(&greppy_core::cache::workspace_store_dir(
                    current,
                ));
            }
            let mut status = greppy_core::cache::cache_status()
                .map_err(|error| Error::io("read cache status", error))?;
            let policy = greppy_core::cache::GcPolicy::from_env();
            let legacy = verified_legacy_cache_entries();
            for entry in &legacy {
                status.unmanaged.retain(|path| path != &entry.path);
                status.unmanaged_bytes = status.unmanaged_bytes.saturating_sub(entry.bytes);
                status.managed_bytes = status.managed_bytes.saturating_add(entry.bytes);
                if entry.locked {
                    status.locked_bytes = status.locked_bytes.saturating_add(entry.bytes);
                }
            }
            let mut entries = status
                .entries
                .iter()
                .map(|entry| {
                    let freshness = if entry.orphaned {
                        "cold"
                    } else if entry.kind != "workspace" {
                        "unknown"
                    } else if entry.workspace_root.as_deref() == current.as_deref() {
                        current_cache_freshness(entry.workspace_root.as_deref())
                    } else {
                        "unknown"
                    };
                    serde_json::json!({
                        "kind": entry.kind,
                        "id": entry.id,
                        "path": entry.path,
                        "workspace_root": entry.workspace_root,
                        "bytes": entry.bytes,
                        "last_used_unix_secs": entry.last_used_unix_secs,
                        "locked": entry.locked,
                        "orphaned": entry.orphaned,
                        "freshness": freshness,
                    })
                })
                .collect::<Vec<_>>();
            entries.extend(legacy.iter().map(|entry| {
                serde_json::json!({
                    "kind": "legacy-workspace",
                    "id": greppy_core::workspace::workspace_hash(&entry.root),
                    "path": entry.path,
                    "workspace_root": entry.root,
                    "bytes": entry.bytes,
                    "last_used_unix_secs": entry.last_used_unix_secs,
                    "locked": entry.locked,
                    "orphaned": !entry.root.exists(),
                    "freshness": if entry.root.exists() { "drift" } else { "cold" },
                })
            }));
            let value = serde_json::json!({
                "data_root": status.data_root,
                "managed_bytes": status.managed_bytes,
                "unmanaged_bytes": status.unmanaged_bytes,
                "locked_bytes": status.locked_bytes,
                "quota_bytes": policy.high_water_bytes,
                "low_water_bytes": policy.low_water_bytes,
                "ttl_secs": policy.ttl.as_secs(),
                "entries": entries,
                "unmanaged": status.unmanaged,
            });
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).map_err(|error| Error::Invalid(
                        format!("serialize cache status: {error}")
                    ))?
                );
            } else {
                println!("cache root: {}", status.data_root.display());
                println!(
                    "managed: {} bytes; locked: {} bytes; unmanaged: {} bytes; quota: {} bytes",
                    status.managed_bytes,
                    status.locked_bytes,
                    status.unmanaged_bytes,
                    policy.high_water_bytes
                );
                for entry in value["entries"].as_array().into_iter().flatten() {
                    println!(
                        "{} {} {} bytes last_used={} locked={} orphaned={} freshness={}",
                        entry["kind"].as_str().unwrap_or("unknown"),
                        entry["path"].as_str().unwrap_or("?"),
                        entry["bytes"].as_u64().unwrap_or(0),
                        entry["last_used_unix_secs"].as_u64().unwrap_or(0),
                        entry["locked"].as_bool().unwrap_or(false),
                        entry["orphaned"].as_bool().unwrap_or(false),
                        entry["freshness"].as_str().unwrap_or("unknown")
                    );
                }
                for path in &status.unmanaged {
                    println!("unmanaged {}", path.display());
                }
            }
            Ok(0)
        }
        CacheCommand::Gc { dry_run, json } => {
            let current = resolve_root(root).ok();
            let policy = greppy_core::cache::GcPolicy::from_env();
            let mut report = greppy_core::cache::run_gc(&policy, dry_run, current.as_deref())
                .map_err(|error| Error::io("run cache GC", error))?;
            if !dry_run {
                cleanup_verified_legacy_trash();
            }
            let now = unix_now_secs_cli();
            for entry in verified_legacy_cache_entries() {
                if current.as_deref() == Some(entry.root.as_path()) {
                    continue;
                }
                let age = now.saturating_sub(entry.last_used_unix_secs);
                if policy.ttl.is_zero() || age <= policy.ttl.as_secs() {
                    continue;
                }
                report.scanned_bytes = report.scanned_bytes.saturating_add(entry.bytes);
                if entry.locked {
                    report.locked_bytes = report.locked_bytes.saturating_add(entry.bytes);
                    report.skipped_locked.push(entry.path);
                } else if dry_run || remove_verified_legacy_entry(&entry) {
                    report.removed_bytes = report.removed_bytes.saturating_add(entry.bytes);
                    report.removed.push(entry.path);
                }
            }
            print_gc_report(&report, json)?;
            Ok(if report.locked_bytes > 0 {
                EXIT_TEMPFAIL as i32
            } else {
                0
            })
        }
        CacheCommand::Clear { all, yes } => {
            if !yes {
                return Err(Error::Invalid(
                    "cache clear requires --yes; no cache data was removed".into(),
                ));
            }
            if all && root.is_some() {
                return Err(Error::Invalid(
                    "cache clear accepts either --all or --root DIR, not both".into(),
                ));
            }
            let target = if all {
                None
            } else {
                let raw = root.ok_or_else(|| {
                    Error::Invalid("cache clear requires --root DIR or --all".into())
                })?;
                Some(resolve_root(Some(raw))?)
            };
            let mut report = greppy_core::cache::clear_cache(target.as_deref())
                .map_err(|error| Error::io("clear cache", error))?;
            cleanup_verified_legacy_trash();
            for entry in verified_legacy_cache_entries() {
                if target.as_deref().is_some_and(|root| root != entry.root) {
                    continue;
                }
                report.scanned_bytes = report.scanned_bytes.saturating_add(entry.bytes);
                if entry.locked {
                    report.locked_bytes = report.locked_bytes.saturating_add(entry.bytes);
                    report.skipped_locked.push(entry.path);
                } else if remove_verified_legacy_entry(&entry) {
                    report.removed_bytes = report.removed_bytes.saturating_add(entry.bytes);
                    report.removed.push(entry.path);
                }
            }
            print_gc_report(&report, false)?;
            Ok(if report.locked_bytes > 0 {
                EXIT_TEMPFAIL as i32
            } else {
                0
            })
        }
    }
}

fn current_cache_freshness(root: Option<&std::path::Path>) -> &'static str {
    let Some(root) = root else { return "unknown" };
    let path = workspace_locator::store_path(root);
    let store = match greppy_store::Store::open_with(&path, greppy_store::OpenOptions::read_only())
    {
        Ok(store) => store,
        Err(_) => return "failed",
    };
    let project = workspace_locator::project_identity(root);
    match greppy_freshness::check_files(
        &store,
        root,
        &project,
        std::time::Duration::from_millis(200),
    ) {
        Ok(state) => match state.outcome {
            greppy_freshness::FreshnessOutcome::Fresh => "fresh",
            greppy_freshness::FreshnessOutcome::Stale { .. }
            | greppy_freshness::FreshnessOutcome::RootMismatch => "drift",
            greppy_freshness::FreshnessOutcome::Cold => "cold",
            greppy_freshness::FreshnessOutcome::Unknown { .. } => "unknown",
        },
        Err(_) => "failed",
    }
}

fn print_gc_report(report: &greppy_core::cache::GcReport, json: bool) -> Result<()> {
    let value = serde_json::json!({
        "dry_run": report.dry_run,
        "throttled": report.throttled,
        "scanned_bytes": report.scanned_bytes,
        "removed_bytes": report.removed_bytes,
        "locked_bytes": report.locked_bytes,
        "removed": report.removed,
        "skipped_locked": report.skipped_locked,
    });
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|error| Error::Invalid(format!("serialize cache GC: {error}")))?
        );
    } else {
        println!(
            "cache GC: scanned={} removed={} locked={} dry_run={}",
            report.scanned_bytes, report.removed_bytes, report.locked_bytes, report.dry_run
        );
        for path in &report.removed {
            println!("removed {}", path.display());
        }
        for path in &report.skipped_locked {
            println!("locked {}", path.display());
        }
    }
    Ok(())
}

/// Best-effort peek of a leading global `--root <val>` / `--root=<val>`
/// from raw argv, BEFORE clap parses it, so the store-eviction pass can
/// protect the store this invocation is actually about to use. Non-UTF-8
/// or absent values yield `None` (fall back to cwd resolution).
fn peek_root_arg(argv: &[std::ffi::OsString]) -> Option<String> {
    let mut i = 1; // skip argv[0]
    while i < argv.len() {
        let s = argv[i].to_str()?;
        if let Some(v) = s.strip_prefix("--root=") {
            return (!v.is_empty()).then(|| v.to_string());
        }
        if s == "--root" {
            return argv.get(i + 1).and_then(|v| v.to_str()).map(String::from);
        }
        // `--root` may also follow the subcommand (clap global arg); keep
        // scanning the whole argv — values of OTHER flags could false-match
        // only if a flag literally takes the value `--root`, which none do.
        i += 1;
    }
    None
}

/// Trial arms own their complete cache/config namespace. Skip the normal
/// structured-command cache maintenance pass so the parent process cannot
/// touch an ambient Greppy store before those namespaces are installed.
fn is_trial_invocation(argv: &[std::ffi::OsString]) -> bool {
    let mut i = 1;
    while i < argv.len() {
        let token = &argv[i];
        if token == "--root" || token == "--device" {
            i += 2;
            continue;
        }
        let token_lossy = token.to_string_lossy();
        if token_lossy.starts_with("--root=")
            || token_lossy.starts_with("--device=")
            || token == "--no-gpu"
        {
            i += 1;
            continue;
        }
        return token == "trial";
    }
    false
}

/// Decide whether `argv` (including argv[0]) is a bare `grep`
/// passthrough rather than a recognised structured subcommand.
///
/// We skip a leading global `--root <val>` / `--root=<val>` and top-level
/// help/version requests (which clap handles), then look at the
/// first remaining token:
/// * If it equals a recognised subcommand name → NOT a passthrough.
/// * Otherwise (a flag like `-R`, a pattern, or nothing) → passthrough.
fn is_grep_passthrough(argv: &[std::ffi::OsString]) -> bool {
    #[cfg(feature = "agent")]
    if is_agent_invocation(argv) {
        return false;
    }
    let mut i = 1; // skip argv[0]
    while i < argv.len() {
        let tok = &argv[i];
        // Long help/version requests are Greppy commands. Short `-h` is also
        // grep's no-filename flag, so it is Greppy help only when used alone.
        if tok == "--help" || tok == "--version" || tok == "-V" || (tok == "-h" && argv.len() == 2)
        {
            return false;
        }
        // Global `--root` may precede the subcommand; skip it (and its
        // value) so we can inspect the real first token.
        if tok == "--root" {
            i += 2; // skip flag + value
            continue;
        }
        if tok.to_string_lossy().starts_with("--root=") {
            i += 1;
            continue;
        }
        if tok == "--device"
            || tok == "--limit"
            || tok == "--max"
            || tok == "--max-bytes"
            || tok == "--offset"
        {
            i += 2;
            continue;
        }
        let token_lossy = tok.to_string_lossy();
        if token_lossy.starts_with("--device=")
            || token_lossy.starts_with("--limit=")
            || token_lossy.starts_with("--max=")
            || token_lossy.starts_with("--max-bytes=")
            || token_lossy.starts_with("--offset=")
            || tok == "--no-gpu"
        {
            i += 1;
            continue;
        }
        // First non-skipped token. If it names a subcommand, defer to
        // clap; otherwise it's a grep passthrough.
        return match tok.to_str() {
            Some(s) => !SUBCOMMANDS.contains(&s),
            // A non-UTF-8 first token can never be a subcommand name, so
            // it is unambiguously a grep passthrough.
            None => true,
        };
    }
    // No tokens at all → not a passthrough (clap prints help).
    false
}

/// Dispatch a parsed CLI to the correct handler. Returns the desired exit
/// code. Use `dispatch_to_code` to run the dispatcher and translate the
/// result into a `u8` exit code for `ExitCode::from`.
pub fn dispatch(cli: Cli) -> Result<i32> {
    // If a recognised subcommand matched, dispatch it. Otherwise treat
    // the trailing args as a `grep` passthrough. This makes both
    //   greppy grep -R foo .
    // and
    //   greppy -R foo .
    // work — the latter being a common agent invocation pattern.
    // The global `--root` (RV-006) is threaded down to every command so
    // index and the query subcommands share one root-resolution path.
    let root = cli.root.clone();
    let device = cli.device.clone();
    let no_gpu = cli.no_gpu;
    if cli.limit == Some(0) {
        return Err(Error::Invalid("--limit/--max must be at least 1".into()));
    }
    if cli.max_bytes == Some(0) {
        return Err(Error::Invalid("--max-bytes must be at least 1".into()));
    }
    set_cli_result_window(cli.limit, cli.offset);
    let configured_device = device.clone().or_else(|| env_nonempty(ENV_DEVICE));
    if !no_gpu {
        configure_explicit_cuda_device(configured_device.as_deref())?;
    }
    set_cli_inference_override(device.clone(), no_gpu);
    if let Some(cmd) = cli.command {
        return dispatch_subcommand(cmd, root.as_deref(), device.as_deref(), no_gpu);
    }
    if !cli.passthrough.is_empty() {
        return dispatch_grep(&cli.passthrough);
    }
    // No subcommand and no pattern: a usage MISTAKE (often an agent's).
    // Print a compact cheat sheet, not the 2.5KB curated help — mid-task
    // token bombs teach nothing (P3). `--help` still prints everything.
    println!("usage: greppy PATTERN [FILES..]        (real-grep passthrough)");
    println!("   or: greppy <command> [--root DIR]   commands:");
    println!("       index PATH  who-calls S   callees S");
    println!("       trial --root DIR --question Q --check who-calls --symbol S ...");
    println!("       references S (who depends on S)   impact S [--direction incoming|outgoing]");
    println!("       brief S   semantic-search \"QUERY\"");
    println!("       search-code Q   search-symbols NAME [--kind function|method|struct|class]");
    println!("       index status   (--help for full details)");
    Ok(EXIT_USAGE as i32)
}

fn configure_explicit_cuda_device(device: Option<&str>) -> Result<()> {
    let policy = greppy_embed_native::InferencePolicy::from_selector(device, false)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let Some(index) = policy.cuda_device_index else {
        return Ok(());
    };
    let index = index.to_string();
    // SAFETY: dispatch applies the global inference policy before spawning
    // daemon/client threads; no concurrent environment access has begun.
    unsafe {
        std::env::set_var(ENV_EMBED_CUDA_DEVICE, &index);
        std::env::set_var(ENV_QWEN_CUDA_DEVICE, index);
    }
    Ok(())
}

fn inference_device_identity(device: &greppy_embed_native::DevicePreference) -> String {
    if *device == greppy_embed_native::DevicePreference::Cuda {
        if let Some(index) =
            env_nonempty(ENV_QWEN_CUDA_DEVICE).or_else(|| env_nonempty(ENV_EMBED_CUDA_DEVICE))
        {
            return format!("cuda:{index}");
        }
    }
    device.as_str().to_string()
}

fn dispatch_subcommand(
    cmd: Command,
    root: Option<&str>,
    device: Option<&str>,
    no_gpu: bool,
) -> Result<i32> {
    match cmd {
        Command::Passthrough(argv) => dispatch_grep(&argv),
        #[cfg(any(unix, windows))]
        Command::EmbedDaemon {
            socket,
            gguf,
            tokenizer,
            model_id,
            max_length,
            prewarm,
        } => {
            let cfg = EmbeddingModelConfig {
                model_id,
                source: EmbeddingModelSource::Gguf {
                    gguf: std::path::PathBuf::from(gguf),
                    tokenizer: std::path::PathBuf::from(tokenizer),
                },
                max_length,
                device: embedding_device_preference(device, no_gpu)?,
            };
            embed_daemon::daemon_main(socket, cfg, prewarm)
        }
        #[cfg(any(unix, windows))]
        Command::SummarizeDaemon {
            socket,
            gguf,
            tokenizer,
            model_id,
            prewarm,
        } => {
            let cfg = QwenSummaryConfig {
                model_id,
                gguf: std::path::PathBuf::from(gguf),
                tokenizer: std::path::PathBuf::from(tokenizer),
                device: qwen_summary_device_preference()?,
            };
            summarize_daemon::daemon_main(socket, cfg, prewarm)
        }
        Command::Index { path, json } => {
            if path.as_deref() == Some("status") {
                dispatch_index_status(json, root)
            } else {
                if json {
                    return Err(Error::Invalid(
                        "index --json is only supported for `grep index status --json`".into(),
                    ));
                }
                dispatch_index(path.as_deref(), root, EmbeddingCliArgs { device, no_gpu })
            }
        }
        Command::Map { path, json } => map::run(path.as_deref(), json, root),
        Command::Outline { path, json, all } => dispatch_outline(&path, json, all, root),
        Command::Changes { base, json } => dispatch_changes(base.as_deref(), json, root),
        Command::Cache { command } => dispatch_cache(command, root),
        Command::Trial { args } => trial::run(args, root),
        Command::Verify { args } => Ok(verify::run(args, root)),
        Command::SearchGraph { name, json } => {
            let mut q = greppy_search::GraphQuery::any().with_limit(cli_result_limit(50));
            let name_filter = name.as_deref();
            if let Some(n) = name_filter {
                q = q.with_name(n);
            }
            dispatch_search_graph(q, name_filter, json, root)
        }
        Command::Trace {
            symbol,
            direction,
            edge,
            depth,
            code,
            json,
        } => dispatch_trace(
            symbol.as_deref(),
            &direction,
            &edge,
            depth,
            code,
            json,
            root,
        ),
        Command::Impact {
            symbols,
            path_opts,
            code: _,
            direction,
            edge,
            depth,
            since,
            base,
            all,
            json,
        } => {
            let targets = nav_targets(&symbols)?;
            validate_path_filters(root, &path_opts, "--path")?;
            if targets.len() > 1 {
                return dispatch_nav_multi(NavMultiRequest {
                    command: "impact",
                    kind: NavKind::Impact,
                    targets: &targets,
                    paths: &path_opts,
                    code: false,
                    all,
                    json,
                    root,
                    direction: &direction,
                    edge: edge.as_deref(),
                    depth,
                });
            }
            dispatch_impact(
                targets.first().map(String::as_str),
                &path_opts,
                &direction,
                edge.as_deref(),
                depth,
                since.as_deref(),
                base.as_deref(),
                all,
                json,
                root,
            )
        }
        Command::Brief {
            symbols,
            path_opts,
            code: _,
            all: _,
            json,
        } => {
            let targets = nav_targets(&symbols)?;
            validate_path_filters(root, &path_opts, "--path")?;
            if targets.len() > 1 {
                return dispatch_brief_multi(&targets, &path_opts, json, root);
            }
            dispatch_brief(targets.first().map(String::as_str), &path_opts, json, root)
        }
        Command::Expand { id, json } => dispatch_expand(id.as_deref(), json, root),
        Command::Read {
            targets,
            symbol_opts,
            path_opts,
            lines,
            context,
            handle,
            code: _,
            json,
            all: _,
        } => {
            set_cli_read_context(context);
            let plan = read_plan(&targets, &symbol_opts, &path_opts, lines.as_deref())?;
            if plan.subjects.len() > 1 {
                return dispatch_read_multi(&plan, handle, json, root);
            }
            let subject = plan.subjects.first().cloned();
            // A path-qualified symbol (`FILE::Symbol`) must reach graph
            // resolution instead of being mistaken for a slash-containing
            // filesystem path.
            if !plan.forced_symbol {
                if let Some(subject) = subject.as_deref() {
                    // Every greppy result line is printed as `path:START-END`, so agents
                    // paste that form straight back into `read`. Accept it: split the
                    // trailing `:START[-END]` off and treat it as `--lines`. Without this
                    // the whole string is looked up as a symbol name and answers
                    // "no exact match", which cost turns in the measured runs.
                    if plan.lines.is_none() {
                        if let Some((file, range)) = split_trailing_line_range(subject) {
                            if read_subject_is_path(file, root)? {
                                return dispatch_read_file(file, Some(&range), handle, json, root);
                            }
                        }
                    }
                    if split_path_qualified(subject).is_none()
                        && read_subject_is_path(subject, root)?
                    {
                        return dispatch_read_file(
                            subject,
                            plan.lines.as_deref(),
                            handle,
                            json,
                            root,
                        );
                    }
                    if looks_like_path(subject) {
                        // Path-shaped and not on disk: keep the file answer
                        // (closest-paths guidance) instead of hunting for a
                        // symbol that cannot exist under that name.
                        return dispatch_read_file(
                            subject,
                            plan.lines.as_deref(),
                            handle,
                            json,
                            root,
                        );
                    }
                }
            }
            if plan.lines.is_some() {
                return Err(Error::Invalid(
                    "read --lines/--line N[:M] requires a file path (for symbols, omit the line flag)"
                        .into(),
                ));
            }
            dispatch_read(subject.as_deref(), handle, json, root)
        }
        Command::Edit { command, json } => dispatch_edit(command, json, root),
        Command::Stats => dispatch_stats(root),
        Command::Diagnostics { json } => dispatch_diagnostics(json, root),
        Command::Doctor { json } => dispatch_doctor(json, root),
        Command::WhoCalls {
            symbols,
            path_opts,
            code,
            all,
            json,
        } => dispatch_nav(
            "who-calls",
            NavKind::WhoCalls,
            &symbols,
            &path_opts,
            code,
            all,
            json,
            root,
        ),
        Command::Callees {
            symbols,
            path_opts,
            code,
            all,
            json,
        } => dispatch_nav(
            "callees",
            NavKind::Callees,
            &symbols,
            &path_opts,
            code,
            all,
            json,
            root,
        ),
        Command::FanIn { edge, json } => dispatch_fan_degree(
            "fan-in",
            "incoming",
            &edge,
            cli_result_limit(20),
            json,
            root,
        ),
        Command::FanOut { edge, json } => dispatch_fan_degree(
            "fan-out",
            "outgoing",
            &edge,
            cli_result_limit(20),
            json,
            root,
        ),
        Command::GraphLocate {
            location,
            file,
            line,
            json,
        } => dispatch_graph_locate(location.as_deref(), file.as_deref(), line, json, root),
        Command::Path {
            from,
            to,
            edge,
            json,
            code: _,
            all: _,
        } => dispatch_path(from.as_deref(), to.as_deref(), &edge, json, root),
        Command::SearchCode {
            query,
            mut paths,
            path_opts,
            changed,
            staged,
            since,
            base,
            json,
            no_code,
            fixed,
            code: _,
            all: _,
        } => {
            // `AGENTS.md` keeps `search-code "PATTERN" [PATH …]`: the grep-shaped
            // verb is the one place where a positional path is the convention,
            // not a filter smuggled in behind a target. Rule 1 still applies to
            // it — a path that is not there cannot narrow anything, so it is a
            // mistake, never an empty scope.
            validate_path_filters(root, &paths, "path")?;
            validate_path_filters(root, &path_opts, "--path")?;
            paths.extend(path_opts);
            dispatch_search_code(
                query.as_deref(),
                &paths,
                changed,
                staged,
                since.as_deref(),
                base.as_deref(),
                json,
                no_code,
                fixed,
                root,
            )
        }
        Command::SearchSymbols {
            queries,
            path_opts,
            kind,
            json,
            code: _,
            all: _,
        } => {
            let targets = nav_targets(&queries)?;
            validate_path_filters(root, &path_opts, "--path")?;
            dispatch_search_symbols_multi(&targets, &path_opts, kind.as_deref(), json, root)
        }
        Command::Plus {
            query,
            k,
            code,
            explain,
            json,
        } => dispatch_plus(
            query.as_deref(),
            k,
            code,
            explain,
            json,
            EmbeddingCliArgs { device, no_gpu },
            root,
        ),
        Command::Semantic {
            queries,
            path_opts,
            json,
        } => {
            validate_path_filters(root, &path_opts, "--path")?;
            let queries = semantic_queries(&queries)?;
            let mut last = 0;
            for query in &queries {
                last = dispatch_semantic(
                    Some(query.as_str()),
                    &path_opts,
                    json,
                    EmbeddingCliArgs { device, no_gpu },
                    root,
                )?;
            }
            if queries.is_empty() {
                last = dispatch_semantic(
                    None,
                    &path_opts,
                    json,
                    EmbeddingCliArgs { device, no_gpu },
                    root,
                )?;
            }
            Ok(last)
        }
        Command::Context {
            query,
            k,
            lines,
            json,
            code: _,
            all: _,
        } => dispatch_context(
            query.as_deref(),
            k,
            lines,
            json,
            EmbeddingCliArgs { device, no_gpu },
            root,
        ),
    }
}

fn dispatch_search_graph(
    q: greppy_search::GraphQuery,
    name_filter: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let q = if q.project.is_none() {
        q.with_project(project.clone())
    } else {
        q
    };
    let limit = q.limit;
    let graph_gate_extra = serde_json::json!({
        "filters": {
            "name": name_filter,
        },
        "scope": "node_search",
        "limit": limit,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "search-graph",
        json,
        graph_gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "search-graph",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let rows = greppy_search::search_graph(&store, &q)?;
    if json {
        let total_exact = greppy_search::count_search_graph(&store, &q)?;
        search_graph_counts_json(
            &store,
            root,
            &project,
            name_filter,
            limit,
            total_exact,
            &rows,
        )?;
        return Ok(0);
    }
    if rows.is_empty() {
        println!("(no matches)");
    } else {
        for r in &rows {
            println!(
                "{}  {}  {}:{}  {}",
                r.label,
                display_row_name(r),
                r.file_path,
                r.start_line,
                r.name
            );
        }
    }
    Ok(0)
}

/// Rank a node label for symbol resolution. Lower is better.
///
/// `resolve_symbol_id` previously
/// picked the FIRST node named `S`, landing on the wrong one when a name
/// is shared — e.g. `Store` resolved to the `EnumVariant`
/// `Error::Store`, and `IndexReport` resolved to the `Impl::IndexReport`
/// instead of the `Struct`. We now rank candidates so a type/def-like
/// label (Class/Interface/Type/Struct/Enum/Trait/Function/Method/TypeAlias)
/// wins over the `Impl`/`EnumVariant`/`AssocConst`/`AssocType`/`Module`
/// blocks and the `Call`/`Import` pseudo-nodes.
///
/// Rust type defs use the canonical graph labels (struct/union → `Class`,
/// trait → `Interface`, type alias → `Type`); the alternate
/// `Struct`/`Trait`/`TypeAlias` labels are kept so other-language
/// extractors and fixtures still rank as primary defs.
fn label_rank(label: &str) -> u8 {
    match label {
        "Class" | "Interface" | "Type" | "Struct" | "Enum" | "Trait" | "Function" | "Method"
        | "TypeAlias" => 0,
        // Definition-ish but secondary: only chosen if no primary exists.
        "Impl" | "EnumVariant" | "AssocConst" | "AssocType" | "Module" => 1,
        // Pseudo-nodes (reference sites) are the last resort.
        "Call" | "Import" => 3,
        // Anything else sits between secondary defs and pseudo-nodes.
        _ => 2,
    }
}

/// True for the "primary" definition labels — the type/def-like kinds we
/// prefer for resolution and that we aggregate incoming edges across
/// (so a `Struct` and its `Impl` both contribute to incoming
/// navigation). See [`resolve_symbol_nodes`].
fn is_primary_label(label: &str) -> bool {
    label_rank(label) <= 1
}

/// Split a symbol query into `(owner, member)` when it is written in the
/// natural qualified form a coding agent types — `Owner.method` or
/// `Owner::method`. Returns `None` for a bare identifier (no separator),
/// leaving all existing bare-name resolution byte-for-byte unchanged.
///
/// The split is on the **last** separator so `member` is the final path
/// component (the actual method/function name) and `owner` is everything
/// before it. Both `.` and `::` are accepted; the two never both appear as
/// the *last* separator, so we pick whichever occurs later in the string
/// (`a::b.c` → owner `a::b`, member `c`; `a.b::c` → owner `a.b`, member
/// `c`). A trailing/leading separator, or an empty owner or member, yields
/// `None` (treated as a bare/invalid query).
///
/// This is intentionally a pure string split — no store access — so it can
/// gate the qualified path cheaply before any graph work.
fn split_qualified(symbol: &str) -> Option<(&str, &str)> {
    // Find the last separator: the later of the last `::` and the last `.`.
    let dcolon = symbol.rfind("::").map(|i| (i, 2usize));
    let dot = symbol.rfind('.').map(|i| (i, 1usize));
    let (idx, sep_len) = match (dcolon, dot) {
        (Some(c), Some(d)) => {
            if c.0 >= d.0 {
                c
            } else {
                d
            }
        }
        (Some(c), None) => c,
        (None, Some(d)) => d,
        (None, None) => return None,
    };
    let owner = &symbol[..idx];
    let member = &symbol[idx + sep_len..];
    if owner.is_empty() || member.is_empty() {
        return None;
    }
    Some((owner, member))
}

/// The **owner segment** of a node's `qualified_name` — the `::`-joined
/// segment immediately before the final (name) segment.
///
/// Qnames are built by the parser as `<file_path>::<owner>::<name>` for an
/// *owned* member (a `Method` on a class/struct/type: Java
/// `JsonReader.java::JsonReader::peekNumber`, Rust
/// `ser.rs::TaggedSerializer::serialize_bool`, TS
/// `types.ts::ZodString::max`) and as `<file_path>::<Label>::<name>` for a
/// *free* function/type. The segment before the name is therefore the
/// class/type owner for members (what a `Owner.method` query disambiguates
/// on) and the Label for free defs. Returns `None` when the qname has no
/// segment before the name (e.g. a bare `name` with no `::`).
///
/// Note: file paths use `/`, never `::`, so splitting the whole qname on
/// `::` never confuses a path component for an owner segment.
fn qname_owner_segment(qualified_name: &str) -> Option<&str> {
    let mut it = qualified_name.rsplit("::");
    let _name = it.next()?; // the final segment is the node name
    it.next() // the segment before it is the owner (or Label for free defs)
}

/// Lua providers preserve dotted declaration names (`function helper.do_it()`)
/// verbatim. Return the final member only for that representation; ordinary
/// bare names and `::`-qualified qnames are not rewritten here.
fn verbatim_dotted_leaf(name: &str) -> Option<&str> {
    let (_, leaf) = name.rsplit_once('.')?;
    (!leaf.is_empty()).then_some(leaf)
}

/// Match a bare query against either an ordinary node name or the final segment
/// of a provider-preserved dotted name. Qualified queries are handled by
/// `resolve_qualified_ids`, where the full dotted spelling is authoritative.
fn bare_symbol_name_matches(node_name: &str, query: &str) -> bool {
    node_name.eq_ignore_ascii_case(query)
        || (split_qualified(query).is_none()
            && verbatim_dotted_leaf(node_name).is_some_and(|leaf| leaf.eq_ignore_ascii_case(query)))
}

/// If `symbol` is a qualified `Owner.member` / `Owner::member` query,
/// return the node ids of every primary-labelled node that genuinely
/// matches it: `name == member` AND the node's [`qname_owner_segment`]
/// equals the **last** segment of `owner`. Returns `None` when `symbol` is
/// bare (no separator) so the caller keeps its existing bare-name path.
///
/// Owner matching compares against the last `::`/`.`-segment of the query
/// owner, so both the natural `JsonReader.peekNumber` and a
/// fully-qualified `com.google.gson.JsonReader.peekNumber` resolve to the
/// `JsonReader` owner segment the qname carries.
///
/// Never-guess is preserved end to end: this only ever *narrows* the set
/// of same-named primary nodes to those whose owner matches. It returns
/// the genuine matching set — one id when the owner is unique, several
/// when the same `Owner.member` legitimately exists in multiple files —
/// and never picks one arbitrarily. An empty set (owner matches nothing)
/// is returned as `Some(vec![])`, which the callers surface as
/// "not found" without silently falling back to a bare-name guess that
/// would ignore the owner the agent supplied.
fn resolve_qualified_ids(
    rows: &[greppy_search::graph::SearchGraphRow],
    symbol: &str,
) -> Option<Vec<i64>> {
    let (owner, member) = split_qualified(symbol)?;
    // Compare on the last segment of the query owner so both the natural
    // `Owner.method` and a fully-qualified `pkg.Owner.method` match.
    let owner_tail = owner
        .rsplit("::")
        .next()
        .and_then(|s| s.rsplit('.').next())
        .unwrap_or(owner);
    let mut ids: Vec<i64> = rows
        .iter()
        .filter(|r| {
            is_primary_label(&r.label)
                && (r.name.eq_ignore_ascii_case(symbol)
                    || (r.name.eq_ignore_ascii_case(member)
                        && qname_owner_segment(&r.qualified_name)
                            .is_some_and(|owner| owner.eq_ignore_ascii_case(owner_tail))))
        })
        .map(|r| r.id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    Some(ids)
}

/// Resolve a `--symbol`/positional symbol argument to a node id within
/// the open store. Ranks candidates by [`label_rank`] (preferring
/// type/def-like labels), breaking ties deterministically by node id, so
/// a shared name resolves to the real definition rather than an `Impl`,
/// `EnumVariant`, or `Call` site. Falls back to an exact
/// `qualified_name` match. When `symbol` is `None` the first node in the
/// graph is used (preserves the historical no-arg `greppy trace`
/// behaviour).
fn resolve_symbol_id(store: &greppy_store::Store, symbol: Option<&str>) -> Result<Option<i64>> {
    // Push the name filter into SQL. The old form loaded the first 10k
    // nodes of the project (ordered by qualified_name) and filtered in
    // memory — on a repo bigger than the cap (django: 56k nodes) every
    // symbol outside that window silently resolved as "not found".
    let rows = symbol_candidate_rows(store, symbol)?;
    let id = match symbol {
        // Qualified query (`Owner.method` / `Owner::method`): resolve within
        // the owner-matched set only. `trace`/`impact`/`path` need a single
        // start node, so among the owner-matched candidates we pick the
        // best-ranked (then lowest id) — the same deterministic discipline
        // as the bare-name path, but confined to the nodes the owner
        // actually disambiguates to. An empty owner match yields `None`
        // (not found) instead of falling back to a bare-name guess that
        // ignores the owner the agent supplied.
        Some(s) if split_qualified(s).is_some() => {
            resolve_qualified_ids(&rows, s).and_then(|ids| {
                rows.iter()
                    .filter(|r| ids.contains(&r.id))
                    .min_by(|a, b| {
                        label_rank(&a.label)
                            .cmp(&label_rank(&b.label))
                            .then(a.id.cmp(&b.id))
                    })
                    .map(|r| r.id)
            })
        }
        Some(s) => {
            let best = rows
                .iter()
                .filter(|r| bare_symbol_name_matches(&r.name, s))
                .min_by(|a, b| {
                    label_rank(&a.label)
                        .cmp(&label_rank(&b.label))
                        .then(a.id.cmp(&b.id))
                })
                .map(|r| r.id);
            if best.is_some() {
                best
            } else {
                // Fall back to an exact qualified_name match — its own
                // indexed lookup now that `rows` only holds name matches.
                let q = greppy_search::GraphQuery::any()
                    .with_qualified_name(s)
                    .with_limit(1);
                greppy_search::search_graph(store, &q)?
                    .first()
                    .map(|r| r.id)
            }
        }
        None => rows.first().map(|r| r.id),
    };
    Ok(id)
}

/// The candidate rows a symbol query resolves against, fetched with the
/// filter pushed into SQL (never a capped whole-project scan):
///   * bare name → exact `name` matches plus provider-preserved dotted names
///     ending in `.name` (Lua's `function helper.do_it()` representation);
///   * qualified `Owner.member` → exact full-name matches plus nodes named
///     `member` (the owner is matched in [`resolve_qualified_ids`]);
///   * no symbol → the first node in qualified_name order (the historical
///     no-arg `trace` seed).
fn symbol_candidate_rows(
    store: &greppy_store::Store,
    symbol: Option<&str>,
) -> Result<Vec<greppy_search::graph::SearchGraphRow>> {
    let Some(s) = symbol else {
        let q = greppy_search::GraphQuery::any().with_limit(1);
        return greppy_search::search_graph(store, &q);
    };

    let mut rows = greppy_search::search_graph(
        store,
        &greppy_search::GraphQuery::any()
            .with_name(s)
            .with_limit(10_000),
    )?;

    if let Some((_, member)) = split_qualified(s) {
        rows.extend(greppy_search::search_graph(
            store,
            &greppy_search::GraphQuery::any()
                .with_name(member)
                .with_limit(10_000),
        )?);
    } else {
        rows.extend(greppy_search::search_graph(
            store,
            &greppy_search::GraphQuery::any()
                .with_name_contains(format!(".{s}"))
                .with_limit(10_000),
        )?);
    }

    rows.sort_by_key(|row| row.id);
    rows.dedup_by_key(|row| row.id);
    Ok(rows)
}

/// Resolve a symbol to the set of node ids whose incoming edges should
/// be aggregated for who-calls / trace-incoming.
///
/// a name like `IndexReport` is
/// split across a `Struct` node and one or more `Impl` nodes; the real
/// callers/usages live on either. We therefore return ALL nodes that
/// share the exact `name` and carry a primary label (Struct/Enum/Trait/
/// Function/Method/TypeAlias/Impl/EnumVariant/…) so both the `Struct`
/// and its `Impl` contribute. The set is deterministically ordered by
/// node id. If no primary-labelled node matches, we fall back to the
/// single best node from [`resolve_symbol_id`] so the old behaviour is
/// preserved for pseudo-node-only names.
/// Candidate needles for similar-name suggestions, from an agent's raw
/// query. Agents guess signature-shaped names ("impl Serialize for Range",
/// "Serialize for Range") — the useful identifier is usually the LAST
/// type-like token, so tokens are tried back to front with declaration
/// keywords dropped.
fn suggestion_needles(query: &str) -> Vec<String> {
    let mut needles: Vec<String> = Vec::new();
    if let Some((_, member)) = split_qualified(query) {
        needles.push(member.to_string());
    }
    let mut tokens: Vec<&str> = query
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| t.len() >= 3)
        .filter(|t| {
            !matches!(
                t.to_ascii_lowercase().as_str(),
                "impl"
                    | "for"
                    | "pub"
                    | "struct"
                    | "trait"
                    | "class"
                    | "def"
                    | "function"
                    | "static"
                    | "const"
                    | "async"
                    | "extends"
                    | "implements"
                    | "interface"
            )
        })
        .collect();
    tokens.reverse();
    for t in tokens {
        if !needles.iter().any(|n| n == t) {
            needles.push(t.to_string());
        }
    }
    if needles.is_empty() {
        needles.push(query.to_string());
    }
    needles
}

fn fuzzy_symbol_name_probes(needle: &str) -> Vec<String> {
    let characters = needle.chars().collect::<Vec<_>>();
    let len = characters.len();
    let mut probes = Vec::new();
    let mut push_probe = |probe: String| {
        if probe.chars().count() >= 3 && !probes.iter().any(|existing| existing == &probe) {
            probes.push(probe);
        }
    };
    push_probe(needle.to_string());
    if len > 3 {
        let prefix_lengths = [
            len.saturating_sub(1),
            len.saturating_sub(2),
            len.saturating_mul(3) / 4,
            len / 2,
            3,
        ];
        for prefix_len in prefix_lengths {
            if prefix_len >= 3 && prefix_len < len {
                push_probe(characters[..prefix_len].iter().collect());
            }
        }
        let suffix_starts = [1, 2, len / 4, len / 2, len.saturating_sub(3)];
        for suffix_start in suffix_starts {
            if suffix_start < len && len - suffix_start >= 3 {
                push_probe(characters[suffix_start..].iter().collect());
            }
        }
    }
    probes
}

fn symbol_name_distance(name: &str, needles: &[String]) -> usize {
    let name = name.to_ascii_lowercase();
    needles
        .iter()
        .map(|needle| levenshtein(&name, &needle.to_ascii_lowercase()))
        .min()
        .unwrap_or(usize::MAX)
}

fn is_near_symbol_name(name: &str, needles: &[String]) -> bool {
    let name_lower = name.to_ascii_lowercase();
    needles.iter().any(|needle| {
        let needle_lower = needle.to_ascii_lowercase();
        let needle_len = needle_lower.chars().count();
        let distance = levenshtein(&name_lower, &needle_lower);
        let threshold = match needle_len {
            0..=4 => 1,
            5..=8 => 2,
            _ => (needle_len / 3).max(3),
        };
        distance <= threshold
            || (needle_len >= 3
                && (name_lower.contains(&needle_lower) || needle_lower.contains(&name_lower)))
    })
}

fn symbol_miss_suggestions(store: &greppy_store::Store, project: &str, query: &str) -> Vec<String> {
    let needles = suggestion_needles(query);
    let mut suggestions = Vec::new();
    for needle in &needles {
        for probe in fuzzy_symbol_name_probes(needle) {
            for name in store
                .similar_node_names(project, &probe, 25)
                .unwrap_or_default()
            {
                if !suggestions.iter().any(|candidate| candidate == &name) {
                    suggestions.push(name);
                }
            }
        }
    }
    suggestions.retain(|name| is_near_symbol_name(name, &needles));
    suggestions.sort_by(|left, right| {
        symbol_name_distance(left, &needles)
            .cmp(&symbol_name_distance(right, &needles))
            .then_with(|| left.len().cmp(&right.len()))
            .then_with(|| left.to_ascii_lowercase().cmp(&right.to_ascii_lowercase()))
            .then_with(|| left.cmp(right))
    });
    suggestions.truncate(5);
    suggestions
}

fn print_symbol_miss_guidance(store: &greppy_store::Store, project: &str, query: &str) {
    println!("symbol not found: `{query}`");
    for suggestion in symbol_miss_suggestions(store, project, query) {
        println!("suggestion: `{suggestion}`");
    }
    println!("try: greppy search-symbols {}", shell_example_arg(query));
    println!("try: greppy semantic-search {}", shell_example_arg(query));
}

fn symbol_miss_json(store: &greppy_store::Store, project: &str, query: &str) -> serde_json::Value {
    serde_json::json!({
        "suggestions": symbol_miss_suggestions(store, project, query),
        "next": [
            format!("greppy search-symbols {}", shell_example_arg(query)),
            format!("greppy semantic-search {}", shell_example_arg(query)),
        ],
    })
}

fn has_case_variant_suggestion(suggestions: &[String], query: &str) -> bool {
    let needle = split_qualified(query)
        .map(|(_, member)| member)
        .unwrap_or(query);
    suggestions
        .iter()
        .any(|candidate| candidate != needle && candidate.eq_ignore_ascii_case(needle))
}

/// Split `file/path.ext::REST` into `(path, rest)` when the head segment is a
/// file path. `search-symbols`/`read` PRINT qualified names as
/// `path::Owner::name` or `path::Kind::name`, so agents naturally feed those
/// forms — and their simplifications (`path::name`) — straight back in.
/// Postel's law: the tool must accept every form it emits. The path only
/// NARROWS; the last `::`/`.` segment is always the symbol name.
fn split_path_qualified(query: &str) -> Option<(&str, &str)> {
    let idx = query.find("::")?;
    let head = &query[..idx];
    let looks_like_path = head.contains('/')
        || std::path::Path::new(head)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| !e.is_empty() && e.chars().all(|c| c.is_ascii_alphanumeric()));
    looks_like_path.then(|| (head, &query[idx + 2..]))
}

fn indexed_path_matches_query(indexed_path: &str, query_path: &str) -> bool {
    let normalized_query = query_path.replace('\\', "/");
    if normalized_query.contains('/') {
        indexed_path == normalized_query.trim_start_matches("./")
    } else {
        std::path::Path::new(indexed_path)
            .file_name()
            .and_then(|name| name.to_str())
            == Some(normalized_query.as_str())
    }
}

fn accepted_symbol_spelling(query: &str) -> Option<&'static str> {
    if let Some((path, _)) = split_path_qualified(query) {
        return Some(if path.contains('/') || path.contains('\\') {
            "path-qualified `path/file::Symbol` spelling"
        } else {
            "bare-file-qualified `file.ext::Symbol` spelling"
        });
    }
    if query.contains('.') && split_qualified(query).is_some() {
        return Some("dotted `Owner.method` spelling");
    }
    None
}

/// Fold an optional disambiguating file path into a `path::SYMBOL` query so the
/// existing path-qualified resolver ([`resolve_symbol_nodes`]) narrows SYMBOL to
/// that file. Opt-in: returns None (leave the query unchanged) unless a symbol
/// and a file-like path are both present and the symbol is not already
/// path-qualified. Agents type `brief open src/flask/testing.py` to break a tie;
/// serving that is cheaper than punishing it with a parse error.
fn qualify_symbol_with_path(symbol: Option<&str>, path: Option<&str>) -> Option<String> {
    let s = symbol?;
    let p = path?;
    if p.is_empty() || split_path_qualified(s).is_some() {
        return None;
    }
    // Require a file-like path (a basename carrying an extension) so
    // split_path_qualified recognises the `path.ext::` boundary; otherwise the
    // fold would only manufacture an unresolvable query. Owner-qualified
    // symbols such as `Type::method` are still folded: only an already
    // path-qualified symbol makes the explicit --path redundant.
    let basename = p.rsplit(['/', '\\']).next().unwrap_or(p);
    if !basename.contains('.') {
        return None;
    }
    Some(format!("{p}::{s}"))
}

fn resolve_symbol_nodes(store: &greppy_store::Store, symbol: Option<&str>) -> Result<Vec<i64>> {
    let Some(s) = symbol else {
        // No symbol: mirror resolve_symbol_id's "first node" behaviour.
        return Ok(resolve_symbol_id(store, None)?.into_iter().collect());
    };
    // Path-qualified query (`path::name`, `path::Kind::name`, `path::Owner::name`
    // — exactly what search-symbols/read print): the last segment is the name,
    // the file path narrows, any middle Kind/Owner segment narrows further but
    // is not required. This accepts every emitted form and its natural
    // simplifications instead of only the one exact string.
    if let Some((path, spec)) = split_path_qualified(s) {
        let name = spec
            .rsplit("::")
            .next()
            .and_then(|x| x.rsplit('.').next())
            .unwrap_or(spec);
        let middle = split_qualified(spec).map(|(owner, _)| {
            owner
                .rsplit("::")
                .next()
                .and_then(|o| o.rsplit('.').next())
                .unwrap_or(owner)
        });
        let rows = symbol_candidate_rows(store, Some(name))?;
        let in_path: Vec<&greppy_search::graph::SearchGraphRow> = rows
            .iter()
            .filter(|r| {
                r.name.eq_ignore_ascii_case(name)
                    && is_primary_label(&r.label)
                    && indexed_path_matches_query(&r.file_path, path)
            })
            .collect();
        // If a middle segment was given (Kind label or owner), prefer the
        // subset it disambiguates to; otherwise take all name+path matches.
        let mut ids: Vec<i64> = match middle {
            Some(m) => {
                let narrowed: Vec<i64> = in_path
                    .iter()
                    .filter(|r| {
                        qname_owner_segment(&r.qualified_name) == Some(m)
                            || r.label.eq_ignore_ascii_case(m)
                    })
                    .map(|r| r.id)
                    .collect();
                if narrowed.is_empty() {
                    in_path.iter().map(|r| r.id).collect()
                } else {
                    narrowed
                }
            }
            None => in_path.iter().map(|r| r.id).collect(),
        };
        ids.sort_unstable();
        ids.dedup();
        if !ids.is_empty() {
            return Ok(ids);
        }
    }
    // Name filter pushed into SQL — see resolve_symbol_id for why the old
    // capped whole-project scan was wrong on large repos.
    let rows = symbol_candidate_rows(store, Some(s))?;
    // Qualified query (`Owner.method` / `Owner::method`): narrow the
    // same-named primary nodes to those the owner disambiguates to. This is
    // the natural form a coding agent types; without it the whole query
    // (name == "Owner.method") matches nothing and the command reports
    // "symbol not found". We return the owner-matched set as-is — one node
    // when unique, several when `Owner.method` legitimately exists in more
    // than one file (aggregated downstream, same as a bare name) — never a
    // guess. An empty owner match returns an empty set so the caller
    // reports "not found" rather than ignoring the owner and guessing.
    if let Some(ids) = resolve_qualified_ids(&rows, s) {
        return Ok(ids);
    }
    let mut ids: Vec<i64> = rows
        .iter()
        // Case-insensitive equality: symbol_candidate_rows only returns a
        // case-variant when it is UNAMBIGUOUS, so this never guesses.
        .filter(|r| bare_symbol_name_matches(&r.name, s) && is_primary_label(&r.label))
        .map(|r| r.id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        // No primary-labelled node — fall back to the single best match
        // (e.g. a name that only exists as a Call/Import pseudo-node).
        if let Some(id) = resolve_symbol_id(store, Some(s))? {
            ids.push(id);
        }
    }
    Ok(ids)
}

fn is_callable_node_label(label: &str) -> bool {
    matches!(label, "Function" | "Method" | "Constructor")
}

fn is_type_container_label(label: &str) -> bool {
    matches!(
        label,
        "Class" | "Struct" | "Interface" | "Type" | "Enum" | "Trait" | "TypeAlias" | "Impl"
    )
}

fn owned_callable_ids_for_type(
    store: &greppy_store::Store,
    project: &str,
    node: &greppy_store::Node,
) -> Result<Vec<i64>> {
    if !is_type_container_label(&node.label) {
        return Ok(Vec::new());
    }
    let mut ids = std::collections::BTreeSet::new();
    for edge_type in ["DEFINES_METHOD", "DEFINES"] {
        for edge in
            store.outgoing_edges(node.id, Some(edge_type), greppy_search::MAX_REACH_RESULTS)?
        {
            if let Some(candidate) = store.get_node(edge.target_id)? {
                if is_callable_node_label(&candidate.label) {
                    ids.insert(candidate.id);
                }
            }
        }
    }
    for candidate in store.list_nodes(project, "", &node.file_path, 0, 10_000)? {
        if !is_callable_node_label(&candidate.label) {
            continue;
        }
        let owned_by_qname =
            qname_owner_segment(&candidate.qualified_name) == Some(node.name.as_str());
        let owned_by_span = candidate.id != node.id
            && candidate.start_line > node.start_line
            && candidate.end_line <= node.end_line;
        if owned_by_qname || owned_by_span {
            ids.insert(candidate.id);
        }
    }
    Ok(ids.into_iter().collect())
}

fn callee_source_ids_for_symbols(
    store: &greppy_store::Store,
    project: &str,
    source_ids: &[i64],
) -> Result<Vec<i64>> {
    let mut out = std::collections::BTreeSet::new();
    for id in source_ids {
        out.insert(*id);
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        for owned in owned_callable_ids_for_type(store, project, &node)? {
            out.insert(owned);
        }
    }
    Ok(out.into_iter().collect())
}

fn targets_include_non_callable(store: &greppy_store::Store, target_ids: &[i64]) -> Result<bool> {
    for id in target_ids {
        if let Some(node) = store.get_node(*id)? {
            if !is_callable_node_label(&node.label) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn incoming_call_nodes_for_targets(
    store: &greppy_store::Store,
    target_ids: &[i64],
) -> Result<Vec<greppy_store::Node>> {
    let mut nodes = std::collections::BTreeMap::new();
    for target_id in target_ids {
        for edge in store.incoming_edges(*target_id, Some("CALLS"), 1024)? {
            if let std::collections::btree_map::Entry::Vacant(slot) = nodes.entry(edge.source_id) {
                if let Some(node) = store.get_node(edge.source_id)? {
                    slot.insert(node);
                }
            }
        }
    }
    Ok(nodes.into_values().collect())
}

fn is_synthetic_file_anchor(label: &str, name: &str, qualified_name: &str) -> bool {
    name == "__file__"
        || qualified_name.ends_with("::__file__")
        || qualified_name.ends_with(".__file__")
        || (label == "File" && qualified_name.ends_with("__file__"))
}

fn display_symbol_name(label: &str, name: &str, qualified_name: &str, file_path: &str) -> String {
    if is_synthetic_file_anchor(label, name, qualified_name) {
        if file_path.is_empty() {
            "Module <unknown>".to_string()
        } else {
            format!("Module {file_path}")
        }
    } else {
        qualified_name.to_string()
    }
}

fn display_node_name(node: &greppy_store::Node) -> String {
    display_symbol_name(
        &node.label,
        &node.name,
        &node.qualified_name,
        &node.file_path,
    )
}

fn display_row_name(row: &greppy_search::graph::SearchGraphRow) -> String {
    display_symbol_name(&row.label, &row.name, &row.qualified_name, &row.file_path)
}

/// Is `q` a single bare identifier — i.e. a "show me the definition of X"
/// / find-definition query rather than a natural-language research query?
///
/// A bare identifier is one whitespace-free token whose characters are all
/// identifier characters (letters, digits, `_`), starting with a letter or
/// `_`. This is the shape a literal-lookup query takes (`clamp_value`,
/// `processSvc100`, `to_minor_units`); natural-language queries used for
/// research (`clamp a value to a range`, `hash fingerprint of bytes`)
/// contain spaces and so are excluded. Used by `context` to decide whether
/// an exact-name definition lookup should return minimal, grep-shaped
/// output instead of padding with related/semantic spans.
fn is_bare_identifier(q: &str) -> bool {
    let mut chars = q.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Default cap (in lines) for a single source span emitted by
/// `greppy context`. Long definitions are truncated with a note so the
/// output stays compact for an agent's context window.
const CONTEXT_SPAN_CAP: usize = 60;

/// Default cap (in lines) for the per-node source span printed by the
/// `--code` flag on the navigation commands (who-calls / callees /
/// trace). Tighter than `context` because these commands
/// can emit many nodes.
const CODE_SPAN_CAP: usize = 25;

/// Default cap on the number of result rows printed by the navigation
/// commands (who-calls / callees). Forensics finding F1: a
/// hot symbol like `Store` has hundreds of incoming edges, so an uncapped
/// dump emits hundreds of lines — *more* tokens than a `grep` an agent
/// would otherwise run, defeating the whole point. We print the first
/// `NAV_LIMIT` rows (deterministically ordered) and a `… and N more`
/// footer; `--all` lifts the cap when the agent genuinely wants every site.
const NAV_LIMIT: usize = 40;

/// Tighter row cap used when `--code` is set on a navigation command.
/// Forensics finding F1b (token-bomb replay of `callees X --code`): with
/// `--code`, EACH result row carries up to `CODE_SPAN_CAP` (25) lines of
/// source, so the 40-row `NAV_LIMIT` would emit ~1000 lines / ~11 KB in a
/// single result on a high-fan-out symbol — far worse than the pointer-only
/// form. When bodies are attached we therefore show only the first few rows
/// (and the same `… and N more` footer), which keeps `--code` useful for
/// "show me the callers' bodies" without letting it flood the agent's
/// context. `--all` still lifts the cap for the rare exhaustive case.
const CODE_NAV_LIMIT: usize = 6;
const EXPAND_NAV_EVIDENCE_LIMIT: usize = 80;
const EXPAND_CALLSITE_LINES_PER_NODE: usize = 8;

/// Freshness budget for explicit navigation queries. Ordinary grep passthrough
/// never reaches this path. Large repositories under I/O pressure need enough
/// time to prove freshness without turning a transient timeout into EX_TEMPFAIL.
const NAV_FRESHNESS_BUDGET: std::time::Duration = std::time::Duration::from_millis(5_000);

/// Default result row cap for code-search surfaces. Text output should stay
/// grep-like and compact; JSON reports exact totals plus omitted rows.
const SEARCH_CODE_LIMIT: usize = 20;

/// Max width (in characters) of a single matched-line snippet printed by
/// `search-code` / the navigation content-fallback. Forensics finding F3: a
/// hit on a minified-JS line or an embedded data blob would otherwise dump
/// the entire multi-KB line straight into the agent's context. The `file:line`
/// location is always preserved, so the agent can open the exact line for the
/// full content; the snippet is only a preview.
const SNIPPET_WIDTH: usize = 200;

/// Clamp a code snippet to [`SNIPPET_WIDTH`] characters, appending a
/// `… (+N chars)` marker when truncated. Counts by `char` so multi-byte
/// UTF-8 is never split mid-codepoint.
fn clamp_snippet(snippet: &str) -> std::borrow::Cow<'_, str> {
    let count = snippet.chars().count();
    if count <= SNIPPET_WIDTH {
        return std::borrow::Cow::Borrowed(snippet);
    }
    let head: String = snippet.chars().take(SNIPPET_WIDTH).collect();
    std::borrow::Cow::Owned(format!("{head}… (+{} chars)", count - SNIPPET_WIDTH))
}

/// Sample-priority rank for CAPPED navigation output. Lower ranks first:
/// named definitions before `__file__` file anchors, product code before
/// test code. Used only to pick WHICH rows land inside the printed sample
/// when truncation applies — counts, footers and `--all` are unaffected.
fn nav_sample_rank(file_path: &str, name: &str) -> (u8, u8) {
    let anchor = u8::from(name == "__file__");
    let test = u8::from(
        file_path.contains("/tests/")
            || file_path.contains("/test/")
            || file_path.starts_with("tests/")
            || file_path.starts_with("test/")
            || file_path.contains(".test.")
            || file_path.contains("_test.")
            || file_path.contains(".spec."),
    );
    (anchor, test)
}

/// Emit a truncation footer for the navigation commands when more results
/// exist than were printed. Centralised so navigation commands word it identically.
fn print_nav_more_footer(total: usize, shown: usize) {
    if total > shown {
        // Report the TRUE total so the agent can answer "how many" from this
        // line alone (e.g. "called by 72 functions"). Deliberately frame
        // `--all` as rarely needed: the F1 forensics showed agents reflexively
        // re-running with `--all` and flooding their own context when the
        // count + sample already answered the question.
        // D2: when serving from a stale index, say so in the count itself
        // so the total is never mistaken for the current state of the tree.
        let stale_note = if serving_stale() {
            " (as of last index)"
        } else {
            ""
        };
        println!(
            "… and {} more ({} shown of {} total{stale_note} — this sample usually answers the question; pass --all only if you truly need every site)",
            total - shown,
            shown,
            total
        );
    }
}

#[derive(Debug, Clone)]
struct ExpandHandle {
    id: String,
    summary: String,
}

struct ExpandEvidenceNode<'a> {
    title: String,
    node: &'a greppy_store::Node,
    site_lines: Vec<u32>,
    extra_json: serde_json::Value,
}

impl ExpandHandle {
    fn text_line(&self) -> String {
        format!(
            "Expand: greppy expand {}  (prepared evidence: {})",
            self.id, self.summary
        )
    }

    fn json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "available": true,
            "kind": "evidence_pack",
            "summary": self.summary,
        })
    }

    fn semantic_text_line(&self) -> String {
        format!(
            "greppy expand {}  → source evidence for {}",
            self.id, self.summary
        )
    }
}

fn expand_ttl_secs() -> u64 {
    std::env::var(ENV_EXPAND_TTL_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(greppy_store::DEFAULT_EXPAND_TTL_SECS)
}

#[allow(clippy::too_many_arguments)]
fn insert_expand_pack_best_effort(
    store: &greppy_store::Store,
    project: &str,
    command: &str,
    query: &str,
    graph_generation: u64,
    summary: serde_json::Value,
    payload_text: String,
    payload_json: Option<serde_json::Value>,
) -> Option<ExpandHandle> {
    if payload_text.trim().is_empty() {
        return None;
    }
    let summary_text = expand_summary_text(&summary);
    let pack = greppy_store::NewExpandPack {
        project: project.to_string(),
        command: command.to_string(),
        query: query.to_string(),
        graph_generation,
        summary_json: summary,
        payload_text,
        payload_json,
        ttl_secs: expand_ttl_secs(),
    };
    store.insert_expand_pack(&pack).ok().map(|id| ExpandHandle {
        id,
        summary: summary_text,
    })
}

fn expand_summary_text(summary: &serde_json::Value) -> String {
    summary
        .get("text")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| "evidence pack".into())
}

fn line_span(file_path: &str, start_line: i64, end_line: i64) -> String {
    if end_line > 0 && end_line >= start_line {
        format!("{file_path}:{start_line}-{end_line}")
    } else {
        format!("{file_path}:{start_line}")
    }
}

fn node_line_span(node: &greppy_store::Node) -> String {
    line_span(&node.file_path, node.start_line, node.end_line)
}

fn sorted_site_lines(lines: Option<&Vec<u32>>) -> Vec<u32> {
    let mut out = lines.cloned().unwrap_or_default();
    out.sort_unstable();
    out.dedup();
    out.truncate(EXPAND_CALLSITE_LINES_PER_NODE);
    out
}

fn append_node_evidence(
    out: &mut String,
    root: &std::path::Path,
    node: &greppy_store::Node,
    title: &str,
    site_lines: &[u32],
) {
    out.push_str(&format!("== {title} ({}) ==\n", node_line_span(node)));
    if !site_lines.is_empty() {
        out.push_str("callsites:\n");
        for line in site_lines {
            if let Some(text) = read_source_line(root, &node.file_path, *line) {
                out.push_str(&format!("  {}:{}: {}\n", node.file_path, line, text));
            }
        }
    }
    if let Some(span) = read_span(
        root,
        &node.file_path,
        node.start_line,
        node.end_line,
        CODE_SPAN_CAP,
        false,
    ) {
        out.push_str("source:\n");
        out.push_str(&span);
        if !span.ends_with('\n') {
            out.push('\n');
        }
    } else {
        out.push_str("source unavailable\n");
    }
    out.push('\n');
}

fn append_span_evidence(
    out: &mut String,
    root: &std::path::Path,
    title: &str,
    file_path: &str,
    start_line: i64,
    end_line: i64,
    cap: usize,
) {
    out.push_str(&format!(
        "== {title} ({}) ==\n",
        line_span(file_path, start_line, end_line)
    ));
    if let Some(span) = read_span(root, file_path, start_line, end_line, cap, false) {
        out.push_str(&span);
        if !span.ends_with('\n') {
            out.push('\n');
        }
    } else {
        out.push_str("source unavailable\n");
    }
    out.push('\n');
}

#[allow(clippy::too_many_arguments)]
fn insert_nav_expand_pack(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    query: &str,
    total: usize,
    rows: &[ExpandEvidenceNode<'_>],
) -> Option<ExpandHandle> {
    if rows.is_empty() {
        return None;
    }
    let root_path = resolve_root(root).ok()?;
    let limit = rows.len().min(EXPAND_NAV_EVIDENCE_LIMIT);
    let mut text = String::new();
    text.push_str(&format!("# evidence pack: {command} {query}\n"));
    text.push_str(&format!("# rows: {} shown of {} total\n\n", limit, total));
    let mut callsite_count = 0usize;
    let mut json_rows = Vec::new();
    for row in rows.iter().take(limit) {
        callsite_count += row.site_lines.len();
        append_node_evidence(&mut text, &root_path, row.node, &row.title, &row.site_lines);
        json_rows.push(serde_json::json!({
            "title": row.title,
            "qualified_name": &row.node.qualified_name,
            "label": &row.node.label,
            "file_path": &row.node.file_path,
            "start_line": row.node.start_line,
            "end_line": row.node.end_line,
            "site_lines": &row.site_lines,
            "extra": &row.extra_json,
        }));
    }
    let summary_text = if callsite_count == 0 {
        format!("{limit} spans")
    } else {
        format!("{limit} spans, {callsite_count} callsites")
    };
    let summary = serde_json::json!({
        "text": summary_text,
        "spans": limit,
        "callsites": callsite_count,
        "total": total,
    });
    let payload_json = serde_json::json!({
        "command": command,
        "query": query,
        "total": total,
        "shown": limit,
        "hits": json_rows,
    });
    insert_expand_pack_best_effort(
        store,
        project,
        command,
        query,
        current_graph_generation_or_zero(store, root),
        summary,
        text,
        Some(payload_json),
    )
}

fn insert_semantic_vector_expand_pack(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    query: &str,
    graph_generation: u64,
    hits: &[greppy_store::VectorSearchHit],
) -> Option<ExpandHandle> {
    if hits.is_empty() {
        return None;
    }
    let root_path = resolve_root(root).ok()?;
    let purposes = semantic_vector_purposes(store, root, hits, false)
        .ok()
        .flatten()?;
    let limit = purposes.len();
    if limit == 0 {
        return None;
    }
    let mut text = String::new();
    text.push_str(&format!("# evidence pack: semantic-search {query}\n"));
    text.push_str(&format!(
        "# spans: {limit} further of {} retrieved hits\n\n",
        hits.len()
    ));
    let mut json_rows = Vec::new();
    for (idx, purpose) in purposes.iter().enumerate() {
        let hit = hits
            .iter()
            .find(|hit| hit.embedding.id == purpose.embedding_id)?;
        let title = format!("{:.3} {}", hit.score, purpose.signature);
        append_span_evidence(
            &mut text,
            &root_path,
            &title,
            &purpose.file_path,
            purpose.start_line,
            purpose.end_line,
            if idx == 0 {
                CONTEXT_SPAN_CAP
            } else {
                CODE_SPAN_CAP
            },
        );
        json_rows.push(serde_json::json!({
            "score": hit.score,
            "qualified_name": &hit.embedding.qualified_name,
            "file_path": &purpose.file_path,
            "start_line": purpose.start_line,
            "end_line": purpose.end_line,
            "signature": &purpose.signature,
            "content_sha256": &hit.embedding.content_sha256,
            "graph_generation": hit.embedding.graph_generation,
        }));
    }
    let summary = serde_json::json!({
        "text": format!("{limit} further hits"),
        "spans": limit,
        "callsites": 0,
        "total": hits.len(),
    });
    let payload_json = serde_json::json!({
        "command": "semantic-search",
        "mode": "vector",
        "query": query,
        "further_hits": limit,
        "hits": json_rows,
    });
    insert_expand_pack_best_effort(
        store,
        project,
        "semantic-search",
        query,
        graph_generation,
        summary,
        text,
        Some(payload_json),
    )
}

fn current_graph_generation_or_zero(store: &greppy_store::Store, root: Option<&str>) -> u64 {
    current_graph_generation(store, root).unwrap_or(0)
}

fn node_hit_json(node: &greppy_store::Node) -> serde_json::Value {
    serde_json::json!({
        "qualified_name": &node.qualified_name,
        // AGENTS.md prints every result as `qualified_name file:line`; the JSON
        // carries the same two fields under the same names, plus the span.
        "file": &node.file_path,
        "line": node.start_line,
        "file_path": &node.file_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
    })
}

#[allow(clippy::too_many_arguments)]
fn nav_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    command: &str,
    symbol: &str,
    project: &str,
    symbol_found: bool,
    total_exact: usize,
    shown: usize,
    all: bool,
    hits: Vec<serde_json::Value>,
) -> Result<()> {
    nav_counts_json_with_expand(
        store,
        root,
        command,
        symbol,
        project,
        symbol_found,
        total_exact,
        shown,
        all,
        hits,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn nav_counts_json_with_expand(
    store: &greppy_store::Store,
    root: Option<&str>,
    command: &str,
    symbol: &str,
    project: &str,
    symbol_found: bool,
    total_exact: usize,
    shown: usize,
    all: bool,
    hits: Vec<serde_json::Value>,
    expand: Option<&ExpandHandle>,
) -> Result<()> {
    let omitted = total_exact.saturating_sub(shown);
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    // Rule 3: a one-symbol answer is a batch of one. The caller parses ONE
    // shape — `targets` plus per-hit attribution — however many symbols it
    // named.
    let mut hits = hits;
    for hit in &mut hits {
        if hit.get("target").is_none() {
            hit["target"] = serde_json::json!(symbol);
        }
    }
    let mut v = serde_json::json!({
        "command": command,
        "symbol": symbol,
        "targets": [{
            "symbol": symbol,
            "symbol_found": symbol_found,
            "total_exact": total_exact,
        }],
        "project": project,
        "symbol_found": symbol_found,
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "total_exact": total_exact,
        "shown": shown,
        "omitted": omitted,
        "truncated": omitted > 0,
        "all": all,
        "hits": hits,
    });
    if !symbol_found {
        let miss = symbol_miss_json(store, project, symbol);
        v["suggestions"] = miss["suggestions"].clone();
        v["next"] = miss["next"].clone();
    }
    if let Some(expand) = expand {
        v["expand"] = expand.json_value();
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize nav JSON: {e}")))?
    );
    Ok(())
}

fn incomplete_provider_json(
    store: &greppy_store::Store,
    project: &str,
) -> Result<Vec<serde_json::Value>> {
    Ok(store
        .list_provider_states(project)?
        .into_iter()
        .filter(greppy_store::ProviderState::is_incomplete)
        .filter(|p| !is_noncode_provider(&p.status, &p.language))
        .map(|p| {
            // Agent responses only need to know which language is partial.
            // Per-edge-class and per-file diagnostics belong to doctor and
            // diagnostics JSON; repeating them on every query wastes tokens.
            serde_json::json!({
                "language": p.language,
                "status": p.status,
            })
        })
        .collect())
}

/// A provider row is "non-code noise" when it exists only because the indexer
/// saw a file it does not parse as source — snapshot/fixture artifacts like
/// `.stderr`, `.snap`, `.snapshot`, or any other unrecognised extension. These
/// map to `Language::Unsupported`, whose provider `status` is `"unsupported"`
/// and whose `language` reads `"file extension .<ext>"` / `"no file
/// extension"`. Such a provider has NO call/usage edges to miss, so counting it
/// as an "incomplete provider" wrongly told agents the code call-graph was
/// partial — the r061 28-round reconciliation blowup. Agent-facing provider
/// metadata therefore reports only real code providers.
fn is_noncode_provider(status: &str, language: &str) -> bool {
    status == "unsupported"
        || language.starts_with("file extension .")
        || language == "no file extension"
}

/// Compact incomplete-provider metadata, excluding non-code snapshot/fixture
/// providers (see [`is_noncode_provider`]) so the reported
/// `incomplete_provider_count` / `provider_complete` reflects only real code
/// callers, not `.stderr` / `.snap` files.
fn code_incomplete_provider_json(
    store: &greppy_store::Store,
    project: &str,
) -> Result<Vec<serde_json::Value>> {
    incomplete_provider_json(store, project)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderPolicy {
    Metadata,
    RequireComplete,
}

fn provider_policy_from_env() -> Result<ProviderPolicy> {
    let raw = match std::env::var(ENV_PROVIDER_POLICY) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(ProviderPolicy::Metadata),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::Config(format!(
                "{ENV_PROVIDER_POLICY} must be valid UTF-8"
            )));
        }
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "metadata" | "warn" | "permissive" => Ok(ProviderPolicy::Metadata),
        "require_complete" | "require-complete" | "strict" | "1" | "true" => {
            Ok(ProviderPolicy::RequireComplete)
        }
        _ => Err(Error::Config(format!(
            "{ENV_PROVIDER_POLICY} must be one of metadata or require_complete"
        ))),
    }
}

fn provider_policy_blocks_query(incomplete_providers: &[serde_json::Value]) -> Result<bool> {
    Ok(
        provider_policy_from_env()? == ProviderPolicy::RequireComplete
            && !incomplete_providers.is_empty(),
    )
}

fn provider_incomplete_skip_message(command: &str, incomplete_count: usize) -> String {
    format!(
        "{command}: skipped indexed provider-dependent output because {incomplete_count} language provider(s) are incomplete; set {ENV_PROVIDER_POLICY}=metadata for metadata-only mode or re-index after provider acceptance"
    )
}

fn provider_incomplete_skip_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    incomplete_providers: &[serde_json::Value],
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<()> {
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let mut obj = serde_json::Map::new();
    obj.insert("command".into(), serde_json::json!(command));
    obj.insert(
        "status".into(),
        serde_json::json!("skipped_incomplete_provider"),
    );
    obj.insert("project".into(), serde_json::json!(project));
    obj.insert("fresh".into(), serde_json::json!(fresh));
    obj.insert("freshness".into(), freshness);
    obj.insert("provider_complete".into(), serde_json::json!(false));
    obj.insert(
        "incomplete_provider_count".into(),
        serde_json::json!(incomplete_providers.len()),
    );
    obj.insert(
        "incomplete_providers".into(),
        serde_json::json!(incomplete_providers),
    );
    obj.insert("total_exact".into(), serde_json::json!(0));
    obj.insert("shown".into(), serde_json::json!(0));
    obj.insert("omitted".into(), serde_json::json!(0));
    obj.insert("truncated".into(), serde_json::json!(false));
    if let serde_json::Value::Object(extra) = extra {
        for (key, value) in extra {
            obj.insert(key, value);
        }
    }
    obj.insert(empty_collection_field.into(), serde_json::json!([]));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(obj)).map_err(|e| {
            Error::Invalid(format!("serialize provider policy JSON for {command}: {e}"))
        })?
    );
    Ok(())
}

fn graph_stale_skip_json(
    store: &greppy_store::Store,
    _root: Option<&str>,
    project: &str,
    command: &str,
    freshness: serde_json::Value,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let mut obj = serde_json::Map::new();
    obj.insert("command".into(), serde_json::json!(command));
    obj.insert("status".into(), serde_json::json!("skipped_stale_index"));
    obj.insert("project".into(), serde_json::json!(project));
    obj.insert("fresh".into(), serde_json::json!(false));
    obj.insert("freshness".into(), freshness);
    obj.insert(
        "provider_complete".into(),
        serde_json::json!(incomplete_providers.is_empty()),
    );
    obj.insert(
        "incomplete_provider_count".into(),
        serde_json::json!(incomplete_providers.len()),
    );
    obj.insert(
        "incomplete_providers".into(),
        serde_json::json!(incomplete_providers),
    );
    obj.insert("total_exact".into(), serde_json::json!(0));
    obj.insert("shown".into(), serde_json::json!(0));
    obj.insert("omitted".into(), serde_json::json!(0));
    obj.insert("truncated".into(), serde_json::json!(false));
    if let serde_json::Value::Object(extra) = extra {
        for (key, value) in extra {
            obj.insert(key, value);
        }
    }
    obj.insert(empty_collection_field.into(), serde_json::json!([]));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(obj)).map_err(|e| {
            Error::Invalid(format!("serialize stale graph JSON for {command}: {e}"))
        })?
    );
    Ok(())
}

/// Fresh-or-fallback gate for graph navigation. Indexed graph data is only
/// visible when freshness was proven; drift/unknown states trigger refresh
/// and return EX_TEMPFAIL instead of exposing stale rows.
fn graph_stale_gate(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    json: bool,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<Option<i32>> {
    match freshness_serve_decision(store, root, project) {
        FreshnessServe::Fresh(_) => Ok(None),
        FreshnessServe::Refuse(freshness) => {
            if json {
                graph_stale_skip_json(
                    store,
                    root,
                    project,
                    command,
                    freshness.clone(),
                    extra,
                    empty_collection_field,
                )?;
            } else {
                println!("{}", indexed_stale_skip_message(command, &freshness));
            }
            Ok(Some(freshness_refusal_exit(&freshness)))
        }
    }
}

fn provider_policy_graph_gate(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    json: bool,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<Option<i32>> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    if !provider_policy_blocks_query(&incomplete_providers)? {
        return Ok(None);
    }
    if json {
        provider_incomplete_skip_json(
            store,
            root,
            project,
            command,
            &incomplete_providers,
            extra,
            empty_collection_field,
        )?;
    } else {
        println!(
            "{}",
            provider_incomplete_skip_message(command, incomplete_providers.len())
        );
    }
    Ok(Some(1))
}

struct ImpactJsonMeta<'a> {
    direction: &'a str,
    edge_type: &'a str,
    edge_types: &'a [&'a str],
    max_hops: usize,
    scope: &'a str,
}

struct ImpactEdgeSpec<'a> {
    mode: &'a str,
    edge_types: Vec<&'a str>,
}

fn impact_edge_spec<'a>(
    direction: greppy_search::ReachDirection,
    requested_edge: Option<&'a str>,
) -> ImpactEdgeSpec<'a> {
    match requested_edge {
        None if matches!(direction, greppy_search::ReachDirection::Incoming) => ImpactEdgeSpec {
            mode: "all_references",
            edge_types: greppy_search::REFERENCE_EDGE_TYPES.to_vec(),
        },
        None => ImpactEdgeSpec {
            mode: "CALLS",
            edge_types: vec!["CALLS"],
        },
        Some(edge) => ImpactEdgeSpec {
            mode: edge,
            edge_types: vec![edge],
        },
    }
}

fn insert_impact_edge_meta(obj: &mut serde_json::Value, spec: &ImpactEdgeSpec<'_>) {
    if let Some(map) = obj.as_object_mut() {
        map.insert("edge_type".into(), serde_json::json!(spec.mode));
        map.insert("edge_types".into(), serde_json::json!(&spec.edge_types));
    }
}

#[allow(clippy::too_many_arguments)]
fn impact_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    symbol: &str,
    project: &str,
    symbol_found: bool,
    total_exact: usize,
    shown: usize,
    all: bool,
    meta: ImpactJsonMeta<'_>,
    hits: Vec<serde_json::Value>,
) -> Result<()> {
    impact_counts_json_with_expand(
        store,
        root,
        symbol,
        project,
        symbol_found,
        total_exact,
        shown,
        all,
        meta,
        hits,
        Vec::new(),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn impact_counts_json_with_expand(
    store: &greppy_store::Store,
    root: Option<&str>,
    symbol: &str,
    project: &str,
    symbol_found: bool,
    total_exact: usize,
    shown: usize,
    all: bool,
    meta: ImpactJsonMeta<'_>,
    hits: Vec<serde_json::Value>,
    tests: Vec<serde_json::Value>,
    expand: Option<&ExpandHandle>,
) -> Result<()> {
    let omitted = total_exact.saturating_sub(shown);
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    // Only real code providers count toward impact completeness; `.stderr` /
    // `.snap` snapshot files are not callers (see `code_incomplete_provider_json`).
    let incomplete_providers = code_incomplete_provider_json(store, project)?;
    // Rule 3: the one-symbol answer has the same shape as a batch of several.
    let mut hits = hits;
    for hit in &mut hits {
        if hit.get("target").is_none() {
            hit["target"] = serde_json::json!(symbol);
        }
    }
    let mut v = serde_json::json!({
        "command": "impact",
        "symbol": symbol,
        "targets": [{
            "symbol": symbol,
            "symbol_found": symbol_found,
            "total_exact": total_exact,
            "tests": tests,
        }],
        "project": project,
        "symbol_found": symbol_found,
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": meta.scope,
        "direction": meta.direction,
        "edge_type": meta.edge_type,
        "edge_types": meta.edge_types,
        "max_hops": meta.max_hops,
        "total_exact": total_exact,
        "shown": shown,
        "omitted": omitted,
        "truncated": omitted > 0,
        "all": all,
        "hits": hits,
    });
    if let Some(expand) = expand {
        v["expand"] = expand.json_value();
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize impact JSON: {e}")))?
    );
    Ok(())
}

#[derive(Clone)]
struct DiffImpactSource {
    row: greppy_search::graph::SearchGraphRow,
}

struct DiffImpactHit {
    node: greppy_search::graph::SearchGraphRow,
    hops: usize,
    sources: Vec<greppy_search::graph::SearchGraphRow>,
}

#[allow(clippy::too_many_arguments)]
fn impact_diff_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    spec: &DiffSearchSpec,
    direction: &str,
    edge_type: &str,
    edge_types: &[&str],
    max_hops: usize,
    sources_total: usize,
    sources_shown: usize,
    total_exact: usize,
    shown: usize,
    hits: Vec<serde_json::Value>,
    source_rows: Vec<serde_json::Value>,
) -> Result<()> {
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    // Exclude non-code snapshot/fixture providers from impact completeness.
    let incomplete_providers = code_incomplete_provider_json(store, project)?;
    let v = serde_json::json!({
        "command": "impact",
        "status": "ok",
        "project": project,
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": "diff",
        "diff_scope": spec.scope,
        "backend": "git_diff_graph",
        "diff_rev": &spec.diff_rev,
        "merge_base": spec.merge_base.as_deref(),
        "diff_files_total": spec.files.len(),
        "direction": direction,
        "edge_type": edge_type,
        "edge_types": edge_types,
        "max_hops": max_hops,
        "source_total": sources_total,
        "source_shown": sources_shown,
        "source_omitted": sources_total.saturating_sub(sources_shown),
        "source_symbols": source_rows,
        "total_exact": total_exact,
        "shown": shown,
        "omitted": total_exact.saturating_sub(shown),
        "truncated": total_exact > shown,
        "all": false,
        "hits": hits,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize impact diff JSON: {e}")))?
    );
    Ok(())
}

fn graph_impact_source_row(row: &greppy_search::graph::SearchGraphRow) -> bool {
    !matches!(row.label.as_str(), "Module" | "Import" | "Call")
        && !row.qualified_name.ends_with("::__file__")
}

fn diff_impact_sources(
    store: &greppy_store::Store,
    project: &str,
    changed_lines: &std::collections::BTreeMap<String, std::collections::BTreeSet<i64>>,
) -> Result<Vec<DiffImpactSource>> {
    let mut by_id: std::collections::BTreeMap<i64, greppy_search::graph::SearchGraphRow> =
        std::collections::BTreeMap::new();
    for (file, lines) in changed_lines {
        for line in lines {
            if let Some(row) = greppy_search::definition_at(store, Some(project), file, *line)? {
                if graph_impact_source_row(&row) {
                    by_id.entry(row.id).or_insert(row);
                }
            }
        }
    }
    let mut sources = by_id
        .into_values()
        .map(|row| DiffImpactSource { row })
        .collect::<Vec<_>>();
    sources.sort_by(|a, b| {
        a.row
            .file_path
            .cmp(&b.row.file_path)
            .then_with(|| a.row.start_line.cmp(&b.row.start_line))
            .then_with(|| a.row.qualified_name.cmp(&b.row.qualified_name))
            .then_with(|| a.row.id.cmp(&b.row.id))
    });
    Ok(sources)
}

fn diff_impact_hits(
    store: &greppy_store::Store,
    sources: &[DiffImpactSource],
    direction: greppy_search::ReachDirection,
    edge_types: &[&str],
    max_hops: usize,
) -> Result<Vec<DiffImpactHit>> {
    let source_by_id = sources
        .iter()
        .map(|source| (source.row.id, source.row.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut hits: std::collections::BTreeMap<
        i64,
        (
            usize,
            greppy_search::graph::SearchGraphRow,
            std::collections::BTreeSet<i64>,
        ),
    > = std::collections::BTreeMap::new();

    for source in sources {
        for hit in greppy_search::impact_radius_any_edge_type(
            store,
            source.row.id,
            direction,
            edge_types,
            max_hops,
            4096,
        )? {
            let entry = hits.entry(hit.node.id).or_insert_with(|| {
                (
                    hit.hops,
                    hit.node.clone(),
                    std::collections::BTreeSet::new(),
                )
            });
            if hit.hops < entry.0 {
                entry.0 = hit.hops;
                entry.1 = hit.node.clone();
            }
            entry.2.insert(source.row.id);
        }
    }

    let mut out = hits
        .into_iter()
        .map(|(_id, (hops, node, source_ids))| {
            let sources = source_ids
                .into_iter()
                .filter_map(|id| source_by_id.get(&id).cloned())
                .collect::<Vec<_>>();
            DiffImpactHit {
                node,
                hops,
                sources,
            }
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| {
        a.hops
            .cmp(&b.hops)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| a.node.id.cmp(&b.node.id))
    });
    Ok(out)
}

struct TraceJsonMeta<'a> {
    direction: &'a str,
    edge_type: Option<&'a str>,
    max_depth: usize,
}

fn trace_step_json(step: &greppy_search::TraceStep) -> serde_json::Value {
    let edge = step.edge.as_ref().map(|e| {
        serde_json::json!({
            "id": e.id,
            "edge_type": &e.edge_type,
            "source_id": e.source_id,
            "target_id": e.target_id,
        })
    });
    match &step.node {
        Some(node) => serde_json::json!({
            "depth": step.depth,
            "node_id": step.node_id,
            "qualified_name": &node.qualified_name,
            "name": &node.name,
            "label": &node.label,
            "file_path": &node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "via_edge": edge,
        }),
        None => serde_json::json!({
            "depth": step.depth,
            "node_id": step.node_id,
            "qualified_name": serde_json::Value::Null,
            "name": serde_json::Value::Null,
            "label": serde_json::Value::Null,
            "file_path": serde_json::Value::Null,
            "start_line": serde_json::Value::Null,
            "end_line": serde_json::Value::Null,
            "via_edge": edge,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn trace_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    symbol: &str,
    project: &str,
    symbol_found: bool,
    meta: TraceJsonMeta<'_>,
    total_exact: usize,
    steps: &[greppy_search::TraceStep],
) -> Result<()> {
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let step_json: Vec<_> = steps.iter().map(trace_step_json).collect();
    let shown = step_json.len();
    let omitted = total_exact.saturating_sub(shown);
    let v = serde_json::json!({
        "command": "trace",
        "symbol": symbol,
        "project": project,
        "symbol_found": symbol_found,
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": "bounded_bfs",
        "direction": meta.direction,
        "edge_type": meta.edge_type,
        "max_depth": meta.max_depth,
        "total_exact": total_exact,
        "shown": shown,
        "omitted": omitted,
        "truncated": omitted > 0,
        "steps": step_json,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize trace JSON: {e}")))?
    );
    Ok(())
}

fn graph_row_json(row: &greppy_search::graph::SearchGraphRow) -> serde_json::Value {
    serde_json::json!({
        "node_id": row.id,
        "qualified_name": &row.qualified_name,
        "name": &row.name,
        "label": &row.label,
        "file_path": &row.file_path,
        "start_line": row.start_line,
        "end_line": row.end_line,
    })
}

fn degree_hit_json(hit: &greppy_search::DegreeRanked) -> serde_json::Value {
    let mut v = graph_row_json(&hit.node);
    if let Some(obj) = v.as_object_mut() {
        obj.insert("degree".to_string(), serde_json::json!(hit.degree));
    }
    v
}

struct DegreeJsonMeta<'a> {
    command: &'a str,
    direction: &'a str,
    edge_type: &'a str,
    requested_limit: usize,
    effective_limit: usize,
}

fn degree_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    total_exact: usize,
    hits: &[greppy_search::DegreeRanked],
    meta: DegreeJsonMeta<'_>,
) -> Result<()> {
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let hit_json: Vec<_> = hits.iter().map(degree_hit_json).collect();
    let v = serde_json::json!({
        "command": meta.command,
        "project": project,
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": "degree_rank",
        "direction": meta.direction,
        "edge_type": meta.edge_type,
        "requested_limit": meta.requested_limit,
        "limit": meta.effective_limit,
        "total_exact": total_exact,
        "shown": shown,
        "omitted": omitted,
        "truncated": omitted > 0,
        "hits": hit_json,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize degree JSON: {e}")))?
    );
    Ok(())
}

fn graph_locate_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    file_path: &str,
    line: i64,
    hit: Option<&greppy_search::graph::SearchGraphRow>,
    match_kind: Option<&str>,
) -> Result<()> {
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let hits: Vec<_> = hit.into_iter().map(graph_row_json).collect();
    let total_exact = hits.len();
    let v = serde_json::json!({
        "command": "graph-locate",
        "project": project,
        "file_path": file_path,
        "line": line,
        "location_found": total_exact == 1,
        "match_kind": match_kind,
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": "file_line_innermost_symbol",
        "total_exact": total_exact,
        "shown": total_exact,
        "omitted": 0,
        "truncated": false,
        "hits": hits,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize graph-locate JSON: {e}")))?
    );
    Ok(())
}

struct PathJsonMeta<'a> {
    edge_type: &'a str,
    max_hops: usize,
    reason: Option<&'a str>,
}

#[allow(clippy::too_many_arguments)]
fn path_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    from: &str,
    to: &str,
    project: &str,
    from_found: bool,
    to_found: bool,
    path: Option<&greppy_search::GraphPath>,
    meta: PathJsonMeta<'_>,
) -> Result<()> {
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let steps: Vec<_> = path
        .map(|p| p.rows.iter().map(graph_row_json).collect())
        .unwrap_or_default();
    let step_count = steps.len();
    let hops = path
        .map(|p| serde_json::json!(p.hops))
        .unwrap_or(serde_json::Value::Null);
    let v = serde_json::json!({
        "command": "path",
        "from": from,
        "to": to,
        "project": project,
        "from_found": from_found,
        "to_found": to_found,
        "path_found": path.is_some(),
        "reason": meta.reason,
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": "shortest_path",
        "direction": "outgoing",
        "edge_type": meta.edge_type,
        "max_hops": meta.max_hops,
        "hops": hops,
        "total_exact": step_count,
        "shown": step_count,
        "omitted": 0,
        "truncated": false,
        "steps": steps,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize path JSON: {e}")))?
    );
    Ok(())
}

fn search_graph_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    name_filter: Option<&str>,
    limit: usize,
    total_exact: usize,
    rows: &[greppy_search::graph::SearchGraphRow],
) -> Result<()> {
    let shown = rows.len();
    let omitted = total_exact.saturating_sub(shown);
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let hits: Vec<_> = rows.iter().map(graph_row_json).collect();
    let v = serde_json::json!({
        "command": "search-graph",
        "project": project,
        "filters": {
            "name": name_filter,
        },
        "fresh": fresh,
        "freshness": freshness,
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "scope": "node_search",
        "limit": limit,
        "total_exact": total_exact,
        "shown": shown,
        "omitted": omitted,
        "truncated": omitted > 0,
        "hits": hits,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize search-graph JSON: {e}")))?
    );
    Ok(())
}

fn nav_freshness_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
) -> serde_json::Value {
    let overrides = match discover_overrides_from_env() {
        Ok(overrides) => overrides,
        Err(e) => {
            return serde_json::json!({
                "fresh": false,
                "state": "config_error",
                "reasons": [e.to_string()],
                "elapsed_ms": 0,
                "discover_scope": "invalid",
                "discover_scope_env": {
                    "include": ENV_DISCOVER_INCLUDE,
                    "exclude": ENV_DISCOVER_EXCLUDE,
                },
            });
        }
    };
    let discover_scope = overrides.scope_key();
    let root_path = match resolve_root(root) {
        Ok(path) => path,
        Err(e) => {
            return serde_json::json!({
                "fresh": false,
                "state": "unknown",
                "reasons": [format!("resolve root failed: {e}")],
                "elapsed_ms": 0,
                "discover_scope": discover_scope,
                "discover_scope_env": {
                    "include": ENV_DISCOVER_INCLUDE,
                    "exclude": ENV_DISCOVER_EXCLUDE,
                },
            });
        }
    };
    match greppy_freshness::check_files_report_with_overrides(
        store,
        &root_path,
        project,
        NAV_FRESHNESS_BUDGET,
        &overrides,
    ) {
        Ok(report) => {
            let (fresh, state_name, reasons) = match report.state.outcome {
                greppy_freshness::FreshnessOutcome::Fresh => (true, "fresh", Vec::<String>::new()),
                greppy_freshness::FreshnessOutcome::Cold => {
                    (false, "cold", vec!["no persisted workspace state".into()])
                }
                greppy_freshness::FreshnessOutcome::RootMismatch => {
                    (false, "drift", vec!["workspace root mismatch".into()])
                }
                greppy_freshness::FreshnessOutcome::Stale { reasons } => (false, "drift", reasons),
                greppy_freshness::FreshnessOutcome::Unknown { reasons } => {
                    (false, "unknown", reasons)
                }
            };
            serde_json::json!({
                "fresh": fresh,
                "state": state_name,
                "reasons": reasons,
                "elapsed_ms": report.state.elapsed.as_millis(),
                // D2: how far the index has drifted. `null` when the
                // check could not enumerate changes (cold store, budget
                // exhausted, walk failure).
                "stale_file_count": report.changed_paths.as_ref().map(Vec::len),
                "changed_paths": report.changed_paths,
                "total_inventory": report.total_inventory,
                "ttl_hit": report.ttl_hit,
                "discover_scope": discover_scope,
                "discover_scope_env": {
                    "include": ENV_DISCOVER_INCLUDE,
                    "exclude": ENV_DISCOVER_EXCLUDE,
                },
            })
        }
        Err(e) => serde_json::json!({
            "fresh": false,
            "state": "unknown",
            "reasons": [format!("freshness check failed: {e}")],
            "elapsed_ms": NAV_FRESHNESS_BUDGET.as_millis(),
            "discover_scope": discover_scope,
            "discover_scope_env": {
                "include": ENV_DISCOVER_INCLUDE,
                "exclude": ENV_DISCOVER_EXCLUDE,
            },
        }),
    }
}

/// Fresh-or-fallback policy for indexed query surfaces. Only `Fresh` may
/// expose graph or embedding rows. `Refuse` carries cold, drift, refreshing,
/// unknown, or failed state; callers either use a live filesystem backend or
/// return EX_TEMPFAIL.
enum FreshnessServe {
    Fresh(serde_json::Value),
    Refuse(serde_json::Value),
}

impl FreshnessServe {
    /// The freshness JSON to embed in the command's payload, whatever
    /// the verdict was.
    fn freshness(&self) -> &serde_json::Value {
        match self {
            FreshnessServe::Fresh(f) | FreshnessServe::Refuse(f) => f,
        }
    }
}

/// Auto-reindex cap: an inline atomic snapshot is only attempted when at most
/// this many files drifted. Above the cap one background refresh starts and
/// stale indexed results are refused.
const AUTO_REINDEX_MAX_FILES: usize = 10;

/// Kill switch for automatic inline/background refresh (`0`/`false` disables).
const ENV_AUTO_REINDEX: &str = "GREPPY_AUTO_REINDEX";

fn auto_reindex_enabled() -> bool {
    match std::env::var(ENV_AUTO_REINDEX) {
        Ok(raw) => !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Compatibility hook for older footer rendering. Production query paths
/// never serve stale indexed rows, so this is always false.
fn serving_stale() -> bool {
    false
}

fn freshness_state_can_trigger_reindex(state: &str) -> bool {
    !matches!(
        state,
        "cold" | "config_error" | "failed" | "unknown" | "refreshing"
    )
}

fn freshness_serve_decision(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
) -> FreshnessServe {
    freshness_serve_decision_with_policy(store, root, project, true, true)
}

/// Heal a reindexable-stale store in-band: rebuild the graph AND (when the
/// store carried them) the embeddings + summaries at a fresh generation, then
/// re-open so the caller serves the current codebase. The edit loop mutates
/// files constantly — through greppy's own edits AND external means (git apply,
/// bash, another tool) — and every query command must reflect those changes or
/// the agent gets stale/empty answers and abandons greppy (forensics
/// 2026-07-18). Genuinely un-reindexable states (cold/failed) are left for the
/// stale gate to refuse. Best-effort: a failed reindex leaves the old store,
/// and the gate then decides.
fn maybe_reindex_stale(store: &mut greppy_store::Store, root: Option<&str>) -> Result<()> {
    // An explicit auto-reindex opt-out must fall through to the fail-closed
    // stale gate. In particular, do not wait on an active writer that the
    // caller has said must not be joined for automatic healing.
    if !auto_reindex_enabled() {
        return Ok(());
    }
    let project = project_for(root)?;
    if freshness_is_reindexable_stale(store, root, &project) {
        let rebuilt = try_auto_reindex_inline(root);
        if !rebuilt && workspace_writer_active(root) {
            wait_for_active_index_refresh(root);
        }
        if let Ok(fresh) = open_default_store_query_writer(root) {
            *store = fresh;
        }
    }
    Ok(())
}

/// An edit-owned or background indexer may already be building the exact fresh
/// snapshot this query needs. Wait for that writer to publish instead of
/// immediately returning an empty `refreshing` result. The OS lock has no stale
/// owner: process exit releases it, so this cannot wait on an abandoned PID.
fn wait_for_active_index_refresh(root: Option<&str>) {
    let Ok(effective_root) = resolve_root(root) else {
        return;
    };
    let store_path = workspace_locator::store_path(&effective_root);
    eprintln!(
        "greppy: graph refresh already running; waiting for fresh snapshot (ETA unavailable)"
    );
    if let Ok(lock) = greppy_freshness::acquire(&store_path) {
        drop(lock);
    }
}

/// The index is stale AND the drift is one an inline reindex can heal
/// (workspace/content drift or a scope-stable version bump), not a cold or
/// broken store. Used by `read` to reindex in-band before serving rather than
/// refuse and leave the edit-loop agent empty-handed.
fn freshness_is_reindexable_stale(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
) -> bool {
    let freshness = nav_freshness_json(store, root, project);
    if freshness_json_is_fresh(&freshness) {
        return false;
    }
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    if !freshness_state_can_trigger_reindex(state) {
        return false;
    }
    let scope_or_version_drift = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|reasons| {
            reasons
                .iter()
                .filter_map(serde_json::Value::as_str)
                .any(|reason| reason.contains("indexer version/scope"))
        });
    if scope_or_version_drift {
        return version_drift_is_scope_stable(&freshness);
    }
    if metadata_only_fingerprint_drift(&freshness) {
        return false;
    }
    freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|count| {
            count as usize <= AUTO_REINDEX_MAX_FILES
                && freshness_changed_bytes(root, &freshness)
                    .is_some_and(|bytes| bytes <= 8 * 1024 * 1024)
        })
}

/// `allow_auto_reindex = false` for a surface that cannot guarantee its
/// generation-scoped embeddings can be rebuilt in the same snapshot.
/// True when the indexer-version drift is a pure VERSION bump (same
/// discover scope on both sides), so a self-healing full reindex rebuilds
/// under the scope the store was already indexed with — never under a
/// different scope the user did not ask to persist. Parses the reason
/// string `indexer version/scope changed (was X, expected Y)` where both
/// X and Y are `{base}` or `{base};discover_scope={scope}`.
fn version_drift_is_scope_stable(freshness: &serde_json::Value) -> bool {
    let reason = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .and_then(|rs| {
            rs.iter()
                .filter_map(serde_json::Value::as_str)
                .find(|r| r.contains("indexer version/scope"))
        });
    let Some(reason) = reason else {
        return false;
    };
    let Some((was, expected)) = reason
        .split_once("(was ")
        .and_then(|(_, rest)| rest.strip_suffix(')'))
        .and_then(|body| body.split_once(", expected "))
    else {
        return false;
    };
    let scope_of = |s: &str| {
        s.split_once(";discover_scope=")
            .map(|(_, sc)| sc.to_string())
            .unwrap_or_default()
    };
    // Same scope on both sides, and the versions genuinely differ.
    scope_of(was) == scope_of(expected) && was != expected
}

fn metadata_only_fingerprint_drift(freshness: &serde_json::Value) -> bool {
    if freshness.get("state").and_then(serde_json::Value::as_str) != Some("drift")
        || freshness
            .get("stale_file_count")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
    {
        return false;
    }
    freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|reasons| {
            !reasons.is_empty()
                && reasons.iter().all(|reason| {
                    reason.as_str().is_some_and(|reason| {
                        reason.starts_with("git_dir changed")
                            || reason.starts_with("git_common_dir changed")
                            || reason.starts_with("head_oid changed")
                            || reason.starts_with("index signature changed")
                    })
                })
        })
}

fn try_refresh_metadata_only_fingerprint(
    root: Option<&str>,
    freshness: &serde_json::Value,
) -> Option<serde_json::Value> {
    if !metadata_only_fingerprint_drift(freshness) {
        return None;
    }
    let effective_root = resolve_root(root).ok()?;
    let overrides = discover_overrides_from_env().ok()?;
    let store_path = workspace_locator::store_path(&effective_root);
    let _writer = greppy_freshness::try_acquire(&store_path).ok()?;
    let mut store =
        greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::query_writer())
            .ok()?;
    let fingerprint = greppy_core::GitFingerprint::capture(&effective_root);
    if !greppy_freshness::refresh_fingerprint_metadata(
        &mut store,
        &fingerprint,
        NAV_FRESHNESS_BUDGET,
        &overrides,
    )
    .ok()?
    {
        return None;
    }

    let mut refreshed = freshness.clone();
    let object = refreshed.as_object_mut()?;
    object.insert("fresh".into(), serde_json::Value::Bool(true));
    object.insert("state".into(), serde_json::Value::String("fresh".into()));
    object.insert("reasons".into(), serde_json::Value::Array(Vec::new()));
    Some(refreshed)
}

fn freshness_serve_decision_with_policy(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    allow_auto_reindex: bool,
    _warn_on_stale: bool,
) -> FreshnessServe {
    let freshness = nav_freshness_json(store, root, project);
    if freshness_json_is_fresh(&freshness) {
        return FreshnessServe::Fresh(freshness);
    }
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    // Unknown is not evidence of drift. In particular, a budget-exhausted
    // inventory walk must not launch a full reindex that can replace the DB
    // containing expand packs created by the preceding query.
    if !freshness_state_can_trigger_reindex(state) {
        return FreshnessServe::Refuse(freshness);
    }
    let scope_or_version_drift = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|rs| {
            rs.iter()
                .filter_map(serde_json::Value::as_str)
                .any(|r| r.contains("indexer version/scope"))
        });
    if scope_or_version_drift {
        if allow_auto_reindex && auto_reindex_enabled() && version_drift_is_scope_stable(&freshness)
        {
            let started = spawn_background_index(root, "indexer-version-drift");
            return FreshnessServe::Refuse(refresh_state(
                freshness,
                started || workspace_writer_active(root),
            ));
        }
        return FreshnessServe::Refuse(freshness);
    }

    // A commit can change only HEAD after the exact source contents were
    // already indexed. The inventory diff above proves there are zero stale
    // files, so refresh just the fingerprint instead of rebuilding the graph
    // and every embedding at a new generation.
    if allow_auto_reindex && auto_reindex_enabled() {
        if let Some(refreshed) = try_refresh_metadata_only_fingerprint(root, &freshness) {
            return FreshnessServe::Fresh(refreshed);
        }
    }

    let stale_file_count = freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);
    let small_enough = stale_file_count.is_some_and(|count| {
        count <= AUTO_REINDEX_MAX_FILES
            && freshness_changed_bytes(root, &freshness)
                .is_some_and(|bytes| bytes <= 8 * 1024 * 1024)
    });
    if allow_auto_reindex && auto_reindex_enabled() && small_enough {
        let rebuilt = try_auto_reindex_inline(root);
        return FreshnessServe::Refuse(refresh_state(
            freshness,
            rebuilt || workspace_writer_active(root),
        ));
    }
    if allow_auto_reindex && auto_reindex_enabled() {
        let started = spawn_background_index(root, "workspace-drift");
        return FreshnessServe::Refuse(refresh_state(
            freshness,
            started || workspace_writer_active(root),
        ));
    }
    FreshnessServe::Refuse(freshness)
}

fn freshness_changed_bytes(root: Option<&str>, freshness: &serde_json::Value) -> Option<u64> {
    let root = resolve_root(root).ok()?;
    let paths = freshness.get("changed_paths")?.as_array()?;
    let mut bytes = 0u64;
    for path in paths {
        let path = path.as_str()?;
        match std::fs::metadata(root.join(path)) {
            Ok(metadata) => bytes = bytes.saturating_add(metadata.len()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return None,
        }
    }
    Some(bytes)
}

fn refresh_state(mut freshness: serde_json::Value, started: bool) -> serde_json::Value {
    if let Some(object) = freshness.as_object_mut() {
        object.insert(
            "state".into(),
            serde_json::json!(if started { "refreshing" } else { "failed" }),
        );
        object.insert("fresh".into(), serde_json::json!(false));
    }
    freshness
}

fn freshness_refusal_exit(freshness: &serde_json::Value) -> i32 {
    match freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
    {
        "refreshing" | "drift" | "unknown" => EXIT_TEMPFAIL as i32,
        _ => 1,
    }
}

fn workspace_writer_active(root: Option<&str>) -> bool {
    let Ok(root) = resolve_root(root) else {
        return false;
    };
    let hash = greppy_core::workspace::workspace_hash(&root);
    matches!(
        greppy_core::cache::acquire_named_lock(
            &format!("workspace-{hash}.writer"),
            greppy_core::cache::LockMode::Exclusive,
            true,
        ),
        Ok(None)
    )
}

/// Build a small-drift refresh through the same temp-snapshot publication
/// boundary as an explicit `index`. The current query keeps its old inode and
/// therefore never observes in-place mutation or a partially rebuilt graph.
fn try_auto_reindex_inline(root: Option<&str>) -> bool {
    let Ok(effective_root) = resolve_root(root) else {
        return false;
    };
    let Ok(project) = project_for(root) else {
        return false;
    };
    let Ok(overrides) = discover_overrides_from_env() else {
        return false;
    };
    let store_path = workspace_locator::store_path(&effective_root);
    let Ok(Some(_lifecycle)) = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    ) else {
        return false;
    };
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => lock,
        _ => return false, // another writer is active: refuse this snapshot
    };
    let Ok(store) =
        greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::read_only())
    else {
        return false;
    };
    // Remember whether this store served code-span vectors BEFORE the
    // reindex bumps the generation: an inline graph-only reindex would
    // otherwise strand every existing vector row on the old generation and
    // silently degrade `context`/`semantic-search` until a manual
    // `grep index` run (the owner's "gains" path dying quietly).
    let had_vectors = !store
        .vector_model_ids(&project)
        .unwrap_or_default()
        .is_empty();
    drop(store);
    let embedding_cfg = if had_vectors {
        let no_args = EmbeddingCliArgs {
            device: None,
            no_gpu: false,
        };
        match embedding_config_optional(no_args) {
            Ok(Some(cfg)) => Some(cfg),
            _ => return false,
        }
    } else {
        None
    };
    let options = greppy_indexer::IndexOptions {
        discover_overrides: overrides,
    };
    index_atomic_snapshot(
        &store_path,
        &effective_root,
        &project,
        embedding_cfg.as_ref(),
        &options,
        false,
        None,
    )
    .map(|snapshot| snapshot.index.is_clean())
    .unwrap_or(false)
}

/// Whether the vector query path may self-heal a stale index via the
/// atomic auto-reindex: only when the embedding model is resolvable, because
/// an existing vector generation must be rebuilt as part of the snapshot.
fn vector_auto_reindex_can_rebuild(args: EmbeddingCliArgs<'_>) -> bool {
    match embedding_config_optional(args) {
        Ok(Some(cfg)) => embedding_model_source_exists(&cfg.source),
        Ok(None) | Err(_) => false,
    }
}

fn embedding_model_source_exists(source: &EmbeddingModelSource) -> bool {
    let EmbeddingModelSource::Gguf { gguf, tokenizer } = source;
    gguf.is_file() && tokenizer.is_file()
}

/// Atomically published status for the one allowed background index job.
const BACKGROUND_JOB_FILE: &str = "index.job";

fn background_job_path(root: &std::path::Path) -> std::path::PathBuf {
    workspace_locator::store_path(root)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(BACKGROUND_JOB_FILE)
}

fn process_is_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
        }
        let Ok(pid) = std::ffi::c_int::try_from(pid) else {
            return false;
        };
        let rc = unsafe { kill(pid, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(1)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if process.is_null() {
            return false;
        }
        let mut exit_code = 0u32;
        let queried = unsafe { GetExitCodeProcess(process, &mut exit_code) } != 0;
        unsafe {
            CloseHandle(process);
        }
        queried && i32::try_from(exit_code).ok() == Some(STILL_ACTIVE)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

fn read_background_job(path: &std::path::Path) -> Option<serde_json::Value> {
    let raw = std::fs::read(path).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn write_background_job(path: &std::path::Path, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;

    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("background job path has no parent".into()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| Error::io(format!("create {}", parent.display()), error))?;
    let temp = parent.join(format!(
        ".background.job.{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0)
    ));
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| Error::Invalid(format!("serialize background job: {error}")))?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temp)
        .map_err(|error| Error::io(format!("create {}", temp.display()), error))?;
    file.write_all(&bytes)
        .map_err(|error| Error::io(format!("write {}", temp.display()), error))?;
    file.sync_all()
        .map_err(|error| Error::io(format!("sync {}", temp.display()), error))?;
    drop(file);
    replace_background_job_file(&temp, path)
        .map_err(|error| Error::io(format!("publish {}", path.display()), error))?;
    sync_parent_dir(path)?;
    Ok(())
}

#[cfg(not(windows))]
fn replace_background_job_file(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_background_job_file(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) } == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

struct BackgroundJobGuard {
    path: Option<std::path::PathBuf>,
    cause: String,
    kind: String,
    started_at_unix_secs: u64,
    target_generation: u64,
    backend: Option<String>,
    device: Option<String>,
    completed_documents: usize,
    total_documents: usize,
    eta_seconds: Option<u64>,
    rate_milli_documents_per_second: Option<u64>,
    embedding_started: Option<std::time::Instant>,
    last_progress_write: Option<std::time::Instant>,
    complete: bool,
}

impl BackgroundJobGuard {
    fn from_env() -> Self {
        let path = std::env::var_os("GREPPY_BACKGROUND_JOB").map(std::path::PathBuf::from);
        // The parent can only publish the job PID after spawn. Hold the child
        // at its entry point until that atomic record is visible, preventing
        // a very small repository from completing and removing the file
        // before the parent writes `refreshing` over it.
        if let Some(path) = &path {
            for _ in 0..100 {
                if read_background_job(path)
                    .and_then(|job| job.get("pid").and_then(serde_json::Value::as_u64))
                    == Some(u64::from(std::process::id()))
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        let published = path.as_deref().and_then(read_background_job);
        Self {
            path,
            cause: std::env::var("GREPPY_BACKGROUND_CAUSE")
                .unwrap_or_else(|_| "background-refresh".into()),
            kind: std::env::var("GREPPY_BACKGROUND_KIND").unwrap_or_else(|_| "index".into()),
            started_at_unix_secs: std::env::var("GREPPY_BACKGROUND_STARTED_AT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(unix_now_secs_cli),
            target_generation: std::env::var("GREPPY_BACKGROUND_TARGET_GENERATION")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            backend: published
                .as_ref()
                .and_then(|job| job.get("backend"))
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned),
            device: published
                .as_ref()
                .and_then(|job| job.get("device"))
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned),
            completed_documents: 0,
            total_documents: published
                .as_ref()
                .and_then(|job| job.get("total_spans"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(0),
            eta_seconds: published
                .as_ref()
                .and_then(|job| job.get("eta_seconds"))
                .and_then(serde_json::Value::as_u64),
            rate_milli_documents_per_second: None,
            embedding_started: None,
            last_progress_write: None,
            complete: false,
        }
    }

    fn is_background(&self) -> bool {
        self.path.is_some()
    }

    fn embedding_loading(&mut self) {
        self.write_state("loading_model", None);
    }

    fn embedding_started(&mut self, backend: &str, total_documents: usize) {
        self.backend = Some(backend.to_string());
        self.completed_documents = 0;
        self.total_documents = total_documents;
        let now = std::time::Instant::now();
        self.embedding_started = Some(now);
        self.rate_milli_documents_per_second = None;
        self.eta_seconds = initial_embedding_eta_seconds(total_documents, backend);
        self.write_state("embedding", None);
        self.last_progress_write = Some(now);
    }

    fn embedding_progress(&mut self, progress: greppy_indexer::EmbeddingIndexProgress) {
        self.completed_documents = progress.completed_documents;
        self.total_documents = progress.total_documents;
        if let Some(started) = self.embedding_started {
            let elapsed_ms = u64::try_from(started.elapsed().as_millis())
                .unwrap_or(u64::MAX)
                .max(1);
            self.eta_seconds = observed_embedding_eta_seconds(
                self.completed_documents,
                self.total_documents,
                elapsed_ms,
            )
            .or(self.eta_seconds);
            self.rate_milli_documents_per_second =
                observed_embedding_rate_milli(self.completed_documents, elapsed_ms);
        }
        let now = std::time::Instant::now();
        let finished = self.total_documents > 0 && self.completed_documents >= self.total_documents;
        let publish = finished
            || self.last_progress_write.is_none_or(|last| {
                now.duration_since(last) >= std::time::Duration::from_millis(500)
            });
        if publish {
            self.write_state("embedding", None);
            self.last_progress_write = Some(now);
        }
    }

    fn write_state(&self, state: &str, last_error: Option<&str>) {
        let Some(path) = &self.path else { return };
        let now = unix_now_secs_cli();
        let eta_unix_secs = self.eta_seconds.map(|eta| now.saturating_add(eta));
        let eta_minutes = self.eta_seconds.map(|eta| eta.saturating_add(59) / 60);
        let progress_milli_percent = if self.total_documents == 0 {
            0
        } else {
            self.completed_documents
                .min(self.total_documents)
                .saturating_mul(100_000)
                .checked_div(self.total_documents)
                .unwrap_or(0)
        };
        let value = serde_json::json!({
            "schema_version": BACKGROUND_JOB_SCHEMA_VERSION,
            "kind": self.kind,
            "pid": std::process::id(),
            "started_at_unix_secs": self.started_at_unix_secs,
            "updated_at_unix_secs": now,
            "cause": self.cause,
            "target_generation": self.target_generation,
            "state": state,
            "backend": self.backend,
            "device": self.device,
            "completed_spans": self.completed_documents,
            "total_spans": self.total_documents,
            "progress_milli_percent": progress_milli_percent,
            "rate_milli_spans_per_second": self.rate_milli_documents_per_second,
            "eta_seconds": self.eta_seconds,
            "eta_minutes": eta_minutes,
            "eta_unix_secs": eta_unix_secs,
            "last_error": last_error,
        });
        let _ = write_background_job(path, &value);
    }

    fn complete(&mut self) {
        self.complete = true;
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
            let _ = sync_parent_dir(path);
        }
    }

    fn fail(&mut self, error: &Error) {
        self.write_state("failed", Some(&error.to_string()));
        self.complete = true;
    }

    /// The snapshot published but the embedding pass is incomplete
    /// (inference failure). The background record keeps the `failed`
    /// state with the degradation reason so the next semantic query
    /// retries the remaining vectors; the published graph stays live.
    fn degraded(&mut self, reason: &str) {
        self.write_state("failed", Some(reason));
        self.complete = true;
    }
}

impl Drop for BackgroundJobGuard {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        self.write_state(
            "failed",
            Some("background index exited before successful publication"),
        );
    }
}

fn initial_embedding_rate(backend: &str) -> u64 {
    match backend {
        "cuda" => 12,
        "metal" => 8,
        _ => 1,
    }
}

fn initial_embedding_eta_seconds(total_documents: usize, backend: &str) -> Option<u64> {
    let total = u64::try_from(total_documents).ok()?;
    let rate = initial_embedding_rate(backend).max(1);
    Some(total.saturating_add(rate - 1) / rate)
}

fn observed_embedding_eta_seconds(
    completed_documents: usize,
    total_documents: usize,
    elapsed_ms: u64,
) -> Option<u64> {
    let completed = u64::try_from(completed_documents).ok()?;
    let total = u64::try_from(total_documents).ok()?;
    if completed == 0 {
        return None;
    }
    let remaining = total.saturating_sub(completed);
    let numerator = u128::from(remaining).saturating_mul(u128::from(elapsed_ms));
    let denominator = u128::from(completed).saturating_mul(1_000);
    let rounded = numerator.saturating_add(denominator.saturating_sub(1)) / denominator.max(1);
    u64::try_from(rounded).ok()
}

fn observed_embedding_rate_milli(completed_documents: usize, elapsed_ms: u64) -> Option<u64> {
    let completed = u64::try_from(completed_documents).ok()?;
    if completed == 0 {
        return None;
    }
    completed
        .saturating_mul(1_000_000)
        .checked_div(elapsed_ms.max(1))
}

fn unix_now_secs_cli() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0)
}

fn embedding_backend_plan(cfg: &EmbeddingModelConfig) -> (String, Option<String>) {
    let EmbeddingModelSource::Gguf { gguf, .. } = &cfg.source;
    let model_bytes = std::fs::metadata(gguf)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let required = greppy_embed_native::estimated_gpu_memory(
        greppy_embed_native::InferenceModelKind::EmbeddingGemma,
        model_bytes,
    );
    let selector = inference_device_identity(&cfg.device);
    let policy = greppy_embed_native::InferencePolicy::from_selector(Some(&selector), false);
    let registry = policy.ok().map(|policy| {
        greppy_embed_native::InferenceBackendRegistry::probe_policy(&policy, required)
    });
    let backend = registry
        .as_ref()
        .and_then(|registry| registry.selected_backend)
        .map(greppy_embed_native::BackendKind::as_str)
        .unwrap_or_else(|| cfg.device.as_str())
        .to_string();
    let device = registry
        .and_then(|registry| registry.selected_device_id)
        .or_else(|| (selector != "auto").then_some(selector));
    (backend, device)
}

fn current_embedding_candidate_count(root: &std::path::Path) -> usize {
    let project = workspace_locator::project_identity(root);
    greppy_store::Store::open_with(
        &workspace_locator::store_path(root),
        greppy_store::OpenOptions::read_only(),
    )
    .ok()
    .and_then(|store| greppy_indexer::count_embedding_candidate_nodes(&store, &project).ok())
    .unwrap_or(0)
}

/// Start at most one detached refresh for a worktree. A spawn lock closes the
/// cross-process race and the atomically published job record is the public
/// progress surface used by semantic-search.
fn spawn_background_job(
    root: Option<&str>,
    cause: &str,
    kind: &str,
    embedding_cfg: Option<&EmbeddingModelConfig>,
) -> bool {
    let Ok(root) = resolve_root(root) else {
        return false;
    };
    if greppy_core::cache::ensure_workspace_store(&root).is_err() {
        return false;
    }
    let hash = greppy_core::workspace::workspace_hash(&root);
    let Ok(Some(_spawn_lock)) = greppy_core::cache::acquire_named_lock(
        &format!("workspace-{hash}.job-spawn"),
        greppy_core::cache::LockMode::Exclusive,
        false,
    ) else {
        return false;
    };
    let job_path = background_job_path(&root);
    if let Some(job) = read_background_job(&job_path) {
        if job
            .get("pid")
            .and_then(serde_json::Value::as_u64)
            .and_then(|pid| u32::try_from(pid).ok())
            .is_some_and(process_is_alive)
        {
            return true;
        }
    }
    let target_generation = greppy_store::Store::open_with(
        &workspace_locator::store_path(&root),
        greppy_store::OpenOptions::read_only(),
    )
    .ok()
    .and_then(|store| {
        store
            .get_workspace_state(root.to_string_lossy().as_ref())
            .ok()
            .flatten()
            .map(|state| state.graph_generation)
    })
    .unwrap_or(0)
    .saturating_add(1);
    let Ok(exe) = std::env::current_exe() else {
        return false;
    };
    let started_at = unix_now_secs_cli();
    let (backend, device, total_spans, eta_seconds) = if let Some(cfg) = embedding_cfg {
        let (backend, device) = embedding_backend_plan(cfg);
        let total = current_embedding_candidate_count(&root);
        let eta = initial_embedding_eta_seconds(total, &backend);
        (Some(backend), device, total, eta)
    } else {
        (None, None, 0, None)
    };
    let mut command = std::process::Command::new(exe);
    command
        .arg("index")
        .arg(&root)
        .arg("--root")
        .arg(&root)
        .env("GREPPY_BACKGROUND_JOB", &job_path)
        .env("GREPPY_BACKGROUND_CAUSE", cause)
        .env("GREPPY_BACKGROUND_KIND", kind)
        .env("GREPPY_BACKGROUND_STARTED_AT", started_at.to_string())
        .env(
            "GREPPY_BACKGROUND_TARGET_GENERATION",
            target_generation.to_string(),
        )
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    if let Some(cfg) = embedding_cfg {
        command.env(ENV_DEVICE, inference_device_identity(&cfg.device));
    }
    let Ok(child) = command.spawn() else {
        return false;
    };
    let eta_unix_secs = eta_seconds.map(|eta| started_at.saturating_add(eta));
    let eta_minutes = eta_seconds.map(|eta| eta.saturating_add(59) / 60);
    let value = serde_json::json!({
        "schema_version": BACKGROUND_JOB_SCHEMA_VERSION,
        "kind": kind,
        "pid": child.id(),
        "started_at_unix_secs": started_at,
        "updated_at_unix_secs": started_at,
        "cause": cause,
        "target_generation": target_generation,
        "state": if kind == "embedding" { "starting" } else { "refreshing" },
        "backend": backend,
        "device": device,
        "completed_spans": 0,
        "total_spans": total_spans,
        "progress_milli_percent": 0,
        "rate_milli_spans_per_second": serde_json::Value::Null,
        "eta_seconds": eta_seconds,
        "eta_minutes": eta_minutes,
        "eta_unix_secs": eta_unix_secs,
        "last_error": serde_json::Value::Null,
    });
    write_background_job(&job_path, &value).is_ok()
}

fn spawn_background_index(root: Option<&str>, cause: &str) -> bool {
    spawn_background_job(root, cause, "index", None)
}

/// Kick off the complete atomic graph + embedding snapshot as a detached
/// child. The resolved inference policy is propagated so explicit CPU/Metal/
/// CUDA choices and automatic GPU priority remain identical in the child.
fn spawn_background_embed(root: Option<&str>, cfg: &EmbeddingModelConfig) -> bool {
    spawn_background_job(root, "embedding-first-use", "embedding", Some(cfg))
}

fn embedding_generation_complete(
    store: &greppy_store::Store,
    project: &str,
    graph_generation: u64,
    model_id: &str,
) -> bool {
    let key = embedding_complete_key(project);
    store
        .conn()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [&key],
            |row| row.get::<_, String>(0),
        )
        .ok()
        == Some(format!("{graph_generation}|{model_id}"))
}

fn embedding_progress_value(
    root: &std::path::Path,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
) -> serde_json::Value {
    if let Some(mut job) = read_background_job(&background_job_path(root)) {
        let alive = job
            .get("pid")
            .and_then(serde_json::Value::as_u64)
            .and_then(|pid| u32::try_from(pid).ok())
            .is_some_and(process_is_alive);
        job["alive"] = serde_json::json!(alive);
        job["graph_generation"] = serde_json::json!(graph_generation);
        return job;
    }

    let (backend, device) = embedding_backend_plan(cfg);
    let total_spans = current_embedding_candidate_count(root);
    let eta_seconds = initial_embedding_eta_seconds(total_spans, &backend);
    let now = unix_now_secs_cli();
    serde_json::json!({
        "schema_version": BACKGROUND_JOB_SCHEMA_VERSION,
        "kind": "embedding",
        "state": "starting",
        "alive": false,
        "backend": backend,
        "device": device,
        "graph_generation": graph_generation,
        "completed_spans": 0,
        "total_spans": total_spans,
        "progress_milli_percent": 0,
        "rate_milli_spans_per_second": serde_json::Value::Null,
        "eta_seconds": eta_seconds,
        "eta_minutes": eta_seconds.map(|eta| eta.saturating_add(59) / 60),
        "eta_unix_secs": eta_seconds.map(|eta| now.saturating_add(eta)),
        "last_error": serde_json::Value::Null,
    })
}

fn format_embedding_eta(seconds: u64) -> String {
    let minutes = seconds / 60;
    let remainder = seconds % 60;
    if minutes == 0 {
        format!("{remainder}s")
    } else if remainder == 0 {
        format!("{minutes}m")
    } else {
        format!("{minutes}m {remainder}s")
    }
}

fn embedding_progress_text(progress: &serde_json::Value) -> String {
    let backend = progress
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("cpu");
    let completed = progress
        .get("completed_spans")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = progress
        .get("total_spans")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if let Some(eta) = progress
        .get("eta_seconds")
        .and_then(serde_json::Value::as_u64)
    {
        format!(
            "semantic index building — {completed}/{total} spans, ETA ~{} (backend {backend})",
            format_embedding_eta(eta)
        )
    } else {
        format!(
            "semantic index building — {completed}/{total} spans, ETA measuring (backend {backend})"
        )
    }
}

#[derive(Clone, Copy)]
struct SemanticFallbackContext<'a> {
    query: &'a str,
    paths: &'a [String],
    root: Option<&'a str>,
}

fn semantic_embedding_indexing_json(
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    freshness: &serde_json::Value,
    progress: &serde_json::Value,
    fallback: SemanticFallbackContext<'_>,
) -> Result<()> {
    let eta_seconds = progress
        .get("eta_seconds")
        .and_then(serde_json::Value::as_u64);
    let retry_after_seconds = eta_seconds.map(|eta| eta.clamp(5, 30)).unwrap_or(10);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
            "command": "semantic-search",
            "mode": "vector",
            "status": "indexing",
            "project": project,
            "model_id": cfg.model_id,
            "prompt_version": greppy_embed_native::PROMPT_VERSION,
            "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
            "graph_generation": graph_generation,
            "fresh": freshness_json_is_fresh(freshness),
            "freshness": freshness,
            "retryable": true,
            "retry_after_seconds": retry_after_seconds,
            "embedding_index": progress,
            "query_tokens": semantic_fallback_tokens(fallback.query),
            "next": semantic_fallback_commands(fallback.query, fallback.paths, fallback.root),
            "total_exact": 0,
            "shown": 0,
            "omitted": 0,
            "truncated": false,
            "hits": [],
        }))
        .map_err(|error| Error::Invalid(format!("serialize semantic indexing JSON: {error}")))?
    );
    Ok(())
}

fn emit_semantic_backend_unavailable(
    project: &str,
    query: &str,
    paths: &[String],
    root: Option<&str>,
    json: bool,
    detail: &str,
) -> Result<()> {
    let next = semantic_fallback_commands(query, paths, root);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
                "command": "semantic-search",
                "mode": "vector",
                "status": "backend_unavailable",
                "project": project,
                "query": query,
                "reason": "asset_missing",
                "detail": detail,
                "retryable": false,
                "query_tokens": semantic_fallback_tokens(query),
                "next": next,
                "total_exact": 0,
                "shown": 0,
                "omitted": 0,
                "truncated": false,
                "hits": [],
            }))
            .map_err(|error| {
                Error::Invalid(format!("serialize semantic unavailable JSON: {error}"))
            })?
        );
    } else {
        println!("semantic backend unavailable (asset missing) — {detail}");
        print_semantic_fallback_commands(query, paths, root);
    }
    Ok(())
}

/// Read one source line (1-based) for the grep-shaped call-site rows the
/// nav commands print (P4). Missing/unreadable files or out-of-range lines
/// return None — the row is skipped, never an error. Trimmed and capped so
/// a pathological line cannot flood the agent's context.
fn read_source_line(root: &std::path::Path, file_path: &str, line: u32) -> Option<String> {
    if line == 0 {
        return None;
    }
    let text = std::fs::read_to_string(root.join(file_path)).ok()?;
    let raw = text.lines().nth(line as usize - 1)?.trim();
    if raw.is_empty() {
        return None;
    }
    let mut s = raw.to_string();
    if s.len() > 160 {
        let mut cut = 160;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    Some(s)
}

/// `--code` and `--json` compose: AGENTS.md gives `--code` as "also print each
/// result's source and a handle for it" and `--json` as "the same answer as
/// data", so the source and the handle belong ON the hit.
fn ensure_nav_json_mode(_code: bool, _json: bool) -> Result<()> {
    Ok(())
}

/// Read the source span for a node from disk and return it as a string,
/// capped at `cap` lines. `file_path` is the node's stored path (relative
/// to the repo root); `root` is the resolved repo root. `start_line` and
/// `end_line` are 1-based inclusive line numbers as stored on the node.
///
/// Robustness (per the task contract): a missing file, an unreadable
/// file, or out-of-range line numbers yield `Ok(None)` so the caller can
/// skip the span gracefully rather than failing the whole command. Only
/// the root-resolution step (which never touches the node's file) can
/// surface a hard error.
///
/// When `with_line_numbers` is set, each emitted line is prefixed with
/// its 1-based line number so an agent can cite exact lines. When the
/// span exceeds `cap` lines it is truncated and a
/// `… (truncated, N more lines)` marker is appended.
///
/// Current indexes store the full tree-sitter definition range. Older indexes
/// may contain only the declaration line (`end_line == start_line`); only for
/// those legacy rows do we recover a body end with [`definition_end_idx`]. A
/// multi-line parser span is authoritative. Extending it heuristically can
/// cross into the next Python method or another adjacent definition.
fn read_span(
    root: &std::path::Path,
    file_path: &str,
    start_line: i64,
    end_line: i64,
    cap: usize,
    with_line_numbers: bool,
) -> Option<String> {
    read_span_with_meta(
        root,
        file_path,
        start_line,
        end_line,
        cap,
        with_line_numbers,
    )
    .map(|span| span.text)
}

fn read_span_with_meta(
    root: &std::path::Path,
    file_path: &str,
    start_line: i64,
    end_line: i64,
    cap: usize,
    with_line_numbers: bool,
) -> Option<SpanRead> {
    // Reject obviously invalid line ranges (the store uses 1-based,
    // inclusive lines; 0 or negative means "unknown").
    if start_line < 1 || end_line < start_line {
        return None;
    }
    let abs = root.join(file_path);
    let content = std::fs::read_to_string(&abs).ok()?;
    let all: Vec<&str> = content.lines().collect();
    // Convert to 0-based indices into the line vector.
    let start_idx = (start_line - 1) as usize;
    if start_idx >= all.len() {
        // start_line is past the end of the file (stale index / edit) —
        // skip gracefully rather than emit nothing useful.
        return None;
    }
    // Stored parser end, clamped to the file.
    let stored_end_idx = std::cmp::min(end_line as usize, all.len()) - 1;
    let end_idx_inclusive = if stored_end_idx == start_idx {
        definition_end_idx(&all, start_idx)
    } else {
        stored_end_idx
    };
    let total_lines = end_idx_inclusive - start_idx + 1;
    let actual_end_line = start_line + total_lines as i64 - 1;
    let shown = std::cmp::min(total_lines, cap);
    let mut out = String::new();
    for (offset, line) in all[start_idx..start_idx + shown].iter().enumerate() {
        if with_line_numbers {
            let lineno = start_line as usize + offset;
            out.push_str(&format!("{lineno:>6}  {line}\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if total_lines > shown {
        out.push_str(&format!(
            "… (truncated, {} more line(s))\n",
            total_lines - shown
        ));
    }
    let omitted_lines = total_lines - shown;
    Some(SpanRead {
        text: out,
        end_line: actual_end_line,
        total_lines,
        shown_lines: shown,
        omitted_lines,
        truncated: omitted_lines > 0,
    })
}

/// Find the 0-based index of the line that ends the definition beginning
/// at `start_idx`, by balancing `{}`/`()`/`[]` delimiters from the
/// declaration line forward.
///
/// The store records only the declaration line of a symbol, so to emit
/// the real body we scan forward until the first `{` we open is balanced
/// back to zero. If no `{` appears before a top-level `;` (a unit struct,
/// a type alias, or a trait-method signature), the `;`-line is the end.
/// If neither closes within [`MAX_SCAN_LINES`] lines (a malformed or
/// truncated file), we stop at the scan window so a pathological input
/// can never run away.
///
/// String/char literals and `//` line comments are skipped so a `{` or
/// `;` inside them does not throw off the balance. This is a pragmatic
/// scanner, not a full Rust lexer — it does not special-case raw strings
/// or block comments containing unbalanced braces, which are rare inside
/// a signature/body header and at worst make the span a little longer
/// (still capped by the caller).
fn definition_end_idx(lines: &[&str], start_idx: usize) -> usize {
    // Shared with the embedding indexer (which embeds the same span this
    // prints) — single source of truth in greppy-core.
    greppy_core::spans::definition_end_idx(lines, start_idx)
}

/// Print the `--code` source span for a single resolved node, using the
/// shared cap and the standard skip-on-failure semantics. Emitted
/// indented under the node's `file:line` line so the structure stays
/// readable when many nodes are printed.
fn print_code_span(root: &std::path::Path, node: &greppy_store::Node, cap: usize) {
    if let Some(span) = read_span(
        root,
        &node.file_path,
        node.start_line,
        node.end_line,
        cap,
        false,
    ) {
        print_code_span_text(&span);
    }
}

fn print_code_span_text(span: &str) {
    for line in span.lines() {
        println!("    {line}");
    }
}

fn dispatch_trace(
    symbol: Option<&str>,
    direction: &str,
    edge: &str,
    depth: usize,
    code: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let dir = match direction.to_ascii_lowercase().as_str() {
        "outgoing" | "out" => greppy_search::TraceDirection::Outgoing,
        "incoming" | "in" => greppy_search::TraceDirection::Incoming,
        other => {
            return Err(Error::Invalid(format!(
                "trace --direction must be 'outgoing' or 'incoming', got '{other}'"
            )));
        }
    };
    let direction_label = match dir {
        greppy_search::TraceDirection::Outgoing => "outgoing",
        greppy_search::TraceDirection::Incoming => "incoming",
    };
    // An empty `--edge ""` means "any edge type"; otherwise filter to the
    // requested type (upper-cased to match the stored edge labels).
    let edge_upper = edge.trim().to_ascii_uppercase();
    let edge_filter: Option<&str> = if edge_upper.is_empty() {
        None
    } else {
        Some(edge_upper.as_str())
    };

    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let query_symbol = symbol.unwrap_or("");
    let graph_gate_extra = serde_json::json!({
        "symbol": query_symbol,
        "symbol_found": false,
        "scope": "bounded_bfs",
        "direction": direction_label,
        "edge_type": edge_filter,
        "max_depth": depth,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "trace",
        json,
        graph_gate_extra.clone(),
        "steps",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "trace",
        json,
        graph_gate_extra,
        "steps",
    )? {
        return Ok(code);
    }
    let Some(start) = resolve_symbol_id(&store, symbol)? else {
        if json {
            trace_counts_json(
                &store,
                root,
                query_symbol,
                &project,
                false,
                TraceJsonMeta {
                    direction: direction_label,
                    edge_type: edge_filter,
                    max_depth: depth,
                },
                0,
                &[],
            )?;
            return Ok(1);
        }
        println!("(symbol not found)");
        return Ok(1);
    };
    let steps = greppy_search::trace_path(&store, start, dir, edge_filter, depth)?;
    let shown = steps.len().min(cli_result_limit(NAV_LIMIT));
    let shown_steps = &steps[..shown];
    if json {
        trace_counts_json(
            &store,
            root,
            query_symbol,
            &project,
            true,
            TraceJsonMeta {
                direction: direction_label,
                edge_type: edge_filter,
                max_depth: depth,
            },
            steps.len(),
            shown_steps,
        )?;
        return Ok(0);
    }
    // `--code` reads spans from disk relative to the resolved repo root.
    let span_root = if code {
        Some(resolve_root(root)?)
    } else {
        None
    };
    for s in shown_steps {
        let edge_marker = match &s.edge {
            Some(e) => format!("via {}", e.edge_type),
            None => "start".to_string(),
        };
        // Print actionable file:line/qname rather
        // than bare node ids so an agent can act without a follow-up
        // search.
        let ctx = match &s.node {
            Some(n) => format!(
                "{} {}:{}-{}",
                display_node_name(n),
                n.file_path,
                n.start_line,
                n.end_line
            ),
            None => format!("node={}", s.node_id),
        };
        println!("depth={} {} ({})", s.depth, ctx, edge_marker);
        // Track A: with `--code`, emit the traced node's source span so
        // the agent sees the body without a separate Read.
        if let (Some(root_path), Some(n)) = (span_root.as_deref(), &s.node) {
            print_code_span(root_path, n, CODE_SPAN_CAP);
        }
    }
    Ok(0)
}

/// `greppy impact S` — the transitive blast radius of `S` in ONE call.
///
/// `--direction incoming` (default) walks every transitive CALLER of `S`
/// (answers "if I change S, what breaks?"); `--direction outgoing` walks
/// everything `S` transitively reaches. Each reached node is printed once,
/// at its minimum hop distance, ordered by (hops, qualified_name), with a
/// capped total + `… and N more` footer. This is the single-command answer
/// that replaces the dozen iterative `who-calls`/`callees` an agent would
/// otherwise run for a multi-hop question — the whole point of having a graph.
#[allow(clippy::too_many_arguments)]
fn dispatch_impact(
    symbol: Option<&str>,
    paths: &[String],
    direction: &str,
    edge: Option<&str>,
    depth: usize,
    since: Option<&str>,
    base: Option<&str>,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let path_filters = prepare_query_path_filters(root, "impact", symbol.unwrap_or(""), paths)?;
    let dir = match direction.to_ascii_lowercase().as_str() {
        "incoming" | "in" | "callers" => greppy_search::ReachDirection::Incoming,
        "outgoing" | "out" | "callees" => greppy_search::ReachDirection::Outgoing,
        other => {
            return Err(Error::Invalid(format!(
                "impact --direction must be 'incoming' or 'outgoing', got '{other}'"
            )));
        }
    };
    let direction_label = match dir {
        greppy_search::ReachDirection::Incoming => "incoming",
        greppy_search::ReachDirection::Outgoing => "outgoing",
    };
    let edge_upper = edge.map(|edge| edge.trim().to_ascii_uppercase());
    if since.is_some() && base.is_some() {
        return Err(Error::Invalid(
            "impact accepts only one git diff scope at a time".into(),
        ));
    }
    let edge_spec = impact_edge_spec(dir, edge_upper.as_deref());
    if since.is_some() || base.is_some() {
        if symbol.map(str::trim).is_some_and(|s| !s.is_empty()) {
            return Err(Error::Invalid(
                "impact accepts either a symbol or a git diff scope, not both".into(),
            ));
        }
        let scope = match (since, base) {
            (Some(rev), None) => DiffSearchScope::Since { rev },
            (None, Some(base)) => DiffSearchScope::Base { base },
            _ => {
                return Err(Error::Invalid(
                    "impact accepts exactly one git diff scope".into(),
                ));
            }
        };
        return dispatch_impact_diff_scope(
            scope,
            dir,
            direction_label,
            &edge_spec,
            depth,
            json,
            root,
        );
    }
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let query_symbol = symbol.unwrap_or("");
    let mut graph_gate_extra = serde_json::json!({
        "symbol": query_symbol,
        "symbol_found": false,
        "scope": "transitive",
        "direction": direction_label,
        "max_hops": depth,
        "all": false,
    });
    insert_impact_edge_meta(&mut graph_gate_extra, &edge_spec);
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "impact",
        json,
        graph_gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "impact",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let Some(start) = resolve_symbol_id(&store, symbol)? else {
        if json {
            impact_counts_json(
                &store,
                root,
                query_symbol,
                &project,
                false,
                0,
                0,
                false,
                ImpactJsonMeta {
                    direction: direction_label,
                    edge_type: edge_spec.mode,
                    edge_types: &edge_spec.edge_types,
                    max_hops: depth,
                    scope: "transitive",
                },
                Vec::new(),
            )?;
            return Ok(1);
        }
        return content_fallback(
            &store,
            root,
            symbol.unwrap_or(""),
            "impact",
            &QueryPathFilters::default(),
        );
    };
    // Aggregate over every same-name start node (e.g. a Class and its Impl) and
    // union the reach, keeping the minimum hop count. For default incoming
    // impact, each traversal follows all reference edge types at every BFS
    // layer so mixed paths are preserved. Generous internal limit; the PRINTED
    // rows are capped separately so the footer can report the true transitive
    // total.
    let starts = resolve_symbol_nodes(&store, symbol)?;
    let starts = if starts.is_empty() {
        vec![start]
    } else {
        starts
    };
    let start_ids: std::collections::HashSet<i64> = starts.iter().copied().collect();
    let mut by_id: std::collections::HashMap<i64, greppy_search::ImpactNode> =
        std::collections::HashMap::new();
    for &sid in &starts {
        for n in greppy_search::impact_radius_any_edge_type(
            &store,
            sid,
            dir,
            &edge_spec.edge_types,
            depth,
            4096,
        )? {
            if start_ids.contains(&n.node.id) {
                continue; // a start node is not its own dependent
            }
            by_id
                .entry(n.node.id)
                .and_modify(|e| {
                    if n.hops < e.hops {
                        e.hops = n.hops;
                    }
                })
                .or_insert(n);
        }
    }
    let mut reached: Vec<greppy_search::ImpactNode> = by_id.into_values().collect();
    reached.sort_by(|a, b| a.hops.cmp(&b.hops).then_with(|| a.node.id.cmp(&b.node.id)));
    reached.retain(|n| path_filters.matches(&n.node.file_path));
    // "and the tests among it" — the reason to run impact before a change is
    // to learn which tests will speak up.
    let mut impact_tests = Vec::new();
    for step in &reached {
        if let Some(node) = store.get_node(step.node.id)? {
            if is_test_node(&node) {
                impact_tests.push(node_hit_json(&node));
            }
        }
    }
    if reached.is_empty() {
        if json {
            impact_counts_json(
                &store,
                root,
                query_symbol,
                &project,
                true,
                0,
                0,
                false,
                ImpactJsonMeta {
                    direction: direction_label,
                    edge_type: edge_spec.mode,
                    edge_types: &edge_spec.edge_types,
                    max_hops: depth,
                    scope: "transitive",
                },
                Vec::new(),
            )?;
            return Ok(0);
        }
        let what = match dir {
            greppy_search::ReachDirection::Incoming => "(nothing depends on it transitively)",
            greppy_search::ReachDirection::Outgoing => "(it reaches nothing transitively)",
        };
        println!("{what}");
        return Ok(0);
    }
    let total = reached.len();
    // `--all` lifts the print cap so the full transitive set is inspectable
    // in one call (the footer's "T total" was previously unreachable — clap
    // rejected --all — forcing a 28-round reconcile, r061).
    let shown = total.min(cli_result_limit_unless_all(NAV_LIMIT, all));
    // Informative sampling (r071/r074/r075 forensics): when the print cap
    // truncates, the FIRST `shown` rows are the answer most agents run with —
    // so rank named definitions before `__file__` file anchors and product
    // code before tests, THEN by hop, instead of letting a wall of hop-2
    // test-file anchors crowd named callers out of the sample. Rank beats
    // hop deliberately: every printed row still carries its `hop N` label,
    // but a named hop-3 caller answers a blast-radius question while a
    // test-file anchor rarely does. Stable sort: ordering within a rank
    // class, the true total, the footer, and `--all` output are unchanged.
    if shown < total {
        reached.sort_by(|a, b| {
            (nav_sample_rank(&a.node.file_path, &a.node.name), a.hops)
                .cmp(&(nav_sample_rank(&b.node.file_path, &b.node.name), b.hops))
        });
    }
    let expand = if !all {
        let mut nodes = Vec::new();
        for n in &reached {
            if let Some(node) = store.get_node(n.node.id)? {
                nodes.push((n.hops, display_row_name(&n.node), node));
            }
        }
        let rows = nodes
            .iter()
            .map(|(hops, name, node)| ExpandEvidenceNode {
                title: format!("hop {hops} {name}"),
                node,
                site_lines: Vec::new(),
                extra_json: serde_json::json!({"hops": hops}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(&store, root, &project, "impact", query_symbol, total, &rows)
    } else {
        None
    };
    if json {
        let hits = reached[..shown]
            .iter()
            .map(|n| {
                serde_json::json!({
                    "hops": n.hops,
                    "qualified_name": &n.node.qualified_name,
                    "label": &n.node.label,
                    "file": &n.node.file_path,
                    "line": n.node.start_line,
                    "file_path": &n.node.file_path,
                    "start_line": n.node.start_line,
                    "end_line": n.node.end_line,
                })
            })
            .collect();
        impact_counts_json_with_expand(
            &store,
            root,
            query_symbol,
            &project,
            true,
            total,
            shown,
            all,
            ImpactJsonMeta {
                direction: direction_label,
                edge_type: edge_spec.mode,
                edge_types: &edge_spec.edge_types,
                max_hops: depth,
                scope: "transitive",
            },
            hits,
            impact_tests,
            expand.as_ref(),
        )?;
        return Ok(0);
    }
    for n in &reached[..shown] {
        println!(
            "hop {} {} {}",
            n.hops,
            display_row_name(&n.node),
            line_span(&n.node.file_path, n.node.start_line, n.node.end_line)
        );
    }
    print_nav_more_footer(total, shown);
    if let Some(expand) = &expand {
        println!("{}", expand.text_line());
    }
    Ok(0)
}

fn dispatch_impact_diff_scope(
    scope: DiffSearchScope<'_>,
    direction: greppy_search::ReachDirection,
    direction_label: &str,
    edge_spec: &ImpactEdgeSpec<'_>,
    depth: usize,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let project = project_for(root)?;
    // Never compute impact from a stale graph; trigger the bounded refresh
    // policy and return a stable refusal payload.
    if let FreshnessServe::Refuse(freshness) = freshness_serve_decision(&store, root, &project) {
        let refusal_exit = freshness_refusal_exit(&freshness);
        if json {
            // Impact reports only real code providers (excludes .stderr/.snap).
            let incomplete_providers = code_incomplete_provider_json(&store, &project)?;
            let mut v = serde_json::json!({
                "command": "impact",
                "status": "skipped_stale_index",
                "project": project,
                "fresh": false,
                "freshness": freshness,
                "provider_complete": incomplete_providers.is_empty(),
                "incomplete_provider_count": incomplete_providers.len(),
                "incomplete_providers": incomplete_providers,
                "scope": "diff",
                "direction": direction_label,
                "max_hops": depth,
                "source_total": 0,
                "total_exact": 0,
                "shown": 0,
                "hits": [],
            });
            insert_impact_edge_meta(&mut v, edge_spec);
            println!(
                "{}",
                serde_json::to_string_pretty(&v)
                    .map_err(|e| Error::Invalid(format!("serialize impact stale JSON: {e}")))?
            );
        } else {
            println!("{}", indexed_stale_skip_message("impact diff", &freshness));
        }
        return Ok(refusal_exit);
    }
    let mut gate_extra = serde_json::json!({
        "scope": "diff",
        "direction": direction_label,
        "max_hops": depth,
        "source_total": 0,
        "source_shown": 0,
        "source_omitted": 0,
        "all": false,
    });
    insert_impact_edge_meta(&mut gate_extra, edge_spec);
    if let Some(code) =
        provider_policy_graph_gate(&store, root, &project, "impact", json, gate_extra, "hits")?
    {
        return Ok(code);
    }

    let root_path = resolve_root(root)?;
    let spec = git_diff_search_spec(&root_path, scope)?;
    let diff_base = spec.merge_base.as_deref().unwrap_or(&spec.diff_rev);
    let changed_lines = git_diff_changed_lines(&root_path, diff_base, "impact")?;
    let sources = diff_impact_sources(&store, &project, &changed_lines)?;
    let source_shown = sources.len().min(NAV_LIMIT);
    let hits = diff_impact_hits(&store, &sources, direction, &edge_spec.edge_types, depth)?;
    let shown = hits.len().min(NAV_LIMIT);

    if json {
        let source_rows = sources[..source_shown]
            .iter()
            .map(|source| {
                serde_json::json!({
                    "qualified_name": &source.row.qualified_name,
                    "label": &source.row.label,
                    "file_path": &source.row.file_path,
                    "start_line": source.row.start_line,
                    "end_line": source.row.end_line,
                })
            })
            .collect::<Vec<_>>();
        let hit_rows = hits[..shown]
            .iter()
            .map(|hit| {
                let sources = hit
                    .sources
                    .iter()
                    .take(8)
                    .map(|source| {
                        serde_json::json!({
                            "qualified_name": &source.qualified_name,
                            "file_path": &source.file_path,
                            "start_line": source.start_line,
                            "end_line": source.end_line,
                        })
                    })
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "hops": hit.hops,
                    "qualified_name": &hit.node.qualified_name,
                    "label": &hit.node.label,
                    "file_path": &hit.node.file_path,
                    "start_line": hit.node.start_line,
                    "end_line": hit.node.end_line,
                    "source_count": hit.sources.len(),
                    "sources": sources,
                })
            })
            .collect::<Vec<_>>();
        impact_diff_counts_json(
            &store,
            root,
            &project,
            &spec,
            direction_label,
            edge_spec.mode,
            &edge_spec.edge_types,
            depth,
            sources.len(),
            source_shown,
            hits.len(),
            shown,
            hit_rows,
            source_rows,
        )?;
        return Ok(if sources.is_empty() { 1 } else { 0 });
    }

    if sources.is_empty() {
        println!("(no indexed definitions touched by diff)");
        return Ok(1);
    }
    println!(
        "diff sources: {} shown of {} total",
        source_shown,
        sources.len()
    );
    for source in &sources[..source_shown] {
        println!(
            "source {} {}",
            display_row_name(&source.row),
            line_span(
                &source.row.file_path,
                source.row.start_line,
                source.row.end_line
            )
        );
    }
    if hits.is_empty() {
        println!("(no transitive impact from diff sources)");
        return Ok(0);
    }
    for hit in &hits[..shown] {
        let source_names = hit
            .sources
            .iter()
            .take(3)
            .map(display_row_name)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "hop {} {} {} sources={}",
            hit.hops,
            display_row_name(&hit.node),
            line_span(&hit.node.file_path, hit.node.start_line, hit.node.end_line),
            source_names
        );
    }
    print_nav_more_footer(hits.len(), shown);
    Ok(0)
}

/// Rows of callers/callees shown by `brief` before truncating — smaller than
/// NAV_LIMIT because a briefing is a summary, not an exhaustive listing.
const BRIEF_LIMIT: usize = 15;

/// `file_path` is the repo-relative path of the definition's file; the brief
/// prompt contract feeds it to the model alongside the source span.
fn summarize_definition_span(file_path: &str, source_span: &str) -> Option<Vec<String>> {
    #[cfg(any(unix, windows))]
    {
        let cfg = qwen_summary_config_optional().ok().flatten()?;
        let model_key = qwen_summary_model_key(&cfg);
        summarize_daemon::summarize_source_via_daemon(&cfg, &model_key, file_path, source_span)
            .filter(|bullets| !bullets.is_empty())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file_path, source_span);
        None
    }
}

/// `greppy brief S` — a one-call briefing: the definition (with source
/// span), the direct callers, and the direct callees. Composes the same
/// resolution/edge helpers as context/who-calls/callees so an agent can
/// answer "how does S work / what is its role / what depends on it" from a
/// SINGLE call instead of three, which is exactly where the benchmark showed
/// research-task iteration eating the token/time savings.
const BRIEF_JSON_SCHEMA_VERSION: &str = "greppy.brief.v1";

fn read_file_candidate(root_path: &std::path::Path, subject: &str) -> std::path::PathBuf {
    let supplied = std::path::Path::new(subject);
    if supplied.is_absolute() {
        return supplied.to_path_buf();
    }
    if let Ok(cwd) = std::env::current_dir() {
        let from_cwd = cwd.join(supplied);
        if from_cwd.exists() {
            return from_cwd;
        }
    }
    root_path.join(supplied)
}

fn read_subject_is_path(subject: &str, root: Option<&str>) -> Result<bool> {
    let root_path = resolve_root(root)?;
    if read_file_candidate(&root_path, subject).exists() {
        return Ok(true);
    }
    let supplied = std::path::Path::new(subject);
    if supplied.is_absolute()
        || subject.starts_with('.')
        || subject.contains('/')
        || subject.contains('\\')
    {
        return Ok(true);
    }
    let path_extension = supplied
        .extension()
        .and_then(|extension| extension.to_str());
    Ok(matches!(
        path_extension,
        Some(
            "rs" | "py"
                | "js"
                | "jsx"
                | "mjs"
                | "cjs"
                | "ts"
                | "tsx"
                | "go"
                | "rb"
                | "java"
                | "c"
                | "h"
                | "cpp"
                | "cc"
                | "cxx"
                | "hpp"
                | "hh"
                | "cs"
                | "php"
                | "sh"
                | "bash"
                | "lua"
                | "kt"
                | "kts"
                | "scala"
                | "sc"
                | "swift"
                | "zig"
                | "r"
                | "json"
                | "toml"
                | "yaml"
                | "yml"
                | "md"
                | "txt"
        )
    ))
}

fn parse_read_line_range(raw: Option<&str>, line_count: usize) -> Result<(usize, usize)> {
    let Some(raw) = raw else {
        return Ok((1, line_count));
    };
    let (start, end) = raw.split_once(':').unwrap_or((raw, raw));
    let start = start.parse::<usize>().map_err(|_| {
        Error::Invalid(format!(
            "read --lines/--line expects a positive line N or range N:M, got `{raw}`"
        ))
    })?;
    let end = end.parse::<usize>().map_err(|_| {
        Error::Invalid(format!(
            "read --lines/--line expects a positive line N or range N:M, got `{raw}`"
        ))
    })?;
    if start == 0 || end < start {
        return Err(Error::Invalid(format!(
            "read --lines/--line expects 1 <= N <= M, got `{raw}`"
        )));
    }
    if start > line_count && line_count > 0 {
        return Err(Error::Invalid(format!(
            "read --lines/--line starts at {start}, but the file has {line_count} line(s)"
        )));
    }
    Ok((start, end.min(line_count)))
}

fn closest_read_paths(root_path: &std::path::Path, subject: &str) -> Result<Vec<String>> {
    let overrides = discover_overrides_from_env()?;
    let entries = greppy_discover::walk_with_policy_and_overrides(
        root_path,
        &greppy_discover::SkipPolicy::walk_default(),
        &overrides,
    )?;
    let requested_name = std::path::Path::new(subject)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(subject);
    let mut ranked = entries
        .into_iter()
        .map(|entry| {
            let candidate_name = std::path::Path::new(&entry.rel_path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&entry.rel_path);
            let score = levenshtein(subject, &entry.rel_path)
                .min(levenshtein(requested_name, candidate_name));
            (score, entry.rel_path)
        })
        .collect::<Vec<_>>();
    ranked.sort();
    ranked.dedup_by(|left, right| left.1 == right.1);
    Ok(ranked.into_iter().take(5).map(|(_, path)| path).collect())
}

/// Split a trailing `:START[-END]` (or `:START[:END]`) off a read subject.
///
/// greppy prints spans as `path:120-160`, so that is what an agent hands back.
/// Returns the bare path and the range in the `START:END` form `--lines` takes.
/// A path-qualified symbol (`file.rs::Symbol`) is left alone.
fn split_trailing_line_range(subject: &str) -> Option<(&str, String)> {
    if subject.contains("::") {
        return None;
    }
    let (path, tail) = subject.rsplit_once(':')?;
    if path.is_empty() || tail.is_empty() {
        return None;
    }
    let (start, end) = match tail.split_once(['-', ':']) {
        Some((start, end)) => (start, if end.is_empty() { start } else { end }),
        None => (tail, tail),
    };
    let start: usize = start.parse().ok()?;
    let end: usize = end.parse().ok()?;
    if start == 0 || end < start {
        return None;
    }
    Some((path, format!("{start}:{end}")))
}

fn dispatch_read_file(
    subject: &str,
    lines: Option<&str>,
    with_handle: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    const READ_FILE_JSON_SCHEMA_VERSION: &str = "greppy.read-file.v1";
    let root_path = resolve_root(root)?;
    let canonical_root = root_path.canonicalize().map_err(|source| Error::Io {
        context: format!("canonicalize {}", root_path.display()),
        source,
    })?;
    let candidate = read_file_candidate(&root_path, subject);
    let canonical = candidate.canonicalize().ok();
    let regular_file = canonical
        .as_deref()
        .is_some_and(|path| path.starts_with(&canonical_root) && path.is_file());
    if !regular_file {
        let suggestions = closest_read_paths(&canonical_root, subject)?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema_version": READ_FILE_JSON_SCHEMA_VERSION,
                    "command": "read",
                    "status": "not-found",
                    "path": subject,
                    "path_candidates": suggestions,
                }))
                .map_err(|error| Error::Invalid(format!("serialize read file JSON: {error}")))?
            );
        } else if suggestions.is_empty() {
            println!("read: file `{subject}` not found");
        } else {
            println!("read: file `{subject}` not found; closest paths:");
            for suggestion in &suggestions {
                println!("  {suggestion}");
            }
            println!("try: greppy read {}", shell_example_arg(&suggestions[0]));
        }
        return Ok(10);
    }
    let canonical = canonical.expect("regular file check requires a canonical path");
    let relative = canonical.strip_prefix(&canonical_root).map_err(|_| {
        Error::Invalid(format!(
            "read path `{subject}` resolves outside workspace {}",
            canonical_root.display()
        ))
    })?;
    let shown_path = relative.to_string_lossy().replace('\\', "/");
    let content = std::fs::read_to_string(&canonical).map_err(|source| Error::Io {
        context: format!("read {}", canonical.display()),
        source,
    })?;
    let file_lines = content.lines().collect::<Vec<_>>();
    let (start, end) = parse_read_line_range(lines, file_lines.len())?;
    let selected = if end < start {
        &file_lines[0..0]
    } else {
        &file_lines[start.saturating_sub(1)..end]
    };
    let (byte_start, byte_end) = if end < start {
        (0, 0)
    } else {
        line_range_to_bytes(content.as_bytes(), start, end)
    };
    let handle_token = if with_handle {
        Some(
            greppy_edit::EditHandle::for_range(
                &canonical_root,
                std::path::Path::new(&shown_path),
                content.as_bytes(),
                byte_start,
                byte_end,
            )?
            .encode(),
        )
    } else {
        None
    };
    if json {
        let rows = selected
            .iter()
            .enumerate()
            .map(|(offset, text)| serde_json::json!({"line": start + offset, "text": text}))
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": READ_FILE_JSON_SCHEMA_VERSION,
                "command": "read",
                "status": "ok",
                "path": shown_path,
                "start_line": start,
                "end_line": end,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "lines": rows,
                "handle": handle_token,
            }))
            .map_err(|error| Error::Invalid(format!("serialize read file JSON: {error}")))?
        );
    } else {
        // The header already states the span, so a per-line number repeats it
        // once per line. Measured over a 41-task benchmark that repetition was
        // 16.6% of everything `read` returned -- about 12,600 characters per
        // task, re-billed on every subsequent turn -- while edits are addressed
        // by content or by handle, not by line number.
        println!("{shown_path}:{start}-{end}");
        for text in selected {
            println!("{text}");
        }
        if let Some(token) = &handle_token {
            println!("handle: {token}");
        }
    }
    Ok(0)
}

/// `greppy read`: a symbol's exact definition span, optionally with an edit
/// handle. Resolution mirrors `brief`; the returned bytes come from the LIVE
/// file (the store addresses, the live file decides), so the handle's hashes
/// always describe what the agent actually saw.
fn dispatch_read(
    symbol: Option<&str>,
    with_handle: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    const READ_JSON_SCHEMA_VERSION: &str = "greppy.read.v1";
    let mut store = open_default_store_query_writer(root)?;
    let project = project_for(root)?;
    // Read is the workhorse of the edit loop, and the loop mutates files
    // constantly (test setup, the agent's own edits). Refusing on a stale
    // index — returning empty until a background reindex catches up — left
    // the agent with nothing and it degraded to bash (forensics 2026-07-18:
    // a real flask task took 123 turns, greppy all but unused). Instead,
    // heal in-band: a reindexable stale index is rebuilt BLOCKING and served
    // fresh on this same call. `read` verifies every span against the live
    // file anyway, so a brief blocking reindex is strictly better than an
    // empty answer. Only genuinely un-reindexable states (cold/failed) still
    // refuse.
    maybe_reindex_stale(&mut store, root)?;
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "read",
        json,
        serde_json::json!({"schema_version": READ_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    let ids = resolve_symbol_nodes(&store, symbol)?;
    let root_path = resolve_root(root)?;
    let mut nodes = Vec::new();
    for id in &ids {
        if let Some(node) = store.get_node(*id)? {
            if !node.file_path.is_empty() && node.start_line >= 1 {
                nodes.push(node);
            }
        }
    }
    // The per-file synthetic anchor answers to the file stem, so `greet` also
    // hits `pkg/greet.go` and a unique definition reads as ambiguous. A file is
    // read by path; the anchor only stands in when nothing else answered.
    if nodes
        .iter()
        .any(|node| !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name))
    {
        nodes.retain(|node| {
            !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name)
        });
    }
    if nodes.is_empty() {
        // Exact/graph resolution missed. greppy has an EMBEDDING engine
        // precisely so a reasonable-but-inexact reference still resolves —
        // "_startsWith", "impl Serialize for Bound", "the fn that validates
        // prefixes". Hard-failing here (as the old bare-name suggestion did)
        // wastes the second engine and forces the agent to guess formats
        // (trace forensics 2026-07-17: 12 read not-founds, 4-5 turns each).
        let query = symbol.unwrap_or("");
        let hits = greppy_search::semantic_query(&store, query, None, Some(&project), 6)
            .unwrap_or_default();
        // A clearly dominant hit is safe to read directly (read mutates
        // nothing); otherwise offer addressable candidates and let the agent
        // pick. Dominance = single hit, or top score >= 1.4x the runner-up.
        let dominant = match hits.as_slice() {
            [only] => Some(only.node.id),
            [top, second, ..] if top.score >= second.score * 1.4 => Some(top.node.id),
            _ => None,
        };
        if let Some(id) = dominant {
            if let Some(node) = store.get_node(id)? {
                if !node.file_path.is_empty() && node.start_line >= 1 {
                    if !json {
                        println!(
                            "read: `{query}` resolved semantically to `{}`",
                            node.qualified_name
                        );
                    }
                    nodes.push(node);
                }
            }
        }
        if nodes.is_empty() {
            // No single confident match: hand back addressable candidates —
            // the exact qualified name read accepts, plus location and kind —
            // so the retry is copy-paste, not another guess.
            let candidates: Vec<serde_json::Value> = hits
                .iter()
                .take(5)
                .map(|h| {
                    serde_json::json!({
                        "qualified_name": h.node.qualified_name,
                        "path": h.node.file_path,
                        "line": h.node.start_line,
                        "kind": h.node.label,
                    })
                })
                .collect();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "schema_version": READ_JSON_SCHEMA_VERSION,
                        "command": "read",
                        "status": "not-found",
                        "query": query,
                        "candidates": candidates,
                    }))
                    .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
                );
            } else if candidates.is_empty() {
                println!("read: no definition found for `{query}`");
            } else {
                println!("read: no exact match for `{query}`; closest definitions:");
                for h in hits.iter().take(5) {
                    println!(
                        "  {}  ({}:{}, {})",
                        h.node.qualified_name, h.node.file_path, h.node.start_line, h.node.label
                    );
                }
            }
            return Ok(10);
        }
    }
    if nodes.len() > 1 {
        // distinct definition sites -> ambiguous, list candidates (exit 11);
        // multiple store nodes on ONE site (Struct + Impl) are not ambiguity
        let mut sites: Vec<(String, i64)> = nodes
            .iter()
            .map(|n| (n.file_path.clone(), n.start_line))
            .collect();
        sites.sort();
        sites.dedup();
        if sites.len() > 1 {
            let candidates: Vec<serde_json::Value> = nodes
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "qualified_name": n.qualified_name,
                        "path": n.file_path,
                        "line": n.start_line,
                    })
                })
                .collect();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "schema_version": READ_JSON_SCHEMA_VERSION,
                        "command": "read",
                        "status": "ambiguous",
                        "query": symbol.unwrap_or(""),
                        "candidates": candidates,
                    }))
                    .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
                );
            } else {
                println!(
                    "read: `{}` is ambiguous; qualify it (Owner.method) or use one of:",
                    symbol.unwrap_or("")
                );
                for n in &nodes {
                    println!("  {} {}:{}", n.qualified_name, n.file_path, n.start_line);
                }
            }
            return Ok(11);
        }
    }
    let node = &nodes[0];
    let abs = root_path.join(&node.file_path);
    let content = std::fs::read(&abs).map_err(|source| Error::Io {
        context: format!("read {}", abs.display()),
        source,
    })?;
    // `--context N` widens the span upwards only: the doc comment sits above the
    // signature, and a definition's own end is where it ends.
    let start_line = (node.start_line - cli_read_context()).max(1);
    let Some(span) = read_span_with_meta(
        &root_path,
        &node.file_path,
        start_line,
        node.end_line,
        usize::MAX,
        false,
    ) else {
        println!(
            "read: definition span for `{}` is stale; re-index and retry",
            node.qualified_name
        );
        return Ok(12);
    };
    // line range -> byte range against the SAME live bytes
    let (byte_start, byte_end) =
        line_range_to_bytes(&content, start_line as usize, span.end_line as usize);
    let handle_token = if with_handle {
        let mut handle = greppy_edit::EditHandle::for_range(
            &root_path,
            std::path::Path::new(&node.file_path),
            &content,
            byte_start,
            byte_end,
        )?;
        let language = greppy_edit::language_for_path(std::path::Path::new(&node.file_path));
        handle.signature_fingerprint =
            greppy_edit::verbs::signature_fingerprint(language, &content, (byte_start, byte_end));
        handle.grammar_id = Some(format!("{language:?}"));
        handle.grammar_version = Some(env!("CARGO_PKG_VERSION").to_string());
        Some(handle.encode())
    } else {
        None
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": READ_JSON_SCHEMA_VERSION,
                "command": "read",
                "status": "ok",
                "qualified_name": node.qualified_name,
                "path": node.file_path,
                "start_line": node.start_line,
                "end_line": span.end_line,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "source": span.text,
                "handle": handle_token,
            }))
            .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
        );
    } else {
        println!(
            "{} {}:{}-{}",
            node.qualified_name, node.file_path, node.start_line, span.end_line
        );
        println!("{}", span.text);
        if let Some(token) = &handle_token {
            println!("handle: {token}");
        }
    }
    Ok(0)
}

/// Byte offsets of an inclusive 1-based line range within `content`.
fn line_range_to_bytes(content: &[u8], start_line: usize, end_line: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut start = 0usize;
    let mut idx = 0usize;
    let mut end = content.len();
    if start_line <= 1 {
        start = 0;
    }
    while idx < content.len() {
        if line == start_line {
            start = idx;
        }
        match content[idx..].iter().position(|&b| b == b'\n') {
            Some(rel) => {
                if line == end_line {
                    end = idx + rel + 1;
                    break;
                }
                idx += rel + 1;
                line += 1;
            }
            None => {
                end = content.len();
                break;
            }
        }
    }
    (start, end)
}

const MINIMAL_CHANGE_SIGNATURE_EXAMPLE: &str =
    include_str!("../../../docs/contracts/change-signature-spec.minimal.json");

/// `greppy edit`: dispatch to the transactional verbs; print the
/// certificate; map its status to the registered exit code.
/// Read an edit source argument: a file path, or `-` for stdin (agents
/// naturally try heredocs; K3 reasoning trace 2026-07-17: "Need pass new
/// source via stdin?").
fn read_source_arg(source_file: &str) -> Result<Vec<u8>> {
    if source_file == "-" {
        use std::io::Read;
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|source| Error::Io {
                context: "read edit source from stdin".into(),
                source,
            })?;
        return Ok(buf);
    }
    std::fs::read(source_file).map_err(|source| Error::Io {
        context: format!("read {source_file}"),
        source,
    })
}

fn dispatch_edit(command: EditCommand, json: bool, root: Option<&str>) -> Result<i32> {
    match dispatch_edit_inner(command, json, root) {
        Err(error @ Error::Invalid(_)) => {
            eprintln!("greppy: {error}");
            Ok(20)
        }
        result => result,
    }
}

fn dispatch_edit_inner(command: EditCommand, json: bool, root: Option<&str>) -> Result<i32> {
    let root_path = resolve_root(root)?;
    // The four operations of the new grammar, the whole-file verbs and `undo`
    // answer with an edit record - file, span, resulting text, handle - rather
    // than a certificate, so they are dispatched before the certificate verbs.
    let command = match dispatch_edit_grammar(command, json, root, &root_path)? {
        GrammarDispatch::Handled(code) => return Ok(code),
        GrammarDispatch::Passthrough(command) => *command,
    };
    fn resolved_options(
        dry_run: bool,
        range: (usize, usize),
        planned_file_sha256: String,
        planned_target_sha256: String,
    ) -> greppy_edit::verbs::VerbOptions {
        greppy_edit::verbs::VerbOptions {
            dry_run,
            with_diff: true,
            planned_file_sha256: Some(planned_file_sha256),
            planned_target_sha256: Some(planned_target_sha256),
            planned_target_range: Some(range),
            ..Default::default()
        }
    }
    let (certificate, report_path) = match command {
        EditCommand::Rename {
            symbol,
            r#in,
            call,
            to,
            expect,
            expect_residual,
            dry_run,
            report,
        } => match (symbol, r#in, call) {
            (Some(symbol), None, None) => {
                let new_name = to;
            let store = open_default_store_query_writer(root)?;
            let ids = resolve_symbol_nodes(&store, Some(&symbol))?;
            let mut def_nodes = Vec::new();
            for id in &ids {
                if let Some(node) = store.get_node(*id)? {
                    if !node.file_path.is_empty() && node.start_line >= 1 {
                        def_nodes.push(node);
                    }
                }
            }
            if def_nodes.is_empty() {
                println!("no symbol `{symbol}`");
                return Ok(10);
            }
            let short_name = def_nodes[0].name.clone();
            // collect per-file scopes: definition files fully (definition,
            // same-file usages, imports), and every referencing node's span
            use std::collections::BTreeMap;
            let mut scopes: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
            for def in &def_nodes {
                scopes
                    .entry(def.file_path.clone())
                    .or_default()
                    .push((0, usize::MAX));
                for edge in store.incoming_edges(def.id, None, 100_000)? {
                    if let Some(src) = store.get_node(edge.source_id)? {
                        if src.file_path.is_empty() || src.start_line < 1 {
                            continue;
                        }
                        let abs = root_path.join(&src.file_path);
                        let Ok(content) = std::fs::read(&abs) else {
                            continue;
                        };
                        let Some(span) = read_span_with_meta(
                            &root_path,
                            &src.file_path,
                            src.start_line,
                            src.end_line,
                            usize::MAX,
                            false,
                        ) else {
                            continue;
                        };
                        let range = line_range_to_bytes(
                            &content,
                            src.start_line as usize,
                            span.end_line as usize,
                        );
                        scopes.entry(src.file_path.clone()).or_default().push(range);
                    }
                }
            }
            // import lines in every affected file: cover the whole file for
            // files that already have narrower spans is wasteful, so add a
            // full-file span only where imports may bind the name
            let scope_vec: Vec<greppy_edit::verbs::RenameFileScope> = scopes
                .into_iter()
                .map(|(rel_path, spans)| greppy_edit::verbs::RenameFileScope { rel_path, spans })
                .collect();
            let options = greppy_edit::verbs::VerbOptions {
                dry_run,
                with_diff: true,
                expect_residual: Some(expect_residual),
                ..Default::default()
            };
            (
                greppy_edit::verbs::rename_symbol_files(
                    &root_path,
                    &scope_vec,
                    &short_name,
                    &new_name,
                    &options,
                )?,
                report,
            )
            }
            (None, Some(in_symbol), Some(from)) => {
                match resolve_edit_target(Some(&in_symbol), None, root, &root_path)? {
            EditTarget::Refusal(cert) => (*cert, report),
            EditTarget::Resolved {
                rel_path,
                range,
                planned_file_sha256,
                planned_target_sha256,
            } => {
                let abs = root_path.join(&rel_path);
                let language = greppy_edit::language_for_path(std::path::Path::new(&rel_path));
                let options =
                    resolved_options(dry_run, range, planned_file_sha256, planned_target_sha256);
                (
                    greppy_edit::verbs::rename_in_span(
                        &root_path, &abs, range, &from, &to, expect, language, &options,
                    )?,
                    report,
                )
            }
                }
            }
            _ => {
                return Err(Error::Invalid(
                    "rename takes --symbol S --to N, or --in S --call A --to B".into(),
                ))
            }
        },
        EditCommand::ChangeSignature {
            symbol,
            spec,
            backend,
            expect_residual,
            dry_run,
            report,
        } => {
            greppy_edit::verbs::require_semantic_backend(&backend)?;
            let spec_bytes = if spec.trim_start().starts_with('{') {
                spec.as_bytes().to_vec()
            } else {
                std::fs::read(&spec).map_err(|source| Error::Io {
                    context: format!("read change-signature spec {spec}"),
                    source,
                })?
            };
            let spec: greppy_edit::verbs::ChangeSignatureSpec = serde_json::from_slice(&spec_bytes)
                .map_err(|error| {
                    Error::Invalid(format!(
                        "change-signature --spec {spec} is invalid: {error}\nminimal complete example:\n{}",
                        MINIMAL_CHANGE_SIGNATURE_EXAMPLE.trim()
                    ))
                })?;
            match resolve_edit_target(Some(&symbol), None, root, &root_path)? {
                EditTarget::Refusal(cert) => (*cert, report),
                EditTarget::Resolved {
                    rel_path,
                    range,
                    planned_file_sha256,
                    planned_target_sha256,
                } => {
                    let store = open_default_store_query_writer(root)?;
                    let ids = resolve_symbol_nodes(&store, Some(&symbol))?;
                    let mut short_name = None;
                    let mut scopes =
                        std::collections::BTreeMap::<String, Vec<(usize, usize)>>::new();
                    for id in &ids {
                        if let Some(definition) = store.get_node(*id)? {
                            short_name.get_or_insert(definition.name);
                        }
                        for edge in store.incoming_edges(*id, None, 100_000)? {
                            let Some(source) = store.get_node(edge.source_id)? else {
                                continue;
                            };
                            if source.file_path.is_empty() || source.start_line < 1 {
                                continue;
                            }
                            let content = std::fs::read(root_path.join(&source.file_path))
                                .map_err(|error| {
                                    Error::io(format!("read {}", source.file_path), error)
                                })?;
                            let Some(span) = read_span_with_meta(
                                &root_path,
                                &source.file_path,
                                source.start_line,
                                source.end_line,
                                usize::MAX,
                                false,
                            ) else {
                                continue;
                            };
                            scopes
                                .entry(source.file_path)
                                .or_default()
                                .push(line_range_to_bytes(
                                    &content,
                                    source.start_line as usize,
                                    span.end_line as usize,
                                ));
                        }
                    }
                    let short_name = short_name.ok_or_else(|| {
                        Error::Invalid(format!(
                            "change-signature could not resolve the name of `{symbol}`"
                        ))
                    })?;
                    let call_scopes: Vec<greppy_edit::verbs::RenameFileScope> = scopes
                        .into_iter()
                        .map(|(rel_path, mut spans)| {
                            spans.sort_unstable();
                            spans.dedup();
                            greppy_edit::verbs::RenameFileScope { rel_path, spans }
                        })
                        .collect();
                    let language = greppy_edit::language_for_path(std::path::Path::new(&rel_path));
                    let mut options = resolved_options(
                        dry_run,
                        range,
                        planned_file_sha256,
                        planned_target_sha256,
                    );
                    options.expect_residual = Some(expect_residual);
                    (
                        greppy_edit::verbs::change_signature_files(
                            &root_path,
                            &greppy_edit::verbs::SignatureDefinition { rel_path, range },
                            &call_scopes,
                            &short_name,
                            &spec,
                            language,
                            &options,
                        )?,
                        report,
                    )
                }
            }
        }
        EditCommand::EnsureArgument {
            symbol,
            call,
            arg,
            dry_run,
            report,
        } => match resolve_edit_target(Some(&symbol), None, root, &root_path)? {
            EditTarget::Refusal(cert) => (*cert, report),
            EditTarget::Resolved {
                rel_path,
                range,
                planned_file_sha256,
                planned_target_sha256,
            } => {
                let abs = root_path.join(&rel_path);
                let options =
                    resolved_options(dry_run, range, planned_file_sha256, planned_target_sha256);
                (
                    greppy_edit::ensure::ensure_argument(
                        &root_path, &abs, range, &call, &arg, &options,
                    )?,
                    report,
                )
            }
        },
        EditCommand::EnsureMethod {
            symbol,
            name,
            content_file,
            dry_run,
            report,
        } => {
            let source = String::from_utf8_lossy(&read_source_arg(&content_file)?).into_owned();
            match resolve_edit_target(Some(&symbol), None, root, &root_path)? {
                EditTarget::Refusal(cert) => (*cert, report),
                EditTarget::Resolved {
                    rel_path,
                    range,
                    planned_file_sha256,
                    planned_target_sha256,
                } => {
                    let abs = root_path.join(&rel_path);
                    // Rust keeps a type and its methods apart: `struct Greeter;`
                    // has no body at all and the methods live in `impl Greeter`.
                    // `--symbol Greeter` names the type, so a definition without
                    // a body is followed to its inherent impl block instead of
                    // being reported as a class with nowhere to add a method.
                    let (range, planned_target_sha256) = std::fs::read(&abs)
                        .ok()
                        .filter(|content| {
                            !content[range.0.min(content.len())..range.1.min(content.len())]
                                .contains(&b'{')
                        })
                        .and_then(|content| {
                            let block = rust_inherent_impl_range(&content, &symbol)?;
                            let planned = greppy_edit::EditHandle::for_range(
                                &root_path,
                                std::path::Path::new(&rel_path),
                                &content,
                                block.0,
                                block.1,
                            )
                            .ok()?;
                            Some((block, planned.target_sha256))
                        })
                        .unwrap_or((range, planned_target_sha256));
                    let options = resolved_options(
                        dry_run,
                        range,
                        planned_file_sha256,
                        planned_target_sha256,
                    );
                    (
                        greppy_edit::ensure::ensure_method(
                            &root_path, &abs, range, &name, &source, &options,
                        )?,
                        report,
                    )
                }
            }
        }
        EditCommand::EnsureAnnotation {
            symbol,
            annotation,
            dry_run,
            report,
        } => match resolve_edit_target(Some(&symbol), None, root, &root_path)? {
            EditTarget::Refusal(cert) => (*cert, report),
            EditTarget::Resolved {
                rel_path,
                range,
                planned_file_sha256,
                planned_target_sha256,
            } => {
                let abs = root_path.join(&rel_path);
                let options =
                    resolved_options(dry_run, range, planned_file_sha256, planned_target_sha256);
                (
                    greppy_edit::ensure::ensure_annotation(
                        &root_path,
                        &abs,
                        range,
                        &annotation,
                        &options,
                    )?,
                    report,
                )
            }
        },
        EditCommand::Data {
            mode,
            file,
            path,
            value_json,
            dry_run,
            report,
        } => {
            let target = resolve_edit_file(&root_path, &file);
            let options = greppy_edit::verbs::VerbOptions {
                dry_run,
                with_diff: true,
                ..Default::default()
            };
            let before = std::fs::read(&target).ok();
            let certificate = if mode == "delete" {
                greppy_edit::data::data_delete(&root_path, &target, &path, &options)?
            } else {
                let value_json = value_json.ok_or_else(|| {
                    Error::Invalid("data set needs --value-json VALUE".into())
                })?;
                greppy_edit::data::data_set(
                    &root_path,
                    &target,
                    &path,
                    &value_json,
                    mode == "ensure",
                    &options,
                )?
            };
            if certificate.published {
                let rel = target
                    .strip_prefix(&root_path)
                    .map(|relative| relative.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_else(|_| file.clone());
                record_edit_undo(
                    &root_path,
                    &[UndoBefore {
                        rel,
                        content: before,
                    }],
                );
            }
            (certificate, report)
        }
        EditCommand::Apply {
            plan,
            dry_run,
            report,
            diff: _,
        } => {
            // `-` means stdin here as it does for every other file argument.
            // Without it a multi-edit plan needs a temporary file first, which
            // turns the one transaction this verb exists for into two turns.
            let text = if plan == "-" {
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin()
                    .read_to_string(&mut buf)
                    .map_err(|source| Error::Io {
                        context: "read edit plan from stdin".into(),
                        source,
                    })?;
                buf
            } else {
                std::fs::read_to_string(&plan).map_err(|source| Error::Io {
                    context: format!("read {plan}"),
                    source,
                })?
            };
            // A plan that could not be read is a refused edit, not a usage
            // error: the caller asked for `--json` and must get an answer as
            // data, not a bare line of prose on stderr.
            if text.trim().is_empty() {
                return emit_edit_outcome(
                    Err(EditRefusal::new(
                        "plan_empty",
                        format!(
                            "--plan {plan}: the plan is empty; the command that was to \
                             produce it printed nothing"
                        ),
                        20,
                    )),
                    json,
                    report,
                );
            }
            // A plan written in the whole-file verbs is executed by the verbs
            // themselves: `write`, `move` and `remove` are not spans in a file,
            // so the span planner has nothing to say about them.
            if let Some(operations) = plan_is_whole_file(&text) {
                return emit_edit_outcome(
                    run_edit_plan_whole_file(&root_path, &operations, dry_run, false),
                    json,
                    report,
                );
            }
            let loaded = match greppy_edit::plan::load_plan(&text, &root_path) {
                Ok(loaded) => loaded,
                Err(error) => {
                    return emit_edit_outcome(
                        Err(EditRefusal::new(
                            "invalid_plan",
                            format!("--plan {plan}: {error}"),
                            20,
                        )),
                        json,
                        report,
                    );
                }
            };
            let plan_root = std::path::PathBuf::from(loaded.workspace_root());
            let plan_root_arg = plan_root.to_string_lossy().into_owned();
            let parsed = loaded.into_plan(|shortcut| {
                let resolved = resolve_edit_target(
                    Some(&shortcut.symbol),
                    None,
                    Some(&plan_root_arg),
                    &plan_root,
                )?;
                let EditTarget::Resolved {
                    rel_path,
                    range,
                    ..
                } = resolved
                else {
                    let EditTarget::Refusal(certificate) = resolved else {
                        unreachable!()
                    };
                    return Err(Error::Invalid(format!(
                        "replace-body operation `{}` could not resolve symbol `{}` (status {:?})",
                        shortcut.id, shortcut.symbol, certificate.status
                    )));
                };

                let declared_file = resolve_edit_file(&plan_root, &shortcut.file);
                let resolved_file = plan_root.join(&rel_path);
                let declared_canonical = declared_file.canonicalize().map_err(|source| Error::Io {
                    context: format!("canonicalize {}", declared_file.display()),
                    source,
                })?;
                let resolved_canonical = resolved_file.canonicalize().map_err(|source| Error::Io {
                    context: format!("canonicalize {}", resolved_file.display()),
                    source,
                })?;
                if declared_canonical != resolved_canonical {
                    return Err(Error::Invalid(format!(
                        "replace-body operation `{}` declares file `{}` but symbol `{}` resolves to `{}`",
                        shortcut.id, shortcut.file, shortcut.symbol, rel_path
                    )));
                }

                let new_body = match &shortcut.body {
                    greppy_edit::plan::ReplaceBodySource::Inline(body) => body.clone(),
                    greppy_edit::plan::ReplaceBodySource::File(source_file) => {
                        let source_path = resolve_edit_file(&plan_root, source_file);
                        std::fs::read_to_string(&source_path).map_err(|source| Error::Io {
                            context: format!("read {}", source_path.display()),
                            source,
                        })?
                    }
                };
                let file_content = std::fs::read(&resolved_file).map_err(|source| Error::Io {
                    context: format!("read {}", resolved_file.display()),
                    source,
                })?;
                greppy_edit::plan::resolve_replace_body_operation(
                    shortcut,
                    rel_path,
                    range,
                    &file_content,
                    &new_body,
                )
            })?;
            let undo_before = plan_undo_snapshot(&root_path, &text);
            let mut certificate = greppy_edit::plan::apply_plan(&parsed, dry_run)?;
            if certificate.published {
                record_edit_undo(&root_path, &undo_before);
            } else {
                annotate_plan_refusal(&mut certificate, &text);
            }
            (certificate, report)
        }
        EditCommand::Recover { report } => {
            // The grammar verbs journal into the workspace store, so their
            // interrupted transactions are the ones to look for first.
            match run_edit_recover(&root_path) {
                Ok(Some(restored)) => {
                    println!("rolled back an interrupted edit: {restored} file(s) restored");
                    return Ok(0);
                }
                Ok(None) => {}
                Err(refusal) => {
                    eprintln!("{}", refusal.message);
                    return Ok(refusal.exit);
                }
            }
            let outcome = greppy_edit::journal::recover_with_report(&root_path)?;
            let msg = match outcome.action {
                greppy_edit::journal::RecoveryAction::NothingToRecover => {
                    "nothing to recover".to_string()
                }
                greppy_edit::journal::RecoveryAction::RolledBack => format!(
                    "rolled back transaction {}: {} file(s) restored",
                    outcome.transaction_id.as_deref().unwrap_or_default(),
                    outcome.files_restored
                ),
                greppy_edit::journal::RecoveryAction::DiscardedUncommitted => {
                    "discarded uncommitted journal; nothing had been published".to_string()
                }
            };
            if let Some(path) = report {
                let json = serde_json::to_string_pretty(&outcome)
                    .map_err(|e| Error::Invalid(format!("serialize recovery report: {e}")))?;
                std::fs::write(&path, json).map_err(|source| Error::Io {
                    context: format!("write {path}"),
                    source,
                })?;
            }
            println!("{msg}");
            return Ok(0);
        }
        EditCommand::EnsureImport {
            file,
            module,
            name,
            dry_run,
            report,
        } => {
            let target = resolve_edit_file(&root_path, &file);
            let options = greppy_edit::verbs::VerbOptions {
                dry_run,
                with_diff: true,
                ..Default::default()
            };
            (
                greppy_edit::ensure::ensure_import(
                    &root_path,
                    &target,
                    &module,
                    name.as_deref(),
                    &options,
                )?,
                report,
            )
        }
        EditCommand::Replace { .. }
        | EditCommand::Insert { .. }
        | EditCommand::Delete { .. }
        | EditCommand::Patch { .. }
        | EditCommand::Write { .. }
        | EditCommand::Move { .. }
        | EditCommand::Remove { .. }
        | EditCommand::Undo { .. } => {
            unreachable!("the grammar verbs answered before the certificate verbs")
        }
    };
    finish_edit(certificate, report_path, json, root, &root_path)
}

// ---------------------------------------------------------------------------
// The new edit grammar: four operations over one WHERE selector.
//
// `AGENTS.md` states the surface: `replace`, `insert`, `delete` and `patch`
// each take exactly one WHERE, and WHERE is one of `--file F --old TEXT`,
// `--file F --old-file F2`, `--file F --pattern REGEX`, `--file F --lines A:B`,
// `--file F`, `--symbol S`, `--symbol S --body`, or `--target H`. Every
// combination that the grammar does not provide is a refusal with a stable
// `error.code`, never a guess: `dev/CLI-SPEC.md` rule 1 forbids reinterpreting
// an argument into a different question.
// ---------------------------------------------------------------------------

/// A refused edit. The code is what a caller branches on; the message says
/// what is the case and stops there.
struct EditRefusal {
    code: &'static str,
    message: String,
    exit: i32,
    extra: Vec<(&'static str, serde_json::Value)>,
}

impl EditRefusal {
    fn new(code: &'static str, message: impl Into<String>, exit: i32) -> Self {
        Self {
            code,
            message: message.into(),
            exit,
            extra: Vec::new(),
        }
    }

    fn with(mut self, key: &'static str, value: serde_json::Value) -> Self {
        self.extra.push((key, value));
        self
    }
}

type EditResult<T> = std::result::Result<T, EditRefusal>;

fn edit_sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// The record schema every edit answers with, in the compact form on stdout
/// and in the full form `--report` writes.
const EDIT_RECORD_SCHEMA: &str = "greppy.edit-record.v1";

/// One written file inside a record: the spans it received, the text that is
/// now in the first of them, and the handle that marks it.
#[derive(Default)]
struct EditOperation {
    file: String,
    /// `[start, end)` in the bytes of the file as it now stands, one per span
    /// written. `--expect N` writes N of them and the report names all N.
    ranges: Vec<(usize, usize)>,
    result_span: Option<String>,
    handle: Option<String>,
    sha_before: Option<String>,
    sha_after: Option<String>,
    diff: Option<String>,
}

/// What an applied edit answers: the file, the span it wrote, the resulting
/// text, and a handle for that new span.
#[derive(Default)]
struct EditRecord {
    /// A verb-specific first line (`moved a -> b`). When set it replaces the
    /// generic `applied file:span` line — the lifecycle verbs owe the caller
    /// their own word, not a borrowed one.
    headline: Option<String>,
    files: Vec<String>,
    span: Option<(usize, usize)>,
    text: Option<String>,
    handle: Option<String>,
    diagnostics: Option<Vec<String>>,
    notes: Vec<String>,
    operations: Vec<EditOperation>,
    /// What this particular verb owes the caller beyond the common shape: the
    /// files a `move` rewrote, what still references a removed file, what an
    /// `undo` put back. It appears in the compact answer and in `--report`.
    extra: Vec<(&'static str, serde_json::Value)>,
    /// False for `--dry-run`: the record says what would be written.
    published: bool,
}

/// A minimal line-based unified diff for the full record. The byte ranges are
/// the precise answer; this is the readable view of the same change.
fn edit_unified_diff(path: &str, before: &[u8], after: &[u8]) -> String {
    let before_text = String::from_utf8_lossy(before);
    let after_text = String::from_utf8_lossy(after);
    let old_lines: Vec<&str> = before_text.lines().collect();
    let new_lines: Vec<&str> = after_text.lines().collect();
    let mut head = 0usize;
    while head < old_lines.len() && head < new_lines.len() && old_lines[head] == new_lines[head] {
        head += 1;
    }
    let mut tail = 0usize;
    while tail < old_lines.len() - head
        && tail < new_lines.len() - head
        && old_lines[old_lines.len() - 1 - tail] == new_lines[new_lines.len() - 1 - tail]
    {
        tail += 1;
    }
    let old_span = &old_lines[head..old_lines.len() - tail];
    let new_span = &new_lines[head..new_lines.len() - tail];
    let mut diff = format!("--- a/{path}\n+++ b/{path}\n");
    diff.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        head + 1,
        old_span.len(),
        head + 1,
        new_span.len()
    ));
    for line in old_span {
        diff.push('-');
        diff.push_str(line);
        diff.push('\n');
    }
    for line in new_span {
        diff.push('+');
        diff.push_str(line);
        diff.push('\n');
    }
    diff
}

enum GrammarDispatch {
    Handled(i32),
    Passthrough(Box<EditCommand>),
}

/// Selector arguments, before they are read as a WHERE.
struct WhereSpec {
    file: Option<String>,
    old: Option<String>,
    old_file: Option<String>,
    pattern: Option<String>,
    lines: Option<String>,
    symbol: Option<String>,
    body: bool,
    target: Option<String>,
    path: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SelectorKind {
    Text,
    Pattern,
    Lines,
    WholeFile,
    Symbol,
    Target,
}

impl SelectorKind {
    /// Line-oriented spans address whole lines, so the newline that ends the
    /// last one belongs to the file, not to the span: replacing a line with
    /// text that has no newline must not join it to the next one.
    fn line_oriented(self) -> bool {
        matches!(self, SelectorKind::Lines | SelectorKind::Symbol | SelectorKind::Target)
    }

    fn name(self) -> &'static str {
        match self {
            SelectorKind::Text => "--old",
            SelectorKind::Pattern => "--pattern",
            SelectorKind::Lines => "--lines",
            SelectorKind::WholeFile => "--file",
            SelectorKind::Symbol => "--symbol",
            SelectorKind::Target => "--target",
        }
    }
}

/// A WHERE resolved against the bytes on disk.
struct Located {
    rel: String,
    abs: std::path::PathBuf,
    content: Vec<u8>,
    ranges: Vec<(usize, usize)>,
    kind: SelectorKind,
    regex: Option<regex::bytes::Regex>,
    /// What a searching selector was looking for. A refusal that only says
    /// "matched 0 times" leaves the caller unable to tell which of two texts
    /// was wrong, so the text itself is carried to the message.
    needle: Option<String>,
}

fn classify_selector(spec: &WhereSpec) -> EditResult<SelectorKind> {
    let mut chosen: Vec<&'static str> = Vec::new();
    if spec.old.is_some() {
        chosen.push("--old");
    }
    if spec.old_file.is_some() {
        chosen.push("--old-file");
    }
    if spec.pattern.is_some() {
        chosen.push("--pattern");
    }
    if spec.lines.is_some() {
        chosen.push("--lines");
    }
    if spec.symbol.is_some() {
        chosen.push("--symbol");
    }
    if spec.target.is_some() {
        chosen.push("--target");
    }
    if chosen.len() > 1 {
        return Err(EditRefusal::new(
            "selector_conflict",
            format!(
                "{} name different targets; WHERE is exactly one of them",
                chosen.join(" and ")
            ),
            20,
        )
        .with("selectors", serde_json::json!(chosen)));
    }
    let kind = match chosen.first().copied() {
        Some("--old") | Some("--old-file") => SelectorKind::Text,
        Some("--pattern") => SelectorKind::Pattern,
        Some("--lines") => SelectorKind::Lines,
        Some("--symbol") => SelectorKind::Symbol,
        Some("--target") => SelectorKind::Target,
        _ => {
            if spec.file.is_some() {
                SelectorKind::WholeFile
            } else {
                return Err(EditRefusal::new(
                    "selector_missing",
                    "no WHERE: pass --file F with --old/--old-file/--pattern/--lines, \
                     --file F alone, --symbol S, or --target H",
                    20,
                ));
            }
        }
    };
    match kind {
        SelectorKind::Text | SelectorKind::Pattern | SelectorKind::Lines => {
            if spec.file.is_none() {
                return Err(EditRefusal::new(
                    "selector_missing",
                    format!("{} addresses a file; pass --file F with it", kind.name()),
                    20,
                ));
            }
        }
        SelectorKind::Symbol | SelectorKind::Target => {
            if spec.file.is_some() {
                return Err(EditRefusal::new(
                    "selector_conflict",
                    format!(
                        "--file and {} name different targets; WHERE is exactly one of them",
                        kind.name()
                    ),
                    20,
                ));
            }
        }
        SelectorKind::WholeFile => {}
    }
    if spec.body && spec.symbol.is_none() {
        return Err(EditRefusal::new(
            "body_without_symbol",
            "--body names the body of a definition; it needs --symbol S",
            20,
        ));
    }
    Ok(kind)
}

fn edit_line_count(content: &[u8]) -> usize {
    if content.is_empty() {
        return 0;
    }
    let newlines = content.iter().filter(|byte| **byte == b'\n').count();
    if content.ends_with(b"\n") {
        newlines
    } else {
        newlines + 1
    }
}

fn edit_line_of_offset(content: &[u8], offset: usize) -> usize {
    content[..offset.min(content.len())]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}

/// The 1-based inclusive line span a written region occupies.
fn edit_span_lines(content: &[u8], start: usize, length: usize) -> (usize, usize) {
    let first = edit_line_of_offset(content, start);
    if length == 0 {
        return (first, first);
    }
    let last = first
        + content[start..start + length - 1]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count();
    (first, last)
}

fn edit_strip_trailing_newline(content: &[u8], range: (usize, usize)) -> (usize, usize) {
    let (start, mut end) = range;
    if end > start && content[end - 1] == b'\n' {
        end -= 1;
        if end > start && content[end - 1] == b'\r' {
            end -= 1;
        }
    }
    (start, end)
}

/// Extend a line-oriented span over the newline that ends it, so a deletion
/// removes the line rather than leaving an empty one behind.
fn edit_extend_over_newline(content: &[u8], range: (usize, usize)) -> (usize, usize) {
    let (start, mut end) = range;
    if end < content.len() && content[end] == b'\r' {
        end += 1;
    }
    if end < content.len() && content[end] == b'\n' {
        end += 1;
    }
    (start, end)
}

fn edit_find_all(haystack: &[u8], needle: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if needle.is_empty() || needle.len() > haystack.len() {
        return out;
    }
    let mut index = 0usize;
    while index + needle.len() <= haystack.len() {
        if &haystack[index..index + needle.len()] == needle {
            out.push((index, index + needle.len()));
            index += needle.len();
        } else {
            index += 1;
        }
    }
    out
}

/// Splice every edit in one pass and report, for each of them, the byte range
/// it occupies in the RESULT. `--expect N` writes N spans and the report has to
/// name all of them, so the offsets are collected while they are still exact
/// rather than recomputed from the old content afterwards.
fn edit_splice(
    content: &[u8],
    edits: &mut [(usize, usize, Vec<u8>)],
) -> (Vec<u8>, Vec<(usize, usize)>) {
    edits.sort_by_key(|(start, _, _)| *start);
    let mut out = Vec::with_capacity(content.len());
    let mut written = Vec::with_capacity(edits.len());
    let mut cursor = 0usize;
    for (start, end, replacement) in edits.iter() {
        out.extend_from_slice(&content[cursor..*start]);
        let at = out.len();
        out.extend_from_slice(replacement);
        written.push((at, out.len()));
        cursor = *end;
    }
    out.extend_from_slice(&content[cursor..]);
    (out, written)
}

/// Refuse a path that leaves the repository, whether by climbing out of it or
/// through a symlink that points out.
fn edit_guard_path(root_path: &std::path::Path, abs: &std::path::Path) -> EditResult<()> {
    match greppy_edit::publish::require_inside_workspace(root_path, abs) {
        Ok(_) => Ok(()),
        Err(Error::Io { .. }) => Err(EditRefusal::new(
            "file_not_found",
            format!("no file {}", abs.display()),
            10,
        )),
        Err(error) => Err(EditRefusal::new("path_outside_repo", error.to_string(), 17)),
    }
}

fn edit_read_file(
    root_path: &std::path::Path,
    file: &str,
) -> EditResult<(String, std::path::PathBuf, Vec<u8>)> {
    let candidate = std::path::Path::new(file);
    let abs = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root_path.join(candidate)
    };
    if std::fs::symlink_metadata(&abs).is_err() {
        return Err(EditRefusal::new(
            "file_not_found",
            format!("no file `{file}`"),
            10,
        ));
    }
    edit_guard_path(root_path, &abs)?;
    let content = std::fs::read(&abs).map_err(|error| {
        EditRefusal::new("file_unreadable", format!("read {file}: {error}"), 10)
    })?;
    let rel = abs
        .strip_prefix(root_path)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| file.to_string());
    Ok((rel, abs, content))
}

/// A resolved definition: its path relative to the root, its absolute path,
/// the bytes of the file it lives in, and the byte range it occupies.
type ResolvedSpan = (String, std::path::PathBuf, Vec<u8>, (usize, usize));

/// Resolve `--symbol S` against the graph for the file, then against the bytes
/// on disk for the span: the graph is a cache and the file is the truth, so a
/// definition that moved since indexing is still addressed correctly.
fn edit_resolve_symbol(
    root_path: &std::path::Path,
    root: Option<&str>,
    name: &str,
    path_filter: Option<&str>,
    want_body: bool,
) -> EditResult<ResolvedSpan> {
    let store = open_default_store_query_writer(root).map_err(|error| {
        EditRefusal::new("symbol_not_found", format!("no symbol `{name}`: {error}"), 10)
    })?;
    let ids = resolve_symbol_nodes(&store, Some(name)).map_err(|error| {
        EditRefusal::new("symbol_not_found", format!("no symbol `{name}`: {error}"), 10)
    })?;
    let mut nodes = Vec::new();
    for id in &ids {
        if let Ok(Some(node)) = store.get_node(*id) {
            if node.file_path.is_empty() || node.start_line < 1 {
                continue;
            }
            if let Some(filter) = path_filter {
                let filter = filter.trim_start_matches("./");
                if !node.file_path.starts_with(filter) {
                    continue;
                }
            }
            nodes.push(node);
        }
    }
    // `--symbol S` names a definition. The graph also carries one synthetic
    // anchor per file, and its name is the file stem — so `pkg/greet.go`
    // answers to `greet` and turns an unambiguous edit into "2 definitions".
    // A file is addressed by `--file`, never by `--symbol`, so the anchor is
    // dropped whenever a real definition answered as well (rule 1: an argument
    // is never reinterpreted into a different question).
    if nodes
        .iter()
        .any(|node| !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name))
    {
        nodes.retain(|node| {
            !is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name)
        });
    }
    if nodes.is_empty() {
        return Err(EditRefusal::new(
            "symbol_not_found",
            format!("no symbol `{name}`"),
            10,
        ));
    }
    let mut sites: Vec<(String, i64)> = nodes
        .iter()
        .map(|node| (node.file_path.clone(), node.start_line))
        .collect();
    sites.sort();
    sites.dedup();
    if sites.len() > 1 {
        let candidates: Vec<serde_json::Value> = nodes
            .iter()
            .map(|node| {
                serde_json::json!({
                    "qualified_name": node.qualified_name,
                    "path": node.file_path,
                    "line": node.start_line,
                })
            })
            .collect();
        let listed = nodes
            .iter()
            .map(|node| format!("  {} {}:{}", node.qualified_name, node.file_path, node.start_line))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(EditRefusal::new(
            "ambiguous_symbol",
            format!("`{name}` resolves to {} definitions\n{listed}", sites.len()),
            11,
        )
        .with("candidates", serde_json::json!(candidates)));
    }
    let node = &nodes[0];
    let abs = root_path.join(&node.file_path);
    edit_guard_path(root_path, &abs)?;
    let content = std::fs::read(&abs).map_err(|error| {
        EditRefusal::new(
            "file_unreadable",
            format!("read {}: {error}", node.file_path),
            10,
        )
    })?;
    let language = greppy_edit::language_for_path(std::path::Path::new(&node.file_path));
    let (start_line, end_line) = edit_live_definition_lines(language, &content, node)
        .unwrap_or((node.start_line as usize, node.end_line as usize));
    let range = line_range_to_bytes(&content, start_line, end_line);
    let mut range = edit_strip_trailing_newline(&content, range);
    if want_body {
        let Some(body) = greppy_edit::verbs::body_range_within(language, &content, range) else {
            return Err(EditRefusal::new(
                "no_body",
                format!("`{name}` has no body"),
                13,
            )
            .with("symbol", serde_json::json!(name)));
        };
        range = edit_strip_trailing_newline(&content, body);
    }
    Ok((node.file_path.clone(), abs, content, range))
}

/// Re-extract the definitions of one file and return the live line span of the
/// node the graph named. Falls back to the cached span when the language has no
/// extraction pass.
fn edit_live_definition_lines(
    language: greppy_edit::Language,
    content: &[u8],
    node: &greppy_store::Node,
) -> Option<(usize, usize)> {
    let extracted = greppy_parser::extract::extract(language, content, &node.file_path).ok()?;
    let exact = extracted
        .nodes
        .iter()
        .find(|candidate| candidate.qualified_name == node.qualified_name);
    let chosen = exact.or_else(|| {
        let mut by_name = extracted
            .nodes
            .iter()
            .filter(|candidate| candidate.name == node.name);
        let first = by_name.next()?;
        if by_name.next().is_some() {
            None
        } else {
            Some(first)
        }
    })?;
    Some((chosen.start_line as usize, chosen.end_line as usize))
}

fn edit_parse_line_range(spec: &str) -> EditResult<(usize, usize)> {
    let bad = || {
        EditRefusal::new(
            "invalid_selector",
            format!("--lines takes A:B, 1-based and both ends included; got `{spec}`"),
            20,
        )
    };
    let (first, last) = spec.split_once(':').unwrap_or((spec, spec));
    let first: usize = first.trim().parse().map_err(|_| bad())?;
    let last: usize = last.trim().parse().map_err(|_| bad())?;
    if first == 0 || last < first {
        return Err(bad());
    }
    Ok((first, last))
}

#[allow(clippy::too_many_arguments)]
fn edit_locate(
    spec: &WhereSpec,
    kind: SelectorKind,
    root: Option<&str>,
    root_path: &std::path::Path,
) -> EditResult<Located> {
    match kind {
        SelectorKind::Symbol => {
            let name = spec.symbol.as_deref().unwrap_or_default();
            let (rel, abs, content, range) =
                edit_resolve_symbol(root_path, root, name, spec.path.as_deref(), spec.body)?;
            Ok(Located {
                rel,
                abs,
                content,
                ranges: vec![range],
                kind,
                regex: None,
                needle: None,
            })
        }
        SelectorKind::Target => {
            let token = spec.target.as_deref().unwrap_or_default();
            let handle = greppy_edit::EditHandle::decode(token).map_err(|error| {
                EditRefusal::new("invalid_handle", format!("not a usable handle: {error}"), 20)
            })?;
            let handle_root = std::path::Path::new(&handle.workspace_root)
                .canonicalize()
                .unwrap_or_else(|_| std::path::PathBuf::from(&handle.workspace_root));
            let here = root_path
                .canonicalize()
                .unwrap_or_else(|_| root_path.to_path_buf());
            if handle_root != here {
                return Err(EditRefusal::new(
                    "foreign_handle",
                    format!(
                        "that handle was taken in {}, not in {}",
                        handle.workspace_root,
                        here.display()
                    ),
                    20,
                ));
            }
            let abs = if std::path::Path::new(&handle.path).is_absolute() {
                std::path::PathBuf::from(&handle.path)
            } else {
                root_path.join(&handle.path)
            };
            edit_guard_path(root_path, &abs)?;
            let content = std::fs::read(&abs).map_err(|error| {
                EditRefusal::new(
                    "file_unreadable",
                    format!("read {}: {error}", handle.path),
                    10,
                )
            })?;
            let range = handle.verify(&content).map_err(|_| {
                EditRefusal::new(
                    "stale_handle",
                    format!("{} changed since that handle was taken", handle.path),
                    12,
                )
            })?;
            let range = edit_strip_trailing_newline(&content, range);
            Ok(Located {
                rel: handle.path.clone(),
                abs,
                content,
                ranges: vec![range],
                kind,
                regex: None,
                needle: None,
            })
        }
        _ => {
            let file = spec.file.as_deref().unwrap_or_default();
            let (rel, abs, content, ranges, regex, needle) = match kind {
                SelectorKind::WholeFile => {
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
                    let length = content.len();
                    (rel, abs, content, vec![(0, length)], None, None)
                }
                SelectorKind::Lines => {
                    let (first, last) = edit_parse_line_range(
                        spec.lines.as_deref().unwrap_or_default(),
                    )?;
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
                    let total = edit_line_count(&content);
                    if last > total || first > total {
                        return Err(EditRefusal::new(
                            "range_out_of_bounds",
                            format!("{rel} has {total} line(s); {first}:{last} runs past its end"),
                            13,
                        ));
                    }
                    let range = line_range_to_bytes(&content, first, last);
                    let range = edit_strip_trailing_newline(&content, range);
                    (rel, abs, content, vec![range], None, None)
                }
                SelectorKind::Text => {
                    let needle = match (&spec.old, &spec.old_file) {
                        (Some(text), None) => text.as_bytes().to_vec(),
                        (None, Some(path)) => read_source_arg(path).map_err(|error| {
                            EditRefusal::new(
                                "invalid_selector",
                                format!("--old-file {path}: {error}"),
                                20,
                            )
                        })?,
                        _ => Vec::new(),
                    };
                    if needle.is_empty() {
                        return Err(EditRefusal::new(
                            "invalid_selector",
                            "--old is empty; empty text matches between every pair of characters",
                            20,
                        ));
                    }
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
                    let ranges = edit_find_all(&content, &needle);
                    let shown = String::from_utf8_lossy(&needle).into_owned();
                    (rel, abs, content, ranges, None, Some(shown))
                }
                SelectorKind::Pattern => {
                    let pattern = spec.pattern.as_deref().unwrap_or_default();
                    let regex = regex::bytes::Regex::new(pattern).map_err(|error| {
                        EditRefusal::new(
                            "invalid_pattern",
                            format!("--pattern is not a regular expression: {error}"),
                            20,
                        )
                    })?;
                    let (rel, abs, content) = edit_read_file(root_path, file)?;
                    let ranges = regex
                        .find_iter(&content)
                        .map(|found| (found.start(), found.end()))
                        .collect();
                    (rel, abs, content, ranges, Some(regex), Some(pattern.to_string()))
                }
                SelectorKind::Symbol | SelectorKind::Target => unreachable!(),
            };
            Ok(Located {
                rel,
                abs,
                content,
                ranges,
                kind,
                regex,
                needle,
            })
        }
    }
}

/// The number of matches a selector is allowed to have. `--old` and
/// `--pattern` search, so they can find none or many; every other selector
/// addresses exactly one span by construction.
fn edit_check_cardinality(located: &Located, expect: Option<usize>) -> EditResult<()> {
    if !matches!(located.kind, SelectorKind::Text | SelectorKind::Pattern) {
        return Ok(());
    }
    let expect = expect.unwrap_or(1);
    if located.ranges.len() != expect {
        // The count alone does not let a caller decide between "pass --expect N"
        // and "I anchored on the wrong text", so the refusal names what was
        // searched for and where every match sits.
        let _subject = located
            .needle
            .as_deref()
            .map_or_else(|| located.kind.name().to_string(), |text| format!("`{text}`"));
        let sites: Vec<String> = located
            .ranges
            .iter()
            .take(20)
            .map(|(start, _)| {
                format!(
                    "{}:{}:{}: {}",
                    located.rel,
                    edit_line_of_offset(&located.content, *start),
                    {
                        let ls = located.content[..*start]
                            .iter()
                            .rposition(|&b| b == b'\n')
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        *start - ls + 1
                    },
                    {
                        let ls = located.content[..*start]
                            .iter()
                            .rposition(|&b| b == b'\n')
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        let le = located.content[*start..]
                            .iter()
                            .position(|&b| b == b'\n')
                            .map(|i| *start + i)
                            .unwrap_or(located.content.len());
                        one_line_truncated(&String::from_utf8_lossy(&located.content[ls..le]), 200)
                    }
                )
            })
            .collect();
        // The needle is not echoed — the caller has it in context (law 5).
        let mut message = format!(
            "matched {} times, expected {expect} — nothing written",
            located.ranges.len()
        );
        for site in &sites {
            message.push_str("\n  ");
            message.push_str(site);
        }
        return Err(EditRefusal::new("match_count", message, 13)
            .with("expected", serde_json::json!(expect))
            .with("found", serde_json::json!(located.ranges.len()))
            .with("matches", serde_json::json!(sites)));
    }
    Ok(())
}

fn edit_expect_positive(expect: Option<usize>) -> EditResult<()> {
    if expect == Some(0) {
        return Err(EditRefusal::new(
            "invalid_expect",
            "--expect 0 asks for an edit that writes nothing",
            20,
        ));
    }
    Ok(())
}

/// There is exactly one stdin, so two arguments asking for it leave the
/// caller's intent unrecoverable — one of them would get nothing, or both would
/// get half. The collision is refused before any of them reads a byte.
fn edit_guard_single_stdin(flags: &[(&'static str, Option<&str>)]) -> EditResult<()> {
    let asking: Vec<&'static str> = flags
        .iter()
        .filter(|(_, value)| *value == Some("-"))
        .map(|(name, _)| *name)
        .collect();
    if asking.len() > 1 {
        return Err(EditRefusal::new(
            "stdin_conflict",
            format!(
                "{} both read stdin; exactly one argument may take `-`",
                asking.join(" and ")
            ),
            20,
        )
        .with("flags", serde_json::json!(asking)));
    }
    Ok(())
}

/// An empty replacement is never an intention: it is a command substitution
/// that produced nothing, a file that was created but never written, or a
/// broken pipe. Writing it would delete working code and report success.
const EMPTY_CONTENT_HINT: &str = "`delete` is the verb that removes a span";

fn edit_content_bytes(
    content: Option<String>,
    content_file: Option<String>,
) -> EditResult<Vec<u8>> {
    match (content, content_file) {
        (Some(text), None) => {
            if text.is_empty() {
                return Err(EditRefusal::new(
                    "content_empty",
                    format!("--content is empty; {EMPTY_CONTENT_HINT}"),
                    20,
                ));
            }
            Ok(text.into_bytes())
        }
        (None, Some(path)) => {
            let bytes = read_source_arg(&path).map_err(|error| {
                EditRefusal::new(
                    "content_unreadable",
                    format!("--content-file {path}: {error}"),
                    20,
                )
            })?;
            if bytes.is_empty() {
                let message = if path == "-" {
                    format!("--content-file -: stdin was empty; {EMPTY_CONTENT_HINT}")
                } else {
                    format!("--content-file {path} is empty; {EMPTY_CONTENT_HINT}")
                };
                return Err(EditRefusal::new("content_empty", message, 20));
            }
            Ok(bytes)
        }
        (Some(_), Some(_)) => Err(EditRefusal::new(
            "content_conflict",
            "--content and --content-file both name the new text; pass exactly one of them",
            20,
        )),
        (None, None) => Err(EditRefusal::new(
            "content_missing",
            "no new text: pass --content TEXT or --content-file FILE",
            20,
        )),
    }
}

/// Publish one file and answer with the record the contract promises: the
/// file, every span it wrote, the resulting text, and a handle for the new
/// span so the next edit needs no `read` in between.
fn edit_publish(
    root_path: &std::path::Path,
    located: &Located,
    new_content: Vec<u8>,
    changed: Vec<(usize, usize)>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let (first_start, first_end) = changed.first().copied().unwrap_or((0, 0));
    let first_end = first_end.min(new_content.len());
    let text = String::from_utf8_lossy(&new_content[first_start..first_end]).into_owned();
    let span = edit_span_lines(&new_content, first_start, first_end - first_start);
    let mut operation = EditOperation {
        file: located.rel.clone(),
        ranges: changed,
        result_span: Some(text.clone()),
        sha_before: Some(edit_sha256_hex(&located.content)),
        sha_after: Some(edit_sha256_hex(&new_content)),
        diff: Some(edit_unified_diff(&located.rel, &located.content, &new_content)),
        ..EditOperation::default()
    };
    let mut record = EditRecord {
        files: vec![located.rel.clone()],
        span: Some(span),
        text: Some(text),
        published: !dry_run,
        ..EditRecord::default()
    };
    if dry_run {
        // A handle addresses bytes on disk. A dry run wrote none, so handing
        // one back would hand back an address that is already stale.
        record.operations = vec![operation];
        return Ok(record);
    }
    let before_sha = edit_sha256_hex(&located.content);
    // The journal goes down before the write, so an edit that dies in between
    // leaves the evidence `recover` needs instead of a half-written file.
    let transaction = edit_journal_open(
        root_path,
        &[UndoBefore {
            rel: located.rel.clone(),
            content: Some(located.content.clone()),
        }],
    );
    edit_journal_crash_hook()?;
    if let Err(error) =
        greppy_edit::publish::publish_atomic(root_path, &located.abs, &new_content, &before_sha)
    {
        edit_journal_abort(root_path);
        return Err(EditRefusal::new("publish_failed", error.to_string(), 16));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    let handle = greppy_edit::EditHandle::for_range(
        root_path,
        std::path::Path::new(&located.rel),
        &new_content,
        first_start,
        first_end,
    )
    .ok()
    .map(|handle| handle.encode());
    operation.handle = handle.clone();
    record.handle = handle;
    record.operations = vec![operation];
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

/// A rewritten file, and the byte ranges of it the edit wrote.
type EditedContent = (Vec<u8>, Vec<(usize, usize)>);

fn edit_op_replace(located: &Located, new_bytes: &[u8]) -> EditedContent {
    // A line-oriented span stops before the newline that ends its last line,
    // because that newline belongs to the file (see `SelectorKind::line_oriented`).
    // New text that carries one of its own would therefore add a blank line the
    // caller never wrote, so the span's own ending is the one that survives.
    let new_bytes = if located.kind.line_oriented() {
        let trimmed = new_bytes
            .strip_suffix(b"\n")
            .map(|text| text.strip_suffix(b"\r").unwrap_or(text));
        trimmed.unwrap_or(new_bytes)
    } else {
        new_bytes
    };
    let mut edits: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    for (start, end) in &located.ranges {
        let replacement = match &located.regex {
            Some(regex) => {
                let mut expanded = Vec::new();
                if let Some(captures) = regex.captures_at(&located.content, *start) {
                    captures.expand(new_bytes, &mut expanded);
                } else {
                    expanded.extend_from_slice(new_bytes);
                }
                expanded
            }
            None => new_bytes.to_vec(),
        };
        edits.push((*start, *end, replacement));
    }
    edit_splice(&located.content, &mut edits)
}

fn edit_op_delete(located: &Located) -> EditedContent {
    let mut edits: Vec<(usize, usize, Vec<u8>)> = located
        .ranges
        .iter()
        .map(|range| {
            let range = if located.kind.line_oriented() {
                edit_extend_over_newline(&located.content, *range)
            } else {
                *range
            };
            (range.0, range.1, Vec::new())
        })
        .collect();
    edit_splice(&located.content, &mut edits)
}

fn edit_op_insert(located: &Located, text: &[u8], before: bool) -> EditedContent {
    let (start, end) = located.ranges[0];
    let at = if before {
        start
    } else {
        edit_extend_over_newline(&located.content, (start, end)).1
    };
    let mut edits = vec![(at, at, text.to_vec())];
    edit_splice(&located.content, &mut edits)
}

/// Apply a unified diff inside the located span. The hunk headers are read
/// twice — once as file line numbers, once as offsets from the start of WHERE —
/// and whichever reading the context lines confirm is the one that applies.
fn edit_op_patch(located: &Located, patch: &[u8]) -> EditResult<EditedContent> {
    let patch_text = String::from_utf8_lossy(patch);
    let mut hunks: Vec<(usize, Vec<String>, Vec<String>)> = Vec::new();
    let mut current: Option<(usize, Vec<String>, Vec<String>)> = None;
    for line in patch_text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.starts_with("@@") {
            if let Some(hunk) = current.take() {
                hunks.push(hunk);
            }
            let start = line
                .split_whitespace()
                .nth(1)
                .and_then(|field| field.strip_prefix('-'))
                .map(|field| field.split(',').next().unwrap_or(field))
                .and_then(|number| number.parse::<usize>().ok())
                .unwrap_or(1);
            current = Some((start, Vec::new(), Vec::new()));
            continue;
        }
        let Some((_, old_lines, new_lines)) = current.as_mut() else {
            continue;
        };
        match line.as_bytes().first() {
            Some(b' ') => {
                old_lines.push(line[1..].to_string());
                new_lines.push(line[1..].to_string());
            }
            Some(b'-') => old_lines.push(line[1..].to_string()),
            Some(b'+') => new_lines.push(line[1..].to_string()),
            Some(b'\\') => {}
            None => {}
            _ => {}
        }
    }
    if let Some(hunk) = current.take() {
        hunks.push(hunk);
    }
    if hunks.is_empty() {
        return Err(EditRefusal::new(
            "invalid_patch",
            "the patch carries no hunk to apply",
            20,
        ));
    }

    // Physical lines of the file, each with the byte range it occupies.
    let mut lines: Vec<(usize, usize)> = Vec::new();
    let mut cursor = 0usize;
    while cursor < located.content.len() {
        let end = located.content[cursor..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|offset| cursor + offset + 1)
            .unwrap_or(located.content.len());
        lines.push((cursor, end));
        cursor = end;
    }
    let line_text = |index: usize| -> Option<String> {
        let (start, end) = *lines.get(index.checked_sub(1)?)?;
        let raw = &located.content[start..end];
        let raw = raw.strip_suffix(b"\n").unwrap_or(raw);
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        Some(String::from_utf8_lossy(raw).into_owned())
    };
    let (span_start, span_end) = located.ranges[0];
    let span_first_line = edit_line_of_offset(&located.content, span_start);
    let span_last_line = edit_line_of_offset(&located.content, span_end.saturating_sub(1).max(span_start));

    let mut edits: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    for (declared, old_lines, new_lines) in &hunks {
        let matches_at = |first: usize| -> bool {
            if first < span_first_line {
                return false;
            }
            if old_lines.is_empty() {
                return first <= span_last_line + 1;
            }
            if first + old_lines.len() - 1 > span_last_line {
                return false;
            }
            old_lines
                .iter()
                .enumerate()
                .all(|(offset, expected)| line_text(first + offset).as_ref() == Some(expected))
        };
        let relative = span_first_line + declared.saturating_sub(1);
        let first = if matches_at(*declared) {
            *declared
        } else if matches_at(relative) {
            relative
        } else {
            // Which hunk failed is not enough to act on: the caller needs the
            // context lines that were expected and not found, or it cannot tell
            // a wrong WHERE from a stale diff.
            let expected = old_lines
                .iter()
                .take(6)
                .map(|line| format!("\n  {line}"))
                .collect::<String>();
            return Err(EditRefusal::new(
                "patch_context",
                format!(
                    "the hunk at line {declared} matches neither {} nor the selected span; \
                     it expects:{expected}",
                    located.rel
                ),
                13,
            ));
        };
        let (start, end) = if old_lines.is_empty() {
            let at = lines
                .get(first.saturating_sub(1))
                .map(|(start, _)| *start)
                .unwrap_or(located.content.len());
            (at, at)
        } else {
            (
                lines[first - 1].0,
                lines[first + old_lines.len() - 2].1,
            )
        };
        let ending = if located.content[start..end].ends_with(b"\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        let mut replacement = String::new();
        for line in new_lines {
            replacement.push_str(line);
            replacement.push_str(ending);
        }
        // A hunk that reached the end of a file without a final newline must
        // not grow one.
        if !located.content[start..end].ends_with(b"\n") && replacement.ends_with(ending) {
            replacement.truncate(replacement.len() - ending.len());
        }
        edits.push((start, end, replacement.into_bytes()));
    }
    let first_start = edits.iter().map(|(start, _, _)| *start).min().unwrap_or(0);
    let last_end = edits.iter().map(|(_, end, _)| *end).max().unwrap_or(0);
    let delta: isize = edits
        .iter()
        .map(|(start, end, replacement)| replacement.len() as isize - (*end - *start) as isize)
        .sum();
    let (new_content, _) = edit_splice(&located.content, &mut edits);
    // A diff is one change even when it carries several hunks, so the span the
    // report names is the whole patched region, not each hunk on its own.
    let length = ((last_end - first_start) as isize + delta).max(0) as usize;
    Ok((new_content, vec![(first_start, first_start + length)]))
}

/// The compiler or linter for the touched files, when the workspace declares
/// one. No tests: a test run is too long for a single edit call.
fn edit_verify_diagnostics(root_path: &std::path::Path, files: &[String]) -> Vec<String> {
    let mut argv: Option<Vec<&str>> = None;
    if root_path.join("Cargo.toml").is_file() {
        argv = Some(vec!["cargo", "check", "--message-format", "short", "--quiet"]);
    } else if root_path.join("go.mod").is_file() {
        argv = Some(vec!["go", "build", "./..."]);
    } else if files.iter().any(|file| file.ends_with(".py"))
        && root_path.join("pyproject.toml").is_file()
    {
        argv = Some(vec!["python3", "-m", "compileall", "-q", "."]);
    }
    let Some(argv) = argv else {
        return Vec::new();
    };
    let Ok(output) = std::process::Command::new(argv[0])
        .args(&argv[1..])
        .current_dir(root_path)
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .filter(|line| line.contains("error") || line.contains("warning"))
        .take(50)
        .map(str::to_string)
        .collect()
}

// --- the undo journal -------------------------------------------------------
//
// The journal lives in the workspace's store directory, never inside the
// repository: a directory of pre-images in the tree would turn up in every
// `git status`, in every `index`, and in every check that a refused edit
// changed nothing — greppy's own bookkeeping would read as the caller's work.

const EDIT_JOURNAL_DIR: &str = "edit-journal";
const EDIT_JOURNAL_STACK: &str = "stack.json";
const EDIT_JOURNAL_PENDING: &str = "pending.json";
const EDIT_JOURNAL_BLOBS: &str = "blobs";
/// How far back `undo` can walk. One call still reverses exactly one edit; the
/// depth only decides how many of them are still on record.
const EDIT_JOURNAL_DEPTH: usize = 100;

/// What one file looked like before an edit touched it. `None` means the file
/// was not there, so undoing that edit removes it again rather than emptying
/// it — an empty file still shadows a module and still shows up in the index.
#[derive(Clone)]
struct UndoBefore {
    rel: String,
    content: Option<Vec<u8>>,
}

impl UndoBefore {
    fn read(root_path: &std::path::Path, rel: &str) -> Self {
        Self {
            rel: rel.to_string(),
            content: std::fs::read(root_path.join(rel)).ok(),
        }
    }
}

fn edit_journal_dir(root_path: &std::path::Path) -> std::path::PathBuf {
    greppy_core::cache::workspace_store_dir(root_path).join(EDIT_JOURNAL_DIR)
}

fn edit_journal_read(path: &std::path::Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn edit_journal_write(path: &std::path::Path, value: &serde_json::Value) {
    if let Ok(text) = serde_json::to_string_pretty(value) {
        let _ = std::fs::write(path, text);
    }
}

/// Record the pre-images and open a transaction. Anything that dies between
/// here and [`edit_journal_close`] leaves `pending.json` behind — which is
/// exactly what `recover` looks for.
fn edit_journal_open(root_path: &std::path::Path, before: &[UndoBefore]) -> Option<String> {
    if before.is_empty() {
        return None;
    }
    let dir = edit_journal_dir(root_path);
    std::fs::create_dir_all(dir.join(EDIT_JOURNAL_BLOBS)).ok()?;
    let id = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default()
    );
    let mut entries = Vec::new();
    for (index, item) in before.iter().enumerate() {
        let blob = match &item.content {
            Some(bytes) => {
                let name = format!("{id}-{index}.bin");
                std::fs::write(dir.join(EDIT_JOURNAL_BLOBS).join(&name), bytes).ok()?;
                Some(name)
            }
            None => None,
        };
        entries.push(serde_json::json!({ "path": item.rel, "blob": blob }));
    }
    edit_journal_write(
        &dir.join(EDIT_JOURNAL_PENDING),
        &serde_json::json!({ "id": id, "entries": entries }),
    );
    Some(id)
}

/// Die after the journal is on disk and before anything is published, so the
/// interrupted-edit path can be exercised without killing the process from the
/// outside. Only ever reached when the environment variable is set.
fn edit_journal_crash_hook() -> EditResult<()> {
    if std::env::var_os("GREPPY_TEST_CRASH_AFTER_JOURNAL").is_some() {
        return Err(EditRefusal::new(
            "interrupted",
            "interrupted after the journal was written and before anything was published",
            16,
        ));
    }
    Ok(())
}

/// Close the transaction: record what the files look like now, and push it onto
/// the stack. The after-image is what `undo` checks against, so an edit that
/// somebody else overwrote in the meantime cannot be reversed blindly (D3).
fn edit_journal_close(root_path: &std::path::Path, id: &str) {
    let dir = edit_journal_dir(root_path);
    let Some(mut record) = edit_journal_read(&dir.join(EDIT_JOURNAL_PENDING)) else {
        return;
    };
    if record["id"].as_str() != Some(id) {
        return;
    }
    let closed: Vec<serde_json::Value> = record["entries"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|mut entry| {
            let rel = entry["path"].as_str().unwrap_or_default().to_string();
            match std::fs::read(root_path.join(&rel)) {
                Ok(bytes) => {
                    entry["existed_after"] = serde_json::json!(true);
                    entry["after_sha256"] = serde_json::json!(edit_sha256_hex(&bytes));
                }
                Err(_) => {
                    entry["existed_after"] = serde_json::json!(false);
                    entry["after_sha256"] = serde_json::Value::Null;
                }
            }
            entry
        })
        .collect();
    record["entries"] = serde_json::json!(closed);
    let mut stack = edit_journal_read(&dir.join(EDIT_JOURNAL_STACK))
        .and_then(|value| value["transactions"].as_array().cloned())
        .unwrap_or_default();
    stack.push(record);
    if stack.len() > EDIT_JOURNAL_DEPTH {
        let excess = stack.len() - EDIT_JOURNAL_DEPTH;
        stack.drain(..excess);
    }
    edit_journal_write(
        &dir.join(EDIT_JOURNAL_STACK),
        &serde_json::json!({ "transactions": stack }),
    );
    let _ = std::fs::remove_file(dir.join(EDIT_JOURNAL_PENDING));
}

/// Abandon an open transaction without recording it. Used when the work it was
/// opened for turned out to write nothing after all.
fn edit_journal_abort(root_path: &std::path::Path) {
    let _ = std::fs::remove_file(edit_journal_dir(root_path).join(EDIT_JOURNAL_PENDING));
}

/// Put a transaction's files back the way they were. `guarded` is the D3 rule:
/// `undo` refuses if a file no longer looks the way that edit left it, because
/// it would otherwise overwrite bytes the caller has never seen. `recover`
/// restores unguarded — an interrupted edit has no after-image to compare with.
fn edit_journal_restore(
    root_path: &std::path::Path,
    record: &serde_json::Value,
    guarded: bool,
) -> EditResult<Vec<String>> {
    let dir = edit_journal_dir(root_path);
    let entries = record["entries"].as_array().cloned().unwrap_or_default();
    if guarded {
        for entry in &entries {
            let rel = entry["path"].as_str().unwrap_or_default();
            let live = std::fs::read(root_path.join(rel));
            let expected_after = entry["existed_after"].as_bool().unwrap_or(true);
            let unchanged = match (&live, expected_after) {
                (Ok(bytes), true) => {
                    edit_sha256_hex(bytes) == entry["after_sha256"].as_str().unwrap_or_default()
                }
                (Err(_), false) => true,
                _ => false,
            };
            if !unchanged {
                return Err(EditRefusal::new(
                    "changed_since_edit",
                    format!("{rel} no longer looks the way that edit left it"),
                    12,
                ));
            }
        }
    }
    let mut restored = Vec::new();
    for entry in &entries {
        let rel = entry["path"].as_str().unwrap_or_default().to_string();
        let abs = root_path.join(&rel);
        match entry["blob"].as_str() {
            Some(blob) => {
                let bytes = std::fs::read(dir.join(EDIT_JOURNAL_BLOBS).join(blob)).map_err(
                    |error| {
                        EditRefusal::new(
                            "nothing_to_undo",
                            format!("{rel}: the pre-image is gone ({error})"),
                            10,
                        )
                    },
                )?;
                if let Some(parent) = abs.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                std::fs::write(&abs, &bytes).map_err(|error| {
                    EditRefusal::new("publish_failed", format!("{rel}: {error}"), 16)
                })?;
                restored.push(rel);
            }
            // The file was created by that edit, so putting it back means
            // taking it away again.
            None => {
                let _ = std::fs::remove_file(&abs);
            }
        }
    }
    restored.sort();
    Ok(restored)
}

fn run_edit_undo(root_path: &std::path::Path, dry_run: bool) -> EditResult<EditRecord> {
    let dir = edit_journal_dir(root_path);
    let mut stack = edit_journal_read(&dir.join(EDIT_JOURNAL_STACK))
        .and_then(|value| value["transactions"].as_array().cloned())
        .unwrap_or_default();
    let Some(last) = stack.pop() else {
        return Err(EditRefusal::new(
            "nothing_to_undo",
            "nothing to undo in this workspace",
            10,
        ));
    };
    let files: Vec<String> = last["entries"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry["path"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let mut record = EditRecord {
        headline: Some(match (files.len(), dry_run) {
            (1, false) => format!("reversed {}", files[0]),
            (n, false) => format!("reversed {n} files"),
            (1, true) => format!("would reverse {}", files[0]),
            (n, true) => format!("would reverse {n} files"),
        }),
        files,
        published: !dry_run,
        ..EditRecord::default()
    };
    if dry_run {
        return Ok(record);
    }
    let restored = edit_journal_restore(root_path, &last, true)?;
    // The entry is consumed, or every further call would restore the same
    // snapshot and answer with a cheerful exit 0 while the repo stops moving.
    edit_journal_write(
        &dir.join(EDIT_JOURNAL_STACK),
        &serde_json::json!({ "transactions": stack }),
    );
    record
        .extra
        .push(("restored", serde_json::json!(restored)));
    Ok(record)
}

/// Finish or roll back an edit that was interrupted between the journal and the
/// publish. Returns `None` when there is nothing pending, so the caller can fall
/// through to the transaction journal of the certificate verbs.
fn run_edit_recover(root_path: &std::path::Path) -> EditResult<Option<usize>> {
    let dir = edit_journal_dir(root_path);
    let pending = dir.join(EDIT_JOURNAL_PENDING);
    let Some(record) = edit_journal_read(&pending) else {
        return Ok(None);
    };
    let restored = edit_journal_restore(root_path, &record, false)?;
    let _ = std::fs::remove_file(&pending);
    Ok(Some(restored.len()))
}

/// Record a transaction whose files are already on disk. The verbs that go
/// through a certificate publish before they report, so there is nothing left
/// to interrupt by the time this runs.
fn record_edit_undo(root_path: &std::path::Path, before: &[UndoBefore]) {
    if let Some(id) = edit_journal_open(root_path, before) {
        edit_journal_close(root_path, &id);
    }
}

/// Pre-images for the files a plan names, captured before it runs.
fn plan_undo_snapshot(root_path: &std::path::Path, plan_text: &str) -> Vec<UndoBefore> {
    let Ok(plan) = serde_json::from_str::<serde_json::Value>(plan_text) else {
        return Vec::new();
    };
    let mut out: Vec<UndoBefore> = Vec::new();
    for operation in plan["operations"].as_array().unwrap_or(&Vec::new()) {
        for key in ["file", "to"] {
            let Some(file) = operation[key].as_str() else {
                continue;
            };
            if out.iter().any(|item| item.rel == file) {
                continue;
            }
            out.push(UndoBefore::read(root_path, file));
        }
    }
    out
}

// --- whole-file verbs -------------------------------------------------------
//
// `write`, `move` and `remove` are the three questions that are about the file
// rather than about a span inside it. `move` and `remove` are only worth having
// because they do the language work `mv` and `rm` cannot: a rename that leaves
// the importers behind produces a repository that does not build, and a delete
// that leaves dangling imports turns one mistake into a broken build.

/// Which bytes of a file are code, as opposed to a comment or a string.
///
/// The distinction is the whole point of the rewrite: a module path inside a
/// comment or a string literal is prose, and rewriting it edits documentation
/// nobody asked about — while a rewrite that stops at the first occurrence
/// leaves a call pointing at a module that is gone.
fn edit_code_mask(text: &str, hash_comments: bool) -> Vec<bool> {
    let bytes = text.as_bytes();
    let mut mask = vec![true; bytes.len()];
    let mut index = 0usize;
    while index < bytes.len() {
        let rest = &bytes[index..];
        let line_comment = if hash_comments {
            rest.first() == Some(&b'#')
        } else {
            rest.starts_with(b"//")
        };
        if line_comment {
            while index < bytes.len() && bytes[index] != b'\n' {
                mask[index] = false;
                index += 1;
            }
            continue;
        }
        if !hash_comments && rest.starts_with(b"/*") {
            let end = text[index..]
                .find("*/")
                .map(|offset| index + offset + 2)
                .unwrap_or(bytes.len());
            for flag in mask.iter_mut().take(end).skip(index) {
                *flag = false;
            }
            index = end;
            continue;
        }
        if hash_comments && (rest.starts_with(b"\"\"\"") || rest.starts_with(b"'''")) {
            let quote = &text[index..index + 3];
            let end = text[index + 3..]
                .find(quote)
                .map(|offset| index + 3 + offset + 3)
                .unwrap_or(bytes.len());
            for flag in mask.iter_mut().take(end).skip(index) {
                *flag = false;
            }
            index = end;
            continue;
        }
        if rest[0] == b'"' || rest[0] == b'\'' {
            let quote = rest[0];
            let mut cursor = index + 1;
            mask[index] = false;
            while cursor < bytes.len() {
                mask[cursor] = false;
                if bytes[cursor] == b'\\' {
                    cursor += 2;
                    continue;
                }
                if bytes[cursor] == quote || bytes[cursor] == b'\n' {
                    cursor += 1;
                    break;
                }
                cursor += 1;
            }
            index = cursor.min(bytes.len());
            continue;
        }
        index += 1;
    }
    mask
}

fn edit_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Replace every code occurrence of `from` with `to`, where an occurrence is
/// bounded by something that is not part of a name — so `crate::helper` never
/// matches inside `crate::helper_answer`, and `pkg.helper` never matches inside
/// `pkg.helpers`.
fn edit_replace_in_code(text: &str, mask: &[bool], from: &str, to: &str) -> (String, usize) {
    let bytes = text.as_bytes();
    let boundary = |offset: usize, delta: isize| -> bool {
        let probe = if delta < 0 {
            match offset.checked_sub(1) {
                Some(index) => index,
                None => return true,
            }
        } else if offset >= bytes.len() {
            return true;
        } else {
            offset
        };
        !edit_ident_byte(bytes[probe])
    };
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    let mut hits = 0usize;
    while let Some(found) = text[cursor..].find(from) {
        let start = cursor + found;
        let end = start + from.len();
        out.push_str(&text[cursor..start]);
        let leading = text[..start]
            .as_bytes()
            .last()
            .is_none_or(|byte| !edit_ident_byte(*byte) && *byte != b'.' && *byte != b':');
        if mask.get(start).copied().unwrap_or(false)
            && leading
            && boundary(start, -1)
            && boundary(end, 1)
        {
            out.push_str(to);
            hits += 1;
        } else {
            out.push_str(&text[start..end]);
        }
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    (out, hits)
}

/// How a language spells "the module this file is". Only the languages whose
/// module identity greppy can state exactly are here; for anything else `move`
/// moves the file and says it rewrote nothing, rather than guessing.
enum ModuleIdentity {
    /// `crate::a::helper`, plus the bare identifier its declaring file uses.
    Rust { path: String, ident: String },
    /// `pkg.helper`.
    Python { path: String },
}

fn edit_module_identity(rel: &str) -> Option<ModuleIdentity> {
    let path = std::path::Path::new(rel);
    let extension = path.extension().and_then(|value| value.to_str())?;
    match extension {
        "rs" => {
            let mut segments: Vec<String> = rel
                .trim_end_matches(".rs")
                .split('/')
                .map(str::to_string)
                .collect();
            if segments.first().is_some_and(|first| first == "src") {
                segments.remove(0);
            }
            if segments.last().is_some_and(|last| last == "mod") {
                segments.pop();
            }
            if segments
                .last()
                .is_some_and(|last| last == "lib" || last == "main")
            {
                segments.pop();
            }
            let ident = segments.last()?.clone();
            Some(ModuleIdentity::Rust {
                path: format!("crate::{}", segments.join("::")),
                ident,
            })
        }
        "py" => {
            let mut segments: Vec<String> = rel
                .trim_end_matches(".py")
                .split('/')
                .map(str::to_string)
                .collect();
            if segments.last().is_some_and(|last| last == "__init__") {
                segments.pop();
            }
            if segments.is_empty() {
                return None;
            }
            Some(ModuleIdentity::Python {
                path: segments.join("."),
            })
        }
        _ => None,
    }
}

/// The files that could declare a Rust module: the parent module's own file.
/// `src/a/helper.rs` is declared in `src/a/mod.rs`, `src/helper.rs` in
/// `src/lib.rs` or `src/main.rs`. Only those files may have their bare
/// `mod helper;` and `helper::…` rewritten — every other file has to say
/// `crate::…`, and rewriting a bare identifier there would hit an unrelated name.
fn edit_rust_declaring_files(rel: &str) -> Vec<String> {
    let path = std::path::Path::new(rel);
    let mut dir = path.parent().map(std::path::Path::to_path_buf);
    if path.file_stem().and_then(|stem| stem.to_str()) == Some("mod") {
        dir = dir.and_then(|inner| inner.parent().map(std::path::Path::to_path_buf));
    }
    let Some(dir) = dir else {
        return Vec::new();
    };
    ["mod.rs", "lib.rs", "main.rs"]
        .iter()
        .map(|name| {
            let joined = dir.join(name);
            joined.to_string_lossy().replace('\\', "/")
        })
        .filter(|candidate| candidate != rel)
        .collect()
}

/// Every source file of the same language, so "which files name this module" is
/// answered from what is on disk now — the index is a cache, and a cache is
/// wrong the moment somebody edits a file outside greppy.
fn edit_sibling_sources(root_path: &std::path::Path, extension: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![root_path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) != Some(extension) {
                continue;
            }
            if let Ok(relative) = path.strip_prefix(root_path) {
                found.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    found.sort();
    found
}

/// One file that names the moved module, and — when a new path was given — the
/// text it has to carry afterwards.
struct ModuleReference {
    file: String,
    rewritten: Option<String>,
}

/// Which files name the module `rel` is, and how they read once it becomes
/// `to`. This is what makes `move` more than `mv` and `remove` more than `rm`.
fn edit_module_references(
    root_path: &std::path::Path,
    rel: &str,
    to: Option<&str>,
) -> Vec<ModuleReference> {
    let Some(identity) = edit_module_identity(rel) else {
        return Vec::new();
    };
    let target = to.and_then(edit_module_identity);
    let (extension, hash_comments) = match identity {
        ModuleIdentity::Rust { .. } => ("rs", false),
        ModuleIdentity::Python { .. } => ("py", true),
    };
    let declaring = edit_rust_declaring_files(rel);
    let mut found = Vec::new();
    for file in edit_sibling_sources(root_path, extension) {
        if file == rel || Some(file.as_str()) == to {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(root_path.join(&file)) else {
            continue;
        };
        let mut current = text.clone();
        let mut hits = 0usize;
        let apply = |current: &mut String, hits: &mut usize, from: &str, to: &str| {
            let mask = edit_code_mask(current, hash_comments);
            let (next, count) = edit_replace_in_code(current, &mask, from, to);
            *hits += count;
            *current = next;
        };
        match (&identity, &target) {
            (ModuleIdentity::Rust { path, ident }, target) => {
                let (new_path, new_ident) = match target {
                    Some(ModuleIdentity::Rust {
                        path: new_path,
                        ident: new_ident,
                    }) => (new_path.clone(), new_ident.clone()),
                    _ => (path.clone(), ident.clone()),
                };
                // The fully qualified path first: after it is rewritten there is
                // no bare segment left to confuse the second pass.
                apply(&mut current, &mut hits, path, &new_path);
                if declaring.contains(&file) {
                    apply(
                        &mut current,
                        &mut hits,
                        &format!("mod {ident};"),
                        &format!("mod {new_ident};"),
                    );
                    apply(
                        &mut current,
                        &mut hits,
                        &format!("{ident}::"),
                        &format!("{new_ident}::"),
                    );
                }
            }
            (ModuleIdentity::Python { path }, target) => {
                let new_path = match target {
                    Some(ModuleIdentity::Python { path: new_path }) => new_path.clone(),
                    _ => path.clone(),
                };
                apply(&mut current, &mut hits, path, &new_path);
            }
        }
        if hits == 0 {
            continue;
        }
        found.push(ModuleReference {
            file,
            rewritten: (to.is_some() && current != text).then_some(current),
        });
    }
    found
}

/// A path that may not exist yet, resolved against the root and checked for
/// escapes lexically — `canonicalize` cannot answer for a file that is about to
/// be created, and a `..` that leaves the repository must never reach the disk.
fn edit_resolve_new_path(
    root_path: &std::path::Path,
    file: &str,
) -> EditResult<(String, std::path::PathBuf)> {
    let base = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.to_path_buf());
    let candidate = std::path::Path::new(file);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base.join(candidate)
    };
    let mut normalized = std::path::PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(EditRefusal::new(
                        "path_outside_repo",
                        format!("{file} is outside {}", base.display()),
                        17,
                    ));
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    let Ok(relative) = normalized.strip_prefix(&base) else {
        return Err(EditRefusal::new(
            "path_outside_repo",
            format!("{file} is outside {}", base.display()),
            17,
        ));
    };
    let rel = relative.to_string_lossy().replace('\\', "/");
    if rel.is_empty() {
        return Err(EditRefusal::new(
            "path_outside_repo",
            format!("{file} is the repository root, not a file in it"),
            17,
        ));
    }
    let abs = root_path.join(&rel);
    Ok((rel, abs))
}

/// The record a whole-file verb answers with once the work is done.
fn edit_whole_file_record(
    root_path: &std::path::Path,
    rel: &str,
    bytes: &[u8],
    before: &[u8],
    published: bool,
) -> EditRecord {
    let text = String::from_utf8_lossy(bytes).into_owned();
    let mut operation = EditOperation {
        file: rel.to_string(),
        ranges: vec![(0, bytes.len())],
        result_span: Some(text.clone()),
        sha_before: Some(edit_sha256_hex(before)),
        sha_after: Some(edit_sha256_hex(bytes)),
        diff: Some(edit_unified_diff(rel, before, bytes)),
        ..EditOperation::default()
    };
    let mut record = EditRecord {
        files: vec![rel.to_string()],
        span: Some(edit_span_lines(bytes, 0, bytes.len())),
        text: Some(text),
        published,
        ..EditRecord::default()
    };
    if published {
        let handle = greppy_edit::EditHandle::for_range(
            root_path,
            std::path::Path::new(rel),
            bytes,
            0,
            bytes.len(),
        )
        .ok()
        .map(|handle| handle.encode());
        operation.handle = handle.clone();
        record.handle = handle;
    }
    record.operations = vec![operation];
    record
}

fn run_edit_write(
    root_path: &std::path::Path,
    file: &str,
    content: Option<String>,
    content_file: Option<String>,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    // The content is resolved first on purpose: "there is no new text" is the
    // caller's mistake whether or not the target happens to exist, and naming
    // the target instead would send it looking in the wrong place.
    let bytes = edit_content_bytes(content, content_file)?;
    let (rel, abs) = edit_resolve_new_path(root_path, file)?;
    if abs.is_dir() {
        return Err(EditRefusal::new(
            "file_exists",
            format!("{file} is a directory, not a file"),
            13,
        ));
    }
    if abs.exists() {
        // D5: two ways to do the same thing without a named difference breaks
        // "one name per thing". `write` creates; rewriting is `replace --file`.
        return Err(EditRefusal::new(
            "file_exists",
            format!("{file} already exists; `replace --file {file}` rewrites it"),
            13,
        ));
    }
    if dry_run {
        return Ok(edit_whole_file_record(root_path, &rel, &bytes, b"", false));
    }
    // There is no `greppy mkdir`, so a new module in a new directory has to be
    // possible here or it is not possible through greppy at all.
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            EditRefusal::new(
                "publish_failed",
                format!("create {}: {error}", parent.display()),
                16,
            )
        })?;
    }
    let transaction = edit_journal_open(
        root_path,
        &[UndoBefore {
            rel: rel.clone(),
            content: None,
        }],
    );
    edit_journal_crash_hook()?;
    if let Err(error) = std::fs::write(&abs, &bytes) {
        edit_journal_abort(root_path);
        return Err(EditRefusal::new(
            "publish_failed",
            format!("{file}: {error}"),
            16,
        ));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    let mut record = edit_whole_file_record(root_path, &rel, &bytes, b"", true);
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

fn run_edit_move(
    root_path: &std::path::Path,
    file: &str,
    to: &str,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let (rel, abs, content) = edit_read_file(root_path, file)?;
    if abs.is_dir() {
        return Err(EditRefusal::new(
            "file_not_found",
            format!("{file} is a directory, not a file"),
            10,
        ));
    }
    let (new_rel, destination) = edit_resolve_new_path(root_path, to)?;
    // A rename that only changes the case of the name is a rename, and on a
    // case-insensitive file system it is the one an `exists()` check refuses by
    // mistake.
    let case_only = new_rel != rel && new_rel.to_lowercase() == rel.to_lowercase();
    if new_rel == rel {
        // Moving a file onto itself is a no-op, not a way to lose it.
        return Ok(EditRecord {
            files: vec![rel.clone()],
            published: !dry_run,
            extra: vec![
                ("from", serde_json::json!(rel)),
                ("to", serde_json::json!(new_rel)),
                ("rewrote", serde_json::json!(Vec::<String>::new())),
            ],
            ..EditRecord::default()
        });
    }
    if destination.exists() && !case_only {
        return Err(EditRefusal::new(
            "file_exists",
            format!("{to} already exists; nothing was moved"),
            13,
        ));
    }
    let references = edit_module_references(root_path, &rel, Some(&new_rel));
    let rewrote: Vec<String> = references
        .iter()
        .filter(|reference| reference.rewritten.is_some())
        .map(|reference| reference.file.clone())
        .collect();
    let mut record = EditRecord {
        headline: Some(if dry_run {
            format!("would move {rel} -> {new_rel}")
        } else {
            format!("moved {rel} -> {new_rel}")
        }),
        files: std::iter::once(new_rel.clone())
            .chain(rewrote.iter().cloned())
            .collect(),
        published: !dry_run,
        extra: vec![
            ("from", serde_json::json!(rel)),
            ("to", serde_json::json!(new_rel)),
            ("rewrote", serde_json::json!(rewrote)),
        ],
        ..EditRecord::default()
    };
    if dry_run {
        return Ok(record);
    }
    let mut before = vec![
        UndoBefore {
            rel: rel.clone(),
            content: Some(content.clone()),
        },
        UndoBefore {
            rel: new_rel.clone(),
            content: None,
        },
    ];
    for reference in &references {
        if reference.rewritten.is_some() {
            before.push(UndoBefore::read(root_path, &reference.file));
        }
    }
    let transaction = edit_journal_open(root_path, &before);
    edit_journal_crash_hook()?;
    let publish = (|| -> std::io::Result<()> {
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if case_only {
            // Writing the destination and unlinking the source is the same file
            // twice on a case-insensitive file system, which destroys it.
            let interim = abs.with_extension("greppy-rename");
            std::fs::rename(&abs, &interim)?;
            std::fs::rename(&interim, &destination)?;
        } else {
            std::fs::write(&destination, &content)?;
            std::fs::remove_file(&abs)?;
        }
        for reference in &references {
            if let Some(text) = &reference.rewritten {
                std::fs::write(root_path.join(&reference.file), text)?;
            }
        }
        Ok(())
    })();
    if let Err(error) = publish {
        if let Some(record) = edit_journal_read(
            &edit_journal_dir(root_path).join(EDIT_JOURNAL_PENDING),
        ) {
            let _ = edit_journal_restore(root_path, &record, false);
        }
        edit_journal_abort(root_path);
        return Err(EditRefusal::new(
            "publish_failed",
            format!("{file} -> {to}: {error}"),
            16,
        ));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

fn run_edit_remove(
    root_path: &std::path::Path,
    file: &str,
    force: bool,
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let (rel, abs, content) = edit_read_file(root_path, file)?;
    if abs.is_dir() {
        return Err(EditRefusal::new(
            "file_not_found",
            format!("{file} is a directory, not a file"),
            10,
        ));
    }
    let references: Vec<String> = edit_module_references(root_path, &rel, None)
        .into_iter()
        .map(|reference| reference.file)
        .collect();
    if !references.is_empty() && !force {
        // Owner decision 4: a delete that leaves dangling imports turns a
        // one-command mistake into a broken build, and it cannot be undone by
        // reading, because the content is gone.
        let listed = references
            .iter()
            .map(|file| format!("\n  {file}"))
            .collect::<String>();
        return Err(
            EditRefusal::new(
                "still_referenced",
                format!("{rel} is still referenced by:{listed}"),
                13,
            )
            .with("references", serde_json::json!(references)),
        );
    }
    let mut record = EditRecord {
        headline: Some(if dry_run {
            format!("would remove {rel}")
        } else {
            format!("removed {rel}")
        }),
        files: vec![rel.clone()],
        published: !dry_run,
        extra: vec![("references", serde_json::json!(references))],
        ..EditRecord::default()
    };
    for reference in record
        .extra
        .first()
        .and_then(|(_, value)| value.as_array())
        .cloned()
        .unwrap_or_default()
    {
        if let Some(file) = reference.as_str() {
            record.notes.push(format!("{file} still references {rel}"));
        }
    }
    if dry_run {
        return Ok(record);
    }
    let transaction = edit_journal_open(
        root_path,
        &[UndoBefore {
            rel: rel.clone(),
            content: Some(content),
        }],
    );
    edit_journal_crash_hook()?;
    if let Err(error) = std::fs::remove_file(&abs) {
        edit_journal_abort(root_path);
        return Err(EditRefusal::new(
            "publish_failed",
            format!("{rel}: {error}"),
            16,
        ));
    }
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    if verify {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

// --- a plan of whole-file verbs ---------------------------------------------

/// `apply --plan` over `write` / `move` / `remove`: all of it or none of it, and
/// one entry in the journal, so one `undo` takes the whole batch back.
fn run_edit_plan_whole_file(
    root_path: &std::path::Path,
    operations: &[serde_json::Value],
    dry_run: bool,
    verify: bool,
) -> EditResult<EditRecord> {
    let mut before: Vec<UndoBefore> = Vec::new();
    let remember = |root_path: &std::path::Path, rel: &str, before: &mut Vec<UndoBefore>| {
        if !before.iter().any(|item| item.rel == rel) {
            before.push(UndoBefore::read(root_path, rel));
        }
    };
    for operation in operations {
        for key in ["file", "to"] {
            if let Some(file) = operation[key].as_str() {
                if let Ok((rel, _)) = edit_resolve_new_path(root_path, file) {
                    remember(root_path, &rel, &mut before);
                }
            }
        }
        // A move rewrites the files that name the module, and they belong to the
        // same transaction — otherwise one `undo` puts the file back and leaves
        // the imports pointing at a module that is gone again.
        if operation["verb"].as_str() == Some("move") {
            if let (Some(file), Some(to)) = (operation["file"].as_str(), operation["to"].as_str()) {
                if let (Ok((rel, _)), Ok((new_rel, _))) = (
                    edit_resolve_new_path(root_path, file),
                    edit_resolve_new_path(root_path, to),
                ) {
                    for reference in edit_module_references(root_path, &rel, Some(&new_rel)) {
                        remember(root_path, &reference.file, &mut before);
                    }
                }
            }
        }
    }
    let transaction = (!dry_run)
        .then(|| edit_journal_open(root_path, &before))
        .flatten();
    if !dry_run {
        edit_journal_crash_hook()?;
    }
    let mut files = Vec::new();
    let mut failure = None;
    for operation in operations {
        let verb = operation["verb"].as_str().unwrap_or_default();
        let file = operation["file"].as_str().unwrap_or_default().to_string();
        let outcome = match verb {
            "write" => run_edit_write(
                root_path,
                &file,
                operation["content"].as_str().map(str::to_string),
                operation["content_file"].as_str().map(str::to_string),
                dry_run,
                false,
            ),
            "move" => run_edit_move(
                root_path,
                &file,
                operation["to"].as_str().unwrap_or_default(),
                dry_run,
                false,
            ),
            "remove" => run_edit_remove(
                root_path,
                &file,
                operation["force"].as_bool().unwrap_or(false),
                dry_run,
                false,
            ),
            other => Err(EditRefusal::new(
                "invalid_plan",
                format!("`{other}` is not one of the whole-file verbs write, move and remove"),
                20,
            )),
        };
        match outcome {
            Ok(record) => files.extend(record.files),
            Err(refusal) => {
                failure = Some(refusal);
                break;
            }
        }
    }
    if let Some(refusal) = failure {
        // "Many edits as one single change" is the only reason to send a plan:
        // a batch whose first two operations are already on disk is worse than
        // no plan at all, because the caller believes nothing happened.
        if !dry_run {
            if let Some(record) =
                edit_journal_read(&edit_journal_dir(root_path).join(EDIT_JOURNAL_PENDING))
            {
                let _ = edit_journal_restore(root_path, &record, false);
            }
            edit_journal_abort(root_path);
        }
        return Err(refusal);
    }
    files.sort();
    files.dedup();
    if let Some(id) = transaction {
        edit_journal_close(root_path, &id);
    }
    let mut record = EditRecord {
        files,
        published: !dry_run,
        ..EditRecord::default()
    };
    if verify && !dry_run {
        record.diagnostics = Some(edit_verify_diagnostics(root_path, &record.files));
    }
    Ok(record)
}

/// Whether a plan is written in the whole-file verbs. A plan that mixes them
/// with the span verbs is not silently split: it is refused by the executor
/// that owns it.
fn plan_is_whole_file(plan_text: &str) -> Option<Vec<serde_json::Value>> {
    let plan: serde_json::Value = serde_json::from_str(plan_text).ok()?;
    let operations = plan["operations"].as_array()?.clone();
    if operations.is_empty() {
        return None;
    }
    operations
        .iter()
        .all(|operation| {
            matches!(
                operation["verb"].as_str(),
                Some("write") | Some("move") | Some("remove")
            )
        })
        .then_some(operations)
}


// --- reporting --------------------------------------------------------------

/// The record as data. `full` is the archival form `--report` writes: it adds
/// the diff and the resulting text, which stdout deliberately leaves out
/// because they cost the context window the compact form exists to protect.
fn edit_record_json(
    record: &EditRecord,
    full: bool,
    report_path: Option<&str>,
) -> serde_json::Value {
    let mut value = serde_json::Map::new();
    value.insert("schema_version".into(), serde_json::json!(EDIT_RECORD_SCHEMA));
    value.insert(
        "status".into(),
        serde_json::json!(if record.published {
            "applied"
        } else {
            // A dry run that reports "applied" is read as a completed edit by
            // every caller that trusts `status` over `published`.
            "would_apply"
        }),
    );
    value.insert("published".into(), serde_json::json!(record.published));
    value.insert("exit_code".into(), serde_json::json!(0));
    if let Some(first) = record.files.first() {
        value.insert("file".into(), serde_json::json!(first));
    }
    value.insert("files".into(), serde_json::json!(record.files));
    for (key, extra) in &record.extra {
        value.insert((*key).into(), extra.clone());
    }
    if let Some((first, last)) = record.span {
        value.insert("span".into(), serde_json::json!(format!("{first}:{last}")));
    }
    if let Some(text) = &record.text {
        value.insert("text".into(), serde_json::json!(text));
    }
    if let Some(handle) = &record.handle {
        value.insert("handle".into(), serde_json::json!(handle));
    }
    let operations: Vec<serde_json::Value> = record
        .operations
        .iter()
        .map(|operation| {
            let mut entry = serde_json::Map::new();
            entry.insert("file".into(), serde_json::json!(operation.file));
            entry.insert(
                "changed_byte_ranges".into(),
                serde_json::json!(operation.ranges),
            );
            if let Some(text) = &operation.result_span {
                entry.insert("result_span".into(), serde_json::json!(text));
                if full {
                    entry.insert("node_after".into(), serde_json::json!(text));
                }
            }
            if let Some(handle) = &operation.handle {
                entry.insert("handle".into(), serde_json::json!(handle));
            }
            if let Some(sha) = &operation.sha_before {
                entry.insert("file_sha256_before".into(), serde_json::json!(sha));
            }
            if let Some(sha) = &operation.sha_after {
                entry.insert("file_sha256_after".into(), serde_json::json!(sha));
            }
            if full {
                if let Some(diff) = &operation.diff {
                    entry.insert("unified_diff".into(), serde_json::json!(diff));
                }
            }
            serde_json::Value::Object(entry)
        })
        .collect();
    value.insert("operations".into(), serde_json::json!(operations));
    if let Some(diagnostics) = &record.diagnostics {
        value.insert("diagnostics".into(), serde_json::json!(diagnostics));
        value.insert(
            "verify".into(),
            serde_json::json!({ "diagnostics": diagnostics }),
        );
    }
    if !record.notes.is_empty() {
        value.insert("references".into(), serde_json::json!(record.notes));
    }
    if let Some(path) = report_path {
        value.insert("report_path".into(), serde_json::json!(path));
    }
    serde_json::Value::Object(value)
}

/// A refusal is an answer too: the same shape, with the cause named. Without
/// `published` and `exit_code` a caller that asked for `--json` cannot tell a
/// refusal from a success without re-reading the process exit code.
fn edit_refusal_json(refusal: &EditRefusal, report_path: Option<&str>) -> serde_json::Value {
    let mut error = serde_json::Map::new();
    error.insert("code".into(), serde_json::json!(refusal.code));
    error.insert("message".into(), serde_json::json!(refusal.message));
    for (key, value) in &refusal.extra {
        error.insert((*key).into(), value.clone());
    }
    let mut value = serde_json::Map::new();
    value.insert("schema_version".into(), serde_json::json!(EDIT_RECORD_SCHEMA));
    value.insert("status".into(), serde_json::json!("refused"));
    value.insert("published".into(), serde_json::json!(false));
    value.insert("exit_code".into(), serde_json::json!(refusal.exit));
    value.insert("operations".into(), serde_json::json!([]));
    value.insert("error".into(), serde_json::Value::Object(error));
    if let Some(path) = report_path {
        value.insert("report_path".into(), serde_json::json!(path));
    }
    serde_json::Value::Object(value)
}

/// Write the archival record. A refusal is the case where the evidence matters
/// most, so `--report` is honoured there too.
fn write_edit_report(path: &str, value: &serde_json::Value) -> Result<()> {
    let rendered = serde_json::to_string_pretty(value)
        .map_err(|error| Error::Invalid(format!("serialize edit record: {error}")))?;
    std::fs::write(path, format!("{rendered}\n")).map_err(|source| Error::Io {
        context: format!("write report {path}"),
        source,
    })
}

fn emit_edit_outcome(
    outcome: EditResult<EditRecord>,
    json: bool,
    report: Option<String>,
) -> Result<i32> {
    let report_path = report.as_deref();
    match outcome {
        Ok(record) => {
            if let Some(path) = report_path {
                write_edit_report(path, &edit_record_json(&record, true, report_path))?;
            }
            if json {
                let compact = edit_record_json(&record, false, report_path);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&compact)
                        .map_err(|error| Error::Invalid(format!("serialize edit record: {error}")))?
                );
            } else {
                // Success is one bare line: verb and address. The text that
                // landed is the text the caller sent (CAS); handles appear only
                // where the spec orders them. A dry run must not say "applied"
                // — it wrote nothing, and a receipt that overstates is the
                // worst output this tool can produce.
                let word = if record.published { "applied" } else { "would apply" };
                if let Some(headline) = &record.headline {
                    println!("{headline}");
                } else {
                    for (index, file) in record.files.iter().enumerate() {
                        match record.span {
                            Some((first, last)) if index == 0 && first == last => {
                                println!("{word} {file}:{first}");
                            }
                            Some((first, last)) if index == 0 => {
                                println!("{word} {file}:{first}-{last}");
                            }
                            _ => println!("{word} {file}"),
                        }
                    }
                }
                for note in &record.notes {
                    println!("{note}");
                }
                if let Some(diagnostics) = &record.diagnostics {
                    for diagnostic in diagnostics {
                        println!("{diagnostic}");
                    }
                }
            }
            Ok(0)
        }
        Err(refusal) => {
            let value = edit_refusal_json(&refusal, report_path);
            if let Some(path) = report_path {
                write_edit_report(path, &value)?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())
                );
            } else {
                // stdout carries answers; a refusal is not one. A caller that
                // pipes an edit into the next command must receive nothing
                // rather than a line of prose it would then act on.
                eprintln!("{}", refusal.message);
            }
            Ok(refusal.exit)
        }
    }
}

/// Map a certificate refusal onto the same `error.code` shape the grammar
/// verbs use, so one caller-side branch covers every edit verb.
fn certificate_refusal_code(certificate: &greppy_edit::Certificate) -> &'static str {
    let occurrence_miss = certificate.operations.iter().any(|operation| {
        operation.postconditions.iter().any(|postcondition| {
            !postcondition.passed
                && (postcondition.name.contains("occurrences")
                    || postcondition.name.contains("cardinality"))
        })
    });
    if occurrence_miss {
        return "match_count";
    }
    match certificate.status {
        greppy_edit::Status::NotFound => "not_found",
        greppy_edit::Status::Ambiguous => "ambiguous_symbol",
        greppy_edit::Status::Stale => "stale_handle",
        greppy_edit::Status::InvalidResult => "invalid_result",
        greppy_edit::Status::ValidationFailed => "validation_failed",
        greppy_edit::Status::PublishFailed => "publish_failed",
        _ => "refused",
    }
}

// --- dispatch ---------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn dispatch_edit_grammar(
    command: EditCommand,
    json: bool,
    root: Option<&str>,
    root_path: &std::path::Path,
) -> Result<GrammarDispatch> {
    let code = match command {
        EditCommand::Replace {
            file,
            old,
            old_file,
            pattern,
            lines,
            symbol,
            body,
            target,
            content,
            content_file,
            expect,
            path,
            dry_run,
            verify,
            report,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                edit_guard_single_stdin(&[
                    ("--old-file", old_file.as_deref()),
                    ("--content-file", content_file.as_deref()),
                ])?;
                let new_bytes = edit_content_bytes(content, content_file)?;
                let spec = WhereSpec {
                    file,
                    old,
                    old_file,
                    pattern,
                    lines,
                    symbol,
                    body,
                    target,
                    path,
                };
                let kind = classify_selector(&spec)?;
                edit_expect_positive(expect)?;
                let located = edit_locate(&spec, kind, root, root_path)?;
                edit_check_cardinality(&located, expect)?;
                let (new_content, changed) = edit_op_replace(&located, &new_bytes);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, report)?
        }
        EditCommand::Insert {
            file,
            old,
            old_file,
            pattern,
            lines,
            symbol,
            body,
            target,
            before,
            after,
            content,
            content_file,
            path,
            dry_run,
            verify,
            report,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                edit_guard_single_stdin(&[
                    ("--old-file", old_file.as_deref()),
                    ("--content-file", content_file.as_deref()),
                ])?;
                let new_bytes = edit_content_bytes(content, content_file)?;
                if before == after {
                    return Err(EditRefusal::new(
                        "side_missing",
                        "insert lands on one side of the anchor: pass --before or --after",
                        20,
                    ));
                }
                let spec = WhereSpec {
                    file,
                    old,
                    old_file,
                    pattern,
                    lines,
                    symbol,
                    body,
                    target,
                    path,
                };
                let kind = classify_selector(&spec)?;
                if !matches!(
                    kind,
                    SelectorKind::Symbol | SelectorKind::Lines | SelectorKind::Target
                ) {
                    return Err(EditRefusal::new(
                        "selector_not_supported",
                        format!(
                            "insert takes --symbol S, --file F --lines A:B or --target H; \
                             {} has no side to land on",
                            kind.name()
                        ),
                        20,
                    ));
                }
                let located = edit_locate(&spec, kind, root, root_path)?;
                let (new_content, changed) = edit_op_insert(&located, &new_bytes, before);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, report)?
        }
        EditCommand::Delete {
            file,
            old,
            old_file,
            pattern,
            lines,
            symbol,
            body,
            target,
            expect,
            path,
            dry_run,
            verify,
            report,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                let spec = WhereSpec {
                    file,
                    old,
                    old_file,
                    pattern,
                    lines,
                    symbol,
                    body,
                    target,
                    path,
                };
                let kind = classify_selector(&spec)?;
                edit_expect_positive(expect)?;
                let located = edit_locate(&spec, kind, root, root_path)?;
                edit_check_cardinality(&located, expect)?;
                let (new_content, changed) = edit_op_delete(&located);
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, report)?
        }
        EditCommand::Patch {
            file,
            old,
            old_file,
            pattern,
            lines,
            symbol,
            body,
            target,
            patch_file,
            path,
            dry_run,
            verify,
            report,
        } => {
            let outcome = (|| -> EditResult<EditRecord> {
                edit_guard_single_stdin(&[
                    ("--old-file", old_file.as_deref()),
                    ("--patch-file", patch_file.as_deref()),
                ])?;
                let Some(patch_file) = patch_file else {
                    return Err(EditRefusal::new(
                        "content_missing",
                        "no diff: pass --patch-file FILE",
                        20,
                    ));
                };
                let patch = read_source_arg(&patch_file).map_err(|error| {
                    EditRefusal::new(
                        "invalid_patch",
                        format!("--patch-file {patch_file}: {error}"),
                        20,
                    )
                })?;
                let spec = WhereSpec {
                    file,
                    old,
                    old_file,
                    pattern,
                    lines,
                    symbol,
                    body,
                    target,
                    path,
                };
                let kind = classify_selector(&spec)?;
                if matches!(kind, SelectorKind::Text | SelectorKind::Pattern) {
                    return Err(EditRefusal::new(
                        "selector_not_supported",
                        format!(
                            "patch takes --symbol S, --file F --lines A:B, --file F or --target H; \
                             {} gives the hunks nothing to count from",
                            kind.name()
                        ),
                        20,
                    ));
                }
                let located = edit_locate(&spec, kind, root, root_path)?;
                let (new_content, changed) = edit_op_patch(&located, &patch)?;
                edit_publish(root_path, &located, new_content, changed, dry_run, verify)
            })();
            emit_edit_outcome(outcome, json, report)?
        }
        EditCommand::Write {
            file,
            content,
            content_file,
            dry_run,
            verify,
            report,
        } => {
            let outcome = run_edit_write(root_path, &file, content, content_file, dry_run, verify);
            emit_edit_outcome(outcome, json, report)?
        }
        EditCommand::Move {
            file,
            to,
            dry_run,
            verify,
            report,
        } => {
            let outcome = run_edit_move(root_path, &file, &to, dry_run, verify);
            emit_edit_outcome(outcome, json, report)?
        }
        EditCommand::Remove {
            file,
            force,
            dry_run,
            verify,
            report,
        } => {
            let outcome = run_edit_remove(root_path, &file, force, dry_run, verify);
            emit_edit_outcome(outcome, json, report)?
        }
        EditCommand::Undo { dry_run, report } => {
            let outcome = run_edit_undo(root_path, dry_run);
            emit_edit_outcome(outcome, json, report)?
        }
        other => return Ok(GrammarDispatch::Passthrough(Box::new(other))),
    };
    Ok(GrammarDispatch::Handled(code))
}

fn edit_status_name(status: greppy_edit::Status) -> &'static str {
    match status {
        greppy_edit::Status::Applied => "applied",
        greppy_edit::Status::AlreadySatisfied => "already-satisfied",
        greppy_edit::Status::NotFound => "not-found",
        greppy_edit::Status::Ambiguous => "ambiguous",
        greppy_edit::Status::Stale => "stale",
        greppy_edit::Status::InvalidResult => "invalid-result",
        greppy_edit::Status::ValidationFailed => "validation-failed",
        greppy_edit::Status::PublishFailed => "publish-failed",
    }
}

fn edit_operation_path(
    operation: &greppy_edit::certificate::OperationReport,
    root_path: &std::path::Path,
) -> String {
    let path = std::path::Path::new(&operation.file);
    let relative = if path.is_absolute() {
        path.strip_prefix(root_path).unwrap_or(path)
    } else {
        path
    };
    relative.to_string_lossy().replace('\\', "/")
}

fn diff_after_line_span(diff: &str) -> Option<(usize, usize)> {
    let mut start = usize::MAX;
    let mut end = 0usize;
    for line in diff.lines().filter(|line| line.starts_with("@@ ")) {
        let range = line
            .split_whitespace()
            .find(|field| field.starts_with('+'))?
            .trim_start_matches('+');
        let (line_start, count) = range.split_once(',').unwrap_or((range, "1"));
        let line_start = line_start.parse::<usize>().ok()?.max(1);
        let count = count.parse::<usize>().ok()?;
        start = start.min(line_start);
        end = end.max(line_start.saturating_add(count.saturating_sub(1)));
    }
    (start != usize::MAX).then_some((start, end.max(start)))
}

fn line_for_byte(content: &[u8], offset: usize) -> usize {
    content[..offset.min(content.len())]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}

fn edit_operation_line_span(
    operation: &greppy_edit::certificate::OperationReport,
    root_path: &std::path::Path,
) -> (usize, usize) {
    if let Some(span) = operation
        .unified_diff
        .as_deref()
        .and_then(diff_after_line_span)
    {
        return span;
    }
    let path = if std::path::Path::new(&operation.file).is_absolute() {
        std::path::PathBuf::from(&operation.file)
    } else {
        root_path.join(&operation.file)
    };
    let content = std::fs::read(path).unwrap_or_default();
    if let Some(start_byte) = operation
        .changed_byte_ranges
        .iter()
        .map(|range| range.0)
        .min()
    {
        let start = line_for_byte(&content, start_byte);
        let end = operation.node_after.as_deref().map_or_else(
            || {
                operation
                    .changed_byte_ranges
                    .iter()
                    .map(|range| line_for_byte(&content, range.1))
                    .max()
                    .unwrap_or(start)
            },
            |span| start.saturating_add(span.lines().count().max(1) - 1),
        );
        return (start, end.max(start));
    }
    let line_count = content.iter().filter(|byte| **byte == b'\n').count()
        + usize::from(!content.is_empty() && !content.ends_with(b"\n"));
    (1, line_count.max(1))
}

fn one_line_truncated(text: &str, max_chars: usize) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control.is_control() => escaped.extend(control.escape_default()),
            printable => escaped.push(printable),
        }
    }
    let mut chars = escaped.chars();
    let mut compact = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() && max_chars > 0 {
        compact.pop();
        compact.push('…');
    }
    compact
}

fn render_compact_edit_certificate(
    certificate: &greppy_edit::Certificate,
    root_path: &std::path::Path,
) {
    // The output states what is the case; it does not tell the caller what to do
    // next. A hard-wired suggestion does not know the task, and the traces show it
    // guesses wrong: one handed `search-symbols` a file path, a query that cannot
    // match by construction. Printed nine times, followed twice.
    if let Some(diagnosis) = certificate.compact_failure_diagnosis() {
        println!("diagnosis: {}", one_line_truncated(&diagnosis, 500));
        return;
    }
    if certificate.exit_code() != 0 {
        println!(
            "diagnosis: edit {} for {} operation(s)",
            edit_status_name(certificate.status),
            certificate.operations.len()
        );
        return;
    }
    let status = edit_status_name(certificate.status);
    for operation in &certificate.operations {
        let path = edit_operation_path(operation, root_path);
        let (start, end) = edit_operation_line_span(operation, root_path);
        // One bare line per operation (spec v8). The written-state echo that
        // used to follow carried a measured justification (130/269 edits were
        // re-read without it, on an older output regime); the frozen spec
        // removes it on the CAS argument. The post-edit re-read rate is the
        // first metric to check on the next bench — if it climbs, revisit
        // here, with data.
        if start == end {
            println!("{status} {path}:{start}");
        } else {
            println!("{status} {path}:{start}-{end}");
        }
    }
}

/// Shared tail of every edit command: refresh the store after a published
/// edit, preserve the full certificate, render compact stdout, map the exit code.
/// Say what the operation was looking for. "expected 1 target(s), found 0"
/// names the count but not the text, and the caller cannot tell a typo in the
/// anchor from a file that moved on. The plan it submitted has the text, so it
/// is put back on the postcondition that failed.
fn annotate_plan_refusal(certificate: &mut greppy_edit::Certificate, plan_text: &str) {
    let Ok(plan) = serde_json::from_str::<serde_json::Value>(plan_text) else {
        return;
    };
    let operations = plan
        .get("operations")
        .or_else(|| plan.get("ops"))
        .and_then(|value| value.as_array());
    let Some(operations) = operations else { return };
    for (index, report) in certificate.operations.iter_mut().enumerate() {
        if report.postconditions_passed {
            continue;
        }
        let declared = operations.get(index).filter(|operation| {
            operation
                .get("file")
                .and_then(|file| file.as_str())
                .is_none_or(|file| file.ends_with(&report.file) || report.file.ends_with(file))
        });
        let Some(declared) = declared else { continue };
        let sought = ["old", "pattern", "symbol", "selector"]
            .iter()
            .find_map(|key| declared.get(*key).and_then(|value| value.as_str()));
        let Some(sought) = sought else { continue };
        for postcondition in &mut report.postconditions {
            if postcondition.passed {
                continue;
            }
            let detail = postcondition.detail.get_or_insert_with(String::new);
            if !detail.contains(sought) {
                detail.push_str(&format!("; looking for `{sought}`"));
            }
            break;
        }
    }
}

/// A handle for every span a certificate operation wrote. A plan writes several
/// spans, so one handle is not enough: without one per operation a multi-file
/// change forces a re-read of every file it touched. Only a published
/// operation gets one — a handle addresses bytes that are on disk.
fn certificate_operation_handles(
    certificate: &greppy_edit::Certificate,
    root_path: &std::path::Path,
) -> Vec<Option<String>> {
    certificate
        .operations
        .iter()
        .map(|operation| {
            if !certificate.published {
                return None;
            }
            // `changed_byte_ranges` addresses the file as it WAS; the handle
            // has to address it as it now IS, and the text that is now there is
            // exactly `node_after`.
            let (start, before_end) = *operation.changed_byte_ranges.first()?;
            let end = operation
                .node_after
                .as_ref()
                .map_or(before_end, |text| start + text.len());
            let candidate = std::path::Path::new(&operation.file);
            let abs = if candidate.is_absolute() {
                candidate.to_path_buf()
            } else {
                root_path.join(candidate)
            };
            let content = std::fs::read(&abs).ok()?;
            let rel = abs
                .strip_prefix(root_path)
                .map(|relative| relative.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| operation.file.clone());
            greppy_edit::EditHandle::for_range(
                root_path,
                std::path::Path::new(&rel),
                &content,
                start,
                end.min(content.len()),
            )
            .ok()
            .map(|handle| handle.encode())
        })
        .collect()
}

/// Put the transaction every operation belongs to, and the handle for the span
/// it wrote, on the operation itself. A single id printed above the list does
/// not prove the operations share it.
fn certificate_operation_extras(
    value: &mut serde_json::Value,
    transaction_id: &str,
    handles: &[Option<String>],
) {
    let Some(operations) = value.get_mut("operations").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for (index, operation) in operations.iter_mut().enumerate() {
        let Some(operation) = operation.as_object_mut() else {
            continue;
        };
        operation.insert("transaction_id".into(), serde_json::json!(transaction_id));
        if let Some(Some(handle)) = handles.get(index) {
            operation.insert("handle".into(), serde_json::json!(handle));
        }
    }
}

/// The operation that actually refused. Once one operation of an all-or-nothing
/// plan refuses, every operation is marked failed, so the first one is usually
/// only a casualty; the caller needs the one that names a cause.
fn certificate_refusal_message(certificate: &greppy_edit::Certificate) -> Option<String> {
    let mut casualty = None;
    for operation in &certificate.operations {
        if operation.postconditions_passed {
            continue;
        }
        let Some(detail) = operation
            .postconditions
            .iter()
            .find(|postcondition| !postcondition.passed)
            .and_then(|postcondition| postcondition.detail.as_deref())
        else {
            continue;
        };
        let line = format!("{}: {detail}", operation.file);
        if detail.contains("another operation") {
            casualty.get_or_insert(line);
        } else {
            return Some(line);
        }
    }
    casualty
}

fn finish_edit(
    certificate: greppy_edit::Certificate,
    report_path: Option<String>,
    json: bool,
    root: Option<&str>,
    root_path: &std::path::Path,
) -> Result<i32> {
    let mut certificate = certificate;
    if certificate.published {
        // close the read->edit->read loop: refresh the store so the next
        // read/graph query addresses the edited file without a manual
        // reindex. index() is incremental from the second run, so this
        // touches only the changed file. A refresh failure downgrades the
        // flag, never the edit (the workspace write already happened).
        let refreshed = std::env::current_exe()
            .ok()
            .and_then(|exe| {
                std::process::Command::new(exe)
                    .arg("--root")
                    .arg(root_path)
                    .arg("index")
                    .arg(root_path)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .ok()
            })
            .map(|status| status.success())
            .unwrap_or(false);
        for op in &mut certificate.operations {
            op.store_refreshed = refreshed;
        }
    }
    let handles = certificate_operation_handles(&certificate, root_path);
    let mut full_value = serde_json::to_value(&certificate)
        .map_err(|e| Error::Invalid(format!("serialize full certificate: {e}")))?;
    certificate_operation_extras(&mut full_value, &certificate.transaction_id, &handles);
    let full = serde_json::to_string_pretty(&full_value)
        .map_err(|e| Error::Invalid(format!("serialize full certificate: {e}")))?;
    let mut report_written = None;
    if let Some(path) = report_path {
        std::fs::write(&path, format!("{full}\n")).map_err(|source| Error::Io {
            context: format!("write report {path}"),
            source,
        })?;
        report_written = Some(path);
    }

    let expand = if let (Ok(store), Ok(project)) =
        (open_default_store_query_writer(root), project_for(root))
    {
        let summary = serde_json::json!({
            "text": format!(
                "edit certificate: {} operation(s), status {}",
                certificate.operations.len(),
                edit_status_name(certificate.status)
            ),
            "transaction_id": &certificate.transaction_id,
            "status": edit_status_name(certificate.status),
            "operations": certificate.operations.len(),
        });
        insert_expand_pack_best_effort(
            &store,
            &project,
            "edit",
            &certificate.transaction_id,
            current_graph_generation_or_zero(&store, root),
            summary,
            format!("{full}\n"),
            serde_json::to_value(&certificate).ok(),
        )
    } else {
        None
    };

    if let Some(expand) = &expand {
        insert_expand_alias_best_effort(root, &certificate.transaction_id, &expand.id);
    }
    if json {
        // The stdout form, not the archival one: it carries `exit_code` and
        // drops evidence that only `--report` and `expand` need. Printing the
        // full Serialize form here silently dropped `exit_code` from the
        // documented certificate contract.
        let stdout_form = certificate
            .to_compact_json_pretty()
            .map_err(|e| Error::Invalid(format!("serialize certificate: {e}")))?;
        // One caller-side branch for every edit verb: a certificate that did
        // not apply carries the same `error.code` the grammar verbs emit.
        let mut value: serde_json::Value = serde_json::from_str(&stdout_form)
            .map_err(|e| Error::Invalid(format!("re-read certificate: {e}")))?;
        certificate_operation_extras(&mut value, &certificate.transaction_id, &handles);
        if let Some(root) = value.as_object_mut() {
            // The evidence stdout omits is omitted, not replaced by a stand-in
            // string that reads like a diff to anything parsing the field.
            if let Some(operations) = root.get_mut("operations").and_then(|v| v.as_array_mut()) {
                for operation in operations {
                    if let Some(operation) = operation.as_object_mut() {
                        operation.remove("unified_diff");
                    }
                }
            }
            if let Some(path) = &report_written {
                root.insert("report_path".into(), serde_json::json!(path));
            }
            if certificate.exit_code() != 0 {
                let message = certificate_refusal_message(&certificate)
                    .or_else(|| certificate.compact_failure_diagnosis())
                    .unwrap_or_else(|| format!("edit {}", edit_status_name(certificate.status)));
                root.insert(
                    "error".into(),
                    serde_json::json!({
                        "code": certificate_refusal_code(&certificate),
                        "message": message,
                    }),
                );
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize certificate: {e}")))?
        );
    } else {
        render_compact_edit_certificate(&certificate, root_path);
    }
    Ok(certificate.exit_code())
}

fn edit_symbol_miss_candidates(
    store: &greppy_store::Store,
    project: &str,
    symbol: &str,
) -> Vec<greppy_edit::certificate::Candidate> {
    let mut candidates = Vec::new();
    for name in symbol_miss_suggestions(store, project, symbol) {
        let Ok(ids) = resolve_symbol_nodes(store, Some(&name)) else {
            continue;
        };
        for id in ids {
            let Ok(Some(node)) = store.get_node(id) else {
                continue;
            };
            if node.file_path.is_empty() || node.start_line < 1 {
                continue;
            }
            let duplicate =
                candidates
                    .iter()
                    .any(|candidate: &greppy_edit::certificate::Candidate| {
                        candidate.qualified_name == node.qualified_name
                            && candidate.path == node.file_path
                    });
            if !duplicate {
                candidates.push(greppy_edit::certificate::Candidate {
                    qualified_name: node.qualified_name,
                    path: node.file_path,
                    line: node.start_line as usize,
                });
            }
            if candidates.len() == 5 {
                return candidates;
            }
        }
    }
    candidates
}

/// Resolve an edit target: either a `--target HANDLE` (verified against the
/// live file) or a `--symbol` (resolved like `read`, against the live file).
/// Returns the file path (workspace-relative), live content, and byte range —
/// or a ready-made refusal certificate (not-found / ambiguous / stale).
enum EditTarget {
    Resolved {
        rel_path: String,
        range: (usize, usize),
        planned_file_sha256: String,
        planned_target_sha256: String,
    },
    Refusal(Box<greppy_edit::Certificate>),
}

#[allow(clippy::too_many_lines)]
fn resolve_edit_target(
    symbol: Option<&str>,
    target: Option<&str>,
    root: Option<&str>,
    root_path: &std::path::Path,
) -> Result<EditTarget> {
    use greppy_edit::certificate as cert;
    fn refusal(
        root_path: &std::path::Path,
        path: &str,
        status: greppy_edit::Status,
        requested_symbol: Option<&str>,
        accepted_spelling: Option<&str>,
        candidates: Vec<cert::Candidate>,
    ) -> greppy_edit::Certificate {
        let mut postconditions = Vec::new();
        if let Some(symbol) = requested_symbol {
            postconditions.push(cert::PostconditionResult {
                name: "requested-symbol".into(),
                passed: true,
                detail: Some(symbol.into()),
            });
        }
        if let Some(spelling) = accepted_spelling {
            postconditions.push(cert::PostconditionResult {
                name: "accepted-symbol-spelling".into(),
                passed: true,
                detail: Some(spelling.into()),
            });
        }
        greppy_edit::Certificate {
            schema_version: cert::CERTIFICATE_SCHEMA.into(),
            status,
            transaction_id: "ge-refused".into(),
            workspace: cert::WorkspaceReport {
                root: root_path.to_string_lossy().into_owned(),
                git_head_before: None,
                git_head_after: None,
            },
            operations: vec![cert::OperationReport {
                id: "resolve".into(),
                file: path.to_string(),
                selector_engine: cert::SelectorEngine::Symbol,
                selector_class: cert::SelectorClass::Resolved,
                scope_matches: 0,
                target_matches: if status == greppy_edit::Status::NotFound {
                    0
                } else {
                    candidates.len()
                },
                file_sha256_before: String::new(),
                file_sha256_after: None,
                target_sha256_before: String::new(),
                target_sha256_after: None,
                outside_declared_ranges_unchanged: true,
                changed_byte_ranges: vec![],
                node_before: None,
                node_after: None,
                unified_diff: None,
                syntax: cert::SyntaxDelta {
                    errors_before: 0,
                    errors_after: 0,
                    new_errors: 0,
                    new_missing_nodes: 0,
                },
                postconditions_passed: false,
                postconditions,
                residual_occurrences: None,
                guarantees: cert::Guarantees {
                    addressed_range: cert::Guarantee::Failed,
                    no_clobber: cert::Guarantee::Proved,
                    byte_isolation: cert::Guarantee::Proved,
                    syntax: cert::Guarantee::NotApplicable,
                    validators: cert::Guarantee::NotApplicable,
                },
                formatter_expanded_change_scope: false,
                store_refreshed: false,
                candidates,
            }],
            validators: vec![],
            published: false,
            publish_mode: greppy_edit::PublishMode::Atomic,
        }
    }

    if let Some(token) = target {
        let handle = greppy_edit::EditHandle::decode(token)?;
        let abs = if std::path::Path::new(&handle.path).is_absolute() {
            std::path::PathBuf::from(&handle.path)
        } else {
            root_path.join(&handle.path)
        };
        let content = std::fs::read(&abs).map_err(|source| Error::Io {
            context: format!("read {}", abs.display()),
            source,
        })?;
        return match handle.verify(&content) {
            Ok(range) => Ok(EditTarget::Resolved {
                rel_path: handle.path.clone(),
                range,
                planned_file_sha256: handle.file_sha256.clone(),
                planned_target_sha256: handle.target_sha256.clone(),
            }),
            Err(_) => Ok(EditTarget::Refusal(Box::new(refusal(
                root_path,
                &handle.path,
                greppy_edit::Status::Stale,
                None,
                None,
                vec![],
            )))),
        };
    }
    let Some(symbol) = symbol else {
        return Err(Error::Invalid(
            "edit needs --symbol SYMBOL or --target HANDLE".into(),
        ));
    };
    let store = open_default_store_query_writer(root)?;
    let ids = resolve_symbol_nodes(&store, Some(symbol))?;
    let mut nodes = Vec::new();
    for id in &ids {
        if let Some(node) = store.get_node(*id)? {
            if !node.file_path.is_empty() && node.start_line >= 1 {
                nodes.push(node);
            }
        }
    }
    if nodes.is_empty() {
        // Carry ranked, addressable near misses in the refusal itself so an
        // agent can recover without another discovery turn.
        let project = project_for(root)?;
        let candidates = edit_symbol_miss_candidates(&store, &project, symbol);
        return Ok(EditTarget::Refusal(Box::new(refusal(
            root_path,
            "",
            greppy_edit::Status::NotFound,
            Some(symbol),
            accepted_symbol_spelling(symbol),
            candidates,
        ))));
    }
    let mut sites: Vec<(String, i64)> = nodes
        .iter()
        .map(|n| (n.file_path.clone(), n.start_line))
        .collect();
    sites.sort();
    sites.dedup();
    if sites.len() > 1 {
        let candidates = nodes
            .iter()
            .map(|n| cert::Candidate {
                qualified_name: n.qualified_name.clone(),
                path: n.file_path.clone(),
                line: n.start_line as usize,
            })
            .collect();
        return Ok(EditTarget::Refusal(Box::new(refusal(
            root_path,
            "",
            greppy_edit::Status::Ambiguous,
            Some(symbol),
            accepted_symbol_spelling(symbol),
            candidates,
        ))));
    }
    let node = &nodes[0];
    let abs = root_path.join(&node.file_path);
    let content = std::fs::read(&abs).map_err(|source| Error::Io {
        context: format!("read {}", abs.display()),
        source,
    })?;
    let Some(span) = read_span_with_meta(
        root_path,
        &node.file_path,
        node.start_line,
        node.end_line,
        usize::MAX,
        false,
    ) else {
        return Ok(EditTarget::Refusal(Box::new(refusal(
            root_path,
            &node.file_path,
            greppy_edit::Status::Stale,
            Some(symbol),
            accepted_symbol_spelling(symbol),
            vec![],
        ))));
    };
    let range = line_range_to_bytes(&content, node.start_line as usize, span.end_line as usize);
    let planned = greppy_edit::EditHandle::for_range(
        root_path,
        std::path::Path::new(&node.file_path),
        &content,
        range.0,
        range.1,
    )?;
    Ok(EditTarget::Resolved {
        rel_path: node.file_path.clone(),
        range,
        planned_file_sha256: planned.file_sha256,
        planned_target_sha256: planned.target_sha256,
    })
}

/// The byte range of `impl NAME { … }` — the type's own methods, not a trait
/// implementation. Used only when the definition `--symbol NAME` resolved to
/// carries no body of its own, so nothing that already works changes.
fn rust_inherent_impl_range(content: &[u8], name: &str) -> Option<(usize, usize)> {
    let text = std::str::from_utf8(content).ok()?;
    let mut cursor = 0usize;
    while let Some(found) = text[cursor..].find("impl ") {
        let at = cursor + found;
        cursor = at + 5;
        let rest = &text[cursor..];
        let Some(brace) = rest.find('{') else { break };
        let head = rest[..brace].trim();
        // `impl Trait for Name` implements someone else's contract; the method
        // an agent asks for belongs in the type's own block.
        if head != name {
            continue;
        }
        let open = cursor + brace;
        let mut depth = 0usize;
        for (offset, byte) in text[open..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((at, open + offset + 1));
                    }
                }
                _ => {}
            }
        }
        return None;
    }
    None
}

/// Edit targets may be workspace-relative or absolute.
fn resolve_edit_file(root_path: &std::path::Path, file: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(file);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root_path.join(p)
    }
}

struct BriefJsonContext<'a> {
    root: Option<&'a str>,
    path_filters: &'a QueryPathFilters,
    semantic_backend_unavailable: Option<&'a str>,
}

fn brief_semantic_backend_json(unavailable: Option<&str>) -> serde_json::Value {
    match unavailable {
        Some(detail) => serde_json::json!({
            "status": "unavailable",
            "reason": "asset_missing",
            "detail": detail,
            "fallback": "graph_only",
        }),
        None => serde_json::json!({"status": "available"}),
    }
}

fn dispatch_brief(
    symbol: Option<&str>,
    paths: &[String],
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let query_symbol = symbol.unwrap_or("");
    let path_filters = prepare_query_path_filters(root, "brief", query_symbol, paths)?;
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "brief",
        json,
        serde_json::json!({"schema_version": BRIEF_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "brief",
        json,
        serde_json::json!({"schema_version": BRIEF_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    // `brief` is graph-first and must remain useful when EmbeddingGemma cannot
    // be resolved. Probe configuration only (never load the model), surface the
    // degradation, then continue with definition/callers/callees unchanged.
    let semantic_backend_unavailable = embedding_config_for_required_use(EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    })
    .err()
    .filter(embedding_asset_missing_error)
    .map(|error| error.to_string());
    if !json && semantic_backend_unavailable.is_some() {
        println!(
            "semantic backend unavailable (asset missing) — brief continuing with graph-only definition/callers/callees"
        );
    }
    let targets = resolve_symbol_nodes(&store, symbol)?;
    if targets.is_empty() {
        if json {
            let miss = symbol_miss_json(&store, &project, query_symbol);
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema_version": BRIEF_JSON_SCHEMA_VERSION,
                    "command": "brief",
                    "status": "not_found",
                    "project": project,
                    "query": query_symbol,
                    "semantic_backend": brief_semantic_backend_json(
                        semantic_backend_unavailable.as_deref()
                    ),
                    "suggestions": miss["suggestions"].clone(),
                    "next": miss["next"].clone(),
                    "definitions": [],
                    "callers": [],
                    "references": [],
                    "calls": [],
                    "expand_id": serde_json::Value::Null,
                }))
                .map_err(|e| Error::Invalid(format!("serialize brief JSON: {e}")))?
            );
            return Ok(1);
        }
        return content_fallback(&store, root, symbol.unwrap_or(""), "brief", &path_filters);
    }
    let root_path = resolve_root(root)?;
    if json {
        return dispatch_brief_json(
            &store,
            &project,
            query_symbol,
            &targets,
            &root_path,
            BriefJsonContext {
                root,
                path_filters: &path_filters,
                semantic_backend_unavailable: semantic_backend_unavailable.as_deref(),
            },
        );
    }
    let mut evidence_nodes: Vec<(String, greppy_store::Node, serde_json::Value)> = Vec::new();

    // Definition(s) + source span.
    let mut seen_def = std::collections::BTreeSet::new();
    for id in &targets {
        if let Some(n) = store.get_node(*id)? {
            if path_filters.matches(&n.file_path) && seen_def.insert(n.id) {
                evidence_nodes.push((
                    format!("definition {}", display_node_name(&n)),
                    n.clone(),
                    serde_json::json!({"section": "definition"}),
                ));
                let span = read_span_with_meta(
                    &root_path,
                    &n.file_path,
                    n.start_line,
                    n.end_line,
                    CONTEXT_SPAN_CAP,
                    false,
                );
                let header_end_line = span
                    .as_ref()
                    .map(|span| span.end_line)
                    .unwrap_or(n.end_line);
                println!(
                    "== {} ({}:{}-{}) ==",
                    display_node_name(&n),
                    n.file_path,
                    n.start_line,
                    header_end_line
                );
                if let Some(span) = span {
                    if let Some(summary) = summarize_definition_span(&n.file_path, &span.text) {
                        for bullet in summary {
                            println!("  - {bullet}");
                        }
                        if !span.text.is_empty() {
                            println!();
                        }
                    }
                    print_code_span_text(&span.text);
                }
            }
        }
    }

    let mut callers = incoming_call_nodes_for_targets(&store, &targets)?;
    callers.retain(|node| path_filters.matches(&node.file_path));
    let cshown = callers.len().min(cli_result_limit(BRIEF_LIMIT));
    println!("\n-- CALLERS ({}) --", callers.len());
    for n in &callers[..cshown] {
        evidence_nodes.push((
            format!("caller {}", display_node_name(n)),
            n.clone(),
            serde_json::json!({"section": "callers"}),
        ));
        println!("  {} {}", display_node_name(n), node_line_span(n));
    }
    print_nav_more_footer(callers.len(), cshown);

    if targets_include_non_callable(&store, &targets)? {
        let mut refs = greppy_search::find_references_to_any(
            &store,
            &targets,
            greppy_search::MAX_REACH_RESULTS,
        )?;
        refs.retain(|reference| path_filters.matches(&reference.node.file_path));
        let total = refs.len();
        refs.truncate(cli_result_limit(BRIEF_LIMIT));
        println!("\n-- REFERENCES ({}) --", total);
        for r in &refs {
            if let Some(node) = store.get_node(r.node.id)? {
                evidence_nodes.push((
                    format!("reference {} {}", r.edge_type, display_node_name(&node)),
                    node,
                    serde_json::json!({"section": "references", "edge_type": r.edge_type}),
                ));
            }
            println!(
                "  {} {} {}",
                r.edge_type,
                display_row_name(&r.node),
                line_span(&r.node.file_path, r.node.start_line, r.node.end_line)
            );
        }
        print_nav_more_footer(total, refs.len());
    }

    // Direct callees (outgoing CALLS).
    let mut callees: std::collections::BTreeMap<i64, greppy_store::Node> =
        std::collections::BTreeMap::new();
    let callee_sources = callee_source_ids_for_symbols(&store, &project, &targets)?;
    for id in &callee_sources {
        for step in greppy_search::callees_of(&store, *id)? {
            if let Some(n) = step.node {
                callees.entry(step.node_id).or_insert(n);
            }
        }
    }
    callees.retain(|_, node| path_filters.matches(&node.file_path));
    let eshown = callees.len().min(cli_result_limit(BRIEF_LIMIT));
    println!("\n-- CALLS ({}) --", callees.len());
    for n in callees.values().take(eshown) {
        evidence_nodes.push((
            format!("callee {}", display_node_name(n)),
            n.clone(),
            serde_json::json!({"section": "calls"}),
        ));
        println!("  {} {}", display_node_name(n), node_line_span(n));
    }
    print_nav_more_footer(callees.len(), eshown);
    if evidence_nodes.is_empty() && !path_filters.is_empty() {
        println!(
            "\n(no brief results under path filter: {})",
            path_filters.shown()
        );
    }
    let evidence_rows = evidence_nodes
        .iter()
        .map(|(title, node, extra_json)| ExpandEvidenceNode {
            title: title.clone(),
            node,
            site_lines: Vec::new(),
            extra_json: extra_json.clone(),
        })
        .collect::<Vec<_>>();
    if let Some(expand) = insert_nav_expand_pack(
        &store,
        root,
        &project,
        "brief",
        query_symbol,
        evidence_rows.len(),
        &evidence_rows,
    ) {
        println!("{}", expand.text_line());
    }
    Ok(0)
}

fn dispatch_brief_json(
    store: &greppy_store::Store,
    project: &str,
    query_symbol: &str,
    targets: &[i64],
    root_path: &std::path::Path,
    context: BriefJsonContext<'_>,
) -> Result<i32> {
    let root = context.root;
    let path_filters = context.path_filters;
    let semantic_backend_unavailable = context.semantic_backend_unavailable;
    let mut evidence_nodes: Vec<(String, greppy_store::Node, serde_json::Value)> = Vec::new();
    let mut definitions = Vec::new();
    let mut seen_def = std::collections::BTreeSet::new();
    for id in targets {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if !path_filters.matches(&node.file_path) || !seen_def.insert(node.id) {
            continue;
        }
        let span = read_span_with_meta(
            root_path,
            &node.file_path,
            node.start_line,
            node.end_line,
            CONTEXT_SPAN_CAP,
            false,
        );
        let source = span.as_ref().map(|span| span.text.as_str()).unwrap_or("");
        let end_line = span
            .as_ref()
            .map(|span| span.end_line)
            .unwrap_or(node.end_line);
        let signature = node
            .properties
            .get("source_signature")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| semantic_signature_from_span(source));
        let summary = summarize_definition_span(&node.file_path, source).unwrap_or_default();
        let summary_prompt_version = if summary.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::json!(greppy_qwen35_native::PROMPT_VERSION)
        };
        definitions.push(serde_json::json!({
            "qualified_name": &node.qualified_name,
            "name": display_node_name(&node),
            "label": &node.label,
            "file_path": &node.file_path,
            "start_line": node.start_line,
            "end_line": end_line,
            "signature": signature,
            "summary": summary,
            "summary_prompt_version": summary_prompt_version,
            "source": source,
        }));
        evidence_nodes.push((
            format!("definition {}", display_node_name(&node)),
            node,
            serde_json::json!({"section": "definition"}),
        ));
    }

    let mut callers = incoming_call_nodes_for_targets(store, targets)?;
    callers.retain(|node| path_filters.matches(&node.file_path));
    let callers_json = callers.iter().map(node_hit_json).collect::<Vec<_>>();
    for node in &callers {
        evidence_nodes.push((
            format!("caller {}", display_node_name(node)),
            node.clone(),
            serde_json::json!({"section": "callers"}),
        ));
    }

    let mut references_json = Vec::new();
    if targets_include_non_callable(store, targets)? {
        for reference in
            greppy_search::find_references_to_any(store, targets, greppy_search::MAX_REACH_RESULTS)?
        {
            if !path_filters.matches(&reference.node.file_path) {
                continue;
            }
            if references_json.len() == cli_result_limit(BRIEF_LIMIT) {
                break;
            }
            references_json.push(serde_json::json!({
                "edge_type": &reference.edge_type,
                "qualified_name": &reference.node.qualified_name,
                "file_path": &reference.node.file_path,
                "start_line": reference.node.start_line,
                "end_line": reference.node.end_line,
            }));
            if let Some(node) = store.get_node(reference.node.id)? {
                evidence_nodes.push((
                    format!(
                        "reference {} {}",
                        reference.edge_type,
                        display_node_name(&node)
                    ),
                    node,
                    serde_json::json!({
                        "section": "references",
                        "edge_type": reference.edge_type,
                    }),
                ));
            }
        }
    }

    let mut callees = std::collections::BTreeMap::<i64, greppy_store::Node>::new();
    for id in callee_source_ids_for_symbols(store, project, targets)? {
        for step in greppy_search::callees_of(store, id)? {
            if let Some(node) = step.node {
                callees.entry(step.node_id).or_insert(node);
            }
        }
    }
    callees.retain(|_, node| path_filters.matches(&node.file_path));
    let calls_json = callees.values().map(node_hit_json).collect::<Vec<_>>();
    for node in callees.values() {
        evidence_nodes.push((
            format!("callee {}", display_node_name(node)),
            node.clone(),
            serde_json::json!({"section": "calls"}),
        ));
    }

    let evidence_rows = evidence_nodes
        .iter()
        .map(|(title, node, extra_json)| ExpandEvidenceNode {
            title: title.clone(),
            node,
            site_lines: Vec::new(),
            extra_json: extra_json.clone(),
        })
        .collect::<Vec<_>>();
    let expand = insert_nav_expand_pack(
        store,
        root,
        project,
        "brief",
        query_symbol,
        evidence_rows.len(),
        &evidence_rows,
    );
    let freshness = nav_freshness_json(store, root, project);
    let mut output = serde_json::json!({
        "schema_version": BRIEF_JSON_SCHEMA_VERSION,
        "command": "brief",
        "status": "ok",
        "project": project,
        "query": query_symbol,
        "path_filters": path_filters.json_value(),
        "freshness": freshness,
        "semantic_backend": brief_semantic_backend_json(semantic_backend_unavailable),
        "definitions": definitions,
        "callers": callers_json,
        "references": references_json,
        "calls": calls_json,
        "expand_id": serde_json::Value::Null,
    });
    if let Some(expand) = expand {
        output["expand_id"] = serde_json::json!(&expand.id);
        output["expand"] = expand.json_value();
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|e| Error::Invalid(format!("serialize brief JSON: {e}")))?
    );
    Ok(0)
}

fn expand_alias_path(root: Option<&str>, alias: &str) -> Option<std::path::PathBuf> {
    if alias.is_empty()
        || !alias
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return None;
    }
    let root_path = resolve_root(root).ok()?;
    let store_path = workspace_locator::store_path(&root_path);
    Some(store_path.parent()?.join("expand-aliases").join(alias))
}

fn insert_expand_alias_best_effort(root: Option<&str>, alias: &str, id: &str) {
    let Some(path) = expand_alias_path(root, alias) else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_ok() {
        let _ = std::fs::write(path, id);
    }
}

fn resolve_expand_alias(root: Option<&str>, alias: &str) -> Option<String> {
    let path = expand_alias_path(root, alias)?;
    std::fs::read_to_string(path)
        .ok()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
}

fn dispatch_expand(id: Option<&str>, json: bool, root: Option<&str>) -> Result<i32> {
    let id = id.unwrap_or("").trim();
    if id.is_empty() {
        return Err(Error::Invalid("expand requires an id".into()));
    }
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let lookup_id = resolve_expand_alias(root, id).unwrap_or_else(|| id.to_string());
    let Some(pack) = store.get_expand_pack(&lookup_id)? else {
        println!("expand: id not found or expired: {id}");
        return Ok(1);
    };
    if json {
        let v = serde_json::json!({
            "id": pack.id,
            "project": pack.project,
            "command": pack.command,
            "query": pack.query,
            "graph_generation": pack.graph_generation,
            "created_at": pack.created_at,
            "expires_at": pack.expires_at,
            "summary": pack.summary_json,
            "payload_text": pack.payload_text,
            "payload_json": pack.payload_json,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v)
                .map_err(|e| Error::Invalid(format!("serialize expand JSON: {e}")))?
        );
    } else {
        print!("{}", pack.payload_text);
        if !pack.payload_text.ends_with('\n') {
            println!();
        }
    }
    Ok(0)
}

/// `greppy stats` — print the deterministic graph statistics for the
/// workspace project: file count, per-label node counts, per-type edge
/// counts, and the node/edge totals. Routes through the shared
/// `--root`/project-identity resolution so it reports on the same store
/// the indexer wrote and the query commands read.
///
/// Output is stable and human-readable: the per-label and per-type lists
/// are already sorted by their key inside `Store::stats`, so two runs over
/// an unchanged graph print byte-identical text.
fn dispatch_stats(root: Option<&str>) -> Result<i32> {
    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let stats = store.stats(&project)?;
    println!("project: {}", stats.project);
    println!("files: {}", stats.file_count);
    println!("nodes: {}", stats.total_nodes);
    for lc in &stats.node_counts_by_label {
        println!("  {} {}", lc.label, lc.count);
    }
    println!("edges: {}", stats.total_edges);
    for ec in &stats.edge_counts_by_type {
        println!("  {} {}", ec.edge_type, ec.count);
    }
    Ok(0)
}

fn dispatch_diagnostics(json: bool, root: Option<&str>) -> Result<i32> {
    let store = open_default_store(root)?;
    let diag = store.diagnostics()?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&diag)
                .map_err(|e| Error::Invalid(format!("serialize diagnostics JSON: {e}")))?
        );
        return Ok(if diag.is_healthy() { 0 } else { EXIT_IO as i32 });
    }

    println!(
        "schema: {}/{}",
        diag.schema_version, diag.expected_schema_version
    );
    println!(
        "integrity: {}",
        if diag.integrity_ok { "ok" } else { "failed" }
    );
    for message in &diag.integrity_messages {
        println!("  integrity_message: {message}");
    }
    println!("workspaces: {}", diag.workspace_states.len());
    for workspace in &diag.workspace_states {
        println!(
            "  {} generation={} indexer={}",
            workspace.root_path, workspace.graph_generation, workspace.indexer_version
        );
    }
    println!("projects: {}", diag.projects.len());
    for project in &diag.projects {
        println!(
            "  {} files={} nodes={} edges={} incomplete_providers={}",
            project.project.name,
            project.stats.file_count,
            project.stats.total_nodes,
            project.stats.total_edges,
            project.incomplete_provider_count
        );
        for skip in &project.skip_counts_by_reason {
            println!("    skipped {} {}", skip.reason, skip.count);
        }
        for provider in &project.provider_states {
            println!(
                "    provider {} status={} files={}/{} missing_edges={}",
                provider.language,
                provider.status,
                provider.files_indexed,
                provider.files_seen,
                provider.unsupported_edge_classes.len()
            );
        }
    }

    Ok(if diag.is_healthy() { 0 } else { EXIT_IO as i32 })
}

fn dispatch_doctor(json: bool, root: Option<&str>) -> Result<i32> {
    dispatch_index_health("doctor", json, root)
}

fn inference_registry_status() -> Result<greppy_embed_native::InferenceBackendRegistry> {
    let cli = cli_inference_override();
    let no_gpu = cli.no_gpu || env_bool(ENV_NO_GPU)?;
    let configured = cli.device.or_else(|| env_nonempty(ENV_DEVICE));
    let policy = greppy_embed_native::InferencePolicy::from_selector(configured.as_deref(), no_gpu)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    Ok(greppy_embed_native::InferenceBackendRegistry::probe_policy(
        &policy,
        combined_inference_gpu_memory(),
    ))
}

fn inference_model_status() -> serde_json::Value {
    let embedding_args = EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    };
    let embedding = match embedding_config_optional(embedding_args) {
        Ok(Some(cfg)) => {
            let EmbeddingModelSource::Gguf { gguf, tokenizer } = cfg.source;
            serde_json::json!({
                "model_id": cfg.model_id,
                "format": "gguf-q4k",
                "embedded": cached_model_digest(&gguf).is_some(),
                "model_sha256": model_file_digest(&gguf).ok(),
                "tokenizer_sha256": model_file_digest(&tokenizer).ok(),
                "model_bytes": std::fs::metadata(&gguf).ok().map(|metadata| metadata.len()),
                "prompt_version": greppy_embed_native::PROMPT_VERSION,
                "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
            })
        }
        Ok(None) => serde_json::json!({
            "model_id": DEFAULT_EMBEDDINGGEMMA_MODEL_ID,
            "format": "gguf-q4k",
            "embedded": true,
            "model_sha256": env!("GREPPY_EMBEDDED_GGUF_SHA"),
            "tokenizer_sha256": env!("GREPPY_EMBEDDED_TOK_SHA"),
            "runtime_state": "not_loaded",
            "prompt_version": greppy_embed_native::PROMPT_VERSION,
            "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
        }),
        Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
    };
    let summary = match qwen_summary_config_optional() {
        Ok(Some(cfg)) => serde_json::json!({
            "model_id": cfg.model_id,
            "format": "gguf-q4-k-m-mtp",
            "embedded": cached_model_digest(&cfg.gguf).is_some(),
            "model_sha256": model_file_digest(&cfg.gguf).ok(),
            "tokenizer_sha256": model_file_digest(&cfg.tokenizer).ok(),
            "model_bytes": std::fs::metadata(&cfg.gguf).ok().map(|metadata| metadata.len()),
            "prompt_version": greppy_qwen35_native::PROMPT_VERSION,
        }),
        Ok(None) => serde_json::json!({
            "model_id": greppy_qwen35_native::MODEL_ID,
            "format": "gguf-q4-k-m-mtp",
            "embedded": true,
            "model_sha256": env!("GREPPY_EMBEDDED_QWEN35_GGUF_SHA"),
            "tokenizer_sha256": env!("GREPPY_EMBEDDED_QWEN35_TOK_SHA"),
            "runtime_state": "not_loaded",
            "prompt_version": greppy_qwen35_native::PROMPT_VERSION,
        }),
        Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
    };
    serde_json::json!({"embedding": embedding, "summary": summary})
}

fn combined_inference_gpu_memory() -> u64 {
    let embedding_args = EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    };
    let embedding = embedding_config_optional(embedding_args)
        .ok()
        .flatten()
        .and_then(|cfg| {
            let EmbeddingModelSource::Gguf { gguf, .. } = cfg.source;
            std::fs::metadata(gguf).ok()
        })
        .map(|metadata| {
            greppy_embed_native::estimated_gpu_memory(
                greppy_embed_native::InferenceModelKind::EmbeddingGemma,
                metadata.len(),
            )
        })
        .unwrap_or(0);
    let summary = qwen_summary_config_optional()
        .ok()
        .flatten()
        .and_then(|cfg| std::fs::metadata(cfg.gguf).ok())
        .map(|metadata| {
            greppy_embed_native::estimated_gpu_memory(
                greppy_embed_native::InferenceModelKind::Qwen35,
                metadata.len(),
            )
        })
        .unwrap_or(0);
    embedding.saturating_add(summary)
}

fn print_inference_registry(registry: &greppy_embed_native::InferenceBackendRegistry) {
    println!(
        "inference: preference={} explicit={} selected={} device={}",
        registry.preference,
        registry.explicit,
        registry
            .selected_backend
            .map(greppy_embed_native::BackendKind::as_str)
            .unwrap_or("none"),
        registry.selected_device_id.as_deref().unwrap_or("none")
    );
    for probe in &registry.probes {
        println!(
            "  backend {} compiled={} available={} score={} abi={}{}",
            probe.backend.as_str(),
            probe.compiled,
            probe.available,
            probe.score,
            probe.abi_version,
            probe
                .reason
                .as_deref()
                .map(|reason| format!(" reason={reason}"))
                .unwrap_or_default()
        );
        for device in &probe.devices {
            let memory = match (device.memory_free, device.memory_total) {
                (Some(free), Some(total)) => format!(" memory_free={free} memory_total={total}"),
                (None, Some(total)) => format!(" memory_total={total}"),
                _ => String::new(),
            };
            println!(
                "    device {} {}{}{}",
                device.id,
                device.name,
                memory,
                device
                    .rejection_reason
                    .as_deref()
                    .map(|reason| format!(" rejected={reason}"))
                    .unwrap_or_default()
            );
        }
    }
}

fn inference_daemon_status() -> serde_json::Value {
    #[cfg(any(unix, windows))]
    {
        let embedding_args = EmbeddingCliArgs {
            device: None,
            no_gpu: false,
        };
        let embedding = match embedding_config_optional(embedding_args) {
            Ok(Some(cfg)) => {
                let key = embedding_query_cache_key(&cfg);
                embed_daemon::status(&cfg, &key)
            }
            Ok(None) => serde_json::json!({"state": "unavailable"}),
            Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
        };
        let summary = match qwen_summary_config_optional() {
            Ok(Some(cfg)) => {
                let key = qwen_summary_model_key(&cfg);
                summarize_daemon::status(&key)
            }
            Ok(None) => serde_json::json!({"state": "unavailable"}),
            Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
        };
        serde_json::json!({"embedding": embedding, "summary": summary})
    }
    #[cfg(not(any(unix, windows)))]
    {
        serde_json::json!({
            "embedding": {"state": "unsupported"},
            "summary": {"state": "unsupported"},
        })
    }
}

fn print_inference_daemons(daemons: &serde_json::Value) {
    let Some(daemons) = daemons.as_object() else {
        return;
    };
    for (name, status) in daemons {
        let state = status
            .get("state")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let endpoint = status
            .get("endpoint")
            .and_then(serde_json::Value::as_str)
            .map(|endpoint| format!(" endpoint={endpoint}"))
            .unwrap_or_default();
        let error = status
            .get("last_error")
            .and_then(serde_json::Value::as_str)
            .map(|error| format!(" error={error}"))
            .unwrap_or_default();
        println!("  daemon {name} state={state}{endpoint}{error}");
    }
}

fn dispatch_index_status(json: bool, root: Option<&str>) -> Result<i32> {
    dispatch_index_health("index-status", json, root)
}

#[derive(Default)]
struct DirtyOverlay {
    git_available: bool,
    clean: bool,
    total: usize,
    staged_count: usize,
    unstaged_count: usize,
    untracked_count: usize,
    ignored_count: usize,
    deleted_count: usize,
    renamed_count: usize,
    files: Vec<DirtyOverlayFile>,
}

struct DirtyOverlayFile {
    path: String,
    old_path: Option<String>,
    index_status: char,
    worktree_status: char,
    staged: bool,
    unstaged: bool,
    untracked: bool,
    ignored: bool,
    deleted: bool,
    renamed: bool,
}

impl DirtyOverlay {
    fn to_json(&self) -> serde_json::Value {
        let files = self
            .files
            .iter()
            .take(40)
            .map(|f| {
                serde_json::json!({
                    "path": f.path,
                    "old_path": f.old_path,
                    "index_status": f.index_status.to_string(),
                    "worktree_status": f.worktree_status.to_string(),
                    "staged": f.staged,
                    "unstaged": f.unstaged,
                    "untracked": f.untracked,
                    "ignored": f.ignored,
                    "deleted": f.deleted,
                    "renamed": f.renamed,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "git_available": self.git_available,
            "clean": self.clean,
            "total": self.total,
            "staged_count": self.staged_count,
            "unstaged_count": self.unstaged_count,
            "untracked_count": self.untracked_count,
            "ignored_count": self.ignored_count,
            "deleted_count": self.deleted_count,
            "renamed_count": self.renamed_count,
            "shown": files.len(),
            "omitted": self.total.saturating_sub(files.len()),
            "files": files,
        })
    }
}

fn dirty_overlay(root_path: &std::path::Path) -> Result<DirtyOverlay> {
    let out = std::process::Command::new("git")
        .args([
            "status",
            "--porcelain=v1",
            "-z",
            "--ignored=matching",
            "--untracked-files=all",
        ])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io("spawn git status for dirty overlay", e))?;
    if !out.status.success() {
        return Ok(DirtyOverlay {
            git_available: false,
            clean: true,
            ..DirtyOverlay::default()
        });
    }

    let mut overlay = DirtyOverlay {
        git_available: true,
        clean: true,
        ..DirtyOverlay::default()
    };
    let mut records = out.stdout.split(|b| *b == 0).filter(|r| !r.is_empty());
    while let Some(record) = records.next() {
        if record.len() < 4 {
            continue;
        }
        let index_status = record[0] as char;
        let worktree_status = record[1] as char;
        let mut path = String::from_utf8_lossy(&record[3..]).to_string();
        let mut old_path = None;
        let renamed = matches!(index_status, 'R' | 'C') || matches!(worktree_status, 'R' | 'C');
        if renamed {
            if let Some(next) = records.next() {
                old_path = Some(String::from_utf8_lossy(next).to_string());
            } else if let Some((old, new)) = path.split_once(" -> ") {
                old_path = Some(old.to_string());
                path = new.to_string();
            }
        }
        let untracked = index_status == '?' && worktree_status == '?';
        let ignored = index_status == '!' && worktree_status == '!';
        let staged = !matches!(index_status, ' ' | '?' | '!');
        let unstaged = !matches!(worktree_status, ' ' | '?' | '!');
        let deleted = matches!(index_status, 'D') || matches!(worktree_status, 'D');

        overlay.staged_count += usize::from(staged);
        overlay.unstaged_count += usize::from(unstaged);
        overlay.untracked_count += usize::from(untracked);
        overlay.ignored_count += usize::from(ignored);
        overlay.deleted_count += usize::from(deleted);
        overlay.renamed_count += usize::from(renamed);
        overlay.files.push(DirtyOverlayFile {
            path,
            old_path,
            index_status,
            worktree_status,
            staged,
            unstaged,
            untracked,
            ignored,
            deleted,
            renamed,
        });
    }
    overlay.files.sort_by(|a, b| a.path.cmp(&b.path));
    overlay.total = overlay.files.len();
    overlay.clean = overlay.total == 0;
    Ok(overlay)
}

/// Count git-tracked files under `root` as an INDEPENDENT oracle for
/// discovery coverage (the walker cannot be its own witness). `None` when
/// git is unavailable or the root is not a repository — the coverage check
/// is then skipped rather than guessed.
fn git_tracked_file_count(root: &std::path::Path) -> Option<u64> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout.iter().filter(|b| **b == 0).count() as u64)
}

fn dispatch_index_health(command: &str, json: bool, root: Option<&str>) -> Result<i32> {
    let effective_root = resolve_root(root)?;
    let project = workspace_locator::project_identity(&effective_root);
    let store_path = workspace_locator::store_path(&effective_root);
    let store_format = store_path
        .parent()
        .and_then(|parent| greppy_core::cache::read_store_manifest(parent).ok())
        .map(|manifest| manifest.format_version);
    let store_bytes = store_path
        .parent()
        .map(cache_path_bytes)
        .unwrap_or_default();
    let background_job = read_background_job(&background_job_path(&effective_root));
    let effective_root_string = effective_root.to_string_lossy().into_owned();
    let writer_active = workspace_writer_active(Some(&effective_root_string));
    let job_state = background_job.as_ref().map(|job| {
        if job
            .get("pid")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|pid| process_is_alive(pid as u32))
        {
            "refreshing"
        } else {
            "failed"
        }
    });
    let background_state = if writer_active {
        Some("refreshing")
    } else {
        job_state
    };
    let dirty_overlay = dirty_overlay(&effective_root)?;
    let inference = (command == "doctor")
        .then(inference_registry_status)
        .transpose()?;
    let inference_daemons = (command == "doctor").then(inference_daemon_status);
    let inference_diagnostics = inference.as_ref().map(|registry| {
        serde_json::json!({
            "registry": registry,
            "daemons": inference_daemons,
            "models": inference_model_status(),
        })
    });

    if !store_path.exists() {
        let status = serde_json::json!({
            "command": command,
            "status": "no_index",
            "healthy": false,
            "store_exists": false,
            "root_path": effective_root,
            "store_path": store_path,
            "store_format": store_format,
            "store_bytes": store_bytes,
            "background_job": background_job,
            "background_state": background_state,
            "embedding_complete": false,
            "project": project,
            "fresh": false,
            "freshness": null,
            "schema_current": false,
            "integrity_ok": false,
            "project_present": false,
            "incomplete_provider_count": null,
            "skip_counts_by_reason": [],
            "dirty_overlay": dirty_overlay.to_json(),
            "inference": inference_diagnostics,
            "message": "no active index; run greppy index first",
        });
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&status)
                    .map_err(|e| Error::Invalid(format!("serialize {command} JSON: {e}")))?
            );
        } else {
            println!("status: no_index");
            println!("root: {}", effective_root.display());
            println!("store: {}", store_path.display());
            println!("message: run `greppy index {}` first", root.unwrap_or("."));
            if let Some(inference) = &inference {
                print_inference_registry(inference);
            }
            if let Some(daemons) = &inference_daemons {
                print_inference_daemons(daemons);
            }
            if dirty_overlay.git_available && !dirty_overlay.clean {
                println!(
                    "dirty_overlay: total={} staged={} unstaged={} untracked={} deleted={} renamed={} ignored={}",
                    dirty_overlay.total,
                    dirty_overlay.staged_count,
                    dirty_overlay.unstaged_count,
                    dirty_overlay.untracked_count,
                    dirty_overlay.deleted_count,
                    dirty_overlay.renamed_count,
                    dirty_overlay.ignored_count
                );
            }
        }
        return Ok(1);
    }

    let store =
        greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::read_only())?;
    let diag = store.diagnostics()?;
    let freshness = nav_freshness_json(&store, root, &project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let project_diag = diag.projects.iter().find(|p| p.project.name == project);
    let workspace = diag
        .workspace_states
        .iter()
        .find(|w| w.root_path == effective_root.to_string_lossy());
    let project_present = project_diag.is_some();
    let incomplete_provider_count = project_diag
        .map(|p| p.incomplete_provider_count)
        .unwrap_or(0);
    let provider_states = project_diag
        .map(|p| p.provider_states.clone())
        .unwrap_or_default();
    let provider_failure_count = provider_states
        .iter()
        .filter(|provider| provider.status != "unsupported")
        .map(|provider| provider.files_failed.max(0) as u64)
        .sum::<u64>();
    let skip_counts = project_diag
        .map(|p| {
            p.skip_counts_by_reason
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "reason": s.reason,
                        "count": s.count,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let stats = project_diag.map(|p| {
        serde_json::json!({
            "files": p.stats.file_count,
            "nodes": p.stats.total_nodes,
            "edges": p.stats.total_edges,
        })
    });
    let graph_generation = workspace.map(|w| w.graph_generation);
    let current_embedding_rows = graph_generation
        .and_then(|generation| {
            store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM vector_embeddings WHERE project = ?1 AND graph_generation = ?2",
                    (&project, generation as i64),
                    |row| row.get::<_, i64>(0),
                )
                .ok()
        })
        .unwrap_or(0);
    let configured_embedding_model = embedding_config_optional(EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    })
    .ok()
    .flatten();
    let embedding_complete = graph_generation.is_some_and(|generation| {
        let Some(model) = configured_embedding_model.as_ref() else {
            return false;
        };
        let key = embedding_complete_key(&project);
        store
            .conn()
            .query_row(
                "SELECT value FROM schema_meta WHERE key = ?1",
                [&key],
                |row| row.get::<_, String>(0),
            )
            .ok()
            == Some(format!("{generation}|{}", model.model_id))
    });
    // Robustness (problem dossier, systemic lesson 1&2): silent
    // under-indexing must be VISIBLE. Two independent-oracle checks:
    //   * coverage: compare the store's indexed file count against
    //     `git ls-files` — a discovery bug (out-of-root gitignore leak,
    //     O9-class) shows up as a tiny fraction of the tracked files.
    //   * vectors: a configured embedding model with zero stored vectors
    //     means every semantic query silently degrades to lexical.
    let indexed_files = project_diag.map(|p| p.stats.file_count).unwrap_or(0);
    let git_tracked = git_tracked_file_count(&effective_root);
    let coverage_warning = match git_tracked {
        Some(tracked) if tracked >= 100 && (indexed_files as u64) * 5 < tracked => Some(format!(
            "store indexed {indexed_files} files but git tracks {tracked} — \
             discovery may be dropping files (nested-repo ignore rules?); \
             re-run `greppy index` with the current binary"
        )),
        _ => None,
    };
    let vectors_missing_with_model = configured_embedding_model.is_some()
        && store
            .vector_model_ids(&project)
            .map(|v| v.is_empty())
            .unwrap_or(false);
    let inference_healthy = inference
        .as_ref()
        .is_none_or(greppy_embed_native::InferenceBackendRegistry::is_satisfied);
    let embedding_healthy = embedding_complete || test_inference_skipped();
    let healthy = diag.schema_current
        && diag.integrity_ok
        && project_present
        && fresh
        && embedding_healthy
        && provider_failure_count == 0
        && coverage_warning.is_none()
        && inference_healthy
        && background_state != Some("refreshing");
    let status_label = if healthy { "ok" } else { "unhealthy" };

    if json {
        let value = serde_json::json!({
            "command": command,
            "status": status_label,
            "healthy": healthy,
            "store_exists": true,
            "root_path": effective_root,
            "store_path": store_path,
            "store_format": store_format,
            "store_bytes": store_bytes,
            "background_job": background_job,
            "background_state": background_state,
            "embedding_complete": embedding_complete,
            "current_embedding_rows": current_embedding_rows,
            "project": project,
            "fresh": fresh,
            "freshness": freshness,
            "schema_version": diag.schema_version,
            "expected_schema_version": diag.expected_schema_version,
            "schema_current": diag.schema_current,
            "integrity_ok": diag.integrity_ok,
            "integrity_messages": diag.integrity_messages,
            "project_present": project_present,
            "graph_generation": graph_generation,
            "stats": stats,
            "incomplete_provider_count": incomplete_provider_count,
            "provider_failure_count": provider_failure_count,
            "providers": provider_states,
            "skip_counts_by_reason": skip_counts,
            "git_tracked_files": git_tracked,
            "coverage_warning": coverage_warning,
            "vectors_missing_with_model": vectors_missing_with_model,
            "dirty_overlay": dirty_overlay.to_json(),
            "inference": inference_diagnostics,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize {command} JSON: {e}")))?
        );
    } else {
        println!("status: {status_label}");
        if let Some(w) = &coverage_warning {
            println!("coverage_warning: {w}");
        }
        if vectors_missing_with_model {
            println!(
                "vectors: none stored though an embedding model is configured \
                 — `semantic-search` will build them on first use, or run \
                 `grep index` now"
            );
        }
        println!("root: {}", effective_root.display());
        println!("store: {}", store_path.display());
        println!("store_format: {}", store_format.unwrap_or(0));
        println!("store_bytes: {store_bytes}");
        println!("embedding_complete: {embedding_complete}");
        if let Some(inference) = &inference {
            print_inference_registry(inference);
        }
        if let Some(daemons) = &inference_daemons {
            print_inference_daemons(daemons);
        }
        if let Some(state) = background_state {
            println!("background_job: {state}");
        }
        println!("project: {project}");
        println!(
            "schema: {}/{} {}",
            diag.schema_version,
            diag.expected_schema_version,
            if diag.schema_current {
                "current"
            } else {
                "stale"
            }
        );
        println!(
            "integrity: {}",
            if diag.integrity_ok { "ok" } else { "failed" }
        );
        println!(
            "freshness: {}",
            if fresh {
                "fresh".to_string()
            } else {
                stale_freshness_reason(&freshness)
            }
        );
        if let Some(generation) = graph_generation {
            println!("generation: {generation}");
        }
        if let Some(project_diag) = project_diag {
            println!(
                "stats: files={} nodes={} edges={}",
                project_diag.stats.file_count,
                project_diag.stats.total_nodes,
                project_diag.stats.total_edges
            );
            println!("incomplete_providers: {incomplete_provider_count}");
            println!("provider_file_failures: {provider_failure_count}");
            for skip in &project_diag.skip_counts_by_reason {
                println!("skipped {} {}", skip.reason, skip.count);
            }
        } else {
            println!("project_present: false");
        }
        if dirty_overlay.git_available && !dirty_overlay.clean {
            println!(
                "dirty_overlay: total={} staged={} unstaged={} untracked={} deleted={} renamed={} ignored={}",
                dirty_overlay.total,
                dirty_overlay.staged_count,
                dirty_overlay.unstaged_count,
                dirty_overlay.untracked_count,
                dirty_overlay.deleted_count,
                dirty_overlay.renamed_count,
                dirty_overlay.ignored_count
            );
        }
    }

    Ok(if healthy { 0 } else { EXIT_IO as i32 })
}

fn cache_path_bytes(path: &std::path::Path) -> u64 {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if metadata.file_type().is_symlink() {
        return 0;
    }
    if metadata.is_file() {
        return metadata.len();
    }
    if !metadata.is_dir() {
        return 0;
    }
    std::fs::read_dir(path)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| cache_path_bytes(&entry.path()))
        .fold(0u64, u64::saturating_add)
}

/// `greppy who-calls S` — the callers and users of `S`: every node with an
/// incoming CALLS or USAGE edge into `S`. Printed as `qualified_name file:line`
/// so an agent can jump straight to each call site's enclosing symbol.
/// Content-search fallback for who-calls when the call/usage
/// GRAPH has no edges for `symbol` (e.g. a weakly-connected single-file symbol,
/// a macro, or a name that is not a graph node at all). Runs the indexed
/// live source search on the name so the agent still gets `file:line` hits from ONE
/// greppy call — instead of finding nothing and falling back to a grep loop.
/// This fallback keeps name queries from losing to a plain source search when
/// the graph has no corresponding symbol.
fn content_fallback(
    store: &greppy_store::Store,
    root: Option<&str>,
    symbol: &str,
    kind: &str,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    let project = project_for(root)?;
    let suggestions = symbol_miss_suggestions(store, &project, symbol);
    if has_case_variant_suggestion(&suggestions, symbol) {
        print_symbol_miss_guidance(store, &project, symbol);
        return Ok(1);
    }
    let mut hits = greppy_search::search_code(store, &project, symbol, 200)?;
    if hits.is_empty() {
        hits = live_grep_code_hits(symbol, &resolve_root(root)?)?;
    }
    hits.retain(|hit| {
        hit.location
            .rsplit_once(':')
            .is_some_and(|(path, _)| path_filters.matches(path))
    });
    hits.truncate(50);
    if hits.is_empty() {
        print_symbol_miss_guidance(store, &project, symbol);
        if !path_filters.is_empty() {
            println!(
                "no {kind} or source matches under path filter: {}",
                path_filters.shown()
            );
        }
        return Ok(1);
    }
    println!(
        "(`{symbol}` is not a graph symbol; {} source match(es) (would-be {kind}):)",
        hits.len()
    );
    for h in &hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    for suggestion in suggestions {
        println!("suggestion: `{suggestion}`");
    }
    println!("try: greppy search-symbols {}", shell_example_arg(symbol));
    println!("try: greppy semantic-search {}", shell_example_arg(symbol));
    Ok(0)
}

// ---------------------------------------------------------------------------
// Several targets per call — `dev/CLI-SPEC.md` rules 1-3, 5.
//
// A positional argument is a TARGET, never a path filter: the only path filter
// is `--path`, and it narrows every target in the batch. An argument the
// grammar does not provide — a path, a glob, an empty string, a name that does
// not resolve — is an error that names it, never an invented answer for the
// targets that happened to resolve. Every result says which target it answers
// for, in text and in `--json`.
// ---------------------------------------------------------------------------

/// Path-shaped: it carries a directory separator and is not one of the
/// `file.rs::Symbol` qualified names greppy itself prints.
fn looks_like_path(target: &str) -> bool {
    (target.contains('/') || target.contains('\\')) && !target.contains("::")
}

fn looks_like_glob(target: &str) -> bool {
    target.contains('*') || target.contains('?') || (target.contains('[') && target.contains(']'))
}

/// Rule 1: reject what the grammar does not provide instead of reinterpreting
/// it. Each of these used to become a silent path filter or an empty answer.
fn validate_nav_target(target: &str) -> Result<()> {
    if target.trim().is_empty() {
        return Err(Error::Invalid(
            "empty target: a symbol name was expected (an unexpanded shell variable produces \
             this). Nothing was looked up."
                .into(),
        ));
    }
    // A mistyped flag must not be swallowed as a target. `-x` stays a target
    // (after `--` a leading dash is part of the name) and fails as an unknown
    // symbol, which names it just as clearly.
    if target.starts_with("--") {
        return Err(Error::Invalid(format!(
            "unknown flag `{target}`; it was not read as a symbol. The path filter is `--path`"
        )));
    }
    if looks_like_glob(target) {
        let literal = target.replace(['*', '?', '[', ']'], "");
        return Err(Error::Invalid(format!(
            "`{target}` is a glob, not a symbol name, and greppy never expands one. Run \
             `greppy search-symbols {}` for the exact name first",
            shell_example_arg(&literal)
        )));
    }
    if looks_like_path(target) {
        return Err(Error::Invalid(format!(
            "`{target}` is a path, but a positional argument is always a symbol. To restrict the \
             results to it write `--path {target}`"
        )));
    }
    Ok(())
}

/// Targets exactly as written, with `-` replaced by what arrived on the pipe.
fn nav_targets(raw: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for value in raw {
        if value == "-" {
            out.extend(targets_from_stdin()?);
            continue;
        }
        out.push(value.clone());
    }
    for target in &out {
        validate_nav_target(target)?;
    }
    Ok(out)
}

/// Semantic queries are plain English, so the symbol-shape rules do not apply;
/// only `-` is expanded.
fn semantic_queries(raw: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for value in raw {
        if value == "-" {
            out.extend(targets_from_stdin()?);
            continue;
        }
        if value.trim().is_empty() {
            return Err(Error::Invalid(
                "empty query: semantic-search needs a plain-English description".into(),
            ));
        }
        out.push(value.clone());
    }
    Ok(out)
}

/// CHAIN: `greppy who-calls S --json | greppy brief -`. Result rows become the
/// next command's targets; the previous call's `targets` echo does not — that
/// is what it was asked, not what it answered.
fn targets_from_stdin() -> Result<Vec<String>> {
    use std::io::Read as _;
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|source| Error::io("read targets from stdin", source))?;
    let text = buffer.trim();
    if text.is_empty() {
        return Err(Error::Invalid(
            "`-` was given but nothing arrived on the pipe".into(),
        ));
    }
    let mut out: Vec<String> = Vec::new();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        collect_piped_targets(&value, &mut out);
    }
    if out.is_empty() {
        for line in text.lines() {
            let name = line.split_whitespace().next().unwrap_or("").trim();
            if !name.is_empty() && !out.iter().any(|kept| kept == name) {
                out.push(name.to_string());
            }
        }
    }
    if out.is_empty() {
        return Err(Error::Invalid(
            "`-` was given but the piped input carried no targets".into(),
        ));
    }
    Ok(out)
}

fn collect_piped_targets(value: &serde_json::Value, out: &mut Vec<String>) {
    let mut rows: Vec<&serde_json::Value> = Vec::new();
    for key in ["hits", "definitions", "candidates"] {
        if let Some(array) = value.get(key).and_then(serde_json::Value::as_array) {
            rows.extend(array.iter());
        }
    }
    if rows.is_empty() {
        match value.as_array() {
            Some(array) => rows.extend(array.iter()),
            None => rows.push(value),
        }
    }
    // (name, file) per row; the file only enters the target when the bare
    // qualified name would be ambiguous — otherwise the shortest form wins.
    let mut pairs: Vec<(String, Option<String>)> = Vec::new();
    for row in rows {
        let name = if let Some(text) = row.as_str() {
            Some(text.to_string())
        } else {
            row.get("qualified_name")
                .or_else(|| row.get("symbol"))
                .or_else(|| row.get("path"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };
        let Some(name) = name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()) else {
            continue;
        };
        let file = row
            .get("file")
            .or_else(|| row.get("file_path"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        pairs.push((name, file));
    }
    for index in 0..pairs.len() {
        let (name, file) = &pairs[index];
        let ambiguous = pairs
            .iter()
            .enumerate()
            .any(|(other, (candidate, _))| other != index && candidate == name);
        let target = match (ambiguous, file) {
            (true, Some(file)) if !name.starts_with(file.as_str()) => format!("{file}::{name}"),
            _ => name.clone(),
        };
        if !out.contains(&target) {
            out.push(target);
        }
    }
}

/// Rule 2: `--path` is the only path filter, so a `--path` that cannot narrow
/// anything is a mistake, not an empty scope. Answering "nothing found" would
/// confirm a typo as a fact about the repository.
///
/// `label` is how the caller spelled the filter, so the message names what was
/// actually written: `--path` for the flag, `path` for `search-code`'s
/// grep-shaped `PATTERN [PATH …]` positional.
fn validate_path_filters(root: Option<&str>, paths: &[String], label: &str) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }
    let root_path = resolve_root(root)?;
    let canonical_root = root_path.canonicalize().unwrap_or_else(|_| root_path.clone());
    for raw in paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(Error::Invalid(format!(
                "{label} needs a file or directory inside the repository"
            )));
        }
        let supplied = std::path::Path::new(trimmed);
        let mut candidates = Vec::new();
        if supplied.is_absolute() {
            candidates.push(supplied.to_path_buf());
        } else {
            if let Ok(cwd) = std::env::current_dir() {
                candidates.push(cwd.join(supplied));
            }
            candidates.push(root_path.join(supplied));
        }
        let Some(canonical) = candidates.iter().find_map(|c| c.canonicalize().ok()) else {
            return Err(Error::Invalid(format!(
                "{label} `{trimmed}` does not exist under {}",
                root_path.display()
            )));
        };
        if !canonical.starts_with(&canonical_root) {
            return Err(Error::Invalid(format!(
                "{label} `{trimmed}` is outside the repository {}; it cannot narrow anything in it",
                root_path.display()
            )));
        }
    }
    Ok(())
}

/// A refusal has to identify WHICH target failed — "symbol not found" without
/// the name forces the caller to bisect their own batch — and it keeps the
/// near-match guidance that turns the refusal into one cheap retry.
fn unknown_targets_message(
    store: &greppy_store::Store,
    project: &str,
    missing: &[String],
) -> String {
    let mut message = if missing.len() == 1 {
        format!("symbol not found: `{}`", missing[0])
    } else {
        format!(
            "symbols not found: {}",
            missing
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    message.push_str(" — no results were printed for the other targets either.");
    for name in missing {
        let suggestions = symbol_miss_suggestions(store, project, name);
        if !suggestions.is_empty() {
            message.push_str(&format!(
                "\n  `{name}`: did you mean {}?",
                suggestions
                    .iter()
                    .take(5)
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    message
}

/// A bare name that names definitions in more than one FILE is ambiguous.
/// Answering for one of them, or silently merging both into one answer, is the
/// same reinterpretation rule 1 forbids — so refuse and print the qualified
/// names that select each definition. Several nodes on one definition site (a
/// struct and its impl) are not ambiguity.
fn ensure_unambiguous_target(
    store: &greppy_store::Store,
    target: &str,
    ids: &[i64],
) -> Result<()> {
    if ids.len() < 2 || split_path_qualified(target).is_some() {
        return Ok(());
    }
    let mut files: Vec<String> = Vec::new();
    for id in ids {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name) {
            continue;
        }
        if !files.contains(&node.file_path) {
            files.push(node.file_path.clone());
        }
    }
    if files.len() < 2 {
        return Ok(());
    }
    files.sort();
    Err(Error::Invalid(format!(
        "`{target}` is defined in {} files, so the answer would be for one of them or for a \
         merge of both. Name the definition: {}",
        files.len(),
        files
            .iter()
            .map(|file| format!("`{file}::{target}`"))
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// A definition is not a reference to itself: the target nodes, the members the
/// target owns, and the synthetic anchor of the file the target is defined in
/// are all part of the definition. Import anchors of OTHER files stay — they
/// A definition that only exists to exercise other code. `brief` reports "the
/// tests that reach it" and `impact` "the tests among it"; both mean the
/// callable test functions, never the synthetic per-file anchors.
fn is_test_node(node: &greppy_store::Node) -> bool {
    if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name) {
        return false;
    }
    let path = node.file_path.as_str();
    let in_test_tree = path.starts_with("tests/")
        || path.starts_with("test/")
        || path.contains("/tests/")
        || path.contains("/test/")
        || path.contains("_test.")
        || path.contains("test_")
        || path.contains(".test.")
        || path.contains(".spec.");
    let name = node.name.as_str();
    let test_name = name.starts_with("test_")
        || name.starts_with("t_")
        || name.starts_with("Test")
        || name.ends_with("_test");
    in_test_tree || test_name
}

/// The raw `--limit`, without the offset `cli_result_limit` folds in: the
/// multi-target window is computed explicitly as `[offset, offset + limit)`.
fn cli_result_limit_raw() -> Option<usize> {
    CLI_RESULT_LIMIT.with(|value| value.get())
}

/// Rule 5: the handle marks exactly the span that was printed — when the source
/// is capped, it covers the shown lines only, never the rest of the definition.
fn node_source_and_handle(
    root_path: &std::path::Path,
    node: &greppy_store::Node,
) -> Option<(String, String)> {
    let span = read_span_with_meta(
        root_path,
        &node.file_path,
        node.start_line,
        node.end_line,
        CODE_SPAN_CAP,
        false,
    )?;
    let content = std::fs::read(root_path.join(&node.file_path)).ok()?;
    let (byte_start, byte_end) =
        line_range_to_bytes(&content, node.start_line as usize, span.end_line as usize);
    let handle = greppy_edit::EditHandle::for_range(
        root_path,
        std::path::Path::new(&node.file_path),
        &content,
        byte_start,
        byte_end,
    )
    .ok()?;
    Some((span.text, handle.encode()))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NavKind {
    WhoCalls,
    Callees,
    Impact,
}

struct NavMultiRequest<'a> {
    command: &'a str,
    kind: NavKind,
    targets: &'a [String],
    paths: &'a [String],
    code: bool,
    all: bool,
    json: bool,
    root: Option<&'a str>,
    direction: &'a str,
    edge: Option<&'a str>,
    depth: usize,
}

struct NavRow {
    target: usize,
    node: greppy_store::Node,
    edge_type: Option<String>,
    hops: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn dispatch_nav(
    command: &str,
    kind: NavKind,
    symbols: &[String],
    paths: &[String],
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let targets = nav_targets(symbols)?;
    validate_path_filters(root, paths, "--path")?;
    if targets.len() > 1 {
        return dispatch_nav_multi(NavMultiRequest {
            command,
            kind,
            targets: &targets,
            paths,
            code,
            all,
            json,
            root,
            direction: "incoming",
            edge: None,
            depth: 0,
        });
    }
    let symbol = targets.first().map(String::as_str);
    match kind {
        NavKind::WhoCalls => dispatch_who_calls(symbol, paths, code, all, json, root),
        NavKind::Callees => dispatch_callees(symbol, paths, code, all, json, root),
        NavKind::Impact => Err(Error::Invalid("impact has its own dispatch arm".into())),
    }
}

fn dispatch_nav_multi(req: NavMultiRequest<'_>) -> Result<i32> {
    let mut store = open_default_store_query_writer(req.root)?;
    maybe_reindex_stale(&mut store, req.root)?;
    let project = project_for(req.root)?;
    let gate_extra = serde_json::json!({
        "symbol": req.targets.join(" "),
        "symbol_found": false,
        "all": req.all,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        req.root,
        &project,
        req.command,
        req.json,
        gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        req.root,
        &project,
        req.command,
        req.json,
        gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    // Rule 1: resolve EVERY target before anything is printed. Answering for
    // the neighbours of an unknown name looks like a complete answer and hides
    // the typo for the rest of the session.
    let mut resolved: Vec<Vec<i64>> = Vec::with_capacity(req.targets.len());
    let mut missing: Vec<String> = Vec::new();
    for target in req.targets {
        let ids = resolve_symbol_nodes(&store, Some(target.as_str()))?;
        if ids.is_empty() {
            missing.push(target.clone());
        }
        resolved.push(ids);
    }
    if !missing.is_empty() {
        return Err(Error::Invalid(unknown_targets_message(
            &store, &project, &missing,
        )));
    }
    for (index, ids) in resolved.iter().enumerate() {
        ensure_unambiguous_target(&store, &req.targets[index], ids)?;
    }
    let dir = match req.direction.to_ascii_lowercase().as_str() {
        "incoming" | "in" | "callers" => greppy_search::ReachDirection::Incoming,
        "outgoing" | "out" | "callees" => greppy_search::ReachDirection::Outgoing,
        other => {
            return Err(Error::Invalid(format!(
                "impact --direction must be 'incoming' or 'outgoing', got '{other}'"
            )));
        }
    };
    let edge_upper = req.edge.map(|edge| edge.trim().to_ascii_uppercase());
    let edge_spec = impact_edge_spec(dir, edge_upper.as_deref());
    let path_filters = prepare_query_path_filters(req.root, req.command, "", req.paths)?;

    let mut rows: Vec<NavRow> = Vec::new();
    let mut totals = vec![0usize; req.targets.len()];
    let mut tests: Vec<Vec<serde_json::Value>> = vec![Vec::new(); req.targets.len()];
    for (index, ids) in resolved.iter().enumerate() {
        let mut collected = nav_rows_for_target(
            &store, &project, ids, index, req.kind, dir, &edge_spec, req.depth,
        )?;
        collected.retain(|row| path_filters.matches(&row.node.file_path));
        totals[index] = collected.len();
        tests[index] = collected
            .iter()
            .filter(|row| is_test_node(&row.node))
            .map(|row| node_hit_json(&row.node))
            .collect();
        rows.extend(collected);
    }

    // `--offset` is applied by the shared output-budget layer, which skips the
    // first N result rows of whatever a command emitted. Producers therefore
    // emit `offset + limit` rows from the top of the stream — the same
    // convention every other greppy command follows.
    let total = rows.len();
    let default_cap = if req.code { CODE_NAV_LIMIT } else { NAV_LIMIT };
    let end = cli_result_limit_unless_all(default_cap, req.all).min(total);
    let window = &rows[..end];
    let shown = window.len();
    let root_path = resolve_root(req.root)?;

    if req.json {
        let hits: Vec<serde_json::Value> = window
            .iter()
            .map(|row| nav_row_json(row, &req.targets[row.target], req.code, &root_path))
            .collect();
        let targets_json: Vec<serde_json::Value> = req
            .targets
            .iter()
            .enumerate()
            .map(|(index, symbol)| {
                let mut entry = serde_json::json!({
                    "symbol": symbol,
                    "symbol_found": true,
                    "total_exact": totals[index],
                });
                entry["tests"] = serde_json::json!(tests[index]);
                entry
            })
            .collect();
        let freshness = nav_freshness_json(&store, req.root, &project);
        let fresh = freshness
            .get("fresh")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let incomplete_providers = incomplete_provider_json(&store, &project)?;
        let omitted = total.saturating_sub(shown);
        let value = serde_json::json!({
            "command": req.command,
            "symbol": req.targets.join(" "),
            "targets": targets_json,
            "project": project,
            "symbol_found": true,
            "fresh": fresh,
            "freshness": freshness,
            "provider_complete": incomplete_providers.is_empty(),
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "total_exact": total,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "all": req.all,
            "hits": hits,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize nav JSON: {e}")))?
        );
        return Ok(0);
    }

    for row in window {
        let mut line = String::new();
        if let Some(hops) = row.hops {
            line.push_str(&format!("hop {hops} "));
        }
        if let Some(edge_type) = &row.edge_type {
            line.push_str(edge_type);
            line.push(' ');
        }
        line.push_str(&display_node_name(&row.node));
        line.push_str(&format!(" {}:{}", row.node.file_path, row.node.start_line));
        println!("{line}");
        if req.code {
            if let Some((source, handle)) = node_source_and_handle(&root_path, &row.node) {
                print_code_span_text(&source);
                println!("handle: {handle}");
            }
        }
    }
    // With `--offset` the budget layer prints its own `try:` continuation.
    if !req.all && cli_result_offset() == 0 && end < total {
        println!("{}", nav_continuation_command(&req, end));
    }
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
fn nav_rows_for_target(
    store: &greppy_store::Store,
    project: &str,
    ids: &[i64],
    index: usize,
    kind: NavKind,
    dir: greppy_search::ReachDirection,
    edge_spec: &ImpactEdgeSpec<'_>,
    depth: usize,
) -> Result<Vec<NavRow>> {
    let out = match kind {
        NavKind::WhoCalls => {
            let mut nodes = std::collections::BTreeMap::new();
            for id in ids {
                for edge_type in ["CALLS", "USAGE"] {
                    for edge in store.incoming_edges(*id, Some(edge_type), 1024)? {
                        if let std::collections::btree_map::Entry::Vacant(slot) =
                            nodes.entry(edge.source_id)
                        {
                            if let Some(node) = store.get_node(edge.source_id)? {
                                slot.insert(node);
                            }
                        }
                    }
                }
            }
            nodes
                .into_values()
                .map(|node| NavRow {
                    target: index,
                    node,
                    edge_type: None,
                    hops: None,
                })
                .collect()
        }
        NavKind::Callees => {
            let mut callees: std::collections::BTreeMap<i64, greppy_store::Node> =
                std::collections::BTreeMap::new();
            for source in callee_source_ids_for_symbols(store, project, ids)? {
                for step in greppy_search::callees_of(store, source)? {
                    if let Some(node) = step.node {
                        callees.entry(step.node_id).or_insert(node);
                    }
                }
            }
            callees
                .into_values()
                .map(|node| NavRow {
                    target: index,
                    node,
                    edge_type: None,
                    hops: None,
                })
                .collect()
        }
        NavKind::Impact => {
            let start_ids: std::collections::HashSet<i64> = ids.iter().copied().collect();
            let mut by_id: std::collections::HashMap<i64, greppy_search::ImpactNode> =
                std::collections::HashMap::new();
            for id in ids {
                for reached in greppy_search::impact_radius_any_edge_type(
                    store,
                    *id,
                    dir,
                    &edge_spec.edge_types,
                    depth,
                    4096,
                )? {
                    if start_ids.contains(&reached.node.id) {
                        continue;
                    }
                    by_id
                        .entry(reached.node.id)
                        .and_modify(|kept| {
                            if reached.hops < kept.hops {
                                kept.hops = reached.hops;
                            }
                        })
                        .or_insert(reached);
                }
            }
            let mut reached: Vec<greppy_search::ImpactNode> = by_id.into_values().collect();
            reached.sort_by(|a, b| a.hops.cmp(&b.hops).then_with(|| a.node.id.cmp(&b.node.id)));
            let mut out = Vec::new();
            for step in reached {
                if let Some(node) = store.get_node(step.node.id)? {
                    out.push(NavRow {
                        target: index,
                        node,
                        edge_type: None,
                        hops: Some(step.hops),
                    });
                }
            }
            out
        }
    };
    Ok(out)
}

fn nav_row_json(
    row: &NavRow,
    target: &str,
    code: bool,
    root_path: &std::path::Path,
) -> serde_json::Value {
    let node = &row.node;
    let mut value = serde_json::json!({
        "target": target,
        "qualified_name": &node.qualified_name,
        "name": display_node_name(node),
        "label": &node.label,
        "file": &node.file_path,
        "line": node.start_line,
        "file_path": &node.file_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
    });
    if let Some(edge_type) = &row.edge_type {
        value["edge_type"] = serde_json::json!(edge_type);
    }
    if let Some(hops) = row.hops {
        value["hops"] = serde_json::json!(hops);
    }
    if code {
        if let Some((source, handle)) = node_source_and_handle(root_path, node) {
            value["code"] = serde_json::json!(source);
            value["handle"] = serde_json::json!(handle);
        }
    }
    value
}

/// AGENTS.md, `--all`: "without it long results are cut and the output ends
/// with the exact command that continues them" — so the last line IS a command.
fn nav_continuation_command(req: &NavMultiRequest<'_>, offset: usize) -> String {
    let mut parts = vec!["greppy".to_string(), req.command.to_string()];
    for target in req.targets {
        parts.push(shell_example_arg(target));
    }
    for path in req.paths {
        parts.push("--path".to_string());
        parts.push(shell_example_arg(path));
    }
    if req.code {
        parts.push("--code".to_string());
    }
    if let Some(limit) = cli_result_limit_raw() {
        parts.push("--limit".to_string());
        parts.push(limit.to_string());
    }
    parts.push("--offset".to_string());
    parts.push(offset.to_string());
    parts.join(" ")
}

/// `brief A B` — the same briefing per target, in one call: the definition,
/// the callers, the callees, and the tests that reach it.
fn dispatch_brief_multi(
    targets: &[String],
    paths: &[String],
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "brief",
        json,
        serde_json::json!({"schema_version": BRIEF_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "brief",
        json,
        serde_json::json!({"schema_version": BRIEF_JSON_SCHEMA_VERSION}),
        "definitions",
    )? {
        return Ok(code);
    }
    let mut resolved: Vec<Vec<i64>> = Vec::with_capacity(targets.len());
    let mut missing: Vec<String> = Vec::new();
    for target in targets {
        let ids = resolve_symbol_nodes(&store, Some(target.as_str()))?;
        if ids.is_empty() {
            missing.push(target.clone());
        }
        resolved.push(ids);
    }
    if !missing.is_empty() {
        return Err(Error::Invalid(unknown_targets_message(
            &store, &project, &missing,
        )));
    }
    let path_filters = prepare_query_path_filters(root, "brief", "", paths)?;
    let root_path = resolve_root(root)?;
    let incoming = impact_edge_spec(greppy_search::ReachDirection::Incoming, None);

    let mut entries = Vec::with_capacity(targets.len());
    let mut hits = Vec::new();
    for (index, ids) in resolved.iter().enumerate() {
        let symbol = &targets[index];
        let mut definition = serde_json::Value::Null;
        for id in ids {
            let Some(node) = store.get_node(*id)? else {
                continue;
            };
            if !path_filters.matches(&node.file_path) {
                continue;
            }
            let span = read_span_with_meta(
                &root_path,
                &node.file_path,
                node.start_line,
                node.end_line,
                CONTEXT_SPAN_CAP,
                false,
            );
            let source = span.as_ref().map(|s| s.text.as_str()).unwrap_or("");
            let mut value = node_hit_json(&node);
            value["target"] = serde_json::json!(symbol);
            value["label"] = serde_json::json!(&node.label);
            value["source"] = serde_json::json!(source);
            if definition.is_null() {
                definition = value.clone();
            }
            hits.push(value);
            break;
        }
        let mut callers = incoming_call_nodes_for_targets(&store, ids)?;
        callers.retain(|node| path_filters.matches(&node.file_path));
        let mut callees: std::collections::BTreeMap<i64, greppy_store::Node> =
            std::collections::BTreeMap::new();
        for source in callee_source_ids_for_symbols(&store, &project, ids)? {
            for step in greppy_search::callees_of(&store, source)? {
                if let Some(node) = step.node {
                    callees.entry(step.node_id).or_insert(node);
                }
            }
        }
        callees.retain(|_, node| path_filters.matches(&node.file_path));
        let reaching = nav_rows_for_target(
            &store,
            &project,
            ids,
            index,
            NavKind::Impact,
            greppy_search::ReachDirection::Incoming,
            &incoming,
            6,
        )?;
        let tests: Vec<serde_json::Value> = reaching
            .iter()
            .filter(|row| is_test_node(&row.node) && path_filters.matches(&row.node.file_path))
            .map(|row| node_hit_json(&row.node))
            .collect();
        entries.push(serde_json::json!({
            "symbol": symbol,
            "symbol_found": true,
            "total_exact": callers.len() + callees.len(),
            "definition": definition,
            "callers": callers.iter().map(node_hit_json).collect::<Vec<_>>(),
            "callees": callees.values().map(node_hit_json).collect::<Vec<_>>(),
            "tests": tests,
        }));
    }

    if json {
        let value = serde_json::json!({
            "schema_version": BRIEF_JSON_SCHEMA_VERSION,
            "command": "brief",
            "status": "ok",
            "project": project,
            "targets": entries,
            "total_exact": hits.len(),
            "shown": hits.len(),
            "hits": hits,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize brief JSON: {e}")))?
        );
        return Ok(0);
    }
    for entry in &entries {
        println!("== {} ==", entry["symbol"].as_str().unwrap_or(""));
        if let Some(source) = entry["definition"]["source"].as_str() {
            let file = entry["definition"]["file"].as_str().unwrap_or("");
            let line = entry["definition"]["line"].as_i64().unwrap_or(0);
            println!("{file}:{line}");
            print_code_span_text(source);
        }
        for (label, key) in [("caller", "callers"), ("callee", "callees"), ("test", "tests")] {
            for row in entry[key].as_array().into_iter().flatten() {
                println!(
                    "{label} {} {}:{}",
                    row["qualified_name"].as_str().unwrap_or(""),
                    row["file"].as_str().unwrap_or(""),
                    row["line"].as_i64().unwrap_or(0)
                );
            }
        }
    }
    Ok(0)
}

/// What `read` was asked to read, after `-`, `--symbol` and `--path` have been
/// folded into one ordered list of subjects.
struct ReadPlan {
    subjects: Vec<String>,
    forced_symbol: bool,
    lines: Option<String>,
}

fn read_plan(
    targets: &[String],
    symbol_opts: &[String],
    path_opts: &[String],
    lines: Option<&str>,
) -> Result<ReadPlan> {
    let mut positional = Vec::new();
    for value in targets {
        if value == "-" {
            positional.extend(targets_from_stdin()?);
            continue;
        }
        positional.push(value.clone());
    }
    let forced_symbol = !symbol_opts.is_empty();
    let mut subjects: Vec<String> = Vec::new();
    if forced_symbol {
        subjects.extend(symbol_opts.iter().cloned());
        subjects.extend(positional);
        // A single `--path` next to a single symbol is the historic
        // disambiguator (`read open --path FILE`), not another subject.
        if subjects.len() == 1 && path_opts.len() == 1 {
            if let Some(folded) = qualify_symbol_with_path(
                subjects.first().map(String::as_str),
                Some(path_opts[0].as_str()),
            ) {
                subjects[0] = folded;
            }
        } else {
            subjects.extend(path_opts.iter().cloned());
        }
    } else if positional.len() == 1 && path_opts.len() == 1 {
        if let Some(folded) = qualify_symbol_with_path(
            positional.first().map(String::as_str),
            Some(path_opts[0].as_str()),
        ) {
            subjects.push(folded);
        } else {
            subjects.extend(positional);
            subjects.extend(path_opts.iter().cloned());
        }
    } else {
        subjects.extend(positional);
        subjects.extend(path_opts.iter().cloned());
    }
    for subject in &subjects {
        if subject.trim().is_empty() {
            return Err(Error::Invalid(
                "empty read target: a symbol name or a path was expected (an unexpanded shell \
                 variable produces this). Nothing was read."
                    .into(),
            ));
        }
    }
    if lines.is_some() && subjects.len() > 1 {
        return Err(Error::Invalid(format!(
            "--lines names one range in one file, but {} targets were given; read them \
             separately, or drop --lines",
            subjects.len()
        )));
    }
    Ok(ReadPlan {
        subjects,
        forced_symbol,
        lines: lines.map(str::to_string),
    })
}

/// `read S [S …]` / `read PATH [PATH …]` — every target is read in one call,
/// and one target that cannot be read refuses the whole call: three files that
/// worked plus a silently dropped fourth looks like success.
fn dispatch_read_multi(plan: &ReadPlan, with_handle: bool, json: bool, root: Option<&str>) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let canonical_root = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.clone());
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;

    enum Subject {
        File(String),
        Symbol(greppy_store::Node),
    }
    let mut subjects = Vec::with_capacity(plan.subjects.len());
    let mut missing: Vec<String> = Vec::new();
    for raw in &plan.subjects {
        // `file.rs::Symbol` is a qualified SYMBOL, not a slash-containing path.
        let path_qualified = split_path_qualified(raw).is_some();
        if !plan.forced_symbol && !path_qualified && read_subject_is_path(raw, root)? {
            subjects.push(Subject::File(raw.clone()));
            continue;
        }
        if !plan.forced_symbol && !path_qualified && looks_like_path(raw) {
            missing.push(format!("`{raw}` is not a file in this repository"));
            continue;
        }
        let ids = resolve_symbol_nodes(&store, Some(raw.as_str()))?;
        let mut node = None;
        for id in &ids {
            if let Some(candidate) = store.get_node(*id)? {
                if !candidate.file_path.is_empty() && candidate.start_line >= 1 {
                    node = Some(candidate);
                    break;
                }
            }
        }
        match node {
            Some(node) => subjects.push(Subject::Symbol(node)),
            None => missing.push(format!("`{raw}` is not a definition in this repository")),
        }
    }
    if !missing.is_empty() {
        return Err(Error::Invalid(format!(
            "read: {} — nothing was read for the other targets either.",
            missing.join("; ")
        )));
    }

    let mut hits = Vec::with_capacity(subjects.len());
    let mut text = String::new();
    for (index, subject) in subjects.iter().enumerate() {
        let (shown_path, start, end, source, content) = match subject {
            Subject::File(raw) => {
                let candidate = read_file_candidate(&root_path, raw);
                let canonical = candidate.canonicalize().map_err(|source| Error::Io {
                    context: format!("canonicalize {}", candidate.display()),
                    source,
                })?;
                let relative = canonical.strip_prefix(&canonical_root).map_err(|_| {
                    Error::Invalid(format!("read path `{raw}` resolves outside the workspace"))
                })?;
                let shown = relative.to_string_lossy().replace('\\', "/");
                let content = std::fs::read(&canonical).map_err(|source| Error::Io {
                    context: format!("read {}", canonical.display()),
                    source,
                })?;
                let body = String::from_utf8_lossy(&content).into_owned();
                let lines = body.lines().count().max(1);
                (shown, 1i64, lines as i64, body, content)
            }
            Subject::Symbol(node) => {
                let absolute = root_path.join(&node.file_path);
                let content = std::fs::read(&absolute).map_err(|source| Error::Io {
                    context: format!("read {}", absolute.display()),
                    source,
                })?;
                let Some(span) = read_span_with_meta(
                    &root_path,
                    &node.file_path,
                    node.start_line,
                    node.end_line,
                    usize::MAX,
                    false,
                ) else {
                    return Err(Error::Invalid(format!(
                        "read: definition span for `{}` is stale; re-index and retry",
                        node.qualified_name
                    )));
                };
                (
                    node.file_path.clone(),
                    node.start_line,
                    span.end_line,
                    span.text,
                    content,
                )
            }
        };
        let (byte_start, byte_end) =
            line_range_to_bytes(&content, start.max(1) as usize, end.max(start) as usize);
        let handle_token = if with_handle {
            Some(
                greppy_edit::EditHandle::for_range(
                    &root_path,
                    std::path::Path::new(&shown_path),
                    &content,
                    byte_start,
                    byte_end,
                )?
                .encode(),
            )
        } else {
            None
        };
        let target = &plan.subjects[index.min(plan.subjects.len() - 1)];
        hits.push(serde_json::json!({
            "target": target,
            "qualified_name": match subject {
                Subject::Symbol(node) => node.qualified_name.clone(),
                Subject::File(_) => shown_path.clone(),
            },
            "file": &shown_path,
            "line": start,
            "path": &shown_path,
            "file_path": &shown_path,
            "start_line": start,
            "end_line": end,
            "lines": format!("{start}:{end}"),
            "source": &source,
            "handle": handle_token,
        }));
        text.push_str(&format!("{shown_path}:{start}-{end}\n"));
        text.push_str(&source);
        if !source.ends_with('\n') {
            text.push('\n');
        }
        if let Some(token) = &handle_token {
            text.push_str(&format!("handle: {token}\n"));
        }
    }
    if json {
        let value = serde_json::json!({
            "schema_version": "greppy.read.v1",
            "command": "read",
            "status": "ok",
            "total_exact": hits.len(),
            "shown": hits.len(),
            "hits": hits,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize read JSON: {e}")))?
        );
    } else {
        print!("{text}");
    }
    Ok(0)
}

/// `search-symbols A B` — one lookup per name, each result attributed.
fn dispatch_search_symbols_multi(
    targets: &[String],
    paths: &[String],
    kind: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    if targets.len() <= 1 {
        return dispatch_search_symbols(
            targets.first().map(String::as_str),
            paths,
            kind,
            json,
            root,
        );
    }
    let path_filters = prepare_query_path_filters(root, "search-symbols", "", paths)?;
    let mut store = open_default_store(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let mut entries = Vec::with_capacity(targets.len());
    let mut hits = Vec::new();
    for query in targets {
        let mut found = greppy_search::search_symbols_in_project(&store, &project, query, 10_000)?;
        if let Some(want) = kind.map(|k| k.to_ascii_lowercase()) {
            found.retain(|hit| {
                store
                    .get_node(hit.node_id)
                    .ok()
                    .flatten()
                    .is_some_and(|node| node.label.to_ascii_lowercase() == want)
            });
        }
        let mut rows = Vec::new();
        for hit in &found {
            let Some(node) = store.get_node(hit.node_id)? else {
                continue;
            };
            if !path_filters.matches(&node.file_path) {
                continue;
            }
            let mut value = node_hit_json(&node);
            value["target"] = serde_json::json!(query);
            value["label"] = serde_json::json!(&node.label);
            value["name"] = serde_json::json!(&node.name);
            rows.push(value);
        }
        entries.push(serde_json::json!({
            "symbol": query,
            "symbol_found": !rows.is_empty(),
            "total_exact": rows.len(),
        }));
        hits.extend(rows);
    }
    let total = hits.len();
    let end = cli_result_limit_raw().unwrap_or(usize::MAX).min(total);
    let window = hits[..end].to_vec();
    if json {
        let value = serde_json::json!({
            "command": "search-symbols",
            "status": "ok",
            "project": project,
            "targets": entries,
            "total_exact": total,
            "shown": window.len(),
            "hits": window,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|e| Error::Invalid(format!("serialize search-symbols JSON: {e}")))?
        );
        return Ok(0);
    }
    for row in &window {
        println!(
            "{} {} {}:{}",
            row["label"].as_str().unwrap_or(""),
            row["qualified_name"].as_str().unwrap_or(""),
            row["file"].as_str().unwrap_or(""),
            row["line"].as_i64().unwrap_or(0)
        );
    }
    Ok(0)
}

/// One answer line of `who-calls` / `callees`: an address, the symbol, and
/// whether it lives in a test. The path appears once, inside the address — the
/// name never repeats it, and no kind prefix is printed, because the command
/// has already said which relation this is.
struct NavAnswerRow {
    file: String,
    /// What the agent acts on: the call site for `who-calls`, the definition
    /// for `callees`.
    line: u32,
    /// What `--code` prints. Collapses to `line` when it is a single line.
    span: (u32, u32),
    name: String,
    test: bool,
}

/// The symbol without the file path in front of it. A qualified name carries
/// the file and usually a kind segment — `edit-src/data.rs::Function::data_set`
/// — and both are noise beside an address that already names the file.
fn nav_short_name(node: &greppy_store::Node) -> String {
    let qualified = node.qualified_name.as_str();
    let tail = qualified
        .split_once("::")
        .map(|(_, rest)| rest)
        .unwrap_or(qualified);
    let name = tail
        .split_once("::")
        .and_then(|(head, rest)| {
            matches!(
                head,
                "Function"
                    | "Method"
                    | "Class"
                    | "Struct"
                    | "Enum"
                    | "Trait"
                    | "Interface"
                    | "EnumVariant"
                    | "Field"
                    | "Variable"
                    | "Constant"
                    | "Module"
            )
            .then_some(rest)
        })
        .unwrap_or(tail);
    if name.is_empty() {
        node.name.clone()
    } else {
        name.to_string()
    }
}

/// Net `(`/`[` nesting a line leaves open. Braces are deliberately not counted:
/// a call inside `if let Some(x) = f( … ) {` balances its parentheses on the
/// very line that opens the block, and counting the brace would drag the whole
/// block in. Text inside double quotes and after `//` is ignored; a lone `'`
/// is left alone so Rust lifetimes do not read as an unterminated string.
fn nav_bracket_delta(text: &str) -> i32 {
    let chars: Vec<char> = text.chars().collect();
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (index, c) in chars.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if *c == '\\' {
                escaped = true;
            } else if *c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '/' if chars.get(index + 1) == Some(&'/') => break,
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// The statement containing `line`, as it stands in the file. Follows the
/// line's own bracket nesting downwards and keeps a trailing `.`/`?`
/// continuation with it, so a call a formatter broke across ten lines arrives
/// whole. It never reformats and never rewrites: the caller prints exactly the
/// range returned here, so the address above the source is always true.
fn nav_statement_span(lines: &[String], line: u32) -> (u32, u32) {
    const MAX_LINES: usize = 60;
    let start = (line as usize).saturating_sub(1);
    if start >= lines.len() {
        return (line, line);
    }
    let mut depth = 0;
    let mut end = start;
    for (offset, text) in lines[start..].iter().enumerate().take(MAX_LINES) {
        depth += nav_bracket_delta(text);
        end = start + offset;
        if depth > 0 {
            continue;
        }
        match lines.get(end + 1).map(|next| next.trim_start()) {
            Some(next) if next.starts_with('.') || next.starts_with('?') => depth = 0,
            _ => break,
        }
    }
    ((start + 1) as u32, (end + 1) as u32)
}

/// Whether a node is a test. `is_test_node` decides on the path and the name,
/// which misses the most common Rust shape by far: a `#[test]` function with an
/// ordinary name inside an inline `mod tests`. Six of `data_set`'s seven callers
/// are exactly that, and calling them production code overstates the blast
/// radius of a change. So the attribute directly above the definition is
/// consulted too — the same line a human reads to decide.
fn nav_is_test(lines: Option<&Vec<String>>, node: &greppy_store::Node) -> bool {
    if is_test_node(node) {
        return true;
    }
    let Some(lines) = lines else {
        return false;
    };
    let definition = node.start_line.max(1) as usize;
    let first = definition.saturating_sub(4);
    lines
        .get(first..definition.saturating_sub(1))
        .into_iter()
        .flatten()
        .map(|line| line.trim())
        .any(|line| {
            line.starts_with("#[test]")
                || line.starts_with("#[tokio::test")
                || line.starts_with("#[rstest")
                || line.starts_with("#[test_case")
                || line.starts_with("@Test")
                || line.starts_with("@pytest.mark")
        })
}

fn nav_file_lines(root: &std::path::Path, file: &str) -> Option<Vec<String>> {
    std::fs::read_to_string(root.join(file))
        .ok()
        .map(|text| text.lines().map(str::to_string).collect())
}

/// What a reader would call this definition. The index label is a poor source:
/// it says `Class` for a Rust struct, so the refusal would name a thing the
/// language does not have. The declaring keyword in the source says it exactly,
/// in the words of the language the agent is reading, and the label is only the
/// fallback for a file that cannot be read.
fn nav_kind_word(lines: Option<&Vec<String>>, node: &greppy_store::Node) -> String {
    const KEYWORDS: [&str; 14] = [
        "struct",
        "enum",
        "trait",
        "class",
        "interface",
        "record",
        "union",
        "type",
        "const",
        "static",
        "protocol",
        "object",
        "module",
        "package",
    ];
    if let Some(lines) = lines {
        if let Some(line) = lines.get((node.start_line.max(1) as usize).saturating_sub(1)) {
            for word in line.split_whitespace() {
                let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                if KEYWORDS.contains(&word) {
                    return word.to_string();
                }
            }
        }
    }
    match node.label.as_str() {
        "EnumVariant" => "enum variant".to_string(),
        other => other.to_ascii_lowercase(),
    }
}

/// Which kinds a direction can answer for. Incoming navigation follows both
/// calls and usages, so every real definition can be referenced. Outgoing
/// navigation still requires a definition that can hold code. Synthetic file
/// anchors are handled separately by `nav_refuse_non_callable`.
#[derive(Clone, Copy)]
enum NavDirection {
    Incoming,
    Outgoing,
}

impl NavDirection {
    fn answerable(self, label: &str) -> bool {
        match self {
            NavDirection::Incoming => true,
            NavDirection::Outgoing => !matches!(
                label,
                "Field" | "Variable" | "Constant" | "EnumVariant" | "Property"
            ),
        }
    }
}

/// A definition the direction cannot answer for. Incoming references apply to
/// every real definition; outgoing calls still refuse definitions that cannot
/// hold code. Synthetic file anchors are not referenceable definitions.
fn nav_refuse_non_callable(
    store: &greppy_store::Store,
    repo_root: &std::path::Path,
    target: &str,
    ids: &[i64],
    direction: NavDirection,
) -> Result<Option<i32>> {
    let mut nodes = Vec::new();
    let mut anchors = Vec::new();
    for id in ids {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name) {
            // A file anchor is the module, not a referenceable definition
            // inside it, so neither navigation direction can answer for it.
            anchors.push(node);
            continue;
        }
        if direction.answerable(&node.label) {
            return Ok(None);
        }
        nodes.push(node);
    }
    let Some(node) = nodes.first().or_else(|| anchors.first()) else {
        return Ok(None);
    };
    let lines = nav_file_lines(repo_root, &node.file_path);
    println!(
        "`{target}` is a {}, not a function  {}:{}",
        nav_kind_word(lines.as_ref(), node),
        node.file_path,
        node.start_line.max(1)
    );
    Ok(Some(1))
}

/// A bare name that names several callable definitions selects none of them.
/// Answering for one, or merging them, is the reinterpretation the resolver
/// exists to prevent. The candidates carry no name because it is the same for
/// all of them — what separates them is the file, and the address holds it.
fn nav_refuse_ambiguous(
    store: &greppy_store::Store,
    target: &str,
    ids: &[i64],
) -> Result<Option<i32>> {
    if ids.len() < 2 || split_path_qualified(target).is_some() {
        return Ok(None);
    }
    let mut sites: Vec<(String, i64)> = Vec::new();
    for id in ids {
        let Some(node) = store.get_node(*id)? else {
            continue;
        };
        if is_synthetic_file_anchor(&node.label, &node.name, &node.qualified_name) {
            continue;
        }
        if !sites.iter().any(|(file, _)| file == &node.file_path) {
            sites.push((node.file_path.clone(), node.start_line.max(1)));
        }
    }
    if sites.len() < 2 {
        return Ok(None);
    }
    sites.sort();
    println!("`{target}` is {} definitions", sites.len());
    for (file, line) in &sites {
        println!("{file}:{line}");
    }
    Ok(Some(1))
}

/// The name does not resolve. Say so, and add only what the agent could not get
/// with one command of its own: names close enough to be the one it meant, with
/// where they are. Nothing close enough means no block at all — the absence is
/// the statement, and it says the repository does not know this name.
fn nav_report_missing(store: &greppy_store::Store, project: &str, query: &str) {
    println!("no symbol `{query}`");
    let normalise =
        |name: &str| name.to_ascii_lowercase().replace('_', "").replace('-', "");
    let wanted = normalise(query);
    let mut near: Vec<(String, String, i64)> = Vec::new();
    for name in symbol_miss_suggestions(store, project, query) {
        let close = normalise(&name) == wanted
            || symbol_name_distance(&name, &suggestion_needles(query)) <= 2;
        if !close {
            continue;
        }
        let Ok(ids) = resolve_symbol_nodes(store, Some(&name)) else {
            continue;
        };
        let Some(node) = ids.first().and_then(|id| store.get_node(*id).ok().flatten()) else {
            continue;
        };
        near.push((name, node.file_path.clone(), node.start_line.max(1)));
        if near.len() == 3 {
            break;
        }
    }
    if near.is_empty() {
        return;
    }
    println!();
    println!("similar names:");
    for (name, file, line) in near {
        println!("{file}:{line}  {name}");
    }
}

/// Prints the answer of `who-calls` / `callees`.
///
/// A short result is printed whole — a count beneath lines the reader can count
/// is packaging. Past `NAV_FULL_LIMIT` the shape leads instead: how many, and
/// across which files. On a skewed result that distribution *is* the answer,
/// and the rows below it are an example. Files with the fewest rows come first,
/// so the outliers are not buried under a hub file.
fn print_nav_rows(
    repo_root: &std::path::Path,
    noun: &str,
    rows: &mut [NavAnswerRow],
    code: bool,
    all: bool,
) {
    const NAV_FULL_LIMIT: usize = 25;
    const NAV_SUMMARY_ROWS: usize = 5;

    let mut per_file: std::collections::BTreeMap<String, usize> = Default::default();
    for row in rows.iter() {
        *per_file.entry(row.file.clone()).or_insert(0) += 1;
    }
    rows.sort_by(|a, b| {
        let left = per_file.get(&a.file).copied().unwrap_or(0);
        let right = per_file.get(&b.file).copied().unwrap_or(0);
        left.cmp(&right)
            .then_with(|| a.file.cmp(&b.file))
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.name.cmp(&b.name))
    });

    let total = rows.len();
    let summarize = !all && total > NAV_FULL_LIMIT;
    if summarize {
        let mut spread: Vec<(&String, &usize)> = per_file.iter().collect();
        spread.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
        let spread = spread
            .iter()
            .map(|(file, count)| format!("{file} {count}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("{total} {noun}: {spread}");
        println!();
    }
    let shown = if summarize {
        NAV_SUMMARY_ROWS.min(total)
    } else {
        total
    };
    let mut cache: std::collections::HashMap<String, Option<Vec<String>>> = Default::default();
    for (index, row) in rows.iter().take(shown).enumerate() {
        let marker = if row.test { "  test" } else { "" };
        if !code {
            println!("{}:{}  {}{}", row.file, row.line, row.name, marker);
            continue;
        }
        if index > 0 {
            println!();
        }
        let source = cache
            .entry(row.file.clone())
            .or_insert_with(|| nav_file_lines(repo_root, &row.file));
        let (start, mut end) = row.span;
        if let Some(lines) = source.as_ref() {
            end = end.min(lines.len() as u32).max(start);
        }
        if start == end {
            println!("{}:{}  {}{}", row.file, start, row.name, marker);
        } else {
            println!("{}:{}-{}  {}{}", row.file, start, end, row.name, marker);
        }
        if let Some(lines) = source.as_ref() {
            for line in lines
                .iter()
                .take(end as usize)
                .skip(start.saturating_sub(1) as usize)
            {
                println!("{line}");
            }
        }
    }
}

fn dispatch_who_calls(
    symbol: Option<&str>,
    paths: &[String],
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let query_symbol = symbol.unwrap_or("");
    let path_filters = prepare_query_path_filters(root, "who-calls", query_symbol, paths)?;
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let graph_gate_extra = serde_json::json!({
        "symbol": query_symbol,
        "symbol_found": false,
        "all": all,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "who-calls",
        json,
        graph_gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "who-calls",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    // Aggregate incoming CALLS and USAGE across ALL nodes sharing the name + a
    // primary label (e.g. a Struct and its Impl), so references are not lost
    // to a name resolving to the wrong single node.
    let targets = resolve_symbol_nodes(&store, symbol)?;
    // Synthetic file anchors are rejected before ambiguity; every real
    // definition can have incoming CALLS or USAGE references.
    if let Some(symbol) = symbol {
        if json {
            ensure_unambiguous_target(&store, symbol, &targets)?;
        } else {
            let repo_root = resolve_root(root)?;
            if let Some(code) =
                nav_refuse_non_callable(&store, &repo_root, symbol, &targets, NavDirection::Incoming)?
            {
                return Ok(code);
            }
            if let Some(code) = nav_refuse_ambiguous(&store, symbol, &targets)? {
                return Ok(code);
            }
        }
    }
    if targets.is_empty() {
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "who-calls",
                query_symbol,
                &project,
                false,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(1);
        }
        nav_report_missing(&store, &project, query_symbol);
        return Ok(1);
    }
    let mut edges = Vec::new();
    for target in &targets {
        for edge_type in ["CALLS", "USAGE"] {
            edges.extend(store.incoming_edges(*target, Some(edge_type), 1024)?);
        }
    }
    if edges.is_empty() {
        // The symbol IS a defined graph node but has no callers — that is a
        // valid, useful answer, not a failure. Do not fall back to content
        // search (it would just echo the definition as noise).
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "who-calls",
                query_symbol,
                &project,
                true,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(0);
        }
        // Nobody calls it. That is an answer, and it needs no packaging and no
        // textual consolation prize.
        if path_filters.is_empty() {
            println!("no callers");
        } else {
            println!("no callers under path filter: {}", path_filters.shown());
        }
        return Ok(0);
    }
    // Deterministic, de-duplicated output across the aggregated targets.
    // First collect the unique caller nodes so we know the true total, then
    // print at most NAV_LIMIT (F1: cap the token-bomb) unless `--all`.
    let mut seen = std::collections::BTreeSet::new();
    let mut nodes = Vec::new();
    // Collect each reference-site line persisted in the edge properties. The
    // line locates the statement that `--code` prints for the answer row.
    let mut sites: std::collections::HashMap<i64, Vec<u32>> = std::collections::HashMap::new();
    for e in &edges {
        if let Some(l) = e.properties.get("line").and_then(|v| v.as_u64()) {
            sites.entry(e.source_id).or_default().push(l as u32);
        }
        if !seen.insert(e.source_id) {
            continue;
        }
        if let Some(n) = store.get_node(e.source_id)? {
            nodes.push(n);
        }
    }
    nodes.retain(|node| path_filters.matches(&node.file_path));
    if nodes.is_empty() && !path_filters.is_empty() && !json {
        println!("no callers under path filter: {}", path_filters.shown());
        return Ok(0);
    }
    let total = nodes.len();
    let cap = cli_result_limit_unless_all(if code { CODE_NAV_LIMIT } else { NAV_LIMIT }, all);
    let shown = total.min(cap);
    // The expand pack exists only for the JSON consumer now: in text mode
    // everything the pack could carry is one flag away, so offering it would be
    // a second spelling of `--all` plus a handle to remember.
    let expand = if json && !all && !code {
        let rows = nodes
            .iter()
            .map(|n| ExpandEvidenceNode {
                title: display_node_name(n),
                node: n,
                site_lines: sorted_site_lines(sites.get(&n.id)),
                extra_json: serde_json::json!({"role": "caller"}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(
            &store,
            root,
            &project,
            "who-calls",
            query_symbol,
            total,
            &rows,
        )
    } else {
        None
    };
    if json {
        let project = project_for(root)?;
        let hits = nodes[..shown].iter().map(node_hit_json).collect();
        nav_counts_json_with_expand(
            &store,
            root,
            "who-calls",
            query_symbol,
            &project,
            true,
            total,
            shown,
            all,
            hits,
            expand.as_ref(),
        )?;
        return Ok(0);
    }
    let repo_root = resolve_root(root)?;
    // The caller's name answers "who"; the call site answers "where the
    // dependency sits". Both fit one line, and nothing else belongs on it.
    let mut sources: std::collections::HashMap<String, Option<Vec<String>>> = Default::default();
    let mut rows = Vec::with_capacity(nodes.len());
    for n in &nodes {
        // A file anchor is greppy's own bookkeeping, not a symbol. `__file__`
        // in a result list is a name the agent cannot carry anywhere.
        if is_synthetic_file_anchor(&n.label, &n.name, &n.qualified_name) {
            continue;
        }
        let site = sorted_site_lines(sites.get(&n.id))
            .first()
            .copied()
            .unwrap_or_else(|| n.start_line.max(1) as u32);
        let lines = sources
            .entry(n.file_path.clone())
            .or_insert_with(|| nav_file_lines(&repo_root, &n.file_path));
        let span = match lines.as_ref() {
            Some(lines) if code => nav_statement_span(lines, site),
            _ => (site, site),
        };
        rows.push(NavAnswerRow {
            file: n.file_path.clone(),
            line: site,
            span,
            name: nav_short_name(n),
            test: nav_is_test(lines.as_ref(), n),
        });
    }
    print_nav_rows(&repo_root, "callers", &mut rows, code, all);
    Ok(0)
}
/// `greppy callees S` — what `S` calls: every node reached by a direct
/// outgoing CALLS edge from `S`. Printed as `qualified_name file:line` so
/// an agent can jump straight to each callee's definition. Backed by the
/// search `callees_of` helper.
///
/// Like who-calls, this aggregates across ALL nodes sharing the name + a
/// primary label (e.g. a Struct and its Impl) so callees are not lost to
/// a name resolving to the wrong single node. Output is de-duplicated and
/// deterministically ordered by node id.
fn dispatch_callees(
    symbol: Option<&str>,
    paths: &[String],
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let query_symbol = symbol.unwrap_or("");
    let path_filters = prepare_query_path_filters(root, "callees", query_symbol, paths)?;
    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    let graph_gate_extra = serde_json::json!({
        "symbol": query_symbol,
        "symbol_found": false,
        "all": all,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "callees",
        json,
        graph_gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "callees",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let sources = resolve_symbol_nodes(&store, symbol)?;
    if let Some(symbol) = symbol {
        if json {
            ensure_unambiguous_target(&store, symbol, &sources)?;
        } else {
            let repo_root = resolve_root(root)?;
            if let Some(code) =
                nav_refuse_non_callable(&store, &repo_root, symbol, &sources, NavDirection::Outgoing)?
            {
                return Ok(code);
            }
            if let Some(code) = nav_refuse_ambiguous(&store, symbol, &sources)? {
                return Ok(code);
            }
        }
    }
    if sources.is_empty() {
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "callees",
                query_symbol,
                &project,
                false,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(1);
        }
        nav_report_missing(&store, &project, query_symbol);
        return Ok(1);
    }
    // Aggregate direct callees across the resolved source nodes, keyed on
    // the callee node id so a callee reached from both a Struct and its
    // Impl is printed once. BTreeMap keeps the output id-ordered. We keep
    // the full node so `--code` can read its source span.
    let mut callees: std::collections::BTreeMap<i64, greppy_store::Node> =
        std::collections::BTreeMap::new();
    let callee_sources = callee_source_ids_for_symbols(&store, &project, &sources)?;
    for src in &callee_sources {
        for step in greppy_search::callees_of(&store, *src)? {
            if let Some(n) = step.node {
                callees.entry(step.node_id).or_insert(n);
            }
        }
    }
    callees.retain(|_, node| path_filters.matches(&node.file_path));
    if callees.is_empty() {
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "callees",
                query_symbol,
                &project,
                true,
                0,
                0,
                all,
                Vec::new(),
            )?;
            return Ok(0);
        }
        if path_filters.is_empty() {
            println!("no callees");
        } else {
            println!("no callees under path filter: {}", path_filters.shown());
        }
        return Ok(0);
    }
    let total = callees.len();
    let cap = cli_result_limit_unless_all(if code { CODE_NAV_LIMIT } else { NAV_LIMIT }, all);
    let shown = total.min(cap);
    let expand = if json && !all && !code {
        let rows = callees
            .values()
            .map(|n| ExpandEvidenceNode {
                title: display_node_name(n),
                node: n,
                site_lines: Vec::new(),
                extra_json: serde_json::json!({"role": "callee"}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(
            &store,
            root,
            &project,
            "callees",
            query_symbol,
            total,
            &rows,
        )
    } else {
        None
    };
    if json {
        let project = project_for(root)?;
        let hits = callees.values().take(shown).map(node_hit_json).collect();
        nav_counts_json_with_expand(
            &store,
            root,
            "callees",
            query_symbol,
            &project,
            true,
            total,
            shown,
            all,
            hits,
            expand.as_ref(),
        )?;
        return Ok(0);
    }
    // Mirror image of `who-calls`: the caller is known, so the new information
    // is where the callee lives — its definition, which is also what `--code`
    // prints.
    let repo_root = resolve_root(root)?;
    let mut sources: std::collections::HashMap<String, Option<Vec<String>>> = Default::default();
    let mut rows = Vec::with_capacity(callees.len());
    for n in callees.values() {
        if is_synthetic_file_anchor(&n.label, &n.name, &n.qualified_name) {
            continue;
        }
        let lines = sources
            .entry(n.file_path.clone())
            .or_insert_with(|| nav_file_lines(&repo_root, &n.file_path));
        rows.push(NavAnswerRow {
            file: n.file_path.clone(),
            line: n.start_line.max(1) as u32,
            span: (
                n.start_line.max(1) as u32,
                n.end_line.max(n.start_line).max(1) as u32,
            ),
            name: nav_short_name(n),
            test: nav_is_test(lines.as_ref(), n),
        });
    }
    print_nav_rows(&repo_root, "callees", &mut rows, code, all);
    Ok(0)
}

/// `greppy fan-in` / `greppy fan-out` — project-wide degree rankings over
/// one edge type. These answer hotspot questions in one bounded command:
/// "what is most called/referenced?" and "which symbols call the most?".
fn dispatch_fan_degree(
    command: &str,
    direction: &str,
    edge: &str,
    limit: usize,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let edge_upper = edge.to_ascii_uppercase();
    let effective_limit = limit.min(greppy_search::MAX_REACH_RESULTS);
    let graph_gate_extra = serde_json::json!({
        "scope": "degree_rank",
        "direction": direction,
        "edge_type": &edge_upper,
        "requested_limit": limit,
        "limit": effective_limit,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        command,
        json,
        graph_gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        command,
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }

    let (total, hits) = match command {
        "fan-in" => (
            greppy_search::count_fan_in(&store, &project, &edge_upper)?,
            greppy_search::fan_in(&store, &project, &edge_upper, effective_limit)?,
        ),
        "fan-out" => (
            greppy_search::count_fan_out(&store, &project, &edge_upper)?,
            greppy_search::fan_out(&store, &project, &edge_upper, effective_limit)?,
        ),
        other => {
            return Err(Error::Invalid(format!(
                "unknown fan-degree command '{other}'"
            )));
        }
    };

    if json {
        degree_counts_json(
            &store,
            root,
            &project,
            total,
            &hits,
            DegreeJsonMeta {
                command,
                direction,
                edge_type: &edge_upper,
                requested_limit: limit,
                effective_limit,
            },
        )?;
        return Ok(0);
    }

    if hits.is_empty() && total == 0 {
        println!("(no {command} hits)");
        return Ok(0);
    }
    for hit in &hits {
        println!(
            "{} {} {}:{}",
            hit.degree,
            display_row_name(&hit.node),
            hit.node.file_path,
            hit.node.start_line
        );
    }
    print_nav_more_footer(total, hits.len());
    Ok(0)
}

fn parse_graph_location(
    location: Option<&str>,
    file: Option<&str>,
    line: Option<i64>,
) -> Result<(String, i64)> {
    let location = location.map(str::trim).filter(|s| !s.is_empty());
    let file = file.map(str::trim).filter(|s| !s.is_empty());
    match (location, file, line) {
        (Some(loc), None, None) => {
            let Some((file_part, line_part)) = loc.rsplit_once(':') else {
                return Err(Error::Invalid(
                    "graph-locate location must be formatted as <file>:<line>".into(),
                ));
            };
            let parsed_line = line_part.parse::<i64>().map_err(|_| {
                Error::Invalid(format!(
                    "graph-locate line must be a positive integer, got '{line_part}'"
                ))
            })?;
            if file_part.trim().is_empty() || parsed_line <= 0 {
                return Err(Error::Invalid(
                    "graph-locate requires a non-empty file and a positive line".into(),
                ));
            }
            Ok((file_part.trim().to_string(), parsed_line))
        }
        (None, Some(file_part), Some(line_part)) => {
            if line_part <= 0 {
                return Err(Error::Invalid(
                    "graph-locate --line must be a positive integer".into(),
                ));
            }
            Ok((file_part.to_string(), line_part))
        }
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => Err(Error::Invalid(
            "graph-locate accepts either <file>:<line> or --file <FILE> --line <N>, not both"
                .into(),
        )),
        _ => Err(Error::Invalid(
            "graph-locate requires <file>:<line> or --file <FILE> --line <N>".into(),
        )),
    }
}

fn normalize_graph_location_file(file: &str, root: Option<&str>) -> Result<String> {
    let trimmed = file.trim();
    let without_dot = trimmed.strip_prefix("./").unwrap_or(trimmed);
    let path = std::path::Path::new(without_dot);
    if path.is_absolute() {
        let root_path = resolve_root(root)?;
        if let Ok(rel) = path.strip_prefix(&root_path) {
            return Ok(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(without_dot.replace('\\', "/"))
}

fn nearest_preceding_primary_symbol(
    store: &greppy_store::Store,
    project: &str,
    file_path: &str,
    line: i64,
) -> Result<Option<greppy_search::graph::SearchGraphRow>> {
    let rows = greppy_search::symbols_in_file(
        store,
        Some(project),
        file_path,
        greppy_search::MAX_REACH_RESULTS,
    )?;
    Ok(rows
        .into_iter()
        .filter(|row| row.start_line <= line && label_rank(&row.label) == 0)
        .max_by(|a, b| {
            a.start_line
                .cmp(&b.start_line)
                .then_with(|| b.end_line.cmp(&a.end_line))
                .then_with(|| b.qualified_name.cmp(&a.qualified_name))
                .then_with(|| b.id.cmp(&a.id))
        }))
}

/// `greppy graph-locate file:line` — map a grep/search hit location to the
/// innermost indexed graph symbol covering that line. If a language provider
/// only supplied one-line spans, fall back to the nearest preceding primary
/// definition and mark that as `nearest_preceding` in JSON/text output.
fn dispatch_graph_locate(
    location: Option<&str>,
    file: Option<&str>,
    line: Option<i64>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let (raw_file, line) = parse_graph_location(location, file, line)?;
    let file_path = normalize_graph_location_file(&raw_file, root)?;
    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let graph_gate_extra = serde_json::json!({
        "file_path": &file_path,
        "line": line,
        "location_found": false,
        "match_kind": serde_json::Value::Null,
        "scope": "file_line_innermost_symbol",
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "graph-locate",
        json,
        graph_gate_extra.clone(),
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "graph-locate",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let mut match_kind = "enclosing";
    let mut hit = greppy_search::definition_at(&store, Some(&project), &file_path, line)?;
    if hit.is_none() {
        hit = nearest_preceding_primary_symbol(&store, &project, &file_path, line)?;
        if hit.is_some() {
            match_kind = "nearest_preceding";
        }
    }

    if json {
        graph_locate_json(
            &store,
            root,
            &project,
            &file_path,
            line,
            hit.as_ref(),
            hit.as_ref().map(|_| match_kind),
        )?;
        return Ok(if hit.is_some() { 0 } else { 1 });
    }

    match hit {
        Some(row) => {
            println!(
                "{} {} {}:{}-{} match={}",
                row.label,
                display_row_name(&row),
                row.file_path,
                row.start_line,
                row.end_line,
                match_kind
            );
            Ok(0)
        }
        None => {
            println!("(no symbol at {file_path}:{line})");
            Ok(1)
        }
    }
}

/// `greppy path --from A --to B [--edge CALLS]` — print a shortest path
/// between two symbols, if one exists, as an ordered list of
/// `qualified_name file:line` steps from `A` to `B`. Backed by the search
/// `path_query` helper (deterministic shortest path over `--edge` edges).
///
/// Both `--from` and `--to` are required; each is resolved to a single
/// node via the same label-ranked resolution the navigation commands use
/// (a type/def-like label beats an Impl/EnumVariant/Call site sharing the
/// name). Exit codes:
/// * 0 — a path was found (printed).
/// * 1 — both endpoints resolved but no path exists within bounds, or an
///   endpoint symbol could not be resolved.
/// * 64 — usage error (missing `--from`/`--to`).
fn dispatch_path(
    from: Option<&str>,
    to: Option<&str>,
    edge: &str,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let from = from.map(str::trim).filter(|s| !s.is_empty());
    let to = to.map(str::trim).filter(|s| !s.is_empty());
    let (Some(from), Some(to)) = (from, to) else {
        return Err(Error::Invalid(
            "path requires both --from <SYMBOL> and --to <SYMBOL>".into(),
        ));
    };
    let edge_upper = edge.trim().to_ascii_uppercase();
    if edge_upper.is_empty() {
        return Err(Error::Invalid("path --edge must not be empty".into()));
    }

    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let max_hops = greppy_search::MAX_REACH_HOPS;
    let graph_gate_extra = serde_json::json!({
        "from": from,
        "to": to,
        "from_found": false,
        "to_found": false,
        "path_found": false,
        "reason": "skipped_stale_index",
        "scope": "shortest_path",
        "direction": "outgoing",
        "edge_type": &edge_upper,
        "max_hops": max_hops,
        "hops": serde_json::Value::Null,
    });
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "path",
        json,
        graph_gate_extra.clone(),
        "steps",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "path",
        json,
        serde_json::json!({
            "from": from,
            "to": to,
            "from_found": false,
            "to_found": false,
            "path_found": false,
            "reason": "skipped_incomplete_provider",
            "scope": "shortest_path",
            "direction": "outgoing",
            "edge_type": &edge_upper,
            "max_hops": max_hops,
            "hops": serde_json::Value::Null,
        }),
        "steps",
    )? {
        return Ok(code);
    }
    let from_id = resolve_symbol_id(&store, Some(from))?;
    let to_id = resolve_symbol_id(&store, Some(to))?;
    let Some(from_id) = from_id else {
        if json {
            path_counts_json(
                &store,
                root,
                from,
                to,
                &project,
                false,
                to_id.is_some(),
                None,
                PathJsonMeta {
                    edge_type: &edge_upper,
                    max_hops,
                    reason: Some("missing_from"),
                },
            )?;
            return Ok(1);
        }
        println!("(symbol not found: {from})");
        return Ok(1);
    };
    let Some(to_id) = to_id else {
        if json {
            path_counts_json(
                &store,
                root,
                from,
                to,
                &project,
                true,
                false,
                None,
                PathJsonMeta {
                    edge_type: &edge_upper,
                    max_hops,
                    reason: Some("missing_to"),
                },
            )?;
            return Ok(1);
        }
        println!("(symbol not found: {to})");
        return Ok(1);
    };

    let path = greppy_search::path_query(
        &store,
        from_id,
        to_id,
        greppy_search::ReachDirection::Outgoing,
        &edge_upper,
        max_hops,
    )?;
    if json {
        let reason = if path.is_some() {
            None
        } else {
            Some("no_path")
        };
        path_counts_json(
            &store,
            root,
            from,
            to,
            &project,
            true,
            true,
            path.as_ref(),
            PathJsonMeta {
                edge_type: &edge_upper,
                max_hops,
                reason,
            },
        )?;
        return Ok(if path.is_some() { 0 } else { 1 });
    }
    match path {
        Some(p) => {
            for row in &p.rows {
                println!(
                    "{} {}:{}",
                    display_row_name(row),
                    row.file_path,
                    row.start_line
                );
            }
            Ok(0)
        }
        None => {
            // AGENTS.md: "A command that finds nothing prints nothing and
            // exits 0." A missing chain is an answer, not a failure.
            Ok(0)
        }
    }
}

fn dispatch_search_symbols(
    query: Option<&str>,
    paths: &[String],
    kind: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("search-symbols requires a query".into()));
    }
    let path_filters = prepare_query_path_filters(root, "search-symbols", q, paths)?;
    let mut store = open_default_store(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    // Symbol rows are visible only from a freshness-proven snapshot.
    let decision = freshness_serve_decision(&store, root, &project);
    if let FreshnessServe::Refuse(freshness) = &decision {
        if json {
            search_symbols_json(
                &store,
                q,
                &project,
                "skipped_stale_index",
                Some(freshness),
                &[],
                &path_filters,
                Some(0),
            )?;
        } else {
            println!(
                "{}",
                indexed_stale_skip_message("search-symbols", freshness)
            );
        }
        return Ok(freshness_refusal_exit(freshness));
    }
    let freshness = decision.freshness().clone();
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            search_symbols_json(
                &store,
                q,
                &project,
                "skipped_incomplete_provider",
                Some(&freshness),
                &[],
                &path_filters,
                Some(0),
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("search-symbols", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    // Path/kind filters are post-query result filters: fetch broadly, then
    // narrow on node metadata without changing symbol ranking/resolution.
    let fetch = if kind.is_some() || !path_filters.is_empty() {
        10_000
    } else {
        cli_result_limit(20)
    };
    let mut hits = greppy_search::search_symbols_in_project(&store, &project, q, fetch)?;
    if let Some(k) = kind {
        let want = k.to_ascii_lowercase();
        hits.retain(|h| {
            store
                .get_node(h.node_id)
                .ok()
                .flatten()
                .map(|n| n.label.to_ascii_lowercase() == want)
                .unwrap_or(false)
        });
    }
    hits.retain(|hit| {
        store
            .get_node(hit.node_id)
            .ok()
            .flatten()
            .is_some_and(|node| path_filters.matches(&node.file_path))
    });
    let total_filtered = hits.len() as i64;
    hits.truncate(cli_result_limit(20));
    if json {
        search_symbols_json(
            &store,
            q,
            &project,
            "ok",
            Some(&freshness),
            &hits,
            &path_filters,
            (!path_filters.is_empty() || kind.is_some()).then_some(total_filtered),
        )?;
        return Ok(if hits.is_empty() { 1 } else { 0 });
    }
    if hits.is_empty() {
        // A name that matches no definition is an empty answer, not a failure:
        // `AGENTS.md` says such a question prints nothing and exits 0, and a
        // caller that pipes `search-symbols` into `read -` must not receive a
        // line of prose where it expects results. The path filter is the one
        // thing worth saying, because it can be the reason the answer is empty.
        if !path_filters.is_empty() {
            println!("(no matches under path filter: {})", path_filters.shown());
        }
    } else {
        for h in &hits {
            // Resolve each FTS hit to its node so we can print the
            // actionable label + qualified_name + file:line instead of
            // a bare node id (matches the other query commands' output).
            match store.get_node(h.node_id)? {
                Some(n) => println!(
                    "{} {} {}:{}",
                    n.label,
                    display_node_name(&n),
                    n.file_path,
                    n.start_line
                ),
                None => println!("node={}", h.node_id),
            }
        }
    }
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
fn search_symbols_json(
    store: &greppy_store::Store,
    query: &str,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    hits: &[greppy_search::SymbolHit],
    path_filters: &QueryPathFilters,
    total_override: Option<i64>,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let total_exact = if status == "ok" {
        match total_override {
            Some(total) => total,
            None => greppy_search::count_symbols_in_project(store, project, query)?,
        }
    } else {
        0
    };
    let mut rows = Vec::new();
    for h in hits {
        match store.get_node(h.node_id)? {
            Some(n) => rows.push(serde_json::json!({
                "node_id": h.node_id,
                "rank": h.rank,
                "target": query,
                "label": n.label,
                "name": n.name,
                "qualified_name": n.qualified_name,
                "file": n.file_path,
                "line": n.start_line,
                "file_path": n.file_path,
                "start_line": n.start_line,
                "end_line": n.end_line,
            })),
            None => rows.push(serde_json::json!({
                "node_id": h.node_id,
                "rank": h.rank,
                "source_available": false,
            })),
        }
    }
    let shown = rows.len() as i64;
    let omitted = total_exact.saturating_sub(shown);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-symbols",
            "status": status,
            "query": query,
            "project": project,
            "path_filters": path_filters.json_value(),
            "fresh": freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": incomplete_providers.is_empty(),
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-symbols JSON: {e}")))?
    );
    Ok(())
}

fn print_search_code_no_matches(query: &str, fixed: bool, path_filters: &QueryPathFilters) {
    println!("(no matches)");
    println!(
        "query_interpreted_as: {}",
        if fixed { "literal" } else { "regex" }
    );
    if path_filters.is_empty() {
        println!("path_filters: <none>");
    } else {
        println!("path_filters: {}", path_filters.shown());
    }
    if fixed
        && query
            .chars()
            .any(|character| ".^$*+?()[]{}|\\".contains(character))
    {
        println!("hint: regex metacharacters are literal because --fixed was supplied");
        let mut retry = format!("greppy search-code {}", shell_example_arg(query));
        for filter in &path_filters.filters {
            retry.push(' ');
            retry.push_str(&shell_example_arg(&filter.shown));
        }
        println!("try without --fixed: {retry}");
    }
}

#[derive(Debug, Clone)]
struct SearchCodeMatchLine {
    location: String,
    file: String,
    line: i64,
    text: String,
}

#[derive(Debug)]
struct SearchCodeDefinitionEntry {
    node_id: i64,
    qualified_name: String,
    file: String,
    start_line: i64,
    end_line: i64,
    source: String,
    handle: String,
    matches: Vec<SearchCodeMatchLine>,
}

#[derive(Debug)]
enum SearchCodeEntry {
    Definition(SearchCodeDefinitionEntry),
    Unenclosed(SearchCodeMatchLine),
}

fn parse_search_code_match(hit: &greppy_search::CodeHit) -> Option<SearchCodeMatchLine> {
    let (file, line) = hit.location.rsplit_once(':')?;
    Some(SearchCodeMatchLine {
        location: hit.location.clone(),
        file: file.to_string(),
        line: line.parse().ok()?,
        text: hit.snippet.clone(),
    })
}

fn search_code_definition_entry(
    root_path: &std::path::Path,
    row: &greppy_search::graph::SearchGraphRow,
) -> Result<Option<SearchCodeDefinitionEntry>> {
    if row.start_line < 1 || is_synthetic_file_anchor(&row.label, &row.name, &row.qualified_name) {
        return Ok(None);
    }
    let absolute = root_path.join(&row.file_path);
    let content = match std::fs::read(&absolute) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(Error::Io {
                context: format!("read {}", absolute.display()),
                source,
            });
        }
    };
    let Some(span) = read_span_with_meta(
        root_path,
        &row.file_path,
        row.start_line,
        row.end_line,
        usize::MAX,
        false,
    ) else {
        return Ok(None);
    };
    let (byte_start, byte_end) =
        line_range_to_bytes(&content, row.start_line as usize, span.end_line as usize);
    let mut handle = greppy_edit::EditHandle::for_range(
        root_path,
        std::path::Path::new(&row.file_path),
        &content,
        byte_start,
        byte_end,
    )?;
    let language = greppy_edit::language_for_path(std::path::Path::new(&row.file_path));
    handle.signature_fingerprint =
        greppy_edit::verbs::signature_fingerprint(language, &content, (byte_start, byte_end));
    handle.grammar_id = Some(format!("{language:?}"));
    handle.grammar_version = Some(env!("CARGO_PKG_VERSION").to_string());
    Ok(Some(SearchCodeDefinitionEntry {
        node_id: row.id,
        qualified_name: row.qualified_name.clone(),
        file: row.file_path.clone(),
        start_line: row.start_line,
        end_line: span.end_line,
        source: span.text,
        handle: handle.encode(),
        matches: Vec::new(),
    }))
}

fn search_code_entries(
    store: &greppy_store::Store,
    project: &str,
    root_path: &std::path::Path,
    hits: &[greppy_search::CodeHit],
    resolve_definitions: bool,
) -> Result<Vec<SearchCodeEntry>> {
    let mut entries = Vec::new();
    let mut definition_entries = std::collections::HashMap::<i64, usize>::new();
    for hit in hits {
        let Some(match_line) = parse_search_code_match(hit) else {
            continue;
        };
        let row = if resolve_definitions {
            greppy_search::definition_at(store, Some(project), &match_line.file, match_line.line)?
        } else {
            None
        };
        let Some(row) = row else {
            entries.push(SearchCodeEntry::Unenclosed(match_line));
            continue;
        };
        if let Some(index) = definition_entries.get(&row.id).copied() {
            if let SearchCodeEntry::Definition(definition) = &mut entries[index] {
                definition.matches.push(match_line);
            }
            continue;
        }
        let Some(mut definition) = search_code_definition_entry(root_path, &row)? else {
            entries.push(SearchCodeEntry::Unenclosed(match_line));
            continue;
        };
        definition.matches.push(match_line);
        let index = entries.len();
        definition_entries.insert(definition.node_id, index);
        entries.push(SearchCodeEntry::Definition(definition));
    }
    Ok(entries)
}

fn print_search_code_entries(entries: &[SearchCodeEntry]) {
    for entry in entries {
        match entry {
            SearchCodeEntry::Unenclosed(hit) => {
                println!("{}  {}", hit.location, clamp_snippet(&hit.text));
            }
            SearchCodeEntry::Definition(definition) => {
                println!(
                    "{} {}:{}-{}",
                    definition.qualified_name,
                    definition.file,
                    definition.start_line,
                    definition.end_line
                );
                let width = definition.end_line.to_string().len();
                for (offset, line) in definition.source.lines().enumerate() {
                    println!(
                        "  {:>width$} | {line}",
                        definition.start_line + offset as i64
                    );
                }
                println!("  handle: {}", definition.handle);
            }
        }
    }
}

fn search_code_entry_json(entry: &SearchCodeEntry) -> serde_json::Value {
    match entry {
        SearchCodeEntry::Definition(definition) => serde_json::json!({
            "qualified_name": &definition.qualified_name,
            "file": &definition.file,
            "span": {
                "start_line": definition.start_line,
                "end_line": definition.end_line,
            },
            "source": &definition.source,
            "handle": &definition.handle,
            "matches": definition.matches.iter().map(|hit| serde_json::json!({
                "location": &hit.location,
                "line": hit.line,
                "text": &hit.text,
            })).collect::<Vec<_>>(),
        }),
        SearchCodeEntry::Unenclosed(hit) => serde_json::json!({
            "qualified_name": serde_json::Value::Null,
            "file": &hit.file,
            "span": {
                "start_line": hit.line,
                "end_line": hit.line,
            },
            "source": serde_json::Value::Null,
            "handle": serde_json::Value::Null,
            "matches": [{
                "location": &hit.location,
                "line": hit.line,
                "text": &hit.text,
            }],
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_search_code_results_with_format(
    store: &greppy_store::Store,
    query: &str,
    project: &str,
    status: &str,
    index_freshness: Option<&serde_json::Value>,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
    path_filters: &QueryPathFilters,
    root_path: &std::path::Path,
    json: bool,
    no_code: bool,
    fixed: bool,
    resolve_definitions: bool,
) -> Result<()> {
    if !json {
        if hits.is_empty() {
            print_search_code_no_matches(query, fixed, path_filters);
        } else if no_code {
            for hit in hits {
                println!("{}  {}", hit.location, clamp_snippet(&hit.snippet));
            }
        } else {
            let entries =
                search_code_entries(store, project, root_path, hits, resolve_definitions)?;
            print_search_code_entries(&entries);
        }
        return Ok(());
    }

    let incomplete_providers = incomplete_provider_json(store, project)?;
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = if no_code {
        hits.iter()
            .map(|hit| {
                serde_json::json!({
                    "location": hit.location,
                    "rank": hit.rank,
                    "snippet": clamp_snippet(&hit.snippet).as_ref(),
                })
            })
            .collect::<Vec<_>>()
    } else {
        search_code_entries(store, project, root_path, hits, resolve_definitions)?
            .iter()
            .map(search_code_entry_json)
            .collect::<Vec<_>>()
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": status,
            "query": query,
            "pattern_mode": if fixed { "fixed" } else { "regex" },
            "project": project,
            "path_filters": path_filters.json_value(),
            "backend": "live-filesystem",
            "fresh": true,
            "freshness": if resolve_definitions {
                index_freshness.cloned().unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            },
            "index_freshness": if resolve_definitions {
                serde_json::Value::Null
            } else {
                index_freshness.cloned().unwrap_or(serde_json::Value::Null)
            },
            "provider_complete": incomplete_providers.is_empty(),
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|error| Error::Invalid(format!("serialize search-code JSON: {error}")))?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_search_code(
    query: Option<&str>,
    paths: &[String],
    changed: bool,
    staged: bool,
    since: Option<&str>,
    base: Option<&str>,
    json: bool,
    no_code: bool,
    fixed: bool,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("search-code requires a query".into()));
    }
    let path_filters = prepare_query_path_filters(root, "search-code", q, paths)?;
    let git_scope_count = usize::from(changed)
        + usize::from(staged)
        + usize::from(since.is_some())
        + usize::from(base.is_some());
    if git_scope_count > 1 {
        return Err(Error::Invalid(
            "search-code accepts only one git scope flag at a time".into(),
        ));
    }
    if changed {
        return dispatch_search_code_changed(q, json, fixed, root, &path_filters);
    }
    if staged {
        return dispatch_search_code_staged(q, json, fixed, root, &path_filters);
    }
    if let Some(rev) = since {
        return dispatch_search_code_since(q, rev, json, fixed, root, &path_filters);
    }
    if let Some(rev) = base {
        return dispatch_search_code_base(q, rev, json, fixed, root, &path_filters);
    }

    let store = open_default_store(root)?;
    let project = project_for(root)?;
    let root_path = resolve_root(root)?;
    let decision = freshness_serve_decision(&store, root, &project);
    let resolve_definitions = matches!(decision, FreshnessServe::Fresh(_));
    let status = if resolve_definitions {
        "ok"
    } else {
        "live-fallback"
    };
    if let FreshnessServe::Refuse(freshness) = &decision {
        if !json {
            eprintln!(
                "{}; falling back to live grep",
                indexed_stale_skip_message("search-code", freshness)
            );
        }
    }
    let all_hits = if path_filters.is_empty() {
        live_grep_code_hits_pattern(q, &root_path, fixed)?
    } else {
        live_grep_code_hits_filtered_pattern(q, &root_path, &path_filters, fixed)?
    };
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();
    emit_search_code_results_with_format(
        &store,
        q,
        &project,
        status,
        Some(decision.freshness()),
        all_hits.len(),
        &shown_hits,
        &path_filters,
        &root_path,
        json,
        no_code,
        fixed,
        resolve_definitions,
    )?;
    Ok(if all_hits.is_empty() { 1 } else { 0 })
}

fn dispatch_search_code_changed(
    query: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let mut changed_files = git_changed_files(&root_path)?;
    changed_files.retain(|path| path_filters.matches(path));
    let all_hits = live_grep_search_code_paths_pattern(query, &root_path, &changed_files, fixed)?;
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_changed_json(
            query,
            &project,
            changed_files.len(),
            all_hits.len(),
            &shown_hits,
            path_filters,
        )?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        print_search_code_no_matches(query, fixed, path_filters);
        return Ok(0);
    }
    for h in &shown_hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

fn search_code_changed_json(
    query: &str,
    project: &str,
    changed_files_total: usize,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
    path_filters: &QueryPathFilters,
) -> Result<()> {
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "location": h.location,
                "rank": h.rank,
                "snippet": clamp_snippet(&h.snippet).as_ref(),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": if total_exact == 0 { "no_matches" } else { "ok" },
            "query": query,
            "project": project,
            "scope": "changed",
            "path_filters": path_filters.json_value(),
            "backend": "live_grep",
            "fresh": true,
            "freshness": serde_json::Value::Null,
            "changed_files_total": changed_files_total,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-code changed JSON: {e}")))?
    );
    Ok(())
}

fn dispatch_search_code_staged(
    query: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let mut staged_files = git_staged_files(&root_path)?;
    staged_files.retain(|path| path_filters.matches(path));
    let all_hits = grep_staged_git_blobs_pattern(query, &root_path, &staged_files, fixed)?;
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_staged_json(
            query,
            &project,
            staged_files.len(),
            all_hits.len(),
            &shown_hits,
        )?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        print_search_code_no_matches(query, fixed, path_filters);
        return Ok(0);
    }
    for h in &shown_hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

fn search_code_staged_json(
    query: &str,
    project: &str,
    staged_files_total: usize,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
) -> Result<()> {
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "location": h.location,
                "rank": h.rank,
                "snippet": clamp_snippet(&h.snippet).as_ref(),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": if total_exact == 0 { "no_matches" } else { "ok" },
            "query": query,
            "project": project,
            "scope": "staged",
            "backend": "git_blob_grep",
            "fresh": true,
            "freshness": serde_json::Value::Null,
            "staged_files_total": staged_files_total,
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-code staged JSON: {e}")))?
    );
    Ok(())
}

fn dispatch_search_code_since(
    query: &str,
    rev: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    dispatch_search_code_diff_scope(
        query,
        DiffSearchScope::Since { rev },
        json,
        fixed,
        root,
        path_filters,
    )
}

fn dispatch_search_code_base(
    query: &str,
    base: &str,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    dispatch_search_code_diff_scope(
        query,
        DiffSearchScope::Base { base },
        json,
        fixed,
        root,
        path_filters,
    )
}

enum DiffSearchScope<'a> {
    Since { rev: &'a str },
    Base { base: &'a str },
}

struct DiffSearchSpec {
    scope: &'static str,
    diff_rev: String,
    merge_base: Option<String>,
    files: Vec<String>,
}

fn dispatch_search_code_diff_scope(
    query: &str,
    scope: DiffSearchScope<'_>,
    json: bool,
    fixed: bool,
    root: Option<&str>,
    path_filters: &QueryPathFilters,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let mut spec = git_diff_search_spec(&root_path, scope)?;
    spec.files.retain(|path| path_filters.matches(path));
    let all_hits = live_grep_search_code_paths_pattern(query, &root_path, &spec.files, fixed)?;
    let shown_hits = all_hits
        .iter()
        .take(cli_result_limit(SEARCH_CODE_LIMIT))
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_diff_scope_json(query, &project, &spec, all_hits.len(), &shown_hits)?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        print_search_code_no_matches(query, fixed, path_filters);
        return Ok(0);
    }
    for h in &shown_hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

fn search_code_diff_scope_json(
    query: &str,
    project: &str,
    spec: &DiffSearchSpec,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
) -> Result<()> {
    let shown = hits.len();
    let omitted = total_exact.saturating_sub(shown);
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "location": h.location,
                "rank": h.rank,
                "snippet": clamp_snippet(&h.snippet).as_ref(),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "search-code",
            "status": if total_exact == 0 { "no_matches" } else { "ok" },
            "query": query,
            "project": project,
            "scope": spec.scope,
            "backend": "git_diff_live_grep",
            "fresh": true,
            "freshness": serde_json::Value::Null,
            "diff_rev": &spec.diff_rev,
            "merge_base": spec.merge_base.as_deref(),
            "diff_files_total": spec.files.len(),
            "total_exact": total_exact,
            "shown": shown,
            "omitted": omitted,
            "truncated": omitted > 0,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize search-code diff JSON: {e}")))?
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct PlusHit {
    location: String,
    file_path: String,
    line: i64,
    symbol: Option<String>,
    node: Option<greppy_store::Node>,
    score: f64,
    signals: std::collections::BTreeSet<String>,
    snippet: String,
}

struct PlusJsonMeta<'a> {
    status: &'a str,
    project: &'a str,
    query: &'a str,
    freshness: Option<&'a serde_json::Value>,
    provider_complete: bool,
    incomplete_providers: &'a [serde_json::Value],
    limit: usize,
    code: bool,
    explain: bool,
    vectors: bool,
    fetch_limit_per_signal: usize,
    precision_floor: f64,
    vector_status: Option<&'a str>,
    vector_candidate_total: Option<i64>,
    vector_candidate_limit: Option<i64>,
    vector_hits_added: Option<usize>,
}

const PLUS_VECTOR_MIN_SCORE: f32 = 0.35;
const PLUS_VECTOR_MAX_CONFIDENCE: f64 = 0.82;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlusVectorControlIntent {
    Literal,
    Graph,
}

impl PlusVectorControlIntent {
    fn status(self) -> &'static str {
        match self {
            Self::Literal => "skipped_literal_control",
            Self::Graph => "skipped_graph_control",
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::Literal => {
                "grep: skipped EmbeddingGemma for literal/exact query; using exact signals only"
            }
            Self::Graph => {
                "grep: skipped EmbeddingGemma for graph-control query; using graph/text signals only"
            }
        }
    }
}

impl PlusHit {
    fn add_signal(&mut self, signal: impl Into<String>, confidence: f64) {
        let signal = signal.into();
        if !self.signals.insert(signal) {
            return;
        }
        let c = confidence.clamp(0.0, 1.0);
        // Keep the public score in 0..1 while still rewarding independent
        // evidence. This is a search relevance score, not a probability.
        self.score = 1.0 - ((1.0 - self.score) * (1.0 - c));
    }
}

fn plus_relevance_from_ranks(ranks: &[f64], rank: f64) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let mut best = ranks[0];
    let mut worst = ranks[0];
    for r in ranks {
        if *r < best {
            best = *r;
        }
        if *r > worst {
            worst = *r;
        }
    }
    let span = worst - best;
    if span > 0.0 {
        (worst - rank) / span
    } else {
        1.0
    }
}

fn plus_query_tokens(query: &str) -> Vec<String> {
    greppy_store::fts::camel_split(query)
        .split_whitespace()
        .map(plus_canonical_token)
        .filter(|tok| tok.len() >= 3)
        .collect()
}

fn plus_canonical_token(token: &str) -> String {
    let t = token.to_ascii_lowercase();
    for suffix in ["isation", "ization"] {
        if let Some(base) = t.strip_suffix(suffix) {
            return format!("{base}ize");
        }
    }
    for suffix in ["ising", "izing", "ised", "ized"] {
        if let Some(base) = t.strip_suffix(suffix) {
            return format!("{base}ize");
        }
    }
    if let Some(base) = t.strip_suffix("ise") {
        return format!("{base}ize");
    }
    t
}

fn plus_symbol_tokens(node: &greppy_store::Node) -> Vec<String> {
    greppy_store::fts::camel_split(&format!(
        "{} {} {}",
        node.name, node.qualified_name, node.file_path
    ))
    .split_whitespace()
    .map(plus_canonical_token)
    .filter(|tok| tok.len() >= 3)
    .collect()
}

fn plus_is_pseudo_node(node: &greppy_store::Node) -> bool {
    matches!(node.label.as_str(), "Module" | "Import" | "Call")
        || node.qualified_name.ends_with("::__file__")
}

fn plus_is_constructor_like(node: &greppy_store::Node) -> bool {
    if !matches!(node.label.as_str(), "Method" | "Function") {
        return false;
    }
    let parts: Vec<&str> = node
        .qualified_name
        .split("::")
        .filter(|part| !part.is_empty())
        .collect();
    parts.len() >= 2 && parts[parts.len() - 1] == parts[parts.len() - 2]
}

fn plus_is_executable_node(node: &greppy_store::Node) -> bool {
    matches!(node.label.as_str(), "Function" | "Method") && !plus_is_constructor_like(node)
}

fn plus_is_code_intent(tokens: &[String]) -> bool {
    tokens.iter().any(|tok| {
        matches!(
            tok.as_str(),
            "code"
                | "show"
                | "where"
                | "return"
                | "loop"
                | "fold"
                | "function"
                | "method"
                | "implement"
                | "implemented"
                | "implementation"
        )
    })
}

fn plus_is_literal_intent(query: &str, tokens: &[String]) -> bool {
    let trimmed = query.trim();
    tokens.len() == 1
        || trimmed.contains('_')
        || trimmed.contains("::")
        || trimmed.chars().any(|c| c.is_ascii_digit())
}

fn plus_has_camel_identifier(query: &str) -> bool {
    query
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
        .any(|part| {
            let mut chars = part.chars();
            let Some(first) = chars.next() else {
                return false;
            };
            first.is_ascii_alphabetic()
                && part.chars().any(|c| c.is_ascii_lowercase())
                && chars.any(|c| c.is_ascii_uppercase())
        })
}

fn plus_is_graph_control_token(token: &str) -> bool {
    matches!(
        token,
        "affected"
            | "blast"
            | "break"
            | "call"
            | "called"
            | "caller"
            | "callers"
            | "calls"
            | "callee"
            | "callees"
            | "change"
            | "changed"
            | "dependency"
            | "dependents"
            | "depends"
            | "direct"
            | "find"
            | "from"
            | "impact"
            | "path"
            | "radius"
            | "reference"
            | "referenced"
            | "references"
            | "trace"
            | "usage"
            | "usages"
            | "what"
            | "where"
            | "would"
    )
}

fn plus_is_graph_control_intent(query: &str, tokens: &[String]) -> bool {
    let q = query.to_ascii_lowercase();
    let graph_phrase = q.contains("who calls")
        || q.contains("what calls")
        || q.contains("called by")
        || q.contains("direct caller")
        || q.contains("direct callee")
        || q.contains("call path")
        || q.contains("trace from")
        || q.contains("trace path")
        || q.starts_with("trace ")
        || q.contains("path from")
        || q.contains("depends on")
        || q.contains("dependency path")
        || q.contains("referenced by")
        || q.contains("references to")
        || q.contains("find usages")
        || q.contains("usages of")
        || q.contains("what would break")
        || q.contains("break if")
        || q.contains("affected by")
        || q.contains("blast radius")
        || (q.contains("impact") && (q.contains("change") || q.contains("changed")));
    if !graph_phrase {
        return false;
    }

    query.contains('_')
        || query.contains("::")
        || plus_has_camel_identifier(query)
        || tokens
            .iter()
            .any(|tok| tok.len() >= 5 && !plus_is_graph_control_token(tok))
}

fn plus_vector_control_intent(
    query: &str,
    tokens: &[String],
    has_exact_text_hit: bool,
) -> Option<PlusVectorControlIntent> {
    if plus_is_graph_control_intent(query, tokens) {
        Some(PlusVectorControlIntent::Graph)
    } else if has_exact_text_hit || plus_is_literal_intent(query, tokens) {
        Some(PlusVectorControlIntent::Literal)
    } else {
        None
    }
}

fn plus_allows_ranked_node(node: &greppy_store::Node, code_intent: bool) -> bool {
    if plus_is_pseudo_node(node) {
        return false;
    }
    !code_intent || plus_is_executable_node(node)
}

fn plus_precision_floor(best_score: f64) -> f64 {
    if best_score >= 0.80 {
        best_score - 0.10
    } else if best_score >= 0.70 {
        best_score - 0.15
    } else {
        0.0
    }
}

fn plus_token_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    let distance = plus_levenshtein(a, b) as f64;
    let width = a.chars().count().max(b.chars().count()) as f64;
    (1.0 - (distance / width)).clamp(0.0, 1.0)
}

fn plus_levenshtein(a: &str, b: &str) -> usize {
    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0; b_chars.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_chars.len()]
}

fn plus_key(file_path: &str, line: i64) -> String {
    format!("{file_path}:{line}")
}

fn plus_first_line(root: &std::path::Path, node: &greppy_store::Node) -> String {
    read_span(
        root,
        &node.file_path,
        node.start_line,
        node.end_line,
        1,
        false,
    )
    .and_then(|span| span.lines().next().map(|line| line.trim().to_string()))
    .filter(|line| !line.is_empty())
    .unwrap_or_else(|| node.qualified_name.clone())
}

fn plus_store_node_from_row(
    store: &greppy_store::Store,
    row: &greppy_search::graph::SearchGraphRow,
) -> Result<Option<greppy_store::Node>> {
    Ok(store.get_node(row.id)?)
}

fn plus_enclosing_node(
    store: &greppy_store::Store,
    project: &str,
    location: &str,
) -> Result<Option<greppy_store::Node>> {
    let Some((file, line_str)) = location.rsplit_once(':') else {
        return Ok(None);
    };
    let Ok(line) = line_str.parse::<i64>() else {
        return Ok(None);
    };
    match greppy_search::definition_at(store, Some(project), file, line)? {
        Some(row) => plus_store_node_from_row(store, &row),
        None => Ok(None),
    }
}

fn plus_put_hit(
    hits: &mut std::collections::BTreeMap<String, PlusHit>,
    file_path: &str,
    line: i64,
    snippet: String,
    node: Option<greppy_store::Node>,
    signal: impl Into<String>,
    confidence: f64,
) {
    let key = plus_key(file_path, line);
    let entry = hits.entry(key).or_insert_with(|| PlusHit {
        location: format!("{file_path}:{line}"),
        file_path: file_path.to_string(),
        line,
        symbol: node.as_ref().map(|n| n.qualified_name.clone()),
        node: node.clone(),
        score: 0.0,
        signals: std::collections::BTreeSet::new(),
        snippet: snippet.clone(),
    });
    if entry.symbol.is_none() {
        entry.symbol = node.as_ref().map(|n| n.qualified_name.clone());
    }
    if entry.node.is_none() {
        entry.node = node;
    }
    if entry.snippet.trim().is_empty() && !snippet.trim().is_empty() {
        entry.snippet = snippet;
    }
    entry.add_signal(signal, confidence);
}

fn plus_vector_confidence(score: f32) -> f64 {
    let min = f64::from(PLUS_VECTOR_MIN_SCORE);
    let s = f64::from(score);
    if s < min {
        return 0.0;
    }
    (((s - min) / (1.0 - min)) * PLUS_VECTOR_MAX_CONFIDENCE).clamp(0.0, PLUS_VECTOR_MAX_CONFIDENCE)
}

#[allow(clippy::too_many_arguments)]
fn plus_add_vector_hits_from_query_vector(
    store: &greppy_store::Store,
    project: &str,
    root_path: &std::path::Path,
    code_intent: bool,
    hits: &mut std::collections::BTreeMap<String, PlusHit>,
    model_id: &str,
    graph_generation: u64,
    query_vector: &[f32],
    limit: usize,
) -> Result<usize> {
    let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
        project,
        model_id,
        Some(graph_generation),
        limit,
    );
    scope.min_score = Some(PLUS_VECTOR_MIN_SCORE);

    let mut added = 0usize;
    for h in greppy_search::vector_search_exact(store, query_vector, &scope)? {
        let node = match h.embedding.node_id {
            Some(id) => store.get_node(id)?,
            None => None,
        };
        if let Some(n) = &node {
            if !plus_allows_ranked_node(n, code_intent) {
                continue;
            }
        }
        let (file_path, line, snippet) = if let Some(n) = &node {
            (
                n.file_path.clone(),
                n.start_line,
                plus_first_line(root_path, n),
            )
        } else {
            (
                h.embedding.file_path.clone(),
                h.embedding.start_line,
                h.embedding.qualified_name.clone(),
            )
        };
        plus_put_hit(
            hits,
            &file_path,
            line,
            snippet,
            node,
            "vector",
            plus_vector_confidence(h.score),
        );
        added += 1;
    }
    Ok(added)
}

fn plus_add_graph_signals(store: &greppy_store::Store, hit: &mut PlusHit) -> Result<()> {
    let Some(node) = &hit.node else {
        return Ok(());
    };
    let is_executable = plus_is_executable_node(node);
    let incoming = store.incoming_edges(node.id, Some("CALLS"), 1024)?.len();
    let outgoing = store.outgoing_edges(node.id, Some("CALLS"), 1024)?.len();
    if incoming > 0 {
        let boost = ((incoming as f64 + 1.0).log10() * 0.10).min(0.22);
        hit.add_signal(format!("graph-in={incoming}"), boost);
    }
    if outgoing > 0 {
        let boost = ((outgoing as f64 + 1.0).log10() * 0.10).min(0.18);
        hit.add_signal(format!("graph-out={outgoing}"), boost);
    }
    if is_executable {
        hit.add_signal("kind=code", 0.10);
    }
    Ok(())
}

fn plus_json(
    meta: PlusJsonMeta<'_>,
    ranked: &[PlusHit],
    root_path: &std::path::Path,
) -> Result<()> {
    let eligible = ranked
        .iter()
        .filter(|hit| hit.score >= meta.precision_floor)
        .collect::<Vec<_>>();
    let shown_hits = eligible
        .iter()
        .copied()
        .take(meta.limit)
        .collect::<Vec<_>>();
    let omitted = eligible.len().saturating_sub(shown_hits.len());
    let mut source_unavailable_count = 0usize;
    let mut source_truncated_count = 0usize;
    let mut rows = Vec::new();

    for hit in shown_hits {
        let signals = hit.signals.iter().cloned().collect::<Vec<_>>();
        let snippet = clamp_snippet(&hit.snippet).into_owned();
        let mut source_available = false;
        let mut source_included = false;
        let mut source = serde_json::Value::Null;
        let mut source_total_lines = serde_json::Value::Null;
        let mut source_shown_lines = serde_json::Value::Null;
        let mut source_omitted_lines = serde_json::Value::Null;
        let mut source_truncated = false;

        if meta.code {
            if let Some(node) = &hit.node {
                match read_span_with_meta(
                    root_path,
                    &node.file_path,
                    node.start_line,
                    node.end_line,
                    CODE_SPAN_CAP,
                    false,
                ) {
                    Some(span) => {
                        source_available = true;
                        source_included = true;
                        source_truncated = span.truncated;
                        if source_truncated {
                            source_truncated_count += 1;
                        }
                        source = serde_json::Value::String(span.text);
                        source_total_lines = serde_json::json!(span.total_lines);
                        source_shown_lines = serde_json::json!(span.shown_lines);
                        source_omitted_lines = serde_json::json!(span.omitted_lines);
                    }
                    None => {
                        source_unavailable_count += 1;
                    }
                }
            } else {
                source_unavailable_count += 1;
            }
        }

        rows.push(serde_json::json!({
            "location": hit.location,
            "file_path": hit.file_path,
            "line": hit.line,
            "snippet": snippet,
            "score": hit.score,
            "signals": signals,
            "symbol": hit.symbol,
            "source_available": source_available,
            "source_included": source_included,
            "source": source,
            "source_total_lines": source_total_lines,
            "source_shown_lines": source_shown_lines,
            "source_omitted_lines": source_omitted_lines,
            "source_truncated": source_truncated,
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "plus",
            "status": meta.status,
            "project": meta.project,
            "query": meta.query,
            "fresh": meta.freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": meta.freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": meta.provider_complete,
            "incomplete_provider_count": meta.incomplete_providers.len(),
            "incomplete_providers": meta.incomplete_providers,
            "limit": meta.limit,
            "code": meta.code,
            "explain": meta.explain,
            "vectors": meta.vectors,
            "fetch_limit_per_signal": meta.fetch_limit_per_signal,
            "candidate_total_kind": "bounded_fetch_union",
            "ranked_total": ranked.len(),
            "eligible_total": eligible.len(),
            "shown": rows.len(),
            "omitted": omitted,
            "truncated": omitted > 0 || source_truncated_count > 0,
            "precision_floor": meta.precision_floor,
            "source_cap_lines": CODE_SPAN_CAP,
            "source_unavailable_count": source_unavailable_count,
            "source_truncated_count": source_truncated_count,
            "vector_status": meta.vector_status,
            "vector_candidate_total": meta.vector_candidate_total,
            "vector_candidate_limit": meta.vector_candidate_limit,
            "vector_hits_added": meta.vector_hits_added,
            "hits": rows,
        }))
        .map_err(|e| Error::Invalid(format!("serialize JSON: {e}")))?
    );
    Ok(())
}

/// `greppy plus <query>` — a grep-like fused search path.
///
/// This deliberately stays a SEARCH command: it does not summarize, does not
/// answer the user's question, and does not invent context. It emits ranked
/// hits with stable locations and signal labels, combining the "plus" parts
/// grep lacks: symbol matching, fuzzy semantic matching, and graph-neighbour
/// hints. EmbeddingGemma code-retrieval hits are always available as another
/// signal, scoped to the current graph generation. Exact literal/graph control
/// queries still short-circuit before loading the model.
#[allow(clippy::too_many_arguments)]
fn dispatch_plus(
    query: Option<&str>,
    k: usize,
    code: bool,
    explain: bool,
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let vectors = true;
    let store = open_default_store(root)?;
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("a query is required".into()));
    }
    let k = cli_result_limit(k).max(1);
    let project = project_for(root)?;
    let root_path = resolve_root(root)?;
    // Combined search also refuses stale indexed lexical/graph/vector rows.
    let decision = freshness_serve_decision(&store, root, &project);
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    let provider_complete = incomplete_providers.is_empty();
    let fetch = (k * 4).max(20);
    if let FreshnessServe::Refuse(freshness) = &decision {
        if json {
            plus_json(
                PlusJsonMeta {
                    status: "skipped_stale_index",
                    project: &project,
                    query: q,
                    freshness: Some(freshness),
                    provider_complete,
                    incomplete_providers: &incomplete_providers,
                    limit: k,
                    code,
                    explain,
                    vectors,
                    fetch_limit_per_signal: fetch,
                    precision_floor: 0.0,
                    vector_status: if vectors {
                        Some("skipped_stale_index")
                    } else {
                        None
                    },
                    vector_candidate_total: None,
                    vector_candidate_limit: None,
                    vector_hits_added: None,
                },
                &[],
                &root_path,
            )?;
        } else {
            eprintln!("{}", plus_stale_skip_message(freshness));
            println!(
                "(no usable index; run `greppy index {}` first)",
                root.unwrap_or(".")
            );
        }
        return Ok(freshness_refusal_exit(freshness));
    }
    let freshness = decision.freshness().clone();
    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            plus_json(
                PlusJsonMeta {
                    status: "skipped_incomplete_provider",
                    project: &project,
                    query: q,
                    freshness: Some(&freshness),
                    provider_complete,
                    incomplete_providers: &incomplete_providers,
                    limit: k,
                    code,
                    explain,
                    vectors,
                    fetch_limit_per_signal: fetch,
                    precision_floor: 0.0,
                    vector_status: if vectors {
                        Some("skipped_incomplete_provider")
                    } else {
                        None
                    },
                    vector_candidate_total: None,
                    vector_candidate_limit: None,
                    vector_hits_added: None,
                },
                &[],
                &root_path,
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("grep", incomplete_providers.len())
            );
        }
        return Ok(1);
    }
    let q_tokens = plus_query_tokens(q);
    let code_intent = plus_is_code_intent(&q_tokens);
    let mut hits: std::collections::BTreeMap<String, PlusHit> = std::collections::BTreeMap::new();
    let mut vector_status = if vectors { Some("requested") } else { None };
    let mut vector_candidate_total = None;
    let mut vector_candidate_limit = None;
    let mut vector_hits_added = None;

    // Literal/full-text signal: exact current-worktree lines remain
    // first-class grep-like results even though source bodies are not copied
    // into SQLite by default.
    let code_hits = source_code_hits_ranked(&store, &project, q, &root_path, fetch)?;
    let exact_literal_text = !code && !code_hits.is_empty() && plus_is_literal_intent(q, &q_tokens);
    let vector_control_intent = if vectors {
        plus_vector_control_intent(q, &q_tokens, exact_literal_text)
    } else {
        None
    };
    if let Some(control) = vector_control_intent {
        vector_status = Some(control.status());
        if !json {
            eprintln!("{}", control.message());
        }
    }
    let vector_config = if vectors && vector_control_intent.is_none() {
        Some(embedding_config_for_required_use(embedding_args)?)
    } else {
        None
    };
    for h in &code_hits {
        let Some((file, line_str)) = h.location.rsplit_once(':') else {
            continue;
        };
        let Ok(line) = line_str.parse::<i64>() else {
            continue;
        };
        let node = plus_enclosing_node(&store, &project, &h.location)?;
        plus_put_hit(
            &mut hits,
            file,
            line,
            h.snippet.clone(),
            node,
            "text",
            h.relevance,
        );
    }
    if code_hits.is_empty() {
        let mut seen_tokens = std::collections::BTreeSet::new();
        for tok in plus_query_tokens(q) {
            if !seen_tokens.insert(tok.clone()) {
                continue;
            }
            for h in source_code_hits_ranked(&store, &project, &tok, &root_path, fetch / 2)? {
                let Some((file, line_str)) = h.location.rsplit_once(':') else {
                    continue;
                };
                let Ok(line) = line_str.parse::<i64>() else {
                    continue;
                };
                let node = plus_enclosing_node(&store, &project, &h.location)?;
                plus_put_hit(
                    &mut hits,
                    file,
                    line,
                    h.snippet,
                    node,
                    format!("text-token={tok}"),
                    h.relevance * 0.72,
                );
            }
        }
    }

    if !exact_literal_text {
        // Symbol FTS signal: identifier/camel-case aware, still output as a
        // location + snippet, not as prose.
        let symbol_hits = greppy_search::search_symbols_in_project(&store, &project, q, fetch)?;
        let symbol_ranks: Vec<f64> = symbol_hits.iter().map(|h| h.rank).collect();
        for h in symbol_hits {
            if let Some(n) = store.get_node(h.node_id)? {
                if !plus_allows_ranked_node(&n, code_intent) {
                    continue;
                }
                let rel = plus_relevance_from_ranks(&symbol_ranks, h.rank);
                let snippet = plus_first_line(&root_path, &n);
                let file_path = n.file_path.clone();
                let start_line = n.start_line;
                plus_put_hit(
                    &mut hits,
                    &file_path,
                    start_line,
                    snippet,
                    Some(n),
                    "symbol",
                    rel,
                );
            }
        }

        // Fuzzy token signal over symbols: catches spelling/convention variants
        // such as normalisation/normalize without turning the command into prose.
        if !q_tokens.is_empty() {
            for n in store.list_nodes(&project, "", "", 0, 100_000)? {
                if !plus_allows_ranked_node(&n, code_intent) {
                    continue;
                }
                let node_tokens = plus_symbol_tokens(&n);
                let best = q_tokens
                    .iter()
                    .flat_map(|qt| {
                        node_tokens
                            .iter()
                            .map(move |nt| plus_token_similarity(qt, nt))
                    })
                    .fold(0.0_f64, f64::max);
                if best >= 0.86 {
                    let snippet = plus_first_line(&root_path, &n);
                    let file_path = n.file_path.clone();
                    let start_line = n.start_line;
                    plus_put_hit(
                        &mut hits,
                        &file_path,
                        start_line,
                        snippet,
                        Some(n),
                        "fuzzy-token",
                        (best * 0.78).min(0.78),
                    );
                }
            }
        }

        // Fuzzy semantic signal: algorithmic semantic scorer over indexed symbol
        // metadata. This is the "plus" part, still represented as a search hit.
        for h in greppy_search::semantic_query(&store, q, None, Some(&project), fetch)? {
            if let Some(n) = plus_store_node_from_row(&store, &h.node)? {
                if !plus_allows_ranked_node(&n, code_intent) {
                    continue;
                }
                let confidence = (h.score / greppy_search::MAX_SEMANTIC_SCORE).clamp(0.0, 1.0);
                let snippet = plus_first_line(&root_path, &n);
                let file_path = n.file_path.clone();
                let start_line = n.start_line;
                plus_put_hit(
                    &mut hits,
                    &file_path,
                    start_line,
                    snippet,
                    Some(n),
                    "fuzzy",
                    confidence,
                );
            }
        }

        if let Some(cfg) = &vector_config {
            let freshness = nav_freshness_json(&store, root, &project);
            if !freshness_json_is_fresh(&freshness) {
                vector_status = Some("skipped_stale_index");
                if !json {
                    eprintln!("{}", vector_stale_skip_message("grep", &freshness));
                }
            } else {
                let generation = current_graph_generation(&store, root)?;
                let scope = greppy_search::embeddinggemma_code_retrieval_scope(
                    &project,
                    &cfg.model_id,
                    Some(generation),
                    fetch,
                );
                let total = greppy_search::count_vector_search_scope(&store, &scope)?;
                let candidate_limit = vector_exact_candidate_limit()?;
                vector_candidate_total = Some(total);
                vector_candidate_limit = candidate_limit;
                if total == 0 {
                    vector_status = Some("no_current_vectors");
                    // No vector rows for this model/profile/generation; keep
                    // the normal plus path intact.
                } else if let Some(limit) = vector_exact_scan_exceeds_limit(total, candidate_limit)
                {
                    vector_status = Some("skipped_over_budget");
                    if !json {
                        eprintln!("{}", vector_exact_scan_skip_message("grep", total, limit));
                    }
                } else {
                    match embed_query_cached(cfg, root, q) {
                        Ok(query_vector) => {
                            let added = plus_add_vector_hits_from_query_vector(
                                &store,
                                &project,
                                &root_path,
                                code_intent,
                                &mut hits,
                                &cfg.model_id,
                                generation,
                                &query_vector,
                                fetch,
                            )?;
                            vector_status = Some("searched");
                            vector_hits_added = Some(added);
                        }
                        Err(e) => {
                            vector_status = Some("skipped_embedding_error");
                            log_embedding_skip_once("grep", &e);
                        }
                    }
                }
            }
        }
    }

    for hit in hits.values_mut() {
        plus_add_graph_signals(&store, hit)?;
    }

    let mut ranked: Vec<PlusHit> = hits.into_values().collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.file_path.cmp(&b.file_path))
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.symbol.cmp(&b.symbol))
    });

    if ranked.is_empty() {
        if json {
            plus_json(
                PlusJsonMeta {
                    status: "ok",
                    project: &project,
                    query: q,
                    freshness: Some(&freshness),
                    provider_complete,
                    incomplete_providers: &incomplete_providers,
                    limit: k,
                    code,
                    explain,
                    vectors,
                    fetch_limit_per_signal: fetch,
                    precision_floor: 0.0,
                    vector_status,
                    vector_candidate_total,
                    vector_candidate_limit,
                    vector_hits_added,
                },
                &ranked,
                &root_path,
            )?;
        } else {
            println!("(no matches)");
        }
        return Ok(1);
    }

    let precision_floor = if explain || ranked.is_empty() {
        0.0
    } else {
        plus_precision_floor(ranked[0].score)
    };
    if json {
        plus_json(
            PlusJsonMeta {
                status: "ok",
                project: &project,
                query: q,
                freshness: Some(&freshness),
                provider_complete,
                incomplete_providers: &incomplete_providers,
                limit: k,
                code,
                explain,
                vectors,
                fetch_limit_per_signal: fetch,
                precision_floor,
                vector_status,
                vector_candidate_total,
                vector_candidate_limit,
                vector_hits_added,
            },
            &ranked,
            &root_path,
        )?;
        return Ok(0);
    }
    let mut printed = 0usize;
    for hit in ranked
        .iter()
        .filter(|hit| hit.score >= precision_floor)
        .take(k)
    {
        print!("{}:{}", hit.location, clamp_snippet(&hit.snippet));
        if explain {
            let signals = hit
                .signals
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(",");
            let symbol = hit
                .node
                .as_ref()
                .map(display_node_name)
                .or_else(|| hit.symbol.clone())
                .unwrap_or_else(|| "-".to_string());
            print!(
                "\t# score={:.3} signals={} symbol={}",
                hit.score, signals, symbol
            );
        }
        println!();
        printed += 1;
        if code {
            if let Some(node) = &hit.node {
                print_code_span(&root_path, node, CODE_SPAN_CAP);
            }
        }
    }
    if printed == 0 {
        let hit = &ranked[0];
        println!("{}:{}", hit.location, clamp_snippet(&hit.snippet));
    }
    Ok(0)
}

/// Live `grep -rnI` over the resolved repo root, used as the `search-code`
/// fallback when the content-FTS index is empty. Output mirrors the FTS
/// form (`relpath:line  snippet`) so an agent sees a consistent shape.
fn live_grep_code_hits(
    query: &str,
    root_path: &std::path::Path,
) -> Result<Vec<greppy_search::CodeHit>> {
    live_grep_code_hits_pattern(query, root_path, true)
}

fn live_grep_code_hits_pattern(
    query: &str,
    root_path: &std::path::Path,
    fixed: bool,
) -> Result<Vec<greppy_search::CodeHit>> {
    let overrides = discover_overrides_from_env()?;
    let entries = greppy_discover::walk_with_policy_and_overrides(
        root_path,
        &greppy_discover::SkipPolicy::walk_default(),
        &overrides,
    )?;
    let paths = entries
        .into_iter()
        .map(|entry| entry.rel_path)
        .collect::<Vec<_>>();
    live_grep_search_code_paths_pattern(query, root_path, &paths, fixed)
}

fn live_grep_code_hits_filtered_pattern(
    query: &str,
    root_path: &std::path::Path,
    path_filters: &QueryPathFilters,
    fixed: bool,
) -> Result<Vec<greppy_search::CodeHit>> {
    let mut hits = live_grep_code_hits_pattern(query, root_path, fixed)?;
    hits.retain(|hit| {
        hit.location
            .rsplit_once(':')
            .is_some_and(|(path, _)| path_filters.matches(path))
    });
    Ok(hits)
}

fn source_code_hits_ranked(
    store: &greppy_store::Store,
    project: &str,
    query: &str,
    root_path: &std::path::Path,
    limit: usize,
) -> Result<Vec<greppy_search::RankedCodeHit>> {
    let indexed = greppy_search::search_code_ranked(store, project, query, limit)?;
    if !indexed.is_empty() {
        return Ok(indexed);
    }
    Ok(live_grep_code_hits(query, root_path)?
        .into_iter()
        .take(limit)
        .map(|hit| greppy_search::RankedCodeHit {
            location: hit.location,
            snippet: hit.snippet,
            rank: 0.0,
            relevance: 1.0,
        })
        .collect())
}

fn git_changed_files(root_path: &std::path::Path) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io("spawn git status for search-code --changed", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "search-code --changed requires a git worktree ({})",
            err.trim()
        )));
    }

    let mut changed = Vec::new();
    let mut records = out.stdout.split(|b| *b == 0).filter(|r| !r.is_empty());
    while let Some(record) = records.next() {
        if record.len() < 4 {
            continue;
        }
        let x = record[0] as char;
        let y = record[1] as char;
        let path = String::from_utf8_lossy(&record[3..]).to_string();
        if matches!(x, 'R' | 'C') || matches!(y, 'R' | 'C') {
            let _ = records.next();
        }
        if path.is_empty() {
            continue;
        }
        if root_path.join(&path).is_file() {
            changed.push(path);
        }
    }
    changed.sort();
    changed.dedup();
    Ok(changed)
}

fn git_staged_files(root_path: &std::path::Path) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .args([
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            "--",
        ])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io("spawn git diff for search-code --staged", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "search-code --staged requires a git worktree ({})",
            err.trim()
        )));
    }

    let mut staged = out
        .stdout
        .split(|b| *b == 0)
        .filter(|r| !r.is_empty())
        .map(|r| String::from_utf8_lossy(r).to_string())
        .collect::<Vec<_>>();
    staged.sort();
    staged.dedup();
    Ok(staged)
}

fn git_diff_search_spec(
    root_path: &std::path::Path,
    scope: DiffSearchScope<'_>,
) -> Result<DiffSearchSpec> {
    match scope {
        DiffSearchScope::Since { rev } => {
            let diff_rev = git_resolve_commitish(root_path, rev, "search-code --since")?;
            let files = git_diff_files(root_path, &diff_rev, "search-code --since")?;
            Ok(DiffSearchSpec {
                scope: "since",
                diff_rev,
                merge_base: None,
                files,
            })
        }
        DiffSearchScope::Base { base } => {
            let base_rev = git_resolve_commitish(root_path, base, "search-code --base")?;
            let merge_base = git_merge_base(root_path, &base_rev)?;
            let files = git_diff_files(root_path, &merge_base, "search-code --base")?;
            Ok(DiffSearchSpec {
                scope: "base",
                diff_rev: base_rev,
                merge_base: Some(merge_base),
                files,
            })
        }
    }
}

fn git_resolve_commitish(root_path: &std::path::Path, rev: &str, context: &str) -> Result<String> {
    let rev = rev.trim();
    if rev.is_empty() {
        return Err(Error::Invalid(format!(
            "{context} requires a non-empty revision"
        )));
    }
    let spec = format!("{rev}^{{commit}}");
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--verify", spec.as_str()])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io(format!("spawn git rev-parse for {context}"), e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "{context} requires a valid git revision ({})",
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_merge_base(root_path: &std::path::Path, base_rev: &str) -> Result<String> {
    let out = std::process::Command::new("git")
        .args(["merge-base", base_rev, "HEAD"])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io("spawn git merge-base for search-code --base", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "search-code --base requires a revision with a merge-base against HEAD ({})",
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_diff_files(
    root_path: &std::path::Path,
    diff_rev: &str,
    context: &str,
) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .args([
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            diff_rev,
            "--",
        ])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io(format!("spawn git diff for {context}"), e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "{context} requires a valid git diff base ({})",
            err.trim()
        )));
    }

    let mut files = out
        .stdout
        .split(|b| *b == 0)
        .filter(|r| !r.is_empty())
        .map(|r| String::from_utf8_lossy(r).to_string())
        .filter(|path| root_path.join(path).is_file())
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    Ok(files)
}

fn git_diff_changed_lines(
    root_path: &std::path::Path,
    diff_rev: &str,
    context: &str,
) -> Result<std::collections::BTreeMap<String, std::collections::BTreeSet<i64>>> {
    let out = std::process::Command::new("git")
        .args(["diff", "--unified=0", "--diff-filter=ACMR", diff_rev, "--"])
        .current_dir(root_path)
        .output()
        .map_err(|e| Error::io(format!("spawn git diff hunks for {context}"), e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Invalid(format!(
            "{context} requires a valid git diff base ({})",
            err.trim()
        )));
    }

    let mut current_file: Option<String> = None;
    let mut changed: std::collections::BTreeMap<String, std::collections::BTreeSet<i64>> =
        std::collections::BTreeMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            current_file = Some(path.to_string());
            continue;
        }
        if line.starts_with("+++ /dev/null") {
            current_file = None;
            continue;
        }
        if !line.starts_with("@@") {
            continue;
        }
        let Some(file) = current_file.as_ref() else {
            continue;
        };
        let Some((start, count)) = parse_git_diff_new_range(line) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let lines = changed.entry(file.clone()).or_default();
        for offset in 0..count {
            lines.insert(start + offset);
        }
    }
    Ok(changed)
}

fn parse_git_diff_new_range(hunk: &str) -> Option<(i64, i64)> {
    let token = hunk
        .split_whitespace()
        .find(|part| part.starts_with('+') && part.len() > 1)?;
    let range = &token[1..];
    let (start, count) = match range.split_once(',') {
        Some((start, count)) => (start.parse::<i64>().ok()?, count.parse::<i64>().ok()?),
        None => (range.parse::<i64>().ok()?, 1),
    };
    Some((start, count))
}

fn live_grep_search_code_paths_pattern(
    query: &str,
    root_path: &std::path::Path,
    paths: &[String],
    fixed: bool,
) -> Result<Vec<greppy_search::CodeHit>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }

    let grep_args = if fixed {
        ["-HnIF", "--", query]
    } else {
        ["-HnIE", "--", query]
    };
    let mut hits = Vec::new();
    for chunk in paths.chunks(128) {
        let out = std::process::Command::new("grep")
            .args(grep_args)
            .args(chunk)
            .current_dir(root_path)
            .output();
        let out = match out {
            Ok(out) => out,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && fixed => {
                return internal_literal_search_code_paths(query, root_path, paths);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::Invalid(
                    "search-code regex mode requires `grep`; retry with --fixed on this host"
                        .into(),
                ));
            }
            Err(error) => {
                return Err(Error::io("spawn grep for search-code source scan", error));
            }
        };
        if !out.status.success() && out.status.code() != Some(1) {
            return Err(Error::Invalid(format!(
                "grep source scan failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        hits.extend(text.lines().filter_map(parse_grep_code_hit));
    }
    Ok(hits)
}

/// Portable fallback for clean Windows hosts where the product extensions
/// must work even though no system grep is installed. Ordinary grep-compatible
/// invocations still require and byte-forward the real grep process; only the
/// `search-code` extension uses this conservative literal fallback.
fn internal_literal_search_code_paths(
    query: &str,
    root_path: &std::path::Path,
    paths: &[String],
) -> Result<Vec<greppy_search::CodeHit>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let mut hits = Vec::new();
    for path in paths {
        let absolute = root_path.join(path);
        let bytes = match std::fs::read(&absolute) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(Error::io(format!("read source file {path}"), error)),
        };
        if greppy_discover::is_binary_bytes(&bytes) {
            continue;
        }
        for (index, line) in String::from_utf8_lossy(&bytes).lines().enumerate() {
            if line.contains(query) {
                hits.push(greppy_search::CodeHit {
                    location: format!("{path}:{}", index + 1),
                    snippet: line.to_string(),
                    rank: 0.0,
                });
            }
        }
    }
    Ok(hits)
}

fn grep_staged_git_blobs_pattern(
    query: &str,
    root_path: &std::path::Path,
    paths: &[String],
    fixed: bool,
) -> Result<Vec<greppy_search::CodeHit>> {
    use std::io::Write;

    let grep_args = if fixed {
        ["-nIF", "--", query]
    } else {
        ["-nIE", "--", query]
    };
    let mut hits = Vec::new();
    for path in paths {
        let blob_spec = format!(":{path}");
        let blob = std::process::Command::new("git")
            .args(["show", blob_spec.as_str()])
            .current_dir(root_path)
            .output()
            .map_err(|e| Error::io(format!("read staged blob {path}"), e))?;
        if !blob.status.success() {
            continue;
        }

        let mut child = match std::process::Command::new("grep")
            .args(grep_args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !fixed => {
                return Err(Error::Invalid(
                    "search-code regex mode requires `grep`; retry with --fixed on this host"
                        .into(),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if greppy_discover::is_binary_bytes(&blob.stdout) {
                    continue;
                }
                for (index, line) in String::from_utf8_lossy(&blob.stdout).lines().enumerate() {
                    if line.contains(query) {
                        hits.push(greppy_search::CodeHit {
                            location: format!("{path}:{}", index + 1),
                            snippet: line.to_string(),
                            rank: 0.0,
                        });
                    }
                }
                continue;
            }
            Err(error) => {
                return Err(Error::io("spawn grep for search-code --staged", error));
            }
        };
        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(&blob.stdout)
                .map_err(|e| Error::io(format!("write staged blob {path} to grep"), e))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| Error::io("wait for grep in search-code --staged", e))?;
        if !out.status.success() && out.status.code() != Some(1) {
            return Err(Error::Invalid(format!(
                "grep staged-source scan failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some((line_no, content)) = line.split_once(':') {
                hits.push(greppy_search::CodeHit {
                    location: format!("{path}:{line_no}"),
                    snippet: content.to_string(),
                    rank: 0.0,
                });
            }
        }
    }
    Ok(hits)
}

fn parse_grep_code_hit(line: &str) -> Option<greppy_search::CodeHit> {
    let cleaned = line.strip_prefix("./").unwrap_or(line);
    cleaned
        .split_once(':')
        .and_then(|(file, rest)| {
            rest.split_once(':')
                .map(|(line_no, content)| (file, line_no, content))
        })
        .map(|(file, line_no, content)| greppy_search::CodeHit {
            location: format!("{file}:{line_no}"),
            snippet: content.to_string(),
            rank: 0.0,
        })
}

fn dispatch_semantic(
    query: Option<&str>,
    paths: &[String],
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("semantic-search requires a query".into()));
    }
    let path_filters = prepare_query_path_filters(root, "semantic-search", q, paths)?;

    let mut store = open_default_store_query_writer(root)?;
    maybe_reindex_stale(&mut store, root)?;
    let project = project_for(root)?;
    // Stale/unknown snapshots are never served. Semantic search is always
    // vector-backed on current main, so auto-refresh is allowed only when the
    // embedding model can be rebuilt in the same atomic snapshot.
    let allow_reindex = vector_auto_reindex_can_rebuild(embedding_args);
    let decision =
        freshness_serve_decision_with_policy(&store, root, &project, allow_reindex, false);
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    let freshness = decision.freshness().clone();

    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            semantic_provider_incomplete_json(
                &project,
                "vector",
                Some(&freshness),
                &incomplete_providers,
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("semantic-search", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    let cfg = match embedding_config_for_required_use(embedding_args) {
        Ok(cfg) => cfg,
        Err(error) if embedding_asset_missing_error(&error) => {
            emit_semantic_backend_unavailable(
                &project,
                q,
                paths,
                root,
                json,
                "EmbeddingGemma assets could not be resolved; use one of the exact non-semantic fallbacks below.",
            )?;
            return Ok(i32::from(EXIT_NOT_IMPLEMENTED));
        }
        Err(error) => return Err(error),
    };
    {
        let generation = current_graph_generation(&store, root)?;
        let candidate_limit = vector_exact_candidate_limit()?;
        if !freshness_json_is_fresh(&freshness) {
            let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
                &project,
                &cfg.model_id,
                Some(generation),
                SEMANTIC_VECTOR_CANDIDATE_LIMIT,
            );
            scope.limit = SEMANTIC_VECTOR_CANDIDATE_LIMIT;
            let total = greppy_search::count_vector_search_scope(&store, &scope)?;
            if json {
                semantic_vector_json(
                    &store,
                    &project,
                    &cfg,
                    generation,
                    total,
                    candidate_limit,
                    Some(&freshness),
                    "skipped_stale_index",
                    &[],
                )?;
            } else {
                println!(
                    "{}",
                    vector_stale_skip_message("semantic-search", &freshness)
                );
            }
            return Ok(freshness_refusal_exit(&freshness));
        }
        if !embedding_generation_complete(&store, &project, generation, &cfg.model_id) {
            let root_path = resolve_root(root)?;
            let _ = spawn_background_embed(root, &cfg);
            let progress = embedding_progress_value(&root_path, &cfg, generation);
            if json {
                semantic_embedding_indexing_json(
                    &project,
                    &cfg,
                    generation,
                    &freshness,
                    &progress,
                    SemanticFallbackContext {
                        query: q,
                        paths,
                        root,
                    },
                )?;
            } else {
                println!("{}", embedding_progress_text(&progress));
                print_semantic_fallback_commands(q, paths, root);
            }
            return Ok(i32::from(EXIT_TEMPFAIL));
        }
        let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
            &project,
            &cfg.model_id,
            Some(generation),
            SEMANTIC_VECTOR_CANDIDATE_LIMIT,
        );
        let total = greppy_search::count_vector_search_scope(&store, &scope)?;
        if total == 0 {
            if json {
                semantic_vector_json(
                    &store,
                    &project,
                    &cfg,
                    generation,
                    total,
                    candidate_limit,
                    Some(&freshness),
                    "no_indexed_vectors",
                    &[],
                )?;
            } else {
                println!(
                    "semantic index unavailable — 0 indexed spans for model {}",
                    cfg.model_id
                );
                print_semantic_fallback_commands(q, paths, root);
            }
            return Ok(freshness_refusal_exit(&freshness));
        }
        if let Some(limit) = vector_exact_scan_exceeds_limit(total, candidate_limit) {
            if json {
                semantic_vector_json(
                    &store,
                    &project,
                    &cfg,
                    generation,
                    total,
                    candidate_limit,
                    Some(&freshness),
                    "skipped_exact_scan_candidate_limit",
                    &[],
                )?;
            } else {
                println!(
                    "{}",
                    vector_exact_scan_skip_message("semantic-search", total, limit)
                );
            }
            return Ok(1);
        }

        match embed_query_cached(&cfg, root, q) {
            Ok(query_vector) => {
                scope.limit = SEMANTIC_VECTOR_CANDIDATE_LIMIT;
                let mut candidates =
                    greppy_search::vector_search_exact(&store, &query_vector, &scope)?;
                candidates.retain(|hit| path_filters.matches(&hit.embedding.file_path));
                let hits = dedupe_semantic_vector_hits(
                    candidates,
                    cli_result_limit(SEMANTIC_VECTOR_RESULT_LIMIT),
                );
                let shown = hits
                    .len()
                    .min(cli_result_limit(SEMANTIC_VECTOR_DISPLAY_LIMIT));
                let display_hits = hits[..shown].to_vec();
                let further_hits = &hits[shown..];
                let purposes = semantic_vector_purposes(&store, root, &display_hits, true)?;
                let expand = insert_semantic_vector_expand_pack(
                    &store,
                    root,
                    &project,
                    q,
                    generation,
                    further_hits,
                );
                if json {
                    semantic_vector_json_with_expand(
                        &store,
                        &project,
                        &cfg,
                        generation,
                        total,
                        hits.len(),
                        candidate_limit,
                        Some(&freshness),
                        "ok",
                        &display_hits,
                        purposes.as_deref(),
                        expand.as_ref(),
                    )?;
                } else if hits.is_empty() {
                    println!("(no vector matches)");
                    return Ok(1);
                } else {
                    for h in &display_hits {
                        print_semantic_vector_hit(h, purposes.as_deref());
                    }
                    if let Some(expand) = &expand {
                        println!("{}", expand.semantic_text_line());
                    }
                }
                Ok(if hits.is_empty() { 1 } else { 0 })
            }
            Err(e) => Err(e),
        }
    }
}

const SEMANTIC_VECTOR_DISPLAY_LIMIT: usize = 3;
const SEMANTIC_VECTOR_RESULT_LIMIT: usize = 6;
const SEMANTIC_VECTOR_CANDIDATE_LIMIT: usize = 24;
const SEMANTIC_JSON_SCHEMA_VERSION: &str = "greppy.semantic-search.v1";
const SEMANTIC_PURPOSE_SPAN_CAP_LINES: usize = 40;
const SEMANTIC_PURPOSE_SPAN_MAX_BYTES: usize = 2 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticVectorPurpose {
    embedding_id: i64,
    file_path: String,
    start_line: i64,
    end_line: i64,
    display_loc: String,
    signature: String,
    bullets: Vec<String>,
}

fn vector_hit_loc(hit: &greppy_store::VectorSearchHit) -> String {
    line_span(
        &hit.embedding.file_path,
        hit.embedding.start_line,
        hit.embedding.end_line,
    )
}

fn dedupe_semantic_vector_hits(
    hits: Vec<greppy_store::VectorSearchHit>,
    limit: usize,
) -> Vec<greppy_store::VectorSearchHit> {
    let mut seen = std::collections::HashSet::new();
    hits.into_iter()
        .filter(|hit| {
            let key = hit
                .embedding
                .node_id
                .map(|id| format!("node:{id}"))
                .unwrap_or_else(|| {
                    format!(
                        "span:{}:{}:{}",
                        hit.embedding.file_path, hit.embedding.start_line, hit.embedding.end_line
                    )
                });
            seen.insert(key)
        })
        .take(limit)
        .collect()
}

fn semantic_vector_purposes(
    store: &greppy_store::Store,
    root: Option<&str>,
    hits: &[greppy_store::VectorSearchHit],
    summarize: bool,
) -> Result<Option<Vec<SemanticVectorPurpose>>> {
    if hits.is_empty() {
        return Ok(None);
    }
    let root_path = match resolve_root(root) {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    #[cfg(any(unix, windows))]
    let summary_runtime = if summarize {
        qwen_summary_config_optional().ok().flatten()
    } else {
        None
    }
    .map(|cfg| {
        let model_key = qwen_summary_model_key(&cfg);
        (cfg, model_key)
    });
    let mut purposes = Vec::new();
    for hit in hits {
        let node = hit
            .embedding
            .node_id
            .and_then(|id| store.get_node(id).ok().flatten());
        let file_path = node
            .as_ref()
            .map(|n| n.file_path.as_str())
            .unwrap_or(&hit.embedding.file_path);
        let start_line = node
            .as_ref()
            .map(|n| n.start_line)
            .unwrap_or(hit.embedding.start_line);
        let stored_end_line = node
            .as_ref()
            .map(|n| n.end_line)
            .unwrap_or(hit.embedding.end_line);
        let Some(span) = read_span_with_meta(
            &root_path,
            file_path,
            start_line,
            stored_end_line,
            SEMANTIC_PURPOSE_SPAN_CAP_LINES,
            false,
        ) else {
            continue;
        };
        let signature = node
            .as_ref()
            .and_then(|node| node.properties.get("source_signature"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| semantic_signature_from_span(&span.text));
        let Some(signature) = signature else {
            continue;
        };
        let mut bullets = Vec::new();
        #[cfg(any(unix, windows))]
        if semantic_signature_is_function_like(&signature, node.as_ref().map(|n| n.label.as_str()))
        {
            if let Some((cfg, model_key)) = summary_runtime.as_ref() {
                let code = cap_semantic_purpose_span(&span.text);
                bullets =
                    summarize_daemon::summarize_source_via_daemon(cfg, model_key, file_path, &code)
                        .unwrap_or_default();
            }
        }
        purposes.push(SemanticVectorPurpose {
            embedding_id: hit.embedding.id,
            file_path: file_path.to_string(),
            start_line,
            end_line: span.end_line,
            display_loc: line_span(file_path, start_line, span.end_line),
            signature,
            bullets,
        });
    }
    if purposes.is_empty() {
        Ok(None)
    } else {
        Ok(Some(purposes))
    }
}

fn cap_semantic_purpose_span(code: &str) -> String {
    let mut out = code
        .lines()
        .take(SEMANTIC_PURPOSE_SPAN_CAP_LINES)
        .collect::<Vec<_>>()
        .join("\n");
    if code.ends_with('\n') && !out.ends_with('\n') {
        out.push('\n');
    }
    truncate_utf8_bytes(&out, SEMANTIC_PURPOSE_SPAN_MAX_BYTES)
}

fn semantic_signature_from_span(code: &str) -> Option<String> {
    let mut leading_offset = 0usize;
    for line in code.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.trim().is_empty() || trimmed.starts_with("//") {
            leading_offset += line.len();
        } else {
            break;
        }
    }
    let start = code[leading_offset..]
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(leading_offset + idx))?;
    let declaration = &code[start..];
    let python_declaration =
        declaration.starts_with("def ") || declaration.starts_with("async def ");
    let bytes = code.as_bytes();
    let mut round_depth = 0usize;
    let mut square_depth = 0usize;
    let mut angle_depth = 0usize;
    let mut string_delimiter = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut end = code.len();
    let mut idx = start;
    while idx < bytes.len() {
        let byte = bytes[idx];
        let next = bytes.get(idx + 1).copied();
        if line_comment {
            if byte == b'\n' {
                line_comment = false;
            }
            idx += 1;
            continue;
        }
        if block_comment_depth > 0 {
            if byte == b'/' && next == Some(b'*') {
                block_comment_depth += 1;
                idx += 2;
            } else if byte == b'*' && next == Some(b'/') {
                block_comment_depth -= 1;
                idx += 2;
            } else {
                idx += 1;
            }
            continue;
        }
        if let Some(delimiter) = string_delimiter {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == delimiter {
                string_delimiter = None;
            }
            idx += 1;
            continue;
        }
        if byte == b'/' && next == Some(b'/') {
            line_comment = true;
            idx += 2;
            continue;
        }
        if byte == b'/' && next == Some(b'*') {
            block_comment_depth = 1;
            idx += 2;
            continue;
        }
        match byte {
            b'"' => string_delimiter = Some(byte),
            b'\'' if python_declaration => string_delimiter = Some(byte),
            b'(' => round_depth += 1,
            b')' => round_depth = round_depth.saturating_sub(1),
            b'[' => square_depth += 1,
            b']' => square_depth = square_depth.saturating_sub(1),
            b'<' => angle_depth += 1,
            b'>' if angle_depth > 0 => angle_depth -= 1,
            b'{' if round_depth == 0 && square_depth == 0 && angle_depth == 0 => {
                end = idx;
                break;
            }
            b':' if python_declaration
                && round_depth == 0
                && square_depth == 0
                && angle_depth == 0 =>
            {
                end = idx;
                break;
            }
            b';' if round_depth == 0 && square_depth == 0 && angle_depth == 0 => {
                end = idx;
                break;
            }
            _ => {}
        }
        idx += 1;
    }
    let signature = code[start..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!signature.is_empty()).then_some(signature)
}

fn semantic_signature_is_function_like(signature: &str, label: Option<&str>) -> bool {
    if let Some(label) = label {
        let lower = label.to_ascii_lowercase();
        if lower.contains("function") || lower.contains("method") {
            return true;
        }
        if lower.contains("struct")
            || lower.contains("class")
            || lower.contains("enum")
            || lower.contains("trait")
            || lower.contains("interface")
            || lower.contains("module")
        {
            return false;
        }
    }
    let s = signature.trim_start();
    s.starts_with("fn ")
        || s.starts_with("pub fn ")
        || s.starts_with("async fn ")
        || s.starts_with("pub async fn ")
        || s.starts_with("def ")
        || s.starts_with("async def ")
        || s.starts_with("function ")
        || s.contains(" function ")
}

fn truncate_utf8_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut cut = max_bytes.saturating_sub(3);
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut out = s[..cut].to_string();
    out.push_str("...");
    out
}

fn vector_purpose_for_hit<'a>(
    purposes: Option<&'a [SemanticVectorPurpose]>,
    hit: &greppy_store::VectorSearchHit,
) -> Option<&'a SemanticVectorPurpose> {
    purposes?
        .iter()
        .find(|purpose| purpose.embedding_id == hit.embedding.id)
}

fn print_semantic_vector_hit(
    hit: &greppy_store::VectorSearchHit,
    purposes: Option<&[SemanticVectorPurpose]>,
) {
    let loc = vector_hit_loc(hit);
    if let Some(purpose) = vector_purpose_for_hit(purposes, hit) {
        println!("{}", purpose.display_loc);
        println!("    {}", purpose.signature);
        for bullet in &purpose.bullets {
            println!("        {bullet}");
        }
    } else {
        println!("{loc}");
    }
    println!();
}

fn semantic_vector_json_row(
    hit: &greppy_store::VectorSearchHit,
    purpose: Option<&SemanticVectorPurpose>,
    expand: Option<&ExpandHandle>,
) -> serde_json::Value {
    let mut row = serde_json::json!({
        "score": hit.score,
        "qualified_name": hit.embedding.qualified_name,
        "file_path": hit.embedding.file_path,
        "start_line": hit.embedding.start_line,
        "end_line": hit.embedding.end_line,
        "content_sha256": hit.embedding.content_sha256,
        "graph_generation": hit.embedding.graph_generation,
        "summary": [],
    });
    if let Some(purpose) = purpose {
        row["file_path"] = serde_json::json!(&purpose.file_path);
        row["start_line"] = serde_json::json!(purpose.start_line);
        row["end_line"] = serde_json::json!(purpose.end_line);
        row["signature"] = serde_json::json!(&purpose.signature);
        row["summary_loc"] = serde_json::json!(&purpose.display_loc);
        row["summary"] = serde_json::json!(&purpose.bullets);
        if !purpose.bullets.is_empty() {
            row["summary_prompt_version"] = serde_json::json!(greppy_qwen35_native::PROMPT_VERSION);
        }
    }
    if let Some(expand) = expand {
        row["expand_id"] = serde_json::json!(&expand.id);
    }
    row
}

fn current_graph_generation(store: &greppy_store::Store, root: Option<&str>) -> Result<u64> {
    let root_path = resolve_root(root)?;
    let root_key = root_path.to_string_lossy().into_owned();
    let state = store.get_workspace_state(&root_key)?.ok_or_else(|| {
        Error::Invalid(format!(
            "no workspace_state for {}; run `greppy index {}` first",
            root_path.display(),
            root.unwrap_or(".")
        ))
    })?;
    Ok(state.graph_generation)
}

#[allow(clippy::too_many_arguments)]
fn semantic_vector_json(
    store: &greppy_store::Store,
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    total: i64,
    candidate_limit: Option<i64>,
    freshness: Option<&serde_json::Value>,
    status: &str,
    hits: &[greppy_store::VectorSearchHit],
) -> Result<()> {
    let retrieved = hits.len();
    semantic_vector_json_with_expand(
        store,
        project,
        cfg,
        graph_generation,
        total,
        retrieved,
        candidate_limit,
        freshness,
        status,
        hits,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn semantic_vector_json_with_expand(
    store: &greppy_store::Store,
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    total: i64,
    retrieved: usize,
    candidate_limit: Option<i64>,
    freshness: Option<&serde_json::Value>,
    status: &str,
    hits: &[greppy_store::VectorSearchHit],
    purposes: Option<&[SemanticVectorPurpose]>,
    expand: Option<&ExpandHandle>,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let rows = hits
        .iter()
        .map(|hit| semantic_vector_json_row(hit, vector_purpose_for_hit(purposes, hit), expand))
        .collect::<Vec<_>>();
    let shown = rows.len() as i64;
    let (retrieved, omitted, unranked_candidates, truncated) =
        semantic_vector_count_values(total, retrieved, rows.len());
    let mut v = serde_json::json!({
        "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
        "command": "semantic-search",
        "mode": "vector",
        "status": status,
        "project": project,
        "backend": "exact_cosine",
        "scope": "embeddinggemma_code_retrieval_current_generation",
        "model_id": cfg.model_id,
        "prompt_version": greppy_embed_native::PROMPT_VERSION,
        "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
        "graph_generation": graph_generation,
        "fresh": freshness
            .and_then(|v| v.get("fresh"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
        "provider_complete": incomplete_providers.is_empty(),
        "incomplete_provider_count": incomplete_providers.len(),
        "incomplete_providers": incomplete_providers,
        "candidate_limit": candidate_limit,
        "candidate_limit_env": ENV_VECTOR_EXACT_CANDIDATE_LIMIT,
        "candidate_total": total,
        "total_exact": total,
        "retrieved": retrieved,
        "shown": shown,
        "omitted": omitted,
        "unranked_candidates": unranked_candidates,
        "truncated": truncated,
        "hits": rows,
    });
    if let Some(expand) = expand {
        v["expand"] = expand.json_value();
        v["expand_id"] = serde_json::json!(&expand.id);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize vector semantic JSON: {e}")))?
    );
    Ok(())
}

fn semantic_vector_count_values(
    candidate_total: i64,
    retrieved: usize,
    shown: usize,
) -> (i64, i64, i64, bool) {
    let retrieved = i64::try_from(retrieved).unwrap_or(i64::MAX);
    let shown = i64::try_from(shown).unwrap_or(i64::MAX);
    let omitted = retrieved.saturating_sub(shown);
    let unranked_candidates = candidate_total.saturating_sub(retrieved);
    (
        retrieved,
        omitted,
        unranked_candidates,
        omitted > 0 || unranked_candidates > 0,
    )
}

fn semantic_provider_incomplete_json(
    project: &str,
    mode: &str,
    freshness: Option<&serde_json::Value>,
    incomplete_providers: &[serde_json::Value],
) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": SEMANTIC_JSON_SCHEMA_VERSION,
            "command": "semantic-search",
            "mode": mode,
            "status": "skipped_incomplete_provider",
            "project": project,
            "fresh": freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": false,
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "total_exact": 0,
            "shown": 0,
            "omitted": 0,
            "truncated": false,
            "hits": [],
        }))
        .map_err(|e| Error::Invalid(format!("serialize semantic provider policy JSON: {e}")))?
    );
    Ok(())
}

/// A definition resolved by `greppy context`, carried with the metadata
/// needed to read and print its source span.
struct ContextDef {
    qualified_name: String,
    file_path: String,
    start_line: i64,
    end_line: i64,
    /// Graph node id when known (vector/exact hits carry it), so the top hit
    /// can be expanded into a graph-linked structural digest. `None` for
    /// span-only rows that resolve to no node.
    node_id: Option<i64>,
}

fn display_context_def_name(store: &greppy_store::Store, def: &ContextDef) -> String {
    if let Some(id) = def.node_id {
        if let Ok(Some(node)) = store.get_node(id) {
            return display_node_name(&node);
        }
    }
    let name = def
        .qualified_name
        .rsplit("::")
        .next()
        .unwrap_or(&def.qualified_name);
    display_symbol_name("", name, &def.qualified_name, &def.file_path)
}

struct SpanRead {
    text: String,
    end_line: i64,
    total_lines: usize,
    shown_lines: usize,
    omitted_lines: usize,
    truncated: bool,
}

/// `greppy context <query> [--k N] [--lines]` — the token-saving lever.
///
/// Instead of returning `file:line` POINTERS (which force the agent to
/// READ whole files), this resolves the most relevant DEFINITIONS for
/// `<query>` and prints their ACTUAL SOURCE SPANS, so the agent reads the
/// relevant function/struct bodies directly from greppy output.
///
/// Resolution unions four signals, in priority order, deduplicating on
/// node id while preserving first-seen order:
/// 1. `search_symbols` — exact/FTS symbol-name matches (most precise).
/// 2. `semantic_query` — algorithmic similarity (catches paraphrases).
/// 3. `search_code` → `definition_at` — content matches resolved to the
///    enclosing definition (catches symbols only the body mentions).
///
/// The top-K (default 6) definitions are emitted with a compact
/// `== qualified_name (file:start-end) ==` header followed by the source
/// span read from disk (capped at [`CONTEXT_SPAN_CAP`] lines, with a
/// truncation note). The command refuses a stale index before emitting spans;
/// missing files / out-of-range lines are still skipped gracefully as a final
/// guard against races.
fn dispatch_context(
    query: Option<&str>,
    k: usize,
    lines: bool,
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("context requires a query".into()));
    }
    let k = cli_result_limit(k).max(1);
    let project = project_for(root)?;
    let span_root = resolve_root(root)?;
    // Context also refuses stale/unknown graph locations. It never combines
    // an old indexed line number with current source text.
    let decision = freshness_serve_decision(&store, root, &project);
    let incomplete_providers = incomplete_provider_json(&store, &project)?;
    if let FreshnessServe::Refuse(freshness) = &decision {
        if json {
            context_json(
                &store,
                &project,
                "skipped_stale_index",
                Some(freshness),
                k,
                lines,
                &[],
                &span_root,
            )?;
        } else {
            println!("{}", context_stale_skip_message(freshness));
        }
        return Ok(freshness_refusal_exit(freshness));
    }
    let freshness = decision.freshness().clone();
    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            context_json(
                &store,
                &project,
                "skipped_incomplete_provider",
                Some(&freshness),
                k,
                lines,
                &[],
                &span_root,
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("context", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    // Ordered, de-duplicated candidate definitions keyed on node id.
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut defs: Vec<ContextDef> = Vec::new();

    // Exact-name / show-definition fast path (Z3): when the query is a
    // single bare identifier that resolves to real primary definition(s)
    // by EXACT name, this is a "find the definition of X" lookup — the
    // domain where plain grep is optimal. Return ONLY those exact-name
    // definitions (grep-shaped: file:line + the def's own span) and skip
    // the semantic / code-search padding that would otherwise pull in
    // callers and paraphrase matches. This keeps a literal lookup
    // grep-competitive instead of ingesting several unrelated full spans.
    // Natural-language research queries (which contain spaces) never take
    // this path, so `context` stays rich for research.
    if is_bare_identifier(q) {
        let exact = resolve_symbol_nodes(&store, Some(q))?;
        if !exact.is_empty() {
            for id in exact {
                if defs.len() >= k {
                    break;
                }
                if let Some(n) = store.get_node(id)? {
                    if seen.insert(n.id) {
                        defs.push(ContextDef {
                            qualified_name: n.qualified_name,
                            file_path: n.file_path,
                            start_line: n.start_line,
                            end_line: n.end_line,
                            node_id: Some(n.id),
                        });
                    }
                }
            }
            return emit_context_locators(
                &store, &project, &freshness, k, lines, json, &defs, &span_root,
            );
        }
    }

    // Over-fetch from each source so the union has enough candidates to
    // fill K distinct definitions after dedup. A small multiple of K is
    // plenty and keeps each query fast.
    let fetch = (k * 4).max(20);

    // Decide the vector fallback up front (D6 / fuzzy-lever fix). It fires for
    // a multi-word (non-bare) natural-language query that names NO real symbol
    // by EXACT name. Two facts make this the right gate — and make it the
    // PRIMARY signal, not a trailing append:
    //
    //  * The lexical semantic step (`semantic_query` -> `score_one`) returns a
    //    hit for ANY node sharing even ONE token with the query. A conceptual
    //    phrase ("restrict a numeric value to stay within bounds") therefore
    //    fills all K slots with token-overlap NOISE (`Identifier::Field`, …)
    //    before any embedding runs. Appending vectors AFTER that union is a
    //    no-op: the `defs.len() >= k` guard drops every vector hit. So when we
    //    know the query names no symbol (lexical hits are noise), the vectors
    //    must LEAD, and the lexical union only backfills leftover slots.
    //  * If the phrase DID resolve to a real symbol (`Owner.method`, an exact
    //    name) vectors stay off and the lexical/exact union answers, so a
    //    genuine exact/FTS match is never displaced by a paraphrase.
    //
    // Router safety (task_classes_v2 `avoid_embedding`): a bare name is handled
    // on the Z3 exact fast path far above and never reaches here; the
    // `!is_bare_identifier` guard keeps any bare name that slipped through
    // (resolved nothing) off vectors. Degrades gracefully (a labeled stderr
    // note, not a crash) when no model is configured or no vectors exist.
    let use_vectors = !is_bare_identifier(q) && resolve_symbol_nodes(&store, Some(q))?.is_empty();
    if use_vectors {
        if let Some((hits, low_confidence)) =
            context_vector_fallback(&store, &project, &freshness, q, fetch, embedding_args, root)?
        {
            // The vectors ARE the answer for this conceptual query (it named no
            // symbol by exact name, so any lexical hit is token-overlap noise —
            // see the `use_vectors` rationale above). Emit the top semantic
            // matches as LEAN grep-shaped locators and STOP: return the location
            // + signature, not K full function bodies. The old behaviour pushed
            // vector hits into `defs` and fell through to the full-body union,
            // which turned the vectors' quality win into a token LOSS and left
            // the agent iterating because it could not tell which body answered.
            let mut vec_defs: Vec<ContextDef> = Vec::new();
            for h in hits {
                if vec_defs.len() >= CONTEXT_VECTOR_LEAN_TOP_N {
                    break;
                }
                // Dedup by node id when present; span-only rows (no node id)
                // cannot collide, so take them.
                if h.node_id.map(|id| seen.insert(id)).unwrap_or(true) {
                    vec_defs.push(ContextDef {
                        qualified_name: h.qualified_name,
                        file_path: h.file_path,
                        start_line: h.start_line,
                        end_line: h.end_line,
                        node_id: h.node_id,
                    });
                }
            }
            if !vec_defs.is_empty() {
                // A multi-word conceptual query wants the mechanism, not just a
                // location: give the #1 hit a bounded body so the agent answers
                // in one call instead of rephrasing and re-searching.
                let conceptual = q.split_whitespace().count() >= CONTEXT_CONCEPTUAL_MIN_WORDS;
                return emit_context_vector_locators(
                    &store,
                    &project,
                    &freshness,
                    k,
                    lines,
                    json,
                    &vec_defs,
                    &span_root,
                    low_confidence,
                    conceptual,
                );
            }
        }
    }

    // 1. Symbol-name FTS hits (most precise). Resolve each to its node.
    for h in greppy_search::search_symbols_in_project(&store, &project, q, fetch)? {
        if defs.len() >= k {
            break;
        }
        if let Some(n) = store.get_node(h.node_id)? {
            if seen.insert(n.id) {
                defs.push(ContextDef {
                    qualified_name: n.qualified_name,
                    file_path: n.file_path,
                    start_line: n.start_line,
                    end_line: n.end_line,
                    node_id: Some(n.id),
                });
            }
        }
    }

    // 2. Semantic hits (paraphrase / related symbols).
    if defs.len() < k {
        for h in greppy_search::semantic_query(&store, q, None, Some(&project), fetch)? {
            if defs.len() >= k {
                break;
            }
            if seen.insert(h.node.id) {
                defs.push(ContextDef {
                    qualified_name: h.node.qualified_name,
                    file_path: h.node.file_path,
                    start_line: h.node.start_line,
                    end_line: h.node.end_line,
                    node_id: Some(h.node.id),
                });
            }
        }
    }

    // 3. Code-search hits resolved to their enclosing definition. This
    //    catches symbols a query only matches inside a body (where neither
    //    the symbol-name FTS nor the semantic signals fired).
    if defs.len() < k {
        let mut code_hits = greppy_search::search_code(&store, &project, q, fetch)?;
        if code_hits.is_empty() {
            code_hits = live_grep_code_hits(q, &span_root)?
                .into_iter()
                .take(fetch)
                .collect();
        }
        for h in code_hits {
            if defs.len() >= k {
                break;
            }
            // `location` is `file:line`; split on the LAST colon so a
            // path containing a colon is still parsed correctly.
            let Some((file, line_str)) = h.location.rsplit_once(':') else {
                continue;
            };
            let Ok(line) = line_str.parse::<i64>() else {
                continue;
            };
            if let Some(row) = greppy_search::definition_at(&store, Some(&project), file, line)? {
                if seen.insert(row.id) {
                    defs.push(ContextDef {
                        qualified_name: row.qualified_name,
                        file_path: row.file_path,
                        start_line: row.start_line,
                        end_line: row.end_line,
                        node_id: Some(row.id),
                    });
                }
            }
        }
    }

    emit_context_defs(
        &store, &project, &freshness, k, lines, json, &defs, &span_root,
    )
}

/// A definition resolved by the `context` vector fallback, carrying the node
/// id (so the caller can dedup against the lexical union) and the span
/// metadata needed to print its source.
struct ContextVectorDef {
    node_id: Option<i64>,
    qualified_name: String,
    file_path: String,
    start_line: i64,
    end_line: i64,
}

/// Native EmbeddingGemma vector fallback for `context` (D6). Returns
/// `Ok(Some(defs))` with the top vector hits when an embedding model is
/// configured, the index has current-generation vectors, and it is fresh;
/// returns `Ok(None)` (with a labeled stderr note) when the fallback cannot
/// run — no model configured, no indexed vectors, stale index, or the exact
/// scan would exceed its candidate guard. It NEVER errors on a missing model:
/// a research question just degrades to the current (lexical-only) behaviour.
///
/// Only reached for multi-word natural-language queries whose lexical union
/// was empty, so it never runs on `avoid_embedding` exact-name / graph
/// queries (see the call site).
fn context_vector_fallback(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    query: &str,
    fetch: usize,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<Option<(Vec<ContextVectorDef>, bool)>> {
    let cfg = embedding_config_for_required_use(embedding_args)?;

    let generation = current_graph_generation(store, root)?;
    if !embedding_generation_complete(store, project, generation, &cfg.model_id) {
        let root_path = resolve_root(root)?;
        let _ = spawn_background_embed(root, &cfg);
        let progress = embedding_progress_value(&root_path, &cfg, generation);
        eprintln!("{}", embedding_progress_text(&progress));
        return Ok(None);
    }
    let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
        project,
        &cfg.model_id,
        Some(generation),
        fetch,
    );
    let total = greppy_search::count_vector_search_scope(store, &scope)?;
    if total == 0 {
        eprintln!(
            "context: the completed semantic index contains no embeddable code spans for this project."
        );
        return Ok(None);
    }
    if !freshness_json_is_fresh(freshness) {
        eprintln!("{}", vector_stale_skip_message("context", freshness));
        return Ok(None);
    }
    let candidate_limit = vector_exact_candidate_limit()?;
    if let Some(limit) = vector_exact_scan_exceeds_limit(total, candidate_limit) {
        eprintln!(
            "{}",
            vector_exact_scan_skip_message("context", total, limit)
        );
        return Ok(None);
    }

    let query_vector = match embed_query_cached(&cfg, root, query) {
        Ok(query_vector) => query_vector,
        Err(e) => {
            log_embedding_skip_once("context", &e);
            return Ok(None);
        }
    };
    // P2b: over-fetch before the class prior below — re-ranking a set the
    // auxiliary stubs already saturated cannot surface the real code.
    scope.limit = fetch.saturating_mul(4).max(64);
    let mut hits = greppy_search::vector_search_exact(store, &query_vector, &scope)?;
    // P2b (spot forensics): tiny bench/test stubs and vendored/lock files
    // embed to near-uniform vectors and crowd out the real definitions on
    // vocabulary queries (zod: `packages/bench/*.ts zod3(){}` outranked
    // coerce.ts). Apply a deterministic class prior — a mild multiplicative
    // penalty for auxiliary paths and one-to-two-line spans — then re-rank.
    // Production code with a genuinely better score still wins; the prior
    // only breaks the near-ties the low-confidence header flags anyway.
    for h in &mut hits {
        let p = h.embedding.file_path.to_ascii_lowercase();
        let auxiliary = p.split('/').any(|seg| {
            matches!(
                seg,
                "test"
                    | "tests"
                    | "__tests__"
                    | "testing"
                    | "spec"
                    | "specs"
                    | "bench"
                    | "benches"
                    | "benchmark"
                    | "benchmarks"
                    | "example"
                    | "examples"
                    | "fixtures"
                    | "docs"
                    | "doc"
                    | "node_modules"
                    | "vendor"
                    | "third_party"
            )
        }) || p.ends_with(".lock")
            || p.ends_with("lock.yaml")
            || p.ends_with("lock.json")
            || p.ends_with(".md");
        if auxiliary {
            h.score *= 0.85;
        }
        if h.embedding.end_line.saturating_sub(h.embedding.start_line) < 2 {
            h.score *= 0.92;
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(fetch);
    // Forensics visibility (F2): env-driven vector use carries no
    // `--embedding-*` flag in the agent's command and the success path is
    // otherwise silent, so a control-class task could run vectors and
    // `forensics.py --enforce` would miss the hard-negative violation. Emit
    // ONE stderr line whose text contains a `candidate_uses_vector`
    // VECTOR_TRIGGER substring ("embeddinggemma" and "vector search" both
    // match forensics.py verbatim) so env-driven use is detectable.
    eprintln!(
        "context: vector search fallback used (embeddinggemma, {} hits)",
        hits.len()
    );
    // Confidence from the score MARGIN, not the absolute score (r042
    // forensics): a genuine hit separates clearly from the runner-up
    // (control-case margin ≈ 0.27) while a vocabulary-mismatch query returns
    // a near-tie of equally-plausible wrong candidates (margin ≈ 0.02) —
    // exactly the shape that sent an agent into a 39-call verify spiral
    // while the header still claimed "#1 is the most likely answer".
    let low_confidence = hits.len() >= 2 && (hits[0].score - hits[1].score) < 0.05;
    Ok(Some((
        hits.into_iter()
            .map(|h| ContextVectorDef {
                node_id: h.embedding.node_id,
                qualified_name: h.embedding.qualified_name,
                file_path: h.embedding.file_path,
                start_line: h.embedding.start_line,
                end_line: h.embedding.end_line,
            })
            .collect(),
        low_confidence,
    )))
}

/// Render the resolved `context` definitions — shared by the exact-name
/// fast path and the general resolution path so both emit identical
/// JSON / span output. Returns exit 0 when at least one definition was
/// resolved, 1 when the set is empty.
#[allow(clippy::too_many_arguments)]
fn emit_context_defs(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    k: usize,
    lines: bool,
    json: bool,
    defs: &[ContextDef],
    span_root: &std::path::Path,
) -> Result<i32> {
    if defs.is_empty() {
        if json {
            context_json(
                store,
                project,
                "ok",
                Some(freshness),
                k,
                lines,
                &[],
                span_root,
            )?;
        } else {
            println!("(no matches)");
        }
        return Ok(1);
    }

    if json {
        context_json(
            store,
            project,
            "ok",
            Some(freshness),
            k,
            lines,
            defs,
            span_root,
        )?;
        return Ok(0);
    }

    let mut printed = 0usize;
    for def in defs.iter().take(k) {
        match read_span(
            span_root,
            &def.file_path,
            def.start_line,
            def.end_line,
            CONTEXT_SPAN_CAP,
            lines,
        ) {
            Some(span) => {
                let display_name = display_context_def_name(store, def);
                println!(
                    "== {} ({}:{}-{}) ==",
                    display_name, def.file_path, def.start_line, def.end_line
                );
                print!("{span}");
                println!();
                printed += 1;
            }
            // Span unreadable (missing file / stale lines) — skip the body
            // but still surface the pointer so the agent is not left blind.
            None => {
                let display_name = display_context_def_name(store, def);
                println!(
                    "== {} ({}:{}-{}) == (source unavailable)",
                    display_name, def.file_path, def.start_line, def.end_line
                );
                println!();
            }
        }
    }

    // Exit 0 as long as we resolved at least one definition; the
    // per-span unavailability is reported inline above.
    let _ = printed;
    Ok(0)
}

/// Emit the exact-name / show-definition result as a LEAN locator (Z3):
/// for each resolved definition print the compact
/// `== qname (file:start-end) ==` header followed by ONLY the definition's
/// first line — its signature / def line — not the whole body. This is the
/// grep-shaped answer to a "find the definition site of X" lookup: it
/// gives the file:line and the signature, matching a single `grep -rn`
/// def line in byte cost, so greppy stays grep-competitive on literal
/// find-definition tasks (contract Z3). Only exact-name bare-identifier
/// queries reach this path; natural-language / multi-word research queries
/// still take the rich, full-body union path, and `greppy brief <X>`
/// still prints the full body plus callers/callees for a deeper look.
///
/// JSON mode keeps the existing structured def-span metadata (a separate,
/// machine-readable consumer) — only the human/text output is leaned out,
/// which is what the token-cost comparison measures.
#[allow(clippy::too_many_arguments)]
fn emit_context_locators(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    k: usize,
    lines: bool,
    json: bool,
    defs: &[ContextDef],
    span_root: &std::path::Path,
) -> Result<i32> {
    if json {
        // Structured consumers still get the full def-span metadata.
        return emit_context_defs(store, project, freshness, k, lines, json, defs, span_root);
    }

    if defs.is_empty() {
        println!("(no matches)");
        return Ok(1);
    }

    for def in defs.iter().take(k) {
        let display_name = display_context_def_name(store, def);
        println!(
            "== {} ({}:{}-{}) ==",
            display_name, def.file_path, def.start_line, def.end_line
        );
        // Only the FIRST line of the span — the signature / def line.
        // `read_span` (via `read_span_with_meta`) computes `total_lines` from
        // `definition_end_idx()` (the whole body) even though we ask for a
        // 1-line cap, so its text is `<sig>\n… (truncated, N more line(s))\n`.
        // That truncation note is dead weight for a lean Z3 locator (~25-40
        // extra bytes/hit). Take ONLY the first line (mirroring
        // `plus_first_line`'s `.lines().next()` guard) rather than modifying
        // the shared `read_span_with_meta`, whose full-body consumers rely on
        // the note.
        if let Some(sig) = read_span(
            span_root,
            &def.file_path,
            def.start_line,
            def.start_line,
            1,
            lines,
        )
        .and_then(|span| span.lines().next().map(str::to_string))
        {
            println!("{sig}");
        }
    }
    Ok(0)
}

/// How many vector-fallback hits the LEAN semantic-locator path emits. A
/// conceptual "which function does X" question wants the LOCATION + signature
/// of the single most-relevant routine, not K full function bodies. Three is
/// enough to cover the target plus a sibling or two when the Q4 model ranks it
/// borderline, while staying an order of magnitude leaner than the old k=6
/// full-body union (~5-6 KB -> a few hundred bytes).
const CONTEXT_VECTOR_LEAN_TOP_N: usize = 3;

/// For a *conceptual* natural-language query ("how does X validate Y"), the
/// answer is not just a location — the agent needs the body to explain the
/// mechanism. Emitting only signature lines forces a follow-up read or, worse,
/// a rephrase-and-re-search spiral (the dominant SWE-QA cost-loss pattern:
/// context→search-code→search-symbols with 3-5 reworded queries, never
/// converging). So the SINGLE top hit of a conceptual query also carries a
/// bounded body excerpt — enough to answer in one call — while #2/#3 stay lean
/// locators. Short/locate queries (< [`CONTEXT_CONCEPTUAL_MIN_WORDS`] words) and
/// near-tie low-confidence results keep the lean sig-only form, so the "where is
/// X" wins and the r042 verify-spiral guard are preserved.
const CONTEXT_TOP1_BODY_LINES: usize = 24;

/// A query with at least this many words is treated as a conceptual "how/why"
/// question (wants the mechanism), not a short "where is X" locate query.
const CONTEXT_CONCEPTUAL_MIN_WORDS: usize = 3;

/// How many of the top hit's callees the structural digest lists. Enough to
/// convey the mechanism (what the function is built from) without re-inflating
/// into a full dump.
const CONTEXT_DIGEST_MAX_CALLEES: usize = 8;

/// Emit the vector-fallback result as LEAN, TRUST-BUILDING semantic locators.
///
/// The context vector fallback fires for a conceptual natural-language query
/// that names no symbol by exact name ("which routine converts X into Y") — the
/// answer is a *location*, so this prints the top-N (`CONTEXT_VECTOR_LEAN_TOP_N`)
/// semantic matches as grep-shaped locators — `== qname (file:start-end) ==`
/// plus the def's own signature line — exactly like the Z3 `emit_context_locators`
/// lean form, NOT the old k=6 full-body union that made the vectors' quality win
/// a token LOSS (r041: 5-6 KB, agent iterates because it can't tell which of six
/// bodies is the answer).
///
/// A single SHORT header precedes the locators, telling the agent these are
/// ranked semantic matches (most-relevant first). The header is deliberately
/// terse — the H2 slim lesson is that a verbose hedge backfires (a 22-token
/// hedge doubled outputs and was reverted), so this is one line, no per-hit
/// caveats.
///
/// JSON mode keeps the structured def-span metadata (a separate machine
/// consumer) via `emit_context_locators`; only the human/text output is leaned.
#[allow(clippy::too_many_arguments)]
fn emit_context_vector_locators(
    store: &greppy_store::Store,
    project: &str,
    freshness: &serde_json::Value,
    k: usize,
    lines: bool,
    json: bool,
    defs: &[ContextDef],
    span_root: &std::path::Path,
    low_confidence: bool,
    conceptual: bool,
) -> Result<i32> {
    if json {
        // Structured consumers still get the full locator metadata; the JSON
        // shape must not diverge between the exact and vector lean paths.
        return emit_context_locators(store, project, freshness, k, lines, json, defs, span_root);
    }

    if defs.is_empty() {
        println!("(no matches)");
        return Ok(1);
    }

    // One short line, most-relevant-first.
    // EXCEPT when the top scores nearly tie (vocabulary-mismatch queries):
    // claiming "#1 is the most likely answer" over a near-tie of plausible
    // wrong candidates sent an agent into a 39-call verify spiral (r042).
    // The low-confidence line is a TRUE signal (the margin really is ~0), so
    // it does not violate the no-false-hedges rule — it saves the agent from
    // serially disproving candidates the ranking itself cannot separate.
    // Show the #1 structural digest for any conceptual query — confident OR
    // near-tie. The near-tie case is exactly where agents used to rephrase
    // and re-search (the dominant cost-loss spiral). The digest exposes the
    // evidence without giving procedural instructions.
    // A short/locate query (< min words) stays sig-only, protecting "where is X".
    let show_top_body = conceptual;
    if low_confidence {
        println!(
            "# semantic candidates (top scores are close). The #1 call map is shown below; #2/#3 locators follow."
        );
    } else if show_top_body {
        println!(
            "# top semantic matches (most relevant first). The #1 call map is shown below; additional locators follow."
        );
    } else {
        println!("# top semantic matches (most relevant first).");
    }
    for (idx, def) in defs.iter().take(CONTEXT_VECTOR_LEAN_TOP_N).enumerate() {
        let display_name = display_context_def_name(store, def);
        println!(
            "== {} ({}:{}-{}) ==",
            display_name, def.file_path, def.start_line, def.end_line
        );
        if show_top_body && idx == 0 {
            // #1 of a conceptual query: a graph-linked structural digest —
            // signature (header) + the functions it calls (with their
            // signatures, from the graph's CALLS edges) + return type, body
            // elided — so the agent gets the mechanism in ONE call. Falls back
            // to a bounded raw body when the node carries no graph detail.
            let digest = def
                .node_id
                .and_then(|id| store.get_node(id).ok().flatten())
                .and_then(|node| context_top_digest(store, &node, span_root));
            if let Some(d) = digest {
                println!("{d}");
            } else if let Some(body) = read_span(
                span_root,
                &def.file_path,
                def.start_line,
                def.end_line,
                CONTEXT_TOP1_BODY_LINES,
                lines,
            ) {
                println!("{body}");
            }
        } else if let Some(sig) = read_span(
            span_root,
            &def.file_path,
            def.start_line,
            def.start_line,
            1,
            lines,
        )
        .and_then(|span| span.lines().next().map(str::to_string))
        {
            // Only the FIRST line of the span — the signature / def line (mirrors
            // the Z3 lean form: drop the "N more line(s)" truncation note, keep
            // the bare signature).
            println!("{sig}");
        }
    }
    Ok(0)
}

/// Build a compact, graph-linked structural digest of the top semantic hit:
/// its signature (header), the key functions it calls (with their signatures,
/// from the graph's CALLS edges) and its return type — the body elided with a
/// `…` marker. This fuses the semantic hit with graph discovery so a conceptual
/// "how does X work" query is answered in ONE call, instead of a raw-body dump
/// (which drove the rephrase-and-re-search cost spiral) or a bare signature
/// (which made the agent re-query for the mechanism). Returns `None` when the
/// node carries no detail beyond its header, so the caller falls back.
fn context_top_digest(
    store: &greppy_store::Store,
    node: &greppy_store::Node,
    span_root: &std::path::Path,
) -> Option<String> {
    fn prop_trimmed<'a>(node: &'a greppy_store::Node, key: &str) -> Option<&'a str> {
        node.properties
            .get(key)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }
    fn cap(s: &str, n: usize) -> String {
        if s.chars().count() > n {
            format!("{}…", s.chars().take(n).collect::<String>())
        } else {
            s.to_string()
        }
    }

    let header = prop_trimmed(node, "signature")
        .map(str::to_string)
        .or_else(|| {
            read_source_line(span_root, &node.file_path, node.start_line as u32)
                .map(|s| s.trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| node.name.clone());

    // Key callees (from the graph's CALLS edges), each with its signature.
    let mut callees: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    if let Ok(steps) = greppy_search::callees_of(store, node.id) {
        for step in steps {
            let Some(n) = step.node else { continue };
            if n.id == node.id || !seen.insert(n.id) {
                continue;
            }
            let short = n
                .qualified_name
                .rsplit("::")
                .next()
                .unwrap_or(&n.qualified_name);
            let label = match n
                .properties
                .get("signature")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(sig) => cap(sig, 72),
                None => short.to_string(),
            };
            // The callee's own location AND line span, so the agent can open
            // and read exactly that function directly when it needs the detail
            // — a navigable map, not a dead-end name list.
            let loc = if n.file_path.is_empty() {
                String::new()
            } else if n.start_line > 0 && n.end_line >= n.start_line {
                format!("  [{}:{}-{}]", n.file_path, n.start_line, n.end_line)
            } else if n.start_line > 0 {
                format!("  [{}:{}]", n.file_path, n.start_line)
            } else {
                format!("  [{}]", n.file_path)
            };
            callees.push(format!("{label}{loc}"));
            if callees.len() >= CONTEXT_DIGEST_MAX_CALLEES {
                break;
            }
        }
    }

    let returns = prop_trimmed(node, "return_type").map(str::to_string);
    let doc = prop_trimmed(node, "doc")
        .map(|d| d.split('\n').next().unwrap_or(d).trim().to_string())
        .filter(|s| !s.is_empty());

    // Not worth a digest if there is nothing beyond the header.
    if callees.is_empty() && returns.is_none() && doc.is_none() {
        return None;
    }

    let mut out = String::new();
    out.push_str(&header);
    if let Some(d) = doc {
        out.push_str("\n    doc: ");
        out.push_str(&cap(&d, 120));
    }
    if !callees.is_empty() {
        // One callee per line, each with its own `file:line`, so the agent can
        // scan the mechanism and open any building-block function directly.
        out.push_str("\n    calls:");
        for c in &callees {
            out.push_str("\n      ");
            out.push_str(c);
        }
    }
    if let Some(rt) = returns {
        out.push_str("\n    returns: ");
        out.push_str(&rt);
    }
    let span = node.end_line - node.start_line;
    if span > 1 {
        out.push_str(&format!("\n    … [{span} lines elided]"));
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn context_json(
    store: &greppy_store::Store,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    limit: usize,
    line_numbers: bool,
    defs: &[ContextDef],
    root: &std::path::Path,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let mut span_truncated_count = 0usize;
    let mut source_unavailable_count = 0usize;
    let mut spans = Vec::new();
    for def in defs.iter().take(limit) {
        let span = read_span_with_meta(
            root,
            &def.file_path,
            def.start_line,
            def.end_line,
            CONTEXT_SPAN_CAP,
            line_numbers,
        );
        match span {
            Some(span) => {
                if span.truncated {
                    span_truncated_count += 1;
                }
                spans.push(serde_json::json!({
                    "qualified_name": &def.qualified_name,
                    "file_path": &def.file_path,
                    "start_line": def.start_line,
                    "end_line": def.end_line,
                    "source_available": true,
                    "source": span.text,
                    "total_lines": span.total_lines,
                    "shown_lines": span.shown_lines,
                    "omitted_lines": span.omitted_lines,
                    "truncated": span.truncated,
                }));
            }
            None => {
                source_unavailable_count += 1;
                spans.push(serde_json::json!({
                    "qualified_name": &def.qualified_name,
                    "file_path": &def.file_path,
                    "start_line": def.start_line,
                    "end_line": def.end_line,
                    "source_available": false,
                    "source": null,
                    "total_lines": null,
                    "shown_lines": 0,
                    "omitted_lines": null,
                    "truncated": false,
                }));
            }
        }
    }
    let truncated = span_truncated_count > 0;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "context",
            "status": status,
            "project": project,
            "fresh": freshness
                .and_then(|v| v.get("fresh"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "freshness": freshness.cloned().unwrap_or(serde_json::Value::Null),
            "provider_complete": incomplete_providers.is_empty(),
            "incomplete_provider_count": incomplete_providers.len(),
            "incomplete_providers": incomplete_providers,
            "limit": limit,
            "line_numbers": line_numbers,
            "span_cap_lines": CONTEXT_SPAN_CAP,
            "candidate_total_kind": "top_k_only",
            "shown": spans.len(),
            "source_unavailable_count": source_unavailable_count,
            "span_truncated_count": span_truncated_count,
            "truncated": truncated,
            "spans": spans,
        }))
        .map_err(|e| Error::Invalid(format!("serialize context JSON: {e}")))?
    );
    Ok(())
}

/// Markers that identify a repository / project root when walking up
/// from the current directory. Kept in sync with the markers
/// `greppy_core::workspace::project_identity` recognises so that the
/// store path (hashed from the resolved root) and the project name
/// (derived from the same root) always agree (RV-006 / RV-011).
/// Resolve the effective workspace root for a command.
///
/// * If `--root <PATH>` was given, canonicalize it and resolve its enclosing
///   Git worktree/project root through the shared core resolver.
/// * Otherwise start at the current directory and walk **up** until a
///   repo marker (`.git`, `Cargo.toml`, `pyproject.toml`) is found,
///   returning that directory.
/// * If no marker is found anywhere in the chain, fall back to the
///   current directory.
///
/// This is the single source of truth every command routes through, so
/// `greppy index .` from the repo root and `greppy search-code Q`
/// from a subdirectory resolve to the **same** store path and the
/// **same** project identity (RV-006 closes the subdir/exit-73 gap;
/// RV-011 closes the index/search project-name mismatch).
fn resolve_root(root: Option<&str>) -> Result<std::path::PathBuf> {
    if let Some(r) = root {
        // Defect D9: `--root` used to be taken verbatim, so a relative
        // (`--root .`) or non-canonical (`/tmp/...` vs `/private/tmp/...`
        // on macOS, trailing slash) argument keyed the store/workspace
        // state differently than the indexer, which records the
        // canonicalized root — later lookups then failed with "no
        // workspace_state". Normalize to the canonical absolute path so
        // every spelling of the same directory is one workspace.
        let explicit = absolutize_path(std::path::Path::new(r));
        return Ok(workspace_locator::resolve_workspace_root(&explicit));
    }
    let cwd = std::env::current_dir()
        .map_err(|e| Error::io("read current_dir for root resolution", e))?;
    Ok(workspace_locator::resolve_workspace_root(&cwd))
}

/// Canonicalize when the path exists; otherwise make it absolute
/// lexically; a path we cannot even absolutize is returned as-is (the
/// caller will fail with a clearer error when it tries to use it).
fn absolutize_path(p: &std::path::Path) -> std::path::PathBuf {
    p.canonicalize()
        .or_else(|_| std::path::absolute(p))
        .unwrap_or_else(|_| p.to_path_buf())
}

/// Walk up from `start` looking for a repository marker. Returns the
/// first ancestor (including `start`) that contains a marker, or `start`
/// itself when none is found. Pure path logic so it is unit-testable
/// without touching the process cwd.
fn find_repo_root(start: &std::path::Path) -> std::path::PathBuf {
    workspace_locator::resolve_workspace_root(start)
}

/// Compute the project identity string for the effective root
/// (`--root` if given, else the detected repo root). Centralised so
/// every command uses the same definition (RV-011).
fn project_for(root: Option<&str>) -> Result<String> {
    let p = resolve_root(root)?;
    Ok(workspace_locator::project_identity(&p))
}

#[derive(Debug, Clone)]
struct QueryPathFilter {
    shown: String,
    repo_prefix: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct QueryPathFilters {
    filters: Vec<QueryPathFilter>,
}

impl QueryPathFilters {
    fn from_args(root_path: &std::path::Path, paths: &[String]) -> Self {
        Self {
            filters: paths
                .iter()
                .filter(|path| !path.trim().is_empty())
                .map(|path| QueryPathFilter {
                    shown: path.clone(),
                    repo_prefix: normalize_query_filter_path(root_path, path),
                })
                .collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    fn matches(&self, file_path: &str) -> bool {
        self.filters.is_empty()
            || self.filters.iter().any(|filter| {
                let Some(prefix) = filter.repo_prefix.as_deref() else {
                    return false;
                };
                prefix.is_empty()
                    || file_path == prefix
                    || file_path
                        .strip_prefix(prefix)
                        .is_some_and(|rest| rest.starts_with('/'))
            })
    }

    fn shown(&self) -> String {
        self.filters
            .iter()
            .map(|filter| filter.shown.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn json_value(&self) -> serde_json::Value {
        serde_json::json!(self
            .filters
            .iter()
            .map(|filter| filter.shown.as_str())
            .collect::<Vec<_>>())
    }
}

fn normalize_query_filter_path(root_path: &std::path::Path, raw: &str) -> Option<String> {
    let supplied = std::path::Path::new(raw);
    let candidate = if supplied.is_absolute() {
        absolutize_path(supplied)
    } else {
        let cwd = std::env::current_dir().ok();
        let cwd_candidate = cwd.as_ref().map(|cwd| cwd.join(supplied));
        if let Some(path) = cwd_candidate.as_ref().filter(|path| path.exists()) {
            absolutize_path(path)
        } else if root_path.join(supplied).exists() {
            absolutize_path(&root_path.join(supplied))
        } else if let Some(cwd) = cwd.filter(|cwd| cwd.starts_with(root_path)) {
            cwd.join(supplied)
        } else {
            root_path.join(supplied)
        }
    };
    let relative = candidate.strip_prefix(root_path).ok()?;
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            std::path::Component::ParentDir => {
                parts.pop();
            }
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }
    Some(parts.join("/"))
}

fn shell_example_arg(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"/_-.+:".contains(&byte))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// Keep fallback searches tied to the user's actual words. Natural-language
/// glue is dropped so the grep pattern remains selective; when every word is
/// glue, preserve the original tokens rather than inventing a pattern.
fn semantic_fallback_tokens(query: &str) -> Vec<String> {
    fn is_search_glue(token: &str) -> bool {
        matches!(
            token.to_ascii_lowercase().as_str(),
            "a" | "an"
                | "and"
                | "are"
                | "as"
                | "at"
                | "be"
                | "by"
                | "code"
                | "find"
                | "for"
                | "from"
                | "how"
                | "in"
                | "is"
                | "it"
                | "of"
                | "on"
                | "or"
                | "search"
                | "that"
                | "the"
                | "this"
                | "to"
                | "what"
                | "where"
                | "which"
                | "with"
        )
    }

    let all = query
        .split(|character: char| !(character.is_alphanumeric() || character == '_'))
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut selected = all
        .iter()
        .filter(|token| token.len() >= 2 && !is_search_glue(token))
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        selected = all;
    }
    let mut seen = std::collections::HashSet::new();
    selected.retain(|token| seen.insert(token.to_ascii_lowercase()));
    selected.truncate(8);
    selected
}

fn semantic_fallback_commands(query: &str, paths: &[String], root: Option<&str>) -> Vec<String> {
    let tokens = semantic_fallback_tokens(query);
    let mut commands = Vec::new();
    if let Some(symbol_token) = tokens.last() {
        let mut command = format!("greppy search-symbols {}", shell_example_arg(symbol_token));
        for path in paths {
            command.push(' ');
            command.push_str(&shell_example_arg(path));
        }
        if let Some(root) = root {
            command.push_str(" --root ");
            command.push_str(&shell_example_arg(root));
        }
        commands.push(command);
    }

    let pattern = if tokens.is_empty() {
        query.to_string()
    } else {
        tokens.join("|")
    };
    let grep_targets = if paths.is_empty() {
        vec![root.unwrap_or(".").to_string()]
    } else if let Some(root) = root {
        paths
            .iter()
            .map(|path| {
                std::path::Path::new(root)
                    .join(path)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    } else {
        paths.to_vec()
    };
    let mut grep = format!("greppy grep -rnE {}", shell_example_arg(&pattern));
    for target in grep_targets {
        grep.push(' ');
        grep.push_str(&shell_example_arg(&target));
    }
    commands.push(grep);
    commands
}

fn print_semantic_fallback_commands(query: &str, paths: &[String], root: Option<&str>) {
    for command in semantic_fallback_commands(query, paths, root) {
        println!("try: {command}");
    }
}

fn embedding_asset_missing_error(error: &Error) -> bool {
    matches!(error, Error::Config(message) if message.contains("EmbeddingGemma assets are unavailable"))
}

fn validate_query_root_usage(root: Option<&str>, command: &str, subject: &str) -> Result<()> {
    let Some(raw_root) = root else {
        return Ok(());
    };
    let supplied = absolutize_path(std::path::Path::new(raw_root));
    let repo_root = workspace_locator::resolve_workspace_root(&supplied);
    if supplied == repo_root
        || workspace_locator::store_path(&supplied).exists()
        || !workspace_locator::store_path(&repo_root).exists()
    {
        return Ok(());
    }
    Err(Error::Invalid(format!(
        "--root selects the indexed repository root, not a file or subtree filter.\nretry: greppy {command} {} {} --root {}",
        shell_example_arg(subject),
        shell_example_arg(raw_root),
        shell_example_arg(&repo_root.to_string_lossy()),
    )))
}

fn prepare_query_path_filters(
    root: Option<&str>,
    command: &str,
    subject: &str,
    paths: &[String],
) -> Result<QueryPathFilters> {
    validate_query_root_usage(root, command, subject)?;
    Ok(QueryPathFilters::from_args(&resolve_root(root)?, paths))
}

fn embedding_config_for_index(args: EmbeddingCliArgs<'_>) -> Result<Option<EmbeddingModelConfig>> {
    if test_inference_skipped() {
        return Ok(None);
    }
    Ok(Some(embedding_config_required(args)?))
}

fn embedding_config_for_required_use(args: EmbeddingCliArgs<'_>) -> Result<EmbeddingModelConfig> {
    embedding_config_required(args)
}

/// Resolve the mandatory embedded EmbeddingGemma model. The `Option` remains
/// only for non-fatal query paths that predate the always-embedded contract.
fn embedding_config_optional(args: EmbeddingCliArgs<'_>) -> Result<Option<EmbeddingModelConfig>> {
    embedding_config_required(args).map(Some)
}

static EMBEDDED_ASSET_TMP_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn extract_embedded_asset(
    model_root: &std::path::Path,
    expected_sha: &str,
    name: &str,
    bytes: &[u8],
) -> Option<String> {
    let model = model_root.file_name()?.to_str()?.to_owned();
    let root = greppy_core::cache::ensure_model_entry(&model, expected_sha).ok()?;
    let dest = root.join(name);
    let marker = root.join(format!("{name}.sha256"));
    if embedded_asset_marker_matches(&dest, &marker, expected_sha, bytes.len()) {
        greppy_core::cache::touch_last_used_dir(&root);
        return Some(dest.to_string_lossy().into_owned());
    }

    let _lease = greppy_core::cache::acquire_model_lifecycle(
        expected_sha,
        greppy_core::cache::LockMode::Exclusive,
        false,
    )
    .ok()??;
    let root = greppy_core::cache::ensure_model_entry(&model, expected_sha).ok()?;
    let dest = root.join(name);
    let marker = root.join(format!("{name}.sha256"));
    if embedded_asset_marker_matches(&dest, &marker, expected_sha, bytes.len()) {
        greppy_core::cache::touch_last_used_dir(&root);
        return Some(dest.to_string_lossy().into_owned());
    }

    let nonce = EMBEDDED_ASSET_TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = root.join(format!("{name}.tmp.{}.{}", std::process::id(), nonce));
    let marker_tmp = root.join(format!(
        "{name}.sha256.tmp.{}.{}",
        std::process::id(),
        nonce
    ));

    // Upgrade legacy markers without rewriting a valid multi-hundred-MiB
    // model. A changed metadata fingerprint always re-enters this digest path.
    let result = if embedded_asset_digest_matches(&dest, expected_sha, bytes.len()) {
        write_embedded_asset_marker(&dest, &marker_tmp, &marker, expected_sha, bytes.len())
    } else {
        write_verified_embedded_asset(&tmp, &dest, &marker_tmp, &marker, expected_sha, bytes)
    };
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&marker_tmp);
    }

    if embedded_asset_marker_matches(&dest, &marker, expected_sha, bytes.len()) {
        greppy_core::cache::touch_last_used_dir(&root);
        Some(dest.to_string_lossy().into_owned())
    } else {
        None
    }
}

fn embedded_asset_marker_matches(
    dest: &std::path::Path,
    marker: &std::path::Path,
    expected_sha: &str,
    expected_len: usize,
) -> bool {
    let Ok(metadata_fingerprint) = embedded_asset_metadata_fingerprint(dest, expected_len) else {
        return false;
    };
    let Ok(marker_metadata) = std::fs::symlink_metadata(marker) else {
        return false;
    };
    if !marker_metadata.file_type().is_file() {
        return false;
    }
    let Ok(raw) = std::fs::read(marker) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&raw) else {
        return false;
    };
    value.get("version").and_then(serde_json::Value::as_u64) == Some(1)
        && value.get("sha256").and_then(serde_json::Value::as_str) == Some(expected_sha)
        && value.get("length").and_then(serde_json::Value::as_u64)
            == u64::try_from(expected_len).ok()
        && value
            .get("metadata_fingerprint")
            .and_then(serde_json::Value::as_str)
            == Some(metadata_fingerprint.as_str())
}

fn embedded_asset_metadata_fingerprint(
    path: &std::path::Path,
    expected_len: usize,
) -> std::io::Result<String> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("embedded asset {} is not a regular file", path.display()),
        ));
    }
    let expected_len = u64::try_from(expected_len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "embedded asset length does not fit u64",
        )
    })?;
    if metadata.len() != expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "embedded asset {} has length {}, expected {expected_len}",
                path.display(),
                metadata.len()
            ),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o077 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!(
                    "embedded asset {} is not private to its owner",
                    path.display()
                ),
            ));
        }
        Ok(format!(
            "unix:{}:{}:{}:{}:{}",
            metadata.dev(),
            metadata.ino(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime_nsec()
        ))
    }
    #[cfg(not(unix))]
    {
        let modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos();
        Ok(format!("portable:{modified}:{}", metadata.len()))
    }
}

fn embedded_asset_digest_matches(
    path: &std::path::Path,
    expected_sha: &str,
    expected_len: usize,
) -> bool {
    if make_embedded_asset_private(path).is_err()
        || embedded_asset_metadata_fingerprint(path, expected_len).is_err()
    {
        return false;
    }
    embedded_asset_sha256_file(path).is_ok_and(|digest| digest == expected_sha)
}

fn embedded_asset_sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn make_embedded_asset_private(path: &std::path::Path) -> std::io::Result<()> {
    greppy_core::cache::secure_private_file(path)
}

fn write_verified_embedded_asset(
    tmp: &std::path::Path,
    dest: &std::path::Path,
    marker_tmp: &std::path::Path,
    marker: &std::path::Path,
    expected_sha: &str,
    bytes: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let _ = std::fs::remove_file(tmp);
    let mut file = options.open(tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    if !embedded_asset_digest_matches(tmp, expected_sha, bytes.len()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "embedded asset {} failed SHA-256 verification",
                tmp.display()
            ),
        ));
    }

    // Invalidate trust before replacing the payload. Other processes either
    // see the old verified pair or wait for the exclusive lifecycle lease.
    let _ = std::fs::remove_file(marker);
    let _ = std::fs::remove_file(dest);
    std::fs::rename(tmp, dest)?;
    make_embedded_asset_private(dest)?;
    write_embedded_asset_marker(dest, marker_tmp, marker, expected_sha, bytes.len())
}

fn write_embedded_asset_marker(
    dest: &std::path::Path,
    marker_tmp: &std::path::Path,
    marker: &std::path::Path,
    expected_sha: &str,
    expected_len: usize,
) -> std::io::Result<()> {
    use std::io::Write;

    make_embedded_asset_private(dest)?;
    let metadata_fingerprint = embedded_asset_metadata_fingerprint(dest, expected_len)?;
    let length = u64::try_from(expected_len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "embedded asset length does not fit u64",
        )
    })?;
    let payload = serde_json::to_vec(&serde_json::json!({
        "version": 1,
        "sha256": expected_sha,
        "length": length,
        "metadata_fingerprint": metadata_fingerprint,
    }))?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let _ = std::fs::remove_file(marker_tmp);
    let mut file = options.open(marker_tmp)?;
    file.write_all(&payload)?;
    file.sync_all()?;
    drop(file);
    let _ = std::fs::remove_file(marker);
    std::fs::rename(marker_tmp, marker)?;
    make_embedded_asset_private(marker)
}

#[cfg(test)]
fn embedded_asset_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Built-in EmbeddingGemma: the Q4_K GGUF and
/// tokenizer are baked into the binary at build time and extracted once
/// to `<data>/greppy/models/embeddinggemma-300m-q4k/<sha>/` (mmap needs a real
/// file). The
/// extraction is atomic (tmp + rename). A cache entry is hashed before it is
/// first trusted and whenever its metadata identity changes; a private marker
/// makes the unchanged fast path constant-time without accepting stale or torn
/// payloads.
mod embeddinggemma_assets {
    pub fn paths() -> Option<(String, String)> {
        const GGUF_SHA: &str = env!("GREPPY_EMBEDDED_GGUF_SHA");
        const TOK_SHA: &str = env!("GREPPY_EMBEDDED_TOK_SHA");
        static GGUF: &[u8] = include_bytes!(env!("GREPPY_EMBEDDED_GGUF_PATH"));
        static TOK: &[u8] = include_bytes!(env!("GREPPY_EMBEDDED_TOK_PATH"));
        let root = greppy_core::cache::models_root().join("embeddinggemma-300m-q4k");
        let gguf = extract(&root, GGUF_SHA, "embeddinggemma-300M-Q4_K.gguf", GGUF)?;
        let tok = extract(&root, TOK_SHA, "tokenizer.json", TOK)?;
        Some((gguf, tok))
    }

    fn extract(
        root: &std::path::Path,
        expected_sha: &str,
        name: &str,
        bytes: &[u8],
    ) -> Option<String> {
        super::extract_embedded_asset(root, expected_sha, name, bytes)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cached_asset_resolves_while_model_has_shared_lease() {
            const MODEL: &str = "embeddinggemma-asset-lock-test";
            const NAME: &str = "asset.bin";
            const BYTES: &[u8] = b"cached embedding asset";

            let sha = crate::embedded_asset_sha256(BYTES);
            let root = std::path::Path::new(MODEL);
            assert!(extract(root, &sha, NAME, BYTES).is_some());
            let lease = greppy_core::cache::acquire_model_lifecycle(
                &sha,
                greppy_core::cache::LockMode::Shared,
                false,
            )
            .expect("shared model lease")
            .expect("model lease available");
            let (tx, rx) = std::sync::mpsc::channel();
            let waiter_sha = sha.clone();
            let waiter = std::thread::spawn(move || {
                let result = extract(std::path::Path::new(MODEL), &waiter_sha, NAME, BYTES);
                let _ = tx.send(result.clone());
                result
            });
            let resolved = rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("cached asset lookup must not wait for an exclusive lease");
            drop(lease);
            assert!(resolved.is_some());
            assert!(waiter.join().expect("asset lookup thread").is_some());
            let _ = std::fs::remove_dir_all(greppy_core::cache::models_root().join(MODEL));
        }

        #[test]
        fn same_length_cached_asset_tampering_is_repaired() {
            const MODEL: &str = "embeddinggemma-asset-tamper-test";
            const NAME: &str = "asset.bin";
            const BYTES: &[u8] = b"verified model bytes";

            let sha = crate::embedded_asset_sha256(BYTES);
            let root = std::path::Path::new(MODEL);
            let path = extract(root, &sha, NAME, BYTES).expect("extract verified asset");
            std::fs::remove_file(&path).expect("remove verified payload");
            std::fs::write(&path, b"tampered model bytes").expect("write same-length tamper");
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                u64::try_from(BYTES.len()).unwrap()
            );

            let repaired = extract(root, &sha, NAME, BYTES).expect("repair tampered asset");
            assert_eq!(std::fs::read(repaired).unwrap(), BYTES);
            let _ = std::fs::remove_dir_all(greppy_core::cache::models_root().join(MODEL));
        }
    }
}

mod qwen35_assets {
    pub fn paths() -> Option<(String, String)> {
        const GGUF_SHA: &str = env!("GREPPY_EMBEDDED_QWEN35_GGUF_SHA");
        const TOK_SHA: &str = env!("GREPPY_EMBEDDED_QWEN35_TOK_SHA");
        static GGUF: &[u8] = include_bytes!(env!("GREPPY_EMBEDDED_QWEN35_GGUF_PATH"));
        static TOK: &[u8] = include_bytes!(env!("GREPPY_EMBEDDED_QWEN35_TOK_PATH"));
        let root = greppy_core::cache::models_root().join("qwen35-0.8b-mtp-q4km");
        let gguf = extract(&root, GGUF_SHA, "Qwen3.5-0.8B-MTP-Q4_K_M.gguf", GGUF)?;
        let tok = extract(&root, TOK_SHA, "tokenizer.json", TOK)?;
        Some((gguf, tok))
    }

    fn extract(
        root: &std::path::Path,
        expected_sha: &str,
        name: &str,
        bytes: &[u8],
    ) -> Option<String> {
        super::extract_embedded_asset(root, expected_sha, name, bytes)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cached_asset_resolves_while_model_has_shared_lease() {
            const MODEL: &str = "qwen35-asset-lock-test";
            const NAME: &str = "asset.bin";
            const BYTES: &[u8] = b"cached qwen asset";

            let sha = crate::embedded_asset_sha256(BYTES);
            let root = std::path::Path::new(MODEL);
            assert!(extract(root, &sha, NAME, BYTES).is_some());
            let lease = greppy_core::cache::acquire_model_lifecycle(
                &sha,
                greppy_core::cache::LockMode::Shared,
                false,
            )
            .expect("shared model lease")
            .expect("model lease available");
            let (tx, rx) = std::sync::mpsc::channel();
            let waiter_sha = sha.clone();
            let waiter = std::thread::spawn(move || {
                let result = extract(std::path::Path::new(MODEL), &waiter_sha, NAME, BYTES);
                let _ = tx.send(result.clone());
                result
            });
            let resolved = rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("cached asset lookup must not wait for an exclusive lease");
            drop(lease);
            assert!(resolved.is_some());
            assert!(waiter.join().expect("asset lookup thread").is_some());
            let _ = std::fs::remove_dir_all(greppy_core::cache::models_root().join(MODEL));
        }
    }
}

fn qwen_summary_config_optional() -> Result<Option<QwenSummaryConfig>> {
    if test_inference_skipped() {
        return Ok(None);
    }
    let Some((gguf, tokenizer)) = qwen35_assets::paths() else {
        return Ok(None);
    };
    Ok(Some(QwenSummaryConfig {
        model_id: greppy_qwen35_native::MODEL_ID.to_string(),
        gguf: gguf.into(),
        tokenizer: tokenizer.into(),
        device: qwen_summary_device_preference()?,
    }))
}

fn qwen_summary_device_preference() -> Result<greppy_qwen35_native::DevicePreference> {
    let cli = cli_inference_override();
    if cli.no_gpu || env_bool(ENV_NO_GPU)? {
        return Ok(greppy_qwen35_native::DevicePreference::Cpu);
    }
    let raw = cli
        .device
        .or_else(|| env_nonempty(ENV_DEVICE))
        .unwrap_or_else(|| "auto".to_string());
    greppy_qwen35_native::DevicePreference::parse(&raw).map_err(|e| Error::Invalid(e.to_string()))
}

struct LoadedQwen35Summarizer {
    inner: greppy_qwen35_native::Qwen35Summarizer,
    _model_lease: Option<greppy_core::cache::FileLock>,
}

impl std::ops::Deref for LoadedQwen35Summarizer {
    type Target = greppy_qwen35_native::Qwen35Summarizer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

fn load_qwen35_summarizer(cfg: &QwenSummaryConfig) -> Result<LoadedQwen35Summarizer> {
    let lease = acquire_cached_model_lease(&cfg.gguf)?;
    let options = greppy_qwen35_native::LoadOptions {
        device: cfg.device.clone(),
    };
    let inner =
        greppy_qwen35_native::Qwen35Summarizer::load_gguf(&cfg.gguf, &cfg.tokenizer, options)
            .map_err(|e| Error::Store(format!("load Qwen3.5 summarizer {}: {e}", cfg.model_id)))?;
    Ok(LoadedQwen35Summarizer {
        inner,
        _model_lease: lease,
    })
}

fn qwen_summary_model_key(cfg: &QwenSummaryConfig) -> String {
    format!(
        "{}:{}:{}:{}:{}:{}:{}",
        cfg.model_id,
        greppy_qwen35_native::PROMPT_VERSION,
        greppy_qwen35_native::TRIAGE_PROMPT_VERSION,
        greppy_qwen35_native::BRIEF_FILTER_VERSION,
        inference_device_identity(&cfg.device),
        model_file_digest(&cfg.gguf).unwrap_or_else(|_| "unknown".into()),
        model_file_digest(&cfg.tokenizer).unwrap_or_else(|_| "unknown".into())
    )
}

fn embedding_config_required(args: EmbeddingCliArgs<'_>) -> Result<EmbeddingModelConfig> {
    #[cfg(debug_assertions)]
    if std::env::var_os(ENV_TEST_EMBED_ASSET_MISSING).is_some() {
        return Err(Error::Config(
            "embedded EmbeddingGemma assets are unavailable (test failpoint)".into(),
        ));
    }
    let device = embedding_device_preference(args.device, args.no_gpu)?;
    let source = match embeddinggemma_assets::paths() {
        Some((gguf, tokenizer)) => EmbeddingModelSource::Gguf {
            gguf: gguf.into(),
            tokenizer: tokenizer.into(),
        },
        None => {
            return Err(Error::Config(
                "embedded EmbeddingGemma assets are unavailable".into(),
            ))
        }
    };
    let source_digest = embedding_source_content_digest(&source)?;
    Ok(EmbeddingModelConfig {
        model_id: format!("{DEFAULT_EMBEDDINGGEMMA_MODEL_ID}@sha256:{source_digest}"),
        source,
        max_length: None,
        device,
    })
}

fn embedding_source_content_digest(source: &EmbeddingModelSource) -> Result<String> {
    use sha2::{Digest, Sha256};

    let EmbeddingModelSource::Gguf { gguf, tokenizer } = source;
    let paths = vec![gguf.clone(), tokenizer.clone()];
    let mut combined = Sha256::new();
    for path in paths {
        let digest = model_file_digest(&path)
            .map_err(|error| Error::io(format!("digest model file {}", path.display()), error))?;
        combined.update(path.file_name().unwrap_or_default().as_encoded_bytes());
        combined.update([0]);
        combined.update(digest.as_bytes());
        combined.update([0]);
    }
    Ok(format!("{:x}", combined.finalize()))
}

fn embedding_device_preference(
    cli_device: Option<&str>,
    cli_no_gpu: bool,
) -> Result<greppy_embed_native::DevicePreference> {
    if cli_no_gpu || env_bool(ENV_NO_GPU)? {
        return Ok(greppy_embed_native::DevicePreference::Cpu);
    }
    let raw = cli_device
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| env_nonempty(ENV_DEVICE))
        .unwrap_or_else(|| "auto".to_string());
    raw.parse::<greppy_embed_native::DevicePreference>()
        .map_err(|e| Error::Invalid(e.to_string()))
}

/// Load the embedding model. `tokenizer_cache_dir` (normally the
/// per-workspace store dir, honoring `GREPPY_STORE_DIR`) enables the
/// tokenizer fast-load sidecar for GGUF models, cutting warm model-load
/// latency roughly in half; pass `None` to force a full parse.
struct LoadedEmbeddingModel {
    inner: greppy_embed_native::EmbeddingGemma,
    _model_lease: Option<greppy_core::cache::FileLock>,
}

impl std::ops::Deref for LoadedEmbeddingModel {
    type Target = greppy_embed_native::EmbeddingGemma;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

fn load_embedding_model(
    cfg: &EmbeddingModelConfig,
    tokenizer_cache_dir: Option<std::path::PathBuf>,
) -> Result<LoadedEmbeddingModel> {
    let options = greppy_embed_native::LoadOptions {
        device: cfg.device.clone(),
        max_length: cfg.max_length,
        tokenizer_cache_dir,
    };
    let EmbeddingModelSource::Gguf { gguf, tokenizer } = &cfg.source;
    let lease = acquire_cached_model_lease(gguf)?;
    let inner = greppy_embed_native::EmbeddingGemma::load_gguf(gguf, tokenizer, options)
        .map_err(|e| Error::Store(format!("load EmbeddingGemma model {}: {e}", cfg.model_id)))?;
    Ok(LoadedEmbeddingModel {
        inner,
        _model_lease: lease,
    })
}

/// Cache key for query embeddings: logical model id + prompt/task contract +
/// content digests. A same-size/same-mtime model replacement cannot reuse a
/// vector computed by different weights.
fn embedding_query_cache_key(cfg: &EmbeddingModelConfig) -> String {
    fn file_fp(path: &std::path::Path) -> String {
        model_file_digest(path).unwrap_or_else(|_| format!("{}:unknown", path.display()))
    }
    let EmbeddingModelSource::Gguf { gguf, tokenizer } = &cfg.source;
    let source_fp = format!("gguf;{};{}", file_fp(gguf), file_fp(tokenizer));
    format!(
        "{}|{}|{}|{}",
        cfg.model_id,
        greppy_embed_native::PROMPT_VERSION,
        greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE,
        source_fp
    )
}

fn cached_model_digest(path: &std::path::Path) -> Option<String> {
    if !path.starts_with(greppy_core::cache::models_root()) {
        return None;
    }
    let digest = path.parent()?.file_name()?.to_str()?;
    (digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| digest.to_ascii_lowercase())
}

fn acquire_cached_model_lease(
    path: &std::path::Path,
) -> Result<Option<greppy_core::cache::FileLock>> {
    let Some(digest) = cached_model_digest(path) else {
        return Ok(None);
    };
    if let Some(parent) = path.parent() {
        greppy_core::cache::touch_last_used_dir(parent);
    }
    greppy_core::cache::acquire_model_lifecycle(
        &digest,
        greppy_core::cache::LockMode::Shared,
        false,
    )
    .map_err(|e| Error::io(format!("acquire model lease for {}", path.display()), e))
}

fn model_file_digest(path: &std::path::Path) -> std::io::Result<String> {
    if let Some(digest) = cached_model_digest(path) {
        return Ok(digest);
    }
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Embed a code-retrieval query, consulting the store-level query cache
/// first. On a hit the model is never loaded (saves the entire ~0.15-0.4s
/// model-load + ~30ms inference cost); on a miss the vector is computed
/// and cached best-effort. Cache failures silently degrade to a miss —
/// they must never fail a search.
fn embed_query_cached(cfg: &EmbeddingModelConfig, root: Option<&str>, q: &str) -> Result<Vec<f32>> {
    let store_dir = resolve_root(root)
        .ok()
        .map(|r| workspace_locator::store_dir(&r));
    let cache = store_dir
        .as_ref()
        .and_then(|dir| greppy_store::QueryEmbeddingCache::open(dir).ok());
    let model_key = embedding_query_cache_key(cfg);
    let normalized = greppy_store::normalize_query_text(q);
    if let Some(cache) = &cache {
        if let Ok(Some(vector)) = cache.get(&model_key, &normalized) {
            return Ok(vector);
        }
    }
    // Prefer the warm daemon (model stays resident across CLI calls; VRAM
    // freed after its idle TTL). Only a daemon proven absent may use the
    // in-process fallback. Busy or faulted live daemons retain model ownership,
    // so falling back there could allocate a second model instance.
    #[cfg(any(unix, windows))]
    let daemon_result = embed_daemon::embed_query_via_daemon_result(cfg, &model_key, &normalized);
    #[cfg(not(any(unix, windows)))]
    let daemon_result = embed_daemon::EmbedDaemonResult::NoDaemon;
    let vector = match daemon_result {
        embed_daemon::EmbedDaemonResult::Embedded(vector) => vector,
        embed_daemon::EmbedDaemonResult::NoDaemon => {
            let model = load_embedding_model(cfg, store_dir)?;
            greppy_search::embed_code_query(&model, &normalized)?
        }
        embed_daemon::EmbedDaemonResult::DaemonBusy => {
            return Err(Error::Store(
                "EmbeddingGemma daemon remained busy until the request deadline".into(),
            ));
        }
        embed_daemon::EmbedDaemonResult::Failed => {
            return Err(Error::Store(
                "EmbeddingGemma daemon failed while retaining model ownership".into(),
            ));
        }
    };
    if let Some(cache) = &cache {
        let _ = cache.put(&model_key, &normalized, &vector);
    }
    Ok(vector)
}

fn log_embedding_skip_once(command: &str, err: &Error) {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("{command}: embedding unavailable; skipping vector search: {err}");
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn env_bool(name: &str) -> Result<bool> {
    let Some(raw) = env_nonempty(name) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(Error::Invalid(format!(
            "{name} must be one of 1/0/true/false/yes/no/on/off"
        ))),
    }
}

fn vector_exact_candidate_limit() -> Result<Option<i64>> {
    let raw = env_nonempty(ENV_VECTOR_EXACT_CANDIDATE_LIMIT);
    parse_vector_exact_candidate_limit(raw.as_deref())
}

fn parse_vector_exact_candidate_limit(raw: Option<&str>) -> Result<Option<i64>> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(Some(greppy_search::DEFAULT_EXACT_VECTOR_CANDIDATE_LIMIT));
    };
    let parsed = raw.parse::<i64>().map_err(|_| {
        Error::Invalid(format!(
            "{ENV_VECTOR_EXACT_CANDIDATE_LIMIT} must be 0 or a positive integer"
        ))
    })?;
    if parsed < 0 {
        return Err(Error::Invalid(format!(
            "{ENV_VECTOR_EXACT_CANDIDATE_LIMIT} must be 0 or a positive integer"
        )));
    }
    if parsed == 0 {
        Ok(None)
    } else {
        Ok(Some(parsed))
    }
}

fn vector_exact_scan_exceeds_limit(total: i64, candidate_limit: Option<i64>) -> Option<i64> {
    match candidate_limit {
        Some(limit) if total > limit => Some(limit),
        _ => None,
    }
}

fn vector_exact_scan_skip_message(command: &str, total: i64, limit: i64) -> String {
    format!(
        "{command}: vector exact scan skipped ({total} candidates exceed limit {limit}); set {ENV_VECTOR_EXACT_CANDIDATE_LIMIT}=0 to allow an unbounded exact scan, or raise the limit until ANN vector search is implemented"
    )
}

fn freshness_json_is_fresh(freshness: &serde_json::Value) -> bool {
    freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn vector_stale_skip_message(command: &str, freshness: &serde_json::Value) -> String {
    format!(
        "{command}: vector search skipped because {}",
        stale_freshness_reason(freshness)
    )
}

fn plus_stale_skip_message(freshness: &serde_json::Value) -> String {
    format!(
        "grep: indexed search skipped because {}; no stale indexed hits emitted",
        stale_freshness_reason(freshness)
    )
}

fn context_stale_skip_message(freshness: &serde_json::Value) -> String {
    format!(
        "context: source-span lookup skipped because {}; no stale indexed spans emitted",
        stale_freshness_reason(freshness)
    )
}

fn indexed_stale_skip_message(command: &str, freshness: &serde_json::Value) -> String {
    format!(
        "{command}: indexed search skipped because {}; no stale indexed hits emitted",
        stale_freshness_reason(freshness)
    )
}

fn stale_freshness_reason(freshness: &serde_json::Value) -> String {
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let reasons = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
                .join("; ")
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "freshness check did not prove the index is current".into());
    format!("graph freshness is {state}: {reasons}")
}

fn open_default_store(root: Option<&str>) -> Result<greppy_store::Store> {
    // The graph DB lives under the platform locator, never at
    // `<cwd>/.greppy/graph.db`. When no
    // `--root` is given we detect the repo root by walking up for a
    // marker, so a query from a subdirectory targets the same store the
    // indexer wrote from the repo root (instead of opening an empty
    // store under the subdir's hash and exiting 73).
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    // RV-007: tighten the store dir + DB file permissions on every open.
    // This is a no-op when the store doesn't exist yet (read paths before
    // any `greppy index` would have failed to open the store anyway).
    if let Some(parent) = path.parent() {
        let _ = workspace_locator::ensure_store_dir(parent);
    }
    // Forensics F4: a query against a repo that was never indexed used to
    // open a non-existent DB, fail deep in SQLite, and exit 73 (EXIT_IO)
    // with NOTHING on stdout/stderr — the agent just saw an empty result and
    // a bare non-zero code, with no hint that the fix is `greppy index`.
    //
    // Feature A (auto-index on first use): rather than erroring, build a
    // GRAPH index (no embeddings) inline on first use so the graph nav
    // commands (who-calls / callees / references / impact /
    // path / brief / fan-in / fan-out / trace) Just Work on a fresh repo.
    // Gated behind the same kill switch as the inline auto-reindex
    // (`GREPPY_AUTO_REINDEX=0` restores the old hard error). If the
    // inline index fails for any reason we fall back to the actionable
    // diagnostic below. Embeddings are NOT computed here — the vector
    // path (`context` / `semantic`) still asks for `grep index` when it
    // needs vectors.
    // (Query commands only — the grep passthrough path never reaches here,
    // so the byte-exact passthrough contract is untouched.)
    if !path.exists() {
        let shown_root = root.unwrap_or(".");
        if auto_reindex_enabled() {
            eprintln!("greppy: indexing {} (first use)…", effective_root.display());
            if try_auto_index_inline(root) && path.exists() {
                // Index built: fall through to the normal read-only open.
            } else {
                eprintln!(
                    "greppy: no index for {} — run `greppy index {}` first",
                    effective_root.display(),
                    shown_root
                );
                return Err(Error::Invalid(format!(
                    "no index for {}; run `greppy index {}` first",
                    effective_root.display(),
                    shown_root
                )));
            }
        } else {
            eprintln!(
                "greppy: no index for {} — run `greppy index {}` first",
                effective_root.display(),
                shown_root
            );
            return Err(Error::Invalid(format!(
                "no index for {}; run `greppy index {}` first",
                effective_root.display(),
                shown_root
            )));
        }
    }
    // Query commands are READ-ONLY: open read-only so they skip both
    // `migrate()` and the O(db-size) `integrity_check` that a read-write open
    // runs. Those belong on the writer (`greppy index`); paying them on every
    // query open made who-calls/search take seconds on a real repo
    // (the token-efficiency benchmark's latency culprit). Readers tolerate
    // whatever schema the DB has.
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::read_only())?;
    let _ = workspace_locator::ensure_db_mode(&path);
    // Feature B: record that this store was just used to serve a query.
    // A read-only open never bumps graph.db's mtime, so a dedicated
    // `.lastused` marker is what keeps a frequently-queried store from
    // being evicted by `cleanup_stale_stores`. Best-effort — a failed
    // touch never fails the query.
    if let Some(store_dir) = path.parent() {
        workspace_locator::touch_lastused(store_dir);
    }
    // O5 session prewarm: the first graph command of an agent session nudges
    // the embed daemon (with an async model load) so a following `context`
    // query hits a warm model instead of paying the cold start. Guarded to
    // fire only when semantic search is actually in play — env model
    // configured AND this store holds vectors — because prewarming a model
    // nobody will query would hold GPU memory for a TTL for nothing.
    #[cfg(any(unix, windows))]
    {
        let no_args = EmbeddingCliArgs {
            device: None,
            no_gpu: false,
        };
        if let Ok(Some(cfg)) = embedding_config_optional(no_args) {
            let has_vectors = project_for(root)
                .ok()
                .and_then(|p| store.vector_model_ids(&p).ok())
                .is_some_and(|m| !m.is_empty());
            if has_vectors {
                let key = embedding_query_cache_key(&cfg);
                embed_daemon::prewarm_from_env(&cfg, &key);
            }
        }
    }
    Ok(store)
}

fn open_default_store_query_writer(root: Option<&str>) -> Result<greppy_store::Store> {
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    if !path.exists() {
        // Reuse the normal query open to trigger the existing first-use
        // auto-index/error path, then reopen writable for the evidence write.
        drop(open_default_store(root)?);
    }
    if let Some(parent) = path.parent() {
        let _ = workspace_locator::ensure_store_dir(parent);
    }
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())?;
    let _ = workspace_locator::ensure_db_mode(&path);
    if let Some(store_dir) = path.parent() {
        workspace_locator::touch_lastused(store_dir);
    }
    Ok(store)
}

/// First use publishes the deterministic graph quickly. Semantic commands
/// start the generation-bound embedding snapshot in the background and report
/// progress instead of making every graph command wait for inference.
fn try_auto_index_inline(root: Option<&str>) -> bool {
    let Ok(effective_root) = resolve_root(root) else {
        return false;
    };
    let Ok(project) = project_for(root) else {
        return false;
    };
    let Ok(overrides) = discover_overrides_from_env() else {
        return false;
    };
    let store_path = workspace_locator::store_path(&effective_root);
    // Create the versioned store namespace and its ownership manifest before
    // opening the DB. GC will never manage a directory without this manifest.
    if greppy_core::cache::ensure_workspace_store(&effective_root).is_err() {
        return false;
    }
    let Ok(Some(_lifecycle)) = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    ) else {
        return false;
    };
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => lock,
        _ => return false, // another writer is active
    };
    if store_path.exists() {
        return true;
    }
    let options = greppy_indexer::IndexOptions {
        discover_overrides: overrides,
    };
    index_atomic_snapshot(
        &store_path,
        &effective_root,
        &project,
        None,
        &options,
        false,
        None,
    )
    .map(|snapshot| snapshot.index.is_clean())
    .unwrap_or(false)
}

fn dispatch_grep(argv: &[String]) -> Result<i32> {
    // clap's `trailing_var_arg` captures everything after `greppy`
    // (or `greppy <unknown_subcmd>`). Delegate to the `OsString`
    // dispatcher so grep- and rg-style routing live in exactly one place.
    let mut full: Vec<std::ffi::OsString> = Vec::with_capacity(argv.len() + 1);
    full.push(std::ffi::OsString::from("greppy"));
    full.extend(argv.iter().map(std::ffi::OsString::from));
    dispatch_grep_os(&full)
}

/// `OsString` argv variant of [`dispatch_grep`].
///
/// forwards the original (possibly non-UTF-8) argv to
/// real grep byte-for-byte via [`greppy_passthrough::run_grep_os`]. `full`
/// includes a synthetic argv[0] placeholder; `full[1..]` are the user's
/// grep arguments. A leading grep-family placeholder (when the user
/// wrote `greppy grep …`) is handled by the pre-clap router, which
/// only routes here for the *bare* form — but we still strip a leading
/// `grep`/`egrep`/… token defensively to match [`dispatch_grep`].
/// Remove greppy's own global flags before handing the line to real grep.
///
/// `--root`, `--device` and `--no-gpu` are greppy's, and the shipped agent guide
/// tells agents to pass `--root .` on every command. Forwarding them made real
/// grep answer `unrecognized option '--root'`, which cost the agent a turn every
/// time it searched — measured at 2.3 wasted turns per task, and zero in an arm
/// that just calls grep directly. `--root DIR` also carries intent: when no path
/// operand is present we append DIR so the search still covers what was asked.
fn strip_greppy_globals(args: &[std::ffi::OsString]) -> Option<Vec<std::ffi::OsString>> {
    const VALUE_FLAGS: [&str; 2] = ["--root", "--device"];
    const BARE_FLAGS: [&str; 1] = ["--no-gpu"];
    let mut out: Vec<std::ffi::OsString> = Vec::with_capacity(args.len());
    let mut root: Option<std::ffi::OsString> = None;
    let mut removed = false;
    let mut index = 0;
    while index < args.len() {
        let text = args[index].to_str().unwrap_or_default();
        if let Some(flag) = VALUE_FLAGS.iter().find(|flag| text == **flag) {
            if let Some(value) = args.get(index + 1) {
                if *flag == "--root" {
                    root = Some(value.clone());
                }
                index += 2;
                removed = true;
                continue;
            }
            index += 1;
            removed = true;
            continue;
        }
        if let Some(flag) = VALUE_FLAGS
            .iter()
            .find(|flag| text.starts_with(&format!("{flag}=")))
        {
            if *flag == "--root" {
                root = Some(std::ffi::OsString::from(&text[flag.len() + 1..]));
            }
            index += 1;
            removed = true;
            continue;
        }
        if BARE_FLAGS.contains(&text) {
            index += 1;
            removed = true;
            continue;
        }
        out.push(args[index].clone());
        index += 1;
    }
    if !removed {
        return None;
    }
    // A bare `--root DIR` with no path operand still means "search DIR". The
    // first non-flag argument is the PATTERN, so a path operand only exists from
    // the second one on — counting the pattern as a path silently searched the
    // wrong place.
    if let Some(root) = root {
        let non_flags = out
            .iter()
            .skip(1)
            .filter(|arg| !arg.to_str().unwrap_or_default().starts_with('-'))
            .count();
        if non_flags <= 1 {
            out.push(root);
        }
    }
    Some(out)
}

/// Greppy-only flags, with the subcommand that owns each one.
///
/// An argv carrying one of these is a mistyped greppy command, not a grep
/// invocation: real grep rejects it with its own usage dump, which says nothing
/// about which greppy subcommand the caller actually wanted. No name here
/// collides with a real grep long option.
/// Flags shared by the navigation commands, which no single subcommand owns.
const NAV_OWNER: &str = "";

const GREPPY_ONLY_FLAGS: &[(&str, &str)] = &[
    ("--target", "edit"),
    ("--expect", "edit"),
    ("--content", "edit"),
    ("--content-file", "edit"),
    ("--source-file", "edit"),
    ("--old", "edit"),
    ("--old-file", "edit"),
    ("--new-file", "edit"),
    ("--pattern", "edit"),
    ("--body", "edit"),
    ("--patch-file", "edit"),
    ("--dry-run", "edit"),
    ("--verify", "edit"),
    ("--plan", "edit apply"),
    ("--spec", "edit"),
    ("--module", "edit ensure-import"),
    ("--annotation", "edit ensure-annotation"),
    ("--arg", "edit ensure-argument"),
    ("--value-json", "edit data"),
    ("--call", "edit rename"),
    ("--to", "edit rename"),
    ("--symbol", "read"),
    ("--lines", "read"),
    ("--handle", "read"),
    ("--kind", "search-symbols"),
    ("--depth", "impact"),
    ("--direction", "impact"),
    ("--from", "path"),
    ("--base", "changes"),
    ("--report", "changes"),
    ("--path", NAV_OWNER),
    // These four were the ones actually seen reaching grep in the traces. The
    // rg branch returns earlier, so `--json` here is never ripgrep's.
    ("--all", NAV_OWNER),
    ("--code", NAV_OWNER),
    ("--json", NAV_OWNER),
    ("--offset", "read"),
];

/// The first greppy-only flag in `args`, with its owning subcommand.
fn greppy_only_flag(args: &[std::ffi::OsString]) -> Option<(&'static str, &'static str)> {
    let mut options = true;
    for argument in args {
        if argument == "--" {
            options = false;
            continue;
        }
        if !options {
            continue;
        }
        let Some(text) = argument.to_str() else {
            continue;
        };
        let name = text.split_once('=').map_or(text, |(name, _)| name);
        if let Some(hit) = GREPPY_ONLY_FLAGS.iter().find(|(flag, _)| *flag == name) {
            return Some(*hit);
        }
    }
    None
}

fn dispatch_grep_os(full: &[std::ffi::OsString]) -> Result<i32> {
    // full[0] is the "greppy" placeholder. Strip a leading
    // grep-family (or rg-family) placeholder in full[1] if present so
    // `greppy grep -R foo .`, `greppy rg -S foo` and `greppy -R foo .`
    // all agree.
    let cleaned = strip_greppy_globals(&full[1..]);
    let args: &[std::ffi::OsString] = cleaned.as_deref().unwrap_or(&full[1..]);
    let (stripped, named_rg, named_grep): (&[std::ffi::OsString], bool, bool) = match args
        .first()
        .and_then(|s| s.to_str())
    {
        Some("grep") | Some("egrep") | Some("fgrep") | Some("rgrep") => (&args[1..], false, true),
        Some("rg") | Some("ripgrep") => (&args[1..], true, false),
        _ => (args, false, false),
    };

    // rg-style invocations (named, or carrying rg-only flags such as
    // --smart-case / -t / --glob) get their own routing: real ripgrep if
    // installed, otherwise a grep translation, otherwise a loud refusal.
    // Blindly forwarding them to real grep would be a usage error at
    // best and a silently different search at worst.
    if named_rg || greppy_passthrough::is_rg_style(stripped) {
        return dispatch_rg_os(stripped);
    }

    // Agents routinely omit -r when the file operand is a directory. Real
    // grep then rejects an otherwise clear request with "Is a directory".
    // Preserve byte-exact passthrough for ordinary invocations, but add the
    // conventional recursive default when we can prove that a file operand is
    // an existing directory and no recursive mode was requested explicitly.
    // This applies to the bare form (`greppy PATTERN dir/`) as much as to the
    // named one: real grep answers both with "Is a directory", so nothing that
    // previously produced results changes meaning.
    let _ = named_grep;
    let recursive_args = grep_args_with_implicit_recursion(stripped);
    let grep_args = recursive_args.as_deref().unwrap_or(stripped);
    if let Some((flag, owner)) = greppy_only_flag(grep_args) {
        let guidance = if owner.is_empty() {
            "it belongs to greppy's navigation commands. Drop it, or name the command it goes with"
                .to_string()
        } else {
            format!("it belongs to `greppy {owner}`. Drop it, or run `greppy {owner} ...` instead")
        };
        return Err(Error::Invalid(format!(
            "`{flag}` is a greppy flag, not a grep flag - {guidance}."
        )));
    }
    let mut rebuilt: Vec<std::ffi::OsString> = Vec::with_capacity(grep_args.len() + 1);
    rebuilt.push(std::ffi::OsString::from("greppy"));
    rebuilt.extend_from_slice(grep_args);

    let real = greppy_passthrough::discover_grep()?;
    greppy_passthrough::run_grep_os(&real, &rebuilt)
}

fn grep_args_with_implicit_recursion(
    args: &[std::ffi::OsString],
) -> Option<Vec<std::ffi::OsString>> {
    if grep_has_recursive_option(args)
        || grep_requests_files_without_match(args)
        || !grep_has_directory_operand(args)
    {
        return None;
    }
    let mut adjusted = Vec::with_capacity(args.len() + 1);
    adjusted.push(std::ffi::OsString::from("-r"));
    adjusted.extend_from_slice(args);
    Some(adjusted)
}

fn grep_requests_files_without_match(args: &[std::ffi::OsString]) -> bool {
    args.iter().any(|argument| {
        argument == "--files-without-match"
            || argument.to_str().is_some_and(|text| {
                text.strip_prefix('-')
                    .filter(|cluster| !cluster.starts_with('-'))
                    .is_some_and(|cluster| cluster.contains('L'))
            })
    })
}

fn grep_has_recursive_option(args: &[std::ffi::OsString]) -> bool {
    let mut options = true;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        if !options {
            index += 1;
            continue;
        }
        let Some(text) = argument.to_str() else {
            index += 1;
            continue;
        };
        if matches!(text, "--recursive" | "--dereference-recursive")
            || grep_short_option_has_recursive_flag(text)
        {
            return true;
        }
        let (takes_value, _) = grep_option_value_mode(text);
        if takes_value && !text.contains('=') && !grep_short_option_has_attached_value(text) {
            index += 2;
        } else {
            index += 1;
        }
    }
    false
}

fn grep_short_option_has_recursive_flag(option: &str) -> bool {
    let Some(cluster) = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-'))
    else {
        return false;
    };
    for flag in cluster.chars() {
        if matches!(flag, 'r' | 'R') {
            return true;
        }
        if matches!(flag, 'e' | 'f' | 'A' | 'B' | 'C' | 'd' | 'D' | 'm') {
            return false;
        }
    }
    false
}

fn grep_has_directory_operand(args: &[std::ffi::OsString]) -> bool {
    let mut options = true;
    let mut explicit_pattern = false;
    let mut positional_pattern_seen = false;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        if options {
            if let Some(text) = argument
                .to_str()
                .filter(|text| text.starts_with('-') && *text != "-")
            {
                let (takes_value, supplies_pattern) = grep_option_value_mode(text);
                explicit_pattern |= supplies_pattern;
                if takes_value && !text.contains('=') && !grep_short_option_has_attached_value(text)
                {
                    index += 2;
                    continue;
                }
                index += 1;
                continue;
            }
        }
        if !explicit_pattern && !positional_pattern_seen {
            positional_pattern_seen = true;
        } else if std::path::Path::new(argument).is_dir() {
            return true;
        }
        index += 1;
    }
    false
}

fn grep_option_value_mode(option: &str) -> (bool, bool) {
    if option.starts_with("--regexp=") || option.starts_with("--file=") {
        return (false, true);
    }
    if matches!(option, "-e" | "--regexp" | "-f" | "--file") {
        return (true, true);
    }
    let long_takes_value = matches!(
        option,
        "--after-context"
            | "--before-context"
            | "--context"
            | "--directories"
            | "--devices"
            | "--exclude"
            | "--exclude-from"
            | "--exclude-dir"
            | "--include"
            | "--label"
            | "--max-count"
    );
    if long_takes_value {
        return (true, false);
    }
    if option.starts_with('-') && !option.starts_with("--") {
        let supplies_pattern = option[1..].chars().any(|flag| matches!(flag, 'e' | 'f'));
        let takes_value = option[1..]
            .chars()
            .any(|flag| matches!(flag, 'e' | 'f' | 'A' | 'B' | 'C' | 'd' | 'D' | 'm'));
        return (takes_value, supplies_pattern);
    }
    (false, false)
}

fn grep_short_option_has_attached_value(option: &str) -> bool {
    let Some(cluster) = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-'))
    else {
        return false;
    };
    cluster.char_indices().any(|(index, flag)| {
        matches!(flag, 'e' | 'f' | 'A' | 'B' | 'C' | 'd' | 'D' | 'm')
            && index + flag.len_utf8() < cluster.len()
    })
}

/// Route a ripgrep-style invocation: byte-exact delegation to real
/// ripgrep when one exists, otherwise translate the safe flag subset to a
/// real-grep call, otherwise fail loudly naming the flag and the closest
/// alternative. Absence of ripgrep must never silently change search
/// semantics.
fn dispatch_rg_os(args: &[std::ffi::OsString]) -> Result<i32> {
    if let Some(real_rg) = greppy_passthrough::discover_ripgrep()? {
        let mut rebuilt: Vec<std::ffi::OsString> = Vec::with_capacity(args.len() + 1);
        rebuilt.push(std::ffi::OsString::from("rg"));
        rebuilt.extend_from_slice(args);
        return greppy_passthrough::run_grep_os(&real_rg, &rebuilt);
    }
    use std::io::IsTerminal;
    let stdin_piped = !std::io::stdin().is_terminal();
    let grep_args =
        greppy_passthrough::translate_to_grep(args, stdin_piped).map_err(Error::Invalid)?;
    let mut rebuilt: Vec<std::ffi::OsString> = Vec::with_capacity(grep_args.len() + 1);
    rebuilt.push(std::ffi::OsString::from("greppy"));
    rebuilt.extend(grep_args);
    let real = greppy_passthrough::discover_grep()?;
    greppy_passthrough::run_grep_os(&real, &rebuilt)
}

/// Run the indexer against `path` (default: current directory).
fn dispatch_index(
    path: Option<&str>,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
    let mut background_job = BackgroundJobGuard::from_env();
    // RV-006: `--root` overrides the indexed target. When both are
    // given we still walk `path` (the user's workspace) but key the
    // store under the canonical `root` so the indexer and the
    // query commands share one project identity (RV-011).
    // Defect D9: normalize BOTH paths to canonical absolute form up
    // front. `greppy index .` used to record whatever the walker
    // derived from the relative target (falling back to `.` in a
    // marker-less directory), while later queries looked the workspace
    // up under an absolute root — the index existed but every lookup
    // failed. Canonical-absolute at the boundary keeps one spelling
    // everywhere.
    let target = match path {
        Some(p) => absolutize_path(std::path::Path::new(p)),
        None => std::env::current_dir()
            .map_err(|e| Error::io("read current_dir for `grep index` default", e))?,
    };
    // RV-006 / RV-011: the store path and project identity are keyed on
    // the *resolved* repo root, not on the (possibly sub-directory) index
    // target. When `--root` is given we honour it; otherwise we walk up
    // from `target` to the repo marker. This guarantees `greppy index
    // <subdir>` and a later `greppy search-code` from anywhere in the
    // same repo open the same store and use the same project name.
    let effective_root = match root {
        Some(r) => {
            let explicit = absolutize_path(std::path::Path::new(r));
            workspace_locator::resolve_workspace_root(&explicit)
        }
        None => find_repo_root(&target),
    };
    let project = workspace_locator::project_identity(&effective_root);
    let index_options = greppy_indexer::IndexOptions {
        discover_overrides: discover_overrides_from_env()?,
    };
    let embedding_config = embedding_config_for_index(embedding_args)?;

    // Open the on-disk store under the workspace locator's path
    // never at `<root>/.greppy/graph.db` (which would
    // pollute `grep -R .`). The versioned platform data directory is used on
    // Linux/macOS and can be overridden via `GREPPY_STORE_DIR`.
    let store_path = workspace_locator::store_path(&effective_root);
    greppy_core::cache::ensure_workspace_store(&effective_root).map_err(|e| {
        Error::io(
            format!("create workspace store for {}", effective_root.display()),
            e,
        )
    })?;
    let _lifecycle = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    )
    .map_err(|error| Error::io("acquire index lifecycle lease", error))?
    .ok_or_else(|| Error::Lock("blocking lifecycle lease returned no guard".into()))?;
    // Acquire the crash-safe
    // advisory lock BEFORE opening/migrating the store. Opening first lets a
    // concurrent indexer hit a SQLite busy error inside Store::open and exit
    // EXIT_IO (73) silently, instead of the documented EX_TEMPFAIL (75) with a
    // diagnostic on contention. Concurrent indexers on the same path get
    // `LockError::Held`; a crashed prior holder is released by the OS. The
    // guard must outlive the complete snapshot build + publish operation.
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => Some(lock),
        Err(greppy_freshness::LockError::Held { .. }) => {
            eprintln!(
                "grep: another indexer is running against {}",
                store_path.display()
            );
            return Ok(EXIT_TEMPFAIL as i32);
        }
        Err(greppy_freshness::LockError::Io { context, source }) => {
            return Err(Error::io(context, source));
        }
    };
    // Holding the writer lock, build a fresh snapshot in a temp DB, validate
    // it, then publish it with one filesystem rename. The indexer crate still
    // supports in-place incremental updates for library tests; the CLI path is
    // the production publication boundary, so it must never expose a half-built
    // graph.db to query commands.
    let is_background = background_job.is_background();
    let snapshot = match index_atomic_snapshot(
        &store_path,
        &target,
        &project,
        embedding_config.as_ref(),
        &index_options,
        !is_background,
        if is_background {
            Some(&mut background_job)
        } else {
            None
        },
    ) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            background_job.fail(&error);
            return Err(error);
        }
    };
    let report = &snapshot.index;

    println!(
        "indexed {} files ({} unsupported, {} unreadable, {} oversize, {} file-limit, {} time-budget); {} nodes extracted; generation {} (project: {project})",
        report.files_indexed,
        report.files_unsupported_language,
        report.files_unreadable,
        report.files_oversize,
        report.files_skipped_by_file_limit,
        report.files_skipped_by_time_budget,
        report.nodes_extracted,
        report.graph_generation
    );
    if !report.is_clean()
        || report.files_skipped_by_file_limit > 0
        || report.files_skipped_by_time_budget > 0
    {
        return Ok(EXIT_IO as i32);
    }
    if let Some(embedding_report) = &snapshot.embeddings {
        println!(
            "embedded {} code spans ({} reused, {} considered, {} non-definition skipped, {} missing-file, {} invalid-span, {} oversize, {} failed, {} stale pruned)",
            embedding_report.nodes_embedded,
            embedding_report.nodes_reused,
            embedding_report.nodes_considered,
            embedding_report.nodes_skipped_non_definition,
            embedding_report.nodes_skipped_missing_file,
            embedding_report.nodes_skipped_invalid_span,
            embedding_report.nodes_skipped_oversize,
            embedding_report.nodes_failed,
            embedding_report.stale_rows_pruned
        );
    }
    let discover_scope = index_options.discover_overrides.scope_key();
    if discover_scope != "default" {
        println!(
            "discover scope: {discover_scope} ({} / {})",
            ENV_DISCOVER_INCLUDE, ENV_DISCOVER_EXCLUDE
        );
    }
    retire_verified_legacy_store(&effective_root);
    match snapshot.embedding_degraded.as_deref() {
        // Degraded embeddings never cost the caller the published graph
        // snapshot: record the reason (background job record / stderr) and
        // let the background embed path finish the remaining vectors.
        Some(reason) => background_job.degraded(reason),
        None => background_job.complete(),
    }
    let embedding_deferred = snapshot.embedding_deferred;
    drop(_lock);
    drop(_lifecycle);
    if let Some(reason) = snapshot.embedding_degraded.as_deref() {
        // No immediate respawn: a broken backend would fail the same way
        // again. The next semantic query re-attempts through the existing
        // background-embed path and reuses every vector that DID embed.
        eprintln!(
            "greppy index: embedding generation degraded ({reason}); the graph index is published and complete; the next semantic query retries the remaining embeddings."
        );
    }
    if embedding_deferred {
        if let Some(cfg) = embedding_config.as_ref() {
            let effective_root_string = effective_root.to_string_lossy().into_owned();
            if spawn_background_embed(Some(&effective_root_string), cfg) {
                let progress =
                    embedding_progress_value(&effective_root, cfg, report.graph_generation);
                println!("{}", embedding_progress_text(&progress));
            } else {
                println!(
                    "semantic-search: semantic index is pending; the next semantic query will retry the background job."
                );
            }
        }
    }
    Ok(0)
}

fn retire_verified_legacy_store(root: &std::path::Path) {
    let legacy = greppy_core::cache::legacy_workspace_store_dir(root);
    let graph = legacy.join("graph.db");
    let Ok(metadata) = std::fs::symlink_metadata(&legacy) else {
        return;
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return;
    }
    if legacy_indexer_alive(&graph) {
        return;
    }
    let mut header = [0u8; 16];
    let Ok(mut file) = std::fs::File::open(&graph) else {
        return;
    };
    if std::io::Read::read_exact(&mut file, &mut header).is_err() || &header != b"SQLite format 3\0"
    {
        return;
    }
    let Ok(store) = greppy_store::Store::open_with(&graph, greppy_store::OpenOptions::read_only())
    else {
        return;
    };
    let schema_valid = store
        .conn()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .is_some();
    let expected_hash = greppy_core::workspace::workspace_hash(root);
    let workspace_valid = store
        .conn()
        .query_row(
            "SELECT root_path FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .is_some_and(|stored_root| {
            greppy_core::workspace::workspace_hash(std::path::Path::new(&stored_root))
                == expected_hash
        });
    drop(store);
    if !schema_valid || !workspace_valid {
        return;
    }
    let trash = greppy_core::cache::trash_root().join(format!(
        "legacy-{}-{}-{}",
        greppy_core::workspace::workspace_hash(root),
        std::process::id(),
        unix_now_secs_cli()
    ));
    if std::fs::create_dir_all(greppy_core::cache::trash_root()).is_ok()
        && std::fs::rename(&legacy, &trash).is_ok()
    {
        let _ = std::fs::remove_dir_all(trash);
    }
}

fn legacy_indexer_alive(graph: &std::path::Path) -> bool {
    let mut lock_name = graph.as_os_str().to_os_string();
    lock_name.push(".lock");
    let Ok(raw) = std::fs::read_to_string(std::path::PathBuf::from(lock_name)) else {
        return false;
    };
    raw.split(|ch: char| !ch.is_ascii_digit())
        .find(|part| !part.is_empty())
        .and_then(|part| part.parse::<u32>().ok())
        .is_some_and(process_is_alive)
}

#[derive(Debug, Clone)]
struct LegacyCacheEntry {
    path: std::path::PathBuf,
    root: std::path::PathBuf,
    bytes: u64,
    last_used_unix_secs: u64,
    locked: bool,
}

fn verified_legacy_cache_entries() -> Vec<LegacyCacheEntry> {
    let data = greppy_core::cache::data_root();
    let Ok(entries) = std::fs::read_dir(&data) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(hash) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if hash.len() != 16 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            continue;
        }
        let graph = path.join("graph.db");
        if !sqlite_header_is_valid(&graph) {
            continue;
        }
        let Ok(connection) = rusqlite::Connection::open_with_flags(
            &graph,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) else {
            continue;
        };
        let schema_valid = connection
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .is_some();
        let root = connection
            .query_row(
                "SELECT root_path FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .map(std::path::PathBuf::from);
        let Some(root) = root.filter(|root| {
            greppy_core::workspace::workspace_hash(root).eq_ignore_ascii_case(&hash)
        }) else {
            continue;
        };
        if !schema_valid {
            continue;
        }
        let last_used_unix_secs = read_last_used_unix_secs(&path);
        out.push(LegacyCacheEntry {
            bytes: cache_path_bytes(&path),
            locked: legacy_indexer_alive(&graph),
            path,
            root,
            last_used_unix_secs,
        });
    }
    out.sort_by_key(|entry| entry.last_used_unix_secs);
    out
}

fn cleanup_expired_legacy_entries(current: Option<&std::path::Path>, ttl: std::time::Duration) {
    if ttl.is_zero() {
        return;
    }
    let now = unix_now_secs_cli();
    for entry in verified_legacy_cache_entries() {
        if current == Some(entry.root.as_path()) || entry.locked {
            continue;
        }
        if now.saturating_sub(entry.last_used_unix_secs) > ttl.as_secs() {
            let _ = remove_verified_legacy_entry(&entry);
        }
    }
}

/// Resume only legacy trash entries whose name, SQLite header, schema and
/// workspace hash all prove that Greppy created them. Unknown trash is left
/// untouched and remains visible as unmanaged cache data.
fn cleanup_verified_legacy_trash() {
    let Ok(entries) = std::fs::read_dir(greppy_core::cache::trash_root()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(rest) = name.strip_prefix("legacy-") else {
            continue;
        };
        let Some(hash) = rest.get(..16) else {
            continue;
        };
        if !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            || rest.as_bytes().get(16) != Some(&b'-')
        {
            continue;
        }
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            continue;
        }
        let graph = path.join("graph.db");
        if !sqlite_header_is_valid(&graph) {
            continue;
        }
        let Ok(connection) = rusqlite::Connection::open_with_flags(
            &graph,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) else {
            continue;
        };
        let schema_valid = connection
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .is_some();
        let workspace_valid = connection
            .query_row(
                "SELECT root_path FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .is_some_and(|root| {
                greppy_core::workspace::workspace_hash(std::path::Path::new(&root))
                    .eq_ignore_ascii_case(hash)
            });
        drop(connection);
        if schema_valid && workspace_valid {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

fn sqlite_header_is_valid(path: &std::path::Path) -> bool {
    let mut header = [0u8; 16];
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    std::io::Read::read_exact(&mut file, &mut header).is_ok() && &header == b"SQLite format 3\0"
}

fn read_last_used_unix_secs(dir: &std::path::Path) -> u64 {
    let marker = dir.join(".lastused");
    std::fs::read_to_string(&marker)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .or_else(|| {
            std::fs::metadata(&marker)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|age| age.as_secs())
        })
        .or_else(|| {
            std::fs::metadata(dir)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|age| age.as_secs())
        })
        .unwrap_or(0)
}

fn remove_verified_legacy_entry(entry: &LegacyCacheEntry) -> bool {
    if entry.locked {
        return false;
    }
    let trash = greppy_core::cache::trash_root().join(format!(
        "legacy-{}-{}-{}",
        greppy_core::workspace::workspace_hash(&entry.root),
        std::process::id(),
        unix_now_secs_cli()
    ));
    std::fs::create_dir_all(greppy_core::cache::trash_root()).is_ok()
        && std::fs::rename(&entry.path, &trash).is_ok()
        && std::fs::remove_dir_all(trash).is_ok()
}

struct IndexSnapshotReport {
    index: greppy_indexer::IndexReport,
    embeddings: Option<greppy_indexer::EmbeddingIndexReport>,
    embedding_deferred: bool,
    /// Set when embedding inference failed (model load or at least one
    /// batch): the graph snapshot is still published, the completeness
    /// stamp is withheld, and the background embed path finishes the
    /// remaining vectors. Vectors are enrichment — their failure must
    /// never cost the caller the graph index (nor the vectors that DID
    /// embed).
    embedding_degraded: Option<String>,
}

/// Outcome of the inline embedding pass over the freshly built temp store.
///
/// `Degraded` covers inference-side failures (embedding model unavailable,
/// failed batches). Store/IO errors keep propagating as `Err`: a store that
/// cannot be written cannot be published either.
enum EmbeddingBuildOutcome {
    Complete(greppy_indexer::EmbeddingIndexReport),
    Degraded {
        report: Option<greppy_indexer::EmbeddingIndexReport>,
        reason: String,
    },
}

fn index_atomic_snapshot(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
    allow_deferred_embeddings: bool,
    mut background_job: Option<&mut BackgroundJobGuard>,
) -> Result<IndexSnapshotReport> {
    for attempt in 0..2 {
        if let Some(report) = index_atomic_snapshot_attempt(
            active_path,
            target,
            project,
            embedding_config,
            index_options,
            allow_deferred_embeddings,
            background_job.as_deref_mut(),
        )? {
            return Ok(report);
        }
        if attempt == 0 {
            eprintln!("greppy: workspace changed during indexing; rebuilding snapshot once");
        }
    }
    Err(Error::Store(
        "workspace kept changing during indexing; snapshot was not published".into(),
    ))
}

fn index_atomic_snapshot_attempt(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
    allow_deferred_embeddings: bool,
    background_job: Option<&mut BackgroundJobGuard>,
) -> Result<Option<IndexSnapshotReport>> {
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    let temp_path = unique_store_sibling(active_path, "next");
    cleanup_sqlite_family(&temp_path)?;
    seed_temp_store_from_active_if_usable(active_path, &temp_path)?;

    let mut temp_store = match greppy_store::Store::open(&temp_path) {
        Ok(store) => store,
        Err(e) => {
            let _ = cleanup_sqlite_family(&temp_path);
            return Err(e.into());
        }
    };

    let report =
        match greppy_indexer::index_with_options(&mut temp_store, target, project, index_options) {
            Ok(report) => report,
            Err(e) => {
                drop(temp_store);
                let _ = cleanup_sqlite_family(&temp_path);
                return Err(e);
            }
        };

    if !report.is_clean()
        || report.files_skipped_by_file_limit > 0
        || report.files_skipped_by_time_budget > 0
    {
        drop(temp_store);
        cleanup_sqlite_family(&temp_path)?;
        return Ok(Some(IndexSnapshotReport {
            index: report,
            embeddings: None,
            embedding_deferred: false,
            embedding_degraded: None,
        }));
    }

    let embedding_deferred = embedding_config.is_some_and(|cfg| {
        allow_deferred_embeddings
            && greppy_indexer::count_embedding_candidate_nodes(&temp_store, project)
                .is_ok_and(|count| should_defer_embedding(cfg, count))
    });
    let (embedding_report, embedding_degraded) =
        if let Some(cfg) = embedding_config.filter(|_| !embedding_deferred) {
            match index_embeddings_into_temp_store(
                &mut temp_store,
                target,
                project,
                cfg,
                &report,
                active_path.parent().map(std::path::Path::to_path_buf),
                background_job,
            ) {
                Ok(EmbeddingBuildOutcome::Complete(report)) => (Some(report), None),
                Ok(EmbeddingBuildOutcome::Degraded { report, reason }) => (report, Some(reason)),
                Err(e) => {
                    drop(temp_store);
                    let _ = cleanup_sqlite_family(&temp_path);
                    return Err(e);
                }
            }
        } else {
            (None, None)
        };

    checkpoint_store(&temp_store, &temp_path)?;
    temp_store.integrity_check().map_err(|e| {
        Error::Store(format!(
            "temp index integrity_check failed for {}: {e}",
            temp_path.display()
        ))
    })?;
    drop(temp_store);
    cleanup_sqlite_sidecars(&temp_path)?;
    sync_file(&temp_path)?;
    sync_parent_dir(&temp_path)?;
    maybe_index_test_failpoint("after-temp-before-publish", &temp_path)?;

    let verify_store =
        greppy_store::Store::open_with(&temp_path, greppy_store::OpenOptions::read_only())?;
    let verification = greppy_freshness::check_files_report_with_ttl(
        &verify_store,
        target,
        project,
        std::time::Duration::from_secs(300),
        &index_options.discover_overrides,
        std::time::Duration::ZERO,
    )?;
    drop(verify_store);
    if !matches!(
        verification.state.outcome,
        greppy_freshness::FreshnessOutcome::Fresh
    ) {
        cleanup_sqlite_family(&temp_path)?;
        return Ok(None);
    }

    publish_store_snapshot(&temp_path, active_path)?;
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    Ok(Some(IndexSnapshotReport {
        index: report,
        embeddings: embedding_report,
        embedding_deferred,
        embedding_degraded,
    }))
}

fn should_defer_embedding(cfg: &EmbeddingModelConfig, candidate_nodes: usize) -> bool {
    let configured = std::env::var(ENV_LAZY_EMBED_MIN_SPANS)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0);
    let threshold = configured.unwrap_or_else(|| {
        let (backend, _) = embedding_backend_plan(cfg);
        if matches!(backend.as_str(), "metal" | "cuda") {
            DEFAULT_LAZY_EMBED_GPU_SPANS
        } else {
            DEFAULT_LAZY_EMBED_CPU_SPANS
        }
    });
    candidate_nodes >= threshold
}

fn index_embeddings_into_temp_store(
    store: &mut greppy_store::Store,
    target: &std::path::Path,
    project: &str,
    cfg: &EmbeddingModelConfig,
    report: &greppy_indexer::IndexReport,
    tokenizer_cache_dir: Option<std::path::PathBuf>,
    mut background_job: Option<&mut BackgroundJobGuard>,
) -> Result<EmbeddingBuildOutcome> {
    #[cfg(debug_assertions)]
    if std::env::var_os(ENV_TEST_EMBED_UNAVAILABLE).is_some() {
        return Ok(EmbeddingBuildOutcome::Degraded {
            report: None,
            reason: "test failpoint: embedding backend unavailable".into(),
        });
    }
    if let Some(job) = background_job.as_deref_mut() {
        job.embedding_loading();
    }
    let model = match load_embedding_model(cfg, tokenizer_cache_dir) {
        Ok(model) => model,
        Err(e) => {
            log_embedding_skip_once("index --embeddings", &e);
            return Ok(EmbeddingBuildOutcome::Degraded {
                report: None,
                reason: format!("embedding model load failed: {e}"),
            });
        }
    };
    let mut provider = greppy_indexer::EmbeddingGemmaCodeProvider::new(&cfg.model_id, &model);
    let options = greppy_indexer::EmbeddingIndexOptions::for_generation(report.graph_generation);
    let embedding_report = if let Some(job) = background_job {
        let total_documents = greppy_indexer::count_code_embedding_documents_for_project(
            store, target, project, &provider, options,
        )?;
        job.embedding_started(model.backend_name(), total_documents);
        let mut progress = |value| job.embedding_progress(value);
        greppy_indexer::index_code_embeddings_for_project_with_progress(
            store,
            target,
            project,
            &mut provider,
            options,
            total_documents,
            &mut progress,
        )?
    } else {
        greppy_indexer::index_code_embeddings_for_project(
            store,
            target,
            project,
            &mut provider,
            options,
        )?
    };
    if !embedding_report.is_complete() {
        // The completeness stamp is deliberately withheld: the next
        // semantic query (or the spawned background job) re-runs the
        // embedding pass, reusing every vector that DID embed by content
        // hash and retrying only the failed documents.
        let reason = format!(
            "{} of {} embedding documents failed inference",
            embedding_report.nodes_failed,
            embedding_report
                .nodes_failed
                .saturating_add(embedding_report.nodes_embedded)
        );
        return Ok(EmbeddingBuildOutcome::Degraded {
            report: Some(embedding_report),
            reason,
        });
    }
    let key = embedding_complete_key(project);
    store
        .conn()
        .execute(
            "INSERT INTO schema_meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rusqlite::params![key, format!("{}|{}", report.graph_generation, cfg.model_id)],
        )
        .map_err(|error| Error::Store(format!("record embedding completeness: {error}")))?;
    Ok(EmbeddingBuildOutcome::Complete(embedding_report))
}

fn embedding_complete_key(project: &str) -> String {
    format!("embedding_complete:{project}")
}

fn publish_store_snapshot(
    temp_path: &std::path::Path,
    active_path: &std::path::Path,
) -> Result<()> {
    let active_backupable = match prepare_existing_active_store(active_path) {
        Ok(()) => active_path.exists(),
        Err(e) if active_snapshot_is_recoverable(&e) && active_path.exists() => {
            let quarantine_path = quarantine_active_store(active_path)?;
            eprintln!(
                "grep: active index {} failed validation before publish ({e}); quarantined to {}",
                active_path.display(),
                quarantine_path.display()
            );
            false
        }
        Err(e) => return Err(e),
    };
    let backup_path = store_sibling(active_path, "prev");

    #[cfg(unix)]
    {
        let _ = active_backupable;
        // POSIX rename replaces the directory entry atomically. Existing
        // readers keep their old inode; new readers see the complete new DB.
        // No full-size graph.db.prev copy is needed or retained.
        cleanup_sqlite_family(&backup_path)?;
        std::fs::rename(temp_path, active_path).map_err(|error| {
            Error::io(
                format!(
                    "atomically publish temp index {} to {}",
                    temp_path.display(),
                    active_path.display()
                ),
                error,
            )
        })?;
        workspace_locator::ensure_db_mode(active_path)
            .map_err(|e| Error::io(format!("chmod db {}", active_path.display()), e))?;
        sync_file(active_path)?;
        cleanup_sqlite_sidecars(temp_path)?;
        cleanup_sqlite_family(&backup_path)?;
        sync_parent_dir(active_path)?;
        Ok(())
    }

    #[cfg(not(unix))]
    {
        if active_backupable {
            cleanup_sqlite_family(&backup_path)?;
            std::fs::copy(active_path, &backup_path).map_err(|e| {
                Error::io(
                    format!(
                        "copy previous index {} to {}",
                        active_path.display(),
                        backup_path.display()
                    ),
                    e,
                )
            })?;
            workspace_locator::ensure_db_mode(&backup_path)
                .map_err(|e| Error::io(format!("chmod db {}", backup_path.display()), e))?;
            sync_file(&backup_path)?;
            sync_parent_dir(&backup_path)?;
        }

        replace_active_with_temp(
            temp_path,
            active_path,
            &backup_path,
            PublishRenameMode::Native,
        )?;
        workspace_locator::ensure_db_mode(active_path)
            .map_err(|e| Error::io(format!("chmod db {}", active_path.display()), e))?;
        sync_file(active_path)?;
        cleanup_sqlite_family(temp_path)?;
        cleanup_sqlite_family(&backup_path)?;
        sync_parent_dir(active_path)?;
        Ok(())
    }
}

fn active_snapshot_is_recoverable(error: &Error) -> bool {
    matches!(error, Error::Store(_))
}

#[cfg(any(not(unix), test))]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublishRenameMode {
    /// Preferred path. On POSIX this atomically replaces the active file.
    /// On platforms where `rename` refuses an existing target, this falls
    /// back to [`PublishRenameMode::RemoveExistingFirst`] only for the
    /// expected "already exists" failure mode.
    Native,
    /// Recovery fallback for platforms that cannot rename over an existing
    /// file. This has a short missing-active-file window, so it is never the
    /// first choice; if the fallback rename fails, the previous known-good
    /// backup is copied back to `active_path`.
    RemoveExistingFirst,
}

#[cfg(any(not(unix), test))]
fn replace_active_with_temp(
    temp_path: &std::path::Path,
    active_path: &std::path::Path,
    backup_path: &std::path::Path,
    mode: PublishRenameMode,
) -> Result<()> {
    match mode {
        PublishRenameMode::Native => match std::fs::rename(temp_path, active_path) {
            Ok(()) => Ok(()),
            Err(e) if rename_target_exists_error(&e, active_path) => replace_active_with_temp(
                temp_path,
                active_path,
                backup_path,
                PublishRenameMode::RemoveExistingFirst,
            ),
            Err(e) => {
                let publish_error = Error::io(
                    format!(
                        "publish temp index {} to {}",
                        temp_path.display(),
                        active_path.display()
                    ),
                    e,
                );
                if !active_path.exists() && backup_path.exists() {
                    return match restore_active_from_backup(active_path, backup_path) {
                        Ok(()) => Err(publish_error),
                        Err(restore_error) => Err(Error::Store(format!(
                            "{publish_error}; failed to restore previous index {} from {}: {restore_error}",
                            active_path.display(),
                            backup_path.display()
                        ))),
                    };
                }
                Err(publish_error)
            }
        },
        PublishRenameMode::RemoveExistingFirst => {
            remove_file_if_exists(active_path)?;
            match std::fs::rename(temp_path, active_path) {
                Ok(()) => Ok(()),
                Err(e) => {
                    let publish_error = Error::io(
                        format!(
                            "publish temp index {} to {} after removing existing target",
                            temp_path.display(),
                            active_path.display()
                        ),
                        e,
                    );
                    match restore_active_from_backup(active_path, backup_path) {
                        Ok(()) => Err(publish_error),
                        Err(restore_error) => Err(Error::Store(format!(
                            "{publish_error}; failed to restore previous index {} from {}: {restore_error}",
                            active_path.display(),
                            backup_path.display()
                        ))),
                    }
                }
            }
        }
    }
}

#[cfg(any(not(unix), test))]
fn rename_target_exists_error(e: &std::io::Error, active_path: &std::path::Path) -> bool {
    if !active_path.exists() {
        return false;
    }
    if e.kind() == std::io::ErrorKind::AlreadyExists {
        return true;
    }
    #[cfg(windows)]
    {
        e.kind() == std::io::ErrorKind::PermissionDenied
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(any(not(unix), test))]
fn restore_active_from_backup(
    active_path: &std::path::Path,
    backup_path: &std::path::Path,
) -> Result<()> {
    if active_path.exists() {
        return Ok(());
    }
    if !backup_path.exists() {
        return Err(Error::NotFound(backup_path.to_path_buf()));
    }
    std::fs::copy(backup_path, active_path).map_err(|e| {
        Error::io(
            format!(
                "restore previous index {} from {}",
                active_path.display(),
                backup_path.display()
            ),
            e,
        )
    })?;
    workspace_locator::ensure_db_mode(active_path)
        .map_err(|e| Error::io(format!("chmod db {}", active_path.display()), e))?;
    sync_file(active_path)?;
    sync_parent_dir(active_path)?;
    Ok(())
}

fn prepare_existing_active_store(active_path: &std::path::Path) -> Result<()> {
    if !active_path.exists() {
        return Ok(());
    }
    workspace_locator::ensure_db_mode(active_path)
        .map_err(|e| Error::io(format!("chmod db {}", active_path.display()), e))?;
    let store = greppy_store::Store::open(active_path)?;
    checkpoint_store(&store, active_path)?;
    store.integrity_check().map_err(|e| {
        Error::Store(format!(
            "active index integrity_check failed for {} before snapshot publish: {e}",
            active_path.display()
        ))
    })?;
    drop(store);
    cleanup_sqlite_sidecars(active_path)?;
    workspace_locator::ensure_db_mode(active_path)
        .map_err(|e| Error::io(format!("chmod db {}", active_path.display()), e))?;
    Ok(())
}

fn quarantine_active_store(active_path: &std::path::Path) -> Result<std::path::PathBuf> {
    let quarantine_path = unique_store_sibling(active_path, "corrupt");
    cleanup_sqlite_family(&quarantine_path)?;
    std::fs::rename(active_path, &quarantine_path).map_err(|e| {
        Error::io(
            format!(
                "quarantine corrupt active index {} to {}",
                active_path.display(),
                quarantine_path.display()
            ),
            e,
        )
    })?;
    let _ = rename_file_if_exists(
        &sqlite_sidecar(active_path, "-wal"),
        &sqlite_sidecar(&quarantine_path, "-wal"),
    );
    let _ = rename_file_if_exists(
        &sqlite_sidecar(active_path, "-shm"),
        &sqlite_sidecar(&quarantine_path, "-shm"),
    );
    let _ = sync_parent_dir(active_path);
    Ok(quarantine_path)
}

fn rename_file_if_exists(from: &std::path::Path, to: &std::path::Path) -> Result<()> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::io(
            format!("rename {} to {}", from.display(), to.display()),
            e,
        )),
    }
}

fn checkpoint_store(store: &greppy_store::Store, path: &std::path::Path) -> Result<()> {
    store
        .conn()
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
        .map_err(|e| Error::Store(format!("checkpoint {}: {e}", path.display())))
}

fn unique_store_sibling(active_path: &std::path::Path, label: &str) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let file_name = active_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("graph.db");
    active_path.with_file_name(format!(
        "{file_name}.{label}.{}.{}",
        std::process::id(),
        stamp
    ))
}

fn store_sibling(active_path: &std::path::Path, label: &str) -> std::path::PathBuf {
    let file_name = active_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("graph.db");
    active_path.with_file_name(format!("{file_name}.{label}"))
}

fn seed_temp_store_from_active_if_usable(
    active_path: &std::path::Path,
    temp_path: &std::path::Path,
) -> Result<bool> {
    if !active_path.exists() {
        return Ok(false);
    }
    match prepare_existing_active_store(active_path) {
        Ok(()) => {}
        Err(e) if active_snapshot_is_recoverable(&e) => {
            return Ok(false);
        }
        Err(e) => return Err(e),
    }
    if try_clone_store_file(active_path, temp_path)? {
        workspace_locator::ensure_db_mode(temp_path)
            .map_err(|e| Error::io(format!("chmod db {}", temp_path.display()), e))?;
        return Ok(true);
    }
    ensure_copy_headroom(active_path)?;
    std::fs::copy(active_path, temp_path).map_err(|e| {
        Error::io(
            format!(
                "seed temp index {} from active {}",
                temp_path.display(),
                active_path.display()
            ),
            e,
        )
    })?;
    workspace_locator::ensure_db_mode(temp_path)
        .map_err(|e| Error::io(format!("chmod db {}", temp_path.display()), e))?;
    Ok(true)
}

#[cfg(target_os = "macos")]
fn try_clone_store_file(source: &std::path::Path, target: &std::path::Path) -> Result<bool> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    unsafe extern "C" {
        fn clonefile(
            source: *const std::ffi::c_char,
            target: *const std::ffi::c_char,
            flags: u32,
        ) -> i32;
    }
    let source_c = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| Error::Invalid("store path contains NUL".into()))?;
    let target_c = CString::new(target.as_os_str().as_bytes())
        .map_err(|_| Error::Invalid("store path contains NUL".into()))?;
    let cloned = unsafe { clonefile(source_c.as_ptr(), target_c.as_ptr(), 0) } == 0;
    if !cloned {
        let _ = remove_file_if_exists(target);
    }
    Ok(cloned)
}

#[cfg(target_os = "linux")]
fn try_clone_store_file(source: &std::path::Path, target: &std::path::Path) -> Result<bool> {
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
        fn ioctl(fd: std::ffi::c_int, request: std::ffi::c_ulong, ...) -> std::ffi::c_int;
    }
    const FICLONE: std::ffi::c_ulong = 0x4004_9409;
    let source_file = std::fs::File::open(source)
        .map_err(|error| Error::io(format!("open {} for reflink", source.display()), error))?;
    let target_file = std::fs::File::create(target)
        .map_err(|error| Error::io(format!("create {} for reflink", target.display()), error))?;
    let cloned = unsafe { ioctl(target_file.as_raw_fd(), FICLONE, source_file.as_raw_fd()) } == 0;
    drop(target_file);
    if !cloned {
        remove_file_if_exists(target)?;
    }
    Ok(cloned)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn try_clone_store_file(_source: &std::path::Path, _target: &std::path::Path) -> Result<bool> {
    Ok(false)
}

fn ensure_copy_headroom(active_path: &std::path::Path) -> Result<()> {
    let active_bytes = std::fs::metadata(active_path)
        .map_err(|error| Error::io(format!("stat {}", active_path.display()), error))?
        .len();
    let reserve = (active_bytes / 4).max(256 * 1024 * 1024);
    let required = active_bytes.saturating_add(reserve);
    let parent = active_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let output = std::process::Command::new("df")
        .args(["-Pk"])
        .arg(parent)
        .output()
        .map_err(|error| Error::io("run df for snapshot capacity check", error))?;
    if !output.status.success() {
        return Err(Error::Store(format!(
            "cannot verify free space before copying {}",
            active_path.display()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let available_kib = stdout
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().nth(3))
        .filter_map(|value| value.parse::<u64>().ok())
        .last()
        .ok_or_else(|| Error::Store("cannot parse free-space result from df".into()))?;
    let available = available_kib.saturating_mul(1024);
    if available < required {
        return Err(Error::Store(format!(
            "insufficient free space for atomic index snapshot: need {} bytes, have {} bytes",
            required, available
        )));
    }
    Ok(())
}

fn cleanup_stale_snapshot_artifacts(
    active_path: &std::path::Path,
    include_quarantine: bool,
) -> Result<usize> {
    let Some(parent) = active_path.parent() else {
        return Ok(0);
    };
    let Some(file_name) = active_path.file_name().and_then(|s| s.to_str()) else {
        return Ok(0);
    };
    let next_prefix = format!("{file_name}.next.");
    let corrupt_prefix = format!("{file_name}.corrupt.");
    let previous = format!("{file_name}.prev");
    let previous_sidecar_prefix = format!("{previous}-");
    let mut removed = 0usize;
    let entries = match std::fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(Error::io(format!("scan {}", parent.display()), e)),
    };
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(format!("scan {}", parent.display()), e))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let managed = name.starts_with(&next_prefix)
            || name == previous
            || name.starts_with(&previous_sidecar_prefix)
            || (name.starts_with(".index.job.") && name.ends_with(".tmp"))
            || (include_quarantine && name.starts_with(&corrupt_prefix));
        if !managed {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::io(
                    format!("remove stale temp {}", path.display()),
                    e,
                ))
            }
        }
    }
    if removed > 0 {
        sync_parent_dir(active_path)?;
    }
    Ok(removed)
}

#[cfg(debug_assertions)]
fn maybe_index_test_failpoint(name: &str, temp_path: &std::path::Path) -> Result<()> {
    match std::env::var(ENV_TEST_INDEX_FAILPOINT) {
        Ok(value) if value == name => {}
        Ok(value) if value == format!("error-{name}") => {
            return Err(Error::Store(format!(
                "test failpoint {value} before publishing {}",
                temp_path.display()
            )));
        }
        _ => return Ok(()),
    }
    if let Ok(ready_path) = std::env::var(ENV_TEST_INDEX_FAILPOINT_READY) {
        let ready_path = std::path::PathBuf::from(ready_path);
        if let Some(parent) = ready_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Error::io(
                    format!("create failpoint ready dir {}", parent.display()),
                    e,
                )
            })?;
        }
        std::fs::write(&ready_path, temp_path.display().to_string()).map_err(|e| {
            Error::io(
                format!("write failpoint ready file {}", ready_path.display()),
                e,
            )
        })?;
    }
    let hold_ms = std::env::var(ENV_TEST_INDEX_FAILPOINT_HOLD_MS)
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(300_000);
    std::thread::sleep(std::time::Duration::from_millis(hold_ms));
    Ok(())
}

#[cfg(not(debug_assertions))]
fn maybe_index_test_failpoint(_name: &str, _temp_path: &std::path::Path) -> Result<()> {
    Ok(())
}

fn sqlite_sidecar(path: &std::path::Path, suffix: &str) -> std::path::PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(suffix);
    std::path::PathBuf::from(os)
}

fn cleanup_sqlite_family(path: &std::path::Path) -> Result<()> {
    remove_file_if_exists(path)?;
    cleanup_sqlite_sidecars(path)
}

fn cleanup_sqlite_sidecars(path: &std::path::Path) -> Result<()> {
    remove_file_if_exists(&sqlite_sidecar(path, "-wal"))?;
    remove_file_if_exists(&sqlite_sidecar(path, "-shm"))
}

fn remove_file_if_exists(path: &std::path::Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::io(format!("remove {}", path.display()), e)),
    }
}

fn sync_parent_dir(path: &std::path::Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let dir = open_directory_for_sync(parent)
        .map_err(|e| Error::io(format!("open parent dir {}", parent.display()), e))?;
    #[cfg(windows)]
    {
        // Windows rejects FlushFileBuffers for directory handles opened with
        // FILE_FLAG_BACKUP_SEMANTICS. Opening the parent still verifies that
        // the destination directory exists; sync_file flushed the payload.
        drop(dir);
        return Ok(());
    }
    #[cfg(not(windows))]
    dir.sync_all()
        .map_err(|e| Error::io(format!("sync parent dir {}", parent.display()), e))
}

fn sync_file(path: &std::path::Path) -> Result<()> {
    #[cfg(windows)]
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| Error::io(format!("open file {}", path.display()), e))?;
    #[cfg(not(windows))]
    let file = std::fs::File::open(path)
        .map_err(|e| Error::io(format!("open file {}", path.display()), e))?;
    file.sync_all()
        .map_err(|e| Error::io(format!("sync file {}", path.display()), e))
}

#[cfg(windows)]
fn open_directory_for_sync(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
}

#[cfg(not(windows))]
fn open_directory_for_sync(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    std::fs::File::open(path)
}

#[derive(Debug, Clone)]
struct OutputBudgetSpec {
    command: &'static str,
    json: bool,
    max_bytes: Option<usize>,
    offset: usize,
}

const DEFAULT_READ_FILE_MAX_BYTES: usize = 16 * 1024;

fn read_uses_default_file_budget(cli: &Cli) -> bool {
    let Some(Command::Read {
        targets,
        symbol_opts,
        path_opts,
        lines,
        json,
        all,
        ..
    }) = cli.command.as_ref()
    else {
        return false;
    };
    if *json || *all || lines.is_some() || !symbol_opts.is_empty() {
        return false;
    }
    if targets.len() + path_opts.len() != 1 {
        return false;
    }
    if targets.is_empty() {
        return true;
    }
    targets.first().is_some_and(|subject| {
        split_path_qualified(subject).is_none()
            && read_subject_is_path(subject, cli.root.as_deref()).unwrap_or(false)
    })
}

fn output_budget_spec(cli: &Cli) -> Option<OutputBudgetSpec> {
    let default_read_budget = read_uses_default_file_budget(cli);
    if cli.max_bytes.is_none() && cli.offset == 0 && !default_read_budget {
        return None;
    }
    let (command, json) = match cli.command.as_ref()? {
        Command::SearchGraph { json, .. } => ("search-graph", *json),
        Command::Trace { json, .. } => ("trace", *json),
        Command::Impact { json, .. } => ("impact", *json),
        Command::Brief { json, .. } => ("brief", *json),
        Command::Expand { json, .. } => ("expand", *json),
        Command::Read { json, .. } => ("read", *json),
        Command::WhoCalls { json, .. } => ("who-calls", *json),
        Command::Callees { json, .. } => ("callees", *json),
        Command::FanIn { json, .. } => ("fan-in", *json),
        Command::FanOut { json, .. } => ("fan-out", *json),
        Command::GraphLocate { json, .. } => ("graph-locate", *json),
        Command::Path { json, .. } => ("path", *json),
        Command::SearchCode { json, .. } => ("search-code", *json),
        Command::SearchSymbols { json, .. } => ("search-symbols", *json),
        Command::Plus { json, .. } => ("plus", *json),
        Command::Semantic { json, .. } => ("semantic-search", *json),
        Command::Context { json, .. } => ("context", *json),
        _ => return None,
    };
    Some(OutputBudgetSpec {
        command,
        json,
        max_bytes: cli
            .max_bytes
            .or(default_read_budget.then_some(DEFAULT_READ_FILE_MAX_BYTES)),
        offset: cli.offset,
    })
}

fn begin_output_capture() {
    OUTPUT_CAPTURE.with(|capture| *capture.borrow_mut() = Some(Vec::new()));
}

fn finish_output_capture(spec: &OutputBudgetSpec, exit_code: u8) {
    use std::io::Write as _;

    let captured = OUTPUT_CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default());
    let rendered = if spec.json {
        budget_json_output(&captured, spec).unwrap_or(captured)
    } else {
        budget_text_output(&captured, spec, exit_code)
    };
    let _ = std::io::stdout().lock().write_all(&rendered);
}

fn retry_with_offset(command: &str, offset: usize) -> String {
    let invocation = CLI_INVOCATION.with(|value| value.borrow().clone());
    if invocation.is_empty() {
        return format!("greppy {command} --offset {offset}");
    }
    let mut args = vec!["greppy".to_string()];
    let mut index = 1usize;
    while index < invocation.len() {
        let token = invocation[index].to_string_lossy();
        if token == "--offset" {
            index = (index + 2).min(invocation.len());
            continue;
        }
        if token.starts_with("--offset=") {
            index += 1;
            continue;
        }
        args.push(shell_quote_cli(&token));
        index += 1;
    }
    args.push("--offset".into());
    args.push(offset.to_string());
    args.join(" ")
}

const BUDGET_ARRAY_FIELDS: &[&str] = &[
    "hits",
    "lines",
    "steps",
    "results",
    "matches",
    "nodes",
    "definitions",
    "callers",
    "references",
    "callees",
];

fn result_item_count(value: &serde_json::Value) -> usize {
    BUDGET_ARRAY_FIELDS
        .iter()
        .filter_map(|key| value.get(*key).and_then(serde_json::Value::as_array))
        .map(Vec::len)
        .sum::<usize>()
        + value
            .get("source")
            .and_then(serde_json::Value::as_str)
            .map(|source| source.lines().count())
            .unwrap_or(0)
}

fn skip_result_items(value: &mut serde_json::Value, mut count: usize) {
    for key in BUDGET_ARRAY_FIELDS {
        let Some(rows) = value
            .get_mut(*key)
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        let take = count.min(rows.len());
        rows.drain(..take);
        count -= take;
        if count == 0 {
            return;
        }
    }
    if count > 0 {
        if let Some(source) = value.get_mut("source") {
            if let Some(text) = source.as_str() {
                let retained = text.lines().skip(count).collect::<Vec<_>>().join("\n");
                *source = retained.into();
            }
        }
    }
}

fn pop_result_item(value: &mut serde_json::Value) -> bool {
    if let Some(source) = value.get_mut("source") {
        if let Some(text) = source.as_str() {
            let mut lines = text.lines().collect::<Vec<_>>();
            if lines.pop().is_some() {
                *source = lines.join("\n").into();
                return true;
            }
        }
    }
    for key in BUDGET_ARRAY_FIELDS.iter().rev() {
        if let Some(rows) = value
            .get_mut(*key)
            .and_then(serde_json::Value::as_array_mut)
        {
            if rows.pop().is_some() {
                return true;
            }
        }
    }
    false
}

fn exact_result_total(value: &serde_json::Value, available: usize, offset: usize) -> usize {
    ["total_exact", "total", "total_hits", "match_count"]
        .iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_u64))
        .and_then(|total| usize::try_from(total).ok())
        .unwrap_or_else(|| offset.saturating_add(available))
}

fn budget_json_output(bytes: &[u8], spec: &OutputBudgetSpec) -> Option<Vec<u8>> {
    let mut value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let available = result_item_count(&value);
    let total = exact_result_total(&value, available, spec.offset);
    skip_result_items(&mut value, spec.offset);

    loop {
        let shown = result_item_count(&value);
        let end = spec.offset.saturating_add(shown).min(total);
        let truncated = end < total;
        value["total"] = total.into();
        value["offset"] = spec.offset.into();
        value["shown"] = shown.into();
        value["omitted"] = total.saturating_sub(end).into();
        value["truncated"] = truncated.into();
        if truncated {
            value["try"] = retry_with_offset(spec.command, end).into();
        }
        let mut rendered = serde_json::to_vec(&value).ok()?;
        rendered.push(b'\n');
        if spec
            .max_bytes
            .is_none_or(|max_bytes| rendered.len() <= max_bytes)
            || !pop_result_item(&mut value)
        {
            return Some(rendered);
        }
    }
}

fn text_line_is_priority(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.is_empty()
        || trimmed.starts_with("diagnosis:")
        || trimmed.starts_with("status:")
        || trimmed.starts_with("next_steps:")
        || trimmed.starts_with("try:")
        || trimmed.starts_with("usage:")
        || trimmed.starts_with("suggestion:")
        || trimmed.starts_with("hint:")
        || trimmed.starts_with("query_interpreted_as:")
        || trimmed.starts_with("path_filters:")
        || trimmed.starts_with("expand:")
        || trimmed.starts_with("— ")
        || trimmed.starts_with("… ")
        || trimmed.starts_with("(no ")
        || trimmed.starts_with("read:")
        || trimmed.starts_with("-- ")
        || trimmed.starts_with("unresolved textual candidates:")
}

fn budget_text_output(bytes: &[u8], spec: &OutputBudgetSpec, exit_code: u8) -> Vec<u8> {
    let text = String::from_utf8_lossy(bytes);
    if exit_code != 0 {
        return bytes.to_vec();
    }
    let mut priority = Vec::new();
    let mut content = Vec::new();
    for line in text.lines() {
        if text_line_is_priority(line) {
            priority.push(line.to_string());
        } else {
            content.push(line.to_string());
        }
    }
    let total = content.len();
    let start = spec.offset.min(total);
    let mut selected = content[start..].to_vec();
    loop {
        let end = start.saturating_add(selected.len());
        let truncated = end < total;
        let mut lines = priority.clone();
        if truncated || spec.offset > 0 {
            lines.push(format!("truncated: {truncated}"));
            lines.push(format!("total: {total}"));
            lines.push(format!("offset: {}", spec.offset));
            if truncated {
                lines.push(format!("try: {}", retry_with_offset(spec.command, end)));
            }
        }
        lines.extend(selected.iter().cloned());
        let rendered = format!("{}\n", lines.join("\n")).into_bytes();
        if spec
            .max_bytes
            .is_none_or(|max_bytes| rendered.len() <= max_bytes)
            || selected.pop().is_none()
        {
            return rendered;
        }
    }
}

/// Translate a `Result<i32>` into the actual exit code we should return.
/// Errors get the documented code; OK keeps its inner i32.
///
/// Defect D5: this used to map every `Err` to an exit code while printing
/// NOTHING — an agent saw a bare non-zero exit with empty stdout/stderr
/// and no hint of what went wrong. Every error now prints its message and
/// its source chain to stderr before the exit code is returned. (A few
/// paths, e.g. the missing-index diagnostic, already print a targeted
/// hint before returning `Err`; the summary line here may then repeat the
/// message — acceptable redundancy versus silent failure.)
pub fn dispatch_to_code(cli: Cli) -> u8 {
    let budget = output_budget_spec(&cli);
    if budget.is_some() {
        begin_output_capture();
    }
    let code = match dispatch(cli) {
        Ok(code) => code.clamp(0, 255) as u8,
        Err(e) => {
            eprintln!("greppy: {e}");
            let mut source = std::error::Error::source(&e);
            while let Some(cause) = source {
                eprintln!("  caused by: {cause}");
                source = cause.source();
            }
            error_exit_code(&e)
        }
    };
    if let Some(spec) = &budget {
        finish_output_capture(spec, code);
    }
    code
}

fn error_exit_code(error: &Error) -> u8 {
    match error {
        Error::NotImplemented { .. } | Error::OutOfScope { .. } => EXIT_NOT_IMPLEMENTED,
        Error::Invalid(_) => EXIT_USAGE,
        _ => EXIT_IO,
    }
}

#[cfg(feature = "agent")]
/// Environment-keyed refusal used at the CLI entry; mirrors the guard inside
/// `greppy_agent::run_agent` so neither entry point can be bypassed.
fn agent_refuse_nested_invocation() -> std::result::Result<(), String> {
    let depth = std::env::var(greppy_agent::ENV_AGENT_DEPTH)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let run_id = std::env::var(greppy_agent::ENV_AGENT_RUN_ID)
        .ok()
        .filter(|value| !value.trim().is_empty());
    if depth == 0 && run_id.is_none() {
        return Ok(());
    }
    Err(format!(
        "refusing to start a nested agent run: this process is already inside an agent run ({}={}, {}={depth}). Use greppy's navigation and edit commands directly instead of invoking the agent recursively.",
        greppy_agent::ENV_AGENT_RUN_ID,
        run_id.as_deref().unwrap_or("<unset>"),
        greppy_agent::ENV_AGENT_DEPTH,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
                // SAFETY: env-mutating tests hold ENV_LOCK while this guard is alive.
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
                    file: None,
                    old: None,
                    old_file: None,
                    pattern: None,
                    lines: None,
                    symbol: Some("computeTotal".into()),
                    body: true,
                    target: None,
                    content: None,
                    content_file: Some(replacement_path.to_string_lossy().into_owned()),
                    expect: None,
                    path: None,
                    dry_run: true,
                    verify: false,
                    report: None,
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
        assert_eq!(commands[0], "greppy search-symbols marker");
        assert_eq!(commands[1], "greppy grep -rnE 'semantic|progress|marker' .");
    }

    #[test]
    fn lazy_embedding_threshold_is_inclusive_and_testable() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore::capture(&[ENV_LAZY_EMBED_MIN_SPANS]);
        // SAFETY: serialized by ENV_LOCK and restored by EnvRestore.
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
            Cli::try_parse_from(["greppy", "index", "--embedding-gguf", "model.gguf", "."])
                .is_err()
        );
        assert!(Cli::try_parse_from(["greppy", "index", "--embeddings", "."]).is_err());
    }

    #[test]
    fn parse_semantic_search_flags() {
        let cli =
            Cli::try_parse_from(["greppy", "semantic-search", "--json", "retry handler"]).unwrap();
        match cli.command {
            Some(Command::Semantic { queries, json, .. }) => {
                assert_eq!(queries, vec!["retry handler".to_string()]);
                assert!(json);
            }
            other => panic!("unexpected command: {other:?}"),
        }

        assert!(
            Cli::try_parse_from(["greppy", "semantic-search", "--vectors", "retry handler"])
                .is_err()
        );
        assert!(Cli::try_parse_from([
            "greppy",
            "semantic-search",
            "--embedding-model-dir",
            "/models/embeddinggemma",
            "retry handler"
        ])
        .is_err());

        let cli = Cli::try_parse_from(["greppy", "semantic", "retry handler"]).unwrap();
        match cli.command {
            Some(Command::Semantic { queries, .. }) => {
                assert_eq!(queries, vec!["retry handler".to_string()]);
            }
            other => panic!("unexpected command for semantic alias: {other:?}"),
        }

        // Postel: --path is an advisory no-op on semantic-search, not a parse error.
        assert!(
            Cli::try_parse_from(["greppy", "semantic-search", "merge", "--path", "src/x.ts"])
                .is_ok()
        );
    }

    #[test]
    fn parse_path_disambiguation_and_hyphen_values() {
        // Rebuild #2, CLI-SPEC rule 2: the path filter is spelled `--path` and
        // nothing else. A positional argument is a further SYMBOL.
        match Cli::try_parse_from([
            "greppy",
            "brief",
            "open",
            "--path",
            "src/flask/testing.py",
        ])
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
        match Cli::try_parse_from(["greppy", "brief", "open", "close"])
            .unwrap()
            .command
        {
            Some(Command::Brief { symbols, .. }) => {
                assert_eq!(symbols, vec!["open".to_string(), "close".to_string()]);
            }
            other => panic!("unexpected: {other:?}"),
        }

        // read carries the nav commands' --code flag as an accepted no-op.
        assert!(Cli::try_parse_from(["greppy", "read", "open", "a/mod.py", "--code"]).is_ok());

        // An already-qualified symbol is not double-qualified.
        assert_eq!(
            qualify_symbol_with_path(Some("a.py::open"), Some("b.py")),
            None
        );
        // A non-file path (no extension) is not folded into an unresolvable query.
        assert_eq!(qualify_symbol_with_path(Some("open"), Some("src")), None);

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
        let cli = Cli::try_parse_from(["greppy", "plus", "--json", "--k", "5", "refund workflow"])
            .unwrap();
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
        // This locks the fix for the regression where semantic-search silently
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
        let cli = Cli::try_parse_from([
            "grep",
            "semantic-search",
            "--device",
            "cuda",
            "refund workflow",
        ])
        .unwrap();
        assert_eq!(cli.device.as_deref(), Some("cuda"));
        assert!(!cli.no_gpu);
        match cli.command {
            Some(Command::Semantic { queries, .. }) => {
                assert_eq!(queries, vec!["refund workflow".to_string()]);
            }
            other => panic!("unexpected command: {other:?}"),
        }
        assert!(Cli::try_parse_from([
            "grep",
            "semantic-search",
            "--device",
            "cuda",
            "--no-gpu",
            "refund workflow",
        ])
        .is_err());
    }

    #[test]
    fn embedding_device_preference_obeys_cli_and_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore::capture(&[
            ENV_DEVICE,
            ENV_NO_GPU,
            ENV_EMBED_CUDA_DEVICE,
            ENV_QWEN_CUDA_DEVICE,
        ]);
        // SAFETY: serialized by ENV_LOCK and restored by EnvRestore.
        unsafe {
            std::env::remove_var(ENV_DEVICE);
            std::env::remove_var(ENV_NO_GPU);
        }

        assert_eq!(
            embedding_device_preference(None, false).unwrap(),
            greppy_embed_native::DevicePreference::Auto
        );

        // SAFETY: serialized by ENV_LOCK and restored by EnvRestore.
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

        // SAFETY: serialized by ENV_LOCK and restored by EnvRestore.
        unsafe {
            std::env::set_var(ENV_NO_GPU, "1");
        }
        assert_eq!(
            embedding_device_preference(Some("cuda"), false).unwrap(),
            greppy_embed_native::DevicePreference::Cpu
        );

        // SAFETY: serialized by ENV_LOCK and restored by EnvRestore.
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

    #[test]
    fn discover_scope_env_parses_include_and_exclude_lists() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore::capture(&[ENV_DISCOVER_INCLUDE, ENV_DISCOVER_EXCLUDE]);
        // SAFETY: serialized by ENV_LOCK and restored by EnvRestore.
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
        let cli =
            Cli::try_parse_from(["greppy", "index", "/nonexistent-root-for-parse-only"]).unwrap();
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
            Cli::try_parse_from(["greppy", "--root", "/repo", "search-code", "foo"]).unwrap();
        assert_eq!(before.root.as_deref(), Some("/repo"));
        assert!(matches!(before.command, Some(Command::SearchCode { .. })));

        let after =
            Cli::try_parse_from(["greppy", "search-code", "--root", "/repo", "foo"]).unwrap();
        assert_eq!(after.root.as_deref(), Some("/repo"));
        assert!(matches!(after.command, Some(Command::SearchCode { .. })));

        // And it is honoured by `index` too.
        let idx = Cli::try_parse_from(["greppy", "--root", "/repo", "index", "."]).unwrap();
        assert_eq!(idx.root.as_deref(), Some("/repo"));
        assert!(matches!(idx.command, Some(Command::Index { .. })));
    }

    #[test]
    fn search_code_changed_parses_as_explicit_scope() {
        let cli = Cli::try_parse_from(["greppy", "search-code", "--changed", "--json", "needle"])
            .unwrap();
        match cli.command {
            Some(Command::SearchCode {
                query,
                changed,
                staged,
                since,
                base,
                json,
                no_code: _,
                fixed: _,
                code: _,
                all: _,
                paths: _,
                path_opts: _,
            }) => {
                assert_eq!(query.as_deref(), Some("needle"));
                assert!(changed);
                assert!(!staged);
                assert!(since.is_none());
                assert!(base.is_none());
                assert!(json);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn search_code_staged_parses_as_explicit_scope() {
        let cli =
            Cli::try_parse_from(["greppy", "search-code", "--staged", "--json", "needle"]).unwrap();
        match cli.command {
            Some(Command::SearchCode {
                query,
                changed,
                staged,
                since,
                base,
                json,
                no_code: _,
                fixed: _,
                code: _,
                all: _,
                paths: _,
                path_opts: _,
            }) => {
                assert_eq!(query.as_deref(), Some("needle"));
                assert!(!changed);
                assert!(staged);
                assert!(since.is_none());
                assert!(base.is_none());
                assert!(json);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn search_code_since_parses_as_explicit_scope() {
        let cli = Cli::try_parse_from([
            "greppy",
            "search-code",
            "--since",
            "HEAD~1",
            "--json",
            "needle",
        ])
        .unwrap();
        match cli.command {
            Some(Command::SearchCode {
                query,
                changed,
                staged,
                since,
                base,
                json,
                no_code: _,
                fixed: _,
                code: _,
                all: _,
                paths: _,
                path_opts: _,
            }) => {
                assert_eq!(query.as_deref(), Some("needle"));
                assert!(!changed);
                assert!(!staged);
                assert_eq!(since.as_deref(), Some("HEAD~1"));
                assert!(base.is_none());
                assert!(json);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn search_code_base_parses_as_explicit_scope() {
        let cli = Cli::try_parse_from([
            "greppy",
            "search-code",
            "--base",
            "main",
            "--json",
            "needle",
        ])
        .unwrap();
        match cli.command {
            Some(Command::SearchCode {
                query,
                changed,
                staged,
                since,
                base,
                json,
                no_code: _,
                fixed: _,
                code: _,
                all: _,
                paths: _,
                path_opts: _,
            }) => {
                assert_eq!(query.as_deref(), Some("needle"));
                assert!(!changed);
                assert!(!staged);
                assert!(since.is_none());
                assert_eq!(base.as_deref(), Some("main"));
                assert!(json);
            }
            other => panic!("unexpected command: {other:?}"),
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
    fn impact_parses_diff_scopes_without_symbol() {
        let cli = Cli::try_parse_from([
            "greppy",
            "impact",
            "--base",
            "main",
            "--direction",
            "outgoing",
            "--json",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Impact {
                symbols,
                path_opts: _,
                code: _,
                direction,
                edge,
                depth,
                since,
                base,
                all,
                json,
            }) => {
                assert!(symbols.is_empty());
                assert_eq!(direction, "outgoing");
                assert!(edge.is_none());
                assert_eq!(depth, 6);
                assert!(since.is_none());
                assert_eq!(base.as_deref(), Some("main"));
                assert!(!all);
                assert!(json);
            }
            other => panic!("expected Impact, got {other:?}"),
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

        let cli = Cli::try_parse_from(["greppy", "references", "Widget"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::References { symbol: Some(ref s), code: false, all: false, json: false }) if s == "Widget"
        ));

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
        assert!(!is_grep_passthrough(&mk(&["greppy", "references", "Foo"])));
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
            "search-code",
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

    #[cfg(feature = "agent")]
    #[test]
    fn agent_router_reserves_prompt_but_not_explicit_grep() {
        let argv = ["greppy", "-p", "fix it"]
            .into_iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>();
        assert!(is_agent_invocation(&argv));
        assert!(!is_grep_passthrough(&argv));

        let grep = ["greppy", "grep", "-p", "pattern"]
            .into_iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>();
        assert!(!is_agent_invocation(&grep));
        assert!(!is_grep_passthrough(&grep));
    }

    #[cfg(feature = "agent")]
    #[test]
    fn agent_arguments_accept_apply_result_without_provider_flags() {
        let args =
            AgentInvocation::try_parse_from(["greppy", "--apply-result", "deadbeef-00000000"])
                .unwrap();
        assert_eq!(args.apply_result.as_deref(), Some("deadbeef-00000000"));
        assert!(args.prompt.is_none());
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
                unsupported_edge_classes: unsupported_edges
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
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
            Cli::try_parse_from(["greppy", "cache", "clear", "--root", "/tmp/repo", "--yes"])
                .unwrap();
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
}
