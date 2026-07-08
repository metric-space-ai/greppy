//! `greppy` CLI — the unified subcommand dispatcher.
//!
//! Subcommand surface:
//! - `grep`         — drop-in wrapper, delegates to real grep.
//! - `index`        — index a repo.
//! - `search-graph` — graph search.
//! - `who-calls` / `callees` / `find-usages` / `impact` / `brief` — graph navigation.
//! - `semantic-search` (`semantic`) — meaning-based code search.
//! - `search-code` / `search-symbols` — indexed code and symbol search.
//! - `install`      — agent installer      (out of scope)
//! - `uninstall`    — agent uninstaller    (out of scope)
//! - `update`       — agent updater        (out of scope)
//! - `config`       — runtime config       (out of scope)
//!
//! Out-of-scope lifecycle subcommands print a structured error and exit
//! with a documented non-zero code (EX_UNAVAILABLE = 69).

#[cfg(not(feature = "embedded-model"))]
compile_error!(
    "greppy cannot be built without the embedded EmbeddingGemma model. \
     Every greppy binary must bake crates/cli/assets/embedded-model/* \
     from this repo into the binary."
);

use clap::{Parser, Subcommand};
use greppy_core::error::{Error, Result};
use greppy_core::workspace as workspace_locator;
use greppy_freshness::LockOutcome;

#[cfg(unix)]
mod embed_daemon;

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

const DEFAULT_EMBEDDINGGEMMA_MODEL_ID: &str = "google/embeddinggemma-300m";
const ENV_EMBED_INDEX: &str = "GREPPY_EMBEDDINGGEMMA_INDEX";
const ENV_EMBED_MODEL_DIR: &str = "GREPPY_EMBEDDINGGEMMA_MODEL";
const ENV_EMBED_GGUF: &str = "GREPPY_EMBEDDINGGEMMA_GGUF";
const ENV_EMBED_TOKENIZER: &str = "GREPPY_EMBEDDINGGEMMA_TOKENIZER";
const ENV_EMBED_MODEL_ID: &str = "GREPPY_EMBEDDINGGEMMA_MODEL_ID";
const ENV_EMBED_MAX_LENGTH: &str = "GREPPY_EMBEDDINGGEMMA_MAX_LENGTH";
const ENV_DEVICE: &str = "GREPPY_DEVICE";
const ENV_NO_GPU: &str = "GREPPY_NO_GPU";
const ENV_VECTOR_EXACT_CANDIDATE_LIMIT: &str = "GREPPY_VECTOR_EXACT_CANDIDATE_LIMIT";
const ENV_PROVIDER_POLICY: &str = "GREPPY_PROVIDER_POLICY";
const ENV_DISCOVER_INCLUDE: &str = "GREPPY_DISCOVER_INCLUDE";
const ENV_DISCOVER_EXCLUDE: &str = "GREPPY_DISCOVER_EXCLUDE";
const ENV_EXPAND_TTL_SECS: &str = "GREPPY_EXPAND_TTL_SECS";
#[cfg(debug_assertions)]
const ENV_TEST_INDEX_FAILPOINT: &str = "GREPPY_TEST_INDEX_FAILPOINT";
#[cfg(debug_assertions)]
const ENV_TEST_INDEX_FAILPOINT_READY: &str = "GREPPY_TEST_INDEX_FAILPOINT_READY";
#[cfg(debug_assertions)]
const ENV_TEST_INDEX_FAILPOINT_HOLD_MS: &str = "GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS";

#[derive(Debug, Clone, Copy)]
struct EmbeddingCliArgs<'a> {
    enabled: bool,
    model_dir: Option<&'a str>,
    gguf: Option<&'a str>,
    tokenizer: Option<&'a str>,
    model_id: Option<&'a str>,
    max_length: Option<usize>,
    device: Option<&'a str>,
    no_gpu: bool,
}

impl EmbeddingCliArgs<'_> {
    fn has_model_source_arg(&self) -> bool {
        self.model_dir.is_some() || self.gguf.is_some() || self.tokenizer.is_some()
    }
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
    SafetensorsDir(std::path::PathBuf),
    Gguf {
        gguf: std::path::PathBuf,
        tokenizer: std::path::PathBuf,
    },
}

#[derive(Debug, Parser)]
#[command(
    name = "grep",
    bin_name = "grep",
    version,
    about = "A drop-in grep that also answers code-structure questions over an indexed codebase (who-calls / callees / impact / semantic-search).",
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
    /// Drop-in replacement for grep.
    #[command(external_subcommand)]
    Passthrough(Vec<String>),
    /// Index a repository.
    Index {
        /// Path to the repository root (default: cwd).
        path: Option<String>,
        /// Build EmbeddingGemma code-span vectors into the published snapshot.
        #[arg(long)]
        embeddings: bool,
        /// Safetensors model directory containing config.json, tokenizer.json and model.safetensors.
        #[arg(long)]
        embedding_model_dir: Option<String>,
        /// Q4 GGUF model file.
        #[arg(long)]
        embedding_gguf: Option<String>,
        /// Tokenizer JSON for --embedding-gguf.
        #[arg(long)]
        embedding_tokenizer: Option<String>,
        /// Logical model id persisted with vector rows.
        #[arg(long)]
        embedding_model_id: Option<String>,
        /// Optional tokenizer/model truncation length.
        #[arg(long)]
        embedding_max_length: Option<usize>,
        /// Embedding backend: auto, cpu, metal, or cuda.
        #[arg(long, value_name = "auto|cpu|metal|cuda")]
        device: Option<String>,
        /// Force CPU embedding inference.
        #[arg(long)]
        no_gpu: bool,
        /// With path `status`, emit machine-readable status JSON.
        #[arg(long)]
        json: bool,
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
        symbol: Option<String>,
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
        symbol: Option<String>,
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
        /// Accepted for agent ergonomics — no-op.
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
    /// Who calls `S` — incoming CALLS edges (the callers of `S`),
    /// printed as `qualified_name file:line`. With `--code`, also prints
    /// each caller's source span so the agent reads the body without a
    /// separate file Read.
    WhoCalls {
        symbol: Option<String>,
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
        symbol: Option<String>,
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
    /// Where `S` is referenced — all incoming references, printed as
    /// `KIND qualified_name file:line`. With `--code`, also prints each
    /// referencing node's source span.
    FindUsages {
        symbol: Option<String>,
        /// Also print the source code span of each result node.
        #[arg(long)]
        code: bool,
        /// Print every usage site (lift the default NAV_LIMIT cap).
        #[arg(long)]
        all: bool,
        /// Emit machine-readable JSON with exact count metadata.
        #[arg(long)]
        json: bool,
    },
    /// Every incoming graph reference to `S` across calls, usages, type refs,
    /// and imports. This is broader than `find-usages`: it answers "who
    /// depends on S?" without mixing in content-search fallback noise.
    References {
        symbol: Option<String>,
        /// Also print the source code span of each referencing node.
        #[arg(long)]
        code: bool,
        /// Print every reference site (lift the default NAV_LIMIT cap).
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
        /// Maximum result rows to print.
        #[arg(long, default_value_t = 20)]
        limit: usize,
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
        /// Maximum result rows to print.
        #[arg(long, default_value_t = 20)]
        limit: usize,
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
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        code: bool,
        /// Accepted for agent ergonomics — no-op.
        #[arg(long)]
        all: bool,
    },
    /// Symbol-only search (search-symbols alias).
    SearchSymbols {
        query: Option<String>,
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
        /// Add EmbeddingGemma vector hits from current-generation indexed code spans.
        #[arg(long)]
        vectors: bool,
        /// Safetensors model directory containing config.json, tokenizer.json and model.safetensors.
        #[arg(long)]
        embedding_model_dir: Option<String>,
        /// Q4 GGUF model file.
        #[arg(long)]
        embedding_gguf: Option<String>,
        /// Tokenizer JSON for --embedding-gguf.
        #[arg(long)]
        embedding_tokenizer: Option<String>,
        /// Logical model id used to select indexed vectors.
        #[arg(long)]
        embedding_model_id: Option<String>,
        /// Override model max sequence length.
        #[arg(long)]
        embedding_max_length: Option<usize>,
        /// Embedding backend: auto, cpu, metal, or cuda.
        #[arg(long, value_name = "auto|cpu|metal|cuda")]
        device: Option<String>,
        /// Force CPU embedding inference.
        #[arg(long)]
        no_gpu: bool,
    },
    /// Semantic query. Default is algorithmic lexical semantic
    /// ranking; `--vectors` uses EmbeddingGemma vector search over indexed
    /// code-span embeddings.
    #[command(name = "semantic-search", alias = "semantic")]
    Semantic {
        query: Option<String>,
        /// Use EmbeddingGemma vector search over indexed code-span embeddings.
        #[arg(long)]
        vectors: bool,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Safetensors model directory containing config.json, tokenizer.json and model.safetensors.
        #[arg(long)]
        embedding_model_dir: Option<String>,
        /// Q4 GGUF model file.
        #[arg(long)]
        embedding_gguf: Option<String>,
        /// Tokenizer JSON for --embedding-gguf.
        #[arg(long)]
        embedding_tokenizer: Option<String>,
        /// Logical model id used to select vector rows.
        #[arg(long)]
        embedding_model_id: Option<String>,
        /// Optional tokenizer/model truncation length.
        #[arg(long)]
        embedding_max_length: Option<usize>,
        /// Embedding backend: auto, cpu, metal, or cuda.
        #[arg(long, value_name = "auto|cpu|metal|cuda")]
        device: Option<String>,
        /// Force CPU embedding inference.
        #[arg(long)]
        no_gpu: bool,
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
        /// Safetensors model directory containing config.json, tokenizer.json and model.safetensors.
        #[arg(long)]
        embedding_model_dir: Option<String>,
        /// Q4 GGUF model file.
        #[arg(long)]
        embedding_gguf: Option<String>,
        /// Tokenizer JSON for --embedding-gguf.
        #[arg(long)]
        embedding_tokenizer: Option<String>,
        /// Logical model id used to select indexed vectors.
        #[arg(long)]
        embedding_model_id: Option<String>,
        /// Optional tokenizer/model truncation length.
        #[arg(long)]
        embedding_max_length: Option<usize>,
        /// Embedding backend: auto, cpu, metal, or cuda.
        #[arg(long, value_name = "auto|cpu|metal|cuda")]
        device: Option<String>,
        /// Force CPU embedding inference.
        #[arg(long)]
        no_gpu: bool,
    },
    /// Agent installer — out of scope.
    Install {
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Agent uninstaller — out of scope.
    Uninstall {
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Agent updater — out of scope.
    Update {
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Runtime config — out of scope.
    Config { subcmd: Option<String> },
    /// Internal: warm embedding daemon (spawned automatically by query
    /// commands; lazy-loads the model, drops it after an idle TTL to free
    /// GPU memory, exits after a longer idle TTL). Not part of the public
    /// surface.
    #[cfg(unix)]
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
        #[arg(long, default_value = "auto")]
        device: String,
        /// Load the model immediately at startup (session prewarm) instead
        /// of on the first request.
        #[arg(long)]
        prewarm: bool,
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
    "stats",
    "diagnostics",
    "doctor",
    "search-graph",
    "trace",
    "impact",
    "brief",
    "expand",
    "who-calls",
    "callees",
    "find-usages",
    "references",
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
    "config",
    "embed-daemon",
];

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
    // Feature B: probabilistically evict stale index stores on binary
    // start (same throttle pattern as the sidecar cleanup). Best-effort
    // and throttled to ~once per 10 min per process, so a tight loop of
    // invocations does not repeatedly walk the cache dir. The `--root`
    // of THIS invocation (peeked from argv before clap runs) is what the
    // eviction must protect — protecting only the cwd store while the
    // command operates on `--root elsewhere` evicted the very store the
    // invocation was about to serve from.
    maybe_run_store_cleanup(peek_root_arg(&argv).as_deref());
    if is_grep_passthrough(&argv) {
        // argv[0] is the binary name; the rest are grep args. Build a
        // synthetic argv for the shared runner whose argv[0] is a
        // placeholder and argv[1..] are the user's (possibly non-UTF-8)
        // arguments, forwarded verbatim.
        let mut full: Vec<std::ffi::OsString> = Vec::with_capacity(argv.len());
        full.push(std::ffi::OsString::from("greppy-grep"));
        full.extend(argv.into_iter().skip(1));
        return match dispatch_grep_os(&full) {
            Ok(code) => code.clamp(0, 255) as u8,
            Err(Error::Invalid(_)) => EXIT_USAGE,
            Err(_) => EXIT_IO,
        };
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
            eprintln!("{first}");
            let sub = argv.get(1).and_then(|s| s.to_str()).unwrap_or("");
            if let Some(usage) = subcommand_usage(sub) {
                eprintln!("usage: {usage}");
            } else {
                eprintln!(
                    "usage: greppy <command> --help  (commands: index, who-calls, callees, \
                     find-usages, impact, brief, semantic-search, search-code, search-symbols, \
                     path, index status)"
                );
            }
            return EXIT_USAGE;
        }
    };
    dispatch_to_code(cli)
}

/// One-line usage per agent-facing subcommand, printed after a short arg
/// error so the failed call carries the correct retry (P3: every failure
/// costs the agent a turn of thinking plus a tool call).
fn subcommand_usage(sub: &str) -> Option<&'static str> {
    Some(match sub {
        "who-calls" => "greppy who-calls SYMBOL [--code|--json] [--all] [--root DIR]",
        "callees" => "greppy callees SYMBOL [--code|--json] [--all] [--root DIR]",
        "find-usages" | "references" => {
            "greppy find-usages SYMBOL [--code|--json] [--all] [--root DIR]"
        }
        "impact" => {
            "greppy impact SYMBOL [--direction incoming|outgoing] [--depth N] [--json] [--root DIR]"
        }
        "brief" => "greppy brief SYMBOL [--root DIR]",
        "expand" => "greppy expand ID [--json] [--root DIR]",
        "semantic-search" | "semantic" => "greppy semantic-search \"QUERY\" [--root DIR]",
        "context" => "greppy context \"QUERY\" [--root DIR]",
        "search-code" => "greppy search-code QUERY [--json] [--root DIR]",
        "search-symbols" => {
            "greppy search-symbols NAME [--kind function|method|struct|class] [--json] [--root DIR]"
        }
        "path" => "greppy path --from SYMBOL --to SYMBOL [--root DIR]",
        "index" => "greppy index PATH [--embeddings --embedding-gguf F --embedding-tokenizer F]",
        _ => return None,
    })
}

/// Feature B — probabilistically evict stale index stores under the
/// shared `<cache>/greppy/` root. Throttled to ~once per 10 minutes
/// per process (the same pattern as the sidecar `cleanup_expired` call)
/// so a tight loop of `grep`/`greppy` invocations does not repeatedly
/// walk the cache dir. Fully best-effort: any failure is swallowed, and
/// the store currently being used by this invocation is preserved as the
/// `keep` dir so we never delete out from under ourselves.
///
/// TTL comes from `GREPPY_STORE_TTL_DAYS` (default 14 days; `0`
/// disables eviction entirely) — see
/// [`greppy_core::workspace::store_ttl_secs`].
pub fn maybe_run_store_cleanup(root: Option<&str>) {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static LAST_RUN: Mutex<Option<Instant>> = Mutex::new(None);
    const MIN_GAP: Duration = Duration::from_secs(10 * 60);

    let should_run = {
        let mut guard = LAST_RUN.lock().unwrap_or_else(|e| e.into_inner());
        match *guard {
            Some(t) if t.elapsed() < MIN_GAP => false,
            _ => {
                *guard = Some(Instant::now());
                true
            }
        }
    };
    if !should_run {
        return;
    }
    let ttl = workspace_locator::store_ttl_secs();
    if ttl == 0 {
        return; // eviction disabled
    }
    let cache_root = match workspace_locator::store_cache_root() {
        Some(c) => c,
        None => return,
    };
    // Preserve the store this invocation is about to use — resolved from
    // the invocation's own `--root` when given, else from cwd (best-effort:
    // if we cannot resolve a root, pass a path that matches nothing).
    let keep = resolve_root(root)
        .map(|r| workspace_locator::store_dir(&r))
        .unwrap_or_else(|_| cache_root.join("\0none"));
    let _ = workspace_locator::cleanup_stale_stores(&cache_root, ttl, &keep);
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

/// Decide whether `argv` (including argv[0]) is a bare `grep`
/// passthrough rather than a recognised structured subcommand.
///
/// We skip a leading global `--root <val>` / `--root=<val>` and any
/// `--help`/`-h`/`--version`/`-V` (which clap handles), then look at the
/// first remaining token:
/// * If it equals a recognised subcommand name → NOT a passthrough.
/// * Otherwise (a flag like `-R`, a pattern, or nothing) → passthrough.
fn is_grep_passthrough(argv: &[std::ffi::OsString]) -> bool {
    let mut i = 1; // skip argv[0]
    while i < argv.len() {
        let tok = &argv[i];
        // Help/version requests must reach clap so it prints them.
        if tok == "--help" || tok == "-h" || tok == "--version" || tok == "-V" {
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
    if let Some(cmd) = cli.command {
        return dispatch_subcommand(cmd, root.as_deref());
    }
    if !cli.passthrough.is_empty() {
        return dispatch_grep(&cli.passthrough);
    }
    // No subcommand and no pattern: a usage MISTAKE (often an agent's).
    // Print a compact cheat sheet, not the 2.5KB curated help — mid-task
    // token bombs teach nothing (P3). `--help` still prints everything.
    println!("usage: grep PATTERN [FILES..]           (drop-in grep)");
    println!("   or: grep <command> [--root DIR]      commands:");
    println!("       index PATH [--embeddings]  who-calls S   callees S   find-usages S");
    println!("       references S (who depends on S)   impact S [--direction incoming|outgoing]");
    println!("       brief S   semantic-search \"QUERY\"");
    println!("       search-code Q   search-symbols NAME [--kind function|method|struct|class]");
    println!("       index status   (--help for full details)");
    Ok(EXIT_USAGE as i32)
}

fn dispatch_subcommand(cmd: Command, root: Option<&str>) -> Result<i32> {
    match cmd {
        Command::Passthrough(argv) => dispatch_grep(&argv),
        #[cfg(unix)]
        Command::EmbedDaemon {
            socket,
            gguf,
            tokenizer,
            model_id,
            max_length,
            device,
            prewarm,
        } => {
            let cfg = EmbeddingModelConfig {
                model_id,
                source: EmbeddingModelSource::Gguf {
                    gguf: std::path::PathBuf::from(gguf),
                    tokenizer: std::path::PathBuf::from(tokenizer),
                },
                max_length,
                device: greppy_embed_native::DevicePreference::parse(&device)
                    .map_err(|e| Error::Invalid(format!("embed-daemon --device: {e}")))?,
            };
            embed_daemon::daemon_main(std::path::PathBuf::from(socket), cfg, prewarm)
        }
        Command::Index {
            path,
            embeddings,
            embedding_model_dir,
            embedding_gguf,
            embedding_tokenizer,
            embedding_model_id,
            embedding_max_length,
            device,
            no_gpu,
            json,
        } => {
            if path.as_deref() == Some("status") {
                if embeddings
                    || embedding_model_dir.is_some()
                    || embedding_gguf.is_some()
                    || embedding_tokenizer.is_some()
                    || embedding_model_id.is_some()
                    || embedding_max_length.is_some()
                    || device.is_some()
                    || no_gpu
                {
                    return Err(Error::Invalid(
                        "index status does not accept embedding index flags".into(),
                    ));
                }
                dispatch_index_status(json, root)
            } else {
                if json {
                    return Err(Error::Invalid(
                        "index --json is only supported for `grep index status --json`".into(),
                    ));
                }
                dispatch_index(
                    path.as_deref(),
                    root,
                    EmbeddingCliArgs {
                        enabled: embeddings,
                        model_dir: embedding_model_dir.as_deref(),
                        gguf: embedding_gguf.as_deref(),
                        tokenizer: embedding_tokenizer.as_deref(),
                        model_id: embedding_model_id.as_deref(),
                        max_length: embedding_max_length,
                        device: device.as_deref(),
                        no_gpu,
                    },
                )
            }
        }
        Command::SearchGraph { name, json } => {
            let mut q = greppy_search::GraphQuery::any().with_limit(50);
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
            symbol,
            code: _,
            direction,
            edge,
            depth,
            since,
            base,
            all,
            json,
        } => dispatch_impact(
            symbol.as_deref(),
            &direction,
            edge.as_deref(),
            depth,
            since.as_deref(),
            base.as_deref(),
            all,
            json,
            root,
        ),
        Command::Brief {
            symbol,
            code: _,
            all: _,
            json: _,
        } => dispatch_brief(symbol.as_deref(), root),
        Command::Expand { id, json } => dispatch_expand(id.as_deref(), json, root),
        Command::Stats => dispatch_stats(root),
        Command::Diagnostics { json } => dispatch_diagnostics(json, root),
        Command::Doctor { json } => dispatch_doctor(json, root),
        Command::WhoCalls {
            symbol,
            code,
            all,
            json,
        } => dispatch_who_calls(symbol.as_deref(), code, all, json, root),
        Command::Callees {
            symbol,
            code,
            all,
            json,
        } => dispatch_callees(symbol.as_deref(), code, all, json, root),
        Command::FindUsages {
            symbol,
            code,
            all,
            json,
        } => dispatch_find_usages(symbol.as_deref(), code, all, json, root),
        Command::References {
            symbol,
            code,
            all,
            json,
        } => dispatch_references(symbol.as_deref(), code, all, json, root),
        Command::FanIn { edge, limit, json } => {
            dispatch_fan_degree("fan-in", "incoming", &edge, limit, json, root)
        }
        Command::FanOut { edge, limit, json } => {
            dispatch_fan_degree("fan-out", "outgoing", &edge, limit, json, root)
        }
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
            changed,
            staged,
            since,
            base,
            json,
            code: _,
            all: _,
        } => dispatch_search_code(
            query.as_deref(),
            changed,
            staged,
            since.as_deref(),
            base.as_deref(),
            json,
            root,
        ),
        Command::SearchSymbols {
            query,
            kind,
            json,
            code: _,
            all: _,
        } => dispatch_search_symbols(query.as_deref(), kind.as_deref(), json, root),
        Command::Plus {
            query,
            k,
            code,
            explain,
            json,
            vectors,
            embedding_model_dir,
            embedding_gguf,
            embedding_tokenizer,
            embedding_model_id,
            embedding_max_length,
            device,
            no_gpu,
        } => dispatch_plus(
            query.as_deref(),
            k,
            code,
            explain,
            json,
            vectors,
            EmbeddingCliArgs {
                enabled: vectors,
                model_dir: embedding_model_dir.as_deref(),
                gguf: embedding_gguf.as_deref(),
                tokenizer: embedding_tokenizer.as_deref(),
                model_id: embedding_model_id.as_deref(),
                max_length: embedding_max_length,
                device: device.as_deref(),
                no_gpu,
            },
            root,
        ),
        Command::Semantic {
            query,
            vectors,
            json,
            embedding_model_dir,
            embedding_gguf,
            embedding_tokenizer,
            embedding_model_id,
            embedding_max_length,
            device,
            no_gpu,
        } => dispatch_semantic(
            query.as_deref(),
            vectors,
            json,
            EmbeddingCliArgs {
                enabled: vectors,
                model_dir: embedding_model_dir.as_deref(),
                gguf: embedding_gguf.as_deref(),
                tokenizer: embedding_tokenizer.as_deref(),
                model_id: embedding_model_id.as_deref(),
                max_length: embedding_max_length,
                device: device.as_deref(),
                no_gpu,
            },
            root,
        ),
        Command::Context {
            query,
            k,
            lines,
            json,
            code: _,
            all: _,
            embedding_model_dir,
            embedding_gguf,
            embedding_tokenizer,
            embedding_model_id,
            embedding_max_length,
            device,
            no_gpu,
        } => dispatch_context(
            query.as_deref(),
            k,
            lines,
            json,
            EmbeddingCliArgs {
                enabled: true,
                model_dir: embedding_model_dir.as_deref(),
                gguf: embedding_gguf.as_deref(),
                tokenizer: embedding_tokenizer.as_deref(),
                model_id: embedding_model_id.as_deref(),
                max_length: embedding_max_length,
                device: device.as_deref(),
                no_gpu,
            },
            root,
        ),
        Command::Install { .. } => Err(Error::out_of_scope("grep install")),
        Command::Uninstall { .. } => Err(Error::out_of_scope("grep uninstall")),
        Command::Update { .. } => Err(Error::out_of_scope("grep update")),
        Command::Config { .. } => Err(Error::out_of_scope("grep config")),
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
/// is shared — e.g. `find-usages Store` resolved to the `EnumVariant`
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
/// (so a `Struct` and its `Impl` both contribute to find-usages /
/// who-calls). See [`resolve_symbol_nodes`].
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
            r.name == member
                && is_primary_label(&r.label)
                && qname_owner_segment(&r.qualified_name) == Some(owner_tail)
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
                .filter(|r| r.name.eq_ignore_ascii_case(s))
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
///   * bare name → all nodes with that exact `name`;
///   * qualified `Owner.member` → all nodes named `member` (the owner is
///     matched in [`resolve_qualified_ids`]);
///   * no symbol → the first node in qualified_name order (the historical
///     no-arg `trace` seed).
fn symbol_candidate_rows(
    store: &greppy_store::Store,
    symbol: Option<&str>,
) -> Result<Vec<greppy_search::graph::SearchGraphRow>> {
    let q = match symbol {
        Some(s) => {
            let lookup_name = split_qualified(s).map(|(_, member)| member).unwrap_or(s);
            greppy_search::GraphQuery::any()
                .with_name(lookup_name)
                .with_limit(10_000)
        }
        None => greppy_search::GraphQuery::any().with_limit(1),
    };
    let rows = greppy_search::search_graph(store, &q)?;
    if !rows.is_empty() {
        return Ok(rows);
    }
    // P3 (agent ergonomics): agents guess casing (`Coerce` vs `coerce`).
    // When the exact name misses, accept a case-variant IF it is
    // unambiguous — every case-insensitive match shares one spelling.
    // Multiple distinct spellings stay unresolved (never-guess), and the
    // not-found path then lists them as suggestions.
    if let Some(s) = symbol {
        let lookup_name = split_qualified(s).map(|(_, member)| member).unwrap_or(s);
        let project = store
            .list_projects()
            .ok()
            .and_then(|ps| ps.into_iter().next().map(|p| p.name));
        if let Some(project) = project {
            if let Ok(similar) = store.similar_node_names(&project, lookup_name, 10) {
                let exact_ci: Vec<&String> = similar
                    .iter()
                    .filter(|n| n.eq_ignore_ascii_case(lookup_name))
                    .collect();
                if exact_ci.len() == 1 {
                    let q = greppy_search::GraphQuery::any()
                        .with_name(exact_ci[0].clone())
                        .with_limit(10_000);
                    return greppy_search::search_graph(store, &q);
                }
            }
        }
    }
    Ok(rows)
}

/// Resolve a symbol to the set of node ids whose incoming edges should
/// be aggregated for who-calls / find-usages / trace-incoming.
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
fn resolve_symbol_nodes(store: &greppy_store::Store, symbol: Option<&str>) -> Result<Vec<i64>> {
    let Some(s) = symbol else {
        // No symbol: mirror resolve_symbol_id's "first node" behaviour.
        return Ok(resolve_symbol_id(store, None)?.into_iter().collect());
    };
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
        .filter(|r| r.name.eq_ignore_ascii_case(s) && is_primary_label(&r.label))
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
/// find-usages / trace). Tighter than `context` because these commands
/// can emit many nodes.
const CODE_SPAN_CAP: usize = 25;

/// Default cap on the number of result rows printed by the navigation
/// commands (who-calls / callees / find-usages). Forensics finding F1: a
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

/// Freshness budget for explicit machine-readable navigation queries. The
/// grep drop-in hotpath uses 200 ms; explicit graph JSON can afford a little
/// more so it can prove freshness on normal debug/test builds instead of
/// reporting false `budget exceeded` staleness.
const NAV_FRESHNESS_BUDGET: std::time::Duration = std::time::Duration::from_millis(1_000);

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

/// Per-target-node incoming-edge LIMIT used by who-calls / find-usages
/// (`incoming_edges(.., 1024)`). When a single target hits this cap the
/// aggregated `total` is a FLOOR, not an exact count, so the completeness
/// footer must never claim "complete" at or above it (H2 / D1).
const NAV_EDGE_LIMIT: usize = 1024;

/// Pluralize a singular edge-class noun ("caller"/"callee"/"usage") by count:
/// `1` keeps the singular, any other count appends `s`. Keeps the H2 footer
/// grammatical ("1 caller (complete)" vs "6 callers found …").
fn pluralize_count<'a>(singular: &'a str, n: usize) -> std::borrow::Cow<'a, str> {
    if n == 1 {
        std::borrow::Cow::Borrowed(singular)
    } else {
        std::borrow::Cow::Owned(format!("{singular}s"))
    }
}

/// Completeness state of a navigation result, driving the unconditional
/// footer the text nav commands (who-calls / callees / find-usages) print.
/// The JSON path already surfaces every field this encodes
/// (`provider_complete`, `truncated`, `total_exact`); this only mirrors it
/// into the human footer so low-count answers (1–2 rows) still carry an
/// explicit count + completeness marker and the agent does not re-iterate to
/// manufacture the number.
struct NavFooter<'a> {
    /// Singular noun for the edge class: "caller" / "callee" / "usage".
    /// Pluralized by count so "1 caller" / "6 callers" both read naturally.
    noun: &'a str,
    /// True unique total across the aggregated target nodes.
    total: usize,
    /// Rows actually printed (capped unless `--all`).
    shown: usize,
    /// True when the provider for the TARGET language is known-incomplete for
    /// this project (partial / parity-candidate / unsupported code provider),
    /// so the call-graph recall may undercount real edges.
    provider_incomplete: bool,
}

impl NavFooter<'_> {
    /// Render the H2 unconditional stop-signal footer to a string.
    ///
    /// Honesty (D1): the word "complete" appears ONLY when the provider is
    /// complete AND the result is not truncated AND the total is a true count
    /// (strictly below the per-node [`NAV_EDGE_LIMIT`] floor). Otherwise the
    /// footer hedges (provider-partial note) or reports the truncation total.
    /// Pure (reads only its fields + the passed `stale` flag) so it is unit
    /// testable without capturing stdout.
    fn render(&self, stale: bool) -> String {
        let stale_note = if stale { " (as of last index)" } else { "" };
        let truncated = self.total > self.shown;
        // A total at/above the per-node edge LIMIT is a floor, never exact,
        // so "complete" is dishonest even when nothing was truncated for print.
        let total_is_exact = self.total < NAV_EDGE_LIMIT;
        let noun = pluralize_count(self.noun, self.total);

        if truncated {
            // Existing >NAV_LIMIT case: report the true total and the escape
            // hatch. Kept byte-for-byte compatible with the prior footer.
            return format!(
                "… and {} more ({} shown of {} total{stale_note} — this sample usually answers the question; pass --all only if you truly need every site)",
                self.total - self.shown,
                self.shown,
                self.total
            );
        }
        // Stop-signal = the COUNT (what the agent over-iterated to manufacture),
        // kept MINIMAL. P2-iterC showed a verbose honest hedge (~22 tokens of
        // "recall is partial … may undercount") DOUBLED low-count outputs and
        // even triggered extra queries — pure overhead once qualified-name
        // already collapsed graph-nav to one round. So: bare count, and a
        // 1-char `+` (never "complete") when the count is a floor — honest
        // (may be more) and cheap.
        let floor = self.provider_incomplete || !total_is_exact;
        let plus = if floor { "+" } else { "" };
        format!("— {}{plus} {noun}{stale_note}", self.total)
    }

    /// Print the H2 unconditional stop-signal footer.
    fn print(&self) {
        println!("{}", self.render(serving_stale()));
    }
}

/// Map a file path's extension to the provider `language` name used in
/// `provider_state` (matching `greppy_parser::Language::name`). Returns the
/// display name for a supported code language, or `None` for extensions that
/// map to an unsupported/non-code provider. Kept in sync with
/// `crates/parser/src/language.rs::language_for_path`; only the supported set
/// matters here because the footer only hedges on a code language's provider.
fn code_language_for_ext(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())?;
    Some(match ext {
        "rs" => "rust",
        "py" => "python",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "ts" => "typescript",
        "tsx" => "tsx",
        "go" => "go",
        "rb" => "ruby",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" | "hh" => "cpp",
        "cs" => "c-sharp",
        "php" => "php",
        "sh" | "bash" => "bash",
        "lua" => "lua",
        "kt" | "kts" => "kotlin",
        "scala" | "sc" => "scala",
        "swift" => "swift",
        "zig" => "zig",
        "r" | "R" => "r",
        _ => return None,
    })
}

/// Decide whether the target symbol's language provider is known-incomplete,
/// and its display name, for the H2 completeness footer. Reads the SAME
/// incomplete-provider set the JSON path emits (`incomplete_provider_json`) so
/// the footer and `provider_complete` never disagree. The target language is
/// derived from the result nodes' file extensions (all rows into/out of a
/// symbol share its language); the first row whose *code* language provider is
/// incomplete wins the hedge.
fn nav_target_provider_incomplete(
    store: &greppy_store::Store,
    project: &str,
    rows: &[&greppy_store::Node],
    edge_class: &str,
) -> Result<(bool, &'static str)> {
    // Languages whose provider does NOT emit `edge_class` for this project.
    // The hedge is scoped to the QUERIED edge class, not the provider's overall
    // completeness (H2 fix): a provider that supports CALLS but omits exotic
    // classes (k8s, gitdiff, …) is complete FOR A who-calls query, so its
    // footer must report an exact count with no `+` floor marker — otherwise
    // the marker triggers a redundant `--all` re-query + grep fallback. We only
    // consider CODE languages: an unsupported non-code file type (.stderr,
    // .snap, …) has no call edges to miss, so it must not trigger a hedge.
    let lacking: std::collections::BTreeSet<String> = store
        .list_provider_states(project)?
        .into_iter()
        .filter(|p| !p.supports_edge_class(edge_class))
        .map(|p| p.language)
        .collect();
    for node in rows {
        if let Some(lang) = code_language_for_ext(&node.file_path) {
            if lacking.contains(lang) {
                return Ok((true, lang));
            }
        }
    }
    Ok((false, ""))
}

/// Same completeness decision as [`nav_target_provider_incomplete`], but keyed
/// on the resolved TARGET node ids — used by the zero-result footer branches,
/// where there are no result rows to read a language from, so the hedge must
/// come from the queried symbol's own language.
fn nav_target_ids_provider_incomplete(
    store: &greppy_store::Store,
    project: &str,
    target_ids: &[i64],
    edge_class: &str,
) -> Result<(bool, &'static str)> {
    let mut nodes = Vec::new();
    for id in target_ids {
        if let Some(n) = store.get_node(*id)? {
            nodes.push(n);
        }
    }
    let refs: Vec<&greppy_store::Node> = nodes.iter().collect();
    nav_target_provider_incomplete(store, project, &refs, edge_class)
}

/// Render the H2 completeness footer for a ZERO-result navigation answer:
/// `— 0 <noun> (complete)` when the target language's provider is complete, or
/// `— 0 <noun> found (<lang> recall partial)` when it is known-incomplete.
/// Pure so it is unit testable.
fn render_zero_nav_footer(
    noun: &str,
    provider_incomplete: bool,
    lang: &str,
    stale: bool,
) -> String {
    let stale_note = if stale { " (as of last index)" } else { "" };
    let noun = pluralize_count(noun, 0);
    let _ = lang;
    // Minimal (see NavFooter::render): a 0-count is a definite stop signal; the
    // `+` marks a floor when the provider is known-incomplete (0 found, may be
    // more), else a bare exact 0.
    let plus = if provider_incomplete { "+" } else { "" };
    format!("— 0{plus} {noun}{stale_note}")
}

/// Print the H2 completeness footer for a ZERO-result navigation answer.
fn print_zero_nav_footer(
    store: &greppy_store::Store,
    project: &str,
    noun: &str,
    target_ids: &[i64],
    edge_class: &str,
) -> Result<()> {
    let (provider_incomplete, lang) =
        nav_target_ids_provider_incomplete(store, project, target_ids, edge_class)?;
    println!(
        "{}",
        render_zero_nav_footer(noun, provider_incomplete, lang, serving_stale())
    );
    Ok(())
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
/// exist than were printed. Centralised so who-calls / callees /
/// find-usages word it identically.
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
}

fn expand_ttl_secs() -> u64 {
    std::env::var(ENV_EXPAND_TTL_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(greppy_store::DEFAULT_EXPAND_TTL_SECS)
}

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

fn insert_semantic_algorithmic_expand_pack(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    query: &str,
    hits: &[greppy_search::SemanticHit],
) -> Option<ExpandHandle> {
    if hits.is_empty() {
        return None;
    }
    let root_path = resolve_root(root).ok()?;
    let limit = hits.len().min(6);
    let mut text = String::new();
    text.push_str(&format!("# evidence pack: semantic-search {query}\n"));
    text.push_str(&format!(
        "# spans: {limit} shown of {} hits\n\n",
        hits.len()
    ));
    let mut json_rows = Vec::new();
    for (idx, hit) in hits.iter().take(limit).enumerate() {
        let title = format!(
            "{:.3} {} {}",
            hit.score,
            hit.node.label,
            display_row_name(&hit.node)
        );
        append_span_evidence(
            &mut text,
            &root_path,
            &title,
            &hit.node.file_path,
            hit.node.start_line,
            hit.node.end_line,
            if idx == 0 {
                CONTEXT_SPAN_CAP
            } else {
                CODE_SPAN_CAP
            },
        );
        json_rows.push(serde_json::json!({
            "score": hit.score,
            "label": &hit.node.label,
            "qualified_name": &hit.node.qualified_name,
            "file_path": &hit.node.file_path,
            "start_line": hit.node.start_line,
            "end_line": hit.node.end_line,
        }));
    }
    let summary = serde_json::json!({
        "text": format!("{limit} spans"),
        "spans": limit,
        "callsites": 0,
        "total": hits.len(),
    });
    let payload_json = serde_json::json!({
        "command": "semantic-search",
        "mode": "algorithmic",
        "query": query,
        "shown": limit,
        "hits": json_rows,
    });
    insert_expand_pack_best_effort(
        store,
        project,
        "semantic-search",
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
    let limit = hits.len().min(6);
    let mut text = String::new();
    text.push_str(&format!(
        "# evidence pack: semantic-search --vectors {query}\n"
    ));
    text.push_str(&format!(
        "# spans: {limit} shown of {} hits\n\n",
        hits.len()
    ));
    let mut json_rows = Vec::new();
    for (idx, hit) in hits.iter().take(limit).enumerate() {
        let title = format!("{:.3} {}", hit.score, hit.embedding.qualified_name);
        append_span_evidence(
            &mut text,
            &root_path,
            &title,
            &hit.embedding.file_path,
            hit.embedding.start_line,
            hit.embedding.end_line,
            if idx == 0 {
                CONTEXT_SPAN_CAP
            } else {
                CODE_SPAN_CAP
            },
        );
        json_rows.push(serde_json::json!({
            "score": hit.score,
            "qualified_name": &hit.embedding.qualified_name,
            "file_path": &hit.embedding.file_path,
            "start_line": hit.embedding.start_line,
            "end_line": hit.embedding.end_line,
            "content_sha256": &hit.embedding.content_sha256,
            "graph_generation": hit.embedding.graph_generation,
        }));
    }
    let summary = serde_json::json!({
        "text": format!("{limit} spans"),
        "spans": limit,
        "callsites": 0,
        "total": hits.len(),
    });
    let payload_json = serde_json::json!({
        "command": "semantic-search",
        "mode": "vector",
        "query": query,
        "shown": limit,
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
    let mut v = serde_json::json!({
        "command": command,
        "symbol": symbol,
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
        .map(|p| {
            serde_json::json!({
                "language": p.language,
                "status": p.status,
                "unsupported_edge_classes": p.unsupported_edge_classes,
                "files_seen": p.files_seen,
                "files_indexed": p.files_indexed,
                "files_failed": p.files_failed,
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
/// partial — the r061 28-round reconciliation blowup. `impact` therefore
/// filters these out and reports only real code providers.
fn is_noncode_provider(status: &str, language: &str) -> bool {
    status == "unsupported"
        || language.starts_with("file extension .")
        || language == "no file extension"
}

/// Incomplete providers for `impact`, excluding non-code snapshot/fixture
/// providers (see [`is_noncode_provider`]) so the reported
/// `incomplete_provider_count` / `provider_complete` reflects only real code
/// callers, not `.stderr` / `.snap` files.
fn code_incomplete_provider_json(
    store: &greppy_store::Store,
    project: &str,
) -> Result<Vec<serde_json::Value>> {
    Ok(store
        .list_provider_states(project)?
        .into_iter()
        .filter(greppy_store::ProviderState::is_incomplete)
        .filter(|p| !is_noncode_provider(&p.status, &p.language))
        .map(|p| {
            serde_json::json!({
                "language": p.language,
                "status": p.status,
                "unsupported_edge_classes": p.unsupported_edge_classes,
                "files_seen": p.files_seen,
                "files_indexed": p.files_indexed,
                "files_failed": p.files_failed,
            })
        })
        .collect())
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

/// D2 fail-open gate for the graph navigation commands. Returns
/// `Some(exit_code)` ONLY when there is no usable index (see
/// [`FreshnessServe::Refuse`]); a merely-stale index proceeds with
/// labeled results (auto-healed first when the drift is small).
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
        FreshnessServe::Fresh(_) | FreshnessServe::StaleLabeled(_) => Ok(None),
        FreshnessServe::Refuse(freshness) => {
            if json {
                graph_stale_skip_json(
                    store,
                    root,
                    project,
                    command,
                    freshness,
                    extra,
                    empty_collection_field,
                )?;
            } else {
                println!("{}", indexed_stale_skip_message(command, &freshness));
            }
            Ok(Some(1))
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
    let mut v = serde_json::json!({
        "command": "impact",
        "symbol": symbol,
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

fn trace_counts_json(
    store: &greppy_store::Store,
    root: Option<&str>,
    symbol: &str,
    project: &str,
    symbol_found: bool,
    meta: TraceJsonMeta<'_>,
    steps: &[greppy_search::TraceStep],
) -> Result<()> {
    let freshness = nav_freshness_json(store, root, project);
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let step_json: Vec<_> = steps.iter().map(trace_step_json).collect();
    let step_count = step_json.len();
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
        "total_exact": step_count,
        "shown": step_count,
        "omitted": 0,
        "truncated": false,
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
                greppy_freshness::FreshnessOutcome::RootMismatch => (
                    false,
                    "root_mismatch",
                    vec!["workspace root mismatch".into()],
                ),
                greppy_freshness::FreshnessOutcome::Stale { reasons } => (false, "stale", reasons),
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

/// D2 fail-open freshness policy for indexed query surfaces.
///
/// The old contract failed CLOSED: any non-fresh verdict (a single
/// edited file, a budget overrun) suppressed ALL indexed output and
/// exited 1 — on large repos the whole plus surface self-disabled and
/// agents silently fell back to plain grep. The new contract:
///
/// - `Fresh`: proceed normally.
/// - `StaleLabeled`: the index exists but has drifted. Serve results
///   FROM THE EXISTING INDEX, honestly labeled (`fresh: false` +
///   `freshness` in every JSON payload, a one-line stderr warning, and
///   `(as of last index)` in completeness footers). Before settling for
///   stale, if the drift is small (≤ [`AUTO_REINDEX_MAX_FILES`] changed
///   files) an inline incremental reindex is attempted so the answer is
///   simply fresh.
/// - `Refuse`: there is no usable index for this root (cold store /
///   root mismatch / invalid discover-scope config). Only this case
///   keeps the old fail-closed behaviour (exit 1 + "run greppy
///   index").
enum FreshnessServe {
    Fresh(serde_json::Value),
    StaleLabeled(serde_json::Value),
    Refuse(serde_json::Value),
}

impl FreshnessServe {
    /// The freshness JSON to embed in the command's payload, whatever
    /// the verdict was.
    fn freshness(&self) -> &serde_json::Value {
        match self {
            FreshnessServe::Fresh(f)
            | FreshnessServe::StaleLabeled(f)
            | FreshnessServe::Refuse(f) => f,
        }
    }
}

/// Auto-reindex cap: an inline incremental reindex is only attempted
/// when at most this many files drifted. Above the cap we serve
/// labeled-stale results instead of stalling the query on a large
/// reindex.
const AUTO_REINDEX_MAX_FILES: usize = 10;

/// Kill switch for the inline auto-reindex (`0`/`false` disables). The
/// labeled-stale serving is NOT affected by this switch.
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

/// Set once a command decides to serve labeled-stale results; the
/// completeness footers read it to append `(as of last index)`.
static SERVED_STALE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn serving_stale() -> bool {
    SERVED_STALE.load(std::sync::atomic::Ordering::Relaxed)
}

fn freshness_serve_decision(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
) -> FreshnessServe {
    freshness_serve_decision_with_policy(store, root, project, true, true)
}

/// `allow_auto_reindex = false` for surfaces whose data is
/// generation-scoped (vector search): an inline reindex without
/// embeddings would bump the generation and invalidate the very rows
/// the query needs. `warn_on_stale = false` for callers that do NOT
/// serve stale results on `StaleLabeled` but emit their own controlled
/// skip output instead (again the vector path) — warning about serving
/// stale results that are then not served would be noise.
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

fn freshness_serve_decision_with_policy(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    allow_auto_reindex: bool,
    warn_on_stale: bool,
) -> FreshnessServe {
    let freshness = nav_freshness_json(store, root, project);
    if freshness_json_is_fresh(&freshness) {
        return FreshnessServe::Fresh(freshness);
    }
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    // No usable index at all: refuse (fail-closed). config_error =
    // invalid discover-scope env, a user error that a stale-labeled
    // answer would mask.
    if matches!(state, "cold" | "root_mismatch" | "config_error") {
        return FreshnessServe::Refuse(freshness);
    }
    // An indexer-version/scope mismatch is a CONFIG mismatch, not time
    // staleness: the persisted rows were produced under different
    // extraction semantics or a different discover scope, so "the same
    // data, slightly old" does not describe them. Serving them labeled
    // "stale" would misrepresent; only a real reindex under the current
    // config helps. This deliberately keeps the pre-D2 fail-closed
    // contract (see discover_scope_env_controls_index_and_query_freshness).
    let scope_or_version_drift = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|rs| {
            rs.iter()
                .filter_map(serde_json::Value::as_str)
                .any(|r| r.contains("indexer version/scope"))
        });
    if scope_or_version_drift {
        // Split the two kinds of drift the one reason string covers:
        //  * VERSION bump (same discover scope, different base version): a
        //    binary upgrade changed the extractor (O3/O8/O9/P4). A store
        //    built by the older/buggy binary must NOT keep serving its
        //    wrong data, and a FULL reindex under the SAME scope is exactly
        //    what a fresh `grep index` produces — so self-heal.
        //  * SCOPE change (different discover scope): the user is querying
        //    with a different discover-scope env than the index was built
        //    with. Silently reindexing under the query's scope would change
        //    what the store holds behind the user's back — keep the
        //    fail-closed contract and refuse.
        if allow_auto_reindex
            && auto_reindex_enabled()
            && version_drift_is_scope_stable(&freshness)
            && try_auto_reindex_inline(root)
        {
            let after = nav_freshness_json(store, root, project);
            if freshness_json_is_fresh(&after) {
                return FreshnessServe::Fresh(after);
            }
        }
        return FreshnessServe::Refuse(freshness);
    }

    // Small drift: heal inline (incremental reindex re-extracts only the
    // changed files), then answer fresh. `stale_file_count == 0` covers
    // pure git-fingerprint drift (e.g. a commit of already-indexed
    // content) — the reindex just refreshes workspace_state.
    let stale_file_count = freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);
    if let Some(n) = stale_file_count {
        if allow_auto_reindex
            && n <= AUTO_REINDEX_MAX_FILES
            && auto_reindex_enabled()
            && try_auto_reindex_inline(root)
        {
            let after = nav_freshness_json(store, root, project);
            if freshness_json_is_fresh(&after) {
                return FreshnessServe::Fresh(after);
            }
            // Reindex ran but freshness still disagrees (races, walk
            // errors): fall through to labeled-stale with the post-
            // reindex view.
            if warn_on_stale {
                warn_serving_stale(store, root, &after);
            }
            return FreshnessServe::StaleLabeled(after);
        }
    }

    if warn_on_stale {
        warn_serving_stale(store, root, &freshness);
    }
    FreshnessServe::StaleLabeled(freshness)
}

/// One-line stderr warning for labeled-stale serving. Printed at most
/// once per invocation; stdout (grep-shaped results / JSON) stays
/// untouched.
fn warn_serving_stale(
    store: &greppy_store::Store,
    root: Option<&str>,
    freshness: &serde_json::Value,
) {
    let already = SERVED_STALE.swap(true, std::sync::atomic::Ordering::Relaxed);
    if already {
        return;
    }
    let generation = current_graph_generation(store, root).unwrap_or(0);
    let drift = match freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
    {
        Some(n) => format!("{n} file(s) changed"),
        None => "extent unknown".to_string(),
    };
    eprintln!(
        "grep: index may be stale ({drift}); results from generation {generation} — run 'grep index'"
    );
}

/// Inline incremental reindex of the CURRENT store, in place, holding
/// the writer lock. Returns true when the index was rebuilt cleanly.
///
/// In-place (not the atomic temp-DB snapshot the `index` subcommand
/// uses) is deliberate: the calling query command already holds an open
/// read-only handle to this DB file, and must see the refreshed rows
/// through it. The indexer's incremental path re-extracts only changed
/// files, so with ≤ [`AUTO_REINDEX_MAX_FILES`] drifted files this is
/// bounded work. Any failure (lock contention, read-only store dir,
/// indexer error) simply reports false — the caller degrades to
/// labeled-stale serving, never to a hard failure.
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
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(LockOutcome::Acquired | LockOutcome::AcquiredFromStale) => {
            greppy_freshness::Lock::new(greppy_freshness::lock_path_for(&store_path))
        }
        _ => return false, // another writer is active: serve labeled-stale
    };
    let Ok(mut store) = greppy_store::Store::open(&store_path) else {
        return false;
    };
    // Remember whether this store served code-span vectors BEFORE the
    // reindex bumps the generation: an inline graph-only reindex would
    // otherwise strand every existing vector row on the old generation and
    // silently degrade `context`/`semantic --vectors` until a manual
    // `grep index --embeddings` run (the owner's "gains" path dying quietly).
    let had_vectors = !store
        .vector_model_ids(&project)
        .unwrap_or_default()
        .is_empty();
    let options = greppy_indexer::IndexOptions {
        discover_overrides: overrides,
    };
    match greppy_indexer::index_with_options(&mut store, &effective_root, &project, &options) {
        Ok(report) => {
            let clean = report.is_clean();
            if clean && had_vectors {
                auto_rebuild_vectors_inline(&mut store, &effective_root, &project, &report);
            }
            clean
        }
        Err(_) => false,
    }
}

/// Whether the vector query path may self-heal a stale index via the
/// inline auto-reindex: only when the embedding model is resolvable from
/// env/HF-cache, because only then does [`auto_rebuild_vectors_inline`]
/// rebuild the vectors for the bumped generation.
fn vector_auto_reindex_can_rebuild(args: EmbeddingCliArgs<'_>) -> bool {
    match embedding_config_optional(args) {
        Ok(Some(cfg)) => embedding_model_source_exists(&cfg.source),
        Ok(None) | Err(_) => false,
    }
}

fn embedding_model_source_exists(source: &EmbeddingModelSource) -> bool {
    match source {
        EmbeddingModelSource::SafetensorsDir(dir) => dir.join("model.safetensors").is_file(),
        EmbeddingModelSource::Gguf { gguf, tokenizer } => gguf.is_file() && tokenizer.is_file(),
    }
}

/// Rebuild code-span vectors after an inline auto-reindex of a store that
/// had them. Incremental by design: unchanged spans keep their vectors (the
/// embedding indexer prunes/re-embeds only what changed), so the common
/// agent-loop case (one edited file) costs a handful of spans, not a full
/// re-embed. Model resolution is env/HF-cache only (no CLI flags exist on
/// this path); when the model cannot be resolved we say so ONCE instead of
/// silently stranding the vectors.
fn auto_rebuild_vectors_inline(
    store: &mut greppy_store::Store,
    root: &std::path::Path,
    project: &str,
    report: &greppy_indexer::IndexReport,
) {
    let no_args = EmbeddingCliArgs {
        enabled: false,
        model_dir: None,
        gguf: None,
        tokenizer: None,
        model_id: None,
        max_length: None,
        device: None,
        no_gpu: false,
    };
    let cfg = match embedding_config_optional(no_args) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => {
            eprintln!(
                "grep: reindex left code-span vectors on an old generation \
                 (embedding model not configured); run `grep index --embeddings ...` to restore vector search"
            );
            return;
        }
        Err(e) => {
            log_embedding_skip_once("auto-reindex vectors", &e);
            return;
        }
    };
    let model = match load_embedding_model(&cfg, None) {
        Ok(m) => m,
        Err(e) => {
            log_embedding_skip_once("auto-reindex vectors", &e);
            return;
        }
    };
    let mut provider = greppy_indexer::EmbeddingGemmaCodeProvider::new(&cfg.model_id, &model);
    if let Err(e) = greppy_indexer::index_code_embeddings_for_project(
        store,
        root,
        project,
        &mut provider,
        greppy_indexer::EmbeddingIndexOptions::for_generation(report.graph_generation),
    ) {
        log_embedding_skip_once("auto-reindex vectors", &e);
    }
}

/// P11: below this node count the first-use embedding build finishes fast
/// enough to run inline (the agent gets semantic results on THIS query);
/// above it, the build goes to a detached background process so the query
/// never blocks. ~3000 nodes embeds in a couple of seconds on Metal/CUDA
/// and still tolerably on CPU; larger repos are the ones that hit minutes.
const INLINE_EMBED_MAX_NODES: i64 = 3000;

fn project_is_small_enough_for_inline_embed(store: &greppy_store::Store, project: &str) -> bool {
    store
        .stats(project)
        .map(|s| s.total_nodes <= INLINE_EMBED_MAX_NODES)
        .unwrap_or(false)
}

/// Kick off `grep index <root> --embeddings` as a detached child so the
/// semantic index builds in the background (P11). The current binary
/// carries the embedded model, so no env is needed. Best-effort: a failure
/// to spawn just means the next query heals inline instead.
fn spawn_background_embed(root: Option<&str>) {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Ok(root_path) = resolve_root(root) else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("index")
        .arg(&root_path)
        .arg("--root")
        .arg(&root_path)
        .arg("--embeddings")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // With null stdio and no wait(), the child is orphaned (reparented to
    // init) when this short-lived CLI exits — it keeps running and cannot
    // receive a terminal SIGHUP (no controlling tty on its fds). Good
    // enough for a background reindex without pulling in a setsid dep.
    let _ = cmd.spawn();
}

/// First-use semantic self-heal (P2): build code-span embeddings for the
/// project when a model is configured but the store has none. Returns true
/// when a build was attempted successfully. Failures are logged (once) and
/// never fail the query — the caller falls back to the honest skip note.
fn build_vectors_first_use(
    cfg: &EmbeddingModelConfig,
    root: Option<&str>,
    project: &str,
    generation: u64,
) -> bool {
    let root_path = match resolve_root(root) {
        Ok(p) => p,
        Err(e) => {
            log_embedding_skip_once("context first-use vectors", &e);
            return false;
        }
    };
    let model = match load_embedding_model(cfg, None) {
        Ok(m) => m,
        Err(e) => {
            log_embedding_skip_once("context first-use vectors", &e);
            return false;
        }
    };
    // The query path holds a READ-ONLY store handle; the heal needs its
    // own writable connection (same DB, WAL allows one writer + readers).
    let store_path = workspace_locator::store_path(&root_path);
    let mut write_store = match greppy_store::Store::open(&store_path) {
        Ok(s) => s,
        Err(e) => {
            log_embedding_skip_once("context first-use vectors", &e.into());
            return false;
        }
    };
    // Use the generation CURRENT AT WRITE TIME: the same context call may
    // have auto-reindexed (bumping the generation) after the caller read
    // it — embeddings written under the stale value are invisible to the
    // retrieval scope and get evicted (observed: 48 of ~8k spans stored).
    let write_generation = match current_graph_generation(&write_store, root) {
        Ok(g) => g,
        Err(_) => generation,
    };
    let mut provider = greppy_indexer::EmbeddingGemmaCodeProvider::new(&cfg.model_id, &model);
    if let Err(e) = greppy_indexer::index_code_embeddings_for_project(
        &mut write_store,
        &root_path,
        project,
        &mut provider,
        greppy_indexer::EmbeddingIndexOptions::for_generation(write_generation),
    ) {
        log_embedding_skip_once("context first-use vectors", &e);
        return false;
    }
    true
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
        s.push_str("…");
    }
    Some(s)
}

fn ensure_nav_json_mode(code: bool, json: bool) -> Result<()> {
    if code && json {
        return Err(Error::Invalid(
            "--json cannot be combined with --code for navigation commands".into(),
        ));
    }
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
/// IMPORTANT (the token-saving design): the store records a node's
/// `start_line`/`end_line` as the **declaration line** of the symbol — in
/// the current indexer `end_line == start_line` for every definition,
/// because the captured tree-sitter node is the name identifier, not the
/// full item. Reading only that single line would defeat the whole point
/// of `context`/`--code` (the agent would still have to open the file to
/// see the body). So we extend the span from `start_line` to the end of
/// the definition by balancing `{}`/`()`/`[]` delimiters across the
/// source (see [`definition_end_idx`]). When a future indexer stores a
/// real multi-line `end_line`, we honour the larger of the stored end and
/// the computed end, so this stays correct either way.
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
    // Stored (declaration) end, clamped to the file. For the current
    // indexer this equals start_idx.
    let stored_end_idx = std::cmp::min(end_line as usize, all.len()) - 1;
    // Computed body end: balance delimiters forward from the declaration
    // line so we capture the whole `{ … }` (or `;`-terminated) item.
    let computed_end_idx = definition_end_idx(&all, start_idx);
    // Honour whichever end is further (forward-compatible with a future
    // indexer that records true multi-line spans).
    let end_idx_inclusive = std::cmp::max(stored_end_idx, computed_end_idx);
    let total_lines = end_idx_inclusive - start_idx + 1;
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
        for line in span.lines() {
            println!("    {line}");
        }
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
                &[],
            )?;
            return Ok(1);
        }
        println!("(symbol not found)");
        return Ok(1);
    };
    let steps = greppy_search::trace_path(&store, start, dir, edge_filter, depth)?;
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
            &steps,
        )?;
        return Ok(0);
    }
    // `--code` reads spans from disk relative to the resolved repo root.
    let span_root = if code {
        Some(resolve_root(root)?)
    } else {
        None
    };
    for s in &steps {
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
    direction: &str,
    edge: Option<&str>,
    depth: usize,
    since: Option<&str>,
    base: Option<&str>,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
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
    let store = open_default_store_query_writer(root)?;
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
        return content_fallback(&store, root, symbol.unwrap_or(""), "impact");
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
    let shown = if all { total } else { total.min(NAV_LIMIT) };
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
    // D2 fail-open: refuse only when no usable index exists; a stale
    // index serves labeled results (auto-healed first when small).
    if let FreshnessServe::Refuse(freshness) = freshness_serve_decision(&store, root, &project) {
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
        return Ok(1);
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

/// `greppy brief S` — a one-call briefing: the definition (with source
/// span), the direct callers, and the direct callees. Composes the same
/// resolution/edge helpers as context/who-calls/callees so an agent can
/// answer "how does S work / what is its role / what depends on it" from a
/// SINGLE call instead of three, which is exactly where the benchmark showed
/// research-task iteration eating the token/time savings.
fn dispatch_brief(symbol: Option<&str>, root: Option<&str>) -> Result<i32> {
    let store = open_default_store_query_writer(root)?;
    let project = project_for(root)?;
    let query_symbol = symbol.unwrap_or("");
    if let Some(code) = graph_stale_gate(
        &store,
        root,
        &project,
        "brief",
        false,
        serde_json::Value::Null,
        "hits",
    )? {
        return Ok(code);
    }
    if let Some(code) = provider_policy_graph_gate(
        &store,
        root,
        &project,
        "brief",
        false,
        serde_json::Value::Null,
        "hits",
    )? {
        return Ok(code);
    }
    let targets = resolve_symbol_nodes(&store, symbol)?;
    if targets.is_empty() {
        return content_fallback(&store, root, symbol.unwrap_or(""), "brief");
    }
    let root_path = resolve_root(root)?;
    let mut evidence_nodes: Vec<(String, greppy_store::Node, serde_json::Value)> = Vec::new();

    // Definition(s) + source span.
    let mut seen_def = std::collections::BTreeSet::new();
    for id in &targets {
        if let Some(n) = store.get_node(*id)? {
            if seen_def.insert(n.id) {
                evidence_nodes.push((
                    format!("definition {}", display_node_name(&n)),
                    n.clone(),
                    serde_json::json!({"section": "definition"}),
                ));
                println!(
                    "== {} ({}:{}-{}) ==",
                    display_node_name(&n),
                    n.file_path,
                    n.start_line,
                    n.end_line
                );
                print_code_span(&root_path, &n, CONTEXT_SPAN_CAP);
            }
        }
    }

    let callers = incoming_call_nodes_for_targets(&store, &targets)?;
    let cshown = callers.len().min(BRIEF_LIMIT);
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
        let total = greppy_search::count_references_to_any(&store, &project, &targets)?;
        let refs = greppy_search::find_references_to_any(&store, &targets, BRIEF_LIMIT)?;
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
    let eshown = callees.len().min(BRIEF_LIMIT);
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

fn dispatch_expand(id: Option<&str>, json: bool, root: Option<&str>) -> Result<i32> {
    let id = id.unwrap_or("").trim();
    if id.is_empty() {
        return Err(Error::Invalid("expand requires an id".into()));
    }
    let store = open_default_store_query_writer(root)?;
    let Some(pack) = store.get_expand_pack(id)? else {
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
    let dirty_overlay = dirty_overlay(&effective_root)?;

    if !store_path.exists() {
        let status = serde_json::json!({
            "command": command,
            "status": "no_index",
            "healthy": false,
            "store_exists": false,
            "root_path": effective_root,
            "store_path": store_path,
            "project": project,
            "fresh": false,
            "freshness": null,
            "schema_current": false,
            "integrity_ok": false,
            "project_present": false,
            "incomplete_provider_count": null,
            "skip_counts_by_reason": [],
            "dirty_overlay": dirty_overlay.to_json(),
            "message": "no active index; run grep index first",
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
            println!("message: run `grep index {}` first", root.unwrap_or("."));
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
             re-run `grep index` with the current binary"
        )),
        _ => None,
    };
    let vectors_missing_with_model = {
        let no_args = EmbeddingCliArgs {
            enabled: false,
            model_dir: None,
            gguf: None,
            tokenizer: None,
            model_id: None,
            max_length: None,
            device: None,
            no_gpu: false,
        };
        matches!(embedding_config_optional(no_args), Ok(Some(_)))
            && store
                .vector_model_ids(&project)
                .map(|v| v.is_empty())
                .unwrap_or(false)
    };
    let healthy = diag.schema_current
        && diag.integrity_ok
        && project_present
        && fresh
        && incomplete_provider_count == 0
        && coverage_warning.is_none();
    let status_label = if healthy { "ok" } else { "unhealthy" };

    if json {
        let value = serde_json::json!({
            "command": command,
            "status": status_label,
            "healthy": healthy,
            "store_exists": true,
            "root_path": effective_root,
            "store_path": store_path,
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
            "skip_counts_by_reason": skip_counts,
            "git_tracked_files": git_tracked,
            "coverage_warning": coverage_warning,
            "vectors_missing_with_model": vectors_missing_with_model,
            "dirty_overlay": dirty_overlay.to_json(),
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
                 — `semantic-search --vectors` will build them on first use, or run \
                 `grep index --embeddings ...` now"
            );
        }
        println!("root: {}", effective_root.display());
        println!("store: {}", store_path.display());
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

/// `greppy who-calls S` — the callers of `S`: every node with an
/// incoming CALLS edge into `S`. Printed as `qualified_name file:line`
/// so an agent can jump straight to each call site's enclosing symbol.
/// Content-search fallback for who-calls / find-usages when the call/usage
/// GRAPH has no edges for `symbol` (e.g. a weakly-connected single-file symbol,
/// a macro, or a name that is not a graph node at all). Runs the indexed
/// content search on the name so the agent still gets `file:line` hits from ONE
/// greppy call — instead of finding nothing and falling back to a grep loop.
/// This was the token-efficiency benchmark's only case where greppy lost to
/// grep (`find-usages GraphIndex`): now greppy is never worse than grep for a
/// name query, since it always returns indexed matches.
fn content_fallback(
    store: &greppy_store::Store,
    root: Option<&str>,
    symbol: &str,
    kind: &str,
) -> Result<i32> {
    let project = project_for(root)?;
    let hits = greppy_search::search_code(store, &project, symbol, 50)?;
    if hits.is_empty() {
        // Truly nothing — not a graph symbol and no indexed text either.
        // Offer the closest indexed names so the dead end carries a next
        // step (P3: agents otherwise retry blind variants or bail to grep).
        let needle = split_qualified(symbol).map(|(_, m)| m).unwrap_or(symbol);
        let similar = store
            .similar_node_names(&project, needle, 5)
            .unwrap_or_default();
        if similar.is_empty() {
            println!("(symbol not found: `{symbol}`; no {kind} and no indexed content matches)");
        } else {
            println!(
                "(symbol not found: `{symbol}`; no {kind}. Similar indexed names: {} — retry with one of these)",
                similar.join(", ")
            );
        }
        return Ok(1);
    }
    println!(
        "(`{symbol}` is not a graph symbol; {} indexed content match(es) (would-be {kind}):)",
        hits.len()
    );
    for h in &hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

fn dispatch_who_calls(
    symbol: Option<&str>,
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let store = open_default_store_query_writer(root)?;
    let query_symbol = symbol.unwrap_or("");
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
    // aggregate incoming CALLS across ALL nodes sharing
    // the name + a primary label (e.g. a Struct and its Impl) so callers
    // are not lost to a name resolving to the wrong single node.
    let targets = resolve_symbol_nodes(&store, symbol)?;
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
        return content_fallback(&store, root, symbol.unwrap_or(""), "callers");
    }
    let mut edges = Vec::new();
    for target in &targets {
        edges.extend(store.incoming_edges(*target, Some("CALLS"), 1024)?);
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
        println!("(no callers)");
        print_zero_nav_footer(&store, &project, "caller", &targets, "calls")?;
        // O6: zero RESOLVED callers on a defined symbol is exactly where
        // dynamic dispatch hides — offer the textual candidates so the
        // agent doesn't re-derive them with its own grep rounds.
        print_textual_call_candidates(&store, &project, query_symbol, &targets, &[])?;
        return Ok(0);
    }
    // `--code` reads spans from disk relative to the resolved repo root.
    let span_root = if code {
        Some(resolve_root(root)?)
    } else {
        None
    };
    // Deterministic, de-duplicated output across the aggregated targets.
    // First collect the unique caller nodes so we know the true total, then
    // print at most NAV_LIMIT (F1: cap the token-bomb) unless `--all`.
    let mut seen = std::collections::BTreeSet::new();
    let mut nodes = Vec::new();
    // P4: collect each caller's CALL-SITE lines (persisted in the edge
    // properties). Printed grep-shaped below so one who-calls answer
    // carries the evidence — the spot forensics showed agents re-reading
    // files after who-calls just to see HOW the call is made.
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
    let total = nodes.len();
    let cap = if code { CODE_NAV_LIMIT } else { NAV_LIMIT };
    let shown = if all { total } else { total.min(cap) };
    let expand = if !all && !code {
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
    for n in &nodes[..shown] {
        println!("{} {}", display_node_name(n), node_line_span(n));
        if let Some(lines) = sites.get(&n.id) {
            let mut lines = lines.clone();
            lines.sort_unstable();
            lines.dedup();
            for l in lines.iter().take(3) {
                if let Some(text) = read_source_line(&repo_root, &n.file_path, *l) {
                    println!("  {}:{}: {}", n.file_path, l, text);
                }
            }
        }
        // Track A: with `--code`, print the caller's body so the agent
        // sees the call site's enclosing symbol without a separate Read.
        if let Some(root_path) = span_root.as_deref() {
            print_code_span(root_path, n, CODE_SPAN_CAP);
        }
    }
    let row_refs: Vec<&greppy_store::Node> = nodes.iter().collect();
    let (provider_incomplete, _) =
        nav_target_provider_incomplete(&store, &project, &row_refs, "calls")?;
    NavFooter {
        noun: "caller",
        total,
        shown,
        provider_incomplete,
    }
    .print();
    print_textual_call_candidates(&store, &project, query_symbol, &targets, &nodes)?;
    if let Some(expand) = &expand {
        println!("{}", expand.text_line());
    }
    Ok(0)
}

/// O6 (django forensics, r044: 26-call grep spiral): the never-guess
/// resolver deliberately does not link dynamic `obj.method()` dispatch, so
/// on dynamic code `who-calls` under-reports and the agent re-derives the
/// rest with its own grep rounds. This prints ONE honestly-labelled section
/// of TEXTUAL call-site candidates from the indexed content FTS — the graph
/// section above stays the authority; this section only saves the agent its
/// own text search.
///
/// Noise control: candidates are only shown for files that do NOT already
/// contain a resolved caller — on statically-resolved code (java/rust) the
/// text sites live in resolved-caller files, so the section stays silent
/// and costs zero tokens; on dynamic code the unresolved files are exactly
/// what is missing. Definition-looking lines are excluded.
fn print_textual_call_candidates(
    store: &greppy_store::Store,
    project: &str,
    symbol: &str,
    target_ids: &[i64],
    resolved: &[greppy_store::Node],
) -> Result<()> {
    const CANDIDATE_CAP: usize = 10;
    let name = symbol
        .rsplit(['.', ':', '#'])
        .next()
        .unwrap_or(symbol)
        .trim();
    // Too-short names text-match half the repo; not worth the noise.
    if name.len() < 4 {
        return Ok(());
    }
    let call_pat = format!("{name}(");
    let resolved_files: std::collections::BTreeSet<&str> =
        resolved.iter().map(|n| n.file_path.as_str()).collect();
    let mut def_locs = std::collections::BTreeSet::new();
    for id in target_ids {
        if let Some(n) = store.get_node(*id)? {
            def_locs.insert(format!("{}:{}", n.file_path, n.start_line));
        }
    }
    let hits = match greppy_search::search_code(store, project, name, 80) {
        Ok(h) => h,
        Err(_) => return Ok(()), // candidates are best-effort, never an error
    };
    let mut out: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for h in hits {
        let snippet = h.snippet.trim();
        if !snippet.contains(&call_pat) {
            continue;
        }
        // Skip definition-shaped lines (`fn name(`, `def name(`, …) and the
        // definition locations themselves.
        let is_def = ["fn ", "def ", "function ", "class ", "async def "]
            .iter()
            .any(|kw| snippet.contains(&format!("{kw}{name}")));
        if is_def || def_locs.contains(&h.location) {
            continue;
        }
        let file = h.location.rsplit_once(':').map(|(f, _)| f).unwrap_or("");
        if resolved_files.contains(file) {
            continue; // already represented by a resolved caller
        }
        if !seen.insert(h.location.clone()) {
            continue;
        }
        let mut line = snippet.to_string();
        if line.len() > 120 {
            line.truncate(120);
            line.push('…');
        }
        out.push((h.location, line));
    }
    if out.is_empty() {
        return Ok(());
    }
    let total = out.len();
    let shown = total.min(CANDIDATE_CAP);
    println!("unresolved textual candidates: {total}");
    for (loc, snippet) in out.iter().take(shown) {
        println!("  {loc}: {snippet}");
    }
    if total > shown {
        println!("  … and {} more candidates", total - shown);
    }
    Ok(())
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
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let store = open_default_store_query_writer(root)?;
    let query_symbol = symbol.unwrap_or("");
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
        println!("(symbol not found)");
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
        println!("(no callees)");
        print_zero_nav_footer(&store, &project, "callee", &sources, "calls")?;
        return Ok(0);
    }
    // `--code` reads spans from disk relative to the resolved repo root.
    let span_root = if code {
        Some(resolve_root(root)?)
    } else {
        None
    };
    let total = callees.len();
    let cap = if code { CODE_NAV_LIMIT } else { NAV_LIMIT };
    let shown = if all { total } else { total.min(cap) };
    let expand = if !all && !code {
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
    for n in callees.values().take(shown) {
        println!("{} {}", display_node_name(n), node_line_span(n));
        // Track A: with `--code`, print the callee's body so the agent
        // sees the callee definition without a separate Read.
        if let Some(root_path) = span_root.as_deref() {
            print_code_span(root_path, n, CODE_SPAN_CAP);
        }
    }
    let row_refs: Vec<&greppy_store::Node> = callees.values().collect();
    let (provider_incomplete, _) =
        nav_target_provider_incomplete(&store, &project, &row_refs, "calls")?;
    NavFooter {
        noun: "callee",
        total,
        shown,
        provider_incomplete,
    }
    .print();
    if let Some(expand) = &expand {
        println!("{}", expand.text_line());
    }
    Ok(0)
}

/// `greppy find-usages S` — where `S` is referenced: every node with an
/// incoming reference edge into `S`. Printed as `KIND qualified_name
/// file:line` so the edge kind is visible.
fn dispatch_find_usages(
    symbol: Option<&str>,
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let store = open_default_store_query_writer(root)?;
    let query_symbol = symbol.unwrap_or("");
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
        "find-usages",
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
        "find-usages",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    // aggregate incoming USAGE across ALL nodes sharing the
    // name + a primary label (e.g. a Class and its Impl). Previously the name
    // resolved to the first node found (often the wrong one — `Store` ->
    // EnumVariant `Error::Store`, `IndexReport` -> `Impl::IndexReport`), so
    // real usages were reported as "(no usages)". The graph now persists every
    // non-call, non-import identifier reference under the single unified
    // `USAGE` label (the former `TYPE_REF` + `USES` passes), so one query
    // covers both type and value references.
    let targets = resolve_symbol_nodes(&store, symbol)?;
    if targets.is_empty() {
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "find-usages",
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
        return content_fallback(&store, root, symbol.unwrap_or(""), "usages");
    }
    let mut edges = Vec::new();
    // P10: "usages" to an agent means EVERY reference. A function whose
    // only references are calls returned a confidently wrong "(no usages)"
    // while who-calls listed its 2 callers (spot forensics, wrap_in_const)
    // — the agent then burned three extra calls distrusting the tool.
    // Aggregate all reference-class edges; each output row is already
    // labelled with its edge type, so the answer stays honest.
    for target in &targets {
        for &et in greppy_search::REFERENCE_EDGE_TYPES {
            edges.extend(store.incoming_edges(*target, Some(et), 1024)?);
        }
    }
    if edges.is_empty() {
        // Resolved graph node with no usages — a valid answer; no content noise.
        if json {
            let project = project_for(root)?;
            nav_counts_json(
                &store,
                root,
                "find-usages",
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
        println!("(no usages)");
        print_zero_nav_footer(&store, &project, "usage", &targets, "usages")?;
        return Ok(0);
    }
    // `--code` reads spans from disk relative to the resolved repo root.
    let span_root = if code {
        Some(resolve_root(root)?)
    } else {
        None
    };
    // Deterministic, de-duplicated output keyed on (edge_type, source).
    // Collect first so we know the true total, then cap at NAV_LIMIT (F1)
    // unless `--all`.
    let mut seen = std::collections::BTreeSet::new();
    let mut rows: Vec<(String, greppy_store::Node)> = Vec::new();
    // P4: collect the reference-site lines per (edge_type, source) so the
    // usage answer prints grep-shaped ' file:line: code' evidence.
    let mut sites: std::collections::HashMap<(String, i64), Vec<u32>> =
        std::collections::HashMap::new();
    for e in &edges {
        if let Some(l) = e.properties.get("line").and_then(|v| v.as_u64()) {
            sites
                .entry((e.edge_type.clone(), e.source_id))
                .or_default()
                .push(l as u32);
        }
        if !seen.insert((e.edge_type.clone(), e.source_id)) {
            continue;
        }
        if let Some(n) = store.get_node(e.source_id)? {
            rows.push((e.edge_type.clone(), n));
        }
    }
    let total = rows.len();
    let cap = if code { CODE_NAV_LIMIT } else { NAV_LIMIT };
    let shown = if all { total } else { total.min(cap) };
    let expand = if !all && !code {
        let evidence_rows = rows
            .iter()
            .map(|(edge_type, n)| ExpandEvidenceNode {
                title: format!("{edge_type} {}", display_node_name(n)),
                node: n,
                site_lines: sorted_site_lines(sites.get(&(edge_type.clone(), n.id))),
                extra_json: serde_json::json!({"edge_type": edge_type}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(
            &store,
            root,
            &project,
            "find-usages",
            query_symbol,
            total,
            &evidence_rows,
        )
    } else {
        None
    };
    if json {
        let project = project_for(root)?;
        let hits = rows[..shown]
            .iter()
            .map(|(edge_type, n)| {
                serde_json::json!({
                    "edge_type": edge_type,
                    "qualified_name": &n.qualified_name,
                    "file_path": &n.file_path,
                    "start_line": n.start_line,
                    "end_line": n.end_line,
                })
            })
            .collect();
        nav_counts_json_with_expand(
            &store,
            root,
            "find-usages",
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
    for (edge_type, n) in &rows[..shown] {
        println!(
            "{} {} {}",
            edge_type,
            display_node_name(n),
            node_line_span(n)
        );
        if let Some(lines) = sites.get(&(edge_type.clone(), n.id)) {
            let mut lines = lines.clone();
            lines.sort_unstable();
            lines.dedup();
            for l in lines.iter().take(3) {
                if let Some(text) = read_source_line(&repo_root, &n.file_path, *l) {
                    println!("  {}:{}: {}", n.file_path, l, text);
                }
            }
        }
        // Track A: with `--code`, print the referencing node's body so
        // the agent sees the usage site without a separate Read.
        if let Some(root_path) = span_root.as_deref() {
            print_code_span(root_path, n, CODE_SPAN_CAP);
        }
    }
    let row_refs: Vec<&greppy_store::Node> = rows.iter().map(|(_, n)| n).collect();
    let (provider_incomplete, _) =
        nav_target_provider_incomplete(&store, &project, &row_refs, "usages")?;
    NavFooter {
        noun: "usage",
        total,
        shown,
        provider_incomplete,
    }
    .print();
    if let Some(expand) = &expand {
        println!("{}", expand.text_line());
    }
    Ok(0)
}

/// `greppy references S` — every incoming graph reference to `S` across
/// CALLS, USAGE, legacy USES/TYPE_REF, and IMPORTS. Unlike find-usages, this
/// intentionally has no content fallback: a "references" answer must stay a
/// graph answer so agents can trust the edge kind and exact count metadata.
fn dispatch_references(
    symbol: Option<&str>,
    code: bool,
    all: bool,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    ensure_nav_json_mode(code, json)?;
    let store = open_default_store_query_writer(root)?;
    let query_symbol = symbol.unwrap_or("");
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
        "references",
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
        "references",
        json,
        graph_gate_extra,
        "hits",
    )? {
        return Ok(code);
    }
    let targets = resolve_symbol_nodes(&store, symbol)?;
    if targets.is_empty() {
        if json {
            nav_counts_json(
                &store,
                root,
                "references",
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
        println!("(symbol not found)");
        return Ok(1);
    }

    let total = greppy_search::count_references_to_any(&store, &project, &targets)?;
    let cap = if code { CODE_NAV_LIMIT } else { NAV_LIMIT };
    let fetch_limit = if all {
        greppy_search::MAX_REACH_RESULTS
    } else {
        EXPAND_NAV_EVIDENCE_LIMIT.max(cap)
    };
    let refs = greppy_search::find_references_to_any(&store, &targets, fetch_limit)?;
    let shown = if all { refs.len() } else { refs.len().min(cap) };
    let expand = if !all && !code {
        let mut nodes = Vec::new();
        for r in &refs {
            if let Some(node) = store.get_node(r.node.id)? {
                nodes.push((r.edge_type.clone(), node));
            }
        }
        let evidence_rows = nodes
            .iter()
            .map(|(edge_type, node)| ExpandEvidenceNode {
                title: format!("{edge_type} {}", display_node_name(node)),
                node,
                site_lines: Vec::new(),
                extra_json: serde_json::json!({"edge_type": edge_type}),
            })
            .collect::<Vec<_>>();
        insert_nav_expand_pack(
            &store,
            root,
            &project,
            "references",
            query_symbol,
            total,
            &evidence_rows,
        )
    } else {
        None
    };

    if json {
        let hits = refs[..shown]
            .iter()
            .map(|r| {
                serde_json::json!({
                    "edge_type": &r.edge_type,
                    "node_id": r.node.id,
                    "qualified_name": &r.node.qualified_name,
                    "name": &r.node.name,
                    "label": &r.node.label,
                    "file_path": &r.node.file_path,
                    "start_line": r.node.start_line,
                    "end_line": r.node.end_line,
                })
            })
            .collect();
        nav_counts_json_with_expand(
            &store,
            root,
            "references",
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

    if refs.is_empty() {
        println!("(no references)");
        return Ok(0);
    }

    let span_root = if code {
        Some(resolve_root(root)?)
    } else {
        None
    };
    for r in &refs[..shown] {
        println!(
            "{} {} {}",
            r.edge_type,
            display_row_name(&r.node),
            line_span(&r.node.file_path, r.node.start_line, r.node.end_line)
        );
        if let Some(root_path) = span_root.as_deref() {
            if let Some(node) = store.get_node(r.node.id)? {
                print_code_span(root_path, &node, CODE_SPAN_CAP);
            }
        }
    }
    print_nav_more_footer(total, shown);
    if let Some(expand) = &expand {
        println!("{}", expand.text_line());
    }
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
            println!("(no path from {from} to {to} via {edge_upper})");
            Ok(1)
        }
    }
}

fn dispatch_search_symbols(
    query: Option<&str>,
    kind: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("search-symbols requires a query".into()));
    }
    let project = project_for(root)?;
    // D2 fail-open: refuse only when no usable index exists; a stale
    // index serves labeled results (auto-healed first when small).
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
            )?;
        } else {
            println!(
                "{}",
                indexed_stale_skip_message("search-symbols", freshness)
            );
        }
        return Ok(1);
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
            )?;
        } else {
            println!(
                "{}",
                provider_incomplete_skip_message("search-symbols", incomplete_providers.len())
            );
        }
        return Ok(1);
    }

    // --kind: fetch a wider candidate set, then keep only nodes whose
    // label matches case-insensitively (agents type `--kind function`).
    let fetch = if kind.is_some() { 100 } else { 20 };
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
        hits.truncate(20);
    }
    if json {
        search_symbols_json(&store, q, &project, "ok", Some(&freshness), &hits)?;
        return Ok(if hits.is_empty() { 1 } else { 0 });
    }
    if hits.is_empty() {
        println!("(no matches)");
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

fn search_symbols_json(
    store: &greppy_store::Store,
    query: &str,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    hits: &[greppy_search::SymbolHit],
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let total_exact = if status == "ok" {
        greppy_search::count_symbols_in_project(store, project, query)?
    } else {
        0
    };
    let mut rows = Vec::new();
    for h in hits {
        match store.get_node(h.node_id)? {
            Some(n) => rows.push(serde_json::json!({
                "node_id": h.node_id,
                "rank": h.rank,
                "label": n.label,
                "name": n.name,
                "qualified_name": n.qualified_name,
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

fn dispatch_search_code(
    query: Option<&str>,
    changed: bool,
    staged: bool,
    since: Option<&str>,
    base: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("search-code requires a query".into()));
    }
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
        return dispatch_search_code_changed(q, json, root);
    }
    if staged {
        return dispatch_search_code_staged(q, json, root);
    }
    if let Some(rev) = since {
        return dispatch_search_code_since(q, rev, json, root);
    }
    if let Some(rev) = base {
        return dispatch_search_code_base(q, rev, json, root);
    }
    let store = open_default_store(root)?;
    // Project identity is derived from the
    // canonical repo root (or `--root` when supplied), not from the
    // cwd basename. Index + search-code + semantic must agree on
    // this value so a search after an index hits the right rows.
    let project = project_for(root)?;
    // D2 fail-open: refuse only when no usable index exists. A stale
    // index is auto-healed when the drift is small; if it stays stale,
    // text output prefers a LIVE grep (strictly fresher than stale
    // indexed rows) while JSON keeps the indexed shape, honestly
    // labeled `fresh: false`.
    let decision = freshness_serve_decision(&store, root, &project);
    match &decision {
        FreshnessServe::Refuse(freshness) => {
            if json {
                search_code_json(
                    &store,
                    q,
                    &project,
                    "skipped_stale_index",
                    Some(freshness),
                    0,
                    &[],
                )?;
                return Ok(1);
            }
            eprintln!(
                "{}; falling back to live grep",
                indexed_stale_skip_message("search-code", freshness)
            );
            return live_grep_search_code(q, root);
        }
        FreshnessServe::StaleLabeled(freshness) => {
            if !json {
                eprintln!(
                    "{}; falling back to live grep",
                    indexed_stale_skip_message("search-code", freshness)
                );
                return live_grep_search_code(q, root);
            }
        }
        FreshnessServe::Fresh(_) => {}
    }
    let freshness = decision.freshness().clone();

    let hits = greppy_search::search_code(&store, &project, q, SEARCH_CODE_LIMIT)?;
    if json {
        let total_exact = store.count_file_content_matches(&project, q)?;
        search_code_json(
            &store,
            q,
            &project,
            "ok",
            Some(&freshness),
            total_exact,
            &hits,
        )?;
        return Ok(if total_exact == 0 { 1 } else { 0 });
    }
    if hits.is_empty() {
        // Content-FTS empty → either no match, or the index was built with
        // content indexing off (GREPPY_NO_CONTENT / a fast index). Either
        // way, fall back to a LIVE grep over the repo — `search_code` finds
        // text patterns via grep, then enriches with the graph. greppy is a
        // grep wrapper, so this is free and keeps `search-code` working
        // without eager content-FTS.
        return live_grep_search_code(q, root);
    }
    for h in &hits {
        println!("{}  {}", h.location, clamp_snippet(&h.snippet));
    }
    Ok(0)
}

fn search_code_json(
    store: &greppy_store::Store,
    query: &str,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    total_exact: usize,
    hits: &[greppy_search::CodeHit],
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
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
            "status": status,
            "query": query,
            "project": project,
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
        .map_err(|e| Error::Invalid(format!("serialize search-code JSON: {e}")))?
    );
    Ok(())
}

fn dispatch_search_code_changed(query: &str, json: bool, root: Option<&str>) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let changed_files = git_changed_files(&root_path)?;
    let all_hits = live_grep_search_code_paths(query, &root_path, &changed_files)?;
    let shown_hits = all_hits
        .iter()
        .take(SEARCH_CODE_LIMIT)
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_changed_json(
            query,
            &project,
            changed_files.len(),
            all_hits.len(),
            &shown_hits,
        )?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        println!("(no matches)");
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

fn dispatch_search_code_staged(query: &str, json: bool, root: Option<&str>) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let staged_files = git_staged_files(&root_path)?;
    let all_hits = grep_staged_git_blobs(query, &root_path, &staged_files)?;
    let shown_hits = all_hits
        .iter()
        .take(SEARCH_CODE_LIMIT)
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
        println!("(no matches)");
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
    root: Option<&str>,
) -> Result<i32> {
    dispatch_search_code_diff_scope(query, DiffSearchScope::Since { rev }, json, root)
}

fn dispatch_search_code_base(
    query: &str,
    base: &str,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    dispatch_search_code_diff_scope(query, DiffSearchScope::Base { base }, json, root)
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
    root: Option<&str>,
) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let project = workspace_locator::project_identity(&root_path);
    let spec = git_diff_search_spec(&root_path, scope)?;
    let all_hits = live_grep_search_code_paths(query, &root_path, &spec.files)?;
    let shown_hits = all_hits
        .iter()
        .take(SEARCH_CODE_LIMIT)
        .cloned()
        .collect::<Vec<_>>();

    if json {
        search_code_diff_scope_json(query, &project, &spec, all_hits.len(), &shown_hits)?;
        return Ok(if all_hits.is_empty() { 1 } else { 0 });
    }

    if shown_hits.is_empty() {
        println!("(no matches)");
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
/// hints. `--vectors` adds EmbeddingGemma code-retrieval hits as one more
/// search signal, scoped to the current graph generation.
#[allow(clippy::too_many_arguments)]
fn dispatch_plus(
    query: Option<&str>,
    k: usize,
    code: bool,
    explain: bool,
    json: bool,
    vectors: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let store = open_default_store(root)?;
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("a query is required".into()));
    }
    let k = k.max(1);
    let project = project_for(root)?;
    let root_path = resolve_root(root)?;
    // D2 fail-open: refuse only when no usable index exists; a stale
    // index serves labeled results (auto-healed first when small).
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
                "(no usable index; run `grep index {}` first)",
                root.unwrap_or(".")
            );
        }
        return Ok(1);
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

    // Literal/full-text signal: exact indexed code lines remain first-class
    // grep-like results.
    let code_hits = greppy_search::search_code_ranked(&store, &project, q, fetch)?;
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
        Some(embedding_config_required(embedding_args)?)
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
            for h in greppy_search::search_code_ranked(&store, &project, &tok, fetch / 2)? {
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
fn live_grep_search_code(query: &str, root: Option<&str>) -> Result<i32> {
    let root_path = resolve_root(root)?;
    let out = std::process::Command::new("grep")
        .args(["-rnI", "--", query])
        .arg(".")
        .current_dir(&root_path)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            return Err(Error::io(
                "spawn grep for search-code fallback".to_string(),
                e,
            ))
        }
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut printed = 0usize;
    for line in text.lines() {
        if printed >= SEARCH_CODE_LIMIT {
            break;
        }
        if let Some(hit) = parse_grep_code_hit(line) {
            println!("{}  {}", hit.location, clamp_snippet(&hit.snippet));
            printed += 1;
        }
    }
    if printed == 0 {
        println!("(no matches)");
    }
    Ok(0)
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

fn live_grep_search_code_paths(
    query: &str,
    root_path: &std::path::Path,
    paths: &[String],
) -> Result<Vec<greppy_search::CodeHit>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }

    let mut hits = Vec::new();
    for chunk in paths.chunks(128) {
        let out = std::process::Command::new("grep")
            .args(["-HnI", "--", query])
            .args(chunk)
            .current_dir(root_path)
            .output()
            .map_err(|e| Error::io("spawn grep for search-code --changed", e))?;
        let text = String::from_utf8_lossy(&out.stdout);
        hits.extend(text.lines().filter_map(parse_grep_code_hit));
    }
    Ok(hits)
}

fn grep_staged_git_blobs(
    query: &str,
    root_path: &std::path::Path,
    paths: &[String],
) -> Result<Vec<greppy_search::CodeHit>> {
    use std::io::Write;

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

        let mut child = std::process::Command::new("grep")
            .args(["-nI", "--", query])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| Error::io("spawn grep for search-code --staged", e))?;
        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(&blob.stdout)
                .map_err(|e| Error::io(format!("write staged blob {path} to grep"), e))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| Error::io("wait for grep in search-code --staged", e))?;
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
    vectors: bool,
    json: bool,
    embedding_args: EmbeddingCliArgs<'_>,
    root: Option<&str>,
) -> Result<i32> {
    let q = query.unwrap_or("").trim();
    if q.is_empty() {
        return Err(Error::Invalid("semantic-search requires a query".into()));
    }

    let store = open_default_store_query_writer(root)?;
    let project = project_for(root)?;
    // D2 fail-open: refuse only when no usable index exists; a stale
    // index serves labeled results (auto-healed first when small). The
    // vector path keeps its own stricter stale check below (vector rows
    // are generation-scoped, so stale-serving them is meaningless). It may
    // self-heal via the inline auto-reindex ONLY when the embedding model
    // is resolvable — the reindex then rebuilds the vectors for the new
    // generation (auto_rebuild_vectors_inline). Without a resolvable model
    // an inline reindex would bump the generation and orphan the very
    // vectors it queries, so we keep the old skip behaviour.
    let allow_reindex = !vectors || vector_auto_reindex_can_rebuild(embedding_args);
    let decision =
        freshness_serve_decision_with_policy(&store, root, &project, allow_reindex, !vectors);
    let incomplete_providers = incomplete_provider_json(&store, &project)?;

    if !vectors {
        if let FreshnessServe::Refuse(freshness) = &decision {
            if json {
                semantic_algorithmic_json(
                    &store,
                    &project,
                    "skipped_stale_index",
                    Some(freshness),
                    &[],
                )?;
            } else {
                println!(
                    "{}",
                    semantic_stale_skip_message("semantic-search", freshness)
                );
            }
            return Ok(1);
        }
    }
    let freshness = decision.freshness().clone();

    if provider_policy_blocks_query(&incomplete_providers)? {
        if json {
            semantic_provider_incomplete_json(
                &project,
                if vectors { "vector" } else { "algorithmic" },
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

    let vector_config = if vectors {
        Some(embedding_config_required(embedding_args)?)
    } else {
        None
    };

    if let Some(cfg) = vector_config {
        let generation = current_graph_generation(&store, root)?;
        let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
            &project,
            &cfg.model_id,
            Some(generation),
            20,
        );
        let total = greppy_search::count_vector_search_scope(&store, &scope)?;
        let candidate_limit = vector_exact_candidate_limit()?;
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
                    "(no vector embeddings for model {}; run `grep index --embeddings ...` first)",
                    cfg.model_id
                );
            }
            return Ok(1);
        }
        if !freshness_json_is_fresh(&freshness) {
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
                    vector_stale_skip_message("semantic-search --vectors", &freshness)
                );
            }
            return Ok(1);
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
                    vector_exact_scan_skip_message("semantic-search --vectors", total, limit)
                );
            }
            return Ok(1);
        }

        match embed_query_cached(&cfg, root, q) {
            Ok(query_vector) => {
                scope.limit = 20;
                let hits = greppy_search::vector_search_exact(&store, &query_vector, &scope)?;
                let expand = insert_semantic_vector_expand_pack(
                    &store, root, &project, q, generation, &hits,
                );
                if json {
                    semantic_vector_json_with_expand(
                        &store,
                        &project,
                        &cfg,
                        generation,
                        total,
                        candidate_limit,
                        Some(&freshness),
                        "ok",
                        &hits,
                        expand.as_ref(),
                    )?;
                } else if hits.is_empty() {
                    println!("(no vector matches)");
                    return Ok(1);
                } else {
                    println!(
                        "# semantic mode: vector (EmbeddingGemma code retrieval; model {}; generation {})",
                        cfg.model_id, generation
                    );
                    for h in &hits {
                        println!(
                            "{:.3}  {}  {}:{}-{}  [vector]",
                            h.score,
                            h.embedding.qualified_name,
                            h.embedding.file_path,
                            h.embedding.start_line,
                            h.embedding.end_line
                        );
                    }
                    if let Some(expand) = &expand {
                        println!("{}", expand.text_line());
                    }
                }
                return Ok(if hits.is_empty() { 1 } else { 0 });
            }
            Err(e) => {
                log_embedding_skip_once("semantic-search --vectors", &e);
            }
        }
    }

    let hits = greppy_search::semantic_query(&store, q, None, Some(&project), 20)?;
    let expand = insert_semantic_algorithmic_expand_pack(&store, root, &project, q, &hits);
    if json {
        semantic_algorithmic_json_with_expand(
            &store,
            &project,
            "ok",
            Some(&freshness),
            &hits,
            expand.as_ref(),
        )?;
        return Ok(if hits.is_empty() { 1 } else { 0 });
    }
    if hits.is_empty() {
        println!("(no matches)");
    } else {
        println!(
            "# semantic mode: algorithmic (TF-IDF + MinHash; pass --vectors for EmbeddingGemma)"
        );
        for h in &hits {
            let mut flags = Vec::new();
            if h.signals.token_overlap {
                flags.push("tok");
            }
            if h.signals.label_affinity {
                flags.push("lbl");
            }
            if h.signals.file_proximity {
                flags.push("file");
            }
            println!(
                "{:.3}  {}  {}  {}  [{}]",
                h.score,
                h.node.label,
                h.node.qualified_name,
                line_span(&h.node.file_path, h.node.start_line, h.node.end_line),
                flags.join(",")
            );
        }
        if let Some(expand) = &expand {
            println!("{}", expand.text_line());
        }
    }
    Ok(0)
}

fn current_graph_generation(store: &greppy_store::Store, root: Option<&str>) -> Result<u64> {
    let root_path = resolve_root(root)?;
    let root_key = root_path.to_string_lossy().into_owned();
    let state = store.get_workspace_state(&root_key)?.ok_or_else(|| {
        Error::Invalid(format!(
            "no workspace_state for {}; run `grep index {}` first",
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
    semantic_vector_json_with_expand(
        store,
        project,
        cfg,
        graph_generation,
        total,
        candidate_limit,
        freshness,
        status,
        hits,
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
    candidate_limit: Option<i64>,
    freshness: Option<&serde_json::Value>,
    status: &str,
    hits: &[greppy_store::VectorSearchHit],
    expand: Option<&ExpandHandle>,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "score": h.score,
                "qualified_name": h.embedding.qualified_name,
                "file_path": h.embedding.file_path,
                "start_line": h.embedding.start_line,
                "end_line": h.embedding.end_line,
                "content_sha256": h.embedding.content_sha256,
                "graph_generation": h.embedding.graph_generation,
            })
        })
        .collect::<Vec<_>>();
    let shown = rows.len() as i64;
    let mut v = serde_json::json!({
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
        "total_exact": total,
        "shown": shown,
        "omitted": total.saturating_sub(shown),
        "truncated": shown < total,
        "hits": rows,
    });
    if let Some(expand) = expand {
        v["expand"] = expand.json_value();
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize vector semantic JSON: {e}")))?
    );
    Ok(())
}

fn semantic_algorithmic_json(
    store: &greppy_store::Store,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    hits: &[greppy_search::SemanticHit],
) -> Result<()> {
    semantic_algorithmic_json_with_expand(store, project, status, freshness, hits, None)
}

fn semantic_algorithmic_json_with_expand(
    store: &greppy_store::Store,
    project: &str,
    status: &str,
    freshness: Option<&serde_json::Value>,
    hits: &[greppy_search::SemanticHit],
    expand: Option<&ExpandHandle>,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let rows = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "score": h.score,
                "label": h.node.label,
                "qualified_name": h.node.qualified_name,
                "file_path": h.node.file_path,
                "start_line": h.node.start_line,
                "end_line": h.node.end_line,
                "signals": {
                    "token_overlap": h.signals.token_overlap,
                    "label_affinity": h.signals.label_affinity,
                    "file_proximity": h.signals.file_proximity,
                    "simhash": h.signals.simhash,
                    "qname_path": h.signals.qname_path,
                    "edge_proximity": h.signals.edge_proximity,
                }
            })
        })
        .collect::<Vec<_>>();
    let mut v = serde_json::json!({
        "command": "semantic-search",
        "mode": "algorithmic",
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
        "total_exact": rows.len(),
        "shown": rows.len(),
        "omitted": 0,
        "truncated": false,
        "hits": rows,
    });
    if let Some(expand) = expand {
        v["expand"] = expand.json_value();
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&v)
            .map_err(|e| Error::Invalid(format!("serialize semantic JSON: {e}")))?
    );
    Ok(())
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
    let k = k.max(1);
    let project = project_for(root)?;
    let span_root = resolve_root(root)?;
    // D2 fail-open: refuse only when no usable index exists; a stale
    // index serves labeled results (auto-healed first when small).
    // Span bodies are read from the CURRENT files on disk, so even
    // labeled-stale context output never shows outdated code — at worst
    // a stale line number drifts, which read_span skips gracefully.
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
        return Ok(1);
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
        for h in greppy_search::search_code(&store, &project, q, fetch)? {
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
    // Model configuration is OPTIONAL here: with no model the command must
    // behave exactly as before (lexical-only), just with a clear note so the
    // operator knows the semantic lever was available but unconfigured.
    let cfg = match embedding_config_optional(embedding_args)? {
        Some(cfg) => cfg,
        None => {
            eprintln!(
                "context: no EmbeddingGemma model configured; skipping vector \
                 fallback for the natural-language query (set \
                 {ENV_EMBED_MODEL_DIR}, or {ENV_EMBED_GGUF}+{ENV_EMBED_TOKENIZER}, \
                 to enable semantic discovery). Returning exact/FTS results only."
            );
            return Ok(None);
        }
    };

    let generation = current_graph_generation(store, root)?;
    let mut scope = greppy_search::embeddinggemma_code_retrieval_scope(
        project,
        &cfg.model_id,
        Some(generation),
        fetch,
    );
    let mut total = greppy_search::count_vector_search_scope(store, &scope)?;
    if total == 0 {
        // P2 (problem dossier): a resolvable model with a vector-less store
        // used to degrade to lexical/FTS SILENTLY — the agent got junk
        // results with no signal (spot-test forensics: pnpm-lock keys as
        // top "semantic" hits). Self-heal instead: build the embeddings
        // now (first semantic use), announced honestly. A second store
        // handle is opened for the write so this read path keeps &Store.
        eprintln!(
            "context: building the semantic index for this project (first \
             use, model {}) — subsequent queries are instant.",
            cfg.model_id
        );
        // P11: a large project's first embedding build takes minutes and
        // used to BLOCK this call. Only build inline when the graph is
        // small enough to finish quickly; otherwise kick off a detached
        // background build and answer THIS query from lexical/FTS now, so
        // the agent never waits. The next query finds the vectors ready.
        if project_is_small_enough_for_inline_embed(store, project) {
            if build_vectors_first_use(&cfg, root, project, generation) {
                total = greppy_search::count_vector_search_scope(store, &scope)?;
            }
        } else {
            spawn_background_embed(root);
            eprintln!(
                "context: this project is large — building the semantic index \
                 in the background; answering from lexical results for now. \
                 Retry in a moment for semantic results."
            );
            return Ok(None);
        }
        if total == 0 {
            eprintln!(
                "context: no indexed vectors for model {} (run `grep index \
                 --embeddings ...`); skipping vector fallback.",
                cfg.model_id
            );
            return Ok(None);
        }
        // The freshly built vectors carry the current generation; re-derive
        // the scope so the freshness/generation filters line up.
        scope = greppy_search::embeddinggemma_code_retrieval_scope(
            project,
            &cfg.model_id,
            Some(current_graph_generation(store, root)?),
            fetch,
        );
    }
    if !freshness_json_is_fresh(freshness) {
        eprintln!(
            "{}",
            vector_stale_skip_message("context --vectors", freshness)
        );
        return Ok(None);
    }
    let candidate_limit = vector_exact_candidate_limit()?;
    if let Some(limit) = vector_exact_scan_exceeds_limit(total, candidate_limit) {
        eprintln!(
            "{}",
            vector_exact_scan_skip_message("context --vectors", total, limit)
        );
        return Ok(None);
    }

    let query_vector = match embed_query_cached(&cfg, root, query) {
        Ok(query_vector) => query_vector,
        Err(e) => {
            log_embedding_skip_once("context --vectors", &e);
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
const ROOT_MARKERS: [&str; 3] = [".git", "Cargo.toml", "pyproject.toml"];

/// Resolve the effective workspace root for a command.
///
/// * If `--root <PATH>` was given, use it verbatim (the user is
///   explicit).
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
        return Ok(absolutize_path(std::path::Path::new(r)));
    }
    let cwd = std::env::current_dir()
        .map_err(|e| Error::io("read current_dir for root resolution", e))?;
    Ok(find_repo_root(&cwd))
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
    let mut cur: &std::path::Path = start;
    loop {
        if ROOT_MARKERS.iter().any(|m| cur.join(m).exists()) {
            return cur.to_path_buf();
        }
        match cur.parent() {
            Some(p) if p != cur => cur = p,
            _ => return start.to_path_buf(),
        }
    }
}

/// Compute the project identity string for the effective root
/// (`--root` if given, else the detected repo root). Centralised so
/// every command uses the same definition (RV-011).
fn project_for(root: Option<&str>) -> Result<String> {
    let p = resolve_root(root)?;
    Ok(workspace_locator::project_identity(&p))
}

fn embedding_config_for_index(args: EmbeddingCliArgs<'_>) -> Result<Option<EmbeddingModelConfig>> {
    let requested = args.enabled || args.has_model_source_arg() || env_bool(ENV_EMBED_INDEX)?;
    if !requested {
        return Ok(None);
    }
    // Out of the box: `index --embeddings` uses the baked-in EmbeddingGemma with
    // NO flags/env, exactly like the query path (embedded_model::paths via
    // embedding_config_optional). Explicit --embedding-* / env still override.
    // Only when embeddings are requested AND no source AND no baked model exists
    // (a non-embedded build) do we surface the clear "model required" error
    // instead of silently skipping the vectors the caller asked for.
    match embedding_config_optional(args)? {
        Some(cfg) => Ok(Some(cfg)),
        None => Ok(Some(embedding_config_required(args)?)),
    }
}

/// Resolve an embedding model config when one is available, without erroring
/// when it is not. Returns `Ok(None)` only when NO model source is configured
/// at all (no `--embedding-*` flag and no `GREPPY_EMBEDDINGGEMMA_*` env),
/// which lets a caller degrade gracefully instead of failing. A partially
/// specified source (e.g. `--embedding-gguf` without a tokenizer) is a real
/// misconfiguration and still surfaces as an error via
/// [`embedding_config_required`].
fn embedding_config_optional(args: EmbeddingCliArgs<'_>) -> Result<Option<EmbeddingModelConfig>> {
    let has_source = cli_or_env(args.model_dir, ENV_EMBED_MODEL_DIR).is_some()
        || cli_or_env(args.gguf, ENV_EMBED_GGUF).is_some()
        || cli_or_env(args.tokenizer, ENV_EMBED_TOKENIZER).is_some();
    if !has_source {
        // Owner rule: semantic search must ALWAYS work. Release binaries
        // carry EmbeddingGemma inside (feature `embedded-model`); when no
        // explicit source is configured, extract it once to the data dir
        // and use it. Env/CLI settings above still override.
        if let Some((gguf, tokenizer)) = embedded_model::paths() {
            let args = EmbeddingCliArgs {
                gguf: Some(&gguf),
                tokenizer: Some(&tokenizer),
                ..args
            };
            return embedding_config_required(args).map(Some);
        }
        return Ok(None);
    }
    embedding_config_required(args).map(Some)
}

/// Built-in EmbeddingGemma (feature `embedded-model`): the Q4_K GGUF and
/// tokenizer are baked into the binary at build time and extracted once
/// to `<data>/greppy/embedded-model/<sha>/` (mmap needs a real file). The
/// extraction is atomic (tmp + rename) and trusts cached files only when their
/// tiny marker file matches the baked SHA and their length matches the baked
/// bytes, so stale or torn cache entries self-repair without hashing the model
/// on every CLI invocation.
mod embedded_model {
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    pub fn paths() -> Option<(String, String)> {
        const GGUF_SHA: &str = env!("GREPPY_EMBEDDED_GGUF_SHA");
        const TOK_SHA: &str = env!("GREPPY_EMBEDDED_TOK_SHA");
        static GGUF: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/embeddinggemma-300M-Q4_K.gguf"));
        static TOK: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tokenizer.json"));
        let root = greppy_core::workspace::store_cache_root()?.join("embedded-model");
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
        let root = root.join(expected_sha);
        let dest = root.join(name);
        let marker = root.join(format!("{name}.sha256"));
        if cache_entry_is_valid(&dest, &marker, expected_sha, bytes.len()) {
            return Some(dest.to_string_lossy().into_owned());
        }

        std::fs::create_dir_all(&root).ok()?;
        if cache_entry_is_valid(&dest, &marker, expected_sha, bytes.len()) {
            return Some(dest.to_string_lossy().into_owned());
        }

        let nonce = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = root.join(format!("{name}.tmp.{}.{}", std::process::id(), nonce));
        let marker_tmp = root.join(format!(
            "{name}.sha256.tmp.{}.{}",
            std::process::id(),
            nonce
        ));

        let written =
            write_verified_cache_entry(&tmp, &dest, &marker_tmp, &marker, expected_sha, bytes);
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp);
            let _ = std::fs::remove_file(&marker_tmp);
        }

        if cache_entry_is_valid(&dest, &marker, expected_sha, bytes.len()) {
            Some(dest.to_string_lossy().into_owned())
        } else {
            None
        }
    }

    fn cache_entry_is_valid(
        dest: &std::path::Path,
        marker: &std::path::Path,
        expected_sha: &str,
        expected_len: usize,
    ) -> bool {
        let marker_ok = std::fs::read_to_string(marker)
            .map(|s| s.trim() == expected_sha)
            .unwrap_or(false);
        if !marker_ok {
            return false;
        }
        std::fs::metadata(dest)
            .map(|m| m.len() == expected_len as u64)
            .unwrap_or(false)
    }

    fn write_verified_cache_entry(
        tmp: &std::path::Path,
        dest: &std::path::Path,
        marker_tmp: &std::path::Path,
        marker: &std::path::Path,
        expected_sha: &str,
        bytes: &[u8],
    ) -> std::io::Result<()> {
        std::fs::write(tmp, bytes)?;
        ensure_len(tmp, bytes.len())?;

        // Invalidate before replacing the data file, so readers never trust an
        // old marker for bytes that are currently being repaired.
        let _ = std::fs::remove_file(marker);
        let _ = std::fs::remove_file(dest);
        std::fs::rename(tmp, dest)?;
        ensure_len(dest, bytes.len())?;

        std::fs::write(marker_tmp, expected_sha.as_bytes())?;
        std::fs::rename(marker_tmp, marker)?;
        Ok(())
    }

    fn ensure_len(path: &std::path::Path, expected_len: usize) -> std::io::Result<()> {
        let len = std::fs::metadata(path)?.len();
        if len == expected_len as u64 {
            return Ok(());
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "embedded model cache entry {} has length {len}, expected {expected_len}",
                path.display()
            ),
        ))
    }
}

fn embedding_config_required(args: EmbeddingCliArgs<'_>) -> Result<EmbeddingModelConfig> {
    let model_dir = cli_or_env(args.model_dir, ENV_EMBED_MODEL_DIR);
    let gguf = cli_or_env(args.gguf, ENV_EMBED_GGUF);
    let tokenizer = cli_or_env(args.tokenizer, ENV_EMBED_TOKENIZER);
    let model_id = cli_or_env(args.model_id, ENV_EMBED_MODEL_ID)
        .unwrap_or_else(|| DEFAULT_EMBEDDINGGEMMA_MODEL_ID.to_string());
    let max_length = match args.max_length {
        Some(v) => Some(v),
        None => env_nonempty(ENV_EMBED_MAX_LENGTH)
            .map(|s| {
                s.parse::<usize>().map_err(|_| {
                    Error::Invalid(format!("{ENV_EMBED_MAX_LENGTH} must be a positive integer"))
                })
            })
            .transpose()?,
    };
    let device = embedding_device_preference(args.device, args.no_gpu)?;

    let source = match (model_dir, gguf, tokenizer) {
        (Some(dir), None, None) => EmbeddingModelSource::SafetensorsDir(dir.into()),
        (None, Some(gguf), Some(tokenizer)) => EmbeddingModelSource::Gguf {
            gguf: gguf.into(),
            tokenizer: tokenizer.into(),
        },
        (Some(_), Some(_), _) | (Some(_), None, Some(_)) => {
            return Err(Error::Invalid(
                "configure either --embedding-model-dir or --embedding-gguf/--embedding-tokenizer, not both"
                    .into(),
            ));
        }
        (None, Some(_), None) => {
            return Err(Error::Invalid(
                "--embedding-gguf requires --embedding-tokenizer (or GREPPY_EMBEDDINGGEMMA_TOKENIZER)"
                    .into(),
            ));
        }
        (None, None, Some(_)) => {
            return Err(Error::Invalid(
                "--embedding-tokenizer requires --embedding-gguf (or GREPPY_EMBEDDINGGEMMA_GGUF)"
                    .into(),
            ));
        }
        (None, None, None) => {
            return Err(Error::Invalid(format!(
                "EmbeddingGemma model required: pass --embedding-model-dir, or --embedding-gguf with --embedding-tokenizer, or set {ENV_EMBED_MODEL_DIR}/{ENV_EMBED_GGUF}/{ENV_EMBED_TOKENIZER}"
            )));
        }
    };
    Ok(EmbeddingModelConfig {
        model_id,
        source,
        max_length,
        device,
    })
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
fn load_embedding_model(
    cfg: &EmbeddingModelConfig,
    tokenizer_cache_dir: Option<std::path::PathBuf>,
) -> Result<greppy_embed_native::EmbeddingGemma> {
    let options = greppy_embed_native::LoadOptions {
        device: cfg.device.clone(),
        max_length: cfg.max_length,
        tokenizer_cache_dir,
    };
    match &cfg.source {
        EmbeddingModelSource::SafetensorsDir(_) => Err(greppy_embed_native::Error::InvalidGguf(
            "native EmbeddingGemma supports GGUF + tokenizer.json only; configure --embedding-gguf with --embedding-tokenizer".into(),
        )),
        EmbeddingModelSource::Gguf { gguf, tokenizer } => {
            greppy_embed_native::EmbeddingGemma::load_gguf(gguf, tokenizer, options)
        }
    }
    .map_err(|e| Error::Store(format!("load EmbeddingGemma model {}: {e}", cfg.model_id)))
}

/// Cache key for query embeddings: logical model id + prompt/task
/// contract + a (len, mtime) fingerprint of the model source files, so
/// swapping the GGUF/tokenizer/safetensors invalidates cached vectors
/// even when the logical model id stays the same.
fn embedding_query_cache_key(cfg: &EmbeddingModelConfig) -> String {
    fn file_fp(path: &std::path::Path) -> String {
        match std::fs::metadata(path) {
            Ok(meta) => {
                let mtime_ns = meta
                    .modified()
                    .ok()
                    .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                format!("{}:{}:{}", path.display(), meta.len(), mtime_ns)
            }
            Err(_) => format!("{}:unknown", path.display()),
        }
    }
    let source_fp = match &cfg.source {
        EmbeddingModelSource::SafetensorsDir(dir) => {
            format!("st;{}", file_fp(&dir.join("model.safetensors")))
        }
        EmbeddingModelSource::Gguf { gguf, tokenizer } => {
            format!("gguf;{};{}", file_fp(gguf), file_fp(tokenizer))
        }
    };
    format!(
        "{}|{}|{}|{}",
        cfg.model_id,
        greppy_embed_native::PROMPT_VERSION,
        greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE,
        source_fp
    )
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
    // freed after its idle TTL); ANY daemon problem falls back to the
    // in-process load below. Embed the NORMALIZED text either way so the
    // cached vector is exactly the vector any query normalizing to the
    // same key would compute.
    #[cfg(unix)]
    let daemon_vector = embed_daemon::embed_query_via_daemon(cfg, &model_key, &normalized);
    #[cfg(not(unix))]
    let daemon_vector: Option<Vec<f32>> = None;
    let vector = match daemon_vector {
        Some(v) => v,
        None => {
            let model = load_embedding_model(cfg, store_dir)?;
            greppy_search::embed_code_query(&model, &normalized)?
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

fn cli_or_env(cli: Option<&str>, env: &str) -> Option<String> {
    cli.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| env_nonempty(env))
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

fn semantic_stale_skip_message(command: &str, freshness: &serde_json::Value) -> String {
    format!(
        "{command}: algorithmic semantic search skipped because {}; no stale indexed hits emitted",
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
    // commands (who-calls / callees / find-usages / references / impact /
    // path / brief / fan-in / fan-out / trace) Just Work on a fresh repo.
    // Gated behind the same kill switch as the inline auto-reindex
    // (`GREPPY_AUTO_REINDEX=0` restores the old hard error). If the
    // inline index fails for any reason we fall back to the actionable
    // diagnostic below. Embeddings are NOT computed here — the vector
    // path (`context` / `semantic`) still asks for `grep index
    // --embeddings` when it needs vectors.
    // (Query commands only — the grep passthrough path never reaches here,
    // so the byte-exact passthrough contract is untouched.)
    if !path.exists() {
        let shown_root = root.unwrap_or(".");
        if auto_reindex_enabled() {
            eprintln!("grep: indexing {} (first use)…", effective_root.display());
            if try_auto_index_inline(root) && path.exists() {
                // Index built: fall through to the normal read-only open.
            } else {
                eprintln!(
                    "grep: no index for {} — run `grep index {}` first",
                    effective_root.display(),
                    shown_root
                );
                return Err(Error::Invalid(format!(
                    "no index for {}; run `grep index {}` first",
                    effective_root.display(),
                    shown_root
                )));
            }
        } else {
            eprintln!(
                "grep: no index for {} — run `grep index {}` first",
                effective_root.display(),
                shown_root
            );
            return Err(Error::Invalid(format!(
                "no index for {}; run `grep index {}` first",
                effective_root.display(),
                shown_root
            )));
        }
    }
    // Query commands are READ-ONLY: open read-only so they skip both
    // `migrate()` and the O(db-size) `integrity_check` that a read-write open
    // runs. Those belong on the writer (`greppy index`); paying them on every
    // query open made who-calls/find-usages/search take seconds on a real repo
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
    #[cfg(unix)]
    {
        let no_args = EmbeddingCliArgs {
            enabled: false,
            model_dir: None,
            gguf: None,
            tokenizer: None,
            model_id: None,
            max_length: None,
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

/// Feature A helper — build a fresh GRAPH index (no embeddings) for the
/// workspace at `root` on an empty/absent store, holding the writer
/// lock. Returns true when the index was written cleanly.
///
/// This reuses the same inline-index machinery as
/// [`try_auto_reindex_inline`]: it opens (creating) the store at the
/// workspace's `store_path` and runs `greppy_indexer::index_with_options`,
/// which builds the graph without touching embeddings. Any failure (lock
/// contention, read-only store dir, indexer error) reports false so the
/// caller degrades to the actionable "run `grep index`" diagnostic.
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
    // Ensure the store dir exists (0700) before we try to create the DB.
    if let Some(parent) = store_path.parent() {
        if workspace_locator::ensure_store_dir(parent).is_err() {
            return false;
        }
    }
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(LockOutcome::Acquired | LockOutcome::AcquiredFromStale) => {
            greppy_freshness::Lock::new(greppy_freshness::lock_path_for(&store_path))
        }
        _ => return false, // another writer is active
    };
    // A concurrent invocation may have built the index while we waited
    // for the lock — if the DB now exists, treat that as success.
    let Ok(mut store) = greppy_store::Store::open(&store_path) else {
        return false;
    };
    let _ = workspace_locator::ensure_db_mode(&store_path);
    let options = greppy_indexer::IndexOptions {
        discover_overrides: overrides,
    };
    match greppy_indexer::index_with_options(&mut store, &effective_root, &project, &options) {
        Ok(report) => report.is_clean(),
        Err(_) => false,
    }
}

fn dispatch_grep(argv: &[String]) -> Result<i32> {
    // clap's `trailing_var_arg` captures everything after `greppy`
    // (or `greppy <unknown_subcmd>`). For the `greppy grep …`
    // invocation, argv[0] is the placeholder "grep" which real grep
    // would otherwise see as a positional file argument — so we strip
    // a leading grep-family placeholder before forwarding.
    //
    // After stripping, argv contains only real grep arguments. We
    // build a synthetic argv where argv[0] is a binary-name placeholder
    // and argv[1..] is the user's args, then call
    // `greppy_grep::run::run_with_optional_augment` which both runs
    // real grep and applies the heuristic + freshness augmentation.
    let stripped: &[String] = match argv.first().map(|s| s.as_str()) {
        Some("grep") | Some("egrep") | Some("fgrep") | Some("rgrep") => &argv[1..],
        _ => argv,
    };

    let mut full = Vec::with_capacity(stripped.len() + 1);
    full.push("greppy-grep".to_string());
    full.extend_from_slice(stripped);

    let parsed = greppy_grep::heuristic::GrepArgs::parse(&full[1..]);
    let real = greppy_grep::discover_grep()?;
    greppy_grep::run::run_with_optional_augment(&real, &full, &parsed)
}

/// `OsString` argv variant of [`dispatch_grep`].
///
/// forwards the original (possibly non-UTF-8) argv to
/// real grep byte-for-byte via the shared
/// [`greppy_grep::run::run_with_optional_augment_os`] path. `full`
/// includes a synthetic argv[0] placeholder; `full[1..]` are the user's
/// grep arguments. A leading grep-family placeholder (when the user
/// wrote `greppy grep …`) is handled by the pre-clap router, which
/// only routes here for the *bare* form — but we still strip a leading
/// `grep`/`egrep`/… token defensively to match [`dispatch_grep`].
fn dispatch_grep_os(full: &[std::ffi::OsString]) -> Result<i32> {
    // full[0] is the "greppy-grep" placeholder. Strip a leading
    // grep-family placeholder in full[1] if present (mirrors the String
    // dispatch_grep behaviour) so `greppy grep -R foo .` and
    // `greppy -R foo .` agree.
    let args: &[std::ffi::OsString] = &full[1..];
    let stripped: &[std::ffi::OsString] = match args.first().and_then(|s| s.to_str()) {
        Some("grep") | Some("egrep") | Some("fgrep") | Some("rgrep") => &args[1..],
        _ => args,
    };

    let mut rebuilt: Vec<std::ffi::OsString> = Vec::with_capacity(stripped.len() + 1);
    rebuilt.push(std::ffi::OsString::from("greppy-grep"));
    rebuilt.extend_from_slice(stripped);

    let parsed = greppy_grep::heuristic::GrepArgs::parse_os(&rebuilt[1..]);
    let real = greppy_grep::discover_grep()?;
    greppy_grep::run::run_with_optional_augment_os(&real, &rebuilt, &parsed)
}

/// Run the indexer against `path` (default: current directory).
fn dispatch_index(
    path: Option<&str>,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
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
        Some(r) => absolutize_path(std::path::Path::new(r)),
        None => find_repo_root(&target),
    };
    let project = workspace_locator::project_identity(&effective_root);
    let index_options = greppy_indexer::IndexOptions {
        discover_overrides: discover_overrides_from_env()?,
    };
    let embedding_config = embedding_config_for_index(embedding_args)?;

    // Open the on-disk store under the workspace locator's path
    // never at `<root>/.greppy/graph.db` (which would
    // pollute `grep -R .`). Default is `$XDG_CACHE_HOME/greppy/<ws-hash>/graph.db`
    // on Linux, `~/Library/Caches/greppy/<ws-hash>/graph.db` on macOS;
    // overridable via `GREPPY_STORE_DIR`.
    let store_path = workspace_locator::store_path(&effective_root);
    if let Some(parent) = store_path.parent() {
        workspace_locator::ensure_store_dir(parent)
            .map_err(|e| Error::io(format!("create store dir {}", parent.display()), e))?;
    }
    // Acquire the crash-safe
    // advisory lock BEFORE opening/migrating the store. Opening first lets a
    // concurrent indexer hit a SQLite busy error inside Store::open and exit
    // EXIT_IO (73) silently, instead of the documented EX_TEMPFAIL (75) with a
    // diagnostic on contention. Concurrent indexers on the same path get
    // `LockError::Held`; a crashed prior holder's lock is taken over via the
    // stale-recovery path. The lock file path is derived inside `try_acquire`
    // from `target` (i.e. `<store_path>.lock`); the RAII handle must wrap the
    // SAME path the lock was created on and must outlive the store mutation.
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(LockOutcome::Acquired | LockOutcome::AcquiredFromStale) => Some(
            greppy_freshness::Lock::new(greppy_freshness::lock_path_for(&store_path)),
        ),
        Ok(LockOutcome::Contended) => {
            eprintln!(
                "grep: another indexer is running against {}",
                store_path.display()
            );
            return Ok(EXIT_TEMPFAIL as i32);
        }
        Err(greppy_freshness::LockError::Held { pid, age_secs, .. }) => {
            eprintln!(
                "grep: lock held by another writer (pid {:?}, age {:?}s) on {}",
                pid,
                age_secs,
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
    let snapshot = index_atomic_snapshot(
        &store_path,
        &target,
        &project,
        embedding_config.as_ref(),
        &index_options,
    )?;
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
    if !report.is_clean() {
        return Ok(EXIT_IO as i32);
    }
    if let Some(embedding_report) = &snapshot.embeddings {
        println!(
            "embedded {} code spans ({} considered, {} non-definition skipped, {} missing-file, {} invalid-span, {} oversize, {} stale pruned)",
            embedding_report.nodes_embedded,
            embedding_report.nodes_considered,
            embedding_report.nodes_skipped_non_definition,
            embedding_report.nodes_skipped_missing_file,
            embedding_report.nodes_skipped_invalid_span,
            embedding_report.nodes_skipped_oversize,
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
    Ok(0)
}

struct IndexSnapshotReport {
    index: greppy_indexer::IndexReport,
    embeddings: Option<greppy_indexer::EmbeddingIndexReport>,
}

fn index_atomic_snapshot(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
) -> Result<IndexSnapshotReport> {
    cleanup_stale_next_snapshots(active_path)?;
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

    if !report.is_clean() {
        drop(temp_store);
        cleanup_sqlite_family(&temp_path)?;
        return Ok(IndexSnapshotReport {
            index: report,
            embeddings: None,
        });
    }

    let embedding_report = if let Some(cfg) = embedding_config {
        match index_embeddings_into_temp_store(
            &mut temp_store,
            target,
            project,
            cfg,
            &report,
            active_path.parent().map(std::path::Path::to_path_buf),
        ) {
            Ok(report) => Some(report),
            Err(e) => {
                drop(temp_store);
                let _ = cleanup_sqlite_family(&temp_path);
                return Err(e);
            }
        }
    } else {
        None
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
    maybe_index_test_failpoint("after-temp-before-publish", &temp_path)?;

    publish_store_snapshot(&temp_path, active_path)?;
    Ok(IndexSnapshotReport {
        index: report,
        embeddings: embedding_report,
    })
}

fn index_embeddings_into_temp_store(
    store: &mut greppy_store::Store,
    target: &std::path::Path,
    project: &str,
    cfg: &EmbeddingModelConfig,
    report: &greppy_indexer::IndexReport,
    tokenizer_cache_dir: Option<std::path::PathBuf>,
) -> Result<greppy_indexer::EmbeddingIndexReport> {
    let model = match load_embedding_model(cfg, tokenizer_cache_dir) {
        Ok(model) => model,
        Err(e) => {
            log_embedding_skip_once("index --embeddings", &e);
            return Ok(greppy_indexer::EmbeddingIndexReport::default());
        }
    };
    let mut provider = greppy_indexer::EmbeddingGemmaCodeProvider::new(&cfg.model_id, &model);
    greppy_indexer::index_code_embeddings_for_project(
        store,
        target,
        project,
        &mut provider,
        greppy_indexer::EmbeddingIndexOptions::for_generation(report.graph_generation),
    )
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
    sync_parent_dir(active_path)?;
    Ok(())
}

fn active_snapshot_is_recoverable(error: &Error) -> bool {
    matches!(error, Error::Store(_))
}

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

fn cleanup_stale_next_snapshots(active_path: &std::path::Path) -> Result<usize> {
    let Some(parent) = active_path.parent() else {
        return Ok(0);
    };
    let Some(file_name) = active_path.file_name().and_then(|s| s.to_str()) else {
        return Ok(0);
    };
    let prefix = format!("{file_name}.next.");
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
        if !name.starts_with(&prefix) {
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
    dir.sync_all()
        .map_err(|e| Error::io(format!("sync parent dir {}", parent.display()), e))
}

fn sync_file(path: &std::path::Path) -> Result<()> {
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
    match dispatch(cli) {
        Ok(code) => code.clamp(0, 255) as u8,
        Err(e) => {
            eprintln!("grep: {e}");
            let mut source = std::error::Error::source(&e);
            while let Some(cause) = source {
                eprintln!("  caused by: {cause}");
                source = cause.source();
            }
            match e {
                Error::NotImplemented { .. } | Error::OutOfScope { .. } => EXIT_NOT_IMPLEMENTED,
                Error::Invalid(_) => EXIT_USAGE,
                _ => EXIT_IO,
            }
        }
    }
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
            "indexer version/scope changed (was greppy-indexer-v1, expected greppy-indexer-v2)"
        )));
        // Same non-default scope, version bumped → self-heal.
        assert!(version_drift_is_scope_stable(&drift_json(
            "indexer version/scope changed (was greppy-indexer-v1;discover_scope=I8:src/*.rs, \
             expected greppy-indexer-v2;discover_scope=I8:src/*.rs)"
        )));
    }

    #[test]
    fn scope_change_is_not_scope_stable() {
        // Different discover scope → NOT stable → refuse (fail-closed).
        assert!(!version_drift_is_scope_stable(&drift_json(
            "indexer version/scope changed (was greppy-indexer-v2;discover_scope=I8:src/*.rs, \
             expected greppy-indexer-v2)"
        )));
        // Version bump AND scope change → scope change dominates → refuse.
        assert!(!version_drift_is_scope_stable(&drift_json(
            "indexer version/scope changed (was greppy-indexer-v1, \
             expected greppy-indexer-v2;discover_scope=I8:src/*.rs)"
        )));
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
    fn parse_index_embedding_flags() {
        let cli = Cli::try_parse_from([
            "greppy",
            "index",
            "--embeddings",
            "--embedding-gguf",
            "model.gguf",
            "--embedding-tokenizer",
            "tokenizer.json",
            "--embedding-model-id",
            "google/embeddinggemma-300m-q4",
            ".",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Index {
                path,
                embeddings,
                embedding_gguf,
                embedding_tokenizer,
                embedding_model_id,
                ..
            }) => {
                assert_eq!(path.as_deref(), Some("."));
                assert!(embeddings);
                assert_eq!(embedding_gguf.as_deref(), Some("model.gguf"));
                assert_eq!(embedding_tokenizer.as_deref(), Some("tokenizer.json"));
                assert_eq!(
                    embedding_model_id.as_deref(),
                    Some("google/embeddinggemma-300m-q4")
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parse_semantic_vector_flags() {
        let cli = Cli::try_parse_from([
            "greppy",
            "semantic-search",
            "--vectors",
            "--json",
            "--embedding-model-dir",
            "/models/embeddinggemma",
            "retry handler",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Semantic {
                query,
                vectors,
                json,
                embedding_model_dir,
                ..
            }) => {
                assert_eq!(query.as_deref(), Some("retry handler"));
                assert!(vectors);
                assert!(json);
                assert_eq!(
                    embedding_model_dir.as_deref(),
                    Some("/models/embeddinggemma")
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }

        let cli = Cli::try_parse_from(["greppy", "semantic", "retry handler"]).unwrap();
        match cli.command {
            Some(Command::Semantic { query, .. }) => {
                assert_eq!(query.as_deref(), Some("retry handler"));
            }
            other => panic!("unexpected command for semantic alias: {other:?}"),
        }
    }

    #[test]
    fn parse_plus_vector_flags() {
        let cli = Cli::try_parse_from([
            "greppy",
            "plus",
            "--json",
            "--vectors",
            "--embedding-gguf",
            "model.gguf",
            "--embedding-tokenizer",
            "tokenizer.json",
            "--embedding-model-id",
            "google/embeddinggemma-300m-q4",
            "--k",
            "5",
            "refund workflow",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Plus {
                query,
                k,
                json,
                vectors,
                embedding_gguf,
                embedding_tokenizer,
                embedding_model_id,
                ..
            }) => {
                assert_eq!(query.as_deref(), Some("refund workflow"));
                assert_eq!(k, 5);
                assert!(json);
                assert!(vectors);
                assert_eq!(embedding_gguf.as_deref(), Some("model.gguf"));
                assert_eq!(embedding_tokenizer.as_deref(), Some("tokenizer.json"));
                assert_eq!(
                    embedding_model_id.as_deref(),
                    Some("google/embeddinggemma-300m-q4")
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn embedding_config_rejects_conflicting_model_sources() {
        let err = embedding_config_required(EmbeddingCliArgs {
            enabled: true,
            model_dir: Some("/models/safetensors"),
            gguf: Some("/models/model.gguf"),
            tokenizer: Some("/models/tokenizer.json"),
            model_id: None,
            max_length: None,
            device: None,
            no_gpu: false,
        })
        .unwrap_err();
        assert!(
            matches!(err, Error::Invalid(msg) if msg.contains("either --embedding-model-dir or --embedding-gguf"))
        );
    }

    #[test]
    fn cli_device_flags_parse_on_embedding_commands() {
        let cli = Cli::try_parse_from([
            "grep",
            "semantic-search",
            "--vectors",
            "--device",
            "cuda",
            "--no-gpu",
            "refund workflow",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Semantic {
                query,
                vectors,
                device,
                no_gpu,
                ..
            }) => {
                assert_eq!(query.as_deref(), Some("refund workflow"));
                assert!(vectors);
                assert_eq!(device.as_deref(), Some("cuda"));
                assert!(no_gpu);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn embedding_device_preference_obeys_cli_and_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore::capture(&[ENV_DEVICE, ENV_NO_GPU]);
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
    fn dispatch_returns_out_of_scope_for_install() {
        let cli = Cli::try_parse_from(["greppy", "install"]).unwrap();
        let r = dispatch(cli);
        assert!(matches!(
            r,
            Err(Error::OutOfScope { ref feature }) if feature == "grep install"
        ));
    }

    #[test]
    fn dispatch_to_code_maps_errors() {
        // `search-graph` is implemented and no longer returns
        // NotImplemented. Verify the not-implemented and usage paths still map
        // to the documented exit codes.
        let cli = Cli::try_parse_from(["greppy", "install"]).unwrap();
        assert_eq!(dispatch_to_code(cli), EXIT_NOT_IMPLEMENTED);

        let cli = Cli::try_parse_from(["greppy", "config"]).unwrap();
        assert_eq!(dispatch_to_code(cli), EXIT_NOT_IMPLEMENTED);

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
                code: _,
                all: _,
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
                code: _,
                all: _,
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
                code: _,
                all: _,
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
                code: _,
                all: _,
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
                symbol,
                code: _,
                direction,
                edge,
                depth,
                since,
                base,
                all,
                json,
            }) => {
                assert!(symbol.is_none());
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
    fn who_calls_and_find_usages_parse_positional_symbol() {
        let cli = Cli::try_parse_from(["greppy", "who-calls", "do_it"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::WhoCalls { symbol: Some(ref s), code: false, all: false, json: false }) if s == "do_it"
        ));

        let cli = Cli::try_parse_from(["greppy", "find-usages", "Widget"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::FindUsages { symbol: Some(ref s), code: false, all: false, json: false }) if s == "Widget"
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
        assert!(matches!(
            cli.command,
            Some(Command::FanIn { ref edge, limit: 7, json: true }) if edge == "USAGE"
        ));

        let cli = Cli::try_parse_from(["greppy", "fan-out"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::FanOut { ref edge, limit: 20, json: false }) if edge == "CALLS"
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
        assert!(!is_grep_passthrough(&mk(&["greppy", "find-usages", "Foo"])));
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

    /// H2 / D1: a full, low-count, non-truncated result from a COMPLETE
    /// provider carries the honest "(complete)" marker — the signal that lets
    /// a 1-caller answer stop the agent iterating to re-derive the count.
    #[test]
    fn footer_prints_complete_for_full_low_count() {
        let footer = NavFooter {
            noun: "caller",
            total: 1,
            shown: 1,
            provider_incomplete: false,
        };
        // Singular noun for a 1-count answer (the prompt's canonical example).
        // Slim form (P2-iterC): bare count, no "(complete)" prose; a complete
        // provider carries no `+` floor marker.
        assert_eq!(footer.render(false), "— 1 caller");
        // Stale index appends the labeled note.
        assert_eq!(footer.render(true), "— 1 caller (as of last index)");
        // A multi-count complete answer pluralizes.
        let many = NavFooter {
            noun: "caller",
            total: 3,
            shown: 3,
            provider_incomplete: false,
        };
        assert_eq!(many.render(false), "— 3 callers");
        // D1: a per-node edge LIMIT floor is marked with `+`, never "complete".
        let floored = NavFooter {
            noun: "caller",
            total: NAV_EDGE_LIMIT,
            shown: NAV_EDGE_LIMIT,
            provider_incomplete: false,
        };
        assert!(!floored.render(false).contains("complete"));
        assert!(floored.render(false).contains('+'));
    }

    /// H2 / D1: when the provider for this language is known-incomplete, the
    /// count carries a `+` floor marker (may be more) and "complete" never
    /// appears — honest under partial call-graph recall, at 1 char (P2-iterC:
    /// the old ~22-token hedge prose was net-negative overhead).
    #[test]
    fn footer_hedges_when_provider_incomplete() {
        let footer = NavFooter {
            noun: "caller",
            total: 6,
            shown: 6,
            provider_incomplete: true,
        };
        let out = footer.render(false);
        assert_eq!(out, "— 6+ callers");
        assert!(!out.contains("complete"), "{out}");
        // Zero-result floor form for the same partial provider.
        assert_eq!(
            render_zero_nav_footer("caller", true, "java", false),
            "— 0+ callers"
        );
        // Zero-result exact form (complete provider).
        assert_eq!(
            render_zero_nav_footer("usage", false, "", false),
            "— 0 usages"
        );

        // The provider-incompleteness decision reads the same source the JSON
        // path uses: a partial Java provider marks Java rows incomplete, while
        // a non-code (.stderr) provider does not.
        let mut store = store_with_defs(&[("Method", "src/A.java", "A", "m")]);
        seed_provider(&mut store, "java", "partial", &["calls"]);
        seed_provider(
            &mut store,
            "file extension .stderr",
            "unsupported",
            &["calls"],
        );
        let node = store
            .get_node(id_of(&store, "src/A.java", "A", "m"))
            .unwrap()
            .unwrap();
        let (incomplete, lang) =
            nav_target_provider_incomplete(&store, "p", &[&node], "calls").unwrap();
        assert!(incomplete);
        assert_eq!(lang, "java");
    }

    /// P2-N regression: a provider that is `is_incomplete()` ONLY because it
    /// omits exotic edge classes (k8s, gitdiff, semantic, …) but fully supports
    /// CALLS must NOT hedge a who-calls footer. This is the real Rust/serde
    /// shape: status "partial", `calls` supported, only irrelevant classes
    /// unsupported. Before the fix every such footer carried a `+` floor marker
    /// that pushed the agent into a redundant `--all` + grep spiral.
    #[test]
    fn footer_does_not_hedge_when_queried_class_is_supported() {
        let mut store = store_with_defs(&[("Function", "src/a.rs", "", "f")]);
        // Mirrors the indexer's real Rust provider_state: partial (parity gate),
        // but `calls` is a SUPPORTED class — only exotic classes are missing.
        seed_provider(
            &mut store,
            "rust",
            "partial",
            &["tests", "k8s", "gitdiff", "semantic"],
        );
        let node = store
            .get_node(id_of(&store, "src/a.rs", "", "f"))
            .unwrap()
            .unwrap();
        // The blanket completeness check still reports the provider incomplete…
        let states = store.list_provider_states("p").unwrap();
        let rust = states.iter().find(|p| p.language == "rust").unwrap();
        assert!(
            rust.is_incomplete(),
            "exotic-class gaps keep is_incomplete true"
        );
        assert!(rust.supports_edge_class("calls"), "but calls is supported");
        // …yet a CALLS-scoped nav query must NOT hedge (no `+` in the footer).
        let (incomplete, _lang) =
            nav_target_provider_incomplete(&store, "p", &[&node], "calls").unwrap();
        assert!(
            !incomplete,
            "who-calls must not hedge when calls is supported"
        );
        // A query for a class this provider genuinely lacks still hedges.
        seed_provider(&mut store, "rust", "partial", &["usages", "k8s"]);
        let (usage_incomplete, _l) =
            nav_target_provider_incomplete(&store, "p", &[&node], "usages").unwrap();
        assert!(
            usage_incomplete,
            "find-usages hedges when usages unsupported"
        );
    }

    /// LEVER 2a: `impact --all` parses (previously clap ERRORED — no such
    /// flag), and the print cap is lifted only when `all` is set.
    #[test]
    fn impact_all_flag_bypasses_limit() {
        let cli = Cli::try_parse_from(["greppy", "impact", "JsonReader", "--all"]).unwrap();
        match cli.command {
            Some(Command::Impact { symbol, all, .. }) => {
                assert_eq!(symbol.as_deref(), Some("JsonReader"));
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

    /// LEVER 2b: `impact`'s incomplete-provider count/`provider_complete`
    /// exclude non-code snapshot/fixture providers (.stderr/.snap/.xml/no-ext),
    /// so the number matches real code callers — the r061 reconciliation fix.
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

        // The unfiltered set counts ALL six (what the OTHER commands see).
        assert_eq!(incomplete_provider_json(&store, "p").unwrap().len(), 6);

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
}
