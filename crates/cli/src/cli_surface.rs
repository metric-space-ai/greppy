//! The command-line surface: every verb, flag and its help text.
//!
//! Five hundred lines of `#[arg]` and doc comments do not belong in the
//! same file as the code that answers the commands.

use super::*;
use clap::{Args, Parser, Subcommand, ValueEnum};

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
    /// query subcommand (search-graph / trace / search-pattern /
    /// search-symbol / search) key the on-disk store and the project
    /// identity on this path instead of detecting the repo root by
    /// walking up from the current directory. `global = true` lets it be
    /// passed either before or after the subcommand:
    ///   grep --root /repo search-pattern foo
    ///   grep search-pattern --root /repo foo
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
                      `search-symbol` looks a name up across the repository; `outline` answers\n\
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
    /// call, instead of iterating search + who-calls + callees separately.
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
    /// Show every call chain between two symbols as one tree of call sites.
    /// The start is printed at its definition; each indented child is the
    /// editable site in its parent that calls the named symbol.
    Path {
        /// Source symbol (the path start).
        #[arg(long)]
        from: Option<String>,
        /// Destination symbol (the path goal).
        #[arg(long)]
        to: Option<String>,
        /// Stored edge type to follow (CALLS, USAGE, TYPE_ASSIGN, IMPORTS).
        #[arg(
            long,
            default_value = "CALLS",
            value_parser = ["CALLS", "USAGE", "TYPE_ASSIGN", "IMPORTS"],
            ignore_case = true
        )]
        edge: String,
        /// Emit machine-readable JSON with exact shortest-path metadata.
        #[arg(long)]
        json: bool,
        #[arg(skip)]
        code: bool,
        #[arg(skip)]
        all: bool,
    },
    /// Every source line matching a regular expression.
    SearchPattern {
        /// The regular expression (or literal text with `--fixed`) to find.
        #[arg(value_name = "REGEX")]
        query: Option<String>,
        /// Emit machine-readable JSON with the existing stable shape.
        #[arg(long)]
        json: bool,
        /// Treat REGEX as literal text instead of a regular expression.
        #[arg(long)]
        fixed: bool,
        /// Restrict matches by the enclosing definition's kind.
        #[arg(
            long,
            value_parser = ["function", "method", "class", "struct", "enum", "trait"]
        )]
        kind: Option<String>,
        /// Also print each matched source line verbatim.
        #[arg(long)]
        code: bool,
        /// Print every result instead of the five-row distribution sample.
        #[arg(long)]
        all: bool,
    },
    /// Definitions whose name contains NAME.
    SearchSymbol {
        /// The name or name fragment to look up.
        #[arg(value_name = "NAME")]
        query: Option<String>,
        /// Restrict matches to one definition kind.
        #[arg(
            long,
            value_parser = ["function", "method", "class", "struct", "enum", "trait"]
        )]
        kind: Option<String>,
        /// Emit machine-readable JSON with the existing stable shape.
        #[arg(long)]
        json: bool,
        /// Also print each definition's source.
        #[arg(long)]
        code: bool,
        /// Print every matching definition.
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
    /// Definitions that do what a plain-English query describes.
    Search {
        /// Everything after the verb is joined into one meaning query.
        #[arg(value_name = "WHAT IT DOES")]
        query_parts: Vec<String>,
        /// Restrict matches to one definition kind.
        #[arg(
            long,
            value_parser = ["function", "method", "class", "struct", "enum", "trait"]
        )]
        kind: Option<String>,
        /// Emit machine-readable JSON with the existing stable shape.
        #[arg(long)]
        json: bool,
        /// Also print each definition's source.
        #[arg(long)]
        code: bool,
        /// Accepted for family consistency; meaning search remains ranked to eight hits.
        #[arg(long)]
        all: bool,
    },
    /// Legacy compatibility command for resolving definitions. Prefer
    /// `search` for meaning-based search and `brief` for a compact
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
