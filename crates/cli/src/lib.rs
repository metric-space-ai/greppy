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

/// A binary without a GPU backend is not buildable, the same way a binary
/// without the embedded models is not buildable. Nothing fails at runtime when
/// the backend is missing — the work just takes twenty times longer, measured
/// on this repo at 7.5 s against 0.3 s for one navigation summary — so the
/// mistake is invisible unless the compiler refuses it. Building on a platform
/// that has no backend, or measuring against the CPU path, is
/// `--features cpu-only`.
#[cfg(not(feature = "cpu-only"))]
const _: () = assert!(
    greppy_embed_native::HAS_GPU_BACKEND,
    "no GPU backend for this target. Metal is enabled for macOS and CUDA for \
     Linux/Windows in crates/cli/Cargo.toml; if this target genuinely has \
     neither, build with --features cpu-only."
);

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

mod cli_surface;
pub use cli_surface::*;
mod nav;
use nav::*;
mod inference;
use inference::*;
mod freshness;
use freshness::*;
mod emit;
use emit::*;
mod resolving;
use resolving::*;
mod vcs;
use vcs::*;
mod edit;
use edit::*;
mod search;
use search::*;
mod read;
use read::*;
mod plus;
use plus::*;
mod indexing;
use indexing::*;
mod passthrough;
use passthrough::*;
#[cfg(feature = "bash-smart")]
mod bash_smart;
mod context;
use context::*;
mod agent;

use clap::{Parser, Subcommand};
use greppy_core::error::{Error, Result};
use greppy_core::workspace as workspace_locator;

#[cfg(any(unix, windows))]
mod embed_daemon;
#[cfg(any(unix, windows))]
mod inference_daemon;
#[cfg(any(unix, windows))]
mod summarize_daemon;
mod trial;

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
    static CLI_JSON_OUTPUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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

fn set_cli_json_output(enabled: bool) {
    CLI_JSON_OUTPUT.with(|value| value.set(enabled));
}

fn cli_json_output() -> bool {
    CLI_JSON_OUTPUT.with(std::cell::Cell::get)
}

fn cli_result_offset() -> usize {
    CLI_RESULT_OFFSET.with(std::cell::Cell::get)
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

/// The trained edit operations. Clap exposes each variant as a top-level verb;
/// this enum is only the small internal dispatch envelope shared by them.
#[derive(Debug)]
pub enum EditCommand {
    Replace {
        symbol: String,
        new: Option<String>,
        body: bool,
        dry_run: bool,
        verify: bool,
    },
    ReplaceText {
        file: String,
        old: String,
        new: Option<String>,
        expect: Option<usize>,
        regex: bool,
        dry_run: bool,
        verify: bool,
    },
    ReplaceLines {
        file: String,
        lines: String,
        new: Option<String>,
        dry_run: bool,
        verify: bool,
    },
    ReplaceSpan {
        handle: String,
        new: Option<String>,
        dry_run: bool,
        verify: bool,
    },
    Write {
        path: String,
        new: Option<String>,
        dry_run: bool,
        verify: bool,
    },
    Delete {
        symbol: String,
        dry_run: bool,
        verify: bool,
    },
    DeleteLines {
        file: String,
        lines: String,
        dry_run: bool,
        verify: bool,
    },
    InsertLines {
        file: String,
        line: usize,
        new: Option<String>,
        dry_run: bool,
        verify: bool,
    },
    Rename {
        symbol: String,
        name: String,
        dry_run: bool,
        verify: bool,
    },
    Undo {
        id: Option<String>,
        dry_run: bool,
        verify: bool,
    },
    Patch {
        diff: Option<String>,
        dry_run: bool,
        verify: bool,
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
    "where-am-i",
    "cache",
    "trial",
    "stats",
    "diagnostics",
    "doctor",
    "search-graph",
    "trace",
    "impact",
    "brief",
    #[cfg(feature = "bash-smart")]
    "bash-smart",
    "expand",
    "read",
    "read-smart",
    "read-file",
    "replace",
    "replace-text",
    "replace-lines",
    "replace-span",
    "write",
    "delete",
    "delete-lines",
    "insert-lines",
    "rename",
    "undo",
    "patch",
    "who-calls",
    "callees",
    "fan-in",
    "fan-out",
    "graph-locate",
    "path",
    "search",
    "search-symbol",
    "search-pattern",
    "plus",
    "context",
    "install",
    "uninstall",
    "update",
    "upgrade",
    "config",
    "embed-daemon",
    "summarize-daemon",
];

/// Names removed from the edit surface are reserved forever: none may fall
/// through to real grep and become a pattern by accident.
const DEAD_EDIT_VERBS: &[&str] = &[
    "edit",
    "insert",
    "move",
    "remove",
    "change-signature",
    "data",
    "apply",
    "recover",
    "ensure-import",
    "ensure-argument",
    "ensure-method",
    "ensure-annotation",
    "text-cas",
    "regex-cas",
    "replace-body",
    "patch-span",
    "insert-after",
    "insert-before",
    "remove-if-present",
];

const DEAD_EDIT_FLAGS: &[&str] = &[
    "--content",
    "--content-file",
    "--old",
    "--old-file",
    "--new-file",
    "--pattern",
    "--target",
    "--source-file",
    "--patch-file",
    "--plan",
    "--spec",
    "--report",
    "--module",
    "--annotation",
    "--arg",
    "--value-json",
    "--call",
];

/// An argv that names a removed command or flag is a Greppy invocation whose
/// vocabulary is no longer valid. Refuse it before passthrough routing.
fn unknown_verb_refusal(argv: &[std::ffi::OsString]) -> Option<String> {
    let rest = grep_passthrough_args(argv);
    let verb = rest.first()?.to_str()?;
    if matches!(
        verb,
        "find-usages"
            | "references"
            | "map"
            | "outline"
            | "changes"
            | "verify"
            | "semantic-search"
            | "semantic"
            | "search-symbols"
            | "search-code"
    ) || DEAD_EDIT_VERBS.contains(&verb)
        || verb.starts_with("ensure-")
    {
        return Some(format!("error: unrecognized subcommand '{verb}'"));
    }
    if let Some(flag) = rest.iter().skip(1).find_map(|token| {
        let token = token.to_str()?;
        DEAD_EDIT_FLAGS
            .iter()
            .find(|flag| token == **flag || token.starts_with(&format!("{flag}=")))
            .copied()
    }) {
        return Some(format!("error: unrecognized argument '{flag}'"));
    }
    if verb.starts_with('-') || SUBCOMMANDS.contains(&verb) {
        return None;
    }
    greppy_only_flag(&rest[1..])?;
    Some(format!(
        "unrecognized command `{verb}`\nusage: greppy <command> --help  (commands: index, trial, who-calls, callees, \
         impact, brief, search, search-symbol, search-pattern, path, read, replace, patch)"
    ))
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
    // `greppy -p` is a structured agent invocation. Intercept before
    // unknown-verb / grep-passthrough so `-p` never becomes a real-grep
    // pattern, and so grep flags like `-p` (Perl regex on GNU grep) still
    // work for non-leading positions via ordinary passthrough.
    if agent::is_agent_p_invocation(&argv) {
        return agent::run_agent_p(&argv);
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
                     impact, brief, search, search-symbol, search-pattern, \
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
        "brief" => "greppy brief SYMBOL [--path PATH] [--json] [--root DIR]",
        "read" => {
            "greppy read SYMBOL [--head M] [--tail N] [--handle] [--root DIR]  \
             or: greppy read FILE [--line N[:M]] [--handle] [--json] [--root DIR]"
        }
        "replace" => "greppy replace S [NEW] [--body] [--dry-run] [--verify]",
        "replace-text" => "greppy replace-text F OLD [NEW] [--expect N] [--regex] [--dry-run] [--verify]",
        "replace-lines" => "greppy replace-lines F A:B [NEW] [--dry-run] [--verify]",
        "replace-span" => "greppy replace-span H [NEW] [--dry-run] [--verify]",
        "write" => "greppy write PATH [NEW] [--dry-run] [--verify]",
        "delete" => "greppy delete S [--dry-run] [--verify]",
        "delete-lines" => "greppy delete-lines F A:B [--dry-run] [--verify]",
        "insert-lines" => "greppy insert-lines F N [NEW] [--dry-run] [--verify]",
        "rename" => "greppy rename S NAME [--dry-run] [--verify]",
        "undo" => "greppy undo [ID] [--dry-run] [--verify]",
        "patch" => "greppy patch [DIFF] [--dry-run] [--verify]",
        "expand" => "greppy expand ID [--json] [--root DIR]",
        "search" => {
            "greppy search WHAT IT DOES [--kind function|method|class|struct|enum|trait] [--code] [--all] [--json] [--root DIR]"
        }
        "context" => "greppy context \"QUERY\" [--root DIR]",
        "search-pattern" => {
            "greppy search-pattern REGEX [--fixed] [--kind function|method|class|struct|enum|trait] [--code] [--all] [--json] [--root DIR]"
        }
        "search-symbol" => {
            "greppy search-symbol NAME [--kind function|method|class|struct|enum|trait] [--code] [--all] [--json] [--root DIR]"
        }
        "path" => "greppy path --from SYMBOL --to SYMBOL [--root DIR]",
        "index" => "greppy index PATH [--device auto|cpu|metal|cuda]",
        "where-am-i" => "greppy where-am-i [--json] [--root DIR]",
        "trial" => {
            "greppy trial --root DIR --question QUESTION --check who-calls --symbol SYMBOL \
             --expect TEXT [--forbid TEXT] --runner pi --provider NAME --model ID"
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
            || tok == "--diagnostics"
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
    set_cli_json_output(command_requests_json(cli.command.as_ref()));
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
    println!("       brief S   search WHAT IT DOES");
    println!("       search-symbol NAME   search-pattern REGEX");
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
        Command::WhereAmI { json } => dispatch_where_am_i(root, json),
        Command::Cache { command } => dispatch_cache(command, root),
        Command::Trial { args } => trial::run(args, root),
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
            all,
            json,
        } => {
            let targets = nav_targets(&symbols)?;
            validate_path_filters(root, &path_opts, "--path")?;
            // The prompt's signature is `impact S` — one symbol. The multi
            // note covers who-calls and callees only; a second symbol here is
            // a usage error, like brief's (the tree of one symbol is already
            // a screen, several at once would break the size law).
            if targets.len() > 1 {
                return Err(Error::Invalid(format!(
                    "impact takes one symbol, got {}: impact them one at a time",
                    targets.len()
                )));
            }
            dispatch_impact(
                targets.first().map(String::as_str),
                &path_opts,
                &direction,
                edge.as_deref(),
                depth,
                all,
                json,
                root,
            )
        }
        Command::Brief {
            symbol,
            path_opts,
            code: _,
            all: _,
            json,
        } => {
            let targets = nav_targets(std::slice::from_ref(&symbol))?;
            validate_path_filters(root, &path_opts, "--path")?;
            if targets.len() > 1 {
                // One SYMBOL is enforced at the clap level; several can only
                // arrive through the `-` pipe, and brief sketches ONE body.
                return Err(Error::Invalid(format!(
                    "brief takes exactly one symbol; the pipe delivered {} — brief them one at a time",
                    targets.len()
                )));
            }
            dispatch_brief(targets.first().map(String::as_str), &path_opts, json, root)
        }
        #[cfg(feature = "bash-smart")]
        Command::BashSmart { regexes, argv } => bash_smart::run(&argv, &regexes, root),
        Command::Expand { id, json } => dispatch_expand(id.as_deref(), json, root),
        Command::Read {
            symbols,
            head,
            tail,
            handle,
            json,
            path_opts,
        } => {
            let symbols = nav_targets(&symbols)?;
            validate_path_filters(root, &path_opts, "--path")?;
            dispatch_read_symbols(&symbols, head, tail, handle, json, &path_opts, root)
        }
        Command::ReadSmart {
            symbols,
            depth,
            handle,
            path_opts,
        } => {
            let symbols = nav_targets(&symbols)?;
            validate_path_filters(root, &path_opts, "--path")?;
            dispatch_read_smart(&symbols, depth, handle, &path_opts, root)
        }
        Command::ReadFile {
            paths,
            lines,
            all,
            handle,
            path_opts,
        } => {
            validate_path_filters(root, &path_opts, "--path")?;
            dispatch_read_files(&paths, lines.as_deref(), all, handle, &path_opts, root)
        }
        Command::Replace {
            symbol,
            new,
            body,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::Replace {
                symbol,
                new,
                body,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::ReplaceText {
            file,
            old,
            new,
            expect,
            regex,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::ReplaceText {
                file,
                old,
                new,
                expect,
                regex,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::ReplaceLines {
            file,
            lines,
            new,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::ReplaceLines {
                file,
                lines,
                new,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::ReplaceSpan {
            handle,
            new,
            dry_run,
            verify,
            json,
        } => {
            let handle = resolve_compact_read_handle(&handle, root)?.unwrap_or(handle);
            dispatch_edit(
                EditCommand::ReplaceSpan {
                    handle,
                    new,
                    dry_run,
                    verify,
                },
                json,
                root,
            )
        }
        Command::Write {
            path,
            new,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::Write {
                path,
                new,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::Delete {
            symbol,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::Delete {
                symbol,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::DeleteLines {
            file,
            lines,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::DeleteLines {
                file,
                lines,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::InsertLines {
            file,
            line,
            new,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::InsertLines {
                file,
                line,
                new,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::Rename {
            symbol,
            name,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::Rename {
                symbol,
                name,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::Undo {
            id,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::Undo {
                id,
                dry_run,
                verify,
            },
            json,
            root,
        ),
        Command::Patch {
            diff,
            dry_run,
            verify,
            json,
        } => dispatch_edit(
            EditCommand::Patch {
                diff,
                dry_run,
                verify,
            },
            json,
            root,
        ),
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
        Command::SearchPattern {
            query,
            json,
            fixed,
            kind,
            code,
            all,
            path_opts,
        } => dispatch_search_code(
            query.as_deref(),
            kind.as_deref(),
            code,
            all,
            json,
            fixed,
            &path_opts,
            root,
        ),
        Command::SearchSymbol {
            query,
            kind,
            json,
            code,
            all,
            path_opts,
        } => dispatch_search_symbols(
            query.as_deref(),
            kind.as_deref(),
            code,
            all,
            json,
            &path_opts,
            EmbeddingCliArgs { device, no_gpu },
            root,
        ),
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
        Command::Search {
            query_parts,
            kind,
            json,
            code,
            all,
            path_opts,
        } => {
            let query = query_parts.join(" ");
            dispatch_semantic(
                (!query.trim().is_empty()).then_some(query.as_str()),
                &path_opts,
                kind.as_deref(),
                code,
                all,
                json,
                EmbeddingCliArgs { device, no_gpu },
                root,
            )
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

fn symbol_miss_json(store: &greppy_store::Store, project: &str, query: &str) -> serde_json::Value {
    serde_json::json!({
        "suggestions": symbol_miss_suggestions(store, project, query),
        "next": [
            format!("greppy search-symbol {}", shell_example_arg(query)),
            format!("greppy search {}", shell_example_arg(query)),
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
/// Rows of the impact tree before the count takes over: one screen, the same
/// law every other command obeys.
const IMPACT_TREE_LIMIT: usize = 40;

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

/// Whether the vector query path may self-heal a stale index via the
/// atomic auto-reindex: only when the embedding model is resolvable, because
/// an existing vector generation must be rebuilt as part of the snapshot.
fn vector_auto_reindex_can_rebuild(args: EmbeddingCliArgs<'_>) -> bool {
    match embedding_config_optional(args) {
        Ok(Some(cfg)) => embedding_model_source_exists(&cfg.source),
        Ok(None) | Err(_) => false,
    }
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

#[derive(Clone, Copy)]
struct SemanticFallbackContext<'a> {
    query: &'a str,
    paths: &'a [String],
    root: Option<&'a str>,
}

/// `--code` and `--json` compose: AGENTS.md gives `--code` as "also print each
/// result's source and a handle for it" and `--json` as "the same answer as
/// data", so the source and the handle belong ON the hit.
fn ensure_nav_json_mode(_code: bool, _json: bool) -> Result<()> {
    Ok(())
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

/// Rows of callers/callees shown by `brief` before truncating — smaller than
/// NAV_LIMIT because a briefing is a summary, not an exhaustive listing.
const BRIEF_LIMIT: usize = 15;

/// Folded into the summary-cache key only (not the daemon identity):
/// bump when generation behavior changes without a prompt-version bump,
/// so cached summaries cannot outlive the code that produced them.
#[cfg(any(unix, windows))]
const SUMMARY_CACHE_GENERATION: &str = "sc1";

/// Summary generations are cached by (model identity, path+span content):
/// a hit skips the daemon entirely, a miss generates and stores best-effort.
/// Cache problems are cache misses — they can never fail the command.
#[cfg(any(unix, windows))]
fn summarize_source_cached(
    cfg: &QwenSummaryConfig,
    model_key: &str,
    cache: Option<&greppy_store::SummaryCache>,
    file_path: &str,
    source: &str,
) -> Option<Vec<String>> {
    let cache_key = format!("{model_key}#{SUMMARY_CACHE_GENERATION}");
    let hash = greppy_store::span_hash(file_path, source);
    if let Some(cache) = cache {
        if let Ok(Some(bullets)) = cache.get(&cache_key, &hash) {
            return Some(bullets);
        }
    }
    let bullets = summarize_daemon::summarize_source_via_daemon(cfg, model_key, file_path, source)
        .filter(|bullets| !bullets.is_empty())?;
    if let Some(cache) = cache {
        let _ = cache.put(&cache_key, &hash, &bullets);
    }
    Some(bullets)
}

/// Async nudge for the Qwen daemon, called ONLY from the dispatches that
/// actually emit summaries (brief, the impact tree, semantic purposes).
/// Deterministic nav commands must never hold a summary model in memory.
#[cfg(any(unix, windows))]
pub(crate) fn prewarm_summary_daemon() {
    if let Ok(Some(cfg)) = qwen_summary_config_optional() {
        summarize_daemon::prewarm_from_env(&cfg);
    }
}
#[cfg(not(any(unix, windows)))]
pub(crate) fn prewarm_summary_daemon() {}

/// `file_path` is the repo-relative path of the definition's file; the brief
/// prompt contract feeds it to the model alongside the source span.
/// `root_path` locates the workspace's summary cache — the command's resolved
/// root, so `--root` invocations hit the right cache.
fn summarize_definition_span(
    root_path: &std::path::Path,
    file_path: &str,
    source_span: &str,
) -> Option<Vec<String>> {
    #[cfg(any(unix, windows))]
    {
        let cfg = qwen_summary_config_optional().ok().flatten()?;
        let model_key = qwen_summary_model_key(&cfg);
        let cache = greppy_store::SummaryCache::open(&workspace_locator::store_dir(root_path)).ok();
        summarize_source_cached(&cfg, &model_key, cache.as_ref(), file_path, source_span)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (root_path, file_path, source_span);
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
    /// The journal transaction that made the CAS visible.
    transaction_id: Option<String>,
    /// The requested bytes were already present, so no write was necessary.
    already_as_sent: bool,
}

struct GrammarDispatch(i32);

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
    Symbol,
    Target,
}

impl SelectorKind {
    /// Line-oriented spans address whole lines, so the newline that ends the
    /// last one belongs to the file, not to the span: replacing a line with
    /// text that has no newline must not join it to the next one.
    fn line_oriented(self) -> bool {
        matches!(
            self,
            SelectorKind::Lines | SelectorKind::Symbol | SelectorKind::Target
        )
    }

    fn name(self) -> &'static str {
        match self {
            SelectorKind::Text => "--old",
            SelectorKind::Pattern => "--pattern",
            SelectorKind::Lines => "--lines",
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

/// A resolved definition: its path relative to the root, its absolute path,
/// the bytes of the file it lives in, and the byte range it occupies.
type ResolvedSpan = (String, std::path::PathBuf, Vec<u8>, (usize, usize));

/// A rewritten file, and the byte ranges of it the edit wrote.
type EditedContent = (Vec<u8>, Vec<(usize, usize)>);

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

// --- reporting --------------------------------------------------------------

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
    #[cfg(feature = "bash-smart")]
    if pack.command == "bash-smart" {
        return bash_smart::expand(&store, pack, json);
    }
    if matches!(pack.command.as_str(), "read-smart" | "read-file") {
        return dispatch_read_expand(&store, &pack, json, root);
    }
    let mut payload_text = pack.payload_text.clone();
    let mut payload_json = pack.payload_json.clone();
    if pack.command == "where-am-i" {
        let Some(metadata) = payload_json.as_ref() else {
            println!("expand: invalid where-am-i inventory pack");
            return Ok(1);
        };
        match where_inventory_expand_payload(&store, root, &pack.project, metadata)? {
            Ok((text, value)) => {
                payload_text = text;
                payload_json = Some(value);
            }
            Err(message) => {
                println!("{message}");
                return Ok(1);
            }
        }
    }
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
            "payload_text": payload_text,
            "payload_json": payload_json,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v)
                .map_err(|e| Error::Invalid(format!("serialize expand JSON: {e}")))?
        );
    } else {
        print!("{payload_text}");
        if !payload_text.ends_with('\n') {
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
    println!("try: greppy search-symbol {}", shell_example_arg(symbol));
    println!("try: greppy search {}", shell_example_arg(symbol));
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
             `greppy search-symbol {}` for the exact name first",
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
    let canonical_root = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.clone());
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
fn ensure_unambiguous_target(store: &greppy_store::Store, target: &str, ids: &[i64]) -> Result<()> {
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
        });
    }
    let symbol = targets.first().map(String::as_str);
    match kind {
        NavKind::WhoCalls => dispatch_who_calls(symbol, paths, code, all, json, root),
        NavKind::Callees => dispatch_callees(symbol, paths, code, all, json, root),
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
    let path_filters = prepare_query_path_filters(req.root, req.command, "", req.paths)?;

    let mut rows: Vec<NavRow> = Vec::new();
    let mut totals = vec![0usize; req.targets.len()];
    let mut tests: Vec<Vec<serde_json::Value>> = vec![Vec::new(); req.targets.len()];
    for (index, ids) in resolved.iter().enumerate() {
        let mut collected = nav_rows_for_target(&store, &project, ids, index, req.kind)?;
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

    // The multi answer is the single answer, grouped: one bare line naming
    // the queried symbol opens each group (it answers "which question", it is
    // not decoration), then the same rows the single-symbol path prints. An
    // empty group states its empty answer instead of vanishing.
    let mut sources: std::collections::HashMap<String, Option<Vec<String>>> =
        std::collections::HashMap::new();
    let empty_word = match req.kind {
        NavKind::WhoCalls => "no callers",
        NavKind::Callees => "no callees",
    };
    for (index, symbol) in req.targets.iter().enumerate() {
        if index > 0 {
            println!();
        }
        println!("{symbol}");
        if totals[index] == 0 {
            println!("{empty_word}");
            continue;
        }
        // A group whose rows all fell behind the cap must not look like an
        // empty answer: count what is missing.
        if !window.iter().any(|row| row.target == index) {
            println!("… {} below the cut", totals[index]);
            continue;
        }
        for row in window.iter().filter(|row| row.target == index) {
            // Same truth as the single-symbol rows: the shared source-aware
            // test detection, not the cheaper node-only heuristic.
            let lines = sources
                .entry(row.node.file_path.clone())
                .or_insert_with(|| nav_file_lines(&root_path, &row.node.file_path));
            let test_suffix = if nav_is_test(lines.as_ref(), &row.node) {
                "  test"
            } else {
                ""
            };
            println!(
                "{}:{}  {}{}",
                row.node.file_path,
                row.node.start_line,
                nav_short_name(&row.node),
                test_suffix
            );
            if req.code {
                if let Some((source, handle)) = node_source_and_handle(&root_path, &row.node) {
                    print_code_span_text(&source);
                    println!("handle: {handle}");
                }
            }
        }
    }
    if !req.all && cli_result_offset() == 0 && end < total {
        println!(
            "… {} more — {}",
            total - end,
            nav_continuation_command(&req, end)
        );
    }
    Ok(0)
}

fn nav_rows_for_target(
    store: &greppy_store::Store,
    project: &str,
    ids: &[i64],
    index: usize,
    kind: NavKind,
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

struct SpanRead {
    text: String,
    end_line: i64,
    total_lines: usize,
    shown_lines: usize,
    omitted_lines: usize,
    truncated: bool,
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

fn vector_stale_skip_message(command: &str, freshness: &serde_json::Value) -> String {
    format!(
        "{command}: vector search skipped because {}",
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
    const BARE_FLAGS: [&str; 2] = ["--no-gpu", "--diagnostics"];
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
    ("--expect", "replace-text"),
    ("--body", "replace"),
    ("--dry-run", "replace"),
    ("--verify", "replace"),
    ("--to", "path"),
    ("--lines", "read-file"),
    ("--handle", "read"),
    ("--kind", "search-symbol"),
    ("--depth", "impact"),
    ("--direction", "impact"),
    ("--from", "path"),
    ("--path", NAV_OWNER),
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

    // Ordinary grep invocations are a byte-exact delegation contract. In
    // particular, a directory operand without `-r` must keep real grep's
    // diagnostic and exit code rather than silently acquiring recursion.
    let _ = named_grep;
    let grep_args = stripped;
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

fn sqlite_header_is_valid(path: &std::path::Path) -> bool {
    let mut header = [0u8; 16];
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    std::io::Read::read_exact(&mut file, &mut header).is_ok() && &header == b"SQLite format 3\0"
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

fn command_requests_json(command: Option<&Command>) -> bool {
    match command {
        Some(Command::SearchGraph { json, .. })
        | Some(Command::Trace { json, .. })
        | Some(Command::Impact { json, .. })
        | Some(Command::Brief { json, .. })
        | Some(Command::Expand { json, .. })
        | Some(Command::Read { json, .. })
        | Some(Command::WhoCalls { json, .. })
        | Some(Command::Callees { json, .. })
        | Some(Command::FanIn { json, .. })
        | Some(Command::FanOut { json, .. })
        | Some(Command::GraphLocate { json, .. })
        | Some(Command::Path { json, .. })
        | Some(Command::SearchPattern { json, .. })
        | Some(Command::SearchSymbol { json, .. })
        | Some(Command::Plus { json, .. })
        | Some(Command::Search { json, .. })
        | Some(Command::Context { json, .. }) => *json,
        _ => false,
    }
}

fn compact_default_json_requested(cli: &Cli) -> bool {
    !cli.diagnostics && command_requests_json(cli.command.as_ref())
}

#[derive(Debug, Clone)]
struct OutputBudgetSpec {
    command: &'static str,
    json: bool,
    max_bytes: Option<usize>,
    offset: usize,
}

fn output_budget_spec(cli: &Cli) -> Option<OutputBudgetSpec> {
    if cli.max_bytes.is_none() && cli.offset == 0 {
        return None;
    }
    let (command, json) = match cli.command.as_ref()? {
        Command::SearchGraph { json, .. } => ("search-graph", *json),
        Command::Trace { json, .. } => ("trace", *json),
        Command::Impact { json, .. } => ("impact", *json),
        Command::Brief { json, .. } => ("brief", *json),
        Command::Expand { json, .. } => ("expand", *json),
        Command::WhoCalls { json, .. } => ("who-calls", *json),
        Command::Callees { json, .. } => ("callees", *json),
        Command::FanIn { json, .. } => ("fan-in", *json),
        Command::FanOut { json, .. } => ("fan-out", *json),
        Command::GraphLocate { json, .. } => ("graph-locate", *json),
        Command::Path { json, .. } => ("path", *json),
        Command::SearchPattern { json, .. } => ("search-pattern", *json),
        Command::SearchSymbol { json, .. } => ("search-symbol", *json),
        Command::Plus { json, .. } => ("plus", *json),
        Command::Search { json, .. } => ("search", *json),
        Command::Context { json, .. } => ("context", *json),
        _ => return None,
    };
    Some(OutputBudgetSpec {
        command,
        json,
        max_bytes: cli.max_bytes,
        offset: cli.offset,
    })
}

fn begin_output_capture() {
    OUTPUT_CAPTURE.with(|capture| *capture.borrow_mut() = Some(Vec::new()));
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

fn json_value_is_empty(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Array(values) => values.is_empty(),
        serde_json::Value::Object(values) => values.is_empty(),
        serde_json::Value::String(value) => value.is_empty(),
        _ => false,
    }
}

fn json_answer_warning(value: &serde_json::Value) -> Option<String> {
    let mut warnings = Vec::new();
    let freshness = value.get("freshness");
    let fresh = value
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            freshness
                .and_then(|freshness| freshness.get("fresh"))
                .and_then(serde_json::Value::as_bool)
        });
    if fresh == Some(false) {
        let state = freshness
            .and_then(|freshness| freshness.get("state"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("stale");
        let stale_files = freshness
            .and_then(|freshness| freshness.get("stale_file_count"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if stale_files > 0 {
            warnings.push(format!("index {state}: {stale_files} stale files"));
        } else {
            warnings.push(format!("index {state}"));
        }
    }
    if let Some(providers) = value
        .get("incomplete_providers")
        .and_then(serde_json::Value::as_array)
        .filter(|providers| !providers.is_empty())
    {
        let mut names = providers
            .iter()
            .filter_map(|provider| provider.get("language"))
            .filter_map(serde_json::Value::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        let count = if names.is_empty() {
            providers.len()
        } else {
            names.len()
        };
        warnings.push(format!(
            "{count} incomplete provider{}; answer may be partial",
            if count == 1 { "" } else { "s" }
        ));
    }
    (!warnings.is_empty()).then(|| warnings.join("; "))
}

fn compact_result_row(value: &mut serde_json::Value, keep_target: bool) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    if !object.contains_key("file") {
        if let Some(file) = object
            .get("file_path")
            .or_else(|| object.get("path"))
            .cloned()
        {
            object.insert("file".into(), file);
        }
    }
    if !object.contains_key("line") {
        if let Some(line) = object.get("start_line").cloned() {
            object.insert("line".into(), line);
        }
    }
    for key in ["node_id", "rank", "schema_version", "file_path", "path"] {
        object.remove(key);
    }
    if !keep_target {
        object.remove("target");
    }
    object.retain(|_, value| !value.is_null());
}

fn compact_result_arrays(value: &mut serde_json::Value, keep_target: bool) {
    for key in BUDGET_ARRAY_FIELDS {
        if let Some(rows) = value
            .get_mut(*key)
            .and_then(serde_json::Value::as_array_mut)
        {
            for row in rows {
                compact_result_row(row, keep_target);
            }
        }
    }
    if let Some(candidates) = value
        .get_mut("candidates")
        .and_then(serde_json::Value::as_array_mut)
    {
        for candidate in candidates {
            compact_result_row(candidate, false);
        }
    }
}

fn compact_nav_json(value: &mut serde_json::Value) {
    let multi_target = value
        .get("targets")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|targets| targets.len() > 1);
    compact_result_arrays(value, multi_target);
    let status = value.get("status").cloned().unwrap_or_else(|| {
        match value
            .get("symbol_found")
            .and_then(serde_json::Value::as_bool)
        {
            Some(false) => serde_json::json!("not_found"),
            _ => serde_json::json!("ok"),
        }
    });
    let mut compact = serde_json::Map::new();
    compact.insert("status".into(), status);
    for key in ["total_exact", "shown", "omitted", "truncated", "expand"] {
        if let Some(field) = value.get(key).cloned() {
            compact.insert(key.into(), field);
        }
    }
    if multi_target {
        if let Some(targets) = value.get("targets").cloned() {
            compact.insert("targets".into(), targets);
        }
    }
    for key in ["suggestions", "next", "warning", "hits"] {
        if let Some(field) = value.get(key).cloned() {
            compact.insert(key.into(), field);
        }
    }
    *value = serde_json::Value::Object(compact);
}

fn compact_search_symbol_json(value: &mut serde_json::Value) {
    compact_result_arrays(value, false);
    if let Some(hits) = value
        .get_mut("hits")
        .and_then(serde_json::Value::as_array_mut)
    {
        for hit in hits {
            if let Some(object) = hit.as_object_mut() {
                object.retain(|key, _| {
                    matches!(
                        key.as_str(),
                        "qualified_name"
                            | "file"
                            | "line"
                            | "start_line"
                            | "end_line"
                            | "source_available"
                    )
                });
            }
        }
    }
    let mut compact = serde_json::Map::new();
    for key in [
        "status",
        "total_exact",
        "shown",
        "omitted",
        "truncated",
        "warning",
        "hits",
        "suggestions",
        "next",
    ] {
        if let Some(field) = value.get(key).cloned() {
            compact.insert(key.into(), field);
        }
    }
    if value.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        if let Some(query) = value.get("query").cloned() {
            compact.insert("query".into(), query);
        }
    }
    if let Some(filters) = value
        .get("path_filters")
        .filter(|value| !json_value_is_empty(value))
    {
        compact.insert("path_filters".into(), filters.clone());
    }
    *value = serde_json::Value::Object(compact);
}

fn compact_read_json(value: &mut serde_json::Value) {
    compact_result_arrays(value, false);
    compact_result_row(value, false);
    let status = value
        .get("status")
        .cloned()
        .unwrap_or_else(|| serde_json::json!("ok"));
    let candidate_count = value
        .get("candidates")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let hit_count = value
        .get("hits")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len);
    let total_exact = value
        .get("total_exact")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| match status.as_str() {
            Some("ok") => hit_count.unwrap_or(1) as u64,
            Some("ambiguous") => candidate_count as u64,
            _ => 0,
        });
    let shown = value
        .get("shown")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| match status.as_str() {
            Some("ok") => hit_count.unwrap_or(1) as u64,
            Some("ambiguous") => candidate_count as u64,
            _ => 0,
        });
    let mut compact = serde_json::Map::new();
    compact.insert("status".into(), status);
    compact.insert("total_exact".into(), total_exact.into());
    compact.insert("shown".into(), shown.into());
    for key in [
        "qualified_name",
        "file",
        "line",
        "start_line",
        "end_line",
        "source",
        "handle",
        "hits",
        "candidates",
        "query",
        "warning",
    ] {
        if let Some(field) = value.get(key).filter(|value| !value.is_null()).cloned() {
            compact.insert(key.into(), field);
        }
    }
    *value = serde_json::Value::Object(compact);
}

fn compact_default_json_output(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let command = value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    if let Some(warning) = json_answer_warning(&value) {
        value["warning"] = warning.into();
    }
    match command.as_deref() {
        Some("who-calls" | "callees") => compact_nav_json(&mut value),
        Some("search-symbol") => compact_search_symbol_json(&mut value),
        Some("read") => compact_read_json(&mut value),
        _ => {
            compact_result_arrays(&mut value, true);
            if let Some(object) = value.as_object_mut() {
                for key in [
                    "command",
                    "project",
                    "fresh",
                    "freshness",
                    "provider_complete",
                    "incomplete_provider_count",
                    "incomplete_providers",
                    "discover_scope",
                    "discover_scope_env",
                    "elapsed_ms",
                    "ttl_hit",
                ] {
                    object.remove(key);
                }
                if object.get("omitted").and_then(serde_json::Value::as_u64) == Some(0) {
                    object.remove("omitted");
                }
                if object.get("truncated").and_then(serde_json::Value::as_bool) == Some(false) {
                    object.remove("truncated");
                }
                if object.get("all").and_then(serde_json::Value::as_bool) == Some(false) {
                    object.remove("all");
                }
                if object.get("path_filters").is_some_and(json_value_is_empty) {
                    object.remove("path_filters");
                }
                // `search-pattern` is the one legacy polymorphic search shape:
                // its stable smoke contract uses `command` to distinguish live
                // filesystem matches from the other search backends.
                if command.as_deref() == Some("search-pattern") {
                    object.insert("command".into(), serde_json::json!("search-pattern"));
                }
                if !object.contains_key("status")
                    && BUDGET_ARRAY_FIELDS
                        .iter()
                        .any(|key| object.contains_key(*key))
                {
                    object.insert("status".into(), serde_json::json!("ok"));
                }
            }
        }
    }
    if let Some(object) = value.as_object_mut() {
        if object.get("omitted").and_then(serde_json::Value::as_u64) == Some(0) {
            object.remove("omitted");
        }
        if object.get("truncated").and_then(serde_json::Value::as_bool) == Some(false) {
            object.remove("truncated");
        }
    }
    let mut rendered = serde_json::to_vec_pretty(&value).ok()?;
    rendered.push(b'\n');
    Some(rendered)
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
        {
            return Some(rendered);
        }
        if result_item_count(&value) > 1 {
            pop_result_item(&mut value);
            continue;
        }
        // Progress guarantee: a budget too small for even one full row must
        // not shrink the answer to zero rows with a same-offset retry — that
        // offer would loop forever delivering nothing. The last row sheds its
        // heavy optional fields instead; only when fully slimmed and still
        // over budget does it ship as-is.
        if !slim_last_result_item(&mut value) {
            return Some(rendered);
        }
    }
}

/// One degrade step on the sole remaining result row: drop the heaviest
/// optional field present. Returns false when nothing is left to shed.
fn slim_last_result_item(value: &mut serde_json::Value) -> bool {
    let Some(items) = value
        .get_mut("hits")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return false;
    };
    let Some(row) = items.first_mut().and_then(|row| row.as_object_mut()) else {
        return false;
    };
    for heavy in ["handle", "source", "snippet"] {
        if row.remove(heavy).is_some() {
            return true;
        }
    }
    if let Some(matches) = row
        .get_mut("matches")
        .and_then(serde_json::Value::as_array_mut)
    {
        if matches.len() > 1 {
            matches.truncate(1);
            return true;
        }
    }
    false
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
    let compact_json = compact_default_json_requested(&cli);
    if budget.is_some() || compact_json {
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
    if budget.is_some() || compact_json {
        finish_output_capture(budget.as_ref(), compact_json, code);
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

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
