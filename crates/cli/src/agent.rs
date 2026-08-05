//! `greppy -p` — one-shot coding agent over an isolated git worktree.
//!
//! Intercepted in [`crate::run_os`] before grep-passthrough routing so that
//! ordinary `greppy -R …` / pattern invocations remain byte-exact real-grep.

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use greppy_agent::{
    run_agent_loop, sandbox as agent_sandbox, AgentConfig, AgentWorkspace, Client, GreppyEnv,
    LoopEvent, LoopStop, ProbeError, RunOutcome, SandboxError, SandboxMode, StreamEvent,
    WorkspaceError, SYSTEM_PROMPT,
};

/// Exit: success (clean, proposal, or applied).
pub const EXIT_OK: u8 = 0;
/// Exit: bad usage / missing model / gateway unreachable.
pub const EXIT_USAGE: u8 = 2;
/// Exit: agent loop / transport / workspace failure.
pub const EXIT_AGENT: u8 = 3;
/// Exit: `--apply` cherry-pick conflict.
pub const EXIT_CONFLICT: u8 = 4;

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8317";
const DEFAULT_MAX_TURNS: usize = 40;
const TOOL_LINE_MAX: usize = 120;

const LONG_HELP: &str = "\
One-shot coding agent. Works in a per-repository agent worktree (reset to
HEAD each run; greppy index built on first use and kept warm afterwards) and
delivers a proposal ref (refs/greppy/agent/<run_id>); inspect with `git show`
or apply with `git cherry-pick -n`. The agent has exactly one tool — `greppy`
— covering search/navigate/read/edit; commands run through that tool as
`bash-smart -- CMD`. The tool is write-confined to the worktree, temp,
greppy data root, ~/.cargo and the platform cache; reads and network stay open.
Pass --no-sandbox (or GREPPY_NO_SANDBOX=1) to disable.

Localhost contract: greppy -p talks to an Anthropic-Messages-compatible
gateway at GREPPY_ENDPOINT (default http://127.0.0.1:8317). The client has
no TLS stack — only plain-HTTP endpoints (localhost gateways) are reachable;
https is impossible. The standard is CLIProxyAPI, which translates all major
chat formats/providers to that wire. GREPPY_MODEL / --model is the model id
passed through unchanged. If the gateway requires an API key (CLIProxyAPI
usually does), set GREPPY_API_KEY; it is sent as x-api-key and
Authorization: Bearer. There is no key flag on purpose — keys do not belong
on the command line.

Leading `-p` is reserved for the agent; to grep for the literal pattern `-p`,
use `greppy -e -p …` (or place `-p` later in the invocation).

Usage:
  greppy -p \"TASK\" [--model M] [--endpoint URL] [--max-turns N]
                   [--apply] [--diff] [--keep-worktree] [--no-sandbox]
                   [--skip-selfcheck]
  greppy -p --help

Flags:
  --model M           Model id (required; env GREPPY_MODEL if flag omitted)
  --endpoint URL      Gateway base URL (env GREPPY_ENDPOINT, else
                      http://127.0.0.1:8317)
  --max-turns N       Cap on assistant turns (default 40)
  --apply             Cherry-pick the proposal into the current checkout
                      (staged, not committed)
  --diff              Print the full proposal patch after the stat
  --keep-worktree     Leave a temporary fallback worktree on disk after success
  --no-sandbox        Disable write-confinement (env GREPPY_NO_SANDBOX=1)
  --skip-selfcheck    Skip the startup capability self-check (env GREPPY_SKIP_SELFCHECK=1)

Exit codes:
  0  ok (clean, proposal saved, or applied)
  2  no gateway / bad usage / missing model
  3  agent or loop error (worktree kept for debugging)
  4  --apply refused (dirty target) or cherry-pick conflict (ref still available)
";

/// Parsed `greppy -p` arguments (everything after the leading `-p` token).
#[derive(Debug, Clone, Parser, PartialEq, Eq)]
#[command(
    name = "greppy -p",
    override_help = LONG_HELP,
    disable_version_flag = true
)]
pub struct AgentArgs {
    /// The task for the agent.
    #[arg(value_name = "TASK")]
    pub task: Option<String>,

    /// Model id (required unless GREPPY_MODEL is set).
    #[arg(long, env = "GREPPY_MODEL")]
    pub model: Option<String>,

    /// Anthropic-Messages gateway base URL.
    #[arg(long, env = "GREPPY_ENDPOINT", default_value = DEFAULT_ENDPOINT)]
    pub endpoint: String,

    /// Maximum assistant turns.
    #[arg(long, default_value_t = DEFAULT_MAX_TURNS, value_name = "N")]
    pub max_turns: usize,

    /// Cherry-pick the proposal into the current checkout (staged).
    #[arg(long)]
    pub apply: bool,

    /// Print the full proposal patch after the stat.
    #[arg(long)]
    pub diff: bool,

    /// Keep a temporary fallback worktree after a successful run.
    #[arg(long)]
    pub keep_worktree: bool,

    /// Disable the write-confinement sandbox for tool subprocesses.
    ///
    /// Also set by env `GREPPY_NO_SANDBOX=1` (Boolish: 1/true/yes/y/on).
    #[arg(
        long,
        env = "GREPPY_NO_SANDBOX",
        value_parser = clap::builder::BoolishValueParser::new(),
        default_value_t = false,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = false,
    )]
    pub no_sandbox: bool,

    /// Skip the startup capability self-check (index answers + worktree writable).
    ///
    /// Also set by env `GREPPY_SKIP_SELFCHECK=1` (Boolish: 1/true/yes/y/on).
    /// Deliberate bypass only — a failed self-check means the agent would
    /// silently degrade to a shell-only fallback.
    #[arg(
        long,
        env = "GREPPY_SKIP_SELFCHECK",
        value_parser = clap::builder::BoolishValueParser::new(),
        default_value_t = false,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = false,
    )]
    pub skip_selfcheck: bool,
}

/// True when argv (after greppy-owned globals) starts with `-p`.
pub fn is_agent_p_invocation(argv: &[std::ffi::OsString]) -> bool {
    let rest = super::grep_passthrough_args(argv);
    rest.first().is_some_and(|t| t == "-p")
}

/// Parse and run `greppy -p …`. Caller must have verified [`is_agent_p_invocation`].
pub fn run_agent_p(argv: &[std::ffi::OsString]) -> u8 {
    // Set for every tool subprocess of a running agent: refuse nesting on
    // every path (plain greppy argv and bash-smart).
    if std::env::var_os(greppy_agent::AGENT_RUN_ENV).is_some() {
        eprintln!(
            "greppy -p: refusing a nested agent run — you are already inside an \
             agent; carry out the task directly."
        );
        return EXIT_USAGE;
    }
    let rest = super::grep_passthrough_args(argv);
    debug_assert!(rest.first().is_some_and(|t| t == "-p"));
    let after_p: Vec<std::ffi::OsString> = rest.iter().skip(1).cloned().collect();

    // Build a synthetic argv for clap: program name + flags/task after -p.
    let mut clap_argv: Vec<std::ffi::OsString> = Vec::with_capacity(after_p.len() + 1);
    clap_argv.push(std::ffi::OsString::from("greppy -p"));
    clap_argv.extend(after_p);

    let args = match AgentArgs::try_parse_from(&clap_argv) {
        Ok(a) => a,
        Err(e) => {
            use clap::error::ErrorKind;
            if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
                let _ = e.print();
                return EXIT_OK;
            }
            // Missing required clap bits: print short usage pointer.
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or("invalid arguments");
            eprintln!("{first}");
            eprintln!("usage: greppy -p \"TASK\" [--model M] …  (details: greppy -p --help)");
            return EXIT_USAGE;
        }
    };

    if let Err(code) = validate_args(&args) {
        return code;
    }

    run_agent(args)
}

fn validate_args(args: &AgentArgs) -> Result<(), u8> {
    let task = args.task.as_deref().map(str::trim).unwrap_or("");
    if task.is_empty() {
        eprintln!("error: missing TASK");
        eprintln!("usage: greppy -p \"TASK\" [--model M] …  (details: greppy -p --help)");
        return Err(EXIT_USAGE);
    }
    let model = args.model.as_deref().map(str::trim).unwrap_or("");
    if model.is_empty() {
        eprintln!("error: --model is required (or set GREPPY_MODEL)");
        eprintln!("details: greppy -p --help");
        return Err(EXIT_USAGE);
    }
    Ok(())
}

fn run_agent(args: AgentArgs) -> u8 {
    let task = args.task.as_deref().unwrap_or("").trim().to_string();
    let model = args.model.as_deref().unwrap_or("").trim().to_string();
    let endpoint = args.endpoint.trim().to_string();

    let mut client = Client::new(&endpoint, &model);
    if let Ok(key) = std::env::var("GREPPY_API_KEY") {
        client = client.with_api_key(key);
    }
    match client.probe() {
        Ok(()) => {}
        Err(ProbeError::Unreachable(_)) => {
            eprintln!(
                "greppy -p needs a local model gateway and found none at {endpoint}.\n\
                 Start one (standard: CLIProxyAPI on 127.0.0.1:8317) or set\n\
                 GREPPY_ENDPOINT / --endpoint. Details: greppy -p --help"
            );
            return EXIT_USAGE;
        }
        Err(ProbeError::BadResponse(detail)) => {
            eprintln!(
                "greppy -p reached {endpoint}, but the gateway rejected the probe:\n\
                 {detail}\n\
                 If it requires an API key, set GREPPY_API_KEY. Details: greppy -p --help"
            );
            return EXIT_USAGE;
        }
    }

    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("greppy -p: cannot resolve current directory: {e}");
            return EXIT_AGENT;
        }
    };

    let run_id = make_run_id();
    let workspace = match AgentWorkspace::create(&cwd, &run_id) {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("greppy -p: workspace create failed: {e}");
            return EXIT_AGENT;
        }
    };

    // Prewarm: build/refresh the worktree's own greppy index before the first
    // model turn so search/where-am-i do not open empty.
    ensure_semantic_index(workspace.worktree_path());

    let sandbox_mode = match resolve_sandbox_mode(&args, workspace.worktree_path()) {
        Ok(mode) => mode,
        Err(code) => {
            keep_worktree_on_error(&workspace);
            return code;
        }
    };

    let mut env = match GreppyEnv::new(workspace.worktree_path().to_path_buf()) {
        Ok(env) => env.with_sandbox(sandbox_mode),
        Err(e) => {
            eprintln!("greppy -p: cannot build greppy env: {e}");
            keep_worktree_on_error(&workspace);
            return EXIT_AGENT;
        }
    };

    // Capability self-check: fail loudly before the model loop rather than
    // silently degrading to a shell-only agent when index or sandbox is broken.
    if !args.skip_selfcheck {
        match env.startup_self_check() {
            Ok(ok) => {
                if ok.unrecognized_census_shape {
                    eprintln!(
                        "self-check ok — index answers (census shape unrecognized), worktree writable"
                    );
                } else {
                    eprintln!("self-check ok — index answers, worktree writable");
                }
            }
            Err(err) => {
                eprintln!("{}", err.diagnostic());
                keep_worktree_on_error(&workspace);
                return EXIT_AGENT;
            }
        }
    }

    let config = AgentConfig {
        max_turns: args.max_turns,
        system: Some(SYSTEM_PROMPT.to_string()),
        model: model.clone(),
        ..AgentConfig::default()
    };

    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let mut tool_line_open = false;
    let mut turns: u64 = 0;

    let loop_result = run_agent_loop(&mut client, &mut env, &config, &task, &mut |event| {
        if matches!(event, LoopEvent::TurnComplete { .. }) {
            turns = turns.saturating_add(1);
        }
        handle_loop_event(event, &mut stdout, &mut stderr, &mut tool_line_open);
    });

    if tool_line_open {
        let _ = writeln!(stderr);
    }

    let loop_result = match loop_result {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(stderr, "greppy -p: agent error: {e}");
            keep_worktree_on_error(&workspace);
            return EXIT_AGENT;
        }
    };

    // Token accounting (stderr only; zero values print as zero).
    let usage = &loop_result.usage;
    let _ = writeln!(
        stderr,
        "tokens: in {} out {} (cache read {}, write {}) over {} turns",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        turns
    );

    // Ensure a trailing newline after streamed assistant text.
    if !loop_result.final_text.is_empty() && !loop_result.final_text.ends_with('\n') {
        let _ = writeln!(stdout);
    }
    let _ = stdout.flush();

    // Stop reason reaches the user BEFORE the proposal block so it is not lost.
    // MaxTurns / Stuck still produce the proposal (or clean) outcome; exit 0
    // when a proposal or clean state exists — exit 3 stays for real errors.
    match &loop_result.stop {
        LoopStop::MaxTurns => {
            let _ = writeln!(
                stderr,
                "stopped: turn limit reached ({}) — the result may be incomplete",
                args.max_turns
            );
        }
        LoopStop::Stuck => {
            let n = config.consecutive_failure_stop;
            let _ = writeln!(
                stderr,
                "stopped: {n} consecutive tool failures — the agent could not make progress"
            );
        }
        LoopStop::EndTurn | LoopStop::MaxTokens => {}
    }

    let commit_message = truncate_chars(&task, 72);
    let outcome = match workspace.finish(&commit_message) {
        Ok(o) => o,
        Err(e) => {
            let _ = writeln!(stderr, "greppy -p: finish failed: {e}");
            keep_worktree_on_error(&workspace);
            return EXIT_AGENT;
        }
    };

    let mut exit = EXIT_OK;
    match outcome {
        RunOutcome::Clean => {
            let _ = writeln!(stdout, "no changes proposed.");
        }
        RunOutcome::Proposal {
            commit,
            ref_name,
            patch,
            stat,
        } => {
            let _ = writeln!(stdout);
            let _ = write!(stdout, "{stat}");
            if !stat.ends_with('\n') {
                let _ = writeln!(stdout);
            }
            let binary_files = stat.lines().filter(|l| l.contains("| Bin")).count();
            if binary_files > 0 {
                let _ = writeln!(
                    stderr,
                    "note: proposal contains {binary_files} binary file(s), likely build \
                     artifacts from verification — is the repo's .gitignore complete?"
                );
            }
            let _ = writeln!(stdout, "proposal saved: {ref_name}");
            let _ = writeln!(stdout, "inspect: git show {ref_name}");
            let _ = writeln!(stdout, "apply:   git cherry-pick -n {ref_name}");

            if args.diff {
                let _ = write!(stdout, "{patch}");
                if !patch.ends_with('\n') {
                    let _ = writeln!(stdout);
                }
            }

            if args.apply {
                match workspace.apply_to(workspace.repo_root(), &commit) {
                    Ok(()) => {
                        let _ = writeln!(stdout, "applied (staged, not committed).");
                    }
                    Err(WorkspaceError::DirtyTarget { ref_name, .. }) => {
                        let _ = writeln!(
                            stderr,
                            "target checkout has uncommitted changes — commit or stash first; \
                             the proposal remains at {ref_name}"
                        );
                        exit = EXIT_CONFLICT;
                    }
                    Err(WorkspaceError::Conflict { ref_name, detail }) => {
                        let _ = writeln!(
                            stderr,
                            "greppy -p: apply conflict: {detail}\n\
                             resolve from {ref_name}:\n\
                             inspect: git show {ref_name}\n\
                             apply:   git cherry-pick -n {ref_name}"
                        );
                        exit = EXIT_CONFLICT;
                    }
                    Err(e) => {
                        let _ = writeln!(stderr, "greppy -p: apply failed: {e}");
                        keep_worktree_on_error(&workspace);
                        return EXIT_AGENT;
                    }
                }
            }
        }
    }

    if exit == EXIT_OK && !args.keep_worktree {
        if let Err(e) = workspace.cleanup() {
            let _ = writeln!(stderr, "greppy -p: worktree cleanup failed: {e}");
        }
    } else if exit != EXIT_OK {
        // Conflict still cleans unless keep — success-path cleanup only when
        // exit is 0. Spec: cleanup on every successful run; keep on error.
        // Conflict is exit 4 (error-ish): keep worktree.
        keep_worktree_on_error(&workspace);
    } else {
        // keep_worktree: drop without cleanup.
        let path = workspace.worktree_path().display().to_string();
        let _ = writeln!(stderr, "worktree kept: {path}");
        drop(workspace);
    }

    exit
}

fn handle_loop_event(
    event: LoopEvent,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    tool_line_open: &mut bool,
) {
    match event {
        LoopEvent::Stream(StreamEvent::TextDelta { text }) => {
            let _ = write!(stdout, "{text}");
            let _ = stdout.flush();
        }
        LoopEvent::Stream(StreamEvent::ThinkingDelta { .. }) => {
            // Thinking is not printed.
        }
        LoopEvent::Stream(_) => {}
        LoopEvent::ToolStart {
            name, arguments, ..
        } => {
            if *tool_line_open {
                let _ = writeln!(stderr);
            }
            let line = format_tool_start(&name, &arguments);
            let _ = write!(stderr, "{line}");
            let _ = stderr.flush();
            *tool_line_open = true;
        }
        LoopEvent::ToolFinish { outcome, .. } => {
            if *tool_line_open {
                if outcome.is_error {
                    let _ = writeln!(stderr, " ✗");
                } else {
                    let _ = writeln!(stderr);
                }
                *tool_line_open = false;
            }
        }
        LoopEvent::TurnComplete { .. } => {}
    }
}

fn format_tool_start(name: &str, arguments: &serde_json::Value) -> String {
    let body = match name {
        "greppy" => {
            let joined = arguments
                .get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            format!("→ greppy {joined}")
        }
        other => {
            let raw = arguments.to_string();
            format!("→ {other} {raw}")
        }
    };
    truncate_chars(&body, TOOL_LINE_MAX)
}

fn keep_worktree_on_error(workspace: &AgentWorkspace) {
    eprintln!(
        "worktree kept for debugging: {}",
        workspace.worktree_path().display()
    );
}

fn make_run_id() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("agent-{secs}-{pid}")
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Resolve the sandbox mode for a `-p` run.
///
/// `--no-sandbox` / `GREPPY_NO_SANDBOX` → `Off` (one stderr line).
/// Otherwise prepare the worktree's writable roots **exactly once**
/// ([`agent_sandbox::resolve_enforce_spec`]: create + full-path symlink
/// validation + canonicalize) and probe the platform backend. The resulting
/// `Enforce` spec carries those fixed canonical roots for the whole agent run;
/// per-tool `apply` never re-resolves them.
///
/// `Unsupported` (Linux without Landlock ABI ≥ V3) warns once and falls back
/// to `Off`; any other error aborts with [`EXIT_AGENT`].
fn resolve_sandbox_mode(args: &AgentArgs, worktree_path: &Path) -> Result<SandboxMode, u8> {
    if args.no_sandbox {
        eprintln!("sandbox disabled");
        return Ok(SandboxMode::Off);
    }
    let raw = writable_roots_for(worktree_path);
    match agent_sandbox::resolve_enforce_spec(&raw) {
        Ok(mode) => Ok(mode),
        Err(SandboxError::Unsupported) => {
            eprintln!(
                "greppy -p: sandbox unsupported on this kernel/platform — continuing unsandboxed"
            );
            Ok(SandboxMode::Off)
        }
        Err(e) => {
            eprintln!("greppy -p: sandbox setup failed: {e}");
            Err(EXIT_AGENT)
        }
    }
}

/// Writable roots for a sandboxed `-p` tool subprocess.
///
/// Keep this list minimal. Each root needs an explicit reason to be writable:
/// 1. the run's worktree — agent edits and worktree-local builds land here,
/// 2. `std::env::temp_dir()` — process temp / intermediate files,
/// 3. greppy's data root — owns `locks/`, `trash/`, and `workspaces/` (index-
///    backed commands acquire lifecycle leases under `locks/` and open
///    `graph.db` under `workspaces/`; one root covers them all),
/// 4. `~/.cargo` — registry/build caches (respects `CARGO_HOME`),
/// 5. the platform user cache dir (`~/Library/Caches` on macOS,
///    `$XDG_CACHE_HOME` or `~/.cache` on Linux) — model/download caches.
fn writable_roots_for(worktree_path: &Path) -> Vec<std::path::PathBuf> {
    vec![
        // Agent proposal edits and worktree-local builds land here.
        worktree_path.to_path_buf(),
        // Process temp (and macOS TMPDIR) for intermediate files.
        std::env::temp_dir(),
        // Greppy's data root owns locks/, trash/, and workspaces/ — index-backed
        // commands acquire lifecycle leases under locks/ and open graph.db under
        // workspaces/. One root covers them all (see greppy_core::cache).
        greppy_core::cache::data_root(),
        // Cargo registry / git / target build caches (respects CARGO_HOME).
        cargo_home_dir(),
        // Platform user cache (e.g. model downloads; XDG cache on Linux).
        platform_user_cache_dir(),
    ]
}

fn cargo_home_dir() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("CARGO_HOME") {
        return std::path::PathBuf::from(home);
    }
    home_dir().join(".cargo")
}

fn platform_user_cache_dir() -> std::path::PathBuf {
    #[cfg(target_os = "macos")]
    {
        home_dir().join("Library").join("Caches")
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            return std::path::PathBuf::from(xdg);
        }
        home_dir().join(".cache")
    }
}

fn home_dir() -> std::path::PathBuf {
    if let Some(h) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(h);
    }
    // Last-resort fallback; should not matter on real agent hosts.
    std::path::PathBuf::from("/")
}

/// Ensure the worktree's greppy index is usable before the first agent turn.
///
/// Skips quietly when `doctor --json` already reports `embedding_complete:
/// true` (warm tree). Otherwise prints one cold-tree line, runs
/// `<current_exe> index` (incremental) with credential scrub and no sandbox,
/// then re-checks doctor. Failure warns with the consequence and continues —
/// the agent can still work via name/text search while embeddings catch up.
fn ensure_semantic_index(worktree_path: &Path) {
    if doctor_reports_embedding_complete(worktree_path) {
        return;
    }

    let Ok(bin) = std::env::current_exe() else {
        eprintln!(
            "greppy -p: cannot resolve current binary to prewarm index — \
             semantic search may report building until index finishes"
        );
        return;
    };

    eprintln!("indexing the agent worktree (first run for this repository)…");

    let mut cmd = Command::new(&bin);
    cmd.arg("index")
        .current_dir(worktree_path)
        .stdin(std::process::Stdio::null());
    scrub_credential_env(&mut cmd);

    match cmd.status() {
        Ok(status) if status.success() => {
            if !doctor_reports_embedding_complete(worktree_path) {
                eprintln!(
                    "greppy -p: index finished but doctor reports not complete — continuing; \
                     semantic search may report building until the index finishes"
                );
            }
        }
        Ok(status) => {
            eprintln!(
                "greppy -p: index prewarm exited {status} — continuing; \
                 semantic search may report building until the index finishes"
            );
        }
        Err(e) => {
            eprintln!(
                "greppy -p: index prewarm failed to start ({e}) — continuing; \
                 semantic search may report building until the index finishes"
            );
        }
    }
}

/// Cheap completeness signal: `doctor --json` → `embedding_complete == true`.
///
/// Any spawn/parse failure means "not known complete" so the caller runs index.
fn doctor_reports_embedding_complete(worktree_path: &Path) -> bool {
    let Ok(bin) = std::env::current_exe() else {
        return false;
    };
    let mut cmd = Command::new(bin);
    cmd.arg("doctor")
        .arg("--json")
        .current_dir(worktree_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    scrub_credential_env(&mut cmd);
    let Ok(output) = cmd.output() else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
        return false;
    };
    value
        .get("embedding_complete")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// Credential env vars stripped from agent-spawned greppy subprocesses
/// (prewarm index / doctor). Mirrors the greppy-agent tool blocklist so
/// API keys held by the agent process never leak into children.
const CREDENTIAL_ENV_BLOCKLIST: &[&str] = &[
    "GREPPY_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "SSH_AUTH_SOCK",
];

fn scrub_credential_env(cmd: &mut Command) {
    for key in CREDENTIAL_ENV_BLOCKLIST {
        cmd.env_remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique(tag: &str) -> PathBuf {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "greppy-agent-cli-{tag}-{}-{}-{}",
            std::process::id(),
            seq,
            nanos
        ))
    }

    fn parse(args: &[&str]) -> Result<AgentArgs, clap::error::Error> {
        let mut full: Vec<OsString> = vec![OsString::from("greppy -p")];
        full.extend(args.iter().map(OsString::from));
        AgentArgs::try_parse_from(full)
    }

    #[test]
    fn parse_task_model_endpoint_max_turns_flags() {
        let a = parse(&[
            "fix the bug",
            "--model",
            "claude-test",
            "--endpoint",
            "http://127.0.0.1:9999",
            "--max-turns",
            "7",
            "--apply",
            "--diff",
            "--keep-worktree",
        ])
        .expect("parse");
        assert_eq!(a.task.as_deref(), Some("fix the bug"));
        assert_eq!(a.model.as_deref(), Some("claude-test"));
        assert_eq!(a.endpoint, "http://127.0.0.1:9999");
        assert_eq!(a.max_turns, 7);
        assert!(a.apply);
        assert!(a.diff);
        assert!(a.keep_worktree);
        assert!(!a.no_sandbox);
    }

    #[test]
    fn parse_defaults() {
        // Clear env influence for this unit test by passing model explicitly.
        let a = parse(&["do it", "--model", "m"]).expect("parse");
        assert_eq!(a.task.as_deref(), Some("do it"));
        assert_eq!(a.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(a.max_turns, DEFAULT_MAX_TURNS);
        assert!(!a.apply);
        assert!(!a.diff);
        assert!(!a.keep_worktree);
        assert!(!a.no_sandbox);
    }

    #[test]
    fn parse_no_sandbox_flag() {
        let a = parse(&["do it", "--model", "m", "--no-sandbox"]).expect("parse");
        assert!(a.no_sandbox);
    }

    #[test]
    fn parse_skip_selfcheck_flag() {
        let a = parse(&["do it", "--model", "m", "--skip-selfcheck"]).expect("parse");
        assert!(a.skip_selfcheck);
        let b = parse(&["do it", "--model", "m"]).expect("parse");
        assert!(!b.skip_selfcheck);
    }

    #[test]
    fn validate_missing_task_errors() {
        let a = parse(&["--model", "m"]).expect("parse allows absent task");
        assert!(a.task.is_none());
        assert_eq!(validate_args(&a), Err(EXIT_USAGE));
    }

    #[test]
    fn validate_missing_model_errors() {
        // Build args without relying on env: construct struct directly.
        let a = AgentArgs {
            task: Some("hi".into()),
            model: None,
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            apply: false,
            diff: false,
            keep_worktree: false,
            no_sandbox: false,
            skip_selfcheck: false,
        };
        assert_eq!(validate_args(&a), Err(EXIT_USAGE));
    }

    #[test]
    fn validate_empty_task_errors() {
        let a = AgentArgs {
            task: Some("   ".into()),
            model: Some("m".into()),
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            apply: false,
            diff: false,
            keep_worktree: false,
            no_sandbox: false,
            skip_selfcheck: false,
        };
        assert_eq!(validate_args(&a), Err(EXIT_USAGE));
    }

    #[test]
    fn help_text_covers_contract_and_exit_codes() {
        // Assert on the RENDERED clap help so override_help is what users see.
        use clap::CommandFactory;
        let help = AgentArgs::command().render_long_help().to_string();
        assert!(help.contains("127.0.0.1:8317"), "help={help}");
        assert!(help.contains("CLIProxyAPI"), "help={help}");
        assert!(help.contains("GREPPY_API_KEY"), "help={help}");
        assert!(help.contains("Exit codes"), "help={help}");
        assert!(help.contains("0  ok"), "help={help}");
        assert!(help.contains("2  no gateway"), "help={help}");
        assert!(help.contains("3  agent"), "help={help}");
        assert!(
            help.contains("4  --apply") || help.contains("4  "),
            "help={help}"
        );
        // Contract text exactly once: no duplicated "Usage:" section from clap.
        let usage_count = help.matches("Usage:").count();
        assert_eq!(
            usage_count, 1,
            "Usage: must appear exactly once; got {usage_count} in:\n{help}"
        );
        // F9: -p collision escape hatch.
        assert!(
            help.contains("greppy -e -p") || help.contains("leading `-p`"),
            "help missing -p escape hatch: {help}"
        );
        // Write-confinement honesty: roots listed, network open, --no-sandbox.
        assert!(
            help.contains("write-confined") || help.contains("write-confinement"),
            "help must describe write-confinement: {help}"
        );
        assert!(
            help.contains("--no-sandbox") || help.contains("no-sandbox"),
            "help must mention --no-sandbox: {help}"
        );
        assert!(
            help.contains("--skip-selfcheck") || help.contains("skip-selfcheck"),
            "help must mention --skip-selfcheck: {help}"
        );
        assert!(
            help.contains("network") || help.contains("reads and network"),
            "help must note network stays open: {help}"
        );
        // Single-tool surface + bash-smart.
        assert!(
            help.contains("exactly one tool") || help.contains("one tool"),
            "help must describe the single greppy tool surface: {help}"
        );
        assert!(
            help.contains("bash-smart"),
            "help must mention bash-smart: {help}"
        );
        // Plain-HTTP / no-TLS localhost contract (client has no TLS stack).
        assert!(
            help.contains("no TLS") || help.contains("plain-HTTP") || help.contains("plain HTTP"),
            "help must state plain-HTTP/no-TLS: {help}"
        );
    }

    #[test]
    fn writable_roots_include_worktree_temp_data_root_cargo_cache() {
        let wt = unique("roots-wt");
        fs::create_dir_all(&wt).unwrap();
        let roots = writable_roots_for(&wt);
        assert!(
            roots.iter().any(|r| r == &wt),
            "worktree missing: {roots:?}"
        );
        assert!(
            roots.iter().any(|r| r == &std::env::temp_dir()),
            "temp missing: {roots:?}"
        );
        let data = greppy_core::cache::data_root();
        assert!(
            roots.iter().any(|r| r == &data),
            "greppy data root missing: {roots:?}"
        );
        // Workspace store / locks / trash all live under data_root; granting the
        // parent is enough (and required for lifecycle leases).
        let store = greppy_core::workspace::store_dir(&wt);
        assert!(
            store.starts_with(&data),
            "store_dir {store:?} must live under data_root {data:?}"
        );
        assert!(
            greppy_core::cache::locks_root().starts_with(&data),
            "locks_root must live under data_root"
        );
        assert!(
            roots
                .iter()
                .any(|r| r.ends_with(".cargo") || r == &cargo_home_dir()),
            "cargo home missing: {roots:?}"
        );
        assert!(
            roots
                .iter()
                .any(|r| r.ends_with("Caches") || r.ends_with(".cache")),
            "platform cache missing: {roots:?}"
        );
        let _ = fs::remove_dir_all(&wt);
    }

    #[test]
    fn resolve_sandbox_mode_no_sandbox_is_off() {
        let a = AgentArgs {
            task: Some("t".into()),
            model: Some("m".into()),
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            apply: false,
            diff: false,
            keep_worktree: false,
            no_sandbox: true,
            skip_selfcheck: false,
        };
        let wt = unique("sb-off");
        fs::create_dir_all(&wt).unwrap();
        let mode = resolve_sandbox_mode(&a, &wt).expect("ok");
        assert!(matches!(mode, SandboxMode::Off));
        let _ = fs::remove_dir_all(&wt);
    }

    #[test]
    fn resolve_sandbox_mode_default_is_enforce() {
        let a = AgentArgs {
            task: Some("t".into()),
            model: Some("m".into()),
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            apply: false,
            diff: false,
            keep_worktree: false,
            no_sandbox: false,
            skip_selfcheck: false,
        };
        let wt = unique("sb-on");
        fs::create_dir_all(&wt).unwrap();
        let mode = resolve_sandbox_mode(&a, &wt).expect("ok");
        match mode {
            SandboxMode::Enforce(spec) => {
                assert!(!spec.writable_roots.is_empty());
                // Roots are pre-resolved (canonical); match via canonicalize.
                let wt_canon = fs::canonicalize(&wt).unwrap();
                assert!(
                    spec.writable_roots.iter().any(|r| r == &wt_canon),
                    "worktree missing from resolved roots: {:?} (want {wt_canon:?})",
                    spec.writable_roots
                );
                // Every root must be absolute (resolve-once invariant).
                assert!(spec.writable_roots.iter().all(|r| r.is_absolute()));
            }
            SandboxMode::Off => {
                // Only acceptable if preflight reported Unsupported (non-mac/linux).
                #[cfg(any(target_os = "macos", target_os = "linux"))]
                panic!("expected Enforce on macOS/Linux");
            }
        }
        let _ = fs::remove_dir_all(&wt);
    }

    #[test]
    fn is_agent_p_detection() {
        let mk = |parts: &[&str]| -> Vec<OsString> {
            parts.iter().map(|s| OsString::from(*s)).collect()
        };
        assert!(is_agent_p_invocation(&mk(&["greppy", "-p", "task"])));
        assert!(is_agent_p_invocation(&mk(&["greppy", "-p", "--help"])));
        assert!(is_agent_p_invocation(&mk(&[
            "greppy", "--root", "/tmp", "-p", "task"
        ])));
        assert!(!is_agent_p_invocation(&mk(&["greppy", "-R", "foo", "."])));
        assert!(!is_agent_p_invocation(&mk(&["greppy", "who-calls", "x"])));
        assert!(!is_agent_p_invocation(&mk(&["greppy", "pattern"])));
        // F9: `-e -p` is a grep passthrough spelling, not the agent.
        assert!(!is_agent_p_invocation(&mk(&["greppy", "-e", "-p", "X"])));
        assert!(!is_agent_p_invocation(&mk(&["greppy", "foo", "-p"])));
    }

    #[test]
    fn format_tool_start_truncates() {
        let long = "x".repeat(200);
        let args = serde_json::json!({"args": ["bash-smart", "--", "echo", long]});
        let line = format_tool_start("greppy", &args);
        assert!(line.starts_with("→ greppy "));
        assert!(line.chars().count() <= TOOL_LINE_MAX);
        assert!(line.ends_with('…'));
    }

    #[test]
    fn format_greppy_tool_line() {
        let args = serde_json::json!({"args": ["who-calls", "foo"]});
        assert_eq!(format_tool_start("greppy", &args), "→ greppy who-calls foo");
    }
}
