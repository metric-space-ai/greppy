//! `greppy agent` / `greppy -p` — interactive and one-shot coding agents over
//! an isolated git worktree.
//!
//! Intercepted in [`crate::run_os`] before grep-passthrough routing so that
//! ordinary `greppy -R …` / pattern invocations remain byte-exact real-grep.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use greppy_agent::{
    run_agent_loop, run_agent_loop_with_history, sandbox as agent_sandbox, AgentConfig,
    AgentWorkspace, Client, GreppyEnv, LoopEvent, LoopStop, ProbeError, RunOutcome, SandboxError,
    SandboxMode, StreamEvent, Usage, WorkspaceError,
};
use greppy_agent::system_prompt;

use crate::agent_tui::{
    bounded_pair, compact_messages, messages_from_protocol, new_session_id,
    protocol_from_persisted, redact_json, SessionCommand, SessionEvent, SessionRecord,
    SessionStore, TuiConfig,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Exit: success (clean, proposal, or applied).
pub const EXIT_OK: u8 = 0;
/// Exit: bad usage / missing model / gateway unreachable.
pub const EXIT_USAGE: u8 = 2;
/// Exit: agent loop / transport / workspace failure.
pub const EXIT_AGENT: u8 = 3;
/// Exit: `--apply` cherry-pick conflict.
pub const EXIT_CONFLICT: u8 = 4;
/// Exit: user cancelled interactive startup.
pub const EXIT_CANCELLED: u8 = 130;

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8317";
const DEFAULT_MAX_TURNS: usize = 40;
const TOOL_LINE_MAX: usize = 120;

const LONG_HELP: &str = "\
Coding agent with interactive and one-shot modes. Uses the installed portable
Chunk-CoW provider and fails before the first model request when its adapter
or persistent mount is not healthy. The immutable baseline includes the pinned
commit plus visible staged, unstaged and untracked state;
ignored files are excluded. It delivers a baseline-bound proposal ref
(refs/greppy/agent/<run_id>); inspect it with `git show` or apply it with
`greppy agent apply REF`. The agent has exactly one tool — `greppy` — covering
search/navigate/read/edit; commands run through that tool as
`bash-smart -- CMD`. The tool is write-confined to the worktree, a per-run
scratch dir (TMPDIR), the worktree's greppy store + lock namespace, and
~/.cargo/{registry,git}; reads and network stay open. Pass --no-sandbox
(or GREPPY_NO_SANDBOX=1) to disable. Repositories with submodules are not
supported yet (the agent worktree cannot reset them safely).

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
  greppy agent [\"INITIAL TASK\"]
  greppy -p \"TASK\" [--model M] [--endpoint URL] [--max-turns N]
                   [--deadline-secs N] [--apply] [--diff] [--keep-worktree]
                   [--no-sandbox] [--skip-selfcheck]
  greppy -p --help

Flags:
  --model M           One-shot model override (`greppy agent` can configure
                      and persist this through /setup)
  --endpoint URL      Gateway base URL (env GREPPY_ENDPOINT, else
                      http://127.0.0.1:8317)
  --max-turns N       Cap on assistant turns (default 40)
  --deadline-secs N   Wall-clock budget in seconds (env GREPPY_DEADLINE_SECS);
                      the loop stops between turns only — a running command is
                      never cut in half
  --apply             Apply only the Agent delta to the exact captured baseline;
                      the existing Git index remains byte-identical
  --diff              Print the full proposal patch after the stat
  --keep-worktree     Preserve the portable namespace and private delta
  --no-sandbox        Disable write-confinement (env GREPPY_NO_SANDBOX=1)
  --skip-selfcheck    Skip the startup capability self-check (env GREPPY_SKIP_SELFCHECK=1)

Interactive keys and commands (`greppy agent`):
  Enter               Send the prompt
  Shift/Alt+Enter     Insert a newline (when the terminal reports it)
  PageUp/PageDown     Scroll by a viewport page
  Mouse wheel         Scroll the transcript (Shift+drag still selects)
  End                 Resume follow-tail
  Tab / Shift+Tab     Completions and overlay choices
  Esc                 Close overlay, then completions
  Ctrl+C              Cancel a run at a safe tool boundary; idle exit;
                      twice to exit after the current non-interruptible tool
  /setup              Configure gateway, model, language, storage, sandbox,
                      acceleration, self-check, and workspace backend
  /help /clear /model /endpoint /usage /tools /copy /sessions /name /compact
  /exit /quit /q      Finish, restore the terminal, publish the proposal

Session flags:
  --continue          Restore this project's most recent interactive session
  --resume ID         Restore a specific session id

Exit codes:
  0  ok (clean, proposal saved, or applied)
  2  no gateway / bad usage / missing model / unsupported repository
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

    /// Model id (required in one-shot mode; interactive mode can configure it).
    #[arg(long, env = "GREPPY_MODEL")]
    pub model: Option<String>,

    /// Anthropic-Messages gateway base URL.
    #[arg(long, env = "GREPPY_ENDPOINT", default_value = DEFAULT_ENDPOINT)]
    pub endpoint: String,

    /// Maximum assistant turns.
    #[arg(long, default_value_t = DEFAULT_MAX_TURNS, value_name = "N")]
    pub max_turns: usize,

    /// Wall-clock budget in seconds (stops the loop between turns only).
    ///
    /// Also set by env `GREPPY_DEADLINE_SECS`.
    #[arg(long, env = "GREPPY_DEADLINE_SECS", value_name = "N")]
    pub deadline_secs: Option<u64>,

    /// Cherry-pick the proposal into the current checkout (staged).
    #[arg(long)]
    pub apply: bool,

    /// Print the full proposal patch after the stat.
    #[arg(long)]
    pub diff: bool,

    /// Preserve the portable namespace and private delta after a successful run.
    #[arg(long)]
    pub keep_worktree: bool,

    /// Disable shared immutable Base Store reuse for this run and build a
    /// complete private index. Intended for diagnostics and safe fallback.
    #[arg(long)]
    pub private_store: bool,

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
        require_equals = true,
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
        require_equals = true,
    )]
    pub skip_selfcheck: bool,

    /// Restore the most recent interactive session for this project.
    #[arg(long = "continue")]
    pub continue_session: bool,

    /// Restore a specific interactive session by id.
    #[arg(long, value_name = "SESSION_ID", conflicts_with = "continue_session")]
    pub resume: Option<String>,
}

/// True when argv (after greppy-owned globals) starts with `-p`.
pub fn is_agent_p_invocation(argv: &[std::ffi::OsString]) -> bool {
    let rest = super::grep_passthrough_args(argv);
    rest.first().is_some_and(|t| t == "-p")
}

/// True when argv (after greppy-owned globals) starts with `agent`.
pub fn is_agent_tui_invocation(argv: &[std::ffi::OsString]) -> bool {
    let rest = super::grep_passthrough_args(argv);
    rest.first().is_some_and(|token| token == "agent")
}

/// Parse and run `greppy -p …`. Caller must have verified [`is_agent_p_invocation`].
pub fn run_agent_p(argv: &[std::ffi::OsString]) -> u8 {
    run_agent_invocation(argv, false)
}

/// Parse and run `greppy agent …` in the full-screen interactive UI.
pub fn run_agent_tui(argv: &[std::ffi::OsString]) -> u8 {
    run_agent_invocation(argv, true)
}

fn run_agent_invocation(argv: &[std::ffi::OsString], interactive: bool) -> u8 {
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
    debug_assert!(rest
        .first()
        .is_some_and(|token| { token == if interactive { "agent" } else { "-p" } }));
    let after_p: Vec<std::ffi::OsString> = rest.iter().skip(1).cloned().collect();

    // Build a synthetic argv for clap: program name + flags/task after -p.
    let mut clap_argv: Vec<std::ffi::OsString> = Vec::with_capacity(after_p.len() + 1);
    clap_argv.push(std::ffi::OsString::from(if interactive {
        "greppy agent"
    } else {
        "greppy -p"
    }));
    clap_argv.extend(after_p.iter().cloned());

    let mut args = match AgentArgs::try_parse_from(&clap_argv) {
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

    let mut settings = crate::agent_tui::AgentSettings::default();
    if interactive {
        settings = crate::agent_tui::AgentSettings::load();
        apply_interactive_settings(&mut args, &after_p, &settings);
    }

    if let Err(code) = validate_args(&args, interactive) {
        return code;
    }
    if interactive && !crate::agent_tui::tty_suitable() {
        return crate::agent_tui::refuse_nontty();
    }

    let bootstrap = if interactive {
        match crate::agent_tui::BootstrapScreen::enter() {
            Ok(screen) => Some(screen),
            Err(error) => {
                eprintln!("greppy agent: cannot initialize startup screen: {error}");
                return EXIT_AGENT;
            }
        }
    } else {
        None
    };

    run_agent(args, interactive, bootstrap, settings)
}

fn has_long_option(argv: &[std::ffi::OsString], name: &str) -> bool {
    argv.iter().any(|value| {
        let value = value.to_string_lossy();
        value == name || value.starts_with(&format!("{name}="))
    })
}

fn apply_interactive_settings(
    args: &mut AgentArgs,
    argv: &[std::ffi::OsString],
    settings: &crate::agent_tui::AgentSettings,
) {
    if args.model.is_none() {
        args.model = settings.model.clone();
    }
    if !has_long_option(argv, "--endpoint") && std::env::var_os("GREPPY_ENDPOINT").is_none() {
        if let Some(endpoint) = settings
            .endpoint
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            args.endpoint = endpoint.clone();
        }
    }
    if !has_long_option(argv, "--private-store") {
        args.private_store |= settings.private_store;
    }
    if !has_long_option(argv, "--no-sandbox") && std::env::var_os("GREPPY_NO_SANDBOX").is_none() {
        args.no_sandbox = settings.no_sandbox;
    }
    if !has_long_option(argv, "--skip-selfcheck")
        && std::env::var_os("GREPPY_SKIP_SELFCHECK").is_none()
    {
        args.skip_selfcheck = settings.skip_selfcheck;
    }
    if std::env::var_os("GREPPY_DEVICE").is_none()
        && std::env::var_os("GREPPY_NO_GPU").is_none()
        && settings.acceleration == "cpu"
    {
        // This runs before workspace/index threads are created.
        unsafe { std::env::set_var("GREPPY_NO_GPU", "1") };
    }
}

fn validate_args(args: &AgentArgs, interactive: bool) -> Result<(), u8> {
    let task = args.task.as_deref().map(str::trim).unwrap_or("");
    if !interactive && task.is_empty() {
        eprintln!("error: missing TASK");
        eprintln!("usage: greppy -p \"TASK\" [--model M] …  (details: greppy -p --help)");
        return Err(EXIT_USAGE);
    }
    let model = args.model.as_deref().map(str::trim).unwrap_or("");
    if model.is_empty() && !interactive {
        eprintln!("error: --model is required (or set GREPPY_MODEL)");
        eprintln!("details: greppy -p --help");
        return Err(EXIT_USAGE);
    }
    if !interactive && (args.continue_session || args.resume.is_some()) {
        eprintln!("error: --continue / --resume are only valid with `greppy agent`");
        return Err(EXIT_USAGE);
    }
    Ok(())
}

fn run_agent(
    args: AgentArgs,
    interactive: bool,
    mut bootstrap: Option<crate::agent_tui::BootstrapScreen>,
    settings: crate::agent_tui::AgentSettings,
) -> u8 {
    let task = args.task.as_deref().unwrap_or("").trim().to_string();
    let mut model = args.model.as_deref().unwrap_or("").trim().to_string();
    let endpoint = args.endpoint.trim().to_string();

    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("greppy -p: cannot resolve current directory: {e}");
            return EXIT_AGENT;
        }
    };
    let shared_data_root = greppy_core::cache::data_root();

    // Stable and disposable agent worktrees have cache/run-id basenames that
    // are unrelated to the source repository. Pin one logical project name so
    // Base and Delta rows resolve identically in every worktree.
    std::env::remove_var(greppy_core::PROJECT_IDENTITY_ENV);
    let logical_project = greppy_core::project_identity(&cwd);
    std::env::set_var(greppy_core::PROJECT_IDENTITY_ENV, &logical_project);

    // Refuse unsupported repositories (e.g. tracked submodules) BEFORE any
    // worktree is created and BEFORE contacting the gateway.
    let run_id = make_run_id();
    let workspace = match AgentWorkspace::create(&cwd, &run_id) {
        Ok(ws) => ws,
        Err(WorkspaceError::Unsupported(reason)) => {
            // Stable user-facing message for the submodule case; fall back to
            // the typed reason for any future Unsupported variants.
            if reason.contains("gitmodules") || reason.contains("submodule") {
                eprintln!(
                    "greppy -p does not support repositories with submodules yet — \
                     the agent worktree cannot reset them safely. Run the task \
                     without -p, or remove the submodule from the working branch."
                );
            } else {
                eprintln!("greppy -p: unsupported repository: {reason}");
            }
            return EXIT_USAGE;
        }
        Err(WorkspaceError::AdapterUnavailable(reason)) => {
            eprintln!("greppy -p: portable CoW adapter is unavailable: {reason}");
            eprintln!("run `greppy workspace setup`, then `greppy workspace doctor --json`");
            return EXIT_USAGE;
        }
        Err(e @ WorkspaceError::Tampered { .. }) => {
            // Create/reuse-reset can surface Tampered when an existing stable
            // tree fails identity during a path that re-raises rather than
            // discards; keep the consistent exit-3 shape.
            report_tampered(&e, None);
            return EXIT_AGENT;
        }
        Err(e) => {
            eprintln!("greppy -p: workspace create failed: {e}");
            return EXIT_AGENT;
        }
    };
    if bootstrap
        .as_ref()
        .is_some_and(crate::agent_tui::BootstrapScreen::cancelled)
    {
        return EXIT_CANCELLED;
    }
    if let Some(screen) = bootstrap.as_mut() {
        screen.advance(1, "Preparing persistent data store");
    }

    let mut client = Client::new(&endpoint, &model);
    if let Ok(key) = std::env::var("GREPPY_API_KEY") {
        client = client.with_api_key(key);
    }
    if !interactive {
        match client.probe() {
            Ok(()) => {}
            Err(ProbeError::Unreachable(_)) => {
                eprintln!(
                    "greppy -p needs a local model gateway and found none at {endpoint}.\n\
                 Start one (standard: CLIProxyAPI on 127.0.0.1:8317) or set\n\
                 GREPPY_ENDPOINT / --endpoint. Details: greppy -p --help"
                );
                keep_worktree_on_error(&workspace);
                return EXIT_USAGE;
            }
            Err(ProbeError::BadResponse(detail)) => {
                eprintln!(
                    "greppy -p reached {endpoint}, but the gateway rejected the probe:\n\
                 {detail}\n\
                 If it requires an API key, set GREPPY_API_KEY. Details: greppy -p --help"
                );
                keep_worktree_on_error(&workspace);
                return EXIT_USAGE;
            }
        }
    } else if model.is_empty() {
        model = "auto".into();
        client = Client::new(&endpoint, &model);
        if let Ok(key) = std::env::var("GREPPY_API_KEY") {
            client = client.with_api_key(key);
        }
    }

    // Isolate greppy's on-disk store for this agent run into a dedicated data
    // root (not the operator's global greppy data). Prewarm + tool children
    // share it via GREPPY_STORE_DIR; the sandbox grants only this tree, not
    // the platform-wide Application Support / XDG data path.
    let agent_data = workspace.agent_data_root();
    if let Err(e) = std::fs::create_dir_all(&agent_data) {
        eprintln!("greppy -p: cannot create agent data root: {e}");
        keep_worktree_on_error(&workspace);
        return EXIT_AGENT;
    }

    let prepared_base = if args.private_store {
        crate::store_cow::configure_private_environment("explicit --private-store");
        if !interactive {
            eprintln!("store mode: private (--private-store)");
        }
        None
    } else if interactive {
        match crate::store_cow::try_reuse_base_store(&workspace, &shared_data_root) {
            Ok(Some(prepared)) => Some(prepared),
            Ok(None) => {
                crate::store_cow::configure_private_environment(
                    "shared Base not ready; using persistent interactive index",
                );
                None
            }
            Err(error) => {
                crate::store_cow::configure_private_environment(&error.to_string());
                None
            }
        }
    } else {
        match crate::store_cow::prepare_base_store(
            &workspace,
            &shared_data_root,
            crate::EmbeddingCliArgs {
                device: None,
                no_gpu: false,
            },
        ) {
            Ok(prepared) => {
                if !interactive {
                    eprintln!(
                        "store mode: overlay (Base {}, {})",
                        &prepared.identity_hash[..12],
                        if prepared.reused {
                            "reused"
                        } else {
                            "published"
                        }
                    );
                }
                Some(prepared)
            }
            Err(error) => {
                eprintln!(
                    "greppy -p: shared Base unavailable ({error}) — agent start aborted before the first model call"
                );
                if args.keep_worktree {
                    keep_worktree_on_error(&workspace);
                } else if let Err(cleanup_error) = workspace.cleanup() {
                    eprintln!(
                        "greppy -p: failed to clean the aborted portable workspace: {cleanup_error}"
                    );
                }
                return EXIT_AGENT;
            }
        }
    };
    if bootstrap
        .as_ref()
        .is_some_and(crate::agent_tui::BootstrapScreen::cancelled)
    {
        return EXIT_CANCELLED;
    }
    if let Some(screen) = bootstrap.as_mut() {
        screen.advance(2, "Starting background services");
    }
    std::env::set_var("GREPPY_STORE_DIR", &agent_data);
    if let Some(prepared) = &prepared_base {
        crate::store_cow::configure_overlay_environment(prepared, workspace.base_commit());
    }

    // Headless runs still prewarm synchronously. Interactive runs launch and
    // monitor the same index job only after the full TUI is visible.
    if !interactive {
        ensure_semantic_index(workspace.worktree_path());
    }
    if bootstrap
        .as_ref()
        .is_some_and(crate::agent_tui::BootstrapScreen::cancelled)
    {
        return EXIT_CANCELLED;
    }
    if let Some(screen) = bootstrap.as_mut() {
        screen.advance(3, "Checking execution environment");
    }

    // Per-run scratch (TMPDIR for tool children). Outside the stable-worktree
    // parent and lock sibling so those stay non-writable to tools.
    let scratch_dir = workspace.agent_scratch_root();
    if let Err(e) = std::fs::create_dir_all(&scratch_dir) {
        eprintln!("greppy -p: cannot create agent scratch dir: {e}");
        keep_worktree_on_error(&workspace);
        return EXIT_AGENT;
    }

    let sandbox_mode = match resolve_sandbox_mode(
        &args,
        workspace.worktree_path(),
        workspace.linked_git_dir(),
        &run_id,
        &scratch_dir,
        &agent_data,
    ) {
        Ok(mode) => mode,
        Err(code) => {
            keep_worktree_on_error(&workspace);
            return code;
        }
    };

    // Point tool children at the per-run scratch (also used by temp-file APIs
    // that honour TMPDIR). Set for the remainder of this process so every
    // sandboxed spawn inherits it without re-plumbing Command env.
    std::env::set_var("TMPDIR", &scratch_dir);
    // Some platforms also honour TMP/TEMP.
    std::env::set_var("TMP", &scratch_dir);
    std::env::set_var("TEMP", &scratch_dir);

    let mut env = match GreppyEnv::new(workspace.worktree_path().to_path_buf()) {
        Ok(env) => env.with_sandbox(sandbox_mode.clone()),
        Err(e) => {
            eprintln!("greppy -p: cannot build greppy env: {e}");
            keep_worktree_on_error(&workspace);
            return EXIT_AGENT;
        }
    };
    if let Some(screen) = bootstrap.as_mut() {
        screen.advance(4, "Opening interactive session");
    }

    // Capability self-check: fail loudly before the model loop rather than
    // silently degrading to a shell-only agent when index or sandbox is broken.
    if !interactive && !args.skip_selfcheck {
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

    // Wall-clock Instant is computed AFTER the self-check (and after
    // prewarm/index) so setup does not eat the budget — only the model loop
    // does. `deadline_total` mirrors the original N so the low-time advisory
    // can fire at 20% remaining.
    let (deadline, deadline_total) = match args.deadline_secs {
        Some(secs) => {
            let total = Duration::from_secs(secs);
            (Some(Instant::now() + total), Some(total))
        }
        None => (None, None),
    };

    let config = AgentConfig {
        max_turns: args.max_turns,
        system: Some(system_prompt()),
        model: model.clone(),
        deadline,
        deadline_total,
        ..AgentConfig::default()
    };

    let repository = cwd
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repository")
        .to_string();
    let session = if interactive {
        // Gateway discovery belongs to the visible setup phase in the worker;
        // never block before entering the alternate-screen UI.
        let models = (model != "auto")
            .then(|| model.clone())
            .into_iter()
            .collect();
        run_interactive_session(
            client,
            env,
            config.clone(),
            task.clone(),
            bootstrap.take(),
            InteractiveLaunch {
                model,
                repository,
                branch: git_branch(&cwd),
                worktree: workspace.worktree_path().display().to_string(),
                sandbox: sandbox_label(&sandbox_mode),
                run_id: workspace.run_id().to_string(),
                data_root: shared_data_root.clone(),
                project: logical_project.clone(),
                continue_session: args.continue_session,
                resume: args.resume.clone(),
                known_models: models,
                endpoint: endpoint.clone(),
                settings,
            },
        )
    } else {
        // The one-shot path needs the same browser wiring as the interactive
        // one: without a parent-owned attach token on fd 4, every `greppy web`
        // tool call dies with "requires a parent-owned attach token". The
        // interactive session sets this up; headless did not, so `greppy -p`
        // could never drive a browser even once it knew the verbs.
        //
        // One deliberate difference: a failure here is NOT fatal. A coding task
        // that never touches the web must not die because a browser token could
        // not be claimed — the agent still gets a clear error if it tries.
        std::env::set_var("GREPPY_RUN_ID", workspace.run_id());
        // Point parent and tool children at this run's own runtime directory --
        // the one the sandbox grants. Without it they fall back to the shared
        // /tmp/greppy-daemon-<uid>, which the child may not write.
        std::env::set_var("GREPPY_RUNTIME_DIR", agent_runtime_dir(workspace.run_id()));
        match crate::web_attach::claim_persistent_parent() {
            Ok(_) => {
                let _ = greppy_agent::greppy_env::PREPARE_ATTACH_FD
                    .set(crate::web_attach::inherit_attach_for_agent);
                // Start the runtime HERE, while this process is still
                // unsandboxed. The runtime sandboxes its own workers, and macOS
                // Seatbelt does not nest: started from a sandboxed tool child it
                // dies with "worker sandbox: Operation not permitted", and the
                // agent only ever sees "web-runtime did not create its socket".
                // Started here, the tool children merely connect.
                if let Err(error) = crate::web::prestart_unsandboxed() {
                    eprintln!("greppy: browser unavailable to the agent ({error})");
                }
            }
            Err(error) => {
                eprintln!("greppy: browser unavailable to the agent (no attach token: {error})");
            }
        }
        // A one-shot run must not leave a runtime behind: a long-lived one
        // degrades until navigation stops working.
        struct ShutdownWebOnDrop;
        impl Drop for ShutdownWebOnDrop {
            fn drop(&mut self) {
                crate::web::shutdown_if_running();
            }
        }
        let _shutdown_web = ShutdownWebOnDrop;
        run_headless_session(&mut client, &mut env, &config, &task)
    };
    let session = match session {
        Ok(session) => session,
        Err(error) => {
            eprintln!(
                "greppy {}: agent error: {error}",
                if interactive { "agent" } else { "-p" }
            );
            keep_worktree_on_error(&workspace);
            return EXIT_AGENT;
        }
    };

    eprintln!(
        "tokens: in {} out {} (cache read {}, write {}) over {} turns",
        session.usage.input_tokens,
        session.usage.output_tokens,
        session.usage.cache_read_input_tokens,
        session.usage.cache_creation_input_tokens,
        session.turns
    );
    report_stop(
        session.last_stop.as_ref(),
        &args,
        &config,
        &mut io::stderr().lock(),
    );

    let commit_subject = if task.is_empty() {
        "interactive agent session"
    } else {
        &task
    };
    let commit_message = truncate_chars(commit_subject, 72);
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let outcome = match workspace.finish(&commit_message) {
        Ok(o) => o,
        Err(e @ WorkspaceError::Tampered { .. }) => {
            report_tampered_to(&e, Some(workspace.worktree_path()), &mut stderr);
            // Tree is already kept by the error path; do not call cleanup.
            drop(workspace);
            return EXIT_AGENT;
        }
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
        let wt_path = workspace.worktree_path().to_path_buf();
        if let Err(e) = workspace.cleanup() {
            // Any cleanup failure is a non-zero exit — a successful run
            // whose cleanup fails is not a success.
            return map_cleanup_error(&e, Some(&wt_path), &mut stderr);
        }
    } else if exit != EXIT_OK {
        // Conflict still cleans unless keep — success-path cleanup only when
        // exit is 0. Spec: cleanup on every successful run; keep on error.
        // Conflict is exit 4 (error-ish): keep worktree.
        keep_worktree_on_error(&workspace);
    } else {
        let path = workspace.worktree_path().display().to_string();
        if let Err(error) = workspace.keep() {
            let _ = writeln!(stderr, "greppy -p: cannot preserve workspace: {error}");
            return EXIT_AGENT;
        }
        let _ = writeln!(stderr, "worktree kept: {path}");
        drop(workspace);
    }

    exit
}

#[derive(Debug, Default)]
struct SessionSummary {
    usage: Usage,
    turns: u64,
    last_stop: Option<LoopStop>,
}

fn run_headless_session(
    client: &mut Client,
    env: &mut GreppyEnv,
    config: &AgentConfig,
    task: &str,
) -> Result<SessionSummary, String> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let mut tool_line_open = false;
    let mut turns = 0u64;
    let result = run_agent_loop(client, env, config, task, &mut |event| {
        if matches!(event, LoopEvent::TurnComplete { .. }) {
            turns = turns.saturating_add(1);
        }
        handle_loop_event(event, &mut stdout, &mut stderr, &mut tool_line_open);
    })
    .map_err(|error| error.to_string())?;

    if tool_line_open {
        let _ = writeln!(stderr);
    }
    if !result.final_text.is_empty() && !result.final_text.ends_with('\n') {
        let _ = writeln!(stdout);
    }
    let _ = stdout.flush();

    Ok(SessionSummary {
        usage: result.usage,
        turns,
        last_stop: Some(result.stop),
    })
}

struct InteractiveLaunch {
    model: String,
    endpoint: String,
    repository: String,
    branch: String,
    worktree: String,
    sandbox: String,
    run_id: String,
    data_root: PathBuf,
    project: String,
    continue_session: bool,
    resume: Option<String>,
    known_models: Vec<String>,
    settings: crate::agent_tui::AgentSettings,
}

fn run_interactive_session(
    mut client: Client,
    mut env: GreppyEnv,
    mut config: AgentConfig,
    initial_task: String,
    bootstrap: Option<crate::agent_tui::BootstrapScreen>,
    launch: InteractiveLaunch,
) -> Result<SessionSummary, String> {
    let store = SessionStore::new(&launch.data_root, &launch.project);
    let mut record = if let Some(id) = launch.resume.as_deref() {
        store
            .load(id)
            .map_err(|error| format!("cannot resume session {id}: {error}"))?
    } else if launch.continue_session {
        store
            .latest()
            .map_err(|error| format!("cannot continue session: {error}"))?
            .ok_or_else(|| "no previous interactive session for this project".to_string())?
    } else {
        SessionRecord::new(
            new_session_id(),
            launch.project.clone(),
            launch.model.clone(),
            launch.run_id.clone(),
        )
    };
    record.model = launch.model.clone();
    record.run_id = launch.run_id.clone();
    // Same run_id for every greppy web subprocess in this interactive session
    // so the supervisor socket and sessions persist across tool calls.
    std::env::set_var("GREPPY_RUN_ID", &launch.run_id);
    crate::web_attach::claim_persistent_parent()
        .map_err(|error| format!("failed to create parent-owned web attach token: {error}"))?;
    let _ = greppy_agent::greppy_env::PREPARE_ATTACH_FD
        .set(crate::web_attach::inherit_attach_for_agent);
    struct ShutdownWebOnDrop;
    impl Drop for ShutdownWebOnDrop {
        fn drop(&mut self) {
            crate::web::shutdown_if_running();
        }
    }
    let _shutdown_web = ShutdownWebOnDrop;
    record.worktree = launch.worktree.clone();
    record.branch = launch.branch.clone();
    if let Err(error) = store.create(&record) {
        if error.kind() != io::ErrorKind::AlreadyExists {
            eprintln!("greppy agent: session save failed: {error}");
        }
    }

    let cancel = Arc::new(AtomicBool::new(false));
    config.cancel = Some(Arc::clone(&cancel));
    config.model = launch.model.clone();
    let ui_cancel = Arc::clone(&cancel);
    let history = protocol_from_persisted(&record.messages);
    let restored_usage = record.usage;
    let restored_turns = record.turns;

    let (command_tx, command_rx) = mpsc::channel();
    let (bridge, intake) = bounded_pair();
    let store_for_worker = store.clone();
    let mut session_id = record.id.clone();
    let startup_worktree = PathBuf::from(&launch.worktree);
    // Gateway/model discovery also runs in the visible setup phase, even when
    // the index is already warm.
    let initializing = true;
    let mut current_endpoint = launch.endpoint.clone();
    let gateway_api_key = std::env::var("GREPPY_API_KEY").ok();

    let index_monitor_cancel = Arc::new(AtomicBool::new(false));
    let monitor_bridge = bridge.clone();
    let monitor_cancel = Arc::clone(&index_monitor_cancel);
    let index_monitor = Some(
        thread::Builder::new()
            .name("greppy-agent-index-monitor".to_string())
            .spawn(move || {
                let ready = if let Some(mut job) = start_semantic_index(&startup_worktree) {
                    match monitor_index_startup(
                        &mut job,
                        &startup_worktree,
                        &monitor_bridge,
                        &monitor_cancel,
                    ) {
                        Ok(ready) => ready,
                        Err(error) => {
                            monitor_bridge.send_discrete(SessionEvent::Warning(format!(
                                "Index monitoring failed: {error}"
                            )));
                            false
                        }
                    }
                } else {
                    true
                };
                if ready || std::env::var_os("GREPPY_TEST_SKIP_INFERENCE").is_some() {
                    monitor_bridge.send_discrete(SessionEvent::BackgroundReady);
                } else if !monitor_cancel.load(Ordering::Relaxed) {
                    monitor_bridge.send_discrete(SessionEvent::SetupBlocked(
                        "The repository's one-time code analysis did not complete. Retry the index before running the agent."
                            .into(),
                    ));
                }
            })
            .map_err(|error| format!("cannot start index monitor: {error}"))?,
    );

    let worker = thread::Builder::new()
        .name("greppy-agent-session".to_string())
        .spawn(move || {
            let mut history = history;
            let mut summary = SessionSummary {
                usage: restored_usage,
                turns: restored_turns,
                last_stop: None,
            };
            if cancel.load(Ordering::Relaxed) {
                return Ok(summary);
            }
            bridge.send_setup_progress(SessionEvent::SetupProgress {
                phase: "Connecting model gateway".into(),
                detail: None,
                unit: "steps".into(),
                completed: 0,
                total: 0,
                rate_milli_per_second: None,
                eta_seconds: None,
                elapsed_seconds: 0,
            });
            let gateway_ready = match client.list_models() {
                Ok(models) if !models.is_empty() => {
                    if config.model == "auto" {
                        config.model = models[0].clone();
                        if let Err(error) = store_for_worker.set_model(&session_id, &config.model) {
                            bridge.send_discrete(SessionEvent::Warning(format!(
                                "session save failed: {error}"
                            )));
                        }
                        client = client_for_endpoint(
                            &current_endpoint,
                            &config.model,
                            gateway_api_key.as_deref(),
                        );
                    }
                    bridge.send_discrete(SessionEvent::Configuration {
                        endpoint: current_endpoint.clone(),
                        model: config.model.clone(),
                        models,
                    });
                    true
                }
                Ok(_) => {
                    bridge.send_discrete(SessionEvent::GatewayRequired(
                        "The gateway reports no available models. Enter another gateway URL below."
                            .into(),
                    ));
                    false
                }
                Err(_) => {
                    bridge.send_discrete(SessionEvent::GatewayRequired(
                        "No model gateway detected. Enter its URL below.".into(),
                    ));
                    false
                }
            };
            if gateway_ready {
                bridge.send_discrete(SessionEvent::SetupReady);
            }
            let mut tool_started = std::collections::HashMap::<String, Instant>::new();
            while let Ok(command) = command_rx.recv() {
                match command {
                    SessionCommand::Quit => break,
                    SessionCommand::Cancel => {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    SessionCommand::SetModel(model) => {
                        config.model = model.clone();
                        if let Err(error) = store_for_worker.set_model(&session_id, &model) {
                            bridge.send_discrete(SessionEvent::Warning(format!(
                                "session save failed: {error}"
                            )));
                        }
                    }
                    SessionCommand::SetEndpoint(endpoint) => {
                        let candidate = client_for_endpoint(
                            &endpoint,
                            &config.model,
                            gateway_api_key.as_deref(),
                        );
                        match candidate.list_models() {
                            Ok(models) if !models.is_empty() => {
                                current_endpoint = endpoint;
                                if config.model == "auto"
                                    || !models.iter().any(|model| model == &config.model)
                                {
                                    config.model = models[0].clone();
                                }
                                client = client_for_endpoint(
                                    &current_endpoint,
                                    &config.model,
                                    gateway_api_key.as_deref(),
                                );
                                bridge.send_discrete(SessionEvent::Configuration {
                                    endpoint: current_endpoint.clone(),
                                    model: config.model.clone(),
                                    models,
                                });
                                bridge.send_discrete(SessionEvent::SetupReady);
                            }
                            Ok(_) => bridge.send_discrete(SessionEvent::EndpointRejected {
                                endpoint,
                                message: "The gateway reports no available models. Enter another gateway URL below."
                                    .into(),
                            }),
                            Err(_) => bridge.send_discrete(SessionEvent::EndpointRejected {
                                endpoint,
                                message: "That gateway is unreachable. Check the URL and GREPPY_API_KEY, then try again."
                                    .into(),
                            }),
                        }
                    }
                    SessionCommand::Resume(next_session_id) => {
                        match store_for_worker.load(&next_session_id) {
                            Ok(next) => {
                                history = protocol_from_persisted(&next.messages);
                                summary.usage = next.usage;
                                summary.turns = next.turns;
                                summary.last_stop = None;
                                config.model = next.model;
                                session_id = next.id;
                                cancel.store(false, Ordering::Relaxed);
                            }
                            Err(error) => {
                                bridge.send_discrete(SessionEvent::Error(format!(
                                    "cannot resume session {next_session_id}: {error}"
                                )));
                            }
                        }
                    }
                    SessionCommand::Compact => {
                        let compacted = compact_messages(&messages_from_protocol(&history), 8);
                        history = protocol_from_persisted(&compacted);
                        if let Err(error) =
                            store_for_worker.append_message_checkpoint(&session_id, &compacted)
                        {
                            bridge.send_discrete(SessionEvent::Warning(format!(
                                "session save failed: {error}"
                            )));
                        }
                        bridge.send_discrete(SessionEvent::Compacted {
                            messages: compacted,
                        });
                    }
                    SessionCommand::Prompt(prompt) => {
                        cancel.store(false, Ordering::Relaxed);
                        let mut prompt_turns = 0u64;
                        let previous_message_count = history.len();
                        let result = run_agent_loop_with_history(
                            &mut client,
                            &mut env,
                            &config,
                            &history,
                            &prompt,
                            &mut |event| {
                                if matches!(event, LoopEvent::TurnComplete { .. }) {
                                    prompt_turns = prompt_turns.saturating_add(1);
                                }
                                match event {
                                    LoopEvent::Stream(StreamEvent::TextDelta { text }) => {
                                        bridge.send_text(&text);
                                    }
                                    LoopEvent::Stream(StreamEvent::ThinkingDelta { text }) => {
                                        bridge.send_thinking(&text);
                                    }
                                    LoopEvent::ToolStart {
                                        call_id,
                                        name,
                                        arguments,
                                    } => {
                                        tool_started.insert(call_id.clone(), Instant::now());
                                        let summary =
                                            format_tool_start(&name, &redact_json(&arguments));
                                        bridge.send_discrete(SessionEvent::ToolStart {
                                            id: call_id,
                                            summary,
                                        });
                                    }
                                    LoopEvent::ToolFinish {
                                        call_id, outcome, ..
                                    } => {
                                        let elapsed_ms = tool_started
                                            .remove(&call_id)
                                            .map(|started| started.elapsed().as_millis() as u64)
                                            .unwrap_or(0);
                                        bridge.send_discrete(SessionEvent::ToolFinish {
                                            id: call_id,
                                            failed: outcome.is_error,
                                            elapsed_ms,
                                            preview: truncate_chars(&outcome.content, 400),
                                        });
                                    }
                                    LoopEvent::Stream(_) | LoopEvent::TurnComplete { .. } => {}
                                }
                            },
                        );

                        match result {
                            Ok(result) => {
                                history = result.messages;
                                add_usage(&mut summary.usage, &result.usage);
                                summary.turns = summary.turns.saturating_add(prompt_turns);
                                summary.last_stop = Some(result.stop.clone());
                                let persisted = messages_from_protocol(&history);
                                let new_messages = messages_from_protocol(
                                    &history[previous_message_count.min(history.len())..],
                                );
                                if let Err(error) =
                                    store_for_worker.append_messages(&session_id, &new_messages)
                                {
                                    bridge.send_discrete(SessionEvent::Warning(format!(
                                        "session save failed: {error}"
                                    )));
                                }
                                if let Err(error) = store_for_worker.append_usage(
                                    &session_id,
                                    &summary.usage,
                                    summary.turns,
                                    stop_label(&result.stop),
                                ) {
                                    bridge.send_discrete(SessionEvent::Warning(format!(
                                        "session save failed: {error}"
                                    )));
                                }
                                bridge.send_discrete(SessionEvent::Done {
                                    input_tokens: result.usage.input_tokens,
                                    output_tokens: result.usage.output_tokens,
                                    cache_read: result.usage.cache_read_input_tokens,
                                    cache_write: result.usage.cache_creation_input_tokens,
                                    turns: prompt_turns,
                                    stop: stop_label(&result.stop).to_string(),
                                    messages: persisted,
                                });
                            }
                            Err(error) => {
                                let message = error.to_string();
                                bridge.send_discrete(SessionEvent::Error(message.clone()));
                                return Err(message);
                            }
                        }
                    }
                }
            }
            Ok(summary)
        })
        .map_err(|error| format!("cannot start session worker: {error}"))?;

    let mut initial_prompts = Vec::new();
    if !initial_task.trim().is_empty() {
        initial_prompts.push(initial_task);
    }
    let mut initial_draft = String::new();
    // Keep bootstrap cleanup ownership through every fallible session setup
    // operation. Transfer only when TerminalGuard is about to adopt the same
    // terminal; any earlier return still restores raw mode and the alt screen.
    if let Some(screen) = bootstrap {
        let handoff = screen.handoff();
        initial_prompts.extend(handoff.queued);
        initial_draft = handoff.draft;
    }
    let ui_result = crate::agent_tui::run(
        TuiConfig {
            model: launch.model,
            endpoint: launch.endpoint,
            repository: launch.repository,
            branch: launch.branch,
            worktree: launch.worktree,
            sandbox: launch.sandbox,
            known_models: launch.known_models,
            cancel: ui_cancel,
            initializing,
            settings: launch.settings,
        },
        record,
        store,
        initial_prompts,
        initial_draft,
        command_tx.clone(),
        intake,
    );
    let _ = command_tx.send(SessionCommand::Quit);
    index_monitor_cancel.store(true, Ordering::Relaxed);
    let worker_result = worker
        .join()
        .map_err(|_| "session worker panicked".to_string())?;
    if let Some(index_monitor) = index_monitor {
        // `/exit` must restore the terminal immediately. A cold-start indexer
        // can be inside model inference or database publication and may not
        // observe cancellation promptly, so only join work that is already
        // complete; dropping the handle detaches the background cleanup.
        if index_monitor.is_finished() {
            let _ = index_monitor.join();
        }
    }
    ui_result.map_err(|error| format!("terminal UI failed: {error}"))?;
    worker_result
}

fn git_branch(path: &Path) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .stdin(std::process::Stdio::null())
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "detached".to_string(),
    }
}

fn sandbox_label(mode: &SandboxMode) -> String {
    match mode {
        SandboxMode::Off => "sandbox off".to_string(),
        SandboxMode::Enforce(_) => {
            if cfg!(target_os = "macos") {
                "sandbox seatbelt".to_string()
            } else if cfg!(target_os = "linux") {
                "sandbox landlock".to_string()
            } else {
                "sandbox on".to_string()
            }
        }
    }
}

fn add_usage(total: &mut Usage, next: &Usage) {
    total.input_tokens = total.input_tokens.saturating_add(next.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(next.output_tokens);
    total.cache_read_input_tokens = total
        .cache_read_input_tokens
        .saturating_add(next.cache_read_input_tokens);
    total.cache_creation_input_tokens = total
        .cache_creation_input_tokens
        .saturating_add(next.cache_creation_input_tokens);
}

fn stop_label(stop: &LoopStop) -> &'static str {
    match stop {
        LoopStop::EndTurn => "ready",
        LoopStop::MaxTokens => "token limit reached",
        LoopStop::MaxTurns => "turn limit reached",
        LoopStop::Stuck => "stopped after repeated tool failures",
        LoopStop::Deadline => "deadline reached",
        LoopStop::Cancelled => "cancelled",
    }
}

fn report_stop(
    stop: Option<&LoopStop>,
    args: &AgentArgs,
    config: &AgentConfig,
    stderr: &mut impl Write,
) {
    match stop {
        Some(LoopStop::MaxTurns) => {
            let _ = writeln!(
                stderr,
                "stopped: turn limit reached ({}) — the result may be incomplete",
                args.max_turns
            );
        }
        Some(LoopStop::Stuck) => {
            let n = config.consecutive_failure_stop;
            let _ = writeln!(
                stderr,
                "stopped: {n} consecutive tool failures — the agent could not make progress"
            );
        }
        Some(LoopStop::Deadline) => {
            let secs = args.deadline_secs.unwrap_or(0);
            let _ = writeln!(
                stderr,
                "stopped: wall-clock deadline reached ({secs}s) — the result may be incomplete"
            );
        }
        Some(LoopStop::Cancelled) => {
            let _ = writeln!(stderr, "stopped: cancelled by user");
        }
        Some(LoopStop::EndTurn | LoopStop::MaxTokens) | None => {}
    }
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
    // Tool arguments are model-controlled and are written directly into a
    // terminal transcript. Keep the useful first line while dropping escape
    // and other control characters before the outer line cap is applied.
    fn clip(text: &str, max: usize) -> String {
        let text = text.trim();
        let first_line = text.lines().next().unwrap_or("").trim_end();
        let mut out: String = first_line
            .chars()
            .filter(|ch| !ch.is_control() || *ch == '\t')
            .take(max)
            .collect();
        if first_line.chars().count() > max || text.lines().count() > 1 {
            out.push('…');
        }
        out
    }

    fn nonblank_field<'a>(arguments: &'a serde_json::Value, field: &str) -> Option<&'a str> {
        arguments
            .get(field)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
    }

    // The production surface currently has exactly one tool. Preserve its
    // argv-oriented summary (which is also the most useful command/path/
    // pattern summary) while redacting values for headless callers too.
    let redacted = redact_json(arguments);
    let body = match name {
        "greppy" => {
            let joined = redacted
                .get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            format!("→ greppy {}", clip(&joined, TOOL_LINE_MAX))
        }
        other => {
            // Keep meaningful invocation fields readable if an extension or
            // future caller supplies a tool-shaped object; raw JSON remains a
            // safe last resort for malformed/opaque arguments.
            let summary = nonblank_field(&redacted, "command")
                .map(str::to_string)
                .or_else(|| {
                    nonblank_field(&redacted, "pattern").map(|pattern| {
                        nonblank_field(&redacted, "path").map_or_else(
                            || pattern.to_string(),
                            |path| format!("{pattern} in {path}"),
                        )
                    })
                })
                .or_else(|| nonblank_field(&redacted, "path").map(str::to_string));
            let summary = summary.unwrap_or_else(|| redacted.to_string());
            format!("→ {other} {}", clip(&summary, TOOL_LINE_MAX))
        }
    };
    truncate_chars(&body, TOOL_LINE_MAX)
}

fn keep_worktree_on_error(workspace: &AgentWorkspace) {
    if let Err(error) = workspace.keep() {
        eprintln!("greppy -p: could not mark failed workspace as kept: {error}");
    }
    eprintln!(
        "worktree kept for debugging: {}",
        workspace.worktree_path().display()
    );
}

/// Consistent Tampered diagnostic: name the **worktree directory** (not a
/// nested `.git` control file) and state that it was kept for inspection.
/// Used by create/reuse-reset, finish, and cleanup paths so exit 3 always has
/// the same shape.
fn report_tampered(err: &WorkspaceError, worktree: Option<&Path>) {
    let mut stderr = io::stderr().lock();
    report_tampered_to(err, worktree, &mut stderr);
}

fn report_tampered_to(err: &WorkspaceError, worktree: Option<&Path>, stderr: &mut impl Write) {
    let kept = match (worktree, err) {
        (Some(wt), _) => wt.to_path_buf(),
        (None, WorkspaceError::Tampered { path, .. }) => {
            // Prefer the worktree directory when the error path is its `.git`.
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n == ".git")
            {
                path.parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| path.clone())
            } else {
                path.clone()
            }
        }
        (None, _) => PathBuf::from("<unknown>"),
    };
    let _ = writeln!(
        stderr,
        "greppy -p: {err}\n\
         worktree kept for inspection: {}",
        kept.display()
    );
}

/// Map a cleanup failure to an exit code.
///
/// **Any** cleanup failure is a non-zero exit: a successful run whose cleanup
/// fails is not a success.
///
/// - [`WorkspaceError::Tampered`] → [`EXIT_AGENT`] (3); message names the
///   worktree directory and states the tree was kept for inspection.
/// - other errors (stable reset/clean, temp removal, …) → [`EXIT_AGENT`] (3);
///   message names the failure and the worktree path that was kept.
fn map_cleanup_error(err: &WorkspaceError, worktree: Option<&Path>, stderr: &mut impl Write) -> u8 {
    match err {
        WorkspaceError::Tampered { .. } => {
            report_tampered_to(err, worktree, stderr);
            EXIT_AGENT
        }
        other => {
            let kept = worktree
                .map(|wt| wt.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("<unknown>"));
            let _ = writeln!(
                stderr,
                "greppy -p: worktree cleanup failed: {other}\n\
                 worktree kept: {}",
                kept.display()
            );
            EXIT_AGENT
        }
    }
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
/// Otherwise prepare the run's narrow writable roots **exactly once**
/// ([`agent_sandbox::resolve_enforce_spec`]: create + full-path symlink
/// validation + canonicalize) and probe the platform backend. The resulting
/// `Enforce` spec carries those fixed canonical roots for the whole agent run;
/// per-tool `apply` never re-resolves them.
///
/// `Unsupported` (Linux without Landlock ABI ≥ V3) warns once and falls back
/// to `Off`; any other error aborts with [`EXIT_AGENT`].
fn resolve_sandbox_mode(
    args: &AgentArgs,
    worktree_path: &Path,
    git_dir: &Path,
    run_id: &str,
    scratch_dir: &Path,
    agent_data: &Path,
) -> Result<SandboxMode, u8> {
    if args.no_sandbox {
        eprintln!("sandbox disabled");
        return Ok(SandboxMode::Off);
    }
    let raw = writable_roots_for(worktree_path, git_dir, run_id, scratch_dir, agent_data);
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

/// Per-run directory for this agent's browser runtime socket.
///
/// The web runtime needs a Unix socket, and `ensure_private_dir` **writes**
/// (`set_permissions`) even when the directory already exists — so a sandboxed
/// tool child cannot use the operator's shared `/tmp/greppy-daemon-<uid>`.
///
/// Giving the run its own directory beats widening the sandbox onto the shared
/// one: the agent's browser can then never touch the sockets of the operator's
/// other greppy daemons, and the grant dies with the run.
///
/// Must stay short. `RuntimeScope::from_env` only honours `GREPPY_RUNTIME_DIR`
/// as a socket directory when it is <= 32 bytes, because a Unix socket path is
/// capped near 104 bytes — which is exactly why the default is under `/tmp`
/// and not the long macOS temp root.
pub(crate) fn agent_runtime_dir(run_id: &str) -> std::path::PathBuf {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in run_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // "/tmp/greppy-agent-" + 8 hex = 26 bytes, comfortably under the cap.
    std::path::PathBuf::from(format!("/tmp/greppy-agent-{:08x}", hash as u32))
}

/// Writable roots for a sandboxed `-p` tool subprocess.
///
/// Deliberately narrow — each entry has a one-line reason. Do **not** re-add
/// the platform cache, global temp root, the operator's greppy data root, or
/// whole Cargo home: those defeat per-worktree isolation and the stable-tree lock.
fn writable_roots_for(
    worktree_path: &Path,
    git_dir: &Path,
    run_id: &str,
    scratch_dir: &Path,
    agent_data: &Path,
) -> Vec<std::path::PathBuf> {
    let mut roots = Vec::with_capacity(7);

    // Worktree: agent proposal edits and worktree-local builds land here.
    roots.push(worktree_path.to_path_buf());

    // Private Git control namespace: index, refs, locks and new objects are a
    // second WorkspaceCore CoW namespace. Normal Git commands must be able to
    // update it without granting the provider mount parent or another Agent's
    // Git state.
    roots.push(git_dir.to_path_buf());

    // Scratch: per-run temp only (TMPDIR points here). Never the global temp root.
    roots.push(scratch_dir.to_path_buf());

    // Isolated greppy data root for this agent worktree (GREPPY_STORE_DIR).
    // Contains the worktree's store, locks/, and trash/ under one tree that is
    // not the operator's global Application Support / XDG greppy data. Index-
    // backed commands call ensure_workspace_store which also touches locks +
    // trash under data_root — granting this isolated root covers them without
    // opening every other workspace index. Lease files live under
    // `<agent_data>/locks/`; the sandbox only grants directories (Seatbelt
    // subpath / Landlock PathBeneath), so locks/ cannot be narrowed to a single
    // lease file without a file-level grant API.
    roots.push(agent_data.to_path_buf());

    // Cargo download caches only — never ~/.cargo/bin, config.toml, credentials*.
    let cargo = cargo_home_dir();
    let cargo_registry = cargo.join("registry");
    let cargo_git = cargo.join("git");
    let _ = std::fs::create_dir_all(&cargo_registry);
    let _ = std::fs::create_dir_all(&cargo_git);
    roots.push(cargo_registry);
    roots.push(cargo_git);

    // This run's own browser-runtime socket directory. Narrow on purpose: NOT
    // the shared /tmp/greppy-daemon-<uid>, so the agent cannot reach the
    // sockets of the operator's other greppy daemons. Without it the tool child
    // dies with "cannot allocate web-runtime socket", because ensure_private_dir
    // calls set_permissions even on an existing directory.
    let runtime_dir = agent_runtime_dir(run_id);
    if std::fs::create_dir_all(&runtime_dir).is_ok() {
        // 0700 up front so the child's own ensure_private_dir finds what it
        // wants and the socket is never group/world reachable.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(
                &runtime_dir,
                std::fs::Permissions::from_mode(0o700),
            );
        }
    }
    roots.push(runtime_dir);

    // Stable-worktree PARENT and the sibling lock file must stay outside every
    // tool-writable root (worktree path itself is root 1; its parent is not).
    roots
}

fn cargo_home_dir() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("CARGO_HOME") {
        return std::path::PathBuf::from(home);
    }
    home_dir().join(".cargo")
}

fn home_dir() -> std::path::PathBuf {
    #[cfg(target_os = "windows")]
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        return std::path::PathBuf::from(profile);
    }
    if let Some(h) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(h);
    }
    std::env::temp_dir()
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

/// Start the existing index command as its normal observable background job.
/// The interactive worker monitors the same JSON record used by `index status`
/// and renders it in the TUI; no indexing or embedding logic is duplicated.
fn start_semantic_index(worktree_path: &Path) -> Option<crate::BackgroundJobLaunch> {
    if doctor_reports_embedding_complete(worktree_path) {
        return None;
    }
    let root = worktree_path.to_string_lossy().into_owned();
    crate::spawn_agent_background_index(Some(&root), "agent-startup")
}

fn client_for_endpoint(endpoint: &str, model: &str, api_key: Option<&str>) -> Client {
    let client = Client::new(endpoint, model);
    match api_key {
        Some(key) => client.with_api_key(key),
        None => client,
    }
}

fn monitor_index_startup(
    launch: &mut crate::BackgroundJobLaunch,
    worktree_path: &Path,
    bridge: &crate::agent_tui::EventBridge,
    cancel: &AtomicBool,
) -> Result<bool, String> {
    let mut missing_ticks = 0usize;
    loop {
        if cancel.load(Ordering::Relaxed) {
            let owned = matches!(launch, crate::BackgroundJobLaunch::Owned { .. });
            cancel_background_job(launch);
            bridge.send_discrete(SessionEvent::Warning(
                if owned {
                    "Indexing cancelled."
                } else {
                    "Startup monitoring stopped; the shared index job continues."
                }
                .into(),
            ));
            return Ok(false);
        }

        if let Some(job) = crate::read_background_job(launch.path()) {
            missing_ticks = 0;
            let state = job
                .get("state")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("starting");
            if state == "failed" {
                let detail = job
                    .get("last_error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown error");
                bridge.send_discrete(SessionEvent::Warning(format!("Indexing failed: {detail}")));
                reap_owned_background_job(launch);
                return Ok(false);
            }
            let completed = job
                .get("completed_spans")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(0);
            let total = job
                .get("total_spans")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(0);
            let backend = job
                .get("backend")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let embedding_phase = match backend {
                "metal" => "Generating embeddings (Metal GPU)",
                "cuda" => "Generating embeddings (CUDA GPU)",
                "cpu" => "Generating embeddings (CPU)",
                _ => "Generating embeddings",
            };
            let phase = match state {
                "analyzing" => "Analyzing source code",
                "storing" | "indexing" => "Writing code index",
                "loading_model" => "Loading embedding model",
                "embedding" => embedding_phase,
                "refreshing" | "starting" => "Preparing code index",
                _ => "Preparing workspace",
            };
            let show_counters = matches!(state, "analyzing" | "embedding");
            let rate_milli_per_second = show_counters
                .then(|| {
                    job.get("rate_milli_spans_per_second")
                        .and_then(serde_json::Value::as_u64)
                })
                .flatten();
            let eta_seconds = show_counters
                .then(|| job.get("eta_seconds").and_then(serde_json::Value::as_u64))
                .flatten();
            let detail = job
                .get("current_detail")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            bridge.send_setup_progress(SessionEvent::BackgroundProgress {
                phase: phase.into(),
                detail,
                unit: if state == "embedding" {
                    "spans".into()
                } else if state == "analyzing" {
                    "files".into()
                } else {
                    "items".into()
                },
                completed: if show_counters { completed } else { 0 },
                total: if show_counters { total } else { 0 },
                rate_milli_per_second,
                eta_seconds,
            });
        } else if doctor_reports_embedding_complete(worktree_path) {
            reap_owned_background_job(launch);
            return Ok(true);
        } else {
            if !owned_background_job_is_running(launch) {
                missing_ticks = missing_ticks.saturating_add(1);
            }
            if missing_ticks >= 20 {
                if std::env::var_os("GREPPY_TEST_SKIP_INFERENCE").is_none() {
                    bridge.send_discrete(SessionEvent::Warning(
                        "The index job ended before embeddings were complete.".into(),
                    ));
                }
                reap_owned_background_job(launch);
                return Ok(false);
            }
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn cancel_background_job(launch: &mut crate::BackgroundJobLaunch) {
    if let crate::BackgroundJobLaunch::Owned { child, path } = launch {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(path);
    }
}

fn reap_owned_background_job(launch: &mut crate::BackgroundJobLaunch) {
    if let crate::BackgroundJobLaunch::Owned { child, .. } = launch {
        let _ = child.wait();
    }
}

fn owned_background_job_is_running(launch: &mut crate::BackgroundJobLaunch) -> bool {
    match launch {
        crate::BackgroundJobLaunch::Owned { child, .. } => {
            matches!(child.try_wait(), Ok(None) | Err(_))
        }
        crate::BackgroundJobLaunch::Attached { .. } => false,
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

pub(crate) fn scrub_credential_env(cmd: &mut Command) {
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
    fn interactive_settings_supply_flag_free_startup_defaults() {
        let mut args = parse(&[]).expect("parse");
        let settings = crate::agent_tui::AgentSettings {
            endpoint: Some("http://127.0.0.1:18318".into()),
            model: Some("configured-model".into()),
            private_store: true,
            no_sandbox: true,
            skip_selfcheck: true,
            ..crate::agent_tui::AgentSettings::default()
        };
        apply_interactive_settings(&mut args, &[], &settings);
        assert_eq!(args.endpoint, "http://127.0.0.1:18318");
        assert_eq!(args.model.as_deref(), Some("configured-model"));
        assert!(args.private_store);
        assert!(args.no_sandbox);
        assert!(args.skip_selfcheck);
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
            "--deadline-secs",
            "1800",
            "--apply",
            "--diff",
            "--keep-worktree",
        ])
        .expect("parse");
        assert_eq!(a.task.as_deref(), Some("fix the bug"));
        assert_eq!(a.model.as_deref(), Some("claude-test"));
        assert_eq!(a.endpoint, "http://127.0.0.1:9999");
        assert_eq!(a.max_turns, 7);
        assert_eq!(a.deadline_secs, Some(1800));
        assert!(a.apply);
        assert!(a.diff);
        assert!(a.keep_worktree);
        assert!(!a.no_sandbox);
    }

    /// Serializes the tests that mutate `GREPPY_DEADLINE_SECS`.
    ///
    #[test]
    fn parse_defaults() {
        // Clear env influence for this unit test by passing model explicitly.
        // Also clear GREPPY_DEADLINE_SECS so env cannot inject a default.
        let _serialized = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("GREPPY_DEADLINE_SECS");
        std::env::remove_var("GREPPY_DEADLINE_SECS");
        let a = parse(&["do it", "--model", "m"]).expect("parse");
        assert_eq!(a.task.as_deref(), Some("do it"));
        assert_eq!(a.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(a.max_turns, DEFAULT_MAX_TURNS);
        assert_eq!(a.deadline_secs, None);
        assert!(!a.apply);
        assert!(!a.diff);
        assert!(!a.keep_worktree);
        assert!(!a.no_sandbox);
        match prev {
            Some(v) => std::env::set_var("GREPPY_DEADLINE_SECS", v),
            None => std::env::remove_var("GREPPY_DEADLINE_SECS"),
        }
    }

    #[test]
    fn parse_deadline_secs_flag() {
        let _serialized = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("GREPPY_DEADLINE_SECS");
        std::env::remove_var("GREPPY_DEADLINE_SECS");
        let a = parse(&["do it", "--model", "m", "--deadline-secs", "42"]).expect("parse");
        assert_eq!(a.deadline_secs, Some(42));
        match prev {
            Some(v) => std::env::set_var("GREPPY_DEADLINE_SECS", v),
            None => std::env::remove_var("GREPPY_DEADLINE_SECS"),
        }
    }

    #[test]
    fn parse_deadline_secs_from_env() {
        let _serialized = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("GREPPY_DEADLINE_SECS");
        std::env::set_var("GREPPY_DEADLINE_SECS", "99");
        let a = parse(&["do it", "--model", "m"]).expect("parse");
        assert_eq!(a.deadline_secs, Some(99));
        match prev {
            Some(v) => std::env::set_var("GREPPY_DEADLINE_SECS", v),
            None => std::env::remove_var("GREPPY_DEADLINE_SECS"),
        }
    }

    #[test]
    fn removed_workspace_backend_and_fresh_flags_are_rejected() {
        assert!(parse(&["do it", "--model", "m", "--fresh"]).is_err());
        assert!(parse(&["do it", "--model", "m", "--workspace-backend", "native",]).is_err());
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
        assert_eq!(validate_args(&a, false), Err(EXIT_USAGE));
        assert_eq!(validate_args(&a, true), Ok(()));
    }

    #[test]
    fn validate_missing_model_errors() {
        // Build args without relying on env: construct struct directly.
        let a = AgentArgs {
            task: Some("hi".into()),
            model: None,
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            deadline_secs: None,
            apply: false,
            diff: false,
            keep_worktree: false,
            private_store: false,
            no_sandbox: false,
            skip_selfcheck: false,
            continue_session: false,
            resume: None,
        };
        assert_eq!(validate_args(&a, false), Err(EXIT_USAGE));
    }

    #[test]
    fn validate_empty_task_errors() {
        let a = AgentArgs {
            task: Some("   ".into()),
            model: Some("m".into()),
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            deadline_secs: None,
            apply: false,
            diff: false,
            keep_worktree: false,
            private_store: false,
            no_sandbox: false,
            skip_selfcheck: false,
            continue_session: false,
            resume: None,
        };
        assert_eq!(validate_args(&a, false), Err(EXIT_USAGE));
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
            help.contains("ignored files are excluded"),
            "help must document ignored-file exclusion: {help}"
        );
        assert!(help.contains("/model"), "help must mention /model: {help}");
        assert!(
            help.contains("--continue"),
            "help must mention --continue: {help}"
        );
        assert!(
            help.contains("--resume"),
            "help must mention --resume: {help}"
        );
        assert!(
            help.contains("ignored build caches") || help.contains("ignored files"),
            "help must document ignored-cache default: {help}"
        );
        assert!(!help.contains("--workspace-backend"), "help={help}");
        assert!(!help.contains("--fresh"), "help={help}");
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
        // Wall-clock deadline: flag + between-turns semantics.
        assert!(
            help.contains("--deadline-secs") || help.contains("deadline-secs"),
            "help must mention --deadline-secs: {help}"
        );
        assert!(
            help.contains("between turns") || help.contains("never cut in half"),
            "help must state deadline stops between turns: {help}"
        );
    }

    #[test]
    fn deadline_stop_line_shape() {
        // Mirror the exact stderr shape printed on LoopStop::Deadline.
        let secs: u64 = 1800;
        let line = format!(
            "stopped: wall-clock deadline reached ({secs}s) — the result may be incomplete"
        );
        assert!(line.contains("stopped: wall-clock deadline reached (1800s)"));
        assert!(line.contains("the result may be incomplete"));
    }

    #[test]
    fn writable_roots_are_narrow_no_global_shared_state() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let wt = unique("roots-wt");
        fs::create_dir_all(&wt).unwrap();
        let git_dir = unique("roots-git-state");
        fs::create_dir_all(&git_dir).unwrap();
        let run_id = "run-roots-test";
        let scratch = unique("roots-scratch");
        fs::create_dir_all(&scratch).unwrap();
        let agent_data = unique("roots-agent-data");
        fs::create_dir_all(&agent_data).unwrap();
        // Point data_root() at the isolated agent data for this test process
        // so any cache helpers agree with writable_roots_for.
        let prev_store = std::env::var_os("GREPPY_STORE_DIR");
        std::env::set_var("GREPPY_STORE_DIR", &agent_data);

        let roots = writable_roots_for(&wt, &git_dir, run_id, &scratch, &agent_data);

        // Worktree + scratch + isolated agent data present.
        assert!(
            roots.iter().any(|r| r == &wt),
            "worktree missing: {roots:?}"
        );
        assert!(
            roots.iter().any(|r| r == &git_dir),
            "private Git CoW namespace missing: {roots:?}"
        );
        assert!(
            roots.iter().any(|r| r == &scratch),
            "scratch missing: {roots:?}"
        );
        assert!(
            roots.iter().any(|r| r == &agent_data),
            "agent data root missing: {roots:?}"
        );

        // No global temp / platform cache / whole cargo home.
        let global_temp = std::env::temp_dir();
        assert!(
            !roots.iter().any(|r| r == &global_temp),
            "global temp must not be a root: {roots:?}"
        );
        let cargo = cargo_home_dir();
        assert!(
            !roots.iter().any(|r| r == &cargo),
            "whole cargo home must not be granted: {roots:?}"
        );
        assert!(
            roots.iter().any(|r| r == &cargo.join("registry")),
            "cargo registry missing: {roots:?}"
        );
        assert!(
            roots.iter().any(|r| r == &cargo.join("git")),
            "cargo git missing: {roots:?}"
        );
        assert!(
            !roots.iter().any(|r| {
                let s = r.to_string_lossy();
                (s.ends_with("Caches") || s.ends_with(".cache"))
                    && !s.contains("greppy-agent-data")
                    && !s.contains("greppy-agent-scratch")
            }),
            "platform cache must not be a root: {roots:?}"
        );

        // Under the isolated agent data, store + locks live beneath agent_data.
        let store = greppy_core::cache::workspace_store_dir(&wt);
        assert!(
            store.starts_with(&agent_data),
            "store {store:?} must live under agent data {agent_data:?}"
        );
        let locks = greppy_core::cache::locks_root();
        assert!(
            locks.starts_with(&agent_data),
            "locks {locks:?} must live under agent data {agent_data:?}"
        );

        match prev_store {
            Some(v) => std::env::set_var("GREPPY_STORE_DIR", v),
            None => std::env::remove_var("GREPPY_STORE_DIR"),
        }
        let _ = fs::remove_dir_all(&wt);
        let _ = fs::remove_dir_all(&git_dir);
        let _ = fs::remove_dir_all(&scratch);
        let _ = fs::remove_dir_all(&agent_data);
    }

    #[test]
    fn resolve_sandbox_mode_no_sandbox_is_off() {
        let a = AgentArgs {
            task: Some("t".into()),
            model: Some("m".into()),
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            deadline_secs: None,
            apply: false,
            diff: false,
            keep_worktree: false,
            private_store: false,
            no_sandbox: true,
            skip_selfcheck: false,
            continue_session: false,
            resume: None,
        };
        let wt = unique("sb-off");
        fs::create_dir_all(&wt).unwrap();
        let git_dir = unique("sb-off-git-state");
        fs::create_dir_all(&git_dir).unwrap();
        let run_id = wt.file_name().unwrap().to_string_lossy().into_owned();
        let scratch = unique("sb-off-scratch");
        fs::create_dir_all(&scratch).unwrap();
        let agent_data = unique("sb-off-agent-data");
        fs::create_dir_all(&agent_data).unwrap();
        let mode =
            resolve_sandbox_mode(&a, &wt, &git_dir, &run_id, &scratch, &agent_data).expect("ok");
        assert!(matches!(mode, SandboxMode::Off));
        let _ = fs::remove_dir_all(&wt);
        let _ = fs::remove_dir_all(&git_dir);
        let _ = fs::remove_dir_all(&scratch);
        let _ = fs::remove_dir_all(&agent_data);
    }

    #[test]
    fn resolve_sandbox_mode_default_is_enforce() {
        let a = AgentArgs {
            task: Some("t".into()),
            model: Some("m".into()),
            endpoint: DEFAULT_ENDPOINT.into(),
            max_turns: 40,
            deadline_secs: None,
            apply: false,
            diff: false,
            keep_worktree: false,
            private_store: false,
            no_sandbox: false,
            skip_selfcheck: false,
            continue_session: false,
            resume: None,
        };
        let wt = unique("sb-on");
        fs::create_dir_all(&wt).unwrap();
        let git_dir = unique("sb-on-git-state");
        fs::create_dir_all(&git_dir).unwrap();
        let run_id = wt.file_name().unwrap().to_string_lossy().into_owned();
        let scratch = unique("sb-on-scratch");
        fs::create_dir_all(&scratch).unwrap();
        let agent_data = unique("sb-on-agent-data");
        fs::create_dir_all(&agent_data).unwrap();
        let mode =
            resolve_sandbox_mode(&a, &wt, &git_dir, &run_id, &scratch, &agent_data).expect("ok");
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
                let git_canon = fs::canonicalize(&git_dir).unwrap();
                assert!(
                    spec.writable_roots.iter().any(|r| r == &git_canon),
                    "private Git namespace missing from resolved roots: {:?}",
                    spec.writable_roots
                );
                // Every root must be absolute (resolve-once invariant).
                assert!(spec.writable_roots.iter().all(|r| r.is_absolute()));
                // Isolated agent data is present; the operator global data root is not
                // (unless GREPPY_STORE_DIR was already overridden to match — compare
                // against the default-looking Application Support / XDG path only when
                // it differs from agent_data).
                let agent_data_canon = fs::canonicalize(&agent_data).unwrap();
                assert!(
                    spec.writable_roots.iter().any(|r| r == &agent_data_canon),
                    "agent data missing from resolved roots: {:?}",
                    spec.writable_roots
                );
            }
            SandboxMode::Off => {
                // Only acceptable if preflight reported Unsupported (non-mac/linux).
                #[cfg(any(target_os = "macos", target_os = "linux"))]
                panic!("expected Enforce on macOS/Linux");
            }
        }
        let _ = fs::remove_dir_all(&wt);
        let _ = fs::remove_dir_all(&git_dir);
        let _ = fs::remove_dir_all(&scratch);
        let _ = fs::remove_dir_all(&agent_data);
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
        assert!(is_agent_tui_invocation(&mk(&["greppy", "agent"])));
        assert!(is_agent_tui_invocation(&mk(&[
            "greppy", "--root", "/tmp", "agent"
        ])));
        assert!(!is_agent_tui_invocation(&mk(&["greppy", "agent.rs"])));
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

    #[test]
    fn format_tool_start_drops_terminal_controls() {
        let args = serde_json::json!({
            "args": ["bash-smart", "--", "echo", "\u{1b}[31mred\u{7f}"]
        });
        let line = format_tool_start("greppy", &args);
        assert!(!line.contains('\u{1b}'));
        assert!(!line.contains('\u{7f}'));
        assert!(line.contains("[31mred"), "line={line}");
    }

    #[test]
    fn format_tool_start_prefers_meaningful_fields_to_raw_json() {
        assert_eq!(
            format_tool_start(
                "extension-tool",
                &serde_json::json!({"pattern": "TODO", "path": "src"})
            ),
            "→ extension-tool TODO in src"
        );
        assert_eq!(
            format_tool_start(
                "extension-tool",
                &serde_json::json!({"command": "cargo test", "path": "ignored"})
            ),
            "→ extension-tool cargo test"
        );
    }

    #[test]
    fn store_cleanup_skipped_under_agent_run_env() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os(greppy_agent::AGENT_RUN_ENV);
        std::env::set_var(greppy_agent::AGENT_RUN_ENV, "1");
        // Must return immediately — no GC under GREPPY_AGENT_RUN (WP21).
        crate::maybe_run_store_cleanup(None);
        match prev {
            Some(v) => std::env::set_var(greppy_agent::AGENT_RUN_ENV, v),
            None => std::env::remove_var(greppy_agent::AGENT_RUN_ENV),
        }
    }

    #[test]
    fn cleanup_tampered_maps_to_exit_agent_and_names_path() {
        // Error path is the worktree directory itself.
        let path = PathBuf::from("/tmp/greppy-agent-wt-inspect");
        let err = WorkspaceError::Tampered {
            path: path.clone(),
            detail: "pointer mismatch".into(),
        };
        let mut stderr = Vec::new();
        let code = map_cleanup_error(&err, None, &mut stderr);
        assert_eq!(code, EXIT_AGENT);
        let msg = String::from_utf8_lossy(&stderr);
        assert!(msg.contains("worktree kept for inspection"), "msg={msg}");
        assert!(
            msg.contains(path.to_string_lossy().as_ref()),
            "msg must name the worktree path: {msg}"
        );
        assert!(
            msg.contains("untrustworthy") || msg.contains("pointer mismatch"),
            "msg={msg}"
        );
    }

    #[test]
    fn cleanup_tampered_names_worktree_when_error_path_is_dot_git() {
        // Common identity-failure path is `<worktree>/.git`; diagnostic must
        // still name the worktree directory, not the control file.
        let wt = PathBuf::from("/tmp/greppy-agent-wt-inspect");
        let git_file = wt.join(".git");
        let err = WorkspaceError::Tampered {
            path: git_file,
            detail: "pointer mismatch".into(),
        };
        let mut stderr = Vec::new();
        let code = map_cleanup_error(&err, None, &mut stderr);
        assert_eq!(code, EXIT_AGENT);
        let msg = String::from_utf8_lossy(&stderr);
        assert!(msg.contains("worktree kept for inspection"), "msg={msg}");
        assert!(
            msg.contains(wt.to_string_lossy().as_ref()),
            "msg must name the worktree directory (not only .git): {msg}"
        );
        // The "kept for inspection" line should end with the worktree, not `.git`.
        let kept_line = msg
            .lines()
            .find(|l| l.contains("worktree kept for inspection"))
            .expect("kept line");
        assert!(
            !kept_line.trim_end().ends_with(".git"),
            "kept path must be the worktree dir, got: {kept_line}"
        );
    }

    #[test]
    fn report_tampered_with_explicit_worktree_is_consistent() {
        // Finish / create paths pass the known worktree directory explicitly.
        let wt = PathBuf::from("/tmp/greppy-agent-finish-wt");
        let err = WorkspaceError::Tampered {
            path: wt.join(".git"),
            detail: "rewritten pointer".into(),
        };
        let mut stderr = Vec::new();
        report_tampered_to(&err, Some(&wt), &mut stderr);
        let msg = String::from_utf8_lossy(&stderr);
        assert!(msg.contains("worktree kept for inspection"), "msg={msg}");
        assert!(
            msg.contains(wt.to_string_lossy().as_ref()),
            "msg must name explicit worktree: {msg}"
        );
        assert!(
            msg.contains("untrustworthy") || msg.contains("rewritten"),
            "msg={msg}"
        );
    }

    #[test]
    fn report_tampered_create_path_shape() {
        // Create/reuse-reset path: no separate worktree arg beyond the error.
        let wt = PathBuf::from("/tmp/greppy-agent-create-wt");
        let err = WorkspaceError::Tampered {
            path: wt.clone(),
            detail: "registration mismatch".into(),
        };
        let mut stderr = Vec::new();
        report_tampered_to(&err, None, &mut stderr);
        let msg = String::from_utf8_lossy(&stderr);
        assert!(msg.starts_with("greppy -p: "), "msg={msg}");
        assert!(msg.contains("worktree kept for inspection"), "msg={msg}");
        assert!(
            msg.contains(wt.to_string_lossy().as_ref()),
            "msg must name worktree: {msg}"
        );
    }

    #[test]
    fn cleanup_non_tampered_forces_nonzero_exit() {
        // Any cleanup failure (reset/clean/remove) is a non-zero exit; a
        // successful run whose cleanup fails is not a success.
        let wt = PathBuf::from("/tmp/greppy-agent-cleanup-fail-wt");
        let err = WorkspaceError::GitFailed {
            command: "git worktree remove".into(),
            stderr: "boom".into(),
            status: Some(128),
        };
        let mut stderr = Vec::new();
        let code = map_cleanup_error(&err, Some(&wt), &mut stderr);
        assert_eq!(
            code, EXIT_AGENT,
            "non-Tampered cleanup failure must force exit 3"
        );
        let msg = String::from_utf8_lossy(&stderr);
        assert!(msg.contains("worktree cleanup failed"), "msg={msg}");
        assert!(
            msg.contains(wt.to_string_lossy().as_ref()),
            "msg must name the worktree path: {msg}"
        );
        assert!(msg.contains("worktree kept"), "msg={msg}");
    }

    #[test]
    fn cleanup_non_tampered_without_worktree_arg_still_exits_nonzero() {
        let err = WorkspaceError::GitFailed {
            command: "git clean".into(),
            stderr: "permission denied".into(),
            status: Some(1),
        };
        let mut stderr = Vec::new();
        let code = map_cleanup_error(&err, None, &mut stderr);
        assert_eq!(code, EXIT_AGENT);
        let msg = String::from_utf8_lossy(&stderr);
        assert!(msg.contains("worktree cleanup failed"), "msg={msg}");
        assert!(msg.contains("worktree kept"), "msg={msg}");
    }

    #[test]
    fn parse_continue_and_resume_flags() {
        let a = parse(&["--model", "m", "--continue"]).expect("parse");
        assert!(a.continue_session);
        assert!(a.task.is_none());
        let b = parse(&["--model", "m", "--resume", "sess-1"]).expect("parse");
        assert_eq!(b.resume.as_deref(), Some("sess-1"));
        assert!(parse(&["--model", "m", "--continue", "--resume", "x"]).is_err());
        let headless = parse(&["task", "--model", "m", "--continue"]).expect("parse");
        assert_eq!(validate_args(&headless, false), Err(EXIT_USAGE));
        assert_eq!(validate_args(&headless, true), Ok(()));
    }

    #[test]
    fn interactive_agent_needs_neither_task_nor_model_flags() {
        let args = parse(&[]).expect("parse flagless interactive invocation");
        assert_eq!(validate_args(&args, true), Ok(()));
        assert_eq!(validate_args(&args, false), Err(EXIT_USAGE));
    }

    #[test]
    fn cancelling_attached_index_job_never_touches_foreign_job_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("foreign-index-job.json");
        fs::write(&path, "{}\n").expect("write job marker");
        let mut launch = crate::BackgroundJobLaunch::Attached { path: path.clone() };

        cancel_background_job(&mut launch);

        assert!(path.exists(), "attached job belongs to another process");
    }

    #[test]
    fn cancelling_owned_index_job_reaps_child_and_removes_its_job_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("owned-index-job.json");
        fs::write(&path, "{}\n").expect("write job marker");
        let child = std::process::Command::new("sh")
            .args(["-c", "sleep 30"])
            .spawn()
            .expect("spawn child");
        let mut launch = crate::BackgroundJobLaunch::Owned {
            child,
            path: path.clone(),
        };

        cancel_background_job(&mut launch);

        assert!(!path.exists());
        let crate::BackgroundJobLaunch::Owned { child, .. } = &mut launch else {
            panic!("expected owned launch");
        };
        assert!(child.try_wait().expect("query child").is_some());
    }

    #[test]
    fn long_help_mentions_submodule_limitation() {
        assert!(
            LONG_HELP.contains("submodule"),
            "LONG_HELP must state the submodule limitation"
        );
    }
}
