//! `greppy -p` — one-shot coding agent over an isolated git worktree.
//!
//! Intercepted in [`crate::run_os`] before grep-passthrough routing so that
//! ordinary `greppy -R …` / pattern invocations remain byte-exact real-grep.

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use greppy_agent::{
    run_agent_loop, AgentConfig, AgentWorkspace, Client, GreppyEnv, LoopEvent, ProbeError,
    RunOutcome, StreamEvent, WorkspaceError, SYSTEM_PROMPT,
};
use greppy_core::workspace as workspace_locator;

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
One-shot coding agent. Runs the task in an isolated git worktree of the current
repository, never writing to your checkout until you apply. When the agent
proposes changes they are pinned to refs/greppy/agent/<run_id> as a review
patch you can inspect (`git show`) or apply (`git cherry-pick -n`).

Localhost contract: greppy -p talks to an Anthropic-Messages-compatible
gateway at GREPPY_ENDPOINT (default http://127.0.0.1:8317). The standard is
CLIProxyAPI, which translates all major chat formats/providers to that wire.
Any compatible server works — e.g. a local llama.cpp/ollama behind such a
gateway. GREPPY_MODEL / --model is the model id passed through unchanged.
If the gateway requires an API key (CLIProxyAPI usually does), set
GREPPY_API_KEY; it is sent as x-api-key and Authorization: Bearer. There is
no key flag on purpose — keys do not belong on the command line.

Usage:
  greppy -p \"TASK\" [--model M] [--endpoint URL] [--max-turns N]
                   [--apply] [--diff] [--keep-worktree]
  greppy -p --help

Flags:
  --model M           Model id (required; env GREPPY_MODEL if flag omitted)
  --endpoint URL      Gateway base URL (env GREPPY_ENDPOINT, else
                      http://127.0.0.1:8317)
  --max-turns N       Cap on assistant turns (default 40)
  --apply             Cherry-pick the proposal into the current checkout
                      (staged, not committed)
  --diff              Print the full proposal patch after the stat
  --keep-worktree     Leave the disposable worktree on disk after success

Exit codes:
  0  ok (clean, proposal saved, or applied)
  2  no gateway / bad usage / missing model
  3  agent or loop error (worktree kept for debugging)
  4  --apply cherry-pick conflict (ref still available)
";

/// Parsed `greppy -p` arguments (everything after the leading `-p` token).
#[derive(Debug, Clone, Parser, PartialEq, Eq)]
#[command(
    name = "greppy -p",
    about = "One-shot coding agent in an isolated worktree; proposals only touch your checkout when you apply.",
    long_about = LONG_HELP,
    disable_version_flag = true,
    after_help = "Set GREPPY_MODEL (or pass --model). Details: greppy -p --help"
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

    /// Keep the disposable worktree after a successful run.
    #[arg(long)]
    pub keep_worktree: bool,
}

/// True when argv (after greppy-owned globals) starts with `-p`.
pub fn is_agent_p_invocation(argv: &[std::ffi::OsString]) -> bool {
    let rest = super::grep_passthrough_args(argv);
    rest.first().is_some_and(|t| t == "-p")
}

/// Parse and run `greppy -p …`. Caller must have verified [`is_agent_p_invocation`].
pub fn run_agent_p(argv: &[std::ffi::OsString]) -> u8 {
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

    seed_store_from_main(workspace.repo_root(), workspace.worktree_path());

    let mut env = match GreppyEnv::new(workspace.worktree_path().to_path_buf()) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("greppy -p: cannot build greppy env: {e}");
            keep_worktree_on_error(&workspace);
            return EXIT_AGENT;
        }
    };

    let config = AgentConfig {
        max_turns: args.max_turns,
        system: Some(SYSTEM_PROMPT.to_string()),
        model: model.clone(),
        ..AgentConfig::default()
    };

    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let mut tool_line_open = false;

    let loop_result = run_agent_loop(&mut client, &mut env, &config, &task, &mut |event| {
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

    // Ensure a trailing newline after streamed assistant text.
    if !loop_result.final_text.is_empty() && !loop_result.final_text.ends_with('\n') {
        let _ = writeln!(stdout);
    }
    let _ = stdout.flush();

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
        "bash" => {
            let cmd = arguments
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            format!("→ bash {cmd}")
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

/// Seed the worktree's cold store from the main checkout's store, if present.
///
/// Prints one stderr line describing the outcome. On macOS prefers
/// `/bin/cp -Rc` (APFS clonefile); falls back to a recursive `std::fs` copy.
pub fn seed_store_from_main(main_root: &Path, worktree_path: &Path) {
    let src = workspace_locator::store_dir(main_root);
    // Dest AFTER the worktree exists so canonicalize/hash is correct.
    let dst = workspace_locator::store_dir(worktree_path);

    if !src.exists() {
        eprintln!("no index to seed — first run will build cold");
        return;
    }
    if dst.exists() {
        // Already present (unusual for a fresh worktree hash); leave it.
        eprintln!("seeded index from main checkout");
        return;
    }

    if let Some(parent) = dst.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!("no index to seed — first run will build cold ({e})");
            return;
        }
    }

    let copied = try_seed_copy(&src, &dst);
    if copied {
        eprintln!("seeded index from main checkout");
    } else {
        // Best-effort: leave dest absent so the agent rebuilds.
        let _ = fs::remove_dir_all(&dst);
        eprintln!("no index to seed — first run will build cold");
    }
}

fn try_seed_copy(src: &Path, dst: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        if try_cp_clonefile(src, dst) {
            return true;
        }
    }
    copy_dir_recursive(src, dst).is_ok()
}

#[cfg(target_os = "macos")]
fn try_cp_clonefile(src: &Path, dst: &Path) -> bool {
    let Some(src_s) = src.to_str() else {
        return false;
    };
    let Some(dst_s) = dst.to_str() else {
        return false;
    };
    match Command::new("/bin/cp").args(["-Rc", src_s, dst_s]).status() {
        Ok(status) => status.success(),
        Err(_) => false,
    }
}

/// Recursive directory copy used as the portable seed fallback.
pub fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if file_type.is_file() {
            fs::copy(&from, &to)?;
        }
        // Symlinks and specials are skipped deliberately.
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
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
        };
        assert_eq!(validate_args(&a), Err(EXIT_USAGE));
    }

    #[test]
    fn help_text_covers_contract_and_exit_codes() {
        let help = LONG_HELP;
        assert!(help.contains("127.0.0.1:8317"));
        assert!(help.contains("CLIProxyAPI"));
        assert!(help.contains("Exit codes"));
        assert!(help.contains("0  ok"));
        assert!(help.contains("2  no gateway"));
        assert!(help.contains("3  agent"));
        assert!(help.contains("4  --apply"));
        assert!(
            help.lines().count() <= 40,
            "help lines={}",
            help.lines().count()
        );
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
    }

    #[test]
    fn copy_dir_recursive_copies_nested_files() {
        let src = unique("copy-src");
        let dst = unique("copy-dst");
        fs::create_dir_all(src.join("nested")).unwrap();
        fs::write(src.join("a.txt"), b"alpha").unwrap();
        fs::write(src.join("nested/b.txt"), b"beta").unwrap();

        copy_dir_recursive(&src, &dst).expect("copy");
        assert_eq!(fs::read_to_string(dst.join("a.txt")).unwrap(), "alpha");
        assert_eq!(
            fs::read_to_string(dst.join("nested/b.txt")).unwrap(),
            "beta"
        );

        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dst);
    }

    #[test]
    fn format_tool_start_truncates() {
        let long = "x".repeat(200);
        let args = serde_json::json!({"command": long});
        let line = format_tool_start("bash", &args);
        assert!(line.starts_with("→ bash "));
        assert!(line.chars().count() <= TOOL_LINE_MAX);
        assert!(line.ends_with('…'));
    }

    #[test]
    fn format_greppy_tool_line() {
        let args = serde_json::json!({"args": ["who-calls", "foo"]});
        assert_eq!(format_tool_start("greppy", &args), "→ greppy who-calls foo");
    }
}
