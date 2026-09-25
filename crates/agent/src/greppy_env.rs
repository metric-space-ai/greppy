//! Production [`ExecutionEnv`](crate::env::ExecutionEnv): one tool over self-invocation.
//!
//! The model sees exactly one tool — `greppy`. It dispatches by spawning the
//! greppy binary as a subprocess with captured stdio (self-invocation). Command
//! execution goes through the same tool as `bash-smart -- CMD`. In production
//! the binary is `std::env::current_exe()`; tests inject a stub via
//! [`GreppyEnv::with_binary`].
//!
//! Capture policy: stdout is read fully, then stderr is appended after stdout
//! (separated by a newline only when both are non-empty). This is deterministic
//! and avoids interleaving races.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::env::{ExecutionEnv, ToolOutcome};
use crate::protocol::ToolDefinition;
use crate::sandbox::{self, SandboxMode};

/// Default wall-clock budget for `greppy bash-smart` invocations (300 s).
pub const DEFAULT_BASH_TIMEOUT: Duration = Duration::from_secs(300);

/// Default wall-clock budget for non-`bash-smart` greppy invocations (120 s).
pub const DEFAULT_GREPPY_TIMEOUT: Duration = Duration::from_secs(120);

/// Default combined stdout+stderr cap (64 KiB).
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 65_536;

/// Installs the parent-owned attach token onto an authorized `greppy web`
/// child via inherited FD 4. The returned value must be held until after
/// `spawn` so the CLOEXEC pipe end stays open through fork.
type AttachHold = Box<dyn Send>;
type PrepareAttachFd = fn(&mut Command) -> io::Result<AttachHold>;
pub static PREPARE_ATTACH_FD: OnceLock<PrepareAttachFd> = OnceLock::new();

/// Credential / secret env vars stripped from every tool subprocess.
///
/// This is a **blocklist**, not a sandbox: PATH, HOME, and everything else
/// still pass through. Tool children must not inherit API keys or tokens
/// that the agent process itself may hold.
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

/// Production execution environment: single `greppy` tool over self-invocation.
#[derive(Debug, Clone)]
pub struct GreppyEnv {
    greppy_bin: PathBuf,
    root: PathBuf,
    bash_timeout: Duration,
    greppy_timeout: Duration,
    max_output_bytes: usize,
    sandbox: SandboxMode,
}

impl GreppyEnv {
    /// Build an env that re-invokes the running binary against `root`.
    pub fn new(root: PathBuf) -> io::Result<Self> {
        Self::with_binary(std::env::current_exe()?, root)
    }

    /// Build an env that invokes an injectable binary (tests use a stub script).
    pub fn with_binary(greppy_bin: PathBuf, root: PathBuf) -> io::Result<Self> {
        Ok(Self {
            greppy_bin,
            root,
            bash_timeout: DEFAULT_BASH_TIMEOUT,
            greppy_timeout: DEFAULT_GREPPY_TIMEOUT,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            sandbox: SandboxMode::Off,
        })
    }

    /// Override the wall-clock timeout used for `bash-smart` greppy invocations.
    pub fn with_bash_timeout(mut self, timeout: Duration) -> Self {
        self.bash_timeout = timeout;
        self
    }

    /// Override the wall-clock timeout used for non-`bash-smart` greppy invocations.
    pub fn with_greppy_timeout(mut self, timeout: Duration) -> Self {
        self.greppy_timeout = timeout;
        self
    }

    /// Override the combined-output byte cap.
    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// Set the write-confinement sandbox applied to every tool subprocess.
    ///
    /// Defaults to [`SandboxMode::Off`] so non-`-p` callers are unchanged.
    /// `greppy -p` enables [`SandboxMode::Enforce`] with the run's writable
    /// roots (worktree, temp, greppy data root, `~/.cargo`, platform cache).
    pub fn with_sandbox(mut self, mode: SandboxMode) -> Self {
        self.sandbox = mode;
        self
    }

    /// Run the startup self-check through this env's production tool path.
    ///
    /// See [`run_startup_self_check`]. Kept as a method so call sites read
    /// naturally next to the env they just built.
    pub fn startup_self_check(&mut self) -> Result<SelfCheckOk, SelfCheckError> {
        run_startup_self_check(self)
    }

    /// Repository root this env operates on.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to the greppy binary (or test stub) that tools re-invoke.
    pub fn greppy_bin(&self) -> &Path {
        &self.greppy_bin
    }

    /// Current sandbox mode.
    pub fn sandbox(&self) -> &SandboxMode {
        &self.sandbox
    }

    fn greppy_tool_def() -> ToolDefinition {
        ToolDefinition {
            name: "greppy".to_string(),
            description: "Answers every question about this repository (search, navigate, read, edit) and runs commands via bash-smart. Pass argv as an array, e.g. [\"who-calls\", \"my_func\"] or [\"bash-smart\", \"--\", \"cargo\", \"test\"]. bash-smart returns compacted output: verdict line, then errors and warnings.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Argv for the greppy command (subcommand first)."
                    }
                },
                "required": ["args"]
            }),
        }
    }

    fn call_greppy(&self, arguments: &Value) -> ToolOutcome {
        let args = match parse_string_array(arguments, "args") {
            Ok(args) => args,
            Err(msg) => return ToolOutcome::err(msg),
        };

        if let Some(msg) = greppy_guard(&args) {
            return ToolOutcome::err(msg);
        }

        let timeout = if args.first().map(String::as_str) == Some("bash-smart") {
            self.bash_timeout
        } else {
            self.greppy_timeout
        };

        let mut cmd = Command::new(&self.greppy_bin);
        if let Err(e) = sandbox::apply(&mut cmd, &self.greppy_bin, &args, &self.sandbox) {
            return ToolOutcome::err(format!("sandbox: {e}"));
        }
        cmd.current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        prepare_tool_env(&mut cmd);
        // Register attach pre_exec AFTER sandbox::apply so callbacks run
        // sandbox first, then child-local dup2 onto FD 4.
        let attach_hold = if args.first().map(String::as_str) == Some("web") {
            match PREPARE_ATTACH_FD.get() {
                Some(prepare) => match prepare(&mut cmd) {
                    Ok(hold) => Some(hold),
                    Err(error) => {
                        return ToolOutcome::err(format!("attach token fd: {error}"));
                    }
                },
                None => {
                    return ToolOutcome::err(
                        "greppy web requires a parent-owned attach token on inherited fd 4",
                    );
                }
            }
        } else {
            None
        };

        match run_capture_held(&mut cmd, Some(timeout), attach_hold) {
            Ok(captured) => {
                let mut outcome = finalize_outcome(captured, self.max_output_bytes);
                if args.first().map(String::as_str) == Some("web")
                    && args.get(1).map(String::as_str) == Some("screenshot")
                {
                    if let Some(data) = extract_png_base64(&outcome.content) {
                        outcome.image_png_base64 = Some(data);
                        outcome.content = "screenshot attached as image for the model".to_owned();
                    }
                }
                outcome
            }
            Err(msg) => ToolOutcome::err(msg),
        }
    }
}

/// Prepare a tool subprocess environment: strip credential env vars and mark
/// the process tree as an agent run so `greppy -p` refuses to nest (a nested
/// `greppy -p` could otherwise launch a second agent).
///
/// `env_remove` wins over any prior command-scoped `.env(...)` entries and over
/// process inheritance, so secrets injected for tests (or accidentally set on
/// the `Command`) cannot leak into the child.
fn prepare_tool_env(cmd: &mut Command) {
    scrub_credential_env(cmd);
    cmd.env(crate::AGENT_RUN_ENV, "1");
}

/// Strip credential / secret env vars from a tool `Command`.
fn scrub_credential_env(cmd: &mut Command) {
    for key in CREDENTIAL_ENV_BLOCKLIST {
        cmd.env_remove(key);
    }
}

impl ExecutionEnv for GreppyEnv {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![Self::greppy_tool_def()]
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        match name {
            "greppy" => self.call_greppy(arguments),
            other => ToolOutcome::unknown_tool(other),
        }
    }
}

/// Guards checked before spawning the greppy tool. Returns an error message on
/// violation (caller must not spawn).
fn greppy_guard(args: &[String]) -> Option<String> {
    if args.is_empty() {
        return Some("greppy tool requires a non-empty args array".to_string());
    }
    let first = args[0].as_str();
    if first == "-p" || first == "agent" {
        return Some(format!(
            "nested agent runs are not supported (first arg {first:?}) — you are \
             the agent; carry out the task directly with the other greppy commands"
        ));
    }
    // Models sometimes nest the binary name: `["greppy", "rg", …]`. The tool
    // already *is* greppy — the first argv element must be the subcommand.
    if first == "greppy" {
        return Some(
            "drop the leading \"greppy\" — pass the subcommand directly, \
             e.g. [\"rg\", \"-n\", \"pattern\"]"
                .to_string(),
        );
    }
    // `bash-smart` is the sanctioned command-execution path under the single
    // greppy tool surface — deliberately allowed here.
    // Reject both `--root` and `--root=…` so the model cannot re-root the
    // execution environment via either spelling.
    if args
        .iter()
        .any(|a| a == "--root" || a.starts_with("--root="))
    {
        return Some(
            "greppy tool forbids --root (the execution environment owns the repository root)"
                .to_string(),
        );
    }
    None
}

fn parse_string_array(arguments: &Value, field: &str) -> Result<Vec<String>, String> {
    let obj = arguments
        .as_object()
        .ok_or_else(|| format!("tool arguments must be a JSON object (missing {field})"))?;
    let value = obj
        .get(field)
        .ok_or_else(|| format!("missing required field: {field}"))?;
    let arr = value
        .as_array()
        .ok_or_else(|| format!("field {field} must be an array of strings"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let s = item
            .as_str()
            .ok_or_else(|| format!("field {field}[{i}] must be a string"))?;
        out.push(s.to_string());
    }
    Ok(out)
}

struct Captured {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    success: bool,
    timed_out: bool,
    timeout: Option<Duration>,
}

/// Spawn `cmd`, optionally enforcing a wall-clock timeout via try_wait + kill.
/// Stdout and stderr are drained on background threads to avoid pipe-buffer
/// deadlocks.
///
/// On Unix the child is placed in its own process group so a timeout kill
/// reaps grandchildren too (e.g. `sh` + `sleep`); otherwise grandchildren keep
/// the pipes open and the drain threads block until they exit naturally.
#[cfg(all(test, unix))]
fn run_capture(cmd: &mut Command, timeout: Option<Duration>) -> Result<Captured, String> {
    run_capture_held(cmd, timeout, None)
}

fn run_capture_held(
    cmd: &mut Command,
    timeout: Option<Duration>,
    attach_hold: Option<AttachHold>,
) -> Result<Captured, String> {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // New session/process group so killpg can reap the whole tree.
        cmd.process_group(0);
    }

    let start = Instant::now();
    let mut executable_busy_attempts = 0;
    let mut child = loop {
        match cmd.spawn() {
            Ok(child) => break child,
            Err(error)
                if executable_busy(&error)
                    && executable_busy_attempts < EXECUTABLE_BUSY_RETRIES =>
            {
                executable_busy_attempts += 1;
                thread::sleep(EXECUTABLE_BUSY_RETRY_DELAY);
            }
            Err(error) => return Err(format!("failed to spawn greppy binary: {error}")),
        }
    };
    drop(attach_hold);

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| "internal error: missing stdout pipe".to_string())?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| "internal error: missing stderr pipe".to_string())?;

    let stdout_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        let mut pipe = stdout_pipe;
        let _ = pipe.read_to_end(&mut buf);
        buf
    });
    let stderr_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        let mut pipe = stderr_pipe;
        let _ = pipe.read_to_end(&mut buf);
        buf
    });

    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if let Some(limit) = timeout {
                    if start.elapsed() >= limit {
                        timed_out = true;
                        kill_child_tree(&mut child);
                        break child
                            .wait()
                            .map_err(|e| format!("wait after kill failed: {e}"))?;
                    }
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(format!("wait for child failed: {e}")),
        }
    };

    let stdout = stdout_handle
        .join()
        .map_err(|_| "stdout drain thread panicked".to_string())?;
    let stderr = stderr_handle
        .join()
        .map_err(|_| "stderr drain thread panicked".to_string())?;

    Ok(Captured {
        stdout,
        stderr,
        success: status.success(),
        timed_out,
        timeout,
    })
}

const EXECUTABLE_BUSY_RETRIES: usize = 10;
const EXECUTABLE_BUSY_RETRY_DELAY: Duration = Duration::from_millis(10);

fn executable_busy(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::ExecutableFileBusy
}

/// Kill the spawned tool process and, on Unix, its process group so shell
/// grandchildren cannot keep stdio pipes open after a timeout.
fn kill_child_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // process_group(0) made the child the leader of a new group with pgid == pid.
        let pid = child.id() as i32;
        if pid > 0 {
            // SAFETY: killpg with the child's own group; best-effort on timeout.
            unsafe {
                libc_kill_pg(pid, 9 /* SIGKILL */);
            }
        }
    }
    let _ = child.kill();
}

/// Thin libc killpg wrapper so we do not take a libc crate dependency.
#[cfg(unix)]
unsafe fn libc_kill_pg(pgid: i32, sig: i32) -> i32 {
    // Declare only what we need; matches POSIX killpg(2).
    unsafe extern "C" {
        fn killpg(pgrp: i32, sig: i32) -> i32;
    }
    unsafe { killpg(pgid, sig) }
}

fn extract_png_base64(stdout: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(stdout).ok()?;
    let result = value.get("result").unwrap_or(&value);
    result
        .get("png_base64")
        .and_then(|value| value.as_str())
        .filter(|data| !data.is_empty())
        .map(str::to_owned)
}

fn finalize_outcome(captured: Captured, max_output_bytes: usize) -> ToolOutcome {
    if captured.timed_out {
        let secs = captured.timeout.map(|d| d.as_secs()).unwrap_or_default();
        // Still surface any partial output so the model has context, then the
        // explicit timeout verdict.
        let mut body = merge_stdio(&captured.stdout, &captured.stderr);
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(&format!("timed out after {secs}s"));
        let body = truncate_output(body, max_output_bytes);
        return ToolOutcome::err(body);
    }

    let body = merge_stdio(&captured.stdout, &captured.stderr);
    // Narrow detection of the retryable semantic-index build status greppy
    // prints on stdout when embeddings are incomplete (exit 1). That status
    // must never reach the model as an error — the agent should retry soon
    // and use name/text search meanwhile.
    //
    // Emitting site: `embedding_progress_text` in `crates/cli/src/inference.rs`
    // (also called from search.rs / context.rs / indexing.rs). Stable prefix
    // is the literal below (em dash U+2014). Match that prefix only — do not
    // re-derive from JSON fields here (text form is what the tool captures).
    if is_retryable_semantic_index_building(&body) {
        let mut msg = body;
        if !msg.is_empty() && !msg.ends_with('\n') {
            msg.push('\n');
        }
        msg.push_str(
            "semantic index still building — not an error. Retry this same command \
             shortly; meanwhile use search-symbol or search-pattern for name/text matches.",
        );
        let msg = truncate_output(msg, max_output_bytes);
        return ToolOutcome::ok(msg);
    }

    // Sandbox write refusals (macOS Seatbelt / Linux Landlock): surface the
    // policy in plain language, then keep the original output so the model
    // still sees the raw errno/path. Narrow — only known refusal signals.
    if !captured.success && is_sandbox_write_refusal(&body) {
        let mut msg = String::from(
            "this run is write-confined to the repository worktree; installing \
             software or writing outside it is not possible. Work with what the \
             repository provides, or finish and report the missing tool.\n",
        );
        msg.push_str(&body);
        let msg = truncate_output(msg, max_output_bytes);
        return ToolOutcome::err(msg);
    }

    let body = truncate_output(body, max_output_bytes);
    if captured.success {
        ToolOutcome::ok(body)
    } else {
        ToolOutcome::err(body)
    }
}

/// Stable prefix of the retryable semantic-index status line.
///
/// Must match the format string in `crates/cli/src/inference.rs`
/// (`embedding_progress_text`): `"semantic index building — {completed}/…"`.
const SEMANTIC_INDEX_BUILDING_PREFIX: &str = "semantic index building —";

/// True when tool output is the retryable "semantic index building" status.
///
/// Matches the CLI's text form by its stable prefix (em dash U+2014). Exit
/// code is ignored — the status prints with exit 1, which would otherwise
/// surface as a tool error. Emitting site: `embedding_progress_text` in
/// `crates/cli/src/inference.rs`.
fn is_retryable_semantic_index_building(body: &str) -> bool {
    body.contains(SEMANTIC_INDEX_BUILDING_PREFIX)
}

/// True when tool output looks like a write-sandbox refusal.
///
/// Narrow signals only:
/// - macOS Seatbelt: `Operation not permitted`
/// - Linux Landlock / generic POSIX: `Permission denied` with `EACCES`/`EPERM`
///   text, or bare `EACCES`/`EPERM` errno tokens.
///
/// Unrelated failures (compile errors, missing files, non-zero greppy exits)
/// must not match.
fn is_sandbox_write_refusal(body: &str) -> bool {
    // macOS Seatbelt (sandbox-exec) denial text.
    if body.contains("Operation not permitted") {
        return true;
    }
    // Linux Landlock / openat failures often surface as "Permission denied"
    // with an errno token nearby. Require both to avoid swallowing ordinary
    // "permission denied" application messages that lack errno context.
    let lower = body.to_ascii_lowercase();
    if lower.contains("permission denied")
        && (body.contains("EACCES")
            || body.contains("EPERM")
            || lower.contains("eacces")
            || lower.contains("eperm"))
    {
        return true;
    }
    // Some shells print only the errno name (e.g. `touch: ...: EACCES`).
    if body.contains("EACCES") || body.contains("EPERM") {
        return true;
    }
    false
}

/// Deterministic merge: stdout first, then stderr appended (with a separating
/// newline when both are non-empty and stdout does not already end with one).
fn merge_stdio(stdout: &[u8], stderr: &[u8]) -> String {
    let out = String::from_utf8_lossy(stdout);
    let err = String::from_utf8_lossy(stderr);
    if err.is_empty() {
        return out.into_owned();
    }
    if out.is_empty() {
        return err.into_owned();
    }
    let mut body = out.into_owned();
    if !body.ends_with('\n') {
        body.push('\n');
    }
    body.push_str(&err);
    body
}

fn truncate_output(body: String, max_output_bytes: usize) -> String {
    if max_output_bytes == 0 {
        return truncation_marker(0);
    }
    if body.len() <= max_output_bytes {
        return body;
    }
    // Cut on a char boundary at or before the cap.
    let mut end = max_output_bytes;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = body[..end].to_string();
    out.push_str(&truncation_marker(max_output_bytes));
    out
}

fn truncation_marker(max_output_bytes: usize) -> String {
    if max_output_bytes > 0 && max_output_bytes.is_multiple_of(1024) {
        format!("\n[output truncated at {} KiB]", max_output_bytes / 1024)
    } else {
        format!("\n[output truncated at {max_output_bytes} bytes]")
    }
}

/// Cap for self-check diagnostic tool output (chars, not bytes).
const SELFCHECK_OUTPUT_CHARS: usize = 500;

/// Successful startup self-check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfCheckOk {
    /// True when `where-am-i` succeeded but its census line could not be
    /// parsed for a file count. Treated as a pass (never fail on formatting
    /// drift); callers should mention it on the success diagnostic line.
    pub unrecognized_census_shape: bool,
}

/// Failed startup self-check probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfCheckError {
    /// Human probe name (`where-am-i` or `bash-smart write probe`).
    pub probe: &'static str,
    /// Verbatim tool output (already truncated for display).
    pub output: String,
    /// Most likely operator-actionable cause.
    pub likely_cause: String,
}

impl SelfCheckError {
    /// One multi-line diagnostic suitable for stderr before aborting.
    pub fn diagnostic(&self) -> String {
        format!(
            "greppy -p: self-check failed on `{}`\n\
             output:\n{}\n\
             likely cause: {}",
            self.probe, self.output, self.likely_cause
        )
    }
}

/// Run the agent startup self-check through the production tool path.
///
/// Two probes, both via [`GreppyEnv::call_tool`] with the env's current
/// [`SandboxMode`] (never a raw `Command`):
/// 1. index-backed navigation: `greppy where-am-i` — must succeed and must
///    not report an empty repository (`N files` == 0). Unrecognized census
///    shape is a pass (formatting drift must not abort).
/// 2. write probe inside the worktree:
///    `bash-smart -- sh -c 'printf ok > .greppy-selfcheck && rm -f .greppy-selfcheck'`.
///
/// On any failure the caller must abort the run (exit 3) and never start the
/// model loop.
pub fn run_startup_self_check(env: &mut GreppyEnv) -> Result<SelfCheckOk, SelfCheckError> {
    // (a) index-backed navigation.
    let where_out = env.call_tool("greppy", &json!({"args": ["where-am-i"]}));
    if where_out.is_error {
        return Err(SelfCheckError {
            probe: "where-am-i",
            output: truncate_chars_for_diag(&where_out.content, SELFCHECK_OUTPUT_CHARS),
            likely_cause: likely_cause_tool_error("where-am-i", &where_out.content),
        });
    }
    let mut unrecognized_census_shape = false;
    match parse_where_am_i_file_count(&where_out.content) {
        Some(0) => {
            return Err(SelfCheckError {
                probe: "where-am-i",
                output: truncate_chars_for_diag(&where_out.content, SELFCHECK_OUTPUT_CHARS),
                likely_cause: "the worktree index is empty (0 files) while the tool exited \
                     successfully — prewarm did not produce a usable index (invalid seed, \
                     wrong store, or sandbox blocked greppy data root)"
                    .to_string(),
            });
        }
        Some(_) => {}
        None => {
            unrecognized_census_shape = true;
        }
    }

    // (b) write probe inside the worktree.
    let write_out = env.call_tool(
        "greppy",
        &json!({
            "args": [
                "bash-smart",
                "--",
                "sh",
                "-c",
                "printf ok > .greppy-selfcheck && rm -f .greppy-selfcheck"
            ]
        }),
    );
    if write_out.is_error {
        return Err(SelfCheckError {
            probe: "bash-smart write probe",
            output: truncate_chars_for_diag(&write_out.content, SELFCHECK_OUTPUT_CHARS),
            likely_cause: likely_cause_tool_error("bash-smart write probe", &write_out.content),
        });
    }

    Ok(SelfCheckOk {
        unrecognized_census_shape,
    })
}

/// Parse the hub census file count from `where-am-i` text output.
///
/// Looks for a line mentioning both `files` and `definitions` and extracts
/// the integer immediately before ` files` (commas allowed). Returns
/// `None` when no such shape is recognized — callers treat that as a pass.
pub fn parse_where_am_i_file_count(output: &str) -> Option<u64> {
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Census root lines always mention both words today; require both so
        // inventory lines like "2 files · 30 defs" cannot false-match.
        let lower = line.to_ascii_lowercase();
        if !lower.contains("files") || !lower.contains("definitions") {
            continue;
        }
        if let Some(count) = extract_count_before_files(line) {
            return Some(count);
        }
        // Line has the words but no parseable count — formatting drift.
        return None;
    }
    None
}

/// Extract the integer immediately before the first ` files` token.
fn extract_count_before_files(line: &str) -> Option<u64> {
    // Case-insensitive locate of " files".
    let lower = line.to_ascii_lowercase();
    let idx = lower.find(" files")?;
    let before = &line[..idx];
    // Take the trailing run of digits and thousand-separator commas.
    let token: String = before
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == ',')
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    if token.is_empty() {
        return None;
    }
    let digits: String = token.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn likely_cause_tool_error(probe: &str, output: &str) -> String {
    if looks_like_permission_failure(output) {
        return "a write the agent needs is outside the sandbox roots — re-run with \
                --no-sandbox to confirm, and report the path"
            .to_string();
    }
    if probe == "where-am-i" {
        "index-backed navigation failed through the sandboxed tool path — check greppy \
         data root permissions and that prewarm succeeded"
            .to_string()
    } else {
        "worktree write probe failed through the sandboxed tool path — re-run with \
         --no-sandbox to confirm, and report the path"
            .to_string()
    }
}

fn looks_like_permission_failure(output: &str) -> bool {
    if output.contains("Operation not permitted")
        || output.contains("EACCES")
        || output.contains("EPERM")
    {
        return true;
    }
    let lower = output.to_ascii_lowercase();
    lower.contains("permission denied") || lower.contains("write-confined")
}

fn truncate_chars_for_diag(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static STUB_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Write an executable `#!/bin/sh` stub into temp_dir; returns its path.
    fn write_stub(body: &str) -> PathBuf {
        let seq = STUB_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "greppy-env-stub-{}-{}-{}",
            std::process::id(),
            seq,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let temporary = path.with_extension("publishing");
        let script = format!("#!/bin/sh\n{body}\n");
        let mut file = fs::File::create(&temporary).expect("create temporary stub");
        std::io::Write::write_all(&mut file, script.as_bytes()).expect("write temporary stub");
        file.sync_all().expect("sync temporary stub");
        drop(file);
        let mut perms = fs::metadata(&temporary).expect("meta").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&temporary, perms).expect("chmod");
        fs::rename(&temporary, &path).expect("publish stub atomically");
        path
    }

    fn temp_root() -> PathBuf {
        let seq = STUB_SEQ.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("greppy-env-root-{}-{}", std::process::id(), seq));
        fs::create_dir_all(&path).expect("mkdir root");
        path
    }

    fn env_with_stub(stub_body: &str) -> (GreppyEnv, PathBuf, PathBuf) {
        let bin = write_stub(stub_body);
        let root = temp_root();
        let env = GreppyEnv::with_binary(bin.clone(), root.clone()).expect("env");
        (env, bin, root)
    }

    #[test]
    fn tool_definitions_exactly_one_named_greppy() {
        let (env, _, _) = env_with_stub("exit 0");
        let defs = env.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "greppy");

        let greppy_schema = &defs[0].input_schema;
        assert_eq!(greppy_schema["type"], "object");
        assert!(greppy_schema["properties"].get("args").is_some());
        assert_eq!(greppy_schema["required"], json!(["args"]));
        assert!(
            defs[0].description.contains("bash-smart"),
            "description must teach bash-smart: {}",
            defs[0].description
        );
    }

    #[test]
    fn greppy_argv_passthrough_and_capture() {
        // Stub prints argv one-per-line (skip $0).
        let (mut env, _, _) = env_with_stub(
            r#"
for a in "$@"; do
  printf '%s\n' "$a"
done
"#,
        );
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["who-calls", "my_func", "--code"]}),
        );
        assert!(!out.is_error, "content={}", out.content);
        assert_eq!(out.content, "who-calls\nmy_func\n--code\n");
    }

    #[test]
    fn exit_code_nonzero_maps_to_is_error_preserving_output() {
        let (mut env, _, _) = env_with_stub(
            r#"
printf 'compiler error: boom\n'
exit 2
"#,
        );
        let out = env.call_tool("greppy", &json!({"args": ["build"]}));
        assert!(out.is_error);
        assert!(
            out.content.contains("compiler error: boom"),
            "content={}",
            out.content
        );
    }

    #[test]
    fn guard_empty_argv_does_not_invoke_stub() {
        let sentinel =
            std::env::temp_dir().join(format!("greppy-env-sentinel-empty-{}", std::process::id()));
        let _ = fs::remove_file(&sentinel);
        let stub = format!("touch '{}'\nexit 0\n", sentinel.display());
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool("greppy", &json!({"args": []}));
        assert!(out.is_error);
        assert!(out.content.contains("non-empty"), "content={}", out.content);
        assert!(
            !sentinel.exists(),
            "stub must not have been invoked for empty argv"
        );
    }

    #[test]
    fn guard_first_arg_dash_p_does_not_invoke_stub() {
        let sentinel =
            std::env::temp_dir().join(format!("greppy-env-sentinel-dashp-{}", std::process::id()));
        let _ = fs::remove_file(&sentinel);
        let stub = format!("touch '{}'\nexit 0\n", sentinel.display());
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool("greppy", &json!({"args": ["-p", "hi"]}));
        assert!(out.is_error);
        assert!(
            out.content.contains("recursive") || out.content.contains("-p"),
            "content={}",
            out.content
        );
        assert!(!sentinel.exists());
    }

    #[test]
    fn guard_first_arg_agent_does_not_invoke_stub() {
        let sentinel =
            std::env::temp_dir().join(format!("greppy-env-sentinel-agent-{}", std::process::id()));
        let _ = fs::remove_file(&sentinel);
        let stub = format!("touch '{}'\nexit 0\n", sentinel.display());
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool("greppy", &json!({"args": ["agent", "run"]}));
        assert!(out.is_error);
        assert!(out.content.contains("agent"), "content={}", out.content);
        assert!(!sentinel.exists());
    }

    #[test]
    fn guard_leading_greppy_rejected() {
        let sentinel = std::env::temp_dir().join(format!(
            "greppy-env-sentinel-nested-greppy-{}",
            std::process::id()
        ));
        let _ = fs::remove_file(&sentinel);
        let stub = format!("touch '{}'\nexit 0\n", sentinel.display());
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool("greppy", &json!({"args": ["greppy", "rg", "--", "429"]}));
        assert!(out.is_error);
        assert!(
            out.content.contains("drop the leading \"greppy\""),
            "content={}",
            out.content
        );
        assert!(
            out.content.contains("[\"rg\""),
            "must show the corrected shape; content={}",
            out.content
        );
        assert!(
            !sentinel.exists(),
            "stub must not have been invoked for nested greppy argv"
        );
    }

    #[test]
    fn guard_root_anywhere_does_not_invoke_stub() {
        let sentinel =
            std::env::temp_dir().join(format!("greppy-env-sentinel-root-{}", std::process::id()));
        let _ = fs::remove_file(&sentinel);
        let stub = format!("touch '{}'\nexit 0\n", sentinel.display());
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["who-calls", "foo", "--root", "/tmp"]}),
        );
        assert!(out.is_error);
        assert!(out.content.contains("--root"), "content={}", out.content);
        assert!(!sentinel.exists());
    }

    #[test]
    fn guard_root_equals_form_does_not_invoke_stub() {
        let sentinel = std::env::temp_dir().join(format!(
            "greppy-env-sentinel-root-eq-{}",
            std::process::id()
        ));
        let _ = fs::remove_file(&sentinel);
        let stub = format!("touch '{}'\nexit 0\n", sentinel.display());
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["who-calls", "foo", "--root=/tmp"]}),
        );
        assert!(out.is_error);
        assert!(out.content.contains("--root"), "content={}", out.content);
        assert!(!sentinel.exists());
    }

    #[test]
    fn bash_smart_argv_is_allowed_and_spawns() {
        let sentinel = std::env::temp_dir().join(format!(
            "greppy-env-sentinel-bash-smart-{}",
            std::process::id()
        ));
        let _ = fs::remove_file(&sentinel);
        let stub = format!("touch '{}'\nexit 0\n", sentinel.display());
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["bash-smart", "--", "cargo", "test"]}),
        );
        assert!(!out.is_error, "content={}", out.content);
        assert!(
            sentinel.exists(),
            "bash-smart is the sanctioned path and must spawn the stub"
        );
        let _ = fs::remove_file(&sentinel);
    }

    #[test]
    fn bash_smart_uses_bash_timeout_budget() {
        // sleep 5 under a 1s bash budget → timeout; greppy budget stays long.
        let (env, _, _) = env_with_stub("sleep 5\nexit 0\n");
        let mut env = env
            .with_bash_timeout(Duration::from_secs(1))
            .with_greppy_timeout(Duration::from_secs(30));
        let start = Instant::now();
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["bash-smart", "--", "sleep", "5"]}),
        );
        let elapsed = start.elapsed();
        assert!(out.is_error, "content={}", out.content);
        assert!(
            out.content.contains("timed out after 1s"),
            "content={}",
            out.content
        );
        assert!(
            elapsed < Duration::from_secs(4),
            "elapsed too long: {elapsed:?}"
        );
    }

    #[test]
    fn plain_greppy_uses_greppy_timeout_budget() {
        let (env, _, _) = env_with_stub("sleep 5\nexit 0\n");
        let mut env = env
            .with_greppy_timeout(Duration::from_secs(1))
            .with_bash_timeout(Duration::from_secs(30));
        let start = Instant::now();
        let out = env.call_tool("greppy", &json!({"args": ["hang"]}));
        let elapsed = start.elapsed();
        assert!(out.is_error, "content={}", out.content);
        assert!(
            out.content.contains("timed out after 1s"),
            "content={}",
            out.content
        );
        assert!(
            elapsed < Duration::from_secs(4),
            "elapsed too long: {elapsed:?}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn executable_busy_spawn_retries_until_atomic_publisher_releases_writer() {
        let bin = write_stub("exit 0");
        let writer = fs::OpenOptions::new()
            .write(true)
            .open(&bin)
            .expect("hold executable open for writing");
        let releaser = thread::spawn(move || {
            thread::sleep(Duration::from_millis(35));
            drop(writer);
        });
        let mut command = Command::new(&bin);
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let captured = run_capture(&mut command, Some(Duration::from_secs(1)))
            .expect("ETXTBSY must be retried after the writer closes");

        releaser.join().expect("writer releaser");
        assert!(captured.success);
        assert!(!captured.timed_out);
    }

    #[test]
    fn retryable_semantic_index_building_is_non_error() {
        // Stub prints the exact greppy status line (emitting site:
        // crates/cli/src/inference.rs::embedding_progress_text) and exits 1.
        let exact = "semantic index building — 3/12 spans, ETA ~9s (backend cuda)\n";
        assert!(
            exact.starts_with(SEMANTIC_INDEX_BUILDING_PREFIX),
            "fixture must use the stable CLI prefix"
        );
        let stub = format!("printf '%s' '{exact}'\nexit 1\n");
        let (mut env, _, _) = env_with_stub(&stub);
        let out = env.call_tool("greppy", &json!({"args": ["search", "retry flow"]}));
        assert!(
            !out.is_error,
            "retryable building status must be non-error; content={}",
            out.content
        );
        assert!(
            out.content.contains(SEMANTIC_INDEX_BUILDING_PREFIX),
            "content={}",
            out.content
        );
        assert!(
            out.content.contains("search-symbol") || out.content.contains("search-pattern"),
            "must advise interim name/text search; content={}",
            out.content
        );
        assert!(
            out.content.to_ascii_lowercase().contains("retry"),
            "must tell the model to retry; content={}",
            out.content
        );
    }

    #[test]
    fn sandbox_refusal_macos_seatbelt_is_clarified() {
        let (mut env, _, _) = env_with_stub(
            r#"
printf 'touch: /Users/x/.local/bin/pytest: Operation not permitted\n' >&2
exit 1
"#,
        );
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["bash-smart", "--", "touch", "/outside"]}),
        );
        assert!(out.is_error, "content={}", out.content);
        assert!(
            out.content
                .contains("this run is write-confined to the repository worktree"),
            "must state the write-confinement rule; content={}",
            out.content
        );
        assert!(
            out.content.contains("Operation not permitted"),
            "must keep the original output; content={}",
            out.content
        );
        // Clarifying sentence comes first.
        let rule_pos = out
            .content
            .find("this run is write-confined")
            .expect("rule");
        let raw_pos = out.content.find("Operation not permitted").expect("raw");
        assert!(
            rule_pos < raw_pos,
            "rule before raw; content={}",
            out.content
        );
    }

    #[test]
    fn sandbox_refusal_linux_eacces_is_clarified() {
        let (mut env, _, _) = env_with_stub(
            r#"
printf 'pip: install failed: [Errno 13] EACCES: /usr/local/lib\n' >&2
exit 1
"#,
        );
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["bash-smart", "--", "pip", "install", "x"]}),
        );
        assert!(out.is_error, "content={}", out.content);
        assert!(
            out.content
                .contains("this run is write-confined to the repository worktree"),
            "content={}",
            out.content
        );
        assert!(
            out.content.contains("EACCES"),
            "must keep original; content={}",
            out.content
        );
    }

    #[test]
    fn ordinary_nonzero_exit_is_not_sandbox_clarified() {
        let (mut env, _, _) = env_with_stub(
            r#"
printf 'error: no such subcommand\n' >&2
exit 2
"#,
        );
        let out = env.call_tool("greppy", &json!({"args": ["nope"]}));
        assert!(out.is_error);
        assert!(
            !out.content.contains("write-confined"),
            "unrelated failures must not get the sandbox rule; content={}",
            out.content
        );
        assert!(
            out.content.contains("no such subcommand"),
            "content={}",
            out.content
        );
    }

    #[test]
    fn agent_run_marker_present_in_tool_subprocesses() {
        // Command-scoped env only — no global set_var (parallel-test safe).
        let (mut env, _, _) =
            env_with_stub(r#"printf 'GREPPY_AGENT_RUN=%s\n' "${GREPPY_AGENT_RUN-}""#);
        let greppy = env.call_tool("greppy", &json!({"args": ["x"]}));
        assert!(!greppy.is_error, "{}", greppy.content);
        assert!(
            greppy.content.contains("GREPPY_AGENT_RUN=1"),
            "{}",
            greppy.content
        );
        // bash-smart path is the same spawn helper — marker must still land.
        let smart = env.call_tool("greppy", &json!({"args": ["bash-smart", "--", "true"]}));
        assert!(
            smart.content.contains("GREPPY_AGENT_RUN=1"),
            "{}",
            smart.content
        );
    }

    #[test]
    fn credential_env_blocklist_stripped_from_tool_subprocesses() {
        // Parallel-safe: no global set_var/remove_var. Inject secrets only as
        // command-scoped `.env(...)` entries, then apply prepare_tool_env and
        // assert env_remove wins (vars absent in the child) while PATH/HOME
        // still inherit.
        let bin = write_stub(
            r#"
printf 'GREPPY_API_KEY=%s\n' "${GREPPY_API_KEY-}"
printf 'ANTHROPIC_API_KEY=%s\n' "${ANTHROPIC_API_KEY-}"
printf 'OPENAI_API_KEY=%s\n' "${OPENAI_API_KEY-}"
printf 'GITHUB_TOKEN=%s\n' "${GITHUB_TOKEN-}"
printf 'PATH_SET=%s\n' "${PATH:+yes}"
printf 'HOME_SET=%s\n' "${HOME:+yes}"
"#,
        );
        let root = temp_root();

        // Build a greppy-tool-shaped Command with secrets set command-locally
        // *before* prepare_tool_env runs — env_remove must win over .env().
        let mut cmd = Command::new(&bin);
        cmd.args(["who-calls", "x"])
            .current_dir(&root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("GREPPY_API_KEY", "sekrit")
            .env("ANTHROPIC_API_KEY", "sekrit")
            .env("OPENAI_API_KEY", "sekrit")
            .env("GITHUB_TOKEN", "sekrit");
        prepare_tool_env(&mut cmd);

        // Command-scoped view: blocklisted keys must be Clear (env_remove),
        // agent-run marker must be set.
        let envs: Vec<(String, Option<String>)> = cmd
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|s| s.to_string_lossy().into_owned()),
                )
            })
            .collect();
        for key in [
            "GREPPY_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GITHUB_TOKEN",
        ] {
            let entry = envs.iter().find(|(k, _)| k == key);
            assert!(
                matches!(entry, Some((_, None))),
                "expected env_remove for {key}, got {entry:?}; all={envs:?}"
            );
        }
        let marker = envs.iter().find(|(k, _)| k == crate::AGENT_RUN_ENV);
        assert_eq!(
            marker.map(|(_, v)| v.as_deref()),
            Some(Some("1")),
            "agent run marker missing; envs={envs:?}"
        );

        // Generous wall-clock margin: under concurrent host load the 5 s budget
        // flaked once in review. Happy path is still near-instant.
        let captured = run_capture(&mut cmd, Some(Duration::from_secs(30))).expect("spawn");
        let body = merge_stdio(&captured.stdout, &captured.stderr);
        assert!(captured.success, "body={body}");
        assert!(body.contains("GREPPY_API_KEY=\n"), "body={body}");
        assert!(body.contains("ANTHROPIC_API_KEY=\n"), "body={body}");
        assert!(body.contains("OPENAI_API_KEY=\n"), "body={body}");
        assert!(body.contains("GITHUB_TOKEN=\n"), "body={body}");
        assert!(
            body.contains("PATH_SET=yes"),
            "PATH must survive; body={body}"
        );
        assert!(
            body.contains("HOME_SET=yes"),
            "HOME must survive; body={body}"
        );

        // Same scrubbing on the bash-tool-shaped Command path.
        let mut bash_cmd = Command::new(&bin);
        bash_cmd
            .args(["bash-smart", "--", "bash", "-lc", "true"])
            .current_dir(&root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("GREPPY_API_KEY", "sekrit")
            .env("ANTHROPIC_API_KEY", "sekrit");
        prepare_tool_env(&mut bash_cmd);
        let bash_envs: Vec<(String, Option<String>)> = bash_cmd
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|s| s.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert!(
            matches!(
                bash_envs.iter().find(|(k, _)| k == "GREPPY_API_KEY"),
                Some((_, None))
            ),
            "bash path must env_remove GREPPY_API_KEY; envs={bash_envs:?}"
        );
        let bash_cap =
            run_capture(&mut bash_cmd, Some(Duration::from_secs(30))).expect("bash spawn");
        assert!(bash_cap.success, "bash scrub path failed");
    }

    #[test]
    fn greppy_timeout_kills_and_reports() {
        let (env, _, _) = env_with_stub("sleep 5\nexit 0\n");
        let mut env = env.with_greppy_timeout(Duration::from_secs(1));
        let start = Instant::now();
        let out = env.call_tool("greppy", &json!({"args": ["hang"]}));
        let elapsed = start.elapsed();
        assert!(out.is_error, "content={}", out.content);
        assert!(
            out.content.contains("timed out after 1s"),
            "content={}",
            out.content
        );
        assert!(
            elapsed < Duration::from_secs(4),
            "elapsed too long: {elapsed:?}"
        );
    }

    #[test]
    fn bash_smart_argv_passthrough_with_cwd_root() {
        // Record argv and cwd for a bash-smart greppy invocation.
        let record =
            std::env::temp_dir().join(format!("greppy-env-bash-record-{}", std::process::id()));
        let _ = fs::remove_file(&record);
        let stub = format!(
            r#"
printf '%s\n' "$@" > '{record}'
printf 'cwd=' >> '{record}'
pwd >> '{record}'
exit 0
"#,
            record = record.display()
        );
        let (mut env, _, root) = env_with_stub(&stub);
        let out = env.call_tool(
            "greppy",
            &json!({"args": ["bash-smart", "--", "echo", "hi"]}),
        );
        assert!(!out.is_error, "content={}", out.content);

        let recorded = fs::read_to_string(&record).expect("record");
        let lines: Vec<&str> = recorded.lines().collect();
        // argv: bash-smart -- echo hi
        assert!(lines.len() >= 4, "recorded={recorded:?}");
        assert_eq!(lines[0], "bash-smart");
        assert_eq!(lines[1], "--");
        assert_eq!(lines[2], "echo");
        assert_eq!(lines[3], "hi");
        let cwd_line = lines.iter().find(|l| l.starts_with("cwd=")).expect("cwd");
        let cwd = cwd_line.trim_start_matches("cwd=");
        let got = fs::canonicalize(cwd).unwrap_or_else(|_| PathBuf::from(cwd));
        let want = fs::canonicalize(&root).unwrap_or(root);
        assert_eq!(got, want);
    }

    #[test]
    fn truncation_appends_marker() {
        // Emit 200 bytes of 'x'; cap at 64.
        let (env, _, _) = env_with_stub(
            r#"
# 200 x's
printf '%s' "$(dd if=/dev/zero bs=200 count=1 2>/dev/null | tr '\0' 'x')"
"#,
        );
        let mut env = env.with_max_output_bytes(64);
        let out = env.call_tool("greppy", &json!({"args": ["x"]}));
        assert!(!out.is_error, "content len={}", out.content.len());
        assert!(
            out.content.contains("[output truncated at 64 bytes]"),
            "content={}",
            out.content
        );
        // Marker is appended after the cut; total can exceed cap by marker length.
        let marker = "\n[output truncated at 64 bytes]";
        assert!(out.content.ends_with(marker));
        assert_eq!(out.content.len(), 64 + marker.len());
    }

    #[test]
    fn unknown_tool_is_error() {
        let (mut env, _, _) = env_with_stub("exit 0");
        let out = env.call_tool("nope", &json!({}));
        assert!(out.is_error);
        assert_eq!(out.content, "unknown tool: nope");
    }

    #[test]
    fn stderr_appended_after_stdout() {
        let (mut env, _, _) = env_with_stub(
            r#"
printf 'OUT\n'
printf 'ERR\n' >&2
exit 0
"#,
        );
        let out = env.call_tool("greppy", &json!({"args": ["x"]}));
        assert!(!out.is_error, "content={}", out.content);
        assert_eq!(out.content, "OUT\nERR\n");
    }

    #[test]
    fn with_sandbox_defaults_to_off_and_is_settable() {
        let (env, _, _) = env_with_stub("exit 0");
        assert!(matches!(env.sandbox(), SandboxMode::Off));
        let roots = crate::sandbox::prepare_writable_roots(&[std::env::temp_dir()])
            .expect("prepare temp root");
        let spec =
            crate::sandbox::SandboxSpec::from_prepared_roots(roots).expect("open prepared roots");
        let env = env.with_sandbox(SandboxMode::Enforce(spec));
        assert!(matches!(env.sandbox(), SandboxMode::Enforce(_)));
    }

    #[test]
    fn parse_where_am_i_file_count_reads_census() {
        let out = "/tmp/repo — rust, 12 files, 34 definitions\n\nsrc/ — 3 files · 10 defs\n";
        assert_eq!(parse_where_am_i_file_count(out), Some(12));
        let empty = "/tmp/repo — 0 files, 0 definitions\n";
        assert_eq!(parse_where_am_i_file_count(empty), Some(0));
        let commas = "/tmp/repo — 1,234 files, 5,678 definitions\n";
        assert_eq!(parse_where_am_i_file_count(commas), Some(1234));
        let drift = "orientation complete; nothing looks like a census\n";
        assert_eq!(parse_where_am_i_file_count(drift), None);
        // Inventory-only lines without "definitions" must not match.
        let inv = "src/ — 2 files · 30 defs — hub\n";
        assert_eq!(parse_where_am_i_file_count(inv), None);
    }

    #[test]
    fn self_check_passes_on_healthy_stub() {
        // where-am-i → non-empty census; bash-smart → success (any exit 0).
        let (mut env, _, _) = env_with_stub(
            r#"
if [ "$1" = "where-am-i" ]; then
  printf '/tmp/fixture — rust, 3 files, 7 definitions\n'
  exit 0
fi
if [ "$1" = "bash-smart" ]; then
  printf 'ok — exit 0\n'
  exit 0
fi
printf 'unexpected argv: %s\n' "$*" >&2
exit 2
"#,
        );
        let ok = run_startup_self_check(&mut env).expect("self-check must pass");
        assert!(!ok.unrecognized_census_shape);
    }

    #[test]
    fn self_check_fails_on_tool_error() {
        let (mut env, _, _) = env_with_stub(
            r#"
printf 'Operation not permitted: locks/\n' >&2
exit 1
"#,
        );
        let err = run_startup_self_check(&mut env).expect_err("must fail");
        assert_eq!(err.probe, "where-am-i");
        assert!(
            err.likely_cause.contains("--no-sandbox"),
            "permission failure must point at --no-sandbox; cause={}",
            err.likely_cause
        );
        assert!(
            err.output.contains("Operation not permitted"),
            "output={}",
            err.output
        );
        let diag = err.diagnostic();
        assert!(diag.contains("self-check failed"), "diag={diag}");
        assert!(diag.contains("where-am-i"), "diag={diag}");
    }

    #[test]
    fn self_check_empty_index_is_failure() {
        let (mut env, _, _) = env_with_stub(
            r#"
if [ "$1" = "where-am-i" ]; then
  printf '/tmp/fixture — 0 files, 0 definitions\n'
  exit 0
fi
printf 'ok\n'
exit 0
"#,
        );
        let err = run_startup_self_check(&mut env).expect_err("empty index must fail");
        assert_eq!(err.probe, "where-am-i");
        assert!(
            err.likely_cause.contains("0 files") || err.likely_cause.contains("empty"),
            "cause={}",
            err.likely_cause
        );
        assert!(err.output.contains("0 files"), "output={}", err.output);
    }

    #[test]
    fn self_check_unrecognized_shape_passes() {
        let (mut env, _, _) = env_with_stub(
            r#"
if [ "$1" = "where-am-i" ]; then
  printf 'repo orientation complete (shape drifted)\n'
  exit 0
fi
if [ "$1" = "bash-smart" ]; then
  printf 'ok — exit 0\n'
  exit 0
fi
exit 0
"#,
        );
        let ok = run_startup_self_check(&mut env).expect("unrecognized shape must pass");
        assert!(
            ok.unrecognized_census_shape,
            "must flag unrecognized census shape"
        );
    }

    #[test]
    fn self_check_write_probe_failure_aborts() {
        let (mut env, _, _) = env_with_stub(
            r#"
if [ "$1" = "where-am-i" ]; then
  printf '/tmp/fixture — 2 files, 4 definitions\n'
  exit 0
fi
printf 'touch: .greppy-selfcheck: Permission denied (EACCES)\n' >&2
exit 1
"#,
        );
        let err = run_startup_self_check(&mut env).expect_err("write probe must fail");
        assert_eq!(err.probe, "bash-smart write probe");
        assert!(
            err.likely_cause.contains("--no-sandbox") || err.likely_cause.contains("sandbox"),
            "cause={}",
            err.likely_cause
        );
    }

    #[cfg(target_os = "macos")]
    mod sandbox_integration {
        use super::*;
        use crate::sandbox::SandboxSpec;
        use std::path::Path;

        /// Stub that, for the bash-smart argv shape (`bash-smart -- bash -lc CMD`
        /// or `bash-smart -- CMD…`), execs the real shell so sandbox write checks
        /// exercise genuine syscalls.
        fn real_bash_stub_body() -> &'static str {
            r#"
if [ "$1" = "bash-smart" ]; then
  shift
  if [ "$1" = "--" ]; then shift; fi
  exec "$@"
fi
printf 'ok\n'
exit 0
"#
        }

        fn sandbox_exec_available() -> bool {
            Path::new("/usr/bin/sandbox-exec").exists()
        }

        #[test]
        fn enforce_write_inside_worktree_succeeds() {
            if !sandbox_exec_available() {
                return;
            }
            let root = temp_root();
            let bin = write_stub(real_bash_stub_body());
            let roots = crate::sandbox::prepare_writable_roots(std::slice::from_ref(&root))
                .expect("prepare roots");
            let spec = SandboxSpec::from_prepared_roots(roots).expect("open prepared roots");
            let mut env = GreppyEnv::with_binary(bin, root.clone())
                .unwrap()
                .with_sandbox(SandboxMode::Enforce(spec));
            let marker = root.join("inside-ok.txt");
            let _ = fs::remove_file(&marker);
            let out = env.call_tool(
                "greppy",
                &json!({"args": ["bash-smart", "--", "bash", "-lc", format!("touch '{}'", marker.display())]}),
            );
            assert!(!out.is_error, "content={}", out.content);
            assert!(marker.exists(), "inside write must create the file");
            let _ = fs::remove_file(&marker);
            let _ = fs::remove_dir_all(&root);
        }

        #[test]
        fn enforce_write_outside_home_is_denied() {
            if !sandbox_exec_available() {
                return;
            }
            let root = temp_root();
            let bin = write_stub(real_bash_stub_body());
            let roots = crate::sandbox::prepare_writable_roots(std::slice::from_ref(&root))
                .expect("prepare roots");
            let spec = SandboxSpec::from_prepared_roots(roots).expect("open prepared roots");
            let mut env = GreppyEnv::with_binary(bin, root.clone())
                .unwrap()
                .with_sandbox(SandboxMode::Enforce(spec));

            let home = std::env::var_os("HOME").expect("HOME");
            let escape = PathBuf::from(&home).join(format!(
                ".greppy-sandbox-escape-proof-env-{}-{}",
                std::process::id(),
                STUB_SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_file(&escape);
            let out = env.call_tool(
                "greppy",
                &json!({"args": ["bash-smart", "--", "bash", "-lc", format!("touch '{}'", escape.display())]}),
            );
            assert!(
                out.is_error,
                "outside write must be an error; content={}",
                out.content
            );
            assert!(
                !escape.exists(),
                "escape file must not exist after sandboxed touch"
            );
            let _ = fs::remove_file(&escape);
            let _ = fs::remove_dir_all(&root);
        }

        #[test]
        fn enforce_git_commit_inside_worktree_succeeds() {
            if !sandbox_exec_available() {
                return;
            }
            let root = temp_root();
            // Fixture worktree with an initial commit ready for a second one.
            let init = Command::new("git")
                .args(["init"])
                .current_dir(&root)
                .output()
                .expect("git init");
            assert!(init.status.success(), "git init");
            fs::write(root.join("file.txt"), b"hello\n").unwrap();
            let add = Command::new("git")
                .args(["add", "file.txt"])
                .current_dir(&root)
                .output()
                .expect("git add");
            assert!(add.status.success(), "git add");
            // First commit outside the sandbox (setup).
            let c1 = Command::new("git")
                .args([
                    "-c",
                    "user.email=t@t.com",
                    "-c",
                    "user.name=t",
                    "commit",
                    "-m",
                    "init",
                ])
                .current_dir(&root)
                .output()
                .expect("git commit setup");
            assert!(
                c1.status.success(),
                "setup commit: {}",
                String::from_utf8_lossy(&c1.stderr)
            );

            fs::write(root.join("file.txt"), b"hello\nworld\n").unwrap();
            let bin = write_stub(real_bash_stub_body());
            let roots =
                crate::sandbox::prepare_writable_roots(&[root.clone(), std::env::temp_dir()])
                    .expect("prepare roots");
            let spec = SandboxSpec::from_prepared_roots(roots).expect("open prepared roots");
            let mut env = GreppyEnv::with_binary(bin, root.clone())
                .unwrap()
                .with_sandbox(SandboxMode::Enforce(spec));
            let out = env.call_tool(
                "greppy",
                &json!({
                    "args": [
                        "bash-smart",
                        "--",
                        "bash",
                        "-lc",
                        "git -c user.email=t@t.com -c user.name=t commit -am second"
                    ]
                }),
            );
            assert!(!out.is_error, "git commit under sandbox: {}", out.content);
            let _ = fs::remove_dir_all(&root);
        }

        #[test]
        fn off_mode_outside_write_succeeds() {
            // Proves the test harness can distinguish Enforce from Off: the
            // same outside touch that Enforce blocks must succeed with Off.
            let root = temp_root();
            let bin = write_stub(real_bash_stub_body());
            let mut env = GreppyEnv::with_binary(bin, root.clone())
                .unwrap()
                .with_sandbox(SandboxMode::Off);

            let home = std::env::var_os("HOME").expect("HOME");
            let probe = PathBuf::from(&home).join(format!(
                ".greppy-sandbox-off-probe-{}-{}",
                std::process::id(),
                STUB_SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_file(&probe);
            let out = env.call_tool(
                "greppy",
                &json!({"args": ["bash-smart", "--", "bash", "-lc", format!("touch '{}'", probe.display())]}),
            );
            assert!(!out.is_error, "Off-mode outside write: {}", out.content);
            assert!(probe.exists(), "Off mode must allow the outside touch");
            let _ = fs::remove_file(&probe);
            let _ = fs::remove_dir_all(&root);
        }

        /// S4 permanent regression (reviewer-verified escape): with only the
        /// worktree + canonical `temp_dir()` roots allowed, a write under
        /// `$TMPDIR/../C/…` must be DENIED. The removed blanket
        /// `/private/var/folders` rule is what previously permitted this.
        #[test]
        fn enforce_tmpdir_sibling_c_escape_denied() {
            if !sandbox_exec_available() {
                return;
            }
            let root = temp_root();
            let tmp = std::env::temp_dir();
            let bin = write_stub(real_bash_stub_body());
            let roots = crate::sandbox::prepare_writable_roots(&[root.clone(), tmp.clone()])
                .expect("prepare roots");
            let spec = SandboxSpec::from_prepared_roots(roots).expect("open prepared roots");
            let mut env = GreppyEnv::with_binary(bin, root.clone())
                .unwrap()
                .with_sandbox(SandboxMode::Enforce(spec));

            let probe_dir = tmp.join("..").join("C");
            let _ = fs::create_dir_all(&probe_dir);
            let probe = probe_dir.join(format!(
                "greppy-sandbox-escape-c-env-{}-{}",
                std::process::id(),
                STUB_SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_file(&probe);
            let out = env.call_tool(
                "greppy",
                &json!({"args": ["bash-smart", "--", "bash", "-lc", format!("touch '{}'", probe.display())]}),
            );
            assert!(
                out.is_error,
                "TMPDIR/../C escape must be an error; content={}",
                out.content
            );
            assert!(
                !probe.exists(),
                "escape probe must not exist after sandboxed touch: {}",
                probe.display()
            );
            // Defensive cleanup.
            let _ = fs::remove_file(&probe);
            let _ = fs::remove_dir_all(&root);
        }

        /// S4(f) / device plumbing: shell redirection to `/dev/null` must work
        /// under Enforce (explicit `/dev/null` allow, not via a broad /dev rule).
        #[test]
        fn enforce_dev_null_redirect_ok() {
            if !sandbox_exec_available() {
                return;
            }
            let root = temp_root();
            let bin = write_stub(real_bash_stub_body());
            let roots =
                crate::sandbox::prepare_writable_roots(&[root.clone(), std::env::temp_dir()])
                    .expect("prepare roots");
            let spec = SandboxSpec::from_prepared_roots(roots).expect("open prepared roots");
            let mut env = GreppyEnv::with_binary(bin, root.clone())
                .unwrap()
                .with_sandbox(SandboxMode::Enforce(spec));
            let out = env.call_tool(
                "greppy",
                &json!({"args": ["bash-smart", "--", "bash", "-lc", "echo x >/dev/null"]}),
            );
            assert!(
                !out.is_error,
                "echo x >/dev/null under sandbox: {}",
                out.content
            );
            let _ = fs::remove_dir_all(&root);
        }
    }
}
