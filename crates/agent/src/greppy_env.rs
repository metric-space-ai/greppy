//! Production [`ExecutionEnv`](crate::env::ExecutionEnv): two tools over self-invocation.
//!
//! The model sees exactly two tools — `greppy` and `bash`. Both dispatch by
//! spawning the greppy binary as a subprocess with captured stdio (self-invocation).
//! In production the binary is `std::env::current_exe()`; tests inject a stub
//! via [`GreppyEnv::with_binary`].
//!
//! Capture policy: stdout is read fully, then stderr is appended after stdout
//! (separated by a newline only when both are non-empty). This is deterministic
//! and avoids interleaving races.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::env::{ExecutionEnv, ToolOutcome};
use crate::protocol::ToolDefinition;

/// Default wall-clock budget for the `bash` tool (300 s).
pub const DEFAULT_BASH_TIMEOUT: Duration = Duration::from_secs(300);

/// Default combined stdout+stderr cap (64 KiB).
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 65_536;

/// Production execution environment: `greppy` + `bash` over self-invocation.
#[derive(Debug, Clone)]
pub struct GreppyEnv {
    greppy_bin: PathBuf,
    root: PathBuf,
    bash_timeout: Duration,
    max_output_bytes: usize,
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
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        })
    }

    /// Override the `bash` tool wall-clock timeout.
    pub fn with_bash_timeout(mut self, timeout: Duration) -> Self {
        self.bash_timeout = timeout;
        self
    }

    /// Override the combined-output byte cap.
    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// Repository root this env operates on.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to the greppy binary (or test stub) that tools re-invoke.
    pub fn greppy_bin(&self) -> &Path {
        &self.greppy_bin
    }

    fn greppy_tool_def() -> ToolDefinition {
        ToolDefinition {
            name: "greppy".to_string(),
            description: "Run one greppy command. Pass argv as an array, e.g. [\"who-calls\", \"my_func\", \"--code\"]. All search, navigate, read and edit commands are available.".to_string(),
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

    fn bash_tool_def() -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Run a shell command in the repository root. Output is compacted: verdict line, then errors/warnings.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run via bash -lc."
                    }
                },
                "required": ["command"]
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

        let mut cmd = Command::new(&self.greppy_bin);
        cmd.args(&args)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        match run_capture(&mut cmd, None) {
            Ok(captured) => finalize_outcome(captured, self.max_output_bytes),
            Err(msg) => ToolOutcome::err(msg),
        }
    }

    fn call_bash(&self, arguments: &Value) -> ToolOutcome {
        let command = match parse_string_field(arguments, "command") {
            Ok(c) => c,
            Err(msg) => return ToolOutcome::err(msg),
        };

        let mut cmd = Command::new(&self.greppy_bin);
        cmd.args(["bash-smart", "--", "bash", "-lc", &command])
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        match run_capture(&mut cmd, Some(self.bash_timeout)) {
            Ok(captured) => finalize_outcome(captured, self.max_output_bytes),
            Err(msg) => ToolOutcome::err(msg),
        }
    }
}

impl ExecutionEnv for GreppyEnv {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![Self::greppy_tool_def(), Self::bash_tool_def()]
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> ToolOutcome {
        match name {
            "greppy" => self.call_greppy(arguments),
            "bash" => self.call_bash(arguments),
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
            "greppy tool forbids recursive agent invocation (first arg {first:?})"
        ));
    }
    if args.iter().any(|a| a == "--root") {
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

fn parse_string_field(arguments: &Value, field: &str) -> Result<String, String> {
    let obj = arguments
        .as_object()
        .ok_or_else(|| format!("tool arguments must be a JSON object (missing {field})"))?;
    let value = obj
        .get(field)
        .ok_or_else(|| format!("missing required field: {field}"))?;
    value
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("field {field} must be a string"))
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
fn run_capture(cmd: &mut Command, timeout: Option<Duration>) -> Result<Captured, String> {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // New session/process group so killpg can reap the whole tree.
        cmd.process_group(0);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn greppy binary: {e}"))?;

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

    let start = Instant::now();
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
    let body = truncate_output(body, max_output_bytes);
    if captured.success {
        ToolOutcome::ok(body)
    } else {
        ToolOutcome::err(body)
    }
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
        let script = format!("#!/bin/sh\n{body}\n");
        fs::write(&path, script).expect("write stub");
        let mut perms = fs::metadata(&path).expect("meta").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).expect("chmod");
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
    fn tool_definitions_exactly_two_named_greppy_and_bash() {
        let (env, _, _) = env_with_stub("exit 0");
        let defs = env.tool_definitions();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "greppy");
        assert_eq!(defs[1].name, "bash");

        let greppy_schema = &defs[0].input_schema;
        assert_eq!(greppy_schema["type"], "object");
        assert!(greppy_schema["properties"].get("args").is_some());
        assert_eq!(greppy_schema["required"], json!(["args"]));

        let bash_schema = &defs[1].input_schema;
        assert_eq!(bash_schema["type"], "object");
        assert!(bash_schema["properties"].get("command").is_some());
        assert_eq!(bash_schema["required"], json!(["command"]));
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
    fn bash_routes_through_bash_smart_with_cwd_root() {
        // Record argv and cwd.
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
        let out = env.call_tool("bash", &json!({"command": "echo hi"}));
        assert!(!out.is_error, "content={}", out.content);

        let recorded = fs::read_to_string(&record).expect("record");
        let lines: Vec<&str> = recorded.lines().collect();
        // argv: bash-smart -- bash -lc <cmd>
        assert!(lines.len() >= 5, "recorded={recorded:?}");
        assert_eq!(lines[0], "bash-smart");
        assert_eq!(lines[1], "--");
        assert_eq!(lines[2], "bash");
        assert_eq!(lines[3], "-lc");
        assert_eq!(lines[4], "echo hi");
        let cwd_line = lines.iter().find(|l| l.starts_with("cwd=")).expect("cwd");
        let cwd = cwd_line.trim_start_matches("cwd=");
        let got = fs::canonicalize(cwd).unwrap_or_else(|_| PathBuf::from(cwd));
        let want = fs::canonicalize(&root).unwrap_or(root);
        assert_eq!(got, want);
    }

    #[test]
    fn bash_timeout_kills_and_reports() {
        let (env, _, _) = env_with_stub("sleep 5\nexit 0\n");
        let mut env = env.with_bash_timeout(Duration::from_secs(1));
        let start = Instant::now();
        let out = env.call_tool("bash", &json!({"command": "sleep 5"}));
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
}
