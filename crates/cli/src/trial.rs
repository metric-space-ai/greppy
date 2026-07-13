use clap::{Args, ValueEnum};
use greppy_core::error::{Error, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "greppy.project-trial.v1";
const PROMPT_VERSION: &str = "greppy.project-trial.prompt.v1";
const EXIT_QUALITY_REGRESSION: i32 = 1;
const EXIT_INCONCLUSIVE: i32 = 2;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Args)]
pub struct TrialArgs {
    /// The project question given unchanged to both arms.
    #[arg(long)]
    question: String,

    /// Mechanical check applied to each final answer.
    #[arg(long, value_enum)]
    check: TrialCheck,

    /// Symbol the question and check concern.
    #[arg(long)]
    symbol: String,

    /// Required case-sensitive literal in each correct answer. Repeatable.
    #[arg(long, required = true)]
    expect: Vec<String>,

    /// Case-sensitive literal that must not occur in a correct answer. Repeatable.
    #[arg(long)]
    forbid: Vec<String>,

    /// Agent runner. The v1 protocol supports Pi only.
    #[arg(long, value_enum)]
    runner: TrialRunner,

    /// Pi provider name.
    #[arg(long)]
    provider: String,

    /// Pi model ID or pattern.
    #[arg(long)]
    model: String,

    /// Per-index and per-arm process timeout.
    #[arg(
        long,
        default_value_t = 300,
        value_parser = clap::value_parser!(u64).range(1..=3600)
    )]
    timeout_seconds: u64,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TrialCheck {
    WhoCalls,
}

impl TrialCheck {
    fn as_str(self) -> &'static str {
        match self {
            Self::WhoCalls => "who-calls",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TrialRunner {
    Pi,
}

impl TrialRunner {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pi => "pi",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Baseline,
    Greppy,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Greppy => "greppy",
        }
    }
}

struct RepositoryIdentity {
    root: PathBuf,
    commit: String,
    tree: String,
}

struct WorktreeState {
    clean: bool,
    head_matches: bool,
}

struct CapturedProcess {
    return_code: Option<i32>,
    timed_out: bool,
    wall_time_ms: u64,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

struct ParsedTrace {
    turns: u64,
    answer: String,
    reported_error: Option<String>,
    tool_calls: Vec<Value>,
    source_open_calls: Vec<Value>,
    tool_results: Vec<Value>,
    tool_result_chars: u64,
    token_counters: BTreeMap<String, u64>,
    first_turn_token_counters: BTreeMap<String, u64>,
    later_turn_token_counters: BTreeMap<String, u64>,
    token_usage_reported: bool,
    first_turn_usage_reported: bool,
    later_turn_usage_reported: bool,
    usage_turns_reported: u64,
    non_json_lines: u64,
    invokes_greppy: bool,
}

struct ArmOutcome {
    valid: bool,
    grade_passed: bool,
    metrics: ComparisonMetrics,
    value: Value,
}

#[derive(Clone, Copy, Default)]
struct TokenSummary {
    input_tokens: Option<u64>,
    uncached_input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_tokens: Option<u64>,
    cache_write_tokens: Option<u64>,
}

struct ComparisonMetrics {
    tool_calls: u64,
    source_open_calls: u64,
    tool_result_chars: u64,
    turns: u64,
    wall_time_ms: u64,
    tokens: TokenSummary,
}

pub(crate) fn run(args: TrialArgs, raw_root: Option<&str>) -> Result<i32> {
    let raw_root = raw_root.ok_or_else(|| {
        Error::Invalid("trial requires --root DIR naming the Git repository root".into())
    })?;
    validate_args(&args)?;

    let request = request_json(&args);
    let result = execute(&args, raw_root);
    let (report, exit_code) = match result {
        Ok(result) => result,
        Err(reason) => (
            json!({
                "schema_version": SCHEMA,
                "status": "inconclusive",
                "repository": { "root": raw_root },
                "request": request,
                "runtime": null,
                "protocol": protocol_json(None, None, None),
                "arms": [],
                "reasons": [reason],
            }),
            EXIT_INCONCLUSIVE,
        ),
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| Error::Invalid(format!("serialize trial report: {error}")))?
    );
    Ok(exit_code)
}

fn validate_args(args: &TrialArgs) -> Result<()> {
    for (name, value) in [
        ("--question", args.question.as_str()),
        ("--symbol", args.symbol.as_str()),
        ("--provider", args.provider.as_str()),
        ("--model", args.model.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(Error::Invalid(format!("trial {name} must not be empty")));
        }
    }
    if args.expect.iter().any(|value| value.is_empty()) {
        return Err(Error::Invalid(
            "trial --expect literals must not be empty".into(),
        ));
    }
    if args.forbid.iter().any(|value| value.is_empty()) {
        return Err(Error::Invalid(
            "trial --forbid literals must not be empty".into(),
        ));
    }
    Ok(())
}

fn request_json(args: &TrialArgs) -> Value {
    json!({
        "question": &args.question,
        "check": args.check.as_str(),
        "symbol": &args.symbol,
        "expected_literals": &args.expect,
        "forbidden_literals": &args.forbid,
        "runner": args.runner.as_str(),
        "provider": &args.provider,
        "model": &args.model,
        "timeout_seconds": args.timeout_seconds,
    })
}

fn execute(args: &TrialArgs, raw_root: &str) -> std::result::Result<(Value, i32), String> {
    let repository = preflight_repository(Path::new(raw_root))?;
    let greppy_bin = std::env::current_exe()
        .map_err(|error| format!("locate the running Greppy executable: {error}"))?
        .canonicalize()
        .map_err(|error| format!("resolve the running Greppy executable: {error}"))?;
    let pi_bin = resolve_executable(args.runner.as_str())?;
    let greppy_sha256 = sha256_file(&greppy_bin)?;
    let pi_sha256 = sha256_file(&pi_bin)?;
    let baseline_prompt = system_prompt(Arm::Baseline, &greppy_bin, &args.symbol);
    let greppy_prompt = system_prompt(Arm::Greppy, &greppy_bin, &args.symbol);
    let prompt_hashes = json!({
        "baseline": sha256_bytes(baseline_prompt.as_bytes()),
        "greppy": sha256_bytes(greppy_prompt.as_bytes()),
    });

    let mut scratch = TrialScratch::create(&repository.root, &repository.commit)?;
    let timeout = Duration::from_secs(args.timeout_seconds);
    let runtime = runtime_identity(
        &scratch,
        &pi_bin,
        &pi_sha256,
        &greppy_bin,
        &greppy_sha256,
        timeout,
    )?;
    let preindex = preindex_greppy(&scratch, &greppy_bin, timeout)?;

    for arm in [Arm::Baseline, Arm::Greppy] {
        let state = inspect_worktree(scratch.worktree(arm), &repository.commit)?;
        if !state.clean || !state.head_matches {
            return Err(format!(
                "{} worktree was not clean at the pinned commit before measurement",
                arm.as_str()
            ));
        }
    }

    let baseline = run_arm(
        Arm::Baseline,
        args,
        &scratch,
        &pi_bin,
        &baseline_prompt,
        &repository.commit,
        timeout,
    )?;
    let greppy = run_arm(
        Arm::Greppy,
        args,
        &scratch,
        &pi_bin,
        &greppy_prompt,
        &repository.commit,
        timeout,
    )?;

    let mut reasons = Vec::new();
    if !baseline.valid {
        reasons.push("baseline arm was invalid".to_string());
    }
    if !greppy.valid {
        reasons.push("greppy arm was invalid".to_string());
    }

    let source_state = inspect_worktree(&repository.root, &repository.commit)?;
    if !source_state.clean || !source_state.head_matches {
        reasons.push("target repository changed during the trial".to_string());
    }

    let cleanup_ok = match scratch.cleanup() {
        Ok(()) => true,
        Err(error) => {
            reasons.push(format!("disposable worktree cleanup failed: {error}"));
            false
        }
    };

    let pair_valid = baseline.valid
        && greppy.valid
        && source_state.clean
        && source_state.head_matches
        && cleanup_ok;
    let comparison = comparison_json(&baseline, &greppy, pair_valid);
    let (status, exit_code) = if pair_valid && baseline.grade_passed && greppy.grade_passed {
        ("valid_observation", 0)
    } else if pair_valid && baseline.grade_passed && !greppy.grade_passed {
        reasons.push(
            "Greppy arm failed the mechanical answer check while baseline passed".to_string(),
        );
        ("quality_regression", EXIT_QUALITY_REGRESSION)
    } else {
        if pair_valid && !baseline.grade_passed {
            reasons.push("baseline arm failed the mechanical answer check".to_string());
        }
        ("inconclusive", EXIT_INCONCLUSIVE)
    };

    Ok((
        json!({
            "schema_version": SCHEMA,
            "status": status,
            "repository": {
                "root": repository.root,
                "commit": repository.commit,
                "tree": repository.tree,
                "clean_committed_root": true,
            },
            "request": request_json(args),
            "runtime": runtime,
            "protocol": protocol_json(Some(prompt_hashes), Some(preindex), Some(cleanup_ok)),
            "arms": [baseline.value, greppy.value],
            "comparison": comparison,
            "reasons": reasons,
        }),
        exit_code,
    ))
}

fn protocol_json(
    prompt_hashes: Option<Value>,
    preindex: Option<Value>,
    worktrees_removed: Option<bool>,
) -> Value {
    json!({
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hashes,
        "arm_order": ["baseline", "greppy"],
        "deterministic_arm_order": true,
        "detached_worktree_per_arm": true,
        "worktrees_outside_target_repository": true,
        "isolated_greppy_store_per_arm": true,
        "isolated_pi_config_and_session_per_arm": true,
        "ambient_context_skills_templates_extensions_disabled": true,
        "preindexed_arm": "greppy",
        "preindex_outside_measurement": true,
        "preindex": preindex,
        "worktrees_removed": worktrees_removed,
    })
}

fn preflight_repository(raw_root: &Path) -> std::result::Result<RepositoryIdentity, String> {
    let root = raw_root
        .canonicalize()
        .map_err(|error| format!("canonicalize --root {}: {error}", raw_root.display()))?;
    if !root.is_dir() {
        return Err(format!("--root is not a directory: {}", root.display()));
    }

    let top = git_stdout(&root, &["rev-parse", "--show-toplevel"])?;
    let top = PathBuf::from(top.trim())
        .canonicalize()
        .map_err(|error| format!("canonicalize Git top-level: {error}"))?;
    if top != root {
        return Err(format!(
            "--root must name the Git top-level exactly (resolved top-level: {})",
            top.display()
        ));
    }

    let commit = git_stdout(&root, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    let commit = commit.trim().to_ascii_lowercase();
    if !valid_git_object_id(&commit) {
        return Err("Git HEAD did not resolve to a full commit object ID".into());
    }
    let tree = git_stdout(&root, &["rev-parse", "--verify", "HEAD^{tree}"])?;
    let tree = tree.trim().to_ascii_lowercase();
    if !valid_git_object_id(&tree) {
        return Err("Git HEAD did not resolve to a full tree object ID".into());
    }

    let status = git_output(
        &root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
    )?;
    if !status.stdout.is_empty() {
        return Err("--root must be clean, including staged and untracked files".into());
    }

    Ok(RepositoryIdentity { root, commit, tree })
}

fn valid_git_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn inspect_worktree(
    root: &Path,
    expected_commit: &str,
) -> std::result::Result<WorktreeState, String> {
    let head = git_stdout(root, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    let status = git_output(
        root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
    )?;
    Ok(WorktreeState {
        clean: status.stdout.is_empty(),
        head_matches: head.trim().eq_ignore_ascii_case(expected_commit),
    })
}

struct GitOutput {
    stdout: Vec<u8>,
}

fn git_output(root: &Path, args: &[&str]) -> std::result::Result<GitOutput, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("run git {}: {error}", args.join(" ")))?;
    if !output.status.success() {
        let stderr = bounded_message(&output.stderr);
        return Err(format!(
            "git {} failed with {}{}",
            args.join(" "),
            display_status(output.status),
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(GitOutput {
        stdout: output.stdout,
    })
}

fn git_stdout(root: &Path, args: &[&str]) -> std::result::Result<String, String> {
    let output = git_output(root, args)?;
    String::from_utf8(output.stdout).map_err(|error| format!("git output was not UTF-8: {error}"))
}

struct TrialScratch {
    source_root: PathBuf,
    base: PathBuf,
    baseline_worktree: PathBuf,
    greppy_worktree: PathBuf,
    cleaned: bool,
}

impl TrialScratch {
    fn create(source_root: &Path, commit: &str) -> std::result::Result<Self, String> {
        let base = create_private_temp_dir(source_root)?;
        let baseline_worktree = base.join("baseline");
        let greppy_worktree = base.join("greppy");
        let mut scratch = Self {
            source_root: source_root.to_path_buf(),
            base,
            baseline_worktree,
            greppy_worktree,
            cleaned: false,
        };
        scratch.add_worktree(Arm::Baseline, commit)?;
        scratch.add_worktree(Arm::Greppy, commit)?;
        for arm in [Arm::Baseline, Arm::Greppy] {
            for kind in ["store", "pi-config", "pi-session"] {
                create_private_directory(&scratch.state_dir(arm, kind))?;
            }
        }
        Ok(scratch)
    }

    fn add_worktree(&mut self, arm: Arm, commit: &str) -> std::result::Result<(), String> {
        let path = self.worktree(arm);
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.source_root)
            .args(["worktree", "add", "--detach"])
            .arg(path)
            .arg(commit)
            .stdin(Stdio::null())
            .output()
            .map_err(|error| format!("create {} worktree: {error}", arm.as_str()))?;
        if !output.status.success() {
            return Err(format!(
                "create {} worktree failed with {}: {}",
                arm.as_str(),
                display_status(output.status),
                bounded_message(&output.stderr)
            ));
        }
        Ok(())
    }

    fn worktree(&self, arm: Arm) -> &Path {
        match arm {
            Arm::Baseline => &self.baseline_worktree,
            Arm::Greppy => &self.greppy_worktree,
        }
    }

    fn state_dir(&self, arm: Arm, kind: &str) -> PathBuf {
        self.base.join(format!("{}-{kind}", arm.as_str()))
    }

    fn cleanup(&mut self) -> std::result::Result<(), String> {
        let mut failures = Vec::new();
        for path in [&self.greppy_worktree, &self.baseline_worktree] {
            if !path.exists() {
                continue;
            }
            let output = Command::new("git")
                .arg("-C")
                .arg(&self.source_root)
                .args(["worktree", "remove", "--force"])
                .arg(path)
                .stdin(Stdio::null())
                .output();
            match output {
                Ok(output) if output.status.success() => {}
                Ok(output) => failures.push(format!(
                    "remove {} ({})",
                    path.display(),
                    display_status(output.status)
                )),
                Err(error) => failures.push(format!("remove {}: {error}", path.display())),
            }
        }
        if let Err(error) = fs::remove_dir_all(&self.base) {
            if self.base.exists() {
                failures.push(format!("remove {}: {error}", self.base.display()));
            }
        }
        self.cleaned = failures.is_empty() && !self.base.exists();
        if self.cleaned {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }
}

impl Drop for TrialScratch {
    fn drop(&mut self) {
        if !self.cleaned {
            let _ = self.cleanup();
        }
    }
}

fn create_private_temp_dir(source_root: &Path) -> std::result::Result<PathBuf, String> {
    let temp_parent = std::env::temp_dir()
        .canonicalize()
        .map_err(|error| format!("canonicalize system temporary directory: {error}"))?;
    let parent = if temp_parent.starts_with(source_root) {
        source_root
            .parent()
            .ok_or_else(|| "cannot place trial worktrees outside --root".to_string())?
            .to_path_buf()
    } else {
        temp_parent
    };
    for _ in 0..100 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = parent.join(format!(
            "greppy-project-trial-{}-{stamp}-{counter}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => {
                create_private_directory(&path)?;
                if path.starts_with(source_root) {
                    let _ = fs::remove_dir(&path);
                    return Err("trial worktree directory resolved inside --root".into());
                }
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "create private trial directory {}: {error}",
                    path.display()
                ));
            }
        }
    }
    Err("could not allocate a unique private trial directory".into())
}

fn create_private_directory(path: &Path) -> std::result::Result<(), String> {
    fs::create_dir_all(path)
        .map_err(|error| format!("create private directory {}: {error}", path.display()))?;
    greppy_core::cache::secure_private_directory(path)
        .map_err(|error| format!("secure private directory {}: {error}", path.display()))
}

fn resolve_executable(name: &str) -> std::result::Result<PathBuf, String> {
    let requested = Path::new(name);
    if requested.components().count() > 1 {
        return canonical_executable(requested);
    }
    let path = std::env::var_os("PATH").ok_or_else(|| "PATH is not set".to_string())?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(name);
        if executable_file(&candidate) {
            return canonical_executable(&candidate);
        }
        #[cfg(windows)]
        if candidate.extension().is_none() {
            for extension in windows_executable_extensions() {
                let candidate = directory.join(format!("{name}{extension}"));
                if executable_file(&candidate) {
                    return canonical_executable(&candidate);
                }
            }
        }
    }
    Err(format!("executable not found on PATH: {name}"))
}

fn canonical_executable(path: &Path) -> std::result::Result<PathBuf, String> {
    if !executable_file(path) {
        return Err(format!("not an executable file: {}", path.display()));
    }
    path.canonicalize()
        .map_err(|error| format!("resolve executable {}: {error}", path.display()))
}

fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(windows)]
fn windows_executable_extensions() -> Vec<String> {
    std::env::var("PATHEXT")
        .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
        .split(';')
        .filter(|extension| !extension.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn sha256_file(path: &Path) -> std::result::Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("open executable {} for hashing: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("hash executable {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn runtime_identity(
    scratch: &TrialScratch,
    pi_bin: &Path,
    pi_sha256: &str,
    greppy_bin: &Path,
    greppy_sha256: &str,
    timeout: Duration,
) -> std::result::Result<Value, String> {
    let version_timeout = timeout.min(Duration::from_secs(15));
    let mut pi_version = Command::new(pi_bin);
    pi_version
        .arg("--version")
        .current_dir(scratch.worktree(Arm::Baseline))
        .env(
            "PI_CODING_AGENT_DIR",
            scratch.state_dir(Arm::Baseline, "pi-config"),
        )
        .env(
            "PI_CODING_AGENT_SESSION_DIR",
            scratch.state_dir(Arm::Baseline, "pi-session"),
        )
        .env("PI_TELEMETRY", "0")
        .env(
            "GREPPY_STORE_DIR",
            scratch.state_dir(Arm::Baseline, "store"),
        );
    let pi_version = run_captured(
        pi_version,
        &scratch.base,
        "runtime-pi-version",
        version_timeout,
    )?;
    require_version_success("Pi", &pi_version)?;

    let mut greppy_version = Command::new(greppy_bin);
    greppy_version
        .arg("--version")
        .current_dir(scratch.worktree(Arm::Greppy))
        .env("GREPPY_STORE_DIR", scratch.state_dir(Arm::Greppy, "store"));
    let greppy_version = run_captured(
        greppy_version,
        &scratch.base,
        "runtime-greppy-version",
        version_timeout,
    )?;
    require_version_success("Greppy", &greppy_version)?;

    Ok(json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "pi": {
            "path": pi_bin,
            "sha256": pi_sha256,
            "version_output": process_text_output(&pi_version),
        },
        "greppy": {
            "path": greppy_bin,
            "sha256": greppy_sha256,
            "version_output": process_text_output(&greppy_version),
        },
    }))
}

fn require_version_success(
    name: &str,
    output: &CapturedProcess,
) -> std::result::Result<(), String> {
    if output.timed_out || output.return_code != Some(0) {
        return Err(format!(
            "{name} --version failed (return_code={}, timed_out={})",
            optional_code(output.return_code),
            output.timed_out
        ));
    }
    Ok(())
}

fn process_text_output(output: &CapturedProcess) -> Value {
    json!({
        "stdout": String::from_utf8_lossy(&output.stdout).trim(),
        "stderr": String::from_utf8_lossy(&output.stderr).trim(),
    })
}

fn preindex_greppy(
    scratch: &TrialScratch,
    greppy_bin: &Path,
    timeout: Duration,
) -> std::result::Result<Value, String> {
    let worktree = scratch.worktree(Arm::Greppy);
    let store = scratch.state_dir(Arm::Greppy, "store");
    let mut command = Command::new(greppy_bin);
    command
        .arg("--root")
        .arg(worktree)
        .arg("index")
        .arg(worktree)
        .current_dir(worktree)
        .env("GREPPY_STORE_DIR", &store);
    let output = run_captured(command, &scratch.base, "preindex", timeout)?;
    if output.timed_out || output.return_code != Some(0) {
        return Err(format!(
            "Greppy pre-index failed (return_code={}, timed_out={})",
            optional_code(output.return_code),
            output.timed_out
        ));
    }
    Ok(json!({
        "return_code": output.return_code,
        "timed_out": output.timed_out,
        "wall_time_ms": output.wall_time_ms,
        "stdout_sha256": sha256_bytes(&output.stdout),
        "stderr_sha256": sha256_bytes(&output.stderr),
    }))
}

fn run_arm(
    arm: Arm,
    args: &TrialArgs,
    scratch: &TrialScratch,
    pi_bin: &Path,
    prompt: &str,
    expected_commit: &str,
    timeout: Duration,
) -> std::result::Result<ArmOutcome, String> {
    let worktree = scratch.worktree(arm);
    let before = inspect_worktree(worktree, expected_commit)?;
    let store = scratch.state_dir(arm, "store");
    let config = scratch.state_dir(arm, "pi-config");
    let session = scratch.state_dir(arm, "pi-session");
    let user_prompt = format!("Question:\n{}", args.question);

    let mut command = Command::new(pi_bin);
    command
        .args(["-p", "--provider"])
        .arg(&args.provider)
        .arg("--model")
        .arg(&args.model)
        .args([
            "--mode",
            "json",
            "--no-session",
            "--thinking",
            "off",
            "--tools",
            "bash,read",
            "--no-context-files",
            "--no-skills",
            "--no-prompt-templates",
            "--no-extensions",
            "--no-themes",
            "--approve",
            "--session-dir",
        ])
        .arg(&session)
        .arg("--system-prompt")
        .arg(prompt)
        .arg(user_prompt)
        .current_dir(worktree)
        .env("GREPPY_STORE_DIR", &store)
        .env("PI_CODING_AGENT_DIR", &config)
        .env("PI_CODING_AGENT_SESSION_DIR", &session)
        .env("PI_TELEMETRY", "0");

    let output = run_captured(
        command,
        &scratch.base,
        &format!("{}-pi", arm.as_str()),
        timeout,
    )?;
    let trace_sha256 = sha256_bytes(&output.stdout);
    let trace = parse_pi_jsonl(&output.stdout);
    let after = inspect_worktree(worktree, expected_commit)?;
    let grade = grade_answer(&trace.answer, &args.expect, &args.forbid);

    let mut invalid_reasons = Vec::new();
    if !before.clean || !before.head_matches {
        invalid_reasons.push("worktree was dirty or moved before the arm".to_string());
    }
    if output.timed_out {
        invalid_reasons.push("Pi timed out".to_string());
    }
    if output.return_code != Some(0) {
        invalid_reasons.push(format!(
            "Pi return code was {}",
            optional_code(output.return_code)
        ));
    }
    if trace.turns == 0 {
        invalid_reasons.push("Pi reported no completed turns".to_string());
    }
    if trace.answer.trim().is_empty() {
        invalid_reasons.push("Pi produced no final answer".to_string());
    }
    if trace.reported_error.is_some() {
        invalid_reasons.push("Pi reported an agent error".to_string());
    }
    if !after.clean || !after.head_matches {
        invalid_reasons.push("worktree was dirty or moved after the arm".to_string());
    }
    if arm == Arm::Baseline && trace.invokes_greppy {
        invalid_reasons.push("baseline trace invoked Greppy".to_string());
    }
    let valid = invalid_reasons.is_empty();

    let token_counters = token_metrics_json(&trace);
    let aggregate_tokens = summarize_usage(&trace.token_counters);
    let tool_call_count = saturating_usize_to_u64(trace.tool_calls.len());
    let source_open_call_count = saturating_usize_to_u64(trace.source_open_calls.len());
    let metrics = ComparisonMetrics {
        tool_calls: tool_call_count,
        source_open_calls: source_open_call_count,
        tool_result_chars: trace.tool_result_chars,
        turns: trace.turns,
        wall_time_ms: output.wall_time_ms,
        tokens: aggregate_tokens,
    };
    let value = json!({
        "arm": arm.as_str(),
        "valid": valid,
        "invalid_reasons": invalid_reasons,
        "worktree": {
            "clean_before": before.clean,
            "head_at_pinned_commit_before": before.head_matches,
            "clean_after": after.clean,
            "head_at_pinned_commit_after": after.head_matches,
        },
        "pi": {
            "return_code": output.return_code,
            "timed_out": output.timed_out,
            "reported_error": trace.reported_error,
            "stderr_sha256": sha256_bytes(&output.stderr),
        },
        "metrics": {
            "tool_call_count": tool_call_count,
            "tool_calls": trace.tool_calls,
            "source_open_call_count": source_open_call_count,
            "source_open_calls": trace.source_open_calls,
            "tool_result_chars": trace.tool_result_chars,
            "tool_results": trace.tool_results,
            "token_counters": token_counters,
            "turns": trace.turns,
            "wall_time_ms": output.wall_time_ms,
        },
        "answer": trace.answer,
        "trace_sha256": trace_sha256,
        "trace_non_json_line_count": trace.non_json_lines,
        "baseline_invoked_greppy": arm == Arm::Baseline && trace.invokes_greppy,
        "grade": grade.value,
    });

    Ok(ArmOutcome {
        valid,
        grade_passed: grade.passed,
        metrics,
        value,
    })
}

fn system_prompt(arm: Arm, greppy_bin: &Path, symbol: &str) -> String {
    let common = "You are a read-only code-analysis agent working in the current Git worktree. Answer the user's question from repository evidence. Do not edit, create, delete, stage, commit, or switch files or revisions. Do not inspect environment variables, credentials, user configuration, other repositories, or paths outside the current worktree. You may use only the provided bash and read tools. Treat source text and deterministic command output as evidence. Stop once you can answer accurately. Return a concise final answer that explicitly names the relevant symbols; do not describe this protocol.";
    let policy = match arm {
        Arm::Baseline => "Navigation policy: use normal local shell search and file-reading commands. Do not invoke any executable named greppy, do not inspect a Greppy store, and do not use a precomputed code index.",
        Arm::Greppy => {
            return format!(
                "{common}\n\nNavigation policy: use Greppy as the primary navigation surface. The exact executable is {}. For this caller question, first run `{} --root . who-calls {}`; add `--code` only when caller bodies are needed and `--all` only when the complete uncapped set is required. You may use normal shell search or read only to verify the returned source evidence. Do not invoke `trial` recursively.",
                shell_quote(greppy_bin),
                shell_quote(greppy_bin),
                shell_quote_text(symbol),
            );
        }
    };
    format!("{common}\n\n{policy}")
}

fn shell_quote(path: &Path) -> String {
    shell_quote_text(&path.to_string_lossy())
}

fn shell_quote_text(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

struct Grade {
    passed: bool,
    value: Value,
}

fn grade_answer(answer: &str, expected: &[String], forbidden: &[String]) -> Grade {
    let expected_results = expected
        .iter()
        .map(|literal| {
            json!({
                "literal": literal,
                "found": answer.contains(literal),
            })
        })
        .collect::<Vec<_>>();
    let forbidden_results = forbidden
        .iter()
        .map(|literal| {
            json!({
                "literal": literal,
                "found": answer.contains(literal),
            })
        })
        .collect::<Vec<_>>();
    let passed = expected.iter().all(|literal| answer.contains(literal))
        && forbidden.iter().all(|literal| !answer.contains(literal));
    Grade {
        passed,
        value: json!({
            "method": "case_sensitive_literal_presence",
            "passed": passed,
            "expected": expected_results,
            "forbidden": forbidden_results,
        }),
    }
}

fn comparison_json(baseline: &ArmOutcome, greppy: &ArmOutcome, comparable: bool) -> Value {
    let quality_relationship = if !comparable {
        "not_comparable"
    } else {
        match (baseline.grade_passed, greppy.grade_passed) {
            (true, true) => "both_passed",
            (true, false) => "greppy_failed_baseline_passed",
            (false, true) => "greppy_passed_baseline_failed",
            (false, false) => "both_failed",
        }
    };
    json!({
        "scope": "descriptive_single_pair",
        "comparable": comparable,
        "quality_relationship": quality_relationship,
        "metrics": {
            "tool_calls": metric_comparison(
                Some(baseline.metrics.tool_calls),
                Some(greppy.metrics.tool_calls),
            ),
            "source_open_calls": metric_comparison(
                Some(baseline.metrics.source_open_calls),
                Some(greppy.metrics.source_open_calls),
            ),
            "tool_result_chars": metric_comparison(
                Some(baseline.metrics.tool_result_chars),
                Some(greppy.metrics.tool_result_chars),
            ),
            "turns": metric_comparison(
                Some(baseline.metrics.turns),
                Some(greppy.metrics.turns),
            ),
            "wall_time_ms": metric_comparison(
                Some(baseline.metrics.wall_time_ms),
                Some(greppy.metrics.wall_time_ms),
            ),
            "input_tokens": metric_comparison(
                baseline.metrics.tokens.input_tokens,
                greppy.metrics.tokens.input_tokens,
            ),
            "uncached_input_tokens": metric_comparison(
                baseline.metrics.tokens.uncached_input_tokens,
                greppy.metrics.tokens.uncached_input_tokens,
            ),
            "output_tokens": metric_comparison(
                baseline.metrics.tokens.output_tokens,
                greppy.metrics.tokens.output_tokens,
            ),
            "cache_read_tokens": metric_comparison(
                baseline.metrics.tokens.cache_read_tokens,
                greppy.metrics.tokens.cache_read_tokens,
            ),
            "cache_write_tokens": metric_comparison(
                baseline.metrics.tokens.cache_write_tokens,
                greppy.metrics.tokens.cache_write_tokens,
            ),
        },
    })
}

fn metric_comparison(baseline: Option<u64>, greppy: Option<u64>) -> Value {
    let (delta, ratio) = match (baseline, greppy) {
        (Some(baseline), Some(greppy)) => (
            signed_delta(greppy, baseline),
            numeric_ratio(greppy, baseline),
        ),
        _ => (Value::Null, Value::Null),
    };
    json!({
        "baseline": baseline,
        "greppy": greppy,
        "greppy_minus_baseline": delta,
        "greppy_over_baseline": ratio,
    })
}

fn signed_delta(greppy: u64, baseline: u64) -> Value {
    if greppy >= baseline {
        return i64::try_from(greppy - baseline)
            .map(Value::from)
            .unwrap_or(Value::Null);
    }
    i64::try_from(baseline - greppy)
        .ok()
        .and_then(i64::checked_neg)
        .map(Value::from)
        .unwrap_or(Value::Null)
}

fn numeric_ratio(greppy: u64, baseline: u64) -> Value {
    if baseline == 0 {
        return Value::Null;
    }
    let Ok(greppy) = greppy.to_string().parse::<f64>() else {
        return Value::Null;
    };
    let Ok(baseline) = baseline.to_string().parse::<f64>() else {
        return Value::Null;
    };
    serde_json::Number::from_f64(greppy / baseline)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn parse_pi_jsonl(raw: &[u8]) -> ParsedTrace {
    let mut parsed = ParsedTrace {
        turns: 0,
        answer: String::new(),
        reported_error: None,
        tool_calls: Vec::new(),
        source_open_calls: Vec::new(),
        tool_results: Vec::new(),
        tool_result_chars: 0,
        token_counters: BTreeMap::new(),
        first_turn_token_counters: BTreeMap::new(),
        later_turn_token_counters: BTreeMap::new(),
        token_usage_reported: false,
        first_turn_usage_reported: false,
        later_turn_usage_reported: false,
        usage_turns_reported: 0,
        non_json_lines: 0,
        invokes_greppy: false,
    };

    for line in String::from_utf8_lossy(raw).lines() {
        let event: Value = match serde_json::from_str(line) {
            Ok(event) => event,
            Err(_) => {
                if !line.trim().is_empty() {
                    parsed.non_json_lines = parsed.non_json_lines.saturating_add(1);
                }
                continue;
            }
        };
        if event.get("type").and_then(Value::as_str) != Some("turn_end") {
            continue;
        }
        parsed.turns = parsed.turns.saturating_add(1);
        let turn = parsed.turns;
        let message = event.get("message").and_then(Value::as_object);

        if let Some(usage) = message
            .and_then(|message| message.get("usage"))
            .and_then(Value::as_object)
        {
            parsed.token_usage_reported = true;
            parsed.usage_turns_reported = parsed.usage_turns_reported.saturating_add(1);
            if turn == 1 {
                parsed.first_turn_usage_reported = true;
            } else {
                parsed.later_turn_usage_reported = true;
            }
            for (name, value) in usage {
                let Some(value) = nonnegative_integer(value) else {
                    continue;
                };
                let total = parsed.token_counters.entry(name.clone()).or_default();
                *total = total.saturating_add(value);
                let window = if turn == 1 {
                    &mut parsed.first_turn_token_counters
                } else {
                    &mut parsed.later_turn_token_counters
                };
                let window_total = window.entry(name.clone()).or_default();
                *window_total = window_total.saturating_add(value);
            }
        }

        let mut turn_answer = String::new();
        if let Some(content) = message
            .and_then(|message| message.get("content"))
            .and_then(Value::as_array)
        {
            for item in content {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            turn_answer.push_str(text);
                        }
                    }
                    Some("toolCall") => {
                        let call = normalized_tool_call(turn, item);
                        if is_source_open_call(item) {
                            parsed.source_open_calls.push(call.clone());
                        }
                        if tool_call_invokes_greppy(item) {
                            parsed.invokes_greppy = true;
                        }
                        parsed.tool_calls.push(call);
                    }
                    _ => {}
                }
            }
        }
        if !turn_answer.trim().is_empty() {
            parsed.answer = turn_answer;
        }
        if let Some(error) = message
            .and_then(|message| message.get("errorMessage"))
            .and_then(Value::as_str)
        {
            parsed.reported_error = Some(error.to_string());
        }

        if let Some(results) = event.get("toolResults").and_then(Value::as_array) {
            for result in results {
                let chars = tool_result_char_count(result);
                parsed.tool_result_chars = parsed.tool_result_chars.saturating_add(chars);
                parsed.tool_results.push(json!({
                    "turn": turn,
                    "tool_call_id": result.get("toolCallId"),
                    "tool_name": result.get("toolName"),
                    "chars": chars,
                }));
            }
        }
    }
    parsed
}

fn normalized_tool_call(turn: u64, item: &Value) -> Value {
    json!({
        "turn": turn,
        "id": item.get("id"),
        "name": item.get("name"),
        "arguments": item.get("arguments").cloned().unwrap_or(Value::Null),
    })
}

fn tool_result_char_count(result: &Value) -> u64 {
    match result.get("content") {
        Some(Value::String(text)) => saturating_usize_to_u64(text.chars().count()),
        Some(Value::Array(content)) => content
            .iter()
            .filter_map(|item| {
                (item.get("type").and_then(Value::as_str) == Some("text"))
                    .then(|| item.get("text").and_then(Value::as_str))
                    .flatten()
            })
            .map(|text| saturating_usize_to_u64(text.chars().count()))
            .fold(0, u64::saturating_add),
        _ => 0,
    }
}

fn token_metrics_json(trace: &ParsedTrace) -> Value {
    if !trace.token_usage_reported {
        return Value::Null;
    }
    json!({
        "reported_turns": trace.usage_turns_reported,
        "aggregate": usage_window_json(&trace.token_counters, true),
        "first_turn": usage_window_json(
            &trace.first_turn_token_counters,
            trace.first_turn_usage_reported,
        ),
        "later_turns": usage_window_json(
            &trace.later_turn_token_counters,
            trace.later_turn_usage_reported,
        ),
    })
}

fn usage_window_json(counters: &BTreeMap<String, u64>, reported: bool) -> Value {
    if !reported {
        return Value::Null;
    }
    let summary = summarize_usage(counters);
    json!({
        "input_tokens": summary.input_tokens,
        "uncached_input_tokens": summary.uncached_input_tokens,
        "output_tokens": summary.output_tokens,
        "cache_read_tokens": summary.cache_read_tokens,
        "cache_write_tokens": summary.cache_write_tokens,
        "reported_counters": counters_object(counters),
    })
}

fn summarize_usage(counters: &BTreeMap<String, u64>) -> TokenSummary {
    TokenSummary {
        input_tokens: sum_present(
            counters,
            &[
                "input",
                "cacheRead",
                "cacheWrite",
                "cacheWrite1h",
                "cacheWrite5m",
            ],
        ),
        uncached_input_tokens: counters.get("input").copied(),
        output_tokens: counters.get("output").copied(),
        cache_read_tokens: counters.get("cacheRead").copied(),
        cache_write_tokens: sum_present(counters, &["cacheWrite", "cacheWrite1h", "cacheWrite5m"]),
    }
}

fn sum_present(counters: &BTreeMap<String, u64>, names: &[&str]) -> Option<u64> {
    let mut found = false;
    let mut total = 0_u64;
    for name in names {
        if let Some(value) = counters.get(*name) {
            found = true;
            total = total.saturating_add(*value);
        }
    }
    found.then_some(total)
}

fn counters_object(counters: &BTreeMap<String, u64>) -> Value {
    let mut object = Map::new();
    for (name, value) in counters {
        object.insert(name.clone(), Value::from(*value));
    }
    Value::Object(object)
}

fn nonnegative_integer(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        value
            .as_i64()
            .filter(|value| *value >= 0)
            .and_then(|value| u64::try_from(value).ok())
    })
}

fn is_source_open_call(item: &Value) -> bool {
    match item.get("name").and_then(Value::as_str) {
        Some("read") => true,
        Some("bash") => item
            .get("arguments")
            .and_then(|arguments| arguments.get("command"))
            .and_then(Value::as_str)
            .is_some_and(shell_command_opens_source),
        _ => false,
    }
}

fn tool_call_invokes_greppy(item: &Value) -> bool {
    let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
    if name == "greppy" || name.starts_with("greppy_") || name.starts_with("greppy-") {
        return true;
    }
    name == "bash"
        && item
            .get("arguments")
            .and_then(|arguments| arguments.get("command"))
            .and_then(Value::as_str)
            .is_some_and(shell_command_invokes_greppy)
}

fn shell_command_opens_source(command: &str) -> bool {
    shell_segments(command).iter().any(|segment| {
        let words = shell_words(segment);
        let Some((executable, arguments)) = command_and_arguments(&words) else {
            return false;
        };
        match executable_basename(executable) {
            "cat" | "head" | "tail" | "less" | "more" | "bat" | "nl" => true,
            "sed" => arguments.iter().any(|argument| argument.starts_with("-n")),
            _ => false,
        }
    })
}

fn shell_command_invokes_greppy(command: &str) -> bool {
    shell_segments(command).iter().any(|segment| {
        let words = shell_words(segment);
        command_and_arguments(&words)
            .is_some_and(|(executable, _)| executable_basename(executable) == "greppy")
    })
}

fn shell_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    for ch in command.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' && quote != Some('\'') {
            current.push(ch);
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"') {
            if quote == Some(ch) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(ch);
            }
            current.push(ch);
            continue;
        }
        if quote.is_none() && matches!(ch, ';' | '|' | '&' | '\n') {
            if !current.trim().is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(ch);
    }
    if !current.trim().is_empty() {
        segments.push(current);
    }
    segments
}

fn shell_words(segment: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    for ch in segment.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' && quote != Some('\'') {
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"') {
            if quote == Some(ch) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(ch);
            } else {
                current.push(ch);
            }
            continue;
        }
        if quote.is_none() && ch.is_whitespace() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn command_and_arguments(words: &[String]) -> Option<(&str, &[String])> {
    let mut index = 0;
    while index < words.len() {
        let word = words[index]
            .trim_start_matches(['(', '{'])
            .trim_end_matches([')', '}']);
        if word.is_empty()
            || matches!(word, "!" | "if" | "then" | "do" | "while" | "until")
            || is_shell_assignment(word)
        {
            index += 1;
            continue;
        }
        if matches!(word, "env" | "command" | "exec" | "nohup" | "time" | "sudo") {
            index += 1;
            while index < words.len()
                && (words[index].starts_with('-') || is_shell_assignment(&words[index]))
            {
                index += 1;
            }
            continue;
        }
        return Some((word, &words[index + 1..]));
    }
    None
}

fn is_shell_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn executable_basename(executable: &str) -> &str {
    Path::new(executable)
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or(executable)
}

fn run_captured(
    mut command: Command,
    output_dir: &Path,
    label: &str,
    timeout: Duration,
) -> std::result::Result<CapturedProcess, String> {
    let stdout_path = output_dir.join(format!("{label}.stdout"));
    let stderr_path = output_dir.join(format!("{label}.stderr"));
    let stdout = private_output_file(&stdout_path)?;
    let stderr = private_output_file(&stderr_path)?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let start = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|error| format!("start process for {label}: {error}"))?;
    let (status, timed_out) = loop {
        match child
            .try_wait()
            .map_err(|error| format!("wait for {label}: {error}"))?
        {
            Some(status) => break (status, false),
            None if start.elapsed() >= timeout => {
                let status = terminate_child(&mut child, label)?;
                break (status, true);
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    };

    let stdout = fs::read(&stdout_path)
        .map_err(|error| format!("read captured stdout for {label}: {error}"))?;
    let stderr = fs::read(&stderr_path)
        .map_err(|error| format!("read captured stderr for {label}: {error}"))?;
    Ok(CapturedProcess {
        return_code: status.code(),
        timed_out,
        wall_time_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        stdout,
        stderr,
    })
}

fn private_output_file(path: &Path) -> std::result::Result<fs::File, String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create private output file {}: {error}", path.display()))?;
    greppy_core::cache::secure_private_file(path)
        .map_err(|error| format!("secure private output file {}: {error}", path.display()))?;
    Ok(file)
}

fn terminate_child(child: &mut Child, label: &str) -> std::result::Result<ExitStatus, String> {
    #[cfg(unix)]
    unsafe {
        if let Ok(process_group) = i32::try_from(child.id()) {
            libc::kill(-process_group, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }

    let grace = Instant::now();
    while grace.elapsed() < Duration::from_millis(250) {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("wait for timed-out {label}: {error}"))?
        {
            return Ok(status);
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    #[cfg(unix)]
    unsafe {
        if let Ok(process_group) = i32::try_from(child.id()) {
            libc::kill(-process_group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    child
        .wait()
        .map_err(|error| format!("reap timed-out {label}: {error}"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn saturating_usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn display_status(status: ExitStatus) -> String {
    status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal".to_string())
}

fn optional_code(code: Option<i32>) -> String {
    code.map(|code| code.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn bounded_message(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .filter(|ch| !ch.is_control() || matches!(ch, '\n' | '\t'))
        .take(500)
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_captures_exact_calls_usage_and_answer() {
        let raw = br#"{"type":"turn_end","toolResults":[{"toolCallId":"a","toolName":"read","content":[{"type":"text","text":"abc"}]}],"message":{"content":[{"type":"toolCall","id":"a","name":"read","arguments":{"path":"src/lib.rs"}},{"type":"text","text":"caller_one"}],"usage":{"input":10,"output":2,"cacheRead":4}}}"#;
        let parsed = parse_pi_jsonl(raw);
        assert_eq!(parsed.turns, 1);
        assert_eq!(parsed.answer, "caller_one");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.source_open_calls.len(), 1);
        assert_eq!(parsed.tool_result_chars, 3);
        assert_eq!(parsed.token_counters.get("input"), Some(&10));
        assert_eq!(parsed.token_counters.get("cacheRead"), Some(&4));
    }

    #[test]
    fn greppy_detection_only_uses_executable_position() {
        assert!(shell_command_invokes_greppy("greppy who-calls target"));
        assert!(shell_command_invokes_greppy(
            "env X=1 /usr/local/bin/greppy who-calls target"
        ));
        assert!(!shell_command_invokes_greppy("grep -R greppy README.md"));
    }

    #[test]
    fn grading_is_literal_and_mechanical() {
        let grade = grade_answer(
            "caller_one calls target",
            &["caller_one".into()],
            &["caller_two".into()],
        );
        assert!(grade.passed);
        let grade = grade_answer("Caller_one calls target", &["caller_one".into()], &[]);
        assert!(!grade.passed);
    }
}
