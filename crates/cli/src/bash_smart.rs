//! Training-free `bash-smart`: byte-preserving command capture, mechanical
//! diagnostic blocks, regex lifts, skeletons, arithmetic repetition collapse,
//! optional embedding novelty, and hash-guarded paged expansion.

use super::*;
use greppy_indexer::CodeEmbeddingProvider;
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
#[cfg(unix)]
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::LazyLock;

const SHORT_TOTAL_LINES: usize = 80;
const STDERR_VERBATIM_LINES: usize = 40;
const HEAD_LINES: usize = 20;
const SUCCESS_TAIL_LINES: usize = 30;
const FAILURE_TAIL_LINES: usize = 60;
const EXPAND_PAGE_LINES: usize = 400;
const EMBED_BATCH_LINES: usize = 16;
const NOVELTY_TOP_K: usize = 3;
const NOVELTY_DISTANCE_FLOOR: f32 = 0.12;
const PACK_SCHEMA_VERSION: u64 = 1;
const PACK_HEAD_BYTES: u64 = 32 * 1024 * 1024;
const PACK_TAIL_BYTES: u64 = 32 * 1024 * 1024;
const INITIAL_HEARTBEAT_MS: u64 = 15_000;
const HEARTBEAT_INTERVAL_MS: u64 = 60_000;
#[cfg(unix)]
const SIGNAL_GRACE: std::time::Duration = std::time::Duration::from_millis(250);

static PATH_TEMPLATE_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#"(?x)(?:[A-Za-z]:[\\/]|\.{0,2}/|/)[^\s`'\"]+"#)
        .expect("bash-smart path template regex")
});
static HEX_TEMPLATE_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"(?i)\b(?:0x)?[0-9a-f]{6,}\b").expect("bash-smart hex template regex")
});
static DIGITS_TEMPLATE_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\d+").expect("bash-smart digits template regex"));
static ERROR_MARKER_RE: LazyLock<regex::bytes::Regex> = LazyLock::new(|| {
    regex::bytes::Regex::new(
        r"(?i-u)^[\t ]*(?:error\b|fatal\b|panic|FAIL(?:ED)?\b|Traceback|Exception\b|assert(?:ion)? ?(?:failed|error)|E:)",
    )
    .expect("bash-smart error marker regex")
});
static WARNING_MARKER_RE: LazyLock<regex::bytes::Regex> = LazyLock::new(|| {
    regex::bytes::Regex::new(r"(?i-u)^[\t ]*(?:warn(?:ing)?\b|deprecat|note:)")
        .expect("bash-smart warning marker regex")
});

fn heartbeat_tail(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    file.seek(std::io::SeekFrom::Start(len.saturating_sub(4096)))
        .ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let line = text
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    let mut shown: String = line.chars().take(240).collect();
    if line.chars().count() > 240 {
        shown.push('…');
    }
    Some(shown)
}

#[cfg(unix)]
static PENDING_SIGNAL: AtomicI32 = AtomicI32::new(0);

#[derive(Clone, Copy)]
struct RawLine<'a> {
    content: &'a [u8],
    raw: &'a [u8],
}

#[derive(Debug, Clone)]
struct CollapseGroup {
    start: usize,
    end: usize,
    representative: Vec<u8>,
    template: String,
}

impl CollapseGroup {
    fn count(&self) -> usize {
        self.end - self.start + 1
    }
}

#[derive(Debug, Clone)]
struct LiftedLine {
    line: usize,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputStream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Error,
    Warning,
}

#[derive(Debug, Clone)]
struct AnswerLine {
    stream: OutputStream,
    line: usize,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct DiagnosticBlock {
    kind: BlockKind,
    lines: Vec<AnswerLine>,
}

#[derive(Debug, Clone)]
struct MatchGroup {
    representative: AnswerLine,
    count: usize,
}

#[derive(Clone)]
struct StoredRaw {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    content_sha256: String,
    payload: serde_json::Value,
}

#[cfg(unix)]
extern "C" fn record_signal(signal: libc::c_int) {
    let _ = PENDING_SIGNAL.compare_exchange(0, signal, Ordering::Relaxed, Ordering::Relaxed);
}

#[cfg(unix)]
struct SignalGuard {
    previous_int: libc::sighandler_t,
    previous_term: libc::sighandler_t,
}

#[cfg(unix)]
impl SignalGuard {
    fn install() -> std::io::Result<Self> {
        PENDING_SIGNAL.store(0, Ordering::Relaxed);
        let handler = record_signal as *const () as libc::sighandler_t;
        let previous_int = unsafe { libc::signal(libc::SIGINT, handler) };
        if previous_int == libc::SIG_ERR {
            return Err(std::io::Error::last_os_error());
        }
        let previous_term = unsafe { libc::signal(libc::SIGTERM, handler) };
        if previous_term == libc::SIG_ERR {
            let error = std::io::Error::last_os_error();
            unsafe {
                libc::signal(libc::SIGINT, previous_int);
            }
            return Err(error);
        }
        Ok(Self {
            previous_int,
            previous_term,
        })
    }

    fn take() -> Option<i32> {
        match PENDING_SIGNAL.swap(0, Ordering::Relaxed) {
            0 => None,
            signal => Some(signal),
        }
    }
}

#[cfg(unix)]
impl Drop for SignalGuard {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGINT, self.previous_int);
            libc::signal(libc::SIGTERM, self.previous_term);
        }
        PENDING_SIGNAL.store(0, Ordering::Relaxed);
    }
}

pub(crate) fn run(argv: &[String], regexes: &[String], root: Option<&str>) -> Result<i32> {
    if argv.is_empty() {
        return Err(Error::Invalid(
            "bash-smart requires a command after `--`".into(),
        ));
    }
    let matchers = compile_matchers(regexes)?;

    #[cfg(unix)]
    let _signal_guard = SignalGuard::install()
        .map_err(|error| Error::io("install bash-smart signal handlers", error))?;

    // Open the pack namespace before spawning: both pipe-drainer threads write
    // straight into this directory while the child runs. The completed output
    // is never accumulated by `Command::output`, and stdout cannot block stderr
    // (or vice versa) at the kernel pipe limit.
    let store = match open_pack_store(root) {
        Ok(store) => Some(store),
        Err(Error::Lock(_)) => {
            eprintln!(
                "bash-smart: index writer active; command execution continues without expansion storage"
            );
            None
        }
        Err(_) => None,
    };
    let spool_dir = spool_dir(root)?;
    let token = spool_token();
    let stdout_path = spool_dir.join(format!("{token}.stdout"));
    let stderr_path = spool_dir.join(format!("{token}.stderr"));
    let stdout_times = spool_dir.join(format!("{token}.stdout.times"));
    let stderr_times = spool_dir.join(format!("{token}.stderr.times"));

    let mut command = command_for_argv(argv)?;
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let exit_code = if error.kind() == std::io::ErrorKind::NotFound {
                127
            } else {
                126
            };
            let mut stdout = std::io::stdout().lock();
            let _ = writeln!(stdout, "{}", verdict_line(exit_code, 0, 0, None));
            let _ = stdout.flush();
            let _ = writeln!(
                std::io::stderr(),
                "bash-smart: failed to run {}: {error}",
                argv[0]
            );
            return Ok(exit_code);
        }
    };
    let stdout = child.stdout.take().expect("piped child stdout");
    let stderr = child.stderr.take().expect("piped child stderr");
    let stdout_thread = spawn_drain(stdout, stdout_path.clone(), stdout_times.clone());
    let stderr_thread = spawn_drain(stderr, stderr_path.clone(), stderr_times.clone());

    let timeout_ms = std::env::var("GREPPY_BASH_SMART_TIMEOUT_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0);
    let heartbeat_override_ms = std::env::var("GREPPY_BASH_SMART_HEARTBEAT_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0);
    let heartbeat_interval =
        std::time::Duration::from_millis(heartbeat_override_ms.unwrap_or(HEARTBEAT_INTERVAL_MS));
    let mut next_heartbeat =
        std::time::Duration::from_millis(heartbeat_override_ms.unwrap_or(INITIAL_HEARTBEAT_MS));
    let started = std::time::Instant::now();
    let mut timed_out = false;
    let mut forwarded_signal = None;
    let status = loop {
        #[cfg(unix)]
        if let Some(signal) = SignalGuard::take() {
            forwarded_signal = Some(signal);
            break wait_after_forwarded_signal(&mut child, signal)
                .map_err(|error| Error::io("wait for interrupted bash-smart command", error))?;
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                let elapsed = started.elapsed();
                if elapsed >= next_heartbeat {
                    let latest =
                        heartbeat_tail(&stderr_path).or_else(|| heartbeat_tail(&stdout_path));
                    if let Some(latest) = latest {
                        eprintln!(
                            "bash-smart: command still running — pid={}, elapsed={}s; latest child output: {latest}",
                            child.id(),
                            elapsed.as_secs()
                        );
                    } else {
                        eprintln!(
                            "bash-smart: command still running — pid={}, elapsed={}s; child output is being captured and will be summarized on exit",
                            child.id(),
                            elapsed.as_secs()
                        );
                    }
                    next_heartbeat = next_heartbeat.saturating_add(heartbeat_interval);
                }
                if timeout_ms.is_some_and(|limit| started.elapsed().as_millis() >= limit as u128) {
                    timed_out = true;
                    kill_child_tree(&mut child);
                    break child.wait().map_err(|error| {
                        Error::io("wait for timed-out bash-smart command", error)
                    })?;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(error) => return Err(Error::io("wait for bash-smart command", error)),
        }
    };
    let capture_end_micros = started.elapsed().as_micros();
    let stdout_capture = join_drain(stdout_thread, "stdout")?;
    let stderr_capture = join_drain(stderr_thread, "stderr")?;
    let timeout_stdout_line = timed_out
        .then(|| line_before_largest_gap(&stdout_capture.timestamps_path, capture_end_micros))
        .flatten();
    let timeout_stderr_line = timed_out
        .then(|| line_before_largest_gap(&stderr_capture.timestamps_path, capture_end_micros))
        .flatten();
    let exit_code = forwarded_signal
        .map(|signal| 128 + signal)
        .unwrap_or_else(|| child_exit_code(&status));
    let raw = StoredRaw::from_capture(stdout_capture, stderr_capture)?;
    let stdout_lines = split_lines(&raw.stdout);
    let stderr_lines = split_lines(&raw.stderr);
    let blocks = detect_blocks(&stdout_lines, &stderr_lines);
    let matches = collect_matches(&stdout_lines, &stderr_lines, &matchers);
    let error_count = blocks
        .iter()
        .filter(|block| block.kind == BlockKind::Error)
        .count();
    let warning_count = blocks.len().saturating_sub(error_count);
    let annotation = termination_annotation(timed_out, forwarded_signal, &status);
    render_answer_prefix(
        &stdout_lines,
        &stderr_lines,
        &blocks,
        &matches,
        &verdict_line(exit_code, error_count, warning_count, annotation.as_deref()),
    );

    let project = project_for(root).unwrap_or_else(|_| "bash-smart".into());
    let query = argv_for_metadata(argv);
    let interrupted = timed_out || forwarded_signal.is_some() || status.code().is_none();
    let short = stdout_lines.len() + stderr_lines.len() <= SHORT_TOTAL_LINES;

    // Kills and timeouts always receive an id, even when the partial wall is
    // short. Normal short output keeps its raw skeleton bytes after the answer
    // prefix.
    if short && !interrupted {
        if let Some(store) = store.as_ref() {
            let ranges = full_line_range(&stdout_lines);
            let _ = insert_pack(store, &project, &query, &raw, "stdout", &ranges);
        }
        write_stream(false, &raw.stdout);
        write_stream(true, &raw.stderr);
        return Ok(exit_code);
    }

    let stdout_folded = !short || interrupted;
    let stderr_folded =
        stderr_lines.len() > STDERR_VERBATIM_LINES || (interrupted && !stderr_lines.is_empty());
    let stdout_all_groups = collapse_groups(&stdout_lines);
    let stderr_all_groups = collapse_groups(&stderr_lines);
    // The embedding lift runs only when the command FAILED. On success the
    // agent reads the tail confirmation and moves on — spending model time
    // there buys lines nobody asked for; on failure it is exactly where the
    // attention belongs. (Owner-approved gating, 2026-08-03. Skeleton and
    // expand id are unaffected; the block classifier will replace this
    // heuristic behind the same gate.)
    let lift_worthwhile = exit_code != 0;
    let mut lifted_stdout = if lift_worthwhile && stdout_lines.len() > SHORT_TOTAL_LINES {
        novelty_lifts(&stdout_lines, &stdout_all_groups, root)
    } else {
        Vec::new()
    };
    let mut lifted_stderr = if lift_worthwhile && stderr_folded {
        novelty_lifts(&stderr_lines, &stderr_all_groups, root)
    } else {
        Vec::new()
    };
    push_line_lift(&mut lifted_stdout, &stdout_lines, timeout_stdout_line);
    push_line_lift(&mut lifted_stderr, &stderr_lines, timeout_stderr_line);

    let stdout_groups = folded_middle_groups(&stdout_lines, exit_code, &stdout_all_groups);
    let stderr_groups = folded_middle_groups(&stderr_lines, exit_code, &stderr_all_groups);
    let stdout_ranges = expansion_ranges(&stdout_lines, exit_code, &stdout_groups, &lifted_stdout);
    let stderr_ranges = expansion_ranges(&stderr_lines, exit_code, &stderr_groups, &lifted_stderr);
    let stdout_id = store.as_ref().and_then(|store| {
        insert_pack(store, &project, &query, &raw, "stdout", &stdout_ranges).ok()
    });
    let stderr_id = if stderr_folded {
        store.as_ref().and_then(|store| {
            insert_pack(store, &project, &query, &raw, "stderr", &stderr_ranges).ok()
        })
    } else {
        None
    };

    if stdout_folded {
        if let (Some(store), Some(id)) = (store.as_ref(), stdout_id.as_deref()) {
            let gated = byte_gate(store, id, "stdout", &lifted_stdout);
            render_folded(false, &stdout_lines, exit_code, id, &stdout_groups, &gated);
        } else {
            write_stream(false, &raw.stdout);
        }
    } else {
        write_stream(false, &raw.stdout);
    }

    if stderr_folded {
        if let (Some(store), Some(id)) = (store.as_ref(), stderr_id.as_deref()) {
            let gated = byte_gate(store, id, "stderr", &lifted_stderr);
            render_folded(true, &stderr_lines, exit_code, id, &stderr_groups, &gated);
        } else {
            write_stream(true, &raw.stderr);
        }
    } else {
        write_stream(true, &raw.stderr);
    }

    if interrupted
        && ((!raw.stdout.is_empty() && !raw.stdout.ends_with(b"\n"))
            || (!raw.stderr.is_empty() && !raw.stderr.ends_with(b"\n")))
    {
        let _ = writeln!(
            std::io::stderr(),
            "bash-smart: partial output ends with an unterminated line"
        );
    }
    if timed_out {
        let _ = writeln!(
            std::io::stderr(),
            "bash-smart: timed out after {} ms; partial output stored as {}",
            timeout_ms.unwrap_or_default(),
            stdout_id.as_deref().unwrap_or("unavailable")
        );
    } else if let Some(signal) = forwarded_signal {
        let _ = writeln!(
            std::io::stderr(),
            "bash-smart: interrupted by signal {signal}; partial output stored as {}",
            stdout_id.as_deref().unwrap_or("unavailable")
        );
    } else if status.code().is_none() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt as _;
            let _ = writeln!(
                std::io::stderr(),
                "bash-smart: terminated by signal {}; partial output stored as {}",
                status.signal().unwrap_or_default(),
                stdout_id.as_deref().unwrap_or("unavailable")
            );
        }
        #[cfg(not(unix))]
        let _ = writeln!(
            std::io::stderr(),
            "bash-smart: terminated; partial output stored as {}",
            stdout_id.as_deref().unwrap_or("unavailable")
        );
    }

    Ok(exit_code)
}

fn compile_matchers(patterns: &[String]) -> Result<Vec<regex::bytes::Regex>> {
    patterns
        .iter()
        .map(|pattern| {
            regex::bytes::Regex::new(pattern).map_err(|error| {
                Error::Invalid(format!("invalid bash-smart -e regex {pattern:?}: {error}"))
            })
        })
        .collect()
}

fn plural(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("1 {singular}")
    } else {
        format!("{count} {plural}")
    }
}

fn verdict_line(
    exit_code: i32,
    errors: usize,
    warnings: usize,
    annotation: Option<&str>,
) -> String {
    if exit_code == 0 {
        let mut counts = Vec::new();
        if errors > 0 {
            counts.push(plural(errors, "error", "errors"));
        }
        if warnings > 0 {
            counts.push(plural(warnings, "warning", "warnings"));
        }
        if counts.is_empty() {
            "ok — exit 0".into()
        } else {
            format!("ok — exit 0, {}", counts.join(", "))
        }
    } else {
        let mut line = format!(
            "FAILED — exit {exit_code}: {}, {}",
            plural(errors, "error", "errors"),
            plural(warnings, "warning", "warnings")
        );
        if let Some(annotation) = annotation {
            line.push_str(" (");
            line.push_str(annotation);
            line.push(')');
        }
        line
    }
}

fn termination_annotation(
    timed_out: bool,
    forwarded_signal: Option<i32>,
    status: &std::process::ExitStatus,
) -> Option<String> {
    if timed_out {
        return Some("timeout".into());
    }
    if let Some(signal) = forwarded_signal {
        return Some(signal_name(signal));
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        if let Some(signal) = status.signal() {
            return Some(signal_name(signal));
        }
    }
    #[cfg(not(unix))]
    let _ = status;
    None
}

fn signal_name(signal: i32) -> String {
    #[cfg(unix)]
    let name = match signal {
        libc::SIGHUP => Some("SIGHUP"),
        libc::SIGINT => Some("SIGINT"),
        libc::SIGQUIT => Some("SIGQUIT"),
        libc::SIGILL => Some("SIGILL"),
        libc::SIGABRT => Some("SIGABRT"),
        libc::SIGFPE => Some("SIGFPE"),
        libc::SIGKILL => Some("SIGKILL"),
        libc::SIGSEGV => Some("SIGSEGV"),
        libc::SIGPIPE => Some("SIGPIPE"),
        libc::SIGALRM => Some("SIGALRM"),
        libc::SIGTERM => Some("SIGTERM"),
        _ => None,
    };
    #[cfg(not(unix))]
    let name: Option<&str> = None;
    name.map(str::to_owned)
        .unwrap_or_else(|| format!("signal {signal}"))
}

/// Mechanical v1 diagnostic detection. This is deliberately the sole function
/// that decides whether lines form error or warning blocks, so a classifier
/// head can replace it without changing the rendering interface. stderr is
/// retained as an origin signal on every line, but never starts a block by
/// itself; an ambiguous stderr singleton therefore cannot be promoted solely
/// because of its stream.
fn detect_blocks(
    stdout_lines: &[RawLine<'_>],
    stderr_lines: &[RawLine<'_>],
) -> Vec<DiagnosticBlock> {
    fn indentation(bytes: &[u8]) -> usize {
        bytes
            .iter()
            .take_while(|byte| matches!(byte, b' ' | b'\t'))
            .count()
    }

    fn blank(bytes: &[u8]) -> bool {
        bytes.iter().all(u8::is_ascii_whitespace)
    }

    fn detail_prefix(bytes: &[u8]) -> bool {
        let trimmed = bytes
            .iter()
            .position(|byte| !byte.is_ascii_whitespace())
            .map(|start| &bytes[start..])
            .unwrap_or_default();
        let lower = trimmed
            .iter()
            .map(u8::to_ascii_lowercase)
            .collect::<Vec<_>>();
        lower.starts_with(b"at ") || lower.starts_with(b"...") || lower.starts_with(b"caused by")
    }

    let mut blocks = Vec::new();
    for (stream, lines) in [
        (OutputStream::Stdout, stdout_lines),
        (OutputStream::Stderr, stderr_lines),
    ] {
        let mut index = 0usize;
        while index < lines.len() {
            let kind = if ERROR_MARKER_RE.is_match(lines[index].content) {
                Some(BlockKind::Error)
            } else if WARNING_MARKER_RE.is_match(lines[index].content) {
                Some(BlockKind::Warning)
            } else {
                None
            };
            let Some(kind) = kind else {
                index += 1;
                continue;
            };

            let marker_indent = indentation(lines[index].content);
            let mut end = index + 1;
            while end < lines.len() {
                let content = lines[end].content;
                if indentation(content) > marker_indent || blank(content) || detail_prefix(content)
                {
                    end += 1;
                } else {
                    break;
                }
            }
            blocks.push(DiagnosticBlock {
                kind,
                lines: (index..end)
                    .map(|line_index| AnswerLine {
                        stream,
                        line: line_index + 1,
                        bytes: lines[line_index].content.to_vec(),
                    })
                    .collect(),
            });
            index = end;
        }
    }
    blocks
}

fn collect_matches(
    stdout_lines: &[RawLine<'_>],
    stderr_lines: &[RawLine<'_>],
    matchers: &[regex::bytes::Regex],
) -> Vec<MatchGroup> {
    if matchers.is_empty() {
        return Vec::new();
    }
    let mut matching = Vec::new();
    for (stream, lines) in [
        (OutputStream::Stdout, stdout_lines),
        (OutputStream::Stderr, stderr_lines),
    ] {
        matching.extend(
            lines
                .iter()
                .enumerate()
                .filter(|(_, line)| {
                    matchers
                        .iter()
                        .any(|matcher| matcher.is_match(line.content))
                })
                .map(|(index, line)| AnswerLine {
                    stream,
                    line: index + 1,
                    bytes: line.content.to_vec(),
                }),
        );
    }

    let mut groups: Vec<MatchGroup> = Vec::new();
    for line in matching {
        let joins_previous = groups.last().is_some_and(|group| {
            group.representative.stream == line.stream
                && group.representative.line + group.count == line.line
                && template_identical(&group.representative.bytes, &line.bytes)
        });
        if joins_previous {
            groups.last_mut().expect("checked match group").count += 1;
        } else {
            groups.push(MatchGroup {
                representative: line,
                count: 1,
            });
        }
    }
    groups
}

fn template_identical(left: &[u8], right: &[u8]) -> bool {
    left == right
        || normalized_template(left)
            .zip(normalized_template(right))
            .is_some_and(|(left, right)| left.0 == right.0)
}

fn answer_line_is_gated(
    stdout_lines: &[RawLine<'_>],
    stderr_lines: &[RawLine<'_>],
    line: &AnswerLine,
) -> bool {
    let lines = match line.stream {
        OutputStream::Stdout => stdout_lines,
        OutputStream::Stderr => stderr_lines,
    };
    lines
        .get(line.line.saturating_sub(1))
        .is_some_and(|stored| stored.content == line.bytes)
}

fn write_answer_line(
    writer: &mut dyn Write,
    stdout_lines: &[RawLine<'_>],
    stderr_lines: &[RawLine<'_>],
    line: &AnswerLine,
) {
    if !answer_line_is_gated(stdout_lines, stderr_lines, line) {
        return;
    }
    let _ = write!(writer, "{}  ", line.line);
    let _ = writer.write_all(&line.bytes);
    let _ = writer.write_all(b"\n");
}

fn write_answer_prefix(
    writer: &mut dyn Write,
    stdout_lines: &[RawLine<'_>],
    stderr_lines: &[RawLine<'_>],
    blocks: &[DiagnosticBlock],
    matches: &[MatchGroup],
    verdict: &str,
) {
    let _ = writeln!(writer, "{verdict}");
    for kind in [BlockKind::Error, BlockKind::Warning] {
        for block in blocks.iter().filter(|block| block.kind == kind) {
            for line in &block.lines {
                write_answer_line(writer, stdout_lines, stderr_lines, line);
            }
        }
    }
    for group in matches {
        let line = &group.representative;
        if !answer_line_is_gated(stdout_lines, stderr_lines, line) {
            continue;
        }
        let _ = write!(writer, "{}  ", line.line);
        let _ = writer.write_all(&line.bytes);
        if group.count > 1 {
            let _ = write!(writer, " ({} matches)", group.count);
        }
        let _ = writer.write_all(b"\n");
    }
}

fn render_answer_prefix(
    stdout_lines: &[RawLine<'_>],
    stderr_lines: &[RawLine<'_>],
    blocks: &[DiagnosticBlock],
    matches: &[MatchGroup],
    verdict: &str,
) {
    let mut stdout = std::io::stdout().lock();
    write_answer_prefix(
        &mut stdout,
        stdout_lines,
        stderr_lines,
        blocks,
        matches,
        verdict,
    );
    let _ = stdout.flush();
}

pub(crate) fn expand(
    store: &greppy_store::Store,
    pack: greppy_store::ExpandPack,
    json: bool,
) -> Result<i32> {
    let Some((pack, raw)) = relocate_or_refuse(store, pack)? else {
        println!("expand: bash-smart pack hash drift; refusing unverified output");
        return Ok(1);
    };
    let stream = pack
        .summary_json
        .get("stream")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("stdout");
    let bytes = if stream == "stderr" {
        &raw.stderr
    } else {
        &raw.stdout
    };
    let lines = split_lines(bytes);
    let ranges = pack_line_ranges(&pack, lines.len());
    let (page_ranges, remaining_ranges) = take_range_page(&ranges, EXPAND_PAGE_LINES);
    let start = page_ranges.first().map(|(start, _)| *start).unwrap_or(1);
    let end_line = page_ranges.last().map(|(_, end)| *end).unwrap_or(0);
    let next_line = remaining_ranges.first().map(|(start, _)| *start);

    let next = if remaining_ranges.is_empty() {
        None
    } else {
        insert_continuation_pack(store, &pack, &raw, stream, &remaining_ranges).ok()
    };

    if json {
        let page = page_ranges
            .iter()
            .flat_map(|(start, end)| lines[start - 1..*end].iter())
            .map(|line| hex_encode(line.raw))
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "id": pack.id,
                "kind": "bash-smart",
                "stream": stream,
                "content_sha256": raw.content_sha256,
                "start_line": start,
                "end_line": end_line,
                "line_count": lines.len(),
                "raw_line_hex": page,
                "next": next.as_ref().zip(next_line).map(|(id, line)| serde_json::json!({
                    "id": id,
                    "line": line,
                    "command": format!("greppy expand {id}"),
                })),
            }))
            .map_err(|error| Error::Invalid(format!("serialize bash-smart expand: {error}")))?
        );
        return Ok(0);
    }

    let mut stdout = std::io::stdout().lock();
    write_line_ranges(&mut stdout, &lines, &page_ranges);
    if let Some((_, end)) = page_ranges.last() {
        if !lines[*end - 1].raw.ends_with(b"\n") {
            let _ = stdout.write_all(b"\n");
        }
    }
    if !remaining_ranges.is_empty() {
        if let (Some(next_id), Some(next_line)) = (next, next_line) {
            let remaining = range_line_count(&remaining_ranges);
            let _ = writeln!(
                stdout,
                "… {remaining} lines — greppy expand {next_id} continues at {next_line}"
            );
        } else {
            // If allocating the next page fails, deliver it now rather than
            // leave a continuation that cannot be opened.
            write_line_ranges(&mut stdout, &lines, &remaining_ranges);
        }
    }
    Ok(0)
}

struct CapturedStream {
    path: PathBuf,
    timestamps_path: PathBuf,
    byte_len: u64,
    line_count: usize,
    sha256: String,
}

fn spool_dir(root: Option<&str>) -> Result<PathBuf> {
    let dir = match resolve_root(root) {
        Ok(root) => workspace_locator::store_dir(&root).join("bash-smart"),
        Err(_) => std::env::temp_dir().join("greppy-bash-smart"),
    };
    std::fs::create_dir_all(&dir)
        .map_err(|error| Error::io("create bash-smart spool directory", error))?;
    Ok(dir)
}

fn spool_token() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{}-{nanos}", std::process::id())
}

fn spawn_drain<R>(
    mut reader: R,
    path: PathBuf,
    timestamps_path: PathBuf,
) -> std::thread::JoinHandle<std::io::Result<CapturedStream>>
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        use std::io::{Seek, SeekFrom};

        let tail_path = path.with_extension("tail-ring");
        let mut output = std::fs::File::create(&path)?;
        let mut tail = std::fs::File::create(&tail_path)?;
        let mut timestamps = std::fs::File::create(&timestamps_path)?;
        let started = std::time::Instant::now();
        let mut byte_len = 0u64;
        let mut overflow_len = 0u64;
        let mut line_count = 0usize;
        let mut saw_bytes_since_newline = false;
        let mut buffer = [0u8; 16 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            let chunk = &buffer[..read];
            let head_remaining = PACK_HEAD_BYTES.saturating_sub(byte_len);
            let head_len = usize::try_from(head_remaining.min(read as u64)).unwrap_or(read);
            if head_len > 0 {
                output.write_all(&chunk[..head_len])?;
            }
            let mut rest = &chunk[head_len..];
            while !rest.is_empty() {
                let ring_at = overflow_len % PACK_TAIL_BYTES;
                let room = usize::try_from(PACK_TAIL_BYTES - ring_at).unwrap_or(rest.len());
                let take = room.min(rest.len());
                tail.seek(SeekFrom::Start(ring_at))?;
                tail.write_all(&rest[..take])?;
                overflow_len += take as u64;
                rest = &rest[take..];
            }
            byte_len = byte_len.saturating_add(read as u64);
            for byte in chunk {
                saw_bytes_since_newline = true;
                if *byte == b'\n' {
                    line_count += 1;
                    writeln!(timestamps, "{}", started.elapsed().as_micros())?;
                    saw_bytes_since_newline = false;
                }
            }
        }
        if saw_bytes_since_newline {
            line_count += 1;
            writeln!(timestamps, "{}", started.elapsed().as_micros())?;
        }

        if overflow_len > 0 {
            tail.flush()?;
            if overflow_len > PACK_TAIL_BYTES {
                let omitted = overflow_len - PACK_TAIL_BYTES;
                writeln!(
                    output,
                    "\n… bash-smart store gap: {omitted} bytes omitted by pack cap …"
                )?;
                let ring_at = overflow_len % PACK_TAIL_BYTES;
                tail.seek(SeekFrom::Start(ring_at))?;
                std::io::copy(
                    &mut (&mut tail).take(PACK_TAIL_BYTES - ring_at),
                    &mut output,
                )?;
                tail.seek(SeekFrom::Start(0))?;
                std::io::copy(&mut (&mut tail).take(ring_at), &mut output)?;
            } else {
                tail.seek(SeekFrom::Start(0))?;
                std::io::copy(&mut tail, &mut output)?;
            }
        }
        output.flush()?;
        timestamps.flush()?;
        drop(output);
        let _ = std::fs::remove_file(tail_path);
        let sha256 = sha256_file(&path)?;
        Ok(CapturedStream {
            path,
            timestamps_path,
            byte_len,
            line_count,
            sha256,
        })
    })
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn join_drain(
    handle: std::thread::JoinHandle<std::io::Result<CapturedStream>>,
    stream: &str,
) -> Result<CapturedStream> {
    handle
        .join()
        .map_err(|_| Error::Invalid(format!("bash-smart {stream} drainer panicked")))?
        .map_err(|error| Error::io(format!("spool bash-smart {stream}"), error))
}

fn line_before_largest_gap(path: &Path, capture_end_micros: u128) -> Option<usize> {
    let raw = std::fs::read_to_string(path).ok()?;
    let timestamps = raw
        .lines()
        .map(str::parse::<u128>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()?;
    let first = *timestamps.first()?;
    let mut largest_gap = 0u128;
    let mut line = 1usize;
    let mut previous = first;
    for (index, current) in timestamps.iter().copied().enumerate().skip(1) {
        let gap = current.saturating_sub(previous);
        if gap > largest_gap {
            largest_gap = gap;
            line = index;
        }
        previous = current;
    }
    let final_gap = capture_end_micros.saturating_sub(previous);
    if final_gap >= largest_gap {
        line = timestamps.len();
    }
    Some(line)
}

fn push_line_lift(lifts: &mut Vec<LiftedLine>, lines: &[RawLine<'_>], line: Option<usize>) {
    let Some(line) = line.filter(|line| *line > 0) else {
        return;
    };
    let Some(raw) = lines.get(line - 1) else {
        return;
    };
    if lifts.iter().any(|lift| lift.line == line) {
        return;
    }
    lifts.push(LiftedLine {
        line,
        bytes: raw.content.to_vec(),
    });
}

fn command_for_argv(argv: &[String]) -> Result<std::process::Command> {
    if argv.len() == 1 {
        #[cfg(windows)]
        {
            if argv[0].contains('|') {
                return Err(Error::Invalid(
                    "bash-smart refuses pipeline scripts on Windows because cmd.exe has no pipefail contract; pass an argv command without a pipeline"
                        .into(),
                ));
            }
            let mut command = std::process::Command::new("cmd");
            command.arg("/C").arg(&argv[0]);
            return Ok(command);
        }
        #[cfg(not(windows))]
        {
            let mut command = std::process::Command::new("bash");
            command.args(["-o", "pipefail", "-c"]).arg(&argv[0]);
            return Ok(command);
        }
    }

    let assignment_count = argv
        .iter()
        .take_while(|arg| env_assignment(arg).is_some())
        .count();
    let command_argv = &argv[assignment_count..];
    if command_argv.is_empty() {
        return Err(Error::Invalid(
            "bash-smart requires a command after environment assignments".into(),
        ));
    }
    if command_argv[0] == "cd" || command_argv.iter().any(|arg| is_shell_operator(arg)) {
        return Err(Error::Invalid(
            "bash-smart received unquoted shell syntax; pass the complete shell expression as one quoted argument, for example `greppy bash-smart -- \"cd DIR && COMMAND\"`"
                .into(),
        ));
    }

    let mut command = std::process::Command::new(&command_argv[0]);
    command.args(&command_argv[1..]);
    for assignment in &argv[..assignment_count] {
        let (name, value) = env_assignment(assignment).expect("counted assignment");
        command.env(name, value);
    }
    Ok(command)
}

fn env_assignment(arg: &str) -> Option<(&str, &str)> {
    let (name, value) = arg.split_once('=')?;
    let mut chars = name.chars();
    let first = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return None;
    }
    Some((name, value))
}

fn is_shell_operator(arg: &str) -> bool {
    matches!(arg, "&&" | "||" | "|" | ";" | "<" | ">" | ">>") || arg.starts_with("2>")
}

#[cfg(unix)]
fn wait_after_forwarded_signal(
    child: &mut std::process::Child,
    signal: i32,
) -> std::io::Result<std::process::ExitStatus> {
    let process_group = child.id() as i32;
    if unsafe { libc::kill(-process_group, signal) } != 0 {
        let _ = unsafe { libc::kill(process_group, signal) };
    }

    let deadline = std::time::Instant::now() + SIGNAL_GRACE;
    let mut status = None;
    while std::time::Instant::now() < deadline {
        if status.is_none() {
            status = child.try_wait()?;
        }
        let group_is_gone = unsafe { libc::kill(-process_group, 0) } != 0
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
        if group_is_gone {
            if let Some(status) = status.take() {
                return Ok(status);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // A descendant may ignore the forwarded signal while retaining a pipe.
    // Reap the whole dedicated process group before joining either drainer.
    let _ = unsafe { libc::kill(-process_group, libc::SIGKILL) };
    if let Some(status) = status {
        Ok(status)
    } else {
        child.wait()
    }
}

fn kill_child_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let process_group = -(child.id() as i32);
        // The child starts a fresh process group, so descendants that inherited
        // its pipes cannot keep the drainer threads alive after a timeout.
        if unsafe { libc::kill(process_group, libc::SIGKILL) } == 0 {
            return;
        }
    }
    let _ = child.kill();
}

fn child_exit_code(status: &std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        status.signal().map(|signal| 128 + signal).unwrap_or(1)
    }
    #[cfg(not(unix))]
    {
        1
    }
}

impl StoredRaw {
    fn from_capture(stdout: CapturedStream, stderr: CapturedStream) -> Result<Self> {
        // Capture itself is always disk-backed. The bounded delivery pass reads
        // the completed spool only after both child pipes have reached EOF.
        let stdout_bytes = std::fs::read(&stdout.path)
            .map_err(|error| Error::io("read captured bash-smart stdout", error))?;
        let stderr_bytes = std::fs::read(&stderr.path)
            .map_err(|error| Error::io("read captured bash-smart stderr", error))?;
        let content_sha256 = combined_sha256(&stdout_bytes, &stderr_bytes);
        let payload = serde_json::json!({
            "schema_version": PACK_SCHEMA_VERSION,
            "kind": "bash-smart",
            "content_sha256": content_sha256,
            "stdout": {
                "sha256": stdout.sha256,
                "byte_len": stdout.byte_len,
                "line_count": stdout.line_count,
                "path": stdout.path,
                "timestamps_path": stdout.timestamps_path,
                "timestamp_unit": "microseconds_since_capture_start",
            },
            "stderr": {
                "sha256": stderr.sha256,
                "byte_len": stderr.byte_len,
                "line_count": stderr.line_count,
                "path": stderr.path,
                "timestamps_path": stderr.timestamps_path,
                "timestamp_unit": "microseconds_since_capture_start",
            },
            "stream_order": "stdout and stderr captured separately; relative order approximate",
        });
        Ok(Self {
            stdout: stdout_bytes,
            stderr: stderr_bytes,
            content_sha256,
            payload,
        })
    }

    fn decode(payload: &serde_json::Value) -> Option<Self> {
        if payload.get("schema_version")?.as_u64()? != PACK_SCHEMA_VERSION
            || payload.get("kind")?.as_str()? != "bash-smart"
        {
            return None;
        }
        let stdout_path = Path::new(payload.get("stdout")?.get("path")?.as_str()?);
        let stderr_path = Path::new(payload.get("stderr")?.get("path")?.as_str()?);
        let stdout = std::fs::read(stdout_path).ok()?;
        let stderr = std::fs::read(stderr_path).ok()?;
        let content_sha256 = combined_sha256(&stdout, &stderr);
        let claimed = payload.get("content_sha256")?.as_str()?;
        let stdout_claimed = payload.get("stdout")?.get("sha256")?.as_str()?;
        let stderr_claimed = payload.get("stderr")?.get("sha256")?.as_str()?;
        (content_sha256 == claimed
            && sha256(&stdout) == stdout_claimed
            && sha256(&stderr) == stderr_claimed)
            .then_some(Self {
                stdout,
                stderr,
                content_sha256,
                payload: payload.clone(),
            })
    }
}

fn open_pack_store(root: Option<&str>) -> Result<greppy_store::Store> {
    // Expansion storage is optional; executing the requested command is not.
    // Opening a writable graph store while an indexer owns the workspace can
    // block in SQLite migration/journal setup before the target process is
    // even spawned. Skip the pack in that state and preserve bash semantics.
    if workspace_writer_active(root) {
        return Err(Error::Lock(
            "index writer active; bash-smart expansion storage unavailable".into(),
        ));
    }
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    if let Some(parent) = path.parent() {
        workspace_locator::ensure_store_dir(parent)
            .map_err(|error| Error::io("create bash-smart pack store", error))?;
    }
    greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())
        .map_err(Error::from)
}

fn insert_pack(
    store: &greppy_store::Store,
    project: &str,
    query: &str,
    raw: &StoredRaw,
    stream: &str,
    line_ranges: &[(usize, usize)],
) -> std::result::Result<String, greppy_store::Error> {
    let start_line = line_ranges.first().map(|(start, _)| *start).unwrap_or(1);
    let summary = serde_json::json!({
        "text": format!("bash-smart {stream} raw output"),
        "kind": "bash-smart",
        "stream": stream,
        "start_line": start_line,
        "line_ranges": line_ranges,
        "content_sha256": raw.content_sha256,
    });
    store.insert_expand_pack(&greppy_store::NewExpandPack {
        project: project.to_string(),
        command: "bash-smart".into(),
        query: query.to_string(),
        graph_generation: 0,
        summary_json: summary,
        payload_text: format!(
            "bash-smart {stream} raw output sha256:{} starts at {start_line}\n",
            raw.content_sha256
        ),
        payload_json: Some(raw.payload.clone()),
        ttl_secs: expand_ttl_secs(),
    })
}

fn insert_continuation_pack(
    store: &greppy_store::Store,
    previous: &greppy_store::ExpandPack,
    raw: &StoredRaw,
    stream: &str,
    line_ranges: &[(usize, usize)],
) -> std::result::Result<String, greppy_store::Error> {
    let start_line = line_ranges.first().map(|(start, _)| *start).unwrap_or(1);
    store.insert_expand_pack(&greppy_store::NewExpandPack {
        project: previous.project.clone(),
        command: "bash-smart".into(),
        query: previous.query.clone(),
        graph_generation: previous.graph_generation,
        summary_json: serde_json::json!({
            "text": format!("bash-smart {stream} raw output"),
            "kind": "bash-smart",
            "stream": stream,
            "start_line": start_line,
            "line_ranges": line_ranges,
            "content_sha256": raw.content_sha256,
        }),
        payload_text: format!(
            "bash-smart {stream} raw output sha256:{} starts at {start_line}\n",
            raw.content_sha256
        ),
        payload_json: Some(raw.payload.clone()),
        ttl_secs: previous.expires_at.saturating_sub(unix_now_secs()).max(1),
    })
}

fn relocate_or_refuse(
    store: &greppy_store::Store,
    pack: greppy_store::ExpandPack,
) -> Result<Option<(greppy_store::ExpandPack, StoredRaw)>> {
    let expected = pack
        .summary_json
        .get("content_sha256")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    if let Some(raw) = pack.payload_json.as_ref().and_then(StoredRaw::decode) {
        if expected.as_deref() == Some(raw.content_sha256.as_str()) {
            return Ok(Some((pack, raw)));
        }
    }
    let Some(expected) = expected else {
        return Ok(None);
    };

    // Packs are immutable in normal operation. If a copied/partially restored
    // row drifted, relocate by content hash to another valid row; never print
    // bytes merely because they occupy the old id.
    let ids = {
        let mut statement = store
            .conn()
            .prepare("SELECT id FROM expand_packs WHERE command = 'bash-smart' AND id <> ?1")
            .map_err(greppy_store::Error::Sqlite)?;
        let ids = statement
            .query_map([&pack.id], |row| row.get::<_, String>(0))
            .map_err(greppy_store::Error::Sqlite)?
            .filter_map(std::result::Result::ok)
            .collect::<Vec<_>>();
        ids
    };
    for id in ids {
        let Some(candidate) = store.get_expand_pack(&id)? else {
            continue;
        };
        if candidate
            .summary_json
            .get("content_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(expected.as_str())
        {
            continue;
        }
        if let Some(raw) = candidate.payload_json.as_ref().and_then(StoredRaw::decode) {
            if raw.content_sha256 == expected {
                return Ok(Some((candidate, raw)));
            }
        }
    }
    Ok(None)
}

fn byte_gate(
    store: &greppy_store::Store,
    id: &str,
    stream: &str,
    candidates: &[LiftedLine],
) -> Vec<LiftedLine> {
    let Some(raw) = store
        .get_expand_pack(id)
        .ok()
        .flatten()
        .and_then(|pack| pack.payload_json.as_ref().and_then(StoredRaw::decode))
    else {
        return Vec::new();
    };
    let lines = split_lines(if stream == "stderr" {
        &raw.stderr
    } else {
        &raw.stdout
    });
    gate_lifts(&lines, candidates)
}

fn gate_lifts(lines: &[RawLine<'_>], candidates: &[LiftedLine]) -> Vec<LiftedLine> {
    candidates
        .iter()
        .filter(|candidate| {
            candidate.line > 0
                && lines
                    .get(candidate.line - 1)
                    .is_some_and(|stored| stored.content == candidate.bytes)
        })
        .cloned()
        .collect()
}

fn folded_middle_bounds(lines: &[RawLine<'_>], exit_code: i32) -> (usize, usize) {
    let tail = if exit_code == 0 {
        SUCCESS_TAIL_LINES
    } else {
        FAILURE_TAIL_LINES
    };
    let head_end = HEAD_LINES.min(lines.len());
    let tail_start = lines.len().saturating_sub(tail).max(head_end);
    (head_end, tail_start)
}

fn folded_middle_groups(
    lines: &[RawLine<'_>],
    exit_code: i32,
    all_groups: &[CollapseGroup],
) -> Vec<CollapseGroup> {
    let (head_end, tail_start) = folded_middle_bounds(lines, exit_code);
    all_groups
        .iter()
        .filter_map(|group| {
            let start = group.start.max(head_end + 1);
            let end = group.end.min(tail_start);
            (start <= end).then(|| CollapseGroup {
                start: start - head_end,
                end: end - head_end,
                representative: lines[start - 1].raw.to_vec(),
                template: group.template.clone(),
            })
        })
        .collect()
}

fn displayed_middle_lines(
    lines: &[RawLine<'_>],
    exit_code: i32,
    groups: &[CollapseGroup],
    lifted: &[LiftedLine],
) -> Vec<usize> {
    let (head_end, tail_start) = folded_middle_bounds(lines, exit_code);
    let mut displayed = groups
        .iter()
        .filter(|group| group.count() > 1)
        .map(|group| head_end + group.start)
        .collect::<Vec<_>>();
    displayed.extend(
        lifted
            .iter()
            .filter(|line| line.line > head_end && line.line <= tail_start)
            .map(|line| line.line),
    );
    displayed.sort_unstable();
    displayed.dedup();
    displayed
}

fn hidden_middle_ranges(
    head_end: usize,
    tail_start: usize,
    displayed: &[usize],
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut range_start = None;
    for line in (head_end + 1)..=tail_start {
        if displayed.binary_search(&line).is_ok() {
            if let Some(start) = range_start.take() {
                ranges.push((start, line - 1));
            }
        } else if range_start.is_none() {
            range_start = Some(line);
        }
    }
    if let Some(start) = range_start {
        ranges.push((start, tail_start));
    }
    ranges
}

fn full_line_range(lines: &[RawLine<'_>]) -> Vec<(usize, usize)> {
    (!lines.is_empty())
        .then_some((1, lines.len()))
        .into_iter()
        .collect()
}

fn expansion_ranges(
    lines: &[RawLine<'_>],
    exit_code: i32,
    groups: &[CollapseGroup],
    lifted: &[LiftedLine],
) -> Vec<(usize, usize)> {
    let (head_end, tail_start) = folded_middle_bounds(lines, exit_code);
    let displayed = displayed_middle_lines(lines, exit_code, groups, lifted);
    let hidden = hidden_middle_ranges(head_end, tail_start, &displayed);
    if hidden.is_empty() {
        full_line_range(lines)
    } else {
        hidden
    }
}

fn pack_line_ranges(pack: &greppy_store::ExpandPack, line_count: usize) -> Vec<(usize, usize)> {
    if let Some(raw_ranges) = pack
        .summary_json
        .get("line_ranges")
        .and_then(serde_json::Value::as_array)
    {
        return raw_ranges
            .iter()
            .filter_map(|range| {
                let range = range.as_array()?;
                let start = usize::try_from(range.first()?.as_u64()?).ok()?.max(1);
                let end = usize::try_from(range.get(1)?.as_u64()?)
                    .ok()?
                    .min(line_count);
                (start <= end).then_some((start, end))
            })
            .collect();
    }

    let start = pack
        .summary_json
        .get("start_line")
        .and_then(serde_json::Value::as_u64)
        .and_then(|line| usize::try_from(line).ok())
        .unwrap_or(1)
        .max(1);
    (start <= line_count)
        .then_some((start, line_count))
        .into_iter()
        .collect()
}

/// Inclusive 1-based line ranges, the shape every expansion page speaks in.
type LineRanges = Vec<(usize, usize)>;

fn take_range_page(ranges: &[(usize, usize)], page_lines: usize) -> (LineRanges, LineRanges) {
    let mut page = Vec::new();
    let mut remaining = Vec::new();
    let mut available = page_lines;
    for &(start, end) in ranges {
        let count = end - start + 1;
        if available == 0 {
            remaining.push((start, end));
        } else if count <= available {
            page.push((start, end));
            available -= count;
        } else {
            page.push((start, start + available - 1));
            remaining.push((start + available, end));
            available = 0;
        }
    }
    (page, remaining)
}

fn range_line_count(ranges: &[(usize, usize)]) -> usize {
    ranges.iter().map(|(start, end)| end - start + 1).sum()
}

fn write_line_ranges(writer: &mut dyn Write, lines: &[RawLine<'_>], ranges: &[(usize, usize)]) {
    for &(start, end) in ranges {
        for line in &lines[start - 1..end] {
            let _ = writer.write_all(line.raw);
        }
    }
}

fn display_line_ranges(ranges: &[(usize, usize)]) -> String {
    if ranges.len() == 1 {
        let (start, end) = ranges[0];
        if start == end {
            return format!("line {start}");
        }
        return format!("lines {start}-{end}");
    }
    let joined = ranges
        .iter()
        .map(|(start, end)| {
            if start == end {
                start.to_string()
            } else {
                format!("{start}-{end}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("lines {joined}")
}

fn render_folded(
    stderr: bool,
    lines: &[RawLine<'_>],
    exit_code: i32,
    id: &str,
    groups: &[CollapseGroup],
    lifted: &[LiftedLine],
) {
    let mut writer: Box<dyn Write> = if stderr {
        Box::new(std::io::stderr().lock())
    } else {
        Box::new(std::io::stdout().lock())
    };
    let (head_end, tail_start) = folded_middle_bounds(lines, exit_code);

    // Layer 1 is invariant: head and tail are literal raw bytes, even when a
    // repeated middle block has the same shape.
    for line in &lines[..head_end] {
        let _ = writer.write_all(line.raw);
    }
    ensure_newline_after_raw(&mut writer, lines, head_end.checked_sub(1));

    for group in groups.iter().filter(|group| group.count() > 1) {
        let _ = writer.write_all(&group.representative);
        if !group.representative.ends_with(b"\n") {
            let _ = writer.write_all(b"\n");
        }
    }
    for line in lifted
        .iter()
        .filter(|line| line.line > head_end && line.line <= tail_start)
    {
        let _ = write!(writer, "{}:", line.line);
        let _ = writer.write_all(&line.bytes);
        let _ = writer.write_all(b"\n");
    }

    let displayed = displayed_middle_lines(lines, exit_code, groups, lifted);
    let hidden_ranges = hidden_middle_ranges(head_end, tail_start, &displayed);
    if !hidden_ranges.is_empty() {
        let hidden_count = hidden_ranges
            .iter()
            .map(|(start, end)| end - start + 1)
            .sum::<usize>();
        let range_text = display_line_ranges(&hidden_ranges);
        if groups.len() == 1 && groups[0].count() > 1 {
            let noun = if hidden_count == 1 {
                "repeat"
            } else {
                "repeats"
            };
            let _ = writeln!(
                writer,
                "… {range_text} ({hidden_count} collapsed `{}` {noun}) — greppy expand {id}",
                groups[0].template
            );
        } else {
            let noun = if hidden_count == 1 { "line" } else { "lines" };
            let _ = writeln!(
                writer,
                "… {range_text} ({hidden_count} collapsed {noun}) — greppy expand {id}"
            );
        }
    } else {
        let _ = writeln!(writer, "… partial output — greppy expand {id}");
    }

    for line in &lines[tail_start..] {
        let _ = writer.write_all(line.raw);
    }
}

fn ensure_newline_after_raw(
    writer: &mut dyn Write,
    lines: &[RawLine<'_>],
    last_index: Option<usize>,
) {
    if last_index.is_some_and(|index| !lines[index].raw.ends_with(b"\n")) {
        let _ = writer.write_all(b"\n");
    }
}

fn novelty_lifts(
    lines: &[RawLine<'_>],
    groups: &[CollapseGroup],
    root: Option<&str>,
) -> Vec<LiftedLine> {
    let middle_end = lines.len().saturating_sub(SUCCESS_TAIL_LINES);
    if !groups
        .iter()
        .any(|group| group.count() == 1 && group.start > HEAD_LINES && group.start <= middle_end)
    {
        return Vec::new();
    }
    if test_inference_skipped() {
        return Vec::new();
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (lines, root);
        return Vec::new();
    }
    #[cfg(any(unix, windows))]
    {
        let args = EmbeddingCliArgs {
            device: None,
            no_gpu: false,
        };
        let Ok(Some(cfg)) = embedding_config_optional(args) else {
            return Vec::new();
        };
        let key = embedding_query_cache_key(&cfg);
        let status = embed_daemon::status(&cfg, &key);
        if status.get("state").and_then(serde_json::Value::as_str) != Some("ready") {
            return Vec::new();
        }
        let _ = root;
        let mut provider = embed_daemon::DaemonCodeEmbeddingProvider::new(&cfg);
        let mut embedded = Vec::<(usize, Vec<f32>)>::new();
        for chunk in groups.chunks(EMBED_BATCH_LINES) {
            let texts = chunk
                .iter()
                .map(|group| std::str::from_utf8(&group.representative).ok())
                .collect::<Vec<_>>();
            let valid = texts
                .iter()
                .enumerate()
                .filter_map(|(index, text)| text.map(|text| (index, (None, text))))
                .collect::<Vec<_>>();
            if valid.is_empty() {
                continue;
            }
            let docs = valid.iter().map(|(_, doc)| *doc).collect::<Vec<_>>();
            let Ok(vectors) = provider.embed_code_documents(&docs) else {
                return Vec::new();
            };
            if vectors.len() != valid.len() {
                return Vec::new();
            }
            for ((index, _), vector) in valid.into_iter().zip(vectors) {
                embedded.push((groups_index(groups, chunk, index), vector));
            }
        }
        rank_novelty(lines, groups, &embedded)
    }
}

fn groups_index(groups: &[CollapseGroup], chunk: &[CollapseGroup], local_index: usize) -> usize {
    let start = chunk
        .first()
        .and_then(|first| groups.iter().position(|group| group.start == first.start))
        .unwrap_or(0);
    start + local_index
}

fn rank_novelty(
    lines: &[RawLine<'_>],
    groups: &[CollapseGroup],
    embedded: &[(usize, Vec<f32>)],
) -> Vec<LiftedLine> {
    let Some(width) = embedded.first().map(|(_, vector)| vector.len()) else {
        return Vec::new();
    };
    if width == 0 || embedded.iter().any(|(_, vector)| vector.len() != width) {
        return Vec::new();
    }
    let mut centroid = vec![0.0f32; width];
    let mut total_weight = 0.0f32;
    for (group_index, vector) in embedded {
        let weight = groups[*group_index].count() as f32;
        total_weight += weight;
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for (sum, value) in centroid.iter_mut().zip(vector) {
            *sum += (*value / norm) * weight;
        }
    }
    if total_weight == 0.0 {
        return Vec::new();
    }
    for value in &mut centroid {
        *value /= total_weight;
    }
    let centroid_norm = centroid
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    for value in &mut centroid {
        *value /= centroid_norm;
    }

    let mut distances = Vec::new();
    let mut weighted_sum = 0.0f32;
    let mut weighted_square_sum = 0.0f32;
    for (group_index, vector) in embedded {
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let cosine = vector
            .iter()
            .zip(&centroid)
            .map(|(value, center)| (*value / norm) * center)
            .sum::<f32>()
            .clamp(-1.0, 1.0);
        let distance = 1.0 - cosine;
        let weight = groups[*group_index].count() as f32;
        weighted_sum += distance * weight;
        weighted_square_sum += distance * distance * weight;
        distances.push((*group_index, distance));
    }
    let mean = weighted_sum / total_weight;
    let variance = (weighted_square_sum / total_weight - mean * mean).max(0.0);
    let threshold = NOVELTY_DISTANCE_FLOOR.max(mean + 1.5 * variance.sqrt());
    let middle_end = lines.len().saturating_sub(SUCCESS_TAIL_LINES);
    distances.retain(|(group_index, distance)| {
        let group = &groups[*group_index];
        group.count() == 1
            && group.start > HEAD_LINES
            && group.start <= middle_end
            && *distance >= threshold
    });
    distances.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    distances
        .into_iter()
        .take(NOVELTY_TOP_K)
        .map(|(group_index, _)| LiftedLine {
            line: groups[group_index].start,
            bytes: lines[groups[group_index].start - 1].content.to_vec(),
        })
        .collect()
}

fn collapse_groups(lines: &[RawLine<'_>]) -> Vec<CollapseGroup> {
    let mut groups = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let normalized = normalized_template(lines[index].content);
        let mut end = index + 1;
        while end < lines.len()
            && (lines[end].content == lines[index].content
                || normalized.as_ref().is_some_and(|template| {
                    normalized_template(lines[end].content).as_ref() == Some(template)
                }))
        {
            end += 1;
        }
        groups.push(CollapseGroup {
            start: index + 1,
            end,
            representative: lines[index].raw.to_vec(),
            template: normalized
                .map(|(_, display)| display)
                .unwrap_or_else(|| template_display(lines[index].content)),
        });
        index = end;
    }
    groups
}

fn normalized_template(line: &[u8]) -> Option<(String, String)> {
    let text = std::str::from_utf8(line).ok()?;
    let masked = PATH_TEMPLATE_RE.replace_all(text, "<PATH>");
    let masked = HEX_TEMPLATE_RE.replace_all(&masked, "<HEX>");
    let masked = DIGITS_TEMPLATE_RE.replace_all(&masked, "<N>");
    // A wall of bare counters (`seq 1 500`) contains no stable shape worth
    // collapsing. Keep enough alphabetic context to distinguish a template
    // from values that merely happen to share a primitive type.
    if masked.chars().filter(|ch| ch.is_alphabetic()).count() < 4 {
        return None;
    }
    let display = masked
        .replace("<PATH>", "…")
        .replace("<HEX>", "…")
        .replace("<N>", "…");
    Some((masked.into_owned(), display))
}

fn template_display(line: &[u8]) -> String {
    String::from_utf8_lossy(line).into_owned()
}

fn split_lines(bytes: &[u8]) -> Vec<RawLine<'_>> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let mut start = 0usize;
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'\n' {
            let physical = &bytes[start..index];
            // Carriage-return progress rewrites are one displayed line. Keep
            // every stored byte in `raw`, but classify/collapse the final
            // visible rewrite rather than counting each update as a line.
            // A final CR is the CRLF terminator, not an empty rewrite.
            let content_end = if physical.ends_with(b"\r") {
                index - 1
            } else {
                index
            };
            let rewrite_start = bytes[start..content_end]
                .iter()
                .rposition(|byte| *byte == b'\r')
                .map(|position| start + position + 1)
                .unwrap_or(start);
            lines.push(RawLine {
                content: &bytes[rewrite_start..content_end],
                raw: &bytes[start..=index],
            });
            start = index + 1;
        }
    }
    if start < bytes.len() {
        let physical = &bytes[start..];
        let rewrite_start = physical
            .iter()
            .rposition(|byte| *byte == b'\r')
            .map(|position| start + position + 1)
            .unwrap_or(start);
        lines.push(RawLine {
            content: &bytes[rewrite_start..],
            raw: &bytes[start..],
        });
    }
    lines
}

fn write_stream(stderr: bool, bytes: &[u8]) {
    if stderr {
        let _ = std::io::stderr().lock().write_all(bytes);
    } else {
        let _ = std::io::stdout().lock().write_all(bytes);
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn combined_sha256(stdout: &[u8], stderr: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"stdout\0");
    hasher.update((stdout.len() as u64).to_le_bytes());
    hasher.update(stdout);
    hasher.update(b"stderr\0");
    hasher.update((stderr.len() as u64).to_le_bytes());
    hasher.update(stderr);
    format!("{:x}", hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn argv_for_metadata(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| format!("{arg:?}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_counters_do_not_template_collapse() {
        let raw = b"1\n2\n3\n";
        let groups = collapse_groups(&split_lines(raw));
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn stable_digit_template_collapses_arithmetically() {
        let raw = b"routine line 1\nroutine line 2\nroutine line 3\n";
        let groups = collapse_groups(&split_lines(raw));
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].count(), 3);
        assert_eq!(groups[0].template, "routine line …");
    }

    #[test]
    fn folded_groups_clipped_from_full_collapse_match_direct_middle_collapse() {
        let mut raw = Vec::new();
        for line in 1..=100 {
            if line <= 25 || line >= 70 {
                writeln!(raw, "routine line {line}").unwrap();
            } else {
                writeln!(raw, "steady state").unwrap();
            }
        }
        let lines = split_lines(&raw);
        let all_groups = collapse_groups(&lines);
        let clipped = folded_middle_groups(&lines, 0, &all_groups);
        let (head_end, tail_start) = folded_middle_bounds(&lines, 0);
        let direct = collapse_groups(&lines[head_end..tail_start]);

        assert_eq!(clipped.len(), direct.len());
        for (clipped, direct) in clipped.iter().zip(&direct) {
            assert_eq!(clipped.start, direct.start);
            assert_eq!(clipped.end, direct.end);
            assert_eq!(clipped.representative, direct.representative);
            assert_eq!(clipped.template, direct.template);
        }
    }

    #[test]
    fn split_lines_preserves_exact_line_bytes() {
        let raw = b"a\r\nb\nlast";
        let lines = split_lines(raw);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].content, b"a");
        assert_eq!(lines[0].raw, b"a\r\n");
        assert_eq!(lines[2].raw, b"last");
    }

    #[test]
    fn carriage_return_rewrites_are_one_visible_line() {
        let raw = b"building 10%\rbuilding 90%\r\ndone\n";
        let lines = split_lines(raw);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].content, b"building 90%");
        assert_eq!(lines[0].raw, b"building 10%\rbuilding 90%\r\n");
    }

    #[test]
    fn verdict_ok_without_diagnostics_omits_zero_counts() {
        assert_eq!(verdict_line(0, 0, 0, None), "ok — exit 0");
    }

    #[test]
    fn verdict_ok_with_warnings_prints_warning_count() {
        assert_eq!(verdict_line(0, 0, 3, None), "ok — exit 0, 3 warnings");
    }

    #[test]
    fn verdict_ok_with_error_blocks_is_still_ok() {
        let stdout = split_lines(b"error: marked despite success\n");
        let blocks = detect_blocks(&stdout, &[]);
        let errors = blocks
            .iter()
            .filter(|block| block.kind == BlockKind::Error)
            .count();
        assert_eq!(
            verdict_line(0, errors, 3, None),
            "ok — exit 0, 1 error, 3 warnings"
        );
    }

    #[test]
    fn verdict_failed_always_prints_both_counts() {
        assert_eq!(
            verdict_line(101, 2, 1, None),
            "FAILED — exit 101: 2 errors, 1 warning"
        );
        assert_eq!(
            verdict_line(1, 0, 0, None),
            "FAILED — exit 1: 0 errors, 0 warnings"
        );
    }

    #[test]
    fn verdict_signal_is_annotated() {
        assert_eq!(
            verdict_line(130, 0, 0, Some("SIGINT")),
            "FAILED — exit 130: 0 errors, 0 warnings (SIGINT)"
        );
        assert_eq!(
            verdict_line(137, 0, 0, Some("timeout")),
            "FAILED — exit 137: 0 errors, 0 warnings (timeout)"
        );
    }

    #[test]
    fn verdict_uses_singular_and_plural_nouns() {
        assert_eq!(
            verdict_line(2, 1, 1, None),
            "FAILED — exit 2: 1 error, 1 warning"
        );
        assert_eq!(
            verdict_line(2, 2, 2, None),
            "FAILED — exit 2: 2 errors, 2 warnings"
        );
    }

    #[test]
    fn traceback_indented_and_caused_by_details_join_one_block() {
        let stdout = split_lines(
            b"Traceback (most recent call last):\n  File \"main.py\", line 3\n    run()\ncaused by missing fixture\nordinary tail\n",
        );
        let blocks = detect_blocks(&stdout, &[]);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].kind, BlockKind::Error);
        assert_eq!(blocks[0].lines.len(), 4);
        assert_eq!(blocks[0].lines[3].bytes, b"caused by missing fixture");
    }

    #[test]
    fn stderr_origin_alone_does_not_create_a_block() {
        let stderr = split_lines(b"compiler stopped here\n");
        assert!(detect_blocks(&[], &stderr).is_empty());
    }

    #[test]
    fn warning_marker_inside_error_detail_stays_in_error_block() {
        let stdout = split_lines(b"error: outer\n  warning: nested context\nwarning: separate\n");
        let blocks = detect_blocks(&stdout, &[]);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].kind, BlockKind::Error);
        assert_eq!(blocks[0].lines.len(), 2);
        assert_eq!(blocks[1].kind, BlockKind::Warning);
    }

    #[test]
    fn repeated_e_patterns_or_together_and_collapse_consecutive_templates() {
        let stdout = split_lines(
            b"test_toml case 1 passed\ntest_toml case 2 passed\nunrelated\njson fixture passed\n",
        );
        let matchers = compile_matchers(&["test_toml".into(), "json".into()]).unwrap();
        let groups = collect_matches(&stdout, &[], &matchers);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].representative.line, 1);
        assert_eq!(groups[0].count, 2);
        assert_eq!(groups[1].representative.line, 4);
        assert_eq!(groups[1].count, 1);

        let mut rendered = Vec::new();
        write_answer_prefix(&mut rendered, &stdout, &[], &[], &groups, "ok — exit 0");
        assert_eq!(
            rendered,
            b"ok \xe2\x80\x94 exit 0\n1  test_toml case 1 passed (2 matches)\n4  json fixture passed\n"
        );
    }

    #[test]
    fn drainer_spools_incrementally_and_records_line_timestamps() {
        let dir = std::env::temp_dir().join(format!("greppy-bash-smart-test-{}", spool_token()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stdout");
        let times = dir.join("stdout.times");
        let capture = join_drain(
            spawn_drain(
                std::io::Cursor::new(b"one\ntwo\nlast".to_vec()),
                path.clone(),
                times.clone(),
            ),
            "test",
        )
        .unwrap();
        assert_eq!(capture.byte_len, 12);
        assert_eq!(capture.line_count, 3);
        assert_eq!(std::fs::read(path).unwrap(), b"one\ntwo\nlast");
        assert_eq!(std::fs::read_to_string(times).unwrap().lines().count(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn timeout_gap_names_the_line_before_the_longest_pause() {
        let dir = std::env::temp_dir().join(format!("greppy-bash-smart-gap-{}", spool_token()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("times");
        std::fs::write(&path, "10\n20\n900\n910\n").unwrap();
        assert_eq!(line_before_largest_gap(&path, 1_000), Some(2));
        std::fs::write(&path, "10\n20\n30\n").unwrap();
        assert_eq!(line_before_largest_gap(&path, 1_000), Some(3));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn lifted_lines_hold_content_bytes_for_the_byte_gate() {
        let raw = b"one\ntwo\n";
        let lines = split_lines(raw);
        let mut lifts = Vec::new();
        push_line_lift(&mut lifts, &lines, Some(2));
        lifts.push(LiftedLine {
            line: 1,
            bytes: b"invented".to_vec(),
        });
        let gated = gate_lifts(&lines, &lifts);
        assert_eq!(gated.len(), 1);
        assert_eq!(gated[0].line, 2);
        assert_eq!(gated[0].bytes, b"two");
    }

    #[test]
    fn child_signal_maps_to_shell_exit_code() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt as _;
            let status = std::process::ExitStatus::from_raw(9);
            assert_eq!(child_exit_code(&status), 137);
        }
    }
}
