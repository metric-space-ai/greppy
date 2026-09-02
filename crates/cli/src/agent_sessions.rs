//! Read-only CLI for persisted `greppy agent` session logs.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::Subcommand;
use greppy_core::error::{Error, Result};
use serde_json::{json, Value};

use crate::agent_tui::{
    list_session_project_dirs, load_path, read_session_log_lines, SessionLogLine, SessionRecord,
    SessionStore,
};

const TOOL_RESULT_PREVIEW: usize = 400;
const TAIL_TEXT_PREVIEW: usize = 200;
const FOLLOW_POLL: Duration = Duration::from_millis(200);

static FOLLOW_STOP: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Subcommand)]
pub enum AgentSessionsCommand {
    /// List sessions for this project, newest first.
    List {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Include sessions from every project under the data root.
        #[arg(long)]
        all_projects: bool,
    },
    /// Show one session header and transcript.
    Show {
        /// Session id or unique prefix.
        id: String,
        /// Emit the session record as JSON, including messages and turn/tool events.
        #[arg(long)]
        json: bool,
        /// Do not truncate tool_result parts.
        #[arg(long)]
        full: bool,
    },
    /// Print the last lines of a session log.
    Tail {
        /// Session id or unique prefix.
        id: String,
        /// Print raw JSONL lines instead of the human rendering.
        #[arg(long)]
        json: bool,
        /// Keep polling for newly appended lines until SIGINT (exit 0).
        #[arg(long)]
        follow: bool,
        /// Number of lines to print (default 40).
        #[arg(long, default_value_t = 40, value_name = "N")]
        lines: usize,
    },
    /// Print the absolute JSONL path for a session.
    Path {
        /// Session id or unique prefix.
        id: String,
    },
}

pub fn run(command: AgentSessionsCommand, root: Option<&str>) -> Result<i32> {
    let repo_root = match root {
        Some(path) => PathBuf::from(path),
        None => std::env::current_dir()
            .map_err(|error| Error::Invalid(format!("cannot resolve cwd: {error}")))?,
    };
    let (data_root, logical_project) = crate::agent::agent_session_store_identity(&repo_root);
    match command {
        AgentSessionsCommand::List { json, all_projects } => {
            list_sessions(&data_root, &logical_project, json, all_projects)
        }
        AgentSessionsCommand::Show { id, json, full } => {
            show_session(&data_root, &logical_project, &id, json, full)
        }
        AgentSessionsCommand::Tail {
            id,
            json,
            follow,
            lines,
        } => tail_session(&data_root, &logical_project, &id, json, follow, lines),
        AgentSessionsCommand::Path { id } => path_session(&data_root, &logical_project, &id),
    }
}

#[derive(Clone)]
pub(crate) struct ListedSession {
    pub(crate) record: SessionRecord,
    source: String,
    pub(crate) path: PathBuf,
    lines: Vec<SessionLogLine>,
}

pub(crate) fn resolve_from_root(
    root: Option<&str>,
    id: &str,
) -> std::result::Result<ListedSession, i32> {
    let repo_root = match root {
        Some(path) => PathBuf::from(path),
        None => match std::env::current_dir() {
            Ok(path) => path,
            Err(error) => {
                eprintln!("greppy: cannot resolve cwd: {error}");
                return Err(2);
            }
        },
    };
    let (data_root, logical_project) = crate::agent::agent_session_store_identity(&repo_root);
    resolve_session(&data_root, &logical_project, id)
}

pub(crate) fn print_session_tail(session: &ListedSession, json: bool, lines: usize) {
    if json {
        let start = session.lines.len().saturating_sub(lines);
        for line in &session.lines[start..] {
            println!("{}", line.raw);
        }
        let _ = std::io::stdout().flush();
        return;
    }
    let rendered: Vec<String> = session
        .lines
        .iter()
        .filter_map(|line| line.value.as_ref().and_then(render_tail_line))
        .collect();
    let start = rendered.len().saturating_sub(lines);
    for line in &rendered[start..] {
        println!("{line}");
    }
    let _ = std::io::stdout().flush();
}

fn list_sessions(
    data_root: &Path,
    logical_project: &str,
    json: bool,
    all_projects: bool,
) -> Result<i32> {
    let mut sessions = if all_projects {
        load_all_projects(data_root)?
    } else {
        load_project(data_root, logical_project)?
    };
    sessions.sort_by_key(|session| std::cmp::Reverse(session.record.created_ms));
    if sessions.is_empty() {
        if json {
            println!("[]");
        } else {
            eprintln!("no sessions for project {logical_project}");
        }
        return Ok(0);
    }
    if json {
        let rows: Vec<Value> = sessions.iter().map(list_json_row).collect();
        println!(
            "{}",
            serde_json::to_string(&rows)
                .map_err(|error| Error::Invalid(format!("serialize sessions: {error}")))?
        );
        return Ok(0);
    }
    println!("ID  CREATED  LIVE  TURNS  STOP  MODEL  TITLE  PROPOSAL");
    for session in sessions {
        let proposal = if session.record.proposal_ref.is_empty() {
            "-"
        } else {
            session.record.proposal_ref.as_str()
        };
        let stop = if session.record.stop.is_empty() {
            "-"
        } else {
            session.record.stop.as_str()
        };
        let (live, _) = live_socket(&session);
        println!(
            "{}  {}  {}  {}  {}  {}  {}  {}",
            session.record.id,
            format_local_iso8601(session.record.created_ms),
            if live { "yes" } else { "no" },
            session.record.turns,
            stop,
            session.record.model,
            session.record.title,
            proposal
        );
    }
    Ok(0)
}

fn show_session(
    data_root: &Path,
    logical_project: &str,
    id: &str,
    json: bool,
    full: bool,
) -> Result<i32> {
    let session = match resolve_session(data_root, logical_project, id) {
        Ok(session) => session,
        Err(code) => return Ok(code),
    };
    if json {
        println!(
            "{}",
            serde_json::to_string(&show_json(&session))
                .map_err(|error| Error::Invalid(format!("serialize session: {error}")))?
        );
        return Ok(0);
    }
    print_show_header(&session);
    print_show_transcript(&session, full);
    Ok(0)
}

fn tail_session(
    data_root: &Path,
    logical_project: &str,
    id: &str,
    json: bool,
    follow: bool,
    lines: usize,
) -> Result<i32> {
    let session = match resolve_session(data_root, logical_project, id) {
        Ok(session) => session,
        Err(code) => return Ok(code),
    };
    print_session_tail(&session, json, lines);
    if !follow {
        return Ok(0);
    }
    let offset =
        complete_file_offset(&session.path).map_err(|error| Error::io("session log", error))?;
    follow_session(&session.path, json, offset)
}

fn path_session(data_root: &Path, logical_project: &str, id: &str) -> Result<i32> {
    let session = match resolve_session(data_root, logical_project, id) {
        Ok(session) => session,
        Err(code) => return Ok(code),
    };
    println!("{}", session.path.display());
    Ok(0)
}

pub(crate) fn resolve_session(
    data_root: &Path,
    logical_project: &str,
    id: &str,
) -> std::result::Result<ListedSession, i32> {
    let sessions = match load_project(data_root, logical_project) {
        Ok(sessions) => sessions,
        Err(error) => {
            eprintln!("greppy: {error}");
            return Err(2);
        }
    };
    if let Some(exact) = sessions.iter().find(|session| session.record.id == id) {
        return Ok(exact.clone());
    }
    let matches: Vec<&ListedSession> = sessions
        .iter()
        .filter(|session| session.record.id.starts_with(id))
        .collect();
    match matches.as_slice() {
        [unique] => Ok((*unique).clone()),
        [] => {
            eprintln!("no session {id} in project {logical_project}");
            Err(2)
        }
        many => {
            eprintln!("ambiguous session prefix {id}:");
            for session in many {
                eprintln!("  {}", session.record.id);
            }
            Err(2)
        }
    }
}

fn load_project(data_root: &Path, project: &str) -> Result<Vec<ListedSession>> {
    let store = SessionStore::new(data_root, project);
    load_dir(&store.project_dir())
}

fn load_all_projects(data_root: &Path) -> Result<Vec<ListedSession>> {
    let mut sessions = Vec::new();
    for dir in
        list_session_project_dirs(data_root).map_err(|error| Error::io("agent-sessions", error))?
    {
        sessions.extend(load_dir(&dir)?);
    }
    Ok(sessions)
}

fn load_dir(dir: &Path) -> Result<Vec<ListedSession>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut sessions = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|error| Error::io("session dir", error))? {
        let entry = entry.map_err(|error| Error::io("session dir", error))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
            continue;
        }
        match inspect_session(&path) {
            Ok(session) => sessions.push(session),
            Err(_) => continue,
        }
    }
    Ok(sessions)
}

fn inspect_session(path: &Path) -> io::Result<ListedSession> {
    let record = load_path(path)?;
    let lines = read_session_log_lines(path)?;
    let source = source_from_lines(&lines);
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Ok(ListedSession {
        record,
        source,
        path,
        lines,
    })
}

fn source_from_lines(lines: &[SessionLogLine]) -> String {
    for line in lines {
        let Some(value) = &line.value else {
            continue;
        };
        if value.get("type").and_then(Value::as_str) == Some("meta") {
            return value
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
    }
    String::new()
}

fn live_socket(session: &ListedSession) -> (bool, Option<PathBuf>) {
    let Some(project_dir) = session.path.parent() else {
        return (false, None);
    };
    let Some(data_root) = project_dir.parent().and_then(Path::parent) else {
        return (false, None);
    };
    let Some(project) = project_dir.file_name().and_then(|name| name.to_str()) else {
        return (false, None);
    };
    let store = SessionStore::new(data_root, project);
    let socket = crate::agent_control::socket_path_for(&store, &session.record.id);
    #[cfg(unix)]
    let live = crate::agent_control::is_live(&socket);
    #[cfg(not(unix))]
    let live = false;
    (live, Some(socket))
}

fn list_json_row(session: &ListedSession) -> Value {
    let (live, socket) = live_socket(session);
    json!({
        "id": session.record.id,
        "project": session.record.project,
        "title": session.record.title,
        "model": session.record.model,
        "created_ms": session.record.created_ms,
        "run_id": session.record.run_id,
        "worktree": session.record.worktree,
        "branch": session.record.branch,
        "proposal_ref": session.record.proposal_ref,
        "turns": session.record.turns,
        "stop": session.record.stop,
        "source": session.source,
        "recovered": session.record.recovered,
        "path": session.path,
        "live": live,
        "socket": socket,
    })
}

fn show_json(session: &ListedSession) -> Value {
    let messages: Vec<Value> = session
        .record
        .messages
        .iter()
        .map(|message| {
            json!({
                "role": message.role,
                "parts": message.parts.iter().map(|part| json!({
                    "kind": part.kind,
                    "text": part.text,
                    "id": part.id,
                    "name": part.name,
                    "is_error": part.is_error,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let events: Vec<Value> = session
        .lines
        .iter()
        .filter_map(|line| {
            let value = line.value.as_ref()?;
            match value.get("type").and_then(Value::as_str) {
                Some("turn") | Some("tool") => Some(value.clone()),
                _ => None,
            }
        })
        .collect();
    json!({
        "id": session.record.id,
        "project": session.record.project,
        "title": session.record.title,
        "model": session.record.model,
        "created_ms": session.record.created_ms,
        "run_id": session.record.run_id,
        "worktree": session.record.worktree,
        "branch": session.record.branch,
        "proposal_ref": session.record.proposal_ref,
        "turns": session.record.turns,
        "stop": session.record.stop,
        "source": session.source,
        "recovered": session.record.recovered,
        "usage": {
            "input": session.record.usage.input_tokens,
            "output": session.record.usage.output_tokens,
            "cache_read": session.record.usage.cache_read_input_tokens,
            "cache_write": session.record.usage.cache_creation_input_tokens,
        },
        "messages": messages,
        "events": events,
    })
}

fn print_show_header(session: &ListedSession) {
    let record = &session.record;
    println!("id: {}", record.id);
    println!("title: {}", record.title);
    println!("project: {}", record.project);
    println!("model: {}", record.model);
    println!("created: {}", format_local_iso8601(record.created_ms));
    println!("run_id: {}", record.run_id);
    println!("worktree: {}", record.worktree);
    println!("branch: {}", record.branch);
    println!("proposal_ref: {}", record.proposal_ref);
    println!("turns: {}", record.turns);
    println!("stop: {}", record.stop);
    println!(
        "usage: in={} out={} cache_read={} cache_write={}",
        record.usage.input_tokens,
        record.usage.output_tokens,
        record.usage.cache_read_input_tokens,
        record.usage.cache_creation_input_tokens
    );
    println!();
}

fn print_show_transcript(session: &ListedSession, full: bool) {
    for line in &session.lines {
        let Some(value) = &line.value else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("message") => print_show_message(value, full),
            Some("tool") => {
                if let Some(rendered) = render_show_tool(value) {
                    println!("{rendered}");
                }
            }
            _ => {}
        }
    }
}

fn print_show_message(value: &Value, full: bool) {
    let role = value
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("message");
    let Some(parts) = value.get("parts").and_then(Value::as_array) else {
        return;
    };
    for part in parts {
        let kind = part.get("kind").and_then(Value::as_str).unwrap_or("");
        let mut text = part
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        match kind {
            "thinking" => continue,
            "tool_result" if !full => text = truncate_chars(&text, TOOL_RESULT_PREVIEW),
            "text" | "tool_result" => {}
            _ => continue,
        }
        if text.is_empty() {
            continue;
        }
        println!("{role}: {text}");
    }
}

fn render_show_tool(value: &Value) -> Option<String> {
    match value.get("event").and_then(Value::as_str)? {
        "start" => {
            let name = value.get("name").and_then(Value::as_str).unwrap_or("tool");
            let summary = value.get("summary").and_then(Value::as_str).unwrap_or("");
            Some(format!("tool ▶ {name} {summary}"))
        }
        "finish" => {
            let elapsed = value.get("elapsed_ms").and_then(Value::as_u64).unwrap_or(0);
            let mark = if value
                .get("failed")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                "✗"
            } else {
                "✓"
            };
            Some(format!("tool {mark} {elapsed} ms"))
        }
        _ => None,
    }
}

pub(crate) fn render_tail_line(value: &Value) -> Option<String> {
    match value.get("type").and_then(Value::as_str)? {
        "message" => {
            let role = value
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("message");
            let preview = message_text_preview(value, TAIL_TEXT_PREVIEW);
            Some(format!("{role}: {preview}"))
        }
        "tool" => match value.get("event").and_then(Value::as_str) {
            Some("start") => {
                let summary = value.get("summary").and_then(Value::as_str).unwrap_or("");
                Some(format!("▶ {summary}"))
            }
            Some("finish") => {
                let elapsed = value.get("elapsed_ms").and_then(Value::as_u64).unwrap_or(0);
                if value
                    .get("failed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let preview = value.get("preview").and_then(Value::as_str).unwrap_or("");
                    Some(format!("✗ {elapsed} ms {preview}"))
                } else {
                    Some(format!("✓ {elapsed} ms"))
                }
            }
            _ => None,
        },
        "turn" => match value.get("event").and_then(Value::as_str) {
            Some("start") => {
                let source = value.get("source").and_then(Value::as_str).unwrap_or("");
                let prompt = value.get("prompt").and_then(Value::as_str).unwrap_or("");
                Some(format!("turn start ({source}): {prompt}"))
            }
            Some("done") => {
                let stop = value.get("stop").and_then(Value::as_str).unwrap_or("");
                let turns = value.get("turns").and_then(Value::as_u64).unwrap_or(0);
                let usage = value.get("usage");
                let input = usage
                    .and_then(|value| value.get("input"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let output = usage
                    .and_then(|value| value.get("output"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                Some(format!(
                    "turn done stop={stop} turns={turns} in={input} out={output}"
                ))
            }
            Some("error") => {
                let message = value.get("message").and_then(Value::as_str).unwrap_or("");
                Some(format!("turn error: {message}"))
            }
            _ => None,
        },
        "usage" => {
            let input = value.get("input").and_then(Value::as_u64).unwrap_or(0);
            let output = value.get("output").and_then(Value::as_u64).unwrap_or(0);
            let turns = value.get("turns").and_then(Value::as_u64).unwrap_or(0);
            let stop = value.get("stop").and_then(Value::as_str).unwrap_or("");
            Some(format!(
                "usage in={input} out={output} turns={turns} stop={stop}"
            ))
        }
        "title" => Some(format!(
            "title: {}",
            value.get("title").and_then(Value::as_str).unwrap_or("")
        )),
        "model" => Some(format!(
            "model: {}",
            value.get("model").and_then(Value::as_str).unwrap_or("")
        )),
        "worktree" => {
            let path = value.get("path").and_then(Value::as_str).unwrap_or("");
            let proposal = value
                .get("proposal_ref")
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(format!("worktree: {path} {proposal}"))
        }
        "meta" => {
            let id = value.get("id").and_then(Value::as_str).unwrap_or("");
            let project = value.get("project").and_then(Value::as_str).unwrap_or("");
            Some(format!("meta: {id} {project}"))
        }
        _ => None,
    }
}

fn message_text_preview(value: &Value, max: usize) -> String {
    let mut text = String::new();
    if let Some(parts) = value.get("parts").and_then(Value::as_array) {
        for part in parts {
            if part.get("kind").and_then(Value::as_str) != Some("text") {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
        }
    }
    truncate_chars(&text, max)
}

fn truncate_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

fn complete_file_offset(path: &Path) -> io::Result<u64> {
    let data = std::fs::read(path)?;
    Ok(match data.iter().rposition(|&byte| byte == b'\n') {
        Some(index) => (index + 1) as u64,
        None => 0,
    })
}

fn follow_session(path: &Path, json: bool, mut offset: u64) -> Result<i32> {
    install_follow_stop();
    loop {
        if FOLLOW_STOP.load(Ordering::Relaxed) {
            return Ok(0);
        }
        let lines = read_new_complete_lines(path, &mut offset)
            .map_err(|error| Error::io("session follow", error))?;
        for raw in lines {
            if json {
                println!("{raw}");
            } else if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                if let Some(rendered) = render_tail_line(&value) {
                    println!("{rendered}");
                }
            }
            let _ = std::io::stdout().flush();
        }
        if FOLLOW_STOP.load(Ordering::Relaxed) {
            return Ok(0);
        }
        std::thread::sleep(FOLLOW_POLL);
    }
}

fn read_new_complete_lines(path: &Path, offset: &mut u64) -> io::Result<Vec<String>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len < *offset {
        *offset = 0;
    }
    file.seek(SeekFrom::Start(*offset))?;
    let mut reader = BufReader::new(file);
    let mut lines = Vec::new();
    loop {
        let mut buf = String::new();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        if !buf.ends_with('\n') {
            break;
        }
        *offset += n as u64;
        if buf.ends_with("\r\n") {
            buf.truncate(buf.len() - 2);
        } else {
            buf.pop();
        }
        if !buf.trim().is_empty() {
            lines.push(buf);
        }
    }
    Ok(lines)
}

fn install_follow_stop() {
    FOLLOW_STOP.store(false, Ordering::Relaxed);
    #[cfg(unix)]
    {
        let handler = follow_sigint as *const () as libc::sighandler_t;
        unsafe {
            libc::signal(libc::SIGINT, handler);
        }
    }
}

#[cfg(unix)]
extern "C" fn follow_sigint(_: libc::c_int) {
    FOLLOW_STOP.store(true, Ordering::Relaxed);
}

fn format_local_iso8601(ms: u64) -> String {
    #[cfg(unix)]
    {
        if let Some(formatted) = format_unix_local(ms) {
            return formatted;
        }
    }
    format_utc(ms)
}

#[cfg(unix)]
fn format_unix_local(ms: u64) -> Option<String> {
    let time = (ms / 1000) as libc::time_t;
    unsafe {
        let mut tm = std::mem::zeroed();
        if libc::localtime_r(&time, &mut tm).is_null() {
            return None;
        }
        let mut buf = [0u8; 64];
        let written = libc::strftime(
            buf.as_mut_ptr().cast(),
            buf.len(),
            c"%Y-%m-%dT%H:%M:%S%z".as_ptr(),
            &tm,
        );
        if written == 0 {
            return None;
        }
        let formatted = std::str::from_utf8(&buf[..written as usize]).ok()?;
        Some(insert_offset_colon(formatted))
    }
}

fn insert_offset_colon(formatted: &str) -> String {
    let n = formatted.len();
    if n >= 5 {
        let sign = formatted.as_bytes()[n - 5];
        if sign == b'+' || sign == b'-' {
            return format!("{}:{}", &formatted[..n - 2], &formatted[n - 2..]);
        }
    }
    formatted.to_string()
}

fn format_utc(ms: u64) -> String {
    let secs = ms / 1000;
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32)
}
