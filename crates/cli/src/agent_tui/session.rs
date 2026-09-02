//! Versioned, append-safe interactive session persistence.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use greppy_agent::{ContentPart, Message, Role, Usage};
use serde_json::{json, Value};

use super::redaction::{redact_json, redact_text};

pub const SESSION_FORMAT: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedMessage {
    pub role: String,
    pub parts: Vec<PersistedPart>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedPart {
    pub kind: String,
    pub text: String,
    pub id: String,
    pub name: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionRecord {
    pub id: String,
    pub project: String,
    pub title: String,
    pub model: String,
    pub created_ms: u64,
    pub run_id: String,
    pub worktree: String,
    pub branch: String,
    pub proposal_ref: String,
    pub source: String,
    pub messages: Vec<PersistedMessage>,
    pub usage: Usage,
    pub turns: u64,
    pub stop: String,
    pub recovered: bool,
}

impl SessionRecord {
    pub fn new(id: String, project: String, model: String, run_id: String) -> Self {
        Self {
            id,
            project,
            title: "untitled".to_string(),
            model,
            created_ms: now_ms(),
            run_id,
            worktree: String::new(),
            branch: String::new(),
            proposal_ref: String::new(),
            source: String::new(),
            messages: Vec::new(),
            usage: Usage::default(),
            turns: 0,
            stop: String::new(),
            recovered: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
    project: String,
}

impl SessionStore {
    pub fn new(data_root: impl Into<PathBuf>, project: impl Into<String>) -> Self {
        Self {
            root: data_root.into().join("agent-sessions"),
            project: sanitize_project(&project.into()),
        }
    }

    pub fn project_dir(&self) -> PathBuf {
        self.root.join(&self.project)
    }

    pub fn path_for(&self, session_id: &str) -> PathBuf {
        self.project_dir().join(format!("{session_id}.jsonl"))
    }

    pub fn create(&self, record: &SessionRecord) -> io::Result<PathBuf> {
        fs::create_dir_all(self.project_dir())?;
        let path = self.path_for(&record.id);
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        writeln!(file, "{}", meta_line(record))?;
        file.flush()?;
        Ok(path)
    }

    pub fn append(&self, session_id: &str, line: &Value) -> io::Result<()> {
        let path = self.path_for(session_id);
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{line}")?;
        file.flush()?;
        Ok(())
    }

    pub fn append_messages(
        &self,
        session_id: &str,
        messages: &[PersistedMessage],
    ) -> io::Result<()> {
        for message in messages {
            self.append(session_id, &message_line(message))?;
        }
        Ok(())
    }

    /// Appends an authoritative message checkpoint without rewriting the JSONL file.
    /// Loading a checkpoint discards all earlier message entries while retaining
    /// later append-only updates and the valid-prefix recovery contract.
    pub fn append_message_checkpoint(
        &self,
        session_id: &str,
        messages: &[PersistedMessage],
    ) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "message_checkpoint",
                "messages": messages.iter().map(message_line).collect::<Vec<_>>(),
            }),
        )
    }

    pub fn append_usage(
        &self,
        session_id: &str,
        usage: &Usage,
        turns: u64,
        stop: &str,
    ) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "usage",
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_read": usage.cache_read_input_tokens,
                "cache_write": usage.cache_creation_input_tokens,
                "turns": turns,
                "stop": stop,
            }),
        )
    }

    pub fn set_title(&self, session_id: &str, title: &str) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "title",
                "title": redact_text(title),
            }),
        )
    }

    pub fn set_model(&self, session_id: &str, model: &str) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "model",
                "model": model,
            }),
        )
    }

    pub fn append_worktree(
        &self,
        session_id: &str,
        path: &str,
        proposal_ref: &str,
    ) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "worktree",
                "path": path,
                "proposal_ref": proposal_ref,
            }),
        )
    }

    pub fn append_turn_start(
        &self,
        session_id: &str,
        source: &str,
        prompt: &str,
    ) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "turn",
                "event": "start",
                "ts_ms": now_ms(),
                "source": source,
                "prompt": redact_text(prompt),
            }),
        )
    }

    pub fn append_tool_start(
        &self,
        session_id: &str,
        id: &str,
        name: &str,
        summary: &str,
    ) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "tool",
                "event": "start",
                "ts_ms": now_ms(),
                "id": id,
                "name": name,
                "summary": summary,
            }),
        )
    }

    pub fn append_tool_finish(
        &self,
        session_id: &str,
        id: &str,
        failed: bool,
        elapsed_ms: u64,
        preview: &str,
    ) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "tool",
                "event": "finish",
                "ts_ms": now_ms(),
                "id": id,
                "failed": failed,
                "elapsed_ms": elapsed_ms,
                "preview": clip_chars(&redact_text(preview), 400),
            }),
        )
    }

    pub fn append_turn_done(
        &self,
        session_id: &str,
        stop: &str,
        turns: u64,
        usage: &Usage,
    ) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "turn",
                "event": "done",
                "ts_ms": now_ms(),
                "stop": stop,
                "turns": turns,
                "usage": usage_object(usage),
            }),
        )
    }

    pub fn append_turn_error(&self, session_id: &str, message: &str) -> io::Result<()> {
        self.append(
            session_id,
            &json!({
                "v": SESSION_FORMAT,
                "type": "turn",
                "event": "error",
                "ts_ms": now_ms(),
                "message": message,
            }),
        )
    }

    pub fn load(&self, session_id: &str) -> io::Result<SessionRecord> {
        load_path(&self.path_for(session_id))
    }

    pub fn latest(&self) -> io::Result<Option<SessionRecord>> {
        let mut best: Option<(SystemTime, SessionRecord)> = None;
        for record in self.list()? {
            let modified = fs::metadata(self.path_for(&record.id))
                .and_then(|meta| meta.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            match &best {
                Some((time, _)) if *time >= modified => {}
                _ => best = Some((modified, record)),
            }
        }
        Ok(best.map(|(_, record)| record))
    }

    pub fn list(&self) -> io::Result<Vec<SessionRecord>> {
        let dir = self.project_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut records = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }
            match load_path(&path) {
                Ok(record) => records.push(record),
                Err(_) => continue,
            }
        }
        records.sort_by_key(|record| std::cmp::Reverse(record.created_ms));
        Ok(records)
    }
}

pub fn load_path(path: &Path) -> io::Result<SessionRecord> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut record = SessionRecord::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("session")
            .to_string(),
        String::new(),
        String::new(),
        String::new(),
    );
    let mut recovered = false;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            recovered = true;
            break;
        };
        if value.get("v").and_then(Value::as_u64) != Some(SESSION_FORMAT as u64) {
            recovered = true;
            break;
        }
        match value.get("type").and_then(Value::as_str) {
            Some("meta") => apply_meta(&mut record, &value),
            Some("message") => {
                if let Some(message) = message_from_value(&value) {
                    record.messages.push(message);
                }
            }
            Some("message_checkpoint") => {
                let Some(messages) = value.get("messages").and_then(Value::as_array) else {
                    recovered = true;
                    break;
                };
                let replacement = messages
                    .iter()
                    .map(message_from_value)
                    .collect::<Option<Vec<_>>>();
                let Some(replacement) = replacement else {
                    recovered = true;
                    break;
                };
                record.messages = replacement;
            }
            Some("usage") => apply_usage(&mut record, &value),
            Some("title") => {
                if let Some(title) = value.get("title").and_then(Value::as_str) {
                    record.title = title.to_string();
                }
            }
            Some("model") => {
                if let Some(model) = value.get("model").and_then(Value::as_str) {
                    record.model = model.to_string();
                }
            }
            Some("worktree") => {
                if let Some(path) = value.get("path").and_then(Value::as_str) {
                    record.worktree = path.to_string();
                }
                if let Some(reference) = value.get("proposal_ref").and_then(Value::as_str) {
                    record.proposal_ref = reference.to_string();
                }
            }
            Some("tool") | Some("turn") => {}
            _ => {}
        }
    }
    record.recovered = recovered;
    Ok(record)
}

pub fn messages_from_protocol(messages: &[Message]) -> Vec<PersistedMessage> {
    messages.iter().map(persist_message).collect()
}

pub fn protocol_from_persisted(messages: &[PersistedMessage]) -> Vec<Message> {
    messages.iter().map(to_protocol).collect()
}

pub fn new_session_id() -> String {
    let ms = now_ms();
    format!("sess-{ms}-{}", std::process::id())
}

pub fn compact_messages(messages: &[PersistedMessage], keep: usize) -> Vec<PersistedMessage> {
    if messages.len() <= keep {
        return messages.to_vec();
    }
    let (old, recent) = messages.split_at(messages.len() - keep);
    let mut summary = String::from("Earlier conversation summary:\n");
    for message in old {
        let preview: String = message
            .parts
            .iter()
            .map(|part| part.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(240)
            .collect();
        if !preview.is_empty() {
            summary.push_str("- ");
            summary.push_str(&message.role);
            summary.push_str(": ");
            summary.push_str(&preview);
            summary.push('\n');
        }
    }
    let mut out = vec![PersistedMessage {
        role: "user".to_string(),
        parts: vec![PersistedPart {
            kind: "text".to_string(),
            text: summary,
            id: String::new(),
            name: String::new(),
            is_error: false,
        }],
    }];
    out.extend(recent.iter().cloned());
    out
}

fn persist_message(message: &Message) -> PersistedMessage {
    let role = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    PersistedMessage {
        role: role.to_string(),
        parts: message.content.iter().map(persist_part).collect(),
    }
}

fn persist_part(part: &ContentPart) -> PersistedPart {
    match part {
        ContentPart::Text { text } => PersistedPart {
            kind: "text".to_string(),
            text: redact_text(text),
            id: String::new(),
            name: String::new(),
            is_error: false,
        },
        ContentPart::Thinking { text } => PersistedPart {
            kind: "thinking".to_string(),
            text: redact_text(text),
            id: String::new(),
            name: String::new(),
            is_error: false,
        },
        ContentPart::ToolCall {
            id,
            name,
            arguments,
        } => PersistedPart {
            kind: "tool_call".to_string(),
            text: redact_json(arguments).to_string(),
            id: id.clone(),
            name: name.clone(),
            is_error: false,
        },
        ContentPart::ToolResult {
            call_id,
            content,
            is_error,
        } => PersistedPart {
            kind: "tool_result".to_string(),
            text: redact_text(content),
            id: call_id.clone(),
            name: String::new(),
            is_error: *is_error,
        },
        ContentPart::Image { media_type, .. } => PersistedPart {
            kind: "image".to_string(),
            text: format!("[{media_type} omitted from session log]"),
            id: String::new(),
            name: media_type.clone(),
            is_error: false,
        },
    }
}

fn to_protocol(message: &PersistedMessage) -> Message {
    let role = if message.role == "assistant" {
        Role::Assistant
    } else {
        Role::User
    };
    Message {
        role,
        content: message.parts.iter().map(to_part).collect(),
    }
}

fn to_part(part: &PersistedPart) -> ContentPart {
    match part.kind.as_str() {
        "thinking" => ContentPart::Thinking {
            text: part.text.clone(),
        },
        "tool_call" => ContentPart::ToolCall {
            id: part.id.clone(),
            name: part.name.clone(),
            arguments: serde_json::from_str(&part.text).unwrap_or(Value::Null),
        },
        "tool_result" => ContentPart::ToolResult {
            call_id: part.id.clone(),
            content: part.text.clone(),
            is_error: part.is_error,
        },
        _ => ContentPart::Text {
            text: part.text.clone(),
        },
    }
}

fn meta_line(record: &SessionRecord) -> Value {
    json!({
        "v": SESSION_FORMAT,
        "type": "meta",
        "id": record.id,
        "project": record.project,
        "title": redact_text(&record.title),
        "model": record.model,
        "created_ms": record.created_ms,
        "run_id": record.run_id,
        "worktree": record.worktree,
        "branch": record.branch,
        "proposal_ref": record.proposal_ref,
        "source": record.source,
    })
}

fn message_line(message: &PersistedMessage) -> Value {
    json!({
        "v": SESSION_FORMAT,
        "type": "message",
        "role": message.role,
        "parts": message.parts.iter().map(|part| json!({
            "kind": part.kind,
            "text": part.text,
            "id": part.id,
            "name": part.name,
            "is_error": part.is_error,
        })).collect::<Vec<_>>(),
    })
}

fn message_from_value(value: &Value) -> Option<PersistedMessage> {
    let role = value.get("role")?.as_str()?.to_string();
    let parts = value
        .get("parts")?
        .as_array()?
        .iter()
        .filter_map(|part| {
            Some(PersistedPart {
                kind: part.get("kind")?.as_str()?.to_string(),
                text: part.get("text")?.as_str().unwrap_or("").to_string(),
                id: part
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                name: part
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                is_error: part
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        })
        .collect();
    Some(PersistedMessage { role, parts })
}

fn apply_meta(record: &mut SessionRecord, value: &Value) {
    if let Some(id) = value.get("id").and_then(Value::as_str) {
        record.id = id.to_string();
    }
    if let Some(project) = value.get("project").and_then(Value::as_str) {
        record.project = project.to_string();
    }
    if let Some(title) = value.get("title").and_then(Value::as_str) {
        record.title = title.to_string();
    }
    if let Some(model) = value.get("model").and_then(Value::as_str) {
        record.model = model.to_string();
    }
    if let Some(created) = value.get("created_ms").and_then(Value::as_u64) {
        record.created_ms = created;
    }
    if let Some(run_id) = value.get("run_id").and_then(Value::as_str) {
        record.run_id = run_id.to_string();
    }
    if let Some(worktree) = value.get("worktree").and_then(Value::as_str) {
        record.worktree = worktree.to_string();
    }
    if let Some(branch) = value.get("branch").and_then(Value::as_str) {
        record.branch = branch.to_string();
    }
    if let Some(reference) = value.get("proposal_ref").and_then(Value::as_str) {
        record.proposal_ref = reference.to_string();
    }
    if let Some(source) = value.get("source").and_then(Value::as_str) {
        record.source = source.to_string();
    }
}

fn usage_object(usage: &Usage) -> Value {
    json!({
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens,
        "cache_write": usage.cache_creation_input_tokens,
    })
}

fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max).collect()
    }
}

fn apply_usage(record: &mut SessionRecord, value: &Value) {
    record.usage.input_tokens = value
        .get("input")
        .and_then(Value::as_u64)
        .unwrap_or(record.usage.input_tokens);
    record.usage.output_tokens = value
        .get("output")
        .and_then(Value::as_u64)
        .unwrap_or(record.usage.output_tokens);
    record.usage.cache_read_input_tokens = value
        .get("cache_read")
        .and_then(Value::as_u64)
        .unwrap_or(record.usage.cache_read_input_tokens);
    record.usage.cache_creation_input_tokens = value
        .get("cache_write")
        .and_then(Value::as_u64)
        .unwrap_or(record.usage.cache_creation_input_tokens);
    record.turns = value
        .get("turns")
        .and_then(Value::as_u64)
        .unwrap_or(record.turns);
    if let Some(stop) = value.get("stop").and_then(Value::as_str) {
        record.stop = stop.to_string();
    }
}

fn sanitize_project(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "project".to_string()
    } else {
        out
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(tag: &str) -> (SessionStore, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "greppy-tui-session-{tag}-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let _ = fs::create_dir_all(&root);
        (SessionStore::new(&root, "demo"), root)
    }

    #[test]
    fn append_and_restore_round_trip() {
        let (store, root) = temp_store("round");
        let mut record = SessionRecord::new(
            "sess-1".into(),
            "demo".into(),
            "test-model".into(),
            "run-1".into(),
        );
        record.title = "first".into();
        store.create(&record).unwrap();
        store
            .append_messages(
                &record.id,
                &[PersistedMessage {
                    role: "user".into(),
                    parts: vec![PersistedPart {
                        kind: "text".into(),
                        text: "hello".into(),
                        id: String::new(),
                        name: String::new(),
                        is_error: false,
                    }],
                }],
            )
            .unwrap();
        store.set_title(&record.id, "renamed").unwrap();
        let loaded = store.load("sess-1").unwrap();
        assert_eq!(loaded.title, "renamed");
        assert_eq!(loaded.messages[0].parts[0].text, "hello");
        assert!(!loaded.recovered);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corrupt_tail_recovers_prefix() {
        let (store, root) = temp_store("corrupt");
        let record = SessionRecord::new("sess-2".into(), "demo".into(), "m".into(), "run".into());
        store.create(&record).unwrap();
        store
            .append_messages(
                &record.id,
                &[PersistedMessage {
                    role: "user".into(),
                    parts: vec![PersistedPart {
                        kind: "text".into(),
                        text: "kept".into(),
                        id: String::new(),
                        name: String::new(),
                        is_error: false,
                    }],
                }],
            )
            .unwrap();
        let path = store.path_for("sess-2");
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(file, "{{not json").unwrap();
        let loaded = store.load("sess-2").unwrap();
        assert!(loaded.recovered);
        assert_eq!(loaded.messages[0].parts[0].text, "kept");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn write_failure_is_io_error() {
        let root = std::env::temp_dir().join(format!(
            "greppy-tui-session-fail-{}-{}",
            std::process::id(),
            now_ms()
        ));
        fs::write(&root, b"not-a-dir").unwrap();
        let store = SessionStore::new(&root, "demo");
        let record = SessionRecord::new("sess".into(), "demo".into(), "m".into(), "r".into());
        assert!(store.create(&record).is_err());
        let _ = fs::remove_file(root);
    }

    #[test]
    fn compact_keeps_recent_and_summary() {
        let messages: Vec<_> = (0..10)
            .map(|i| PersistedMessage {
                role: "user".into(),
                parts: vec![PersistedPart {
                    kind: "text".into(),
                    text: format!("m{i}"),
                    id: String::new(),
                    name: String::new(),
                    is_error: false,
                }],
            })
            .collect();
        let compacted = compact_messages(&messages, 4);
        assert_eq!(compacted.len(), 5);
        assert!(compacted[0].parts[0].text.contains("Earlier conversation"));
        assert_eq!(compacted.last().unwrap().parts[0].text, "m9");
    }

    #[test]
    fn checkpoint_replaces_prior_messages_and_allows_later_appends() {
        let (store, root) = temp_store("checkpoint");
        let record = SessionRecord::new(
            "sess-checkpoint".into(),
            "demo".into(),
            "m".into(),
            "run".into(),
        );
        store.create(&record).unwrap();
        let message = |text: &str| PersistedMessage {
            role: "user".into(),
            parts: vec![PersistedPart {
                kind: "text".into(),
                text: text.into(),
                id: String::new(),
                name: String::new(),
                is_error: false,
            }],
        };
        store
            .append_messages(&record.id, &[message("old-1"), message("old-2")])
            .unwrap();
        store
            .append_message_checkpoint(&record.id, &[message("summary"), message("recent")])
            .unwrap();
        store
            .append_messages(&record.id, &[message("after")])
            .unwrap();

        let loaded = store.load(&record.id).unwrap();
        let texts = loaded
            .messages
            .iter()
            .map(|message| message.parts[0].text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(texts, vec!["summary", "recent", "after"]);
        assert!(!loaded.recovered);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn usage_entries_are_cumulative_snapshots() {
        let (store, root) = temp_store("usage-snapshot");
        let record =
            SessionRecord::new("sess-usage".into(), "demo".into(), "m".into(), "run".into());
        store.create(&record).unwrap();
        let first = Usage {
            input_tokens: 10,
            output_tokens: 4,
            ..Usage::default()
        };
        store
            .append_usage(&record.id, &first, 1, "tool_use")
            .unwrap();
        let cumulative = Usage {
            input_tokens: 25,
            output_tokens: 9,
            cache_read_input_tokens: 3,
            ..Usage::default()
        };
        store
            .append_usage(&record.id, &cumulative, 3, "end_turn")
            .unwrap();

        let loaded = store.load(&record.id).unwrap();
        assert_eq!(loaded.usage.input_tokens, 25);
        assert_eq!(loaded.usage.output_tokens, 9);
        assert_eq!(loaded.usage.cache_read_input_tokens, 3);
        assert_eq!(loaded.turns, 3);
        assert_eq!(loaded.stop, "end_turn");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn source_defaults_empty_on_legacy_meta() {
        let (store, root) = temp_store("legacy-source");
        let path = store.path_for("sess-legacy");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            r#"{"v":1,"type":"meta","id":"sess-legacy","project":"demo","title":"untitled","model":"m","created_ms":1,"run_id":"run","worktree":"","branch":"","proposal_ref":""}
{"v":1,"type":"message","role":"user","parts":[{"kind":"text","text":"hi","id":"","name":"","is_error":false}]}
{"v":1,"type":"usage","input":4,"output":2,"cache_read":0,"cache_write":0,"turns":1,"stop":"ready"}
"#,
        )
        .unwrap();
        let loaded = store.load("sess-legacy").unwrap();
        assert_eq!(loaded.source, "");
        assert_eq!(loaded.messages[0].parts[0].text, "hi");
        assert_eq!(loaded.usage.input_tokens, 4);
        assert!(!loaded.recovered);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn tool_and_turn_event_lines_do_not_affect_messages_or_usage() {
        let (store, root) = temp_store("events");
        let mut record = SessionRecord::new(
            "sess-events".into(),
            "demo".into(),
            "m".into(),
            "run".into(),
        );
        record.source = "interactive".into();
        store.create(&record).unwrap();
        let message = PersistedMessage {
            role: "user".into(),
            parts: vec![PersistedPart {
                kind: "text".into(),
                text: "prompt".into(),
                id: String::new(),
                name: String::new(),
                is_error: false,
            }],
        };
        store.append_messages(&record.id, &[message]).unwrap();
        let usage = Usage {
            input_tokens: 11,
            output_tokens: 5,
            cache_read_input_tokens: 1,
            cache_creation_input_tokens: 2,
        };
        store.append_usage(&record.id, &usage, 2, "ready").unwrap();
        store
            .append_turn_start(&record.id, "interactive", "secret token sk-test")
            .unwrap();
        store
            .append_tool_start(&record.id, "c1", "greppy", "→ greppy who-calls foo")
            .unwrap();
        store
            .append_tool_finish(&record.id, "c1", false, 12, &"x".repeat(500))
            .unwrap();
        store
            .append_turn_done(&record.id, "ready", 1, &usage)
            .unwrap();
        store.append_turn_error(&record.id, "boom").unwrap();
        store
            .append_worktree(&record.id, "/tmp/wt", "refs/greppy/agent/run")
            .unwrap();

        let loaded = store.load(&record.id).unwrap();
        assert_eq!(loaded.source, "interactive");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].parts[0].text, "prompt");
        assert_eq!(loaded.usage.input_tokens, 11);
        assert_eq!(loaded.usage.output_tokens, 5);
        assert_eq!(loaded.usage.cache_read_input_tokens, 1);
        assert_eq!(loaded.usage.cache_creation_input_tokens, 2);
        assert_eq!(loaded.turns, 2);
        assert_eq!(loaded.stop, "ready");
        assert_eq!(loaded.worktree, "/tmp/wt");
        assert_eq!(loaded.proposal_ref, "refs/greppy/agent/run");
        assert!(!loaded.recovered);

        let raw = fs::read_to_string(store.path_for(&record.id)).unwrap();
        assert!(raw.contains(r#""type":"turn""#));
        assert!(raw.contains(r#""type":"tool""#));
        let finish = raw
            .lines()
            .find(|line| line.contains(r#""event":"finish""#))
            .unwrap();
        let value: Value = serde_json::from_str(finish).unwrap();
        assert_eq!(value["preview"].as_str().unwrap().chars().count(), 400);
        let _ = fs::remove_dir_all(root);
    }
}
