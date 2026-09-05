//! Shared, source-addressable experimental inputs for frozen embedding heads.
//! No labels, selections or replacement text are produced here.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const INPUT_SCHEMA: &str = "greppy.heads.prepared-input.v1";
pub const CONTRACT: &str = "greppy.heads.target-context-json.v1";

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Head {
    LogClassifier,
    LogRanker,
    WebRanker,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub sha256: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: String,
    pub head: Head,
    pub target: Span,
    pub context: Vec<Span>,
    pub task: Option<String>,
    pub observation_id: Option<String>,
    pub goal_version: Option<u64>,
    /// Null denotes an unknown action. It is never rewritten as a success.
    pub last_action: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_tokens: usize,
    pub max_target_bytes: usize,
    pub max_context_bytes: usize,
    pub max_parts: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            max_target_bytes: 2048,
            max_context_bytes: 2048,
            max_parts: 4096,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedInput {
    pub schema: String,
    pub contract_sha256: String,
    pub id: String,
    pub candidate_id: String,
    pub source_id: String,
    pub source_sha256: String,
    pub head: Head,
    pub original_target: Span,
    pub target: Span,
    pub target_sha256: String,
    pub context_used: Vec<Span>,
    pub context_omitted: Vec<Span>,
    pub conditioning_sha256: String,
    pub observation_id: Option<String>,
    pub goal_version: Option<u64>,
    pub prompt: String,
    pub input_sha256: String,
    pub token_count: usize,
}

fn json_hash<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|v| sha256(&v))
        .map_err(|e| e.to_string())
}

pub fn contract_hash(limits: Limits) -> Result<String, String> {
    if !(32..=2048).contains(&limits.max_tokens)
        || !(4..=65536).contains(&limits.max_target_bytes)
        || limits.max_context_bytes > 65536
        || !(1..=1_000_000).contains(&limits.max_parts)
    {
        return Err("invalid head input limits".into());
    }
    json_hash(&(
        CONTRACT,
        limits,
        "embeddinggemma-classification-prefix",
        "json-target-context",
        "split-target-at-utf8-boundary",
        "drop-whole-context-from-tail",
        "no-target-truncation",
    ))
}

fn span_text(source: &str, span: Span) -> Result<&str, String> {
    if span.start >= span.end {
        return Err("empty or reversed source span".into());
    }
    source
        .get(span.start..span.end)
        .ok_or_else(|| "source span is out of bounds or splits UTF-8".into())
}

fn boundary(text: &str, mut end: usize) -> usize {
    end = end.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Every physical line, including CRLF and the final unterminated line.
/// This iterator traverses the whole source; no prefix-only candidate selection.
pub fn log_spans(text: &str) -> impl Iterator<Item = Span> + '_ {
    let mut start = 0;
    text.split_inclusive('\n').map(move |line| {
        let span = Span {
            start,
            end: start + line.len(),
        };
        start = span.end;
        span
    })
}

/// Validate the source once, then prepare many targets without rehashing the
/// entire output for every line. Borrowing keeps verified bytes immutable.
pub struct VerifiedSource<'a> {
    source: &'a Source,
}
impl<'a> VerifiedSource<'a> {
    pub fn new(source: &'a Source) -> Result<Self, String> {
        if source.id.is_empty()
            || source.id.len() > 256
            || source.sha256 != sha256(source.text.as_bytes())
        {
            return Err("invalid source identity or content checksum".into());
        }
        Ok(Self { source })
    }

    /// Token counter must be the exact native tokenizer with no truncation.
    /// Failure returns no partial candidate. Splits need span-level annotation;
    /// a parent's label must never be copied by majority projection.
    pub fn prepare<F>(
        &self,
        candidate: &Candidate,
        limits: Limits,
        mut token_len: F,
    ) -> Result<Vec<PreparedInput>, String>
    where
        F: FnMut(&str) -> Result<usize, String>,
    {
        let source = self.source;
        let contract = contract_hash(limits)?;
        if candidate.id.is_empty() || candidate.id.len() > 256 || candidate.context.len() > 256 {
            return Err("invalid candidate identity or context count".into());
        }
        span_text(&source.text, candidate.target)?;
        let mut prior_end = None;
        for span in &candidate.context {
            span_text(&source.text, *span)?;
            if (span.start < candidate.target.end && span.end > candidate.target.start)
                || prior_end.is_some_and(|end| span.start < end)
            {
                return Err("context spans overlap target or are not disjoint source order".into());
            }
            prior_end = Some(span.end);
        }
        match candidate.head {
            Head::LogClassifier => {
                if candidate.task.is_some()
                    || candidate.last_action.is_some()
                    || candidate.observation_id.is_some()
                    || candidate.goal_version.is_some()
                {
                    return Err("severity input must not carry task-dependent conditioning".into());
                }
            }
            Head::LogRanker | Head::WebRanker => {
                if !candidate
                    .task
                    .as_ref()
                    .is_some_and(|s| !s.trim().is_empty() && s.len() <= 8192)
                {
                    return Err("ranker requires an explicit bounded task".into());
                }
            }
        }
        if candidate.head == Head::WebRanker {
            if !candidate
                .observation_id
                .as_ref()
                .is_some_and(|s| !s.is_empty() && s.len() <= 256)
                || !candidate.goal_version.is_some_and(|v| v > 0)
            {
                return Err(
                    "Web ranker requires observation identity and positive goal version".into(),
                );
            }
        } else if candidate.observation_id.is_some()
            || candidate.goal_version.is_some()
            || candidate.last_action.is_some()
        {
            return Err("Web observation conditioning on a log candidate".into());
        }
        if serde_json::to_vec(&candidate.last_action)
            .map_err(|e| e.to_string())?
            .len()
            > 8192
        {
            return Err("action context exceeds input limit".into());
        }
        let conditioning = json_hash(&(
            candidate.head,
            &candidate.task,
            &candidate.observation_id,
            candidate.goal_version,
            &candidate.last_action,
        ))?;
        let mut output = Vec::new();
        let mut start = candidate.target.start;
        while start < candidate.target.end {
            if output.len() == limits.max_parts {
                return Err("candidate exceeds part budget; retain deterministic output".into());
            }
            let remaining = &source.text[start..candidate.target.end];
            let mut size = boundary(remaining, limits.max_target_bytes);
            loop {
                let target = Span {
                    start,
                    end: start + size,
                };
                let mut used = Vec::new();
                let mut context_bytes = 0;
                for span in &candidate.context {
                    if context_bytes + span.end - span.start <= limits.max_context_bytes {
                        used.push(*span);
                        context_bytes += span.end - span.start;
                    }
                }
                let (prompt, tokens) = loop {
                    let contexts = used
                        .iter()
                        .map(|s| &source.text[s.start..s.end])
                        .collect::<Vec<_>>();
                    let body = serde_json::json!({"head": candidate.head,
                        "target": &source.text[target.start..target.end], "context": contexts,
                        "task": candidate.task, "last_action": candidate.last_action});
                    let prompt = format!("task: classification | query: {}", body);
                    let tokens = token_len(&prompt)?;
                    if tokens == 0 {
                        return Err("native tokenizer returned no tokens".into());
                    }
                    if tokens <= limits.max_tokens || used.is_empty() {
                        break (prompt, tokens);
                    }
                    used.pop();
                };
                if tokens <= limits.max_tokens {
                    let input_hash = sha256(prompt.as_bytes());
                    let omitted = candidate
                        .context
                        .iter()
                        .copied()
                        .filter(|s| !used.contains(s))
                        .collect();
                    output.push(PreparedInput {
                        schema: INPUT_SCHEMA.into(),
                        contract_sha256: contract.clone(),
                        id: json_hash(&(
                            &source.id,
                            &source.sha256,
                            &candidate.id,
                            target,
                            &conditioning,
                            &contract,
                            &input_hash,
                        ))?,
                        candidate_id: candidate.id.clone(),
                        source_id: source.id.clone(),
                        source_sha256: source.sha256.clone(),
                        head: candidate.head,
                        original_target: candidate.target,
                        target,
                        target_sha256: sha256(source.text[target.start..target.end].as_bytes()),
                        context_used: used,
                        context_omitted: omitted,
                        conditioning_sha256: conditioning.clone(),
                        observation_id: candidate.observation_id.clone(),
                        goal_version: candidate.goal_version,
                        prompt,
                        input_sha256: input_hash,
                        token_count: tokens,
                    });
                    start = target.end;
                    break;
                }
                let next_size = boundary(remaining, size / 2);
                if next_size == 0 {
                    return Err(
                        "task and minimal target exceed token budget; no truncation permitted"
                            .into(),
                    );
                }
                size = next_size;
            }
        }
        Ok(output)
    }
}
