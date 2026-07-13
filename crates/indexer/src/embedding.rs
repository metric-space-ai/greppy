//! Symbol/code-span embedding indexing.
//!
//! This module is the R5 bridge between graph indexing and vector search. It
//! does not invent summaries or Markdown context: every stored vector is derived
//! from the exact source span of a persisted graph node.

use std::collections::HashMap;
use std::path::Path;

use greppy_core::{Error, Result};
use greppy_embed_native::{
    tokenizer::DEFAULT_MAX_LENGTH, EmbedTask, EmbeddingGemma, CODE_RETRIEVAL_PROFILE,
    PROMPT_VERSION,
};
use greppy_store::{file_state::sha256_hex, NewVectorEmbedding, ReusableVectorEmbeddingKey, Store};

const NODE_PAGE_SIZE: usize = 1000;
const DEFAULT_MAX_SPAN_BYTES: usize = 32 * 1024;
const MAX_DERIVED_DEF_LINES: usize = 400;

/// Number of code documents embedded per forward pass. Batching is what makes
/// the embedding-index step GPU-parallelizable: one padded forward over
/// `(batch, seq_len)` instead of `batch` serial single-doc passes. Overridable
/// via `GREPPY_EMBED_BATCH` (clamped to `>= 1`); `1` reproduces the old
/// serial path exactly.
const DEFAULT_EMBED_BATCH: usize = 16;
const EMBED_BATCH_ENV: &str = "GREPPY_EMBED_BATCH";
const EMBED_SCHEDULE_BATCHES: usize = 16;

fn embed_batch_size() -> usize {
    parse_embed_batch_size(std::env::var(EMBED_BATCH_ENV).ok().as_deref())
}

fn parse_embed_batch_size(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_EMBED_BATCH)
}

fn embed_schedule_window(batch_size: usize) -> usize {
    batch_size
        .saturating_mul(EMBED_SCHEDULE_BATCHES)
        .max(batch_size)
}

/// Provider interface used by the indexer to embed real code spans.
pub trait CodeEmbeddingProvider {
    fn model_id(&self) -> &str;
    fn prompt_version(&self) -> &str;
    fn task_profile(&self) -> &str;
    fn embed_code_document(&mut self, title: Option<&str>, content: &str) -> Result<Vec<f32>>;
    fn max_input_tokens(&self) -> usize {
        DEFAULT_MAX_LENGTH
    }
    fn document_token_len(&self, title: Option<&str>, content: &str) -> Result<usize> {
        Ok(EmbedTask::document_with_title(title, content)
            .split_whitespace()
            .count()
            .max(1))
    }

    /// Embed a batch of `(title, content)` documents in one call, returning
    /// vectors in input order.
    ///
    /// The default implementation loops over [`embed_code_document`] so any
    /// existing provider keeps working unchanged. Providers backed by a model
    /// with true batch inference (e.g. [`EmbeddingGemmaCodeProvider`]) override
    /// this to run a single padded forward pass over the whole batch, which is
    /// what makes large-repo embedding indexing GPU-parallelizable. The
    /// per-vector result MUST match the serial path (that equivalence is
    /// covered by a test).
    fn embed_code_documents(&mut self, docs: &[(Option<&str>, &str)]) -> Result<Vec<Vec<f32>>> {
        docs.iter()
            .map(|(title, content)| self.embed_code_document(*title, content))
            .collect()
    }
}

/// Production provider backed by native EmbeddingGemma inference.
pub struct EmbeddingGemmaCodeProvider<'a> {
    model_id: String,
    model: &'a EmbeddingGemma,
}

impl<'a> EmbeddingGemmaCodeProvider<'a> {
    pub fn new(model_id: impl Into<String>, model: &'a EmbeddingGemma) -> Self {
        Self {
            model_id: model_id.into(),
            model,
        }
    }
}

impl CodeEmbeddingProvider for EmbeddingGemmaCodeProvider<'_> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn prompt_version(&self) -> &str {
        PROMPT_VERSION
    }

    fn task_profile(&self) -> &str {
        CODE_RETRIEVAL_PROFILE
    }

    fn embed_code_document(&mut self, title: Option<&str>, content: &str) -> Result<Vec<f32>> {
        self.model
            .embed_document(title, content)
            .map_err(|e| Error::Store(format!("embeddinggemma document embedding: {e}")))
    }

    fn max_input_tokens(&self) -> usize {
        self.model.max_length()
    }

    fn document_token_len(&self, title: Option<&str>, content: &str) -> Result<usize> {
        self.model
            .document_token_len(title, content)
            .map_err(|e| Error::Store(format!("embeddinggemma document tokenization: {e}")))
    }

    fn embed_code_documents(&mut self, docs: &[(Option<&str>, &str)]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed_documents(docs)
            .map_err(|e| Error::Store(format!("embeddinggemma document embedding: {e}")))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingIndexOptions {
    pub graph_generation: u64,
    pub max_span_bytes: usize,
    pub prune_before_generation: bool,
}

impl EmbeddingIndexOptions {
    pub fn for_generation(graph_generation: u64) -> Self {
        Self {
            graph_generation,
            max_span_bytes: DEFAULT_MAX_SPAN_BYTES,
            prune_before_generation: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingIndexReport {
    pub nodes_considered: usize,
    pub nodes_embedded: usize,
    pub nodes_reused: usize,
    pub nodes_skipped_non_definition: usize,
    pub nodes_skipped_missing_file: usize,
    pub nodes_skipped_invalid_span: usize,
    pub nodes_skipped_oversize: usize,
    pub stale_rows_pruned: usize,
}

/// Index vectors for all embeddable symbol nodes in `project`.
pub fn index_code_embeddings_for_project(
    store: &mut Store,
    root: &Path,
    project: &str,
    provider: &mut dyn CodeEmbeddingProvider,
    options: EmbeddingIndexOptions,
) -> Result<EmbeddingIndexReport> {
    let mut report = EmbeddingIndexReport::default();
    let mut file_cache: HashMap<String, Option<String>> = HashMap::new();
    let mut offset = 0usize;

    let batch_size = embed_batch_size();
    let schedule_window = embed_schedule_window(batch_size);
    // Candidate documents accumulated until a full batch is ready. Owned
    // `title`/`chunk` (already owned by construction) plus the whole `Node`
    // so the exact upsert fields are preserved on flush.
    let mut pending: Vec<PendingDoc> = Vec::with_capacity(schedule_window);
    let mut embedding_batch_failed = false;

    loop {
        let nodes = store.list_nodes(project, "", "", offset, NODE_PAGE_SIZE)?;
        if nodes.is_empty() {
            break;
        }
        offset += nodes.len();

        for node in nodes {
            report.nodes_considered += 1;
            if !is_embedding_candidate_label(&node.label) {
                report.nodes_skipped_non_definition += 1;
                continue;
            }

            let source = match cached_file_source(&mut file_cache, root, &node.file_path) {
                Ok(Some(source)) => source,
                Ok(None) => {
                    report.nodes_skipped_missing_file += 1;
                    continue;
                }
                Err(_) => {
                    report.nodes_skipped_missing_file += 1;
                    continue;
                }
            };
            // Skip minified / generated spans (a single line longer than this is
            // machine-generated): embedding them is noise and pathologically slow.
            const MAX_EMBED_LINE_BYTES: usize = 2048;
            if span_has_overlong_line(source, node.start_line, node.end_line, MAX_EMBED_LINE_BYTES)
            {
                report.nodes_skipped_oversize += 1;
                continue;
            }
            let title = format!(
                "{}:{}-{} {}",
                node.file_path, node.start_line, node.end_line, node.qualified_name
            );
            let chunks = embedding_chunks(
                source,
                node.start_line,
                node.end_line,
                &title,
                provider,
                options.max_span_bytes,
            )?;
            if chunks.is_empty() {
                report.nodes_skipped_invalid_span += 1;
                continue;
            }

            for chunk in chunks {
                let content_sha256 = sha256_hex(chunk.text.as_bytes());
                if let Some(existing) =
                    store.find_reusable_vector_embedding(&ReusableVectorEmbeddingKey {
                        project,
                        model_id: provider.model_id(),
                        prompt_version: provider.prompt_version(),
                        task: provider.task_profile(),
                        qualified_name: &node.qualified_name,
                        chunk_idx: chunk.chunk_idx,
                        content_sha256: &content_sha256,
                    })?
                {
                    store.upsert_vector_embedding(&NewVectorEmbedding {
                        project: node.project.clone(),
                        model_id: provider.model_id().to_string(),
                        prompt_version: provider.prompt_version().to_string(),
                        task: provider.task_profile().to_string(),
                        node_id: Some(node.id),
                        chunk_idx: chunk.chunk_idx,
                        qualified_name: node.qualified_name.clone(),
                        file_path: node.file_path.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        content_sha256,
                        graph_generation: options.graph_generation,
                        vector: existing.vector,
                    })?;
                    report.nodes_embedded += 1;
                    report.nodes_reused += 1;
                    continue;
                }
                pending.push(PendingDoc {
                    node: node.clone(),
                    title: title.clone(),
                    prompt_token_len: provider.document_token_len(Some(&title), &chunk.text)?,
                    chunk,
                });
                if pending.len() >= schedule_window {
                    let flush = flush_embedding_batch(
                        store,
                        provider,
                        options.graph_generation,
                        batch_size,
                        &mut pending,
                    )?;
                    report.nodes_embedded += flush.written;
                    embedding_batch_failed |= flush.failed;
                }
            }
        }
    }

    if !pending.is_empty() {
        let flush = flush_embedding_batch(
            store,
            provider,
            options.graph_generation,
            batch_size,
            &mut pending,
        )?;
        report.nodes_embedded += flush.written;
        embedding_batch_failed |= flush.failed;
    }

    if embedding_batch_failed {
        return Err(Error::Store(
            "embedding generation incomplete: at least one batch failed".into(),
        ));
    }

    if options.prune_before_generation {
        report.stale_rows_pruned =
            store.prune_vector_embeddings_before_generation(project, options.graph_generation)?;
    }
    Ok(report)
}

/// A candidate code document awaiting embedding: the source `node` (whose owned
/// fields become the upsert record) plus the already-computed prompt `title`
/// and one exact source chunk.
struct PendingDoc {
    node: greppy_store::Node,
    title: String,
    prompt_token_len: usize,
    chunk: EmbeddingChunk,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EmbeddingChunk {
    chunk_idx: i64,
    start_line: i64,
    end_line: i64,
    text: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FlushEmbeddingBatchReport {
    written: usize,
    failed: bool,
}

/// Embed every buffered document in `pending` in length-sorted inference calls
/// and upsert the resulting vectors, then empty the buffer.
///
/// This preserves the exact per-node upsert fields of the old serial path — the
/// only change is that each `vector` comes from a full batch of nearby prompt
/// lengths. The scheduling window is bounded independently of repository size,
/// avoiding both pathological padding and unbounded source retention.
fn flush_embedding_batch(
    store: &mut Store,
    provider: &mut dyn CodeEmbeddingProvider,
    graph_generation: u64,
    batch_size: usize,
    pending: &mut Vec<PendingDoc>,
) -> Result<FlushEmbeddingBatchReport> {
    if pending.is_empty() {
        return Ok(FlushEmbeddingBatchReport::default());
    }

    pending.sort_by_key(|doc| doc.prompt_token_len);
    let mut report = FlushEmbeddingBatchReport::default();
    while !pending.is_empty() {
        let batch_len = pending.len().min(batch_size);
        let mut batch_docs = pending.drain(..batch_len).collect::<Vec<_>>();
        let bucket_report =
            flush_length_sorted_batch(store, provider, graph_generation, &mut batch_docs)?;
        report.written += bucket_report.written;
        report.failed |= bucket_report.failed;
    }
    Ok(report)
}

fn flush_length_sorted_batch(
    store: &mut Store,
    provider: &mut dyn CodeEmbeddingProvider,
    graph_generation: u64,
    pending: &mut Vec<PendingDoc>,
) -> Result<FlushEmbeddingBatchReport> {
    let docs = pending
        .iter()
        .map(|doc| (Some(doc.title.as_str()), doc.chunk.text.as_str()))
        .collect::<Vec<_>>();
    let vectors = match provider.embed_code_documents(&docs) {
        Ok(vectors) => vectors,
        Err(e) => {
            log_embedding_skip_once(&e);
            pending.clear();
            return Ok(FlushEmbeddingBatchReport {
                written: 0,
                failed: true,
            });
        }
    };
    if vectors.len() != pending.len() {
        log_embedding_skip_once(&Error::Store(format!(
            "embedding provider returned {} vectors for a batch of {} documents",
            vectors.len(),
            pending.len()
        )));
        pending.clear();
        return Ok(FlushEmbeddingBatchReport {
            written: 0,
            failed: true,
        });
    }
    let model_id = provider.model_id().to_string();
    let prompt_version = provider.prompt_version().to_string();
    let task = provider.task_profile().to_string();
    let mut written = 0usize;
    for (doc, vector) in pending.drain(..).zip(vectors) {
        let node = doc.node;
        store.upsert_vector_embedding(&NewVectorEmbedding {
            project: node.project,
            model_id: model_id.clone(),
            prompt_version: prompt_version.clone(),
            task: task.clone(),
            node_id: Some(node.id),
            chunk_idx: doc.chunk.chunk_idx,
            qualified_name: node.qualified_name,
            file_path: node.file_path,
            start_line: doc.chunk.start_line,
            end_line: doc.chunk.end_line,
            content_sha256: sha256_hex(doc.chunk.text.as_bytes()),
            graph_generation,
            vector,
        })?;
        written += 1;
    }
    Ok(FlushEmbeddingBatchReport {
        written,
        failed: false,
    })
}

fn log_embedding_skip_once(err: &Error) {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("greppy index: embedding unavailable; skipping embedding batch: {err}");
    }
}

fn cached_file_source<'a>(
    cache: &'a mut HashMap<String, Option<String>>,
    root: &Path,
    rel_path: &str,
) -> Result<Option<&'a str>> {
    if !cache.contains_key(rel_path) {
        let path = root.join(rel_path);
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => Some(s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                return Err(Error::io(format!("read {}", path.display()), e));
            }
        };
        cache.insert(rel_path.to_string(), source);
    }
    Ok(cache.get(rel_path).and_then(|s| s.as_deref()))
}

/// The code documents a graph node contributes to the semantic index.
///
/// Every chunk starts with the definition header/signature, then carries a
/// consecutive body window whose full `title: ... | text: ...` prompt is within
/// the provider's token budget. Short definitions stay one document; long ones
/// are tiled with overlap until the final source line is covered.
fn embedding_chunks(
    source: &str,
    start_line: i64,
    end_line: i64,
    title: &str,
    provider: &dyn CodeEmbeddingProvider,
    max_bytes: usize,
) -> Result<Vec<EmbeddingChunk>> {
    if start_line <= 0 || end_line < start_line {
        return Ok(Vec::new());
    }
    let lines: Vec<&str> = source.lines().collect();
    let start_idx = (start_line as usize).saturating_sub(1);
    if start_idx >= lines.len() {
        return Ok(Vec::new());
    }

    let def_end_idx = if end_line > start_line {
        ((end_line as usize).saturating_sub(1)).min(lines.len().saturating_sub(1))
    } else {
        definition_end_idx_full(&lines, start_idx, max_bytes)
    };
    if def_end_idx < start_idx {
        return Ok(Vec::new());
    }

    let header_end_idx = signature_end_idx(&lines, start_idx, def_end_idx);
    let header = join_lines(&lines[start_idx..=header_end_idx]);
    let full = join_lines(&lines[start_idx..=def_end_idx]);
    let max_tokens = provider.max_input_tokens().max(1);
    if prompt_fits(provider, title, &full, max_tokens)? {
        return Ok(vec![EmbeddingChunk {
            chunk_idx: 0,
            start_line,
            end_line: (def_end_idx + 1) as i64,
            text: full,
        }]);
    }

    let body_start_idx = header_end_idx + 1;
    if !prompt_fits(provider, title, &header, max_tokens)?
        || body_start_idx > def_end_idx
        || max_fitting_segment_len(title, &header, lines[body_start_idx], provider, max_tokens)?
            .filter(|&n| n > 0)
            .is_none()
    {
        return split_oversize_text_with_header(
            title,
            &full,
            &header,
            start_line,
            def_end_idx,
            provider,
            max_tokens,
        );
    }

    let mut out = Vec::new();
    let mut body_idx = body_start_idx;
    while body_idx <= def_end_idx {
        let Some(body_end_idx) = max_fitting_body_end(
            &lines,
            &header,
            body_idx,
            def_end_idx,
            title,
            provider,
            max_tokens,
        )?
        else {
            append_oversize_body_line_chunks(
                &mut out,
                title,
                &header,
                lines[body_idx],
                body_idx,
                provider,
                max_tokens,
            )?;
            body_idx += 1;
            continue;
        };

        let text = chunk_text(&header, &lines[body_idx..=body_end_idx]);
        let chunk_start_line = if out.is_empty() {
            start_line
        } else {
            (body_idx + 1) as i64
        };
        out.push(EmbeddingChunk {
            chunk_idx: out.len() as i64,
            start_line: chunk_start_line,
            end_line: (body_end_idx + 1) as i64,
            text,
        });

        if body_end_idx >= def_end_idx {
            break;
        }
        let next_idx =
            overlapped_next_body_idx(&lines, &header, body_idx, body_end_idx, title, provider)?;
        body_idx = if next_idx <= body_idx {
            body_end_idx + 1
        } else {
            next_idx
        };
    }
    Ok(out)
}

fn prompt_fits(
    provider: &dyn CodeEmbeddingProvider,
    title: &str,
    content: &str,
    max_tokens: usize,
) -> Result<bool> {
    Ok(provider.document_token_len(Some(title), content)? <= max_tokens)
}

fn join_lines(lines: &[&str]) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn chunk_text(header: &str, body_lines: &[&str]) -> String {
    let mut out = String::with_capacity(
        header.len() + body_lines.iter().map(|line| line.len() + 1).sum::<usize>(),
    );
    out.push_str(header);
    for line in body_lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn signature_end_idx(lines: &[&str], start_idx: usize, def_end_idx: usize) -> usize {
    for (idx, line) in lines
        .iter()
        .enumerate()
        .take(def_end_idx + 1)
        .skip(start_idx)
    {
        let trimmed = line.trim_end();
        if trimmed.contains('{') || trimmed.contains("=>") || trimmed.ends_with(':') {
            return idx;
        }
    }
    start_idx
}

/// Same brace-balancing rules as `greppy_core::spans`, but without that
/// module's display-oriented scan cap for genuine graph spans. When the graph
/// only gives a declaration line, the derived extent is still bounded so a
/// malformed unclosed definition cannot consume the rest of the file.
fn definition_end_idx_full(lines: &[&str], start_idx: usize, max_bytes: usize) -> usize {
    let scan_end = derived_definition_scan_end_idx(lines, start_idx, max_bytes);
    let start_indent = leading_whitespace_len(lines[start_idx]);
    let mut depth: i32 = 0;
    let mut opened = false;
    for (idx, raw) in lines.iter().enumerate().take(scan_end + 1).skip(start_idx) {
        if idx > start_idx
            && opened
            && depth > 0
            && leading_whitespace_len(raw) <= start_indent
            && looks_like_definition_start(raw)
        {
            return previous_nonblank_idx(lines, start_idx, idx.saturating_sub(1));
        }
        let mut in_str: Option<char> = None;
        let mut prev = '\0';
        let chars: Vec<char> = raw.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if let Some(q) = in_str {
                if c == q && prev != '\\' {
                    in_str = None;
                }
                prev = c;
                i += 1;
                continue;
            }
            if c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                break;
            }
            match c {
                '"' | '\'' => in_str = Some(c),
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' if opened => {
                    depth -= 1;
                    if depth <= 0 {
                        return idx;
                    }
                }
                ';' if !opened => return idx,
                _ => {}
            }
            prev = c;
            i += 1;
        }
    }
    if opened {
        scan_end
    } else {
        greppy_core::spans::definition_end_idx(&lines[..=scan_end], start_idx)
    }
}

fn derived_definition_scan_end_idx(lines: &[&str], start_idx: usize, max_bytes: usize) -> usize {
    let last = lines.len().saturating_sub(1);
    let line_cap = start_idx.saturating_add(MAX_DERIVED_DEF_LINES).min(last);
    if max_bytes == 0 {
        return start_idx.min(last);
    }

    let mut bytes = 0usize;
    let mut end = start_idx.min(last);
    for (idx, line) in lines.iter().enumerate().take(line_cap + 1).skip(start_idx) {
        let next = bytes.saturating_add(line.len()).saturating_add(1);
        if idx > start_idx && next > max_bytes {
            break;
        }
        bytes = next;
        end = idx;
    }
    end
}

fn leading_whitespace_len(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).count()
}

fn previous_nonblank_idx(lines: &[&str], start_idx: usize, mut idx: usize) -> usize {
    while idx > start_idx && lines[idx].trim().is_empty() {
        idx -= 1;
    }
    idx
}

fn looks_like_definition_start(line: &str) -> bool {
    let trimmed = line.trim_start();
    let without_visibility = trimmed
        .strip_prefix("pub ")
        .or_else(|| trimmed.strip_prefix("pub(crate) "))
        .or_else(|| trimmed.strip_prefix("public "))
        .or_else(|| trimmed.strip_prefix("private "))
        .or_else(|| trimmed.strip_prefix("protected "))
        .unwrap_or(trimmed);
    let without_async = without_visibility
        .strip_prefix("async ")
        .unwrap_or(without_visibility);
    without_async.starts_with("fn ")
        || without_async.starts_with("def ")
        || without_async.starts_with("class ")
        || without_async.starts_with("struct ")
        || without_async.starts_with("enum ")
        || without_async.starts_with("trait ")
        || without_async.starts_with("interface ")
        || without_async.starts_with("function ")
        || without_async.starts_with("export function ")
        || without_async.starts_with("export class ")
}

fn max_fitting_body_end(
    lines: &[&str],
    header: &str,
    body_start_idx: usize,
    def_end_idx: usize,
    title: &str,
    provider: &dyn CodeEmbeddingProvider,
    max_tokens: usize,
) -> Result<Option<usize>> {
    let mut lo = body_start_idx;
    let mut hi = def_end_idx;
    let mut best = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let text = chunk_text(header, &lines[body_start_idx..=mid]);
        if prompt_fits(provider, title, &text, max_tokens)? {
            best = Some(mid);
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    Ok(best)
}

fn body_token_count(
    lines: &[&str],
    header: &str,
    body_start_idx: usize,
    body_end_idx: usize,
    title: &str,
    provider: &dyn CodeEmbeddingProvider,
) -> Result<usize> {
    let whole = provider.document_token_len(
        Some(title),
        &chunk_text(header, &lines[body_start_idx..=body_end_idx]),
    )?;
    let header_only = provider.document_token_len(Some(title), header)?;
    Ok(whole.saturating_sub(header_only))
}

fn overlapped_next_body_idx(
    lines: &[&str],
    header: &str,
    body_start_idx: usize,
    body_end_idx: usize,
    title: &str,
    provider: &dyn CodeEmbeddingProvider,
) -> Result<usize> {
    if body_end_idx <= body_start_idx {
        return Ok(body_end_idx + 1);
    }
    let body_tokens =
        body_token_count(lines, header, body_start_idx, body_end_idx, title, provider)?;
    let target = ((body_tokens as f64) * 0.15).ceil().max(1.0) as usize;
    // `body_token_count(idx..body_end)` grows monotonically as `idx` decreases,
    // so the largest `idx` whose overlap still reaches `target` can be found with
    // a binary search instead of re-tokenizing `[idx..body_end]` for every line
    // (which is O(body_bytes^2) and stalls on minified / very long bodies).
    let mut lo = body_start_idx + 1;
    let mut hi = body_end_idx;
    let mut ans = body_end_idx + 1;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let overlap_tokens = body_token_count(lines, header, mid, body_end_idx, title, provider)?;
        if overlap_tokens >= target {
            ans = mid;
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    Ok(ans)
}

fn append_oversize_body_line_chunks(
    out: &mut Vec<EmbeddingChunk>,
    title: &str,
    header: &str,
    line: &str,
    line_idx: usize,
    provider: &dyn CodeEmbeddingProvider,
    max_tokens: usize,
) -> Result<()> {
    let mut byte_start = 0usize;
    while byte_start < line.len() {
        let segment_len =
            max_fitting_segment_len(title, header, &line[byte_start..], provider, max_tokens)?;
        let Some(take) = segment_len.filter(|&n| n > 0) else {
            break;
        };
        let byte_end = byte_start + take;
        let mut text = String::with_capacity(header.len() + take + 1);
        text.push_str(header);
        text.push_str(&line[byte_start..byte_end]);
        text.push('\n');
        out.push(EmbeddingChunk {
            chunk_idx: out.len() as i64,
            start_line: (line_idx + 1) as i64,
            end_line: (line_idx + 1) as i64,
            text,
        });
        byte_start = byte_end;
    }
    Ok(())
}

fn split_oversize_text_with_header(
    title: &str,
    text: &str,
    header: &str,
    start_line: i64,
    def_end_idx: usize,
    provider: &dyn CodeEmbeddingProvider,
    max_tokens: usize,
) -> Result<Vec<EmbeddingChunk>> {
    let Some(header_prefix_len) =
        max_header_prefix_len_for_progress(title, header, text, provider, max_tokens)?
    else {
        return Ok(Vec::new());
    };
    let header_prefix = &header[..header_prefix_len];
    let suffix = &text[header_prefix_len..];
    let mut chunks = Vec::new();
    let mut byte_start = 0usize;
    while byte_start < suffix.len() {
        let segment_len = max_fitting_segment_len(
            title,
            header_prefix,
            &suffix[byte_start..],
            provider,
            max_tokens,
        )?;
        let Some(take) = segment_len.filter(|&n| n > 0) else {
            break;
        };
        let byte_end = byte_start + take;
        let mut chunk = String::with_capacity(header_prefix.len() + take + 1);
        chunk.push_str(header_prefix);
        chunk.push_str(&suffix[byte_start..byte_end]);
        if !chunk.ends_with('\n') {
            chunk.push('\n');
        }
        chunks.push(EmbeddingChunk {
            chunk_idx: chunks.len() as i64,
            start_line,
            end_line: (def_end_idx + 1) as i64,
            text: chunk,
        });
        byte_start = byte_end;
    }
    Ok(chunks)
}

fn max_header_prefix_len_for_progress(
    title: &str,
    header: &str,
    text: &str,
    provider: &dyn CodeEmbeddingProvider,
    max_tokens: usize,
) -> Result<Option<usize>> {
    let boundaries = header
        .char_indices()
        .map(|(i, c)| i + c.len_utf8())
        .collect::<Vec<_>>();
    if boundaries.is_empty() {
        return Ok(Some(0));
    }

    let mut lo = 1usize;
    let mut hi = boundaries.len();
    let mut best = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let prefix_len = boundaries[mid - 1];
        let prefix = &header[..prefix_len];
        let suffix = &text[prefix_len..];
        let fits = max_fitting_segment_len(title, prefix, suffix, provider, max_tokens)?
            .filter(|&n| n > 0)
            .is_some();
        if fits {
            best = Some(prefix_len);
            lo = mid + 1;
        } else if mid == 1 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    Ok(best)
}

fn max_fitting_segment_len(
    title: &str,
    prefix: &str,
    text: &str,
    provider: &dyn CodeEmbeddingProvider,
    max_tokens: usize,
) -> Result<Option<usize>> {
    if text.is_empty() {
        return Ok(Some(0));
    }
    // A fitting segment holds at most `max_tokens` tokens, so its byte length is
    // bounded. Search within a byte window rather than the whole `text`, so each
    // call is O(window) instead of O(text.len()) - the callers invoke this in a
    // `while byte_start < len` loop, and without the window a minified /
    // single-line blob (e.g. a bundled JS file hundreds of KB long) makes that
    // loop O(n^2) and stalls indexing for minutes. If the whole window turns out
    // to fit, the window doubles and retries, so the exact maximum is still
    // returned for genuinely token-sparse text.
    let mut window = max_tokens.saturating_mul(64).max(1);
    loop {
        let mut cap = window.min(text.len());
        while cap < text.len() && !text.is_char_boundary(cap) {
            cap += 1;
        }
        let search = &text[..cap];
        let boundaries = search
            .char_indices()
            .map(|(i, c)| i + c.len_utf8())
            .collect::<Vec<_>>();
        let mut lo = 0usize;
        let mut hi = boundaries.len();
        let mut best = None;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let end = boundaries[mid];
            let mut candidate = String::with_capacity(prefix.len() + end + 1);
            candidate.push_str(prefix);
            candidate.push_str(&text[..end]);
            if !candidate.ends_with('\n') {
                candidate.push('\n');
            }
            if prompt_fits(provider, title, &candidate, max_tokens)? {
                best = Some(end);
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if best == Some(cap) && cap < text.len() {
            window = window.saturating_mul(2);
            continue;
        }
        return Ok(best);
    }
}

#[cfg(test)]
fn source_span(source: &str, start_line: i64, end_line: i64) -> Option<String> {
    if start_line <= 0 || end_line < start_line {
        return None;
    }
    let start = start_line as usize;
    let end = end_line as usize;
    let mut out = String::new();
    for (idx, line) in source.lines().enumerate() {
        let line_no = idx + 1;
        if line_no < start {
            continue;
        }
        if line_no > end {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// True if any source line within `[start_line, end_line]` is longer than
/// `max_line_bytes`. Such a line is machine-generated (minified / bundled), not
/// human-authored code; embedding it is noise for semantic search and, because
/// many symbols can share one huge line, pathologically expensive to chunk.
fn span_has_overlong_line(
    source: &str,
    start_line: i64,
    end_line: i64,
    max_line_bytes: usize,
) -> bool {
    if start_line <= 0 {
        return false;
    }
    let start = (start_line as usize).saturating_sub(1);
    let end = if end_line >= start_line {
        (end_line as usize).saturating_sub(1)
    } else {
        start
    };
    for (idx, line) in source.lines().enumerate() {
        if idx < start {
            continue;
        }
        if idx > end {
            break;
        }
        if line.len() > max_line_bytes {
            return true;
        }
    }
    false
}

fn is_embedding_candidate_label(label: &str) -> bool {
    matches!(
        label,
        "Function"
            | "Method"
            | "Class"
            | "Interface"
            | "Type"
            | "Enum"
            | "Struct"
            | "Trait"
            | "TypeAlias"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_store::{NewNode, Project, VectorSearchQuery};

    struct DeterministicProvider;

    impl CodeEmbeddingProvider for DeterministicProvider {
        fn model_id(&self) -> &str {
            "test-code-embedder"
        }

        fn prompt_version(&self) -> &str {
            "test-prompt-v1"
        }

        fn task_profile(&self) -> &str {
            "embeddinggemma_code_retrieval"
        }

        fn embed_code_document(&mut self, _title: Option<&str>, content: &str) -> Result<Vec<f32>> {
            if content.contains("refund") {
                Ok(vec![0.0, 1.0])
            } else {
                Ok(vec![1.0, 0.0])
            }
        }
    }

    struct TokenBudgetProvider {
        max_tokens: usize,
        docs: std::rc::Rc<std::cell::RefCell<Vec<(String, String)>>>,
    }

    impl TokenBudgetProvider {
        fn new(max_tokens: usize) -> Self {
            Self {
                max_tokens,
                docs: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            }
        }

        fn token_count(&self, title: Option<&str>, content: &str) -> usize {
            EmbedTask::document_with_title(title, content)
                .split_whitespace()
                .count()
                .max(1)
        }
    }

    impl CodeEmbeddingProvider for TokenBudgetProvider {
        fn model_id(&self) -> &str {
            "test-code-embedder"
        }

        fn prompt_version(&self) -> &str {
            "test-prompt-v1"
        }

        fn task_profile(&self) -> &str {
            "embeddinggemma_code_retrieval"
        }

        fn embed_code_document(&mut self, title: Option<&str>, content: &str) -> Result<Vec<f32>> {
            self.docs
                .borrow_mut()
                .push((title.unwrap_or("none").to_string(), content.to_string()));
            Ok(test_vector_for_chunk(content))
        }

        fn embed_code_documents(&mut self, docs: &[(Option<&str>, &str)]) -> Result<Vec<Vec<f32>>> {
            let mut recorded = self.docs.borrow_mut();
            for (title, content) in docs {
                recorded.push((title.unwrap_or("none").to_string(), (*content).to_string()));
            }
            Ok(docs
                .iter()
                .map(|(_, content)| test_vector_for_chunk(content))
                .collect())
        }

        fn max_input_tokens(&self) -> usize {
            self.max_tokens
        }

        fn document_token_len(&self, title: Option<&str>, content: &str) -> Result<usize> {
            Ok(self.token_count(title, content))
        }
    }

    struct CharBudgetProvider {
        max_tokens: usize,
    }

    impl CodeEmbeddingProvider for CharBudgetProvider {
        fn model_id(&self) -> &str {
            "test-code-embedder"
        }

        fn prompt_version(&self) -> &str {
            "test-prompt-v1"
        }

        fn task_profile(&self) -> &str {
            "embeddinggemma_code_retrieval"
        }

        fn embed_code_document(&mut self, _title: Option<&str>, content: &str) -> Result<Vec<f32>> {
            Ok(test_vector_for_chunk(content))
        }

        fn max_input_tokens(&self) -> usize {
            self.max_tokens
        }

        fn document_token_len(&self, title: Option<&str>, content: &str) -> Result<usize> {
            Ok(EmbedTask::document_with_title(title, content)
                .chars()
                .count()
                .max(1))
        }
    }

    fn test_vector_for_chunk(content: &str) -> Vec<f32> {
        if content.contains("new_marker") {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        }
    }

    fn long_function_source(name: &str, body_lines: usize, marker_line: Option<usize>) -> String {
        let mut src = format!("pub fn {name}() {{\n");
        for i in 0..body_lines {
            if marker_line == Some(i) {
                src.push_str(&format!("    let value_{i} = new_marker_{i};\n"));
            } else {
                src.push_str(&format!("    let value_{i} = body_token_{i};\n"));
            }
        }
        src.push_str("}\n");
        src
    }

    fn tempdir_via_env() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "greppy-indexer-embedding-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let p = base.join(unique);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn store_with_project(root: &Path) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-07-01T00:00:00Z".into(),
            root_path: root.to_string_lossy().into_owned(),
        })
        .unwrap();
        s
    }

    fn insert_node(
        store: &mut Store,
        qname: &str,
        name: &str,
        label: &str,
        file_path: &str,
        start_line: i64,
        end_line: i64,
    ) {
        store
            .insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: name.into(),
                qualified_name: qname.into(),
                file_path: file_path.into(),
                start_line,
                end_line,
                properties: serde_json::json!({}),
            })
            .unwrap();
    }

    #[test]
    fn long_definition_indexes_multiple_header_prefixed_chunks_covering_body() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        let src = long_function_source("huge", 80, None);
        std::fs::write(root.join("src/lib.rs"), &src).unwrap();
        let last_line = src.lines().count() as i64;
        let title = "src/lib.rs:1-1 p.huge";
        let provider = TokenBudgetProvider::new(34);
        let chunks =
            embedding_chunks(&src, 1, 1, title, &provider, usize::MAX).expect("chunk long def");

        assert!(chunks.len() > 1, "expected multiple chunks, got {chunks:?}");
        assert_eq!(chunks.last().unwrap().end_line, last_line);
        assert_eq!(chunks[0].start_line, 1);
        for pair in chunks.windows(2) {
            assert!(
                pair[1].start_line <= pair[0].end_line + 1,
                "chunk line ranges must cover without gaps: {pair:?}"
            );
        }
        for chunk in &chunks {
            assert!(
                chunk.text.starts_with("pub fn huge() {\n"),
                "chunk did not start with header: {:?}",
                chunk.text
            );
        }

        let mut store = store_with_project(&root);
        insert_node(&mut store, "p.huge", "huge", "Function", "src/lib.rs", 1, 1);
        let mut provider = TokenBudgetProvider::new(34);
        let report = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(9),
        )
        .unwrap();
        assert_eq!(report.nodes_embedded, chunks.len());
        assert_eq!(
            store
                .count_vector_embeddings(
                    "p",
                    "test-code-embedder",
                    "test-prompt-v1",
                    "embeddinggemma_code_retrieval",
                    Some(9),
                )
                .unwrap(),
            chunks.len() as i64
        );
        let docs = provider.docs.borrow();
        assert_eq!(docs.len(), chunks.len());
        assert!(docs
            .iter()
            .all(|(_, content)| content.starts_with("pub fn huge() {\n")));
    }

    #[test]
    fn small_definition_produces_one_embedding_chunk() {
        let src = "pub fn tiny() {\n    body_token();\n}\n";
        let provider = TokenBudgetProvider::new(128);
        let chunks =
            embedding_chunks(src, 1, 1, "src/lib.rs:1-1 p.tiny", &provider, usize::MAX).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn embedding_chunks_respect_prompt_token_budget() {
        let src = long_function_source("budgeted", 60, None);
        let title = "src/lib.rs:1-1 p.budgeted";
        let provider = TokenBudgetProvider::new(31);
        let chunks = embedding_chunks(&src, 1, 1, title, &provider, usize::MAX).unwrap();
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            let tokens = provider
                .document_token_len(Some(title), &chunk.text)
                .unwrap();
            assert!(
                tokens <= provider.max_input_tokens(),
                "chunk {} exceeded budget: {tokens} > {}",
                chunk.chunk_idx,
                provider.max_input_tokens()
            );
        }
    }

    #[test]
    fn pathological_huge_header_chunks_stay_within_prompt_budget() {
        let title = "t";
        let provider = CharBudgetProvider { max_tokens: 96 };
        let header_tail = "H".repeat(240);
        let body = "B".repeat(240);
        let src = format!("pub fn pathological_{header_tail}() {{ {body} }}\n");
        assert!(
            provider.document_token_len(Some(title), &src).unwrap() > provider.max_input_tokens(),
            "test setup must exceed the provider budget"
        );

        let chunks = embedding_chunks(&src, 1, 1, title, &provider, usize::MAX).unwrap();
        assert!(chunks.len() > 1, "expected multiple chunks, got {chunks:?}");
        assert!(
            chunks.iter().any(|chunk| chunk.text.contains('B')),
            "chunks should make progress into the body"
        );
        for chunk in &chunks {
            assert!(
                chunk.text.starts_with("pub fn pathological_"),
                "chunk did not keep the truncated header prefix: {:?}",
                chunk.text
            );
            let tokens = provider
                .document_token_len(Some(title), &chunk.text)
                .unwrap();
            assert!(
                tokens <= provider.max_input_tokens(),
                "chunk {} exceeded budget: {tokens} > {}",
                chunk.chunk_idx,
                provider.max_input_tokens()
            );
        }
    }

    #[test]
    fn unclosed_definition_does_not_embed_next_definition() {
        let src = "\
pub fn broken() {
    let keep = 1;

pub fn next_definition() {
    unrelated();
}
";
        let title = "src/lib.rs:1-1 p.broken";
        let provider = TokenBudgetProvider::new(512);
        let chunks = embedding_chunks(src, 1, 1, title, &provider, usize::MAX).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].end_line, 2);
        assert!(chunks[0].text.contains("broken"));
        assert!(
            !chunks[0].text.contains("next_definition"),
            "malformed definition included the following definition: {:?}",
            chunks[0].text
        );
    }

    #[test]
    fn incremental_long_function_reembed_prunes_old_chunks_and_search_dedups_node() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        let src_v1 = long_function_source("evolving", 70, None);
        std::fs::write(root.join("src/lib.rs"), &src_v1).unwrap();
        let mut store = store_with_project(&root);
        insert_node(
            &mut store,
            "p.evolving",
            "evolving",
            "Function",
            "src/lib.rs",
            1,
            1,
        );

        let mut provider = TokenBudgetProvider::new(34);
        index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(1),
        )
        .unwrap();
        assert!(
            store
                .count_vector_embeddings(
                    "p",
                    "test-code-embedder",
                    "test-prompt-v1",
                    "embeddinggemma_code_retrieval",
                    Some(1),
                )
                .unwrap()
                > 1
        );

        let src_v2 = long_function_source("evolving", 70, Some(44));
        std::fs::write(root.join("src/lib.rs"), &src_v2).unwrap();
        let mut provider = TokenBudgetProvider::new(34);
        index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(2),
        )
        .unwrap();

        let hits = store
            .vector_search_exact(
                &[1.0, 0.0],
                &VectorSearchQuery {
                    project: "p",
                    model_id: "test-code-embedder",
                    prompt_version: "test-prompt-v1",
                    task: "embeddinggemma_code_retrieval",
                    graph_generation: Some(2),
                    file_path: None,
                    limit: 5,
                    min_score: None,
                },
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].embedding.qualified_name, "p.evolving");
        assert_eq!(hits[0].embedding.graph_generation, 2);
        assert_eq!(
            store
                .count_vector_embeddings(
                    "p",
                    "test-code-embedder",
                    "test-prompt-v1",
                    "embeddinggemma_code_retrieval",
                    Some(1),
                )
                .unwrap(),
            0
        );
    }

    struct FailingBatchProvider {
        failed: bool,
    }

    impl CodeEmbeddingProvider for FailingBatchProvider {
        fn model_id(&self) -> &str {
            "test-code-embedder"
        }

        fn prompt_version(&self) -> &str {
            "test-prompt-v1"
        }

        fn task_profile(&self) -> &str {
            "embeddinggemma_code_retrieval"
        }

        fn embed_code_document(
            &mut self,
            _title: Option<&str>,
            _content: &str,
        ) -> Result<Vec<f32>> {
            Err(Error::Store("intentional embedding failure".into()))
        }

        fn embed_code_documents(
            &mut self,
            _docs: &[(Option<&str>, &str)],
        ) -> Result<Vec<Vec<f32>>> {
            if !self.failed {
                self.failed = true;
                Err(Error::Store("intentional embedding failure".into()))
            } else {
                Ok(Vec::new())
            }
        }
    }

    #[test]
    fn embedding_batch_failure_fails_build_and_keeps_prior_generation() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn current() {}\n").unwrap();
        let mut store = store_with_project(&root);
        insert_node(
            &mut store,
            "p.current",
            "current",
            "Function",
            "src/lib.rs",
            1,
            1,
        );
        store
            .upsert_vector_embedding(&NewVectorEmbedding {
                project: "p".into(),
                model_id: "test-code-embedder".into(),
                prompt_version: "test-prompt-v1".into(),
                task: "embeddinggemma_code_retrieval".into(),
                node_id: None,
                chunk_idx: 0,
                qualified_name: "p.old".into(),
                file_path: "src/old.rs".into(),
                start_line: 1,
                end_line: 1,
                content_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    .into(),
                graph_generation: 1,
                vector: vec![1.0, 0.0],
            })
            .unwrap();

        let mut provider = FailingBatchProvider { failed: false };
        let error = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(5),
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("embedding generation incomplete"));
        assert_eq!(
            store
                .count_vector_embeddings(
                    "p",
                    "test-code-embedder",
                    "test-prompt-v1",
                    "embeddinggemma_code_retrieval",
                    Some(1)
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn indexes_real_symbol_spans_into_vector_store() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/payments.rs"),
            "pub fn charge() {\n    settle();\n}\n\npub fn refund_payment() {\n    refund();\n}\n",
        )
        .unwrap();
        let mut store = store_with_project(&root);
        insert_node(
            &mut store,
            "p.charge",
            "charge",
            "Function",
            "src/payments.rs",
            1,
            3,
        );
        insert_node(
            &mut store,
            "p.refund_payment",
            "refund_payment",
            "Function",
            "src/payments.rs",
            5,
            7,
        );
        insert_node(
            &mut store,
            "src/payments.rs::__file__",
            "__file__",
            "Module",
            "src/payments.rs",
            1,
            1,
        );

        let mut provider = DeterministicProvider;
        let report = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(4),
        )
        .unwrap();

        assert_eq!(report.nodes_considered, 3);
        assert_eq!(report.nodes_embedded, 2);
        assert_eq!(report.nodes_skipped_non_definition, 1);
        let hits = store
            .vector_search_exact(
                &[0.0, 1.0],
                &VectorSearchQuery {
                    project: "p",
                    model_id: "test-code-embedder",
                    prompt_version: "test-prompt-v1",
                    task: "embeddinggemma_code_retrieval",
                    graph_generation: Some(4),
                    file_path: None,
                    limit: 2,
                    min_score: None,
                },
            )
            .unwrap();
        assert_eq!(hits[0].embedding.qualified_name, "p.refund_payment");
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.embedding.graph_generation == 4));
    }

    #[test]
    fn embedding_index_prunes_only_after_successful_generation() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn current() {}\n").unwrap();
        let mut store = store_with_project(&root);
        insert_node(
            &mut store,
            "p.current",
            "current",
            "Function",
            "src/lib.rs",
            1,
            1,
        );
        store
            .upsert_vector_embedding(&NewVectorEmbedding {
                project: "p".into(),
                model_id: "test-code-embedder".into(),
                prompt_version: "test-prompt-v1".into(),
                task: "embeddinggemma_code_retrieval".into(),
                node_id: None,
                chunk_idx: 0,
                qualified_name: "p.old".into(),
                file_path: "src/old.rs".into(),
                start_line: 1,
                end_line: 1,
                content_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .into(),
                graph_generation: 1,
                vector: vec![1.0, 0.0],
            })
            .unwrap();

        let mut provider = DeterministicProvider;
        let report = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(5),
        )
        .unwrap();
        assert_eq!(report.nodes_embedded, 1);
        assert_eq!(report.stale_rows_pruned, 1);
        assert_eq!(
            store
                .count_vector_embeddings(
                    "p",
                    "test-code-embedder",
                    "test-prompt-v1",
                    "embeddinggemma_code_retrieval",
                    None
                )
                .unwrap(),
            1
        );
    }

    /// Provider whose per-document vector is a pure, unique function of the
    /// span content, and that records the size and prompt lengths of every
    /// batch call. Lets tests prove batching behavior while checking that each
    /// node still receives the vector for its own content.
    struct RecordingProvider {
        batch_sizes: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
        batch_prompt_lengths: std::rc::Rc<std::cell::RefCell<Vec<Vec<usize>>>>,
    }

    fn content_vector(content: &str) -> Vec<f32> {
        // Distinct, deterministic per content; not order-dependent.
        let mut h: u32 = 2166136261;
        for b in content.as_bytes() {
            h ^= u32::from(*b);
            h = h.wrapping_mul(16777619);
        }
        vec![(h & 0xffff) as f32, (h >> 16) as f32, content.len() as f32]
    }

    fn test_prompt_token_len(title: Option<&str>, content: &str) -> usize {
        EmbedTask::document_with_title(title, content)
            .split_whitespace()
            .count()
            .max(1)
    }

    impl CodeEmbeddingProvider for RecordingProvider {
        fn model_id(&self) -> &str {
            "test-code-embedder"
        }
        fn prompt_version(&self) -> &str {
            "test-prompt-v1"
        }
        fn task_profile(&self) -> &str {
            "embeddinggemma_code_retrieval"
        }
        fn embed_code_document(&mut self, title: Option<&str>, content: &str) -> Result<Vec<f32>> {
            self.batch_sizes.borrow_mut().push(1);
            self.batch_prompt_lengths
                .borrow_mut()
                .push(vec![test_prompt_token_len(title, content)]);
            Ok(content_vector(content))
        }
        // Override so the indexer's batched path is exercised, recording the
        // real batch size and returning a per-document vector in input order.
        fn embed_code_documents(&mut self, docs: &[(Option<&str>, &str)]) -> Result<Vec<Vec<f32>>> {
            self.batch_sizes.borrow_mut().push(docs.len());
            self.batch_prompt_lengths.borrow_mut().push(
                docs.iter()
                    .map(|(title, content)| test_prompt_token_len(*title, content))
                    .collect(),
            );
            Ok(docs.iter().map(|(_, c)| content_vector(c)).collect())
        }

        fn document_token_len(&self, title: Option<&str>, content: &str) -> Result<usize> {
            Ok(test_prompt_token_len(title, content))
        }
    }

    /// The batched index loop must (a) actually batch and (b) store, for every
    /// node, exactly the vector its own span content maps to — i.e. batching
    /// preserves per-node correctness and ordering versus the serial path.
    #[test]
    fn batched_index_loop_preserves_per_node_vectors() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        // 20 functions => spans multiple default (16) batches plus a remainder.
        let n = 20usize;
        let mut src = String::new();
        for i in 0..n {
            src.push_str(&format!("pub fn f{i}() {{\n    body_{i}();\n}}\n\n"));
        }
        std::fs::write(root.join("src/lib.rs"), &src).unwrap();
        let mut store = store_with_project(&root);
        for i in 0..n {
            let start = (i * 4 + 1) as i64;
            insert_node(
                &mut store,
                &format!("p.f{i}"),
                &format!("f{i}"),
                "Function",
                "src/lib.rs",
                start,
                start + 2,
            );
        }

        let batch_sizes = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut provider = RecordingProvider {
            batch_sizes: batch_sizes.clone(),
            batch_prompt_lengths: Default::default(),
        };
        let report = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(7),
        )
        .unwrap();
        assert_eq!(report.nodes_embedded, n);

        // Batching actually occurred: with default batch 16 and 20 nodes we
        // expect batches [16, 4] and never a size-1 (serial) call.
        let sizes = batch_sizes.borrow().clone();
        assert!(
            sizes.iter().sum::<usize>() == n && sizes.iter().all(|&s| s > 1),
            "expected multi-doc batches summing to {n}, got {sizes:?}"
        );

        // Every stored vector equals the vector for that node's own span.
        for i in 0..n {
            let start = (i * 4 + 1) as i64;
            let span = source_span(&src, start, start + 2).unwrap();
            let expected = content_vector(&span);
            let hits = store
                .vector_search_exact(
                    &expected,
                    &VectorSearchQuery {
                        project: "p",
                        model_id: "test-code-embedder",
                        prompt_version: "test-prompt-v1",
                        task: "embeddinggemma_code_retrieval",
                        graph_generation: Some(7),
                        file_path: None,
                        limit: 1,
                        min_score: None,
                    },
                )
                .unwrap();
            assert_eq!(
                hits[0].embedding.qualified_name,
                format!("p.f{i}"),
                "node f{i} did not get its own content vector"
            );
        }
    }

    #[test]
    fn dissimilar_prompt_lengths_are_sorted_into_nearby_batches() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        let mut src = String::new();
        let mut spans = Vec::new();
        for i in 0..(DEFAULT_EMBED_BATCH * 2) {
            let start_line = src.lines().count() as i64 + 1;
            src.push_str(&format!("pub fn varied_{i}() {{\n"));
            let body_lines = if i % 2 == 0 { 1 } else { 32 };
            for line in 0..body_lines {
                src.push_str(&format!("    let value_{i}_{line} = token_{i}_{line};\n"));
            }
            src.push_str("}\n\n");
            let end_line = src.lines().count() as i64 - 1;
            spans.push((i, start_line, end_line));
        }
        std::fs::write(root.join("src/lib.rs"), &src).unwrap();

        let mut store = store_with_project(&root);
        for &(i, start_line, end_line) in &spans {
            insert_node(
                &mut store,
                &format!("p.varied_{i}"),
                &format!("varied_{i}"),
                "Function",
                "src/lib.rs",
                start_line,
                end_line,
            );
        }

        let batch_sizes = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let batch_prompt_lengths = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut provider = RecordingProvider {
            batch_sizes: batch_sizes.clone(),
            batch_prompt_lengths: batch_prompt_lengths.clone(),
        };
        let report = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(8),
        )
        .unwrap();
        assert_eq!(report.nodes_embedded, DEFAULT_EMBED_BATCH * 2);

        let batches = batch_prompt_lengths.borrow();
        assert_eq!(
            batches.iter().map(Vec::len).sum::<usize>(),
            DEFAULT_EMBED_BATCH * 2
        );
        assert_eq!(batches.len(), 2, "expected two full batches: {batches:?}");
        for lengths in batches.iter() {
            assert!(
                lengths.windows(2).all(|pair| pair[0] <= pair[1]),
                "one forward was not sorted by prompt length: {lengths:?}"
            );
        }
        assert!(
            batches[0].last().unwrap() < batches[1].first().unwrap(),
            "short and long prompts were mixed across forwards: {batches:?}"
        );

        for &(i, start_line, end_line) in &spans {
            let span = source_span(&src, start_line, end_line).unwrap();
            let hits = store
                .vector_search_exact(
                    &content_vector(&span),
                    &VectorSearchQuery {
                        project: "p",
                        model_id: "test-code-embedder",
                        prompt_version: "test-prompt-v1",
                        task: "embeddinggemma_code_retrieval",
                        graph_generation: Some(8),
                        file_path: None,
                        limit: 1,
                        min_score: None,
                    },
                )
                .unwrap();
            assert_eq!(hits[0].embedding.qualified_name, format!("p.varied_{i}"));
        }
    }

    /// The `GREPPY_EMBED_BATCH` override is honored and clamps to >= 1.
    #[test]
    fn embed_batch_env_override_is_parsed_and_clamped() {
        assert_eq!(super::parse_embed_batch_size(Some("4")), 4);
        assert_eq!(
            super::parse_embed_batch_size(Some("0")),
            super::DEFAULT_EMBED_BATCH
        );
        assert_eq!(
            super::parse_embed_batch_size(Some("garbage")),
            super::DEFAULT_EMBED_BATCH
        );
        assert_eq!(
            super::parse_embed_batch_size(None),
            super::DEFAULT_EMBED_BATCH
        );
    }
}
