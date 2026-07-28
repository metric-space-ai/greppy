//! Training-free `bash-smart`: byte-preserving command capture, mechanical
//! skeletons, arithmetic repetition collapse, optional embedding novelty, and
//! hash-guarded paged expansion.

use super::*;
use sha2::{Digest, Sha256};
use std::io::Write;

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

#[derive(Clone)]
struct StoredRaw {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    content_sha256: String,
    payload: serde_json::Value,
}

pub(crate) fn run(argv: &[String], root: Option<&str>) -> Result<i32> {
    if argv.is_empty() {
        return Err(Error::Invalid(
            "bash-smart requires a command after `--`".into(),
        ));
    }

    let mut command = command_for_argv(argv);
    let output = match command.output() {
        Ok(output) => output,
        Err(error) => {
            let _ = writeln!(
                std::io::stderr(),
                "bash-smart: failed to run {}: {error}",
                argv[0]
            );
            return Ok(if error.kind() == std::io::ErrorKind::NotFound {
                127
            } else {
                126
            });
        }
    };
    let exit_code = child_exit_code(&output.status);
    let raw = StoredRaw::new(output.stdout, output.stderr);
    let stdout_lines = split_lines(&raw.stdout);
    let stderr_lines = split_lines(&raw.stderr);

    // Every invocation is put in the expand economy, including short and empty
    // output. Short output deliberately prints no id: storage is durability,
    // not an invitation to spend another call.
    let store = open_pack_store(root).ok();
    let project = project_for(root).unwrap_or_else(|_| "bash-smart".into());
    let query = argv_for_metadata(argv);

    if stdout_lines.len() + stderr_lines.len() <= SHORT_TOTAL_LINES {
        if let Some(store) = store.as_ref() {
            let _ = insert_pack(store, &project, &query, &raw, "stdout", 1);
        }
        write_stream(false, &raw.stdout);
        write_stream(true, &raw.stderr);
        return Ok(exit_code);
    }

    let stdout_folded = stdout_lines.len() > SHORT_TOTAL_LINES;
    let stderr_folded = stderr_lines.len() > STDERR_VERBATIM_LINES;
    let stdout_id = store.as_ref().and_then(|store| {
        insert_pack(store, &project, &query, &raw, "stdout", HEAD_LINES + 1).ok()
    });
    let stderr_id = if stderr_folded {
        store.as_ref().and_then(|store| {
            insert_pack(store, &project, &query, &raw, "stderr", HEAD_LINES + 1).ok()
        })
    } else {
        None
    };

    let lifted_stdout = if stdout_folded {
        novelty_lifts(&stdout_lines, root)
    } else {
        Vec::new()
    };
    let lifted_stderr = if stderr_folded {
        novelty_lifts(&stderr_lines, root)
    } else {
        Vec::new()
    };

    if stdout_folded {
        if let (Some(store), Some(id)) = (store.as_ref(), stdout_id.as_deref()) {
            let gated = byte_gate(store, id, "stdout", &lifted_stdout);
            render_folded(false, &stdout_lines, exit_code, id, &gated);
        } else {
            // A missing pack must never turn truncation into data loss.
            write_stream(false, &raw.stdout);
        }
    } else {
        write_stream(false, &raw.stdout);
    }

    if stderr_folded {
        if let (Some(store), Some(id)) = (store.as_ref(), stderr_id.as_deref()) {
            let gated = byte_gate(store, id, "stderr", &lifted_stderr);
            render_folded(true, &stderr_lines, exit_code, id, &gated);
        } else {
            write_stream(true, &raw.stderr);
        }
    } else {
        write_stream(true, &raw.stderr);
    }

    Ok(exit_code)
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
    let start = pack
        .summary_json
        .get("start_line")
        .and_then(serde_json::Value::as_u64)
        .and_then(|line| usize::try_from(line).ok())
        .unwrap_or(1)
        .max(1);
    let bytes = if stream == "stderr" {
        &raw.stderr
    } else {
        &raw.stdout
    };
    let lines = split_lines(bytes);
    let begin = start.saturating_sub(1).min(lines.len());
    let end = begin.saturating_add(EXPAND_PAGE_LINES).min(lines.len());
    let next_line = end + 1;

    let next = if end < lines.len() {
        insert_continuation_pack(store, &pack, &raw, stream, next_line).ok()
    } else {
        None
    };

    if json {
        let page = lines[begin..end]
            .iter()
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
                "end_line": end,
                "line_count": lines.len(),
                "raw_line_hex": page,
                "next": next.as_ref().map(|id| serde_json::json!({
                    "id": id,
                    "line": next_line,
                    "command": format!("greppy expand {id}"),
                })),
            }))
            .map_err(|error| Error::Invalid(format!("serialize bash-smart expand: {error}")))?
        );
        return Ok(0);
    }

    let mut stdout = std::io::stdout().lock();
    for line in &lines[begin..end] {
        let _ = stdout.write_all(line.raw);
    }
    if end > begin && !lines[end - 1].raw.ends_with(b"\n") {
        let _ = stdout.write_all(b"\n");
    }
    if end < lines.len() {
        if let Some(next_id) = next {
            let remaining = lines.len() - end;
            let _ = writeln!(
                stdout,
                "… {remaining} lines — greppy expand {next_id} continues at {next_line}"
            );
        } else {
            // If allocating the next page fails, deliver it now rather than
            // leave a continuation that cannot be opened.
            for line in &lines[end..] {
                let _ = stdout.write_all(line.raw);
            }
        }
    }
    Ok(0)
}

fn command_for_argv(argv: &[String]) -> std::process::Command {
    if argv.len() == 1 {
        #[cfg(windows)]
        {
            let mut command = std::process::Command::new("cmd");
            command.arg("/C").arg(&argv[0]);
            command
        }
        #[cfg(not(windows))]
        {
            let mut command = std::process::Command::new("sh");
            command.arg("-c").arg(&argv[0]);
            command
        }
    } else {
        let mut command = std::process::Command::new(&argv[0]);
        command.args(&argv[1..]);
        command
    }
}

fn child_exit_code(status: &std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        return status.signal().map(|signal| 128 + signal).unwrap_or(1);
    }
    #[cfg(not(unix))]
    {
        1
    }
}

impl StoredRaw {
    fn new(stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        let stdout_sha256 = sha256(&stdout);
        let stderr_sha256 = sha256(&stderr);
        let content_sha256 = combined_sha256(&stdout, &stderr);
        let payload = serde_json::json!({
            "schema_version": PACK_SCHEMA_VERSION,
            "kind": "bash-smart",
            "content_sha256": content_sha256,
            "stdout": {
                "sha256": stdout_sha256,
                "byte_len": stdout.len(),
                "line_count": split_lines(&stdout).len(),
                "hex": hex_encode(&stdout),
            },
            "stderr": {
                "sha256": stderr_sha256,
                "byte_len": stderr.len(),
                "line_count": split_lines(&stderr).len(),
                "hex": hex_encode(&stderr),
            },
        });
        Self {
            stdout,
            stderr,
            content_sha256,
            payload,
        }
    }

    fn decode(payload: &serde_json::Value) -> Option<Self> {
        if payload.get("schema_version")?.as_u64()? != PACK_SCHEMA_VERSION
            || payload.get("kind")?.as_str()? != "bash-smart"
        {
            return None;
        }
        let stdout = hex_decode(payload.get("stdout")?.get("hex")?.as_str()?)?;
        let stderr = hex_decode(payload.get("stderr")?.get("hex")?.as_str()?)?;
        let decoded = Self::new(stdout, stderr);
        let claimed = payload.get("content_sha256")?.as_str()?;
        let stdout_claimed = payload.get("stdout")?.get("sha256")?.as_str()?;
        let stderr_claimed = payload.get("stderr")?.get("sha256")?.as_str()?;
        (decoded.content_sha256 == claimed
            && sha256(&decoded.stdout) == stdout_claimed
            && sha256(&decoded.stderr) == stderr_claimed)
            .then_some(decoded)
    }
}

fn open_pack_store(root: Option<&str>) -> Result<greppy_store::Store> {
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
    start_line: usize,
) -> std::result::Result<String, greppy_store::Error> {
    let summary = serde_json::json!({
        "text": format!("bash-smart {stream} raw output"),
        "kind": "bash-smart",
        "stream": stream,
        "start_line": start_line,
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
    start_line: usize,
) -> std::result::Result<String, greppy_store::Error> {
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

fn render_folded(
    stderr: bool,
    lines: &[RawLine<'_>],
    exit_code: i32,
    id: &str,
    lifted: &[LiftedLine],
) {
    let tail = if exit_code == 0 {
        SUCCESS_TAIL_LINES
    } else {
        FAILURE_TAIL_LINES
    };
    let groups = collapse_groups(lines);
    let mut repeated_line = vec![false; lines.len()];
    for group in groups.iter().filter(|group| group.count() > 1) {
        for index in group.start - 1..group.end {
            repeated_line[index] = true;
        }
    }

    let mut writer: Box<dyn Write> = if stderr {
        Box::new(std::io::stderr().lock())
    } else {
        Box::new(std::io::stdout().lock())
    };
    let head_end = HEAD_LINES.min(lines.len());
    for index in 0..head_end {
        if !repeated_line[index] {
            let _ = writer.write_all(lines[index].raw);
        }
    }
    ensure_newline_after_raw(
        &mut writer,
        lines,
        (0..head_end).rev().find(|i| !repeated_line[*i]),
    );

    enum Block<'a> {
        Collapse(&'a CollapseGroup),
        Lift(&'a LiftedLine),
    }
    let mut block = groups
        .iter()
        .filter(|group| group.count() > 1)
        .map(|group| (group.start, Block::Collapse(group)))
        .chain(lifted.iter().map(|line| (line.line, Block::Lift(line))))
        .collect::<Vec<_>>();
    block.sort_by_key(|(line, _)| *line);
    block.dedup_by(|left, right| left.0 == right.0);
    for (_, item) in block {
        match item {
            Block::Collapse(group) => {
                let _ = writer.write_all(&group.representative);
                if !group.representative.ends_with(b"\n") {
                    let _ = writer.write_all(b"\n");
                }
                let _ = writeln!(
                    writer,
                    "… {} weitere `{}`-Zeilen",
                    group.count() - 1,
                    group.template
                );
            }
            Block::Lift(line) => {
                let _ = write!(writer, "{}:", line.line);
                let _ = writer.write_all(&line.bytes);
                let _ = writer.write_all(b"\n");
            }
        }
    }

    let middle = lines.len().saturating_sub(HEAD_LINES + tail);
    if middle > 0 {
        let _ = writeln!(
            writer,
            "… {middle} lines — greppy expand {id} continues at {}",
            HEAD_LINES + 1
        );
    }

    let tail_start = lines.len().saturating_sub(tail).max(head_end);
    for index in tail_start..lines.len() {
        if !repeated_line[index] {
            let _ = writer.write_all(lines[index].raw);
        }
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

fn novelty_lifts(lines: &[RawLine<'_>], root: Option<&str>) -> Vec<LiftedLine> {
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
        let cache_dir = resolve_root(root)
            .ok()
            .map(|path| workspace_locator::store_dir(&path));
        let Ok(model) = load_embedding_model(&cfg, cache_dir) else {
            return Vec::new();
        };
        let groups = collapse_groups(lines);
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
            let Ok(vectors) = model.embed_documents(&docs) else {
                return Vec::new();
            };
            if vectors.len() != valid.len() {
                return Vec::new();
            }
            for ((index, _), vector) in valid.into_iter().zip(vectors) {
                embedded.push((groups_index(&groups, chunk, index), vector));
            }
        }
        rank_novelty(lines, &groups, &embedded)
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
            bytes: groups[group_index].representative.clone(),
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
    let path_re = regex::Regex::new(r#"(?x)(?:[A-Za-z]:[\\/]|\.{0,2}/|/)[^\s`'\"]+"#).ok()?;
    let hex_re = regex::Regex::new(r"(?i)\b(?:0x)?[0-9a-f]{6,}\b").ok()?;
    let digits_re = regex::Regex::new(r"\d+").ok()?;
    let masked = path_re.replace_all(text, "<PATH>");
    let masked = hex_re.replace_all(&masked, "<HEX>");
    let masked = digits_re.replace_all(&masked, "<N>");
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
            let mut content_end = index;
            if content_end > start && bytes[content_end - 1] == b'\r' {
                content_end -= 1;
            }
            lines.push(RawLine {
                content: &bytes[start..content_end],
                raw: &bytes[start..=index],
            });
            start = index + 1;
        }
    }
    if start < bytes.len() {
        lines.push(RawLine {
            content: &bytes[start..],
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

fn hex_decode(encoded: &str) -> Option<Vec<u8>> {
    if encoded.len() % 2 != 0 {
        return None;
    }
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_nibble(pair[0])?;
            let low = hex_nibble(pair[1])?;
            Some((high << 4) | low)
        })
        .collect()
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
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
    fn split_lines_preserves_exact_line_bytes() {
        let raw = b"a\r\nb\nlast";
        let lines = split_lines(raw);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].content, b"a");
        assert_eq!(lines[0].raw, b"a\r\n");
        assert_eq!(lines[2].raw, b"last");
    }
}
