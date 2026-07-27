//! Structured-data editing: `data set` / `data ensure` for JSON, TOML and
//! YAML, format-preserving.
//!
//! JSON: a span tokenizer locates the VALUE bytes of a `$.a.b[2].c` path and
//! replaces exactly those bytes — every other byte (whitespace, key order,
//! comments in JSON5-ish files are out of scope) stays untouched.
//! TOML: `toml_edit` validates the document while a span locator replaces only
//! the selected scalar bytes, preserving comments and decoration verbatim.
//! YAML: scalar values addressed by mapping path are replaced in-line; block
//! scalars and sequences refuse rather than reformatting the document.

use std::path::Path;

use crate::certificate::{Certificate, SelectorClass, SelectorEngine, Status};
use crate::txn::{PlannedOp, Snapshot};
use crate::verbs::{
    planned_precondition_refusal_for, run_pipeline_public, single_refusal_certificate,
    single_status_certificate, VerbOptions,
};
use greppy_core::{Error, Result};

/// Path segment of a `$.a.b[2].c` style path.
#[derive(Debug, Clone, PartialEq)]
enum Seg {
    Key(String),
    Index(usize),
}

fn parse_path(path: &str) -> Result<Vec<Seg>> {
    let body = path
        .strip_prefix("$.")
        .ok_or_else(|| Error::Invalid(format!("data path must start with `$.`: {path}")))?;
    if body.is_empty() || body.ends_with('.') || body.contains("..") {
        return Err(Error::Invalid(format!("invalid data path: {path}")));
    }
    let mut out = Vec::new();
    for raw in body.split('.') {
        if raw.is_empty() {
            return Err(Error::Invalid(format!("invalid data path: {path}")));
        }
        let mut rest = raw;
        if let Some(idx) = rest.find('[') {
            let (key, brackets) = rest.split_at(idx);
            if key.is_empty() {
                return Err(Error::Invalid(format!(
                    "missing key before index in path: {path}"
                )));
            }
            out.push(Seg::Key(key.to_string()));
            rest = brackets;
            while let Some(inner) = rest.strip_prefix('[') {
                let end = inner
                    .find(']')
                    .ok_or_else(|| Error::Invalid(format!("unclosed index in path: {path}")))?;
                let n: usize = inner[..end]
                    .parse()
                    .map_err(|_| Error::Invalid(format!("non-numeric index in path: {path}")))?;
                out.push(Seg::Index(n));
                rest = &inner[end + 1..];
            }
            if !rest.is_empty() {
                return Err(Error::Invalid(format!(
                    "invalid index suffix in path: {path}"
                )));
            }
        } else {
            out.push(Seg::Key(rest.to_string()));
        }
    }
    Ok(out)
}

#[derive(Debug)]
enum PathLookup {
    Missing,
    Ambiguous(usize),
    Unique(usize, usize, String),
}

/// `greppy edit data set|ensure`.
pub fn data_set(
    workspace_root: &Path,
    file: &Path,
    path: &str,
    value_json: &str,
    _ensure: bool,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    if let Some(certificate) = planned_precondition_refusal_for(
        workspace_root,
        &snapshot,
        options,
        SelectorEngine::DataPath,
        SelectorClass::StructuredData,
    ) {
        return Ok(certificate);
    }
    let segs = parse_path(path)?;
    let new_value: serde_json::Value = serde_json::from_str(value_json)
        .map_err(|e| Error::Invalid(format!("--value-json is not valid JSON: {e}")))?;

    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let lookup = match ext.as_str() {
        "json" => {
            serde_json::from_slice::<serde_json::Value>(&snapshot.content)
                .map_err(|e| Error::Parse(format!("JSON parse: {e}")))?;
            classify_spans(
                json_value_spans(&snapshot.content, &segs),
                value_json.trim().to_string(),
            )
        }
        "toml" => {
            // Parse first so malformed input is never treated as a path miss.
            std::str::from_utf8(&snapshot.content)
                .map_err(|e| Error::Parse(format!("TOML is not UTF-8: {e}")))?
                .parse::<toml_edit::DocumentMut>()
                .map_err(|e| Error::Parse(format!("TOML parse: {e}")))?;
            classify_spans(
                toml_value_spans(&snapshot.content, &segs),
                json_to_toml(&new_value)?.to_string(),
            )
        }
        "yaml" | "yml" => classify_spans(
            yaml_scalar_spans(&snapshot.content, &segs),
            yaml_scalar(&new_value),
        ),
        _ => {
            return Err(Error::Invalid(format!(
                "data edits support .json, .yaml/.yml, and .toml files; got {}",
                file.display()
            )))
        }
    };

    let (start, end, new_text) = match lookup {
        PathLookup::Missing => {
            return Ok(single_refusal_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::DataPath,
                SelectorClass::StructuredData,
                Status::NotFound,
                options,
            ))
        }
        PathLookup::Ambiguous(matches) => {
            return Ok(single_status_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::DataPath,
                SelectorClass::StructuredData,
                Status::Ambiguous,
                matches,
                options,
            ))
        }
        PathLookup::Unique(start, end, replacement) => (start, end, replacement),
    };

    if snapshot.content[start..end] == *new_text.as_bytes() {
        return Ok(single_status_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::DataPath,
            SelectorClass::StructuredData,
            Status::AlreadySatisfied,
            1,
            options,
        ));
    }
    let ops = vec![PlannedOp {
        id: "data-set".into(),
        range: (start, end),
        replacement: new_text.into_bytes(),
    }];

    // Validate the projected document before publication. Data parsers are not
    // tree-sitter postconditions, so this check must happen before the shared
    // pipeline writes anything.
    let projected = crate::txn::apply_in_memory(&snapshot, &ops)?;
    match ext.as_str() {
        "json" => {
            serde_json::from_slice::<serde_json::Value>(&projected.content)
                .map_err(|e| Error::Invalid(format!("data set would produce invalid JSON: {e}")))?;
        }
        "toml" => {
            std::str::from_utf8(&projected.content)
                .map_err(|e| Error::Invalid(format!("data set would produce non-UTF-8 TOML: {e}")))?
                .parse::<toml_edit::DocumentMut>()
                .map_err(|e| Error::Invalid(format!("data set would produce invalid TOML: {e}")))?;
        }
        _ => {}
    }

    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::DataPath,
        SelectorClass::StructuredData,
        None,
        options,
    )
}

/// Remove the entry a data path points at.
///
/// Deleting is not "replace the value with nothing": the key and the punctuation
/// that joins the entry to its neighbours have to go too, or the document is left
/// malformed. The span from `*_value_spans` covers only the value, so it is widened
/// to the whole entry here, and the projected document is parsed before anything is
/// written — the same guard `data_set` uses, for the same reason.
pub fn data_delete(
    workspace_root: &Path,
    file: &Path,
    path: &str,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    if let Some(certificate) = planned_precondition_refusal_for(
        workspace_root,
        &snapshot,
        options,
        SelectorEngine::DataPath,
        SelectorClass::StructuredData,
    ) {
        return Ok(certificate);
    }
    let segs = parse_path(path)?;

    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let value_spans = match ext.as_str() {
        "json" => {
            serde_json::from_slice::<serde_json::Value>(&snapshot.content)
                .map_err(|e| Error::Parse(format!("JSON parse: {e}")))?;
            json_value_spans(&snapshot.content, &segs)
        }
        "toml" => {
            std::str::from_utf8(&snapshot.content)
                .map_err(|e| Error::Parse(format!("TOML is not UTF-8: {e}")))?
                .parse::<toml_edit::DocumentMut>()
                .map_err(|e| Error::Parse(format!("TOML parse: {e}")))?;
            toml_value_spans(&snapshot.content, &segs)
        }
        "yaml" | "yml" => yaml_scalar_spans(&snapshot.content, &segs),
        _ => {
            return Err(Error::Invalid(format!(
                "data edits support .json, .yaml/.yml, and .toml files; got {}",
                file.display()
            )))
        }
    };

    let (start, end) = match value_spans.as_slice() {
        [] => {
            return Ok(single_refusal_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::DataPath,
                SelectorClass::StructuredData,
                Status::NotFound,
                options,
            ))
        }
        [(start, end)] => (*start, *end),
        many => {
            return Ok(single_status_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::DataPath,
                SelectorClass::StructuredData,
                Status::Ambiguous,
                many.len(),
                options,
            ))
        }
    };

    let (start, end) = widen_to_entry(&snapshot.content, start, end, &ext);
    let ops = vec![PlannedOp {
        id: "data-delete".into(),
        range: (start, end),
        replacement: Vec::new(),
    }];

    let projected = crate::txn::apply_in_memory(&snapshot, &ops)?;
    match ext.as_str() {
        "json" => {
            serde_json::from_slice::<serde_json::Value>(&projected.content).map_err(|e| {
                Error::Invalid(format!("data delete would produce invalid JSON: {e}"))
            })?;
        }
        "toml" => {
            std::str::from_utf8(&projected.content)
                .map_err(|e| {
                    Error::Invalid(format!("data delete would produce non-UTF-8 TOML: {e}"))
                })?
                .parse::<toml_edit::DocumentMut>()
                .map_err(|e| {
                    Error::Invalid(format!("data delete would produce invalid TOML: {e}"))
                })?;
        }
        _ => {}
    }

    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::DataPath,
        SelectorClass::StructuredData,
        None,
        options,
    )
}

/// Widen a value span to the whole entry that holds it.
///
/// JSON and TOML: take the key in front of the value, then one separator — a
/// trailing comma if there is one, otherwise a leading one, so the neighbours stay
/// joined. YAML: take the line, since an entry there is a line.
fn widen_to_entry(content: &[u8], start: usize, end: usize, ext: &str) -> (usize, usize) {
    let Ok(text) = std::str::from_utf8(content) else {
        return (start, end);
    };
    if ext == "yaml" || ext == "yml" {
        let line_start = text[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = text[end..]
            .find('\n')
            .map(|i| end + i + 1)
            .unwrap_or(text.len());
        return (line_start, line_end);
    }

    // Walk back over the separator and the key to the start of the entry.
    let mut key_start = start;
    for (index, ch) in text[..start].char_indices().rev() {
        if matches!(ch, '{' | '[' | ',' | '\n') {
            key_start = index + ch.len_utf8();
            break;
        }
        key_start = index;
    }
    let entry_start = text[..key_start]
        .rfind(|c: char| !c.is_whitespace())
        .map(|i| i + 1)
        .unwrap_or(key_start)
        .min(key_start);

    // Prefer the trailing comma; fall back to the leading one so the document does
    // not end up with two separators in a row.
    let after = text[end..]
        .char_indices()
        .find(|(_, c)| !c.is_whitespace())
        .map(|(i, c)| (end + i, c));
    if let Some((index, ',')) = after {
        return (entry_start, index + 1);
    }
    let before = text[..entry_start]
        .char_indices()
        .rev()
        .find(|(_, c)| !c.is_whitespace())
        .map(|(i, c)| (i, c));
    if let Some((index, ',')) = before {
        return (index, end);
    }
    (entry_start, end)
}

fn classify_spans(spans: Vec<(usize, usize)>, replacement: String) -> PathLookup {
    match spans.as_slice() {
        [] => PathLookup::Missing,
        [(start, end)] => PathLookup::Unique(*start, *end, replacement),
        many => PathLookup::Ambiguous(many.len()),
    }
}

fn json_to_toml(v: &serde_json::Value) -> Result<toml_edit::Item> {
    Ok(match v {
        serde_json::Value::String(s) => toml_edit::value(s.as_str()),
        serde_json::Value::Bool(b) => toml_edit::value(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml_edit::value(i)
            } else {
                toml_edit::value(n.as_f64().unwrap_or(0.0))
            }
        }
        _ => {
            return Err(Error::Invalid(
                "TOML data set supports scalar values only".into(),
            ))
        }
    })
}

fn toml_value_spans(content: &[u8], segs: &[Seg]) -> Vec<(usize, usize)> {
    let wanted: Vec<String> = segs
        .iter()
        .map(|seg| match seg {
            Seg::Key(key) => Ok(key.clone()),
            Seg::Index(_) => Err(()),
        })
        .collect::<std::result::Result<_, _>>()
        .unwrap_or_default();
    if wanted.len() != segs.len() {
        return Vec::new();
    }
    let Ok(text) = std::str::from_utf8(content) else {
        return Vec::new();
    };
    let mut section: Vec<String> = Vec::new();
    let mut offset = 0usize;
    let mut out = Vec::new();
    for line in text.split_inclusive('\n') {
        let stripped = line.trim_end_matches(['\r', '\n']);
        let body = stripped.trim_start();
        let body_offset = stripped.len() - body.len();
        if body.starts_with('[') && !body.starts_with("[[") {
            if let Some(end) = find_unquoted(body, ']') {
                section = parse_dotted_keys(&body[1..end]);
            }
            offset += line.len();
            continue;
        }
        if body.is_empty() || body.starts_with('#') {
            offset += line.len();
            continue;
        }
        let Some(equal) = find_unquoted(body, '=') else {
            offset += line.len();
            continue;
        };
        let mut path = section.clone();
        path.extend(parse_dotted_keys(body[..equal].trim()));
        if path == wanted {
            let rest = &body[equal + 1..];
            let leading = rest.len() - rest.trim_start().len();
            let value_and_comment = &rest[leading..];
            let comment = find_unquoted(value_and_comment, '#').unwrap_or(value_and_comment.len());
            let value = value_and_comment[..comment].trim_end();
            if !value.is_empty() {
                let start = offset + body_offset + equal + 1 + leading;
                out.push((start, start + value.len()));
            }
        }
        offset += line.len();
    }
    out
}

fn parse_dotted_keys(text: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let mut start = 0usize;
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && ch == '\\' {
            escaped = true;
            continue;
        }
        match (quote, ch) {
            (None, '"' | '\'') => quote = Some(ch),
            (Some(current), found) if current == found => quote = None,
            (None, '.') => {
                keys.push(unquote_key(text[start..index].trim()).to_string());
                start = index + 1;
            }
            _ => {}
        }
    }
    keys.push(unquote_key(text[start..].trim()).to_string());
    keys
}

fn unquote_key(key: &str) -> &str {
    key.strip_prefix('"')
        .and_then(|key| key.strip_suffix('"'))
        .or_else(|| {
            key.strip_prefix('\'')
                .and_then(|key| key.strip_suffix('\''))
        })
        .unwrap_or(key)
}

fn find_unquoted(text: &str, needle: char) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && ch == '\\' {
            escaped = true;
            continue;
        }
        match (quote, ch) {
            (None, '"' | '\'') => quote = Some(ch),
            (Some(current), found) if current == found => quote = None,
            (None, found) if found == needle => return Some(index),
            _ => {}
        }
    }
    None
}

fn yaml_scalar(v: &serde_json::Value) -> String {
    // JSON scalar/flow syntax is valid YAML 1.2 and quoting strings prevents
    // `#`, `:`, booleans, and null-like words from changing meaning.
    v.to_string()
}

/// Locate all plain/quoted scalar values at a YAML mapping path. Indentation is
/// inferred from the document rather than fixed at two spaces. Inline comments
/// are excluded from the replacement span.
fn yaml_scalar_spans(content: &[u8], segs: &[Seg]) -> Vec<(usize, usize)> {
    if segs.iter().any(|seg| matches!(seg, Seg::Index(_))) {
        return Vec::new();
    }
    let Ok(text) = std::str::from_utf8(content) else {
        return Vec::new();
    };
    let wanted: Vec<&str> = segs
        .iter()
        .filter_map(|seg| match seg {
            Seg::Key(key) => Some(key.as_str()),
            Seg::Index(_) => None,
        })
        .collect();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut offset = 0usize;
    let mut out = Vec::new();
    for line in text.split_inclusive('\n') {
        let stripped = line.trim_end_matches(['\r', '\n']);
        let indent = stripped.len() - stripped.trim_start().len();
        let body = stripped.trim_start();
        if body.is_empty() || body.starts_with('#') || body.starts_with('-') {
            offset += line.len();
            continue;
        }
        while stack.last().is_some_and(|(level, _)| *level >= indent) {
            stack.pop();
        }
        let Some(colon) = find_unquoted(body, ':') else {
            offset += line.len();
            continue;
        };
        let key = unquote_key(body[..colon].trim());
        let rest = &body[colon + 1..];
        let mut path: Vec<&str> = stack.iter().map(|(_, key)| key.as_str()).collect();
        path.push(key);
        if path == wanted {
            let leading = rest.len() - rest.trim_start().len();
            let value_and_comment = &rest[leading..];
            let comment = find_unquoted(value_and_comment, '#').unwrap_or(value_and_comment.len());
            let value = value_and_comment[..comment].trim_end();
            if !value.is_empty() && !value.starts_with('|') && !value.starts_with('>') {
                let body_offset = stripped.len() - body.len();
                let start = offset + body_offset + colon + 1 + leading;
                out.push((start, start + value.len()));
            }
        }
        if rest.trim().is_empty() {
            stack.push((indent, key.to_string()));
        }
        offset += line.len();
    }
    out
}

/// Locate every JSON value matching `segs`. Duplicate object keys therefore
/// produce an ambiguous structured selector instead of silently choosing one.
fn json_value_spans(content: &[u8], segs: &[Seg]) -> Vec<(usize, usize)> {
    let Ok(text) = std::str::from_utf8(content) else {
        return Vec::new();
    };
    let Some(root) = skip_ws(text, 0) else {
        return Vec::new();
    };
    let mut positions = vec![root];
    for seg in segs {
        let mut next = Vec::new();
        for pos in positions {
            match seg {
                Seg::Key(key) => next.extend(json_object_values(text, pos, key)),
                Seg::Index(index) => {
                    if let Some(value) = json_array_value(text, pos, *index) {
                        next.push(value);
                    }
                }
            }
        }
        positions = next;
        if positions.is_empty() {
            break;
        }
    }
    positions
        .into_iter()
        .filter_map(|start| skip_json_value(text, start).map(|end| (start, end)))
        .collect()
}

fn json_object_values(text: &str, mut pos: usize, wanted: &str) -> Vec<usize> {
    let mut out = Vec::new();
    if text.as_bytes().get(pos) != Some(&b'{') {
        return out;
    }
    pos += 1;
    while let Some(key_start) = skip_ws(text, pos) {
        if text.as_bytes().get(key_start) == Some(&b'}') {
            break;
        }
        let Some((key, after_key)) = parse_json_string(text, key_start) else {
            break;
        };
        let Some(colon) = skip_ws(text, after_key) else {
            break;
        };
        if text.as_bytes().get(colon) != Some(&b':') {
            break;
        }
        let Some(value_start) = skip_ws(text, colon + 1) else {
            break;
        };
        let Some(value_end) = skip_json_value(text, value_start) else {
            break;
        };
        if key == wanted {
            out.push(value_start);
        }
        let Some(delimiter) = skip_ws(text, value_end) else {
            break;
        };
        match text.as_bytes().get(delimiter) {
            Some(b',') => pos = delimiter + 1,
            Some(b'}') => break,
            _ => break,
        }
    }
    out
}

fn json_array_value(text: &str, mut pos: usize, wanted: usize) -> Option<usize> {
    if text.as_bytes().get(pos) != Some(&b'[') {
        return None;
    }
    pos += 1;
    let mut index = 0usize;
    loop {
        pos = skip_ws(text, pos)?;
        if text.as_bytes().get(pos) == Some(&b']') {
            return None;
        }
        let end = skip_json_value(text, pos)?;
        if index == wanted {
            return Some(pos);
        }
        let delimiter = skip_ws(text, end)?;
        if text.as_bytes().get(delimiter) != Some(&b',') {
            return None;
        }
        pos = delimiter + 1;
        index += 1;
    }
}

fn skip_ws(text: &str, mut pos: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    (pos <= bytes.len()).then_some(pos)
}

fn parse_json_string(text: &str, pos: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    if bytes.get(pos) != Some(&b'"') {
        return None;
    }
    let mut index = pos + 1;
    let mut escaped = false;
    while index < bytes.len() {
        match bytes[index] {
            b'"' if !escaped => {
                let end = index + 1;
                let decoded: String = serde_json::from_str(&text[pos..end]).ok()?;
                return Some((decoded, end));
            }
            b'\\' if !escaped => escaped = true,
            _ => escaped = false,
        }
        index += 1;
    }
    None
}

fn skip_json_value(text: &str, pos: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    match bytes.get(pos)? {
        b'"' => parse_json_string(text, pos).map(|(_, end)| end),
        b'{' | b'[' => {
            let mut stack = vec![bytes[pos]];
            let mut index = pos + 1;
            while index < bytes.len() {
                match bytes[index] {
                    b'"' => {
                        let (_, end) = parse_json_string(text, index)?;
                        index = end;
                        continue;
                    }
                    b'{' | b'[' => stack.push(bytes[index]),
                    b'}' if stack.last() == Some(&b'{') => {
                        stack.pop();
                        if stack.is_empty() {
                            return Some(index + 1);
                        }
                    }
                    b']' if stack.last() == Some(&b'[') => {
                        stack.pop();
                        if stack.is_empty() {
                            return Some(index + 1);
                        }
                    }
                    _ => {}
                }
                index += 1;
            }
            None
        }
        _ => {
            let mut index = pos;
            while index < bytes.len() && !b",}] \t\r\n".contains(&bytes[index]) {
                index += 1;
            }
            Some(index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn json_set_preserves_formatting() {
        let dir = ws();
        let f = dir.path().join("config.json");
        std::fs::write(
            &f,
            b"{\n  \"server\": {\n    \"port\": 9000,\n    \"host\": \"x\"\n  }\n}\n",
        )
        .unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.server.port",
            "8080",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("\"port\": 8080"), "{out}");
        assert!(out.contains("\"host\": \"x\""), "{out}");
        // byte-preserving: same shape, only the value changed
        assert!(out.starts_with("{\n  \"server\": {\n"), "{out}");
    }

    #[test]
    fn json_missing_path_refuses() {
        let dir = ws();
        let f = dir.path().join("c.json");
        std::fs::write(&f, b"{\"a\": 1}\n").unwrap();
        let cert = data_set(dir.path(), &f, "$.b", "2", false, &VerbOptions::default()).unwrap();
        assert_eq!(cert.status, Status::NotFound);
    }

    #[test]
    fn json_array_index() {
        let dir = ws();
        let f = dir.path().join("c.json");
        std::fs::write(&f, b"{\"items\": [1, 2, 3]}\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.items[1]",
            "99",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(std::fs::read_to_string(&f).unwrap().contains("[1, 99, 3]"));
    }

    #[test]
    fn ensure_is_idempotent() {
        let dir = ws();
        let f = dir.path().join("c.json");
        std::fs::write(&f, b"{\"port\": 8080}\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.port",
            "8080",
            true,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
    }

    #[test]
    fn toml_set_scalar() {
        let dir = ws();
        let f = dir.path().join("Cargo.toml");
        std::fs::write(&f, b"[package]\nname = \"x\"\nversion = \"0.1.0\"\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.package.version",
            "\"0.2.0\"",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("version = \"0.2.0\""), "{out}");
        assert!(out.contains("name = \"x\""), "{out}");
    }

    #[test]
    fn yaml_scalar_set() {
        let dir = ws();
        let f = dir.path().join("c.yaml");
        std::fs::write(&f, b"server:\n  port: 9000\n  host: x\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.server.port",
            "8080",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("port: 8080"), "{out}");
        assert!(out.contains("host: x"), "{out}");
    }
}
