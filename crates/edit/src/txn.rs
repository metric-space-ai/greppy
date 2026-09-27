//! The edit transaction core: snapshot, overlap rejection, in-memory apply,
//! reparse, changed-range accounting.
//!
//! All byte ranges are computed against one immutable snapshot; multiple
//! operations on a file are applied from the highest to the lowest offset so
//! earlier applications never shift later targets. Overlapping ranges are
//! rejected before anything is applied.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use crate::hash::sha256_hex;
use greppy_core::{Error, Result};
use greppy_parser::Language;

/// One planned mutation of a byte range within a snapshot.
#[derive(Debug, Clone)]
pub struct PlannedOp {
    pub id: String,
    pub range: (usize, usize),
    pub replacement: Vec<u8>,
}

/// An immutable view of one file at plan time.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub path: PathBuf,
    pub content: Vec<u8>,
    pub file_sha256: String,
}

impl Snapshot {
    pub fn read(path: &Path) -> Result<Self> {
        let meta = std::fs::symlink_metadata(path).map_err(|source| Error::Io {
            context: format!("stat {}", path.display()),
            source,
        })?;
        if meta.file_type().is_symlink() {
            return Err(Error::Workspace(format!(
                "refusing to edit through symlink: {}",
                path.display()
            )));
        }
        let content = std::fs::read(path).map_err(|source| Error::Io {
            context: format!("read {}", path.display()),
            source,
        })?;
        let file_sha256 = sha256_hex(&content);
        Ok(Self {
            path: path.to_path_buf(),
            content,
            file_sha256,
        })
    }
}

/// Result of applying planned operations in memory.
#[derive(Debug)]
pub struct Applied {
    pub content: Vec<u8>,
    pub file_sha256: String,
    /// Ranges (in ORIGINAL coordinates) that were replaced.
    pub changed_ranges: Vec<(usize, usize)>,
}

/// Reject overlaps, then apply high→low against the snapshot.
pub fn apply_in_memory(snapshot: &Snapshot, ops: &[PlannedOp]) -> Result<Applied> {
    for op in ops {
        let (start, end) = op.range;
        if start > end || end > snapshot.content.len() {
            return Err(Error::Invalid(format!(
                "operation {}: range {start}..{end} outside file of {} bytes",
                op.id,
                snapshot.content.len()
            )));
        }
    }
    let mut sorted: Vec<&PlannedOp> = ops.iter().collect();
    sorted.sort_by_key(|op| op.range.0);
    for (index, first) in sorted.iter().enumerate() {
        for second in &sorted[index + 1..] {
            if mutation_ranges_overlap(first.range, second.range) {
                return Err(Error::Invalid(format!(
                    "operations {} and {} overlap ({}..{} vs {}..{}); nothing was changed",
                    first.id,
                    second.id,
                    first.range.0,
                    first.range.1,
                    second.range.0,
                    second.range.1
                )));
            }
        }
    }
    let mut content = snapshot.content.clone();
    for op in sorted.iter().rev() {
        content.splice(op.range.0..op.range.1, op.replacement.iter().copied());
    }
    let changed_ranges = sorted.iter().map(|op| op.range).collect();
    let file_sha256 = sha256_hex(&content);
    Ok(Applied {
        content,
        file_sha256,
        changed_ranges,
    })
}

fn mutation_ranges_overlap(first: (usize, usize), second: (usize, usize)) -> bool {
    match (first.0 == first.1, second.0 == second.1) {
        // Several insertions at one boundary are deterministic: stable sorting
        // plus reverse application preserves plan order in the output.
        (true, true) => false,
        (true, false) => second.0 <= first.0 && first.0 < second.1,
        (false, true) => first.0 <= second.0 && second.0 < first.1,
        (false, false) => first.0 < second.1 && second.0 < first.1,
    }
}

/// Verify that every byte outside the declared original ranges maps
/// unchanged into the result (accounting for length deltas of the edits).
pub fn outside_ranges_unchanged(before: &[u8], after: &[u8], ops: &[PlannedOp]) -> bool {
    let mut sorted: Vec<&PlannedOp> = ops.iter().collect();
    sorted.sort_by_key(|op| op.range.0);
    let mut b = 0usize; // cursor in before
    let mut a = 0usize; // cursor in after
    for op in &sorted {
        let (start, end) = op.range;
        if before.get(b..start) != after.get(a..a + (start - b)) {
            return false;
        }
        a += start - b + op.replacement.len();
        b = end;
    }
    before.get(b..) == after.get(a..)
}

/// ERROR/MISSING counts of a parse tree, for syntax postconditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntaxCounts {
    pub errors: usize,
    pub missing: usize,
}

/// Build a validation-only view for TypeScript import-type queries that the
/// bundled grammar reports as errors below an outer `typeof`.
///
/// Candidates are recognized by a small lexical scanner, only in normal code
/// (never comments, strings, or templates), and only when the raw parse has an
/// ERROR/MISSING node at that candidate. Horizontal whitespace is accepted,
/// while newlines and line continuations are deliberately left untouched.
/// Replacing only the `import("literal")` portion with a one-byte identifier
/// plus spaces preserves every byte and line coordinate.
fn syntax_validation_content(language: Language, content: &[u8]) -> Cow<'_, [u8]> {
    if language != (Language::TypeScript { tsx: false }) &&
        language != (Language::TypeScript { tsx: true })
    {
        return Cow::Borrowed(content);
    }

    let Ok(raw_tree) = greppy_parser::parse(language, content) else {
        return Cow::Borrowed(content);
    };
    let mut error_offsets = Vec::new();
    let mut tree_cursor = raw_tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = tree_cursor.node();
        if node.is_error() || node.is_missing() {
            error_offsets.push(node.start_byte());
        }
        if tree_cursor.goto_first_child() {
            continue;
        }
        loop {
            if tree_cursor.goto_next_sibling() {
                break;
            }
            if !tree_cursor.goto_parent() {
                reached_root = true;
                break;
            }
        }
    }
    if error_offsets.is_empty() {
        return Cow::Borrowed(content);
    }

    fn identifier_byte(byte: u8) -> bool {
        byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$'
    }
    fn horizontal_space(byte: u8) -> bool {
        matches!(byte, b' ' | b'\t' | 0x0c)
    }
    fn skip_quoted(content: &[u8], start: usize, quote: u8) -> usize {
        let mut cursor = start + 1;
        while let Some(&byte) = content.get(cursor) {
            if byte == b'\\' {
                cursor += 2;
                continue;
            }
            cursor += 1;
            if byte == quote || byte == b'\n' || byte == b'\r' {
                break;
            }
        }
        cursor
    }

    let mut normalized: Option<Vec<u8>> = None;
    let mut cursor = 0usize;
    while cursor < content.len() {
        match (content.get(cursor), content.get(cursor + 1)) {
            (Some(b'/'), Some(b'/')) => {
                cursor += 2;
                while !matches!(content.get(cursor), None | Some(b'\n' | b'\r')) {
                    cursor += 1;
                }
                continue;
            },
            (Some(b'/'), Some(b'*')) => {
                cursor += 2;
                while cursor + 1 < content.len() &&
                    !matches!((content[cursor], content[cursor + 1]), (b'*', b'/'))
                {
                    cursor += 1;
                }
                cursor = (cursor + 2).min(content.len());
                continue;
            },
            (Some(&quote @ (b'\'' | b'"' | b'`')), _) => {
                cursor = skip_quoted(content, cursor, quote);
                continue;
            },
            _ => {},
        }

        const TYPEOF: &[u8] = b"typeof";
        if !content[cursor..].starts_with(TYPEOF) ||
            cursor.checked_sub(1).and_then(|at| content.get(at)).is_some_and(|b| identifier_byte(*b)) ||
            content.get(cursor + TYPEOF.len()).is_some_and(|b| identifier_byte(*b))
        {
            cursor += 1;
            continue;
        }
        let start = cursor;
        cursor += TYPEOF.len();
        let whitespace_start = cursor;
        while content.get(cursor).is_some_and(|b| horizontal_space(*b)) {
            cursor += 1;
        }
        if cursor == whitespace_start || !content[cursor..].starts_with(b"import") ||
            content.get(cursor + b"import".len()).is_some_and(|b| identifier_byte(*b))
        {
            continue;
        }
        let import_start = cursor;
        cursor += b"import".len();
        while content.get(cursor).is_some_and(|b| horizontal_space(*b)) {
            cursor += 1;
        }
        if content.get(cursor) != Some(&b'(') {
            continue;
        }
        cursor += 1;
        while content.get(cursor).is_some_and(|b| horizontal_space(*b)) {
            cursor += 1;
        }
        let quote_at = cursor;
        let Some(&quote @ (b'\'' | b'"')) = content.get(quote_at) else {
            continue;
        };
        cursor = quote_at + 1;
        let mut escaped = false;
        while let Some(&byte) = content.get(cursor) {
            if escaped {
                if byte == b'\n' || byte == b'\r' {
                    break;
                }
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == quote {
                break;
            } else if byte == b'\n' || byte == b'\r' {
                break;
            }
            cursor += 1;
        }
        if content.get(cursor) != Some(&quote) {
            continue;
        }
        cursor += 1;
        while content.get(cursor).is_some_and(|b| horizontal_space(*b)) {
            cursor += 1;
        }
        if content.get(cursor) != Some(&b')') {
            continue;
        }
        let import_end = cursor + 1;
        if !error_offsets
            .iter()
            .any(|error| *error >= start && *error <= import_end + 1)
        {
            cursor = import_end;
            continue;
        }
        let output = normalized.get_or_insert_with(|| content.to_vec());
        let span = &mut output[import_start..import_end];
        span.fill(b' ');
        span[0] = b'T';
        cursor = import_end;
    }
    normalized.map_or(Cow::Borrowed(content), Cow::Owned)
}

/// First parser failure in the proposed content. Coordinates are one-based;
/// columns count bytes, as in tree-sitter, rather than displayed characters.
pub fn first_syntax_diagnostic(language: Language, content: &[u8]) -> Option<String> {
    let validation_content = syntax_validation_content(language, content);
    let tree = greppy_parser::parse(language, &validation_content).ok()?;
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        if node.is_error() || node.is_missing() {
            let start = node.start_position();
            let reason = if node.is_missing() {
                format!("missing `{}`", node.kind())
            } else {
                "unexpected syntax".to_string()
            };
            return Some(format!(
                "{}:{} (tree-sitter: {reason}; column is a byte offset)",
                start.row + 1,
                start.column + 1
            ));
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return None;
            }
        }
    }
}

/// The kinds of the ancestor chain (parent -> root, leaf excluded) of the
/// smallest node covering `range`. This is the structural CONTEXT the edited
/// bytes live in.
///
/// Counting ERROR/MISSING nodes alone is not a sufficient syntax gate:
/// tree-sitter's error recovery silently reinterprets many malformations
/// without emitting ERROR nodes (proven 2026-07-17: replacing a Go method
/// body with a whole file's text — copyright header, package decl, imports —
/// yielded new_errors=0 while gofmt rejected the file, so the certificate
/// falsely reported `syntax: proved`). A structural edit must not change the
/// context its target sits in: a body stays inside its function, a function
/// stays a top-level declaration. When the surrounding context's kind chain
/// changes, the edit broke the grammar in a way tree-sitter recovered past.
fn context_kinds(language: Language, content: &[u8], range: (usize, usize)) -> Option<Vec<String>> {
    let validation_content = syntax_validation_content(language, content);
    let tree = greppy_parser::parse(language, &validation_content).ok()?;
    let leaf = tree
        .root_node()
        .descendant_for_byte_range(range.0, range.1.saturating_sub(1).max(range.0))?;
    let mut kinds = Vec::new();
    let mut node = leaf.parent();
    while let Some(cur) = node {
        kinds.push(cur.kind().to_string());
        node = cur.parent();
    }
    Some(kinds)
}

/// Does the edited region still sit in the same structural context after the
/// edit? `before_range` is the target in the pre-edit content; `after_range`
/// is the changed span in the post-edit content. Returns true (permissive)
/// when either side cannot be parsed — that path is covered by the
/// ERROR/MISSING count, which reports not-applicable.
pub fn structural_context_preserved(
    language: Language,
    before: &[u8],
    before_range: (usize, usize),
    after: &[u8],
    after_range: (usize, usize),
) -> bool {
    match (
        context_kinds(language, before, before_range),
        context_kinds(language, after, after_range),
    ) {
        (Some(b), Some(a)) => a == b,
        _ => true,
    }
}

/// Parse `content` and count ERROR and MISSING nodes. `None` when the
/// language is not tree-sitter-supported (postcondition then reports
/// not-applicable rather than silently passing).
pub fn syntax_counts(language: Language, content: &[u8]) -> Option<SyntaxCounts> {
    let validation_content = syntax_validation_content(language, content);
    let tree = greppy_parser::parse(language, &validation_content).ok()?;
    let mut errors = 0usize;
    let mut missing = 0usize;
    let mut cursor = tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = cursor.node();
        if node.is_error() {
            errors += 1;
        }
        if node.is_missing() {
            missing += 1;
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                reached_root = true;
                break;
            }
        }
    }
    Some(SyntaxCounts { errors, missing })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(content: &[u8]) -> Snapshot {
        Snapshot {
            path: PathBuf::from("mem"),
            content: content.to_vec(),
            file_sha256: sha256_hex(content),
        }
    }

    fn op(id: &str, range: (usize, usize), replacement: &[u8]) -> PlannedOp {
        PlannedOp {
            id: id.into(),
            range,
            replacement: replacement.to_vec(),
        }
    }

    #[test]
    fn css_named_container_queries_are_valid_without_weakening_syntax_errors() {
        let css = greppy_parser::language_for_path(std::path::Path::new("editor.css"));
        let valid = br#"@container mail-content-editor (max-width: 460px) {
  .editor { color: red; }
}
@container sidebar style(--theme: dark) {
  .message { display: block; }
}
"#;
        assert_eq!(
            syntax_counts(css, valid),
            Some(SyntaxCounts {
                errors: 0,
                missing: 0
            })
        );
        assert_eq!(first_syntax_diagnostic(css, valid), None);

        let malformed_query = br#"@container mail-content-editor (max-width 460px) {
  .editor { color: red; }
}
"#;
        let query_counts = syntax_counts(css, malformed_query).unwrap();
        assert!(
            query_counts.errors + query_counts.missing > 0,
            "malformed container query must remain an atomic edit failure"
        );
        assert!(first_syntax_diagnostic(css, malformed_query).is_some());

        let malformed_body = br#"@container mail-content-editor (max-width: 460px) {
  .editor { color: red; }
"#;
        let body_counts = syntax_counts(css, malformed_body).unwrap();
        assert!(
            body_counts.errors + body_counts.missing > 0,
            "malformed container body must remain an atomic edit failure"
        );
        assert!(first_syntax_diagnostic(css, malformed_body).is_some());
    }

    #[test]
    fn typescript_import_type_query_is_valid_without_weakening_syntax_errors() {
        let language = Language::TypeScript { tsx: false };
        let valid = br#"vi.mock("node:child_process", async (importOriginal) => {
  const original = await importOriginal<typeof import("node:child_process")>();
  return { ...original, spawn: vi.fn(original.spawn) };
});
"#;
        assert_eq!(
            syntax_counts(language, valid),
            Some(SyntaxCounts {
                errors: 0,
                missing: 0
            })
        );
        assert_eq!(first_syntax_diagnostic(language, valid), None);

        for valid_whitespace in [
            br#"type ChildProcess = typeof  import ("node:child_process");"#.as_slice(),
            b"type ChildProcess = typeof\timport\t(\t'node:child_process'\t);".as_slice(),
        ] {
            assert_eq!(
                syntax_counts(language, valid_whitespace),
                Some(SyntaxCounts {
                    errors: 0,
                    missing: 0
                }),
                "{}",
                String::from_utf8_lossy(valid_whitespace)
            );
        }

        let malformed = br#"vi.mock("node:child_process", async (importOriginal) => {
  const original = await importOriginal<typeof import("node:child_process")>();
  return { ...original, spawn: vi.fn(original.spawn) };
);
"#;
        let counts = syntax_counts(language, malformed).unwrap();
        assert!(
            counts.errors + counts.missing > 0,
            "unbalanced TypeScript must remain an atomic edit failure"
        );
        assert!(first_syntax_diagnostic(language, malformed).is_some());

        for lexically_invalid in [
            br#"/* typeof import("*/") */"#.as_slice(),
            br#"const value = "typeof import(\"node:child_process\")"#.as_slice(),
        ] {
            assert!(
                syntax_counts(language, lexically_invalid)
                    .is_some_and(|counts| counts.errors + counts.missing > 0),
                "malformed comment/string must not be hidden: {}",
                String::from_utf8_lossy(lexically_invalid)
            );
            assert_eq!(
                syntax_validation_content(language, lexically_invalid).as_ref(),
                lexically_invalid,
                "comment/string contents must never be rewritten"
            );
        }

        let escaped_newline =
            b"type ChildProcess = typeof import(\"node:\\\nchild_process\");";
        assert_eq!(
            syntax_validation_content(language, escaped_newline).as_ref(),
            escaped_newline,
            "line continuations must not be rewritten because that would move diagnostics"
        );
    }

    #[test]
    fn applies_high_to_low_without_shifting() {
        let s = snap(b"aaa bbb ccc");
        let applied =
            apply_in_memory(&s, &[op("1", (0, 3), b"XXXXX"), op("2", (8, 11), b"Y")]).unwrap();
        assert_eq!(applied.content, b"XXXXX bbb Y");
        assert!(outside_ranges_unchanged(
            &s.content,
            &applied.content,
            &[op("1", (0, 3), b"XXXXX"), op("2", (8, 11), b"Y")]
        ));
    }

    #[test]
    fn rejects_overlap_without_changing_anything() {
        let s = snap(b"0123456789");
        let err = apply_in_memory(&s, &[op("a", (0, 5), b""), op("b", (3, 7), b"")]);
        assert!(err.is_err());
    }

    #[test]
    fn same_boundary_insertions_preserve_plan_order() {
        let s = snap(b"ab");
        let applied =
            apply_in_memory(&s, &[op("first", (1, 1), b"X"), op("second", (1, 1), b"Y")]).unwrap();
        assert_eq!(applied.content, b"aXYb");
        assert!(outside_ranges_unchanged(
            &s.content,
            &applied.content,
            &[op("first", (1, 1), b"X"), op("second", (1, 1), b"Y")]
        ));
    }

    #[test]
    fn outside_check_detects_clobber() {
        let before = b"aaa bbb ccc";
        // simulate a buggy apply that also mutated untouched bytes
        let after = b"XXX bbb cZc";
        assert!(!outside_ranges_unchanged(
            before,
            after,
            &[op("1", (0, 3), b"XXX")]
        ));
    }

    #[test]
    fn syntax_counts_flag_broken_rust() {
        let ok = syntax_counts(Language::Rust, b"fn main() {}\n").unwrap();
        assert_eq!(
            ok,
            SyntaxCounts {
                errors: 0,
                missing: 0
            }
        );
        let broken = syntax_counts(Language::Rust, b"fn main( {}\n").unwrap();
        assert!(broken.errors + broken.missing > 0);
    }

    #[test]
    fn syntax_counts_accept_valid_rust_macro_and_let_else_forms() {
        for source in [
            "fn f() { let raw = 1; let _ = &raw; }",
            "fn f() { let raw = [1, 2]; let _ = &raw[..]; let _ = &raw[0]; }",
            "fn f() { let mut raw = 1; let _ = &raw const raw; let _ = &raw mut raw; }",
            "fn f() -> Result<(), ()> { let Ok(root) = std::env::var(\"ROOT\") else { return Ok(()); }; Ok(()) }",
            "fn f() { let payload = serde_json::json!({\"id\": task.message_key, \"status\": \"running\"}).to_string(); }",
            "fn f() -> Result<(), ()> { let (raw, revision): (String, String) = db.query_row(\"SELECT\", params![key], |row| Ok((row.get(0)?, row.get(1)?)),)?; Ok(()) }",
        ] {
            let counts = syntax_counts(Language::Rust, source.as_bytes()).unwrap();
            assert_eq!(counts, SyntaxCounts { errors: 0, missing: 0 }, "{source}\n{}", greppy_parser::parse(Language::Rust, source.as_bytes()).unwrap().root_node().to_sexp());
        }
    }

    #[test]
    fn property_random_mutation_never_corrupts() {
        // deterministic pseudo-random walk: any post-snapshot mutation must be
        // caught by the hash check before publish (verified here via sha
        // comparison, the same check publish performs)
        let s = snap(b"the quick brown fox jumps over the lazy dog");
        let mut seed = 0x9e3779b97f4a7c15u64;
        for _ in 0..500 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (seed >> 33) as usize % s.content.len();
            let mut live = s.content.clone();
            live[idx] ^= 0x20;
            assert_ne!(
                sha256_hex(&live),
                s.file_sha256,
                "mutation must change hash"
            );
        }
    }
}
