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

/// Select the syntax grammar from the immutable pre-edit snapshot. A `.h`
/// header can contain either C or C++; prefer C++ only when it explains the
/// existing bytes strictly better without adding either kind of diagnostic.
/// The same selected grammar must validate both sides of the transaction.
pub fn syntax_language_for_path(path: &Path, before: &[u8]) -> Language {
    let language = greppy_parser::language_for_path(path);
    if language != Language::C || path.extension().and_then(|value| value.to_str()) != Some("h") {
        return language;
    }
    if let (Some(c), Some(cpp)) = (
        syntax_counts(Language::C, before),
        syntax_counts(Language::Cpp, before),
    ) {
        if cpp.errors <= c.errors && cpp.missing <= c.missing && cpp != c {
            return Language::Cpp;
        }
    }
    language
}

/// Resolve a C-compatible `.h` edit without interpreting valid new C++ as
/// broken C. A fallback requires both snapshots to parse completely as C++;
/// existing recovery diagnostics cannot become a license to change grammar.
/// Validate both snapshots with the returned grammar.
pub fn syntax_language_for_edit(path: &Path, before: &[u8], after: &[u8]) -> Language {
    let language = syntax_language_for_path(path, before);
    if language != Language::C || path.extension().and_then(|value| value.to_str()) != Some("h") {
        return language;
    }
    if let (Some(c_before), Some(c_after), Some(cpp_before), Some(cpp_after)) = (
        syntax_counts(Language::C, before),
        syntax_counts(Language::C, after),
        syntax_counts(Language::Cpp, before),
        syntax_counts(Language::Cpp, after),
    ) {
        let c_regressed = c_after.errors > c_before.errors || c_after.missing > c_before.missing;
        let cpp_clean = cpp_before.errors == 0
            && cpp_before.missing == 0
            && cpp_after.errors == 0
            && cpp_after.missing == 0;
        if c_regressed && cpp_clean {
            return Language::Cpp;
        }
    }
    language
}

/// Validation-only recovery for Bash's read/write redirect omitted by the grammar.
fn bash_validation_content(content: &[u8]) -> Cow<'_, [u8]> {
    let Ok(tree) = greppy_parser::parse(Language::Bash, content) else {
        return Cow::Borrowed(content);
    };
    let mut normalized: Option<Vec<u8>> = None;
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        if node.is_error() && content.get(node.start_byte()..node.end_byte()) == Some(b">") {
            if let Some(redirect) = node.parent().filter(|p| p.kind() == "file_redirect") {
                let start = node.start_byte();
                let end = node.end_byte();
                let destination = redirect.child_by_field_name("destination");
                let exact_operator = (0..redirect.child_count())
                    .filter_map(|i| redirect.child(i))
                    .any(|child| {
                        child.kind() == "<"
                            && child.end_byte() == start
                            && child.start_byte().checked_add(1) == Some(start)
                    });
                if exact_operator
                    && destination.is_some_and(|d| !d.has_error() && d.start_byte() >= end)
                {
                    // tree-sitter-bash 0.25.1 omits Bash's read/write `<>`.
                    // Validate an equally sized `>>` view of this exact recovered
                    // redirect; proposed bytes and diagnostic positions stay intact.
                    let output = normalized.get_or_insert_with(|| content.to_vec());
                    output[start - 1] = b'>';
                }
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return normalized.map_or(Cow::Borrowed(content), Cow::Owned);
            }
        }
    }
}

/// Build a validation-only view for the exact import-type recovery shape
/// emitted by the bundled TypeScript grammar.
///
/// For `fn<typeof import("module")>()`, the grammar parses `<` as a binary
/// operator, retains `typeof import("module")` as a
/// `binary_expression -> unary_expression -> call_expression(import)` right
/// operand, then emits one ERROR over the exact `>()` suffix. All of that raw
/// AST and byte evidence must match before validation substitutes the import
/// call. Lookalikes in comments, strings, templates, regexes, value
/// expressions, and unrelated malformed code cannot qualify. The substituted
/// span preserves length and every newline.
fn syntax_validation_content(language: Language, content: &[u8]) -> Cow<'_, [u8]> {
    if matches!(language, Language::C | Language::Cpp) {
        return guarded_linkage_validation_content(language, content);
    }
    if language.name() == "json" {
        return json_validation_content(content);
    }
    if matches!(language, Language::Bash) {
        return bash_validation_content(content);
    }

    if !matches!(language, Language::TypeScript { .. }) {
        return Cow::Borrowed(content);
    }
    let Ok(raw_tree) = greppy_parser::parse(language, content) else {
        return Cow::Borrowed(content);
    };

    let mut errors = Vec::new();
    let mut tree_cursor = raw_tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = tree_cursor.node();
        if node.is_error() || node.is_missing() {
            errors.push((node.start_byte(), node.end_byte(), node.is_missing()));
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
    if errors.is_empty() {
        return Cow::Borrowed(content);
    }

    let mut normalized: Option<Vec<u8>> = None;
    // The shared grammar also accepts Flow's exact-object `{| ... |}`
    // delimiters. A compact malformed TypeScript union such as `string|}`
    // can be recovered as that delimiter without an ERROR. Never erase
    // those non-TypeScript type arguments while repairing a typed tag.
    // tree-sitter-typescript 0.23.2 omits type_arguments on template_call.

    // Its exact recovery is a complete instantiation_expression followed by
    // a fabricated missing `!` inside non_null_expression. Validate a view
    // without those type arguments; never alter the proposed source bytes.
    let mut cursor = raw_tree.walk();
    loop {
        let node = cursor.node();
        if node.is_missing() && node.kind() == "!" {
            if let Some(non_null) = node.parent().filter(|p| p.kind() == "non_null_expression") {
                if let Some(call) = non_null.parent().filter(|p| p.kind() == "call_expression") {
                    let mut expression = non_null.named_child(0);
                    if expression.is_some_and(|p| p.kind() == "yield_expression") {
                        expression = expression.and_then(|p| p.named_child(0));
                    }
                    if let (Some(instance), Some(template)) = (
                        expression.filter(|p| p.kind() == "instantiation_expression"),
                        call.child_by_field_name("arguments")
                            .filter(|p| p.kind() == "template_string" && !p.has_error()),
                    ) {
                        if let Some(types) = instance.child_by_field_name("type_arguments") {
                            let has_flow_object_delimiter = || {
                                let mut cursor = types.walk();
                                loop {
                                    if matches!(cursor.node().kind(), "{|" | "|}") {
                                        return true;
                                    }
                                    if cursor.goto_first_child() {
                                        continue;
                                    }
                                    loop {
                                        if cursor.goto_next_sibling() {
                                            break;
                                        }
                                        if !cursor.goto_parent() {
                                            return false;
                                        }
                                    }
                                }
                            };
                            let start = types.start_byte();

                            let end = types.end_byte();
                            if !instance.has_error()
                                && types.named_child_count() > 0
                                && !has_flow_object_delimiter()
                                && end == node.start_byte()
                                && end <= template.start_byte()
                                && content.get(start) == Some(&b'<')
                                && content.get(end.wrapping_sub(1)) == Some(&b'>')
                                && content[end..template.start_byte()]
                                    .iter()
                                    .all(u8::is_ascii_whitespace)
                            {
                                let output = normalized.get_or_insert_with(|| content.to_vec());
                                for byte in &mut output[start..end] {
                                    if !matches!(*byte, b'\n' | b'\r') {
                                        *byte = b' ';
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                break;
            }
        }
        if cursor.node() == raw_tree.root_node() {
            break;
        }
    }

    fn identifier_byte(byte: u8) -> bool {
        byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$'
    }
    let mut scan = 0usize;
    while scan + b"typeof".len() <= content.len() {
        let Some(relative) = content[scan..]
            .windows(b"typeof".len())
            .position(|window| window == b"typeof")
        else {
            break;
        };
        let start = scan + relative;
        scan = start + b"typeof".len();
        if start
            .checked_sub(1)
            .and_then(|at| content.get(at))
            .is_some_and(|byte| identifier_byte(*byte))
            || content.get(scan).is_some_and(|byte| identifier_byte(*byte))
        {
            continue;
        }
        let Some(type_node) = raw_tree.root_node().descendant_for_byte_range(start, scan) else {
            continue;
        };
        let Some(unary_expression) = std::iter::successors(Some(type_node), |node| node.parent())
            .find(|node| node.kind() == "unary_expression")
        else {
            continue;
        };
        if !matches!(
            unary_expression.parent(),
            Some(node) if node.kind() == "binary_expression"
        ) {
            continue;
        }
        let mut before_typeof = start;
        while before_typeof > 0 && content[before_typeof - 1].is_ascii_whitespace() {
            before_typeof -= 1;
        }
        if before_typeof == 0 || content[before_typeof - 1] != b'<' {
            continue;
        }

        let whitespace_start = scan;
        while content
            .get(scan)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            scan += 1;
        }
        if scan == whitespace_start || !content[scan..].starts_with(b"import") {
            continue;
        }
        let import_start = scan;
        scan += b"import".len();
        if content.get(scan).is_some_and(|byte| identifier_byte(*byte)) {
            continue;
        }
        while content
            .get(scan)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            scan += 1;
        }
        if content.get(scan) != Some(&b'(') {
            continue;
        }
        scan += 1;
        while content
            .get(scan)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            scan += 1;
        }
        let Some(&quote @ (b'\'' | b'"')) = content.get(scan) else {
            continue;
        };
        scan += 1;
        let mut escaped = false;
        let mut line_continuation = false;
        while let Some(&byte) = content.get(scan) {
            if escaped {
                line_continuation |= matches!(byte, b'\n' | b'\r');
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == quote || matches!(byte, b'\n' | b'\r') {
                break;
            }
            scan += 1;
        }
        if line_continuation || content.get(scan) != Some(&quote) {
            continue;
        }
        scan += 1;
        while content
            .get(scan)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            scan += 1;
        }
        if content.get(scan) != Some(&b')') {
            continue;
        }
        let import_end = scan + 1;
        let suffix_end = import_end + 3;
        if content.get(import_end..suffix_end) != Some(b">()")
            || !errors.iter().any(|(error_start, error_end, missing)| {
                !missing && *error_start == import_end && *error_end == suffix_end
            })
        {
            continue;
        }

        let output = normalized.get_or_insert_with(|| content.to_vec());
        for (offset, byte) in output[import_start..import_end].iter_mut().enumerate() {
            if !matches!(*byte, b'\n' | b'\r') {
                *byte = if offset == 0 { b'T' } else { b' ' };
            }
        }
        scan = import_end;
    }
    normalized.map_or(Cow::Borrowed(content), Cow::Owned)
}

/// The bundled JSON grammar omits `+` in numeric exponents. Only normalize
/// strict JSON accepted independently, outside strings, in a same-length view.
/// This cannot turn malformed JSON into an accepted edit or alter written bytes.
fn json_validation_content(content: &[u8]) -> Cow<'_, [u8]> {
    let mut parser = serde_json::Deserializer::from_slice(content);
    if <serde::de::IgnoredAny as serde::Deserialize>::deserialize(&mut parser).is_err()
        || parser.end().is_err()
    {
        return Cow::Borrowed(content);
    }
    let mut normalized = None;
    let mut quoted = false;
    let mut escaped = false;
    for (i, &byte) in content.iter().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
        } else if byte == b'+'
            && i >= 2
            && matches!(content[i - 1], b'e' | b'E')
            && content[i - 2].is_ascii_digit()
            && content.get(i + 1).is_some_and(u8::is_ascii_digit)
        {
            // `2.84e+18` and `2.84e018` have identical numeric meaning and
            // AST boundaries. The original bytes remain the edit payload.
            normalized.get_or_insert_with(|| content.to_vec())[i] = b'0';
        }
    }
    normalized.map_or(Cow::Borrowed(content), Cow::Owned)
}

/// Validate only a complete unique conventional linkage pair. The body and
/// other directives stay parsed; spaces preserve diagnostic coordinates.
fn guarded_linkage_validation_content(language: Language, content: &[u8]) -> Cow<'_, [u8]> {
    const OPEN: &[u8] = b"#ifdef __cplusplus\nextern \"C\" {\n#endif\n";
    const CLOSE: &[u8] = b"#ifdef __cplusplus\n}\n#endif\n";
    const OPEN_CRLF: &[u8] = b"#ifdef __cplusplus\r\nextern \"C\" {\r\n#endif\r\n";
    const CLOSE_CRLF: &[u8] = b"#ifdef __cplusplus\r\n}\r\n#endif\r\n";
    let unique_line = |variants: [&[u8]; 2]| {
        let mut matches = variants.into_iter().flat_map(|needle| {
            content
                .windows(needle.len())
                .enumerate()
                .filter_map(move |(i, bytes)| {
                    (bytes == needle && (i == 0 || content[i - 1] == b'\n'))
                        .then_some((i, needle.len()))
                })
        });
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    };
    let (Some((open, open_len)), Some((close, close_len))) = (
        unique_line([OPEN, OPEN_CRLF]),
        unique_line([CLOSE, CLOSE_CRLF]),
    ) else {
        return Cow::Borrowed(content);
    };
    if open + open_len > close {
        return Cow::Borrowed(content);
    }
    let Ok(tree) = greppy_parser::parse(language, content) else {
        return Cow::Borrowed(content);
    };
    for offset in [open, close] {
        let Some(mut node) = tree
            .root_node()
            .descendant_for_byte_range(offset, offset + 1)
        else {
            return Cow::Borrowed(content);
        };
        loop {
            if matches!(
                node.kind(),
                "comment" | "string_literal" | "raw_string_literal"
            ) {
                return Cow::Borrowed(content);
            }
            let Some(parent) = node.parent() else { break };
            node = parent;
        }
    }
    let mut normalized = content.to_vec();
    for (start, len) in [(open, open_len), (close, close_len)] {
        for byte in &mut normalized[start..start + len] {
            if !matches!(*byte, b'\r' | b'\n') {
                *byte = b' ';
            }
        }
    }
    // Linkage specifications are only allowed at namespace scope. Erasing
    // these wrappers inside a function/struct must not make invalid C++ valid.
    let Ok(view) = greppy_parser::parse(language, &normalized) else {
        return Cow::Borrowed(content);
    };
    let mut enclosing = Vec::new();
    for offset in [open, close] {
        let mut ancestors = Vec::new();
        let Some(mut node) = view
            .root_node()
            .descendant_for_byte_range(offset, offset + 1)
        else {
            return Cow::Borrowed(content);
        };
        loop {
            if !matches!(
                node.kind(),
                "translation_unit"
                    | "preproc_if"
                    | "preproc_ifdef"
                    | "preproc_else"
                    | "preproc_elif"
            ) {
                return Cow::Borrowed(content);
            }
            ancestors.push((node.kind(), node.start_byte(), node.end_byte()));
            let Some(parent) = node.parent() else { break };
            node = parent;
        }
        enclosing.push(ancestors);
    }
    if enclosing[0] != enclosing[1] {
        return Cow::Borrowed(content);
    }
    Cow::Owned(normalized)
}
/// First parser failure in the proposed content. Coordinates are one-based;
/// columns count bytes, as in tree-sitter, rather than displayed characters.
// The JS/TS grammars accept raw line breaks in ordinary quoted strings
// without an ERROR node. ECMAScript does not. JSX attribute strings have
// different lexical rules and may contain raw line breaks.
fn js_ts_string_line_breaks(
    language: Language,
    kind: &str,
    jsx_attribute: bool,
    bytes: &[u8],
) -> (usize, Option<usize>) {
    if !matches!(language, Language::JavaScript | Language::TypeScript { .. })
        || kind != "string"
        || jsx_attribute
        || !matches!(bytes.first(), Some(b'\'' | b'"'))
    {
        return (0, None);
    }
    let mut count = 0;
    let mut first = None;
    let mut offset = 1;
    while offset < bytes.len() {
        match bytes[offset] {
            b'\\' => {
                offset += 1;
                // A backslash followed by CRLF is one legal continuation.
                if bytes.get(offset) == Some(&b'\r') && bytes.get(offset + 1) == Some(&b'\n') {
                    offset += 1;
                }
            }
            b'\r' | b'\n' => {
                count += 1;
                first.get_or_insert(offset);
                if bytes[offset] == b'\r' && bytes.get(offset + 1) == Some(&b'\n') {
                    offset += 1;
                }
            }
            _ => {}
        }
        offset += 1;
    }
    (count, first)
}

// tree-sitter accepts empty Python suites and module/class-level returns
// without ERROR nodes. Those recoveries must not certify a breaking edit.
// Inspect scope boundaries rather than accepting any outer function ancestor:
// a class declared inside a function is still not a return-capable scope.
fn python_syntax_diagnostics(content: &[u8]) -> Vec<(usize, usize, &'static str)> {
    let Ok(tree) = greppy_parser::parse_for_syntax_validation(Language::Python, content) else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        let reason = if node.kind() == "block" {
            let mut cursor = node.walk();
            let has_statement = node
                .named_children(&mut cursor)
                .any(|child| child.kind() != "comment");
            (!has_statement).then_some("Python suite requires a statement; --body replacements must include indentation (for example, four spaces before return); use pass for an empty body")
        } else if node.kind() == "return_statement" {
            let mut ancestor = node.parent();
            let mut in_function = false;
            while let Some(scope) = ancestor {
                match scope.kind() {
                    "function_definition" => {
                        in_function = true;
                        break;
                    }
                    "class_definition" | "module" => break,
                    _ => ancestor = scope.parent(),
                }
            }
            (!in_function).then_some(
                "Python return must remain inside its function; preserve the body's indentation",
            )
        } else {
            None
        };
        if let Some(reason) = reason {
            let position = node.start_position();
            issues.push((position.row + 1, position.column + 1, reason));
        }
        let mut cursor = node.walk();
        stack.extend(node.named_children(&mut cursor));
    }
    issues.sort_by_key(|&(row, column, _)| (row, column));
    issues
}

pub fn first_syntax_diagnostic(language: Language, content: &[u8]) -> Option<String> {
    if language == Language::Python {
        if let Some((row, column, reason)) = python_syntax_diagnostics(content).first() {
            return Some(format!(
                "{row}:{column} ({reason}; column is a byte offset)"
            ));
        }
    }
    let validation_content = syntax_validation_content(language, content);
    let tree = greppy_parser::parse_for_syntax_validation(language, &validation_content).ok()?;
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        let (_, line_break) = js_ts_string_line_breaks(
            language,
            node.kind(),
            node.parent().is_some_and(|p| p.kind() == "jsx_attribute"),
            &validation_content[node.byte_range()],
        );
        if let Some(relative) = line_break {
            let offset = node.start_byte() + relative;
            let prefix = &validation_content[..offset];
            let row = prefix.iter().filter(|&&byte| byte == b'\n').count() + 1;
            let column = offset
                - prefix
                    .iter()
                    .rposition(|&byte| byte == b'\n')
                    .map_or(0, |i| i + 1)
                + 1;
            return Some(format!(
                "{row}:{column} (unescaped line break in quoted JavaScript/TypeScript string; use an escaped newline or a template literal; column is a byte offset)"
            ));
        }
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
    let tree = greppy_parser::parse_for_syntax_validation(language, &validation_content).ok()?;
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
    let tree = greppy_parser::parse_for_syntax_validation(language, &validation_content).ok()?;
    let mut errors = if language == Language::Python {
        python_syntax_diagnostics(content).len()
    } else {
        0
    };
    let mut missing = 0usize;
    let mut cursor = tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = cursor.node();
        errors += js_ts_string_line_breaks(
            language,
            node.kind(),
            node.parent().is_some_and(|p| p.kind() == "jsx_attribute"),
            &validation_content[node.byte_range()],
        )
        .0;
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
    #[test]
    fn python_syntax_counts_reject_empty_suites_and_escaped_returns() {
        for invalid in [
            "def f():\nreturn 1\n",
            "def f():\n    # no statement\n",
            "if True:\nprint(1)\n",
            "class Empty:\n# no statement\n",
            "return 1\n",
            "def f():\n    pass\nreturn 1\n",
            "def f():\n    class C:\n        return 1\n",
        ] {
            assert!(
                syntax_counts(Language::Python, invalid.as_bytes())
                    .unwrap()
                    .errors
                    > 0,
                "{invalid}"
            );
            assert!(
                first_syntax_diagnostic(Language::Python, invalid.as_bytes())
                    .unwrap()
                    .contains("Python")
            );
        }
        for valid in [
            "def f():\n    return 1\n",
            "def f(): return 1\n",
            "def f():\n    pass\n",
            "def f():\n    ...\n",
            "def f():\n    \"docstring\"\n",
            "def f():\r\n\treturn 1\r\n",
            "@decorate\nasync def f():\n    return await g()\n",
            "def f():\n    class C:\n        def g(self):\n            return 1\n    return C\n",
            "def f():\n    if True:\n        return \"\"\"multiline\nreturn 2\n\"\"\"\n",
        ] {
            assert_eq!(
                syntax_counts(Language::Python, valid.as_bytes())
                    .unwrap()
                    .errors,
                0,
                "{valid}"
            );
            assert_eq!(
                first_syntax_diagnostic(Language::Python, valid.as_bytes()),
                None,
                "{valid}"
            );
        }
    }

    #[test]
    fn js_ts_quoted_strings_reject_unescaped_line_breaks() {
        for language in [
            Language::JavaScript,
            Language::TypeScript { tsx: false },
            Language::TypeScript { tsx: true },
        ] {
            for invalid in [
                "const value = 'a\nb';\n",
                "const value = \"a\r\nb\";\n",
                "const value = 'a\rb';\n",
                "const value = 'a\\\\\nb';\n",
            ] {
                assert!(syntax_counts(language, invalid.as_bytes()).unwrap().errors > 0);
                assert!(first_syntax_diagnostic(language, invalid.as_bytes())
                    .unwrap()
                    .contains("unescaped line break"));
            }
            let invalid = "// π\nconst value = 'a\nb';\n";
            assert!(first_syntax_diagnostic(language, invalid.as_bytes())
                .unwrap()
                .starts_with("2:17 "));
            let two_breaks = "const value = 'a\nb\nc';\n";
            assert_eq!(
                syntax_counts(language, two_breaks.as_bytes())
                    .unwrap()
                    .errors,
                2
            );
        }
    }

    #[test]
    fn js_ts_string_guard_preserves_valid_lexical_contexts() {
        for language in [
            Language::JavaScript,
            Language::TypeScript { tsx: false },
            Language::TypeScript { tsx: true },
        ] {
            for valid in [
                "const value = 'a\\nb';\n",
                "const value = 'a\\\nb';\n",
                "const value = \"a\\\r\nb\";\n",
                "const value = 'a\\\rb';\n",
                "const value = `a\nb`;\n",
                "/* 'a\nb' */ const value = 1;\n",
                "const value = /['\"]/;\n",
                "const value = 'a\u{2028}b\u{2029}c';\n",
            ] {
                assert_eq!(
                    syntax_counts(language, valid.as_bytes()).unwrap(),
                    SyntaxCounts {
                        errors: 0,
                        missing: 0
                    },
                    "{valid:?}"
                );
                assert!(first_syntax_diagnostic(language, valid.as_bytes()).is_none());
            }
        }
        for language in [Language::JavaScript, Language::TypeScript { tsx: true }] {
            let valid = "const view = <div title=\"a\nb\" />;\n";
            assert_eq!(
                syntax_counts(language, valid.as_bytes()).unwrap(),
                SyntaxCounts {
                    errors: 0,
                    missing: 0
                }
            );
            let invalid = "const view = <div title={'a\nb'} />;\n";
            assert!(syntax_counts(language, invalid.as_bytes()).unwrap().errors > 0);
        }
    }

    #[test]
    fn c_va_arg_type_operands_use_validation_view_only() {
        let valid = b"#include <stdarg.h>\nvoid *get(int key,...) {va_list ap;va_start(ap,key);void *p=va_arg(ap,void *);va_end(ap);return p;}\n";
        assert_eq!(
            syntax_counts(Language::C, valid),
            Some(SyntaxCounts {
                errors: 0,
                missing: 0
            })
        );
        assert_eq!(first_syntax_diagnostic(Language::C, valid), None);
        let invalid = b"void *get(void){return va_arg(ap,void *+);}";
        let counts = syntax_counts(Language::C, invalid).unwrap();
        assert!(counts.errors > 0 || counts.missing > 0);
        assert!(first_syntax_diagnostic(Language::C, invalid).is_some());
    }

    #[test]
    fn guarded_linkage_crlf_keeps_offsets_and_rejects_malformed_source() {
        let source = b"#ifdef __cplusplus\r\nextern \"C\" {\r\n#endif\r\nint value;\r\n#ifdef __cplusplus\r\n}\r\n#endif\r\n";
        let view = guarded_linkage_validation_content(Language::C, source);
        assert!(matches!(view, Cow::Owned(_)));
        assert_eq!(view.len(), source.len());
        for (before, after) in source.iter().zip(view.iter()) {
            if matches!(*before, b'\r' | b'\n') {
                assert_eq!(before, after);
            }
        }
        assert_eq!(
            syntax_counts(Language::C, source).unwrap(),
            SyntaxCounts {
                errors: 0,
                missing: 0
            }
        );
        let invalid = String::from_utf8(source.to_vec())
            .unwrap()
            .replace("int value;", "int value(");
        let counts = syntax_counts(Language::C, invalid.as_bytes()).unwrap();
        assert!(counts.errors > 0 || counts.missing > 0);
        let lookalike = format!("/*\r\n{}*/\r\n", String::from_utf8_lossy(source));
        assert!(matches!(
            guarded_linkage_validation_content(Language::C, lookalike.as_bytes()),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn guarded_linkage_lookalikes_keep_valid_comment_and_string_bytes() {
        let pair = "#ifdef __cplusplus\nextern \"C\" {\n#endif\nint value;\n#ifdef __cplusplus\n}\n#endif\n";
        let comment = format!("/*\n{pair}*/\nint value;\n");
        let raw_string = format!("const char *value = R\"guard(\n{pair})guard\";\n");
        // Ordinary strings encode newlines and quotes; they must remain literal
        // data, independently of the raw-string/comment ancestor exclusions.
        let escaped = pair
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        let string = format!("const char *value = \"{escaped}\";\n");
        for (language, source) in [
            (Language::C, comment),
            (Language::Cpp, string),
            (Language::Cpp, raw_string),
        ] {
            let tree = greppy_parser::parse(language, source.as_bytes()).unwrap();
            assert!(
                !tree.root_node().has_error(),
                "{source}\n{}",
                tree.root_node().to_sexp()
            );
            let view = guarded_linkage_validation_content(language, source.as_bytes());
            assert!(
                matches!(view, Cow::Borrowed(_)),
                "lookalike normalized: {source}"
            );
            assert_eq!(view.as_ref(), source.as_bytes());
            assert_eq!(
                syntax_counts(language, source.as_bytes()).unwrap(),
                SyntaxCounts {
                    errors: 0,
                    missing: 0
                }
            );
        }
    }
    use super::*;

    #[test]
    fn json_positive_exponents_are_valid_without_mutating_payloads() {
        let language = syntax_language_for_path(Path::new("evidence.json"), b"{}");
        assert_eq!(language.name(), "json");
        for source in [
            r#"{"finite_max_error": 2.842105616405627e+18}"#,
            r#"[-1E+3, 0e+0, 2.5e-18, "2e+18", "escaped\"2e+18"]"#,
            "1e+9999",
        ] {
            let original = source.as_bytes().to_vec();
            let counts = syntax_counts(language, &original).unwrap();
            assert_eq!(
                counts,
                SyntaxCounts {
                    errors: 0,
                    missing: 0
                },
                "{source}"
            );
            assert!(first_syntax_diagnostic(language, &original).is_none());
            assert_eq!(original, source.as_bytes());
        }
        let quoted = br#"{"text":"2e+18"}"#;
        assert_eq!(json_validation_content(quoted).as_ref(), quoted);
        for source in [
            "{\"x\":2e+}",
            "{\"x\":+2}",
            "{\"x\":2e++18}",
            "{\"x\":2e+18,}",
        ] {
            assert!(matches!(
                json_validation_content(source.as_bytes()),
                Cow::Borrowed(_)
            ));
            assert!(
                syntax_counts(language, source.as_bytes()).unwrap().errors > 0,
                "{source}"
            );
            assert!(first_syntax_diagnostic(language, source.as_bytes()).is_some());
        }
    }

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
    fn ambiguous_cpp_header_uses_baseline_grammar_without_weakening_c_errors() {
        let before = b"#include <memory>\nnamespace KWin { class GLFramebuffer; class Layer { std::unique_ptr<GLFramebuffer> buffer; }; }\n";
        let after = b"#include <memory>\nnamespace KWin { class GLFramebuffer; class GLRenderTimeQuery; class Layer { std::unique_ptr<GLFramebuffer> buffer; std::unique_ptr<GLRenderTimeQuery> query; }; }\n";
        let language = syntax_language_for_path(Path::new("layer.h"), before);
        assert_eq!(language, Language::Cpp);
        let baseline = syntax_counts(language, before).unwrap();
        assert_eq!(syntax_counts(language, after).unwrap(), baseline);
        let malformed =
            b"namespace KWin { class Layer { std::unique_ptr<GLFramebuffer> query( ; }; }\n";
        let invalid = syntax_counts(language, malformed).unwrap();
        assert!(invalid.errors > baseline.errors || invalid.missing > baseline.missing);

        let c = b"struct AtomicState { _Atomic(int) value; };\n";
        let c_language = syntax_language_for_path(Path::new("state.h"), c);
        assert_eq!(c_language, Language::C);
        assert_eq!(
            syntax_language_for_path(Path::new("state.c"), before),
            Language::C
        );
        let invalid_c =
            syntax_counts(c_language, b"struct AtomicState { _Atomic(int value; };\n").unwrap();
        let valid_c = syntax_counts(c_language, c).unwrap();
        assert!(invalid_c.errors > valid_c.errors || invalid_c.missing > valid_c.missing);
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
        let raw_tree = greppy_parser::parse(language, valid).unwrap();
        let typeof_start = valid
            .windows(b"typeof".len())
            .position(|window| window == b"typeof")
            .unwrap();
        let mut ancestor = raw_tree
            .root_node()
            .descendant_for_byte_range(typeof_start, typeof_start + b"typeof".len());
        assert!(
            std::iter::from_fn(|| {
                let node = ancestor?;
                ancestor = node.parent();
                Some(node.kind())
            })
            .collect::<Vec<_>>()
            .windows(2)
            .any(|kinds| kinds == ["unary_expression", "binary_expression"]),
            "the raw recovery tree must retain the observed unary/binary import-type shape"
        );
        let suffix_start = valid
            .windows(b">()".len())
            .position(|window| window == b">()")
            .unwrap();
        let error = raw_tree
            .root_node()
            .descendant_for_byte_range(suffix_start, suffix_start + b">()".len())
            .expect("ERROR covering the recovered generic call suffix");
        assert!(error.is_error());
        assert_eq!(&valid[error.start_byte()..error.end_byte()], b">()");
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
            b"const value = call<typeof\n  import(\n    \"node:child_process\"\n  )>();".as_slice(),
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
            br#"const value = `typeof import("node:child_process")` + ;"#.as_slice(),
            br#"const r = /typeof import("/")/;"#.as_slice(),
        ] {
            assert!(
                syntax_counts(language, lexically_invalid)
                    .is_some_and(|counts| counts.errors + counts.missing > 0),
                "malformed comment/string/template/regex must not be hidden: {}",
                String::from_utf8_lossy(lexically_invalid)
            );
            assert_eq!(
                syntax_validation_content(language, lexically_invalid).as_ref(),
                lexically_invalid,
                "comment/string/template/regex contents must never be rewritten"
            );
        }

        let escaped_newline = b"type ChildProcess = typeof import(\"node:\\\nchild_process\");";
        assert_eq!(
            syntax_validation_content(language, escaped_newline).as_ref(),
            escaped_newline,
            "line continuations must not be rewritten because that would move diagnostics"
        );
    }

    #[test]
    fn typescript_typed_tag_validation_preserves_real_failures() {
        for tsx in [false, true] {
            let language = Language::TypeScript { tsx };
            for valid in [
                "function* run() { const rows = yield* sql<{ readonly workspace_root: string | null }>`SELECT workspace_root`; return rows; }",
                "const rows = db.sql<Array<{ id: number }>>`SELECT ${id}`;",
                "const rows = sql<{id: | string | null}>`SELECT id`;",
                "const rows = sql<{id: \"|}\"}>`SELECT id`;",
                "const rows = sql<{id: string /* |} */}>`SELECT id`;",

                "const rows = sql<\n{ readonly id: number },\nstring\n>`SELECT id`;",
            ] {
                let counts = syntax_counts(language, valid.as_bytes()).unwrap();
                assert_eq!((counts.errors, counts.missing), (0, 0), "{valid}");
                assert!(first_syntax_diagnostic(language, valid.as_bytes()).is_none());
                let view = syntax_validation_content(language, valid.as_bytes());
                assert_eq!(view.len(), valid.len());
                assert_eq!(view.iter().enumerate().filter(|(_, b)| **b == b'\n').map(|(i, _)| i).collect::<Vec<_>>(), valid.bytes().enumerate().filter(|(_, b)| *b == b'\n').map(|(i, _)| i).collect::<Vec<_>>());
            }
            for invalid in [
                "const rows = sql<{ id: }>`SELECT id`;",
                "const rows = sql<{id:string|}>`SELECT id`;",
                "const rows = sql<{|id:string|}>`SELECT id`;",

                "const rows = sql<{id:string&}>`SELECT id`;",
                "function* run() {const rows=yield* sql<{readonly workspace_root:string|}>`SELECT workspace_root`;return rows;}",

                "const rows = sql<{ id: number }>`SELECT id;",
                "const rows = sql<{ id: number }>`SELECT ${}`;",
                "const rows = sql<{ id: number }>`SELECT id`; const broken = ;",
                "const rows = sql<{ id: number }>!`SELECT id`; const broken = ;",
            ] {
                let counts = syntax_counts(language, invalid.as_bytes()).unwrap();
                assert!(counts.errors + counts.missing > 0, "{invalid}");
                assert!(first_syntax_diagnostic(language, invalid.as_bytes()).is_some());
            }
            for literal in [
                "const s = 'sql<{ id: number }>`SELECT id`';",
                "// sql<{ id: number }>`SELECT id`\nconst x = 1;",
                "const s = `sql<{ id: number }> SELECT id`;",
                "const r = /sql<id>!/;",
            ] {
                assert_eq!(
                    syntax_validation_content(language, literal.as_bytes()).as_ref(),
                    literal.as_bytes()
                );
            }
        }
    }

    #[test]
    fn bash_readwrite_redirect_validation_preserves_real_failures() {
        for valid in [
            "#!/bin/bash\nexec 9<>/mnt/nvme1/.greppy-heavy.lock\n",
            "exec <> file",
            "exec 9<>\"space name\"",
            "exec 9<>$lock; echo ok",
        ] {
            let counts = syntax_counts(Language::Bash, valid.as_bytes()).unwrap();
            assert_eq!((counts.errors, counts.missing), (0, 0), "{valid}");
            assert!(first_syntax_diagnostic(Language::Bash, valid.as_bytes()).is_none());
            assert_eq!(bash_validation_content(valid.as_bytes()).len(), valid.len());
        }
        for invalid in [
            "exec 9< >file",
            "exec 9<>\n",
            "exec 9<>>file",
            "exec 9<>file; if then",
            "exec 9<>\"unclosed",
        ] {
            let counts = syntax_counts(Language::Bash, invalid.as_bytes()).unwrap();
            assert!(counts.errors + counts.missing > 0, "{invalid}");
            assert!(first_syntax_diagnostic(Language::Bash, invalid.as_bytes()).is_some());
        }
        for literal in [
            "echo '9<>file'",
            "# exec 9<>file\necho ok",
            "cat <<EOF\n9<>file\nEOF\n",
        ] {
            assert_eq!(
                bash_validation_content(literal.as_bytes()).as_ref(),
                literal.as_bytes()
            );
        }
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
