//! Structured Markdown to ratatui lines, with a best-effort code highlighter.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::theme::Theme;

pub fn render_markdown(source: &str, theme: Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut rest = source;
    while !rest.is_empty() {
        if rest.starts_with("```") {
            let after = &rest[3..];
            let (lang, body_start) = match after.find('\n') {
                Some(idx) => (after[..idx].trim().to_string(), idx + 1),
                None => (String::new(), after.len()),
            };
            let body = &after[body_start..];
            if let Some(end) = body.find("\n```") {
                let code = &body[..end];
                lines.extend(render_code_block(&lang, code, theme));
                rest = body[end + 4..].trim_start_matches('\n');
                if rest.starts_with("```") {
                    rest = rest.trim_start_matches('`');
                    if let Some(stripped) = rest.strip_prefix('\n') {
                        rest = stripped;
                    }
                }
                continue;
            }
            if let Some(end) = body.find("```") {
                lines.extend(render_code_block(&lang, &body[..end], theme));
                rest = &body[end + 3..];
                continue;
            }
            lines.extend(render_code_block(&lang, body, theme));
            break;
        }

        let (line, tail) = split_line(rest);
        rest = tail;
        if line.is_empty() {
            lines.push(Line::default());
            continue;
        }
        if let Some(heading) = heading(line) {
            lines.push(Line::from(Span::styled(
                enrich_inline(&heading, theme),
                theme.heading().add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if let Some(item) = list_item(line) {
            let mut spans = vec![Span::raw("  • ".to_string())];
            spans.extend(inline_spans(&item, theme));
            lines.push(Line::from(spans));
            continue;
        }
        lines.push(Line::from(inline_spans(line, theme)));
    }
    if lines.is_empty() {
        lines.push(Line::raw(source.to_string()));
    }
    lines
}

fn split_line(source: &str) -> (&str, &str) {
    match source.find('\n') {
        Some(idx) => (&source[..idx], &source[idx + 1..]),
        None => (source, ""),
    }
}

fn heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|ch| *ch == '#').count();
    if (1..=6).contains(&hashes) && trimmed.as_bytes().get(hashes) == Some(&b' ') {
        Some(trimmed[hashes..].trim().to_string())
    } else {
        None
    }
}

fn list_item(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    for prefix in ["- ", "* ", "+ "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return Some(rest.to_string());
        }
    }
    let digits = trimmed.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digits > 0 {
        if let Some(rest) = trimmed[digits..].strip_prefix(". ") {
            return Some(rest.to_string());
        }
    }
    None
}

fn inline_spans(line: &str, theme: Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let enriched_line = enrich_inline(line, theme);
    let mut rest = enriched_line.as_str();
    while !rest.is_empty() {
        if let Some(stripped) = rest.strip_prefix('`') {
            if let Some(end) = stripped.find('`') {
                spans.push(Span::styled(
                    stripped[..end].to_string(),
                    theme.inline_code(),
                ));
                rest = &stripped[end + 1..];
                continue;
            }
        }
        if let Some(stripped) = rest.strip_prefix('[') {
            if let Some(label_end) = stripped.find(']') {
                if stripped[label_end..].starts_with("](") {
                    let after = &stripped[label_end + 2..];
                    if let Some(url_end) = after.find(')') {
                        let label = &stripped[..label_end];
                        spans.push(Span::styled(label.to_string(), theme.link()));
                        rest = &after[url_end + 1..];
                        continue;
                    }
                }
            }
        }
        if let Some(stripped) = rest.strip_prefix("**") {
            if let Some(end) = stripped.find("**") {
                spans.push(Span::styled(
                    stripped[..end].to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                rest = &stripped[end + 2..];
                continue;
            }
        }
        let next = rest.find(['`', '[', '*']).unwrap_or(rest.len());
        if next == 0 {
            let ch = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            spans.push(Span::raw(rest[..ch].to_string()));
            rest = &rest[ch..];
        } else {
            spans.push(Span::raw(rest[..next].to_string()));
            rest = &rest[next..];
        }
    }
    spans
}

fn render_code_block(lang: &str, code: &str, theme: Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let language = highlight_language(lang);
    let label = language_label(lang, language);
    if !label.is_empty() {
        lines.push(Line::from(Span::styled(label, theme.muted())));
    }
    if language == HighlightLanguage::Mermaid {
        lines.extend(render_mermaid_lines(code, theme));
        return lines;
    }
    for raw in code.split('\n') {
        lines.push(Line::from(highlight_line(lang, raw, theme)));
    }
    lines
}

fn highlight_line(lang: &str, line: &str, theme: Theme) -> Vec<Span<'static>> {
    let keywords = keywords_for(lang);
    if keywords.is_empty() {
        return vec![Span::styled(line.to_string(), theme.code_block())];
    }
    let mut spans = Vec::new();
    let mut rest = line;
    while !rest.is_empty() {
        if rest.starts_with("//") || rest.starts_with('#') {
            spans.push(Span::styled(rest.to_string(), theme.muted()));
            break;
        }
        if rest.starts_with('"') || rest.starts_with('\'') {
            let quote = rest.as_bytes()[0] as char;
            let mut end = 1;
            let bytes = rest.as_bytes();
            while end < rest.len() {
                if bytes[end] == b'\\' && end + 1 < rest.len() {
                    end += 2;
                    continue;
                }
                if bytes[end] == quote as u8 {
                    end += 1;
                    break;
                }
                end += 1;
            }
            spans.push(Span::styled(rest[..end].to_string(), theme.inline_code()));
            rest = &rest[end..];
            continue;
        }
        let ch = rest.chars().next().unwrap();
        if ch.is_ascii_digit() {
            let n = rest
                .find(|c: char| !c.is_ascii_digit() && c != '.')
                .unwrap_or(rest.len());
            spans.push(Span::styled(rest[..n].to_string(), theme.user()));
            rest = &rest[n..];
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let n = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            let word = &rest[..n];
            let style = if keywords.contains(&word) {
                theme.status()
            } else {
                theme.code_block()
            };
            spans.push(Span::styled(word.to_string(), style));
            rest = &rest[n..];
            continue;
        }
        let len = ch.len_utf8();
        spans.push(Span::styled(rest[..len].to_string(), theme.code_block()));
        rest = &rest[len..];
    }
    spans
}

fn keywords_for(lang: &str) -> &'static [&'static str] {
    match highlight_language(lang) {
        HighlightLanguage::Rust => &[
            "fn", "let", "mut", "pub", "struct", "enum", "impl", "use", "mod", "match", "if",
            "else", "return", "self", "Self", "async", "await", "const", "static",
        ],
        HighlightLanguage::Python => &[
            "def", "class", "return", "import", "from", "if", "else", "elif", "for", "while",
            "yield", "async", "await", "self",
        ],
        HighlightLanguage::JavaScript | HighlightLanguage::TypeScript => &[
            "function", "const", "let", "return", "class", "if", "else", "await", "async",
            "import", "export",
        ],
        HighlightLanguage::Bash => &["if", "then", "fi", "do", "done", "export", "return"],
        HighlightLanguage::Go => &[
            "package",
            "import",
            "func",
            "return",
            "var",
            "const",
            "type",
            "struct",
            "if",
            "else",
            "for",
            "range",
            "go",
            "defer",
            "interface",
        ],
        HighlightLanguage::C | HighlightLanguage::Cpp => &[
            "auto",
            "bool",
            "break",
            "case",
            "char",
            "class",
            "const",
            "else",
            "for",
            "if",
            "include",
            "int",
            "namespace",
            "nullptr",
            "return",
            "struct",
            "switch",
            "this",
            "void",
            "while",
        ],
        HighlightLanguage::Json => &["true", "false", "null"],
        HighlightLanguage::Toml
        | HighlightLanguage::Yaml
        | HighlightLanguage::Diff
        | HighlightLanguage::Markdown
        | HighlightLanguage::Mermaid
        | HighlightLanguage::Plain => &[],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HighlightLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Bash,
    Go,
    C,
    Cpp,
    Json,
    Toml,
    Yaml,
    Diff,
    Markdown,
    Mermaid,
    Plain,
}

fn fence_tag(lang: &str) -> &str {
    lang.split_whitespace()
        .next()
        .unwrap_or("")
        .split([',', ';'])
        .next()
        .unwrap_or("")
}

fn highlight_language(lang: &str) -> HighlightLanguage {
    match fence_tag(lang).to_ascii_lowercase().as_str() {
        "rust" | "rs" => HighlightLanguage::Rust,
        "python" | "py" => HighlightLanguage::Python,
        "javascript" | "js" => HighlightLanguage::JavaScript,
        "typescript" | "ts" | "tsx" => HighlightLanguage::TypeScript,
        "bash" | "sh" | "zsh" | "shell" => HighlightLanguage::Bash,
        "go" | "golang" => HighlightLanguage::Go,
        "c" => HighlightLanguage::C,
        "cpp" | "c++" | "cc" | "cxx" => HighlightLanguage::Cpp,
        "json" | "jsonc" => HighlightLanguage::Json,
        "toml" => HighlightLanguage::Toml,
        "yaml" | "yml" => HighlightLanguage::Yaml,
        "diff" | "patch" => HighlightLanguage::Diff,
        "markdown" | "md" => HighlightLanguage::Markdown,
        "mermaid" | "mmd" => HighlightLanguage::Mermaid,
        _ => HighlightLanguage::Plain,
    }
}

fn language_label(lang: &str, language: HighlightLanguage) -> String {
    let tag = fence_tag(lang);
    if tag.is_empty() {
        return String::new();
    }
    match language {
        HighlightLanguage::Rust => "rust".to_string(),
        HighlightLanguage::Python => "python".to_string(),
        HighlightLanguage::JavaScript => "javascript".to_string(),
        HighlightLanguage::TypeScript => "typescript".to_string(),
        HighlightLanguage::Bash => "bash".to_string(),
        HighlightLanguage::Go => "go".to_string(),
        HighlightLanguage::C => "c".to_string(),
        HighlightLanguage::Cpp => "cpp".to_string(),
        HighlightLanguage::Json => "json".to_string(),
        HighlightLanguage::Toml => "toml".to_string(),
        HighlightLanguage::Yaml => "yaml".to_string(),
        HighlightLanguage::Diff => "diff".to_string(),
        HighlightLanguage::Markdown => "markdown".to_string(),
        HighlightLanguage::Mermaid => "mermaid".to_string(),
        HighlightLanguage::Plain => tag.to_string(),
    }
}

/// Convert common LaTeX commands to legible Unicode without matching a
/// command prefix (for example, `\\to` must not alter `\\top`).
fn latex_to_unicode(latex: &str) -> String {
    let replacements = [
        (r"\alpha", "α"),
        (r"\beta", "β"),
        (r"\gamma", "γ"),
        (r"\delta", "δ"),
        (r"\epsilon", "ε"),
        (r"\theta", "θ"),
        (r"\lambda", "λ"),
        (r"\mu", "μ"),
        (r"\pi", "π"),
        (r"\sigma", "σ"),
        (r"\tau", "τ"),
        (r"\phi", "φ"),
        (r"\omega", "ω"),
        (r"\times", "×"),
        (r"\div", "÷"),
        (r"\pm", "±"),
        (r"\le", "≤"),
        (r"\ge", "≥"),
        (r"\ne", "≠"),
        (r"\approx", "≈"),
        (r"\infty", "∞"),
        (r"\to", "→"),
        (r"\gets", "←"),
        (r"\sum", "∑"),
        (r"\prod", "∏"),
        (r"\sqrt", "√"),
    ];
    let mut out = String::with_capacity(latex.len());
    let mut cursor = 0;
    while cursor < latex.len() {
        let remaining = &latex[cursor..];
        if let Some(&(from, to)) = replacements.iter().find(|&&(from, _)| {
            remaining.strip_prefix(from).is_some_and(|suffix| {
                suffix
                    .chars()
                    .next()
                    .is_none_or(|next| !next.is_ascii_alphabetic())
            })
        }) {
            out.push_str(to);
            cursor += from.len();
            continue;
        }
        let Some(character) = remaining.chars().next() else {
            break;
        };
        out.push(character);
        cursor += character.len_utf8();
    }
    out
}

fn hex_swatch_marker(theme: Theme) -> &'static str {
    if theme.ascii {
        "[#]"
    } else {
        "■"
    }
}

/// Prefix standalone three- or six-digit hex colors with a printable swatch.
/// The marker is deliberately textual, so it remains safe for ratatui and
/// terminals with color disabled.
fn render_hex_swatches(text: &str, theme: Theme) -> String {
    let mut out = String::with_capacity(text.len() + 32);
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '#' {
            let prev_ok = i == 0 || !chars[i - 1].is_ascii_alphanumeric();
            let digit_count = [6, 3].into_iter().find(|digit_count| {
                let end = i + 1 + digit_count;
                end <= chars.len()
                    && chars[i + 1..end].iter().all(char::is_ascii_hexdigit)
                    && chars
                        .get(end)
                        .is_none_or(|next| !next.is_ascii_alphanumeric())
            });
            if prev_ok {
                if let Some(digit_count) = digit_count {
                    let end = i + 1 + digit_count;
                    out.push_str(hex_swatch_marker(theme));
                    out.push(' ');
                    out.extend(chars[i..end].iter().copied());
                    i = end;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn is_path_or_url(chunk: &str) -> bool {
    chunk.contains('/')
        || chunk.contains("://")
        || chunk.contains(":\\")
        || chunk.starts_with("\\\\")
}

fn enrich_plain(text: &str, theme: Theme) -> String {
    let mut out = String::with_capacity(text.len());
    let mut start = 0;
    let mut whitespace = None;
    for (index, character) in text.char_indices() {
        let current_whitespace = character.is_whitespace();
        if whitespace.is_some_and(|was_whitespace| was_whitespace != current_whitespace) {
            let chunk = &text[start..index];
            if whitespace == Some(true) || is_path_or_url(chunk) {
                out.push_str(chunk);
            } else {
                out.push_str(&render_hex_swatches(&latex_to_unicode(chunk), theme));
            }
            start = index;
        }
        whitespace = Some(current_whitespace);
    }
    let chunk = &text[start..];
    if whitespace == Some(true) || is_path_or_url(chunk) {
        out.push_str(chunk);
    } else {
        out.push_str(&render_hex_swatches(&latex_to_unicode(chunk), theme));
    }
    out
}

fn inline_code_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let delimiter_len = bytes[start..]
        .iter()
        .take_while(|byte| **byte == b'`')
        .count();
    let mut cursor = start + delimiter_len;
    while cursor < bytes.len() {
        if bytes[cursor] != b'`' {
            cursor += 1;
            continue;
        }
        let run_len = bytes[cursor..]
            .iter()
            .take_while(|byte| **byte == b'`')
            .count();
        if run_len == delimiter_len {
            return Some(cursor + run_len);
        }
        cursor += run_len;
    }
    None
}

fn link_destination_end(text: &str, start: usize) -> Option<usize> {
    if !text[start..].starts_with("](") {
        return None;
    }
    let bytes = text.as_bytes();
    let mut depth = 1_usize;
    let mut cursor = start + 2;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'\\' => cursor = (cursor + 2).min(bytes.len()),
            b'(' => {
                depth += 1;
                cursor += 1;
            }
            b')' => {
                depth -= 1;
                cursor += 1;
                if depth == 0 {
                    return Some(cursor);
                }
            }
            _ => cursor += 1,
        }
    }
    None
}

fn enrich_inline(text: &str, theme: Theme) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut plain_start = 0;
    let mut cursor = 0;
    while cursor < bytes.len() {
        let protected_end = match bytes[cursor] {
            b'`' => inline_code_end(text, cursor),
            b']' => link_destination_end(text, cursor),
            b'<' => text[cursor + 1..]
                .find('>')
                .map(|offset| cursor + offset + 2),
            _ => None,
        };
        if let Some(end) = protected_end {
            out.push_str(&enrich_plain(&text[plain_start..cursor], theme));
            out.push_str(&text[cursor..end]);
            cursor = end;
            plain_start = end;
        } else {
            let character_len = text[cursor..].chars().next().map_or(1, char::len_utf8);
            cursor += character_len;
        }
    }
    out.push_str(&enrich_plain(&text[plain_start..], theme));
    out
}

fn render_mermaid_lines(source: &str, theme: Theme) -> Vec<Line<'static>> {
    const MAX_WIDTH: usize = 80;
    let max_width = MAX_WIDTH;
    let line_limit = max_width.saturating_sub(4);
    let (left, right, horizontal) = if theme.ascii {
        ('+', '+', '-')
    } else {
        ('┌', '┐', '─')
    };
    let (vertical, bottom_left, bottom_right) = if theme.ascii {
        ('|', '+', '+')
    } else {
        ('│', '└', '┘')
    };
    let title = " [Mermaid Diagram] ";
    let mut top = String::new();
    top.push(left);
    top.push_str(&horizontal.to_string().repeat(2));
    top.push_str(title);
    while top.chars().count() < max_width.saturating_sub(1) {
        top.push(horizontal);
    }
    top.push(right);
    let mut lines = vec![Line::from(Span::styled(top, theme.code_block()))];
    for source_line in source.lines() {
        let trimmed = source_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let content: String = trimmed.chars().take(line_limit).collect();
        lines.push(Line::from(Span::styled(
            format!(
                "{vertical} {content:<width$} {vertical}",
                width = line_limit
            ),
            theme.code_block(),
        )));
    }
    let bottom = format!(
        "{bottom_left}{}{bottom_right}",
        horizontal.to_string().repeat(max_width.saturating_sub(2))
    );
    lines.push(Line::from(Span::styled(bottom, theme.code_block())));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_headings_lists_code_and_links() {
        let theme = Theme {
            color: false,
            ascii: true,
        };
        let lines = render_markdown(
            "# Title\n\n- item `code`\n\nSee [docs](https://example.test).\n\n```rust\nfn main() {}\n```\n",
            theme,
        );
        let joined: String = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("Title"));
        assert!(joined.contains("item"));
        assert!(joined.contains("docs"));
        assert!(joined.contains("fn"));
        assert!(joined.contains("main"));
    }

    #[test]
    fn normalizes_fence_aliases_and_highlights_extended_languages() {
        let theme = Theme {
            color: false,
            ascii: true,
        };
        let lines = render_markdown("```TSX strict\nconst answer = 42;\n```", theme);
        assert_eq!(lines[0].spans[0].content, "typescript");
        assert!(lines[1].spans.iter().any(|span| span.content == "const"));
        assert!(lines[1].spans.iter().any(|span| span.content == "42"));
    }

    #[test]
    fn enriches_prose_but_preserves_code_and_destinations() {
        let theme = Theme {
            color: false,
            ascii: true,
        };
        let lines = render_markdown(
            r"Math \alpha and color #abc. Inline `\beta #123456` and [link](https://example.test/#abc).",
            theme,
        );
        let joined: String = lines
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.content.as_ref()))
            .collect();
        assert!(joined.contains("α"));
        assert!(joined.contains("[#] #abc"));
        assert!(joined.contains(r"\beta #123456"));
        assert!(joined.contains("link"));
        assert!(!joined.contains("https://example.test/"));
    }

    #[test]
    fn renders_mermaid_as_deterministic_plain_text() {
        let theme = Theme {
            color: false,
            ascii: true,
        };
        let lines = render_markdown("```mmd\ngraph TD\n  A --> B\n```", theme);
        let joined: String = lines
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.content.as_ref()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("+-- [Mermaid Diagram]"));
        assert!(joined.contains("A --> B"));
        assert!(!joined.contains('\u{1b}'));
    }
}
