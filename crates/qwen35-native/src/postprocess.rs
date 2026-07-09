pub const MAX_BRIEF_BULLET_CHARS: usize = 140;
pub const MAX_TRIAGE_REASON_CHARS: usize = 48;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriageVerdict {
    pub read: bool,
    pub reason: String,
}

const PREFIXES: &[&str] = &[
    "summary:",
    "purpose:",
    "function:",
    "the function:",
    "this function:",
    "answer:",
];

pub fn postprocess_brief_output(raw: &str, prompt: &str) -> Vec<String> {
    if looks_like_thinking_output(raw) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut in_code_fence = false;
    for raw_line in raw.lines() {
        let mut line = raw_line.trim();
        if line.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence || line.is_empty() {
            continue;
        }
        line = strip_bullet_marker(line);
        line = strip_known_prefix(line);
        let Some(mut line) = normalize_statement(line) else {
            continue;
        };
        let Some(rewritten) = rewrite_brief_statement(&line) else {
            continue;
        };
        line = rewritten;
        if looks_like_prompt_echo(&line, prompt) {
            return Vec::new();
        }
        if looks_like_code(&line) {
            continue;
        }
        if looks_malformed_brief_statement(&line) {
            continue;
        }
        let line = cap_line(&line, MAX_BRIEF_BULLET_CHARS);
        if !line.is_empty() && !out.iter().any(|prev| prev == &line) {
            out.push(line);
        }
        if out.len() == 2 {
            break;
        }
    }
    out
}

pub fn postprocess_triage_output(raw: &str, prompt: &str) -> TriageVerdict {
    if looks_like_thinking_output(raw) {
        return conservative_read();
    }
    let Some(line) = first_clean_triage_line(raw) else {
        return conservative_read();
    };
    if looks_like_triage_prompt_echo(&line, prompt) {
        return conservative_read();
    }
    parse_triage_line(&line).unwrap_or_else(conservative_read)
}

fn first_clean_triage_line(raw: &str) -> Option<String> {
    let mut in_code_fence = false;
    for raw_line in raw.lines() {
        let mut line = raw_line.trim();
        if line.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence || line.is_empty() {
            continue;
        }
        line = strip_bullet_marker(line);
        line = strip_triage_prefix(line);
        let Some(line) = normalize_statement(line) else {
            continue;
        };
        if !line.is_empty() {
            return Some(line);
        }
    }
    None
}

fn strip_triage_prefix(line: &str) -> &str {
    let lower = line.to_ascii_lowercase();
    for prefix in ["verdict:", "decision:", "result:", "answer:"] {
        if lower.starts_with(prefix) {
            return line[prefix.len()..].trim_start();
        }
    }
    line
}

fn parse_triage_line(line: &str) -> Option<TriageVerdict> {
    let line = line.trim();
    let lower = line.to_ascii_lowercase();
    let (read, rest) = if lower.starts_with("read") {
        (true, &line[4..])
    } else if lower.starts_with("skip") {
        (false, &line[4..])
    } else {
        return None;
    };
    let Some(reason) = normalize_triage_reason(rest, read) else {
        return Some(conservative_read());
    };
    Some(TriageVerdict { read, reason })
}

fn normalize_triage_reason(raw: &str, read: bool) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_start_matches(['-', ':', '|', '—', '–'])
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim();
    let mut words = trimmed
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| c == ',' || c == '.' || c == ';' || c == ':'))
        .filter(|w| !w.is_empty())
        .take(5)
        .collect::<Vec<_>>()
        .join(" ");
    if words.is_empty() {
        words = if read {
            "possible relevant span".to_string()
        } else {
            "likely unrelated span".to_string()
        };
    }
    if looks_malformed_short_text(&words) {
        return None;
    }
    Some(cap_line(&words, MAX_TRIAGE_REASON_CHARS))
}

fn conservative_read() -> TriageVerdict {
    TriageVerdict {
        read: true,
        reason: "uncertain relevant span".to_string(),
    }
}

fn strip_bullet_marker(mut line: &str) -> &str {
    loop {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("- ") {
            line = rest;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* ") {
            line = rest;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("+ ") {
            line = rest;
            continue;
        }
        if let Some(rest) = strip_numbered_prefix(trimmed) {
            line = rest;
            continue;
        }
        return trimmed;
    }
}

fn strip_numbered_prefix(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i + 1 >= bytes.len() {
        return None;
    }
    if (bytes[i] == b'.' || bytes[i] == b')') && bytes[i + 1].is_ascii_whitespace() {
        Some(line[i + 2..].trim_start())
    } else {
        None
    }
}

fn strip_known_prefix(line: &str) -> &str {
    let lower = line.to_ascii_lowercase();
    for prefix in PREFIXES {
        if lower.starts_with(prefix) {
            return line[prefix.len()..].trim_start();
        }
    }
    line
}

fn normalize_statement(line: &str) -> Option<String> {
    let line = line
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .trim_end_matches([';', ','])
        .trim();
    if line.is_empty() {
        return None;
    }
    Some(strip_inline_markdown(line))
}

fn rewrite_brief_statement(line: &str) -> Option<String> {
    let line = trim_discourse_tail(line).trim();
    if line.is_empty() {
        return None;
    }
    let lower = line.to_ascii_lowercase();
    if lower.contains("known as") {
        return None;
    }
    for prefix in [
        "this code snippet describes a logic that ",
        "this code snippet describes that ",
        "this code snippet describes ",
        "this code describes a logic that ",
        "this code describes that ",
        "this code describes ",
        "this function is a helper utility designed to ",
        "this function is primarily designed to ",
        "this function is designed to ",
        "this function is used to ",
        "this function is primarily for ",
        "this function is for ",
    ] {
        if lower.starts_with(prefix) {
            let rest = clean_rewritten_brief(line[prefix.len()..].trim());
            return (!rest.is_empty()).then_some(rest);
        }
    }
    if let Some(idx) = lower.find(" designed to ") {
        let rest = clean_rewritten_brief(line[idx + " designed to ".len()..].trim());
        if !rest.is_empty() {
            return Some(rest);
        }
    }
    Some(clean_rewritten_brief(line))
}

fn clean_rewritten_brief(line: &str) -> String {
    line.split_once(" (")
        .map(|(head, _)| head)
        .unwrap_or(line)
        .trim_end_matches(',')
        .trim()
        .to_string()
}

fn trim_discourse_tail(line: &str) -> &str {
    let lower = line.to_ascii_lowercase();
    let mut end = line.len();
    for marker in [
        " specifically,",
        " the logic is as follows",
        " here is a breakdown",
        " when you ",
        " in short:",
    ] {
        if let Some(idx) = lower.find(marker) {
            end = end.min(idx);
        }
    }
    line[..end].trim()
}

fn strip_inline_markdown(line: &str) -> String {
    line.chars()
        .filter(|c| !matches!(c, '`' | '*'))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn looks_like_prompt_echo(statement: &str, prompt: &str) -> bool {
    let lower = statement.to_ascii_lowercase();
    if lower.contains("summarize: what is this function for?") {
        return true;
    }
    let prompt_first_code_line = prompt
        .lines()
        .skip_while(|line| !line.trim_start().starts_with("fn "))
        .next()
        .map(str::trim)
        .unwrap_or("");
    !prompt_first_code_line.is_empty() && statement.trim() == prompt_first_code_line
}

fn looks_like_triage_prompt_echo(statement: &str, prompt: &str) -> bool {
    let lower = statement.to_ascii_lowercase();
    lower.contains("given the user's question and one code span")
        || lower.contains("do not answer the question")
        || statement.trim() == prompt.lines().next().unwrap_or_default().trim()
}

fn looks_like_thinking_output(raw: &str) -> bool {
    let lower = raw.to_ascii_lowercase();
    lower.contains("<think>")
        || lower.contains("</think>")
        || lower.contains("[start thinking]")
        || lower.contains("thinking process")
        || lower.contains("analyze the request")
        || lower.contains("let's analyze")
}

fn looks_like_code(statement: &str) -> bool {
    let s = statement.trim_start();
    s.starts_with("fn ")
        || s.starts_with("pub fn ")
        || s.starts_with("async fn ")
        || s.starts_with("impl ")
        || s.starts_with("class ")
        || s.contains(" {")
        || s.ends_with('{')
        || s.contains(");")
        || s.contains("->")
}

fn looks_malformed_brief_statement(statement: &str) -> bool {
    let s = statement.trim();
    let lower = s.to_ascii_lowercase();
    if lower.contains("here is a breakdown") || lower.starts_with("in short:") {
        return true;
    }
    if looks_malformed_short_text(s) {
        return true;
    }
    let words = s.split_whitespace().collect::<Vec<_>>();
    let alpha_words = words
        .iter()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .count();
    if alpha_words == 0 {
        return true;
    }
    if words.len() >= 8 {
        let tiny_words = words
            .iter()
            .filter(|word| {
                let letters = word.chars().filter(|c| c.is_alphabetic()).count();
                letters > 0 && letters <= 2
            })
            .count();
        if tiny_words * 2 > words.len() {
            return true;
        }
    }
    words.iter().any(|word| looks_like_noise_word(word))
}

fn looks_malformed_short_text(text: &str) -> bool {
    if text.contains('\u{fffd}') || text.chars().any(|c| c.is_control() && c != '\t') {
        return true;
    }
    let mut letters = 0usize;
    let mut non_latin_letters = 0usize;
    for c in text.chars() {
        if c.is_alphabetic() {
            letters += 1;
            if !is_latin_letter(c) {
                non_latin_letters += 1;
            }
        }
    }
    letters > 0 && non_latin_letters > 0
}

fn is_latin_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
        || matches!(
            c as u32,
            0x00c0..=0x024f | 0x1e00..=0x1eff
        )
        || c == 'µ'
}

fn looks_like_noise_word(word: &str) -> bool {
    let letters = word.chars().filter(|c| c.is_alphabetic()).count();
    if letters < 24 {
        return false;
    }
    let vowels = word
        .chars()
        .filter(|c| matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u' | 'y'))
        .count();
    vowels * 8 < letters
}

fn cap_line(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_string();
    }
    let candidate = line
        .split_once(" (")
        .map(|(head, _)| head)
        .filter(|head| !head.trim().is_empty() && head.chars().count() <= max_chars)
        .unwrap_or(line);
    if candidate.chars().count() <= max_chars {
        return candidate.trim_end_matches(['.', ',']).trim().to_string();
    }
    let mut out = candidate.chars().take(max_chars).collect::<String>();
    while !out.is_empty() && !out.ends_with(char::is_whitespace) {
        out.pop();
    }
    if out.trim().is_empty() {
        out = candidate.chars().take(max_chars).collect();
    }
    out.trim_end_matches(['.', ',']).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brief_prompt;

    #[test]
    fn cleans_prefixes_and_limits_to_two_lines() {
        let prompt = brief_prompt("fn dispatch_brief() {}\n");
        let out = postprocess_brief_output(
            "Summary: Builds the compact symbol briefing.\n\n- Falls back to content search.\n- Extra detail.\n",
            &prompt,
        );
        assert_eq!(
            out,
            vec![
                "Builds the compact symbol briefing.".to_string(),
                "Falls back to content search.".to_string()
            ]
        );
    }

    #[test]
    fn drops_code_fences_and_code_echo() {
        let prompt = brief_prompt("fn dispatch_brief() {}\n");
        let out = postprocess_brief_output(
            "```rust\nfn dispatch_brief() {}\n```\nPurpose: Resolves a symbol briefing.\n",
            &prompt,
        );
        assert_eq!(out, vec!["Resolves a symbol briefing.".to_string()]);
    }

    #[test]
    fn trims_long_parenthetical_without_ellipsis() {
        let prompt = brief_prompt("fn add_user() {}\n");
        let out = postprocess_brief_output(
            "This function is a helper utility designed to add a user's name to a list of strings (likely representing a database record, a JSON object, or a list of items).\n\nHere is a breakdown of what it does:",
            &prompt,
        );
        assert_eq!(
            out,
            vec!["add a user's name to a list of strings".to_string()]
        );
    }

    #[test]
    fn drops_method_call_fragments() {
        let prompt = brief_prompt("fn add_user() {}\n");
        assert!(
            postprocess_brief_output("users.push(name.trim().to_string();:", &prompt).is_empty()
        );
    }

    #[test]
    fn removes_generic_leadin_and_discourse_tail() {
        let prompt = brief_prompt("fn steal_into() {}\n");
        let out = postprocess_brief_output(
            "This code snippet describes a logic that processes tasks from multiple queues into a single queue. Specifically, the rest repeats.",
            &prompt,
        );
        assert_eq!(
            out,
            vec!["processes tasks from multiple queues into a single queue.".to_string()]
        );
    }

    #[test]
    fn drops_known_as_hallucination_shape() {
        let prompt = brief_prompt("fn run_task() {}\n");
        assert!(postprocess_brief_output(
            "This function, known as steal_into, is a standard helper.",
            &prompt
        )
        .is_empty());
    }

    #[test]
    fn drops_prompt_echo() {
        let prompt = brief_prompt("fn dispatch_brief() {}\n");
        assert!(postprocess_brief_output(
            "Summarize: What is this function for?\n\nfn dispatch_brief() {}",
            &prompt
        )
        .is_empty());
    }

    #[test]
    fn drops_malformed_multiscript_gibberish() {
        let prompt = brief_prompt("fn dispatch_brief() {}\n");
        let raw = "bli sport JB ảoillon bankrupt induceonconomykids包istent ro物的یکی旗和平快快ность是一项staولوج-errorsificar万岁жно上新tractorложеныcribedttikenorauingpr";

        assert!(postprocess_brief_output(raw, &prompt).is_empty());
    }

    #[test]
    fn drops_thinking_output() {
        let prompt = brief_prompt("fn dispatch_brief() {}\n");
        let raw =
            "[Start thinking]\nHere's a thinking process that leads to the suggested summary.";

        assert!(postprocess_brief_output(raw, &prompt).is_empty());
    }

    #[test]
    fn keeps_latin_extended_summary_text() {
        let prompt = brief_prompt("fn dispatch_brief() {}\n");

        assert_eq!(
            postprocess_brief_output("Löst den Symbolkontext für brief auf.", &prompt),
            vec!["Löst den Symbolkontext für brief auf.".to_string()]
        );
    }

    #[test]
    fn parses_triage_verdicts() {
        let prompt = crate::triage_prompt("who starts workers?", "x.rs:1", "fn start() {}\n");
        assert_eq!(
            postprocess_triage_output("READ - defines worker startup", &prompt),
            TriageVerdict {
                read: true,
                reason: "defines worker startup".to_string()
            }
        );
        assert_eq!(
            postprocess_triage_output("Verdict: SKIP - unrelated parser helper", &prompt),
            TriageVerdict {
                read: false,
                reason: "unrelated parser helper".to_string()
            }
        );
    }

    #[test]
    fn triage_with_malformed_reason_fails_open_to_read() {
        let prompt = crate::triage_prompt("query", "x.rs:1", "fn f() {}\n");
        assert_eq!(
            postprocess_triage_output("SKIP - 包和平жно", &prompt),
            TriageVerdict {
                read: true,
                reason: "uncertain relevant span".to_string()
            }
        );
    }

    #[test]
    fn triage_thinking_output_fails_open_to_read() {
        let prompt = crate::triage_prompt("query", "x.rs:1", "fn f() {}\n");
        assert_eq!(
            postprocess_triage_output(
                "<think>\nmaybe unrelated\n</think>\nSKIP - unrelated",
                &prompt
            ),
            TriageVerdict {
                read: true,
                reason: "uncertain relevant span".to_string()
            }
        );
    }

    #[test]
    fn triage_fails_open_to_read() {
        let prompt = crate::triage_prompt("query", "x.rs:1", "fn f() {}\n");
        assert_eq!(
            postprocess_triage_output("I cannot tell.", &prompt),
            TriageVerdict {
                read: true,
                reason: "uncertain relevant span".to_string()
            }
        );
    }
}
