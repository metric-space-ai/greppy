//! Bounded source display only; matching, counts and source bytes stay intact.
pub(crate) const MAX_SOURCE_LINE_BYTES: usize = 4096;
const WINDOW_BYTES: usize = 3072;
const LEADING_CONTEXT_BYTES: usize = 512;

pub(crate) struct Preview<'a> {
    pub text: &'a str,
    pub omitted_before: usize,
    pub omitted_after: usize,
}

pub(crate) fn line(text: &str, match_start: Option<usize>) -> Preview<'_> {
    if text.len() <= MAX_SOURCE_LINE_BYTES {
        return Preview {
            text,
            omitted_before: 0,
            omitted_after: 0,
        };
    }
    // An unavailable display anchor must not turn a valid grep hit into an
    // error. The caller labels this as a preview and always offers source access.
    let anchor = match_start.unwrap_or(0).min(text.len());
    let mut start = anchor
        .saturating_sub(LEADING_CONTEXT_BYTES)
        .min(text.len().saturating_sub(WINDOW_BYTES));
    while !text.is_char_boundary(start) {
        start += 1;
    }
    let mut end = (start + WINDOW_BYTES).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    Preview {
        text: &text[start..end],
        omitted_before: start,
        omitted_after: text.len() - end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_lines_are_unchanged() {
        for value in [
            "",
            "short\r\n",
            "🙂 source",
            &"x".repeat(MAX_SOURCE_LINE_BYTES),
        ] {
            let preview = line(value, Some(5));
            assert_eq!(preview.text, value);
            assert_eq!(preview.omitted_before + preview.omitted_after, 0);
        }
    }

    #[test]
    fn huge_line_preserves_middle_match_and_exact_omission_counts() {
        let source = format!("{}needle{}", "x".repeat(100_000), "y".repeat(100_000));
        let preview = line(&source, source.find("needle"));
        assert!(preview.text.contains("needle"));
        assert!(preview.text.len() <= WINDOW_BYTES);
        assert_eq!(
            preview.omitted_before + preview.text.len() + preview.omitted_after,
            source.len()
        );
        assert_eq!(
            &source[preview.omitted_before..source.len() - preview.omitted_after],
            preview.text
        );
    }

    #[test]
    fn utf8_boundary_and_end_anchor_are_safe() {
        let source = format!("{}needle", "🙂".repeat(10_001));
        for anchor in [None, Some(1), source.find("needle"), Some(usize::MAX)] {
            let preview = line(&source, anchor);
            assert!(preview.text.len() <= WINDOW_BYTES);
            assert_eq!(
                preview.omitted_before + preview.text.len() + preview.omitted_after,
                source.len()
            );
        }
        assert!(line(&source, source.find("needle"))
            .text
            .ends_with("needle"));
    }
}
