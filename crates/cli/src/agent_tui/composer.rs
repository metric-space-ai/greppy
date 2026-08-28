//! Multiline Unicode-safe prompt editor.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composer {
    text: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    draft: String,
    scroll: usize,
}

impl Default for Composer {
    fn default() -> Self {
        Self::new()
    }
}

impl Composer {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
            draft: String::new(),
            scroll: 0,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    #[allow(dead_code)]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.cursor = self.text.len();
        self.history_index = None;
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.scroll = 0;
        self.history_index = None;
    }

    pub fn insert_text(&mut self, text: &str) {
        let filtered: String = text.chars().filter(|ch| *ch != '\u{0}').collect();
        self.text.insert_str(self.cursor, &filtered);
        self.cursor += filtered.len();
        self.history_index = None;
    }

    pub fn insert_char(&mut self, ch: char) {
        if ch != '\u{0}' {
            self.text.insert(self.cursor, ch);
            self.cursor += ch.len_utf8();
            self.history_index = None;
        }
    }

    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    pub fn backspace(&mut self) {
        if let Some(prev) = prev_grapheme(self.text.as_str(), self.cursor) {
            self.text.drain(prev..self.cursor);
            self.cursor = prev;
        }
    }

    pub fn delete(&mut self) {
        if let Some(next) = next_grapheme(self.text.as_str(), self.cursor) {
            self.text.drain(self.cursor..next);
        }
    }

    pub fn move_left(&mut self) {
        if let Some(prev) = prev_grapheme(self.text.as_str(), self.cursor) {
            self.cursor = prev;
        }
    }

    pub fn move_right(&mut self) {
        if let Some(next) = next_grapheme(self.text.as_str(), self.cursor) {
            self.cursor = next;
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = current_line_start(&self.text, self.cursor);
    }

    pub fn move_end(&mut self) {
        self.cursor = current_line_end(&self.text, self.cursor);
    }

    pub fn move_word_left(&mut self) {
        let prefix = &self.text[..self.cursor];
        let trimmed = prefix.trim_end();
        if trimmed.len() < prefix.len() {
            self.cursor = trimmed.len();
        }
        let prefix = &self.text[..self.cursor];
        if let Some(idx) = prefix.rfind(|ch: char| ch.is_whitespace()) {
            self.cursor = idx + 1;
            while self.cursor < prefix.len() && prefix.as_bytes().get(self.cursor) == Some(&b' ') {
                self.cursor += 1;
            }
            if let Some(last_ws) = prefix.rfind(char::is_whitespace) {
                self.cursor = last_ws + 1;
            } else {
                self.cursor = 0;
            }
        } else {
            self.cursor = 0;
        }
        self.cursor = snap_to_grapheme(&self.text, self.cursor);
    }

    pub fn move_word_right(&mut self) {
        let rest = &self.text[self.cursor..];
        let bytes = rest.as_bytes();
        let mut i = 0;
        while i < rest.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        while i < rest.len() && !bytes[i].is_ascii_whitespace() {
            i += rest[i..]
                .chars()
                .next()
                .map(|ch| ch.len_utf8())
                .unwrap_or(1);
        }
        self.cursor += i;
        self.cursor = snap_to_grapheme(&self.text, self.cursor);
    }

    pub fn move_line_up(&mut self) {
        let start = current_line_start(&self.text, self.cursor);
        if start == 0 {
            return;
        }
        let col = display_col(&self.text[start..self.cursor]);
        let prev_end = start.saturating_sub(1);
        let prev_start = current_line_start(&self.text, prev_end);
        self.cursor = offset_for_col(&self.text[prev_start..prev_end], col) + prev_start;
    }

    pub fn move_line_down(&mut self) {
        let end = current_line_end(&self.text, self.cursor);
        if end == self.text.len() {
            return;
        }
        let start = current_line_start(&self.text, self.cursor);
        let col = display_col(&self.text[start..self.cursor]);
        let next_start = end + 1;
        let next_end = current_line_end(&self.text, next_start);
        self.cursor = offset_for_col(&self.text[next_start..next_end], col) + next_start;
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.history_index {
            None => {
                self.draft = self.text.clone();
                self.history_index = Some(self.history.len() - 1);
            }
            Some(0) => return,
            Some(idx) => self.history_index = Some(idx - 1),
        }
        if let Some(idx) = self.history_index {
            self.text = self.history[idx].clone();
            self.cursor = self.text.len();
        }
    }

    pub fn history_down(&mut self) {
        let Some(idx) = self.history_index else {
            return;
        };
        if idx + 1 >= self.history.len() {
            self.history_index = None;
            self.text = self.draft.clone();
            self.cursor = self.text.len();
            return;
        }
        self.history_index = Some(idx + 1);
        self.text = self.history[idx + 1].clone();
        self.cursor = self.text.len();
    }

    pub fn remember(&mut self, prompt: &str) {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.history.last().map(String::as_str) != Some(trimmed) {
            self.history.push(trimmed.to_string());
        }
        self.history_index = None;
        self.draft.clear();
    }

    pub fn submit(&mut self) -> Option<String> {
        let prompt = self.text.trim().to_string();
        if prompt.is_empty() {
            return None;
        }
        self.remember(&prompt);
        self.clear();
        Some(prompt)
    }

    pub fn visual_cursor(&self, width: u16) -> (u16, u16) {
        let width = width.max(1) as usize;
        let before = &self.text[..self.cursor];
        let mut row = 0u16;
        let mut col = 0usize;
        for grapheme in before.graphemes(true) {
            if grapheme == "\n" {
                row = row.saturating_add(1);
                col = 0;
                continue;
            }
            let w = grapheme.width().max(1);
            if col + w > width {
                row = row.saturating_add(1);
                col = w;
            } else {
                col += w;
            }
        }
        (row, col.min(u16::MAX as usize) as u16)
    }

    pub fn line_count(&self) -> usize {
        self.text.split('\n').count().max(1)
    }

    pub fn ensure_cursor_visible(&mut self, height: u16, width: u16) {
        let height = height.max(1) as usize;
        let (row, _) = self.visual_cursor(width);
        let row = row as usize;
        if row < self.scroll {
            self.scroll = row;
        } else if row >= self.scroll + height {
            self.scroll = row + 1 - height;
        }
    }

    pub fn scroll(&self) -> usize {
        self.scroll
    }

    pub fn history_index_active(&self) -> bool {
        self.history_index.is_some()
    }

    pub fn visible_text(&self, height: u16, width: u16) -> String {
        self.ensure_cursor_visible_copy(height, width);
        let width = width.max(1) as usize;
        let mut rows: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut col = 0usize;
        for grapheme in self.text.graphemes(true) {
            if grapheme == "\n" {
                rows.push(std::mem::take(&mut current));
                col = 0;
                continue;
            }
            let w = grapheme.width().max(1);
            if col + w > width && !current.is_empty() {
                rows.push(std::mem::take(&mut current));
                col = 0;
            }
            current.push_str(grapheme);
            col += w;
        }
        rows.push(current);
        let start = self.scroll.min(rows.len().saturating_sub(1));
        rows.into_iter()
            .skip(start)
            .take(height.max(1) as usize)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn ensure_cursor_visible_copy(&self, height: u16, width: u16) {
        let _ = (height, width);
    }
}

fn prev_grapheme(text: &str, cursor: usize) -> Option<usize> {
    text[..cursor]
        .graphemes(true)
        .next_back()
        .map(|g| cursor - g.len())
}

fn next_grapheme(text: &str, cursor: usize) -> Option<usize> {
    text[cursor..]
        .graphemes(true)
        .next()
        .map(|g| cursor + g.len())
}

fn snap_to_grapheme(text: &str, mut offset: usize) -> usize {
    if offset > text.len() {
        offset = text.len();
    }
    while offset > 0 && !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn current_line_start(text: &str, cursor: usize) -> usize {
    text[..cursor].rfind('\n').map_or(0, |idx| idx + 1)
}

fn current_line_end(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .find('\n')
        .map_or(text.len(), |idx| cursor + idx)
}

fn display_col(text: &str) -> usize {
    text.graphemes(true).map(UnicodeWidthStr::width).sum()
}

fn offset_for_col(line: &str, col: usize) -> usize {
    let mut offset = 0usize;
    let mut used = 0usize;
    for grapheme in line.graphemes(true) {
        if used >= col {
            return offset;
        }
        used += grapheme.width();
        offset += grapheme.len();
        if used >= col {
            return offset;
        }
    }
    line.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combining_mark_is_one_grapheme() {
        let mut editor = Composer::new();
        editor.insert_text("e\u{0301}x");
        editor.move_left();
        editor.backspace();
        assert_eq!(editor.text(), "x");
        assert_eq!(editor.cursor(), 0);
    }

    #[test]
    fn emoji_zwj_sequence_deletes_as_one_grapheme() {
        let mut editor = Composer::new();
        editor.insert_text("a👨‍👩‍👧b");
        editor.move_left();
        editor.backspace();
        assert_eq!(editor.text(), "ab");
    }

    #[test]
    fn cjk_width_and_boundary_delete() {
        let mut editor = Composer::new();
        editor.insert_text("a日b");
        editor.move_left();
        editor.backspace();
        assert_eq!(editor.text(), "ab");
        editor.insert_text("日");
        assert!(editor.visual_cursor(10).1 >= 1);
    }

    #[test]
    fn multiline_paste_and_line_navigation() {
        let mut editor = Composer::new();
        editor.insert_text("one\ntwo\nthree");
        editor.move_home();
        assert!(editor.text()[editor.cursor()..].starts_with("three"));
        editor.move_line_up();
        editor.move_home();
        assert!(editor.text()[editor.cursor()..].starts_with("two"));
        editor.move_end();
        editor.delete();
        assert_eq!(editor.text(), "one\ntwothree");
    }

    #[test]
    fn history_navigation_when_empty_or_after_submit() {
        let mut editor = Composer::new();
        editor.set_text("first");
        let first = editor.submit().unwrap();
        assert_eq!(first, "first");
        editor.set_text("second");
        let _ = editor.submit();
        editor.history_up();
        assert_eq!(editor.text(), "second");
        editor.history_up();
        assert_eq!(editor.text(), "first");
        editor.history_down();
        assert_eq!(editor.text(), "second");
    }
}
