//! Restrained colour and glyph choices that degrade with terminal capability.

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub color: bool,
    pub ascii: bool,
}

impl Theme {
    pub fn detect() -> Self {
        let no_color = std::env::var_os("NO_COLOR").is_some();
        let ascii = ascii_requested();
        Self {
            color: !no_color,
            ascii,
        }
    }

    pub fn user(self) -> Style {
        self.fg(Color::Cyan).add_modifier(Modifier::BOLD)
    }

    pub fn assistant(self) -> Style {
        self.fg(Color::Green).add_modifier(Modifier::BOLD)
    }

    pub fn thinking(self) -> Style {
        self.fg(Color::Gray).add_modifier(Modifier::ITALIC)
    }

    pub fn tool_running(self) -> Style {
        self.fg(Color::Yellow)
    }

    pub fn tool_ok(self) -> Style {
        self.fg(Color::Green)
    }

    pub fn tool_fail(self) -> Style {
        self.fg(Color::Red)
    }

    pub fn warning(self) -> Style {
        self.fg(Color::Yellow)
    }

    pub fn error(self) -> Style {
        self.fg(Color::Red).add_modifier(Modifier::BOLD)
    }

    pub fn muted(self) -> Style {
        self.fg(Color::Gray)
    }

    pub fn heading(self) -> Style {
        Style::default().add_modifier(Modifier::BOLD)
    }

    pub fn inline_code(self) -> Style {
        self.fg(Color::Magenta)
    }

    pub fn code_block(self) -> Style {
        self.fg(Color::White)
    }

    pub fn link(self) -> Style {
        self.fg(Color::Blue).add_modifier(Modifier::UNDERLINED)
    }

    pub fn status(self) -> Style {
        self.fg(Color::Yellow)
    }

    pub fn spinner(self, tick: usize) -> &'static str {
        if self.ascii {
            ASCII_SPINNER[tick % ASCII_SPINNER.len()]
        } else {
            UNICODE_SPINNER[tick % UNICODE_SPINNER.len()]
        }
    }

    pub fn arrow(self) -> &'static str {
        if self.ascii {
            "->"
        } else {
            "→"
        }
    }

    fn fg(self, color: Color) -> Style {
        if self.color {
            Style::default().fg(color)
        } else {
            Style::default()
        }
    }
}

const UNICODE_SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const ASCII_SPINNER: [&str; 4] = ["-", "\\", "|", "/"];

pub fn ascii_requested() -> bool {
    if std::env::var_os("GREPPY_ASCII").is_some() {
        return true;
    }
    matches!(std::env::var("TERM").ok().as_deref(), Some("dumb" | "DUMB"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_color_drops_foreground() {
        let theme = Theme {
            color: false,
            ascii: true,
        };
        assert_eq!(theme.user().fg, None);
        assert_eq!(theme.spinner(0), "-");
        assert_eq!(theme.arrow(), "->");
    }

    #[test]
    fn color_theme_keeps_identity_styles() {
        let theme = Theme {
            color: true,
            ascii: false,
        };
        assert_eq!(theme.user().fg, Some(Color::Cyan));
        assert_eq!(theme.spinner(0), "⠋");
    }
}
