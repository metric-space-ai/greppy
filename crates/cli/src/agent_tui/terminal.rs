//! RAII terminal lifecycle: raw mode, alt screen, mouse, paste, cursor, title.

use std::io::{self, Stdout, Write};
use std::panic::{self, PanicHookInfo};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, SetTitle,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use super::theme::ascii_requested;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LifecycleState {
    armed: bool,
    raw_mode: bool,
    alternate_screen: bool,
    bracketed_paste: bool,
    cursor_hidden: bool,
    title_changed: bool,
    mouse_capture: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CleanupAction {
    DisableRawMode,
    DisableMouseCapture,
    DisableBracketedPaste,
    LeaveAlternateScreen,
    ShowCursor,
    ResetTitle,
}

static ACTIVE_TERMINAL: std::sync::Mutex<LifecycleState> = std::sync::Mutex::new(LifecycleState {
    armed: false,
    raw_mode: false,
    alternate_screen: false,
    bracketed_paste: false,
    cursor_hidden: false,
    title_changed: false,
    mouse_capture: false,
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCaps {
    pub color: bool,
    pub ascii: bool,
    pub mouse: bool,
}

impl TerminalCaps {
    pub fn detect() -> Self {
        Self {
            color: std::env::var_os("NO_COLOR").is_none(),
            ascii: ascii_requested(),
            mouse: std::env::var_os("GREPPY_TUI_NO_MOUSE").is_none(),
        }
    }
}

pub fn tty_suitable() -> bool {
    use std::io::IsTerminal;
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub fn unsupported_tty_message() -> &'static str {
    "greppy agent needs a TTY on stdin and stdout for the interactive UI; use `greppy -p` for one-shot mode"
}

pub struct TerminalGuard;

impl TerminalGuard {
    pub fn enter(caps: &TerminalCaps) -> io::Result<(Terminal<CrosstermBackend<Stdout>>, Self)> {
        install_panic_hook();

        // Arm cleanup before the first terminal mutation. Each capability is
        // recorded immediately before its enabling operation, because a
        // failed terminal write may still have emitted a partial sequence.
        let guard = Self::arm()?;
        guard.mark(|state| state.raw_mode = true);
        enable_raw_mode()?;

        let mut stdout = io::stdout();

        guard.mark(|state| state.alternate_screen = true);
        execute!(stdout, EnterAlternateScreen)?;

        guard.mark(|state| state.bracketed_paste = true);
        execute!(stdout, EnableBracketedPaste)?;

        guard.mark(|state| state.cursor_hidden = true);
        execute!(stdout, Hide)?;

        guard.mark(|state| state.title_changed = true);
        execute!(stdout, SetTitle("greppy agent"))?;

        if caps.mouse {
            guard.mark(|state| state.mouse_capture = true);
            execute!(stdout, EnableMouseCapture)?;
        }
        stdout.flush()?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok((terminal, guard))
    }

    fn arm() -> io::Result<Self> {
        let mut state = active_state();
        if state.armed {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "greppy agent already owns the terminal",
            ));
        }
        *state = LifecycleState {
            armed: true,
            ..LifecycleState::default()
        };
        Ok(Self)
    }

    fn mark(&self, update: impl FnOnce(&mut LifecycleState)) {
        let mut state = active_state();
        if state.armed {
            update(&mut state);
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_active();
    }
}

fn active_state() -> std::sync::MutexGuard<'static, LifecycleState> {
    ACTIVE_TERMINAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn take_active_state() -> LifecycleState {
    let mut active = active_state();
    std::mem::take(&mut *active)
}

fn restore_active() {
    restore(take_active_state());
}

fn restore(state: LifecycleState) {
    if !state.armed {
        return;
    }

    let mut stdout = io::stdout();
    for action in cleanup_plan(state) {
        match action {
            CleanupAction::DisableRawMode => {
                let _ = disable_raw_mode();
            }
            CleanupAction::DisableMouseCapture => {
                let _ = execute!(stdout, DisableMouseCapture);
            }
            CleanupAction::DisableBracketedPaste => {
                let _ = execute!(stdout, DisableBracketedPaste);
            }
            CleanupAction::LeaveAlternateScreen => {
                let _ = execute!(stdout, LeaveAlternateScreen);
            }
            CleanupAction::ShowCursor => {
                let _ = execute!(stdout, Show);
            }
            CleanupAction::ResetTitle => {
                let _ = execute!(stdout, SetTitle(""));
            }
        }
    }
    let _ = stdout.flush();
}

fn cleanup_plan(state: LifecycleState) -> Vec<CleanupAction> {
    let mut plan = Vec::with_capacity(6);
    // As in Pi's suspend path, return termios to cooked mode first so the
    // shell regains echo and signal handling even if later escape writes fail.
    if state.raw_mode {
        plan.push(CleanupAction::DisableRawMode);
    }
    if state.mouse_capture {
        plan.push(CleanupAction::DisableMouseCapture);
    }
    if state.bracketed_paste {
        plan.push(CleanupAction::DisableBracketedPaste);
    }
    if state.alternate_screen {
        plan.push(CleanupAction::LeaveAlternateScreen);
    }
    if state.cursor_hidden {
        plan.push(CleanupAction::ShowCursor);
    }
    if state.title_changed {
        plan.push(CleanupAction::ResetTitle);
    }
    plan
}

fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info: &PanicHookInfo<'_>| {
            restore_active();
            previous(info);
        }));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_message_names_headless_escape() {
        assert!(unsupported_tty_message().contains("greppy -p"));
    }

    #[test]
    fn partial_setup_tracks_only_requested_cleanup() {
        let state = LifecycleState {
            armed: true,
            raw_mode: true,
            alternate_screen: true,
            ..LifecycleState::default()
        };
        assert_eq!(
            cleanup_plan(state),
            vec![
                CleanupAction::DisableRawMode,
                CleanupAction::LeaveAlternateScreen,
            ]
        );
    }

    #[test]
    fn full_cleanup_returns_cooked_mode_before_terminal_sequences() {
        let state = LifecycleState {
            armed: true,
            raw_mode: true,
            alternate_screen: true,
            bracketed_paste: true,
            cursor_hidden: true,
            title_changed: true,
            mouse_capture: true,
        };
        assert_eq!(
            cleanup_plan(state),
            vec![
                CleanupAction::DisableRawMode,
                CleanupAction::DisableMouseCapture,
                CleanupAction::DisableBracketedPaste,
                CleanupAction::LeaveAlternateScreen,
                CleanupAction::ShowCursor,
                CleanupAction::ResetTitle,
            ]
        );
    }

    #[test]
    fn taking_lifecycle_state_makes_cleanup_idempotent() {
        let mut state = LifecycleState {
            armed: true,
            raw_mode: true,
            alternate_screen: true,
            bracketed_paste: true,
            cursor_hidden: true,
            title_changed: true,
            mouse_capture: true,
        };
        let first = std::mem::take(&mut state);
        let second = std::mem::take(&mut state);
        assert!(first.armed);
        assert_eq!(second, LifecycleState::default());
    }
}
