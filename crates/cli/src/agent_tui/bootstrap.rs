//! Immediate full-screen loader shown before the session state exists.
//!
//! The terminal chrome appears before workspace allocation. Expensive graph
//! and embedding work is started later and reported inside the session TUI.
//! This bridge keeps startup visually continuous without returning to
//! scrollback between bootstrap and session state construction.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, size, EnterAlternateScreen, LeaveAlternateScreen,
};

const INPUT_POLL: Duration = Duration::from_millis(50);
pub struct BootstrapScreen {
    label: Arc<Mutex<String>>,
    input: Arc<Mutex<String>>,
    queued: Arc<Mutex<Vec<String>>>,
    generation: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    transferred: bool,
}

pub struct BootstrapHandoff {
    pub queued: Vec<String>,
    pub draft: String,
}

impl BootstrapScreen {
    pub fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen, Hide) {
            let _ = disable_raw_mode();
            return Err(error);
        }

        let label = Arc::new(Mutex::new("Creating agent workspace".to_string()));
        let input = Arc::new(Mutex::new(String::new()));
        let queued = Arc::new(Mutex::new(Vec::new()));
        let generation = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::new(AtomicBool::new(false));
        if let Err(error) = draw_frame(&mut stdout, "Creating agent workspace", "", 0, false, true)
        {
            restore_terminal();
            return Err(error);
        }

        let worker_label = Arc::clone(&label);
        let worker_input = Arc::clone(&input);
        let worker_queued = Arc::clone(&queued);
        let worker_generation = Arc::clone(&generation);
        let worker_stop = Arc::clone(&stop);
        let worker_cancelled = Arc::clone(&cancelled);
        let worker = match thread::Builder::new()
            .name("greppy-agent-bootstrap-ui".into())
            .spawn(move || {
                run_event_loop(
                    worker_label,
                    worker_input,
                    worker_queued,
                    worker_generation,
                    worker_stop,
                    worker_cancelled,
                )
            }) {
            Ok(worker) => worker,
            Err(error) => {
                restore_terminal();
                return Err(error);
            }
        };

        Ok(Self {
            label,
            input,
            queued,
            generation,
            stop,
            cancelled,
            worker: Some(worker),
            transferred: false,
        })
    }

    pub fn advance(&mut self, _phase: usize, label: &str) {
        if self.cancelled() {
            return;
        }
        *self
            .label
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = label.to_string();
        self.generation.fetch_add(1, Ordering::Release);
    }

    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Keep raw mode and the alternate screen active. The session TUI adopts
    /// the same terminal immediately afterwards, avoiding a scrollback flash.
    pub fn handoff(mut self) -> BootstrapHandoff {
        self.stop_worker();
        self.transferred = true;
        BootstrapHandoff {
            queued: std::mem::take(
                &mut *self
                    .queued
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            ),
            draft: std::mem::take(
                &mut *self
                    .input
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            ),
        }
    }

    fn stop_worker(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for BootstrapScreen {
    fn drop(&mut self) {
        self.stop_worker();
        if !self.transferred {
            let label = self
                .label
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            restore_terminal();
            eprintln!("greppy agent: startup stopped during {label}");
        }
    }
}

fn run_event_loop(
    label: Arc<Mutex<String>>,
    input: Arc<Mutex<String>>,
    queued: Arc<Mutex<Vec<String>>>,
    generation: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
) {
    // `enter` already painted this size. Do not immediately clear and paint
    // the identical frame again when the input worker starts.
    let mut last_size = Some(size().unwrap_or((80, 24)));
    let mut rendered_generation = generation.load(Ordering::Acquire);
    let mut last_pulse = Instant::now();
    let mut pulse_on = false;
    let mut stdout = io::stdout();
    while !stop.load(Ordering::Relaxed) {
        if event::poll(INPUT_POLL).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(key)) if key.kind != KeyEventKind::Release => match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        cancelled.store(true, Ordering::Relaxed);
                        *label
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            "Cancelling startup".into();
                        generation.fetch_add(1, Ordering::Release);
                    }
                    KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        input
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(ch);
                        generation.fetch_add(1, Ordering::Release);
                    }
                    KeyCode::Backspace => {
                        input
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .pop();
                        generation.fetch_add(1, Ordering::Release);
                    }
                    KeyCode::Enter => {
                        let mut draft = input
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if !draft.trim().is_empty() {
                            queued
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .push(std::mem::take(&mut *draft));
                        }
                        generation.fetch_add(1, Ordering::Release);
                    }
                    _ => {}
                },
                Ok(Event::Paste(text)) => {
                    input
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push_str(&text);
                    generation.fetch_add(1, Ordering::Release);
                }
                _ => {}
            }
        }
        let current_size = size().unwrap_or((80, 24));
        let resized = last_size.replace(current_size) != Some(current_size);
        let current_generation = generation.load(Ordering::Acquire);
        let queued_count = queued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        let pulse = queued_count > 0 && last_pulse.elapsed() >= Duration::from_millis(500);
        if pulse {
            pulse_on = !pulse_on;
            last_pulse = Instant::now();
        }
        if resized || current_generation != rendered_generation || pulse {
            let current_label = label
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            let current_input = input
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            let _ = draw_frame(
                &mut stdout,
                &current_label,
                &current_input,
                queued_count,
                pulse_on,
                resized,
            );
            rendered_generation = current_generation;
        }
    }
}

fn draw_frame(
    stdout: &mut io::Stdout,
    label: &str,
    input: &str,
    queued_count: usize,
    pulse_on: bool,
    clear: bool,
) -> io::Result<()> {
    let (cols, rows) = size().unwrap_or((80, 24));
    let clear_sequence = if clear { "\x1b[2J" } else { "" };
    let ascii = std::env::var_os("GREPPY_ASCII").is_some();
    let (top_left, top_right, bottom_left, bottom_right, horizontal, vertical) = if ascii {
        ('+', '+', '+', '+', '-', '|')
    } else {
        ('┌', '┐', '└', '┘', '─', '│')
    };
    let inner = usize::from(cols.saturating_sub(2));
    let title = " prompt ";
    let top = format!(
        "{top_left}{title}{}{top_right}",
        horizontal
            .to_string()
            .repeat(inner.saturating_sub(title.chars().count()))
    );
    let shown_input = fit_line(input, inner);
    let middle = format!(
        "{vertical}{shown_input}{}{vertical}",
        " ".repeat(inner.saturating_sub(shown_input.chars().count()))
    );
    let bottom = format!(
        "{bottom_left}{}{bottom_right}",
        horizontal.to_string().repeat(inner)
    );
    let composer_top = rows.saturating_sub(3).max(2);
    let composer_middle = composer_top.saturating_add(1).min(rows);
    let composer_bottom = composer_top.saturating_add(2).min(rows);
    let status = fit_line(
        if queued_count > 0 {
            "Queued — starts after this repository's one-time code analysis completes"
        } else {
            label
        },
        usize::from(cols.saturating_sub(4)),
    );
    let status_style = if queued_count == 0 {
        "\x1b[2m"
    } else if pulse_on {
        "\x1b[1;38;5;214m"
    } else {
        "\x1b[2;38;5;214m"
    };
    write!(
        stdout,
        "{clear_sequence}\x1b[1;1H\x1b[2Kgreppy agent {}\
         \x1b[{composer_top};1H\x1b[2K\x1b[2m{top}\x1b[0m\
         \x1b[{composer_middle};1H\x1b[2K\x1b[2m{middle}\x1b[0m\
         \x1b[{composer_bottom};1H\x1b[2K\x1b[2m{bottom}\x1b[0m\
         \x1b[{rows};1H\x1b[2K\x1b[38;5;214m·\x1b[0m {status_style}{status}\x1b[0m",
        env!("CARGO_PKG_VERSION")
    )?;
    stdout.flush()
}

fn fit_line(text: &str, max_width: usize) -> String {
    if text.chars().count() <= max_width {
        return text.to_string();
    }
    if max_width <= 3 {
        return ".".repeat(max_width);
    }
    let mut fitted: String = text.chars().take(max_width - 3).collect();
    fitted.push_str("...");
    fitted
}

fn restore_terminal() {
    let mut stdout = io::stdout();
    let _ = execute!(stdout, Show, LeaveAlternateScreen);
    let _ = disable_raw_mode();
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::fit_line;

    #[test]
    fn labels_fit_without_overwriting_the_frame() {
        assert_eq!(fit_line("abcdefgh", 8), "abcdefgh");
        assert_eq!(fit_line("abcdefgh", 6), "abc...");
        assert_eq!(fit_line("abcdefgh", 2), "..");
    }
}
