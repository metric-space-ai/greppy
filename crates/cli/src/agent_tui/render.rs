//! Layout and painting for the interactive agent TUI.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

use super::markdown::render_markdown;
use super::overlay::Overlay;
use super::state::{App, RunPhase, ToolPhase, TranscriptItem, MIN_COLS, MIN_ROWS};
use super::theme::Theme;

pub fn render(frame: &mut Frame<'_>, app: &mut App) {
    app.cols = frame.area().width;
    app.rows = frame.area().height;
    if app.cols < MIN_COLS || app.rows < MIN_ROWS {
        render_too_small(frame, app);
        return;
    }

    let composer_h = composer_height(app);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(composer_h),
            Constraint::Length(1),
        ])
        .split(frame.area());

    render_header(frame, chunks[0], app);
    render_transcript(frame, chunks[1], app);
    if composer_h > 0 {
        render_composer(frame, chunks[2], app);
    }
    render_footer(frame, chunks[3], app);
    render_overlay(frame, frame.area(), app);
}

fn composer_height(app: &App) -> u16 {
    let lines = app.composer.line_count() as u16;
    let cap = if app.rows < 24 { 4 } else { 8 };
    lines
        .saturating_add(2)
        .clamp(3, cap.min(app.rows.saturating_sub(8).max(3)))
}

fn render_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let line = header_line(app, area.width);
    frame.render_widget(Paragraph::new(Line::from(Span::raw(line))), area);
}

pub fn header_line(app: &App, width: u16) -> String {
    let mut fields = vec![
        format!("greppy agent {}", env!("CARGO_PKG_VERSION")),
        app.header.repository.clone(),
        app.header.branch.clone(),
        app.header.worktree.clone(),
        app.header.model.clone(),
        app.header.endpoint.clone(),
        app.header.sandbox.clone(),
    ];
    loop {
        let line = fields.join("  ·  ");
        if line.width() <= width as usize || fields.len() <= 2 {
            if line.width() <= width as usize {
                return line;
            }
            return truncate_width(&line, width as usize);
        }
        // Drop secondary fields from the right, keeping identity first.
        if fields.len() > 2 {
            fields.remove(fields.len() - 1);
        }
    }
}

fn render_transcript(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let block = Block::default().borders(Borders::NONE);
    let inner = block.inner(area);
    app.viewport_height = inner.height;
    let text = transcript_text(app, inner.width);
    let visual = visual_line_count(&text, inner.width);
    app.max_scroll = visual
        .saturating_sub(inner.height as usize)
        .min(u16::MAX as usize) as u16;
    if app.follow_tail {
        app.scroll = app.max_scroll;
    } else {
        app.scroll = app.scroll.min(app.max_scroll);
    }
    let paragraph = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll, 0));
    frame.render_widget(paragraph, area);
}

pub fn transcript_text(app: &App, width: u16) -> Text<'static> {
    if matches!(
        app.phase,
        RunPhase::Setup | RunPhase::Configuring | RunPhase::Blocked
    ) || (app.phase == RunPhase::Cancelling && app.items.is_empty())
    {
        return setup_text(app, width);
    }
    let mut lines = Vec::new();
    for (index, item) in app.items.iter().enumerate() {
        if index > 0 {
            lines.push(Line::default());
        }
        lines.extend(item_lines(item, app.theme, app.thinking_expanded, width));
    }
    Text::from(lines)
}

fn setup_text(app: &App, width: u16) -> Text<'static> {
    let ratio = if app.setup_total == 0 {
        0.0
    } else {
        app.setup_completed.min(app.setup_total) as f64 / app.setup_total as f64
    };
    let bar_width = usize::from(width.saturating_sub(18).clamp(12, 56));
    let filled = (ratio * bar_width as f64).round() as usize;
    let ascii = std::env::var_os("GREPPY_ASCII").is_some();
    let done = if ascii { '=' } else { '━' };
    let pending = if ascii { '-' } else { '─' };
    let bar = (app.setup_total > 0).then(|| {
        Line::from(vec![
            Span::styled(
                done.to_string().repeat(filled.min(bar_width)),
                app.theme.tool_ok(),
            ),
            Span::styled(
                pending.to_string().repeat(bar_width.saturating_sub(filled)),
                Style::default().add_modifier(Modifier::DIM),
            ),
        ])
    });
    let progress = if app.setup_total > 0 {
        format!(
            "{} / {} {}  ·  {:>3}%",
            app.setup_completed,
            app.setup_total,
            app.setup_unit,
            (ratio * 100.0).round() as u64
        )
    } else {
        String::new()
    };
    let rate = app
        .setup_rate_milli_per_second
        .map(|value| format!("{:.1}/s", value as f64 / 1000.0))
        .unwrap_or_default();
    let eta = app
        .setup_eta_seconds
        .map(format_duration)
        .map(|value| format!("ETA {value}"))
        .unwrap_or_default();
    let content_lines: u16 =
        if app.setup_total > 0 { 4 } else { 1 } + u16::from(app.setup_detail.is_some());
    let top_padding = usize::from(
        app.rows
            .saturating_sub(content_lines + composer_height(app) + 2)
            / 2,
    );
    let mut lines = vec![Line::default(); top_padding];
    if app.phase == RunPhase::Configuring {
        lines.push(Line::from(Span::styled(
            "Configure model gateway",
            Style::default().add_modifier(Modifier::BOLD),
        )));
        if let Some(message) = &app.setup_error {
            lines.push(Line::from(message.clone()));
        }
    } else if let Some(error) = &app.setup_error {
        lines.push(Line::from(Span::styled(
            "Startup failed",
            app.theme.error().add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(error.clone()));
    } else {
        if app.setup_total > 0 {
            lines.push(Line::from(Span::styled(
                app.status.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            )));
            lines.push(bar.expect("known setup total has a progress bar"));
            lines.push(Line::from(progress));
            let mut telemetry = Vec::new();
            if !rate.is_empty() {
                telemetry.push(rate);
            }
            if !eta.is_empty() {
                telemetry.push(eta);
            }
            telemetry.push(format!(
                "elapsed {}",
                format_duration(app.setup_elapsed_seconds)
            ));
            lines.push(Line::from(telemetry.join("  ·  ")));
        } else {
            let pulse_style = if app.spinner_tick % 2 == 0 {
                app.theme.status().add_modifier(Modifier::BOLD)
            } else {
                app.theme.status().add_modifier(Modifier::DIM)
            };
            lines.push(Line::from(Span::styled(
                app.status.clone(),
                if app.queued.is_empty() {
                    Style::default().add_modifier(Modifier::BOLD)
                } else {
                    pulse_style
                },
            )));
        }
        if let Some(detail) = &app.setup_detail {
            lines.push(Line::from(Span::styled(
                format!("symbol  ·  {detail}"),
                Style::default().add_modifier(Modifier::DIM),
            )));
        }
    }
    Text::from(lines)
}

pub(super) fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    }
}

fn item_lines(
    item: &TranscriptItem,
    theme: Theme,
    thinking_expanded: bool,
    width: u16,
) -> Vec<Line<'static>> {
    match item {
        TranscriptItem::User { text } => labeled("you", theme.user(), text, false, theme),
        TranscriptItem::Assistant { text } => {
            let mut lines = vec![Line::from(Span::styled("greppy", theme.assistant()))];
            lines.extend(render_markdown(text, theme));
            lines
        }
        TranscriptItem::Thinking { text, streaming } => {
            let label = if *streaming {
                "thinking…"
            } else {
                "thinking"
            };
            if thinking_expanded {
                labeled(label, theme.thinking(), text, false, theme)
            } else {
                let summary = if *streaming { "streaming" } else { "collapsed" };
                vec![Line::from(Span::styled(
                    format!("{label} ({summary})"),
                    theme.thinking(),
                ))]
            }
        }
        TranscriptItem::Tool {
            summary,
            phase,
            elapsed_ms,
            preview,
            expanded,
            ..
        } => {
            let (tag, style) = match phase {
                ToolPhase::Running => ("tool …", theme.tool_running()),
                ToolPhase::Success => ("tool ok", theme.tool_ok()),
                ToolPhase::Failure => ("tool !", theme.tool_fail()),
            };
            let arrow = theme.arrow();
            let mut line = format!("{tag}  {arrow} {summary}");
            if *elapsed_ms > 0 {
                line.push_str(&format!("  {elapsed_ms}ms"));
            }
            let mut lines = vec![Line::from(Span::styled(
                truncate_width(&line, width as usize),
                style,
            ))];
            if *expanded || matches!(phase, ToolPhase::Failure) {
                for preview_line in preview.lines().take(8) {
                    lines.push(Line::from(Span::styled(
                        format!("    {preview_line}"),
                        theme.muted(),
                    )));
                }
            } else if !preview.is_empty() {
                if let Some(first) = preview.lines().next() {
                    lines.push(Line::from(Span::styled(
                        format!(
                            "    {}",
                            truncate_width(first, width.saturating_sub(4) as usize)
                        ),
                        theme.muted(),
                    )));
                }
            }
            lines
        }
        TranscriptItem::Warning { text } => labeled("warn", theme.warning(), text, false, theme),
        TranscriptItem::Error { text } => labeled("error", theme.error(), text, true, theme),
        TranscriptItem::Queued { text } => labeled("queued", theme.muted(), text, false, theme),
    }
}

fn labeled(
    label: &str,
    style: Style,
    text: &str,
    failed: bool,
    _theme: Theme,
) -> Vec<Line<'static>> {
    let _ = failed;
    let mut lines = Vec::new();
    let mut body = text.lines();
    let first = body.next().unwrap_or("");
    lines.push(Line::from(vec![
        Span::styled(format!("{label}  "), style),
        Span::raw(first.to_string()),
    ]));
    for line in body {
        lines.push(Line::from(vec![
            Span::raw("      ".to_string()),
            Span::raw(line.to_string()),
        ]));
    }
    lines
}

fn render_composer(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let title = match app.phase {
        RunPhase::Configuring => " Gateway URL · Enter to connect · Ctrl-C to quit ",
        RunPhase::Setup => " prompt · indexing continues in background ",
        RunPhase::Blocked => " startup failed ",
        RunPhase::Busy | RunPhase::Cancelling => " queued follow-ups accepted ",
        RunPhase::Idle => " prompt ",
    };
    let inner_h = area.height.saturating_sub(2).max(1);
    let inner_w = area.width.saturating_sub(2).max(1);
    app.composer.ensure_cursor_visible(inner_h, inner_w);
    let paragraph = Paragraph::new(app.composer.visible_text(inner_h, inner_w))
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);

    if app.phase == RunPhase::Blocked {
        return;
    }

    if let Some(menu) = &app.completion {
        let height = (menu.items.len() as u16)
            .saturating_add(2)
            .min(8)
            .min(area.y);
        if height >= 3 && area.y >= height {
            let popup = Rect {
                x: area.x.saturating_add(1),
                y: area.y.saturating_sub(height),
                width: area.width.saturating_sub(2).max(10),
                height,
            };
            let lines: Vec<Line> = menu
                .items
                .iter()
                .enumerate()
                .map(|(idx, item)| {
                    let style = if idx == menu.selected {
                        Style::default().add_modifier(Modifier::REVERSED)
                    } else {
                        Style::default()
                    };
                    Line::from(Span::styled(item.clone(), style))
                })
                .collect();
            frame.render_widget(Clear, popup);
            frame.render_widget(
                Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" / ")),
                popup,
            );
        }
    }

    let (row, col) = app.composer.visual_cursor(inner_w);
    let row = row.saturating_sub(app.composer.scroll() as u16);
    let x = area
        .x
        .saturating_add(1)
        .saturating_add(col)
        .min(area.right().saturating_sub(2));
    let y = area
        .y
        .saturating_add(1)
        .saturating_add(row)
        .min(area.bottom().saturating_sub(2));
    frame.set_cursor_position((x, y));
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if matches!(
        app.phase,
        RunPhase::Setup | RunPhase::Configuring | RunPhase::Blocked
    ) {
        let line = match app.phase {
            RunPhase::Setup => "Ctrl-C cancels startup",
            RunPhase::Configuring => "Enter connects  ·  Ctrl-C quits",
            RunPhase::Blocked => "Startup blocked  ·  Ctrl-C quits",
            _ => unreachable!(),
        };
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
    let activity = match app.phase {
        RunPhase::Busy | RunPhase::Cancelling => {
            format!("{} {}", app.theme.spinner(app.spinner_tick), app.status)
        }
        RunPhase::Idle => app.status.clone(),
        RunPhase::Setup | RunPhase::Configuring | RunPhase::Blocked => unreachable!(),
    };
    let extra = app.copy_status.as_deref().unwrap_or("");
    let extra_sep = if extra.is_empty() { "" } else { "  ·  " };
    let background = app
        .background_status
        .as_deref()
        .map(|status| format!("  ·  {status}"))
        .unwrap_or_default();
    let line = format!(
        "{activity}{background}  ·  {model}  ·  in {inn} out {out} cache {cr}/{cw}  ·  {turns} turns  ·  {queued} queued{extra_sep}{extra}",
        model = app.header.model,
        inn = app.input_tokens,
        out = app.output_tokens,
        cr = app.cache_read,
        cw = app.cache_write,
        turns = app.turns,
        queued = app.queued.len(),
    );
    frame.render_widget(
        Paragraph::new(truncate_width(&line, area.width as usize)).style(app.theme.status()),
        area,
    );
}

fn render_overlay(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let overlay_lines = match &app.overlay {
        Overlay::None => return,
        Overlay::TooSmall { cols, rows } => {
            render_too_small(frame, app);
            let _ = (cols, rows);
            return;
        }
        Overlay::Help => Overlay::help_lines(),
        Overlay::ConfirmClear => vec![
            "Clear the visible transcript?".into(),
            "This drops unsaved conversational context from the display.".into(),
            String::new(),
            "y confirm   n cancel".into(),
        ],
        Overlay::Usage => app.usage_lines(),
        Overlay::Model(picker) | Overlay::Sessions(picker) => {
            let mut lines = vec![
                picker.title.clone(),
                format!("filter: {}", picker.filter),
                String::new(),
            ];
            for (idx, item) in picker.visible().into_iter().enumerate() {
                let marker = if idx == picker.selected { ">" } else { " " };
                lines.push(format!("{marker} {}  {}", item.label, item.detail));
            }
            if picker.visible().is_empty() {
                lines.push("(no matches)".into());
            }
            lines
        }
        Overlay::Tools(tools) => {
            let mut lines = vec!["tool executions".into(), String::new()];
            let mut idx = 0usize;
            for item in &app.items {
                if let TranscriptItem::Tool {
                    summary,
                    phase,
                    elapsed_ms,
                    expanded,
                    ..
                } = item
                {
                    let marker = if idx == tools.selected { ">" } else { " " };
                    let state = match phase {
                        ToolPhase::Running => "running",
                        ToolPhase::Success => "ok",
                        ToolPhase::Failure => "failed",
                    };
                    let flag = if *expanded { "+" } else { "-" };
                    lines.push(format!(
                        "{marker} {flag} {state}  {summary}  {elapsed_ms}ms"
                    ));
                    idx += 1;
                }
            }
            if idx == 0 {
                lines.push("no tools yet".into());
            }
            lines
        }
        Overlay::Setup(menu) => {
            let rows = [
                ("Gateway", app.header.endpoint.clone(), "edit now"),
                ("Model", app.header.model.clone(), "select now"),
                ("Language", app.settings.language.clone(), "active"),
                (
                    "Session store",
                    if app.settings.private_store {
                        "private".into()
                    } else {
                        "automatic".into()
                    },
                    "toggle · next launch",
                ),
                (
                    "Sandbox",
                    if app.settings.no_sandbox {
                        "off".into()
                    } else {
                        "enabled".into()
                    },
                    "toggle · next launch",
                ),
                (
                    "Startup self-check",
                    if app.settings.skip_selfcheck {
                        "skipped".into()
                    } else {
                        "enabled".into()
                    },
                    "toggle · next launch",
                ),
                (
                    "Acceleration",
                    app.settings.acceleration.clone(),
                    "cycle · next launch",
                ),
                (
                    "Workspace backend",
                    app.settings.workspace_backend.clone(),
                    "cycle · next launch",
                ),
                ("Close", "Enter / Esc".into(), ""),
            ];
            let mut lines = vec![
                "Setup".into(),
                "All interactive agent settings".into(),
                String::new(),
            ];
            for (index, (label, value, detail)) in rows.into_iter().enumerate() {
                let marker = if index == menu.selected { ">" } else { " " };
                lines.push(format!("{marker} {label:<18} {value:<14} {detail}"));
            }
            lines.push(String::new());
            lines.push("↑/↓ select  ·  Enter change  ·  Esc close".into());
            lines
        }
    };
    let width = area.width.saturating_mul(3) / 4 + area.width / 8;
    let height = (overlay_lines.len() as u16)
        .saturating_add(2)
        .min(area.height.saturating_sub(2));
    let popup = centered(area, width.min(area.width.saturating_sub(2)), height);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(
            overlay_lines
                .into_iter()
                .map(Line::from)
                .collect::<Vec<_>>(),
        )
        .block(Block::default().borders(Borders::ALL).title(" greppy "))
        .wrap(Wrap { trim: false }),
        popup,
    );
}

fn render_too_small(frame: &mut Frame<'_>, app: &App) {
    let msg = format!(
        "terminal too small ({}x{}) — need at least {MIN_COLS}x{MIN_ROWS}",
        app.cols, app.rows
    );
    frame.render_widget(
        Paragraph::new(msg).alignment(ratatui::layout::Alignment::Center),
        frame.area(),
    );
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn visual_line_count(text: &Text<'_>, width: u16) -> usize {
    let width = width.max(1) as usize;
    text.lines
        .iter()
        .map(|line| {
            let used: usize = line.spans.iter().map(|span| span.content.width()).sum();
            used.max(1).div_ceil(width)
        })
        .sum()
}

fn truncate_width(input: &str, max: usize) -> String {
    if input.width() <= max {
        return input.to_string();
    }
    let mut out = String::new();
    let mut used = 0usize;
    for grapheme in unicode_segmentation::UnicodeSegmentation::graphemes(input, true) {
        let w = grapheme.width();
        if used + w + 1 > max {
            break;
        }
        out.push_str(grapheme);
        used += w;
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_tui::session::SessionRecord;
    use crate::agent_tui::state::HeaderState;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn sample_app() -> App {
        let session =
            SessionRecord::new("sess".into(), "demo".into(), "model-x".into(), "run".into());
        let mut app = App::new(
            HeaderState {
                repository: "greppy".into(),
                branch: "main".into(),
                worktree: "worktree".into(),
                model: "model-x".into(),
                endpoint: "http://127.0.0.1:8317".into(),
                sandbox: "sandbox off".into(),
            },
            Theme {
                color: false,
                ascii: true,
            },
            &session,
        );
        app.cols = 80;
        app.rows = 24;
        app
    }

    fn paint(app: &mut App, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, app)).unwrap();
        buffer_string(terminal.backend())
    }

    pub fn buffer_string(backend: &TestBackend) -> String {
        let buf = backend.buffer();
        let area = buf.area();
        let mut out = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                let cell = &buf[(x, y)];
                out.push_str(cell.symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn header_drops_secondary_fields_before_identity() {
        let app = sample_app();
        let wide = header_line(&app, 120);
        assert!(wide.contains("greppy"));
        assert!(wide.contains("model-x"));
        let narrow = header_line(&app, 28);
        assert!(narrow.contains("greppy"));
        assert!(!narrow.contains("sandbox"));
        assert!(narrow.width() <= 28);
    }

    #[test]
    fn idle_snapshot_fits_supported_sizes() {
        let mut app = sample_app();
        app.push_user("inspect the parser".into());
        app.append_assistant("I'll look at `parse_config`.");
        for (w, h) in [(120, 36), (80, 24), (60, 18)] {
            let out = paint(&mut app, w, h);
            assert!(!out.contains("terminal too small"), "{w}x{h}\n{out}");
            assert!(out.contains("greppy"), "{w}x{h}\n{out}");
        }
    }

    #[test]
    fn too_small_view_is_stable() {
        let mut app = sample_app();
        let out = paint(&mut app, 40, 10);
        assert!(out.contains("terminal too small"), "{out}");
    }
}
