//! Deterministic PNG export of the production renderer.

#![allow(dead_code)]

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::Terminal;

use super::render::render;
use super::session::SessionRecord;
use super::state::{App, HeaderState, RunPhase, TranscriptItem};
use super::theme::Theme;

const CELL_W: u32 = 8;
const CELL_H: u32 = 16;

pub fn preview_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/assets/tui")
}

pub fn sample_idle_app() -> App {
    let session = SessionRecord::new(
        "sess-preview".into(),
        "greppy".into(),
        "local-model".into(),
        "agent-1".into(),
    );
    let mut app = App::new(
        HeaderState {
            repository: "greppy".into(),
            branch: "main".into(),
            worktree: "worktree".into(),
            model: "local-model".into(),
            endpoint: "http://127.0.0.1:8317".into(),
            sandbox: "sandbox seatbelt".into(),
        },
        Theme {
            color: true,
            ascii: false,
        },
        &session,
    );
    app.push_user("inspect parse_config and add tests".into());
    app.append_assistant(
        "I'll start from the graph.\n\n```rust\nfn parse_config(src: &str) -> Config {}\n```\n\nNext I'll search callers.",
    );
    app.start_tool("t1".into(), "greppy who-calls parse_config".into());
    app.finish_tool(
        "t1",
        false,
        42,
        "crates/core/src/lib.rs:12  load_config".into(),
    );
    app.status = "ready".into();
    app.input_tokens = 1200;
    app.output_tokens = 340;
    app.turns = 2;
    app.cols = 120;
    app.rows = 36;
    app
}

pub fn sample_busy_app() -> App {
    let mut app = sample_idle_app();
    app.phase = RunPhase::Busy;
    app.status = "working".into();
    app.append_thinking("ranking who-calls hits");
    app.append_assistant(" Callers look confined to the loader.");
    app.start_tool("t2".into(), "greppy read parse_config".into());
    app.queued.push_back("then write the tests".into());
    app.items.push(TranscriptItem::Queued {
        text: "then write the tests".into(),
    });
    app.spinner_tick = 0;
    app
}

pub fn write_previews(dir: &Path) -> io::Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(dir)?;
    let wide = dir.join("agent-tui-120x36.png");
    let mid = dir.join("agent-tui-80x24.png");
    let mut idle = sample_idle_app();
    encode_app(&mut idle, 120, 36, &wide)?;
    let mut busy = sample_busy_app();
    encode_app(&mut busy, 80, 24, &mid)?;
    Ok((wide, mid))
}

pub fn encode_app(app: &mut App, width: u16, height: u16, path: &Path) -> io::Result<()> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal.draw(|frame| render(frame, app)).expect("draw");
    encode_png(terminal.backend(), path)
}

fn encode_png(backend: &TestBackend, path: &Path) -> io::Result<()> {
    let buf = backend.buffer();
    let area: Rect = *buf.area();
    let width = u32::from(area.width) * CELL_W;
    let height = u32::from(area.height) * CELL_H;
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..area.height {
        for x in 0..area.width {
            let cell = &buf[(x, y)];
            let fg = ansi_rgb(cell.fg, true);
            let bg = ansi_rgb(cell.bg, false);
            blit_cell(
                &mut rgba,
                width,
                u32::from(x),
                u32::from(y),
                cell.symbol(),
                fg,
                bg,
            );
        }
    }
    write_png(path, width, height, &rgba)
}

fn blit_cell(
    rgba: &mut [u8],
    stride_w: u32,
    cx: u32,
    cy: u32,
    symbol: &str,
    fg: [u8; 3],
    bg: [u8; 3],
) {
    let glyph = glyph_for(symbol);
    for row in 0..CELL_H {
        for col in 0..CELL_W {
            let on = glyph_bit(glyph, col, row);
            let color = if on { fg } else { bg };
            let px = (cy * CELL_H + row) * stride_w + (cx * CELL_W + col);
            let i = (px * 4) as usize;
            rgba[i] = color[0];
            rgba[i + 1] = color[1];
            rgba[i + 2] = color[2];
            rgba[i + 3] = 255;
        }
    }
}

fn glyph_for(symbol: &str) -> u8 {
    symbol.chars().next().unwrap_or(' ') as u8
}

fn glyph_bit(ch: u8, col: u32, row: u32) -> bool {
    if !(32..=126).contains(&ch) {
        return col == 0 || col + 1 == CELL_W || row == 0 || row + 1 == CELL_H;
    }
    if ch == b' ' {
        return false;
    }
    // 5x7 glyph scaled into 8x16 with padding. Distinct per character so
    // screenshots stay deterministic without a bundled bitmap font.
    let gx = col.saturating_sub(1);
    let gy = row.saturating_sub(2) / 2;
    if gx >= 5 || gy >= 7 {
        return false;
    }
    let pattern = u32::from(ch).wrapping_mul(0x9E37_79B9);
    let bit = gy * 5 + gx;
    ((pattern >> (bit % 31)) & 1) == 1 || gx == 0 || gy == 0
}

fn ansi_rgb(color: Color, fg: bool) -> [u8; 3] {
    match color {
        Color::Reset if fg => [220, 220, 220],
        Color::Reset => [18, 18, 18],
        Color::Black => [0, 0, 0],
        Color::Red => [220, 80, 80],
        Color::Green => [80, 180, 110],
        Color::Yellow => [210, 180, 70],
        Color::Blue => [80, 140, 220],
        Color::Magenta => [180, 110, 200],
        Color::Cyan => [70, 190, 200],
        Color::Gray => [150, 150, 150],
        Color::DarkGray => [90, 90, 90],
        Color::LightRed => [255, 120, 120],
        Color::LightGreen => [120, 220, 140],
        Color::LightYellow => [240, 220, 120],
        Color::LightBlue => [120, 170, 240],
        Color::LightMagenta => [210, 140, 230],
        Color::LightCyan => [120, 220, 230],
        Color::White => [240, 240, 240],
        Color::Rgb(r, g, b) => [r, g, b],
        Color::Indexed(n) => indexed(n),
    }
}

fn indexed(n: u8) -> [u8; 3] {
    match n {
        0 => [0, 0, 0],
        1 => [180, 60, 60],
        2 => [60, 160, 80],
        3 => [180, 160, 50],
        4 => [60, 100, 180],
        5 => [160, 80, 170],
        6 => [50, 160, 170],
        7 => [200, 200, 200],
        _ => [160, 160, 160],
    }
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> io::Result<()> {
    let mut raw = Vec::new();
    for row in 0..height as usize {
        raw.push(0);
        let start = row * width as usize * 4;
        raw.extend_from_slice(&rgba[start..start + width as usize * 4]);
    }
    let deflated = deflate_store(&raw);
    let mut out = Vec::new();
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    write_chunk(&mut out, b"IHDR", &ihdr(width, height));
    write_chunk(&mut out, b"IDAT", &deflated);
    write_chunk(&mut out, b"IEND", &[]);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(&out)?;
    Ok(())
}

fn ihdr(width: u32, height: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&width.to_be_bytes());
    v.extend_from_slice(&height.to_be_bytes());
    v.extend_from_slice(&[8, 6, 0, 0, 0]);
    v
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_src = Vec::with_capacity(4 + data.len());
    crc_src.extend_from_slice(kind);
    crc_src.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_src).to_be_bytes());
}

fn deflate_store(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut offset = 0usize;
    while offset < data.len() {
        let end = (offset + 65535).min(data.len());
        let chunk = &data[offset..end];
        let last = end == data.len();
        out.push(if last { 1 } else { 0 });
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice((!len).to_le_bytes().as_ref());
        out.extend_from_slice(chunk);
        offset = end;
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let bit = crc & 1;
            crc >>= 1;
            if bit != 0 {
                crc ^= 0xEDB88320;
            }
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for byte in data {
        a = (a + u32::from(*byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_encoder_writes_signature() {
        let dir = std::env::temp_dir().join(format!(
            "greppy-tui-png-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).unwrap();
        let mut app = sample_idle_app();
        let path = dir.join("t.png");
        encode_app(&mut app, 60, 18, &path).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        let _ = fs::remove_dir_all(dir);
    }
}
