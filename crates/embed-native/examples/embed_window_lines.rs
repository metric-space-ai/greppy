//! Per-line states from ONE forward pass over a window of lines ("Fall A").
//!
//! The pooled path (`embed_prompts`, one vector per line, lines embedded in
//! isolation) measured 6-8% precision on the rare classes — a pooled vector
//! keeps semantics and discards position, so a head fitted to it cannot localise
//! anything. The spec asks for the opposite: "the collapsed middle runs through
//! the resident model in ~64-line windows with ~8-line overlap; a small head
//! reads the hidden state at each line-end token — ONE forward pass per window".
//!
//! This emits exactly those per-line states, so the head can be trained on the
//! representation it will see at inference.
//!
//! Input: a JSONL stream, one object per wall: {"id": "...", "lines": [...]}.
//! Output: one JSON object per wall: {"id", "vectors": [[f32; D]; L]} — one
//! vector per input line, in order.
//!
//!   GREPPY_EMBEDDINGGEMMA_GGUF=… GREPPY_EMBEDDINGGEMMA_TOKENIZER=… \
//!   cargo run --release -p greppy-embed-native --example embed_window_lines \
//!       --features cuda -- cuda 64 8 < walls.jsonl > states.jsonl

use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

use greppy_embed_native::{DevicePreference, EmbeddingGemma, LoadOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = PathBuf::from(std::env::var("GREPPY_EMBEDDINGGEMMA_GGUF")?);
    let tok = PathBuf::from(std::env::var("GREPPY_EMBEDDINGGEMMA_TOKENIZER")?);
    let mut args = std::env::args().skip(1);
    let device: DevicePreference = args
        .next()
        .unwrap_or_else(|| "auto".into())
        .parse()
        .unwrap_or(DevicePreference::Auto);
    let window: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);
    let overlap: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
    let stride = window.saturating_sub(overlap).max(1);

    let model = EmbeddingGemma::load_gguf(
        &gguf,
        &tok,
        LoadOptions {
            device,
            ..LoadOptions::default()
        },
    )?;
    eprintln!(
        "backend {} dim {} window {window} overlap {overlap}",
        model.backend_name(),
        model.embedding_dim()
    );

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut walls = 0usize;

    for raw in stdin.lock().lines() {
        let raw = raw?;
        if raw.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&raw)?;
        let id = value["id"].as_str().unwrap_or("?").to_string();
        let lines: Vec<String> = value["lines"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();
        if lines.is_empty() {
            continue;
        }

        // Overlapping windows; a line covered twice keeps the state from the
        // window whose centre it is nearest to, so every line is read with the
        // most context the windowing can give it.
        let dim = model.embedding_dim();
        let mut acc = vec![vec![0f32; dim]; lines.len()];
        let mut best = vec![usize::MAX; lines.len()];

        let mut start = 0usize;
        while start < lines.len() {
            let end = (start + window).min(lines.len());
            let slice = &lines[start..end];
            let states = model.line_states(slice)?;
            let centre = start + (end - start) / 2;
            for (offset, vector) in states.into_iter().enumerate() {
                let index = start + offset;
                let distance = index.abs_diff(centre);
                if distance < best[index] {
                    best[index] = distance;
                    acc[index] = vector;
                }
            }
            if end == lines.len() {
                break;
            }
            start += stride;
        }

        let body: Vec<String> = acc
            .iter()
            .map(|v| {
                let nums: Vec<String> = v.iter().map(|x| format!("{x:.5}")).collect();
                format!("[{}]", nums.join(","))
            })
            .collect();
        writeln!(out, "{{\"id\":\"{id}\",\"vectors\":[{}]}}", body.join(","))?;
        walls += 1;
        if walls % 200 == 0 {
            eprintln!("  {walls} walls");
            out.flush()?;
        }
    }
    out.flush()?;
    eprintln!("done: {walls} walls");
    Ok(())
}
