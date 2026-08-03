//! Embed JSONL block batches with the shipped classification prompt.
//!
//! Input/output records are {"id": ..., "texts": [...]}/
//! {"id": ..., "vectors": [...]}. Newlines stay inside JSON strings, so a
//! block (plus its previous-block context line) is one pooled embedding.

use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

use greppy_embed_native::{DevicePreference, EmbedTask, EmbeddingGemma, LoadOptions};
use serde_json::{json, Value};

const BATCH: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = PathBuf::from(std::env::var("GREPPY_EMBEDDINGGEMMA_GGUF")?);
    let tokenizer = PathBuf::from(std::env::var("GREPPY_EMBEDDINGGEMMA_TOKENIZER")?);
    let mut args = std::env::args().skip(1);
    let device = match args.next().as_deref() {
        Some("--device") => args.next().unwrap_or_else(|| "auto".into()),
        Some(value) => value.into(),
        None => "auto".into(),
    };
    let device: DevicePreference = device.parse().unwrap_or(DevicePreference::Auto);
    let model = EmbeddingGemma::load_gguf(
        &gguf,
        &tokenizer,
        LoadOptions {
            device,
            ..LoadOptions::default()
        },
    )?;
    eprintln!(
        "backend {}, dim {}",
        model.backend_name(),
        model.embedding_dim()
    );

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut total = 0usize;
    for raw in stdin.lock().lines() {
        let value: Value = serde_json::from_str(&raw?)?;
        let id = value.get("id").cloned().unwrap_or(Value::Null);
        let texts: Vec<&str> = value["texts"]
            .as_array()
            .ok_or("texts must be an array")?
            .iter()
            .map(|text| text.as_str().ok_or("text must be a string"))
            .collect::<Result<_, _>>()?;
        let mut vectors = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(BATCH) {
            let prompts = chunk
                .iter()
                .map(|text| EmbedTask::Classification.prompt(text));
            vectors.extend(model.embed_prompts(prompts)?);
        }
        total += texts.len();
        serde_json::to_writer(&mut out, &json!({"id": id, "vectors": vectors}))?;
        writeln!(out)?;
        if total % 8192 < BATCH {
            eprintln!("embedded {total} blocks");
        }
    }
    out.flush()?;
    eprintln!("done: {total} blocks");
    Ok(())
}
