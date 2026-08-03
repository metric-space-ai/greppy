//! Emit the product's own embeddings for lines on stdin, one JSON vector per line.
//!
//! The bash-smart classifier head is an adapter on top of a frozen encoder. It
//! must be fitted to the vectors the product actually produces — the shipped
//! Q4_K GGUF through this crate's kernels — not to full-precision vectors from
//! the upstream checkpoint. Those are different numbers, and a head fitted to
//! the wrong ones scores well in training and drifts at inference.
//!
//! No gradient flows through the quantized weights and none needs to: the
//! encoder is frozen, this is a forward pass, and the head is fitted on its
//! output. Same path the product takes, via `EmbeddingGemma::load_gguf`.
//!
//!   GREPPY_EMBEDDINGGEMMA_GGUF=…/embeddinggemma-300M-Q4_K.gguf \
//!   GREPPY_EMBEDDINGGEMMA_TOKENIZER=…/tokenizer.json \
//!   cargo run --release -p greppy-embed-native --example embed_lines \
//!       --features cuda -- --device cuda < lines.txt > vectors.jsonl

use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

use greppy_embed_native::{DevicePreference, EmbedTask, EmbeddingGemma, LoadOptions};

const BATCH: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = PathBuf::from(
        std::env::var("GREPPY_EMBEDDINGGEMMA_GGUF")
            .map_err(|_| "set GREPPY_EMBEDDINGGEMMA_GGUF to the shipped Q4_K model")?,
    );
    let tokenizer = PathBuf::from(
        std::env::var("GREPPY_EMBEDDINGGEMMA_TOKENIZER")
            .map_err(|_| "set GREPPY_EMBEDDINGGEMMA_TOKENIZER to the shipped tokenizer.json")?,
    );
    let device: DevicePreference = std::env::args()
        .nth(2)
        .as_deref()
        .unwrap_or("auto")
        .parse()
        .unwrap_or(DevicePreference::Auto);

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
    let lines: Vec<String> = stdin.lock().lines().collect::<Result<_, _>>()?;
    eprintln!("embedding {} lines", lines.len());

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut done = 0usize;
    for chunk in lines.chunks(BATCH) {
        // The classification task template, because that is what the head is:
        // a classifier over one output line. Using the document template would
        // embed the line as retrievable prose and fit the head to the wrong
        // vectors.
        let prompts: Vec<String> = chunk
            .iter()
            .map(|line| EmbedTask::Classification.prompt(line))
            .collect();
        for vector in model.embed_prompts(prompts)? {
            let body: Vec<String> = vector.iter().map(|v| format!("{v:.5}")).collect();
            writeln!(out, "[{}]", body.join(","))?;
        }
        done += chunk.len();
        if done % 8192 < BATCH {
            eprintln!("  {done}/{}", lines.len());
        }
    }
    out.flush()?;
    Ok(())
}
