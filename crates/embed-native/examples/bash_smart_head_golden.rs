//! End-to-end fixed-block parity/latency harness for EmbeddingGemma + R5 head.
//!
//! Usage: bash_smart_head_golden DEVICE GGUF TOKENIZER CLASSIFIER BLOCKS_JSON

use std::time::Instant;

use greppy_embed_native::{
    BlockClassifier, DevicePreference, EmbedTask, EmbeddingGemma, LoadOptions,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    blocks: Vec<Block>,
}

#[derive(Deserialize)]
struct Block {
    text: String,
    python_probs: [f32; 4],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(
            "usage: bash_smart_head_golden DEVICE GGUF TOKENIZER CLASSIFIER BLOCKS_JSON".into(),
        );
    }
    let device: DevicePreference = args[0].parse()?;
    let head = BlockClassifier::from_bytes(&std::fs::read(&args[3])?)?;
    let golden: Golden = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    let load_started = Instant::now();
    let model = EmbeddingGemma::load_gguf(
        &args[1],
        &args[2],
        LoadOptions {
            device,
            ..LoadOptions::default()
        },
    )?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let forward_started = Instant::now();
    let vectors = model.embed_prompts(
        golden
            .blocks
            .iter()
            .map(|block| EmbedTask::Classification.prompt(&block.text)),
    )?;
    let probs = head.probabilities_batch(&vectors)?;
    let forward_ms = forward_started.elapsed().as_secs_f64() * 1000.0;
    let max_abs_diff = probs
        .iter()
        .zip(&golden.blocks)
        .flat_map(|(actual, block)| {
            actual
                .iter()
                .zip(block.python_probs)
                .map(|(actual, expected)| (*actual - expected).abs())
        })
        .fold(0.0f32, f32::max);
    let error_threshold = head.error_threshold();
    let decision_disagreements = probs
        .iter()
        .zip(&golden.blocks)
        .filter(|(actual, block)| {
            (actual[0] >= error_threshold) != (block.python_probs[0] >= error_threshold)
        })
        .count();
    println!(
        "device={} n={} load_ms={load_ms:.3} forward_ms={forward_ms:.3} max_abs_diff={max_abs_diff:.9} error_decision_disagreements={decision_disagreements}",
        model.backend_name(),
        golden.blocks.len()
    );
    Ok(())
}
