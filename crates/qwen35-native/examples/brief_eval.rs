//! Evaluate external Qwen GGUF candidates through the production summarizer.
//!
//! Input is JSONL with `id`, `path` (repo-relative file path), and `source`
//! fields. `path` is mandatory: the prompt contract bakes it into training,
//! so eval rows without it must fail loudly instead of silently diverging.
//! Output is one JSON object per line so model checkpoints can be compared
//! without rebuilding the embedded CLI binary.

use std::io::BufRead;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .ok_or("usage: brief_eval MODEL.gguf TOKENIZER.json INPUT.jsonl [DEVICE]")?;
    let tokenizer = args
        .next()
        .ok_or("usage: brief_eval MODEL.gguf TOKENIZER.json INPUT.jsonl [DEVICE]")?;
    let input = args
        .next()
        .ok_or("usage: brief_eval MODEL.gguf TOKENIZER.json INPUT.jsonl [DEVICE]")?;
    let device = args
        .next()
        .as_deref()
        .unwrap_or("auto")
        .parse::<greppy_qwen35_native::DevicePreference>()?;

    let summarizer = greppy_qwen35_native::Qwen35Summarizer::load_gguf(
        gguf,
        tokenizer,
        greppy_qwen35_native::LoadOptions { device },
    )?;
    let reader = std::io::BufReader::new(std::fs::File::open(input)?);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let item: serde_json::Value = serde_json::from_str(&line)?;
        let source = item
            .get("source")
            .and_then(serde_json::Value::as_str)
            .ok_or("brief_eval row is missing string field `source`")?;
        let path = item
            .get("path")
            .and_then(serde_json::Value::as_str)
            .filter(|path| !path.trim().is_empty())
            .ok_or("brief_eval row is missing mandatory string field `path` (repo-relative file path; required so eval never silently diverges from the trained prompt)")?;
        let started = Instant::now();
        let summary = summarizer.summarize_source(path, source)?;
        println!(
            "{}",
            serde_json::json!({
                "id": item.get("id").cloned().unwrap_or(serde_json::Value::Null),
                "backend": summarizer.backend_name(),
                "elapsed_ms": started.elapsed().as_millis(),
                "summary": summary,
            })
        );
    }
    Ok(())
}
