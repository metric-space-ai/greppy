use greppy_embed_native::head_input::{sha256, Candidate, Limits, Source, VerifiedSource};
use greppy_embed_native::{
    DevicePreference, EmbeddingGemma, LoadOptions, PromptTokenizer, TokenizerConfig,
};
use serde::Deserialize;
use std::io::{BufRead, Write};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    source: Source,
    candidates: Vec<Candidate>,
    limits: Limits,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let tokenizer_path = args
        .next()
        .ok_or("usage: head_inputs TOKENIZER prepare|cpu|metal|cuda [GGUF]")?;
    let mode = args.next().ok_or("explicit mode required")?;
    let model_path = args.next();
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let tokenizer_hash = sha256(&std::fs::read(&tokenizer_path)?);
    let binary_hash = sha256(&std::fs::read(std::env::current_exe()?)?);
    let tokenizer = PromptTokenizer::from_file(&tokenizer_path, TokenizerConfig::default())?;
    let model = if mode == "prepare" {
        if model_path.is_some() {
            return Err("prepare must not load a model".into());
        }
        None
    } else {
        let device: DevicePreference = mode.parse()?;
        if device == DevicePreference::Auto {
            return Err("explicit backend required".into());
        }
        let path = model_path.as_ref().ok_or("GGUF required")?;
        Some(EmbeddingGemma::load_gguf(
            path,
            &tokenizer_path,
            LoadOptions {
                device,
                ..LoadOptions::default()
            },
        )?)
    };
    let model_hash = match model_path {
        Some(path) => Some(sha256(&std::fs::read(path)?)),
        None => None,
    };
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut sources = std::collections::HashSet::new();
    for raw in std::io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&raw?)?;
        if !sources.insert(request.source.id.clone()) {
            return Err("duplicate source".into());
        }
        let source = VerifiedSource::new(&request.source)?;
        let mut ids = std::collections::HashSet::new();
        for c in &request.candidates {
            if !ids.insert(&c.id) {
                return Err("duplicate candidate".into());
            }
        }
        if request.candidates.is_empty() {
            return Err("empty candidates".into());
        }
        for c in &request.candidates {
            let inputs = source.prepare(c, request.limits, |s| {
                tokenizer.token_len(s).map_err(|e| e.to_string())
            })?;
            for chunk in inputs.chunks(16) {
                let exact = tokenizer.encode_prompts_exact(chunk.iter().map(|x| &x.prompt))?;
                for (row, mask) in chunk.iter().zip(&exact.attention_mask) {
                    if row.token_count != mask.iter().map(|x| *x as usize).sum::<usize>() {
                        return Err("prepared/native token count mismatch".into());
                    }
                }
                let vectors = match &model {
                    Some(m) => Some(m.embed_prompts_exact(chunk.iter().map(|x| &x.prompt))?),
                    None => None,
                };
                if let Some(batch) = &vectors {
                    if batch.len() != chunk.len() {
                        return Err("embedding count mismatch".into());
                    }
                    for v in batch {
                        if v.len() != 768 || v.iter().any(|n| !n.is_finite()) {
                            return Err("invalid vector".into());
                        }
                    }
                }
                let backend = model.as_ref().map(|m| m.backend_name());
                for (i, input) in chunk.iter().enumerate() {
                    let vector = vectors.as_ref().map(|v| &v[i]);
                    let row = serde_json::json!({"schema": "greppy.heads.native-feature.v1", "input": input,
                        "tokenizer_sha256": tokenizer_hash, "binary_sha256": binary_hash,
                        "model_sha256": model_hash, "backend": backend, "vector": vector,
                        "representation": "frozen-final-mean-dense2-dense3-l2", "production_eligible": false});
                    serde_json::to_writer(&mut out, &row)?;
                    writeln!(out)?;
                }
                out.flush()?;
            }
        }
    }
    Ok(())
}
