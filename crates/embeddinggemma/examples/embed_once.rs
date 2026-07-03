use std::path::PathBuf;

use grepplus_embeddinggemma::{DevicePreference, EmbedTask, EmbeddingGemma, LoadOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut json = false;
    let mut device = DevicePreference::Cpu;
    let mut tokenizer_json: Option<PathBuf> = None;
    while let Some(flag) = args.first().map(String::as_str) {
        match flag {
            "--json" => {
                json = true;
                args.remove(0);
            }
            "--auto" => {
                device = LoadOptions::auto().device;
                args.remove(0);
            }
            #[cfg(feature = "metal")]
            "--metal" => {
                device = DevicePreference::Metal(0);
                args.remove(0);
            }
            "--tokenizer" => {
                args.remove(0);
                let path = args
                    .first()
                    .map(PathBuf::from)
                    .ok_or("--tokenizer requires a tokenizer.json path")?;
                tokenizer_json = Some(path);
                args.remove(0);
            }
            _ => break,
        }
    }
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .ok_or("usage: embed_once [--json] [--auto|--metal] [--tokenizer tokenizer.json] <model-dir|model.gguf> <text>")?;
    let text = args.iter().skip(1).cloned().collect::<Vec<_>>().join(" ");
    if text.trim().is_empty() {
        return Err("usage: embed_once [--json] [--auto|--metal] [--tokenizer tokenizer.json] <model-dir|model.gguf> <text>".into());
    }

    let options = LoadOptions {
        device,
        ..LoadOptions::default()
    };
    let model = if model_dir.extension().and_then(|s| s.to_str()) == Some("gguf") {
        let tokenizer_json =
            tokenizer_json.ok_or("GGUF loading requires --tokenizer tokenizer.json")?;
        EmbeddingGemma::load_gguf(model_dir, tokenizer_json, options)?
    } else {
        EmbeddingGemma::load_safetensors(model_dir, options)?
    };
    let vector = model.embed_one(EmbedTask::CodeRetrievalQuery, &text)?;
    if json {
        println!("{}", serde_json::to_string(&vector)?);
        return Ok(());
    }
    let norm = vector
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    let preview = vector
        .iter()
        .take(8)
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("dim={} norm={norm:.6} first8=[{preview}]", vector.len());
    Ok(())
}
