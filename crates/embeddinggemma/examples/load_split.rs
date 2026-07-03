//! Measure the cold-start cost split of `EmbeddingGemma::load_gguf`:
//! GGUF header parse, tensor upload, tokenizer JSON read + parse, and one
//! query embed. Used to decide which part of warm-invocation latency to
//! attack (mmap vs tokenizer sidecar vs query cache).
//!
//! usage: load_split <model.gguf> <tokenizer.json> [iters]

use std::io::BufReader;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use grepplus_embeddinggemma::{EmbedTask, EmbeddingGemma, LoadOptions};

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let gguf_path = args.next().ok_or("usage: load_split <gguf> <tokenizer.json> [iters]")?;
    let tok_path = args.next().ok_or("usage: load_split <gguf> <tokenizer.json> [iters]")?;
    let iters: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);

    for i in 0..iters {
        // 1. GGUF header/metadata parse via eager BufReader (status quo).
        let t = Instant::now();
        let mut reader = BufReader::new(std::fs::File::open(&gguf_path)?);
        let _ct = gguf_file::Content::read(&mut reader)?;
        let gguf_header_ms = ms(t);
        drop(reader);

        // 2. Tokenizer: raw file read vs serde parse.
        let t = Instant::now();
        let tok_bytes = std::fs::read(&tok_path)?;
        let tok_read_ms = ms(t);
        let t = Instant::now();
        let tok = tokenizers::Tokenizer::from_bytes(&tok_bytes).map_err(|e| e.to_string())?;
        let tok_parse_ms = ms(t);

        // 2b. Compact re-serialized tokenizer parse cost.
        let compact = tok.to_string(false).map_err(|e| e.to_string())?;
        let t = Instant::now();
        let _tok2 =
            tokenizers::Tokenizer::from_bytes(compact.as_bytes()).map_err(|e| e.to_string())?;
        let tok_compact_parse_ms = ms(t);

        // 2c. Pure JSON scan cost (serde_json::Value) — isolates JSON parsing
        // from tokenizers' typed deserialization + BPE merges-map build.
        let t = Instant::now();
        let _v: serde_json::Value = serde_json::from_slice(&tok_bytes)?;
        let tok_value_parse_ms = ms(t);

        // 2d. MessagePack roundtrip: serialize once, measure deserialize.
        let t = Instant::now();
        let rmp_bytes = rmp_serde::to_vec_named(&tok);
        let (rmp_bytes_len, rmp_ser_ms, rmp_de_ms) = match rmp_bytes {
            Ok(b) => {
                let ser_ms = ms(t);
                let t = Instant::now();
                let de: Result<tokenizers::Tokenizer, _> = rmp_serde::from_slice(&b);
                let de_ms = ms(t);
                match de {
                    Ok(_) => (b.len() as i64, ser_ms, de_ms),
                    Err(e) => {
                        println!("rmp deserialize FAILED: {e}");
                        (b.len() as i64, ser_ms, -1.0)
                    }
                }
            }
            Err(e) => {
                println!("rmp serialize FAILED: {e}");
                (-1, -1.0, -1.0)
            }
        };

        // 2e. BPE model rebuild from pre-parsed vocab+merges (sidecar
        // candidate): how much of tok_parse is BPE construction vs JSON?
        let v: serde_json::Value = serde_json::from_slice(&tok_bytes)?;
        let model = &v["model"];
        let vocab: std::collections::HashMap<String, u32> =
            serde_json::from_value(model["vocab"].clone())?;
        let merges: Vec<(String, String)> = model["merges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p[0].as_str().unwrap().to_string(),
                    p[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let t = Instant::now();
        let bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(
                vocab
                    .into_iter()
                    .collect::<tokenizers::models::bpe::Vocab>(),
                merges,
            )
            .unk_token("<unk>".into())
            .fuse_unk(true)
            .byte_fallback(true)
            .build()
            .map_err(|e| e.to_string())?;
        let bpe_build_ms = ms(t);
        drop(bpe);

        // 2f. rmp deserialize of plain vocab+merges (binary sidecar read cost).
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Plain {
            vocab: Vec<(String, u32)>,
            merges: Vec<(String, String)>,
        }
        let vocab2: Vec<(String, u32)> =
            serde_json::from_value::<std::collections::HashMap<String, u32>>(
                model["vocab"].clone(),
            )?
            .into_iter()
            .collect();
        let merges2: Vec<(String, String)> = model["merges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p[0].as_str().unwrap().to_string(),
                    p[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let plain = Plain { vocab: vocab2, merges: merges2 };
        let plain_bytes = rmp_serde::to_vec(&plain)?;
        let t = Instant::now();
        let _plain2: Plain = rmp_serde::from_slice(&plain_bytes)?;
        let plain_de_ms = ms(t);
        println!(
            "iter={i} bpe_build_ms={bpe_build_ms:.1} plain_sidecar_bytes={} plain_de_ms={plain_de_ms:.1}",
            plain_bytes.len()
        );

        // 5. mmap'd GGUF header parse via Cursor (candidate replacement).
        let t = Instant::now();
        let file = std::fs::File::open(&gguf_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let _ct2 = gguf_file::Content::read(&mut cursor)?;
        let gguf_mmap_header_ms = ms(t);
        println!(
            "iter={i} tok_value_parse_ms={tok_value_parse_ms:.1} rmp_bytes={rmp_bytes_len} \
             rmp_ser_ms={rmp_ser_ms:.1} rmp_de_ms={rmp_de_ms:.1} gguf_mmap_header_ms={gguf_mmap_header_ms:.1}"
        );

        // 3. Full model load (GGUF parse + tensor construction + tokenizer).
        let t = Instant::now();
        let model = EmbeddingGemma::load_gguf(
            &gguf_path,
            &tok_path,
            LoadOptions {
                max_length: Some(128),
                ..LoadOptions::default()
            },
        )?;
        let full_load_ms = ms(t);

        // 3b. Full model load WITH tokenizer sidecar cache (first call in
        // the process run writes the cache; later iters read it).
        let cache_dir = std::env::temp_dir().join("grepplus-load-split-cache");
        let t = Instant::now();
        let model_cached = EmbeddingGemma::load_gguf(
            &gguf_path,
            &tok_path,
            LoadOptions {
                max_length: Some(128),
                tokenizer_cache_dir: Some(cache_dir),
                ..LoadOptions::default()
            },
        )?;
        let full_load_cached_ms = ms(t);
        let vc =
            model_cached.embed_one(EmbedTask::CodeRetrievalQuery, "find order total calculation")?;

        // 4. One query embed.
        let t = Instant::now();
        let v = model.embed_one(EmbedTask::CodeRetrievalQuery, "find order total calculation")?;
        let embed_ms = ms(t);

        println!(
            "iter={i} gguf_header_ms={gguf_header_ms:.1} tok_read_ms={tok_read_ms:.1} \
             tok_parse_ms={tok_parse_ms:.1} tok_compact_bytes={} tok_compact_parse_ms={tok_compact_parse_ms:.1} \
             full_load_ms={full_load_ms:.1} full_load_cached_ms={full_load_cached_ms:.1} \
             embed_ms={embed_ms:.1} dim={} cached_vs_plain_max_delta={:.2e}",
            compact.len(),
            v.len(),
            v.iter()
                .zip(vc.iter())
                .map(|(a, b)| (a - b).abs() as f64)
                .fold(0.0, f64::max)
        );
    }
    Ok(())
}
