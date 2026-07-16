//! Raw production-path Qwen samples for the inference performance contract.

use std::io::BufRead;
use std::time::Instant;

use serde::Deserialize;
use tokenizers::Tokenizer;

const RAW_SCHEMA_VERSION: &str = "greppy.inference-performance.raw.v1";
const PP512_SEMANTICS: &str = "qwen_target_prefill_exact_512_v1";
const TG128_SEMANTICS: &str = "qwen_greedy_generation_exact_128_v1";
const BRIEF_SEMANTICS: &str = "greppy_brief_production_mtp_v1";

#[derive(Deserialize)]
struct PromptCase {
    id: String,
    /// Repo-relative file path of the source span; part of the trained
    /// brief prompt contract and therefore mandatory.
    path: String,
    source: String,
    max_output_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let gguf = args.next().ok_or(usage())?;
    let tokenizer_path = args.next().ok_or(usage())?;
    let prompts_path = args.next().ok_or(usage())?;
    let device = args
        .next()
        .ok_or(usage())?
        .parse::<greppy_qwen35_native::DevicePreference>()?;
    let device_name = device.as_str();
    let samples = parse_count(args.next(), "SAMPLES", 5)?;
    let warmups = parse_count(args.next(), "WARMUPS", 1)?;
    if args.next().is_some() {
        return Err(usage().into());
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let cases = read_cases(&prompts_path)?;
    let prepared = cases
        .into_iter()
        .map(|case| {
            if case.max_output_tokens == 0 || case.max_output_tokens > 64 {
                return Err(format!("{}: max_output_tokens must be in 1..=64", case.id));
            }
            let prompt = production_chat_prompt(&case.path, &case.source);
            let encoding = tokenizer
                .encode(prompt, true)
                .map_err(|error| format!("{}: cannot tokenize prompt: {error}", case.id))?;
            let token_ids = encoding.get_ids().to_vec();
            if !(100..=500).contains(&token_ids.len()) {
                return Err(format!(
                    "{}: production prompt has {} tokens; contract requires 100..=500",
                    case.id,
                    token_ids.len()
                ));
            }
            Ok((case, token_ids))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let summarizer = greppy_qwen35_native::Qwen35Summarizer::load_gguf(
        &gguf,
        &tokenizer_path,
        greppy_qwen35_native::LoadOptions { device },
    )?;
    if !summarizer.inventory().has_mtp() {
        return Err("Qwen benchmark model has no MTP layer".into());
    }

    let (_, benchmark_ids) = prepared
        .first()
        .ok_or("benchmark prompt preparation returned no cases")?;
    let pp512_ids = exact_token_count(
        benchmark_ids,
        greppy_qwen35_native::DIAGNOSTIC_TARGET_PREFILL_TOKENS,
    )?;

    for _ in 0..warmups {
        std::hint::black_box(summarizer.diagnostic_target_prefill_512(&pp512_ids)?);
    }
    for sample_index in 0..samples {
        let result = summarizer.diagnostic_target_prefill_512(&pp512_ids)?;
        println!(
            "{}",
            serde_json::json!({
                "schema_version": RAW_SCHEMA_VERSION,
                "model_family": "qwen35_mtp",
                "workload": "qwen_pp512",
                "semantics": PP512_SEMANTICS,
                "generation_path": "target_prefill",
                "case_id": "qwen_pp512_exact",
                "sample_index": sample_index,
                "elapsed_ns": duration_ns(result.elapsed)?,
                "input_token_ids": pp512_ids,
                "output_token_ids": [],
                "output_limit": 0,
                "backend": summarizer.backend_name(),
                "device": device_name,
            })
        );
    }

    for _ in 0..warmups {
        std::hint::black_box(summarizer.diagnostic_generate_mtp_greedy(
            benchmark_ids,
            greppy_qwen35_native::DIAGNOSTIC_MAX_OUTPUT_TOKENS,
        )?);
    }
    for sample_index in 0..samples {
        let result = summarizer.diagnostic_generate_mtp_greedy(
            benchmark_ids,
            greppy_qwen35_native::DIAGNOSTIC_MAX_OUTPUT_TOKENS,
        )?;
        println!(
            "{}",
            serde_json::json!({
                "schema_version": RAW_SCHEMA_VERSION,
                "model_family": "qwen35_mtp",
                "workload": "qwen_tg128",
                "semantics": TG128_SEMANTICS,
                "generation_path": "production_mtp",
                "case_id": "qwen_tg128_exact",
                "sample_index": sample_index,
                "elapsed_ns": duration_ns(result.decode)?,
                "input_token_ids": benchmark_ids,
                "output_token_ids": result.output_token_ids,
                "output_limit": greppy_qwen35_native::DIAGNOSTIC_MAX_OUTPUT_TOKENS,
                "target_prefill_ns": duration_ns(result.target_prefill)?,
                "mtp_prefill_ns": duration_ns(result.mtp_prefill)?,
                "decode_ns": duration_ns(result.decode)?,
                "mtp_used": result.mtp.used,
                "mtp_cycles": result.mtp.cycles,
                "mtp_drafted_tokens": result.mtp.drafted_tokens,
                "mtp_accepted_tokens": result.mtp.accepted_tokens,
                "mtp_fallback": result.mtp.fallback,
                "backend": summarizer.backend_name(),
                "device": device_name,
            })
        );
    }

    for (case, token_ids) in prepared {
        for _ in 0..warmups {
            std::hint::black_box(summarizer.summarize_source(&case.path, &case.source)?);
        }
        for sample_index in 0..samples {
            let started = Instant::now();
            let summary = summarizer.summarize_source(&case.path, &case.source)?;
            let elapsed_ns = u64::try_from(started.elapsed().as_nanos())
                .map_err(|_| "sample duration does not fit u64 nanoseconds")?;
            std::hint::black_box(&summary);
            println!(
                "{}",
                serde_json::json!({
                    "schema_version": RAW_SCHEMA_VERSION,
                    "model_family": "qwen35_mtp",
                    "workload": "greppy_brief",
                    "semantics": BRIEF_SEMANTICS,
                    "generation_path": "production_mtp",
                    "case_id": case.id,
                    "sample_index": sample_index,
                    "elapsed_ns": elapsed_ns,
                    "input_token_ids": token_ids,
                    "output_token_ids": [],
                    "output_limit": case.max_output_tokens,
                    "backend": summarizer.backend_name(),
                    "summary_bullets": summary.len(),
                    "device": device_name,
                })
            );
        }
    }
    Ok(())
}

fn exact_token_count(
    token_ids: &[u32],
    target: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if token_ids.is_empty() {
        return Err("cannot expand an empty token sequence".into());
    }
    Ok(token_ids.iter().copied().cycle().take(target).collect())
}

fn duration_ns(duration: std::time::Duration) -> Result<u64, Box<dyn std::error::Error>> {
    let nanos = u64::try_from(duration.as_nanos())
        .map_err(|_| "sample duration does not fit u64 nanoseconds")?;
    if nanos == 0 {
        return Err("sample duration must be positive".into());
    }
    Ok(nanos)
}

fn read_cases(path: &str) -> Result<Vec<PromptCase>, Box<dyn std::error::Error>> {
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut cases = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: PromptCase = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{}: {error}", path, index + 1))?;
        if case.id.trim().is_empty() || case.path.trim().is_empty() || case.source.trim().is_empty()
        {
            return Err(format!(
                "{}:{}: id, path, and source must be non-empty",
                path,
                index + 1
            )
            .into());
        }
        cases.push(case);
    }
    if cases.is_empty() {
        return Err(format!("{path}: no prompt cases").into());
    }
    Ok(cases)
}

fn production_chat_prompt(path: &str, source: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        greppy_qwen35_native::brief_prompt(path, source).trim()
    )
}

fn parse_count(
    value: Option<String>,
    name: &str,
    default: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let count = value
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| format!("{name} must be a positive integer"))?
        .unwrap_or(default);
    if count == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(count)
}

fn usage() -> &'static str {
    "usage: qwen_inference_contract MODEL.gguf TOKENIZER.json PROMPTS.jsonl DEVICE [SAMPLES] [WARMUPS]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_token_count_repeats_without_conversion() {
        let ids = exact_token_count(&[3, 5, 7], 8).expect("expand token IDs");
        assert_eq!(ids, [3, 5, 7, 3, 5, 7, 3, 5]);
    }

    #[test]
    fn duration_contract_rejects_zero() {
        assert!(duration_ns(std::time::Duration::ZERO).is_err());
        assert_eq!(duration_ns(std::time::Duration::from_nanos(7)).unwrap(), 7);
    }
}
