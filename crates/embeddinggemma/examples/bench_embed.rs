use std::collections::BTreeMap;
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use grepplus_embeddinggemma::{
    DevicePreference, EmbedTask, EmbeddingGemma, LoadOptions, EMBEDDING_DIM, PROMPT_VERSION,
};

#[derive(Debug, Clone, Copy)]
enum TaskProfile {
    CodeQuery,
    Document,
    SentenceSimilarity,
}

impl TaskProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "code-query" | "code" => Ok(Self::CodeQuery),
            "document" | "retrieval-document" => Ok(Self::Document),
            "sentence-similarity" | "similarity" => Ok(Self::SentenceSimilarity),
            other => Err(format!(
                "unknown --task {other}; expected code-query, document or sentence-similarity"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::CodeQuery => "code-query",
            Self::Document => "document",
            Self::SentenceSimilarity => "sentence-similarity",
        }
    }

    fn prompt(self, text: &str) -> String {
        match self {
            Self::CodeQuery => EmbedTask::CodeRetrievalQuery.prompt(text),
            Self::Document => EmbedTask::document_with_title(None, text),
            Self::SentenceSimilarity => EmbedTask::SentenceSimilarity.prompt(text),
        }
    }
}

#[derive(Debug)]
struct Args {
    json: bool,
    device_label: &'static str,
    device: DevicePreference,
    tokenizer_json: Option<PathBuf>,
    max_length: Option<usize>,
    warmup: usize,
    iters: usize,
    batch_size: usize,
    task: TaskProfile,
    model_path: PathBuf,
    text: String,
}

#[derive(Debug, Default)]
struct GgufStats {
    tensor_count: usize,
    tensor_dtypes: BTreeMap<String, usize>,
}

#[derive(Debug)]
struct HostInfo {
    os: &'static str,
    arch: &'static str,
    family: &'static str,
    available_parallelism: Option<usize>,
    cpu_brand: Option<String>,
    physical_cpus: Option<String>,
    logical_cpus: Option<String>,
    uname: Option<String>,
    rustc: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if args.iters == 0 {
        return Err("--iters must be greater than 0".into());
    }
    if args.batch_size == 0 {
        return Err("--batch-size must be greater than 0".into());
    }

    let model_format = if is_gguf(&args.model_path) {
        "gguf"
    } else {
        "safetensors"
    };
    let model_bytes = std::fs::metadata(&args.model_path).ok().map(|m| m.len());
    let host = host_info();
    let tokenizer_bytes = args
        .tokenizer_json
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len());
    let gguf_stats = if is_gguf(&args.model_path) {
        Some(read_gguf_stats(&args.model_path)?)
    } else {
        None
    };

    let load_options = LoadOptions {
        device: args.device.clone(),
        max_length: args.max_length,
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let model = if is_gguf(&args.model_path) {
        let tokenizer = args
            .tokenizer_json
            .as_ref()
            .ok_or("GGUF loading requires --tokenizer tokenizer.json")?;
        EmbeddingGemma::load_gguf(&args.model_path, tokenizer, load_options)?
    } else {
        EmbeddingGemma::load_safetensors(&args.model_path, load_options)?
    };
    let load_ms = elapsed_ms(load_start);

    let prompts = make_prompts(args.task, &args.text, args.batch_size);
    let warmup_start = Instant::now();
    for _ in 0..args.warmup {
        black_box(model.embed_prompts(&prompts)?);
    }
    let warmup_ms = elapsed_ms(warmup_start);

    let mut durations = Vec::with_capacity(args.iters);
    let mut checksum = 0.0f64;
    let mut first_norm = None;
    for _ in 0..args.iters {
        let start = Instant::now();
        let vectors = model.embed_prompts(&prompts)?;
        durations.push(elapsed_ms(start));
        if first_norm.is_none() {
            first_norm = vectors.first().map(|v| l2_norm(v));
        }
        checksum += vectors
            .iter()
            .flat_map(|v| v.iter().take(8))
            .map(|v| f64::from(*v))
            .sum::<f64>();
        black_box(&vectors);
    }

    let stats = timing_stats(&durations);
    let total_embeddings = args.iters * args.batch_size;
    let encode_total_ms = durations.iter().sum::<f64>();
    let embeddings_per_s = if encode_total_ms > 0.0 {
        (total_embeddings as f64) / (encode_total_ms / 1000.0)
    } else {
        0.0
    };
    let ms_per_embedding = if total_embeddings > 0 {
        encode_total_ms / total_embeddings as f64
    } else {
        0.0
    };

    if args.json {
        let gguf_json = gguf_stats.as_ref().map(|s| {
            serde_json::json!({
                "tensor_count": s.tensor_count,
                "tensor_dtypes": s.tensor_dtypes,
            })
        });
        println!(
            "{}",
            serde_json::json!({
                "tool": "grepplus-embeddinggemma-bench",
                "prompt_version": PROMPT_VERSION,
                "host": host.to_json(),
                "model_path": args.model_path,
                "model_format": model_format,
                "model_bytes": model_bytes,
                "tokenizer_path": args.tokenizer_json,
                "tokenizer_bytes": tokenizer_bytes,
                "device": args.device_label,
                "task": args.task.as_str(),
                "max_length": args.max_length,
                "embedding_dim": model.embedding_dim(),
                "expected_embedding_dim": EMBEDDING_DIM,
                "warmup": args.warmup,
                "iters": args.iters,
                "batch_size": args.batch_size,
                "load_ms": load_ms,
                "warmup_ms": warmup_ms,
                "encode_total_ms": encode_total_ms,
                "mean_ms": stats.mean_ms,
                "p50_ms": stats.p50_ms,
                "p95_ms": stats.p95_ms,
                "min_ms": stats.min_ms,
                "max_ms": stats.max_ms,
                "embeddings_per_s": embeddings_per_s,
                "ms_per_embedding": ms_per_embedding,
                "first_norm": first_norm,
                "checksum": checksum,
                "gguf": gguf_json,
            })
        );
    } else {
        println!("EmbeddingGemma CPU/Q4 benchmark");
        println!("model: {}", args.model_path.display());
        println!("format: {model_format}");
        println!("device: {}", args.device_label);
        println!("host: {} {} ({})", host.os, host.arch, host.family);
        if let Some(cpu) = &host.cpu_brand {
            println!("cpu_brand: {cpu}");
        }
        if let Some(n) = host.available_parallelism {
            println!("available_parallelism: {n}");
        }
        println!("task: {}", args.task.as_str());
        println!("prompt_version: {PROMPT_VERSION}");
        println!("max_length: {:?}", args.max_length);
        println!("batch_size: {}", args.batch_size);
        println!("warmup: {}", args.warmup);
        println!("iters: {}", args.iters);
        println!("load_ms: {load_ms:.3}");
        println!("warmup_ms: {warmup_ms:.3}");
        println!(
            "encode_ms: mean={:.3} p50={:.3} p95={:.3} min={:.3} max={:.3}",
            stats.mean_ms, stats.p50_ms, stats.p95_ms, stats.min_ms, stats.max_ms
        );
        println!("ms_per_embedding: {ms_per_embedding:.3}");
        println!("embeddings_per_s: {embeddings_per_s:.3}");
        if let Some(norm) = first_norm {
            println!("first_norm: {norm:.6}");
        }
        if let Some(s) = gguf_stats {
            println!("gguf_tensor_count: {}", s.tensor_count);
            println!("gguf_tensor_dtypes: {:?}", s.tensor_dtypes);
        }
    }

    Ok(())
}

impl HostInfo {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "os": self.os,
            "arch": self.arch,
            "family": self.family,
            "available_parallelism": self.available_parallelism,
            "cpu_brand": self.cpu_brand,
            "physical_cpus": self.physical_cpus,
            "logical_cpus": self.logical_cpus,
            "uname": self.uname,
            "rustc": self.rustc,
        })
    }
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut raw = std::env::args().skip(1).collect::<Vec<_>>();
    let mut json = false;
    let mut device = DevicePreference::Cpu;
    let mut device_label = "cpu";
    let mut tokenizer_json = None;
    let mut max_length = Some(128usize);
    let mut warmup = 2usize;
    let mut iters = 20usize;
    let mut batch_size = 1usize;
    let mut task = TaskProfile::CodeQuery;

    while let Some(flag) = raw.first().map(String::as_str) {
        match flag {
            "--json" => {
                json = true;
                raw.remove(0);
            }
            "--cpu" => {
                device = DevicePreference::Cpu;
                device_label = "cpu";
                raw.remove(0);
            }
            "--auto" => {
                device = LoadOptions::auto().device;
                device_label = "auto";
                raw.remove(0);
            }
            #[cfg(feature = "metal")]
            "--metal" => {
                device = DevicePreference::Metal(0);
                device_label = "metal";
                raw.remove(0);
            }
            "--tokenizer" => {
                raw.remove(0);
                tokenizer_json = Some(take_path_arg(&mut raw, "--tokenizer")?);
            }
            "--max-length" => {
                raw.remove(0);
                max_length = Some(take_usize_arg(&mut raw, "--max-length")?);
            }
            "--no-max-length" => {
                max_length = None;
                raw.remove(0);
            }
            "--warmup" => {
                raw.remove(0);
                warmup = take_usize_arg(&mut raw, "--warmup")?;
            }
            "--iters" => {
                raw.remove(0);
                iters = take_usize_arg(&mut raw, "--iters")?;
            }
            "--batch-size" => {
                raw.remove(0);
                batch_size = take_usize_arg(&mut raw, "--batch-size")?;
            }
            "--task" => {
                raw.remove(0);
                let value = take_string_arg(&mut raw, "--task")?;
                task = TaskProfile::parse(&value)?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => break,
        }
    }

    let model_path = raw
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| usage_error("missing <model-dir|model.gguf>"))?;
    raw.remove(0);
    let text = if raw.is_empty() {
        "find order total calculation".to_string()
    } else {
        raw.join(" ")
    };

    Ok(Args {
        json,
        device_label,
        device,
        tokenizer_json,
        max_length,
        warmup,
        iters,
        batch_size,
        task,
        model_path,
        text,
    })
}

fn take_path_arg(
    args: &mut Vec<String>,
    flag: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(take_string_arg(args, flag)?))
}

fn take_string_arg(
    args: &mut Vec<String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if args.is_empty() {
        return Err(format!("{flag} requires a value").into());
    }
    Ok(args.remove(0))
}

fn take_usize_arg(args: &mut Vec<String>, flag: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let raw = take_string_arg(args, flag)?;
    raw.parse::<usize>()
        .map_err(|e| format!("{flag} requires an unsigned integer, got {raw:?}: {e}").into())
}

fn usage_error(msg: &str) -> Box<dyn std::error::Error> {
    format!("{msg}\n\n{}", usage()).into()
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "usage: bench_embed [--json] [--cpu|--auto] [--tokenizer tokenizer.json] [--max-length N|--no-max-length] [--warmup N] [--iters N] [--batch-size N] [--task code-query|document|sentence-similarity] <model-dir|model.gguf> [text]"
}

fn is_gguf(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()) == Some("gguf")
}

fn host_info() -> HostInfo {
    HostInfo {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        family: std::env::consts::FAMILY,
        available_parallelism: std::thread::available_parallelism().ok().map(|n| n.get()),
        cpu_brand: cpu_brand(),
        physical_cpus: sysctl("hw.physicalcpu").or_else(|| linux_cpuinfo_value("cpu cores")),
        logical_cpus: sysctl("hw.logicalcpu").or_else(|| linux_cpuinfo_value("siblings")),
        uname: command_output("uname", &["-a"]),
        rustc: command_output("rustc", &["-Vv"]),
    }
}

fn cpu_brand() -> Option<String> {
    sysctl("machdep.cpu.brand_string")
        .or_else(|| linux_cpuinfo_value("model name"))
        .or_else(|| command_output("lscpu", &[]).and_then(|out| lscpu_value(&out, "Model name")))
}

fn sysctl(key: &str) -> Option<String> {
    command_output("sysctl", &["-n", key])
}

fn linux_cpuinfo_value(key: &str) -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let key_lower = key.to_ascii_lowercase();
    text.lines().find_map(|line| {
        let (k, v) = line.split_once(':')?;
        if k.trim().eq_ignore_ascii_case(&key_lower) {
            Some(v.trim().to_string())
        } else {
            None
        }
    })
}

fn lscpu_value(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (k, v) = line.split_once(':')?;
        if k.trim() == key {
            Some(v.trim().to_string())
        } else {
            None
        }
    })
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn read_gguf_stats(path: &Path) -> Result<GgufStats, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let content = gguf_file::Content::read(&mut reader)?;
    let mut stats = GgufStats {
        tensor_count: content.tensor_infos.len(),
        tensor_dtypes: BTreeMap::new(),
    };
    for info in content.tensor_infos.values() {
        *stats
            .tensor_dtypes
            .entry(format!("{:?}", info.ggml_dtype))
            .or_insert(0) += 1;
    }
    Ok(stats)
}

fn make_prompts(task: TaskProfile, text: &str, batch_size: usize) -> Vec<String> {
    (0..batch_size)
        .map(|i| {
            if batch_size == 1 {
                task.prompt(text)
            } else {
                task.prompt(&format!("{text}\nbench item {i}"))
            }
        })
        .collect()
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt()
}

#[derive(Debug)]
struct TimingStats {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn timing_stats(values: &[f64]) -> TimingStats {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sum = sorted.iter().sum::<f64>();
    TimingStats {
        mean_ms: sum / sorted.len() as f64,
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        min_ms: *sorted.first().unwrap(),
        max_ms: *sorted.last().unwrap(),
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let idx = ((sorted.len() as f64 * pct).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}
