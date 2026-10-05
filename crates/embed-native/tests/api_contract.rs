use std::path::{Path, PathBuf};

use greppy_embed_native::{
    DevicePreference, EmbedTask, EmbeddingGemma, LoadOptions, CODE_RETRIEVAL_PROFILE,
    EMBEDDING_DIM, PROMPT_VERSION,
};

fn main() {
    assert_eq!(EMBEDDING_DIM, 768);
    assert_eq!(PROMPT_VERSION, "embeddinggemma-code-retrieval-st-v3");
    assert_eq!(CODE_RETRIEVAL_PROFILE, "embeddinggemma_code_retrieval");
    assert_eq!(
        EmbedTask::CodeRetrievalQuery.prompt("find retry handler"),
        "task: code retrieval | query: find retry handler"
    );
    assert_eq!(
        EmbedTask::document_with_title(Some("src/lib.rs"), "fn main() {}"),
        "title: src/lib.rs | text: fn main() {}"
    );

    let Some((gguf, tokenizer)) = model_paths() else {
        eprintln!(
            "skipping production API contract: EMBED_NATIVE_GGUF or EMBED_NATIVE_TOKENIZER unset"
        );
        return;
    };

    if let Ok(raw_device) = std::env::var("EMBED_NATIVE_API_CONTRACT_DEVICE") {
        let device = raw_device
            .parse::<DevicePreference>()
            .unwrap_or_else(|e| panic!("invalid EMBED_NATIVE_API_CONTRACT_DEVICE: {e}"));
        let model = EmbeddingGemma::load_gguf(
            &gguf,
            &tokenizer,
            LoadOptions {
                device,
                max_length: Some(8),
                tokenizer_cache_dir: None,
            },
        )
        .unwrap_or_else(|e| {
            panic!(
                "load requested-device EmbeddingGemma {}: {e}",
                gguf.display()
            )
        });
        println!("requested {raw_device} backend: {}", model.backend_name());
        assert_embedding_shape(
            &model
                .embed_one(EmbedTask::CodeRetrievalQuery, "x")
                .expect("requested-device one-token-ish query"),
        );
        return;
    }

    let cpu_model = EmbeddingGemma::load_gguf(
        &gguf,
        &tokenizer,
        LoadOptions {
            device: DevicePreference::Cpu,
            max_length: Some(8),
            tokenizer_cache_dir: None,
        },
    )
    .unwrap_or_else(|e| panic!("load native CPU EmbeddingGemma {}: {e}", gguf.display()));
    println!("forced cpu backend: {}", cpu_model.backend_name());
    assert_eq!(cpu_model.backend_name(), "cpu");
    assert_embedding_shape(
        &cpu_model
            .embed_one(EmbedTask::CodeRetrievalQuery, "x")
            .expect("forced CPU one-token-ish query"),
    );

    if std::env::var_os("EMBED_NATIVE_API_CONTRACT_CPU_ONLY").is_some() {
        return;
    }

    let model = EmbeddingGemma::load_gguf(
        &gguf,
        &tokenizer,
        LoadOptions {
            max_length: Some(8),
            ..LoadOptions::auto()
        },
    )
    .unwrap_or_else(|e| panic!("load native EmbeddingGemma {}: {e}", gguf.display()));

    println!("auto backend: {}", model.backend_name());
    #[cfg(all(feature = "metal", target_os = "macos"))]
    assert_eq!(model.backend_name(), "metal");
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    assert_eq!(model.backend_name(), "cuda");
    assert_eq!(model.embedding_dim(), EMBEDDING_DIM);

    let empty = model.embed_documents(&[]).expect("empty batch");
    assert!(empty.is_empty(), "empty batch must return empty result");

    let one_token = model
        .embed_one(EmbedTask::CodeRetrievalQuery, "x")
        .expect("one-token-ish query");
    assert_embedding_shape(&one_token);

    let docs = [
        (Some("src/a.rs"), "fn a() {}"),
        (
            Some("src/long.rs"),
            "fn long_name() { let value = 42; value.to_string(); }",
        ),
        (None, "x"),
    ];
    let batch = model.embed_documents(&docs).expect("variable-length batch");
    assert_eq!(batch.len(), docs.len());
    for vector in &batch {
        assert_embedding_shape(vector);
    }

    let oversize = "token ".repeat(10_000);
    let truncated = model
        .embed_one(EmbedTask::CodeRetrievalQuery, &oversize)
        .expect("oversize input should truncate through tokenizer");
    assert_embedding_shape(&truncated);
}

fn model_paths() -> Option<(PathBuf, PathBuf)> {
    let gguf = std::env::var_os("EMBED_NATIVE_GGUF").map(PathBuf::from)?;
    let tokenizer = std::env::var_os("EMBED_NATIVE_TOKENIZER")
        .map(PathBuf::from)
        .or_else(|| {
            let plan = std::env::var("GREPPY_EMBEDDINGGEMMA_TOKENIZER").unwrap_or_default();
            let plan_path = Path::new(&plan);
            plan_path.exists().then(|| plan_path.to_path_buf())
        })?;
    Some((gguf, tokenizer))
}

fn assert_embedding_shape(vector: &[f32]) {
    assert_eq!(vector.len(), EMBEDDING_DIM);
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (0.99..=1.01).contains(&norm),
        "expected approximately l2-normalized embedding, got norm {norm}"
    );
    assert!(
        vector.iter().all(|v| v.is_finite()),
        "embedding contains non-finite values"
    );
}
