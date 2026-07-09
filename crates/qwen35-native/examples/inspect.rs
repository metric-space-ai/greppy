fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: cargo run -p greppy-qwen35-native --example inspect -- MODEL.gguf")?;
    let model = greppy_embed_native::GgufModel::open(path)?;
    println!("metadata:");
    for key in [
        "general.architecture",
        "qwen35.block_count",
        "qwen35.context_length",
        "qwen35.embedding_length",
        "qwen35.feed_forward_length",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length",
        "qwen35.attention.value_length",
        "qwen35.rope.dimension_count",
        "qwen35.full_attention_interval",
        "qwen35.ssm.inner_size",
        "qwen35.ssm.group_count",
        "qwen35.ssm.time_step_rank",
    ] {
        if let Some(value) = model.metadata().get(key) {
            println!("  {key}: {value:?}");
        }
    }
    println!("tensors:");
    let mut names = model.tensor_infos().keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        let info = &model.tensor_infos()[&name];
        println!("  {name:34} {:>5} {:?}", info.dtype, info.shape);
    }
    Ok(())
}
