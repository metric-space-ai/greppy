use greppy_embed_native::{GgmlDType, GgufModel, TensorInfo, Value};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Inventory {
    pub architecture: String,
    pub block_count: usize,
    pub hidden_size: usize,
    pub feed_forward_size: usize,
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub rope_dim: usize,
    pub context_length: usize,
    pub full_attention_interval: usize,
    pub ssm_inner_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35Expected {
    pub architecture: &'static str,
    pub block_count: usize,
    pub hidden_size: usize,
    pub feed_forward_size: usize,
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub rope_dim: usize,
    pub full_attention_interval: usize,
    pub ssm_inner_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
}

pub const QWEN35_08B_EXPECTED: Qwen35Expected = Qwen35Expected {
    architecture: "qwen35",
    block_count: 24,
    hidden_size: 1024,
    feed_forward_size: 3584,
    vocab_size: 248_320,
    attention_heads: 8,
    kv_heads: 2,
    head_dim: 256,
    value_dim: 256,
    rope_dim: 64,
    full_attention_interval: 4,
    ssm_inner_size: 2048,
    ssm_group_count: 16,
    ssm_time_step_rank: 16,
};

impl Qwen35Inventory {
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let architecture = metadata_str(model, "general.architecture")?.to_string();
        let block_count = metadata_usize(model, "qwen35.block_count")?;
        let hidden_size = metadata_usize(model, "qwen35.embedding_length")?;
        let feed_forward_size = metadata_usize(model, "qwen35.feed_forward_length")?;
        let attention_heads = metadata_usize(model, "qwen35.attention.head_count")?;
        let kv_heads = metadata_usize(model, "qwen35.attention.head_count_kv")?;
        let head_dim = metadata_usize(model, "qwen35.attention.key_length").unwrap_or_else(|_| {
            hidden_size
                .checked_mul(2)
                .and_then(|v| v.checked_div(attention_heads.max(1)))
                .unwrap_or(0)
        });
        let value_dim = metadata_usize(model, "qwen35.attention.value_length").unwrap_or(head_dim);
        let rope_dim = metadata_usize(model, "qwen35.rope.dimension_count")?;
        let context_length = metadata_usize(model, "qwen35.context_length")
            .or_else(|_| metadata_usize(model, "qwen35.context_length_train"))
            .unwrap_or(262_144);
        let full_attention_interval = metadata_usize(model, "qwen35.full_attention_interval")?;
        let ssm_inner_size = metadata_usize(model, "qwen35.ssm.inner_size")?;
        let ssm_group_count = metadata_usize(model, "qwen35.ssm.group_count")?;
        let ssm_time_step_rank = metadata_usize(model, "qwen35.ssm.time_step_rank")?;
        let vocab_size = tensor_outer_dim(model, "token_embd.weight")?;
        let inv = Self {
            architecture,
            block_count,
            hidden_size,
            feed_forward_size,
            vocab_size,
            attention_heads,
            kv_heads,
            head_dim,
            value_dim,
            rope_dim,
            context_length,
            full_attention_interval,
            ssm_inner_size,
            ssm_group_count,
            ssm_time_step_rank,
        };
        inv.validate()?;
        Ok(inv)
    }

    pub fn validate(&self) -> Result<()> {
        let expected = QWEN35_08B_EXPECTED;
        require_eq(
            "architecture",
            self.architecture.as_str(),
            expected.architecture,
        )?;
        require_eq("block_count", self.block_count, expected.block_count)?;
        require_eq("hidden_size", self.hidden_size, expected.hidden_size)?;
        require_eq(
            "feed_forward_size",
            self.feed_forward_size,
            expected.feed_forward_size,
        )?;
        require_eq("vocab_size", self.vocab_size, expected.vocab_size)?;
        require_eq(
            "attention_heads",
            self.attention_heads,
            expected.attention_heads,
        )?;
        require_eq("kv_heads", self.kv_heads, expected.kv_heads)?;
        require_eq("head_dim", self.head_dim, expected.head_dim)?;
        require_eq("value_dim", self.value_dim, expected.value_dim)?;
        require_eq("rope_dim", self.rope_dim, expected.rope_dim)?;
        require_eq(
            "full_attention_interval",
            self.full_attention_interval,
            expected.full_attention_interval,
        )?;
        require_eq(
            "ssm_inner_size",
            self.ssm_inner_size,
            expected.ssm_inner_size,
        )?;
        require_eq(
            "ssm_group_count",
            self.ssm_group_count,
            expected.ssm_group_count,
        )?;
        require_eq(
            "ssm_time_step_rank",
            self.ssm_time_step_rank,
            expected.ssm_time_step_rank,
        )?;
        Ok(())
    }

    pub fn validate_core_tensors(&self, model: &GgufModel) -> Result<()> {
        require_quant_tensor(
            model,
            "token_embd.weight",
            &[self.vocab_size, self.hidden_size],
        )?;
        require_f32_tensor(model, "output_norm.weight", &[self.hidden_size])?;
        for layer in 0..self.block_count {
            require_f32_tensor(
                model,
                &format!("blk.{layer}.attn_norm.weight"),
                &[self.hidden_size],
            )?;
            require_f32_tensor(
                model,
                &format!("blk.{layer}.post_attention_norm.weight"),
                &[self.hidden_size],
            )?;
            require_quant_tensor(
                model,
                &format!("blk.{layer}.ffn_gate.weight"),
                &[self.feed_forward_size, self.hidden_size],
            )?;
            require_quant_tensor(
                model,
                &format!("blk.{layer}.ffn_up.weight"),
                &[self.feed_forward_size, self.hidden_size],
            )?;
            require_quant_tensor(
                model,
                &format!("blk.{layer}.ffn_down.weight"),
                &[self.hidden_size, self.feed_forward_size],
            )?;
            if self.is_full_attention_layer(layer) {
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.attn_q.weight"),
                    &[self.attention_heads * self.head_dim * 2, self.hidden_size],
                )?;
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.attn_k.weight"),
                    &[self.kv_heads * self.head_dim, self.hidden_size],
                )?;
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.attn_v.weight"),
                    &[self.kv_heads * self.value_dim, self.hidden_size],
                )?;
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.attn_output.weight"),
                    &[self.hidden_size, self.attention_heads * self.value_dim],
                )?;
                require_f32_tensor(
                    model,
                    &format!("blk.{layer}.attn_q_norm.weight"),
                    &[self.head_dim],
                )?;
                require_f32_tensor(
                    model,
                    &format!("blk.{layer}.attn_k_norm.weight"),
                    &[self.head_dim],
                )?;
            } else {
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.attn_qkv.weight"),
                    &[self.ssm_inner_size * 3, self.hidden_size],
                )?;
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.attn_gate.weight"),
                    &[self.ssm_inner_size, self.hidden_size],
                )?;
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.ssm_beta.weight"),
                    &[self.ssm_group_count, self.hidden_size],
                )?;
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.ssm_alpha.weight"),
                    &[self.ssm_time_step_rank, self.hidden_size],
                )?;
                require_f32_tensor(
                    model,
                    &format!("blk.{layer}.ssm_conv1d.weight"),
                    &[self.ssm_inner_size * 3, 4],
                )?;
                require_f32_tensor(
                    model,
                    &format!("blk.{layer}.ssm_a"),
                    &[self.ssm_group_count],
                )?;
                require_f32_tensor(
                    model,
                    &format!("blk.{layer}.ssm_dt.bias"),
                    &[self.ssm_time_step_rank],
                )?;
                require_f32_tensor(
                    model,
                    &format!("blk.{layer}.ssm_norm.weight"),
                    &[self.ssm_inner_size / self.ssm_group_count],
                )?;
                require_quant_tensor(
                    model,
                    &format!("blk.{layer}.ssm_out.weight"),
                    &[self.hidden_size, self.ssm_inner_size],
                )?;
            }
        }
        Ok(())
    }

    pub fn is_full_attention_layer(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attention_interval == 0
    }
}

fn metadata_str<'a>(model: &'a GgufModel, key: &str) -> Result<&'a str> {
    model
        .metadata()
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Gguf(format!("metadata key `{key}` is not a string")))
}

fn metadata_usize(model: &GgufModel, key: &str) -> Result<usize> {
    model
        .metadata()
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| Error::Gguf(format!("metadata key `{key}` is not an integer")))
}

fn tensor_outer_dim(model: &GgufModel, name: &str) -> Result<usize> {
    let info = tensor_info(model, name)?;
    info.shape
        .first()
        .copied()
        .ok_or_else(|| Error::Gguf(format!("tensor `{name}` has empty shape")))
}

fn tensor_info<'a>(model: &'a GgufModel, name: &str) -> Result<&'a TensorInfo> {
    model
        .tensor_infos()
        .get(name)
        .ok_or_else(|| Error::Gguf(format!("missing tensor `{name}`")))
}

fn require_f32_tensor(model: &GgufModel, name: &str, expected_shape: &[usize]) -> Result<()> {
    require_tensor(
        model,
        name,
        expected_shape,
        TensorDType::Exact(GgmlDType::F32),
    )
}

fn require_quant_tensor(model: &GgufModel, name: &str, expected_shape: &[usize]) -> Result<()> {
    require_tensor(model, name, expected_shape, TensorDType::Qwen35Q4KmSet)
}

#[derive(Debug, Clone, Copy)]
enum TensorDType {
    Exact(GgmlDType),
    Qwen35Q4KmSet,
}

fn require_tensor(
    model: &GgufModel,
    name: &str,
    expected_shape: &[usize],
    expected_dtype: TensorDType,
) -> Result<()> {
    let info = tensor_info(model, name)?;
    if info.shape != expected_shape {
        return Err(Error::Gguf(format!(
            "tensor `{name}` shape {:?}, expected {:?}",
            info.shape, expected_shape
        )));
    }
    match expected_dtype {
        TensorDType::Exact(dtype) if info.dtype != dtype => {
            return Err(Error::Gguf(format!(
                "tensor `{name}` dtype {}, expected {}",
                info.dtype, dtype
            )));
        }
        TensorDType::Qwen35Q4KmSet if !is_qwen35_q4km_dtype(info.dtype) => {
            return Err(Error::Gguf(format!(
                "tensor `{name}` dtype {}, expected one of Q4_K/Q5_K/Q6_K/Q8_0/Q5_0 for qwen35 q4_k_m inference",
                info.dtype
            )));
        }
        _ => {}
    }
    Ok(())
}

fn is_qwen35_q4km_dtype(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8_0 | GgmlDType::Q5_0
    )
}

fn require_eq<T>(what: &str, got: T, expected: T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if got == expected {
        Ok(())
    } else {
        Err(Error::Gguf(format!(
            "{what} mismatch: got {:?}, expected {:?}",
            got, expected
        )))
    }
}
