//! Quantized CPU matmul/embedding helpers for GGUF weights.
//!
//! The block layouts and dot-product formulas are adapted from
//! `candle-core 0.11.0::quantized::k_quants`, kept local so this crate has no
//! candle dependency. We keep GGUF weights quantized and quantize each f32
//! activation row to Q8 blocks before computing f32 outputs.

use half::f16;
use rayon::prelude::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

use crate::gguf::{GgufModel, TensorView};
use crate::quant::GgmlDType;
use crate::{Error, Result};

const QK5_0: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;

const Q5_0_SIZE: usize = 2 + 4 + QK5_0 / 2;
const Q8_0_SIZE: usize = 2 + QK8_0;
const Q4_K_SIZE: usize = 2 + 2 + 12 + QK_K / 2;
const Q5_K_SIZE: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
const Q6_K_SIZE: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

pub fn cpu_simd_backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        return x86_kernel_kind().name();
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return "neon";
    }
    #[allow(unreachable_code)]
    "scalar"
}

#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X86KernelKind {
    Scalar,
    Avx2,
}

#[cfg(target_arch = "x86_64")]
impl X86KernelKind {
    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn x86_kernel_kind() -> X86KernelKind {
    static KIND: OnceLock<X86KernelKind> = OnceLock::new();
    *KIND.get_or_init(|| {
        let requested = std::env::var("EMBED_NATIVE_CPU_SIMD")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase());
        match requested.as_deref() {
            Some("scalar" | "none" | "off" | "0") => return X86KernelKind::Scalar,
            Some("avx2") if is_x86_feature_detected!("avx2") => return X86KernelKind::Avx2,
            _ => {}
        }

        if is_x86_feature_detected!("avx2") {
            X86KernelKind::Avx2
        } else {
            X86KernelKind::Scalar
        }
    })
}

#[derive(Debug, Clone)]
pub struct QuantMatrix {
    name: String,
    rows: usize,
    cols: usize,
    storage: QuantStorage,
}

#[derive(Debug, Clone)]
enum QuantStorage {
    F32(Vec<f32>),
    Q4K {
        blocks: Vec<BlockQ4K>,
        #[cfg(target_arch = "aarch64")]
        x8: Option<Vec<BlockQ4Kx8>>,
    },
    Q5K(Vec<BlockQ5K>),
    Q6K(Vec<BlockQ6K>),
    Q8_0(Vec<BlockQ8_0>),
    Q5_0(Vec<BlockQ5_0>),
}

#[derive(Debug, Clone)]
struct BlockQ4K {
    d: f16,
    dmin: f16,
    scales: [u8; 12],
    qs: [u8; QK_K / 2],
}

#[derive(Debug, Clone)]
struct BlockQ5K {
    d: f16,
    dmin: f16,
    scales: [u8; 12],
    qh: [u8; QK_K / 8],
    qs: [u8; QK_K / 2],
}

#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BlockQ4Kx8 {
    d: [f16; 8],
    dmin: [f16; 8],
    scales: [u8; 96],
    qs: [u8; 1024],
}

#[derive(Debug, Clone)]
struct BlockQ6K {
    ql: [u8; QK_K / 2],
    qh: [u8; QK_K / 4],
    scales: [i8; QK_K / 16],
    d: f16,
}

#[derive(Debug, Clone)]
struct BlockQ8K {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

#[derive(Debug, Clone)]
struct BlockQ8_0 {
    d: f16,
    qs: [i8; QK8_0],
}

#[derive(Debug, Clone)]
struct BlockQ5_0 {
    d: f16,
    qh: u32,
    qs: [u8; QK5_0 / 2],
}

impl QuantMatrix {
    pub fn from_model(model: &GgufModel, name: &str) -> Result<Self> {
        let tensor = model.tensor(name)?;
        Self::from_tensor_impl(name, tensor, false)
    }

    /// Qwen CPU path: prepack Q4_K rows for the Apple `sdot` x8 kernel.
    /// Other consumers keep the compact GGUF layout unless explicitly opted in.
    pub fn from_model_q4_x8(model: &GgufModel, name: &str) -> Result<Self> {
        let tensor = model.tensor(name)?;
        Self::from_tensor_impl(name, tensor, true)
    }

    pub fn from_tensor(name: &str, tensor: TensorView<'_>) -> Result<Self> {
        Self::from_tensor_impl(name, tensor, false)
    }

    fn from_tensor_impl(name: &str, tensor: TensorView<'_>, force_q4_x8: bool) -> Result<Self> {
        if tensor.shape.len() != 2 {
            return Err(Error::InvalidGguf(format!(
                "matrix tensor '{name}' must be rank 2, got {:?}",
                tensor.shape
            )));
        }
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        let storage = match tensor.dtype {
            GgmlDType::F32 => QuantStorage::F32(tensor.to_f32()?),
            GgmlDType::Q4K => {
                let blocks = parse_q4k(tensor.raw_bytes)?;
                #[cfg(target_arch = "aarch64")]
                {
                    let use_x8 = match std::env::var("EMBED_NATIVE_Q4_X8_MODE").as_deref() {
                        Ok("all") => true,
                        Ok("ffn") => {
                            name.contains(".ffn_gate.weight") || name.contains(".ffn_up.weight")
                        }
                        Ok("attn_qk") => {
                            name.contains(".attn_q.weight") || name.contains(".attn_k.weight")
                        }
                        Ok("attn_o") => name.contains(".attn_output.weight"),
                        Ok("attn") => {
                            name.contains(".attn_q.weight")
                                || name.contains(".attn_k.weight")
                                || name.contains(".attn_output.weight")
                        }
                        _ => false,
                    };
                    let x8 = ((force_q4_x8 || use_x8)
                        && std::arch::is_aarch64_feature_detected!("dotprod")
                        && rows % 8 == 0)
                        .then(|| pack_to_q4kx8(&blocks, rows));
                    QuantStorage::Q4K { blocks, x8 }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    QuantStorage::Q4K { blocks }
                }
            }
            GgmlDType::Q5K => QuantStorage::Q5K(parse_q5k(tensor.raw_bytes)?),
            GgmlDType::Q6K => QuantStorage::Q6K(parse_q6k(tensor.raw_bytes)?),
            GgmlDType::Q8_0 => QuantStorage::Q8_0(parse_q8_0(tensor.raw_bytes)?),
            GgmlDType::Q5_0 => QuantStorage::Q5_0(parse_q5_0(tensor.raw_bytes)?),
            other => return Err(Error::UnsupportedDType(other)),
        };
        let blocks = storage.block_count();
        let expected_blocks = rows
            .checked_mul(cols / tensor.dtype.block_size())
            .ok_or_else(|| Error::InvalidGguf(format!("matrix '{name}' block count overflows")))?;
        if cols % tensor.dtype.block_size() != 0 || blocks != expected_blocks {
            return Err(Error::InvalidGguf(format!(
                "matrix '{name}' shape {:?} incompatible with dtype {}: parsed {blocks} blocks, expected {expected_blocks}",
                tensor.shape, tensor.dtype
            )));
        }
        Ok(Self {
            name: name.to_string(),
            rows,
            cols,
            storage,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn matmul(&self, lhs: &[f32], lhs_rows: usize) -> Result<Vec<f32>> {
        if lhs.len() != lhs_rows * self.cols {
            return Err(Error::InvalidGguf(format!(
                "{} matmul lhs len {}, expected {}x{}",
                self.name,
                lhs.len(),
                lhs_rows,
                self.cols
            )));
        }
        if std::env::var_os("EMBED_NATIVE_DEQUANT_MATMUL").is_some() {
            return self.matmul_dequant(lhs, lhs_rows);
        }
        let mut out = vec![0.0f32; lhs_rows * self.rows];
        match &self.storage {
            QuantStorage::F32(rhs) => {
                if lhs_rows == 1 {
                    out.par_chunks_mut(64)
                        .enumerate()
                        .for_each(|(chunk_idx, dst)| {
                            let first = chunk_idx * 64;
                            for (local, value) in dst.iter_mut().enumerate() {
                                let out_col = first + local;
                                let y = &rhs[out_col * self.cols..(out_col + 1) * self.cols];
                                *value = dot_f32(lhs, y);
                            }
                        });
                } else {
                    out.par_chunks_mut(self.rows)
                        .enumerate()
                        .for_each(|(row_idx, dst)| {
                            let x = &lhs[row_idx * self.cols..(row_idx + 1) * self.cols];
                            for (out_col, y) in rhs.chunks_exact(self.cols).enumerate() {
                                dst[out_col] = dot_f32(x, y);
                            }
                        });
                }
            }
            QuantStorage::Q4K {
                blocks: rhs,
                #[cfg(target_arch = "aarch64")]
                x8,
            } => {
                let row_blocks = self.cols / QK_K;
                if lhs_rows >= 4 {
                    #[cfg(target_arch = "aarch64")]
                    if let Some(x8) = x8 {
                        out = matmul_q4kx8_batched(x8, lhs, lhs_rows, self.rows, self.cols);
                    } else {
                        out = matmul_q4k_batched(rhs, lhs, lhs_rows, self.rows, self.cols);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        out = matmul_q4k_batched(rhs, lhs, lhs_rows, self.rows, self.cols);
                    }
                } else if lhs_rows == 1 {
                    let xq = quantize_q8k(lhs);
                    #[cfg(target_arch = "aarch64")]
                    let used_x8 = if let Some(x8) = x8 {
                        out.par_chunks_mut(64)
                            .enumerate()
                            .for_each(|(chunk_idx, dst)| {
                                let first_group = chunk_idx * 8;
                                for (local_group, values) in dst.chunks_exact_mut(8).enumerate() {
                                    let group = first_group + local_group;
                                    let y = &x8[group * row_blocks..(group + 1) * row_blocks];
                                    values.copy_from_slice(unsafe { &dot8_q4k_q8k_neon(y, &xq) });
                                }
                            });
                        true
                    } else {
                        false
                    };
                    #[cfg(not(target_arch = "aarch64"))]
                    let used_x8 = false;
                    if !used_x8 {
                        out.par_chunks_mut(64)
                            .enumerate()
                            .for_each(|(chunk_idx, dst)| {
                                let first = chunk_idx * 64;
                                let n_quad = dst.len() & !3;
                                for local in (0..n_quad).step_by(4) {
                                    let out_col = first + local;
                                    let y0 = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                    let y1 = &rhs
                                        [(out_col + 1) * row_blocks..(out_col + 2) * row_blocks];
                                    let y2 = &rhs
                                        [(out_col + 2) * row_blocks..(out_col + 3) * row_blocks];
                                    let y3 = &rhs
                                        [(out_col + 3) * row_blocks..(out_col + 4) * row_blocks];
                                    let (d0, d1, d2, d3) = dot4_q4k_q8k(y0, y1, y2, y3, &xq);
                                    dst[local] = d0;
                                    dst[local + 1] = d1;
                                    dst[local + 2] = d2;
                                    dst[local + 3] = d3;
                                }
                                for (local, value) in dst.iter_mut().enumerate().skip(n_quad) {
                                    let out_col = first + local;
                                    let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                    *value = dot_q4k_q8k(y, &xq);
                                }
                            });
                    }
                } else {
                    out.par_chunks_mut(self.rows)
                        .enumerate()
                        .for_each(|(row_idx, dst)| {
                            let x = &lhs[row_idx * self.cols..(row_idx + 1) * self.cols];
                            let xq = quantize_q8k(x);
                            #[cfg(target_arch = "aarch64")]
                            if let Some(x8) = x8 {
                                for group in 0..self.rows / 8 {
                                    let y = &x8[group * row_blocks..(group + 1) * row_blocks];
                                    let values = unsafe { dot8_q4k_q8k_neon(y, &xq) };
                                    dst[group * 8..group * 8 + 8].copy_from_slice(&values);
                                }
                                return;
                            }
                            let n_quad = self.rows & !3;
                            for out_col in (0..n_quad).step_by(4) {
                                let y0 = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                let y1 =
                                    &rhs[(out_col + 1) * row_blocks..(out_col + 2) * row_blocks];
                                let y2 =
                                    &rhs[(out_col + 2) * row_blocks..(out_col + 3) * row_blocks];
                                let y3 =
                                    &rhs[(out_col + 3) * row_blocks..(out_col + 4) * row_blocks];
                                let (d0, d1, d2, d3) = dot4_q4k_q8k(y0, y1, y2, y3, &xq);
                                dst[out_col] = d0;
                                dst[out_col + 1] = d1;
                                dst[out_col + 2] = d2;
                                dst[out_col + 3] = d3;
                            }
                            for out_col in n_quad..self.rows {
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                dst[out_col] = dot_q4k_q8k(y, &xq);
                            }
                        });
                }
            }
            QuantStorage::Q5K(rhs) => {
                let row_blocks = self.cols / QK_K;
                if lhs_rows >= 4 {
                    out = matmul_q5k_batched(rhs, lhs, lhs_rows, self.rows, self.cols);
                } else if lhs_rows == 1 {
                    let xq = quantize_q8k(lhs);
                    out.par_chunks_mut(64)
                        .enumerate()
                        .for_each(|(chunk_idx, dst)| {
                            let first = chunk_idx * 64;
                            for (local, value) in dst.iter_mut().enumerate() {
                                let out_col = first + local;
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                *value = dot_q5k_q8k(y, &xq);
                            }
                        });
                } else {
                    out.par_chunks_mut(self.rows)
                        .enumerate()
                        .for_each(|(row_idx, dst)| {
                            let x = &lhs[row_idx * self.cols..(row_idx + 1) * self.cols];
                            let xq = quantize_q8k(x);
                            for out_col in 0..self.rows {
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                dst[out_col] = dot_q5k_q8k(y, &xq);
                            }
                        });
                }
            }
            QuantStorage::Q6K(rhs) => {
                let row_blocks = self.cols / QK_K;
                if lhs_rows >= 4 {
                    out = matmul_q6k_batched(rhs, lhs, lhs_rows, self.rows, self.cols);
                } else if lhs_rows == 1 {
                    let xq = quantize_q8k(lhs);
                    out.par_chunks_mut(64)
                        .enumerate()
                        .for_each(|(chunk_idx, dst)| {
                            let first = chunk_idx * 64;
                            let n_quad = dst.len() & !3;
                            for local in (0..n_quad).step_by(4) {
                                let out_col = first + local;
                                let y0 = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                let y1 =
                                    &rhs[(out_col + 1) * row_blocks..(out_col + 2) * row_blocks];
                                let y2 =
                                    &rhs[(out_col + 2) * row_blocks..(out_col + 3) * row_blocks];
                                let y3 =
                                    &rhs[(out_col + 3) * row_blocks..(out_col + 4) * row_blocks];
                                let (d0, d1, d2, d3) = dot4_q6k_q8k(y0, y1, y2, y3, &xq);
                                dst[local] = d0;
                                dst[local + 1] = d1;
                                dst[local + 2] = d2;
                                dst[local + 3] = d3;
                            }
                            for (local, value) in dst.iter_mut().enumerate().skip(n_quad) {
                                let out_col = first + local;
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                *value = dot_q6k_q8k(y, &xq);
                            }
                        });
                } else {
                    out.par_chunks_mut(self.rows)
                        .enumerate()
                        .for_each(|(row_idx, dst)| {
                            let x = &lhs[row_idx * self.cols..(row_idx + 1) * self.cols];
                            let xq = quantize_q8k(x);
                            let n_quad = self.rows & !3;
                            for out_col in (0..n_quad).step_by(4) {
                                let y0 = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                let y1 =
                                    &rhs[(out_col + 1) * row_blocks..(out_col + 2) * row_blocks];
                                let y2 =
                                    &rhs[(out_col + 2) * row_blocks..(out_col + 3) * row_blocks];
                                let y3 =
                                    &rhs[(out_col + 3) * row_blocks..(out_col + 4) * row_blocks];
                                let (d0, d1, d2, d3) = dot4_q6k_q8k(y0, y1, y2, y3, &xq);
                                dst[out_col] = d0;
                                dst[out_col + 1] = d1;
                                dst[out_col + 2] = d2;
                                dst[out_col + 3] = d3;
                            }
                            for out_col in n_quad..self.rows {
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                dst[out_col] = dot_q6k_q8k(y, &xq);
                            }
                        });
                }
            }
            QuantStorage::Q8_0(rhs) => {
                let row_blocks = self.cols / QK8_0;
                if lhs_rows >= 4 {
                    out = matmul_q8_0_batched(rhs, lhs, lhs_rows, self.rows, self.cols);
                } else if lhs_rows == 1 {
                    let xq = quantize_q8_0(lhs);
                    out.par_chunks_mut(64)
                        .enumerate()
                        .for_each(|(chunk_idx, dst)| {
                            let first = chunk_idx * 64;
                            let n_quad = dst.len() & !3;
                            for local in (0..n_quad).step_by(4) {
                                let out_col = first + local;
                                let y0 = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                let y1 =
                                    &rhs[(out_col + 1) * row_blocks..(out_col + 2) * row_blocks];
                                let y2 =
                                    &rhs[(out_col + 2) * row_blocks..(out_col + 3) * row_blocks];
                                let y3 =
                                    &rhs[(out_col + 3) * row_blocks..(out_col + 4) * row_blocks];
                                let (d0, d1, d2, d3) = dot4_q8_0_q8_0(y0, y1, y2, y3, &xq);
                                dst[local] = d0;
                                dst[local + 1] = d1;
                                dst[local + 2] = d2;
                                dst[local + 3] = d3;
                            }
                            for (local, value) in dst.iter_mut().enumerate().skip(n_quad) {
                                let out_col = first + local;
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                *value = dot_q8_0_q8_0(y, &xq);
                            }
                        });
                } else {
                    out.par_chunks_mut(self.rows)
                        .enumerate()
                        .for_each(|(row_idx, dst)| {
                            let x = &lhs[row_idx * self.cols..(row_idx + 1) * self.cols];
                            let xq = quantize_q8_0(x);
                            let n_quad = self.rows & !3;
                            for out_col in (0..n_quad).step_by(4) {
                                let y0 = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                let y1 =
                                    &rhs[(out_col + 1) * row_blocks..(out_col + 2) * row_blocks];
                                let y2 =
                                    &rhs[(out_col + 2) * row_blocks..(out_col + 3) * row_blocks];
                                let y3 =
                                    &rhs[(out_col + 3) * row_blocks..(out_col + 4) * row_blocks];
                                let (d0, d1, d2, d3) = dot4_q8_0_q8_0(y0, y1, y2, y3, &xq);
                                dst[out_col] = d0;
                                dst[out_col + 1] = d1;
                                dst[out_col + 2] = d2;
                                dst[out_col + 3] = d3;
                            }
                            for out_col in n_quad..self.rows {
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                dst[out_col] = dot_q8_0_q8_0(y, &xq);
                            }
                        });
                }
            }
            QuantStorage::Q5_0(rhs) => {
                let row_blocks = self.cols / QK5_0;
                if lhs_rows == 1 {
                    let xq = quantize_q8_0(lhs);
                    out.par_chunks_mut(64)
                        .enumerate()
                        .for_each(|(chunk_idx, dst)| {
                            let first = chunk_idx * 64;
                            for (local, value) in dst.iter_mut().enumerate() {
                                let out_col = first + local;
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                *value = dot_q5_0_q8_0(y, &xq);
                            }
                        });
                } else {
                    out.par_chunks_mut(self.rows)
                        .enumerate()
                        .for_each(|(row_idx, dst)| {
                            let x = &lhs[row_idx * self.cols..(row_idx + 1) * self.cols];
                            let xq = quantize_q8_0(x);
                            for out_col in 0..self.rows {
                                let y = &rhs[out_col * row_blocks..(out_col + 1) * row_blocks];
                                dst[out_col] = dot_q5_0_q8_0(y, &xq);
                            }
                        });
                }
            }
        }
        Ok(out)
    }

    fn matmul_dequant(&self, lhs: &[f32], lhs_rows: usize) -> Result<Vec<f32>> {
        let mut rhs = vec![0.0f32; self.rows * self.cols];
        for row in 0..self.rows {
            self.dequantize_row(row, &mut rhs[row * self.cols..(row + 1) * self.cols])?;
        }
        let mut out = vec![0.0f32; lhs_rows * self.rows];
        out.par_chunks_mut(self.rows)
            .enumerate()
            .for_each(|(row_idx, dst)| {
                let x = &lhs[row_idx * self.cols..(row_idx + 1) * self.cols];
                for (out_col, y) in rhs.chunks_exact(self.cols).enumerate() {
                    dst[out_col] = dot_f32(x, y);
                }
            });
        Ok(out)
    }

    pub fn embedding_rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; ids.len() * self.cols];
        for (out_row, &id) in ids.iter().enumerate() {
            let row = id as usize;
            if row >= self.rows {
                return Err(Error::InvalidGguf(format!(
                    "{} embedding id {row} out of range for {} rows",
                    self.name, self.rows
                )));
            }
            self.dequantize_row(
                row,
                &mut out[out_row * self.cols..(out_row + 1) * self.cols],
            )?;
        }
        Ok(out)
    }

    fn dequantize_row(&self, row: usize, dst: &mut [f32]) -> Result<()> {
        if dst.len() != self.cols {
            return Err(Error::InvalidGguf(format!(
                "{} row dst len {}, expected {}",
                self.name,
                dst.len(),
                self.cols
            )));
        }
        match &self.storage {
            QuantStorage::F32(values) => {
                dst.copy_from_slice(&values[row * self.cols..(row + 1) * self.cols]);
            }
            QuantStorage::Q4K { blocks, .. } => {
                let row_blocks = self.cols / QK_K;
                dequantize_q4k_row(&blocks[row * row_blocks..(row + 1) * row_blocks], dst);
            }
            QuantStorage::Q5K(blocks) => {
                let row_blocks = self.cols / QK_K;
                dequantize_q5k_row(&blocks[row * row_blocks..(row + 1) * row_blocks], dst);
            }
            QuantStorage::Q6K(blocks) => {
                let row_blocks = self.cols / QK_K;
                dequantize_q6k_row(&blocks[row * row_blocks..(row + 1) * row_blocks], dst);
            }
            QuantStorage::Q8_0(blocks) => {
                let row_blocks = self.cols / QK8_0;
                dequantize_q8_0_row(&blocks[row * row_blocks..(row + 1) * row_blocks], dst);
            }
            QuantStorage::Q5_0(blocks) => {
                let row_blocks = self.cols / QK5_0;
                dequantize_q5_0_row(&blocks[row * row_blocks..(row + 1) * row_blocks], dst);
            }
        }
        Ok(())
    }
}

fn transpose_batched_output(transposed: &[f32], lhs_rows: usize, matrix_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; lhs_rows * matrix_rows];
    out.par_chunks_mut(matrix_rows)
        .enumerate()
        .for_each(|(input_row, dst)| {
            for output_row in 0..matrix_rows {
                dst[output_row] = transposed[output_row * lhs_rows + input_row];
            }
        });
    out
}

fn matmul_q4k_batched(
    rhs: &[BlockQ4K],
    lhs: &[f32],
    lhs_rows: usize,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Vec<f32> {
    let row_blocks = matrix_cols / QK_K;
    let activations = lhs
        .par_chunks(matrix_cols)
        .map(quantize_q8k)
        .collect::<Vec<_>>();
    let mut transposed = vec![0.0f32; matrix_rows * lhs_rows];
    let chunk_values = 64 * lhs_rows;
    transposed
        .par_chunks_mut(chunk_values)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let first = chunk_idx * 64;
            let output_rows = chunk.len() / lhs_rows;
            let n_quad = output_rows & !3;
            for local in (0..n_quad).step_by(4) {
                let output_row = first + local;
                let y0 = &rhs[output_row * row_blocks..(output_row + 1) * row_blocks];
                let y1 = &rhs[(output_row + 1) * row_blocks..(output_row + 2) * row_blocks];
                let y2 = &rhs[(output_row + 2) * row_blocks..(output_row + 3) * row_blocks];
                let y3 = &rhs[(output_row + 3) * row_blocks..(output_row + 4) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    let (d0, d1, d2, d3) = dot4_q4k_q8k(y0, y1, y2, y3, xq);
                    chunk[local * lhs_rows + input_row] = d0;
                    chunk[(local + 1) * lhs_rows + input_row] = d1;
                    chunk[(local + 2) * lhs_rows + input_row] = d2;
                    chunk[(local + 3) * lhs_rows + input_row] = d3;
                }
            }
            for local in n_quad..output_rows {
                let output_row = first + local;
                let y = &rhs[output_row * row_blocks..(output_row + 1) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    chunk[local * lhs_rows + input_row] = dot_q4k_q8k(y, xq);
                }
            }
        });
    transpose_batched_output(&transposed, lhs_rows, matrix_rows)
}

#[cfg(target_arch = "aarch64")]
fn matmul_q4kx8_batched(
    rhs: &[BlockQ4Kx8],
    lhs: &[f32],
    lhs_rows: usize,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Vec<f32> {
    let row_blocks = matrix_cols / QK_K;
    let activations = lhs
        .par_chunks(matrix_cols)
        .map(quantize_q8k)
        .collect::<Vec<_>>();
    let mut transposed = vec![0.0f32; matrix_rows * lhs_rows];
    let chunk_values = 64 * lhs_rows;
    transposed
        .par_chunks_mut(chunk_values)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let first_group = chunk_idx * 8;
            for (local_group, group_dst) in chunk.chunks_exact_mut(8 * lhs_rows).enumerate() {
                let group = first_group + local_group;
                let weights = &rhs[group * row_blocks..(group + 1) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    let values = unsafe { dot8_q4k_q8k_neon(weights, xq) };
                    for output_lane in 0..8 {
                        group_dst[output_lane * lhs_rows + input_row] = values[output_lane];
                    }
                }
            }
        });
    transpose_batched_output(&transposed, lhs_rows, matrix_rows)
}

fn matmul_q6k_batched(
    rhs: &[BlockQ6K],
    lhs: &[f32],
    lhs_rows: usize,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Vec<f32> {
    let row_blocks = matrix_cols / QK_K;
    let activations = lhs
        .par_chunks(matrix_cols)
        .map(quantize_q8k)
        .collect::<Vec<_>>();
    let mut transposed = vec![0.0f32; matrix_rows * lhs_rows];
    let chunk_values = 64 * lhs_rows;
    transposed
        .par_chunks_mut(chunk_values)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let first = chunk_idx * 64;
            let output_rows = chunk.len() / lhs_rows;
            let n_quad = output_rows & !3;
            for local in (0..n_quad).step_by(4) {
                let output_row = first + local;
                let y0 = &rhs[output_row * row_blocks..(output_row + 1) * row_blocks];
                let y1 = &rhs[(output_row + 1) * row_blocks..(output_row + 2) * row_blocks];
                let y2 = &rhs[(output_row + 2) * row_blocks..(output_row + 3) * row_blocks];
                let y3 = &rhs[(output_row + 3) * row_blocks..(output_row + 4) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    let (d0, d1, d2, d3) = dot4_q6k_q8k(y0, y1, y2, y3, xq);
                    chunk[local * lhs_rows + input_row] = d0;
                    chunk[(local + 1) * lhs_rows + input_row] = d1;
                    chunk[(local + 2) * lhs_rows + input_row] = d2;
                    chunk[(local + 3) * lhs_rows + input_row] = d3;
                }
            }
            for local in n_quad..output_rows {
                let output_row = first + local;
                let y = &rhs[output_row * row_blocks..(output_row + 1) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    chunk[local * lhs_rows + input_row] = dot_q6k_q8k(y, xq);
                }
            }
        });
    transpose_batched_output(&transposed, lhs_rows, matrix_rows)
}

fn matmul_q5k_batched(
    rhs: &[BlockQ5K],
    lhs: &[f32],
    lhs_rows: usize,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Vec<f32> {
    let row_blocks = matrix_cols / QK_K;
    let activations = lhs
        .par_chunks(matrix_cols)
        .map(quantize_q8k)
        .collect::<Vec<_>>();
    let mut transposed = vec![0.0f32; matrix_rows * lhs_rows];
    let chunk_values = 64 * lhs_rows;
    transposed
        .par_chunks_mut(chunk_values)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let first = chunk_idx * 64;
            let output_rows = chunk.len() / lhs_rows;
            for local in 0..output_rows {
                let output_row = first + local;
                let y = &rhs[output_row * row_blocks..(output_row + 1) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    chunk[local * lhs_rows + input_row] = dot_q5k_q8k(y, xq);
                }
            }
        });
    transpose_batched_output(&transposed, lhs_rows, matrix_rows)
}

fn matmul_q8_0_batched(
    rhs: &[BlockQ8_0],
    lhs: &[f32],
    lhs_rows: usize,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Vec<f32> {
    let row_blocks = matrix_cols / QK8_0;
    let activations = lhs
        .par_chunks(matrix_cols)
        .map(quantize_q8_0)
        .collect::<Vec<_>>();
    let mut transposed = vec![0.0f32; matrix_rows * lhs_rows];
    let chunk_values = 64 * lhs_rows;
    transposed
        .par_chunks_mut(chunk_values)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let first = chunk_idx * 64;
            let output_rows = chunk.len() / lhs_rows;
            let n_quad = output_rows & !3;
            for local in (0..n_quad).step_by(4) {
                let output_row = first + local;
                let y0 = &rhs[output_row * row_blocks..(output_row + 1) * row_blocks];
                let y1 = &rhs[(output_row + 1) * row_blocks..(output_row + 2) * row_blocks];
                let y2 = &rhs[(output_row + 2) * row_blocks..(output_row + 3) * row_blocks];
                let y3 = &rhs[(output_row + 3) * row_blocks..(output_row + 4) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    let (d0, d1, d2, d3) = dot4_q8_0_q8_0(y0, y1, y2, y3, xq);
                    chunk[local * lhs_rows + input_row] = d0;
                    chunk[(local + 1) * lhs_rows + input_row] = d1;
                    chunk[(local + 2) * lhs_rows + input_row] = d2;
                    chunk[(local + 3) * lhs_rows + input_row] = d3;
                }
            }
            for local in n_quad..output_rows {
                let output_row = first + local;
                let y = &rhs[output_row * row_blocks..(output_row + 1) * row_blocks];
                for (input_row, xq) in activations.iter().enumerate() {
                    chunk[local * lhs_rows + input_row] = dot_q8_0_q8_0(y, xq);
                }
            }
        });
    transpose_batched_output(&transposed, lhs_rows, matrix_rows)
}

impl QuantStorage {
    fn block_count(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Q4K { blocks, .. } => blocks.len(),
            Self::Q5K(v) => v.len(),
            Self::Q6K(v) => v.len(),
            Self::Q8_0(v) => v.len(),
            Self::Q5_0(v) => v.len(),
        }
    }
}

fn parse_q4k(raw: &[u8]) -> Result<Vec<BlockQ4K>> {
    if raw.len() % Q4_K_SIZE != 0 {
        return Err(Error::InvalidGguf(format!(
            "Q4_K raw len {} is not divisible by {Q4_K_SIZE}",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / Q4_K_SIZE);
    for (idx, block) in raw.chunks_exact(Q4_K_SIZE).enumerate() {
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&block[4..16]);
        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&block[16..144]);
        let d = read_f16(&block[0..2]);
        let dmin = read_f16(&block[2..4]);
        validate_f16_scale("Q4_K.d", idx, d)?;
        validate_f16_scale("Q4_K.dmin", idx, dmin)?;
        out.push(BlockQ4K {
            d,
            dmin,
            scales,
            qs,
        });
    }
    Ok(out)
}

#[cfg(target_arch = "aarch64")]
fn pack_to_q4kx8(blocks: &[BlockQ4K], n: usize) -> Vec<BlockQ4Kx8> {
    if n == 0 || n % 8 != 0 {
        return Vec::new();
    }
    let k_blocks = blocks.len() / n;
    let n_groups = n / 8;
    let mut packed = Vec::with_capacity(n_groups * k_blocks);
    for g in 0..n_groups {
        for b in 0..k_blocks {
            let mut p = BlockQ4Kx8 {
                d: [f16::ZERO; 8],
                dmin: [f16::ZERO; 8],
                scales: [0; 96],
                qs: [0; 1024],
            };
            let src: [&BlockQ4K; 8] = std::array::from_fn(|i| &blocks[(g * 8 + i) * k_blocks + b]);
            for (i, s) in src.iter().enumerate() {
                p.d[i] = s.d;
                p.dmin[i] = s.dmin;
            }
            for i in 0..128usize {
                let col = i % 8;
                let off = (i / 8) * 8;
                p.qs[i * 8..i * 8 + 8].copy_from_slice(&src[col].qs[off..off + 8]);
            }
            for i in 0..4usize {
                let mut s = [0u8; 8];
                let mut m = [0u8; 8];
                for j in 0..8 {
                    s[j] = src[j].scales[i] & 63;
                    m[j] = src[j].scales[i + 4] & 63;
                }
                let b12 = i * 12;
                p.scales[b12] = (s[0] & 63) + ((s[4] & 48) << 2);
                p.scales[b12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                p.scales[b12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                p.scales[b12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                p.scales[b12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                p.scales[b12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                p.scales[b12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                p.scales[b12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                p.scales[b12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                p.scales[b12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                p.scales[b12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                p.scales[b12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }
            for i in 0..4usize {
                let mut s = [0u8; 8];
                let mut m = [0u8; 8];
                for j in 0..8 {
                    s[j] = ((src[j].scales[i] & 192) >> 2) | (src[j].scales[i + 8] & 15);
                    m[j] =
                        ((src[j].scales[i + 4] & 192) >> 2) | ((src[j].scales[i + 8] & 240) >> 4);
                }
                let b12 = i * 12 + 48;
                p.scales[b12] = (s[0] & 63) + ((s[4] & 48) << 2);
                p.scales[b12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                p.scales[b12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                p.scales[b12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                p.scales[b12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                p.scales[b12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                p.scales[b12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                p.scales[b12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                p.scales[b12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                p.scales[b12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                p.scales[b12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                p.scales[b12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }
            packed.push(p);
        }
    }
    packed
}

fn parse_q5k(raw: &[u8]) -> Result<Vec<BlockQ5K>> {
    if raw.len() % Q5_K_SIZE != 0 {
        return Err(Error::InvalidGguf(format!(
            "Q5_K raw len {} is not divisible by {Q5_K_SIZE}",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / Q5_K_SIZE);
    for (idx, block) in raw.chunks_exact(Q5_K_SIZE).enumerate() {
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&block[4..16]);
        let mut qh = [0u8; QK_K / 8];
        qh.copy_from_slice(&block[16..48]);
        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&block[48..176]);
        let d = read_f16(&block[0..2]);
        let dmin = read_f16(&block[2..4]);
        validate_f16_scale("Q5_K.d", idx, d)?;
        validate_f16_scale("Q5_K.dmin", idx, dmin)?;
        out.push(BlockQ5K {
            d,
            dmin,
            scales,
            qh,
            qs,
        });
    }
    Ok(out)
}

fn parse_q6k(raw: &[u8]) -> Result<Vec<BlockQ6K>> {
    if raw.len() % Q6_K_SIZE != 0 {
        return Err(Error::InvalidGguf(format!(
            "Q6_K raw len {} is not divisible by {Q6_K_SIZE}",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / Q6_K_SIZE);
    for (idx, block) in raw.chunks_exact(Q6_K_SIZE).enumerate() {
        let mut ql = [0u8; QK_K / 2];
        ql.copy_from_slice(&block[0..128]);
        let mut qh = [0u8; QK_K / 4];
        qh.copy_from_slice(&block[128..192]);
        let mut scales = [0i8; QK_K / 16];
        for (dst, &src) in scales.iter_mut().zip(&block[192..208]) {
            *dst = src as i8;
        }
        let d = read_f16(&block[208..210]);
        validate_f16_scale("Q6_K.d", idx, d)?;
        out.push(BlockQ6K { ql, qh, scales, d });
    }
    Ok(out)
}

fn parse_q8_0(raw: &[u8]) -> Result<Vec<BlockQ8_0>> {
    if raw.len() % Q8_0_SIZE != 0 {
        return Err(Error::InvalidGguf(format!(
            "Q8_0 raw len {} is not divisible by {Q8_0_SIZE}",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / Q8_0_SIZE);
    for (idx, block) in raw.chunks_exact(Q8_0_SIZE).enumerate() {
        let mut qs = [0i8; QK8_0];
        for (dst, &src) in qs.iter_mut().zip(&block[2..34]) {
            *dst = src as i8;
        }
        let d = read_f16(&block[0..2]);
        validate_f16_scale("Q8_0.d", idx, d)?;
        out.push(BlockQ8_0 { d, qs });
    }
    Ok(out)
}

fn parse_q5_0(raw: &[u8]) -> Result<Vec<BlockQ5_0>> {
    if raw.len() % Q5_0_SIZE != 0 {
        return Err(Error::InvalidGguf(format!(
            "Q5_0 raw len {} is not divisible by {Q5_0_SIZE}",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / Q5_0_SIZE);
    for (idx, block) in raw.chunks_exact(Q5_0_SIZE).enumerate() {
        let mut qs = [0u8; QK5_0 / 2];
        qs.copy_from_slice(&block[6..22]);
        let d = read_f16(&block[0..2]);
        validate_f16_scale("Q5_0.d", idx, d)?;
        out.push(BlockQ5_0 {
            d,
            qh: read_u32_le(&block[2..6]),
            qs,
        });
    }
    Ok(out)
}

fn quantize_q8k(xs: &[f32]) -> Vec<BlockQ8K> {
    let mut out = Vec::with_capacity(xs.len() / QK_K);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        out.resize_with(xs.len() / QK_K, || BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        });
        unsafe {
            quantize_q8k_neon(xs, &mut out);
        }
        return out;
    }
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        out.resize_with(xs.len() / QK_K, || BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        });
        unsafe {
            quantize_q8k_avx2(xs, &mut out);
        }
        return out;
    }

    for x in xs.chunks_exact(QK_K) {
        let mut y = BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in x {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        if amax != 0.0 {
            let iscale = -127.0f32 / max;
            for (q, &v) in y.qs.iter_mut().zip(x) {
                *q = (iscale * v).round().clamp(-128.0, 127.0) as i8;
            }
            for j in 0..QK_K / 16 {
                let mut sum = 0i32;
                for ii in 0..16 {
                    sum += y.qs[j * 16 + ii] as i32;
                }
                y.bsums[j] = sum as i16;
            }
            y.d = 1.0 / iscale;
        }
        out.push(y);
    }
    out
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn merge_signed_max(
    abs_a: float32x4_t,
    smax_a: float32x4_t,
    abs_b: float32x4_t,
    smax_b: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    (
        vmaxq_f32(abs_a, abs_b),
        vbslq_f32(vcgtq_f32(abs_b, abs_a), smax_b, smax_a),
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn quantize_q8k_neon(xs: &[f32], ys: &mut [BlockQ8K]) {
    for (chunk, y) in xs.chunks_exact(QK_K).zip(ys.iter_mut()) {
        let (mut vabs_max0, mut vsmax0) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut vabs_max1, mut vsmax1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut vabs_max2, mut vsmax2) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut vabs_max3, mut vsmax3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut p = chunk.as_ptr();
        for _ in 0..QK_K / 16 {
            let (v0, v1) = (vld1q_f32(p), vld1q_f32(p.add(4)));
            let (v2, v3) = (vld1q_f32(p.add(8)), vld1q_f32(p.add(12)));
            p = p.add(16);
            (vabs_max0, vsmax0) = merge_signed_max(vabs_max0, vsmax0, vabsq_f32(v0), v0);
            (vabs_max1, vsmax1) = merge_signed_max(vabs_max1, vsmax1, vabsq_f32(v1), v1);
            (vabs_max2, vsmax2) = merge_signed_max(vabs_max2, vsmax2, vabsq_f32(v2), v2);
            (vabs_max3, vsmax3) = merge_signed_max(vabs_max3, vsmax3, vabsq_f32(v3), v3);
        }
        let (abs01, smax01) = merge_signed_max(vabs_max0, vsmax0, vabs_max1, vsmax1);
        let (abs23, smax23) = merge_signed_max(vabs_max2, vsmax2, vabs_max3, vsmax3);
        let (abs_v, smax_v) = merge_signed_max(abs01, smax01, abs23, smax23);
        let mask_lohi = vcgt_f32(vget_high_f32(abs_v), vget_low_f32(abs_v));
        let abs_pair = vmax_f32(vget_low_f32(abs_v), vget_high_f32(abs_v));
        let smax_pair = vbsl_f32(mask_lohi, vget_high_f32(smax_v), vget_low_f32(smax_v));
        let max_signed = if vget_lane_f32::<1>(abs_pair) > vget_lane_f32::<0>(abs_pair) {
            vget_lane_f32::<1>(smax_pair)
        } else {
            vget_lane_f32::<0>(smax_pair)
        };

        if max_signed == 0.0 {
            y.d = 0.0;
            y.qs.fill(0);
            y.bsums.fill(0);
            continue;
        }

        let iscale = -127.0f32 / max_signed;
        let vscale = vdupq_n_f32(iscale);
        let mut out = y.qs.as_mut_ptr();
        let mut p = chunk.as_ptr();
        for j in 0..QK_K / 16 {
            let f0 = vmulq_f32(vld1q_f32(p), vscale);
            let f1 = vmulq_f32(vld1q_f32(p.add(4)), vscale);
            let f2 = vmulq_f32(vld1q_f32(p.add(8)), vscale);
            let f3 = vmulq_f32(vld1q_f32(p.add(12)), vscale);
            p = p.add(16);
            let s01 = vcombine_s16(
                vqmovn_s32(vcvtaq_s32_f32(f0)),
                vqmovn_s32(vcvtaq_s32_f32(f1)),
            );
            let s23 = vcombine_s16(
                vqmovn_s32(vcvtaq_s32_f32(f2)),
                vqmovn_s32(vcvtaq_s32_f32(f3)),
            );
            let q = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
            vst1q_s8(out, q);
            out = out.add(16);
            y.bsums[j] = vaddvq_s32(vpaddlq_s16(vpaddlq_s8(q))) as i16;
        }
        y.d = 1.0 / iscale;
    }
}

fn quantize_q8_0(xs: &[f32]) -> Vec<BlockQ8_0> {
    let mut out = Vec::with_capacity(xs.len() / QK8_0);
    for x in xs.chunks_exact(QK8_0) {
        let mut y = BlockQ8_0 {
            d: f16::ZERO,
            qs: [0; QK8_0],
        };
        let mut amax = 0.0f32;
        for &v in x {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y.d = f16::from_f32(d);
        for (q, &v) in y.qs.iter_mut().zip(x) {
            *q = (v * id).round() as i8;
        }
        out.push(y);
    }
    out
}

fn dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot_q4k_q8k_neon(xs, ys);
    }
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        unsafe {
            return dot_q4k_q8k_avx2(xs, ys);
        }
    }

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];
    let mut sumf = 0.0f32;

    for (x, y) in xs.iter().zip(ys) {
        aux32.fill(0);

        let mut a_offset = 0;
        let mut q4_offset = 0;
        for _ in 0..QK_K / 64 {
            for l in 0..32 {
                aux8[a_offset + l] = (x.qs[q4_offset + l] & 0x0f) as i8;
            }
            a_offset += 32;
            for l in 0..32 {
                aux8[a_offset + l] = (x.qs[q4_offset + l] >> 4) as i8;
            }
            a_offset += 32;
            q4_offset += 32;
        }

        let mut utmp = [0u32; 4];
        utmp[0] = read_u32_le(&x.scales[0..4]);
        utmp[1] = read_u32_le(&x.scales[4..8]);
        utmp[2] = read_u32_le(&x.scales[8..12]);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        write_u32_le_into(&utmp[0..2], &mut scales);
        write_u32_le_into(&utmp[2..4], &mut mins);

        let mut sumi = 0i32;
        for j in 0..QK_K / 16 {
            sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
        }

        let mut a_offset = 0;
        let mut q8_offset = 0;
        for scale in scales {
            let scale = scale as i32;
            for _ in 0..4 {
                for l in 0..8 {
                    aux16[l] = y.qs[q8_offset + l] as i16 * aux8[a_offset + l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8_offset += 8;
                a_offset += 8;
            }
        }

        let d = x.d.to_f32() * y.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        let dmin = x.dmin.to_f32() * y.d;
        sumf -= dmin * sumi as f32;
    }

    sumf + sums.iter().sum::<f32>()
}

fn dot_q5k_q8k(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot_q5k_q8k_neon(xs, ys);
    }

    let mut sum = 0.0f32;
    let mut values = [0.0f32; QK_K];
    for (x, y) in xs.iter().zip(ys) {
        dequantize_q5k_row(std::slice::from_ref(x), &mut values);
        for (value, quantized) in values.iter().zip(&y.qs) {
            sum += *value * (*quantized as f32 * y.d);
        }
    }
    sum
}

fn dot_q6k_q8k(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot_q6k_q8k_neon(xs, ys);
    }
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        unsafe {
            return dot_q6k_q8k_avx2(xs, ys);
        }
    }

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0.0f32; 8];

    for (x, y) in xs.iter().zip(ys) {
        aux32.fill(0.0);

        for j in (0..QK_K).step_by(128) {
            for l in 0..32 {
                aux8[j + l] =
                    (((x.ql[j / 2 + l] & 0x0f) | ((x.qh[j / 4 + l] & 3) << 4)) as i32 - 32) as i8;
                aux8[j + l + 32] =
                    (((x.ql[j / 2 + l + 32] & 0x0f) | (((x.qh[j / 4 + l] >> 2) & 3) << 4)) as i32
                        - 32) as i8;
                aux8[j + l + 64] = (((x.ql[j / 2 + l] >> 4) | (((x.qh[j / 4 + l] >> 4) & 3) << 4))
                    as i32
                    - 32) as i8;
                aux8[j + l + 96] =
                    (((x.ql[j / 2 + l + 32] >> 4) | (((x.qh[j / 4 + l] >> 6) & 3) << 4)) as i32
                        - 32) as i8;
            }
        }

        for (j, &scale) in x.scales.iter().enumerate() {
            let scale = scale as f32;
            let q8 = &y.qs[16 * j..];
            let a = &aux8[16 * j..];
            for l in 0..8 {
                aux16[l] = q8[l] as i16 * a[l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as f32;
            }
            let q8 = &q8[8..];
            let a = &a[8..];
            for l in 0..8 {
                aux16[l] = q8[l] as i16 * a[l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as f32;
            }
        }

        let d = x.d.to_f32() * y.d;
        for (sum, &a) in sums.iter_mut().zip(&aux32) {
            *sum += a * d;
        }
    }

    sums.iter().sum()
}

fn dot_q8_0_q8_0(xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    #[cfg(not(target_arch = "x86_64"))]
    if std::env::var_os("EMBED_NATIVE_Q8_SCALAR").is_some() {
        return dot_q8_0_q8_0_scalar(xs, ys);
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot_q8_0_q8_0_neon(xs, ys);
    }
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        unsafe {
            return dot_q8_0_q8_0_avx2(xs, ys);
        }
    }

    dot_q8_0_q8_0_scalar(xs, ys)
}

fn dot_q8_0_q8_0_scalar(xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    let mut sumf = 0.0f32;
    for (x, y) in xs.iter().zip(ys) {
        let mut sum_i = 0i32;
        for (&a, &b) in x.qs.iter().zip(&y.qs) {
            sum_i += a as i32 * b as i32;
        }
        sumf += sum_i as f32 * x.d.to_f32() * y.d.to_f32();
    }
    sumf
}

fn dot_q5_0_q8_0(xs: &[BlockQ5_0], ys: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        unsafe {
            return dot_q5_0_q8_0_avx2(xs, ys);
        }
    }

    let mut sumf = 0.0f32;
    for (x, y) in xs.iter().zip(ys) {
        let mut sumi = 0i32;
        for j in 0..QK5_0 / 2 {
            let xh_0 = (((x.qh & (1u32 << j)) >> j) << 4) as u8;
            let xh_1 = ((x.qh & (1u32 << (j + 16))) >> (j + 12)) as u8;
            let x0 = ((x.qs[j] & 0x0f) as i32 | xh_0 as i32) - 16;
            let x1 = ((x.qs[j] >> 4) as i32 | xh_1 as i32) - 16;
            sumi += (x0 * y.qs[j] as i32) + (x1 * y.qs[j + QK5_0 / 2] as i32);
        }
        sumf += sumi as f32 * x.d.to_f32() * y.d.to_f32();
    }
    sumf
}

fn dot4_q4k_q8k(
    xs0: &[BlockQ4K],
    xs1: &[BlockQ4K],
    xs2: &[BlockQ4K],
    xs3: &[BlockQ4K],
    ys: &[BlockQ8K],
) -> (f32, f32, f32, f32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot4_q4k_q8k_neon(xs0, xs1, xs2, xs3, ys);
    }
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        unsafe {
            return dot4_q4k_q8k_avx2(xs0, xs1, xs2, xs3, ys);
        }
    }

    (
        dot_q4k_q8k(xs0, ys),
        dot_q4k_q8k(xs1, ys),
        dot_q4k_q8k(xs2, ys),
        dot_q4k_q8k(xs3, ys),
    )
}

fn dot4_q6k_q8k(
    xs0: &[BlockQ6K],
    xs1: &[BlockQ6K],
    xs2: &[BlockQ6K],
    xs3: &[BlockQ6K],
    ys: &[BlockQ8K],
) -> (f32, f32, f32, f32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot4_q6k_q8k_neon(xs0, xs1, xs2, xs3, ys);
    }
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        unsafe {
            return (
                dot_q6k_q8k_avx2(xs0, ys),
                dot_q6k_q8k_avx2(xs1, ys),
                dot_q6k_q8k_avx2(xs2, ys),
                dot_q6k_q8k_avx2(xs3, ys),
            );
        }
    }

    (
        dot_q6k_q8k(xs0, ys),
        dot_q6k_q8k(xs1, ys),
        dot_q6k_q8k(xs2, ys),
        dot_q6k_q8k(xs3, ys),
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot4_q4k_q8k_neon(
    xs0: &[BlockQ4K],
    xs1: &[BlockQ4K],
    xs2: &[BlockQ4K],
    xs3: &[BlockQ4K],
    ys: &[BlockQ8K],
) -> (f32, f32, f32, f32) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    let mut utmp = [0u32; 4];
    let mut sc0 = [0u8; 16];
    let mut sc1 = [0u8; 16];
    let mut sc2 = [0u8; 16];
    let mut sc3 = [0u8; 16];

    macro_rules! decode_q4k_scales {
        ($x:ident, $sc:ident) => {{
            utmp[0] = read_u32_le(&$x.scales[0..4]);
            utmp[1] = read_u32_le(&$x.scales[4..8]);
            utmp[2] = read_u32_le(&$x.scales[8..12]);
            let mins_arr = [
                utmp[1] & KMASK1,
                ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
            ];
            let mins8 = vld1_u32(mins_arr.as_ptr());
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[0] &= KMASK1;
            write_u32_le_into(&utmp, &mut $sc);
            vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)))
        }};
    }

    macro_rules! dot_col {
        ($q4:ident, $sc:ident, $vsum1:ident, $vsum2:ident, $q8lo:ident, $q8hi:ident, $j:ident, $m4b:ident) => {{
            let bits = vld1q_u8_x2($q4);
            $q4 = $q4.add(32);
            let q4lo = int8x16x2_t(
                vreinterpretq_s8_u8(vandq_u8(bits.0, $m4b)),
                vreinterpretq_s8_u8(vandq_u8(bits.1, $m4b)),
            );
            $vsum1 = vmlaq_n_s32(
                $vsum1,
                neon_vdotq_s32_pair(q4lo.0, $q8lo.0, q4lo.1, $q8lo.1),
                $sc[2 * $j] as i32,
            );
            let q4hi = int8x16x2_t(
                vreinterpretq_s8_u8(vshrq_n_u8(bits.0, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(bits.1, 4)),
            );
            $vsum2 = vmlaq_n_s32(
                $vsum2,
                neon_vdotq_s32_pair(q4hi.0, $q8hi.0, q4hi.1, $q8hi.1),
                $sc[2 * $j + 1] as i32,
            );
        }};
    }

    let m4b = vdupq_n_u8(0x0f);

    for ((((x0, x1), x2), x3), y) in xs0.iter().zip(xs1).zip(xs2).zip(xs3).zip(ys) {
        let yd = y.d;
        let q8sums = vpaddq_s16(
            vld1q_s16(y.bsums.as_ptr()),
            vld1q_s16(y.bsums.as_ptr().add(8)),
        );

        let mins0 = decode_q4k_scales!(x0, sc0);
        let mins1 = decode_q4k_scales!(x1, sc1);
        let mins2 = decode_q4k_scales!(x2, sc2);
        let mins3 = decode_q4k_scales!(x3, sc3);

        let d0 = yd * x0.d.to_f32();
        let d1 = yd * x1.d.to_f32();
        let d2 = yd * x2.d.to_f32();
        let d3 = yd * x3.d.to_f32();

        macro_rules! min_correct {
            ($mins:ident, $dmin:expr, $sum:ident) => {{
                let prod = vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums), vget_low_s16($mins)),
                    vmull_s16(vget_high_s16(q8sums), vget_high_s16($mins)),
                );
                $sum -= $dmin as f64 * vaddvq_s32(prod) as f64;
            }};
        }
        min_correct!(mins0, yd * x0.dmin.to_f32(), sum0);
        min_correct!(mins1, yd * x1.dmin.to_f32(), sum1);
        min_correct!(mins2, yd * x2.dmin.to_f32(), sum2);
        min_correct!(mins3, yd * x3.dmin.to_f32(), sum3);

        let mut q4_0 = x0.qs.as_ptr();
        let mut q4_1 = x1.qs.as_ptr();
        let mut q4_2 = x2.qs.as_ptr();
        let mut q4_3 = x3.qs.as_ptr();
        let mut q8 = y.qs.as_ptr();

        let mut s0a = vdupq_n_s32(0);
        let mut s0b = vdupq_n_s32(0);
        let mut s1a = vdupq_n_s32(0);
        let mut s1b = vdupq_n_s32(0);
        let mut s2a = vdupq_n_s32(0);
        let mut s2b = vdupq_n_s32(0);
        let mut s3a = vdupq_n_s32(0);
        let mut s3b = vdupq_n_s32(0);

        for j in 0..QK_K / 64 {
            let q8lo = vld1q_s8_x2(q8);
            q8 = q8.add(32);
            let q8hi = vld1q_s8_x2(q8);
            q8 = q8.add(32);

            dot_col!(q4_0, sc0, s0a, s0b, q8lo, q8hi, j, m4b);
            dot_col!(q4_1, sc1, s1a, s1b, q8lo, q8hi, j, m4b);
            dot_col!(q4_2, sc2, s2a, s2b, q8lo, q8hi, j, m4b);
            dot_col!(q4_3, sc3, s3a, s3b, q8lo, q8hi, j, m4b);
        }

        sum0 += d0 as f64 * vaddvq_s32(vaddq_s32(s0a, s0b)) as f64;
        sum1 += d1 as f64 * vaddvq_s32(vaddq_s32(s1a, s1b)) as f64;
        sum2 += d2 as f64 * vaddvq_s32(vaddq_s32(s2a, s2b)) as f64;
        sum3 += d3 as f64 * vaddvq_s32(vaddq_s32(s3a, s3b)) as f64;
    }

    (sum0 as f32, sum1 as f32, sum2 as f32, sum3 as f32)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot4_q6k_q8k_neon(
    xs0: &[BlockQ6K],
    xs1: &[BlockQ6K],
    xs2: &[BlockQ6K],
    xs3: &[BlockQ6K],
    ys: &[BlockQ8K],
) -> (f32, f32, f32, f32) {
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let m4b = vdupq_n_u8(0x0f);
    let mone = vdupq_n_u8(3);

    for ((((x0, x1), x2), x3), y) in xs0.iter().zip(xs1).zip(xs2).zip(xs3).zip(ys) {
        let yd = y.d;
        let q8sums = vld1q_s16_x2(y.bsums.as_ptr());

        macro_rules! col_isum_mins {
            ($x:ident) => {{
                let scales_v = vld1q_s8($x.scales.as_ptr());
                let q6sc = int16x8x2_t(
                    vmovl_s8(vget_low_s8(scales_v)),
                    vmovl_s8(vget_high_s8(scales_v)),
                );
                let prod = vaddq_s32(
                    vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums.0), vget_low_s16(q6sc.0)),
                        vmull_s16(vget_high_s16(q8sums.0), vget_high_s16(q6sc.0)),
                    ),
                    vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums.1), vget_low_s16(q6sc.1)),
                        vmull_s16(vget_high_s16(q8sums.1), vget_high_s16(q6sc.1)),
                    ),
                );
                vaddvq_s32(prod)
            }};
        }

        let isum_mins0 = col_isum_mins!(x0);
        let isum_mins1 = col_isum_mins!(x1);
        let isum_mins2 = col_isum_mins!(x2);
        let isum_mins3 = col_isum_mins!(x3);

        let mut q6_0 = x0.ql.as_ptr();
        let mut qh_0 = x0.qh.as_ptr();
        let mut sc_0 = x0.scales.as_ptr();
        let mut q6_1 = x1.ql.as_ptr();
        let mut qh_1 = x1.qh.as_ptr();
        let mut sc_1 = x1.scales.as_ptr();
        let mut q6_2 = x2.ql.as_ptr();
        let mut qh_2 = x2.qh.as_ptr();
        let mut sc_2 = x2.scales.as_ptr();
        let mut q6_3 = x3.ql.as_ptr();
        let mut qh_3 = x3.qh.as_ptr();
        let mut sc_3 = x3.scales.as_ptr();
        let mut q8 = y.qs.as_ptr();

        let mut isum0 = 0i32;
        let mut isum1 = 0i32;
        let mut isum2 = 0i32;
        let mut isum3 = 0i32;

        for _ in 0..QK_K / 128 {
            let q8lo = vld1q_s8_x4(q8);
            q8 = q8.add(64);
            let q8hi = vld1q_s8_x4(q8);
            q8 = q8.add(64);

            macro_rules! process_col {
                ($q6:ident, $qh:ident, $sc:ident, $isum:ident) => {{
                    let qhb = vld1q_u8_x2($qh);
                    $qh = $qh.add(32);
                    let q6b = vld1q_u8_x4($q6);
                    $q6 = $q6.add(64);

                    let qh00 = vshlq_n_u8(vandq_u8(mone, qhb.0), 4);
                    let qh01 = vshlq_n_u8(vandq_u8(mone, qhb.1), 4);
                    let qh10 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.0, 2)), 4);
                    let qh11 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.1, 2)), 4);

                    let q6b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.0, m4b), qh00));
                    let q6b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.1, m4b), qh01));
                    let q6b2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.2, m4b), qh10));
                    let q6b3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.3, m4b), qh11));

                    let p0 = neon_vdotq_s32(q6b0, q8lo.0);
                    let p1 = neon_vdotq_s32(q6b1, q8lo.1);
                    $isum += vaddvq_s32(p0) * (*$sc as i32) + vaddvq_s32(p1) * (*$sc.add(1) as i32);
                    $sc = $sc.add(2);

                    let p2 = neon_vdotq_s32(q6b2, q8lo.2);
                    let p3 = neon_vdotq_s32(q6b3, q8lo.3);
                    $isum += vaddvq_s32(p2) * (*$sc as i32) + vaddvq_s32(p3) * (*$sc.add(1) as i32);
                    $sc = $sc.add(2);

                    let qh20 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.0, 4)), 4);
                    let qh21 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.1, 4)), 4);
                    let qh30 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.0, 6)), 4);
                    let qh31 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.1, 6)), 4);

                    let q6b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.0, 4), qh20));
                    let q6b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.1, 4), qh21));
                    let q6b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.2, 4), qh30));
                    let q6b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.3, 4), qh31));

                    let p0 = neon_vdotq_s32(q6b0, q8hi.0);
                    let p1 = neon_vdotq_s32(q6b1, q8hi.1);
                    $isum += vaddvq_s32(p0) * (*$sc as i32) + vaddvq_s32(p1) * (*$sc.add(1) as i32);
                    $sc = $sc.add(2);

                    let p2 = neon_vdotq_s32(q6b2, q8hi.2);
                    let p3 = neon_vdotq_s32(q6b3, q8hi.3);
                    $isum += vaddvq_s32(p2) * (*$sc as i32) + vaddvq_s32(p3) * (*$sc.add(1) as i32);
                    $sc = $sc.add(2);
                }};
            }

            process_col!(q6_0, qh_0, sc_0, isum0);
            process_col!(q6_1, qh_1, sc_1, isum1);
            process_col!(q6_2, qh_2, sc_2, isum2);
            process_col!(q6_3, qh_3, sc_3, isum3);
        }

        sum0 += x0.d.to_f32() as f64 * yd as f64 * ((isum0 - 32 * isum_mins0) as f64);
        sum1 += x1.d.to_f32() as f64 * yd as f64 * ((isum1 - 32 * isum_mins1) as f64);
        sum2 += x2.d.to_f32() as f64 * yd as f64 * ((isum2 - 32 * isum_mins2) as f64);
        sum3 += x3.d.to_f32() as f64 * yd as f64 * ((isum3 - 32 * isum_mins3) as f64);
    }

    (sum0 as f32, sum1 as f32, sum2 as f32, sum3 as f32)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_f16x4(ptr: *const f16) -> float32x4_t {
    let raw = vld1_u64(ptr as *const u64);
    let mut result: float32x4_t;
    core::arch::asm!(
        "fcvtl {out:v}.4s, {inp:v}.4h",
        inp = in(vreg) raw,
        out = out(vreg) result,
        options(nostack, nomem),
    );
    result
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_acc(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut out = acc;
    core::arch::asm!(
        "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
        out = inout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nostack, nomem),
    );
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn decode_q4kx8_scales(scales_in: *const u8) -> (int16x8_t, int16x8_t) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let sm0 = (scales_in as *const u32).read_unaligned();
    let sm1 = (scales_in.add(4) as *const u32).read_unaligned();
    let sm2 = (scales_in.add(8) as *const u32).read_unaligned();
    let mins_0_3 = sm1 & KMASK1;
    let mins_4_7 = ((sm2 >> 4) & KMASK2) | (((sm1 >> 6) & KMASK3) << 4);
    let out_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vcreate_u32(
        (mins_0_3 as u64) | ((mins_4_7 as u64) << 32),
    ))));
    let sc_0 = sm0 & KMASK1;
    let sc_1 = (sm2 & KMASK2) | (((sm0 >> 6) & KMASK3) << 4);
    let out_scales = vmovl_s8(vreinterpret_s8_u8(vreinterpret_u8_u32(vcreate_u32(
        (sc_0 as u64) | ((sc_1 as u64) << 32),
    ))));
    (out_mins, out_scales)
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot8_q4k_q8k_neon(xs: &[BlockQ4Kx8], ys: &[BlockQ8K]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    let mut vacc_0 = vdupq_n_f32(0.0);
    let mut vacc_1 = vdupq_n_f32(0.0);
    let m4b = vdupq_n_u8(0x0f);

    for (q4, q8) in xs.iter().zip(ys.iter()) {
        let q8d_v = vdupq_n_f32(q8.d);
        let sb_scale_0 = vmulq_f32(load_f16x4(q4.d.as_ptr()), q8d_v);
        let sb_scale_1 = vmulq_f32(load_f16x4(q4.d.as_ptr().add(4)), q8d_v);
        let sb_min_0 = vmulq_f32(load_f16x4(q4.dmin.as_ptr()), q8d_v);
        let sb_min_1 = vmulq_f32(load_f16x4(q4.dmin.as_ptr().add(4)), q8d_v);
        let bsums = vpaddq_s16(
            vld1q_s16(q8.bsums.as_ptr()),
            vld1q_s16(q8.bsums.as_ptr().add(8)),
        );
        let mut bias_0 = vdupq_n_s32(0);
        let mut bias_1 = vdupq_n_s32(0);

        macro_rules! process_sb {
            ($sb:literal) => {{
                let (mins0, sc0) = decode_q4kx8_scales(q4.scales.as_ptr().add($sb * 24));
                let (mins1, sc1) = decode_q4kx8_scales(q4.scales.as_ptr().add($sb * 24 + 12));
                let q8p = q8.qs.as_ptr().add($sb * 64);
                let q8_0 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p as *const i64));
                let q8_1 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(8) as *const i64));
                let q8_2 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(16) as *const i64));
                let q8_3 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(24) as *const i64));
                let q8_4 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(32) as *const i64));
                let q8_5 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(40) as *const i64));
                let q8_6 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(48) as *const i64));
                let q8_7 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(56) as *const i64));
                let q4p = q4.qs.as_ptr().add($sb * QK_K);
                let s0 = vld1q_u8_x2(q4p);
                let s0h = vld1q_u8_x2(q4p.add(32));
                let s1 = vld1q_u8_x2(q4p.add(64));
                let s1h = vld1q_u8_x2(q4p.add(96));
                let s2 = vld1q_u8_x2(q4p.add(128));
                let s2h = vld1q_u8_x2(q4p.add(160));
                let s3 = vld1q_u8_x2(q4p.add(192));
                let s3h = vld1q_u8_x2(q4p.add(224));
                let (b00, b10) = (s0.0, s0.1);
                let (b20, b30) = (s0h.0, s0h.1);
                let (b01, b11) = (s1.0, s1.1);
                let (b21, b31) = (s1h.0, s1h.1);
                let (b02, b12) = (s2.0, s2.1);
                let (b22, b32) = (s2h.0, s2h.1);
                let (b03, b13) = (s3.0, s3.1);
                let (b23, b33) = (s3h.0, s3h.1);

                let mut a0 = vdupq_n_s32(0);
                a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b00, m4b)), q8_0);
                a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b01, m4b)), q8_1);
                a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b02, m4b)), q8_2);
                a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b03, m4b)), q8_3);
                let mut h0 = vdupq_n_s32(0);
                h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b00, 4)), q8_4);
                h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b01, 4)), q8_5);
                h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b02, 4)), q8_6);
                h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b03, 4)), q8_7);
                let mut a1 = vdupq_n_s32(0);
                a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b10, m4b)), q8_0);
                a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b11, m4b)), q8_1);
                a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b12, m4b)), q8_2);
                a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b13, m4b)), q8_3);
                let mut h1 = vdupq_n_s32(0);
                h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b10, 4)), q8_4);
                h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b11, 4)), q8_5);
                h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b12, 4)), q8_6);
                h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b13, 4)), q8_7);
                let sumf_lo_03 =
                    vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_low_s16(sc0)), vpaddq_s32(a0, a1)));
                vacc_0 = vfmaq_f32(vacc_0, sb_scale_0, sumf_lo_03);
                let sumf_hi_03 =
                    vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_low_s16(sc1)), vpaddq_s32(h0, h1)));
                vacc_0 = vfmaq_f32(vacc_0, sb_scale_0, sumf_hi_03);

                let mut a2 = vdupq_n_s32(0);
                a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b20, m4b)), q8_0);
                a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b21, m4b)), q8_1);
                a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b22, m4b)), q8_2);
                a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b23, m4b)), q8_3);
                let mut h2 = vdupq_n_s32(0);
                h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b20, 4)), q8_4);
                h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b21, 4)), q8_5);
                h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b22, 4)), q8_6);
                h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b23, 4)), q8_7);
                let mut a3 = vdupq_n_s32(0);
                a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b30, m4b)), q8_0);
                a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b31, m4b)), q8_1);
                a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b32, m4b)), q8_2);
                a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b33, m4b)), q8_3);
                let mut h3 = vdupq_n_s32(0);
                h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b30, 4)), q8_4);
                h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b31, 4)), q8_5);
                h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b32, 4)), q8_6);
                h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b33, 4)), q8_7);
                let sumf_lo_47 =
                    vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_high_s16(sc0)), vpaddq_s32(a2, a3)));
                vacc_1 = vfmaq_f32(vacc_1, sb_scale_1, sumf_lo_47);
                let sumf_hi_47 =
                    vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_high_s16(sc1)), vpaddq_s32(h2, h3)));
                vacc_1 = vfmaq_f32(vacc_1, sb_scale_1, sumf_hi_47);

                let bl = vdup_n_s16(vgetq_lane_s16::<{ $sb * 2 }>(bsums));
                let bh = vdup_n_s16(vgetq_lane_s16::<{ $sb * 2 + 1 }>(bsums));
                bias_0 = vmlal_s16(bias_0, bl, vget_low_s16(mins0));
                bias_0 = vmlal_s16(bias_0, bh, vget_low_s16(mins1));
                bias_1 = vmlal_s16(bias_1, bl, vget_high_s16(mins0));
                bias_1 = vmlal_s16(bias_1, bh, vget_high_s16(mins1));
            }};
        }
        process_sb!(0);
        process_sb!(1);
        process_sb!(2);
        process_sb!(3);
        vacc_0 = vmlsq_f32(vacc_0, vcvtq_f32_s32(bias_0), sb_min_0);
        vacc_1 = vmlsq_f32(vacc_1, vcvtq_f32_s32(bias_1), sb_min_1);
    }
    vst1q_f32(out.as_mut_ptr(), vacc_0);
    vst1q_f32(out.as_mut_ptr().add(4), vacc_1);
    out
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn neon_vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        let mut acc: int32x4_t = vdupq_n_s32(0);
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(nostack, nomem),
        );
        return acc;
    }
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn neon_vdotq_s32_pair(
    a0: int8x16_t,
    b0: int8x16_t,
    a1: int8x16_t,
    b1: int8x16_t,
) -> int32x4_t {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        let mut acc: int32x4_t = vdupq_n_s32(0);
        core::arch::asm!(
            "sdot {acc:v}.4s, {a0:v}.16b, {b0:v}.16b",
            "sdot {acc:v}.4s, {a1:v}.16b, {b1:v}.16b",
            acc = inout(vreg) acc,
            a0 = in(vreg) a0,
            b0 = in(vreg) b0,
            a1 = in(vreg) a1,
            b1 = in(vreg) b1,
            options(nostack, nomem),
        );
        return acc;
    }
    let p0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
    let p1 = vmull_s8(vget_high_s8(a0), vget_high_s8(b0));
    let p2 = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
    let p3 = vmull_s8(vget_high_s8(a1), vget_high_s8(b1));
    vaddq_s32(
        vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
        vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)),
    )
}

fn dot4_q8_0_q8_0(
    xs0: &[BlockQ8_0],
    xs1: &[BlockQ8_0],
    xs2: &[BlockQ8_0],
    xs3: &[BlockQ8_0],
    ys: &[BlockQ8_0],
) -> (f32, f32, f32, f32) {
    #[cfg(not(target_arch = "x86_64"))]
    if std::env::var_os("EMBED_NATIVE_Q8_SCALAR").is_some() {
        return (
            dot_q8_0_q8_0_scalar(xs0, ys),
            dot_q8_0_q8_0_scalar(xs1, ys),
            dot_q8_0_q8_0_scalar(xs2, ys),
            dot_q8_0_q8_0_scalar(xs3, ys),
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot4_q8_0_q8_0_neon(xs0, xs1, xs2, xs3, ys);
    }
    #[cfg(target_arch = "x86_64")]
    if x86_kernel_kind() == X86KernelKind::Avx2 {
        unsafe {
            return (
                dot_q8_0_q8_0_avx2(xs0, ys),
                dot_q8_0_q8_0_avx2(xs1, ys),
                dot_q8_0_q8_0_avx2(xs2, ys),
                dot_q8_0_q8_0_avx2(xs3, ys),
            );
        }
    }

    (
        dot_q8_0_q8_0(xs0, ys),
        dot_q8_0_q8_0(xs1, ys),
        dot_q8_0_q8_0(xs2, ys),
        dot_q8_0_q8_0(xs3, ys),
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot4_q8_0_q8_0_neon(
    xs0: &[BlockQ8_0],
    xs1: &[BlockQ8_0],
    xs2: &[BlockQ8_0],
    xs3: &[BlockQ8_0],
    ys: &[BlockQ8_0],
) -> (f32, f32, f32, f32) {
    let mut sum0 = vdupq_n_f32(0.0f32);
    let mut sum1 = vdupq_n_f32(0.0f32);
    let mut sum2 = vdupq_n_f32(0.0f32);
    let mut sum3 = vdupq_n_f32(0.0f32);
    for ((((x0, x1), x2), x3), y) in xs0.iter().zip(xs1).zip(xs2).zip(xs3).zip(ys) {
        let y0 = vld1q_s8(y.qs.as_ptr());
        let y1 = vld1q_s8(y.qs.as_ptr().add(16));
        let yd = y.d.to_f32();
        let p0 = neon_vdotq_s32_pair(
            vld1q_s8(x0.qs.as_ptr()),
            y0,
            vld1q_s8(x0.qs.as_ptr().add(16)),
            y1,
        );
        let p1 = neon_vdotq_s32_pair(
            vld1q_s8(x1.qs.as_ptr()),
            y0,
            vld1q_s8(x1.qs.as_ptr().add(16)),
            y1,
        );
        let p2 = neon_vdotq_s32_pair(
            vld1q_s8(x2.qs.as_ptr()),
            y0,
            vld1q_s8(x2.qs.as_ptr().add(16)),
            y1,
        );
        let p3 = neon_vdotq_s32_pair(
            vld1q_s8(x3.qs.as_ptr()),
            y0,
            vld1q_s8(x3.qs.as_ptr().add(16)),
            y1,
        );
        sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(p0), x0.d.to_f32() * yd);
        sum1 = vmlaq_n_f32(sum1, vcvtq_f32_s32(p1), x1.d.to_f32() * yd);
        sum2 = vmlaq_n_f32(sum2, vcvtq_f32_s32(p2), x2.d.to_f32() * yd);
        sum3 = vmlaq_n_f32(sum3, vcvtq_f32_s32(p3), x3.d.to_f32() * yd);
    }
    (
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot_q8_0_q8_0_neon(xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    let mut sumv0 = vdupq_n_f32(0.0f32);
    for (x, y) in xs.iter().zip(ys) {
        let x0 = vld1q_s8(x.qs.as_ptr());
        let x1 = vld1q_s8(x.qs.as_ptr().add(16));
        let y0 = vld1q_s8(y.qs.as_ptr());
        let y1 = vld1q_s8(y.qs.as_ptr().add(16));
        let p0 = neon_vdotq_s32(x0, y0);
        let p1 = neon_vdotq_s32(x1, y1);
        sumv0 = vmlaq_n_f32(
            sumv0,
            vcvtq_f32_s32(vaddq_s32(p0, p1)),
            x.d.to_f32() * y.d.to_f32(),
        );
    }
    vaddvq_f32(sumv0)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot_q5k_q8k_neon(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4b = vdupq_n_u8(0x0f);
    let mone = vdupq_n_u8(1);
    let mtwo = vdupq_n_u8(2);
    let mut sum = 0.0f32;

    for (x, y) in xs.iter().zip(ys) {
        let d = y.d * x.d.to_f32();
        let dmin = y.d * x.dmin.to_f32();
        let q8sums = vpaddq_s16(
            vld1q_s16(y.bsums.as_ptr()),
            vld1q_s16(y.bsums.as_ptr().add(8)),
        );

        let mut packed = [0u32; 4];
        packed[0] = read_u32_le(&x.scales[0..4]);
        packed[1] = read_u32_le(&x.scales[4..8]);
        packed[2] = read_u32_le(&x.scales[8..12]);
        packed[3] = ((packed[2] >> 4) & KMASK2) | (((packed[1] >> 6) & KMASK3) << 4);
        let aux = packed[1] & KMASK1;
        packed[1] = (packed[2] & KMASK2) | (((packed[0] >> 6) & KMASK3) << 4);
        packed[2] = aux;
        packed[0] &= KMASK1;
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        write_u32_le_into(&packed[..2], &mut scales);
        write_u32_le_into(&packed[2..], &mut mins);
        let mut min_sum = 0i32;
        for i in 0..8 {
            min_sum += (y.bsums[2 * i] as i32 + y.bsums[2 * i + 1] as i32) * mins[i] as i32;
        }

        let mut qhbits = vld1q_u8_x2(x.qh.as_ptr());
        let mut q5 = x.qs.as_ptr();
        let mut q8 = y.qs.as_ptr();
        let mut scaled_sum = 0i32;
        for group in 0..QK_K / 64 {
            let q5bits = vld1q_u8_x2(q5);
            q5 = q5.add(32);
            let q8bytes = vld1q_s8_x4(q8);
            q8 = q8.add(64);

            let q5h0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
            let q5h1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
            let q5h2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
            let q5h3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);
            qhbits.0 = vshrq_n_u8(qhbits.0, 2);
            qhbits.1 = vshrq_n_u8(qhbits.1, 2);

            let q50 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.0, m4b), q5h0));
            let q51 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.1, m4b), q5h1));
            let q52 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.0, 4), q5h2));
            let q53 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.1, 4), q5h3));
            scaled_sum += vaddvq_s32(neon_vdotq_s32_pair(q50, q8bytes.0, q51, q8bytes.1))
                * scales[2 * group] as i32;
            scaled_sum += vaddvq_s32(neon_vdotq_s32_pair(q52, q8bytes.2, q53, q8bytes.3))
                * scales[2 * group + 1] as i32;
        }
        sum += d * scaled_sum as f32 - dmin * min_sum as f32;
    }
    sum
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot_q6k_q8k_neon(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f64;
    let m4b = vdupq_n_u8(0x0f);
    let mone = vdupq_n_u8(3);

    for (x, y) in xs.iter().zip(ys) {
        let d_all = x.d.to_f32();
        let mut q6 = x.ql.as_ptr();
        let mut qh = x.qh.as_ptr();
        let mut q8 = y.qs.as_ptr();
        let mut scale = x.scales.as_ptr();

        let q8sums = vld1q_s16_x2(y.bsums.as_ptr());
        let scales = vld1q_s8(scale);
        let q6scales = int16x8x2_t(
            vmovl_s8(vget_low_s8(scales)),
            vmovl_s8(vget_high_s8(scales)),
        );
        let prod = vaddq_s32(
            vaddq_s32(
                vmull_s16(vget_low_s16(q8sums.0), vget_low_s16(q6scales.0)),
                vmull_s16(vget_high_s16(q8sums.0), vget_high_s16(q6scales.0)),
            ),
            vaddq_s32(
                vmull_s16(vget_low_s16(q8sums.1), vget_low_s16(q6scales.1)),
                vmull_s16(vget_high_s16(q8sums.1), vget_high_s16(q6scales.1)),
            ),
        );
        let isum_mins = vaddvq_s32(prod);
        let mut isum = 0i32;

        for _ in 0..QK_K / 128 {
            let qhbits = vld1q_u8_x2(qh);
            qh = qh.add(32);
            let q6bits = vld1q_u8_x4(q6);
            q6 = q6.add(64);
            let q8bytes = vld1q_s8_x4(q8);
            q8 = q8.add(64);

            let q6h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
            let q6h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
            let q6h_2 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.0, 2)), 4);
            let q6h_3 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.1, 2)), 4);

            let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.0, m4b), q6h_0));
            let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.1, m4b), q6h_1));
            let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.2, m4b), q6h_2));
            let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.3, m4b), q6h_3));

            let p0 = neon_vdotq_s32(q6bytes_0, q8bytes.0);
            let p1 = neon_vdotq_s32(q6bytes_1, q8bytes.1);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
            scale = scale.add(2);

            let p2 = neon_vdotq_s32(q6bytes_2, q8bytes.2);
            let p3 = neon_vdotq_s32(q6bytes_3, q8bytes.3);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
            scale = scale.add(2);

            let q8bytes = vld1q_s8_x4(q8);
            q8 = q8.add(64);

            let q6h_0 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.0, 4)), 4);
            let q6h_1 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.1, 4)), 4);
            let q6h_2 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.0, 6)), 4);
            let q6h_3 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.1, 6)), 4);

            let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.0, 4), q6h_0));
            let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.1, 4), q6h_1));
            let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.2, 4), q6h_2));
            let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.3, 4), q6h_3));

            let p0 = neon_vdotq_s32(q6bytes_0, q8bytes.0);
            let p1 = neon_vdotq_s32(q6bytes_1, q8bytes.1);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
            scale = scale.add(2);

            let p2 = neon_vdotq_s32(q6bytes_2, q8bytes.2);
            let p3 = neon_vdotq_s32(q6bytes_3, q8bytes.3);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
            scale = scale.add(2);
        }
        sum += d_all as f64 * y.d as f64 * ((isum - 32 * isum_mins) as f64);
    }
    sum as f32
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot_q4k_q8k_neon(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sumf = 0.0f64;
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];
    let m4b = vdupq_n_u8(0x0f);

    for (x, y) in xs.iter().zip(ys) {
        let d = y.d * x.d.to_f32();
        let dmin = y.d * x.dmin.to_f32();

        let q8sums = vpaddq_s16(
            vld1q_s16(y.bsums.as_ptr()),
            vld1q_s16(y.bsums.as_ptr().add(8)),
        );

        utmp[0] = read_u32_le(&x.scales[0..4]);
        utmp[1] = read_u32_le(&x.scales[4..8]);
        utmp[2] = read_u32_le(&x.scales[8..12]);

        let mins_arr = [
            utmp[1] & KMASK1,
            ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
        ];
        let mins8 = vld1_u32(mins_arr.as_ptr());
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[0] &= KMASK1;

        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        sumf -= dmin as f64 * vaddvq_s32(prod) as f64;

        write_u32_le_into(&utmp, &mut scales);

        let mut q4 = x.qs.as_ptr();
        let mut q8 = y.qs.as_ptr();
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        for j in 0..QK_K / 64 {
            let q4bits = vld1q_u8_x2(q4);
            q4 = q4.add(32);
            let q8bytes = vld1q_s8_x2(q8);
            q8 = q8.add(32);
            let q4bytes = int8x16x2_t(
                vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b)),
                vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b)),
            );
            let p0 = neon_vdotq_s32(q4bytes.0, q8bytes.0);
            let p1 = neon_vdotq_s32(q4bytes.1, q8bytes.1);
            sumi1 += vaddvq_s32(vaddq_s32(p0, p1)) * scales[2 * j] as i32;

            let q8bytes = vld1q_s8_x2(q8);
            q8 = q8.add(32);
            let q4bytes = int8x16x2_t(
                vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)),
            );
            let p2 = neon_vdotq_s32(q4bytes.0, q8bytes.0);
            let p3 = neon_vdotq_s32(q4bytes.1, q8bytes.1);
            sumi2 += vaddvq_s32(vaddq_s32(p2, p3)) * scales[2 * j + 1] as i32;
        }
        sumf += d as f64 * (sumi1 + sumi2) as f64;
    }
    sumf as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_q8k_avx2(xs: &[f32], ys: &mut [BlockQ8K]) {
    let half_pos = _mm256_set1_ps(0.5);
    let half_neg = _mm256_set1_ps(-0.5);
    let min_q = _mm256_set1_ps(-128.0);
    let max_q = _mm256_set1_ps(127.0);

    for (chunk, y) in xs.chunks_exact(QK_K).zip(ys.iter_mut()) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in chunk {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            y.d = 0.0;
            y.qs.fill(0);
            y.bsums.fill(0);
            continue;
        }

        let iscale = -127.0f32 / max;
        let scale = _mm256_set1_ps(iscale);
        let mut tmp = [0i32; 8];
        for j in 0..QK_K / 8 {
            let values = _mm256_loadu_ps(chunk.as_ptr().add(j * 8));
            let scaled = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(values, scale), min_q), max_q);
            let adj = _mm256_blendv_ps(half_pos, half_neg, scaled);
            let rounded = _mm256_cvttps_epi32(_mm256_add_ps(scaled, adj));
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, rounded);
            for (lane, &q) in tmp.iter().enumerate() {
                y.qs[j * 8 + lane] = q as i8;
            }
        }

        for j in 0..QK_K / 16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += y.qs[j * 16 + ii] as i32;
            }
            y.bsums[j] = sum as i16;
        }
        y.d = 1.0 / iscale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4k_q8k_avx2(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    let m4b = _mm256_set1_epi8(0x0f);
    let mut sum = 0.0f64;

    for (x, y) in xs.iter().zip(ys) {
        let (scales, mins) = decode_q4k_scales_mins(&x.scales);
        let mut bias = 0i32;
        for j in 0..QK_K / 16 {
            bias += y.bsums[j] as i32 * mins[j / 2] as i32;
        }
        sum -= (x.dmin.to_f32() * y.d) as f64 * bias as f64;

        let mut isum = 0i32;
        for j in 0..QK_K / 64 {
            let q4bits = _mm256_loadu_si256(x.qs.as_ptr().add(j * 32) as *const __m256i);
            let q4lo = _mm256_and_si256(q4bits, m4b);
            let q8lo = _mm256_loadu_si256(y.qs.as_ptr().add(j * 64) as *const __m256i);
            isum += dot_u8_i8_32_avx2(q4lo, q8lo) * scales[2 * j] as i32;

            let q4hi = _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits), m4b);
            let q8hi = _mm256_loadu_si256(y.qs.as_ptr().add(j * 64 + 32) as *const __m256i);
            isum += dot_u8_i8_32_avx2(q4hi, q8hi) * scales[2 * j + 1] as i32;
        }
        sum += (x.d.to_f32() * y.d) as f64 * isum as f64;
    }

    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot4_q4k_q8k_avx2(
    xs0: &[BlockQ4K],
    xs1: &[BlockQ4K],
    xs2: &[BlockQ4K],
    xs3: &[BlockQ4K],
    ys: &[BlockQ8K],
) -> (f32, f32, f32, f32) {
    let m4b = _mm256_set1_epi8(0x0f);
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    for ((((x0, x1), x2), x3), y) in xs0.iter().zip(xs1).zip(xs2).zip(xs3).zip(ys) {
        let (sc0, mins0) = decode_q4k_scales_mins(&x0.scales);
        let (sc1, mins1) = decode_q4k_scales_mins(&x1.scales);
        let (sc2, mins2) = decode_q4k_scales_mins(&x2.scales);
        let (sc3, mins3) = decode_q4k_scales_mins(&x3.scales);

        macro_rules! apply_bias {
            ($x:ident, $mins:ident, $sum:ident) => {{
                let mut bias = 0i32;
                for j in 0..QK_K / 16 {
                    bias += y.bsums[j] as i32 * $mins[j / 2] as i32;
                }
                $sum -= ($x.dmin.to_f32() * y.d) as f64 * bias as f64;
            }};
        }
        apply_bias!(x0, mins0, sum0);
        apply_bias!(x1, mins1, sum1);
        apply_bias!(x2, mins2, sum2);
        apply_bias!(x3, mins3, sum3);

        let mut isum0 = 0i32;
        let mut isum1 = 0i32;
        let mut isum2 = 0i32;
        let mut isum3 = 0i32;
        for j in 0..QK_K / 64 {
            let q8lo = _mm256_loadu_si256(y.qs.as_ptr().add(j * 64) as *const __m256i);
            let q8hi = _mm256_loadu_si256(y.qs.as_ptr().add(j * 64 + 32) as *const __m256i);
            macro_rules! dot_col {
                ($x:ident, $sc:ident, $isum:ident) => {{
                    let q4bits = _mm256_loadu_si256($x.qs.as_ptr().add(j * 32) as *const __m256i);
                    let q4lo = _mm256_and_si256(q4bits, m4b);
                    $isum += dot_u8_i8_32_avx2(q4lo, q8lo) * $sc[2 * j] as i32;
                    let q4hi = _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits), m4b);
                    $isum += dot_u8_i8_32_avx2(q4hi, q8hi) * $sc[2 * j + 1] as i32;
                }};
            }
            dot_col!(x0, sc0, isum0);
            dot_col!(x1, sc1, isum1);
            dot_col!(x2, sc2, isum2);
            dot_col!(x3, sc3, isum3);
        }

        sum0 += (x0.d.to_f32() * y.d) as f64 * isum0 as f64;
        sum1 += (x1.d.to_f32() * y.d) as f64 * isum1 as f64;
        sum2 += (x2.d.to_f32() * y.d) as f64 * isum2 as f64;
        sum3 += (x3.d.to_f32() * y.d) as f64 * isum3 as f64;
    }

    (sum0 as f32, sum1 as f32, sum2 as f32, sum3 as f32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q6k_q8k_avx2(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f64;
    let mut aux8 = [0i8; QK_K];

    for (x, y) in xs.iter().zip(ys) {
        decode_q6k_i8(x, &mut aux8);
        let mut isum = 0i32;
        for j in 0..QK_K / 16 {
            let dot = dot_i8_i8_16_avx2(aux8.as_ptr().add(j * 16), y.qs.as_ptr().add(j * 16));
            isum += dot * x.scales[j] as i32;
        }
        sum += x.d.to_f32() as f64 * y.d as f64 * isum as f64;
    }

    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q5_0_q8_0_avx2(xs: &[BlockQ5_0], ys: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f64;
    let mut q5 = [0i8; QK5_0];
    for (x, y) in xs.iter().zip(ys) {
        for j in 0..QK5_0 / 2 {
            let xh_0 = (((x.qh & (1u32 << j)) >> j) << 4) as u8;
            let xh_1 = ((x.qh & (1u32 << (j + 16))) >> (j + 12)) as u8;
            q5[j] = ((x.qs[j] & 0x0f) as i32 | xh_0 as i32) as i8 - 16;
            q5[j + QK5_0 / 2] = ((x.qs[j] >> 4) as i32 | xh_1 as i32) as i8 - 16;
        }
        let dot = dot_i8_i8_32_avx2(q5.as_ptr(), y.qs.as_ptr());
        sum += dot as f64 * x.d.to_f32() as f64 * y.d.to_f32() as f64;
    }
    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q8_0_q8_0_avx2(xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f64;
    for (x, y) in xs.iter().zip(ys) {
        let dot = dot_i8_i8_32_avx2(x.qs.as_ptr(), y.qs.as_ptr());
        sum += dot as f64 * x.d.to_f32() as f64 * y.d.to_f32() as f64;
    }
    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_32_avx2(a_unsigned: __m256i, b_signed: __m256i) -> i32 {
    let products = _mm256_maddubs_epi16(a_unsigned, b_signed);
    let sums = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    hsum_i32x8_avx2(sums)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_16_avx2(a: *const i8, b: *const i8) -> i32 {
    let av = _mm_loadu_si128(a as *const __m128i);
    let bv = _mm_loadu_si128(b as *const __m128i);
    let a16 = _mm256_cvtepi8_epi16(av);
    let b16 = _mm256_cvtepi8_epi16(bv);
    hsum_i32x8_avx2(_mm256_madd_epi16(a16, b16))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_32_avx2(a: *const i8, b: *const i8) -> i32 {
    dot_i8_i8_16_avx2(a, b) + dot_i8_i8_16_avx2(a.add(16), b.add(16))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32x8_avx2(v: __m256i) -> i32 {
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
    tmp.iter().sum()
}

#[cfg(target_arch = "x86_64")]
fn decode_q4k_scales_mins(q: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp = [0u32; 4];
    utmp[0] = read_u32_le(&q[0..4]);
    utmp[1] = read_u32_le(&q[4..8]);
    utmp[2] = read_u32_le(&q[8..12]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    write_u32_le_into(&utmp[0..2], &mut scales);
    write_u32_le_into(&utmp[2..4], &mut mins);
    (scales, mins)
}

#[cfg(target_arch = "x86_64")]
fn decode_q6k_i8(x: &BlockQ6K, aux8: &mut [i8; QK_K]) {
    for j in (0..QK_K).step_by(128) {
        for l in 0..32 {
            aux8[j + l] =
                (((x.ql[j / 2 + l] & 0x0f) | ((x.qh[j / 4 + l] & 3) << 4)) as i32 - 32) as i8;
            aux8[j + l + 32] = (((x.ql[j / 2 + l + 32] & 0x0f)
                | (((x.qh[j / 4 + l] >> 2) & 3) << 4)) as i32
                - 32) as i8;
            aux8[j + l + 64] =
                (((x.ql[j / 2 + l] >> 4) | (((x.qh[j / 4 + l] >> 4) & 3) << 4)) as i32 - 32) as i8;
            aux8[j + l + 96] = (((x.ql[j / 2 + l + 32] >> 4) | (((x.qh[j / 4 + l] >> 6) & 3) << 4))
                as i32
                - 32) as i8;
        }
    }
}

fn dequantize_q4k_row(blocks: &[BlockQ4K], dst: &mut [f32]) {
    for (block, y) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let mut scale_idx = 0;
        let mut out_idx = 0;
        for j in (0..QK_K).step_by(64) {
            let q = &block.qs[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(scale_idx, &block.scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(scale_idx + 1, &block.scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for &q in q {
                y[out_idx] = d1 * (q & 0x0f) as f32 - m1;
                out_idx += 1;
            }
            for &q in q {
                y[out_idx] = d2 * (q >> 4) as f32 - m2;
                out_idx += 1;
            }
            scale_idx += 2;
        }
    }
}

fn dequantize_q5k_row(blocks: &[BlockQ5K], dst: &mut [f32]) {
    for (block, y) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let mut scale_idx = 0;
        let mut qs_base = 0;
        let mut high_lo = 1u8;
        let mut high_hi = 2u8;
        for out_base in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(scale_idx, &block.scales);
            let (sc2, m2) = get_scale_min_k4(scale_idx + 1, &block.scales);
            let scale1 = d * sc1 as f32;
            let scale2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;
            for l in 0..32 {
                let q4 = block.qs[qs_base + l] & 0x0f;
                let high = if block.qh[l] & high_lo != 0 { 16 } else { 0 };
                y[out_base + l] = scale1 * (q4 as f32 + high as f32) - min1;
            }
            for l in 0..32 {
                let q4 = block.qs[qs_base + l] >> 4;
                let high = if block.qh[l] & high_hi != 0 { 16 } else { 0 };
                y[out_base + 32 + l] = scale2 * (q4 as f32 + high as f32) - min2;
            }
            scale_idx += 2;
            qs_base += 32;
            high_lo <<= 2;
            high_hi <<= 2;
        }
    }
}

fn dequantize_q6k_row(blocks: &[BlockQ6K], dst: &mut [f32]) {
    for (block, y) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
        let d = block.d.to_f32();
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &block.scales[8 * idx..];
            let ql = &block.ql[64 * idx..];
            let qh = &block.qh[32 * idx..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                y[n + l] = d * sc[is] as f32 * q1 as f32;
                y[n + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                y[n + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                y[n + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
    }
}

fn dequantize_q8_0_row(blocks: &[BlockQ8_0], dst: &mut [f32]) {
    for (block, y) in blocks.iter().zip(dst.chunks_exact_mut(QK8_0)) {
        let d = block.d.to_f32();
        for (dst, &src) in y.iter_mut().zip(&block.qs) {
            *dst = src as f32 * d;
        }
    }
}

fn dequantize_q5_0_row(blocks: &[BlockQ5_0], dst: &mut [f32]) {
    for (block, y) in blocks.iter().zip(dst.chunks_exact_mut(QK5_0)) {
        let d = block.d.to_f32();
        for j in 0..QK5_0 / 2 {
            let xh_0 = (((block.qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((block.qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((block.qs[j] & 0x0f) | xh_0) as i32 - 16;
            let x1 = ((block.qs[j] >> 4) | xh_1) as i32 - 16;
            y[j] = x0 as f32 * d;
            y[j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
}

fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn write_u32_le_into(src: &[u32], dst: &mut [u8]) {
    for (chunk, value) in dst.chunks_exact_mut(4).zip(src) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
}

fn dot_q5k_f32(blocks: &[BlockQ5K], input: &[f32]) -> f32 {
    let mut tmp = [0.0f32; QK_K];
    let mut acc = 0.0f32;
    for (block_idx, block) in blocks.iter().enumerate() {
        dequantize_q5k_row(std::slice::from_ref(block), &mut tmp);
        let x = &input[block_idx * QK_K..(block_idx + 1) * QK_K];
        acc += dot_f32(&tmp, x);
    }
    acc
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn read_f16(bytes: &[u8]) -> f16 {
    let bits = match bytes {
        [lo, hi] => u16::from_le_bytes([*lo, *hi]),
        _ => 0,
    };
    f16::from_bits(bits)
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    match bytes {
        [b0, b1, b2, b3] => u32::from_le_bytes([*b0, *b1, *b2, *b3]),
        _ => 0,
    }
}

fn validate_f16_scale(name: &str, block_idx: usize, value: f16) -> Result<()> {
    if value.to_f32().is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidGguf(format!(
            "{name} is non-finite in block {block_idx}"
        )))
    }
}
