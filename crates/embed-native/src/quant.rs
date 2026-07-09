//! CPU dequantization for GGML block formats used by EmbeddingGemma.
//!
//! The formulas are adapted from candle-core 0.11.0's `quantized::k_quants`
//! (itself a Rust port of ggml/llama.cpp), but decode from raw bytes directly
//! so mmap alignment never matters.

use std::fmt;

use half::{bf16, f16};

use crate::{Error, Result};

const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
const QK8_0: usize = 32;
const QK8_1: usize = 32;
const QK_K: usize = 256;

const Q4_0_SIZE: usize = 2 + QK4_0 / 2;
const Q4_1_SIZE: usize = 2 + 2 + QK4_1 / 2;
const Q5_0_SIZE: usize = 2 + 4 + QK5_0 / 2;
const Q5_1_SIZE: usize = 2 + 2 + 4 + QK5_1 / 2;
const Q8_0_SIZE: usize = 2 + QK8_0;
const Q8_1_SIZE: usize = 2 + 2 + QK8_1;
const Q2_K_SIZE: usize = QK_K / 16 + QK_K / 4 + 2 + 2;
const Q3_K_SIZE: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
const Q4_K_SIZE: usize = 2 + 2 + 12 + QK_K / 2;
const Q5_K_SIZE: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
const Q6_K_SIZE: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
const Q8_K_SIZE: usize = 4 + QK_K + QK_K / 16 * 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub fn from_u32(u: u32) -> Result<Self> {
        match u {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            30 => Ok(Self::BF16),
            _ => Err(Error::InvalidGguf(format!("unknown GGML dtype id {u}"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => QK4_0,
            Self::Q4_1 => QK4_1,
            Self::Q5_0 => QK5_0,
            Self::Q5_1 => QK5_1,
            Self::Q8_0 => QK8_0,
            Self::Q8_1 => QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => QK_K,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => Q4_0_SIZE,
            Self::Q4_1 => Q4_1_SIZE,
            Self::Q5_0 => Q5_0_SIZE,
            Self::Q5_1 => Q5_1_SIZE,
            Self::Q8_0 => Q8_0_SIZE,
            Self::Q8_1 => Q8_1_SIZE,
            Self::Q2K => Q2_K_SIZE,
            Self::Q3K => Q3_K_SIZE,
            Self::Q4K => Q4_K_SIZE,
            Self::Q5K => Q5_K_SIZE,
            Self::Q6K => Q6_K_SIZE,
            Self::Q8K => Q8_K_SIZE,
        }
    }
}

impl fmt::Display for GgmlDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub fn dequantize(dtype: GgmlDType, raw: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    validate_raw(dtype, raw, elem_count)?;
    let mut out = vec![0.0f32; elem_count];
    match dtype {
        GgmlDType::F32 => dequantize_f32(raw, &mut out)?,
        GgmlDType::F16 => dequantize_f16(raw, &mut out)?,
        GgmlDType::BF16 => dequantize_bf16(raw, &mut out)?,
        GgmlDType::Q5_0 => dequantize_q5_0(raw, &mut out)?,
        GgmlDType::Q8_0 => dequantize_q8_0(raw, &mut out)?,
        GgmlDType::Q4K => dequantize_q4_k(raw, &mut out)?,
        GgmlDType::Q5K => dequantize_q5_k(raw, &mut out)?,
        GgmlDType::Q6K => dequantize_q6_k(raw, &mut out)?,
        other => return Err(Error::UnsupportedDType(other)),
    }
    ensure_finite(dtype.name(), &out)?;
    Ok(out)
}

fn validate_raw(dtype: GgmlDType, raw: &[u8], elem_count: usize) -> Result<()> {
    let block_size = dtype.block_size();
    if elem_count % block_size != 0 {
        return Err(Error::InvalidGguf(format!(
            "{elem_count} elements is not divisible by {block_size} for {dtype}"
        )));
    }
    let expected = elem_count / block_size * dtype.type_size();
    if raw.len() != expected {
        return Err(Error::InvalidGguf(format!(
            "{dtype} raw byte length mismatch: got {}, expected {expected} for {elem_count} elems",
            raw.len()
        )));
    }
    Ok(())
}

fn dequantize_f32(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (dst, bytes) in out.iter_mut().zip(raw.chunks_exact(4)) {
        *dst = f32::from_le_bytes(read_array(bytes, "f32")?);
    }
    Ok(())
}

fn dequantize_f16(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (dst, bytes) in out.iter_mut().zip(raw.chunks_exact(2)) {
        *dst = read_f16(bytes)?;
    }
    Ok(())
}

fn dequantize_bf16(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (dst, bytes) in out.iter_mut().zip(raw.chunks_exact(2)) {
        *dst = bf16::from_bits(u16::from_le_bytes(read_array(bytes, "bf16")?)).to_f32();
    }
    Ok(())
}

fn dequantize_q5_0(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (block, y) in raw.chunks_exact(Q5_0_SIZE).zip(out.chunks_exact_mut(QK5_0)) {
        let d = read_f16(&block[0..2])?;
        let qh = u32::from_le_bytes(read_array(&block[2..6], "Q5_0 qh")?);
        let qs = &block[6..22];
        for j in 0..(QK5_0 / 2) {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0f) | xh_0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
            y[j] = x0 as f32 * d;
            y[j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
    Ok(())
}

fn dequantize_q8_0(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (block, y) in raw.chunks_exact(Q8_0_SIZE).zip(out.chunks_exact_mut(QK8_0)) {
        let d = read_f16(&block[0..2])?;
        for (dst, &src) in y.iter_mut().zip(&block[2..34]) {
            *dst = (src as i8) as f32 * d;
        }
    }
    Ok(())
}

fn dequantize_q4_k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (block, y) in raw.chunks_exact(Q4_K_SIZE).zip(out.chunks_exact_mut(QK_K)) {
        let d = read_f16(&block[0..2])?;
        let dmin = read_f16(&block[2..4])?;
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut is = 0;
        let mut out_idx = 0;
        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
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
            is += 2;
        }
    }
    Ok(())
}

fn dequantize_q5_k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (block_idx, block) in raw.chunks_exact(Q5_K_SIZE).enumerate() {
        let d = read_f16(&block[0..2])?;
        let dmin = read_f16(&block[2..4])?;
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let y = &mut out[block_idx * QK_K..(block_idx + 1) * QK_K];

        let mut scale_idx = 0;
        let mut qs_base = 0;
        let mut high_lo = 1u8;
        let mut high_hi = 2u8;
        for out_base in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(scale_idx, scales);
            let (sc2, m2) = get_scale_min_k4(scale_idx + 1, scales);
            let scale1 = d * sc1 as f32;
            let scale2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;
            for l in 0..32 {
                let q4 = qs[qs_base + l] & 0x0f;
                let high = if qh[l] & high_lo != 0 { 16 } else { 0 };
                y[out_base + l] = scale1 * (q4 as f32 + high as f32) - min1;
            }
            for l in 0..32 {
                let q4 = qs[qs_base + l] >> 4;
                let high = if qh[l] & high_hi != 0 { 16 } else { 0 };
                y[out_base + 32 + l] = scale2 * (q4 as f32 + high as f32) - min2;
            }
            scale_idx += 2;
            qs_base += 32;
            high_lo <<= 2;
            high_hi <<= 2;
        }
    }
    Ok(())
}

fn dequantize_q6_k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    for (block_idx, block) in raw.chunks_exact(Q6_K_SIZE).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = read_f16(&block[208..210])?;
        let y = &mut out[block_idx * QK_K..(block_idx + 1) * QK_K];

        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql = &ql[64 * idx..];
            let qh = &qh[32 * idx..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                y[n + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[n + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[n + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[n + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
    Ok(())
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
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

fn read_f16(bytes: &[u8]) -> Result<f32> {
    Ok(f16::from_bits(u16::from_le_bytes(read_array(bytes, "f16")?)).to_f32())
}

fn read_array<const N: usize>(bytes: &[u8], what: &str) -> Result<[u8; N]> {
    bytes.try_into().map_err(|_| {
        Error::InvalidGguf(format!(
            "{what} read expected {N} bytes, got {}",
            bytes.len()
        ))
    })
}

fn ensure_finite(context: &str, values: &[f32]) -> Result<()> {
    if let Some((idx, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Error::InvalidGguf(format!(
            "{context} dequantized non-finite value {value} at element {idx}"
        )));
    }
    Ok(())
}
