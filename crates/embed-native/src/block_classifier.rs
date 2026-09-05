//! Versioned loader and portable forward pass for the R5 bash-smart block head.
//!
//! The encoder remains [`crate::EmbeddingGemma`], which selects CPU, CUDA, or
//! Metal. This module consumes its 768-wide pooled block vectors. The small
//! head intentionally uses one portable f32 implementation on every backend so
//! exported weights have one decode and one numerical contract.

use crate::{Error, Result, EMBEDDING_DIM};

const MAGIC: &[u8; 8] = b"GRPYR5H1";
const FORMAT_VERSION: u32 = 1;
const HEADER_BYTES: usize = 128;
const HIDDEN: usize = 256;
const OUTPUTS: usize = 4;
const LABELS: usize = 4;
const EXPECTED_FLOATS: usize =
    EMBEDDING_DIM * 2 + HIDDEN * EMBEDDING_DIM + HIDDEN + OUTPUTS * HIDDEN + OUTPUTS;

/// Frozen class order emitted by the R5 checkpoint.
pub const BLOCK_CLASSIFIER_LABELS: [&str; OUTPUTS] = ["error", "warning", "progress", "text"];

#[derive(Debug, Clone)]
pub struct BlockClassifier {
    scaler_mean: Vec<f32>,
    scaler_scale: Vec<f32>,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
    error_threshold: f32,
    warning_threshold: f32,
    source_checkpoint_sha256: [u8; 32],
    frozen_thresholds_sha256: [u8; 32],
}

impl BlockClassifier {
    /// Decode the little-endian `greppy.r5-classifier.f32le.1` asset.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
            return Err(Error::InvalidGguf(
                "invalid R5 classifier magic/header".into(),
            ));
        }
        let version = read_u32(bytes, 8)?;
        let header_bytes = read_u32(bytes, 12)? as usize;
        let input = read_u32(bytes, 16)? as usize;
        let hidden = read_u32(bytes, 20)? as usize;
        let outputs = read_u32(bytes, 24)? as usize;
        let labels = read_u32(bytes, 28)? as usize;
        let float_count = read_u64(bytes, 40)? as usize;
        if version != FORMAT_VERSION
            || header_bytes != HEADER_BYTES
            || input != EMBEDDING_DIM
            || hidden != HIDDEN
            || outputs != OUTPUTS
            || labels != LABELS
            || float_count != EXPECTED_FLOATS
        {
            return Err(Error::InvalidGguf(format!(
                "unsupported R5 classifier layout version={version} header={header_bytes} dims={input}x{hidden}x{outputs} labels={labels} floats={float_count}"
            )));
        }
        let expected_bytes = HEADER_BYTES
            .checked_add(float_count.checked_mul(4).ok_or_else(|| {
                Error::InvalidGguf("R5 classifier payload length overflow".into())
            })?)
            .ok_or_else(|| Error::InvalidGguf("R5 classifier file length overflow".into()))?;
        if bytes.len() != expected_bytes {
            return Err(Error::InvalidGguf(format!(
                "R5 classifier length {}, expected {expected_bytes}",
                bytes.len()
            )));
        }
        if bytes[112..HEADER_BYTES].iter().any(|byte| *byte != 0) {
            return Err(Error::InvalidGguf(
                "R5 classifier reserved header bytes are nonzero".into(),
            ));
        }
        let mut source_checkpoint_sha256 = [0u8; 32];
        source_checkpoint_sha256.copy_from_slice(&bytes[48..80]);
        let mut frozen_thresholds_sha256 = [0u8; 32];
        frozen_thresholds_sha256.copy_from_slice(&bytes[80..112]);

        let values = bytes[HEADER_BYTES..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidGguf(
                "R5 classifier contains non-finite values".into(),
            ));
        }
        let mut cursor = 0usize;
        let mut take = |count: usize| {
            let start = cursor;
            cursor += count;
            values[start..cursor].to_vec()
        };
        let scaler_mean = take(EMBEDDING_DIM);
        let scaler_scale = take(EMBEDDING_DIM);
        let linear1_weight = take(HIDDEN * EMBEDDING_DIM);
        let linear1_bias = take(HIDDEN);
        let linear2_weight = take(OUTPUTS * HIDDEN);
        let linear2_bias = take(OUTPUTS);
        debug_assert_eq!(cursor, values.len());
        if scaler_scale.iter().any(|scale| *scale <= 0.0) {
            return Err(Error::InvalidGguf(
                "R5 classifier scaler has a non-positive scale".into(),
            ));
        }

        Ok(Self {
            scaler_mean,
            scaler_scale,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            error_threshold: read_f32(bytes, 32)?,
            warning_threshold: read_f32(bytes, 36)?,
            source_checkpoint_sha256,
            frozen_thresholds_sha256,
        })
    }

    pub fn error_threshold(&self) -> f32 {
        self.error_threshold
    }

    pub fn warning_threshold(&self) -> f32 {
        self.warning_threshold
    }

    pub fn source_checkpoint_sha256(&self) -> [u8; 32] {
        self.source_checkpoint_sha256
    }

    pub fn frozen_thresholds_sha256(&self) -> [u8; 32] {
        self.frozen_thresholds_sha256
    }

    /// Return logits in the frozen `error, warning, progress, text` order.
    pub fn logits(&self, vector: &[f32]) -> Result<[f32; OUTPUTS]> {
        if vector.len() != EMBEDDING_DIM {
            return Err(Error::InvalidGguf(format!(
                "R5 classifier expected {EMBEDDING_DIM} inputs, got {}",
                vector.len()
            )));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidGguf(
                "R5 classifier input contains non-finite values".into(),
            ));
        }
        let mut normalized = [0.0f32; EMBEDDING_DIM];
        for index in 0..EMBEDDING_DIM {
            normalized[index] =
                (vector[index] - self.scaler_mean[index]) / self.scaler_scale[index];
        }
        let mut hidden = [0.0f32; HIDDEN];
        for (row, slot) in hidden.iter_mut().enumerate() {
            let weights = &self.linear1_weight[row * EMBEDDING_DIM..(row + 1) * EMBEDDING_DIM];
            let mut sum = self.linear1_bias[row];
            for index in 0..EMBEDDING_DIM {
                sum += normalized[index] * weights[index];
            }
            *slot = gelu(sum);
        }
        let mut logits = [0.0f32; OUTPUTS];
        for (row, slot) in logits.iter_mut().enumerate() {
            let weights = &self.linear2_weight[row * HIDDEN..(row + 1) * HIDDEN];
            let mut sum = self.linear2_bias[row];
            for index in 0..HIDDEN {
                sum += hidden[index] * weights[index];
            }
            *slot = sum;
        }
        Ok(logits)
    }

    pub fn probabilities(&self, vector: &[f32]) -> Result<[f32; OUTPUTS]> {
        let mut values = self.logits(vector)?;
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in &mut values {
            *value = (*value - max).exp();
            sum += *value;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(Error::InvalidGguf(
                "R5 classifier softmax produced an invalid denominator".into(),
            ));
        }
        for value in &mut values {
            *value /= sum;
        }
        Ok(values)
    }

    pub fn probabilities_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<[f32; OUTPUTS]>> {
        vectors
            .iter()
            .map(|vector| self.probabilities(vector))
            .collect()
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::InvalidGguf("truncated R5 classifier header".into()))?;
    Ok(u32::from_le_bytes(raw.try_into().expect("four bytes")))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::InvalidGguf("truncated R5 classifier header".into()))?;
    Ok(u64::from_le_bytes(raw.try_into().expect("eight bytes")))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32> {
    let value = f32::from_bits(read_u32(bytes, offset)?);
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| Error::InvalidGguf("non-finite R5 classifier threshold".into()))
}

// Abramowitz-Stegun 7.1.26, max |error| ~= 1.5e-7. PyTorch's exported head
// uses exact GELU (erf form), not the tanh approximation.
pub(crate) fn gelu(value: f32) -> f32 {
    let x = value * std::f32::consts::FRAC_1_SQRT_2;
    0.5 * value * (1.0 + erf(x))
}

fn erf(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truncated_asset() {
        assert!(BlockClassifier::from_bytes(b"GRPYR5H1").is_err());
    }

    #[test]
    fn erf_anchor_points() {
        assert!(erf(0.0).abs() < 2e-7);
        assert!((erf(1.0) - 0.842_700_8).abs() < 2e-7);
        assert!((erf(-1.0) + 0.842_700_8).abs() < 2e-7);
    }
}
