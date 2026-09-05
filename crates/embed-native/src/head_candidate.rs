//! Checked portable arithmetic for experimental head exports.
//!
//! This format deliberately has no production release or backend calibration.
//! Loading it cannot authorize automatic model activation.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

const MAGIC: &[u8; 8] = b"GRPYHD01";
const HEADER: usize = 64;
const DIM: usize = 768;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct CandidateError(String);

type Result<T> = std::result::Result<T, CandidateError>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeadKind {
    LogClassifier,
    LogRanker,
    WebRanker,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    Classification,
    Ordinal,
    Pairwise,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateManifest {
    pub schema: String,
    pub role: String,
    pub head: HeadKind,
    pub objective: Objective,
    pub input_dimension: usize,
    pub hidden_dimension: usize,
    pub input_contract_sha256: String,
    pub representation_sha256: String,
    pub source_run_id: String,
    pub weights_sha256: String,
    pub golden_sha256: String,
    pub validated_backends: Vec<String>,
    pub calibration: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HeadOutput {
    Classification { probabilities: [f32; 4] },
    Relevance { score: f32 },
}

#[derive(Debug)]
pub struct CandidateHead {
    manifest: CandidateManifest,
    values: Vec<f32>,
}

fn failure(message: &str) -> CandidateError {
    CandidateError(message.into())
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("checked header"),
    )
}

impl CandidateHead {
    /// The caller supplies the input/representation contract it actually used.
    /// An asset's self-declared contract is not sufficient to establish a match.
    pub fn load(
        manifest_json: &[u8],
        weights: &[u8],
        expected_input_contract: &str,
        expected_representation: &str,
    ) -> Result<Self> {
        if manifest_json.len() > 65536 {
            return Err(failure("candidate manifest exceeds size limit"));
        }
        let manifest: CandidateManifest = serde_json::from_slice(manifest_json)
            .map_err(|_| failure("invalid candidate manifest"))?;
        if manifest.schema != "greppy.heads.candidate.v1"
            || !matches!(
                manifest.role.as_str(),
                "synthetic_pipeline_test" | "development_candidate"
            )
            || manifest.input_dimension != DIM
            || !matches!(manifest.hidden_dimension, 0 | 128 | 256)
            || !manifest.validated_backends.is_empty()
            || manifest.calibration.is_some()
        {
            return Err(failure(
                "unsupported candidate contract; not a production release format",
            ));
        }
        for hash in [
            &manifest.input_contract_sha256,
            &manifest.representation_sha256,
            &manifest.source_run_id,
            &manifest.weights_sha256,
            &manifest.golden_sha256,
        ] {
            if !valid_hash(hash) {
                return Err(failure("invalid candidate checksum"));
            }
        }
        if manifest.input_contract_sha256 != expected_input_contract
            || manifest.representation_sha256 != expected_representation
        {
            return Err(failure(
                "candidate input or representation contract mismatch",
            ));
        }
        if (manifest.head == HeadKind::LogClassifier)
            != (manifest.objective == Objective::Classification)
        {
            return Err(failure("head kind does not match its objective"));
        }
        let hidden = manifest.hidden_dimension;
        let outputs = if manifest.objective == Objective::Classification {
            4
        } else {
            1
        };
        let cuts = if manifest.objective == Objective::Ordinal {
            3
        } else {
            0
        };
        let expected_floats = DIM * 2
            + if hidden == 0 {
                outputs * DIM + outputs
            } else {
                hidden * DIM + hidden + outputs * hidden + outputs
            }
            + cuts;
        if weights.len() != HEADER + 4 * expected_floats
            || &weights[..8] != MAGIC
            || u32_at(weights, 8) != 1
            || u32_at(weights, 12) as usize != DIM
            || u32_at(weights, 16) as usize != hidden
            || u32_at(weights, 20) as usize != outputs
            || u32_at(weights, 24) != manifest.objective as u32
            || u32_at(weights, 28) != manifest.head as u32
            || u64::from_le_bytes(weights[32..40].try_into().expect("checked header")) as usize
                != expected_floats
            || weights[40..HEADER].iter().any(|byte| *byte != 0)
        {
            return Err(failure("invalid candidate weight layout"));
        }
        if format!("{:x}", Sha256::digest(weights)) != manifest.weights_sha256 {
            return Err(failure("candidate weight checksum mismatch"));
        }
        let values: Vec<f32> = weights[HEADER..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("float chunk")))
            .collect();
        if values.iter().any(|v| !v.is_finite())
            || values[DIM..DIM * 2].iter().any(|scale| *scale <= 0.0)
        {
            return Err(failure("invalid candidate parameters"));
        }
        if cuts == 3 {
            let end = values.len();
            if !(values[end - 3] < values[end - 2] && values[end - 2] < values[end - 1]) {
                return Err(failure("ordinal cutpoints must increase strictly"));
            }
        }
        Ok(Self { manifest, values })
    }

    pub fn manifest(&self) -> &CandidateManifest {
        &self.manifest
    }

    /// Arithmetic validation only; this uncalibrated method grants no backend release.
    pub fn predict_for_validation(&self, vector: &[f32]) -> Result<HeadOutput> {
        if vector.len() != DIM || vector.iter().any(|v| !v.is_finite()) {
            return Err(failure("invalid candidate input vector"));
        }
        let mut normalized = vec![0.0; DIM];
        for i in 0..DIM {
            normalized[i] = (vector[i] - self.values[i]) / self.values[DIM + i];
        }
        if normalized.iter().any(|v| !v.is_finite()) {
            return Err(failure("candidate normalization overflow"));
        }
        let hidden = self.manifest.hidden_dimension;
        let outputs = if self.manifest.objective == Objective::Classification {
            4
        } else {
            1
        };
        let mut cursor = DIM * 2;
        let mut affine = |input: &[f32], rows: usize| -> Result<Vec<f32>> {
            let size = input.len() * rows;
            let weights = &self.values[cursor..cursor + size];
            let biases = &self.values[cursor + size..cursor + size + rows];
            cursor += size + rows;
            let mut result = Vec::with_capacity(rows);
            for row in 0..rows {
                let mut value = biases[row];
                for (i, x) in input.iter().enumerate() {
                    value += *x * weights[row * input.len() + i];
                }
                if !value.is_finite() {
                    return Err(failure("candidate affine overflow"));
                }
                result.push(value);
            }
            Ok(result)
        };
        let logits = if hidden == 0 {
            affine(&normalized, outputs)?
        } else {
            let mut activated = affine(&normalized, hidden)?;
            for value in &mut activated {
                *value = crate::block_classifier::gelu(*value);
                if !value.is_finite() {
                    return Err(failure("candidate activation overflow"));
                }
            }
            affine(&activated, outputs)?
        };
        match self.manifest.objective {
            Objective::Classification => {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probabilities = [0.0; 4];
                for i in 0..4 {
                    probabilities[i] = (logits[i] - max).exp();
                }
                let sum: f32 = probabilities.iter().sum();
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(failure("invalid candidate probability denominator"));
                }
                for value in &mut probabilities {
                    *value /= sum;
                }
                Ok(HeadOutput::Classification { probabilities })
            }
            Objective::Pairwise => Ok(HeadOutput::Relevance { score: logits[0] }),
            Objective::Ordinal => {
                let mut score = 0.0;
                for cut in &self.values[self.values.len() - 3..] {
                    let x = logits[0] - *cut;
                    if !x.is_finite() {
                        return Err(failure("ordinal score overflow"));
                    }
                    // Stable sigmoid for both tails; no overflowing exp().
                    score += if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let e = x.exp();
                        e / (1.0 + e)
                    };
                }
                Ok(HeadOutput::Relevance { score })
            }
        }
    }

    /// Keep caller identities exactly; any invalid row rejects the entire batch.
    pub fn batch_for_validation(
        &self,
        rows: &[(&str, &[f32])],
    ) -> Result<Vec<(String, HeadOutput)>> {
        let mut seen = HashSet::new();
        let mut result = Vec::with_capacity(rows.len());
        for (id, vector) in rows {
            if id.is_empty() || id.len() > 256 || !seen.insert(*id) {
                return Err(failure("invalid or duplicate candidate source identity"));
            }
            result.push(((*id).to_string(), self.predict_for_validation(vector)?));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(objective: Objective) -> (CandidateManifest, Vec<u8>) {
        let head = if objective == Objective::Classification {
            HeadKind::LogClassifier
        } else {
            HeadKind::WebRanker
        };
        let outputs = if objective == Objective::Classification {
            4
        } else {
            1
        };
        let cuts = if objective == Objective::Ordinal {
            3
        } else {
            0
        };
        let mut values = vec![0.0f32; DIM * 2 + DIM * outputs + outputs + cuts];
        values[DIM..DIM * 2].fill(1.0);
        if cuts == 3 {
            let end = values.len();
            values[end - 3..].copy_from_slice(&[-1.0, 0.0, 1.0]);
        }
        let mut bytes = vec![0; HEADER];
        bytes[..8].copy_from_slice(MAGIC);
        for (offset, value) in [
            (8, 1),
            (12, DIM as u32),
            (16, 0),
            (20, outputs as u32),
            (24, objective as u32),
            (28, head as u32),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[32..40].copy_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let manifest = CandidateManifest {
            schema: "greppy.heads.candidate.v1".into(),
            role: "synthetic_pipeline_test".into(),
            head,
            objective,
            input_dimension: DIM,
            hidden_dimension: 0,
            input_contract_sha256: "a".repeat(64),
            representation_sha256: "b".repeat(64),
            source_run_id: "c".repeat(64),
            weights_sha256: format!("{:x}", Sha256::digest(&bytes)),
            golden_sha256: "d".repeat(64),
            validated_backends: vec![],
            calibration: None,
        };
        (manifest, bytes)
    }

    fn load(manifest: &CandidateManifest, bytes: &[u8]) -> Result<CandidateHead> {
        CandidateHead::load(
            &serde_json::to_vec(manifest).unwrap(),
            bytes,
            &"a".repeat(64),
            &"b".repeat(64),
        )
    }

    #[test]
    fn analytic_classification_and_ordinal_values() {
        let (manifest, bytes) = fixture(Objective::Classification);
        assert_eq!(
            load(&manifest, &bytes)
                .unwrap()
                .predict_for_validation(&vec![0.0; DIM])
                .unwrap(),
            HeadOutput::Classification {
                probabilities: [0.25; 4]
            }
        );
        let (manifest, bytes) = fixture(Objective::Ordinal);
        let HeadOutput::Relevance { score } = load(&manifest, &bytes)
            .unwrap()
            .predict_for_validation(&vec![0.0; DIM])
            .unwrap()
        else {
            panic!("wrong output")
        };
        assert!((score - 1.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_layout_checksum_contract_and_release_claim() {
        let (mut manifest, mut bytes) = fixture(Objective::Classification);
        assert!(load(&manifest, &bytes[..10]).is_err());
        bytes[HEADER] ^= 1;
        assert!(load(&manifest, &bytes).is_err());
        bytes[HEADER] ^= 1;
        manifest.input_contract_sha256 = "e".repeat(64);
        assert!(load(&manifest, &bytes).is_err());
        manifest.input_contract_sha256 = "a".repeat(64);
        manifest.validated_backends.push("cuda".into());
        assert!(load(&manifest, &bytes).is_err());
    }

    #[test]
    fn rejects_nonfinite_parameters_even_with_matching_checksum() {
        let (mut manifest, mut bytes) = fixture(Objective::Pairwise);
        bytes[HEADER..HEADER + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        manifest.weights_sha256 = format!("{:x}", Sha256::digest(&bytes));
        assert!(load(&manifest, &bytes).is_err());
    }

    #[test]
    fn rejects_nonincreasing_cutpoints_and_wrong_head_kind() {
        let (mut manifest, mut bytes) = fixture(Objective::Ordinal);
        let end = bytes.len();
        bytes[end - 4..].copy_from_slice(&(-1.0f32).to_le_bytes());
        manifest.weights_sha256 = format!("{:x}", Sha256::digest(&bytes));
        assert!(load(&manifest, &bytes).is_err());
        let (mut manifest, bytes) = fixture(Objective::Classification);
        manifest.head = HeadKind::LogRanker;
        assert!(load(&manifest, &bytes).is_err());
    }

    #[test]
    fn preserves_source_ids_and_rejects_partial_or_duplicate_batches() {
        let (manifest, bytes) = fixture(Objective::Pairwise);
        let model = load(&manifest, &bytes).unwrap();
        let vector = vec![0.0; DIM];
        let result = model
            .batch_for_validation(&[("source:7", &vector)])
            .unwrap();
        assert_eq!(result[0].0, "source:7");
        assert!(model
            .batch_for_validation(&[("source:7", &vector), ("source:7", &vector)])
            .is_err());
        assert!(model
            .batch_for_validation(&[("source:7", &vector), ("source:8", &[0.0])])
            .is_err());
        assert!(model
            .predict_for_validation(&vec![f32::INFINITY; DIM])
            .is_err());
    }

    #[test]
    fn rejects_finite_parameter_arithmetic_overflow() {
        let (mut manifest, mut bytes) = fixture(Objective::Pairwise);
        bytes[HEADER + DIM * 8..HEADER + DIM * 8 + 4].copy_from_slice(&f32::MAX.to_le_bytes());
        manifest.weights_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let model = load(&manifest, &bytes).unwrap();
        let mut vector = vec![0.0; DIM];
        vector[0] = 2.0;
        assert!(model.predict_for_validation(&vector).is_err());
    }
}
