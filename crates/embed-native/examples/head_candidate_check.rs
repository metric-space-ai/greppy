//! Check every exported candidate against bound same-vector reference cases.
use greppy_embed_native::head_candidate::{CandidateHead, CandidateManifest, HeadOutput};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, error::Error, fs, path::Path};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Golden {
    schema: String,
    cases: Vec<Case>,
    scope: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    vector: Vec<f32>,
    expected: HeadOutput,
}

#[derive(Deserialize)]
struct ExportReport {
    schema: String,
    candidates: usize,
    golden_cases: usize,
    exports: Vec<ExportItem>,
}
#[derive(Deserialize)]
struct ExportItem {
    run_id: String,
    manifest_sha256: String,
}
fn main() -> Result<(), Box<dyn Error>> {
    let root = std::env::args()
        .nth(1)
        .ok_or("provide candidate export directory")?;
    let report: ExportReport =
        serde_json::from_slice(&fs::read(Path::new(&root).join("export-report.json"))?)?;
    let mut expected: HashMap<_, _> = report
        .exports
        .iter()
        .map(|item| (item.run_id.clone(), item.manifest_sha256.clone()))
        .collect();
    if report.schema != "greppy.heads.candidate-export.v1"
        || report.candidates == 0
        || expected.len() != report.candidates
        || expected.len() != report.exports.len()
    {
        return Err("invalid candidate inventory".into());
    }
    let mut candidates = 0usize;
    let mut cases = 0usize;
    let mut max_difference = 0.0f32;
    for entry in fs::read_dir(Path::new(&root))? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let raw_manifest = fs::read(path.join("manifest.json"))?;
        let manifest: CandidateManifest = serde_json::from_slice(&raw_manifest)?;
        let expected_sha = expected
            .remove(&manifest.source_run_id)
            .ok_or("unexpected or duplicate candidate")?;
        if format!("{:x}", Sha256::digest(&raw_manifest)) != expected_sha {
            return Err("candidate manifest checksum mismatch".into());
        }
        let model = CandidateHead::load(
            &raw_manifest,
            &fs::read(path.join("weights.f32le"))?,
            &manifest.input_contract_sha256,
            &manifest.representation_sha256,
        )?;
        let raw_golden = fs::read(path.join("golden.json"))?;
        if format!("{:x}", Sha256::digest(&raw_golden)) != manifest.golden_sha256 {
            return Err("golden checksum mismatch".into());
        }
        let golden: Golden = serde_json::from_slice(&raw_golden)?;
        if golden.schema != "greppy.heads.candidate-golden.v1"
            || golden.cases.is_empty()
            || golden.scope.is_empty()
        {
            return Err("invalid golden cases".into());
        }
        for case in golden.cases {
            let actual = model.predict_for_validation(&case.vector)?;
            let difference = match (actual, case.expected) {
                (
                    HeadOutput::Classification { probabilities: a },
                    HeadOutput::Classification { probabilities: b },
                ) => a
                    .into_iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max),
                (HeadOutput::Relevance { score: a }, HeadOutput::Relevance { score: b }) => {
                    (a - b).abs()
                }
                _ => return Err("golden output kind mismatch".into()),
            };
            if !difference.is_finite() || difference > 5e-5 {
                return Err(format!(
                    "candidate {} differs by {difference}",
                    manifest.source_run_id
                )
                .into());
            }
            max_difference = max_difference.max(difference);
            cases += 1;
        }
        candidates += 1;
    }
    if !expected.is_empty() || candidates != report.candidates || cases != report.golden_cases {
        return Err("incomplete candidate or golden coverage".into());
    }
    println!(
        "{}",
        serde_json::json!({
            "candidates": candidates, "cases": cases, "max_absolute_difference": max_difference,
            "absolute_tolerance": 5e-5, "production_eligible": false,
            "scope": "same-vector portable head arithmetic"
        })
    );
    Ok(())
}
