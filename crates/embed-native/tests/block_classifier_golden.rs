//! Python/PyTorch -> Rust golden for the frozen R5 classifier.

use greppy_embed_native::BlockClassifier;
use sha2::{Digest, Sha256};

const N: usize = 64;
const INPUT: usize = 768;
const OUTPUTS: usize = 4;
const BUDGET: f32 = 5.0e-5;

#[test]
fn python_r5_outputs_match_portable_rust_forward() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cli/assets/bash-smart-head-r5");
    let asset = std::fs::read(root.join("classifier-v1.f32le")).expect("read classifier asset");
    assert_eq!(
        format!("{:x}", Sha256::digest(&asset)),
        "523c23339149d0cae8f30d15c422d10d82b7be5fcc9935aba3bcc805790a6a1c"
    );
    let classifier = BlockClassifier::from_bytes(&asset).expect("load classifier asset");
    let bytes = std::fs::read(root.join("golden-v1.f32le")).expect("read classifier golden");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "b922a254670028da5a413e12a74bdf0697d8175656a980b940a7aed6317f7e6e"
    );
    assert_eq!(&bytes[..8], b"GRPYR5G1");
    assert_eq!(u32_at(&bytes, 8), 1);
    assert_eq!(u32_at(&bytes, 12), 64);
    assert_eq!(u32_at(&bytes, 16) as usize, N);
    assert_eq!(u32_at(&bytes, 20) as usize, INPUT);
    let expected_len = 64 + (N * INPUT + N * OUTPUTS * 2) * 4;
    assert_eq!(bytes.len(), expected_len);
    let values = bytes[64..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    let vectors = &values[..N * INPUT];
    let python_logits = &values[N * INPUT..N * (INPUT + OUTPUTS)];
    let python_probs = &values[N * (INPUT + OUTPUTS)..];

    let mut max_abs_diff = 0.0f32;
    for row in 0..N {
        let vector = &vectors[row * INPUT..(row + 1) * INPUT];
        let logits = classifier.logits(vector).expect("Rust logits");
        let probs = classifier
            .probabilities(vector)
            .expect("Rust probabilities");
        for col in 0..OUTPUTS {
            max_abs_diff = max_abs_diff
                .max((logits[col] - python_logits[row * OUTPUTS + col]).abs())
                .max((probs[col] - python_probs[row * OUTPUTS + col]).abs());
        }
    }
    assert!(
        max_abs_diff <= BUDGET,
        "R5 Python/Rust max_abs_diff={max_abs_diff:.9} exceeds budget={BUDGET:.9}"
    );
    eprintln!("R5 classifier golden N={N} max_abs_diff={max_abs_diff:.9} budget={BUDGET:.9}");
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
