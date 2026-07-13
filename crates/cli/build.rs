//! Build-time embedding of the EmbeddingGemma-300M and Qwen3.5-0.8B Q4_K models.
//!
//! Owner rule: greppy works OUT OF THE BOX — semantic search must ALWAYS
//! work, so the model ships INSIDE the binary. The asset BYTES are hosted as
//! GitHub release assets (free and unlimited for public repos; Git LFS storage
//! and bandwidth were the org's dominant cost). WHAT ships stays pinned in
//! this repo: `crates/cli/assets/MODEL_ASSETS.json` plus `*.sha256` sidecars,
//! materialized and digest-verified by `tools/fetch_model_assets.sh` before
//! any build. The shipped binary is unchanged: no download, external path,
//! feature switch, or environment variable is required at runtime. The
//! sole exception is the compile-guarded `ci-test-assets` debug feature used by
//! non-inference tests; release builds cannot enable it.
//!
//! A plain `cargo build` verifies the in-repo assets before passing their
//! absolute paths to `lib.rs` for `include_bytes!`. The compiler therefore
//! embeds the verified repo files directly without creating another ~1 GiB
//! copy per Cargo build fingerprint. A binary without either model is not
//! buildable.

use std::path::{Path, PathBuf};

const GGUF_NAME: &str = "embeddinggemma-300M-Q4_K.gguf";
const GGUF_SHA: &str = "53f7d1c0d5c84a81e46f3bea8e0f17c94f459ffbaa8b06f7f52f1f09e58996f2";
const TOK_NAME: &str = "tokenizer.json";
const TOK_SHA: &str = "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e";
const QWEN_GGUF_NAME: &str = "Qwen3.5-0.8B-MTP-Q4_K_M.gguf";
const QWEN_TOK_NAME: &str = "tokenizer.json";

const CI_EMBED_GGUF_SHA: &str = "5d653fbbddef916720120b139d5647b921e48007eb86e3b1eb3e182025bc6b13";
const CI_EMBED_TOK_SHA: &str = "91350500ab5af78f0b0027547d4a40b96bf81432b63b06da7b36bb1d87fa4999";
const CI_QWEN_GGUF_SHA: &str = "b14c40dfa0c3e2428232027344341fe7cb1b4495b5086d983835cea46b7267d8";
const CI_QWEN_TOK_SHA: &str = "8a3b4f437b9ee58c2190c1e409e24c5b31c0b697041f22925705fca5365db5ca";

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    if std::env::var_os("CARGO_FEATURE_CI_TEST_ASSETS").is_some() {
        assert_ne!(
            std::env::var("PROFILE").as_deref(),
            Ok("release"),
            "ci-test-assets is forbidden in release builds"
        );
        assert!(
            std::env::var("CI").is_ok_and(|value| matches!(value.as_str(), "true" | "1")),
            "ci-test-assets is an internal CI fixture mode, not a product build option"
        );
        configure_ci_test_assets(&manifest);
        return;
    }

    println!("cargo:rustc-env=GREPPY_EMBEDDED_GGUF_SHA={GGUF_SHA}");
    println!("cargo:rustc-env=GREPPY_EMBEDDED_TOK_SHA={TOK_SHA}");
    let assets = manifest.join("assets").join("embeddinggemma-300m-q4k");
    let embedding_gguf = assets.join(GGUF_NAME);
    let embedding_tokenizer = assets.join(TOK_NAME);
    verify(&embedding_gguf, GGUF_NAME, GGUF_SHA);
    verify(&embedding_tokenizer, TOK_NAME, TOK_SHA);
    export_path("GREPPY_EMBEDDED_GGUF_PATH", &embedding_gguf);
    export_path("GREPPY_EMBEDDED_TOK_PATH", &embedding_tokenizer);
    let qwen_assets = manifest.join("assets").join("qwen35-0.8b-mtp-q4km");
    let qwen_gguf_sha = read_sha256_sidecar(&qwen_assets.join(format!("{QWEN_GGUF_NAME}.sha256")));
    let qwen_tok_sha = read_sha256_sidecar(&qwen_assets.join(format!("{QWEN_TOK_NAME}.sha256")));
    println!("cargo:rustc-env=GREPPY_EMBEDDED_QWEN35_GGUF_SHA={qwen_gguf_sha}");
    println!("cargo:rustc-env=GREPPY_EMBEDDED_QWEN35_TOK_SHA={qwen_tok_sha}");
    let qwen_gguf = qwen_assets.join(QWEN_GGUF_NAME);
    let qwen_tokenizer = qwen_assets.join(QWEN_TOK_NAME);
    verify(&qwen_gguf, QWEN_GGUF_NAME, &qwen_gguf_sha);
    verify(&qwen_tokenizer, QWEN_TOK_NAME, &qwen_tok_sha);
    export_path("GREPPY_EMBEDDED_QWEN35_GGUF_PATH", &qwen_gguf);
    export_path("GREPPY_EMBEDDED_QWEN35_TOK_PATH", &qwen_tokenizer);
}

fn configure_ci_test_assets(manifest: &Path) {
    let assets = manifest.join("tests").join("assets").join("ci-test");
    let embedding_gguf = assets.join("embedding-model.bin");
    let embedding_tokenizer = assets.join("embedding-tokenizer.json");
    let qwen_gguf = assets.join("qwen-model.bin");
    let qwen_tokenizer = assets.join("qwen-tokenizer.json");

    verify(&embedding_gguf, "CI embedding sentinel", CI_EMBED_GGUF_SHA);
    verify(
        &embedding_tokenizer,
        "CI embedding tokenizer sentinel",
        CI_EMBED_TOK_SHA,
    );
    verify(&qwen_gguf, "CI Qwen sentinel", CI_QWEN_GGUF_SHA);
    verify(
        &qwen_tokenizer,
        "CI Qwen tokenizer sentinel",
        CI_QWEN_TOK_SHA,
    );

    println!("cargo:rustc-env=GREPPY_EMBEDDED_GGUF_SHA={CI_EMBED_GGUF_SHA}");
    println!("cargo:rustc-env=GREPPY_EMBEDDED_TOK_SHA={CI_EMBED_TOK_SHA}");
    println!("cargo:rustc-env=GREPPY_EMBEDDED_QWEN35_GGUF_SHA={CI_QWEN_GGUF_SHA}");
    println!("cargo:rustc-env=GREPPY_EMBEDDED_QWEN35_TOK_SHA={CI_QWEN_TOK_SHA}");
    export_path("GREPPY_EMBEDDED_GGUF_PATH", &embedding_gguf);
    export_path("GREPPY_EMBEDDED_TOK_PATH", &embedding_tokenizer);
    export_path("GREPPY_EMBEDDED_QWEN35_GGUF_PATH", &qwen_gguf);
    export_path("GREPPY_EMBEDDED_QWEN35_TOK_PATH", &qwen_tokenizer);
}

/// Verify the exact repo file that rustc embeds. Panics with a precise error
/// instead of allowing a missing, LFS-pointer, or modified model into a build.
fn verify(repo_asset: &Path, name: &str, want_sha: &str) {
    println!("cargo:rerun-if-changed={}", repo_asset.display());
    let src = repo_asset.to_path_buf();
    assert!(
        src.exists(),
        "embedded model asset `{name}` not found at {}.\n\
         Fetch the release-hosted model assets first: run `./tools/fetch_model_assets.sh`.\n\
         Refusing to build a binary without its repo-owned model.",
        src.display(),
    );
    // A Git-LFS pointer file (a few hundred bytes) is not the real asset —
    // catch the common "asset not fetched" case with a clear message.
    let got = sha256_file(&src);
    assert_eq!(
        got,
        want_sha,
        "embedded model `{name}` at {} has the wrong SHA256 (got {got}).\n\
         If this is a ~130-byte Git-LFS pointer, run `git lfs pull`. Refusing to bake an unverified model.",
        src.display(),
    );
}

fn export_path(name: &str, path: &Path) {
    let canonical = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize embedded asset {}: {error}", path.display()));
    println!("cargo:rustc-env={name}={}", canonical.display());
}

fn sha256_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut f = std::fs::File::open(path).expect("open model file for hashing");
    std::io::copy(&mut f, &mut hasher).expect("hash model file");
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn read_sha256_sidecar(path: &Path) -> String {
    println!("cargo:rerun-if-changed={}", path.display());
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "embedded Qwen3.5 sidecar {} is required for every greppy build: {e}",
            path.display()
        )
    });
    let sha = raw.trim();
    assert!(
        sha.len() == 64
            && sha
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "embedded Qwen3.5 sidecar {} must contain a 64-byte lowercase SHA256",
        path.display()
    );
    sha.to_string()
}
