//! Public qualified type usage contract; inference is not required.
use std::process::Command;

#[test]
fn who_calls_two_same_named_response_types_keeps_generic_and_variant_users_separate() {
    let fixture = tempfile::tempdir().unwrap();
    let repo = fixture.path().join("repo");
    let store = fixture.path().join("store");
    std::fs::create_dir_all(repo.join("src/scrape")).unwrap();
    std::fs::create_dir_all(repo.join("src/doc_stack")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "mod scrape; mod doc_stack;\n").unwrap();
    std::fs::write(
        repo.join("src/scrape/mod.rs"),
        r#"
pub mod semantic_enrichment;
pub struct Marker;
pub enum LocalEmbeddingSocketResponse { Ready { dimension: u32 }, Failure }
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("src/scrape/semantic_enrichment.rs"),
        r#"
use super::{Marker, LocalEmbeddingSocketResponse};
pub fn enrich() {
    let _ = serde_json::from_str::<LocalEmbeddingSocketResponse>("{}");
    let _ = LocalEmbeddingSocketResponse::Ready { dimension: 1 };
    let _ = LocalEmbeddingSocketResponse::Failure;
}
pub fn shadowed<LocalEmbeddingSocketResponse>() {
    let _ = serde_json::from_str::<LocalEmbeddingSocketResponse>("{}");
}
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("src/doc_stack/mod.rs"),
        r#"
pub enum LocalEmbeddingSocketResponse { Ready { dimension: u32 }, Failure }
pub fn parse_document() {
    let _ = serde_json::from_str::<LocalEmbeddingSocketResponse>("{}");
    let _ = LocalEmbeddingSocketResponse::Ready { dimension: 2 };
    let _ = LocalEmbeddingSocketResponse::Failure;
}
"#,
    )
    .unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_greppy"))
            .args(args)
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", &store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_AUTO_REINDEX", "0")
            .output()
            .unwrap()
    };
    let indexed = run(&["index", "."]);
    assert!(indexed.status.success(), "{indexed:?}");
    let scrape = "src/scrape/mod.rs::LocalEmbeddingSocketResponse";
    let doc = "src/doc_stack/mod.rs::LocalEmbeddingSocketResponse";
    for (target, expected, rejected) in [
        (scrape, "enrich", "parse_document"),
        (doc, "parse_document", "enrich"),
    ] {
        let result = run(&["who-calls", target]);
        assert!(result.status.success(), "{result:?}");
        let text = String::from_utf8(result.stdout).unwrap();
        assert!(text.contains(expected), "{text}");
        assert!(!text.contains(rejected), "{text}");
        assert!(!text.contains("shadowed"), "{text}");
    }
    let both = run(&["who-calls", scrape, doc]);
    assert!(both.status.success(), "{both:?}");
    let text = String::from_utf8(both.stdout).unwrap();
    assert!(
        text.contains("enrich") && text.contains("parse_document"),
        "{text}"
    );
    assert!(!text.contains("shadowed"), "{text}");
}
