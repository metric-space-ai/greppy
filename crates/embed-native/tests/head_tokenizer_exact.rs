use greppy_embed_native::{PromptTokenizer, TokenizerConfig};
use std::path::PathBuf;

struct Fixture(PathBuf);
impl Fixture {
    fn tokenizer(&self, max_length: usize) -> PromptTokenizer {
        PromptTokenizer::from_file(
            &self.0,
            TokenizerConfig {
                max_length,
                pad_token_id: 0,
            },
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn fixture() -> Fixture {
    let path =
        std::env::temp_dir().join(format!("head-exact-tokenizer-{}.json", std::process::id()));
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": {"direction":"Right","max_length":2,"strategy":"LongestFirst","stride":0},
        "padding": {"strategy":{"Fixed":8},"direction":"Right","pad_to_multiple_of":null,
                    "pad_id":0,"pad_type_id":0,"pad_token":"[PAD]"},
        "added_tokens": [], "normalizer": null, "pre_tokenizer":{"type":"Whitespace"},
        "post_processor": null, "decoder": null,
        "model":{"type":"WordLevel","vocab":{"[PAD]":0,"[UNK]":1,"alpha":2,"beta":3,"gamma":4},
                 "unk_token":"[UNK]"}
    });
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    serde_json::to_writer(&mut file, &json).unwrap();
    Fixture(path)
}
#[test]
fn serialized_truncation_padding_and_exact_limits() {
    let fixture = fixture();
    let tokenizer = fixture.tokenizer(4);
    // The file requests two-token truncation and eight-token padding; raw IDs
    // must still retain every real token and exclude every padding token.
    assert_eq!(
        tokenizer.encode_ids("alpha beta gamma").unwrap(),
        vec![2, 3, 4]
    );
    assert_eq!(tokenizer.token_len("alpha beta gamma").unwrap(), 3);
    let batch = tokenizer
        .encode_prompts_exact(["alpha beta gamma", "beta"])
        .unwrap();
    assert_eq!(batch.token_ids, vec![vec![2, 3, 4], vec![3, 0, 0]]);
    assert_eq!(batch.attention_mask, vec![vec![1, 1, 1], vec![1, 0, 0]]);
    assert!(tokenizer
        .encode_prompts_exact(["alpha beta gamma alpha beta"])
        .is_err());
    assert!(tokenizer.encode_prompts_exact([""]).is_err());
    assert!(tokenizer
        .encode_prompts_exact(Vec::<String>::new())
        .unwrap()
        .is_empty());
    let narrow = fixture.tokenizer(2);
    assert_eq!(
        narrow
            .encode_prompts(["alpha beta gamma"])
            .unwrap()
            .token_ids[0],
        vec![2, 3]
    );
    assert!(narrow.encode_prompts_exact(["alpha beta gamma"]).is_err());
    // A huge single lexical token fits the token budget but the old byte guard
    // would drop the final known word. Exact mode must detect that change.
    let huge = "x".repeat(9000) + " beta";
    assert_eq!(tokenizer.encode_ids(&huge).unwrap(), vec![1, 3]);
    assert!(tokenizer.encode_prompts_exact([huge]).is_err());
}
