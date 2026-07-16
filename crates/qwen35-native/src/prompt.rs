/// Versioned prompt/output contract used by the daemon protocol.
pub const PROMPT_VERSION: &str = "qwen35-brief-path-v5";
pub const TRIAGE_PROMPT_VERSION: &str = "qwen35-triage-v3";

/// Exact prompt used for every definition source span (path format).
///
/// The finetuned model is trained on this exact byte layout: the task tag,
/// one space, the repo-relative file path, a newline, then the unmodified
/// source span. The student sees the same path context the teacher had when
/// the training data was generated.
pub fn brief_prompt(path: &str, source: &str) -> String {
    format!("brief: {path}\n{source}")
}

/// Chat wrapper for `brief` (path format): the finetuned model is trained on
/// this exact prefix and needs no empty think block to stay in non-thinking
/// mode.
pub fn brief_chat_prompt(path: &str, source: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        brief_prompt(path, source).trim()
    )
}

pub fn triage_prompt(query: &str, span_loc: &str, span_code: &str) -> String {
    format!(
        "Given the user's question and ONE code span, decide if this span is worth the developer opening. Reply with a verdict `READ` or `SKIP` and a 3-5 word reason. If unsure, READ. Do NOT answer the question. Do NOT explain the code.\n\nQuestion:\n{query}\n\nSpan: {span_loc}\n{span_code}"
    )
}

/// Qwen3.5's tokenizer_config chat template emits this assistant prefix when
/// `add_generation_prompt=true` and `enable_thinking=false`.
pub fn non_thinking_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        user_prompt.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_exact_contract() {
        assert_eq!(
            brief_prompt("src/lib.rs", "fn f() {}\n"),
            "brief: src/lib.rs\nfn f() {}\n"
        );
        assert_eq!(
            brief_chat_prompt("src/lib.rs", "fn f() {}\n"),
            "<|im_start|>user\nbrief: src/lib.rs\nfn f() {}<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn prompt_keeps_path_and_source_verbatim() {
        // One space after the colon, one newline before the source, and the
        // source itself byte-identical — training and inference must agree.
        let prompt = brief_prompt("crates/core/src/graph store.rs", "  fn g() {}\n\n");
        assert_eq!(
            prompt,
            "brief: crates/core/src/graph store.rs\n  fn g() {}\n\n"
        );
    }

    #[test]
    fn triage_prompt_contains_query_loc_and_code() {
        let prompt = triage_prompt("who starts workers?", "src/lib.rs:10-20", "fn start() {}\n");
        assert!(prompt.starts_with("Given the user's question and ONE code span"));
        assert!(prompt.contains("If unsure, READ."));
        assert!(prompt.contains("Question:\nwho starts workers?"));
        assert!(prompt.contains("Span: src/lib.rs:10-20\nfn start() {}\n"));
    }

    #[test]
    fn non_thinking_chat_prompt_matches_qwen_template_prefix() {
        assert_eq!(
            non_thinking_chat_prompt("Summarize: What is this function for?\n\nfn f() {}\n"),
            "<|im_start|>user\nSummarize: What is this function for?\n\nfn f() {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
