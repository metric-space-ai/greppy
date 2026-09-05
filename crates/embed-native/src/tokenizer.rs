//! EmbeddingGemma prompt templates and tokenizer configuration.
//!
//! This mirrors `crates/embeddinggemma/src/lib.rs` without taking a candle
//! dependency. Token IDs and masks are verified against candle-minted goldens.

use std::borrow::Cow;
use std::path::Path;

use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams,
};

use crate::{gguf::GgufModel, Error, Result};

pub const DEFAULT_MAX_LENGTH: usize = 2048;
pub const DEFAULT_PAD_TOKEN_ID: u32 = 0;
/// Dev/test convenience source for the tokenizer path: the runtime env
/// (`GREPPY_EMBEDDINGGEMMA_TOKENIZER`) rather than a machine-specific
/// hardcoded path, so no local model-cache path is baked into the source.
pub fn plan_tokenizer_path() -> String {
    std::env::var("GREPPY_EMBEDDINGGEMMA_TOKENIZER").unwrap_or_default()
}

/// EmbeddingGemma task prompt selection. The string forms intentionally match
/// the model's SentenceTransformer prompt table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedTask {
    RetrievalQuery,
    RetrievalDocument,
    CodeRetrievalQuery,
    QuestionAnswering,
    FactVerification,
    Classification,
    Clustering,
    SentenceSimilarity,
}

impl EmbedTask {
    pub fn prompt(self, content: &str) -> String {
        match self {
            Self::RetrievalQuery => format!("task: search result | query: {content}"),
            Self::RetrievalDocument => format!("title: none | text: {content}"),
            Self::CodeRetrievalQuery => format!("task: code retrieval | query: {content}"),
            Self::QuestionAnswering => format!("task: question answering | query: {content}"),
            Self::FactVerification => format!("task: fact checking | query: {content}"),
            Self::Classification => format!("task: classification | query: {content}"),
            Self::Clustering => format!("task: clustering | query: {content}"),
            Self::SentenceSimilarity => format!("task: sentence similarity | query: {content}"),
        }
    }

    pub fn document_with_title(title: Option<&str>, content: &str) -> String {
        format!("title: {} | text: {content}", title.unwrap_or("none"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub max_length: usize,
    pub pad_token_id: u32,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: DEFAULT_MAX_LENGTH,
            pad_token_id: DEFAULT_PAD_TOKEN_ID,
        }
    }
}

impl TokenizerConfig {
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let max_length = model.metadata_u32("gemma-embedding.context_length")? as usize;
        if max_length == 0 {
            return Err(Error::InvalidGguf(
                "gemma-embedding.context_length must be non-zero".into(),
            ));
        }
        Ok(Self {
            max_length,
            pad_token_id: model.metadata_u32("tokenizer.ggml.padding_token_id")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedBatch {
    pub token_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
}

impl TokenizedBatch {
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    pub fn batch_size(&self) -> usize {
        self.token_ids.len()
    }

    pub fn seq_len(&self) -> usize {
        self.token_ids.first().map(Vec::len).unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct PromptTokenizer {
    tokenizer: Tokenizer,
    raw_tokenizer: Tokenizer,
    config: TokenizerConfig,
}

impl PromptTokenizer {
    pub fn from_file(path: impl AsRef<Path>, config: TokenizerConfig) -> Result<Self> {
        let mut tokenizer =
            Tokenizer::from_file(path).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut raw_tokenizer = tokenizer.clone();
        // Serialized tokenizer settings must never truncate or pad raw counts.
        raw_tokenizer.with_padding(None);
        raw_tokenizer
            .with_truncation(None)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        configure_tokenizer(&mut tokenizer, config)?;
        Ok(Self {
            tokenizer,
            raw_tokenizer,
            config,
        })
    }

    pub fn from_plan_path() -> Result<Self> {
        Self::from_file(plan_tokenizer_path(), TokenizerConfig::default())
    }

    pub fn encode_task(&self, task: EmbedTask, content: &str) -> Result<TokenizedBatch> {
        let prompt = task.prompt(content);
        self.encode_prompts([prompt])
    }

    /// Encode a single prompt exactly as text, without padding or truncation.
    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        self.raw_tokenizer
            .encode(text, true)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Encode a text FRAGMENT: no special tokens. A line inside a window is not
    /// a standalone prompt; per-line BOS/EOS would put two foreign tokens into
    /// every line span the head pools over.
    pub fn encode_fragment_ids(&self, text: &str) -> Result<Vec<u32>> {
        self.raw_tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Number of tokens in a single prompt without padding or truncation.
    pub fn token_len(&self, text: &str) -> Result<usize> {
        self.encode_ids(text).map(|ids| ids.len())
    }

    pub fn max_length(&self) -> usize {
        self.config.max_length
    }

    /// Encode without allowing either the byte pre-limit or token truncation to
    /// alter a source-spanned head input. Compare actual nonpadding token IDs.
    pub fn encode_prompts_exact<S, I>(&self, prompts: I) -> Result<TokenizedBatch>
    where
        S: AsRef<str>,
        I: IntoIterator<Item = S>,
    {
        let prompts = prompts.into_iter().collect::<Vec<_>>();
        let expected = prompts
            .iter()
            .map(|p| self.encode_ids(p.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        if expected
            .iter()
            .any(|ids| ids.is_empty() || ids.len() > self.config.max_length)
        {
            return Err(Error::Tokenizer(
                "exact head input exceeds native token limit or is empty".into(),
            ));
        }
        let batch = self.encode_prompts(prompts.iter().map(|p| p.as_ref()))?;
        if batch.batch_size() != expected.len() {
            return Err(Error::Tokenizer(
                "exact head input batch size changed".into(),
            ));
        }
        for ((ids, mask), expected) in batch
            .token_ids
            .iter()
            .zip(&batch.attention_mask)
            .zip(expected)
        {
            let actual = ids
                .iter()
                .zip(mask)
                .filter_map(|(id, active)| (*active == 1).then_some(*id))
                .collect::<Vec<_>>();
            if actual != expected || mask.iter().any(|v| *v > 1) {
                return Err(Error::Tokenizer(
                    "exact head input was altered by tokenizer truncation or padding".into(),
                ));
            }
        }
        Ok(batch)
    }

    pub fn encode_prompts<S, I>(&self, prompts: I) -> Result<TokenizedBatch>
    where
        S: AsRef<str>,
        I: IntoIterator<Item = S>,
    {
        let prompts = prompts.into_iter().collect::<Vec<_>>();
        if prompts.is_empty() {
            return Ok(TokenizedBatch {
                token_ids: Vec::new(),
                attention_mask: Vec::new(),
            });
        }

        let truncated = prompts
            .iter()
            .map(|prompt| self.truncate_prompt(prompt.as_ref()))
            .collect::<Vec<_>>();
        let prompt_refs = truncated
            .iter()
            .map(|prompt| prompt.as_ref())
            .collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(prompt_refs, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let seq_len = encodings
            .first()
            .ok_or_else(|| Error::Tokenizer("tokenizer returned empty batch".into()))?
            .len();
        let mut token_ids = Vec::with_capacity(encodings.len());
        let mut attention_mask = Vec::with_capacity(encodings.len());
        for enc in encodings {
            if enc.len() != seq_len {
                return Err(Error::Tokenizer(
                    "tokenizer padding failed to produce a rectangular batch".into(),
                ));
            }
            token_ids.push(enc.get_ids().to_vec());
            attention_mask.push(enc.get_attention_mask().to_vec());
        }
        Ok(TokenizedBatch {
            token_ids,
            attention_mask,
        })
    }

    fn truncate_prompt<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        let max_bytes = self.config.max_length.saturating_mul(128).max(8 * 1024);
        if prompt.len() <= max_bytes {
            return Cow::Borrowed(prompt);
        }
        let mut end = max_bytes;
        while end > 0 && !prompt.is_char_boundary(end) {
            end -= 1;
        }
        Cow::Owned(prompt[..end].to_string())
    }
}

fn configure_tokenizer(tokenizer: &mut Tokenizer, config: TokenizerConfig) -> Result<()> {
    if config.max_length == 0 {
        return Err(Error::Tokenizer(
            "max tokenizer length must be non-zero".into(),
        ));
    }
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_id: config.pad_token_id,
            pad_type_id: 0,
            pad_token: "<pad>".into(),
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length: config.max_length,
            direction: TruncationDirection::Right,
            ..Default::default()
        }))
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    Ok(())
}
