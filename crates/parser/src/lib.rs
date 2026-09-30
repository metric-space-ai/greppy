//! `greppy-parser` — tree-sitter based AST extraction.
//!
//! Implements:
//! - A small [`Language`] registry mapping language names to tree-sitter
//!   grammars. Currently ships Rust; other languages are explicitly
//!   reported as `unsupported` (or omitted from `supported()`).
//! - A [`Parser`] wrapper around `tree_sitter::Parser` that takes bytes,
//!   parses with the chosen grammar, and exposes the root [`tree_sitter::Tree`].
//! - Per-language extraction passes: definitions, imports, calls.
//!   Each pass returns a `Vec<ExtractedNode>` or `Vec<ExtractedEdge>` so the
//!   indexer can pipe them into the store.

#![deny(rust_2018_idioms)]
// The per-language `src/langs/*.rs` modules carry rich doc comments with
// indented AST/grammar sketches; clippy's pedantic doc-list-indentation lint
// flags that cosmetic style. It is not a correctness signal here.
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::doc_lazy_continuation)]

pub mod extract;
pub mod grounded_hint;
pub mod langs;
pub mod language;
pub mod provider;
pub mod query;
pub mod registry;
pub mod spec;

pub use extract::{extract, ExtractedEdge, ExtractedNode, ExtractionResult};
pub use language::{language_for_path, Language, SUPPORTED_LANGUAGES};
pub use provider::{
    manifest_for_language, EdgeClass, ProviderContractError, ProviderEdge, ProviderManifest,
    ProviderNode, ProviderOutput, ProviderStatus,
};
pub use query::{CompiledQuery, QueryKind};
pub use registry::LangDef;

use greppy_core::Result;
use tree_sitter::{Parser, Tree};

/// Parse `source` as `language`. Returns the parse tree.
///
/// On any tree-sitter error, returns
/// `greppy_core::Error::Store(format!("tree-sitter: ..."))`.
pub fn parse(language: Language, source: &[u8]) -> Result<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&language.grammar())
        .map_err(|e| greppy_core::Error::Parse(format!("set_language: {e}")))?;
    parser
        .parse(source, None)
        .ok_or_else(|| greppy_core::Error::Parse("tree-sitter parse returned None".into()))
}
