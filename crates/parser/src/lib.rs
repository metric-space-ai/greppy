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
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| greppy_core::Error::Parse("tree-sitter parse returned None".into()))?;
    if !matches!(language, Language::C)
        || !source.windows(13).any(|token| token == b"_Thread_local")
    {
        return Ok(tree);
    }
    // The pinned C grammar recognizes C23 thread_local and GNU __thread, but
    // omits the C11 keyword _Thread_local. Reparse only exact identifier tokens
    // with an equivalent storage-class spelling. Padding keeps every following
    // byte/line offset unchanged; the caller's source is never modified.
    let mut replacements = Vec::new();
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        if matches!(node.kind(), "identifier" | "type_identifier")
            && source.get(node.byte_range()) == Some(b"_Thread_local".as_slice())
        {
            let mut ancestor = node.parent();
            let mut protected = false;
            while let Some(parent) = ancestor {
                if matches!(
                    parent.kind(),
                    "comment"
                        | "string_literal"
                        | "char_literal"
                        | "preproc_arg"
                        | "preproc_def"
                        | "preproc_function_def"
                        | "preproc_call"
                ) {
                    protected = true;
                    break;
                }
                ancestor = parent.parent();
            }
            if !protected {
                replacements.push(node.byte_range());
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                if replacements.is_empty() {
                    return Ok(tree);
                }
                let mut view = source.to_vec();
                for range in replacements {
                    view[range].copy_from_slice(b" thread_local");
                }
                return parser.parse(&view, None).ok_or_else(|| {
                    greppy_core::Error::Parse("tree-sitter C11 parse returned None".into())
                });
            }
        }
    }
}

#[cfg(test)]
mod c11_tests {
    use super::*;

    #[test]
    fn c11_thread_local_storage_keeps_source_and_declaration_offsets() {
        for qualifier in ["", "static ", "extern "] {
            let source = format!(
                "/* _Thread_local */\n{qualifier}_Thread_local int counter=-1;\nconst char *text=\"_Thread_local\";\nint _Thread_local_suffix;\nint main(void){{return counter;}}\n"
            );
            let bytes = source.as_bytes().to_vec();
            let tree = parse(Language::C, &bytes).unwrap();
            assert!(!tree.root_node().has_error(), "{}", tree.root_node().to_sexp());
            let at = source.find("counter").unwrap();
            let name = tree.root_node().descendant_for_byte_range(at, at + 7).unwrap();
            assert_eq!(name.kind(), "identifier");
            assert_eq!(name.utf8_text(&bytes).unwrap(), "counter");
            let at = source.find("_Thread_local_suffix").unwrap();
            let name = tree.root_node().descendant_for_byte_range(at, at + 20).unwrap();
            assert_eq!(name.utf8_text(&bytes).unwrap(), "_Thread_local_suffix");
            assert_eq!(bytes, source.as_bytes());
        }
    }

    #[test]
    fn c11_thread_local_does_not_hide_broken_initializers_or_bodies() {
        for source in [
            "static _Thread_local int counter=;\n",
            "static _Thread_local int counter=-1;\nint main(void){return counter;\n",
            "static _Thread_local int counter=-1\nint main(void){return counter;}\n",
        ] {
            let tree = parse(Language::C, source.as_bytes()).unwrap();
            assert!(tree.root_node().has_error(), "{source}");
        }
    }
}
