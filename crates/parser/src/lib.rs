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
        return if matches!(language, Language::C) {
            parse_c_va_arg(&mut parser, source, tree)
        } else {
            Ok(tree)
        };
    }
    // The pinned C grammar recognizes C23 thread_local and GNU __thread, but
    // omits the C11 keyword _Thread_local. Reparse only exact identifier tokens
    // with an equivalent storage-class spelling. Padding keeps every following
    // byte/line offset unchanged; the caller's source is never modified.
    let mut replacements = Vec::new();
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        // An explicit macro definition owns this spelling; do not reinterpret
        // its uses as a built-in storage class without preprocessing evidence.
        if matches!(node.kind(), "preproc_def" | "preproc_function_def")
            && node.child_by_field_name("name").is_some_and(|name| {
                source.get(name.byte_range()) == Some(b"_Thread_local".as_slice())
            })
        {
            drop(cursor);
            return parse_c_va_arg(&mut parser, source, tree);
        }
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
                drop(cursor);
                if replacements.is_empty() {
                    return parse_c_va_arg(&mut parser, source, tree);
                }
                let mut view = source.to_vec();
                for range in replacements {
                    view[range].copy_from_slice(b" thread_local");
                }
                let tree = parser.parse(&view, None).ok_or_else(|| {
                    greppy_core::Error::Parse("tree-sitter C11 parse returned None".into())
                })?;
                return parse_c_va_arg(&mut parser, &view, tree);
            }
        }
    }
}

/// The C grammar treats all call operands as expressions. Validate the standard
/// va_arg type operand with its type_descriptor rule before replacing just that
/// operand in a same-width parse view. This is syntax validation, not C semantic
/// checking (typedef resolution, completeness and promotions require a compiler).
fn parse_c_va_arg(parser: &mut Parser, source: &[u8], tree: Tree) -> Result<Tree> {
    if !source.windows(6).any(|s| s == b"va_arg") {
        return Ok(tree);
    }
    let mut names = Vec::new();
    let mut bound = false;
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        if matches!(node.kind(), "preproc_def" | "preproc_function_def")
            && node
                .child_by_field_name("name")
                .is_some_and(|name| source.get(name.byte_range()) == Some(b"va_arg".as_slice()))
        {
            bound = true;
        }
        if node.kind() == "identifier"
            && source.get(node.byte_range()) == Some(b"va_arg".as_slice())
            && node.parent().is_some_and(|parent| {
                parent.kind() == "call_expression"
                    && parent.child_by_field_name("function") == Some(node)
            })
        {
            names.push(node.end_byte());
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                drop(cursor);
                if bound {
                    return Ok(tree);
                }
                let mut view = source.to_vec();
                let mut changed = false;
                for end in names {
                    let Some(range) = c_va_arg_type_range(source, end) else {
                        continue;
                    };
                    let prefix = b"void f(void){sizeof(";
                    let mut probe = prefix.to_vec();
                    probe.extend_from_slice(&source[range.clone()]);
                    probe.extend_from_slice(b");}");
                    let Some(probe_tree) = parser.parse(&probe, None) else {
                        continue;
                    };
                    if probe_tree.root_node().has_error() {
                        continue;
                    }
                    let Some(operand) = probe_tree
                        .root_node()
                        .descendant_for_byte_range(prefix.len(), prefix.len() + range.len())
                    else {
                        continue;
                    };
                    // Require a type, rather than accepting sizeof(expression).
                    if operand.kind() != "type_descriptor" {
                        continue;
                    }
                    // Preserve newlines as well as bytes; surrounding symbols and
                    // the va_arg callee keep their original source identities.
                    for byte in &mut view[range.clone()] {
                        if !matches!(*byte, b'\n' | b'\r') {
                            *byte = b' ';
                        }
                    }
                    view[range.start] = b'0';
                    changed = true;
                }
                return if changed {
                    parser.parse(&view, None).ok_or_else(|| {
                        greppy_core::Error::Parse("tree-sitter va_arg parse returned None".into())
                    })
                } else {
                    Ok(tree)
                };
            }
        }
    }
}

/// Find exactly two balanced call operands, skipping comments and literals.
/// The first operand remains subject to the ordinary expression grammar.
fn c_va_arg_type_range(source: &[u8], mut at: usize) -> Option<std::ops::Range<usize>> {
    while source.get(at).is_some_and(u8::is_ascii_whitespace) {
        at += 1;
    }
    if source.get(at) != Some(&b'(') {
        return None;
    }
    let mut stack = vec![b')'];
    let mut comma = None;
    at += 1;
    while at < source.len() {
        match source[at] {
            b'/' if source.get(at + 1) == Some(&b'*') => {
                at += 2;
                while at + 1 < source.len() && &source[at..at + 2] != b"*/" {
                    at += 1;
                }
                if at + 1 == source.len() {
                    return None;
                }
                at += 2;
                continue;
            }
            b'/' if source.get(at + 1) == Some(&b'/') => {
                while at < source.len() && source[at] != b'\n' {
                    at += 1;
                }
                continue;
            }
            quote @ (b'\'' | b'"') => {
                at += 1;
                while at < source.len() && source[at] != quote {
                    if source[at] == b'\\' {
                        at += 1;
                    }
                    at += 1;
                }
                if at == source.len() {
                    return None;
                }
            }
            b'(' => stack.push(b')'),
            b'[' => stack.push(b']'),
            b'{' => stack.push(b'}'),
            b')' | b']' | b'}' => {
                if stack.pop() != Some(source[at]) {
                    return None;
                }
                if stack.is_empty() {
                    let mut start = comma? + 1;
                    let mut end = at;
                    while start < end && source[start].is_ascii_whitespace() {
                        start += 1;
                    }
                    while end > start && source[end - 1].is_ascii_whitespace() {
                        end -= 1;
                    }
                    return (start < end).then_some(start..end);
                }
            }
            b',' if stack.len() == 1 => {
                if comma.replace(at).is_some() {
                    return None;
                }
            }
            _ => {}
        }
        at += 1;
    }
    None
}

#[cfg(test)]
mod va_arg_tests {
    use super::*;

    #[test]
    fn standard_type_operands_preserve_calls_and_source_locations() {
        for ty in [
            "void *",
            "const char *",
            "unsigned long",
            "struct item *",
            "int (*)(int)",
        ] {
            let source = format!(
                "#include <stdarg.h>\nstruct item {{ int value; }};\nvoid *get(int key,...) {{va_list ap;va_start(ap,key);void *p=va_arg(ap,{ty});va_end(ap);return p;}}\n"
            );
            let bytes = source.as_bytes().to_vec();
            let tree = parse(Language::C, &bytes).unwrap();
            assert!(
                !tree.root_node().has_error(),
                "{ty}: {}",
                tree.root_node().to_sexp()
            );
            for name in ["get", "va_start", "va_arg", "va_end"] {
                let at = source.find(name).unwrap();
                let node = tree
                    .root_node()
                    .descendant_for_byte_range(at, at + name.len())
                    .unwrap();
                assert_eq!(node.kind(), "identifier", "{name}");
                assert_eq!(node.utf8_text(&bytes).unwrap(), name);
                assert_eq!(node.start_byte(), at);
            }
            let at = source.find("va_arg(ap").unwrap() + 7;
            let operand = tree
                .root_node()
                .descendant_for_byte_range(at, at + 2)
                .unwrap();
            assert_eq!(operand.kind(), "identifier");
            assert_eq!(operand.utf8_text(&bytes).unwrap(), "ap");
            assert_eq!(bytes, source.as_bytes());
        }
    }

    #[test]
    fn malformed_operands_and_surrounding_syntax_remain_errors() {
        for call in [
            "va_arg(ap,void *+)",
            "va_arg(ap,int (*)",
            "va_arg(ap,void *,int)",
            "va_arg(,void *)",
            "va_arg(ap,int +)",
        ] {
            let source = format!("void *get(void){{return {call};}}");
            assert!(
                parse(Language::C, source.as_bytes())
                    .unwrap()
                    .root_node()
                    .has_error(),
                "{source}"
            );
        }
        for source in [
            "void *get(void){void *p=va_arg(ap,void *);return ; broken = ;}",
            "void *get(void){return va_arg(ap,void *);",
        ] {
            assert!(
                parse(Language::C, source.as_bytes())
                    .unwrap()
                    .root_node()
                    .has_error(),
                "{source}"
            );
        }
    }

    #[test]
    fn explicit_macro_binding_and_other_calls_are_not_reinterpreted() {
        for source in [
            "#define va_arg(a,b) custom(a,b)\nvoid *get(void){return va_arg(ap,void *);}",
            "#define va_arg custom\nvoid *get(void){return va_arg(ap,void *);}",
            "void *get(void){return custom(ap,void *);}",
        ] {
            let mut raw = Parser::new();
            raw.set_language(&Language::C.grammar()).unwrap();
            let original = raw.parse(source, None).unwrap();
            let actual = parse(Language::C, source.as_bytes()).unwrap();
            assert_eq!(actual.root_node().to_sexp(), original.root_node().to_sexp());
        }
    }

    #[test]
    fn thread_local_compatibility_composes_with_va_arg() {
        let source = b"_Thread_local int counter;\nvoid *get(void){return va_arg(ap,void *);}";
        assert!(!parse(Language::C, source).unwrap().root_node().has_error());
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
            assert!(
                !tree.root_node().has_error(),
                "{}",
                tree.root_node().to_sexp()
            );
            let at = source.find("counter").unwrap();
            let name = tree
                .root_node()
                .descendant_for_byte_range(at, at + 7)
                .unwrap();
            assert_eq!(name.kind(), "identifier");
            assert_eq!(name.utf8_text(&bytes).unwrap(), "counter");
            let at = source.find("_Thread_local_suffix").unwrap();
            let name = tree
                .root_node()
                .descendant_for_byte_range(at, at + 20)
                .unwrap();
            assert_eq!(name.utf8_text(&bytes).unwrap(), "_Thread_local_suffix");
            assert_eq!(bytes, source.as_bytes());
        }
    }

    #[test]
    fn c11_thread_local_macro_binding_remains_opaque() {
        let source = b"#define _Thread_local int\n_Thread_local counter;\n";
        let tree = parse(Language::C, source).unwrap();
        assert!(
            !tree.root_node().has_error(),
            "{}",
            tree.root_node().to_sexp()
        );
        let at = source
            .windows(7)
            .position(|word| word == b"counter")
            .unwrap();
        assert_eq!(
            tree.root_node()
                .descendant_for_byte_range(at, at + 7)
                .unwrap()
                .utf8_text(source)
                .unwrap(),
            "counter"
        );
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
