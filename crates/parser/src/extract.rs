//! Extraction passes: walk a parse tree with a compiled query and
//! produce `ExtractedNode` / `ExtractedEdge` values the indexer can
//! pipe into the store.
//!
//! Single-file scope: emits real `ExtractedEdge` entries for `CALLS`
//! (caller function → callee name) and `IMPORTS` (file → imported path).
//! The indexer resolves the qnames to node ids and persists the edges.
//!
//! Method qnames include the enclosing
//! `impl`/`trait` type so two methods named `new` on different impls
//! do not collide on `{file}::Function::new`.

use serde::{Deserialize, Serialize};
use tree_sitter::{Node, QueryCursor, StreamingIterator};

use crate::language::Language;
use crate::query::QueryKind;

/// One graph node extracted from source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedNode {
    pub label: String,
    pub name: String,
    pub qualified_name: String,
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub properties: serde_json::Value,
}

/// One graph edge extracted from source.
///
/// `source_qualified_name` and `target_qualified_name` refer to
/// `qualified_name` values the indexer must look up in the store.
/// For `CALLS`, the source is the enclosing function and the target
/// is the callee's qname (resolved within the same file, per the
/// 2026-06-29 single-file scope-call). For `IMPORTS`, the source is
/// the file-level synthetic qname `<file>::__file__` and the target
/// is the use-tree path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedEdge {
    pub edge_type: String,
    pub source_qualified_name: String,
    pub target_qualified_name: String,
    pub file_path: String,
    pub line: u32,
    pub properties: serde_json::Value,
}

/// All extractions from one file.
#[derive(Debug, Clone, Default)]
pub struct ExtractionResult {
    pub nodes: Vec<ExtractedNode>,
    pub edges: Vec<ExtractedEdge>,
}

impl ExtractionResult {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    pub fn extend(&mut self, other: ExtractionResult) {
        self.nodes.extend(other.nodes);
        self.edges.extend(other.edges);
    }
}

/// Run all extraction passes for `language` over `source`.
///
/// Currently implements Rust only; other languages return an explicit
/// `Error::NotImplemented`.
pub fn extract(
    language: Language,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    match language {
        Language::Rust => extract_rust(source, file_path),
        Language::Python => extract_python(source, file_path),
        Language::JavaScript => extract_js_ts(language, source, file_path),
        Language::TypeScript { .. } => extract_js_ts(language, source, file_path),
        Language::Go => extract_go(source, file_path),
        Language::Ruby => extract_ruby(source, file_path),
        Language::Java => extract_java(source, file_path),
        Language::C => extract_c_cpp(Language::C, source, file_path),
        Language::Cpp => extract_c_cpp(Language::Cpp, source, file_path),
        Language::CSharp => extract_csharp(source, file_path),
        Language::Php => extract_php(source, file_path),
        Language::Bash => extract_bash(source, file_path),
        Language::Lua => extract_lua(source, file_path),
        Language::Kotlin => extract_kotlin(source, file_path),
        Language::Scala => extract_scala(source, file_path),
        Language::Swift => extract_swift(source, file_path),
        Language::Zig => extract_zig(source, file_path),
        Language::R => extract_r(source, file_path),
        Language::Registered(d) => {
            // Elixir's tree-sitter grammar models every `def`/`defp`/`defmodule`
            // as a generic `call` node, so the uniform spec template cannot express
            // a richer taxonomy (defmodule → Class, def/defp/defmacro → Function, and a
            // bare-name CALLS pass sourced from the file module). A bespoke pass —
            // reaches full coverage where the generic path only emits coarse Functions.
            if d.name == "elixir" {
                return extract_elixir(source, file_path);
            }
            // Same rationale as elixir: the generic spec path only emits coarse
            // def-nodes for these grammars; a bespoke pass reaches full coverage.
            if d.name == "ocaml" {
                return extract_ocaml(d, source, file_path);
            }
            if d.name == "julia" {
                return extract_julia(language, d, source, file_path);
            }
            if d.name == "haskell" {
                return extract_haskell(language, d, source, file_path);
            }
            if d.name == "dart" {
                return extract_dart(language, d, source, file_path);
            }
            // Clojure's tree-sitter grammar (`tree-sitter-clojure-orchard`)
            // models every form as a generic `list_lit`, so the uniform spec
            // template mislabels every def-form as a coarse `Function` and its
            // CALLS/IMPORTS passes emit nothing. A bespoke pass labels `defrecord`/`deftype` → `Struct`,
            // `defprotocol`/`definterface` → `Interface`, every other def head
            // → `Function`, emits `CALLS` from the file Module per applied
            // symbol, and `IMPORTS` per `(ns .. (:require ..))` clause entry.
            if d.name == "clojure" {
                return extract_clojure(d, source, file_path);
            }
            // Racket's `tree-sitter-racket` grammar is a generic s-expression
            // grammar: every parenthesised form is a `list`, every atom a
            // `symbol`, and there is NO `name:` field or dedicated def node. The
            // uniform spec template captures `(define ...)` forms as coarse
            // `Function`s but misses `(struct ..)` / `(define-struct ..)` (C's
            // Struct heads) and emits no resolved CALLS (they are sourced from the
            // file Module, which the generic enclosing-callable resolution
            // cannot name). A bespoke pass handles `struct` / `define-struct` /
            // `define-record-type` → `Struct`, every other def head →
            // `Function`, and one `CALLS` from the file Module per applied
            // symbol (same-file target, cross-file callee_name fallback).
            if d.name == "racket" {
                return extract_racket(d, source, file_path);
            }
            // D's tree-sitter grammar exposes clean, distinct def kinds, but the
            // generic spec path only emits coarse Struct/Class def-nodes and no
            // resolved CALLS/USAGE/IMPORTS. A bespoke pass covers the full D
            // taxonomy: struct/class/union → Class, interface → Interface, enum →
            // Enum, every `function_declaration` (free or method) → a free
            // Function keyed `{module}.{name}` (no owner segment, so same-named
            // methods across types collapse when the store dedups), and
            // Module-sourced CALLS / USAGE plus File-resolving IMPORTS.
            if d.name == "d" {
                return extract_d(d, source, file_path);
            }
            // Same registry-path rationale: a bespoke pass (in the style of
            // extract_ruby/extract_ocaml) resolves the full edge set where the
            // generic spec path only emits coarse def-nodes.
            if d.name == "crystal" {
                return extract_crystal(d, source, file_path);
            }
            if d.name == "elm" {
                return extract_elm(language, d, source, file_path);
            }
            if d.name == "erlang" {
                return extract_erlang(language, d, source, file_path);
            }
            if d.name == "fsharp" {
                return extract_fsharp(d, source, file_path);
            }
            // Solidity's grammar exposes clean, distinct def kinds, but the
            // taxonomy needs a bespoke pass to (a) relabel
            // contract/library/struct → Class and interface → Interface, (b)
            // double-count state vars (Field + Variable) and struct members
            // (Field only), (c) double-count owned functions/modifiers
            // (Method + Function) while emitting free functions once, (d) emit
            // DEFINES_METHOD / CALLS / USAGE / INHERITS / IMPLEMENTS. The
            // generic spec path only emits coarse Contract/Library/Struct
            // def-nodes; this pass resolves the full edge set.
            if d.name == "solidity" {
                return extract_solidity(d, source, file_path);
            }
            if d.name == "gleam" {
                return extract_gleam(language, d, source, file_path);
            }
            if d.name == "groovy" {
                return extract_groovy(d, source, file_path);
            }
            if d.name == "purescript" {
                return extract_purescript(language, d, source, file_path);
            }
            if d.name == "fortran" {
                return extract_fortran(d, source, file_path);
            }
            if d.name == "scheme" {
                return extract_scheme(d, source, file_path);
            }
            // Objective-C's tree-sitter grammar is C-derived, but its
            // `@interface`/`@implementation`/`@protocol`/`method_definition`
            // nodes carry their name on an anonymous `identifier` child (no
            // `name:` field), so the uniform spec template cannot resolve them.
            // A bespoke pass handles the Objective-C def / method / usage work:
            // `class_interface`/`class_implementation` → "Class" (collapsed by
            // qname), `protocol_declaration` → "Interface",
            // `method_definition` (inside `@implementation`) → "Method" +
            // DEFINES_METHOD, free `function_definition` emits NO node (no
            // Function/Field/Variable for objc), a `message_expression`
            // selector → CALLS, and every reference identifier → USAGE.
            if d.name == "objc" {
                return extract_objc(language, d, source, file_path);
            }
            let queries = d.compiled_queries().map_err(|e| {
                greppy_core::Error::Parse(format!("compile {} queries: {e}", d.name))
            })?;
            crate::spec::spec_extract(language, d.spec, queries, source, file_path)
        }
        Language::Unsupported(s) => Err(greppy_core::Error::not_implemented(
            "language extraction",
            format!(
                "language {s} is not supported in this build; supported: \
                 [rust, python, javascript, typescript, go, ruby, java, c, cpp, csharp, php, bash, \
                 lua, kotlin, scala, swift, zig, r]"
            ),
        )),
    }
}

/// A flat record of one definition capture plus the impl/trait context
/// we resolved by walking the tree before the query pass. Currently we
/// only consume the resolved qname during PASS 1 (PASS 2 / PASS 3 do
/// their own ancestor walks). The other fields are kept so future
/// per-file diff / cross-reference passes can read the resolved
/// span list — they are not "dead" in the sense of "never used", but
/// `clippy` cannot see that yet.
struct DefinitionSpan {
    /// Symbol's effective label: `Function` or `Method` etc.
    #[allow(dead_code)]
    label: String,
    /// Symbol name as captured.
    #[allow(dead_code)]
    name: String,
    /// What we'll record as the node's qname.
    #[allow(dead_code)]
    qname: String,
    #[allow(dead_code)]
    start_line: u32,
    #[allow(dead_code)]
    end_line: u32,
    /// For functions/methods, the qname of the enclosing function.
    #[allow(dead_code)]
    enclosing_function_qname: Option<String>,
}

/// Return the *owner type* name of an `impl_item`/`trait_item` node — the
/// type the block qualifies methods/assoc-items under.
///
/// For a `trait_item` it is the trait's `name` field. For an `impl_item` it is
/// the `type:` field: crucially, for a *trait impl* (`impl Trait for Type`) the
/// owner is the implementing `Type` (the `type:` field), NOT the trait (the
/// `trait:` field). Reading the `type:` field by name avoids the
/// first-`type_identifier`-child bug where `impl Trait for Type` would
/// otherwise resolve to `Trait`.
fn impl_owner_type<'a>(source: &'a [u8], item: Node<'_>) -> Option<&'a str> {
    match item.kind() {
        "trait_item" => item
            .child_by_field_name("name")
            .map(|n| node_text(source, n)),
        "impl_item" => {
            let ty = item.child_by_field_name("type")?;
            Some(impl_type_name(source, ty))
        }
        _ => None,
    }
}

/// The final name of a `type:` field on an `impl_item`. The grammar may give a
/// bare `type_identifier`, a `scoped_type_identifier` (`a::B`), or a
/// `generic_type` (`Foo<T>`); we want the base type identifier.
fn impl_type_name<'a>(source: &'a [u8], type_node: Node<'_>) -> &'a str {
    match type_node.kind() {
        "type_identifier" => node_text(source, type_node),
        "scoped_type_identifier" => named_child_of_kinds(type_node, &["type_identifier"])
            .map(|n| node_text(source, n))
            .unwrap_or_else(|| node_text(source, type_node)),
        "generic_type" => type_node
            .child_by_field_name("type")
            .map(|n| impl_type_name(source, n))
            .unwrap_or_else(|| node_text(source, type_node)),
        _ => node_text(source, type_node),
    }
}

/// Walk `node`'s ancestors and return the owner type of the first impl/trait
/// block it sits inside, if any. Used to qualify method / associated-item
/// qnames.
fn enclosing_impl_type<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if matches!(cur.kind(), "impl_item" | "trait_item") {
            return impl_owner_type(source, cur);
        }
        p = cur.parent();
    }
    None
}

/// Walk `node`'s ancestors and return the first enclosing
/// `function_item`'s qname (constructed with the same collision-
/// avoiding rules as the definition pass).
fn enclosing_function_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "function_item" {
            // Find the function's name child.
            for i in 0..cur.child_count() {
                if let Some(child) = cur.child(i) {
                    if matches!(child.kind(), "identifier" | "type_identifier") {
                        let name = node_text(source, child);
                        let impl_ctx = enclosing_impl_type(source, cur);
                        return Some(match impl_ctx {
                            Some(t) => format!("{file_path}::{t}::{name}"),
                            None => format!("{file_path}::Function::{name}"),
                        });
                    }
                }
            }
            return None;
        }
        p = cur.parent();
    }
    None
}

/// Resolve the owner type of an unambiguous Rust receiver call when the AST
/// carries enough local type evidence. `self.method()` inherits the enclosing
/// impl owner; named receivers are accepted for explicitly typed parameters or
/// locals and for locals initialized directly from a struct literal. Other
/// inferred values deliberately return `None` so the graph resolver cannot
/// attach the call to an unrelated same-named method.
fn rust_receiver_owner<'a>(source: &'a [u8], callee: Node<'_>) -> Option<&'a str> {
    let field = callee.parent()?;
    if field.kind() != "field_expression" {
        return None;
    }
    let receiver = field.child_by_field_name("value")?;
    let receiver_text = node_text(source, receiver).trim();
    if receiver_text == "self" {
        return enclosing_impl_type(source, callee);
    }
    if receiver.kind() != "identifier" || receiver_text.is_empty() {
        return None;
    }

    let mut ancestor = field.parent();
    while let Some(node) = ancestor {
        if node.kind() == "function_item" {
            let (local_found, local_owner) =
                rust_local_binding_owner(source, callee, receiver_text);
            if local_found {
                return local_owner;
            }
            let parameters = node.child_by_field_name("parameters")?;
            for index in 0..parameters.named_child_count() {
                let Some(parameter) = parameters.named_child(index) else {
                    continue;
                };
                if parameter.kind() != "parameter" {
                    continue;
                }
                let Some(pattern) = parameter.child_by_field_name("pattern") else {
                    continue;
                };
                if node_text(source, pattern).trim() != receiver_text {
                    continue;
                }
                let ty = parameter.child_by_field_name("type")?;
                let mut names = Vec::new();
                type_identifiers_in(source, ty, &mut names);
                return names.into_iter().next();
            }
            return None;
        }
        ancestor = node.parent();
    }
    None
}

/// Return the nearest preceding local binding whose owner is explicit in the
/// syntax. A type annotation is authoritative; a struct literal also names its
/// constructed type directly. Calls and arbitrary expressions are excluded
/// because their return type would require semantic type inference.
fn rust_local_binding_owner<'a>(
    source: &'a [u8],
    callee: Node<'_>,
    receiver_name: &str,
) -> (bool, Option<&'a str>) {
    let mut ancestor = callee.parent();
    while let Some(node) = ancestor {
        if node.kind() == "function_item" {
            break;
        }
        if node.kind() == "block" {
            // Only direct statements in this lexical block are visible here.
            // Searching descendants would incorrectly pick bindings from an
            // already-closed sibling block.
            for index in (0..node.named_child_count()).rev() {
                let Some(binding) = node.named_child(index) else {
                    continue;
                };
                if binding.end_byte() > callee.start_byte() || binding.kind() != "let_declaration" {
                    continue;
                }
                let matches_receiver = binding
                    .child_by_field_name("pattern")
                    .is_some_and(|pattern| node_text(source, pattern).trim() == receiver_name);
                if !matches_receiver {
                    continue;
                }
                if let Some(ty) = binding.child_by_field_name("type") {
                    let mut names = Vec::new();
                    type_identifiers_in(source, ty, &mut names);
                    return (true, names.into_iter().next());
                }
                if let Some(value) = binding.child_by_field_name("value") {
                    if value.kind() == "struct_expression" {
                        let type_node = value
                            .child_by_field_name("name")
                            .or_else(|| value.named_child(0));
                        if let Some(type_node) = type_node {
                            let mut names = Vec::new();
                            type_identifiers_in(source, type_node, &mut names);
                            return (true, names.into_iter().next());
                        }
                    }
                }
                // A local binding shadows any parameter of the same name, but
                // its arbitrary initializer does not reveal a safe owner.
                return (true, None);
            }
        }
        ancestor = node.parent();
    }
    (false, None)
}

/// Walk `node`'s ancestors and return the qname of the nearest enclosing
/// *definition* (function/method, struct, enum, or trait) — whichever the
/// reference sits inside. Used as the `source` endpoint for TYPE_REF and
/// USES edges so they hang off a resolvable symbol.
///
/// Construction mirrors PASS 1's qname scheme so the endpoints resolve
/// against the nodes emitted there.
fn enclosing_def_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        match cur.kind() {
            "function_item" => {
                let name_node = named_child_of_kinds(cur, &["identifier", "type_identifier"])?;
                let name = node_text(source, name_node);
                return Some(match enclosing_impl_type(source, cur) {
                    Some(t) => format!("{file_path}::{t}::{name}"),
                    None => format!("{file_path}::Function::{name}"),
                });
            }
            "struct_item" | "union_item" => {
                let name_node = named_child_of_kinds(cur, &["type_identifier"])?;
                let name = node_text(source, name_node);
                return Some(format!("{file_path}::Class::{name}"));
            }
            "enum_item" => {
                let name_node = named_child_of_kinds(cur, &["type_identifier"])?;
                let name = node_text(source, name_node);
                return Some(format!("{file_path}::Enum::{name}"));
            }
            "trait_item" => {
                let name_node = named_child_of_kinds(cur, &["type_identifier"])?;
                let name = node_text(source, name_node);
                return Some(format!("{file_path}::Interface::{name}"));
            }
            _ => {}
        }
        p = cur.parent();
    }
    None
}

/// The declared type text of a `field_declaration` node (the `type:` field),
/// recorded as the `return_type` property on the emitted `Field` definition.
/// Returns `None` when the field is untyped
/// (the query already requires a `type:` child, so this is a safety net).
fn field_declared_type(source: &[u8], field_decl: Node<'_>) -> Option<String> {
    let ty = field_decl.child_by_field_name("type")?;
    let text = node_text(source, ty).trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Return the first direct child of `node` whose kind is in `kinds`.
fn named_child_of_kinds<'t>(node: Node<'t>, kinds: &[&str]) -> Option<Node<'t>> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if kinds.contains(&child.kind()) {
                return Some(child);
            }
        }
    }
    None
}

/// Strip wrapper syntax from a type-position node and return the base type
/// identifier(s): it descends through `reference_type` (`&T`),
/// `generic_type` (`Vec<T>`),
/// `scoped_type_identifier` (`a::B`), `array_type`/`slice` and tuple wrappers
/// to the inner `type_identifier`(s).
///
/// Returns every concrete `type_identifier` found (so `Result<Foo, Bar>`
/// yields both `Foo` and `Bar`), de-duplicated by the caller as needed.
fn type_identifiers_in<'a>(source: &'a [u8], node: Node<'_>, out: &mut Vec<&'a str>) {
    match node.kind() {
        "type_identifier" => {
            out.push(node_text(source, node));
        }
        "scoped_type_identifier" => {
            // `a::b::Foo` — the final `type_identifier` is the type name.
            if let Some(name) = named_child_of_kinds(node, &["type_identifier"]) {
                out.push(node_text(source, name));
            }
        }
        _ => {
            // Descend into wrapper nodes (reference_type, generic_type,
            // array_type, tuple_type, type_arguments, …).
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    type_identifiers_in(source, child, out);
                }
            }
        }
    }
}

/// Rust primitive / builtin types that must NOT generate TYPE_REF edges
/// (Rust-relevant subset).
fn is_builtin_rust_type(name: &str) -> bool {
    matches!(
        name,
        "u8" | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f32"
            | "f64"
            | "bool"
            | "char"
            | "str"
            | "Self"
    )
}

/// Reserved words / non-reference identifiers that should never count as a
/// USAGE reference. The set is deliberately broader than
/// the Rust language keywords: it also excludes the ubiquitous prelude / std
/// names (`Some`, `None`, `Ok`, `Err`, `Vec`, `String`, `Box`, `Rc`, `Arc`,
/// `Option`, `Result`) and the common macros (`println`, `format`, `assert*`,
/// `derive`, `cfg`, …). Excluding these keeps the USAGE edge count
/// from over-shooting on real corpora that use those names.
fn is_rust_keyword_or_self(name: &str) -> bool {
    matches!(
        name,
        // ── true Rust keywords (reserved + weak/2018) ──
        "as" | "async"
            | "await"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "dyn"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            // ── reserved-for-future / historical keywords ──
            | "abstract"
            | "become"
            | "box"
            | "do"
            | "final"
            | "macro"
            | "override"
            | "priv"
            | "try"
            | "typeof"
            | "unsized"
            | "virtual"
            | "yield"
            // ── prelude enum variants / std types (excluded as keywords) ──
            | "Some"
            | "None"
            | "Ok"
            | "Err"
            | "Vec"
            | "String"
            | "Box"
            | "Rc"
            | "Arc"
            | "Option"
            | "Result"
            // ── common macros (excluded as keywords) ──
            | "println"
            | "eprintln"
            | "format"
            | "write"
            | "writeln"
            | "print"
            | "eprint"
            | "panic"
            | "assert"
            | "assert_eq"
            | "assert_ne"
            | "debug_assert"
            | "todo"
            | "unimplemented"
            | "cfg"
            | "derive"
            // ── common attributes (excluded as keywords) ──
            | "test"
            | "allow"
            | "deny"
            | "warn"
            | "forbid"
            | "deprecated"
    )
}

/// True if `node` is the *name* field of its parent definition (so it is a
/// definition, not a usage).
fn is_definition_name(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    if let Some(name_field) = parent.child_by_field_name("name") {
        return name_field.start_byte() == node.start_byte()
            && name_field.end_byte() == node.end_byte();
    }
    false
}

/// True if `node` has an ancestor whose kind is in `kinds`, within the
/// parent-depth bound used by several usage passes.
fn is_inside_kind(node: Node<'_>, kinds: &[&str]) -> bool {
    const MAX_PARENT_DEPTH: usize = 12;
    let mut cur = node.parent();
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= MAX_PARENT_DEPTH {
            break;
        }
        if kinds.contains(&n.kind()) {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// The Rust reference-node kinds the usage pass recognises: the common
/// `identifier` / `type_identifier`, plus Rust's
/// `field_identifier` and `scoped_identifier`. Any node of
/// one of these kinds that is not inside a call/import and is not a definition
/// name is a candidate usage.
fn is_rust_reference_kind(kind: &str) -> bool {
    matches!(
        kind,
        "identifier"
            | "type_identifier"
            | "field_identifier"
            | "scoped_identifier"
            | "scoped_type_identifier"
    )
}

/// Import ancestor kinds that suppress a Rust usage. Call expressions are
/// handled separately so we can suppress the callee path without dropping
/// argument references such as `make(types::Marker)`.
const RUST_USAGE_IMPORT_SUPPRESSORS: &[&str] = &["use_declaration", "extern_crate_declaration"];

fn node_contains(parent: Node<'_>, child: Node<'_>) -> bool {
    child.start_byte() >= parent.start_byte() && child.end_byte() <= parent.end_byte()
}

fn rust_node_is_call_target(node: Node<'_>, call: Node<'_>) -> bool {
    call.child_by_field_name("function")
        .map(|function| node_contains(function, node))
        .unwrap_or(false)
}

fn rust_reference_leaf<'t>(node: Node<'t>) -> Node<'t> {
    if !matches!(node.kind(), "scoped_identifier" | "scoped_type_identifier") {
        return node;
    }
    let mut cursor = node.walk();
    let mut best = None;
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" | "type_identifier" | "field_identifier" => best = Some(child),
            "scoped_identifier" | "scoped_type_identifier" => {
                best = Some(rust_reference_leaf(child))
            }
            _ => {}
        }
    }
    best.unwrap_or(node)
}

/// True if `node` is inside a callee/import suppressor within the
/// 10-parent bound.
fn rust_usage_is_suppressed(node: Node<'_>) -> bool {
    const MAX_PARENT_DEPTH: usize = 10;
    let mut cur = node.parent();
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= MAX_PARENT_DEPTH {
            break;
        }
        if RUST_USAGE_IMPORT_SUPPRESSORS.contains(&n.kind()) {
            return true;
        }
        if n.kind() == "call_expression" && rust_node_is_call_target(node, n) {
            return true;
        }
        if n.kind() == "macro_invocation" {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// The Rust USAGE walker. Visits every
/// node in the subtree rooted at `node` (pre-order), and for each
/// reference-kind node that is NOT inside a call/import,
/// NOT a definition name, and NOT a keyword, invokes `emit(ref_node, text)`.
///
/// The caller resolves each `ref_name` against the project's registered
/// symbols and keeps only unique matches. Non-resolving references (locals,
/// params with no matching def, etc.) are emitted here but dropped at
/// resolution.
fn walk_rust_usages<'t, F: FnMut(Node<'t>, &str)>(source: &[u8], node: Node<'t>, emit: &mut F) {
    // Try to emit a usage for THIS node.
    if is_rust_reference_kind(node.kind())
        && !rust_usage_is_suppressed(node)
        && !is_definition_name(node)
    {
        let name_node = rust_reference_leaf(node);
        let text = node_text(source, name_node);
        if !text.is_empty() && !is_rust_keyword_or_self(text) {
            emit(name_node, text);
        }
        if matches!(node.kind(), "scoped_identifier" | "scoped_type_identifier") {
            return;
        }
    }
    // Descend into every child.
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            walk_rust_usages(source, child, emit);
        }
    }
}

/// Doc text is truncated to this many bytes.
pub(crate) const MAX_COMMENT_LEN: usize = 500;

/// True if `kind` is a comment node that can carry a Rust doc comment.
/// Rust's grammar emits `line_comment` for `//`-style and `block_comment`
/// for `/* */`-style comments; doc variants (`///`, `/** */`, `//!`, `/*! */`)
/// are the same node kinds with an inner `outer_doc_comment_marker` /
/// `inner_doc_comment_marker` child.
fn is_comment_kind(kind: &str) -> bool {
    matches!(kind, "line_comment" | "block_comment")
}

/// True if `comment` is a *doc* comment (`///`, `/** */`, `//!`, `/*! */`)
/// rather than an ordinary comment. Detected by the presence of a doc-marker
/// child, which the tree-sitter-rust grammar only produces for doc comments.
fn is_doc_comment(comment: Node<'_>) -> bool {
    for i in 0..comment.child_count() {
        if let Some(child) = comment.child(i) {
            if matches!(
                child.kind(),
                "outer_doc_comment_marker" | "inner_doc_comment_marker"
            ) {
                return true;
            }
        }
    }
    false
}

/// Pull the human-readable text out of a single Rust doc comment node,
/// stripping the leading marker (`///`, `//!`, `/**`, `/*!`), the trailing
/// `*/`, and per-line `*` / `///` prefixes, normalising Rust's marker
/// syntax first.
fn doc_comment_text(source: &[u8], comment: Node<'_>) -> String {
    let raw = node_text(source, comment);
    let mut out = String::new();
    if raw.starts_with("/*") {
        // Block comment: drop `/**` / `/*!` / `/*` opener and `*/` closer,
        // then strip a leading `*` from each interior line.
        let inner = raw
            .trim_start_matches("/**")
            .trim_start_matches("/*!")
            .trim_start_matches("/*")
            .trim_end_matches("*/");
        for line in inner.lines() {
            let line = line.trim();
            let line = line.strip_prefix('*').unwrap_or(line).trim();
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
        }
    } else {
        // Line comment(s): strip `///` / `//!` / `//` from each line.
        for line in raw.lines() {
            let line = line.trim();
            let line = line
                .strip_prefix("///")
                .or_else(|| line.strip_prefix("//!"))
                .or_else(|| line.strip_prefix("//"))
                .unwrap_or(line)
                .trim();
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
        }
    }
    out
}

/// Find the doc comment attached to definition `node` and return its text.
///
/// Walk *backwards* over the immediately
/// preceding siblings, collecting consecutive doc-comment lines (so a block of
/// `///` lines becomes one docstring), stopping at the first non-comment
/// sibling. Returns `None` when there is no leading doc comment. The result is
/// truncated to `MAX_COMMENT_LEN` bytes (on a char boundary).
fn extract_docstring(source: &[u8], node: Node<'_>) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut prev = node.prev_sibling();
    while let Some(cur) = prev {
        if !is_comment_kind(cur.kind()) || !is_doc_comment(cur) {
            break;
        }
        lines.push(doc_comment_text(source, cur));
        prev = cur.prev_sibling();
    }
    if lines.is_empty() {
        return None;
    }
    // We collected nearest-first; reverse to source order.
    lines.reverse();
    let mut doc = lines.join("\n").trim().to_string();
    if doc.is_empty() {
        return None;
    }
    if doc.len() > MAX_COMMENT_LEN {
        // Truncate on a char boundary at/under the limit.
        let mut end = MAX_COMMENT_LEN;
        while end > 0 && !doc.is_char_boundary(end) {
            end -= 1;
        }
        doc.truncate(end);
    }
    Some(doc)
}

/// The first non-empty line of a docstring — its summary. Used for the node's
/// `doc` property so list/search views show a one-line summary.
pub(crate) fn docstring_summary(doc: &str) -> &str {
    doc.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
}

/// One captured function/method parameter: its binding name (or `self`) and
/// the textual type annotation.
#[derive(Debug, Clone, Serialize)]
struct ParamInfo {
    name: String,
    #[serde(rename = "type")]
    ty: String,
}

/// The pieces of a function/method signature we attach as node properties:
/// the `signature`, per-parameter info, and the `return_type`.
/// `signature` is the parameter list text plus the
/// return type (the human-readable signature line, body excluded).
struct SignatureInfo {
    signature: String,
    params: Vec<ParamInfo>,
    return_type: Option<String>,
}

/// Read the `parameters` and `return_type` of a `function_item` and build a
/// [`SignatureInfo`]. Returns `None` if there is no parameter list (e.g. a
/// malformed node); a parameterless `fn f()` yields an empty `params` list and
/// a `signature` of `"()"` (plus `-> T` when a return type is present).
fn signature_info(source: &[u8], func: Node<'_>) -> Option<SignatureInfo> {
    let params_node = func.child_by_field_name("parameters")?;
    let params_text = node_text(source, params_node);

    let mut params: Vec<ParamInfo> = Vec::new();
    for i in 0..params_node.named_child_count() {
        let Some(p) = params_node.named_child(i) else {
            continue;
        };
        match p.kind() {
            // `self`, `&self`, `&mut self` — a receiver, no explicit type node.
            "self_parameter" => {
                params.push(ParamInfo {
                    name: "self".to_string(),
                    ty: node_text(source, p).to_string(),
                });
            }
            "parameter" => {
                let name = p
                    .child_by_field_name("pattern")
                    .map(|n| node_text(source, n).to_string())
                    .unwrap_or_default();
                let ty = p
                    .child_by_field_name("type")
                    .map(|n| node_text(source, n).to_string())
                    .unwrap_or_default();
                params.push(ParamInfo { name, ty });
            }
            // Variadic (`...`) and other forms: keep the raw text as the type so
            // the param count stays faithful, with no binding name.
            _ => {
                params.push(ParamInfo {
                    name: String::new(),
                    ty: node_text(source, p).to_string(),
                });
            }
        }
    }

    let return_type = func
        .child_by_field_name("return_type")
        .map(|n| node_text(source, n).to_string());

    let signature = match &return_type {
        Some(rt) => format!("{params_text} -> {rt}"),
        None => params_text.to_string(),
    };

    Some(SignatureInfo {
        signature,
        params,
        return_type,
    })
}

/// Return the complete source-level function signature, including visibility,
/// modifiers, name, generics, parameters, return type, and where-clause. The
/// body opener and a declaration-only trailing semicolon are excluded.
fn function_source_signature(source: &[u8], func: Node<'_>) -> Option<String> {
    let end = func
        .child_by_field_name("body")
        .map(|body| body.start_byte())
        .unwrap_or_else(|| func.end_byte());
    let raw = std::str::from_utf8(source.get(func.start_byte()..end)?).ok()?;
    let joined = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let signature = joined.trim().trim_end_matches(';').trim();
    (!signature.is_empty()).then(|| signature.to_string())
}

/// Modifier flags captured off a `function_item`
/// (`visibility` + `async`/`unsafe`/`const`).
#[derive(Default)]
struct ModifierInfo {
    /// `pub`, `pub(crate)`, `pub(super)`, … — the full visibility text, or
    /// `None` for a private (no-modifier) item.
    visibility: Option<String>,
    is_async: bool,
    is_unsafe: bool,
    is_const: bool,
}

/// Read the `visibility_modifier` and `function_modifiers` children of a def
/// node into a [`ModifierInfo`]. Works for any item kind: structs/traits/enums
/// carry only a `visibility_modifier`; functions also carry a
/// `function_modifiers` node with `async`/`unsafe`/`const` keyword children.
fn modifier_info(source: &[u8], item: Node<'_>) -> ModifierInfo {
    let mut info = ModifierInfo::default();
    for i in 0..item.child_count() {
        let Some(child) = item.child(i) else { continue };
        match child.kind() {
            "visibility_modifier" => {
                info.visibility = Some(node_text(source, child).to_string());
            }
            "function_modifiers" => {
                for j in 0..child.child_count() {
                    if let Some(m) = child.child(j) {
                        match m.kind() {
                            "async" => info.is_async = true,
                            "unsafe" => info.is_unsafe = true,
                            "const" => info.is_const = true,
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    info
}

/// One generic bound: a type parameter constrained by a trait, captured so the
/// resolver can link `fn f<T: Trait>`
/// (or a `where T: Trait`) to the bound trait.
struct GenericBound {
    /// The constrained type parameter (`T`).
    type_param: String,
    /// The bound trait's base name (`Trait`), generic args stripped.
    bound: String,
}

/// Collect every `type_param: Trait` constraint from a `function_item`'s
/// `type_parameters` list and its `where_clause`. Lifetimes, plain
/// (unconstrained) type params, and builtin/primitive bounds are skipped.
///
/// Sources:
/// - `type_parameters` → `type_parameter` with a `trait_bounds` child
///   (`fn f<T: A + B>`), giving `(T, A)` and `(T, B)`.
/// - `where_clause` → `where_predicate` (`left: type_identifier`, `trait_bounds`)
///   (`where T: A`).
fn generic_bounds(source: &[u8], func: Node<'_>) -> Vec<GenericBound> {
    let mut out: Vec<GenericBound> = Vec::new();

    // Angle-bracket bounds: `<T: A + B, U>`.
    if let Some(tps) = func.child_by_field_name("type_parameters") {
        for i in 0..tps.named_child_count() {
            let Some(tp) = tps.named_child(i) else {
                continue;
            };
            if tp.kind() != "type_parameter" {
                continue;
            }
            // The constrained type param is the leading `type_identifier`.
            let Some(name_node) = named_child_of_kinds(tp, &["type_identifier"]) else {
                continue;
            };
            let type_param = node_text(source, name_node).to_string();
            if let Some(bounds) = named_child_of_kinds(tp, &["trait_bounds"]) {
                push_trait_bounds(source, bounds, &type_param, &mut out);
            }
        }
    }

    // `where` predicates: `where T: A, U: B`.
    let mut cursor = func.walk();
    for child in func.children(&mut cursor) {
        if child.kind() != "where_clause" {
            continue;
        }
        for i in 0..child.named_child_count() {
            let Some(pred) = child.named_child(i) else {
                continue;
            };
            if pred.kind() != "where_predicate" {
                continue;
            }
            let left = pred
                .child_by_field_name("left")
                .or_else(|| named_child_of_kinds(pred, &["type_identifier"]));
            let Some(left) = left else { continue };
            let type_param = node_text(source, left).to_string();
            if let Some(bounds) = named_child_of_kinds(pred, &["trait_bounds"]) {
                push_trait_bounds(source, bounds, &type_param, &mut out);
            }
        }
    }

    out
}

/// Push one [`GenericBound`] per concrete trait inside a `trait_bounds` node
/// (`: A + B + 'a`). Lifetimes and builtin/primitive type names are skipped.
fn push_trait_bounds(
    source: &[u8],
    bounds: Node<'_>,
    type_param: &str,
    out: &mut Vec<GenericBound>,
) {
    for i in 0..bounds.named_child_count() {
        let Some(b) = bounds.named_child(i) else {
            continue;
        };
        // Each bound is a type (type_identifier / generic_type /
        // scoped_type_identifier / higher_ranked_trait_bound) or a lifetime.
        // Reuse the type-stripping walker to get the base trait name(s).
        if b.kind() == "lifetime" {
            continue;
        }
        let mut names: Vec<&str> = Vec::new();
        // For a generic bound `Iterator<Item = T>` we only want the base
        // `Iterator`, so take the first concrete type identifier.
        type_identifiers_in(source, b, &mut names);
        if let Some(&bound) = names.first() {
            if bound.is_empty() || is_builtin_rust_type(bound) {
                continue;
            }
            out.push(GenericBound {
                type_param: type_param.to_string(),
                bound: bound.to_string(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// JavaScript / TypeScript extraction
// ---------------------------------------------------------------------------
//
// Mirrors the Rust/Python passes at the level the JS/TS grammars support,
// reusing the same `ExtractedNode` / `ExtractedEdge` conventions and the same
// name-based resolution keys (`callee_name`, `imported_name`) so the indexer's
// existing two-phase resolver links JS/TS edges cross-file with NO indexer
// change:
//
//   * DEFINITIONS — `function_declaration` / `class_declaration` /
//     `method_definition` / arrow-or-function assigned to a binding, plus the
//     TypeScript-only `interface_declaration` / `type_alias_declaration` /
//     `enum_declaration`. A method (a `method_definition` inside a class body)
//     is owned by its class: qname `{file}::{Class}::{name}`. A free function /
//     arrow is `{file}::Function::{name}`; a class is `{file}::Class::{name}`;
//     an interface is `{file}::Interface::{name}`; a type alias is
//     `{file}::Type::{name}`; an enum is `{file}::Enum::{name}`.
//   * CALLS — final callee identifier → `CALLS` edge with the `callee_name`
//     property (the bare final identifier the cross-file resolver keys on),
//     sourced from the enclosing function/method/arrow qname. `require(...)`
//     callees are dropped here (the imports pass owns them).
//   * IMPORTS — `import` statements (default / named / namespace / aliased /
//     side-effect-only) and `const x = require("m")` → `IMPORTS` edges, one per
//     bound name, with `imported_name` keying the resolver, plus a searchable
//     `Import` node per name.
//   * docstrings — a leading JSDoc block comment (`/** … */`) immediately
//     preceding the definition becomes the node's `doc` (one-line summary) and
//     `doc_full` properties.

/// Run all JS/TS extraction passes. Shared by JavaScript and both TypeScript
/// variants — the node kinds and conventions are identical; the TypeScript
/// query set merely adds interface/type/enum definitions, which the shared
/// definition loop handles by node kind.
fn extract_js_ts(
    language: Language,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let (queries, spec) = match language {
        Language::JavaScript => (
            crate::query::cached_query_set(&language).map_err(|e| {
                greppy_core::Error::Parse(format!("compile javascript queries: {e}"))
            })?,
            &crate::spec::JAVASCRIPT,
        ),
        Language::TypeScript { .. } => (
            crate::query::cached_query_set(&language).map_err(|e| {
                greppy_core::Error::Parse(format!("compile typescript queries: {e}"))
            })?,
            &crate::spec::TYPESCRIPT,
        ),
        other => {
            return Err(greppy_core::Error::Parse(format!(
                "extract_js_ts called with non-JS/TS language: {}",
                other.name()
            )))
        }
    };
    // The shared spec engine covers Function / Class / Method / Interface /
    // Type / Enum definitions plus CALLS and IMPORTS. JS/TS additionally needs
    // `Variable` definition nodes that the spec engine does not emit —
    // module-level `const`/`let`/`var` bindings and enum members. Those
    // Variables are real definition nodes and each gets a File→DEFINES edge, so
    // we add them here in a JS/TS-specific pass that covers module-level
    // variables and enum members, leaving the shared `spec_extract` (and every
    // other language it drives) untouched.
    let mut result = crate::spec::spec_extract(language, spec, queries, source, file_path)?;
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    extract_js_ts_variables(root, source, file_path, &mut result);

    // CALLS — the shared spec engine hangs a call's source endpoint off the
    // nearest ancestor whose *kind* matches a callable `DefRule`. For JS/TS
    // that includes `variable_declarator`, so a call inside
    // `const rec = normalizeRecord(...)` is (wrongly) attributed to a
    // `Function::rec` that never existed — the indexer then drops the edge for
    // want of a real source node. The spec engine also drops file-scope calls
    // entirely (no enclosing callable). To capture both, we drop the spec
    // engine's JS/TS CALLS and re-emit them with our own enclosing-function
    // model: the nearest ancestor that is a `function_declaration` /
    // `method_definition` / `arrow_function` / `function_expression`, named the
    // way the def pass named it, with a `__file__` fallback at module scope.
    result.edges.retain(|e| e.edge_type != "CALLS");
    extract_js_ts_calls(root, source, file_path, &mut result);

    // USAGE — a per-language reference pass: every bare
    // `identifier` / `type_identifier` that is NOT the callee/argument of a
    // call, NOT inside an import-bearing statement, NOT the `name:` of its own
    // definition, and NOT a language keyword becomes a `USAGE` edge from its
    // enclosing function (or the `__file__` node) keyed on `ref_name`. The
    // shared indexer resolves `ref_name` to any registered symbol and drops it
    // unless unique, so unresolved references never become edges (no
    // over-emission).
    extract_js_ts_usages(root, source, file_path, &mut result);
    Ok(result)
}

/// JS/TS grammar node kinds treated as an *enclosing function*. A call's /
/// usage's `source` endpoint is the nearest ancestor of one of these kinds.
const JS_TS_FUNC_KINDS: &[&str] = &[
    "function_declaration",
    "method_definition",
    "arrow_function",
    "function_expression",
    "generator_function",
    "generator_function_declaration",
];

/// JS/TS class node kinds: a method/arrow owned by one of
/// these is qualified `{file}::{Class}::{name}`, matching the def pass.
const JS_TS_CLASS_KINDS: &[&str] = &["class_declaration", "class"];

/// JS/TS call node kinds.
const JS_TS_CALL_KINDS: &[&str] = &["call_expression", "new_expression"];

/// JS/TS import-bearing node kinds. A reference inside any
/// of these is an import binding, not a usage (note `lexical_declaration` is
/// here — a `const x: T = …` annotation is NOT a usage, which is
/// exactly why `runPipeline`'s `const rows: Entry[]` yields no USAGE edge).
const JS_TS_IMPORT_KINDS: &[&str] = &[
    "import_statement",
    "lexical_declaration",
    "export_statement",
    "import",
    "extends",
    "require",
];

/// JS/TS keyword + well-known-global set. A
/// reference whose text is in this set is never a USAGE (keeps `console`,
/// `Math`, `Promise`, … from resolving to same-named user symbols).
const JS_TS_KEYWORDS: &[&str] = &[
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "let",
    "new",
    "null",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "undefined",
    "var",
    "void",
    "while",
    "with",
    "yield",
    "async",
    "await",
    "of",
    "static",
    "get",
    "set",
    "from",
    "as",
    "constructor",
    "prototype",
    "console",
    "window",
    "document",
    "process",
    "module",
    "exports",
    "require",
    "Array",
    "Object",
    "String",
    "Number",
    "Boolean",
    "Symbol",
    "Map",
    "Set",
    "Promise",
    "Error",
    "RegExp",
    "Date",
    "Math",
    "JSON",
    "parseInt",
    "parseFloat",
    "setTimeout",
    "setInterval",
    "clearTimeout",
    "clearInterval",
];

/// The `source` qname for a call/usage at `node`: the nearest enclosing
/// function's def qname, or the per-file `__file__` node at module scope.
///
/// Compute the enclosing-function qualified name:
///   * find the nearest ancestor in [`JS_TS_FUNC_KINDS`];
///   * name it — the `name:` field for a declaration/method, or (for an
///     `arrow_function` / `function_expression`) the enclosing
///     `variable_declarator`'s bound name;
///   * a `function_declaration` is always a free `Function` (the def pass
///     never owns it); a `method_definition` or a name-bearing arrow/function
///     nested in a `class` body is owned `{file}::{Class}::{name}`; everything
///     else is `{file}::Function::{name}`;
///   * an anonymous inline callback is skipped and the walk continues to the
///     nearest NAMED scope, so a callback nested in a named function is still
///     attributed to that function;
///   * if no named enclosing function at all, fall back to
///     `{file}::__file__`.
fn js_ts_enclosing_qname(node: Node<'_>, source: &[u8], file_path: &str) -> String {
    let file_qname = format!("{file_path}::__file__");
    let mut p = node.parent();
    while let Some(cur) = p {
        if JS_TS_FUNC_KINDS.contains(&cur.kind()) {
            if let Some((name, node_for_owner)) = js_ts_func_name(cur, source) {
                // `function_declaration` is never class-owned by the def pass.
                let owner = if cur.kind() == "function_declaration" {
                    None
                } else {
                    js_ts_enclosing_class_name(node_for_owner, source)
                };
                return match owner {
                    Some(class) => format!("{file_path}::{class}::{name}"),
                    None => format!("{file_path}::Function::{name}"),
                };
            }
            // An unnamed enclosing function (anonymous inline callback).
            // Rather than attributing the call to the file node — which would
            // erase the real caller whenever the callback sits inside a named
            // function (`function outer() { arr.map(x => helper(x)) }` would
            // lose `outer -> helper`) — keep walking to the nearest NAMED scope;
            // module-level callbacks still fall through to `__file__` below.
        }
        p = cur.parent();
    }
    file_qname
}

/// The name of a JS/TS enclosing-function node plus the node whose ancestry
/// decides class ownership. Returns `None` when the function is anonymous (so
/// the caller falls back to the `__file__` node).
fn js_ts_func_name<'a, 't>(func: Node<'t>, source: &'a [u8]) -> Option<(&'a str, Node<'t>)> {
    // A `name:` field covers `function_declaration`, `method_definition`, and a
    // named `function_expression`.
    if let Some(name_node) = func.child_by_field_name("name") {
        let name = node_text(source, name_node);
        if !name.is_empty() {
            return Some((name, func));
        }
    }
    // Arrow / anonymous function-expression bound to a declarator:
    // `const f = () => {}` / `const f = function () {}`. The def pass emits
    // this via the `variable_declarator` rule, named from the declarator.
    if matches!(func.kind(), "arrow_function" | "function_expression") {
        if let Some(parent) = func.parent() {
            if parent.kind() == "variable_declarator" {
                if let Some(vname) = parent.child_by_field_name("name") {
                    if vname.kind() == "identifier" {
                        let name = node_text(source, vname);
                        if !name.is_empty() {
                            // Class ownership is decided from the declarator's
                            // ancestry (same as the def pass).
                            return Some((name, parent));
                        }
                    }
                }
            }
        }
    }
    None
}

/// The name of the nearest enclosing `class` of `node`, if any (drives
/// `{file}::{Class}::{member}` ownership, matching `Owner::EnclosingName`).
fn js_ts_enclosing_class_name<'a>(node: Node<'_>, source: &'a [u8]) -> Option<&'a str> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if JS_TS_CLASS_KINDS.contains(&cur.kind()) {
            return cur
                .child_by_field_name("name")
                .map(|n| node_text(source, n))
                .filter(|s| !s.is_empty());
        }
        p = cur.parent();
    }
    None
}

/// Walk every `call_expression` / `new_expression` and emit one `CALLS` edge
/// per call, keyed on the callee's simple (last-segment) name — exactly the key
/// the shared resolver resolves on. `require` is skipped (the imports pass
/// owns it).
fn extract_js_ts_calls(
    root: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if JS_TS_CALL_KINDS.contains(&node.kind()) {
            if let Some(callee) = js_ts_callee_name(node, source) {
                if !callee.is_empty() && callee != "require" {
                    let src = js_ts_enclosing_qname(node, source, file_path);
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: src,
                        target_qualified_name: format!("{file_path}::Function::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": callee,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
}

/// The simple (last-segment) callee name of a JS/TS `call_expression` /
/// `new_expression`. For a member call `a.b.c()` this is `c`; for a bare call
/// `f()` it is `f`; for `new T()` it is `T`. This is the resolver's lookup key.
fn js_ts_callee_name<'a>(call: Node<'_>, source: &'a [u8]) -> Option<&'a str> {
    if call.kind() == "new_expression" {
        // `constructor:` field, else the first named child (the type).
        let ctor = call
            .child_by_field_name("constructor")
            .or_else(|| call.named_child(0))?;
        return Some(js_ts_simple_name(ctor, source));
    }
    let func = call.child_by_field_name("function")?;
    Some(js_ts_simple_name(func, source))
}

/// The simple name of a callee expression: the `property:` of a
/// `member_expression`, else the text's last dotted segment.
fn js_ts_simple_name<'a>(node: Node<'_>, source: &'a [u8]) -> &'a str {
    if node.kind() == "member_expression" {
        if let Some(prop) = node.child_by_field_name("property") {
            return node_text(source, prop);
        }
    }
    let text = node_text(source, node);
    text.rsplit('.').next().unwrap_or(text)
}

/// Emit `USAGE` edges for a JS/TS file. See the call-site comment in
/// `extract_js_ts` for the contract; the four skip guards are applied below.
fn extract_js_ts_usages(
    root: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        try_emit_js_ts_usage(node, source, file_path, result);
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
}

fn try_emit_js_ts_usage(
    node: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    // JS/TS references are `identifier` / `type_identifier`.
    if !matches!(node.kind(), "identifier" | "type_identifier") {
        return;
    }
    // Skip callees/arguments of a call and import bindings (C checks up to 10
    // ancestors for each set).
    if js_ts_ancestor_in(node, JS_TS_CALL_KINDS) || js_ts_inside_import(node) {
        return;
    }
    // Skip a node that IS the `name:` field of its own parent (a definition
    // name, not a reference).
    if js_ts_is_definition_name(node) {
        return;
    }
    let name = node_text(source, node);
    if name.is_empty() || JS_TS_KEYWORDS.contains(&name) {
        return;
    }
    let src = js_ts_enclosing_qname(node, source, file_path);
    result.edges.push(ExtractedEdge {
        edge_type: "USAGE".into(),
        source_qualified_name: src,
        // No real target qname exists; the indexer resolves `ref_name` to any
        // registered symbol and drops it unless unique.
        target_qualified_name: format!("{file_path}::__ref__::{name}"),
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
        properties: serde_json::json!({ "ref_name": name }),
    });
}

/// Whether any ancestor of `node` within 10 levels has a kind in `kinds`.
fn js_ts_ancestor_in(node: Node<'_>, kinds: &[&str]) -> bool {
    let mut p = node.parent();
    let mut depth = 0;
    while let Some(cur) = p {
        if depth >= 10 {
            break;
        }
        if kinds.contains(&cur.kind()) {
            return true;
        }
        p = cur.parent();
        depth += 1;
    }
    false
}

/// Whether `node` sits inside an import-bearing statement, with an
/// `is_export_of_declaration` refinement. An `export_statement` that wraps a
/// *declaration*
/// (`export function f(x: T) {}`) is NOT an import boundary, so type references
/// in an exported declaration's signature still count as usages; only a bare
/// re-export (`export { X } from './m'`) suppresses them.
fn js_ts_inside_import(node: Node<'_>) -> bool {
    let mut p = node.parent();
    let mut depth = 0;
    while let Some(cur) = p {
        if depth >= 10 {
            break;
        }
        if JS_TS_IMPORT_KINDS.contains(&cur.kind())
            && !(cur.kind() == "export_statement" && js_ts_export_of_declaration(cur))
        {
            return true;
        }
        p = cur.parent();
        depth += 1;
    }
    false
}

/// Whether an `export_statement` wraps a declaration child (vs. a bare
/// re-export).
fn js_ts_export_of_declaration(export: Node<'_>) -> bool {
    let mut cursor = export.walk();
    let found = export.children(&mut cursor).any(|c| {
        matches!(
            c.kind(),
            "function_declaration"
                | "class_declaration"
                | "lexical_declaration"
                | "abstract_class_declaration"
                | "interface_declaration"
                | "enum_declaration"
                | "type_alias_declaration"
                | "variable_declaration"
                | "generator_function_declaration"
        )
    });
    found
}

/// Whether `node` is the `name:` field of its own parent.
fn js_ts_is_definition_name(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    match parent.child_by_field_name("name") {
        Some(name_field) => {
            name_field.start_byte() == node.start_byte() && name_field.end_byte() == node.end_byte()
        }
        None => false,
    }
}

/// Emit `Variable` definition nodes for a JS/TS file:
///
///   * Walk the **module-level** children of the program
///     root, matching `lexical_declaration` / `variable_declaration`,
///     unwrapping an `export_statement` wrapper — so
///     `export const x = 1` is covered.
///   * Iterate each declaration's `variable_declarator`s and
///     skip any whose `value` is an `arrow_function` / `function_expression` /
///     `generator_function` (those are Functions, emitted by the def pass, not
///     Variables). A `require(...)`-valued declarator is a `call_expression`, so
///     it is NOT skipped — its bound name(s) become Variables (this covers
///     `const { X } = require(...)`).
///   * Destructured bindings (`object_pattern` / `array_pattern`) emit one
///     Variable per bound identifier.
///   * Emit one `Variable` per member of a
///     `enum_declaration`, owned by the enum.
///
/// Qnames follow the `{file}::Variable::{name}` scheme (as in
/// `extract_rust`); the structural pass keys File→DEFINES off the node label,
/// so any collision-free qname yields the correct DEFINES edge.
fn extract_js_ts_variables(
    root: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    // Module-level `const`/`let`/`var`. Only top-level children of the program
    // root qualify, so class-body / function-body locals are NOT module
    // Variables.
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        match child.kind() {
            "lexical_declaration" | "variable_declaration" => {
                emit_js_ts_declarators(child, source, file_path, result);
            }
            // Unwrap an `export`/`statement`/`expression_statement` wrapper and
            // look one level in for a variable declaration or enum
            // (`export const x = 1`, `export enum E { ... }`).
            "export_statement" | "statement" | "expression_statement" => {
                let mut inner = child.walk();
                for grand in child.named_children(&mut inner) {
                    match grand.kind() {
                        "lexical_declaration" | "variable_declaration" => {
                            emit_js_ts_declarators(grand, source, file_path, result);
                        }
                        "enum_declaration" => {
                            emit_js_ts_enum_members(grand, source, file_path, result);
                        }
                        _ => {}
                    }
                }
            }
            "enum_declaration" => {
                emit_js_ts_enum_members(child, source, file_path, result);
            }
            _ => {}
        }
    }
}

/// Push one `Variable` node per non-function `variable_declarator` in a
/// `lexical_declaration` / `variable_declaration`, expanding destructuring.
fn emit_js_ts_declarators(
    decl: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut cursor = decl.walk();
    for vd in decl.named_children(&mut cursor) {
        if vd.kind() != "variable_declarator" {
            continue;
        }
        // Skip function-valued declarators (they are Functions, not Variables).
        if let Some(value) = vd.child_by_field_name("value") {
            if matches!(
                value.kind(),
                "arrow_function" | "function_expression" | "generator_function"
            ) {
                continue;
            }
        }
        let Some(name_node) = vd.child_by_field_name("name") else {
            continue;
        };
        match name_node.kind() {
            "object_pattern" | "array_pattern" => {
                emit_js_ts_destructured(name_node, vd, source, file_path, result);
            }
            _ => {
                push_js_ts_variable(node_text(source, name_node), vd, file_path, result);
            }
        }
    }
}

/// Emit one `Variable` per bound identifier in a destructuring pattern.
fn emit_js_ts_destructured(
    pattern: Node<'_>,
    decl: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut cursor = pattern.walk();
    for pat_child in pattern.named_children(&mut cursor) {
        let ident = match pat_child.kind() {
            "shorthand_property_identifier_pattern" | "identifier" => Some(pat_child),
            // `{ a: b }` binds `b` (the pattern's `value`).
            "pair_pattern" => pat_child.child_by_field_name("value"),
            // rest_pattern, assignment_pattern, object_assignment_pattern, … →
            // first named child (fallback).
            _ => pat_child.named_child(0),
        };
        let Some(ident) = ident else { continue };
        // A nested pattern (`{ a: { b } }`) recurses; a bare identifier emits.
        match ident.kind() {
            "object_pattern" | "array_pattern" => {
                emit_js_ts_destructured(ident, decl, source, file_path, result);
            }
            _ => {
                let text = node_text(source, ident);
                if !text.is_empty() {
                    push_js_ts_variable(text, decl, file_path, result);
                }
            }
        }
    }
}

/// Emit one `Variable` per member of a TS `enum_declaration`, owned by the enum
/// (each member is labeled `Variable`).
fn emit_js_ts_enum_members(
    enum_node: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let body = enum_node
        .child_by_field_name("body")
        .or_else(|| find_child_of_kind(enum_node, "enum_body"));
    let Some(body) = body else { return };
    let Some(enum_name) = enum_node
        .child_by_field_name("name")
        .map(|n| node_text(source, n))
    else {
        return;
    };
    let mut cursor = body.walk();
    for member in body.named_children(&mut cursor) {
        // TS enum members parse as `property_identifier` (bare) or
        // `enum_assignment` (`A = 1`), whose `name:` is the member identifier.
        let name_node = match member.kind() {
            "property_identifier" => Some(member),
            "enum_assignment" | "enum_member" | "enum_member_declaration" => member
                .child_by_field_name("name")
                .or_else(|| find_child_of_kind(member, "property_identifier"))
                .or_else(|| find_child_of_kind(member, "identifier")),
            _ => None,
        };
        let Some(name_node) = name_node else { continue };
        let name = node_text(source, name_node);
        if name.is_empty() {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: name.to_string(),
            qualified_name: format!("{file_path}::{enum_name}::{name}"),
            file_path: file_path.to_string(),
            start_line: member.start_position().row as u32 + 1,
            end_line: member.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// Push a single `Variable` node (`decl` supplies the line span, recording the
/// declarator's position).
fn push_js_ts_variable(name: &str, decl: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    if name.is_empty() {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Variable".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Variable::{name}"),
        file_path: file_path.to_string(),
        start_line: decl.start_position().row as u32 + 1,
        end_line: decl.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// First direct child of `node` whose kind is `kind`, if any.
fn find_child_of_kind<'t>(node: Node<'t>, kind: &str) -> Option<Node<'t>> {
    let mut cursor = node.walk();
    let found = node.children(&mut cursor).find(|c| c.kind() == kind);
    found
}

// ---------------------------------------------------------------------------
// Python extraction
// ---------------------------------------------------------------------------
//
// Mirrors the Rust passes at the level Python's grammar supports, reusing the
// same `ExtractedNode` / `ExtractedEdge` conventions and the same name-based
// resolution keys (`callee_name`, `imported_name`) so the indexer's existing
// two-phase resolver links Python edges cross-file with NO indexer change:
//
//   * DEFINITIONS — `function_definition` / `class_definition` →
//     `Function` / `Method` / `Class` nodes. A method (function nested in a
//     class body) is owned by its class: qname `{file}::{Class}::{name}`. A
//     free function is `{file}::Function::{name}`; a class is
//     `{file}::Class::{name}`.
//   * CALLS — final callee identifier → `CALLS` edge with the `callee_name`
//     property (the bare final identifier the cross-file resolver keys on),
//     sourced from the enclosing function/method qname.
//   * IMPORTS — `import` / `from x import y` → `IMPORTS` edges, one per bound
//     name, with `imported_name` keying the resolver, plus a searchable
//     `Import` node per name.
//   * docstrings — the first string statement in a def/class body becomes the
//     node's `doc` (one-line summary) and `doc_full` properties.

fn extract_python(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Python)
        .map_err(|e| greppy_core::Error::Parse(format!("compile python queries: {e}")))?;
    let mut result = crate::spec::spec_extract(
        Language::Python,
        &crate::spec::PYTHON,
        queries,
        source,
        file_path,
    )?;

    // MODULE-LEVEL VARIABLE PASS.
    //
    // The uniform spec engine only models Function/Method/Class for Python; we
    // additionally emit a `Variable` def for every *module-level* assignment:
    //
    //   * only `assignment` / `augmented_assignment` nodes qualify;
    //   * they must sit at module level — a direct child of the `module` root,
    //     or wrapped in a top-level `expression_statement` whose parent is the
    //     `module`; assignments inside a function or class body are NOT
    //     variables;
    //   * the name is the `left:` field, and *only* when it is a plain
    //     `identifier` (so `a, b = …`, `obj.attr = …`, `d[k] = …` are skipped);
    //   * the `_` placeholder and empty names are skipped.
    //
    // qname follows the Rust extractor's `Variable` convention
    // (`{file}::Variable::{name}`); the File→DEFINES edge the indexer hangs off
    // this node keys on the node id, not the qname text.
    let tree = crate::parse(Language::Python, source)?;
    let root = tree.root_node();
    if root.kind() == "module" {
        let mut cursor = root.walk();
        for child in root.named_children(&mut cursor) {
            // A top-level `assignment` / `augmented_assignment` is wrapped in an
            // `expression_statement`; a bare one can also appear directly. Look
            // through the wrapper.
            let assign = match child.kind() {
                "assignment" | "augmented_assignment" => Some(child),
                "expression_statement" => child
                    .named_child(0)
                    .filter(|n| matches!(n.kind(), "assignment" | "augmented_assignment")),
                _ => None,
            };
            let Some(assign) = assign else { continue };
            let Some(left) = assign.child_by_field_name("left") else {
                continue;
            };
            if left.kind() != "identifier" {
                continue;
            }
            let vname = node_text(source, left);
            if vname.is_empty() || vname == "_" {
                continue;
            }
            result.nodes.push(ExtractedNode {
                label: "Variable".into(),
                name: vname.to_string(),
                qualified_name: format!("{file_path}::Variable::{vname}"),
                file_path: file_path.to_string(),
                start_line: assign.start_position().row as u32 + 1,
                end_line: assign.end_position().row as u32 + 1,
                properties: serde_json::json!({}),
            });
        }
    }

    // MODULE-SCOPE CALLS PASS.
    //
    // The shared `spec_calls` only emits a `CALLS` edge when the call has an
    // enclosing callable (`enclosing_callable_qname` → `Some`). We additionally
    // fall back to the *file* node when a call sits at module scope with no
    // enclosing function, so a top-level `main()` (e.g. under
    // `if __name__ == "__main__":`) still produces a `CALLS` edge
    // `<file>::__file__ → main`: walk every `call` whose final callee is an
    // identifier/attribute and which has NO enclosing `function_definition`,
    // and emit the edge from the file Module node. The name-based resolver drops
    // callees that don't resolve (builtins like `print`/`len`), so this never
    // over-emits resolved edges.
    let file_module_qname = format!("{file_path}::__file__");
    emit_python_module_scope_calls(source, root, file_path, &file_module_qname, &mut result);

    // IMPORTS COLLAPSE.
    //
    // We model an import as ONE edge per import *statement*, targeting the
    // imported *module* (`from a.b.c import x, y` → a single edge to module
    // `a.b.c`). The shared `py_expand_imports` instead yields
    // one item per bound *name*, so a multi-name `from … import x, y` produces
    // two `IMPORTS` edges that resolve to two distinct symbols and both
    // survive. Collapse the shared pass's per-name edges back
    // to per-statement/per-module granularity: keep only the first
    // `IMPORTS` edge for each `(source file, module prefix)` pair. Single-name
    // imports (the overwhelming majority) and multi-*module* `import a, b`
    // statements are unaffected because their module prefixes differ.
    collapse_python_imports(source, root, &mut result);

    // USAGE PASS.
    //
    // Emit one `USAGE` for every identifier /
    // attribute reference that is NOT:
    //   * a definition *name* (the `name:` field of its parent — for Python
    //     this is only the `function_definition` / `class_definition` name;
    //     an `assignment` uses a `left:` field, so its LHS is a usage — e.g.
    //     `x = 1` emits a usage of `x`);
    //   * inside a `call` / `with_statement` (the callee is
    //     already a `CALLS` edge, and every argument nested under the call is
    //     suppressed by the ancestor scan);
    //   * inside an `import_statement`; note that a
    //     `from … import …` (`import_from_statement`) is NOT suppressed, so
    //     the imported name and dotted module path DO emit usages;
    //   * a Python keyword / common builtin.
    //
    // The reference kinds for Python are `identifier`, `type_identifier`, and
    // `attribute`. The source endpoint
    // is the enclosing function/method qname, or the file-level `__file__`
    // Module node at module scope. The
    // indexer resolves `ref_name` against any registered symbol and drops it
    // if not unique, so no real target qname is needed.
    emit_python_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// Reference node kinds for the Python usage pass: a bare/typed identifier or
/// an `attribute` expression.
fn is_python_reference_kind(kind: &str) -> bool {
    matches!(kind, "identifier" | "type_identifier" | "attribute")
}

/// True if `node` sits within an ancestor of any of `kinds`, using a
/// parent-depth bound of 10. A dedicated Python copy so the bound is 10
/// (the shared `is_inside_kind` uses a depth of 12).
fn python_is_inside_kind(node: Node<'_>, kinds: &[&str]) -> bool {
    const MAX_PARENT_DEPTH: usize = 10;
    let mut cur = node.parent();
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= MAX_PARENT_DEPTH {
            break;
        }
        if kinds.contains(&n.kind()) {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// Python keyword / common-builtin filter. References
/// whose text is one of these are never emitted as a usage.
fn is_python_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "while"
            | "with"
            | "yield"
            | "self"
            | "cls"
            | "__init__"
            | "__name__"
            | "__main__"
            | "super"
            | "print"
            | "len"
            | "range"
            | "enumerate"
            | "zip"
            | "map"
            | "filter"
            | "type"
            | "int"
            | "str"
            | "float"
            | "bool"
            | "list"
            | "dict"
            | "set"
            | "tuple"
            | "bytes"
    )
}

/// The qname of the nearest enclosing Python callable (`function_definition`)
/// for a usage's source endpoint, constructed with the same ownership scheme
/// the definition pass uses: a method owned by an enclosing `class_definition`
/// gets `{file}::{Class}::{name}`, a free function `{file}::Function::{name}`.
/// Returns `None` when the reference is at module scope (the caller then uses
/// the file-level `__file__` Module node).
fn python_enclosing_usage_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "function_definition" {
            let name = cur
                .child_by_field_name("name")
                .map(|n| node_text(source, n).to_string())?;
            // Determine ownership: the nearest enclosing `class_definition`
            // above this function makes it a method (qname `{file}::{Class}::
            // {name}`); otherwise it is a free function.
            let mut owner: Option<String> = None;
            let mut q = cur.parent();
            while let Some(anc) = q {
                if anc.kind() == "class_definition" {
                    if let Some(cn) = anc.child_by_field_name("name") {
                        owner = Some(node_text(source, cn).to_string());
                    }
                    break;
                }
                if anc.kind() == "function_definition" {
                    // A nested function is owned by its enclosing function's
                    // scope, not the outer class — stop the class search.
                    break;
                }
                q = anc.parent();
            }
            return Some(match owner {
                Some(cls) => format!("{file_path}::{cls}::{name}"),
                None => format!("{file_path}::Function::{name}"),
            });
        }
        p = cur.parent();
    }
    None
}

/// Recursively emit `USAGE` edges for every Python reference node under
/// `node`. Walks all named descendants (every reference kind — `identifier`,
/// `type_identifier`, `attribute` — is a named node, so anonymous children can
/// never be references and are safely skipped).
fn emit_python_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if is_python_reference_kind(node.kind())
        && !python_is_inside_kind(node, &["call", "with_statement"])
        && !python_is_inside_kind(node, &["import_statement"])
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_python_usage_keyword(text) {
            let source_qname = python_enclosing_usage_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                // Name-based only: the indexer resolves `ref_name` against any
                // registered symbol and drops it if not unique, so the target
                // qname is a placeholder that never needs to resolve directly.
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({
                    "ref_name": text,
                }),
            });
        }
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        emit_python_usages(source, child, file_path, file_module_qname, result);
    }
}

/// The final callee identifier of a Python `call` node, mirroring the shared
/// CALLS query (`python_queries::CALLS`): a bare `function: (identifier)` gives
/// that identifier; a `function: (attribute attribute: (identifier))` gives the
/// final attribute segment. Returns `None` for other callee shapes (e.g. a call
/// whose callee is itself a subscript or another call).
fn python_call_callee_text<'a>(source: &'a [u8], call: Node<'_>) -> Option<&'a str> {
    let func = call.child_by_field_name("function")?;
    match func.kind() {
        "identifier" => Some(node_text(source, func)),
        "attribute" => {
            let attr = func.child_by_field_name("attribute")?;
            if attr.kind() == "identifier" {
                Some(node_text(source, attr))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Emit `CALLS` edges for `call`s that sit at *module scope* (no enclosing
/// `function_definition`), sourced from the file's `__file__` Module node.
/// This is the file-node fallback that the shared `spec_calls` omits (it only
/// emits when an enclosing callable exists). Recurses through the tree, skipping
/// the bodies of function definitions (those calls already have an enclosing
/// callable and are handled by `spec_calls`).
fn emit_python_module_scope_calls(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    // A function body's calls are owned by that function — `spec_calls` already
    // emits them. Do not descend into nested definitions here.
    if node.kind() == "function_definition" {
        return;
    }
    if node.kind() == "call" {
        if let Some(text) = python_call_callee_text(source, node) {
            if !text.is_empty() {
                result.edges.push(ExtractedEdge {
                    edge_type: "CALLS".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::Function::{text}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "callee_text": text,
                        "callee_name": text,
                    }),
                });
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        emit_python_module_scope_calls(source, child, file_path, file_module_qname, result);
    }
}

/// Collapse the shared IMPORTS pass's per-*name* edges to per-*module*
/// granularity. We model an import as one edge per statement targeting the
/// imported *module*, deduped by module across statements:
/// `from a.b.c import x, y` and a later
/// `from a.b.c import z` together produce a single edge to module `a.b.c`.
/// The shared `py_expand_imports` instead yields one item per bound *name*, so
/// each `from`-name resolves to a distinct symbol and survives — inflating the
/// count for multi-name / repeated-module imports.
///
/// Retain only the first `IMPORTS` edge per `(source file, module)` pair. The
/// module of an edge is derived from its originating statement in the AST (via
/// the edge's line): a `from M import …` contributes module `M`; a plain
/// `import A.B` contributes module `A.B`. Deriving the module from the AST
/// rather than the ambiguous `path` property keeps plain `import a.b` /
/// `import a.c` (distinct modules) correctly separate.
fn collapse_python_imports(source: &[u8], root: Node<'_>, result: &mut ExtractionResult) {
    use std::collections::{HashMap, HashSet};

    // line (1-based) → module name, for every import statement in the file.
    // From-imports key on the `module_name` field; plain imports on the base
    // module of each imported binding (the full dotted path, alias stripped).
    let mut line_module: HashMap<u32, String> = HashMap::new();
    collect_python_import_modules(source, root, &mut line_module);

    let mut seen: HashSet<(String, String)> = HashSet::new();
    result.edges.retain(|edge| {
        if edge.edge_type != "IMPORTS" {
            return true;
        }
        // Fall back to the `path` property's module prefix if the line is not
        // mapped (defensive — every IMPORTS edge should sit on a mapped line).
        let module = line_module.get(&edge.line).cloned().unwrap_or_else(|| {
            let path = edge
                .properties
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            match path.rsplit_once('.') {
                Some((prefix, _last)) => prefix.to_string(),
                None => path.to_string(),
            }
        });
        seen.insert((edge.source_qualified_name.clone(), module))
    });
}

/// Walk the tree and record, for each import statement's start line, the
/// module it imports (the per-statement import target). Used by
/// [`collapse_python_imports`] to dedup per module. A `from M import …`
/// statement maps its line to `M`; a plain `import A.B[.C] as d` maps its line
/// to the first binding's dotted module path (`A.B.C`).
fn collect_python_import_modules(
    source: &[u8],
    node: Node<'_>,
    out: &mut std::collections::HashMap<u32, String>,
) {
    match node.kind() {
        "import_from_statement" => {
            if let Some(m) = node.child_by_field_name("module_name") {
                let line = node.start_position().row as u32 + 1;
                out.insert(line, node_text(source, m).to_string());
            }
        }
        "import_statement" => {
            // `import a.b, c` — key the line on the first bound module. The
            // shared pass emits one edge per module here; multi-module plain
            // imports are rare, and each module's edge still dedups against its
            // own module in `retain` via the `path`-property fallback, so a
            // single mapped line for the statement is sufficient for the common
            // single-module case.
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                let module = match child.kind() {
                    "aliased_import" => child
                        .child_by_field_name("name")
                        .map(|n| node_text(source, n).to_string()),
                    "dotted_name" | "identifier" => Some(node_text(source, child).to_string()),
                    _ => None,
                };
                if let Some(module) = module {
                    let line = node.start_position().row as u32 + 1;
                    out.entry(line).or_insert(module);
                    break;
                }
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_python_import_modules(source, child, out);
    }
}

fn extract_rust(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Rust, source)?;
    let queries = crate::query::cached_query_set(&Language::Rust)
        .map_err(|e| greppy_core::Error::Parse(format!("compile rust queries: {e}")))?;

    let mut result = ExtractionResult::default();

    // File-level synthetic qname for `IMPORTS` edges. We emit
    // per-file edges `file → imported_module` and approximate
    // the file endpoint with `<file>::__file__`; the indexer can
    // resolve it against the project row (or we accept that it
    // currently does not — IMPORTS edges are therefore emitted but
    // their source endpoint may not resolve; that's a documented
    // gap in v1 single-file).
    let file_qname = format!("{file_path}::__file__");

    // PASS 1 — definitions. We resolve the impl/trait context here
    // so the qnames are collision-free.
    let mut defs: Vec<DefinitionSpan> = Vec::new();
    for cq in queries
        .iter()
        .filter(|cq| cq.kind == QueryKind::Definitions)
    {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, tree.root_node(), source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };

                // MEMBER PASS — struct/union fields → `Field` nodes and
                // top-level const/static → `Variable` nodes:
                //   - a `field_declaration` with a typed name becomes a `Field`
                //     owned by its enclosing struct/union, qname
                //     `{struct_qname}::{field}` (`::` is the separator
                //     throughout);
                //   - a module-level `const_item`/`static_item` becomes a
                //     `Variable` (qname `{file}::Variable::{name}`), skipping
                //     `_`. An impl/trait associated const has an owner and is
                //     handled by the AssocConst pass, so we filter those out.
                if name == "field" {
                    let node = cap.node;
                    let fname = node_text(source, node);
                    if fname.is_empty() {
                        continue;
                    }
                    // The owning struct/union's def node is the ancestor whose
                    // kind is a type container; its qname mirrors PASS 1
                    // (`enclosing_def_qname` walks to the nearest struct/enum/
                    // trait/function and rebuilds its qname with the same
                    // label scheme).
                    let Some(owner_qn) = enclosing_def_qname(source, node, file_path) else {
                        continue;
                    };
                    let decl = node.parent().unwrap_or(node);
                    let mut properties = serde_json::Map::new();
                    if let Some(ty) = field_declared_type(source, decl) {
                        properties.insert("return_type".into(), serde_json::Value::String(ty));
                    }
                    result.nodes.push(ExtractedNode {
                        label: "Field".into(),
                        name: fname.to_string(),
                        qualified_name: format!("{owner_qn}::{fname}"),
                        file_path: file_path.to_string(),
                        start_line: decl.start_position().row as u32 + 1,
                        end_line: decl.end_position().row as u32 + 1,
                        properties: serde_json::Value::Object(properties),
                    });
                    continue;
                }
                if name == "var" {
                    let node = cap.node;
                    let vname = node_text(source, node);
                    // Skip empty names and the `_`
                    // placeholder; associated (impl/trait) consts are owned and
                    // handled elsewhere, so only module-level items qualify.
                    if vname.is_empty()
                        || vname == "_"
                        || enclosing_impl_type(source, node).is_some()
                    {
                        continue;
                    }
                    let item = node.parent().unwrap_or(node);
                    result.nodes.push(ExtractedNode {
                        label: "Variable".into(),
                        name: vname.to_string(),
                        qualified_name: format!("{file_path}::Variable::{vname}"),
                        file_path: file_path.to_string(),
                        start_line: item.start_position().row as u32 + 1,
                        end_line: item.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                    continue;
                }

                if name != "name" {
                    continue;
                }
                let node = cap.node;
                let text = node_text(source, node);
                let label = match node.parent().map(|p| p.kind()) {
                    Some(k) => match k {
                        "function_item" => {
                            let impl_ctx = enclosing_impl_type(source, node);
                            // Free function vs. method: we label
                            // methods "Method" to make the type
                            // obvious in the graph UI later, but
                            // keep the qname-disambiguation rule
                            // for both.
                            match impl_ctx {
                                Some(_) => "Method",
                                None => "Function",
                            }
                        }
                        // Type-def label mapping: Rust
                        // `struct_item`/`union_item` → "Class", `trait_item` →
                        // "Interface", `enum_item` → "Enum", `type_item` →
                        // "Type". Rust structs are reported as `Class`, not
                        // `Struct`.
                        // `impl_item` produces NO def node: an impl block
                        // contributes only its *methods* as defs and never a
                        // definition for the impl itself. We handle that
                        // by labeling the impl-name capture "Impl" here and then
                        // skipping the node push below (the impl context is still
                        // resolved for method qnames via `enclosing_impl_type`,
                        // which walks the AST and does not depend on a node).
                        "struct_item" | "union_item" => "Class",
                        "enum_item" => "Enum",
                        "trait_item" => "Interface",
                        "impl_item" => "Impl",
                        "type_item" => "Type",
                        _ => "Item",
                    }
                    .to_string(),
                    None => "Item".to_string(),
                };
                // An `impl` block is not a definition of its own, so
                // emit no node for it (only the methods in its body are defs).
                // The impl-name capture still resolves method ownership above.
                if label == "Impl" {
                    continue;
                }
                let qname = match label.as_str() {
                    "Method" | "Function" => {
                        let impl_ctx = enclosing_impl_type(source, node);
                        match impl_ctx {
                            Some(t) => format!("{file_path}::{t}::{text}"),
                            None => format!("{file_path}::Function::{text}"),
                        }
                    }
                    _ => format!("{file_path}::{label}::{text}"),
                };
                // Captures name the identifier, while the parent is the full
                // definition item. Persist the item span so navigation and
                // semantic output cover the complete function body.
                let def_node = node.parent().unwrap_or(node);
                // The enclosing-function qname of this def itself
                // (used when emitting a call edge whose endpoint is
                // THIS function — i.e., a method's qname when a
                // nested call is found inside it).
                let enclosing = if matches!(label.as_str(), "Method" | "Function") {
                    Some(qname.clone())
                } else {
                    None
                };
                defs.push(DefinitionSpan {
                    label: label.clone(),
                    name: text.to_string(),
                    qname: qname.clone(),
                    start_line: def_node.start_position().row as u32 + 1,
                    end_line: def_node.end_position().row as u32 + 1,
                    enclosing_function_qname: enclosing,
                });
                // Docstring: the leading `///` / `/** */` doc comment attached
                // to this definition. We attach the one-line summary as the
                // node's `doc` property,
                // and the full (possibly multi-line) text as `doc_full`. The
                // comment is a preceding sibling of the *definition* node, not
                // the name identifier, so we walk up to it.
                let mut properties = serde_json::Map::new();
                if let Some(doc) = extract_docstring(source, def_node) {
                    let summary = docstring_summary(&doc).to_string();
                    properties.insert("doc".into(), serde_json::Value::String(summary));
                    properties.insert("doc_full".into(), serde_json::Value::String(doc));
                }

                // Modifiers: visibility + async/unsafe/const. Captured for every
                // def kind (structs/traits/enums carry visibility only).
                let mods = modifier_info(source, def_node);
                if let Some(vis) = &mods.visibility {
                    properties.insert("visibility".into(), serde_json::Value::String(vis.clone()));
                }
                if mods.is_async {
                    properties.insert("is_async".into(), serde_json::Value::Bool(true));
                }
                if mods.is_unsafe {
                    properties.insert("is_unsafe".into(), serde_json::Value::Bool(true));
                }
                if mods.is_const {
                    properties.insert("is_const".into(), serde_json::Value::Bool(true));
                }

                // Signature + params + return type for functions/methods, plus
                // BOUND edges for each generic constraint. Captures the
                // signature/params/return type on `function_item`.
                if matches!(label.as_str(), "Method" | "Function")
                    && def_node.kind() == "function_item"
                {
                    if let Some(sig) = signature_info(source, def_node) {
                        properties
                            .insert("signature".into(), serde_json::Value::String(sig.signature));
                        if let Some(rt) = sig.return_type {
                            properties.insert("return_type".into(), serde_json::Value::String(rt));
                        }
                        if !sig.params.is_empty() {
                            properties.insert(
                                "params".into(),
                                serde_json::to_value(&sig.params)
                                    .unwrap_or(serde_json::Value::Null),
                            );
                        }
                    }
                    if let Some(signature) = function_source_signature(source, def_node) {
                        properties.insert(
                            "source_signature".into(),
                            serde_json::Value::String(signature),
                        );
                    }

                    // BOUND edges: def → bound trait, one per `T: Trait`
                    // constraint (angle-bracket + where-clause). The `name`
                    // property carries the bare trait name for the resolver;
                    // `type_param` records which generic it constrains.
                    let line = node.start_position().row as u32 + 1;
                    for gb in generic_bounds(source, def_node) {
                        result.edges.push(ExtractedEdge {
                            edge_type: "BOUND".into(),
                            source_qualified_name: qname.clone(),
                            // Same-file guess; resolves directly to a trait
                            // defined in this file, else falls back to the
                            // name-based resolver via the `name` property.
                            // Traits are labeled "Interface".
                            target_qualified_name: format!("{file_path}::Interface::{}", gb.bound),
                            file_path: file_path.to_string(),
                            line,
                            properties: serde_json::json!({
                                "name": gb.bound,
                                "bound": gb.bound,
                                "type_param": gb.type_param,
                            }),
                        });
                    }
                }

                let properties = serde_json::Value::Object(properties);
                result.nodes.push(ExtractedNode {
                    label,
                    name: text.to_string(),
                    qualified_name: qname,
                    file_path: file_path.to_string(),
                    start_line: def_node.start_position().row as u32 + 1,
                    end_line: def_node.end_position().row as u32 + 1,
                    properties,
                });
            }
        }
    }

    // PASS 2 — imports. We emit **exactly one** import per top-level
    // `use_declaration` — we do NOT expand brace groups, renames, or globs.
    // `use a::{B, C};` is a single import whose `module_path` is the whole
    // `a::{B, C}` text and whose representative symbol is the FIRST
    // group member (`B`). The resolver then links that one symbol.
    //
    // Emitting one IMPORTS edge per `use_declaration` (rather than one per
    // braced name) avoids over-counting IMPORTS (`use x::{A, B}` would
    // otherwise produce 2 edges). The representative name goes in
    // `imported_name` (what the indexer's name-based resolver keys on) so the
    // edge still resolves to a single definition.
    for cq in queries.iter().filter(|cq| cq.kind == QueryKind::Imports) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, tree.root_node(), source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };
                if name != "path" {
                    continue;
                }
                let node = cap.node;
                let line = node.start_position().row as u32 + 1;
                // The full `use`-tree text is the module path, minus the
                // `use `/`;` that the query capture already excludes — the
                // capture is the `argument:` subtree.
                let full_path = node_text(source, node).trim().to_string();
                // The representative imported symbol — the first group member,
                // the original of a rename, or the last path segment. Empty
                // only for a bare `a::*` glob whose base is a keyword; the edge
                // is still emitted (one per `use`) but will not resolve.
                let items = expand_use_tree(source, node, "");
                let (imported_name, original_name, is_glob) = match items.first() {
                    Some(item) => (
                        import_representative_name(item),
                        item.original_name.clone(),
                        item.is_glob,
                    ),
                    None => (String::new(), String::new(), false),
                };
                // NOTE: no `Import` pseudo-node (forensics F2 + index perf).
                // The IMPORTS edge has the real `__file__` Module node as its
                // source and resolves its target by `imported_name`.
                result.edges.push(ExtractedEdge {
                    edge_type: "IMPORTS".into(),
                    source_qualified_name: file_qname.clone(),
                    target_qualified_name: format!("{file_path}::Import::{full_path}"),
                    file_path: file_path.to_string(),
                    line,
                    properties: serde_json::json!({
                        "path": full_path,
                        "imported_name": imported_name,
                        "original_name": original_name,
                        "glob": is_glob,
                    }),
                });
            }
        }
    }

    // PASS 3 — calls. Emit a `Call` node (existing behaviour) PLUS
    // a `CALLS` edge when we can determine the enclosing function
    // Multi-callee field accesses (`Foo::bar`) emit a
    // call node for `bar` so we keep the searchable surface, but
    // the edge uses the full path.
    for cq in queries.iter().filter(|cq| cq.kind == QueryKind::Calls) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, tree.root_node(), source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };
                if name != "callee" {
                    continue;
                }
                let node = cap.node;
                let text = node_text(source, node);
                // Rust's grammar makes receiver dispatch unambiguous here:
                // `value.method()` captures the field_identifier below a
                // field_expression, while bare/module/associated calls use
                // identifier/scoped_identifier nodes. Preserve that shape so
                // the resolver never guesses a same-named free function for a
                // receiver call.
                let callee_form = if node
                    .parent()
                    .is_some_and(|parent| parent.kind() == "field_expression")
                {
                    "receiver"
                } else {
                    "direct"
                };
                let receiver_owner = rust_receiver_owner(source, node);
                // NOTE: no `Call` pseudo-node (forensics F2 + index perf). The
                // CALLS edge below targets the real `file::Function::<text>`
                // qname (resolved by name when cross-file); the Call node was
                // never a resolution endpoint, only dead weight + search noise.
                if let Some(caller_qname) = enclosing_function_qname(source, node, file_path) {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: caller_qname,
                        // Target endpoint = a same-file *guess* qname.
                        // `text` is now the FINAL callee identifier
                        // (e.g. `do_it` of `helper::do_it()`), so this
                        // resolves directly for same-file free-function
                        // calls. When it does NOT resolve (cross-file,
                        // or a method), the indexer falls back to the
                        // name-based cross-file resolver, keyed on the
                        // `callee_name` property below.
                        target_qualified_name: format!("{file_path}::Function::{text}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        // `callee_name` is the bare final identifier the
                        // resolver matches against Function/Method node
                        // names project-wide. `callee_text` is kept for
                        // backwards-compatible diagnostics.
                        properties: {
                            let mut properties = serde_json::json!({
                            "callee_text": text,
                            "callee_name": text,
                            "callee_form": callee_form,
                            });
                            if let (Some(owner), Some(object)) =
                                (receiver_owner, properties.as_object_mut())
                            {
                                object.insert(
                                    "receiver_owner".into(),
                                    serde_json::Value::String(owner.to_string()),
                                );
                            }
                            properties
                        },
                    });
                }
            }
        }
    }

    // PASS 4+5 — usages (the unified reference model). There are NO separate
    // `TYPE_REF`/`USES` passes: every non-call, non-import identifier
    // reference is a single `USAGE` edge from the enclosing function (or the
    // file node) to whatever registered symbol its name resolves to — a type,
    // callable, `Variable`, or `Field`.
    //
    // We walk the whole tree once and emit a `USAGE` edge for each reference
    // node (`identifier` / `type_identifier` / `field_identifier` /
    // `scoped_identifier`) that is not inside a `call_expression` /
    // `macro_invocation` / `use_declaration` / `extern_crate_declaration`
    // (bounded 10-parent walk), not a definition name, and
    // not a keyword. The indexer resolves `ref_name` against the project's
    // registered symbols and keeps only unique matches (`resolve_unique`),
    // so references to locals/params
    // with no matching definition are emitted here and dropped at
    // resolution. This subsumes the old TYPE_REF pass (a `type_identifier`
    // in a type position is just another reference node) so structs, enums,
    // and traits still get their usage edges.
    {
        let mut emit = |node: Node<'_>, text: &str| {
            // The nearest enclosing function's qname, with the same file-node
            // fallback the resolver applies when the reference is not inside
            // any function.
            let source_qname = enclosing_function_qname(source, node, file_path)
                .unwrap_or_else(|| file_qname.clone());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                // The target is name-based: the resolver matches `ref_name`
                // against every registered symbol project-wide. The `__ref__`
                // placeholder is never a real node, so resolution always goes
                // through the name path (`USAGE_LABELS`).
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({
                    "ref_name": text,
                }),
            });
        };
        walk_rust_usages(source, tree.root_node(), &mut emit);
    }

    // PASS 6 — declared-type assignments (a variable's declared type). For each
    // `let x: T = …`, `const C: T = …`, `static S: T = …`, and struct
    // `field: T`, emit a `TYPE_ASSIGN` edge from the enclosing definition to
    // the declared type T. This is distinct from `TYPE_REF` (any type
    // mention): a `TYPE_ASSIGN` specifically records that a *named binding*
    // has declared type T, so the resolver can answer "what type is `x`?".
    // The `var_name` property carries the binding, `type_name` the bare type
    // for the name-based resolver; builtin primitives are skipped.
    for cq in queries
        .iter()
        .filter(|cq| cq.kind == QueryKind::TypeAssigns)
    {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, tree.root_node(), source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(cap_name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };
                if cap_name != "assign" {
                    continue;
                }
                let node = cap.node;
                // The variable/binding name field differs by node kind:
                // `let` uses `pattern`, the others use `name`.
                let var_node = node
                    .child_by_field_name("pattern")
                    .or_else(|| node.child_by_field_name("name"));
                let Some(var_node) = var_node else { continue };
                // Only simple identifier bindings carry a single declared
                // type (`let (a, b): (X, Y)` tuple patterns are skipped —
                // this requires an `identifier` left side).
                if !matches!(var_node.kind(), "identifier" | "field_identifier") {
                    continue;
                }
                let var_name = node_text(source, var_node);
                let Some(type_node) = node.child_by_field_name("type") else {
                    continue;
                };
                let Some(source_qname) = enclosing_def_qname(source, node, file_path) else {
                    // Top-level const/static with no enclosing def — skip;
                    // type-assigns are attached to an enclosing function.
                    continue;
                };
                let mut type_names: Vec<&str> = Vec::new();
                type_identifiers_in(source, type_node, &mut type_names);
                // The declared type is the first (outermost) concrete type
                // identifier; record it. Skip builtin primitives.
                let Some(&ty) = type_names.first() else {
                    continue;
                };
                if ty.is_empty() || is_builtin_rust_type(ty) {
                    continue;
                }
                result.edges.push(ExtractedEdge {
                    edge_type: "TYPE_ASSIGN".into(),
                    source_qualified_name: source_qname,
                    // Same-file guess; falls back to the name-based resolver
                    // for types defined elsewhere.
                    target_qualified_name: format!("{file_path}::Class::{ty}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "var_name": var_name,
                        "type_name": ty,
                    }),
                });
            }
        }
    }

    // PASS 7 — inheritance + enum members + associated items
    // (base-classes / enum-members / impls).
    //
    //  * `impl Trait for Type` → an `IMPLEMENTS` edge from the implementing
    //    type to the trait (target name in `name`/`trait_name` for the
    //    resolver). Inherent `impl Type` carries no `trait:` field and so does
    //    not match the `@impl_trait` pattern — no edge, as intended.
    //  * each `enum_variant` → an `EnumVariant` node plus a `DEFINES` edge from
    //    the owning enum, with the enum in the variant's qname.
    //  * associated `const`/`type` items inside an impl/trait block → a node
    //    (`AssocConst` / `AssocType`) whose qname is owned by the enclosing
    //    impl/trait type, mirroring the method qname scheme.
    for cq in queries
        .iter()
        .filter(|cq| cq.kind == QueryKind::Inheritance)
    {
        let trait_name_idx = cq.capture_index("trait_name");
        let impl_type_idx = cq.capture_index("impl_type");
        let enum_name_idx = cq.capture_index("enum_name");
        let enum_variant_idx = cq.capture_index("enum_variant");
        let assoc_const_idx = cq.capture_index("assoc_const");
        let assoc_type_idx = cq.capture_index("assoc_type");

        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, tree.root_node(), source);
        while let Some(m) = matches.next() {
            // ---- IMPLEMENTS: `impl Trait for Type` ----
            if let (Some(ti), Some(ii)) = (trait_name_idx, impl_type_idx) {
                let trait_cap = m.captures.iter().find(|c| c.index == ti);
                let type_cap = m.captures.iter().find(|c| c.index == ii);
                if let (Some(trait_cap), Some(type_cap)) = (trait_cap, type_cap) {
                    let trait_name = impl_type_name(source, trait_cap.node);
                    let impl_type = impl_type_name(source, type_cap.node);
                    if !trait_name.is_empty() && !impl_type.is_empty() {
                        let line = type_cap.node.start_position().row as u32 + 1;
                        result.edges.push(ExtractedEdge {
                            edge_type: "IMPLEMENTS".into(),
                            // Source = the implementing type. Same-file guess
                            // qname (resolves directly to a struct/enum defined
                            // in this file); the `type_name` property keys the
                            // name-based cross-file resolver.
                            source_qualified_name: format!("{file_path}::Class::{impl_type}"),
                            // Target = the trait. Same-file guess; `name` /
                            // `trait_name` carry the bare trait name for the
                            // resolver. Traits are labeled "Interface".
                            target_qualified_name: format!("{file_path}::Interface::{trait_name}"),
                            file_path: file_path.to_string(),
                            line,
                            properties: serde_json::json!({
                                "name": trait_name,
                                "trait_name": trait_name,
                                "type_name": impl_type,
                            }),
                        });
                    }
                }
            }

            // ---- enum variants ----
            if let (Some(eni), Some(evi)) = (enum_name_idx, enum_variant_idx) {
                let enum_cap = m.captures.iter().find(|c| c.index == eni);
                let variant_cap = m.captures.iter().find(|c| c.index == evi);
                if let (Some(enum_cap), Some(variant_cap)) = (enum_cap, variant_cap) {
                    let enum_name = node_text(source, enum_cap.node);
                    let variant_name = node_text(source, variant_cap.node);
                    if !enum_name.is_empty() && !variant_name.is_empty() {
                        let start_line = variant_cap.node.start_position().row as u32 + 1;
                        let end_line = variant_cap.node.end_position().row as u32 + 1;
                        let variant_qname = format!("{file_path}::{enum_name}::{variant_name}");
                        let enum_qname = format!("{file_path}::Enum::{enum_name}");
                        result.nodes.push(ExtractedNode {
                            label: "EnumVariant".into(),
                            name: variant_name.to_string(),
                            qualified_name: variant_qname.clone(),
                            file_path: file_path.to_string(),
                            start_line,
                            end_line,
                            properties: serde_json::json!({
                                "enum": enum_name,
                            }),
                        });
                        result.edges.push(ExtractedEdge {
                            edge_type: "DEFINES".into(),
                            source_qualified_name: enum_qname,
                            target_qualified_name: variant_qname,
                            file_path: file_path.to_string(),
                            line: start_line,
                            properties: serde_json::json!({
                                "member": "enum_variant",
                                "name": variant_name,
                                "enum": enum_name,
                            }),
                        });
                    }
                }
            }

            // ---- associated const ----
            if let Some(aci) = assoc_const_idx {
                if let Some(cap) = m.captures.iter().find(|c| c.index == aci) {
                    // Only associated consts (inside an impl/trait block) — a
                    // top-level / function-local `const` has no impl/trait
                    // owner and is left to the existing TYPE_ASSIGN pass.
                    if let Some(owner) = enclosing_impl_type(source, cap.node) {
                        let name = node_text(source, cap.node);
                        let node = cap.node;
                        let item = node.parent().unwrap_or(node);
                        result.nodes.push(ExtractedNode {
                            label: "AssocConst".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::{owner}::{name}"),
                            file_path: file_path.to_string(),
                            start_line: item.start_position().row as u32 + 1,
                            end_line: item.end_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "owner": owner,
                                "kind": "associated_const",
                            }),
                        });
                    }
                }
            }

            // ---- associated type ----
            if let Some(ati) = assoc_type_idx {
                if let Some(cap) = m.captures.iter().find(|c| c.index == ati) {
                    if let Some(owner) = enclosing_impl_type(source, cap.node) {
                        let name = node_text(source, cap.node);
                        let node = cap.node;
                        let item = node.parent().unwrap_or(node);
                        result.nodes.push(ExtractedNode {
                            label: "AssocType".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::{owner}::{name}"),
                            file_path: file_path.to_string(),
                            start_line: item.start_position().row as u32 + 1,
                            end_line: item.end_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "owner": owner,
                                "kind": "associated_type",
                            }),
                        });
                    }
                }
            }
        }
    }

    // We intentionally do not use `defs` after this point — the
    // node QNames produced in PASS 1 are what end up in the store;
    // future per-file diff / cross-reference passes can read the
    // resolved DefinitionSpan list. Touching `defs` here keeps the
    // Rust compiler from complaining about an unused local in the
    // PASS-2/3 cases where PASS 1 finds nothing.
    let _ = &defs;

    Ok(result)
}

/// One imported name extracted from a `use` declaration's use-tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImportedItem {
    /// The full `::`-joined path of the import, with any brace group / rename
    /// resolved (`use a::{B, C as D}` → `a::B` and `a::C`).
    pub(crate) path: String,
    /// The local binding name the resolver matches against definitions:
    /// the final segment, or the alias for a rename (`C as D` → `D`).
    pub(crate) imported_name: String,
    /// For a rename, the *original* name (`C as D` → `C`); equal to
    /// `imported_name` otherwise. Distinguishes the imported symbol from its
    /// local alias.
    pub(crate) original_name: String,
    /// True for a glob import (`use a::*`), which stays a single edge with an
    /// empty `imported_name`.
    pub(crate) is_glob: bool,
}

/// Expand a `use` declaration's argument node into one `ImportedItem` per
/// imported name. Handles plain paths (`a::b::C`), brace groups
/// (`a::{B, C as D}`, nested), renames (`a::B as C`), `self` in a group
/// (`a::{self, B}` → imports `a` and `a::B`), and globs (`a::*`, kept as a
/// single glob item). Handles brace-group / rename import expansion.
fn expand_use_tree(source: &[u8], node: Node<'_>, prefix: &str) -> Vec<ImportedItem> {
    let mut out = Vec::new();
    expand_use_tree_into(source, node, prefix, &mut out);
    out
}

/// The single representative symbol name for a whole `use` declaration.
/// Because one IMPORTS edge is emitted per `use` statement, it needs one name
/// to resolve against a project definition; it takes the first brace-group
/// member, the original name of a rename, or the last path segment (for a glob
/// or plain import). We derive that from the FIRST expanded item:
///   - a rename → the *original* name (strip ` as <alias>` and keep `X`),
///   - a glob → the last `::` segment of the pre-`*` path,
///   - otherwise → the item's `imported_name` (first group member / last
///     path segment / `self`'s prefix segment).
fn import_representative_name(first: &ImportedItem) -> String {
    if first.is_glob {
        // `a::b::*` → `b` (strip the trailing `*`/`::` then take the last segment).
        return first
            .path
            .trim_end_matches('*')
            .trim_end_matches("::")
            .rsplit("::")
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
    }
    // For a rename `X as Y`, resolve the ORIGINAL symbol `X`, not the
    // local alias `Y`. `original_name` already carries `X`'s last segment.
    if first.original_name != first.imported_name && !first.original_name.is_empty() {
        return first.original_name.clone();
    }
    first.imported_name.clone()
}

fn join_path(prefix: &str, seg: &str) -> String {
    if prefix.is_empty() {
        seg.to_string()
    } else {
        format!("{prefix}::{seg}")
    }
}

fn expand_use_tree_into(source: &[u8], node: Node<'_>, prefix: &str, out: &mut Vec<ImportedItem>) {
    match node.kind() {
        // `a::*` — glob, single edge, empty imported_name.
        "use_wildcard" => {
            let text = node_text(source, node);
            // Path = everything before `::*` joined onto the prefix.
            let base = text.trim_end_matches('*').trim_end_matches("::").trim();
            let path = join_path(prefix, base);
            out.push(ImportedItem {
                path,
                imported_name: String::new(),
                original_name: String::new(),
                is_glob: true,
            });
        }
        // `a::b::{...}` — scoped use list: descend into the brace group with
        // the path prefix applied.
        "scoped_use_list" => {
            let new_prefix = match node.child_by_field_name("path") {
                Some(p) => join_path(prefix, node_text(source, p)),
                None => prefix.to_string(),
            };
            if let Some(list) = node.child_by_field_name("list") {
                expand_use_tree_into(source, list, &new_prefix, out);
            }
        }
        // `{B, C as D, sub::E}` — iterate the group's named children.
        "use_list" => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    expand_use_tree_into(source, child, prefix, out);
                }
            }
        }
        // `B as D` — rename. The local binding is the alias.
        "use_as_clause" => {
            let orig_node = node.child_by_field_name("path");
            let alias_node = node.child_by_field_name("alias");
            if let (Some(orig), Some(alias)) = (orig_node, alias_node) {
                let orig_path = node_text(source, orig);
                let alias_name = node_text(source, alias);
                // The original may itself be a path (`a::B as C`); the
                // imported symbol's name is the final segment of the original.
                let orig_name = orig_path.rsplit("::").next().unwrap_or(orig_path);
                out.push(ImportedItem {
                    path: join_path(prefix, orig_path),
                    imported_name: alias_name.to_string(),
                    original_name: orig_name.to_string(),
                    is_glob: false,
                });
            }
        }
        // `self` inside a group (`a::{self, B}`) — binds the prefix's last
        // segment.
        "self" => {
            let name = prefix.rsplit("::").next().unwrap_or(prefix).to_string();
            out.push(ImportedItem {
                path: prefix.to_string(),
                imported_name: name.clone(),
                original_name: name,
                is_glob: false,
            });
        }
        // A plain `identifier` / `scoped_identifier` leaf — one import.
        "identifier" | "scoped_identifier" | "type_identifier" | "crate" | "super" => {
            let seg = node_text(source, node);
            let name = seg.rsplit("::").next().unwrap_or(seg).to_string();
            out.push(ImportedItem {
                path: join_path(prefix, seg),
                imported_name: name.clone(),
                original_name: name,
                is_glob: false,
            });
        }
        // Unknown wrapper — descend into named children so we don't silently
        // drop imports the grammar nests differently.
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    expand_use_tree_into(source, child, prefix, out);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Go extraction
// ---------------------------------------------------------------------------
//
// Handles the passes at the level Go's grammar supports, reusing the same
// `ExtractedNode` / `ExtractedEdge` conventions and the same name-based
// resolution keys (`callee_name`, `imported_name`) so the indexer's existing
// two-phase resolver links Go edges cross-file with NO indexer change:
//
//   * DEFINITIONS — `function_declaration` → `Function`; `method_declaration`
//     → `Method` owned by its receiver type (`{file}::{RecvType}::{name}`);
//     `type_spec` → a *type* node whose label follows this taxonomy:
//     a `struct_type` body → **"Class"** (NOT "Struct"), an
//     `interface_type` body → **"Interface"**, and any other body (a plain type
//     alias like `type Celsius int`) → the default **"Class"**. A free
//     function is `{file}::Function::{name}`.
//   * CALLS — final callee identifier (`add` of `add()`, `Println` of
//     `fmt.Println()`) → `CALLS` edge with the `callee_name` property, sourced
//     from the enclosing function/method qname.
//   * IMPORTS — each `import_spec` → an `IMPORTS` edge whose `imported_name` is
//     the final path segment of the imported package (`math/rand` → `rand`), or
//     the explicit alias when present (`m "math/rand"` → `m`), plus a
//     searchable `Import` node.
//   * docstrings — a run of leading `//` line comments immediately preceding the
//     declaration becomes the node's `doc` (one-line summary) and `doc_full`.

/// Go keyword / predeclared-identifier filter for the usages pass: the 25 Go
/// reserved words plus the predeclared builtins (`append`/`len`/`make`/…) and
/// the untyped constants (`true`/`false`/`nil`/`iota`). A reference to one of
/// these — e.g. `len` or `nil` — never becomes a USAGE.
fn is_go_keyword(name: &str) -> bool {
    matches!(
        name,
        "break"
            | "case"
            | "chan"
            | "const"
            | "continue"
            | "default"
            | "defer"
            | "else"
            | "fallthrough"
            | "for"
            | "func"
            | "go"
            | "goto"
            | "if"
            | "import"
            | "interface"
            | "map"
            | "package"
            | "range"
            | "return"
            | "select"
            | "struct"
            | "switch"
            | "type"
            | "var"
            | "true"
            | "false"
            | "nil"
            | "iota"
            | "append"
            | "cap"
            | "close"
            | "complex"
            | "copy"
            | "delete"
            | "imag"
            | "len"
            | "make"
            | "new"
            | "panic"
            | "print"
            | "println"
            | "real"
            | "recover"
    )
}

/// The receiver base type of a Go `method_declaration` (`*Adder` → `Adder`),
/// so a usage's enclosing-method qname is identical to the def-pass qname the
/// indexer registered for that method (`{file}::{RecvType}::{name}`). Returns
/// `None` for a free function.
fn go_receiver_type_name<'a>(source: &'a [u8], method: Node<'_>) -> Option<&'a str> {
    let receiver = method.child_by_field_name("receiver")?;
    let mut decl = None;
    for i in 0..receiver.named_child_count() {
        if let Some(c) = receiver.named_child(i) {
            if c.kind() == "parameter_declaration" {
                decl = Some(c);
                break;
            }
        }
    }
    let ty = decl?.child_by_field_name("type")?;
    Some(go_base_type_text(source, ty))
}

/// Strip pointer / generic / qualified wrappers off a Go receiver type node and
/// return the base identifier text.
fn go_base_type_text<'a>(source: &'a [u8], node: Node<'_>) -> &'a str {
    match node.kind() {
        "type_identifier" => node_text(source, node),
        "pointer_type" => node
            .named_child(0)
            .map(|n| go_base_type_text(source, n))
            .unwrap_or_else(|| node_text(source, node)),
        "generic_type" => node
            .child_by_field_name("type")
            .map(|n| go_base_type_text(source, n))
            .unwrap_or_else(|| node_text(source, node)),
        "qualified_type" => node
            .child_by_field_name("name")
            .map(|n| node_text(source, n))
            .unwrap_or_else(|| node_text(source, node)),
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(c) = node.named_child(i) {
                    if c.kind() == "type_identifier" {
                        return node_text(source, c);
                    }
                }
            }
            node_text(source, node)
        }
    }
}

/// The qname of the Go function/method that lexically encloses `node`, built to
/// match the Go definitions pass exactly (`{file}::Function::{name}` for a free
/// function, `{file}::{RecvType}::{name}` for a receiver method). It walks to
/// the nearest `function_declaration` / `method_declaration` and, when the
/// function is a method, prefixes the receiver type. A `func_literal` (closure)
/// has no name, so the walk continues past it to the outer named function.
/// Returns `None` at file scope; the caller then
/// uses the per-file `__file__` module qname.
fn go_enclosing_func_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        match cur.kind() {
            "function_declaration" => {
                let name_node = cur.child_by_field_name("name")?;
                let name = node_text(source, name_node);
                return Some(format!("{file_path}::Function::{name}"));
            }
            "method_declaration" => {
                let name_node = cur.child_by_field_name("name")?;
                let name = node_text(source, name_node);
                return Some(match go_receiver_type_name(source, cur) {
                    Some(t) => format!("{file_path}::{t}::{name}"),
                    None => format!("{file_path}::Function::{name}"),
                });
            }
            _ => {}
        }
        p = cur.parent();
    }
    None
}

fn extract_go(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Go)
        .map_err(|e| greppy_core::Error::Parse(format!("compile go queries: {e}")))?;
    let mut result =
        crate::spec::spec_extract(Language::Go, &crate::spec::GO, queries, source, file_path)?;

    // C-taxonomy relabel for Go `type_spec` nodes.
    //
    // The shared `spec::adjusted_rule` labels a Go `type_spec` by its inner
    // body: `struct_type` → "Struct", `interface_type` → "Interface", anything
    // else → "Type". This extractor uses a single canonical taxonomy instead:
    // a struct body is a **"Class"** (not "Struct"), an interface body is
    // an **"Interface"**, and every other `type_spec` — a plain type alias such
    // as `type Celsius int` — falls through to the *default* label **"Class"**.
    // Node tables count by label, so this pass normalises Go type labels
    // without touching the shared spec machinery (owned by a
    // parallel component) or any other language's `extract_*`.
    //
    // Only `type_spec` rules can emit "Struct"/"Interface"/"Type" in the Go spec
    // (functions/methods emit "Function"/"Method"), so keying off the emitted
    // label is unambiguous. The qname's type segment mirrors the label
    // (`{file}::{label}::{name}`, see `spec::def_label_and_qname`), so it is
    // rewritten in lock-step to keep node identity and any name-based resolution
    // consistent.
    for node in &mut result.nodes {
        let new_label = match node.label.as_str() {
            "Struct" | "Type" => "Class",
            _ => continue,
        };
        let old_seg = format!("::{}::", node.label);
        let new_seg = format!("::{new_label}::");
        node.qualified_name = node.qualified_name.replacen(&old_seg, &new_seg, 1);
        node.label = new_label.to_string();
    }

    // USAGE pass. For each Go reference node — `identifier`,
    // `type_identifier`, `field_identifier`, or `package_identifier` — emit a
    // `USAGE` edge from the enclosing
    // function/method (or the per-file `__file__` module qname at file scope)
    // to the identifier text, unless the reference:
    //   * is a definition *name* (the `name:` field of its parent — the def
    //     itself),
    //   * sits inside a `call_expression` (already a CALLS edge; note this
    //     excludes the *arguments* of a call too, since the walk covers the
    //     whole call subtree), or
    //   * sits inside an import declaration, or
    //   * is a Go keyword / predeclared builtin.
    // The indexer resolves `ref_name` against every registered symbol and drops
    // it when the name is not unique project-wide (`USAGE_LABELS`), so no real
    // target qname is needed — the `<file>::__usage__::{name}` placeholder is
    // never resolved directly.
    let tree = crate::parse(Language::Go, source)?;
    for cq in queries.iter().filter(|cq| cq.kind == QueryKind::Usages) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, tree.root_node(), source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(cap_name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };
                if cap_name != "use" {
                    continue;
                }
                let node = cap.node;
                let text = node_text(source, node);
                if text.is_empty() || is_go_keyword(text) {
                    continue;
                }
                if is_definition_name(node) {
                    continue;
                }
                if is_inside_kind(node, &["call_expression"]) {
                    continue;
                }
                if is_inside_kind(node, &["import_declaration", "import_spec"]) {
                    continue;
                }
                let source_qname = go_enclosing_func_qname(source, node, file_path)
                    .unwrap_or_else(|| format!("{file_path}::__file__"));
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: source_qname,
                    target_qualified_name: format!("{file_path}::__usage__::{text}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "ref_name": text,
                    }),
                });
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Ruby extraction
// ---------------------------------------------------------------------------
//
// The uniform `spec_extract` template gives the base three passes (definitions,
// calls, imports); Ruby layers several more definition facets on top that the
// template cannot express. `extract_ruby` runs the spec path and then
// supplements its output so the node/edge tables are complete.
//
// The supplementary walk does, per file:
//
//   * one per-file **Module** node (the `program` root) — added by the indexer's
//     structural pass here, so the spec path must NOT relabel a real `module`
//     decl to keep that slot;
//   * every `class` AND `module` declaration → a **"Class"** node
//     (a Ruby `module` is not one of the Interface/Enum/Type special cases, so
//     `module Foo` is a "Class");
//   * every `method` / `singleton_method` node → a **"Function"** node
//     (Ruby never sees a receiver so the label stays "Function"), AND, when the
//     def sits in a class/module body, ALSO a **"Method"** node owned by that
//     type. Every Ruby def is therefore
//     counted once as Function and — if nested — once as Method;
//   * every module-level `assignment` (LHS `identifier` or `constant`) → a
//     **"Variable"**: both file-top-level assignments and
//     assignments directly in a class/module body.
//     Assignments inside a method body are NOT variables.
//
// Edges, on top of the spec's CALLS / IMPORTS:
//   * **DEFINES_METHOD** — one per owned method, from the owning class/module
//     node to the method node.
//   * **USES** — every Ruby bare `identifier` read that is not a definition,
//     call endpoint, assignment target, or keyword;
//   * **TYPE_REF** — statically named constant receivers such as `Widget.new`
//     and `Helper.do_it`, sourced from the enclosing method/module qname.
//
// The spec path already labels a `module` decl "Module"; we relabel those to
// "Class" here (leaving the indexer's synthetic per-file Module untouched, since
// that node is added later and never passes through the parser output).

fn extract_ruby(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Ruby)
        .map_err(|e| greppy_core::Error::Parse(format!("compile ruby queries: {e}")))?;
    let mut result = crate::spec::spec_extract(
        Language::Ruby,
        &crate::spec::RUBY,
        queries,
        source,
        file_path,
    )?;

    // (1) Relabel real `module` decls: the spec path stamps them "Module" with
    // qname `{file}::Module::{name}`; C labels a Ruby module declaration a
    // "Class". Rewrite the label and the qname's type segment in lock-step so
    // node identity stays consistent. Method owner qnames use the owner's *name*
    // (`{file}::{ModuleName}::{method}`), not the label, so they are unaffected.
    for node in &mut result.nodes {
        if node.label == "Module" {
            node.label = "Class".into();
            let old = "::Module::";
            let new = "::Class::";
            node.qualified_name = node.qualified_name.replacen(old, new, 1);
        }
    }

    let tree = crate::parse(Language::Ruby, source)?;
    let root = tree.root_node();

    // Preserve statically named receivers (`Helper.do_it`) so the indexer can
    // resolve the call to the owned singleton Method instead of its duplicate
    // free-Function facet.
    ruby_annotate_receiver_calls(source, root, file_path, &mut result);
    ruby_annotate_require_imports(source, &mut result);

    // (2) Function-per-def + DEFINES_METHOD + Variable passes, walking the tree
    // once. `spec_definitions` already emitted the Method (owned) / Function
    // (free) node for each def; we ADD the second "Function" node for every
    // def C double-counts, plus the class/module → method DEFINES_METHOD edge.
    ruby_defs_pass(source, root, file_path, &mut result);

    // (3) TYPE_REF / USES pass.
    let file_module_qname = format!("{file_path}::__file__");
    ruby_emit_references(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// Mark Ruby calls whose receiver is a statically named constant as receiver
/// dispatch. `def self.x` is indexed as an owned Method (`Owner::x`) plus a
/// compatibility Function facet; this metadata lets the indexer select the
/// Method deterministically for `Owner.x` call sites, including cross-file
/// singleton methods on modules.
fn ruby_annotate_receiver_calls(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            stack.push(child);
        }
        if node.kind() != "call" {
            continue;
        }
        let Some(receiver) = node.child_by_field_name("receiver") else {
            continue;
        };
        if receiver.kind() != "constant" {
            continue;
        }
        let Some(method) = node.child_by_field_name("method") else {
            continue;
        };
        let owner = node_text(source, receiver);
        let name = node_text(source, method);
        if owner.is_empty() || name.is_empty() {
            continue;
        }
        let line = method.start_position().row as u32 + 1;
        let Some(edge) = result.edges.iter_mut().find(|edge| {
            edge.edge_type == "CALLS"
                && edge.line == line
                && edge
                    .properties
                    .get("callee_name")
                    .and_then(|value| value.as_str())
                    == Some(name)
                && edge.properties.get("receiver_owner").is_none()
        }) else {
            continue;
        };
        edge.target_qualified_name = format!("{file_path}::{owner}::{name}");
        if let Some(properties) = edge.properties.as_object_mut() {
            properties.insert(
                "callee_form".into(),
                serde_json::Value::String("receiver".into()),
            );
            properties.insert(
                "receiver_owner".into(),
                serde_json::Value::String(owner.to_string()),
            );
        }
    }
}

/// Mark Ruby `require` imports as file-module imports. `require_relative` also
/// carries its relative-path semantics so the indexer can resolve the exact
/// `<required-file>::__file__` Module node rather than guessing a definition
/// whose spelling happens to match the file stem.
fn ruby_annotate_require_imports(source: &[u8], result: &mut ExtractionResult) {
    let lines: Vec<&[u8]> = source.split(|byte| *byte == b'\n').collect();
    for edge in result
        .edges
        .iter_mut()
        .filter(|edge| edge.edge_type == "IMPORTS")
    {
        let line = edge
            .line
            .checked_sub(1)
            .and_then(|index| lines.get(index as usize))
            .and_then(|line| std::str::from_utf8(line).ok())
            .map(str::trim_start)
            .unwrap_or("");
        if let Some(properties) = edge.properties.as_object_mut() {
            properties.insert(
                "filesystem_module_import".into(),
                serde_json::Value::Bool(true),
            );
            if line.starts_with("require_relative") {
                properties.insert(
                    "ruby_require_relative".into(),
                    serde_json::Value::Bool(true),
                );
            }
        }
    }
}

/// The nearest enclosing Ruby callable qname for `node`'s source endpoint:
/// the closest `method` /
/// `singleton_method` ancestor, owned by its nearest enclosing `class`/`module`
/// (`{file}::{Owner}::{name}`) or free (`{file}::Function::{name}`). Returns
/// `None` at file scope (the caller substitutes the file Module qname).
fn ruby_enclosing_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if matches!(cur.kind(), "method" | "singleton_method") {
            let name = cur
                .child_by_field_name("name")
                .map(|n| node_text(source, n).to_string())?;
            // Nearest enclosing class/module (its `name:`) owns the method.
            let mut owner: Option<String> = None;
            let mut q = cur.parent();
            while let Some(anc) = q {
                match anc.kind() {
                    "class" | "module" => {
                        if let Some(cn) = anc.child_by_field_name("name") {
                            owner = Some(node_text(source, cn).to_string());
                        }
                        break;
                    }
                    "method" | "singleton_method" => break,
                    _ => {}
                }
                q = anc.parent();
            }
            return Some(match owner {
                Some(o) => format!("{file_path}::{o}::{name}"),
                None => format!("{file_path}::Function::{name}"),
            });
        }
        p = cur.parent();
    }
    None
}

/// The owning class/module *name* for a def node (its nearest enclosing
/// `class`/`module`), or `None` when the def is at file scope (a free
/// `Function`). Used to reconstruct the Method qname and the DEFINES_METHOD
/// endpoints so they line up with the spec-emitted nodes.
fn ruby_def_owner_name<'a>(source: &'a [u8], def_node: Node<'_>) -> Option<&'a str> {
    let mut p = def_node.parent();
    while let Some(cur) = p {
        if matches!(cur.kind(), "class" | "module") {
            return cur
                .child_by_field_name("name")
                .map(|n| node_text(source, n));
        }
        p = cur.parent();
    }
    None
}

/// Walk the tree emitting, for every `method` / `singleton_method`:
///   * an ADDITIONAL "Function" node labelling every Ruby def node "Function"
///     regardless of nesting — the spec path already emitted the
///     "Method"/"Function" node, so this is the second count;
///   * a `DEFINES_METHOD` edge from the owning class/module node to the Method
///     node when the def is nested.
/// And, for every `class` / `module` def, extracts its body's direct-child
/// `assignment`s as module-level "Variable" nodes.
/// File-top-level `assignment`s are handled by the same routine on the root.
fn ruby_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    // ── Variables: file top level (program's direct children). ──────────
    // Iterate the file-root children (unwrapping simple statement wrappers) and
    // take each `assignment` whose LHS is an identifier/constant.
    ruby_emit_body_variables(source, root, file_path, result);

    // ── Function nodes + DEFINES_METHOD + class/module body variables. ──
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "method" | "singleton_method" => {
                if let Some(name) = node
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n))
                {
                    if !name.is_empty() {
                        // Second, "Function"-labelled count for this def.
                        result.nodes.push(ExtractedNode {
                            label: "Function".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::Function::{name}"),
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                        // DEFINES_METHOD: owner class/module → the owned Method
                        // node the spec pass emitted at `{file}::{Owner}::{name}`.
                        if let Some(owner) = ruby_def_owner_name(source, node) {
                            result.edges.push(ExtractedEdge {
                                edge_type: "DEFINES_METHOD".into(),
                                source_qualified_name: format!("{file_path}::Class::{owner}"),
                                target_qualified_name: format!("{file_path}::{owner}::{name}"),
                                file_path: file_path.to_string(),
                                line: node.start_position().row as u32 + 1,
                                properties: serde_json::json!({}),
                            });
                        }
                    }
                }
                // Do not descend into a method body: any nested def is rare and
                // C does not re-walk method bodies for further defs.
            }
            "class" | "module" => {
                // Class/module body direct-child assignments → Variables.
                if let Some(body) = node.child_by_field_name("body") {
                    ruby_emit_body_variables(source, body, file_path, result);
                }
                // Descend so nested classes/modules and their methods are
                // reached (their owner is computed by parent walk).
                let mut c = node.walk();
                for child in node.named_children(&mut c) {
                    stack.push(child);
                }
            }
            _ => {
                let mut c = node.walk();
                for child in node.named_children(&mut c) {
                    stack.push(child);
                }
            }
        }
    }
}

/// Emit a "Variable" node for each direct-child `assignment` of `container`
/// (the file root's body, or a class/module `body_statement`) whose LHS is a
/// plain `identifier` or `constant`: the `left:` field must be `identifier`
/// or `constant`; anything else (multiple assignment, `obj.attr =`, `h[k] =`)
/// is skipped. The `_` placeholder and empty names are dropped.
fn ruby_emit_body_variables(
    source: &[u8],
    container: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut c = container.walk();
    for child in container.named_children(&mut c) {
        // Unwrap a bare statement wrapper if the grammar ever introduces one;
        // in current tree-sitter-ruby a top-level / body assignment is a direct
        // `assignment` child, so this is just a defensive passthrough.
        let assign = if child.kind() == "assignment" {
            Some(child)
        } else {
            None
        };
        let Some(assign) = assign else { continue };
        let Some(left) = assign.child_by_field_name("left") else {
            continue;
        };
        if !matches!(left.kind(), "identifier" | "constant") {
            continue;
        }
        let vname = node_text(source, left);
        if vname.is_empty() || vname == "_" {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: vname.to_string(),
            qualified_name: format!("{file_path}::Variable::{vname}"),
            file_path: file_path.to_string(),
            start_line: assign.start_position().row as u32 + 1,
            end_line: assign.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// Recursively classify Ruby references. Bare identifier reads become `USES`;
/// a constant used as the receiver of a call (`Widget.new`, `Helper.do_it`)
/// becomes `TYPE_REF` because Ruby constants are the grammar's statically named
/// class/module surface. Scope-resolution constants are handled separately as
/// value references. Certified Ruby references preserve these logical labels
/// when persisted instead of folding them into the compatibility `USAGE` kind.
fn ruby_emit_references(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let is_identifier_use = node.kind() == "identifier"
        && !is_inside_kind(node, &["call", "command_call"])
        && !is_definition_name(node)
        && !ruby_is_assignment_lhs(node);
    let is_type_ref = node.kind() == "constant"
        && (ruby_is_constant_call_receiver(node) || ruby_is_scope_owner(node));
    let is_constant_use = node.kind() == "constant"
        && !is_type_ref
        && !ruby_is_constant_declaration_name(node)
        && !ruby_is_assignment_lhs(node);
    if is_identifier_use || is_constant_use || is_type_ref {
        let text = node_text(source, node);
        if !text.is_empty() && !is_ruby_usage_keyword(text) {
            let source_qname = ruby_enclosing_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: if is_type_ref { "TYPE_REF" } else { "USES" }.into(),
                source_qualified_name: source_qname,
                target_qualified_name: if is_type_ref {
                    format!("{file_path}::Class::{text}")
                } else {
                    format!("{file_path}::__ref__::{text}")
                },
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: if is_type_ref {
                    serde_json::json!({
                        "type_name": text,
                        "preserve_reference_kind": true,
                    })
                } else {
                    serde_json::json!({
                        "ref_name": text,
                        "preserve_reference_kind": true,
                    })
                },
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        ruby_emit_references(source, child, file_path, file_module_qname, result);
    }
}

fn ruby_is_constant_declaration_name(node: Node<'_>) -> bool {
    if node
        .parent()
        .is_some_and(|parent| parent.kind() == "scope_resolution")
    {
        return false;
    }
    is_definition_name(node)
}

fn ruby_is_assignment_lhs(node: Node<'_>) -> bool {
    node.parent().is_some_and(|parent| {
        parent.kind() == "assignment"
            && parent.child_by_field_name("left").is_some_and(|left| {
                left.start_byte() == node.start_byte() && left.end_byte() == node.end_byte()
            })
    })
}

fn ruby_is_constant_call_receiver(node: Node<'_>) -> bool {
    node.parent().is_some_and(|parent| {
        parent.kind() == "call"
            && parent
                .child_by_field_name("receiver")
                .is_some_and(|receiver| {
                    receiver.start_byte() == node.start_byte()
                        && receiver.end_byte() == node.end_byte()
                })
    })
}

fn ruby_is_scope_owner(node: Node<'_>) -> bool {
    node.parent().is_some_and(|parent| {
        parent.kind() == "scope_resolution"
            && parent.child_by_field_name("scope").is_some_and(|scope| {
                scope.start_byte() == node.start_byte() && scope.end_byte() == node.end_byte()
            })
    })
}

/// Ruby keyword / literal filter. References whose text is one of these never
/// emit a usage.
fn is_ruby_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "nil"
            | "None"
            | "undefined"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "struct"
            | "enum"
            | "interface"
            | "trait"
            | "impl"
            | "import"
            | "export"
            | "package"
            | "module"
            | "use"
            | "require"
            | "include"
            | "new"
            | "delete"
            | "this"
            | "self"
            | "super"
            | "public"
            | "private"
            | "protected"
            | "static"
            | "const"
            | "var"
            | "let"
            | "function"
            | "def"
            | "fn"
            | "func"
            | "fun"
            | "proc"
            | "sub"
            | "method"
            | "async"
            | "await"
            | "yield"
    )
}

// ---------------------------------------------------------------------------
// Java extraction
// ---------------------------------------------------------------------------
//
// Handles the passes at the level Java's grammar supports, reusing the same
// `ExtractedNode` / `ExtractedEdge` conventions and name-based resolution keys
// (`callee_name`, `imported_name`) so the indexer's existing two-phase resolver
// links Java edges cross-file with NO indexer change:
//
//   * DEFINITIONS — `class` / `interface` / `enum` declarations → `Class` /
//     `Interface` / `Enum`; `method` / `constructor` declarations → `Method`
//     owned by the enclosing class (`{file}::{Class}::{name}`). A method with
//     no enclosing class is treated as a free `Function`.
//   * CALLS — the final `name:` of a `method_invocation` → `CALLS` edge with the
//     `callee_name` property, sourced from the enclosing method qname.
//   * IMPORTS — each `import_declaration` → an `IMPORTS` edge whose
//     `imported_name` is the final segment of the imported path
//     (`java.util.List` → `List`), plus a searchable `Import` node.
//   * docstrings — a leading Javadoc block comment (`/** … */`) immediately
//     preceding the definition becomes the node's `doc` (one-line summary) and
//     `doc_full` properties.

fn extract_java(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Java)
        .map_err(|e| greppy_core::Error::Parse(format!("compile java queries: {e}")))?;
    // The uniform spec path already emits the type nodes (Class / Interface /
    // Enum), the owned Method / constructor nodes, and the CALLS / IMPORTS
    // edges. Java additionally emits member
    // definitions the uniform template cannot express, so we layer a bespoke
    // member pass on top of the spec output:
    //
    //   * every class-body `field_declaration` yields BOTH a `Field` node
    //     (qname `{type}::{name}`) AND a `Variable`
    //     node (qname `{file}::Variable::{name}`) — one of each;
    //   * every `enum_constant` yields a `Variable` node (qname `{enum}::{name}`);
    //   * every owned `method_declaration` / `constructor_declaration` yields a
    //     `DEFINES_METHOD` edge from its enclosing type node to the method node.
    let mut result = crate::spec::spec_extract(
        Language::Java,
        &crate::spec::JAVA,
        queries,
        source,
        file_path,
    )?;
    java_member_pass(source, file_path, &mut result)?;
    java_usage_pass(source, file_path, &mut result)?;
    java_collapse_imports_per_package(&mut result);
    Ok(result)
}

/// Collapse this file's `IMPORTS` edges to one per (file, PACKAGE).
///
/// A Java import like `import a.b.C;` names a symbol in package `a.b`. Two
/// imports from the same package (`import a.b.C; import a.b.D;`) resolve to the
/// same package, so we collapse them to a single `IMPORTS` edge per package
/// rather than one per imported symbol. The package is the path minus its final
/// member segment; an on-demand `a.b.*` import is already a package.
///
/// Choosing the representative: the USAGE/CALLS disambiguation reads back the
/// resolved `IMPORTS` edge targets, so to preserve reference resolution we keep
/// the import whose symbol is actually referenced by a `USAGE` in this file when
/// there is one, and the first import otherwise. Either way the surviving edge's
/// `imported_name` resolves to a real symbol, so exactly one edge is emitted per
/// (file, package) — the choice only decides *which* same-package symbol carries
/// the single edge.
fn java_collapse_imports_per_package(result: &mut ExtractionResult) {
    // Names referenced by a USAGE edge (per source file). An import whose
    // symbol appears here must survive the collapse so the reference resolves.
    let referenced: std::collections::HashSet<&str> = result
        .edges
        .iter()
        .filter(|e| e.edge_type == "USAGE")
        .filter_map(|e| e.properties.get("ref_name").and_then(|v| v.as_str()))
        .collect();

    // Index of the kept IMPORTS edge for each (source Module qname, package).
    let mut kept: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    // Indices to drop (superseded same-package imports).
    let mut drop: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for (i, edge) in result.edges.iter().enumerate() {
        if edge.edge_type != "IMPORTS" {
            continue;
        }
        let path = edge
            .properties
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let imported_name = edge
            .properties
            .get("imported_name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let key = (
            edge.source_qualified_name.clone(),
            java_import_package(path),
        );
        match kept.get(&key) {
            None => {
                kept.insert(key, i);
            }
            Some(&prev) => {
                // Prefer the import whose symbol is referenced by a USAGE. If
                // the incumbent is referenced (or the newcomer is not), drop
                // the newcomer; otherwise the newcomer supersedes it.
                let prev_referenced = result.edges[prev]
                    .properties
                    .get("imported_name")
                    .and_then(|v| v.as_str())
                    .map(|n| referenced.contains(n))
                    .unwrap_or(false);
                if prev_referenced || !referenced.contains(imported_name) {
                    drop.insert(i);
                } else {
                    drop.insert(prev);
                    kept.insert(key, i);
                }
            }
        }
    }

    let mut idx = 0;
    result.edges.retain(|_| {
        let keep = !drop.contains(&idx);
        idx += 1;
        keep
    });
}

/// The PACKAGE an `import` path resolves to. Progressively strip the trailing
/// dotted segment until the prefix matches a package key:
///   * a symbol import `a.b.C` → strip the member → package `a.b`;
///   * an on-demand import `a.b.*` → the `*` is the member placeholder, so the
///     package is `a.b` directly (strip only the `.*`);
///   * a single-segment import (`Foo`) has no package prefix, so it keys on
///     itself and is never merged with another import.
fn java_import_package(path: &str) -> String {
    // On-demand import: the trailing `.*` already denotes "everything in the
    // package", so the package is the path with the glob removed.
    if let Some(pkg) = path.strip_suffix(".*") {
        return pkg.trim_end_matches('.').to_string();
    }
    let trimmed = path.trim_end_matches('.');
    match trimmed.rfind('.') {
        Some(idx) => trimmed[..idx].to_string(),
        None => trimmed.to_string(),
    }
}

/// Java type-declaration node kinds that own members, mapped to the label the
/// spec path stamps on their node (`class_declaration` → "Class", …). The
/// member pass uses this both to find bodies to scan and to reconstruct the
/// owning type node's qname for `DEFINES_METHOD` edges (`{file}::{label}::{name}`).
fn java_type_label(kind: &str) -> Option<&'static str> {
    match kind {
        "class_declaration" => Some("Class"),
        "interface_declaration" => Some("Interface"),
        "enum_declaration" => Some("Enum"),
        _ => None,
    }
}

/// Supplementary Java member pass: appends `Field` / `Variable` member nodes and
/// `DEFINES_METHOD` edges to a spec-extracted result. Emits a "Field" and a
/// "Variable" node per class field, a "Variable" per enum member, and a
/// `DEFINES_METHOD` edge from each type to its methods.
fn java_member_pass(
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let tree = crate::parse(Language::Java, source)?;
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if let Some(label) = java_type_label(node.kind()) {
            java_type_members(source, file_path, node, label, result);
        }
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                stack.push(child);
            }
        }
    }
    Ok(())
}

/// Emit the member nodes + `DEFINES_METHOD` edges for one Java type declaration
/// `type_node` (kind → `label`). Only the type's OWN body is scanned; nested
/// types are reached by the outer walk, so their members are attributed to the
/// correct owner.
fn java_type_members(
    source: &[u8],
    file_path: &str,
    type_node: Node<'_>,
    label: &str,
    result: &mut ExtractionResult,
) {
    let Some(type_name) = type_node
        .child_by_field_name("name")
        .map(|n| node_text(source, n))
    else {
        return;
    };
    // The spec path names the type node `{file}::{label}::{name}` (a
    // `DefRule::ty` free qname) and names an owned member `{file}::{type}::{name}`
    // (the enclosing-owner qname). Reconstruct both so our edges/nodes line up
    // with the nodes the spec pass already emitted.
    let type_qname = format!("{file_path}::{label}::{type_name}");
    let member_owner_prefix = format!("{file_path}::{type_name}");

    let Some(body) = type_node.child_by_field_name("body") else {
        return;
    };

    for i in 0..body.named_child_count() {
        let Some(child) = body.named_child(i) else {
            continue;
        };
        match child.kind() {
            // `field_declaration` → one Field node + one Variable node, both
            // keyed on the first `variable_declarator`'s name (C takes the
            // first declarator only).
            "field_declaration" => {
                let Some(fname) = java_field_name(source, child) else {
                    continue;
                };
                if fname.is_empty() || fname == "_" {
                    continue;
                }
                let start = child.start_position().row as u32 + 1;
                let end = child.end_position().row as u32 + 1;
                let mut props = serde_json::Map::new();
                if let Some(ty) = child.child_by_field_name("type") {
                    props.insert(
                        "return_type".into(),
                        serde_json::Value::String(node_text(source, ty).to_string()),
                    );
                }
                // Field: owned by the enclosing type (C qname `{type}.{name}`;
                // greppy uses `::` throughout — `{type_owner}::{name}`).
                result.nodes.push(ExtractedNode {
                    label: "Field".into(),
                    name: fname.to_string(),
                    qualified_name: format!("{member_owner_prefix}::{fname}"),
                    file_path: file_path.to_string(),
                    start_line: start,
                    end_line: end,
                    properties: serde_json::Value::Object(props),
                });
                // Variable: a distinct module-scoped Variable node for the
                // same field (a different qname, so the two never collide).
                result.nodes.push(ExtractedNode {
                    label: "Variable".into(),
                    name: fname.to_string(),
                    qualified_name: format!("{file_path}::Variable::{fname}"),
                    file_path: file_path.to_string(),
                    start_line: start,
                    end_line: end,
                    properties: serde_json::json!({}),
                });
            }
            // Enum members → Variable nodes.
            "enum_constant" => {
                let Some(mname) = child
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n))
                else {
                    continue;
                };
                if mname.is_empty() {
                    continue;
                }
                result.nodes.push(ExtractedNode {
                    label: "Variable".into(),
                    name: mname.to_string(),
                    qualified_name: format!("{type_qname}::{mname}"),
                    file_path: file_path.to_string(),
                    start_line: child.start_position().row as u32 + 1,
                    end_line: child.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            // Owned method / constructor → a `DEFINES_METHOD` edge from the
            // enclosing type node to the method node. The method node itself is
            // already emitted by the spec definitions pass with qname
            // `{file}::{type}::{name}`; the indexer resolves this edge's two
            // endpoints by direct qname lookup (its default edge-type path).
            "method_declaration" | "constructor_declaration" => {
                let Some(mname) = child
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n))
                else {
                    continue;
                };
                if mname.is_empty() {
                    continue;
                }
                result.edges.push(ExtractedEdge {
                    edge_type: "DEFINES_METHOD".into(),
                    source_qualified_name: type_qname.clone(),
                    target_qualified_name: format!("{member_owner_prefix}::{mname}"),
                    file_path: file_path.to_string(),
                    line: child.start_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            _ => {}
        }
    }
}

/// The name of a Java `field_declaration`'s first `variable_declarator`.
fn java_field_name<'a>(source: &'a [u8], field: Node<'_>) -> Option<&'a str> {
    let decl = field
        .child_by_field_name("declarator")
        .or_else(|| first_child_of_kind_java(field, "variable_declarator"))?;
    decl.child_by_field_name("name")
        .map(|n| node_text(source, n))
}

/// First named child of `node` whose kind is `kind`.
fn first_child_of_kind_java<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    (0..node.named_child_count())
        .filter_map(|i| node.named_child(i))
        .find(|c| c.kind() == kind)
}

/// How far to walk up the parent chain when deciding whether a
/// reference sits inside a call/import, so we neither miss nor over-emit a
/// usage.
const JAVA_USAGE_MAX_PARENT_DEPTH: usize = 10;

/// Node kinds that make a reference a CALL (already counted as a CALLS edge, so
/// NOT a usage).
const JAVA_CALL_NODE_KINDS: &[&str] = &["method_invocation", "object_creation_expression"];

/// Node kinds that make a reference part of an import (NOT a usage):
/// `import_declaration`, `extends`, `import`.
const JAVA_IMPORT_NODE_KINDS: &[&str] = &["import_declaration", "extends", "import"];

/// Java keyword / builtin-type set that usages are filtered against. This is NOT
/// just language keywords: it also lists the common JDK type names (`System`,
/// `String`, `List`, …) so a reference to one of them never becomes a USAGE
/// edge.
const JAVA_USAGE_KEYWORDS: &[&str] = &[
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "true",
    "try",
    "void",
    "volatile",
    "while",
    "var",
    "record",
    "sealed",
    "permits",
    "yield",
    "System",
    "String",
    "Integer",
    "Long",
    "Double",
    "Float",
    "Boolean",
    "Object",
    "List",
    "Map",
    "Set",
    "Optional",
    "Stream",
    "Arrays",
    "Collections",
];

/// Whether `node` is a reference-bearing identifier for Java usages: the
/// `identifier` / `type_identifier` kinds are the only references (Java's
/// grammar uses no `simple_identifier`).
fn java_is_reference_node(kind: &str) -> bool {
    matches!(kind, "identifier" | "type_identifier")
}

/// Whether `node` sits inside a call / import node within
/// [`JAVA_USAGE_MAX_PARENT_DEPTH`] ancestors.
fn java_ref_inside(node: Node<'_>, kinds: &[&str]) -> bool {
    let mut cur = node.parent();
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= JAVA_USAGE_MAX_PARENT_DEPTH {
            break;
        }
        if kinds.contains(&n.kind()) {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// Whether `node` is the `name:` field of its own parent — i.e. it names a
/// definition rather than referencing one.
fn java_is_definition_name(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    parent
        .child_by_field_name("name")
        .map(|name_field| {
            name_field.start_byte() == node.start_byte() && name_field.end_byte() == node.end_byte()
        })
        .unwrap_or(false)
}

/// Name of the nearest enclosing Java type declaration
/// (`class_declaration` / `interface_declaration` / `enum_declaration`) — the
/// spec pass's `owner_kinds` for Java. Used to reconstruct an owned method's
/// `{file}::{OwnerType}::{name}` qname.
fn java_enclosing_type_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut cur = node.parent();
    while let Some(n) = cur {
        if matches!(
            n.kind(),
            "class_declaration" | "interface_declaration" | "enum_declaration"
        ) {
            return n.child_by_field_name("name").map(|c| node_text(source, c));
        }
        cur = n.parent();
    }
    None
}

/// The qname of the definition enclosing `node`, used as a USAGE
/// edge's source: the nearest enclosing
/// method / constructor (qualified by its owning type), else the file-level
/// synthetic Module node (`<file>::__file__`). The method /
/// constructor qname matches the spec definition pass
/// (`{file}::{OwnerType}::{name}`) so the endpoint resolves to a real node.
fn java_usage_source_qname(source: &[u8], node: Node<'_>, file_path: &str) -> String {
    let mut cur = node.parent();
    while let Some(n) = cur {
        if matches!(n.kind(), "method_declaration" | "constructor_declaration") {
            if let Some(name) = n.child_by_field_name("name").map(|c| node_text(source, c)) {
                // Owner type name = nearest enclosing type declaration, matching
                // the spec pass's `{file}::{OwnerType}::{name}` method qname.
                let owner = java_enclosing_type_name(source, n);
                return match owner {
                    Some(t) => format!("{file_path}::{t}::{name}"),
                    // A method with no enclosing type is not valid Java, but be
                    // defensive and fall back to the free method qname the spec
                    // pass would emit.
                    None => format!("{file_path}::Method::{name}"),
                };
            }
        }
        cur = n.parent();
    }
    format!("{file_path}::__file__")
}

/// Supplementary Java usage pass: emits `USAGE` edges for identifier /
/// type-identifier references that are NOT calls (already CALLS edges), NOT
/// inside imports, NOT definition names, and NOT keywords. The indexer's
/// shared USAGE arm resolves the `ref_name` to a unique registered symbol and
/// drops it otherwise, and dedups per `(source, target, USAGE)`, so multiple
/// references to the same symbol from the same enclosing definition collapse to
/// one edge.
fn java_usage_pass(
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let tree = crate::parse(Language::Java, source)?;
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        // Push every child (named + anonymous) so the walk visits every node.
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                stack.push(child);
            }
        }
        if !java_is_reference_node(node.kind()) {
            continue;
        }
        if java_ref_inside(node, JAVA_CALL_NODE_KINDS)
            || java_ref_inside(node, JAVA_IMPORT_NODE_KINDS)
        {
            continue;
        }
        if java_is_definition_name(node) {
            continue;
        }
        let name = node_text(source, node);
        if name.is_empty() || JAVA_USAGE_KEYWORDS.contains(&name) {
            continue;
        }
        let source_qname = java_usage_source_qname(source, node, file_path);
        result.edges.push(ExtractedEdge {
            edge_type: "USAGE".into(),
            source_qualified_name: source_qname,
            // No real direct-target qname exists; the indexer's USAGE arm
            // resolves `ref_name` by name against any registered symbol.
            target_qualified_name: format!("{file_path}::__ref__::{name}"),
            file_path: file_path.to_string(),
            line: node.start_position().row as u32 + 1,
            properties: serde_json::json!({ "ref_name": name }),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// C / C++ extraction
// ---------------------------------------------------------------------------
//
// Handles the passes at the level supported by both grammars, reusing the same
// `ExtractedNode` / `ExtractedEdge` conventions and name-based resolution keys
// (`callee_name`, `imported_name`) so the indexer's existing two-phase resolver
// links C / C++ edges cross-file with NO indexer change. The two languages share
// one implementation: C++ is a superset, and the extra node kinds
// (`class_specifier`, `namespace_definition`, `qualified_identifier` callees,
// `using_declaration` imports) are handled by node kind so a C file simply never
// produces them.
//
//   * DEFINITIONS — `function_definition` → `Function` (or `Method` when owned
//     by a class / out-of-line `Class::method`); `struct` / `union` / `enum`
//     specifiers → `Struct` / `Union` / `Enum`; `typedef` → `Type`; (C++)
//     `class_specifier` → `Class`, `namespace_definition` → `Namespace`.
//   * CALLS — final callee identifier (bare, `obj.fn()` / `ptr->fn()`, or
//     `ns::fn()`) → `CALLS` edge with the `callee_name` property, sourced from
//     the enclosing function/method qname.
//   * IMPORTS — `#include <x>` / `"x"` → an `IMPORTS` edge whose
//     `imported_name` is the header basename; (C++) `using` declarations →
//     `IMPORTS` whose `imported_name` is the used name / namespace.
//   * docstrings — a leading block (`/* */`) or run of line (`//`) comments
//     immediately preceding the definition becomes the node's `doc` (one-line
//     summary) and `doc_full`.

/// Run all C / C++ extraction passes. Shared by C and C++; `language` selects
/// the grammar / query set. The node kinds and conventions are identical except
/// for the C++-only definition / call / import forms, which are dispatched by
/// node kind.
fn extract_c_cpp(
    language: Language,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let (queries, spec) = match language {
        Language::C => (
            crate::query::cached_query_set(&Language::C)
                .map_err(|e| greppy_core::Error::Parse(format!("compile c queries: {e}")))?,
            &crate::spec::C,
        ),
        Language::Cpp => (
            crate::query::cached_query_set(&Language::Cpp)
                .map_err(|e| greppy_core::Error::Parse(format!("compile cpp queries: {e}")))?,
            &crate::spec::CPP,
        ),
        other => {
            return Err(greppy_core::Error::Parse(format!(
                "extract_c_cpp called with non-C/C++ language: {}",
                other.name()
            )))
        }
    };
    let mut result = crate::spec::spec_extract(language, spec, queries, source, file_path)?;

    // The uniform spec path emits the type / function / method nodes plus CALLS
    // / IMPORTS edges. A bespoke pass layered on top normalises the node + edge
    // set:
    //
    //   * every `struct` / `union` / `class` specifier is labelled "Class"
    //     (a single label for every non-enum / non-typedef type); the spec path
    //     stamps "Struct" / "Union", so we
    //     relabel them here (also rewriting the `::{label}::` qname segment);
    //   * `type_definition` (typedef) nodes carry NO label — a typedef whose
    //     bare name equals its struct/enum (`typedef struct Vec Vec;`) would
    //     collide, and no standalone `Type` nodes are emitted for the remainder,
    //     so we drop every spec-emitted "Type" node;
    //   * `namespace_definition` nodes are NOT graph nodes here (they fold into
    //     the module spine), so we drop every "Namespace" node;
    //   * struct / class / union body `field_declaration`s yield "Field" nodes
    //     (qname `{class}::{name}`);
    //   * enum body `enumerator`s yield "Variable" nodes (qname `{enum}::{name}`);
    //   * `#define` / function-like `#define` yield "Macro" nodes
    //     (incl. header-guard `#define X_H`).
    c_cpp_relabel_and_prune(&mut result);
    c_cpp_member_pass(language, source, file_path, &mut result)?;

    // USAGE — a per-language reference pass: every `identifier` /
    // `type_identifier` that is NOT inside a call, NOT inside an import, NOT a
    // definition name, and NOT a language keyword becomes a `USAGE` edge from
    // its enclosing function (or the `__file__` node) keyed on `ref_name`. The
    // shared indexer resolves `ref_name` to any registered symbol and drops it
    // unless unique.
    c_cpp_usage_pass(source, file_path, &mut result)?;

    Ok(result)
}

/// Drop the spec path's typedef `Type` and `Namespace` nodes (neither is a
/// graph node here), then relabel `Struct` / `Union` → `Class`
/// (rewriting the `::{label}::` qname segment). Every C / C++ `struct` /
/// `union` / `class` specifier folds into "Class", the resolvable type target
/// the shared indexer expects.
fn c_cpp_relabel_and_prune(result: &mut ExtractionResult) {
    result
        .nodes
        .retain(|n| n.label != "Type" && n.label != "Namespace");
    for node in &mut result.nodes {
        if node.label == "Struct" || node.label == "Union" {
            let old = format!("::{}::", node.label);
            if let Some(pos) = node.qualified_name.find(&old) {
                node.qualified_name
                    .replace_range(pos..pos + old.len(), "::Class::");
            }
            node.label = "Class".into();
        }
    }
}

/// The C / C++ type-declaration node kinds that own members, mapped to the label
/// their (relabelled) type node carries. `struct` / `union` / `class` fold into
/// "Class"; the label reconstructs the owning type
/// node's qname (`{file}::{label}::{name}`) for member ownership.
fn c_cpp_type_label(kind: &str) -> Option<&'static str> {
    match kind {
        "struct_specifier" | "union_specifier" | "class_specifier" => Some("Class"),
        "enum_specifier" => Some("Enum"),
        _ => None,
    }
}

/// Supplementary C / C++ member + macro pass: appends `Field` / `Variable`
/// member nodes and `Macro` nodes to a spec-extracted result. Emits a "Field"
/// per struct/class/union field, a "Variable" per enum member, and a "Macro"
/// per `#define`.
fn c_cpp_member_pass(
    language: Language,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let tree = crate::parse(language, source)?;
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        match node.kind() {
            // `#define NAME ...` / `#define FN(x) ...` → a Macro node. The macro
            // body is a `preproc_arg`, so there is nothing to descend into.
            "preproc_def" | "preproc_function_def" => {
                c_cpp_macro_def(source, file_path, node, result);
            }
            k => {
                if let Some(label) = c_cpp_type_label(k) {
                    c_cpp_type_members(source, file_path, node, label, result);
                }
            }
        }
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                stack.push(child);
            }
        }
    }
    Ok(())
}

/// Emit a `Macro` node for a `#define`. The qname is `{file}::Macro::{name}`
/// so a macro never collides with a same-named function / type.
fn c_cpp_macro_def(source: &[u8], file_path: &str, node: Node<'_>, result: &mut ExtractionResult) {
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Macro".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Macro::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// The `field_declaration_list` / `enumerator_list` body of a C / C++ type
/// specifier (the `body` field). The sole container whose DIRECT
/// children are the type's own members, so a nested type is attributed to the
/// correct owner by the outer walk rather than double-counted.
fn c_cpp_type_body<'a>(type_node: Node<'a>) -> Option<Node<'a>> {
    type_node.child_by_field_name("body")
}

/// Emit the `Field` / `Variable` member nodes for one C / C++ type specifier
/// `type_node` (kind → `label`). Only the type's OWN body is scanned; nested
/// types are reached by the outer walk in [`c_cpp_member_pass`].
fn c_cpp_type_members(
    source: &[u8],
    file_path: &str,
    type_node: Node<'_>,
    label: &str,
    result: &mut ExtractionResult,
) {
    let Some(type_name) = type_node
        .child_by_field_name("name")
        .map(|n| node_text(source, n))
    else {
        return;
    };
    if type_name.is_empty() {
        return;
    }
    // The (relabelled) type node's qname is `{file}::{label}::{name}`; a member
    // hangs off it as `{file}::{label}::{type}::{member}`.
    let owner_prefix = format!("{file_path}::{label}::{type_name}");

    let Some(body) = c_cpp_type_body(type_node) else {
        return;
    };

    for i in 0..body.named_child_count() {
        let Some(child) = body.named_child(i) else {
            continue;
        };
        match child.kind() {
            // Struct / class / union body field → a Field node
            // (qname `{class}.{name}`). Function-pointer fields are skipped.
            "field_declaration" => {
                if label != "Class" {
                    continue;
                }
                if c_cpp_is_func_ptr_field(child) {
                    continue;
                }
                let Some(name_node) = c_cpp_field_name_node(child) else {
                    continue;
                };
                let fname = node_text(source, name_node);
                if fname.is_empty() {
                    continue;
                }
                let mut props = serde_json::Map::new();
                if let Some(ty) = child.child_by_field_name("type") {
                    props.insert(
                        "return_type".into(),
                        serde_json::Value::String(node_text(source, ty).to_string()),
                    );
                }
                result.nodes.push(ExtractedNode {
                    label: "Field".into(),
                    name: fname.to_string(),
                    qualified_name: format!("{owner_prefix}::{fname}"),
                    file_path: file_path.to_string(),
                    start_line: child.start_position().row as u32 + 1,
                    end_line: child.end_position().row as u32 + 1,
                    properties: serde_json::Value::Object(props),
                });
            }
            // Enum body member → a Variable node (qname `{enum}.{name}`).
            "enumerator" => {
                if label != "Enum" {
                    continue;
                }
                let Some(mname) = child
                    .child_by_field_name("name")
                    .or_else(|| first_child_of_kind_c_cpp(child, "identifier"))
                    .map(|n| node_text(source, n))
                else {
                    continue;
                };
                if mname.is_empty() {
                    continue;
                }
                result.nodes.push(ExtractedNode {
                    label: "Variable".into(),
                    name: mname.to_string(),
                    qualified_name: format!("{owner_prefix}::{mname}"),
                    file_path: file_path.to_string(),
                    start_line: child.start_position().row as u32 + 1,
                    end_line: child.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            _ => {}
        }
    }
}

/// The identifier node naming a C / C++ `field_declaration`: the `declarator`
/// field, unwrapping a single `pointer_declarator` / `array_declarator` layer
/// to its inner declarator. Returns `None` when the field has no plain
/// declarator (e.g. an anonymous struct member).
fn c_cpp_field_name_node(field: Node<'_>) -> Option<Node<'_>> {
    let name_node = field
        .child_by_field_name("declarator")
        .or_else(|| field.child_by_field_name("name"))?;
    match name_node.kind() {
        "pointer_declarator" | "array_declarator" => name_node.child_by_field_name("declarator"),
        _ => Some(name_node),
    }
}

/// Whether a `field_declaration`'s declarator chain is a function pointer
/// (`void (*fn)(int)`), which is skipped.
fn c_cpp_is_func_ptr_field(field: Node<'_>) -> bool {
    let mut decl = field.child_by_field_name("declarator");
    let mut depth = 0;
    while let Some(cur) = decl {
        if depth >= 8 {
            break;
        }
        if cur.kind() == "function_declarator" {
            return true;
        }
        decl = cur
            .child_by_field_name("declarator")
            .or_else(|| (0..cur.named_child_count()).find_map(|k| cur.named_child(k)));
        depth += 1;
    }
    false
}

/// First named child of `node` whose kind is `kind`.
fn first_child_of_kind_c_cpp<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    (0..node.named_child_count())
        .filter_map(|i| node.named_child(i))
        .find(|c| c.kind() == kind)
}

/// Ancestor node kinds that mark a C / C++ reference as being INSIDE a call.
/// C uses only `call_expression`;
/// C++ adds the operator / new / delete / index forms whose operands are
/// call-context, not standalone references.
const C_CALL_KINDS: &[&str] = &["call_expression"];
const CPP_CALL_KINDS: &[&str] = &[
    "call_expression",
    "field_expression",
    "subscript_expression",
    "new_expression",
    "delete_expression",
    "binary_expression",
    "unary_expression",
    "update_expression",
];

/// Ancestor node kinds that mark a reference as being INSIDE an import.
/// C uses only `preproc_include`; C++ adds `template_function` and
/// `declaration` (any identifier inside a plain declaration is a declared name,
/// not a reference).
const C_IMPORT_KINDS: &[&str] = &["preproc_include"];
const CPP_IMPORT_KINDS: &[&str] = &["preproc_include", "template_function", "declaration"];

/// C / C++ keyword names a bare reference must not be.
const C_CPP_KEYWORDS: &[&str] = &[
    "true",
    "false",
    "null",
    "nil",
    "None",
    "undefined",
    "void",
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "return",
    "throw",
    "try",
    "catch",
    "finally",
    "class",
    "struct",
    "enum",
    "interface",
    "trait",
    "impl",
    "import",
    "export",
    "package",
    "module",
    "use",
    "require",
    "include",
    "new",
    "delete",
    "this",
    "self",
    "super",
    "public",
    "private",
    "protected",
    "static",
    "const",
    "var",
    "let",
    "function",
    "def",
    "fn",
    "func",
    "fun",
    "proc",
    "sub",
    "method",
    "async",
    "await",
    "yield",
];

/// USAGE pass for a C / C++ file. Visits every node;
/// each `identifier` / `type_identifier` that survives the four skip guards
/// (inside-call, inside-import, definition-name, keyword) becomes a `USAGE`
/// edge from its enclosing function keyed on `ref_name`. The shared indexer
/// resolves `ref_name` to any registered symbol and keeps it only if unique.
fn c_cpp_usage_pass(
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    // Every C-family extension EXCEPT `.c` is treated as C++
    // (`.h` / `.hpp` / `.cc` / … → C++). Routing `.h` to `Language::C` would
    // otherwise cause the usage set to diverge two ways: (1) the narrower
    // C-dialect import-suppression (only `preproc_include`, so `declaration`
    // prototypes leak) and (2) the C-dialect grammar parses a bare
    // `typedef enum E E;` as `type_definition` + a standalone `type_identifier`
    // alias that the C++ grammar folds into the enum. Drive BOTH the grammar
    // and the skip-sets off
    // the extension: only a `.c` file is treated as plain C here.
    let is_plain_c = file_path.rsplit('.').next() == Some("c");
    let (usage_language, call_kinds, import_kinds): (Language, &[&str], &[&str]) = if is_plain_c {
        (Language::C, C_CALL_KINDS, C_IMPORT_KINDS)
    } else {
        (Language::Cpp, CPP_CALL_KINDS, CPP_IMPORT_KINDS)
    };
    let tree = crate::parse(usage_language, source)?;
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        c_cpp_try_emit_usage(node, source, file_path, call_kinds, import_kinds, result);
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                stack.push(child);
            }
        }
    }
    Ok(())
}

/// Try to emit one C / C++ `USAGE` edge for `node`.
fn c_cpp_try_emit_usage(
    node: Node<'_>,
    source: &[u8],
    file_path: &str,
    call_kinds: &[&str],
    import_kinds: &[&str],
    result: &mut ExtractionResult,
) {
    // C / C++ references are `identifier` / `type_identifier`.
    if !matches!(node.kind(), "identifier" | "type_identifier") {
        return;
    }
    let name = node_text(source, node);
    if name.is_empty() || C_CPP_KEYWORDS.contains(&name) {
        return;
    }
    // Type positions have to be classified before the generic definition-name
    // guard: `::Widget` is a `qualified_identifier` whose `name:` child is the
    // `type_identifier`, but that field is a reference, not a declaration.
    if c_cpp_is_type_reference(node) && !c_cpp_ancestor_in(node, import_kinds) {
        result.edges.push(ExtractedEdge {
            edge_type: "TYPE_REF".into(),
            source_qualified_name: c_cpp_reference_source_qname(source, node, file_path),
            target_qualified_name: format!("{file_path}::Class::{name}"),
            file_path: file_path.to_string(),
            line: node.start_position().row as u32 + 1,
            properties: serde_json::json!({ "type_name": name }),
        });
        return;
    }
    // Skip a reference sitting inside a call or an import (checks up to 10
    // ancestors for each set).
    if c_cpp_ancestor_in(node, call_kinds) || c_cpp_ancestor_in(node, import_kinds) {
        return;
    }
    // Skip a node that IS the `name:` field of its own parent (a definition
    // name, not a reference).
    if c_cpp_is_definition_name(node) {
        return;
    }
    // Every C / C++ usage is attributed to the per-file MODULE node, not the
    // enclosing function. Keying the source on `{file}::__file__` dedups the
    // file's usages by (module, resolved-symbol) over the module source id.
    result.edges.push(ExtractedEdge {
        edge_type: "USAGE".into(),
        source_qualified_name: format!("{file_path}::__file__"),
        // No real target qname exists; the indexer resolves `ref_name` to any
        // registered symbol and drops it unless unique.
        target_qualified_name: format!("{file_path}::__ref__::{name}"),
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
        properties: serde_json::json!({ "ref_name": name }),
    });
}

/// Whether `node` is contained by a C/C++ syntax node's `type:` field. This
/// catches parameter, return, field, and local declaration types while excluding
/// the `name:` of a type declaration.
fn c_cpp_is_type_reference(node: Node<'_>) -> bool {
    let mut current = node.parent();
    let mut depth = 0;
    while let Some(parent) = current {
        if depth >= 10 {
            break;
        }
        if parent.child_by_field_name("type").is_some_and(|type_node| {
            type_node.start_byte() <= node.start_byte() && type_node.end_byte() >= node.end_byte()
        }) {
            return true;
        }
        if matches!(
            parent.kind(),
            "compound_statement" | "declaration_list" | "translation_unit"
        ) {
            break;
        }
        current = parent.parent();
        depth += 1;
    }
    false
}

/// Reconstruct the qname of the C/C++ function containing a reference. Free
/// functions use `{file}::Function::{name}`; class-owned and out-of-line methods
/// use `{file}::{Owner}::{name}`. File-scope type references use the Module node.
fn c_cpp_reference_source_qname(source: &[u8], node: Node<'_>, file_path: &str) -> String {
    let mut current = node.parent();
    while let Some(parent) = current {
        if parent.kind() == "function_definition" {
            if let Some((name, owner)) = c_cpp_function_name(source, parent) {
                return match owner {
                    Some(owner) => format!("{file_path}::{owner}::{name}"),
                    None => format!("{file_path}::Function::{name}"),
                };
            }
        }
        current = parent.parent();
    }
    format!("{file_path}::__file__")
}

fn c_cpp_function_name<'a>(
    source: &'a [u8],
    function: Node<'_>,
) -> Option<(&'a str, Option<&'a str>)> {
    let declarator = function.child_by_field_name("declarator")?;
    let (name, qualifier) = c_cpp_declarator_name(source, declarator)?;
    let owner = qualifier.or_else(|| c_cpp_enclosing_class_name(source, function));
    Some((name, owner))
}

fn c_cpp_declarator_name<'a>(
    source: &'a [u8],
    declarator: Node<'_>,
) -> Option<(&'a str, Option<&'a str>)> {
    match declarator.kind() {
        "identifier" | "field_identifier" => Some((node_text(source, declarator), None)),
        "qualified_identifier" => {
            let name = declarator.child_by_field_name("name")?;
            let owner = declarator
                .child_by_field_name("scope")
                .map(|scope| node_text(source, scope));
            let (leaf, _) = c_cpp_declarator_name(source, name)?;
            Some((leaf, owner))
        }
        _ => c_cpp_declarator_name(source, declarator.child_by_field_name("declarator")?),
    }
}

fn c_cpp_enclosing_class_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut current = node.parent();
    while let Some(parent) = current {
        if matches!(parent.kind(), "class_specifier" | "struct_specifier") {
            return parent
                .child_by_field_name("name")
                .map(|name| node_text(source, name));
        }
        current = parent.parent();
    }
    None
}

/// Whether any ancestor of `node` within 10 levels has a kind in `kinds`
/// (the inside-call / inside-import test, bounded at a parent depth of 10).
fn c_cpp_ancestor_in(node: Node<'_>, kinds: &[&str]) -> bool {
    let mut p = node.parent();
    let mut depth = 0;
    while let Some(cur) = p {
        if depth >= 10 {
            break;
        }
        if kinds.contains(&cur.kind()) {
            return true;
        }
        p = cur.parent();
        depth += 1;
    }
    false
}

/// Whether `node` is a defined name rather than a reference to one, plus the
/// enum-typedef-alias case.
///
/// A node equal to its parent's `name:` field is a definition name. In addition,
/// no usage is emitted for the alias of a bare `typedef enum E E;`, which the
/// parse exposes as the `type_definition`'s `declarator` (a standalone
/// `type_identifier`). A `typedef struct S S;` alias IS a real reference and is
/// kept, so this is scoped to `type:` being an `enum_specifier`.
fn c_cpp_is_definition_name(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    let byte_eq =
        |n: Node<'_>| n.start_byte() == node.start_byte() && n.end_byte() == node.end_byte();
    if parent.child_by_field_name("name").is_some_and(byte_eq) {
        return true;
    }
    parent.kind() == "type_definition"
        && parent
            .child_by_field_name("declarator")
            .is_some_and(byte_eq)
        && parent
            .child_by_field_name("type")
            .is_some_and(|t| t.kind() == "enum_specifier")
}

// ---------------------------------------------------------------------------
// PHP / Bash — onboarded purely via the data-driven spec path (Track A).
// Each is a `LangSpec` + three query sources; no bespoke extraction logic.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// C# extraction
// ---------------------------------------------------------------------------
//
// C# is handled the same way as Java: the uniform spec path emits the type
// nodes, the owned `Method` / constructor nodes, and the CALLS / IMPORTS edges;
// a bespoke pass on top adds the member definitions and method-ownership edges
// the template cannot express:
//
//   * `struct_declaration` / `record_declaration` are labelled "Class" — every
//     non-interface / non-enum / non-type-alias type declaration is a "Class"
//     (the spec path instead stamps "Struct" / "Record", so we relabel them
//     here);
//   * every class-body `field_declaration` yields BOTH a `Field` node
//     (qname `{type}::{name}`) AND a `Variable` node (qname `{file}::{name}`);
//   * every `property_declaration` yields a `Field` node only (property nodes
//     are not treated as variables);
//   * every `enum_member_declaration` yields a `Variable` node
//     (qname `{enum}::{name}`);
//   * every owned `method_declaration` / `constructor_declaration` yields a
//     `DEFINES_METHOD` edge from its enclosing type node to the method node.
//
// A final reference pass classifies user-defined type positions as `TYPE_REF`
// and value reads as `USES`. The indexer resolves both conservatively by name;
// their persisted graph label remains the unified `USAGE` contract.
fn extract_csharp(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::CSharp)
        .map_err(|e| greppy_core::Error::Parse(format!("compile csharp queries: {e}")))?;
    let mut result = crate::spec::spec_extract(
        Language::CSharp,
        &crate::spec::CSHARP,
        queries,
        source,
        file_path,
    )?;
    csharp_relabel_types_as_class(&mut result);
    csharp_member_pass(source, file_path, &mut result)?;
    csharp_reference_pass(source, file_path, &mut result)?;
    Ok(result)
}

/// Relabel the spec path's `Struct` / `Record` type nodes to `Class`, rewriting
/// both the node label and the `::{label}::` segment of its qname. Every C#
/// `struct_declaration` / `record_declaration` is labelled "Class", and `Class`
/// is the resolvable inheritance / import target the shared indexer expects
/// (`IMPORTABLE_LABELS`), so this keeps both the node count and cross-file
/// resolution consistent. Method / Field member qnames are unaffected: they are
/// keyed on the bare type NAME (`{file}::{TypeName}::{member}`), not on the type
/// node's label segment.
fn csharp_relabel_types_as_class(result: &mut ExtractionResult) {
    for node in &mut result.nodes {
        if node.label == "Struct" || node.label == "Record" {
            let old = format!("::{}::", node.label);
            let new = "::Class::".to_string();
            if let Some(pos) = node.qualified_name.find(&old) {
                node.qualified_name
                    .replace_range(pos..pos + old.len(), &new);
            }
            node.label = "Class".into();
        }
    }
}

/// C# type-declaration node kinds that own members, mapped to the label the
/// (relabelled) type node carries. `struct` / `record` are folded into "Class";
/// the label is used to reconstruct the owning type node's qname
/// (`{file}::{label}::{name}`) for `DEFINES_METHOD` edges.
fn csharp_type_label(kind: &str) -> Option<&'static str> {
    match kind {
        "class_declaration" | "struct_declaration" | "record_declaration" => Some("Class"),
        "interface_declaration" => Some("Interface"),
        "enum_declaration" => Some("Enum"),
        _ => None,
    }
}

/// Supplementary C# member pass: appends `Field` / `Variable` member nodes and
/// `DEFINES_METHOD` edges to a spec-extracted result. Emits "Field" / "Variable"
/// member nodes and a `DEFINES_METHOD` edge per owned method.
fn csharp_member_pass(
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let tree = crate::parse(Language::CSharp, source)?;
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if let Some(label) = csharp_type_label(node.kind()) {
            csharp_type_members(source, file_path, node, label, result);
        }
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                stack.push(child);
            }
        }
    }
    Ok(())
}

/// The `declaration_list` (class / struct / record / interface) or
/// `enum_member_declaration_list` (enum) body of a C# type declaration (the
/// `body` field) — the sole container whose *direct* children are the type's own
/// members, so nested types are attributed to the correct owner by the outer
/// walk rather than double-counted.
fn csharp_type_body<'a>(type_node: Node<'a>) -> Option<Node<'a>> {
    type_node.child_by_field_name("body")
}

/// Emit the member nodes + `DEFINES_METHOD` edges for one C# type declaration
/// `type_node` (kind → `label`). Only the type's OWN body is scanned; nested
/// types are reached by the outer walk in [`csharp_member_pass`].
fn csharp_type_members(
    source: &[u8],
    file_path: &str,
    type_node: Node<'_>,
    label: &str,
    result: &mut ExtractionResult,
) {
    let Some(type_name) = type_node
        .child_by_field_name("name")
        .map(|n| node_text(source, n))
    else {
        return;
    };
    // The spec path names the type node `{file}::{label}::{name}` and names an
    // owned member `{file}::{type}::{name}`. Reconstruct both so our edges /
    // nodes line up with the nodes the spec pass already emitted.
    let type_qname = format!("{file_path}::{label}::{type_name}");
    let member_owner_prefix = format!("{file_path}::{type_name}");

    let Some(body) = csharp_type_body(type_node) else {
        return;
    };

    for i in 0..body.named_child_count() {
        let Some(child) = body.named_child(i) else {
            continue;
        };
        match child.kind() {
            // `field_declaration` → one Field node AND one module-scoped
            // Variable node. Only the first `variable_declarator` is taken.
            "field_declaration" => {
                let Some((fname, fname_node)) = csharp_field_name(source, child) else {
                    continue;
                };
                if fname.is_empty() || fname == "_" {
                    continue;
                }
                let start = child.start_position().row as u32 + 1;
                let end = child.end_position().row as u32 + 1;
                let mut props = serde_json::Map::new();
                if let Some(ty) = csharp_field_type(child) {
                    props.insert(
                        "return_type".into(),
                        serde_json::Value::String(node_text(source, ty).to_string()),
                    );
                }
                // Field: owned by the enclosing type (qname
                // `{type_owner}::{name}`). A Field is only emitted when the
                // declaration carries a resolvable type; a `field_declaration`
                // always has a `variable_declaration.type`, so this holds.
                result.nodes.push(ExtractedNode {
                    label: "Field".into(),
                    name: fname.to_string(),
                    qualified_name: format!("{member_owner_prefix}::{fname}"),
                    file_path: file_path.to_string(),
                    start_line: start,
                    end_line: end,
                    properties: serde_json::Value::Object(props),
                });
                // Variable: a distinct module-scoped Variable for the same
                // field (qname `{file}::{name}`), keyed on the declarator so it
                // never collides with the Field.
                result.nodes.push(ExtractedNode {
                    label: "Variable".into(),
                    name: fname.to_string(),
                    qualified_name: format!("{file_path}::Variable::{fname}"),
                    file_path: file_path.to_string(),
                    start_line: fname_node.start_position().row as u32 + 1,
                    end_line: fname_node.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            // `property_declaration` → a Field node ONLY (a property is a
            // field, not a variable, so no Variable is pushed for it).
            "property_declaration" => {
                let Some(pname) = child
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n))
                else {
                    continue;
                };
                if pname.is_empty() || pname == "_" {
                    continue;
                }
                let mut props = serde_json::Map::new();
                if let Some(ty) = child.child_by_field_name("type") {
                    props.insert(
                        "return_type".into(),
                        serde_json::Value::String(node_text(source, ty).to_string()),
                    );
                }
                result.nodes.push(ExtractedNode {
                    label: "Field".into(),
                    name: pname.to_string(),
                    qualified_name: format!("{member_owner_prefix}::{pname}"),
                    file_path: file_path.to_string(),
                    start_line: child.start_position().row as u32 + 1,
                    end_line: child.end_position().row as u32 + 1,
                    properties: serde_json::Value::Object(props),
                });
            }
            // Enum members → Variable nodes (qname `{enum}::{name}`).
            "enum_member_declaration" => {
                let Some(mname) = child
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n))
                    .or_else(|| {
                        first_child_of_kind_csharp(child, "identifier")
                            .map(|n| node_text(source, n))
                    })
                else {
                    continue;
                };
                if mname.is_empty() {
                    continue;
                }
                result.nodes.push(ExtractedNode {
                    label: "Variable".into(),
                    name: mname.to_string(),
                    qualified_name: format!("{type_qname}::{mname}"),
                    file_path: file_path.to_string(),
                    start_line: child.start_position().row as u32 + 1,
                    end_line: child.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            // Owned method / constructor → a `DEFINES_METHOD` edge from the
            // enclosing type node to the method node. The method node itself is
            // already emitted by the spec definitions pass with qname
            // `{file}::{type}::{name}`; the indexer resolves this edge's two
            // endpoints by direct qname lookup (its default edge-type path).
            "method_declaration" | "constructor_declaration" => {
                let Some(mname) = child
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n))
                else {
                    continue;
                };
                if mname.is_empty() {
                    continue;
                }
                result.edges.push(ExtractedEdge {
                    edge_type: "DEFINES_METHOD".into(),
                    source_qualified_name: type_qname.clone(),
                    target_qualified_name: format!("{member_owner_prefix}::{mname}"),
                    file_path: file_path.to_string(),
                    line: child.start_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            _ => {}
        }
    }
}

/// The name (+ its identifier node) of a C# `field_declaration`'s first
/// `variable_declarator`, read via `field_declaration > variable_declaration >
/// variable_declarator(.name)`.
fn csharp_field_name<'a>(source: &'a [u8], field: Node<'a>) -> Option<(&'a str, Node<'a>)> {
    let decl = first_child_of_kind_csharp(field, "variable_declaration")?;
    let declarator = first_child_of_kind_csharp(decl, "variable_declarator")?;
    let name_node = declarator
        .child_by_field_name("name")
        .or_else(|| first_child_of_kind_csharp(declarator, "identifier"))?;
    Some((node_text(source, name_node), name_node))
}

/// The `type` node of a C# `field_declaration` (nested inside the
/// `variable_declaration`), used to stamp the Field's `return_type`.
fn csharp_field_type<'a>(field: Node<'a>) -> Option<Node<'a>> {
    let decl = first_child_of_kind_csharp(field, "variable_declaration")?;
    decl.child_by_field_name("type")
}

/// First named child of `node` whose kind is `kind`.
fn first_child_of_kind_csharp<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    (0..node.named_child_count())
        .filter_map(|i| node.named_child(i))
        .find(|c| c.kind() == kind)
}

/// Supplementary C# reference pass. Identifiers in a `type:` / `returns:`
/// subtree become `TYPE_REF`; other value reads become `USES`. Declaration
/// names, namespace/import segments, and invocation callees/arguments are
/// excluded because their dedicated passes own those syntax positions.
fn csharp_reference_pass(
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let tree = crate::parse(Language::CSharp, source)?;
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                stack.push(child);
            }
        }
        if node.kind() != "identifier"
            || csharp_is_declaration_name(node)
            || csharp_inside_any(node, &["using_directive"])
            || csharp_is_namespace_name(node)
        {
            continue;
        }
        let name = node_text(source, node);
        if name.is_empty() || csharp_is_reference_keyword(name) {
            continue;
        }
        let source_qname = csharp_reference_source_qname(source, node, file_path);
        if csharp_is_type_reference(node) {
            result.edges.push(ExtractedEdge {
                edge_type: "TYPE_REF".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::Class::{name}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({ "type_name": name }),
            });
        } else if !csharp_inside_any(node, &["invocation_expression"]) {
            result.edges.push(ExtractedEdge {
                edge_type: "USES".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{name}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({ "ref_name": name }),
            });
        }
    }
    Ok(())
}

/// Whether `node` names a declaration rather than referring to one. C# also
/// uses `name:` for member-access selectors, so this must be restricted to
/// declaration node kinds instead of applying the generic field-name check.
fn csharp_is_declaration_name(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    if !matches!(
        parent.kind(),
        "class_declaration"
            | "struct_declaration"
            | "interface_declaration"
            | "record_declaration"
            | "enum_declaration"
            | "enum_member_declaration"
            | "method_declaration"
            | "constructor_declaration"
            | "property_declaration"
            | "variable_declarator"
            | "parameter"
            | "type_parameter"
    ) {
        return false;
    }
    parent.child_by_field_name("name").is_some_and(|name| {
        name.start_byte() == node.start_byte() && name.end_byte() == node.end_byte()
    })
}

/// Whether `node` is contained by the `name:` field of a namespace declaration.
/// The namespace body is deliberately not suppressed.
fn csharp_is_namespace_name(node: Node<'_>) -> bool {
    let mut current = node.parent();
    while let Some(parent) = current {
        if parent.kind() == "namespace_declaration" {
            return parent.child_by_field_name("name").is_some_and(|name| {
                name.start_byte() <= node.start_byte() && name.end_byte() >= node.end_byte()
            });
        }
        current = parent.parent();
    }
    false
}

/// Whether an ancestor within the bounded syntax context has one of `kinds`.
fn csharp_inside_any(node: Node<'_>, kinds: &[&str]) -> bool {
    let mut current = node.parent();
    let mut depth = 0;
    while let Some(parent) = current {
        if depth >= 12 {
            break;
        }
        if kinds.contains(&parent.kind()) {
            return true;
        }
        current = parent.parent();
        depth += 1;
    }
    false
}

/// A C# type reference is any identifier contained by a parent's `type:` or
/// method `returns:` field. This covers parameters, locals, fields/properties,
/// casts, object creation, and method return types, including wrapped generic
/// and qualified type nodes.
fn csharp_is_type_reference(node: Node<'_>) -> bool {
    let mut current = node.parent();
    let mut depth = 0;
    while let Some(parent) = current {
        if depth >= 12 {
            break;
        }
        for field in ["type", "returns"] {
            if parent.child_by_field_name(field).is_some_and(|field_node| {
                field_node.start_byte() <= node.start_byte()
                    && field_node.end_byte() >= node.end_byte()
            }) {
                return true;
            }
        }
        if matches!(
            parent.kind(),
            "block" | "declaration_list" | "compilation_unit"
        ) {
            break;
        }
        current = parent.parent();
        depth += 1;
    }
    false
}

/// Reconstruct the spec-emitted qname of the nearest C# method/constructor;
/// references outside a callable attach to the per-file Module node.
fn csharp_reference_source_qname(source: &[u8], node: Node<'_>, file_path: &str) -> String {
    let mut current = node.parent();
    while let Some(parent) = current {
        if matches!(
            parent.kind(),
            "method_declaration" | "constructor_declaration"
        ) {
            if let Some(name_node) = parent.child_by_field_name("name") {
                let name = node_text(source, name_node);
                let mut owner = parent.parent();
                while let Some(candidate) = owner {
                    if csharp_type_label(candidate.kind()).is_some() {
                        if let Some(owner_name) = candidate.child_by_field_name("name") {
                            return format!(
                                "{file_path}::{}::{name}",
                                node_text(source, owner_name)
                            );
                        }
                    }
                    owner = candidate.parent();
                }
                return format!("{file_path}::Method::{name}");
            }
        }
        current = parent.parent();
    }
    format!("{file_path}::__file__")
}

fn csharp_is_reference_keyword(name: &str) -> bool {
    matches!(
        name,
        "this"
            | "base"
            | "true"
            | "false"
            | "null"
            | "var"
            | "void"
            | "object"
            | "string"
            | "bool"
            | "byte"
            | "sbyte"
            | "short"
            | "ushort"
            | "int"
            | "uint"
            | "long"
            | "ulong"
            | "float"
            | "double"
            | "decimal"
            | "char"
    )
}

fn extract_php(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Php)
        .map_err(|e| greppy_core::Error::Parse(format!("compile php queries: {e}")))?;
    let mut result =
        crate::spec::spec_extract(Language::Php, &crate::spec::PHP, queries, source, file_path)?;

    let tree = crate::parse(Language::Php, source)?;
    let root = tree.root_node();
    let file_module_qname = format!("{file_path}::__file__");

    // TRAIT-AS-CLASS RELABEL.
    //
    // The uniform spec stamps `trait_declaration` with label "Trait", but a PHP
    // trait is labelled "Class": only `interface_declaration` → "Interface" /
    // `enum_declaration` → "Enum"; a `trait_declaration` falls through to the
    // default "Class" (the `trait_item` / `trait_definition` kinds that DO map
    // to "Interface" are a different grammar and never appear in PHP). Relabel
    // the node and rewrite its qname's label segment so the `DEFINES_METHOD`
    // source below (computed from the same label mapping) resolves to it.
    for node in result.nodes.iter_mut() {
        if node.label == "Trait" {
            node.label = "Class".into();
            node.qualified_name = format!("{file_path}::Class::{}", node.name);
        }
    }

    // ENUM-METHOD FREE-FUNCTION PASS + DEFINES_METHOD PASS.
    //
    // Two behaviours, both driven off type-declaration bodies:
    //
    //  * DEFINES_METHOD — a `DEFINES_METHOD` edge from a type node to every
    //    method it owns. The uniform spec engine never emits it (only the
    //    bespoke Java pass does), so emit it for every PHP method owned by a
    //    class / interface / trait / enum.
    //
    //  * ENUM-METHOD Function — a type body is walked only when its container is
    //    a `declaration_list` (class/interface/trait). A PHP `enum_declaration`'s
    //    body is an `enum_declaration_list`, which is NOT recognised as a body
    //    container, so the enum's `method_declaration`s are re-visited as
    //    definitions and emit an ADDITIONAL file-scoped "Function" node for each
    //    (on top of the "Method" node already produced). Class/interface/trait
    //    bodies ARE recognised, so their methods are never re-walked and get no
    //    such duplicate. Reproduce that enum-only duplication.
    emit_php_type_members(source, root, file_path, &mut result);

    // TYPE-REFERENCE PASS.
    //
    // PHP's shared query set has no reference query. Walk the grammar's
    // `named_type` nodes so parameter and return annotations (including union,
    // intersection, and nullable wrappers) participate in the same cross-file
    // TYPE_REF resolution used by the other certified providers.
    emit_php_type_references(source, root, file_path, &file_module_qname, &mut result);

    // MODULE-SCOPE CALLS PASS.
    //
    // The shared `spec_calls` only emits a `CALLS` edge when the call has an
    // enclosing callable. For a call at module scope, fall back to the file
    // node, so a top-level `main();` still produces `<file>::__file__ → main`:
    // walk calls that are NOT inside any function/method definition and emit the
    // edge from the file Module node. The name-based resolver drops callees that
    // don't resolve (PHP builtins like `require`/`printf`).
    emit_php_module_scope_calls(source, root, file_path, &file_module_qname, &mut result);

    // IMPORTS COLLAPSE (per-namespace).
    //
    // A PHP `use` is one edge per statement, but the target resolves to the
    // *declaring file's* Module node via the namespace map and identical
    // (source-file, target) edges are deduped. Multiple `use App\Core\X;` in one
    // file all resolve to the same `App\Core` file → a single edge. The shared
    // `php_expand_use` instead
    // yields one edge per clause, each resolving to a distinct imported *symbol*,
    // produces one edge per imported symbol, so a file importing N classes from
    // one namespace produces N edges. Collapse to per-(source file, namespace)
    // granularity: keep only the first `IMPORTS` edge for each namespace prefix.
    // (The target is the imported symbol, not the declaring file's Module node —
    // a resolver choice outside this extractor.)
    collapse_php_imports(source, root, &mut result);

    Ok(result)
}

/// PHP type-declaration node kinds that own members, mapped to the label the
/// (post-relabel) spec path stamps on their node. `trait_declaration` maps to
/// "Class". Used to reconstruct the owning type node's qname
/// (`{file}::{label}::{name}`) for `DEFINES_METHOD` edges.
fn php_type_label(kind: &str) -> Option<&'static str> {
    match kind {
        "class_declaration" | "trait_declaration" => Some("Class"),
        "interface_declaration" => Some("Interface"),
        "enum_declaration" => Some("Enum"),
        _ => None,
    }
}

/// Emit a `TYPE_REF` edge for each PHP `named_type`. The grammar uses this node
/// only in declared type positions; primitive types have their own
/// `primitive_type` node and are therefore excluded structurally.
fn emit_php_type_references(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if node.kind() == "named_type" {
        if let Some(name_node) = php_named_type_name(node) {
            let name = node_text(source, name_node);
            if !name.is_empty() {
                let source_qname = php_reference_source_qname(source, node, file_path)
                    .unwrap_or_else(|| file_module_qname.to_string());
                result.edges.push(ExtractedEdge {
                    edge_type: "TYPE_REF".into(),
                    source_qualified_name: source_qname,
                    target_qualified_name: format!("{file_path}::Class::{name}"),
                    file_path: file_path.to_string(),
                    line: name_node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "type_name": name }),
                });
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        emit_php_type_references(source, child, file_path, file_module_qname, result);
    }
}

/// Return the final unqualified `name` inside a PHP `named_type`.
fn php_named_type_name(node: Node<'_>) -> Option<Node<'_>> {
    let mut stack = vec![node];
    let mut found = None;
    while let Some(current) = stack.pop() {
        if current.kind() == "name"
            && found.is_none_or(|previous: Node<'_>| current.start_byte() > previous.start_byte())
        {
            found = Some(current);
        }
        for index in 0..current.named_child_count() {
            if let Some(child) = current.named_child(index) {
                stack.push(child);
            }
        }
    }
    found
}

/// Reconstruct the spec-emitted qname of the nearest PHP function or method.
fn php_reference_source_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut current = node.parent();
    while let Some(parent) = current {
        match parent.kind() {
            "function_definition" => {
                let name = parent.child_by_field_name("name")?;
                return Some(format!(
                    "{file_path}::Function::{}",
                    node_text(source, name)
                ));
            }
            "method_declaration" => {
                let name = parent.child_by_field_name("name")?;
                let mut owner = parent.parent();
                while let Some(candidate) = owner {
                    if php_type_label(candidate.kind()).is_some() {
                        let owner_name = candidate.child_by_field_name("name")?;
                        return Some(format!(
                            "{file_path}::{}::{}",
                            node_text(source, owner_name),
                            node_text(source, name)
                        ));
                    }
                    owner = candidate.parent();
                }
            }
            _ => {}
        }
        current = parent.parent();
    }
    None
}

/// For every PHP type declaration under `node`, emit a `DEFINES_METHOD` edge for
/// each `method_declaration` it owns and — for `enum_declaration` bodies only —
/// an additional file-scoped `Function` node per method (the enum-body re-walk).
/// Only a type's OWN body is scanned; nested types are reached by the recursive
/// walk so members attribute to the correct owner.
fn emit_php_type_members(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    if let Some(label) = php_type_label(node.kind()) {
        if let Some(type_name) = node
            .child_by_field_name("name")
            .map(|n| node_text(source, n))
        {
            let type_qname = format!("{file_path}::{label}::{type_name}");
            let member_owner_prefix = format!("{file_path}::{type_name}");
            let is_enum = node.kind() == "enum_declaration";
            if let Some(body) = node.child_by_field_name("body") {
                let mut cursor = body.walk();
                for member in body.named_children(&mut cursor) {
                    if member.kind() != "method_declaration" {
                        continue;
                    }
                    let Some(mname) = member
                        .child_by_field_name("name")
                        .map(|n| node_text(source, n))
                    else {
                        continue;
                    };
                    if mname.is_empty() {
                        continue;
                    }
                    // DEFINES_METHOD: owner type node → the owned method node.
                    // Both endpoints resolve by direct qname lookup; the method
                    // node itself was already emitted by the spec definitions
                    // pass with qname `{file}::{type}::{name}`.
                    result.edges.push(ExtractedEdge {
                        edge_type: "DEFINES_METHOD".into(),
                        source_qualified_name: type_qname.clone(),
                        target_qualified_name: format!("{member_owner_prefix}::{mname}"),
                        file_path: file_path.to_string(),
                        line: member.start_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                    // Enum methods are ALSO re-walked as file-scoped free
                    // Functions (see `extract_php`). Add that duplicate node.
                    if is_enum {
                        result.nodes.push(ExtractedNode {
                            label: "Function".into(),
                            name: mname.to_string(),
                            qualified_name: format!("{file_path}::Function::{mname}"),
                            file_path: file_path.to_string(),
                            start_line: member.start_position().row as u32 + 1,
                            end_line: member.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                }
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        emit_php_type_members(source, child, file_path, result);
    }
}

/// The final callee name of a PHP call for the module-scope fallback: a bare
/// `function_call_expression` (`function: (name)`), a member call
/// (`->m()` / `?->m()`, `name: (name)`), or a static call (`C::m()`,
/// `name: (name)`). Returns `None` for other callee shapes (dynamic names,
/// variable callees) — matching the shared CALLS query's capture set.
fn php_call_callee_text<'a>(source: &'a [u8], call: Node<'_>) -> Option<&'a str> {
    let field = match call.kind() {
        "function_call_expression" => "function",
        "member_call_expression" | "nullsafe_member_call_expression" | "scoped_call_expression" => {
            "name"
        }
        _ => return None,
    };
    let callee = call.child_by_field_name(field)?;
    if callee.kind() == "name" {
        Some(node_text(source, callee))
    } else {
        None
    }
}

/// Emit `CALLS` edges for calls at *module scope* (not inside any
/// `function_definition` / `method_declaration`), sourced from the file's
/// `__file__` Module node — the file-node fallback the shared `spec_calls` omits
/// (it only emits with an enclosing callable). Recurses but does not descend
/// into function/method bodies — those calls already have an enclosing callable
/// and are handled by `spec_calls`.
fn emit_php_module_scope_calls(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if matches!(node.kind(), "function_definition" | "method_declaration") {
        return;
    }
    if matches!(
        node.kind(),
        "function_call_expression"
            | "member_call_expression"
            | "nullsafe_member_call_expression"
            | "scoped_call_expression"
    ) {
        if let Some(text) = php_call_callee_text(source, node) {
            if !text.is_empty() {
                result.edges.push(ExtractedEdge {
                    edge_type: "CALLS".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::Function::{text}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "callee_text": text,
                        "callee_name": text,
                    }),
                });
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        emit_php_module_scope_calls(source, child, file_path, file_module_qname, result);
    }
}

/// Collapse the shared PHP IMPORTS pass's per-*clause* edges to
/// per-(source file, namespace) granularity. Every `use` in a file resolves to
/// the declaring file's Module node and identical edges are deduped, so all
/// `use App\Core\X;` in one file collapse to a single edge. Retain only the
/// first `IMPORTS` edge per `(source file, namespace prefix)` pair; the
/// namespace prefix is the `use` path with its final (class/function) segment
/// dropped. Single-clause imports are unaffected.
fn collapse_php_imports(source: &[u8], root: Node<'_>, result: &mut ExtractionResult) {
    use std::collections::{HashMap, HashSet};

    // line (1-based) → namespace prefix, for every `use` clause in the file.
    let mut line_namespace: HashMap<u32, String> = HashMap::new();
    collect_php_use_namespaces(source, root, &mut line_namespace);

    let mut seen: HashSet<(String, String)> = HashSet::new();
    result.edges.retain(|edge| {
        if edge.edge_type != "IMPORTS" {
            return true;
        }
        // Derive the namespace from the edge's `path` property (the full dotted
        // `App\Core\X`), falling back through the AST line map. Dropping the last
        // `\`-segment yields the namespace `App\Core`.
        let namespace = edge
            .properties
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| match p.rsplit_once('\\') {
                Some((prefix, _last)) => prefix.to_string(),
                None => p.to_string(),
            })
            .or_else(|| line_namespace.get(&edge.line).cloned())
            .unwrap_or_default();
        seen.insert((edge.source_qualified_name.clone(), namespace))
    });
}

/// Record, for each `namespace_use_declaration` clause's start line, the
/// namespace prefix it imports (the `use` path minus its final segment). Used
/// by [`collapse_php_imports`] as a fallback when the edge `path` property is
/// absent.
fn collect_php_use_namespaces(
    source: &[u8],
    node: Node<'_>,
    out: &mut std::collections::HashMap<u32, String>,
) {
    if node.kind() == "namespace_use_declaration" {
        let line = node.start_position().row as u32 + 1;
        let text = node_text(source, node);
        // Strip the leading `use`/`use function`/`use const` keyword and take
        // the first path token; drop its final `\`-segment for the namespace.
        if let Some(path) = text
            .trim_start_matches("use")
            .trim_start()
            .trim_start_matches("function")
            .trim_start_matches("const")
            .trim_start()
            .split([';', ',', ' ', '{'])
            .next()
        {
            let ns = match path.trim().rsplit_once('\\') {
                Some((prefix, _last)) => prefix.to_string(),
                None => path.trim().to_string(),
            };
            out.entry(line).or_insert(ns);
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_php_use_namespaces(source, child, out);
    }
}

// ---------------------------------------------------------------------------
// Bash extraction
// ---------------------------------------------------------------------------
//
// The uniform spec (`crate::spec::BASH`) already produces the three passes it
// can express:
//
//   * DEFINITIONS — every `function_definition` becomes a `Function` node named
//     by its `name:` `word`. Bash has no class ownership, so nothing is a
//     `Method`.
//   * CALLS — every `command`'s `command_name` → a `CALLS` edge whose source is
//     the nearest enclosing `function_definition` and whose target is
//     `{file}::Function::{callee}`; `source` / `.` callees are dropped (owned by
//     the imports pass).
//   * IMPORTS — `source` / `.` commands become `IMPORTS` edges on the file
//     Module node.
//
// Bash has two more definition/edge kinds the uniform template cannot express,
// so `extract_bash` layers them on top of the spec output — the same way
// `extract_lua` layers module-level Variables and module-scope CALLS onto its
// own spec output:
//
//   * module-level `Variable`s — a `variable_assignment` is both a variable and
//     an assignment node, and only the file's DIRECT children are walked, so a
//     top-level `NAME=value` yields ONE `Variable` (the `name:` field text).
//     `declare -A NAME` is a `declaration_command`, not a `variable_assignment`,
//     so it is NOT a Variable; and `local`/loop assignments inside a
//     `function_definition` body are below module level, so they are excluded
//     too. Each Variable gets a `File → DEFINES → Variable` edge from the shared
//     structural pass.
//   * module-scope `CALLS` — a top-level `command` (not inside any
//     `function_definition`) has no enclosing callable, so the shared
//     `spec_calls` drops it. Fall back to the file node instead, so `main "$@"`
//     at the end of an entry-point script surfaces as a `CALLS` edge sourced
//     from the file's `__file__` node. We source it from that `{file}::__file__`
//     per-file Module node and target `{file}::Function::{callee}`, so it
//     resolves by name like any other call; `source` / `.` are skipped (owned by
//     imports) and the name-based resolver drops any callee with no `Function`
//     def (bash builtins like `echo` / `printf`).
//
// Bash emits NO `USAGE` edges: only `identifier` / `simple_identifier` /
// `type_identifier` are usage references, and tree-sitter-bash uses none of
// those for variable/command references (they are `variable_name` / `word`), so
// the usage walk finds nothing. The spec path likewise emits no usages, so bash
// needs no usage pass.
fn extract_bash(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Bash)
        .map_err(|e| greppy_core::Error::Parse(format!("compile bash queries: {e}")))?;
    let mut result = crate::spec::spec_extract(
        Language::Bash,
        &crate::spec::BASH,
        queries,
        source,
        file_path,
    )?;

    let tree = crate::parse(Language::Bash, source)?;
    let root = tree.root_node();
    let file_module_qname = format!("{file_path}::__file__");

    // (1) module-level `variable_assignment`s → `Variable` nodes.
    bash_emit_module_variables(source, root, file_path, &mut result);

    // (2) module-scope `CALLS` — commands not inside any `function_definition`,
    //     sourced from the file Module node (the file-node fallback).
    bash_emit_module_scope_calls(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// Emit one module-level `Variable` per top-level `variable_assignment`
/// (module-level only): `variable_assignment` is the variable node type, and
/// only the file root's DIRECT children are scanned, so assignments inside a
/// `function_definition` body are not module Variables. The name is the `name:`
/// field text (a `variable_name`; a `subscript` name like `ARR[k]=v` is not a
/// top-level definition here and is skipped). `declare -A` is a
/// `declaration_command`, not a `variable_assignment`, so it is excluded — the
/// gate keys purely on the `variable_assignment` kind.
fn bash_emit_module_variables(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        if child.kind() != "variable_assignment" {
            continue;
        }
        let Some(name_node) = child.child_by_field_name("name") else {
            continue;
        };
        // Read the `name:` field verbatim; a `subscript` LHS (`arr[k]=v`) is
        // not a bare top-level variable definition, so restrict to `variable_name`.
        if name_node.kind() != "variable_name" {
            continue;
        }
        let vname = node_text(source, name_node);
        if vname.is_empty() {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: vname.to_string(),
            qualified_name: format!("{file_path}::Variable::{vname}"),
            file_path: file_path.to_string(),
            start_line: child.start_position().row as u32 + 1,
            end_line: child.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// Emit `CALLS` edges for `command`s at *module scope* (not inside any
/// `function_definition`), sourced from the file's `__file__` Module node — the
/// file-node fallback the shared `spec_calls` omits (it only emits with an
/// enclosing callable). The callee is the `command_name`'s `word` text, matching
/// the `spec_calls` target scheme (`{file}::Function::{callee}`); `source` / `.`
/// are skipped (owned by the imports pass) and the name-based resolver drops any
/// callee that does not resolve to a `Function` (bash builtins like `echo` /
/// `printf`). Recurses over the tree but never descends into a
/// `function_definition` body — those calls already have an enclosing callable
/// and are handled by `spec_calls`.
fn bash_emit_module_scope_calls(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if node.kind() == "function_definition" {
        return;
    }
    if node.kind() == "command" {
        if let Some(name_node) = node.child_by_field_name("name") {
            // `command_name` wraps a `word` (bare command) — match the spec's
            // callee capture, which reads the inner `word`.
            if let Some(word) = find_child_of_kind(name_node, "word") {
                let text = node_text(source, word);
                if !text.is_empty() && text != "source" && text != "." {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Function::{text}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": text,
                            "callee_name": text,
                        }),
                    });
                }
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        bash_emit_module_scope_calls(source, child, file_path, file_module_qname, result);
    }
}

// ---------------------------------------------------------------------------
// Lua extraction
// ---------------------------------------------------------------------------
//
// The uniform spec (`crate::spec::LUA`) already produces the three passes it can
// express:
//
//   * DEFINITIONS — every `function_declaration` (`function f()`,
//     `local function f()`, `function M.f()`, `function M:f()`) becomes a
//     `Function` node named by the whole `name:` field text (`f`, `M.f`,
//     `M:f`). Lua has no class ownership, so nothing is a `Method`. The name is
//     the `name:` field read verbatim.
//   * CALLS — a `function_call`'s bare/dotted/method callee → a `CALLS` edge
//     targeting `{file}::Function::{callee}`, so a dotted call `M.f(...)`
//     resolves to the dotted `Function M.f` def (the callee is the `name:`
//     field text).
//   * IMPORTS — `require("path")` → an `IMPORTS` edge on the file Module node.
//
// Lua has three more definition/edge kinds the uniform template cannot express,
// so `extract_lua` layers them on top of the spec output:
//
//   * ANONYMOUS-FUNCTION Functions — `local f = function() … end` and
//     `M.f = function() … end` bind an anonymous `function_definition` to a
//     name resolved from the enclosing `assignment_statement`'s first variable,
//     so the lambda surfaces as a `Function` named `f` / `M.f`.
//   * module-level `Variable`s — a top-level `variable_declaration`
//     (`local x = …`) whose value is NOT a `function_definition` yields one
//     `Variable` (the first bound name). Bare `x = …` assignments are NOT
//     `variable_declaration`s, so they are not Variables.
//   * `USAGE` edges — every `identifier` reference that is not a definition
//     name, not inside a `function_call` / import, and not a keyword, keyed on
//     `ref_name` for the indexer's name-based resolver.
fn extract_lua(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Lua)
        .map_err(|e| greppy_core::Error::Parse(format!("compile lua queries: {e}")))?;
    let mut result =
        crate::spec::spec_extract(Language::Lua, &crate::spec::LUA, queries, source, file_path)?;

    let tree = crate::parse(Language::Lua, source)?;
    let root = tree.root_node();
    let file_module_qname = format!("{file_path}::__file__");

    // (1) Anonymous `function_definition`s bound to a name → `Function` nodes.
    lua_emit_anon_functions(source, root, file_path, &mut result);

    // (2) Module-level `variable_declaration`s (non-function values) →
    //     `Variable` nodes.
    lua_emit_module_variables(source, root, file_path, &mut result);

    // (3) module-scope `CALLS` — calls not inside any function, sourced from
    //     the file Module node (C `calls_find_source` file-node fallback).
    lua_emit_module_scope_calls(source, root, file_path, &file_module_qname, &mut result);

    // (4) `USAGE` edges for identifier references (C `pass_usages`).
    lua_emit_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The Lua `function_call` / import call node kind (both the call and import
/// containers are `function_call`). An identifier inside one of these is a
/// call/import argument, never a `USAGE`.
const LUA_CALL_KINDS: &[&str] = &["function_call"];

/// Emit `CALLS` edges for `function_call`s at *module scope* (not inside any
/// `function_declaration` / `function_definition`), sourced from the file's
/// `__file__` Module node — the file-node fallback the shared `spec_calls` omits
/// (it only emits with an enclosing callable). The callee is the `name:` field
/// text (bare or dotted), matching the `spec_calls` target scheme
/// (`{file}::Function::{callee}`); `require` is skipped (owned by the imports
/// pass) and the name-based resolver drops any callee that does not resolve (Lua
/// builtins like `print`). Recurses but does not descend into function bodies —
/// those calls already have an enclosing callable and are handled by
/// `spec_calls`.
fn lua_emit_module_scope_calls(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if matches!(node.kind(), "function_declaration" | "function_definition") {
        return;
    }
    if node.kind() == "function_call" {
        if let Some(name_node) = node.child_by_field_name("name") {
            if matches!(
                name_node.kind(),
                "identifier" | "dot_index_expression" | "method_index_expression"
            ) {
                let text = node_text(source, name_node);
                if !text.is_empty() && text != "require" {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Function::{text}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": text,
                            "callee_name": text,
                        }),
                    });
                }
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        lua_emit_module_scope_calls(source, child, file_path, file_module_qname, result);
    }
}

/// Emit a `Function` node for every anonymous `function_definition` bound to a
/// name. Walk from the `function_definition` up through its `expression_list` to
/// the enclosing `assignment_statement` and take the first variable of the
/// `variable_list`/`variables:` as the name (`M`, `f`, or a dotted `M.f`). The
/// `function_declaration` forms (`function f()`, `local function f()`) are
/// already handled by the spec definitions pass and use a different node kind,
/// so they are never double-counted here.
fn lua_emit_anon_functions(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    if node.kind() == "function_definition" {
        if let Some(name) = lua_anon_function_name(source, node) {
            if !name.is_empty() {
                result.nodes.push(ExtractedNode {
                    label: "Function".into(),
                    name: name.clone(),
                    qualified_name: format!("{file_path}::Function::{name}"),
                    file_path: file_path.to_string(),
                    start_line: node.start_position().row as u32 + 1,
                    end_line: node.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        lua_emit_anon_functions(source, child, file_path, result);
    }
}

/// Resolve the bound name of an anonymous `function_definition`: the
/// definition's value slot sits in an `expression_list`; its parent is the
/// `assignment_statement`; the first variable of that assignment's `variables:`
/// field (or `variable_list` child) is the name. Returns `None` for an un-bound
/// lambda (e.g. a callback passed directly as an argument).
fn lua_anon_function_name(source: &[u8], func_def: Node<'_>) -> Option<String> {
    let mut parent = func_def.parent()?;
    if parent.kind() == "expression_list" {
        parent = parent.parent()?;
    }
    if parent.kind() != "assignment_statement" {
        return None;
    }
    let vars = parent
        .child_by_field_name("variables")
        .or_else(|| first_child_of_kind_lua(parent, "variable_list"))?;
    let first = vars.named_child(0)?;
    let text = node_text(source, first);
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Emit one module-level `Variable` per top-level `variable_declaration` whose
/// value is not a `function_definition` (module-level only): a
/// `variable_declaration` wraps an `assignment_statement`; its
/// `expression_list`'s first value decides whether the binding is a `Variable`
/// (any non-lambda value) or a `Function` (a lambda, handled by
/// [`lua_emit_anon_functions`]). The variable name is the first entry of the
/// assignment's `variables:` / `variable_list`. Only the file's direct children
/// are scanned, so locals inside function bodies are not module Variables.
fn lua_emit_module_variables(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        if child.kind() != "variable_declaration" {
            continue;
        }
        let Some(assign) = first_child_of_kind_lua(child, "assignment_statement") else {
            continue;
        };
        // Skip a lambda-valued declaration — its name is a Function, not a
        // Variable.
        if let Some(expr_list) = first_child_of_kind_lua(assign, "expression_list") {
            if let Some(val) = expr_list.named_child(0) {
                if val.kind() == "function_definition" {
                    continue;
                }
            }
        }
        let Some(vars) = assign
            .child_by_field_name("variables")
            .or_else(|| first_child_of_kind_lua(assign, "variable_list"))
        else {
            continue;
        };
        let Some(first) = vars.named_child(0) else {
            continue;
        };
        let vname = node_text(source, first);
        if vname.is_empty() || vname == "_" {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: vname.to_string(),
            qualified_name: format!("{file_path}::Variable::{vname}"),
            file_path: file_path.to_string(),
            start_line: child.start_position().row as u32 + 1,
            end_line: child.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// First (named-or-anonymous) child of `node` whose kind is `kind`.
fn first_child_of_kind_lua<'t>(node: Node<'t>, kind: &str) -> Option<Node<'t>> {
    for i in 0..node.child_count() {
        let child = node.child(i)?;
        if child.kind() == kind {
            return Some(child);
        }
    }
    None
}

/// Recursively emit `USAGE` edges for Lua `identifier` references. Only the
/// common `identifier` is a reference node (Lua has no language-specific
/// reference kinds). A reference emits a usage unless it is a definition *name*
/// (the `name:` field of its parent), sits inside a `function_call` (the same
/// container is used for calls and imports, so this one check covers both the
/// CALLS and require suppressions), or is a keyword. The `ref_name` is resolved
/// project-wide by the indexer, so the target qname is a placeholder that never
/// resolves directly.
fn lua_emit_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if node.kind() == "identifier"
        && !is_inside_kind(node, LUA_CALL_KINDS)
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_lua_usage_keyword(text) {
            let source_qname = lua_enclosing_func_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({ "ref_name": text }),
            });
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        lua_emit_usages(source, child, file_path, file_module_qname, result);
    }
}

/// The nearest enclosing Lua callable qname for `node`'s USAGE source endpoint:
/// the closest `function_declaration` ancestor's `name:` field text
/// (`{file}::Function::{name}`; Lua has no ownership so the label is always
/// `Function`). Returns `None` at file scope (the caller substitutes the file
/// Module qname). An anonymous `function_definition` has no name, so the walk
/// continues past it to the nearest *named* declaration.
fn lua_enclosing_func_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "function_declaration" {
            if let Some(nm) = cur.child_by_field_name("name") {
                let name = node_text(source, nm);
                if !name.is_empty() {
                    return Some(format!("{file_path}::Function::{name}"));
                }
            }
        }
        p = cur.parent();
    }
    None
}

/// Lua keyword / literal filter. Lua uses the same generic keyword table as
/// Ruby. References whose text is one of these (notably `self`, `nil`,
/// `require`) never emit a usage.
fn is_lua_usage_keyword(name: &str) -> bool {
    is_ruby_usage_keyword(name)
}

fn extract_kotlin(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Kotlin)
        .map_err(|e| greppy_core::Error::Parse(format!("compile kotlin queries: {e}")))?;
    // Base pass (definitions for `class_declaration` / `object_declaration` +
    // `function_declaration`, calls, imports): the spec engine already emits the
    // "Class" node per `class_declaration` (Kotlin's grammar labels class /
    // interface / `enum class` all `class_declaration`, all "Class"), a "Method"
    // node owned by its enclosing type for every `function_declaration` inside a
    // type body, a free "Function" node for every top-level `fun`, the CALLS
    // pass and the IMPORTS pass. What the uniform template does NOT model is
    // added below: `object_declaration` is relabelled "Object" → "Class"; every
    // `type_alias` → a "Type" node; every body / module-level
    // `property_declaration` → a "Variable" node; the companion-object method is
    // removed; the DEFINES_METHOD edges; and the USAGE walk.
    let mut result = crate::spec::spec_extract(
        Language::Kotlin,
        &crate::spec::KOTLIN,
        queries,
        source,
        file_path,
    )?;

    // The spec `DefRule::ty("object_declaration", "Object")` labels a Kotlin
    // `object`/`companion object` "Object", but an `object_declaration` is a
    // class type (it does not match the interface/enum/type-alias arms), so it
    // should be "Class". Relabel — this also makes the node registrable in the
    // resolver's IMPORTABLE_LABELS / TYPE_LABELS / DEF_LABELS sets ("Object" is
    // in none of them), so an `import` of an object (and any type/usage
    // reference to it) resolves correctly.
    for node in &mut result.nodes {
        if node.label == "Object" {
            node.label = "Class".into();
            let prefix = format!("{file_path}::Object::");
            if let Some(rest) = node.qualified_name.strip_prefix(&prefix) {
                node.qualified_name = format!("{file_path}::Class::{rest}");
            }
        }
    }

    let tree = crate::parse(Language::Kotlin, source)?;
    let root = tree.root_node();

    kotlin_defs_pass(source, root, file_path, &mut result);

    let file_module_qname = format!("{file_path}::__file__");
    kotlin_emit_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The `name:` (`identifier`) of a Kotlin type / object declaration
/// (`class_declaration` / `object_declaration`), or `None` (an anonymous
/// `companion object` has no `name:` field).
fn kotlin_type_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    node.child_by_field_name("name")
        .map(|n| node_text(source, n))
}

/// Second definitions pass over the Kotlin tree, adding what the uniform spec
/// template does not model:
///
///   * every `type_alias` → a "Type" node (its name is a plain `identifier`
///     child, no `name:` field).
///   * every `property_declaration` that is a direct child of a type body
///     (`class_body`) OR a top-level (module-scope) child → a "Variable" node
///     (the class body's direct `property_declaration` children and the file
///     root's direct `property_declaration` children, both label "Variable").
///     Constructor-parameter `val`/`var`s (`class_parameter` inside
///     `primary_constructor`) are NOT `property_declaration` nodes, so they are
///     not Variables. A `companion object`'s properties are inside its own
///     `class_body`, which is never descended into (a name-less
///     `object_declaration`/`companion_object` is skipped before its variables
///     are extracted), so they are skipped here too.
///   * DEFINES_METHOD: each type → every method it owns, pointing at the
///     spec-emitted Method node.
///   * removal of the companion-object method the spec pass wrongly attributed
///     to the enclosing class: only a class body's DIRECT `function_declaration`
///     children are Methods, and a name-less `companion_object` is never
///     descended into, so a `fun` declared in a companion object is neither a
///     Method nor a Function.
fn kotlin_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    // Collect the qnames of companion-object-nested methods the spec pass
    // emitted so they can be removed (they are owned by the nearest enclosing
    // *named* type, since `companion_object` is not one of the spec's
    // `owner_kinds`). Also collect Variables and DEFINES_METHOD as we go.
    let mut drop_method_qnames: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "type_alias" => {
                // `typealias Slot = Map<...>` — the name is a plain `identifier`
                // child (no `name:` field in tree-sitter-kotlin-ng).
                if let Some(name) = kotlin_first_identifier(source, node) {
                    if !name.is_empty() {
                        result.nodes.push(ExtractedNode {
                            label: "Type".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::Type::{name}"),
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                }
                // A type alias has no members to descend into.
            }
            "class_declaration" | "object_declaration" => {
                let owner = kotlin_type_name(source, node);
                if let (Some(owner), Some(body)) = (owner, kotlin_class_body(node)) {
                    let mut bc = body.walk();
                    for member in body.named_children(&mut bc) {
                        match member.kind() {
                            "function_declaration" => {
                                if let Some(m) = kotlin_func_name(source, member) {
                                    if !m.is_empty() {
                                        result.edges.push(ExtractedEdge {
                                            edge_type: "DEFINES_METHOD".into(),
                                            source_qualified_name: format!(
                                                "{file_path}::Class::{owner}"
                                            ),
                                            target_qualified_name: format!(
                                                "{file_path}::{owner}::{m}"
                                            ),
                                            file_path: file_path.to_string(),
                                            line: member.start_position().row as u32 + 1,
                                            properties: serde_json::json!({}),
                                        });
                                    }
                                }
                            }
                            "property_declaration" => {
                                kotlin_emit_variable(source, member, file_path, result);
                            }
                            "companion_object" => {
                                // A (name-less) companion object is never
                                // descended into: it is skipped before its
                                // methods / variables are extracted. The spec
                                // query, however, captured its `fun`s and
                                // attributed them to THIS class (the nearest
                                // named owner). Mark those for removal.
                                let mut cc = member.walk();
                                for inner in member.named_children(&mut cc) {
                                    let cb = if inner.kind() == "class_body" {
                                        Some(inner)
                                    } else {
                                        None
                                    };
                                    if let Some(cb) = cb {
                                        let mut cbw = cb.walk();
                                        for cm in cb.named_children(&mut cbw) {
                                            if cm.kind() == "function_declaration" {
                                                if let Some(m) = kotlin_func_name(source, cm) {
                                                    drop_method_qnames.insert(format!(
                                                        "{file_path}::{owner}::{m}"
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // Descend only into nested *type* declarations so the body's
                // members (handled above) are not re-processed (only class-type
                // children of the body are pushed onto the defs stack).
                if let Some(body) = kotlin_class_body(node) {
                    kotlin_push_nested_types(body, &mut stack);
                }
            }
            "property_declaration" => {
                // A file-top-level property (a type-body property is handled in
                // the enclosing class arm and never re-descended into).
                kotlin_emit_variable(source, node, file_path, result);
            }
            "function_declaration" => {
                // A free function's body — do not descend (Kotlin function
                // bodies are not re-walked for further defs).
            }
            _ => {
                let mut c = node.walk();
                for child in node.named_children(&mut c) {
                    stack.push(child);
                }
            }
        }
    }

    if !drop_method_qnames.is_empty() {
        result
            .nodes
            .retain(|n| !(n.label == "Method" && drop_method_qnames.contains(&n.qualified_name)));
    }
}

/// The Kotlin `class_body` / `enum_class_body` of a type declaration, or `None`.
/// tree-sitter-kotlin-ng exposes the body as a `body:` field on
/// `class_declaration` in some shapes and as a plain child in others; fall back
/// to a child scan so both `class_body` and `enum_class_body` are found.
fn kotlin_class_body(node: Node<'_>) -> Option<Node<'_>> {
    if let Some(b) = node.child_by_field_name("body") {
        return Some(b);
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        if matches!(child.kind(), "class_body" | "enum_class_body") {
            return Some(child);
        }
    }
    None
}

/// The name of a Kotlin `function_declaration` — its `name:` field
/// (`identifier`), matching the spec engine's `Capture` name strategy so the
/// Method / DEFINES_METHOD qnames line up with the spec-emitted Method node.
fn kotlin_func_name<'a>(source: &'a [u8], func: Node<'_>) -> Option<&'a str> {
    func.child_by_field_name("name")
        .map(|n| node_text(source, n))
}

/// The first plain `identifier` child of `node` (used for `type_alias`, whose
/// name has no `name:` field). Skips the `typealias` keyword token (an unnamed
/// leaf), returning the first *named* `identifier`.
fn kotlin_first_identifier<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        if child.kind() == "identifier" {
            return Some(node_text(source, child));
        }
    }
    None
}

/// Push every nested `class_declaration` / `object_declaration` / `type_alias`
/// found directly under a type `body` onto the defs stack (so a nested type
/// gets its own Type / Variable / DEFINES_METHOD treatment) WITHOUT re-visiting
/// the body's method / property members. A `companion_object` is deliberately
/// NOT pushed: it has no name.
fn kotlin_push_nested_types<'a>(body: Node<'a>, stack: &mut Vec<Node<'a>>) {
    let mut inner = vec![body];
    while let Some(cur) = inner.pop() {
        let mut c = cur.walk();
        for child in cur.named_children(&mut c) {
            match child.kind() {
                "class_declaration" | "object_declaration" | "type_alias" => stack.push(child),
                // A method / property body can itself hold a locally-declared
                // type; keep scanning through non-type, non-companion nodes.
                "companion_object" => {}
                _ => inner.push(child),
            }
        }
    }
}

/// Emit a "Variable" node for a `property_declaration` (empty names and the `_`
/// placeholder are dropped). The name is the `variable_declaration`'s
/// `identifier`.
fn kotlin_emit_variable(
    source: &[u8],
    prop: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let Some(name) = kotlin_property_name(source, prop) else {
        return;
    };
    if name.is_empty() || name == "_" {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Variable".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Variable::{name}"),
        file_path: file_path.to_string(),
        start_line: prop.start_position().row as u32 + 1,
        end_line: prop.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// The Variable name of a Kotlin `property_declaration`: the `name:` field
/// first, else a direct `simple_identifier` / `identifier`, else the
/// `identifier` inside a nested `variable_declaration` (tree-sitter-kotlin-ng's
/// shape: `property_declaration > variable_declaration > identifier`).
fn kotlin_property_name<'a>(source: &'a [u8], prop: Node<'_>) -> Option<&'a str> {
    if let Some(n) = prop.child_by_field_name("name") {
        return Some(node_text(source, n));
    }
    let mut c = prop.walk();
    for child in prop.named_children(&mut c) {
        match child.kind() {
            "simple_identifier" | "identifier" => return Some(node_text(source, child)),
            "variable_declaration" => {
                let mut vc = child.walk();
                for id in child.named_children(&mut vc) {
                    if matches!(id.kind(), "simple_identifier" | "identifier") {
                        return Some(node_text(source, id));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// The nearest enclosing Kotlin callable qname for `node`: the closest
/// `function_declaration` ancestor, owned by its nearest enclosing named type
/// (`{file}::{Owner}::{name}`) or free (`{file}::Function::{name}`). Returns
/// `None` at file / type scope (the caller substitutes the file Module qname).
fn kotlin_enclosing_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "function_declaration" {
            let name = kotlin_func_name(source, cur)?;
            return Some(match kotlin_func_owner_name(source, cur) {
                Some(owner) => format!("{file_path}::{owner}::{name}"),
                None => format!("{file_path}::Function::{name}"),
            });
        }
        p = cur.parent();
    }
    None
}

/// The owning type *name* for a `function_declaration` (its nearest enclosing
/// `class_declaration` / `object_declaration`), or `None` when the func is free
/// (file scope). Mirrors the spec engine's `enclosing_owner_name` (owner_kinds =
/// class/object declarations) so the Method qname lines up with the spec nodes.
fn kotlin_func_owner_name<'a>(source: &'a [u8], func: Node<'_>) -> Option<&'a str> {
    let mut p = func.parent();
    while let Some(cur) = p {
        if matches!(cur.kind(), "class_declaration" | "object_declaration") {
            return kotlin_type_name(source, cur);
        }
        p = cur.parent();
    }
    None
}

/// Reference pass for Kotlin. Identifiers under a `user_type` are emitted as
/// `TYPE_REF`; other references are emitted as `USES`. Definition names, call
/// endpoints/arguments, imports, and keywords are excluded. Kotlin preserves
/// these logical labels in the persisted graph (rather than folding them to the
/// compatibility `USAGE` label) so provider completeness is externally visible.
fn kotlin_emit_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let kind = node.kind();
    if matches!(kind, "simple_identifier" | "identifier" | "type_identifier")
        && kotlin_is_usage_reference(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_kotlin_usage_keyword(text) && !is_kotlin_builtin_type(text) {
            // SOURCE ENDPOINT — the nearest enclosing callable's qname (its
            // method/function node), falling back to the per-file Module node at
            // class / file scope. tree-sitter-kotlin-ng always field-labels the
            // function name, so the enclosing-callable lookup is deterministic.
            let source_qname = kotlin_enclosing_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            let is_type_ref = kotlin_is_type_reference(node);
            result.edges.push(ExtractedEdge {
                edge_type: if is_type_ref { "TYPE_REF" } else { "USES" }.into(),
                source_qualified_name: source_qname,
                target_qualified_name: if is_type_ref {
                    format!("{file_path}::Class::{text}")
                } else {
                    format!("{file_path}::__ref__::{text}")
                },
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: if is_type_ref {
                    serde_json::json!({
                        "type_name": text,
                        "preserve_reference_kind": true,
                    })
                } else {
                    serde_json::json!({
                        "ref_name": text,
                        "preserve_reference_kind": true,
                    })
                },
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        kotlin_emit_usages(source, child, file_path, file_module_qname, result);
    }
}

/// Whether a Kotlin identifier is part of a user-defined type reference. The
/// grammar wraps parameter, return, property, and generic types in `user_type`;
/// builtin names are filtered before this classifier is called.
fn kotlin_is_type_reference(node: Node<'_>) -> bool {
    let mut current = node.parent();
    let mut depth = 0;
    while let Some(parent) = current {
        if depth >= 8 {
            break;
        }
        if parent.kind() == "user_type" {
            return true;
        }
        if matches!(
            parent.kind(),
            "block" | "function_body" | "source_file" | "import"
        ) {
            break;
        }
        current = parent.parent();
        depth += 1;
    }
    false
}

/// Decide whether an `identifier` reference is a genuine Kotlin reference, for
/// the tree-sitter-kotlin-ng tree shape.
///
/// A usage is emitted for every reference that is NOT a definition name, NOT
/// inside a call node (`call_expression` / `navigation_expression`), and NOT
/// inside an `import`. On tree-sitter-kotlin-ng the receiver is the first
/// `identifier` child of the `navigation_expression`, so a blanket "inside a
/// navigation_expression" test would wrongly drop it. This function encodes the
/// intent structurally:
///
///   * DROP the callee of a bare call (`call_expression`'s function-position
///     identifier) and the member name (the identifier after the `.` in a
///     `navigation_expression`) — those are the CALLS endpoints, not usages.
///   * DROP call arguments (`value_arguments`) — not counted.
///   * DROP import / package-header identifiers, definition names, and the
///     bound names of `variable_declaration` / `enum_entry` / parameter decls.
///   * KEEP a `navigation_expression` receiver, a `user_type` reference, and any
///     free identifier reference.
fn kotlin_is_usage_reference(node: Node<'_>) -> bool {
    // `is_definition_name` only suppresses a reference when its parent carries
    // the name on a `name:` field. tree-sitter-kotlin-ng field-labels the
    // class/object/function `name:`, so a blanket `is_definition_name` would
    // over-suppress the class/property name tokens (which are legitimate
    // usages). Only suppress a *function/method* name here (its parent is a
    // `function_declaration`).
    if let Some(parent) = node.parent() {
        if is_definition_name(node) && parent.kind() == "function_declaration" {
            return false;
        }
    }
    if kotlin_is_decl_name(node) {
        return false;
    }
    // Walk ancestors to classify context (bounded at a depth-10 scan).
    let mut cur = node;
    let mut child = node;
    let mut depth = 0;
    while let Some(parent) = cur.parent() {
        if depth >= 12 {
            break;
        }
        match parent.kind() {
            // Any identifier under an import directive or the file's package
            // header is not a usage. (A `qualified_identifier` also appears in
            // fully-qualified type references, so only suppress when it is
            // actually under an import / package header.)
            "import" | "package_header" | "qualified_identifier"
                if kotlin_under_import_or_package(parent) =>
            {
                return false;
            }
            // Call arguments and generic type-arguments are suppressed (they
            // sit inside the `call_expression` subtree, e.g. the `Record` in
            // `ArrayList<Record>()` or the `key` in `put(key)`).
            "value_arguments" | "type_arguments" => return false,
            // The function-position child of a bare call is the callee (a
            // `navigation_expression` callee is handled by the arm below when we
            // reach IT as the parent): `call_expression > identifier value_args`.
            "call_expression" if matches!(child.kind(), "identifier" | "simple_identifier") => {
                return false;
            }
            // `navigation_expression > receiver `.` member`: the RECEIVER (first
            // identifier) is a usage, the MEMBER (after the `.`) is the
            // call/property member and is suppressed.
            "navigation_expression" if !kotlin_is_navigation_receiver(parent, node) => {
                return false;
            }
            _ => {}
        }
        child = parent;
        cur = parent;
        depth += 1;
    }
    true
}

/// True when `id` is the receiver (first `identifier`/`simple_identifier` child)
/// of `nav`, a `navigation_expression`. The member name (after the `.`) returns
/// false.
fn kotlin_is_navigation_receiver(nav: Node<'_>, id: Node<'_>) -> bool {
    let mut c = nav.walk();
    for child in nav.named_children(&mut c) {
        if matches!(child.kind(), "identifier" | "simple_identifier") {
            return child == id;
        }
        // The receiver may itself be a nested navigation_expression / call;
        // in that case `id` is inside it, not the direct member, so it is a
        // usage (return true) unless it is the trailing member.
        if child.byte_range().contains(&id.start_byte())
            && child.kind() != "identifier"
            && child.kind() != "simple_identifier"
        {
            return true;
        }
    }
    false
}

/// True when any ancestor chain from `node` up is an `import` or `package_header`
/// (used to suppress qualified-identifier segments in those directives).
fn kotlin_under_import_or_package(node: Node<'_>) -> bool {
    let mut cur = Some(node);
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= 12 {
            break;
        }
        if matches!(n.kind(), "import" | "package_header") {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// True when `node` is the `identifier` that names the *declaration* part of a
/// `property_declaration` / `type_alias` / `class_parameter` / `parameter` /
/// `enum_entry` — the definition side. tree-sitter-kotlin-ng often nests the
/// bound name in a `variable_declaration` with no `name:` field, so the
/// field-based `is_definition_name` misses it; catch those here.
fn kotlin_is_decl_name(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    match parent.kind() {
        // `variable_declaration > identifier` is the bound var name (def side).
        "variable_declaration" => true,
        // `enum_entry > identifier` is an enum constant declaration name.
        "enum_entry" => true,
        // `type_alias > identifier(name) = user_type`: the leading identifier is
        // the alias NAME (a definition, already emitted as a "Type" node), not a
        // usage. C does not emit a usage for it either.
        "type_alias" => {
            let mut c = parent.walk();
            for ch in parent.named_children(&mut c) {
                if ch.kind() == "identifier" {
                    return ch == node;
                }
            }
            false
        }
        // A parameter / class_parameter's leading identifier is its name (a
        // declaration, not a reference) — `is_definition_name` catches the ones
        // carried on a `name:` field; the plain-identifier shape is caught here.
        // The *type* of the parameter is a separate `user_type` child.
        "parameter" | "class_parameter" => {
            // Only the first identifier child (the name) is a definition; a
            // later identifier is the type (handled by the builtin filter or
            // emitted as a genuine type usage).
            let mut c = parent.walk();
            for ch in parent.named_children(&mut c) {
                if ch.kind() == "identifier" {
                    return ch == node;
                }
            }
            false
        }
        _ => false,
    }
}

/// Kotlin keyword filter. A reference whose text is one of these never emits a
/// usage.
fn is_kotlin_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "as" | "break"
            | "class"
            | "continue"
            | "do"
            | "else"
            | "false"
            | "for"
            | "fun"
            | "if"
            | "in"
            | "interface"
            | "is"
            | "null"
            | "object"
            | "package"
            | "return"
            | "super"
            | "this"
            | "throw"
            | "true"
            | "try"
            | "typealias"
            | "typeof"
            | "val"
            | "var"
            | "when"
            | "while"
    )
}

/// Kotlin/JVM builtin type filter, consulted when classifying type references.
/// In tree-sitter-kotlin-ng a builtin type such as `Int` / `String` appears as a
/// plain `identifier` inside a `user_type`, so it would otherwise flood the
/// USAGE walk. Filtering the builtins here keeps the walk emitting only
/// user-defined references.
fn is_kotlin_builtin_type(name: &str) -> bool {
    matches!(
        name,
        "Int"
            | "Int8"
            | "Int16"
            | "Int32"
            | "Int64"
            | "UInt"
            | "UInt8"
            | "UInt16"
            | "UInt32"
            | "UInt64"
            | "Float"
            | "Double"
            | "String"
            | "Bool"
            | "Boolean"
            | "Byte"
            | "Short"
            | "Long"
            | "Char"
            | "Unit"
            | "Void"
            | "Any"
            | "Nothing"
            | "Dynamic"
            | "Number"
            | "List"
            | "MutableList"
            | "Map"
            | "MutableMap"
            | "Set"
            | "MutableSet"
            | "Array"
            | "ArrayList"
            | "HashMap"
            | "HashSet"
            | "Pair"
            | "Triple"
            | "Collection"
            | "Iterable"
            | "Sequence"
    )
}

// ===========================================================================
// Groovy — bespoke extraction pass.
// ===========================================================================
//
// The crates.io `tree-sitter-groovy` grammar the registry uses is Java-derived,
// so its tree shape differs from a Groovy-native grammar. The Groovy taxonomy
// this pass emits is:
//
//   * **Class** — one per `class`/`interface` declaration (both `class Foo` and
//     `interface Foo` are folded into "Class"; there is no Interface/Enum label
//     for Groovy). Enums emit no node. Qname `{file}::Class::{Name}`.
//   * **Method** — one per method that is a DIRECT member of a class body AND
//     has a body → Method `{Owner}.{name}`. Constructors (name == class) are NOT
//     extracted, and an `interface` body carries only signatures (no body) so it
//     yields NO methods. Qname `{file}::{Owner}::{name}`.
//   * **Function** — one per top-level (script) function definition
//     (`def f(){}` / `int f(){}`). Qname `{file}::Function::{name}`.
//   * **Variable** — one per class field declaration, MODULE-scoped (file-level
//     `{Module}.{field}`, not `{Class}.{field}`). Qname `{file}::Variable::{name}`.
//
// Edges, on top of the spec's CALLS / IMPORTS:
//   * **DEFINES_METHOD** — owner Class → each owned Method.
//   * **USAGE** — one usage per referenced identifier that resolves to a unique
//     def. Two things drive the count: (a) each method/function DEFINITION NAME
//     is a usage (see `groovy_emit_def_name_usages`), and (b) every body
//     identifier that is not inside a call/import and resolves uniquely by name
//     (a field ref, a param/local whose name collides with a unique def, a
//     field's TYPE identifier). The source is always the per-file Module.
//
// The generic spec path (Java grammar) mislabels `interface`→Interface and
// `enum`→Enum, counts constructors + interface signatures as Methods, and emits
// no Variable / DEFINES_METHOD / USAGE. `extract_groovy` walks the Java-grammar
// tree directly to emit the taxonomy above, and reuses the spec's CALLS
// (constructor self-CALLS are excluded as out of scope) and IMPORTS.

/// Java-grammar class-like declaration kinds. `interface_declaration` is folded
/// into "Class" (there is no distinct interface node kind for Groovy).
const GROOVY_CLASS_KINDS: [&str; 2] = ["class_declaration", "interface_declaration"];

fn extract_groovy(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    // Base: run the generic spec path only for its CALLS + IMPORTS passes; drop
    // its (mislabelled) node set and re-emit nodes from the tree walk below.
    let queries = d
        .compiled_queries()
        .map_err(|e| greppy_core::Error::Parse(format!("compile {} queries: {e}", d.name)))?;
    let mut base =
        crate::spec::spec_extract(Language::Registered(d), d.spec, queries, source, file_path)?;
    // Keep only CALLS / IMPORTS edges from the spec run; discard its nodes and
    // any other edge kinds (there are none, but be explicit).
    base.nodes.clear();
    base.edges
        .retain(|e| e.edge_type == "CALLS" || e.edge_type == "IMPORTS");

    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let file_module_qname = format!("{file_path}::__file__");

    groovy_defs_pass(source, root, file_path, &mut base);
    groovy_usages_pass(source, root, &file_module_qname, file_path, &mut base);

    Ok(base)
}

/// Emit the Class / Method / Function / Variable nodes and DEFINES_METHOD edges
/// for one Groovy file, following the taxonomy documented above.
fn groovy_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "class_declaration" | "interface_declaration" => {
                let Some(name) = node
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n).to_string())
                else {
                    // No name — descend so any nested defs are still reached.
                    let mut c = node.walk();
                    for child in node.named_children(&mut c) {
                        stack.push(child);
                    }
                    continue;
                };
                // The class/interface itself → a "Class" node.
                result.nodes.push(ExtractedNode {
                    label: "Class".into(),
                    name: name.clone(),
                    qualified_name: format!("{file_path}::Class::{name}"),
                    file_path: file_path.to_string(),
                    start_line: node.start_position().row as u32 + 1,
                    end_line: node.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
                // Members: an `interface_declaration` body has only bodyless
                // signatures (no Method, no field); a `class_declaration`
                // body contributes concrete Methods + field Variables.
                if node.kind() == "class_declaration" {
                    if let Some(body) = node.child_by_field_name("body") {
                        groovy_class_members(source, body, &name, file_path, result);
                    }
                }
                // Do NOT descend past the class into its bodies via the generic
                // stack: `groovy_class_members` already handled direct members,
                // and C does not re-walk method bodies for further defs. But a
                // nested class inside the body is rare and out of scope here.
            }
            // Top-level (script) function → a "Function". The Java grammar
            // parses a `def f(){}` as `function_definition` and a *typed*
            // top-level function (`int f(){}`) as a `method_declaration`;
            // both are labelled a top-level "Function". Only emit when NOT
            // inside a class/interface body (a `method_declaration` there is
            // a Method, handled by `groovy_class_members`).
            "function_definition" | "method_declaration" => {
                if !groovy_inside_class(node) {
                    if let Some(name) = node
                        .child_by_field_name("name")
                        .map(|n| node_text(source, n).to_string())
                    {
                        if !name.is_empty() {
                            result.nodes.push(ExtractedNode {
                                label: "Function".into(),
                                name: name.clone(),
                                qualified_name: format!("{file_path}::Function::{name}"),
                                file_path: file_path.to_string(),
                                start_line: node.start_position().row as u32 + 1,
                                end_line: node.end_position().row as u32 + 1,
                                properties: serde_json::json!({}),
                            });
                        }
                    }
                }
                // Do not descend into the function body.
            }
            _ => {
                let mut c = node.walk();
                for child in node.named_children(&mut c) {
                    stack.push(child);
                }
            }
        }
    }
}

/// True if `node` sits inside a class/interface declaration (used to decide
/// whether a `function_definition` is a free Function or a class Method).
fn groovy_inside_class(node: Node<'_>) -> bool {
    let mut p = node.parent();
    while let Some(cur) = p {
        if GROOVY_CLASS_KINDS.contains(&cur.kind()) {
            return true;
        }
        p = cur.parent();
    }
    false
}

/// Walk one class `body` node (Java grammar `class_body`) emitting owned Method
/// nodes (for concrete `method_declaration` members), field Variable nodes, and
/// the DEFINES_METHOD edge per method. Constructors and bodyless abstract
/// methods are skipped.
fn groovy_class_members(
    source: &[u8],
    body: Node<'_>,
    owner: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut c = body.walk();
    for member in body.named_children(&mut c) {
        match member.kind() {
            // A concrete method with a body → Method + DEFINES_METHOD.
            "method_declaration" => {
                // Skip bodyless (abstract) members; a `method_declaration` with
                // no `body:` block is an abstract signature.
                if member.child_by_field_name("body").is_none() {
                    continue;
                }
                let Some(name) = member
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n).to_string())
                else {
                    continue;
                };
                if name.is_empty() || name == owner {
                    // A member whose name equals the class name is a
                    // constructor — not emitted as a Method.
                    continue;
                }
                result.nodes.push(ExtractedNode {
                    label: "Method".into(),
                    name: name.clone(),
                    qualified_name: format!("{file_path}::{owner}::{name}"),
                    file_path: file_path.to_string(),
                    start_line: member.start_position().row as u32 + 1,
                    end_line: member.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
                result.edges.push(ExtractedEdge {
                    edge_type: "DEFINES_METHOD".into(),
                    source_qualified_name: format!("{file_path}::Class::{owner}"),
                    target_qualified_name: format!("{file_path}::{owner}::{name}"),
                    file_path: file_path.to_string(),
                    line: member.start_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            // Constructors are a distinct kind in the Java grammar; no Method
            // is emitted for them.
            "constructor_declaration" | "compact_constructor_declaration" => {}
            // A class field → a MODULE-scoped Variable (one per declarator).
            "field_declaration" => {
                groovy_emit_field_variables(source, member, file_path, result);
            }
            _ => {}
        }
    }
}

/// Emit a module-scoped "Variable" node for each declarator of one
/// `field_declaration`. Groovy fields are registered at file/module scope,
/// `{file}::Variable::{name}`.
fn groovy_emit_field_variables(
    source: &[u8],
    field: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut c = field.walk();
    for child in field.named_children(&mut c) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        let Some(name_node) = child.child_by_field_name("name") else {
            continue;
        };
        let name = node_text(source, name_node);
        if name.is_empty() || name == "_" {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: name.to_string(),
            qualified_name: format!("{file_path}::Variable::{name}"),
            file_path: file_path.to_string(),
            start_line: field.start_position().row as u32 + 1,
            end_line: field.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// Emit USAGE edges over the Groovy tree. The source is always the per-file
/// Module; each candidate `identifier` reference emits a usage whose `ref_name`
/// the indexer resolves to a unique def (unresolved / ambiguous refs are dropped
/// by the resolver).
///
/// A candidate reference is any `identifier` node that is NOT inside a call or
/// import, and NOT a keyword. The Java grammar puts a def's name in the `name:`
/// field, so `is_definition_name` excludes it here; each method/function NAME is
/// itself a usage, re-emitted via `groovy_emit_def_name_usages`.
fn groovy_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    // (a) Each method/function/class NAME → a self-usage.
    groovy_emit_def_name_usages(source, root, file_module_qname, file_path, result);
    // (b) Body identifier / type-identifier references. `identifier` and
    // `type_identifier` are treated alike, so a field's TYPE (`Catalog catalog`
    // → the `type_identifier` Catalog) is a reference too.
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if matches!(node.kind(), "identifier" | "type_identifier")
            && !is_inside_kind(node, &GROOVY_CALL_KINDS)
            && !is_inside_kind(node, &GROOVY_IMPORT_KINDS)
            && !is_definition_name(node)
            && !groovy_is_field_name(node)
        {
            let text = node_text(source, node);
            if !text.is_empty() && !groovy_is_keyword(text) {
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::__ref__::{text}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "ref_name": text }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit each Method / Function definition NAME as a USAGE (the def head
/// identifier is treated as a plain reference). The resolver dedups a
/// (Module, target) pair, so a def name that is also body-referenced still
/// counts once.
fn groovy_emit_def_name_usages(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let name_of = |n: Node<'_>| -> Option<String> {
            n.child_by_field_name("name")
                .map(|x| node_text(source, x).to_string())
        };
        // Only method / function definition NAMES are self-usages; a
        // class/interface NAME is NOT (no usage is emitted for the type's own
        // name — a reference to the type only appears where it is *used*, e.g.
        // a field's type, which the body-identifier walk covers via
        // `type_identifier`).
        let emit_name = match node.kind() {
            "function_definition" | "method_declaration" => name_of(node),
            _ => None,
        };
        if let Some(name) = emit_name {
            if !name.is_empty() && !groovy_is_keyword(&name) {
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::__ref__::{name}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "ref_name": name }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// True if `node` is the `name:` identifier of a `field_declaration`'s
/// `variable_declarator` (a field name, already a Variable def — not a usage).
fn groovy_is_field_name(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    if parent.kind() != "variable_declarator" {
        return false;
    }
    parent
        .child_by_field_name("name")
        .map(|n| n.start_byte() == node.start_byte() && n.end_byte() == node.end_byte())
        .unwrap_or(false)
}

/// Groovy call node kinds for the Java grammar: an identifier inside one of
/// these is already a CALLS edge, so it is not a USAGE.
const GROOVY_CALL_KINDS: [&str; 2] = ["method_invocation", "object_creation_expression"];

/// Groovy import node kinds for the Java grammar: an identifier inside an import
/// statement is not a USAGE.
const GROOVY_IMPORT_KINDS: [&str; 1] = ["import_declaration"];

/// Minimal Groovy keyword filter for the USAGE pass. Filtering the common
/// value/type keywords avoids spurious references from control-flow and
/// primitive-type tokens.
fn groovy_is_keyword(name: &str) -> bool {
    matches!(
        name,
        "def"
            | "class"
            | "interface"
            | "enum"
            | "extends"
            | "implements"
            | "package"
            | "import"
            | "return"
            | "if"
            | "else"
            | "for"
            | "in"
            | "while"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "new"
            | "this"
            | "super"
            | "static"
            | "final"
            | "abstract"
            | "public"
            | "private"
            | "protected"
            | "void"
            | "true"
            | "false"
            | "null"
            | "int"
            | "long"
            | "short"
            | "byte"
            | "char"
            | "float"
            | "double"
            | "boolean"
            | "String"
            | "List"
            | "Map"
            | "Set"
            | "Object"
            | "try"
            | "catch"
            | "finally"
            | "throw"
            | "throws"
            | "assert"
    )
}

// ===========================================================================
// OCaml — bespoke pass.
// ===========================================================================
//
// This pass models OCaml definitions and references as follows:
//
//   * **Class** — one importable compilation-unit namespace per source file,
//     named by OCaml's capitalized file stem (`helper.ml` -> `Helper`).
//   * **Function** — one per reachable `value_definition` (`let ... = ...`),
//     whether parameterized or constant. The first `let_binding`'s `pattern:`
//     field supplies the name; nested local lets are not emitted.
//   * **Type** — one per `type_binding`, covering records, variants, and aliases.
//   * **CALLS / USAGE / TYPE_REF** — source is the nearest emitted Function,
//     falling back to the per-file Module at file scope. Targets resolve by the
//     final path segment.
//   * **IMPORTS** — `open` / `include` module paths resolve to the compilation-
//     unit Class node named by their final segment.
//
// DEFINES and CONTAINS_* are auto-derived by the indexer's structural pass.
fn extract_ocaml(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    ocaml_emit_compilation_unit(file_path, root, &mut result);
    ocaml_defs_pass(source, root, file_path, &mut result);
    ocaml_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    ocaml_imports_pass(d, source, root, &file_module_qname, file_path, &mut result)?;
    ocaml_type_refs_pass(source, root, &file_module_qname, file_path, &mut result);
    ocaml_usages_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// Emit an importable namespace node for the OCaml compilation unit. OCaml
/// capitalizes the source-file stem (`helper.ml` -> `Helper`) when naming the
/// module. The shared IMPORTS resolver deliberately excludes synthetic Module
/// nodes, so the language pass models this namespace like Elixir's `defmodule`:
/// as a Class node carrying an explicit `ocaml_compilation_unit` marker.
fn ocaml_emit_compilation_unit(file_path: &str, root: Node<'_>, result: &mut ExtractionResult) {
    let file_name = file_path.rsplit('/').next().unwrap_or(file_path);
    let stem = file_name
        .rsplit_once('.')
        .map_or(file_name, |(stem, _)| stem);
    let mut chars = stem.chars();
    let Some(first) = chars.next() else {
        return;
    };
    let name = first.to_uppercase().chain(chars).collect::<String>();
    result.nodes.push(ExtractedNode {
        label: "Class".into(),
        name: name.clone(),
        qualified_name: format!("{file_path}::Class::{name}"),
        file_path: file_path.to_string(),
        start_line: 1,
        end_line: root.end_position().row as u32 + 1,
        properties: serde_json::json!({ "kind": "ocaml_compilation_unit" }),
    });
}

/// The OCaml node kinds that become a "Function" node.
const OCAML_FUNC_KINDS: [&str; 3] = [
    "value_definition",
    "constructor_declaration",
    "method_definition",
];

/// The OCaml call node kinds this pass reads: an `application_expression`'s
/// head, plus `infix_expression` (whose operator never names a user Function,
/// so it yields no resolvable edge).
const OCAML_CALL_KINDS: [&str; 2] = ["application_expression", "infix_expression"];

/// The OCaml import node kinds: a reference must NOT sit inside one of these to
/// count as a USAGE (`open Foo` / `include Bar`).
const OCAML_IMPORT_KINDS: [&str; 2] = ["open_module", "include_module"];

/// Emit one "Function" node per OCaml definition node. Walks the WHOLE tree
/// (module bodies included), descending into `module ... = struct ... end`
/// bodies even though the module itself emits no node.
fn ocaml_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "type_binding" {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(source, name_node);
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Type".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Type::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
        }
        if OCAML_FUNC_KINDS.contains(&node.kind()) {
            if let Some(name) = ocaml_def_name(source, node) {
                // Drop empty names and the literal "function"; the `_`
                // wildcard pattern is not a real binding.
                if !name.is_empty() && name != "function" && name != "_" {
                    result.nodes.push(ExtractedNode {
                        label: "Function".into(),
                        name: name.clone(),
                        qualified_name: format!("{file_path}::Function::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            // Stop after a definition — do NOT descend into a
            // function/definition body. So a local `let x = .. in ..` binding
            // inside a function is NOT a Function. Module bodies ARE descended
            // into, which the generic recursion below still reaches because
            // module_definition is not a func kind.
            continue;
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The definition NAME for one OCaml def node: for a `value_definition`, the
/// FIRST `let_binding`'s `pattern:`; otherwise the generic `name:` field for
/// `constructor_declaration` / `method_definition`.
fn ocaml_def_name(source: &[u8], node: Node<'_>) -> Option<String> {
    if node.kind() == "value_definition" {
        // First `let_binding` child only (the first match wins, so
        // `let a = .. and b = ..` names just `a`).
        let mut c = node.walk();
        let binding = node
            .named_children(&mut c)
            .find(|ch| ch.kind() == "let_binding")?;
        let pattern = binding.child_by_field_name("pattern")?;
        return Some(node_text(source, pattern).to_string());
    }
    node.child_by_field_name("name")
        .map(|n| node_text(source, n).to_string())
}

/// Resolve the nearest enclosing emitted OCaml callable. For top-level values,
/// the callable name lives under the first `let_binding`'s `pattern:` field,
/// not a generic `name:` field. Walking to the containing `value_definition`
/// also skips nested local `let` bindings, which do not emit Function nodes.
fn ocaml_enclosing_callable_qname(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
) -> Option<String> {
    let mut parent = node.parent();
    while let Some(candidate) = parent {
        if OCAML_FUNC_KINDS.contains(&candidate.kind())
            && !ocaml_is_nested_callable_definition(candidate)
        {
            let name = ocaml_def_name(source, candidate)?;
            if !name.is_empty() {
                return Some(format!("{file_path}::Function::{name}"));
            }
        }
        parent = candidate.parent();
    }
    None
}

/// The defs walk stops after emitting a callable, so a callable-shaped node
/// nested inside another callable is not a graph node and cannot own edges.
fn ocaml_is_nested_callable_definition(node: Node<'_>) -> bool {
    let mut parent = node.parent();
    while let Some(candidate) = parent {
        if OCAML_FUNC_KINDS.contains(&candidate.kind()) {
            return true;
        }
        parent = candidate.parent();
    }
    false
}

/// Emit CALLS edges from the nearest callable (or per-file Module) to each applied function.
/// The callee is resolved by its final path segment; the indexer's resolver
/// then links it same-file (by the direct `{file}::Function::{seg}` qname) or
/// cross-file (by unique name).
fn ocaml_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if OCAML_CALL_KINDS.contains(&node.kind()) {
            if let Some(callee) = ocaml_callee_name(source, node) {
                if !callee.is_empty() {
                    let source_qname = ocaml_enclosing_callable_qname(source, node, file_path)
                        .unwrap_or_else(|| file_module_qname.to_string());
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: source_qname,
                        target_qualified_name: format!("{file_path}::Function::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": callee,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The callee NAME of an OCaml call node. For `application_expression` the head
/// (`function:` field) is read; only a `value_path` / bare name names a
/// resolvable Function — the final `value_name` segment is the resolvable name.
/// `infix_expression` yields the operator, which never names a user Function,
/// so it produces `None`.
fn ocaml_callee_name(source: &[u8], node: Node<'_>) -> Option<String> {
    if node.kind() != "application_expression" {
        return None;
    }
    let head = node.child_by_field_name("function")?;
    match head.kind() {
        // `M.f` / `f`: the resolvable name is the final `value_name`.
        "value_path" => ocaml_value_path_leaf(source, head),
        // A bare constructor application (`Some x`) — the constructor is not a
        // Function node, so its callee ("Some") never resolves. We still take
        // the leaf so the resolver simply finds no match (no spurious edge).
        "constructor_path" => ocaml_value_path_leaf(source, head),
        _ => None,
    }
}

/// The final path segment of a `value_path` / `constructor_path`: its last
/// `value_name` / `constructor_name` child (`Str_ext.banner` → `banner`).
/// Falls back to the whole node's text when no segment child is present.
fn ocaml_value_path_leaf(source: &[u8], node: Node<'_>) -> Option<String> {
    let mut c = node.walk();
    let leaf = node
        .named_children(&mut c)
        .filter(|ch| matches!(ch.kind(), "value_name" | "constructor_name"))
        .last();
    match leaf {
        Some(l) => Some(node_text(source, l).to_string()),
        None => Some(node_text(source, node).to_string()),
    }
}

/// Emit one file-sourced IMPORTS edge for each `open` / `include` module path.
/// The target name is the final module segment; compilation-unit Class nodes
/// provide importable namespace targets without changing shared resolution.
fn ocaml_imports_pass(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let language = (d.grammar)();
    let query = tree_sitter::Query::new(&language, d.import_query)
        .map_err(|e| greppy_core::Error::Parse(format!("compile ocaml imports query: {e}")))?;
    let imported_capture = query
        .capture_index_for_name("imported")
        .ok_or_else(|| greppy_core::Error::Parse("ocaml imports query lacks @imported".into()))?;
    let mut cursor = QueryCursor::new();
    let mut captures = cursor.captures(&query, root, source);
    while let Some((query_match, capture_index)) = captures.next() {
        let capture = query_match.captures[*capture_index];
        if capture.index != imported_capture {
            continue;
        }
        let path = node_text(source, capture.node);
        let imported_name = path.rsplit('.').next().unwrap_or(path);
        if path.is_empty() || imported_name.is_empty() {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "IMPORTS".into(),
            source_qualified_name: file_module_qname.to_string(),
            target_qualified_name: format!("{file_path}::__import__::{path}"),
            file_path: file_path.to_string(),
            line: capture.node.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "path": path,
                "imported_name": imported_name,
                "original_name": imported_name,
                "glob": true,
            }),
        });
    }
    Ok(())
}

/// Emit TYPE_REF edges for qualified and bare OCaml type constructors. A path
/// such as `Types.widget` resolves by its final segment (`widget`) to the Type
/// node emitted from the corresponding `type_binding`.
fn ocaml_type_refs_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let is_path = node.kind() == "type_constructor_path";
        let is_bare = node.kind() == "type_constructor"
            && node.parent().map(|parent| parent.kind()) != Some("type_constructor_path");
        if (is_path || is_bare) && !is_definition_name(node) {
            let path = node_text(source, node);
            let type_name = path.rsplit('.').next().unwrap_or(path);
            if !type_name.is_empty() {
                let source_qname = ocaml_enclosing_callable_qname(source, node, file_path)
                    .unwrap_or_else(|| file_module_qname.to_string());
                result.edges.push(ExtractedEdge {
                    edge_type: "TYPE_REF".into(),
                    source_qualified_name: source_qname,
                    target_qualified_name: format!("{file_path}::__type__::{type_name}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "type_name": type_name }),
                });
            }
            if is_path {
                continue;
            }
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            stack.push(child);
        }
    }
}

/// Emit USAGE edges for OCaml: every `value_path` / `constructor_path`
/// reference that is NOT inside a call or an `open`/`include`, and is not a
/// definition name. The source is the per-file `Module` node; the reference
/// resolves by its final segment.
fn ocaml_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if matches!(node.kind(), "value_path" | "constructor_path")
            && !ocaml_is_inside(node, &OCAML_CALL_KINDS)
            && !ocaml_is_inside(node, &OCAML_IMPORT_KINDS)
            && !is_definition_name(node)
        {
            if let Some(refname) = ocaml_value_path_leaf(source, node) {
                if !refname.is_empty() && !ocaml_is_keyword(&refname) {
                    let source_qname = ocaml_enclosing_callable_qname(source, node, file_path)
                        .unwrap_or_else(|| file_module_qname.to_string());
                    result.edges.push(ExtractedEdge {
                        edge_type: "USAGE".into(),
                        source_qualified_name: source_qname,
                        target_qualified_name: format!("{file_path}::__ref__::{refname}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({ "ref_name": refname }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// True if `node` sits inside an ancestor of one of `kinds`, within a
/// 10-parent depth bound.
fn ocaml_is_inside(node: Node<'_>, kinds: &[&str]) -> bool {
    const MAX_PARENT_DEPTH: usize = 10;
    let mut cur = node.parent();
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= MAX_PARENT_DEPTH {
            break;
        }
        if kinds.contains(&n.kind()) {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// A minimal OCaml keyword filter for the USAGE pass. A `value_path` leaf is a
/// lowercase identifier, so only the value-position keywords can appear;
/// filtering them avoids spurious references.
fn ocaml_is_keyword(name: &str) -> bool {
    matches!(
        name,
        "let"
            | "in"
            | "and"
            | "rec"
            | "fun"
            | "function"
            | "match"
            | "with"
            | "if"
            | "then"
            | "else"
            | "begin"
            | "end"
            | "module"
            | "struct"
            | "sig"
            | "type"
            | "open"
            | "include"
            | "true"
            | "false"
            | "when"
            | "as"
            | "of"
            | "val"
            | "mutable"
            | "ref"
            | "for"
            | "to"
            | "do"
            | "done"
            | "while"
    )
}

// ===========================================================================
// Crystal — bespoke pass.
// ===========================================================================
//
// This pass models Crystal like this:
//
//   * **Class** — one per `class_def` / `struct_def` / `module_def` /
//     `enum_def` / `annotation_def`. Every one of these kinds is labelled
//     "Class" (none matches the Interface / Enum / Type keyword lists — note
//     Crystal's `enum_def` is NOT the `enum_specifier` / `enum_declaration` /
//     `enum_item` that maps to "Enum"). `type_declaration` (an `@ivar : T` field
//     decl) carries no `name:` field, so it resolves no name and emits nothing.
//     The per-file synthetic node is the only "Module".
//   * **Method** — one per `method_def` that is a DIRECT member of a class-type
//     body (each `method_def` in the body becomes a Method scoped
//     `{Owner}.{name}`), with a DEFINES_METHOD edge from the owning type.
//   * **Function** — one per `method_def`, re-walked (descending into class
//     bodies but not into method bodies). The free-Function qname carries NO
//     owner segment, so two same-named methods in one file collapse to one node
//     via store dedup (this is why two `initialize` methods in index.cr yield
//     Method 47 but Function 46).
//   * **CALLS** — a `call` whose `method:` is an `identifier` (operator calls
//     carry an `operator`, receiver calls still expose the final method
//     identifier). Source is the enclosing `method_def`'s Method qname, or the
//     per-file Module (`{file}::__file__`) at top level.
//   * **USAGE** — an `identifier` reference that is not inside a call, not a
//     definition name, and not a keyword (a bare `identifier` is a reference;
//     Crystal `constant`s are a different node kind and are not references).
//
// DEFINES (File→def) is auto-derived by the structural pass from the node set
// above. `require "..."` → IMPORTS and constructor (`.new`) CALLS do NOT resolve
// on cross-file/constructor paths and are out of this pass's scope, so none are
// emitted.

/// The Crystal type-declaration kinds that become a "Class" (excluding
/// `type_declaration`, which has no resolvable name).
const CRYSTAL_CLASS_KINDS: [&str; 5] = [
    "class_def",
    "struct_def",
    "module_def",
    "enum_def",
    "annotation_def",
];

fn extract_crystal(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    crystal_defs_pass(source, root, file_path, &mut result);
    crystal_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    crystal_usages_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// The `name:` (`constant` for a type, `identifier` for a method) of a Crystal
/// def node.
fn crystal_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    node.child_by_field_name("name")
        .map(|n| node_text(source, n))
}

/// The `name:` of the nearest ancestor class-type of `node` (its owner), or
/// `None` when `node` is not inside any class/struct/module/enum.
fn crystal_owner_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if CRYSTAL_CLASS_KINDS.contains(&cur.kind()) {
            return crystal_name(source, cur);
        }
        p = cur.parent();
    }
    None
}

/// Defs pass: Class nodes (every class-type), Method nodes (owned methods) with
/// their DEFINES_METHOD edge, and the double-counted free Function node per
/// method (re-walk; owner-less qname → same-name dedup).
fn crystal_defs_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        if CRYSTAL_CLASS_KINDS.contains(&kind) {
            // `type_declaration` is excluded from CRYSTAL_CLASS_KINDS, so every
            // node here has a `name:` constant → a "Class" node.
            if let Some(name) = crystal_name(source, node) {
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Class".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Class::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
        } else if kind == "method_def" {
            if let Some(name) = crystal_name(source, node) {
                if !name.is_empty() {
                    let start = node.start_position().row as u32 + 1;
                    let end = node.end_position().row as u32 + 1;
                    // Owned method → "Method" (scoped) + DEFINES_METHOD.
                    if let Some(owner) = crystal_owner_name(source, node) {
                        result.nodes.push(ExtractedNode {
                            label: "Method".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::{owner}::{name}"),
                            file_path: file_path.to_string(),
                            start_line: start,
                            end_line: end,
                            properties: serde_json::json!({}),
                        });
                        result.edges.push(ExtractedEdge {
                            edge_type: "DEFINES_METHOD".into(),
                            source_qualified_name: format!("{file_path}::Class::{owner}"),
                            target_qualified_name: format!("{file_path}::{owner}::{name}"),
                            file_path: file_path.to_string(),
                            line: start,
                            properties: serde_json::json!({}),
                        });
                    }
                    // Every method_def is ALSO re-walked into a free "Function".
                    // Owner-less qname → same-name dedup.
                    result.nodes.push(ExtractedNode {
                        label: "Function".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Function::{name}"),
                        file_path: file_path.to_string(),
                        start_line: start,
                        end_line: end,
                        properties: serde_json::json!({}),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// CALLS pass: a `call` whose `method:` is an `identifier` (excludes operator
/// calls). Source is the enclosing method's Method qname, else the per-file
/// Module. Target resolves by `callee_name` (same-file `{file}::Function::seg`
/// or cross-file by unique name) exactly like the spec CALLS pass.
fn crystal_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "call" {
            if let Some(callee) = node.child_by_field_name("method") {
                if callee.kind() == "identifier" {
                    let name = node_text(source, callee);
                    if !name.is_empty() && !crystal_is_keyword(name) {
                        let src = crystal_enclosing_callable_qname(source, node, file_path)
                            .unwrap_or_else(|| file_module_qname.to_string());
                        result.edges.push(ExtractedEdge {
                            edge_type: "CALLS".into(),
                            source_qualified_name: src,
                            target_qualified_name: format!("{file_path}::Function::{name}"),
                            file_path: file_path.to_string(),
                            line: node.start_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "callee_text": name,
                                "callee_name": name,
                            }),
                        });
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The Method qname of the nearest enclosing `method_def`, matching the def
/// pass (`{file}::{owner}::{name}`); `None` when the node is not inside a
/// method (top-level → the CALLS source falls back to the file module).
fn crystal_enclosing_callable_qname(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "method_def" {
            let name = crystal_name(source, cur)?;
            let owner = crystal_owner_name(source, cur)?;
            return Some(format!("{file_path}::{owner}::{name}"));
        }
        p = cur.parent();
    }
    None
}

/// USAGE pass: an `identifier` reference that is not inside a call, not a
/// definition name, and not a keyword. Source is the enclosing `method_def`,
/// else the per-file Module; the ref resolves by `ref_name`. Sourcing on the
/// enclosing method (not the file) gives per-function attribution, so two
/// references to the same name from DIFFERENT methods are two distinct edges.
fn crystal_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "identifier"
            && !is_inside_kind(node, &["call"])
            && !is_definition_name(node)
        {
            let name = node_text(source, node);
            if !name.is_empty() && !crystal_is_keyword(name) {
                let src = crystal_enclosing_callable_qname(source, node, file_path)
                    .unwrap_or_else(|| file_module_qname.to_string());
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: src,
                    target_qualified_name: format!("{file_path}::__ref__::{name}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "ref_name": name }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Crystal keyword filter. A callee / reference `identifier` is a lowercase
/// word, so only the value-position keywords can appear here.
fn crystal_is_keyword(name: &str) -> bool {
    matches!(
        name,
        "abstract"
            | "alias"
            | "as"
            | "begin"
            | "break"
            | "case"
            | "class"
            | "def"
            | "do"
            | "else"
            | "elsif"
            | "end"
            | "ensure"
            | "enum"
            | "extend"
            | "false"
            | "for"
            | "fun"
            | "if"
            | "in"
            | "include"
            | "is_a?"
            | "lib"
            | "macro"
            | "module"
            | "next"
            | "nil"
            | "of"
            | "out"
            | "private"
            | "protected"
            | "require"
            | "rescue"
            | "return"
            | "select"
            | "self"
            | "struct"
            | "super"
            | "then"
            | "true"
            | "typeof"
            | "uninitialized"
            | "union"
            | "unless"
            | "until"
            | "when"
            | "while"
            | "with"
            | "yield"
    )
}

// ===========================================================================
// Solidity — bespoke pass (registry language).
// ===========================================================================
//
// The Solidity taxonomy (verified by dumping the graph for a real .sol corpus):
//
//   * DEFINITIONS:
//       - `contract_declaration` / `library_declaration` / `struct_declaration`
//         → "Class". `interface_declaration` → "Interface".
//         `enum_declaration` → "Enum". (contract/library/struct all collapse to
//         the generic "Class" label; only interfaces get "Interface".)
//       - `struct_member` → "Field" (struct fields, single node).
//       - `state_variable_declaration` (a contract-level variable) →
//         "Field" (scoped, owned by the enclosing class) AND "Variable"
//         (file-level twin) — a double-count for member+variable.
//       - `function_definition` / `modifier_definition` → "Method" (scoped,
//         owned by the enclosing class) + DEFINES_METHOD, AND a "Function"
//         twin (re-walk). A top-level (free) function that is not inside any
//         class emits ONLY a "Function" (no owner, no twin).
//   * INHERITS / IMPLEMENTS (`inheritance_specifier`): each ancestor of a
//     `contract_declaration` links the derived Class to the base. When the base
//     resolves to an "Interface" → IMPLEMENTS, else (a "Class") → INHERITS.
//     Resolved by unique base name project-wide.
//   * CALLS: a `call_expression` whose `function:` is a plain `identifier` (or a
//     `member_expression` whose `property:` is the callee).
//     Source = enclosing callable's Method qname, else the per-file Module.
//     Target = same-file `{file}::Function::{callee}` guess + `callee_name`, so
//     the shared resolver links same-file calls directly and unique cross-file
//     names by name (ambiguous cross-file stays unresolved — honesty guard).
//   * USAGE: every `identifier` reference not inside a call / import / using
//     directive, not a definition name, and not a keyword. Source = enclosing
//     callable (per-function attribution), else the per-file Module; resolved by
//     `ref_name`.
//
// OUT OF SCOPE on this fixture (noted, not forced): IMPORTS (resolving the
// relative `import "..."` path to the imported file's Module node — a file-path
// import resolution the shared plumbing's name-based IMPORTS pass and
// IMPORTABLE_LABELS, which excludes Module, cannot express) and SIMILAR_TO
// (SEMANTICALLY_RELATED embeddings).

/// The Solidity owner kinds: node kinds that become graph def-nodes with a
/// `name:` field and own the functions/members lexically inside them.
const SOLIDITY_OWNER_KINDS: [&str; 3] = [
    "contract_declaration",
    "interface_declaration",
    "library_declaration",
];

fn extract_solidity(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    solidity_defs_pass(source, root, file_path, &mut result);
    solidity_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    solidity_usages_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// The `name:` identifier text of a Solidity def node.
fn solidity_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    node.child_by_field_name("name")
        .map(|n| node_text(source, n))
        .filter(|s| !s.is_empty())
}

/// The label for a Solidity type-def kind: "Interface" for interfaces, "Enum"
/// for enums, "Class" for everything else (contract/library/struct).
fn solidity_type_label(kind: &str) -> &'static str {
    match kind {
        "interface_declaration" => "Interface",
        "enum_declaration" => "Enum",
        _ => "Class",
    }
}

/// The nearest ancestor owner (contract/interface/library) name of `node`, or
/// `None` when `node` is not inside one (a free / file-level function).
fn solidity_owner_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if SOLIDITY_OWNER_KINDS.contains(&cur.kind()) {
            return solidity_name(source, cur);
        }
        p = cur.parent();
    }
    None
}

/// The Method qname of the nearest enclosing function/modifier (matching the
/// def pass, `{file}::{owner}::{name}`), or `None` when the reference is not
/// inside an OWNED callable (top-level / free-function bodies fall back to the
/// per-file Module for CALLS/USAGE sourcing).
fn solidity_enclosing_callable_qname(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if matches!(cur.kind(), "function_definition" | "modifier_definition") {
            let name = solidity_name(source, cur)?;
            let owner = solidity_owner_name(source, cur)?;
            return Some(format!("{file_path}::{owner}::{name}"));
        }
        p = cur.parent();
    }
    None
}

/// DEFS pass — Class/Interface/Enum type nodes, struct-member Fields,
/// state-variable Field+Variable twins, and Method (owned) + Function twins.
fn solidity_defs_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        match kind {
            "contract_declaration"
            | "interface_declaration"
            | "library_declaration"
            | "struct_declaration"
            | "enum_declaration" => {
                if let Some(name) = solidity_name(source, node) {
                    let label = solidity_type_label(kind);
                    result.nodes.push(ExtractedNode {
                        label: label.into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::{label}::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            "struct_member" => {
                // A struct field → "Field" only (single node, owner = the
                // enclosing struct).
                if let (Some(name), Some(owner)) = (
                    solidity_name(source, node),
                    solidity_owner_struct_name(source, node),
                ) {
                    result.nodes.push(ExtractedNode {
                        label: "Field".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::{owner}::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            "state_variable_declaration" => {
                // A contract-level variable → "Field" (scoped) + "Variable"
                // (file-level twin), a double-count.
                if let Some(name) = solidity_name(source, node) {
                    let start = node.start_position().row as u32 + 1;
                    let end = node.end_position().row as u32 + 1;
                    if let Some(owner) = solidity_owner_name(source, node) {
                        result.nodes.push(ExtractedNode {
                            label: "Field".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::{owner}::{name}"),
                            file_path: file_path.to_string(),
                            start_line: start,
                            end_line: end,
                            properties: serde_json::json!({}),
                        });
                    }
                    result.nodes.push(ExtractedNode {
                        label: "Variable".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Variable::{name}"),
                        file_path: file_path.to_string(),
                        start_line: start,
                        end_line: end,
                        properties: serde_json::json!({}),
                    });
                }
            }
            "function_definition" | "modifier_definition" => {
                if let Some(name) = solidity_name(source, node) {
                    let start = node.start_position().row as u32 + 1;
                    let end = node.end_position().row as u32 + 1;
                    // Owned function/modifier → "Method" + DEFINES_METHOD.
                    if let Some(owner) = solidity_owner_name(source, node) {
                        result.nodes.push(ExtractedNode {
                            label: "Method".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::{owner}::{name}"),
                            file_path: file_path.to_string(),
                            start_line: start,
                            end_line: end,
                            properties: serde_json::json!({}),
                        });
                        // Owner label: interface → Interface, else Class.
                        let owner_label = solidity_owner_label(node);
                        result.edges.push(ExtractedEdge {
                            edge_type: "DEFINES_METHOD".into(),
                            source_qualified_name: format!("{file_path}::{owner_label}::{owner}"),
                            target_qualified_name: format!("{file_path}::{owner}::{name}"),
                            file_path: file_path.to_string(),
                            line: start,
                            properties: serde_json::json!({}),
                        });
                    }
                    // Every function/modifier is ALSO re-walked into a free
                    // "Function". A free (top-level) function with no owner is
                    // emitted here exactly once.
                    result.nodes.push(ExtractedNode {
                        label: "Function".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Function::{name}"),
                        file_path: file_path.to_string(),
                        start_line: start,
                        end_line: end,
                        properties: serde_json::json!({}),
                    });
                }
            }
            _ => {}
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The nearest enclosing `struct_declaration` name of a `struct_member`.
fn solidity_owner_struct_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "struct_declaration" {
            return solidity_name(source, cur);
        }
        p = cur.parent();
    }
    None
}

/// The label of the nearest owner (Interface for `interface_declaration`, else
/// Class) — used to source DEFINES_METHOD from the correct def-node qname.
fn solidity_owner_label(node: Node<'_>) -> &'static str {
    let mut p = node.parent();
    while let Some(cur) = p {
        if SOLIDITY_OWNER_KINDS.contains(&cur.kind()) {
            return solidity_type_label(cur.kind());
        }
        p = cur.parent();
    }
    "Class"
}

// INHERITS / IMPLEMENTS are OUT OF SCOPE on this fixture. They would come from
// each `contract Derived is Base` specifier, but on this corpus every base
// (IERC20, IVault, Ownable, Token) is defined in a DIFFERENT file, so the edge
// is a cross-file inheritance link. The shared plumbing only name-resolves
// CALLS / TYPE_REF / USES / USAGE (INHERITS/IMPLEMENTS fall through to a
// direct-qname target which cannot name a cross-file base at extract time), so
// these are the same cross-file-resolution category the honesty guard excludes.
// We therefore emit no INHERITS/IMPLEMENTS and note the omission.

/// CALLS pass. A `call_expression` whose callee is a plain `identifier` (or the
/// `property:` identifier of a `member_expression`). Source = enclosing Method
/// qname, else per-file Module. Target = same-file `{file}::Function::{callee}`
/// guess + `callee_name`, resolved by the shared plumbing.
fn solidity_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "call_expression" {
            if let Some(callee) = solidity_callee_name(source, node) {
                if !callee.is_empty() && !solidity_is_keyword(callee) {
                    let src = solidity_enclosing_callable_qname(source, node, file_path)
                        .unwrap_or_else(|| file_module_qname.to_string());
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: src,
                        target_qualified_name: format!("{file_path}::Function::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": callee,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The callee name of a `call_expression`: the trailing identifier of its
/// `function:` expression (plain `identifier`, or the `property:` of a
/// `member_expression`).
fn solidity_callee_name<'a>(source: &'a [u8], call: Node<'_>) -> Option<&'a str> {
    let func = call.child_by_field_name("function")?;
    // `function:` is an `expression` wrapping the actual callee.
    let inner = func.named_child(0).unwrap_or(func);
    match inner.kind() {
        "identifier" => Some(node_text(source, inner)),
        "member_expression" => inner
            .child_by_field_name("property")
            .map(|p| node_text(source, p)),
        _ => {
            // Fall back to the last identifier under `function:`.
            let mut last = None;
            let mut stack = vec![inner];
            while let Some(n) = stack.pop() {
                if n.kind() == "identifier" {
                    last = Some(node_text(source, n));
                }
                let mut c = n.walk();
                for child in n.named_children(&mut c) {
                    stack.push(child);
                }
            }
            last
        }
    }
}

/// Names of every `enum_declaration` in the file (its `name:` identifier).
/// References to these are excluded from the USAGE pass (a Solidity usage is
/// never resolved to an Enum node).
fn solidity_enum_names(source: &[u8], root: Node<'_>) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "enum_declaration" {
            if let Some(name) = solidity_name(source, node) {
                names.insert(name.to_string());
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
    names
}

/// USAGE pass. Every `identifier` reference not inside a call / import / using
/// directive / inheritance / definition name / type position, and not a
/// keyword. Source = enclosing callable, else per-file Module; resolved by
/// `ref_name`.
///
/// One deviation from the naive walk: a reference to an ENUM type name (e.g.
/// `Status` in `Status.Frozen`, `Role` in `Role.Admin` or a `Role role` param
/// type) is NOT a USAGE. A Solidity usage is never resolved to an `Enum` node
/// (Enum is absent from the usage-target label set), so those edges must not be
/// emitted. We collect the enum names declared in the file and skip identifiers
/// that match one — every enum referenced on this fixture is declared in the
/// same file it is used in.
fn solidity_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let enum_names = solidity_enum_names(source, root);
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "identifier"
            && !is_inside_kind(
                node,
                &["call_expression", "import_directive", "using_directive"],
            )
            && !is_definition_name(node)
        {
            let name = node_text(source, node);
            if !name.is_empty() && !solidity_is_keyword(name) && !enum_names.contains(name) {
                let src = solidity_enclosing_callable_qname(source, node, file_path)
                    .unwrap_or_else(|| file_module_qname.to_string());
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: src,
                    target_qualified_name: format!("{file_path}::__ref__::{name}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "ref_name": name }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Solidity value-position keyword filter (a callee / reference identifier is a
/// lowercase word, so only value-position keywords can appear here).
fn solidity_is_keyword(name: &str) -> bool {
    matches!(
        name,
        "abstract"
            | "address"
            | "as"
            | "assembly"
            | "bool"
            | "break"
            | "constant"
            | "constructor"
            | "continue"
            | "contract"
            | "delete"
            | "do"
            | "else"
            | "emit"
            | "enum"
            | "event"
            | "external"
            | "false"
            | "for"
            | "function"
            | "if"
            | "import"
            | "indexed"
            | "interface"
            | "internal"
            | "is"
            | "library"
            | "mapping"
            | "memory"
            | "modifier"
            | "new"
            | "payable"
            | "pragma"
            | "private"
            | "public"
            | "pure"
            | "require"
            | "return"
            | "returns"
            | "storage"
            | "string"
            | "struct"
            | "true"
            | "using"
            | "view"
            | "virtual"
            | "while"
    )
}

// ===========================================================================
// Erlang — bespoke pass (registry language).
// ===========================================================================
//
// The Erlang passes:
//
//   * DEFINITIONS:
//       - `function_clause` → "Function". The grammar wraps every clause in its
//         own `fun_decl`, so a multi-clause function yields one Function per
//         clause (same name/qname) — an intentional over-count.
//       - `type_alias` → "Type". Name = the `name:` `type_name` node's text.
//       - `record_decl` / `pp_define` → "Variable", but ONLY at file-root scope.
//         The name is the first child of kind {atom, var, macro_lhs}.
//   * IMPORTS: one import per root-level child whose kind is in the import set
//     ({module_attribute, import, include}). Only `-module(x)` parses as a
//     `module_attribute`; `-import`/`-include` parse as `import_attribute` /
//     `pp_include`, which do NOT match, so each file contributes exactly one
//     IMPORTS whose target is the file's own per-file Module ("x"). The imported
//     name comes from the node's `name:` field.
//   * CALLS: the callee of a `call` node is the text of its FIRST child, i.e.
//     the `expr:` atom. A call links ONLY when the callee resolves to a
//     same-file Function (a remote `mod:fun(...)` whose inner `call` names `fun`
//     does NOT resolve cross-file). We reproduce this by emitting a CALLS edge
//     with a direct same-file target qname and NO `callee_name` property, so the
//     indexer's cross-file name fallback is skipped (only the direct same-file
//     qname match resolves). Source = enclosing `function_clause` qname; dedup
//     by (source, target) is applied by the store's upsert.
//   * THROWS: identical to CALLS — every resolvable call is ALSO a THROWS edge
//     (same source/target, same dedup).
//   * USAGE: every `atom` / `var` reference not inside a call / import, not a
//     definition name, and not a keyword. Resolved by name against the project's
//     defs by the indexer.
fn extract_erlang(
    language: Language,
    _d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    erlang_defs_pass(source, root, file_path, &mut result);
    erlang_imports_pass(source, root, file_path, &mut result);
    erlang_calls_pass(source, root, file_path, &mut result);
    erlang_usages_pass(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The Erlang call kinds — a reference inside one of these is a CALLS candidate,
/// so the usage walk skips it.
const ERLANG_CALL_KINDS: [&str; 1] = ["call"];

/// The Erlang import kinds — a reference inside one of these is skipped by the
/// usage walk. Only `module_attribute` (`-module(x)`) actually occurs at the
/// point references live; `import`/`include` are the declared kinds but the
/// grammar emits `import_attribute` / `pp_include` (which never match), so in
/// practice only `module_attribute` suppresses a reference. Listed for
/// completeness.
const ERLANG_IMPORT_KINDS: [&str; 3] = ["module_attribute", "import", "include"];

/// DEFINITIONS pass. Emits one "Function" per `function_clause` (anywhere in
/// the tree — the whole file is descended), one "Type" per `type_alias`, and
/// one "Variable" per file-root `record_decl` / `pp_define`.
fn erlang_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    // Functions + Types: full-tree walk.
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "function_clause" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let name = node_text(source, name_node);
                    if !name.is_empty() {
                        result.nodes.push(ExtractedNode {
                            label: "Function".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::Function::{name}"),
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                }
                // Stop after a function (no descent into the clause body), so
                // nested `call`s never spawn a Function.
                continue;
            }
            "type_alias" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    // Read the text of the whole `name:` node (`type_name`),
                    // e.g. "money()" — one Type per `type_alias`.
                    let name = node_text(source, name_node);
                    if !name.is_empty() {
                        result.nodes.push(ExtractedNode {
                            label: "Type".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::Type::{name}"),
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                }
            }
            _ => {}
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }

    // Variables: file-root `record_decl` / `pp_define` only (module-level
    // guard). The name is the first named child of kind {atom, var, macro_lhs}.
    let mut rc = root.walk();
    for child in root.named_children(&mut rc) {
        if !matches!(child.kind(), "record_decl" | "pp_define") {
            continue;
        }
        let Some(name) = erlang_var_name(source, child) else {
            continue;
        };
        if name.is_empty() || name == "_" {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: name.to_string(),
            qualified_name: format!("{file_path}::Variable::{name}"),
            file_path: file_path.to_string(),
            start_line: child.start_position().row as u32 + 1,
            end_line: child.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// The variable NAME of a `record_decl` / `pp_define`: the first named child of
/// kind {atom, var, macro_lhs}. For `-record(account, {...})` this is the `atom`
/// "account"; for `-define(MAX, 1000)` it is the `macro_lhs` (whose text is
/// "MAX").
fn erlang_var_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        if matches!(child.kind(), "atom" | "var" | "macro_lhs") {
            return Some(node_text(source, child));
        }
    }
    None
}

/// IMPORTS pass. One IMPORTS edge per root-level child whose kind is in the
/// import set (in practice only `module_attribute`, i.e. `-module(x)`). The
/// imported name is the node's `name:` field (the module atom), which the
/// indexer resolves to the per-file Module node "x". Source = the file's
/// per-file Module node.
fn erlang_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let file_module_qname = format!("{file_path}::__file__");
    let mut rc = root.walk();
    for child in root.named_children(&mut rc) {
        if !ERLANG_IMPORT_KINDS.contains(&child.kind()) {
            continue;
        }
        // The module atom is the `name:` field.
        let Some(name_node) = child.child_by_field_name("name") else {
            continue;
        };
        let imported = node_text(source, name_node);
        if imported.is_empty() {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "IMPORTS".into(),
            source_qualified_name: file_module_qname.clone(),
            target_qualified_name: format!("{file_path}::__import__::{imported}"),
            file_path: file_path.to_string(),
            line: child.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "imported_name": imported,
                "import_path": imported,
            }),
        });
    }
}

/// CALLS + THROWS pass. For every `call` node the callee is the text of its
/// FIRST child (the `expr:` atom). We emit a CALLS edge (and an identical THROWS
/// edge) whose target is the same-file `{file}::Function::{callee}` qname and —
/// crucially — WITHOUT a `callee_name` property, so the indexer resolves ONLY
/// the direct same-file qname match and never the cross-file name fallback
/// (remote `mod:fun(...)` calls are not linked). Source = the enclosing
/// `function_clause` qname. The store upsert dedups by (source, target, type).
fn erlang_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "call" {
            if let Some(callee) = erlang_callee_name(source, node) {
                if !callee.is_empty() && !erlang_is_keyword(callee) {
                    if let Some(src_qn) = erlang_enclosing_func_qname(source, node, file_path) {
                        let target = format!("{file_path}::Function::{callee}");
                        // CALLS and THROWS are the same resolvable set for
                        // Erlang ({call}).
                        for edge_type in ["CALLS", "THROWS"] {
                            result.edges.push(ExtractedEdge {
                                edge_type: edge_type.into(),
                                source_qualified_name: src_qn.clone(),
                                target_qualified_name: target.clone(),
                                file_path: file_path.to_string(),
                                line: node.start_position().row as u32 + 1,
                                // No `callee_name`: same-file direct target only.
                                properties: serde_json::json!({}),
                            });
                        }
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The callee NAME of an Erlang `call` node: the text of its first child, which
/// is the `expr:` atom (`log`, `system_time`, ...). Returns `None` when the call
/// has no children or the first child is not an atom-like name.
fn erlang_callee_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let first = node.child(0)?;
    // The `expr:` position of a `call` is an `atom` for a resolvable local
    // callee. (For a remote `mod:fun` the inner `call`'s first child is still
    // the `fun` atom.) Non-atom heads (rare) yield no resolvable callee.
    if first.kind() == "atom" {
        Some(node_text(source, first))
    } else {
        None
    }
}

/// The nearest enclosing `function_clause` qname for `node`
/// (`{file}::Function::{name}`), using the func kind set {function_clause}.
/// `None` at file scope.
fn erlang_enclosing_func_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "function_clause" {
            let name = cur.child_by_field_name("name")?;
            let name = node_text(source, name);
            if name.is_empty() {
                return None;
            }
            return Some(format!("{file_path}::Function::{name}"));
        }
        p = cur.parent();
    }
    None
}

/// USAGE pass for Erlang. Every `atom` / `var` reference that is NOT inside a
/// call / import, NOT a definition name, and NOT a keyword emits a USAGE edge
/// keyed on `ref_name`, resolved project-wide by the indexer. Source = the
/// nearest enclosing `function_clause` qname, else the per-file Module node.
fn erlang_usages_pass(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if matches!(node.kind(), "atom" | "var")
        && !is_inside_kind(node, &ERLANG_CALL_KINDS)
        && !is_inside_kind(node, &ERLANG_IMPORT_KINDS)
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !erlang_is_keyword(text) {
            let source_qname = erlang_enclosing_func_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({ "ref_name": text }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        erlang_usages_pass(source, child, file_path, file_module_qname, result);
    }
}

/// Erlang keyword / literal filter. Covers the reserved words that can appear
/// in an `atom` / `var` reference position so they never emit a usage or a
/// call.
fn erlang_is_keyword(name: &str) -> bool {
    matches!(
        name,
        "after"
            | "and"
            | "andalso"
            | "band"
            | "begin"
            | "bnot"
            | "bor"
            | "bsl"
            | "bsr"
            | "bxor"
            | "case"
            | "catch"
            | "cond"
            | "div"
            | "end"
            | "fun"
            | "if"
            | "let"
            | "not"
            | "of"
            | "or"
            | "orelse"
            | "receive"
            | "rem"
            | "try"
            | "when"
            | "xor"
            | "true"
            | "false"
    )
}

// ==================== F# ====================
//
// F# is a REGISTRY language (see `crate::langs::fsharp`). The generic spec
// path only emits coarse Type/Function defs and no CALLS/USAGE, so this bespoke
// pass handles F#:
//
//   * DEFINITIONS: `function_or_value_defn` → "Function" (name = the first
//     `identifier`/`long_identifier` child of its `function_declaration_left` /
//     `value_declaration_left`; a bare value binding whose lhs has neither —
//     e.g. `let pi = 3.14` — resolves to a NULL name and is SKIPPED, via a
//     non-recursive lookup). `type_definition` → "Type", upgraded to "Class"
//     when it has a `primary_constr_args` or `class_inherits_decl` descendant
//     (the OOP-class rule). `exception_definition` carries its name on
//     `exception_name`, not a `type_name`, so the type-name resolver finds
//     nothing and SKIPS it.
//   * CALLS: an `application_expression` whose head (first named child) is a
//     `long_identifier_or_op` / `long_identifier` / `identifier`. The callee
//     is that head's final path segment (a Function). The source is always
//     the per-file module: F# `function_or_value_defn` is not a func kind, so
//     no function endpoint is attached (identical to OCaml — source =
//     `<file>::__file__`).
//   * USAGE (generic path): every `identifier` reference NOT inside a call
//     (`application_expression` / `dot_expression`) or an import
//     (`import_decl` / `open_expression` / `instance`), and not a definition
//     name. Source = the per-file module.
//
// DEFINES / INHERITS / CONTAINS_* are auto-emitted by the shared indexer
// structure pass from the node set, so this pass emits only the nodes plus
// the CALLS and USAGE reference edges.

/// F# def nodes that become a "Function".
const FSHARP_FUNC_KINDS: [&str; 3] = [
    "function_declaration",
    "value_declaration",
    "function_or_value_defn",
];

/// F# def nodes that become a type/class node.
const FSHARP_CLASS_KINDS: [&str; 2] = ["type_definition", "exception_definition"];

/// F# call kinds — the nodes a reference must NOT sit inside to count as a
/// USAGE, and the CALLS source kinds (`dot_expression` never carries a
/// resolvable free-function callee, so it yields no CALLS edge but still masks
/// its inner identifiers from the USAGE pass).
const FSHARP_CALL_KINDS: [&str; 2] = ["application_expression", "dot_expression"];

/// F# import kinds: the nodes a reference must NOT sit inside to count as a
/// USAGE (`open Foo` / `import ...`).
const FSHARP_IMPORT_KINDS: [&str; 3] = ["import_decl", "open_expression", "instance"];

fn extract_fsharp(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    fsharp_defs_pass(source, root, file_path, &mut result);
    // Names of the "Type"-labeled defs this file emitted. A reference to one of
    // these never becomes a USAGE edge (the usage registry indexes only
    // Function/Method/Class/Interface), so the usage pass skips them.
    let type_names: std::collections::HashSet<String> = result
        .nodes
        .iter()
        .filter(|n| n.label == "Type")
        .map(|n| n.name.clone())
        .collect();
    fsharp_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    fsharp_usages_pass(
        source,
        root,
        &file_module_qname,
        file_path,
        &type_names,
        &mut result,
    );
    fsharp_inherits_pass(source, root, file_path, &mut result);

    Ok(result)
}

/// Emit one INHERITS edge per `type Foo(..) = inherit Base(..)`: a
/// `class_inherits_decl` → its `simple_type` base. Source = the derived class
/// node (`{file}::Class::{name}`); target = the base class's same-file qname
/// (`{file}::Class::{base}`). A base that names no in-file Class simply does not
/// resolve (no spurious edge).
fn fsharp_inherits_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if FSHARP_CLASS_KINDS.contains(&node.kind()) {
            if let Some((name, _label)) = fsharp_type_name_and_label(source, node) {
                if let Some(inh) = first_descendant_of_kind(node, "class_inherits_decl") {
                    if let Some(st) = first_descendant_of_kind(inh, "simple_type") {
                        let base = fsharp_simple_type_leaf(source, st);
                        if !base.is_empty() && !name.is_empty() {
                            result.edges.push(ExtractedEdge {
                                edge_type: "INHERITS".into(),
                                source_qualified_name: format!("{file_path}::Class::{name}"),
                                target_qualified_name: format!("{file_path}::Class::{base}"),
                                file_path: file_path.to_string(),
                                line: node.start_position().row as u32 + 1,
                                properties: serde_json::json!({
                                    "name": base,
                                    "base_name": base,
                                }),
                            });
                        }
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The final `identifier` segment of a `simple_type` base-type node
/// (`simple_type > long_identifier > identifier` → the last identifier).
fn fsharp_simple_type_leaf(source: &[u8], node: Node<'_>) -> String {
    match first_descendant_of_kind(node, "long_identifier") {
        Some(li) => fsharp_ident_leaf(source, li),
        None => match first_descendant_of_kind(node, "identifier") {
            Some(id) => node_text(source, id).to_string(),
            None => node_text(source, node).to_string(),
        },
    }
}

/// Emit one Function / Type / Class node per F# definition node. Functions are
/// matched FIRST, and a function body is NOT descended into, so nested `let`
/// bindings are not defs. Class/type bodies ARE descended into, but F# members
/// (`member_defn`) are not in any def-type set, so no Method nodes are emitted.
fn fsharp_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        if FSHARP_FUNC_KINDS.contains(&kind) {
            if let Some(name) = fsharp_func_name(source, node) {
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Function".into(),
                        name: name.clone(),
                        qualified_name: format!("{file_path}::Function::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            // Stop after a func match (F# not in the descend-into-func set) —
            // a local `let .. in ..` binding inside a function body is therefore
            // NOT a definition.
            continue;
        }
        if FSHARP_CLASS_KINDS.contains(&kind) {
            if let Some((name, label)) = fsharp_type_name_and_label(source, node) {
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: label.into(),
                        name: name.clone(),
                        qualified_name: format!("{file_path}::{label}::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            // Descend into the type body (class bodies are descended) so any
            // nested type / function is still visited.
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The definition NAME for one F# `function_or_value_defn`: find the
/// `function_declaration_left` (or `value_declaration_left`) child, then its
/// first *direct* `identifier` / `long_identifier` child. A pure value binding
/// (`value_declaration_left` wrapping an `identifier_pattern`) has no such
/// direct child → `None`, so it is skipped.
fn fsharp_func_name(source: &[u8], node: Node<'_>) -> Option<String> {
    let lhs = first_child_of_kind(node, "function_declaration_left")
        .or_else(|| first_child_of_kind(node, "value_declaration_left"))?;
    let name = first_child_of_kind(lhs, "identifier")
        .or_else(|| first_child_of_kind(lhs, "long_identifier"))?;
    Some(node_text(source, name).to_string())
}

/// The definition NAME and LABEL for one F# `type_definition` /
/// `exception_definition`: the name is the first descendant `type_name` node's
/// `type_name:` field (or its first `identifier` child). `exception_definition`
/// has no `type_name` descendant → `None` (skipped). The label is "Type" unless
/// the def has a `primary_constr_args` or `class_inherits_decl` descendant, in
/// which case it is an OOP "Class".
fn fsharp_type_name_and_label(source: &[u8], node: Node<'_>) -> Option<(String, &'static str)> {
    let tn = first_descendant_of_kind(node, "type_name")?;
    let id = tn
        .child_by_field_name("type_name")
        .or_else(|| first_child_of_kind(tn, "identifier"))?;
    let name = node_text(source, id).to_string();
    let mut label = "Type";
    if first_descendant_of_kind(node, "primary_constr_args").is_some()
        || first_descendant_of_kind(node, "class_inherits_decl").is_some()
    {
        label = "Class";
    }
    Some((name, label))
}

/// Emit CALLS edges from the per-file `Module` node to each applied function.
/// The callee is resolved by its final path segment; the indexer's resolver
/// links it same-file (direct `{file}::Function::{seg}` qname) or cross-file
/// (by unique name).
fn fsharp_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "application_expression" {
            if let Some(callee) = fsharp_callee_name(source, node) {
                if !callee.is_empty() {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Function::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": callee,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The callee NAME of an F# `application_expression`: the head is the first
/// named child; when it is a `long_identifier_or_op` / `long_identifier` /
/// `identifier`, the resolvable name is that head's final path segment
/// (`System.Math.sqrt` → `sqrt`, `add` → `add`).
fn fsharp_callee_name(source: &[u8], node: Node<'_>) -> Option<String> {
    let mut c = node.walk();
    let head = node.named_children(&mut c).next()?;
    match head.kind() {
        "long_identifier_or_op" | "long_identifier" | "identifier" => {
            Some(fsharp_ident_leaf(source, head))
        }
        _ => None,
    }
}

/// The final `identifier` segment of a (possibly dotted) F# identifier node.
/// `long_identifier_or_op`/`long_identifier` wrap a chain of `identifier`
/// children; the last one is the resolvable name. A bare `identifier` is
/// itself the leaf.
fn fsharp_ident_leaf(source: &[u8], node: Node<'_>) -> String {
    if node.kind() == "identifier" {
        return node_text(source, node).to_string();
    }
    let mut c = node.walk();
    let leaf = node
        .named_children(&mut c)
        .filter(|ch| ch.kind() == "identifier")
        .last();
    match leaf {
        Some(l) => node_text(source, l).to_string(),
        None => node_text(source, node).to_string(),
    }
}

/// Emit USAGE edges for F# (the generic reference walk): every `identifier`
/// reference that is NOT inside a call / import, and is not a definition name.
/// The source is the per-file `Module`; the reference resolves by name.
///
/// The usage resolver only registers Function/Method/Class/Interface labels, so
/// a reference to a plain "Type"-labeled definition (record / union / interface
/// / type-alias without a primary constructor or `inherit`) must never resolve
/// to an edge — including that definition's own name self-reference, a
/// same-named module, a field/annotation type, or a union case type. greppy's
/// shared resolver additionally accepts "Type" (its `USAGE_LABELS` is a
/// superset), and that resolver is off-limits here — so we apply the same screen
/// at the extraction site: skip any identifier whose text names a "Type"-labeled
/// definition of this file. "Class"-labeled names (Counter / Box / BaseShape)
/// are NOT in `type_names`, so their name usage is kept.
fn fsharp_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    type_names: &std::collections::HashSet<String>,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "identifier"
            && !is_inside_kind(node, &FSHARP_CALL_KINDS)
            && !is_inside_kind(node, &FSHARP_IMPORT_KINDS)
            && !is_definition_name(node)
        {
            let refname = node_text(source, node);
            if !refname.is_empty() && !fsharp_is_keyword(refname) && !type_names.contains(refname) {
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::__ref__::{refname}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "ref_name": refname }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// First *direct* named-or-anonymous child of `node` whose kind is `kind`
/// (non-recursive).
fn first_child_of_kind<'t>(node: Node<'t>, kind: &str) -> Option<Node<'t>> {
    let mut c = node.walk();
    let found = node.children(&mut c).find(|ch| ch.kind() == kind);
    found
}

/// First descendant of `node` (pre-order, `node` excluded) whose kind is
/// `kind`.
fn first_descendant_of_kind<'t>(node: Node<'t>, kind: &str) -> Option<Node<'t>> {
    let mut stack: Vec<Node<'t>> = Vec::new();
    // Push direct children in reverse so we pop them in source order.
    let mut c = node.walk();
    let children: Vec<Node<'t>> = node.children(&mut c).collect();
    for ch in children.into_iter().rev() {
        stack.push(ch);
    }
    while let Some(cur) = stack.pop() {
        if cur.kind() == kind {
            return Some(cur);
        }
        let mut cc = cur.walk();
        let kids: Vec<Node<'t>> = cur.children(&mut cc).collect();
        for ch in kids.into_iter().rev() {
            stack.push(ch);
        }
    }
    None
}

/// F# keyword filter for the USAGE pass. An `identifier` node only ever holds a
/// real identifier token in this grammar, but keywords occasionally surface as
/// `identifier` in member/type positions; filtering the common ones avoids
/// spurious references.
fn fsharp_is_keyword(name: &str) -> bool {
    matches!(
        name,
        "let"
            | "in"
            | "and"
            | "rec"
            | "fun"
            | "function"
            | "match"
            | "with"
            | "if"
            | "then"
            | "else"
            | "elif"
            | "begin"
            | "end"
            | "module"
            | "namespace"
            | "type"
            | "member"
            | "abstract"
            | "override"
            | "inherit"
            | "interface"
            | "open"
            | "import"
            | "static"
            | "mutable"
            | "new"
            | "of"
            | "val"
            | "do"
            | "done"
            | "for"
            | "to"
            | "downto"
            | "while"
            | "when"
            | "as"
            | "true"
            | "false"
            | "null"
            | "this"
            | "base"
            | "exception"
            | "raise"
            | "try"
            | "finally"
            | "return"
            | "yield"
            | "use"
            | "lazy"
    )
}

fn extract_scala(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Scala)
        .map_err(|e| greppy_core::Error::Parse(format!("compile scala queries: {e}")))?;
    // Base pass: the Scala DEFINITIONS query now captures ONLY
    // `function_definition`, so the spec engine emits exactly the "Method"
    // (owned by its enclosing class/object/trait via `owner_kinds`) or free
    // "Function" node, plus the CALLS pass and the IMPORTS pass. Everything the
    // uniform template does NOT model is added by the second pass below:
    //
    //   * the *type* declarations, labelled: `class_definition` /
    //     `object_definition` → "Class", `trait_definition` → "Interface",
    //     `enum_definition` → "Enum", `type_definition` → "Type" (the spec's own
    //     object→"Object" / trait→"Trait" labels are wrong here, which is why
    //     those kinds were dropped from the query);
    //   * the double-counted free "Function" node kept for every method (the
    //     re-walk visits the `template_body`, which is not recognised as a class
    //     body, so each `function_definition` inside is re-extracted as a free
    //     "Function" on top of the emitted "Method");
    //   * a "Variable" node for every class/object/trait-body and module-level
    //     `val` / `var`;
    //   * DEFINES_METHOD: each type → every method it owns;
    //   * the USAGE walk.
    let mut result = crate::spec::spec_extract(
        Language::Scala,
        &crate::spec::SCALA,
        queries,
        source,
        file_path,
    )?;

    let tree = crate::parse(Language::Scala, source)?;
    let root = tree.root_node();

    scala_defs_pass(source, root, file_path, &mut result);

    let file_module_qname = format!("{file_path}::__file__");
    scala_emit_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The Scala type-declaration kinds that become type/class nodes. Each is
/// labelled by `scala_type_label`.
const SCALA_TYPE_KINDS: [&str; 5] = [
    "class_definition",
    "object_definition",
    "trait_definition",
    "enum_definition",
    "type_definition",
];

/// The label for a Scala type declaration: trait → "Interface", enum → "Enum",
/// type → "Type", everything else (class / object) → "Class".
fn scala_type_label(kind: &str) -> &'static str {
    match kind {
        "trait_definition" => "Interface",
        "enum_definition" => "Enum",
        "type_definition" => "Type",
        _ => "Class",
    }
}

/// The `name:` (`identifier` / `type_identifier`) of a Scala definition node, or
/// `None`.
fn scala_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    node.child_by_field_name("name")
        .map(|n| node_text(source, n))
}

/// Second definitions pass over the Scala tree. Emits the type nodes with their
/// labels, the double-counted free `Function` node per method, the class/module
/// `Variable`s, and the DEFINES_METHOD edges. The spec base pass already emitted
/// the `Method` / free `Function` nodes and the CALLS / IMPORTS edges.
fn scala_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    // Module-level `val` / `var` (file-root direct children only).
    let mut rc = root.walk();
    for child in root.named_children(&mut rc) {
        if matches!(
            child.kind(),
            "val_definition" | "var_definition" | "val_declaration" | "var_declaration"
        ) {
            scala_emit_variable(source, child, file_path, result);
        }
    }

    // Every type declaration, wherever it sits (top-level or nested in another
    // template body). A full tree walk finds them all.
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if SCALA_TYPE_KINDS.contains(&node.kind()) {
            scala_emit_type(source, node, file_path, result);
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit the "Class"/"Interface"/"Enum"/"Type" node for one type declaration,
/// plus — for every `function_definition` that is a DIRECT member of its body —
/// the DEFINES_METHOD edge (type → Method) and the double-counted free
/// "Function" node, and for every direct-member `val`/`var` a "Variable" node.
/// Emits the type node, its methods and variables, plus the re-walk that
/// double-counts each method as a free Function.
fn scala_emit_type(source: &[u8], node: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let Some(name) = scala_name(source, node) else {
        return;
    };
    if name.is_empty() {
        return;
    }
    let label = scala_type_label(node.kind());
    result.nodes.push(ExtractedNode {
        label: label.into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::{label}::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });

    // Walk the DIRECT members of the type body (only the body's direct
    // children — a `val` inside a method body is not a class variable, and a
    // nested type is reached by the outer tree walk, not re-processed here).
    let Some(body) = node.child_by_field_name("body") else {
        return;
    };
    let type_qname = format!("{file_path}::{label}::{name}");
    let mut bc = body.walk();
    for member in body.named_children(&mut bc) {
        match member.kind() {
            // `function_definition` = a concrete `def name(...) = body`;
            // `function_declaration` = an abstract `def name(...)` (no body) in a
            // trait/abstract class. Both are a "Method" (spec pass) + a
            // double-counted free "Function" (re-walk) + a DEFINES_METHOD edge
            // from the owning type.
            "function_definition" | "function_declaration" => {
                let Some(m) = scala_name(source, member) else {
                    continue;
                };
                if m.is_empty() {
                    continue;
                }
                // The spec base pass emits the "Method" node for a concrete
                // `function_definition` (it is in `SCALA.defs`), but NOT for an
                // abstract `function_declaration` (which the spec does not
                // model). Both should be a "Method", so emit the abstract
                // method's "Method" node here (qname `{file}::{Owner}::{method}`,
                // matching the DEFINES_METHOD target and the spec's
                // concrete-method qname scheme).
                if member.kind() == "function_declaration" {
                    result.nodes.push(ExtractedNode {
                        label: "Method".into(),
                        name: m.to_string(),
                        qualified_name: format!("{file_path}::{name}::{m}"),
                        file_path: file_path.to_string(),
                        start_line: member.start_position().row as u32 + 1,
                        end_line: member.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
                // DEFINES_METHOD: type → the Method
                // (`{file}::{Owner}::{method}`).
                result.edges.push(ExtractedEdge {
                    edge_type: "DEFINES_METHOD".into(),
                    source_qualified_name: type_qname.clone(),
                    target_qualified_name: format!("{file_path}::{name}::{m}"),
                    file_path: file_path.to_string(),
                    line: member.start_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
                // The double-counted free "Function" node (re-walk). Its qname
                // carries NO owner segment, so two methods of the same name in
                // the same file collapse to one node via store dedup.
                result.nodes.push(ExtractedNode {
                    label: "Function".into(),
                    name: m.to_string(),
                    qualified_name: format!("{file_path}::Function::{m}"),
                    file_path: file_path.to_string(),
                    start_line: member.start_position().row as u32 + 1,
                    end_line: member.end_position().row as u32 + 1,
                    properties: serde_json::json!({}),
                });
            }
            "val_definition" | "var_definition" | "val_declaration" | "var_declaration" => {
                scala_emit_variable(source, member, file_path, result);
            }
            _ => {}
        }
    }
}

/// Emit a "Variable" node for a `val`/`var` definition or declaration. The name
/// is the `pattern:` field text (definitions) or the `name:` field
/// (declarations). The `_` placeholder and empty names are dropped. The qname
/// carries NO owner segment (the bare name), so same-named vals in one file
/// collapse to one node.
fn scala_emit_variable(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let name_node = node
        .child_by_field_name("pattern")
        .or_else(|| node.child_by_field_name("name"));
    let Some(name_node) = name_node else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() || name == "_" {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Variable".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Variable::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// USAGE pass for Scala. Every `identifier` / `type_identifier` reference emits
/// a USAGE edge unless it sits inside a call node (Scala call kinds:
/// `call_expression` / `generic_function` / `field_expression` /
/// `infix_expression` / `instance_expression` — those references are already
/// CALLS candidates), sits inside an import (`import_declaration` /
/// `using_directive`), is a definition *name*, or is a keyword. The `ref_name`
/// is resolved project-wide by the indexer, so the target qname is a
/// placeholder. The source is the nearest enclosing callable qname (a
/// `function_definition` owned by its type, or free) falling back to the
/// per-file Module node.
fn scala_emit_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let kind = node.kind();
    if matches!(kind, "identifier" | "type_identifier")
        && !is_inside_kind(
            node,
            &[
                "call_expression",
                "generic_function",
                "field_expression",
                "infix_expression",
                "instance_expression",
                "import_declaration",
                "using_directive",
            ],
        )
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_scala_usage_keyword(text) {
            let source_qname = scala_enclosing_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({
                    "ref_name": text,
                }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        scala_emit_usages(source, child, file_path, file_module_qname, result);
    }
}

/// The nearest enclosing Scala callable qname for `node`: the closest
/// `function_definition` ancestor, owned by its nearest enclosing type
/// (`{file}::{Owner}::{name}`) or free (`{file}::Function::{name}`). Returns
/// `None` at file / type scope (the caller substitutes the file Module qname).
fn scala_enclosing_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        // Both a concrete `function_definition` and an abstract
        // `function_declaration` are callables, so a reference inside an
        // abstract method's signature (e.g. the `Record` type annotation of
        // `def add(record: Record)`) attributes to that method — not the
        // enclosing file Module.
        if matches!(cur.kind(), "function_definition" | "function_declaration") {
            let name = scala_name(source, cur)?;
            return Some(match scala_owner_name(source, cur) {
                Some(owner) => format!("{file_path}::{owner}::{name}"),
                None => format!("{file_path}::Function::{name}"),
            });
        }
        p = cur.parent();
    }
    None
}

/// The owning type *name* for a `function_definition` (its nearest enclosing
/// `class_definition` / `object_definition` / `trait_definition` /
/// `enum_definition`), or `None` when the function is free. Mirrors the spec
/// engine's `enclosing_owner_name` (the same `owner_kinds`) so the Method qname
/// used as the USAGE source matches the spec-emitted Method node.
fn scala_owner_name<'a>(source: &'a [u8], func: Node<'_>) -> Option<&'a str> {
    let mut p = func.parent();
    while let Some(cur) = p {
        if matches!(
            cur.kind(),
            "class_definition" | "object_definition" | "trait_definition" | "enum_definition"
        ) {
            return scala_name(source, cur);
        }
        p = cur.parent();
    }
    None
}

// ===========================================================================
// Julia — bespoke pass
// ===========================================================================
//
// The generic registry/spec path cannot express Julia
// extraction, so this bespoke pass handles it directly:
//
//   * DEFINITIONS:
//       - `function_definition` (long form) → "Function". The name is the first
//         `identifier` reached by walking first-named-children through
//         `signature > call_expression`.
//       - `struct_definition` / `abstract_definition` / `primitive_definition`
//         → "Class" (all three are labelled "Class").
//         The name is the first `identifier` descendant of the `type_head`
//         child.
//       - short-form `f(x) = …` is parsed by tree-sitter-julia as a plain
//         `assignment` (NOT a `short_function_definition` node), so no
//         name is resolved for it and NO node is
//         emitted: short-form defs and module-level
//         `const`/`assignment` yield NO node (zero Variables here —
//         the `assignment` var arm needs a direct `identifier` child, which the
//         `call_expression`-LHS short form does not have).
//   * CALLS: every `call_expression` /
//     `broadcast_call_expression` callee identifier becomes a CALLS edge whose
//     SOURCE is the file's per-file Module node (`<file>::__file__`) — every
//     resolved call is attributed to the file module — and whose
//     `callee_name` the indexer resolves by name to a unique Function. This
//     also picks up each definition's own signature `call_expression`
//     (`function f(x)` → a self-call to `f`).
//   * USAGE: every `identifier` reference not
//     inside a call/import, not a definition-name, and not a keyword becomes a
//     USAGE edge keyed on `ref_name`, resolved by the indexer. The struct-name
//     identifier inside a `type_head` is NOT a `name:` field (Julia struct
//     nodes have no `name` field), so it surfaces as a self-USAGE
//     onto the struct's own Class node.
//
// IMPORTS are intentionally not emitted: the fixture's `using`/`import` name
// external packages with no in-repo Module target, so this pass resolves
// zero IMPORTS edges (honesty guard: cross-file/external imports are out of
// scope).
fn extract_julia(
    language: Language,
    _def: &crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    julia_defs_pass(source, root, file_path, &mut result);
    julia_calls_pass(source, root, file_path, &file_module_qname, &mut result);
    julia_usages_pass(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The Julia type-declaration kinds routed through the class path.
/// All three are labelled "Class".
const JULIA_CLASS_KINDS: [&str; 3] = [
    "struct_definition",
    "abstract_definition",
    "primitive_definition",
];

/// The Julia call kinds treated as calls.
const JULIA_CALL_KINDS: [&str; 2] = ["call_expression", "broadcast_call_expression"];

/// Resolve a Julia `function_definition`'s name: walk first-named-children
/// (through `signature` → `call_expression`) to the first `identifier` /
/// `operator_identifier`.
fn julia_func_name<'a>(source: &'a [u8], func: Node<'_>) -> Option<&'a str> {
    let mut current = func;
    for _ in 0..8 {
        let first = current.named_child(0)?;
        let k = first.kind();
        if k == "identifier" || k == "operator_identifier" {
            return Some(node_text(source, first));
        }
        current = first;
    }
    None
}

/// Resolve a Julia struct/abstract/primitive definition's name: the first
/// `identifier` descendant of its `type_head` child
/// (handles both `struct Foo` and `struct Foo <: Bar`).
fn julia_class_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let head = named_child_of_kinds(node, &["type_head"])?;
    julia_first_identifier(source, head)
}

/// First `identifier` node in a subtree (DFS, pre-order), or `None`.
fn julia_first_identifier<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    if node.kind() == "identifier" {
        return Some(node_text(source, node));
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        if let Some(found) = julia_first_identifier(source, child) {
            return Some(found);
        }
    }
    None
}

/// DEFINITIONS pass: emit "Function" for each long-form `function_definition`
/// and "Class" for each struct/abstract/primitive definition, walking the whole
/// tree (nested defs are reached too). Short-form `f(x)=…`
/// (`assignment`) and module-level `const`/`assignment` emit nothing on
/// this grammar.
fn julia_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "function_definition" => {
                if let Some(name) = julia_func_name(source, node) {
                    if !name.is_empty() {
                        result.nodes.push(ExtractedNode {
                            label: "Function".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::Function::{name}"),
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                }
            }
            k if JULIA_CLASS_KINDS.contains(&k) => {
                if let Some(name) = julia_class_name(source, node) {
                    if !name.is_empty() {
                        result.nodes.push(ExtractedNode {
                            label: "Class".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::Class::{name}"),
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                }
            }
            _ => {}
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// CALLS pass: one CALLS edge per `call_expression` / `broadcast_call_expression`
/// callee identifier. Source is the file Module (`<file>::__file__`);
/// the indexer resolves `callee_name` to a
/// unique Function and dedups by (source, target, type).
fn julia_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if JULIA_CALL_KINDS.contains(&node.kind()) {
            if let Some(callee) = node.named_child(0) {
                if callee.kind() == "identifier" || callee.kind() == "operator_identifier" {
                    let text = node_text(source, callee);
                    if !text.is_empty() {
                        result.edges.push(ExtractedEdge {
                            edge_type: "CALLS".into(),
                            source_qualified_name: file_module_qname.to_string(),
                            target_qualified_name: format!("{file_path}::Function::{text}"),
                            file_path: file_path.to_string(),
                            line: callee.start_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "callee_text": text,
                                "callee_name": text,
                            }),
                        });
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// USAGE pass for Julia. Every
/// `identifier` reference that is NOT inside a call/import, NOT a definition
/// name, and NOT a keyword emits a USAGE edge keyed on `ref_name`. The source
/// is the nearest enclosing `function_definition` qname, falling back to the
/// file Module.
fn julia_usages_pass(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if node.kind() == "identifier"
        && !is_inside_kind(
            node,
            &[
                "call_expression",
                "broadcast_call_expression",
                "import_statement",
                "using_statement",
                "export_statement",
                "selected_import",
            ],
        )
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_julia_usage_keyword(text) {
            let source_qname = julia_enclosing_func_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({ "ref_name": text }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        julia_usages_pass(source, child, file_path, file_module_qname, result);
    }
}

/// Nearest enclosing Julia `function_definition` qname for `node`
/// (`<file>::Function::<name>`). `None` at
/// module / struct scope (the caller substitutes the file Module qname).
fn julia_enclosing_func_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "function_definition" {
            let name = julia_func_name(source, cur)?;
            return Some(format!("{file_path}::Function::{name}"));
        }
        p = cur.parent();
    }
    None
}

/// Julia keyword / literal filter — the generic keyword table.
/// A reference whose text is one of these never emits a usage.
fn is_julia_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "nil"
            | "None"
            | "undefined"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "struct"
            | "enum"
            | "interface"
            | "trait"
            | "impl"
            | "import"
            | "export"
            | "package"
            | "module"
            | "use"
            | "require"
            | "include"
            | "new"
            | "delete"
            | "this"
            | "self"
            | "super"
            | "public"
            | "private"
            | "protected"
            | "static"
            | "const"
            | "var"
            | "let"
            | "function"
            | "def"
            | "fn"
            | "func"
            | "fun"
            | "proc"
            | "sub"
            | "method"
            | "async"
            | "await"
            | "yield"
    )
}

/// Scala keyword / literal filter — the generic keyword table, the same
/// one used by the other data-path languages. A reference whose text
/// is one of these never emits a usage.
fn is_scala_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "nil"
            | "None"
            | "undefined"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "struct"
            | "enum"
            | "interface"
            | "trait"
            | "impl"
            | "import"
            | "export"
            | "package"
            | "module"
            | "use"
            | "require"
            | "include"
            | "new"
            | "delete"
            | "this"
            | "self"
            | "super"
            | "public"
            | "private"
            | "protected"
            | "static"
            | "const"
            | "var"
            | "let"
            | "val"
            | "function"
            | "def"
            | "fn"
            | "func"
            | "fun"
            | "proc"
            | "sub"
            | "method"
            | "async"
            | "await"
            | "yield"
    )
}

// ---------------------------------------------------------------------------
// Haskell (registry language onboarded via `crate::langs::haskell`).
//
// The generic spec path (`spec_extract`) already emits everything the
// function pass needs: a free "Function" node per top-level
// `function` / `bind` (and — because the query walks the whole tree — per
// `function` inside a `class` / `instance` body), the CALLS pass over
// `apply` / `infix` call nodes, and the
// File→DEFINES edges the structural pass auto-adds. `extract_haskell` layers
// the two facets the template cannot express:
//
//   * a "Class" node for every `class` (typeclass), `data_type` and `newtype`.
//     The type-declaration set is exactly `{class, data_type, newtype}`, and
//     all three are labelled "Class" (none matches the
//     Interface/Enum/Type kinds). `type_synomym` (`type X = …`) is in NO type
//     list, so it is deliberately NOT emitted.
//   * an IMPORTS pass over explicit import lists, keyed by each imported name;
//   * the `pass_usages` USAGE walk: every
//     `variable` / `constructor` reference that is not inside a runtime call
//     (`apply` / `infix`) or import (`import` / `instance`), is not a
//     definition name, and is not a keyword. Constructor applies in a function
//     LHS pattern are usages, not calls.
// ---------------------------------------------------------------------------

/// The Haskell type-declaration kinds routed through the class path.
/// All three are labelled "Class"
/// (Haskell has no Interface/Enum/Type kind).
const HASKELL_TYPE_KINDS: [&str; 3] = ["class", "data_type", "newtype"];

/// The Haskell call kinds — a reference inside one of these is a CALLS
/// candidate, so the usage pass skips it.
const HASKELL_CALL_KINDS: [&str; 2] = ["apply", "infix"];

/// The Haskell import kinds — a reference inside one of these is skipped by the
/// usage walk. Note `instance` is treated as an import
/// container, so references inside an `instance` body never emit usages.
const HASKELL_IMPORT_KINDS: [&str; 2] = ["import", "instance"];

fn extract_haskell(
    language: Language,
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    // Haskell's generic spec path is deliberately NOT used: the tree-sitter
    // def/call queries visit the *whole* tree, so they over-count `bind`s
    // (every `where`-bound local binding) and under-count calls (they miss
    // `infix` operators and `constructor` applies). Instead this runs three
    // explicit walk passes:
    //   * defs: Function per top-level `function`/
    //     `bind` and per class-body `signature`/`function`, but NOT `where`-bound
    //     locals (a function is not descended into);
    //     "Class" per `class`/`data_type`/`newtype`.
    //   * calls: one call candidate per runtime `apply`
    //     (callee = first child, if `variable`/`constructor`) and per `infix`
    //     (callee = the `operator:` field); LHS constructor patterns are skipped.
    //   * imports: one IMPORTS edge per explicit import-list name.
    //   * usages: one USAGE per `variable`/
    //     `constructor` reference not inside a runtime call/import, not a def
    //     name; LHS constructor patterns are included.
    // The Module/File/Folder/Project structural nodes and the File→DEFINES /
    // CONTAINS edges are added by the indexer's shared structural pass.
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();

    haskell_walk_defs(source, root, file_path, &mut result);
    haskell_walk_calls(source, root, file_path, &mut result);

    let file_module_qname = format!("{file_path}::__file__");
    haskell_imports_pass(d, source, root, file_path, &file_module_qname, &mut result)?;
    haskell_emit_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The Haskell defs walk (an explicit
/// stack, no recursion into function bodies). For each node:
///   * a `function` / `bind` → emit a free "Function" node, then STOP (do not
///     descend — this is why `where`-bound locals are not extracted);
///   * a `signature` → no node, STOP;
///   * a `class` / `data_type` / `newtype` → emit a "Class" node, then descend
///     into its children so class-body `signature`/`function` decls are reached
///     (a Haskell `class_declarations` body is not a recognised
///     body-kind, so all children are pushed);
///   * anything else → descend into all children.
fn haskell_walk_defs(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        // {function, signature, bind}: extract (no-op for
        // `signature`) then STOP — do not descend into the body.
        if matches!(kind, "function" | "signature" | "bind") {
            if kind != "signature" {
                haskell_emit_function(source, node, file_path, result);
            }
            continue;
        }
        // class_types = {class, data_type, newtype}: emit "Class", then descend
        // into the body (class methods become free Functions on the next pops).
        if HASKELL_TYPE_KINDS.contains(&kind) {
            haskell_emit_type(source, node, file_path, result);
        }
        // Push children in reverse so they pop in source order (cosmetic; counts
        // are order-independent). Uses ALL children (named + unnamed),
        // though only named nodes ever match.
        let n = node.child_count();
        for i in (0..n).rev() {
            if let Some(child) = node.child(i) {
                stack.push(child);
            }
        }
    }
}

/// Emit a free "Function" node for a `function` / `bind` def node. The name is
/// the `name:` field, or — for a multi-clause
/// `function` whose grammar shape puts the name on the first named `variable`
/// child — that child. Empty names are dropped.
fn haskell_emit_function(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let name_node = node
        .child_by_field_name("name")
        .or_else(|| haskell_first_variable_child(node));
    let Some(name_node) = name_node else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Function".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Function::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// The first named child if it is a `variable` /
/// `name`, else that child's own first `variable` / `name`.
fn haskell_first_variable_child(node: Node<'_>) -> Option<Node<'_>> {
    let head = node.named_child(0)?;
    if matches!(head.kind(), "variable" | "name") {
        return Some(head);
    }
    let inner = head.named_child(0)?;
    if matches!(inner.kind(), "variable" | "name") {
        return Some(inner);
    }
    None
}

/// The Haskell calls walk: a full-tree
/// walk that, at every `apply` / `infix` node, emits one CALLS candidate.
///   * `apply` → callee = the FIRST child, if it is a `variable` / `constructor`;
///     a nested `apply`
///     first child yields no callee here (its own visit emits the inner call).
///   * `infix` → callee = the `operator:` field text.
/// Keyword callees are dropped. The source is the nearest
/// enclosing function qname (or the per-file module qname).
fn haskell_walk_calls(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let file_module_qname = format!("{file_path}::__file__");
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let callee = if haskell_is_function_lhs_pattern(node) {
            None
        } else {
            haskell_call_callee(source, node)
        };
        if let Some(callee) = callee {
            if !callee.is_empty() && !is_haskell_usage_keyword(callee) {
                let source_qname = haskell_enclosing_qname(source, node, file_path)
                    .unwrap_or_else(|| file_module_qname.clone());
                result.edges.push(ExtractedEdge {
                    edge_type: "CALLS".into(),
                    source_qualified_name: source_qname,
                    target_qualified_name: format!("{file_path}::__callee__::{callee}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "callee_name": callee,
                    }),
                });
            }
        }
        let n = node.child_count();
        for i in (0..n).rev() {
            if let Some(child) = node.child(i) {
                stack.push(child);
            }
        }
    }
}

/// The callee text for one call node, or `None` if the
/// node is not an `apply` / `infix` or its callee position is not an
/// identifier-like node.
fn haskell_call_callee<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    match node.kind() {
        "apply" => {
            let callee = node.child(0)?;
            if matches!(callee.kind(), "variable" | "constructor" | "name") {
                Some(node_text(source, callee))
            } else {
                None
            }
        }
        "infix" => {
            let op = node.child_by_field_name("operator")?;
            Some(node_text(source, op))
        }
        _ => None,
    }
}

/// Whether `node` is inside the `patterns:` field of an enclosing function.
/// Constructor application syntax on the left-hand side is destructuring, not
/// a runtime call, and is instead classified by the USAGE pass.
fn haskell_is_function_lhs_pattern(node: Node<'_>) -> bool {
    let mut current = Some(node);
    let mut patterns = None;
    while let Some(candidate) = current {
        if candidate.kind() == "patterns" {
            patterns = Some(candidate);
        }
        if candidate.kind() == "function" {
            return patterns.is_some_and(|pattern_node| {
                candidate
                    .child_by_field_name("patterns")
                    .is_some_and(|field| field.id() == pattern_node.id())
            });
        }
        current = candidate.parent();
    }
    false
}

/// Emit the "Class" node for one `class` / `data_type` / `newtype`. The name is
/// the `name:` field (a `name` node in tree-sitter-haskell).
/// Empty names are dropped.
fn haskell_emit_type(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Class".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Class::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// Emit one file-sourced IMPORTS edge per explicit `import_name`. The provider
/// query maps `import -> module + import_list -> import_name`; resolution keys
/// on the imported symbol while retaining the module path for disambiguation.
fn haskell_imports_pass(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let language = (d.grammar)();
    let query = tree_sitter::Query::new(&language, d.import_query)
        .map_err(|e| greppy_core::Error::Parse(format!("compile haskell imports query: {e}")))?;
    let module_capture = query
        .capture_index_for_name("module")
        .ok_or_else(|| greppy_core::Error::Parse("haskell imports query lacks @module".into()))?;
    let imported_capture = query
        .capture_index_for_name("imported")
        .ok_or_else(|| greppy_core::Error::Parse("haskell imports query lacks @imported".into()))?;
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, source);
    while let Some(query_match) = matches.next() {
        let mut module = None;
        let mut imported = None;
        for capture in query_match.captures {
            if capture.index == module_capture {
                module = Some(node_text(source, capture.node));
            } else if capture.index == imported_capture {
                imported = capture
                    .node
                    .child_by_field_name("variable")
                    .or_else(|| capture.node.child_by_field_name("type"))
                    .or_else(|| capture.node.child_by_field_name("operator"))
                    .map(|name| node_text(source, name));
            }
        }
        let (Some(path), Some(imported_name)) = (module, imported) else {
            continue;
        };
        if path.is_empty() || imported_name.is_empty() {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "IMPORTS".into(),
            source_qualified_name: file_module_qname.to_string(),
            target_qualified_name: format!("{file_path}::__import__::{path}::{imported_name}"),
            file_path: file_path.to_string(),
            line: query_match.captures[0].node.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "path": path,
                "imported_name": imported_name,
                "original_name": imported_name,
                "glob": false,
            }),
        });
    }
    Ok(())
}

/// USAGE pass for Haskell over reference nodes
/// (`variable` / `constructor`). Every such
/// reference emits a USAGE edge unless it sits inside a call node
/// (`apply` / `infix`), inside an import (`import` / `instance`), is a
/// definition *name*, or is a keyword. The `ref_name` is resolved project-wide
/// by the indexer, so the target qname is a placeholder. The source is the
/// nearest enclosing callable qname (a `function` / `bind`, resolved free as
/// `{file}::Function::{name}`) falling back to the per-file module qname.
fn haskell_emit_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let kind = node.kind();
    if matches!(kind, "variable" | "constructor")
        && (!haskell_is_inside(node, &HASKELL_CALL_KINDS) || haskell_is_function_lhs_pattern(node))
        && !haskell_is_inside(node, &HASKELL_IMPORT_KINDS)
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_haskell_usage_keyword(text) {
            let source_qname = haskell_enclosing_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({
                    "ref_name": text,
                }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        haskell_emit_usages(source, child, file_path, file_module_qname, result);
    }
}

/// True if `node` has an ancestor whose kind is in `kinds`, within a
/// `MAX_PARENT_DEPTH` of 10. A dedicated helper — rather
/// than the shared `is_inside_kind` (depth 12) — keeps this bound so the
/// USAGE count is stable.
fn haskell_is_inside(node: Node<'_>, kinds: &[&str]) -> bool {
    const MAX_PARENT_DEPTH: usize = 10;
    let mut cur = node.parent();
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= MAX_PARENT_DEPTH {
            break;
        }
        if kinds.contains(&n.kind()) {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// The qname of the enclosing Haskell definition for `node`, resolved to
/// `{file}::Function::{name}` — but ONLY when that def is a genuine TOP-LEVEL
/// `function` / `bind` (a direct child of the file's `declarations`). Otherwise
/// returns `None` and the caller substitutes the per-file module qname.
///
/// The enclosing-function qn attributes a
/// reference to the module (not the def) whenever it is NOT directly in a
/// top-level body:
///   * `where` / `let` bindings (`local_binds`): every `digestOf`/`unDigest`
///     call in `Index.hs`'s `where` clauses is a single `Index → …` edge, not
///     one per call site; the `let cfg = defaultConfig` reference in `main`
///     attributes to the Main module.
///   * class-/instance-body methods (`class_declarations` /
///     `instance_declarations`): the `name` call in the `describe` default
///     method attributes to the module (`Types → name`), not `describe → name`.
/// Only a `function`/`bind` whose parent is `declarations` is an enclosing
/// scope. This also keeps the source a real node — the defs walk never descends
/// into a body, so a `where`-bound or class-body inner def is not emitted and
/// attributing to it would dangle.
fn haskell_enclosing_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    // The nearest `function` / `bind` ancestor.
    let mut def: Option<Node<'_>> = None;
    let mut p = node.parent();
    while let Some(cur) = p {
        if matches!(cur.kind(), "function" | "bind") {
            def = Some(cur);
            break;
        }
        p = cur.parent();
    }
    let def = def?;
    // Only a TOP-LEVEL def (direct child of `declarations`) is an enclosing
    // scope; a `where`/`let`-bound def (parent `local_binds`) or a class/
    // instance-body method (parent `class_declarations`/`instance_declarations`)
    // attributes to the module instead.
    if def.parent().map(|p| p.kind()) != Some("declarations") {
        return None;
    }
    let name_node = def
        .child_by_field_name("name")
        .or_else(|| haskell_first_variable_child(def))?;
    let name = node_text(source, name_node);
    if name.is_empty() {
        return None;
    }
    Some(format!("{file_path}::Function::{name}"))
}

/// Haskell keyword / literal filter — the generic keyword table,
/// the same list used by the other data-path languages. A reference
/// whose text is one of these never emits a usage.
fn is_haskell_usage_keyword(name: &str) -> bool {
    is_scala_usage_keyword(name)
}

// ===========================================================================
// Dart — registry language with a bespoke pass.
//
// The base spec engine (`DART_SPEC` in `langs/dart.rs`) captures only the
// `function_signature` def-nodes, so it emits `Method` (class members) and
// `Function` (free) nodes plus the per-file `Module`. It does NOT emit the
// `Class` / `Enum` type nodes, the enum-constant `Variable`s, the
// `DEFINES_METHOD` edges, correctly attributed `CALLS` / `USAGE` edges, or
// relative-file `IMPORTS` edges. `extract_dart` runs the base pass and adds
// those bespoke passes.
//
// Grammar note: this crate uses `tree-sitter-dart` 0.2, so a class is a
// `class_declaration` node. The emitted *counts / labels* are grammar-independent,
// reached with THIS grammar's kinds.

/// Dart type-declaration kinds and their labels. In
/// this grammar a class is `class_declaration` and an enum is
/// `enum_declaration`. A `mixin_declaration` is NOT counted
/// (its members are not extracted either), and a `type_alias` (typedef) does
/// not surface as a node, so both are excluded here.
fn dart_type_label(kind: &str) -> Option<&'static str> {
    match kind {
        "class_declaration" => Some("Class"),
        "enum_declaration" => Some("Enum"),
        _ => None,
    }
}

/// The `name:` identifier text of a Dart declaration node, or `None`.
fn dart_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    node.child_by_field_name("name")
        .map(|n| node_text(source, n))
}

fn extract_dart(
    language: Language,
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    // Base pass: `Method` (class/mixin members) + free `Function` + per-file
    // `Module` nodes. The shared CALLS pass cannot cross the signature/body
    // sibling boundary and the registry import query is empty, so bespoke
    // passes below supply calls and relative imports.
    let queries = d
        .compiled_queries()
        .map_err(|e| greppy_core::Error::Parse(format!("compile {} queries: {e}", d.name)))?;
    let mut result = crate::spec::spec_extract(language, d.spec, queries, source, file_path)?;

    let tree = crate::parse(language, source)?;
    let root = tree.root_node();

    // (a) Drop the Method nodes the base pass created for `mixin_declaration`
    // members — C does not extract mixin members. We identify them by their
    // owner: the base pass qualifies a mixin method as `{file}::{Mixin}::{m}`.
    // Rather than string-match owners, re-walk the tree and collect the set of
    // mixin-member method qnames to remove.
    let mixin_method_qnames = dart_mixin_method_qnames(source, root, file_path);
    if !mixin_method_qnames.is_empty() {
        result
            .nodes
            .retain(|n| !(n.label == "Method" && mixin_method_qnames.contains(&n.qualified_name)));
    }

    // (b) Type nodes (Class/Enum) + enum-constant Variables + DEFINES_METHOD.
    dart_defs_pass(source, root, file_path, &mut result);

    // (c) Calls and relative imports. Dart places a declaration's signature
    // and body beside each other, so the shared ancestor-only call pass cannot
    // attribute body callsites to their Function/Method node.
    dart_emit_calls(source, root, file_path, &mut result);
    dart_emit_relative_imports(source, root, file_path, &mut result);

    // The enum names declared in this file. A USAGE never resolves
    // to an `Enum` node (every usage resolves to
    // Class / Method / Function / Variable, none to
    // `Enum:Category` / `Enum:Severity`). A reference to a *same-file* enum is
    // the one such case the parser can recognise without cross-file knowledge,
    // so it is filtered out of the USAGE walk below.
    let local_enums: Vec<String> = result
        .nodes
        .iter()
        .filter(|n| n.label == "Enum")
        .map(|n| n.name.clone())
        .collect();

    // (d) USAGE walk for Dart.
    let file_module_qname = format!("{file_path}::__file__");
    dart_emit_usages(
        source,
        root,
        file_path,
        &file_module_qname,
        &local_enums,
        &mut result,
    );

    Ok(result)
}

/// Collect the qnames the base spec pass assigned to methods that are members
/// of a `mixin_declaration` (so they can be removed — C does not extract mixin
/// members). The base pass qualifies such a method as `{file}::{Mixin}::{m}`.
fn dart_mixin_method_qnames(source: &[u8], root: Node<'_>, file_path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "mixin_declaration" {
            let Some(mixin_name) = dart_name(source, node) else {
                continue;
            };
            if let Some(body) = node.child_by_field_name("body") {
                let mut bc = body.walk();
                for member in body.named_children(&mut bc) {
                    if member.kind() != "class_member" {
                        continue;
                    }
                    // class_member > method_declaration > method_signature >
                    // function_signature name: identifier
                    if let Some(m) = dart_member_method_name(source, member) {
                        out.push(format!("{file_path}::{mixin_name}::{m}"));
                    }
                }
            }
            continue;
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
    out
}

/// The method name of a `class_member` that wraps a `method_declaration` whose
/// signature carries a plain `function_signature` (i.e. what the base spec pass
/// counts as a Method). Getter/setter/operator/constructor members carry a
/// different signature kind and yield `None`.
fn dart_member_method_name<'a>(source: &'a [u8], class_member: Node<'_>) -> Option<&'a str> {
    let mut mc = class_member.walk();
    for md in class_member.named_children(&mut mc) {
        if md.kind() != "method_declaration" {
            continue;
        }
        let sig = md.child_by_field_name("signature")?;
        // method_signature > function_signature name: identifier
        let mut sc = sig.walk();
        for fs in sig.named_children(&mut sc) {
            if fs.kind() == "function_signature" {
                if let Some(nm) = fs.child_by_field_name("name") {
                    return Some(node_text(source, nm));
                }
            }
        }
    }
    None
}

/// Second definitions pass: emit `Class`/`Enum` type nodes,
/// the enum-constant `Variable`s, and the
/// `DEFINES_METHOD` edges (type → each Method it owns). The base spec pass
/// already emitted the `Method` / free `Function` / `Module` nodes.
fn dart_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "top_level_variable_declaration" {
            dart_emit_top_level_variables(source, node, file_path, result);
        }
        if let Some(label) = dart_type_label(node.kind()) {
            dart_emit_type(source, node, label, file_path, result);
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit `Variable` nodes for initialized top-level Dart declarations. These
/// definitions are cross-file USAGE targets (for example imported constants).
fn dart_emit_top_level_variables(
    source: &[u8],
    declaration: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![declaration];
    while let Some(node) = stack.pop() {
        if matches!(
            node.kind(),
            "initialized_variable_definition" | "static_final_declaration"
        ) {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(source, name_node);
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Variable".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Variable::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            continue;
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            stack.push(child);
        }
    }
}

/// Emit the `Class`/`Enum` node for one type declaration. For a `class`, also
/// emit a `DEFINES_METHOD` edge to every direct-member method. For an `enum`,
/// emit a `Variable` node for every enum constant (C extracts enum constants as
/// Variables owned by the enum, qname `{file}::{Enum}::{const}`).
fn dart_emit_type(
    source: &[u8],
    node: Node<'_>,
    label: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let Some(name) = dart_name(source, node) else {
        return;
    };
    if name.is_empty() {
        return;
    }
    let type_qname = format!("{file_path}::{label}::{name}");
    result.nodes.push(ExtractedNode {
        label: label.into(),
        name: name.to_string(),
        qualified_name: type_qname.clone(),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });

    let Some(body) = node.child_by_field_name("body") else {
        return;
    };

    if label == "Enum" {
        // enum_body > enum_constant name: identifier
        let mut bc = body.walk();
        for member in body.named_children(&mut bc) {
            if member.kind() != "enum_constant" {
                continue;
            }
            let Some(cname) = dart_name(source, member) else {
                continue;
            };
            if cname.is_empty() {
                continue;
            }
            result.nodes.push(ExtractedNode {
                label: "Variable".into(),
                name: cname.to_string(),
                qualified_name: format!("{file_path}::{name}::{cname}"),
                file_path: file_path.to_string(),
                start_line: member.start_position().row as u32 + 1,
                end_line: member.end_position().row as u32 + 1,
                properties: serde_json::json!({}),
            });
        }
        return;
    }

    // Class: DEFINES_METHOD → each direct-member method (matching the base
    // pass's Method qname `{file}::{Class}::{method}`).
    let mut bc = body.walk();
    for member in body.named_children(&mut bc) {
        if member.kind() != "class_member" {
            continue;
        }
        if let Some(m) = dart_member_method_name(source, member) {
            result.edges.push(ExtractedEdge {
                edge_type: "DEFINES_METHOD".into(),
                source_qualified_name: type_qname.clone(),
                target_qualified_name: format!("{file_path}::{name}::{m}"),
                file_path: file_path.to_string(),
                line: member.start_position().row as u32 + 1,
                properties: serde_json::json!({}),
            });
        }
    }
}

/// Emit Dart CALLS edges with a source reconstructed from the enclosing
/// declaration. The grammar makes `function_signature` and `function_body`
/// siblings, so the shared spec walk sees the callee but cannot find its caller.
fn dart_emit_calls(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            stack.push(child);
        }
        if node.kind() != "call_expression" {
            continue;
        }
        let Some(function) = node.child_by_field_name("function") else {
            continue;
        };
        let Some(callee) = dart_call_name_node(function) else {
            continue;
        };
        let name = node_text(source, callee);
        if name.is_empty() {
            continue;
        }
        let Some(source_qname) = dart_enclosing_qname(source, node, file_path) else {
            continue;
        };
        if result.edges.iter().any(|edge| {
            edge.edge_type == "CALLS"
                && edge.line == callee.start_position().row as u32 + 1
                && edge.source_qualified_name == source_qname
                && edge
                    .properties
                    .get("callee_name")
                    .and_then(|value| value.as_str())
                    == Some(name)
        }) {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "CALLS".into(),
            source_qualified_name: source_qname,
            target_qualified_name: format!("{file_path}::Function::{name}"),
            file_path: file_path.to_string(),
            line: callee.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "callee_text": name,
                "callee_name": name,
            }),
        });
    }
}

/// Return the final statically named callee from a direct or member call.
fn dart_call_name_node(function: Node<'_>) -> Option<Node<'_>> {
    match function.kind() {
        "identifier" => Some(function),
        "member_expression" | "null_aware_member_expression" => {
            function.child_by_field_name("property")
        }
        _ => None,
    }
}

/// Emit one filesystem import candidate for each relative Dart `import` URI.
/// The indexer resolves it lexically against the importing file and fans the
/// import out to the target library's public definition nodes.
fn dart_emit_relative_imports(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            stack.push(child);
        }
        if node.kind() != "import_specification" {
            continue;
        }
        let Some(path) = dart_import_path(source, node) else {
            continue;
        };
        if path.contains(':') || path.contains('$') {
            continue;
        }
        let imported_name = path.rsplit('/').next().unwrap_or(path);
        result.edges.push(ExtractedEdge {
            edge_type: "IMPORTS".into(),
            source_qualified_name: format!("{file_path}::__file__"),
            target_qualified_name: format!("{file_path}::Import::{path}"),
            file_path: file_path.to_string(),
            line: node.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "path": path,
                "imported_name": imported_name,
                "original_name": imported_name,
                "glob": true,
                "filesystem_module_import": true,
                "dart_relative_import": true,
            }),
        });
    }
}

fn dart_import_path<'a>(source: &'a [u8], import: Node<'_>) -> Option<&'a str> {
    let uri = import.child_by_field_name("uri")?;
    let text = node_text(source, uri).trim();
    let text = text.strip_prefix('r').unwrap_or(text);
    text.strip_prefix('\'')
        .and_then(|inner| inner.strip_suffix('\''))
        .or_else(|| {
            text.strip_prefix('"')
                .and_then(|inner| inner.strip_suffix('"'))
        })
        .filter(|path| !path.is_empty())
}

/// USAGE pass for Dart. Every
/// `identifier` / `type_identifier` reference emits a USAGE edge unless it sits
/// inside a call node (this grammar's call/
/// constructor kinds), sits inside an import, is a definition *name*, or is a
/// keyword. The indexer resolves `ref_name` project-wide (so the target qname
/// is a `__ref__` placeholder). The source is the nearest enclosing callable
/// qname (a class method or free function), falling back to the per-file
/// Module node.
fn dart_emit_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    local_enums: &[String],
    result: &mut ExtractionResult,
) {
    let kind = node.kind();
    if matches!(kind, "identifier" | "type_identifier")
        && !is_inside_kind(node, DART_CALL_SKIP_KINDS)
        && !is_inside_kind(node, DART_IMPORT_KINDS)
        && !is_definition_name(node)
        && !dart_is_pattern_type_qualifier(node)
    {
        let text = node_text(source, node);
        if !text.is_empty()
            && !is_dart_usage_keyword(text)
            && !local_enums.iter().any(|e| e == text)
        {
            let source_qname = dart_enclosing_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({ "ref_name": text }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        dart_emit_usages(
            source,
            child,
            file_path,
            file_module_qname,
            local_enums,
            result,
        );
    }
}

/// Call/constructor node kinds a Dart reference must NOT be inside to count as a
/// USAGE (those references are already CALLS candidates). The call types are
/// `{selector, new_expression}`; in THIS grammar the equivalent invocation
/// wrappers are `member_expression` (the `obj.method` / `.field` access) and
/// `new_expression`. Call arguments remain ordinary value/type references —
/// for example `Widget(answer + HELPER_VALUE)` must retain the constant usage.
/// A bare direct call `foo()` leaves its callee `foo` as a plain `identifier`
/// under `call_expression` (NOT inside either wrapper), so direct-call callees
/// still count as usages while `obj.method()` selectors do not.
const DART_CALL_SKIP_KINDS: &[&str] = &["member_expression", "new_expression"];

/// True if `node` is the *type qualifier* of a `constant_pattern` — the `X` in a
/// `case X.member:` enum-value pattern (`constant_pattern > identifier(X) . name`).
/// The meaningful reference in such a pattern is the constant member, not the
/// enum type, which is why a USAGE is emitted for the member and NOT
/// for the qualifier. (In this grammar a `constant_pattern` is a flat sequence
/// `identifier '.' identifier`; the qualifier is the first named child.)
fn dart_is_pattern_type_qualifier(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    if parent.kind() != "constant_pattern" {
        return false;
    }
    parent
        .named_child(0)
        .is_some_and(|first| first.id() == node.id())
}

/// Import node kinds a Dart reference must not be inside.
const DART_IMPORT_KINDS: &[&str] = &["import_or_export", "import_specification", "library_import"];

/// The nearest enclosing Dart callable qname for `node`: the closest
/// `function_signature` (owned by its enclosing class → `{file}::{Class}::{m}`,
/// or free → `{file}::Function::{name}`). Returns `None` at file / type scope
/// (the caller substitutes the file Module qname).
fn dart_enclosing_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut current = node.parent();
    while let Some(candidate) = current {
        let signature = match candidate.kind() {
            "function_signature" => Some(candidate),
            "function_declaration" => candidate.child_by_field_name("signature"),
            "method_declaration" => candidate
                .child_by_field_name("signature")
                .and_then(dart_function_signature_child),
            _ => None,
        };
        if let Some(signature) = signature {
            let name = signature
                .child_by_field_name("name")
                .map(|name| node_text(source, name))?;
            return Some(match dart_owner_type_name(source, candidate) {
                Some(owner) => format!("{file_path}::{owner}::{name}"),
                None => format!("{file_path}::Function::{name}"),
            });
        }
        current = candidate.parent();
    }
    None
}

fn dart_function_signature_child(node: Node<'_>) -> Option<Node<'_>> {
    if node.kind() == "function_signature" {
        return Some(node);
    }
    let mut cursor = node.walk();
    let signature = node
        .named_children(&mut cursor)
        .find(|child| child.kind() == "function_signature");
    signature
}

/// The owning type *name* for a `function_signature` (its nearest enclosing
/// `class_declaration`), or `None` when free or inside a mixin (mixin members
/// are not extracted, so a reference in one attributes to the file Module).
fn dart_owner_type_name<'a>(source: &'a [u8], func: Node<'_>) -> Option<&'a str> {
    let mut p = func.parent();
    while let Some(cur) = p {
        match cur.kind() {
            "class_declaration" => return dart_name(source, cur),
            "mixin_declaration" => return None,
            _ => {}
        }
        p = cur.parent();
    }
    None
}

/// Dart keyword / literal filter — the generic keyword table.
/// A reference whose text is one of these never emits a
/// usage.
fn is_dart_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "enum"
            | "mixin"
            | "extends"
            | "implements"
            | "with"
            | "abstract"
            | "interface"
            | "import"
            | "export"
            | "part"
            | "library"
            | "new"
            | "this"
            | "super"
            | "static"
            | "const"
            | "final"
            | "var"
            | "late"
            | "get"
            | "set"
            | "typedef"
            | "factory"
            | "async"
            | "await"
            | "yield"
            | "in"
            | "is"
            | "as"
    )
}

fn extract_swift(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Swift)
        .map_err(|e| greppy_core::Error::Parse(format!("compile swift queries: {e}")))?;
    // Base pass (defs for class_declaration + function_declaration,
    // calls, imports): the spec engine already emits the
    // Module node, one "Class" node per `class_declaration` (Swift's grammar
    // labels class / struct / enum all `class_declaration`),
    // a "Method" node owned by its enclosing type for
    // every `function_declaration` inside a type body, a free "Function" node
    // for every top-level `func`, the CALLS pass and the IMPORTS pass. What the
    // uniform template does NOT model is
    // added below: `protocol_declaration` → "Interface"; every
    // `property_declaration` → "Variable"; the enum-method double-count;
    // the DEFINES_METHOD / IMPLEMENTS edges; and the `pass_usages` USAGE walk.
    let mut result = crate::spec::spec_extract(
        Language::Swift,
        &crate::spec::SWIFT,
        queries,
        source,
        file_path,
    )?;

    let tree = crate::parse(Language::Swift, source)?;
    let root = tree.root_node();

    swift_defs_pass(source, root, file_path, &mut result);

    let file_module_qname = format!("{file_path}::__file__");
    swift_emit_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The keyword child (`class` / `struct` / `enum`) of a Swift
/// `class_declaration`, read off the `declaration_kind:` field. Swift's grammar
/// collapses all three into `class_declaration`; every one is
/// labelled "Class", so the spec engine's single "Class" label already
/// matches — this is used only to detect the *enum* case (whose body node is an
/// `enum_class_body`, which is NOT recognised as
/// a class body, so its `func`s are re-walked as free Functions in addition to the
/// Method already emitted — the double-count handled below).
fn swift_declaration_kind<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    node.child_by_field_name("declaration_kind")
        .map(|k| node_text(source, k))
}

/// The `name:` (`type_identifier`) of a Swift type declaration
/// (`class_declaration` / `protocol_declaration`), or `None`.
fn swift_type_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    node.child_by_field_name("name")
        .map(|n| node_text(source, n))
}

/// The owning type *name* for a `function_declaration` (its nearest enclosing
/// `class_declaration`), or `None` when the func is free (file scope). Mirrors
/// the spec engine's `enclosing_owner_name` so the Method qname and the
/// DEFINES_METHOD endpoints line up with the spec-emitted nodes.
fn swift_func_owner_name<'a>(source: &'a [u8], func: Node<'_>) -> Option<&'a str> {
    let mut p = func.parent();
    while let Some(cur) = p {
        if cur.kind() == "class_declaration" {
            return swift_type_name(source, cur);
        }
        p = cur.parent();
    }
    None
}

/// The nearest enclosing Swift callable qname for `node`:
/// the closest `function_declaration` ancestor, owned
/// by its nearest enclosing type (`{file}::{Owner}::{name}`) or free
/// (`{file}::Function::{name}`). Returns `None` at file / type scope (the caller
/// substitutes the file Module qname).
fn swift_enclosing_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if cur.kind() == "function_declaration" {
            let name = swift_type_name(source, cur)?; // `name:` field on the func
            return Some(match swift_func_owner_name(source, cur) {
                Some(owner) => format!("{file_path}::{owner}::{name}"),
                None => format!("{file_path}::Function::{name}"),
            });
        }
        p = cur.parent();
    }
    None
}

/// The Variable *name* of a `property_declaration` — its `name:` field is a
/// `pattern` whose `bound_identifier:` is the `simple_identifier`.
/// The name is resolved off
/// the property's pattern. Returns `None` for anonymous / non-simple patterns.
fn swift_property_name<'a>(source: &'a [u8], prop: Node<'_>) -> Option<&'a str> {
    let pattern = prop.child_by_field_name("name")?;
    let ident = match pattern.child_by_field_name("bound_identifier") {
        Some(n) => Some(n),
        None => {
            let mut c = pattern.walk();
            let found = pattern
                .named_children(&mut c)
                .find(|n| n.kind() == "simple_identifier");
            found
        }
    }?;
    Some(node_text(source, ident))
}

/// Second definitions pass over the Swift tree, adding what the uniform spec
/// template does not model:
///
///   * `protocol_declaration` → an "Interface" node. Its body
///     (`protocol_body`) holds `protocol_function_declaration` /
///     `protocol_property_declaration` — neither is a `function_declaration` /
///     `property_declaration`, so no Method / Variable is emitted for it.
///   * every `property_declaration` (top-level or inside a type body) → a
///     "Variable" node.
///   * every `function_declaration` directly inside an `enum_class_body` → an
///     ADDITIONAL "Function" node: `enum_class_body` is not
///     recognised as a class body, so the defs walk re-walks
///     the enum's `func`s and labels each one "Function" —
///     on top of the "Method" the spec pass already emitted for it.
///   * DEFINES_METHOD: each type → every method it owns,
///     pointing at the spec-emitted Method node.
///   * IMPLEMENTS: each `class_declaration` / `protocol_declaration` →
///     every type named in its `inheritance_specifier`.
fn swift_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "protocol_declaration" => {
                if let Some(name) = swift_type_name(source, node) {
                    if !name.is_empty() {
                        result.nodes.push(ExtractedNode {
                            label: "Interface".into(),
                            name: name.to_string(),
                            qualified_name: format!("{file_path}::Interface::{name}"),
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                }
                swift_emit_implements(source, node, file_path, "Interface", result);
                // Descend only into nested *type* declarations (a protocol body
                // holds `protocol_function_declaration` / `protocol_property_
                // declaration`, which are neither a `function_declaration` nor a
                // `property_declaration`, so they emit nothing — but a rare
                // nested type must still be reached without re-processing the
                // body's requirement members).
                if let Some(body) = node.child_by_field_name("body") {
                    swift_push_nested_types(body, &mut stack);
                }
            }
            "class_declaration" => {
                let is_enum = swift_declaration_kind(source, node) == Some("enum");
                swift_emit_implements(source, node, file_path, "Class", result);

                // DEFINES_METHOD + (enum only) the double-counted Function node.
                let owner = swift_type_name(source, node);
                if let (Some(owner), Some(body)) = (owner, node.child_by_field_name("body")) {
                    let mut bc = body.walk();
                    for member in body.named_children(&mut bc) {
                        match member.kind() {
                            "function_declaration" => {
                                if let Some(m) = swift_type_name(source, member) {
                                    if !m.is_empty() {
                                        result.edges.push(ExtractedEdge {
                                            edge_type: "DEFINES_METHOD".into(),
                                            source_qualified_name: format!(
                                                "{file_path}::Class::{owner}"
                                            ),
                                            target_qualified_name: format!(
                                                "{file_path}::{owner}::{m}"
                                            ),
                                            file_path: file_path.to_string(),
                                            line: member.start_position().row as u32 + 1,
                                            properties: serde_json::json!({}),
                                        });
                                        if is_enum {
                                            // Second, "Function"-labelled count.
                                            result.nodes.push(ExtractedNode {
                                                label: "Function".into(),
                                                name: m.to_string(),
                                                qualified_name: format!(
                                                    "{file_path}::Function::{m}"
                                                ),
                                                file_path: file_path.to_string(),
                                                start_line: member.start_position().row as u32 + 1,
                                                end_line: member.end_position().row as u32 + 1,
                                                properties: serde_json::json!({}),
                                            });
                                        }
                                    }
                                }
                            }
                            "property_declaration" => {
                                swift_emit_variable(source, member, file_path, result);
                            }
                            _ => {}
                        }
                    }
                }

                // Descend only into nested *type* declarations so the body's
                // `function_declaration` / `property_declaration` members (already
                // handled above) are not re-processed into duplicate work.
                if let Some(body) = node.child_by_field_name("body") {
                    swift_push_nested_types(body, &mut stack);
                }
            }
            "property_declaration" => {
                // A file-top-level property (a type-body property is handled in
                // the enclosing class arm and never re-descended into).
                swift_emit_variable(source, node, file_path, result);
            }
            "function_declaration" => {
                // A free function's body — do not descend
                // (Swift function bodies are not re-walked for further defs).
            }
            _ => {
                let mut c = node.walk();
                for child in node.named_children(&mut c) {
                    stack.push(child);
                }
            }
        }
    }
}

/// Push every nested `class_declaration` / `protocol_declaration` found under a
/// type `body` onto the defs stack (so a nested type gets its own Interface /
/// Variable / DEFINES_METHOD / IMPLEMENTS treatment), WITHOUT re-visiting the
/// body's method / property members. Only the type-declaration children of a
/// class body are pushed — the enclosing
/// class arm already emitted the members, so a plain child descent would
/// re-process them. Descent stops at the boundary of a found type declaration
/// (its own body is walked when it is popped).
fn swift_push_nested_types<'a>(body: Node<'a>, stack: &mut Vec<Node<'a>>) {
    let mut inner = vec![body];
    while let Some(cur) = inner.pop() {
        let mut c = cur.walk();
        for child in cur.named_children(&mut c) {
            match child.kind() {
                "class_declaration" | "protocol_declaration" => stack.push(child),
                // A method / property body can itself hold a locally-declared
                // type; keep scanning through non-type nodes to reach it.
                _ => inner.push(child),
            }
        }
    }
}

/// Emit a "Variable" node for a `property_declaration`
/// (drops empty names and the `_` placeholder).
fn swift_emit_variable(
    source: &[u8],
    prop: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let Some(name) = swift_property_name(source, prop) else {
        return;
    };
    if name.is_empty() || name == "_" {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Variable".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Variable::{name}"),
        file_path: file_path.to_string(),
        start_line: prop.start_position().row as u32 + 1,
        end_line: prop.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// Emit an `IMPLEMENTS` edge from a type node (`{file}::{label}::{name}`) to
/// each `type_identifier` named in its `inheritance_specifier`(s): one
/// IMPLEMENTS edge per named super-type. The
/// resolver keys IMPLEMENTS on the target qname directly (no name-based branch),
/// so — like the Rust `impl Trait for Type` extractor — the target is
/// the same-file guess qname `{file}::Interface::{base}` (a Swift protocol
/// conformance names an Interface node). It resolves when the protocol is
/// declared in the same file; a genuinely cross-file conformance would need the
/// resolver's honesty guard relaxed and is left unresolved (exact
/// here for same-file conformances, the only shape a `by_qname` target reaches).
fn swift_emit_implements(
    source: &[u8],
    type_node: Node<'_>,
    file_path: &str,
    label: &str,
    result: &mut ExtractionResult,
) {
    let Some(name) = swift_type_name(source, type_node) else {
        return;
    };
    if name.is_empty() {
        return;
    }
    let src_qname = format!("{file_path}::{label}::{name}");
    let mut c = type_node.walk();
    for child in type_node.named_children(&mut c) {
        if child.kind() != "inheritance_specifier" {
            continue;
        }
        // Each inheritance_specifier wraps one `user_type` (`inherits_from:`)
        // whose leading `type_identifier` names the super-type.
        let base = swift_first_type_identifier(source, child);
        let Some(base) = base else { continue };
        if base.is_empty() || base == name {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "IMPLEMENTS".into(),
            source_qualified_name: src_qname.clone(),
            target_qualified_name: format!("{file_path}::Interface::{base}"),
            file_path: file_path.to_string(),
            line: child.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "name": base,
                "trait_name": base,
                "type_name": name,
            }),
        });
    }
}

/// The text of the first `type_identifier` at or under `node` (the leading
/// name of a `user_type`, unwrapping any qualifier), or `None`.
fn swift_first_type_identifier<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    if node.kind() == "type_identifier" {
        return Some(node_text(source, node));
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        if let Some(t) = swift_first_type_identifier(source, child) {
            return Some(t);
        }
    }
    None
}

/// USAGE pass for Swift. Every
/// `simple_identifier` / `type_identifier` / `identifier` reference emits a
/// USAGE edge unless it is a definition *name*, sits inside a call node
/// (`call_expression` / `constructor_expression` / `macro_invocation` /
/// `navigation_expression` — already a CALLS edge, and its nested references
/// suppressed), sits inside an import, or is a Swift keyword. The `ref_name` is
/// resolved project-wide by the indexer, so the target qname is a placeholder
/// that never resolves directly. The source is the nearest enclosing callable
/// qname, falling back to the per-file Module node at file / type scope.
fn swift_emit_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let kind = node.kind();
    if matches!(kind, "simple_identifier" | "type_identifier" | "identifier")
        && !is_inside_kind(
            node,
            &[
                "call_expression",
                "constructor_expression",
                "macro_invocation",
                "navigation_expression",
                "import_declaration",
            ],
        )
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_swift_usage_keyword(text) {
            let source_qname = swift_enclosing_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({
                    "ref_name": text,
                }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        swift_emit_usages(source, child, file_path, file_module_qname, result);
    }
}

/// Swift keyword / literal filter — the generic keyword table.
/// A reference whose text is one of these never emits a
/// usage.
fn is_swift_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "nil"
            | "None"
            | "undefined"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "struct"
            | "enum"
            | "interface"
            | "trait"
            | "impl"
            | "import"
            | "export"
            | "package"
            | "module"
            | "use"
            | "require"
            | "include"
            | "new"
            | "delete"
            | "this"
            | "self"
            | "super"
            | "public"
            | "private"
            | "protected"
            | "static"
            | "const"
            | "var"
            | "let"
            | "function"
            | "def"
            | "fn"
            | "func"
            | "fun"
            | "proc"
            | "sub"
            | "method"
            | "async"
            | "await"
            | "yield"
    )
}

// ---------------------------------------------------------------------------
// Elixir extraction
// ---------------------------------------------------------------------------
//
// Elixir's tree-sitter grammar has NO distinct definition kinds: `def`, `defp`,
// `defmacro` and `defmodule` all parse as generic `call` nodes whose first child
// is an `identifier` naming the macro. The uniform spec template keys DefRules on
// node kinds and cannot express this, so — like the other hand-written languages
// — Elixir gets a bespoke pass.
//
// Per file the walk produces:
//
//   * one per-file **Module** node (added by the structural pass, not
//     here — the parser output must not emit it);
//   * every `defmodule Foo do … end` → a **"Class"** node named by the module
//     alias, and the walk then
//     descends ONLY into that module's `do_block`, visiting its direct `call`
//     children;
//   * every `def` / `defp` / `defmacro` call in a module body → a **"Function"**
//     node named by the head identifier of its first argument.
//     NB `defmacrop` is NOT in the set, so it is not
//     extracted; def bodies are not re-walked, so nested defs are not reached.
//
// Edges, on top of the structural DEFINES (auto, one File→def per def node):
//   * **CALLS** — the calls walk visits every `call` node and takes a callee
//     (a bare `identifier`, or a `dot` whose trailing `identifier` is the
//     method). The source is the nearest enclosing `def` / `defp` / `defmacro`
//     qname, falling back to the file Module only for module-scope calls.
//   * **IMPORTS** — the provider query captures aliases from `alias` / `import`
//     / `require`; each becomes a file-sourced edge keyed by the alias's final
//     dotted segment (`Types.Widget` -> `Widget`).
//   * **TYPE_REF** — `%Widget{}` parses as a `struct` containing an `alias`;
//     that alias is attributed to the nearest enclosing definition.

/// The Elixir macro keywords whose `call` nodes are definitions.
/// `defmacrop` is deliberately absent (the set is
/// exactly def/defp/defmacro).
fn elixir_is_func_macro(kw: &str) -> bool {
    matches!(kw, "def" | "defp" | "defmacro")
}

/// The first child of an Elixir `call` node, i.e. the callee/macro node.
fn elixir_call_head(node: Node<'_>) -> Option<Node<'_>> {
    node.child(0)
}

/// The `arguments` node of an Elixir `call`, falling back to the second child.
fn elixir_call_args(node: Node<'_>) -> Option<Node<'_>> {
    node.child_by_field_name("arguments").or_else(|| {
        if node.child_count() > 1 {
            node.child(1)
        } else {
            None
        }
    })
}

/// The defined name is the head identifier
/// of the def's first argument — either a `call` (`def add(a, b)`), whose first
/// child is the name, or a bare `identifier` (`def add` with no parens).
fn elixir_func_def_name<'a>(source: &'a [u8], call: Node<'_>) -> Option<&'a str> {
    let args = elixir_call_args(call)?;
    let first_arg = args.child(0)?;
    match first_arg.kind() {
        "call" => first_arg.child(0).map(|n| node_text(source, n)),
        "identifier" => Some(node_text(source, first_arg)),
        _ => None,
    }
}

/// Resolve the nearest enclosing Elixir definition to the same free-Function
/// qname emitted by `elixir_defs_pass`. Elixir represents definitions as
/// ordinary `call` nodes, so the shared callable-kind walk cannot see them.
fn elixir_enclosing_callable_qname(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
) -> Option<String> {
    let mut parent = node.parent();
    while let Some(candidate) = parent {
        if candidate.kind() == "call" {
            if let Some(head) = elixir_call_head(candidate) {
                if head.kind() == "identifier" && elixir_is_func_macro(node_text(source, head)) {
                    if let Some(name) = elixir_func_def_name(source, candidate) {
                        if !name.is_empty() {
                            return Some(format!("{file_path}::Function::{name}"));
                        }
                    }
                }
            }
        }
        parent = candidate.parent();
    }
    None
}

/// True when `node` is the function-head `call` nested inside a `def name(...)`
/// macro. It is syntax for the definition, not a runtime self-call.
fn elixir_is_definition_head_call(source: &[u8], node: Node<'_>) -> bool {
    let Some(args) = node.parent().filter(|parent| parent.kind() == "arguments") else {
        return false;
    };
    if args.child(0).map(|first| first.id()) != Some(node.id()) {
        return false;
    }
    let Some(def_call) = args.parent().filter(|parent| parent.kind() == "call") else {
        return false;
    };
    elixir_call_head(def_call)
        .filter(|head| head.kind() == "identifier")
        .map(|head| elixir_is_func_macro(node_text(source, head)))
        .unwrap_or(false)
}

/// The trailing bare name of an Elixir call's callee node: for a `dot`
/// (`Product.product_label`) the last `identifier`; for a bare `identifier`
/// (`format_user_label`) itself. Returns `None` for anything else. This is the
/// name the resolver matches against project Function names (the whole
/// dotted callee is stored and matched on the last `.`-segment downstream).
fn elixir_callee_name<'a>(source: &'a [u8], head: Node<'_>) -> Option<&'a str> {
    match head.kind() {
        "identifier" => Some(node_text(source, head)),
        "dot" => {
            // `dot` children: <operand> '.' identifier — take the last identifier.
            let mut last: Option<Node<'_>> = None;
            let mut c = head.walk();
            for child in head.named_children(&mut c) {
                if child.kind() == "identifier" {
                    last = Some(child);
                }
            }
            last.map(|n| node_text(source, n))
        }
        _ => None,
    }
}

/// The Elixir definition walk. Emits Class nodes for
/// `defmodule` and Function nodes for `def`/`defp`/`defmacro`, descending only
/// into module `do_block`s.
fn elixir_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    // Seed with the top-level `call` nodes (each top-level
    // `call` is handled directly, without descending generically).
    let mut stack: Vec<Node<'_>> = Vec::new();
    let mut rc = root.walk();
    for child in root.named_children(&mut rc) {
        if child.kind() == "call" {
            stack.push(child);
        }
    }

    while let Some(cur) = stack.pop() {
        let Some(head) = elixir_call_head(cur) else {
            continue;
        };
        if head.kind() != "identifier" {
            continue;
        }
        let macro_kw = node_text(source, head);

        if elixir_is_func_macro(macro_kw) {
            if let Some(name) = elixir_func_def_name(source, cur) {
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Function".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Function::{name}"),
                        file_path: file_path.to_string(),
                        start_line: cur.start_position().row as u32 + 1,
                        end_line: cur.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
        } else if macro_kw == "defmodule" {
            let name = elixir_call_args(cur)
                .and_then(|args| args.child(0))
                .map(|n| node_text(source, n));
            if let Some(name) = name {
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Class".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Class::{name}"),
                        file_path: file_path.to_string(),
                        start_line: cur.start_position().row as u32 + 1,
                        end_line: cur.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            // Descend into the module's do_block, visiting its direct `call`
            // children (C pushes each `call` child of the do_block).
            if let Some(do_block) = named_child_of_kinds(cur, &["do_block"]) {
                let mut dc = do_block.walk();
                for child in do_block.named_children(&mut dc) {
                    if child.kind() == "call" {
                        stack.push(child);
                    }
                }
            }
        }
    }
}

/// The Elixir calls walk: visit every node, and for each runtime `call` whose
/// callee resolves to a bare name that is not a keyword, emit a CALLS edge from
/// the nearest enclosing Elixir definition (or the file Module at module scope).
/// The indexer's name resolver drops callees with no matching project Function.
fn elixir_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "call" && !elixir_is_definition_head_call(source, node) {
            if let Some(head) = elixir_call_head(node) {
                if let Some(name) = elixir_callee_name(source, head) {
                    if !name.is_empty() && !is_elixir_call_keyword(name) {
                        let source_qname = elixir_enclosing_callable_qname(source, node, file_path)
                            .unwrap_or_else(|| file_module_qname.to_string());
                        result.edges.push(ExtractedEdge {
                            edge_type: "CALLS".into(),
                            source_qualified_name: source_qname,
                            target_qualified_name: format!("{file_path}::Function::{name}"),
                            file_path: file_path.to_string(),
                            line: node.start_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "callee_text": name,
                                "callee_name": name,
                            }),
                        });
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Elixir callee keyword filter — the generic keyword table. A callee whose bare name
/// is one of these never becomes a CALLS candidate, so `def add(…)` (callee
/// "def") and builtins like `new`/`if` drop out before resolution.
fn is_elixir_call_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "nil"
            | "None"
            | "undefined"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "struct"
            | "enum"
            | "interface"
            | "trait"
            | "impl"
            | "import"
            | "export"
            | "package"
            | "module"
            | "use"
            | "require"
            | "include"
            | "new"
            | "delete"
            | "this"
            | "self"
            | "super"
            | "public"
            | "private"
            | "protected"
            | "static"
            | "const"
            | "var"
            | "let"
            | "function"
            | "def"
            | "fn"
            | "func"
            | "fun"
            | "proc"
            | "sub"
            | "method"
            | "async"
            | "await"
            | "yield"
    )
}

/// Emit one file-sourced IMPORTS edge for every provider-query capture from an
/// `alias`, `import`, or `require` call. Resolution keys on the final dotted
/// segment so `alias Types.Widget` links to the graph symbol named `Widget`.
fn elixir_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) -> greppy_core::Result<()> {
    let language = tree_sitter_elixir::LANGUAGE.into();
    let query = tree_sitter::Query::new(&language, crate::langs::elixir::IMPORTS)
        .map_err(|e| greppy_core::Error::Parse(format!("compile elixir imports query: {e}")))?;
    let imported_capture = query
        .capture_index_for_name("imported")
        .ok_or_else(|| greppy_core::Error::Parse("elixir imports query lacks @imported".into()))?;
    let mut cursor = QueryCursor::new();
    let mut captures = cursor.captures(&query, root, source);
    while let Some((query_match, capture_index)) = captures.next() {
        let capture = query_match.captures[*capture_index];
        if capture.index != imported_capture {
            continue;
        }
        let path = node_text(source, capture.node);
        let imported_name = path.rsplit('.').next().unwrap_or(path);
        if path.is_empty() || imported_name.is_empty() {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "IMPORTS".into(),
            source_qualified_name: file_module_qname.to_string(),
            target_qualified_name: format!("{file_path}::__import__::{path}"),
            file_path: file_path.to_string(),
            line: capture.node.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "path": path,
                "imported_name": imported_name,
                "original_name": imported_name,
                "glob": false,
            }),
        });
    }
    Ok(())
}

/// Emit TYPE_REF for `%Alias{}` struct literals. The grammar places the alias
/// directly under a `struct` node, making this a narrow pass that does not
/// confuse ordinary module aliases or remote calls with type references.
fn elixir_struct_refs_pass(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if node.kind() == "struct" {
        if let Some(alias) = node.named_child(0).filter(|child| child.kind() == "alias") {
            let path = node_text(source, alias);
            let ref_name = path.rsplit('.').next().unwrap_or(path);
            if !ref_name.is_empty() {
                let source_qname = elixir_enclosing_callable_qname(source, node, file_path)
                    .unwrap_or_else(|| file_module_qname.to_string());
                result.edges.push(ExtractedEdge {
                    edge_type: "TYPE_REF".into(),
                    source_qualified_name: source_qname,
                    target_qualified_name: format!("{file_path}::__type__::{ref_name}"),
                    file_path: file_path.to_string(),
                    line: alias.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "type_name": ref_name }),
                });
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        elixir_struct_refs_pass(source, child, file_path, file_module_qname, result);
    }
}

/// Bespoke Elixir extraction. See the module-level
/// comment above `elixir_defs_pass` for the full node/edge mapping.
fn extract_elixir(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_elixir::LANGUAGE.into())
        .map_err(|e| greppy_core::Error::Parse(format!("set elixir language: {e}")))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| greppy_core::Error::Parse("tree-sitter parse returned None".into()))?;
    let root = tree.root_node();

    let mut result = ExtractionResult::default();
    elixir_defs_pass(source, root, file_path, &mut result);

    let file_module_qname = format!("{file_path}::__file__");
    elixir_calls_pass(source, root, file_path, &file_module_qname, &mut result);
    elixir_imports_pass(source, root, file_path, &file_module_qname, &mut result)?;
    elixir_struct_refs_pass(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

// ===========================================================================
// Clojure — registry language with a bespoke Lisp-aware pass.
//
// Clojure is driven entirely through its Lisp special-cases, NOT the uniform
// spec template:
//   * The DEFS pass special-cases every `list_lit`: it reads the head symbol
//     of the form and — when it is a def head (`def`, `defn`, `defn-`,
//     `defmacro`, `defmulti`, `defmethod`, `defprotocol`, `defrecord`,
//     `deftype`, `definterface`, `defonce`) — emits ONE definition node whose
//     label is:
//       `defrecord`/`deftype`            → "Struct"
//       `defprotocol`/`definterface`     → "Interface"
//       everything else                  → "Function"
//     It then descends into the children, so nested def forms are captured too.
//     The always-present per-file "Module" node is emitted by the shared
//     structural pass (the indexer), not here.
//   * The CALLS pass matches the call node kind {list_lit} and reads the head
//     symbol; the callee resolves (Function/Method only) against project defs.
//     Clojure has no function node kinds, so the enclosing func is always the
//     file module — every CALLS is sourced from the per-file Module node, and
//     the call graph dedups by (caller, callee).
//   * The IMPORTS pass walks every `list_lit` and, for a
//     `(ns name (:require ..) (:use ..))` form, pushes one import per module in
//     each dependency clause (`[app.util :as u]` → module `app.util`).
//
// `extract_clojure` implements exactly those three passes.

/// The Clojure def-form heads recognised (the Clojure subset — Scheme/Racket
/// heads never appear in `.clj`). The head is the first named child's symbol
/// text of a `list_lit`.
const CLOJURE_DEF_HEADS: [&str; 11] = [
    "def",
    "defn",
    "defn-",
    "defmacro",
    "defmulti",
    "defmethod",
    "defprotocol",
    "defrecord",
    "deftype",
    "definterface",
    "defonce",
];

/// The label for a def head: `defrecord`/`deftype` →
/// "Struct", `defprotocol`/`definterface` → "Interface", every other def head →
/// "Function".
fn clojure_def_label(head: &str) -> &'static str {
    match head {
        "defrecord" | "deftype" => "Struct",
        "defprotocol" | "definterface" => "Interface",
        _ => "Function",
    }
}

fn extract_clojure(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    clojure_defs_pass(source, root, file_path, &mut result);
    clojure_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    clojure_imports_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// The head symbol text of a `list_lit` (its first named `sym_lit` child's
/// text), or `None` when the first named child is not a symbol. The full symbol text
/// is returned (`util/square` stays qualified) so it can be tested against the
/// def-head / keyword tables directly.
fn clojure_head_sym<'a, 't>(source: &'a [u8], list: Node<'t>) -> Option<(&'a str, Node<'t>)> {
    let head = list.named_child(0)?;
    if head.kind() != "sym_lit" {
        return None;
    }
    Some((node_text(source, head), head))
}

/// The resolvable trailing name of a Clojure symbol (`sym_lit`): its `sym_name`
/// child text (`util/square` → `square`, `add` → `add`). Falls back to the whole
/// symbol text when there is no `sym_name` child.
fn clojure_sym_leaf<'a>(source: &'a [u8], sym: Node<'_>) -> &'a str {
    match find_child_of_kind(sym, "sym_name") {
        Some(n) => node_text(source, n),
        None => node_text(source, sym),
    }
}

/// DEFS pass for Clojure. Walks every
/// `list_lit` (C falls through and descends after each), and for a form whose
/// head symbol is a def head emits ONE node with the Struct/Interface/Function
/// label. The name is the second named child; when that child is itself a list
/// (`(define (foo ..) ..)` — not idiomatic Clojure but handled for fidelity) the
/// name is that list's head symbol. The `sym_name` leaf is used so a namespaced
/// def name resolves on its trailing segment.
fn clojure_defs_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "list_lit" && node.named_child_count() >= 2 {
            if let Some((head_text, _)) = clojure_head_sym(source, node) {
                if CLOJURE_DEF_HEADS.contains(&head_text) {
                    // Name target: the second named child. If it is a nested
                    // list, the name is that list's first named child (the
                    // `(define (foo ..))` case). Otherwise the symbol itself.
                    if let Some(target) = node.named_child(1) {
                        let name_node = if target.kind() == "list_lit" {
                            target.named_child(0)
                        } else {
                            Some(target)
                        };
                        if let Some(name_node) = name_node {
                            let name = clojure_sym_leaf(source, name_node);
                            if !name.is_empty() {
                                let label = clojure_def_label(head_text);
                                result.nodes.push(ExtractedNode {
                                    label: label.into(),
                                    name: name.to_string(),
                                    qualified_name: format!("{file_path}::{label}::{name}"),
                                    file_path: file_path.to_string(),
                                    start_line: node.start_position().row as u32 + 1,
                                    end_line: node.end_position().row as u32 + 1,
                                    properties: serde_json::json!({}),
                                });
                            }
                        }
                    }
                }
            }
        }
        // Descend into every form's children so nested defs are reached.
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// CALLS pass for Clojure: every `list_lit` (the sole call node kind) whose
/// head is a symbol is a call whose callee is that head symbol. The callee
/// resolves against project Function/Method defs (`CALLABLE_LABELS`), so
/// def-form heads (`defn`, …), special forms, and unresolved names simply drop
/// out. The source is always the per-file Module node (Clojure has no function
/// node kinds, so the enclosing func is always the module). The call graph
/// dedups by (caller, callee); the indexer's `ON CONFLICT(source_id,
/// target_id, edge_type)` upsert reproduces that, so naive per-form emission
/// collapses to the same edge set.
fn clojure_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "list_lit" {
            if let Some((head_text, _)) = clojure_head_sym(source, node) {
                // The callee is the WHOLE head-symbol text; the resolver then
                // splits only on the last `.`, NOT on `/`. So a
                // NAMESPACE-qualified call (`util/square`, `product/build-product`)
                // keeps its `ns/` prefix and never matches a bare Function name —
                // only bare same-file calls resolve. Key the resolver on the
                // short name (text after the last `.`), which retains any `/` so
                // qualified calls stay unresolved, while a bare call (`add`,
                // `square`) resolves same-file via the direct
                // `{file}::Function::{name}` qname.
                let callee = clojure_call_short(head_text);
                if !callee.is_empty()
                    && !is_clojure_keyword(callee)
                    && !CLOJURE_DEF_HEADS.contains(&head_text)
                {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Function::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": head_text,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// IMPORTS pass for Clojure:
/// walk every `list_lit`, and for a `(ns name (:require ..) (:use ..) ..)` form
/// push one IMPORTS edge per module named in each dependency clause. A clause
/// entry may be a bare symbol (`(:use app.io)` → `app.io`) or a vector
/// (`[app.util :as u]` → its first symbol `app.util`). The source is the
/// per-file Module node; the module name resolves cross-file to the declaring
/// file (out of the in-scope node/label set, but emitted for edge parity).
fn clojure_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "list_lit" {
            if let Some((head_text, _)) = clojure_head_sym(source, node) {
                if head_text == "ns" {
                    // Dependency clauses are the keyword-headed lists after the
                    // namespace symbol (child 0 = `ns`, child 1 = ns name).
                    let mut c = node.walk();
                    for (i, clause) in node.named_children(&mut c).enumerate() {
                        if i < 2 || clause.kind() != "list_lit" {
                            continue;
                        }
                        clojure_push_clause_modules(
                            source,
                            clause,
                            file_module_qname,
                            file_path,
                            result,
                        );
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Push one IMPORTS edge per module named in a `(:require ..)` / `(:use ..)` /
/// `(:import ..)` clause. The clause head is a
/// `kwd_lit` keyword; each following entry names a module: a `vec_lit` /
/// `list_lit` yields its first symbol, a bare symbol yields itself.
fn clojure_push_clause_modules(
    source: &[u8],
    clause: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    // Only `:require` / `:use` / `:import` clauses denote dependencies; other
    // keyword clauses (`:gen-class`, `:refer-clojure`, …) are not imports.
    let head = clause.named_child(0);
    let is_dep = head
        .filter(|h| h.kind() == "kwd_lit")
        .and_then(|h| find_child_of_kind(h, "kwd_name"))
        .map(|kw| matches!(node_text(source, kw), "require" | "use" | "import"))
        .unwrap_or(false);
    if !is_dep {
        return;
    }
    let mut c = clause.walk();
    for (i, item) in clause.named_children(&mut c).enumerate() {
        if i == 0 {
            continue; // skip the leading keyword head
        }
        let module = match item.kind() {
            // `[app.util :as u]` / `(app.util :as u)` — the module is the
            // vector/list's first symbol.
            "vec_lit" | "list_lit" => item
                .named_child(0)
                .filter(|n| n.kind() == "sym_lit")
                .map(|n| node_text(source, n)),
            // bare symbol: `app.io`.
            "sym_lit" => Some(node_text(source, item)),
            _ => None,
        };
        if let Some(module) = module {
            if !module.is_empty() {
                result.edges.push(ExtractedEdge {
                    edge_type: "IMPORTS".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::__import__::{module}"),
                    file_path: file_path.to_string(),
                    line: item.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "import_path": module,
                        "module_path": module,
                        "local_name": module.rsplit('.').next().unwrap_or(module),
                    }),
                });
            }
        }
    }
}

/// The resolver-facing short name of a Clojure callee: the substring after the
/// LAST `.`, or the whole text when there is none. Crucially this does NOT split
/// on `/`, so a namespace-qualified call (`util/square`) keeps its `ns/` prefix
/// and never matches a bare `Function` name, leaving qualified Clojure calls
/// unresolved.
fn clojure_call_short(callee: &str) -> &str {
    match callee.rfind('.') {
        Some(idx) => &callee[idx + 1..],
        None => callee,
    }
}

/// Clojure keyword / special-form / literal filter for the CALLS pass. Clojure
/// runs through the generic keyword table; a callee whose trailing name is one
/// of these never becomes a resolved CALLS candidate. Reuses the shared generic
/// list (identical to the other data-path languages).
fn is_clojure_keyword(name: &str) -> bool {
    is_scala_usage_keyword(name)
}

// ===========================================================================
// PureScript — registry language with a bespoke pass.
//
// PureScript takes the *generic* (non-FP-special-cased) path: it is not in the
// Haskell/OCaml callee branch nor the Haskell reference-node arm, and its
// imports fall to the generic spec-import parser. Its node-kind spec is:
//   * function node kinds = {function}
//   * class node kinds    = {class_declaration, data, newtype, type_alias}
//   * module node kinds   = {module}
//   * call node kinds     = {exp_apply}
//   * import node kinds   = {import, import_item, instance}
//   * variable node kinds = {signature}
//
// What this pass EMITS on this grammar:
//
//   DEFS:
//     * a top-level `function` → a free "Function" node (name = `name:` field,
//       a `variable`). The pass STOPS after emitting — it does NOT
//       descend into the body, so `let`-bound locals inside a value binding are
//       NOT emitted (this is the ONLY difference from a generic tree-sitter
//       def query, which would over-count those nested `function` nodes).
//     * `data` / `newtype` → a "Class" node (name = `name:` field, a `type`);
//       the label mapping sends neither to Interface/Enum/Type, so "Class".
//     * `type_alias` → a "Type" node (the label mapping special-cases the
//       kind string `type_alias` → "Type"); name = `name:` field.
//     * `class_declaration` → NO node: its name is nested under
//       `class_head > class_name > type`, NOT a direct `name:` field, so the
//       `name` field lookup is null
//       and the def is dropped. The pass then descends into the class body,
//       but the body holds only `signature`
//       method decls (a variable kind, not a `function`), so nothing is
//       emitted there either. Type-class declarations therefore contribute
//       zero nodes.
//     * `signature` → NO node (a top-level `f :: T` type signature; the
//       variable path would emit a "Variable" only for a
//       `signature` that exposes a bound `name` in a value position, which the
//       PureScript `signature` does not — zero Variables here).
//
//   CALLS: **zero**. PureScript's call node is `exp_apply`, whose head is an
//     `exp_name` wrapper (not a bare `identifier`/`variable`/`constructor`/
//     `value_path`). Since PureScript is not in the FP callee branch, the
//     callee lookup finds no `function:`/`name:`/`method:` field and no
//     `identifier` first child, so no FP callee is reached and the
//     generic fallback fails → NULL for every `exp_apply`. No CALLS are emitted.
//
//   IMPORTS: one edge per top-level `import` declaration, keyed by the LAST
//     segment of the imported module path (the `module:` field, so
//     `path_last("Data.Shape")` = "Shape"). The shared indexer resolves that
//     `imported_name` to the unique project-wide definition among
//     `IMPORTABLE_LABELS`; unresolved segments (`Prelude`, `Geometry`, …) are
//     dropped. On the fixture this yields: `Data.Shape`
//     (imported ×4) → Class `Shape`, `Data.Color` (×2) → Class `Color`.
//
//   USAGE: **zero**. The reference-node predicate for PureScript hits only the
//     common `identifier`/`simple_identifier`/`type_identifier` kinds (there is
//     no PureScript arm), which the tree-sitter-purescript grammar never uses
//     for references (`variable` / `qualified_variable` / `constructor` / …).
//     No USAGE edges are emitted.
//
// The Module/File/Folder/Project structural nodes and the File→DEFINES /
// CONTAINS edges are added by the indexer's shared structural pass.

/// PureScript class kinds routed through the class-def path.
const PURESCRIPT_CLASS_KINDS: [&str; 4] = ["class_declaration", "data", "newtype", "type_alias"];

fn extract_purescript(
    language: Language,
    _d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();

    purescript_defs_pass(source, root, file_path, &mut result);
    purescript_imports_pass(source, root, file_path, &mut result);

    Ok(result)
}

/// The label for a PureScript class kind: only
/// `type_alias` → "Type"; `data` / `newtype` → "Class". `class_declaration` is
/// handled separately (no node) since its name is not on a `name:` field.
fn purescript_class_label(kind: &str) -> &'static str {
    if kind == "type_alias" {
        "Type"
    } else {
        "Class"
    }
}

/// DEFS pass for PureScript: an explicit
/// stack that, for each node, routes `function` → Function (then STOPS, no
/// descent into the body) and the class kinds → their def node (then descends
/// into the body). `signature` is neither, so it is simply descended-through and
/// emits nothing.
fn purescript_defs_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        // `function` → free "Function"; STOP here (do NOT descend into the
        // value binding's body, so nested `let`-bound `function` nodes are not
        // emitted).
        if kind == "function" {
            purescript_emit_function(source, node, file_path, result);
            continue;
        }
        // `data`/`newtype`/`type_alias` → a Class/Type node; `class_declaration`
        // resolves to no name, so it emits nothing but we still descend into the
        // body — the body holds only
        // `signature` decls, which are not `function` nodes, so no node is
        // produced there.
        if PURESCRIPT_CLASS_KINDS.contains(&kind) {
            purescript_emit_type(source, node, file_path, result);
        }
        let n = node.child_count();
        for i in (0..n).rev() {
            if let Some(child) = node.child(i) {
                stack.push(child);
            }
        }
    }
}

/// Emit a free "Function" node for a `function` def node. The name is the
/// `name:` field (a `variable`). Empty names
/// are dropped, as is the literal "function".
fn purescript_emit_function(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() || name == "function" {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Function".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Function::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// Emit the "Class"/"Type" node for one `data` / `newtype` / `type_alias`, or
/// nothing for a `class_declaration` (whose name is not a direct `name:` field,
/// so it is dropped). The name is the `name:` field (a
/// `type` node).
fn purescript_emit_type(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() {
        return;
    }
    let label = purescript_class_label(node.kind());
    result.nodes.push(ExtractedNode {
        label: label.into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::{label}::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// Emit one IMPORTS edge per top-level `import` declaration by walking the
/// module's DIRECT children only:
/// read the `module:` field and key the edge on the
/// LAST segment of the dotted module path. The source is the
/// per-file `Module` node (`<file>::__file__`); the shared indexer resolves
/// `imported_name` to the unique project-wide `IMPORTABLE_LABELS` definition and
/// drops any segment that names no (or an ambiguous) definition.
fn purescript_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let file_module_qname = format!("{file_path}::__file__");
    let mut c = root.walk();
    for node in root.named_children(&mut c) {
        if node.kind() != "import" {
            continue;
        }
        let Some(module_node) = node.child_by_field_name("module") else {
            continue;
        };
        // The last `module` segment of the `qualified_module` (`Data.Shape` →
        // `Shape`). Fall back to the whole node text when there are no segment
        // children (a bare module name).
        let mut mc = module_node.walk();
        let last_seg = module_node
            .named_children(&mut mc)
            .filter(|ch| ch.kind() == "module")
            .last()
            .map(|seg| node_text(source, seg))
            .unwrap_or_else(|| node_text(source, module_node));
        let module_path = node_text(source, module_node);
        if last_seg.is_empty() {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "IMPORTS".into(),
            source_qualified_name: file_module_qname.clone(),
            target_qualified_name: format!("{file_path}::__import__::{last_seg}"),
            file_path: file_path.to_string(),
            line: node.start_position().row as u32 + 1,
            properties: serde_json::json!({
                "imported_name": last_seg,
                "module_path": module_path,
                "local_name": last_seg,
            }),
        });
    }
}

// ===========================================================================
// Objective-C — registry language with a bespoke pass.
// ===========================================================================
//
// The Objective-C passes:
//
//   * DEFINITIONS:
//       - `class_interface` / `class_implementation` (objc class kinds) →
//         "Class" (the default label). The name is the node's
//         first `identifier` child (the Objective-C
//         first-identifier fallback). The `@interface` and `@implementation`
//         for the same class share one qname (`{file}::Class::{Name}`), so
//         they collapse to one Class
//         node — de-duped here per (file, qname).
//       - `protocol_declaration` (an objc class kind) → "Interface".
//         Name = first `identifier` child.
//       - `method_definition` inside a `class_implementation`'s
//         `implementation_definition` → "Method"
//         with qname `{file}::{Class}::{method}` + a `DEFINES_METHOD` edge from
//         the owning Class node. Method name = first `identifier` child.
//         `@interface` `method_declaration`s
//         are NOT emitted (only `implementation_definition` bodies are walked).
//       - free `function_definition` emits NO node. The def pass never
//         reaches a top-level C function as a def (it only routes
//         objc class/method kinds), so ZERO Function nodes are emitted —
//         and zero Field / Variable nodes (properties/ivars are not extracted).
//   * CALLS: every `message_expression`'s selector (the
//     `method:` field) is the callee. Source = the file's
//     per-file Module node (all CALLS are `Module -> Method`);
//     the callee resolves cross/same-file to a unique `Method` by the shared
//     plumbing. C-style `call_expression`s resolve to nothing (no Function nodes
//     exist), so they contribute no edge.
//   * USAGE: every `identifier` /
//     `type_identifier` reference not inside a call / import, not a definition
//     name, and not a keyword → a `USAGE` edge from the per-file Module keyed on
//     `ref_name`; the indexer resolves it to any unique registered symbol
//     (Class / Interface / Method). Deduped by (Module, resolved-symbol)
//     over the module source.
//   * IMPORTS are emitted by the shared registry query path (the `#import` /
//     `#include` expander), so no import edge is emitted
//     here.
//
// OUT OF SCOPE: cross-file `INHERITS`
// (`class_interface : Base` where Base is defined in another file — the shared
// plumbing name-resolves only CALLS / USAGE, not INHERITS) and `SIMILAR_TO`
// (SEMANTICALLY_RELATED family).

/// Objective-C value-position keyword filter (the generic keyword
/// table used for every non-special language). A reference or
/// selector whose name is one of these never becomes a USAGE / CALLS edge.
const OBJC_KEYWORDS: &[&str] = &[
    "true",
    "false",
    "null",
    "nil",
    "NULL",
    "YES",
    "NO",
    "None",
    "undefined",
    "void",
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "return",
    "throw",
    "try",
    "catch",
    "finally",
    "class",
    "struct",
    "enum",
    "interface",
    "trait",
    "impl",
    "import",
    "export",
    "package",
    "module",
    "use",
    "require",
    "include",
    "new",
    "delete",
    "this",
    "self",
    "super",
    "public",
    "private",
    "protected",
    "static",
    "const",
    "var",
    "let",
    "function",
    "def",
    "fn",
    "func",
    "fun",
    "proc",
    "sub",
    "method",
    "async",
    "await",
    "yield",
    "id",
    "instancetype",
    "in",
];

/// The name of an Objective-C def node: its first `identifier` child.
fn objc_first_identifier<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    find_child_of_kind(node, "identifier")
        .map(|n| node_text(source, n))
        .filter(|s| !s.is_empty())
}

fn extract_objc(
    language: Language,
    _d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    objc_defs_pass(source, root, file_path, &mut result);
    objc_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    objc_usages_pass(source, root, &file_module_qname, file_path, &mut result);
    objc_imports_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// IMPORTS pass — `#import <…>` / `#import "…"` / `#include …` all parse as
/// `preproc_include` (the tree-sitter-objc grammar reduces both
/// import kinds to `preproc_include`). One IMPORTS edge per
/// directive, keyed on the path's basename (`imported_name`) so the indexer's
/// `resolve_file_imports` pass links a bare `"Shape.m"` import to that File node.
fn objc_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if matches!(node.kind(), "preproc_include" | "preproc_import") {
            if let Some(path_node) = node.child_by_field_name("path") {
                let raw = node_text(source, path_node);
                let path = raw
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .trim_matches('"')
                    .to_string();
                if !path.is_empty() {
                    let basename = path.rsplit('/').next().unwrap_or(&path).to_string();
                    result.edges.push(ExtractedEdge {
                        edge_type: "IMPORTS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Import::{path}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "path": path,
                            "imported_name": basename,
                            "original_name": basename,
                            "glob": false,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// DEFS pass — Class (interface/implementation, collapsed by qname), Interface
/// (protocol), and Method (impl-body `method_definition`) nodes plus the
/// DEFINES_METHOD ownership edges.
fn objc_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    // Class qnames already emitted, so `@interface` + `@implementation` for the
    // same class collapse to ONE node (deduped by qname).
    let mut seen_class_qnames: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "protocol_declaration" => {
                if let Some(name) = objc_first_identifier(source, node) {
                    result.nodes.push(ExtractedNode {
                        label: "Interface".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Interface::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
            "class_interface" | "class_implementation" => {
                if let Some(name) = objc_first_identifier(source, node) {
                    let qname = format!("{file_path}::Class::{name}");
                    if seen_class_qnames.insert(qname.clone()) {
                        result.nodes.push(ExtractedNode {
                            label: "Class".into(),
                            name: name.to_string(),
                            qualified_name: qname,
                            file_path: file_path.to_string(),
                            start_line: node.start_position().row as u32 + 1,
                            end_line: node.end_position().row as u32 + 1,
                            properties: serde_json::json!({}),
                        });
                    }
                    // Methods live only in `@implementation` bodies: walk the
                    // class node's `implementation_definition` children.
                    if node.kind() == "class_implementation" {
                        objc_emit_impl_methods(source, node, name, file_path, result);
                    }
                }
            }
            _ => {}
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit `Method` nodes (+ `DEFINES_METHOD`) for every `method_definition` inside
/// a `class_implementation`'s `implementation_definition` children.
/// The method name is the definition's first
/// `identifier` child; the qname is `{file}::{Class}::{method}` and the owner is
/// the Class node `{file}::Class::{Class}`.
fn objc_emit_impl_methods(
    source: &[u8],
    class_node: Node<'_>,
    class_name: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let class_qname = format!("{file_path}::Class::{class_name}");
    let mut c = class_node.walk();
    for impl_def in class_node.named_children(&mut c) {
        if impl_def.kind() != "implementation_definition" {
            continue;
        }
        let mut ic = impl_def.walk();
        for m in impl_def.named_children(&mut ic) {
            if m.kind() != "method_definition" {
                continue;
            }
            let Some(mname) = objc_first_identifier(source, m) else {
                continue;
            };
            let method_qname = format!("{file_path}::{class_name}::{mname}");
            result.nodes.push(ExtractedNode {
                label: "Method".into(),
                name: mname.to_string(),
                qualified_name: method_qname.clone(),
                file_path: file_path.to_string(),
                start_line: m.start_position().row as u32 + 1,
                end_line: m.end_position().row as u32 + 1,
                properties: serde_json::json!({}),
            });
            result.edges.push(ExtractedEdge {
                edge_type: "DEFINES_METHOD".into(),
                source_qualified_name: class_qname.clone(),
                target_qualified_name: method_qname,
                file_path: file_path.to_string(),
                line: m.start_position().row as u32 + 1,
                properties: serde_json::json!({}),
            });
        }
    }
}

/// CALLS pass — every `message_expression`'s selector (`method:` field, the
/// first selector segment) is the callee. Source = the per-file Module node
/// (all CALLS are `Module -> Method`); target is the same-file
/// `{file}::Method::{callee}` guess plus a `callee_name` property, resolved
/// cross/same-file to a unique `Method` by the shared plumbing.
fn objc_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "message_expression" {
            if let Some(sel) = node.child_by_field_name("method") {
                let callee = node_text(source, sel);
                if !callee.is_empty() && !OBJC_KEYWORDS.contains(&callee) {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Method::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": callee,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// USAGE pass for
/// Objective-C. Every `identifier` / `type_identifier` reference not inside a
/// call (`message_expression` / `call_expression`) or import (`preproc_include`
/// / `preproc_import`), not a definition NAME, and not a keyword becomes a
/// `USAGE` edge from the per-file Module keyed on `ref_name`. The indexer
/// resolves `ref_name` to any unique registered symbol and dedups by
/// (Module, resolved-symbol).
///
/// NB the definition-name guard is the shared [`is_definition_name`] (parent's
/// `name:` field only).
/// Objective-C class / protocol / method names sit on an anonymous first
/// `identifier` child (no `name:` field), so they are NOT treated as definition
/// names: every occurrence of a class / method name — including the one in
/// `@interface X` / `@implementation X` / a `method_definition` header — is a
/// USAGE candidate that resolves to its own def node (deduped to one edge). This
/// is why a class self-reference (`Circle` → Class Circle) and a same-file method
/// name (`area` → Method) each yield a USAGE.
fn objc_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    const CALL_KINDS: &[&str] = &["message_expression", "call_expression"];
    const IMPORT_KINDS: &[&str] = &["preproc_include", "preproc_import"];
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if matches!(node.kind(), "identifier" | "type_identifier")
            && !is_inside_kind(node, CALL_KINDS)
            && !is_inside_kind(node, IMPORT_KINDS)
            && !is_definition_name(node)
        {
            let name = node_text(source, node);
            if !name.is_empty() && !OBJC_KEYWORDS.contains(&name) {
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::__ref__::{name}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "ref_name": name }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

// ---------------------------------------------------------------------------
// Racket (registry language, bespoke `extract_racket`)
// ---------------------------------------------------------------------------
//
// `tree-sitter-racket` is a generic s-expression grammar: every parenthesised
// form is a `list`, every atom a `symbol`, the file root is `program`, and no
// def node exposes a `name:` field. Racket is handled exactly like
// Clojure/Scheme, so this pass is `extract_clojure` re-pointed at
// Racket's node kinds (`list`/`symbol`, not `list_lit`/`sym_lit`) and the
// Racket-relevant def-head set.
// `require`→Module IMPORTS resolve
// path-to-Module, a mechanism the shared name-based IMPORTS resolver does
// not model (out of scope), so no IMPORTS edge is
// emitted here.

/// The recognised def-form heads. This is the full set,
/// shared across Clojure/Racket/Scheme; the Clojure-only
/// heads (`defn`, `defrecord`, …) never appear in `.rkt` and the Racket-only
/// heads (`define`, `struct`, …) never appear in `.clj`, so a single shared
/// list works without a per-language subset. Only heads that occur in
/// Racket are exercised.
const RACKET_DEF_HEADS: [&str; 19] = [
    "defn",
    "defn-",
    "def",
    "defmacro",
    "defmulti",
    "defmethod",
    "defprotocol",
    "defrecord",
    "deftype",
    "definterface",
    "defonce",
    "define",
    "define-syntax",
    "define-values",
    "define-syntax-rule",
    "define-struct",
    "define-record-type",
    "define/contract",
    "struct",
];

/// The label for a Racket def head (identical rule to Clojure's, but
/// keyed on the Racket heads): `struct` / `define-struct` / `define-record-type`
/// (plus the Clojure `defrecord` / `deftype`) → "Struct";
/// `definterface` / `defprotocol` → "Interface"; every other def head →
/// "Function". Racket in practice only ever hits the "Struct"/"Function" arms.
fn racket_def_label(head: &str) -> &'static str {
    match head {
        "struct" | "define-struct" | "define-record-type" | "defrecord" | "deftype" => "Struct",
        "definterface" | "defprotocol" => "Interface",
        _ => "Function",
    }
}

/// Fortran registry extractor for the
/// `fortran_small` fixture.
///
/// Base: the generic spec path already emits the 19 module-procedure
/// `Function` nodes (procedures are free Functions — the
/// `module` node exposes no owner name). But it ALSO emits nodes we do NOT want:
///   * one `Module` per `module_statement` — the only module kind we keep
///     is `translation_unit`, i.e. the per-file `__file__` Module the indexer's
///     structural pass already adds (10 Modules = 10 files, not 10+8).
///   * one `Type` per `derived_type_statement` — a derived-type
///     class yields ZERO nodes in this grammar (its name resolver finds no
///     name on that node kind), so no Class/Struct/Type/
///     Enum is emitted for derived types at all.
/// So we KEEP the spec's Function nodes and DROP every Module/Type node.
///
/// Edges: the spec CALLS pass emits nothing (a `call_expression` is a sibling of
/// the header `function_statement`, so the callable-ancestor walk fails). The
/// CALLS / USAGE passes instead source both from the per-file Module:
///   * every `call_expression` (a `foo(...)` function reference — NOT a `call`
///     statement, which is a distinct grammar node) → a `CALLS` edge from the
///     file Module to the callee `Function`, resolved by short name.
///   * every `subroutine_call` (`call sub(...)`) → a `USAGE` edge from the file
///     Module to the referenced `Function` (the call kinds exclude
///     `subroutine_call`, so it surfaces as a generic reference/usage, not a
///     call). Both dedup by (source, target) via the indexer's ON CONFLICT
///     upsert.
fn extract_fortran(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let queries = d
        .compiled_queries()
        .map_err(|e| greppy_core::Error::Parse(format!("compile {} queries: {e}", d.name)))?;
    let mut result =
        crate::spec::spec_extract(Language::Registered(d), d.spec, queries, source, file_path)?;

    // Keep only the spec's `Function` nodes (the 19 module procedures). Drop the
    // `module_statement` Modules and the `derived_type_statement` Types — we emit
    // neither (see doc comment above). Drop all spec edges: the spec CALLS pass
    // produces none for Fortran, and we emit the CALLS/USAGE edges below.
    result.nodes.retain(|n| n.label == "Function");
    result.edges.clear();

    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let file_module_qname = format!("{file_path}::__file__");

    fortran_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    fortran_usages_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// The callee `identifier` text of a Fortran call node (its first `identifier`
/// child — the leading name in `foo(args)` / `call foo(args)`), or `None`.
fn fortran_call_callee<'a>(source: &'a [u8], call: Node<'_>) -> Option<&'a str> {
    let name = find_child_of_kind(call, "identifier")?;
    let text = node_text(source, name);
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// CALLS pass for Fortran (call kinds =
/// {call_expression, keyword_argument, call} — the resolvable head being the
/// `call_expression`'s leading `identifier`). Every `call_expression` is a
/// function reference `foo(...)`; the callee is its leading `identifier`. The
/// source is ALWAYS the per-file Module node (`{file}::__file__`): a call's
/// nearest enclosing captured def is the header `function_statement`, but the
/// call lives in the sibling body, so the enclosing-func walk falls back to the
/// file Module. The target is a
/// same-file `{file}::Function::{callee}` direct qname; the `callee_name`
/// property lets the shared resolver pick up a project-wide unique cross-file
/// callee (Fortran procedure names are global, so a `use`d procedure resolves by
/// bare name). Builtins with no Function def
/// (`sqrt`, …) resolve to nothing and drop out.
fn fortran_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "call_expression" {
            if let Some(callee) = fortran_call_callee(source, node) {
                result.edges.push(ExtractedEdge {
                    edge_type: "CALLS".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::Function::{callee}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "callee_text": callee,
                        "callee_name": callee,
                    }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// USAGE pass for Fortran. A `call sub(...)` statement
/// parses as a `subroutine_call` node — which is NOT in the call kinds
/// (`{call_expression, keyword_argument, call}`), so it is never a CALLS site.
/// Its callee `identifier` is instead a generic reference, resolved
/// against every registered def (Function/Method/… — Fortran
/// procedures are Functions). Emit one `USAGE` edge per `subroutine_call` from
/// the per-file Module to the referenced name; the shared resolver keys on
/// `ref_name` and links same-file first, then project-wide unique. Dedup by
/// (source, target) is the indexer's ON CONFLICT upsert.
fn fortran_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "subroutine_call" {
            if let Some(name) = fortran_call_callee(source, node) {
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::__ref__::{name}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({
                        "ref_name": name,
                    }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

fn extract_racket(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    racket_defs_pass(source, root, file_path, &mut result);
    racket_calls_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// The head symbol text of a Racket `list` (its first named `symbol` child's
/// text), or `None` when the first named child is not a symbol. The full symbol
/// text is returned so it can be tested against the def-head table verbatim.
fn racket_head_sym<'a, 't>(source: &'a [u8], list: Node<'t>) -> Option<(&'a str, Node<'t>)> {
    let head = list.named_child(0)?;
    if head.kind() != "symbol" {
        return None;
    }
    Some((node_text(source, head), head))
}

/// DEFS pass for Racket. Walks every
/// `list` (descending into each), and for a form whose head
/// symbol is a def head emits ONE node with a Struct/Interface/Function label.
/// The name target is the second named child; when that child is itself a `list`
/// (`(define (foo args) ..)` / `(struct foo (fields) ..)`'s function-shape name)
/// the name is that list's head symbol, else the symbol itself.
fn racket_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "list" && node.named_child_count() >= 2 {
            if let Some((head_text, _)) = racket_head_sym(source, node) {
                if RACKET_DEF_HEADS.contains(&head_text) {
                    if let Some(target) = node.named_child(1) {
                        // (define (foo args) ..) — the name is the head symbol of
                        // the nested signature list.
                        let name_node = if target.kind() == "list" {
                            target.named_child(0)
                        } else {
                            Some(target)
                        };
                        if let Some(name_node) = name_node {
                            let name = node_text(source, name_node);
                            if !name.is_empty() {
                                let label = racket_def_label(head_text);
                                result.nodes.push(ExtractedNode {
                                    label: label.into(),
                                    name: name.to_string(),
                                    qualified_name: format!("{file_path}::{label}::{name}"),
                                    file_path: file_path.to_string(),
                                    start_line: node.start_position().row as u32 + 1,
                                    end_line: node.end_position().row as u32 + 1,
                                    properties: serde_json::json!({}),
                                });
                            }
                        }
                    }
                }
            }
        }
        // Descend into every form's children so nested defs are reached.
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// CALLS pass for Racket: every
/// `list` (the sole call kind) whose head is a `symbol` is a call whose
/// callee is that head symbol. The callee resolves against project
/// Function/Method defs, so def-form heads (`define`, `struct`, …), special
/// forms (`if`, `cond`, `let`, …), operators, and unresolved names simply drop
/// out. The source is always the per-file Module node (Racket has no function
/// node kinds, so the enclosing func is always the module). The
/// resolver is keyed on the short name (text after the last `.`) so the
/// same-file `{file}::Function::{callee}` direct qname resolves same-file, and
/// the `callee_name` property lets the shared resolver pick up a project-wide
/// unique cross-file callee (`square` required from `math.rkt`). The call graph
/// dedups by
/// (caller, callee); the indexer's `ON CONFLICT(source_id, target_id,
/// edge_type)` upsert reproduces that.
fn racket_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "list" {
            if let Some((head_text, _)) = racket_head_sym(source, node) {
                let callee = racket_call_short(head_text);
                if !callee.is_empty()
                    && !is_clojure_keyword(callee)
                    && !RACKET_DEF_HEADS.contains(&head_text)
                {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Function::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": head_text,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The resolver-facing short name of a Racket callee: the substring after the
/// LAST `.`, or the whole text when there is none. A namespaced/qualified symbol
/// keeps any `/` prefix (no split on `/`), so it never matches a bare `Function`
/// name.
fn racket_call_short(callee: &str) -> &str {
    match callee.rfind('.') {
        Some(idx) => &callee[idx + 1..],
        None => callee,
    }
}

// ---------------------------------------------------------------------------
// Scheme (registry language, bespoke `extract_scheme`)
// ---------------------------------------------------------------------------
//
// `tree-sitter-scheme` is the same generic s-expression grammar family as
// `tree-sitter-racket`: every parenthesised form is a `list`, every atom a
// `symbol`, the file root is `program`, and no def node exposes a `name:`
// field. Scheme is handled identically to Racket/Clojure
// (call kind `{"list"}`, var kind `{"symbol"}`,
// module kind `{"program"}`), so this pass is `extract_racket`
// re-pointed at Scheme's `LangDef`. The Racket helpers (`racket_defs_pass`,
// `racket_calls_pass`, `RACKET_DEF_HEADS`, `racket_def_label`) already key on
// exactly the `list`/`symbol` node kinds and the full def-head
// set, so they apply to Scheme verbatim.
//
// On `bench/agent_efficiency/corpus/scheme_small`: `define-record-type` →
// `Struct` (the only Struct head that occurs in idiomatic Scheme), every other
// def head incl. value bindings (`(define origin (make-point 0 0))`) →
// `Function`, and one `CALLS` from the file Module per applied same-file
// symbol. `(import (a) (b c) ..)` → one raw IMPORTS edge per import group; the
// shared `resolve_file_imports` pass links each whose bare module name matches
// a project File stem — a `(sub mod)` group's local name is the whole inner
// text, which never contains a `/`/`.`, so multi-symbol groups
// like `(util math)` do not match a file and drop.
fn extract_scheme(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    racket_defs_pass(source, root, file_path, &mut result);
    racket_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    scheme_imports_pass(source, root, &file_module_qname, file_path, &mut result);

    Ok(result)
}

/// IMPORTS pass for Scheme: walk every `list`, and for a form whose head symbol is
/// `import` / `require` / `use` / `load` / `include`, push one IMPORTS edge per
/// following module datum. A datum that is itself a `list` (`(util math)`,
/// `(stack)`) contributes its inner text (symbols joined by a single space, i.e.
/// the whole list text minus the parens); a bare `symbol` contributes its own
/// text. The edge's `imported_name` drives the shared `resolve_file_imports`
/// pass, which links it to a File when the name is a bare stem matching exactly
/// one project file — so single-symbol groups (`(stack)`) resolve and
/// multi-symbol groups (`(util math)`) drop (the resolver only splits on
/// `/`/`.`/`:`).
fn scheme_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "list" {
            if let Some((head_text, _)) = racket_head_sym(source, node) {
                if matches!(head_text, "import" | "require" | "use" | "load" | "include") {
                    let mut c = node.walk();
                    for (i, item) in node.named_children(&mut c).enumerate() {
                        if i == 0 {
                            continue; // skip the leading import head
                        }
                        let module = scheme_import_module_name(source, item);
                        if let Some(module) = module {
                            if !module.is_empty() {
                                result.edges.push(ExtractedEdge {
                                    edge_type: "IMPORTS".into(),
                                    source_qualified_name: file_module_qname.to_string(),
                                    target_qualified_name: format!(
                                        "{file_path}::__import__::{module}"
                                    ),
                                    file_path: file_path.to_string(),
                                    line: item.start_position().row as u32 + 1,
                                    properties: serde_json::json!({
                                        "imported_name": module,
                                        "module_path": module,
                                        "local_name": module,
                                    }),
                                });
                            }
                        }
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The module name of one Scheme import datum:
///   * a bare `symbol` (`(require util)`) → its own text;
///   * a nested `list` (`(util math)`, `(stack)`) → the inner text with parens
///     removed, i.e. the child symbols joined by a single space. This is the
///     raw slice between the parens; for the space-separated s-expression forms
///     used here that equals the symbols joined by " ".
fn scheme_import_module_name(source: &[u8], item: Node<'_>) -> Option<String> {
    match item.kind() {
        "symbol" => Some(node_text(source, item).to_string()),
        "list" => {
            let mut c = item.walk();
            let parts: Vec<&str> = item
                .named_children(&mut c)
                .filter(|n| n.kind() == "symbol")
                .map(|n| node_text(source, n))
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" "))
            }
        }
        _ => None,
    }
}

// ===========================================================================
// D — bespoke pass.
// ===========================================================================
//
// D is modelled as:
//
//   * **Class / Interface / Enum** — the class kinds route
//     `class_declaration` / `struct_declaration` / `union_declaration` through
//     the class-def path, labelling them all "Class";
//     `interface_declaration` → "Interface"; `enum_declaration` → "Enum".
//     (`module_declaration` / `module_def` are also in the set but resolve no
//     plain name — the name is a dotted `module_fqn`, not a bare `identifier`
//     child — so they emit no def node; the only "Module" is the per-file
//     synthetic node.) The def name is the container's direct `identifier`
//     child.
//   * **Function** — the func kinds route `function_declaration` through
//     the func-def path (a free "Function"). D members lack a `name:` owner
//     field, so no Method is keyed: every `function_declaration` — free OR a
//     class/struct/interface member — becomes a free Function whose qname
//     carries NO owner segment (`{module}.{name}`). Two same-named methods in
//     one file therefore collapse to one node in the store (this is why
//     `area`/`name`, defined on both Circle and Rect, count once each).
//     `constructor` / `destructor` are also func kinds, but their name is the
//     `this`/`~this` keyword, which is dropped, so a `this` constructor emits no
//     Function.
//   * **DEFINES** (File→def) is auto-derived by the structural pass from
//     the node set above (plus the per-file Module and README Section), so this
//     pass emits none.
//   * **CALLS** — a `call_expression` / `new_expression` head identifier. The
//     source is the per-file Module node (`{file}::__file__`); the callee
//     resolves same-file by the direct `{file}::Function::{callee}` qname or
//     cross-file / to a Class (constructor `new C()` / struct literal `S()`) by
//     the unique `callee_name`.
//   * **USAGE** — every `identifier` reference that is not inside a call or an
//     import and is not a keyword, sourced from the per-file Module and resolved
//     by name to a project definition (Function / Class / Interface / Enum).
//   * **IMPORTS** — one per `import_declaration`, sourced from the per-file
//     Module. The `resolve_file_imports` pass links the importer to the
//     imported FILE by its bare stem (last path segment), yielding one
//     per-file IMPORTS edge on this fixture.
//
// Cross-file INHERITS/IMPLEMENTS and README markdown Heading/Section nodes are
// out of this pass's scope; none are emitted here.

/// The label for a D class kind.
fn d_class_label(kind: &str) -> &'static str {
    match kind {
        "interface_declaration" => "Interface",
        "enum_declaration" => "Enum",
        // class_declaration / struct_declaration / union_declaration
        _ => "Class",
    }
}

/// The container kinds that become a Class/Interface/Enum node (their name is a
/// direct `identifier` child).
const D_TYPE_KINDS: [&str; 5] = [
    "class_declaration",
    "struct_declaration",
    "union_declaration",
    "interface_declaration",
    "enum_declaration",
];

/// The def NAME of a D container / function: its first direct `identifier`
/// child (the grammar puts the name there — the return type is a `type`,
/// parameters live under `parameters`, base types under `base_class`).
fn d_def_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut c = node.walk();
    let found = node
        .named_children(&mut c)
        .find(|ch| ch.kind() == "identifier")
        .map(|n| node_text(source, n));
    found
}

fn extract_d(
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(Language::Registered(d), source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_module_qname = format!("{file_path}::__file__");

    d_defs_pass(source, root, file_path, &mut result);
    // An enum-name reference should not produce a USAGE edge (only
    // Function/Method/Class/Interface are USAGE targets). The shared USAGE
    // resolver DOES resolve to Enum labels, so the names of enums
    // defined in this file are collected and skipped in the USAGE pass.
    let enum_names: std::collections::HashSet<String> = result
        .nodes
        .iter()
        .filter(|n| n.label == "Enum")
        .map(|n| n.name.clone())
        .collect();
    // A same-file `name → label` map so a `class C : Base` base can be routed
    // to IMPLEMENTS (Base is an Interface) or INHERITS (Base is a Class),
    // keying on the resolved base label.
    let type_labels: std::collections::HashMap<String, String> = result
        .nodes
        .iter()
        .filter(|n| matches!(n.label.as_str(), "Class" | "Interface" | "Enum"))
        .map(|n| (n.name.clone(), n.label.clone()))
        .collect();
    d_calls_pass(source, root, &file_module_qname, file_path, &mut result);
    d_usages_pass(
        source,
        root,
        &file_module_qname,
        file_path,
        &enum_names,
        &mut result,
    );
    d_imports_pass(source, root, &file_module_qname, file_path, &mut result);
    d_inherits_pass(source, root, file_path, &type_labels, &mut result);

    Ok(result)
}

/// Emit an IMPLEMENTS / INHERITS edge from each type with a `base_class` to the
/// base type. The base is the `identifier`
/// inside a `base_class` child. The resolver keys these edges on the target
/// qname directly, so the target is the same-file guess qname
/// `{file}::{Label}::{base}`: when the base is a same-file Interface the edge is
/// IMPLEMENTS (`{file}::Interface::{base}`), otherwise INHERITS
/// (`{file}::Class::{base}`). A genuinely cross-file base resolves only if it is
/// declared in this file (only same-file conformances that
/// a `by_qname` target reaches are captured).
fn d_inherits_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    type_labels: &std::collections::HashMap<String, String>,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if D_TYPE_KINDS.contains(&node.kind()) {
            if let Some(name) = d_def_name(source, node) {
                let src_label = d_class_label(node.kind());
                let src_qname = format!("{file_path}::{src_label}::{name}");
                let mut c = node.walk();
                for child in node.named_children(&mut c) {
                    if child.kind() != "base_class" {
                        continue;
                    }
                    if let Some(base) = d_last_identifier(source, child) {
                        if base.is_empty() || base == name {
                            continue;
                        }
                        // Route on the base's same-file label: Interface →
                        // IMPLEMENTS, anything else (Class / unknown) → INHERITS.
                        let (edge_type, base_label) =
                            match type_labels.get(base).map(String::as_str) {
                                Some("Interface") => ("IMPLEMENTS", "Interface"),
                                _ => ("INHERITS", "Class"),
                            };
                        result.edges.push(ExtractedEdge {
                            edge_type: edge_type.into(),
                            source_qualified_name: src_qname.clone(),
                            target_qualified_name: format!("{file_path}::{base_label}::{base}"),
                            file_path: file_path.to_string(),
                            line: child.start_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "name": base,
                                "trait_name": base,
                                "type_name": name,
                            }),
                        });
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit one node per D definition. Container kinds
/// (class/struct/union/interface/enum) → Class/Interface/Enum; every
/// `function_declaration` (free or member) → a free Function keyed
/// `{file}::Function::{name}` (no owner segment, so same-named members across
/// types collapse in the store). Constructors emit nothing.
fn d_defs_pass(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        if D_TYPE_KINDS.contains(&kind) {
            if let Some(name) = d_def_name(source, node) {
                if !name.is_empty() {
                    let label = d_class_label(kind);
                    result.nodes.push(ExtractedNode {
                        label: label.into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::{label}::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
        } else if kind == "function_declaration" {
            if let Some(name) = d_def_name(source, node) {
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Function".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Function::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
        }
        // Descend into every node (class/struct bodies included) so member
        // functions are reached — but not into a `function_declaration`'s own
        // body (function bodies are not recursed into, so a local nested
        // `function_declaration` is not a def).
        if kind == "function_declaration" {
            continue;
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The callee NAME of a D call node. For a
/// `call_expression` / `function_call_expression` the callee is the head child
/// (an `identifier` for a bare call, or the trailing member of a `.`-qualified
/// receiver call). For a `new_expression` (`new Circle(..)`) the callee is the
/// constructed type identifier.
fn d_callee_name<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    match node.kind() {
        "new_expression" => {
            // `new Circle(args)` — the type is the first `identifier`/`type`
            // descendant; take the last `identifier` leaf of the type.
            let mut c = node.walk();
            let found = node
                .named_children(&mut c)
                .find_map(|ch| d_last_identifier(source, ch));
            found
        }
        // call / function_call: the head is the first named child; its trailing
        // identifier is the callee name.
        _ => node
            .named_child(0)
            .and_then(|h| d_last_identifier(source, h)),
    }
}

/// The trailing `identifier` leaf of a call head: for a bare `identifier` it is
/// the node itself; for a member/qualified head (`a.b.f`) it is the last
/// `identifier` descendant. Returns `None` for non-identifier heads (literals,
/// operators) that never name a resolvable definition.
fn d_last_identifier<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    if node.kind() == "identifier" {
        return Some(node_text(source, node));
    }
    // Walk descendants and keep the last identifier (deepest trailing member).
    let mut last: Option<&str> = None;
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        if n.kind() == "identifier" {
            last = Some(node_text(source, n));
        }
        let mut c = n.walk();
        for ch in n.named_children(&mut c) {
            stack.push(ch);
        }
    }
    last
}

/// The D call node kinds.
const D_CALL_KINDS: [&str; 3] = [
    "call_expression",
    "function_call_expression",
    "new_expression",
];

/// Emit CALLS edges from the per-file Module to each call's callee.
/// Same-file callees resolve by the direct
/// `{file}::Function::{callee}` qname; cross-file / Class callees resolve by the
/// unique `callee_name`.
fn d_calls_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if D_CALL_KINDS.contains(&node.kind()) {
            if let Some(callee) = d_callee_name(source, node) {
                if !callee.is_empty() && !d_is_keyword(callee) {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Function::{callee}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "callee_text": callee,
                            "callee_name": callee,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit USAGE edges: every `identifier`
/// reference that is NOT inside a call or an import declaration and is not a
/// keyword. The source is the per-file Module; the reference resolves by name to
/// a project definition (Function / Class / Interface / Enum).
fn d_usages_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    enum_names: &std::collections::HashSet<String>,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "identifier"
            && !d_is_inside(node, &D_CALL_KINDS)
            && !d_is_inside(node, &["import_declaration", "module_declaration"])
        {
            let refname = node_text(source, node);
            if !refname.is_empty() && !d_is_keyword(refname) && !enum_names.contains(refname) {
                result.edges.push(ExtractedEdge {
                    edge_type: "USAGE".into(),
                    source_qualified_name: file_module_qname.to_string(),
                    target_qualified_name: format!("{file_path}::__ref__::{refname}"),
                    file_path: file_path.to_string(),
                    line: node.start_position().row as u32 + 1,
                    properties: serde_json::json!({ "ref_name": refname }),
                });
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit one IMPORTS edge per `import_declaration`, sourced from the per-file
/// Module. The dotted module path (`util.math`) is reduced to its bare stem
/// (`math`) in `imported_name` so the `resolve_file_imports` pass links the
/// importer to the imported FILE (one IMPORTS edge per import).
fn d_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_module_qname: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "import_declaration" {
            // The imported module path is the `module_fqn` under `imported`.
            if let Some(fqn) = d_find_descendant(node, "module_fqn") {
                let full = node_text(source, fqn);
                let stem = full.rsplit('.').next().unwrap_or(full);
                if !stem.is_empty() {
                    result.edges.push(ExtractedEdge {
                        edge_type: "IMPORTS".into(),
                        source_qualified_name: file_module_qname.to_string(),
                        target_qualified_name: format!("{file_path}::__import__::{stem}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: serde_json::json!({
                            "imported_name": stem,
                            "module_path": full,
                            "local_name": stem,
                        }),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// The first descendant of `node` with the given kind (breadth-first).
fn d_find_descendant<'t>(node: Node<'t>, kind: &str) -> Option<Node<'t>> {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        let mut c = n.walk();
        for ch in n.named_children(&mut c) {
            if ch.kind() == kind {
                return Some(ch);
            }
            stack.push(ch);
        }
    }
    None
}

/// True if `node` sits inside an ancestor of one of `kinds`, within a
/// 10-ancestor depth bound.
fn d_is_inside(node: Node<'_>, kinds: &[&str]) -> bool {
    const MAX_PARENT_DEPTH: usize = 10;
    let mut cur = node.parent();
    let mut depth = 0;
    while let Some(n) = cur {
        if depth >= MAX_PARENT_DEPTH {
            break;
        }
        if kinds.contains(&n.kind()) {
            return true;
        }
        cur = n.parent();
        depth += 1;
    }
    false
}

/// D keywords / built-in type names that a bare `identifier` reference must not
/// count as a USAGE / CALLS name (they never name a user definition).
fn d_is_keyword(name: &str) -> bool {
    matches!(
        name,
        "this"
            | "super"
            | "return"
            | "if"
            | "else"
            | "for"
            | "foreach"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "new"
            | "delete"
            | "import"
            | "module"
            | "class"
            | "struct"
            | "interface"
            | "union"
            | "enum"
            | "auto"
            | "void"
            | "int"
            | "uint"
            | "long"
            | "ulong"
            | "short"
            | "ushort"
            | "byte"
            | "ubyte"
            | "bool"
            | "float"
            | "double"
            | "real"
            | "char"
            | "wchar"
            | "dchar"
            | "string"
            | "true"
            | "false"
            | "null"
            | "cast"
            | "typeof"
            | "sizeof"
            | "in"
            | "out"
            | "ref"
            | "const"
            | "immutable"
            | "static"
            | "public"
            | "private"
            | "protected"
    )
}

fn extract_zig(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::Zig)
        .map_err(|e| greppy_core::Error::Parse(format!("compile zig queries: {e}")))?;
    // Base pass. The spec engine emits the per-file Module node, one "Function"
    // per `function_declaration` (free AND member — tree-sitter-zig's unnamed
    // `struct_declaration` / `enum_declaration` /
    // `union_declaration` container nodes cannot be named, so the class-def path
    // pushes no Class/Enum/Field and re-walks the
    // container's `function_declaration`s at file scope — every method too), and
    // the IMPORTS pass. The spec CALLS query captures as `@zig_call` (not
    // `@callee`), so `spec_calls` is a no-op; Zig owns CALLS below. What the
    // uniform template does NOT cover is added
    // below: `test_declaration` → an additional "Function"; every top-level
    // `variable_declaration` → a "Variable" (`const X =
    // struct{…}`, `const std = @import(…)`, `var`, `const` are ALL Variables —
    // Zig's only module-scope def kind besides Function); the CALLS pass (bare
    // and member `recv.method()` callees); and the USAGE pass.
    let mut result =
        crate::spec::spec_extract(Language::Zig, &crate::spec::ZIG, queries, source, file_path)?;

    let tree = crate::parse(Language::Zig, source)?;
    let root = tree.root_node();
    let file_module_qname = format!("{file_path}::__file__");

    // MODULE / IMPORT PASS.
    //
    // The shared import resolver only targets importable definition labels,
    // not synthetic Module nodes. Model each Zig source file as an importable
    // Class namespace named by its file stem, normalize `.zig` import basenames
    // to that stem, and materialize extensionless package imports (for example
    // `std`) as external namespace stubs. This keeps the fix local to Zig while
    // giving every @import edge a real graph target.
    emit_zig_file_namespace(root, file_path, &mut result);
    normalize_zig_imports(file_path, &mut result);

    // FUNCTION (test) PASS.
    //
    // The Zig func kinds = {function_declaration, test_declaration,
    // function_signature}. `function_declaration`s are already emitted by the
    // spec query; `test_declaration`s are not (they have no `name:` identifier —
    // they are named from the test string). Emit one
    // "Function" per `test_declaration`, named by its string literal.
    emit_zig_test_functions(source, root, file_path, &mut result);

    // VARIABLE PASS.
    //
    // The variable pass (name
    // field → declarator → first identifier child). Zig's variable kind
    // is {variable_declaration}; only the file root's *module-level*
    // `variable_declaration` children are candidates (locals inside function
    // bodies are never module vars). The name is
    // the declaration's first `identifier` child (the bound name).
    // `_` and empty names are dropped.
    emit_zig_variables(source, root, file_path, &mut result);

    // CALLS PASS.
    //
    // The CALLS pass (zig call kinds =
    // {call_expression, builtin_function}). For every `call_expression`, the
    // callee is the `function:` field: a bare `identifier` (`helper(...)`) or a
    // `field_expression` (`recv.method(...)` / `mod.func(...)`), whose LAST
    // identifier segment is the callee name resolved against
    // Function/Method defs (member-call `.method()` and module-qualified
    // `mod.func()` both key on the trailing name). `builtin_function`
    // (`@import`, `@intCast`, …) has no `function:` field and the generic
    // fallback finds no `identifier` child (child 0 is `builtin_identifier`), so
    // builtins yield no CALLS — achieved by only walking `call_expression`.
    // Source = nearest enclosing `function_declaration` / `test_declaration`
    // qname (all `{file}::Function::{name}`, since Zig methods flatten to
    // Functions), else the file Module node. CALLS dedup by (caller, callee).
    emit_zig_calls(source, root, file_path, &file_module_qname, &mut result);

    // USAGE PASS.
    //
    // The USAGE pass. Zig has no
    // language-specific reference kind, so a
    // reference is a bare `identifier` / `type_identifier`. Emit a USAGE unless
    // it sits inside a `call_expression` / `builtin_function` (a callee or
    // call-argument reference is suppressed) or a
    // `builtin_function` (an `@import` argument), is a
    // definition *name* (the `name:` field of its parent), or is a generic
    // keyword. `ref_name` is resolved
    // project-wide by the indexer (same-file preference, then uniqueness) against
    // Variable/Function/… defs, so an unresolvable reference (e.g. a
    // `container_field` name that is never emitted as a node) is dropped by the
    // registry lookup. Source = enclosing function qname, else Module.
    emit_zig_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// The importable namespace name for a Zig source path or `@import` path.
/// `foo/bar.zig` becomes `bar`; extensionless package names such as `std`
/// remain unchanged.
fn zig_module_basename(path: &str) -> &str {
    let basename = path.rsplit(['/', '\\']).next().unwrap_or(path);
    basename.strip_suffix(".zig").unwrap_or(basename)
}

/// Emit the current source file's importable namespace. The shared resolver
/// intentionally excludes synthetic Module nodes, so Zig uses the same
/// importable-Class technique as other language-specific module passes.
fn emit_zig_file_namespace(
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let module_name = zig_module_basename(file_path);
    if module_name.is_empty() {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Class".into(),
        name: module_name.to_string(),
        qualified_name: format!("{file_path}::Class::{module_name}"),
        file_path: file_path.to_string(),
        start_line: 1,
        end_line: root.end_position().row as u32 + 1,
        properties: serde_json::json!({ "kind": "zig_file_namespace" }),
    });
}

/// Normalize Zig IMPORTS keys to the imported module basename without the
/// `.zig` suffix. Project files resolve to the Class namespace emitted above;
/// extensionless packages such as `std` get a local external namespace stub so
/// their IMPORTS edge remains navigable even though the package is outside the
/// indexed repository.
fn normalize_zig_imports(file_path: &str, result: &mut ExtractionResult) {
    let mut external_modules = std::collections::BTreeSet::new();
    for edge in result
        .edges
        .iter_mut()
        .filter(|edge| edge.edge_type == "IMPORTS")
    {
        let Some(path) = edge
            .properties
            .get("path")
            .and_then(|value| value.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        let imported_name = zig_module_basename(&path);
        if imported_name.is_empty() {
            continue;
        }
        if let Some(properties) = edge.properties.as_object_mut() {
            properties.insert("imported_name".into(), serde_json::json!(imported_name));
        }
        if !path.ends_with(".zig") {
            external_modules.insert(imported_name.to_string());
        }
    }

    for module_name in external_modules {
        result.nodes.push(ExtractedNode {
            label: "Class".into(),
            name: module_name.clone(),
            qualified_name: format!("{file_path}::Class::{module_name}"),
            file_path: file_path.to_string(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({ "kind": "zig_external_module" }),
        });
    }
}

/// The nearest enclosing Zig callable qname for `node`, keyed on the func kinds
/// {function_declaration,
/// test_declaration}: the closest such ancestor, named by its `name:` field
/// (`function_declaration`) or its test string (`test_declaration`). Because the
/// class-qn branch cannot name tree-sitter-zig's unnamed container nodes, EVERY
/// Zig function — free or a struct/enum/union method — is qualified as
/// `{file}::Function::{name}`, matching the flattened Function nodes. Returns
/// `None` at file / container scope (the caller substitutes the file Module
/// qname), which is the source of those CALLS / USAGE edges.
fn zig_enclosing_qname(source: &[u8], node: Node<'_>, file_path: &str) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        match cur.kind() {
            "function_declaration" => {
                let name = cur
                    .child_by_field_name("name")
                    .map(|n| node_text(source, n))?;
                return Some(format!("{file_path}::Function::{name}"));
            }
            // A `test_declaration` IS an enclosing-func kind, but the qname is
            // read from the `name:` FIELD — a `test_declaration` has none (its
            // DEF name is derived separately from the test string). So the
            // name lookup returns nothing and the enclosing scope falls back to
            // the file Module node. Return `None` (→ Module) once the
            // nearest enclosing callable is a test: a call / usage inside a test
            // body is sourced from the Module, not the test Function.
            "test_declaration" => return None,
            _ => {}
        }
        p = cur.parent();
    }
    None
}

/// The name of a `test_declaration` — its string literal's `string_content`.
/// A nameless `test { … }` has no string child and
/// is skipped (returns `None`).
fn zig_test_name<'a>(source: &'a [u8], test_node: Node<'_>) -> Option<&'a str> {
    let mut c = test_node.walk();
    let string_node = test_node
        .named_children(&mut c)
        .find(|n| n.kind() == "string")?;
    let content = find_child_of_kind(string_node, "string_content")?;
    Some(node_text(source, content))
}

/// Emit a "Function" node for every `test_declaration` (the func kinds
/// include it, named from the test string). Named tests only.
fn emit_zig_test_functions(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "test_declaration" {
            if let Some(name) = zig_test_name(source, node) {
                if !name.is_empty() {
                    result.nodes.push(ExtractedNode {
                        label: "Function".into(),
                        name: name.to_string(),
                        qualified_name: format!("{file_path}::Function::{name}"),
                        file_path: file_path.to_string(),
                        start_line: node.start_position().row as u32 + 1,
                        end_line: node.end_position().row as u32 + 1,
                        properties: serde_json::json!({}),
                    });
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit a "Variable" node for each *module-level* `variable_declaration`
/// (only the file root's
/// direct children are module vars — a `variable_declaration` inside a function
/// body or a container body is a local / field, not a module Variable). The name
/// is the declaration's first `identifier` child;
/// `_` / empty names are dropped.
fn emit_zig_variables(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let mut c = root.walk();
    for child in root.named_children(&mut c) {
        if child.kind() != "variable_declaration" {
            continue;
        }
        let Some(name_node) = find_child_of_kind(child, "identifier") else {
            continue;
        };
        let vname = node_text(source, name_node);
        if vname.is_empty() || vname == "_" {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: vname.to_string(),
            qualified_name: format!("{file_path}::Variable::{vname}"),
            file_path: file_path.to_string(),
            start_line: child.start_position().row as u32 + 1,
            end_line: child.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// The callee of a Zig `call_expression`, as `(full_text, resolve_name)`. A bare
/// `identifier` (`helper`) yields `("helper", "helper")`; a `field_expression`
/// (`recv.method` / `mod.func`) yields the full dotted text (`"math.sub"`) plus
/// its trailing `identifier` (`"sub"`).
///
/// The split matters for the keyword filter: the keyword test runs on the FULL
/// callee text (`math.sub`, never the bare segment), so a qualified call whose
/// method happens to be a generic keyword (`math.sub` — `sub` IS a generic
/// keyword) is NOT filtered; only a genuinely bare keyword callee is. The
/// resolver then keys on the trailing segment. Returns `None` for any other
/// callee shape (a `builtin_function` has no `function:` field, so `@import` /
/// `@intCast` yield no call — the generic fallback finds no `identifier` child).
fn zig_callee_name<'a>(source: &'a [u8], call: Node<'_>) -> Option<(&'a str, &'a str)> {
    let func = call.child_by_field_name("function")?;
    match func.kind() {
        "identifier" => {
            let t = node_text(source, func);
            Some((t, t))
        }
        "field_expression" => {
            let full = node_text(source, func);
            let mut last = None;
            let mut c = func.walk();
            for ch in func.named_children(&mut c) {
                if ch.kind() == "identifier" {
                    last = Some(ch);
                }
            }
            last.map(|n| (full, node_text(source, n)))
        }
        _ => None,
    }
}

/// Emit `CALLS` edges for every Zig `call_expression`:
/// the source is the enclosing function/test qname or the file Module node; the
/// callee name is the trailing identifier of the `function:` callee (bare or
/// member). Generic keywords are skipped. The call graph is deduplicated by
/// (caller, callee), so two
/// call sites to the same callee from the same enclosing scope collapse to one
/// edge — via the per-`(source, callee)` dedup below.
fn emit_zig_calls(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "call_expression" {
            if let Some((full, callee)) = zig_callee_name(source, node) {
                // Keyword filter on the FULL callee text, so a
                // qualified `mod.method` whose trailing segment is a generic
                // keyword still counts; resolution keys on the trailing segment.
                if !full.is_empty() && !callee.is_empty() && !is_zig_keyword(full) {
                    let source_qname = zig_enclosing_qname(source, node, file_path)
                        .unwrap_or_else(|| file_module_qname.to_string());
                    // Dedup by (caller, callee) — the resolvable trailing name.
                    if seen.insert((source_qname.clone(), callee.to_string())) {
                        result.edges.push(ExtractedEdge {
                            edge_type: "CALLS".into(),
                            source_qualified_name: source_qname,
                            target_qualified_name: format!("{file_path}::Function::{callee}"),
                            file_path: file_path.to_string(),
                            line: node.start_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "callee_text": full,
                                "callee_name": callee,
                            }),
                        });
                    }
                }
            }
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// USAGE pass for Zig. Every
/// `identifier` / `type_identifier` reference emits a USAGE edge unless it is a
/// definition *name*, sits inside a call (`call_expression` / `builtin_function`
/// — already a CALLS edge or a builtin, and its nested references suppressed),
/// sits inside an import (`builtin_function`, i.e. `@import`'s argument), or is a
/// generic keyword. The `ref_name` is resolved project-wide by the indexer, so
/// the target qname is a placeholder that never resolves directly. The source is
/// the nearest enclosing callable qname, falling back to the per-file Module
/// node at file / container scope.
fn emit_zig_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let kind = node.kind();
    if matches!(kind, "identifier" | "type_identifier")
        && !is_inside_kind(node, &["call_expression", "builtin_function"])
        && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_zig_keyword(text) {
            let source_qname = zig_enclosing_qname(source, node, file_path)
                .unwrap_or_else(|| file_module_qname.to_string());
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({
                    "ref_name": text,
                }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        emit_zig_usages(source, child, file_path, file_module_qname, result);
    }
}

/// Zig keyword / literal filter — Zig runs through the generic
/// keyword table. A reference or callee whose
/// text is one of these never emits a usage / call.
fn is_zig_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "nil"
            | "None"
            | "undefined"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "struct"
            | "enum"
            | "interface"
            | "trait"
            | "impl"
            | "import"
            | "export"
            | "package"
            | "module"
            | "use"
            | "require"
            | "include"
            | "new"
            | "delete"
            | "this"
            | "self"
            | "super"
            | "public"
            | "private"
            | "protected"
            | "static"
            | "const"
            | "var"
            | "let"
            | "function"
            | "def"
            | "fn"
            | "func"
            | "fun"
            | "proc"
            | "sub"
            | "method"
            | "async"
            | "await"
            | "yield"
    )
}

// ===========================================================================
// Elm — registry language with a bespoke pass.
// ===========================================================================
//
// Elm is modelled as three families of definition node, walked by the def pass:
//
//   * func kinds = {`value_declaration`, `function_declaration`} →
//     "Function". The name is the `functionDeclarationLeft`
//     child's FIRST named identifier (a top-level `f a b = …` value). The def
//     walk STOPS after a func node, so it does NOT descend into the body — a
//     nested `let x = … in …` binding is therefore NOT a Function. Every
//     top-level binding (`origin = …`, `defaultWidth = …`, `sampleShapes = …`)
//     is a plain `value_declaration`, so each is one Function (Elm has no
//     separate top-level-variable node, so zero Variables are emitted
//     for Elm).
//   * class kinds = {`type_declaration`, `type_alias_declaration`,
//     `module_declaration`} → routed through the class-def path, labelled:
//     `type_alias_declaration` → "Type"; both
//     `type_declaration` (a custom `type Foo = A | B`) and `module_declaration`
//     (the file's `module Foo exposing (…)` header) → "Class". The name is the
//     `name:` field (`upper_case_identifier` for the two type kinds, the whole
//     `upper_case_qid` — e.g. `Math.Util` — for a module).
//
// Everything else is produced by the shared plumbing
// and is NOT this pass's job:
//   * File / Folder / Project / per-file Module structural nodes and the
//     DEFINES / CONTAINS_* edges are auto-derived by the indexer's structural
//     pass from the definition nodes above.
//   * The reference-node predicate has NO Elm arm (it falls to
//     `false`), so ZERO USAGE edges are emitted for Elm — this pass
//     emits none either.
//   * No CALLS edge resolves: an Elm call is a `function_call_expr` whose callee
//     sits on the `target:` field (a `value_expr`, not an `identifier`), so the
//     callee lookup — which only reads `function`/`name`/`method` fields
//     or a bare first-`identifier` child — returns NULL for every Elm call.
//     Zero resolved CALLS are emitted.
//   * IMPORTS: `import X` resolves to a sibling *file*'s Module node
//     (File/Module → Module). That is the `require`→File shape the shared
//     indexer deliberately does not resolve (its IMPORTS pass keys on
//     `IMPORTABLE_LABELS`, which excludes `Module`) — the same documented
//     carve-out as Ruby's `require` and Dart's `import` (out of scope), so none
//     are emitted here.
fn extract_elm(
    language: Language,
    _d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();

    elm_walk_defs(source, root, file_path, &mut result);

    Ok(result)
}

/// The Elm func kinds — routed through the func-def path (→ "Function").
const ELM_FUNC_KINDS: [&str; 2] = ["value_declaration", "function_declaration"];

/// The label for an Elm type/module declaration:
/// `type_alias_declaration` → "Type"; `type_declaration`
/// and `module_declaration` → "Class". Returns `None` for any other kind.
fn elm_class_label(kind: &str) -> Option<&'static str> {
    match kind {
        "type_alias_declaration" => Some("Type"),
        "type_declaration" | "module_declaration" => Some("Class"),
        _ => None,
    }
}

/// DEFS pass for Elm (explicit stack,
/// no recursion into function bodies). For each node:
///   * a `value_declaration` / `function_declaration` → emit a "Function", then
///     STOP (do not descend,
///     which is why nested `let … in` bindings are not extracted);
///   * a `type_declaration` / `type_alias_declaration` / `module_declaration` →
///     emit the "Class"/"Type" node, then descend into its children (a
///     `module_declaration` has no nested defs, and a `type_declaration` body is
///     just union variants, so this is only for completeness);
///   * anything else → descend into all named children.
fn elm_walk_defs(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        if ELM_FUNC_KINDS.contains(&kind) {
            elm_emit_function(source, node, file_path, result);
            // Do NOT descend into a func body, so `let`-bound locals are not
            // Functions.
            continue;
        }
        if let Some(label) = elm_class_label(kind) {
            elm_emit_type(source, node, label, file_path, result);
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// Emit a "Function" node for one Elm `value_declaration` / `function_declaration`.
/// The name is the
/// `functionDeclarationLeft` child's FIRST named identifier. A
/// `function_declaration` (rare in this grammar) carries a plain `name:` field.
/// Empty names are dropped.
fn elm_emit_function(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let name_node = elm_func_name_node(node);
    let Some(name_node) = name_node else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: "Function".into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::Function::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

/// C `resolve_elm_func_name`: the FIRST named child of the
/// `functionDeclarationLeft` field (or, failing the field, a
/// `function_declaration_left` child). Falls back to a plain `name:` field
/// (`function_declaration`).
fn elm_func_name_node(node: Node<'_>) -> Option<Node<'_>> {
    let fdl = node
        .child_by_field_name("functionDeclarationLeft")
        .or_else(|| named_child_of_kinds(node, &["function_declaration_left"]));
    if let Some(fdl) = fdl {
        if let Some(first) = fdl.named_child(0) {
            return Some(first);
        }
    }
    node.child_by_field_name("name")
}

/// Emit the "Class"/"Type" node for one Elm type/module declaration. The name is
/// the `name:` field text (`upper_case_identifier` for a type, the whole
/// `upper_case_qid` — e.g. `Math.Util` — for a module). Empty names are dropped.
fn elm_emit_type(
    source: &[u8],
    node: Node<'_>,
    label: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let name = node_text(source, name_node);
    if name.is_empty() {
        return;
    }
    result.nodes.push(ExtractedNode {
        label: label.into(),
        name: name.to_string(),
        qualified_name: format!("{file_path}::{label}::{name}"),
        file_path: file_path.to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        properties: serde_json::json!({}),
    });
}

// ===========================================================================
// Gleam — registry language handled with a thin
// bespoke layer on top of the generic `spec_extract`.
// ===========================================================================
//
// Gleam maps so closely onto the uniform definition template that the generic
// spec path already covers EVERY node label and most edges:
//
//   * Function definitions = {`function`, `anonymous_function`,
//     `external_function`} → "Function" (the spec's `DefRule::func("function")`;
//     the fixture uses no `anonymous_function` / `external_function`, and neither
//     does most Gleam).
//   * Type definitions = {`type_definition`, `type_alias`, `custom_type`} —
//     both `type_definition` (a `pub type Foo { .. }` custom type) and
//     `type_alias` (`pub type Money = Int`) → "Type" (the spec keys the type
//     DefRule on `type_name`, the name-bearing child of BOTH kinds). `custom_type`
//     is not a real node kind in this grammar, so it never fires.
//   * Variable candidates = {`let`, `constant`} — but ZERO Variables are emitted
//     from `.gleam` source: a module-level `pub const` carries no `name:` field a
//     top-level Variable resolver reads, and `let` bindings live inside function
//     bodies, which the definition walk never descends into. The spec has no
//     Variable DefRule for Gleam, so 0 Variables is the natural, not forced,
//     result.
//   * Field candidates = {`field`} name a *field* only; there is no `field`
//     NODE kind in the grammar, so 0 Field nodes are emitted.
//
// The File / Folder / Project / per-file Module structural spine and the
// DEFINES / CONTAINS_FILE / CONTAINS_FOLDER edges are auto-derived by the
// indexer's structural pass.
//
// Two edge families need the bespoke layer below:
//
//   * CALLS — the callee resolver reads a `function_call`'s `function:` field
//     only when it is a bare `identifier` (`with_tax(..)`); a module-qualified
//     call whose `function:` is a `field_access` (`string.append`, `int.to_string`)
//     resolves to no project Function and yields no edge. The generic CALLS query
//     already agrees (its `field_access` branch captures a name that resolves to
//     nothing). The ONE wrinkle: the generic pass keeps a self-recursion call as a
//     `caller → caller` self-loop; we instead treat self-recursion as a node
//     property and emit NO self-loop CALLS edge. So we run the generic CALLS pass,
//     then drop the self-loops (source qname == target qname).
//   * IMPORTS — one edge per `import <local module>` clause. The shared resolver
//     keys IMPORTS on `imported_name` against `IMPORTABLE_LABELS`, which
//     (deliberately) excludes `Module`, so an import→Module target can never
//     resolve there. Gleam's `import mod.{type T, f}` clause DOES bind concrete
//     importable symbols (`T` a Type, `f` a Function), so we emit one IMPORTS
//     edge per local import clause whose `imported_name` is the clause's first
//     importable unqualified symbol — a real binding the resolver links to the
//     unique project definition. `import gleam/...` (stdlib) resolves to nothing
//     and is skipped. The endpoint label is a Type/Function (not a Module)
//     because the resolver cannot target a Module — the same carve-out documented
//     for Ruby `require` / Dart `import`.
fn extract_gleam(
    language: Language,
    d: &'static crate::registry::LangDef,
    source: &[u8],
    file_path: &str,
) -> greppy_core::Result<ExtractionResult> {
    let queries = d
        .compiled_queries()
        .map_err(|e| greppy_core::Error::Parse(format!("compile {} queries: {e}", d.name)))?;
    // Base pass: definitions (Function / Type), DEFINES, CALLS, and (empty) imports.
    let mut result = crate::spec::spec_extract(language, d.spec, queries, source, file_path)?;

    // CALLS FIX — drop self-recursion self-loops. Self-recursion is recorded as
    // a `self_recursive` property on the function node rather than a
    // `caller → caller` CALLS edge; the generic pass keeps the self-loop. Remove
    // edges whose resolved source and target qnames are identical.
    result
        .edges
        .retain(|e| e.edge_type != "CALLS" || e.source_qualified_name != e.target_qualified_name);

    // IMPORTS — one edge per `import <local module>` clause (see the module
    // doc-comment). The generic import query is empty, so `result` carries no
    // IMPORTS yet; we add them here.
    let tree = crate::parse(language, source)?;
    gleam_imports_pass(source, tree.root_node(), file_path, &mut result);

    Ok(result)
}

/// Emit one `IMPORTS` edge per local (`import <module>` NOT under `gleam/`)
/// import clause. The source is the per-file Module node (`{file}::__file__`);
/// the target resolves by `imported_name` — the clause's first importable
/// unqualified symbol (a `type X` → its `type_identifier`, or a bare value
/// `identifier`) — against the unique project definition. Stdlib imports
/// (`import gleam/int`) name no project symbol and are skipped.
fn gleam_imports_pass(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let file_qname = format!("{file_path}::__file__");
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "import" {
            if let Some(module_node) = node.child_by_field_name("module") {
                let module_path = node_text(source, module_node);
                // Stdlib (`gleam/...`) and any non-local module resolve to no
                // project definition; skip them.
                if !module_path.is_empty() && !gleam_is_stdlib_module(module_path) {
                    if let Some(imported) = gleam_first_importable_name(source, node) {
                        result.edges.push(ExtractedEdge {
                            edge_type: "IMPORTS".into(),
                            source_qualified_name: file_qname.clone(),
                            target_qualified_name: format!("{file_path}::Import::{module_path}"),
                            file_path: file_path.to_string(),
                            line: node.start_position().row as u32 + 1,
                            properties: serde_json::json!({
                                "path": module_path,
                                "imported_name": imported,
                                "original_name": imported,
                                "glob": false,
                            }),
                        });
                    }
                }
            }
            // An `import` clause has no nested `import` — no need to descend.
            continue;
        }
        let mut c = node.walk();
        for child in node.named_children(&mut c) {
            stack.push(child);
        }
    }
}

/// True for a module path that names the Gleam standard library (`gleam/...`)
/// or the bare `gleam` module — none of which is a project definition, so they
/// resolve to nothing and emit no IMPORTS edge.
fn gleam_is_stdlib_module(module_path: &str) -> bool {
    module_path == "gleam" || module_path.starts_with("gleam/")
}

/// The first importable unqualified symbol NAME bound by an `import` clause:
/// the `name:` of the first `unqualified_import` child of the `imports:`
/// `unqualified_imports` group (a `type_identifier` for `type X`, or an
/// `identifier` for a value). Returns `None` for a bare `import mod` with no
/// `{ .. }` group (nothing importable to resolve against).
fn gleam_first_importable_name(source: &[u8], import_node: Node<'_>) -> Option<String> {
    let group = import_node.child_by_field_name("imports")?;
    let mut c = group.walk();
    for child in group.named_children(&mut c) {
        if child.kind() == "unqualified_import" {
            if let Some(name_node) = child.child_by_field_name("name") {
                let name = node_text(source, name_node);
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

fn extract_r(source: &[u8], file_path: &str) -> greppy_core::Result<ExtractionResult> {
    let queries = crate::query::cached_query_set(&Language::R)
        .map_err(|e| greppy_core::Error::Parse(format!("compile r queries: {e}")))?;
    // The shared engine emits the `Function` nodes (`name <- function(...)`,
    // via `DefRule::func("binary_operator")` + `r_def_name`) and the shared
    // IMPORTS pass (`library`/`require`/`requireNamespace`). The R CALLS query
    // captures as `@r_call` (not `@callee`), so `spec_calls` is a no-op — R
    // owns its CALLS pass below.
    let mut result =
        crate::spec::spec_extract(Language::R, &crate::spec::R, queries, source, file_path)?;

    let tree = crate::parse(Language::R, source)?;
    let root = tree.root_node();
    let file_module_qname = format!("{file_path}::__file__");

    // VARIABLE PASS.
    //
    // The pass walks the file-root's direct children and, for each
    // `binary_operator` (R's variable node kind), emits a "Variable" node UNLESS
    // a named child is a `function_definition` (those are the `Function` defs
    // above). The name is the assignment's left-hand side when it is an
    // `identifier` / `string` (a `constant` LHS is also accepted, though R's
    // grammar does not produce it); the `_` placeholder and empty names are
    // dropped. Only top-level assignments are variables — the walk never recurses
    // into function bodies for module-level vars.
    emit_r_variables(source, root, file_path, &mut result);

    // CALLS PASS.
    //
    // Every `call` whose callee is a bare `identifier` that is not a keyword
    // becomes a CALLS edge.
    //
    // SOURCE ENDPOINT — the file node, for EVERY R call. Enclosing-function
    // resolution reads the `name:` field of the enclosing `function_definition`,
    // but in tree-sitter-r that field is the anonymous `function` / `\` keyword
    // token, never the assigned symbol (an R function is named by the *outer*
    // `binary_operator`'s LHS, which the field does not carry). So there is no
    // resolvable enclosing-function name, and every R call — inside a function or
    // at module scope — maps to the file node rather than a Function node,
    // sourced from `{file}::__file__`. Because references then dedup by
    // (caller, callee) and every call in a file shares that one caller, the
    // file's CALLS collapse to one edge per distinct callee — enforced by the
    // per-`(source, callee)` dedup below.
    // `library`/`require`/`requireNamespace` callees are skipped (imports); the
    // name-based resolver drops callees with no unique project definition (R
    // builtins like `list`, `paste`, `sapply`).
    emit_r_calls(source, root, file_path, &file_module_qname, &mut result);

    // USAGE PASS.
    //
    // R has no language-specific reference kind, so a reference is a bare
    // `identifier`. It emits a USAGE unless it sits inside a `call` (R's call and
    // import node kinds are both `{call}`, so a callee or any argument reference
    // is suppressed — already a CALLS or IMPORTS edge), is a definition *name*
    // (the `name:` field of its parent — R assignments use `lhs`/`rhs`, not
    // `name`, so an assignment's LHS IS a usage), or is a generic keyword.
    //
    // SOURCE ENDPOINT — the file node, for every function-body and module-scope
    // reference, exactly like CALLS: an R function can never be named as the
    // enclosing scope (enclosing-function resolution reads the anonymous
    // `function`/`\` keyword token, not the assigned symbol), so every such
    // reference falls back to the file node. References then dedup by
    // (source, ref) — with one shared source per file, all references to a name
    // in a file collapse to a single USAGE edge. Sourcing from
    // `{file}::__file__` produces that.
    //
    // (References sitting in a function's *parameter defaults* would route to the
    // File node instead of the Module node — a second per-file source that keeps
    // a param-default reference distinct from a body/module reference of the same
    // name. The parser cannot name the File node (its qname needs the project
    // prefix, which lives in the indexer), so this fixture references shared
    // constants from function *bodies*, never as parameter defaults — the shape
    // that routes uniformly to the one file node.)
    emit_r_usages(source, root, file_path, &file_module_qname, &mut result);

    Ok(result)
}

/// True if `node`'s named children contain a `function_definition` (i.e. the
/// assignment defines a function — a `Function`, not a `Variable`).
fn r_assignment_defines_function(node: Node<'_>) -> bool {
    let mut c = node.walk();
    let mut found = false;
    for ch in node.named_children(&mut c) {
        if ch.kind() == "function_definition" {
            found = true;
            break;
        }
    }
    found
}

/// Emit a "Variable" node for each top-level `binary_operator` assignment whose
/// right-hand side is NOT a function definition. Only the file root's direct
/// `binary_operator` children are candidates; the name is the LHS when it is an
/// `identifier` or `string`; `_` / empty names are dropped.
fn emit_r_variables(source: &[u8], root: Node<'_>, file_path: &str, result: &mut ExtractionResult) {
    let mut c = root.walk();
    for child in root.named_children(&mut c) {
        if child.kind() != "binary_operator" {
            continue;
        }
        if r_assignment_defines_function(child) {
            continue;
        }
        // LHS: the `lhs` field, falling back to the first named child.
        let Some(lhs) = child
            .child_by_field_name("lhs")
            .or_else(|| child.named_child(0))
        else {
            continue;
        };
        if !matches!(lhs.kind(), "identifier" | "string") {
            continue;
        }
        let vname = node_text(source, lhs);
        if vname.is_empty() || vname == "_" {
            continue;
        }
        result.nodes.push(ExtractedNode {
            label: "Variable".into(),
            name: vname.to_string(),
            qualified_name: format!("{file_path}::Variable::{vname}"),
            file_path: file_path.to_string(),
            start_line: child.start_position().row as u32 + 1,
            end_line: child.end_position().row as u32 + 1,
            properties: serde_json::json!({}),
        });
    }
}

/// Emit `CALLS` edges for every R `call` with a bare-identifier callee. The
/// source is the enclosing function qname or the file Module node;
/// `library`/`require`/`requireNamespace` callees (imports) and generic keywords
/// are skipped.
///
/// The call graph is **deduplicated by (caller, callee)**, so two call sites to
/// the same callee from the same enclosing function collapse to a single edge.
/// Collect per-file, dedup on `(source_qname, callee_text)`, then push.
fn emit_r_calls(
    source: &[u8],
    root: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    let mut collected: Vec<(String, String, u32)> = Vec::new();
    collect_r_calls(source, root, file_module_qname, &mut collected);

    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    for (source_qname, text, line) in collected {
        if !seen.insert((source_qname.clone(), text.clone())) {
            continue;
        }
        result.edges.push(ExtractedEdge {
            edge_type: "CALLS".into(),
            source_qualified_name: source_qname,
            target_qualified_name: format!("{file_path}::Function::{text}"),
            file_path: file_path.to_string(),
            line,
            properties: serde_json::json!({
                "callee_text": text,
                "callee_name": text,
            }),
        });
    }
}

/// Recursively collect `(source_qname, callee_text, line)` for every R `call`
/// with a bare-identifier callee (before dedup). Every call is sourced from
/// `file_module_qname` (see `extract_r`): enclosing-function resolution never
/// recovers an R function name, so all calls fall back to the file node.
fn collect_r_calls(
    source: &[u8],
    node: Node<'_>,
    file_module_qname: &str,
    out: &mut Vec<(String, String, u32)>,
) {
    if node.kind() == "call" {
        if let Some(func) = node.child_by_field_name("function") {
            if func.kind() == "identifier" {
                let text = node_text(source, func);
                if !text.is_empty()
                    && !matches!(text, "library" | "require" | "requireNamespace")
                    && !is_r_usage_keyword(text)
                {
                    out.push((
                        file_module_qname.to_string(),
                        text.to_string(),
                        node.start_position().row as u32 + 1,
                    ));
                }
            }
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        collect_r_calls(source, child, file_module_qname, out);
    }
}

/// Recursively emit `USAGE` edges for R `identifier` references. A reference is
/// suppressed if it is inside a `call` (both the call and import checks use
/// `{call}` for R), is a definition name (the `name:` field of its parent —
/// never set on R assignments), or is a generic keyword. Every surviving
/// reference is sourced from the file node (`file_module_qname`); the store's
/// `UNIQUE(source, target, type)` then collapses repeated references to a name
/// within a file to one edge.
fn emit_r_usages(
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
    file_module_qname: &str,
    result: &mut ExtractionResult,
) {
    if node.kind() == "identifier" && !is_inside_kind(node, &["call"]) && !is_definition_name(node)
    {
        let text = node_text(source, node);
        if !text.is_empty() && !is_r_usage_keyword(text) {
            let source_qname = file_module_qname.to_string();
            result.edges.push(ExtractedEdge {
                edge_type: "USAGE".into(),
                source_qualified_name: source_qname,
                target_qualified_name: format!("{file_path}::__ref__::{text}"),
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
                properties: serde_json::json!({
                    "ref_name": text,
                }),
            });
        }
    }
    let mut c = node.walk();
    for child in node.named_children(&mut c) {
        emit_r_usages(source, child, file_path, file_module_qname, result);
    }
}

/// R keyword / literal filter — R uses the default generic keyword/literal
/// table (no R-specific arm).
fn is_r_usage_keyword(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "null"
            | "nil"
            | "None"
            | "undefined"
            | "void"
            | "if"
            | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "throw"
            | "try"
            | "catch"
            | "finally"
            | "class"
            | "struct"
            | "enum"
            | "interface"
            | "trait"
            | "impl"
            | "import"
            | "export"
            | "package"
            | "module"
            | "use"
            | "require"
            | "include"
            | "new"
            | "delete"
            | "this"
            | "self"
            | "super"
            | "public"
            | "private"
            | "protected"
            | "static"
            | "const"
            | "var"
            | "let"
            | "function"
            | "def"
            | "fn"
            | "func"
            | "fun"
            | "proc"
            | "sub"
            | "method"
            | "async"
            | "await"
            | "yield"
    )
}

pub(crate) fn node_text<'a>(source: &'a [u8], node: Node<'_>) -> &'a str {
    std::str::from_utf8(&source[node.byte_range()]).unwrap_or("<non-utf8>")
}

#[cfg(test)]
mod tests {
    use crate::extract;
    use crate::language::Language;

    const SIMPLE_RS: &str = r#"
        use std::collections::HashMap;

        fn hello() {
            let m: HashMap<String, String> = HashMap::new();
            m.insert("k".to_string(), "v".to_string());
        }

        struct Greeter {
            name: String,
        }

        impl Greeter {
            fn greet(&self) -> String {
                format!("hi {}", self.name)
            }
        }
    "#;

    #[test]
    fn extract_rust_finds_function_struct_impl() {
        let r = extract(Language::Rust, SIMPLE_RS.as_bytes(), "src/lib.rs").unwrap();
        let names: Vec<&str> = r.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(
            names.contains(&"hello"),
            "missing function 'hello': {names:?}"
        );
        assert!(
            names.contains(&"Greeter"),
            "missing struct 'Greeter': {names:?}"
        );
        assert!(
            names.contains(&"greet"),
            "missing method 'greet': {names:?}"
        );
        // Imports are edge-only now (no Import pseudo-node): assert the
        // IMPORTS edge carries the imported path rather than a node name.
        assert!(
            r.edges.iter().any(|e| e.edge_type == "IMPORTS"
                && e.properties.get("path").and_then(|v| v.as_str())
                    == Some("std::collections::HashMap")),
            "missing IMPORTS edge for std::collections::HashMap: {:?}",
            r.edges
        );
    }

    #[test]
    fn extract_unsupported_language_errors_out() {
        let r = extract(Language::Unsupported("go"), b"package main", "main.go");
        match r {
            Err(greppy_core::Error::NotImplemented { feature, .. }) => {
                assert!(feature.contains("language extraction"));
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn extracted_node_has_correct_label_for_function() {
        let r = extract(Language::Rust, SIMPLE_RS.as_bytes(), "src/lib.rs").unwrap();
        let hello = r.nodes.iter().find(|n| n.name == "hello").unwrap();
        assert_eq!(hello.label, "Function");
        assert_eq!(hello.file_path, "src/lib.rs");
        assert!(hello.start_line >= 1);
    }

    #[test]
    fn extracted_node_has_correct_label_for_struct() {
        let r = extract(Language::Rust, SIMPLE_RS.as_bytes(), "src/lib.rs").unwrap();
        let s = r.nodes.iter().find(|n| n.name == "Greeter").unwrap();
        // Rust struct defs are labeled `Class`.
        assert_eq!(s.label, "Class");
    }

    #[test]
    fn extracted_node_has_correct_label_for_impl_method() {
        let r = extract(Language::Rust, SIMPLE_RS.as_bytes(), "src/lib.rs").unwrap();
        let greet = r.nodes.iter().find(|n| n.name == "greet").unwrap();
        assert_eq!(greet.label, "Method");
    }

    #[test]
    fn method_qname_includes_impl_type_to_avoid_collisions() {
        // Two impls with `fn new` must produce two distinct qnames.
        const TWO_NEWS: &str = r#"
            struct Foo;
            struct Bar;
            impl Foo {
                fn new() -> Foo { Foo }
            }
            impl Bar {
                fn new() -> Bar { Bar }
            }
        "#;
        let r = extract(Language::Rust, TWO_NEWS.as_bytes(), "src/lib.rs").unwrap();
        let new_qnames: Vec<&str> = r
            .nodes
            .iter()
            .filter(|n| n.name == "new" && n.label == "Method")
            .map(|n| n.qualified_name.as_str())
            .collect();
        assert_eq!(
            new_qnames.len(),
            2,
            "expected two 'new' methods, got {new_qnames:?}"
        );
        assert!(
            new_qnames.contains(&"src/lib.rs::Foo::new"),
            "missing Foo::new qname; got {new_qnames:?}"
        );
        assert!(
            new_qnames.contains(&"src/lib.rs::Bar::new"),
            "missing Bar::new qname; got {new_qnames:?}"
        );
    }

    #[test]
    fn extract_emits_calls_edges_for_caller_callee_pairs() {
        // A CALLS edge from `hello` (the enclosing
        // function) to the callee text.
        const CALLS_RS: &str = r#"
            fn a() {
                b();
            }
            fn b() {}
        "#;
        let r = extract(Language::Rust, CALLS_RS.as_bytes(), "src/lib.rs").unwrap();
        let calls: Vec<_> = r.edges.iter().filter(|e| e.edge_type == "CALLS").collect();
        assert!(
            !calls.is_empty(),
            "expected at least one CALLS edge, got {calls:?}"
        );
        let src_qnames: std::collections::HashSet<_> = calls
            .iter()
            .map(|e| e.source_qualified_name.clone())
            .collect();
        assert!(
            src_qnames.contains("src/lib.rs::Function::a"),
            "expected caller 'a' in CALLS edges, got {src_qnames:?}"
        );
    }

    #[test]
    fn calls_capture_final_callee_identifier_for_scoped_and_method_calls() {
        // The CALLS pass must capture the FINAL callee identifier, not
        // the first path segment. `helper::do_it()` → `do_it`,
        // `Foo::new()` → `new`, `x.run()` → `run`, `bare()` → `bare`.
        const CALLS_RS: &str = r#"
            fn caller() {
                helper::do_it();
                Foo::new();
                bare();
                let v = Vec::<u8>::new();
            }
        "#;
        let r = extract(Language::Rust, CALLS_RS.as_bytes(), "src/lib.rs").unwrap();
        // Edge targets carry the final callee identifier in
        // `callee_name`.
        let callee_names: std::collections::HashSet<String> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        assert!(
            callee_names.contains("do_it"),
            "scoped call must capture final `do_it`, got {callee_names:?}"
        );
        assert!(
            callee_names.contains("new"),
            "constructor call must capture `new`, got {callee_names:?}"
        );
        assert!(
            callee_names.contains("bare"),
            "bare call must capture `bare`, got {callee_names:?}"
        );
        assert!(
            !callee_names.contains("helper"),
            "must NOT capture first path segment `helper`, got {callee_names:?}"
        );
        assert!(
            !callee_names.contains("Foo"),
            "must NOT capture type path `Foo`, got {callee_names:?}"
        );
    }

    #[test]
    fn rust_calls_preserve_receiver_dispatch_shape() {
        const SOURCE: &str = r#"
            fn caller(value: Buffer) {
                value.method_call();
                free_call(value);
            }
        "#;
        let result = extract(Language::Rust, SOURCE.as_bytes(), "src/lib.rs").unwrap();
        let form_for = |name: &str| {
            result
                .edges
                .iter()
                .find(|edge| {
                    edge.edge_type == "CALLS"
                        && edge.properties.get("callee_name").and_then(|v| v.as_str()) == Some(name)
                })
                .and_then(|edge| edge.properties.get("callee_form"))
                .and_then(|value| value.as_str())
        };

        assert_eq!(form_for("method_call"), Some("receiver"));
        assert_eq!(form_for("free_call"), Some("direct"));
        let receiver_owner = result
            .edges
            .iter()
            .find(|edge| {
                edge.properties.get("callee_name").and_then(|v| v.as_str()) == Some("method_call")
            })
            .and_then(|edge| edge.properties.get("receiver_owner"))
            .and_then(|value| value.as_str());
        assert_eq!(receiver_owner, Some("Buffer"));
    }

    #[test]
    fn rust_receiver_owner_uses_struct_literal_local_without_guessing_calls() {
        const SOURCE: &str = r#"
            fn caller(shadowed: ParameterType) {
                let report = types::IndexReport { files: 0 };
                report.total();
                let unknown = make_report();
                unknown.total();
                let sibling = make_report();
                {
                    let sibling = SiblingType {};
                    sibling.total();
                }
                sibling.total();
                let shadowed = make_report();
                shadowed.total();
            }
        "#;
        let result = extract(Language::Rust, SOURCE.as_bytes(), "src/lib.rs").unwrap();
        let owners: Vec<_> = result
            .edges
            .iter()
            .filter(|edge| {
                edge.edge_type == "CALLS"
                    && edge.properties.get("callee_name").and_then(|v| v.as_str()) == Some("total")
            })
            .map(|edge| {
                edge.properties
                    .get("receiver_owner")
                    .and_then(|value| value.as_str())
            })
            .collect();

        assert_eq!(
            owners,
            vec![Some("IndexReport"), None, Some("SiblingType"), None, None]
        );
    }

    #[test]
    fn extract_emits_imports_edges_for_each_use() {
        const USE_RS: &str = r#"
            use std::collections::HashMap;
            use std::io::Read;
        "#;
        let r = extract(Language::Rust, USE_RS.as_bytes(), "src/lib.rs").unwrap();
        let imp: Vec<_> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .collect();
        assert_eq!(
            imp.len(),
            2,
            "expected one IMPORTS edge per use-statement, got {imp:?}"
        );
    }

    #[test]
    fn extract_handles_pub_visibility_modifier() {
        const PUB_STRUCT: &str = r#"
            pub struct Greeter {
                name: String,
            }
        "#;
        let r = extract(Language::Rust, PUB_STRUCT.as_bytes(), "src/lib.rs").unwrap();
        let g = r.nodes.iter().find(|n| n.name == "Greeter");
        assert!(
            g.is_some(),
            "Greeter struct must be extracted; got: {:?}",
            r.nodes
        );
        assert_eq!(g.unwrap().label, "Class");
    }

    // ---- USAGE pass ----
    //
    // A SINGLE usage pass emits one `USAGE` edge per identifier reference (type
    // refs, value refs, field refs). There is no separate TYPE_REF/USES pass — a
    // type reference is just another reference node captured by the unified usage
    // walk. These tests assert type-position references still appear as USAGE
    // edges.

    /// Collect `(source_qname, ref_name)` for every USAGE edge.
    fn usages(src: &str) -> Vec<(String, String)> {
        let r = extract(Language::Rust, src.as_bytes(), "src/lib.rs").unwrap();
        r.edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.properties
                        .get("ref_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn usages_capture_param_return_and_field_type_references() {
        const SRC: &str = r#"
            struct Config { handler: Handler, count: u32 }
            fn build(input: Request) -> Response {
                let cfg: Config = make();
                cfg
            }
        "#;
        let us = usages(SRC);
        // Parameter type `Request` on `build` — the enclosing function is the
        // source.
        assert!(
            us.contains(&("src/lib.rs::Function::build".into(), "Request".into())),
            "param type Request missing: {us:?}"
        );
        // Return type `Response` on `build`.
        assert!(
            us.contains(&("src/lib.rs::Function::build".into(), "Response".into())),
            "return type Response missing: {us:?}"
        );
        // Field type `Handler` — inside the struct body, not a function, so it
        // is sourced from the file node (`__file__`).
        assert!(
            us.iter().any(|(_, n)| n == "Handler"),
            "field type Handler missing: {us:?}"
        );
        // `let cfg: Config` binding type reference.
        assert!(
            us.contains(&("src/lib.rs::Function::build".into(), "Config".into())),
            "let-binding type Config missing: {us:?}"
        );
    }

    #[test]
    fn usages_capture_value_references_with_enclosing_func() {
        const SRC: &str = r#"
            fn run() {
                let total = base;
                let other = total;
            }
        "#;
        let us = usages(SRC);
        // `base` is a bare value reference inside `run`.
        assert!(
            us.contains(&("src/lib.rs::Function::run".into(), "base".into())),
            "expected USAGE base from run: {us:?}"
        );
        // `total` is read on the RHS of `other`.
        assert!(
            us.iter()
                .any(|(s, n)| s == "src/lib.rs::Function::run" && n == "total"),
            "expected USAGE total from run: {us:?}"
        );
    }

    #[test]
    fn usages_exclude_definition_names_calls_and_keywords() {
        const SRC: &str = r#"
            fn run() {
                helper();
                let x = value;
            }
        "#;
        let names: Vec<String> = usages(SRC).into_iter().map(|(_, n)| n).collect();
        // The callee `helper` is inside a call_expression → suppressed (it is
        // a CALLS edge, not a USAGE).
        assert!(
            !names.contains(&"helper".to_string()),
            "callee `helper` must not be a USAGE ref: {names:?}"
        );
        // The defined name `run` must not be a usage of itself.
        assert!(
            !names.contains(&"run".to_string()),
            "definition name `run` must not be a USAGE ref: {names:?}"
        );
        // A real RHS reference is captured.
        assert!(
            names.contains(&"value".to_string()),
            "expected `value` usage: {names:?}"
        );
    }

    #[test]
    fn usages_capture_scoped_leaf_names_and_call_arguments() {
        const SRC: &str = r#"
            fn render(w: types::Widget) {
                make(types::Marker);
            }
        "#;
        let names: Vec<String> = usages(SRC).into_iter().map(|(_, n)| n).collect();
        assert!(
            names.contains(&"Widget".to_string()),
            "scoped type should emit leaf ref Widget: {names:?}"
        );
        assert!(
            names.contains(&"Marker".to_string()),
            "call argument should emit leaf ref Marker: {names:?}"
        );
        assert!(
            !names.contains(&"types::Widget".to_string()) && !names.contains(&"types".to_string()),
            "scoped ref must not emit module prefix/full path noise: {names:?}"
        );
        assert!(
            !names.contains(&"make".to_string()),
            "call target make must remain a CALLS edge, not USAGE: {names:?}"
        );
    }

    #[test]
    fn usages_exclude_identifiers_inside_use_declarations() {
        const SRC: &str = r#"
            use std::collections::HashMap;
            fn f() { let _x = HashMap::new(); }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        // No USAGE edge should be sourced for the `std`/`collections` path
        // segments — those live inside a `use_declaration` and are suppressed.
        // `HashMap` is inside a `call_expression` (`HashMap::new()`) so it is
        // suppressed too.
        let import_path_refs: Vec<_> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .filter(|e| {
                let n = e
                    .properties
                    .get("ref_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                n == "std" || n == "collections"
            })
            .collect();
        assert!(
            import_path_refs.is_empty(),
            "use-declaration path segments must not be USAGE: {import_path_refs:?}"
        );
    }

    #[test]
    fn usages_skip_builtin_and_call_interior_but_keep_type_refs() {
        // Builtin primitives (`u32`, `bool`) are not registered symbols, so
        // even though they are emitted as usages they never resolve — but the
        // parser still emits a raw USAGE for a non-builtin type reference.
        const SRC: &str = r#"
            fn f(a: Foo, n: u32) -> Bar { g() }
        "#;
        let names: std::collections::HashSet<String> =
            usages(SRC).into_iter().map(|(_, t)| t).collect();
        assert!(names.contains("Foo"), "Foo type ref kept: {names:?}");
        assert!(names.contains("Bar"), "Bar return type kept: {names:?}");
        // `g` is the callee of a call → suppressed.
        assert!(
            !names.contains("g"),
            "call callee `g` must be suppressed: {names:?}"
        );
    }

    // ---- richer IMPORTS (imported_name property) ----

    #[test]
    fn imports_carry_final_imported_name_property() {
        const SRC: &str = r#"
            use crate::foo::Bar;
            use std::io::Read;
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        let by_name = |want: &str| {
            r.edges
                .iter()
                .filter(|e| e.edge_type == "IMPORTS")
                .any(|e| e.properties.get("imported_name").and_then(|v| v.as_str()) == Some(want))
        };
        assert!(
            by_name("Bar"),
            "use crate::foo::Bar must carry imported_name=Bar"
        );
        assert!(
            by_name("Read"),
            "use std::io::Read must carry imported_name=Read"
        );
    }

    // ---- brace-group / rename import expansion ----

    /// Collect `(path, imported_name, original_name, glob)` for IMPORTS edges.
    fn imports(src: &str) -> Vec<(String, String, String, bool)> {
        let r = extract(Language::Rust, src.as_bytes(), "src/lib.rs").unwrap();
        r.edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .map(|e| {
                let p = &e.properties;
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                (
                    s("path"),
                    s("imported_name"),
                    s("original_name"),
                    p.get("glob").and_then(|v| v.as_bool()).unwrap_or(false),
                )
            })
            .collect()
    }

    #[test]
    fn brace_group_import_stays_one_edge_with_first_member_name() {
        // A `use` with a brace group is ONE import, NOT one per name. The whole
        // `a::{B, C}` text is the module path and the representative symbol is the
        // FIRST group member (`B`).
        const SRC: &str = r#"
            use std::collections::{HashMap, HashSet as Set};
        "#;
        let imp = imports(SRC);
        assert_eq!(imp.len(), 1, "brace group must stay ONE edge: {imp:?}");
        let (path, name, ..) = &imp[0];
        assert!(
            path.contains("{HashMap, HashSet as Set}"),
            "the module path is the whole use-tree text: {imp:?}"
        );
        assert_eq!(
            name, "HashMap",
            "representative symbol is the first brace-group member: {imp:?}"
        );
    }

    #[test]
    fn nested_brace_group_import_stays_one_edge() {
        // Still one import per `use` statement, no expansion.
        const SRC: &str = r#"
            use a::b::{C, d::{E, F}};
        "#;
        let imp = imports(SRC);
        assert_eq!(
            imp.len(),
            1,
            "nested brace group must stay ONE edge: {imp:?}"
        );
        let (_, name, ..) = &imp[0];
        assert_eq!(
            name, "C",
            "representative symbol is the first member `C`: {imp:?}"
        );
    }

    #[test]
    fn glob_import_stays_single_edge() {
        const SRC: &str = r#"
            use std::io::prelude::*;
        "#;
        let imp = imports(SRC);
        assert_eq!(imp.len(), 1, "glob must stay a single edge: {imp:?}");
        let (_path, name, _orig, glob) = &imp[0];
        assert!(glob, "glob edge must set glob=true: {imp:?}");
        // The trailing `*`/`::` is stripped and the last path segment
        // (`prelude`) is taken as the representative symbol.
        assert_eq!(
            name, "prelude",
            "glob representative is the last path segment: {imp:?}"
        );
    }

    #[test]
    fn top_level_rename_import_binds_original_symbol() {
        // A rename resolves to the ORIGINAL symbol (`Read`), not the local alias
        // (`R`) — the ` as <alias>` suffix is stripped.
        const SRC: &str = r#"
            use std::io::Read as R;
        "#;
        let imp = imports(SRC);
        assert_eq!(imp.len(), 1, "single rename → single edge: {imp:?}");
        let (path, name, orig, _) = &imp[0];
        assert_eq!(path, "std::io::Read as R");
        assert_eq!(name, "Read", "representative is the original symbol");
        assert_eq!(orig, "Read", "original symbol recorded in original_name");
    }

    // ---- TYPE_ASSIGN pass ----

    /// Collect `(source_qname, var_name, type_name)` for TYPE_ASSIGN edges.
    fn type_assigns(src: &str) -> Vec<(String, String, String)> {
        let r = extract(Language::Rust, src.as_bytes(), "src/lib.rs").unwrap();
        r.edges
            .iter()
            .filter(|e| e.edge_type == "TYPE_ASSIGN")
            .map(|e| {
                let p = &e.properties;
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                (
                    e.source_qualified_name.clone(),
                    s("var_name"),
                    s("type_name"),
                )
            })
            .collect()
    }

    #[test]
    fn type_assign_captures_let_binding_declared_type() {
        const SRC: &str = r#"
            fn build() {
                let cfg: Config = make();
                let n: u32 = 0;
            }
        "#;
        let tas = type_assigns(SRC);
        assert!(
            tas.contains(&(
                "src/lib.rs::Function::build".into(),
                "cfg".into(),
                "Config".into()
            )),
            "let cfg: Config must emit TYPE_ASSIGN(build, cfg, Config): {tas:?}"
        );
        // Builtin primitive types are skipped (matches TYPE_REF behaviour).
        assert!(
            !tas.iter().any(|(_, _, t)| t == "u32"),
            "primitive `u32` declared type must be skipped: {tas:?}"
        );
    }

    #[test]
    fn type_assign_captures_field_and_const_types() {
        const SRC: &str = r#"
            struct S { handler: Handler }
            const DEFAULT: Config = Config::new();
            fn f() {}
        "#;
        let tas = type_assigns(SRC);
        // Field type: source is the enclosing struct.
        assert!(
            tas.contains(&(
                "src/lib.rs::Class::S".into(),
                "handler".into(),
                "Handler".into()
            )),
            "struct field must emit TYPE_ASSIGN(S, handler, Handler): {tas:?}"
        );
        // A top-level const with no enclosing def is skipped (type-assigns
        // attach to an enclosing function only).
        assert!(
            !tas.iter().any(|(_, v, _)| v == "DEFAULT"),
            "top-level const with no enclosing def must be skipped: {tas:?}"
        );
    }

    #[test]
    fn type_assign_distinct_from_usage_but_both_present() {
        // A `let x: T` produces BOTH a USAGE (the `T` type reference, via the
        // unified usage walk) and a TYPE_ASSIGN (a binding's declared type) —
        // they are distinct edges with different properties.
        const SRC: &str = r#"
            fn f() { let x: Widget = build(); }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        let has_usage = r.edges.iter().any(|e| {
            e.edge_type == "USAGE"
                && e.properties.get("ref_name").and_then(|v| v.as_str()) == Some("Widget")
        });
        let has_type_assign = r.edges.iter().any(|e| {
            e.edge_type == "TYPE_ASSIGN"
                && e.properties.get("var_name").and_then(|v| v.as_str()) == Some("x")
                && e.properties.get("type_name").and_then(|v| v.as_str()) == Some("Widget")
        });
        assert!(
            has_usage,
            "expected a USAGE for the Widget type reference: {:?}",
            r.edges
        );
        assert!(
            has_type_assign,
            "expected a distinct TYPE_ASSIGN(x, Widget): {:?}",
            r.edges
        );
    }

    // ---- docstrings ----

    /// Look up the `doc` / `doc_full` property of a named definition node.
    fn node_doc(src: &str, name: &str) -> Option<(String, String)> {
        let r = extract(Language::Rust, src.as_bytes(), "src/lib.rs").unwrap();
        r.nodes.iter().find(|n| n.name == name).map(|n| {
            let p = &n.properties;
            let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
            (s("doc"), s("doc_full"))
        })
    }

    #[test]
    fn line_doc_comment_attaches_summary_and_full_text() {
        const SRC: &str = r#"
/// Builds the greeter.
/// Second line of detail.
fn build() {}
"#;
        let (doc, full) = node_doc(SRC, "build").expect("build node must exist");
        assert_eq!(doc, "Builds the greeter.", "doc summary = first line");
        assert_eq!(
            full, "Builds the greeter.\nSecond line of detail.",
            "doc_full keeps all lines, markers stripped"
        );
    }

    #[test]
    fn block_doc_comment_attaches_to_struct() {
        const SRC: &str = r#"
/** A configuration struct. */
struct Config { x: u32 }
"#;
        let (doc, _full) = node_doc(SRC, "Config").expect("Config node must exist");
        assert_eq!(doc, "A configuration struct.");
    }

    #[test]
    fn ordinary_comment_is_not_a_docstring() {
        // A plain `//` comment is NOT a doc comment and must not attach.
        const SRC: &str = r#"
// just a regular note
fn plain() {}
"#;
        let (doc, full) = node_doc(SRC, "plain").expect("plain node must exist");
        assert_eq!(doc, "", "regular `//` comment must not become a docstring");
        assert_eq!(full, "");
    }

    #[test]
    fn definition_without_doc_has_no_doc_property() {
        const SRC: &str = r#"
            fn bare() {}
        "#;
        let (doc, full) = node_doc(SRC, "bare").expect("bare node must exist");
        assert_eq!(doc, "");
        assert_eq!(full, "");
    }

    // ---- inheritance: IMPLEMENTS edges ----

    /// Collect `(source_qname, target_qname, trait_name, type_name)` for every
    /// IMPLEMENTS edge.
    fn implements(src: &str) -> Vec<(String, String, String, String)> {
        let r = extract(Language::Rust, src.as_bytes(), "src/lib.rs").unwrap();
        r.edges
            .iter()
            .filter(|e| e.edge_type == "IMPLEMENTS")
            .map(|e| {
                let p = &e.properties;
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                    s("trait_name"),
                    s("type_name"),
                )
            })
            .collect()
    }

    #[test]
    fn trait_impl_emits_implements_edge_type_to_trait() {
        const SRC: &str = r#"
            trait Animal { fn speak(&self) -> String; }
            struct Dog;
            impl Animal for Dog {
                fn speak(&self) -> String { String::new() }
            }
        "#;
        let imps = implements(SRC);
        assert_eq!(
            imps.len(),
            1,
            "expected exactly one IMPLEMENTS edge: {imps:?}"
        );
        let (src_q, tgt_q, trait_name, type_name) = &imps[0];
        // Edge goes FROM the implementing type TO the trait.
        assert_eq!(
            src_q, "src/lib.rs::Class::Dog",
            "source must be the type: {imps:?}"
        );
        assert_eq!(
            tgt_q, "src/lib.rs::Interface::Animal",
            "target must be the trait: {imps:?}"
        );
        // Name properties for the resolver.
        assert_eq!(trait_name, "Animal", "trait_name property: {imps:?}");
        assert_eq!(type_name, "Dog", "type_name property: {imps:?}");
    }

    #[test]
    fn inherent_impl_emits_no_implements_edge() {
        // `impl Type { ... }` (no trait) must NOT produce an IMPLEMENTS edge.
        const SRC: &str = r#"
            struct Widget;
            impl Widget {
                fn build(&self) {}
            }
        "#;
        let imps = implements(SRC);
        assert!(
            imps.is_empty(),
            "inherent impl must not emit IMPLEMENTS: {imps:?}"
        );
    }

    #[test]
    fn generic_trait_impl_implements_edge_uses_base_type_names() {
        // `impl Display for Vec<Foo>` — the implementing type's base name is
        // `Vec`; the trait is `Display`.
        const SRC: &str = r#"
            impl Display for Wrapper<Foo> {
                fn fmt(&self) {}
            }
        "#;
        let imps = implements(SRC);
        assert_eq!(imps.len(), 1, "expected one IMPLEMENTS edge: {imps:?}");
        let (_src_q, _tgt_q, trait_name, type_name) = &imps[0];
        assert_eq!(trait_name, "Display", "trait must be Display: {imps:?}");
        assert_eq!(
            type_name, "Wrapper",
            "implementing type must be the base `Wrapper`, not the generic arg: {imps:?}"
        );
    }

    #[test]
    fn trait_impl_method_qname_owned_by_implementing_type_not_trait() {
        // Regression: in `impl Trait for Type`, a method's qname owner must be
        // the implementing *Type*, not the *Trait*.
        const SRC: &str = r#"
            trait Animal { fn speak(&self); }
            struct Dog;
            impl Animal for Dog {
                fn speak(&self) {}
            }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        let speak = r
            .nodes
            .iter()
            .find(|n| n.name == "speak" && n.label == "Method")
            .expect("speak method node must exist");
        assert_eq!(
            speak.qualified_name, "src/lib.rs::Dog::speak",
            "trait-impl method must be owned by the implementing type Dog, got {}",
            speak.qualified_name
        );
    }

    // ---- inheritance: enum variants ----

    #[test]
    fn enum_variants_emit_nodes_and_defines_edges() {
        const SRC: &str = r#"
            enum Color {
                Red,
                Green,
                Rgb(u8, u8, u8),
                Named { name: String },
            }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();

        // One EnumVariant node per variant, qname qualified by the enum.
        let variants: std::collections::HashSet<(String, String)> = r
            .nodes
            .iter()
            .filter(|n| n.label == "EnumVariant")
            .map(|n| (n.name.clone(), n.qualified_name.clone()))
            .collect();
        for (name, qname) in [
            ("Red", "src/lib.rs::Color::Red"),
            ("Green", "src/lib.rs::Color::Green"),
            ("Rgb", "src/lib.rs::Color::Rgb"),
            ("Named", "src/lib.rs::Color::Named"),
        ] {
            assert!(
                variants.contains(&(name.to_string(), qname.to_string())),
                "missing EnumVariant node {name} ({qname}): {variants:?}"
            );
        }
        assert_eq!(
            variants.len(),
            4,
            "expected exactly 4 variants: {variants:?}"
        );

        // A DEFINES edge from the enum to each variant.
        let defines: std::collections::HashSet<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "DEFINES")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                )
            })
            .collect();
        for qname in [
            "src/lib.rs::Color::Red",
            "src/lib.rs::Color::Green",
            "src/lib.rs::Color::Rgb",
            "src/lib.rs::Color::Named",
        ] {
            assert!(
                defines.contains(&("src/lib.rs::Enum::Color".to_string(), qname.to_string())),
                "missing DEFINES edge Enum::Color -> {qname}: {defines:?}"
            );
        }
    }

    #[test]
    fn enum_variant_carries_owning_enum_property() {
        const SRC: &str = r#"
            enum Status { Open, Closed }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        let open = r
            .nodes
            .iter()
            .find(|n| n.label == "EnumVariant" && n.name == "Open")
            .expect("Open variant must exist");
        assert_eq!(
            open.properties.get("enum").and_then(|v| v.as_str()),
            Some("Status"),
            "variant must record its owning enum: {:?}",
            open.properties
        );
    }

    // ---- inheritance: associated consts / types ----

    /// Find an extracted node by exact (label, name).
    fn find_node<'a>(
        nodes: &'a [crate::ExtractedNode],
        label: &str,
        name: &str,
    ) -> Option<&'a crate::ExtractedNode> {
        nodes.iter().find(|n| n.label == label && n.name == name)
    }

    #[test]
    fn associated_const_and_type_in_impl_owned_by_type() {
        const SRC: &str = r#"
            struct Dog;
            impl Dog {
                const LEGS: u32 = 4;
                type Output = Bark;
                fn bark(&self) {}
            }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();

        let legs = find_node(&r.nodes, "AssocConst", "LEGS")
            .expect("associated const LEGS must be a node");
        assert_eq!(
            legs.qualified_name, "src/lib.rs::Dog::LEGS",
            "assoc const qname must be owned by Dog: {}",
            legs.qualified_name
        );
        assert_eq!(
            legs.properties.get("owner").and_then(|v| v.as_str()),
            Some("Dog")
        );

        let output = find_node(&r.nodes, "AssocType", "Output")
            .expect("associated type Output must be a node");
        assert_eq!(
            output.qualified_name, "src/lib.rs::Dog::Output",
            "assoc type qname must be owned by Dog: {}",
            output.qualified_name
        );
        assert_eq!(
            output.properties.get("owner").and_then(|v| v.as_str()),
            Some("Dog")
        );
    }

    #[test]
    fn associated_const_and_type_in_trait_owned_by_trait() {
        const SRC: &str = r#"
            trait Animal {
                const LEGS: u32 = 4;
                type Output;
                fn speak(&self) -> String;
            }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();

        let legs = find_node(&r.nodes, "AssocConst", "LEGS")
            .expect("trait associated const LEGS must be a node");
        assert_eq!(
            legs.qualified_name, "src/lib.rs::Animal::LEGS",
            "trait assoc const qname must be owned by Animal: {}",
            legs.qualified_name
        );

        // `type Output;` in a trait is an `associated_type` node (no `= ...`).
        let output = find_node(&r.nodes, "AssocType", "Output")
            .expect("trait associated type Output must be a node");
        assert_eq!(
            output.qualified_name, "src/lib.rs::Animal::Output",
            "trait assoc type qname must be owned by Animal: {}",
            output.qualified_name
        );
    }

    #[test]
    fn assoc_const_in_trait_impl_owned_by_implementing_type() {
        // In `impl Trait for Type`, an associated const's owner must be the
        // implementing Type, not the Trait.
        const SRC: &str = r#"
            trait HasLegs { const LEGS: u32; }
            struct Spider;
            impl HasLegs for Spider {
                const LEGS: u32 = 8;
            }
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        // Two LEGS assoc consts: one in the trait (owner HasLegs), one in the
        // impl (owner Spider).
        let owners: std::collections::HashSet<String> = r
            .nodes
            .iter()
            .filter(|n| n.label == "AssocConst" && n.name == "LEGS")
            .map(|n| n.qualified_name.clone())
            .collect();
        assert!(
            owners.contains("src/lib.rs::HasLegs::LEGS"),
            "trait decl LEGS must be owned by HasLegs: {owners:?}"
        );
        assert!(
            owners.contains("src/lib.rs::Spider::LEGS"),
            "impl LEGS must be owned by implementing type Spider, not HasLegs: {owners:?}"
        );
    }

    #[test]
    fn top_level_const_is_not_an_assoc_const_node() {
        // A top-level `const` (no impl/trait owner) must NOT become an
        // AssocConst node — it has no owner to qualify under.
        const SRC: &str = r#"
            const MAX: u32 = 10;
            fn f() {}
        "#;
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        assert!(
            find_node(&r.nodes, "AssocConst", "MAX").is_none(),
            "top-level const must not be an AssocConst node: {:?}",
            r.nodes
        );
    }

    // ---- signature + params + return type ----

    /// Look up a named node and return its `properties` object.
    fn node_props(src: &str, name: &str) -> serde_json::Value {
        let r = extract(Language::Rust, src.as_bytes(), "src/lib.rs").unwrap();
        r.nodes
            .iter()
            .find(|n| n.name == name)
            .map(|n| n.properties.clone())
            .unwrap_or(serde_json::Value::Null)
    }

    #[test]
    fn function_node_carries_signature_params_and_return_type() {
        const SRC: &str = r#"
            fn build(input: Request, count: u32) -> Response {
                todo!()
            }
        "#;
        let p = node_props(SRC, "build");
        assert_eq!(
            p.get("signature").and_then(|v| v.as_str()),
            Some("(input: Request, count: u32) -> Response"),
            "signature must include params + return type: {p}"
        );
        assert_eq!(
            p.get("return_type").and_then(|v| v.as_str()),
            Some("Response"),
            "return_type property: {p}"
        );
        let params = p
            .get("params")
            .and_then(|v| v.as_array())
            .expect("params array");
        assert_eq!(params.len(), 2, "two params: {p}");
        assert_eq!(
            params[0].get("name").and_then(|v| v.as_str()),
            Some("input")
        );
        assert_eq!(
            params[0].get("type").and_then(|v| v.as_str()),
            Some("Request")
        );
        assert_eq!(
            params[1].get("name").and_then(|v| v.as_str()),
            Some("count")
        );
        assert_eq!(params[1].get("type").and_then(|v| v.as_str()), Some("u32"));
    }

    #[test]
    fn rust_function_node_uses_full_item_span_and_source_signature() {
        const SRC: &str = "impl Rules {\n    pub fn rename_by_rules(&mut self, rules: RenameAllRules) {\n        self.serialize = rules.serialize;\n        self.deserialize = rules.deserialize;\n    }\n}\n";
        let r = extract(Language::Rust, SRC.as_bytes(), "src/lib.rs").unwrap();
        let node = r
            .nodes
            .iter()
            .find(|node| node.name == "rename_by_rules")
            .expect("rename_by_rules node");

        assert_eq!((node.start_line, node.end_line), (2, 5));
        assert_eq!(
            node.properties
                .get("source_signature")
                .and_then(|value| value.as_str()),
            Some("pub fn rename_by_rules(&mut self, rules: RenameAllRules)")
        );
        assert!(!node
            .properties
            .get("source_signature")
            .and_then(|value| value.as_str())
            .unwrap()
            .contains("-> ()"));
    }

    #[test]
    fn rust_source_signature_preserves_modifiers_generics_and_where_clause() {
        const SRC: &str = r#"
            pub unsafe extern "C" fn transform<'a, T: Clone>(
                value: &'a T,
            ) -> Option<&'a T>
            where
                T: Send,
            {
                Some(value)
            }
        "#;
        let p = node_props(SRC, "transform");

        assert_eq!(
            p.get("source_signature").and_then(|value| value.as_str()),
            Some(
                "pub unsafe extern \"C\" fn transform<'a, T: Clone>( value: &'a T, ) -> Option<&'a T> where T: Send,"
            )
        );
    }

    #[test]
    fn method_signature_captures_self_receiver_and_no_return() {
        const SRC: &str = r#"
            struct Greeter { name: String }
            impl Greeter {
                fn touch(&mut self, n: u32) {}
            }
        "#;
        let p = node_props(SRC, "touch");
        // Parameterless-return method: signature is just the param list.
        assert_eq!(
            p.get("signature").and_then(|v| v.as_str()),
            Some("(&mut self, n: u32)"),
            "signature for method without return type: {p}"
        );
        assert!(
            p.get("return_type").is_none(),
            "no return_type property when method returns unit: {p}"
        );
        let params = p.get("params").and_then(|v| v.as_array()).expect("params");
        assert_eq!(params[0].get("name").and_then(|v| v.as_str()), Some("self"));
        assert_eq!(
            params[0].get("type").and_then(|v| v.as_str()),
            Some("&mut self")
        );
        assert_eq!(params[1].get("name").and_then(|v| v.as_str()), Some("n"));
    }

    #[test]
    fn struct_node_has_no_signature_property() {
        const SRC: &str = r#"
            struct Config { x: u32 }
        "#;
        let p = node_props(SRC, "Config");
        assert!(
            p.get("signature").is_none(),
            "non-function defs must not carry a signature: {p}"
        );
    }

    // ---- modifiers ----

    #[test]
    fn function_modifiers_visibility_async_unsafe_const() {
        const SRC: &str = r#"
            pub async unsafe fn a() {}
            pub(crate) const fn b() {}
            fn c() {}
        "#;
        let a = node_props(SRC, "a");
        assert_eq!(a.get("visibility").and_then(|v| v.as_str()), Some("pub"));
        assert_eq!(a.get("is_async").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(a.get("is_unsafe").and_then(|v| v.as_bool()), Some(true));
        assert!(a.get("is_const").is_none(), "a is not const: {a}");

        let b = node_props(SRC, "b");
        assert_eq!(
            b.get("visibility").and_then(|v| v.as_str()),
            Some("pub(crate)")
        );
        assert_eq!(b.get("is_const").and_then(|v| v.as_bool()), Some(true));
        assert!(b.get("is_async").is_none(), "b is not async: {b}");

        let c = node_props(SRC, "c");
        assert!(
            c.get("visibility").is_none(),
            "private fn has no visibility property: {c}"
        );
        assert!(c.get("is_async").is_none());
        assert!(c.get("is_unsafe").is_none());
        assert!(c.get("is_const").is_none());
    }

    #[test]
    fn struct_visibility_modifier_captured() {
        const SRC: &str = r#"
            pub struct Public { x: u32 }
            struct Private { y: u32 }
        "#;
        assert_eq!(
            node_props(SRC, "Public")
                .get("visibility")
                .and_then(|v| v.as_str()),
            Some("pub"),
            "pub struct must carry visibility"
        );
        assert!(
            node_props(SRC, "Private").get("visibility").is_none(),
            "private struct has no visibility property"
        );
    }

    // ---- generic bounds: BOUND edges ----

    /// Collect `(source_qname, target_qname, type_param, bound_name)` for each
    /// BOUND edge.
    fn bounds(src: &str) -> Vec<(String, String, String, String)> {
        let r = extract(Language::Rust, src.as_bytes(), "src/lib.rs").unwrap();
        r.edges
            .iter()
            .filter(|e| e.edge_type == "BOUND")
            .map(|e| {
                let p = &e.properties;
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                    s("type_param"),
                    s("name"),
                )
            })
            .collect()
    }

    #[test]
    fn angle_bracket_generic_bounds_emit_bound_edges() {
        const SRC: &str = r#"
            fn f<T: Clone + Send, U>(a: T, b: U) {}
        "#;
        let bs = bounds(SRC);
        // `T: Clone + Send` → two BOUND edges; `U` (unconstrained) → none.
        assert!(
            bs.contains(&(
                "src/lib.rs::Function::f".into(),
                "src/lib.rs::Interface::Clone".into(),
                "T".into(),
                "Clone".into()
            )),
            "expected BOUND f -> Clone for T: {bs:?}"
        );
        assert!(
            bs.iter()
                .any(|(_, _, tp, name)| tp == "T" && name == "Send"),
            "expected BOUND for T: Send: {bs:?}"
        );
        assert!(
            !bs.iter().any(|(_, _, tp, _)| tp == "U"),
            "unconstrained U must not emit a BOUND edge: {bs:?}"
        );
    }

    #[test]
    fn where_clause_bounds_emit_bound_edges() {
        const SRC: &str = r#"
            fn g<T, U>(a: T, b: U) where T: Default, U: Iterator {}
        "#;
        let bs = bounds(SRC);
        assert!(
            bs.iter()
                .any(|(_, _, tp, name)| tp == "T" && name == "Default"),
            "expected where-clause BOUND T: Default: {bs:?}"
        );
        assert!(
            bs.iter()
                .any(|(_, _, tp, name)| tp == "U" && name == "Iterator"),
            "expected where-clause BOUND U: Iterator: {bs:?}"
        );
    }

    #[test]
    fn generic_bound_target_resolves_to_local_trait() {
        // When the bound trait is defined in the same file, the BOUND edge's
        // target qname must point at that trait node's qname.
        const SRC: &str = r#"
            trait Speak {}
            fn announce<T: Speak>(t: T) {}
        "#;
        let bs = bounds(SRC);
        assert!(
            bs.contains(&(
                "src/lib.rs::Function::announce".into(),
                "src/lib.rs::Interface::Speak".into(),
                "T".into(),
                "Speak".into()
            )),
            "BOUND target must be the local trait qname: {bs:?}"
        );
    }

    #[test]
    fn method_generic_bound_owned_by_impl_type() {
        // A generic-bounded method's BOUND source must be the method's
        // impl-qualified qname (collision-free), not a bare Function qname.
        const SRC: &str = r#"
            trait Render {}
            struct View;
            impl View {
                fn draw<T: Render>(&self, t: T) {}
            }
        "#;
        let bs = bounds(SRC);
        assert!(
            bs.iter()
                .any(|(src, _, tp, name)| src == "src/lib.rs::View::draw"
                    && tp == "T"
                    && name == "Render"),
            "method BOUND source must be View::draw: {bs:?}"
        );
    }

    // =======================================================================
    // Python extraction (Track A — second language)
    // =======================================================================

    fn py(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Python, src.as_bytes(), file).unwrap()
    }

    #[test]
    fn python_extracts_functions_and_classes() {
        const SRC: &str = r#"
def top_level():
    pass

class Greeter:
    def greet(self):
        return "hi"
"#;
        let r = py(SRC, "app/a.py");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let f = by("Function", "top_level").expect("free function node");
        assert_eq!(f.qualified_name, "app/a.py::Function::top_level");
        let c = by("Class", "Greeter").expect("class node");
        assert_eq!(c.qualified_name, "app/a.py::Class::Greeter");
        // A method nested in a class is owned by that class.
        let m = by("Method", "greet").expect("method node");
        assert_eq!(
            m.qualified_name, "app/a.py::Greeter::greet",
            "method qname must be owned by its class"
        );
    }

    #[test]
    fn python_module_level_variables_become_variable_nodes() {
        // Only *module-level* `assignment` / `augmented_assignment` whose `left`
        // is a plain identifier become `Variable` nodes. Assignments inside a
        // function/class body, tuple targets, attribute/subscript targets, and
        // the `_` placeholder are all skipped.
        const SRC: &str = r#"
CONST = 1
name: str = "probe"
COUNTER = 0
COUNTER += 1
a, b = 1, 2
obj.attr = 3
d[k] = 4
_ = 5

def f():
    local_var = 6

class C:
    field = 7
"#;
        let r = py(SRC, "m.py");
        let vars: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.label == "Variable")
            .map(|n| n.name.as_str())
            .collect();
        // Module-level plain-identifier assignments (incl. annotated + augmented).
        assert!(vars.contains("CONST"), "CONST missing: {vars:?}");
        assert!(vars.contains("name"), "annotated name missing: {vars:?}");
        assert!(vars.contains("COUNTER"), "COUNTER missing: {vars:?}");
        // Not variables: tuple / attribute / subscript targets, `_`, and
        // assignments nested inside a function or class body.
        assert!(
            !vars.contains("a"),
            "tuple target must be skipped: {vars:?}"
        );
        assert!(
            !vars.contains("b"),
            "tuple target must be skipped: {vars:?}"
        );
        assert!(
            !vars.contains("_"),
            "`_` placeholder must be skipped: {vars:?}"
        );
        assert!(
            !vars.contains("local_var"),
            "function-body assignment is not a module Variable: {vars:?}"
        );
        assert!(
            !vars.contains("field"),
            "class-body assignment is not a module Variable: {vars:?}"
        );
        // `COUNTER` appears twice in source (=, +=); one Variable def is pushed
        // per qualifying assignment node — assert the qname is the module-scoped
        // one and that the count matches the per-assignment push.
        let counter_defs = r
            .nodes
            .iter()
            .filter(|n| n.label == "Variable" && n.name == "COUNTER")
            .count();
        assert_eq!(
            counter_defs, 2,
            "one Variable per module-level assignment node (= and +=)"
        );
        let const_var = r
            .nodes
            .iter()
            .find(|n| n.label == "Variable" && n.name == "CONST")
            .unwrap();
        assert_eq!(const_var.qualified_name, "m.py::Variable::CONST");
    }

    #[test]
    fn python_method_qnames_do_not_collide_across_classes() {
        // Two `__init__` methods on different classes must get distinct qnames.
        const SRC: &str = r#"
class Foo:
    def __init__(self):
        pass

class Bar:
    def __init__(self):
        pass
"#;
        let r = py(SRC, "m.py");
        let inits: std::collections::HashSet<String> = r
            .nodes
            .iter()
            .filter(|n| n.label == "Method" && n.name == "__init__")
            .map(|n| n.qualified_name.clone())
            .collect();
        assert!(
            inits.contains("m.py::Foo::__init__"),
            "missing Foo::__init__: {inits:?}"
        );
        assert!(
            inits.contains("m.py::Bar::__init__"),
            "missing Bar::__init__: {inits:?}"
        );
    }

    #[test]
    fn python_calls_capture_final_callee_and_enclosing_caller() {
        const SRC: &str = r#"
def caller():
    bare()
    obj.method()
    pkg.mod.deep()
"#;
        let r = py(SRC, "c.py");
        let edges: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .map(|e| {
                (
                    e.source_qualified_name.as_str(),
                    e.properties
                        .get("callee_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                )
            })
            .collect();
        // bare() → callee `bare`, caller `caller`.
        assert!(
            edges.contains(&("c.py::Function::caller", "bare")),
            "bare call: {edges:?}"
        );
        // obj.method() → final identifier `method`.
        assert!(
            edges.contains(&("c.py::Function::caller", "method")),
            "attribute call must capture final `method`: {edges:?}"
        );
        // pkg.mod.deep() → final identifier `deep`.
        assert!(
            edges.contains(&("c.py::Function::caller", "deep")),
            "chained call must capture final `deep`: {edges:?}"
        );
        // The receiver object is NOT the callee.
        assert!(
            !edges.iter().any(|(_, callee)| *callee == "obj"),
            "receiver `obj` must not be a callee: {edges:?}"
        );
    }

    #[test]
    fn python_import_statement_emits_imports_edge() {
        const SRC: &str = r#"
import os
import a.b as c
"#;
        let r = py(SRC, "i.py");
        let imp: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .map(|e| {
                let p = &e.properties;
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                (s("path"), s("imported_name"))
            })
            .collect();
        // `import os` → path os, bound name os.
        assert!(
            imp.contains(&("os".into(), "os".into())),
            "import os: {imp:?}"
        );
        // `import a.b as c` → path a.b, bound name c (the alias).
        assert!(
            imp.contains(&("a.b".into(), "c".into())),
            "aliased import binds the alias: {imp:?}"
        );
    }

    #[test]
    fn python_from_import_emits_imports_edge_with_imported_name() {
        // A single `from pkg.mod import helper` keeps the shared pass's
        // `imported_name` / `original_name` properties for the cross-file
        // resolver.
        const SRC: &str = r#"
from pkg.mod import helper
"#;
        let r = py(SRC, "f.py");
        let imp: Vec<_> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .collect();
        assert_eq!(imp.len(), 1, "one import: {imp:?}");
        assert_eq!(
            imp[0]
                .properties
                .get("imported_name")
                .and_then(|v| v.as_str()),
            Some("helper"),
            "from-import must bind `helper`"
        );
    }

    #[test]
    fn python_imports_collapse_to_one_edge_per_module() {
        // An import is ONE edge per statement targeting the imported *module*,
        // deduped by module across statements. `extract_python`'s collapse: two
        // `from pkg.mod import …` statements (same module) yield a single IMPORTS
        // edge, and a multi-name `from a import x, y` counts once.
        const SRC: &str = r#"
from pkg.mod import helper
from pkg.mod import thing as aliased
from other import x, y
"#;
        let r = py(SRC, "f.py");
        let imports: Vec<_> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .collect();
        // `pkg.mod` (two statements) collapses to one; `other` (multi-name)
        // collapses to one → two module imports total.
        assert_eq!(
            imports.len(),
            2,
            "same-module + multi-name imports collapse per module: {imports:?}"
        );
        let modules: std::collections::HashSet<&str> = imports
            .iter()
            .filter_map(|e| e.properties.get("path").and_then(|v| v.as_str()))
            .map(|p| p.rsplit_once('.').map(|(m, _)| m).unwrap_or(p))
            .collect();
        assert!(modules.contains("pkg.mod"), "keeps pkg.mod: {modules:?}");
        assert!(modules.contains("other"), "keeps other: {modules:?}");
    }

    #[test]
    fn python_wildcard_import_is_single_glob_edge() {
        const SRC: &str = "from pkg import *\n";
        let r = py(SRC, "w.py");
        let imp: Vec<_> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .collect();
        assert_eq!(imp.len(), 1, "wildcard is a single edge: {imp:?}");
        assert_eq!(
            imp[0].properties.get("glob").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            imp[0]
                .properties
                .get("imported_name")
                .and_then(|v| v.as_str()),
            Some(""),
            "glob carries an empty imported_name"
        );
    }

    #[test]
    fn python_docstrings_attach_to_function_and_class() {
        const SRC: &str = "
def documented():
    \"\"\"Does a thing.

    Extended description.
    \"\"\"
    pass

class Widget:
    '''A widget.'''
    pass

def bare():
    pass
";
        let r = py(SRC, "d.py");
        let doc = |name: &str| -> (String, String) {
            let n = r.nodes.iter().find(|n| n.name == name).unwrap();
            let p = &n.properties;
            let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
            (s("doc"), s("doc_full"))
        };
        let (fdoc, ffull) = doc("documented");
        assert_eq!(fdoc, "Does a thing.", "doc summary = first non-empty line");
        assert!(
            ffull.contains("Extended description."),
            "doc_full keeps the body: {ffull:?}"
        );
        let (cdoc, _) = doc("Widget");
        assert_eq!(cdoc, "A widget.", "class docstring");
        // A def whose first statement is not a string has no doc property.
        let (bdoc, bfull) = doc("bare");
        assert_eq!(bdoc, "");
        assert_eq!(bfull, "");
    }

    #[test]
    fn python_cross_file_call_resolves_by_callee_name() {
        // a.py calls a function defined in b.py. The indexer's name-based
        // resolver keys on `callee_name`; here we assert the producer side
        // emits the matching name on both ends so resolution succeeds with NO
        // indexer change.
        const A: &str = r#"
from b import shared

def use_it():
    shared()
"#;
        const B: &str = r#"
def shared():
    return 1
"#;
        let ra = py(A, "a.py");
        let rb = py(B, "b.py");
        // a.py emits a CALLS edge whose callee_name is `shared`.
        let callee_names: std::collections::HashSet<String> = ra
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        assert!(
            callee_names.contains("shared"),
            "a.py must emit CALLS callee_name=shared: {callee_names:?}"
        );
        // b.py emits a Function node named `shared` — the resolver target.
        assert!(
            rb.nodes
                .iter()
                .any(|n| n.label == "Function" && n.name == "shared"),
            "b.py must define a Function named shared: {:?}",
            rb.nodes
        );
        // The bare callee name on the edge equals the definition's `name`,
        // which is exactly the key the two-phase resolver matches on.
        let def = rb
            .nodes
            .iter()
            .find(|n| n.label == "Function" && n.name == "shared")
            .unwrap();
        assert!(
            callee_names.contains(&def.name),
            "callee_name must match the cross-file definition's name"
        );
    }

    #[test]
    fn python_unsupported_no_longer_returned_for_py() {
        // extract() must NOT return NotImplemented for Python anymore.
        let r = extract(Language::Python, b"def f():\n    pass\n", "x.py");
        assert!(r.is_ok(), "Python extraction must succeed: {r:?}");
        assert!(
            !r.unwrap().is_empty(),
            "Python extraction must produce nodes"
        );
    }

    // ---- JavaScript extraction ----

    const JS_SRC: &str = r#"
import { helper, util as u } from "./lib";
import def from "./d";
import * as ns from "./n";
const cjs = require("./cjs");
const { destructured } = require("./d2");

/** Adds two numbers together. */
function add(a, b) {
    return a + b;
}

const mul = (a, b) => a * b;

class Calc {
    /** Computes a value. */
    compute(n) {
        return add(n, mul(n, 2));
    }
}

obj.run();
"#;

    fn js(src: &str, path: &str) -> crate::extract::ExtractionResult {
        extract(Language::JavaScript, src.as_bytes(), path).unwrap()
    }

    #[test]
    fn js_extract_returns_ok_for_all_extensions() {
        for path in ["a.js", "a.jsx", "a.mjs", "a.cjs"] {
            let lang = crate::language::language_for_path(std::path::Path::new(path));
            let r = extract(lang, JS_SRC.as_bytes(), path);
            assert!(r.is_ok(), "extract must be Ok for {path}: {r:?}");
        }
    }

    #[test]
    fn js_finds_functions_arrow_class_and_method() {
        let r = js(JS_SRC, "src/a.js");
        let by_name =
            |n: &str, label: &str| r.nodes.iter().any(|x| x.name == n && x.label == label);
        assert!(
            by_name("add", "Function"),
            "missing function add: {:?}",
            r.nodes
        );
        assert!(
            by_name("mul", "Function"),
            "missing arrow mul: {:?}",
            r.nodes
        );
        assert!(
            by_name("Calc", "Class"),
            "missing class Calc: {:?}",
            r.nodes
        );
        assert!(
            by_name("compute", "Method"),
            "missing method compute: {:?}",
            r.nodes
        );
    }

    #[test]
    fn js_method_qname_is_owned_by_class() {
        let r = js(JS_SRC, "src/a.js");
        let compute = r
            .nodes
            .iter()
            .find(|n| n.name == "compute" && n.label == "Method")
            .unwrap();
        assert_eq!(compute.qualified_name, "src/a.js::Calc::compute");
    }

    #[test]
    fn js_method_qnames_do_not_collide_across_classes() {
        const SRC: &str = r#"
class Foo { run() {} }
class Bar { run() {} }
"#;
        let r = js(SRC, "src/a.js");
        let qnames: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.name == "run" && n.label == "Method")
            .map(|n| n.qualified_name.as_str())
            .collect();
        assert!(
            qnames.contains("src/a.js::Foo::run") && qnames.contains("src/a.js::Bar::run"),
            "expected distinct Foo::run and Bar::run qnames, got {qnames:?}"
        );
    }

    #[test]
    fn js_jsdoc_becomes_doc_property() {
        let r = js(JS_SRC, "src/a.js");
        let add = r.nodes.iter().find(|n| n.name == "add").unwrap();
        assert_eq!(
            add.properties.get("doc").and_then(|v| v.as_str()),
            Some("Adds two numbers together."),
            "add JSDoc summary missing: {:?}",
            add.properties
        );
        let compute = r.nodes.iter().find(|n| n.name == "compute").unwrap();
        assert_eq!(
            compute.properties.get("doc").and_then(|v| v.as_str()),
            Some("Computes a value."),
            "compute JSDoc summary missing: {:?}",
            compute.properties
        );
    }

    #[test]
    fn js_module_level_var_and_require_bindings_become_variables() {
        // Module-level `const`/`let`/`var` bindings are `Variable` definition
        // nodes — including the names bound by `const { x } = require(...)` (a
        // require-valued declarator is a call_expression, NOT skipped). A
        // declarator whose value is an arrow / function expression is a Function,
        // not a Variable.
        const SRC: &str = r#"
const plain = 42;
let counter = 0;
const { alpha, beta } = require("./m");
const [first, second] = arr;
const arrow = (x) => x;
function real() {}
"#;
        let r = js(SRC, "src/a.js");
        let var_names: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.label == "Variable")
            .map(|n| n.name.as_str())
            .collect();
        for expected in ["plain", "counter", "alpha", "beta", "first", "second"] {
            assert!(
                var_names.contains(expected),
                "missing Variable {expected}: {var_names:?}"
            );
        }
        // Arrow/function-valued declarator is a Function, never a Variable.
        assert!(
            !var_names.contains("arrow"),
            "arrow binding must not be a Variable: {var_names:?}"
        );
        assert!(
            r.nodes
                .iter()
                .any(|n| n.name == "arrow" && n.label == "Function"),
            "arrow binding should be a Function"
        );
        // A require-bound Variable carries the greppy qname scheme.
        let alpha = r
            .nodes
            .iter()
            .find(|n| n.name == "alpha" && n.label == "Variable")
            .unwrap();
        assert_eq!(alpha.qualified_name, "src/a.js::Variable::alpha");
    }

    #[test]
    fn js_function_body_locals_are_not_module_variables() {
        // Only module-level declarations are Variables; a `const` inside a
        // function body is a local, not a module Variable.
        const SRC: &str = r#"
function f() {
    const localOnly = 1;
    return localOnly;
}
"#;
        let r = js(SRC, "src/a.js");
        assert!(
            !r.nodes
                .iter()
                .any(|n| n.name == "localOnly" && n.label == "Variable"),
            "function-body local must not be a module Variable: {:?}",
            r.nodes
        );
    }

    #[test]
    fn ts_enum_members_become_variables_owned_by_enum() {
        // Each TS enum member is a `Variable` owned by the enum (qname
        // `{file}::{Enum}::{member}`).
        const SRC: &str = r#"
export enum Color {
    Red,
    Green = 2,
    Blue,
}
"#;
        let r = extract(
            Language::TypeScript { tsx: false },
            SRC.as_bytes(),
            "src/a.ts",
        )
        .unwrap();
        let member_qnames: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.label == "Variable")
            .map(|n| n.qualified_name.as_str())
            .collect();
        for q in [
            "src/a.ts::Color::Red",
            "src/a.ts::Color::Green",
            "src/a.ts::Color::Blue",
        ] {
            assert!(
                member_qnames.contains(q),
                "missing enum member {q}: {member_qnames:?}"
            );
        }
        // The enum itself is still an Enum node (from the shared def pass).
        assert!(
            r.nodes
                .iter()
                .any(|n| n.name == "Color" && n.label == "Enum"),
            "enum Color should be an Enum node"
        );
    }

    #[test]
    fn js_calls_capture_final_callee_and_skip_require() {
        // `obj.run()` sits inside `add` so it produces a CALLS edge (a
        // top-level call has no enclosing function and only yields a Call node).
        const SRC: &str = r#"
const cjs = require("./cjs");
function add(a, b) {
    obj.run();
    return helper(a) + b;
}
const mul = (a, b) => compute(a) * b;
"#;
        let r = js(SRC, "src/a.js");
        let callees: std::collections::HashSet<String> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();
        assert!(
            callees.contains("helper"),
            "bare call helper missing: {callees:?}"
        );
        assert!(
            callees.contains("compute"),
            "arrow body call missing: {callees:?}"
        );
        assert!(
            callees.contains("run"),
            "member call obj.run() must capture `run`: {callees:?}"
        );
        assert!(
            !callees.contains("require"),
            "require() must be owned by imports, not CALLS: {callees:?}"
        );
        assert!(
            !callees.contains("obj"),
            "must not capture receiver `obj`: {callees:?}"
        );
        // The require() call also must not surface as a Call node.
        assert!(
            !r.nodes
                .iter()
                .any(|n| n.label == "Call" && n.name == "require"),
            "require must not produce a Call node: {:?}",
            r.nodes
        );
    }

    #[test]
    fn js_call_source_is_enclosing_function() {
        let r = js(JS_SRC, "src/a.js");
        // `compute` calls `add` and `mul`; the CALLS edge source must be the
        // method's owned qname.
        let from_compute: Vec<&str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter(|e| e.source_qualified_name == "src/a.js::Calc::compute")
            .filter_map(|e| e.properties.get("callee_name").and_then(|v| v.as_str()))
            .collect();
        assert!(
            from_compute.contains(&"add") && from_compute.contains(&"mul"),
            "compute must CALL add and mul: {from_compute:?}"
        );
    }

    #[test]
    fn js_imports_named_default_namespace_alias_and_require() {
        let r = js(JS_SRC, "src/a.js");
        // imported_name -> (path, original_name) for IMPORTS edges.
        let imports: std::collections::HashMap<String, (String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .map(|e| {
                (
                    e.properties
                        .get("imported_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    (
                        e.properties
                            .get("path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        e.properties
                            .get("original_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                )
            })
            .collect();
        // Named import.
        assert_eq!(imports.get("helper").map(|p| p.0.as_str()), Some("./lib"));
        // Aliased named import: local binding `u`, original `util`.
        assert_eq!(
            imports.get("u").map(|p| (p.0.as_str(), p.1.as_str())),
            Some(("./lib", "util")),
            "aliased import util as u not resolved: {imports:?}"
        );
        // Default import.
        assert_eq!(imports.get("def").map(|p| p.0.as_str()), Some("./d"));
        // Namespace import.
        assert_eq!(imports.get("ns").map(|p| p.0.as_str()), Some("./n"));
        // require() whole-module + destructured.
        assert_eq!(imports.get("cjs").map(|p| p.0.as_str()), Some("./cjs"));
        assert_eq!(
            imports.get("destructured").map(|p| p.0.as_str()),
            Some("./d2"),
            "destructured require binding missing: {imports:?}"
        );
    }

    #[test]
    fn js_cross_file_call_resolves_by_callee_name() {
        // a.js calls `shared`, defined in b.js. The cross-file resolver keys on
        // the callee_name matching the definition's `name` — verify both sides.
        let ra = js("function caller() { shared(); }", "a.js");
        let callee_names: std::collections::HashSet<String> = ra
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();
        assert!(
            callee_names.contains("shared"),
            "a.js must emit CALLS callee_name=shared: {callee_names:?}"
        );
        let rb = js("function shared() {}", "b.js");
        let def = rb
            .nodes
            .iter()
            .find(|n| n.label == "Function" && n.name == "shared")
            .expect("b.js must define Function shared");
        assert!(
            callee_names.contains(&def.name),
            "callee_name must match the cross-file definition's name"
        );
    }

    // ---- TypeScript extraction ----

    const TS_SRC: &str = r#"
import { Shape } from "./shape";

/** A unique identifier. */
type ID = string | number;

interface Repo {
    find(id: ID): Shape;
}

enum Color { Red, Green, Blue }

function load(id: ID): Shape {
    return lookup(id);
}

class FileRepo implements Repo {
    find(id: ID): Shape {
        return load(id);
    }
}
"#;

    fn ts(src: &str, path: &str) -> crate::extract::ExtractionResult {
        let lang = crate::language::language_for_path(std::path::Path::new(path));
        extract(lang, src.as_bytes(), path).unwrap()
    }

    #[test]
    fn ts_extract_returns_ok_for_ts_and_tsx() {
        let r_ts = extract(
            Language::TypeScript { tsx: false },
            TS_SRC.as_bytes(),
            "a.ts",
        );
        assert!(r_ts.is_ok(), "extract must be Ok for .ts: {r_ts:?}");

        const TSX: &str = r#"
const App = () => <div className="x">hi</div>;
export function Comp(): JSX.Element { return <span/>; }
"#;
        let r_tsx = extract(Language::TypeScript { tsx: true }, TSX.as_bytes(), "a.tsx");
        assert!(r_tsx.is_ok(), "extract must be Ok for .tsx: {r_tsx:?}");
        // The arrow component and the function must both be found.
        let r_tsx = r_tsx.unwrap();
        assert!(
            r_tsx.nodes.iter().any(|n| n.name == "App"),
            "tsx arrow App missing: {:?}",
            r_tsx.nodes
        );
        assert!(
            r_tsx.nodes.iter().any(|n| n.name == "Comp"),
            "tsx function Comp missing: {:?}",
            r_tsx.nodes
        );
    }

    #[test]
    fn ts_finds_interface_type_enum_function_class_method() {
        let r = ts(TS_SRC, "src/a.ts");
        let has = |n: &str, label: &str| r.nodes.iter().any(|x| x.name == n && x.label == label);
        assert!(
            has("Repo", "Interface"),
            "missing interface Repo: {:?}",
            r.nodes
        );
        assert!(has("ID", "Type"), "missing type alias ID: {:?}", r.nodes);
        assert!(has("Color", "Enum"), "missing enum Color: {:?}", r.nodes);
        assert!(
            has("load", "Function"),
            "missing function load: {:?}",
            r.nodes
        );
        assert!(
            has("FileRepo", "Class"),
            "missing class FileRepo: {:?}",
            r.nodes
        );
        assert!(has("find", "Method"), "missing method find: {:?}", r.nodes);
    }

    #[test]
    fn ts_type_alias_jsdoc_doc_property() {
        let r = ts(TS_SRC, "src/a.ts");
        let id = r.nodes.iter().find(|n| n.name == "ID").unwrap();
        assert_eq!(
            id.properties.get("doc").and_then(|v| v.as_str()),
            Some("A unique identifier."),
            "type alias JSDoc missing: {:?}",
            id.properties
        );
    }

    #[test]
    fn ts_interface_and_type_qnames() {
        let r = ts(TS_SRC, "src/a.ts");
        let repo = r.nodes.iter().find(|n| n.name == "Repo").unwrap();
        assert_eq!(repo.qualified_name, "src/a.ts::Interface::Repo");
        let id = r.nodes.iter().find(|n| n.name == "ID").unwrap();
        assert_eq!(id.qualified_name, "src/a.ts::Type::ID");
        let find = r
            .nodes
            .iter()
            .find(|n| n.name == "find" && n.label == "Method")
            .unwrap();
        assert_eq!(find.qualified_name, "src/a.ts::FileRepo::find");
    }

    #[test]
    fn ts_cross_file_call_resolves_by_callee_name() {
        // FileRepo.find calls `load`; `load` calls `lookup`. The `lookup`
        // definition lives in another file — verify callee_name keying.
        let ra = ts(TS_SRC, "a.ts");
        let callees: std::collections::HashSet<String> = ra
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();
        assert!(
            callees.contains("lookup"),
            "load must call lookup: {callees:?}"
        );
        assert!(callees.contains("load"), "find must call load: {callees:?}");
        // The cross-file target definition.
        let rb = ts("export function lookup(id) { return id; }", "b.ts");
        let def = rb
            .nodes
            .iter()
            .find(|n| n.label == "Function" && n.name == "lookup")
            .expect("b.ts must define Function lookup");
        assert!(callees.contains(&def.name));
    }

    #[test]
    fn ts_import_edge_emitted() {
        let r = ts(TS_SRC, "src/a.ts");
        let shape = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .find(|e| e.properties.get("imported_name").and_then(|v| v.as_str()) == Some("Shape"));
        assert!(shape.is_some(), "Shape import edge missing: {:?}", r.edges);
        assert_eq!(
            shape
                .unwrap()
                .properties
                .get("path")
                .and_then(|v| v.as_str()),
            Some("./shape")
        );
    }

    // -----------------------------------------------------------------------
    // Go
    // -----------------------------------------------------------------------

    const GO_SRC: &str = r#"
// Package main does things.
package main

import (
    "fmt"
    "strings"
    m "math/rand"
)

// Adder holds a base value.
type Adder struct {
    base int
}

// Shape is a thing with area.
type Shape interface {
    Area() float64
}

// add returns the sum of a and b.
func add(a int, b int) int {
    return a + b
}

func (r *Adder) Compute(n int) int {
    x := add(n, r.base)
    fmt.Println(x)
    return strings.ToUpper("hi") == "" || m.Intn(3) == 0
}
"#;

    fn go(src: &str, path: &str) -> crate::extract::ExtractionResult {
        extract(Language::Go, src.as_bytes(), path).unwrap()
    }

    #[test]
    fn go_extract_returns_ok() {
        let lang = crate::language::language_for_path(std::path::Path::new("main.go"));
        let r = extract(lang, GO_SRC.as_bytes(), "main.go");
        assert!(r.is_ok(), "extract must be Ok for .go: {r:?}");
    }

    #[test]
    fn go_finds_function_method_struct_and_interface() {
        let r = go(GO_SRC, "src/a.go");
        let by = |n: &str, label: &str| r.nodes.iter().any(|x| x.name == n && x.label == label);
        assert!(by("add", "Function"), "missing func add: {:?}", r.nodes);
        assert!(
            by("Compute", "Method"),
            "missing method Compute: {:?}",
            r.nodes
        );
        // A Go `struct` type is labeled `Class` (NOT
        // `Struct`).
        assert!(
            by("Adder", "Class"),
            "missing struct Adder (labeled Class): {:?}",
            r.nodes
        );
        assert!(
            by("Shape", "Interface"),
            "missing interface Shape: {:?}",
            r.nodes
        );
    }

    #[test]
    fn go_method_qname_is_owned_by_receiver_type() {
        let r = go(GO_SRC, "src/a.go");
        let compute = r
            .nodes
            .iter()
            .find(|n| n.name == "Compute" && n.label == "Method")
            .unwrap();
        assert_eq!(compute.qualified_name, "src/a.go::Adder::Compute");
    }

    #[test]
    fn go_method_qnames_do_not_collide_across_receivers() {
        const SRC: &str = r#"
package main
type Foo struct{}
type Bar struct{}
func (f Foo) Run() {}
func (b Bar) Run() {}
"#;
        let r = go(SRC, "src/a.go");
        let qnames: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.name == "Run" && n.label == "Method")
            .map(|n| n.qualified_name.as_str())
            .collect();
        assert!(
            qnames.contains("src/a.go::Foo::Run") && qnames.contains("src/a.go::Bar::Run"),
            "expected distinct Foo::Run and Bar::Run qnames, got {qnames:?}"
        );
    }

    #[test]
    fn go_doc_comment_becomes_doc_property() {
        let r = go(GO_SRC, "src/a.go");
        let add = r.nodes.iter().find(|n| n.name == "add").unwrap();
        assert_eq!(
            add.properties.get("doc").and_then(|v| v.as_str()),
            Some("add returns the sum of a and b."),
            "add doc summary missing: {:?}",
            add.properties
        );
        let adder = r.nodes.iter().find(|n| n.name == "Adder").unwrap();
        assert_eq!(
            adder.properties.get("doc").and_then(|v| v.as_str()),
            Some("Adder holds a base value."),
            "Adder doc summary missing: {:?}",
            adder.properties
        );
    }

    #[test]
    fn go_calls_capture_bare_and_selector_callee() {
        let r = go(GO_SRC, "src/a.go");
        let callees: std::collections::HashSet<String> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        // bare `add()` and selector `fmt.Println()` / `strings.ToUpper()`.
        assert!(
            callees.contains("add"),
            "missing bare callee add: {callees:?}"
        );
        assert!(
            callees.contains("Println"),
            "missing selector callee Println: {callees:?}"
        );
        assert!(
            callees.contains("ToUpper"),
            "missing selector callee ToUpper: {callees:?}"
        );
    }

    #[test]
    fn go_cross_file_call_resolves_by_callee_name() {
        // `b.go` defines `helper`; `a.go` calls it. The CALLS edge from a.go
        // carries `callee_name = helper`, matching the Function node b.go emits.
        const A: &str = r#"
package main
func caller() int {
    return helper(1)
}
"#;
        const B: &str = r#"
package main
func helper(n int) int { return n + 1 }
"#;
        let a = go(A, "a.go");
        let b = go(B, "b.go");
        // a.go emits a CALLS edge whose callee_name is `helper`.
        let call_edge = a
            .edges
            .iter()
            .find(|e| {
                e.edge_type == "CALLS"
                    && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("helper")
            })
            .expect("a.go must emit CALLS edge for helper");
        assert_eq!(
            call_edge.source_qualified_name, "a.go::Function::caller",
            "CALLS source must be the enclosing caller: {call_edge:?}"
        );
        // b.go emits a Function node named `helper` the resolver keys on.
        assert!(
            b.nodes
                .iter()
                .any(|n| n.name == "helper" && n.label == "Function"),
            "b.go must define Function helper: {:?}",
            b.nodes
        );
    }

    #[test]
    fn go_imports_emit_edges_with_final_segment_and_alias() {
        let r = go(GO_SRC, "src/a.go");
        let imported: std::collections::HashMap<&str, &str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .filter_map(|e| {
                let name = e.properties.get("imported_name").and_then(|v| v.as_str())?;
                let path = e.properties.get("path").and_then(|v| v.as_str())?;
                Some((name, path))
            })
            .collect();
        // plain import -> final segment as binding.
        assert_eq!(
            imported.get("fmt"),
            Some(&"fmt"),
            "fmt import: {imported:?}"
        );
        assert_eq!(
            imported.get("strings"),
            Some(&"strings"),
            "strings import: {imported:?}"
        );
        // aliased import `m "math/rand"` -> binding `m`, path `math/rand`.
        assert_eq!(
            imported.get("m"),
            Some(&"math/rand"),
            "aliased import m: {imported:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Ruby
    // -----------------------------------------------------------------------

    const RB_SRC: &str = r#"
require "json"
require_relative "./helper"

# Greeter greets people.
class Greeter
  # initialize stores the name.
  def initialize(name)
    @name = name
  end

  def greet
    puts "hi"
    helper_fn(@name)
  end
end

module Util
  def self.run
    Greeter.new("x").greet
  end
end

def top_level
  Util.run
end
"#;

    fn rb(src: &str, path: &str) -> crate::extract::ExtractionResult {
        extract(Language::Ruby, src.as_bytes(), path).unwrap()
    }

    #[test]
    fn ruby_extract_returns_ok() {
        let lang = crate::language::language_for_path(std::path::Path::new("main.rb"));
        let r = extract(lang, RB_SRC.as_bytes(), "main.rb");
        assert!(r.is_ok(), "extract must be Ok for .rb: {r:?}");
    }

    #[test]
    fn ruby_finds_class_module_method_and_singleton() {
        let r = rb(RB_SRC, "src/a.rb");
        let by = |n: &str, label: &str| r.nodes.iter().any(|x| x.name == n && x.label == label);
        assert!(
            by("Greeter", "Class"),
            "missing class Greeter: {:?}",
            r.nodes
        );
        // A Ruby `module` declaration is labelled a "Class" (the label
        // defaults to "Class"; a module is not Interface/Enum/Type). The
        // per-file "Module" slot is the synthetic node the indexer adds, not
        // the decl.
        assert!(
            by("Util", "Class"),
            "module Util must be a Class: {:?}",
            r.nodes
        );
        assert!(by("greet", "Method"), "missing method greet: {:?}", r.nodes);
        assert!(
            by("run", "Method"),
            "missing singleton method run: {:?}",
            r.nodes
        );
        assert!(
            by("top_level", "Function"),
            "missing top-level def top_level: {:?}",
            r.nodes
        );
    }

    #[test]
    fn ruby_method_qname_is_owned_by_class() {
        let r = rb(RB_SRC, "src/a.rb");
        let greet = r
            .nodes
            .iter()
            .find(|n| n.name == "greet" && n.label == "Method")
            .unwrap();
        assert_eq!(greet.qualified_name, "src/a.rb::Greeter::greet");
    }

    #[test]
    fn ruby_method_qnames_do_not_collide_across_classes() {
        const SRC: &str = r#"
class Foo
  def run; end
end
class Bar
  def run; end
end
"#;
        let r = rb(SRC, "src/a.rb");
        let qnames: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.name == "run" && n.label == "Method")
            .map(|n| n.qualified_name.as_str())
            .collect();
        assert!(
            qnames.contains("src/a.rb::Foo::run") && qnames.contains("src/a.rb::Bar::run"),
            "expected distinct Foo::run and Bar::run qnames, got {qnames:?}"
        );
    }

    #[test]
    fn ruby_doc_comment_becomes_doc_property() {
        let r = rb(RB_SRC, "src/a.rb");
        let greeter = r.nodes.iter().find(|n| n.name == "Greeter").unwrap();
        assert_eq!(
            greeter.properties.get("doc").and_then(|v| v.as_str()),
            Some("Greeter greets people."),
            "Greeter doc summary missing: {:?}",
            greeter.properties
        );
        let init = r.nodes.iter().find(|n| n.name == "initialize").unwrap();
        assert_eq!(
            init.properties.get("doc").and_then(|v| v.as_str()),
            Some("initialize stores the name."),
            "initialize doc summary missing: {:?}",
            init.properties
        );
    }

    #[test]
    fn ruby_calls_capture_method_name_and_skip_require() {
        let r = rb(RB_SRC, "src/a.rb");
        let callees: std::collections::HashSet<String> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        assert!(
            callees.contains("helper_fn"),
            "missing callee helper_fn: {callees:?}"
        );
        assert!(callees.contains("puts"), "missing callee puts: {callees:?}");
        // require / require_relative are imports, not CALLS.
        assert!(
            !callees.contains("require") && !callees.contains("require_relative"),
            "require must not appear as a call: {callees:?}"
        );
    }

    #[test]
    fn ruby_constant_receiver_call_preserves_singleton_owner() {
        let r = rb("def caller\n  Helper.do_it\nend\n", "app.rb");
        let edge = r
            .edges
            .iter()
            .find(|edge| {
                edge.edge_type == "CALLS"
                    && edge.properties.get("callee_name").and_then(|v| v.as_str()) == Some("do_it")
            })
            .expect("Helper.do_it must emit a CALLS edge");
        assert_eq!(edge.target_qualified_name, "app.rb::Helper::do_it");
        assert_eq!(
            edge.properties.get("callee_form").and_then(|v| v.as_str()),
            Some("receiver")
        );
        assert_eq!(
            edge.properties
                .get("receiver_owner")
                .and_then(|v| v.as_str()),
            Some("Helper")
        );
    }

    #[test]
    fn ruby_classifies_constant_receiver_type_refs_and_identifier_uses() {
        let r = rb(
            "def render(input)\n  widget = Widget.new\n  widget\nend\n",
            "app.rb",
        );
        let has_ref = |kind: &str, property: &str, name: &str| {
            r.edges.iter().any(|edge| {
                edge.edge_type == kind
                    && edge.source_qualified_name == "app.rb::Function::render"
                    && edge
                        .properties
                        .get(property)
                        .and_then(|value| value.as_str())
                        == Some(name)
                    && edge
                        .properties
                        .get("preserve_reference_kind")
                        .and_then(|value| value.as_bool())
                        == Some(true)
            })
        };
        assert!(
            has_ref("TYPE_REF", "type_name", "Widget"),
            "Widget.new receiver must classify as TYPE_REF: {:?}",
            r.edges
        );
        assert!(
            has_ref("USES", "ref_name", "widget"),
            "returned local must classify as USES: {:?}",
            r.edges
        );
        assert!(
            !r.edges.iter().any(|edge| edge.edge_type == "USAGE"),
            "Ruby reference kinds must not collapse during extraction: {:?}",
            r.edges
        );
    }

    #[test]
    fn ruby_classifies_qualified_and_bare_constants_as_uses() {
        let r = rb(
            "def report_limit\n  Helper::LIMIT\n  LIMIT\nend\n",
            "app.rb",
        );
        let refs: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter_map(|edge| {
                let name = edge
                    .properties
                    .get(if edge.edge_type == "TYPE_REF" {
                        "type_name"
                    } else {
                        "ref_name"
                    })?
                    .as_str()?;
                Some((edge.edge_type.as_str(), name))
            })
            .collect();
        assert!(
            refs.contains(&("TYPE_REF", "Helper")),
            "scope owner Helper must classify as TYPE_REF: {refs:?}"
        );
        assert_eq!(
            refs.iter()
                .filter(|(kind, name)| *kind == "USES" && *name == "LIMIT")
                .count(),
            2,
            "qualified and bare LIMIT reads must classify as USES: {refs:?}"
        );
    }

    #[test]
    fn ruby_cross_file_call_resolves_by_callee_name() {
        // `b.rb` defines `helper_fn`; `a.rb` calls it. The CALLS edge carries
        // `callee_name = helper_fn`, matching the Function node b.rb emits.
        const A: &str = r#"
def caller
  helper_fn(1)
end
"#;
        const B: &str = r#"
def helper_fn(n)
  n + 1
end
"#;
        let a = rb(A, "a.rb");
        let b = rb(B, "b.rb");
        let call_edge = a
            .edges
            .iter()
            .find(|e| {
                e.edge_type == "CALLS"
                    && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("helper_fn")
            })
            .expect("a.rb must emit CALLS edge for helper_fn");
        assert_eq!(
            call_edge.source_qualified_name, "a.rb::Function::caller",
            "CALLS source must be the enclosing caller: {call_edge:?}"
        );
        assert!(
            b.nodes
                .iter()
                .any(|n| n.name == "helper_fn" && n.label == "Function"),
            "b.rb must define Function helper_fn: {:?}",
            b.nodes
        );
    }

    #[test]
    fn ruby_require_emits_import_edges() {
        let r = rb(RB_SRC, "src/a.rb");
        let imported: std::collections::HashMap<&str, &str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .filter_map(|e| {
                let name = e.properties.get("imported_name").and_then(|v| v.as_str())?;
                let path = e.properties.get("path").and_then(|v| v.as_str())?;
                Some((name, path))
            })
            .collect();
        // `require "json"` -> binding json, path json.
        assert_eq!(
            imported.get("json"),
            Some(&"json"),
            "json require: {imported:?}"
        );
        // `require_relative "./helper"` -> binding helper, path ./helper.
        assert_eq!(
            imported.get("helper"),
            Some(&"./helper"),
            "helper require_relative: {imported:?}"
        );
        let helper = r
            .edges
            .iter()
            .find(|edge| {
                edge.edge_type == "IMPORTS"
                    && edge
                        .properties
                        .get("imported_name")
                        .and_then(|value| value.as_str())
                        == Some("helper")
            })
            .expect("require_relative helper edge");
        assert_eq!(
            helper
                .properties
                .get("filesystem_module_import")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            helper
                .properties
                .get("ruby_require_relative")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    // -----------------------------------------------------------------------
    // Java
    // -----------------------------------------------------------------------

    const JAVA_SRC: &str = r#"
import java.util.List;
import java.util.Map.Entry;

/** Greeter greets people. */
public class Greeter {
    /** initialize stores the name. */
    public Greeter(String name) {
        this.name = name;
    }

    public void greet() {
        System.out.println("hi");
        helperFn(name);
    }
}

interface Shape {
    double area();
}

enum Color { RED, GREEN }
"#;

    fn java(src: &str, path: &str) -> crate::extract::ExtractionResult {
        extract(Language::Java, src.as_bytes(), path).unwrap()
    }

    #[test]
    fn java_extract_returns_ok() {
        let lang = crate::language::language_for_path(std::path::Path::new("Main.java"));
        let r = extract(lang, JAVA_SRC.as_bytes(), "Main.java");
        assert!(r.is_ok(), "extract must be Ok for .java: {r:?}");
    }

    #[test]
    fn java_finds_class_interface_enum_and_method() {
        let r = java(JAVA_SRC, "src/G.java");
        let by = |n: &str, label: &str| r.nodes.iter().any(|x| x.name == n && x.label == label);
        assert!(
            by("Greeter", "Class"),
            "missing class Greeter: {:?}",
            r.nodes
        );
        assert!(
            by("Shape", "Interface"),
            "missing interface Shape: {:?}",
            r.nodes
        );
        assert!(by("Color", "Enum"), "missing enum Color: {:?}", r.nodes);
        assert!(by("greet", "Method"), "missing method greet: {:?}", r.nodes);
    }

    #[test]
    fn java_method_qname_is_owned_by_class() {
        let r = java(JAVA_SRC, "src/G.java");
        let greet = r
            .nodes
            .iter()
            .find(|n| n.name == "greet" && n.label == "Method")
            .unwrap();
        assert_eq!(greet.qualified_name, "src/G.java::Greeter::greet");
    }

    #[test]
    fn java_method_qnames_do_not_collide_across_classes() {
        const SRC: &str = r#"
class Foo { void run() {} }
class Bar { void run() {} }
"#;
        let r = java(SRC, "src/a.java");
        let qnames: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.name == "run" && n.label == "Method")
            .map(|n| n.qualified_name.as_str())
            .collect();
        assert!(
            qnames.contains("src/a.java::Foo::run") && qnames.contains("src/a.java::Bar::run"),
            "expected distinct Foo::run and Bar::run qnames, got {qnames:?}"
        );
    }

    #[test]
    fn java_javadoc_becomes_doc_property() {
        let r = java(JAVA_SRC, "src/G.java");
        let greeter = r
            .nodes
            .iter()
            .find(|n| n.name == "Greeter" && n.label == "Class")
            .unwrap();
        assert_eq!(
            greeter.properties.get("doc").and_then(|v| v.as_str()),
            Some("Greeter greets people."),
            "Greeter javadoc summary missing: {:?}",
            greeter.properties
        );
    }

    #[test]
    fn java_calls_capture_final_method_name() {
        let r = java(JAVA_SRC, "src/G.java");
        let callees: std::collections::HashSet<String> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        // bare `helperFn(...)` and qualified `System.out.println(...)`.
        assert!(
            callees.contains("helperFn"),
            "missing bare callee helperFn: {callees:?}"
        );
        assert!(
            callees.contains("println"),
            "missing qualified callee println: {callees:?}"
        );
    }

    #[test]
    fn java_cross_file_call_resolves_by_callee_name() {
        // `B.java` defines `helperFn` (a method); `A.java` calls it. The CALLS
        // edge from A.java carries `callee_name = helperFn`, matching the
        // method node B.java emits (resolved cross-file by name).
        const A: &str = r#"
class A {
    int caller() { return helperFn(1); }
}
"#;
        const B: &str = r#"
class B {
    int helperFn(int n) { return n + 1; }
}
"#;
        let a = java(A, "A.java");
        let b = java(B, "B.java");
        let call_edge = a
            .edges
            .iter()
            .find(|e| {
                e.edge_type == "CALLS"
                    && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("helperFn")
            })
            .expect("A.java must emit CALLS edge for helperFn");
        assert_eq!(
            call_edge.source_qualified_name, "A.java::A::caller",
            "CALLS source must be the enclosing caller: {call_edge:?}"
        );
        assert!(
            b.nodes
                .iter()
                .any(|n| n.name == "helperFn" && n.label == "Method"),
            "B.java must define method helperFn: {:?}",
            b.nodes
        );
    }

    #[test]
    fn java_imports_emit_edges_with_final_segment() {
        let r = java(JAVA_SRC, "src/G.java");
        let imported: std::collections::HashMap<&str, &str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .filter_map(|e| {
                let name = e.properties.get("imported_name").and_then(|v| v.as_str())?;
                let path = e.properties.get("path").and_then(|v| v.as_str())?;
                Some((name, path))
            })
            .collect();
        // `java.util.List` -> binding List.
        assert_eq!(
            imported.get("List"),
            Some(&"java.util.List"),
            "List import: {imported:?}"
        );
        // `java.util.Map.Entry` -> binding Entry.
        assert_eq!(
            imported.get("Entry"),
            Some(&"java.util.Map.Entry"),
            "Entry import: {imported:?}"
        );
    }

    #[test]
    fn java_usage_pass_emits_type_reference_usages() {
        // A type used in a non-call, non-import position (return type / local
        // variable type) becomes a USAGE edge from the enclosing method; the
        // callee of a `Foo.bar()` invocation and the `new Foo()` receiver stay
        // CALLS, not USAGE (references inside calls are skipped).
        const SRC: &str = r#"
package corpus.service;
import corpus.core.Widget;

public final class Svc {
    public Widget process(int n) {
        Widget w = Widget.build(n);
        return w;
    }
}
"#;
        let r = java(SRC, "src/main/java/corpus/service/Svc.java");
        let usages: Vec<&str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .filter_map(|e| e.properties.get("ref_name").and_then(|v| v.as_str()))
            .collect();
        // `Widget` appears as the return type and the local-variable type
        // (both usages); it must also appear as a USAGE. Its receiver form
        // `Widget.build(n)` is inside a call and must NOT double-count there,
        // but the two type positions are enough to require a USAGE.
        assert!(
            usages.contains(&"Widget"),
            "expected a USAGE for the type `Widget`, got {usages:?}"
        );
        // The usage source is the enclosing method's owned qname, so the edge
        // hangs off a real definition node.
        let widget_usage = r
            .edges
            .iter()
            .find(|e| {
                e.edge_type == "USAGE"
                    && e.properties.get("ref_name").and_then(|v| v.as_str()) == Some("Widget")
            })
            .expect("Widget USAGE edge");
        assert_eq!(
            widget_usage.source_qualified_name,
            "src/main/java/corpus/service/Svc.java::Svc::process",
            "usage source must be the enclosing method qname"
        );
        // Keywords / JDK builtins are filtered (e.g. `int`, `final`,
        // `public`) and never appear as usages.
        for kw in ["int", "final", "public", "String", "System"] {
            assert!(
                !usages.contains(&kw),
                "keyword/builtin `{kw}` must not be a USAGE, got {usages:?}"
            );
        }
    }

    #[test]
    fn java_same_package_imports_collapse_to_one_edge() {
        // Two imports from the SAME package must collapse to a single IMPORTS
        // edge (C models Java imports per package: both resolve to the same
        // package target and dedup). Imports from DIFFERENT packages each keep
        // their own edge.
        const SRC: &str = r#"
package corpus.service;
import corpus.core.Alpha;
import corpus.core.Beta;
import corpus.other.Gamma;

public final class Svc {
    public Alpha run() {
        return Alpha.make();
    }
}
"#;
        let r = java(SRC, "src/main/java/corpus/service/Svc.java");
        let import_names: Vec<&str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .filter_map(|e| e.properties.get("imported_name").and_then(|v| v.as_str()))
            .collect();
        // Exactly two IMPORTS survive: one for package `corpus.core`, one for
        // `corpus.other`.
        assert_eq!(
            import_names.len(),
            2,
            "same-package imports must collapse: {import_names:?}"
        );
        // The `corpus.other` package keeps its only symbol.
        assert!(
            import_names.contains(&"Gamma"),
            "distinct-package import must survive: {import_names:?}"
        );
        // The kept `corpus.core` import is the one referenced by a USAGE
        // (`Alpha` is the return type), so cross-file reference resolution is
        // preserved.
        assert!(
            import_names.contains(&"Alpha"),
            "referenced same-package import must be the survivor: {import_names:?}"
        );
    }

    // -----------------------------------------------------------------------
    // C
    // -----------------------------------------------------------------------

    const C_SRC: &str = r#"
#include <stdio.h>
#include "sub/helper.h"

/* add returns the sum. */
int add(int a, int b) {
    printf("%d", a);
    return helper(a) + b;
}

struct Point { int x; int y; };
union U { int i; float f; };
enum E { A, B };
typedef int MyInt;
"#;

    fn c(src: &str, path: &str) -> crate::extract::ExtractionResult {
        extract(Language::C, src.as_bytes(), path).unwrap()
    }

    #[test]
    fn c_extract_returns_ok_for_c_and_h() {
        // `.c` and `.h` both map to C.
        for path in ["a.c", "a.h"] {
            let lang = crate::language::language_for_path(std::path::Path::new(path));
            assert_eq!(lang, Language::C, "for {path}");
            let r = extract(lang, C_SRC.as_bytes(), path);
            assert!(r.is_ok(), "extract must be Ok for {path}: {r:?}");
        }
    }

    #[test]
    fn c_finds_function_struct_union_enum_and_typedef() {
        let r = c(C_SRC, "src/a.c");
        let by = |n: &str, label: &str| r.nodes.iter().any(|x| x.name == n && x.label == label);
        assert!(by("add", "Function"), "missing func add: {:?}", r.nodes);
        // `struct` / `union` are labelled "Class",
        // not "Struct" / "Union".
        assert!(by("Point", "Class"), "missing Class Point: {:?}", r.nodes);
        assert!(by("U", "Class"), "missing Class U: {:?}", r.nodes);
        assert!(by("E", "Enum"), "missing enum E: {:?}", r.nodes);
        // Enum members become module-scoped `Variable` nodes.
        assert!(by("A", "Variable"), "missing enum member A: {:?}", r.nodes);
        assert!(by("B", "Variable"), "missing enum member B: {:?}", r.nodes);
        // struct / union body fields become `Field` nodes.
        assert!(by("x", "Field"), "missing Field Point.x: {:?}", r.nodes);
        assert!(by("i", "Field"), "missing Field U.i: {:?}", r.nodes);
        // A typedef emits NO `Type` node — the UNIQUE(qname) store collapses
        // it, so zero standalone Type nodes are emitted.
        assert!(
            !r.nodes.iter().any(|x| x.label == "Type"),
            "typedef must not emit a Type node: {:?}",
            r.nodes
        );
    }

    #[test]
    fn c_doc_comment_becomes_doc_property() {
        let r = c(C_SRC, "src/a.c");
        let add = r.nodes.iter().find(|n| n.name == "add").unwrap();
        assert_eq!(
            add.properties.get("doc").and_then(|v| v.as_str()),
            Some("add returns the sum."),
            "add doc summary missing: {:?}",
            add.properties
        );
    }

    #[test]
    fn c_calls_capture_bare_and_member_callee() {
        const SRC: &str = r#"
void f(struct T *p) {
    bare();
    p->run();
    p.go();
}
"#;
        let r = c(SRC, "src/a.c");
        let callees: std::collections::HashSet<String> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        assert!(callees.contains("bare"), "missing bare callee: {callees:?}");
        assert!(
            callees.contains("run"),
            "missing arrow member callee run: {callees:?}"
        );
        assert!(
            callees.contains("go"),
            "missing dot member callee go: {callees:?}"
        );
    }

    #[test]
    fn c_cross_file_call_resolves_by_callee_name() {
        const A: &str = r#"
int caller(void) { return helper(1); }
"#;
        const B: &str = r#"
int helper(int n) { return n + 1; }
"#;
        let a = c(A, "a.c");
        let b = c(B, "b.c");
        let call_edge = a
            .edges
            .iter()
            .find(|e| {
                e.edge_type == "CALLS"
                    && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("helper")
            })
            .expect("a.c must emit CALLS edge for helper");
        assert_eq!(
            call_edge.source_qualified_name, "a.c::Function::caller",
            "CALLS source must be the enclosing caller: {call_edge:?}"
        );
        assert!(
            b.nodes
                .iter()
                .any(|n| n.name == "helper" && n.label == "Function"),
            "b.c must define Function helper: {:?}",
            b.nodes
        );
    }

    #[test]
    fn c_includes_emit_import_edges_with_basename() {
        let r = c(C_SRC, "src/a.c");
        let imported: std::collections::HashMap<&str, &str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .filter_map(|e| {
                let name = e.properties.get("imported_name").and_then(|v| v.as_str())?;
                let path = e.properties.get("path").and_then(|v| v.as_str())?;
                Some((name, path))
            })
            .collect();
        // `<stdio.h>` -> binding stdio.h.
        assert_eq!(
            imported.get("stdio.h"),
            Some(&"stdio.h"),
            "stdio include: {imported:?}"
        );
        // `"sub/helper.h"` -> binding basename helper.h, path sub/helper.h.
        assert_eq!(
            imported.get("helper.h"),
            Some(&"sub/helper.h"),
            "helper include basename: {imported:?}"
        );
    }

    // -----------------------------------------------------------------------
    // C++
    // -----------------------------------------------------------------------

    const CPP_SRC: &str = r#"
#include <vector>
#include "helper.h"
using std::vector;

namespace geo {

// A shape.
class Shape {
public:
    double area();
    void scale(double f) { helper(f); }
};

double Shape::area() {
    obj.run();
    return 0.0;
}

}

int add(int a, int b) { return a + b; }
"#;

    fn cpp(src: &str, path: &str) -> crate::extract::ExtractionResult {
        extract(Language::Cpp, src.as_bytes(), path).unwrap()
    }

    #[test]
    fn cpp_extract_returns_ok_for_cpp_and_hpp() {
        for path in ["a.cpp", "a.hpp", "a.cc", "a.cxx", "a.hh"] {
            let lang = crate::language::language_for_path(std::path::Path::new(path));
            assert_eq!(lang, Language::Cpp, "for {path}");
            let r = extract(lang, CPP_SRC.as_bytes(), path);
            assert!(r.is_ok(), "extract must be Ok for {path}: {r:?}");
        }
    }

    #[test]
    fn cpp_finds_class_namespace_struct_and_function() {
        let r = cpp(CPP_SRC, "src/s.cpp");
        let by = |n: &str, label: &str| r.nodes.iter().any(|x| x.name == n && x.label == label);
        assert!(by("Shape", "Class"), "missing class Shape: {:?}", r.nodes);
        // A `namespace_definition` is NOT a graph node
        // (it is folded into the module spine), so no Namespace node
        // is emitted.
        assert!(
            !r.nodes.iter().any(|x| x.label == "Namespace"),
            "namespace must not emit a Namespace node: {:?}",
            r.nodes
        );
        assert!(by("add", "Function"), "missing func add: {:?}", r.nodes);
        assert!(by("scale", "Method"), "missing method scale: {:?}", r.nodes);
    }

    #[test]
    fn cpp_inline_and_out_of_line_methods_owned_by_class() {
        let r = cpp(CPP_SRC, "src/s.cpp");
        // inline method `scale` defined inside the class body.
        let scale = r
            .nodes
            .iter()
            .find(|n| n.name == "scale" && n.label == "Method")
            .unwrap();
        assert_eq!(scale.qualified_name, "src/s.cpp::Shape::scale");
        // out-of-line definition `double Shape::area()` owned by Shape.
        let area = r
            .nodes
            .iter()
            .find(|n| n.name == "area" && n.label == "Method")
            .expect("missing out-of-line method area");
        assert_eq!(area.qualified_name, "src/s.cpp::Shape::area");
    }

    #[test]
    fn cpp_method_qnames_do_not_collide_across_classes() {
        const SRC: &str = r#"
class Foo { public: void run() {} };
class Bar { public: void run() {} };
"#;
        let r = cpp(SRC, "src/a.cpp");
        let qnames: std::collections::HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.name == "run" && n.label == "Method")
            .map(|n| n.qualified_name.as_str())
            .collect();
        assert!(
            qnames.contains("src/a.cpp::Foo::run") && qnames.contains("src/a.cpp::Bar::run"),
            "expected distinct Foo::run and Bar::run qnames, got {qnames:?}"
        );
    }

    #[test]
    fn cpp_doc_comment_becomes_doc_property() {
        let r = cpp(CPP_SRC, "src/s.cpp");
        let shape = r
            .nodes
            .iter()
            .find(|n| n.name == "Shape" && n.label == "Class")
            .unwrap();
        assert_eq!(
            shape.properties.get("doc").and_then(|v| v.as_str()),
            Some("A shape."),
            "Shape doc summary missing: {:?}",
            shape.properties
        );
    }

    #[test]
    fn cpp_calls_capture_bare_member_and_qualified_callee() {
        const SRC: &str = r#"
void f() {
    bare();
    obj.doIt();
    ptr->run();
    geo::helper();
}
"#;
        let r = cpp(SRC, "src/a.cpp");
        let callees: std::collections::HashSet<String> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| {
                e.properties
                    .get("callee_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        for want in ["bare", "doIt", "run", "helper"] {
            assert!(callees.contains(want), "missing callee {want}: {callees:?}");
        }
    }

    #[test]
    fn cpp_cross_file_call_resolves_by_callee_name() {
        const A: &str = r#"
int caller() { return helper(1); }
"#;
        const B: &str = r#"
int helper(int n) { return n + 1; }
"#;
        let a = cpp(A, "a.cpp");
        let b = cpp(B, "b.cpp");
        let call_edge = a
            .edges
            .iter()
            .find(|e| {
                e.edge_type == "CALLS"
                    && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("helper")
            })
            .expect("a.cpp must emit CALLS edge for helper");
        assert_eq!(
            call_edge.source_qualified_name, "a.cpp::Function::caller",
            "CALLS source must be the enclosing caller: {call_edge:?}"
        );
        assert!(
            b.nodes
                .iter()
                .any(|n| n.name == "helper" && n.label == "Function"),
            "b.cpp must define Function helper: {:?}",
            b.nodes
        );
    }

    #[test]
    fn cpp_parameter_type_emits_type_ref_from_enclosing_function() {
        let r = cpp(
            r#"
namespace app {
int render(::Widget w) { return w.w; }
}
"#,
            "src/main.cpp",
        );
        let edge = r.edges.iter().find(|edge| {
            edge.edge_type == "TYPE_REF"
                && edge.properties.get("type_name").and_then(|v| v.as_str()) == Some("Widget")
        });
        let edge = edge.unwrap_or_else(|| panic!("missing Widget TYPE_REF: {:?}", r.edges));
        assert_eq!(edge.source_qualified_name, "src/main.cpp::Function::render");
    }

    #[test]
    fn cpp_includes_and_using_emit_import_edges() {
        let r = cpp(CPP_SRC, "src/s.cpp");
        // Collect (path, imported_name) pairs so the `<vector>` include and the
        // `using std::vector` (which share the binding `vector`) are both
        // visible — keyed by their distinct paths.
        let pairs: std::collections::HashSet<(&str, &str)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .filter_map(|e| {
                let name = e.properties.get("imported_name").and_then(|v| v.as_str())?;
                let path = e.properties.get("path").and_then(|v| v.as_str())?;
                Some((path, name))
            })
            .collect();
        // `#include <vector>` -> path vector, basename binding vector.
        assert!(
            pairs.contains(&("vector", "vector")),
            "vector include missing: {pairs:?}"
        );
        // `#include "helper.h"` -> path helper.h, basename binding helper.h.
        assert!(
            pairs.contains(&("helper.h", "helper.h")),
            "helper include missing: {pairs:?}"
        );
        // `using std::vector;` -> path std::vector, binding vector.
        assert!(
            pairs.contains(&("std::vector", "vector")),
            "using std::vector missing: {pairs:?}"
        );
    }

    // =======================================================================
    // C# / PHP / Bash — onboarded purely through the data-driven spec path.
    // These prove the LangSpec data path produces Definitions / Calls /
    // Imports + docstrings at parity with the other non-Rust languages.
    // =======================================================================

    fn cs(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::CSharp, src.as_bytes(), file).unwrap()
    }
    fn php(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Php, src.as_bytes(), file).unwrap()
    }
    fn bash(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Bash, src.as_bytes(), file).unwrap()
    }

    fn calls_edges(r: &crate::ExtractionResult) -> Vec<(String, String)> {
        r.edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.properties
                        .get("callee_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect()
    }

    fn import_pairs(r: &crate::ExtractionResult) -> Vec<(String, String)> {
        r.edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .map(|e| {
                let p = &e.properties;
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                (s("path"), s("imported_name"))
            })
            .collect()
    }

    #[test]
    fn extract_ok_for_cs_php_sh() {
        // The task's acceptance check: extract() returns Ok for .cs/.php/.sh.
        assert!(extract(Language::CSharp, b"class A {}", "A.cs").is_ok());
        assert!(extract(Language::Php, b"<?php class A {}", "A.php").is_ok());
        assert!(extract(Language::Bash, b"f() { :; }", "a.sh").is_ok());
    }

    #[test]
    fn csharp_defs_methods_owned_by_class() {
        const SRC: &str = r#"
using System;

namespace App {
    /// <summary>A widget.</summary>
    public class Widget {
        public Widget() { Setup(); }
        public int Compute(int x) { return Helper.Run(x); }
        private void Setup() {}
    }
    public interface IShape { double Area(); }
}
"#;
        let r = cs(SRC, "app/Widget.cs");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let c = by("Class", "Widget").expect("class node");
        assert_eq!(c.qualified_name, "app/Widget.cs::Class::Widget");
        // The class docstring (`///`) is attached.
        assert_eq!(
            c.properties.get("doc").and_then(|v| v.as_str()),
            Some("<summary>A widget.</summary>")
        );
        // Methods + constructor are owned by their class.
        let m = by("Method", "Compute").expect("method node");
        assert_eq!(m.qualified_name, "app/Widget.cs::Widget::Compute");
        let ctor = by("Method", "Widget").expect("constructor node");
        assert_eq!(ctor.qualified_name, "app/Widget.cs::Widget::Widget");
        let iface = by("Interface", "IShape").expect("interface node");
        assert_eq!(iface.qualified_name, "app/Widget.cs::Interface::IShape");
    }

    #[test]
    fn csharp_calls_capture_final_callee_and_caller() {
        const SRC: &str = r#"
class C {
    void Caller() {
        Bare();
        Helper.Run();
    }
}
"#;
        let r = cs(SRC, "C.cs");
        let edges = calls_edges(&r);
        assert!(
            edges.contains(&("C.cs::C::Caller".into(), "Bare".into())),
            "bare call: {edges:?}"
        );
        // Member call `Helper.Run()` captures the final `Run`.
        assert!(
            edges.contains(&("C.cs::C::Caller".into(), "Run".into())),
            "member call must capture final `Run`: {edges:?}"
        );
        assert!(
            !edges.iter().any(|(_, c)| c == "Helper"),
            "receiver `Helper` must not be a callee: {edges:?}"
        );
    }

    #[test]
    fn csharp_using_imports() {
        const SRC: &str = r#"
using System;
using System.Collections.Generic;
using IO = System.IO;
"#;
        let r = cs(SRC, "u.cs");
        let pairs = import_pairs(&r);
        assert!(
            pairs.contains(&("System".into(), "System".into())),
            "{pairs:?}"
        );
        assert!(
            pairs.contains(&("System.Collections.Generic".into(), "Generic".into())),
            "qualified using: {pairs:?}"
        );
        // Aliased `using IO = System.IO;` binds `IO`.
        assert!(
            pairs.contains(&("System.IO".into(), "IO".into())),
            "aliased using must bind alias: {pairs:?}"
        );
    }

    #[test]
    fn csharp_cross_file_call_resolves_by_callee_name() {
        // A call to `Run` in one file keys on the callee name so the indexer's
        // name-based resolver links it to a `Run` defined in another file.
        let a = cs("class A { void f() { Run(); } }", "a.cs");
        let b = cs("class B { public void Run() {} }", "b.cs");
        let callee = a
            .edges
            .iter()
            .find(|e| e.edge_type == "CALLS")
            .and_then(|e| e.properties.get("callee_name").and_then(|v| v.as_str()))
            .expect("a.cs must emit a CALLS edge");
        assert_eq!(callee, "Run");
        assert!(
            b.nodes
                .iter()
                .any(|n| n.label == "Method" && n.name == "Run"),
            "b.cs must define method Run for the resolver to match"
        );
    }

    #[test]
    fn csharp_struct_and_record_labelled_class() {
        // Every C#
        // `struct_declaration` / `record_declaration` is labelled "Class"; no
        // `Struct` or `Record` node survives, and the qname's label segment is
        // rewritten.
        let r = cs(
            "struct P { public int X; } record D { public int Y { get; init; } }",
            "m.cs",
        );
        assert!(
            !r.nodes
                .iter()
                .any(|n| n.label == "Struct" || n.label == "Record"),
            "no Struct/Record labels: {:?}",
            r.nodes
                .iter()
                .map(|n| (&n.label, &n.name))
                .collect::<Vec<_>>()
        );
        let p = r
            .nodes
            .iter()
            .find(|n| n.name == "P")
            .expect("struct P node");
        assert_eq!(p.label, "Class");
        assert_eq!(p.qualified_name, "m.cs::Class::P");
        let d = r
            .nodes
            .iter()
            .find(|n| n.name == "D")
            .expect("record D node");
        assert_eq!(d.label, "Class");
        assert_eq!(d.qualified_name, "m.cs::Class::D");
    }

    #[test]
    fn csharp_fields_variables_and_enum_members() {
        // The member model:
        //   * a `field_declaration` → one Field (owned by the type) + one
        //     module-scoped Variable;
        //   * a `property_declaration` → one Field only (properties are not in
        //     `cs_var_types`, so no Variable);
        //   * an `enum_member_declaration` → one Variable (owned by the enum).
        const SRC: &str = r#"
class C {
    public int Score { get; set; }
    private string _label;
}
enum E { A, B }
"#;
        let r = cs(SRC, "c.cs");
        let node =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        // Property → Field only (no Variable named Score).
        let score = node("Field", "Score").expect("property Field");
        assert_eq!(score.qualified_name, "c.cs::C::Score");
        assert!(
            node("Variable", "Score").is_none(),
            "no Variable for a property"
        );
        // Field → Field + Variable.
        let label_field = node("Field", "_label").expect("field Field");
        assert_eq!(label_field.qualified_name, "c.cs::C::_label");
        let label_var = node("Variable", "_label").expect("field Variable");
        assert_eq!(label_var.qualified_name, "c.cs::Variable::_label");
        // Enum members → Variable owned by the enum.
        let a = node("Variable", "A").expect("enum member A");
        assert_eq!(a.qualified_name, "c.cs::Enum::E::A");
        assert!(node("Variable", "B").is_some(), "enum member B");
    }

    #[test]
    fn csharp_defines_method_edges_owned_by_type() {
        // Every owned method / constructor gets a DEFINES_METHOD edge from its
        // enclosing type node to the method node.
        const SRC: &str = r#"
class Svc {
    public Svc() {}
    public int Work(int x) { return x; }
}
"#;
        let r = cs(SRC, "s.cs");
        let dm: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "DEFINES_METHOD")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                )
            })
            .collect();
        assert!(
            dm.contains(&("s.cs::Class::Svc".into(), "s.cs::Svc::Work".into())),
            "method DEFINES_METHOD: {dm:?}"
        );
        assert!(
            dm.contains(&("s.cs::Class::Svc".into(), "s.cs::Svc::Svc".into())),
            "constructor DEFINES_METHOD: {dm:?}"
        );
    }

    #[test]
    fn csharp_classifies_type_refs_and_qualified_enum_uses() {
        const SRC: &str = r#"
using Payload = Fixture.Types.Payload;
using HelperCode = Fixture.Helpers.HelperCode;
namespace Fixture.App {
    class Flow {
        Payload caller(Payload input) {
            int seed = (int)HelperCode.Seed;
            return input;
        }
    }
}
"#;
        let r = cs(SRC, "Main.cs");
        let refs: Vec<(&str, &str, &str)> = r
            .edges
            .iter()
            .filter_map(|edge| {
                let name = edge
                    .properties
                    .get(if edge.edge_type == "TYPE_REF" {
                        "type_name"
                    } else {
                        "ref_name"
                    })?
                    .as_str()?;
                Some((
                    edge.edge_type.as_str(),
                    edge.source_qualified_name.as_str(),
                    name,
                ))
            })
            .collect();
        assert!(
            refs.contains(&("TYPE_REF", "Main.cs::Flow::caller", "Payload")),
            "return/parameter Payload must be a TYPE_REF from caller: {refs:?}"
        );
        assert!(
            refs.contains(&("USES", "Main.cs::Flow::caller", "Seed")),
            "HelperCode.Seed must classify the enum member read as USES: {refs:?}"
        );
        assert!(
            !refs
                .iter()
                .any(|(_, source, _)| *source == "Main.cs::__file__"),
            "using aliases must not leak into the reference pass: {refs:?}"
        );
    }

    #[test]
    fn csharp_object_creation_is_a_call() {
        // `new Foo()` is a CALLS edge whose callee is the constructed type name,
        // keyed so the resolver links it to
        // the type's constructor `Method`. A generic `new List<int>()` keys on
        // the bare `List`.
        const SRC: &str = r#"
class C {
    void M() {
        var a = new Foo();
        var b = new Bar<int>();
    }
}
"#;
        let r = cs(SRC, "c.cs");
        let callees: Vec<&str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .filter_map(|e| e.properties.get("callee_name").and_then(|v| v.as_str()))
            .collect();
        assert!(callees.contains(&"Foo"), "bare ctor: {callees:?}");
        assert!(
            callees.contains(&"Bar"),
            "generic ctor keys on base name: {callees:?}"
        );
    }

    #[test]
    fn php_defs_methods_owned_by_class() {
        const SRC: &str = r#"<?php
namespace App;

/** A user. */
class User {
    public function __construct() { $this->setup(); }
    public function greet() { return Helper::run(); }
    private function setup() {}
}
function topLevel() { greet(); }
interface Shape { public function area(); }
trait T { public function shared() {} }
"#;
        let r = php(SRC, "User.php");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let c = by("Class", "User").expect("class node");
        assert_eq!(c.qualified_name, "User.php::Class::User");
        assert_eq!(
            c.properties.get("doc").and_then(|v| v.as_str()),
            Some("A user.")
        );
        let m = by("Method", "greet").expect("method node");
        assert_eq!(m.qualified_name, "User.php::User::greet");
        let f = by("Function", "topLevel").expect("free function");
        assert_eq!(f.qualified_name, "User.php::Function::topLevel");
        assert!(by("Interface", "Shape").is_some(), "interface def");
        // A PHP `trait_declaration` is labelled "Class"
        // (only Rust `trait_item` / `trait_definition` map to "Interface"), so a
        // PHP trait is a "Class" node with a `::Class::` qname, never a "Trait".
        let t = by("Class", "T").expect("trait def is labelled Class");
        assert_eq!(t.qualified_name, "User.php::Class::T");
        assert!(
            by("Trait", "T").is_none(),
            "a PHP trait must not keep the Trait label"
        );
        // The trait's own method gets a DEFINES_METHOD edge from the (Class) trait
        // node to the owned method, exactly like a class.
        assert!(
            r.edges.iter().any(|e| e.edge_type == "DEFINES_METHOD"
                && e.source_qualified_name == "User.php::Class::T"
                && e.target_qualified_name == "User.php::T::shared"),
            "trait method DEFINES_METHOD: {:?}",
            r.edges
        );
    }

    #[test]
    fn php_named_types_emit_type_refs_from_enclosing_callable() {
        const SRC: &str = r#"<?php
class Service {
    public function render(Widget $widget): Result|Error {
        return new Result();
    }
}
function load(): ?Vendor\\Payload { return null; }
"#;
        let r = php(SRC, "types.php");
        let refs: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter(|edge| edge.edge_type == "TYPE_REF")
            .filter_map(|edge| {
                Some((
                    edge.source_qualified_name.as_str(),
                    edge.properties.get("type_name")?.as_str()?,
                ))
            })
            .collect();
        assert!(
            refs.contains(&("types.php::Service::render", "Widget")),
            "parameter type: {refs:?}"
        );
        assert!(
            refs.contains(&("types.php::Service::render", "Result")),
            "union return type: {refs:?}"
        );
        assert!(
            refs.contains(&("types.php::Service::render", "Error")),
            "union return type: {refs:?}"
        );
        assert!(
            refs.contains(&("types.php::Function::load", "Payload")),
            "qualified nullable return type uses final name: {refs:?}"
        );
        assert_eq!(refs.len(), 4, "only declared named types: {refs:?}");
    }

    #[test]
    fn php_calls_capture_final_callee() {
        const SRC: &str = r#"<?php
class C {
    function caller() {
        bare();
        $this->member();
        Helper::staticCall();
    }
}
"#;
        let r = php(SRC, "c.php");
        let edges = calls_edges(&r);
        assert!(
            edges.contains(&("c.php::C::caller".into(), "bare".into())),
            "bare: {edges:?}"
        );
        assert!(
            edges.contains(&("c.php::C::caller".into(), "member".into())),
            "member call final name: {edges:?}"
        );
        assert!(
            edges.contains(&("c.php::C::caller".into(), "staticCall".into())),
            "static call final name: {edges:?}"
        );
    }

    #[test]
    fn php_enum_methods_double_as_functions_and_module_calls() {
        // A backed enum with a static factory + an instance method, and a
        // top-level call at module scope.
        const SRC: &str = r#"<?php
namespace App;

enum Severity: int {
    case Low = 1;
    case High = 3;

    public static function fromScore(int $s): Severity {
        if ($s >= 66) { return Severity::High; }
        return Severity::Low;
    }

    public function label(): string {
        return 'x';
    }
}

function boot(): void {}
boot();
"#;
        let r = php(SRC, "e.php");
        let node =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        // The enum itself is an Enum.
        assert!(node("Enum", "Severity").is_some(), "enum node");
        // Each enum method is BOTH a Method (owned by the enum) and a file-scoped
        // free Function — C double-emits these because a PHP `enum_declaration`
        // body is re-walked as top-level functions.
        for m in ["fromScore", "label"] {
            let meth = node("Method", m).unwrap_or_else(|| panic!("Method {m}"));
            assert_eq!(meth.qualified_name, format!("e.php::Severity::{m}"));
            let func = node("Function", m).unwrap_or_else(|| panic!("Function {m}"));
            assert_eq!(func.qualified_name, format!("e.php::Function::{m}"));
        }
        // DEFINES_METHOD from the enum type node to each enum method.
        for m in ["fromScore", "label"] {
            assert!(
                r.edges.iter().any(|e| e.edge_type == "DEFINES_METHOD"
                    && e.source_qualified_name == "e.php::Enum::Severity"
                    && e.target_qualified_name == format!("e.php::Severity::{m}")),
                "enum DEFINES_METHOD for {m}: {:?}",
                r.edges
            );
        }
        // The top-level `boot();` produces a module-scope CALLS edge from the
        // file Module node (the call-source file fallback).
        assert!(
            r.edges.iter().any(|e| e.edge_type == "CALLS"
                && e.source_qualified_name == "e.php::__file__"
                && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("boot")),
            "module-scope CALLS for boot(): {:?}",
            calls_edges(&r)
        );
    }

    #[test]
    fn php_class_methods_not_duplicated_as_functions() {
        // A regular class body is recognised by the body-container walk, so its
        // methods are NOT re-emitted as free Functions (only enum bodies are).
        const SRC: &str = r#"<?php
class C {
    public function m(): void {}
}
"#;
        let r = php(SRC, "c.php");
        assert!(
            r.nodes.iter().any(|n| n.label == "Method" && n.name == "m"),
            "class method m is a Method"
        );
        assert!(
            !r.nodes
                .iter()
                .any(|n| n.label == "Function" && n.name == "m"),
            "class method m must NOT also be a free Function"
        );
        // But it still gets a DEFINES_METHOD edge from its class.
        assert!(
            r.edges.iter().any(|e| e.edge_type == "DEFINES_METHOD"
                && e.source_qualified_name == "c.php::Class::C"
                && e.target_qualified_name == "c.php::C::m"),
            "class DEFINES_METHOD: {:?}",
            r.edges
        );
    }

    #[test]
    fn php_use_imports_collapse_per_namespace() {
        // Every clause here imports from the SAME namespace `App\Lib`. C resolves
        // each `use App\Lib\X` through the namespace map to the first file
        // declaring `App\Lib` and dedups identical (source-file, target) edges, so
        // all of them collapse to ONE IMPORTS edge. `collapse_php_imports`
        // reproduces that per-(file, namespace) granularity, keeping only the
        // first clause (`Helper`).
        const SRC: &str = r#"<?php
use App\Lib\Helper;
use App\Lib\{Foo, Bar as B};
"#;
        let r = php(SRC, "i.php");
        let pairs = import_pairs(&r);
        assert_eq!(
            pairs.len(),
            1,
            "same-namespace uses collapse to one edge: {pairs:?}"
        );
        assert!(
            pairs.contains(&("App\\Lib\\Helper".into(), "Helper".into())),
            "the surviving edge is the first clause: {pairs:?}"
        );
    }

    #[test]
    fn php_use_imports_distinct_namespaces_kept() {
        // Imports from DIFFERENT namespaces resolve to different targets in C and
        // are NOT collapsed — one IMPORTS edge each.
        const SRC: &str = r#"<?php
use App\Core\Helper;
use App\Service\Runner;
"#;
        let r = php(SRC, "i.php");
        let pairs = import_pairs(&r);
        assert_eq!(
            pairs.len(),
            2,
            "distinct namespaces stay separate: {pairs:?}"
        );
        assert!(pairs.contains(&("App\\Core\\Helper".into(), "Helper".into())));
        assert!(pairs.contains(&("App\\Service\\Runner".into(), "Runner".into())));
    }

    #[test]
    fn php_cross_file_call_resolves_by_callee_name() {
        let a = php("<?php class A { function f() { run(); } }", "a.php");
        let b = php("<?php function run() {} ", "b.php");
        let callee = a
            .edges
            .iter()
            .find(|e| e.edge_type == "CALLS")
            .and_then(|e| e.properties.get("callee_name").and_then(|v| v.as_str()))
            .expect("a.php must emit a CALLS edge");
        assert_eq!(callee, "run");
        assert!(
            b.nodes.iter().any(|n| n.name == "run"),
            "b.php must define run"
        );
    }

    #[test]
    fn bash_defs_calls_and_source_imports() {
        const SRC: &str = r#"#!/bin/bash
source ./lib.sh
. ./other.sh

# Greet the user
greet() {
    helper arg
}

function build {
    greet
}
"#;
        let r = bash(SRC, "build.sh");
        let by = |name: &str| {
            r.nodes
                .iter()
                .find(|n| n.label == "Function" && n.name == name)
        };
        let g = by("greet").expect("greet function");
        assert_eq!(g.qualified_name, "build.sh::Function::greet");
        // Leading `#` comment becomes the docstring.
        assert_eq!(
            g.properties.get("doc").and_then(|v| v.as_str()),
            Some("Greet the user")
        );
        assert!(by("build").is_some(), "build function (function kw form)");

        // Calls: `greet` inside build; `helper` inside greet. `source`/`.` are
        // imports, not calls.
        let edges = calls_edges(&r);
        assert!(
            edges.contains(&("build.sh::Function::build".into(), "greet".into())),
            "greet call: {edges:?}"
        );
        assert!(
            edges.contains(&("build.sh::Function::greet".into(), "helper".into())),
            "helper call: {edges:?}"
        );
        assert!(
            !edges.iter().any(|(_, c)| c == "source" || c == "."),
            "source/. must not be calls: {edges:?}"
        );

        // Imports: `source ./lib.sh` and `. ./other.sh`.
        let pairs = import_pairs(&r);
        assert!(
            pairs.contains(&("./lib.sh".into(), "lib.sh".into())),
            "source import: {pairs:?}"
        );
        assert!(
            pairs.contains(&("./other.sh".into(), "other.sh".into())),
            "dot import: {pairs:?}"
        );
    }

    #[test]
    fn bash_cross_file_call_resolves_by_callee_name() {
        let a = bash("caller() { shared; }", "a.sh");
        let b = bash("shared() { echo hi; }", "b.sh");
        let callee = a
            .edges
            .iter()
            .find(|e| e.edge_type == "CALLS")
            .and_then(|e| e.properties.get("callee_name").and_then(|v| v.as_str()))
            .expect("a.sh must emit a CALLS edge");
        assert_eq!(callee, "shared");
        assert!(
            b.nodes.iter().any(|n| n.name == "shared"),
            "b.sh must define shared"
        );
    }

    // =======================================================================
    // Batch-onboarded languages (Lua, Kotlin, Scala, Swift, Zig, R) — each
    // added purely through the LangSpec data path. For every language:
    //   * extract() returns Ok for the extension,
    //   * definitions are found,
    //   * a CROSS-FILE call (caller in one file, callee defined in another)
    //     resolves by `callee_name`.
    // =======================================================================

    fn lua(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Lua, src.as_bytes(), file).unwrap()
    }
    fn kotlin(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Kotlin, src.as_bytes(), file).unwrap()
    }
    fn scala(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Scala, src.as_bytes(), file).unwrap()
    }
    fn swift(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Swift, src.as_bytes(), file).unwrap()
    }
    fn dart(src: &str, file: &str) -> crate::ExtractionResult {
        // Dart is a registry language; resolve the `Registered` variant by path
        // so the `extract_dart` bespoke pass runs.
        let lang = crate::language::language_for_path(std::path::Path::new(file));
        extract(lang, src.as_bytes(), file).unwrap()
    }
    fn zig(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::Zig, src.as_bytes(), file).unwrap()
    }
    fn rlang(src: &str, file: &str) -> crate::ExtractionResult {
        extract(Language::R, src.as_bytes(), file).unwrap()
    }
    fn ocaml(src: &str, file: &str) -> crate::ExtractionResult {
        // OCaml is a registry language; resolve its `LangDef` by `.ml` path so
        // the `Language::Registered` dispatch reaches `extract_ocaml`.
        let d = crate::registry::LangDef::for_path(std::path::Path::new(file))
            .expect("ocaml LangDef registered for .ml");
        extract(Language::Registered(d), src.as_bytes(), file).unwrap()
    }

    fn solidity(src: &str, file: &str) -> crate::ExtractionResult {
        // Solidity is a registry language; resolve its `LangDef` by `.sol`
        // path so the `Language::Registered` dispatch reaches `extract_solidity`.
        let d = crate::registry::LangDef::for_path(std::path::Path::new(file))
            .expect("solidity LangDef registered for .sol");
        extract(Language::Registered(d), src.as_bytes(), file).unwrap()
    }

    #[test]
    fn solidity_contract_library_struct_are_class_interface_is_interface() {
        let src = r#"
interface IThing { function go() external; }
library Lib { function help() internal pure returns (uint256) { return 1; } }
contract C is IThing {
    struct Rec { uint256 amount; }
    enum State { On, Off }
    uint256 public total;
    modifier guard() { _; }
    function go() public { help(); }
}
"#;
        let r = solidity(src, "a.sol");
        let has =
            |label: &str, name: &str| r.nodes.iter().any(|n| n.label == label && n.name == name);
        // contract / library / struct all → Class; interface → Interface.
        assert!(has("Interface", "IThing"));
        assert!(has("Class", "Lib"));
        assert!(has("Class", "C"));
        assert!(has("Class", "Rec")); // struct → Class
        assert!(has("Enum", "State"));
        // struct member → Field only (no Variable twin).
        assert!(has("Field", "amount"));
        assert!(!r
            .nodes
            .iter()
            .any(|n| n.label == "Variable" && n.name == "amount"));
        // contract state variable → Field + Variable twin.
        assert!(has("Field", "total"));
        assert!(has("Variable", "total"));
        // owned function/modifier → Method + Function twin + DEFINES_METHOD.
        assert!(has("Method", "go") && has("Function", "go"));
        assert!(has("Method", "guard") && has("Function", "guard"));
        assert!(r
            .edges
            .iter()
            .any(|e| e.edge_type == "DEFINES_METHOD" && e.target_qualified_name == "a.sol::C::go"));
    }

    #[test]
    fn solidity_free_function_is_function_only_and_calls_resolve_same_file() {
        let src = r#"
contract C {
    function a() public { b(); }
    function b() public {}
}
function freeHelper(uint256 x) pure returns (uint256) { return x; }
"#;
        let r = solidity(src, "a.sol");
        // Free (top-level) function → exactly one Function node, no Method twin.
        assert_eq!(r.nodes.iter().filter(|n| n.name == "freeHelper").count(), 1);
        assert!(r
            .nodes
            .iter()
            .any(|n| n.label == "Function" && n.name == "freeHelper"));
        assert!(!r
            .nodes
            .iter()
            .any(|n| n.label == "Method" && n.name == "freeHelper"));
        // Same-file CALLS: `a` calls `b`; source is the enclosing Method qname.
        assert!(r.edges.iter().any(|e| e.edge_type == "CALLS"
            && e.source_qualified_name == "a.sol::C::a"
            && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("b")));
    }

    #[test]
    fn solidity_enum_type_reference_is_not_a_usage() {
        let src = r#"
contract C {
    enum State { On, Off }
    State public s;
    function set() public { s = State.On; }
}
"#;
        let r = solidity(src, "a.sol");
        // `State` (enum type name) is never a USAGE ref — Solidity usages
        // are not resolved to Enum nodes.
        assert!(!r.edges.iter().any(|e| e.edge_type == "USAGE"
            && e.properties.get("ref_name").and_then(|v| v.as_str()) == Some("State")));
    }

    /// The `callee_name` of the first CALLS edge in `r`, for cross-file tests.
    fn first_callee(r: &crate::ExtractionResult) -> Option<String> {
        r.edges
            .iter()
            .find(|e| e.edge_type == "CALLS")
            .and_then(|e| e.properties.get("callee_name").and_then(|v| v.as_str()))
            .map(|s| s.to_string())
    }

    // ---- Objective-C (registry language, bespoke `extract_objc`) ----------

    fn objc(src: &str, file: &str) -> crate::ExtractionResult {
        // Objective-C is a registry language; resolve its `LangDef` by `.m` path
        // so the `Language::Registered` dispatch reaches `extract_objc`.
        let d = crate::registry::LangDef::for_path(std::path::Path::new(file))
            .expect("objc LangDef registered for .m");
        extract(Language::Registered(d), src.as_bytes(), file).unwrap()
    }

    #[test]
    fn objc_interface_impl_collapse_to_class_protocol_is_interface_methods_owned() {
        // `@interface`/`@implementation` for the same class collapse to ONE
        // Class node (via `UNIQUE(qualified_name)`); `@protocol` → Interface; only
        // `@implementation`-body `method_definition`s become Method nodes (+
        // DEFINES_METHOD); a free C `function_definition` emits NO node.
        let src = r#"
#import <Foundation/Foundation.h>

@protocol Drawable <NSObject>
- (double)area;
@end

@interface Shape : NSObject <Drawable>
@property (nonatomic, copy) NSString *name;
- (double)area;
@end

@implementation Shape
- (double)area { return 0.0; }
- (NSString *)describe { return self.name; }
@end

double helper(double v) { return v; }
"#;
        let r = objc(src, "Shape.m");
        let has =
            |label: &str, name: &str| r.nodes.iter().any(|n| n.label == label && n.name == name);
        // interface + implementation collapse to exactly one Class node.
        assert_eq!(
            r.nodes
                .iter()
                .filter(|n| n.label == "Class" && n.name == "Shape")
                .count(),
            1
        );
        assert!(has("Interface", "Drawable"));
        // impl-body methods → Method; the class-name identifier is NOT a def.
        assert!(has("Method", "area") && has("Method", "describe"));
        // free C function → NO node (C emits zero Function for objc).
        assert!(!r.nodes.iter().any(|n| n.name == "helper"));
        assert!(!r.nodes.iter().any(|n| n.label == "Function"));
        // no Field/Variable nodes for objc properties/ivars.
        assert!(!r
            .nodes
            .iter()
            .any(|n| n.label == "Field" || n.label == "Variable"));
        // DEFINES_METHOD from the Class node to each owned method.
        assert!(r.edges.iter().any(|e| e.edge_type == "DEFINES_METHOD"
            && e.source_qualified_name == "Shape.m::Class::Shape"
            && e.target_qualified_name == "Shape.m::Shape::area"));
    }

    #[test]
    fn objc_message_send_is_calls_and_references_are_usage() {
        let src = r#"
@implementation Shape
- (double)area { return 0.0; }
- (double)twice { return [self area] + [self area]; }
@end
"#;
        let r = objc(src, "Shape.m");
        // A `message_expression` selector → CALLS keyed on the selector name.
        assert!(r.edges.iter().any(|e| e.edge_type == "CALLS"
            && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("area")));
        // `#import` → IMPORTS is emitted for a directive.
        let src2 = r#"
#import "Shape.m"
@implementation Circle
- (double)area { return 1.0; }
@end
"#;
        let r2 = objc(src2, "Circle.m");
        assert!(r2.edges.iter().any(|e| e.edge_type == "IMPORTS"
            && e.properties.get("imported_name").and_then(|v| v.as_str()) == Some("Shape.m")));
        // A bare reference identifier (the same-file method name) → USAGE.
        assert!(r2.edges.iter().any(|e| e.edge_type == "USAGE"
            && e.properties.get("ref_name").and_then(|v| v.as_str()) == Some("area")));
    }

    #[test]
    fn extract_ok_for_batch_onboarded_extensions() {
        assert!(extract(Language::Lua, b"function f() end", "a.lua").is_ok());
        assert!(extract(Language::Kotlin, b"fun f() {}", "a.kt").is_ok());
        assert!(extract(
            Language::Scala,
            b"object O { def f(): Unit = {} }",
            "a.scala"
        )
        .is_ok());
        assert!(extract(Language::Swift, b"func f() {}", "a.swift").is_ok());
        assert!(extract(Language::Zig, b"fn f() void {}", "a.zig").is_ok());
        assert!(extract(Language::R, b"f <- function() { 1 }", "a.r").is_ok());
        let hs = crate::language::language_for_path(std::path::Path::new("A.hs"));
        assert!(extract(hs, b"module A where\nf x = x\n", "A.hs").is_ok());
    }

    // ---- Haskell (registry language, bespoke `extract_haskell`) -----------

    fn haskell(src: &str, path: &str) -> crate::ExtractionResult {
        let lang = crate::language::language_for_path(std::path::Path::new(path));
        extract(lang, src.as_bytes(), path).expect("haskell extract")
    }

    #[test]
    fn haskell_types_functions_and_classlabels() {
        // `data`/`newtype`/`class` are all "Class" (there is
        // no Interface/Enum/Type kind for Haskell); a `type` synonym is NOT
        // emitted (not in any type list); top-level equations and class-body
        // methods are free "Function"s, but `where`-bound locals are NOT
        // (the defs walk does not descend into a function body).
        const SRC: &str = r#"
module M (Shape(..), Named(..), area) where

type Radius = Double

data Shape = Circle Radius | Rect Double Double

newtype Wrapper = Wrapper Int

class Named a where
  name :: a -> String
  describe :: a -> String
  describe x = "n:" ++ name x

area :: Shape -> Double
area (Circle r) = pi * r * r

helper :: Int -> Int
helper n = doubled
  where
    doubled = n + n
"#;
        let r = haskell(SRC, "M.hs");
        let names_of = |label: &str| {
            let mut v: Vec<&str> = r
                .nodes
                .iter()
                .filter(|n| n.label == label)
                .map(|n| n.name.as_str())
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        // Class: data + newtype + class (NOT the `type Radius`).
        assert_eq!(names_of("Class"), vec!["Named", "Shape", "Wrapper"]);
        // Function: top-level `area`/`helper` + the class-body default method
        // `describe` (which has an equation). `name` is signature-ONLY (no
        // equation here — no instance), so it is NOT a Function; and the
        // `where`-bound `doubled` is NOT emitted (the defs walk never descends
        // into a function body).
        assert_eq!(names_of("Function"), vec!["area", "describe", "helper"]);
        // Class node qname uses the free `{file}::Class::{name}` scheme.
        assert!(r
            .nodes
            .iter()
            .any(|n| n.label == "Class" && n.qualified_name == "M.hs::Class::Shape"));
    }

    #[test]
    fn haskell_calls_infix_apply_and_constructor() {
        // `apply` callee = first child if variable/constructor; `infix` callee =
        // the operator; keyword-y callees are dropped. A call inside a top-level
        // body attributes to that def; a call inside a `where` attributes to the
        // module (so per-file duplicates collapse).
        const SRC: &str = r#"
module M (f, g, render) where

data Widget = Widget Int

f :: Int -> Int
f x = g (h x)

g :: Int -> Int
g y = h y
  where
    h z = z + z

render :: Widget -> Int
render (Widget n) = g n
"#;
        let r = haskell(SRC, "M.hs");
        let calls = calls_edges(&r);
        // `f x = g (h x)` — both `g` and `h` are apply callees, attributed to f.
        assert!(
            calls.contains(&("M.hs::Function::f".into(), "g".into())),
            "{calls:?}"
        );
        assert!(
            calls.contains(&("M.hs::Function::f".into(), "h".into())),
            "{calls:?}"
        );
        // The `where`-bound `h z = z + z` body call attributes to the MODULE,
        // not to any (unemitted) `where` def.
        assert!(
            !calls.iter().any(|(src, _)| src.contains("::Function::h")),
            "no call may be sourced from the unemitted where-def h: {calls:?}"
        );
        // `(Widget n)` is an `apply` in the grammar, but it is in the function's
        // LHS `patterns:` field: classify it as USAGE, never CALLS.
        assert!(
            !calls.iter().any(|(_, target)| target == "Widget"),
            "{calls:?}"
        );
        assert!(r.edges.iter().any(|edge| {
            edge.edge_type == "USAGE"
                && edge.source_qualified_name == "M.hs::Function::render"
                && edge.properties.get("ref_name").and_then(|v| v.as_str()) == Some("Widget")
        }));
    }

    #[test]
    fn haskell_usages_export_list_and_body_refs() {
        // Export-list names and non-call body references emit USAGE edges;
        // references inside a call or an import do not.
        const SRC: &str = r#"
module M (topA, topB) where

import Other (dep)

topA :: Int
topA = topB

topB :: Int
topB = 1
"#;
        let r = haskell(SRC, "M.hs");
        let usage_refs: Vec<&str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .filter_map(|e| e.properties.get("ref_name").and_then(|v| v.as_str()))
            .collect();
        // Export-list `topA`, `topB` and the body reference `topB` in topA.
        assert!(usage_refs.contains(&"topA"), "{usage_refs:?}");
        assert!(usage_refs.contains(&"topB"), "{usage_refs:?}");
        // The imported `dep` is inside an `import` — never a usage, but the
        // explicit import-list mapping emits an IMPORTS edge keyed by `dep`.
        assert!(!usage_refs.contains(&"dep"), "{usage_refs:?}");
        assert!(r.edges.iter().any(|edge| {
            edge.edge_type == "IMPORTS"
                && edge.source_qualified_name == "M.hs::__file__"
                && edge.properties.get("path").and_then(|v| v.as_str()) == Some("Other")
                && edge
                    .properties
                    .get("imported_name")
                    .and_then(|v| v.as_str())
                    == Some("dep")
        }));
        // Every USAGE source qname is a real node qname (a Function or the file
        // module), never a dangling `where`/`let`-bound name.
        for e in r.edges.iter().filter(|e| e.edge_type == "USAGE") {
            let src = &e.source_qualified_name;
            assert!(
                src == "M.hs::__file__" || src.starts_with("M.hs::Function::"),
                "unexpected usage source {src}"
            );
        }
    }

    // ---- Lua --------------------------------------------------------------

    #[test]
    fn lua_defs_calls_and_require_imports() {
        const SRC: &str = r#"
-- A greeter.
function greet(name)
    return helper(name)
end
local lib = require("mylib")
"#;
        let r = lua(SRC, "greet.lua");
        let greet = r
            .nodes
            .iter()
            .find(|n| n.label == "Function" && n.name == "greet")
            .expect("greet function");
        assert_eq!(greet.qualified_name, "greet.lua::Function::greet");
        // `--` docstring is attached.
        assert_eq!(
            greet.properties.get("doc").and_then(|v| v.as_str()),
            Some("A greeter.")
        );
        // Bare call `helper(...)` is captured; `require` is owned by imports.
        let edges = calls_edges(&r);
        assert!(
            edges.contains(&("greet.lua::Function::greet".into(), "helper".into())),
            "{edges:?}"
        );
        assert!(
            !edges.iter().any(|(_, c)| c == "require"),
            "require must not be a call: {edges:?}"
        );
        // `require("mylib")` import.
        assert!(
            import_pairs(&r).contains(&("mylib".into(), "mylib".into())),
            "{:?}",
            import_pairs(&r)
        );
    }

    #[test]
    fn lua_cross_file_call_resolves_by_callee_name() {
        let a = lua("function caller() return shared() end", "a.lua");
        let b = lua("function shared() return 1 end", "b.lua");
        assert_eq!(first_callee(&a).as_deref(), Some("shared"));
        assert!(
            b.nodes.iter().any(|n| n.name == "shared"),
            "b.lua must define shared"
        );
    }

    // ---- Dart -------------------------------------------------------------

    #[test]
    fn dart_types_members_enum_constants_and_usage() {
        // Exercises every pass the bespoke `extract_dart` adds on top of
        // the registry spec path: `class_declaration` → Class, `enum_declaration`
        // → Enum; enum constants → Variable owned by the enum; the mixin's members
        // are NOT extracted (mixins are not modelled); DEFINES_METHOD (class →
        // its methods); and the USAGE walk (direct-call callees kept,
        // `obj.method()` selectors skipped, same-file enum references suppressed
        // so no USAGE resolves to an Enum).
        const SRC: &str = r#"
enum Mode {
  fast,
  slow,
}

mixin Logging {
  String get channel;
  void logInfo(String message) {
    channel;
  }
}

class Token {
  final String lexeme;
  Token(this.lexeme);
  int widthOfToken() {
    return lexeme.length;
  }
}

class Runner with Logging {
  final Token token;
  Runner(this.token);
  @override
  String get channel => 'runner';
  int drive(Token other) {
    return other.widthOfToken();
  }
}

String labelOfMode(Mode mode) {
  switch (mode) {
    case Mode.fast:
      return 'f';
    case Mode.slow:
      return 's';
  }
}
"#;
        let r = dart(SRC, "app.dart");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let count = |label: &str, name: &str| {
            r.nodes
                .iter()
                .filter(|n| n.label == label && n.name == name)
                .count()
        };

        // class_declaration → Class; enum_declaration → Enum; the mixin is NOT a
        // Class node (mixins are not modelled).
        assert_eq!(
            by("Class", "Token").expect("class Token").qualified_name,
            "app.dart::Class::Token"
        );
        assert!(by("Class", "Runner").is_some(), "class is a Class");
        assert!(by("Class", "Logging").is_none(), "mixin is NOT a Class");
        assert_eq!(
            by("Enum", "Mode").expect("enum Mode").qualified_name,
            "app.dart::Enum::Mode"
        );

        // Enum constants → Variable owned by the enum (qname {file}::{Enum}::{c}).
        for c in ["fast", "slow"] {
            assert_eq!(
                by("Variable", c)
                    .unwrap_or_else(|| panic!("enum const {c}"))
                    .qualified_name,
                format!("app.dart::Mode::{c}")
            );
        }

        // Methods: class methods only. The mixin's `logInfo` is NOT a Method (or
        // a Function), and the getter `channel` is not extracted either.
        assert_eq!(count("Method", "widthOfToken"), 1);
        assert_eq!(count("Method", "drive"), 1);
        assert_eq!(count("Method", "logInfo"), 0, "mixin member not extracted");
        assert_eq!(
            count("Function", "logInfo"),
            0,
            "mixin member not a Function"
        );
        assert_eq!(count("Method", "channel"), 0, "getter not extracted");
        // Free function → Function (not Method).
        assert_eq!(
            by("Function", "labelOfMode")
                .expect("free fn")
                .qualified_name,
            "app.dart::Function::labelOfMode"
        );

        // DEFINES_METHOD: owner Class → each of its methods (never the mixin).
        let defm: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "DEFINES_METHOD")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                )
            })
            .collect();
        assert!(
            defm.contains(&(
                "app.dart::Class::Token".into(),
                "app.dart::Token::widthOfToken".into()
            )),
            "{defm:?}"
        );
        assert!(
            defm.contains(&(
                "app.dart::Class::Runner".into(),
                "app.dart::Runner::drive".into()
            )),
            "{defm:?}"
        );
        assert!(
            !defm.iter().any(|(s, _)| s.contains("Logging")),
            "no DEFINES_METHOD from the mixin: {defm:?}"
        );

        // USAGE. Collect (source, ref_name).
        let usages: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .filter_map(|e| {
                e.properties
                    .get("ref_name")
                    .and_then(|v| v.as_str())
                    .map(|rn| (e.source_qualified_name.as_str(), rn))
            })
            .collect();
        // A parameter-type reference to a Class is a USAGE (here `Token` in
        // `drive(Token other)`, attributed to the method via its signature).
        assert!(
            usages.contains(&("app.dart::Runner::drive", "Token")),
            "param-type USAGE of Token: {usages:?}"
        );
        // `obj.method()` selectors are NOT usages (the `.widthOfToken` in
        // `other.widthOfToken()` is a CALLS candidate, not a USAGE).
        assert!(
            !usages.iter().any(|(_, rn)| *rn == "widthOfToken"),
            "member-call selector must not emit USAGE: {usages:?}"
        );
        // A same-file enum reference never emits a USAGE (C emits none that
        // resolve to an Enum): neither the `Mode` param type nor the
        // `Mode.fast` pattern qualifiers.
        assert!(
            !usages.iter().any(|(_, rn)| *rn == "Mode"),
            "same-file enum reference must not emit USAGE: {usages:?}"
        );
        // Keywords never emit a USAGE.
        assert!(
            !usages
                .iter()
                .any(|(_, rn)| matches!(*rn, "this" | "return" | "switch" | "case")),
            "keywords must not emit USAGE: {usages:?}"
        );
    }

    #[test]
    fn dart_body_calls_usages_and_relative_imports_use_function_scope() {
        const SRC: &str = r#"
import 'helper.dart';

int caller() {
  final answer = do_it();
  return answer + HELPER_VALUE;
}
"#;
        let r = dart(SRC, "lib/main.dart");
        let call = r
            .edges
            .iter()
            .find(|edge| {
                edge.edge_type == "CALLS"
                    && edge
                        .properties
                        .get("callee_name")
                        .and_then(|value| value.as_str())
                        == Some("do_it")
            })
            .expect("do_it CALLS edge");
        assert_eq!(
            call.source_qualified_name,
            "lib/main.dart::Function::caller"
        );
        let usage = r
            .edges
            .iter()
            .find(|edge| {
                edge.edge_type == "USAGE"
                    && edge
                        .properties
                        .get("ref_name")
                        .and_then(|value| value.as_str())
                        == Some("HELPER_VALUE")
            })
            .expect("HELPER_VALUE usage");
        assert_eq!(
            usage.source_qualified_name,
            "lib/main.dart::Function::caller"
        );
        let import = r
            .edges
            .iter()
            .find(|edge| edge.edge_type == "IMPORTS")
            .expect("relative Dart import");
        assert_eq!(
            import
                .properties
                .get("path")
                .and_then(|value| value.as_str()),
            Some("helper.dart")
        );
        assert_eq!(
            import
                .properties
                .get("dart_relative_import")
                .and_then(|value| value.as_bool()),
            Some(true)
        );

        let helper = dart("const int HELPER_VALUE = 7;\n", "lib/helper.dart");
        assert!(
            helper
                .nodes
                .iter()
                .any(|node| node.label == "Variable" && node.name == "HELPER_VALUE"),
            "top-level constants must be resolvable usage targets: {:?}",
            helper.nodes
        );
    }

    // ---- Kotlin -----------------------------------------------------------

    #[test]
    fn kotlin_defs_methods_owned_by_class() {
        const SRC: &str = r#"
package app
import kotlin.math.max
/** A widget. */
class Widget {
    fun compute(x: Int): Int {
        return helper(x)
    }
}
fun freeFn() { freeOther() }
"#;
        let r = kotlin(SRC, "app/Widget.kt");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let c = by("Class", "Widget").expect("class node");
        assert_eq!(c.qualified_name, "app/Widget.kt::Class::Widget");
        assert_eq!(
            c.properties.get("doc").and_then(|v| v.as_str()),
            Some("A widget.")
        );
        // Method owned by its class.
        let m = by("Method", "compute").expect("method node");
        assert_eq!(m.qualified_name, "app/Widget.kt::Widget::compute");
        // Free function.
        let f = by("Function", "freeFn").expect("free function");
        assert_eq!(f.qualified_name, "app/Widget.kt::Function::freeFn");
        // Import final segment.
        assert!(
            import_pairs(&r).contains(&("kotlin.math.max".into(), "max".into())),
            "{:?}",
            import_pairs(&r)
        );
    }

    #[test]
    fn kotlin_cross_file_call_resolves_by_callee_name() {
        let a = kotlin("fun caller() { shared() }", "a.kt");
        let b = kotlin("fun shared() {}", "b.kt");
        assert_eq!(first_callee(&a).as_deref(), Some("shared"));
        assert!(b.nodes.iter().any(|n| n.name == "shared"));
    }

    #[test]
    fn kotlin_classifies_type_refs_and_constant_uses() {
        let r = kotlin(
            r#"
import grid.helper.HELPER_VALUE
import grid.types.Payload
fun caller(payload: Payload): Payload {
    val total = HELPER_VALUE
    return payload
}
"#,
            "src/main.kt",
        );
        let edge = |kind: &str, property: &str, name: &str| {
            r.edges.iter().any(|edge| {
                edge.edge_type == kind
                    && edge.source_qualified_name == "src/main.kt::Function::caller"
                    && edge.properties.get(property).and_then(|v| v.as_str()) == Some(name)
                    && edge
                        .properties
                        .get("preserve_reference_kind")
                        .and_then(|v| v.as_bool())
                        == Some(true)
            })
        };
        assert!(
            edge("TYPE_REF", "type_name", "Payload"),
            "Payload parameter/return type must classify as TYPE_REF: {:?}",
            r.edges
        );
        assert!(
            edge("USES", "ref_name", "HELPER_VALUE"),
            "HELPER_VALUE read must classify as USES: {:?}",
            r.edges
        );
        assert!(
            !r.edges.iter().any(|edge| edge.edge_type == "USAGE"),
            "Kotlin references must not collapse to generic USAGE at extraction: {:?}",
            r.edges
        );
    }

    // ---- Scala ------------------------------------------------------------

    #[test]
    fn scala_defs_methods_owned_by_type_and_imports() {
        const SRC: &str = r#"
package app
import scala.collection.mutable.Map
class Widget {
  val threshold: Int = 3
  def compute(x: Int): Int = helper(x)
  def caller(): Unit = { compute(2); Helper.run() }
}
object Helper { def run(): Unit = {} }
"#;
        let r = scala(SRC, "app/Widget.scala");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        assert_eq!(
            by("Class", "Widget").expect("class").qualified_name,
            "app/Widget.scala::Class::Widget"
        );
        assert_eq!(
            by("Method", "compute").expect("method").qualified_name,
            "app/Widget.scala::Widget::compute"
        );
        // An `object` is labelled "Class" (not "Object") and a
        // companion object is deduped into the same qname as its class.
        assert_eq!(
            by("Class", "Helper").expect("object").qualified_name,
            "app/Widget.scala::Class::Helper"
        );
        // A class-body `val` is a "Variable" (qname carries no owner segment).
        assert_eq!(
            by("Variable", "threshold").expect("val").qualified_name,
            "app/Widget.scala::Variable::threshold"
        );
        // Every method is ALSO double-counted as a free "Function" (the defs
        // walk re-walks the `template_body`), qname without the owner segment.
        assert_eq!(
            by("Function", "compute")
                .expect("free fn twin")
                .qualified_name,
            "app/Widget.scala::Function::compute"
        );
        // DEFINES_METHOD: the owning type → each Method it defines.
        let defmeth: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "DEFINES_METHOD")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                )
            })
            .collect();
        assert!(
            defmeth.contains(&(
                "app/Widget.scala::Class::Widget".into(),
                "app/Widget.scala::Widget::compute".into()
            )),
            "{defmeth:?}"
        );
        assert!(
            defmeth.contains(&(
                "app/Widget.scala::Class::Helper".into(),
                "app/Widget.scala::Helper::run".into()
            )),
            "{defmeth:?}"
        );
        // Member call `Helper.run()` captures the final `run`.
        let edges = calls_edges(&r);
        assert!(
            edges.contains(&("app/Widget.scala::Widget::caller".into(), "run".into())),
            "member call final segment: {edges:?}"
        );
        // Import final `path:` segment.
        assert!(
            import_pairs(&r).contains(&("scala.collection.mutable.Map".into(), "Map".into())),
            "{:?}",
            import_pairs(&r)
        );
    }

    #[test]
    fn scala_trait_is_interface_and_abstract_methods_owned() {
        // A trait → "Interface"; its abstract `def` (a `function_declaration`,
        // no body) is a "Method" owned by the trait, double-counted as a free
        // "Function", and the target of a DEFINES_METHOD edge — for both
        // concrete and abstract members.
        const SRC: &str = r#"
package app
trait Sink {
  def emit(level: String): Unit
}
"#;
        let r = scala(SRC, "app/Sink.scala");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        assert_eq!(
            by("Interface", "Sink").expect("trait").qualified_name,
            "app/Sink.scala::Interface::Sink"
        );
        assert_eq!(
            by("Method", "emit")
                .expect("abstract method")
                .qualified_name,
            "app/Sink.scala::Sink::emit"
        );
        assert!(
            by("Function", "emit").is_some(),
            "abstract method double-counted as free Function"
        );
        let defmeth = r.edges.iter().any(|e| {
            e.edge_type == "DEFINES_METHOD"
                && e.source_qualified_name == "app/Sink.scala::Interface::Sink"
                && e.target_qualified_name == "app/Sink.scala::Sink::emit"
        });
        assert!(defmeth, "trait defines its abstract method");
    }

    #[test]
    fn scala_cross_file_call_resolves_by_callee_name() {
        let a = scala("object A { def caller(): Unit = { shared() } }", "a.scala");
        let b = scala("object B { def shared(): Unit = {} }", "b.scala");
        assert_eq!(first_callee(&a).as_deref(), Some("shared"));
        assert!(b.nodes.iter().any(|n| n.name == "shared"));
    }

    // ---- Swift ------------------------------------------------------------

    #[test]
    fn swift_defs_methods_owned_by_type_and_imports() {
        const SRC: &str = r#"
import Foundation
/// A widget.
class Widget {
    func compute(x: Int) -> Int {
        return helper(x)
    }
}
func freeFn() { freeOther() }
"#;
        let r = swift(SRC, "Widget.swift");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let c = by("Class", "Widget").expect("class");
        assert_eq!(c.qualified_name, "Widget.swift::Class::Widget");
        assert_eq!(
            c.properties.get("doc").and_then(|v| v.as_str()),
            Some("A widget.")
        );
        assert_eq!(
            by("Method", "compute").expect("method").qualified_name,
            "Widget.swift::Widget::compute"
        );
        assert_eq!(
            by("Function", "freeFn").expect("free fn").qualified_name,
            "Widget.swift::Function::freeFn"
        );
        assert!(
            import_pairs(&r).contains(&("Foundation".into(), "Foundation".into())),
            "{:?}",
            import_pairs(&r)
        );
    }

    #[test]
    fn swift_cross_file_call_resolves_by_callee_name() {
        let a = swift("func caller() { shared() }", "a.swift");
        let b = swift("func shared() {}", "b.swift");
        assert_eq!(first_callee(&a).as_deref(), Some("shared"));
        assert!(b.nodes.iter().any(|n| n.name == "shared"));
    }

    #[test]
    fn ocaml_defs_calls_usages_match_c_model() {
        // Exercises the OCaml extractor in one file: a
        // param-less binding (`origin`), a function (`make_point`), a nested
        // module whose bindings ARE captured (`fib`), a local `let .. in`
        // binding that is NOT (`dx`), a same-file CALL (`square`), and a
        // non-call value_path USAGE (`origin`).
        const SRC: &str = r#"
open Helper

type widget = { value : int }

let render (w : Helper.widget) = w.value

let origin = 0

let square n = n * n

let alias = origin

let dist a =
  let dx = square a in
  dx

module Fib = struct
  let fib n = n
end
"#;
        let r = ocaml(SRC, "m.ml");
        let fns: Vec<&str> = r
            .nodes
            .iter()
            .filter(|n| n.label == "Function")
            .map(|n| n.name.as_str())
            .collect();
        // Every value_definition (top-level AND module-nested) is a Function;
        // the local `let dx = .. in` inside `dist`'s body is NOT.
        assert!(fns.contains(&"origin"), "param-less binding: {fns:?}");
        assert!(fns.contains(&"square"), "function: {fns:?}");
        assert!(fns.contains(&"dist"), "function: {fns:?}");
        assert!(fns.contains(&"fib"), "module-nested binding: {fns:?}");
        assert!(
            !fns.contains(&"dx"),
            "local let..in must NOT be a Function: {fns:?}"
        );
        // The source file becomes the importable OCaml compilation unit `M`;
        // the nested `module Fib` is still not emitted as a separate container.
        assert!(r.nodes.iter().any(|n| {
            n.label == "Class"
                && n.name == "M"
                && n.properties.get("kind").and_then(|v| v.as_str())
                    == Some("ocaml_compilation_unit")
        }));
        assert!(!r.nodes.iter().any(|n| n.name == "Fib"));
        assert!(r
            .nodes
            .iter()
            .any(|n| n.label == "Type" && n.name == "widget"));
        assert!(r.edges.iter().any(|edge| {
            edge.edge_type == "IMPORTS"
                && edge
                    .properties
                    .get("imported_name")
                    .and_then(|v| v.as_str())
                    == Some("Helper")
        }));
        assert!(r.edges.iter().any(|edge| {
            edge.edge_type == "TYPE_REF"
                && edge.source_qualified_name == "m.ml::Function::render"
                && edge.properties.get("type_name").and_then(|v| v.as_str()) == Some("widget")
        }));

        // CALLS source is the enclosing `dist`; `square` resolves same-file.
        let call = r
            .edges
            .iter()
            .find(|e| e.edge_type == "CALLS")
            .expect("a CALLS edge");
        assert_eq!(call.source_qualified_name, "m.ml::Function::dist");
        assert_eq!(
            call.properties.get("callee_name").and_then(|v| v.as_str()),
            Some("square")
        );

        // USAGE: `origin` referenced (not in a call) → one USAGE from the
        // enclosing `alias` binding, keyed by `ref_name`.
        let usage = r
            .edges
            .iter()
            .find(|e| {
                e.edge_type == "USAGE"
                    && e.properties.get("ref_name").and_then(|v| v.as_str()) == Some("origin")
            })
            .expect("a USAGE of origin");
        assert_eq!(usage.source_qualified_name, "m.ml::Function::alias");
    }

    #[test]
    fn swift_protocol_struct_enum_members_and_edges() {
        // Exercises every pass the bespoke `extract_swift` adds on top
        // of the uniform spec path: `protocol_declaration` → Interface;
        // `property_declaration` (top-level + type-body) → Variable; the enum
        // double-count (a `func` in an `enum_class_body` is BOTH a Method and a
        // free Function, as the defs walk re-walks the unrecognised body);
        // DEFINES_METHOD; IMPLEMENTS (same-file protocol conformance); and USAGE.
        const SRC: &str = r#"
import Foundation

let topLevelFlag = 1

protocol Greeter {
    var greeting: String { get }
    func makeGreeting() -> String
}

struct Badge {
    let title: String
    var count: Int

    func widthOfBadge() -> Int {
        return count
    }
}

enum Mode {
    case fast
    case slow

    func labelOfMode() -> String {
        return "m"
    }
}

class Printer: Greeter {
    var greeting: String
    let prefix: String

    func makeGreeting() -> String {
        return greeting
    }
}
"#;
        let r = swift(SRC, "kit.swift");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let count = |label: &str, name: &str| {
            r.nodes
                .iter()
                .filter(|n| n.label == label && n.name == name)
                .count()
        };

        // Protocol → Interface (NOT Class); struct / class / enum → Class.
        assert_eq!(
            by("Interface", "Greeter")
                .expect("interface")
                .qualified_name,
            "kit.swift::Interface::Greeter"
        );
        assert!(by("Class", "Badge").is_some(), "struct is a Class");
        assert!(by("Class", "Mode").is_some(), "enum is a Class");
        assert!(by("Class", "Printer").is_some(), "class is a Class");
        // Protocol requirements emit no Method / Function / Variable.
        assert_eq!(count("Method", "makeGreeting"), 1, "one real impl only");

        // Variables: top-level + every type-body property (protocol requirement
        // `greeting` on the protocol contributes none; the class's does).
        assert!(by("Variable", "topLevelFlag").is_some());
        for v in ["title", "count", "prefix"] {
            assert!(by("Variable", v).is_some(), "missing Variable {v}");
        }
        // `greeting` appears once — the class property (not the protocol req).
        assert_eq!(count("Variable", "greeting"), 1);

        // Enum method double-count: Method AND free Function.
        assert_eq!(count("Method", "labelOfMode"), 1, "enum method as Method");
        assert_eq!(
            count("Function", "labelOfMode"),
            1,
            "enum method ALSO as Function (C double-count)"
        );
        // A struct/class method is a Method only (its body is a recognised
        // `class_body`, so it is NOT re-walked as a Function).
        assert_eq!(count("Function", "widthOfBadge"), 0);
        assert_eq!(count("Method", "widthOfBadge"), 1);

        // DEFINES_METHOD: owner Class node → its Method.
        let defm: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "DEFINES_METHOD")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                )
            })
            .collect();
        assert!(
            defm.contains(&(
                "kit.swift::Class::Badge".into(),
                "kit.swift::Badge::widthOfBadge".into()
            )),
            "{defm:?}"
        );
        assert!(
            defm.contains(&(
                "kit.swift::Class::Mode".into(),
                "kit.swift::Mode::labelOfMode".into()
            )),
            "enum owner DEFINES_METHOD: {defm:?}"
        );

        // IMPLEMENTS: same-file protocol conformance resolves via the target
        // Interface qname.
        let imps: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPLEMENTS")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                )
            })
            .collect();
        assert_eq!(
            imps,
            vec![(
                "kit.swift::Class::Printer".into(),
                "kit.swift::Interface::Greeter".into()
            )],
            "one same-file IMPLEMENTS edge"
        );

        // USAGE: a non-call reference (here the `greeting` returned from
        // `makeGreeting`) emits a USAGE keyed on `ref_name`, sourced from the
        // enclosing method. It is NOT a definition name, call, or keyword.
        let usages: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .filter_map(|e| {
                e.properties
                    .get("ref_name")
                    .and_then(|v| v.as_str())
                    .map(|rn| (e.source_qualified_name.as_str(), rn))
            })
            .collect();
        assert!(
            usages.contains(&("kit.swift::Printer::makeGreeting", "greeting")),
            "method-body USAGE of `greeting`: {usages:?}"
        );
        // A keyword / def-name / call callee never emits a USAGE.
        assert!(
            !usages.iter().any(|(_, rn)| *rn == "self" || *rn == "func"),
            "keywords must not emit USAGE: {usages:?}"
        );
    }

    // ---- Elixir -----------------------------------------------------------

    fn elixir(src: &str, file: &str) -> crate::ExtractionResult {
        super::extract_elixir(src.as_bytes(), file).unwrap()
    }

    #[test]
    fn elixir_module_is_class_and_defs_are_functions() {
        // `defmodule` → Class;
        // `def`/`defp`/`defmacro` → Function; `defmacrop` is NOT extracted (it is
        // absent from the macro set); `alias`/`defstruct` emit no node.
        const SRC: &str = r#"
defmodule Shop.Cart do
  @moduledoc "cart"
  alias Shop.Product
  defstruct items: [], owner: nil

  def add_to_cart(cart, product) do
    Product.product_label(product)
  end

  defp sum_prices(items) do
    Enum.reduce(items, 0, fn e, acc -> acc + e.price end)
  end

  defmacro const(value) do
    quote do
      unquote(value)
    end
  end

  defmacrop internal_only(x) do
    x
  end
end
"#;
        let r = elixir(SRC, "lib/cart.ex");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let count = |label: &str| r.nodes.iter().filter(|n| n.label == label).count();

        // `defmodule Shop.Cart` → one Class named by the module alias.
        assert_eq!(
            by("Class", "Shop.Cart")
                .expect("module class")
                .qualified_name,
            "lib/cart.ex::Class::Shop.Cart"
        );
        assert_eq!(count("Class"), 1, "exactly one Class per defmodule");

        // def / defp / defmacro → Function; defmacrop is NOT extracted.
        for f in ["add_to_cart", "sum_prices", "const"] {
            assert_eq!(
                by("Function", f)
                    .unwrap_or_else(|| panic!("missing Function {f}"))
                    .qualified_name,
                format!("lib/cart.ex::Function::{f}")
            );
        }
        assert!(
            by("Function", "internal_only").is_none(),
            "defmacrop is not a C Function"
        );
        assert_eq!(count("Function"), 3, "def+defp+defmacro only");
        // `alias`/`defstruct`/`@moduledoc` produce no def node.
        assert_eq!(r.nodes.len(), 4, "1 Class + 3 Function, nothing else");
    }

    #[test]
    fn elixir_calls_sourced_from_file_module_by_bare_name() {
        // The CALLS source is the file Module (`<file>::__file__`,
        // Elixir defs are `call` nodes, never a func kind, so the enclosing-func
        // lookup falls back to the module). The callee is the bare name — the
        // trailing segment of a dotted `Mod.fun` call — so cross-module calls
        // resolve to the project Function by name. Keyword callees (`def`,
        // builtins) never become CALLS candidates.
        const SRC: &str = r#"
defmodule Shop.Cart do
  def add_to_cart(cart, product) do
    Product.product_label(product)
  end

  def total(cart) do
    sum_prices(cart.items)
  end
end
"#;
        let r = elixir(SRC, "lib/cart.ex");
        let callees: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.properties
                        .get("callee_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect();

        // Dotted call → bare trailing name; every CALLS is sourced from the file
        // Module node, never from the enclosing def.
        assert!(
            callees.contains(&("lib/cart.ex::__file__".into(), "product_label".into())),
            "dotted Product.product_label → bare `product_label` from file Module: {callees:?}"
        );
        assert!(
            callees.contains(&("lib/cart.ex::__file__".into(), "sum_prices".into())),
            "bare sum_prices call: {callees:?}"
        );
        // `def` (a keyword) is never a CALLS candidate — the def-header inner
        // call (`add_to_cart(cart, product)`) IS, and resolves to the
        // Function by name; but the outer `def` macro callee is keyword-filtered.
        assert!(
            !callees.iter().any(|(_, c)| c == "def"),
            "the `def` macro callee is keyword-filtered: {callees:?}"
        );
        // `defmodule` is NOT keyword-filtered here (it is not in the generic
        // keyword table), so it appears as a candidate — but the indexer's name
        // resolver drops it (no project Function named `defmodule`).
        assert!(
            callees.iter().any(|(_, c)| c == "add_to_cart"),
            "def-header self-call resolves to the Function: {callees:?}"
        );
        // Every CALLS edge is sourced from the file Module.
        assert!(
            r.edges
                .iter()
                .filter(|e| e.edge_type == "CALLS")
                .all(|e| e.source_qualified_name == "lib/cart.ex::__file__"),
            "all CALLS sourced from the file Module"
        );
    }

    // ---- Clojure ----------------------------------------------------------

    fn clojure(src: &str, file: &str) -> crate::ExtractionResult {
        let d = crate::registry::LangDef::for_path(std::path::Path::new("x.clj"))
            .expect("clojure registered");
        super::extract_clojure(d, src.as_bytes(), file).unwrap()
    }

    #[test]
    fn clojure_def_labels_calls_and_imports() {
        // Clojure def / callee / import handling:
        //   * `def` / `defn` → "Function"; `defrecord` / `deftype` → "Struct";
        //     `defprotocol` → "Interface";
        //   * every CALLS is sourced from the per-file Module; a BARE call to a
        //     same-file Function resolves, a NAMESPACE-qualified call
        //     (`u/square`) keeps its `ns/` prefix and does NOT resolve to a bare
        //     Function name (C splits only on `.`, never `/`);
        //   * a `(ns .. (:require [pkg :as p]) (:use other))` form emits one
        //     IMPORTS edge per module named in each dependency clause.
        const SRC: &str = r#"(ns app.core
  (:require [app.util :as u]
            [app.io :as io])
  (:use app.shared))

(def pi 3.14)

(defn add [a b]
  (+ a b))

(defn area [r]
  (add pi (u/square r)))

(defrecord Point [x y])

(deftype Box [w h])

(defprotocol Shape
  (area-of [this]))
"#;
        let r = clojure(SRC, "src/core.clj");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let count = |label: &str| r.nodes.iter().filter(|n| n.label == label).count();

        // Labels: def/defn → Function, defrecord/deftype → Struct,
        // defprotocol → Interface.
        for f in ["pi", "add", "area"] {
            assert_eq!(
                by("Function", f)
                    .unwrap_or_else(|| panic!("missing Function {f}"))
                    .qualified_name,
                format!("src/core.clj::Function::{f}")
            );
        }
        assert_eq!(count("Function"), 3, "def + two defn → 3 Functions");
        assert_eq!(
            by("Struct", "Point")
                .expect("defrecord → Struct")
                .qualified_name,
            "src/core.clj::Struct::Point"
        );
        assert!(by("Struct", "Box").is_some(), "deftype → Struct");
        assert_eq!(count("Struct"), 2, "defrecord + deftype → 2 Structs");
        assert_eq!(
            by("Interface", "Shape")
                .expect("defprotocol → Interface")
                .qualified_name,
            "src/core.clj::Interface::Shape"
        );
        assert_eq!(count("Interface"), 1);
        // The `area-of` sig inside the protocol body is NOT a def head → no node.
        assert!(by("Function", "area-of").is_none());

        // CALLS: all sourced from the file Module; the bare same-file call `add`
        // targets `add`; the qualified `u/square` keeps its prefix (unresolved).
        let calls: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "CALLS")
            .map(|e| {
                (
                    e.source_qualified_name.as_str(),
                    e.properties
                        .get("callee_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                )
            })
            .collect();
        assert!(
            calls
                .iter()
                .all(|(src, _)| *src == "src/core.clj::__file__"),
            "all CALLS sourced from the file Module: {calls:?}"
        );
        assert!(
            calls.iter().any(|(_, c)| *c == "add"),
            "bare same-file call `add`: {calls:?}"
        );
        assert!(
            calls.iter().any(|(_, c)| *c == "u/square"),
            "qualified call keeps its `ns/` prefix so it never matches a bare \
             Function name: {calls:?}"
        );
        // A def head (`defn`, `def`, `defrecord`, …) is never a call callee (the
        // DEFINITIONS pass owns those forms). Non-def heads like `ns` / `+` /
        // `area-of` MAY appear as raw callees exactly as the calls walk emits
        // them, but resolve to nothing (no project Function of that name).
        assert!(
            !calls.iter().any(|(_, c)| matches!(
                *c,
                "defn" | "def" | "defrecord" | "deftype" | "defprotocol"
            )),
            "def-form heads are not CALLS callees: {calls:?}"
        );

        // IMPORTS: one per dependency-clause module (`app.util`, `app.io` from
        // `:require`, `app.shared` from `:use`), all sourced from the Module.
        let imports: Vec<&str> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "IMPORTS")
            .map(|e| {
                e.properties
                    .get("module_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
            })
            .collect();
        assert_eq!(imports.len(), 3, "three imported modules: {imports:?}");
        for m in ["app.util", "app.io", "app.shared"] {
            assert!(imports.contains(&m), "missing import {m}: {imports:?}");
        }
    }

    // ---- Kotlin -----------------------------------------------------------

    #[test]
    fn kotlin_object_typealias_companion_variables_and_edges() {
        // Exercises every pass the bespoke `extract_kotlin` adds on top
        // of the uniform spec path:
        //   * `object_declaration` → "Class", NOT
        //     "Object" (so it resolves in the import/type/def label sets);
        //   * `type_alias` → "Type";
        //   * body / module-level `property_declaration` → "Variable" (a
        //     constructor-param `val`/`var` is NOT a property_declaration, so it
        //     is not a Variable);
        //   * a `fun` in a `companion object` is neither a Method nor a Function
        //     (the name-less companion is never descended into);
        //   * DEFINES_METHOD (owner Class node → its Method);
        //   * non-call value references emit USES and type positions emit TYPE_REF.
        const SRC: &str = r#"
package app

import app.util.Helper

const val MAX = 16

typealias Slot = Map<Int, String>

interface Store {
    fun put(key: String)
}

object Registry {
    const val SEED = 7

    fun make(): Int {
        return SEED
    }
}

class Cache(val capacity: Int) : Store {
    private val hits = HashMap<String, Int>()

    override fun put(key: String) {
        hits.put(key, capacity)
    }

    fun peek(key: String): Int {
        return MAX
    }

    companion object {
        const val MISS = -1

        fun empty(): Cache {
            return Cache(0)
        }
    }
}
"#;
        let r = kotlin(SRC, "app/App.kt");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);
        let count = |label: &str, name: &str| {
            r.nodes
                .iter()
                .filter(|n| n.label == label && n.name == name)
                .count()
        };

        // `object` → Class (NOT Object); interface / class → Class.
        assert_eq!(
            by("Class", "Registry")
                .expect("object is a Class")
                .qualified_name,
            "app/App.kt::Class::Registry"
        );
        assert!(by("Object", "Registry").is_none(), "no Object label");
        assert!(by("Class", "Store").is_some(), "interface is a Class");
        assert!(by("Class", "Cache").is_some(), "class is a Class");

        // `typealias` → Type.
        assert_eq!(
            by("Type", "Slot").expect("typealias").qualified_name,
            "app/App.kt::Type::Slot"
        );

        // Variables: module-level `const val`, an object's body `const val`, and
        // a class-body property. Constructor-param `val capacity` is NOT one.
        assert!(by("Variable", "MAX").is_some(), "module-level const val");
        assert!(by("Variable", "SEED").is_some(), "object body const val");
        assert!(by("Variable", "hits").is_some(), "class body property");
        assert!(
            by("Variable", "capacity").is_none(),
            "constructor-param val is not a property_declaration → no Variable"
        );
        // The companion object's `const val MISS` is inside a name-less
        // companion that is never descended into.
        assert!(by("Variable", "MISS").is_none(), "companion const skipped");

        // Methods owned by their type; the companion `fun empty` is neither a
        // Method nor a Function (the companion is never descended into).
        assert_eq!(
            by("Method", "make").expect("object method").qualified_name,
            "app/App.kt::Registry::make"
        );
        assert_eq!(count("Method", "peek"), 1);
        assert_eq!(count("Method", "put"), 2, "interface put + override put");
        assert_eq!(count("Method", "empty"), 0, "companion fun is not a Method");
        assert_eq!(count("Function", "empty"), 0, "…nor a free Function");

        // DEFINES_METHOD: owner Class node → its Method (including the object's).
        let defm: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "DEFINES_METHOD")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.target_qualified_name.clone(),
                )
            })
            .collect();
        assert!(
            defm.contains(&(
                "app/App.kt::Class::Registry".into(),
                "app/App.kt::Registry::make".into()
            )),
            "object owner DEFINES_METHOD: {defm:?}"
        );
        assert!(
            defm.contains(&(
                "app/App.kt::Class::Cache".into(),
                "app/App.kt::Cache::peek".into()
            )),
            "{defm:?}"
        );
        assert!(
            !defm.iter().any(|(_, t)| t == "app/App.kt::Cache::empty"),
            "no DEFINES_METHOD for a companion fun: {defm:?}"
        );

        // IMPORTS: the final path segment is the imported name.
        assert!(
            import_pairs(&r).iter().any(|(p, _)| p == "app.util.Helper"),
            "{:?}",
            import_pairs(&r)
        );

        // USES: non-call references (`MAX` returned from `peek`, `SEED` from
        // `make`) are keyed on `ref_name` and sourced from the enclosing method.
        // Keywords / def-names / call args never emit one.
        let usages: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USES")
            .filter_map(|e| {
                e.properties
                    .get("ref_name")
                    .and_then(|v| v.as_str())
                    .map(|rn| (e.source_qualified_name.as_str(), rn))
            })
            .collect();
        assert!(
            usages.contains(&("app/App.kt::Cache::peek", "MAX")),
            "method-body USES of `MAX`: {usages:?}"
        );
        assert!(
            usages.contains(&("app/App.kt::Registry::make", "SEED")),
            "object-method-body USES of `SEED`: {usages:?}"
        );
        assert!(
            !usages
                .iter()
                .any(|(_, rn)| *rn == "fun" || *rn == "val" || *rn == "return" || *rn == "String"),
            "keywords / builtins must not emit USES: {usages:?}"
        );
    }

    // ---- Zig --------------------------------------------------------------

    #[test]
    fn zig_defs_calls_and_import_builtin() {
        const SRC: &str = r#"
const std = @import("std");
/// A greeter.
fn greet(name: i32) i32 {
    return helper(name);
}
"#;
        let r = zig(SRC, "greet.zig");
        let greet = r
            .nodes
            .iter()
            .find(|n| n.label == "Function" && n.name == "greet")
            .expect("greet function");
        assert_eq!(greet.qualified_name, "greet.zig::Function::greet");
        assert_eq!(
            greet.properties.get("doc").and_then(|v| v.as_str()),
            Some("A greeter.")
        );
        let edges = calls_edges(&r);
        assert!(
            edges.contains(&("greet.zig::Function::greet".into(), "helper".into())),
            "{edges:?}"
        );
        // `@import("std")` import.
        assert!(
            import_pairs(&r).contains(&("std".into(), "std".into())),
            "{:?}",
            import_pairs(&r)
        );
    }

    #[test]
    fn zig_cross_file_call_resolves_by_callee_name() {
        let a = zig("fn caller() void { _ = shared(); }", "a.zig");
        let b = zig("fn shared() i32 { return 1; }", "b.zig");
        assert_eq!(first_callee(&a).as_deref(), Some("shared"));
        assert!(b.nodes.iter().any(|n| n.name == "shared"));
    }

    #[test]
    fn zig_variables_members_member_calls_test_and_usages() {
        // Exercises every pass `extract_zig` adds on top of the uniform
        // spec path: every top-level `variable_declaration` → Variable (INCLUDING
        // `const P = struct{…}`, `const std = @import(…)`, `var`); a
        // `struct`-nested `function_declaration` flattened to a free Function
        // (tree-sitter-zig's unnamed container nodes cannot be named); a
        // `test_declaration` → a Function named by its string; member/qualified
        // CALLS (`recv.method()` / `mod.func()`, resolved by the trailing name);
        // and the USAGE walk (identifiers not in a call/import/def-name/keyword).
        const SRC: &str = r#"
const std = @import("std");
const other = @import("other.zig");

pub const Point = struct {
    x: i32,
    y: i32,

    pub fn magnitude(self: Point) i32 {
        return other.absValue(self.x);
    }
};

var counter: i32 = 0;
const MAX: i32 = 100;

pub fn build(p: Point) i32 {
    return p.magnitude();
}

test "builds a point" {
    _ = build(Point{ .x = 1, .y = 2 });
}
"#;
        let r = zig(SRC, "geo.zig");
        let by =
            |label: &str, name: &str| r.nodes.iter().find(|n| n.label == label && n.name == name);

        // Top-level `variable_declaration`s are Variables — struct binding,
        // import bindings, `var`, and typed `const` alike.
        for v in ["std", "other", "Point", "counter", "MAX"] {
            let n = by("Variable", v).unwrap_or_else(|| panic!("Variable {v} missing"));
            assert_eq!(n.qualified_name, format!("geo.zig::Variable::{v}"));
        }
        // A `container_field` (`x` / `y`) is NEVER a node (class-def name
        // resolution fails, so fields are never extracted).
        assert!(by("Field", "x").is_none() && by("Variable", "x").is_none());
        // The struct method flattens to a free Function (no Class/Method nodes).
        assert_eq!(
            by("Function", "magnitude")
                .expect("magnitude fn")
                .qualified_name,
            "geo.zig::Function::magnitude"
        );
        assert!(by("Class", "Point").is_none() && by("Method", "magnitude").is_none());
        // The test becomes a Function named by its string.
        assert!(by("Function", "builds a point").is_some(), "test fn");

        // Member/qualified CALLS resolve by the trailing segment; the source is
        // the enclosing (flattened) Function; a call inside the test sources from
        // the file Module (`__file__`) — a test cannot be named as the source.
        let calls = calls_edges(&r);
        assert!(
            calls.contains(&("geo.zig::Function::magnitude".into(), "absValue".into())),
            "member/qualified call other.absValue: {calls:?}"
        );
        assert!(
            calls.contains(&("geo.zig::Function::build".into(), "magnitude".into())),
            "member call p.magnitude: {calls:?}"
        );
        assert!(
            calls.contains(&("geo.zig::__file__".into(), "build".into())),
            "test-body call sources from Module: {calls:?}"
        );

        // USAGE: a type reference in a signature (`p: Point`) emits a USAGE for
        // `Point`; identifiers inside calls/imports do not. `self` is a keyword.
        let usages: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.properties
                        .get("ref_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect();
        assert!(
            usages.contains(&("geo.zig::Function::build".into(), "Point".into())),
            "param-type usage Point in build: {usages:?}"
        );
        assert!(
            !usages.iter().any(|(_, r)| r == "self"),
            "`self` is a keyword, never a usage: {usages:?}"
        );
    }

    // ---- R ----------------------------------------------------------------

    #[test]
    fn r_defs_calls_and_library_imports() {
        const SRC: &str = r#"
library(stats)
# a greeter
greet <- function(name) {
    helper(name)
}
"#;
        let r = rlang(SRC, "greet.r");
        let greet = r
            .nodes
            .iter()
            .find(|n| n.label == "Function" && n.name == "greet")
            .expect("greet function");
        assert_eq!(greet.qualified_name, "greet.r::Function::greet");
        assert_eq!(
            greet.properties.get("doc").and_then(|v| v.as_str()),
            Some("a greeter")
        );
        let edges = calls_edges(&r);
        // Every R call is sourced from the file node — the enclosing-function
        // resolver never recovers an R function name (it reads the anonymous
        // `function` keyword, not the assigned symbol), so both module-scope and
        // in-function calls fall back to the file module (`greet.r::__file__`).
        assert!(
            edges.contains(&("greet.r::__file__".into(), "helper".into())),
            "{edges:?}"
        );
        // `library` must not be counted as a call.
        assert!(
            !edges.iter().any(|(_, c)| c == "library"),
            "library must not be a call: {edges:?}"
        );
        assert!(
            import_pairs(&r).contains(&("stats".into(), "stats".into())),
            "{:?}",
            import_pairs(&r)
        );
    }

    #[test]
    fn r_top_level_variables_and_usages() {
        // A top-level assignment whose RHS is NOT a function is a `Variable`;
        // one whose RHS IS a function is a `Function`. Bare identifier
        // references (outside a call) emit `USAGE` edges sourced from the file
        // module; identifiers inside a call are suppressed (they are the CALLS
        // edge or its arguments).
        const SRC: &str = r#"
threshold <- 10
scale <- function(x) {
    x * threshold
}
"#;
        let r = rlang(SRC, "vars.r");
        // `threshold` is a Variable, `scale` is a Function.
        assert!(
            r.nodes
                .iter()
                .any(|n| n.label == "Variable" && n.name == "threshold"),
            "{:?}",
            r.nodes
        );
        assert!(
            r.nodes
                .iter()
                .any(|n| n.label == "Function" && n.name == "scale"),
            "{:?}",
            r.nodes
        );
        // USAGE edges are all sourced from the file module node.
        let usages: Vec<(String, String)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "USAGE")
            .map(|e| {
                (
                    e.source_qualified_name.clone(),
                    e.properties
                        .get("ref_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                )
            })
            .collect();
        assert!(
            usages.iter().all(|(src, _)| src == "vars.r::__file__"),
            "all R usages source from the file module: {usages:?}"
        );
        // The in-function reference `threshold` emits a usage.
        assert!(usages.iter().any(|(_, n)| n == "threshold"), "{usages:?}");
    }

    #[test]
    fn r_cross_file_call_resolves_by_callee_name() {
        let a = rlang("caller <- function() { shared() }", "a.r");
        let b = rlang("shared <- function() { 1 }", "b.r");
        assert_eq!(first_callee(&a).as_deref(), Some("shared"));
        assert!(b.nodes.iter().any(|n| n.name == "shared"));
    }

    // =======================================================================
    // Behaviour-preservation: the eight migrated languages run through the
    // data-driven `spec_extract` engine and must produce the SAME output the
    // bespoke extractors produced. The per-language tests above (and the ~131
    // pre-existing tests) assert that. This test pins the structural
    // invariants the migration must never regress.
    // =======================================================================

    #[test]
    fn migrated_languages_unchanged_invariants() {
        // Python: method owned by class, free function segment.
        let r = py(
            "class K:\n    def m(self):\n        pass\ndef g():\n    pass\n",
            "k.py",
        );
        assert!(r.nodes.iter().any(|n| n.qualified_name == "k.py::K::m"));
        assert!(r
            .nodes
            .iter()
            .any(|n| n.qualified_name == "k.py::Function::g"));

        // Go: receiver-owned method qname (the nuance the generic spec must
        // express via Owner::GoReceiver).
        let go_src = "package p\ntype Adder struct{}\nfunc (a *Adder) Add() {}\nfunc Free() {}\n";
        let rg = extract(Language::Go, go_src.as_bytes(), "g.go").unwrap();
        assert!(
            rg.nodes
                .iter()
                .any(|n| n.qualified_name == "g.go::Adder::Add"),
            "Go receiver ownership must survive migration: {:?}",
            rg.nodes
                .iter()
                .map(|n| &n.qualified_name)
                .collect::<Vec<_>>()
        );

        // C++: out-of-line `Class::method` ownership (CppClass nuance).
        let cpp_src = "struct S { void m(); };\nvoid S::m() {}\n";
        let rc = extract(Language::Cpp, cpp_src.as_bytes(), "s.cpp").unwrap();
        assert!(
            rc.nodes
                .iter()
                .any(|n| n.label == "Method" && n.qualified_name == "s.cpp::S::m"),
            "C++ out-of-line method ownership must survive migration: {:?}",
            rc.nodes
                .iter()
                .map(|n| (&n.label, &n.qualified_name))
                .collect::<Vec<_>>()
        );
    }

    /// Java member pass: class-body `field_declaration`s must yield BOTH a
    /// `Field` and a `Variable` node (the extractor pushes one of each), enum
    /// constants must yield `Variable` nodes, every owned method/constructor
    /// must get a `DEFINES_METHOD` edge from its type, and `new T(...)` must be
    /// a CALLS callee — the four behaviours exercised on java_medium.
    #[test]
    fn java_fields_variables_defines_method_and_constructor_calls() {
        const SRC: &str = r#"
package p;
public final class Normalizer {
    public final int score;
    public final long checksum;

    private Normalizer(int score, long checksum) {
        this.score = score;
        this.checksum = checksum;
    }

    public static Normalizer make(int s) {
        return new Normalizer(s, 0L);
    }
}

enum Color { RED, GREEN }
"#;
        let r = extract(Language::Java, SRC.as_bytes(), "N.java").unwrap();

        // Each of the two class-body fields → one Field + one Variable.
        for f in ["score", "checksum"] {
            assert!(
                r.nodes.iter().any(|n| n.label == "Field"
                    && n.name == f
                    && n.qualified_name == format!("N.java::Normalizer::{f}")),
                "expected Field {f}: {:?}",
                r.nodes
                    .iter()
                    .map(|n| (&n.label, &n.qualified_name))
                    .collect::<Vec<_>>()
            );
            assert!(
                r.nodes.iter().any(|n| n.label == "Variable"
                    && n.name == f
                    && n.qualified_name == format!("N.java::Variable::{f}")),
                "expected Variable {f}"
            );
        }
        assert_eq!(
            r.nodes.iter().filter(|n| n.label == "Field").count(),
            2,
            "exactly two Field nodes"
        );

        // Enum constants → Variable nodes.
        for c in ["RED", "GREEN"] {
            assert!(
                r.nodes.iter().any(|n| n.label == "Variable" && n.name == c),
                "expected enum-member Variable {c}"
            );
        }

        // DEFINES_METHOD edge from the class node to each owned method /
        // constructor. Two members here: the constructor `Normalizer` and the
        // method `make`.
        let dm: Vec<(&str, &str)> = r
            .edges
            .iter()
            .filter(|e| e.edge_type == "DEFINES_METHOD")
            .map(|e| {
                (
                    e.source_qualified_name.as_str(),
                    e.target_qualified_name.as_str(),
                )
            })
            .collect();
        assert!(
            dm.contains(&(
                "N.java::Class::Normalizer",
                "N.java::Normalizer::Normalizer"
            )),
            "constructor DEFINES_METHOD: {dm:?}"
        );
        assert!(
            dm.contains(&("N.java::Class::Normalizer", "N.java::Normalizer::make")),
            "method DEFINES_METHOD: {dm:?}"
        );

        // `new Normalizer(...)` is a constructor CALL (C counts
        // object_creation_expression as a call).
        assert!(
            r.edges.iter().any(|e| e.edge_type == "CALLS"
                && e.properties.get("callee_name").and_then(|v| v.as_str()) == Some("Normalizer")),
            "expected constructor CALLS edge to Normalizer: {:?}",
            r.edges
                .iter()
                .filter(|e| e.edge_type == "CALLS")
                .map(|e| e.properties.get("callee_name").cloned())
                .collect::<Vec<_>>()
        );
    }
}
