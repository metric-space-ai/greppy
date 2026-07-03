//! F# — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-fsharp` dependency.
//!
//! Status: **experimental**. The `tree-sitter-fsharp` grammar (0.3.0, built on
//! `tree-sitter-language`, tree-sitter 0.25-compatible) models a `let`
//! function/value binding as a `function_or_value_defn` whose *signature* is a
//! `function_declaration_left` node. That signature node has NO `name:` field —
//! the function name is simply its first `identifier` child — so with the
//! `Capture` name strategy the definition node is the `function_declaration_left`
//! (parent of the `@name` identifier). Types are captured through the
//! `type_name` node that every `record`/`union`/`interface`/`enum`/`abbrev`
//! type definition carries (its `type_name:` field holds the identifier).
//!
//! Calls: function application `f a b` parses as nested `application_expression`
//! nodes; the callee is the `identifier` inside the `long_identifier_or_op` in
//! the innermost function position. The callee is captured, but because
//! `function_declaration_left` exposes no `name:` field, the engine's
//! enclosing-callable resolution (which needs `child_by_field_name("name")`)
//! cannot attach a source endpoint to a call sitting in a function body — so,
//! exactly like Julia, CALLS *edges whose source is an F# function are not
//! resolved*. Definition/type extraction works; call-graph edges are best-effort
//! only. NOT claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// `let f a b = …` → the `function_declaration_left` signature becomes a
/// `Function`. `type T = …` (record/union/interface/enum/abbrev) → the
/// `type_name` node becomes a `Type`. Both route their name identifier's parent
/// to the captured def node, so the `Capture` strategy applies uniformly.
static FSHARP_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function_declaration_left"),
        DefRule::ty("type_name", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // F# imports (`open Foo`) are not extracted yet (import_query is empty); any
    // variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineSlashComment,
};

/// A `let f a b = …` binding names the function in the FIRST `identifier` child
/// of its `function_declaration_left`; that node is the `@def`. A type
/// definition carries its name in the `type_name:` field of its `type_name`
/// node; capturing that identifier makes `type_name` the `@def`.
const DEFINITIONS: &str = r#"
    (function_declaration_left
      .
      (identifier) @name) @def

    (type_name
      type_name: (identifier) @name) @def
"#;

/// Function application `f a b` is `(application_expression (application_expression
/// (long_identifier_or_op (identifier "f")) …) …)`; the callee is the
/// `identifier` inside the `long_identifier_or_op` in the innermost function
/// (first-child) position.
const CALLS: &str = r#"
    (application_expression
      .
      (long_identifier_or_op (identifier) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "fsharp",
        extensions: &["fs", "fsx"],
        filenames: &[],
        grammar: || tree_sitter_fsharp::LANGUAGE_FSHARP.into(),
        spec: &FSHARP_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
