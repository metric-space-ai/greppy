//! Julia — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically).
//!
//! Status: **experimental**. Julia's tree-sitter grammar does not expose a
//! `name:` field on `function_definition` — the name lives nested at
//! `function_definition > signature > call_expression > identifier`. With the
//! `Capture` name strategy the definition node is therefore the inner
//! `call_expression` (the one that wraps the function name + parameter list),
//! not the whole `function_definition`. That is enough to emit Function nodes,
//! but because the engine's enclosing-callable resolution needs a `name:` field
//! (which `call_expression` lacks), CALLS edges whose source is a Julia function
//! are NOT resolved. Extraction is intentionally NOT claimed as `supported`
//! (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Function definitions are captured through the inner `call_expression` that
/// carries the function name. Both the long form (`function f(...) ... end`) and
/// the short form (`f(...) = ...`) route their name identifier's parent to a
/// `call_expression`, so a single `DefRule::func("call_expression")` covers both
/// — and the DEFINITIONS query below only ever matches the *signature* / short
/// assignment call, never an ordinary call expression, so no spurious defs leak.
static JULIA_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("call_expression")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Julia imports (`using` / `import`) are not extracted yet (import_query is
    // empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// A `function f(...)`/`macro m(...)` names the callee inside its `signature`; a
/// short-form `f(...) = expr` names it in the `call_expression` on the left of
/// an `assignment`. In every case the name is the FIRST `identifier` child of a
/// `call_expression` (function position), captured as `@name`; its parent
/// `call_expression` is the `@def`.
const DEFINITIONS: &str = r#"
    (function_definition
      (signature
        (call_expression
          .
          (identifier) @name) @def))

    (macro_definition
      (signature
        (call_expression
          .
          (identifier) @name) @def))

    (assignment
      .
      (call_expression
        .
        (identifier) @name) @def
      (operator) @_eq
      (#eq? @_eq "="))
"#;

/// Ordinary `foo(...)` call expressions. The callee is the first `identifier`
/// child (function position). This also matches the name-position call inside a
/// signature / short-form definition, but those are self-references and are
/// harmless (the enclosing-callable source cannot be resolved for Julia, so no
/// edge is materialised for them anyway).
const CALLS: &str = r#"
    (call_expression
      .
      (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "julia",
        extensions: &["jl"],
        filenames: &[],
        grammar: || tree_sitter_julia::LANGUAGE.into(),
        spec: &JULIA_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
