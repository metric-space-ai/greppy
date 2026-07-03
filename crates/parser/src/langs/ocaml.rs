//! OCaml — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically).
//!
//! Status: **experimental**. OCaml models a function as a `let_binding` whose
//! `pattern:` field holds the `value_name` and which carries one or more
//! `parameter` children (a value binding with no `parameter` is a plain
//! constant, not a function). Definition extraction is precise, but CALLS
//! edges are LIMITED: the engine resolves an enclosing callable's own name via
//! `child_by_field_name("name")`, while OCaml exposes the name under the
//! `pattern:` field, so a call's *source* callable cannot be named by the
//! generic engine and the CALLS edge is dropped. Callee capture itself works.
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// A `let ... = <params> ...` binding becomes a Function definition. With
/// `NameStrategy::Capture` the engine takes the `@name` node and uses its
/// PARENT as the definition node, so the `@name` here is the `value_name`
/// under a `let_binding`; the parent kind `"let_binding"` is what the DefRule
/// keys on. Requiring a `(parameter)` child excludes plain value bindings
/// (`let x = 3`) from being counted as functions.
static OCAML_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("let_binding")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // OCaml `open`/`include` imports are not extracted yet (import_query is
    // empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::None,
};

/// `let add a b = ...` parses as `(value_definition (let_binding pattern:
/// (value_name "add") (parameter ...) ... ))`; capture the `value_name` as the
/// name and require at least one `parameter` so only functions match. The
/// whole `let_binding` is captured as `@def` for documentation, though the
/// Capture strategy derives the def node from `@name`'s parent.
const DEFINITIONS: &str = r#"
    (let_binding
      pattern: (value_name) @name
      (parameter)) @def
"#;

/// Function application `f x` / `M.f x` parses as `(application_expression
/// function: (value_path ... (value_name) @callee) ...)`; the final
/// `value_name` inside the callee `value_path` is the called function.
const CALLS: &str = r#"
    (application_expression
      function: (value_path (value_name) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "ocaml",
        extensions: &["ml", "mli"],
        filenames: &[],
        grammar: || tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        spec: &OCAML_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
