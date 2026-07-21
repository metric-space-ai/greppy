//! OCaml — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically).
//!
//! Status: **experimental**. OCaml models a function as a `let_binding` whose
//! `pattern:` field holds the `value_name`; type declarations use `type_binding`
//! with a `name:` field. The bespoke extractor consumes these provider queries
//! and supplies callable attribution, imports, and type references. Not claimed
//! as `supported` (no verification corpus).

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
    defs: &[
        DefRule::func("let_binding"),
        DefRule::ty("type_binding", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // The bespoke OCaml extractor consumes IMPORTS below directly. The Bash
    // strategy is inert on this path.
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
    (type_binding
      name: (type_constructor) @name) @def
"#;

/// Function application `f x` / `M.f x` parses as `(application_expression
/// function: (value_path ... (value_name) @callee) ...)`; the final
/// `value_name` inside the callee `value_path` is the called function.
const CALLS: &str = r#"
    (application_expression
      function: (value_path (value_name) @callee))
"#;

/// OCaml namespace imports. The module expression is captured generically so
/// both simple (`open Helper`) and qualified module paths are available to the
/// bespoke import pass.
pub(crate) const IMPORTS: &str = r#"
    [
      (open_module module: (_) @imported)
      (include_module module: (_) @imported)
    ] @import
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
        import_query: IMPORTS,
    }
}
