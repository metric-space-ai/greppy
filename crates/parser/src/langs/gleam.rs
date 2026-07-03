//! Gleam — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-gleam` dependency.
//!
//! Status: **experimental**. The `tree-sitter-gleam` grammar (v1.0.0, built on
//! `tree-sitter-language`, ABI-compatible with tree-sitter 0.25) models a
//! function as a `function` node with a `name:` field (an `identifier`), and a
//! type declaration as `type_definition > type_name`, where `type_name` carries
//! the `name:` field (a `type_identifier`). Calls are `function_call` nodes with
//! a `function:` field that is either a bare `identifier` (`add(...)`) or a
//! `field_access` for module-qualified calls (`io.println(...)`); in the latter
//! case the callee is the `field:` `label`. Extraction is best-effort (no
//! golden-master vs C), so it is NOT claimed as `supported`.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `function` — top-level `fn` / `pub fn` declarations → `Function`. The
///    name is the `name:` `identifier`, whose parent is the `function` node, so
///    `DefRule::func("function")` keys on that parent kind.
///  * `type_name` — the name-bearing child of a `type_definition`. With the
///    `Capture` strategy the captured `type_identifier`'s parent is `type_name`,
///    so the type DefRule keys on `"type_name"` (labelled `Type`).
///
/// Gleam has no methods (functions are never owned by a type), so `owner_kinds`
/// is empty and every function is free.
static GLEAM_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function"),
        DefRule::ty("type_name", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Gleam imports (`import gleam/io`) are not extracted yet (import_query is
    // empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::None,
};

/// `pub fn add(...) { ... }` parses as `(function name: (identifier) @name)`;
/// the type declaration `pub type Cat { ... }` parses as
/// `(type_definition (type_name name: (type_identifier) @name))`. In both cases
/// the captured name node's parent is the def node (`function` / `type_name`).
const DEFINITIONS: &str = r#"
    (function
      name: (identifier) @name) @def
    (type_name
      name: (type_identifier) @name) @def
"#;

/// Calls parse as `function_call` with the callee in the `function:` field:
///  * a bare `identifier` for local calls (`add(1, 2)`), or
///  * a `field_access` for module-qualified calls (`io.println(...)`), whose
///    `field:` `label` is the callee name.
const CALLS: &str = r#"
    (function_call
      function: (identifier) @callee)
    (function_call
      function: (field_access
        field: (label) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "gleam",
        extensions: &["gleam"],
        filenames: &[],
        grammar: || tree_sitter_gleam::LANGUAGE.into(),
        spec: &GLEAM_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
