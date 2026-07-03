//! Haskell — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically).
//!
//! Status: **experimental**. The tree-sitter-haskell grammar models a
//! top-level equation `f x = ...` as a `function` node (with a `name:`
//! field holding a `variable`) and a nullary binding `f = ...` as a `bind`
//! node. Both are treated as Function definitions here. Function
//! application `f a b` parses as nested `apply` nodes; the callee is the
//! `variable` in the innermost `function:` position. This capture is a
//! best-effort heuristic (no golden-master vs C), so it is NOT claimed as
//! `supported`.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Top-level `function` equations and nullary `bind`ings both become Function
/// definitions. The name is taken from the `@name` capture (a `variable`).
static HASKELL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("function"), DefRule::func("bind")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Haskell imports (`import Foo`) are not extracted yet (import_query is
    // empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineDashComment,
};

/// `add a b = ...` parses as `(function name: (variable) @name)`; the nullary
/// `main = ...` parses as `(bind name: (variable) @name)`. Only top-level
/// declarations carry a `name:` field, so this scopes to real definitions.
const DEFINITIONS: &str = r#"
    (function
      name: (variable) @name) @def
    (bind
      name: (variable) @name) @def
"#;

/// Function application `f a b` is `(apply function: (apply function:
/// (variable "f") ...) ...)`; capturing the `variable` in the innermost
/// `function:` position yields the callee identifier.
const CALLS: &str = r#"
    (apply
      function: (variable) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "haskell",
        extensions: &["hs"],
        filenames: &[],
        grammar: || tree_sitter_haskell::LANGUAGE.into(),
        spec: &HASKELL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}