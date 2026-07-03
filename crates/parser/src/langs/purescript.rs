//! PureScript — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-purescript` dependency (a git dependency on
//! `postsolar/tree-sitter-purescript` v0.3.0 — there is no crates.io release —
//! which builds against tree-sitter 0.25 via the `tree-sitter-language` shim).
//!
//! Status: **experimental / partial**. The grammar (verified with
//! `examples/dump_purs.rs`) exposes clean, distinct node kinds:
//!
//!   * `function`   — a value binding `f a b = …`, name on the `name:` field
//!                    (a `variable`)                              → `Function`
//!   * `data`       — `data T = …`, name on `name:` (a `type`)    → `Data`
//!   * `newtype`    — `newtype T = …`, name on `name:` (a `type`) → `Data`
//!   * `type_alias` — `type T = …`, name on `name:` (a `type`)    → `Type`
//!
//! Every captured def node exposes its name as a *direct* child of the `name:`
//! field, so the `Capture` strategy applies uniformly: the def node is the
//! captured name's parent, exactly the node keyed by each `DefRule`.
//!
//! Imprecision / honesty:
//!   * A top-level type SIGNATURE (`f :: Int -> Int`) parses as a separate
//!     `signature` node; only the value `function` binding is captured, so a
//!     function with an equation is emitted once (the signature is ignored, not
//!     double-counted).
//!   * `class_declaration` is intentionally NOT captured: its name is nested
//!     under `class_head > class_name > type` (not a direct `name:` child), so
//!     the `Capture` (name-parent) rule cannot key it without emitting a
//!     `class_name` def node. Classes are therefore omitted.
//!   * CALLS captures only the *head* of an application `exp_apply` (the applied
//!     function). Operator applications (`a + b`), qualified calls
//!     (`Data.List.sum`), and record/section forms are best-effort or dropped.
//!   * `import` declarations are not expanded into IMPORTS edges: no PureScript
//!     import strategy exists in `ImportStrategy`, so `import_query` is empty.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `function`   → `Function` (free; PureScript has no methods/ownership).
///  * `data` / `newtype` → `Data` types.
///  * `type_alias` → `Type`.
///
/// Every def node carries its name in the `name:` field (a `variable` for a
/// `function`, a `type` for the type kinds), so the `Capture` strategy (name =
/// `@name`, def = its parent) applies uniformly. No ownership is modelled
/// (`owner_kinds` empty): PureScript top-level bindings are not methods.
static PURESCRIPT_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function"),
        DefRule::ty("data", "Data"),
        DefRule::ty("newtype", "Data"),
        DefRule::ty("type_alias", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // PureScript `import` declarations are not extracted (no PureScript import
    // strategy exists); `import_query` is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // PureScript line comments use `--`; block/doc comments are `{- -}` / `-- |`
    // which the line-dash helper collapses into a leading `--` run.
    docs: DocStyle::LineDashComment,
};

/// Each def node carries its name in the `name:` field. Capture that node as
/// `@name`; the engine derives the def node as its parent (`function` / `data`
/// / `newtype` / `type_alias`) and keys the matching `DefRule` on that parent's
/// kind. `function` names are a `variable`; the type kinds' names are a `type`.
const DEFINITIONS: &str = r#"
    (function   name: (variable) @name) @def
    (data       name: (type)     @name) @def
    (newtype    name: (type)     @name) @def
    (type_alias name: (type)     @name) @def
"#;

/// A function application `f a b` parses as `(exp_apply (exp_name (variable
/// "f")) (exp_name …) …)`; the callee is the head `variable`, captured by
/// anchoring `.` to the FIRST child of `exp_apply` so the arguments are not
/// mistaken for the callee. The engine hangs the CALLS edge off the enclosing
/// `function` (which exposes a `name:` field, so the source endpoint resolves).
const CALLS: &str = r#"
    (exp_apply
      . (exp_name (variable) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "purescript",
        extensions: &["purs"],
        filenames: &[],
        grammar: || tree_sitter_purescript::LANGUAGE.into(),
        spec: &PURESCRIPT_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
