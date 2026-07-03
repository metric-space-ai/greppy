//! D — onboarded via the parallel-safe registry (`crate::registry`). This whole
//! file is the entire surface: it declares the spec + queries + grammar and
//! self-registers with `inventory::submit!`. No shared file is edited (build.rs
//! discovers this module automatically); the only Cargo.toml line added is the
//! `tree-sitter-d` dependency (crates.io `tree-sitter-d` v0.8.2, which builds
//! against tree-sitter 0.25 via the `tree-sitter-language` shim).
//!
//! Status: **experimental**. Node kinds were verified against the real grammar
//! with `examples/dump_d.rs`. The `tree-sitter-d` grammar exposes clean, distinct
//! definition kinds, but — crucially — it puts NO `name:` field on any of them:
//! the name is an *unnamed* `identifier` child (`fields` is empty `{}` in the
//! grammar's `node-types.json`). The relevant kinds are:
//!
//!   * `function_declaration` — `int add(int a, int b) { … }`; the name is the
//!     sole direct `identifier` child (parameters live under `parameters`, the
//!     return type under `type`)                                    → `Function`
//!   * `struct_declaration`   — `struct Point { … }`; name = direct `identifier`
//!                                                                   → `Struct`
//!   * `class_declaration`    — `class Widget { … }`; name = direct `identifier`
//!                                                                   → `Class`
//!
//! With the `Capture` name strategy the definition node is the *parent* of the
//! captured `@name` identifier — exactly the `function_declaration` /
//! `struct_declaration` / `class_declaration` node keyed by each `DefRule`. A
//! single anchored `identifier` capture per container yields the right def node
//! and name.
//!
//! Imprecision / honesty:
//!   * CALLS edges are captured (a `call_expression`'s head `identifier` is the
//!     callee) but they are **not resolved to a source endpoint**: the engine's
//!     `enclosing_callable_qname` reads the enclosing function's name via
//!     `child_by_field_name("name")`, and `function_declaration` has NO `name:`
//!     field, so the enclosing-callable lookup returns `None` and the CALLS edge
//!     is dropped — the same limitation the codebase already documents for Julia.
//!     The `call_query` is kept (it is correct and harmless: it simply produces
//!     no surviving edges under the current generic resolver) so that if the
//!     engine ever grows unnamed-name resolution, D calls light up for free.
//!   * Methods inside a `class`/`struct` `aggregate_body` are captured as free
//!     `Function`s (no ownership): D members also lack a `name:` field, so the
//!     `EnclosingName` owner rule cannot key them; modelling ownership would
//!     require a bespoke walker. This is an intentional, honest over-flattening.
//!   * `import`/`module` declarations are not expanded into IMPORTS edges: no D
//!     import strategy exists in `ImportStrategy`, so `import_query` is empty.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `function_declaration` → `Function` (free; D method ownership is not
///     modelled — members lack a `name:` field, see the module docs).
///  * `struct_declaration`   → `Struct`.
///  * `class_declaration`    → `Class`.
///
/// Every def node carries its name as an unnamed direct `identifier` child, so
/// the `Capture` strategy (name = `@name`, def = its parent) applies uniformly.
static D_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function_declaration"),
        DefRule::ty("struct_declaration", "Struct"),
        DefRule::ty("class_declaration", "Class"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // D `import`/`module` declarations are not extracted (no D import strategy
    // exists); `import_query` is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // D uses C-style comments (`//` line and `/* */` / `/** */` block).
    docs: DocStyle::CBlockOrLine,
};

/// Each def node carries its name as the sole direct `identifier` child (the
/// return type is a `type`, parameters live under `parameters`, so no other bare
/// `identifier` is a direct child). Capture that `identifier` as `@name`; the
/// engine derives the def node as its parent (`function_declaration` /
/// `struct_declaration` / `class_declaration`) and keys the matching `DefRule`.
/// The `.` anchor pins each capture to a direct child of the container.
const DEFINITIONS: &str = r#"
    (function_declaration (identifier) @name)
    (struct_declaration   (identifier) @name)
    (class_declaration    (identifier) @name)
"#;

/// A function call `total(a, b)` parses as `(call_expression (identifier
/// "total") (named_arguments …))`; the callee is the head `identifier`, captured
/// by anchoring `.` to the FIRST child of `call_expression` so nested argument
/// identifiers are not mistaken for the callee. (See module docs: these edges do
/// not currently resolve a source endpoint because `function_declaration` has no
/// `name:` field, but the capture itself is correct.)
const CALLS: &str = r#"
    (call_expression
      . (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "d",
        extensions: &["d"],
        filenames: &[],
        grammar: || tree_sitter_d::LANGUAGE.into(),
        spec: &D_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
