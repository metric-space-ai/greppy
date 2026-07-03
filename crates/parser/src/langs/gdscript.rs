//! GDScript — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-gdscript` dependency (a crates.io release, v6.1.0,
//! which builds against tree-sitter 0.25 via the `tree-sitter-language` shim —
//! its grammar accessor is `tree_sitter_gdscript::LANGUAGE`).
//!
//! Status: **experimental / partial**. GDScript is Godot's scripting language.
//! The grammar (verified with `examples/dump_gd.rs`) exposes clean, distinct
//! node kinds, each carrying its name on a `name:` field whose child is a
//! `name` node:
//!
//!   * `function_definition` — `func f(...)`, name on `name:` (a `name`)
//!                             → `Function` (free) / `Method` (owned)
//!   * `class_definition`    — inner `class C:`, name on `name:` (a `name`)
//!                             → `Class`
//!   * `class_name_statement`— the script's own `class_name Player`, name on
//!                             `name:` (a `name`)                    → `Class`
//!   * `const_statement`     — `const X = …`, name on `name:` (a `name`)
//!                             → `Const`
//!   * `variable_statement`  — `var y = …`, name on `name:` (a `name`)
//!                             → `Variable`
//!   * `signal_statement`    — `signal died`, name on `name:` (a `name`)
//!                             → `Signal`
//!
//! Every captured def node exposes its name as a *direct* child of the `name:`
//! field, so the `Capture` strategy applies uniformly: the def node is the
//! captured name's parent, exactly the node keyed by each `DefRule`.
//!
//! Ownership: a `function_definition` nested inside an inner `class_definition`
//! (via that class's `class_body`) is a `Method` owned by that class's `name:`
//! (`owner_kinds` below). A top-level `func` is a free `Function`. Because both
//! the owner (`class_definition`) and the `function_definition` expose a `name:`
//! field, the engine's enclosing-callable resolution succeeds, so CALLS edges
//! whose source is a GDScript method ARE resolved.
//!
//! CALLS: a free call parses as `(call (identifier) @callee arguments: …)` where
//! the callee is the FIRST child (anchored with `.` so argument identifiers are
//! not mistaken for the callee). A qualified/receiver call (`self.bar()`,
//! `Global.qux()`) parses as `(attribute … (attribute_call (identifier) @callee
//! arguments: …))`, so the final method segment is captured while the receiver
//! is dropped (best-effort). The engine hangs the CALLS edge off the enclosing
//! `function_definition` (which exposes a `name:` field, so the source endpoint
//! resolves).
//!
//! Imprecision / honesty:
//!   * GDScript `preload("res://…")` / `load(…)` are not expanded into IMPORTS
//!     edges: no GDScript import strategy exists in `ImportStrategy`, so
//!     `import_query` is empty (the IMPORTS pass is inert).
//!   * `const` / `var` / `signal` are surfaced as structural definitions
//!     (`Const` / `Variable` / `Signal`) so a GDScript file is greppable as
//!     structure; they are not callable and never owned.
//!   * `enum` bodies and inner enum members are not captured.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `function_definition` → `Function` when free, `Method` when owned by an
///    enclosing inner `class_definition` (via that class's `name:`).
///  * `class_definition` / `class_name_statement` → `Class`.
///  * `const_statement` / `variable_statement` / `signal_statement` →
///    `Const` / `Variable` / `Signal` (structural, never owned, not callable).
///
/// Every def node carries its name in the `name:` field (a `name` node), so the
/// `Capture` strategy (name = `@name`, def = its parent) applies uniformly.
static GDSCRIPT_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::method("function_definition"),
        DefRule::ty("class_definition", "Class"),
        DefRule::ty("class_name_statement", "Class"),
        DefRule::ty("const_statement", "Const"),
        DefRule::ty("variable_statement", "Variable"),
        DefRule::ty("signal_statement", "Signal"),
    ],
    owner_kinds: &["class_definition"],
    calls: CallSpec { skip_callees: &[] },
    // GDScript `preload`/`load` are not extracted (no GDScript import strategy
    // exists); `import_query` is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // GDScript line comments use `#` (there are no block comments).
    docs: DocStyle::LineHashComment,
};

/// Each def node carries its name in the `name:` field (a `name` node). Capture
/// that node as `@name`; the engine derives the def node as its parent
/// (`function_definition` / `class_definition` / `class_name_statement` /
/// `const_statement` / `variable_statement` / `signal_statement`) and keys the
/// matching `DefRule` on that parent's kind.
const DEFINITIONS: &str = r#"
    (function_definition  name: (name) @name) @def
    (class_definition     name: (name) @name) @def
    (class_name_statement name: (name) @name) @def
    (const_statement      name: (name) @name) @def
    (variable_statement   name: (name) @name) @def
    (signal_statement     name: (name) @name) @def
"#;

/// A free call parses as `(call (identifier) @callee arguments: …)`; the callee
/// is the head `identifier`, captured by anchoring `.` to the FIRST child of
/// `call` so an argument identifier is never mistaken for the callee. A
/// qualified call (`self.bar()`, `Global.qux()`) parses as `(attribute …
/// (attribute_call (identifier) @callee …))`, capturing the final method
/// segment (receiver dropped). The engine hangs the CALLS edge off the
/// enclosing `function_definition` (which exposes a `name:` field, so the source
/// endpoint resolves).
const CALLS: &str = r#"
    (call . (identifier) @callee)
    (attribute_call (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "gdscript",
        extensions: &["gd"],
        filenames: &[],
        grammar: || tree_sitter_gdscript::LANGUAGE.into(),
        spec: &GDSCRIPT_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}