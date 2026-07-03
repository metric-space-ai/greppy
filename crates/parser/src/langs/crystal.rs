//! Crystal — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-crystal` dependency (a git dependency on
//! `crystal-lang-tools/tree-sitter-crystal` — there is no crates.io release —
//! which builds against tree-sitter 0.25 via the `tree-sitter-language` shim,
//! exactly like `tree-sitter-purescript`).
//!
//! Status: **experimental**. The grammar (verified with
//! `examples/dump_crystal.rs`) exposes clean, distinct node kinds, each carrying
//! its name on a `name:` field:
//!
//!   * `method_def`  — `def f(...)`, name on `name:` (an `identifier`)
//!                     → `Function` (free) / `Method` (owned)
//!   * `class_def`   — `class C`, name on `name:` (a `constant`)   → `Class`
//!   * `module_def`  — `module M`, name on `name:` (a `constant`)  → `Module`
//!   * `struct_def`  — `struct S`, name on `name:` (a `constant`)  → `Struct`
//!   * `enum_def`    — `enum E`, name on `name:` (a `constant`)    → `Enum`
//!
//! Every captured def node exposes its name as a *direct* child of the `name:`
//! field, so the `Capture` strategy applies uniformly: the def node is the
//! captured name's parent, exactly the node keyed by each `DefRule`.
//!
//! Ownership: a `method_def` nested in a `class_def` / `module_def` /
//! `struct_def` / `enum_def` is a `Method` owned by that type's `name:`
//! constant (`owner_kinds` below). Because every owner AND every `method_def`
//! exposes a `name:` field, the engine's enclosing-callable resolution succeeds,
//! so CALLS edges whose source is a Crystal method ARE resolved.
//!
//! CALLS: a call site parses as `(call method: (identifier) @callee …)`. A
//! qualified/receiver call (`Math.sqrt(1)`) still carries `method: (identifier
//! "sqrt")`, so the final method segment is captured while the receiver is
//! dropped (best-effort). Operator calls (`a + b`) parse as `(call method:
//! (operator "+"))`, so keying the callee capture on `(identifier)` naturally
//! excludes operators. `private def total` wraps the `method_def` in a
//! `visibility_modifier`, which is transparent to both the `name:`-parent
//! capture and the ancestor-walk ownership resolution.
//!
//! Imprecision / honesty:
//!   * Crystal `require "..."` statements are not expanded into IMPORTS edges:
//!     no Crystal import strategy exists in `ImportStrategy`, so `import_query`
//!     is empty (the IMPORTS pass is inert).
//!   * `struct`/`enum`/`module` members other than `method_def` are not
//!     captured; only the top-level type and its methods are surfaced.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `method_def` → `Function` when free, `Method` when owned by an enclosing
///    class/module/struct/enum (via that owner's `name:` constant).
///  * `class_def` / `module_def` / `struct_def` / `enum_def` → the matching type
///    label (never owned, not callable).
///
/// Every def node carries its name in the `name:` field (an `identifier` for a
/// `method_def`, a `constant` for the type kinds), so the `Capture` strategy
/// (name = `@name`, def = its parent) applies uniformly.
static CRYSTAL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::method("method_def"),
        DefRule::ty("class_def", "Class"),
        DefRule::ty("module_def", "Module"),
        DefRule::ty("struct_def", "Struct"),
        DefRule::ty("enum_def", "Enum"),
    ],
    owner_kinds: &["class_def", "module_def", "struct_def", "enum_def"],
    calls: CallSpec { skip_callees: &[] },
    // Crystal `require "..."` is not extracted (no Crystal import strategy
    // exists); `import_query` is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // Crystal line comments use `#` (there are no block comments).
    docs: DocStyle::LineHashComment,
};

/// Each def node carries its name in the `name:` field. Capture that node as
/// `@name`; the engine derives the def node as its parent (`method_def` /
/// `class_def` / `module_def` / `struct_def` / `enum_def`) and keys the matching
/// `DefRule` on that parent's kind. `method_def` names are an `identifier`; the
/// type kinds' names are a `constant`.
const DEFINITIONS: &str = r#"
    (method_def name: (identifier) @name) @def
    (class_def  name: (constant)   @name) @def
    (module_def name: (constant)   @name) @def
    (struct_def name: (constant)   @name) @def
    (enum_def   name: (constant)   @name) @def
"#;

/// A call site parses as `(call method: (identifier) @callee …)`. Keying on
/// `(identifier)` (not `operator`) excludes operator applications; a qualified
/// call (`Math.sqrt`) still carries `method: (identifier "sqrt")`, so the final
/// method segment is captured (receiver dropped). The engine hangs the CALLS
/// edge off the enclosing `method_def` (which exposes a `name:` field, so the
/// source endpoint resolves).
const CALLS: &str = r#"
    (call
      method: (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "crystal",
        extensions: &["cr"],
        filenames: &[],
        grammar: || tree_sitter_crystal::LANGUAGE.into(),
        spec: &CRYSTAL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
