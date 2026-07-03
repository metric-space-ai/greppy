//! Solidity — onboarded via the parallel-safe registry (`crate::registry`).
//! The whole surface is this one file: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-solidity` dependency.
//!
//! Status: **experimental**. The `tree-sitter-solidity` grammar exposes clean,
//! distinct node kinds: `contract_declaration` / `interface_declaration` /
//! `library_declaration` are types, and `function_definition` /
//! `modifier_definition` carry a `name:` field (an `identifier`), so the
//! `Capture` name strategy applies uniformly. Functions/modifiers lexically
//! enclosed by a contract/interface/library are owned by it (→ `Method`);
//! top-level (free) functions become `Function`s. Calls parse as
//! `call_expression` whose `function:` field wraps an `expression` holding the
//! callee `identifier`. `constructor_definition` is intentionally NOT captured
//! (the grammar gives it no `name:` field). Imprecision: member/qualified calls
//! (`x.foo()`, `Lib.bar()`) resolve to the trailing identifier only, and
//! `import` directives are not expanded (no Solidity import strategy exists).
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `contract_declaration` / `interface_declaration` / `library_declaration`
///    → types (`Contract` / `Interface` / `Library`).
///  * `struct_declaration` / `enum_declaration` → types.
///  * `function_definition` / `modifier_definition` — owned by the enclosing
///    contract/interface/library (→ `Method`) or free (→ `Function`).
///
/// Every captured def node exposes a `name:` field whose value is an
/// `identifier`, so the `Capture` strategy (name = `@name`, def = its parent)
/// applies uniformly.
static SOLIDITY_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("contract_declaration", "Contract"),
        DefRule::ty("interface_declaration", "Interface"),
        DefRule::ty("library_declaration", "Library"),
        DefRule::ty("struct_declaration", "Struct"),
        DefRule::ty("enum_declaration", "Enum"),
        DefRule::method("function_definition"),
        DefRule::method("modifier_definition"),
    ],
    owner_kinds: &[
        "contract_declaration",
        "interface_declaration",
        "library_declaration",
    ],
    calls: CallSpec { skip_callees: &[] },
    // Solidity `import` directives are not extracted (no Solidity import
    // strategy exists); import_query is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    docs: DocStyle::CBlockOrLine,
};

/// Each def node carries its name in the `name:` field (an `identifier`).
/// Capture the identifier as `@name`; the engine derives the def node as its
/// parent and keys the DefRule on that parent's kind.
const DEFINITIONS: &str = r#"
    (contract_declaration  name: (identifier) @name) @def
    (interface_declaration name: (identifier) @name) @def
    (library_declaration   name: (identifier) @name) @def
    (struct_declaration    name: (identifier) @name) @def
    (enum_declaration      name: (identifier) @name) @def
    (function_definition   name: (identifier) @name) @def
    (modifier_definition   name: (identifier) @name) @def
"#;

/// A call `foo(...)` parses as `(call_expression function: (expression
/// (identifier "foo")))`; the callee is that inner `identifier`. Member calls
/// (`x.foo()`) wrap the callee in a `member_expression`, so this simple form
/// captures the plain-identifier calls (best-effort).
const CALLS: &str = r#"
    (call_expression
      function: (expression (identifier) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "solidity",
        extensions: &["sol"],
        filenames: &[],
        grammar: || tree_sitter_solidity::LANGUAGE.into(),
        spec: &SOLIDITY_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
