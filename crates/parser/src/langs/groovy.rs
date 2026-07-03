//! Groovy — onboarded via the parallel-safe registry (`crate::registry`).
//! The whole surface is this one file: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-groovy` dependency.
//!
//! Status: **experimental**. The `tree-sitter-groovy` grammar is Java-derived,
//! so it models Groovy with Java-style node kinds: top-level scripts use
//! `function_definition`, class/interface members use `method_declaration`, and
//! calls are `method_invocation`. This works well for explicitly-parenthesised
//! Groovy (which is the vast majority of `.gradle` / typed `.groovy` code), but
//! Groovy's paren-less command syntax (`task foo`, `println bar`) is not always
//! parsed as a call by this grammar, so call extraction is best-effort. Not
//! claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `function_definition` — top-level (script) functions → `Function`.
///  * `method_declaration` / `constructor_declaration` — members owned by the
///    enclosing class/interface/enum → `Method`.
///  * `class_declaration` / `interface_declaration` / `enum_declaration` → types.
///
/// All def nodes expose a `name:` field whose value is an `identifier`, so the
/// `Capture` strategy (name = `@name`, def = its parent) applies uniformly.
static GROOVY_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_declaration", "Class"),
        DefRule::ty("interface_declaration", "Interface"),
        DefRule::ty("enum_declaration", "Enum"),
        DefRule::func("function_definition"),
        DefRule::method("method_declaration"),
        DefRule::method("constructor_declaration"),
    ],
    owner_kinds: &[
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
    ],
    calls: CallSpec { skip_callees: &[] },
    // Groovy imports mirror Java's `import_declaration` grammar exactly, so the
    // Java import expander applies (walks the `scoped_identifier`).
    imports: ImportStrategy::Java,
    docs: DocStyle::CBlockOrLine,
};

/// Each def node carries its name in the `name:` field (an `identifier`).
/// Capture the identifier as `@name`; the engine derives the def node as its
/// parent and keys the DefRule on that parent's kind.
const DEFINITIONS: &str = r#"
    (function_definition   name: (identifier) @name) @def
    (method_declaration    name: (identifier) @name) @def
    (constructor_declaration name: (identifier) @name) @def
    (class_declaration     name: (identifier) @name) @def
    (interface_declaration name: (identifier) @name) @def
    (enum_declaration      name: (identifier) @name) @def
"#;

/// Calls parse as `method_invocation` with the callee in the `name:` field.
const CALLS: &str = r#"
    (method_invocation name: (identifier) @callee)
"#;

/// `import java.util.List` / `import static java.lang.Math.max` /
/// `import java.util.*` — one `import_declaration` per statement.
const IMPORTS: &str = r#"
    (import_declaration) @import
"#;

inventory::submit! {
    LangDef {
        name: "groovy",
        extensions: &["groovy", "gradle"],
        filenames: &[],
        grammar: || tree_sitter_groovy::LANGUAGE.into(),
        spec: &GROOVY_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: IMPORTS,
    }
}
