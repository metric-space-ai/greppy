//! GLSL — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-glsl` dependency (a crates.io release, v0.2.0,
//! which builds against tree-sitter 0.25 via the `tree-sitter-language` shim
//! — its accessor is the `LANGUAGE_GLSL` `LanguageFn` constant).
//!
//! Status: **experimental / partial**. GLSL (the OpenGL Shading Language) is a
//! C-derived shading language: the `tree-sitter-glsl` grammar is forked from
//! `tree-sitter-c`, so its definition nodes are exactly the C shapes. Verified
//! with `examples/dump_glsl.rs`, the grammar exposes:
//!
//!   * `function_definition` — `vec3 shade(...) { ... }`. The name is nested
//!     inside `declarator: (function_declarator declarator: (identifier))`
//!     (possibly behind pointer/array declarators), NOT on a `name:` field, so
//!     the `CStructural` name strategy (which walks the declarator) applies —
//!     identical to C.                                       → `Function`
//!   * `struct_specifier` — `struct Light { ... };`, name on the `name:` field
//!     (a `type_identifier`).                                → `Struct`
//!   * `type_definition` — `typedef` (GLSL rarely uses it, but the C grammar
//!     accepts it); name resolved structurally off the declarator. → `Type`
//!
//! Ownership: GLSL has no classes/methods, so `owner_kinds` is empty and every
//! `function_definition` resolves to a free `Function` (the `Owner::CppClass`
//! rule falls through to the free branch when no enclosing class/qualifier is
//! found — exactly as for a free C function).
//!
//! CALLS: a call site parses as `(call_expression function: (identifier)
//! @callee ...)`, e.g. `attenuation(d)` inside `shade`. The engine hangs the
//! CALLS edge off the enclosing `function_definition` (whose name resolves
//! structurally, so the source endpoint resolves). A swizzle/member access
//! (`light.color`) is a `field_expression`, not a call, so it is not captured.
//!
//! Imprecision / honesty:
//!   * Built-in intrinsics (`length`, `normalize`, `dot`, `mix`, …) are
//!     captured as CALLS callees like any user function — there is no built-in
//!     symbol table to filter them, so the edge targets a `Function::length`
//!     qname that will simply never resolve to a local definition. This is
//!     best-effort (same behaviour as C calling libc functions).
//!   * GLSL `#include` (an ARB/EXT extension, not core GLSL) is not expanded
//!     into IMPORTS edges: `import_query` is empty so the IMPORTS pass is inert.
//!   * Only top-level `function_definition` / `struct_specifier` /
//!     `type_definition` are surfaced; uniforms, `in`/`out` globals, and
//!     interface blocks are not captured as definitions.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy, Owner};

/// Definitions mirror the C spec (GLSL's grammar is a C fork):
///  * `function_definition` → `Function` (free; GLSL has no methods, so the
///    `Owner::CppClass` rule falls through to the free branch).
///  * `struct_specifier` → `Struct`.
///  * `type_definition`  → `Type`.
///
/// Names are resolved structurally (`NameStrategy::CStructural`): a function's
/// name is nested inside its `function_declarator`, and a tagged type's name is
/// the `name:` `type_identifier` on the specifier.
static GLSL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::CStructural,
    defs: &[
        DefRule {
            node_kind: "function_definition",
            label: "Function",
            method_label: "Method",
            owner: Owner::CppClass,
            callable: true,
        },
        DefRule::ty("struct_specifier", "Struct"),
        DefRule::ty("type_definition", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // GLSL `#include` is a non-core extension; no GLSL import strategy exists so
    // `import_query` is empty and the IMPORTS pass is inert (variant arbitrary).
    imports: ImportStrategy::Bash,
    // GLSL uses C-style `/* */` block and `//` line comments.
    docs: DocStyle::CBlockOrLine,
};

/// Capture the whole definition node as `@def`; the `CStructural` extractor
/// walks the declarator (functions) or reads the `name:` field (structs) to
/// resolve the name. A `struct_specifier` is only captured when it carries a
/// `name:` (an anonymous struct in a declaration has no def name).
const DEFINITIONS: &str = r#"
    (function_definition) @def

    (struct_specifier
        name: (type_identifier)) @def

    (type_definition) @def
"#;

/// A call site parses as `(call_expression function: (identifier) @callee ...)`.
/// Keying on the `function:` `identifier` captures user-function and built-in
/// intrinsic calls alike; the engine hangs the CALLS edge off the enclosing
/// `function_definition` (whose name resolves structurally).
const CALLS: &str = r#"
    (call_expression
        function: (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "glsl",
        extensions: &["glsl", "frag", "vert"],
        filenames: &[],
        grammar: || tree_sitter_glsl::LANGUAGE_GLSL.into(),
        spec: &GLSL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
