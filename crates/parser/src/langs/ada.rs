//! Ada — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-ada` dependency.
//!
//! Status: **experimental / partial**. The `tree-sitter-ada` grammar (briot's
//! grammar, exposed through the `tree-sitter-language` 0.1 shim so it builds
//! against the workspace's tree-sitter 0.25) models an Ada subprogram as a
//! `subprogram_body` wrapping a `function_specification` or
//! `procedure_specification`. The subprogram's name is a `name:` field (an
//! `identifier`) on that *specification*, NOT on the enclosing
//! `subprogram_body`. The definition query tags that body as `@def` and reads
//! the name from the nested specification. The specification kinds are not
//! DefRules, so the Function span is the whole `subprogram_body` (including
//! `begin` … `end`), which is what `read` returns. Spec-only declarations
//! without a body are not emitted.
//!
//! Package units carry their name directly: `package_declaration` (a package
//! spec) and `package_body` both expose a `name:` `identifier`, so capturing it
//! makes the def node the package node. A named `full_type_declaration`
//! (`type Color is …`) carries the type name as its first plain `identifier`
//! child (the `type` keyword's next sibling), so capturing that direct child
//! makes the def node the `full_type_declaration` (enum members live nested
//! inside `enumeration_type_definition`, so they are never captured).
//!
//! Calls parse as `function_call` (in an expression) or
//! `procedure_call_statement`, each with a `name:` field naming the callee.
//! The body is inside `subprogram_body`, which has no `name:` field (`endname`
//! is not `name`), so the calls pass reuses the qname recorded for that
//! definition. Imprecision: the callee `name:` can itself be a dotted
//! `selected_component` for a qualified call (`Pkg.Op(…)`); this query captures
//! only the plain `identifier` callee form (best-effort). Ada `with` clauses
//! (imports) are not extracted (no Ada import strategy exists; import_query is
//! empty). Not claimed as `supported` (no verification corpus).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `subprogram_body` — a function or procedure body → Function. The name is
///    the nested specification's `name:` identifier.
///  * `package_declaration` / `package_body` — the parent of a package's
///    `name:` identifier → Package.
///  * `full_type_declaration` — the parent of a type's plain `identifier` child
///    → Type.
///
/// Packages and types use the `Capture` strategy (def node = the captured
/// name's parent). Subprograms use the `@def` body container. No ownership is
/// modelled (Ada nested subprograms/methods are not distinguished), so
/// subprograms are always free Functions.
static ADA_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("subprogram_body"),
        DefRule::ty("package_declaration", "Package"),
        DefRule::ty("package_body", "Package"),
        DefRule::ty("full_type_declaration", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Ada `with` clauses are not extracted yet (import_query is empty); any
    // variant is inert without a query.
    imports: ImportStrategy::Bash,
    // Ada comments start with `--`.
    docs: DocStyle::LineDashComment,
};

/// Capture the name identifier of each definition; the engine derives the def
/// node as that identifier's parent and keys the DefRule on that parent's kind.
///
///  * `function_specification` / `procedure_specification` expose `name:`.
///  * `package_declaration` / `package_body` expose `name:`.
///  * `full_type_declaration` carries the type name as its first plain
///    `identifier` child (anchored so the query matches only that direct child,
///    not the enum-member identifiers nested inside the type definition).
const DEFINITIONS: &str = r#"
    (subprogram_body
      (function_specification name: (identifier) @name)) @def
    (subprogram_body
      (procedure_specification name: (identifier) @name)) @def
    (package_declaration     name: (identifier) @name)
    (package_body            name: (identifier) @name)
    (full_type_declaration . (identifier) @name)
"#;

/// A call `Compute(…)` parses as `(function_call name: (identifier) @callee …)`;
/// a bare `Do_Thing(…);` statement parses as `(procedure_call_statement name:
/// (identifier) @callee …)`. Qualified callees (`Pkg.Op`) sit in a
/// `selected_component` rather than a plain `identifier`, so this simple form
/// captures the unqualified callee (best-effort).
const CALLS: &str = r#"
    (function_call          name: (identifier) @callee)
    (procedure_call_statement name: (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "ada",
        extensions: &["adb", "ads"],
        filenames: &[],
        grammar: || tree_sitter_ada::LANGUAGE.into(),
        spec: &ADA_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
