//! Fortran — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-fortran` dependency.
//!
//! Status: **experimental**. The `tree-sitter-fortran` grammar (0.5.x, built on
//! the `tree-sitter-language` 0.1 shim so it links against the workspace
//! tree-sitter 0.25) models a procedure as a `function` / `subroutine` node
//! whose *header* `function_statement` / `subroutine_statement` carries the
//! `name:` field (a `name` node). The definition query tags the outer
//! procedure as `@def`, so the Function span covers the body. Fortran does not
//! use the generic CALLS pass: `extract_fortran` drops spec edges and attributes
//! each `call_expression` to the enclosing procedure (falling back to the file
//! module when the call is not inside one).
//!
//! Module / derived-type headers likewise carry the name on the header
//! statement (`module_statement` has a `name` child; `derived_type_statement`
//! has a `type_name` child), so those are captured as top-level definitions via
//! `DefRule::ty` keyed on the header statement kind.
//!
//! Imprecision: (1) Fortran does not distinguish a function call from an array
//! subscript syntactically — both parse as `call_expression` — so some CALLS
//! edges may actually be array indexing (best-effort over-count). (2) Module
//! procedures are emitted as free `Function`s, not `Method`s: the `module`
//! node exposes no `name:` field for the ownership resolver, and Fortran module
//! procedures are plain procedures rather than OO methods. (3) `use` module
//! imports are not extracted (no Fortran import strategy exists; import_query
//! is empty). Not claimed as `supported` (no verification corpus).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `module_statement`        — a `module NAME` header          → `Module`
///  * `derived_type_statement`  — a `type :: NAME` header         → `Type`
///  * `function`                — a whole function procedure        → `Function`
///  * `subroutine`              — a whole subroutine procedure      → `Function`
///
/// Procedure names are captured from the header statement; the def node is the
/// outer procedure so the span includes the body. Module and type rules stay
/// on the header statement. No class/method ownership is modelled.
static FORTRAN_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("module_statement", "Module"),
        DefRule::ty("derived_type_statement", "Type"),
        DefRule::func("function"),
        DefRule::func("subroutine"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Fortran `use MODULE` imports are not extracted (no Fortran import strategy
    // exists); import_query is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // Fortran comments start with `!`, for which there is no DocStyle marker
    // (the line-comment-run helpers key on `//` / `#` / `--`); so no docs.
    docs: DocStyle::None,
};

/// `function area(r) result(a)` parses as `(function (function_statement name:
/// (name) @name …))`. Capture the header name and the outer `function` /
/// `subroutine` as `@def` so the span covers the body. A derived type's name
/// sits on a `type_name` child of `derived_type_statement`; a module's on a
/// `name` child of `module_statement`.
const DEFINITIONS: &str = r#"
    (module_statement       (name) @name)
    (derived_type_statement (type_name) @name)
    (function
      (function_statement name: (name) @name)) @def
    (subroutine
      (subroutine_statement name: (name) @name)) @def
"#;

/// `square(r)` / `area(r)` parse as `(call_expression (identifier) @callee
/// (argument_list …))`; the callee is the leading `identifier`. Fortran does not
/// distinguish a call from an array subscript syntactically, so this also
/// captures array indexing (best-effort over-count).
const CALLS: &str = r#"
    (call_expression
      (identifier) @callee
      (argument_list))
"#;

inventory::submit! {
    LangDef {
        name: "fortran",
        extensions: &["f90", "f95", "f"],
        filenames: &[],
        grammar: || tree_sitter_fortran::LANGUAGE.into(),
        spec: &FORTRAN_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
