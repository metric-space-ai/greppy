//! SQL — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-sequel` dependency.
//!
//! Status: **experimental / partial**. SQL is a data-definition + query
//! language, not a general-purpose programming language. The
//! `tree-sitter-sequel` grammar (crate `tree-sitter-sequel`, grammar accessor
//! `LANGUAGE`, C symbol `tree_sitter_sql`) models each top-level statement as a
//! `statement` wrapping a concrete node: `create_function`, `create_table`,
//! `create_view`, etc. NONE of these expose a `name:` field — the object's
//! name sits as an `object_reference` child (which itself wraps an
//! `identifier`). With the `Capture` name strategy the definition node is the
//! *parent* of the `@name` capture, so we capture the `object_reference` (whose
//! text is the bare name, e.g. `add_one`) and its parent is exactly the
//! `create_function` / `create_table` / `create_view` node we want.
//!
//! What is surfaced:
//!   * `create_function` → `Function` (the only callable SQL definition; a
//!     stored routine such as a PL/pgSQL function).
//!   * `create_table`    → `Table`.
//!   * `create_view`     → `View`.
//!
//! CALLS: a function invocation inside a routine body parses as an `invocation`
//! whose callee is `(object_reference (identifier))`. We capture that inner
//! `identifier` as `@callee`. IMPORTANT LIMITATION: the generic engine resolves
//! a CALLS edge's *source* endpoint (the enclosing routine) via the callable
//! node's `name:` FIELD (`callable_name` in `spec.rs`), but `create_function`
//! has NO `name:` field — its name lives in an `object_reference` child. So,
//! exactly like Julia (whose `call_expression` also lacks a `name:` field), the
//! enclosing-callable resolution returns `None` and NO CALLS edge is
//! materialised, even though the callee query matches. Fixing this would
//! require editing the shared `spec.rs` engine, which the parallel-safe
//! onboarding surface forbids. Call *definitions* and *tables/views* are
//! extracted correctly; only cross-function CALL edges are absent.
//!
//! Imprecision: SQL is dialect-heavy and this grammar is generic; qualified
//! names (`schema.func`) collapse to their trailing identifier via the
//! `object_reference` text, aggregate/built-in calls (`count(...)`) would be
//! counted as invocations, and no import concept exists (import_query empty).
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// SQL definitions are its top-level `CREATE …` statements. `create_function`
/// is callable (a stored routine, so CALLS edges hang off it); tables and views
/// are non-callable structural types (`DefRule::ty`). `Capture` sets the def
/// node = the `@name` (`object_reference`) node's parent, which is precisely the
/// `create_function` / `create_table` / `create_view` node keyed here.
static SQL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("create_function"),
        DefRule::ty("create_table", "Table"),
        DefRule::ty("create_view", "View"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // SQL has no import syntax; the IMPORTS pass is inert (import_query empty).
    // Any variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // SQL line comments start with `--`; the run-collector handles that marker.
    docs: DocStyle::LineDashComment,
};

/// Each `CREATE` statement carries its object name in an `object_reference`
/// child (`create_function (object_reference (identifier "add_one"))`), not in a
/// `name:` field. Capture the `object_reference` as `@name`: its text is the
/// bare object name and the engine derives the def node as its parent
/// (`create_function` / `create_table` / `create_view`).
const DEFINITIONS: &str = r#"
    (create_function (object_reference) @name)
    (create_table    (object_reference) @name)
    (create_view     (object_reference) @name)
"#;

/// A routine-body call parses as `(invocation (object_reference (identifier)
/// @callee) …)`. Capture the callee identifier; the engine emits a CALLS edge
/// from the enclosing `create_function` routine to that name. Nested
/// invocations (`f(g(x))`) each match separately.
const CALLS: &str = r#"
    (invocation
      (object_reference (identifier) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "sql",
        extensions: &["sql"],
        filenames: &[],
        grammar: || tree_sitter_sequel::LANGUAGE.into(),
        spec: &SQL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
