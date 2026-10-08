//! Pascal / Object Pascal (Delphi, Free Pascal) — onboarded via the
//! parallel-safe registry (`crate::registry`). This whole file is the entire
//! surface: it declares the spec + queries + grammar and self-registers with
//! `inventory::submit!`. No shared file is edited (build.rs discovers this
//! module automatically); the only Cargo.toml line added is the
//! `tree-sitter-pascal` dependency.
//!
//! Status: **experimental / partial**. The `tree-sitter-pascal` grammar
//! (0.10.x, built on the `tree-sitter-language` shim so it links against
//! workspace tree-sitter 0.25) models a procedure/function definition as:
//!
//! ```text
//! (defProc
//!    header: (declProc kFunction name: (identifier) args: (declArgs …) …)
//!    body:   (block …))
//! ```
//!
//! The NAME sits on the `header`'s inner `declProc` node (`name:` field, an
//! `identifier`) — NOT on the outer `defProc`. The definition query therefore
//! tags the body-containing `defProc` as `@def` and the header name as `@name`.
//! `declProc` itself is not a DefRule, so the engine keeps `defProc` and the
//! stored span covers the body. `kFunction` and `kProcedure` are just keyword
//! children of the same `declProc` kind, so functions and procedures are
//! captured uniformly (both are labelled `Function`; no return-value distinction
//! is drawn). Forward declarations (`declProc` without a `defProc` body) are
//! not emitted.
//!
//! A call (`exprCall`) lives in `body: (block …)`, which is inside `defProc`.
//! `defProc` has no `name:` field, so the calls pass reuses the qname recorded
//! for that definition node. Qualified/member calls are still not captured.
//!
//! Other imprecision: member/qualified calls (`obj.Method(…)`) and `uses`
//! clauses are not modelled. Not claimed as `supported` (no verification corpus).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions: `defProc` is the body-containing node. The name is read from
/// `header: (declProc name: …)`, which is not itself a DefRule. Both
/// `function`s and `procedure`s become `Function`. No class/record ownership
/// is modelled (kept experimental/partial).
static PASCAL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("defProc")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Pascal `uses` clauses are not extracted yet (import_query is empty); any
    // variant is inert without a query.
    imports: ImportStrategy::Bash,
    // Pascal comments are `{ … }` / `(* … *)` / `//`; the generic doc helpers
    // key on `//` / `#` / `--` line runs, which do not match Pascal's brace/
    // paren block comments cleanly, so docstrings are left off.
    docs: DocStyle::None,
};

/// A `function Foo(...)` / `procedure Bar(...)` parses as
/// `(defProc header: (declProc name: (identifier) @name …) body: (block …))`.
/// Capture the header name and the body-containing `defProc`.
const DEFINITIONS: &str = r#"
    (defProc
      header: (declProc
        name: (identifier) @name)) @def
"#;

/// A call `Foo(...)` parses as `(exprCall entity: (identifier) @callee …)`.
/// Qualified/member calls (`obj.Method(…)`) wrap the entity differently and are
/// not captured (best-effort).
const CALLS: &str = r#"
    (exprCall
      entity: (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "pascal",
        extensions: &["pas", "pp"],
        filenames: &[],
        grammar: || tree_sitter_pascal::LANGUAGE.into(),
        spec: &PASCAL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
