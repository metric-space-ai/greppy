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
//! `identifier`) — NOT on the outer `defProc`. With the `Capture` name strategy
//! the definition node is the `@name` identifier's PARENT, which is the
//! `declProc` header. So the `DefRule` keys on `"declProc"` and one Function is
//! emitted per procedure/function/method. `kFunction` and `kProcedure` are just
//! keyword children of the same `declProc` kind, so functions and procedures are
//! captured uniformly (both are labelled `Function`; no return-value distinction
//! is drawn).
//!
//! IMPRECISION — CALLS edges are NOT attributed (edge count is 0). A call
//! (`exprCall`) lives inside the callable's BODY (`body: (block …)`), which is a
//! SIBLING of the name-bearing `header` (`declProc`) under the wrapping
//! `defProc`. The generic engine attributes a CALLS edge's *source* to the
//! nearest def-rule ancestor of the call site whose name it can resolve; here
//! that ancestor is `defProc`, whose name lives one level down on `header` and
//! is therefore NOT reachable via the engine's `child_by_field_name("name")`
//! lookup. Keying defs on `declProc` (correct for definition extraction) means
//! `declProc` is never an ancestor of a call, so no enclosing callable is found
//! and no CALLS edge is emitted. The CALLS query below is retained (it correctly
//! identifies the callee identifier of `exprCall`), but with this grammar's
//! header/body split the generic engine cannot hang a source endpoint off it
//! without a bespoke extractor. Definition extraction (the node pass) is
//! unaffected and complete for plain `function`/`procedure` definitions.
//!
//! Other imprecision: member/qualified calls (`obj.Method(…)`) and `uses`
//! clauses are not modelled. Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions: `declProc` is the header node that carries the `name:` field
/// (`Capture` → def node = the name identifier's parent = `declProc`). Both
/// `function`s and `procedure`s parse as `declProc`, so both become `Function`.
/// No class/record ownership is modelled (kept experimental/partial).
static PASCAL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("declProc")],
    owner_kinds: &[],
    // The CALLS query captures callee identifiers, but with this grammar's
    // header/body split the generic engine cannot resolve a call's enclosing
    // callable (see module docs), so 0 CALLS edges are emitted in practice.
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
/// Capture the header's `name:` identifier; the engine derives the def node as
/// its parent `declProc` and keys `DefRule::func("declProc")` on it.
const DEFINITIONS: &str = r#"
    (declProc
      name: (identifier) @name) @def
"#;

/// A call `Foo(...)` parses as `(exprCall entity: (identifier) @callee …)`.
/// The capture is correct, but the generic engine cannot attribute the call to
/// its enclosing callable for this grammar (see module docs), so no CALLS edge
/// is materialised. Qualified/member calls (`obj.Method(…)`) wrap the entity
/// differently and are not captured (best-effort).
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
