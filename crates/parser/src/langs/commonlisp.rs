//! Common Lisp — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-commonlisp` dependency.
//!
//! Status: **experimental**. The `tree-sitter-commonlisp` grammar models a
//! `(defun name (args) ...)` form as a `defun` node containing a
//! `defun_header` whose `function_name:` field is a `sym_lit`. With the
//! `Capture` name strategy the definition node is therefore the `defun_header`
//! (the parent of the `@name` `sym_lit`), which is enough to emit Function
//! nodes. Every other Lisp form parses as a `list_lit`, and — following the
//! grammar's own `tags.scm` — a `list_lit` whose first element is a symbol is
//! treated as a call to that symbol. Because `defun_header` exposes the name
//! under a `function_name:` field rather than the `name:` field the engine's
//! enclosing-callable resolver consults, CALLS edges whose *source* is a Lisp
//! function are NOT resolved (same limitation as Julia). Call extraction is
//! also structural/heuristic (it cannot distinguish a real function call from a
//! macro/special-form head such as `let`/`if`), so it is best-effort and NOT
//! claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// `(defun f (args) ...)` parses as `(defun (defun_header function_name:
/// (sym_lit) ...))`. Capturing the `sym_lit` as `@name` makes its parent —
/// the `defun_header` — the definition node, so the rule keys on
/// `"defun_header"`. `defmacro` / `defun`-family headers all share this node
/// kind, so a single `DefRule::func("defun_header")` covers them.
static COMMONLISP_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("defun_header")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Common Lisp `require` / `defpackage` / `use-package` imports are not
    // extracted (import_query is empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineDashComment,
};

/// The function name is the `function_name:` field (a `sym_lit`) of a
/// `defun_header`. Capture it as `@name`; the engine derives the def node as
/// its parent (`defun_header`) and keys the DefRule on that kind.
const DEFINITIONS: &str = r#"
    (defun_header
      function_name: (sym_lit) @name) @def
"#;

/// Following the grammar's own `tags.scm`: a `list_lit` whose FIRST element is
/// a `sym_lit` is a call to that symbol. The anchor `.` pins the capture to the
/// list's first named child (the operator position), so argument symbols are
/// not mistaken for callees. This also matches macro/special-form heads
/// (`let`, `if`, …) — an accepted imprecision for this heuristic grammar.
const CALLS: &str = r#"
    (list_lit
      .
      (sym_lit) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "commonlisp",
        extensions: &["lisp", "cl"],
        filenames: &[],
        grammar: || tree_sitter_commonlisp::LANGUAGE_COMMONLISP.into(),
        spec: &COMMONLISP_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
