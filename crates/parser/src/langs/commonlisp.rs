//! Common Lisp — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-commonlisp` dependency.
//!
//! Status: **experimental**. The `tree-sitter-commonlisp` grammar models a
//! `(defun name (args) ...)` form as a `defun` node containing a
//! `defun_header` whose `function_name:` field is a `sym_lit`. The definition
//! query tags the body-containing `defun` as `@def` (the header is not a
//! DefRule), so the Function span covers the body and calls inside it source
//! from that function. Every other Lisp form parses as a `list_lit`, and —
//! following the grammar's own `tags.scm` — a `list_lit` whose first element is
//! a symbol is treated as a call to that symbol. That also treats lambda-list
//! heads and special forms (`let`/`if`) as calls. Best-effort, NOT claimed as
//! `supported` (no verification corpus).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// `(defun f (args) ...)` parses as `(defun (defun_header function_name:
/// (sym_lit) ...))`. The def node is the `defun` (body container), not the
/// header. `defmacro` is a separate node kind and is not emitted.
static COMMONLISP_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("defun")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Common Lisp `require` / `defpackage` / `use-package` imports are not
    // extracted (import_query is empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineDashComment,
};

/// The function name is the `function_name:` field (a `sym_lit`) of a
/// `defun_header`. Capture it as `@name` and the enclosing `defun` as `@def`
/// so the span includes the body forms.
const DEFINITIONS: &str = r#"
    (defun
      (defun_header
        function_name: (sym_lit) @name)) @def
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
