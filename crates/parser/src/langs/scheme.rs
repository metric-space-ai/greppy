//! Scheme — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-scheme` dependency (`0.24.7`, which builds against
//! tree-sitter 0.25 through the `tree-sitter-language` shim — the same shim
//! `tree-sitter-racket 0.24.7` already uses in this workspace).
//!
//! Status: **experimental / partial**. Scheme's tree-sitter grammar is
//! deliberately *homogeneous* (verified by dumping the tree-sitter parse of a
//! sample against `tree_sitter_scheme::LANGUAGE`): every parenthesised form is a
//! single `list` node whose children are `symbol` /
//! `number` / `string` / nested `list` atoms. There is NO distinct `define`
//! node, no `function_definition` kind, and no `name:` field anywhere. A
//! definition therefore has to be recognised structurally by matching the
//! literal `define` keyword symbol at the head of a `list`:
//!
//!   * `(define (f a b) …)`  — a procedure definition. The name `f` is the head
//!     `symbol` of the inner *parameter* `list`; its parent is that inner
//!     `list`, so with the `Capture` strategy the def node is the parameter
//!     `list` (its span covers the signature).
//!   * `(define x 42)`       — a value binding. The name `x` is the `symbol`
//!     directly after the `define` keyword; its parent is the outer `list`, so
//!     the def node is the whole `(define x 42)` form.
//!
//! Because the def node kind is `list` in BOTH cases, a single
//! `DefRule::ty("list", "Define")` keys them (the query is what restricts the
//! match to real `define` heads — a bare `@name` whose parent is any other
//! `list` is never captured). Names come out correct (`square`,
//! `sum-of-squares`, `pi`, …).
//!
//! Honesty / imprecision:
//!   * Scheme is treated as a **structural / config-style** language (like TOML
//!     here): definitions are surfaced as `Define` nodes via `DefRule::ty`, and
//!     NO CALLS or IMPORTS edges are produced (both those queries are empty).
//!     This is a deliberate, honest choice, not an oversight: the grammar
//!     exposes no `name:` field on any node, so the engine's enclosing-callable
//!     resolution (which reads `child_by_field_name("name")`) cannot resolve a
//!     Scheme procedure as a CALLS *source* — every CALLS edge would be dropped
//!     unresolved. Rather than emit a call query whose edges can never land, the
//!     CALLS pass is left inert.
//!   * A procedure definition's def-node span is the parameter `list` (the
//!     signature), not the entire `(define …)` form; a value binding's span is
//!     the full form. Both point at the definition site.
//!   * `define-syntax`, `let`-bound locals, and other binding forms are not
//!     captured — only top-level (or nested) `(define …)` forms.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Scheme definitions are `(define …)` forms, surfaced structurally as `Define`
/// nodes. Both the procedure and value-binding patterns land on a `list` def
/// node (the `@name` symbol's parent), so a single `DefRule::ty("list",
/// "Define")` keys them. None are callable and none are owned (Scheme exposes no
/// `name:` field, so CALLS cannot resolve a Scheme source — the CALLS pass is
/// left inert with an empty `call_query`).
static SCHEME_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::ty("list", "Define")],
    owner_kinds: &[],
    // Scheme has no resolvable call source (no `name:` field); the CALLS pass is
    // inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // Scheme `(import …)` / `(require …)` are not extracted (no Scheme import
    // strategy exists); `import_query` is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // Scheme line comments start with `;`, for which there is no DocStyle marker
    // (the line-comment-run helpers key on `//` / `#` / `--`); so no docs.
    docs: DocStyle::None,
};

/// Two structural shapes of a `(define …)` form, both keyed on the literal
/// `define` head symbol:
///
///   * Procedure: `(define (name args…) body)` — the name is the head `symbol`
///     of the inner parameter `list`. Its `.parent()` is that inner `list`, so
///     the `Capture` strategy makes the def node the parameter `list`.
///   * Value:     `(define name value)` — the name is the `symbol` immediately
///     after the `define` keyword. Its `.parent()` is the outer `list`, so the
///     def node is the whole `(define …)` form.
///
/// Anchors (`.`) pin the `define` keyword to the first symbol after the opening
/// `"("`, and pin each `@name` to the head of its containing list, so no nested
/// atom is mistaken for a definition name.
const DEFINITIONS: &str = r#"
    (list
      .
      "("
      .
      (symbol) @_kw
      .
      (list . "(" . (symbol) @name)
      (#eq? @_kw "define"))

    (list
      .
      "("
      .
      (symbol) @_kw
      .
      (symbol) @name
      (#eq? @_kw "define"))
"#;

inventory::submit! {
    LangDef {
        name: "scheme",
        extensions: &["scm", "ss"],
        filenames: &[],
        grammar: || tree_sitter_scheme::LANGUAGE.into(),
        spec: &SCHEME_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
