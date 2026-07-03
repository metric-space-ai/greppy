//! Racket — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-racket` dependency.
//!
//! Status: **experimental**. The `tree-sitter-racket` grammar is a generic
//! s-expression grammar: it has NO `function_definition` / `call` node kinds —
//! *every* parenthesised form is a `list` and every atom is a `symbol`. There is
//! no `name:` field anywhere. Definitions are therefore recognised purely by
//! shape: a `list` whose first element is the `symbol` `define` (also `define/…`
//! macros such as `define/public`, `define-values`, `define-syntax`). Two
//! define shapes exist:
//!
//!   * function form  `(define (f a b) body)` — the name is the first `symbol`
//!     of the *inner* `list` (the `(f a b)` signature). With the `Capture`
//!     strategy the def node becomes that inner signature `list` (its parent),
//!     which is enough to emit a Function node but means the reported line-span
//!     is the signature list, not the whole `define` form.
//!   * value form     `(define x expr)`       — the name is a bare `symbol`
//!     whose parent is the outer `define` `list`; the whole form is the def.
//!
//! Because no def node exposes a `name:` field, the engine's enclosing-callable
//! resolution (which reads `child_by_field_name("name")`) cannot name a Racket
//! function, so CALLS edges whose *source* is a Racket define are NOT resolved
//! (identical to the Julia limitation). Definition extraction still works.
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Both define shapes route their name `symbol`'s parent to a `list`
/// (the inner signature list for the function form, the outer define list for
/// the value form), so a single `DefRule::func("list")` covers both. The
/// DEFINITIONS query only ever matches a `symbol` that is the name position of a
/// `define` form, so ordinary call `list`s never reach the def pass — no
/// spurious defs leak despite `list` being the universal node kind.
static RACKET_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("list")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Racket imports (`require`) are not extracted yet (import_query is empty);
    // any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// A `define` form is a `list` whose first `symbol` is a `define`-family keyword.
///
///   * function form  `(define (f a b) …)` — capture the first `symbol` of the
///     inner signature `list` as `@name`; its parent is that inner `list` (the
///     `@def`).
///   * value form     `(define x …)`       — capture the bare `symbol` after the
///     keyword as `@name`; its parent is the outer `define` `list` (the `@def`).
///
/// The keyword is matched with `#any-of?` against the common `define` variants
/// so `define`, `define/public`, `define-values`, `define-syntax`, … all count.
const DEFINITIONS: &str = r#"
    (list
      .
      (symbol) @_kw
      .
      (list . (symbol) @name)
      (#any-of? @_kw
        "define" "define*" "define/public" "define/private" "define/override"
        "define/contract" "define/match" "define-values" "define-syntax"
        "define-syntax-rule" "define-for-syntax" "define/augment")) @def

    (list
      .
      (symbol) @_kw
      .
      (symbol) @name
      (#any-of? @_kw
        "define" "define*" "define/public" "define/private" "define/override"
        "define/contract" "define/match" "define-values" "define-syntax"
        "define-syntax-rule" "define-for-syntax" "define/augment")) @def
"#;

/// Every application `(f a b)` is a `list` whose first `symbol` is the callee.
/// This also matches the head `symbol` of `define` forms and other special
/// forms (`if`, `let`, …); those are harmless noise for a call graph, and the
/// name-position `symbol`s of definitions are self-references whose enclosing
/// source cannot be resolved for Racket anyway (no edge is materialised).
const CALLS: &str = r#"
    (list
      .
      (symbol) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "racket",
        extensions: &["rkt"],
        filenames: &[],
        grammar: || tree_sitter_racket::LANGUAGE.into(),
        spec: &RACKET_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
