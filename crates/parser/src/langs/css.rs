//! CSS — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-css` dependency (a crates.io release, `0.25`,
//! which builds against tree-sitter 0.25 directly — no git/path shim needed).
//!
//! Status: **experimental / partial**. CSS is a styling/markup language, not a
//! programming language: it has no functions and no user call semantics, so
//! there is nothing to extract as a `Function`/`Method` and no CALLS or IMPORTS
//! edges are produced (both those queries are intentionally empty). What the
//! registry *can* surface — and what makes a CSS file greppable as structure —
//! are its top-level definition nodes:
//!
//!   * `rule_set`            — a `<selectors> { … }` rule            → `Rule`
//!   * `keyframes_statement` — an `@keyframes name { … }` block      → `Keyframes`
//!
//! The grammar (verified with `examples/dump_css.rs`) exposes these node kinds:
//!
//!   stylesheet
//!     rule_set
//!       selectors          ← captured as @name; text is ".button" / "#main"
//!         class_selector / id_selector / …
//!       block { … }
//!     keyframes_statement
//!       @keyframes
//!       keyframes_name     ← captured as @name; text is the animation name
//!       block { … }
//!
//! Neither `rule_set` nor `keyframes_statement` exposes a `name:` field, so the
//! `Capture` name strategy is used: the definition node is the *parent* of the
//! captured node. Capturing the whole `selectors` node yields a def node =
//! `rule_set` whose name is the full selector text (`.button`, `.button:hover`,
//! `#main`), which is exactly the greppable identity of a CSS rule. Capturing
//! `keyframes_name` yields a def node = `keyframes_statement` named by the
//! animation. This is best-effort structural extraction (no golden-master vs C),
//! so it is NOT claimed as `supported`.
//!
//! Imprecision / honesty:
//!   * The rule's `name` is the *entire* selector list, verbatim (e.g.
//!     `.a, .b > c:hover`), not a decomposed set of class/id names. Two rules
//!     with distinct selector text are distinct defs; there is no cross-rule
//!     "reference" edge (a descendant/compound selector like `.button:hover`
//!     shares text with `.button` only lexically, which greppable search over
//!     the selector name already surfaces — no synthetic CALLS edge is made).
//!   * `@media` / `@supports` blocks nest `rule_set`s; those nested rules ARE
//!     captured (the query matches at any depth), but the at-rule wrapper itself
//!     is not surfaced as its own def.
//!   * `@import "…"` is not expanded into IMPORTS edges: no CSS import strategy
//!     exists in `ImportStrategy`, so `import_query` is empty (the IMPORTS pass
//!     is inert).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// CSS definitions are its structural containers. None are callable and none
/// are owned (CSS has no method/class semantics), so every rule is a
/// `DefRule::ty`. `Capture` sets the def node = the `@name` node's parent, which
/// is precisely the `rule_set` / `keyframes_statement` node keyed here.
static CSS_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("rule_set", "Rule"),
        DefRule::ty("keyframes_statement", "Keyframes"),
    ],
    owner_kinds: &[],
    // CSS has no call syntax; the CALLS pass is inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // CSS `@import` is not extracted (no CSS import strategy exists); the
    // IMPORTS pass is inert (import_query is empty). Any variant is dead weight
    // without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // CSS has only `/* … */` block comments; none of the line-comment doc
    // helpers apply, so docstrings are not extracted.
    docs: DocStyle::None,
};

/// Capture the selector list of each rule and the name of each keyframes block
/// as `@name`; the engine derives the def node as that node's parent
/// (`rule_set` / `keyframes_statement`) and keys the matching `DefRule` on that
/// parent's kind.
///
/// The `selectors` node is a *direct* child of `rule_set`, so its `.parent()`
/// is the `rule_set` itself; its text is the full selector list, which is the
/// rule's greppable name. `keyframes_name` is a *direct* child of
/// `keyframes_statement`, so its `.parent()` is that statement; its text is the
/// animation name.
const DEFINITIONS: &str = r#"
    (rule_set (selectors) @name)
    (keyframes_statement (keyframes_name) @name)
"#;

inventory::submit! {
    LangDef {
        name: "css",
        extensions: &["css"],
        filenames: &[],
        grammar: || tree_sitter_css::LANGUAGE.into(),
        spec: &CSS_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
