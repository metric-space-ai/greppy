//! Vue (single-file components, `.vue`) — onboarded via the parallel-safe
//! registry (`crate::registry`). This whole file is the entire surface: it
//! declares the spec + queries + grammar and self-registers with
//! `inventory::submit!`. No shared file is edited (build.rs discovers this
//! module automatically); the only Cargo.toml line added is the
//! `tree-sitter-vue-next` dependency.
//!
//! `tree-sitter-vue-next` (v0.1.0) is a crates.io release that builds against
//! tree-sitter 0.25 via the `tree-sitter-language` shim (its `tree-sitter`
//! dependency is a *dev*-dependency only; the library itself depends solely on
//! `tree-sitter-language` 0.1, exactly like `tree-sitter-crystal` /
//! `tree-sitter-purescript`). It is NOT a local path dependency. The grammar
//! accessor is `tree_sitter_vue_next::LANGUAGE` (a `LanguageFn`).
//!
//! Status: **experimental / partial**. A Vue SFC is a markup document, not a
//! programming language: it has no functions and no call expressions, so there
//! is nothing to extract as a `Function`/`Method` and no CALLS or IMPORTS edges
//! are produced (both those queries are intentionally empty — the `<script>`
//! body is an opaque `raw_text` token this grammar does not parse into JS). What
//! the registry *can* surface — and what makes an SFC greppable as structure —
//! are its markup elements, keyed on each element's `tag_name`:
//!
//!   * `start_tag`        — the opening tag of a paired element
//!                          (`<div>…</div>`, and the SFC sections
//!                          `<template>` / `<script>` / `<style>`) → `Element`
//!   * `self_closing_tag` — a self-closing element (`<UserCard … />`) → `Element`
//!
//! The grammar (verified with `examples/dump_vue.rs`) exposes NO `name:` field
//! on any node; each element's name is a `tag_name` child of its opening tag.
//! With the `Capture` name strategy the definition node is therefore the
//! *parent* of the captured `tag_name` — precisely the `start_tag` /
//! `self_closing_tag` node we want. The queries anchor `tag_name` inside the
//! opening-tag nodes only, so an `end_tag`'s `tag_name` (`</div>`) is never
//! captured and elements are not double-counted.
//!
//! Imprecision / honesty:
//!   * A component reference like `<UserCard>` (a definition-site *usage* of a
//!     component) is surfaced as an `Element` named `UserCard`; it is NOT linked
//!     by an edge to any component definition (SFCs live one-per-file and this
//!     grammar does not resolve the `<script>` `export default { name: … }`).
//!   * Directive bindings (`@click="greet"`, `:name="userName"`) are NOT
//!     extracted: they reference `<script>` symbols this grammar leaves as
//!     opaque `raw_text`, so no CALLS/reference edge could be resolved. The
//!     CALLS query is empty (the `Capture`/name-field CALLS source resolution
//!     also cannot apply — Vue nodes carry no `name:` field).
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Vue definitions are its markup elements. None are callable and none are
/// owned (Vue markup has no method/class semantics), so every rule is a
/// `DefRule::ty`. `Capture` sets the def node = the `@name` `tag_name`'s parent,
/// which is precisely the `start_tag` / `self_closing_tag` node keyed here.
static VUE_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("start_tag", "Element"),
        DefRule::ty("self_closing_tag", "Element"),
    ],
    owner_kinds: &[],
    // Vue markup has no call syntax; the CALLS pass is inert (call_query empty).
    calls: CallSpec { skip_callees: &[] },
    // Vue has no import syntax reachable by this grammar (the `<script>` body is
    // opaque `raw_text`); the IMPORTS pass is inert (import_query empty). Any
    // variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // Vue templates use HTML `<!-- -->` comments; none of the DocStyle helpers
    // model them, so no docstrings are extracted.
    docs: DocStyle::None,
};

/// Capture the `tag_name` of each element's opening tag as `@name`; the engine
/// derives the def node as that `tag_name`'s parent (`start_tag` /
/// `self_closing_tag`) and keys the matching `DefRule` on that parent's kind.
///
/// The `tag_name` is anchored INSIDE `start_tag` / `self_closing_tag`, so an
/// `end_tag`'s `tag_name` (`</div>`) is excluded — its parent would be
/// `end_tag`, which has no `DefRule`, so it is dropped. This yields exactly one
/// `Element` per opening tag (paired elements are counted once, at their
/// `start_tag`).
const DEFINITIONS: &str = r#"
    (start_tag        (tag_name) @name)
    (self_closing_tag (tag_name) @name)
"#;

inventory::submit! {
    LangDef {
        name: "vue",
        extensions: &["vue"],
        filenames: &[],
        grammar: || tree_sitter_vue_next::LANGUAGE.into(),
        spec: &VUE_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
