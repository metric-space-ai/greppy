//! Svelte — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-svelte-ng` dependency (a crates.io release —
//! v1.0.2 — which builds against tree-sitter 0.25 via the
//! `tree-sitter-language` shim, exactly like `tree-sitter-purescript` /
//! `tree-sitter-crystal`; its grammar accessor is `LANGUAGE`).
//!
//! Status: **experimental / partial**. Svelte is a *markup* language: the
//! grammar (`tree-sitter-svelte-ng`, verified with `examples/dump_svelte.rs`)
//! parses the component's **template**, but treats the `<script>` and `<style>`
//! bodies as opaque `raw_text` — the JS/TS inside a `<script>` block is NOT
//! parsed, so there are no `function` / `class` nodes to surface and no call
//! expressions to extract from component logic. What the registry *can* surface
//! — and what makes a `.svelte` file greppable as structure — are its template
//! definition nodes:
//!
//!   * `snippet_start` — the header of a reusable `{#snippet name(args)}` block,
//!                       whose name is a direct `snippet_name` child   → `Snippet`
//!   * `start_tag`     — an element's opening tag `<button …>`, whose name is a
//!                       direct `tag_name` child                       → `Element`
//!   * `self_closing_tag` — an element `<Foo … />`, `tag_name` child   → `Element`
//!
//! None of these nodes expose a `name:` field (the grammar uses no field names);
//! the name sits as an anonymous `snippet_name` / `tag_name` child. With the
//! `Capture` name strategy the definition node is therefore the *parent* of the
//! captured name — exactly the `snippet_start` / `start_tag` / `self_closing_tag`
//! node we want — so a single name capture per container yields the right def
//! node and name.
//!
//! Imprecision / honesty:
//!   * A `{#snippet row(x)}…{/snippet}` reusable block is referenced elsewhere by
//!     `{@render row(item)}`; that reference parses as `render_tag > svelte_raw_text
//!     "row(item)"`, i.e. the callee name is embedded in an *opaque* `raw_text`
//!     token (name + args, not a separate identifier node). There is no clean
//!     `@callee` identifier to capture, so no CALLS edges are produced (the CALLS
//!     pass is intentionally empty). The snippet-definition/`@render` relationship
//!     is therefore not resolved.
//!   * `<script>` / `<style>` bodies are opaque `raw_text`: functions, imports,
//!     and calls written in the embedded JS/TS/CSS are not extracted. Svelte has
//!     no import strategy in `ImportStrategy`, so `import_query` is empty (the
//!     IMPORTS pass is inert).
//!   * Every `<script>` / `<style>` block also carries a `start_tag` with a
//!     `tag_name` (`script` / `style`), so those tags are surfaced as `Element`
//!     defs alongside real markup elements — best-effort structural extraction.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Svelte definitions are its template structural containers. None are callable
/// and none are owned (Svelte templates have no method/class semantics), so
/// every rule is a `DefRule::ty`. `Capture` sets the def node = the `@name`
/// name-node's parent, which is precisely the `snippet_start` / `start_tag` /
/// `self_closing_tag` node keyed here.
static SVELTE_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("snippet_start", "Snippet"),
        DefRule::ty("start_tag", "Element"),
        DefRule::ty("self_closing_tag", "Element"),
    ],
    owner_kinds: &[],
    // Svelte `{@render name(args)}` references embed the callee in an opaque
    // `raw_text` token (no `@callee` identifier node), and `<script>` bodies are
    // not parsed, so the CALLS pass is inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // Svelte has no import syntax the grammar exposes (script bodies are opaque);
    // the IMPORTS pass is inert (import_query is empty). Any variant is dead
    // weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // Svelte template comments are HTML `<!-- -->`; the grammar names them
    // `comment` but they are not a leading-line-run style, so no docstrings are
    // extracted in practice.
    docs: DocStyle::None,
};

/// Capture the name of each structural container as `@name`; the engine derives
/// the def node as that name's parent (`snippet_start` / `start_tag` /
/// `self_closing_tag`) and keys the matching `DefRule` on that parent's kind.
///
/// A snippet header's name is a `snippet_name` child of `snippet_start`
/// (`{#snippet row(x)}`); an element's name is a `tag_name` child of its
/// `start_tag` (`<button …>`) or `self_closing_tag` (`<Foo … />`). In every case
/// the captured name is a *direct* child of the container, so its `.parent()` is
/// the container itself.
const DEFINITIONS: &str = r#"
    (snippet_start (snippet_name) @name)
    (start_tag (tag_name) @name)
    (self_closing_tag (tag_name) @name)
"#;

inventory::submit! {
    LangDef {
        name: "svelte",
        extensions: &["svelte"],
        filenames: &[],
        grammar: || tree_sitter_svelte_ng::LANGUAGE.into(),
        spec: &SVELTE_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
