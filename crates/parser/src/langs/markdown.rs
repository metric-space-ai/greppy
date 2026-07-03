//! Markdown — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-md` dependency (a crates.io release that builds
//! against tree-sitter 0.25 via the `tree-sitter-language` 0.1 shim, exactly
//! like the other grammars in this crate). The block grammar accessor is
//! `tree_sitter_md::LANGUAGE` (the `tree_sitter_markdown` block parser; the
//! crate also exposes a separate `INLINE_LANGUAGE` inline parser we do not use).
//!
//! Status: **experimental / partial**. Markdown is a *markup* language: it has
//! no functions and no call expressions, so there is nothing to extract as a
//! `Function` / `Method` and no CALLS or IMPORTS edges are produced (both those
//! queries are intentionally empty). What the registry *can* surface — and what
//! makes a Markdown file greppable as structure — are its top-level definition
//! nodes: its section headings. Both heading syntaxes are captured (verified
//! with `examples/dump_md.rs` against the real grammar):
//!
//!   * `atx_heading`    — `# Title`, `## add`, `### deep`             → `Heading`
//!   * `setext_heading` — `Title` underlined by `===` / `---`         → `Heading`
//!
//! The grammar carries the heading text on the `heading_content:` field: an
//! `inline` node for an `atx_heading`, and a `paragraph` (wrapping an `inline`)
//! for a `setext_heading`. Neither heading node exposes a `name:` field, so with
//! the `Capture` name strategy the definition node is the *parent* of the
//! captured `heading_content` child — precisely the `atx_heading` /
//! `setext_heading` node we want — and the captured child's text is the heading
//! title (the section name).
//!
//! Imprecision / honesty:
//!   * The block grammar does NOT parse inline links (`[text](#target)`): inside
//!     a heading/paragraph they render as separate punctuation tokens, not a
//!     `link`/`link_destination` node (that only happens under the separate
//!     `INLINE_LANGUAGE` parser). So Markdown cross-references are NOT modelled
//!     as CALLS edges; `call_query` is empty and the CALLS pass is inert.
//!   * A `setext_heading`'s captured name is the underlying `paragraph`'s text,
//!     which may carry a trailing newline; it is not trimmed here.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Markdown definitions are its section headings. Neither is callable and
/// neither is owned (Markdown has no method/class semantics), so every rule is a
/// `DefRule::ty`. `Capture` sets the def node = the captured `heading_content`
/// child's parent, which is precisely the `atx_heading` / `setext_heading` node
/// keyed here.
static MARKDOWN_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    // C-reference parity: the C markdown extractor labels every atx/setext
    // heading a `Section` (extract_defs.c:2951-2953, CBM_LANG_MARKDOWN), not a
    // `Heading`. Same nodes, C's label — closes the golden-master `Section`
    // (grepplus-missing) and `Heading` (grepplus-only) rows on every fixture.
    defs: &[
        DefRule::ty("atx_heading", "Section"),
        DefRule::ty("setext_heading", "Section"),
    ],
    owner_kinds: &[],
    // Markdown has no call syntax (the block grammar does not parse inline
    // links); the CALLS pass is inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // Markdown has no import syntax; the IMPORTS pass is inert (import_query is
    // empty). Any variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // Markdown has no code-doc comment convention the doc helpers recognise
    // (`<!-- -->` HTML comments are not `//`/`#`/`--` line runs); no docs.
    docs: DocStyle::None,
};

/// Capture the `heading_content` of each heading as `@name`; the engine derives
/// the def node as that child's parent (`atx_heading` / `setext_heading`) and
/// keys the matching `DefRule::ty` on that parent's kind. An `atx_heading`'s
/// content is an `inline`; a `setext_heading`'s content is a `paragraph`. In
/// both cases the captured node is a *direct* child of the heading, so its
/// `.parent()` is the heading itself.
const DEFINITIONS: &str = r#"
    (atx_heading    heading_content: (inline)    @name)
    (setext_heading heading_content: (paragraph) @name)
"#;

inventory::submit! {
    LangDef {
        name: "markdown",
        extensions: &["md"],
        filenames: &[],
        grammar: || tree_sitter_md::LANGUAGE.into(),
        spec: &MARKDOWN_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
