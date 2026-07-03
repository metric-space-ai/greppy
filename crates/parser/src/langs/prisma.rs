//! Prisma — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-prisma-io` dependency (a crates.io release, v1.6.0,
//! which builds against tree-sitter 0.25 via the `tree-sitter-language` shim,
//! exactly like the `LANGUAGE`-constant grammars — verified: the crate pins
//! `tree-sitter-language = "0.1"` and its accessor is
//! `tree_sitter_prisma_io::LANGUAGE`).
//!
//! Status: **experimental / partial**. Prisma schema (`.prisma`) is a schema /
//! configuration DSL, not a programming language: it has no functions and no
//! call expressions in the "one function invokes another" sense, so nothing is
//! extracted as a `Function`/`Method` and no CALLS or IMPORTS edges are produced
//! (both those queries are intentionally empty). What the registry *can* surface
//! — and what makes a schema greppable as structure — are its top-level
//! declaration nodes, verified with `examples/dump_prisma.rs`:
//!
//!   * `model_declaration`      — `model User { … }`      → `Model`
//!   * `enum_declaration`        — `enum Role { … }`        → `Enum`
//!   * `datasource_declaration`  — `datasource db { … }`    → `Datasource`
//!   * `generator_declaration`   — `generator client { … }` → `Generator`
//!   * `type_declaration`        — `type Address { … }`     → `Type`
//!
//! The grammar does NOT expose a `name:` field on any of these; the declaration
//! name is an anonymous `identifier` sitting as a *direct* child of the
//! declaration node, immediately after the leading keyword (`model` / `enum` /
//! …). With the `Capture` name strategy the definition node is the *parent* of
//! the captured `identifier` — exactly the `*_declaration` node we want — and
//! anchoring the identifier as the FIRST `identifier` under each declaration
//! (`. (identifier) @name`, skipping the keyword token which is anonymous)
//! keys the right def node and name. This is best-effort structural extraction
//! (no golden-master vs C), so it is NOT claimed as `supported`.
//!
//! Imprecision / honesty:
//!   * A `model`/`type` references another model through a `column_type`'s
//!     `identifier` (e.g. `author User`, `posts Post[]`). These cross-model
//!     references are NOT emitted as CALLS/edges: the enclosing declaration is a
//!     type (not a callable), so the CALLS source endpoint would not resolve.
//!     The sample's `Post.author : User` relation is therefore captured only as
//!     structure (both models are surfaced), not as an explicit edge.
//!   * Individual `column_declaration` fields are not surfaced as their own def
//!     nodes; only the top-level declarations are.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Prisma definitions are its top-level declarations. None are callable and none
/// are owned (Prisma has no method/class-member semantics at this level), so
/// every rule is a `DefRule::ty`. `Capture` sets the def node = the `@name`
/// identifier's parent, which is precisely the `*_declaration` node keyed here.
static PRISMA_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("model_declaration", "Model"),
        DefRule::ty("enum_declaration", "Enum"),
        DefRule::ty("datasource_declaration", "Datasource"),
        DefRule::ty("generator_declaration", "Generator"),
        DefRule::ty("type_declaration", "Type"),
    ],
    owner_kinds: &[],
    // Prisma has no call syntax; the CALLS pass is inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // Prisma has no import syntax; the IMPORTS pass is inert (import_query is
    // empty). Any variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // Prisma line comments use `//`; `///` are doc comments.
    docs: DocStyle::LineSlashComment,
};

/// Capture the name identifier of each top-level declaration as `@name`; the
/// engine derives the def node as that identifier's parent (`model_declaration`
/// / `enum_declaration` / …) and keys the matching `DefRule` on that parent's
/// kind.
///
/// Each declaration is `<keyword> <identifier> <block>` where the keyword token
/// (`model`, `enum`, `datasource`, `generator`, `type`) is anonymous, so the
/// name is the FIRST `identifier` child. Anchoring `.` to the first named child
/// captures exactly that identifier and never a nested one (a nested identifier
/// — e.g. a `column_type`'s reference — lives under `statement_block`, whose
/// parent is not a def node).
const DEFINITIONS: &str = r#"
    (model_declaration      . (identifier) @name)
    (enum_declaration       . (identifier) @name)
    (datasource_declaration . (identifier) @name)
    (generator_declaration  . (identifier) @name)
    (type_declaration       . (identifier) @name)
"#;

inventory::submit! {
    LangDef {
        name: "prisma",
        extensions: &["prisma"],
        filenames: &[],
        grammar: || tree_sitter_prisma_io::LANGUAGE.into(),
        spec: &PRISMA_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
