//! Dockerfile — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-containerfile` dependency (a crates.io
//! release that builds against tree-sitter 0.25 via the `tree-sitter-language`
//! ^0.1 shim — exactly like `tree-sitter-crystal` / `tree-sitter-purescript`;
//! its accessor is `tree_sitter_containerfile::LANGUAGE`). The grammar covers
//! both Containerfiles and Dockerfiles.
//!
//! Status: **experimental / partial**. A Dockerfile is a build/config language,
//! not a programming language: it has no functions and no call expressions in
//! the programming sense, so nothing is extracted as a `Function`/`Method` and
//! no CALLS or IMPORTS edges are produced (both those queries are intentionally
//! empty). What the registry *can* surface — and what makes a Dockerfile
//! greppable as structure — are its named build stages:
//!
//!   * `from_instruction` — a `FROM <image> AS <name>` build stage, named by
//!     its `as:` field (an `image_alias`)                          → `Stage`
//!
//! Verified with `examples/dump_dockerfile.rs`. A multi-stage Dockerfile such
//! as
//!
//!   FROM golang:1.21 AS builder      → Stage "builder"
//!   FROM builder    AS tester        → Stage "tester" (references "builder")
//!   FROM alpine:3.19 AS runtime      → Stage "runtime"
//!
//! parses each stage as a `from_instruction` whose alias sits on the `as:`
//! field as an `image_alias`. With the `Capture` name strategy the definition
//! node is the *parent* of the captured alias — precisely the
//! `from_instruction` we want — so a single alias capture per stage yields the
//! right def node and name.
//!
//! Imprecision / honesty:
//!   * A base-image-only `FROM alpine` (no `AS <name>`) is NOT captured: it has
//!     no `image_alias`, so nothing keys the `Capture` rule (a stage without an
//!     explicit name is an anonymous, un-greppable stage — omitting it is
//!     correct, not a bug).
//!   * Cross-stage *references* are not emitted as edges. A later stage names an
//!     earlier one via `FROM builder AS …` (`image_spec name: (image_name)`) or
//!     `COPY --from=builder …`. Neither is surfaced as a CALLS/USES edge: a
//!     `from_instruction`'s name lives on its `as:` field (not the `name:` field
//!     the generic CALLS source-resolver reads), and the grammar hides the
//!     `--from=` value as an unnamed token inside a `param` node, so it is not
//!     capturable. Stages are therefore modelled as non-callable `DefRule::ty`
//!     nodes (like the TOML config-language spec), and the CALLS pass is inert.
//!   * `ARG` / `ENV` / `LABEL` pairs and other instructions are not surfaced;
//!     only the top-level named build stages are.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// A Dockerfile's definitions are its named build stages. A stage is not
/// callable and is never owned (a Dockerfile has no method/class semantics), so
/// the single rule is a `DefRule::ty`. `Capture` sets the def node = the
/// `@name` alias's parent, which is precisely the `from_instruction` keyed here.
static DOCKERFILE_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::ty("from_instruction", "Stage")],
    owner_kinds: &[],
    // A Dockerfile has no call syntax; the CALLS pass is inert (call_query is
    // empty). Cross-stage references live on `as:`-not-`name:` fields / hidden
    // `--from=` tokens, so they are not resolvable as CALLS edges (see header).
    calls: CallSpec { skip_callees: &[] },
    // A Dockerfile has no import syntax the engine models; the IMPORTS pass is
    // inert (import_query is empty). Any variant is dead weight without a query.
    imports: ImportStrategy::Bash,
    // Dockerfile comments use `#`.
    docs: DocStyle::LineHashComment,
};

/// Capture the alias of each named build stage as `@name`; the engine derives
/// the def node as that alias's parent (`from_instruction`) and keys the
/// `DefRule` on that parent's kind. The alias is an `image_alias` on the
/// `as:` field of the `from_instruction`, a direct child, so its `.parent()`
/// is the `from_instruction` itself. A `FROM` with no `AS <name>` has no
/// `image_alias` and so matches nothing (anonymous stages are omitted).
const DEFINITIONS: &str = r#"
    (from_instruction as: (image_alias) @name)
"#;

inventory::submit! {
    LangDef {
        name: "dockerfile",
        extensions: &["dockerfile"],
        filenames: &[],
        grammar: || tree_sitter_containerfile::LANGUAGE.into(),
        spec: &DOCKERFILE_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
