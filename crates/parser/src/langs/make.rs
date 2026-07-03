//! Make (Makefiles) — onboarded via the parallel-safe registry
//! (`crate::registry`). This whole file is the entire surface: it declares the
//! spec + queries + grammar and self-registers with `inventory::submit!`. No
//! shared file is edited (build.rs discovers this module automatically); the
//! only Cargo.toml line added is the `tree-sitter-make` dependency (crates.io
//! `tree-sitter-make = "1.1.1"`, which builds against tree-sitter 0.25 via the
//! `tree-sitter-language` shim — its grammar accessor is
//! `tree_sitter_make::LANGUAGE`).
//!
//! Status: **experimental / partial**. Make is a build/configuration language,
//! not a programming language: it has no functions and no call expressions in
//! the programming sense, so no CALLS or IMPORTS edges are produced (both those
//! queries are intentionally empty). What the registry *can* surface — and what
//! makes a Makefile greppable as structure — are its top-level definition
//! nodes, verified with `examples/dump_make.rs` against the real grammar:
//!
//!   * `rule`                 — a `target: prereqs …` recipe    → `Target`
//!   * `variable_assignment`  — a `NAME = value` binding         → `Variable`
//!
//! Node-kind facts (from `dump_make.rs`):
//!   * A `rule` has a `targets` child (holding one or more `word` targets) and,
//!     for a normal rule, a `[normal] prerequisites` child (holding `word`
//!     prerequisites that reference *other* targets — e.g. `build: main.o`).
//!     The grammar exposes NO `name:` field on `rule`.
//!   * A `variable_assignment` exposes its name on the `[name]` field, whose
//!     value is a `word` (e.g. `CC = gcc`).
//!
//! Because the grammar puts no `name:` field on `rule`, the `Capture` name
//! strategy is used with the def node = the *parent* of the captured `@name`
//! node: capturing the `targets` node yields def node = its parent `rule`
//! (and name = the target text, e.g. `all`); capturing the `variable_assignment`
//! `name:` `word` yields def node = its parent `variable_assignment` (and name =
//! the variable text, e.g. `CC`). A single capture per container therefore
//! yields exactly the def node and name we want.
//!
//! Imprecision / honesty:
//!   * A rule with multiple targets (`a b c: dep`) is surfaced ONCE, named by
//!     the concatenated `targets` text (the grammar groups all target `word`s
//!     under one `targets` node), not once per target.
//!   * Prerequisites (`build: main.o`) DO reference other targets, but Make
//!     targets are not callables: a `rule` is a `DefRule::ty` (not callable), so
//!     no CALLS edge is emitted for a prerequisite → target reference. The
//!     cross-reference is visible as structure only (the prerequisite text lives
//!     inside the rule's span), not as a resolved edge. This is best-effort
//!     structural extraction (no golden-master vs C), so it is NOT claimed as
//!     `supported`.
//!   * `include` directives are not expanded into IMPORTS edges: no Make import
//!     strategy exists in `ImportStrategy`, so `import_query` is empty.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Make definitions are its structural containers. Neither is callable and
/// neither is owned (Make has no method/class semantics), so every rule is a
/// `DefRule::ty`. `Capture` sets the def node = the `@name` node's parent, which
/// is precisely the `rule` (parent of `targets`) / `variable_assignment`
/// (parent of the `name:` `word`) node keyed here.
static MAKE_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("rule", "Target"),
        DefRule::ty("variable_assignment", "Variable"),
    ],
    owner_kinds: &[],
    // Make has no call syntax the template models; the CALLS pass is inert
    // (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // Make `include` is not extracted (no Make import strategy exists);
    // import_query is empty so any variant is inert. Pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // Make line comments use `#`.
    docs: DocStyle::LineHashComment,
};

/// Capture the name of each structural container as `@name`; the engine derives
/// the def node as that node's parent and keys the matching `DefRule` on that
/// parent's kind.
///
/// * A `rule`'s target list is a `targets` node; its parent is the `rule`, so
///   capturing `(targets)` yields def node = `rule` and name = the target text.
/// * A `variable_assignment`'s name is a `word` on the `name:` field; its parent
///   is the `variable_assignment`, so capturing that `word` yields def node =
///   `variable_assignment` and name = the variable text.
const DEFINITIONS: &str = r#"
    (rule (targets) @name)
    (variable_assignment name: (word) @name)
"#;

inventory::submit! {
    LangDef {
        name: "make",
        extensions: &["mk"],
        filenames: &["Makefile", "makefile", "GNUmakefile"],
        grammar: || tree_sitter_make::LANGUAGE.into(),
        spec: &MAKE_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
