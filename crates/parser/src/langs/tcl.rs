//! Tcl — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `bca-tree-sitter-tcl` dependency (the maintained
//! `tree-sitter-grammars` Tcl fork, whose Rust binding compiles the C grammar
//! and exposes it through the `tree-sitter-language` 0.1 shim, so it links
//! cleanly against the workspace's tree-sitter 0.25).
//!
//! Status: **experimental**. Tcl IS a programming language: a `proc name {args}
//! {body}` parses as a distinct `procedure` node whose `name:` field is a
//! `simple_word`, so the `Capture` name strategy applies (def node = the
//! `simple_word`'s parent = `procedure`) and each proc becomes a `Function`.
//! Tcl has no class/method concept, so no ownership is modelled.
//!
//! Calls: every other Tcl command parses as a `command` node with a `name:`
//! `simple_word` (e.g. `total`, `return`, `set`). A call `[total $a $b]` nests
//! its `command` inside a `command_substitution`, but the inner node is still a
//! `command` with `name: (simple_word)`, so it is captured. Because `procedure`
//! is its OWN node kind (not a `command`), proc *definitions* are never picked
//! up by the CALLS query. Imprecision: Tcl does not distinguish user procs from
//! built-in commands syntactically, so built-ins invoked as commands
//! (`return`, `expr`, `set`, `puts`, …) are captured as calls too (best-effort,
//! same as Elixir/Erlang capturing keyword-ish calls). `set`/`namespace`
//! statements have their own node kinds and are not commands, so those are not
//! double-counted. Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Each `proc` becomes a Function definition. The `procedure` node (the parent
/// of the `@name` `simple_word`) is what `DefRule::func("procedure")` keys on.
/// Tcl has no methods/classes, so no ownership is modelled.
static TCL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("procedure")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Tcl `source`/`package require` are not extracted yet (import_query is
    // empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// `proc add {a b} {…}` parses as `(procedure name: (simple_word) @name)`.
/// Capture the `simple_word` as `@name`; the engine derives the def node as its
/// parent `procedure` and keys `DefRule::func("procedure")` on it.
const DEFINITIONS: &str = r#"
    (procedure
      name: (simple_word) @name) @def
"#;

/// A command invocation `foo bar` (or `[foo bar]`) parses as `(command name:
/// (simple_word) @callee …)`. `procedure` is a distinct node kind, so proc
/// definitions are not matched here; only real command calls are. Built-in
/// commands invoked this way are captured too (best-effort).
const CALLS: &str = r#"
    (command
      name: (simple_word) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "tcl",
        extensions: &["tcl"],
        filenames: &[],
        grammar: || tree_sitter_tcl::LANGUAGE.into(),
        spec: &TCL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
