//! LaTeX — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `codebook-tree-sitter-latex` dependency (a crates.io release
//! that builds against tree-sitter 0.25 via the `tree-sitter-language` shim —
//! exactly like the git shims used by `crystal.rs` / `purescript.rs`, but a
//! published crate rather than a git dependency).
//!
//! NOTE on crate choice: the plainly-named `tree-sitter-latex` v0.1.0 crate on
//! crates.io is UNBUILDABLE — its `parser.c` references an external scanner
//! (`tree_sitter_latex_external_scanner_*`) but the crate ships no `scanner.c`,
//! so linking fails with undefined symbols. `codebook-tree-sitter-latex` v0.6.1
//! is the LaTeX grammar packaged for `codebook`; it ships both `parser.c` and
//! `scanner.c`, uses the `tree-sitter-language` shim, and links cleanly against
//! the workspace's tree-sitter 0.25. Its grammar accessor is
//! `codebook_tree_sitter_latex::LANGUAGE`.
//!
//! Status: **experimental / partial**. LaTeX is a markup/macro language, not a
//! programming language: it has no classes/methods and no notion of ownership.
//! The grammar (verified with `examples/dump_latex.rs`) exposes clean, distinct
//! definition node kinds, each carrying its name on a field:
//!
//!   * `\newcommand{\foo}{...}` parses as
//!       `(new_command_definition declaration: (curly_group_command_name
//!          command: (command_name "\foo")) implementation: (curly_group ...))`.
//!     The macro NAME (`\foo`) is a `command_name`. Capturing that `command_name`
//!     as `@name` makes the def node its parent — a `curly_group_command_name`
//!     for the braced form `\newcommand{\foo}`, or the `new_command_definition`
//!     itself for the brace-less form `\newcommand\foo`. Both parents are keyed
//!     → `Macro`. Capturing the *name* (not the wrapping curly group) keeps the
//!     def's name equal to `\foo`, so a later `\foo` reference RESOLVES against
//!     it (see CALLS below).
//!   * `\newenvironment{myenv}{begin}{end}` parses as
//!       `(environment_definition name: (curly_group_text ...) begin: ... end: ...)`.
//!     The name field is a `curly_group_text`; capturing it makes the def node
//!     the `environment_definition`               → `Environment`.
//!   * `\label{sec:intro}` parses as
//!       `(label_definition name: (curly_group_label label: (label ...)))`.
//!     Capturing the `curly_group_label` name field makes the def node the
//!     `label_definition`                          → `Label`.
//!
//! CALLS: a macro use `\foo` parses as `(generic_command command: (command_name
//! "\foo"))`. The callee capture yields `\foo`, which equals the name recorded
//! for a `\newcommand{\foo}` definition, so the callee endpoint resolves by
//! name. HOWEVER — see the honesty note below — the *source* endpoint does NOT
//! resolve: the enclosing callable (a macro definition) exposes its name under
//! the grammar's `declaration:` field, not `name:`, and the engine's generic
//! `Capture` callable-name lookup reads only `name:`. So call sites are dropped
//! for lack of a resolvable source, and in practice NO CALLS edges are emitted.
//! The `call_query` is retained (it is correct and would light up if the engine
//! ever resolved macro sources), but it is effectively inert today.
//!
//! Imprecision / honesty:
//!   * NO CALLS edges are produced. A macro reference inside another macro's
//!     body (`\newcommand{\greet}{\hello,...}`) is attributed to the enclosing
//!     `new_command_definition`, whose macro name lives on `declaration:` (a
//!     `curly_group_command_name`), not on a `name:` field. The engine resolves
//!     an enclosing callable's name only via `name:` (`callable_name` in
//!     `spec.rs`), so it returns `None` and the call site is discarded. This is
//!     a structural limitation of the generic template for a grammar that names
//!     macros via `declaration:` rather than `name:`; fixing it would require an
//!     engine change, which is out of scope for a registry-only onboarding.
//!   * Environment and label NAMES retain their surrounding braces text form
//!     only insofar as the grammar's `curly_group_*` byte range — in practice
//!     `curly_group_text`/`curly_group_label` span the braces, so those names
//!     read as `{myenv}` / `{sec:intro}`. Macro names (the common case) are the
//!     clean `\foo` `command_name` text, which is what CALLS needs.
//!   * `generic_command` captures EVERY command invocation as a callee,
//!     including built-ins (`\section`, `\textbf`, …) that have no local
//!     definition. Those simply produce unresolved call sites (dropped at
//!     resolution) — they do not create spurious nodes.
//!   * `\input` / `\include` / `\usepackage` are NOT expanded into IMPORTS
//!     edges: no LaTeX import strategy exists in `ImportStrategy`, so
//!     `import_query` is empty (the IMPORTS pass is inert).
//!   * Sectioning commands (`\section`, `\chapter`, …) are intentionally NOT
//!     captured as definitions: their `text:` field is a generic `curly_group`
//!     that also wraps ordinary prose, so keying on it would emit noise.
//!   * LaTeX comments are `%`-to-end-of-line, which no `DocStyle` variant models,
//!     so `DocStyle::None` is used (no docstrings extracted).
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions are LaTeX's top-level macro / environment / label declarations.
/// None are owned (LaTeX has no class/method semantics), so `owner_kinds` is
/// empty and every rule is `DefRule::ty` except the macro forms, which are
/// `DefRule::func` so a macro can be the source endpoint of a CALLS edge.
///
/// `Capture` sets the def node = the `@name` capture's parent:
///   * `command_name` (a macro name) → parent is `curly_group_command_name`
///     (braced `\newcommand{\foo}`) or `new_command_definition` (brace-less
///     `\newcommand\foo`); both are keyed → `Macro`.
///   * `curly_group_text`  (an env name field) → parent `environment_definition`.
///   * `curly_group_label` (a label name field) → parent `label_definition`.
static LATEX_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        // Braced macro form: `\newcommand{\foo}` → def node curly_group_command_name.
        DefRule::func("curly_group_command_name"),
        // Brace-less macro form: `\newcommand\foo` → def node new_command_definition.
        DefRule::func("new_command_definition"),
        DefRule::ty("environment_definition", "Environment"),
        DefRule::ty("label_definition", "Label"),
    ],
    owner_kinds: &[],
    // A macro use `\foo` is `(generic_command command: (command_name))`. The
    // callee name resolves, but the source (enclosing macro) does not — its name
    // is on `declaration:`, not `name:` — so no CALLS edges are emitted today.
    calls: CallSpec { skip_callees: &[] },
    // No LaTeX import strategy exists in `ImportStrategy`; `import_query` is
    // empty so any variant is inert. Pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // LaTeX comments are `%`-to-EOL, which no DocStyle models → None.
    docs: DocStyle::None,
};

/// Capture each definition's NAME node as `@name`; the engine derives the def
/// node as that node's parent and keys the DefRule on the parent's kind.
///
///   * The macro name is the `command_name` under a command definition's
///     `declaration:` field. For the braced form its parent is
///     `curly_group_command_name`; for the brace-less form its parent is the
///     `new_command_definition`. Both parents are registered → `Macro`. We
///     capture the `command_name` (not the wrapping curly group) so the def
///     name equals `\foo`, matching a later `\foo` reference for CALLS.
///   * The environment name is the `curly_group_text` on the `name:` field of
///     `environment_definition`; its parent is the `environment_definition`.
///   * The label name is the `curly_group_label` on the `name:` field of
///     `label_definition`; its parent is the `label_definition`.
const DEFINITIONS: &str = r#"
    (new_command_definition
      declaration: (curly_group_command_name
        command: (command_name) @name))
    (new_command_definition
      declaration: (command_name) @name)
    (environment_definition
      name: (curly_group_text) @name)
    (label_definition
      name: (curly_group_label) @name)
"#;

/// A macro use `\foo` parses as `(generic_command command: (command_name
/// "\foo"))`. Keying the callee on `command_name` captures `\foo`, which equals
/// the name recorded for the `\newcommand{\foo}` definition — so the *callee*
/// endpoint would resolve by name. The *source* endpoint does not (the enclosing
/// macro's name is on `declaration:`, not `name:`), so the engine drops the call
/// and no CALLS edge is emitted today (see the module honesty notes). The query
/// is kept correct so it lights up if the engine gains macro-source resolution.
const CALLS: &str = r#"
    (generic_command
      command: (command_name) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "latex",
        extensions: &["tex"],
        filenames: &[],
        grammar: || codebook_tree_sitter_latex::LANGUAGE.into(),
        spec: &LATEX_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
