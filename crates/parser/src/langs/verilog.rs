//! Verilog / SystemVerilog — onboarded via the parallel-safe registry
//! (`crate::registry`). This whole file is the entire surface: it declares the
//! spec + queries + grammar and self-registers with `inventory::submit!`. No
//! shared file is edited (build.rs discovers this module automatically); the
//! only Cargo.toml line added is the `tree-sitter-systemverilog` dependency
//! (crates.io `0.3.1`, which builds against tree-sitter 0.25 via the
//! `tree-sitter-language` 0.1 shim — its accessor is `LANGUAGE`, and it parses
//! both `.v` and `.sv`).
//!
//! Status: **experimental / partial**. The grammar (verified with
//! `examples/dump_sv.rs`) does not expose a single node whose `name:` field is
//! the definition — names sit on nested header/body nodes:
//!
//!   * `module_ansi_header` / `module_nonansi_header` — the header of a
//!     `module … endmodule`, carrying the module identifier on its `name:`
//!     field (a `simple_identifier`). Captured as a `Module` def.  → `Module`
//!   * `function_body_declaration` — the body of a `function … endfunction`,
//!     carrying the function identifier on its `name:` field.        → `Function`
//!   * `task_body_declaration` — the body of a `task … endtask`, likewise.
//!                                                                    → `Function`
//!
//! Because the `Capture` name strategy sets the definition node = the captured
//! `@name`'s PARENT, keying each `DefRule` on the *header* / *body* kind is
//! exactly right: `@name`'s parent is `module_ansi_header` /
//! `function_body_declaration` / `task_body_declaration` respectively.
//!
//! Imprecision / honesty:
//!   * The def NODE for a module is its `module_ansi_header`, not the enclosing
//!     `module_declaration`, so the reported span covers the header line, not
//!     the whole `module … endmodule`. This is the direct consequence of the
//!     `Capture` (name-parent) rule and the grammar's nesting; it is accepted
//!     rather than adding a bespoke structural name strategy.
//!   * CALLS captures the head identifier of a `tf_call` (a task/function
//!     subroutine call). The engine hangs the edge off the nearest enclosing
//!     callable (`function_body_declaration` / `task_body_declaration`), so a
//!     call made at module level (e.g. inside a `continuous_assign`) has no
//!     enclosing callable and is dropped — only intra-subprogram calls resolve.
//!   * System calls (`$display`), method calls, and hierarchical/qualified
//!     calls beyond the leading `simple_identifier` are best-effort or dropped.
//!   * Verilog has no import syntax modelled here (`import_query` is empty), so
//!     no IMPORTS edges are produced; SystemVerilog `import pkg::*;` is not
//!     expanded (no matching `ImportStrategy` variant).
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `module_ansi_header` / `module_nonansi_header` → `Module` (never owned).
///  * `function_body_declaration` / `task_body_declaration` → `Function`
///    (Verilog subprograms are free; module ownership is not modelled, so
///    `owner_kinds` is empty and these use `DefRule::func`, keeping them
///    callable so intra-subprogram CALLS edges can hang off them).
///
/// Every captured `@name` is a direct `name:` child of the keyed node, so the
/// `Capture` strategy (name = `@name`, def = its parent) applies uniformly.
static VERILOG_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("module_ansi_header", "Module"),
        DefRule::ty("module_nonansi_header", "Module"),
        DefRule::func("function_body_declaration"),
        DefRule::func("task_body_declaration"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Verilog has no import syntax modelled here; the IMPORTS pass is inert
    // (import_query is empty). Any variant is dead weight without a query.
    imports: ImportStrategy::Bash,
    // Verilog line comments use `//`; block comments are `/* */`. The
    // C-block-or-line helper collapses a leading `/* */` block or a run of `//`
    // lines into the docstring.
    docs: DocStyle::CBlockOrLine,
};

/// Each definition carries its identifier on the `name:` field of its
/// header/body node. Capture that identifier as `@name`; the engine derives the
/// def node as its parent (`module_ansi_header` / `module_nonansi_header` /
/// `function_body_declaration` / `task_body_declaration`) and keys the matching
/// `DefRule` on that parent's kind.
const DEFINITIONS: &str = r#"
    (module_ansi_header    name: (simple_identifier) @name)
    (module_nonansi_header name: (simple_identifier) @name)
    (function_body_declaration name: (simple_identifier) @name)
    (task_body_declaration     name: (simple_identifier) @name)
"#;

/// A task/function subroutine call `f(a, b)` parses as `(tf_call
/// (hierarchical_identifier (simple_identifier "f")) (list_of_arguments …))`;
/// the callee is the leading `simple_identifier` of the `tf_call`'s
/// `hierarchical_identifier`. Anchoring the capture inside `tf_call` keeps
/// argument identifiers (which sit under `list_of_arguments > … > primary`) out
/// of the callee capture. The engine hangs the CALLS edge off the enclosing
/// `function_body_declaration` / `task_body_declaration`.
const CALLS: &str = r#"
    (tf_call (hierarchical_identifier (simple_identifier) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "verilog",
        extensions: &["v", "sv"],
        filenames: &[],
        grammar: || tree_sitter_systemverilog::LANGUAGE.into(),
        spec: &VERILOG_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
