//! PowerShell — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-powershell` dependency (crates.io `0.26.4`,
//! which builds against tree-sitter 0.25 via the `tree-sitter-language` shim —
//! its only tree-sitter dependency is `tree-sitter-language`, so it links
//! against the workspace's 0.25 runtime).
//!
//! Status: **experimental**. The `tree-sitter-powershell` grammar (verified
//! with `examples/dump_ps.rs`) models a function definition as a
//! `function_statement` whose name is a plain `function_name` child — NOT a
//! `name:` field. With the `Capture` name strategy the def node is therefore
//! the `function_name`'s parent (`function_statement`), so a single
//! `DefRule::func("function_statement")` emits one Function per definition.
//!
//! Honesty / imprecision:
//!   * `function_statement` does NOT expose a `name:` field (the name is a bare
//!     `function_name` child), so the engine's enclosing-callable resolution
//!     (`callable_name`, which reads `child_by_field_name("name")`) returns
//!     `None`. Consequently CALLS edges whose source is a PowerShell function
//!     are NOT resolved — the same limitation as Julia. Call *targets* are still
//!     captured (`command_name`), they simply have no enclosing-function source
//!     to hang off, so no CALLS edge is emitted for calls made inside a
//!     function body. Top-level commands likewise have no enclosing callable.
//!   * A call parses as `(command command_name: (command_name) @callee …)`;
//!     the `command_name` text is the invoked command / function.
//!   * `import`/`using`/dot-sourcing are not expanded into IMPORTS edges (no
//!     PowerShell import strategy exists in `ImportStrategy`), so `import_query`
//!     is empty.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Each `function_statement` (the parent of the `@name` `function_name`) becomes
/// a Function definition. No class/method ownership is modelled (PowerShell
/// functions defined at script scope are free functions).
static POWERSHELL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("function_statement")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // PowerShell `using` / dot-sourcing / `Import-Module` are not extracted
    // (no PowerShell import strategy exists); `import_query` is empty so any
    // variant is inert.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// `function Foo { … }` parses as `(function_statement (function_name) @name …)`.
/// Capture the `function_name` as `@name`; the engine derives the def node as
/// its parent `function_statement` and keys `DefRule::func("function_statement")`
/// on it. (`function_name` is a plain child, not a `name:` field.)
const DEFINITIONS: &str = r#"
    (function_statement
      (function_name) @name) @def
"#;

/// A command invocation parses as `(command command_name: (command_name) @callee
/// command_elements: …)`. Capture the `command_name` as the callee. Because
/// `function_statement` has no `name:` field the enclosing-callable source often
/// does not resolve (see module docs), but the query is correct and harmless.
const CALLS: &str = r#"
    (command
      command_name: (command_name) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "powershell",
        extensions: &["ps1", "psm1"],
        filenames: &[],
        grammar: || tree_sitter_powershell::LANGUAGE.into(),
        spec: &POWERSHELL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
