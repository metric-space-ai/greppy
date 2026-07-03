//! MATLAB — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-matlab` dependency (crates.io `tree-sitter-matlab`
//! v1.3.0, which builds against tree-sitter 0.25 via the `tree-sitter-language`
//! shim — same mechanism as the PureScript grammar).
//!
//! Status: **experimental**. The grammar (verified with `examples/dump_ts.rs`)
//! exposes clean, distinct node kinds:
//!
//!   * `function_definition` — `function r = add(a, b) … end`, name on the
//!                             `name:` field (an `identifier`)     → `Function`
//!   * `function_call`       — `add(a, b)`, callee on the `name:` field (an
//!                             `identifier`)                        → CALLS edge
//!
//! Because `function_definition` DOES expose a `name:` field, the engine's
//! enclosing-callable resolution succeeds, so CALLS edges whose source is a
//! MATLAB function ARE resolved.
//!
//! Imprecision / honesty:
//!   * MATLAB's `function_call` node is shared between real calls (`add(a, b)`)
//!     and array indexing (`x(1)`) — the grammar cannot distinguish them, so a
//!     variable index reads as a call to that variable name (best-effort; the
//!     callee simply won't resolve to any def and is dropped downstream).
//!   * Command-syntax calls (`disp foo`) and method/dotted calls
//!     (`obj.method(...)`) are captured only where they surface a `name:`
//!     `identifier` head; qualified segments are dropped.
//!   * `import` statements are not expanded into IMPORTS edges: no MATLAB import
//!     strategy exists in `ImportStrategy`, so `import_query` is empty.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// `function r = add(a, b) … end` parses as a `function_definition` whose name
/// sits on the `name:` field (an `identifier`). With the `Capture` name
/// strategy the def node is the `@name` identifier's parent — precisely the
/// `function_definition` node keyed here. No class/method ownership is modelled
/// (`owner_kinds` empty); nested/local functions are still emitted as free
/// `Function`s.
static MATLAB_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("function_definition")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // MATLAB `import` statements are not extracted (no MATLAB import strategy
    // exists); `import_query` is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // MATLAB comments start with `%` (and block `%{ … %}`), for which there is
    // no DocStyle marker (the line-comment-run helpers key on `//` / `#` / `--`);
    // so no docs.
    docs: DocStyle::None,
};

/// The function name is a direct child of the `name:` field. Capture that
/// `identifier` as `@name`; the engine derives the def node as its parent
/// (`function_definition`) and keys `DefRule::func("function_definition")` on it.
const DEFINITIONS: &str = r#"
    (function_definition
      name: (identifier) @name) @def
"#;

/// A call `add(a, b)` parses as `(function_call name: (identifier) @callee …)`.
/// Capture the `name:` identifier as the callee. The engine hangs the CALLS
/// edge off the enclosing `function_definition` (which exposes a `name:` field,
/// so the source endpoint resolves).
const CALLS: &str = r#"
    (function_call
      name: (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "matlab",
        extensions: &["matlab"],
        filenames: &[],
        grammar: || tree_sitter_matlab::LANGUAGE.into(),
        spec: &MATLAB_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
