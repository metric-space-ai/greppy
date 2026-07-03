//! Erlang — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-erlang` dependency.
//!
//! Status: **experimental**. The `tree-sitter-erlang` grammar models a function
//! definition as a `fun_decl` wrapping one or more `function_clause` nodes, and
//! it carries the function name on the `function_clause`'s `name:` field (an
//! `atom`) — NOT on the `fun_decl`. With the `Capture` name strategy the def
//! node is therefore the `function_clause` (the `@name` atom's parent), so a
//! single `DefRule::func("function_clause")` emits one Function per clause.
//! This grammar also splits a multi-clause function (`f(0) -> …; f(N) -> …`)
//! into SEPARATE `fun_decl`s each holding one `function_clause`, so a
//! multi-clause function yields one Function node per clause (same name, same
//! qname) — an intentional, harmless over-count.
//!
//! Because `function_clause` DOES expose a `name:` field, the engine's
//! enclosing-callable resolution succeeds, so CALLS edges whose source is an
//! Erlang function ARE resolved (unlike Julia). Call extraction captures the
//! callee `atom` in a `call` node's `expr:` position; for a remote call
//! (`mod:fun(…)`) the inner `call` still carries `expr: (atom "fun")`, so the
//! function segment is captured while the module qualifier is dropped
//! (best-effort).
//!
//! Status: **golden-master parity** on `bench/agent_efficiency/corpus/
//! erlang_small` vs the pinned C reference (all node labels + CALLS / THROWS /
//! USAGE / DEFINES / CONTAINS_* edges). Erlang no longer runs through the
//! generic `spec_extract` path — the `Language::Registered("erlang")` arm in
//! `extract.rs` dispatches to the bespoke `extract_erlang`, a faithful port of
//! the C reference's Erlang passes (`type_alias` → Type, `record_decl` /
//! `pp_define` → Variable, same-file-only CALLS mirrored as THROWS, and the
//! `atom`/`var` USAGE walk). The spec/queries below are retained for the
//! registry declaration (grammar + extensions) but are NOT consulted for
//! extraction. The one row that stays non-parity is the C `-module(x)`
//! self-import (a self-referential IMPORTS edge onto the file's own Module
//! node), which grepplus's importable-symbol IMPORTS resolver — which never
//! targets a structural Module node and drops self-loops — cannot model
//! without indexer changes; this is the out-of-scope "import → File/Module"
//! family.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Each `function_clause` (the parent of the `@name` atom) becomes a Function
/// definition. No class/module ownership is modelled (Erlang has no methods).
static ERLANG_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("function_clause")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Erlang `-import(...)` / `-include(...)` attributes are not extracted yet
    // (import_query is empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    // Erlang comments start with `%`, for which there is no DocStyle marker
    // (the line-comment-run helpers key on `//` / `#` / `--`); so no docs.
    docs: DocStyle::None,
};

/// `add(A, B) -> …` parses as `(fun_decl (function_clause name: (atom) @name))`.
/// Capture the `atom` as `@name`; the engine derives the def node as its parent
/// `function_clause` and keys the `DefRule::func("function_clause")` on it.
const DEFINITIONS: &str = r#"
    (function_clause
      name: (atom) @name) @def
"#;

/// Local calls parse as `(call expr: (atom) @callee …)`. A remote call
/// `mod:fun(…)` wraps an inner `(call expr: (atom "fun"))`, so this also
/// captures the function segment of remote calls (module qualifier dropped).
const CALLS: &str = r#"
    (call
      expr: (atom) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "erlang",
        extensions: &["erl", "hrl"],
        filenames: &[],
        grammar: || tree_sitter_erlang::LANGUAGE.into(),
        spec: &ERLANG_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
