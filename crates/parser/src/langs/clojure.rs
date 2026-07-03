//! Clojure — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-clojure-orchard` dependency.
//!
//! Status: **experimental**. Clojure is homoiconic — the grammar
//! (`tree-sitter-clojure-orchard`) models *every* form as a generic `list_lit`
//! whose children are `sym_lit` symbols. There is no `function_definition`
//! kind and no `name:` field: a `(defn foo [..] ..)` definition is just a
//! `list_lit` whose first symbol is the literal `defn` and whose second symbol
//! is the name. Definition/call extraction is therefore predicate-based (keyed
//! off the leading symbol text), exactly like the Elixir onboarding.
//!
//! Because `list_lit` exposes no `name:` field, the engine's enclosing-callable
//! resolution (which reads `child_by_field_name("name")`) cannot attribute a
//! call to its surrounding `defn`, so CALLS edges whose *source* is a Clojure
//! function are NOT materialised (same limitation as Julia). Definition nodes
//! (the important signal) ARE extracted. Not claimed as `supported` (no
//! golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Every top-level definition form (`def`, `defn`, `defn-`, `defmacro`,
/// `defmulti`, `defmethod`, `defprotocol`, `defrecord`, `deftype`,
/// `definline`, `defonce`) parses as a `list_lit`. With the `Capture` strategy
/// the def node is the `@name` capture's PARENT, so we capture the *name*
/// `sym_lit` (whose parent is the enclosing `list_lit`) and key the single
/// `DefRule` on `"list_lit"`. The DEFINITIONS query only ever matches def-form
/// lists (it filters on the leading keyword), so no ordinary call-list leaks in.
static CLOJURE_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("list_lit")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Clojure `(require ...)` / `(:require ...)` / `(ns ... (:require ..))`
    // imports are not extracted yet (import_query is empty); any variant is
    // inert without a query.
    imports: ImportStrategy::Lua,
    docs: DocStyle::None,
};

/// `(defn add [a b] ...)` parses as `(list_lit (sym_lit (sym_name "defn"))
/// (sym_lit "add") ...)`. Capture the leading symbol's `sym_name` as `@_kw` to
/// filter on the def keyword, and the SECOND `sym_lit` as `@name` (its parent
/// is the `list_lit` def node). The `.` anchors pin `@_kw` to the first symbol
/// and `@name` to the one immediately after it (the definition name).
const DEFINITIONS: &str = r#"
    (list_lit
      .
      (sym_lit (sym_name) @_kw)
      .
      (sym_lit) @name
      (#any-of? @_kw
        "def" "defn" "defn-" "defmacro" "defmulti" "defmethod"
        "defprotocol" "defrecord" "deftype" "definline" "defonce")) @def
"#;

/// An ordinary application `(add x 10)` is a `list_lit` whose leading `sym_lit`
/// names the callee. Capture that leading symbol's `sym_name` as `@callee`,
/// excluding the definition keywords (owned by the DEFINITIONS pass) and the
/// most common special forms / macros that are not user-defined calls.
const CALLS: &str = r#"
    (list_lit
      .
      (sym_lit (sym_name) @callee)
      (#not-any-of? @callee
        "def" "defn" "defn-" "defmacro" "defmulti" "defmethod"
        "defprotocol" "defrecord" "deftype" "definline" "defonce"
        "ns" "let" "letfn" "if" "if-not" "if-let" "when" "when-not"
        "when-let" "cond" "condp" "case" "do" "fn" "quote" "loop"
        "recur" "for" "doseq" "dotimes" "try" "catch" "finally"
        "throw" "->" "->>" "as->" "some->" "some->>" "and" "or"))
"#;

inventory::submit! {
    LangDef {
        name: "clojure",
        extensions: &["clj", "cljs", "cljc"],
        filenames: &[],
        grammar: || tree_sitter_clojure_orchard::LANGUAGE.into(),
        spec: &CLOJURE_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
