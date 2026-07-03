//! TOML — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-toml-ng` dependency.
//!
//! Status: **experimental / partial**. TOML is a configuration/data language,
//! not a programming language: it has no functions and no call expressions, so
//! there is nothing to extract as a `Function`/`Method` and no CALLS or IMPORTS
//! edges are produced (both those queries are intentionally empty). What the
//! registry *can* surface — and what makes a TOML file greppable as structure —
//! are its top-level definition nodes:
//!
//!   * `table`               — a `[section]` header               → `Table`
//!   * `table_array_element` — a `[[array.of.tables]]` header     → `Table`
//!   * `pair`                — a `key = value` assignment          → `Key`
//!
//! The `tree-sitter-toml-ng` grammar does not expose a `name:` field on any of
//! these; the key sits as an anonymous `bare_key` / `dotted_key` child. With the
//! `Capture` name strategy the definition node is therefore the *parent* of the
//! captured key — exactly the `table` / `table_array_element` / `pair` node we
//! want — so a single key capture per container yields the right def node and
//! name. This is best-effort structural extraction (no golden-master vs C), so
//! it is NOT claimed as `supported`.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// TOML definitions are its structural containers. None are callable and none
/// are owned (TOML has no method/class semantics), so every rule is a
/// `DefRule::ty`. `Capture` sets the def node = the `@name` key's parent, which
/// is precisely the `table` / `table_array_element` / `pair` node keyed here.
static TOML_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    // C-reference parity: the C config extractor labels a TOML `table` /
    // `table_array_element` a "Class" (extract_defs.c:2944-2946,
    // CBM_LANG_TOML) and a top-level `pair` a "Variable" (its variable pass).
    // grepplus emitted "Table" / "Key" — grepplus-only rows in the golden
    // master, while C's Class/Variable rows (from Cargo.toml / pyproject.toml)
    // read as grepplus-missing. Same nodes, C's labels: on rust_medium this
    // turns Table 2 -> Class (reaching Class 3=3 with the Record struct) and
    // Key 4 -> Variable 4=4; on python_large Class 1=1 and Variable 2=2.
    defs: &[
        DefRule::ty("table", "Class"),
        DefRule::ty("table_array_element", "Class"),
        DefRule::ty("pair", "Variable"),
    ],
    owner_kinds: &[],
    // TOML has no call syntax; the CALLS pass is inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // TOML has no import syntax; the IMPORTS pass is inert (import_query is
    // empty). Any variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// Capture the key of each structural container as `@name`; the engine derives
/// the def node as that key's parent (`table` / `table_array_element` / `pair`)
/// and keys the DefRule on that parent's kind.
///
/// A section header's key is either a `bare_key` (`[server]`) or a `dotted_key`
/// (`[servers.config]`); a pair's key is likewise `bare_key` (`port = 8080`) or
/// `dotted_key` (`a.b.c = 1`). In every case the captured key is a *direct*
/// child of the container, so its `.parent()` is the container itself. The
/// `dotted_key` alternative is anchored inside its container so its own nested
/// `bare_key` children are never captured (their parent would be `dotted_key`,
/// not a def node).
const DEFINITIONS: &str = r#"
    (table (bare_key) @name)
    (table (dotted_key) @name)
    (table_array_element (bare_key) @name)
    (table_array_element (dotted_key) @name)
    (pair (bare_key) @name)
    (pair (dotted_key) @name)
"#;

inventory::submit! {
    LangDef {
        name: "toml",
        extensions: &["toml"],
        filenames: &[],
        grammar: || tree_sitter_toml_ng::LANGUAGE.into(),
        spec: &TOML_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
