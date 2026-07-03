//! JSON — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-json` dependency.
//!
//! Status: **experimental** (data language, not code). JSON has no functions,
//! classes, calls, or imports — it is a pure data format. To still surface a
//! useful symbol index, the top-level keys of the root `object` are captured as
//! `Key` definitions: each `pair` whose parent is the document's root object
//! becomes one `Key` node named after its string key. This makes `stats`/search
//! see the document's structure (which sections a config file declares) instead
//! of nothing.
//!
//! Imprecision to be honest about:
//!   * The tree-sitter-json grammar has NO field/kind that exposes a key's raw
//!     text without quotes, and the `Capture` strategy takes the def node = the
//!     `@name` capture's PARENT. The only child of a `pair` whose parent is the
//!     `pair` is the `key:` `string` node, so `@name` is that `string` and the
//!     extracted name INCLUDES the surrounding double quotes (e.g. `"server"`).
//!   * Only ROOT-level keys are captured (via `(document (object (pair ...)))`),
//!     not nested keys — nesting every key would explode the node count on large
//!     config files and add little value.
//!   * There are no CALLS or IMPORTS in JSON, so those queries are empty.
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Top-level `pair`s (a `document`'s root object's members) become `Key`
/// definitions. With the `Capture` strategy the def node is the `@name`
/// capture's parent; capturing the `pair`'s `key:` `string` makes the parent the
/// `pair`, so the DefRule keys on `"pair"`.
static JSON_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::ty("pair", "Key")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // JSON has no import syntax; import_query is empty so this variant is inert.
    imports: ImportStrategy::Bash,
    docs: DocStyle::None,
};

/// Root-level `pair`s only: `(document (object (pair key: (string) @name)))`.
/// The captured `string` (the key) is `@name`; its parent — the `pair` — is the
/// `@def`. Nested pairs (inside a value `object`/`array`) are intentionally not
/// matched.
const DEFINITIONS: &str = r#"
    (document
      (object
        (pair
          key: (string) @name) @def))
"#;

inventory::submit! {
    LangDef {
        name: "json",
        extensions: &["json"],
        filenames: &[],
        grammar: || tree_sitter_json::LANGUAGE.into(),
        spec: &JSON_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
