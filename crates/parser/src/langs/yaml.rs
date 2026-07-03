//! YAML — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-yaml` dependency (v0.7.2, crates.io — it builds
//! against workspace tree-sitter 0.25 via the `tree-sitter-language` shim,
//! confirmed by `cargo build -p grepplus-parser`).
//!
//! Status: **experimental / partial**. YAML is a data/markup language, not a
//! programming language: it has no functions, no call expressions, and no
//! import syntax, so nothing is emitted as a callable `Function`/`Method` and
//! both the CALLS and IMPORTS passes are intentionally empty (inert). What the
//! registry *can* surface — and what makes a YAML file greppable as structure —
//! are its mapping entries: every `key: value` pair is a `block_mapping_pair`
//! node, which is captured as a `Key` definition.
//!
//! Grammar shape (verified with `examples/dump_yaml.rs`):
//!
//!   stream > document > block_node > block_mapping >
//!     block_mapping_pair
//!       key:   flow_node (plain_scalar > string_scalar)
//!       value: flow_node | block_node
//!
//! The `block_mapping_pair` exposes its key on the `key:` field (a `flow_node`).
//! With the `Capture` name strategy the definition node is the *parent* of the
//! captured `@name`, so capturing the key `flow_node` yields the enclosing
//! `block_mapping_pair` as the def node and the key text (e.g. `jobs`, `steps`)
//! as its name. Both top-level and nested pairs are captured — the same
//! best-effort structural extraction TOML uses for its `pair`s. This is NOT
//! claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// YAML definitions are its mapping entries. None are callable and none are
/// owned (YAML has no method/class semantics), so the single rule is a
/// `DefRule::ty` keyed on `block_mapping_pair` → `Key`. `Capture` sets the def
/// node = the `@name` key's parent, which is precisely that
/// `block_mapping_pair`.
static YAML_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::ty("block_mapping_pair", "Key")],
    owner_kinds: &[],
    // YAML has no call syntax; the CALLS pass is inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // YAML has no import syntax; the IMPORTS pass is inert (import_query is
    // empty). Any variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // YAML comments are `#` line comments; a run preceding a pair is its doc.
    docs: DocStyle::LineHashComment,
};

/// Capture the key of each mapping entry as `@name`; the engine derives the def
/// node as that key's parent (`block_mapping_pair`) and keys the DefRule on that
/// parent's kind. The key is a `flow_node` (a plain/quoted scalar) held on the
/// pair's `key:` field, so its `.parent()` is the `block_mapping_pair` itself.
const DEFINITIONS: &str = r#"
    (block_mapping_pair key: (flow_node) @name)
"#;

inventory::submit! {
    LangDef {
        name: "yaml",
        extensions: &["yaml", "yml"],
        filenames: &[],
        grammar: || tree_sitter_yaml::LANGUAGE.into(),
        spec: &YAML_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
