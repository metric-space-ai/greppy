//! GraphQL — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-graphql` dependency.
//!
//! Status: **experimental / partial**. GraphQL is a schema/data + query
//! language, not a general-purpose programming language: its `type` / `interface`
//! / `enum` / `input` / `scalar` / `union` declarations describe a schema, and
//! its `query` / `mutation` / `subscription` operations + `fragment`s describe
//! executable documents. There are no functions and no call expressions in the
//! imperative sense, so no CALLS or IMPORTS edges are produced (both those
//! queries are intentionally empty). What the registry *can* surface — and what
//! makes a `.graphql` file greppable as structure — are its top-level
//! definition nodes, captured here as types:
//!
//!   * `object_type_definition`       — `type User { … }`        → `Type`
//!   * `interface_type_definition`    — `interface Node { … }`   → `Interface`
//!   * `enum_type_definition`         — `enum Role { … }`        → `Enum`
//!   * `input_object_type_definition` — `input CreateInput { … }`→ `Input`
//!   * `scalar_type_definition`       — `scalar DateTime`        → `Scalar`
//!   * `union_type_definition`        — `union Result = A | B`   → `Union`
//!   * `operation_definition`         — `query GetUser { … }`    → `Operation`
//!   * `fragment_definition`          — `fragment F on T { … }`  → `Fragment`
//!
//! The `tree-sitter-graphql` grammar (v0.1) exposes NO `name:` field on any
//! node — every child is positional. In every listed definition the identifier
//! sits as a direct `name` child of the definition node (for `fragment` the
//! `name` is wrapped in a `fragment_name`, so that one alternative is anchored
//! specifically). With the `Capture` name strategy the definition node is the
//! *parent* of the captured `name`, which is exactly the type/operation node we
//! want. This is best-effort structural extraction (no golden-master vs C), so
//! it is NOT claimed as `supported`.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// GraphQL definitions are its schema/operation containers. None are callable
/// and none are owned (GraphQL has no method/receiver semantics), so every rule
/// is a `DefRule::ty`. `Capture` sets the def node = the `@name` node's parent,
/// which is precisely the `*_type_definition` / `operation_definition` /
/// `fragment_definition` node keyed here.
static GRAPHQL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("object_type_definition", "Type"),
        DefRule::ty("interface_type_definition", "Interface"),
        DefRule::ty("enum_type_definition", "Enum"),
        DefRule::ty("input_object_type_definition", "Input"),
        DefRule::ty("scalar_type_definition", "Scalar"),
        DefRule::ty("union_type_definition", "Union"),
        DefRule::ty("operation_definition", "Operation"),
        DefRule::ty("fragment_definition", "Fragment"),
    ],
    owner_kinds: &[],
    // GraphQL has no call syntax; the CALLS pass is inert (call_query is empty).
    calls: CallSpec { skip_callees: &[] },
    // GraphQL has no import syntax; the IMPORTS pass is inert (import_query is
    // empty). Any variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // GraphQL comments start with `#`, matching the line-hash doc helper. A
    // leading `#` comment run above a definition becomes its docstring.
    docs: DocStyle::LineHashComment,
};

/// Capture the identifier of each top-level definition as `@name`; the engine
/// derives the def node as that capture's PARENT and keys the DefRule on the
/// parent's kind.
///
/// The `tree-sitter-graphql` grammar carries the identifier as a direct `name`
/// child of every `*_type_definition` and of `operation_definition`, so the
/// captured `name`'s parent IS the definition node — exactly the def kind keyed
/// in `GRAPHQL_SPEC`. `fragment_definition` is the one exception: its identifier
/// lives one level deeper, inside a `fragment_name` wrapper. Capturing the bare
/// `name` there would make the def node resolve to `fragment_name` (which has no
/// DefRule) and be dropped, so the fragment alternative instead captures the
/// whole `fragment_name` node: its parent is the `fragment_definition` (correct
/// def node) and its text is just the fragment identifier.
const DEFINITIONS: &str = r#"
    (object_type_definition       (name) @name)
    (interface_type_definition    (name) @name)
    (enum_type_definition         (name) @name)
    (input_object_type_definition (name) @name)
    (scalar_type_definition       (name) @name)
    (union_type_definition        (name) @name)
    (operation_definition         (name) @name)
    (fragment_definition          (fragment_name) @name)
"#;

inventory::submit! {
    LangDef {
        name: "graphql",
        extensions: &["graphql", "gql"],
        filenames: &[],
        grammar: || tree_sitter_graphql::LANGUAGE.into(),
        spec: &GRAPHQL_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
