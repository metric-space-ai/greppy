//! Protobuf (`.proto`) — onboarded via the parallel-safe registry
//! (`crate::registry`). This whole file is the entire surface: it declares the
//! spec + queries + grammar and self-registers with `inventory::submit!`. No
//! shared file is edited (build.rs discovers this module automatically); the
//! only Cargo.toml line added is the `tree-sitter-proto` dependency.
//!
//! Status: **experimental / partial**. Protobuf is an interface-definition /
//! data-description language, not a programming language: it has no function
//! bodies and no call expressions, so there is nothing to extract as a
//! `Function`/`Method` and no CALLS edges are produced (that query is
//! intentionally empty). What the registry *can* surface — and what makes a
//! `.proto` file greppable as structure — are its top-level definition nodes:
//!
//!   * `message` — a `message Foo { … }` declaration         → `Message`
//!   * `enum`    — an `enum Color { … }` declaration          → `Enum`
//!   * `service` — a `service Geometry { … }` declaration     → `Service`
//!   * `rpc`     — an `rpc Distance(Point) returns (Point);`  → `Rpc`
//!
//! The `tree-sitter-proto` grammar does NOT expose a `name:` field on any of
//! these nodes; the name lives in a dedicated child wrapper node
//! (`message_name` / `enum_name` / `service_name` / `rpc_name`), each of which
//! holds a single `identifier`. With the `Capture` name strategy the definition
//! node is the *parent* of the captured `@name` node, so we capture the wrapper
//! node itself (e.g. `message_name`): its text is exactly the declared name and
//! its `.parent()` is precisely the `message` / `enum` / `service` / `rpc` node
//! we want as the def node.
//!
//! Because these nodes carry no `name:` field, the engine's enclosing-callable
//! resolution (which reads `child_by_field_name("name")`) cannot resolve an
//! `rpc` as a CALLS source; combined with the absence of real call syntax, no
//! CALLS edges are emitted. `import "path";` statements are likewise NOT
//! extracted: the grammar puts the quoted path on the `import` node's `path:`
//! field, but no existing import expander reads that field (the `Bash` source
//! expander keys on an `argument:` field a proto `import` does not have), so the
//! import_query is intentionally left empty rather than claim an inert edge.
//! This is best-effort structural extraction (no golden-master vs C), so it is
//! NOT claimed as `supported`.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Protobuf definitions are its structural containers. None are callable and
/// none are owned (there is no method semantics that the uniform template can
/// express here), so every rule is a `DefRule::ty`. `Capture` sets the def node
/// = the `@name` wrapper's parent, which is precisely the `message` / `enum` /
/// `service` / `rpc` node keyed here.
static PROTOBUF_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("message", "Message"),
        DefRule::ty("enum", "Enum"),
        DefRule::ty("service", "Service"),
        DefRule::ty("rpc", "Rpc"),
    ],
    owner_kinds: &[],
    // Protobuf has no call syntax; the CALLS pass is inert (call_query empty).
    calls: CallSpec { skip_callees: &[] },
    // `import "path";` statements are not extracted (no expander reads the
    // `import` node's `path:` field); import_query is empty so any variant is
    // inert. Pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // Protobuf comments are `//` line / `/* */` block; use the C-style extractor.
    docs: DocStyle::CBlockOrLine,
};

/// Capture the name-wrapper of each container as `@name`; the engine derives the
/// def node as that wrapper's parent (`message` / `enum` / `service` / `rpc`)
/// and keys the DefRule on that parent's kind. The wrapper node's text is the
/// declared identifier itself (e.g. `message_name` spans exactly `Point`), so
/// `node_text(@name)` yields the right name.
const DEFINITIONS: &str = r#"
    (message (message_name) @name)
    (enum    (enum_name)    @name)
    (service (service_name) @name)
    (rpc     (rpc_name)     @name)
"#;

inventory::submit! {
    LangDef {
        name: "protobuf",
        extensions: &["proto"],
        filenames: &[],
        grammar: || tree_sitter_proto::LANGUAGE.into(),
        spec: &PROTOBUF_SPEC,
        def_query: DEFINITIONS,
        call_query: "",
        import_query: "",
    }
}
