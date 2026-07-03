//! Dart — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically).
//!
//! Status: **experimental**. Dart's tree-sitter grammar
//! (`tree-sitter-dart` 0.2) models both top-level functions and class methods
//! with an inner `function_signature` carrying the `name:` identifier
//! (top-level: `function_declaration > function_signature > identifier`;
//! method: `method_declaration > method_signature > function_signature >
//! identifier`). The generic `NameStrategy::Capture` extractor keys the
//! DefRule off the `@name` identifier's *direct parent*, which is
//! `function_signature` in BOTH cases, so a single `DefRule::method`
//! "function_signature" rule covers both — `Owner::EnclosingName` promotes a
//! def inside a `class_declaration` (etc.) to `Method` and leaves a truly
//! top-level def as `Function`. Getter/setter/operator/constructor members are
//! NOT extracted (their signatures are different node kinds), so definition
//! extraction is real but incomplete. No golden-master vs C exists, so it is
//! intentionally NOT claimed as `supported`.
//!
//! CALLS caveat: Dart splits the signature (`function_signature`) from the body
//! (`function_body`) as *siblings* under the declaration, so the def node
//! (`function_signature`) is NOT an ancestor of the call sites in the body. The
//! generic caller-attribution (`spec::enclosing_callable_qname`) resolves a
//! call's *source* function by walking the call's ancestors for a callable
//! DefRule node — which never reaches `function_signature`. The `call_query`
//! below matches callee names correctly, but no CALLS *edge* is emitted
//! (verified: 0 edges on the fixture). Dart call-graph extraction therefore
//! needs an engine change and is NOT yet supported; definitions are the working
//! surface.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// A single rule on `function_signature` (the `@name` identifier's direct
/// parent for both free functions and methods). `Owner::EnclosingName` decides
/// Function vs Method by whether an `owner_kinds` node encloses it.
static DART_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::method("function_signature")],
    owner_kinds: &[
        "class_declaration",
        "mixin_declaration",
        "extension_declaration",
        "enum_declaration",
    ],
    calls: CallSpec { skip_callees: &[] },
    // Dart imports (`import`/`export`/`part`) are not extracted yet
    // (import_query is empty); any variant is inert without a query.
    imports: ImportStrategy::JsTs,
    docs: DocStyle::LineSlashComment,
};

/// Top-level functions and class methods that carry a plain function
/// signature. The `@name` identifier's parent is `function_signature` in both
/// forms. Getters/setters/operators/constructors are intentionally left out
/// (their signatures are different node kinds).
const DEFINITIONS: &str = r#"
    (function_declaration
      signature: (function_signature name: (identifier) @name)) @def

    (method_declaration
      signature: (method_signature
        (function_signature name: (identifier) @name))) @def
"#;

/// Direct `foo(...)` calls and method calls `recv.foo(...)` (including
/// null-aware `recv?.foo(...)`). The callee name is the bare identifier.
const CALLS: &str = r#"
    (call_expression
      function: (identifier) @callee)

    (call_expression
      function: (member_expression property: (identifier) @callee))

    (call_expression
      function: (null_aware_member_expression property: (identifier) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "dart",
        extensions: &["dart"],
        filenames: &[],
        grammar: || tree_sitter_dart::LANGUAGE.into(),
        spec: &DART_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
