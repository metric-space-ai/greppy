//! Elm — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-elm` dependency.
//!
//! Status: **experimental**. The `tree-sitter-elm` grammar models a top-level
//! value/function `f a b = ...` as a `value_declaration` whose name lives in a
//! nested `function_declaration_left` (as the FIRST positional
//! `lower_case_identifier`, NOT a `name:` field). With the `Capture` strategy
//! the definition node is therefore that `function_declaration_left` (the parent
//! of the captured name), which is enough to emit Function nodes. Type/type-alias
//! declarations expose a real `name:` field and become `Type` / `TypeAlias`.
//!
//! Because `function_declaration_left` has no `name:` field, the engine's
//! enclosing-callable resolution (which reads `child_by_field_name("name")`)
//! cannot recover a Function's own name, so CALLS edges whose SOURCE is an Elm
//! function are NOT resolved (same limitation as Julia). The CALLS query still
//! captures the correct callee identifier for every `function_call_expr`. Not
//! claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `function_declaration_left` — top-level `f a b = ...` values/functions →
///    `Function` (the def node is the name identifier's parent).
///  * `type_declaration` — `type Foo = ...` custom types → `Type`.
///  * `type_alias_declaration` — `type alias Bar = ...` → `TypeAlias`.
///
/// All three route their captured `@name` identifier's parent to one of these
/// kinds, so the `Capture` strategy (name = `@name`, def = its parent) applies
/// uniformly.
static ELM_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function_declaration_left"),
        DefRule::ty("type_declaration", "Type"),
        DefRule::ty("type_alias_declaration", "TypeAlias"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Elm imports (`import Foo exposing (..)`) are not extracted yet
    // (import_query is empty); any variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineDashComment,
};

/// A top-level `add a b = ...` parses as
/// `(value_declaration (function_declaration_left (lower_case_identifier) @name
/// (lower_pattern ...)*))`; the function name is the FIRST child identifier of
/// `function_declaration_left` (parameters are wrapped in `lower_pattern`, so
/// they are not direct `lower_case_identifier` children and do not match). The
/// anchor `.` pins the capture to that first identifier. Type declarations carry
/// their name in a real `name:` field.
const DEFINITIONS: &str = r#"
    (function_declaration_left
      .
      (lower_case_identifier) @name)

    (type_declaration
      name: (upper_case_identifier) @name)

    (type_alias_declaration
      name: (upper_case_identifier) @name)
"#;

/// A call `add x x` parses as `(function_call_expr target: (value_expr name:
/// (value_qid (lower_case_identifier) @callee)) arg: ...)`. Capturing the
/// `lower_case_identifier` in the `target`'s `value_qid` yields the callee name.
const CALLS: &str = r#"
    (function_call_expr
      target: (value_expr
        name: (value_qid (lower_case_identifier) @callee)))
"#;

inventory::submit! {
    LangDef {
        name: "elm",
        extensions: &["elm"],
        filenames: &[],
        grammar: || tree_sitter_elm::LANGUAGE.into(),
        spec: &ELM_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
