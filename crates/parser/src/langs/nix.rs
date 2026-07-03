//! Nix — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-nix` dependency.
//!
//! Status: **experimental**. Nix is an expression/config language: every
//! top-level definition — whether it holds a function (`add = a: b: ...`) or a
//! plain value (`greeting = "hello"`) — is a `binding` node of the form
//! `attrpath = expr ;`. The tree-sitter-nix grammar carries the bound name in
//! the `attrpath:` field (a dotted path), NOT in a `name:` field, so with the
//! `Capture` strategy the `@name` capture is the `attrpath` and its parent (the
//! `binding`) is the definition node. Bindings are modelled as `Function`
//! definitions (a Nix binding is the closest analogue to a top-level def and is
//! frequently a lambda). Function application `f x y` parses as nested
//! `apply_expression`s whose head `variable_expression` names the callee, so
//! CALLS targets are extracted. However, because a `binding` has no `name:`
//! field, the engine's enclosing-callable resolution (which reads `name:`)
//! cannot attach a CALLS edge's *source* to the enclosing binding — CALLS edges
//! are therefore emitted only when the enclosing callable exposes a `name:`
//! field, which Nix bindings do not, so most Nix call sites contribute a callee
//! target without a resolved source. This is a best-effort heuristic (no
//! golden-master vs C); it is intentionally NOT claimed as `supported`.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// A Nix `binding` (`attrpath = expr ;`) is the unit of definition — inside a
/// `let ... in` block, an attribute set, or a `rec { }`. Each is treated as a
/// `Function` definition (Nix has no distinct def kind; bindings routinely hold
/// lambdas, and this keeps callable-source resolution best-effort rather than
/// discarding it). The name is the `attrpath` (`@name`); its parent `binding`
/// is the def node, so the `DefRule` keys on `"binding"`.
static NIX_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("binding")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // Nix imports use `import ./path` expressions rather than a distinct import
    // statement kind; they are not extracted yet (import_query is empty), so any
    // variant is inert without a query.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// `name = expr ;` parses as `(binding attrpath: (attrpath (attr: (identifier)))
/// = (expression))`. Capture the `attrpath` as `@name` (its text is the bound
/// name, e.g. `add`), so the engine derives the def node as its parent — the
/// `binding` — and keys the `DefRule` on `"binding"`.
const DEFINITIONS: &str = r#"
    (binding
      attrpath: (attrpath) @name) @def
"#;

/// Function application `f x y` parses as nested `apply_expression`s:
/// `(apply_expression function: (apply_expression function: (variable_expression
/// (identifier "f") ...) ...) ...)`. Capturing the `identifier` inside the head
/// `variable_expression` in `function:` position yields the callee name. This
/// fires once per application level, naming each function that is applied.
const CALLS: &str = r#"
    (apply_expression
      function: (variable_expression (identifier) @callee))
"#;

inventory::submit! {
    LangDef {
        name: "nix",
        extensions: &["nix"],
        filenames: &[],
        grammar: || tree_sitter_nix::LANGUAGE.into(),
        spec: &NIX_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
