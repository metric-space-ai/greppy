//! HCL (HashiCorp Configuration Language, incl. Terraform `.tf`) — onboarded
//! via the parallel-safe registry (`crate::registry`). This whole file is the
//! entire surface: it declares the spec + queries + grammar and self-registers
//! with `inventory::submit!`. No shared file is edited (build.rs discovers this
//! module automatically); the only Cargo.toml line added is the
//! `tree-sitter-hcl` dependency.
//!
//! Status: **experimental / partial**. HCL is a configuration/data language,
//! not a programming language: its unit of structure is the *block*
//! (`resource "aws_instance" "web" { … }`, `variable "env" { … }`,
//! `module "vpc" { … }`, `locals { … }`). The `tree-sitter-hcl` grammar models
//! every one of these as a single `block` node whose FIRST child is an
//! `identifier` naming the block *type* (`resource` / `variable` / `module` /
//! `locals` / `provider` / …), optionally followed by one or more `string_lit`
//! labels. There are no `function_definition`-style nodes.
//!
//! What the registry *can* surface — and what makes an HCL file greppable as
//! structure — are those top-level `block` nodes, captured as definitions via
//! `DefRule::ty`. The grammar exposes NO `name:` field on `block` (its fields
//! map is empty; the type keyword and labels are anonymous positional
//! children), so with the `Capture` name strategy the definition node is the
//! *parent* of the captured `identifier` — precisely the `block` node — and the
//! extracted name is the block *type keyword* (`resource`, `variable`, …), not
//! the block's label. That is a real imprecision: `resource "aws_instance"
//! "web"` and `resource "aws_s3_bucket" "logs"` both surface as name
//! `"resource"` (distinguished only by their line range and qname collision is
//! avoided by nothing — this is best-effort structural extraction). It is
//! intentionally NOT claimed as `supported` (no golden-master vs C).
//!
//! CALLS: HCL expressions can invoke built-in / provider functions
//! (`length(var.subnets)`, `cidrsubnet(var.base_cidr, 8, 1)`), which the
//! grammar parses as a `function_call` whose first child `identifier` is the
//! callee. Those are extracted as CALLS edges (best-effort). Because a `block`
//! is captured as a *type* (`callable == false`), a call's source endpoint
//! resolves to the nearest enclosing callable — of which HCL has none — so
//! CALLS edges are only emitted when such an enclosing callable exists; in
//! practice HCL yields definition nodes and the raw `function_call` sites, and
//! the callee identifier is what is greppable.
//!
//! IMPORTS: HCL has no import statement (a `module` block's `source` is a plain
//! attribute, not a syntactic import), so the IMPORTS pass is inert
//! (import_query is empty).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// HCL's only structural definition is the `block`. It is not callable and not
/// owned (HCL has no method/class semantics), so the rule is a `DefRule::ty`.
/// `Capture` sets the def node = the captured `identifier`'s parent, which is
/// exactly the `block` node keyed here; the name is the block *type keyword*.
static HCL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::ty("block", "Block")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // HCL has no import syntax; the IMPORTS pass is inert (import_query is
    // empty). Any variant is dead weight without a query — pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // HCL comments are `#` (and `//` / `/* */`); the line-comment-run helper
    // keys on `#`, so use LineHashComment for the `#` doc style.
    docs: DocStyle::LineHashComment,
};

/// Capture each `block`'s leading type-keyword `identifier` as `@name`; the
/// engine derives the def node as that identifier's parent (`block`) and keys
/// the `DefRule::ty("block", …)` on it. The `identifier` is a *direct* child of
/// `block`, so its `.parent()` is the `block` itself. Nested `attribute` /
/// `function_call` identifiers are never captured here because they are not
/// direct children of a `block`.
const DEFINITIONS: &str = r#"
    (block (identifier) @name)
"#;

/// A function invocation `foo(...)` parses as `(function_call (identifier)
/// @callee (function_arguments …))`; the callee is that leading `identifier`.
/// It is a direct child of `function_call`, so nested argument identifiers are
/// not captured.
const CALLS: &str = r#"
    (function_call (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "hcl",
        extensions: &["hcl", "tf"],
        filenames: &[],
        grammar: || tree_sitter_hcl::LANGUAGE.into(),
        spec: &HCL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
