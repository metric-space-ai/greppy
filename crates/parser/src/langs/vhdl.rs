//! VHDL — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-vhdl` dependency (a crates.io release, v1.4.0,
//! which builds against tree-sitter 0.25 via the `tree-sitter-language` shim —
//! its accessor is `tree_sitter_vhdl::LANGUAGE`).
//!
//! Status: **experimental / partial**. VHDL is a hardware-description language.
//! Its "definitions" are structural design units and subprograms. The grammar
//! (verified with `examples/dump_vhdl.rs`) exposes these node kinds, each
//! carrying its name on a *field* whose name is the keyword (NOT `name:`):
//!
//!   * `entity_declaration`     — `entity E is …`, name on `entity:` field
//!                                (an `identifier`)                → `Entity`
//!   * `architecture_definition`— `architecture A of E is …`, name on
//!                                `architecture:` (an `identifier`)→ `Architecture`
//!   * `package_declaration`    — `package P is …`, name on `package:`
//!                                (an `identifier`)                → `Package`
//!   * `function_specification` — `function f(…) return T`, name on `function:`
//!                                (an `identifier`)                → `Function`
//!   * `procedure_specification`— `procedure p(…)`, name on `procedure:`
//!                                (an `identifier`)                → `Procedure`
//!
//! Every captured name is a *direct* child of its def node (on a keyword field),
//! so the `Capture` strategy applies uniformly: the def node is the captured
//! name's parent, exactly the node keyed by each `DefRule`. A subprogram's name
//! lives on `function_specification` / `procedure_specification` (which is a
//! child of the `subprogram_definition` wrapper), so those specifier nodes are
//! captured directly as the def nodes (they appear both in a full
//! `subprogram_definition` body and in a `subprogram_declaration` header — both
//! contain the same specifier, so declarations and definitions are captured).
//!
//! CALLS (best-effort, mostly inert): a function call `inc(cnt)` parses as
//! `(name (identifier) @callee (parenthesis_group …))`, and a procedure call as
//! `(procedure_call_statement (name (identifier) @callee (parenthesis_group)))`.
//! The callee query below captures that leading identifier. HOWEVER the engine's
//! source-endpoint resolution (`enclosing_callable_qname` → `callable_name`)
//! looks up the enclosing callable's name via `child_by_field_name("name")`, and
//! VHDL's grammar puts the name on the `function:` / `procedure:` field, NOT
//! `name`. So the enclosing-callable name never resolves and CALLS edges from
//! inside a subprogram body are effectively NOT emitted. This is an honest
//! limitation of the uniform template for a grammar that does not use a `name:`
//! field; the call query is kept so that the pass is correct-by-construction if
//! the template ever generalises, but in practice VHDL CALLS coverage is ~nil.
//!
//! Imprecision / honesty:
//!   * No ownership is modelled (`owner_kinds` empty): the enclosing-owner walk
//!     also relies on a `name:` field VHDL lacks, so subprograms are emitted as
//!     free `Function` / `Procedure`, never as owned methods of their package /
//!     architecture. Names are file-scoped, so same-named subprograms in
//!     different design units share a qname (rare in practice).
//!   * `library` / `use` clauses are NOT expanded into IMPORTS edges: no VHDL
//!     import strategy exists in `ImportStrategy`, so `import_query` is empty.
//!   * Signals, constants, types, components and processes are not surfaced;
//!     only the design units + subprograms above are.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// VHDL definitions are its design units + subprograms. None are owned (VHDL's
/// grammar exposes names on keyword fields, not a `name:` field, so the uniform
/// ownership walk cannot resolve owners), so entities/architectures/packages are
/// `DefRule::ty` and subprograms are `DefRule::func`. `Capture` sets the def node
/// = the `@name` identifier's parent, which is precisely the node keyed here.
/// `function_specification` is `func` (labelled `Function`, callable);
/// `procedure_specification` is `ty` (labelled `Procedure`, not callable) —
/// VHDL procedures are distinct from functions and, since CALLS never resolves
/// (no `name:` field, see below), marking them callable would add nothing.
static VHDL_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("entity_declaration", "Entity"),
        DefRule::ty("architecture_definition", "Architecture"),
        DefRule::ty("package_declaration", "Package"),
        DefRule::func("function_specification"),
        DefRule::ty("procedure_specification", "Procedure"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // VHDL `library` / `use` clauses are not extracted (no VHDL import strategy
    // exists); `import_query` is empty so any variant is inert.
    imports: ImportStrategy::Bash,
    // VHDL line comments use `--`.
    docs: DocStyle::LineDashComment,
};

/// Each def node carries its name on a keyword field (`entity:` /
/// `architecture:` / `package:` / `function:` / `procedure:`), all of which hold
/// an `identifier`. Capture that identifier as `@name`; the engine derives the
/// def node as its parent and keys the matching `DefRule` on that parent's kind.
/// `function_specification` / `procedure_specification` are captured directly
/// (their parent is a `subprogram_definition` / `subprogram_declaration` wrapper
/// that carries no name of its own).
const DEFINITIONS: &str = r#"
    (entity_declaration      entity:       (identifier) @name)
    (architecture_definition architecture: (identifier) @name)
    (package_declaration     package:      (identifier) @name)
    (function_specification  function:     (identifier) @name)
    (procedure_specification procedure:    (identifier) @name)
"#;

/// A subprogram call parses as a `name` whose first child is the callee
/// `identifier` and whose following child is the `parenthesis_group` of
/// arguments: `(name (identifier) @callee (parenthesis_group))`. Anchoring `.`
/// to the first child pins the callee to the leading identifier (so a qualified
/// receiver's leading segment is captured best-effort). NOTE: the engine only
/// emits a CALLS edge when the enclosing callable's name resolves via a `name:`
/// field, which VHDL does not use — so this pass is effectively inert (see the
/// module docs). It is kept correct-by-construction, not for coverage.
const CALLS: &str = r#"
    (name
      . (identifier) @callee
      (parenthesis_group))
"#;

inventory::submit! {
    LangDef {
        name: "vhdl",
        extensions: &["vhd", "vhdl"],
        filenames: &[],
        grammar: || tree_sitter_vhdl::LANGUAGE.into(),
        spec: &VHDL_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
