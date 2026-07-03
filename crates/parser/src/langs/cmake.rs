//! CMake — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-cmake` dependency.
//!
//! Status: **experimental / partial**. CMake is a hybrid: it has real
//! user-defined callables (`function(name …) … endfunction()` and
//! `macro(name …) … endmacro()`) *and* a large surface of command-style
//! "definitions" (`set`, `option`, `project`, `add_library`,
//! `add_executable`, `add_custom_target`) that name build variables/targets.
//! The `tree-sitter-cmake` grammar carries NONE of these names on a `name:`
//! field — every command's arguments live positionally inside an
//! `argument_list`, and a `function`/`macro` header is itself just a
//! `function_command` / `macro_command` whose first `argument` is the name.
//!
//! With the `Capture` name strategy the definition node is the *parent* of the
//! captured `@name` node, so the two def families are separated by capturing at
//! two different depths so their parents differ:
//!
//!   * function / macro name — capture the whole first `argument`; its parent is
//!     the `argument_list`, so the def node is `argument_list` → `Function`.
//!   * command name (set / add_library / …) — capture the first
//!     `unquoted_argument`; its parent is the `argument`, so the def node is
//!     `argument` → `Command`.
//!
//! IMPRECISION (honest): because a function/macro's def node is its *name*
//! `argument_list` (not the enclosing `function_def` that contains the `body`),
//! the body's call sites are NOT descendants of the def node. The engine's
//! enclosing-callable walk therefore never attributes a CALLS edge to a CMake
//! function (same limitation the module notes for Julia), so callee identifiers
//! are recognised by the CALLS query but **0 CALLS edges are emitted**. The
//! grammar exposes no distinct name node under `function_def`/`macro_def` that
//! would let `Capture` pick the body-containing node as the def, so this is a
//! deliberate trade to keep names + node kinds correct. Definition spans are
//! the name line only. Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `argument_list` — the def node of a `function`/`macro` header (the parent
///    of the captured first `argument`) → `Function`.
///  * `argument` — the def node of a whitelisted command definition (the parent
///    of the captured first `unquoted_argument`) → `Command`.
///
/// No ownership is modelled (CMake has no class/method semantics). The
/// `Function` rule is `func` (callable) so callee resolution *would* attribute
/// to it if a call were a descendant of its def node — it is not (see the
/// module note), so no CALLS edges result, but keeping it callable is correct.
static CMAKE_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("argument_list"),
        DefRule::ty("argument", "Command"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // CMake `include()` / `add_subdirectory()` are not extracted as imports (no
    // CMake import strategy exists); import_query is empty so any variant is
    // inert. Pick one arbitrarily.
    imports: ImportStrategy::Bash,
    // CMake comments start with `#`.
    docs: DocStyle::LineHashComment,
};

/// `function(greet name)` parses as
/// `(function_def (function_command (argument_list (argument (unquoted_argument "greet")) …)))`.
/// Capture the *first* `argument` (anchored with `.`) as `@name`; its parent is
/// the `argument_list`, which is the `DefRule::func("argument_list")` node.
///
/// A command definition (`set(SOURCES …)`) parses as
/// `(normal_command (identifier "set") (argument_list (argument (unquoted_argument "SOURCES")) …))`.
/// Capture the first `unquoted_argument` as `@name` (parent = `argument`, the
/// `DefRule::ty("argument", …)` node), gated to the command names that actually
/// introduce a build variable / target so ordinary calls are not captured.
const DEFINITIONS: &str = r#"
    (function_def
      (function_command
        (argument_list . (argument) @name)))
    (macro_def
      (macro_command
        (argument_list . (argument) @name)))
    ((normal_command
       (identifier) @_cmd
       (argument_list . (argument (unquoted_argument) @name)))
      (#any-of? @_cmd
        "set" "option" "project"
        "add_library" "add_executable" "add_custom_target"))
"#;

/// Every command invocation is `(normal_command (identifier) @callee …)`; the
/// callee is that leading identifier. This captures `message(…)`, a call to a
/// user `function`/`macro`, and the built-in commands alike (best-effort). The
/// def-introducing commands are NOT excluded here, so e.g. `set` appears both as
/// a `Command` def and a callee — harmless, and the enclosing-callable walk
/// never resolves a CMake call to a source function anyway (see module note).
const CALLS: &str = r#"
    (normal_command
      (identifier) @callee)
"#;

inventory::submit! {
    LangDef {
        name: "cmake",
        extensions: &["cmake"],
        filenames: &[],
        grammar: || tree_sitter_cmake::LANGUAGE.into(),
        spec: &CMAKE_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: "",
    }
}
