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
//! A function/macro definition is the body-containing `function_def` /
//! `macro_def` (tagged `@def`). The name is still the first `argument` of the
//! header; that argument's parent (`argument_list`) is not itself a DefRule, so
//! the engine keeps the `@def` ancestor. Calls inside the body source from that
//! function. Command definitions (`set`, `option`, …) stay on the `argument`
//! node and are not callables. Not claimed as `supported` (no verification
//! corpus). Signature-only `function()` declarations without a body are not
//! emitted.

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// Definitions:
///  * `function_def` / `macro_def` — the body-containing command wrapper → `Function`.
///  * `argument` — the def node of a whitelisted command definition (the parent
///    of the captured first `unquoted_argument`) → `Command`.
///
/// No ownership is modelled (CMake has no class/method semantics).
static CMAKE_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function_def"),
        DefRule::func("macro_def"),
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
/// Capture the *first* `argument` (anchored with `.`) as `@name` and the
/// enclosing `function_def` / `macro_def` as `@def`. The name's parent
/// (`argument_list`) has no DefRule, so the engine keeps the body container.
///
/// A command definition (`set(SOURCES …)`) parses as
/// `(normal_command (identifier "set") (argument_list (argument (unquoted_argument "SOURCES")) …))`.
/// Capture the first `unquoted_argument` as `@name` (parent = `argument`, the
/// `DefRule::ty("argument", …)` node), gated to the command names that actually
/// introduce a build variable / target so ordinary calls are not captured.
const DEFINITIONS: &str = r#"
    (function_def
      (function_command
        (argument_list . (argument) @name))) @def
    (macro_def
      (macro_command
        (argument_list . (argument) @name))) @def
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
/// a `Command` def and a callee — harmless. `function()` / `macro()` themselves
/// are not `normal_command`s, so the definition name is not a self-call.
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
