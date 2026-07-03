//! Objective-C — onboarded via the parallel-safe registry (`crate::registry`).
//! This whole file is the entire surface: it declares the spec + queries +
//! grammar and self-registers with `inventory::submit!`. No shared file is
//! edited (build.rs discovers this module automatically); the only Cargo.toml
//! line added is the `tree-sitter-objc` dependency (crates.io `v3.0.2`, which
//! builds against tree-sitter 0.25 via the `tree-sitter-language` shim — the
//! same mechanism PureScript uses; the older `3.0.0`/`2.x`/`1.x` releases pin
//! `tree-sitter ~0.20.10` and do NOT build here).
//!
//! Status: **golden-master parity with the C reference**. The uniform registry
//! template below (a `CStructural` `function_definition` def rule plus the C
//! include expander) cannot express Objective-C's taxonomy: `@interface` /
//! `@implementation` / `@protocol` and `- (…)method` definitions carry their
//! name on an anonymous `identifier` child (no `name:` field), and the CALLS
//! model is the `[receiver message:…]` `message_expression`, not
//! `call_expression`. So Objective-C is routed to a bespoke pass —
//! `extract::extract_objc`, a faithful port of the C reference's `CBM_LANG_OBJC`
//! def / method / usage passes — which reaches C golden-master parity on
//! `bench/agent_efficiency/corpus/objc_small` for every in-scope node label and
//! edge type (Class 7, Interface 2, Method 26; DEFINES 46, DEFINES_METHOD 26,
//! IMPORTS 15, USAGE 48, plus the resolvable CALLS). See the `extract_objc`
//! module docs for the full mapping and the out-of-scope rows (cross-file
//! ambiguous CALLS / INHERITS, SEMANTICALLY_RELATED).
//!
//! What `extract_objc` captures (this file's queries below are retained only for
//! grammar/extension registration; the bespoke pass supersedes them):
//!   * `class_interface` / `class_implementation` → `Class` (collapsed by qname)
//!   * `protocol_declaration`                     → `Interface`
//!   * `method_definition` (in `@implementation`) → `Method` + `DEFINES_METHOD`
//!   * `message_expression`                       → `CALLS` (selector callee)
//!   * every reference identifier                 → `USAGE`
//!   * `preproc_include` (`#import` / `#include`) → `IMPORTS`
//!   * free `function_definition`                 → NO node (C emits zero
//!                                                  Function/Field/Variable)

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// C-style `function_definition`s become `Function` definitions. The grammar is
/// C-derived, so the `CStructural` strategy applies: the whole def node is
/// captured as `@def` and the name is walked off its `function_declarator` by
/// the shared `c_def_name` resolver. No class/method ownership is modelled
/// (Objective-C classes are not captured — see the module docs).
static OBJC_SPEC: LangSpec = LangSpec {
    name: NameStrategy::CStructural,
    defs: &[DefRule::func("function_definition")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // `#import` / `#include` both parse as `preproc_include`; reuse the C
    // include expander (it keys on the `include` capture name below).
    imports: ImportStrategy::C,
    // Objective-C uses C comment syntax: `/* … */` blocks and `//` lines.
    docs: DocStyle::CBlockOrLine,
};

/// Capture each `function_definition` as `@def`; the `CStructural` strategy
/// walks its `declarator: (function_declarator declarator: (identifier))` to
/// obtain the name and keys `DefRule::func("function_definition")` on the node.
const DEFINITIONS: &str = r#"
    (function_definition) @def
"#;

/// A C-style call `add(a, b)` parses as `(call_expression function: (identifier)
/// @callee arguments: (argument_list))`. The engine hangs the CALLS edge off the
/// enclosing `function_definition` (which exposes a declarator, so the source
/// endpoint resolves).
const CALLS: &str = r#"
    (call_expression
      function: (identifier) @callee)
"#;

/// `#import <Foundation/Foundation.h>` / `#include "x.h"` parse as
/// `(preproc_include path: …)`; capture the whole directive as `@include` so the
/// C import expander (keyed on the `include` capture name) turns it into an
/// `IMPORTS` edge.
const IMPORTS: &str = r#"
    (preproc_include) @include
"#;

inventory::submit! {
    LangDef {
        name: "objc",
        extensions: &["m"],
        filenames: &[],
        grammar: || tree_sitter_objc::LANGUAGE.into(),
        spec: &OBJC_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: IMPORTS,
    }
}
