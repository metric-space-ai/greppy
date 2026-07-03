//! CUDA — onboarded via the parallel-safe registry (`crate::registry`). This
//! whole file is the entire surface: it declares the spec + queries + grammar
//! and self-registers with `inventory::submit!`. No shared file is edited
//! (build.rs discovers this module automatically); the only Cargo.toml line
//! added is the `tree-sitter-cuda` dependency (crates.io `v0.21.1`, which builds
//! against tree-sitter 0.25 via the `tree-sitter-language` shim — the same
//! mechanism PureScript / Objective-C use; the accessor is
//! `tree_sitter_cuda::LANGUAGE`).
//!
//! Status: **experimental / partial**. The `tree-sitter-cuda` grammar is a
//! C/C++-derived grammar (verified with `examples/dump_cuda.rs`): CUDA source is
//! parsed exactly like C, with CUDA-specific extras layered on top. A function —
//! including a `__global__` / `__device__` / `__host__` kernel — is a
//! `function_definition` whose name is nested inside
//! `declarator: (function_declarator declarator: (identifier))`, identical to the
//! C grammar this crate already extracts. The `__global__` etc. execution-space
//! qualifiers appear as leading anonymous children of the `function_definition`
//! and are transparent to the declarator walk. So CUDA reuses the `CStructural`
//! name strategy (the def node is captured as `@def`; the name is walked off the
//! declarator by the shared `c_def_name` resolver) and the C `#include` import
//! expander.
//!
//! What is captured (all exactly as for C — same node kinds):
//!   * `function_definition` — a C-style function `__global__ void k(…){…}` or a
//!                             plain host function → `Function`. Its name resolves
//!                             structurally and it exposes a declarator, so CALLS
//!                             edges whose source is such a function ARE resolved.
//!   * `struct_specifier`    — `struct Point { … };`            → `Struct`
//!   * `union_specifier`     — `union U { … };`                 → `Union`
//!   * `enum_specifier`      — `enum E { … };`                  → `Enum`
//!   * `type_definition`     — `typedef unsigned int uint32;`   → `Type`
//!   * `preproc_include`     — `#include <…>` / `#include "…"`  → `IMPORTS` edges.
//!
//! CALLS: a C-style call `square(n)` parses as `(call_expression function:
//! (identifier) @callee …)`. A CUDA kernel launch `addKernel<<<1,n>>>(…)` ALSO
//! parses as a `call_expression` with `function: (identifier "addKernel")` (the
//! `<<<…>>>` execution-configuration is a separate `kernel_call_syntax` child, so
//! the callee identifier is still captured). The engine hangs the CALLS edge off
//! the enclosing `function_definition` (which exposes a declarator, so the source
//! endpoint resolves). In the fixture `host()` calls both `addKernel<<<…>>>(…)`
//! and `square(…)`, so CALLS edges are produced.
//!
//! Honesty / omissions:
//!   * Method / class ownership is NOT modelled (`owner_kinds` empty). A CUDA C++
//!     class member function nested in a `class_specifier` is captured as a free
//!     `Function` (its structural name resolves), not a `Method` owned by the
//!     class. Out-of-line `Class::method` qualifiers are not resolved to an owner
//!     here (unlike the bespoke C++ path, which the uniform template does not
//!     express). This mirrors how `objc.rs` treats C-family functions.
//!   * A `struct`/`class` DEFINED at file scope is captured; C++ `class_specifier`
//!     is intentionally NOT in the def set (kept identical to the C rule set) —
//!     add it if C++-style CUDA classes need surfacing.
//!   * The `<<<…>>>` launch configuration itself is not surfaced as data; only the
//!     launched kernel name becomes a CALLS callee.
//!
//! Not claimed as `supported` (no golden-master vs C).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// CUDA is a C/C++-derived grammar, so the `CStructural` strategy applies: each
/// def node is captured as `@def` and its name is walked off the declarator (for
/// functions/typedefs) or read from the `name:` field (for tagged types) by the
/// shared `c_def_name` resolver. The rule set mirrors C's exactly — functions,
/// tagged types, and typedefs. No class/method ownership is modelled (CUDA C++
/// classes are captured as free functions — see the module docs).
static CUDA_SPEC: LangSpec = LangSpec {
    name: NameStrategy::CStructural,
    defs: &[
        DefRule::func("function_definition"),
        DefRule::ty("struct_specifier", "Struct"),
        DefRule::ty("union_specifier", "Union"),
        DefRule::ty("enum_specifier", "Enum"),
        DefRule::ty("type_definition", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // `#include` parses as `preproc_include`; reuse the C include expander (it
    // keys on the `include` capture name below).
    imports: ImportStrategy::C,
    // CUDA uses C comment syntax: `/* … */` blocks and `//` lines.
    docs: DocStyle::CBlockOrLine,
};

/// Capture each definition node as `@def`; the `CStructural` strategy walks a
/// `function_definition`'s `declarator: (function_declarator declarator:
/// (identifier))` for the name, reads the `name:` field of a tagged type, or
/// walks a `type_definition`'s declarator — then keys the matching `DefRule` on
/// the node's kind.
const DEFINITIONS: &str = r#"
    (function_definition) @def
    (struct_specifier    name: (type_identifier)) @def
    (union_specifier     name: (type_identifier)) @def
    (enum_specifier      name: (type_identifier)) @def
    (type_definition) @def
"#;

/// A C-style call `square(n)` parses as `(call_expression function: (identifier)
/// @callee arguments: (argument_list))`. A CUDA kernel launch
/// `addKernel<<<1,n>>>(…)` also parses as a `call_expression` whose `function:`
/// is the launched-kernel `identifier`, so keying the callee on `function:
/// (identifier)` captures both. The engine hangs the CALLS edge off the enclosing
/// `function_definition` (which exposes a declarator, so the source resolves).
const CALLS: &str = r#"
    (call_expression
      function: (identifier) @callee)
"#;

/// `#include <cuda_runtime.h>` / `#include "kernel.cuh"` parse as
/// `(preproc_include path: …)`; capture the whole directive as `@include` so the
/// C import expander (keyed on the `include` capture name) turns it into an
/// `IMPORTS` edge.
const IMPORTS: &str = r#"
    (preproc_include) @include
"#;

inventory::submit! {
    LangDef {
        name: "cuda",
        extensions: &["cu", "cuh"],
        filenames: &[],
        grammar: || tree_sitter_cuda::LANGUAGE.into(),
        spec: &CUDA_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: IMPORTS,
    }
}
