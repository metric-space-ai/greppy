//! Data-driven language onboarding.
//!
//! The eight non-Rust languages this crate extracts (Python, JavaScript,
//! TypeScript, Go, Ruby, Java, C, C++) all follow the **same** three-pass
//! extraction template:
//!
//!   * PASS 1 — DEFINITIONS: a small set of grammar node-kinds map to labels
//!     (`Function` / `Method` / `Class` / …). A name is read off the node, and
//!     when the node is a function/method it is *owned* by an enclosing
//!     class/type so two same-named methods do not collide on
//!     `{file}::Function::{name}`.
//!   * PASS 2 — CALLS: a `@callee` capture names the final callee identifier; a
//!     `CALLS` edge is emitted from the enclosing function/method qname to that
//!     name, keyed on `callee_name` for the cross-file resolver.
//!   * PASS 3 — IMPORTS: an import/include/require statement expands into one
//!     `IMPORTS` edge per bound name, keyed on `imported_name`.
//!
//! Reaching 156 languages by hand-writing one bespoke `extract_*` per language
//! is infeasible. This module captures that uniform template as a declarative
//! [`LangSpec`] so a *new* uniform language is config, not code: a slice of
//! [`DefRule`]s, a [`CallSpec`], an [`ImportStrategy`], and a [`DocStyle`]. The
//! generic [`spec_extract`] consumes a `LangSpec` and produces exactly the same
//! `ExtractedNode` / `ExtractedEdge` output the per-language extractors produce
//! — the eight existing languages are migrated onto it byte-for-byte (their
//! ~131 tests stay green unchanged), and three new languages (C#, PHP, Bash)
//! are onboarded purely as specs.
//!
//! Rust is intentionally **not** modelled here: it keeps its bespoke seven-pass
//! path (type-refs, usages, type-assigns, inheritance) which the uniform
//! template does not express.

use serde_json::json;
use tree_sitter::{Node, QueryCursor, StreamingIterator};

use crate::extract::{
    docstring_summary, node_text, ExtractedEdge, ExtractedNode, ExtractionResult, ImportedItem,
    MAX_COMMENT_LEN,
};
use crate::language::Language;
use crate::query::{CompiledQuery, QueryKind};

/// How a definition node's *name* is obtained from a Definitions-query match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameStrategy {
    /// The query captures the name node as `@name`; the definition node is its
    /// parent. Used by every language whose grammar exposes a `name:` field
    /// (Python, JS/TS, Go, Ruby, Java, C#, PHP, Bash).
    Capture,
    /// The query captures the definition node as `@def`; the name is resolved
    /// structurally by walking the declarator. Used by C / C++, whose function
    /// name is nested inside a `function_declarator` (possibly behind pointer /
    /// qualified declarators) and whose tagged-type name is a field on the
    /// specifier.
    CStructural,
    /// The query captures the whole assignment node as `@def`; the name is the
    /// left-hand `identifier` of an `name <- function(...)` binding. Used by R,
    /// whose functions are anonymous `function_definition`s bound by a
    /// `binary_operator` assignment (the name is the binding's left sibling, not
    /// a child of the function).
    RAssign,
}

/// How a definition is *owned* by an enclosing scope, which decides its qname
/// (`{file}::{Owner}::{name}` for an owned member vs `{file}::{segment}::{name}`
/// for a free function / a type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    /// Never owned. Top-level types (classes, interfaces, enums, structs,
    /// namespaces) and free functions in languages where functions are never
    /// methods (Bash).
    None,
    /// Owned by the nearest enclosing node listed in [`LangSpec::owner_kinds`],
    /// via that node's `name:` field. Covers Python, JS/TS, Ruby, Java, C#, PHP.
    EnclosingName,
    /// Owned by the method's *receiver* type (Go `method_declaration`).
    GoReceiver,
    /// Owned by the enclosing C++ class/struct, OR by an out-of-line
    /// `Class::method` qualifier on the declarator (resolved structurally).
    CppClass,
}

/// One definition rule: a grammar node-kind, its label, and ownership.
#[derive(Debug, Clone, Copy)]
pub struct DefRule {
    /// The grammar node kind of the definition (`"function_definition"`,
    /// `"class_declaration"`, …).
    pub node_kind: &'static str,
    /// The label/qname-segment used when the node is *not* owned (a free
    /// function or a type): `"Function"`, `"Class"`, `"Interface"`, …
    pub label: &'static str,
    /// The label applied when the node *is* owned by an enclosing scope
    /// (usually `"Method"`). Only consulted when `owner != Owner::None`.
    pub method_label: &'static str,
    /// The ownership rule.
    pub owner: Owner,
    /// Whether this rule names a *function-like* definition. The CALLS pass
    /// hangs a call's source endpoint off the nearest enclosing callable; types
    /// set this `false`.
    pub callable: bool,
}

impl DefRule {
    /// A top-level type definition (class/struct/enum/interface/namespace):
    /// never owned, not callable.
    pub const fn ty(node_kind: &'static str, label: &'static str) -> Self {
        DefRule {
            node_kind,
            label,
            method_label: label,
            owner: Owner::None,
            callable: false,
        }
    }

    /// A free function that is never a method (no ownership).
    pub const fn func(node_kind: &'static str) -> Self {
        DefRule {
            node_kind,
            label: "Function",
            method_label: "Function",
            owner: Owner::None,
            callable: true,
        }
    }

    /// A function/method owned by an enclosing class/type via its `name:` field.
    pub const fn method(node_kind: &'static str) -> Self {
        DefRule {
            node_kind,
            label: "Function",
            method_label: "Method",
            owner: Owner::EnclosingName,
            callable: true,
        }
    }
}

/// The CALLS pass configuration.
#[derive(Debug, Clone, Copy)]
pub struct CallSpec {
    /// Callee identifiers to *skip* (so they are not double-counted as calls):
    /// e.g. `require` / `require_relative`, which the imports pass owns.
    pub skip_callees: &'static [&'static str],
}

/// The IMPORTS pass strategy — which expander turns an import-bearing capture
/// into [`ImportedItem`]s, and the `kind` string stamped on the emitted nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportStrategy {
    /// Python `import` / `from x import y` (multi-binding, globs).
    Python,
    /// JS/TS ES `import` statements + CommonJS `require()` declarators.
    JsTs,
    /// Go `import_spec` (one package per spec).
    Go,
    /// Ruby `require` / `require_relative` calls.
    Ruby,
    /// Java `import_declaration` (final segment).
    Java,
    /// C `#include` only.
    C,
    /// C++ `#include` + `using` declarations.
    Cpp,
    /// C# `using_directive`.
    CSharp,
    /// PHP `namespace_use_declaration`.
    Php,
    /// Bash `source` / `.` builtins.
    Bash,
    /// Lua `require("…")` calls.
    Lua,
    /// Kotlin `import` directives (final qualified-name segment).
    Kotlin,
    /// Scala `import_declaration` (final `path:` segment).
    Scala,
    /// Swift `import_declaration` (module identifier).
    Swift,
    /// Zig `@import("…")` builtin calls.
    Zig,
    /// R `library(…)` / `require(…)` calls.
    R,
}

/// How a definition's docstring is extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocStyle {
    /// No docstrings.
    None,
    /// The first bare string statement in the definition body (Python).
    PythonBodyString,
    /// A run of leading `//` line `comment` nodes (Go).
    LineSlashComment,
    /// A leading `/** … */` block comment, JSDoc-style (JS/TS, Java).
    BlockJsdoc,
    /// A run of leading `#` line `comment` nodes (Ruby, Bash, PHP `#`, R).
    LineHashComment,
    /// A leading `/* */` block OR a run of `//` line comments (C/C++, C#, PHP).
    CBlockOrLine,
    /// A run of leading `--` line `comment` nodes (Lua).
    LineDashComment,
}

/// A declarative description of a uniform language's extraction template.
///
/// This is the whole onboarding surface: to add a new uniform language you
/// build its three query sources (Definitions/Calls/Imports — see
/// [`crate::query`]) and one `LangSpec`. No bespoke `extract_*` function.
#[derive(Debug, Clone, Copy)]
pub struct LangSpec {
    /// How definition names are obtained (PASS 1).
    pub name: NameStrategy,
    /// Definition rules (PASS 1).
    pub defs: &'static [DefRule],
    /// Enclosing-owner node kinds for [`Owner::EnclosingName`]. The definition
    /// pass and the CALLS source walk both consult this so they agree on which
    /// class/type owns a member.
    pub owner_kinds: &'static [&'static str],
    /// CALLS configuration (PASS 2).
    pub calls: CallSpec,
    /// IMPORTS strategy (PASS 3).
    pub imports: ImportStrategy,
    /// Docstring style.
    pub docs: DocStyle,
}

// ---------------------------------------------------------------------------
// Language specs — the declarative onboarding surface
// ---------------------------------------------------------------------------
//
// Each `LangSpec` below fully describes a uniform language's extraction. The
// eight migrated languages reproduce their previous bespoke output exactly; the
// three new languages (C#, PHP, Bash) are onboarded purely as specs.

/// Python: `function_definition` / `class_definition`; methods owned by their
/// enclosing class; first-body-string docstrings.
pub const PYTHON: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_definition", "Class"),
        DefRule::method("function_definition"),
    ],
    owner_kinds: &["class_definition"],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Python,
    docs: DocStyle::PythonBodyString,
};

/// JavaScript: functions, classes, methods, arrow/function bindings; methods
/// owned by their class; JSDoc docstrings; `require` callee skipped.
pub const JAVASCRIPT: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function_declaration"),
        DefRule::ty("class_declaration", "Class"),
        DefRule::method("method_definition"),
        DefRule::method("variable_declarator"),
    ],
    owner_kinds: &["class_declaration", "class"],
    calls: CallSpec {
        skip_callees: &["require"],
    },
    imports: ImportStrategy::JsTs,
    docs: DocStyle::BlockJsdoc,
};

/// TypeScript: the JavaScript rules plus `interface` / `type` / `enum`.
pub const TYPESCRIPT: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function_declaration"),
        DefRule::ty("class_declaration", "Class"),
        DefRule::method("method_definition"),
        DefRule::method("variable_declarator"),
        DefRule::ty("interface_declaration", "Interface"),
        DefRule::ty("type_alias_declaration", "Type"),
        DefRule::ty("enum_declaration", "Enum"),
    ],
    owner_kinds: &["class_declaration", "class"],
    calls: CallSpec {
        skip_callees: &["require"],
    },
    imports: ImportStrategy::JsTs,
    docs: DocStyle::BlockJsdoc,
};

/// Go: functions, receiver-owned methods, and `type_spec` struct/interface/type.
pub const GO: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::func("function_declaration"),
        DefRule {
            node_kind: "method_declaration",
            label: "Function",
            method_label: "Method",
            owner: Owner::GoReceiver,
            callable: true,
        },
        DefRule::ty("type_spec", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Go,
    docs: DocStyle::LineSlashComment,
};

/// Ruby: methods/singleton-methods owned by class/module; classes/modules;
/// `require`/`require_relative` callees skipped; `#` docstrings.
pub const RUBY: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class", "Class"),
        DefRule::ty("module", "Module"),
        DefRule::method("method"),
        DefRule::method("singleton_method"),
    ],
    owner_kinds: &["class", "module"],
    calls: CallSpec {
        skip_callees: &["require", "require_relative"],
    },
    imports: ImportStrategy::Ruby,
    docs: DocStyle::LineHashComment,
};

/// Java: class/interface/enum; methods/constructors owned by their type;
/// Javadoc docstrings.
pub const JAVA: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_declaration", "Class"),
        DefRule::ty("interface_declaration", "Interface"),
        DefRule::ty("enum_declaration", "Enum"),
        DefRule::method("method_declaration"),
        DefRule::method("constructor_declaration"),
    ],
    owner_kinds: &[
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
    ],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Java,
    docs: DocStyle::BlockJsdoc,
};

/// C: functions, tagged types, typedefs; structural names; `#include` imports.
pub const C: LangSpec = LangSpec {
    name: NameStrategy::CStructural,
    defs: &[
        DefRule {
            node_kind: "function_definition",
            label: "Function",
            method_label: "Method",
            owner: Owner::CppClass,
            callable: true,
        },
        DefRule::ty("struct_specifier", "Struct"),
        DefRule::ty("union_specifier", "Union"),
        DefRule::ty("enum_specifier", "Enum"),
        DefRule::ty("type_definition", "Type"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::C,
    docs: DocStyle::CBlockOrLine,
};

/// C++: the C rules plus `class_specifier` / `namespace_definition`; out-of-line
/// `Class::method` ownership; `#include` + `using` imports.
pub const CPP: LangSpec = LangSpec {
    name: NameStrategy::CStructural,
    defs: &[
        DefRule {
            node_kind: "function_definition",
            label: "Function",
            method_label: "Method",
            owner: Owner::CppClass,
            callable: true,
        },
        DefRule::ty("struct_specifier", "Struct"),
        DefRule::ty("union_specifier", "Union"),
        DefRule::ty("enum_specifier", "Enum"),
        DefRule::ty("type_definition", "Type"),
        DefRule::ty("class_specifier", "Class"),
        DefRule::ty("namespace_definition", "Namespace"),
    ],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Cpp,
    docs: DocStyle::CBlockOrLine,
};

/// C# (new, data-path only): class/struct/interface/record/enum; methods +
/// constructors owned by their type; `using` directives; `///` / `/** */` docs.
pub const CSHARP: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_declaration", "Class"),
        DefRule::ty("struct_declaration", "Struct"),
        DefRule::ty("interface_declaration", "Interface"),
        DefRule::ty("record_declaration", "Record"),
        DefRule::ty("enum_declaration", "Enum"),
        DefRule::method("method_declaration"),
        DefRule::method("constructor_declaration"),
    ],
    owner_kinds: &[
        "class_declaration",
        "struct_declaration",
        "interface_declaration",
        "record_declaration",
    ],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::CSharp,
    docs: DocStyle::CBlockOrLine,
};

/// PHP (new, data-path only): class/interface/trait/enum; methods/functions
/// owned by their type; `use` imports; `/** */` / `//` / `#` docs.
pub const PHP: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_declaration", "Class"),
        DefRule::ty("interface_declaration", "Interface"),
        DefRule::ty("trait_declaration", "Trait"),
        DefRule::ty("enum_declaration", "Enum"),
        DefRule::func("function_definition"),
        DefRule::method("method_declaration"),
    ],
    owner_kinds: &[
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "enum_declaration",
    ],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Php,
    docs: DocStyle::CBlockOrLine,
};

/// Bash (new, data-path only): `function_definition` defs; command calls;
/// `source` / `.` imports; `#` docstrings. No class ownership.
pub const BASH: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("function_definition")],
    owner_kinds: &[],
    calls: CallSpec {
        skip_callees: &["source", "."],
    },
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// Lua (new, data-path only): free `function_declaration`s named with a bare
/// identifier; `require("…")` imports; `--` line docstrings. No class
/// ownership (dotted / method definitions are not captured).
pub const LUA: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("function_declaration")],
    owner_kinds: &[],
    calls: CallSpec {
        skip_callees: &["require"],
    },
    imports: ImportStrategy::Lua,
    docs: DocStyle::LineDashComment,
};

/// Kotlin (new, data-path only): class/interface (`class_declaration`) /
/// `object`; functions owned by their enclosing type; `import` directives;
/// `/** */` KDoc docstrings.
pub const KOTLIN: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_declaration", "Class"),
        DefRule::ty("object_declaration", "Object"),
        DefRule::method("function_declaration"),
    ],
    owner_kinds: &["class_declaration", "object_declaration"],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Kotlin,
    docs: DocStyle::BlockJsdoc,
};

/// Scala (new, data-path only): class/object/trait; `def` functions owned by
/// their enclosing type; `import` declarations; `/** */` ScalaDoc docstrings.
pub const SCALA: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_definition", "Class"),
        DefRule::ty("object_definition", "Object"),
        DefRule::ty("trait_definition", "Trait"),
        DefRule::method("function_definition"),
    ],
    owner_kinds: &["class_definition", "object_definition", "trait_definition"],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Scala,
    docs: DocStyle::BlockJsdoc,
};

/// Swift (new, data-path only): class/struct/enum (all `class_declaration`);
/// `func` functions owned by their enclosing type; `import` declarations;
/// `///` line docstrings.
pub const SWIFT: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[
        DefRule::ty("class_declaration", "Class"),
        DefRule::method("function_declaration"),
    ],
    owner_kinds: &["class_declaration"],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Swift,
    docs: DocStyle::CBlockOrLine,
};

/// Zig (new, data-path only): free `function_declaration`s (Zig methods live in
/// anonymous structs, so ownership is not modelled); `@import("…")` imports;
/// `///` doc-comment docstrings.
pub const ZIG: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("function_declaration")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    imports: ImportStrategy::Zig,
    docs: DocStyle::CBlockOrLine,
};

/// R (new, data-path only): top-level `name <- function(...)` assignments
/// (name resolved structurally from the left-hand identifier); `library(…)` /
/// `require(…)` imports; `#` line docstrings. No class ownership.
pub const R: LangSpec = LangSpec {
    name: NameStrategy::RAssign,
    defs: &[DefRule::func("binary_operator")],
    owner_kinds: &[],
    calls: CallSpec {
        skip_callees: &["library", "require", "requireNamespace"],
    },
    imports: ImportStrategy::R,
    docs: DocStyle::LineHashComment,
};

// ---------------------------------------------------------------------------
// Generic extraction engine
// ---------------------------------------------------------------------------

/// Run the three uniform passes for `spec` over `source`, producing exactly the
/// same `ExtractedNode` / `ExtractedEdge` output a bespoke per-language
/// extractor would. `queries` is the language's compiled
/// Definitions/Calls/Imports set.
pub fn spec_extract(
    language: Language,
    spec: &LangSpec,
    queries: &[CompiledQuery],
    source: &[u8],
    file_path: &str,
) -> grepplus_core::Result<ExtractionResult> {
    let tree = crate::parse(language, source)?;
    let root = tree.root_node();
    let mut result = ExtractionResult::default();
    let file_qname = format!("{file_path}::__file__");

    spec_definitions(spec, queries, root, source, file_path, &mut result);
    spec_calls(spec, queries, root, source, file_path, &mut result);
    spec_imports(
        spec,
        queries,
        root,
        source,
        file_path,
        &file_qname,
        &mut result,
    );

    Ok(result)
}

/// Look up the [`DefRule`] for a node kind, if the spec has one.
fn rule_for<'s>(spec: &'s LangSpec, kind: &str) -> Option<&'s DefRule> {
    spec.defs.iter().find(|r| r.node_kind == kind)
}

/// Resolve `(label, qname)` for one definition `def_node` of a known rule.
fn def_label_and_qname(
    spec: &LangSpec,
    rule: &DefRule,
    source: &[u8],
    def_node: Node<'_>,
    name: &str,
    file_path: &str,
) -> (String, String) {
    let free = || {
        (
            rule.label.to_string(),
            format!("{file_path}::{}::{name}", rule.label),
        )
    };
    let owned = |owner: &str| {
        (
            rule.method_label.to_string(),
            format!("{file_path}::{owner}::{name}"),
        )
    };
    match rule.owner {
        Owner::None => free(),
        Owner::EnclosingName => match enclosing_owner_name(source, def_node, spec.owner_kinds) {
            Some(owner) => owned(owner),
            None => free(),
        },
        Owner::GoReceiver => match go_receiver_type(source, def_node) {
            Some(t) => owned(t),
            None => free(),
        },
        Owner::CppClass => match cpp_function_owner(source, def_node) {
            Some(t) => owned(&t),
            None => free(),
        },
    }
}

fn spec_definitions(
    spec: &LangSpec,
    queries: &[CompiledQuery],
    root: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    // `Capture` languages tag the name as `@name`; `CStructural` languages tag
    // the whole node as `@def`.
    let want = match spec.name {
        NameStrategy::Capture => "name",
        NameStrategy::CStructural | NameStrategy::RAssign => "def",
    };
    for cq in queries
        .iter()
        .filter(|cq| cq.kind == QueryKind::Definitions)
    {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, root, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(cap_name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };
                if cap_name != want {
                    continue;
                }

                let (def_node, name): (Node<'_>, String) = match spec.name {
                    NameStrategy::Capture => {
                        let node = cap.node;
                        (
                            node.parent().unwrap_or(node),
                            node_text(source, node).to_string(),
                        )
                    }
                    NameStrategy::CStructural => {
                        let node = cap.node;
                        match c_def_name(source, node) {
                            Some(n) => (node, n),
                            None => continue,
                        }
                    }
                    NameStrategy::RAssign => {
                        let node = cap.node;
                        match r_def_name(source, node) {
                            Some(n) => (node, n),
                            None => continue,
                        }
                    }
                };

                let Some(rule) = rule_for(spec, def_node.kind()) else {
                    continue;
                };
                // A few rules resolve their label dynamically from the body
                // (Go `type_spec` → Struct / Interface / Type).
                let rule = adjusted_rule(rule, def_node).unwrap_or(*rule);

                let (label, qname) =
                    def_label_and_qname(spec, &rule, source, def_node, &name, file_path);

                let mut properties = serde_json::Map::new();
                if let Some(doc) = extract_doc(spec.docs, source, def_node) {
                    let summary = docstring_summary(&doc).to_string();
                    properties.insert("doc".into(), serde_json::Value::String(summary));
                    properties.insert("doc_full".into(), serde_json::Value::String(doc));
                }

                result.nodes.push(ExtractedNode {
                    label,
                    name,
                    qualified_name: qname,
                    file_path: file_path.to_string(),
                    start_line: def_node.start_position().row as u32 + 1,
                    end_line: def_node.end_position().row as u32 + 1,
                    properties: serde_json::Value::Object(properties),
                });
            }
        }
    }
}

/// Go `type_spec` carries its concrete kind in the body; rewrite the rule's
/// label (`Struct` / `Interface` / `Type`) accordingly. Returns `None` when no
/// adjustment is needed.
fn adjusted_rule(rule: &DefRule, def_node: Node<'_>) -> Option<DefRule> {
    if def_node.kind() != "type_spec" {
        return None;
    }
    let label = match def_node.child_by_field_name("type").map(|n| n.kind()) {
        Some("struct_type") => "Struct",
        Some("interface_type") => "Interface",
        _ => "Type",
    };
    Some(DefRule {
        label,
        method_label: label,
        ..*rule
    })
}

fn spec_calls(
    spec: &LangSpec,
    queries: &[CompiledQuery],
    root: Node<'_>,
    source: &[u8],
    file_path: &str,
    result: &mut ExtractionResult,
) {
    for cq in queries.iter().filter(|cq| cq.kind == QueryKind::Calls) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, root, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(cap_name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };
                if cap_name != "callee" {
                    continue;
                }
                let node = cap.node;
                let text = node_text(source, node);
                if spec.calls.skip_callees.contains(&text) {
                    continue;
                }
                // NOTE: we deliberately do NOT materialise a `Call` pseudo-node.
                // The CALLS edge below targets the real `file::Function::<text>`
                // qname (resolved by name when cross-file), so a `Call` node was
                // never a resolution endpoint — it was pure dead weight that
                // inflated the node count ~4x (slowing indexing) and flooded
                // symbol/semantic search (forensics F2). The call's information
                // lives entirely in the edge + its `callee_name` property.
                if let Some(caller_qname) = enclosing_callable_qname(spec, source, node, file_path)
                {
                    result.edges.push(ExtractedEdge {
                        edge_type: "CALLS".into(),
                        source_qualified_name: caller_qname,
                        target_qualified_name: format!("{file_path}::Function::{text}"),
                        file_path: file_path.to_string(),
                        line: node.start_position().row as u32 + 1,
                        properties: json!({ "callee_text": text, "callee_name": text }),
                    });
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spec_imports(
    spec: &LangSpec,
    queries: &[CompiledQuery],
    root: Node<'_>,
    source: &[u8],
    file_path: &str,
    file_qname: &str,
    result: &mut ExtractionResult,
) {
    for cq in queries.iter().filter(|cq| cq.kind == QueryKind::Imports) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&cq.query, root, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let Some(cap_name) = cq.capture_names.get(cap.index as usize) else {
                    continue;
                };
                let node = cap.node;
                let items = expand_import(spec.imports, source, cap_name, node);
                if items.is_empty() {
                    continue;
                }
                let line = node.start_position().row as u32 + 1;
                for item in &items {
                    // NOTE: no `Import` pseudo-node (same rationale as Call,
                    // forensics F2). The IMPORTS edge's source is the real
                    // per-file `__file__` Module node and its target is resolved
                    // by `imported_name`; the Import node was never looked up.
                    result.edges.push(ExtractedEdge {
                        edge_type: "IMPORTS".into(),
                        source_qualified_name: file_qname.to_string(),
                        target_qualified_name: format!("{file_path}::Import::{}", item.path),
                        file_path: file_path.to_string(),
                        line,
                        properties: json!({
                            "path": item.path,
                            "imported_name": item.imported_name,
                            "original_name": item.original_name,
                            "glob": item.is_glob,
                        }),
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ownership / enclosing-scope resolution
// ---------------------------------------------------------------------------

/// The `name:` of the nearest ancestor of `node` whose kind is in `kinds`.
fn enclosing_owner_name<'a>(source: &'a [u8], node: Node<'_>, kinds: &[&str]) -> Option<&'a str> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if kinds.contains(&cur.kind()) {
            return cur
                .child_by_field_name("name")
                .map(|n| node_text(source, n));
        }
        p = cur.parent();
    }
    None
}

/// Walk `node`'s ancestors and return the qname of the nearest enclosing
/// *callable* definition, constructed with the same ownership rules the
/// definition pass used. Used as the `source` endpoint for CALLS edges.
fn enclosing_callable_qname(
    spec: &LangSpec,
    source: &[u8],
    node: Node<'_>,
    file_path: &str,
) -> Option<String> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if let Some(rule) = rule_for(spec, cur.kind()) {
            if rule.callable {
                // Resolve the callable's own name the same way the def pass did.
                let name = callable_name(spec, source, cur)?;
                let (_, qname) = def_label_and_qname(spec, rule, source, cur, &name, file_path);
                return Some(qname);
            }
        }
        p = cur.parent();
    }
    None
}

/// The name of a callable definition node, by the spec's name strategy.
fn callable_name(spec: &LangSpec, source: &[u8], def_node: Node<'_>) -> Option<String> {
    match spec.name {
        NameStrategy::Capture => def_node
            .child_by_field_name("name")
            .map(|n| node_text(source, n).to_string()),
        NameStrategy::CStructural => c_def_name(source, def_node),
        NameStrategy::RAssign => r_def_name(source, def_node),
    }
}

/// Resolve the name of an R `name <- function(...)` assignment: the left-hand
/// `identifier` of the `binary_operator`. Returns `None` for assignments whose
/// right-hand side is not a `function_definition`.
fn r_def_name(source: &[u8], def_node: Node<'_>) -> Option<String> {
    if def_node.kind() != "binary_operator" {
        return None;
    }
    let lhs = def_node.child_by_field_name("lhs")?;
    if lhs.kind() != "identifier" {
        return None;
    }
    let rhs = def_node.child_by_field_name("rhs")?;
    if rhs.kind() != "function_definition" {
        return None;
    }
    Some(node_text(source, lhs).to_string())
}

// ---------------------------------------------------------------------------
// Go receiver-type ownership
// ---------------------------------------------------------------------------

/// The receiver base type of a Go `method_declaration` (`*Adder` → `Adder`).
fn go_receiver_type<'a>(source: &'a [u8], method: Node<'_>) -> Option<&'a str> {
    let receiver = method.child_by_field_name("receiver")?;
    let mut decl = None;
    for i in 0..receiver.named_child_count() {
        if let Some(c) = receiver.named_child(i) {
            if c.kind() == "parameter_declaration" {
                decl = Some(c);
                break;
            }
        }
    }
    let ty = decl?.child_by_field_name("type")?;
    Some(go_base_type_name(source, ty))
}

/// Strip pointer / generic / qualified wrappers off a Go type node and return
/// the base `type_identifier` text.
fn go_base_type_name<'a>(source: &'a [u8], node: Node<'_>) -> &'a str {
    match node.kind() {
        "type_identifier" => node_text(source, node),
        "pointer_type" => node
            .named_child(0)
            .map(|n| go_base_type_name(source, n))
            .unwrap_or_else(|| node_text(source, node)),
        "generic_type" => node
            .child_by_field_name("type")
            .map(|n| go_base_type_name(source, n))
            .unwrap_or_else(|| node_text(source, node)),
        "qualified_type" => node
            .child_by_field_name("name")
            .map(|n| node_text(source, n))
            .unwrap_or_else(|| node_text(source, node)),
        _ => {
            if let Some(inner) = first_child_of_kind(node, "type_identifier") {
                node_text(source, inner)
            } else {
                node_text(source, node)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C / C++ structural name + ownership
// ---------------------------------------------------------------------------

/// Resolve the name of a C / C++ definition node (`@def`). Functions are read
/// off the declarator; tagged types and namespaces off the `name:` field;
/// typedefs off the declarator.
fn c_def_name(source: &[u8], def_node: Node<'_>) -> Option<String> {
    match def_node.kind() {
        "function_definition" => c_function_name(source, def_node).map(|(n, _)| n.to_string()),
        "struct_specifier"
        | "union_specifier"
        | "enum_specifier"
        | "class_specifier"
        | "namespace_definition" => def_node
            .child_by_field_name("name")
            .map(|n| node_text(source, n).to_string()),
        "type_definition" => c_typedef_name(source, def_node).map(|n| n.to_string()),
        _ => None,
    }
}

/// The owner of a C / C++ `function_definition`: an out-of-line `Class::`
/// qualifier or the lexically enclosing class. `None` for a free function.
fn cpp_function_owner(source: &[u8], def_node: Node<'_>) -> Option<String> {
    if def_node.kind() != "function_definition" {
        return None;
    }
    let (_, owner) = c_function_name(source, def_node)?;
    owner.map(|s| s.to_string())
}

/// The base function name of a C / C++ `function_declarator`'s `declarator:`,
/// plus the `Class::` qualifier for an out-of-line C++ method.
fn c_declarator_name<'a>(
    source: &'a [u8],
    declarator: Node<'_>,
) -> Option<(&'a str, Option<&'a str>)> {
    match declarator.kind() {
        "identifier" | "field_identifier" => Some((node_text(source, declarator), None)),
        "qualified_identifier" => {
            let name = declarator.child_by_field_name("name")?;
            let owner = declarator
                .child_by_field_name("scope")
                .map(|n| node_text(source, n));
            let (leaf, _) = c_declarator_name(source, name)?;
            Some((leaf, owner))
        }
        "function_declarator"
        | "pointer_declarator"
        | "parenthesized_declarator"
        | "reference_declarator" => {
            let inner = declarator.child_by_field_name("declarator")?;
            c_declarator_name(source, inner)
        }
        _ => {
            let inner = declarator.child_by_field_name("declarator")?;
            c_declarator_name(source, inner)
        }
    }
}

/// `(name, owner)` of a C / C++ `function_definition`.
fn c_function_name<'a>(source: &'a [u8], func: Node<'_>) -> Option<(&'a str, Option<&'a str>)> {
    let declarator = func.child_by_field_name("declarator")?;
    let (name, qualifier) = c_declarator_name(source, declarator)?;
    let owner = qualifier.or_else(|| cpp_enclosing_class(source, func));
    Some((name, owner))
}

/// The name of the nearest enclosing C++ `class_specifier` / `struct_specifier`.
fn cpp_enclosing_class<'a>(source: &'a [u8], node: Node<'_>) -> Option<&'a str> {
    let mut p = node.parent();
    while let Some(cur) = p {
        if matches!(cur.kind(), "class_specifier" | "struct_specifier") {
            if let Some(name) = cur.child_by_field_name("name") {
                return Some(node_text(source, name));
            }
        }
        p = cur.parent();
    }
    None
}

/// The declared type name of a `type_definition` (`typedef … Name;`).
fn c_typedef_name<'a>(source: &'a [u8], typedef: Node<'_>) -> Option<&'a str> {
    let declarator = typedef.child_by_field_name("declarator")?;
    c_typedef_declarator_name(source, declarator)
}

fn c_typedef_declarator_name<'a>(source: &'a [u8], declarator: Node<'_>) -> Option<&'a str> {
    match declarator.kind() {
        "type_identifier" => Some(node_text(source, declarator)),
        "pointer_declarator" | "parenthesized_declarator" | "function_declarator" => {
            let inner = declarator.child_by_field_name("declarator")?;
            c_typedef_declarator_name(source, inner)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Docstrings
// ---------------------------------------------------------------------------

/// Extract a definition's docstring per the language's [`DocStyle`].
fn extract_doc(style: DocStyle, source: &[u8], def_node: Node<'_>) -> Option<String> {
    let doc = match style {
        DocStyle::None => None,
        DocStyle::PythonBodyString => py_body_string(source, def_node),
        DocStyle::LineSlashComment => line_comment_run(source, doc_anchor(def_node), "//", &['/']),
        DocStyle::LineHashComment => line_comment_run(source, doc_anchor(def_node), "#", &['#']),
        DocStyle::BlockJsdoc => block_jsdoc(source, doc_anchor(def_node)),
        DocStyle::CBlockOrLine => c_block_or_line(source, def_node),
        DocStyle::LineDashComment => line_comment_run(source, doc_anchor(def_node), "--", &['-']),
    }?;
    Some(truncate_doc(doc))
}

/// Truncate `doc` to [`MAX_COMMENT_LEN`] bytes on a char boundary; `None`-ish
/// empties are filtered by the callers.
fn truncate_doc(mut doc: String) -> String {
    if doc.len() > MAX_COMMENT_LEN {
        let mut end = MAX_COMMENT_LEN;
        while end > 0 && !doc.is_char_boundary(end) {
            end -= 1;
        }
        doc.truncate(end);
    }
    doc
}

/// The anchor whose preceding siblings carry the doc comment. Walks up the
/// wrapper nodes that sit between a definition and its leading comment:
///   * Go `type_spec` → the enclosing single `type_declaration`.
///   * Ruby `method` first in a `body_statement` → the `body_statement`.
fn doc_anchor(def_node: Node<'_>) -> Node<'_> {
    // Go: a lone `type_spec` is wrapped in a `type_declaration` that carries the
    // leading comment.
    if def_node.kind() == "type_spec" {
        if let Some(parent) = def_node.parent() {
            if parent.kind() == "type_declaration" {
                return parent;
            }
        }
    }
    // Ruby: a method/class with no prev sibling sits first in a wrapping
    // `body_statement`; the comment is a sibling of that body_statement.
    if def_node.prev_sibling().is_none() {
        if let Some(parent) = def_node.parent() {
            if parent.kind() == "body_statement" {
                return parent;
            }
        }
    }
    def_node
}

/// Python: the first bare string statement in the definition `body`.
fn py_body_string(source: &[u8], def_node: Node<'_>) -> Option<String> {
    let body = def_node.child_by_field_name("body")?;
    let first = body.named_child(0)?;
    if first.kind() != "expression_statement" {
        return None;
    }
    let string_node = first.named_child(0)?;
    if string_node.kind() != "string" {
        return None;
    }
    let raw = node_text(source, string_node);
    let text = py_string_literal_text(raw);
    let doc = text.trim().to_string();
    if doc.is_empty() {
        None
    } else {
        Some(doc)
    }
}

/// Strip Python string-literal syntax (prefix + surrounding quotes).
fn py_string_literal_text(raw: &str) -> &str {
    let trimmed = raw.trim_start_matches(['r', 'R', 'b', 'B', 'f', 'F', 'u', 'U']);
    for q in ["\"\"\"", "'''", "\"", "'"] {
        if let Some(inner) = trimmed.strip_prefix(q) {
            return inner.strip_suffix(q).unwrap_or(inner);
        }
    }
    trimmed
}

/// A run of consecutive leading line `comment` nodes with the given marker.
/// `marker` is the literal prefix the comment must start with (`//` / `#`);
/// `trim_chars` are stripped from the front of each comment line.
fn line_comment_run(
    source: &[u8],
    anchor: Node<'_>,
    marker: &str,
    trim_chars: &[char],
) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut prev = anchor.prev_sibling();
    while let Some(cur) = prev {
        if cur.kind() != "comment" {
            break;
        }
        let raw = node_text(source, cur);
        if !raw.starts_with(marker) {
            break;
        }
        let text = raw.trim_start_matches(trim_chars).trim().to_string();
        lines.push(text);
        prev = cur.prev_sibling();
    }
    if lines.is_empty() {
        return None;
    }
    lines.reverse();
    let doc = lines.join("\n").trim().to_string();
    if doc.is_empty() {
        None
    } else {
        Some(doc)
    }
}

/// A single leading `/** … */` JSDoc/Javadoc block comment. The grammar names
/// the node `comment` (JS/TS) or `block_comment` (Java); we accept either.
fn block_jsdoc(source: &[u8], anchor: Node<'_>) -> Option<String> {
    let prev = anchor.prev_sibling()?;
    if !matches!(prev.kind(), "comment" | "block_comment") {
        return None;
    }
    let raw = node_text(source, prev);
    if !raw.starts_with("/**") {
        return None;
    }
    let inner = raw
        .trim_start_matches("/**")
        .trim_start_matches("/*")
        .trim_end_matches("*/");
    let mut out = String::new();
    for line in inner.lines() {
        let line = line.trim();
        let line = line.strip_prefix('*').unwrap_or(line).trim();
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
    }
    let doc = out.trim().to_string();
    if doc.is_empty() {
        None
    } else {
        Some(doc)
    }
}

/// C / C++ style: a leading `/* */` block OR a run of `//` line comments. The
/// grammar names both `comment`.
fn c_block_or_line(source: &[u8], def_node: Node<'_>) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut prev = def_node.prev_sibling();
    while let Some(cur) = prev {
        if cur.kind() != "comment" {
            break;
        }
        let raw = node_text(source, cur);
        if raw.starts_with("/*") {
            let inner = raw
                .trim_start_matches("/**")
                .trim_start_matches("/*")
                .trim_end_matches("*/");
            let mut block = String::new();
            for line in inner.lines() {
                let line = line.trim();
                let line = line.strip_prefix('*').unwrap_or(line).trim();
                if !block.is_empty() {
                    block.push('\n');
                }
                block.push_str(line);
            }
            lines.push(block.trim().to_string());
            break;
        } else if raw.starts_with("//") {
            // `//` and C#/doc `///` line comments: drop every leading slash.
            lines.push(raw.trim_start_matches('/').trim().to_string());
            prev = cur.prev_sibling();
        } else {
            break;
        }
    }
    if lines.is_empty() {
        return None;
    }
    lines.reverse();
    let doc = lines.join("\n").trim().to_string();
    if doc.is_empty() {
        None
    } else {
        Some(doc)
    }
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

/// Expand one import-bearing capture into [`ImportedItem`]s per the strategy.
/// `cap_name` distinguishes multi-capture queries (e.g. JS `import` vs
/// `require`, C++ `include` vs `using`).
fn expand_import(
    strategy: ImportStrategy,
    source: &[u8],
    cap_name: &str,
    node: Node<'_>,
) -> Vec<ImportedItem> {
    match strategy {
        ImportStrategy::Python => {
            if cap_name == "import" || cap_name == "from_import" {
                py_expand_imports(source, node)
            } else {
                Vec::new()
            }
        }
        ImportStrategy::JsTs => match cap_name {
            "import" => js_expand_import(source, node),
            "require" => js_expand_require(source, node),
            _ => Vec::new(),
        },
        ImportStrategy::Go => {
            if cap_name == "import" {
                go_expand_import(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Ruby => {
            if cap_name == "require" {
                rb_expand_require(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Java => {
            if cap_name == "import" {
                java_expand_import(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::C => {
            if cap_name == "include" {
                c_expand_include(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Cpp => match cap_name {
            "include" => c_expand_include(source, node).into_iter().collect(),
            "using" => cpp_expand_using(source, node).into_iter().collect(),
            _ => Vec::new(),
        },
        ImportStrategy::CSharp => {
            if cap_name == "import" {
                cs_expand_using(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Php => {
            if cap_name == "import" {
                php_expand_use(source, node)
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Bash => {
            if cap_name == "source" {
                bash_expand_source(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Lua => {
            if cap_name == "require" {
                lua_expand_require(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Kotlin => {
            if cap_name == "import" {
                kotlin_expand_import(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Scala => {
            if cap_name == "import" {
                scala_expand_import(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Swift => {
            if cap_name == "import" {
                swift_expand_import(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::Zig => {
            if cap_name == "import" {
                zig_expand_import(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        ImportStrategy::R => {
            if cap_name == "require" {
                r_expand_library(source, node).into_iter().collect()
            } else {
                Vec::new()
            }
        }
    }
}

// ---- Python imports -------------------------------------------------------

fn py_expand_imports(source: &[u8], node: Node<'_>) -> Vec<ImportedItem> {
    let mut out = Vec::new();
    match node.kind() {
        "import_statement" => {
            for i in 0..node.named_child_count() {
                let Some(child) = node.named_child(i) else {
                    continue;
                };
                py_push_dotted_or_aliased(source, child, &mut out);
            }
        }
        "import_from_statement" => {
            let module = node
                .child_by_field_name("module_name")
                .map(|n| node_text(source, n).to_string())
                .unwrap_or_default();
            let mut has_wildcard = false;
            let mut name_children: Vec<Node<'_>> = Vec::new();
            for i in 0..node.named_child_count() {
                let Some(child) = node.named_child(i) else {
                    continue;
                };
                if child.kind() == "wildcard_import" {
                    has_wildcard = true;
                } else if Some(child) != node.child_by_field_name("module_name") {
                    name_children.push(child);
                }
            }
            if has_wildcard {
                out.push(ImportedItem {
                    path: module.clone(),
                    imported_name: String::new(),
                    original_name: String::new(),
                    is_glob: true,
                });
                return out;
            }
            for child in name_children {
                let (orig, alias) = py_name_and_alias(source, child);
                let final_seg = orig.rsplit('.').next().unwrap_or(&orig).to_string();
                let imported = alias.clone().unwrap_or_else(|| final_seg.clone());
                let path = if module.is_empty() {
                    orig.clone()
                } else {
                    format!("{module}.{orig}")
                };
                out.push(ImportedItem {
                    path,
                    imported_name: imported,
                    original_name: final_seg,
                    is_glob: false,
                });
            }
        }
        _ => {}
    }
    out
}

fn py_push_dotted_or_aliased(source: &[u8], child: Node<'_>, out: &mut Vec<ImportedItem>) {
    let (path, alias) = py_name_and_alias(source, child);
    let final_seg = path.rsplit('.').next().unwrap_or(&path).to_string();
    let imported_name = match &alias {
        Some(a) => a.clone(),
        None => path.split('.').next().unwrap_or(&path).to_string(),
    };
    out.push(ImportedItem {
        path,
        imported_name,
        original_name: final_seg,
        is_glob: false,
    });
}

fn py_name_and_alias(source: &[u8], node: Node<'_>) -> (String, Option<String>) {
    if node.kind() == "aliased_import" {
        let name = node
            .child_by_field_name("name")
            .map(|n| node_text(source, n).to_string())
            .unwrap_or_default();
        let alias = node
            .child_by_field_name("alias")
            .map(|n| node_text(source, n).to_string());
        (name, alias)
    } else {
        (node_text(source, node).to_string(), None)
    }
}

// ---- JS / TS imports ------------------------------------------------------

fn js_expand_import(source: &[u8], node: Node<'_>) -> Vec<ImportedItem> {
    let mut out = Vec::new();
    let module = node
        .child_by_field_name("source")
        .map(|n| js_string_text(source, n))
        .unwrap_or_default();

    let mut top = node.walk();
    let clause = node
        .named_children(&mut top)
        .find(|c| c.kind() == "import_clause");

    let Some(clause) = clause else {
        out.push(ImportedItem {
            path: module.clone(),
            imported_name: String::new(),
            original_name: String::new(),
            is_glob: false,
        });
        return out;
    };

    let mut cursor = clause.walk();
    for child in clause.named_children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                let name = node_text(source, child).to_string();
                out.push(ImportedItem {
                    path: module.clone(),
                    imported_name: name.clone(),
                    original_name: name,
                    is_glob: false,
                });
            }
            "namespace_import" => {
                let name = first_child_of_kind(child, "identifier")
                    .map(|n| node_text(source, n).to_string())
                    .unwrap_or_default();
                out.push(ImportedItem {
                    path: module.clone(),
                    imported_name: name,
                    original_name: String::new(),
                    is_glob: true,
                });
            }
            "named_imports" => {
                let mut spec_cursor = child.walk();
                for spec in child.named_children(&mut spec_cursor) {
                    if spec.kind() != "import_specifier" {
                        continue;
                    }
                    let orig = spec
                        .child_by_field_name("name")
                        .map(|n| node_text(source, n).to_string())
                        .unwrap_or_default();
                    let alias = spec
                        .child_by_field_name("alias")
                        .map(|n| node_text(source, n).to_string());
                    let imported = alias.clone().unwrap_or_else(|| orig.clone());
                    out.push(ImportedItem {
                        path: module.clone(),
                        imported_name: imported,
                        original_name: orig,
                        is_glob: false,
                    });
                }
            }
            _ => {}
        }
    }

    if out.is_empty() {
        out.push(ImportedItem {
            path: module,
            imported_name: String::new(),
            original_name: String::new(),
            is_glob: false,
        });
    }
    out
}

fn js_expand_require(source: &[u8], declarator: Node<'_>) -> Vec<ImportedItem> {
    let mut out = Vec::new();
    let Some(value) = declarator.child_by_field_name("value") else {
        return out;
    };
    let module = value
        .child_by_field_name("arguments")
        .and_then(|args| args.named_child(0))
        .map(|s| js_string_text(source, s))
        .unwrap_or_default();

    let Some(name_node) = declarator.child_by_field_name("name") else {
        return out;
    };
    match name_node.kind() {
        "identifier" => {
            let name = node_text(source, name_node).to_string();
            out.push(ImportedItem {
                path: module,
                imported_name: name.clone(),
                original_name: name,
                is_glob: false,
            });
        }
        "object_pattern" => {
            let mut cursor = name_node.walk();
            for child in name_node.named_children(&mut cursor) {
                match child.kind() {
                    "shorthand_property_identifier_pattern" => {
                        let name = node_text(source, child).to_string();
                        out.push(ImportedItem {
                            path: module.clone(),
                            imported_name: name.clone(),
                            original_name: name,
                            is_glob: false,
                        });
                    }
                    "pair_pattern" => {
                        let orig = child
                            .child_by_field_name("key")
                            .map(|n| node_text(source, n).to_string())
                            .unwrap_or_default();
                        let bind = child
                            .child_by_field_name("value")
                            .map(|n| node_text(source, n).to_string())
                            .unwrap_or_else(|| orig.clone());
                        out.push(ImportedItem {
                            path: module.clone(),
                            imported_name: bind,
                            original_name: orig,
                            is_glob: false,
                        });
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    out
}

fn js_string_text(source: &[u8], node: Node<'_>) -> String {
    if let Some(frag) = first_child_of_kind(node, "string_fragment") {
        return node_text(source, frag).to_string();
    }
    let raw = node_text(source, node);
    raw.trim_matches(['"', '\'', '`']).to_string()
}

// ---- Go imports -----------------------------------------------------------

fn go_expand_import(source: &[u8], spec: Node<'_>) -> Option<ImportedItem> {
    let path_node = spec.child_by_field_name("path")?;
    let path = go_string_text(source, path_node);
    if path.is_empty() {
        return None;
    }
    let final_seg = path.rsplit('/').next().unwrap_or(&path).to_string();
    let imported_name = match spec.child_by_field_name("name") {
        Some(name_node) => node_text(source, name_node).to_string(),
        None => final_seg.clone(),
    };
    Some(ImportedItem {
        path,
        imported_name,
        original_name: final_seg,
        is_glob: false,
    })
}

fn go_string_text(source: &[u8], node: Node<'_>) -> String {
    if let Some(frag) = first_child_of_kind(node, "interpreted_string_literal_content") {
        return node_text(source, frag).to_string();
    }
    let raw = node_text(source, node);
    raw.trim_matches(['"', '`']).to_string()
}

// ---- Ruby imports ---------------------------------------------------------

fn rb_expand_require(source: &[u8], call: Node<'_>) -> Option<ImportedItem> {
    let args = call.child_by_field_name("arguments")?;
    let mut string_node = None;
    for i in 0..args.named_child_count() {
        if let Some(c) = args.named_child(i) {
            if c.kind() == "string" {
                string_node = Some(c);
                break;
            }
        }
    }
    let path = rb_string_text(source, string_node?);
    if path.is_empty() {
        return None;
    }
    let trimmed = path.trim_end_matches(".rb");
    let final_seg = trimmed.rsplit('/').next().unwrap_or(trimmed).to_string();
    Some(ImportedItem {
        path,
        imported_name: final_seg.clone(),
        original_name: final_seg,
        is_glob: false,
    })
}

fn rb_string_text(source: &[u8], node: Node<'_>) -> String {
    if let Some(frag) = first_child_of_kind(node, "string_content") {
        return node_text(source, frag).to_string();
    }
    let raw = node_text(source, node);
    raw.trim_matches(['"', '\'']).to_string()
}

// ---- Java imports ---------------------------------------------------------

fn java_expand_import(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    let path = java_import_path(source, node);
    if path.is_empty() {
        return None;
    }
    let imported_name = java_import_name(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: imported_name.clone(),
        original_name: imported_name,
        is_glob: false,
    })
}

fn java_import_name(path: &str) -> &str {
    let trimmed = path.trim_end_matches(".*").trim_end_matches('.');
    trimmed.rsplit('.').next().unwrap_or(trimmed)
}

fn java_import_path(source: &[u8], node: Node<'_>) -> String {
    for i in 0..node.named_child_count() {
        if let Some(c) = node.named_child(i) {
            if matches!(c.kind(), "scoped_identifier" | "identifier") {
                return node_text(source, c).to_string();
            }
        }
    }
    node_text(source, node)
        .trim_start_matches("import")
        .trim()
        .trim_start_matches("static")
        .trim()
        .trim_end_matches(';')
        .trim()
        .to_string()
}

// ---- C / C++ imports ------------------------------------------------------

fn c_expand_include(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    let path_node = node.child_by_field_name("path")?;
    let raw = node_text(source, path_node);
    let path = raw
        .trim_start_matches('<')
        .trim_end_matches('>')
        .trim_matches('"')
        .to_string();
    if path.is_empty() {
        return None;
    }
    let basename = path.rsplit('/').next().unwrap_or(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: basename.clone(),
        original_name: basename,
        is_glob: false,
    })
}

fn cpp_expand_using(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i) else {
            continue;
        };
        match child.kind() {
            "qualified_identifier" => {
                let path = node_text(source, child).to_string();
                let name = path.rsplit("::").next().unwrap_or(&path).to_string();
                return Some(ImportedItem {
                    path,
                    imported_name: name.clone(),
                    original_name: name,
                    is_glob: false,
                });
            }
            "identifier" => {
                let name = node_text(source, child).to_string();
                return Some(ImportedItem {
                    path: name.clone(),
                    imported_name: name.clone(),
                    original_name: name,
                    is_glob: true,
                });
            }
            _ => {}
        }
    }
    None
}

// ---- C# imports -----------------------------------------------------------

/// `using System.Collections.Generic;` → name `Generic`, path
/// `System.Collections.Generic`. `using static System.Math;` and aliased
/// `using IO = System.IO;` are handled by reading the qualified name.
fn cs_expand_using(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    // The directive's qualified name is the (last) `qualified_name` / `identifier`
    // child; an alias (`name = qualified;`) puts the binding on the `name:` field.
    let alias = node
        .child_by_field_name("name")
        .map(|n| node_text(source, n).to_string());
    let mut path_node = None;
    for i in 0..node.named_child_count() {
        if let Some(c) = node.named_child(i) {
            if matches!(
                c.kind(),
                "qualified_name" | "identifier" | "member_access_expression"
            ) {
                // Skip the alias identifier itself.
                if Some(c) == node.child_by_field_name("name") {
                    continue;
                }
                path_node = Some(c);
            }
        }
    }
    let path_node = path_node.or_else(|| node.child_by_field_name("name"))?;
    let path = node_text(source, path_node).to_string();
    if path.is_empty() {
        return None;
    }
    let final_seg = path.rsplit('.').next().unwrap_or(&path).to_string();
    let imported_name = alias.unwrap_or(final_seg.clone());
    Some(ImportedItem {
        path,
        imported_name,
        original_name: final_seg,
        is_glob: false,
    })
}

// ---- PHP imports ----------------------------------------------------------

/// `use App\Models\User;` → name `User`, path `App\Models\User`;
/// `use App\Models\User as U;` → name `U`. A grouped
/// `use App\{Foo, Bar as B};` expands to one item per clause.
fn php_expand_use(source: &[u8], node: Node<'_>) -> Vec<ImportedItem> {
    let mut out = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "namespace_use_clause" => {
                if let Some(item) = php_use_clause(source, child, "") {
                    out.push(item);
                }
            }
            "namespace_use_group" => {
                // `use App\Lib\{Foo, Bar as B};` — the prefix `namespace_name`
                // precedes the `body: namespace_use_group`; each member is a
                // `namespace_use_clause`.
                let prefix = node
                    .named_children(&mut node.walk())
                    .find(|c| c.kind() == "namespace_name")
                    .map(|n| node_text(source, n).to_string())
                    .unwrap_or_default();
                let mut gc = child.walk();
                for clause in child.named_children(&mut gc) {
                    if clause.kind() == "namespace_use_clause" {
                        if let Some(item) = php_use_clause(source, clause, &prefix) {
                            out.push(item);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn php_use_clause(source: &[u8], clause: Node<'_>, prefix: &str) -> Option<ImportedItem> {
    // A clause holds a qualified/namespace/bare name and an optional
    // `alias: name`.
    let mut name = None;
    let mut alias = None;
    let mut cursor = clause.walk();
    for i in 0..clause.child_count() {
        let Some(child) = clause.child(i) else {
            continue;
        };
        let field = clause.field_name_for_child(i as u32);
        match child.kind() {
            "qualified_name" | "namespace_name" if name.is_none() => {
                name = Some(node_text(source, child).to_string());
            }
            "name" => {
                if field == Some("alias") {
                    alias = Some(node_text(source, child).to_string());
                } else if name.is_none() {
                    name = Some(node_text(source, child).to_string());
                }
            }
            _ => {}
        }
    }
    let _ = &mut cursor;
    let name = name?;
    let path = if prefix.is_empty() {
        name.clone()
    } else {
        format!("{}\\{}", prefix.trim_end_matches('\\'), name)
    };
    let final_seg = path.rsplit('\\').next().unwrap_or(&path).to_string();
    let imported_name = alias.unwrap_or(final_seg.clone());
    Some(ImportedItem {
        path,
        imported_name,
        original_name: final_seg,
        is_glob: false,
    })
}

// ---- Bash imports ---------------------------------------------------------

/// `source ./lib.sh` / `. ./lib.sh` → path `./lib.sh`, name `lib.sh`.
fn bash_expand_source(source: &[u8], cmd: Node<'_>) -> Option<ImportedItem> {
    // A `command` carries the sourced file in its `argument:` field; take the
    // first argument after the `command_name` (`source` / `.`).
    let arg = cmd.child_by_field_name("argument")?;
    let raw = node_text(source, arg);
    let path = raw.trim_matches(['"', '\'']).to_string();
    if path.is_empty() {
        return None;
    }
    let final_seg = path.rsplit('/').next().unwrap_or(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: final_seg.clone(),
        original_name: final_seg,
        is_glob: false,
    })
}

// ---- Lua imports ----------------------------------------------------------

/// `require("lib")` / `require "lib"` → path `lib`, name `lib`.
fn lua_expand_require(source: &[u8], call: Node<'_>) -> Option<ImportedItem> {
    let args = call.child_by_field_name("arguments")?;
    let mut string_node = None;
    for i in 0..args.named_child_count() {
        if let Some(c) = args.named_child(i) {
            if c.kind() == "string" {
                string_node = Some(c);
                break;
            }
        }
    }
    let path = lua_string_text(source, string_node?);
    if path.is_empty() {
        return None;
    }
    let final_seg = path.rsplit(['.', '/']).next().unwrap_or(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: final_seg.clone(),
        original_name: final_seg,
        is_glob: false,
    })
}

fn lua_string_text(source: &[u8], node: Node<'_>) -> String {
    if let Some(frag) = first_child_of_kind(node, "string_content") {
        return node_text(source, frag).to_string();
    }
    let raw = node_text(source, node);
    raw.trim_matches(['"', '\'', '[', ']']).to_string()
}

// ---- Kotlin imports -------------------------------------------------------

/// `import kotlin.math.max` → path `kotlin.math.max`, name `max`;
/// `import a.b.*` → glob, name `b`.
fn kotlin_expand_import(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    let qi = node
        .named_children(&mut node.walk())
        .find(|c| c.kind() == "qualified_identifier")?;
    let path = node_text(source, qi).to_string();
    if path.is_empty() {
        return None;
    }
    let raw = node_text(source, node);
    let is_glob = raw.trim_end().ends_with(".*") || raw.trim_end().ends_with('*');
    let final_seg = path.rsplit('.').next().unwrap_or(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: final_seg.clone(),
        original_name: final_seg,
        is_glob,
    })
}

// ---- Scala imports --------------------------------------------------------

/// `import scala.collection.mutable.Map` → path `scala.collection.mutable.Map`,
/// name `Map`. The grammar emits one `path:`-field child per dotted segment;
/// the imported name is the final identifier segment.
fn scala_expand_import(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    let mut segments: Vec<&str> = Vec::new();
    for i in 0..node.child_count() {
        let Some(child) = node.child(i) else { continue };
        if node.field_name_for_child(i as u32) == Some("path") && child.kind() == "identifier" {
            segments.push(node_text(source, child));
        }
    }
    if segments.is_empty() {
        return None;
    }
    let path = segments.join(".");
    let final_seg = segments.last().copied().unwrap_or(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: final_seg.clone(),
        original_name: final_seg,
        is_glob: false,
    })
}

// ---- Swift imports --------------------------------------------------------

/// `import Foundation` → path `Foundation`, name `Foundation`.
fn swift_expand_import(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    let ident = node
        .named_children(&mut node.walk())
        .find(|c| c.kind() == "identifier")?;
    let path = node_text(source, ident).to_string();
    if path.is_empty() {
        return None;
    }
    let final_seg = path.rsplit('.').next().unwrap_or(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: final_seg.clone(),
        original_name: final_seg,
        is_glob: false,
    })
}

// ---- Zig imports ----------------------------------------------------------

/// `@import("std")` → path `std`, name `std`. The basename of a relative
/// path (`@import("foo/bar.zig")` → `bar.zig`) keys the imported name.
fn zig_expand_import(source: &[u8], node: Node<'_>) -> Option<ImportedItem> {
    let args = node
        .named_children(&mut node.walk())
        .find(|c| c.kind() == "arguments")?;
    let string_node = args
        .named_children(&mut args.walk())
        .find(|c| c.kind() == "string")?;
    let path = zig_string_text(source, string_node);
    if path.is_empty() {
        return None;
    }
    let final_seg = path.rsplit('/').next().unwrap_or(&path).to_string();
    Some(ImportedItem {
        path,
        imported_name: final_seg.clone(),
        original_name: final_seg,
        is_glob: false,
    })
}

fn zig_string_text(source: &[u8], node: Node<'_>) -> String {
    if let Some(frag) = first_child_of_kind(node, "string_content") {
        return node_text(source, frag).to_string();
    }
    let raw = node_text(source, node);
    raw.trim_matches('"').to_string()
}

// ---- R imports ------------------------------------------------------------

/// `library(stats)` / `require("stats")` → path `stats`, name `stats`. The
/// first argument's identifier or string value names the package.
fn r_expand_library(source: &[u8], call: Node<'_>) -> Option<ImportedItem> {
    let args = call.child_by_field_name("arguments")?;
    let arg = args
        .named_children(&mut args.walk())
        .find(|c| c.kind() == "argument")?;
    let value = arg.child_by_field_name("value")?;
    let path = match value.kind() {
        "string" => r_string_text(source, value),
        _ => node_text(source, value).to_string(),
    };
    if path.is_empty() {
        return None;
    }
    Some(ImportedItem {
        path: path.clone(),
        imported_name: path.clone(),
        original_name: path,
        is_glob: false,
    })
}

fn r_string_text(source: &[u8], node: Node<'_>) -> String {
    if let Some(frag) = first_child_of_kind(node, "string_content") {
        return node_text(source, frag).to_string();
    }
    let raw = node_text(source, node);
    raw.trim_matches(['"', '\'']).to_string()
}

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/// The first direct child (named or unnamed) of `node` whose kind is `kind`.
fn first_child_of_kind<'t>(node: Node<'t>, kind: &str) -> Option<Node<'t>> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if child.kind() == kind {
                return Some(child);
            }
        }
    }
    None
}
