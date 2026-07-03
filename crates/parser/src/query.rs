//! Compiled tree-sitter queries for the extraction passes.
//!
//! Phase 3 ships Rust queries only. Each query is a small S-expression
//! over the language's grammar that captures the AST nodes we want to
//! extract. The extraction module walks the matched nodes and turns
//! them into `ExtractedNode` / `ExtractedEdge` values.

use tree_sitter::Query;

/// The extraction passes we run for every supported language.
///
/// `Definitions`, `Imports`, and `Calls` are the original three. `TypeRefs`
/// and `Usages` are ported from upstream `extract_type_refs.c` /
/// `extract_usages.c`: they capture references to types in signatures and
/// fields (`TYPE_REF` edges) and bare identifier usages (`USES` edges) so the
/// indexer's name-based resolver can support "find usages".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    Definitions,
    Imports,
    Calls,
    TypeRefs,
    Usages,
    TypeAssigns,
    Inheritance,
}

impl QueryKind {
    pub fn name(self) -> &'static str {
        match self {
            QueryKind::Definitions => "definitions",
            QueryKind::Imports => "imports",
            QueryKind::Calls => "calls",
            QueryKind::TypeRefs => "type_refs",
            QueryKind::Usages => "usages",
            QueryKind::TypeAssigns => "type_assigns",
            QueryKind::Inheritance => "inheritance",
        }
    }
}

/// A pre-compiled query plus its capture names.
///
/// We keep these around (rather than recompiling per file) because
/// tree-sitter query compilation is the most expensive part of an
/// indexer pass.
#[derive(Debug)]
pub struct CompiledQuery {
    pub kind: QueryKind,
    pub language: tree_sitter::Language,
    pub query: Query,
    /// Capture index → name, in declaration order. The extraction
    /// passes use these names to decide what each capture means.
    pub capture_names: Vec<String>,
}

impl CompiledQuery {
    pub fn new(
        kind: QueryKind,
        language: tree_sitter::Language,
        source: &str,
    ) -> Result<Self, tree_sitter::QueryError> {
        let query = Query::new(&language, source)?;
        let capture_names: Vec<String> = query
            .capture_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        Ok(Self {
            kind,
            language,
            query,
            capture_names,
        })
    }

    pub fn capture_index(&self, name: &str) -> Option<u32> {
        self.capture_names
            .iter()
            .position(|n| n == name)
            .map(|i| i as u32)
    }
}

/// Rust query sources. These are deliberately small — they cover the
/// structures a typical agent asks about (function/method/struct/trait
/// definitions, `use` statements, function-call expressions).
pub mod rust_queries {
    /// Captures:
    /// - `name` — the identifier of a top-level type/function definition
    /// - `def` — the entire definition (function_item, struct_item, …)
    /// - `field` — a struct/union `field_declaration` name (→ `Field` node,
    ///   ported from the C reference's `extract_class_fields`, which emits a
    ///   `Field` def for every typed struct field)
    /// - `var` — a top-level `const_item`/`static_item` name (→ `Variable`
    ///   node, ported from the C reference's `extract_variables`, whose
    ///   `rust_var_types` = `{static_item, const_item}`; the extractor filters
    ///   to module-level items so an impl/trait associated const stays an
    ///   `AssocConst`)
    pub const DEFINITIONS: &str = r#"
        (function_item
            name: (identifier) @name) @def

        (struct_item
            name: (type_identifier) @name) @def

        (union_item
            name: (type_identifier) @name) @def

        (enum_item
            name: (type_identifier) @name) @def

        (trait_item
            name: (type_identifier) @name) @def

        (impl_item
            type: (type_identifier) @name) @def

        (type_item
            name: (type_identifier) @name) @def

        (field_declaration
            name: (field_identifier) @field
            type: (_)) @field_decl

        (const_item
            name: (identifier) @var) @var_item

        (static_item
            name: (identifier) @var) @var_item
    "#;

    /// Captures:
    /// - `path` — the use-tree root
    pub const IMPORTS: &str = r#"
        (use_declaration
            argument: (_) @path)
    "#;

    /// Captures:
    /// - `callee` — the FINAL callee identifier of the called
    ///   expression. For a bare call `bare()` this is `bare`; for a
    ///   method call `x.do_it()` it is `do_it`; for a scoped call
    ///   `helper::do_it()` it is `do_it` (the `name` field of the
    ///   `scoped_identifier`, NOT its first `path` segment); for a
    ///   constructor `Foo::new()` it is `new`.
    ///
    /// Capturing the final segment is what lets the indexer's
    /// name-based cross-file resolver find the callee's definition in
    /// another file — the first segment (`helper`, `Foo`) is a
    /// module/type path, not the function name.
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @callee)
        (call_expression
            function: (field_expression
                field: (field_identifier) @callee))
        (call_expression
            function: (scoped_identifier
                name: (identifier) @callee))
        (call_expression
            function: (generic_function
                function: (scoped_identifier
                    name: (identifier) @callee)))
        (call_expression
            function: (generic_function
                function: (identifier) @callee))
    "#;

    /// Type references in signatures, fields, returns and let-bindings.
    /// Ported from upstream `extract_type_refs.c` (param/return/field/local
    /// type annotations).
    ///
    /// Captures:
    /// - `type` — a `type_identifier` node appearing in a position that
    ///   names a type: a function parameter type, a return type, a struct
    ///   field type, or a `let x: T` binding. The extractor strips
    ///   wrapper nodes (references, generics, slices) by relying on the
    ///   query matching the inner `type_identifier`, and discards builtin
    ///   primitive types.
    ///
    /// We match the `type_identifier` wherever it sits inside a
    /// `(reference_type)`, `(generic_type)`, `(scoped_type_identifier)`,
    /// etc., because tree-sitter descends into those for us. The single
    /// `(type_identifier) @type` pattern (anchored only by being inside a
    /// type position via the parent patterns below) over-captures, so we
    /// enumerate the concrete type-bearing positions instead.
    pub const TYPE_REFS: &str = r#"
        (parameter
            type: (_) @type)

        (function_item
            return_type: (_) @type)

        (field_declaration
            type: (_) @type)

        (let_declaration
            type: (_) @type)
    "#;

    /// Bare identifier usages — variable / function / type *uses* that are
    /// not themselves definitions, calls, or imports. Ported from upstream
    /// `extract_usages.c`. The extractor filters out identifiers that are
    /// definition names, the callee of a call, or inside a `use`
    /// declaration, and drops keywords/builtins.
    ///
    /// Captures:
    /// - `use` — an `identifier` / `field_identifier` reference node.
    pub const USAGES: &str = r#"
        (identifier) @use
        (field_identifier) @use
    "#;

    /// Declared-type assignments. Ported from upstream `extract_type_assigns.c`
    /// (a variable's declared type). Where upstream infers the type from a
    /// constructor on the RHS (`let x = Foo::new()`), the Rust port captures
    /// the *explicit* type annotation, which is unambiguous and matches the
    /// task's stated intent (`let x: T = expr;`, plus field / const / static
    /// type annotations).
    ///
    /// Captures the whole declaration node as `@assign` so the extractor can
    /// read its `pattern`/`name` (the variable) and `type` (the declared
    /// type) fields. We only capture declarations that carry a `type:` field
    /// so an annotation is guaranteed to be present.
    ///
    /// Captures:
    /// - `assign` — a `let_declaration`, `const_item`, `static_item`, or
    ///   `field_declaration` that has an explicit type annotation.
    pub const TYPE_ASSIGNS: &str = r#"
        (let_declaration
            type: (_)) @assign

        (const_item
            type: (_)) @assign

        (static_item
            type: (_)) @assign

        (field_declaration
            type: (_)) @assign
    "#;

    /// Inheritance + enum-member + associated-item structural captures.
    /// Ported from upstream `extract_defs.c` base-class / enum-member / impl
    /// handling.
    ///
    /// We capture three families in one pass so the extractor can resolve the
    /// enclosing owner (impl/trait/enum) once per match:
    ///
    /// - `impl_trait` — an `impl_item` that carries a `trait:` field, i.e. a
    ///   *trait impl* (`impl Trait for Type`). The `trait_name` and `impl_type`
    ///   captures give the trait and the implementing type so the extractor can
    ///   emit an `IMPLEMENTS` edge from the type to the trait. An inherent
    ///   `impl Type { ... }` carries no `trait:` field and so does NOT match
    ///   here (no IMPLEMENTS edge), matching upstream's trait-impl distinction.
    ///
    /// - `enum_variant` — one capture per variant inside an `enum_item`, with
    ///   `enum_name` for the owning enum so the extractor can build the
    ///   `{file}::{Enum}::{Variant}` qname and a DEFINES/MEMBER edge.
    ///
    /// - `assoc_const` / `assoc_type` — associated `const`/`type` items inside
    ///   an `impl`/`trait` block. (Associated fns/methods are already captured
    ///   by the Definitions pass.) These get qnames owned by the enclosing
    ///   impl/trait type, mirroring the method qname scheme.
    pub const INHERITANCE: &str = r#"
        (impl_item
            trait: (_) @trait_name
            type: (_) @impl_type) @impl_trait

        (enum_item
            name: (type_identifier) @enum_name
            body: (enum_variant_list
                (enum_variant
                    name: (identifier) @enum_variant)))

        (const_item
            name: (identifier) @assoc_const) @assoc_const_item

        (associated_type
            name: (type_identifier) @assoc_type) @assoc_type_item

        (type_item
            name: (type_identifier) @assoc_type) @assoc_type_item
    "#;
}

/// Python query sources. These mirror the Rust passes at the level Python's
/// grammar supports: definitions (functions + classes, methods owned by their
/// class), calls (final callee identifier), and imports (`import` /
/// `from x import y`). Docstrings are handled structurally in the extractor
/// (first string statement in a def/class body), not via a query.
pub mod python_queries {
    /// Captures:
    /// - `name` — the identifier of a `function_definition` / `class_definition`
    /// - `def` — the entire definition node
    ///
    /// Methods are distinguished from free functions in the extractor by
    /// walking ancestors to find an enclosing `class_definition` (mirroring the
    /// Rust impl/trait ownership walk).
    pub const DEFINITIONS: &str = r#"
        (function_definition
            name: (identifier) @name) @def

        (class_definition
            name: (identifier) @name) @def
    "#;

    /// Captures:
    /// - `callee` — the FINAL callee identifier of a call expression. For a
    ///   bare call `bare()` this is `bare`; for an attribute call `x.do_it()`
    ///   it is `do_it` (the `attribute` field of the `attribute` node);
    ///   for a chained call `a.b.c()` it is the final `c`.
    ///
    /// Capturing the final segment is what lets the indexer's name-based
    /// cross-file resolver find the callee's definition in another file — the
    /// receiver (`x`, `a.b`) is an object/module path, not the function name.
    pub const CALLS: &str = r#"
        (call
            function: (identifier) @callee)
        (call
            function: (attribute
                attribute: (identifier) @callee))
    "#;

    /// Captures the whole import statement so the extractor can expand it into
    /// one imported name per binding.
    ///
    /// - `import` — an `import_statement` (`import a`, `import a.b as c`,
    ///   `import a, b`)
    /// - `from_import` — an `import_from_statement`
    ///   (`from x import y`, `from x import y as z`, `from x import *`)
    pub const IMPORTS: &str = r#"
        (import_statement) @import
        (import_from_statement) @from_import
    "#;
}

/// JavaScript query sources. These mirror the Rust/Python passes at the level
/// JavaScript's grammar supports: definitions (function declarations, arrow /
/// function expressions assigned to a `const`/`let`/`var`, classes + methods),
/// calls (final callee identifier), and imports (`import` statements +
/// `require()` calls). JSDoc docstrings (`/** … */`) are handled structurally
/// in the extractor (the leading comment sibling), not via a query.
///
/// The tree-sitter-javascript grammar parses JSX, so the same queries cover
/// `.jsx` / `.mjs` / `.cjs`.
pub mod js_queries {
    /// Captures:
    /// - `name` — the identifier of a `function_declaration` /
    ///   `class_declaration`, the `name:` of a method, or the binding name of a
    ///   `variable_declarator` whose value is an arrow / function expression.
    /// - `def` — the entire definition node.
    ///
    /// Methods (`method_definition` inside a `class_body`) are owned by their
    /// enclosing class in the extractor (mirroring the Rust impl / Python class
    /// ownership walk). A `variable_declarator` only counts as a definition when
    /// its `value` is an `arrow_function` or `function_expression`, so plain
    /// data bindings are not treated as functions.
    pub const DEFINITIONS: &str = r#"
        (function_declaration
            name: (identifier) @name) @def

        (class_declaration
            name: (identifier) @name) @def

        (method_definition
            name: (property_identifier) @name) @def

        (variable_declarator
            name: (identifier) @name
            value: [(arrow_function) (function_expression)]) @def
    "#;

    /// Captures:
    /// - `callee` — the FINAL callee identifier of a call expression. For a
    ///   bare call `bare()` this is `bare`; for a member call `x.do_it()` it is
    ///   `do_it` (the `property` field of the `member_expression`); for a
    ///   chained call `a.b.c()` it is the final `c`.
    ///
    /// `require(...)` is matched here too (its callee is the bare `require`),
    /// but the imports pass owns require-as-import; the extractor drops the
    /// `require` callee from CALLS so it is not double-counted.
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @callee)
        (call_expression
            function: (member_expression
                property: (property_identifier) @callee))
    "#;

    /// Captures whole import-bearing statements so the extractor can expand
    /// each into one bound name.
    ///
    /// - `import` — an `import_statement` (`import a from "m"`,
    ///   `import {a, b as c} from "m"`, `import * as ns from "m"`,
    ///   `import "m"` side-effect-only).
    /// - `require` — a `variable_declarator` whose value is a `require("m")`
    ///   call (`const x = require("m")`, `const {a} = require("m")`).
    pub const IMPORTS: &str = r#"
        (import_statement) @import

        (variable_declarator
            value: (call_expression
                function: (identifier) @require_fn
                (#eq? @require_fn "require"))) @require
    "#;
}

/// TypeScript query sources. These extend the JavaScript passes with the extra
/// definition forms TypeScript adds — `interface`, `type` alias, and `enum` —
/// and otherwise share the JavaScript call / import shapes (the TS grammar is a
/// superset). The same module is used for both the plain TypeScript and the TSX
/// grammar; both accept these node kinds.
pub mod ts_queries {
    /// Captures (superset of [`super::js_queries::DEFINITIONS`]):
    /// - `name` / `def` — functions, classes, methods, and arrow / function
    ///   expressions assigned to a binding (as in JavaScript).
    /// - plus `interface_declaration`, `type_alias_declaration`, and
    ///   `enum_declaration` (TypeScript-only type definitions).
    pub const DEFINITIONS: &str = r#"
        (function_declaration
            name: (identifier) @name) @def

        (class_declaration
            name: (type_identifier) @name) @def

        (method_definition
            name: (property_identifier) @name) @def

        (variable_declarator
            name: (identifier) @name
            value: [(arrow_function) (function_expression)]) @def

        (interface_declaration
            name: (type_identifier) @name) @def

        (type_alias_declaration
            name: (type_identifier) @name) @def

        (enum_declaration
            name: (identifier) @name) @def
    "#;

    /// Same call shape as JavaScript: final callee identifier of a bare or
    /// member call.
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @callee)
        (call_expression
            function: (member_expression
                property: (property_identifier) @callee))
    "#;

    /// Same import shape as JavaScript: `import` statements plus
    /// `const x = require("m")`.
    pub const IMPORTS: &str = r#"
        (import_statement) @import

        (variable_declarator
            value: (call_expression
                function: (identifier) @require_fn
                (#eq? @require_fn "require"))) @require
    "#;
}

/// Go query sources. These mirror the Rust/Python/JS passes at the level Go's
/// grammar supports: definitions (`function_declaration` + `method_declaration`
/// owned by the receiver type, plus `type_declaration` structs/interfaces),
/// calls (final callee identifier of a bare or selector call), and imports
/// (`import_spec` -> one IMPORTS edge per imported package). Doc comments
/// (leading `//` lines) are handled structurally in the extractor, not via a
/// query.
pub mod go_queries {
    /// Captures:
    /// - `name` -- the identifier / field_identifier / type_identifier naming
    ///   the definition.
    /// - `def` -- the entire definition node (`function_declaration`,
    ///   `method_declaration`, or `type_spec`).
    ///
    /// A `method_declaration` carries a `receiver:` parameter list; the
    /// extractor reads the receiver's base type so the method qname is owned by
    /// that type (`{file}::{RecvType}::{name}`). A `type_spec` inside a
    /// `type_declaration` names a struct / interface / alias.
    pub const DEFINITIONS: &str = r#"
        (function_declaration
            name: (identifier) @name) @def

        (method_declaration
            name: (field_identifier) @name) @def

        (type_spec
            name: (type_identifier) @name) @def
    "#;

    /// Captures:
    /// - `callee` -- the FINAL callee identifier of a call expression. For a
    ///   bare call `add()` this is `add`; for a selector call `fmt.Println()`
    ///   it is the `field` of the `selector_expression` (`Println`), which is
    ///   what the cross-file resolver keys on -- the operand (`fmt`) is a
    ///   package/value path, not the function name.
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @callee)
        (call_expression
            function: (selector_expression
                field: (field_identifier) @callee))
    "#;

    /// Captures each `import_spec` so the extractor can emit one IMPORTS edge
    /// per imported package. Covers both the grouped form
    /// (`import ( "fmt"; m "math/rand" )`) and the single form
    /// (`import "fmt"`), since both produce `import_spec` nodes.
    ///
    /// - `import` -- an `import_spec` (`path:` string literal, optional `name:`
    ///   alias / `.` / `_`).
    pub const IMPORTS: &str = r#"
        (import_spec) @import
    "#;

    /// Usage references. Ported from upstream `extract_usages.c`
    /// (`is_reference_node`, Go arm ~L60): a Go reference node is an
    /// `identifier`, `type_identifier`, `field_identifier`, or
    /// `package_identifier`. The extractor drops any capture that is a
    /// definition name, sits inside a call or import node, or is a Go keyword,
    /// then emits a `USAGE` edge whose `ref_name` the indexer resolves against
    /// every registered symbol (dropping non-unique names) — mirroring the C
    /// `pass_usages` name-based resolution.
    ///
    /// Captures:
    /// - `use` — an `identifier` / `type_identifier` / `field_identifier` /
    ///   `package_identifier` reference node.
    pub const USAGES: &str = r#"
        (identifier) @use
        (type_identifier) @use
        (field_identifier) @use
        (package_identifier) @use
    "#;
}

/// Ruby query sources. These mirror the passes at the level Ruby's grammar
/// supports: definitions (`method` / `singleton_method`, plus `class` /
/// `module` which own nested method qnames), calls (`call` -> method name),
/// and `require` / `require_relative` -> imports. Leading `#` comment blocks
/// are handled structurally in the extractor, not via a query.
pub mod ruby_queries {
    /// Captures:
    /// - `name` -- the identifier / constant naming the definition.
    /// - `def` -- the entire definition node (`method`, `singleton_method`,
    ///   `class`, or `module`).
    ///
    /// A `method` nested inside a `class`/`module` body is owned by that
    /// class/module in the extractor (qname `{file}::{Class}::{name}`); a
    /// `singleton_method` (`def self.x`) likewise. A free `method` at the top
    /// level is `{file}::Function::{name}`.
    pub const DEFINITIONS: &str = r#"
        (method
            name: (_) @name) @def

        (singleton_method
            name: (_) @name) @def

        (class
            name: (constant) @name) @def

        (module
            name: (constant) @name) @def
    "#;

    /// Captures:
    /// - `callee` -- the method name of a `call`. Ruby parses both
    ///   `receiver.method(args)` and the command form `helper arg1, arg2` as a
    ///   `call` node with a `method:` field, so a single pattern covers both. A
    ///   bare zero-argument reference (`foo` with no args / parens) parses as a
    ///   plain `identifier`, not a `call`, and is therefore not captured -- this
    ///   matches the other languages, which also only treat explicit call
    ///   expressions as CALLS.
    pub const CALLS: &str = r#"
        (call
            method: (identifier) @callee)
    "#;

    /// Captures `require` / `require_relative` calls so the extractor can emit
    /// one IMPORTS edge per required path. Both are ordinary `call`s whose
    /// `method:` is the bare identifier `require` / `require_relative`; the
    /// `(#match?)` predicate restricts the capture to those two names.
    ///
    /// - `require` -- a `call` to `require` / `require_relative`. The extractor
    ///   reads the first string argument as the imported path.
    pub const IMPORTS: &str = r#"
        (call
            method: (identifier) @require_fn
            (#match? @require_fn "^(require|require_relative)$")) @require
    "#;
}

/// Java query sources. These mirror the passes at the level Java's grammar
/// supports: definitions (`class` / `interface` / `enum` declarations, plus
/// `method` / `constructor` declarations owned by their enclosing class),
/// calls (`method_invocation` -> final method name), and imports
/// (`import_declaration` -> final segment). Javadoc `/** … */` block comments
/// are handled structurally in the extractor, not via a query.
pub mod java_queries {
    /// Captures:
    /// - `name` -- the `identifier` naming the definition.
    /// - `def` -- the entire definition node (`class_declaration`,
    ///   `interface_declaration`, `enum_declaration`, `method_declaration`,
    ///   or `constructor_declaration`).
    ///
    /// A `method_declaration` / `constructor_declaration` nested inside a
    /// `class_body` / `interface_body` / `enum_body` is owned by that type in
    /// the extractor (qname `{file}::{Class}::{name}`).
    pub const DEFINITIONS: &str = r#"
        (class_declaration
            name: (identifier) @name) @def

        (interface_declaration
            name: (identifier) @name) @def

        (enum_declaration
            name: (identifier) @name) @def

        (method_declaration
            name: (identifier) @name) @def

        (constructor_declaration
            name: (identifier) @name) @def
    "#;

    /// Captures:
    /// - `callee` -- the FINAL method name of a `method_invocation`. Java's
    ///   grammar puts the called method's name in the `name:` field regardless
    ///   of whether the call is bare (`helperFn()`), qualified by an object
    ///   (`obj.greet()`), or static (`Helper.run()` / `a.b.c()`), so a single
    ///   pattern keyed on `name:` covers every form. The receiver (`obj`,
    ///   `Helper`, `a.b`) is an object/type path, not the function name, so it
    ///   is not captured.
    /// - `callee` -- the constructed type of an `object_creation_expression`
    ///   (`new Foo(...)`). The C reference counts constructor calls as CALLS
    ///   (`java_call_types` includes `object_creation_expression`, resolved via
    ///   `extract_constructor_callee` to the bare type name), so a `new Foo()`
    ///   yields a CALLS edge to `Foo` just like `Foo.bar()` does. The `type:`
    ///   field holds the bare `type_identifier`; a generic `new Foo<T>()` wraps
    ///   it in a `generic_type`, whose leading `type_identifier` is the name.
    pub const CALLS: &str = r#"
        (method_invocation
            name: (identifier) @callee)

        (object_creation_expression
            type: (type_identifier) @callee)

        (object_creation_expression
            type: (generic_type
                (type_identifier) @callee))
    "#;

    /// Captures each `import_declaration` so the extractor can emit one IMPORTS
    /// edge per import. The imported name is the final segment of the imported
    /// path (`java.util.List` -> `List`, `java.util.Map.Entry` -> `Entry`).
    ///
    /// - `import` -- an `import_declaration` (covers plain, `static`, and
    ///   on-demand `.*` imports).
    pub const IMPORTS: &str = r#"
        (import_declaration) @import
    "#;
}

/// C query sources. These mirror the passes at the level C's grammar supports:
/// definitions (`function_definition`, plus `struct` / `union` / `enum`
/// specifiers and `typedef`s), calls (`call_expression` -> final callee
/// identifier), and includes (`#include "x"` / `<x>` -> header basename). A
/// leading block/line comment is handled structurally in the extractor.
pub mod c_queries {
    /// Captures:
    /// - `def` -- the entire definition node (`function_definition`,
    ///   `struct_specifier`, `union_specifier`, `enum_specifier`, or
    ///   `type_definition`).
    ///
    /// The *name* is resolved structurally in the extractor rather than via a
    /// query capture, because C nests the function name inside a
    /// `function_declarator` (possibly behind pointer declarators) and the
    /// tagged-type name is a `type_identifier` on the specifier. Capturing the
    /// whole node and walking to the name keeps the query simple and robust.
    pub const DEFINITIONS: &str = r#"
        (function_definition) @def

        (struct_specifier
            name: (type_identifier)) @def

        (union_specifier
            name: (type_identifier)) @def

        (enum_specifier
            name: (type_identifier)) @def

        (type_definition) @def
    "#;

    /// Captures:
    /// - `callee` -- the FINAL callee identifier of a `call_expression`. For a
    ///   bare call `helper()` this is `helper`; for a member call `obj.fn()` /
    ///   `ptr->fn()` it is the `field` of the `field_expression` (`fn`), which
    ///   is what the cross-file resolver keys on -- the receiver (`obj`, `ptr`)
    ///   is a value path, not the function name.
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @callee)
        (call_expression
            function: (field_expression
                field: (field_identifier) @callee))
    "#;

    /// Captures each `preproc_include` so the extractor can emit one IMPORTS
    /// edge per include. The imported name is the header basename
    /// (`<stdio.h>` -> `stdio.h`, `"sub/helper.h"` -> `helper.h`).
    ///
    /// - `include` -- a `preproc_include` (`#include <x>` or `#include "x"`).
    pub const IMPORTS: &str = r#"
        (preproc_include) @include
    "#;
}

/// C++ query sources. These extend the C passes with the extra definition forms
/// C++ adds -- `class_specifier` and `namespace_definition` -- and otherwise
/// share the C call / include shapes plus a `qualified_identifier` callee form
/// (`geo::helper()`) and `using` declarations as imports.
pub mod cpp_queries {
    /// Captures (superset of [`super::c_queries::DEFINITIONS`]):
    /// - `def` -- a `function_definition` (free function, in-class method, or
    ///   out-of-line `Class::method` definition), `struct` / `union` / `enum`
    ///   specifier, `type_definition`, plus the C++-only `class_specifier` and
    ///   `namespace_definition`.
    ///
    /// Names are resolved structurally in the extractor (the function name may
    /// be a plain `identifier`, a `field_identifier` for an in-class method, or
    /// a `qualified_identifier` for an out-of-line `Class::method` definition).
    pub const DEFINITIONS: &str = r#"
        (function_definition) @def

        (struct_specifier
            name: (type_identifier)) @def

        (union_specifier
            name: (type_identifier)) @def

        (enum_specifier
            name: (type_identifier)) @def

        (type_definition) @def

        (class_specifier
            name: (type_identifier)) @def

        (namespace_definition
            name: (namespace_identifier)) @def
    "#;

    /// Captures:
    /// - `callee` -- the FINAL callee identifier of a `call_expression`. Covers
    ///   bare calls (`helper()`), member calls (`obj.doIt()` / `ptr->run()`)
    ///   via the `field_expression` `field:`, and namespace-qualified calls
    ///   (`geo::helper()`) via the `qualified_identifier` `name:`.
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @callee)
        (call_expression
            function: (field_expression
                field: (field_identifier) @callee))
        (call_expression
            function: (qualified_identifier
                name: (identifier) @callee))
    "#;

    /// Captures `#include`s and `using` declarations so the extractor can emit
    /// one IMPORTS edge each. An `#include` imports a header (basename keyed);
    /// a `using` declaration imports a name (`using std::vector;` -> `vector`)
    /// or a whole namespace (`using namespace std;` -> `std`).
    ///
    /// - `include` -- a `preproc_include`.
    /// - `using` -- a `using_declaration`.
    pub const IMPORTS: &str = r#"
        (preproc_include) @include
        (using_declaration) @using
    "#;
}

/// C# query sources (data-path onboarding). Definitions for class / struct /
/// interface / record / enum and method / constructor (owned by their type);
/// calls via `invocation_expression`; `using` directives as imports. XML-doc
/// `///` and `/** */` comments are handled structurally by the spec engine.
pub mod csharp_queries {
    /// Captures `name` (the definition's `name:` identifier) and `def` (the
    /// whole node), mirroring the other `@name`-capturing languages.
    pub const DEFINITIONS: &str = r#"
        (class_declaration
            name: (identifier) @name) @def

        (struct_declaration
            name: (identifier) @name) @def

        (interface_declaration
            name: (identifier) @name) @def

        (record_declaration
            name: (identifier) @name) @def

        (enum_declaration
            name: (identifier) @name) @def

        (method_declaration
            name: (identifier) @name) @def

        (constructor_declaration
            name: (identifier) @name) @def
    "#;

    /// Captures the final callee identifier of an `invocation_expression`:
    /// bare (`Setup()`) or member (`Helper.Run()`, via the
    /// `member_access_expression` `name:`); and the constructed type of an
    /// `object_creation_expression` (`new Foo()` / `new Foo<T>()`), which the C
    /// reference (`extract_constructor_callee`) resolves to the type's
    /// constructor `Method` — matching its per-type constructor call count. The
    /// bare type name is captured so a generic `new List<T>()` keys on `List`.
    pub const CALLS: &str = r#"
        (invocation_expression
            function: (identifier) @callee)
        (invocation_expression
            function: (member_access_expression
                name: (identifier) @callee))
        (object_creation_expression
            type: (identifier) @callee)
        (object_creation_expression
            type: (generic_name
                (identifier) @callee))
    "#;

    /// Captures each `using_directive` as `@import`; the spec engine reads its
    /// qualified name (and optional `name:` alias).
    pub const IMPORTS: &str = r#"
        (using_directive) @import
    "#;
}

/// PHP query sources (data-path onboarding). Definitions for class / interface
/// / trait / enum and function / method (owned by their type); calls via the
/// three call-expression forms; `use` declarations as imports. `/** */`, `//`
/// and `#` comments are handled structurally by the spec engine.
pub mod php_queries {
    /// Captures `name` (the definition's `name:` `name` node) and `def`.
    pub const DEFINITIONS: &str = r#"
        (class_declaration
            name: (name) @name) @def

        (interface_declaration
            name: (name) @name) @def

        (trait_declaration
            name: (name) @name) @def

        (enum_declaration
            name: (name) @name) @def

        (function_definition
            name: (name) @name) @def

        (method_declaration
            name: (name) @name) @def
    "#;

    /// Captures the final callee `name` of a call: bare
    /// (`function_call_expression`), member (`$o->m()` via
    /// `member_call_expression`), or static (`C::m()` via
    /// `scoped_call_expression`).
    pub const CALLS: &str = r#"
        (function_call_expression
            function: (name) @callee)
        (member_call_expression
            name: (name) @callee)
        (scoped_call_expression
            name: (name) @callee)
    "#;

    /// Captures each `namespace_use_declaration` as `@import`; the spec engine
    /// expands plain and grouped `use` clauses (with aliases).
    pub const IMPORTS: &str = r#"
        (namespace_use_declaration) @import
    "#;
}

/// Bash query sources (data-path onboarding). Definitions for
/// `function_definition`; calls for every `command` (its `command_name`); and
/// `source` / `.` builtins as imports (the imports pass owns them, so they are
/// skipped as calls). Leading `#` comments are handled by the spec engine.
pub mod bash_queries {
    /// Captures the function `name:` `word` and the whole `def`.
    pub const DEFINITIONS: &str = r#"
        (function_definition
            name: (word) @name) @def
    "#;

    /// Captures the `word` naming a `command`'s `command_name` as `@callee`.
    /// `source` / `.` callees are dropped by the spec (the imports pass owns
    /// them).
    pub const CALLS: &str = r#"
        (command
            name: (command_name (word) @callee))
    "#;

    /// Captures a `command` whose `command_name` is `source` or `.` as
    /// `@source`; the spec engine reads its first argument as the sourced file.
    pub const IMPORTS: &str = r#"
        (command
            name: (command_name (word) @src_fn)
            (#match? @src_fn "^(source|\.)$")) @source
    "#;
}

/// Lua query sources (data-path onboarding). Definitions for every
/// `function_declaration`, whatever its name shape: a bare `identifier`
/// (`function f()` / `local function f()`), a `dot_index_expression`
/// (`function M.f()` → name `M.f`), or a `method_index_expression`
/// (`function M:f()` → name `M:f`). This mirrors the C reference, whose
/// `func_name_node` reads the whole `name:` field's text as the function name
/// (`resolve_func_name` in `extract_defs.c`), so tables-as-modules and method
/// definitions each surface as a `Function` node (Lua has no class ownership,
/// so nothing is relabelled a `Method`). Calls capture the bare and
/// dotted/method callee forms; `require("…")` calls are imports; leading `--`
/// line comments are docstrings. Anonymous `function_definition`s bound to a
/// name (`local f = function() … end`, `M.f = function() … end`) and
/// module-level `Variable`s / `USAGE` edges are handled by the bespoke
/// `extract_lua` passes (the uniform spec cannot express them).
pub mod lua_queries {
    /// Captures the `name:` of a `function_declaration` — an `identifier`,
    /// `dot_index_expression`, or `method_index_expression` — and the whole
    /// `def`. The spec's `Capture` strategy reads the capture node's *text* as
    /// the function name, so a dotted/method name yields `M.f` / `M:f`
    /// verbatim (matching C's `func_name_node`).
    pub const DEFINITIONS: &str = r#"
        (function_declaration
            name: [
                (identifier)
                (dot_index_expression)
                (method_index_expression)
            ] @name) @def
    "#;

    /// Captures the callee of a `function_call`: a bare `identifier`
    /// (`helper(...)`), a `dot_index_expression` (`M.greet(...)` → `M.greet`),
    /// or a `method_index_expression` (`obj:run(...)` → `obj:run`). C's
    /// `extract_callee_from_fields` reads the `function_call`'s `name:` field
    /// text as the callee, so a dotted call's callee is its whole dotted text
    /// — which resolves to the matching dotted `Function` def node.
    pub const CALLS: &str = r#"
        (function_call
            name: [
                (identifier)
                (dot_index_expression)
                (method_index_expression)
            ] @callee)
    "#;

    /// Captures `require("…")` calls so the spec engine reads the first string
    /// argument as the imported path.
    pub const IMPORTS: &str = r#"
        (function_call
            name: (identifier) @require_fn
            (#eq? @require_fn "require")) @require
    "#;
}

/// Kotlin query sources (data-path onboarding). Definitions for `class` /
/// `interface` (both `class_declaration`) / `object` and functions (owned by
/// their enclosing class/object); calls via the bare-identifier form of
/// `call_expression`; `import` directives as imports. Leading `/** */`
/// KDoc/JSDoc block comments are docstrings.
pub mod kotlin_queries {
    /// Captures the `identifier` naming a `class_declaration`,
    /// `object_declaration`, or `function_declaration`, plus the whole `def`.
    pub const DEFINITIONS: &str = r#"
        (class_declaration
            name: (identifier) @name) @def

        (object_declaration
            name: (identifier) @name) @def

        (function_declaration
            name: (identifier) @name) @def
    "#;

    /// Captures the bare-identifier callee of a `call_expression`
    /// (`helper(...)`). A member call (`Helper.run()`) names the function with
    /// a `navigation_expression`, not a direct identifier, so only bare calls
    /// are captured (the resolver keys on the final callee name).
    pub const CALLS: &str = r#"
        (call_expression
            (identifier) @callee
            (value_arguments))
    "#;

    /// Captures each `import` directive; the spec engine reads its qualified
    /// name's final segment.
    pub const IMPORTS: &str = r#"
        (import) @import
    "#;
}

/// Scala query sources (data-path onboarding). Definitions for `class` /
/// `object` / `trait` and `def` functions (owned by their enclosing
/// class/object/trait); calls via the bare and member forms of
/// `call_expression`; `import` declarations as imports. Leading `/** */`
/// ScalaDoc block comments are docstrings.
pub mod scala_queries {
    /// Captures the `identifier` naming a `function_definition`, plus the whole
    /// `def`. The spec base pass emits exactly the `Method` (owned) / `Function`
    /// (free) node for each `def`; the *type* declarations
    /// (`class_definition` / `object_definition` / `trait_definition` /
    /// `enum_definition` / `type_definition`) are NOT captured here — the
    /// bespoke `extract_scala` second pass owns them so it can reproduce C's
    /// `class_label_for_kind` mapping (object → "Class", trait → "Interface",
    /// enum → "Enum", type → "Type") rather than the uniform spec labels, and
    /// emit the double-counted free `Function` node C keeps for every method
    /// (its `walk_defs` re-walks a Scala `template_body`, which
    /// `push_class_body_children` does not recognise as a class body).
    pub const DEFINITIONS: &str = r#"
        (function_definition
            name: (identifier) @name) @def

        (function_declaration
            name: (identifier) @name) @def
    "#;

    /// Captures the final callee identifier of a `call_expression`: bare
    /// (`helper(...)`) or member (`Helper.run()` via the `field_expression`
    /// `field:`).
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @callee)
        (call_expression
            function: (field_expression
                field: (identifier) @callee))
    "#;

    /// Captures each `import_declaration` as `@import`; the spec engine reads
    /// the final `path:` segment.
    pub const IMPORTS: &str = r#"
        (import_declaration) @import
    "#;
}

/// Swift query sources (data-path onboarding). Definitions for `class` /
/// `struct` / `enum` (all `class_declaration`, distinguished by
/// `declaration_kind:`) and `func` functions (owned by their enclosing type);
/// calls via the bare and member forms of `call_expression`; `import`
/// declarations as imports. Leading `///` line comments are docstrings.
pub mod swift_queries {
    /// Captures the `type_identifier` naming a `class_declaration` (class /
    /// struct / enum) or the `simple_identifier` naming a
    /// `function_declaration`, plus the whole `def`.
    pub const DEFINITIONS: &str = r#"
        (class_declaration
            name: (type_identifier) @name) @def

        (function_declaration
            name: (simple_identifier) @name) @def
    "#;

    /// Captures the callee of a `call_expression`: bare (`helper(...)` via the
    /// leading `simple_identifier`) or member (`Helper.run()` via the
    /// `navigation_expression`'s final `navigation_suffix`).
    pub const CALLS: &str = r#"
        (call_expression
            (simple_identifier) @callee
            (call_suffix))
        (call_expression
            (navigation_expression
                suffix: (navigation_suffix
                    suffix: (simple_identifier) @callee))
            (call_suffix))
    "#;

    /// Captures each `import_declaration` as `@import`; the spec engine reads
    /// its module identifier.
    pub const IMPORTS: &str = r#"
        (import_declaration) @import
    "#;
}

/// Zig query sources (data-path onboarding). Definitions for
/// `function_declaration` (always free — Zig methods live in anonymous
/// `struct_declaration`s named on an enclosing `variable_declaration`, so
/// ownership is not modelled); calls via the bare-identifier form of
/// `call_expression`; `@import("…")` builtins as imports. Leading `///` doc
/// comments are docstrings.
pub mod zig_queries {
    /// Captures the `identifier` naming a `function_declaration` and the whole
    /// `def`. Struct/enum/union methods are `function_declaration`s nested in a
    /// `struct_declaration` / `enum_declaration` / `union_declaration`; the C
    /// reference flattens every one to a free `Function` (its class-def name
    /// resolution fails on tree-sitter-zig's unnamed container nodes, so
    /// `push_class_body_children` re-walks the methods at file scope), so a
    /// single `function_declaration` capture — matched anywhere in the tree —
    /// already reproduces C's Function set for methods. `extract_zig` adds the
    /// `test_declaration` Functions on top (C's `zig_func_types` lists them,
    /// named from the test string).
    pub const DEFINITIONS: &str = r#"
        (function_declaration
            name: (identifier) @name) @def
    "#;

    /// Zig owns its CALLS pass in `extract_zig` (`emit_zig_calls`), a port of
    /// the C `walk_calls` + `extract_callee_from_fields`: a `call_expression`
    /// whose callee is a bare `identifier` (`helper(...)`) OR a
    /// `field_expression` (`recv.method(...)` / `mod.func(...)`, resolved by the
    /// trailing method identifier). The capture is deliberately NOT named
    /// `@callee`, so the shared `spec_calls` pass — which only consumes
    /// `@callee` — emits nothing for Zig and the bespoke pass is the sole
    /// source of CALLS.
    pub const CALLS: &str = r#"
        (call_expression
            function: (identifier) @zig_call)
    "#;

    /// Captures `@import("…")` builtin calls so the spec engine reads the first
    /// string argument as the imported path.
    pub const IMPORTS: &str = r#"
        (builtin_function
            (builtin_identifier) @builtin
            (#eq? @builtin "@import")) @import
    "#;
}

/// R query sources (data-path onboarding). Definitions for top-level
/// `name <- function(...) {...}` assignments (captured as the whole
/// `binary_operator`, with the name resolved structurally from the left-hand
/// identifier); calls via `call` with a bare-identifier function; `library(…)`
/// / `require(…)` calls as imports. Leading `#` line comments are docstrings.
pub mod r_queries {
    /// Captures the whole assignment `binary_operator` whose right-hand side is
    /// a `function_definition` as `@def`. The name (left-hand `identifier`) is
    /// resolved structurally by the spec engine.
    pub const DEFINITIONS: &str = r#"
        (binary_operator
            lhs: (identifier)
            rhs: (function_definition)) @def
    "#;

    /// Captures the bare-identifier callee of a `call` (`helper(...)`).
    ///
    /// NOTE: the capture is deliberately named `@r_call` (NOT the `@callee`
    /// the shared `spec_calls` consumes), so the generic calls pass emits
    /// nothing for R. R owns its CALLS pass in `extract_r`
    /// (`emit_r_calls`), which mirrors the C reference `walk_calls` +
    /// `cbm_enclosing_func_qn`: the enclosing source is the nearest
    /// `function_definition`'s assigned name, or the file Module node at
    /// module scope. The shared pass cannot express this because R's
    /// `enclosing_callable_qname` walks up to the nearest `binary_operator`,
    /// which for a call nested in a non-function assignment
    /// (`x <- sapply(...)`) resolves to `None` and drops the edge.
    pub const CALLS: &str = r#"
        (call
            function: (identifier) @r_call)
    "#;

    /// Captures `library(pkg)` / `require(pkg)` calls so the spec engine reads
    /// the first argument as the imported package.
    pub const IMPORTS: &str = r#"
        (call
            function: (identifier) @lib_fn
            (#match? @lib_fn "^(library|require|requireNamespace)$")) @require
    "#;
}

/// Compile a language's query set ONCE and cache it for every subsequent
/// file. tree-sitter `Query::new` is the single most expensive part of an
/// extraction pass; the per-language `*_query_set()` builders below recompiled
/// every query for EVERY file, which was the dominant cold-index cost (~1.3 s
/// for a 423-file repo — 11x more than the parse itself). This memoises each
/// language's compiled set behind its own `OnceLock`, so the extract path
/// compiles queries exactly once per process. `CompiledQuery` is `Sync`
/// (immutable `Query` + `Vec<String>`), so sharing the `&'static` slice across
/// the parallel-extract worker threads is safe.
pub fn cached_query_set(
    language: &crate::Language,
) -> Result<&'static [CompiledQuery], tree_sitter::QueryError> {
    use crate::Language;
    macro_rules! once {
        ($lock:ident, $build:expr) => {{
            static $lock: std::sync::OnceLock<Vec<CompiledQuery>> = std::sync::OnceLock::new();
            if let Some(v) = $lock.get() {
                return Ok(v.as_slice());
            }
            let v = $build?;
            Ok($lock.get_or_init(|| v).as_slice())
        }};
    }
    match language {
        Language::Rust => once!(RUST, rust_query_set()),
        Language::Python => once!(PY, python_query_set()),
        Language::JavaScript => once!(JS, javascript_query_set()),
        Language::TypeScript { tsx: false } => once!(TS, typescript_query_set(false)),
        Language::TypeScript { tsx: true } => once!(TSX, typescript_query_set(true)),
        Language::Go => once!(GO, go_query_set()),
        Language::Ruby => once!(RUBY, ruby_query_set()),
        Language::Java => once!(JAVA, java_query_set()),
        Language::C => once!(CLANG, c_query_set()),
        Language::Cpp => once!(CPP, cpp_query_set()),
        Language::CSharp => once!(CSHARP, csharp_query_set()),
        Language::Php => once!(PHP, php_query_set()),
        Language::Bash => once!(BASH, bash_query_set()),
        Language::Lua => once!(LUA, lua_query_set()),
        Language::Kotlin => once!(KOTLIN, kotlin_query_set()),
        Language::Scala => once!(SCALA, scala_query_set()),
        Language::Swift => once!(SWIFT, swift_query_set()),
        Language::Zig => once!(ZIG, zig_query_set()),
        Language::R => once!(RLANG, r_query_set()),
        // Registry languages cache their own compiled queries in the LangDef's
        // OnceLock (see `registry::LangDef::compiled_queries`).
        Language::Registered(d) => d.compiled_queries(),
        Language::Unsupported(_) => Ok(&[]),
    }
}

/// Build the CompiledQuery objects for Lua (definitions, calls, imports).
pub fn lua_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_lua::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            lua_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), lua_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), lua_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Kotlin (definitions, calls, imports).
pub fn kotlin_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_kotlin_ng::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            kotlin_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), kotlin_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), kotlin_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Scala (definitions, calls, imports).
pub fn scala_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_scala::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            scala_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), scala_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), scala_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Swift (definitions, calls, imports).
pub fn swift_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_swift::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            swift_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), swift_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), swift_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Zig (definitions, calls, imports).
pub fn zig_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_zig::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            zig_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), zig_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), zig_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for R (definitions, calls, imports).
pub fn r_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_r::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(QueryKind::Definitions, lang.clone(), r_queries::DEFINITIONS)?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), r_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), r_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Java (definitions, calls, imports).
pub fn java_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_java::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            java_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), java_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), java_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for C (definitions, calls, includes).
pub fn c_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_c::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(QueryKind::Definitions, lang.clone(), c_queries::DEFINITIONS)?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), c_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), c_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for C++ (definitions, calls, includes).
pub fn cpp_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_cpp::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            cpp_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), cpp_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), cpp_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for C# (definitions, calls, imports).
pub fn csharp_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_c_sharp::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            csharp_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), csharp_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), csharp_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for PHP (definitions, calls, imports).
pub fn php_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_php::LANGUAGE_PHP.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            php_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), php_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), php_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Bash (definitions, calls, imports).
pub fn bash_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_bash::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            bash_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), bash_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), bash_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Go (definitions, calls, imports).
pub fn go_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_go::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            go_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), go_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), go_queries::IMPORTS)?,
        CompiledQuery::new(QueryKind::Usages, lang.clone(), go_queries::USAGES)?,
    ])
}

/// Build the CompiledQuery objects for Ruby (definitions, calls, imports).
pub fn ruby_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_ruby::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            ruby_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), ruby_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), ruby_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for JavaScript (definitions, calls, imports).
pub fn javascript_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_javascript::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            js_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), js_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), js_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for TypeScript (definitions, calls,
/// imports). `tsx` selects the TSX grammar; the query sources are identical.
pub fn typescript_query_set(tsx: bool) -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = if tsx {
        tree_sitter_typescript::LANGUAGE_TSX.into()
    } else {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    };
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            ts_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), ts_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), ts_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Python (definitions, calls, imports).
pub fn python_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            python_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), python_queries::CALLS)?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), python_queries::IMPORTS)?,
    ])
}

/// Build the CompiledQuery objects for Rust (one per extraction pass).
pub fn rust_query_set() -> Result<Vec<CompiledQuery>, tree_sitter::QueryError> {
    let lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    Ok(vec![
        CompiledQuery::new(
            QueryKind::Definitions,
            lang.clone(),
            rust_queries::DEFINITIONS,
        )?,
        CompiledQuery::new(QueryKind::Imports, lang.clone(), rust_queries::IMPORTS)?,
        CompiledQuery::new(QueryKind::Calls, lang.clone(), rust_queries::CALLS)?,
        CompiledQuery::new(QueryKind::TypeRefs, lang.clone(), rust_queries::TYPE_REFS)?,
        CompiledQuery::new(QueryKind::Usages, lang.clone(), rust_queries::USAGES)?,
        CompiledQuery::new(
            QueryKind::TypeAssigns,
            lang.clone(),
            rust_queries::TYPE_ASSIGNS,
        )?,
        CompiledQuery::new(
            QueryKind::Inheritance,
            lang.clone(),
            rust_queries::INHERITANCE,
        )?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_query_set_compiles() {
        let qs = rust_query_set().expect("rust queries must compile");
        assert_eq!(qs.len(), 7);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Imports);
        assert_eq!(qs[2].kind, QueryKind::Calls);
        assert_eq!(qs[3].kind, QueryKind::TypeRefs);
        assert_eq!(qs[4].kind, QueryKind::Usages);
        assert_eq!(qs[5].kind, QueryKind::TypeAssigns);
        assert_eq!(qs[6].kind, QueryKind::Inheritance);
    }

    #[test]
    fn python_query_set_compiles() {
        let qs = python_query_set().expect("python queries must compile");
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Calls);
        assert_eq!(qs[2].kind, QueryKind::Imports);
    }

    #[test]
    fn javascript_query_set_compiles() {
        let qs = javascript_query_set().expect("javascript queries must compile");
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Calls);
        assert_eq!(qs[2].kind, QueryKind::Imports);
    }

    #[test]
    fn go_query_set_compiles() {
        let qs = go_query_set().expect("go queries must compile");
        assert_eq!(qs.len(), 4);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Calls);
        assert_eq!(qs[2].kind, QueryKind::Imports);
        assert_eq!(qs[3].kind, QueryKind::Usages);
    }

    #[test]
    fn ruby_query_set_compiles() {
        let qs = ruby_query_set().expect("ruby queries must compile");
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Calls);
        assert_eq!(qs[2].kind, QueryKind::Imports);
    }

    #[test]
    fn java_query_set_compiles() {
        let qs = java_query_set().expect("java queries must compile");
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Calls);
        assert_eq!(qs[2].kind, QueryKind::Imports);
    }

    #[test]
    fn c_query_set_compiles() {
        let qs = c_query_set().expect("c queries must compile");
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Calls);
        assert_eq!(qs[2].kind, QueryKind::Imports);
    }

    #[test]
    fn cpp_query_set_compiles() {
        let qs = cpp_query_set().expect("cpp queries must compile");
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0].kind, QueryKind::Definitions);
        assert_eq!(qs[1].kind, QueryKind::Calls);
        assert_eq!(qs[2].kind, QueryKind::Imports);
    }

    #[test]
    fn csharp_php_bash_query_sets_compile() {
        for (name, qs) in [
            ("csharp", csharp_query_set()),
            ("php", php_query_set()),
            ("bash", bash_query_set()),
        ] {
            let qs = qs.unwrap_or_else(|e| panic!("{name} queries must compile: {e}"));
            assert_eq!(qs.len(), 3, "{name}");
            assert_eq!(qs[0].kind, QueryKind::Definitions, "{name}");
            assert_eq!(qs[1].kind, QueryKind::Calls, "{name}");
            assert_eq!(qs[2].kind, QueryKind::Imports, "{name}");
        }
    }

    #[test]
    fn batch_onboarded_query_sets_compile() {
        for (name, qs) in [
            ("lua", lua_query_set()),
            ("kotlin", kotlin_query_set()),
            ("scala", scala_query_set()),
            ("swift", swift_query_set()),
            ("zig", zig_query_set()),
            ("r", r_query_set()),
        ] {
            let qs = qs.unwrap_or_else(|e| panic!("{name} queries must compile: {e}"));
            assert_eq!(qs.len(), 3, "{name}");
            assert_eq!(qs[0].kind, QueryKind::Definitions, "{name}");
            assert_eq!(qs[1].kind, QueryKind::Calls, "{name}");
            assert_eq!(qs[2].kind, QueryKind::Imports, "{name}");
        }
    }

    #[test]
    fn typescript_query_sets_compile_for_ts_and_tsx() {
        for tsx in [false, true] {
            let qs = typescript_query_set(tsx).expect("typescript queries must compile");
            assert_eq!(qs.len(), 3, "tsx={tsx}");
            assert_eq!(qs[0].kind, QueryKind::Definitions);
            assert_eq!(qs[1].kind, QueryKind::Calls);
            assert_eq!(qs[2].kind, QueryKind::Imports);
        }
    }
}
