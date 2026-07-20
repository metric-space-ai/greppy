//! `greppy-resolver` — import, call, and (eventually) type resolution.
//!
//! v1 (Rust-only) ships a real, name-based **cross-file call
//! resolver**. Without the Hybrid LSP layer (DD-1) we cannot do full
//! type resolution, so the call resolver is deliberately conservative:
//! it resolves a callee *name* to a unique `Function`/`Method`
//! definition node anywhere in the project. If no callable target can be
//! resolved, it falls back to a unique constructable class/type definition
//! so `Widget(...)`-style construction links to the `Class`/type node. It
//! still **refuses to guess** when the target set is ambiguous or absent.
//!
//! The indexer drives this: after PASS-1 inserts every node for every
//! file, it walks the unresolved CALLS edges and asks
//! [`resolve_call`] for each callee name. On a unique hit it inserts
//! the cross-file CALLS edge; on zero/ambiguous it skips.
//!
//! Wave 4 adds two more name-based resolvers that share the exact same
//! *uniqueness* discipline (resolve only when the name maps to a single
//! definition project-wide; never guess on ambiguity or absence):
//!
//! - [`resolve_type_ref`] — resolves a referenced **type name** (a
//!   function parameter/return/field/`let` type) to a unique
//!   `Struct`/`Enum`/`Trait`/`TypeAlias` definition anywhere in the
//!   project. Drives cross-file `TYPE_REF` edges.
//! - [`resolve_use`] — resolves a bare **usage name** to a unique
//!   definition of *any* resolvable kind (`Function`/`Method`/`Struct`/
//!   `Enum`/`Trait`/`TypeAlias`). Drives cross-file `USES` edges.
//!
//! Cross-file `IMPORTS` resolution is handled in the indexer (it owns
//! the per-file module node creation); the resolver only contributes
//! the shared name-lookup primitive [`unique_def_named`].

#![deny(rust_2018_idioms)]

use greppy_store::{Node, Store};

/// Labels that count as a callable definition target. A `CALLS` edge
/// may only resolve to one of these.
const CALLABLE_LABELS: [&str; 2] = ["Function", "Method"];

/// Labels that count as constructable type targets for a `CALLS` fallback.
/// This lets class/type construction sites depend on the class when there is
/// no callable constructor node to link to.
const CONSTRUCTABLE_LABELS: [&str; 4] = ["Class", "Struct", "Type", "Enum"];

/// Labels that count as a *type* definition target. A `TYPE_REF` edge
/// may only resolve to one of these. The Rust extractor labels type defs
/// with the C-reference scheme (`class_label_for_kind`): struct/union →
/// `Class`, trait → `Interface`, enum → `Enum`, type alias → `Type`. The
/// legacy `Struct`/`Trait`/`TypeAlias` labels are kept so existing
/// resolver fixtures (and any future extractor that emits them) still
/// resolve.
const TYPE_LABELS: [&str; 7] = [
    "Class",
    "Interface",
    "Type",
    "Enum",
    "Struct",
    "Trait",
    "TypeAlias",
];

/// Labels that count as a resolvable definition for a bare usage. A
/// `USES` edge may resolve to any of these — every kind of project
/// symbol a reference could plausibly name. `Import`/`Call`/`Module`
/// synthetic-ish labels are deliberately excluded so a usage resolves
/// to a real definition, not another reference node.
const DEF_LABELS: [&str; 11] = [
    "Function",
    "Method",
    "Class",
    "Interface",
    "Type",
    "Enum",
    "Struct",
    "Trait",
    "TypeAlias",
    "Variable",
    "Field",
];

/// Outcome of resolving one callee name against the project graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallResolution {
    /// Exactly one callable definition matched the name. Carries the
    /// resolved node id.
    Unique(i64),
    /// No callable definition with that name exists in the project.
    Unresolved,
    /// More than one callable definition matched; we refuse to guess.
    /// Carries the candidate node ids for diagnostics.
    Ambiguous(Vec<i64>),
}

impl CallResolution {
    /// The resolved id if and only if the resolution is unique.
    pub fn unique_id(&self) -> Option<i64> {
        match self {
            CallResolution::Unique(id) => Some(*id),
            _ => None,
        }
    }
}

/// Resolve a call from `caller` to a callee named `callee_name`
/// somewhere in `project`.
///
/// Strategy (most specific first):
/// 1. **Same-file preference.** If exactly one callable definition with
///    that name lives in the caller's own file, take it. This keeps
///    same-file calls correct even when an identically-named function
///    exists elsewhere in the project.
/// 2. **Project-wide uniqueness.** Otherwise, gather every callable
///    definition with that name across the whole project. If exactly
///    one exists, resolve to it (the cross-file case). If none,
///    `Unresolved`.
/// 3. **Import disambiguation.** If several remain, look at what the
///    caller's *file* imports (its `IMPORTS` edges). If exactly one
///    candidate is among the file's imported definitions, prefer it.
///    Only if the file imports zero — or more than one — of the
///    candidates do we stay `Ambiguous`. We still never guess.
///
/// `caller` is the resolved definition node of the enclosing function;
/// it is used only for the same-file preference and to avoid resolving
/// a call to the caller itself when that would be the sole same-file
/// candidate is *not* excluded (direct recursion is a legitimate
/// self-call, so we keep it).
pub fn resolve_call(
    store: &Store,
    project: &str,
    caller: &Node,
    callee_name: &str,
) -> Result<CallResolution, greppy_core::Error> {
    let candidates = callable_defs_named(store, project, callee_name)?;
    let callable = resolve_unique_with_imports(store, project, &candidates, caller)?;
    if matches!(callable, CallResolution::Unique(_)) {
        return Ok(callable);
    }
    let constructable = defs_named(store, project, &CONSTRUCTABLE_LABELS, callee_name)?;
    match resolve_unique_with_imports(store, project, &constructable, caller)? {
        unique @ CallResolution::Unique(_) => Ok(unique),
        CallResolution::Unresolved if matches!(callable, CallResolution::Unresolved) => {
            Ok(CallResolution::Unresolved)
        }
        CallResolution::Ambiguous(ids) if matches!(callable, CallResolution::Unresolved) => {
            Ok(CallResolution::Ambiguous(ids))
        }
        _ => Ok(callable),
    }
}

/// Every `Function`/`Method` node in `project` whose `name` equals
/// `callee_name`. The store has no by-name index, so we list by the
/// callable labels and filter — fine for v1 project sizes; a dedicated
/// store query can replace this later without touching callers.
fn callable_defs_named(
    store: &Store,
    project: &str,
    callee_name: &str,
) -> Result<Vec<Node>, greppy_core::Error> {
    defs_named(store, project, &CALLABLE_LABELS, callee_name)
}

/// Resolve a referenced **type name** to a unique type definition
/// (`Struct`/`Enum`/`Trait`/`TypeAlias`) anywhere in `project`.
///
/// Same conservative discipline as [`resolve_call`]: if the type lives
/// in the referrer's own file (same-file preference) take it; otherwise
/// resolve only when the name is project-wide unique. Ambiguous (>1) or
/// absent (0) → never guess. Drives cross-file `TYPE_REF` edges.
pub fn resolve_type_ref(
    store: &Store,
    project: &str,
    referrer: &Node,
    type_name: &str,
) -> Result<CallResolution, greppy_core::Error> {
    let candidates = defs_named(store, project, &TYPE_LABELS, type_name)?;
    resolve_unique_with_imports(store, project, &candidates, referrer)
}

/// Resolve a bare **usage name** to a unique definition of any
/// resolvable kind (`Function`/`Method`/`Struct`/`Enum`/`Trait`/
/// `TypeAlias`) anywhere in `project`.
///
/// Same conservative discipline as [`resolve_call`]: same-file
/// preference first, then project-wide uniqueness; never guess on
/// ambiguity or absence. Drives cross-file `USES` edges.
pub fn resolve_use(
    store: &Store,
    project: &str,
    referrer: &Node,
    ref_name: &str,
) -> Result<CallResolution, greppy_core::Error> {
    let candidates = defs_named(store, project, &DEF_LABELS, ref_name)?;
    resolve_unique_with_imports(store, project, &candidates, referrer)
}

/// Resolve `name` to a single definition node among `labels`,
/// project-wide, returning the node id only when it is unique (the
/// same-file preference does NOT apply — this is the primitive the
/// indexer uses for `IMPORTS`, whose source is a file/module node with
/// no meaningful "same file" notion for the imported symbol). Returns
/// `None` on zero or ambiguous matches.
pub fn unique_def_named(
    store: &Store,
    project: &str,
    labels: &[&str],
    name: &str,
) -> Result<Option<i64>, greppy_core::Error> {
    let candidates = defs_named(store, project, labels, name)?;
    Ok(if candidates.len() == 1 {
        Some(candidates[0].id)
    } else {
        None
    })
}

/// Resolve an imported symbol to a single definition node, using the
/// import **path** to break ties between same-named definitions.
///
/// This is [`unique_def_named`] with a path-aware fallback: if the bare
/// name is project-wide unique we return it (same as before). If it is
/// ambiguous, we consult `path` — a Rust use-path such as `b::dup` or
/// `crate::b::dup` — and keep only the candidates whose defining file
/// matches the path's **module segment** (the segment immediately before
/// the imported name). `b::dup` selects the `dup` defined in a file named
/// `b` (`src/b.rs`, `src/b/mod.rs`); `crate::b::dup` does the same. If
/// exactly one candidate matches, we resolve to it. Zero or several
/// matches → `None` (still never guess).
///
/// This is what lets a `use b::dup;` link to the *right* `dup` when two
/// files define one. It is the precise, author-supplied disambiguation
/// the reference-edge resolver later reads back via the file's IMPORTS
/// edges. `path` may be empty (brace-group / glob), in which case only
/// the project-wide-unique case can resolve.
pub fn unique_def_named_with_path(
    store: &Store,
    project: &str,
    labels: &[&str],
    name: &str,
    path: &str,
) -> Result<Option<i64>, greppy_core::Error> {
    let candidates = defs_named(store, project, labels, name)?;
    match candidates.len() {
        0 => return Ok(None),
        1 => return Ok(Some(candidates[0].id)),
        _ => {}
    }
    // Ambiguous by name: disambiguate by the path's module segment.
    let Some(module_seg) = path_module_segment(path, name) else {
        return Ok(None);
    };
    let matched: Vec<&Node> = candidates
        .iter()
        .filter(|n| file_stem_matches(&n.file_path, module_seg))
        .collect();
    Ok(if matched.len() == 1 {
        Some(matched[0].id)
    } else {
        None
    })
}

/// The module segment of a Rust use-`path` for a given final `name`: the
/// path segment immediately before `name`. `b::dup` → `Some("b")`,
/// `crate::b::dup` → `Some("b")`, `dup` → `None` (no module qualifier),
/// `self::dup` / `crate::dup` → `None` (the qualifier is not a module
/// file we can map). Whitespace around `::` is tolerated.
fn path_module_segment<'a>(path: &'a str, name: &str) -> Option<&'a str> {
    let segs: Vec<&str> = path
        .split("::")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    // The path must end with the imported name and have a real module
    // segment before it.
    let last = segs.last()?;
    if *last != name || segs.len() < 2 {
        return None;
    }
    let module = segs[segs.len() - 2];
    // `crate` / `self` / `super` are not file-named modules.
    if matches!(module, "crate" | "self" | "super") {
        return None;
    }
    Some(module)
}

/// Whether a node's `file_path` belongs to a module named `module`:
/// `src/b.rs` or `src/b/mod.rs` both match module `b`.
fn file_stem_matches(file_path: &str, module: &str) -> bool {
    let p = std::path::Path::new(file_path);
    // `src/b/mod.rs` → parent dir name is the module.
    if p.file_name().and_then(|s| s.to_str()) == Some("mod.rs") {
        if let Some(parent) = p
            .parent()
            .and_then(|d| d.file_name())
            .and_then(|s| s.to_str())
        {
            return parent == module;
        }
    }
    p.file_stem().and_then(|s| s.to_str()) == Some(module)
}

/// The label set [`unique_def_named`] should use to resolve an imported
/// symbol name. Exposed so the indexer and resolver agree on what an
/// `IMPORTS` edge may point at.
///
/// This is the importable subset of [`DEF_LABELS`]: an `import`/`use`/`from …
/// import` names a top-level type, free function, or enum — never a method,
/// field, or variable. Keeping `Method` here made every import ambiguous in languages
/// whose constructor shares the type's name (Java `class Checksum { Checksum()
/// … }` yields both a `Class` and a `Method` named `Checksum` in the SAME
/// file, so the name-tie could not be broken by module segment and every such
/// import was dropped — java_medium IMPORTS resolved 0 of 152). Value
/// references (`USES`/`USAGE`) DO include `Method` (you can reference a
/// method, field, or variable), which is why they use [`DEF_LABELS`] and
/// imports use this subset.
pub const IMPORTABLE_LABELS: [&str; 8] = [
    "Function",
    "Class",
    "Interface",
    "Type",
    "Enum",
    "Struct",
    "Trait",
    "TypeAlias",
];

/// Shared uniqueness resolution: same-file preference, then project-wide
/// uniqueness, mirroring [`resolve_call`]'s two-step strategy. Factored
/// out so TYPE_REF and USES behave identically to CALLS.
fn resolve_unique(
    candidates: &[Node],
    referrer: &Node,
) -> Result<CallResolution, greppy_core::Error> {
    if candidates.is_empty() {
        return Ok(CallResolution::Unresolved);
    }
    let same_file: Vec<&Node> = candidates
        .iter()
        .filter(|n| n.file_path == referrer.file_path)
        .collect();
    if same_file.len() == 1 {
        return Ok(CallResolution::Unique(same_file[0].id));
    }
    if candidates.len() == 1 {
        return Ok(CallResolution::Unique(candidates[0].id));
    }
    Ok(CallResolution::Ambiguous(
        candidates.iter().map(|n| n.id).collect(),
    ))
}

/// [`resolve_unique`] with one extra disambiguation step layered on top:
/// when the project-wide result is `Ambiguous`, consult the referrer
/// file's imports and prefer the single imported candidate, if there is
/// exactly one.
///
/// Rationale: without type information we cannot in general pick between
/// two same-named definitions — but a Rust file's `use` declarations are
/// a precise, author-supplied statement of *which* of them this file
/// refers to. The indexer turns each `use` into an `IMPORTS` edge from
/// the file's per-file `Module` node (`<file>::__file__`) to the unique
/// imported definition. So if the referrer's file imports exactly one of
/// the ambiguous candidates, that candidate is the right answer and we
/// resolve to it.
///
/// Discipline is preserved: this only ever *narrows* an `Ambiguous`
/// result to a `Unique` one when the import signal is itself
/// unambiguous (exactly one candidate imported). Zero imported, or more
/// than one imported, stays `Ambiguous` — we never guess. `Unique` and
/// `Unresolved` results from [`resolve_unique`] are returned untouched.
fn resolve_unique_with_imports(
    store: &Store,
    project: &str,
    candidates: &[Node],
    referrer: &Node,
) -> Result<CallResolution, greppy_core::Error> {
    let base = resolve_unique(candidates, referrer)?;
    let CallResolution::Ambiguous(_) = base else {
        return Ok(base);
    };

    // The referrer's file imports a set of definition node ids (its
    // `IMPORTS` edge targets). Keep only the candidates in that set.
    let imported = imported_target_ids(store, project, &referrer.file_path)?;
    if imported.is_empty() {
        return Ok(base);
    }
    let preferred: Vec<&Node> = candidates
        .iter()
        .filter(|n| imported.contains(&n.id))
        .collect();
    if preferred.len() == 1 {
        return Ok(CallResolution::Unique(preferred[0].id));
    }
    // Zero or several candidates imported → still ambiguous, never guess.
    Ok(base)
}

/// The set of definition node ids that `file`'s per-file `Module` node
/// imports — i.e. the resolved `target_id`s of the `IMPORTS` edges whose
/// source is `<file>::__file__`.
///
/// Returns an empty set when the file has no Module node yet (e.g. the
/// referrer is not a real file node) or imports nothing resolvable. This
/// is the precise signal used to disambiguate same-named definitions:
/// the file told us which symbol it pulled in.
fn imported_target_ids(
    store: &Store,
    project: &str,
    file: &str,
) -> Result<std::collections::HashSet<i64>, greppy_core::Error> {
    let module_qname = format!("{file}::__file__");
    let Some(module) = store.get_node_by_qname(project, &module_qname)? else {
        return Ok(std::collections::HashSet::new());
    };
    // A file is unlikely to carry more than this many imports; the cap
    // only guards against a pathological generated file.
    const IMPORT_LIMIT: usize = 100_000;
    let edges = store.outgoing_edges(module.id, Some("IMPORTS"), IMPORT_LIMIT)?;
    Ok(edges.into_iter().map(|e| e.target_id).collect())
}

/// Every node in `project` whose `name` equals `name` and whose `label`
/// is in `labels`.
///
/// **Index-backed (Wave 4).** This goes straight at the
/// `idx_nodes_name` index on `nodes(project, name)` via
/// [`Store::list_nodes_by_name`] and then keeps only the rows whose
/// label is one of `labels`. The previous implementation listed *every*
/// node of each candidate label (a 100k-row scan per label) and filtered
/// by name — O(nodes) work on every single edge to be resolved. Going
/// by name first means we materialise only the (tiny) set of nodes that
/// actually share the looked-up name, so the per-edge cost is
/// proportional to the number of same-named symbols rather than the size
/// of the whole graph.
///
/// The returned set is **identical** to the old by-label path: the same
/// `(name, label)` predicate selects the same rows. `list_nodes_by_name`
/// orders by `qualified_name`, giving a deterministic, label-interleaved
/// order; resolution outcomes ([`resolve_unique`] / [`unique_def_named`])
/// depend only on the *set* and the same-file count, so order does not
/// change any resolution, and the `Ambiguous` id list is deterministic.
fn defs_named(
    store: &Store,
    project: &str,
    labels: &[&str],
    name: &str,
) -> Result<Vec<Node>, greppy_core::Error> {
    // A generous cap: a single name is extremely unlikely to be borne by
    // more definitions than this across one project. The by-name index
    // makes this lookup cheap regardless.
    const NAME_LIMIT: usize = 100_000;
    let nodes = store.list_nodes_by_name(project, name, NAME_LIMIT)?;
    Ok(nodes
        .into_iter()
        .filter(|n| labels.contains(&n.label.as_str()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_store::{NewEdge, NewNode, Project, Store};

    fn store_with_project(name: &str) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: name.into(),
            indexed_at: "2026-06-29T00:00:00Z".into(),
            root_path: format!("/repos/{name}"),
        })
        .unwrap();
        s
    }

    fn insert_fn(s: &mut Store, project: &str, label: &str, file: &str, name: &str) -> Node {
        let qname = format!("{file}::Function::{name}");
        let id = s
            .insert_node(&NewNode {
                project: project.into(),
                label: label.into(),
                name: name.into(),
                qualified_name: qname.clone(),
                file_path: file.into(),
                start_line: 1,
                end_line: 2,
                properties: Default::default(),
            })
            .unwrap();
        s.get_node(id).unwrap().unwrap()
    }

    #[test]
    fn resolves_unique_cross_file_callee() {
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let target = insert_fn(&mut s, "p", "Function", "src/helper.rs", "do_it");
        let r = resolve_call(&s, "p", &caller, "do_it").unwrap();
        assert_eq!(r, CallResolution::Unique(target.id));
    }

    #[test]
    fn unresolved_when_no_definition_exists() {
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let r = resolve_call(&s, "p", &caller, "nope").unwrap();
        assert_eq!(r, CallResolution::Unresolved);
    }

    #[test]
    fn ambiguous_when_two_definitions_in_other_files() {
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let a = insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        let b = insert_fn(&mut s, "p", "Function", "src/b.rs", "dup");
        let r = resolve_call(&s, "p", &caller, "dup").unwrap();
        match r {
            CallResolution::Ambiguous(ids) => {
                let set: std::collections::HashSet<i64> = ids.into_iter().collect();
                assert!(set.contains(&a.id) && set.contains(&b.id));
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn same_file_preference_breaks_a_tie() {
        // Two definitions named `dup`: one same-file, one cross-file.
        // Same-file wins instead of going Ambiguous.
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let local = insert_fn(&mut s, "p", "Function", "src/lib.rs", "dup");
        let _remote = insert_fn(&mut s, "p", "Function", "src/other.rs", "dup");
        let r = resolve_call(&s, "p", &caller, "dup").unwrap();
        assert_eq!(r, CallResolution::Unique(local.id));
    }

    #[test]
    fn resolves_method_label_callee() {
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let m = insert_fn(&mut s, "p", "Method", "src/widget.rs", "render");
        let r = resolve_call(&s, "p", &caller, "render").unwrap();
        assert_eq!(r, CallResolution::Unique(m.id));
    }

    /// Insert a node of any label/name in `file`, returning the node.
    fn insert_node_named(
        s: &mut Store,
        project: &str,
        label: &str,
        file: &str,
        name: &str,
    ) -> Node {
        let qname = format!("{file}::{label}::{name}");
        let id = s
            .insert_node(&NewNode {
                project: project.into(),
                label: label.into(),
                name: name.into(),
                qualified_name: qname,
                file_path: file.into(),
                start_line: 1,
                end_line: 2,
                properties: Default::default(),
            })
            .unwrap();
        s.get_node(id).unwrap().unwrap()
    }

    #[test]
    fn resolve_call_falls_back_to_unique_constructable_class() {
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "build");
        let class = insert_node_named(&mut s, "p", "Class", "src/runner.py", "RunnerFilter");
        let r = resolve_call(&s, "p", &caller, "RunnerFilter").unwrap();
        assert_eq!(r, CallResolution::Unique(class.id));
    }

    #[test]
    fn resolve_type_ref_unique_cross_file() {
        let mut s = store_with_project("p");
        // A function in lib.rs referencing a Struct defined in types.rs.
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "use_widget");
        let widget = insert_node_named(&mut s, "p", "Struct", "src/types.rs", "Widget");
        let r = resolve_type_ref(&s, "p", &referrer, "Widget").unwrap();
        assert_eq!(r, CallResolution::Unique(widget.id));
    }

    #[test]
    fn resolve_type_ref_resolves_enum_and_trait_and_alias() {
        for label in ["Enum", "Trait", "TypeAlias"] {
            let mut s = store_with_project("p");
            let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "f");
            let ty = insert_node_named(&mut s, "p", label, "src/types.rs", "T");
            let r = resolve_type_ref(&s, "p", &referrer, "T").unwrap();
            assert_eq!(
                r,
                CallResolution::Unique(ty.id),
                "label {label} must resolve"
            );
        }
    }

    #[test]
    fn resolve_type_ref_ignores_non_type_labels() {
        // A Function named `Widget` must NOT satisfy a TYPE_REF.
        let mut s = store_with_project("p");
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "f");
        let _fn_widget = insert_fn(&mut s, "p", "Function", "src/other.rs", "Widget");
        let r = resolve_type_ref(&s, "p", &referrer, "Widget").unwrap();
        assert_eq!(r, CallResolution::Unresolved);
    }

    #[test]
    fn resolve_type_ref_ambiguous_is_not_guessed() {
        let mut s = store_with_project("p");
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "f");
        insert_node_named(&mut s, "p", "Struct", "src/a.rs", "Dup");
        insert_node_named(&mut s, "p", "Enum", "src/b.rs", "Dup");
        let r = resolve_type_ref(&s, "p", &referrer, "Dup").unwrap();
        assert!(matches!(r, CallResolution::Ambiguous(_)));
    }

    #[test]
    fn resolve_use_unique_cross_file_function() {
        let mut s = store_with_project("p");
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let helper = insert_fn(&mut s, "p", "Function", "src/helper.rs", "helper");
        let r = resolve_use(&s, "p", &referrer, "helper").unwrap();
        assert_eq!(r, CallResolution::Unique(helper.id));
    }

    #[test]
    fn resolve_use_resolves_a_type_definition() {
        // A bare identifier referencing a Struct resolves to it.
        let mut s = store_with_project("p");
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "f");
        let cfg = insert_node_named(&mut s, "p", "Struct", "src/config.rs", "Config");
        let r = resolve_use(&s, "p", &referrer, "Config").unwrap();
        assert_eq!(r, CallResolution::Unique(cfg.id));
    }

    #[test]
    fn resolve_use_resolves_named_values_and_fields() {
        for label in ["Variable", "Field"] {
            let mut s = store_with_project("p");
            let referrer = insert_fn(&mut s, "p", "Function", "src/lib.cs", "caller");
            let value = insert_node_named(&mut s, "p", label, "src/values.cs", "Seed");
            let r = resolve_use(&s, "p", &referrer, "Seed").unwrap();
            assert_eq!(
                r,
                CallResolution::Unique(value.id),
                "USES must resolve a {label} target"
            );
        }
    }

    #[test]
    fn resolve_use_ambiguous_is_not_guessed() {
        let mut s = store_with_project("p");
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "f");
        insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        insert_node_named(&mut s, "p", "Struct", "src/b.rs", "dup");
        let r = resolve_use(&s, "p", &referrer, "dup").unwrap();
        assert!(matches!(r, CallResolution::Ambiguous(_)));
    }

    #[test]
    fn unique_def_named_resolves_only_when_unique() {
        let mut s = store_with_project("p");
        let target = insert_node_named(&mut s, "p", "Struct", "src/types.rs", "Widget");
        let got = unique_def_named(&s, "p", &IMPORTABLE_LABELS, "Widget").unwrap();
        assert_eq!(got, Some(target.id));

        // Ambiguous → None (no guess).
        insert_node_named(&mut s, "p", "Enum", "src/other.rs", "Widget");
        let got = unique_def_named(&s, "p", &IMPORTABLE_LABELS, "Widget").unwrap();
        assert_eq!(got, None);

        // Absent → None.
        let got = unique_def_named(&s, "p", &IMPORTABLE_LABELS, "Nope").unwrap();
        assert_eq!(got, None);
    }

    /// Insert a per-file `Module` node (qname `<file>::__file__`) and one
    /// `IMPORTS` edge from it to `target`. Mirrors what the indexer
    /// persists so the resolver's disambiguation has a real signal.
    fn import_in_file(s: &mut Store, project: &str, file: &str, target: &Node) {
        // The canonical per-file Module node uses the qname
        // `<file>::__file__`, which is what the resolver looks up.
        let id = s
            .insert_node(&NewNode {
                project: project.into(),
                label: "Module".into(),
                name: file.into(),
                qualified_name: format!("{file}::__file__"),
                file_path: file.into(),
                start_line: 1,
                end_line: 1,
                properties: Default::default(),
            })
            .unwrap();
        s.insert_edge(&NewEdge {
            project: project.into(),
            source_id: id,
            target_id: target.id,
            edge_type: "IMPORTS".into(),
            properties: Default::default(),
        })
        .unwrap();
    }

    #[test]
    fn defs_named_uses_by_name_index_path() {
        // Micro-assertion that resolution goes through the by-name index
        // rather than a by-label full scan: insert MANY nodes of the
        // candidate label under OTHER names, plus exactly one match. The
        // result is the single match — and (the point) the candidate set
        // `defs_named` builds has length 1, i.e. it did not materialise
        // the whole label population. We assert that directly via the
        // private `defs_named`.
        let mut s = store_with_project("p");
        for i in 0..500 {
            insert_fn(
                &mut s,
                "p",
                "Function",
                "src/noise.rs",
                &format!("noise_{i}"),
            );
        }
        let target = insert_fn(&mut s, "p", "Function", "src/helper.rs", "needle");
        let cands = defs_named(&s, "p", &CALLABLE_LABELS, "needle").unwrap();
        assert_eq!(
            cands.len(),
            1,
            "by-name path must return only same-named candidates, not the whole label set"
        );
        assert_eq!(cands[0].id, target.id);
        // And the public resolver still resolves it.
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        assert_eq!(
            resolve_call(&s, "p", &caller, "needle").unwrap(),
            CallResolution::Unique(target.id)
        );
    }

    #[test]
    fn defs_named_filters_by_label_not_just_name() {
        // A node sharing the name but with a non-candidate label must be
        // excluded — confirms the label filter is applied after the
        // by-name lookup.
        let mut s = store_with_project("p");
        let _wrong_label = insert_node_named(&mut s, "p", "Struct", "src/a.rs", "thing");
        let right = insert_fn(&mut s, "p", "Function", "src/b.rs", "thing");
        let cands = defs_named(&s, "p", &CALLABLE_LABELS, "thing").unwrap();
        assert_eq!(
            cands.len(),
            1,
            "Struct named `thing` must be filtered out for CALLS"
        );
        assert_eq!(cands[0].id, right.id);
    }

    #[test]
    fn import_disambiguates_ambiguous_call() {
        // Two functions named `dup` in different files. The caller's file
        // imports exactly one of them → the call resolves to the imported
        // one instead of staying Ambiguous.
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let _a = insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        let b = insert_fn(&mut s, "p", "Function", "src/b.rs", "dup");
        // src/lib.rs imports the `dup` from src/b.rs.
        import_in_file(&mut s, "p", "src/lib.rs", &b);
        let r = resolve_call(&s, "p", &caller, "dup").unwrap();
        assert_eq!(
            r,
            CallResolution::Unique(b.id),
            "the imported candidate must win the tie"
        );
    }

    #[test]
    fn no_import_keeps_call_ambiguous() {
        // Same two-`dup` setup but the caller's file imports NEITHER → we
        // refuse to guess and stay Ambiguous.
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let a = insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        let b = insert_fn(&mut s, "p", "Function", "src/b.rs", "dup");
        let r = resolve_call(&s, "p", &caller, "dup").unwrap();
        match r {
            CallResolution::Ambiguous(ids) => {
                let set: std::collections::HashSet<i64> = ids.into_iter().collect();
                assert!(set.contains(&a.id) && set.contains(&b.id));
            }
            other => panic!("expected Ambiguous without a disambiguating import, got {other:?}"),
        }
    }

    #[test]
    fn import_of_both_candidates_stays_ambiguous() {
        // If the file somehow imports BOTH same-named candidates, the
        // import signal is itself ambiguous → never guess.
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let a = insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        let b = insert_fn(&mut s, "p", "Function", "src/b.rs", "dup");
        // Import both (parser/indexer would only ever do this in odd
        // cases, but the resolver must still refuse to guess).
        let module_id = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Module".into(),
                name: "src/lib.rs".into(),
                qualified_name: "src/lib.rs::__file__".into(),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: Default::default(),
            })
            .unwrap();
        for tgt in [&a, &b] {
            s.insert_edge(&NewEdge {
                project: "p".into(),
                source_id: module_id,
                target_id: tgt.id,
                edge_type: "IMPORTS".into(),
                properties: Default::default(),
            })
            .unwrap();
        }
        assert!(matches!(
            resolve_call(&s, "p", &caller, "dup").unwrap(),
            CallResolution::Ambiguous(_)
        ));
    }

    #[test]
    fn import_disambiguates_type_ref() {
        // TYPE_REF: two `Widget` structs; the referrer's file imports one.
        let mut s = store_with_project("p");
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "f");
        let _a = insert_node_named(&mut s, "p", "Struct", "src/a.rs", "Widget");
        let b = insert_node_named(&mut s, "p", "Struct", "src/b.rs", "Widget");
        import_in_file(&mut s, "p", "src/lib.rs", &b);
        assert_eq!(
            resolve_type_ref(&s, "p", &referrer, "Widget").unwrap(),
            CallResolution::Unique(b.id)
        );
    }

    #[test]
    fn import_disambiguates_use() {
        // USES: two same-named definitions of mixed kinds; the file
        // imports exactly one.
        let mut s = store_with_project("p");
        let referrer = insert_fn(&mut s, "p", "Function", "src/lib.rs", "f");
        let _a = insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        let b = insert_node_named(&mut s, "p", "Struct", "src/b.rs", "dup");
        import_in_file(&mut s, "p", "src/lib.rs", &b);
        assert_eq!(
            resolve_use(&s, "p", &referrer, "dup").unwrap(),
            CallResolution::Unique(b.id)
        );
    }

    #[test]
    fn import_in_another_file_does_not_disambiguate() {
        // The import must be in the REFERRER's file. An import in a
        // different file is irrelevant and the call stays ambiguous.
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let _a = insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        let b = insert_fn(&mut s, "p", "Function", "src/b.rs", "dup");
        // Import is recorded for src/other.rs, NOT src/lib.rs.
        import_in_file(&mut s, "p", "src/other.rs", &b);
        assert!(matches!(
            resolve_call(&s, "p", &caller, "dup").unwrap(),
            CallResolution::Ambiguous(_)
        ));
    }

    #[test]
    fn path_module_segment_extracts_module() {
        assert_eq!(path_module_segment("b::dup", "dup"), Some("b"));
        assert_eq!(path_module_segment("crate::b::dup", "dup"), Some("b"));
        assert_eq!(path_module_segment("a::b::dup", "dup"), Some("b"));
        // No module qualifier / non-file qualifiers → None.
        assert_eq!(path_module_segment("dup", "dup"), None);
        assert_eq!(path_module_segment("crate::dup", "dup"), None);
        assert_eq!(path_module_segment("self::dup", "dup"), None);
        assert_eq!(path_module_segment("super::dup", "dup"), None);
        // Final segment must equal the imported name.
        assert_eq!(path_module_segment("b::other", "dup"), None);
    }

    #[test]
    fn file_stem_matches_handles_rs_and_mod() {
        assert!(file_stem_matches("src/b.rs", "b"));
        assert!(file_stem_matches("src/b/mod.rs", "b"));
        assert!(!file_stem_matches("src/a.rs", "b"));
        assert!(!file_stem_matches("src/other/mod.rs", "b"));
    }

    #[test]
    fn unique_def_named_with_path_disambiguates_by_module() {
        // Two `dup` functions; `b::dup` selects the one in src/b.rs.
        let mut s = store_with_project("p");
        let _a = insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        let b = insert_fn(&mut s, "p", "Function", "src/b.rs", "dup");
        let got = unique_def_named_with_path(&s, "p", &IMPORTABLE_LABELS, "dup", "b::dup").unwrap();
        assert_eq!(got, Some(b.id));
        // crate-qualified path works too.
        let got = unique_def_named_with_path(&s, "p", &IMPORTABLE_LABELS, "dup", "crate::b::dup")
            .unwrap();
        assert_eq!(got, Some(b.id));
    }

    #[test]
    fn unique_def_named_with_path_still_unique_without_path() {
        // Unique name resolves regardless of path (mirrors unique_def_named).
        let mut s = store_with_project("p");
        let only = insert_fn(&mut s, "p", "Function", "src/x.rs", "solo");
        let got = unique_def_named_with_path(&s, "p", &IMPORTABLE_LABELS, "solo", "").unwrap();
        assert_eq!(got, Some(only.id));
    }

    #[test]
    fn unique_def_named_with_path_no_path_match_stays_unresolved() {
        // Ambiguous name and the path's module matches NEITHER file → None.
        let mut s = store_with_project("p");
        insert_fn(&mut s, "p", "Function", "src/a.rs", "dup");
        insert_fn(&mut s, "p", "Function", "src/b.rs", "dup");
        let got =
            unique_def_named_with_path(&s, "p", &IMPORTABLE_LABELS, "dup", "zzz::dup").unwrap();
        assert_eq!(got, None);
        // No module qualifier at all → cannot disambiguate → None.
        let got = unique_def_named_with_path(&s, "p", &IMPORTABLE_LABELS, "dup", "dup").unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn same_file_preference_still_wins_over_import() {
        // The same-file preference (a candidate in the referrer's own
        // file) is resolved by `resolve_unique` before disambiguation is
        // ever consulted, so it remains authoritative.
        let mut s = store_with_project("p");
        let caller = insert_fn(&mut s, "p", "Function", "src/lib.rs", "caller");
        let local = insert_fn(&mut s, "p", "Function", "src/lib.rs", "dup");
        let remote = insert_fn(&mut s, "p", "Function", "src/other.rs", "dup");
        // Even if the file imported the remote one, the same-file def wins.
        import_in_file(&mut s, "p", "src/lib.rs", &remote);
        assert_eq!(
            resolve_call(&s, "p", &caller, "dup").unwrap(),
            CallResolution::Unique(local.id)
        );
    }
}
