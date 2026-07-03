//! Structural node model — faithful port of the C reference's
//! `pass_structure` (`src/pipeline/pipeline.c`) plus the File→DEFINES edge
//! creation in `pass_definitions.c` (`process_def`, ~L288-320).
//!
//! The C pipeline materializes a *structural spine* over the discovered
//! file set, independent of what any language extractor produces:
//!
//! * one **Project** node (the repo root) — `label="Project"`,
//!   `name = qualified_name = project_name`, no file path.
//! * a **Folder** node for every unique directory that contains an indexed
//!   file (and every ancestor directory up to the root) — `label="Folder"`,
//!   `name = directory basename`, `qualified_name = project.dir.parts`
//!   (dot-joined; see [`folder_qn`]), `file_path = rel_dir`.
//! * a **File** node for every discovered file — `label="File"`,
//!   `name = basename`, `qualified_name = project.dir.__file__`
//!   (see [`file_qn`]), `file_path = rel_path`,
//!   `properties = {"extension": ".rs"}`.
//!
//! and the connecting edges:
//!
//! * **CONTAINS_FOLDER** — `parent -> child` for every folder in the chain
//!   (Project → top-level folder, folder → subfolder …). One per folder.
//! * **CONTAINS_FILE** — `parent -> file` (the file's directory, or the
//!   Project when the file is at the repo root). One per file.
//! * **DEFINES** — `File -> def` for every definition extracted from that
//!   file (`process_def` in `pass_definitions.c` inserts a File→DEFINES edge
//!   for *every* def it materializes).
//!
//! ## Relationship to grepplus's synthetic `Module` node
//!
//! grepplus already persists a per-file synthetic **Module** node (qname
//! `<rel_path>::__file__`) in `apply_file_nodes`, used as the resolvable
//! source endpoint for `IMPORTS` edges. The C reference *also* keeps a
//! per-file Module node (its `get_graph_schema` reports both `File` and
//! `Module` at 83 each on rust_medium), so we KEEP the Module node exactly
//! as it is and add the File/Folder/Project spine alongside it. To avoid a
//! `qualified_name` collision with the Module node (whose qname is
//! `<rel>::__file__`), the File node uses the C dotted scheme
//! `project.dir.__file__`; the two never collide.
//!
//! ## Faithfulness notes / documented deviations
//!
//! * **Node QNs and edge shapes are byte-faithful to C.** File/Folder QNs
//!   use the same `fqn_compute` / `fqn_folder` dotted scheme (path
//!   separators → dots, extension stripped, `__init__`/`index` handling).
//! * **DEFINES count** is bounded by grepplus's *extracted* definition set,
//!   which is currently smaller than C's (C's Rust extractor emits more
//!   definition kinds — Class/Variable/Field/Section — that grepplus does
//!   not yet extract). The File→DEFINES *mechanism* is a faithful port; the
//!   absolute count trails C by exactly the count of unextracted defs and
//!   will converge automatically as the extractor-parity gap closes. The
//!   Project/Folder/File/CONTAINS_FILE/CONTAINS_FOLDER counts reach full
//!   C parity.

use std::collections::{BTreeSet, HashMap};

use grepplus_core::Result;
use grepplus_discover::InventoryEntry;
use grepplus_store::{NewEdge, NewNode, Store};

/// Labels that are part of the structural spine and therefore are NOT
/// targets of a File→DEFINES edge. Every OTHER node living in a file is a
/// definition the File node DEFINES — INCLUDING the per-file `Module` node.
///
/// The C reference makes the module a genuine definition: its
/// `cbm_extract_definitions` (`internal/cbm/extract_defs.c:5469-5480`) pushes
/// a `label="Module"` `CBMDefinition` as the *first* definition of every file,
/// which then flows through `process_def` and gets a `File→DEFINES→Module`
/// edge exactly like any other def. Excluding `Module` here made grepplus's
/// DEFINES count trail C's by precisely one-per-file (rust 88 vs 171 = 88
/// defs + 83 modules; python 425 vs 850; go 16 vs 32; java 246 vs 327). So
/// `Module` is a DEFINES target, and only the File/Folder/Project spine is not.
const NON_DEF_LABELS: [&str; 3] = ["File", "Folder", "Project"];

/// Normalize backslashes to `/` (C `cbm_normalize_path_sep`) and return an
/// owned copy. rel_paths from the discover walk are already `/`-separated on
/// every platform, but we mirror the C step for exactness.
fn norm(path: &str) -> String {
    path.replace('\\', "/")
}

/// The directory portion of a rel_path (everything before the last `/`), or
/// `""` when the file is at the repo root. Mirrors the C `strrchr(rel, '/')`
/// basename split.
fn parent_dir(rel_path: &str) -> &str {
    match rel_path.rfind('/') {
        Some(i) => &rel_path[..i],
        None => "",
    }
}

/// The basename of a rel_path (everything after the last `/`). Mirrors the
/// C `slash ? slash + 1 : rel`.
fn basename(rel_path: &str) -> &str {
    match rel_path.rfind('/') {
        Some(i) => &rel_path[i + 1..],
        None => rel_path,
    }
}

/// File extension including the leading dot (`src/a.rs` → `.rs`), or `""`
/// when the basename has no dot. Mirrors the C `strrchr(basename, '.')`.
fn extension(rel_path: &str) -> String {
    let b = basename(rel_path);
    match b.rfind('.') {
        Some(i) => b[i..].to_string(),
        None => String::new(),
    }
}

/// Strip the file extension from the last path component. Mirrors the C
/// `strip_file_extension`: only the final segment's extension is removed.
fn strip_file_extension(path: &str) -> String {
    let (dir, last) = match path.rfind('/') {
        Some(i) => (&path[..=i], &path[i + 1..]),
        None => ("", path),
    };
    let stripped = match last.rfind('.') {
        Some(i) => &last[..i],
        None => last,
    };
    format!("{dir}{stripped}")
}

/// The File node's qualified_name: `project.dir.parts.__file__`. Faithful
/// port of `cbm_pipeline_fqn_compute(project, rel, "__file__")`:
///
/// 1. normalize separators, strip the final extension,
/// 2. split into `/`-segments, prefix `project`,
/// 3. `__init__`/`index` handling: since a non-empty `name` (`__file__`) is
///    provided, a trailing `__init__`/`index` segment is dropped,
/// 4. append the `__file__` name segment,
/// 5. dot-join.
fn file_qn(project: &str, rel_path: &str) -> String {
    let path = strip_file_extension(&norm(rel_path));
    let mut segs: Vec<String> = vec![project.to_string()];
    for tok in path.split('/') {
        if !tok.is_empty() {
            segs.push(tok.to_string());
        }
    }
    // strip_init_or_index: drop a trailing __init__/index segment when a
    // name is provided (it always is here: "__file__").
    if segs.len() > 1 {
        let last = segs.last().map(String::as_str);
        if last == Some("__init__") || last == Some("index") {
            segs.pop();
        }
    }
    segs.push("__file__".to_string());
    segs.join(".")
}

/// A Folder node's qualified_name: `project.dir.parts`. Faithful port of
/// `cbm_pipeline_fqn_folder(project, rel_dir)` — no extension stripping, no
/// `__init__`/`index` handling; just `project` + the dir segments dot-joined.
fn folder_qn(project: &str, rel_dir: &str) -> String {
    let dir = norm(rel_dir);
    let mut segs: Vec<String> = vec![project.to_string()];
    for tok in dir.split('/') {
        if !tok.is_empty() {
            segs.push(tok.to_string());
        }
    }
    segs.join(".")
}

/// Create the Project / Folder / File structural nodes and the
/// CONTAINS_FOLDER / CONTAINS_FILE / DEFINES edges for `project`, faithfully
/// mirroring the C `pass_structure` + the File→DEFINES half of `process_def`.
///
/// Called from `index()` AFTER all per-file nodes (the synthetic Module node
/// and every extracted definition) have been written, so the DEFINES targets
/// exist. Runs over the FULL discovered `entries` set (every file gets a File
/// node, exactly like C's structural pass, regardless of language support).
///
/// Idempotent: nodes are upserted on `(project, qualified_name)` and edges on
/// `(source_id, target_id, edge_type)`, so a re-index that re-creates the same
/// spine is a no-op on counts.
pub(crate) fn build_structural(
    store: &mut Store,
    project: &str,
    entries: &[InventoryEntry],
) -> Result<()> {
    // ── Project node ────────────────────────────────────────────────
    // label="Project", name = qn = project_name, no file path.
    let project_id = store.insert_node(&NewNode {
        project: project.to_string(),
        label: "Project".into(),
        name: project.to_string(),
        qualified_name: project.to_string(),
        file_path: String::new(),
        start_line: 0,
        end_line: 0,
        properties: serde_json::json!({}),
    })?;

    // ── Folder nodes ────────────────────────────────────────────────
    // Collect every directory (and ancestor) that appears in the file set.
    // C's create_folder_chain walks each file's dir upward, creating a
    // Folder for each not-yet-seen directory and a CONTAINS_FOLDER edge
    // parent→child. We first materialize all folder nodes, then wire the
    // chain, so id lookups always succeed regardless of discovery order.
    let mut dirs: BTreeSet<String> = BTreeSet::new();
    for entry in entries {
        let mut dir = parent_dir(&norm(&entry.rel_path)).to_string();
        while !dir.is_empty() {
            if !dirs.insert(dir.clone()) {
                break; // ancestor already recorded (and its ancestors too)
            }
            dir = parent_dir(&dir).to_string();
        }
    }

    // Insert Folder nodes; remember their ids by rel_dir for edge wiring.
    let mut folder_id: HashMap<String, i64> = HashMap::new();
    for dir in &dirs {
        let id = store.insert_node(&NewNode {
            project: project.to_string(),
            label: "Folder".into(),
            name: basename(dir).to_string(),
            qualified_name: folder_qn(project, dir),
            file_path: dir.clone(),
            start_line: 0,
            end_line: 0,
            properties: serde_json::json!({}),
        })?;
        folder_id.insert(dir.clone(), id);
    }

    // CONTAINS_FOLDER: parent(dir) → dir. The parent is the enclosing
    // directory's Folder node, or the Project node when the dir is
    // top-level. One edge per folder (matches C's per-folder emission).
    let mut edges: Vec<NewEdge> = Vec::new();
    for dir in &dirs {
        let child = folder_id[dir];
        let parent_rel = parent_dir(dir);
        let parent = if parent_rel.is_empty() {
            project_id
        } else {
            folder_id[parent_rel]
        };
        edges.push(NewEdge {
            project: project.to_string(),
            source_id: parent,
            target_id: child,
            edge_type: "CONTAINS_FOLDER".into(),
            properties: serde_json::json!({}),
        });
    }

    // ── File nodes + CONTAINS_FILE + DEFINES edges ──────────────────
    // Track the qnames we (re)create this run so the cleanup pass below can
    // drop structural nodes left behind by a now-deleted file/folder.
    let mut valid_files: BTreeSet<String> = BTreeSet::new();
    for entry in entries {
        let rel = norm(&entry.rel_path);
        let qn = file_qn(project, &rel);
        valid_files.insert(qn.clone());
        let file_id = store.insert_node(&NewNode {
            project: project.to_string(),
            label: "File".into(),
            name: basename(&rel).to_string(),
            qualified_name: qn,
            file_path: rel.clone(),
            start_line: 0,
            end_line: 0,
            properties: serde_json::json!({ "extension": extension(&rel) }),
        })?;

        // CONTAINS_FILE: parent(dir) → file (Project when at repo root).
        let parent_rel = parent_dir(&rel);
        let parent = if parent_rel.is_empty() {
            project_id
        } else {
            folder_id[parent_rel]
        };
        edges.push(NewEdge {
            project: project.to_string(),
            source_id: parent,
            target_id: file_id,
            edge_type: "CONTAINS_FILE".into(),
            properties: serde_json::json!({}),
        });

        // DEFINES: File → every definition extracted from this file. C's
        // process_def inserts a File→DEFINES edge for every def; here the
        // defs are the already-persisted nodes for this file that are NOT
        // part of the structural spine or the synthetic Module node. We
        // list every label in this file (empty label filter) and skip the
        // structural/synthetic labels.
        for def in store.list_nodes(project, "", &rel, 0, usize::MAX)? {
            if NON_DEF_LABELS.contains(&def.label.as_str()) {
                continue;
            }
            edges.push(NewEdge {
                project: project.to_string(),
                source_id: file_id,
                target_id: def.id,
                edge_type: "DEFINES".into(),
                properties: serde_json::json!({}),
            });
        }
    }

    // Persist every structural edge. `insert_edge` upserts on the unique
    // (source, target, type) triple, so re-indexing is idempotent.
    for e in &edges {
        store.insert_edge(e)?;
    }

    // ── Incremental cleanup (review finding P1: stale spine) ─────────
    // Node upserts add/refresh but never REMOVE, so a file or directory
    // deleted since the last run would leave an orphan File/Folder node
    // (and, for a now-empty folder, an orphan CONTAINS_FOLDER edge that the
    // per-file `delete_nodes_for_file` cascade never touches, because a
    // Folder's `file_path` is a directory, not a file). We therefore drop
    // any Project-scoped File/Folder node whose qualified_name is NOT in the
    // set we just (re)built from the CURRENT `entries`. Surviving nodes keep
    // their ids (upsert-by-qname preserved them above), so this only removes
    // genuine leftovers and never churns the live graph; a full index has no
    // leftovers and so is a no-op. Deleting a node cascades its edges via the
    // FK on edges.source_id / edges.target_id.
    let valid_folders: BTreeSet<String> = dirs.iter().map(|d| folder_qn(project, d)).collect();
    for node in store.list_nodes_by_label(project, "Folder", usize::MAX)? {
        if !valid_folders.contains(&node.qualified_name) {
            store.delete_node(node.id)?;
        }
    }
    for node in store.list_nodes_by_label(project, "File", usize::MAX)? {
        if !valid_files.contains(&node.qualified_name) {
            store.delete_node(node.id)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use grepplus_store::Store;
    use std::path::PathBuf;

    fn entry(rel: &str) -> InventoryEntry {
        InventoryEntry {
            rel_path: rel.to_string(),
            abs_path: PathBuf::from(format!("/tmp/{rel}")),
            size: None,
            mtime_ns: None,
        }
    }

    // nodes.project has an FK to projects(name); the real index() upserts the
    // project row before any node write, so the test must too.
    fn store_with_project(p: &str) -> Store {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&grepplus_store::Project {
                name: p.to_string(),
                indexed_at: "1970-01-01T00:00:00Z".to_string(),
                root_path: "/tmp".to_string(),
            })
            .unwrap();
        store
    }

    fn names(store: &Store, project: &str, label: &str) -> Vec<String> {
        let mut v: Vec<String> = store
            .list_nodes_by_label(project, label, usize::MAX)
            .unwrap()
            .into_iter()
            .map(|n| n.file_path)
            .collect();
        v.sort();
        v
    }

    // Review finding P1: a file/folder deleted between indexes must not leave
    // an orphan File/Folder node behind. build_structural rebuilds the spine
    // from the CURRENT entries and drops any structural node no longer backed
    // by one, so a re-run with a different file set fully replaces the spine.
    #[test]
    fn stale_file_and_now_empty_folder_are_removed_on_reindex() {
        let p = "proj";
        let mut store = store_with_project(p);

        build_structural(&mut store, p, &[entry("a/x.rs")]).unwrap();
        assert_eq!(names(&store, p, "File"), vec!["a/x.rs".to_string()]);
        assert_eq!(names(&store, p, "Folder"), vec!["a".to_string()]);

        // x.rs deleted, y.rs added in a different folder. The `a` folder is now
        // empty and must disappear; `a/x.rs` must disappear; `b` + `b/y.rs`
        // must appear. The Project node is always kept.
        build_structural(&mut store, p, &[entry("b/y.rs")]).unwrap();
        assert_eq!(names(&store, p, "File"), vec!["b/y.rs".to_string()]);
        assert_eq!(names(&store, p, "Folder"), vec!["b".to_string()]);
        assert_eq!(store.count_nodes_by_label(p, "Project").unwrap(), 1);
    }

    // A full re-index with the SAME entries is a no-op for the cleanup pass
    // (nothing stale) and preserves node identity via upsert-by-qname.
    #[test]
    fn identical_reindex_preserves_spine_ids() {
        let p = "proj";
        let mut store = store_with_project(p);
        build_structural(&mut store, p, &[entry("src/a.rs"), entry("src/b.rs")]).unwrap();
        let before: Vec<i64> = store
            .list_nodes_by_label(p, "File", usize::MAX)
            .unwrap()
            .into_iter()
            .map(|n| n.id)
            .collect();
        build_structural(&mut store, p, &[entry("src/a.rs"), entry("src/b.rs")]).unwrap();
        let after: Vec<i64> = store
            .list_nodes_by_label(p, "File", usize::MAX)
            .unwrap()
            .into_iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(before, after, "upsert-by-qname must preserve File node ids");
        assert_eq!(store.count_nodes_by_label(p, "File").unwrap(), 2);
    }
}
