//! Trace call paths through the graph.
//!
//! Direction is `Outgoing` (follow `source -> target` edges) for
//! "who does this call?" and `Incoming` (follow `target -> source`
//! edges) for "who calls this?". Both directions honour `max_depth`
//! and `edge_type` filtering.

use std::collections::{HashSet, VecDeque};

use grepplus_core::Result;
use grepplus_store::{Edge, Node, Store};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceDirection {
    Outgoing,
    Incoming,
}

/// One step in the trace.
///
/// R-012 / WP-R012: `qualified_name`, `file_path`, and the
/// `start_line..end_line` span are populated for every step so the
/// caller can present agent-actionable context (file/line/symbol)
/// without needing a follow-up `grepplus search-graph` query.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceStep {
    /// Sequential depth from the start node (0 = start, 1 = direct
    /// neighbour, …).
    pub depth: usize,
    /// The node id at this step.
    pub node_id: i64,
    /// The edge that brought us here (`None` at depth 0).
    pub edge: Option<Edge>,
    /// Cached node metadata so the CLI can print actionable
    /// information without re-querying.
    pub node: Option<Node>,
}

impl TraceStep {
    /// The resolved qualified name of this step's node, if the store
    /// could resolve it. `None` when the node metadata was not cached
    /// (e.g. a dangling edge target the store no longer holds).
    pub fn qualified_name(&self) -> Option<&str> {
        self.node.as_ref().map(|n| n.qualified_name.as_str())
    }

    /// The resolved file path of this step's node, if available.
    pub fn file_path(&self) -> Option<&str> {
        self.node.as_ref().map(|n| n.file_path.as_str())
    }

    /// The resolved label/kind of this step's node, if available.
    pub fn label(&self) -> Option<&str> {
        self.node.as_ref().map(|n| n.label.as_str())
    }

    /// The resolved `start_line..=end_line` span of this step's node,
    /// if available.
    pub fn line_span(&self) -> Option<(i64, i64)> {
        self.node.as_ref().map(|n| (n.start_line, n.end_line))
    }

    /// The edge type that brought the BFS to this step, if any
    /// (`None` at the start node).
    pub fn via_edge_type(&self) -> Option<&str> {
        self.edge.as_ref().map(|e| e.edge_type.as_str())
    }

    /// The resolved leaf `name` of this step's node, if available. The
    /// complement to [`TraceStep::qualified_name`]: where the qualified
    /// name locates the symbol in the module path, this is the bare
    /// identifier the CLI's path/impact output shows as the label.
    pub fn name(&self) -> Option<&str> {
        self.node.as_ref().map(|n| n.name.as_str())
    }

    /// The resolved start line of this step's node, if available. A
    /// convenience over [`TraceStep::line_span`] for callers that only
    /// need the definition's first line (the jump-to target).
    pub fn start_line(&self) -> Option<i64> {
        self.node.as_ref().map(|n| n.start_line)
    }

    /// The resolved end line of this step's node, if available.
    pub fn end_line(&self) -> Option<i64> {
        self.node.as_ref().map(|n| n.end_line)
    }

    /// A single-line presentation of this step's resolved context:
    /// `qualified_name file_path:start-end`. Returns `None` when the
    /// node metadata is unresolved, so a caller can fall back to the
    /// bare `node_id`. Mirrors exactly what the CLI prints today, so
    /// the CLI can call this instead of re-formatting by hand.
    pub fn context_line(&self) -> Option<String> {
        self.node.as_ref().map(|n| {
            format!(
                "{} {}:{}-{}",
                n.qualified_name, n.file_path, n.start_line, n.end_line
            )
        })
    }
}

/// BFS-traverse from `start_id` up to `max_depth`. Returns one
/// `TraceStep` per visited node, in BFS order. The `edge` field on the
/// start node is `None`; on every other node it is the edge that
/// brought the BFS to that node.
pub fn trace_path(
    store: &Store,
    start_id: i64,
    direction: TraceDirection,
    edge_type: Option<&str>,
    max_depth: usize,
) -> Result<Vec<TraceStep>> {
    let mut visited: HashSet<i64> = HashSet::new();
    let mut out: Vec<TraceStep> = Vec::new();
    // (depth, node_id, edge_that_brought_us_here). The start has edge=None.
    let mut queue: VecDeque<(usize, i64, Option<Edge>)> = VecDeque::new();
    queue.push_back((0, start_id, None));

    while let Some((depth, node_id, incoming_edge)) = queue.pop_front() {
        if !visited.insert(node_id) {
            continue;
        }
        // R-012 / WP-R012: refuse to emit a phantom step for a
        // non-existent start node. R-025 flagged that the prior
        // implementation happily emitted a step with `node_id =
        // -1` for an empty queue. We additionally skip start nodes
        // that the store cannot resolve.
        let node_meta = match store.get_node(node_id) {
            Ok(Some(n)) => Some(n),
            Ok(None) => {
                if depth == 0 {
                    // The starting id doesn't exist — return an
                    // empty trace rather than a phantom step.
                    return Ok(Vec::new());
                }
                None
            }
            Err(e) => {
                return Err(grepplus_core::Error::Store(format!(
                    "get_node({node_id}): {e}"
                )));
            }
        };
        out.push(TraceStep {
            depth,
            node_id,
            edge: incoming_edge,
            node: node_meta.clone(),
        });
        if depth >= max_depth {
            continue;
        }
        let neighbours = match direction {
            TraceDirection::Outgoing => store.outgoing_edges(node_id, edge_type, 1024)?,
            TraceDirection::Incoming => store.incoming_edges(node_id, edge_type, 1024)?,
        };
        for e in neighbours {
            let next = match direction {
                TraceDirection::Outgoing => e.target_id,
                TraceDirection::Incoming => e.source_id,
            };
            if visited.contains(&next) {
                continue;
            }
            queue.push_back((depth + 1, next, Some(e)));
        }
    }
    Ok(out)
}

/// The direct callees of `node_id`: the nodes it reaches over a single
/// outgoing `CALLS` edge. This is the one-level convenience form of an
/// outgoing [`trace_path`] — it answers "what does this function call?"
/// without the caller having to set up a `TraceDirection` and a
/// `max_depth` of 1.
///
/// Returns one [`TraceStep`] per direct callee (depth `1`, with the
/// `CALLS` edge that reached it), in the same BFS order [`trace_path`]
/// yields. The start node itself is **not** included (unlike
/// `trace_path`, which emits the start at depth 0), so the result is
/// exactly the neighbour set. A missing start node yields an empty vec.
/// Deterministic: neighbours come from the id-ordered edge table.
pub fn callees_of(store: &Store, node_id: i64) -> Result<Vec<TraceStep>> {
    one_level(store, node_id, TraceDirection::Outgoing)
}

/// The direct callers of `node_id`: the nodes that reach it over a
/// single incoming `CALLS` edge. The one-level convenience form of an
/// incoming [`trace_path`] — it answers "who calls this function?".
///
/// Same shape and guarantees as [`callees_of`]: one [`TraceStep`] per
/// direct caller at depth `1`, start node excluded, deterministic
/// id-ordered output, empty when the start node is missing.
pub fn callers_of(store: &Store, node_id: i64) -> Result<Vec<TraceStep>> {
    one_level(store, node_id, TraceDirection::Incoming)
}

/// Shared one-hop CALLS expansion for [`callees_of`]/[`callers_of`].
/// Drops the depth-0 start step a `trace_path` would emit and dedups
/// repeated edges to the same neighbour, preserving id order.
fn one_level(store: &Store, node_id: i64, direction: TraceDirection) -> Result<Vec<TraceStep>> {
    if store.get_node(node_id)?.is_none() {
        return Ok(Vec::new());
    }
    let neighbours = match direction {
        TraceDirection::Outgoing => store.outgoing_edges(node_id, Some("CALLS"), 1024)?,
        TraceDirection::Incoming => store.incoming_edges(node_id, Some("CALLS"), 1024)?,
    };
    let mut seen: HashSet<i64> = HashSet::new();
    let mut out: Vec<TraceStep> = Vec::new();
    for e in neighbours {
        let next = match direction {
            TraceDirection::Outgoing => e.target_id,
            TraceDirection::Incoming => e.source_id,
        };
        // A self-loop is not a one-level neighbour; skip it.
        if next == node_id || !seen.insert(next) {
            continue;
        }
        let node = store.get_node(next)?;
        out.push(TraceStep {
            depth: 1,
            node_id: next,
            edge: Some(e),
            node,
        });
    }
    Ok(out)
}

/// One node in a bounded [`CallTree`]: the resolved step plus the ids of
/// its children one level deeper. Children ids index back into
/// [`CallTree::nodes`] so the structure stays flat (no recursive owned
/// boxes) while remaining fully reconstructable.
#[derive(Debug, Clone, PartialEq)]
pub struct CallTreeNode {
    /// The resolved step (depth, node id, the edge that reached it, and
    /// cached node metadata). The root has `depth == 0` and `edge ==
    /// None`.
    pub step: TraceStep,
    /// Node ids of this node's direct children at the next depth, in
    /// deterministic (edge-id) order. Empty at a leaf or at `max_depth`.
    pub children: Vec<i64>,
}

/// A bounded, deterministic call tree rooted at a single symbol,
/// expanded over `CALLS` edges in **both** directions at once.
///
/// Where [`trace_path`] walks a single direction, `CallTree` captures the
/// full local call structure: for each node it records the children
/// reached by following `CALLS` edges in the requested [`TraceDirection`]
/// — outgoing for a callee tree ("what does this transitively call?"),
/// incoming for a caller tree ("what transitively calls this?"). It is a
/// *tree*, not a set: a node reached by two distinct paths appears once
/// (first-reached wins, BFS order), so cycles terminate and the output
/// is finite.
#[derive(Debug, Clone, PartialEq)]
pub struct CallTree {
    /// The root node id the tree was built around.
    pub root_id: i64,
    /// Every node in the tree in BFS order (root first). Each carries its
    /// own children list; index by position is not significant — match on
    /// `node.step.node_id`.
    pub nodes: Vec<CallTreeNode>,
}

impl CallTree {
    /// Total number of nodes in the tree (including the root).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty (root did not resolve).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The maximum depth actually reached (0 when only the root is
    /// present, or the tree is empty).
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.step.depth).max().unwrap_or(0)
    }
}

/// Build a bounded call tree rooted at `start_id`, following `CALLS`
/// edges in `direction` up to `max_depth` levels.
///
/// Determinism and bounds:
/// - BFS expands children in ascending edge-`id` order, so the tree
///   structure is reproducible across runs and independent of insertion
///   timing.
/// - A node is placed at the *first* (shallowest) depth it is reached,
///   exactly once; later sightings are not re-added (cycles terminate,
///   diamonds collapse to their nearest occurrence).
/// - `max_depth` is honoured: nodes at `max_depth` have an empty
///   `children` list even if they have further callers/callees.
/// - A missing `start_id` yields an empty tree (`nodes` empty).
///
/// This is the both-directions-friendly companion to [`trace_path`]: call
/// it twice (Outgoing then Incoming) to assemble a full "who calls this,
/// and what does it call" picture, each half deterministic.
pub fn call_tree(
    store: &Store,
    start_id: i64,
    direction: TraceDirection,
    max_depth: usize,
) -> Result<CallTree> {
    let mut nodes: Vec<CallTreeNode> = Vec::new();
    if store.get_node(start_id)?.is_none() {
        return Ok(CallTree {
            root_id: start_id,
            nodes,
        });
    }

    // node_id -> index into `nodes`, recording the first (shallowest)
    // placement so cycles/diamonds resolve to a single occurrence.
    let mut placed: HashSet<i64> = HashSet::new();
    placed.insert(start_id);
    nodes.push(CallTreeNode {
        step: TraceStep {
            depth: 0,
            node_id: start_id,
            edge: None,
            node: store.get_node(start_id)?,
        },
        children: Vec::new(),
    });
    // Queue of (index-into-nodes, depth) still to expand.
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((0, 0));

    while let Some((idx, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        let node_id = nodes[idx].step.node_id;
        let neighbours = match direction {
            TraceDirection::Outgoing => store.outgoing_edges(node_id, Some("CALLS"), 1024)?,
            TraceDirection::Incoming => store.incoming_edges(node_id, Some("CALLS"), 1024)?,
        };
        let mut children: Vec<i64> = Vec::new();
        for e in neighbours {
            let next = match direction {
                TraceDirection::Outgoing => e.target_id,
                TraceDirection::Incoming => e.source_id,
            };
            if next == node_id {
                // Skip self-recursion; it would never terminate as a
                // tree edge and adds no structural information.
                continue;
            }
            if !placed.insert(next) {
                // Already placed at a shallower (or equal) depth; do not
                // duplicate it, but it is still a child of this node.
                children.push(next);
                continue;
            }
            let child_idx = nodes.len();
            nodes.push(CallTreeNode {
                step: TraceStep {
                    depth: depth + 1,
                    node_id: next,
                    edge: Some(e),
                    node: store.get_node(next)?,
                },
                children: Vec::new(),
            });
            children.push(next);
            queue.push_back((child_idx, depth + 1));
        }
        nodes[idx].children = children;
    }

    Ok(CallTree {
        root_id: start_id,
        nodes,
    })
}

/// Summarise a trace as the ordered list of resolved qualified names,
/// from the start node to each subsequently visited node (the BFS
/// "leaf" order). Steps whose node metadata could not be resolved fall
/// back to a `#<node_id>` placeholder so the summary stays positional
/// (one entry per step) and never silently drops a hop.
///
/// This is the cheap presentation companion to [`trace_path`]: a caller
/// that already holds the `Vec<TraceStep>` can render the path "a -> b
/// -> c" without re-querying the store. Order matches the input slice
/// exactly (BFS order from `trace_path`).
pub fn path_summary(steps: &[TraceStep]) -> Vec<String> {
    steps
        .iter()
        .map(|s| match s.qualified_name() {
            Some(q) => q.to_string(),
            None => format!("#{}", s.node_id),
        })
        .collect()
}

/// One node of a recursive [`CallHierarchy`]: the resolved step plus its
/// owned child subtrees.
///
/// Unlike [`CallTreeNode`] — which keeps the tree flat by referencing
/// children through ids into a shared [`CallTree::nodes`] vector and places
/// every node *once globally* (diamonds collapse) — a `CallHierarchyNode`
/// owns its children directly, so the same callee reached under two distinct
/// parents appears as a distinct subtree under each parent. That is the
/// genuine *call hierarchy* an IDE shows: the full incoming-or-outgoing tree,
/// not a deduplicated reachability set.
///
/// Cycle-safety is per *ancestor path*, not global: a node is expanded only
/// if it does not already appear on the path from the root to it. When a
/// child would re-enter an ancestor (a back-edge / recursion), it is emitted
/// as a leaf with [`CallHierarchyNode::cyclic`] set to `true` and **not**
/// expanded, so the tree is always finite.
#[derive(Debug, Clone, PartialEq)]
pub struct CallHierarchyNode {
    /// The resolved step (depth, node id, the edge that reached it, and
    /// cached node metadata). The root has `depth == 0` and `edge == None`.
    pub step: TraceStep,
    /// `true` when this node closes a cycle: its id already appears on the
    /// path from the root, so it is emitted as a leaf (no `children`) to
    /// guarantee termination. The edge that reached it is still recorded on
    /// `step`, so the back-edge is visible to the caller.
    pub cyclic: bool,
    /// Whether expansion stopped here because the depth budget (`max_depth`)
    /// was reached rather than because the node is a genuine leaf. Lets a
    /// caller tell "no more callers/callees" apart from "tree truncated".
    pub truncated: bool,
    /// The owned child subtrees, one per distinct neighbour, in deterministic
    /// (edge-`id`) order. Empty at a genuine leaf, at `max_depth`
    /// (`truncated == true`), or at a cycle-closing node (`cyclic == true`).
    pub children: Vec<CallHierarchyNode>,
}

impl CallHierarchyNode {
    /// The node id at this position in the hierarchy.
    pub fn node_id(&self) -> i64 {
        self.step.node_id
    }

    /// Total number of nodes in the subtree rooted here, including this one.
    pub fn count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(CallHierarchyNode::count)
            .sum::<usize>()
    }

    /// The maximum depth (relative to the overall root) reached anywhere in
    /// the subtree rooted here.
    pub fn max_depth(&self) -> usize {
        self.children
            .iter()
            .map(CallHierarchyNode::max_depth)
            .max()
            .unwrap_or(self.step.depth)
    }
}

/// A recursive, bounded, deterministic *call hierarchy* rooted at one symbol,
/// expanded over `CALLS` edges in a single [`TraceDirection`] (outgoing for a
/// callee tree, incoming for a caller tree) up to `max_depth` levels.
///
/// The companion to [`CallTree`]. Where `CallTree` returns a flat,
/// globally-deduplicated reachability set (each node once, diamonds
/// collapsed), `CallHierarchy` returns the genuine expanded tree: a callee
/// reached under two distinct callers is shown under each. This is what an
/// IDE's "call hierarchy" view presents.
#[derive(Debug, Clone, PartialEq)]
pub struct CallHierarchy {
    /// The direction the tree follows: `Outgoing` = callees, `Incoming` =
    /// callers.
    pub direction: TraceDirection,
    /// The root node, or `None` when `start_id` did not resolve.
    pub root: Option<CallHierarchyNode>,
}

impl CallHierarchy {
    /// Whether the hierarchy is empty (root did not resolve).
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Total node count across the whole tree (0 when empty).
    pub fn len(&self) -> usize {
        self.root
            .as_ref()
            .map(CallHierarchyNode::count)
            .unwrap_or(0)
    }

    /// The maximum depth reached (0 when empty or only the root is present).
    pub fn max_depth(&self) -> usize {
        self.root
            .as_ref()
            .map(CallHierarchyNode::max_depth)
            .unwrap_or(0)
    }
}

/// Build a recursive [`CallHierarchy`] rooted at `start_id`, following
/// `CALLS` edges in `direction` up to `max_depth` levels deep.
///
/// Semantics:
/// - **Full tree, not a set.** A callee/caller reached along two different
///   ancestor paths is expanded under each — unlike [`call_tree`], which
///   places every node once globally. This yields the genuine hierarchy.
/// - **Per-path cycle-safety.** A child whose id already appears on the path
///   from the root to it is emitted as a leaf with `cyclic == true` and is
///   not expanded, so recursion and back-edges terminate and the tree is
///   finite.
/// - **Deterministic.** Children are expanded in ascending edge-`id` order
///   (the order the id-keyed edge table yields), duplicate edges to the same
///   neighbour collapse to one child, and self-loops are skipped — so the
///   structure is byte-stable across runs and independent of insertion
///   timing.
/// - **Bounded depth.** Nodes at `max_depth` are emitted with `truncated ==
///   true` and an empty `children` list even when they have further
///   callers/callees. A `max_depth` of `0` yields just the root.
/// - A missing `start_id` yields `CallHierarchy { root: None, .. }`.
pub fn call_hierarchy(
    store: &Store,
    start_id: i64,
    direction: TraceDirection,
    max_depth: usize,
) -> Result<CallHierarchy> {
    let Some(root_node) = store.get_node(start_id)? else {
        return Ok(CallHierarchy {
            direction,
            root: None,
        });
    };

    // The ancestor path (root..=current node, exclusive of the node being
    // expanded's own children) used for per-path cycle detection.
    let mut path: HashSet<i64> = HashSet::new();
    let root = expand_hierarchy(
        store,
        start_id,
        Some(root_node),
        None,
        0,
        direction,
        max_depth,
        &mut path,
    )?;
    Ok(CallHierarchy {
        direction,
        root: Some(root),
    })
}

/// Recursive worker for [`call_hierarchy`]. `path` holds the ids of every
/// ancestor strictly above this node; the node's own id is inserted before
/// recursing into children and removed afterwards, so the set always
/// reflects the current root-to-node path exactly.
#[allow(clippy::too_many_arguments)]
fn expand_hierarchy(
    store: &Store,
    node_id: i64,
    node_meta: Option<Node>,
    incoming_edge: Option<Edge>,
    depth: usize,
    direction: TraceDirection,
    max_depth: usize,
    path: &mut HashSet<i64>,
) -> Result<CallHierarchyNode> {
    let step = TraceStep {
        depth,
        node_id,
        edge: incoming_edge,
        node: node_meta,
    };

    if depth >= max_depth {
        return Ok(CallHierarchyNode {
            step,
            cyclic: false,
            truncated: true,
            children: Vec::new(),
        });
    }

    let neighbours = match direction {
        TraceDirection::Outgoing => store.outgoing_edges(node_id, Some("CALLS"), 1024)?,
        TraceDirection::Incoming => store.incoming_edges(node_id, Some("CALLS"), 1024)?,
    };

    path.insert(node_id);
    let mut seen: HashSet<i64> = HashSet::new();
    let mut children: Vec<CallHierarchyNode> = Vec::new();
    for e in neighbours {
        let next = match direction {
            TraceDirection::Outgoing => e.target_id,
            TraceDirection::Incoming => e.source_id,
        };
        // Skip self-loops and duplicate edges to the same neighbour.
        if next == node_id || !seen.insert(next) {
            continue;
        }
        if path.contains(&next) {
            // Back-edge into an ancestor: emit a cyclic leaf, do not expand.
            children.push(CallHierarchyNode {
                step: TraceStep {
                    depth: depth + 1,
                    node_id: next,
                    edge: Some(e),
                    node: store.get_node(next)?,
                },
                cyclic: true,
                truncated: false,
                children: Vec::new(),
            });
            continue;
        }
        let child_meta = store.get_node(next)?;
        let child = expand_hierarchy(
            store,
            next,
            child_meta,
            Some(e),
            depth + 1,
            direction,
            max_depth,
            path,
        )?;
        children.push(child);
    }
    path.remove(&node_id);

    Ok(CallHierarchyNode {
        step,
        cyclic: false,
        truncated: false,
        children,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use grepplus_store::{NewEdge, NewNode, Project, Store};

    fn seed_chain() -> (Store, i64, i64, i64) {
        // a -> b -> c  with all CALLS edges.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let a = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "a".into(),
                qualified_name: "p::Function::a".into(),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        let b = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "b".into(),
                qualified_name: "p::Function::b".into(),
                file_path: "b.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        let c = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "c".into(),
                qualified_name: "p::Function::c".into(),
                file_path: "c.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: a,
            target_id: b,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: b,
            target_id: c,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        (s, a, b, c)
    }

    #[test]
    fn outgoing_trace_walks_chain() {
        let (s, a, b, c) = seed_chain();
        let steps = trace_path(&s, a, TraceDirection::Outgoing, Some("CALLS"), 5).unwrap();
        let ids: Vec<i64> = steps.iter().map(|s| s.node_id).collect();
        assert_eq!(ids, vec![a, b, c], "BFS should walk the entire chain");
        // The start node has no incoming edge in this trace.
        assert!(steps[0].edge.is_none());
        // Depth 1 and 2 entries have edges attached.
        assert!(steps[1].edge.is_some());
        assert!(steps[2].edge.is_some());
    }

    #[test]
    fn incoming_trace_walks_backwards() {
        let (s, a, b, c) = seed_chain();
        // c is called by b; b is called by a. Incoming trace from c
        // walks back through b to a.
        let steps = trace_path(&s, c, TraceDirection::Incoming, Some("CALLS"), 5).unwrap();
        let names: Vec<i64> = steps.iter().map(|s| s.node_id).collect();
        assert_eq!(names, vec![c, b, a]);
    }

    #[test]
    fn trace_respects_max_depth() {
        let (s, a, _, _) = seed_chain();
        let steps = trace_path(&s, a, TraceDirection::Outgoing, Some("CALLS"), 1).unwrap();
        // depth 0 (start) + depth 1 (b). c (depth 2) must NOT be present.
        let max_d = steps.iter().map(|s| s.depth).max().unwrap_or(0);
        assert_eq!(max_d, 1);
    }

    #[test]
    fn trace_with_no_edges_returns_only_start() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let a = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "lonely".into(),
                qualified_name: "p::Function::lonely".into(),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        let steps = trace_path(&s, a, TraceDirection::Outgoing, None, 5).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].node_id, a);
    }

    #[test]
    fn trace_step_exposes_resolved_node_context() {
        let (s, a, b, _c) = seed_chain();
        let steps = trace_path(&s, a, TraceDirection::Outgoing, Some("CALLS"), 5).unwrap();
        // Start node: resolved context, no incoming edge.
        assert_eq!(steps[0].qualified_name(), Some("p::Function::a"));
        assert_eq!(steps[0].file_path(), Some("a.rs"));
        assert_eq!(steps[0].label(), Some("Function"));
        assert_eq!(steps[0].line_span(), Some((1, 1)));
        assert_eq!(steps[0].via_edge_type(), None);
        // Depth-1 node b arrived via a CALLS edge.
        assert_eq!(steps[1].node_id, b);
        assert_eq!(steps[1].qualified_name(), Some("p::Function::b"));
        assert_eq!(steps[1].via_edge_type(), Some("CALLS"));
        // context_line mirrors the CLI's existing format exactly.
        assert_eq!(
            steps[0].context_line().as_deref(),
            Some("p::Function::a a.rs:1-1")
        );
    }

    #[test]
    fn path_summary_lists_qualified_names_in_bfs_order() {
        let (s, a, _b, _c) = seed_chain();
        let steps = trace_path(&s, a, TraceDirection::Outgoing, Some("CALLS"), 5).unwrap();
        let summary = path_summary(&steps);
        assert_eq!(
            summary,
            vec![
                "p::Function::a".to_string(),
                "p::Function::b".to_string(),
                "p::Function::c".to_string(),
            ]
        );
    }

    #[test]
    fn path_summary_falls_back_to_id_placeholder_for_unresolved_step() {
        // A step whose node metadata is absent (None) should yield a
        // positional `#id` placeholder rather than being dropped.
        let unresolved = TraceStep {
            depth: 1,
            node_id: 42,
            edge: None,
            node: None,
        };
        let summary = path_summary(&[unresolved]);
        assert_eq!(summary, vec!["#42".to_string()]);
    }

    #[test]
    fn path_summary_is_positional_one_entry_per_step() {
        let (s, _a, _b, c) = seed_chain();
        // Incoming trace from c walks c, b, a — three steps, three names.
        let steps = trace_path(&s, c, TraceDirection::Incoming, Some("CALLS"), 5).unwrap();
        let summary = path_summary(&steps);
        assert_eq!(summary.len(), steps.len());
        assert_eq!(summary.first().map(String::as_str), Some("p::Function::c"));
    }

    #[test]
    fn callees_of_returns_direct_outgoing_calls_only() {
        // a -> b -> c. a's direct callees are just {b}, not c.
        let (s, a, b, c) = seed_chain();
        let callees = callees_of(&s, a).unwrap();
        let ids: Vec<i64> = callees.iter().map(|s| s.node_id).collect();
        assert_eq!(ids, vec![b]);
        // Start node is excluded; the step is at depth 1 with its edge.
        assert_eq!(callees[0].depth, 1);
        assert_eq!(callees[0].via_edge_type(), Some("CALLS"));
        assert!(callees.iter().all(|s| s.node_id != a && s.node_id != c));
    }

    #[test]
    fn callers_of_returns_direct_incoming_calls_only() {
        // a -> b -> c. c's direct callers are just {b}, not a.
        let (s, a, b, c) = seed_chain();
        let callers = callers_of(&s, c).unwrap();
        let ids: Vec<i64> = callers.iter().map(|s| s.node_id).collect();
        assert_eq!(ids, vec![b]);
        assert!(callers.iter().all(|s| s.node_id != a && s.node_id != c));
    }

    #[test]
    fn callees_and_callers_of_missing_node_is_empty() {
        let (s, _a, _b, _c) = seed_chain();
        assert!(callees_of(&s, 999_999).unwrap().is_empty());
        assert!(callers_of(&s, 999_999).unwrap().is_empty());
    }

    #[test]
    fn callees_of_resolves_node_context() {
        let (s, a, b, _c) = seed_chain();
        let callees = callees_of(&s, a).unwrap();
        assert_eq!(callees[0].node_id, b);
        assert_eq!(callees[0].qualified_name(), Some("p::Function::b"));
        assert_eq!(callees[0].file_path(), Some("b.rs"));
    }

    #[test]
    fn callees_of_dedups_repeated_edges_and_skips_self_loop() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mk = |s: &mut Store, name: &str| {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: format!("p::{name}"),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap()
        };
        let a = mk(&mut s, "a");
        let b = mk(&mut s, "b");
        let edge = |s: &mut Store, src: i64, tgt: i64| {
            s.insert_edge(&NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(&mut s, a, a); // self-loop
        edge(&mut s, a, b);
        edge(&mut s, a, b); // duplicate edge to b
        let callees = callees_of(&s, a).unwrap();
        let ids: Vec<i64> = callees.iter().map(|s| s.node_id).collect();
        assert_eq!(ids, vec![b], "self-loop skipped, duplicate b collapsed");
    }

    #[test]
    fn call_tree_outgoing_captures_bounded_structure() {
        let (s, a, b, c) = seed_chain();
        let tree = call_tree(&s, a, TraceDirection::Outgoing, 5).unwrap();
        assert_eq!(tree.root_id, a);
        // BFS order: a (root), b, c.
        let ids: Vec<i64> = tree.nodes.iter().map(|n| n.step.node_id).collect();
        assert_eq!(ids, vec![a, b, c]);
        // Root's child is b; b's child is c; c is a leaf.
        let root = &tree.nodes[0];
        assert_eq!(root.children, vec![b]);
        assert_eq!(root.step.depth, 0);
        assert!(root.step.edge.is_none());
        let bn = tree.nodes.iter().find(|n| n.step.node_id == b).unwrap();
        assert_eq!(bn.children, vec![c]);
        let cn = tree.nodes.iter().find(|n| n.step.node_id == c).unwrap();
        assert!(cn.children.is_empty());
        assert_eq!(tree.max_depth(), 2);
    }

    #[test]
    fn call_tree_incoming_walks_backwards() {
        let (s, a, b, c) = seed_chain();
        let tree = call_tree(&s, c, TraceDirection::Incoming, 5).unwrap();
        let ids: Vec<i64> = tree.nodes.iter().map(|n| n.step.node_id).collect();
        assert_eq!(ids, vec![c, b, a]);
    }

    #[test]
    fn call_tree_respects_max_depth() {
        let (s, a, b, c) = seed_chain();
        let tree = call_tree(&s, a, TraceDirection::Outgoing, 1).unwrap();
        let ids: Vec<i64> = tree.nodes.iter().map(|n| n.step.node_id).collect();
        // Only a (depth 0) and b (depth 1); c (depth 2) excluded.
        assert_eq!(ids, vec![a, b]);
        assert!(!ids.contains(&c));
        // b is at max depth, so its children list is empty even though it
        // actually calls c.
        let bn = tree.nodes.iter().find(|n| n.step.node_id == b).unwrap();
        assert!(bn.children.is_empty());
        assert_eq!(tree.max_depth(), 1);
    }

    #[test]
    fn call_tree_terminates_on_cycle() {
        // a -> b -> a (cycle). The tree must terminate, placing each node
        // once; b's child a points back at the already-placed root.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mk = |s: &mut Store, name: &str| {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: format!("p::{name}"),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap()
        };
        let a = mk(&mut s, "a");
        let b = mk(&mut s, "b");
        let edge = |s: &mut Store, src: i64, tgt: i64| {
            s.insert_edge(&NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(&mut s, a, b);
        edge(&mut s, b, a);
        let tree = call_tree(&s, a, TraceDirection::Outgoing, 10).unwrap();
        let ids: Vec<i64> = tree.nodes.iter().map(|n| n.step.node_id).collect();
        assert_eq!(ids, vec![a, b], "each node placed exactly once");
        // b's child is a (the already-placed root) — edge recorded.
        let bn = tree.nodes.iter().find(|n| n.step.node_id == b).unwrap();
        assert_eq!(bn.children, vec![a]);
    }

    #[test]
    fn call_tree_missing_root_is_empty() {
        let (s, _a, _b, _c) = seed_chain();
        let tree = call_tree(&s, 999_999, TraceDirection::Outgoing, 5).unwrap();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.max_depth(), 0);
    }

    #[test]
    fn call_tree_is_deterministic_across_runs() {
        let (s, a, _b, _c) = seed_chain();
        let first = call_tree(&s, a, TraceDirection::Outgoing, 10).unwrap();
        let second = call_tree(&s, a, TraceDirection::Outgoing, 10).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn trace_step_exposes_name_and_per_line_accessors() {
        let (s, a, b, _c) = seed_chain();
        let steps = trace_path(&s, a, TraceDirection::Outgoing, Some("CALLS"), 5).unwrap();
        // Start node a.
        assert_eq!(steps[0].name(), Some("a"));
        assert_eq!(steps[0].start_line(), Some(1));
        assert_eq!(steps[0].end_line(), Some(1));
        // Depth-1 node b: name + line accessors agree with line_span.
        let bn = steps.iter().find(|s| s.node_id == b).unwrap();
        assert_eq!(bn.name(), Some("b"));
        assert_eq!(bn.qualified_name(), Some("p::Function::b"));
        assert_eq!(bn.file_path(), Some("b.rs"));
        let (lo, hi) = bn.line_span().unwrap();
        assert_eq!(bn.start_line(), Some(lo));
        assert_eq!(bn.end_line(), Some(hi));
    }

    #[test]
    fn trace_step_accessors_are_none_for_unresolved_node() {
        let unresolved = TraceStep {
            depth: 1,
            node_id: 42,
            edge: None,
            node: None,
        };
        assert_eq!(unresolved.name(), None);
        assert_eq!(unresolved.start_line(), None);
        assert_eq!(unresolved.end_line(), None);
        assert_eq!(unresolved.qualified_name(), None);
        assert_eq!(unresolved.file_path(), None);
    }

    /// Helper: build a store and an edge-insert closure for hierarchy tests.
    fn seed_empty() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s
    }

    fn mk_node(s: &mut Store, name: &str) -> i64 {
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: name.into(),
            qualified_name: format!("p::{name}"),
            file_path: format!("{name}.rs"),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap()
    }

    fn mk_call(s: &mut Store, src: i64, tgt: i64) {
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: src,
            target_id: tgt,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
    }

    #[test]
    fn call_hierarchy_outgoing_builds_recursive_tree() {
        let (s, a, b, c) = seed_chain();
        let h = call_hierarchy(&s, a, TraceDirection::Outgoing, 5).unwrap();
        assert_eq!(h.direction, TraceDirection::Outgoing);
        let root = h.root.as_ref().unwrap();
        assert_eq!(root.node_id(), a);
        assert_eq!(root.step.depth, 0);
        assert!(root.step.edge.is_none());
        assert!(!root.cyclic && !root.truncated);
        // a -> b -> c, one child each, owned recursively.
        assert_eq!(root.children.len(), 1);
        let bn = &root.children[0];
        assert_eq!(bn.node_id(), b);
        assert_eq!(bn.step.via_edge_type(), Some("CALLS"));
        assert_eq!(bn.children.len(), 1);
        let cn = &bn.children[0];
        assert_eq!(cn.node_id(), c);
        assert!(cn.children.is_empty());
        assert_eq!(h.len(), 3);
        assert_eq!(h.max_depth(), 2);
    }

    #[test]
    fn call_hierarchy_incoming_walks_callers() {
        let (s, a, b, c) = seed_chain();
        let h = call_hierarchy(&s, c, TraceDirection::Incoming, 5).unwrap();
        let root = h.root.as_ref().unwrap();
        assert_eq!(root.node_id(), c);
        assert_eq!(root.children[0].node_id(), b);
        assert_eq!(root.children[0].children[0].node_id(), a);
    }

    #[test]
    fn call_hierarchy_expands_diamond_under_each_parent() {
        // a calls b and c; both b and c call d. Unlike `call_tree` (which
        // places d once globally), the hierarchy expands d under BOTH b and c.
        let mut s = seed_empty();
        let a = mk_node(&mut s, "a");
        let b = mk_node(&mut s, "b");
        let c = mk_node(&mut s, "c");
        let d = mk_node(&mut s, "d");
        mk_call(&mut s, a, b);
        mk_call(&mut s, a, c);
        mk_call(&mut s, b, d);
        mk_call(&mut s, c, d);
        let h = call_hierarchy(&s, a, TraceDirection::Outgoing, 5).unwrap();
        let root = h.root.as_ref().unwrap();
        assert_eq!(root.children.len(), 2);
        let bn = root.children.iter().find(|n| n.node_id() == b).unwrap();
        let cn = root.children.iter().find(|n| n.node_id() == c).unwrap();
        // d appears as a child of both b and c.
        assert_eq!(bn.children.len(), 1);
        assert_eq!(bn.children[0].node_id(), d);
        assert_eq!(cn.children.len(), 1);
        assert_eq!(cn.children[0].node_id(), d);
        // Five nodes total: a, b, c, d-under-b, d-under-c.
        assert_eq!(h.len(), 5);
    }

    #[test]
    fn call_hierarchy_marks_cycle_leaf_and_terminates() {
        // a -> b -> a (cycle). The second a is a cyclic leaf, not expanded.
        let mut s = seed_empty();
        let a = mk_node(&mut s, "a");
        let b = mk_node(&mut s, "b");
        mk_call(&mut s, a, b);
        mk_call(&mut s, b, a);
        let h = call_hierarchy(&s, a, TraceDirection::Outgoing, 10).unwrap();
        let root = h.root.as_ref().unwrap();
        assert!(!root.cyclic);
        let bn = &root.children[0];
        assert_eq!(bn.node_id(), b);
        assert_eq!(bn.children.len(), 1);
        let back = &bn.children[0];
        assert_eq!(back.node_id(), a);
        assert!(back.cyclic, "back-edge into ancestor marked cyclic");
        assert!(back.children.is_empty(), "cyclic leaf is not expanded");
        // a, b, a(cyclic-leaf) = 3 nodes; terminates despite the cycle.
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn call_hierarchy_self_loop_is_skipped() {
        let mut s = seed_empty();
        let a = mk_node(&mut s, "a");
        mk_call(&mut s, a, a);
        let h = call_hierarchy(&s, a, TraceDirection::Outgoing, 5).unwrap();
        let root = h.root.as_ref().unwrap();
        assert!(root.children.is_empty(), "self-loop is not a child");
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn call_hierarchy_dedups_duplicate_edges() {
        let mut s = seed_empty();
        let a = mk_node(&mut s, "a");
        let b = mk_node(&mut s, "b");
        mk_call(&mut s, a, b);
        mk_call(&mut s, a, b); // duplicate (upserted, but exercise dedup)
        let h = call_hierarchy(&s, a, TraceDirection::Outgoing, 5).unwrap();
        let root = h.root.as_ref().unwrap();
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].node_id(), b);
    }

    #[test]
    fn call_hierarchy_respects_max_depth_and_marks_truncated() {
        let (s, a, b, c) = seed_chain();
        let h = call_hierarchy(&s, a, TraceDirection::Outgoing, 1).unwrap();
        let root = h.root.as_ref().unwrap();
        let bn = &root.children[0];
        assert_eq!(bn.node_id(), b);
        // b is at max_depth: truncated, no children even though it calls c.
        assert!(bn.truncated);
        assert!(bn.children.is_empty());
        assert!(!hierarchy_contains(&h, c));
        assert_eq!(h.max_depth(), 1);
    }

    #[test]
    fn call_hierarchy_depth_zero_yields_only_root() {
        let (s, a, _b, _c) = seed_chain();
        let h = call_hierarchy(&s, a, TraceDirection::Outgoing, 0).unwrap();
        let root = h.root.as_ref().unwrap();
        assert!(root.truncated);
        assert!(root.children.is_empty());
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn call_hierarchy_missing_root_is_empty() {
        let (s, _a, _b, _c) = seed_chain();
        let h = call_hierarchy(&s, 999_999, TraceDirection::Outgoing, 5).unwrap();
        assert!(h.is_empty());
        assert!(h.root.is_none());
        assert_eq!(h.len(), 0);
        assert_eq!(h.max_depth(), 0);
    }

    #[test]
    fn call_hierarchy_is_deterministic_across_runs() {
        let mut s = seed_empty();
        let a = mk_node(&mut s, "a");
        let b = mk_node(&mut s, "b");
        let c = mk_node(&mut s, "c");
        let d = mk_node(&mut s, "d");
        mk_call(&mut s, a, b);
        mk_call(&mut s, a, c);
        mk_call(&mut s, b, d);
        mk_call(&mut s, c, d);
        mk_call(&mut s, d, a); // back-edge to root
        let first = call_hierarchy(&s, a, TraceDirection::Outgoing, 10).unwrap();
        let second = call_hierarchy(&s, a, TraceDirection::Outgoing, 10).unwrap();
        assert_eq!(first, second);
    }

    /// Test-only helper: does any node in the hierarchy carry `node_id == id`?
    fn hierarchy_contains(h: &CallHierarchy, id: i64) -> bool {
        fn walk(n: &CallHierarchyNode, id: i64) -> bool {
            n.node_id() == id || n.children.iter().any(|c| walk(c, id))
        }
        h.root.as_ref().is_some_and(|r| walk(r, id))
    }
}
