//! Structured graph search.
//!
//! Mirrors the upstream's `search_graph` MCP tool surface. Each filter
//! is optional; absent filters match everything. The combination is
//! `AND`.

use grepplus_core::Result;
use grepplus_store::Store;

impl GraphQuery {
    /// Build a query that matches everything.
    pub fn any() -> Self {
        Self::default()
    }

    /// Build a query for a single project (filter empty otherwise).
    pub fn in_project(project: impl Into<String>) -> Self {
        Self {
            project: Some(project.into()),
            ..Self::default()
        }
    }

    /// Restrict to a single label (Function, Struct, Import, …).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Restrict to a name (exact match against `nodes.name`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Restrict to a substring match on `nodes.qualified_name`.
    pub fn with_qualified_name_contains(mut self, qname_substr: impl Into<String>) -> Self {
        self.qname_contains = Some(qname_substr.into());
        self
    }

    /// Restrict to file paths matching a glob-like substring.
    pub fn with_file_path_contains(mut self, file_substr: impl Into<String>) -> Self {
        self.file_path_contains = Some(file_substr.into());
        self
    }

    /// Restrict to nodes whose `name` contains the given substring
    /// (case-sensitive, mirrors openCypher `name CONTAINS '…'`).
    pub fn with_name_contains(mut self, name_substr: impl Into<String>) -> Self {
        self.name_contains = Some(name_substr.into());
        self
    }

    /// Restrict to nodes whose `name` starts with the given prefix
    /// (case-sensitive, mirrors openCypher `name STARTS WITH '…'`).
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = Some(prefix.into());
        self
    }

    /// Restrict to nodes whose `file_path` starts with the given
    /// prefix. Useful to scope a query to a directory subtree.
    pub fn with_file_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.file_path_prefix = Some(prefix.into());
        self
    }

    /// Restrict to nodes that have at least one **outgoing** edge of
    /// the given type (e.g. `"CALLS"` → only nodes that call
    /// something). Mirrors a Cypher `(n)-[:TYPE]->()` existence
    /// predicate. Composable with all other filters (`AND`).
    pub fn with_outgoing_edge(mut self, edge_type: impl Into<String>) -> Self {
        self.has_outgoing_edge = Some(edge_type.into());
        self
    }

    /// Restrict to nodes that have at least one **incoming** edge of
    /// the given type (e.g. `"CALLS"` → only nodes that are called by
    /// something). Mirrors a Cypher `()-[:TYPE]->(n)` existence
    /// predicate. Composable with all other filters (`AND`).
    pub fn with_incoming_edge(mut self, edge_type: impl Into<String>) -> Self {
        self.has_incoming_edge = Some(edge_type.into());
        self
    }
}

/// Ordering applied to [`search_graph`] results.
///
/// The default — [`GraphOrder::QualifiedName`] — reproduces the historic
/// behaviour (sort by `qualified_name`, tie-break on `id`), so existing
/// callers that never set an order observe no change. Every variant
/// produces a *total* order (the final tie-break is always the node `id`,
/// which SQLite assigns monotonically), so the result set is byte-stable
/// across runs regardless of the chosen key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphOrder {
    /// Ascending `qualified_name`, then `id`. The historic default.
    #[default]
    QualifiedName,
    /// Ascending leaf `name`, then `qualified_name`, then `id`.
    Name,
    /// Ascending `file_path`, then `start_line`, then `id` — groups a
    /// file's symbols together in source order.
    File,
    /// Descending out-degree (number of outgoing `edge_type` edges),
    /// then `qualified_name`, then `id`. Most-connected sources first.
    OutDegreeDesc,
    /// Descending in-degree (number of incoming `edge_type` edges),
    /// then `qualified_name`, then `id`. Most-referenced targets first.
    InDegreeDesc,
}

/// Run the query against the store.
pub fn search_graph(store: &Store, q: &GraphQuery) -> Result<Vec<SearchGraphRow>> {
    let mut sql = String::from(
        "SELECT id, project, label, name, qualified_name, file_path, start_line, end_line
         FROM nodes WHERE 1=1",
    );
    let mut binds: Vec<String> = Vec::new();
    append_graph_filters(&mut sql, &mut binds, q);

    // Column-based orders can be pushed into SQL with a `LIMIT` so the
    // store does the sort and truncation. Degree-based orders need a
    // count per row, so they fetch the full candidate set ordered
    // deterministically and sort/truncate in Rust below. Either way the
    // final order is total (id is always the last tie-break).
    let pushdown_order: Option<&str> = match q.order_by {
        GraphOrder::QualifiedName => Some("qualified_name, id"),
        GraphOrder::Name => Some("name, qualified_name, id"),
        GraphOrder::File => Some("file_path, start_line, id"),
        // Degree orders sort in Rust; fetch in a stable base order so the
        // candidate set itself is reproducible before re-sorting.
        GraphOrder::OutDegreeDesc | GraphOrder::InDegreeDesc => None,
    };

    let limit = q.limit as i64;
    let conn = store.conn();

    if let Some(order_clause) = pushdown_order {
        sql.push_str(" ORDER BY ");
        sql.push_str(order_clause);
        sql.push_str(" LIMIT ?");
        let mut stmt = conn.prepare(&sql).map_err(grepplus_store::Error::Sqlite)?;
        let params_iter: Vec<&dyn rusqlite::ToSql> = {
            let mut v: Vec<&dyn rusqlite::ToSql> =
                binds.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            v.push(&limit as &dyn rusqlite::ToSql);
            v
        };
        let rows = stmt
            .query_map(params_iter.as_slice(), row_to_search_row)
            .map_err(grepplus_store::Error::Sqlite)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(grepplus_store::Error::Sqlite)?;
        return Ok(rows);
    }

    // Degree ordering: fetch the full filtered candidate set in a stable
    // base order, then sort by degree (descending) in Rust and truncate.
    sql.push_str(" ORDER BY qualified_name, id");
    let mut stmt = conn.prepare(&sql).map_err(grepplus_store::Error::Sqlite)?;
    let params_iter: Vec<&dyn rusqlite::ToSql> =
        binds.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
    let mut rows = stmt
        .query_map(params_iter.as_slice(), row_to_search_row)
        .map_err(grepplus_store::Error::Sqlite)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(grepplus_store::Error::Sqlite)?;
    drop(stmt);

    // Degree counts respect the same edge-type filter the predicates use:
    // when the caller scoped the order to an edge type via `order_edge_type`
    // only those edges count; otherwise all edge types count.
    let edge_filter = q.order_edge_type.as_deref();
    let mut keyed: Vec<(i64, SearchGraphRow)> = Vec::with_capacity(rows.len());
    for row in rows.drain(..) {
        let degree = match q.order_by {
            GraphOrder::OutDegreeDesc => store
                .outgoing_edges(row.id, edge_filter, MAX_REACH_RESULTS)?
                .len() as i64,
            GraphOrder::InDegreeDesc => store
                .incoming_edges(row.id, edge_filter, MAX_REACH_RESULTS)?
                .len() as i64,
            _ => 0,
        };
        keyed.push((degree, row));
    }
    // Descending degree, then the stable base order (qualified_name, id).
    keyed.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.qualified_name.cmp(&b.1.qualified_name))
            .then_with(|| a.1.id.cmp(&b.1.id))
    });
    keyed.truncate(q.limit);
    Ok(keyed.into_iter().map(|(_, r)| r).collect())
}

/// Count the full candidate set for a [`GraphQuery`] without applying its
/// display limit or ordering.
pub fn count_search_graph(store: &Store, q: &GraphQuery) -> Result<usize> {
    let mut sql = String::from("SELECT COUNT(*) FROM nodes WHERE 1=1");
    let mut binds: Vec<String> = Vec::new();
    append_graph_filters(&mut sql, &mut binds, q);

    let conn = store.conn();
    let mut stmt = conn.prepare(&sql).map_err(grepplus_store::Error::Sqlite)?;
    let params_iter: Vec<&dyn rusqlite::ToSql> =
        binds.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
    let count: i64 = stmt
        .query_row(params_iter.as_slice(), |row| row.get(0))
        .map_err(grepplus_store::Error::Sqlite)?;
    Ok(count.max(0) as usize)
}

fn append_graph_filters(sql: &mut String, binds: &mut Vec<String>, q: &GraphQuery) {
    if let Some(p) = &q.project {
        sql.push_str(" AND project = ?");
        binds.push(p.clone());
    }
    if let Some(l) = &q.label {
        sql.push_str(" AND label = ?");
        binds.push(l.clone());
    }
    if let Some(n) = &q.name {
        sql.push_str(" AND name = ?");
        binds.push(n.clone());
    }
    if let Some(s) = &q.qname_contains {
        sql.push_str(" AND qualified_name LIKE ? ESCAPE '\\'");
        binds.push(format!("%{}%", like_escape(s)));
    }
    if let Some(s) = &q.file_path_contains {
        sql.push_str(" AND file_path LIKE ? ESCAPE '\\'");
        binds.push(format!("%{}%", like_escape(s)));
    }
    if let Some(s) = &q.name_contains {
        sql.push_str(" AND name LIKE ? ESCAPE '\\'");
        binds.push(format!("%{}%", like_escape(s)));
    }
    if let Some(s) = &q.name_prefix {
        sql.push_str(" AND name LIKE ? ESCAPE '\\'");
        binds.push(format!("{}%", like_escape(s)));
    }
    if let Some(s) = &q.file_path_prefix {
        sql.push_str(" AND file_path LIKE ? ESCAPE '\\'");
        binds.push(format!("{}%", like_escape(s)));
    }
    if let Some(s) = &q.file_path_exact {
        sql.push_str(" AND file_path = ?");
        binds.push(s.clone());
    }
    // Neighbour / edge-existence predicates. A correlated EXISTS
    // subquery keeps the predicate composable (`AND`) with every other
    // filter and avoids materialising the whole edge set in Rust.
    if let Some(t) = &q.has_outgoing_edge {
        sql.push_str(
            " AND EXISTS (SELECT 1 FROM edges e WHERE e.source_id = nodes.id AND e.edge_type = ?)",
        );
        binds.push(t.clone());
    }
    if let Some(t) = &q.has_incoming_edge {
        sql.push_str(
            " AND EXISTS (SELECT 1 FROM edges e WHERE e.target_id = nodes.id AND e.edge_type = ?)",
        );
        binds.push(t.clone());
    }
}

/// Direction for a bounded reachability traversal.
///
/// `Outgoing` follows `source -> target` edges ("what does this node
/// reach / call?"); `Incoming` follows them in reverse ("what reaches /
/// calls this node?"). Mirrors [`crate::TraceDirection`] but is kept
/// local to the graph-query surface so a caller can run reachability
/// without pulling in the `trace` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReachDirection {
    Outgoing,
    Incoming,
}

/// One node reached by [`reachable_within`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReachableNode {
    /// Minimum number of hops from the start node (`1` for a direct
    /// neighbour; the start node itself is never emitted).
    pub hops: usize,
    /// The reached node's row (same shape as [`search_graph`] results).
    pub node: SearchGraphRow,
}

/// Hard cap on the hop depth accepted by [`reachable_within`]. Requests
/// above this are clamped, bounding traversal cost on adversarial input.
pub const MAX_REACH_HOPS: usize = 32;

/// Hard cap on the number of nodes [`reachable_within`] will return.
/// Requests above this are clamped.
pub const MAX_REACH_RESULTS: usize = 10_000;

/// Per-node fan-out cap when expanding neighbours. Bounds the work done
/// at any single node regardless of its degree.
const REACH_FANOUT: usize = 4096;

/// Bounded multi-hop reachability query.
///
/// Returns every node reachable from `start_id` within `max_hops` edges
/// of type `edge_type`, following edges in `direction`. This is the
/// graph-query counterpart to [`crate::trace_path`]: where `trace_path`
/// yields an ordered BFS walk (with the edge that reached each node and
/// cached node metadata), `reachable_within` yields the deduplicated
/// *set* of reachable nodes as `search_graph`-shaped rows, ready to feed
/// the rest of the query surface.
///
/// Determinism and bounds:
/// - The BFS expands neighbours in ascending edge-`id` order (the order
///   [`Store::outgoing_edges`]/[`Store::incoming_edges`] already
///   guarantee), so the discovered `hops` for each node is stable.
/// - `max_hops` is clamped to [`MAX_REACH_HOPS`]; `limit` to
///   [`MAX_REACH_RESULTS`]. A `max_hops` of `0` (or a missing start
///   node) yields an empty result.
/// - The start node is **never** included in the output.
/// - The returned vec is sorted by `(hops asc, qualified_name asc, id
///   asc)` for a total, reproducible order independent of insertion or
///   traversal timing.
///
/// Reuses the existing edge predicates: neighbour expansion goes through
/// the same `Store` edge accessors that back the `with_outgoing_edge` /
/// `with_incoming_edge` EXISTS filters, so an edge type that matches
/// nothing yields an empty result exactly as those predicates do.
pub fn reachable_within(
    store: &Store,
    start_id: i64,
    direction: ReachDirection,
    edge_type: &str,
    max_hops: usize,
    limit: usize,
) -> Result<Vec<ReachableNode>> {
    use std::collections::{HashMap, VecDeque};

    let max_hops = max_hops.min(MAX_REACH_HOPS);
    let limit = limit.min(MAX_REACH_RESULTS);
    if max_hops == 0 || limit == 0 {
        return Ok(Vec::new());
    }
    // The start node must exist; otherwise there is nothing to expand.
    if store.get_node(start_id)?.is_none() {
        return Ok(Vec::new());
    }

    // node_id -> minimum hop count at which it was first reached. The
    // start node is seeded at hop 0 but never emitted.
    let mut best_hops: HashMap<i64, usize> = HashMap::new();
    best_hops.insert(start_id, 0);
    let mut queue: VecDeque<(i64, usize)> = VecDeque::new();
    queue.push_back((start_id, 0));

    while let Some((node_id, hops)) = queue.pop_front() {
        if hops >= max_hops {
            continue;
        }
        let neighbours = match direction {
            ReachDirection::Outgoing => {
                store.outgoing_edges(node_id, Some(edge_type), REACH_FANOUT)?
            }
            ReachDirection::Incoming => {
                store.incoming_edges(node_id, Some(edge_type), REACH_FANOUT)?
            }
        };
        for e in neighbours {
            let next = match direction {
                ReachDirection::Outgoing => e.target_id,
                ReachDirection::Incoming => e.source_id,
            };
            let next_hops = hops + 1;
            // BFS in edge-id order means the first time we see a node is
            // at its minimum hop count; later sightings never improve it.
            if let std::collections::hash_map::Entry::Vacant(slot) = best_hops.entry(next) {
                slot.insert(next_hops);
                queue.push_back((next, next_hops));
            }
        }
    }

    // Resolve every reached node (excluding the start) to a row.
    let mut out: Vec<ReachableNode> = Vec::new();
    for (node_id, hops) in best_hops.iter() {
        if *node_id == start_id {
            continue;
        }
        if let Some(node) = store.get_node(*node_id)? {
            out.push(ReachableNode {
                hops: *hops,
                node: SearchGraphRow {
                    id: node.id,
                    project: node.project,
                    label: node.label,
                    name: node.name,
                    qualified_name: node.qualified_name,
                    file_path: node.file_path,
                    start_line: node.start_line,
                    end_line: node.end_line,
                },
            });
        }
    }
    // Total, deterministic order: nearest first, then alphabetical,
    // then id (generation) as the final tie-break.
    out.sort_by(|a, b| {
        a.hops
            .cmp(&b.hops)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| a.node.id.cmp(&b.node.id))
    });
    out.truncate(limit);
    Ok(out)
}

/// The direct (1-hop) neighbours of `node_id` over `edge_type` in the
/// given `direction`, as `search_graph`-shaped rows.
///
/// This is the convenience form of a single BFS layer: where
/// [`reachable_within`] does a bounded multi-hop walk, `neighbors`
/// answers the common "who does this node directly call / who directly
/// calls it?" question in one call. `Outgoing` returns edge *targets*
/// (e.g. callees); `Incoming` returns edge *sources* (e.g. callers).
///
/// Determinism and bounds:
/// - Neighbours are gathered in ascending edge-`id` order (the order the
///   `Store` edge accessors guarantee), then the resolved rows are
///   sorted by `(qualified_name asc, id asc)` for a total, reproducible
///   order independent of insertion timing.
/// - Fan-out is capped at [`MAX_REACH_RESULTS`]; the result is further
///   truncated to `limit` (also clamped to [`MAX_REACH_RESULTS`]).
/// - A duplicated edge to the same neighbour yields a single row.
/// - A missing start node yields an empty result.
pub fn neighbors(
    store: &Store,
    node_id: i64,
    edge_type: &str,
    direction: ReachDirection,
    limit: usize,
) -> Result<Vec<SearchGraphRow>> {
    use std::collections::HashSet;

    let limit = limit.min(MAX_REACH_RESULTS);
    if limit == 0 {
        return Ok(Vec::new());
    }
    if store.get_node(node_id)?.is_none() {
        return Ok(Vec::new());
    }

    let edges = match direction {
        ReachDirection::Outgoing => {
            store.outgoing_edges(node_id, Some(edge_type), MAX_REACH_RESULTS)?
        }
        ReachDirection::Incoming => {
            store.incoming_edges(node_id, Some(edge_type), MAX_REACH_RESULTS)?
        }
    };

    let mut seen: HashSet<i64> = HashSet::new();
    let mut out: Vec<SearchGraphRow> = Vec::new();
    for e in edges {
        let nid = match direction {
            ReachDirection::Outgoing => e.target_id,
            ReachDirection::Incoming => e.source_id,
        };
        // A self-loop is not a neighbour; skip it.
        if nid == node_id || !seen.insert(nid) {
            continue;
        }
        if let Some(node) = store.get_node(nid)? {
            out.push(node_to_row(node));
        }
    }
    out.sort_by(|a, b| {
        a.qualified_name
            .cmp(&b.qualified_name)
            .then_with(|| a.id.cmp(&b.id))
    });
    out.truncate(limit);
    Ok(out)
}

/// One edge in a [`Subgraph`], by endpoint node id and type. Kept as
/// plain ids (not resolved rows) so the edge list stays compact; resolve
/// against [`Subgraph::nodes`] when a row is needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgraphEdge {
    pub source_id: i64,
    pub target_id: i64,
    pub edge_type: String,
}

/// The bounded neighbourhood of a symbol: the center plus every node
/// within `max_hops` over the requested edge types, and the edges
/// among the collected nodes. Returned by [`subgraph_around`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subgraph {
    /// The center node the subgraph was built around.
    pub center: SearchGraphRow,
    /// Every node in the neighbourhood, **including** the center,
    /// sorted by `(qualified_name asc, id asc)`.
    pub nodes: Vec<SearchGraphRow>,
    /// Every edge (of the requested types) whose *both* endpoints are in
    /// `nodes`, sorted by `(source_id, target_id, edge_type)`. Induced:
    /// edges leaving the collected node set are excluded.
    pub edges: Vec<SubgraphEdge>,
}

/// Build the bounded subgraph around `center_id`.
///
/// Collects every node within `max_hops` of the center, treating the
/// requested `edge_types` as **undirected** (an edge is followed from
/// either endpoint) so the neighbourhood captures both callers and
/// callees, importers and imported, etc. Then returns the *induced*
/// edge set: every edge of a requested type whose endpoints are both in
/// the collected node set.
///
/// Determinism and bounds:
/// - `max_hops` is clamped to [`MAX_REACH_HOPS`]; the collected node
///   count is clamped to [`MAX_REACH_RESULTS`] (the center always
///   counts toward the budget and is always present if it exists).
/// - BFS expands neighbours in ascending edge-`id` order; the returned
///   `nodes` and `edges` are each sorted by a total key, so the result
///   is byte-stable across runs and independent of insertion timing.
/// - `edge_types` is deduplicated and each type is traversed in the
///   order given; an empty `edge_types` yields just the center with no
///   edges.
/// - A missing center yields an empty subgraph (`nodes` empty).
pub fn subgraph_around(
    store: &Store,
    center_id: i64,
    edge_types: &[&str],
    max_hops: usize,
) -> Result<Option<Subgraph>> {
    use std::collections::{HashSet, VecDeque};

    let max_hops = max_hops.min(MAX_REACH_HOPS);

    let Some(center_node) = store.get_node(center_id)? else {
        return Ok(None);
    };
    let center_row = node_to_row(center_node);

    // Deduplicate the requested edge types, preserving first-seen order.
    let mut types: Vec<&str> = Vec::new();
    for t in edge_types {
        if !types.contains(t) {
            types.push(t);
        }
    }

    // BFS over the undirected union of the requested edge types.
    let mut collected: HashSet<i64> = HashSet::new();
    collected.insert(center_id);
    let mut queue: VecDeque<(i64, usize)> = VecDeque::new();
    queue.push_back((center_id, 0));

    while let Some((nid, hops)) = queue.pop_front() {
        if hops >= max_hops {
            continue;
        }
        for ty in &types {
            // Both directions: undirected neighbourhood.
            let mut adj: Vec<i64> = Vec::new();
            for e in store.outgoing_edges(nid, Some(ty), MAX_REACH_RESULTS)? {
                adj.push(e.target_id);
            }
            for e in store.incoming_edges(nid, Some(ty), MAX_REACH_RESULTS)? {
                adj.push(e.source_id);
            }
            for next in adj {
                if next == nid {
                    continue;
                }
                if collected.len() >= MAX_REACH_RESULTS && !collected.contains(&next) {
                    continue;
                }
                if collected.insert(next) {
                    queue.push_back((next, hops + 1));
                }
            }
        }
    }

    // Resolve collected ids to rows.
    let mut nodes: Vec<SearchGraphRow> = Vec::new();
    for id in &collected {
        if let Some(n) = store.get_node(*id)? {
            nodes.push(node_to_row(n));
        }
    }
    nodes.sort_by(|a, b| {
        a.qualified_name
            .cmp(&b.qualified_name)
            .then_with(|| a.id.cmp(&b.id))
    });

    // Induced edge set: every requested-type edge with both endpoints in
    // the collected set. Collected via outgoing edges from each node to
    // avoid double counting, deduplicated by the unique (src,tgt,type)
    // triple.
    let mut edge_seen: HashSet<(i64, i64, String)> = HashSet::new();
    let mut edges: Vec<SubgraphEdge> = Vec::new();
    for id in &collected {
        for ty in &types {
            for e in store.outgoing_edges(*id, Some(ty), MAX_REACH_RESULTS)? {
                if !collected.contains(&e.target_id) {
                    continue;
                }
                let key = (e.source_id, e.target_id, e.edge_type.clone());
                if edge_seen.insert(key) {
                    edges.push(SubgraphEdge {
                        source_id: e.source_id,
                        target_id: e.target_id,
                        edge_type: e.edge_type,
                    });
                }
            }
        }
    }
    edges.sort_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then_with(|| a.target_id.cmp(&b.target_id))
            .then_with(|| a.edge_type.cmp(&b.edge_type))
    });

    Ok(Some(Subgraph {
        center: center_row,
        nodes,
        edges,
    }))
}

/// Map a `nodes` result row (id, project, label, name, qualified_name,
/// file_path, start_line, end_line) onto a [`SearchGraphRow`]. Shared by
/// every `search_graph` fetch path so the column order stays in one place.
fn row_to_search_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SearchGraphRow> {
    Ok(SearchGraphRow {
        id: row.get(0)?,
        project: row.get(1)?,
        label: row.get(2)?,
        name: row.get(3)?,
        qualified_name: row.get(4)?,
        file_path: row.get(5)?,
        start_line: row.get(6)?,
        end_line: row.get(7)?,
    })
}

/// Map a store [`grepplus_store::Node`] onto a [`SearchGraphRow`] (the
/// shape the rest of the query surface speaks).
fn node_to_row(node: grepplus_store::Node) -> SearchGraphRow {
    SearchGraphRow {
        id: node.id,
        project: node.project,
        label: node.label,
        name: node.name,
        qualified_name: node.qualified_name,
        file_path: node.file_path,
        start_line: node.start_line,
        end_line: node.end_line,
    }
}

/// Escape SQL `LIKE` wildcards (`%`, `_`) and the escape char itself
/// in a user-supplied substring so a substring/prefix filter matches
/// the literal text rather than treating `%`/`_` as wildcards. The
/// caller must pair the bound value with `ESCAPE '\'`.
fn like_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

/// One row returned by `search_graph`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchGraphRow {
    pub id: i64,
    pub project: String,
    pub label: String,
    pub name: String,
    pub qualified_name: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
}

/// Default maximum rows returned by `search_graph` when the caller
/// does not specify one. The upstream's `search_graph` tool returns
/// up to 50 by default; we use a higher default because callers in
/// Phase 5 freshness checks iterate the whole graph.
pub const DEFAULT_LIMIT: usize = 10_000;

/// Query spec for `search_graph`.
#[derive(Debug, Clone)]
pub struct GraphQuery {
    pub project: Option<String>,
    pub label: Option<String>,
    pub name: Option<String>,
    pub qname_contains: Option<String>,
    pub file_path_contains: Option<String>,
    /// Substring match on `nodes.name`.
    pub name_contains: Option<String>,
    /// Prefix match on `nodes.name`.
    pub name_prefix: Option<String>,
    /// Prefix match on `nodes.file_path`.
    pub file_path_prefix: Option<String>,
    /// Exact match on `nodes.file_path` (used by the label+file composite).
    pub file_path_exact: Option<String>,
    /// If set, keep only nodes with an outgoing edge of this type.
    pub has_outgoing_edge: Option<String>,
    /// If set, keep only nodes with an incoming edge of this type.
    pub has_incoming_edge: Option<String>,
    /// Ordering/ranking applied to the result set. Defaults to
    /// [`GraphOrder::QualifiedName`] (the historic behaviour).
    pub order_by: GraphOrder,
    /// Edge type that the degree-based orders ([`GraphOrder::OutDegreeDesc`]
    /// / [`GraphOrder::InDegreeDesc`]) count. `None` counts edges of every
    /// type. Ignored by the column-based orders.
    pub order_edge_type: Option<String>,
    pub limit: usize,
}

impl Default for GraphQuery {
    fn default() -> Self {
        Self {
            project: None,
            label: None,
            name: None,
            qname_contains: None,
            file_path_contains: None,
            name_contains: None,
            name_prefix: None,
            file_path_prefix: None,
            file_path_exact: None,
            has_outgoing_edge: None,
            has_incoming_edge: None,
            order_by: GraphOrder::default(),
            order_edge_type: None,
            limit: DEFAULT_LIMIT,
        }
    }
}

impl GraphQuery {
    /// Set the maximum number of rows to return.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
    /// Restrict the query to one project (R-025).
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set the result ordering/ranking. Back-compatible: the default is
    /// [`GraphOrder::QualifiedName`], so omitting this leaves the historic
    /// order unchanged. Every order is total and deterministic.
    pub fn with_order(mut self, order: GraphOrder) -> Self {
        self.order_by = order;
        self
    }

    /// Scope the degree-based orders ([`GraphOrder::OutDegreeDesc`] /
    /// [`GraphOrder::InDegreeDesc`]) to a single edge type. No effect on
    /// the column-based orders. When unset, every edge type counts toward
    /// the degree.
    pub fn with_order_edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.order_edge_type = Some(edge_type.into());
        self
    }
}

/// Find nodes by label **and** file in one call: the common "what symbols
/// of kind X live in file Y?" composite.
///
/// `file` matches against `nodes.file_path`. When `exact_file` is true the
/// path must equal `file` exactly; otherwise `file` is treated as a
/// substring (the same literal-escaped `LIKE` the
/// [`GraphQuery::with_file_path_contains`] filter uses), so a directory
/// fragment or partial path also matches. `project` scopes the search
/// (R-025); pass `None` only for a single-project store.
///
/// This is sugar over [`search_graph`] — it builds the equivalent
/// `GraphQuery` and runs it — so it inherits the same determinism,
/// ordering (sorted by `file_path` then source line, grouping a file's
/// symbols in source order), and limit behaviour. Back-compatible:
/// `search_graph` and every existing filter are untouched.
pub fn find_by_label_and_file(
    store: &Store,
    project: Option<&str>,
    label: &str,
    file: &str,
    exact_file: bool,
    limit: usize,
) -> Result<Vec<SearchGraphRow>> {
    let mut q = GraphQuery::any()
        .with_label(label)
        .with_order(GraphOrder::File)
        .with_limit(limit);
    if let Some(p) = project {
        q = q.with_project(p);
    }
    if exact_file {
        // An exact file match: reuse the `name` exact path by binding
        // file_path directly. The existing struct has no exact-file
        // field, so express it as a prefix+contains pair would be lossy;
        // instead match exactly via a dedicated branch below.
        q.file_path_exact = Some(file.to_string());
    } else {
        q = q.with_file_path_contains(file);
    }
    search_graph(store, &q)
}

/// All symbol definitions in one file, in source order.
///
/// Returns every node whose `file_path` equals `file` exactly, ordered by
/// `(start_line, end_line, qualified_name, id)` — i.e. top-of-file first,
/// an enclosing definition before the definitions it contains (a struct
/// before its methods, since the struct opens on an earlier or equal line
/// and closes later), and a total tie-break on `qualified_name` then the
/// monotonic `id`. `project` scopes the search (R-025); pass `None` only
/// for a single-project store.
///
/// This is the "what is defined in this file?" companion to
/// [`find_by_label_and_file`]: where that filters by a *label*, this returns
/// every kind. It is sugar over [`search_graph`] (exact file match +
/// [`GraphOrder::File`]) and then applies the `end_line`/`qualified_name`
/// tie-breaks in Rust, so it inherits the same determinism and limit
/// behaviour. Back-compatible: every existing API is untouched.
pub fn symbols_in_file(
    store: &Store,
    project: Option<&str>,
    file: &str,
    limit: usize,
) -> Result<Vec<SearchGraphRow>> {
    let mut q = GraphQuery::any()
        .with_order(GraphOrder::File)
        .with_limit(DEFAULT_LIMIT);
    if let Some(p) = project {
        q = q.with_project(p);
    }
    q.file_path_exact = Some(file.to_string());
    let mut rows = search_graph(store, &q)?;
    // `GraphOrder::File` already sorts by (file_path, start_line, id). Within
    // a single file that is (start_line, id); refine the tie-break so an
    // enclosing definition (same start_line, larger end_line) sorts before
    // the things it contains, then fall back to qualified_name and id for a
    // total, store-independent order.
    rows.sort_by(|a, b| {
        a.start_line
            .cmp(&b.start_line)
            .then_with(|| b.end_line.cmp(&a.end_line))
            .then_with(|| a.qualified_name.cmp(&b.qualified_name))
            .then_with(|| a.id.cmp(&b.id))
    });
    rows.truncate(limit);
    Ok(rows)
}

/// The definition of the symbol *at* a `file:line` location: the nearest
/// enclosing definition whose `[start_line, end_line]` span contains
/// `line`.
///
/// Models the editor "go to definition of the symbol under the cursor"
/// gesture against the indexed graph. Among every node in `file` whose span
/// brackets `line` (`start_line <= line <= end_line`), the **innermost** one
/// wins — the definition with the largest `start_line` (and, on a tie, the
/// smallest `end_line`), i.e. the tightest span around the location. So a
/// `line` inside a method body resolves to the method, not the enclosing
/// struct/impl; a `line` between two top-level functions resolves to the one
/// it falls inside, or `None` if it falls inside neither (e.g. a blank line
/// or a top-level import region with no covering definition).
///
/// `project` scopes the search (R-025). `line` is 1-based, matching the
/// stored `start_line`/`end_line` convention.
///
/// Determinism: the candidate set comes from [`symbols_in_file`] (a total
/// order); the innermost pick uses `(start_line desc, end_line asc,
/// qualified_name asc, id asc)`, so identical inputs always resolve to the
/// same node. Additive: built on existing query surface, no API changed.
pub fn definition_at(
    store: &Store,
    project: Option<&str>,
    file: &str,
    line: i64,
) -> Result<Option<SearchGraphRow>> {
    let rows = symbols_in_file(store, project, file, DEFAULT_LIMIT)?;
    let best = rows
        .into_iter()
        .filter(|r| r.start_line <= line && line <= r.end_line)
        .min_by(|a, b| {
            // Innermost = largest start_line, then smallest end_line
            // (tightest span), then a total tie-break.
            b.start_line
                .cmp(&a.start_line)
                .then_with(|| a.end_line.cmp(&b.end_line))
                .then_with(|| a.qualified_name.cmp(&b.qualified_name))
                .then_with(|| a.id.cmp(&b.id))
        });
    Ok(best)
}

/// A node ranked by its edge degree, returned by [`most_connected`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegreeRanked {
    /// The degree (number of matching edges) used to rank this node.
    pub degree: usize,
    /// The node row (same shape as [`search_graph`] results).
    pub node: SearchGraphRow,
}

/// Direction of the degree counted by [`most_connected`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegreeKind {
    /// Count outgoing edges (how many things this node points at).
    Out,
    /// Count incoming edges (how many things point at this node).
    In,
    /// Count both directions (total connectivity). A self-loop counts
    /// once in each direction, i.e. twice.
    Total,
}

/// Top-`n` most-connected nodes by edge degree — a simple "hub"
/// query ported from the upstream's graph-analysis surface.
///
/// Counts edges of type `edge_type` (or, when `edge_type` is `None`,
/// edges of every type) in the requested `kind` (out / in / total) for
/// every node matching the optional `project` scope, then returns the
/// `n` nodes with the highest degree.
///
/// Determinism and bounds:
/// - Per-node fan-out is capped at [`MAX_REACH_RESULTS`] edges in each
///   direction, bounding the work at any single node.
/// - The result is sorted by `(degree desc, qualified_name asc, id asc)`
///   — a total order — so it is byte-stable across runs regardless of
///   insertion or row-scan timing, and the leaf tie-break is the node
///   `id` (a monotonic generation proxy).
/// - `n` is the post-sort truncation; nodes with degree `0` are kept
///   (they may legitimately be the answer in a sparse graph) but always
///   sort last among the candidate set.
///
/// This is sugar over the same `Store` edge accessors that back the
/// degree-ordered [`search_graph`] path; it differs by *returning the
/// degree* alongside each row so a caller can show "called by N places".
pub fn most_connected(
    store: &Store,
    project: Option<&str>,
    edge_type: Option<&str>,
    kind: DegreeKind,
    n: usize,
) -> Result<Vec<DegreeRanked>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut q = GraphQuery::any().with_limit(DEFAULT_LIMIT);
    if let Some(p) = project {
        q = q.with_project(p);
    }
    let rows = search_graph(store, &q)?;

    let mut ranked: Vec<DegreeRanked> = Vec::with_capacity(rows.len());
    for row in rows {
        let out = match kind {
            DegreeKind::Out | DegreeKind::Total => store
                .outgoing_edges(row.id, edge_type, MAX_REACH_RESULTS)?
                .len(),
            DegreeKind::In => 0,
        };
        let inc = match kind {
            DegreeKind::In | DegreeKind::Total => store
                .incoming_edges(row.id, edge_type, MAX_REACH_RESULTS)?
                .len(),
            DegreeKind::Out => 0,
        };
        ranked.push(DegreeRanked {
            degree: out + inc,
            node: row,
        });
    }
    ranked.sort_by(|a, b| {
        b.degree
            .cmp(&a.degree)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| a.node.id.cmp(&b.node.id))
    });
    ranked.truncate(n);
    Ok(ranked)
}

/// Rank nodes by incoming `edge_type` degree within one project.
///
/// Unlike [`most_connected`], this uses a single SQL aggregation over the
/// edge table, emits only positive-degree nodes, and has a companion exact
/// count helper. It is the production path behind `grepplus fan-in`.
pub fn fan_in(
    store: &Store,
    project: &str,
    edge_type: &str,
    limit: usize,
) -> Result<Vec<DegreeRanked>> {
    fan_degree(store, project, edge_type, "target_id", limit)
}

/// Rank nodes by outgoing `edge_type` degree within one project.
///
/// This is the production path behind `grepplus fan-out`.
pub fn fan_out(
    store: &Store,
    project: &str,
    edge_type: &str,
    limit: usize,
) -> Result<Vec<DegreeRanked>> {
    fan_degree(store, project, edge_type, "source_id", limit)
}

/// Count positive incoming-degree nodes for `edge_type` without applying a
/// display limit.
pub fn count_fan_in(store: &Store, project: &str, edge_type: &str) -> Result<usize> {
    count_fan_degree(store, project, edge_type, "target_id")
}

/// Count positive outgoing-degree nodes for `edge_type` without applying a
/// display limit.
pub fn count_fan_out(store: &Store, project: &str, edge_type: &str) -> Result<usize> {
    count_fan_degree(store, project, edge_type, "source_id")
}

fn fan_degree(
    store: &Store,
    project: &str,
    edge_type: &str,
    endpoint_col: &str,
    limit: usize,
) -> Result<Vec<DegreeRanked>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let sql = format!(
        "SELECT n.id, n.project, n.label, n.name, n.qualified_name, n.file_path,
                n.start_line, n.end_line, COUNT(e.id) AS degree
         FROM edges e
         JOIN nodes n ON n.id = e.{endpoint_col} AND n.project = e.project
         WHERE e.project = ?1 AND e.edge_type = ?2
         GROUP BY n.id
         ORDER BY degree DESC, n.qualified_name ASC, n.id ASC
         LIMIT ?3"
    );
    let limit = limit.min(MAX_REACH_RESULTS) as i64;
    let mut stmt = store
        .conn()
        .prepare(&sql)
        .map_err(grepplus_store::Error::Sqlite)?;
    let rows = stmt
        .query_map(rusqlite::params![project, edge_type, limit], |row| {
            let degree: i64 = row.get(8)?;
            Ok(DegreeRanked {
                degree: degree.max(0) as usize,
                node: row_to_search_row(row)?,
            })
        })
        .map_err(grepplus_store::Error::Sqlite)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(grepplus_store::Error::Sqlite)?;
    Ok(rows)
}

fn count_fan_degree(
    store: &Store,
    project: &str,
    edge_type: &str,
    endpoint_col: &str,
) -> Result<usize> {
    let sql = format!(
        "SELECT COUNT(*) FROM (
             SELECT n.id
             FROM edges e
             JOIN nodes n ON n.id = e.{endpoint_col} AND n.project = e.project
             WHERE e.project = ?1 AND e.edge_type = ?2
             GROUP BY n.id
         )"
    );
    let count: i64 = store
        .conn()
        .query_row(&sql, rusqlite::params![project, edge_type], |row| {
            row.get(0)
        })
        .map_err(grepplus_store::Error::Sqlite)?;
    Ok(count.max(0) as usize)
}

/// A concrete path returned by [`path_query`]: the ordered list of node
/// ids from the start to the goal (inclusive of both endpoints).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPath {
    /// Node ids in walk order: `nodes[0] == from`, `nodes.last() == to`.
    /// A direct edge yields a two-element path; `from == to` yields a
    /// single-element path.
    pub nodes: Vec<i64>,
    /// The resolved rows for each id in `nodes`, in the same order, so a
    /// caller can render the path without a second lookup.
    pub rows: Vec<SearchGraphRow>,
    /// Number of edges traversed (`nodes.len() - 1`); `0` when
    /// `from == to`.
    pub hops: usize,
}

/// Find a shortest path of `edge_type` edges from `from_id` to `to_id`
/// within `max_hops`, following edges in `direction`. Returns the path
/// (start..=goal) or `None` if no such path exists within the bound.
///
/// This is the path-existence counterpart to [`reachable_within`]: where
/// `reachable_within` returns the *set* of reachable nodes,
/// `path_query` answers "is there a path A → B, and what is it?" and
/// returns a single concrete shortest path.
///
/// Determinism and bounds:
/// - BFS expands neighbours in ascending edge-`id` order (the order the
///   `Store` edge accessors guarantee) and records the *first* predecessor
///   that reaches each node. Because BFS reaches every node at its minimum
///   hop count and ties are broken by edge id, the returned path is the
///   unique shortest path under that total order — byte-stable across
///   runs.
/// - `max_hops` is clamped to [`MAX_REACH_HOPS`]; per-node fan-out is
///   capped at [`REACH_FANOUT`].
/// - `from == to` returns a length-zero path (`[from]`) when the node
///   exists, `None` otherwise.
/// - A missing `from` or `to` node yields `None`.
pub fn path_query(
    store: &Store,
    from_id: i64,
    to_id: i64,
    direction: ReachDirection,
    edge_type: &str,
    max_hops: usize,
) -> Result<Option<GraphPath>> {
    use std::collections::{HashMap, VecDeque};

    let max_hops = max_hops.min(MAX_REACH_HOPS);

    // Both endpoints must exist.
    if store.get_node(from_id)?.is_none() || store.get_node(to_id)?.is_none() {
        return Ok(None);
    }

    // Trivial path: start == goal.
    if from_id == to_id {
        let row = node_to_row(store.get_node(from_id)?.expect("checked above"));
        return Ok(Some(GraphPath {
            nodes: vec![from_id],
            rows: vec![row],
            hops: 0,
        }));
    }
    if max_hops == 0 {
        return Ok(None);
    }

    // predecessor[next] = the node we arrived from when first reaching
    // `next`. The start has no predecessor. BFS in edge-id order means the
    // first time a node is seen is at its minimum hop count via the
    // lowest-edge-id route, so the reconstructed path is the deterministic
    // shortest one.
    let mut pred: HashMap<i64, i64> = HashMap::new();
    let mut seen_hops: HashMap<i64, usize> = HashMap::new();
    seen_hops.insert(from_id, 0);
    let mut queue: VecDeque<(i64, usize)> = VecDeque::new();
    queue.push_back((from_id, 0));

    while let Some((node_id, hops)) = queue.pop_front() {
        if hops >= max_hops {
            continue;
        }
        let neighbours = match direction {
            ReachDirection::Outgoing => {
                store.outgoing_edges(node_id, Some(edge_type), REACH_FANOUT)?
            }
            ReachDirection::Incoming => {
                store.incoming_edges(node_id, Some(edge_type), REACH_FANOUT)?
            }
        };
        for e in neighbours {
            let next = match direction {
                ReachDirection::Outgoing => e.target_id,
                ReachDirection::Incoming => e.source_id,
            };
            if let std::collections::hash_map::Entry::Vacant(slot) = seen_hops.entry(next) {
                slot.insert(hops + 1);
                pred.insert(next, node_id);
                if next == to_id {
                    // Reconstruct: walk predecessors back to the start.
                    let mut chain = vec![to_id];
                    let mut cur = to_id;
                    while let Some(&p) = pred.get(&cur) {
                        chain.push(p);
                        cur = p;
                        if cur == from_id {
                            break;
                        }
                    }
                    chain.reverse();
                    let mut rows = Vec::with_capacity(chain.len());
                    for id in &chain {
                        if let Some(node) = store.get_node(*id)? {
                            rows.push(node_to_row(node));
                        }
                    }
                    let hops = chain.len() - 1;
                    return Ok(Some(GraphPath {
                        nodes: chain,
                        rows,
                        hops,
                    }));
                }
                queue.push_back((next, hops + 1));
            }
        }
    }
    Ok(None)
}

/// One node in an [`impact_radius`] result: a reached node, the minimum
/// number of `CALLS` hops it sits from the impacted symbol, and the
/// node's resolved row.
///
/// Distinct from [`ReachableNode`] only in name/intent — it carries the
/// same `(hops, node)` payload but is the type the *impact / blast-radius*
/// surface speaks, so a caller reading "what breaks if I change S?" sees a
/// purpose-named result rather than a generic reachability one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactNode {
    /// Minimum hops from the impacted (changed) symbol over the requested
    /// edge type (`1` for a direct caller/callee; the source is never
    /// emitted).
    pub hops: usize,
    /// The reached node's row (same shape as [`search_graph`] results).
    pub node: SearchGraphRow,
}

/// Impact / blast-radius query: every node transitively reachable from
/// `source_id` over `edge_type` within `max_hops`, deduplicated, each
/// tagged with the minimum hop depth at which it was reached.
///
/// This is the analyst-facing framing of [`reachable_within`] for the
/// common "what is affected if I change S?" question. Direction encodes
/// the two flavours of blast radius:
/// - [`ReachDirection::Incoming`] over `CALLS` → the *callers* that
///   transitively depend on `source_id` (what might break if S changes).
/// - [`ReachDirection::Outgoing`] over `CALLS` → everything S transitively
///   *calls* (its dependency cone).
///
/// Determinism and bounds are inherited verbatim from
/// [`reachable_within`]: BFS in ascending edge-id order yields each node
/// at its minimum hop count; `max_hops` is clamped to [`MAX_REACH_HOPS`]
/// and `limit` to [`MAX_REACH_RESULTS`]; the source node is never in the
/// output; and the result is sorted by `(hops asc, qualified_name asc, id
/// asc)` for a total, reproducible order. A missing source, a `max_hops`
/// of `0`, or an `edge_type` matching nothing yields an empty result.
///
/// Additive: this wraps the existing `reachable_within` and re-shapes its
/// rows into [`ImpactNode`]; no existing API changes.
pub fn impact_radius(
    store: &Store,
    source_id: i64,
    direction: ReachDirection,
    edge_type: &str,
    max_hops: usize,
    limit: usize,
) -> Result<Vec<ImpactNode>> {
    let reached = reachable_within(store, source_id, direction, edge_type, max_hops, limit)?;
    Ok(reached
        .into_iter()
        .map(|r| ImpactNode {
            hops: r.hops,
            node: r.node,
        })
        .collect())
}

/// One co-change candidate returned by [`co_change_candidates`]: a node
/// that shares one or more edge-neighbours with the seed symbol, ranked by
/// how many neighbours they share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoChangeCandidate {
    /// Number of distinct neighbours this candidate shares with the seed
    /// (the "co-change strength"). Always `>= 1`.
    pub shared: usize,
    /// The candidate node's row (same shape as [`search_graph`] results).
    pub node: SearchGraphRow,
}

/// "Co-change candidates" query: the nodes that share at least one
/// edge-neighbour with `seed_id` over `edge_type`, ranked by the number of
/// shared neighbours.
///
/// This is a structural proxy for the classic version-control "files that
/// change together" heuristic, computed from the static call graph instead
/// of history: two symbols that both call (or are both called by) the same
/// set of functions sit in the same neighbourhood and tend to evolve
/// together. Concretely, for the seed's neighbour set `N` (its callees when
/// `direction == Outgoing`, its callers when `Incoming`), every *other*
/// node that also neighbours some `m` in `N` in the **opposite** orientation
/// is a candidate, scored by the number of shared neighbours.
///
/// Worked example (`direction == Outgoing`, `edge_type == "CALLS"`): the
/// seed S calls `m`. Any other node X that *also* calls `m` shares the
/// callee `m` with S, so X is a co-change candidate with `shared >= 1`.
/// The more callees S and X have in common, the higher X ranks.
///
/// Determinism and bounds:
/// - Neighbour expansion uses the same id-ordered `Store` edge accessors
///   the rest of this module relies on; per-step fan-out is capped at
///   [`REACH_FANOUT`].
/// - The seed itself is never a candidate; a node is counted once per
///   distinct shared neighbour (a duplicated edge does not inflate the
///   score).
/// - The result is sorted by `(shared desc, qualified_name asc, id asc)`
///   — a total order — then truncated to `limit` (clamped to
///   [`MAX_REACH_RESULTS`]).
/// - A missing seed, a `limit` of `0`, or an `edge_type` matching nothing
///   yields an empty result.
///
/// Additive: a new function plus two result types; no existing API
/// changes.
pub fn co_change_candidates(
    store: &Store,
    seed_id: i64,
    direction: ReachDirection,
    edge_type: &str,
    limit: usize,
) -> Result<Vec<CoChangeCandidate>> {
    use std::collections::{HashMap, HashSet};

    let limit = limit.min(MAX_REACH_RESULTS);
    if limit == 0 {
        return Ok(Vec::new());
    }
    if store.get_node(seed_id)?.is_none() {
        return Ok(Vec::new());
    }

    // The seed's direct neighbour set N over `edge_type` in `direction`.
    // For Outgoing these are callees; for Incoming, callers.
    let seed_neighbours = direct_neighbour_ids(store, seed_id, direction, edge_type)?;

    // For each shared neighbour m in N, find every *other* node that also
    // neighbours m in the opposite orientation, and count how many distinct
    // m's it shares with the seed. Opposite orientation: if the seed
    // reaches m as a callee (Outgoing), a co-caller of m is found by
    // looking at m's *incoming* edges (who else calls m).
    let opposite = match direction {
        ReachDirection::Outgoing => ReachDirection::Incoming,
        ReachDirection::Incoming => ReachDirection::Outgoing,
    };

    let mut shared_counts: HashMap<i64, usize> = HashMap::new();
    for m in &seed_neighbours {
        let co = direct_neighbour_ids(store, *m, opposite, edge_type)?;
        // De-dup per neighbour m so a node that reaches m via several
        // edges still only earns one increment for this m.
        let mut counted_for_m: HashSet<i64> = HashSet::new();
        for other in co {
            if other == seed_id {
                continue; // the seed is not its own candidate
            }
            if counted_for_m.insert(other) {
                *shared_counts.entry(other).or_insert(0) += 1;
            }
        }
    }

    let mut out: Vec<CoChangeCandidate> = Vec::new();
    for (node_id, shared) in shared_counts {
        if let Some(node) = store.get_node(node_id)? {
            out.push(CoChangeCandidate {
                shared,
                node: node_to_row(node),
            });
        }
    }
    // Strongest co-change first, then a total tie-break.
    out.sort_by(|a, b| {
        b.shared
            .cmp(&a.shared)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| a.node.id.cmp(&b.node.id))
    });
    out.truncate(limit);
    Ok(out)
}

/// The deduplicated set of direct (1-hop) neighbour ids of `node_id` over
/// `edge_type` in `direction`, in ascending edge-id discovery order with
/// self-loops excluded. Shared helper for the co-change query; bounded by
/// [`REACH_FANOUT`] per call.
fn direct_neighbour_ids(
    store: &Store,
    node_id: i64,
    direction: ReachDirection,
    edge_type: &str,
) -> Result<Vec<i64>> {
    use std::collections::HashSet;
    let edges = match direction {
        ReachDirection::Outgoing => store.outgoing_edges(node_id, Some(edge_type), REACH_FANOUT)?,
        ReachDirection::Incoming => store.incoming_edges(node_id, Some(edge_type), REACH_FANOUT)?,
    };
    let mut seen: HashSet<i64> = HashSet::new();
    let mut out: Vec<i64> = Vec::new();
    for e in edges {
        let nid = match direction {
            ReachDirection::Outgoing => e.target_id,
            ReachDirection::Incoming => e.source_id,
        };
        if nid == node_id {
            continue;
        }
        if seen.insert(nid) {
            out.push(nid);
        }
    }
    Ok(out)
}

/// A connected dependency cluster returned by [`dependency_cluster`]: the
/// set of nodes mutually reachable from a seed by treating `IMPORTS` edges
/// as **undirected**, i.e. the connected component of the import graph the
/// seed belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyCluster {
    /// Every node in the component (including the seed), sorted by
    /// `(qualified_name asc, id asc)` for a total, reproducible order.
    pub nodes: Vec<SearchGraphRow>,
    /// True when the traversal hit the [`MAX_REACH_RESULTS`] node budget
    /// (or the caller-supplied `limit`) and stopped early, so the cluster
    /// may be incomplete. False when the whole component fit.
    pub truncated: bool,
}

/// Dependency-cluster query: the connected component of the **`IMPORTS`**
/// graph that `seed_id` belongs to, treating import edges as undirected so
/// the cluster captures both "what the seed imports" and "what imports the
/// seed", transitively.
///
/// This answers "which symbols form one import-coupled unit?" — a module
/// and everything it pulls in or is pulled into, bounded so a pathological
/// graph cannot blow up. It is the `IMPORTS`-specific, *whole-component*
/// counterpart to [`subgraph_around`] (which is hop-bounded and multi-edge):
/// here there is no hop limit, the traversal runs until the component is
/// exhausted or the node budget is hit.
///
/// Determinism and bounds:
/// - BFS expands neighbours in ascending edge-`id` order (the order the
///   `Store` edge accessors guarantee); the returned `nodes` are sorted by
///   `(qualified_name asc, id asc)`, so the result is byte-stable across
///   runs and independent of insertion or traversal timing.
/// - The collected node count is clamped to `limit` (itself clamped to
///   [`MAX_REACH_RESULTS`]); when the clamp stops the walk early,
///   `truncated` is `true`. The seed always counts toward the budget and is
///   always present when it exists.
/// - Per-node fan-out is capped at [`MAX_REACH_RESULTS`] edges in each
///   direction.
/// - A missing seed yields `None`.
///
/// Additive: a new function and a new result type; no existing API changes.
pub fn dependency_cluster(
    store: &Store,
    seed_id: i64,
    limit: usize,
) -> Result<Option<DependencyCluster>> {
    use std::collections::{HashSet, VecDeque};

    let limit = limit.clamp(1, MAX_REACH_RESULTS);

    if store.get_node(seed_id)?.is_none() {
        return Ok(None);
    }

    let mut collected: HashSet<i64> = HashSet::new();
    collected.insert(seed_id);
    let mut queue: VecDeque<i64> = VecDeque::new();
    queue.push_back(seed_id);
    let mut truncated = false;

    while let Some(nid) = queue.pop_front() {
        // Undirected union of IMPORTS in both directions.
        let mut adj: Vec<i64> = Vec::new();
        for e in store.outgoing_edges(nid, Some("IMPORTS"), MAX_REACH_RESULTS)? {
            adj.push(e.target_id);
        }
        for e in store.incoming_edges(nid, Some("IMPORTS"), MAX_REACH_RESULTS)? {
            adj.push(e.source_id);
        }
        for next in adj {
            if next == nid || collected.contains(&next) {
                continue;
            }
            if collected.len() >= limit {
                // Budget exhausted: the component is larger than we will
                // report. Flag it and stop adding new frontier nodes.
                truncated = true;
                continue;
            }
            collected.insert(next);
            queue.push_back(next);
        }
    }

    let mut nodes: Vec<SearchGraphRow> = Vec::with_capacity(collected.len());
    for id in &collected {
        if let Some(n) = store.get_node(*id)? {
            nodes.push(node_to_row(n));
        }
    }
    nodes.sort_by(|a, b| {
        a.qualified_name
            .cmp(&b.qualified_name)
            .then_with(|| a.id.cmp(&b.id))
    });

    Ok(Some(DependencyCluster { nodes, truncated }))
}

/// One detected cycle returned by [`cycles`]: a closed walk over `CALLS`
/// edges, expressed as the ordered list of node ids on the cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    /// Node ids in cycle order. The first node is the lexicographically
    /// smallest `id` on the cycle (the canonical rotation), and the list
    /// does **not** repeat that node at the end — a cycle of length `k`
    /// has exactly `k` entries, and the implied closing edge runs from
    /// `nodes.last()` back to `nodes[0]`.
    pub nodes: Vec<i64>,
}

impl Cycle {
    /// The cycle length (number of edges, equal to the number of nodes).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the cycle is empty (never true for a value produced by
    /// [`cycles`], which only emits cycles of length `>= 1`).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Hard cap on the cycle length [`cycles`] will search for. Requests above
/// this are clamped, bounding the depth-first search on dense graphs.
pub const MAX_CYCLE_LEN: usize = 16;

/// Cycle detector: report every distinct simple cycle over `CALLS` edges of
/// length at most `max_len`, scoped to `project`.
///
/// A *simple* cycle visits each node at most once before closing back to its
/// start. Self-loops (a node that `CALLS` itself) are reported as length-1
/// cycles. Each cycle is canonicalised — rotated so its smallest node `id`
/// comes first — and deduplicated, so the same cyclic structure discovered
/// from different entry points is reported once.
///
/// This is the static-analysis "find recursion / call cycles" query: a
/// non-empty result means the call graph has back-edges, useful for spotting
/// unbounded recursion, cyclic module coupling, or layering violations.
///
/// Determinism and bounds:
/// - Candidate start nodes come from [`search_graph`] in its total
///   `(qualified_name, id)` order; neighbour expansion uses the id-ordered
///   `Store` edge accessors. Each cycle is canonicalised to its
///   smallest-id rotation, the cycle set is deduplicated, and the returned
///   vec is sorted by `(len asc, nodes lexicographically)` — so the result
///   is byte-stable across runs.
/// - `max_len` is clamped to [`MAX_CYCLE_LEN`]; a `max_len` of `0` yields no
///   cycles. Per-node fan-out is capped at [`MAX_REACH_RESULTS`].
/// - To bound output on adversarial input the number of distinct cycles is
///   capped at `limit` (after sorting, so the smallest cycles survive the
///   truncation).
///
/// Additive: a new function plus a result type and two constants; no
/// existing API changes.
pub fn cycles(
    store: &Store,
    project: Option<&str>,
    max_len: usize,
    limit: usize,
) -> Result<Vec<Cycle>> {
    use std::collections::HashSet;

    let max_len = max_len.min(MAX_CYCLE_LEN);
    if max_len == 0 || limit == 0 {
        return Ok(Vec::new());
    }

    // Candidate node set, in a total deterministic order.
    let mut q = GraphQuery::any().with_limit(DEFAULT_LIMIT);
    if let Some(p) = project {
        q = q.with_project(p);
    }
    let rows = search_graph(store, &q)?;
    // The set of node ids in scope; an edge whose target is outside this set
    // (e.g. a cross-project edge) is ignored so cycles stay within `project`.
    let in_scope: HashSet<i64> = rows.iter().map(|r| r.id).collect();

    // Cache each node's outgoing CALLS targets that are in scope, in
    // ascending edge-id order (the accessor's order), so the DFS is both
    // cheap and deterministic.
    use std::collections::HashMap;
    let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();
    for r in &rows {
        let mut targets: Vec<i64> = Vec::new();
        for e in store.outgoing_edges(r.id, Some("CALLS"), MAX_REACH_RESULTS)? {
            if in_scope.contains(&e.target_id) {
                targets.push(e.target_id);
            }
        }
        adj.insert(r.id, targets);
    }

    // Collect canonicalised cycles in a set to dedup. Canonical form: rotate
    // so the smallest id is first, keeping cyclic order.
    let mut found: HashSet<Vec<i64>> = HashSet::new();

    // DFS from each node, only extending the path with neighbours whose id is
    // >= the path's start id. This restriction guarantees each simple cycle is
    // discovered exactly once from its smallest-id member (and lets us canonical
    // -ise trivially), and prunes the search space without missing any cycle.
    for start_row in &rows {
        let start = start_row.id;
        let mut path: Vec<i64> = vec![start];
        let mut on_path: HashSet<i64> = HashSet::new();
        on_path.insert(start);
        dfs_cycles(
            start,
            start,
            &adj,
            &mut path,
            &mut on_path,
            max_len,
            &mut found,
        );
    }

    let mut out: Vec<Cycle> = found.into_iter().map(|nodes| Cycle { nodes }).collect();
    out.sort_by(|a, b| {
        a.nodes
            .len()
            .cmp(&b.nodes.len())
            .then_with(|| a.nodes.cmp(&b.nodes))
    });
    out.truncate(limit);
    Ok(out)
}

/// Depth-first walk that records every simple `CALLS` cycle that returns to
/// `start`, restricted to nodes with `id >= start` so each cycle is found
/// once from its smallest member. `path` is the current walk (starting at
/// `start`); `on_path` is its node set for O(1) revisit checks.
#[allow(clippy::too_many_arguments)]
fn dfs_cycles(
    start: i64,
    node: i64,
    adj: &std::collections::HashMap<i64, Vec<i64>>,
    path: &mut Vec<i64>,
    on_path: &mut std::collections::HashSet<i64>,
    max_len: usize,
    found: &mut std::collections::HashSet<Vec<i64>>,
) {
    if let Some(targets) = adj.get(&node) {
        for &next in targets {
            if next == start {
                // Closed the cycle back to the start. path is already the
                // canonical (smallest-first) rotation because every other
                // node on it has id > start.
                if path.len() <= max_len {
                    found.insert(path.clone());
                }
                continue;
            }
            // Only extend through nodes strictly greater than the start (so
            // `start` stays the minimum) and not already on the path (simple
            // cycle), and respect the length bound.
            if next < start || on_path.contains(&next) || path.len() >= max_len {
                continue;
            }
            path.push(next);
            on_path.insert(next);
            dfs_cycles(start, next, adj, path, on_path, max_len, found);
            on_path.remove(&next);
            path.pop();
        }
    }
}

/// The edge types that constitute a *reference* to a symbol in the unified
/// [`find_references`] query and the [`unused_symbols`] zero-reference test.
///
/// These are exactly the incoming relations that mean "something elsewhere
/// depends on this definition": it is `CALLS`ed, referenced (the unified
/// C-reference `USAGE` edge — formerly the separate `USES` / `TYPE_REF`
/// passes, retained here so an older graph still traverses), or `IMPORTS`ed.
/// Listed in a fixed order so the merged reference scan visits edge types
/// reproducibly; the final result is sorted by a total key regardless.
pub const REFERENCE_EDGE_TYPES: &[&str] = &["CALLS", "USAGE", "USES", "TYPE_REF", "IMPORTS"];

/// One reference returned by [`find_references`]: the referencing node plus
/// the edge type by which it refers to the target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// The relation by which `node` refers to the queried symbol — one of
    /// [`REFERENCE_EDGE_TYPES`].
    pub edge_type: String,
    /// The referencing node's row (same shape as [`search_graph`] results).
    pub node: SearchGraphRow,
}

/// Unified "find references" query: every node that refers to `target_id`
/// over **any** of [`REFERENCE_EDGE_TYPES`] (`CALLS`, `USES`, `TYPE_REF`,
/// `IMPORTS`), i.e. every incoming reference edge in one pass.
///
/// This is the structured analogue of an IDE's "Find all references": where
/// [`neighbors`] answers one edge type at a time, this merges all four
/// reference relations into a single deduplicated, ranked list so a caller
/// asking "who depends on this symbol?" gets the complete answer in one call.
///
/// Determinism and bounds:
/// - Each edge type is scanned with the id-ordered store accessors, capped at
///   [`MAX_REACH_RESULTS`] edges per type.
/// - A `(referencing-node, edge_type)` pair is emitted once even if the store
///   holds several such edges; the same node referring via two *different*
///   edge types yields two rows (one per relation), so the caller sees how it
///   depends.
/// - Self-references (a node referring to itself) are excluded.
/// - The result is sorted by `(qualified_name asc, id asc, edge_type asc)` for
///   a total, reproducible order, then truncated to `limit` (clamped to
///   [`MAX_REACH_RESULTS`]).
/// - A missing `target_id` or a `limit` of `0` yields an empty result.
pub fn find_references(store: &Store, target_id: i64, limit: usize) -> Result<Vec<Reference>> {
    find_references_to_any(store, &[target_id], limit)
}

/// Unified reference query across multiple target nodes that represent the
/// same user-facing symbol name. Rows are de-duplicated by `(source, edge_type)`
/// so a `Struct`/`Impl` name collision cannot double-count the same caller.
pub fn find_references_to_any(
    store: &Store,
    target_ids: &[i64],
    limit: usize,
) -> Result<Vec<Reference>> {
    use std::collections::HashSet;

    let limit = limit.min(MAX_REACH_RESULTS);
    if limit == 0 || target_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Dedup on (source node id, edge type): one row per distinct relation a
    // given node has to the target.
    let mut seen: HashSet<(i64, &str)> = HashSet::new();
    let mut out: Vec<Reference> = Vec::new();
    for &target_id in target_ids {
        if store.get_node(target_id)?.is_none() {
            continue;
        }
        for &ty in REFERENCE_EDGE_TYPES {
            for e in store.incoming_edges(target_id, Some(ty), MAX_REACH_RESULTS)? {
                let src = e.source_id;
                if src == target_id {
                    // A symbol referring to itself is not an external reference.
                    continue;
                }
                if !seen.insert((src, ty)) {
                    continue;
                }
                if let Some(node) = store.get_node(src)? {
                    out.push(Reference {
                        edge_type: ty.to_string(),
                        node: node_to_row(node),
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| {
        a.node
            .qualified_name
            .cmp(&b.node.qualified_name)
            .then_with(|| a.node.id.cmp(&b.node.id))
            .then_with(|| a.edge_type.cmp(&b.edge_type))
    });
    out.truncate(limit);
    Ok(out)
}

/// Count the full unified reference set for one target node without applying
/// display limits.
pub fn count_references(store: &Store, project: &str, target_id: i64) -> Result<usize> {
    count_references_to_any(store, project, &[target_id])
}

/// Count the full unified reference set for multiple target nodes without
/// applying display limits. The count uses the same `(source, edge_type)`
/// de-duplication and self-reference exclusion as [`find_references_to_any`].
pub fn count_references_to_any(store: &Store, project: &str, target_ids: &[i64]) -> Result<usize> {
    if target_ids.is_empty() {
        return Ok(0);
    }

    let mut existing_targets = Vec::new();
    for &target_id in target_ids {
        if store.get_node(target_id)?.is_some() {
            existing_targets.push(target_id);
        }
    }
    if existing_targets.is_empty() {
        return Ok(0);
    }

    let placeholders = (0..existing_targets.len())
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT COUNT(*) FROM (
             SELECT source_id, edge_type
             FROM edges
             WHERE project = ?
               AND target_id IN ({placeholders})
               AND source_id != target_id
               AND edge_type IN ('CALLS', 'USAGE', 'USES', 'TYPE_REF', 'IMPORTS')
             GROUP BY source_id, edge_type
         )"
    );

    let mut params_iter: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(existing_targets.len() + 1);
    params_iter.push(&project);
    for id in &existing_targets {
        params_iter.push(id);
    }

    let count: i64 = store
        .conn()
        .query_row(&sql, params_iter.as_slice(), |row| row.get(0))
        .map_err(grepplus_store::Error::Sqlite)?;
    Ok(count.max(0) as usize)
}

/// "Unused symbols" query: every definition node in `project` that has **zero
/// incoming reference edges** of any [`REFERENCE_EDGE_TYPES`] kind — nothing
/// `CALLS`, `USES`, `TYPE_REF`s, or `IMPORTS` it.
///
/// This is the static dead-code heuristic: a definition no other symbol
/// refers to is a candidate for removal (modulo entry points, exported API,
/// dynamic dispatch, and reflection, which the static graph cannot see — so
/// the result is *candidates*, not a proof). It is the complement of
/// [`find_references`]: a symbol is unused exactly when `find_references`
/// would return empty for it.
///
/// `labels` optionally restricts the scan to definition kinds (e.g.
/// `&["Function", "Method"]`); an empty slice scans every node in the project.
/// Import nodes are typically *not* definitions and have no incoming reference
/// edges by construction, so callers usually pass a definition-label filter to
/// avoid flagging every import — but the function itself imposes no implicit
/// label policy.
///
/// Determinism and bounds:
/// - Candidates come from [`search_graph`] (already a total `(qualified_name,
///   id)` order) scoped to `project`.
/// - For each candidate, every reference edge type is checked with the
///   id-ordered store accessors; a single incoming reference (even a
///   self-reference) disqualifies the node.
/// - The result preserves `search_graph`'s `(qualified_name asc, id asc)`
///   order and is truncated to `limit` (clamped to [`MAX_REACH_RESULTS`]).
/// - An empty project, or a `limit` of `0`, yields an empty result.
pub fn unused_symbols(
    store: &Store,
    project: &str,
    labels: &[&str],
    limit: usize,
) -> Result<Vec<SearchGraphRow>> {
    use std::collections::HashSet;

    let limit = limit.min(MAX_REACH_RESULTS);
    if limit == 0 {
        return Ok(Vec::new());
    }

    let mut q = GraphQuery::in_project(project).with_limit(MAX_REACH_RESULTS);
    let allowed: HashSet<&str> = labels.iter().copied().collect();
    // When exactly one label is requested, push it into the SQL filter; for
    // multiple labels filter in Rust (search_graph takes a single label).
    if labels.len() == 1 {
        q = q.with_label(labels[0]);
    }
    let rows = search_graph(store, &q)?;

    let mut out: Vec<SearchGraphRow> = Vec::new();
    for row in rows {
        if !allowed.is_empty() && !allowed.contains(row.label.as_str()) {
            continue;
        }
        let mut has_ref = false;
        for &ty in REFERENCE_EDGE_TYPES {
            // We only need to know whether *any* reference exists; cap at 1.
            if !store.incoming_edges(row.id, Some(ty), 1)?.is_empty() {
                has_ref = true;
                break;
            }
        }
        if !has_ref {
            out.push(row);
            if out.len() >= limit {
                break;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use grepplus_store::{NewNode, Project, Store};

    fn seed() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for (label, name) in [
            ("Function", "alpha"),
            ("Function", "beta"),
            ("Struct", "Gamma"),
            ("Import", "std::collections::HashMap"),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: name.into(),
                qualified_name: format!("p::{label}::{name}"),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        s
    }

    #[test]
    fn search_by_label() {
        let s = seed();
        let r = search_graph(&s, &GraphQuery::in_project("p").with_label("Function")).unwrap();
        assert_eq!(r.len(), 2);
        let names: Vec<&str> = r.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn count_search_graph_ignores_display_limit() {
        let s = seed();
        let q = GraphQuery::in_project("p")
            .with_label("Function")
            .with_limit(1);

        let shown = search_graph(&s, &q).unwrap();
        let total = count_search_graph(&s, &q).unwrap();

        assert_eq!(shown.len(), 1);
        assert_eq!(total, 2);
    }

    #[test]
    fn search_by_name() {
        let s = seed();
        let r = search_graph(&s, &GraphQuery::in_project("p").with_name("Gamma")).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].label, "Struct");
    }

    #[test]
    fn search_by_qname_substring() {
        let s = seed();
        let q = GraphQuery::in_project("p").with_qualified_name_contains("Function::");
        let r = search_graph(&s, &q).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn search_respects_limit() {
        let s = seed();
        let r = search_graph(&s, &GraphQuery::in_project("p").with_limit(1)).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn search_with_no_project_returns_empty_when_no_rows() {
        let s = seed();
        let r = search_graph(&s, &GraphQuery::in_project("nonexistent")).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_by_name_prefix() {
        let s = seed();
        // "alpha" and "beta" — only "alpha" starts with "al".
        let r = search_graph(&s, &GraphQuery::in_project("p").with_name_prefix("al")).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].name, "alpha");
    }

    #[test]
    fn search_by_name_contains() {
        let s = seed();
        // Both "alpha" and "Gamma" contain "a"... but name_contains is
        // case-sensitive: "ph" appears only in "alpha".
        let r = search_graph(&s, &GraphQuery::in_project("p").with_name_contains("ph")).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].name, "alpha");
    }

    #[test]
    fn name_predicates_are_composable_and() {
        let s = seed();
        // prefix "be" AND contains "ta" -> "beta" only.
        let q = GraphQuery::in_project("p")
            .with_name_prefix("be")
            .with_name_contains("ta");
        let r = search_graph(&s, &q).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].name, "beta");
    }

    #[test]
    fn like_wildcards_are_escaped_as_literals() {
        // A name containing a literal '%' must not be treated as a
        // wildcard: searching for "x" must NOT match "a%b".
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for name in ["a%b", "axb"] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: format!("p::{name}"),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        // Literal "a%b" must match exactly one row (not both via wildcard).
        let r = search_graph(&s, &GraphQuery::in_project("p").with_name_contains("a%b")).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].name, "a%b");
    }

    #[test]
    fn search_by_file_path_prefix() {
        let mut s = seed();
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "other".into(),
            qualified_name: "p::other".into(),
            file_path: "tests/it.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap();
        let r = search_graph(
            &s,
            &GraphQuery::in_project("p").with_file_path_prefix("src/"),
        )
        .unwrap();
        // The four seed nodes live under src/lib.rs; the extra node under tests/.
        assert_eq!(r.len(), 4);
        assert!(r.iter().all(|row| row.file_path.starts_with("src/")));
    }

    fn seed_with_edges() -> (Store, i64, i64) {
        // caller --CALLS--> callee
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let caller = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "caller".into(),
                qualified_name: "p::caller".into(),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        let callee = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "callee".into(),
                qualified_name: "p::callee".into(),
                file_path: "src/lib.rs".into(),
                start_line: 5,
                end_line: 5,
                properties: serde_json::json!({}),
            })
            .unwrap();
        // An isolated node with no edges at all.
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "island".into(),
            qualified_name: "p::island".into(),
            file_path: "src/lib.rs".into(),
            start_line: 9,
            end_line: 9,
            properties: serde_json::json!({}),
        })
        .unwrap();
        s.insert_edge(&grepplus_store::NewEdge {
            project: "p".into(),
            source_id: caller,
            target_id: callee,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        (s, caller, callee)
    }

    #[test]
    fn outgoing_edge_predicate_keeps_only_sources() {
        let (s, caller, _callee) = seed_with_edges();
        let r = search_graph(&s, &GraphQuery::in_project("p").with_outgoing_edge("CALLS")).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, caller);
        assert_eq!(r[0].name, "caller");
    }

    #[test]
    fn incoming_edge_predicate_keeps_only_targets() {
        let (s, _caller, callee) = seed_with_edges();
        let r = search_graph(&s, &GraphQuery::in_project("p").with_incoming_edge("CALLS")).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, callee);
        assert_eq!(r[0].name, "callee");
    }

    #[test]
    fn edge_predicate_wrong_type_matches_nothing() {
        let (s, _, _) = seed_with_edges();
        let r = search_graph(
            &s,
            &GraphQuery::in_project("p").with_outgoing_edge("IMPORTS"),
        )
        .unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn edge_predicate_composes_with_label() {
        let (s, caller, _) = seed_with_edges();
        // Function label AND has an outgoing CALLS edge -> just caller.
        let q = GraphQuery::in_project("p")
            .with_label("Function")
            .with_outgoing_edge("CALLS");
        let r = search_graph(&s, &q).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, caller);
    }

    /// a -CALLS-> b -CALLS-> c -CALLS-> d, plus a -CALLS-> e (a second
    /// branch) and an isolated island. Returns the store and the ids in
    /// (a, b, c, d, e) order.
    fn seed_multihop() -> (Store, i64, i64, i64, i64, i64) {
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
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap()
        };
        let a = mk(&mut s, "a");
        let b = mk(&mut s, "b");
        let c = mk(&mut s, "c");
        let d = mk(&mut s, "d");
        let e = mk(&mut s, "e");
        let _island = mk(&mut s, "island");
        let edge = |s: &mut Store, src: i64, tgt: i64| {
            s.insert_edge(&grepplus_store::NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(&mut s, a, b);
        edge(&mut s, b, c);
        edge(&mut s, c, d);
        edge(&mut s, a, e);
        (s, a, b, c, d, e)
    }

    #[test]
    fn reachable_within_outgoing_collects_all_within_depth() {
        let (s, a, b, _c, _d, _e) = seed_multihop();
        // 2 hops from a over CALLS: b,e at hop 1; c at hop 2. d is at
        // hop 3 -> excluded.
        let r = reachable_within(&s, a, ReachDirection::Outgoing, "CALLS", 2, 100).unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        assert_eq!(names, vec!["b", "e", "c"], "ordering: hop then qname");
        // Start node never appears.
        assert!(r.iter().all(|x| x.node.id != a));
        let hop = |name: &str| r.iter().find(|x| x.node.name == name).unwrap().hops;
        assert_eq!(hop("b"), 1);
        assert_eq!(hop("e"), 1);
        assert_eq!(hop("c"), 2);
        let _ = b;
    }

    #[test]
    fn reachable_within_respects_hop_cap() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        // 1 hop -> only the direct neighbours b and e.
        let r = reachable_within(&s, a, ReachDirection::Outgoing, "CALLS", 1, 100).unwrap();
        let mut names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["b", "e"]);
    }

    #[test]
    fn reachable_within_incoming_walks_backwards() {
        let (s, a, _b, _c, d, _e) = seed_multihop();
        // Who reaches d within 3 hops over CALLS? c(1), b(2), a(3).
        let r = reachable_within(&s, d, ReachDirection::Incoming, "CALLS", 3, 100).unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        assert_eq!(names, vec!["c", "b", "a"]);
        let _ = a;
    }

    #[test]
    fn reachable_within_min_hops_on_diamond() {
        // a -> b -> d and a -> d directly: d should be reported at its
        // MINIMUM hop count (1), not 2.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mut ids = std::collections::HashMap::new();
        for n in ["a", "b", "d"] {
            let id = s
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: n.into(),
                    qualified_name: format!("p::{n}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            ids.insert(n, id);
        }
        let mut edge = |src: i64, tgt: i64| {
            s.insert_edge(&grepplus_store::NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(ids["a"], ids["b"]);
        edge(ids["b"], ids["d"]);
        edge(ids["a"], ids["d"]);
        let r = reachable_within(&s, ids["a"], ReachDirection::Outgoing, "CALLS", 5, 100).unwrap();
        let d = r.iter().find(|x| x.node.name == "d").unwrap();
        assert_eq!(d.hops, 1, "d is reachable directly -> min hop is 1");
    }

    #[test]
    fn reachable_within_wrong_edge_type_is_empty() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let r = reachable_within(&s, a, ReachDirection::Outgoing, "IMPORTS", 5, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn reachable_within_missing_start_is_empty() {
        let (s, _a, _b, _c, _d, _e) = seed_multihop();
        let r = reachable_within(&s, 999_999, ReachDirection::Outgoing, "CALLS", 5, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn reachable_within_zero_hops_is_empty() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let r = reachable_within(&s, a, ReachDirection::Outgoing, "CALLS", 0, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn reachable_within_respects_result_limit() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        // Full reach from a is b,e,c,d (4 nodes). Cap to 2.
        let r = reachable_within(&s, a, ReachDirection::Outgoing, "CALLS", 10, 2).unwrap();
        assert_eq!(r.len(), 2);
        // The two nearest (hop 1) are b and e, alphabetical.
        let names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        assert_eq!(names, vec!["b", "e"]);
    }

    #[test]
    fn reachable_within_is_deterministic_across_runs() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let first = reachable_within(&s, a, ReachDirection::Outgoing, "CALLS", 10, 100).unwrap();
        let second = reachable_within(&s, a, ReachDirection::Outgoing, "CALLS", 10, 100).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn reachable_within_clamps_hops_to_max() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        // A huge hop request is clamped to MAX_REACH_HOPS but still
        // returns the full finite reachable set without panicking.
        let r = reachable_within(
            &s,
            a,
            ReachDirection::Outgoing,
            "CALLS",
            usize::MAX,
            usize::MAX,
        )
        .unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        assert_eq!(names, vec!["b", "e", "c", "d"]);
    }

    #[test]
    fn neighbors_outgoing_returns_direct_callees() {
        let (s, a, b, _c, _d, _e) = seed_multihop();
        // a directly calls b and e.
        let r = neighbors(&s, a, "CALLS", ReachDirection::Outgoing, 100).unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, vec!["b", "e"], "sorted by qualified_name");
        let _ = b;
    }

    #[test]
    fn neighbors_incoming_returns_direct_callers() {
        let (s, _a, b, c, _d, _e) = seed_multihop();
        // Who directly calls c? Only b.
        let r = neighbors(&s, c, "CALLS", ReachDirection::Incoming, 100).unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, vec!["b"]);
        let _ = b;
    }

    #[test]
    fn neighbors_excludes_two_hop_nodes() {
        let (s, a, _b, c, _d, _e) = seed_multihop();
        // c is two hops from a; it must NOT appear among a's neighbours.
        let r = neighbors(&s, a, "CALLS", ReachDirection::Outgoing, 100).unwrap();
        assert!(r.iter().all(|x| x.id != c));
    }

    #[test]
    fn neighbors_wrong_edge_type_is_empty() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let r = neighbors(&s, a, "IMPORTS", ReachDirection::Outgoing, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn neighbors_missing_node_is_empty() {
        let (s, _a, _b, _c, _d, _e) = seed_multihop();
        let r = neighbors(&s, 999_999, "CALLS", ReachDirection::Outgoing, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn neighbors_respects_limit_and_is_deterministic() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let r = neighbors(&s, a, "CALLS", ReachDirection::Outgoing, 1).unwrap();
        assert_eq!(r.len(), 1);
        // Alphabetically first neighbour is "b".
        assert_eq!(r[0].name, "b");
        let again = neighbors(&s, a, "CALLS", ReachDirection::Outgoing, 1).unwrap();
        assert_eq!(r, again);
    }

    #[test]
    fn neighbors_skips_self_loop() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let n = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "recur".into(),
                qualified_name: "p::recur".into(),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        s.insert_edge(&grepplus_store::NewEdge {
            project: "p".into(),
            source_id: n,
            target_id: n,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        let r = neighbors(&s, n, "CALLS", ReachDirection::Outgoing, 100).unwrap();
        assert!(r.is_empty(), "a self-loop is not a neighbour");
    }

    #[test]
    fn subgraph_around_collects_center_and_neighbourhood() {
        // a -> b -> c, plus a -> e. Subgraph around b within 1 hop
        // (undirected over CALLS) is {a, b, c}.
        let (s, a, b, c, _d, _e) = seed_multihop();
        let sg = subgraph_around(&s, b, &["CALLS"], 1).unwrap().unwrap();
        assert_eq!(sg.center.id, b);
        let ids: std::collections::HashSet<i64> = sg.nodes.iter().map(|n| n.id).collect();
        assert_eq!(
            ids,
            [a, b, c].into_iter().collect(),
            "undirected 1-hop around b is a,b,c"
        );
        // Induced edges among {a,b,c}: a->b and b->c.
        assert_eq!(sg.edges.len(), 2);
        assert!(sg
            .edges
            .iter()
            .any(|e| e.source_id == a && e.target_id == b));
        assert!(sg
            .edges
            .iter()
            .any(|e| e.source_id == b && e.target_id == c));
    }

    #[test]
    fn subgraph_around_excludes_edges_leaving_the_set() {
        // Around b within 1 hop = {a,b,c}. The edge a->e leaves the set
        // (e not collected), so it must NOT appear in the induced edges.
        let (s, _a, b, _c, _d, e) = seed_multihop();
        let sg = subgraph_around(&s, b, &["CALLS"], 1).unwrap().unwrap();
        assert!(sg.nodes.iter().all(|n| n.id != e));
        assert!(sg.edges.iter().all(|edge| edge.target_id != e));
    }

    #[test]
    fn subgraph_around_respects_hop_budget() {
        // Around a within 1 hop = {a,b,e}; c (2 hops) excluded.
        let (s, a, b, c, _d, e) = seed_multihop();
        let sg = subgraph_around(&s, a, &["CALLS"], 1).unwrap().unwrap();
        let ids: std::collections::HashSet<i64> = sg.nodes.iter().map(|n| n.id).collect();
        assert_eq!(ids, [a, b, e].into_iter().collect());
        assert!(!ids.contains(&c));
    }

    #[test]
    fn subgraph_around_missing_center_is_none() {
        let (s, _a, _b, _c, _d, _e) = seed_multihop();
        let sg = subgraph_around(&s, 999_999, &["CALLS"], 2).unwrap();
        assert!(sg.is_none());
    }

    #[test]
    fn subgraph_around_empty_edge_types_is_center_only() {
        let (s, _a, b, _c, _d, _e) = seed_multihop();
        let sg = subgraph_around(&s, b, &[], 3).unwrap().unwrap();
        assert_eq!(sg.nodes.len(), 1);
        assert_eq!(sg.nodes[0].id, b);
        assert!(sg.edges.is_empty());
    }

    #[test]
    fn subgraph_around_is_deterministic() {
        let (s, _a, _b, c, _d, _e) = seed_multihop();
        let first = subgraph_around(&s, c, &["CALLS"], 5).unwrap();
        let second = subgraph_around(&s, c, &["CALLS"], 5).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn subgraph_around_dedups_repeated_edge_types() {
        // Passing "CALLS" twice must not double-collect nodes or edges.
        let (s, a, b, c, _d, _e) = seed_multihop();
        let once = subgraph_around(&s, b, &["CALLS"], 1).unwrap().unwrap();
        let twice = subgraph_around(&s, b, &["CALLS", "CALLS"], 1)
            .unwrap()
            .unwrap();
        assert_eq!(once, twice);
        let _ = (a, c);
    }

    #[test]
    fn deterministic_order_is_stable_across_runs() {
        let (s, _, _) = seed_with_edges();
        let q = GraphQuery::in_project("p");
        let first = search_graph(&s, &q).unwrap();
        let second = search_graph(&s, &q).unwrap();
        assert_eq!(first, second);
        // Ordered by qualified_name: caller, callee, island ->
        // alphabetical: p::callee, p::caller, p::island.
        let names: Vec<&str> = first.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["callee", "caller", "island"]);
    }

    #[test]
    fn default_order_is_qualified_name() {
        // Omitting `with_order` must reproduce the historic order exactly.
        let s = seed();
        let explicit = search_graph(
            &s,
            &GraphQuery::in_project("p").with_order(GraphOrder::QualifiedName),
        )
        .unwrap();
        let implicit = search_graph(&s, &GraphQuery::in_project("p")).unwrap();
        assert_eq!(explicit, implicit);
    }

    #[test]
    fn order_by_name_sorts_by_leaf_name() {
        let s = seed();
        // Leaf names: alpha, beta, Gamma, std::collections::HashMap node
        // has name "std::collections::HashMap". ASCII sort: capitals
        // sort before lowercase, so "Gamma" and the "std::..." name order
        // accordingly. Assert the result is sorted by `name` ascending.
        let r = search_graph(
            &s,
            &GraphQuery::in_project("p").with_order(GraphOrder::Name),
        )
        .unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "rows must be ordered by leaf name");
    }

    #[test]
    fn order_by_file_groups_by_path_then_line() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mk = |s: &mut Store, name: &str, file: &str, line: i64| {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: format!("p::{name}"),
                file_path: file.into(),
                start_line: line,
                end_line: line,
                properties: serde_json::json!({}),
            })
            .unwrap()
        };
        mk(&mut s, "z_in_a", "a.rs", 10);
        mk(&mut s, "a_in_a", "a.rs", 5);
        mk(&mut s, "x_in_b", "b.rs", 1);
        let r = search_graph(
            &s,
            &GraphQuery::in_project("p").with_order(GraphOrder::File),
        )
        .unwrap();
        // a.rs:5, a.rs:10, b.rs:1 — file then start_line.
        let order: Vec<(&str, i64)> = r
            .iter()
            .map(|x| (x.file_path.as_str(), x.start_line))
            .collect();
        assert_eq!(order, vec![("a.rs", 5), ("a.rs", 10), ("b.rs", 1)]);
    }

    /// hub --CALLS--> {leaf1, leaf2, leaf3}; spoke --CALLS--> leaf1.
    /// Out-degree: hub=3, spoke=1, leaves=0. In-degree: leaf1=2,
    /// leaf2=1, leaf3=1, others 0. Returns the store.
    fn seed_degree_graph() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mut id = std::collections::HashMap::new();
        for n in ["hub", "spoke", "leaf1", "leaf2", "leaf3"] {
            let nid = s
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: n.into(),
                    qualified_name: format!("p::{n}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            id.insert(n, nid);
        }
        let mut edge = |src: i64, tgt: i64| {
            s.insert_edge(&grepplus_store::NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(id["hub"], id["leaf1"]);
        edge(id["hub"], id["leaf2"]);
        edge(id["hub"], id["leaf3"]);
        edge(id["spoke"], id["leaf1"]);
        s
    }

    #[test]
    fn order_by_out_degree_desc_ranks_most_connected_source_first() {
        let s = seed_degree_graph();
        let r = search_graph(
            &s,
            &GraphQuery::in_project("p").with_order(GraphOrder::OutDegreeDesc),
        )
        .unwrap();
        // hub (out 3) first, then spoke (out 1), then the leaves (out 0)
        // in qualified_name order.
        let names: Vec<&str> = r.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names[0], "hub");
        assert_eq!(names[1], "spoke");
        // Remaining are the three leaves, alphabetical.
        assert_eq!(&names[2..], &["leaf1", "leaf2", "leaf3"]);
    }

    #[test]
    fn order_by_in_degree_desc_ranks_most_referenced_first() {
        let s = seed_degree_graph();
        let r = search_graph(
            &s,
            &GraphQuery::in_project("p").with_order(GraphOrder::InDegreeDesc),
        )
        .unwrap();
        // leaf1 has in-degree 2 -> first.
        assert_eq!(r[0].name, "leaf1");
    }

    #[test]
    fn order_by_degree_respects_edge_type_filter() {
        // Add an IMPORTS edge into leaf2 so that, counted across all edge
        // types, leaf2 ties leaf1 — but scoped to CALLS, leaf1 still wins.
        let mut s = seed_degree_graph();
        // Find ids.
        let rows = search_graph(&s, &GraphQuery::in_project("p")).unwrap();
        let id = |name: &str| rows.iter().find(|r| r.name == name).unwrap().id;
        s.insert_edge(&grepplus_store::NewEdge {
            project: "p".into(),
            source_id: id("hub"),
            target_id: id("leaf2"),
            edge_type: "IMPORTS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        let r = search_graph(
            &s,
            &GraphQuery::in_project("p")
                .with_order(GraphOrder::InDegreeDesc)
                .with_order_edge_type("CALLS"),
        )
        .unwrap();
        // Scoped to CALLS, leaf1 (in 2) still beats leaf2 (CALLS in 1).
        assert_eq!(r[0].name, "leaf1");
    }

    #[test]
    fn order_by_degree_respects_limit_and_is_deterministic() {
        let s = seed_degree_graph();
        let q = GraphQuery::in_project("p")
            .with_order(GraphOrder::OutDegreeDesc)
            .with_limit(2);
        let first = search_graph(&s, &q).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].name, "hub");
        assert_eq!(first[1].name, "spoke");
        let second = search_graph(&s, &q).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn find_by_label_and_file_exact_matches_only_that_file() {
        let mut s = seed();
        // Add a Function in a different file.
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "elsewhere".into(),
            qualified_name: "p::elsewhere".into(),
            file_path: "src/other.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap();
        let r = find_by_label_and_file(&s, Some("p"), "Function", "src/lib.rs", true, 100).unwrap();
        // The two seed Functions (alpha, beta) live in src/lib.rs; the
        // Struct/Import are filtered by label; "elsewhere" by file.
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|x| x.label == "Function"));
        assert!(r.iter().all(|x| x.file_path == "src/lib.rs"));
        let names: Vec<&str> = r.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
        assert!(!names.contains(&"elsewhere"));
    }

    #[test]
    fn find_by_label_and_file_substring_matches_partial_path() {
        let s = seed();
        // "lib" is a substring of "src/lib.rs".
        let r = find_by_label_and_file(&s, Some("p"), "Function", "lib", false, 100).unwrap();
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|x| x.file_path.contains("lib")));
    }

    #[test]
    fn find_by_label_and_file_wrong_label_is_empty() {
        let s = seed();
        let r = find_by_label_and_file(&s, Some("p"), "Enum", "src/lib.rs", true, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn find_by_label_and_file_orders_by_source_line() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for (name, line) in [("late", 50), ("early", 5), ("mid", 20)] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: format!("p::{name}"),
                file_path: "src/f.rs".into(),
                start_line: line,
                end_line: line,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        let r = find_by_label_and_file(&s, Some("p"), "Function", "src/f.rs", true, 100).unwrap();
        let order: Vec<&str> = r.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(order, vec!["early", "mid", "late"], "source-line order");
    }

    #[test]
    fn most_connected_ranks_hub_first_by_out_degree() {
        let s = seed_degree_graph();
        // Out-degree: hub=3, spoke=1, leaves=0.
        let r = most_connected(&s, Some("p"), Some("CALLS"), DegreeKind::Out, 3).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].node.name, "hub");
        assert_eq!(r[0].degree, 3);
        assert_eq!(r[1].node.name, "spoke");
        assert_eq!(r[1].degree, 1);
        // The third slot is the alphabetically-first degree-0 leaf.
        assert_eq!(r[2].degree, 0);
        assert_eq!(r[2].node.name, "leaf1");
    }

    #[test]
    fn most_connected_in_degree_ranks_most_referenced_first() {
        let s = seed_degree_graph();
        // In-degree: leaf1=2, leaf2=1, leaf3=1, hub=0, spoke=0.
        let r = most_connected(&s, Some("p"), Some("CALLS"), DegreeKind::In, 2).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].node.name, "leaf1");
        assert_eq!(r[0].degree, 2);
        // Second is leaf2 or leaf3 (both degree 1); qname tie-break -> leaf2.
        assert_eq!(r[1].degree, 1);
        assert_eq!(r[1].node.name, "leaf2");
    }

    #[test]
    fn most_connected_total_counts_both_directions() {
        let s = seed_degree_graph();
        // Total degree: hub=3 (out), leaf1=2 (in), spoke=1, leaf2=1, leaf3=1.
        let r = most_connected(&s, Some("p"), Some("CALLS"), DegreeKind::Total, 5).unwrap();
        assert_eq!(r[0].node.name, "hub");
        assert_eq!(r[0].degree, 3);
        assert_eq!(r[1].node.name, "leaf1");
        assert_eq!(r[1].degree, 2);
    }

    #[test]
    fn most_connected_respects_n_and_is_deterministic() {
        let s = seed_degree_graph();
        let first = most_connected(&s, Some("p"), Some("CALLS"), DegreeKind::Out, 1).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].node.name, "hub");
        let second = most_connected(&s, Some("p"), Some("CALLS"), DegreeKind::Out, 1).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn most_connected_zero_n_is_empty() {
        let s = seed_degree_graph();
        let r = most_connected(&s, Some("p"), Some("CALLS"), DegreeKind::Out, 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn most_connected_unknown_edge_type_all_zero_degree() {
        let s = seed_degree_graph();
        let r = most_connected(&s, Some("p"), Some("IMPORTS"), DegreeKind::Out, 3).unwrap();
        // No IMPORTS edges -> every node has degree 0, but rows still
        // returned in qualified_name order.
        assert!(r.iter().all(|x| x.degree == 0));
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn fan_in_returns_positive_in_degree_with_exact_count() {
        let s = seed_degree_graph();
        let shown = fan_in(&s, "p", "CALLS", 1).unwrap();
        let total = count_fan_in(&s, "p", "CALLS").unwrap();

        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].node.name, "leaf1");
        assert_eq!(shown[0].degree, 2);
        assert_eq!(total, 3);
    }

    #[test]
    fn fan_out_returns_positive_out_degree_with_exact_count() {
        let s = seed_degree_graph();
        let shown = fan_out(&s, "p", "CALLS", 1).unwrap();
        let total = count_fan_out(&s, "p", "CALLS").unwrap();

        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].node.name, "hub");
        assert_eq!(shown[0].degree, 3);
        assert_eq!(total, 2);
    }

    #[test]
    fn fan_degree_unknown_edge_type_is_empty() {
        let s = seed_degree_graph();

        assert!(fan_in(&s, "p", "IMPORTS", 10).unwrap().is_empty());
        assert!(fan_out(&s, "p", "IMPORTS", 10).unwrap().is_empty());
        assert_eq!(count_fan_in(&s, "p", "IMPORTS").unwrap(), 0);
        assert_eq!(count_fan_out(&s, "p", "IMPORTS").unwrap(), 0);
    }

    #[test]
    fn path_query_finds_direct_edge() {
        let (s, caller, callee) = seed_with_edges();
        let p = path_query(&s, caller, callee, ReachDirection::Outgoing, "CALLS", 5)
            .unwrap()
            .unwrap();
        assert_eq!(p.nodes, vec![caller, callee]);
        assert_eq!(p.hops, 1);
        assert_eq!(p.rows.len(), 2);
        assert_eq!(p.rows[0].name, "caller");
        assert_eq!(p.rows[1].name, "callee");
    }

    #[test]
    fn path_query_finds_multihop_chain() {
        // a -> b -> c -> d.
        let (s, a, _b, _c, d, _e) = seed_multihop();
        let p = path_query(&s, a, d, ReachDirection::Outgoing, "CALLS", 5)
            .unwrap()
            .unwrap();
        let names: Vec<&str> = p.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c", "d"]);
        assert_eq!(p.hops, 3);
        assert_eq!(*p.nodes.first().unwrap(), a);
        assert_eq!(*p.nodes.last().unwrap(), d);
    }

    #[test]
    fn path_query_returns_shortest_on_diamond() {
        // a -> b -> d and a -> d directly: shortest is the direct edge.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mut ids = std::collections::HashMap::new();
        for n in ["a", "b", "d"] {
            let id = s
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: n.into(),
                    qualified_name: format!("p::{n}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            ids.insert(n, id);
        }
        // Insert the long route first, the direct edge second, to prove
        // BFS (not edge order) yields the shortest path.
        let mut edge = |src: i64, tgt: i64| {
            s.insert_edge(&grepplus_store::NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(ids["a"], ids["b"]);
        edge(ids["b"], ids["d"]);
        edge(ids["a"], ids["d"]);
        let p = path_query(&s, ids["a"], ids["d"], ReachDirection::Outgoing, "CALLS", 5)
            .unwrap()
            .unwrap();
        assert_eq!(p.hops, 1, "the direct a->d edge is the shortest path");
        assert_eq!(p.nodes, vec![ids["a"], ids["d"]]);
    }

    #[test]
    fn path_query_respects_hop_bound() {
        // a -> b -> c -> d is 3 hops; bounding to 2 finds no path.
        let (s, a, _b, _c, d, _e) = seed_multihop();
        let p = path_query(&s, a, d, ReachDirection::Outgoing, "CALLS", 2).unwrap();
        assert!(p.is_none(), "d is 3 hops away; a 2-hop bound finds nothing");
    }

    #[test]
    fn path_query_incoming_walks_backwards() {
        // d is reachable from a forwards; backwards a is reachable from d.
        let (s, a, _b, _c, d, _e) = seed_multihop();
        let p = path_query(&s, d, a, ReachDirection::Incoming, "CALLS", 5)
            .unwrap()
            .unwrap();
        let names: Vec<&str> = p.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["d", "c", "b", "a"]);
    }

    #[test]
    fn path_query_no_path_is_none() {
        // island has no edges; no path from a to island.
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let rows = search_graph(&s, &GraphQuery::in_project("p")).unwrap();
        let island = rows.iter().find(|r| r.name == "island").unwrap().id;
        let p = path_query(&s, a, island, ReachDirection::Outgoing, "CALLS", 10).unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn path_query_same_node_is_zero_hop_path() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let p = path_query(&s, a, a, ReachDirection::Outgoing, "CALLS", 5)
            .unwrap()
            .unwrap();
        assert_eq!(p.hops, 0);
        assert_eq!(p.nodes, vec![a]);
        assert_eq!(p.rows.len(), 1);
    }

    #[test]
    fn path_query_missing_endpoint_is_none() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let p = path_query(&s, a, 999_999, ReachDirection::Outgoing, "CALLS", 5).unwrap();
        assert!(p.is_none());
        let p2 = path_query(&s, 999_999, a, ReachDirection::Outgoing, "CALLS", 5).unwrap();
        assert!(p2.is_none());
    }

    #[test]
    fn path_query_wrong_edge_type_is_none() {
        let (s, a, _b, _c, d, _e) = seed_multihop();
        let p = path_query(&s, a, d, ReachDirection::Outgoing, "IMPORTS", 10).unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn path_query_is_deterministic_across_runs() {
        let (s, a, _b, _c, d, _e) = seed_multihop();
        let first = path_query(&s, a, d, ReachDirection::Outgoing, "CALLS", 10).unwrap();
        let second = path_query(&s, a, d, ReachDirection::Outgoing, "CALLS", 10).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn impact_radius_incoming_collects_transitive_callers_with_depth() {
        // a -> b -> c -> d. The blast radius of d (Incoming over CALLS) is
        // its transitive callers: c(1), b(2), a(3).
        let (s, a, b, c, d, _e) = seed_multihop();
        let r = impact_radius(&s, d, ReachDirection::Incoming, "CALLS", 5, 100).unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        assert_eq!(names, vec!["c", "b", "a"], "nearest caller first");
        let hop = |name: &str| r.iter().find(|x| x.node.name == name).unwrap().hops;
        assert_eq!(hop("c"), 1);
        assert_eq!(hop("b"), 2);
        assert_eq!(hop("a"), 3);
        // The source d is never in its own blast radius.
        assert!(r.iter().all(|x| x.node.id != d));
        let _ = (a, b, c);
    }

    #[test]
    fn impact_radius_outgoing_is_dependency_cone() {
        // From a, the outgoing impact (what a transitively calls) within 2
        // hops is b,e (hop 1) and c (hop 2); d is 3 hops -> excluded.
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let r = impact_radius(&s, a, ReachDirection::Outgoing, "CALLS", 2, 100).unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        assert_eq!(names, vec!["b", "e", "c"]);
    }

    #[test]
    fn impact_radius_dedupes_diamond_to_min_hops() {
        // a -> b -> d and a -> d directly: d must appear once at hop 1.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mut ids = std::collections::HashMap::new();
        for n in ["a", "b", "d"] {
            let id = s
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: n.into(),
                    qualified_name: format!("p::{n}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            ids.insert(n, id);
        }
        let mut edge = |src: i64, tgt: i64| {
            s.insert_edge(&grepplus_store::NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(ids["a"], ids["b"]);
        edge(ids["b"], ids["d"]);
        edge(ids["a"], ids["d"]);
        let r = impact_radius(&s, ids["a"], ReachDirection::Outgoing, "CALLS", 5, 100).unwrap();
        let d_hits: Vec<_> = r.iter().filter(|x| x.node.id == ids["d"]).collect();
        assert_eq!(d_hits.len(), 1, "d appears exactly once");
        assert_eq!(d_hits[0].hops, 1, "min-hop reporting");
    }

    #[test]
    fn impact_radius_respects_limit_and_is_deterministic() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        let r = impact_radius(&s, a, ReachDirection::Outgoing, "CALLS", 10, 2).unwrap();
        assert_eq!(r.len(), 2);
        let again = impact_radius(&s, a, ReachDirection::Outgoing, "CALLS", 10, 2).unwrap();
        assert_eq!(r, again);
    }

    #[test]
    fn impact_radius_missing_source_or_wrong_type_is_empty() {
        let (s, a, _b, _c, _d, _e) = seed_multihop();
        assert!(
            impact_radius(&s, 999_999, ReachDirection::Incoming, "CALLS", 5, 100)
                .unwrap()
                .is_empty()
        );
        assert!(
            impact_radius(&s, a, ReachDirection::Outgoing, "IMPORTS", 5, 100)
                .unwrap()
                .is_empty()
        );
    }

    /// s and x both CALL the shared callee m; x also CALLS n (which s does
    /// not). y CALLS only n. Returns (store, s, x, y, m, n).
    fn seed_co_change() -> (Store, i64, i64, i64, i64, i64) {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let mk = |store: &mut Store, name: &str| {
            store
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: name.into(),
                    qualified_name: format!("p::{name}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap()
        };
        let s = mk(&mut store, "s");
        let x = mk(&mut store, "x");
        let y = mk(&mut store, "y");
        let m = mk(&mut store, "m");
        let n = mk(&mut store, "n");
        let mut edge = |src: i64, tgt: i64| {
            store
                .insert_edge(&grepplus_store::NewEdge {
                    project: "p".into(),
                    source_id: src,
                    target_id: tgt,
                    edge_type: "CALLS".into(),
                    properties: serde_json::json!({}),
                })
                .unwrap();
        };
        edge(s, m); // s -> m
        edge(x, m); // x -> m  (x shares callee m with s)
        edge(x, n); // x -> n
        edge(y, n); // y -> n  (y shares nothing with s)
        (store, s, x, y, m, n)
    }

    #[test]
    fn co_change_candidates_finds_co_callers_of_shared_callee() {
        // Outgoing over CALLS: s's callee set is {m}. x also calls m, so x
        // is a candidate with shared==1. y calls only n -> not a candidate.
        let (store, s, x, y, _m, _n) = seed_co_change();
        let r = co_change_candidates(&store, s, ReachDirection::Outgoing, "CALLS", 100).unwrap();
        let names: Vec<&str> = r.iter().map(|c| c.node.name.as_str()).collect();
        assert_eq!(names, vec!["x"], "only x shares a callee with s");
        assert_eq!(r[0].shared, 1);
        assert!(r.iter().all(|c| c.node.id != s), "seed excluded");
        assert!(r.iter().all(|c| c.node.id != y));
        let _ = x;
    }

    #[test]
    fn co_change_candidates_ranks_by_shared_neighbour_count() {
        // s -> m1, m2. x -> m1, m2 (shares 2). z -> m1 only (shares 1).
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let mk = |store: &mut Store, name: &str| {
            store
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: name.into(),
                    qualified_name: format!("p::{name}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap()
        };
        let s = mk(&mut store, "s");
        let x = mk(&mut store, "x");
        let z = mk(&mut store, "z");
        let m1 = mk(&mut store, "m1");
        let m2 = mk(&mut store, "m2");
        let mut edge = |src: i64, tgt: i64| {
            store
                .insert_edge(&grepplus_store::NewEdge {
                    project: "p".into(),
                    source_id: src,
                    target_id: tgt,
                    edge_type: "CALLS".into(),
                    properties: serde_json::json!({}),
                })
                .unwrap();
        };
        edge(s, m1);
        edge(s, m2);
        edge(x, m1);
        edge(x, m2);
        edge(z, m1);
        let r = co_change_candidates(&store, s, ReachDirection::Outgoing, "CALLS", 100).unwrap();
        assert_eq!(r[0].node.name, "x");
        assert_eq!(r[0].shared, 2, "x shares both callees");
        assert_eq!(r[1].node.name, "z");
        assert_eq!(r[1].shared, 1, "z shares one callee");
    }

    #[test]
    fn co_change_candidates_incoming_finds_co_callees_of_shared_caller() {
        // Incoming over CALLS: build c -> s and c -> t, so s and t share
        // the caller c. From s (Incoming) the neighbour set is {c};
        // co-callees of c (Outgoing) are s and t -> candidate t.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let mk = |store: &mut Store, name: &str| {
            store
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: name.into(),
                    qualified_name: format!("p::{name}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap()
        };
        let s = mk(&mut store, "s");
        let t = mk(&mut store, "t");
        let c = mk(&mut store, "c");
        let mut edge = |src: i64, tgt: i64| {
            store
                .insert_edge(&grepplus_store::NewEdge {
                    project: "p".into(),
                    source_id: src,
                    target_id: tgt,
                    edge_type: "CALLS".into(),
                    properties: serde_json::json!({}),
                })
                .unwrap();
        };
        edge(c, s);
        edge(c, t);
        let r = co_change_candidates(&store, s, ReachDirection::Incoming, "CALLS", 100).unwrap();
        let names: Vec<&str> = r.iter().map(|x| x.node.name.as_str()).collect();
        assert_eq!(names, vec!["t"], "t shares caller c with s");
        assert_eq!(r[0].shared, 1);
    }

    #[test]
    fn co_change_candidates_dedupes_multiple_edges_to_same_neighbour() {
        // s -> m and x -> m twice (duplicate edge). x must still count m
        // once, so shared == 1.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let mk = |store: &mut Store, name: &str| {
            store
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: name.into(),
                    qualified_name: format!("p::{name}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap()
        };
        let s = mk(&mut store, "s");
        let x = mk(&mut store, "x");
        let m = mk(&mut store, "m");
        let mut edge = |src: i64, tgt: i64| {
            store
                .insert_edge(&grepplus_store::NewEdge {
                    project: "p".into(),
                    source_id: src,
                    target_id: tgt,
                    edge_type: "CALLS".into(),
                    properties: serde_json::json!({}),
                })
                .unwrap();
        };
        edge(s, m);
        edge(x, m);
        edge(x, m); // duplicate
        let r = co_change_candidates(&store, s, ReachDirection::Outgoing, "CALLS", 100).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].shared, 1, "duplicate edge does not inflate the score");
    }

    #[test]
    fn co_change_candidates_missing_seed_or_limit_zero_is_empty() {
        let (store, s, _x, _y, _m, _n) = seed_co_change();
        assert!(
            co_change_candidates(&store, 999_999, ReachDirection::Outgoing, "CALLS", 100)
                .unwrap()
                .is_empty()
        );
        assert!(
            co_change_candidates(&store, s, ReachDirection::Outgoing, "CALLS", 0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn co_change_candidates_is_deterministic() {
        let (store, s, _x, _y, _m, _n) = seed_co_change();
        let a = co_change_candidates(&store, s, ReachDirection::Outgoing, "CALLS", 100).unwrap();
        let b = co_change_candidates(&store, s, ReachDirection::Outgoing, "CALLS", 100).unwrap();
        assert_eq!(a, b);
    }

    /// Seed a single file with nested spans. `Widget` struct spans lines
    /// 1..=20 (outermost); its `render` method spans 5..=9 and `update`
    /// spans 11..=15; a sibling `helper` function spans 25..=30 after the
    /// struct. Inserted out of source order on purpose so the ordering
    /// helpers are exercised, not the insertion order.
    fn seed_spans() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for (label, name, qn, file, sl, el) in [
            ("Function", "helper", "p::helper", "src/widget.rs", 25, 30),
            (
                "Method",
                "update",
                "p::Widget::update",
                "src/widget.rs",
                11,
                15,
            ),
            ("Struct", "Widget", "p::Widget", "src/widget.rs", 1, 20),
            (
                "Method",
                "render",
                "p::Widget::render",
                "src/widget.rs",
                5,
                9,
            ),
            // A symbol in a different file must never leak in.
            ("Function", "other", "p::other", "src/other.rs", 1, 3),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: name.into(),
                qualified_name: qn.into(),
                file_path: file.into(),
                start_line: sl,
                end_line: el,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        s
    }

    #[test]
    fn symbols_in_file_returns_only_that_file_in_source_order() {
        let s = seed_spans();
        let rows = symbols_in_file(&s, Some("p"), "src/widget.rs", 100).unwrap();
        // Only the widget.rs symbols, never src/other.rs.
        assert!(rows.iter().all(|r| r.file_path == "src/widget.rs"));
        let order: Vec<&str> = rows.iter().map(|r| r.qualified_name.as_str()).collect();
        // Source order with enclosing-before-contained tie-break:
        // Widget (1..20) opens first; render (5) before update (11);
        // helper (25) last.
        assert_eq!(
            order,
            vec![
                "p::Widget",
                "p::Widget::render",
                "p::Widget::update",
                "p::helper",
            ]
        );
    }

    #[test]
    fn symbols_in_file_is_deterministic_and_respects_limit() {
        let s = seed_spans();
        let a = symbols_in_file(&s, Some("p"), "src/widget.rs", 100).unwrap();
        let b = symbols_in_file(&s, Some("p"), "src/widget.rs", 100).unwrap();
        assert_eq!(a, b);
        let limited = symbols_in_file(&s, Some("p"), "src/widget.rs", 2).unwrap();
        assert_eq!(limited.len(), 2);
        assert_eq!(limited[0].qualified_name, "p::Widget");
    }

    #[test]
    fn symbols_in_file_unknown_file_is_empty() {
        let s = seed_spans();
        assert!(symbols_in_file(&s, Some("p"), "src/nope.rs", 100)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn definition_at_resolves_innermost_enclosing_symbol() {
        let s = seed_spans();
        // A line inside `render`'s body resolves to the method, not Widget.
        let d = definition_at(&s, Some("p"), "src/widget.rs", 7)
            .unwrap()
            .expect("line 7 is inside render");
        assert_eq!(d.qualified_name, "p::Widget::render");

        // A line inside Widget but outside both methods (e.g. a field
        // declaration on line 3) resolves to the struct itself.
        let d = definition_at(&s, Some("p"), "src/widget.rs", 3)
            .unwrap()
            .expect("line 3 is inside Widget but no method");
        assert_eq!(d.qualified_name, "p::Widget");

        // The exact start line of a definition counts as inside it.
        let d = definition_at(&s, Some("p"), "src/widget.rs", 11)
            .unwrap()
            .expect("line 11 is update's start");
        assert_eq!(d.qualified_name, "p::Widget::update");

        // A line inside the sibling function.
        let d = definition_at(&s, Some("p"), "src/widget.rs", 27)
            .unwrap()
            .expect("line 27 is inside helper");
        assert_eq!(d.qualified_name, "p::helper");
    }

    #[test]
    fn definition_at_returns_none_when_no_span_covers_the_line() {
        let s = seed_spans();
        // Line 22 is between Widget (..20) and helper (25..) — no cover.
        assert!(definition_at(&s, Some("p"), "src/widget.rs", 22)
            .unwrap()
            .is_none());
        // A line past the end of the file.
        assert!(definition_at(&s, Some("p"), "src/widget.rs", 999)
            .unwrap()
            .is_none());
    }

    #[test]
    fn definition_at_is_deterministic() {
        let s = seed_spans();
        let a = definition_at(&s, Some("p"), "src/widget.rs", 7).unwrap();
        let b = definition_at(&s, Some("p"), "src/widget.rs", 7).unwrap();
        assert_eq!(a, b);
    }

    /// Build a store with a named node helper and an IMPORTS/CALLS edge
    /// helper, returning the store plus a name->id map for assertions.
    fn seed_named(
        project: &str,
        names: &[&str],
    ) -> (Store, std::collections::HashMap<String, i64>) {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: project.into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mut ids = std::collections::HashMap::new();
        for n in names {
            let id = s
                .insert_node(&NewNode {
                    project: project.into(),
                    label: "Function".into(),
                    name: (*n).into(),
                    qualified_name: format!("{project}::{n}"),
                    file_path: "src/lib.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            ids.insert((*n).to_string(), id);
        }
        (s, ids)
    }

    fn add_edge(s: &mut Store, project: &str, src: i64, tgt: i64, ty: &str) {
        s.insert_edge(&grepplus_store::NewEdge {
            project: project.into(),
            source_id: src,
            target_id: tgt,
            edge_type: ty.into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
    }

    #[test]
    fn dependency_cluster_collects_connected_imports_component() {
        // a -IMPORTS-> b -IMPORTS-> c form one cluster; d -IMPORTS-> e a
        // second, disjoint one. f is isolated.
        let (mut s, ids) = seed_named("p", &["a", "b", "c", "d", "e", "f"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "IMPORTS");
        add_edge(&mut s, "p", ids["b"], ids["c"], "IMPORTS");
        add_edge(&mut s, "p", ids["d"], ids["e"], "IMPORTS");

        // From b: undirected reach pulls in a (importer) and c (imported).
        let cluster = dependency_cluster(&s, ids["b"], 100).unwrap().unwrap();
        let names: Vec<&str> = cluster.nodes.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
        assert!(!cluster.truncated);
        // The second component is not pulled in.
        assert!(!cluster.nodes.iter().any(|r| r.name == "d"));

        // From f (isolated): the cluster is just f.
        let solo = dependency_cluster(&s, ids["f"], 100).unwrap().unwrap();
        assert_eq!(solo.nodes.len(), 1);
        assert_eq!(solo.nodes[0].name, "f");
    }

    #[test]
    fn dependency_cluster_ignores_other_edge_types() {
        // A CALLS edge must NOT widen an IMPORTS cluster.
        let (mut s, ids) = seed_named("p", &["a", "b", "c"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "IMPORTS");
        add_edge(&mut s, "p", ids["b"], ids["c"], "CALLS");
        let cluster = dependency_cluster(&s, ids["a"], 100).unwrap().unwrap();
        let names: Vec<&str> = cluster.nodes.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"], "CALLS must not extend the cluster");
    }

    #[test]
    fn dependency_cluster_respects_limit_and_flags_truncation() {
        // A chain a-b-c-d-e over IMPORTS; a limit of 2 stops early.
        let (mut s, ids) = seed_named("p", &["a", "b", "c", "d", "e"]);
        for w in ["a", "b", "c", "d", "e"].windows(2) {
            add_edge(&mut s, "p", ids[w[0]], ids[w[1]], "IMPORTS");
        }
        let cluster = dependency_cluster(&s, ids["a"], 2).unwrap().unwrap();
        assert_eq!(cluster.nodes.len(), 2);
        assert!(cluster.truncated, "limit hit -> truncated must be true");
    }

    #[test]
    fn dependency_cluster_missing_seed_is_none() {
        let (s, _ids) = seed_named("p", &["a"]);
        assert!(dependency_cluster(&s, 9999, 100).unwrap().is_none());
    }

    #[test]
    fn dependency_cluster_is_deterministic() {
        let (mut s, ids) = seed_named("p", &["a", "b", "c"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "IMPORTS");
        add_edge(&mut s, "p", ids["b"], ids["c"], "IMPORTS");
        let x = dependency_cluster(&s, ids["b"], 100).unwrap();
        let y = dependency_cluster(&s, ids["b"], 100).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn cycles_detects_a_simple_calls_cycle() {
        // a -> b -> c -> a is a 3-cycle; d -> e is acyclic.
        let (mut s, ids) = seed_named("p", &["a", "b", "c", "d", "e"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "CALLS");
        add_edge(&mut s, "p", ids["b"], ids["c"], "CALLS");
        add_edge(&mut s, "p", ids["c"], ids["a"], "CALLS");
        add_edge(&mut s, "p", ids["d"], ids["e"], "CALLS");

        let cy = cycles(&s, Some("p"), 8, 100).unwrap();
        assert_eq!(cy.len(), 1, "exactly one cycle expected: {cy:?}");
        let c = &cy[0];
        assert_eq!(c.len(), 3);
        // Canonical: starts at the smallest id on the cycle.
        let min_id = *[ids["a"], ids["b"], ids["c"]].iter().min().unwrap();
        assert_eq!(c.nodes[0], min_id);
        // The cycle contains exactly a, b, c.
        let mut got = c.nodes.clone();
        got.sort_unstable();
        let mut want = vec![ids["a"], ids["b"], ids["c"]];
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn cycles_reports_self_loop_as_length_one() {
        let (mut s, ids) = seed_named("p", &["a", "b"]);
        add_edge(&mut s, "p", ids["a"], ids["a"], "CALLS"); // self-loop
        add_edge(&mut s, "p", ids["a"], ids["b"], "CALLS");
        let cy = cycles(&s, Some("p"), 8, 100).unwrap();
        assert_eq!(cy.len(), 1);
        assert_eq!(cy[0].nodes, vec![ids["a"]]);
        assert_eq!(cy[0].len(), 1);
    }

    #[test]
    fn cycles_respects_max_len() {
        // A 4-cycle a->b->c->d->a. With max_len=3 it must NOT be reported.
        let (mut s, ids) = seed_named("p", &["a", "b", "c", "d"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "CALLS");
        add_edge(&mut s, "p", ids["b"], ids["c"], "CALLS");
        add_edge(&mut s, "p", ids["c"], ids["d"], "CALLS");
        add_edge(&mut s, "p", ids["d"], ids["a"], "CALLS");
        assert!(cycles(&s, Some("p"), 3, 100).unwrap().is_empty());
        // With max_len=4 it appears once.
        let cy = cycles(&s, Some("p"), 4, 100).unwrap();
        assert_eq!(cy.len(), 1);
        assert_eq!(cy[0].len(), 4);
    }

    #[test]
    fn cycles_deduplicates_and_canonicalises() {
        // Two overlapping cycles sharing node a:
        //   a -> b -> a   (2-cycle)
        //   a -> c -> a   (2-cycle)
        // Each must be reported exactly once, canonicalised to smallest-first.
        let (mut s, ids) = seed_named("p", &["a", "b", "c"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "CALLS");
        add_edge(&mut s, "p", ids["b"], ids["a"], "CALLS");
        add_edge(&mut s, "p", ids["a"], ids["c"], "CALLS");
        add_edge(&mut s, "p", ids["c"], ids["a"], "CALLS");
        let cy = cycles(&s, Some("p"), 8, 100).unwrap();
        assert_eq!(cy.len(), 2, "two distinct 2-cycles: {cy:?}");
        for c in &cy {
            assert_eq!(c.len(), 2);
            // smallest id first.
            assert!(c.nodes[0] <= c.nodes[1]);
        }
        // Sorted by (len, nodes lexicographically): both length 2, so by ids.
        assert!(cy[0].nodes <= cy[1].nodes);
    }

    #[test]
    fn cycles_acyclic_graph_yields_nothing() {
        let (mut s, ids) = seed_named("p", &["a", "b", "c"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "CALLS");
        add_edge(&mut s, "p", ids["b"], ids["c"], "CALLS");
        assert!(cycles(&s, Some("p"), 8, 100).unwrap().is_empty());
    }

    #[test]
    fn cycles_is_deterministic() {
        let (mut s, ids) = seed_named("p", &["a", "b", "c"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "CALLS");
        add_edge(&mut s, "p", ids["b"], ids["c"], "CALLS");
        add_edge(&mut s, "p", ids["c"], ids["a"], "CALLS");
        let x = cycles(&s, Some("p"), 8, 100).unwrap();
        let y = cycles(&s, Some("p"), 8, 100).unwrap();
        assert_eq!(x, y);
    }

    fn add_labeled_node(s: &mut Store, project: &str, label: &str, name: &str) -> i64 {
        s.insert_node(&NewNode {
            project: project.into(),
            label: label.into(),
            name: name.into(),
            qualified_name: format!("{project}::{name}"),
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap()
    }

    #[test]
    fn find_references_merges_all_reference_edge_types() {
        // target is CALLED by caller, USED by user, TYPE_REF'd by typer,
        // and IMPORTED by importer — find_references returns all four.
        let (mut s, ids) = seed_named(
            "p",
            &["target", "caller", "user", "typer", "importer", "unrelated"],
        );
        add_edge(&mut s, "p", ids["caller"], ids["target"], "CALLS");
        add_edge(&mut s, "p", ids["user"], ids["target"], "USES");
        add_edge(&mut s, "p", ids["typer"], ids["target"], "TYPE_REF");
        add_edge(&mut s, "p", ids["importer"], ids["target"], "IMPORTS");
        // An outgoing edge from target must NOT appear (we want incoming refs).
        add_edge(&mut s, "p", ids["target"], ids["unrelated"], "CALLS");

        let refs = find_references(&s, ids["target"], 100).unwrap();
        assert_eq!(refs.len(), 4);
        let mut by_node: std::collections::HashMap<i64, &str> = std::collections::HashMap::new();
        for r in &refs {
            by_node.insert(r.node.id, r.edge_type.as_str());
        }
        assert_eq!(by_node.get(&ids["caller"]).copied(), Some("CALLS"));
        assert_eq!(by_node.get(&ids["user"]).copied(), Some("USES"));
        assert_eq!(by_node.get(&ids["typer"]).copied(), Some("TYPE_REF"));
        assert_eq!(by_node.get(&ids["importer"]).copied(), Some("IMPORTS"));
        // `unrelated` is only an outgoing callee, never an incoming reference.
        assert!(!by_node.contains_key(&ids["unrelated"]));
    }

    #[test]
    fn count_references_ignores_display_limit_and_deduplicates() {
        let (mut s, ids) = seed_named("p", &["target", "caller", "user", "alias"]);
        add_edge(&mut s, "p", ids["caller"], ids["target"], "CALLS");
        add_edge(&mut s, "p", ids["user"], ids["target"], "USAGE");
        // Duplicate relation from the same source must count once.
        add_edge(&mut s, "p", ids["user"], ids["alias"], "USAGE");

        let shown = find_references_to_any(&s, &[ids["target"], ids["alias"]], 1).unwrap();
        let total = count_references_to_any(&s, "p", &[ids["target"], ids["alias"]]).unwrap();

        assert_eq!(shown.len(), 1);
        assert_eq!(total, 2);
        assert_eq!(count_references(&s, "p", ids["target"]).unwrap(), 2);
    }

    #[test]
    fn find_references_same_node_two_relations_yields_two_rows() {
        let (mut s, ids) = seed_named("p", &["target", "both"]);
        add_edge(&mut s, "p", ids["both"], ids["target"], "CALLS");
        add_edge(&mut s, "p", ids["both"], ids["target"], "USES");
        let refs = find_references(&s, ids["target"], 100).unwrap();
        assert_eq!(refs.len(), 2);
        let types: Vec<&str> = refs.iter().map(|r| r.edge_type.as_str()).collect();
        assert!(types.contains(&"CALLS"));
        assert!(types.contains(&"USES"));
        assert!(refs.iter().all(|r| r.node.id == ids["both"]));
    }

    #[test]
    fn find_references_excludes_self_reference() {
        let (mut s, ids) = seed_named("p", &["target"]);
        add_edge(&mut s, "p", ids["target"], ids["target"], "CALLS");
        let refs = find_references(&s, ids["target"], 100).unwrap();
        assert!(
            refs.is_empty(),
            "a self-reference is not an external reference"
        );
    }

    #[test]
    fn find_references_missing_target_and_zero_limit_are_empty() {
        let (mut s, ids) = seed_named("p", &["target", "caller"]);
        add_edge(&mut s, "p", ids["caller"], ids["target"], "CALLS");
        assert!(find_references(&s, 999_999, 100).unwrap().is_empty());
        assert_eq!(count_references(&s, "p", 999_999).unwrap(), 0);
        assert!(find_references(&s, ids["target"], 0).unwrap().is_empty());
    }

    #[test]
    fn find_references_is_sorted_and_deterministic() {
        let (mut s, ids) = seed_named("p", &["target", "zeta", "alpha"]);
        add_edge(&mut s, "p", ids["zeta"], ids["target"], "CALLS");
        add_edge(&mut s, "p", ids["alpha"], ids["target"], "USES");
        let first = find_references(&s, ids["target"], 100).unwrap();
        let second = find_references(&s, ids["target"], 100).unwrap();
        assert_eq!(first, second);
        // Sorted by qualified_name asc: p::alpha before p::zeta.
        assert_eq!(first[0].node.name, "alpha");
        assert_eq!(first[1].node.name, "zeta");
    }

    #[test]
    fn unused_symbols_flags_definitions_with_zero_incoming_refs() {
        // used <- caller (CALLS); orphan_a, orphan_b have no incoming refs.
        let (mut s, ids) = seed_named("p", &["used", "caller", "orphan_a", "orphan_b"]);
        add_edge(&mut s, "p", ids["caller"], ids["used"], "CALLS");
        let unused = unused_symbols(&s, "p", &[], 100).unwrap();
        let names: Vec<&str> = unused.iter().map(|r| r.name.as_str()).collect();
        // `used` is referenced; `caller` and the orphans are not.
        assert!(names.contains(&"orphan_a"));
        assert!(names.contains(&"orphan_b"));
        assert!(names.contains(&"caller"));
        assert!(!names.contains(&"used"));
        // Deterministic qualified_name ordering.
        let again = unused_symbols(&s, "p", &[], 100).unwrap();
        assert_eq!(unused, again);
    }

    #[test]
    fn unused_symbols_counts_every_reference_edge_type() {
        // Each target gets exactly one kind of incoming reference; none of
        // them should be flagged as unused.
        let (mut s, ids) = seed_named(
            "p",
            &["t_call", "t_use", "t_type", "t_import", "src", "lonely"],
        );
        add_edge(&mut s, "p", ids["src"], ids["t_call"], "CALLS");
        add_edge(&mut s, "p", ids["src"], ids["t_use"], "USES");
        add_edge(&mut s, "p", ids["src"], ids["t_type"], "TYPE_REF");
        add_edge(&mut s, "p", ids["src"], ids["t_import"], "IMPORTS");
        let unused = unused_symbols(&s, "p", &[], 100).unwrap();
        let names: Vec<&str> = unused.iter().map(|r| r.name.as_str()).collect();
        for referenced in ["t_call", "t_use", "t_type", "t_import"] {
            assert!(!names.contains(&referenced), "{referenced} is referenced");
        }
        // `src` has no incoming refs, `lonely` has none either.
        assert!(names.contains(&"src"));
        assert!(names.contains(&"lonely"));
    }

    #[test]
    fn unused_symbols_respects_label_filter() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let func = add_labeled_node(&mut s, "p", "Function", "orphan_fn");
        let _strukt = add_labeled_node(&mut s, "p", "Struct", "OrphanStruct");
        let _ = func;
        // Only Function-labelled unused defs.
        let unused = unused_symbols(&s, "p", &["Function"], 100).unwrap();
        let names: Vec<&str> = unused.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["orphan_fn"]);
        // Multi-label filter scans both kinds.
        let both = unused_symbols(&s, "p", &["Function", "Struct"], 100).unwrap();
        assert_eq!(both.len(), 2);
    }

    #[test]
    fn unused_symbols_empty_project_and_zero_limit_are_empty() {
        let (mut s, ids) = seed_named("p", &["a", "b"]);
        add_edge(&mut s, "p", ids["a"], ids["b"], "CALLS");
        assert!(unused_symbols(&s, "nonexistent", &[], 100)
            .unwrap()
            .is_empty());
        assert!(unused_symbols(&s, "p", &[], 0).unwrap().is_empty());
    }

    #[test]
    fn unused_symbols_is_complement_of_find_references() {
        let (mut s, ids) = seed_named("p", &["used", "caller", "orphan"]);
        add_edge(&mut s, "p", ids["caller"], ids["used"], "CALLS");
        let unused = unused_symbols(&s, "p", &[], 100).unwrap();
        for row in &unused {
            // Every "unused" node must have zero references.
            assert!(
                find_references(&s, row.id, 100).unwrap().is_empty(),
                "{} flagged unused but has references",
                row.name
            );
        }
        // And `used` (which has a reference) must NOT be in the unused set.
        assert!(!unused.iter().any(|r| r.id == ids["used"]));
    }
}
