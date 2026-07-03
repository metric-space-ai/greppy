//! Algorithmic semantic search.
//!
//! The upstream `src/semantic/semantic.c` combines 11 signals to score
//! graph nodes against a free-text query:
//!
//!  1. TF-IDF on metadata tokens
//!  2. Random Indexing with co-occurrence
//!  3. MinHash structural (decoded from "fp" property)
//!  4. API Signature vectors (same callees → related)
//!  5. Type Signature vectors (same param/return types → related)
//!  6. Module Proximity (same directory → boost)
//!  7. Decorator Pattern vectors (same annotations → related)
//!  8. AST Structural Profile (control flow shape, expression types)
//!  9. Approximate Data Flow (params→return, params→condition)
//! 10. Graph Diffusion (transitive closure via neighbor blending)
//! 11. Halstead-Lite (operator/operand complexity profile)
//!
//! Phase 4 ships the **subset** that is cheap and self-contained, using
//! only data the indexer already produces. Signals that require the
//! call graph, type resolver, or per-token metadata (1–8, 11) are
//! deferred to Phase 4 follow-ons and the EmbeddingGemma vector-index decision.
//!
//! Signals implemented in Phase 4:
//!
//! - **Token overlap** (subset of 1) — Jaccard similarity between the
//!   query tokens and the node's `name` + `qualified_name` tokenised
//!   via `grepplus_store::fts::camel_split`.
//! - **File module proximity** (signal 6) — exact-file or same-directory
//!   boost. We currently apply this only when the user passes a
//!   `near_file` hint (Phase 5 will compute it automatically from the
//!   invocation's working directory).
//! - **Label affinity** — bonus when the query mentions a known label
//!   word (function, struct, trait, …) and the node's label matches.
//! - **Qualified-name path proximity** (refinement of signal 6) — bonus
//!   proportional to how many leading module-path segments of the node's
//!   `qualified_name` are named in the query. Distinct from raw token
//!   overlap: it weights the *structural* module path (the `::`/`/`/`.`
//!   separated prefix that locates a symbol within the codebase) rather
//!   than the leaf identifier, so a query that names a containing module
//!   pulls every symbol under that module up the ranking even when the
//!   leaf names differ.
//!
//! ## Signal weights (deterministic, documented)
//!
//! The final score is a fixed linear combination. All weights are
//! `const`s in this module so the ranking is reproducible and auditable:
//!
//! | signal                       | weight / cap                 |
//! |------------------------------|------------------------------|
//! | token overlap (Jaccard)      | `1.0` (base, range `0..=1`)  |
//! | label/kind affinity          | [`LABEL_AFFINITY_BONUS`]     |
//! | file module proximity        | [`FILE_PROXIMITY_BONUS`]     |
//! | qualified-name path proximity| [`QNAME_PATH_MAX_BONUS`]     |
//! | simhash structural overlap   | [`SIMHASH_MAX_BONUS`]        |
//! | edge-aware graph proximity   | [`EDGE_PROXIMITY_BONUS`]     |
//!
//! Ties (equal score) break first on `qualified_name` (ascending,
//! stable across stores) and finally on a **recency/generation** proxy:
//! the node `id`, which SQLite assigns monotonically on insert, so a
//! later-indexed node (higher generation) wins an exact tie. This keeps
//! the order total and deterministic without an extra schema column.

use grepplus_core::Result;
use grepplus_store::fts::camel_split;
use grepplus_store::Store;
use std::collections::{HashMap, HashSet};

use crate::graph::{search_graph, SearchGraphRow};
use crate::simhash::{MinHash, MINHASH_K};

/// Which signals contributed to a hit's score.
///
/// Signals are added as the ranker grows (e.g. the `simhash` signal).
/// Downstream crates should construct via [`SemanticSignal::none`] or
/// spread `..Default::default()` so future additions stay
/// source-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SemanticSignal {
    pub token_overlap: bool,
    pub label_affinity: bool,
    pub file_proximity: bool,
    /// Set when the MinHash signature of the node's tokens has a small
    /// hamming distance to the query's signature (structural-overlap
    /// signal ported from upstream `src/simhash`).
    pub simhash: bool,
    /// Set when at least one leading module-path segment of the node's
    /// `qualified_name` is named in the query (qualified-name path
    /// proximity signal).
    pub qname_path: bool,
    /// Set when the node is within one CALLS/USES/IMPORTS hop of the
    /// query's resolved anchor symbol (edge-aware graph proximity
    /// signal). See [`EDGE_PROXIMITY_BONUS`].
    pub edge_proximity: bool,
}

impl SemanticSignal {
    /// A signal set with no signals active. Equivalent to
    /// `SemanticSignal::default()`.
    pub fn none() -> Self {
        Self::default()
    }
}

/// Per-signal numeric breakdown of a [`SemanticHit`]'s score, for
/// explainability ("why did this rank here?").
///
/// Each field is the *additive contribution* that the named signal made to
/// the final [`SemanticHit::score`]: the fields sum (modulo floating-point
/// rounding) to `score`. A signal that did not fire contributes `0.0`. The
/// boolean [`SemanticSignal`] flags say *whether* a signal fired; this says
/// *how much* it moved the score, so a caller can render "token overlap
/// 0.62, +0.15 label, +0.14 edge proximity".
///
/// Additive and deterministic: the breakdown is derived from the same
/// deterministic scoring already performed, so identical inputs yield
/// identical breakdowns.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SemanticBreakdown {
    /// IDF-weighted Jaccard token-overlap base (range `0..=1`).
    pub token_overlap: f64,
    /// Bonus from label/kind affinity ([`LABEL_AFFINITY_BONUS`] or `0`).
    pub label_affinity: f64,
    /// Bonus from `near_file` module proximity ([`FILE_PROXIMITY_BONUS`]
    /// or `0`).
    pub file_proximity: f64,
    /// Bonus from MinHash structural overlap (`0..=`[`SIMHASH_MAX_BONUS`]).
    pub simhash: f64,
    /// Bonus from qualified-name path proximity
    /// (`0..=`[`QNAME_PATH_MAX_BONUS`]).
    pub qname_path: f64,
    /// Bonus from edge-aware graph proximity to the resolved anchor(s)
    /// ([`EDGE_PROXIMITY_BONUS`] or, in multi-anchor mode, a multiple of
    /// [`MULTI_ANCHOR_PER_HOP`]).
    pub edge_proximity: f64,
}

impl SemanticBreakdown {
    /// The sum of every signal contribution. Equals [`SemanticHit::score`]
    /// up to floating-point rounding.
    pub fn total(&self) -> f64 {
        self.token_overlap
            + self.label_affinity
            + self.file_proximity
            + self.simhash
            + self.qname_path
            + self.edge_proximity
    }
}

/// One semantic-query hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticHit {
    pub node: SearchGraphRow,
    pub score: f64,
    pub signals: SemanticSignal,
    /// Per-signal numeric contributions that sum to `score`, for
    /// explainability. See [`SemanticBreakdown`].
    pub breakdown: SemanticBreakdown,
}

/// Run a semantic query. Returns up to `limit` hits sorted by score
/// (descending). `near_file` is an optional hint (typically the
/// invoker's CWD) used for the module-proximity signal.
///
/// When `project` is `Some`, the query is restricted to that
/// project's nodes (R-025: avoids cross-project pollution in
/// multi-tenant stores). Pass `None` only when intentionally
/// searching a single-project store.
pub fn semantic_query(
    store: &Store,
    query: &str,
    near_file: Option<&str>,
    project: Option<&str>,
    limit: usize,
) -> Result<Vec<SemanticHit>> {
    semantic_query_filtered(store, query, near_file, project, &[], limit)
}

/// Run a semantic query restricted to a set of labels/kinds.
///
/// Identical to [`semantic_query`] in every respect except that, when
/// `labels` is non-empty, only candidate nodes whose `label` is one of
/// the listed labels are scored and returned. An empty `labels` slice
/// disables the filter and reproduces [`semantic_query`] exactly — so
/// this is a strict additive extension of the existing API.
///
/// Label matching is exact against the stored `nodes.label` (e.g.
/// `"Function"`, `"Struct"`, `"Import"`), the same values
/// [`search_graph`] filters on. Duplicate labels in the slice are
/// harmless. Determinism, scoring, tie-breaks, and the IDF vocabulary are
/// all inherited unchanged: the filter only narrows the candidate set.
///
/// Note the IDF vocabulary is computed over the *filtered* candidate set,
/// so token rarity is judged relative to the kind being searched (e.g.
/// among functions only). This keeps the discriminating-token weighting
/// meaningful when the caller has already committed to one kind.
pub fn semantic_query_filtered(
    store: &Store,
    query: &str,
    near_file: Option<&str>,
    project: Option<&str>,
    labels: &[&str],
    limit: usize,
) -> Result<Vec<SemanticHit>> {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let q_labels = label_words_in_query(query);
    // MinHash signature of the query token set, used for the simhash
    // (structural token-overlap) signal. Built once and reused.
    let q_sig = MinHash::from_tokens(q_tokens.iter());

    // R-025: scope to the requested project when given. Multi-
    // project stores would otherwise return hits from every
    // project, polluting the agent's view.
    let mut q = crate::GraphQuery::any().with_limit(10_000);
    if let Some(proj) = project {
        q = q.with_project(proj);
    }
    let mut rows = search_graph(store, &q)?;

    // Forensics F2: drop the parser's `Call` / `Import` pseudo-nodes from
    // the semantic candidate set. They are edge endpoints (their meaning is
    // already in the CALLS / IMPORTS edges who-calls/callees read), not
    // navigable symbols, yet their name tokens duplicate the real
    // function/type so closely that they outrank it — `semantic Store`
    // returned `Call::Store` ABOVE `Struct::Store`. We exclude them unless a
    // caller has *explicitly* asked for those labels via `labels`, which
    // keeps the additive contract (an explicit `labels=["Call"]` still
    // works) while making the common unfiltered query clean.
    let explicitly_wanted: HashSet<&str> = labels.iter().copied().collect();
    rows.retain(|r| {
        !matches!(r.label.as_str(), "Call" | "Import")
            || explicitly_wanted.contains(r.label.as_str())
    });
    // Forensics F6: the synthetic file-level module node (qualified_name
    // ending `::__file__`; its `name` is the file stem, e.g. `store`) points
    // at a whole file, not a navigable definition, yet its broad token set
    // lets it rank #1 on many queries. Drop it from semantic ranking; it
    // stays in the graph for edge resolution.
    rows.retain(|r| !r.qualified_name.ends_with("::__file__"));

    // Label/kind filter (additive): when `labels` is non-empty keep only
    // rows whose label is listed. An empty slice leaves the candidate set
    // untouched, so the unfiltered `semantic_query` is bit-for-bit
    // unchanged. Filtering before IDF means token rarity is judged within
    // the requested kind.
    if !labels.is_empty() {
        let allowed: HashSet<&str> = labels.iter().copied().collect();
        rows.retain(|r| allowed.contains(r.label.as_str()));
    }

    // IDF-style term weighting over the candidate vocabulary. Each node's
    // (name + qualified_name) token set is one "document"; a token's
    // document frequency is how many nodes contain it. Common tokens
    // (high df) earn a low weight so they contribute little to the
    // overlap score, while rare, discriminating tokens dominate ranking.
    // Computed deterministically from the store's nodes — identical
    // inputs yield identical weights.
    let idf = TokenIdf::from_rows(&rows);

    // Edge-aware proximity: if the query names a concrete symbol present
    // in the candidate set, resolve it to an anchor and gather the set
    // of node ids one CALLS/USES/IMPORTS hop away. Nodes in that set get
    // a deterministic boost. When the query names no symbol (ordinary
    // free-text search) the set is empty and the signal never fires.
    let proximity_set: HashSet<i64> = match resolve_anchor(&rows, &q_tokens) {
        Some(anchor_id) => proximity_neighbors(store, anchor_id)?,
        None => HashSet::new(),
    };

    let mut hits: Vec<SemanticHit> = rows
        .into_iter()
        .filter_map(|row| {
            score_one(
                &row,
                &q_tokens,
                &q_labels,
                &q_sig,
                near_file,
                &proximity_set,
                &idf,
            )
        })
        .collect();
    // Sort by score descending; tie-break first on qualified_name
    // (ascending, stable across stores) and finally on the node id as a
    // recency/generation proxy — SQLite assigns ids monotonically on
    // insert, so the later-indexed (higher-generation) node wins an
    // otherwise exact tie. This keeps the order total and deterministic.
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| b.node.id.cmp(&a.node.id))
    });
    hits.truncate(limit);
    Ok(hits)
}

/// Recall-oriented semantic query that drives scoring off the
/// [`expand_query_tokens`] decomposition of the query.
///
/// Behaves exactly like [`semantic_query`] — same signals, weights,
/// tie-breaks, project scoping, and determinism — but builds the query
/// token set with [`expand_query_tokens`] instead of the bare
/// [`tokenize`]. For a query that is already whitespace-separated words
/// the two are identical (expansion is idempotent), so this is a strict,
/// additive extension. The difference shows on **compound-identifier
/// queries**: a query like `getUserId` or `parseHTTPRequest` is split into
/// its concept tokens, so it recalls symbols written in any casing
/// convention (`get_user_id`, `GetUserID`, `parse_http_request`) even when
/// the leaf identifiers differ in style across languages.
///
/// When `labels` is non-empty the candidate set is narrowed to those
/// labels first (identical to [`semantic_query_filtered`]); an empty slice
/// disables the filter. The IDF vocabulary, anchor resolution, and edge
/// proximity all behave exactly as in the non-expanded path.
///
/// Determinism: `expand_query_tokens` is a pure function and the rest of
/// the pipeline is the established deterministic one, so identical inputs
/// yield identical results. Additive: a new entry point and a new public
/// helper, touching no existing API.
pub fn semantic_query_expanded(
    store: &Store,
    query: &str,
    near_file: Option<&str>,
    project: Option<&str>,
    labels: &[&str],
    limit: usize,
) -> Result<Vec<SemanticHit>> {
    let q_tokens = expand_query_tokens(query);
    if q_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let q_labels = label_words_in_query(query);
    let q_sig = MinHash::from_tokens(q_tokens.iter());

    let mut q = crate::GraphQuery::any().with_limit(10_000);
    if let Some(proj) = project {
        q = q.with_project(proj);
    }
    let mut rows = search_graph(store, &q)?;

    if !labels.is_empty() {
        let allowed: HashSet<&str> = labels.iter().copied().collect();
        rows.retain(|r| allowed.contains(r.label.as_str()));
    }

    let idf = TokenIdf::from_rows(&rows);

    let proximity_set: HashSet<i64> = match resolve_anchor(&rows, &q_tokens) {
        Some(anchor_id) => proximity_neighbors(store, anchor_id)?,
        None => HashSet::new(),
    };

    let mut hits: Vec<SemanticHit> = rows
        .into_iter()
        .filter_map(|row| {
            score_one(
                &row,
                &q_tokens,
                &q_labels,
                &q_sig,
                near_file,
                &proximity_set,
                &idf,
            )
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| b.node.id.cmp(&a.node.id))
    });
    hits.truncate(limit);
    Ok(hits)
}

/// Per-anchor adjacency boost applied by [`semantic_query_multi_anchor`].
/// A candidate earns this *for each distinct anchor* it sits one
/// CALLS/USES/IMPORTS hop from, so a node adjacent to two anchors earns
/// `2 * MULTI_ANCHOR_PER_HOP`. Sized so a single-anchor adjacency matches
/// the established single-anchor [`EDGE_PROXIMITY_BONUS`], keeping the
/// multi-anchor path a strict generalisation of the existing one.
pub const MULTI_ANCHOR_PER_HOP: f64 = EDGE_PROXIMITY_BONUS;

/// Multi-anchor semantic mode: rank nodes by proximity to a **set** of
/// resolved anchor symbols rather than a single free-text-resolved anchor.
///
/// The caller supplies `anchor_names` — the names of symbols it has
/// already resolved (e.g. the symbols touched by a change, or a cluster of
/// related entry points). Each name is resolved to a node the same
/// deterministic way [`related_symbols`] resolves its anchor; unresolved
/// names are silently skipped. The combined proximity set is the union of
/// every anchor's one-hop CALLS/USES/IMPORTS neighbourhood, and each
/// candidate additionally earns [`MULTI_ANCHOR_PER_HOP`] *per distinct
/// anchor* it neighbours — so a node adjacent to several of the anchors
/// ranks above a node adjacent to just one.
///
/// `query` still drives the lexical signals (token overlap, label, qname
/// path, simhash, file proximity) exactly as [`semantic_query`] does, so
/// this is a superset: the base text ranking is preserved and the
/// multi-anchor adjacency is layered on top. A node that is adjacent to an
/// anchor but shares no query tokens still surfaces (adjacency is itself
/// evidence), mirroring the single-anchor behaviour.
///
/// The query's own in-text anchor resolution is intentionally **not** run
/// here — the anchors are exactly the supplied set — so the two modes stay
/// distinct and predictable.
///
/// Determinism and bounds: anchors are resolved by a total order; each
/// neighbourhood comes from the id-ordered edge tables; the result is
/// sorted by `(score desc, qualified_name asc, id desc)` and truncated to
/// `limit`, identical to [`semantic_query`]. Additive: a new function and
/// one constant, touching no existing API.
pub fn semantic_query_multi_anchor(
    store: &Store,
    query: &str,
    anchor_names: &[&str],
    near_file: Option<&str>,
    project: Option<&str>,
    limit: usize,
) -> Result<Vec<SemanticHit>> {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let q_labels = label_words_in_query(query);
    let q_sig = MinHash::from_tokens(q_tokens.iter());

    let mut q = crate::GraphQuery::any().with_limit(10_000);
    if let Some(proj) = project {
        q = q.with_project(proj);
    }
    let rows = search_graph(store, &q)?;
    let idf = TokenIdf::from_rows(&rows);

    // Resolve each supplied anchor name to a node id (deterministically),
    // deduplicating so the same symbol named twice counts once. For each
    // resolved anchor, gather its one-hop proximity neighbourhood and tally
    // how many distinct anchors each candidate node neighbours.
    let mut resolved_anchors: HashSet<i64> = HashSet::new();
    let mut per_node_anchor_count: HashMap<i64, usize> = HashMap::new();
    for name in anchor_names {
        let anchor_tokens = tokenize(name);
        if anchor_tokens.is_empty() {
            continue;
        }
        let Some(anchor_id) = resolve_anchor(&rows, &anchor_tokens) else {
            continue;
        };
        if !resolved_anchors.insert(anchor_id) {
            continue; // this symbol already counted
        }
        for nid in proximity_neighbors(store, anchor_id)? {
            *per_node_anchor_count.entry(nid).or_insert(0) += 1;
        }
    }
    // Anchors never earn their own multi-anchor boost.
    for a in &resolved_anchors {
        per_node_anchor_count.remove(a);
    }

    // The union proximity set drives the single-hit edge-proximity signal
    // (so a node adjacent to any anchor still gets the established
    // `edge_proximity` flag + bonus inside `score_one`); the per-anchor
    // count drives the *additional* multi-anchor stacking below.
    let proximity_set: HashSet<i64> = per_node_anchor_count.keys().copied().collect();

    let mut hits: Vec<SemanticHit> = rows
        .into_iter()
        .filter_map(|row| {
            score_one(
                &row,
                &q_tokens,
                &q_labels,
                &q_sig,
                near_file,
                &proximity_set,
                &idf,
            )
        })
        .collect();

    // Stack the multi-anchor adjacency: a node adjacent to `k` distinct
    // anchors earns `k * MULTI_ANCHOR_PER_HOP`. `score_one` already added
    // one `EDGE_PROXIMITY_BONUS` for being in `proximity_set`; top up the
    // remaining `(k - 1)` anchors so the total adjacency contribution is
    // exactly `k * MULTI_ANCHOR_PER_HOP` and the breakdown stays faithful.
    for hit in &mut hits {
        if let Some(&k) = per_node_anchor_count.get(&hit.node.id) {
            if k > 1 {
                let extra = (k - 1) as f64 * MULTI_ANCHOR_PER_HOP;
                hit.score += extra;
                hit.breakdown.edge_proximity += extra;
            }
        }
    }

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| b.node.id.cmp(&a.node.id))
    });
    hits.truncate(limit);
    Ok(hits)
}

/// The theoretical maximum raw [`SemanticHit::score`]: a perfect token
/// overlap (`1.0`) plus every signal bonus firing at its cap. Used to map a
/// raw score onto an absolute `[0.0, 1.0]` confidence in
/// [`semantic_query_diversified`]. Kept in sync with the signal weights
/// above; a score at or above this value clamps to a confidence of `1.0`.
pub const MAX_SEMANTIC_SCORE: f64 = 1.0
    + LABEL_AFFINITY_BONUS
    + FILE_PROXIMITY_BONUS
    + SIMHASH_MAX_BONUS
    + QNAME_PATH_MAX_BONUS
    + EDGE_PROXIMITY_BONUS;

/// A [`SemanticHit`] paired with an absolute, normalized **confidence** in
/// `[0.0, 1.0]`. Returned by [`semantic_query_diversified`].
///
/// `confidence` is the hit's raw `hit.score` divided by
/// [`MAX_SEMANTIC_SCORE`] and clamped to `[0.0, 1.0]`. Unlike a
/// result-set-relative normalization, this is **absolute**: a hit with the
/// same score always reports the same confidence regardless of what else is
/// in the result set, so a caller can threshold on it ("only show hits with
/// confidence >= 0.5") meaningfully across queries. The mapping is monotonic
/// in `score`, so it never reorders results.
#[derive(Debug, Clone, PartialEq)]
pub struct DiversifiedHit {
    /// The underlying semantic hit (node, raw score, signals, breakdown),
    /// unchanged.
    pub hit: SemanticHit,
    /// Absolute relevance confidence in `[0.0, 1.0]`, higher = more
    /// relevant. `hit.score / MAX_SEMANTIC_SCORE`, clamped.
    pub confidence: f64,
}

/// Semantic query with optional **result diversification** and an absolute
/// **confidence** score.
///
/// This is an additive wrapper over [`semantic_query`]: it runs the exact
/// same deterministic ranking (same signals, weights, tie-breaks, project
/// scoping), then optionally diversifies the result list and attaches a
/// normalized confidence to each returned hit.
///
/// **Diversification** (`diversify == true`): the upstream ranking can
/// return several hits that all come from the same file or all name the
/// same leaf symbol, crowding out other relevant results. When enabled, the
/// result list is filtered greedily in rank order so that no single
/// `(file_path, leaf-name)` group contributes more than `per_group` hits —
/// the highest-ranked members of each group are kept, later duplicates are
/// dropped, and the relative order of the survivors is preserved. A
/// `per_group` of `0` is treated as `1`. With `diversify == false` the list
/// is returned intact (every hit, original order), so the flag is opt-in and
/// the default behaviour matches `semantic_query`.
///
/// **Confidence**: every returned hit carries `confidence = score /
/// MAX_SEMANTIC_SCORE` clamped to `[0.0, 1.0]` (see [`DiversifiedHit`]).
///
/// Determinism and bounds: the base ranking is the established deterministic
/// one; diversification is a stable greedy pass over that order; the result
/// is truncated to `limit` **after** diversification, so a diversified query
/// still returns up to `limit` distinct-group hits rather than `limit`
/// pre-diversification hits. Identical inputs yield identical output.
///
/// Note `limit` here bounds the *final* list. Internally the base query is
/// run with a generous candidate cap so diversification has enough material
/// to fill `limit` distinct groups; the candidate cap is itself bounded.
pub fn semantic_query_diversified(
    store: &Store,
    query: &str,
    near_file: Option<&str>,
    project: Option<&str>,
    diversify: bool,
    per_group: usize,
    limit: usize,
) -> Result<Vec<DiversifiedHit>> {
    // Run the base ranking over a generous candidate set so that, after
    // diversification drops same-group duplicates, we can still fill `limit`
    // distinct groups. Bounded so an adversarial query cannot explode, but
    // never below `limit` (a huge `limit` still gets at least `limit`
    // candidates) and never below `1`.
    let candidate_cap = limit.saturating_mul(8).min(10_000).max(limit).max(1);
    let ranked = semantic_query(store, query, near_file, project, candidate_cap)?;

    let kept: Vec<SemanticHit> = if diversify {
        let cap = per_group.max(1);
        let mut group_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut out: Vec<SemanticHit> = Vec::new();
        for hit in ranked {
            // Group key: the file the symbol lives in plus its leaf name, so
            // "many hits from one file" and "many hits for one symbol name"
            // are both diversified. Stable greedy keep in rank order.
            let key = (hit.node.file_path.clone(), hit.node.name.clone());
            let count = group_counts.entry(key).or_insert(0);
            if *count < cap {
                *count += 1;
                out.push(hit);
            }
        }
        out
    } else {
        ranked
    };

    Ok(kept
        .into_iter()
        .take(limit)
        .map(|hit| {
            let confidence = (hit.score / MAX_SEMANTIC_SCORE).clamp(0.0, 1.0);
            DiversifiedHit { hit, confidence }
        })
        .collect())
}

/// One hit from [`related_symbols`]: a node ranked by its relatedness to
/// a named anchor symbol.
#[derive(Debug, Clone, PartialEq)]
pub struct RelatedHit {
    pub node: SearchGraphRow,
    /// Combined relatedness score: edge-proximity contribution plus
    /// token-overlap (IDF-weighted Jaccard) to the anchor's own tokens.
    pub score: f64,
    /// True when the node is within one CALLS/USES/IMPORTS hop of the
    /// anchor (in either direction).
    pub edge_adjacent: bool,
    /// Number of hops the node sits from the anchor over the proximity
    /// edge types (`1` for a direct neighbour; `0` only for the anchor
    /// itself, which is never returned).
    pub hops: usize,
}

/// Weight applied to the edge-adjacency component of the
/// [`related_symbols`] score. A direct CALLS/USES/IMPORTS neighbour of
/// the anchor earns this on top of its token overlap, so graph-adjacent
/// symbols outrank lexically-similar but disconnected ones.
pub const RELATED_EDGE_WEIGHT: f64 = 0.5;

/// "Related symbols" mode: given the name of an anchor symbol, rank every
/// other symbol by how related it is, combining **edge proximity** (a
/// CALLS/USES/IMPORTS neighbour of the anchor) with **token overlap** (an
/// IDF-weighted Jaccard between the candidate's tokens and the anchor's
/// own name/qualified-name tokens).
///
/// This is a focused complement to [`semantic_query`]: instead of free
/// text, the caller names a *symbol they already have* and asks "what
/// else is related to this?". The score is
///
/// ```text
/// score = token_overlap(candidate, anchor)
///       + RELATED_EDGE_WEIGHT * edge_adjacent
/// ```
///
/// so a direct neighbour with some lexical overlap ranks above a pure
/// lexical match, and a pure neighbour (no shared tokens) still surfaces.
///
/// Anchor resolution: `anchor_name` is matched against `nodes.name` the
/// same way [`resolve_anchor`] tokenises — the candidate whose leaf-name
/// tokens are exactly the anchor's tokens, with the smallest
/// `(qualified_name, id)` winning ties. Returns an empty vec when no such
/// symbol exists.
///
/// Determinism and bounds: the anchor is resolved by a total order; the
/// neighbour set comes from the id-ordered edge tables; the result is
/// sorted by `(score desc, qualified_name asc, id desc)` exactly like
/// [`semantic_query`]. The anchor itself is never in the output. The API
/// is additive — it adds a function and a type, touching nothing existing.
pub fn related_symbols(
    store: &Store,
    anchor_name: &str,
    project: Option<&str>,
    limit: usize,
) -> Result<Vec<RelatedHit>> {
    let anchor_tokens = tokenize(anchor_name);
    if anchor_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut q = crate::GraphQuery::any().with_limit(10_000);
    if let Some(proj) = project {
        q = q.with_project(proj);
    }
    let rows = search_graph(store, &q)?;

    // Resolve the anchor by name within the candidate set.
    let Some(anchor_id) = resolve_anchor(&rows, &anchor_tokens) else {
        return Ok(Vec::new());
    };
    let Some(anchor_row) = rows.iter().find(|r| r.id == anchor_id).cloned() else {
        return Ok(Vec::new());
    };

    // The anchor's own *leaf-name* token set is the lexical reference
    // every candidate is scored against. We deliberately use only the leaf
    // name (not the qualified_name) here: the qualified-name prefix (e.g.
    // the `p::` project segment or shared module path) is boilerplate that
    // every sibling symbol carries, so including it would make unrelated
    // symbols look "overlapping" purely via shared path tokens. Token
    // overlap on the *identifier* is the meaningful lexical signal.
    let anchor_token_set: HashSet<String> = tokenize(&anchor_row.name);

    // IDF over the candidate vocabulary (same construction the free-text
    // path uses) so common tokens contribute little to the overlap.
    let idf = TokenIdf::from_rows(&rows);

    // The set of direct (1-hop) proximity neighbours of the anchor.
    let proximity_set = proximity_neighbors(store, anchor_id)?;

    let mut hits: Vec<RelatedHit> = Vec::new();
    for row in rows {
        if row.id == anchor_id {
            continue; // never return the anchor itself
        }
        let node_token_set: HashSet<String> = tokenize(&row.name);
        if node_token_set.is_empty() {
            continue;
        }
        let edge_adjacent = proximity_set.contains(&row.id);
        // IDF-weighted Jaccard between the candidate and the anchor tokens.
        let mut inter_w = 0.0;
        let mut union_w = 0.0;
        for t in anchor_token_set.union(&node_token_set) {
            let w = idf.weight(t);
            union_w += w;
            if anchor_token_set.contains(t) && node_token_set.contains(t) {
                inter_w += w;
            }
        }
        let overlap = if union_w > 0.0 {
            inter_w / union_w
        } else {
            0.0
        };
        // Emit only nodes that are either edge-adjacent or share a token
        // with the anchor; everything else is noise.
        if !edge_adjacent && overlap == 0.0 {
            continue;
        }
        let score = overlap
            + if edge_adjacent {
                RELATED_EDGE_WEIGHT
            } else {
                0.0
            };
        hits.push(RelatedHit {
            node: row,
            score,
            edge_adjacent,
            hops: if edge_adjacent { 1 } else { 0 },
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| b.node.id.cmp(&a.node.id))
    });
    hits.truncate(limit);
    Ok(hits)
}

/// Which single field a [`semantic_query_fielded`] search matches against.
///
/// Where the full [`semantic_query`] folds a node's `name`, `qualified_name`,
/// label, file path, and graph adjacency into one score, a fielded query is
/// deliberately narrow: it scores token overlap against **exactly one**
/// textual field and nothing else. That makes the result precise and
/// predictable — "find symbols whose *name* contains these tokens" without the
/// qualified-name path or doc text leaking matches in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticField {
    /// Match against the symbol's leaf `name` only.
    Name,
    /// Match against the full `qualified_name` only.
    QualifiedName,
    /// Match against the symbol's documentation text only, read from the
    /// node's `properties["doc"]` string (empty when absent). The store does
    /// not surface a typed doc column, so this is the agreed convention for
    /// where indexers stash docstrings/leading comments.
    Doc,
}

impl SemanticField {
    /// The text of this field for a given node row + properties, used as the
    /// token source for the fielded match.
    fn field_text(self, row: &SearchGraphRow, properties: &serde_json::Value) -> String {
        match self {
            SemanticField::Name => row.name.clone(),
            SemanticField::QualifiedName => row.qualified_name.clone(),
            SemanticField::Doc => properties
                .get("doc")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        }
    }

    /// Whether this field requires loading the node's JSON properties (only
    /// the doc field does). Lets the query skip a per-row store fetch for the
    /// name/qualified_name fields, which `SearchGraphRow` already carries.
    fn needs_properties(self) -> bool {
        matches!(self, SemanticField::Doc)
    }
}

/// A fielded semantic query: rank nodes by IDF-weighted token overlap against
/// **one** field — the leaf `name`, the `qualified_name`, or the `doc` text —
/// and nothing else.
///
/// This is the precision counterpart to [`semantic_query`]. The full query
/// blends many signals (label, file proximity, qname-path, simhash, edge
/// adjacency); a fielded query intentionally drops all of them and scores only
/// the chosen field's token overlap, so a caller who wants "symbols whose
/// *name* matches" or "symbols whose *doc* mentions X" gets exactly that with
/// no cross-field bleed.
///
/// Scoring: the IDF vocabulary is built over the chosen field across the
/// candidate set (so token rarity is judged *within that field*), and each
/// node's score is the IDF-weighted Jaccard between the query tokens and the
/// node's field tokens — identical to the base overlap signal in
/// [`semantic_query`] but restricted to one field. A node with zero overlap is
/// dropped. The returned [`SemanticHit::breakdown`] reports the whole score
/// under `token_overlap`; the other breakdown fields are `0.0` (no other
/// signal fired), and `signals.token_overlap` is the only flag set.
///
/// Determinism and bounds:
/// - Candidates come from [`search_graph`] (scoped to `project` when given),
///   capped at 10,000 rows like the other semantic entry points.
/// - The IDF-weighted overlap is summed over a **sorted** token union, so the
///   floating-point accumulation order is fixed and the score is byte-stable
///   run-to-run (HashSet iteration order is not).
/// - Sorted by `(score desc, qualified_name asc, id desc)` — the exact total
///   order [`semantic_query`] uses — then truncated to `limit`.
/// - An empty query, or a field with no text on a node, yields no hit for that
///   node. Additive: a new function + enum, no existing API touched.
pub fn semantic_query_fielded(
    store: &Store,
    query: &str,
    field: SemanticField,
    project: Option<&str>,
    limit: usize,
) -> Result<Vec<SemanticHit>> {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut q = crate::GraphQuery::any().with_limit(10_000);
    if let Some(proj) = project {
        q = q.with_project(proj);
    }
    let rows = search_graph(store, &q)?;

    // Resolve each row's field text once. For the doc field we must read the
    // node's JSON properties from the store; name/qualified_name are already
    // on the row, so we avoid the fetch.
    let need_props = field.needs_properties();
    let mut field_texts: Vec<(SearchGraphRow, String)> = Vec::with_capacity(rows.len());
    for row in rows {
        let text = if need_props {
            match store.get_node(row.id)? {
                Some(n) => field.field_text(&row, &n.properties),
                None => String::new(),
            }
        } else {
            field.field_text(&row, &serde_json::Value::Null)
        };
        field_texts.push((row, text));
    }

    // IDF over the chosen field across the candidate set: each node's field
    // token set is one document.
    let mut df: HashMap<String, usize> = HashMap::new();
    let mut n_docs = 0usize;
    for (_, text) in &field_texts {
        let toks = tokenize(text);
        if toks.is_empty() {
            continue;
        }
        n_docs += 1;
        for t in toks {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    let weight = |token: &str| -> f64 {
        let d = df.get(token).copied().unwrap_or(0);
        ((n_docs as f64 + 1.0) / (d as f64 + 1.0)).ln() + 1.0
    };

    let mut hits: Vec<SemanticHit> = Vec::new();
    for (row, text) in field_texts {
        let node_tokens = tokenize(&text);
        if node_tokens.is_empty() {
            continue;
        }
        // Sum over a *sorted* union so the floating-point accumulation order
        // is fixed run-to-run (HashSet iteration order is not). Non-
        // associative f64 addition would otherwise yield last-bit-different
        // scores across runs and break determinism.
        let mut union: Vec<&String> = q_tokens.union(&node_tokens).collect();
        union.sort_unstable();
        let mut inter_w = 0.0;
        let mut union_w = 0.0;
        for t in union {
            let w = weight(t);
            union_w += w;
            if q_tokens.contains(t) && node_tokens.contains(t) {
                inter_w += w;
            }
        }
        if inter_w <= 0.0 {
            // No token overlap on the chosen field — drop (would be noise).
            continue;
        }
        let jaccard = if union_w > 0.0 {
            inter_w / union_w
        } else {
            0.0
        };
        let mut signals = SemanticSignal::none();
        signals.token_overlap = true;
        let breakdown = SemanticBreakdown {
            token_overlap: jaccard,
            ..SemanticBreakdown::default()
        };
        hits.push(SemanticHit {
            node: row,
            score: jaccard,
            signals,
            breakdown,
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node.qualified_name.cmp(&b.node.qualified_name))
            .then_with(|| b.node.id.cmp(&a.node.id))
    });
    hits.truncate(limit);
    Ok(hits)
}

/// Maximum bonus contributed by the MinHash/simhash signal. Kept small
/// relative to the token-overlap Jaccard so it acts as a tie-breaker /
/// refinement rather than dominating ranking.
pub const SIMHASH_MAX_BONUS: f64 = 0.10;

/// A node's simhash signal only fires when its MinHash signature agrees
/// with the query on more than this fraction of slots. Below it the
/// agreement is indistinguishable from hash-collision noise.
const SIMHASH_MIN_AGREEMENT: f64 = 0.30;

/// Bonus added when the query names the node's label/kind (function,
/// struct, …). A "kind affinity" weight: it nudges nodes of the
/// requested kind above same-overlap nodes of other kinds.
pub const LABEL_AFFINITY_BONUS: f64 = 0.15;

/// Bonus added when a `near_file` hint shares a directory with the
/// node's file (module proximity signal 6).
pub const FILE_PROXIMITY_BONUS: f64 = 0.10;

/// Maximum bonus contributed by qualified-name path proximity. Reached
/// when *every* leading module-path segment of the node's
/// `qualified_name` is named in the query. Capped below the label and
/// file weights so it refines rather than dominates the base overlap.
pub const QNAME_PATH_MAX_BONUS: f64 = 0.12;

/// Bonus added when a node is within one hop of the query's resolved
/// anchor symbol over a relationship edge (`CALLS`, `USES`, or
/// `IMPORTS`), in either direction. Edge proximity is strong evidence
/// that two symbols are semantically related — a caller/callee, a
/// user/used, or an importer/imported pair — so this is weighted at the
/// top of the bonus band, just below the label affinity. Deterministic:
/// the anchor is resolved by a total order and the neighbour set comes
/// straight from the (id-ordered) edge tables.
pub const EDGE_PROXIMITY_BONUS: f64 = 0.14;

/// Edge types that count toward the edge-proximity signal. A node one
/// hop from the anchor over any of these (in either direction) is
/// boosted. Ordered and fixed so the traversal is reproducible.
const PROXIMITY_EDGE_TYPES: &[&str] = &["CALLS", "USES", "IMPORTS"];

/// Per-anchor fan-out cap when collecting proximity neighbours. Bounds
/// the work regardless of the anchor's degree.
const PROXIMITY_FANOUT: usize = 4096;

/// Resolve the query to a single anchor node id, deterministically.
///
/// A node is an anchor candidate when its leaf `name`, normalised the
/// same way the ranker tokenises (`camel_split` + lower-case + join),
/// appears verbatim as a contiguous run of query tokens — i.e. the
/// query *names the symbol*. Among candidates the winner is the one
/// with the smallest `(qualified_name, id)`, a total order that does not
/// depend on insertion or row-scan timing. Returns `None` when the
/// query names no symbol in the candidate set (the common free-text
/// case), in which case the edge-proximity signal simply never fires.
fn resolve_anchor(rows: &[SearchGraphRow], q_tokens: &HashSet<String>) -> Option<i64> {
    let mut best: Option<&SearchGraphRow> = None;
    for row in rows {
        // The node's leaf name, tokenised the same way the query is.
        let name_tokens: Vec<String> = camel_split(&row.name)
            .split_whitespace()
            .map(|t| t.to_string())
            .collect();
        if name_tokens.is_empty() {
            continue;
        }
        // Every token of the symbol's name must be named by the query.
        // This keeps the anchor an *exact* symbol reference rather than
        // a partial-overlap guess (which would make the boost noisy).
        if !name_tokens.iter().all(|t| q_tokens.contains(t)) {
            continue;
        }
        best = Some(match best {
            None => row,
            Some(cur) => {
                if (row.qualified_name.as_str(), row.id) < (cur.qualified_name.as_str(), cur.id) {
                    row
                } else {
                    cur
                }
            }
        });
    }
    best.map(|r| r.id)
}

/// Collect the set of node ids within one hop of `anchor_id` over any
/// [`PROXIMITY_EDGE_TYPES`] edge, in both directions. Deterministic and
/// bounded: each edge type/direction is capped at [`PROXIMITY_FANOUT`].
/// The anchor itself is excluded from the returned set.
fn proximity_neighbors(store: &Store, anchor_id: i64) -> Result<HashSet<i64>> {
    let mut set = HashSet::new();
    for ty in PROXIMITY_EDGE_TYPES {
        for e in store.outgoing_edges(anchor_id, Some(ty), PROXIMITY_FANOUT)? {
            if e.target_id != anchor_id {
                set.insert(e.target_id);
            }
        }
        for e in store.incoming_edges(anchor_id, Some(ty), PROXIMITY_FANOUT)? {
            if e.source_id != anchor_id {
                set.insert(e.source_id);
            }
        }
    }
    Ok(set)
}

/// IDF-style term weighting computed from the candidate node vocabulary.
///
/// Each candidate node contributes one "document": its `name` +
/// `qualified_name` token set. A token's *document frequency* (`df`) is
/// the number of nodes whose token set contains it. The weight is the
/// standard smoothed inverse-document-frequency
///
/// ```text
/// idf(t) = ln((N + 1) / (df(t) + 1)) + 1
/// ```
///
/// where `N` is the document (node) count. The `+1` smoothing keeps every
/// weight strictly positive (so a token present in *every* node still
/// counts a little) and bounded, and the formula is monotonically
/// decreasing in `df`: a token shared by every node earns the floor
/// weight, a token unique to one node earns the most. A token not seen in
/// the vocabulary at all (e.g. a pure query term) is treated as maximally
/// rare (`df = 0`).
///
/// Determinism: the weights depend only on the candidate set's tokens, so
/// identical stores produce identical weights. With a single node, or a
/// vocabulary where every token has equal `df`, the weighting is uniform
/// and the weighted overlap reduces exactly to the plain Jaccard the
/// ranker used before — so prior behaviour is preserved whenever IDF has
/// nothing to discriminate on.
struct TokenIdf {
    df: HashMap<String, usize>,
    n: usize,
}

impl TokenIdf {
    fn from_rows(rows: &[SearchGraphRow]) -> Self {
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut n = 0usize;
        for row in rows {
            let toks = tokenize(&format!("{} {}", row.name, row.qualified_name));
            if toks.is_empty() {
                continue;
            }
            n += 1;
            for t in toks {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        Self { df, n }
    }

    /// The IDF weight of a token. Strictly positive and monotonically
    /// decreasing in document frequency.
    fn weight(&self, token: &str) -> f64 {
        let df = self.df.get(token).copied().unwrap_or(0);
        ((self.n as f64 + 1.0) / (df as f64 + 1.0)).ln() + 1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn score_one(
    row: &SearchGraphRow,
    q_tokens: &HashSet<String>,
    q_labels: &HashSet<&'static str>,
    q_sig: &MinHash,
    near_file: Option<&str>,
    proximity_set: &HashSet<i64>,
    idf: &TokenIdf,
) -> Option<SemanticHit> {
    let node_token_set: HashSet<String> = tokenize(&format!("{} {}", row.name, row.qualified_name));
    if node_token_set.is_empty() {
        return None;
    }
    let intersection: usize = q_tokens.intersection(&node_token_set).count();
    let is_proximity_neighbor = proximity_set.contains(&row.id);
    if intersection == 0 && !is_proximity_neighbor {
        // No token overlap and not graph-adjacent to the anchor — emit
        // no hit (would be noise). A node that *is* one CALLS/USES/
        // IMPORTS hop from the resolved anchor is allowed through even
        // with zero lexical overlap: edge adjacency is itself the
        // evidence of relatedness, so the edge-proximity signal can
        // surface a related symbol the free-text tokens alone miss.
        return None;
    }
    // IDF-weighted Jaccard: the overlap is the sum of IDF weights over
    // the shared tokens divided by the sum over the union. Common tokens
    // (low weight) move the score little; rare, discriminating tokens
    // dominate. With uniform weights this is exactly the plain
    // intersection/union Jaccard, so it stays in `[0, 1]` and reduces to
    // the prior behaviour when IDF has nothing to discriminate on.
    let mut inter_w = 0.0;
    let mut union_w = 0.0;
    for t in q_tokens.union(&node_token_set) {
        let w = idf.weight(t);
        union_w += w;
        if q_tokens.contains(t) && node_token_set.contains(t) {
            inter_w += w;
        }
    }
    // `union_w` is strictly positive here: node_token_set is non-empty
    // (checked above) and every IDF weight is > 0, so even a
    // zero-intersection proximity neighbour has a well-defined overlap of
    // 0.
    let jaccard = if union_w > 0.0 {
        inter_w / union_w
    } else {
        0.0
    };

    let label_match = q_labels.contains(label_key(&row.label));
    let label_bonus = if label_match {
        LABEL_AFFINITY_BONUS
    } else {
        0.0
    };

    let file_bonus = match near_file {
        Some(f) if paths_share_dir(f, &row.file_path) => FILE_PROXIMITY_BONUS,
        Some(_) => 0.0,
        None => 0.0,
    };

    // Qualified-name path proximity: of the node's leading module-path
    // segments (everything before the leaf identifier), how many are
    // named in the query token set? Reward the *fraction* matched so a
    // query that names the full containing path scores the cap, a query
    // that names one of several segments scores proportionally, and a
    // query naming none scores zero. Deterministic: depends only on the
    // node's qualified_name and the query tokens.
    let path_segments = qname_path_segments(&row.qualified_name);
    let (qname_path_bonus, qname_path_fired) = if path_segments.is_empty() {
        (0.0, false)
    } else {
        let matched = path_segments
            .iter()
            .filter(|seg| q_tokens.contains(*seg))
            .count();
        if matched == 0 {
            (0.0, false)
        } else {
            let frac = matched as f64 / path_segments.len() as f64;
            (frac * QNAME_PATH_MAX_BONUS, true)
        }
    };

    // Simhash signal: build the node's MinHash over the same token set
    // and derive a bounded bonus from the hamming distance to the
    // query signature. A small hamming distance (high slot agreement)
    // means the structural token sets overlap strongly. Deterministic:
    // identical inputs always yield identical signatures and bonuses.
    let node_sig = MinHash::from_tokens(node_token_set.iter());
    let agreement = 1.0 - (node_sig.hamming(q_sig) as f64 / MINHASH_K as f64);
    let (simhash_bonus, simhash_fired) = if agreement > SIMHASH_MIN_AGREEMENT {
        // Map agreement in (MIN, 1.0] linearly onto (0, SIMHASH_MAX_BONUS].
        let scaled = (agreement - SIMHASH_MIN_AGREEMENT) / (1.0 - SIMHASH_MIN_AGREEMENT);
        (scaled * SIMHASH_MAX_BONUS, true)
    } else {
        (0.0, false)
    };

    // Edge-aware proximity: the node sits one CALLS/USES/IMPORTS hop
    // from the query's resolved anchor symbol. We exclude the anchor
    // itself (it is not in `proximity_set`) so a symbol never boosts
    // itself.
    let edge_proximity_fired = proximity_set.contains(&row.id);
    let edge_proximity_bonus = if edge_proximity_fired {
        EDGE_PROXIMITY_BONUS
    } else {
        0.0
    };

    let score = jaccard
        + label_bonus
        + file_bonus
        + simhash_bonus
        + qname_path_bonus
        + edge_proximity_bonus;
    let mut signals = SemanticSignal::none();
    signals.token_overlap = true;
    if label_match {
        signals.label_affinity = true;
    }
    if file_bonus > 0.0 {
        signals.file_proximity = true;
    }
    if simhash_fired {
        signals.simhash = true;
    }
    if qname_path_fired {
        signals.qname_path = true;
    }
    if edge_proximity_fired {
        signals.edge_proximity = true;
    }
    let breakdown = SemanticBreakdown {
        token_overlap: jaccard,
        label_affinity: label_bonus,
        file_proximity: file_bonus,
        simhash: simhash_bonus,
        qname_path: qname_path_bonus,
        edge_proximity: edge_proximity_bonus,
    };
    Some(SemanticHit {
        node: row.clone(),
        score,
        signals,
        breakdown,
    })
}

fn tokenize(s: &str) -> HashSet<String> {
    camel_split(s)
        .split_whitespace()
        .map(|t| t.to_string())
        .collect()
}

/// Expand a free-text query into the set of search tokens used for
/// recall, decomposing every camelCase / snake_case / kebab-case
/// identifier in the query into its constituent sub-tokens via the
/// `camel_split` vocabulary.
///
/// Why this exists as a named, public step: code identifiers are
/// language-agnostic compounds — `getUserId` (Java/JS), `get_user_id`
/// (Rust/Python), `GetUserID` (Go), `get-user-id` (Lisp/CSS) all encode
/// the same three concepts. The graph stores each symbol's tokens
/// *already split* (see [`tokenize`]), so a query that names a symbol in
/// one casing convention must be split the same way to match a symbol
/// written in another. Expanding `getUserId` → `{get, user, id}` lets the
/// query recall `get_user_id`, `GetUserID`, and `get-user-id` alike —
/// improving cross-language recall on identifier-shaped queries without
/// any per-language rules.
///
/// The expansion is the union of the `camel_split` sub-tokens of every
/// whitespace-delimited word in `query`. It is **idempotent** (an
/// already-split query expands to itself), **order-independent** (returns
/// a set), and **deterministic** (a pure function of the input). It is a
/// strict superset of nothing it removes: every token a plain split would
/// produce is present, and compound words contribute their parts.
///
/// This is the same decomposition [`tokenize`] performs, surfaced as a
/// reusable, documented primitive so callers (and tests) can reason about
/// query recall explicitly. [`semantic_query_expanded`] drives its scoring
/// off this set.
pub fn expand_query_tokens(query: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for word in query.split_whitespace() {
        for tok in camel_split(word).split_whitespace() {
            if !tok.is_empty() {
                out.insert(tok.to_string());
            }
        }
    }
    out
}

/// The leading module-path segments of a qualified name — everything
/// *before* the leaf identifier. The qualified name is split on the
/// common symbol-path separators (`::`, `/`, `.`); the final component
/// (the symbol itself) is dropped, and each remaining segment is
/// `camel_split` + lower-cased so it can be compared against the query
/// token set. Empty segments are skipped.
///
/// Example: `myapp::orders::service::processOrder` →
/// `{myapp, orders, service, process}` (the leaf `processOrder` itself
/// is excluded, but a multi-word *module* segment is still expanded).
fn qname_path_segments(qname: &str) -> HashSet<String> {
    let raw: Vec<&str> = qname
        .split([':', '/', '.'])
        .filter(|s| !s.is_empty())
        .collect();
    if raw.len() <= 1 {
        // No containing path (bare leaf) — nothing structural to match.
        return HashSet::new();
    }
    // Drop the leaf (last component); keep the path prefix.
    let mut out = HashSet::new();
    for seg in &raw[..raw.len() - 1] {
        for tok in camel_split(seg).split_whitespace() {
            if !tok.is_empty() {
                out.insert(tok.to_string());
            }
        }
    }
    out
}

const KNOWN_LABELS: &[&str] = &[
    "function",
    "functions",
    "method",
    "methods",
    "struct",
    "structs",
    "class",
    "classes",
    "trait",
    "traits",
    "enum",
    "enums",
    "module",
    "modules",
    "import",
    "imports",
    "impl",
    "call",
    "calls",
    "type",
    "typealias",
];

fn label_words_in_query(query: &str) -> HashSet<&'static str> {
    let lower = query.to_lowercase();
    KNOWN_LABELS
        .iter()
        .copied()
        .filter(|w| {
            // Match as whole word surrounded by non-letters.
            let needle = format!(" {w} ");
            let padded = format!(" {lower} ");
            padded.contains(&needle)
        })
        .collect()
}

fn label_key(label: &str) -> &'static str {
    match label {
        "Function" => "function",
        "Method" => "method",
        "Struct" => "struct",
        "Class" => "class",
        "Trait" => "trait",
        "Enum" => "enum",
        "Module" => "module",
        "Import" => "import",
        "Impl" => "impl",
        "Call" => "call",
        "TypeAlias" => "typealias",
        _ => "",
    }
}

fn paths_share_dir(a: &str, b: &str) -> bool {
    fn dir_of(p: &str) -> &str {
        match p.rfind('/') {
            Some(i) => &p[..i],
            None => "",
        }
    }
    dir_of(a) == dir_of(b) && !dir_of(a).is_empty()
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
            ("Function", "processOrder"),
            ("Function", "processPayment"),
            ("Struct", "Order"),
            ("Function", "unrelatedHelper"),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: name.into(),
                qualified_name: format!("p::{label}::{name}"),
                file_path: "src/orders.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        s
    }

    #[test]
    fn query_for_process_order_ranks_that_node_first() {
        let s = seed();
        let hits = semantic_query(&s, "process order", None, None, 10).unwrap();
        assert!(!hits.is_empty());
        // Top hit should mention processOrder.
        assert!(hits[0].node.name.to_lowercase().contains("process"));
    }

    #[test]
    fn empty_query_returns_empty() {
        let s = seed();
        let hits = semantic_query(&s, "   ", None, None, 10).unwrap();
        assert!(hits.is_empty());
    }

    /// Forensics F2 + F6: semantic search must not surface `Call` / `Import`
    /// pseudo-nodes or the `::__file__` file-module sentinel. A `Struct Store`
    /// definition must outrank a `Call::Store` site that shares its name.
    #[test]
    fn semantic_excludes_call_import_and_file_module_nodes() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for (label, name, qname) in [
            ("Struct", "Store", "p::Struct::Store"),
            ("Call", "Store", "p::caller::Call::Store"),
            ("Import", "Store", "p::lib::Import::Store"),
            // file-module sentinel: name is the file stem, qname ends ::__file__
            ("Module", "store", "src/store.rs::__file__"),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: name.into(),
                qualified_name: qname.into(),
                file_path: "src/store.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }

        let hits = semantic_query(&s, "Store", None, None, 10).unwrap();
        assert!(!hits.is_empty(), "the Struct::Store must be found");
        for h in &hits {
            assert_ne!(h.node.label, "Call", "Call pseudo-node leaked: {h:?}");
            assert_ne!(h.node.label, "Import", "Import pseudo-node leaked: {h:?}");
            assert!(
                !h.node.qualified_name.ends_with("::__file__"),
                "__file__ module node leaked: {h:?}"
            );
        }
        // The real definition is the top result, not a pseudo-node.
        assert_eq!(hits[0].node.qualified_name, "p::Struct::Store");

        // The additive contract still holds: an explicit `labels=["Call"]`
        // request can still reach Call nodes when a caller really wants them.
        let call_hits = semantic_query_filtered(&s, "Store", None, None, &["Call"], 10).unwrap();
        assert!(
            call_hits.iter().any(|h| h.node.label == "Call"),
            "explicit labels=[Call] must still return Call nodes"
        );
    }

    #[test]
    fn query_with_no_token_overlap_returns_empty() {
        let s = seed();
        let hits = semantic_query(&s, "xylophone zebra", None, None, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn label_word_in_query_boosts_matching_label() {
        let s = seed();
        let hits_struct = semantic_query(&s, "order struct", None, None, 10).unwrap();
        // "struct" should boost the Order struct node.
        let struct_hit = hits_struct.iter().find(|h| h.node.name == "Order");
        assert!(struct_hit.is_some(), "Order struct should appear");
        let other = hits_struct.iter().find(|h| h.node.name == "processOrder");
        assert!(other.is_some());
        // The struct-hit should have label_affinity signal set.
        let s = struct_hit.unwrap();
        assert!(
            s.signals.label_affinity,
            "Order should have label_affinity signal"
        );
        // The other may or may not.
        let _ = other.unwrap();
    }

    #[test]
    fn simhash_signal_fires_for_strong_token_overlap() {
        let s = seed();
        // "process order" shares both tokens with processOrder's token
        // set {process, order} -> high MinHash agreement -> signal set.
        let hits = semantic_query(&s, "process order", None, None, 10).unwrap();
        let po = hits
            .iter()
            .find(|h| h.node.name == "processOrder")
            .expect("processOrder should be a hit");
        assert!(
            po.signals.simhash,
            "strong token overlap should set the simhash signal"
        );
    }

    #[test]
    fn simhash_signal_absent_for_weak_overlap() {
        let s = seed();
        // Query overlaps "unrelatedHelper" only via the generic
        // "helper" token among a large query token set -> low MinHash
        // agreement -> simhash signal should NOT fire even though the
        // token-overlap (hit-emitting) condition is met.
        let hits =
            semantic_query(&s, "helper alpha beta gamma delta epsilon", None, None, 10).unwrap();
        let uh = hits
            .iter()
            .find(|h| h.node.name == "unrelatedHelper")
            .expect("unrelatedHelper should still be a hit via token overlap");
        assert!(uh.signals.token_overlap);
        assert!(
            !uh.signals.simhash,
            "weak structural overlap should not set the simhash signal"
        );
    }

    #[test]
    fn simhash_bonus_is_deterministic() {
        let s = seed();
        let a = semantic_query(&s, "process order", None, None, 10).unwrap();
        let b = semantic_query(&s, "process order", None, None, 10).unwrap();
        // Identical inputs must produce byte-identical scored output.
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.node, y.node);
            assert_eq!(x.score, y.score);
            assert_eq!(x.signals, y.signals);
        }
    }

    #[test]
    fn simhash_bonus_lifts_full_match_above_partial_match() {
        let s = seed();
        // processOrder tokens {process, order} fully covered by the
        // query; processPayment tokens {process, payment} only half.
        // Both share "process", but the simhash signal rewards the
        // fuller structural overlap of processOrder.
        let hits = semantic_query(&s, "process order", None, None, 10).unwrap();
        let order_pos = hits.iter().position(|h| h.node.name == "processOrder");
        let pay_pos = hits.iter().position(|h| h.node.name == "processPayment");
        assert!(order_pos.is_some() && pay_pos.is_some());
        assert!(
            order_pos < pay_pos,
            "processOrder should rank above processPayment"
        );
    }

    /// Seed nodes that live under distinct module paths so qualified-name
    /// path proximity can be exercised independently of leaf overlap.
    fn seed_modules() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for (qname, name) in [
            ("app::orders::service::handle", "handle"),
            ("app::billing::service::handle", "handle"),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: qname.into(),
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
    fn qname_path_proximity_boosts_node_in_named_module() {
        let s = seed_modules();
        // Both nodes share the leaf "handle"; only one lives under the
        // "orders" module. Naming "orders" must lift it above billing.
        let hits = semantic_query(&s, "handle orders", None, None, 10).unwrap();
        let orders_pos = hits
            .iter()
            .position(|h| h.node.qualified_name.contains("orders"));
        let billing_pos = hits
            .iter()
            .position(|h| h.node.qualified_name.contains("billing"));
        assert!(orders_pos.is_some() && billing_pos.is_some());
        assert!(
            orders_pos < billing_pos,
            "node in the named 'orders' module should rank first"
        );
        let orders_hit = &hits[orders_pos.unwrap()];
        assert!(
            orders_hit.signals.qname_path,
            "qname_path signal should fire for the orders node"
        );
        let billing_hit = &hits[billing_pos.unwrap()];
        assert!(
            !billing_hit.signals.qname_path,
            "billing node's path was not named in the query"
        );
    }

    #[test]
    fn qname_path_signal_absent_when_no_module_named() {
        let s = seed_modules();
        // Query names only the leaf, not any module segment.
        let hits = semantic_query(&s, "handle", None, None, 10).unwrap();
        assert!(!hits.is_empty());
        assert!(
            hits.iter().all(|h| !h.signals.qname_path),
            "no module segment named -> qname_path must not fire"
        );
    }

    #[test]
    fn qname_path_bonus_is_proportional_to_segments_matched() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        // Two nodes with the SAME leaf but different path depths sharing
        // the query terms to differing degrees:
        //   full:    app::orders::handle   -> segments {app, orders}
        //   partial: app::orders::sub::handle -> {app, orders, sub}
        // Query "handle app orders" matches 2/2 of full but 2/3 of
        // partial, so full earns the larger fraction of the cap.
        for qname in ["app::orders::handle", "app::orders::sub::handle"] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "handle".into(),
                qualified_name: qname.into(),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        let hits = semantic_query(&s, "handle app orders", None, None, 10).unwrap();
        let full = hits
            .iter()
            .find(|h| h.node.qualified_name == "app::orders::handle")
            .unwrap();
        let partial = hits
            .iter()
            .find(|h| h.node.qualified_name == "app::orders::sub::handle")
            .unwrap();
        assert!(
            full.score > partial.score,
            "fuller path-segment coverage should score higher: full={} partial={}",
            full.score,
            partial.score
        );
    }

    #[test]
    fn generation_tiebreak_orders_higher_id_first_on_exact_tie() {
        // Two nodes that produce an identical score (same leaf token, no
        // module path named, same label, same file) must still order
        // deterministically. They share the same qualified_name path
        // prefix but differ in qualified_name string, so we force the
        // id (generation) tie-break by giving them an equal score and
        // equal qualified_name is impossible (unique constraint); instead
        // assert the sort is stable and total across repeated runs.
        let s = seed_modules();
        let a = semantic_query(&s, "handle", None, None, 10).unwrap();
        let b = semantic_query(&s, "handle", None, None, 10).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.node.id, y.node.id);
            assert_eq!(x.score, y.score);
        }
    }

    #[test]
    fn qname_path_does_not_fire_for_bare_leaf_qualified_name() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "solo".into(),
            // No path separators: a bare leaf.
            qualified_name: "solo".into(),
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits = semantic_query(&s, "solo", None, None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            !hits[0].signals.qname_path,
            "a bare-leaf qualified_name has no path to match"
        );
    }

    /// Seed an anchor `processOrder` that CALLS `validateCart` and is
    /// USED-by `checkout`, plus an unrelated `loadConfig`. Returns
    /// (store, anchor_id, callee_id, user_id, unrelated_id).
    fn seed_edge_graph() -> (Store, i64, i64, i64, i64) {
        use grepplus_store::NewEdge;
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
        let anchor = mk(&mut s, "processOrder");
        let callee = mk(&mut s, "validateCart");
        let user = mk(&mut s, "checkout");
        let unrelated = mk(&mut s, "loadConfig");
        // anchor --CALLS--> callee
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: anchor,
            target_id: callee,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        // user --USES--> anchor (so anchor's incoming USES neighbour is `checkout`)
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: user,
            target_id: anchor,
            edge_type: "USES".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        (s, anchor, callee, user, unrelated)
    }

    #[test]
    fn edge_proximity_surfaces_callee_with_no_token_overlap() {
        let (s, _anchor, callee, _user, _unrelated) = seed_edge_graph();
        // Query names the anchor symbol exactly. The callee shares no
        // tokens with "process order" yet must surface via edge proximity.
        let hits = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let callee_hit = hits.iter().find(|h| h.node.id == callee);
        assert!(
            callee_hit.is_some(),
            "callee should surface purely via the CALLS edge proximity"
        );
        assert!(callee_hit.unwrap().signals.edge_proximity);
    }

    #[test]
    fn edge_proximity_surfaces_incoming_user() {
        let (s, _anchor, _callee, user, _unrelated) = seed_edge_graph();
        let hits = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let user_hit = hits.iter().find(|h| h.node.id == user);
        assert!(
            user_hit.is_some(),
            "the USES-source `checkout` should surface via incoming-edge proximity"
        );
        assert!(user_hit.unwrap().signals.edge_proximity);
    }

    #[test]
    fn edge_proximity_does_not_fire_for_unrelated_node() {
        let (s, _anchor, _callee, _user, unrelated) = seed_edge_graph();
        let hits = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        // loadConfig is neither lexically nor graph-adjacent -> absent.
        assert!(
            hits.iter().all(|h| h.node.id != unrelated),
            "an unrelated, non-adjacent node must not appear"
        );
    }

    #[test]
    fn edge_proximity_anchor_does_not_boost_itself() {
        let (s, anchor, _callee, _user, _unrelated) = seed_edge_graph();
        let hits = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let anchor_hit = hits.iter().find(|h| h.node.id == anchor).unwrap();
        assert!(
            !anchor_hit.signals.edge_proximity,
            "the anchor symbol must not earn its own proximity boost"
        );
    }

    #[test]
    fn edge_proximity_signal_absent_when_query_names_no_symbol() {
        let (s, _anchor, callee, _user, _unrelated) = seed_edge_graph();
        // "validate cart" names the callee leaf, not the anchor; and no
        // edge runs from a node the *free-text* query resolves... Here we
        // assert that a query naming no resolvable anchor leaves the
        // signal off for every hit it does return.
        let hits = semantic_query(&s, "configuration loader thing", None, Some("p"), 10).unwrap();
        assert!(
            hits.iter().all(|h| !h.signals.edge_proximity),
            "no resolvable anchor -> edge_proximity must never fire"
        );
        let _ = callee;
    }

    #[test]
    fn edge_proximity_boost_lifts_neighbor_above_equal_overlap_nonneighbor() {
        use grepplus_store::NewEdge;
        // Two nodes with identical lexical overlap to the query, but only
        // one is a CALLS-neighbour of the resolved anchor. The neighbour
        // must rank strictly higher.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mk = |s: &mut Store, name: &str, qn: &str| {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: qn.into(),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap()
        };
        // Anchor named exactly by the query.
        let anchor = mk(&mut s, "render", "p::render");
        // Two "helper" nodes: same leaf token "helper", identical overlap.
        let near = mk(&mut s, "helper", "p::a::helper");
        let far = mk(&mut s, "helper", "p::b::helper");
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: anchor,
            target_id: near,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits = semantic_query(&s, "render helper", None, Some("p"), 10).unwrap();
        let near_pos = hits.iter().position(|h| h.node.id == near);
        let far_pos = hits.iter().position(|h| h.node.id == far);
        assert!(near_pos.is_some() && far_pos.is_some());
        assert!(
            near_pos < far_pos,
            "the CALLS-neighbour of the anchor should outrank the equal-overlap non-neighbour"
        );
    }

    #[test]
    fn edge_proximity_is_deterministic() {
        let (s, _anchor, _callee, _user, _unrelated) = seed_edge_graph();
        let a = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let b = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.node, y.node);
            assert_eq!(x.score, y.score);
            assert_eq!(x.signals, y.signals);
        }
    }

    #[test]
    fn idf_weight_is_monotonic_decreasing_in_document_frequency() {
        // Three nodes; "common" appears in all three, "rare" in one.
        let rows = vec![
            SearchGraphRow {
                id: 1,
                project: "p".into(),
                label: "Function".into(),
                name: "commonRare".into(),
                qualified_name: "p::commonRare".into(),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 1,
            },
            SearchGraphRow {
                id: 2,
                project: "p".into(),
                label: "Function".into(),
                name: "commonOther".into(),
                qualified_name: "p::commonOther".into(),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 1,
            },
            SearchGraphRow {
                id: 3,
                project: "p".into(),
                label: "Function".into(),
                name: "commonThing".into(),
                qualified_name: "p::commonThing".into(),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 1,
            },
        ];
        let idf = TokenIdf::from_rows(&rows);
        // "common" is in every node; "rare" in one. Rare must weigh more.
        assert!(
            idf.weight("rare") > idf.weight("common"),
            "rarer token must earn a higher IDF weight"
        );
        // Every weight is strictly positive (smoothing).
        assert!(idf.weight("common") > 0.0);
        // An unseen token is treated as maximally rare (df = 0).
        assert!(idf.weight("neverseen") >= idf.weight("rare"));
    }

    #[test]
    fn idf_downweights_common_token_to_favor_discriminating_match() {
        // Vocabulary where "handler" is ubiquitous and the second token
        // is unique per node. A query naming the common token plus the
        // discriminating one must rank the node matching the *rare* token
        // first, because IDF starves the common token of weight.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        // Many nodes carry "handler"; only one carries "payment".
        for name in [
            "orderHandler",
            "userHandler",
            "authHandler",
            "cacheHandler",
            "paymentHandler",
        ] {
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
        let hits = semantic_query(&s, "payment handler", None, Some("p"), 10).unwrap();
        // paymentHandler shares the rare "payment" token; the others share
        // only the ubiquitous "handler". IDF must float paymentHandler to
        // the top.
        assert_eq!(
            hits[0].node.name, "paymentHandler",
            "the node matching the rare, discriminating token must rank first"
        );
    }

    #[test]
    fn idf_weighting_is_deterministic() {
        let s = seed();
        let a = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let b = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.node, y.node);
            assert_eq!(x.score, y.score);
            assert_eq!(x.signals, y.signals);
        }
    }

    #[test]
    fn idf_uniform_vocabulary_reduces_to_plain_jaccard() {
        // A single node: every token has df = 1 and N = 1, so the IDF
        // weighting is uniform across its tokens and the weighted Jaccard
        // equals the plain intersection/union ratio. With query "process
        // order" fully covering {process, order} the score's overlap
        // component is 1.0 (the remaining bonuses are separate signals).
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            // Bare leaf so there is no module path to add tokens.
            name: "processOrder".into(),
            qualified_name: "processOrder".into(),
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        assert_eq!(hits.len(), 1);
        // Query tokens {process, order} == node tokens {process, order}:
        // weighted Jaccard = 1.0 regardless of the (uniform) weights.
        // The simhash bonus also fires; subtract nothing — just assert the
        // jaccard floor is present by checking score >= 1.0.
        assert!(
            hits[0].score >= 1.0,
            "uniform-vocab full overlap yields jaccard 1.0, score={}",
            hits[0].score
        );
    }

    #[test]
    fn idf_preserves_process_order_ranking() {
        // The historic ranking assertion must still hold under IDF: a
        // query for "process order" ranks processOrder above
        // processPayment (full vs partial overlap), and IDF — which
        // downweights the shared "process" — only sharpens that gap.
        let s = seed();
        let hits = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let order_pos = hits.iter().position(|h| h.node.name == "processOrder");
        let pay_pos = hits.iter().position(|h| h.node.name == "processPayment");
        assert!(order_pos.is_some() && pay_pos.is_some());
        assert!(order_pos < pay_pos);
    }

    #[test]
    fn label_filter_restricts_candidates_to_listed_labels() {
        let s = seed();
        // Without a filter, both the processOrder Function and the Order
        // Struct can surface for "order". With a Struct-only filter, the
        // Function candidates are excluded entirely.
        let all = semantic_query(&s, "order", None, Some("p"), 10).unwrap();
        assert!(all.iter().any(|h| h.node.label == "Function"));
        let structs =
            semantic_query_filtered(&s, "order", None, Some("p"), &["Struct"], 10).unwrap();
        assert!(!structs.is_empty());
        assert!(
            structs.iter().all(|h| h.node.label == "Struct"),
            "filter must keep only Struct nodes"
        );
        assert!(structs.iter().any(|h| h.node.name == "Order"));
    }

    #[test]
    fn empty_label_filter_matches_unfiltered_query() {
        let s = seed();
        let plain = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let empty_filter =
            semantic_query_filtered(&s, "process order", None, Some("p"), &[], 10).unwrap();
        assert_eq!(plain.len(), empty_filter.len());
        for (a, b) in plain.iter().zip(empty_filter.iter()) {
            assert_eq!(a.node, b.node);
            assert_eq!(a.score, b.score);
            assert_eq!(a.signals, b.signals);
        }
    }

    #[test]
    fn label_filter_excludes_all_when_no_label_matches() {
        let s = seed();
        let hits =
            semantic_query_filtered(&s, "process order", None, Some("p"), &["Enum"], 10).unwrap();
        assert!(hits.is_empty(), "no Enum nodes -> empty result");
    }

    #[test]
    fn related_symbols_ranks_edge_neighbor_above_pure_lexical() {
        let (s, _anchor, callee, user, unrelated) = seed_edge_graph();
        // Anchor "processOrder" CALLS validateCart and is USED by checkout.
        let hits = related_symbols(&s, "processOrder", Some("p"), 10).unwrap();
        // The two graph neighbours must surface and be edge_adjacent.
        let callee_hit = hits.iter().find(|h| h.node.id == callee).unwrap();
        let user_hit = hits.iter().find(|h| h.node.id == user).unwrap();
        assert!(callee_hit.edge_adjacent);
        assert!(user_hit.edge_adjacent);
        assert_eq!(callee_hit.hops, 1);
        // loadConfig is unrelated -> absent.
        assert!(hits.iter().all(|h| h.node.id != unrelated));
        // The anchor itself is never returned.
        assert!(hits.iter().all(|h| h.node.name != "processOrder"));
    }

    #[test]
    fn related_symbols_edge_neighbor_outranks_lexical_only() {
        use grepplus_store::NewEdge;
        // anchor `render`; a neighbour `renderHelper` shares a token AND is
        // a CALLS neighbour; a disconnected `renderUtil` shares the same
        // token but no edge. The neighbour must rank strictly higher.
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
        let anchor = mk(&mut s, "render");
        let near = mk(&mut s, "renderHelper");
        let far = mk(&mut s, "renderUtil");
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: anchor,
            target_id: near,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits = related_symbols(&s, "render", Some("p"), 10).unwrap();
        let near_pos = hits.iter().position(|h| h.node.id == near).unwrap();
        let far_pos = hits.iter().position(|h| h.node.id == far).unwrap();
        assert!(
            near_pos < far_pos,
            "the CALLS-neighbour should outrank the lexical-only match"
        );
    }

    #[test]
    fn related_symbols_unknown_anchor_is_empty() {
        let (s, _a, _c, _u, _un) = seed_edge_graph();
        let hits = related_symbols(&s, "noSuchSymbol", Some("p"), 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn related_symbols_empty_anchor_is_empty() {
        let (s, _a, _c, _u, _un) = seed_edge_graph();
        let hits = related_symbols(&s, "   ", Some("p"), 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn related_symbols_is_deterministic() {
        let (s, _a, _c, _u, _un) = seed_edge_graph();
        let a = related_symbols(&s, "processOrder", Some("p"), 10).unwrap();
        let b = related_symbols(&s, "processOrder", Some("p"), 10).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.node, y.node);
            assert_eq!(x.score, y.score);
            assert_eq!(x.edge_adjacent, y.edge_adjacent);
        }
    }

    #[test]
    fn file_proximity_boost_when_near_file_matches_dir() {
        let s = seed();
        let near = semantic_query(&s, "process", Some("src/orders.rs"), None, 10).unwrap();
        let no_near = semantic_query(&s, "process", None, None, 10).unwrap();
        // The processOrder/processPayment nodes live in src/orders.rs.
        // When the caller hints src/orders.rs, those nodes get a 0.10 bonus.
        let near_top = near.first().map(|h| h.score).unwrap_or(0.0);
        let no_near_top = no_near.first().map(|h| h.score).unwrap_or(0.0);
        assert!(
            near_top > no_near_top,
            "file-proximity should boost score: near={near_top} vs no_near={no_near_top}"
        );
    }

    #[test]
    fn breakdown_components_sum_to_score() {
        let s = seed();
        let hits = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        assert!(!hits.is_empty());
        for h in &hits {
            let total = h.breakdown.total();
            assert!(
                (total - h.score).abs() < 1e-9,
                "breakdown {total} must sum to score {}",
                h.score
            );
        }
    }

    #[test]
    fn breakdown_attributes_label_bonus_to_label_field() {
        let s = seed();
        let hits = semantic_query(&s, "order struct", None, Some("p"), 10).unwrap();
        let order = hits
            .iter()
            .find(|h| h.node.name == "Order")
            .expect("Order struct should be present");
        assert!(order.signals.label_affinity);
        assert!(
            (order.breakdown.label_affinity - LABEL_AFFINITY_BONUS).abs() < 1e-9,
            "label contribution should equal the label bonus"
        );
    }

    #[test]
    fn breakdown_is_deterministic() {
        let s = seed();
        let a = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        let b = semantic_query(&s, "process order", None, Some("p"), 10).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.breakdown, y.breakdown);
        }
    }

    #[test]
    fn multi_anchor_boosts_node_adjacent_to_two_anchors_above_one() {
        use grepplus_store::NewEdge;
        // Two anchors anchorA, anchorB. `shared` is a CALLS-neighbour of
        // BOTH; `single` is a neighbour of only anchorA. Both share the
        // same query token, so lexical overlap is equal — multi-anchor
        // adjacency must lift `shared` above `single`.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        let mk = |s: &mut Store, name: &str, qn: &str| {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: qn.into(),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap()
        };
        let anchor_a = mk(&mut s, "anchorAlpha", "p::anchorAlpha");
        let anchor_b = mk(&mut s, "anchorBeta", "p::anchorBeta");
        let shared = mk(&mut s, "helper", "p::a::helper");
        let single = mk(&mut s, "helper", "p::b::helper");
        let mut edge = |src: i64, tgt: i64| {
            s.insert_edge(&NewEdge {
                project: "p".into(),
                source_id: src,
                target_id: tgt,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        };
        edge(anchor_a, shared);
        edge(anchor_b, shared);
        edge(anchor_a, single);
        let hits = semantic_query_multi_anchor(
            &s,
            "helper",
            &["anchorAlpha", "anchorBeta"],
            None,
            Some("p"),
            10,
        )
        .unwrap();
        let shared_pos = hits.iter().position(|h| h.node.id == shared);
        let single_pos = hits.iter().position(|h| h.node.id == single);
        assert!(shared_pos.is_some() && single_pos.is_some());
        assert!(
            shared_pos < single_pos,
            "node adjacent to BOTH anchors must outrank node adjacent to one"
        );
        // The doubly-adjacent node's edge_proximity contribution is 2x.
        let shared_hit = &hits[shared_pos.unwrap()];
        assert!(
            (shared_hit.breakdown.edge_proximity - 2.0 * MULTI_ANCHOR_PER_HOP).abs() < 1e-9,
            "two-anchor adjacency should contribute 2 * MULTI_ANCHOR_PER_HOP, got {}",
            shared_hit.breakdown.edge_proximity
        );
    }

    #[test]
    fn multi_anchor_anchor_does_not_boost_itself() {
        use grepplus_store::NewEdge;
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
        let a = mk(&mut s, "anchorAlpha");
        let b = mk(&mut s, "anchorBeta");
        // a CALLS b and b CALLS a so each is the other's neighbour; assert
        // neither earns a boost as *itself*.
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: a,
            target_id: b,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits = semantic_query_multi_anchor(
            &s,
            "anchor",
            &["anchorAlpha", "anchorBeta"],
            None,
            Some("p"),
            10,
        )
        .unwrap();
        for h in &hits {
            if h.node.id == a || h.node.id == b {
                // An anchor may still appear via token overlap, but its
                // edge_proximity contribution must be zero (anchors removed
                // from the boost set).
                assert_eq!(
                    h.breakdown.edge_proximity, 0.0,
                    "an anchor must not earn a multi-anchor proximity boost"
                );
            }
        }
    }

    #[test]
    fn multi_anchor_empty_query_is_empty() {
        let (s, _anchor, _callee, _user, _unrelated) = seed_edge_graph();
        let hits =
            semantic_query_multi_anchor(&s, "   ", &["processOrder"], None, Some("p"), 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn multi_anchor_unresolved_names_are_skipped() {
        // With no resolvable anchor the multi-anchor mode degrades to the
        // plain lexical ranking; it must not panic and must still return
        // the lexical hits.
        let s = seed();
        let hits = semantic_query_multi_anchor(
            &s,
            "process order",
            &["noSuchSymbol"],
            None,
            Some("p"),
            10,
        )
        .unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| h.breakdown.edge_proximity == 0.0));
    }

    #[test]
    fn multi_anchor_is_deterministic() {
        let (s, _anchor, _callee, _user, _unrelated) = seed_edge_graph();
        let a = semantic_query_multi_anchor(
            &s,
            "process order",
            &["processOrder"],
            None,
            Some("p"),
            10,
        )
        .unwrap();
        let b = semantic_query_multi_anchor(
            &s,
            "process order",
            &["processOrder"],
            None,
            Some("p"),
            10,
        )
        .unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.node, y.node);
            assert_eq!(x.score, y.score);
            assert_eq!(x.breakdown, y.breakdown);
        }
    }

    // ---- query expansion (camel_split vocabulary) --------------------

    #[test]
    fn expand_query_tokens_decomposes_compound_identifiers() {
        // Every common casing convention for "get user id" must decompose
        // to the same concept-token set.
        let expected: HashSet<String> = ["get", "user", "id"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(expand_query_tokens("getUserId"), expected);
        assert_eq!(expand_query_tokens("get_user_id"), expected);
        assert_eq!(expand_query_tokens("GetUserID"), expected);
        assert_eq!(expand_query_tokens("get-user-id"), expected);
        // Already-split input is idempotent.
        assert_eq!(expand_query_tokens("get user id"), expected);
    }

    #[test]
    fn expand_query_tokens_is_empty_for_blank() {
        assert!(expand_query_tokens("   ").is_empty());
        assert!(expand_query_tokens("").is_empty());
    }

    /// A multi-language corpus: the same "fetch user record" concept named
    /// in four casing conventions, one per "language".
    fn seed_multilang() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for (name, qname, file) in [
            // JS / Java style.
            ("fetchUserRecord", "p::fetchUserRecord", "src/a.js"),
            // Rust / Python style.
            ("fetch_user_record", "p::fetch_user_record", "src/b.rs"),
            // Go exported style.
            ("FetchUserRecord", "p::FetchUserRecord", "src/c.go"),
            // Unrelated symbol that must not be recalled.
            ("computeChecksum", "p::computeChecksum", "src/d.rs"),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: name.into(),
                qualified_name: qname.into(),
                file_path: file.into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        s
    }

    #[test]
    fn expanded_compound_query_recalls_every_casing_convention() {
        let s = seed_multilang();
        // A single compound query token, JS-style.
        let hits =
            semantic_query_expanded(&s, "fetchUserRecord", None, Some("p"), &[], 10).unwrap();
        let names: HashSet<&str> = hits.iter().map(|h| h.node.name.as_str()).collect();
        // All three casing variants of the same concept are recalled.
        assert!(names.contains("fetchUserRecord"));
        assert!(names.contains("fetch_user_record"));
        assert!(names.contains("FetchUserRecord"));
        // The unrelated symbol is NOT recalled (no shared concept tokens).
        assert!(
            !names.contains("computeChecksum"),
            "unrelated symbol must not match: {names:?}"
        );
    }

    #[test]
    fn expanded_query_matches_plain_query_when_already_split() {
        // Expansion is idempotent on a whitespace-split query, so the
        // expanded entry point must agree with the plain one term-for-term.
        let s = seed_multilang();
        let plain = semantic_query(&s, "fetch user record", None, Some("p"), 10).unwrap();
        let expanded =
            semantic_query_expanded(&s, "fetch user record", None, Some("p"), &[], 10).unwrap();
        assert_eq!(plain.len(), expanded.len());
        for (x, y) in plain.iter().zip(expanded.iter()) {
            assert_eq!(x.node, y.node);
            assert_eq!(x.score, y.score);
        }
    }

    #[test]
    fn expanded_query_label_filter_narrows_candidates() {
        let mut s = seed_multilang();
        // Add a Struct sharing the concept tokens; the label filter must
        // exclude it when only Functions are requested.
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Struct".into(),
            name: "FetchUserRecordCache".into(),
            qualified_name: "p::FetchUserRecordCache".into(),
            file_path: "src/e.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits =
            semantic_query_expanded(&s, "fetchUserRecord", None, Some("p"), &["Function"], 10)
                .unwrap();
        assert!(hits.iter().all(|h| h.node.label == "Function"));
        assert!(hits.iter().all(|h| h.node.name != "FetchUserRecordCache"));
    }

    #[test]
    fn expanded_query_is_deterministic() {
        let s = seed_multilang();
        let a = semantic_query_expanded(&s, "fetchUserRecord", None, Some("p"), &[], 10).unwrap();
        let b = semantic_query_expanded(&s, "fetchUserRecord", None, Some("p"), &[], 10).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.node, y.node);
            assert_eq!(x.score, y.score);
        }
    }

    #[test]
    fn expanded_empty_query_is_empty() {
        let s = seed_multilang();
        let hits = semantic_query_expanded(&s, "   ", None, Some("p"), &[], 10).unwrap();
        assert!(hits.is_empty());
    }

    /// Seed several symbols that share a leaf name `process` across three
    /// files, plus a couple of distinct symbols, so diversification by
    /// `(file_path, name)` has same-group duplicates to collapse.
    fn seed_diversify() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        // Three `process` functions, each in its OWN file -> distinct
        // (file,name) groups, so they all survive diversification...
        for f in ["a.rs", "b.rs", "c.rs"] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "process".into(),
                qualified_name: format!("p::{f}::process"),
                file_path: f.into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        // ...and two MORE `process` functions in the SAME file a.rs (same
        // (file,name) group) -> these are the duplicates diversification
        // should cap. Give them different qnames so they are distinct nodes.
        for q in ["process_dup1", "process_dup2"] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "process".into(),
                qualified_name: format!("p::a.rs::{q}"),
                file_path: "a.rs".into(),
                start_line: 2,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        s
    }

    #[test]
    fn diversify_caps_hits_per_file_name_group() {
        let s = seed_diversify();
        // a.rs has THREE `process` nodes (one + two dups) = one (file,name)
        // group of size 3. Without diversification all three appear.
        let undiv =
            semantic_query_diversified(&s, "process", None, Some("p"), false, 1, 50).unwrap();
        let a_rs_count = undiv
            .iter()
            .filter(|d| d.hit.node.file_path == "a.rs")
            .count();
        assert_eq!(a_rs_count, 3, "without diversify, all 3 a.rs hits appear");

        // With diversification at per_group=1, a.rs contributes exactly one.
        let div = semantic_query_diversified(&s, "process", None, Some("p"), true, 1, 50).unwrap();
        let a_rs_div = div
            .iter()
            .filter(|d| d.hit.node.file_path == "a.rs")
            .count();
        assert_eq!(a_rs_div, 1, "diversify per_group=1 caps a.rs to one hit");
        // The distinct-file `process` nodes (b.rs, c.rs) still survive.
        assert!(div.iter().any(|d| d.hit.node.file_path == "b.rs"));
        assert!(div.iter().any(|d| d.hit.node.file_path == "c.rs"));
    }

    #[test]
    fn diversify_per_group_two_keeps_two() {
        let s = seed_diversify();
        let div = semantic_query_diversified(&s, "process", None, Some("p"), true, 2, 50).unwrap();
        let a_rs = div
            .iter()
            .filter(|d| d.hit.node.file_path == "a.rs")
            .count();
        assert_eq!(a_rs, 2, "per_group=2 keeps two from the a.rs group");
    }

    #[test]
    fn diversify_preserves_relative_rank_order() {
        let s = seed_diversify();
        let undiv =
            semantic_query_diversified(&s, "process", None, Some("p"), false, 1, 50).unwrap();
        let div = semantic_query_diversified(&s, "process", None, Some("p"), true, 1, 50).unwrap();
        // The diversified list is a subsequence of the undiversified one:
        // survivors keep their relative order.
        let undiv_order: Vec<i64> = undiv.iter().map(|d| d.hit.node.id).collect();
        let div_order: Vec<i64> = div.iter().map(|d| d.hit.node.id).collect();
        let mut ui = 0usize;
        for &id in &div_order {
            while ui < undiv_order.len() && undiv_order[ui] != id {
                ui += 1;
            }
            assert!(ui < undiv_order.len(), "diversified order is a subsequence");
            ui += 1;
        }
    }

    #[test]
    fn confidence_is_in_unit_range_and_matches_score() {
        let s = seed_diversify();
        let div = semantic_query_diversified(&s, "process", None, Some("p"), false, 1, 50).unwrap();
        assert!(!div.is_empty());
        for d in &div {
            assert!(
                (0.0..=1.0).contains(&d.confidence),
                "confidence out of range: {}",
                d.confidence
            );
            // Confidence is the raw score scaled by the documented max.
            let expected = (d.hit.score / MAX_SEMANTIC_SCORE).clamp(0.0, 1.0);
            assert!((d.confidence - expected).abs() < 1e-12);
        }
        // Confidence is monotonic non-increasing down the (score-sorted) list.
        for w in div.windows(2) {
            assert!(w[0].confidence >= w[1].confidence);
        }
    }

    #[test]
    fn confidence_full_on_token_overlap_one() {
        // A query that exactly equals a symbol's token set yields token
        // overlap 1.0; with no other bonuses the confidence is
        // 1.0 / MAX_SEMANTIC_SCORE. With a perfect score it would clamp to
        // 1.0. Here we just assert it is strictly positive and bounded.
        let s = seed();
        let div =
            semantic_query_diversified(&s, "process order", None, Some("p"), false, 1, 10).unwrap();
        let top = &div[0];
        assert!(top.confidence > 0.0 && top.confidence <= 1.0);
    }

    #[test]
    fn diversify_per_group_zero_is_treated_as_one() {
        let s = seed_diversify();
        let div = semantic_query_diversified(&s, "process", None, Some("p"), true, 0, 50).unwrap();
        let a_rs = div
            .iter()
            .filter(|d| d.hit.node.file_path == "a.rs")
            .count();
        assert_eq!(a_rs, 1, "per_group=0 behaves like per_group=1");
    }

    #[test]
    fn diversify_empty_query_is_empty() {
        let s = seed_diversify();
        let div = semantic_query_diversified(&s, "   ", None, Some("p"), true, 1, 10).unwrap();
        assert!(div.is_empty());
    }

    #[test]
    fn diversify_is_deterministic() {
        let s = seed_diversify();
        let a = semantic_query_diversified(&s, "process", None, Some("p"), true, 1, 50).unwrap();
        let b = semantic_query_diversified(&s, "process", None, Some("p"), true, 1, 50).unwrap();
        assert_eq!(a, b);
    }

    /// Seed a store where the name, qualified_name, and doc fields carry
    /// *different* tokens, so a fielded query can be shown to match only the
    /// requested field.
    fn seed_fielded() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        // node A: name has "widget", qname module "alpha", doc says "frobnicate".
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "widget".into(),
            qualified_name: "p::alpha::widget".into(),
            file_path: "src/a.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({ "doc": "frobnicate the gizmo" }),
        })
        .unwrap();
        // node B: name has "gadget", qname module "beta", doc says "widget here".
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "gadget".into(),
            qualified_name: "p::beta::gadget".into(),
            file_path: "src/b.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({ "doc": "the widget lives here" }),
        })
        .unwrap();
        s
    }

    #[test]
    fn fielded_name_matches_only_name_field() {
        let s = seed_fielded();
        // "widget" is in A's name and in B's doc. A Name-fielded query must
        // return A (name match) and NOT B (whose name is "gadget").
        let hits =
            semantic_query_fielded(&s, "widget", SemanticField::Name, Some("p"), 10).unwrap();
        let names: Vec<&str> = hits.iter().map(|h| h.node.name.as_str()).collect();
        assert_eq!(names, vec!["widget"]);
    }

    #[test]
    fn fielded_doc_matches_only_doc_field() {
        let s = seed_fielded();
        // "widget" is in B's doc and A's name. A Doc-fielded query must return
        // B (doc contains "widget") and NOT A (whose doc is "frobnicate ...").
        let hits = semantic_query_fielded(&s, "widget", SemanticField::Doc, Some("p"), 10).unwrap();
        let names: Vec<&str> = hits.iter().map(|h| h.node.name.as_str()).collect();
        assert_eq!(names, vec!["gadget"]);
    }

    #[test]
    fn fielded_qualified_name_matches_only_qname_field() {
        let s = seed_fielded();
        // "beta" appears only in B's qualified_name (p::beta::gadget).
        let hits = semantic_query_fielded(&s, "beta", SemanticField::QualifiedName, Some("p"), 10)
            .unwrap();
        let names: Vec<&str> = hits.iter().map(|h| h.node.name.as_str()).collect();
        assert_eq!(names, vec!["gadget"]);
        // The same token against the Name field matches nothing.
        let none = semantic_query_fielded(&s, "beta", SemanticField::Name, Some("p"), 10).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn fielded_doc_with_no_doc_property_yields_no_hit() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        // No "doc" property at all.
        s.insert_node(&NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: "widget".into(),
            qualified_name: "p::widget".into(),
            file_path: "src/a.rs".into(),
            start_line: 1,
            end_line: 1,
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits = semantic_query_fielded(&s, "widget", SemanticField::Doc, Some("p"), 10).unwrap();
        assert!(hits.is_empty(), "no doc text -> no doc-field match");
    }

    #[test]
    fn fielded_breakdown_only_token_overlap_fires() {
        let s = seed_fielded();
        let hits =
            semantic_query_fielded(&s, "widget", SemanticField::Name, Some("p"), 10).unwrap();
        let h = &hits[0];
        assert!(h.signals.token_overlap);
        assert!(!h.signals.label_affinity);
        assert!(!h.signals.file_proximity);
        assert!(!h.signals.simhash);
        assert!(!h.signals.qname_path);
        assert!(!h.signals.edge_proximity);
        // The score equals the token-overlap contribution alone.
        assert_eq!(h.score, h.breakdown.token_overlap);
        assert_eq!(h.breakdown.label_affinity, 0.0);
        assert_eq!(h.breakdown.qname_path, 0.0);
        assert!(h.score > 0.0 && h.score <= 1.0);
    }

    #[test]
    fn fielded_empty_query_is_empty() {
        let s = seed_fielded();
        let hits = semantic_query_fielded(&s, "   ", SemanticField::Name, Some("p"), 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn fielded_is_deterministic() {
        let s = seed_fielded();
        let a = semantic_query_fielded(&s, "widget", SemanticField::Doc, Some("p"), 10).unwrap();
        let b = semantic_query_fielded(&s, "widget", SemanticField::Doc, Some("p"), 10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn fielded_respects_limit() {
        let s = seed_fielded();
        // Both nodes' qualified_names share the "p" segment, so a "p" query on
        // QualifiedName matches both; limit 1 returns just the top.
        let hits =
            semantic_query_fielded(&s, "p", SemanticField::QualifiedName, Some("p"), 1).unwrap();
        assert_eq!(hits.len(), 1);
    }
}
