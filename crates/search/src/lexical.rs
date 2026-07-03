//! Lexical search over indexed file content.
//!
//! R-011 / WP-R011: replaces the symbol-metadata-only FTS path with
//! a real grep-like code-content search.
//!
//! Two entry points:
//!
//! - [`search_code`] — grep-like code search; finds any literal
//!   snippet in indexed file contents. Backed by
//!   `grepplus_store::Store::search_file_content`.
//!
//! - [`search_symbols`] — the old `nodes_fts` symbol-metadata
//!   search, renamed so we keep the old behaviour available for
//!   callers that specifically wanted symbol-only matching.

use grepplus_core::Result;
use grepplus_store::fts::{self, FtsHit};

/// One FTS hit over file content, plus convenience fields for the
/// search-callable surface.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeHit {
    /// `file_path::line` string for human-friendly display (e.g.,
    /// `src/lib.rs:42`). Always populated for `search_code` hits.
    pub location: String,
    /// The matched snippet (one indexed line).
    pub snippet: String,
    /// BM25 rank (lower is better).
    pub rank: f64,
}

/// One FTS hit over the symbol metadata FTS (the old behaviour).
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolHit {
    pub node_id: i64,
    pub rank: f64,
}

/// A code hit carrying a **normalized** relevance score in addition to
/// the raw BM25 rank. Returned by [`search_code_ranked`].
///
/// `rank` is the raw SQLite `bm25()` value the store produced (negative,
/// where a more-negative value is more relevant). `relevance` is that
/// rank mapped onto `[0.0, 1.0]` with **higher = more relevant**, matching
/// the upstream convention of presenting a positive relevance score
/// rather than a raw, scale-dependent BM25 number. See
/// [`search_code_ranked`] for the exact normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedCodeHit {
    /// `file_path::line` location string (e.g. `src/lib.rs:42`).
    pub location: String,
    /// The matched snippet (one indexed line).
    pub snippet: String,
    /// Raw BM25 rank as returned by the store (lower / more negative is
    /// better). Preserved verbatim so callers that already reason about
    /// the raw value are unaffected.
    pub rank: f64,
    /// Normalized relevance in `[0.0, 1.0]`, higher = more relevant. The
    /// best hit in a result set scores `1.0`; the worst scores `0.0` when
    /// the set spans a range of ranks, or all hits score `1.0` when they
    /// are equally relevant.
    pub relevance: f64,
}

impl From<FtsHit> for SymbolHit {
    fn from(h: FtsHit) -> Self {
        Self {
            node_id: h.node_id,
            rank: h.rank,
        }
    }
}

/// grep-like code search over indexed file contents.
///
/// Pass any literal pattern; whitespace splits tokens, every token
/// must match. Camel-case queries still work because the indexed
/// snippets are stored verbatim (so `processOrder` matches a line
/// containing `processOrder` exactly, and `process Order` also
/// matches via the unicode61 tokenizer that splits on whitespace
/// inside the snippet).
///
/// **Phrase / exact-match support.** A query wrapped in double quotes
/// — e.g. `"build cache"` — is treated as an *exact phrase*: the inner
/// text (between the quotes) is sent to the FTS layer as ordinary tokens
/// for recall, then hits whose snippet contains the inner text as a
/// **literal, contiguous substring** (case-insensitive) are boosted ahead
/// of hits that merely contain the tokens scattered. This makes
/// `"foo bar"` rank a line reading `foo bar baz` above one reading
/// `bar ... foo`, matching the grep "quoted = exact" intuition while
/// preserving FTS recall. An unquoted query behaves exactly as before, so
/// this is a strict, additive extension. The quote stripping recognises a
/// query that *starts and ends* with `"` (after trimming surrounding
/// whitespace); an unbalanced or interior quote is left untouched and
/// searched literally.
///
/// Determinism: the boost is a stable two-key sort — exact-substring hits
/// first (preserving their relative BM25 order), then the rest — so
/// identical inputs yield an identical, totally-ordered result.
pub fn search_code(
    store: &grepplus_store::Store,
    project: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<CodeHit>> {
    let phrase = exact_phrase(query);
    // For the FTS recall pass, search by the inner (unquoted) text when a
    // phrase was requested; otherwise the query verbatim. The store splits
    // on whitespace and ANDs the tokens either way.
    let recall_query = phrase.as_deref().unwrap_or(query);
    let hits = store.search_file_content(project, recall_query, limit)?;
    let mut out: Vec<CodeHit> = hits
        .into_iter()
        .map(|h| CodeHit {
            location: format!("{}:{}", h.rel_path, h.line),
            snippet: h.snippet,
            rank: h.rank,
        })
        .collect();

    // Phrase boost: when an exact phrase was requested, stably partition so
    // snippets containing the literal phrase (case-insensitive) come first,
    // each partition keeping its incoming best-first BM25 order. A stable
    // sort on a boolean key does exactly this without disturbing ties.
    if let Some(p) = &phrase {
        let needle = p.to_lowercase();
        if !needle.is_empty() {
            out.sort_by_key(|h| !h.snippet.to_lowercase().contains(&needle));
        }
    }
    Ok(out)
}

/// If `query` is a double-quoted phrase (starts and ends with `"` after
/// trimming surrounding whitespace, with at least one character between the
/// quotes), return the inner text; otherwise `None`. A bare `""`, a single
/// `"`, or a string with only one quote is treated as not-a-phrase so it is
/// searched literally.
fn exact_phrase(query: &str) -> Option<String> {
    let t = query.trim();
    let bytes = t.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"' {
        let inner = &t[1..t.len() - 1];
        if !inner.is_empty() {
            return Some(inner.to_string());
        }
    }
    None
}

/// grep-like code search returning a **normalized relevance** alongside
/// the raw BM25 rank, moving the presented score toward the upstream's
/// positive-relevance convention without altering the underlying store
/// ranking.
///
/// The store ranks by raw SQLite `bm25()`: a negative number where a
/// *more negative* value is more relevant, and results already arrive in
/// best-first order. Raw BM25 magnitudes are corpus- and query-dependent,
/// so they are awkward to threshold or display. This function preserves
/// that order and the raw `rank`, and adds a `relevance` in `[0.0, 1.0]`
/// (higher = better) via a stable min/max normalization over the returned
/// set:
///
/// ```text
/// relevance(h) = (worst_rank - rank(h)) / (worst_rank - best_rank)
/// ```
///
/// where `best_rank` is the smallest (most negative) rank in the set and
/// `worst_rank` the largest. The best hit scores `1.0`, the worst `0.0`,
/// and the mapping is **monotonic** in relevance so it never reorders the
/// results. When every hit shares one rank (or there is a single hit) the
/// denominator is zero; all hits then score `1.0` (equally relevant).
///
/// Determinism: the normalization is a pure function of the store's
/// best-first result vector, so identical inputs yield identical
/// relevances. This is additive — [`search_code`] and [`CodeHit`] are
/// untouched.
pub fn search_code_ranked(
    store: &grepplus_store::Store,
    project: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedCodeHit>> {
    let hits = store.search_file_content(project, query, limit)?;
    if hits.is_empty() {
        return Ok(Vec::new());
    }
    // The store returns best-first (ascending raw rank). best = smallest
    // (most negative) rank; worst = largest. Compute over the actual set
    // rather than assuming the first/last positions, so the normalization
    // is robust even if the store's order ever changes — the relevance
    // mapping stays monotonic in the raw rank regardless.
    let mut best = hits[0].rank;
    let mut worst = hits[0].rank;
    for h in &hits {
        if h.rank < best {
            best = h.rank;
        }
        if h.rank > worst {
            worst = h.rank;
        }
    }
    let span = worst - best;
    Ok(hits
        .into_iter()
        .map(|h| {
            // span == 0 -> every hit equally relevant -> 1.0. Otherwise a
            // smaller (more negative) rank maps nearer 1.0.
            let relevance = if span > 0.0 {
                (worst - h.rank) / span
            } else {
                1.0
            };
            RankedCodeHit {
                location: format!("{}:{}", h.rel_path, h.line),
                snippet: h.snippet,
                rank: h.rank,
                relevance,
            }
        })
        .collect())
}

/// Symbol-only FTS search. Backed by the historical `nodes_fts`
/// table (keyed on node metadata: name/qualified_name/label/
/// file_path). Finds "this project contains a symbol whose
/// camelSplit-tokens cover my query" — NOT general code search.
///
/// Kept for callers that explicitly want symbol-only matching; the
/// CLI dispatcher exposes this as `search-symbols`.
pub fn search_symbols(
    store: &grepplus_store::Store,
    query: &str,
    limit: usize,
) -> Result<Vec<SymbolHit>> {
    let hits = fts::search_fts(store, query, limit)?;
    Ok(hits.into_iter().map(SymbolHit::from).collect())
}

/// Project-scoped symbol-only FTS search. This is the user-facing variant:
/// a shared store can contain multiple workspace snapshots, but a symbol
/// query from one repo must not leak rows from another.
pub fn search_symbols_in_project(
    store: &grepplus_store::Store,
    project: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<SymbolHit>> {
    let hits = fts::search_fts_in_project(store, project, query, limit)?;
    Ok(hits.into_iter().map(SymbolHit::from).collect())
}

/// Exact count for project-scoped symbol FTS matches, using the same
/// pseudo-node and project filters as [`search_symbols_in_project`].
pub fn count_symbols_in_project(
    store: &grepplus_store::Store,
    project: &str,
    query: &str,
) -> Result<i64> {
    Ok(fts::count_fts_in_project(store, project, query)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use grepplus_store::{ContentRow, NewNode, Project, Store};

    #[test]
    fn search_code_finds_literal_text_in_indexed_file() {
        // R-011: indexed file content is searchable as text, not
        // only as symbol metadata.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "// this file is a hello world example".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "fn helper() { let x = 42; }".into(),
                },
                ContentRow {
                    line: 3,
                    snippet: "let processOrder = build_default_order();".into(),
                },
            ],
        )
        .unwrap();

        let hits = search_code(&s, "p", "processOrder", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "expected at least one hit for 'processOrder' in file content, got 0"
        );
        assert_eq!(hits[0].location, "src/lib.rs:3");
        assert!(hits[0].snippet.contains("processOrder"));

        let hits = search_code(&s, "p", "hello world", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "expected a hit for the comment 'hello world'"
        );
    }

    #[test]
    fn search_symbols_remains_backed_by_node_metadata_fts() {
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
            name: "processOrder".into(),
            qualified_name: "p::Function::processOrder".into(),
            file_path: "src/orders.rs".into(),
            start_line: 1,
            end_line: 5,
            properties: serde_json::json!({}),
        })
        .unwrap();
        let hits = search_symbols(&s, "process", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "search_symbols should hit the function node"
        );
        assert_eq!(hits[0].node_id, 1);
    }

    #[test]
    fn search_symbols_project_scope_and_count_match_result_scope() {
        let mut s = Store::open_memory().unwrap();
        for project in ["p1", "p2"] {
            s.upsert_project(&Project {
                name: project.into(),
                indexed_at: "x".into(),
                root_path: format!("/{project}"),
            })
            .unwrap();
            s.insert_node(&NewNode {
                project: project.into(),
                label: "Function".into(),
                name: "SharedSymbol".into(),
                qualified_name: format!("{project}::SharedSymbol"),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 5,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }

        let hits = search_symbols_in_project(&s, "p1", "Shared", 20).unwrap();
        assert_eq!(hits.len(), 1);
        let node = s.get_node(hits[0].node_id).unwrap().unwrap();
        assert_eq!(node.project, "p1");
        assert_eq!(count_symbols_in_project(&s, "p1", "Shared").unwrap(), 1);
    }

    #[test]
    fn search_code_empty_query_returns_empty() {
        let s = Store::open_memory().unwrap();
        let hits = search_code(&s, "p", "   ", 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_code_ranked_pins_relevance_order_on_known_corpus() {
        // A small, deterministic corpus where one line mentions the query
        // term twice and the others once, plus a non-matching line. BM25
        // must rank the term-dense line first; the normalized relevance
        // must be 1.0 for the best hit and within [0,1] and monotonically
        // non-increasing down the result list.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "let cache = build_cache(cache_size);".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "fn cache_lookup() { todo!() }".into(),
                },
                ContentRow {
                    line: 3,
                    snippet: "// unrelated comment about widgets".into(),
                },
            ],
        )
        .unwrap();

        let hits = search_code_ranked(&s, "p", "cache", 10).unwrap();
        assert!(hits.len() >= 2, "expected >=2 'cache' hits, got {hits:?}");
        // Pin: the term-dense line 1 (two 'cache' occurrences) ranks first.
        assert_eq!(
            hits[0].location, "src/lib.rs:1",
            "the line with the most 'cache' occurrences must rank first"
        );
        // Best hit normalizes to exactly 1.0.
        assert_eq!(hits[0].relevance, 1.0);
        // Relevance is in [0,1] and monotonically non-increasing (the
        // store returns best-first, normalization is order-preserving).
        for h in &hits {
            assert!(
                (0.0..=1.0).contains(&h.relevance),
                "relevance out of range: {}",
                h.relevance
            );
        }
        for w in hits.windows(2) {
            assert!(
                w[0].relevance >= w[1].relevance,
                "relevance must not increase down the result list: {} then {}",
                w[0].relevance,
                w[1].relevance
            );
            // Raw rank order (ascending) must agree with relevance order.
            assert!(w[0].rank <= w[1].rank);
        }
        // The non-matching comment line must not appear.
        assert!(hits.iter().all(|h| h.location != "src/lib.rs:3"));
    }

    #[test]
    fn search_code_ranked_single_hit_is_full_relevance() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[ContentRow {
                line: 7,
                snippet: "let unique_marker_token = 1;".into(),
            }],
        )
        .unwrap();
        let hits = search_code_ranked(&s, "p", "unique_marker_token", 10).unwrap();
        assert_eq!(hits.len(), 1);
        // A single hit has a zero-span normalization -> full relevance.
        assert_eq!(hits[0].relevance, 1.0);
        assert_eq!(hits[0].location, "src/lib.rs:7");
    }

    #[test]
    fn search_code_ranked_empty_query_is_empty() {
        let s = Store::open_memory().unwrap();
        let hits = search_code_ranked(&s, "p", "   ", 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_code_ranked_is_deterministic() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "alpha beta alpha".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "alpha gamma".into(),
                },
            ],
        )
        .unwrap();
        let a = search_code_ranked(&s, "p", "alpha", 10).unwrap();
        let b = search_code_ranked(&s, "p", "alpha", 10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn search_code_ranks_consistently_across_languages() {
        // A known multi-language corpus. `search_code` is literal text
        // search over stored snippets (the unicode61 FTS tokenizer splits
        // on separators/whitespace but is case-insensitive and does NOT
        // split camelCase), so we pick a shared token written with a word
        // boundary — `request` — that every language file contains. The
        // query-term density is strictly distinct per file (3, 2, 1, then
        // a non-matching file), so BM25 yields a *total* order we can pin
        // exactly, guarding against future drift in cross-language ranking.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        // Go: token appears 3x -> most relevant.
        s.insert_file_content_rows(
            "p",
            "server.go",
            &[ContentRow {
                line: 10,
                snippet: "func handle(request Request) { log(request); send(request) }".into(),
            }],
        )
        .unwrap();
        // Rust: token appears 2x.
        s.insert_file_content_rows(
            "p",
            "server.rs",
            &[ContentRow {
                line: 20,
                snippet: "fn handle(request: Request) { log(request) }".into(),
            }],
        )
        .unwrap();
        // Python: token appears 1x.
        s.insert_file_content_rows(
            "p",
            "server.py",
            &[ContentRow {
                line: 30,
                snippet: "def handle(request): pass".into(),
            }],
        )
        .unwrap();
        // JS: no occurrence of the query token.
        s.insert_file_content_rows(
            "p",
            "server.js",
            &[ContentRow {
                line: 40,
                snippet: "function handle(payload) {}".into(),
            }],
        )
        .unwrap();

        let hits = search_code(&s, "p", "request", 10).unwrap();
        let order: Vec<&str> = hits.iter().map(|h| h.location.as_str()).collect();
        // Pinned, language-agnostic order: density 3 > 2 > 1, JS excluded.
        assert_eq!(order, vec!["server.go:10", "server.rs:20", "server.py:30"]);
        // Ranks are best-first (ascending raw BM25).
        for w in hits.windows(2) {
            assert!(
                w[0].rank <= w[1].rank,
                "ranks must be ascending (best-first): {} then {}",
                w[0].rank,
                w[1].rank
            );
        }
        // Determinism: a second identical query yields the identical order.
        let again = search_code(&s, "p", "request", 10).unwrap();
        assert_eq!(hits, again);
    }

    #[test]
    fn search_symbols_ranks_consistently_across_languages() {
        // The symbol-metadata FTS must also rank a shared concept the same
        // way regardless of the source language's casing convention. Four
        // symbols name the same "parse config" concept across languages;
        // one decoy names something else. A query for "parse config" must
        // recall exactly the four concept symbols, in a deterministic
        // order, and exclude the decoy.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        for (name, qname, file) in [
            ("parseConfig", "p::parseConfig", "a.js"),
            ("parse_config", "p::parse_config", "b.rs"),
            ("ParseConfig", "p::ParseConfig", "c.go"),
            ("parse-config", "p::parse-config", "d.css"),
            ("computeHash", "p::computeHash", "e.rs"),
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

        let hits = search_symbols(&s, "parse config", 10).unwrap();
        let ids: Vec<i64> = hits.iter().map(|h| h.node_id).collect();
        // The four casing variants (node ids 1..=4) are recalled; the
        // decoy computeHash (id 5) is not.
        assert_eq!(ids.len(), 4, "expected the four concept symbols: {ids:?}");
        assert!(!ids.contains(&5), "decoy must not be recalled");
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4]);
        // Ranks ascending (best-first), and the result is deterministic.
        for w in hits.windows(2) {
            assert!(w[0].rank <= w[1].rank);
        }
        let again = search_symbols(&s, "parse config", 10).unwrap();
        assert_eq!(hits, again);
    }

    #[test]
    fn search_code_quoted_phrase_boosts_exact_substring() {
        // Two lines both contain the tokens "build" and "cache", but only
        // one contains the contiguous phrase "build cache". A quoted query
        // must rank the contiguous line first, while an unquoted query is
        // free to rank by BM25 alone.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[
                // Tokens present but NOT contiguous as "build cache".
                ContentRow {
                    line: 1,
                    snippet: "cache = init(); later we build cache".into(),
                },
                // Exact phrase "build cache" present.
                ContentRow {
                    line: 2,
                    snippet: "fn build_cache() { return build cache here }".into(),
                },
            ],
        )
        .unwrap();

        // Quoted phrase: the line with the contiguous "build cache" wins.
        let hits = search_code(&s, "p", "\"build cache\"", 10).unwrap();
        assert!(hits.len() >= 2, "expected both rows recalled: {hits:?}");
        assert_eq!(
            hits[0].location, "src/lib.rs:2",
            "the contiguous 'build cache' line must be boosted to the top"
        );
        assert!(hits[0].snippet.to_lowercase().contains("build cache"));
        // Every exact-substring hit precedes every non-substring hit.
        let first_non_exact = hits
            .iter()
            .position(|h| !h.snippet.to_lowercase().contains("build cache"));
        if let Some(idx) = first_non_exact {
            assert!(
                hits[idx..]
                    .iter()
                    .all(|h| !h.snippet.to_lowercase().contains("build cache")),
                "exact-substring hits must all come before non-substring ones"
            );
        }

        // Determinism: identical query yields identical order.
        let again = search_code(&s, "p", "\"build cache\"", 10).unwrap();
        assert_eq!(hits, again);
    }

    #[test]
    fn search_code_unquoted_query_is_unchanged_by_phrase_logic() {
        // The same corpus searched without quotes must behave exactly like
        // the plain token search (phrase boost is inert).
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "alpha beta alpha".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "alpha gamma".into(),
                },
            ],
        )
        .unwrap();
        // Unquoted single token: identical to a direct store search mapping.
        let via_search = search_code(&s, "p", "alpha", 10).unwrap();
        let direct = s.search_file_content("p", "alpha", 10).unwrap();
        let mapped: Vec<(String, f64)> = direct
            .iter()
            .map(|h| (format!("{}:{}", h.rel_path, h.line), h.rank))
            .collect();
        let got: Vec<(String, f64)> = via_search
            .iter()
            .map(|h| (h.location.clone(), h.rank))
            .collect();
        assert_eq!(got, mapped, "unquoted query must not reorder vs the store");
    }

    #[test]
    fn search_code_degenerate_quotes_are_searched_literally() {
        // A lone `""` is not a phrase; an interior quote is not a phrase.
        // Neither should panic, and both fall back to literal token search.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[ContentRow {
                line: 1,
                snippet: "the marker token lives here".into(),
            }],
        )
        .unwrap();
        // Empty quotes -> no phrase, recall query "\"\"" has no word tokens.
        let empty = search_code(&s, "p", "\"\"", 10).unwrap();
        assert!(empty.is_empty(), "empty-quote query matches nothing");
        // A normal token still works.
        let hits = search_code(&s, "p", "marker", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].location, "src/lib.rs:1");
    }

    #[test]
    fn search_code_finds_text_that_is_not_a_symbol() {
        // This is the regression R-011 fixes: a literal that is
        // only in a function body (not in any extracted symbol
        // name) must still be findable.
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[ContentRow {
                line: 5,
                snippet: "let cached_payment_retry = 0;".into(),
            }],
        )
        .unwrap();
        let hits = search_code(&s, "p", "payment_retry", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "R-011: literal in function body must be findable; got 0 hits"
        );
    }
}
