//! Ad-hoc profiling harness for the indexer scale work (Track A,
//! Finding #1). NOT a test — run manually:
//!
//! ```sh
//! cargo run -p grepplus-indexer --example profile_index --release -- /path/to/corpus
//! ```
//!
//! It indexes the given repo into an in-memory store and prints the wall
//! time plus the report counters, so we can compare the O(n^2) hotspot
//! before/after the fix on a 500/1000-file corpus.

use std::time::Instant;

fn main() {
    let root = std::env::args()
        .nth(1)
        .expect("usage: profile_index <repo_root>");
    let mut store = grepplus_store::Store::open_memory().expect("open store");
    let path = std::path::Path::new(&root);

    let t0 = Instant::now();
    let r0 = grepplus_indexer::index(&mut store, path, "profile").expect("full index run");
    let dt0 = t0.elapsed();
    println!(
        "[full]        {} files, {} nodes, {} edges, skipped {} in {:?}",
        r0.files_indexed, r0.nodes_extracted, r0.edges_extracted, r0.files_skipped, dt0
    );

    // Second run with no edits: exercises the incremental no-op path.
    let t1 = Instant::now();
    let r1 = grepplus_indexer::index(&mut store, path, "profile").expect("incremental run");
    let dt1 = t1.elapsed();
    println!(
        "[incremental] {} files, {} nodes, {} edges, skipped {} in {:?}",
        r1.files_indexed, r1.nodes_extracted, r1.edges_extracted, r1.files_skipped, dt1
    );
}
