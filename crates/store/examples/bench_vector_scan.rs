//! Micro-benchmark for the exact vector scan (`vector_search_exact`).
//!
//! Usage: `cargo run --release -p greppy-store --example bench_vector_scan -- [N]`
//! Fills a temp store with N synthetic 768-dim embeddings and times queries.
//! Documents the brute-force headroom that made a kd-tree unnecessary: at
//! 768 dims a kd-tree's measured pruning rate is 0%, while this scan is
//! memory-bandwidth-bound.

use greppy_store::{NewVectorEmbedding, Project, Store, VectorSearchQuery};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    const DIM: usize = 768;
    let dir = std::env::temp_dir().join(format!("gp-vecbench-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut store = Store::open(&dir.join("graph.db")).unwrap();
    store
        .upsert_project(&Project {
            name: "bench".into(),
            indexed_at: "2026-01-01T00:00:00Z".into(),
            root_path: "/bench".into(),
        })
        .unwrap();

    // Deterministic pseudo-random vectors (xorshift) — no rand dep.
    let mut state = 0x243F6A8885A308D3u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32 / u64::MAX as f32) - 0.5
    };

    let t0 = std::time::Instant::now();
    for i in 0..n {
        let v: Vec<f32> = (0..DIM).map(|_| next()).collect();
        store
            .upsert_vector_embedding(&NewVectorEmbedding {
                project: "bench".into(),
                model_id: "m".into(),
                prompt_version: "v1".into(),
                task: "code".into(),
                node_id: None,
                chunk_idx: 0,
                qualified_name: format!("f{i}"),
                file_path: format!("src/{}.rs", i % 977),
                start_line: 1,
                end_line: 1,
                content_sha256: format!("{i:064x}"),
                graph_generation: 1,
                vector: v,
            })
            .unwrap();
    }
    eprintln!("inserted {n} x {DIM} in {:?}", t0.elapsed());

    let query: Vec<f32> = (0..DIM).map(|_| next()).collect();
    let q = VectorSearchQuery {
        project: "bench",
        model_id: "m",
        prompt_version: "v1",
        task: "code",
        graph_generation: Some(1),
        file_path: None,
        path_prefixes: None,
        min_score: None,
        limit: 20,
    };
    // warm the page cache, then measure
    let _ = store.vector_search_exact(&query, &q).unwrap();
    let reps = 5;
    let t1 = std::time::Instant::now();
    let mut top_score = 0.0f32;
    for _ in 0..reps {
        let hits = store.vector_search_exact(&query, &q).unwrap();
        top_score = hits.first().map(|h| h.score).unwrap_or(0.0);
    }
    let per = t1.elapsed() / reps;
    println!(
        "N={n}: {per:?} per query ({:.1}M dots/s equivalent, top={top_score:.4})",
        n as f64 / per.as_secs_f64() / 1e6
    );
    let _ = std::fs::remove_dir_all(&dir);
}
