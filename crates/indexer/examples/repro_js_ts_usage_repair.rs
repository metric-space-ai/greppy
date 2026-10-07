//! Bounded structural-only reproduction of first-use JS/TS usage repair.
//! This does not load inference or mark the semantic index complete.
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(args.next().ok_or("repository path required")?);
    let db = PathBuf::from(args.next().ok_or("scratch store path required")?);
    if args.next().is_some() {
        return Err("expected repository and scratch store paths".into());
    }
    let mut store = greppy_store::Store::open(&db)?;
    let report = greppy_indexer::index(&mut store, &root, "reproduction")?;
    let repaired = greppy_indexer::js_ts_usages_repaired(&store)?;
    let nodes = store
        .list_nodes("reproduction", "", "", 0, i64::MAX as usize)?
        .len();
    println!(
        "structural_reproduction files_indexed={} nodes={} js_ts_repaired={repaired}",
        report.files_indexed, nodes
    );
    if !repaired {
        return Err("JS/TS repair marker missing".into());
    }
    println!("structural_only=true; no embedding or full CLI health claim");
    Ok(())
}
