//! Parallel-safe language modules. Each `src/langs/<lang>.rs` is a fully
//! self-contained language that self-registers via `inventory::submit!` — see
//! [`crate::registry`]. The `pub mod <lang>;` declarations below are GENERATED
//! by `build.rs` from the files present in this directory, so adding a language
//! requires NO edit here (no merge conflict when adding languages in parallel).
//!
//! To add a language: create `src/langs/<lang>.rs` following the template in an
//! existing file (e.g. `elixir.rs`) and add the `tree-sitter-<lang>` dependency
//! to `crates/parser/Cargo.toml`. That is the entire surface.

include!(concat!(env!("OUT_DIR"), "/langs_generated.rs"));
