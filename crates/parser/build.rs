//! Auto-discovers language modules under `src/langs/*.rs` so that adding a
//! new language is a ZERO-shared-edit operation: drop `src/langs/<lang>.rs`
//! (self-registering via `inventory::submit!`) and add the one
//! `tree-sitter-<lang>` line to Cargo.toml. This build script generates the
//! `pub mod <lang>;` declarations, so no shared `mod.rs` edit (and therefore
//! no merge conflict) is needed when many agents add languages in parallel.

use std::{env, fs, path::Path};

fn main() {
    let langs_dir = Path::new("src/langs");
    let mut decls = String::new();
    if let Ok(entries) = fs::read_dir(langs_dir) {
        let mut stems: Vec<String> = entries
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    let stem = p.file_stem()?.to_str()?.to_string();
                    if stem != "mod" {
                        return Some(stem);
                    }
                }
                None
            })
            .collect();
        stems.sort(); // deterministic order
                      // `mod` declarations inside an `include!`d file resolve relative to
                      // OUT_DIR, not the includer, so emit an explicit absolute `#[path]`.
        let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
        for stem in stems {
            decls.push_str(&format!(
                "#[path = \"{manifest}/src/langs/{stem}.rs\"]\npub mod {stem};\n"
            ));
            println!("cargo:rerun-if-changed=src/langs/{stem}.rs");
        }
    }
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("langs_generated.rs");
    fs::write(out, decls).unwrap();
    println!("cargo:rerun-if-changed=src/langs");
}
