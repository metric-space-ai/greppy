#[path = "../../assets/prompts/contract.rs"]
mod prompt_contract;
fn main() {
    let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    prompt_contract::write_snapshots(&manifest.join("../.."), &out)
        .unwrap_or_else(|e| panic!("{e}"));
}
