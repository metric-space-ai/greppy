#[path = "../../assets/prompts/contract.rs"]
mod prompt_contract;
fn main() {
    let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let (public, adapter) = prompt_contract::verify_repository(&manifest.join("../.."))
        .unwrap_or_else(|e| panic!("{e}"));
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(out.join("canonical-prompt.md"), public)
        .expect("write verified prompt snapshot");
    std::fs::write(out.join("agent-adapter.md"), adapter).expect("write verified adapter snapshot");
}
