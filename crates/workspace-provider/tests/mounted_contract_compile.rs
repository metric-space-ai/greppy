#[path = "common/mounted_contract.rs"]
mod mounted_contract;

#[test]
fn shared_mounted_contract_has_one_cross_platform_entrypoint() {
    let _: fn(&std::path::Path, &greppy_workspace_core::WorkspaceCore) =
        mounted_contract::exercise_mounted_contract;
}
