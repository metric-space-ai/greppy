//! Prompt inspection must work without a repository, graph or inference store.
use std::process::Command;

#[test]
fn prompt_export_matches_embedded_bytes_without_preparing_a_workspace() {
    let scratch = tempfile::tempdir().unwrap();
    let store = scratch.path().join("store-is-a-file");
    std::fs::write(&store, "unchanged store sentinel").unwrap();
    let run = |args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .args(args)
            .current_dir(scratch.path())
            .env("GREPPY_STORE_DIR", &store)
            .env_remove(greppy_agent::AGENT_RUN_ENV)
            .output()
            .expect("run prompt export");
        assert!(
            output.status.success(),
            "{:?}: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.stderr.is_empty(),
            "{:?}: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };
    for external in [false, true] {
        let json_args: &[&str] = if external {
            &["prompt", "--external", "--json"]
        } else {
            &["prompt", "--json"]
        };
        let actual: serde_json::Value = serde_json::from_str(&run(json_args)).unwrap();
        let expected = greppy_agent::prompt_metadata(external);
        assert_eq!(actual, expected);
        assert_eq!(actual["version"], env!("CARGO_PKG_VERSION"));
        assert!(actual["prompt"]
            .as_str()
            .unwrap()
            .starts_with(&format!("greppy {}\n\n", env!("CARGO_PKG_VERSION"))));
        let plain_args: &[&str] = if external {
            &["prompt", "--external"]
        } else {
            &["prompt"]
        };
        assert_eq!(
            run(plain_args),
            format!(
                "prompt-sha256: {}\n{}",
                expected["prompt_sha256"].as_str().unwrap(),
                expected["prompt"].as_str().unwrap()
            )
        );
        assert_eq!(
            std::fs::read_to_string(&store).unwrap(),
            "unchanged store sentinel"
        );
        assert!(!scratch.path().join(".greppy").exists());
        assert_eq!(std::fs::read_dir(scratch.path()).unwrap().count(), 1);
    }
}
