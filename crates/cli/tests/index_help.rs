//! Storage help is available before indexing and must not create a store.
use std::process::Command;

#[test]
fn help_version_and_refused_usage_do_not_trigger_store_gc() {
    for (name, args, successful) in [
        ("help", vec!["index", "--help"], true),
        ("version", vec!["--version"], true),
        ("usage", vec!["replace-text"], false),
    ] {
        let base = std::env::temp_dir().join(format!("greppy-no-gc-{name}-{}", std::process::id()));
        assert!(!base.exists(), "test needs an unused storage path");
        let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .args(args)
            .env("GREPPY_STORE_DIR", &base)
            .output()
            .expect("run help/version/usage");
        assert_eq!(output.status.success(), successful, "{name}: {output:?}");
        assert!(!base.exists(), "{name} created store files at {base:?}");
    }
}

#[test]
fn index_help_explains_store_and_inference_roots_without_creating_them() {
    let base = std::env::temp_dir().join(format!("greppy-index-help-{}", std::process::id()));
    assert!(!base.exists(), "test needs an unused storage path");
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["index", "--help"])
        .env("GREPPY_STORE_DIR", base.join("store"))
        .env("GREPPY_SHARED_INFERENCE_ROOT", base.join("inference"))
        .output()
        .expect("run index help");
    assert!(!base.exists(), "help must not create storage directories");
    assert!(output.status.success(), "{:?}", output);
    let help = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "GREPPY_STORE_DIR",
        "GREPPY_SHARED_INFERENCE_ROOT",
        "TMPDIR controls temporary files",
        "XDG_CACHE_HOME does not select",
        "same value for status, queries and later edits",
        "does not move or delete an existing one",
        "greppy index status --json",
    ] {
        assert!(help.contains(expected), "missing {expected:?} in {help}");
    }
}
