//! WP19/WP21 regression: sandboxed greppy tool must write the *narrow*
//! greppy store paths (isolated agent data root + locks) without granting
//! the operator's global greppy data root, platform cache, or global temp.
//!
//! macOS-gated: Seatbelt is the platform that reproduced the production
//! failure. Exercises the real GreppyEnv Enforce path with the real greppy
//! binary — not a stub.
//!
//! The gate is on the whole file, not on the two test functions. With only the
//! tests gated, every helper below them is dead code on Linux, and `clippy
//! -D warnings` turns that into ten compile errors — which is how this file
//! took the ubuntu job red.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use greppy_agent::{sandbox, ExecutionEnv, GreppyEnv, SandboxMode};
use serde_json::json;

static SEQ: AtomicU64 = AtomicU64::new(0);
/// Both tests mutate process-global sandbox/build environment; serialize.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct ScopedEnv {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl ScopedEnv {
    fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }

    fn remove(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        std::env::remove_var(key);
        Self { key, previous }
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        if let Some(value) = &self.previous {
            std::env::set_var(self.key, value);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique(tag: &str) -> PathBuf {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!(
        "greppy-wp21-{tag}-{}-{}-{}",
        std::process::id(),
        seq,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn git(cwd: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .expect("spawn git");
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn init_git_fixture(root: &Path) {
    git(root, &["init"]);
    git(root, &["checkout", "-b", "main"]);
    git(root, &["config", "user.name", "fixture"]);
    git(root, &["config", "user.email", "fixture@test.local"]);
    git(root, &["config", "commit.gpgsign", "false"]);
    std::fs::write(root.join("main.rs"), b"fn main() {}\n").unwrap();
    git(root, &["add", "main.rs"]);
    git(root, &["commit", "-m", "initial"]);
}

/// Narrow root list matching `crates/cli/src/agent.rs::writable_roots_for`
/// after WP21 (worktree, per-run scratch, isolated agent data, cargo
/// registry/git). Kept local so the regression stays focused on the Enforce
/// path rather than exporting the CLI helper.
fn agent_writable_roots(worktree: &Path, scratch: &Path, agent_data: &Path) -> Vec<PathBuf> {
    let cargo = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/"));
            home.join(".cargo")
        });
    let registry = cargo.join("registry");
    let git_cache = cargo.join("git");
    let _ = std::fs::create_dir_all(&registry);
    let _ = std::fs::create_dir_all(&git_cache);
    vec![
        worktree.to_path_buf(),
        scratch.to_path_buf(),
        agent_data.to_path_buf(),
        registry,
        git_cache,
    ]
}

fn sandbox_exec_available() -> bool {
    Path::new("/usr/bin/sandbox-exec").exists()
}

#[cfg(target_os = "macos")]
#[test]
fn enforce_index_backed_where_am_i_not_permission_denied() {
    if !sandbox_exec_available() {
        return;
    }
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Isolated greppy data (mirrors production GREPPY_STORE_DIR for agent runs).
    let base = unique("base");
    let worktree = base.join("wt");
    std::fs::create_dir_all(&worktree).unwrap();
    init_git_fixture(&worktree);

    let agent_data = base.join("greppy-agent-data");
    std::fs::create_dir_all(&agent_data).unwrap();
    let scratch = base.join("greppy-agent-scratch").join("run");
    std::fs::create_dir_all(&scratch).unwrap();
    let shared_base = base.join("shared-base");
    std::fs::create_dir_all(&shared_base).unwrap();
    let shared_base_graph = shared_base.join("graph.db");
    std::fs::write(&shared_base_graph, b"immutable-base").unwrap();

    // Capture the process global temp BEFORE overriding TMPDIR for children.
    let global_temp = std::env::temp_dir();

    // Safety: set only for this process; integration tests run in their own
    // binary process so this cannot poison parallel unit-test threads.
    let store_env = ScopedEnv::set("GREPPY_STORE_DIR", &agent_data);
    let tmp_env = ScopedEnv::set("TMPDIR", &scratch);
    let base_store_env = ScopedEnv::set("GREPPY_AGENT_BASE_STORE", &shared_base_graph);

    let roots = agent_writable_roots(&worktree, &scratch, &agent_data);
    assert!(
        roots.iter().any(|r| r == &agent_data),
        "test roots must include isolated agent data: {roots:?}"
    );
    // Must NOT include global shared state (whole temp / platform cache / cargo home).
    assert!(
        !roots.iter().any(|r| r == &global_temp),
        "global temp must not be a root: {roots:?}"
    );
    assert!(
        !roots.iter().any(|r| {
            let s = r.to_string_lossy();
            (s.ends_with("/Caches") || s.ends_with("Caches") || s.ends_with("/.cache"))
                && !s.contains("greppy-agent")
        }),
        "platform cache must not be a root: {roots:?}"
    );
    assert!(
        !roots.iter().any(|r| r.ends_with(".cargo")),
        "whole cargo home must not be a root: {roots:?}"
    );
    assert!(
        !roots.iter().any(|root| shared_base_graph.starts_with(root)),
        "published Base must stay outside every writable sandbox root: {roots:?}"
    );

    let mode = match sandbox::resolve_enforce_spec(&roots) {
        Ok(m) => m,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&base);
            panic!("resolve_enforce_spec failed: {e}");
        }
    };
    assert!(
        matches!(mode, SandboxMode::Enforce(_)),
        "expected Enforce on macOS with sandbox-exec"
    );

    let mut env = GreppyEnv::with_binary(binary_path(), worktree.clone())
        .expect("GreppyEnv")
        .with_sandbox(mode)
        .with_greppy_timeout(std::time::Duration::from_secs(60));

    // Index-backed command: opens graph.db and acquires a lifecycle lease under
    // agent_data/locks. Must NOT fail with a sandbox write refusal.
    let out = env.call_tool("greppy", &json!({"args": ["where-am-i"]}));
    let body = &out.content;
    assert!(
        !body.contains("Operation not permitted"),
        "index-backed where-am-i must not hit seatbelt denial; content={body}"
    );
    assert!(
        !(body.contains("lifecycle lease") && body.to_ascii_lowercase().contains("not permitted")),
        "lifecycle lease must not fail as a permission error; content={body}"
    );
    assert!(
        !body.contains("this run is write-confined to the repository worktree"),
        "must not be classified as a sandbox write refusal; content={body}"
    );

    // Confinement: $HOME write denied.
    let home = std::env::var_os("HOME").expect("HOME");
    let escape = PathBuf::from(&home).join(format!(
        ".greppy-wp21-home-escape-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_file(&escape);
    let home_out = env.call_tool(
        "greppy",
        &json!({"args": [
            "bash-smart",
            "--",
            "bash",
            "-lc",
            format!("touch '{}'", escape.display())
        ]}),
    );
    assert!(
        home_out.is_error,
        "HOME write must be denied; content={}",
        home_out.content
    );
    assert!(
        !escape.exists(),
        "escape file under HOME must not exist: {}",
        escape.display()
    );
    let _ = std::fs::remove_file(&escape);

    // Confinement: write into platform cache OUTSIDE the worktree denied.
    let platform_cache = PathBuf::from(&home).join("Library").join("Caches");
    let cache_probe = platform_cache.join(format!(
        "greppy-wp21-cache-escape-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_file(&cache_probe);

    // Store-CoW invariant: the tool child may read the published Base through
    // SQLite, but the Base path itself is never writable inside the sandbox.
    let base_write = env.call_tool(
        "greppy",
        &json!({"args": [
            "bash-smart",
            "--",
            "bash",
            "-lc",
            format!("printf tamper >> '{}'", shared_base_graph.display())
        ]}),
    );
    assert!(
        base_write.is_error,
        "published Base write must be denied; content={}",
        base_write.content
    );
    assert_eq!(
        std::fs::read(&shared_base_graph).unwrap(),
        b"immutable-base",
        "sandboxed agent must not change published Base bytes"
    );
    let cache_out = env.call_tool(
        "greppy",
        &json!({"args": [
            "bash-smart",
            "--",
            "bash",
            "-lc",
            format!("touch '{}'", cache_probe.display())
        ]}),
    );
    assert!(
        cache_out.is_error,
        "platform-cache write outside worktree must be denied; content={}",
        cache_out.content
    );
    assert!(
        !cache_probe.exists(),
        "cache escape probe must not exist: {}",
        cache_probe.display()
    );
    let _ = std::fs::remove_file(&cache_probe);

    // TMPDIR for the tool child must be inside the granted scratch dir.
    let tmp_out = env.call_tool(
        "greppy",
        &json!({"args": [
            "bash-smart",
            "--",
            "bash",
            "-lc",
            "printf '%s' \"$TMPDIR\""
        ]}),
    );
    assert!(
        !tmp_out.is_error,
        "reading TMPDIR must succeed; content={}",
        tmp_out.content
    );
    let reported = tmp_out.content.trim();
    // bash-smart may wrap output; accept any line that contains the scratch path.
    let scratch_s = scratch.to_string_lossy();
    assert!(
        reported.contains(scratch_s.as_ref())
            || std::fs::canonicalize(&scratch)
                .ok()
                .map(|c| reported.contains(&*c.to_string_lossy()))
                .unwrap_or(false),
        "TMPDIR must be inside scratch; reported={reported:?} scratch={scratch_s}"
    );

    drop((base_store_env, tmp_env, store_env));
    let _ = std::fs::remove_dir_all(&base);
}

#[cfg(target_os = "macos")]
#[test]
fn enforce_cargo_test_in_scratch_crate_still_works() {
    if !sandbox_exec_available() {
        return;
    }
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let base = unique("cargo");
    let worktree = base.join("wt");
    std::fs::create_dir_all(&worktree).unwrap();
    init_git_fixture(&worktree);

    let agent_data = base.join("greppy-agent-data");
    std::fs::create_dir_all(&agent_data).unwrap();
    let scratch = base.join("greppy-agent-scratch").join("run");
    std::fs::create_dir_all(&scratch).unwrap();
    let store_env = ScopedEnv::set("GREPPY_STORE_DIR", &agent_data);
    let tmp_env = ScopedEnv::set("TMPDIR", &scratch);
    // A host/CI target directory may sit outside the deliberately narrow
    // sandbox roots. This fixture verifies worktree-local builds, so force
    // Cargo back to its ordinary `tiny/target` rather than granting a shared
    // writable build cache to the agent.
    let cargo_target_env = ScopedEnv::remove("CARGO_TARGET_DIR");

    let roots = agent_writable_roots(&worktree, &scratch, &agent_data);
    let mode = match sandbox::resolve_enforce_spec(&roots) {
        Ok(m) => m,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&base);
            panic!("resolve_enforce_spec failed: {e}");
        }
    };

    // Tiny crate under the worktree; cargo test must be able to use registry/git caches.
    let crate_dir = worktree.join("tiny");
    std::fs::create_dir_all(crate_dir.join("src")).unwrap();
    std::fs::write(
        crate_dir.join("Cargo.toml"),
        b"[package]\nname = \"tiny\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(crate_dir.join("src/lib.rs"), b"pub fn n() -> i32 { 1 }\n#[cfg(test)]\nmod tests { #[test] fn t() { assert_eq!(super::n(), 1); } }\n").unwrap();

    let mut env = GreppyEnv::with_binary(binary_path(), worktree.clone())
        .expect("GreppyEnv")
        .with_sandbox(mode)
        .with_bash_timeout(std::time::Duration::from_secs(180));

    let out = env.call_tool(
        "greppy",
        &json!({"args": [
            "bash-smart",
            "--",
            "bash",
            "-lc",
            "cd tiny && cargo test -q"
        ]}),
    );
    assert!(
        !out.is_error,
        "cargo test in scratch crate must succeed under narrow roots; content={}",
        out.content
    );

    drop((cargo_target_env, tmp_env, store_env));
    let _ = std::fs::remove_dir_all(&base);
}
