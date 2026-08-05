//! WP19 regression: sandboxed greppy tool must write greppy's data root
//! (locks / lifecycle leases / workspaces). Without that root, index-backed
//! commands fail with `Operation not permitted` / `lifecycle lease`.
//!
//! macOS-gated: Seatbelt is the platform that reproduced the production
//! failure. Exercises the real GreppyEnv Enforce path with the real greppy
//! binary — not a stub.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use greppy_agent::{sandbox, ExecutionEnv, GreppyEnv, SandboxMode};
use greppy_core::cache;
use serde_json::json;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique(tag: &str) -> PathBuf {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!(
        "greppy-wp19-{tag}-{}-{}-{}",
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

/// Same root list shape as `crates/cli/src/agent.rs::writable_roots_for`
/// (worktree, temp, greppy data root, cargo home, platform cache). Kept local
/// so the regression stays focused on the Enforce path rather than exporting
/// the CLI helper.
fn agent_writable_roots(worktree: &Path) -> Vec<PathBuf> {
    let cargo = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/"));
            home.join(".cargo")
        });
    #[cfg(target_os = "macos")]
    let platform_cache = {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/"));
        home.join("Library").join("Caches")
    };
    #[cfg(not(target_os = "macos"))]
    let platform_cache = {
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            PathBuf::from(xdg)
        } else {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/"));
            home.join(".cache")
        }
    };
    vec![
        worktree.to_path_buf(),
        std::env::temp_dir(),
        cache::data_root(),
        cargo,
        platform_cache,
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

    // Isolate greppy data so the test does not contend with a live store under
    // the operator's real Application Support path, and so prepare_writable_roots
    // can create the data root without side effects.
    let base = unique("data");
    let store_override = base.join("greppy-data");
    std::fs::create_dir_all(&store_override).unwrap();
    // Safety: set only for this process; integration tests run in their own
    // binary process so this cannot poison parallel unit-test threads.
    std::env::set_var("GREPPY_STORE_DIR", &store_override);

    let worktree = unique("wt");
    init_git_fixture(&worktree);

    let roots = agent_writable_roots(&worktree);
    assert!(
        roots.iter().any(|r| r == &cache::data_root()),
        "test roots must include greppy data root: {roots:?}"
    );

    let mode = match sandbox::resolve_enforce_spec(&roots) {
        Ok(m) => m,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&base);
            let _ = std::fs::remove_dir_all(&worktree);
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
    // data_root/locks. Must NOT fail with a sandbox write refusal — that was
    // the WP19 production bug (every index-backed tool abandoned greppy).
    //
    // Design: do not require a prebuilt index. Assert only that the failure mode
    // 'Operation not permitted' / 'lifecycle lease' does not appear. A cold
    // empty-store where-am-i is still free of that permission error once the
    // data root is writable.
    let out = env.call_tool("greppy", &json!({"args": ["where-am-i"]}));
    let body = &out.content;
    // The WP19 failure mode: seatbelt blocks data_root/locks → lease open fails
    // with "Operation not permitted" (and often "lifecycle lease" in the
    // message). Catch either signal; do not require a prebuilt index.
    assert!(
        !body.contains("Operation not permitted"),
        "index-backed where-am-i must not hit seatbelt denial; content={body}"
    );
    assert!(
        !(body.contains("lifecycle lease") && body.to_ascii_lowercase().contains("not permitted")),
        "lifecycle lease must not fail as a permission error; content={body}"
    );
    // The clarifying sandbox rule must not fire either (it keys off the same
    // refusal signals).
    assert!(
        !body.contains("this run is write-confined to the repository worktree"),
        "must not be classified as a sandbox write refusal; content={body}"
    );

    // Confinement still holds: $HOME write denied.
    let home = std::env::var_os("HOME").expect("HOME");
    let escape = PathBuf::from(&home).join(format!(
        ".greppy-wp19-home-escape-{}-{}",
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

    // Confinement still holds: $TMPDIR/../C sibling escape denied.
    let tmp = std::env::temp_dir();
    let probe_dir = tmp.join("..").join("C");
    let _ = std::fs::create_dir_all(&probe_dir);
    let probe = probe_dir.join(format!(
        "greppy-wp19-c-escape-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_file(&probe);
    let c_out = env.call_tool(
        "greppy",
        &json!({"args": [
            "bash-smart",
            "--",
            "bash",
            "-lc",
            format!("touch '{}'", probe.display())
        ]}),
    );
    assert!(
        c_out.is_error,
        "TMPDIR/../C write must be denied; content={}",
        c_out.content
    );
    assert!(
        !probe.exists(),
        "escape probe must not exist: {}",
        probe.display()
    );
    let _ = std::fs::remove_file(&probe);

    std::env::remove_var("GREPPY_STORE_DIR");
    let _ = std::fs::remove_dir_all(&base);
    let _ = std::fs::remove_dir_all(&worktree);
}
