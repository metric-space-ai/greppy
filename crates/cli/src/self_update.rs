//! `greppy update`: move this installation to the newest (or a named) release.
//!
//! The work is the public installer's: it resolves the platform asset, verifies the release's
//! SHA256SUMS and switches the command link. `update` only decides the target tag, skips the
//! download when this binary already is that release, and keeps the user's agent setup as it is
//! (`--no-agents`). A binary from a system package (.deb/.rpm/.pkg) is updated through the
//! package manager (`--system`, the installer asks for sudo).

use std::path::{Path, PathBuf};
use std::process::Command;

const INSTALL_URL: &str = "https://greppy.xyz/install.sh";
const LATEST_RELEASE_API: &str =
    "https://api.github.com/repos/metric-space-ai/greppy/releases/latest";

fn versions_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("GREPPY_VERSIONS_DIR") {
        return Some(PathBuf::from(dir));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share/greppy/versions"))
}

/// True when this binary lives in the installer's per-user versions tree.
fn is_user_install(exe: &Path, versions: Option<&Path>) -> bool {
    versions.is_some_and(|dir| {
        let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        exe.starts_with(&dir)
    })
}

/// The shell program run for the update. Values arrive through the environment, never spliced
/// into the script text.
fn update_script() -> &'static str {
    r#"set -eu
tag="${GREPPY_UPDATE_TAG:-}"
if [ -z "$tag" ]; then
  tag=$(curl -fsSL "$GREPPY_UPDATE_API" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)
  [ -n "$tag" ] || { echo "greppy update: could not determine the latest release" >&2; exit 69; }
fi
if [ "$tag" = "v$GREPPY_UPDATE_CURRENT" ]; then
  echo "greppy $GREPPY_UPDATE_CURRENT is the latest release; nothing to update."
  exit 0
fi
echo "greppy update: $GREPPY_UPDATE_CURRENT -> $tag"
if [ -n "${GREPPY_UPDATE_SYSTEM:-}" ]; then
  curl -fsSL "$GREPPY_UPDATE_INSTALLER" | sh -s -- --yes --no-agents --system --version "$tag"
else
  curl -fsSL "$GREPPY_UPDATE_INSTALLER" | sh -s -- --yes --no-agents --version "$tag"
fi
"#
}

pub(crate) fn run(version: Option<String>) -> i32 {
    let exe = std::env::current_exe()
        .and_then(|path| path.canonicalize())
        .unwrap_or_default();
    let user = is_user_install(&exe, versions_dir().as_deref());
    let tag = version.map(|v| {
        if v.starts_with('v') {
            v
        } else {
            format!("v{v}")
        }
    });
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(update_script())
        .env("GREPPY_UPDATE_CURRENT", env!("CARGO_PKG_VERSION"))
        .env("GREPPY_UPDATE_TAG", tag.unwrap_or_default())
        .env(
            "GREPPY_UPDATE_INSTALLER",
            std::env::var("GREPPY_INSTALL_URL").unwrap_or_else(|_| INSTALL_URL.to_string()),
        )
        .env(
            "GREPPY_UPDATE_API",
            std::env::var("GREPPY_RELEASE_API").unwrap_or_else(|_| LATEST_RELEASE_API.to_string()),
        );
    if user {
        cmd.env_remove("GREPPY_UPDATE_SYSTEM");
    } else {
        cmd.env("GREPPY_UPDATE_SYSTEM", "1");
    }
    match cmd.status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(error) => {
            eprintln!("greppy update: cannot run sh: {error}");
            69
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn user_install_is_recognised_by_the_versions_tree() {
        let root = std::env::temp_dir().join(format!("greppy-update-test-{}", std::process::id()));
        let exe_dir = root.join("versions/v0.4.3/bin");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let exe = exe_dir.canonicalize().unwrap().join("greppy");
        assert!(is_user_install(&exe, Some(&root.join("versions"))));
        assert!(!is_user_install(
            Path::new("/usr/lib/greppy/bin/greppy"),
            Some(&root.join("versions"))
        ));
        assert!(!is_user_install(&exe, None));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn same_release_is_a_no_op_without_network() {
        let out = Command::new("sh")
            .arg("-c")
            .arg(update_script())
            .env("GREPPY_UPDATE_CURRENT", "9.9.9")
            .env("GREPPY_UPDATE_TAG", "v9.9.9")
            .env("GREPPY_UPDATE_INSTALLER", "http://127.0.0.1:9/never")
            .env("GREPPY_UPDATE_API", "http://127.0.0.1:9/never")
            .output()
            .unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("nothing to update"));
    }
}
