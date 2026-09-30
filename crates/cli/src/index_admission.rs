//! Automatic index jobs participate in a configured host-wide development gate.
//! Admission wraps the whole child lifetime, rather than a racy status probe.
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) fn command(executable: &Path, job: &Path) -> io::Result<(Command, Option<PathBuf>)> {
    let configured = std::env::var_os("GREPPY_HEAVY_GATE").map(PathBuf::from);
    let default = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".codex/bin/dev-heavy-run.py"));
    let gate = match configured {
        Some(path) if path.is_file() => Some(path),
        Some(_) => {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "GREPPY_HEAVY_GATE must name an existing admission script; no index work started",
            ));
        }
        None => default.filter(|path| path.is_file()),
    };
    let Some(gate) = gate else {
        return Ok((Command::new(executable), None));
    };
    // A real inherited lease keeps an already-admitted workflow from trying to
    // acquire its own exclusive lease again. An environment flag is not proof.
    if default_gate_lease_is_inherited(&gate) {
        return Ok((Command::new(executable), None));
    }
    gated_command(executable, &gate, job)
}

fn gated_command(
    executable: &Path,
    gate: &Path,
    job: &Path,
) -> io::Result<(Command, Option<PathBuf>)> {
    let stderr = job.with_extension("admission-stderr");
    let mut options = fs::OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let output = options.open(&stderr)?;
    let mut command = Command::new(if cfg!(target_os = "macos") {
        "/usr/bin/python3"
    } else {
        "python3"
    });
    command
        .arg(gate)
        .arg("--owner")
        .arg(format!("greppy-auto-index-{}", std::process::id()))
        .arg("--project")
        .arg("greppy")
        .arg("--task")
        .arg(format!("auto-index-{}", std::process::id()))
        .arg("--")
        .arg(executable)
        .stderr(output);
    Ok((command, Some(stderr)))
}

pub(crate) fn failure_detail(
    stderr: Option<&Path>,
    status: std::process::ExitStatus,
) -> Option<String> {
    let path = stderr?;
    use io::Read;
    let mut bytes = Vec::new();
    fs::File::open(path)
        .ok()?
        .take(4096)
        .read_to_end(&mut bytes)
        .ok()?;
    let detail = String::from_utf8_lossy(&bytes).trim().to_string();
    Some(if status.code() == Some(75) {
        format!(
            "Automatic indexing deferred by shared host admission; no index work started. {detail} Retry the original command when host capacity is available. For an immediate bounded source read, use greppy read-file PATH --lines A:B."
        )
    } else {
        format!("Automatic index admission runner exited {status}: {detail}")
    })
}

#[cfg(unix)]
fn default_gate_lease_is_inherited(gate: &Path) -> bool {
    use std::os::unix::{fs::MetadataExt, io::AsRawFd};
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };
    if gate != home.join(".codex/bin/dev-heavy-run.py") {
        return false;
    }
    let lock = home.join(".codex/run/heavy-job.lock");
    let Ok(probe) = fs::File::open(&lock) else {
        return false;
    };
    let Ok(metadata) = probe.metadata() else {
        return false;
    };
    let Ok(record) = fs::read(&lock) else {
        return false;
    };
    if record.len() > 4096 {
        return false;
    }
    let Ok(record) = serde_json::from_slice::<serde_json::Value>(&record) else {
        return false;
    };
    let Some(owner) = record.get("pid").and_then(serde_json::Value::as_u64) else {
        return false;
    };
    // The gate ancestor is still waiting for this process and cannot release
    // the lock during the descriptor check. This excludes stale lease records.
    if !ancestor_contains(owner) {
        return false;
    }
    unsafe {
        if libc::flock(probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 {
            libc::flock(probe.as_raw_fd(), libc::LOCK_UN);
            return false;
        }
    }
    let directory = if cfg!(target_os = "linux") {
        "/proc/self/fd"
    } else {
        "/dev/fd"
    };
    let Ok(entries) = fs::read_dir(directory) else {
        return false;
    };
    for entry in entries.flatten() {
        let Some(fd) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<i32>().ok())
        else {
            continue;
        };
        if fd < 3 || fd == probe.as_raw_fd() {
            continue;
        }
        let Ok(candidate) = fs::metadata(entry.path()) else {
            continue;
        };
        if candidate.dev() != metadata.dev() || candidate.ino() != metadata.ino() {
            continue;
        }
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags < 0 || flags & libc::FD_CLOEXEC != 0 {
                continue;
            }
            // flock succeeds on the same inherited open-file description, but
            // fails on another descriptor while the gate ancestor holds it.
            if libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) == 0 {
                return true;
            }
        }
    }
    false
}

#[cfg(unix)]
fn ancestor_contains(owner: u64) -> bool {
    let Ok(output) = Command::new("/bin/ps")
        .args(["-axo", "pid=,ppid="])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let parents: std::collections::HashMap<u64, u64> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            Some((fields.next()?.parse().ok()?, fields.next()?.parse().ok()?))
        })
        .collect();
    let mut pid = u64::from(std::process::id());
    for _ in 0..32 {
        let Some(parent) = parents.get(&pid).copied() else {
            return false;
        };
        if parent == owner {
            return true;
        }
        if parent <= 1 || parent == pid {
            return false;
        }
        pid = parent;
    }
    false
}

#[cfg(not(unix))]
fn default_gate_lease_is_inherited(_: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejecting_gate_never_runs_index_and_preserves_capacity_diagnostic() {
        let tmp = tempfile::tempdir().unwrap();
        let gate = tmp.path().join("gate.py");
        fs::write(
            &gate,
            "import sys\nprint('Capacity gate: tmp below 20 GiB', file=sys.stderr)\nsys.exit(75)\n",
        )
        .unwrap();
        let (mut command, stderr) = gated_command(
            Path::new("__index_must_not_execute__"),
            &gate,
            &tmp.path().join("job.json"),
        )
        .unwrap();
        command.arg("index");
        let status = command.status().unwrap();
        assert_eq!(status.code(), Some(75));
        let detail = failure_detail(stderr.as_deref(), status).unwrap();
        assert!(detail.contains("no index work started"));
        assert!(detail.contains("tmp below 20 GiB"));
        assert!(detail.contains("Retry the original command"));
    }
    #[test]
    fn admitted_gate_receives_literal_child_arguments() {
        let tmp = tempfile::tempdir().unwrap();
        let gate = tmp.path().join("gate.py");
        fs::write(&gate, "import sys, subprocess\ni=sys.argv.index('--')\nsys.exit(subprocess.call(sys.argv[i+1:]))\n").unwrap();
        let python = if cfg!(target_os = "macos") {
            "/usr/bin/python3"
        } else {
            "python3"
        };
        let (mut command, _) =
            gated_command(Path::new(python), &gate, &tmp.path().join("job.json")).unwrap();
        command.args([
            "-c",
            "import sys; assert sys.argv[1] == 'path with ; literal spaces'",
            "path with ; literal spaces",
        ]);
        assert!(command.status().unwrap().success());
    }
    #[cfg(unix)]
    #[test]
    fn an_unrelated_pid_cannot_authorize_lease_reuse() {
        assert!(!ancestor_contains(u64::MAX));
    }
}
