//! Automatic index jobs participate in a configured host-wide development gate.
//! Admission wraps the whole child lifetime, rather than a racy status probe.
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) fn command(executable: &Path, job: &Path) -> io::Result<(Command, Option<PathBuf>)> {
    let Some(gate) = configured_gate()? else {
        return Ok((Command::new(executable), None));
    };
    // A real inherited lease keeps an already-admitted workflow from trying to
    // acquire its own exclusive lease again. An environment flag is not proof.
    if default_gate_lease_is_inherited(&gate) {
        return Ok((Command::new(executable), None));
    }
    gated_command(executable, &gate, job)
}

fn configured_gate() -> io::Result<Option<PathBuf>> {
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
    Ok(gate)
}

/// Ungated inline refresh must never compete with an admitted host job.
/// Returning false routes normal queries to the existing gated child refresh.
pub(crate) fn inline_refresh_is_admitted() -> bool {
    match configured_gate() {
        Ok(gate) => inline_refresh_allowed_for_gate(gate.as_deref()),
        Err(_) => false,
    }
}

fn inline_refresh_allowed_for_gate(gate: Option<&Path>) -> bool {
    match gate {
        Some(gate) => default_gate_lease_is_inherited(gate),
        None => true,
    }
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
            "Automatic indexing deferred by shared host admission; no index work started. {detail} Retry the original command when host capacity is available. For an immediate bounded source read, use greppy read-file PATH --lines A:B. For an edit without a graph refresh, use greppy replace-text PATH OLD NEW; it refuses missing or non-unique matches."
        )
    } else {
        format!("Automatic index admission runner exited {status}: {detail}")
    })
}

#[cfg(unix)]
fn default_gate_lease_is_inherited(gate: &Path) -> bool {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };
    if gate != home.join(".codex/bin/dev-heavy-run.py") {
        return false;
    }
    inherited_lease_owned_by_ancestor(&home.join(".codex/run/heavy-job.lock"))
}

#[cfg(unix)]
fn inherited_lease_owned_by_ancestor(lock: &Path) -> bool {
    use std::os::unix::{fs::MetadataExt, io::AsRawFd};
    let Ok(probe) = fs::File::open(lock) else {
        return false;
    };
    let Ok(metadata) = probe.metadata() else {
        return false;
    };
    let Ok(record) = fs::read(lock) else {
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
        if io::Error::last_os_error().kind() != io::ErrorKind::WouldBlock {
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
        // macOS stat(/dev/fd/N) reports the fdesc filesystem's device,
        // not the backing file's device. Compare the descriptor itself.
        let mut candidate = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(fd, candidate.as_mut_ptr()) } != 0 {
            continue;
        }
        let candidate = unsafe { candidate.assume_init() };
        // libc stat field widths and signedness differ between macOS and Linux.
        #[allow(clippy::unnecessary_cast)]
        let same_backing_file =
            candidate.st_dev as u64 == metadata.dev() && candidate.st_ino as u64 == metadata.ino();
        if !same_backing_file {
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
    // Shell/Node/Bun subprocess APIs may close inherited descriptors. The
    // admitted ancestor still waits for the child and holds the real lease.
    // Authenticate the ancestor and verify its still-held open-file-description
    // lease. An open descriptor or a stale lock record alone is insufficient.
    ancestor_lease_witness(owner, &record, &metadata)
}

#[cfg(unix)]
fn ancestor_lease_witness(owner: u64, record: &serde_json::Value, metadata: &fs::Metadata) -> bool {
    use std::io::{BufRead, Write};
    use std::os::unix::{fs::MetadataExt, io::AsRawFd, net::UnixStream};
    let Some(socket) = record.get("lease_witness").and_then(serde_json::Value::as_str) else {
        return false;
    };
    let Ok(mut stream) = UnixStream::connect(socket) else {
        return false;
    };
    let timeout = Some(std::time::Duration::from_millis(500));
    if stream.set_read_timeout(timeout).is_err() || stream.set_write_timeout(timeout).is_err() {
        return false;
    }
    // Authenticate the server's kernel PID, not a PID supplied in its JSON.
    #[cfg(target_os = "linux")]
    let peer = {
        let mut credentials = std::mem::MaybeUninit::<libc::ucred>::uninit();
        let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        let result = unsafe { libc::getsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, credentials.as_mut_ptr().cast(), &mut length) };
        if result != 0 { return false; }
        unsafe { credentials.assume_init() }.pid
    };
    #[cfg(target_os = "macos")]
    let peer = {
        let mut pid: libc::pid_t = 0;
        let mut length = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
        // Darwin sys/un.h: SOL_LOCAL=0, LOCAL_PEERPID=2.
        let result = unsafe { libc::getsockopt(stream.as_raw_fd(), 0, 2, (&mut pid as *mut libc::pid_t).cast(), &mut length) };
        if result != 0 { return false; }
        pid
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let peer: libc::pid_t = 0;
    if u64::try_from(peer).ok() != Some(owner) { return false; }
    let challenge = format!("{}-{:?}", std::process::id(), std::time::SystemTime::now());
    let request = serde_json::json!({"challenge": challenge});
    if writeln!(stream, "{request}").is_err() { return false; }
    let mut line = String::new();
    if std::io::BufReader::new(std::io::Read::take(stream, 2048)).read_line(&mut line).is_err() { return false; }
    let Ok(reply) = serde_json::from_str::<serde_json::Value>(&line) else { return false; };
    reply.get("challenge").and_then(serde_json::Value::as_str) == Some(challenge.as_str())
        && reply.get("owns_lease").and_then(serde_json::Value::as_bool) == Some(true)
        && reply.get("dev").and_then(serde_json::Value::as_u64) == Some(metadata.dev())
        && reply.get("ino").and_then(serde_json::Value::as_u64) == Some(metadata.ino())
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
    fn configured_gate_requires_child_admission_before_inline_refresh() {
        let tmp = tempfile::tempdir().unwrap();
        let gate = tmp.path().join("gate.py");
        fs::write(&gate, "raise SystemExit(75)\n").unwrap();
        assert!(!inline_refresh_allowed_for_gate(Some(&gate)));
        assert!(inline_refresh_allowed_for_gate(None));
    }

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
        assert!(detail.contains("greppy replace-text PATH OLD NEW"));
        assert!(detail.contains("refuses missing or non-unique matches"));
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

    #[cfg(unix)]
    #[test]
    fn lease_probe_child() {
        let Some(path) = std::env::var_os("GREPPY_TEST_ADMISSION_LEASE") else {
            return;
        };
        let expected = std::env::var("GREPPY_TEST_ADMISSION_INHERITED").unwrap() == "yes";
        assert_eq!(
            inherited_lease_owned_by_ancestor(Path::new(&path)),
            expected
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn authenticated_ancestor_witness_survives_closed_fds_but_not_foreign_locks() {
        let tmp = tempfile::tempdir().unwrap();
        let python = if cfg!(target_os = "macos") { "/usr/bin/python3" } else { "python3" };
        let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/dev_heavy_lease.py");
        let script = r#"import fcntl,json,os,subprocess,sys,importlib.util
spec=importlib.util.spec_from_file_location('witness',sys.argv[4]); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with open(sys.argv[1],'w+') as lease:
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with module.LeaseWitness(lease,sys.argv[1]) as witness:
        json.dump({'pid':os.getpid(),'lease_witness':witness.path},lease);lease.flush()
        holder=None
        if sys.argv[3]=='foreign':
            fcntl.flock(lease,fcntl.LOCK_UN)
            holder=subprocess.Popen([sys.executable,'-c',"import fcntl,sys,time; f=open(sys.argv[1]);fcntl.flock(f,fcntl.LOCK_EX);print('locked',flush=True);time.sleep(20)",sys.argv[1]],stdout=subprocess.PIPE,text=True)
            assert holder.stdout.readline().strip()=='locked'
        env=dict(os.environ,GREPPY_TEST_ADMISSION_LEASE=sys.argv[1],GREPPY_TEST_ADMISSION_INHERITED='no' if holder else 'yes')
        try: result=subprocess.run([sys.argv[2],'--exact','index_admission::tests::lease_probe_child','--nocapture'],env=env,close_fds=True)
        finally:
            if holder: holder.terminate();holder.wait(timeout=5)
        sys.exit(result.returncode)
"#;
        for mode in ["owned", "foreign"] {
            let output = Command::new(python).arg("-c").arg(script)
                .arg(tmp.path().join("lease.lock")).arg(std::env::current_exe().unwrap())
                .arg(mode).arg(&helper).output().unwrap();
            assert!(output.status.success(), "{} {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        }
    }

    #[cfg(unix)]
    #[test]
    fn legacy_gate_without_witness_remains_fail_closed_after_descriptor_closure() {
        let tmp = tempfile::tempdir().unwrap();
        let python = if cfg!(target_os = "macos") {
            "/usr/bin/python3"
        } else {
            "python3"
        };
        let script = r#"import fcntl,json,os,subprocess,sys
with open(sys.argv[1], 'w+') as lease:
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    json.dump({'pid':os.getpid()}, lease); lease.flush()
    env=dict(os.environ,GREPPY_TEST_ADMISSION_LEASE=sys.argv[1],GREPPY_TEST_ADMISSION_INHERITED='yes' if sys.argv[3]=='inherited' else 'no')
    if sys.argv[3]=='released': fcntl.flock(lease, fcntl.LOCK_UN)
    inherited=(lease.fileno(),) if sys.argv[3]=='inherited' else ()
    result=subprocess.run([sys.argv[2],'--exact','index_admission::tests::lease_probe_child','--nocapture'],env=env,pass_fds=inherited)
    sys.exit(result.returncode)
"#;
        for expected in ["inherited", "closed", "released"] {
            let output = Command::new(python)
                .arg("-c")
                .arg(script)
                .arg(tmp.path().join("lease.lock"))
                .arg(std::env::current_exe().unwrap())
                .arg(expected)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        }
    }
}
