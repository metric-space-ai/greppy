//! Automatic index jobs participate in a configured host-wide development gate.
//! Admission wraps the whole child lifetime, rather than a racy status probe.
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

/// What an automatic job asks of the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Work {
    /// A first index or embedding work: a heavy job under the host gate.
    Heavy,
    /// A structural refresh of an already published index. It is bounded
    /// graph work, so it does not queue behind builds for the host's
    /// exclusive lease; greppy limits how many run at once by itself.
    Refresh,
}

/// Set for the child of a [`Work::Refresh`] job: it must hold a refresh slot.
pub(crate) const ENV_REFRESH_LANE: &str = "GREPPY_REFRESH_LANE";
/// How many structural refreshes may run on this host at once (default 2).
/// `0` turns the lane off: every refresh is a heavy job again.
pub(crate) const ENV_REFRESH_SLOTS: &str = "GREPPY_REFRESH_SLOTS";
const DEFAULT_REFRESH_SLOTS: usize = 2;

pub(crate) fn refresh_slots() -> usize {
    std::env::var(ENV_REFRESH_SLOTS)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map_or(DEFAULT_REFRESH_SLOTS, |slots| slots.min(16))
}

/// Take one of the host's refresh slots for the life of this process.
/// `Ok(None)`: all of them are held by other refreshes.
pub(crate) fn acquire_refresh_slot() -> io::Result<Option<greppy_core::cache::FileLock>> {
    // Host-wide, not per store directory: agents with separate stores still
    // share the machine's memory.
    let root = std::env::temp_dir().join(format!("greppy-refresh-lane-{}", user_tag()));
    for slot in 0..refresh_slots() {
        if let Some(lock) = greppy_core::cache::acquire_named_lock_in(
            &root,
            &format!("refresh-slot-{slot}"),
            greppy_core::cache::LockMode::Exclusive,
            true,
        )? {
            return Ok(Some(lock));
        }
    }
    Ok(None)
}

#[cfg(unix)]
fn user_tag() -> String {
    unsafe { libc::geteuid() }.to_string()
}

#[cfg(not(unix))]
fn user_tag() -> String {
    std::env::var("USERNAME").unwrap_or_else(|_| "user".into())
}

fn refresh_lane_command(executable: &Path, job: &Path) -> io::Result<(Command, Option<PathBuf>)> {
    let stderr = job.with_extension("admission-stderr");
    let mut options = fs::OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let output = options.open(&stderr)?;
    let mut command = Command::new(executable);
    command.env(ENV_REFRESH_LANE, "1").stderr(output);
    Ok((command, Some(stderr)))
}

pub(crate) fn command(
    executable: &Path,
    job: &Path,
    work: Work,
) -> io::Result<(Command, Option<PathBuf>)> {
    if work == Work::Refresh {
        return refresh_lane_command(executable, job);
    }
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
    let mut command = gate_command(gate, "auto-index", executable);
    command.stderr(output);
    Ok((command, Some(stderr)))
}

/// `executable` run by the host gate as task `<task>-<pid>` of project greppy.
fn gate_command(gate: &Path, task: &str, executable: &Path) -> Command {
    let mut command = Command::new(if cfg!(target_os = "macos") {
        "/usr/bin/python3"
    } else {
        "python3"
    });
    // The gate's disk floor protects the volume the job writes to. A greppy
    // index job writes its store and inference cache (and the TMPDIR the gate
    // assigns); tell the gate where, so a store on another volume is not
    // refused for a full system disk it never touches.
    if let Ok(writes) = std::env::join_paths([
        greppy_core::cache::data_root(),
        greppy_core::cache::shared_inference_root(),
    ]) {
        command.env("DEV_HEAVY_WRITES", writes);
    }
    let pid = std::process::id();
    command
        .arg(gate)
        .arg("--owner")
        .arg(format!("greppy-{task}-{pid}"))
        .arg("--project")
        .arg("greppy")
        .arg("--task")
        .arg(format!("{task}-{pid}"))
        .arg("--")
        .arg(executable);
    command
}

/// Set for an index command the host gate admitted: it runs the work itself.
const ENV_INDEX_ADMITTED: &str = "GREPPY_INDEX_ADMITTED";

/// Run this `greppy index` command again under the host gate and return its
/// exit code: the gate's own 75 when the host is busy. `None` when there is
/// no gate, this command already is the admitted one, or it runs inside a
/// job the gate admitted.
pub(crate) fn rerun_under_gate() -> io::Result<Option<i32>> {
    if std::env::var_os(ENV_INDEX_ADMITTED).is_some() {
        return Ok(None);
    }
    let Some(gate) = configured_gate()? else {
        return Ok(None);
    };
    if default_gate_lease_is_inherited(&gate) {
        return Ok(None);
    }
    let executable = std::env::current_exe()?;
    let status = gate_command(&gate, "index", &executable)
        .args(std::env::args_os().skip(1))
        .env(ENV_INDEX_ADMITTED, "1")
        .status()?;
    Ok(Some(status.code().unwrap_or(70)))
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
    let Some(socket) = record
        .get("lease_witness")
        .and_then(serde_json::Value::as_str)
    else {
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
        let result = unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                credentials.as_mut_ptr().cast(),
                &mut length,
            )
        };
        if result != 0 {
            return false;
        }
        unsafe { credentials.assume_init() }.pid
    };
    #[cfg(target_os = "macos")]
    let peer = {
        let mut pid: libc::pid_t = 0;
        let mut length = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
        // Darwin sys/un.h: SOL_LOCAL=0, LOCAL_PEERPID=2.
        let result = unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                0,
                2,
                (&mut pid as *mut libc::pid_t).cast(),
                &mut length,
            )
        };
        if result != 0 {
            return false;
        }
        pid
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let peer: libc::pid_t = 0;
    if u64::try_from(peer).ok() != Some(owner) {
        return false;
    }
    let challenge = format!("{}-{:?}", std::process::id(), std::time::SystemTime::now());
    let request = serde_json::json!({"challenge": challenge});
    if writeln!(stream, "{request}").is_err() {
        return false;
    }
    let mut line = String::new();
    if std::io::BufReader::new(std::io::Read::take(stream, 2048))
        .read_line(&mut line)
        .is_err()
    {
        return false;
    }
    let Ok(reply) = serde_json::from_str::<serde_json::Value>(&line) else {
        return false;
    };
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

/// The default host gate is about to wrap a new child. If another owner
/// already holds its lease, queueing behind `dev-heavy-run.py` is a silent
/// multi-minute wait that still ends in exit 75. Record the deferral now.
pub(crate) fn occupied_gate_deferral() -> Option<String> {
    let gate = configured_gate().ok().flatten()?;
    if default_gate_lease_is_inherited(&gate) {
        return None;
    }
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    if gate != home.join(".codex/bin/dev-heavy-run.py") {
        return None;
    }
    if !foreign_lock_is_held(&home.join(".codex/run/heavy-job.lock")) {
        return None;
    }
    Some(
        "Automatic indexing deferred by shared host admission; no index work started. Capacity gate: another owner holds the heavy-job lease. Retry the original command when host capacity is available. For an immediate bounded source read, use greppy read-file PATH --lines A:B. For an edit without a graph refresh, use greppy replace-text PATH OLD NEW; it refuses missing or non-unique matches."
            .to_string(),
    )
}

/// `true` when a different open-file description already holds an exclusive
/// flock. A missing lock is free. Tests must pass a temporary path, never
/// the user's real heavy-job lock.
pub(crate) fn foreign_lock_is_held(lock: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if !lock.is_file() {
            return false;
        }
        let Ok(file) = fs::File::open(lock) else {
            return false;
        };
        unsafe {
            if libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 {
                let _ = libc::flock(file.as_raw_fd(), libc::LOCK_UN);
                return false;
            }
            let error = io::Error::last_os_error();
            return error.kind() == io::ErrorKind::WouldBlock
                || error.raw_os_error() == Some(libc::EAGAIN)
                || error.raw_os_error() == Some(libc::EWOULDBLOCK);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = lock;
        false
    }
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
        let python = if cfg!(target_os = "macos") {
            "/usr/bin/python3"
        } else {
            "python3"
        };
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
            let output = Command::new(python)
                .arg("-c")
                .arg(script)
                .arg(tmp.path().join("lease.lock"))
                .arg(std::env::current_exe().unwrap())
                .arg(mode)
                .arg(&helper)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
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

    #[cfg(unix)]
    #[test]
    fn foreign_lock_reports_only_a_held_lease() {
        use std::io::Read;
        let tmp = tempfile::tempdir().unwrap();
        let lock = tmp.path().join("heavy-job.lock");
        fs::write(&lock, b"{}\n").unwrap();
        assert!(!foreign_lock_is_held(&lock));
        let python = if cfg!(target_os = "macos") {
            "/usr/bin/python3"
        } else {
            "python3"
        };
        let helper = tmp.path().join("hold.py");
        fs::write(
            &helper,
            "import fcntl,sys,time\n\
             lease=open(sys.argv[1],'a+')\n\
             fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)\n\
             sys.stdout.write('held\\n')\n\
             sys.stdout.flush()\n\
             time.sleep(30)\n",
        )
        .unwrap();
        let mut child = Command::new(python)
            .arg(&helper)
            .arg(&lock)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let mut buf = [0u8; 5];
        let n = stdout.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"held\n");
        assert!(
            foreign_lock_is_held(&lock),
            "a foreign flock holder must not look free"
        );
        let _ = child.kill();
        let _ = child.wait();
        assert!(!foreign_lock_is_held(&lock));
    }
}
