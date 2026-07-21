#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CACHE_MAGIC: &[u8] = b"greppy-verify-cache-v1\0";
const MIRROR_CANDIDATES: &[&str] = &[".tox", ".venv", "venv", ".nox", "node_modules", "target"];

#[derive(Debug, Clone)]
pub(crate) struct CommandRun {
    pub(crate) exit_code: Option<i32>,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) timed_out: bool,
    pub(crate) spawn_error: Option<String>,
}

impl CommandRun {
    fn from_spawn_error(error: std::io::Error) -> Self {
        Self {
            exit_code: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            timed_out: false,
            spawn_error: Some(error.to_string()),
        }
    }

    pub(crate) fn combined_text(&self) -> String {
        let mut bytes = Vec::with_capacity(self.stdout.len() + self.stderr.len() + 1);
        bytes.extend_from_slice(&self.stdout);
        if !self.stdout.is_empty() && !self.stderr.is_empty() && !self.stdout.ends_with(b"\n") {
            bytes.push(b'\n');
        }
        bytes.extend_from_slice(&self.stderr);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub(crate) fn repository_root(cwd: &Path) -> Result<PathBuf, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|error| format!("cannot start git: {error}"))?;
    if !output.status.success() {
        return Err(first_output_line(
            &output.stderr,
            "not inside a git worktree",
        ));
    }
    let root = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if root.is_empty() {
        return Err("git returned an empty worktree root".into());
    }
    Ok(PathBuf::from(root))
}

pub(crate) fn resolve_revision(root: &Path, revision: &str) -> Result<String, String> {
    let expression = format!("{revision}^{{commit}}");
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--verify"])
        .arg(expression)
        .output()
        .map_err(|error| format!("cannot start git: {error}"))?;
    if !output.status.success() {
        return Err(first_output_line(
            &output.stderr,
            &format!("baseline revision `{revision}` is not a commit"),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Hash the index entries and live bytes of every tracked path. Untracked
/// files are deliberately excluded: this is a dirstate attestation, not a
/// repository archive hash.
pub(crate) fn workspace_digest(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-s", "-z"])
        .output()
        .map_err(|error| format!("cannot start git for workspace digest: {error}"))?;
    if !output.status.success() {
        return Err(first_output_line(
            &output.stderr,
            "git ls-files failed while computing workspace digest",
        ));
    }

    let mut hasher = Sha256::new();
    for record in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|r| !r.is_empty())
    {
        hasher.update((record.len() as u64).to_le_bytes());
        hasher.update(record);
        let Some(tab) = record.iter().position(|byte| *byte == b'\t') else {
            return Err("unexpected git ls-files record without path".into());
        };
        let path_bytes = &record[tab + 1..];
        let relative = path_from_git_bytes(path_bytes)?;
        let path = root.join(relative);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                hasher.update(b"symlink\0");
                let target = fs::read_link(&path)
                    .map_err(|error| format!("read symlink {}: {error}", path.display()))?;
                hasher.update(os_str_bytes(target.as_os_str()));
            }
            Ok(metadata) if metadata.is_file() => {
                hasher.update(b"file\0");
                let mut file = fs::File::open(&path)
                    .map_err(|error| format!("open tracked file {}: {error}", path.display()))?;
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let read = file.read(&mut buffer).map_err(|error| {
                        format!("read tracked file {}: {error}", path.display())
                    })?;
                    if read == 0 {
                        break;
                    }
                    hasher.update(&buffer[..read]);
                }
            }
            Ok(_) => hasher.update(b"other\0"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                hasher.update(b"missing\0")
            }
            Err(error) => {
                return Err(format!("stat tracked path {}: {error}", path.display()));
            }
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(unix)]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    use std::os::unix::ffi::OsStrExt;
    Ok(PathBuf::from(OsStr::from_bytes(bytes)))
}

#[cfg(windows)]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    String::from_utf8(bytes.to_vec())
        .map(PathBuf::from)
        .map_err(|_| "git emitted a non-UTF-8 tracked path on Windows".into())
}

#[cfg(unix)]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

#[cfg(windows)]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    value.to_string_lossy().as_bytes().to_vec()
}

pub(crate) fn run_command(argv: &[String], cwd: &Path, timeout: Duration) -> CommandRun {
    let Some(program) = argv.first() else {
        return CommandRun::from_spawn_error(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "test command is empty",
        ));
    };
    let mut child = match Command::new(program)
        .args(&argv[1..])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return CommandRun::from_spawn_error(error),
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = thread::spawn(move || read_pipe(stdout));
    let stderr_reader = thread::spawn(move || read_pipe(stderr));
    let started = Instant::now();
    let mut timed_out = false;
    let status: Option<ExitStatus> = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() < timeout => thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                timed_out = true;
                let _ = child.kill();
                break child.wait().ok();
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let stdout = stdout_reader.join().unwrap_or_default();
                let stderr = stderr_reader.join().unwrap_or_default();
                return CommandRun {
                    exit_code: None,
                    stdout,
                    stderr,
                    timed_out: false,
                    spawn_error: Some(format!("wait for test command: {error}")),
                };
            }
        }
    };
    CommandRun {
        exit_code: status.and_then(|value| value.code()),
        stdout: stdout_reader.join().unwrap_or_default(),
        stderr: stderr_reader.join().unwrap_or_default(),
        timed_out,
        spawn_error: None,
    }
}

fn read_pipe<R: Read>(pipe: Option<R>) -> Vec<u8> {
    let mut bytes = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = pipe.read_to_end(&mut bytes);
    }
    bytes
}

pub(crate) struct TemporaryWorktree {
    repository: PathBuf,
    path: PathBuf,
    active: bool,
}

impl TemporaryWorktree {
    pub(crate) fn add(repository: &Path, revision: &str) -> Result<Self, String> {
        let path = unique_worktree_path();
        let output = Command::new("git")
            .arg("-C")
            .arg(repository)
            .args(["worktree", "add", "--detach"])
            .arg(&path)
            .arg(revision)
            .output()
            .map_err(|error| format!("cannot start git worktree add: {error}"))?;
        if !output.status.success() {
            return Err(first_output_line(&output.stderr, "git worktree add failed"));
        }
        Ok(Self {
            repository: repository.to_path_buf(),
            path,
            active: true,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn cleanup(mut self) -> Result<(), String> {
        let result = self.remove();
        self.active = false;
        result
    }

    fn remove(&mut self) -> Result<(), String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.repository)
            .args(["worktree", "remove", "--force"])
            .arg(&self.path)
            .output()
            .map_err(|error| format!("cannot start git worktree remove: {error}"))?;
        let remove_dir_result = if self.path.exists() {
            fs::remove_dir_all(&self.path)
        } else {
            Ok(())
        };
        let _ = Command::new("git")
            .arg("-C")
            .arg(&self.repository)
            .args(["worktree", "prune"])
            .status();
        if !output.status.success() {
            return Err(first_output_line(
                &output.stderr,
                "git worktree remove failed",
            ));
        }
        remove_dir_result.map_err(|error| {
            format!(
                "remove temporary worktree directory {}: {error}",
                self.path.display()
            )
        })
    }
}

impl Drop for TemporaryWorktree {
    fn drop(&mut self) {
        if self.active {
            let _ = self.remove();
        }
    }
}

fn unique_worktree_path() -> PathBuf {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("greppy-verify-{}-{epoch}", std::process::id()))
}

#[derive(Debug, Clone)]
pub(crate) struct Mirror {
    pub(crate) relative: PathBuf,
    pub(crate) source: PathBuf,
}

pub(crate) fn discover_mirrors(root: &Path, relative_cwd: &Path) -> Vec<Mirror> {
    let mut candidates = BTreeSet::new();
    for name in MIRROR_CANDIDATES {
        candidates.insert(PathBuf::from(name));
        if !relative_cwd.as_os_str().is_empty() {
            candidates.insert(relative_cwd.join(name));
        }
    }
    candidates
        .into_iter()
        .filter_map(|relative| {
            let source = root.join(&relative);
            if !source.is_dir() || !is_gitignored(root, &relative) {
                return None;
            }
            Some(Mirror { relative, source })
        })
        .collect()
}

fn is_gitignored(root: &Path, relative: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["check-ignore", "-q", "--"])
        .arg(relative)
        .status()
        .is_ok_and(|status| status.success())
}

pub(crate) fn install_mirrors(worktree: &Path, mirrors: &[Mirror]) -> Result<(), String> {
    for mirror in mirrors {
        let destination = worktree.join(&mirror.relative);
        if destination.exists() || fs::symlink_metadata(&destination).is_ok() {
            continue;
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create mirror parent {}: {error}", parent.display()))?;
        }
        create_dir_symlink(&mirror.source, &destination).map_err(|error| {
            format!(
                "mirror {} at {}: {error}",
                mirror.source.display(),
                destination.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(unix)]
fn create_dir_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, destination)
}

#[cfg(windows)]
fn create_dir_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(source, destination)
}

pub(crate) fn cache_key(revision: &str, argv: &[String], mirrors: &[Mirror]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(revision.as_bytes());
    hasher.update([0]);
    for arg in argv {
        hasher.update(arg.as_bytes());
        hasher.update([0]);
    }
    for mirror in mirrors {
        hasher.update(os_str_bytes(mirror.relative.as_os_str()));
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn read_cache(path: &Path) -> Option<CommandRun> {
    let mut file = fs::File::open(path).ok()?;
    let mut magic = vec![0_u8; CACHE_MAGIC.len()];
    file.read_exact(&mut magic).ok()?;
    if magic != CACHE_MAGIC {
        return None;
    }
    let exit_code = read_i32(&mut file)?;
    let stdout = read_blob(&mut file)?;
    let stderr = read_blob(&mut file)?;
    Some(CommandRun {
        exit_code: if exit_code == i32::MIN {
            None
        } else {
            Some(exit_code)
        },
        stdout,
        stderr,
        timed_out: false,
        spawn_error: None,
    })
}

pub(crate) fn write_cache(path: &Path, run: &CommandRun) -> Result<(), String> {
    if run.timed_out || run.spawn_error.is_some() {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("cache path {} has no parent", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("create verify cache {}: {error}", parent.display()))?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = fs::File::create(&temporary)
        .map_err(|error| format!("create verify cache {}: {error}", temporary.display()))?;
    file.write_all(CACHE_MAGIC)
        .and_then(|_| file.write_all(&run.exit_code.unwrap_or(i32::MIN).to_le_bytes()))
        .and_then(|_| write_blob(&mut file, &run.stdout))
        .and_then(|_| write_blob(&mut file, &run.stderr))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("write verify cache {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("publish verify cache {}: {error}", path.display()))
}

fn read_i32(reader: &mut impl Read) -> Option<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(i32::from_le_bytes(bytes))
}

fn read_blob(reader: &mut impl Read) -> Option<Vec<u8>> {
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length).ok()?;
    let length = u64::from_le_bytes(length);
    let length: usize = length.try_into().ok()?;
    if length > 256 * 1024 * 1024 {
        return None;
    }
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

fn write_blob(writer: &mut impl Write, bytes: &[u8]) -> std::io::Result<()> {
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(bytes)
}

fn first_output_line(output: &[u8], fallback: &str) -> String {
    String::from_utf8_lossy(output)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or(fallback)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "greppy-verify-unit-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn command_timeout_is_reported() {
        #[cfg(unix)]
        let argv = vec!["sh".into(), "-c".into(), "sleep 2".into()];
        #[cfg(windows)]
        let argv = vec!["cmd".into(), "/C".into(), "ping -n 3 127.0.0.1 >NUL".into()];
        let run = run_command(&argv, Path::new("."), Duration::from_millis(20));
        assert!(run.timed_out);
    }

    #[test]
    fn workspace_digest_changes_with_tracked_bytes_not_untracked_bytes() {
        let root = temporary_directory("digest");
        assert!(Command::new("git")
            .arg("init")
            .arg("-q")
            .arg(&root)
            .status()
            .unwrap()
            .success());
        fs::write(root.join("tracked.txt"), "one").unwrap();
        assert!(Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["add", "tracked.txt"])
            .status()
            .unwrap()
            .success());
        let first = workspace_digest(&root).unwrap();
        fs::write(root.join("untracked.txt"), "ignored by digest").unwrap();
        assert_eq!(workspace_digest(&root).unwrap(), first);
        fs::write(root.join("tracked.txt"), "two").unwrap();
        assert_ne!(workspace_digest(&root).unwrap(), first);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cache_round_trip_preserves_process_result() {
        let root = temporary_directory("cache");
        let path = root.join("entry");
        let run = CommandRun {
            exit_code: Some(7),
            stdout: b"out".to_vec(),
            stderr: b"err".to_vec(),
            timed_out: false,
            spawn_error: None,
        };
        write_cache(&path, &run).unwrap();
        let restored = read_cache(&path).unwrap();
        assert_eq!(restored.exit_code, Some(7));
        assert_eq!(restored.stdout, b"out");
        assert_eq!(restored.stderr, b"err");
        let _ = fs::remove_dir_all(root);
    }
}
