use crate::{ChunkId, ChunkStore, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File, Metadata};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EntryKind {
    File,
    Symlink,
    Tombstone,
}

/// One path in the immutable dirty layer. Clean tracked paths are resolved
/// lazily from `base_commit` and therefore never appear here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub path: String,
    pub kind: EntryKind,
    pub mode: u32,
    pub size: u64,
    pub modified_unix_ns: i64,
    pub content_hash: String,
    pub chunks: Vec<ChunkId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    pub repository: PathBuf,
    pub base_commit: String,
    pub baseline_hash: String,
    pub index_hash: String,
    pub index_chunks: Vec<ChunkId>,
    pub entries: Vec<BaselineEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Observation {
    head: String,
    index_path: PathBuf,
    index_hash: String,
    status_hash: String,
    dirty_paths: Vec<String>,
}

/// Captures the committed base plus the visible staged, unstaged and untracked
/// state. Ignored files are deliberately absent. The complete observation and
/// every captured path are verified a second time; one retry is allowed.
pub fn capture_repository(repo: impl AsRef<Path>, store: &ChunkStore) -> Result<BaselineSnapshot> {
    let repository = repository_root(repo.as_ref())?;
    preflight_repository(&repository)?;

    for attempt in 0..2 {
        let first = observe_repository(&repository)?;
        let mut pinned = Vec::new();
        let result = capture_once(&repository, store, &first, &mut pinned);
        match result {
            Ok(snapshot) => return Ok(snapshot),
            Err(Error::ConcurrentRepositoryMutation) if attempt == 0 => {
                for id in pinned {
                    let _ = store.unpin(id);
                }
            }
            Err(error) => {
                for id in pinned {
                    let _ = store.unpin(id);
                }
                return Err(error);
            }
        }
    }
    Err(Error::ConcurrentRepositoryMutation)
}

fn capture_once(
    repository: &Path,
    store: &ChunkStore,
    first: &Observation,
    pinned: &mut Vec<ChunkId>,
) -> Result<BaselineSnapshot> {
    let mut entries = Vec::with_capacity(first.dirty_paths.len());
    for relative in &first.dirty_paths {
        let entry = capture_entry(repository, relative, store)?;
        for id in &entry.chunks {
            store.pin(*id)?;
            pinned.push(*id);
        }
        entries.push(entry);
    }

    let index_bytes = match fs::read(&first.index_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let (index_chunks, _) = store.put_stream(index_bytes.as_slice())?;
    for id in &index_chunks {
        store.pin(*id)?;
        pinned.push(*id);
    }

    let second = observe_repository(repository)?;
    if &second != first {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    for expected in &entries {
        let actual = fingerprint_entry(repository, &expected.path)?;
        if actual.kind != expected.kind
            || actual.mode != expected.mode
            || actual.size != expected.size
            || actual.content_hash != expected.content_hash
        {
            return Err(Error::ConcurrentRepositoryMutation);
        }
    }

    let canonical =
        serde_json::to_vec(&(&first.head, &first.index_hash, &first.status_hash, &entries))?;
    let baseline_hash = blake3::hash(&canonical).to_hex().to_string();
    Ok(BaselineSnapshot {
        repository: repository.to_path_buf(),
        base_commit: first.head.clone(),
        baseline_hash,
        index_hash: first.index_hash.clone(),
        index_chunks,
        entries,
    })
}

fn capture_entry(repository: &Path, relative: &str, store: &ChunkStore) -> Result<BaselineEntry> {
    validate_relative_path(relative)?;
    let path = repository.join(relative);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(BaselineEntry {
                path: relative.to_string(),
                kind: EntryKind::Tombstone,
                mode: 0,
                size: 0,
                modified_unix_ns: 0,
                content_hash: blake3::hash(&[]).to_hex().to_string(),
                chunks: Vec::new(),
            });
        }
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        let target = fs::read_link(&path)?;
        let bytes = path_bytes(target.as_os_str());
        crate::path_policy::validate_symlink_target(relative, &bytes)?;
        let (chunks, size) = store.put_stream(bytes.as_slice())?;
        return Ok(BaselineEntry {
            path: relative.to_string(),
            kind: EntryKind::Symlink,
            mode: 0o120000,
            size,
            modified_unix_ns: modified_unix_ns(&metadata),
            content_hash: blake3::hash(&bytes).to_hex().to_string(),
            chunks,
        });
    }
    if !metadata.is_file() {
        return Err(Error::UnsupportedRepository(format!(
            "dirty path is not a regular file or symbolic link: {relative}"
        )));
    }
    let file = File::open(&path)?;
    let (chunks, size) = store.put_stream(file)?;
    let content_hash = hash_file(&path)?;
    Ok(BaselineEntry {
        path: relative.to_string(),
        kind: EntryKind::File,
        mode: git_mode(&metadata),
        size,
        modified_unix_ns: modified_unix_ns(&metadata),
        content_hash,
        chunks,
    })
}

fn fingerprint_entry(repository: &Path, relative: &str) -> Result<BaselineEntry> {
    validate_relative_path(relative)?;
    let path = repository.join(relative);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(BaselineEntry {
                path: relative.to_string(),
                kind: EntryKind::Tombstone,
                mode: 0,
                size: 0,
                modified_unix_ns: 0,
                content_hash: blake3::hash(&[]).to_hex().to_string(),
                chunks: Vec::new(),
            });
        }
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        let target = fs::read_link(path)?;
        let bytes = path_bytes(target.as_os_str());
        return Ok(BaselineEntry {
            path: relative.to_string(),
            kind: EntryKind::Symlink,
            mode: 0o120000,
            size: bytes.len() as u64,
            modified_unix_ns: modified_unix_ns(&metadata),
            content_hash: blake3::hash(&bytes).to_hex().to_string(),
            chunks: Vec::new(),
        });
    }
    if !metadata.is_file() {
        return Err(Error::UnsupportedRepository(format!(
            "dirty path is not a regular file or symbolic link: {relative}"
        )));
    }
    Ok(BaselineEntry {
        path: relative.to_string(),
        kind: EntryKind::File,
        mode: git_mode(&metadata),
        size: metadata.len(),
        modified_unix_ns: modified_unix_ns(&metadata),
        content_hash: hash_file(&path)?,
        chunks: Vec::new(),
    })
}

fn modified_unix_ns(metadata: &Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .and_then(|duration| i64::try_from(duration.as_nanos()).ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .and_then(|duration| i64::try_from(duration.as_nanos()).ok())
                .unwrap_or(0)
        })
}

fn observe_repository(repository: &Path) -> Result<Observation> {
    let head = git_text(repository, &["rev-parse", "HEAD"])?;
    let index_path = PathBuf::from(git_text(
        repository,
        &["rev-parse", "--path-format=absolute", "--git-path", "index"],
    )?);
    let index = match fs::read(&index_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let status = git_output(
        repository,
        &[
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=dirty",
        ],
    )?;
    let mut paths = BTreeSet::new();
    for path in nul_paths(&git_output(
        repository,
        &["diff", "--name-only", "-z", "HEAD", "--"],
    )?)? {
        paths.insert(path);
    }
    for path in nul_paths(&git_output(
        repository,
        &["ls-files", "--others", "--exclude-standard", "-z", "--"],
    )?)? {
        paths.insert(path);
    }
    Ok(Observation {
        head,
        index_path,
        index_hash: blake3::hash(&index).to_hex().to_string(),
        status_hash: blake3::hash(&status).to_hex().to_string(),
        dirty_paths: paths.into_iter().collect(),
    })
}

fn preflight_repository(repository: &Path) -> Result<()> {
    let git_dir = PathBuf::from(git_text(
        repository,
        &["rev-parse", "--path-format=absolute", "--absolute-git-dir"],
    )?);
    for marker in [
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_LOG",
        "rebase-apply",
        "rebase-merge",
    ] {
        if git_dir.join(marker).exists() {
            return Err(Error::UnsupportedRepository(format!(
                "Git operation in progress ({marker})"
            )));
        }
    }

    let tree = git_output(repository, &["ls-tree", "-r", "-z", "HEAD"])?;
    for record in tree
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
    {
        if record.starts_with(b"160000 ") {
            return Err(Error::UnsupportedRepository(
                "tracked submodules/gitlinks are not supported by portable CoW".into(),
            ));
        }
    }

    let tracked = git_output(repository, &["ls-files", "-z"])?;
    let mut check = Command::new("git")
        .args(["check-attr", "-z", "--stdin", "filter"])
        .current_dir(repository)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    check
        .stdin
        .take()
        .ok_or_else(|| Error::Git {
            command: "git check-attr filter".into(),
            detail: "stdin pipe unavailable".into(),
        })?
        .write_all(&tracked)?;
    let filters = check.wait_with_output()?;
    if !filters.status.success() {
        return Err(git_error("git check-attr filter", &filters));
    }
    let fields: Vec<&[u8]> = filters
        .stdout
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .collect();
    if fields.len() % 3 != 0 {
        return Err(Error::Git {
            command: "git check-attr filter".into(),
            detail: "unexpected -z output shape".into(),
        });
    }
    for triple in fields.chunks_exact(3) {
        let value = triple[2];
        if value != b"unspecified" && value != b"unset" {
            return Err(Error::UnsupportedRepository(format!(
                "Git filter attribute is active for {}; smudge/process filters (including Git LFS) are not supported by portable CoW",
                String::from_utf8_lossy(triple[0])
            )));
        }
    }
    Ok(())
}

fn repository_root(path: &Path) -> Result<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(path)
        .output()?;
    if !output.status.success() {
        return Err(Error::NotGitRepository {
            path: path.to_path_buf(),
            detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|_| Error::NotGitRepository {
            path: path.to_path_buf(),
            detail: "repository root is not valid UTF-8".into(),
        })?
        .trim_end()
        .to_string();
    fs::canonicalize(value).map_err(Into::into)
}

fn git_text(repository: &Path, args: &[&str]) -> Result<String> {
    let output = git_output(repository, args)?;
    String::from_utf8(output)
        .map(|value| value.trim_end().to_string())
        .map_err(|_| Error::Git {
            command: format!("git {}", args.join(" ")),
            detail: "stdout is not valid UTF-8".into(),
        })
}

fn git_output(repository: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repository)
        .output()?;
    if !output.status.success() {
        return Err(git_error(&format!("git {}", args.join(" ")), &output));
    }
    Ok(output.stdout)
}

fn git_error(command: &str, output: &Output) -> Error {
    Error::Git {
        command: command.to_string(),
        detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    }
}

fn nul_paths(bytes: &[u8]) -> Result<Vec<String>> {
    bytes
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            String::from_utf8(path.to_vec()).map_err(|_| {
                Error::UnsupportedRepository(
                    "non-UTF-8 Git paths are not supported by the cross-platform workspace".into(),
                )
            })
        })
        .collect()
}

fn validate_relative_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(Error::InvalidPath(path.display().to_string()));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => return Err(Error::InvalidPath(path.display().to_string())),
        }
    }
    if path.components().next().map(Component::as_os_str) == Some(OsStr::new(".git")) {
        return Err(Error::InvalidPath(
            "repository metadata cannot be captured as workspace content".into(),
        ));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut file, &mut HasherWriter(&mut hasher))?;
    Ok(hasher.finalize().to_hex().to_string())
}

struct HasherWriter<'a>(&'a mut blake3::Hasher);

impl Write for HasherWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(unix)]
fn git_mode(metadata: &Metadata) -> u32 {
    if metadata.permissions().mode() & 0o111 != 0 {
        0o100755
    } else {
        0o100644
    }
}

#[cfg(not(unix))]
fn git_mode(_metadata: &Metadata) -> u32 {
    0o100644
}

#[cfg(unix)]
fn path_bytes(path: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_bytes(path: &OsStr) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git(path: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn fixture() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        git(temp.path(), &["init", "-q"]);
        git(temp.path(), &["config", "user.email", "test@example.test"]);
        git(temp.path(), &["config", "user.name", "Test"]);
        fs::write(temp.path().join("tracked.txt"), "base\n").unwrap();
        fs::write(temp.path().join(".gitignore"), "ignored/\n").unwrap();
        git(temp.path(), &["add", "."]);
        git(temp.path(), &["commit", "-qm", "base"]);
        temp
    }

    #[test]
    fn captures_staged_unstaged_and_untracked_but_not_ignored() {
        let repo = fixture();
        fs::write(repo.path().join("tracked.txt"), "staged\n").unwrap();
        git(repo.path(), &["add", "tracked.txt"]);
        fs::write(repo.path().join("tracked.txt"), "visible\n").unwrap();
        fs::write(repo.path().join("untracked.txt"), "new\n").unwrap();
        fs::create_dir(repo.path().join("ignored")).unwrap();
        fs::write(repo.path().join("ignored/cache"), "cache\n").unwrap();
        let store_root = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(store_root.path()).unwrap();
        let snapshot = capture_repository(repo.path(), &store).unwrap();
        let names: Vec<_> = snapshot
            .entries
            .iter()
            .map(|entry| entry.path.as_str())
            .collect();
        assert_eq!(names, ["tracked.txt", "untracked.txt"]);
        let tracked = &snapshot.entries[0];
        let visible: Vec<u8> = tracked
            .chunks
            .iter()
            .flat_map(|id| store.read(*id).unwrap())
            .collect();
        assert_eq!(visible, b"visible\n");
        assert!(!snapshot.index_chunks.is_empty());
    }

    #[test]
    fn rejects_an_in_progress_merge_before_copying_content() {
        let repo = fixture();
        let git_dir = PathBuf::from(git_text(repo.path(), &["rev-parse", "--git-dir"]).unwrap());
        let git_dir = if git_dir.is_absolute() {
            git_dir
        } else {
            repo.path().join(git_dir)
        };
        fs::write(git_dir.join("MERGE_HEAD"), "deadbeef\n").unwrap();
        let store_root = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(store_root.path()).unwrap();
        let error = capture_repository(repo.path(), &store).unwrap_err();
        assert!(matches!(error, Error::UnsupportedRepository(_)));
        assert_eq!(store.stats().unwrap().chunk_count, 0);
    }

    #[test]
    fn a_deleted_tracked_path_becomes_a_tombstone() {
        let repo = fixture();
        fs::remove_file(repo.path().join("tracked.txt")).unwrap();
        let store_root = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(store_root.path()).unwrap();
        let snapshot = capture_repository(repo.path(), &store).unwrap();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].kind, EntryKind::Tombstone);
    }
}
