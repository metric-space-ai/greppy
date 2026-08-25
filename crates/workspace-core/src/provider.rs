use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const PROVIDER_PROTOCOL_VERSION: u32 = 1;
const MAX_HEARTBEAT_AGE: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdapterKind {
    Fuse3,
    FsKit,
    WinFsp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderState {
    Starting,
    Ready,
    Recovering,
    Broken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub hard_links: bool,
    pub symbolic_links: bool,
    pub byte_range_locks: bool,
    pub memory_maps: bool,
    pub atomic_rename: bool,
    pub case_preserving: bool,
}

impl ProviderCapabilities {
    fn validate(&self) -> Result<()> {
        let missing = [
            ("hard-links", self.hard_links),
            ("symbolic-links", self.symbolic_links),
            ("byte-range-locks", self.byte_range_locks),
            ("memory-maps", self.memory_maps),
            ("atomic-rename", self.atomic_rename),
            ("case-preserving", self.case_preserving),
        ]
        .into_iter()
        .filter_map(|(name, available)| (!available).then_some(name))
        .collect::<Vec<_>>();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::AdapterUnhealthy(format!(
                "adapter is missing required capabilities: {}",
                missing.join(", ")
            )))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderManifest {
    pub protocol_version: u32,
    pub adapter_version: String,
    pub adapter_kind: AdapterKind,
    pub state: ProviderState,
    pub instance_id: String,
    pub data_root: PathBuf,
    pub mount_root: PathBuf,
    pub heartbeat_unix_ms: u64,
    pub capabilities: ProviderCapabilities,
}

/// A verified connection to the one persistent per-user mount. Merely finding
/// a directory is insufficient: the control manifest and the marker served by
/// the mounted adapter must agree byte-for-byte on protocol, instance and root.
#[derive(Debug, Clone)]
pub struct ProviderInstallation {
    data_root: PathBuf,
    manifest: ProviderManifest,
}

impl ProviderInstallation {
    pub fn require_healthy(data_root: impl AsRef<Path>) -> Result<Self> {
        Self::require_healthy_at(data_root, SystemTime::now())
    }

    fn require_healthy_at(data_root: impl AsRef<Path>, now: SystemTime) -> Result<Self> {
        let data_root = absolute_clean(data_root.as_ref())?;
        let control_path = data_root.join("provider.json");
        let control_bytes = fs::read(&control_path).map_err(|error| {
            Error::AdapterUnavailable(format!("cannot read {}: {error}", control_path.display()))
        })?;
        let manifest: ProviderManifest = serde_json::from_slice(&control_bytes)?;
        validate_manifest(&manifest, &data_root, now)?;

        let marker_path = manifest.mount_root.join(".greppy-provider.json");
        let marker_bytes = fs::read(&marker_path).map_err(|error| {
            Error::AdapterUnhealthy(format!(
                "mount marker {} is unavailable: {error}",
                marker_path.display()
            ))
        })?;
        let marker: ProviderManifest = serde_json::from_slice(&marker_bytes)?;
        if marker != manifest {
            return Err(Error::AdapterUnhealthy(
                "control manifest and mounted provider identity differ".into(),
            ));
        }
        manifest.capabilities.validate()?;
        Ok(Self {
            data_root,
            manifest,
        })
    }

    pub fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    pub fn mount_root(&self) -> &Path {
        &self.manifest.mount_root
    }

    pub fn workspace_path(&self, workspace_id: &str) -> Result<PathBuf> {
        validate_id(workspace_id)?;
        Ok(self.mount_root().join("workspaces").join(workspace_id))
    }

    /// Exercise the mounted adapter through ordinary filesystem APIs. This is
    /// intentionally independent of the core database; a stale mount cannot
    /// pass by merely reporting a healthy JSON state.
    pub fn doctor_io(&self, probe_id: &str) -> Result<()> {
        validate_id(probe_id)?;
        let scratch_root = self.mount_root().join("doctor");
        let first = scratch_root.join(format!("{probe_id}.tmp"));
        let second = scratch_root.join(format!("{probe_id}.renamed"));
        let result = (|| -> std::io::Result<()> {
            fs::create_dir_all(&scratch_root)?;
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&first)?;
            file.write_all(b"portable-cow-doctor")?;
            file.seek(SeekFrom::Start(9))?;
            file.write_all(b"COW")?;
            file.sync_all()?;
            file.seek(SeekFrom::Start(0))?;
            let mut observed = Vec::new();
            file.read_to_end(&mut observed)?;
            if observed != b"portable-COW-doctor" {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "partial write was not visible",
                ));
            }
            drop(file);
            fs::rename(&first, &second)?;
            if first.exists() || !second.is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "atomic rename contract was not preserved",
                ));
            }
            fs::remove_file(&second)?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_file(&first);
            let _ = fs::remove_file(&second);
            return Err(Error::AdapterUnhealthy(format!(
                "mounted read/write/rename/delete smoke test failed: {error}"
            )));
        }
        Ok(())
    }
}

fn validate_manifest(
    manifest: &ProviderManifest,
    expected_data_root: &Path,
    now: SystemTime,
) -> Result<()> {
    if manifest.protocol_version != PROVIDER_PROTOCOL_VERSION {
        return Err(Error::AdapterUnhealthy(format!(
            "provider protocol {} is incompatible with required protocol {}",
            manifest.protocol_version, PROVIDER_PROTOCOL_VERSION
        )));
    }
    if manifest.state != ProviderState::Ready {
        return Err(Error::AdapterUnhealthy(format!(
            "provider state is {:?}",
            manifest.state
        )));
    }
    if manifest.data_root != expected_data_root {
        return Err(Error::AdapterUnhealthy(format!(
            "provider data root {} does not match {}",
            manifest.data_root.display(),
            expected_data_root.display()
        )));
    }
    if !manifest.mount_root.is_absolute() || manifest.mount_root == manifest.data_root {
        return Err(Error::AdapterUnhealthy(
            "provider mount root is invalid or aliases its private data root".into(),
        ));
    }
    validate_id(&manifest.instance_id)?;
    if manifest.adapter_version.trim().is_empty() {
        return Err(Error::AdapterUnhealthy(
            "provider did not report an adapter version".into(),
        ));
    }
    let heartbeat = UNIX_EPOCH
        .checked_add(Duration::from_millis(manifest.heartbeat_unix_ms))
        .ok_or_else(|| Error::AdapterUnhealthy("provider heartbeat overflowed".into()))?;
    let age = now
        .duration_since(heartbeat)
        .map_err(|_| Error::AdapterUnhealthy("provider heartbeat is in the future".into()))?;
    if age > MAX_HEARTBEAT_AGE {
        return Err(Error::AdapterUnhealthy(format!(
            "provider heartbeat is stale by {} seconds",
            age.as_secs()
        )));
    }
    Ok(())
}

fn absolute_clean(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err(Error::AdapterUnavailable(format!(
            "provider data root must be absolute: {}",
            path.display()
        )));
    }
    Ok(path.to_path_buf())
}

fn validate_id(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(Error::InvalidPath(format!(
            "identifier contains unsupported characters: {value:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(data: &Path, mount: &Path, now: SystemTime) -> ProviderManifest {
        ProviderManifest {
            protocol_version: PROVIDER_PROTOCOL_VERSION,
            adapter_version: "0.3.4-test".into(),
            adapter_kind: AdapterKind::Fuse3,
            state: ProviderState::Ready,
            instance_id: "test-instance".into(),
            data_root: data.to_path_buf(),
            mount_root: mount.to_path_buf(),
            heartbeat_unix_ms: now.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            capabilities: ProviderCapabilities {
                hard_links: true,
                symbolic_links: true,
                byte_range_locks: true,
                memory_maps: true,
                atomic_rename: true,
                case_preserving: true,
            },
        }
    }

    fn publish(manifest: &ProviderManifest) {
        fs::create_dir_all(&manifest.data_root).unwrap();
        fs::create_dir_all(&manifest.mount_root).unwrap();
        let bytes = serde_json::to_vec(manifest).unwrap();
        fs::write(manifest.data_root.join("provider.json"), &bytes).unwrap();
        fs::write(manifest.mount_root.join(".greppy-provider.json"), bytes).unwrap();
    }

    #[test]
    fn requires_matching_live_control_and_mount_identity() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        let mount = temp.path().join("mount");
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let value = manifest(&data, &mount, now);
        publish(&value);
        let installation = ProviderInstallation::require_healthy_at(&data, now).unwrap();
        assert_eq!(installation.manifest(), &value);

        let mut other = value;
        other.instance_id = "different-instance".into();
        fs::write(
            mount.join(".greppy-provider.json"),
            serde_json::to_vec(&other).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            ProviderInstallation::require_healthy_at(&data, now),
            Err(Error::AdapterUnhealthy(_))
        ));
    }

    #[test]
    fn rejects_a_stale_heartbeat() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        let mount = temp.path().join("mount");
        let heartbeat = UNIX_EPOCH + Duration::from_secs(1_000);
        publish(&manifest(&data, &mount, heartbeat));
        let now = heartbeat + MAX_HEARTBEAT_AGE + Duration::from_millis(1);
        assert!(matches!(
            ProviderInstallation::require_healthy_at(&data, now),
            Err(Error::AdapterUnhealthy(_))
        ));
    }

    #[test]
    fn doctor_exercises_partial_write_rename_and_delete() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        let mount = temp.path().join("mount");
        let now = SystemTime::now();
        publish(&manifest(&data, &mount, now));
        let installation = ProviderInstallation::require_healthy_at(&data, now).unwrap();
        installation.doctor_io("smoke").unwrap();
        assert!(!mount.join("doctor/smoke.tmp").exists());
        assert!(!mount.join("doctor/smoke.renamed").exists());
    }
}
