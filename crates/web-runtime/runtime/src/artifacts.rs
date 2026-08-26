//! Content-addressed artifact store (guide §15).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactManifest {
    pub contract: String,
    pub digest: DigestFields,
    pub byte_count: u64,
    pub media_type: String,
    pub producing_operation: String,
    pub session_id: String,
    pub run_id: String,
    pub timestamp: String,
    pub redaction_status: String,
    pub sensitive: bool,
    pub credential_labeled: bool,
    pub object_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DigestFields {
    pub algorithm: String,
    pub hex: String,
}

pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    pub fn new(root: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(root.join("objects").join("sha256"))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn put(
        &self,
        bytes: &[u8],
        media_type: &str,
        session_id: &str,
        run_id: &str,
        producing_operation: &str,
        sensitive: bool,
    ) -> io::Result<ArtifactManifest> {
        let hex = hex_sha256(bytes);
        let object_path = format!("objects/sha256/{hex}");
        let dest = self.root.join("objects").join("sha256").join(&hex);
        if !dest.exists() {
            fs::write(&dest, bytes)?;
        }
        let timestamp = rfc3339_now();
        let manifest = ArtifactManifest {
            contract: "greppy.web-runtime.artifact-manifest.v1".to_owned(),
            digest: DigestFields {
                algorithm: "sha256".to_owned(),
                hex: hex.clone(),
            },
            byte_count: bytes.len() as u64,
            media_type: media_type.to_owned(),
            producing_operation: producing_operation.to_owned(),
            session_id: session_id.to_owned(),
            run_id: run_id.to_owned(),
            timestamp,
            redaction_status: if sensitive {
                "redacted_for_model".to_owned()
            } else {
                "not_redacted".to_owned()
            },
            sensitive,
            credential_labeled: sensitive,
            object_path,
        };
        let manifest_path = self
            .root
            .join("sessions")
            .join(session_id)
            .join("artifacts")
            .join(format!("{hex}.json"));
        if let Some(parent) = manifest_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            manifest_path,
            serde_json::to_vec_pretty(&manifest).map_err(io::Error::other)?,
        )?;
        Ok(manifest)
    }

    pub fn list_session(&self, session_id: &str) -> io::Result<Vec<ArtifactManifest>> {
        let dir = self
            .root
            .join("sessions")
            .join(session_id)
            .join("artifacts");
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut manifests = Vec::new();
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)?;
            if let Ok(manifest) = serde_json::from_slice::<ArtifactManifest>(&bytes) {
                manifests.push(manifest);
            }
        }
        manifests.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(manifests)
    }
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn rfc3339_now() -> String {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}Z", elapsed.as_secs(), elapsed.subsec_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_bytes_once_and_lists_manifests() {
        let root = std::env::temp_dir().join(format!("greppy-art-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let store = ArtifactStore::new(root.clone()).unwrap();
        let first = store
            .put(b"hello", "text/plain", "wrs_1", "run", "web.read", false)
            .unwrap();
        let second = store
            .put(b"hello", "text/plain", "wrs_1", "run", "web.read", false)
            .unwrap();
        assert_eq!(first.digest.hex, second.digest.hex);
        assert_eq!(first.byte_count, 5);
        let listed = store.list_session("wrs_1").unwrap();
        assert_eq!(listed.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }
}
