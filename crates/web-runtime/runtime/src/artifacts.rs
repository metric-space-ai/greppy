//! Content-addressed artifact store (guide §15).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read};
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

impl ArtifactManifest {
    /// True when model-facing payloads must not include raw object bytes.
    pub fn is_restricted(&self) -> bool {
        self.sensitive || self.credential_labeled
    }

    /// Compact model-facing reference. Never includes object bytes, full_text, or html.
    /// Fail-closed: an empty digest is not success.
    pub fn model_facing_ref(&self) -> Result<serde_json::Value, String> {
        if self.digest.hex.is_empty() {
            return Err("artifact digest missing".to_owned());
        }
        Ok(serde_json::json!({
            "digest": self.digest.hex,
            "path": self.object_path,
            "label": self.producing_operation,
            "redaction_status": self.redaction_status,
        }))
    }
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

    pub fn read_object(&self, digest_hex: &str) -> io::Result<Vec<u8>> {
        if digest_hex.len() != 64 || !digest_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "artifact digest is not sha256 hex",
            ));
        }
        fs::read(self.root.join("objects").join("sha256").join(digest_hex))
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

/// Stream a file through SHA-256 without slurping it into memory.
pub fn hex_sha256_file(path: &Path) -> io::Result<String> {
    let file = fs::File::open(path)?;
    let mut reader = io::BufReader::with_capacity(256 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
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
        assert_eq!(store.read_object(&first.digest.hex).unwrap(), b"hello");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_object_rejects_non_hex_digest() {
        let root = std::env::temp_dir().join(format!("greppy-art-bad-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let store = ArtifactStore::new(root.clone()).unwrap();
        assert!(store.read_object("../objects").is_err());
        assert!(store.read_object(&"ab".repeat(32)).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn file_digest_matches_in_memory_digest_for_multi_buffer_payload() {
        let root = std::env::temp_dir().join(format!("greppy-sha-file-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let path = root.join("blob.bin");
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &bytes).unwrap();
        assert_eq!(hex_sha256_file(&path).unwrap(), hex_sha256(&bytes));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sensitive_model_facing_ref_omits_raw_bytes_and_keeps_digest_path_label() {
        let root = std::env::temp_dir().join(format!("greppy-art-sens-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let store = ArtifactStore::new(root.clone()).unwrap();
        let secret = b"SUPER_SECRET_TOKEN_value=hunter2";
        let manifest = store
            .put(
                secret,
                "text/plain",
                "wrs_sens",
                "run",
                "web.read",
                true,
            )
            .unwrap();
        assert!(manifest.is_restricted());
        assert_eq!(manifest.redaction_status, "redacted_for_model");
        let facing = manifest.model_facing_ref().unwrap();
        let dumped = facing.to_string();
        assert!(
            !dumped.contains("SUPER_SECRET_TOKEN"),
            "sensitive bytes leaked into model-facing payload: {dumped}"
        );
        assert!(
            !dumped.contains("hunter2"),
            "sensitive bytes leaked into model-facing payload: {dumped}"
        );
        assert_eq!(facing["digest"], manifest.digest.hex);
        assert_eq!(facing["path"], manifest.object_path);
        assert_eq!(facing["label"], manifest.producing_operation);
        assert_eq!(facing["redaction_status"], "redacted_for_model");
        assert!(facing.get("bytes").is_none());
        assert!(facing.get("full_text").is_none());
        assert!(facing.get("html").is_none());
        assert!(facing.get("text").is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn model_facing_ref_fails_closed_without_digest() {
        let manifest = ArtifactManifest {
            contract: "greppy.web-runtime.artifact-manifest.v1".to_owned(),
            digest: DigestFields {
                algorithm: "sha256".to_owned(),
                hex: String::new(),
            },
            byte_count: 3,
            media_type: "text/plain".to_owned(),
            producing_operation: "web.read".to_owned(),
            session_id: "wrs_1".to_owned(),
            run_id: "run".to_owned(),
            timestamp: "0.000Z".to_owned(),
            redaction_status: "redacted_for_model".to_owned(),
            sensitive: true,
            credential_labeled: true,
            object_path: "objects/sha256/missing".to_owned(),
        };
        let err = manifest.model_facing_ref().unwrap_err();
        assert!(err.contains("digest"), "{err}");
    }
}
