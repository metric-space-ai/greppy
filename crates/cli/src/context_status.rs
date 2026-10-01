//! Small contextual transition engine. Never opens SQLite or starts preparation.
//! Only an observed preparation restriction may authorize a later readiness hint.
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};
fn bounded_read(path: &Path) -> Option<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > MAX_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    (bytes.len() as u64 <= MAX_BYTES).then_some(bytes)
}

const MAX_BYTES: u64 = 16 * 1024;
const MAX_SCOPES: usize = 32;
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum Capability {
    Graph,
    Semantic,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Fingerprint {
    bytes: u64,
    modified: u128,
}
fn fingerprint(path: &Path) -> Option<Fingerprint> {
    let m = std::fs::symlink_metadata(path).ok()?;
    if !m.is_file() {
        return None;
    }
    Some(Fingerprint {
        bytes: m.len(),
        modified: m
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos(),
    })
}
#[derive(Default, Serialize, Deserialize)]
struct Status {
    publication: Option<Publication>,
    restrictions: Vec<Restriction>,
}
#[derive(Serialize, Deserialize)]
struct Publication {
    generation: u64,
    graph: bool,
    semantic: bool,
    fingerprint: Fingerprint,
}
#[derive(Serialize, Deserialize)]
struct Restriction {
    scope: String,
    generation: u64,
    capability: Capability,
    announced: bool,
}
impl Status {
    fn restrict(&mut self, scope: String, generation: u64, capability: Capability) {
        if self
            .restrictions
            .iter()
            .any(|r| r.scope == scope && r.generation == generation && r.capability == capability)
        {
            return;
        }
        // A new generation invalidates old restrictions for this caller.
        self.restrictions
            .retain(|r| r.scope != scope || r.generation == generation);
        if self.restrictions.len() >= MAX_SCOPES {
            self.restrictions.remove(0);
        }
        self.restrictions.push(Restriction {
            scope,
            generation,
            capability,
            announced: false,
        });
        if capability == Capability::Semantic {
            if let Some(p) = self
                .publication
                .as_mut()
                .filter(|p| p.generation == generation)
            {
                p.semantic = false;
            }
        }
    }
    fn take_notice(
        &mut self,
        scope: &str,
        current: &Fingerprint,
        blocked: bool,
    ) -> Option<&'static str> {
        let p = self.publication.as_ref()?;
        if blocked || &p.fingerprint != current {
            return None;
        }
        let index = self.restrictions.iter().position(|r| {
            !r.announced
                && r.scope == scope
                && r.generation == p.generation
                && match r.capability {
                    Capability::Graph => p.graph,
                    Capability::Semantic => p.semantic,
                }
        })?;
        let r = &mut self.restrictions[index];
        r.announced = true;
        Some(match r.capability {
            Capability::Graph => "greppy: graph preparation completed; search-symbol, who-calls and impact are available again. These commands validate source freshness before answering.",
            Capability::Semantic => "greppy: semantic embedding preparation completed; search is available again. It validates source freshness before answering.",
        })
    }
}
fn status_path(root: &Path) -> PathBuf {
    super::background_job_path(root).with_file_name("context-status.json")
}
fn scope() -> Option<String> {
    let value = std::env::var("CODEX_THREAD_ID").ok()?;
    (value.len() <= 128 && !value.is_empty()).then_some(value)
}
fn update<T>(root: &Path, action: impl FnOnce(&mut Status) -> T) -> Option<T> {
    let path = status_path(root);
    let parent = path.parent()?;
    // Never create a store as a side effect of a literal read.
    if !parent.is_dir() {
        return None;
    }
    let _lock = greppy_core::cache::acquire_named_lock_in(
        parent,
        "context-status",
        greppy_core::cache::LockMode::Exclusive,
        true,
    )
    .ok()??;
    let mut state = if let Ok(m) = std::fs::symlink_metadata(&path) {
        if !m.is_file() || m.len() > MAX_BYTES {
            return None;
        }
        serde_json::from_slice::<Status>(&bounded_read(&path)?).ok()?
    } else {
        Status::default()
    };
    if state.restrictions.len() > MAX_SCOPES {
        return None;
    }
    let previous = serde_json::to_vec(&state).ok()?;
    let result = action(&mut state);
    let bytes = serde_json::to_vec(&state).ok()?;
    if previous == bytes {
        return Some(result);
    }
    if bytes.len() as u64 > MAX_BYTES {
        return None;
    }
    let temp = parent.join(format!(".context-status.{}.tmp", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    use std::io::Write;
    let mut file = options.open(&temp).ok()?;
    let written = file
        .write_all(&bytes)
        .and_then(|_| std::fs::rename(&temp, &path));
    let _ = std::fs::remove_file(&temp);
    written.ok()?;
    Some(result)
}
pub(crate) fn restricted(root: &Path, generation: u64, capability: Capability) {
    let Some(scope) = scope() else {
        return;
    };
    let _ = update(root, |s| s.restrict(scope, generation, capability));
}
pub(crate) fn published(root: &Path, generation: u64, graph: bool, semantic: bool) {
    let Some(fingerprint) = fingerprint(&super::workspace_locator::store_path(root)) else {
        return;
    };
    let _ = update(root, |s| {
        s.publication = Some(Publication {
            generation,
            graph,
            semantic,
            fingerprint,
        })
    });
}
pub(crate) fn semantic_published(root: &Path, generation: u64, complete: bool) {
    let Some(fingerprint) = fingerprint(&super::workspace_locator::store_path(root)) else {
        return;
    };
    let _ = update(root, |s| {
        if let Some(p) = s
            .publication
            .as_mut()
            .filter(|p| p.generation == generation)
        {
            p.fingerprint = fingerprint;
            p.semantic = p.graph && complete;
        }
    });
}
pub(crate) fn attach_read_notice(root: Option<&str>) {
    if super::cli_json_output() {
        return;
    }
    let Some(scope) = scope() else {
        return;
    };
    let Ok(root) = super::resolve_root(root) else {
        return;
    };
    let job = bounded_read(&super::background_job_path(&root))
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let live = job
        .as_ref()
        .and_then(|j| j.get("pid").and_then(serde_json::Value::as_u64))
        .and_then(|pid| u32::try_from(pid).ok())
        .is_some_and(super::process_is_alive);
    if live {
        if let Some(j) = job.as_ref().filter(|j| {
            !matches!(
                j.get("state").and_then(serde_json::Value::as_str),
                Some("failed" | "cancelled")
            )
        }) {
            if j.get("kind").and_then(serde_json::Value::as_str) == Some("index") {
                if let Some(generation) = j
                    .get("target_generation")
                    .and_then(serde_json::Value::as_u64)
                {
                    restricted(&root, generation, Capability::Graph);
                }
            } else if j.get("kind").and_then(serde_json::Value::as_str) == Some("embedding") {
                let _ = update(&root, |s| {
                    if let Some(p) = s.publication.as_ref() {
                        s.restrict(scope.clone(), p.generation, Capability::Semantic);
                    }
                });
            }
        }
    }
    // Missing state is the common path: do not create any status/lock files.
    if !status_path(&root).is_file() {
        return;
    }
    let Some(current) = fingerprint(&super::workspace_locator::store_path(&root)) else {
        return;
    };
    let blocked = job.as_ref().is_some_and(|j| {
        matches!(
            j.get("state").and_then(serde_json::Value::as_str),
            Some("failed" | "cancelled")
        )
    }) || job
        .as_ref()
        .is_some_and(|j| j.get("kind").and_then(serde_json::Value::as_str) != Some("embedding"));
    if let Some(Some(line)) = update(&root, |s| s.take_notice(&scope, &current, blocked)) {
        eprintln!("{line}");
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fp(n: u64) -> Fingerprint {
        Fingerprint {
            bytes: n,
            modified: n as u128,
        }
    }
    #[test]
    fn transition_requires_restriction_and_is_consumed_once() {
        let mut s = Status::default();
        s.publication = Some(Publication {
            generation: 4,
            graph: true,
            semantic: false,
            fingerprint: fp(1),
        });
        assert!(s.take_notice("a", &fp(1), false).is_none());
        s.restrict("a".into(), 4, Capability::Graph);
        assert!(s.take_notice("b", &fp(1), false).is_none());
        assert!(s
            .take_notice("a", &fp(1), false)
            .unwrap()
            .contains("graph preparation"));
        assert!(s.take_notice("a", &fp(1), false).is_none());
    }
    #[test]
    fn graph_publication_never_claims_semantic_readiness() {
        let mut s = Status::default();
        s.restrict("a".into(), 4, Capability::Semantic);
        s.publication = Some(Publication {
            generation: 4,
            graph: true,
            semantic: false,
            fingerprint: fp(1),
        });
        assert!(s.take_notice("a", &fp(1), false).is_none());
        s.publication.as_mut().unwrap().semantic = true;
        assert!(s
            .take_notice("a", &fp(1), false)
            .unwrap()
            .contains("semantic embedding"));
    }
    #[test]
    fn failure_stale_snapshot_and_new_generation_stay_quiet() {
        let mut s = Status::default();
        s.restrict("a".into(), 4, Capability::Graph);
        s.publication = Some(Publication {
            generation: 4,
            graph: true,
            semantic: true,
            fingerprint: fp(1),
        });
        assert!(s.take_notice("a", &fp(1), true).is_none());
        assert!(s.take_notice("a", &fp(2), false).is_none());
        s.publication.as_mut().unwrap().generation = 5;
        assert!(s.take_notice("a", &fp(1), false).is_none());
    }
    #[test]
    fn bounded_metadata_refuses_oversize_and_roundtrips_dedupe() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("metadata");
        std::fs::write(&path, vec![0; MAX_BYTES as usize + 1]).unwrap();
        assert!(bounded_read(&path).is_none());
        let mut s = Status::default();
        s.restrict("a".into(), 4, Capability::Graph);
        s.publication = Some(Publication {
            generation: 4,
            graph: true,
            semantic: false,
            fingerprint: fp(1),
        });
        assert!(s.take_notice("a", &fp(1), false).is_some());
        std::fs::write(&path, serde_json::to_vec(&s).unwrap()).unwrap();
        let mut restored: Status = serde_json::from_slice(&bounded_read(&path).unwrap()).unwrap();
        restored.restrict("a".into(), 4, Capability::Graph);
        assert!(restored.take_notice("a", &fp(1), false).is_none());
    }
    #[test]
    fn restriction_state_is_bounded() {
        let mut s = Status::default();
        for i in 0..100 {
            s.restrict(i.to_string(), 4, Capability::Graph);
        }
        assert_eq!(s.restrictions.len(), MAX_SCOPES);
        s.restrict("99".into(), 5, Capability::Graph);
        assert_eq!(s.restrictions.iter().filter(|r| r.scope == "99").count(), 1);
    }
}
