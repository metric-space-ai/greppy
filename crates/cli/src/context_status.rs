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
    revision: u64,
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
    after_revision: u64,
}
impl Status {
    fn restrict(&mut self, scope: String, generation: u64, capability: Capability) {
        if let Some(r) = self
            .restrictions
            .iter_mut()
            .find(|r| r.scope == scope && r.generation == generation && r.capability == capability)
        {
            if r.announced {
                r.announced = false;
                r.after_revision = self.publication.as_ref().map_or(0, |p| p.revision);
            }
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
            after_revision: self.publication.as_ref().map_or(0, |p| p.revision),
        });
    }
    fn take_notice(
        &mut self,
        scope: &str,
        current: &Fingerprint,
        blocked: [bool; 2],
    ) -> Option<&'static str> {
        let p = self.publication.as_ref()?;
        if &p.fingerprint != current {
            return None;
        }
        let index = self.restrictions.iter().position(|r| {
            !r.announced
                && r.scope == scope
                && r.generation == p.generation
                && p.revision > r.after_revision
                && match r.capability {
                    Capability::Graph => p.graph && !blocked[0],
                    Capability::Semantic => p.semantic && !blocked[1],
                }
        })?;
        let r = &mut self.restrictions[index];
        r.announced = true;
        Some(match r.capability {
            Capability::Graph => {
                "greppy: graph preparation completed; use search-symbol, who-calls or impact."
            }
            Capability::Semantic => "greppy: semantic embedding preparation completed; use search.",
        })
    }
}
fn status_path(root: &Path) -> PathBuf {
    super::background_job_path(root).with_file_name("context-status.json")
}
fn scope() -> Option<String> {
    let value = std::env::var("GREPPY_CONTEXT_SCOPE")
        .or_else(|_| std::env::var("CODEX_THREAD_ID"))
        .ok()?;
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
pub(crate) fn invalidate(root: &Path) {
    if !status_path(root).is_file() {
        return;
    }
    let _ = update(root, |s| {
        if let Some(p) = s.publication.as_mut() {
            p.graph = false;
            p.semantic = false;
        }
    });
}
pub(crate) fn acknowledge(root: &Path, capability: Capability) {
    let Some(scope) = scope() else {
        return;
    };
    acknowledge_scoped(root, &scope, capability);
}
fn acknowledge_scoped(root: &Path, scope: &str, capability: Capability) {
    let _ = update(root, |s| {
        for r in s
            .restrictions
            .iter_mut()
            .filter(|r| r.scope == scope && r.capability == capability)
        {
            r.announced = true;
        }
    });
}
pub(crate) fn published(root: &Path, generation: u64, graph: bool, semantic: bool) {
    let Some(fingerprint) = fingerprint(&super::workspace_locator::store_path(root)) else {
        return;
    };
    let _ = update(root, |s| {
        let revision = s
            .publication
            .as_ref()
            .map_or(1, |p| p.revision.saturating_add(1));
        s.publication = Some(Publication {
            revision,
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
            p.revision = p.revision.saturating_add(1);
            p.fingerprint = fingerprint;
            p.semantic = p.graph && complete;
        }
    });
}
pub(crate) fn notice(root: &Path, scope: &str) -> Option<String> {
    if !status_path(root).is_file() {
        return None;
    }
    let current = fingerprint(&super::workspace_locator::store_path(root))?;
    let job_path = super::background_job_path(root);
    let job = match std::fs::symlink_metadata(&job_path) {
        Ok(_) => Some(serde_json::from_slice::<serde_json::Value>(&bounded_read(&job_path)?).ok()?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => return None,
    };
    let graph_blocked = job
        .as_ref()
        .is_some_and(|j| j.get("kind").and_then(serde_json::Value::as_str) != Some("embedding"));
    let semantic_blocked = graph_blocked
        || job.as_ref().is_some_and(|j| {
            matches!(
                j.get("state").and_then(serde_json::Value::as_str),
                Some("failed" | "cancelled")
            )
        });
    let blocked = [graph_blocked, semantic_blocked];
    update(root, |s| {
        let mut lines = Vec::new();
        while let Some(line) = s.take_notice(scope, &current, blocked) {
            lines.push(line);
        }
        (!lines.is_empty()).then(|| lines.join("\n"))
    })?
}
pub(crate) fn agent_notice(root: &Path, args: &[String], scope: &str) -> Option<String> {
    // Exclude explicit roots rather than acknowledge another workspace's status.
    if args
        .iter()
        .any(|a| a == "--root" || a.starts_with("--root="))
    {
        return None;
    }
    let root = super::workspace_locator::resolve_workspace_root(root);
    let mut argv = vec![std::ffi::OsString::from("greppy")];
    argv.extend(args.iter().map(std::ffi::OsString::from));
    let tail = super::grep_passthrough_args(&argv);
    let verb = tail.first().and_then(|a| a.to_str());
    // This hook is called only for actual successful subprocess completion,
    // never the envelope's special conversion of pending semantic exit-1.
    let used = match verb {
        Some(
            "search-symbol" | "search-symbols" | "who-calls" | "callees" | "impact" | "brief"
            | "path" | "read" | "read-smart",
        ) => Some(Capability::Graph),
        Some("search" | "semantic-search" | "semantic") => Some(Capability::Semantic),
        _ => None,
    };
    if let Some(capability) = used {
        acknowledge_scoped(&root, scope, capability);
        return None;
    }
    if args.iter().any(|a| a == "--json" || a == "--jsonl") {
        return None;
    }
    if !matches!(verb, Some("rg" | "ripgrep" | "grep" | "read-file")) {
        return None;
    }
    notice(&root, scope)
}
pub(crate) fn attach_read_notice(root: Option<&str>) {
    if super::cli_json_output() || std::env::var_os("GREPPY_CONTEXT_ENVELOPE").is_some() {
        return;
    }
    let Some(scope) = scope() else {
        return;
    };
    let Ok(root) = super::resolve_root(root) else {
        return;
    };
    if let Some(line) = notice(&root, &scope) {
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
        let mut s = Status {
            publication: Some(Publication {
                revision: 1,
                generation: 4,
                graph: true,
                semantic: false,
                fingerprint: fp(1),
            }),
            ..Status::default()
        };
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_none());
        s.restrict("a".into(), 4, Capability::Graph);
        s.publication.as_mut().unwrap().revision = 2;
        assert!(s.take_notice("b", &fp(1), [false; 2]).is_none());
        assert!(s
            .take_notice("a", &fp(1), [false; 2])
            .unwrap()
            .contains("graph preparation"));
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_none());
    }
    #[test]
    fn graph_publication_never_claims_semantic_readiness() {
        let mut s = Status::default();
        s.restrict("a".into(), 4, Capability::Semantic);
        s.publication = Some(Publication {
            revision: 1,
            generation: 4,
            graph: true,
            semantic: false,
            fingerprint: fp(1),
        });
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_none());
        s.publication.as_mut().unwrap().semantic = true;
        assert!(s
            .take_notice("a", &fp(1), [false; 2])
            .unwrap()
            .contains("semantic embedding"));
    }
    #[test]
    fn failure_stale_snapshot_and_new_generation_stay_quiet() {
        let mut s = Status::default();
        s.restrict("a".into(), 4, Capability::Graph);
        s.publication = Some(Publication {
            revision: 1,
            generation: 4,
            graph: true,
            semantic: true,
            fingerprint: fp(1),
        });
        assert!(s.take_notice("a", &fp(1), [true; 2]).is_none());
        assert!(s.take_notice("a", &fp(2), [false; 2]).is_none());
        s.publication.as_mut().unwrap().generation = 5;
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_none());
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
            revision: 1,
            generation: 4,
            graph: true,
            semantic: false,
            fingerprint: fp(1),
        });
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_some());
        std::fs::write(&path, serde_json::to_vec(&s).unwrap()).unwrap();
        let mut restored: Status = serde_json::from_slice(&bounded_read(&path).unwrap()).unwrap();
        restored.restrict("a".into(), 4, Capability::Graph);
        assert!(restored.take_notice("a", &fp(1), [false; 2]).is_none());
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
    #[test]
    fn a_session_restriction_does_not_erase_shared_readiness() {
        let mut s = Status {
            publication: Some(Publication {
                revision: 1,
                generation: 4,
                graph: true,
                semantic: true,
                fingerprint: fp(1),
            }),
            ..Status::default()
        };
        s.restrict("a".into(), 4, Capability::Semantic);
        assert!(s.publication.as_ref().unwrap().semantic);
        // A prior ready publication does not satisfy a later restriction.
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_none());
        s.publication.as_mut().unwrap().revision = 2;
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_some());
    }
    #[cfg(unix)]
    #[test]
    fn existing_agent_envelope_attaches_rg_and_read_once_per_session() {
        use greppy_agent::ExecutionEnv;
        use std::os::unix::fs::PermissionsExt;
        let _guard = crate::TEST_ENV_LOCK.lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("GREPPY_STORE_DIR");
        struct Restore(Option<std::ffi::OsString>);
        impl Drop for Restore {
            fn drop(&mut self) {
                match self.0.take() {
                    Some(value) => std::env::set_var("GREPPY_STORE_DIR", value),
                    None => std::env::remove_var("GREPPY_STORE_DIR"),
                }
            }
        }
        let _restore = Restore(previous);
        std::env::set_var("GREPPY_STORE_DIR", temp.path().join("cache"));
        let root = temp.path().join("source");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        let db = super::super::workspace_locator::store_path(&root);
        assert!(db.starts_with(temp.path()));
        std::fs::create_dir_all(db.parent().unwrap()).unwrap();
        std::fs::write(&db, "test graph metadata; never opened as SQLite").unwrap();
        update(&root, |s| {
            s.restrict("session-a".into(), 4, Capability::Graph);
            s.restrict("session-b".into(), 4, Capability::Semantic);
        })
        .unwrap();
        published(&root, 4, true, false);
        std::fs::write(super::super::background_job_path(&root), serde_json::to_vec(&serde_json::json!({"kind":"embedding", "state":"embedding", "pid":std::process::id()})).unwrap()).unwrap();
        let binary = temp.path().join("tool-stub");
        std::fs::write(&binary, "#!/bin/sh\nprintf 'literal:%s:%s\\n' \"$GREPPY_CONTEXT_SCOPE\" \"$GREPPY_CONTEXT_ENVELOPE\"\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut a = greppy_agent::GreppyEnv::with_binary(binary.clone(), root.clone())
            .unwrap()
            .with_context_status("session-a".into(), agent_notice);
        let mut b = greppy_agent::GreppyEnv::with_binary(binary, root.clone())
            .unwrap()
            .with_context_status("session-b".into(), agent_notice);
        let request = serde_json::json!({"args": ["rg", "needle", "file"]});
        let first = a.call_tool("greppy", &request);
        assert!(!first.is_error);
        assert!(first.content.starts_with("literal:session-a:1\n"));
        assert!(first.content.contains("graph preparation completed"));
        assert!(!first
            .content
            .contains("semantic embedding preparation completed"));
        assert_eq!(
            a.call_tool("greppy", &request).content,
            "literal:session-a:1\n"
        );
        assert_eq!(
            b.call_tool("greppy", &request).content,
            "literal:session-b:1\n"
        );
        semantic_published(&root, 4, true);
        let read = serde_json::json!({"args": ["read-file", "file"]});
        assert!(b
            .call_tool("greppy", &read)
            .content
            .contains("semantic embedding preparation completed"));
        assert_eq!(
            b.call_tool("greppy", &read).content,
            "literal:session-b:1\n"
        );
        // Machine/other-root calls do not consume a new session's signal.
        update(&root, |s| {
            s.restrict("session-c".into(), 4, Capability::Semantic)
        })
        .unwrap();
        published(&root, 4, true, true);
        assert!(agent_notice(
            &root,
            &["rg".into(), "--json".into(), "needle".into()],
            "session-c"
        )
        .is_none());
        assert!(agent_notice(
            &root,
            &[
                "--root".into(),
                "other".into(),
                "rg".into(),
                "needle".into()
            ],
            "session-c"
        )
        .is_none());
        std::fs::write(
            super::super::background_job_path(&root),
            serde_json::to_vec(&serde_json::json!({"kind":"embedding", "state":"cancelled"}))
                .unwrap(),
        )
        .unwrap();
        assert!(agent_notice(&root, &["rg".into(), "needle".into()], "session-c").is_none());
        std::fs::write(
            super::super::background_job_path(&root),
            vec![0; MAX_BYTES as usize + 1],
        )
        .unwrap();
        assert!(agent_notice(&root, &["rg".into(), "needle".into()], "session-c").is_none());
        std::fs::remove_file(super::super::background_job_path(&root)).unwrap();
        invalidate(&root);
        assert!(agent_notice(&root, &["rg".into(), "needle".into()], "session-c").is_none());
        published(&root, 4, true, true);
        assert!(agent_notice(&root, &["rg".into(), "needle".into()], "session-c").is_some());
        update(&root, |s| {
            s.restrict("session-d".into(), 4, Capability::Graph);
            s.restrict("session-d".into(), 4, Capability::Semantic);
        })
        .unwrap();
        published(&root, 4, true, true);
        std::fs::write(
            super::super::background_job_path(&root),
            serde_json::to_vec(&serde_json::json!({"kind":"embedding", "state":"failed"})).unwrap(),
        )
        .unwrap();
        a.set_context_scope("session-d");
        let failed_embedding = a.call_tool("greppy", &request);
        assert!(failed_embedding
            .content
            .contains("graph preparation completed"));
        assert!(!failed_embedding
            .content
            .contains("semantic embedding preparation completed"));
        assert_eq!(
            a.call_tool("greppy", &request).content,
            "literal:session-d:1\n"
        );
        std::fs::remove_file(super::super::background_job_path(&root)).unwrap();
        semantic_published(&root, 4, true);
        assert!(a
            .call_tool("greppy", &request)
            .content
            .contains("semantic embedding preparation completed"));

        for (session, capability, query) in [
            ("session-e", Capability::Graph, "who-calls"),
            ("session-f", Capability::Semantic, "search"),
        ] {
            update(&root, |s| s.restrict(session.into(), 4, capability)).unwrap();
            published(&root, 4, true, true);
            a.set_context_scope(session);
            let advanced = a.call_tool("greppy", &serde_json::json!({"args":[query,"target"]}));
            assert_eq!(advanced.content, format!("literal:{session}:1\n"));
            // A successful original capability means a following rg gets no
            // redundant readiness hint, even when the original query was fresh
            // already and never entered a preparation wait loop.
            assert_eq!(
                a.call_tool("greppy", &request).content,
                format!("literal:{session}:1\n")
            );
        }
    }
    #[test]
    fn a_new_preparation_cycle_rearms_after_announcement_only() {
        let mut s = Status::default();
        s.restrict("a".into(), 4, Capability::Semantic);
        s.publication = Some(Publication {
            revision: 1,
            generation: 4,
            graph: true,
            semantic: true,
            fingerprint: fp(1),
        });
        // Repeated pending boundaries do not move the original revision.
        s.restrict("a".into(), 4, Capability::Semantic);
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_some());
        s.restrict("a".into(), 4, Capability::Semantic);
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_none());
        s.publication.as_mut().unwrap().revision = 2;
        s.restrict("a".into(), 4, Capability::Semantic);
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_some());
        assert!(s.take_notice("a", &fp(1), [false; 2]).is_none());
    }
    #[test]
    fn graph_readiness_survives_failed_embedding_preparation() {
        let mut s = Status::default();
        s.restrict("a".into(), 4, Capability::Graph);
        s.restrict("a".into(), 4, Capability::Semantic);
        s.publication = Some(Publication {
            revision: 1,
            generation: 4,
            graph: true,
            semantic: true,
            fingerprint: fp(1),
        });
        assert!(s
            .take_notice("a", &fp(1), [false, true])
            .unwrap()
            .contains("graph preparation completed"));
        assert!(s.take_notice("a", &fp(1), [false, true]).is_none());
        assert!(s
            .take_notice("a", &fp(1), [false; 2])
            .unwrap()
            .contains("semantic embedding preparation completed"));
    }
}
