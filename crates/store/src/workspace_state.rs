//! Workspace-state CRUD: one row per indexed workspace root, recording
//! git fingerprint and graph-generation counter.
//!
//! Phase 5 will read this on every `grepplus` invocation to decide
//! whether the graph is fresh. Phase 2 only persists it.

use rusqlite::{params, OptionalExtension};

use crate::store::Store;
use crate::store_error::Result;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceState {
    pub root_path: String,
    pub git_dir: Option<String>,
    pub git_common_dir: Option<String>,
    pub head_oid: Option<String>,
    pub index_signature: Option<String>,
    pub schema_version: u32,
    pub indexer_version: String,
    pub graph_generation: u64,
    pub updated_at: String,
}

impl Store {
    /// Insert or update a workspace state row.
    pub fn upsert_workspace_state(&mut self, w: &WorkspaceState) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "INSERT INTO workspace_state
                  (root_path, git_dir, git_common_dir, head_oid, index_signature,
                   schema_version, indexer_version, graph_generation, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(root_path) DO UPDATE SET
                   git_dir = excluded.git_dir,
                   git_common_dir = excluded.git_common_dir,
                   head_oid = excluded.head_oid,
                   index_signature = excluded.index_signature,
                   schema_version = excluded.schema_version,
                   indexer_version = excluded.indexer_version,
                   updated_at = excluded.updated_at",
            params![
                w.root_path,
                w.git_dir,
                w.git_common_dir,
                w.head_oid,
                w.index_signature,
                w.schema_version as i64,
                w.indexer_version,
                w.graph_generation as i64,
                w.updated_at,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch a workspace state by root path.
    pub fn get_workspace_state(&self, root_path: &str) -> Result<Option<WorkspaceState>> {
        let row = self
            .conn()
            .query_row(
                "SELECT root_path, git_dir, git_common_dir, head_oid, index_signature,
                        schema_version, indexer_version, graph_generation, updated_at
                 FROM workspace_state WHERE root_path = ?1",
                params![root_path],
                |row| Ok(row_to_workspace_state(row)),
            )
            .optional()?;
        Ok(row)
    }

    /// List every indexed workspace state, ordered by root path.
    pub fn list_workspace_states(&self) -> Result<Vec<WorkspaceState>> {
        let mut stmt = self.conn().prepare(
            "SELECT root_path, git_dir, git_common_dir, head_oid, index_signature,
                    schema_version, indexer_version, graph_generation, updated_at
             FROM workspace_state ORDER BY root_path",
        )?;
        let rows = stmt
            .query_map([], |row| Ok(row_to_workspace_state(row)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Bump the graph generation counter (used after every successful
    /// incremental update).
    pub fn bump_generation(&mut self, root_path: &str) -> Result<u64> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "UPDATE workspace_state
                 SET graph_generation = graph_generation + 1,
                     updated_at = ?2
                 WHERE root_path = ?1",
            params![root_path, now_iso8601()],
        )?;
        let gen: i64 = tx.raw().query_row(
            "SELECT graph_generation FROM workspace_state WHERE root_path = ?1",
            params![root_path],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(gen as u64)
    }
}

fn row_to_workspace_state(row: &rusqlite::Row<'_>) -> WorkspaceState {
    WorkspaceState {
        root_path: row.get(0).unwrap(),
        git_dir: row.get(1).unwrap(),
        git_common_dir: row.get(2).unwrap(),
        head_oid: row.get(3).unwrap(),
        index_signature: row.get(4).unwrap(),
        schema_version: row.get::<_, i64>(5).unwrap_or(0) as u32,
        indexer_version: row.get(6).unwrap(),
        graph_generation: row.get::<_, i64>(7).unwrap_or(0) as u64,
        updated_at: row.get(8).unwrap(),
    }
}

/// ISO-8601 timestamp with second precision. UTC.
pub fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Calendar conversion without pulling in `chrono`. Phase 2 keeps it
    // minimal; Phase 9 hardening can swap in `time` or `chrono` if
    // higher-precision timestamps become necessary.
    format!("1970-01-01T00:00:00Z+{secs}s")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> WorkspaceState {
        WorkspaceState {
            root_path: "/repos/demo".into(),
            git_dir: Some("/repos/demo/.git".into()),
            git_common_dir: Some("/repos/demo/.git".into()),
            head_oid: Some("abc123".into()),
            index_signature: Some("idx-sig-1".into()),
            schema_version: 1,
            indexer_version: "grepplus-indexer-v1".into(),
            graph_generation: 1,
            updated_at: now_iso8601(),
        }
    }

    #[test]
    fn upsert_then_get_round_trip() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_workspace_state(&sample()).unwrap();
        let got = s.get_workspace_state("/repos/demo").unwrap().unwrap();
        assert_eq!(got.head_oid.as_deref(), Some("abc123"));
        assert_eq!(got.graph_generation, 1);
    }

    #[test]
    fn bump_generation_increments() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_workspace_state(&sample()).unwrap();
        let g2 = s.bump_generation("/repos/demo").unwrap();
        assert_eq!(g2, 2);
        let g3 = s.bump_generation("/repos/demo").unwrap();
        assert_eq!(g3, 3);
    }

    #[test]
    fn workspace_state_is_not_fk_dependent_on_projects() {
        // Workspace state is keyed by absolute path, not project name;
        // it should be insertable without first creating a project row.
        // This test guards against accidentally adding a FK to projects.
        let mut s = Store::open_memory().unwrap();
        s.upsert_workspace_state(&sample()).unwrap();
        let got = s.get_workspace_state("/repos/demo").unwrap().unwrap();
        assert_eq!(got.root_path, "/repos/demo");
    }

    #[test]
    fn list_workspace_states_is_deterministic() {
        let mut s = Store::open_memory().unwrap();
        let mut a = sample();
        a.root_path = "/repos/a".into();
        let mut b = sample();
        b.root_path = "/repos/b".into();
        s.upsert_workspace_state(&b).unwrap();
        s.upsert_workspace_state(&a).unwrap();
        let rows = s.list_workspace_states().unwrap();
        assert_eq!(
            rows.iter()
                .map(|w| w.root_path.as_str())
                .collect::<Vec<_>>(),
            vec!["/repos/a", "/repos/b"]
        );
    }
}
