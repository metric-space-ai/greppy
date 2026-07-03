//! Per-project language-provider state.
//!
//! This is R3 diagnostic data: the indexer records which language providers
//! contributed to the active index and which edge classes are still missing.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::store::Store;
use crate::store_error::Result;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderState {
    pub project: String,
    pub language: String,
    pub provider_version: String,
    pub status: String,
    pub supported_edge_classes: Vec<String>,
    pub unsupported_edge_classes: Vec<String>,
    pub files_seen: i64,
    pub files_indexed: i64,
    pub files_failed: i64,
    pub diagnostics: Vec<String>,
    pub last_indexed_generation: u64,
    pub updated_at: String,
}

impl ProviderState {
    pub fn is_incomplete(&self) -> bool {
        self.status != "accepted" || !self.unsupported_edge_classes.is_empty()
    }

    /// Whether this provider emits the given edge class for the project.
    ///
    /// This is deliberately narrower than [`is_incomplete`]: a provider can be
    /// "incomplete" only because it omits exotic edge classes (k8s, gitdiff,
    /// semantic, …) while fully supporting the call-graph classes an agent
    /// actually queries. A navigation footer for a *specific* edge class must
    /// hedge on THAT class, not on the provider's overall completeness — else a
    /// fully-supported `who-calls` (CALLS) answer is falsely marked a floor,
    /// which pushes the agent into a redundant `--all` re-query and grep
    /// fallback (H2 spiral). `class` matches the lowercase provider-state
    /// spelling ("calls" / "usages" / "type_refs" / …).
    pub fn supports_edge_class(&self, class: &str) -> bool {
        // A wholly-unsupported provider emits nothing; otherwise the class is
        // supported unless it is explicitly listed as unsupported. The indexer
        // always classifies the call-graph classes (calls/usages/type_refs/…)
        // into exactly one of the two lists, so this is exact for real data.
        self.status != "unsupported" && !self.unsupported_edge_classes.iter().any(|c| c == class)
    }
}

impl Store {
    pub fn upsert_provider_state(&mut self, p: &ProviderState) -> Result<()> {
        let supported = serde_json::to_string(&p.supported_edge_classes)
            .map_err(|e| crate::Error::Store(format!("serialize provider supported edges: {e}")))?;
        let unsupported = serde_json::to_string(&p.unsupported_edge_classes).map_err(|e| {
            crate::Error::Store(format!("serialize provider unsupported edges: {e}"))
        })?;
        let diagnostics = serde_json::to_string(&p.diagnostics)
            .map_err(|e| crate::Error::Store(format!("serialize provider diagnostics: {e}")))?;
        let tx = self.transaction()?;
        tx.raw().execute(
            "INSERT INTO provider_state
                  (project, language, provider_version, status,
                   supported_edge_classes, unsupported_edge_classes,
                   files_seen, files_indexed, files_failed, diagnostics,
                   last_indexed_generation, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                 ON CONFLICT(project, language) DO UPDATE SET
                   provider_version = excluded.provider_version,
                   status = excluded.status,
                   supported_edge_classes = excluded.supported_edge_classes,
                   unsupported_edge_classes = excluded.unsupported_edge_classes,
                   files_seen = excluded.files_seen,
                   files_indexed = excluded.files_indexed,
                   files_failed = excluded.files_failed,
                   diagnostics = excluded.diagnostics,
                   last_indexed_generation = excluded.last_indexed_generation,
                   updated_at = excluded.updated_at",
            params![
                p.project,
                p.language,
                p.provider_version,
                p.status,
                supported,
                unsupported,
                p.files_seen,
                p.files_indexed,
                p.files_failed,
                diagnostics,
                p.last_indexed_generation as i64,
                p.updated_at,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn replace_provider_states(
        &mut self,
        project: &str,
        states: &[ProviderState],
    ) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "DELETE FROM provider_state WHERE project = ?1",
            params![project],
        )?;
        tx.commit()?;
        for state in states {
            self.upsert_provider_state(state)?;
        }
        Ok(())
    }

    pub fn get_provider_state(
        &self,
        project: &str,
        language: &str,
    ) -> Result<Option<ProviderState>> {
        if !self.provider_state_table_exists()? {
            return Ok(None);
        }
        let row = self
            .conn()
            .query_row(
                "SELECT project, language, provider_version, status,
                        supported_edge_classes, unsupported_edge_classes,
                        files_seen, files_indexed, files_failed, diagnostics,
                        last_indexed_generation, updated_at
                 FROM provider_state WHERE project = ?1 AND language = ?2",
                params![project, language],
                row_to_provider_state,
            )
            .optional()?;
        Ok(row)
    }

    pub fn list_provider_states(&self, project: &str) -> Result<Vec<ProviderState>> {
        if !self.provider_state_table_exists()? {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn().prepare(
            "SELECT project, language, provider_version, status,
                    supported_edge_classes, unsupported_edge_classes,
                    files_seen, files_indexed, files_failed, diagnostics,
                    last_indexed_generation, updated_at
             FROM provider_state
             WHERE project = ?1
             ORDER BY language",
        )?;
        let rows = stmt
            .query_map(params![project], row_to_provider_state)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn incomplete_provider_states(&self, project: &str) -> Result<Vec<ProviderState>> {
        Ok(self
            .list_provider_states(project)?
            .into_iter()
            .filter(ProviderState::is_incomplete)
            .collect())
    }

    fn provider_state_table_exists(&self) -> Result<bool> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type='table' AND name='provider_state'",
            [],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }
}

fn row_to_provider_state(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProviderState> {
    let supported_json: String = row.get(4)?;
    let unsupported_json: String = row.get(5)?;
    let diagnostics_json: String = row.get(9)?;
    Ok(ProviderState {
        project: row.get(0)?,
        language: row.get(1)?,
        provider_version: row.get(2)?,
        status: row.get(3)?,
        supported_edge_classes: parse_json_vec(&supported_json),
        unsupported_edge_classes: parse_json_vec(&unsupported_json),
        files_seen: row.get(6)?,
        files_indexed: row.get(7)?,
        files_failed: row.get(8)?,
        diagnostics: parse_json_vec(&diagnostics_json),
        last_indexed_generation: row.get::<_, i64>(10).unwrap_or(0) as u64,
        updated_at: row.get(11)?,
    })
}

fn parse_json_vec(raw: &str) -> Vec<String> {
    serde_json::from_str(raw).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{workspace_state as ws, Project};

    fn store_with_project() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s
    }

    #[test]
    fn provider_state_round_trips_and_marks_incomplete() {
        let mut s = store_with_project();
        let state = ProviderState {
            project: "p".into(),
            language: "rust".into(),
            provider_version: "v1".into(),
            status: "partial".into(),
            supported_edge_classes: vec!["definitions".into(), "calls".into()],
            unsupported_edge_classes: vec!["tests".into()],
            files_seen: 2,
            files_indexed: 1,
            files_failed: 1,
            diagnostics: vec!["provider is partial".into()],
            last_indexed_generation: 7,
            updated_at: ws::now_iso8601(),
        };

        s.upsert_provider_state(&state).unwrap();
        let got = s.get_provider_state("p", "rust").unwrap().unwrap();
        assert_eq!(got, state);
        assert!(got.is_incomplete());
        assert_eq!(s.incomplete_provider_states("p").unwrap(), vec![state]);
    }

    #[test]
    fn replace_provider_states_removes_stale_languages() {
        let mut s = store_with_project();
        let mk = |language: &str| ProviderState {
            project: "p".into(),
            language: language.into(),
            provider_version: "v1".into(),
            status: "accepted".into(),
            supported_edge_classes: Vec::new(),
            unsupported_edge_classes: Vec::new(),
            files_seen: 1,
            files_indexed: 1,
            files_failed: 0,
            diagnostics: Vec::new(),
            last_indexed_generation: 1,
            updated_at: ws::now_iso8601(),
        };
        s.upsert_provider_state(&mk("rust")).unwrap();
        s.upsert_provider_state(&mk("python")).unwrap();

        s.replace_provider_states("p", &[mk("rust")]).unwrap();
        let states = s.list_provider_states("p").unwrap();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].language, "rust");
    }
}
