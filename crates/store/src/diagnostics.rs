//! Store health diagnostics for R3.
//!
//! This is intentionally read-only and deterministic: it surfaces the store
//! schema version, SQLite integrity result, workspace state, graph stats and
//! provider completeness in one object that a CLI or review harness can print.

use serde::{Deserialize, Serialize};

use crate::migrate::CURRENT_VERSION;
use crate::store_error::Result;
use crate::{
    GraphStats, IndexSkip, IndexSkipReasonCount, Project, ProviderState, Store, WorkspaceState,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreDiagnostics {
    pub schema_version: u32,
    pub expected_schema_version: u32,
    pub schema_current: bool,
    pub integrity_ok: bool,
    pub integrity_messages: Vec<String>,
    pub workspace_states: Vec<WorkspaceState>,
    pub projects: Vec<ProjectDiagnostics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectDiagnostics {
    pub project: Project,
    pub stats: GraphStats,
    pub provider_states: Vec<ProviderState>,
    pub incomplete_provider_count: usize,
    pub index_skips: Vec<IndexSkip>,
    pub skip_counts_by_reason: Vec<IndexSkipReasonCount>,
}

impl StoreDiagnostics {
    pub fn is_healthy(&self) -> bool {
        self.schema_current
            && self.integrity_ok
            && self
                .projects
                .iter()
                .all(|p| p.incomplete_provider_count == 0)
    }
}

impl Store {
    /// Build a read-only diagnostic snapshot of the active store.
    ///
    /// Provider incompleteness is deliberately part of health: a database can
    /// be structurally valid while still being unsafe to market as full graph
    /// parity because one or more language providers are partial.
    pub fn diagnostics(&self) -> Result<StoreDiagnostics> {
        let schema_version = self.schema_version()?;
        let integrity_messages = self.integrity_check_messages()?;
        let integrity_ok = matches!(integrity_messages.as_slice(), [single] if single == "ok");
        let mut projects = Vec::new();
        for project in self.list_projects()? {
            let stats = self.stats(&project.name)?;
            let provider_states = self.list_provider_states(&project.name)?;
            let incomplete_provider_count =
                provider_states.iter().filter(|p| p.is_incomplete()).count();
            let index_skips = self.list_index_skips(&project.name)?;
            let skip_counts_by_reason = self.index_skip_counts_by_reason(&project.name)?;
            projects.push(ProjectDiagnostics {
                project,
                stats,
                provider_states,
                incomplete_provider_count,
                index_skips,
                skip_counts_by_reason,
            });
        }
        Ok(StoreDiagnostics {
            schema_version,
            expected_schema_version: CURRENT_VERSION,
            schema_current: schema_version == CURRENT_VERSION,
            integrity_ok,
            integrity_messages,
            workspace_states: self.list_workspace_states()?,
            projects,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{workspace_state as ws, IndexSkip, Project, ProviderState, WorkspaceState};

    #[test]
    fn diagnostics_expose_schema_integrity_workspace_and_provider_incompleteness() {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: "/p".into(),
                git_dir: None,
                git_common_dir: None,
                head_oid: None,
                index_signature: None,
                schema_version: CURRENT_VERSION,
                indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
                graph_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();
        store
            .upsert_provider_state(&ProviderState {
                project: "p".into(),
                language: "rust".into(),
                provider_version: "v1".into(),
                status: "partial".into(),
                supported_edge_classes: vec!["definitions".into()],
                unsupported_edge_classes: vec!["tests".into()],
                files_seen: 1,
                files_indexed: 1,
                files_failed: 0,
                diagnostics: vec!["missing test extraction".into()],
                last_indexed_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();
        store
            .upsert_index_skip(&IndexSkip {
                project: "p".into(),
                rel_path: "generated.rs".into(),
                language: "rust".into(),
                reason: "oversize".into(),
                detail: "size exceeds cap".into(),
                size: 99,
                mtime_ns: 1,
                ctime_ns: Some(2),
                file_id: Some(3),
                last_indexed_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();

        let diag = store.diagnostics().unwrap();
        assert_eq!(diag.schema_version, CURRENT_VERSION);
        assert!(diag.schema_current);
        assert!(diag.integrity_ok);
        assert_eq!(diag.workspace_states.len(), 1);
        assert_eq!(diag.projects.len(), 1);
        assert_eq!(diag.projects[0].incomplete_provider_count, 1);
        assert_eq!(diag.projects[0].index_skips.len(), 1);
        assert_eq!(diag.projects[0].skip_counts_by_reason[0].reason, "oversize");
        assert!(!diag.is_healthy());
    }
}
