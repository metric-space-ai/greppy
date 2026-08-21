//! Reliable filesystem identity associated with a persisted file state.

use std::collections::HashMap;

use rusqlite::params;

use crate::store::Store;
use crate::store_error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileIdentity {
    pub ctime_ns: Option<i64>,
    pub file_id: Option<u64>,
}

impl Store {
    pub fn upsert_file_identity(
        &self,
        project: &str,
        rel_path: &str,
        identity: FileIdentity,
    ) -> Result<()> {
        self.conn().execute(
            "INSERT INTO main.file_identity(project, rel_path, ctime_ns, file_id)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(project, rel_path) DO UPDATE SET
               ctime_ns = excluded.ctime_ns,
               file_id = excluded.file_id",
            params![
                project,
                rel_path,
                identity.ctime_ns,
                identity.file_id.map(|value| value as i64),
            ],
        )?;
        Ok(())
    }

    pub fn list_file_identities(&self, project: &str) -> Result<HashMap<String, FileIdentity>> {
        let mut stmt = self.conn().prepare(
            "SELECT rel_path, ctime_ns, file_id
             FROM file_identity WHERE project = ?1",
        )?;
        let rows = stmt.query_map(params![project], |row| {
            let raw_id: Option<i64> = row.get(2)?;
            Ok((
                row.get::<_, String>(0)?,
                FileIdentity {
                    ctime_ns: row.get(1)?,
                    file_id: raw_id.map(|value| value as u64),
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<HashMap<_, _>>>()?)
    }
}
