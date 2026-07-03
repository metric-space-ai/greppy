//! Per-file state tracking. This is the `file_hashes` table from the
//! upstream schema extended with the parser_version / extractor_version /
//! last_indexed_generation columns required by the phase plan.

use rusqlite::{params, OptionalExtension};

use crate::store::Store;
use crate::store_error::Result;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq)]
pub struct FileState {
    pub project: String,
    pub rel_path: String,
    pub language: String,
    pub sha256: String,
    pub mtime_ns: i64,
    pub size: i64,
    pub parser_version: String,
    pub extractor_version: String,
    pub last_indexed_generation: u64,
}

impl Store {
    /// Insert or update one file's state. The unique key is
    /// `(project, rel_path)`.
    pub fn upsert_file_state(&mut self, f: &FileState) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "INSERT INTO file_state
                  (project, rel_path, language, sha256, mtime_ns, size,
                   parser_version, extractor_version, last_indexed_generation)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(project, rel_path) DO UPDATE SET
                   language = excluded.language,
                   sha256 = excluded.sha256,
                   mtime_ns = excluded.mtime_ns,
                   size = excluded.size,
                   parser_version = excluded.parser_version,
                   extractor_version = excluded.extractor_version,
                   last_indexed_generation = excluded.last_indexed_generation",
            params![
                f.project,
                f.rel_path,
                f.language,
                f.sha256,
                f.mtime_ns,
                f.size,
                f.parser_version,
                f.extractor_version,
                f.last_indexed_generation as i64,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch a single file's state.
    pub fn get_file_state(&self, project: &str, rel_path: &str) -> Result<Option<FileState>> {
        let row = self
            .conn()
            .query_row(
                "SELECT project, rel_path, language, sha256, mtime_ns, size,
                        parser_version, extractor_version, last_indexed_generation
                 FROM file_state WHERE project = ?1 AND rel_path = ?2",
                params![project, rel_path],
                |row| Ok(row_to_file_state(row)),
            )
            .optional()?;
        Ok(row)
    }

    /// Delete a file's state. Used during incremental updates when a
    /// file is removed from the workspace.
    pub fn delete_file_state(&mut self, project: &str, rel_path: &str) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "DELETE FROM file_state WHERE project = ?1 AND rel_path = ?2",
            params![project, rel_path],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// List all `(project, rel_path)` pairs. Used by freshness checks.
    pub fn list_file_states(&self, project: &str) -> Result<Vec<FileState>> {
        let mut stmt = self.conn().prepare(
            "SELECT project, rel_path, language, sha256, mtime_ns, size,
                    parser_version, extractor_version, last_indexed_generation
             FROM file_state WHERE project = ?1 ORDER BY rel_path",
        )?;
        let rows = stmt
            .query_map(params![project], |row| Ok(row_to_file_state(row)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }
}

fn row_to_file_state(row: &rusqlite::Row<'_>) -> FileState {
    FileState {
        project: row.get(0).unwrap(),
        rel_path: row.get(1).unwrap(),
        language: row.get(2).unwrap(),
        sha256: row.get(3).unwrap(),
        mtime_ns: row.get(4).unwrap(),
        size: row.get(5).unwrap(),
        parser_version: row.get(6).unwrap(),
        extractor_version: row.get(7).unwrap(),
        last_indexed_generation: row.get::<_, i64>(8).unwrap_or(0) as u64,
    }
}

/// Compute the sha256 hex digest of a byte slice. Used both at insert
/// time (so callers do not have to import `sha2` directly) and in the
/// freshness check (so a fast path of `mtime_ns` + `size` can verify
/// before re-hashing).
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;

    fn store_with_project(name: &str) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: name.into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: format!("/repos/{name}"),
        })
        .unwrap();
        s
    }

    fn new_state(project: &str, rel: &str, content: &[u8]) -> FileState {
        FileState {
            project: project.into(),
            rel_path: rel.into(),
            language: "rust".into(),
            sha256: sha256_hex(content),
            mtime_ns: 1_700_000_000_000_000_000,
            size: content.len() as i64,
            parser_version: "tree-sitter-0.21".into(),
            extractor_version: "grepplus-extractor-v1".into(),
            last_indexed_generation: 1,
        }
    }

    #[test]
    fn sha256_hex_is_deterministic_and_lowercase() {
        let a = sha256_hex(b"hello");
        let b = sha256_hex(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn upsert_then_get_round_trip() {
        let mut s = store_with_project("p");
        s.upsert_file_state(&new_state("p", "src/lib.rs", b"fn a() {}"))
            .unwrap();
        let got = s.get_file_state("p", "src/lib.rs").unwrap().unwrap();
        assert_eq!(got.sha256, sha256_hex(b"fn a() {}"));
        assert_eq!(got.size, 9);
    }

    #[test]
    fn upsert_updates_existing_row() {
        let mut s = store_with_project("p");
        s.upsert_file_state(&new_state("p", "src/lib.rs", b"v1"))
            .unwrap();
        // "v2-longer" is 9 bytes: v 2 - l o n g e r
        s.upsert_file_state(&new_state("p", "src/lib.rs", b"v2-longer"))
            .unwrap();
        let got = s.get_file_state("p", "src/lib.rs").unwrap().unwrap();
        assert_eq!(got.sha256, sha256_hex(b"v2-longer"));
        assert_eq!(got.size, 9);
    }

    #[test]
    fn delete_removes_row() {
        let mut s = store_with_project("p");
        s.upsert_file_state(&new_state("p", "src/lib.rs", b"x"))
            .unwrap();
        s.delete_file_state("p", "src/lib.rs").unwrap();
        assert!(s.get_file_state("p", "src/lib.rs").unwrap().is_none());
    }
}
