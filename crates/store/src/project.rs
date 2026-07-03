//! Project metadata CRUD.

use rusqlite::{params, OptionalExtension};

use crate::store::Store;
use crate::store_error::Result;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub name: String,
    pub indexed_at: String,
    pub root_path: String,
}

impl Store {
    /// Insert or update a project row.
    pub fn upsert_project(&mut self, p: &Project) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "INSERT INTO projects (name, indexed_at, root_path)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(name) DO UPDATE SET
                   indexed_at = excluded.indexed_at,
                   root_path = excluded.root_path",
            params![p.name, p.indexed_at, p.root_path],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch a project by name.
    pub fn get_project(&self, name: &str) -> Result<Option<Project>> {
        let row = self
            .conn()
            .query_row(
                "SELECT name, indexed_at, root_path FROM projects WHERE name = ?1",
                params![name],
                |row| {
                    Ok(Project {
                        name: row.get(0)?,
                        indexed_at: row.get(1)?,
                        root_path: row.get(2)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// List all projects.
    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT name, indexed_at, root_path FROM projects ORDER BY name")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Project {
                    name: row.get(0)?,
                    indexed_at: row.get(1)?,
                    root_path: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Delete a project by name. Cascades to all dependent tables via FK.
    pub fn delete_project(&mut self, name: &str) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw()
            .execute("DELETE FROM projects WHERE name = ?1", params![name])?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_then_get() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "demo".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/tmp/demo".into(),
        })
        .unwrap();
        let p = s.get_project("demo").unwrap().unwrap();
        assert_eq!(p.root_path, "/tmp/demo");
    }

    #[test]
    fn upsert_updates_root_path() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/old".into(),
        })
        .unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-06-28T21:00:00Z".into(),
            root_path: "/new".into(),
        })
        .unwrap();
        assert_eq!(s.get_project("p").unwrap().unwrap().root_path, "/new");
    }

    #[test]
    fn list_and_delete() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "a".into(),
            indexed_at: "x".into(),
            root_path: "/a".into(),
        })
        .unwrap();
        s.upsert_project(&Project {
            name: "b".into(),
            indexed_at: "x".into(),
            root_path: "/b".into(),
        })
        .unwrap();
        assert_eq!(s.list_projects().unwrap().len(), 2);
        s.delete_project("a").unwrap();
        assert_eq!(s.list_projects().unwrap().len(), 1);
    }
}
