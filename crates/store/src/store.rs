//! Connection wrapper, open modes, and the high-level entry point.

use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::migrate;
use crate::store_error::{Error, Result};

/// Open-mode flags for [`Store::open`].
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Open the database read-only. SQLite will refuse any write attempt.
    pub read_only: bool,
    /// Run `PRAGMA integrity_check` after opening a writable store.
    pub integrity_check: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            read_only: false,
            integrity_check: true,
        }
    }
}

impl OpenOptions {
    pub const fn read_only() -> Self {
        Self {
            read_only: true,
            integrity_check: false,
        }
    }

    /// Writable query hotpath open: apply migrations and allow small writes,
    /// but skip the full DB integrity scan reserved for index writers.
    pub const fn query_writer() -> Self {
        Self {
            read_only: false,
            integrity_check: false,
        }
    }
}

/// Handle to an open graph store.
///
/// A `Store` owns a single `rusqlite::Connection`. It is **not** `Clone` —
/// cloning a connection across threads requires `Send + Sync`, which
/// `rusqlite::Connection` provides only behind a `Mutex`. We deliberately
/// keep the type single-threaded; a `StorePool` handles
/// concurrent reads.
pub struct Store {
    conn: Connection,
    // Shared lifecycle lease prevents GC from renaming/removing the workspace
    // directory for as long as this SQLite handle is alive. In-memory and
    // non-workspace test databases legitimately have no lease.
    _lifecycle: Option<greppy_core::cache::FileLock>,
    overlay: Option<OverlayInfo>,
}

#[derive(Debug, Clone)]
struct OverlayInfo {
    base_path: PathBuf,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").finish_non_exhaustive()
    }
}

impl Store {
    /// Open an in-memory database and run migrations.
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(conn)
    }

    /// Open a database at `path`. Creates the file if it does not exist.
    pub fn open(path: &Path) -> Result<Self> {
        let opts = OpenOptions::default();
        Self::open_with(path, opts)
    }

    /// Open with explicit options.
    pub fn open_with(path: &Path, opts: OpenOptions) -> Result<Self> {
        let lifecycle = workspace_lifecycle_for_path(path).map_err(|e| Error::Io {
            context: format!("acquire lifecycle lease for {}", path.display()),
            source: e,
        })?;
        let conn = if opts.read_only {
            Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(
                |e| Error::Io {
                    context: format!("open read-only {}", path.display()),
                    source: std::io::Error::other(e.to_string()),
                },
            )?
        } else {
            Connection::open(path).map_err(|e| Error::Io {
                context: format!("open {}", path.display()),
                source: std::io::Error::other(e.to_string()),
            })?
        };
        if let Some(parent) = path.parent() {
            greppy_core::cache::touch_last_used_dir(parent);
        }
        Self::from_connection_with_options_and_lease(conn, opts, lifecycle)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        Self::from_connection_with_options(conn, OpenOptions::default())
    }

    fn from_connection_with_options(conn: Connection, opts: OpenOptions) -> Result<Self> {
        Self::from_connection_with_options_and_lease(conn, opts, None)
    }

    fn from_connection_with_options_and_lease(
        conn: Connection,
        opts: OpenOptions,
        lifecycle: Option<greppy_core::cache::FileLock>,
    ) -> Result<Self> {
        // Performance pragmas for the WRITE path (i.e. `greppy index`).
        // Default SQLite is journal_mode=DELETE + synchronous=FULL, which
        // fsyncs on every transaction commit. The indexer commits once per
        // file (batching), so a 423-file repo paid ~423 fsyncs — the
        // dominant cost of cold indexing (measured: ~1.2 s of a 2.65 s
        // python_large index was fsync). WAL + synchronous=NORMAL is the
        // standard crash-safe bulk-write configuration: it fsyncs only at
        // checkpoints, not per commit, and WAL is atomic so a crash can never
        // corrupt the DB (worst case loses the last checkpoint, and the index
        // is a rebuildable cache anyway). temp_store=MEMORY keeps FTS merge
        // scratch off disk. Readers don't set these (they open read-only and
        // tolerate whatever the DB has).
        if !opts.read_only {
            // journal_mode returns a row; use query_row, not execute.
            let _: String = conn
                .query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))
                .unwrap_or_default();
            let _ = conn.execute_batch(
                "PRAGMA synchronous = NORMAL; PRAGMA temp_store = MEMORY; PRAGMA cache_size = -16000;",
            );
        }
        // Apply pending migrations up-front — but only on writers.
        // A read-only open against a DB whose persisted
        // `schema_version` is older than `CURRENT_VERSION` would
        // attempt to CREATE / ALTER tables on a read-only
        // connection and fail (this is what the
        // `freshness-probe` bench was tripping on 2026-06-29).
        // Readers tolerate whatever schema the DB has; the
        // `greppy index` writer upgrades on the next write.
        if !opts.read_only {
            migrate::migrate(&conn)?;
        }
        let s = Self {
            conn,
            _lifecycle: lifecycle,
            overlay: None,
        };
        // Verify integrity on WRITE opens (i.e. `greppy index`) only.
        // `PRAGMA integrity_check` is O(db-size) — hundreds of ms on a large
        // store — so running it on every READ-ONLY open (the query hotpath:
        // who-calls / find-usages / trace / the grep freshness gate) would make
        // every greppy invocation pay that scan. The writer verifies before
        // it mutates; a read-only query against a genuinely corrupt DB still
        // fails loudly at the offending statement (SQLite errors on a malformed
        // image), so it never silently returns wrong data. This keeps the
        // agent-facing query path fast (the token-efficiency benchmark showed
        // per-open integrity_check was the dominant query latency).
        if !opts.read_only && opts.integrity_check {
            s.integrity_check()?;
        }
        if !opts.read_only {
            // Evidence packs are intentionally ephemeral. Prune them on every
            // writable maintenance/open path so an actively used workspace
            // cannot retain expired payloads indefinitely.
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let _ = s
                .conn
                .execute("DELETE FROM expand_packs WHERE expires_at <= ?1", [now]);
        }
        Ok(s)
    }

    /// Returns the current schema version recorded in `schema_meta`.
    pub fn schema_version(&self) -> Result<u32> {
        let v: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .ok();
        Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0))
    }

    /// Run SQLite's built-in `PRAGMA integrity_check`. Returns `Ok(())`
    /// when the database reports `ok`, otherwise returns the diagnostic
    /// text as an error.
    pub fn integrity_check(&self) -> Result<()> {
        let rows = self.integrity_check_messages()?;
        match rows.as_slice() {
            [single] if single == "ok" => Ok(()),
            other => Err(Error::Store(format!("integrity_check reported: {other:?}"))),
        }
    }

    /// Return SQLite's raw `PRAGMA integrity_check` messages.
    ///
    /// Diagnostics use this instead of [`Store::integrity_check`] so they can
    /// report an unhealthy store without hiding the exact SQLite messages
    /// behind an early error.
    pub fn integrity_check_messages(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("PRAGMA integrity_check")?;
        let rows: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Borrow the underlying connection.
    ///
    /// Public so peer crates (search, freshness, …) can issue raw SQL
    /// without us wrapping every query in a typed method. Callers must
    /// treat the returned `&Connection` as read-only-by-convention; the
    /// store's own helpers (`insert_node`, `insert_edge`, …) own the
    /// write paths.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Open a writable private Delta with an immutable Base attached behind
    /// one set of TEMP views. Existing read queries continue to address the
    /// ordinary table names; writes must explicitly target `main.<table>`.
    pub fn open_overlay(
        base_path: &Path,
        delta_path: &Path,
        visibility: &crate::VisibilityIndex,
    ) -> Result<Self> {
        Self::open_overlay_with(
            base_path,
            delta_path,
            visibility,
            OpenOptions::query_writer(),
        )
    }

    /// Open an immutable Base and private Delta for the query hot path.
    /// TEMP tables and views remain writable even though the persisted Delta
    /// connection is read-only, avoiding migrations and write-path PRAGMAs on
    /// every command invocation.
    pub fn open_overlay_read_only(
        base_path: &Path,
        delta_path: &Path,
        visibility: &crate::VisibilityIndex,
    ) -> Result<Self> {
        Self::open_overlay_with(base_path, delta_path, visibility, OpenOptions::read_only())
    }

    fn open_overlay_with(
        base_path: &Path,
        delta_path: &Path,
        visibility: &crate::VisibilityIndex,
        options: OpenOptions,
    ) -> Result<Self> {
        Self::open_with(delta_path, options)?.attach_overlay(base_path, visibility)
    }

    /// Attach an immutable Base to an already-open Delta. Query callers use
    /// this after reading the persisted visibility from the same read-only
    /// connection, avoiding a second SQLite open on every overlay command.
    pub fn attach_overlay(
        mut self,
        base_path: &Path,
        visibility: &crate::VisibilityIndex,
    ) -> Result<Self> {
        let base_uri = sqlite_read_only_uri(base_path)?;
        self.conn
            .execute("ATTACH DATABASE ?1 AS greppy_base", [base_uri])?;
        self.conn.execute_batch(
            "CREATE TEMP TABLE greppy_hidden_paths (
                 path TEXT PRIMARY KEY
             ) WITHOUT ROWID;",
        )?;
        {
            let tx = self.conn.transaction()?;
            {
                let mut insert = tx.prepare("INSERT INTO greppy_hidden_paths(path) VALUES (?1)")?;
                for path in visibility.dirty_paths().chain(visibility.deleted_paths()) {
                    insert.execute([path])?;
                }
            }
            tx.commit()?;
        }
        self.conn.execute_batch(OVERLAY_VIEWS_SQL)?;
        self.overlay = Some(OverlayInfo {
            base_path: base_path.to_path_buf(),
        });
        Ok(self)
    }

    pub fn is_overlay(&self) -> bool {
        self.overlay.is_some()
    }

    pub fn overlay_base_path(&self) -> Option<&Path> {
        self.overlay.as_ref().map(|info| info.base_path.as_path())
    }

    /// Begin a write transaction. Rolls back on drop if neither
    /// `commit()` nor `rollback()` is called explicitly.
    pub fn transaction(&mut self) -> Result<Transaction<'_>> {
        let tx = self.conn.transaction()?;
        Ok(Transaction { tx })
    }
}

fn sqlite_read_only_uri(path: &Path) -> Result<String> {
    let absolute = std::path::absolute(path).map_err(|error| Error::Io {
        context: format!("resolve immutable Base path {}", path.display()),
        source: error,
    })?;
    let raw = absolute.to_str().ok_or_else(|| {
        Error::Store(format!(
            "immutable Base path is not valid UTF-8: {}",
            absolute.display()
        ))
    })?;
    let mut encoded = String::with_capacity(raw.len() + 32);
    for byte in raw.bytes() {
        match byte {
            b'%' => encoded.push_str("%25"),
            b'?' => encoded.push_str("%3F"),
            b'#' => encoded.push_str("%23"),
            b' ' => encoded.push_str("%20"),
            _ => encoded.push(byte as char),
        }
    }
    Ok(format!("file:{encoded}?mode=ro&immutable=1"))
}

/// Base ids are negated while Delta ids remain positive. SQLite AUTOINCREMENT
/// starts at one, so the two namespaces cannot collide.
const OVERLAY_VIEWS_SQL: &str = r#"
CREATE TEMP VIEW projects AS
SELECT * FROM main.projects
UNION ALL
SELECT b.* FROM greppy_base.projects b
WHERE NOT EXISTS (SELECT 1 FROM main.projects d WHERE d.name = b.name);

CREATE TEMP VIEW nodes AS
SELECT * FROM main.nodes
UNION ALL
SELECT -b.id AS id, b.project, b.label, b.name, b.qualified_name,
       b.file_path, b.start_line, b.end_line, b.properties
FROM greppy_base.nodes b
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.file_path
)
AND NOT EXISTS (
    SELECT 1 FROM main.nodes d
    WHERE d.project = b.project AND d.qualified_name = b.qualified_name
);

CREATE TEMP VIEW raw_edges AS
SELECT * FROM main.raw_edges
UNION ALL
SELECT -b.id AS id, b.project, b.file_path, b.source_qname,
       b.target_qname, b.edge_type, b.properties
FROM greppy_base.raw_edges b
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.file_path
);

CREATE TEMP VIEW edges AS
SELECT d.id, d.project, visible_source.id AS source_id,
       visible_target.id AS target_id, d.edge_type, d.properties,
       json_extract(d.properties, '$.url_path') AS url_path_gen
FROM main.overlay_edges d
JOIN nodes visible_source
  ON visible_source.project = d.project
 AND visible_source.qualified_name = d.source_qualified_name
JOIN nodes visible_target
  ON visible_target.project = d.project
 AND visible_target.qualified_name = d.target_qualified_name
UNION ALL
SELECT -e.id AS id, e.project, visible_source.id AS source_id,
       visible_target.id AS target_id, e.edge_type, e.properties,
       e.url_path_gen
FROM greppy_base.edges e
JOIN greppy_base.nodes base_source ON base_source.id = e.source_id
JOIN greppy_base.nodes base_target ON base_target.id = e.target_id
JOIN nodes visible_source
  ON visible_source.project = base_source.project
 AND visible_source.qualified_name = base_source.qualified_name
JOIN nodes visible_target
  ON visible_target.project = base_target.project
 AND visible_target.qualified_name = base_target.qualified_name
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = base_source.file_path
)
AND NOT EXISTS (
    SELECT 1 FROM main.overlay_edges d
    WHERE d.project = e.project
      AND d.source_qualified_name = base_source.qualified_name
      AND d.target_qualified_name = base_target.qualified_name
      AND d.edge_type = e.edge_type
);

CREATE TEMP VIEW file_state AS
SELECT * FROM main.file_state
UNION ALL
SELECT b.* FROM greppy_base.file_state b
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.rel_path
);

CREATE TEMP VIEW file_content AS
SELECT * FROM main.file_content
UNION ALL
SELECT -b.id AS id, b.project, b.rel_path, b.line, b.snippet, b.file_path
FROM greppy_base.file_content b
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.file_path
);

CREATE TEMP VIEW vector_embeddings AS
SELECT * FROM main.vector_embeddings
UNION ALL
SELECT -b.id AS id, b.project, b.model_id, b.prompt_version, b.task,
       CASE WHEN b.node_id IS NULL THEN NULL ELSE -b.node_id END AS node_id,
       b.chunk_idx, b.qualified_name, b.file_path, b.start_line, b.end_line,
       b.content_sha256, b.graph_generation, b.dim, b.vector_norm,
       b.vector, b.created_at, b.vector_i8, b.i8_scale
FROM greppy_base.vector_embeddings b
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.file_path
);

CREATE TEMP VIEW provider_state AS
SELECT * FROM main.provider_state
UNION ALL
SELECT b.* FROM greppy_base.provider_state b
WHERE NOT EXISTS (
    SELECT 1 FROM main.provider_state d
    WHERE d.project = b.project AND d.language = b.language
);

CREATE TEMP VIEW index_skips AS
SELECT * FROM main.index_skips
UNION ALL
SELECT b.* FROM greppy_base.index_skips b
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.rel_path
);

CREATE TEMP VIEW file_identity AS
SELECT * FROM main.file_identity
UNION ALL
SELECT b.* FROM greppy_base.file_identity b
WHERE NOT EXISTS (
    SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.rel_path
);

CREATE TEMP VIEW workspace_state AS
SELECT * FROM main.workspace_state
UNION ALL
SELECT b.* FROM greppy_base.workspace_state b
WHERE NOT EXISTS (
    SELECT 1 FROM main.workspace_state d WHERE d.root_path = b.root_path
);
"#;

fn workspace_lifecycle_for_path(
    path: &Path,
) -> std::io::Result<Option<greppy_core::cache::FileLock>> {
    let Some(parent) = path.parent() else {
        return Ok(None);
    };
    let Ok(manifest) = greppy_core::cache::read_store_manifest(parent) else {
        return Ok(None);
    };
    greppy_core::cache::acquire_workspace_lifecycle(
        &manifest.canonical_root,
        greppy_core::cache::LockMode::Shared,
        false,
    )
    .inspect(|lease| {
        debug_assert!(
            lease.is_some(),
            "blocking lifecycle lock must return a guard"
        );
    })
}

/// A write transaction. Use `Store::transaction()` to acquire.
pub struct Transaction<'a> {
    tx: rusqlite::Transaction<'a>,
}

impl<'a> Transaction<'a> {
    pub fn commit(self) -> Result<()> {
        self.tx.commit().map_err(Error::Sqlite)
    }

    pub fn rollback(self) -> Result<()> {
        self.tx.rollback().map_err(Error::Sqlite)
    }

    /// Borrow the underlying rusqlite transaction. Crate-internal.
    pub(crate) fn raw(&self) -> &rusqlite::Transaction<'a> {
        &self.tx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_memory_creates_db_with_schema() {
        let s = Store::open_memory().unwrap();
        // Schema is at CURRENT_VERSION after migrations run.
        assert_eq!(s.schema_version().unwrap(), crate::migrate::CURRENT_VERSION);
    }

    #[test]
    fn integrity_check_passes_on_fresh_db() {
        let s = Store::open_memory().unwrap();
        s.integrity_check().unwrap();
    }

    #[test]
    fn open_persistent_path_round_trip() {
        let tmp = tempdir_via_env();
        let path = tmp.join("test.db");
        {
            let s = Store::open(&path).unwrap();
            assert_eq!(s.schema_version().unwrap(), crate::migrate::CURRENT_VERSION);
        }
        let s2 = Store::open(&path).unwrap();
        assert_eq!(
            s2.schema_version().unwrap(),
            crate::migrate::CURRENT_VERSION
        );
        s2.integrity_check().unwrap();
    }

    #[test]
    fn read_only_overlay_can_build_temp_views_but_not_mutate_delta() {
        let tmp = tempdir_via_env();
        let base_path = tmp.join("base.db");
        let delta_path = tmp.join("delta.db");
        drop(Store::open(&base_path).unwrap());
        drop(Store::open(&delta_path).unwrap());

        let delta = Store::open_with(&delta_path, OpenOptions::read_only()).unwrap();
        let store = delta
            .attach_overlay(&base_path, &crate::VisibilityIndex::default())
            .unwrap();
        let projects: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
            .unwrap();
        assert_eq!(projects, 0);
        assert!(store
            .conn()
            .execute("DELETE FROM main.projects", [])
            .is_err());
    }

    fn tempdir_via_env() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "greppy-store-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let p = base.join(unique);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
