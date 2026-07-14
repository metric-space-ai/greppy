//! Versioned migration runner.
//!
//! Each migration is a `(version, name, sql)` triple. The runner applies
//! every migration whose version is greater than the current `schema_version`
//! recorded in the `schema_meta` table, in version order.

use rusqlite::Connection;

use crate::Error;

/// One versioned schema change.
///
/// SQL must be idempotent where possible (use `CREATE TABLE IF NOT EXISTS`,
/// `CREATE INDEX IF NOT EXISTS`) so that opening an already-migrated DB is a
/// no-op. For destructive migrations (column add, table rebuild) bump the
/// version and ship a new migration.
#[derive(Debug, Clone)]
pub struct Migration {
    pub version: u32,
    pub name: &'static str,
    pub sql: &'static str,
}

/// All migrations in version order. Append-only.
pub const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "initial_schema",
        sql: include_str!("migrations/0001_initial.sql"),
    },
    Migration {
        version: 2,
        name: "file_content_fts",
        sql: include_str!("migrations/0002_file_content_fts.sql"),
    },
    Migration {
        version: 3,
        name: "file_content_fts_external",
        sql: include_str!("migrations/0003_file_content_fts_external.sql"),
    },
    Migration {
        version: 4,
        name: "file_content_fts_triggers",
        sql: include_str!("migrations/0004_file_content_fts_triggers.sql"),
    },
    Migration {
        version: 5,
        name: "nodes_by_name_index",
        sql: include_str!("migrations/0005_nodes_by_name_index.sql"),
    },
    Migration {
        version: 6,
        name: "nodes_fts_rebuild",
        sql: include_str!("migrations/0006_nodes_fts_rebuild.sql"),
    },
    Migration {
        version: 7,
        name: "raw_edges",
        sql: include_str!("migrations/0007_raw_edges.sql"),
    },
    Migration {
        version: 8,
        name: "provider_state",
        sql: include_str!("migrations/0008_provider_state.sql"),
    },
    Migration {
        version: 9,
        name: "index_skips",
        sql: include_str!("migrations/0009_index_skips.sql"),
    },
    Migration {
        version: 10,
        name: "vector_embeddings",
        sql: include_str!("migrations/0010_vector_embeddings.sql"),
    },
    Migration {
        version: 11,
        name: "vector_embeddings_i8",
        sql: include_str!("migrations/0011_vector_embeddings_i8.sql"),
    },
    Migration {
        version: 12,
        name: "vector_embeddings_chunks",
        sql: include_str!("migrations/0012_vector_embeddings_chunks.sql"),
    },
    Migration {
        version: 13,
        name: "expand_packs",
        sql: include_str!("migrations/0013_expand_packs.sql"),
    },
    Migration {
        version: 14,
        name: "file_identity",
        sql: include_str!("migrations/0014_file_identity.sql"),
    },
    Migration {
        version: 15,
        name: "index_skip_identity",
        sql: include_str!("migrations/0015_index_skip_identity.sql"),
    },
];

/// Current schema version this crate knows about.
pub const CURRENT_VERSION: u32 = 15;

/// Apply pending migrations. Returns the number of migrations applied.
pub fn migrate(conn: &Connection) -> Result<usize, Error> {
    // Ensure the meta table exists so we can read the current version
    // even on a fresh database.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
         );",
    )
    .map_err(|e| Error::Store(format!("create schema_meta: {e}")))?;

    let current: u32 = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut applied = 0usize;
    for m in MIGRATIONS {
        if m.version > current {
            apply_migration_atomic(conn, m)?;
            applied += 1;
        }
    }
    Ok(applied)
}

/// Apply one migration's DDL **and** its `schema_version` bump inside ONE
/// transaction so the pair is all-or-nothing.
///
/// Without this wrapper a crash (or a failing statement) part-way through
/// a migration could leave the DB with the new tables half-created but the
/// recorded version unchanged — or vice versa — yielding a store that can
/// never be opened or re-migrated cleanly. With the transaction the
/// version advances atomically with the schema it describes: either the
/// whole migration lands and the version moves, or nothing changes and the
/// next open retries from the same version.
///
/// SQLite runs migration DDL fine inside an explicit transaction (FTS5
/// virtual-table create/drop included). We BEGIN, apply, bump, and COMMIT;
/// any error rolls the whole unit back before surfacing.
fn apply_migration_atomic(conn: &Connection, m: &Migration) -> Result<(), Error> {
    conn.execute_batch("BEGIN")
        .map_err(|e| Error::Store(format!("begin migration v{}: {e}", m.version)))?;
    let step = (|| -> Result<(), Error> {
        conn.execute_batch(m.sql)
            .map_err(|e| Error::Store(format!("migration v{} ({}): {e}", m.version, m.name)))?;
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rusqlite::params![m.version.to_string()],
        )
        .map_err(|e| Error::Store(format!("record schema_version: {e}")))?;
        Ok(())
    })();
    match step {
        Ok(()) => {
            conn.execute_batch("COMMIT")
                .map_err(|e| Error::Store(format!("commit migration v{}: {e}", m.version)))?;
            Ok(())
        }
        Err(e) => {
            // Roll the half-applied migration back so the version and
            // schema stay consistent; surface the original error.
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{migrate, CURRENT_VERSION};
    use crate::store::Store;
    use rusqlite::Connection;

    #[test]
    fn migrate_fresh_db_applies_initial() {
        let s = Store::open_memory().unwrap();
        assert_eq!(s.schema_version().unwrap(), CURRENT_VERSION);
    }

    #[test]
    fn migrate_idempotent() {
        let s = Store::open_memory().unwrap();
        // Re-open by serialising nothing — we just verify schema_version
        // is stable across migrate_to calls.
        let again = s.schema_version().unwrap();
        assert_eq!(again, CURRENT_VERSION);
    }

    /// Running `migrate()` a second time on an already-migrated connection
    /// must apply zero migrations and leave the version untouched.
    #[test]
    fn migrate_twice_applies_nothing_the_second_time() {
        let conn = Connection::open_in_memory().unwrap();
        let first = migrate(&conn).unwrap();
        assert_eq!(
            first, CURRENT_VERSION as usize,
            "fresh DB applies every migration"
        );
        let second = migrate(&conn).unwrap();
        assert_eq!(second, 0, "re-running migrate must be a no-op");
        let v: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v.parse::<u32>().unwrap(), CURRENT_VERSION);
    }

    /// Simulate an existing DB that was created at an older schema version
    /// (pre-0005, before the by-name index was guaranteed) and verify the
    /// new migration applies cleanly and only the pending migration runs.
    #[test]
    fn migration_applies_cleanly_on_existing_older_db() {
        let conn = Connection::open_in_memory().unwrap();
        // Build the schema up to version 4 only.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )
        .unwrap();
        for m in super::MIGRATIONS.iter().filter(|m| m.version <= 4) {
            conn.execute_batch(m.sql).unwrap();
        }
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', '4')",
            [],
        )
        .unwrap();
        // The by-name index does already exist (created by 0001), so drop it
        // to model a DB that genuinely predates the guarantee.
        conn.execute("DROP INDEX IF EXISTS idx_nodes_name", [])
            .unwrap();

        let applied = migrate(&conn).unwrap();
        // Every migration with version > 4 is pending here (v5 by-name
        // index, v6 nodes_fts rebuild, …); exactly those should apply.
        let expected_pending = super::MIGRATIONS.iter().filter(|m| m.version > 4).count();
        assert_eq!(
            applied, expected_pending,
            "exactly the migrations newer than v4 should apply"
        );

        // The index must now exist again.
        let idx_present: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='index' AND name='idx_nodes_name'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx_present, 1, "0005 must (re)create idx_nodes_name");

        // And the recorded version advances to CURRENT_VERSION.
        let v: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v.parse::<u32>().unwrap(), CURRENT_VERSION);

        // A second migrate is a no-op.
        assert_eq!(migrate(&conn).unwrap(), 0);
    }

    /// The 0007 raw_edges migration must apply cleanly on an existing DB
    /// that was created at v6 (before raw_edges existed): the table is
    /// absent beforehand, present afterward, only the pending migration
    /// runs, and a second migrate is a no-op.
    #[test]
    fn raw_edges_migration_applies_on_existing_v6_db() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )
        .unwrap();
        // Build the schema up to v6 only (everything before raw_edges).
        for m in super::MIGRATIONS.iter().filter(|m| m.version <= 6) {
            conn.execute_batch(m.sql).unwrap();
        }
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', '6')",
            [],
        )
        .unwrap();

        // raw_edges does not exist yet on this older DB.
        let before: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='raw_edges'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(before, 0, "v6 DB must not have raw_edges yet");

        // Exactly the migrations newer than v6 should apply (just 0007 here).
        let applied = migrate(&conn).unwrap();
        let expected_pending = super::MIGRATIONS.iter().filter(|m| m.version > 6).count();
        assert_eq!(applied, expected_pending);

        // raw_edges now exists and is queryable, with its file index.
        let after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='raw_edges'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(after, 1, "0007 must create raw_edges");
        conn.query_row("SELECT COUNT(*) FROM raw_edges", [], |r| r.get::<_, i64>(0))
            .unwrap();
        let idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='index' AND name='idx_raw_edges_file'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, 1, "0007 must create idx_raw_edges_file");

        // Version advanced to CURRENT and a second migrate is a no-op.
        let v: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v.parse::<u32>().unwrap(), CURRENT_VERSION);
        assert_eq!(migrate(&conn).unwrap(), 0);
    }

    /// The 0008 provider-state migration adds the file language stamp and the
    /// per-provider diagnostics table to existing stores.
    #[test]
    fn provider_state_migration_applies_on_existing_v7_db() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )
        .unwrap();
        for m in super::MIGRATIONS.iter().filter(|m| m.version <= 7) {
            conn.execute_batch(m.sql).unwrap();
        }
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', '7')",
            [],
        )
        .unwrap();

        let before_provider: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='provider_state'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(before_provider, 0);
        let before_language: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('file_state') WHERE name='language'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(before_language, 0);

        let applied = migrate(&conn).unwrap();
        let expected_pending = super::MIGRATIONS.iter().filter(|m| m.version > 7).count();
        assert_eq!(applied, expected_pending);

        let after_provider: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='provider_state'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(after_provider, 1);
        let after_language: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('file_state') WHERE name='language'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(after_language, 1);

        let v: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v.parse::<u32>().unwrap(), CURRENT_VERSION);
        assert_eq!(migrate(&conn).unwrap(), 0);
    }

    /// The 0009 index-skips migration adds persistent skipped-file metadata.
    #[test]
    fn index_skips_migration_applies_on_existing_v8_db() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )
        .unwrap();
        for m in super::MIGRATIONS.iter().filter(|m| m.version <= 8) {
            conn.execute_batch(m.sql).unwrap();
        }
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', '8')",
            [],
        )
        .unwrap();

        let before: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='index_skips'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(before, 0);

        let applied = migrate(&conn).unwrap();
        let expected_pending = super::MIGRATIONS.iter().filter(|m| m.version > 8).count();
        assert_eq!(applied, expected_pending);

        let after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='index_skips'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(after, 1);
        conn.query_row("SELECT COUNT(*) FROM index_skips", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap();

        let v: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v.parse::<u32>().unwrap(), CURRENT_VERSION);
        assert_eq!(migrate(&conn).unwrap(), 0);
    }

    #[test]
    fn index_skip_identity_migration_preserves_existing_rows() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )
        .unwrap();
        for migration in super::MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 14)
        {
            conn.execute_batch(migration.sql).unwrap();
        }
        conn.execute_batch(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', '14');
             INSERT INTO projects(name, indexed_at, root_path) VALUES('p', 'x', '/p');
             INSERT INTO index_skips(
                 project, rel_path, language, reason, detail, size, mtime_ns,
                 last_indexed_generation, updated_at
             ) VALUES('p', 'bad.rs', 'rust', 'parse_failed', 'fixture', 4, 5, 6, 'x');",
        )
        .unwrap();

        assert_eq!(migrate(&conn).unwrap(), 1);
        let identity: (Option<i64>, Option<i64>) = conn
            .query_row(
                "SELECT ctime_ns, file_id FROM index_skips
                 WHERE project = 'p' AND rel_path = 'bad.rs'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(identity, (None, None));
        let version: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "15");
    }

    /// a migration that fails part-way must not advance
    /// `schema_version` — the DDL and the version bump are one atomic
    /// transaction, so a failing migration rolls back to the prior version
    /// and leaves no half-created schema behind.
    #[test]
    fn failed_migration_does_not_advance_version_atomically() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO schema_meta(key, value) VALUES('schema_version','0');",
        )
        .unwrap();

        // A migration whose SQL creates one table and THEN fails (duplicate
        // create). We drive it through the REAL production helper
        // `apply_migration_atomic`, so this test guards the actual code
        // path `migrate()` uses — not a copy of it. Without the
        // transaction wrapper the first table + the version bump would
        // persist; with it, the whole unit rolls back.
        let bad = super::Migration {
            version: 1,
            name: "intentionally_failing",
            sql: "CREATE TABLE marker_atomic (x INTEGER);
                  CREATE TABLE marker_atomic (x INTEGER);", // 2nd create fails
        };
        let result = super::apply_migration_atomic(&conn, &bad);
        assert!(
            result.is_err(),
            "the failing migration must surface an error"
        );

        // Version stayed at 0 (the bump was rolled back with the DDL).
        let v: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v, "0", "version must not advance on a failed migration");

        // The first statement's table was rolled back too — no half-applied
        // schema, and the connection is left in autocommit (not stuck in an
        // open transaction), so the next migrate retries cleanly.
        let marker_present: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='marker_atomic'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            marker_present, 0,
            "a failed migration must leave NO partially-created tables"
        );
        assert!(
            conn.is_autocommit(),
            "rollback must release the transaction so the next open retries"
        );

        // A subsequent good migration still applies and advances the
        // version — proving the rollback left the DB usable.
        let good = super::Migration {
            version: 1,
            name: "good",
            sql: "CREATE TABLE marker_ok (x INTEGER);",
        };
        super::apply_migration_atomic(&conn, &good).unwrap();
        let v2: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v2, "1", "a good migration after a failure must advance");
    }

    /// The real runner advances the version atomically: after `migrate`
    /// the version equals CURRENT_VERSION and the FTS-rebuild migration
    /// (0006) left a usable, empty `nodes_fts` table.
    #[test]
    fn runner_applies_all_migrations_atomically_to_current_version() {
        let conn = Connection::open_in_memory().unwrap();
        let applied = migrate(&conn).unwrap();
        assert_eq!(applied, CURRENT_VERSION as usize);
        let v: String = conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key='schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v.parse::<u32>().unwrap(), CURRENT_VERSION);
        // nodes_fts exists and is queryable (0006 rebuilt it).
        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='nodes_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "0006 must (re)create nodes_fts");
    }
}
