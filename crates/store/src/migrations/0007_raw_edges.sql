-- 0007_raw_edges.sql
-- Store-owned, normalised raw-edge table.
--
-- The indexer currently keeps an ad-hoc `indexer_raw_edges` sidecar created
-- via raw `conn()` DDL (a `(project, file_path, edge_json)` blob). This
-- migration introduces a *store-owned* `raw_edges` table with a normalised
-- shape — one column per logical field — so a future wave can switch the
-- indexer from ad-hoc DDL to the typed store API
-- (`insert_raw_edges` / `list_raw_edges` / `delete_raw_edges_for_file`).
--
-- This migration does NOT touch the indexer or its existing
-- `indexer_raw_edges` table; the two coexist until the indexer is migrated.
--
-- Columns:
--   project        the owning project (FK to projects, cascade on delete)
--   file_path      the file whose extraction produced the edge; rows are
--                  keyed by it so a re-extract replaces exactly that file's
--                  contribution (delete-then-insert per file)
--   source_qname   unresolved source qualified-name
--   target_qname   unresolved target qualified-name / identifier
--   edge_type      e.g. CALLS, IMPORTS, TYPE_REF
--   properties     JSON blob of extra per-edge attributes (default '{}')
--
-- Idempotent: CREATE TABLE / INDEX IF NOT EXISTS so re-applying on an
-- already-migrated DB is a no-op.

CREATE TABLE IF NOT EXISTS raw_edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project      TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    file_path    TEXT    NOT NULL DEFAULT '',
    source_qname TEXT    NOT NULL,
    target_qname TEXT    NOT NULL,
    edge_type    TEXT    NOT NULL,
    properties   TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_raw_edges_project ON raw_edges(project);
CREATE INDEX IF NOT EXISTS idx_raw_edges_file    ON raw_edges(project, file_path);
