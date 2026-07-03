-- 0001_initial.sql
-- Phase 2 initial schema. Mirrors the upstream's
--   src/store/store.c init_schema() DDL block with the addition of
--   the per-file state columns required by the phase plan
--   (`parser_version`, `extractor_version`, `last_indexed_generation`)
--   plus a separate `workspace_state` table that the upstream keeps
--   implicit in `file_hashes`.

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    name       TEXT PRIMARY KEY,
    indexed_at TEXT NOT NULL,
    root_path  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_state (
    project              TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    rel_path             TEXT    NOT NULL,
    sha256               TEXT    NOT NULL,
    mtime_ns             INTEGER NOT NULL DEFAULT 0,
    size                 INTEGER NOT NULL DEFAULT 0,
    parser_version       TEXT    NOT NULL DEFAULT '',
    extractor_version    TEXT    NOT NULL DEFAULT '',
    last_indexed_generation INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project, rel_path)
);

CREATE TABLE IF NOT EXISTS nodes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project        TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    label          TEXT    NOT NULL,
    name           TEXT    NOT NULL,
    qualified_name TEXT    NOT NULL,
    file_path      TEXT    NOT NULL DEFAULT '',
    start_line     INTEGER NOT NULL DEFAULT 0,
    end_line       INTEGER NOT NULL DEFAULT 0,
    properties     TEXT    NOT NULL DEFAULT '{}',
    UNIQUE(project, qualified_name)
);

CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project    TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    source_id  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    edge_type  TEXT    NOT NULL,
    properties TEXT    NOT NULL DEFAULT '{}',
    url_path_gen TEXT GENERATED ALWAYS AS (json_extract(properties, '$.url_path')),
    UNIQUE(source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS project_summaries (
    project     TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_state (
    root_path        TEXT    PRIMARY KEY,
    git_dir          TEXT,
    git_common_dir   TEXT,
    head_oid         TEXT,
    index_signature  TEXT,
    schema_version   INTEGER NOT NULL,
    indexer_version  TEXT    NOT NULL,
    graph_generation INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_label         ON nodes(project, label);
CREATE INDEX IF NOT EXISTS idx_nodes_name          ON nodes(project, name);
CREATE INDEX IF NOT EXISTS idx_nodes_file          ON nodes(project, file_path);
CREATE INDEX IF NOT EXISTS idx_edges_source        ON edges(source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_target        ON edges(target_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_type          ON edges(project, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_target_type   ON edges(project, target_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_source_type   ON edges(project, source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_url_path      ON edges(project, url_path_gen);
CREATE INDEX IF NOT EXISTS idx_file_state_project  ON file_state(project);

-- FTS5 contentless virtual table for BM25 full-text search.
-- Contentless (`content=''`) means FTS5 stores only the inverted index,
-- not a copy of the source text. This is what the upstream does
-- (`create_user_indexes` / `nodes_fts`).
-- We feed it `camel_split(name)` at insert time so that camelCase
-- queries match split tokens; the helper lives in `fts.rs`.
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name,
    qualified_name,
    label,
    file_path,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);
