-- 0016_overlay_edges.sql
--
-- A Store-CoW Delta cannot persist SQLite node ids for edges whose target is
-- supplied by the immutable Base: Base ids live in another database. Persist
-- logical endpoint identities instead and resolve them through the composed
-- `nodes` view when the overlay is opened.

CREATE TABLE IF NOT EXISTS overlay_edges (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    project               TEXT NOT NULL,
    source_qualified_name TEXT NOT NULL,
    target_qualified_name TEXT NOT NULL,
    edge_type             TEXT NOT NULL,
    properties            TEXT NOT NULL DEFAULT '{}',
    UNIQUE(project, source_qualified_name, target_qualified_name, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_overlay_edges_source
    ON overlay_edges(project, source_qualified_name, edge_type);
CREATE INDEX IF NOT EXISTS idx_overlay_edges_target
    ON overlay_edges(project, target_qualified_name, edge_type);
