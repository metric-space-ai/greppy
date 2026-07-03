-- 0002_file_content_fts.sql
--
-- Adds a backing table + contentless FTS5 index for real, grep-like,
-- indexed code search. Each `file_content` row is one indexed line
-- of one file; the surrogate `id` column is the FTS5 rowid and is
-- globally unique (so different files at the same line number do not
-- collide on the FTS rowid space).
--
-- R-011 / WP-R011: replaces the symbol-metadata-only
-- `grepplus_search::search_code` with one that finds arbitrary
-- literal text in file contents. The previous `nodes_fts` (driven
-- from `ExtractedNode` metadata) is kept as the backing store for
-- `search-symbols`, which is a renamed alias of the old behaviour.

CREATE TABLE IF NOT EXISTS file_content (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    rel_path    TEXT    NOT NULL,
    line        INTEGER NOT NULL,
    snippet     TEXT    NOT NULL,
    UNIQUE(project, rel_path, line)
);

CREATE INDEX IF NOT EXISTS idx_file_content_path ON file_content(project, rel_path);

-- Contentless FTS5 mirror — we feed it explicitly from the indexer
-- because `snippet` is multi-line aware (a ContentRow carries
-- one line plus the file path) and we want consistent `file_path`
-- tokens alongside the snippet tokens so file-restricted queries
-- work without joining on every hit.
CREATE VIRTUAL TABLE IF NOT EXISTS file_content_fts USING fts5(
    snippet,
    file_path,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);
