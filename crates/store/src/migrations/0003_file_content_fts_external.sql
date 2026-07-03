-- 0003_file_content_fts_external.sql
--
-- RV-009 / WP-R011 follow-up: switch the `file_content_fts` mirror
-- from contentless mode (`content = ''`) to external-content mode
-- (`content = 'file_content'`). The contentless declaration does not
-- allow DELETE from the FTS table, which means the mirror accumulated
-- orphan rows on every re-index (verified: 6 re-indexes produced 6
-- file_content rows + 12 file_content_fts rows).
--
-- External-content mode means the FTS table reads its indexed columns
-- (`snippet`, `file_path`) from the parent `file_content` table via
-- the parent's `id` column (declared as `content_rowid`). When
-- `file_content` rows are deleted, the corresponding FTS rows go with
-- them automatically — no manual prune needed.
--
-- `file_path` is the renamed equivalent of `rel_path`. To avoid
-- breaking the indexer's existing `rel_path` usage everywhere, we
-- add `file_path` as a new column and sync it via an AFTER INSERT
-- trigger (or just by writing it in the same INSERT in
-- `Store::insert_file_content_rows`).

ALTER TABLE file_content ADD COLUMN file_path TEXT;

-- Backfill: every existing row gets its file_path = rel_path. New
-- rows will be written with both columns set in the application.
UPDATE file_content SET file_path = rel_path WHERE file_path IS NULL;

DROP TABLE IF EXISTS file_content_fts;
CREATE VIRTUAL TABLE file_content_fts USING fts5(
    snippet,
    file_path,
    content = 'file_content',
    content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Index on file_path for filter queries (e.g. restrict FTS hits to
-- a single file_path). The FTS5 MATCH itself stays as the primary
-- filter; this index is for the secondary `file_path = ?` filter we
-- apply when narrowing a search to one file.
CREATE INDEX IF NOT EXISTS idx_file_content_path_v2
    ON file_content(project, file_path);