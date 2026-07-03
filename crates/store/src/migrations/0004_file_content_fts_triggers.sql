-- 0004_file_content_fts_triggers.sql
--
-- RV-009 (orphan-row leak) final fix.
--
-- History: migration 0002 declared `file_content_fts` as a
-- *contentless* FTS5 table (`content = ''`). Contentless tables do not
-- support DELETE, so every re-index left the previously-inserted FTS
-- rows behind — 6 re-indexes of a 1-line file produced 2 content rows
-- but 12 FTS rows. Migration 0003 switched the declaration to
-- *external-content* mode (`content = 'file_content'`) but the
-- application code kept doing `INSERT INTO file_content_fts(rowid)`,
-- which for an external-content table writes a malformed index entry
-- and corrupts the database ("database disk image is malformed").
--
-- This migration makes the mirror correct and self-maintaining:
--
--   * `file_content_fts` stays external-content (it reads `snippet` /
--     `file_path` from the parent `file_content` row at query time).
--   * Three triggers keep the FTS index in lock-step with the content
--     table using the FTS5 'delete' command for removals. With these
--     triggers the application no longer touches the FTS table at all,
--     so re-indexing (delete-for-file + re-insert) can never leave an
--     orphan FTS row.
--
-- The table is dropped and recreated so any orphan/malformed rows left
-- by the 0002/0003 code are discarded, then rebuilt from the current
-- contents of `file_content`.

DROP TRIGGER IF EXISTS file_content_fts_ai;
DROP TRIGGER IF EXISTS file_content_fts_ad;
DROP TRIGGER IF EXISTS file_content_fts_au;
DROP TABLE IF EXISTS file_content_fts;

CREATE VIRTUAL TABLE file_content_fts USING fts5(
    snippet,
    file_path,
    content = 'file_content',
    content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Keep the external-content mirror in sync. These are the canonical
-- FTS5 external-content triggers: inserts add the rowid, deletes use
-- the special 'delete' command, and updates are a delete + insert.
CREATE TRIGGER file_content_fts_ai AFTER INSERT ON file_content BEGIN
    INSERT INTO file_content_fts(rowid, snippet, file_path)
    VALUES (new.id, new.snippet, new.file_path);
END;

CREATE TRIGGER file_content_fts_ad AFTER DELETE ON file_content BEGIN
    INSERT INTO file_content_fts(file_content_fts, rowid, snippet, file_path)
    VALUES ('delete', old.id, old.snippet, old.file_path);
END;

CREATE TRIGGER file_content_fts_au AFTER UPDATE ON file_content BEGIN
    INSERT INTO file_content_fts(file_content_fts, rowid, snippet, file_path)
    VALUES ('delete', old.id, old.snippet, old.file_path);
    INSERT INTO file_content_fts(rowid, snippet, file_path)
    VALUES (new.id, new.snippet, new.file_path);
END;

-- Rebuild the index from whatever rows currently exist so the mirror
-- starts from a clean, orphan-free state.
INSERT INTO file_content_fts(file_content_fts) VALUES ('rebuild');
