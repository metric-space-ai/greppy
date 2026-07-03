-- 0006_nodes_fts_rebuild.sql
--
-- P0 (re-review): `nodes_fts` is a CONTENTLESS FTS5 table
-- (`content = ''`, migration 0001). For a contentless table the FTS5
-- `'delete'` command requires the ORIGINAL token values that were
-- inserted — passing empty strings (which `delete_node` /
-- `delete_nodes_for_file` did) does NOT remove the previously-indexed
-- tokens. Every symbol delete therefore leaked its tokens; after a few
-- rename/edit reindex cycles `nodes_fts` accumulated orphan postings
-- and eventually corrupted ("database disk image is malformed"), so
-- `search-symbols` exited 73 while `PRAGMA integrity_check` (which does
-- NOT descend into FTS5 shadow tables) still reported `ok`.
--
-- This is the SAME class of bug fixed for `file_content_fts` in
-- migrations 0003/0004. We cannot make `nodes_fts` external-content the
-- way `file_content_fts` is, because the indexed columns are NOT the raw
-- `nodes` columns: they are the `camel_split` tokenisations of
-- name/qualified_name (so a query for `processOrder` matches the symbol
-- `ProcessOrder`). External-content mode would force FTS5 to read the
-- raw column values from `nodes`, which would silently change the
-- tokenisation and break camelCase symbol search.
--
-- Fix taken (the task's second, "whichever is correct and leak-free"
-- option): keep `nodes_fts` CONTENTLESS, but
--   (a) drop + recreate the table here so any orphan/corrupt postings
--       left by the empty-string deletes are discarded, and
--   (b) change the application (`insert_node` / `delete_node` /
--       `delete_nodes_for_file` in node.rs) to pass the REAL token
--       values — the exact same `camel_split`-derived strings used at
--       insert time — to the FTS5 `'delete'` command, so every delete
--       removes precisely the postings its insert added.
--
-- The table starts empty; the next index run (per-file delete-then-
-- insert, R-018) repopulates it. There is no SQL `'rebuild'` here
-- because a contentless table has no backing content table to rebuild
-- from — the application owns the token values.

DROP TABLE IF EXISTS nodes_fts;

CREATE VIRTUAL TABLE nodes_fts USING fts5(
    name,
    qualified_name,
    label,
    file_path,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);
