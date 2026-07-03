-- 0009_index_skips.sql
--
-- R3 diagnostics / large-repo controls: persist every file the indexer
-- deliberately skipped or could not process, with a stable reason and enough
-- stat metadata to audit the decision later.

CREATE TABLE IF NOT EXISTS index_skips (
    project                 TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    rel_path                TEXT    NOT NULL,
    language                TEXT    NOT NULL DEFAULT '',
    reason                  TEXT    NOT NULL,
    detail                  TEXT    NOT NULL DEFAULT '',
    size                    INTEGER NOT NULL DEFAULT 0,
    mtime_ns                INTEGER NOT NULL DEFAULT 0,
    last_indexed_generation INTEGER NOT NULL DEFAULT 0,
    updated_at              TEXT    NOT NULL,
    PRIMARY KEY (project, rel_path)
);

CREATE INDEX IF NOT EXISTS idx_index_skips_project_reason
    ON index_skips(project, reason);
