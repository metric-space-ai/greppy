-- 0008_provider_state.sql
--
-- R3 diagnostics: persist the detected language per file and the current
-- provider completeness state per project/language. This lets diagnostics
-- expose "indexed, but only partial provider support" without inferring it
-- from marketing text or parser internals.

ALTER TABLE file_state ADD COLUMN language TEXT NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS provider_state (
    project                  TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    language                 TEXT    NOT NULL,
    provider_version         TEXT    NOT NULL,
    status                   TEXT    NOT NULL,
    supported_edge_classes   TEXT    NOT NULL DEFAULT '[]',
    unsupported_edge_classes TEXT    NOT NULL DEFAULT '[]',
    files_seen               INTEGER NOT NULL DEFAULT 0,
    files_indexed            INTEGER NOT NULL DEFAULT 0,
    files_failed             INTEGER NOT NULL DEFAULT 0,
    diagnostics              TEXT    NOT NULL DEFAULT '[]',
    last_indexed_generation  INTEGER NOT NULL DEFAULT 0,
    updated_at               TEXT    NOT NULL,
    PRIMARY KEY (project, language)
);

CREATE INDEX IF NOT EXISTS idx_provider_state_project_status
    ON provider_state(project, status);
