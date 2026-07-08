-- 0013: short-lived evidence packs for agent-facing expand.

CREATE TABLE IF NOT EXISTS expand_packs (
    id               TEXT    PRIMARY KEY,
    project          TEXT    NOT NULL,
    command          TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    graph_generation INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL,
    expires_at       INTEGER NOT NULL,
    summary_json     TEXT    NOT NULL,
    payload_text     TEXT    NOT NULL,
    payload_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_expand_packs_expiry
    ON expand_packs(expires_at);

CREATE INDEX IF NOT EXISTS idx_expand_packs_project
    ON expand_packs(project, graph_generation);
