-- 0010_vector_embeddings.sql
--
-- R5 vector-search substrate: persist real embedding vectors for graph
-- nodes/code spans. The key includes the model, prompt/task contract, content
-- hash and graph generation so query paths can reject stale embeddings instead
-- of silently ranking them.

CREATE TABLE IF NOT EXISTS vector_embeddings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project          TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    model_id         TEXT    NOT NULL,
    prompt_version   TEXT    NOT NULL,
    task             TEXT    NOT NULL,
    node_id          INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
    qualified_name   TEXT    NOT NULL,
    file_path        TEXT    NOT NULL,
    start_line       INTEGER NOT NULL DEFAULT 0,
    end_line         INTEGER NOT NULL DEFAULT 0,
    content_sha256   TEXT    NOT NULL,
    graph_generation INTEGER NOT NULL DEFAULT 0,
    dim              INTEGER NOT NULL,
    vector_norm      REAL    NOT NULL,
    vector           BLOB    NOT NULL,
    created_at       TEXT    NOT NULL,
    UNIQUE(project, model_id, prompt_version, task, qualified_name, content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_scope
    ON vector_embeddings(project, model_id, prompt_version, task, graph_generation);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_file
    ON vector_embeddings(project, file_path);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_node
    ON vector_embeddings(node_id);
