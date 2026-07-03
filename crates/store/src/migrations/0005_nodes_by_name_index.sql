-- 0005_nodes_by_name_index.sql
-- Track C: by-name lookup foundation.
--
-- Ensure an index on nodes(project, name) exists so the resolver can do
-- an indexed by-name lookup instead of an O(edges*nodes) full scan.
--
-- `idx_nodes_name` is already created by 0001_initial on fresh databases,
-- but databases created before this guarantee (or any future schema edit
-- that drops it) may lack it. `CREATE INDEX IF NOT EXISTS` makes this
-- migration idempotent: a no-op where the index already exists, and a
-- repair where it does not. This is what lets list_nodes_by_name /
-- count_nodes_by_name run against an index on every DB at this version.

CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(project, name);
