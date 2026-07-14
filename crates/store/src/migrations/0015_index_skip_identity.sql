-- Persist the same filesystem identity used by the normal file_state
-- freshness fast path. Existing rows remain NULL and are conservatively
-- re-dispositioned once before the indexer records a complete identity.
ALTER TABLE index_skips ADD COLUMN ctime_ns INTEGER;
ALTER TABLE index_skips ADD COLUMN file_id INTEGER;
