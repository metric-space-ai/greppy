use crate::{Error, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Logical CoW block size. Every file, including a small one, is represented
/// exclusively by one or more chunks of at most this size.
pub const CHUNK_SIZE: usize = 1024 * 1024;
const SEGMENT_TARGET_SIZE: u64 = 256 * 1024 * 1024;
const RECORD_MAGIC: &[u8; 4] = b"GCW1";
const RECORD_HEADER_LEN: u64 = 4 + 4 + 32;

/// BLAKE3 identity of an immutable chunk.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub [u8; 32]);

impl ChunkId {
    pub fn from_hex(value: &str) -> Result<Self> {
        if value.len() != 64 {
            return Err(Error::Corrupt(format!(
                "chunk id has {} hex characters, expected 64",
                value.len()
            )));
        }
        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let pair = std::str::from_utf8(pair)
                .map_err(|_| Error::Corrupt("chunk id is not UTF-8 hex".into()))?;
            bytes[index] = u8::from_str_radix(pair, 16)
                .map_err(|_| Error::Corrupt("chunk id is not hexadecimal".into()))?;
        }
        Ok(Self(bytes))
    }

    pub fn to_hex(self) -> String {
        let mut value = String::with_capacity(64);
        for byte in self.0 {
            use fmt::Write as _;
            let _ = write!(value, "{byte:02x}");
        }
        value
    }
}

impl fmt::Debug for ChunkId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl fmt::Display for ChunkId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ChunkStoreStats {
    pub chunk_count: u64,
    pub referenced_chunks: u64,
    pub logical_bytes: u64,
    pub segment_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ChunkGcReport {
    pub removed_chunks: u64,
    pub removed_logical_bytes: u64,
    pub retained_chunks: u64,
    pub segment_bytes_before: u64,
    pub segment_bytes_after: u64,
}

/// Append-only, content-addressed storage. The SQLite committed length is the
/// authority for every segment, making a crash between `fsync` and the
/// metadata commit recoverable by truncating the uncommitted tail.
pub struct ChunkStore {
    root: PathBuf,
    connection: Mutex<Connection>,
}

impl ChunkStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("segments"))?;
        // Keep CAS reference accounting in its own WAL database. Namespace
        // transactions frequently pin or unpin chunks while their manifest
        // transaction is open; sharing one SQLite file would turn that valid
        // lock ordering into a cross-connection write lock.
        let connection = Connection::open(root.join("chunks.sqlite3"))?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS cow_segments (
                 id INTEGER PRIMARY KEY,
                 committed_len INTEGER NOT NULL CHECK(committed_len >= 0)
             );
             CREATE TABLE IF NOT EXISTS cow_chunks (
                 hash BLOB PRIMARY KEY CHECK(length(hash) = 32),
                 segment_id INTEGER NOT NULL REFERENCES cow_segments(id),
                 payload_offset INTEGER NOT NULL CHECK(payload_offset >= 0),
                 len INTEGER NOT NULL CHECK(len >= 0),
                 refs INTEGER NOT NULL DEFAULT 0 CHECK(refs >= 0)
             );
             INSERT OR IGNORE INTO cow_segments(id, committed_len) VALUES(1, 0);",
        )?;
        let store = Self {
            root,
            connection: Mutex::new(connection),
        };
        store.recover_uncommitted_tails()?;
        store.remove_orphan_segment_files()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn put(&self, bytes: &[u8]) -> Result<ChunkId> {
        if bytes.len() > CHUNK_SIZE {
            return Err(Error::Corrupt(format!(
                "chunk has {} bytes, maximum is {CHUNK_SIZE}",
                bytes.len()
            )));
        }
        let id = ChunkId(*blake3::hash(bytes).as_bytes());
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
        let existing: Option<i64> = connection
            .query_row(
                "SELECT len FROM cow_chunks WHERE hash = ?1",
                params![&id.0[..]],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(len) = existing {
            if len != bytes.len() as i64 {
                return Err(Error::Corrupt(format!(
                    "chunk hash {id} is registered with conflicting length {len}"
                )));
            }
            return Ok(id);
        }

        let transaction = connection.transaction()?;
        let (mut segment_id, mut committed_len): (i64, u64) = transaction.query_row(
            "SELECT id, committed_len FROM cow_segments ORDER BY id DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as u64)),
        )?;
        let record_len = RECORD_HEADER_LEN + bytes.len() as u64;
        if committed_len > 0 && committed_len + record_len > SEGMENT_TARGET_SIZE {
            segment_id += 1;
            committed_len = 0;
            transaction.execute(
                "INSERT INTO cow_segments(id, committed_len) VALUES(?1, 0)",
                params![segment_id],
            )?;
        }

        let path = self.segment_path(segment_id);
        let mut segment = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        segment.set_len(committed_len)?;
        segment.seek(SeekFrom::Start(committed_len))?;
        segment.write_all(RECORD_MAGIC)?;
        segment.write_all(&(bytes.len() as u32).to_le_bytes())?;
        segment.write_all(&id.0)?;
        segment.write_all(bytes)?;
        segment.sync_data()?;

        let payload_offset = committed_len + RECORD_HEADER_LEN;
        let new_committed_len = committed_len + record_len;
        transaction.execute(
            "INSERT INTO cow_chunks(hash, segment_id, payload_offset, len, refs)
             VALUES(?1, ?2, ?3, ?4, 0)",
            params![
                &id.0[..],
                segment_id,
                payload_offset as i64,
                bytes.len() as i64
            ],
        )?;
        transaction.execute(
            "UPDATE cow_segments SET committed_len = ?2 WHERE id = ?1",
            params![segment_id, new_committed_len as i64],
        )?;
        transaction.commit()?;
        Ok(id)
    }

    pub fn put_stream(&self, mut reader: impl Read) -> Result<(Vec<ChunkId>, u64)> {
        let mut ids = Vec::new();
        let mut total = 0_u64;
        loop {
            let mut buffer = vec![0_u8; CHUNK_SIZE];
            let mut filled = 0;
            while filled < buffer.len() {
                let read = reader.read(&mut buffer[filled..])?;
                if read == 0 {
                    break;
                }
                filled += read;
            }
            if filled == 0 {
                break;
            }
            buffer.truncate(filled);
            ids.push(self.put(&buffer)?);
            total += filled as u64;
        }
        if total == 0 {
            ids.push(self.put(&[])?);
        }
        Ok((ids, total))
    }

    pub fn read(&self, id: ChunkId) -> Result<Vec<u8>> {
        let (segment_id, offset, len): (i64, u64, usize) = {
            let connection = self
                .connection
                .lock()
                .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
            connection
                .query_row(
                    "SELECT segment_id, payload_offset, len FROM cow_chunks WHERE hash = ?1",
                    params![&id.0[..]],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get::<_, i64>(1)? as u64,
                            row.get::<_, i64>(2)? as usize,
                        ))
                    },
                )
                .optional()?
                .ok_or_else(|| Error::Corrupt(format!("unknown chunk {id}")))?
        };
        let mut segment = File::open(self.segment_path(segment_id))?;
        segment.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0_u8; len];
        segment.read_exact(&mut bytes)?;
        let actual = ChunkId(*blake3::hash(&bytes).as_bytes());
        if actual != id {
            return Err(Error::Corrupt(format!(
                "chunk {id} failed content verification (read {actual})"
            )));
        }
        Ok(bytes)
    }

    pub fn pin(&self, id: ChunkId) -> Result<()> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
        let changed = connection.execute(
            "UPDATE cow_chunks SET refs = refs + 1 WHERE hash = ?1",
            params![&id.0[..]],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt(format!("cannot pin unknown chunk {id}")));
        }
        Ok(())
    }

    pub fn unpin(&self, id: ChunkId) -> Result<()> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
        let changed = connection.execute(
            "UPDATE cow_chunks SET refs = refs - 1 WHERE hash = ?1 AND refs > 0",
            params![&id.0[..]],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt(format!(
                "cannot unpin unknown or unreferenced chunk {id}"
            )));
        }
        Ok(())
    }

    pub fn stats(&self) -> Result<ChunkStoreStats> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
        let (chunk_count, referenced_chunks, logical_bytes): (i64, i64, i64) = connection
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(refs > 0), 0), COALESCE(SUM(len), 0)
                 FROM cow_chunks",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        let segment_bytes: i64 = connection.query_row(
            "SELECT COALESCE(SUM(committed_len), 0) FROM cow_segments",
            [],
            |row| row.get(0),
        )?;
        Ok(ChunkStoreStats {
            chunk_count: chunk_count as u64,
            referenced_chunks: referenced_chunks as u64,
            logical_bytes: logical_bytes as u64,
            segment_bytes: segment_bytes as u64,
        })
    }

    pub fn verify(&self) -> Result<()> {
        let rows: Vec<(ChunkId, i64, u64, usize)> = {
            let connection = self
                .connection
                .lock()
                .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
            let mut statement = connection.prepare(
                "SELECT hash, segment_id, payload_offset, len FROM cow_chunks ORDER BY segment_id, payload_offset",
            )?;
            let mapped = statement.query_map([], |row| {
                let hash: Vec<u8> = row.get(0)?;
                let mut bytes = [0_u8; 32];
                bytes.copy_from_slice(&hash);
                Ok((
                    ChunkId(bytes),
                    row.get(1)?,
                    row.get::<_, i64>(2)? as u64,
                    row.get::<_, i64>(3)? as usize,
                ))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for (id, segment_id, offset, len) in rows {
            let mut segment = File::open(self.segment_path(segment_id))?;
            if offset < RECORD_HEADER_LEN {
                return Err(Error::Corrupt(format!(
                    "chunk {id} has invalid payload offset {offset}"
                )));
            }
            segment.seek(SeekFrom::Start(offset - RECORD_HEADER_LEN))?;
            let mut header = [0_u8; RECORD_HEADER_LEN as usize];
            segment.read_exact(&mut header)?;
            if &header[..4] != RECORD_MAGIC {
                return Err(Error::Corrupt(format!(
                    "chunk {id} has invalid record magic"
                )));
            }
            let recorded_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            if recorded_len != len || header[8..40] != id.0 {
                return Err(Error::Corrupt(format!(
                    "chunk {id} record header disagrees with metadata"
                )));
            }
            drop(segment);
            let _ = self.read(id)?;
        }
        Ok(())
    }

    /// Compact referenced chunks into fresh append-only segments. Fresh files
    /// are durable before the SQLite transaction switches every row to them;
    /// old files are deleted only after that commit. Either side of a crash is
    /// therefore recoverable without guessing which bytes are authoritative.
    pub fn gc(&self) -> Result<ChunkGcReport> {
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
        let (before_chunks, before_bytes, before_segments): (i64, i64, i64) = connection
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(len), 0),
                        COALESCE((SELECT SUM(committed_len) FROM cow_segments), 0)
                 FROM cow_chunks",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        let rows: Vec<(ChunkId, i64, u64, usize, i64)> = {
            let mut statement = connection.prepare(
                "SELECT hash, segment_id, payload_offset, len, refs
                 FROM cow_chunks WHERE refs > 0 ORDER BY hash",
            )?;
            let mapped = statement.query_map([], |row| {
                let hash: Vec<u8> = row.get(0)?;
                let mut bytes = [0_u8; 32];
                bytes.copy_from_slice(&hash);
                Ok((
                    ChunkId(bytes),
                    row.get(1)?,
                    row.get::<_, i64>(2)? as u64,
                    row.get::<_, i64>(3)? as usize,
                    row.get(4)?,
                ))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let old_segments: Vec<i64> = {
            let mut statement = connection.prepare("SELECT id FROM cow_segments ORDER BY id")?;
            let values = statement
                .query_map([], |row| row.get(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            values
        };
        let mut segment_id = old_segments.last().copied().unwrap_or(0) + 1;
        let mut segment_len = 0_u64;
        let mut segments = vec![(segment_id, 0_u64)];
        let mut updates = Vec::with_capacity(rows.len());
        let mut current = new_segment_file(&self.segment_path(segment_id))?;
        for (id, old_segment, old_offset, len, refs) in &rows {
            let mut source = File::open(self.segment_path(*old_segment))?;
            source.seek(SeekFrom::Start(*old_offset))?;
            let mut bytes = vec![0_u8; *len];
            source.read_exact(&mut bytes)?;
            if blake3::hash(&bytes).as_bytes() != &id.0 {
                return Err(Error::Corrupt(format!(
                    "chunk {id} failed verification during GC"
                )));
            }
            let record_len = RECORD_HEADER_LEN + *len as u64;
            if segment_len > 0 && segment_len + record_len > SEGMENT_TARGET_SIZE {
                current.sync_data()?;
                segments.last_mut().unwrap().1 = segment_len;
                segment_id += 1;
                segment_len = 0;
                segments.push((segment_id, 0));
                current = new_segment_file(&self.segment_path(segment_id))?;
            }
            current.write_all(RECORD_MAGIC)?;
            current.write_all(&(*len as u32).to_le_bytes())?;
            current.write_all(&id.0)?;
            current.write_all(&bytes)?;
            updates.push((*id, segment_id, segment_len + RECORD_HEADER_LEN, *refs));
            segment_len += record_len;
        }
        current.sync_data()?;
        segments.last_mut().unwrap().1 = segment_len;
        sync_directory(&self.root.join("segments"))?;

        let transaction = connection.transaction()?;
        for (id, len) in &segments {
            transaction.execute(
                "INSERT INTO cow_segments(id, committed_len) VALUES(?1, ?2)",
                params![id, *len as i64],
            )?;
        }
        transaction.execute("DELETE FROM cow_chunks WHERE refs = 0", [])?;
        for (id, new_segment, offset, refs) in &updates {
            let changed = transaction.execute(
                "UPDATE cow_chunks
                 SET segment_id = ?2, payload_offset = ?3, refs = ?4
                 WHERE hash = ?1",
                params![&id.0[..], new_segment, *offset as i64, refs],
            )?;
            if changed != 1 {
                return Err(Error::Corrupt(format!("chunk {id} disappeared during GC")));
            }
        }
        for id in &old_segments {
            transaction.execute("DELETE FROM cow_segments WHERE id = ?1", params![id])?;
        }
        transaction.commit()?;
        for id in old_segments {
            match fs::remove_file(self.segment_path(id)) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        sync_directory(&self.root.join("segments"))?;
        let after_bytes = segments.iter().map(|(_, len)| *len).sum();
        let retained_logical: i64 = rows.iter().map(|(_, _, _, len, _)| *len as i64).sum();
        Ok(ChunkGcReport {
            removed_chunks: (before_chunks - rows.len() as i64) as u64,
            removed_logical_bytes: (before_bytes - retained_logical) as u64,
            retained_chunks: rows.len() as u64,
            segment_bytes_before: before_segments as u64,
            segment_bytes_after: after_bytes,
        })
    }

    fn recover_uncommitted_tails(&self) -> Result<()> {
        let segments: Vec<(i64, u64)> = {
            let connection = self
                .connection
                .lock()
                .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
            let mut statement =
                connection.prepare("SELECT id, committed_len FROM cow_segments ORDER BY id")?;
            let rows =
                statement.query_map([], |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as u64)))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for (id, committed_len) in segments {
            let path = self.segment_path(id);
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(path)?;
            let actual = file.metadata()?.len();
            if actual < committed_len {
                return Err(Error::Corrupt(format!(
                    "segment {id} is truncated: {actual} bytes, metadata commits {committed_len}"
                )));
            }
            if actual > committed_len {
                file.set_len(committed_len)?;
                file.sync_data()?;
            }
        }
        Ok(())
    }

    fn remove_orphan_segment_files(&self) -> Result<()> {
        let known: std::collections::HashSet<PathBuf> = {
            let connection = self
                .connection
                .lock()
                .map_err(|_| Error::Corrupt("chunk metadata mutex poisoned".into()))?;
            let mut statement = connection.prepare("SELECT id FROM cow_segments")?;
            let paths = statement
                .query_map([], |row| row.get::<_, i64>(0))?
                .map(|id| id.map(|id| self.segment_path(id)))
                .collect::<std::result::Result<_, _>>()?;
            paths
        };
        for entry in fs::read_dir(self.root.join("segments"))? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "gcws")
                && !known.contains(&path)
            {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    fn segment_path(&self, segment_id: i64) -> PathBuf {
        self.root
            .join("segments")
            .join(format!("{segment_id:016x}.gcws"))
    }
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn new_segment_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_deduplicated_and_verified() {
        let temp = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(temp.path()).unwrap();
        let first = store.put(b"hello").unwrap();
        let second = store.put(b"hello").unwrap();
        assert_eq!(first, second);
        assert_eq!(store.read(first).unwrap(), b"hello");
        store.pin(first).unwrap();
        let stats = store.stats().unwrap();
        assert_eq!(stats.chunk_count, 1);
        assert_eq!(stats.referenced_chunks, 1);
        store.verify().unwrap();
    }

    #[test]
    fn streams_are_split_only_at_chunk_boundaries() {
        let temp = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(temp.path()).unwrap();
        let input = vec![7_u8; CHUNK_SIZE + 17];
        let (ids, len) = store.put_stream(input.as_slice()).unwrap();
        assert_eq!(len, input.len() as u64);
        assert_eq!(ids.len(), 2);
        assert_eq!(store.read(ids[0]).unwrap().len(), CHUNK_SIZE);
        assert_eq!(store.read(ids[1]).unwrap().len(), 17);
    }

    #[test]
    fn reopening_truncates_an_uncommitted_segment_tail() {
        let temp = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(temp.path()).unwrap();
        let id = store.put(b"committed").unwrap();
        let segment = store.segment_path(1);
        let committed = fs::metadata(&segment).unwrap().len();
        drop(store);
        OpenOptions::new()
            .append(true)
            .open(&segment)
            .unwrap()
            .write_all(b"orphaned-after-crash")
            .unwrap();
        assert!(fs::metadata(&segment).unwrap().len() > committed);
        let recovered = ChunkStore::open(temp.path()).unwrap();
        assert_eq!(fs::metadata(&segment).unwrap().len(), committed);
        assert_eq!(recovered.read(id).unwrap(), b"committed");
    }

    #[test]
    fn gc_rewrites_only_referenced_chunks_and_keeps_them_readable() {
        let temp = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(temp.path()).unwrap();
        let kept = store.put(b"kept").unwrap();
        store.pin(kept).unwrap();
        let removed = store.put(b"remove-me").unwrap();
        let before = store.stats().unwrap();
        let report = store.gc().unwrap();
        assert_eq!(report.removed_chunks, 1);
        assert_eq!(report.retained_chunks, 1);
        assert!(report.segment_bytes_after < before.segment_bytes);
        assert_eq!(store.read(kept).unwrap(), b"kept");
        assert!(store.read(removed).is_err());
        store.verify().unwrap();
    }
}
