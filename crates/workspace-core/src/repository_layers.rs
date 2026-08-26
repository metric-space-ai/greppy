use crate::{BaselineSnapshot, ChunkId, ChunkStore, EntryKind, Error, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub(crate) struct LayerEntry {
    pub kind: LayerKind,
    pub mode: u32,
    pub size: u64,
    pub modified_unix_ns: i64,
    pub chunks: Vec<ChunkId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    File,
    Directory,
    Symlink,
    Tombstone,
}

#[derive(Debug)]
struct TreeEntry {
    path: String,
    parent: String,
    name: String,
    kind: LayerKind,
    mode: u32,
    oid: Option<String>,
    size: u64,
    chunks: Vec<ChunkId>,
}

pub(crate) fn install_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS cow_repository_bases (
             id TEXT PRIMARY KEY,
             repository TEXT NOT NULL,
             base_commit TEXT NOT NULL,
             state TEXT NOT NULL CHECK(state IN ('ready', 'broken')),
             UNIQUE(repository, base_commit)
         );
         CREATE TABLE IF NOT EXISTS cow_repository_base_entries (
             base_id TEXT NOT NULL REFERENCES cow_repository_bases(id) ON DELETE CASCADE,
             path TEXT NOT NULL,
             parent TEXT NOT NULL,
             name TEXT NOT NULL,
             kind TEXT NOT NULL CHECK(kind IN ('file', 'directory', 'symlink')),
             mode INTEGER NOT NULL,
             size INTEGER NOT NULL CHECK(size >= 0),
             chunks_json BLOB NOT NULL,
             PRIMARY KEY(base_id, path)
         );
         CREATE INDEX IF NOT EXISTS cow_repository_base_parent
             ON cow_repository_base_entries(base_id, parent, name);
         CREATE TABLE IF NOT EXISTS cow_dirty_layers (
             id TEXT PRIMARY KEY,
             base_id TEXT NOT NULL REFERENCES cow_repository_bases(id),
             baseline_json BLOB NOT NULL
         );
         CREATE TABLE IF NOT EXISTS cow_dirty_entries (
             layer_id TEXT NOT NULL REFERENCES cow_dirty_layers(id) ON DELETE CASCADE,
             path TEXT NOT NULL,
             parent TEXT NOT NULL,
             name TEXT NOT NULL,
             kind TEXT NOT NULL CHECK(kind IN ('file', 'symlink', 'tombstone')),
             mode INTEGER NOT NULL,
             size INTEGER NOT NULL CHECK(size >= 0),
             modified_unix_ns INTEGER NOT NULL,
             chunks_json BLOB NOT NULL,
             PRIMARY KEY(layer_id, path)
         );
         CREATE INDEX IF NOT EXISTS cow_dirty_parent
             ON cow_dirty_entries(layer_id, parent, name);
         CREATE TABLE IF NOT EXISTS cow_workspace_layers (
             workspace_id TEXT PRIMARY KEY REFERENCES cow_workspaces(id) ON DELETE CASCADE,
             base_id TEXT NOT NULL REFERENCES cow_repository_bases(id),
             dirty_layer_id TEXT NOT NULL REFERENCES cow_dirty_layers(id)
         );
         CREATE TABLE IF NOT EXISTS cow_overlay_templates (
             id TEXT PRIMARY KEY,
             dirty_layer_id TEXT NOT NULL UNIQUE REFERENCES cow_dirty_layers(id)
         );",
    )?;
    Ok(())
}

pub(crate) fn ensure_layers(
    connection: &mut Connection,
    store: &ChunkStore,
    baseline: &BaselineSnapshot,
    captured_snapshot_owns_chunks: bool,
    empty_base: bool,
) -> Result<(String, String)> {
    let base_id = if empty_base {
        ensure_empty_base(connection, baseline)?
    } else {
        ensure_repository_base(connection, store, baseline)?
    };
    let dirty_id = format!("dirty:{}", baseline.baseline_hash);
    let exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM cow_dirty_layers WHERE id = ?1)",
        params![dirty_id],
        |row| row.get(0),
    )?;
    if exists {
        if captured_snapshot_owns_chunks {
            release_snapshot_chunks(store, baseline)?;
        }
        return Ok((base_id, dirty_id));
    }

    if !captured_snapshot_owns_chunks {
        retain_snapshot_chunks(store, baseline)?;
    }
    let baseline_json = serde_json::to_vec(baseline)?;
    let transaction = connection.transaction()?;
    let inserted = (|| -> Result<()> {
        transaction.execute(
            "INSERT INTO cow_dirty_layers(id, base_id, baseline_json) VALUES(?1, ?2, ?3)",
            params![dirty_id, base_id, baseline_json],
        )?;
        for entry in &baseline.entries {
            let (parent, name) = split_parent(&entry.path);
            transaction.execute(
                "INSERT INTO cow_dirty_entries(
                     layer_id, path, parent, name, kind, mode, size,
                     modified_unix_ns, chunks_json
                 ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    dirty_id,
                    entry.path,
                    parent,
                    name,
                    kind_text(entry.kind),
                    entry.mode as i64,
                    entry.size as i64,
                    entry.modified_unix_ns,
                    serde_json::to_vec(&entry.chunks)?
                ],
            )?;
        }
        transaction.commit()?;
        Ok(())
    })();
    if let Err(error) = inserted {
        if !captured_snapshot_owns_chunks {
            let _ = release_snapshot_chunks(store, baseline);
        }
        return Err(error);
    }
    Ok((base_id, dirty_id))
}

fn ensure_empty_base(connection: &mut Connection, baseline: &BaselineSnapshot) -> Result<String> {
    if !baseline.base_commit.starts_with("virtual-empty:") {
        return Err(Error::UnsupportedRepository(
            "empty overlay base is missing its virtual-empty identity".into(),
        ));
    }
    let repository = baseline.repository.to_str().ok_or_else(|| {
        Error::UnsupportedRepository("overlay identity is not valid UTF-8".into())
    })?;
    if let Some(id) = connection
        .query_row(
            "SELECT id FROM cow_repository_bases
             WHERE repository = ?1 AND base_commit = ?2 AND state = 'ready'",
            params![repository, baseline.base_commit],
            |row| row.get(0),
        )
        .optional()?
    {
        return Ok(id);
    }
    let base_id = format!(
        "empty-base:{}",
        blake3::hash(format!("{repository}\0{}", baseline.base_commit).as_bytes()).to_hex()
    );
    connection.execute(
        "INSERT INTO cow_repository_bases(id, repository, base_commit, state)
         VALUES(?1, ?2, ?3, 'ready')",
        params![base_id, repository, baseline.base_commit],
    )?;
    Ok(base_id)
}

fn ensure_repository_base(
    connection: &mut Connection,
    store: &ChunkStore,
    baseline: &BaselineSnapshot,
) -> Result<String> {
    let repository = baseline
        .repository
        .to_str()
        .ok_or_else(|| Error::UnsupportedRepository("repository path is not valid UTF-8".into()))?;
    if let Some(id) = connection
        .query_row(
            "SELECT id FROM cow_repository_bases
             WHERE repository = ?1 AND base_commit = ?2 AND state = 'ready'",
            params![repository, baseline.base_commit],
            |row| row.get(0),
        )
        .optional()?
    {
        return Ok(id);
    }

    let base_id = format!(
        "base:{}",
        blake3::hash(format!("{repository}\0{}", baseline.base_commit).as_bytes()).to_hex()
    );
    let mut entries = list_tree(&baseline.repository, &baseline.base_commit)?;
    hydrate_blobs(&baseline.repository, store, &mut entries)?;
    let mut retained = HashMap::<ChunkId, u64>::new();
    for entry in &entries {
        for chunk in entry_chunks(entry) {
            *retained.entry(*chunk).or_default() += 1;
        }
    }
    let retained = retained.into_iter().collect::<Vec<_>>();
    store.pin_many(&retained)?;

    let transaction = connection.transaction()?;
    let inserted = (|| -> Result<()> {
        transaction.execute(
            "INSERT INTO cow_repository_bases(id, repository, base_commit, state)
             VALUES(?1, ?2, ?3, 'ready')",
            params![base_id, repository, baseline.base_commit],
        )?;
        for entry in &entries {
            transaction.execute(
                "INSERT INTO cow_repository_base_entries(
                     base_id, path, parent, name, kind, mode, size, chunks_json
                 ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    base_id,
                    entry.path,
                    entry.parent,
                    entry.name,
                    layer_kind_text(entry.kind),
                    entry.mode as i64,
                    entry.size as i64,
                    serde_json::to_vec(entry_chunks(entry))?
                ],
            )?;
        }
        transaction.commit()?;
        Ok(())
    })();
    if let Err(error) = inserted {
        let _ = store.unpin_many(&retained);
        return Err(error);
    }
    Ok(base_id)
}

pub(crate) fn link_workspace(
    transaction: &rusqlite::Transaction<'_>,
    workspace_id: &str,
    base_id: &str,
    dirty_layer_id: &str,
) -> Result<()> {
    transaction.execute(
        "INSERT INTO cow_workspace_layers(workspace_id, base_id, dirty_layer_id)
         VALUES(?1, ?2, ?3)",
        params![workspace_id, base_id, dirty_layer_id],
    )?;
    Ok(())
}

pub(crate) fn retain_overlay_template(
    transaction: &rusqlite::Transaction<'_>,
    baseline_hash: &str,
    dirty_layer_id: &str,
) -> Result<()> {
    transaction.execute(
        "INSERT OR IGNORE INTO cow_overlay_templates(id, dirty_layer_id) VALUES(?1, ?2)",
        params![baseline_hash, dirty_layer_id],
    )?;
    let observed: String = transaction.query_row(
        "SELECT dirty_layer_id FROM cow_overlay_templates WHERE id = ?1",
        params![baseline_hash],
        |row| row.get(0),
    )?;
    if observed != dirty_layer_id {
        return Err(Error::Corrupt(format!(
            "overlay template {baseline_hash} points at an unexpected immutable layer"
        )));
    }
    Ok(())
}

pub(crate) fn lookup(
    connection: &Connection,
    workspace_id: &str,
    path: &str,
) -> Result<Option<LayerEntry>> {
    let (base_id, dirty_id) = workspace_layers(connection, workspace_id)?;
    if let Some(entry) = lookup_table(connection, "cow_dirty_entries", "layer_id", &dirty_id, path)?
    {
        return if entry.kind == LayerKind::Tombstone {
            Ok(None)
        } else {
            Ok(Some(entry))
        };
    }
    lookup_table(
        connection,
        "cow_repository_base_entries",
        "base_id",
        &base_id,
        path,
    )
}

pub(crate) fn list_names(
    connection: &Connection,
    workspace_id: &str,
    parent: &str,
) -> Result<Vec<String>> {
    let (base_id, dirty_id) = workspace_layers(connection, workspace_id)?;
    let mut names = std::collections::BTreeSet::new();
    for (table, column, id) in [
        ("cow_repository_base_entries", "base_id", base_id.as_str()),
        ("cow_dirty_entries", "layer_id", dirty_id.as_str()),
    ] {
        let sql = format!("SELECT name FROM {table} WHERE {column} = ?1 AND parent = ?2");
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(params![id, parent], |row| row.get::<_, String>(0))?;
        for row in rows {
            names.insert(row?);
        }
    }
    Ok(names.into_iter().collect())
}

pub(crate) fn has_descendant(
    connection: &Connection,
    workspace_id: &str,
    path: &str,
) -> Result<bool> {
    let (base_id, dirty_id) = workspace_layers(connection, workspace_id)?;
    let prefix = if path.is_empty() {
        "%".to_string()
    } else {
        format!("{}/%", escape_like(path))
    };
    for (table, column, id) in [
        ("cow_repository_base_entries", "base_id", base_id.as_str()),
        ("cow_dirty_entries", "layer_id", dirty_id.as_str()),
    ] {
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {table}
             WHERE {column} = ?1 AND path LIKE ?2 ESCAPE '\\')"
        );
        let exists: bool = connection.query_row(&sql, params![id, prefix], |row| row.get(0))?;
        if exists {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn count_references(connection: &Connection) -> Result<Vec<Vec<ChunkId>>> {
    let mut all = Vec::new();
    for table in ["cow_repository_base_entries", "cow_dirty_entries"] {
        let mut statement = connection.prepare(&format!("SELECT chunks_json FROM {table}"))?;
        let rows = statement.query_map([], |row| row.get::<_, Vec<u8>>(0))?;
        for row in rows {
            all.push(serde_json::from_slice(&row?)?);
        }
    }
    let mut statement = connection.prepare("SELECT baseline_json FROM cow_dirty_layers")?;
    let rows = statement.query_map([], |row| row.get::<_, Vec<u8>>(0))?;
    for row in rows {
        let snapshot: BaselineSnapshot = serde_json::from_slice(&row?)?;
        all.push(snapshot.index_chunks);
    }
    Ok(all)
}

pub(crate) fn cached_snapshot(
    connection: &Connection,
    repository: &Path,
    tracker_epoch: u64,
) -> Result<Option<BaselineSnapshot>> {
    let repository = repository
        .to_str()
        .ok_or_else(|| Error::UnsupportedRepository("repository path is not valid UTF-8".into()))?;
    let mut statement = connection.prepare(
        "SELECT d.baseline_json
         FROM cow_dirty_layers d
         JOIN cow_repository_bases b ON b.id = d.base_id
         WHERE b.repository = ?1
         ORDER BY d.rowid DESC",
    )?;
    let rows = statement.query_map(params![repository], |row| row.get::<_, Vec<u8>>(0))?;
    for row in rows {
        let snapshot: BaselineSnapshot = serde_json::from_slice(&row?)?;
        if snapshot.tracker_epoch == Some(tracker_epoch) && snapshot.tracker_generation.is_some() {
            return Ok(Some(snapshot));
        }
    }
    Ok(None)
}

pub(crate) fn remove_unreferenced(connection: &mut Connection, store: &ChunkStore) -> Result<()> {
    let unreferenced_dirty: Vec<String> = {
        let mut statement = connection.prepare(
            "SELECT id FROM cow_dirty_layers d
             WHERE NOT EXISTS(
                 SELECT 1 FROM cow_workspace_layers w WHERE w.dirty_layer_id = d.id
             ) AND NOT EXISTS(
                 SELECT 1 FROM cow_proposals p
                 WHERE ('dirty:' || p.baseline_hash) = d.id
             ) AND NOT EXISTS(
                 SELECT 1 FROM cow_overlay_templates t WHERE t.dirty_layer_id = d.id
             )",
        )?;
        let ids = statement
            .query_map([], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ids
    };
    let mut released = Vec::new();
    for id in &unreferenced_dirty {
        let baseline: Vec<u8> = connection.query_row(
            "SELECT baseline_json FROM cow_dirty_layers WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        let baseline: BaselineSnapshot = serde_json::from_slice(&baseline)?;
        released.extend(snapshot_chunks(&baseline));
    }
    let transaction = connection.transaction()?;
    for id in &unreferenced_dirty {
        transaction.execute("DELETE FROM cow_dirty_layers WHERE id = ?1", params![id])?;
    }
    transaction.commit()?;

    let unreferenced_bases: Vec<String> = {
        let mut statement = connection.prepare(
            "SELECT id FROM cow_repository_bases b
             WHERE NOT EXISTS(SELECT 1 FROM cow_dirty_layers d WHERE d.base_id = b.id)",
        )?;
        let ids = statement
            .query_map([], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ids
    };
    for id in &unreferenced_bases {
        let mut statement = connection
            .prepare("SELECT chunks_json FROM cow_repository_base_entries WHERE base_id = ?1")?;
        let rows = statement.query_map(params![id], |row| row.get::<_, Vec<u8>>(0))?;
        for row in rows {
            released.extend(serde_json::from_slice::<Vec<ChunkId>>(&row?)?);
        }
    }
    let transaction = connection.transaction()?;
    for id in &unreferenced_bases {
        transaction.execute(
            "DELETE FROM cow_repository_bases WHERE id = ?1",
            params![id],
        )?;
    }
    transaction.commit()?;
    for chunk in released {
        store.unpin(chunk)?;
    }
    Ok(())
}

fn workspace_layers(connection: &Connection, workspace_id: &str) -> Result<(String, String)> {
    connection
        .query_row(
            "SELECT base_id, dirty_layer_id FROM cow_workspace_layers WHERE workspace_id = ?1",
            params![workspace_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
        .ok_or_else(|| Error::Corrupt(format!("workspace {workspace_id} has no layer binding")))
}

fn lookup_table(
    connection: &Connection,
    table: &str,
    id_column: &str,
    id: &str,
    path: &str,
) -> Result<Option<LayerEntry>> {
    let sql = format!(
        "SELECT kind, mode, size, chunks_json{} FROM {table}
         WHERE {id_column} = ?1 AND path = ?2",
        if table == "cow_dirty_entries" {
            ", modified_unix_ns"
        } else {
            ", 0"
        }
    );
    connection
        .query_row(&sql, params![id, path], |row| {
            let kind: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(3)?;
            Ok((
                kind,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                bytes,
                row.get(4)?,
            ))
        })
        .optional()?
        .map(|(kind, mode, size, chunks, modified)| {
            Ok(LayerEntry {
                kind: parse_kind(&kind)?,
                mode: mode as u32,
                size: size as u64,
                modified_unix_ns: modified,
                chunks: serde_json::from_slice(&chunks)?,
            })
        })
        .transpose()
}

fn list_tree(repository: &Path, commit: &str) -> Result<Vec<TreeEntry>> {
    let output = Command::new("git")
        .args(["ls-tree", "-r", "-t", "-z", "-l", commit])
        .current_dir(repository)
        .output()?;
    if !output.status.success() {
        return Err(git_error("git ls-tree", &output.stderr));
    }
    let mut entries = Vec::new();
    for record in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|r| !r.is_empty())
    {
        let tab = record
            .iter()
            .position(|byte| *byte == b'\t')
            .ok_or_else(|| Error::Git {
                command: "git ls-tree".into(),
                detail: "record has no path separator".into(),
            })?;
        let header = String::from_utf8(record[..tab].to_vec())
            .map_err(|_| Error::UnsupportedRepository("non-UTF-8 Git tree header".into()))?;
        let path = String::from_utf8(record[tab + 1..].to_vec())
            .map_err(|_| Error::UnsupportedRepository("non-UTF-8 Git path".into()))?;
        let fields: Vec<_> = header.split_whitespace().collect();
        if fields.len() != 4 {
            return Err(Error::Git {
                command: "git ls-tree".into(),
                detail: format!("unexpected record: {header}"),
            });
        }
        let mut mode = u32::from_str_radix(fields[0], 8).map_err(|_| Error::Git {
            command: "git ls-tree".into(),
            detail: format!("invalid mode {}", fields[0]),
        })?;
        let (kind, oid, size) = match fields[1] {
            "tree" => {
                mode = 0o040755;
                (LayerKind::Directory, None, 0)
            }
            "blob" => (
                if mode == 0o120000 {
                    LayerKind::Symlink
                } else {
                    LayerKind::File
                },
                Some(fields[2].to_string()),
                fields[3].parse::<u64>().map_err(|_| Error::Git {
                    command: "git ls-tree".into(),
                    detail: format!("invalid blob size {}", fields[3]),
                })?,
            ),
            other => {
                return Err(Error::UnsupportedRepository(format!(
                    "unsupported Git object type {other}"
                )))
            }
        };
        let (parent, name) = split_parent(&path);
        entries.push(TreeEntry {
            path,
            parent,
            name,
            kind,
            mode,
            oid,
            size,
            chunks: Vec::new(),
        });
    }
    Ok(entries)
}

fn hydrate_blobs(
    repository: &Path,
    store: &ChunkStore,
    entries: &mut [TreeEntry],
) -> Result<usize> {
    let mut by_oid = BTreeMap::<String, Vec<usize>>::new();
    for (index, entry) in entries.iter().enumerate() {
        if let Some(oid) = &entry.oid {
            by_oid.entry(oid.clone()).or_default().push(index);
        }
    }
    let mut child = Command::new("git")
        .args(["cat-file", "--batch"])
        .current_dir(repository)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut input = child.stdin.take().ok_or_else(|| Error::Git {
        command: "git cat-file --batch".into(),
        detail: "stdin unavailable".into(),
    })?;
    let output = child.stdout.take().ok_or_else(|| Error::Git {
        command: "git cat-file --batch".into(),
        detail: "stdout unavailable".into(),
    })?;
    let mut output = BufReader::new(output);
    for (oid, indexes) in &by_oid {
        writeln!(input, "{oid}")?;
        input.flush()?;
        let mut header = String::new();
        output.read_line(&mut header)?;
        let fields: Vec<_> = header.split_whitespace().collect();
        if fields.len() != 3 || fields[0] != oid || fields[1] != "blob" {
            return Err(Error::Git {
                command: "git cat-file --batch".into(),
                detail: format!("unexpected response: {}", header.trim_end()),
            });
        }
        let size = fields[2].parse::<usize>().map_err(|_| Error::Git {
            command: "git cat-file --batch".into(),
            detail: format!("invalid size {}", fields[2]),
        })?;
        for index in indexes {
            if size as u64 != entries[*index].size {
                return Err(Error::Corrupt(format!(
                    "Git blob {oid} changed size during Base import"
                )));
            }
        }
        let mut bytes = vec![0_u8; size];
        output.read_exact(&mut bytes)?;
        let mut terminator = [0_u8; 1];
        output.read_exact(&mut terminator)?;
        if terminator[0] != b'\n' {
            return Err(Error::Git {
                command: "git cat-file --batch".into(),
                detail: "blob terminator missing".into(),
            });
        }
        let (chunks, _) = store.put_stream(bytes.as_slice())?;
        for index in indexes {
            entries[*index].chunks.clone_from(&chunks);
        }
    }
    drop(input);
    let status = child.wait()?;
    if !status.success() {
        let mut stderr = String::new();
        if let Some(mut stream) = child.stderr.take() {
            let _ = stream.read_to_string(&mut stderr);
        }
        return Err(Error::Git {
            command: "git cat-file --batch".into(),
            detail: stderr.trim().into(),
        });
    }
    Ok(by_oid.len())
}

fn entry_chunks(entry: &TreeEntry) -> &[ChunkId] {
    &entry.chunks
}

fn retain_snapshot_chunks(store: &ChunkStore, baseline: &BaselineSnapshot) -> Result<()> {
    store.pin_many(&chunk_counts(snapshot_chunks(baseline)))
}

fn release_snapshot_chunks(store: &ChunkStore, baseline: &BaselineSnapshot) -> Result<()> {
    store.unpin_many(&chunk_counts(snapshot_chunks(baseline)))
}

fn chunk_counts(chunks: impl Iterator<Item = ChunkId>) -> Vec<(ChunkId, u64)> {
    let mut counts = HashMap::<ChunkId, u64>::new();
    for chunk in chunks {
        *counts.entry(chunk).or_default() += 1;
    }
    counts.into_iter().collect()
}

fn snapshot_chunks(snapshot: &BaselineSnapshot) -> impl Iterator<Item = ChunkId> + '_ {
    snapshot
        .entries
        .iter()
        .flat_map(|entry| entry.chunks.iter().copied())
        .chain(snapshot.index_chunks.iter().copied())
}

fn split_parent(path: &str) -> (String, String) {
    path.rsplit_once('/')
        .map(|(parent, name)| (parent.to_string(), name.to_string()))
        .unwrap_or_else(|| (String::new(), path.to_string()))
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn kind_text(kind: EntryKind) -> &'static str {
    match kind {
        EntryKind::File => "file",
        EntryKind::Symlink => "symlink",
        EntryKind::Tombstone => "tombstone",
    }
}

fn layer_kind_text(kind: LayerKind) -> &'static str {
    match kind {
        LayerKind::File => "file",
        LayerKind::Directory => "directory",
        LayerKind::Symlink => "symlink",
        LayerKind::Tombstone => "tombstone",
    }
}

fn parse_kind(kind: &str) -> Result<LayerKind> {
    match kind {
        "file" => Ok(LayerKind::File),
        "directory" => Ok(LayerKind::Directory),
        "symlink" => Ok(LayerKind::Symlink),
        "tombstone" => Ok(LayerKind::Tombstone),
        other => Err(Error::Corrupt(format!(
            "invalid repository layer kind {other}"
        ))),
    }
}

fn git_error(command: &str, stderr: &[u8]) -> Error {
    Error::Git {
        command: command.into(),
        detail: String::from_utf8_lossy(stderr).trim().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn git(repository: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(repository)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn repeated_tree_oids_are_hydrated_once() {
        let repository = tempfile::tempdir().unwrap();
        git(repository.path(), &["init", "-q"]);
        git(
            repository.path(),
            &["config", "user.email", "test@example.test"],
        );
        git(repository.path(), &["config", "user.name", "Test"]);
        fs::write(repository.path().join("first.txt"), b"shared blob\n").unwrap();
        fs::write(repository.path().join("second.txt"), b"shared blob\n").unwrap();
        git(repository.path(), &["add", "."]);
        git(repository.path(), &["commit", "-qm", "base"]);

        let commit = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(repository.path())
            .output()
            .unwrap();
        assert!(commit.status.success());
        let commit = String::from_utf8(commit.stdout).unwrap();
        let mut entries = list_tree(repository.path(), commit.trim()).unwrap();
        let storage = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(storage.path()).unwrap();

        assert_eq!(
            hydrate_blobs(repository.path(), &store, &mut entries).unwrap(),
            1
        );
        let files = entries
            .iter()
            .filter(|entry| entry.oid.is_some())
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].chunks, files[1].chunks);
        assert_eq!(store.stats().unwrap().chunk_count, 1);
    }
}
