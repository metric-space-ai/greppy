use crate::{Error, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const HEARTBEAT_INTERVAL_MS: u64 = 1_000;
pub const HEARTBEAT_STALE_AFTER_MS: u64 = 5 * HEARTBEAT_INTERVAL_MS;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RepositoryTrackerState {
    Requested,
    Active,
    Gap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepositoryTrackerStatus {
    pub repository: PathBuf,
    pub state: RepositoryTrackerState,
    pub epoch: u64,
    pub generation: u64,
    pub owner_pid: u32,
    pub heartbeat_unix_ms: u64,
    pub detail: Option<String>,
}

impl RepositoryTrackerStatus {
    pub fn is_live_at(&self, now_unix_ms: u64) -> bool {
        self.state == RepositoryTrackerState::Active
            && self.owner_pid != 0
            && process_is_alive(self.owner_pid)
            && now_unix_ms.saturating_sub(self.heartbeat_unix_ms) <= HEARTBEAT_STALE_AFTER_MS
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryChangeBatch {
    pub epoch: u64,
    pub generation: u64,
    pub paths: Vec<String>,
}

pub(crate) fn install_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS cow_repository_trackers (
             repository TEXT PRIMARY KEY,
             state TEXT NOT NULL CHECK(state IN ('requested', 'active', 'gap')),
             epoch INTEGER NOT NULL DEFAULT 0 CHECK(epoch >= 0),
             generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
             owner_pid INTEGER NOT NULL DEFAULT 0 CHECK(owner_pid >= 0),
             heartbeat_unix_ms INTEGER NOT NULL DEFAULT 0 CHECK(heartbeat_unix_ms >= 0),
             detail TEXT
         );
         CREATE TABLE IF NOT EXISTS cow_repository_events (
             repository TEXT NOT NULL REFERENCES cow_repository_trackers(repository)
                 ON DELETE CASCADE,
             epoch INTEGER NOT NULL CHECK(epoch >= 0),
             generation INTEGER NOT NULL CHECK(generation >= 0),
             path TEXT NOT NULL,
             PRIMARY KEY(repository, epoch, generation, path)
         );
         CREATE INDEX IF NOT EXISTS cow_repository_events_since
             ON cow_repository_events(repository, epoch, generation);
         CREATE TABLE IF NOT EXISTS cow_repository_fences (
             repository TEXT NOT NULL REFERENCES cow_repository_trackers(repository)
                 ON DELETE CASCADE,
             epoch INTEGER NOT NULL CHECK(epoch >= 0),
             path TEXT NOT NULL,
             observed_unix_ms INTEGER NOT NULL CHECK(observed_unix_ms >= 0),
             PRIMARY KEY(repository, epoch, path)
         );",
    )?;
    ensure_column(
        connection,
        "cow_repository_trackers",
        "owner_pid",
        "INTEGER NOT NULL DEFAULT 0 CHECK(owner_pid >= 0)",
    )?;
    Ok(())
}

fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    declaration: &str,
) -> Result<()> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        if row? == column {
            return Ok(());
        }
    }
    connection.execute_batch(&format!(
        "ALTER TABLE {table} ADD COLUMN {column} {declaration}"
    ))?;
    Ok(())
}

pub(crate) fn request(
    connection: &mut Connection,
    repository: &Path,
    now_unix_ms: u64,
) -> Result<()> {
    let repository = path_text(repository)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let existing: Option<(String, i64, i64)> = transaction
        .query_row(
            "SELECT state, owner_pid, heartbeat_unix_ms
             FROM cow_repository_trackers WHERE repository = ?1",
            params![repository],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let preserve_active = existing.as_ref().is_some_and(|(state, owner, heartbeat)| {
        let owner = u32::try_from(*owner).unwrap_or(0);
        state == "active"
            && owner != 0
            && process_is_alive(owner)
            && now_unix_ms.saturating_sub(*heartbeat as u64) <= HEARTBEAT_STALE_AFTER_MS
    });
    if existing.is_none() {
        transaction.execute(
            "INSERT INTO cow_repository_trackers(repository, state, owner_pid)
             VALUES(?1, 'requested', 0)",
            params![repository],
        )?;
    } else if !preserve_active {
        transaction.execute(
            "UPDATE cow_repository_trackers
             SET state = 'requested', generation = 0, owner_pid = 0,
                 heartbeat_unix_ms = 0, detail = NULL
             WHERE repository = ?1",
            params![repository],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

pub(crate) fn pending(connection: &Connection) -> Result<Vec<PathBuf>> {
    let mut statement = connection.prepare(
        "SELECT repository FROM cow_repository_trackers
         WHERE state = 'requested' ORDER BY repository",
    )?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(PathBuf::from)
        .collect())
}

pub(crate) fn activate(
    connection: &mut Connection,
    repository: &Path,
    heartbeat_unix_ms: u64,
) -> Result<RepositoryTrackerStatus> {
    let repository_text = path_text(repository)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let previous: Option<(String, i64)> = transaction
        .query_row(
            "SELECT state, epoch FROM cow_repository_trackers WHERE repository = ?1",
            params![repository_text],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let (state, previous_epoch) = previous.ok_or_else(|| {
        Error::Corrupt(format!(
            "cannot activate unregistered repository tracker: {repository_text}"
        ))
    })?;
    if state != "requested" {
        return Err(Error::Corrupt(format!(
            "cannot activate repository tracker from {state}: {repository_text}"
        )));
    }
    let epoch = previous_epoch as u64 + 1;
    let changed = transaction.execute(
        "UPDATE cow_repository_trackers
         SET state = 'active', epoch = ?2, generation = 0,
             owner_pid = ?3, heartbeat_unix_ms = ?4, detail = NULL
         WHERE repository = ?1 AND state = 'requested'",
        params![
            repository_text,
            epoch as i64,
            std::process::id() as i64,
            heartbeat_unix_ms as i64
        ],
    )?;
    if changed != 1 {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    transaction.execute(
        "DELETE FROM cow_repository_events WHERE repository = ?1",
        params![repository_text],
    )?;
    transaction.execute(
        "DELETE FROM cow_repository_fences WHERE repository = ?1",
        params![repository_text],
    )?;
    transaction.commit()?;
    status(connection, repository)?
        .ok_or_else(|| Error::Corrupt(format!("repository tracker disappeared: {repository_text}")))
}

pub(crate) fn record(
    connection: &mut Connection,
    repository: &Path,
    paths: &[String],
    heartbeat_unix_ms: u64,
) -> Result<()> {
    let repository = path_text(repository)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let (state, epoch, generation, owner_pid): (String, i64, i64, i64) = transaction
        .query_row(
            "SELECT state, epoch, generation, owner_pid FROM cow_repository_trackers
             WHERE repository = ?1",
            params![repository],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?
        .ok_or_else(|| Error::Corrupt(format!("unregistered repository tracker {repository}")))?;
    if state == "requested" {
        // The native watcher is installed before activation. Its successful
        // pre-activation events are covered by the first full double-capture.
        return Ok(());
    }
    if state != "active" {
        return Err(Error::Corrupt(format!(
            "repository tracker is not active: {repository} ({state})"
        )));
    }
    if owner_pid != std::process::id() as i64 {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    let next = generation + 1;
    transaction.execute(
        "UPDATE cow_repository_trackers
         SET generation = ?2, heartbeat_unix_ms = ?3
         WHERE repository = ?1",
        params![repository, next, heartbeat_unix_ms as i64],
    )?;
    for path in paths {
        transaction.execute(
            "INSERT OR IGNORE INTO cow_repository_events(
                 repository, epoch, generation, path
             ) VALUES(?1, ?2, ?3, ?4)",
            params![repository, epoch, next, path],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

pub(crate) fn record_fences(
    connection: &mut Connection,
    repository: &Path,
    paths: &[String],
    heartbeat_unix_ms: u64,
) -> Result<()> {
    let repository = path_text(repository)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let (state, epoch, owner_pid): (String, i64, i64) = transaction
        .query_row(
            "SELECT state, epoch, owner_pid FROM cow_repository_trackers
             WHERE repository = ?1",
            params![repository],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or_else(|| Error::Corrupt(format!("unregistered repository tracker {repository}")))?;
    if state == "requested" {
        return Ok(());
    }
    if state != "active" {
        return Err(Error::Corrupt(format!(
            "repository tracker is not active: {repository} ({state})"
        )));
    }
    if owner_pid != std::process::id() as i64 {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    transaction.execute(
        "UPDATE cow_repository_trackers SET heartbeat_unix_ms = ?2
         WHERE repository = ?1",
        params![repository, heartbeat_unix_ms as i64],
    )?;
    for path in paths {
        transaction.execute(
            "INSERT INTO cow_repository_fences(repository, epoch, path, observed_unix_ms)
             VALUES(?1, ?2, ?3, ?4)
             ON CONFLICT(repository, epoch, path) DO UPDATE
             SET observed_unix_ms = excluded.observed_unix_ms",
            params![repository, epoch, path, heartbeat_unix_ms as i64],
        )?;
    }
    transaction.execute(
        "DELETE FROM cow_repository_fences
         WHERE repository = ?1 AND observed_unix_ms < ?2",
        params![repository, heartbeat_unix_ms.saturating_sub(60_000) as i64],
    )?;
    transaction.commit()?;
    Ok(())
}

pub(crate) fn consume_fence(
    connection: &mut Connection,
    repository: &Path,
    epoch: u64,
    path: &str,
) -> Result<Option<u64>> {
    let repository = path_text(repository)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let (state, current_epoch, generation): (String, i64, i64) = transaction
        .query_row(
            "SELECT state, epoch, generation FROM cow_repository_trackers
             WHERE repository = ?1",
            params![repository],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or_else(|| Error::Corrupt(format!("unknown repository tracker: {repository}")))?;
    if state != "active" || current_epoch as u64 != epoch {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    let removed = transaction.execute(
        "DELETE FROM cow_repository_fences
         WHERE repository = ?1 AND epoch = ?2 AND path = ?3",
        params![repository, epoch as i64, path],
    )?;
    transaction.commit()?;
    Ok((removed == 1).then_some(generation as u64))
}

pub(crate) fn mark_gap(
    connection: &Connection,
    repository: &Path,
    detail: &str,
    heartbeat_unix_ms: u64,
) -> Result<()> {
    let changed = connection.execute(
        "UPDATE cow_repository_trackers
         SET state = 'gap', owner_pid = 0, detail = ?2, heartbeat_unix_ms = ?3
         WHERE repository = ?1
           AND (state = 'requested' OR (state = 'active' AND owner_pid = ?4))",
        params![
            path_text(repository)?,
            detail,
            heartbeat_unix_ms as i64,
            std::process::id() as i64
        ],
    )?;
    if changed == 0 {
        match status(connection, repository)? {
            None => {
                return Err(Error::Corrupt(format!(
                    "cannot mark unknown repository tracker as gap: {}",
                    repository.display()
                )))
            }
            Some(status) if status.state == RepositoryTrackerState::Gap => {}
            Some(_) => return Err(Error::ConcurrentRepositoryMutation),
        }
    }
    Ok(())
}

pub(crate) fn heartbeat(
    connection: &Connection,
    repository: &Path,
    heartbeat_unix_ms: u64,
) -> Result<()> {
    let changed = connection.execute(
        "UPDATE cow_repository_trackers SET heartbeat_unix_ms = ?2
         WHERE repository = ?1 AND state = 'active' AND owner_pid = ?3",
        params![
            path_text(repository)?,
            heartbeat_unix_ms as i64,
            std::process::id() as i64
        ],
    )?;
    if changed != 1 {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    Ok(())
}

pub(crate) fn status(
    connection: &Connection,
    repository: &Path,
) -> Result<Option<RepositoryTrackerStatus>> {
    connection
        .query_row(
            "SELECT state, epoch, generation, owner_pid, heartbeat_unix_ms, detail
             FROM cow_repository_trackers WHERE repository = ?1",
            params![path_text(repository)?],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Option<String>>(5)?,
                ))
            },
        )
        .optional()?
        .map(|(state, epoch, generation, owner_pid, heartbeat, detail)| {
            Ok(RepositoryTrackerStatus {
                repository: repository.to_path_buf(),
                state: parse_state(&state)?,
                epoch: epoch as u64,
                generation: generation as u64,
                owner_pid: u32::try_from(owner_pid).map_err(|_| {
                    Error::Corrupt(format!("invalid repository tracker owner pid {owner_pid}"))
                })?,
                heartbeat_unix_ms: heartbeat as u64,
                detail,
            })
        })
        .transpose()
}

#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    unsafe extern "C" {
        fn kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
    }
    let Ok(pid) = std::ffi::c_int::try_from(pid) else {
        return false;
    };
    let rc = unsafe { kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(1)
}

#[cfg(windows)]
fn process_is_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if process.is_null() {
        return false;
    }
    let mut exit_code = 0u32;
    let queried = unsafe { GetExitCodeProcess(process, &mut exit_code) } != 0;
    unsafe { CloseHandle(process) };
    queried && i32::try_from(exit_code).ok() == Some(STILL_ACTIVE)
}

#[cfg(not(any(unix, windows)))]
fn process_is_alive(_pid: u32) -> bool {
    false
}

pub(crate) fn changes_since(
    connection: &Connection,
    repository: &Path,
    epoch: u64,
    generation: u64,
) -> Result<RepositoryChangeBatch> {
    let status = status(connection, repository)?.ok_or_else(|| {
        Error::Corrupt(format!(
            "unknown repository tracker: {}",
            repository.display()
        ))
    })?;
    if status.state != RepositoryTrackerState::Active || status.epoch != epoch {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    let mut statement = connection.prepare(
        "SELECT DISTINCT path FROM cow_repository_events
         WHERE repository = ?1 AND epoch = ?2 AND generation > ?3
         ORDER BY path",
    )?;
    let rows = statement.query_map(
        params![path_text(repository)?, epoch as i64, generation as i64],
        |row| row.get::<_, String>(0),
    )?;
    Ok(RepositoryChangeBatch {
        epoch,
        generation: status.generation,
        paths: rows.collect::<std::result::Result<Vec<_>, _>>()?,
    })
}

fn parse_state(value: &str) -> Result<RepositoryTrackerState> {
    match value {
        "requested" => Ok(RepositoryTrackerState::Requested),
        "active" => Ok(RepositoryTrackerState::Active),
        "gap" => Ok(RepositoryTrackerState::Gap),
        other => Err(Error::Corrupt(format!(
            "invalid repository tracker state {other}"
        ))),
    }
}

fn path_text(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        Error::UnsupportedRepository(format!(
            "repository tracker path is not valid UTF-8: {}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_epoch_and_generation_fail_closed_across_gaps() {
        let temp = tempfile::tempdir().unwrap();
        let mut connection = Connection::open(temp.path().join("tracker.sqlite3")).unwrap();
        connection.execute_batch("PRAGMA foreign_keys=ON").unwrap();
        install_schema(&connection).unwrap();
        let repository = temp.path().join("repo");
        request(&mut connection, &repository, 8).unwrap();
        record(
            &mut connection,
            &repository,
            &["before-active.txt".into()],
            9,
        )
        .unwrap();
        assert_eq!(
            status(&connection, &repository)
                .unwrap()
                .unwrap()
                .generation,
            0
        );
        assert_eq!(
            pending(&connection).unwrap().as_slice(),
            std::slice::from_ref(&repository)
        );

        let active = activate(&mut connection, &repository, 10).unwrap();
        assert_eq!(active.state, RepositoryTrackerState::Active);
        assert_eq!(active.owner_pid, std::process::id());
        record(
            &mut connection,
            &repository,
            &["src/lib.rs".into(), "README.md".into()],
            11,
        )
        .unwrap();
        let changes = changes_since(&connection, &repository, active.epoch, 0).unwrap();
        assert_eq!(changes.paths, ["README.md", "src/lib.rs"]);
        assert_eq!(changes.generation, 1);

        mark_gap(&connection, &repository, "watcher overflow", 12).unwrap();
        assert!(changes_since(&connection, &repository, active.epoch, 1).is_err());
        assert!(activate(&mut connection, &repository, 13).is_err());
        request(&mut connection, &repository, 13).unwrap();
        let restarted = activate(&mut connection, &repository, 14).unwrap();
        assert!(restarted.epoch > active.epoch);
        assert_eq!(restarted.generation, 0);
    }

    #[test]
    fn request_preserves_a_fresh_live_owner() {
        let mut connection = Connection::open_in_memory().unwrap();
        install_schema(&connection).unwrap();
        let repository = Path::new("/repo");
        request(&mut connection, repository, 100).unwrap();
        let active = activate(&mut connection, repository, 101).unwrap();

        request(&mut connection, repository, 102).unwrap();
        assert_eq!(status(&connection, repository).unwrap().unwrap(), active);
    }

    #[test]
    fn request_reclaims_a_dead_owner_and_advances_epoch() {
        let mut connection = Connection::open_in_memory().unwrap();
        install_schema(&connection).unwrap();
        let repository = Path::new("/repo");
        request(&mut connection, repository, 100).unwrap();
        let active = activate(&mut connection, repository, 101).unwrap();
        connection
            .execute(
                "UPDATE cow_repository_trackers SET owner_pid = ?1 WHERE repository = ?2",
                params![u32::MAX as i64, path_text(repository).unwrap()],
            )
            .unwrap();

        request(&mut connection, repository, 102).unwrap();
        let requested = status(&connection, repository).unwrap().unwrap();
        assert_eq!(requested.state, RepositoryTrackerState::Requested);
        assert_eq!(requested.owner_pid, 0);
        let reclaimed = activate(&mut connection, repository, 103).unwrap();
        assert!(reclaimed.epoch > active.epoch);
        assert_eq!(reclaimed.owner_pid, std::process::id());
    }

    #[test]
    fn request_reclaims_a_stale_heartbeat_from_a_live_pid() {
        let mut connection = Connection::open_in_memory().unwrap();
        install_schema(&connection).unwrap();
        let repository = Path::new("/repo");
        request(&mut connection, repository, 100).unwrap();
        let active = activate(&mut connection, repository, 101).unwrap();

        request(
            &mut connection,
            repository,
            active.heartbeat_unix_ms + HEARTBEAT_STALE_AFTER_MS + 1,
        )
        .unwrap();
        let requested = status(&connection, repository).unwrap().unwrap();
        assert_eq!(requested.state, RepositoryTrackerState::Requested);
        assert_eq!(requested.owner_pid, 0);
    }

    #[test]
    fn fence_acknowledgements_do_not_advance_repository_generation() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection.execute_batch("PRAGMA foreign_keys=ON").unwrap();
        install_schema(&connection).unwrap();
        let repository = Path::new("/repo");
        request(&mut connection, repository, 100).unwrap();
        let active = activate(&mut connection, repository, 101).unwrap();
        let fence = ".git/greppy-tracker-fence-123-456-0".to_string();

        record_fences(
            &mut connection,
            repository,
            std::slice::from_ref(&fence),
            102,
        )
        .unwrap();
        assert_eq!(
            status(&connection, repository).unwrap().unwrap().generation,
            active.generation
        );
        assert_eq!(
            consume_fence(&mut connection, repository, active.epoch, &fence).unwrap(),
            Some(active.generation)
        );
        assert_eq!(
            consume_fence(&mut connection, repository, active.epoch, &fence).unwrap(),
            None
        );

        record(&mut connection, repository, &["src/lib.rs".into()], 103).unwrap();
        let changes =
            changes_since(&connection, repository, active.epoch, active.generation).unwrap();
        assert_eq!(changes.generation, active.generation + 1);
        assert_eq!(changes.paths, ["src/lib.rs"]);
    }

    #[test]
    fn schema_upgrade_adds_owner_pid_idempotently() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE cow_repository_trackers (
                     repository TEXT PRIMARY KEY,
                     state TEXT NOT NULL,
                     epoch INTEGER NOT NULL DEFAULT 0,
                     generation INTEGER NOT NULL DEFAULT 0,
                     heartbeat_unix_ms INTEGER NOT NULL DEFAULT 0,
                     detail TEXT
                 );",
            )
            .unwrap();
        install_schema(&connection).unwrap();
        install_schema(&connection).unwrap();
        let owner_columns = connection
            .prepare("PRAGMA table_info(cow_repository_trackers)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|column| column == "owner_pid")
            .count();
        assert_eq!(owner_columns, 1);
    }
}
