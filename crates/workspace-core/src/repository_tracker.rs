use crate::{Error, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    pub heartbeat_unix_ms: u64,
    pub detail: Option<String>,
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
             ON cow_repository_events(repository, epoch, generation);",
    )?;
    Ok(())
}

pub(crate) fn request(connection: &Connection, repository: &Path) -> Result<()> {
    let repository = path_text(repository)?;
    connection.execute(
        "INSERT INTO cow_repository_trackers(repository, state)
         VALUES(?1, 'requested')
         ON CONFLICT(repository) DO UPDATE SET
             state = CASE
                 WHEN cow_repository_trackers.state = 'active' THEN 'active'
                 ELSE 'requested'
             END,
             detail = CASE
                 WHEN cow_repository_trackers.state = 'active' THEN cow_repository_trackers.detail
                 ELSE NULL
             END",
        params![repository],
    )?;
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
             heartbeat_unix_ms = ?3, detail = NULL
         WHERE repository = ?1 AND state = 'requested'",
        params![repository_text, epoch as i64, heartbeat_unix_ms as i64],
    )?;
    if changed != 1 {
        return Err(Error::ConcurrentRepositoryMutation);
    }
    transaction.execute(
        "DELETE FROM cow_repository_events WHERE repository = ?1",
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
    let (state, epoch, generation): (String, i64, i64) = transaction
        .query_row(
            "SELECT state, epoch, generation FROM cow_repository_trackers
             WHERE repository = ?1",
            params![repository],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
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

pub(crate) fn mark_gap(
    connection: &Connection,
    repository: &Path,
    detail: &str,
    heartbeat_unix_ms: u64,
) -> Result<()> {
    let changed = connection.execute(
        "UPDATE cow_repository_trackers
         SET state = 'gap', detail = ?2, heartbeat_unix_ms = ?3
         WHERE repository = ?1 AND state != 'gap'",
        params![path_text(repository)?, detail, heartbeat_unix_ms as i64],
    )?;
    if changed == 0 && status(connection, repository)?.is_none() {
        return Err(Error::Corrupt(format!(
            "cannot mark unknown repository tracker as gap: {}",
            repository.display()
        )));
    }
    Ok(())
}

pub(crate) fn status(
    connection: &Connection,
    repository: &Path,
) -> Result<Option<RepositoryTrackerStatus>> {
    connection
        .query_row(
            "SELECT state, epoch, generation, heartbeat_unix_ms, detail
             FROM cow_repository_trackers WHERE repository = ?1",
            params![path_text(repository)?],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            },
        )
        .optional()?
        .map(|(state, epoch, generation, heartbeat, detail)| {
            Ok(RepositoryTrackerStatus {
                repository: repository.to_path_buf(),
                state: parse_state(&state)?,
                epoch: epoch as u64,
                generation: generation as u64,
                heartbeat_unix_ms: heartbeat as u64,
                detail,
            })
        })
        .transpose()
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
        request(&connection, &repository).unwrap();
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
        request(&connection, &repository).unwrap();
        let restarted = activate(&mut connection, &repository, 14).unwrap();
        assert!(restarted.epoch > active.epoch);
        assert_eq!(restarted.generation, 0);
    }
}
