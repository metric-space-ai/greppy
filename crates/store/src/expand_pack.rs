//! Short-lived prepared evidence packs for `greppy expand <id>`.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::store::Store;
use crate::store_error::{Error, Result};

pub const DEFAULT_EXPAND_TTL_SECS: u64 = 24 * 60 * 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewExpandPack {
    pub project: String,
    pub command: String,
    pub query: String,
    pub graph_generation: u64,
    pub summary_json: serde_json::Value,
    pub payload_text: String,
    pub payload_json: Option<serde_json::Value>,
    pub ttl_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpandPack {
    pub id: String,
    pub project: String,
    pub command: String,
    pub query: String,
    pub graph_generation: u64,
    pub created_at: u64,
    pub expires_at: u64,
    pub summary_json: serde_json::Value,
    pub payload_text: String,
    pub payload_json: Option<serde_json::Value>,
}

impl Store {
    pub fn insert_expand_pack(&self, pack: &NewExpandPack) -> Result<String> {
        let now = unix_now_secs();
        let ttl = pack.ttl_secs.max(1);
        let expires_at = now.saturating_add(ttl);
        self.prune_expired_expand_packs_at(now)?;

        let summary_json = serde_json::to_string(&pack.summary_json)
            .map_err(|e| Error::Store(format!("serialize expand summary: {e}")))?;
        let payload_json = pack
            .payload_json
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| Error::Store(format!("serialize expand payload JSON: {e}")))?;

        for attempt in 0..16u8 {
            let id = expand_pack_id(pack, now, attempt);
            let inserted = self.conn().execute(
                "INSERT OR IGNORE INTO expand_packs
                    (id, project, command, query, graph_generation, created_at,
                     expires_at, summary_json, payload_text, payload_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    id,
                    pack.project,
                    pack.command,
                    pack.query,
                    pack.graph_generation as i64,
                    now as i64,
                    expires_at as i64,
                    summary_json,
                    pack.payload_text,
                    payload_json,
                ],
            )?;
            if inserted == 1 {
                return Ok(id);
            }
        }

        Err(Error::Store(
            "could not allocate unique expand pack id after 16 attempts".into(),
        ))
    }

    pub fn get_expand_pack(&self, id: &str) -> Result<Option<ExpandPack>> {
        let id = id.trim();
        if id.is_empty() {
            return Ok(None);
        }
        let now = unix_now_secs();
        self.prune_expired_expand_packs_at(now)?;
        self.conn()
            .query_row(
                "SELECT id, project, command, query, graph_generation, created_at,
                        expires_at, summary_json, payload_text, payload_json
                 FROM expand_packs
                 WHERE id = ?1 AND expires_at > ?2",
                params![id, now as i64],
                row_to_expand_pack,
            )
            .optional()
            .map_err(Error::Sqlite)
    }

    pub fn prune_expired_expand_packs(&self) -> Result<usize> {
        self.prune_expired_expand_packs_at(unix_now_secs())
    }

    pub fn prune_expired_expand_packs_at(&self, now: u64) -> Result<usize> {
        self.conn()
            .execute(
                "DELETE FROM expand_packs WHERE expires_at <= ?1",
                params![now as i64],
            )
            .map_err(Error::Sqlite)
    }
}

fn row_to_expand_pack(row: &rusqlite::Row<'_>) -> rusqlite::Result<ExpandPack> {
    let summary_raw: String = row.get(7)?;
    let payload_json_raw: Option<String> = row.get(9)?;
    Ok(ExpandPack {
        id: row.get(0)?,
        project: row.get(1)?,
        command: row.get(2)?,
        query: row.get(3)?,
        graph_generation: row.get::<_, i64>(4)? as u64,
        created_at: row.get::<_, i64>(5)? as u64,
        expires_at: row.get::<_, i64>(6)? as u64,
        summary_json: serde_json::from_str(&summary_raw).unwrap_or(serde_json::Value::Null),
        payload_text: row.get(8)?,
        payload_json: payload_json_raw
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok()),
    })
}

fn expand_pack_id(pack: &NewExpandPack, now: u64, attempt: u8) -> String {
    let mut h = Sha256::new();
    h.update(pack.project.as_bytes());
    h.update([0]);
    h.update(pack.command.as_bytes());
    h.update([0]);
    h.update(pack.query.as_bytes());
    h.update([0]);
    h.update(pack.graph_generation.to_le_bytes());
    h.update(now.to_le_bytes());
    h.update([attempt]);
    h.update(pack.payload_text.as_bytes());
    let digest = h.finalize();
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pack() -> NewExpandPack {
        NewExpandPack {
            project: "p".into(),
            command: "semantic".into(),
            query: "how tasks schedule".into(),
            graph_generation: 7,
            summary_json: serde_json::json!({"spans": 2, "callsites": 1}),
            payload_text: "== evidence ==\nfn schedule() {}\n".into(),
            payload_json: Some(serde_json::json!({"hits": [{"file_path": "src/lib.rs"}]})),
            ttl_secs: DEFAULT_EXPAND_TTL_SECS,
        }
    }

    #[test]
    fn expand_pack_round_trips() {
        let store = Store::open_memory().unwrap();
        let id = store.insert_expand_pack(&sample_pack()).unwrap();
        assert_eq!(id.len(), 16);
        let got = store.get_expand_pack(&id).unwrap().unwrap();
        assert_eq!(got.id, id);
        assert_eq!(got.project, "p");
        assert_eq!(got.graph_generation, 7);
        assert_eq!(got.summary_json["spans"], 2);
        assert!(got.payload_text.contains("schedule"));
        assert_eq!(
            got.payload_json.unwrap()["hits"][0]["file_path"],
            "src/lib.rs"
        );
    }

    #[test]
    fn expired_expand_pack_is_not_returned() {
        let store = Store::open_memory().unwrap();
        let mut pack = sample_pack();
        pack.ttl_secs = 1;
        let id = store.insert_expand_pack(&pack).unwrap();
        let future = unix_now_secs() + 2;
        assert_eq!(store.prune_expired_expand_packs_at(future).unwrap(), 1);
        assert!(store.get_expand_pack(&id).unwrap().is_none());
    }

    #[test]
    fn missing_expand_pack_returns_none() {
        let store = Store::open_memory().unwrap();
        assert!(store.get_expand_pack("missing").unwrap().is_none());
    }
}
