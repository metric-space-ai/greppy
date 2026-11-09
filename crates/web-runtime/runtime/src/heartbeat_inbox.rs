use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

struct Entry {
    publication: u64,
    received: Option<Instant>,
}

#[derive(Default)]
struct State {
    publication: u64,
    entries: HashMap<String, Entry>,
    has_pending: bool,
}

#[derive(Default)]
pub(crate) struct HeartbeatInbox {
    state: Mutex<State>,
}

impl HeartbeatInbox {
    // The daemon's published live-session inventory is the only admission source.
    // Existing keys are reused: publication and repeated heartbeats clone no IDs.
    pub(crate) fn publish<'a>(&self, sessions: impl Iterator<Item = &'a str>) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let publication = state.publication.wrapping_add(1);
        state.publication = publication;
        for session in sessions {
            if let Some(entry) = state.entries.get_mut(session) {
                entry.publication = publication;
            } else {
                state.entries.insert(
                    session.to_owned(),
                    Entry {
                        publication,
                        received: None,
                    },
                );
            }
        }
        state
            .entries
            .retain(|_, entry| entry.publication == publication);
        // A closed large inventory must not retain its historical bucket array.
        if state.entries.capacity() > state.entries.len().saturating_mul(4) {
            state.entries.shrink_to_fit();
        }
    }

    pub(crate) fn record(&self, session: &str, received: Instant) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(entry) = state.entries.get_mut(session) else {
            return false;
        };
        entry.received = Some(entry.received.map_or(received, |old| old.max(received)));
        state.has_pending = true;
        true
    }

    // Only a short in-memory update of the existing session map runs under this
    // lock. No receipt Vec or per-message String allocation is needed.
    pub(crate) fn drain(&self, mut apply: impl FnMut(&str, Instant)) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if !state.has_pending {
            return;
        }
        for (session, entry) in &mut state.entries {
            if let Some(received) = entry.received.take() {
                apply(session, received);
            }
        }
        state.has_pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::HeartbeatInbox;
    use std::time::{Duration, Instant};

    #[test]
    fn known_sessions_coalesce_and_consumed_or_closed_receipts_do_not_replay() {
        let inbox = HeartbeatInbox::default();
        inbox.publish(["a", "b"].into_iter());
        let first = Instant::now();
        let latest = first + Duration::from_secs(1);
        for index in 0..20_000 {
            assert!(inbox.record("a", latest));
            assert!(
                !inbox.record(&format!("unknown-{index}"), first),
                "unknown session heartbeat was admitted"
            );
        }
        // Even an older supplied receipt cannot move a known timestamp backwards.
        assert!(inbox.record("a", first));
        let mut receipts = Vec::new();
        inbox.drain(|id, time| receipts.push((id.to_owned(), time)));
        assert_eq!(
            receipts,
            vec![("a".to_owned(), latest)],
            "heartbeat messages were not coalesced at their latest receipt time"
        );
        assert!(inbox.record("b", first));
        receipts.clear();
        inbox.drain(|id, time| receipts.push((id.to_owned(), time)));
        assert_eq!(
            receipts,
            vec![("b".to_owned(), first)],
            "drained heartbeat was replayed"
        );
        assert!(inbox.record("a", latest));
        inbox.publish(["b"].into_iter());
        assert!(!inbox.record("a", latest));
        receipts.clear();
        inbox.drain(|id, time| receipts.push((id.to_owned(), time)));
        assert!(receipts.is_empty(), "closed session heartbeat was replayed");
        inbox.publish(std::iter::empty());
        assert!(!inbox.record("b", latest));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn repeated_heartbeats_release_private_memory_in_a_fresh_process() {
        const CHILD: &str = "GREPPY_HEARTBEAT_MEMORY_CONTRACT_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "heartbeat_inbox::tests::repeated_heartbeats_release_private_memory_in_a_fresh_process",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "fresh-process heartbeat memory contract failed:\n{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            print!("{}", String::from_utf8_lossy(&result.stdout));
            return;
        }
        fn private_bytes() -> u64 {
            let rollup = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap();
            ["Private_Clean:", "Private_Dirty:", "Private_Hugetlb:"]
                .iter()
                .map(|key| {
                    rollup
                        .lines()
                        .find_map(|line| {
                            let mut fields = line.split_whitespace();
                            (fields.next() == Some(*key))
                                .then(|| fields.next().unwrap().parse::<u64>().unwrap() * 1024)
                        })
                        .unwrap_or(0)
                })
                .sum()
        }
        let inbox = HeartbeatInbox::default();
        let session = "s".repeat(16 * 1024);
        inbox.publish(std::iter::once(session.as_str()));
        let received = Instant::now();
        for _ in 0..32 {
            assert!(inbox.record(&session, received));
        }
        let before = private_bytes();
        for _ in 0..4096 {
            assert!(inbox.record(&session, received));
        }
        let growth = private_bytes().saturating_sub(before);
        println!("heartbeat_receipt_private_growth_bytes={growth}");
        assert!(
            growth < 8 * 1024 * 1024,
            "heartbeat receipts retained private memory: {growth} bytes"
        );
        let mut delivered = 0;
        inbox.drain(|id, time| {
            assert_eq!(id, session);
            assert_eq!(time, received);
            delivered += 1;
        });
        assert_eq!(delivered, 1, "repeated heartbeats were not coalesced");
    }
}
