//! Run-owned session state machine (guide §6.3).

use crate::limits::SessionLimits;
use crate::policy::NetworkProfile;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct LocatorSnapshot {
    pub token: String,
    pub page_id: String,
    pub ref_ceiling: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionState {
    Creating,
    Ready,
    Busy,
    Failed,
    Closing,
    Closed,
}

#[derive(Debug)]
pub struct Session {
    pub id: String,
    pub run_id: String,
    pub state: SessionState,
    pub operation_id: Option<String>,
    pub last_heartbeat: Instant,
    pub profile: NetworkProfile,
    pub limits: SessionLimits,
    /// The page every operation without an explicit target acts on.
    pub page_id: Option<String>,
    /// Content-worker browser context that owns this session's storage.
    pub engine_context_id: Option<String>,
    /// Every page this session holds open, oldest first. `page_id` names the
    /// active one; a session with several tabs keeps them all here so a
    /// caller can switch without losing the others.
    pub tabs: Vec<String>,
    /// Tabs snapshotted to disk so the content worker can drop their documents.
    pub swapped_tabs: Vec<String>,
    /// Old tab id to the page id created when that snapshot was restored.
    pub tab_aliases: HashMap<String, String>,
    /// Document-bound recipes per page. Observing one tab must not replace
    /// another tab's scope. The worker independently checks actual node identity.
    pub locator_snapshots: HashMap<String, LocatorSnapshot>,
    pub pages: u32,
    pub contexts: u32,
    pub requests: u64,
    pub network_bytes: u64,
    pub artifact_bytes: u64,
    pub download_bytes: u64,
    pub console_bytes: u64,
    pub peak_rss_bytes: u64,
    /// CPU charged to this session's serialized operation intervals. These
    /// counters survive worker respawns; the daemon adds only the process CPU
    /// delta observed around each operation, so one session cannot inherit
    /// another session's shared-worker work. The limit check uses the
    /// navigation counters below, which reset on each navigation.
    pub content_cpu_used_ns: u64,
    pub controller_cpu_used_ns: u64,
    /// Content and controller CPU since the last navigation. A heavy page
    /// must not exhaust the next page's budget.
    pub navigation_content_cpu_ns: u64,
    pub navigation_controller_cpu_ns: u64,
    /// Response bytes since the last navigation. `network_bytes` keeps the
    /// session total for metrics; the limit uses this counter.
    pub navigation_network_bytes: u64,
    /// Response ids already charged, so a page that retains its log does not
    /// bill those bytes again on the next navigation.
    seen_response_ids: HashSet<String>,
    pub started: Instant,
    pub inflight_engine_request_id: Option<u64>,
    pub inflight_engine_method: Option<String>,
    pub discarded_engine_results: u64,
    pub persistent_profile: Option<String>,
    pub owner: Option<String>,
    pub trace: Option<crate::playwright_trace::TraceRecorder>,
}

impl Session {
    pub fn new(id: impl Into<String>, run_id: impl Into<String>, profile: NetworkProfile) -> Self {
        Self {
            id: id.into(),
            run_id: run_id.into(),
            state: SessionState::Creating,
            operation_id: None,
            last_heartbeat: Instant::now(),
            profile,
            limits: {
                let mut limits = SessionLimits::for_profile(profile.as_str());
                limits.apply_host_shape();
                limits
            },
            page_id: None,
            engine_context_id: None,
            tabs: Vec::new(),
            swapped_tabs: Vec::new(),
            tab_aliases: HashMap::new(),
            locator_snapshots: HashMap::new(),
            pages: 0,
            contexts: 0,
            requests: 0,
            network_bytes: 0,
            artifact_bytes: 0,
            download_bytes: 0,
            console_bytes: 0,
            peak_rss_bytes: 0,
            content_cpu_used_ns: 0,
            controller_cpu_used_ns: 0,
            navigation_content_cpu_ns: 0,
            navigation_controller_cpu_ns: 0,
            navigation_network_bytes: 0,
            seen_response_ids: HashSet::new(),
            started: Instant::now(),
            inflight_engine_request_id: None,
            inflight_engine_method: None,
            discarded_engine_results: 0,
            persistent_profile: None,
            owner: None,
            trace: None,
        }
    }

    /// Drop the CPU and network bytes charged to the page being replaced.
    /// The session totals stay, so a worker restart still cannot underflow them.
    pub fn begin_navigation(&mut self) {
        self.navigation_content_cpu_ns = 0;
        self.navigation_controller_cpu_ns = 0;
        self.navigation_network_bytes = 0;
    }

    pub(crate) fn charge_new_responses(&mut self, rows: &[serde_json::Value]) -> u64 {
        let mut current_ids = HashSet::new();
        let mut added = 0u64;
        for row in rows {
            let Some(id) = row
                .get("requestId")
                .and_then(|value| value.as_str())
                .filter(|id| !id.is_empty())
            else {
                continue;
            };
            current_ids.insert(id.to_owned());
            if self.seen_response_ids.insert(id.to_owned()) {
                added = added.saturating_add(
                    row.get("byteLength")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0),
                );
            }
        }
        self.seen_response_ids.retain(|id| current_ids.contains(id));
        self.navigation_network_bytes = self.navigation_network_bytes.saturating_add(added);
        self.network_bytes = self.network_bytes.saturating_add(added);
        added
    }

    pub fn transition(&mut self, next: SessionState) -> Result<(), String> {
        let allowed = matches!(
            (self.state, next),
            (SessionState::Creating, SessionState::Ready)
                | (SessionState::Ready, SessionState::Busy)
                | (SessionState::Busy, SessionState::Ready)
                | (SessionState::Busy, SessionState::Failed)
                | (SessionState::Ready, SessionState::Failed)
                | (SessionState::Creating, SessionState::Failed)
                | (SessionState::Failed, SessionState::Ready)
                | (SessionState::Failed, SessionState::Busy)
                | (SessionState::Ready, SessionState::Closing)
                | (SessionState::Busy, SessionState::Closing)
                | (SessionState::Failed, SessionState::Closing)
                | (SessionState::Closing, SessionState::Closed)
        );
        if !allowed {
            return Err(format!(
                "illegal session transition {:?} -> {:?}",
                self.state, next
            ));
        }
        if next != SessionState::Busy {
            self.operation_id = None;
            self.inflight_engine_request_id = None;
            self.inflight_engine_method = None;
        }
        self.state = next;
        self.last_heartbeat = Instant::now();
        Ok(())
    }

    pub fn begin_operation(&mut self, operation_id: impl Into<String>) -> Result<(), String> {
        self.transition(SessionState::Busy)?;
        self.operation_id = Some(operation_id.into());
        Ok(())
    }

    pub fn canonical_tab(&self, tab: &str) -> String {
        let mut id = tab.to_string();
        for _ in 0..8 {
            match self.tab_aliases.get(&id) {
                Some(next) if next != &id => id = next.clone(),
                _ => break,
            }
        }
        id
    }

    pub fn owns_tab(&self, tab: &str) -> bool {
        let id = self.canonical_tab(tab);
        self.page_id.as_deref() == Some(id.as_str())
            || self.tabs.iter().any(|page| page == &id)
            || self
                .swapped_tabs
                .iter()
                .any(|page| page == &id || page == tab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapped_tab_stays_addressable_until_restore() {
        let mut session = Session::new("wrs_1", "run", crate::policy::NetworkProfile::Project);
        session.swapped_tabs.push("tab-old".into());
        assert!(session.owns_tab("tab-old"));
        session
            .tab_aliases
            .insert("tab-old".into(), "tab-new".into());
        session.tabs.push("tab-new".into());
        session.swapped_tabs.clear();
        assert_eq!(session.canonical_tab("tab-old"), "tab-new");
        assert!(session.owns_tab("tab-old"));
    }

    #[test]
    fn ready_busy_ready_close() {
        let mut session = Session::new("wrs_1", "run", crate::policy::NetworkProfile::Project);
        session.transition(SessionState::Ready).unwrap();
        session.begin_operation("wrq_1").unwrap();
        assert_eq!(session.operation_id.as_deref(), Some("wrq_1"));
        session.transition(SessionState::Ready).unwrap();
        assert!(session.operation_id.is_none());
        session.transition(SessionState::Closing).unwrap();
        session.transition(SessionState::Closed).unwrap();
    }

    #[test]
    fn rejects_closed_to_ready() {
        let mut session = Session::new("wrs_1", "run", crate::policy::NetworkProfile::Project);
        session.transition(SessionState::Ready).unwrap();
        session.transition(SessionState::Closing).unwrap();
        session.transition(SessionState::Closed).unwrap();
        assert!(session.transition(SessionState::Ready).is_err());
    }

    #[test]
    fn failed_session_accepts_the_next_operation() {
        let mut session = Session::new("wrs_1", "run", crate::policy::NetworkProfile::Project);
        session.transition(SessionState::Ready).unwrap();
        session.begin_operation("wrq_1").unwrap();
        session.transition(SessionState::Failed).unwrap();
        session.begin_operation("wrq_2").unwrap();
        assert_eq!(session.state, SessionState::Busy);
        assert_eq!(session.operation_id.as_deref(), Some("wrq_2"));
        session.transition(SessionState::Ready).unwrap();
        assert_eq!(session.state, SessionState::Ready);
    }

    #[test]
    fn default_session_accepts_multiple_operations_after_two_minutes() {
        let mut session = Session::new("wrs_old", "run", crate::policy::NetworkProfile::Project);
        session.started = Instant::now() - std::time::Duration::from_secs(121);
        session.transition(SessionState::Ready).unwrap();

        for operation in ["wrq_late_1", "wrq_late_2"] {
            session
                .limits
                .check_wall_time(session.started.elapsed())
                .unwrap();
            session.begin_operation(operation).unwrap();
            session.transition(SessionState::Ready).unwrap();
        }
    }
}
