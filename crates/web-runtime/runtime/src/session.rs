//! Run-owned session state machine (guide §6.3).

use crate::limits::SessionLimits;
use crate::policy::NetworkProfile;
use std::time::Instant;

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
    pub page_id: Option<String>,
    pub pages: u32,
    pub contexts: u32,
    pub requests: u64,
    pub network_bytes: u64,
    pub artifact_bytes: u64,
    pub download_bytes: u64,
    pub console_bytes: u64,
    pub peak_rss_bytes: u64,
    pub started: Instant,
    pub inflight_engine_request_id: Option<u64>,
    pub inflight_engine_method: Option<String>,
    pub discarded_engine_results: u64,
    pub persistent_profile: Option<String>,
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
            limits: SessionLimits::for_profile(profile.as_str()),
            page_id: None,
            pages: 0,
            contexts: 0,
            requests: 0,
            network_bytes: 0,
            artifact_bytes: 0,
            download_bytes: 0,
            console_bytes: 0,
            peak_rss_bytes: 0,
            started: Instant::now(),
            inflight_engine_request_id: None,
            inflight_engine_method: None,
            discarded_engine_results: 0,
            persistent_profile: None,
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
