//! Run-owned session state machine (guide §6.3).

use crate::limits::SessionLimits;
use crate::policy::NetworkProfile;
use greppy_web_client::{
    ActionOperation, ActionOutcome, ActionTicket, ObservationContext, ObservationContextError,
    ObservationContextState,
};
use std::time::Instant;

#[derive(Debug)]
pub struct LocatorSnapshot {
    pub token: String,
    pub page_id: String,
    pub ref_count: u64,
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
    /// Every page this session holds open, oldest first. `page_id` names the
    /// active one; a session with several tabs keeps them all here so a
    /// caller can switch without losing the others.
    pub tabs: Vec<String>,
    /// Locator recipes created by the most recent `web.observe`. Recipes are
    /// deliberately session-, page-, and document-bound; the content worker
    /// independently verifies the document token before resolving a ref.
    pub locator_snapshot: Option<LocatorSnapshot>,
    pub pages: u32,
    pub contexts: u32,
    pub requests: u64,
    pub network_bytes: u64,
    pub artifact_bytes: u64,
    pub download_bytes: u64,
    pub console_bytes: u64,
    pub peak_rss_bytes: u64,
    /// CPU baselines so the session budget measures THIS session's work, not
    /// the content/controller process lifetime (finding 039: the verb path
    /// charged every session for all CPU ever burned, so a long-lived daemon
    /// went permanently unusable after ~30s of cumulative work). Paired with
    /// the pid so a worker respawn resets the baseline instead of producing
    /// a bogus negative delta.
    pub content_cpu_baseline: Option<(u32, u64)>,
    pub controller_cpu_baseline: Option<(u32, u64)>,
    pub started: Instant,
    pub inflight_engine_request_id: Option<u64>,
    pub inflight_engine_method: Option<String>,
    pub discarded_engine_results: u64,
    pub persistent_profile: Option<String>,
    pub owner: Option<String>,
    observation_context: ObservationContextState,
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
            tabs: Vec::new(),
            locator_snapshot: None,
            pages: 0,
            contexts: 0,
            requests: 0,
            network_bytes: 0,
            artifact_bytes: 0,
            download_bytes: 0,
            console_bytes: 0,
            peak_rss_bytes: 0,
            content_cpu_baseline: None,
            controller_cpu_baseline: None,
            started: Instant::now(),
            inflight_engine_request_id: None,
            inflight_engine_method: None,
            discarded_engine_results: 0,
            persistent_profile: None,
            owner: None,
            observation_context: ObservationContextState::default(),
        }
    }

    /// Goal-only initialization; profiles, limits, and page/snapshot identity are unchanged.
    pub fn new_with_goal(
        id: impl Into<String>,
        run_id: impl Into<String>,
        profile: NetworkProfile,
        goal: Option<String>,
    ) -> Result<Self, ObservationContextError> {
        let context = ObservationContextState::with_goal(goal)?;
        let mut session = Self::new(id, run_id, profile);
        session.observation_context = context;
        Ok(session)
    }

    pub fn observation_context(&self) -> ObservationContext {
        self.observation_context.snapshot()
    }

    pub fn set_observation_goal(
        &mut self,
        goal: Option<String>,
        expected_goal_version: u64,
    ) -> Result<ObservationContext, ObservationContextError> {
        self.observation_context
            .set_goal(goal, expected_goal_version)
    }

    pub fn begin_observation_action(
        &mut self,
        operation: ActionOperation,
        request_id: &str,
        page_id: Option<&str>,
    ) -> Result<ActionTicket, ObservationContextError> {
        self.observation_context
            .begin_action(operation, request_id, page_id)
    }

    pub fn complete_observation_action(
        &mut self,
        ticket: &ActionTicket,
        outcome: ActionOutcome,
    ) -> Result<(), ObservationContextError> {
        self.observation_context.complete_action(ticket, outcome)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_changes_are_isolated_and_do_not_change_session_lifecycle_or_snapshot() {
        let mut a = Session::new_with_goal(
            "wrs_a",
            "run",
            NetworkProfile::Project,
            Some("Choose item".into()),
        )
        .unwrap();
        let b = Session::new("wrs_b", "run", NetworkProfile::Project);
        a.page_id = Some("page_a".into());
        a.locator_snapshot = Some(LocatorSnapshot {
            token: "snapshot_a".into(),
            page_id: "page_a".into(),
            ref_count: 2,
        });
        let heartbeat = a.last_heartbeat;
        assert_eq!(
            a.set_observation_goal(Some("Other".into()), 0),
            Err(ObservationContextError::GoalVersionConflict {
                current_goal_version: 1
            })
        );
        let cleared = a.set_observation_goal(None, 1).unwrap();
        assert!(!cleared.has_explicit_goal());
        assert_eq!(cleared.goal_version, 2);
        assert_eq!(a.state, SessionState::Creating);
        assert_eq!(a.last_heartbeat, heartbeat);
        assert_eq!(a.page_id.as_deref(), Some("page_a"));
        assert_eq!(a.locator_snapshot.as_ref().unwrap().token, "snapshot_a");
        assert_eq!(b.observation_context().goal_version, 0);
        assert!(!b.observation_context().has_explicit_goal());
    }

    #[test]
    fn action_tickets_do_not_cross_sessions_and_capture_dispatch_goal() {
        let mut a =
            Session::new_with_goal("wrs_a", "run", NetworkProfile::Project, Some("Edit".into()))
                .unwrap();
        let mut b = Session::new("wrs_b", "run", NetworkProfile::Project);
        let ticket = a
            .begin_observation_action(ActionOperation::Fill, "wrq_a", Some("page_a"))
            .unwrap();
        assert!(b
            .complete_observation_action(&ticket, ActionOutcome::Success)
            .is_err());
        a.set_observation_goal(Some("Save".into()), 1).unwrap();
        a.complete_observation_action(&ticket, ActionOutcome::Failure)
            .unwrap();
        let receipt = a.observation_context().last_action.unwrap();
        assert_eq!(receipt.goal_version, 1);
        assert_eq!(receipt.outcome, ActionOutcome::Failure);
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
}
