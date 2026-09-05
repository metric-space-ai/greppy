//! Versioned, value-free task context for optional observation ranking.
//! A successful action receipt means tool dispatch completed, not task success.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const MAX_GOAL_BYTES: usize = 8192;
pub const MAX_IDENTITY_BYTES: usize = 128;
pub const MAX_PENDING_ACTIONS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ObservationContextSchema {
    #[serde(rename = "greppy.web.observation-context.v1")]
    V1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ActionOperation {
    #[serde(rename = "web.goto")]
    Navigate,
    #[serde(rename = "web.back")]
    Back,
    #[serde(rename = "web.forward")]
    Forward,
    #[serde(rename = "web.reload")]
    Reload,
    #[serde(rename = "web.click")]
    Click,
    #[serde(rename = "web.fill")]
    Fill,
    #[serde(rename = "web.type")]
    Type,
    #[serde(rename = "web.clear")]
    Clear,
    #[serde(rename = "web.select")]
    Select,
    #[serde(rename = "web.check")]
    Check,
    #[serde(rename = "web.uncheck")]
    Uncheck,
    #[serde(rename = "web.press")]
    Press,
    #[serde(rename = "web.hover")]
    Hover,
    #[serde(rename = "web.scroll")]
    Scroll,
    #[serde(rename = "web.upload")]
    Upload,
    #[serde(rename = "web.tab.new")]
    NewTab,
    #[serde(rename = "web.tab.switch")]
    SwitchTab,
    #[serde(rename = "web.tab.close")]
    CloseTab,
}

impl ActionOperation {
    /// Readers and unknown operations cannot replace the last completed action.
    /// Script/chain integration must identify actual actions at completion sites.
    pub fn from_rpc_operation(operation: &str) -> Option<Self> {
        serde_json::from_value(serde_json::Value::String(operation.into())).ok()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionOutcome {
    Success,
    Failure,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionReceipt {
    pub action_seq: u64,
    pub request_id: String,
    pub operation: ActionOperation,
    /// Opaque runtime identity at dispatch; never a URL, selector, or user value.
    pub page_id: Option<String>,
    pub goal_version: u64,
    pub outcome: ActionOutcome,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationContext {
    pub schema: ObservationContextSchema,
    pub goal: Option<String>,
    pub goal_version: u64,
    pub last_action: Option<ActionReceipt>,
}

impl ObservationContext {
    pub fn has_explicit_goal(&self) -> bool {
        self.goal.is_some()
    }
}

/// Missing goal is an error; only an explicit JSON null requests clearing it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetGoalRequest {
    pub session_id: String,
    #[serde(deserialize_with = "explicit_goal")]
    pub goal: Option<String>,
    pub expected_goal_version: u64,
}

fn explicit_goal<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    Option::<String>::deserialize(d)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum ObservationContextError {
    GoalVersionConflict { current_goal_version: u64 },
    GoalEmpty,
    GoalTooLong,
    InvalidIdentity,
    VersionExhausted,
    SequenceExhausted,
    TooManyPendingActions,
    UnknownActionTicket,
}

pub fn validate_goal(goal: &str) -> Result<(), ObservationContextError> {
    if goal.trim().is_empty() {
        return Err(ObservationContextError::GoalEmpty);
    }
    if goal.len() > MAX_GOAL_BYTES {
        return Err(ObservationContextError::GoalTooLong);
    }
    Ok(())
}

fn validate_identity(id: &str) -> Result<(), ObservationContextError> {
    if id.is_empty()
        || id.len() > MAX_IDENTITY_BYTES
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-:.".contains(&b))
    {
        return Err(ObservationContextError::InvalidIdentity);
    }
    Ok(())
}

/// Session-bound token. Callers cannot construct or retarget it.
#[derive(Debug)]
pub struct ActionTicket {
    owner: Arc<()>,
    sequence: u64,
}

#[derive(Debug)]
struct PendingAction {
    request_id: String,
    operation: ActionOperation,
    page_id: Option<String>,
    goal_version: u64,
}

/// Mutable state is separate from the wire snapshot and is not deserializable.
#[derive(Debug)]
pub struct ObservationContextState {
    owner: Arc<()>,
    goal: Option<String>,
    goal_version: u64,
    action_seq: u64,
    pending: BTreeMap<u64, PendingAction>,
    last_action: Option<ActionReceipt>,
}

impl Default for ObservationContextState {
    fn default() -> Self {
        Self {
            owner: Arc::new(()),
            goal: None,
            goal_version: 0,
            action_seq: 0,
            pending: BTreeMap::new(),
            last_action: None,
        }
    }
}

impl ObservationContextState {
    pub fn with_goal(goal: Option<String>) -> Result<Self, ObservationContextError> {
        let mut state = Self::default();
        if goal.is_some() {
            state.set_goal(goal, 0)?;
        }
        Ok(state)
    }

    pub fn snapshot(&self) -> ObservationContext {
        ObservationContext {
            schema: ObservationContextSchema::V1,
            goal: self.goal.clone(),
            goal_version: self.goal_version,
            last_action: self.last_action.clone(),
        }
    }

    /// Every accepted explicit set/clear advances the version, including a repeated goal.
    /// A CAS conflict or invalid goal changes no state and leaves pending actions intact.
    pub fn set_goal(
        &mut self,
        goal: Option<String>,
        expected: u64,
    ) -> Result<ObservationContext, ObservationContextError> {
        if expected != self.goal_version {
            return Err(ObservationContextError::GoalVersionConflict {
                current_goal_version: self.goal_version,
            });
        }
        if let Some(value) = &goal {
            validate_goal(value)?;
        }
        let next = self
            .goal_version
            .checked_add(1)
            .ok_or(ObservationContextError::VersionExhausted)?;
        self.goal = goal;
        self.goal_version = next;
        Ok(self.snapshot())
    }

    /// Identities must originate from the runtime, not from action arguments.
    /// No value, text, selector, token, upload body, or arbitrary operation string is accepted.
    pub fn begin_action(
        &mut self,
        operation: ActionOperation,
        request_id: &str,
        page_id: Option<&str>,
    ) -> Result<ActionTicket, ObservationContextError> {
        validate_identity(request_id)?;
        if !request_id.starts_with("wrq_") {
            return Err(ObservationContextError::InvalidIdentity);
        }
        if let Some(id) = page_id {
            validate_identity(id)?;
        }
        if self.pending.len() >= MAX_PENDING_ACTIONS {
            return Err(ObservationContextError::TooManyPendingActions);
        }
        let sequence = self
            .action_seq
            .checked_add(1)
            .ok_or(ObservationContextError::SequenceExhausted)?;
        self.pending.insert(
            sequence,
            PendingAction {
                request_id: request_id.into(),
                operation,
                page_id: page_id.map(str::to_owned),
                goal_version: self.goal_version,
            },
        );
        self.action_seq = sequence;
        Ok(ActionTicket {
            owner: Arc::clone(&self.owner),
            sequence,
        })
    }

    /// Completion order defines "last completed", while action_seq records dispatch order.
    /// Call for failures as well as successes. A ticket cannot complete twice or cross sessions.
    pub fn complete_action(
        &mut self,
        ticket: &ActionTicket,
        outcome: ActionOutcome,
    ) -> Result<(), ObservationContextError> {
        if !Arc::ptr_eq(&self.owner, &ticket.owner) {
            return Err(ObservationContextError::UnknownActionTicket);
        }
        let pending = self
            .pending
            .remove(&ticket.sequence)
            .ok_or(ObservationContextError::UnknownActionTicket)?;
        self.last_action = Some(ActionReceipt {
            action_seq: ticket.sequence,
            request_id: pending.request_id,
            operation: pending.operation,
            page_id: pending.page_id,
            goal_version: pending.goal_version,
            outcome,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_goal_cas_clear_and_session_isolation() {
        let mut a = ObservationContextState::with_goal(Some("Choose item".into())).unwrap();
        let b = ObservationContextState::default();
        assert_eq!(a.snapshot().goal_version, 1);
        assert!(!b.snapshot().has_explicit_goal());
        let before = a.snapshot();
        assert_eq!(
            a.set_goal(Some("Other".into()), 0),
            Err(ObservationContextError::GoalVersionConflict {
                current_goal_version: 1
            })
        );
        assert_eq!(a.snapshot(), before);
        assert_eq!(a.set_goal(None, 1).unwrap().goal_version, 2);
        assert!(!a.snapshot().has_explicit_goal());
        assert_eq!(a.set_goal(Some("New".into()), 2).unwrap().goal_version, 3);
        assert_eq!(b.snapshot().goal_version, 0);
    }

    #[test]
    fn invalid_goal_and_exhaustion_do_not_mutate() {
        let mut state = ObservationContextState::default();
        assert_eq!(
            state.set_goal(Some(" ".into()), 0),
            Err(ObservationContextError::GoalEmpty)
        );
        assert_eq!(
            state.set_goal(Some("ä".repeat(MAX_GOAL_BYTES)), 0),
            Err(ObservationContextError::GoalTooLong)
        );
        assert_eq!(state.snapshot().goal_version, 0);
        state.goal_version = u64::MAX;
        assert_eq!(
            state.set_goal(None, u64::MAX),
            Err(ObservationContextError::VersionExhausted)
        );
    }

    #[test]
    fn action_uses_dispatch_goal_and_readers_do_not_replace_it() {
        let mut state = ObservationContextState::with_goal(Some("Edit quantity".into())).unwrap();
        let ticket = state
            .begin_action(ActionOperation::Fill, "wrq_1", Some("page_1"))
            .unwrap();
        assert!(state.snapshot().last_action.is_none());
        state.set_goal(Some("Save".into()), 1).unwrap();
        state
            .complete_action(&ticket, ActionOutcome::Failure)
            .unwrap();
        let snapshot = state.snapshot();
        assert_eq!(snapshot.last_action.as_ref().unwrap().goal_version, 1);
        assert_eq!(
            snapshot.last_action.as_ref().unwrap().outcome,
            ActionOutcome::Failure
        );
        for reader in [
            "web.observe",
            "web.inspect",
            "web.status",
            "web.handshake",
            "web.session.set_goal",
        ] {
            assert!(ActionOperation::from_rpc_operation(reader).is_none());
        }
        assert_eq!(state.snapshot(), snapshot);
    }

    #[test]
    fn tickets_are_bounded_fenced_and_ordered_by_completion() {
        let mut a = ObservationContextState::default();
        let mut b = ObservationContextState::default();
        let first = a
            .begin_action(ActionOperation::Click, "wrq_1", None)
            .unwrap();
        assert_eq!(
            b.complete_action(&first, ActionOutcome::Success),
            Err(ObservationContextError::UnknownActionTicket)
        );
        let second = a
            .begin_action(ActionOperation::Click, "wrq_2", None)
            .unwrap();
        a.complete_action(&second, ActionOutcome::Success).unwrap();
        a.complete_action(&first, ActionOutcome::Failure).unwrap();
        assert_eq!(a.snapshot().last_action.unwrap().action_seq, 1);
        assert_eq!(
            a.complete_action(&first, ActionOutcome::Success),
            Err(ObservationContextError::UnknownActionTicket)
        );
        for _ in 0..MAX_PENDING_ACTIONS {
            a.begin_action(ActionOperation::Click, "wrq_3", None)
                .unwrap();
        }
        assert_eq!(
            a.begin_action(ActionOperation::Click, "wrq_4", None)
                .unwrap_err(),
            ObservationContextError::TooManyPendingActions
        );
    }

    #[test]
    fn receipt_cannot_accept_input_values_and_ids_are_bounded() {
        let mut state = ObservationContextState::default();
        assert!(state
            .begin_action(
                ActionOperation::Fill,
                "wrq_1",
                Some("https://private/?token=x")
            )
            .is_err());
        let ticket = state
            .begin_action(ActionOperation::Fill, "wrq_1", Some("page_1"))
            .unwrap();
        state
            .complete_action(&ticket, ActionOutcome::Success)
            .unwrap();
        let receipt = state.snapshot().last_action.unwrap();
        let mut value = serde_json::to_value(&receipt).unwrap();
        value["value"] = serde_json::json!("private password");
        assert!(serde_json::from_value::<ActionReceipt>(value).is_err());
        assert!(!serde_json::to_string(&receipt)
            .unwrap()
            .contains("password"));
    }

    #[test]
    fn clearing_goal_requires_explicit_null_and_snapshot_schema_is_versioned() {
        assert!(serde_json::from_value::<SetGoalRequest>(serde_json::json!({
            "session_id":"wrs_1","expected_goal_version":0
        }))
        .is_err());
        let request: SetGoalRequest = serde_json::from_value(serde_json::json!({
            "session_id":"wrs_1","expected_goal_version":0,"goal":null
        }))
        .unwrap();
        assert!(request.goal.is_none());
        let mut value =
            serde_json::to_value(ObservationContextState::default().snapshot()).unwrap();
        value["schema"] = serde_json::json!("unknown");
        assert!(serde_json::from_value::<ObservationContext>(value).is_err());
    }
}
