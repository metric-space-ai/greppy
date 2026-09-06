//! Declarative runtime workflow input. No caller-provided JavaScript predicates.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const WORKFLOW_VERSION: u32 = 1;
pub const MAX_WORKFLOW_STEPS: usize = 64;
const MAX_TEXT_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Workflow {
    pub version: u32,
    pub session_id: String,
    pub tab_id: Option<String>,
    pub steps: Vec<WorkflowStep>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorkflowStep {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<WorkflowAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect: Option<WorkflowExpectation>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorkflowExpectation {
    pub condition: WorkflowCondition,
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorkflowCondition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub absent: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowSelector {
    Css {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nth: Option<i64>,
    },
    Xpath {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nth: Option<i64>,
    },
    Text {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nth: Option<i64>,
    },
    Role {
        role: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nth: Option<i64>,
    },
    Ref {
        value: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowAction {
    Goto {
        url: String,
    },
    Back,
    Forward,
    Reload,
    Click {
        selector: WorkflowSelector,
    },
    Fill {
        selector: WorkflowSelector,
        value: String,
    },
    Type {
        selector: WorkflowSelector,
        text: String,
    },
    Select {
        selector: WorkflowSelector,
        value: String,
    },
    Check {
        selector: WorkflowSelector,
    },
    Uncheck {
        selector: WorkflowSelector,
    },
    Hover {
        selector: WorkflowSelector,
    },
    Press {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        selector: Option<WorkflowSelector>,
        key: String,
    },
    Scroll {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        selector: Option<WorkflowSelector>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delta_y: Option<i64>,
    },
    Upload {
        selector: WorkflowSelector,
        files: Vec<String>,
    },
}

fn bounded(value: &str, label: &str, empty: bool) -> Result<(), String> {
    if (!empty && value.trim().is_empty()) || value.len() > MAX_TEXT_BYTES {
        Err(format!(
            "{label} must be {}at most {MAX_TEXT_BYTES} bytes",
            if empty { "" } else { "nonempty and " }
        ))
    } else {
        Ok(())
    }
}

impl WorkflowSelector {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Ref { value: 0 } => Err("reference numbering starts at 1".into()),
            Self::Ref { .. } => Ok(()),
            Self::Css { value, .. } | Self::Xpath { value, .. } | Self::Text { value, .. } => {
                bounded(value, "selector", false)
            }
            Self::Role { role, name, .. } => {
                bounded(role, "role", false)?;
                if let Some(name) = name {
                    bounded(name, "name", true)?;
                }
                Ok(())
            }
        }
    }
}

impl WorkflowAction {
    pub fn operation(&self) -> &'static str {
        match self {
            Self::Goto { .. } => "web.goto",
            Self::Back => "web.back",
            Self::Forward => "web.forward",
            Self::Reload => "web.reload",
            Self::Click { .. } => "web.click",
            Self::Fill { .. } => "web.fill",
            Self::Type { .. } => "web.type",
            Self::Select { .. } => "web.select",
            Self::Check { .. } => "web.check",
            Self::Uncheck { .. } => "web.uncheck",
            Self::Hover { .. } => "web.hover",
            Self::Press { .. } => "web.press",
            Self::Scroll { .. } => "web.scroll",
            Self::Upload { .. } => "web.upload",
        }
    }

    pub fn selector(&self) -> Option<&WorkflowSelector> {
        match self {
            Self::Click { selector }
            | Self::Fill { selector, .. }
            | Self::Type { selector, .. }
            | Self::Select { selector, .. }
            | Self::Check { selector }
            | Self::Uncheck { selector }
            | Self::Hover { selector }
            | Self::Upload { selector, .. } => Some(selector),
            Self::Press { selector, .. } | Self::Scroll { selector, .. } => selector.as_ref(),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(selector) = self.selector() {
            selector.validate()?;
        }
        match self {
            Self::Goto { url } => bounded(url, "url", false),
            Self::Fill { value, .. } | Self::Select { value, .. } => bounded(value, "value", true),
            Self::Type { text, .. } => bounded(text, "text", true),
            Self::Press { key, .. } => bounded(key, "key", false),
            Self::Scroll { selector, delta_y } if selector.is_some() == delta_y.is_some() => {
                Err("scroll needs exactly one of selector or delta_y".into())
            }
            Self::Upload { files, .. } => {
                if files.is_empty() || files.len() > 64 {
                    return Err("upload needs 1 to 64 staged files".into());
                }
                for file in files {
                    bounded(file, "staged file", false)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn payload(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("workflow action is serializable");
        value
            .as_object_mut()
            .expect("tagged action")
            .remove("operation");
        value
    }
}

impl WorkflowCondition {
    pub fn validate(&self) -> Result<(), String> {
        if self.query.is_none() && self.url.is_none() && self.title.is_none() {
            return Err("condition needs query, url or title".into());
        }
        for value in [&self.query, &self.url, &self.title].into_iter().flatten() {
            bounded(value, "condition", false)?;
        }
        self.reference()?;
        Ok(())
    }

    pub fn reference(&self) -> Result<Option<u64>, String> {
        let Some(query) = self
            .query
            .as_deref()
            .map(str::trim)
            .filter(|q| q.starts_with('@'))
        else {
            return Ok(None);
        };
        let digits = &query[1..];
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err("condition reference must be @ followed by digits".into());
        }
        let value = digits
            .parse::<u64>()
            .map_err(|_| "condition reference is out of range")?;
        if value == 0 {
            return Err("reference numbering starts at 1".into());
        }
        Ok(Some(value))
    }

    /// Engine-owned interpreter consumes JSON data, never caller JavaScript.
    pub fn source(&self) -> String {
        format!("({})({}, typeof __greppyConditionNodes === 'undefined' ? null : __greppyConditionNodes)",
                include_str!("workflow-condition.js"), serde_json::to_string(self).expect("condition is serializable"))
    }
}

impl Workflow {
    /// Shape/size validation. Engine syntax preflight MUST also pass before mutation.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != WORKFLOW_VERSION {
            return Err("unsupported workflow version".into());
        }
        bounded(&self.session_id, "session_id", false)?;
        if let Some(tab) = &self.tab_id {
            bounded(tab, "tab_id", false)?;
        }
        if self.steps.is_empty() || self.steps.len() > MAX_WORKFLOW_STEPS {
            return Err("workflow needs 1 to 64 steps".into());
        }
        for (index, step) in self.steps.iter().enumerate() {
            let validate = || -> Result<(), String> {
                if step.action.is_none() && step.expect.is_none() {
                    return Err("step needs action or expectation".into());
                }
                if let Some(action) = &step.action {
                    action.validate()?;
                }
                if let Some(expect) = &step.expect {
                    expect.condition.validate()?;
                    if expect.timeout_ms == 0 || expect.timeout_ms > 300_000 {
                        return Err("expectation timeout must be 1 to 300000 ms".into());
                    }
                }
                Ok(())
            };
            validate().map_err(|error| format!("step {}: {error}", index + 1))?;
        }
        if serde_json::to_vec(self).map_err(|e| e.to_string())?.len() > crate::MAX_FRAME_BYTES / 2 {
            return Err("workflow exceeds half the protocol frame budget".into());
        }
        Ok(())
    }

    pub fn preflight_source(&self) -> String {
        let inputs: Vec<Value> = self
            .steps
            .iter()
            .map(|step| {
                json!({
                    "selector": step.action.as_ref().and_then(WorkflowAction::selector),
                    "condition": step.expect.as_ref().map(|expect| &expect.condition),
                })
            })
            .collect();
        format!(
            "({})({}, {})",
            include_str!("workflow-preflight.js"),
            serde_json::to_string(&inputs).expect("inputs are serializable"),
            include_str!("workflow-condition.js")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn valid() -> Workflow {
        Workflow {
            version: 1,
            session_id: "s".into(),
            tab_id: Some("p".into()),
            steps: vec![WorkflowStep {
                action: Some(WorkflowAction::Click {
                    selector: WorkflowSelector::Ref { value: 7 },
                }),
                expect: Some(WorkflowExpectation {
                    condition: WorkflowCondition {
                        query: Some("css=#done".into()),
                        ..WorkflowCondition::default()
                    },
                    timeout_ms: 500,
                }),
            }],
        }
    }

    #[test]
    fn later_invalid_step_is_rejected_before_execution_can_begin() {
        let mut workflow = valid();
        workflow.steps.push(WorkflowStep {
            action: Some(WorkflowAction::Click {
                selector: WorkflowSelector::Ref { value: 0 },
            }),
            expect: None,
        });
        assert!(workflow.validate().unwrap_err().starts_with("step 2:"));
        workflow.steps[1] = WorkflowStep {
            action: None,
            expect: None,
        };
        assert!(workflow.validate().is_err());
    }

    #[test]
    fn arbitrary_predicates_unknown_steps_and_ambiguous_shapes_are_rejected() {
        let mut encoded = serde_json::to_value(valid()).unwrap();
        encoded["steps"][0]["expect"]["condition"]["source"] =
            json!("document.body.remove(); true");
        assert!(serde_json::from_value::<Workflow>(encoded).is_err());
        assert!(serde_json::from_value::<WorkflowAction>(
            json!({"operation":"evaluate","source":"true"})
        )
        .is_err());
        assert!(serde_json::from_value::<WorkflowSelector>(
            json!({"type":"ref","value":1,"nth":0})
        )
        .is_err());
        let mut workflow = valid();
        workflow.steps[0].action = Some(WorkflowAction::Scroll {
            selector: Some(WorkflowSelector::Ref { value: 1 }),
            delta_y: Some(100),
        });
        assert!(workflow.validate().is_err());
    }

    #[test]
    fn bounds_and_invalid_absence_refs_remain_failures() {
        for query in ["@", "@0", "@-1", "@1x", "@18446744073709551616"] {
            let condition = WorkflowCondition {
                query: Some(query.into()),
                absent: true,
                ..WorkflowCondition::default()
            };
            assert!(condition.validate().is_err(), "{query}");
        }
        let mut workflow = valid();
        workflow.steps[0].expect.as_mut().unwrap().timeout_ms = 0;
        assert!(workflow.validate().is_err());
        workflow.steps[0].expect.as_mut().unwrap().timeout_ms = 300001;
        assert!(workflow.validate().is_err());
        let mut workflow = valid();
        workflow.steps = vec![workflow.steps[0].clone(); MAX_WORKFLOW_STEPS + 1];
        assert!(workflow.validate().is_err());
    }

    #[test]
    fn preflight_contains_selectors_and_conditions_without_action_values() {
        let mut workflow = valid();
        workflow.steps[0].action = Some(WorkflowAction::Fill {
            selector: WorkflowSelector::Css {
                value: "#password".into(),
                nth: None,
            },
            value: "secret-not-in-preflight".into(),
        });
        workflow.validate().unwrap();
        let source = workflow.preflight_source();
        assert!(source.contains("#password"));
        assert!(source.contains("#done"));
        assert!(!source.contains("secret-not-in-preflight"));
        let action = workflow.steps[0].action.as_ref().unwrap();
        assert_eq!(action.operation(), "web.fill");
        assert_eq!(action.payload()["value"], "secret-not-in-preflight");
        assert!(action.payload().get("operation").is_none());
    }
}
