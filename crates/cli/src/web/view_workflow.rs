//! Presentation only: known workflow envelopes, never inferred task success.
use serde_json::{Map, Value};

fn known(object: &Map<String, Value>, fields: &[&str]) -> bool {
    object.keys().all(|key| fields.contains(&key.as_str()))
}

fn detail(value: &Value, workflow: &Value, status: &str) -> Option<Value> {
    let mut result = value.as_object()?.clone();
    // A future detail schema may give even familiar keys different meaning.
    // Keep that entire detail, rather than applying today's default omissions.
    if !known(
        &result,
        &[
            "session_id",
            "tab_id",
            "untrusted_content_boundary",
            "ok",
            "dispatch",
            "held",
            "waited_ms",
            "document_id",
        ],
    ) {
        return Some(value.clone());
    }
    for key in ["session_id", "tab_id"] {
        if result.get(key) == workflow.get(key) {
            result.remove(key);
        }
    }
    if result
        .get("untrusted_content_boundary")
        .and_then(Value::as_str)
        == Some("UNTRUSTED_PAGE_CONTENT")
    {
        result.remove("untrusted_content_boundary");
    }
    // Only this redundant success flag is implicit in the stage's status.
    // In particular, held=false, missing held and ok=false are never promoted.
    if status == "ok" && result.get("ok") == Some(&Value::Bool(true)) {
        result.remove("ok");
    }
    Some(Value::Object(result))
}

/// Unknown envelope versions/fields and malformed data use the original view.
/// Unknown action/expectation detail fields are retained verbatim in the detail.
pub(super) fn render(payload: &Value) -> Option<String> {
    if payload.get("operation")?.as_str()? != "web.workflow" {
        return None;
    }
    let workflow = payload.get("result")?;
    let object = workflow.as_object()?;
    if workflow.get("workflow_version")?.as_u64()? != 1
        || !known(
            object,
            &[
                "workflow_version",
                "session_id",
                "tab_id",
                "completed_steps",
                "total_steps",
                "actions_attempted",
                "steps",
                "page_state",
                "rolled_back",
                "failed_step",
                "phase",
                "untrusted_content_boundary",
            ],
        )
    {
        return None;
    }
    // The caller renders the available snapshot separately. Other observation
    // shapes retain the existing full partial-result / unavailable diagnostics.
    let state = workflow.get("page_state")?;
    if state.get("schema")?.as_str()? != "greppy.web.page-state.v1"
        || state.get("status")?.as_str()? != "available"
        || !state.get("snapshot")?.is_object()
    {
        return None;
    }
    workflow.get("session_id")?.as_str()?;
    workflow.get("tab_id")?.as_str()?;
    for key in ["completed_steps", "total_steps", "actions_attempted"] {
        workflow.get(key)?.as_u64()?;
    }
    workflow.get("rolled_back")?.as_bool()?;
    let mut summary = object.clone();
    summary.remove("steps");
    summary.remove("page_state");
    if summary
        .get("untrusted_content_boundary")
        .and_then(Value::as_str)
        == Some("UNTRUSTED_PAGE_CONTENT")
    {
        summary.remove("untrusted_content_boundary");
    }
    let mut output = format!("workflow: {}\n", Value::Object(summary));
    for step in workflow.get("steps")?.as_array()? {
        let step_object = step.as_object()?;
        if !known(
            step_object,
            &["step", "action", "expectation", "failed_phase"],
        ) || (!step_object.contains_key("action") && !step_object.contains_key("expectation"))
        {
            return None;
        }
        output.push_str(&format!("step {}:", step.get("step")?.as_u64()?));
        for (name, key) in [("action", "receipt"), ("expectation", "result")] {
            let Some(stage) = step.get(name) else {
                continue;
            };
            if !known(
                stage.as_object()?,
                if name == "action" {
                    &["operation", "status", "receipt"]
                } else {
                    &["status", "result"]
                },
            ) {
                return None;
            }
            let status = stage.get("status")?.as_str()?;
            if !matches!(status, "ok" | "error") {
                return None;
            }
            output.push_str(&format!(" {name}"));
            if name == "action" {
                stage.get("operation")?.as_str()?;
                output.push_str(&format!("={}", stage["operation"]));
            }
            output.push_str(&format!(" status={}", stage["status"]));
            let compact = detail(stage.get(key)?, workflow, status)?;
            if !compact.as_object()?.is_empty() {
                output.push_str(&format!(" {key}={compact}"));
            }
        }
        if let Some(phase) = step.get("failed_phase") {
            output.push_str(&format!(" failed_phase={phase}"));
        }
        output.push('\n');
    }
    Some(output)
}
