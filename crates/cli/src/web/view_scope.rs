//! Human-only modal projection. The caller must archive the complete payload
//! before using this projection. Unknown or ambiguous scope schemas fall back.
use serde_json::{json, Value};
use std::collections::HashSet;

pub(super) fn project(payload: &Value) -> Option<Value> {
    let receipt = payload
        .get("result")
        .filter(|v| !v.is_null())
        .unwrap_or(payload);
    let nested = receipt.get("page_state").is_some_and(|state| {
        state["schema"] == "greppy.web.page-state.v1"
            && state["status"] == "available"
            && state["snapshot"].is_object()
    });
    let snapshot = if nested {
        &receipt["page_state"]["snapshot"]
    } else {
        receipt
    };
    let scope = snapshot.get("working_scope")?;
    if scope["schema"] != "greppy.web.working-scope.v1" || scope["kind"] != "modal" {
        return None;
    }
    let fields = scope.as_object()?;
    if fields.keys().any(|key| {
        ![
            "schema",
            "kind",
            "scope_ref",
            "role",
            "name",
            "provenance",
            "native_modal_detection",
            "modal_candidates",
            "focus_ref",
            "focus_source",
            "ancestry",
            "ancestry_truncated",
            "actionable_refs",
            "background_count",
            "background_returned",
            "background_location",
            "text",
            "text_truncated",
        ]
        .contains(&key.as_str())
    }) {
        return None;
    }
    if !matches!(
        scope["provenance"].as_str(),
        Some("native_modal" | "declared_aria_modal")
    ) || !scope["scope_ref"].is_string()
        || !scope["text"].is_string()
        || !scope["text_truncated"].is_boolean()
        || !scope["ancestry"].is_array()
        || scope["background_location"] != "snapshot.actionables"
    {
        return None;
    }
    let actions = snapshot["actionables"].as_array()?;
    let refs = scope["actionable_refs"].as_array()?;
    let foreground: HashSet<&str> = refs.iter().map(Value::as_str).collect::<Option<_>>()?;
    let all: HashSet<&str> = actions
        .iter()
        .map(|node| node["ref"].as_str())
        .collect::<Option<_>>()?;
    if foreground.len() != refs.len() || all.len() != actions.len() || !foreground.is_subset(&all) {
        return None;
    }
    let omitted = actions.len().checked_sub(foreground.len())? as u64;
    if scope["background_returned"].as_u64()? != omitted
        || scope["background_count"].as_u64()? < omitted
    {
        return None;
    }
    let selected: Vec<Value> = actions
        .iter()
        .filter(|node| foreground.contains(node["ref"].as_str().unwrap()))
        .cloned()
        .collect();
    let context = json!({
        "kind":"modal", "ref":scope["scope_ref"], "role":scope["role"],
        "name":scope["name"], "provenance":scope["provenance"],
        "native_modal_detection":scope["native_modal_detection"],
        "modal_candidates":scope["modal_candidates"],
        "focus_ref":scope["focus_ref"], "focus_source":scope["focus_source"],
        "ancestry":scope["ancestry"], "ancestry_truncated":scope["ancestry_truncated"],
        "text_truncated":scope["text_truncated"],
        "background_controls_returned":omitted,
        "background_controls_total":scope["background_count"],
    });
    let mut projected = payload.clone();
    let receipt = if payload.get("result").is_some_and(|value| !value.is_null()) {
        projected.get_mut("result")?
    } else {
        &mut projected
    };
    let snapshot = if nested {
        &mut receipt["page_state"]["snapshot"]
    } else {
        receipt
    };
    let object = snapshot.as_object_mut()?;
    object.insert("actionables".into(), json!(selected));
    object.insert("text".into(), scope["text"].clone());
    object.remove("working_scope");
    // These are page-wide projections, retained in the archived original.
    object.remove("headings");
    object.remove("links");
    object.insert("working_context".into(), context);
    Some(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn modal() -> Value {
        json!({"operation":"web.observe","result":{
            "text":"Background table Secret row Dialog text", "headings":["Background"],
            "actionables":[{"ref":"@1","name":"Background"},{"ref":"@2","name":"Confirm"}],
            "working_scope":{"schema":"greppy.web.working-scope.v1","kind":"modal",
                "scope_ref":"@3","provenance":"native_modal","text":"Dialog text",
                "text_truncated":false,"ancestry":[],"actionable_refs":["@2"],
                "background_count":1,"background_returned":1,"background_location":"snapshot.actionables"}
        }})
    }
    #[test]
    fn modal_foregrounds_controls_and_text_without_mutating_original() {
        let original = modal();
        let projected = project(&original).unwrap();
        assert_eq!(
            projected["result"]["actionables"].as_array().unwrap().len(),
            1
        );
        assert_eq!(projected["result"]["text"], "Dialog text");
        assert_eq!(
            original["result"]["actionables"].as_array().unwrap().len(),
            2
        );
        assert!(original["result"]["text"]
            .as_str()
            .unwrap()
            .contains("Secret row"));
    }
    #[test]
    fn ambiguous_future_and_inconsistent_scopes_are_not_filtered() {
        for (key, value) in [
            ("kind", json!("ambiguous")),
            ("schema", json!("future")),
            ("new_important_field", json!(true)),
            ("background_returned", json!(0)),
            ("actionable_refs", json!(["@99"])),
            ("actionable_refs", json!(["@2", "@2"])),
        ] {
            let mut value_original = modal();
            value_original["result"]["working_scope"][key] = value;
            assert!(project(&value_original).is_none(), "{key}");
        }
    }
    #[test]
    fn declared_aria_keeps_its_provenance_and_failure_receipt() {
        let mut snapshot = modal()["result"].clone();
        snapshot["working_scope"]["provenance"] = json!("declared_aria_modal");
        let original = json!({"status":"error","error":{"code":"TIMEOUT"},"result":{
            "completed_steps":1,"page_state":{"schema":"greppy.web.page-state.v1","status":"available","snapshot":snapshot}}});
        let projected = project(&original).unwrap();
        assert_eq!(projected["error"], original["error"]);
        assert_eq!(projected["result"]["completed_steps"], 1);
        assert_eq!(
            projected["result"]["page_state"]["snapshot"]["working_context"]["provenance"],
            "declared_aria_modal"
        );
    }
}
