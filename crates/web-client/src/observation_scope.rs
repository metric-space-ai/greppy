//! Fail closed when an older peer ignores the optional observation query.
use crate::{ErrorObject, Response};
use serde_json::Value;

pub fn observation_scope_roots(result: &Value, query: &str) -> Result<u64, &'static str> {
    let scope = result
        .get("observation_scope")
        .ok_or("missing observation scope")?;
    if scope.get("schema").and_then(Value::as_str) != Some("greppy.web.observation-scope.v1")
        || scope.get("query").and_then(Value::as_str) != Some(query)
    {
        return Err("observation scope does not match the requested query");
    }
    let count = |key| {
        scope
            .get(key)
            .and_then(Value::as_u64)
            .ok_or("invalid observation scope count")
    };
    let matched = count("matched_elements")?;
    let visible = count("visible_matches")?;
    let total = count("roots_total")?;
    let returned = count("roots_returned")?;
    if matched < visible
        || visible < total
        || returned != total.min(20)
        || scope.get("roots_truncated").and_then(Value::as_bool) != Some(total > returned)
    {
        return Err("inconsistent observation scope bounds");
    }
    for key in ["text_truncated", "headings_truncated", "links_truncated"] {
        if scope.get(key).and_then(Value::as_bool).is_none() {
            return Err("missing observation truncation evidence");
        }
    }
    Ok(returned)
}

pub fn guard_scoped_observation(query: &str, mut response: Response) -> Response {
    let roots = response
        .result
        .as_ref()
        .and_then(|result| observation_scope_roots(result, query).ok());
    if response.status == "ok" && !matches!(roots, Some(1..)) {
        response.status = "error".into();
        response.error = Some(ErrorObject::new(
            "unsupported_observation_scope",
            "runtime did not prove the requested observation scope; unfiltered data was discarded",
            response.request_id.clone(), 34,
            "install the CLI and web runtime from the same build, then stop the old runtime with `greppy web runtime stop` and retry",
        ));
        response.result = None;
    } else if response.status != "ok" && roots.is_none() {
        // Preserve the original failure, but do not print an unrelated page
        // snapshot from a peer which does not implement scoped observations.
        response.result = None;
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Request;
    use serde_json::json;

    fn tree(query: &str, total: u64) -> Value {
        json!({"text":"scoped", "observation_scope": {
            "schema":"greppy.web.observation-scope.v1", "query":query,
            "matched_elements":total,"visible_matches":total,"roots_total":total,
            "roots_returned":total.min(20),"roots_truncated":total>20,
            "text_truncated":false,"headings_truncated":false,"links_truncated":false
        }})
    }

    #[test]
    fn scope_bounds_and_query_identity_are_validated() {
        for total in [0, 1, 20, 21, 100] {
            assert_eq!(
                observation_scope_roots(&tree("role=dialog", total), "role=dialog"),
                Ok(total.min(20))
            );
        }
        assert!(observation_scope_roots(&tree("role=dialog", 1), "css=body").is_err());
        for (key, value) in [
            ("matched_elements", json!(0)),
            ("visible_matches", json!(0)),
            ("roots_returned", json!(2)),
            ("roots_truncated", json!(true)),
            ("text_truncated", Value::Null),
        ] {
            let mut value_tree = tree("role=dialog", 1);
            value_tree["observation_scope"][key] = value;
            assert!(
                observation_scope_roots(&value_tree, "role=dialog").is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn older_peer_success_cannot_widen_a_scoped_request() {
        let request = Request::new("test", "web.observe", json!({"query":"role=dialog"}));
        for result in [
            json!({"text":"UNRELATED_PAGE"}),
            tree("css=body", 1),
            tree("role=dialog", 0),
        ] {
            let response = guard_scoped_observation("role=dialog", Response::ok(&request, result));
            assert_eq!(response.status, "error");
            assert_eq!(
                response.error.unwrap().code,
                "unsupported_observation_scope"
            );
            assert!(response.result.is_none());
        }
    }

    #[test]
    fn verified_scope_is_preserved_byte_for_byte() {
        let request = Request::new("test", "web.observe", json!({}));
        let response = Response::ok(&request, tree("role=dialog", 21));
        let before = serde_json::to_value(&response).unwrap();
        assert_eq!(
            serde_json::to_value(guard_scoped_observation("role=dialog", response)).unwrap(),
            before
        );
    }

    #[test]
    fn failures_keep_their_error_but_discard_unscoped_page_content() {
        let request = Request::new("test", "web.observe", json!({}));
        let mut response = Response::error(
            &request,
            ErrorObject::new("TIMEOUT", "deadline", "op", 37, "retry"),
        );
        response.result = Some(json!({"text":"UNRELATED_PAGE"}));
        let response = guard_scoped_observation("role=dialog", response);
        assert_eq!(response.error.unwrap().code, "TIMEOUT");
        assert!(response.result.is_none());
    }

    #[test]
    fn scoped_no_match_retains_its_empty_scope_evidence() {
        let request = Request::new("test", "web.observe", json!({}));
        let mut response = Response::error(
            &request,
            ErrorObject::new("NO_MATCH", "absent", "op", 32, "inspect query"),
        );
        response.result = Some(tree("role=dialog", 0));
        let before = serde_json::to_value(&response).unwrap();
        assert_eq!(
            serde_json::to_value(guard_scoped_observation("role=dialog", response)).unwrap(),
            before
        );
    }
}
