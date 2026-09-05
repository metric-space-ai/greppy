//! Waiting and checking. Both verbs ask the same question — does the page
//! satisfy a condition — so both are polls over `web.evaluate` rather than
//! separate engine operations. `wait` polls until the deadline, `assert`
//! asks once and turns the answer into an exit code.

use super::common::*;
use clap::{Args, Subcommand};
use greppy_core::error::Result;
use greppy_web_client::ErrorObject;
use serde_json::json;
use std::time::{Duration, Instant};

/// Exit code for a condition that did not hold. Distinct from a protocol
/// error: the command worked, the page just does not satisfy the claim.
const EXIT_ASSERT_FAILED: i32 = 18;
/// Exit code for a deadline that expired.
const EXIT_WAIT_TIMEOUT: i32 = 13;

#[cfg(test)]
mod response_tests {
    use super::*;
    use greppy_web_client::{Request, Response};

    fn reply(result: serde_json::Value) -> Response {
        Response::ok(
            &Request::new("condition-test", "web.evaluate", json!({})),
            result,
        )
    }

    #[test]
    fn malformed_runtime_values_cannot_prove_absence() {
        for value in [
            json!(null),
            json!({}),
            json!({"holds": null}),
            json!({"holds": "false"}),
            json!({"holds": 0}),
            json!({"holds": []}),
        ] {
            let response = reply(json!({"value": value}));
            let error = decode_condition_response(response)
                .expect_err("a missing boolean must not become false and satisfy --absent");
            assert_eq!(error.code, "INVALID_CONDITION_RESULT");
            assert_eq!(error.exit_code, 34);
        }
        assert!(decode_condition_response(reply(json!({}))).is_err());
    }

    #[test]
    fn valid_boolean_and_details_are_preserved() {
        for holds in [false, true] {
            let detail = json!({"matched": if holds { 1 } else { 0 }});
            let actual = decode_condition_response(reply(json!({
                "value": {"holds": holds, "detail": detail}
            })))
            .unwrap();
            assert_eq!(actual, (holds, detail));
        }
    }

    #[test]
    fn error_status_cannot_be_overridden_by_a_boolean_value() {
        let mut response = reply(json!({"value": {"holds": false}}));
        response.status = "error".into();
        assert!(decode_condition_response(response).is_err());
    }

    #[test]
    fn typed_runtime_errors_are_preserved_without_inversion() {
        let request = Request::new("condition-test", "web.evaluate", json!({}));
        let response = Response::error(
            &request,
            ErrorObject::new(
                "STALE_REF",
                "node was replaced",
                request.request_id.clone(),
                34,
                "observe the current document",
            ),
        );
        let error = decode_condition_response(response).unwrap_err();
        assert_eq!(error.code, "STALE_REF");
        assert_eq!(error.exit_code, 34);
        assert_eq!(error.operation_id, request.request_id);
    }
}

#[derive(Debug, Args, Clone)]
pub struct Condition {
    /// Node query: css=, xpath=, text=, text~/re/, role=, id=, tag=.
    /// text= matches normalized whole-element text; text~/re/ matches a part.
    /// A bare argument is a CSS selector. Time durations are not conditions.
    pub query: Option<String>,
    /// Match on the document URL instead of a node, e.g. `--url '~/\/done$/'`.
    #[arg(long)]
    pub url: Option<String>,
    /// Match on the document title.
    #[arg(long)]
    pub title: Option<String>,
    /// Require the condition to be false rather than true.
    #[arg(long)]
    pub absent: bool,
    #[arg(long)]
    pub session: Option<String>,
    /// Check a particular tab in the selected session.
    #[arg(long)]
    pub tab: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Subcommand)]
pub enum ExpectCommand {
    /// Wait until a condition holds, or fail at the deadline.
    ///
    ///   greppy web wait 'css=#late'
    ///   greppy web wait --url '~/\/dashboard/' --timeout 10000
    Wait {
        #[command(flatten)]
        condition: Condition,
        /// Deadline in milliseconds.
        #[arg(long, default_value_t = 10_000)]
        timeout: u64,
        /// Poll interval in milliseconds.
        #[arg(long, default_value_t = 120)]
        interval: u64,
    },
    /// Check a condition once. Exit 18 when it does not hold.
    ///
    ///   greppy web assert 'text~/welcome/i'
    ///   greppy web assert 'css=.error' --absent
    Assert {
        #[command(flatten)]
        condition: Condition,
    },
}

pub(super) fn dispatch(command: ExpectCommand, root: Option<&str>) -> Result<i32> {
    match command {
        ExpectCommand::Wait {
            condition,
            timeout,
            interval,
        } => wait(root, condition, timeout, interval),
        ExpectCommand::Assert { condition } => assert_once(root, condition),
    }
}

/// Build the expression that answers the condition with `{ holds, detail }`.
fn condition_expression(condition: &Condition) -> std::result::Result<String, String> {
    let mut checks: Vec<String> = Vec::new();
    if let Some(query) = &condition.query {
        super::see::validate_condition_query(query)?;
        let body = "return { holds: nodes.length > 0, detail: { matched: nodes.length } };";
        checks.push(super::see::query_expression_pub(query, body));
    }
    if let Some(pattern) = &condition.url {
        checks.push(text_check("location.href", pattern)?);
    }
    if let Some(pattern) = &condition.title {
        checks.push(text_check("document.title", pattern)?);
    }
    match checks.len() {
        0 => Err("needs a query, --url or --title".into()),
        1 => Ok(checks.remove(0)),
        _ => {
            // Several parts must all hold; report the first that does not so
            // the caller learns which half of the condition is missing.
            let parts = checks.join(", ");
            Ok(format!(
                "(function(){{ var rs = [{parts}]; \
                 var bad = rs.find(function(r) {{ return !r.holds; }}); \
                 return bad ? bad : {{ holds: true, detail: rs.map(function(r) {{ return r.detail; }}) }}; }})()"
            ))
        }
    }
}

/// `value` compared against either `/regex/flags` or an exact string.
fn text_check(accessor: &str, pattern: &str) -> std::result::Result<String, String> {
    let pattern = pattern.trim();
    let expr = if let Some(rest) = pattern.strip_prefix('~') {
        let rest = rest.trim();
        let body = rest
            .strip_prefix('/')
            .ok_or("regex must be ~/pattern/flags")?;
        let close = body.rfind('/').ok_or("regex must be ~/pattern/flags")?;
        let (re, flags) = body.split_at(close);
        let flags = &flags[1..];
        if !flags.chars().all(|f| "imsu".contains(f)) {
            return Err(format!("unsupported regex flags `{flags}`"));
        }
        // Compile here so a malformed pattern is a usage error rather than a
        // JavaScript exception surfacing from inside the page.
        regex::Regex::new(re).map_err(|error| format!("invalid regex: {error}"))?;
        format!(
            "new RegExp({}, {}).test(String({accessor}))",
            serde_json::Value::String(re.to_owned()),
            serde_json::Value::String(flags.to_owned())
        )
    } else {
        format!(
            "String({accessor}) === {}",
            serde_json::Value::String(pattern.to_owned())
        )
    };
    Ok(format!(
        "(function(){{ var v = String({accessor}); return {{ holds: {expr}, detail: {{ value: v.slice(0, 200) }} }}; }})()"
    ))
}

/// Ask the selected page once. Invalid replies cannot prove presence or absence.
fn evaluate_condition(
    root: Option<&str>,
    session: Option<String>,
    tab: Option<String>,
    source: &str,
) -> std::result::Result<(bool, serde_json::Value), ErrorObject> {
    let session = resolve_session(root, session)?;
    let mut payload = json!({ "session_id": session, "source": source });
    if let Some(tab) = resolve_tab(root, tab) {
        payload["tab_id"] = json!(tab);
    }
    let response = rpc_response(root, "web.evaluate", payload, Some(session))?;
    decode_condition_response(response)
}

fn decode_condition_response(
    response: greppy_web_client::Response,
) -> std::result::Result<(bool, serde_json::Value), ErrorObject> {
    if let Some(error) = response.error {
        return Err(error);
    }
    let invalid_result = || {
        ErrorObject::new(
        "INVALID_CONDITION_RESULT",
        "condition evaluation returned an invalid reply; expected status=ok and a boolean value.holds",
        response.request_id.clone(),
        34,
        "inspect the runtime response contract; an invalid reply cannot prove presence or absence",
    )
    };
    if response.status != "ok" {
        return Err(invalid_result());
    }
    let value = response
        .result
        .as_ref()
        .and_then(|result| result.get("value"))
        .ok_or_else(invalid_result)?;
    let holds = value
        .get("holds")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(invalid_result)?;
    let detail = value.get("detail").cloned().unwrap_or(json!(null));
    Ok((holds, detail))
}

fn wait(root: Option<&str>, condition: Condition, timeout: u64, interval: u64) -> Result<i32> {
    let source = match condition_expression(&condition) {
        Ok(source) => source,
        Err(message) => {
            return emit_error(condition.json, invalid(&format!("web wait: {message}")));
        }
    };
    let deadline = Instant::now() + Duration::from_millis(timeout);
    let started = Instant::now();
    let mut last_detail = json!(null);
    loop {
        match evaluate_condition(
            root,
            condition.session.clone(),
            condition.tab.clone(),
            &source,
        ) {
            Err(error) => return emit_error(condition.json, error),
            Ok((holds, detail)) => {
                last_detail = detail;
                if holds != condition.absent {
                    return ok_envelope(
                        condition.json,
                        "web.wait",
                        json!({
                            "held": true,
                            "waited_ms": started.elapsed().as_millis() as u64,
                            "detail": last_detail,
                        }),
                        0,
                    );
                }
            }
        }
        if Instant::now() >= deadline {
            return ok_envelope(
                condition.json,
                "web.wait",
                json!({
                    "held": false,
                    "waited_ms": started.elapsed().as_millis() as u64,
                    "timeout_ms": timeout,
                    "detail": last_detail,
                }),
                EXIT_WAIT_TIMEOUT,
            );
        }
        std::thread::sleep(Duration::from_millis(interval.max(10)));
    }
}

fn assert_once(root: Option<&str>, condition: Condition) -> Result<i32> {
    let source = match condition_expression(&condition) {
        Ok(source) => source,
        Err(message) => {
            return emit_error(condition.json, invalid(&format!("web assert: {message}")));
        }
    };
    match evaluate_condition(
        root,
        condition.session.clone(),
        condition.tab.clone(),
        &source,
    ) {
        Err(error) => emit_error(condition.json, error),
        Ok((holds, detail)) => {
            let satisfied = holds != condition.absent;
            ok_envelope(
                condition.json,
                "web.assert",
                json!({
                    "held": satisfied,
                    "expected_absent": condition.absent,
                    "detail": detail,
                }),
                if satisfied { 0 } else { EXIT_ASSERT_FAILED },
            )
        }
    }
}

/// Emit the condition verdict. An unmet condition has error status and a
/// nonzero exit code even though the evaluation itself was well formed.
fn ok_envelope(
    json_out: bool,
    operation: &str,
    result: serde_json::Value,
    code: i32,
) -> Result<i32> {
    emit_web(
        json_out,
        &json!({
            "schema": "greppy.web-runtime.v1",
            "status": if code == 0 { "ok" } else { "error" },
            "operation": operation,
            "result": result,
        }),
    )?;
    Ok(code)
}
