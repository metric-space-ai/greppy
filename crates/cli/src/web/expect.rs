//! Waiting and checking share a condition compiler. Assert evaluates once.
//! Wait retains its legacy polling backend unless --native explicitly selects
//! the experimental single-request runtime wait. No automatic fallback occurs.

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

    #[derive(clap::Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: ExpectCommand,
    }

    #[test]
    fn reference_conditions_use_bound_nodes_not_css_and_reject_bad_refs() {
        use clap::Parser;
        let parsed = TestCli::try_parse_from(["test", "assert", "@3", "--absent"]).unwrap();
        let ExpectCommand::Assert { condition } = parsed.command else {
            panic!("assert")
        };
        let source = condition_expression(&condition).unwrap();
        assert!(source.contains("__greppyConditionNodes"));
        assert!(!source.contains("querySelector"));
        let mut payload = json!({});
        bind_condition_payload(&condition, &mut payload).unwrap();
        assert_eq!(payload["condition_ref"], json!({"type":"ref", "value":3}));
        for malformed in ["@", "@0", "@-1", "@3x", "@18446744073709551616"] {
            let parsed = TestCli::try_parse_from(["test", "assert", malformed]).unwrap();
            let ExpectCommand::Assert { condition } = parsed.command else {
                panic!("assert")
            };
            assert!(condition_expression(&condition).is_err(), "{malformed}");
        }
    }

    #[test]
    fn native_wait_is_explicit_and_rejects_a_conflicting_poll_interval() {
        use clap::Parser;
        let native = TestCli::try_parse_from(["test", "wait", "css=#late", "--native"]).unwrap();
        assert!(matches!(
            native.command,
            ExpectCommand::Wait { native: true, .. }
        ));
        let legacy = TestCli::try_parse_from(["test", "wait", "css=#late"]).unwrap();
        assert!(matches!(
            legacy.command,
            ExpectCommand::Wait { native: false, .. }
        ));
        let error =
            TestCli::try_parse_from(["test", "wait", "css=#late", "--native", "--interval", "20"])
                .err()
                .expect("an explicit polling option cannot be silently ignored");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn native_wait_preserves_result_evidence_and_never_accepts_missing_confirmation() {
        let state = json!({"status":"available", "snapshot":{"title":"Saved"}});
        let result = normalize_native_wait_response(
            reply(json!({
                "held": true, "waited_ms": 7, "page_state": state, "future_detail": {"value":false}
            })),
            100,
            9,
        )
        .unwrap();
        let value = result.result.unwrap();
        assert_eq!(value["page_state"], state);
        assert_eq!(value["future_detail"]["value"], false);
        assert_eq!(value["waited_ms"], 7);
        assert_eq!(value["wait_backend"], "native_v1");
        for invalid in [
            json!(null),
            json!({}),
            json!({"held":false}),
            json!({"held":"true"}),
        ] {
            assert_eq!(
                normalize_native_wait_response(reply(invalid), 100, 9)
                    .unwrap_err()
                    .code,
                "INVALID_WAIT_RESULT"
            );
        }
    }

    #[test]
    fn native_timeout_keeps_recovery_and_legacy_timeout_status() {
        let request = Request::new("wait-test", "web.wait", json!({}));
        for code in [
            "TIMEOUT",
            "STALE_REF",
            "resource_limit",
            "INVALID_WAIT_SOURCE",
        ] {
            let response = Response::error(
                &request,
                ErrorObject::new(
                    code,
                    "specific runtime failure",
                    request.request_id.clone(),
                    34,
                    "specific recovery",
                ),
            );
            let normalized = normalize_native_wait_response(response, 50, 51).unwrap();
            assert_eq!(normalized.status, "error");
            let error = normalized.error.unwrap();
            assert_eq!(error.code, code);
            assert_eq!(error.next_action, "specific recovery");
            assert_eq!(error.operation_id, request.request_id);
            if code == "TIMEOUT" {
                assert_eq!(error.exit_code, EXIT_WAIT_TIMEOUT);
                let result = normalized.result.unwrap();
                assert_eq!(result["held"], false);
                assert_eq!(result["timeout_ms"], 50);
            } else {
                assert_eq!(error.exit_code, 34);
                assert!(normalized.result.is_none());
            }
        }
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
    /// @N is a node from this tab's observation; stale refs remain errors,
    /// including with --absent. A bare argument is a CSS selector.
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
        /// Poll interval in milliseconds for the legacy backend.
        #[arg(long, default_value_t = 120, conflicts_with = "native")]
        interval: u64,
        /// Experimental single-request native wait; requires a matching runtime.
        /// Runtime errors remain errors; this mode never falls back to polling.
        #[arg(long)]
        native: bool,
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
            native,
        } => {
            if native {
                wait_native(root, condition, timeout)
            } else {
                wait(root, condition, timeout, interval)
            }
        }
        ExpectCommand::Assert { condition } => assert_once(root, condition),
    }
}

/// Build the expression that answers the condition with `{ holds, detail }`.
fn condition_expression(condition: &Condition) -> std::result::Result<String, String> {
    let mut checks: Vec<String> = Vec::new();
    if let Some(query) = &condition.query {
        if condition_ref_selector(condition)?.is_some() {
            // This lexical binding is supplied by the runtime only after
            // session/page/snapshot and live node identity have been checked.
            // An older runtime must fail, not treat an unbound ref as absent.
            checks.push("(function(){ if (typeof __greppyConditionNodes === 'undefined') throw new Error('REF_CONDITION_UNSUPPORTED: use a matching CLI and runtime'); return { holds: __greppyConditionNodes.length > 0, detail: { matched: __greppyConditionNodes.length } }; })()".into());
        } else {
            super::see::validate_condition_query(query)?;
            let body = "return { holds: nodes.length > 0, detail: { matched: nodes.length } };";
            checks.push(super::see::query_expression_pub(query, body));
        }
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

fn condition_ref_selector(
    condition: &Condition,
) -> std::result::Result<Option<serde_json::Value>, String> {
    let Some(query) = condition
        .query
        .as_deref()
        .filter(|q| q.trim().starts_with('@'))
    else {
        return Ok(None);
    };
    parse_target(query, false, false, None)
        .map(|target| Some(target.selector))
        .map_err(|error| error.message.to_string())
}

fn bind_condition_payload(
    condition: &Condition,
    payload: &mut serde_json::Value,
) -> std::result::Result<(), ErrorObject> {
    if let Some(selector) =
        condition_ref_selector(condition).map_err(|message| query_syntax(&message))?
    {
        payload["condition_ref"] = selector;
    }
    Ok(())
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
    condition: &Condition,
    source: &str,
) -> std::result::Result<(bool, serde_json::Value), ErrorObject> {
    let session = resolve_session(root, condition.session.clone())?;
    let mut payload = json!({ "session_id": session, "source": source });
    bind_condition_payload(condition, &mut payload)?;
    if let Some(tab) = resolve_tab(root, condition.tab.clone()) {
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

/// Adapt the existing typed condition result to the native Boolean contract.
/// Validate before IPC; neither a missing value nor an object can prove absence.
fn native_condition_source(condition: &Condition) -> std::result::Result<String, String> {
    let source = condition_expression(condition)?;
    Ok(format!(
        "(function(){{ var r = ({source}); \
         if (!r || typeof r.holds !== 'boolean') throw new Error('INVALID_WAIT_PREDICATE'); \
         return r.holds !== {}; }})()",
        condition.absent
    ))
}

fn normalize_native_wait_response(
    mut response: greppy_web_client::Response,
    timeout: u64,
    waited_ms: u64,
) -> std::result::Result<greppy_web_client::Response, ErrorObject> {
    if let Some(error) = response.error.as_mut() {
        if error.code == "TIMEOUT" {
            // Keep the CLI's existing wait-timeout exit code and fields while
            // retaining the runtime's typed error, metrics and recovery text.
            error.exit_code = EXIT_WAIT_TIMEOUT;
            let result = response.result.get_or_insert_with(|| json!({}));
            if let Some(result) = result.as_object_mut() {
                result.insert("held".into(), json!(false));
                result.insert("timeout_ms".into(), json!(timeout));
                result.insert("waited_ms".into(), json!(waited_ms));
                result.entry("detail").or_insert(json!(null));
                result.insert("wait_backend".into(), json!("native_v1"));
            }
        }
        // Other errors are never inverted or used to trigger another backend.
        return Ok(response);
    }
    if response.status != "ok"
        || response
            .result
            .as_ref()
            .and_then(|r| r.get("held"))
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        return Err(ErrorObject::new(
            "INVALID_WAIT_RESULT",
            "native wait returned an invalid reply; expected status=ok and held=true",
            response.request_id.clone(),
            EXIT_WEB_ENGINE,
            "inspect the runtime response contract; no condition was confirmed",
        ));
    }
    let result = response
        .result
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
        .expect("a held field requires an object");
    result.entry("detail").or_insert(json!(null));
    result.entry("waited_ms").or_insert(json!(waited_ms));
    result.insert("wait_backend".into(), json!("native_v1"));
    Ok(response)
}

fn wait_native(root: Option<&str>, condition: Condition, timeout: u64) -> Result<i32> {
    let source = match native_condition_source(&condition) {
        Ok(source) => source,
        Err(message) => {
            return emit_error(condition.json, invalid(&format!("web wait: {message}")))
        }
    };
    if Instant::now()
        .checked_add(Duration::from_millis(timeout.saturating_add(5_000)))
        .is_none()
    {
        return emit_error(
            condition.json,
            invalid("web wait: timeout exceeds the supported monotonic clock range"),
        );
    }
    let session = match resolve_session(root, condition.session.clone()) {
        Ok(session) => session,
        Err(error) => return emit_error(condition.json, error),
    };
    let mut payload = json!({"session_id": session, "source": source, "timeout_ms": timeout});
    if let Err(error) = bind_condition_payload(&condition, &mut payload) {
        return emit_error(condition.json, error);
    }
    if let Some(tab) = resolve_tab(root, condition.tab.clone()) {
        payload["tab_id"] = json!(tab);
    }
    let started = Instant::now();
    match rpc_response(root, "web.wait", payload, Some(session)).and_then(|response| {
        normalize_native_wait_response(
            response,
            timeout,
            started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        )
    }) {
        Ok(response) => emit_response(condition.json, response),
        Err(error) => emit_error(condition.json, error),
    }
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
        match evaluate_condition(root, &condition, &source) {
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
    match evaluate_condition(root, &condition, &source) {
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
