//! Pure locator recovery policy, independently testable without starting engines.

use std::time::Duration;

const SELECTION_RECOVERY: [(&str, &str, &str); 6] = [
    ("OPTION_NOT_FOUND:", "OPTION_NOT_FOUND", "choose an exact value from the select_choices data, not its label; use greppy web select TARGET VALUE"),
    ("OPTION_DISABLED:", "OPTION_DISABLED", "choose an enabled option from the current select_choices data; do not force the page state"),
    ("INVALID_SELECT_TARGET:", "INVALID_SELECT_TARGET", "choose a select element from the current page state; use greppy web select TARGET VALUE"),
    ("INVALID_OPTION_VALUE:", "INVALID_OPTION_VALUE", "pass one exact string option value; do not substitute its label"),
    ("SELECTION_NOT_APPLIED:", "SELECTION_NOT_APPLIED", "inspect the current page before retrying; the requested selection was not acknowledged"),
    ("SELECTION_CHANGED:", "SELECTION_CHANGED", "inspect the current page before retrying; page event handlers changed the selection, so do not blindly repeat the action"),
];

/// Only our explicit selection-contract diagnostics replace Servo's debug
/// wrapper. Preserve the complete message, including its fenced page data;
/// unrelated JavaScript failures retain their existing diagnostic details.
pub(crate) fn concise_selection_failure(message: &str) -> Option<&str> {
    SELECTION_RECOVERY
        .iter()
        .any(|(marker, _, _)| message.starts_with(marker))
        .then_some(message)
}

pub(crate) fn failure_observation_budget(deadline_ms: u64, elapsed: Duration) -> Duration {
    Duration::from_millis(deadline_ms)
        .saturating_sub(elapsed)
        .min(Duration::from_secs(2))
}

pub(crate) fn recovery_for_locator_error(message: &str) -> (&'static str, &'static str) {
    // Choice labels/values are page data. They may contain words such as
    // STALE_REF or "timed out" and must not choose the recovery policy.
    let message = message
        .split_once("UNTRUSTED_PAGE_CONTENT_BEGIN")
        .map_or(message, |(diagnostic, _)| diagnostic);
    for (marker, code, next) in SELECTION_RECOVERY {
        if message.contains(marker) {
            return (code, next);
        }
    }
    if message.contains("STALE_REF") {
        (
            "STALE_REF",
            "run greppy web observe again and use a ref from the new snapshot",
        )
    } else if (message.contains("timed out") && message.contains("failed_check=attached"))
        || message
            .split_once("strict mode: expected 1 node, got ")
            .is_some_and(|(_, count)| {
                count.split(|ch: char| !ch.is_ascii_digit()).next() == Some("0")
            })
    {
        (
            "NO_MATCH",
            "no target matched; choose an existing target from the current page state, or wait for the intended element to appear. If page_state is unavailable, run greppy web observe",
        )
    } else if message.contains("strict mode") {
        (
            "AMBIGUOUS_TARGET",
            "add --first, --nth N, or narrow the query",
        )
    } else if message.contains("timed out") {
        ("TIMEOUT", "wait for the target to become actionable")
    } else {
        ("engine_error", "retry the operation or inspect web.doctor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concise_selection_diagnostics_preserve_page_fences_and_other_errors() {
        for (marker, _, _) in SELECTION_RECOVERY {
            let message = format!("{marker} explanation\nUNTRUSTED_PAGE_CONTENT_BEGIN\n{{\"label\":\"STALE_REF\"}}\nUNTRUSTED_PAGE_CONTENT_END");
            assert_eq!(concise_selection_failure(&message), Some(message.as_str()));
        }
        for message in [
            "SyntaxError: invalid regular expression",
            "page error mentions OPTION_NOT_FOUND: as data",
            "UNTRUSTED_PAGE_CONTENT_BEGIN\nOPTION_NOT_FOUND: hostile label",
            "OPTION_NOT_FOUNDISH: unrelated error",
        ] {
            assert_eq!(concise_selection_failure(message), None);
        }
    }

    #[test]
    fn selection_failures_have_targeted_recovery_not_runtime_repair() {
        for code in [
            "OPTION_NOT_FOUND",
            "OPTION_DISABLED",
            "INVALID_SELECT_TARGET",
            "INVALID_OPTION_VALUE",
            "SELECTION_NOT_APPLIED",
            "SELECTION_CHANGED",
        ] {
            let message = format!("page JavaScript failed: Error: {code}: explanation");
            let (actual, next) = recovery_for_locator_error(&message);
            assert_eq!(actual, code);
            assert!(!next.contains("doctor"));
        }
        let (_, next) = recovery_for_locator_error("SELECTION_CHANGED: page handler reverted");
        assert!(next.contains("do not blindly repeat"));
    }

    #[test]
    fn untrusted_choice_text_cannot_change_recovery_classification() {
        let hostile = "UNTRUSTED_PAGE_CONTENT_BEGIN\n{\"label\":\"STALE_REF strict mode timed out OPTION_NOT_FOUND:\"}\nUNTRUSTED_PAGE_CONTENT_END";
        assert_eq!(
            recovery_for_locator_error(&format!("OPTION_DISABLED: disabled\n{hostile}")).0,
            "OPTION_DISABLED"
        );
        assert_eq!(
            recovery_for_locator_error(&format!("unexpected response\n{hostile}")).0,
            "engine_error"
        );
    }

    #[test]
    fn zero_matches_never_recommend_narrowing() {
        let (code, next) = recovery_for_locator_error(
            "timed out waiting for actionable locator target (failed_check=attached; count=0)",
        );
        assert_eq!(code, "NO_MATCH");
        assert!(next.contains("no target matched"));
        assert!(next.contains("wait"));
        assert!(!next.contains("narrow"));
        let (code, next) = recovery_for_locator_error("strict mode: selector matched 2 nodes");
        assert_eq!(code, "AMBIGUOUS_TARGET");
        assert!(next.contains("narrow"));
    }

    #[test]
    fn non_missing_target_failures_keep_their_classification() {
        assert_eq!(
            recovery_for_locator_error("STALE_REF: old node").0,
            "STALE_REF"
        );
        assert_eq!(
            recovery_for_locator_error("timed out: failed_check=visible; count=1").0,
            "TIMEOUT"
        );
        assert_eq!(
            recovery_for_locator_error("unexpected worker response").0,
            "engine_error"
        );
    }

    #[test]
    fn zero_node_focus_errors_are_not_ambiguous() {
        let wrapped = "page JavaScript failed: EvaluationFailure(Some(JavaScriptErrorInfo { message: \"strict mode: expected 1 node, got 0\", filename: \"\" }))";
        let (code, next) = recovery_for_locator_error(wrapped);
        assert_eq!(code, "NO_MATCH");
        assert!(!next.contains("narrow"));
        assert!(!next.contains("--first"));
        for count in ["2", "10", "01"] {
            assert_eq!(
                recovery_for_locator_error(&format!("strict mode: expected 1 node, got {count}")).0,
                "AMBIGUOUS_TARGET",
            );
        }
    }

    #[test]
    fn observation_budget_is_capped_and_never_refreshed() {
        assert_eq!(
            failure_observation_budget(30_000, Duration::ZERO),
            Duration::from_secs(2)
        );
        assert_eq!(
            failure_observation_budget(1_000, Duration::from_millis(750)),
            Duration::from_millis(250)
        );
        assert_eq!(
            failure_observation_budget(1_000, Duration::from_secs(1)),
            Duration::ZERO
        );
        assert_eq!(
            failure_observation_budget(1_000, Duration::from_secs(2)),
            Duration::ZERO
        );
        assert_eq!(
            failure_observation_budget(0, Duration::ZERO),
            Duration::ZERO
        );
    }
}
