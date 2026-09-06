//! Budget and diagnostics for the experimental internal Boolean wait bridge.

use std::time::{Duration, Instant};

/// CLOCK_MONOTONIC is shared by these local Unix processes. An Instant cannot
/// be serialized, and wall-clock timestamps can jump during a queued call.
pub(crate) fn monotonic_ns() -> std::io::Result<u64> {
    let mut time: libc::timespec = unsafe { std::mem::zeroed() };
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let seconds = u64::try_from(time.tv_sec).map_err(std::io::Error::other)?;
    let nanos = u64::try_from(time.tv_nsec).map_err(std::io::Error::other)?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or_else(|| std::io::Error::other("monotonic timestamp overflow"))
}

pub(crate) fn worker_remaining(now_ns: u64, deadline_ns: u64, maximum: Duration) -> Duration {
    Duration::from_nanos(deadline_ns.saturating_sub(now_ns)).min(maximum)
}

pub(crate) fn remaining_wait_budget(
    request_ms: u64,
    timeout_ms: u64,
    elapsed: Duration,
    session_remaining: Duration,
) -> Duration {
    Duration::from_millis(request_ms.min(timeout_ms))
        .saturating_sub(elapsed)
        .min(session_remaining)
}

pub(crate) fn wait_io_budget(deadline: Option<Instant>, fallback: Duration) -> Duration {
    let budget = deadline
        .map(|end| end.saturating_duration_since(Instant::now()).min(fallback))
        .unwrap_or(fallback);
    if deadline.is_some() && budget < Duration::from_millis(1) {
        Duration::ZERO
    } else {
        budget
    }
}

pub(crate) fn wait_error_detail(message: &str) -> (&'static str, &'static str, &'static str) {
    if message.contains("INVALID_WAIT_PREDICATE") {
        (
            "INVALID_WAIT_PREDICATE",
            "wait condition must return a Boolean, not an object or other value",
            "correct the condition to return true or false",
        )
    } else if message.contains("INVALID_WAIT_SOURCE") || message.contains("SyntaxError") {
        (
            "INVALID_WAIT_SOURCE",
            "invalid wait expression or regular expression",
            "correct the expression syntax; retrying it unchanged cannot succeed",
        )
    } else if message.contains("STALE_REF") {
        (
            "STALE_REF",
            "the wait target no longer belongs to the observed document or node",
            "observe the current page and explicitly choose a new target",
        )
    } else if message.contains("timeout") || message.contains("timed out") {
        (
            "TIMEOUT",
            "wait condition was not confirmed within the remaining request/session budget",
            "inspect the page and condition; a new explicit wait requires remaining session budget",
        )
    } else {
        (
            "WAIT_FAILED",
            "the native wait could not confirm the condition",
            "inspect the session and condition before issuing another operation",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_worker_calls_spend_the_original_deadline() {
        let maximum = Duration::from_millis(100);
        assert_eq!(
            worker_remaining(70_000_000, 100_000_000, maximum),
            Duration::from_millis(30)
        );
        assert_eq!(
            worker_remaining(100_000_000, 100_000_000, maximum),
            Duration::ZERO
        );
        assert_eq!(
            worker_remaining(110_000_000, 100_000_000, maximum),
            Duration::ZERO
        );
        assert_eq!(worker_remaining(0, u64::MAX, maximum), maximum);
        let before = monotonic_ns().unwrap();
        assert!(monotonic_ns().unwrap() >= before);
    }

    #[test]
    fn wait_never_refreshes_or_exceeds_any_budget() {
        let session = Duration::from_secs(10);
        assert_eq!(
            remaining_wait_budget(1000, 800, Duration::from_millis(300), session),
            Duration::from_millis(500)
        );
        assert_eq!(
            remaining_wait_budget(
                1000,
                800,
                Duration::from_millis(300),
                Duration::from_millis(100)
            ),
            Duration::from_millis(100)
        );
        assert_eq!(
            remaining_wait_budget(1000, 800, Duration::from_millis(800), session),
            Duration::ZERO
        );
        assert_eq!(
            remaining_wait_budget(0, 800, Duration::ZERO, session),
            Duration::ZERO
        );
        assert_eq!(
            wait_io_budget(Some(Instant::now()), Duration::from_millis(80)),
            Duration::ZERO
        );
        assert_eq!(
            wait_io_budget(None, Duration::from_millis(80)),
            Duration::from_millis(80)
        );
    }

    #[test]
    fn invalid_conditions_never_recommend_doctor_or_blind_retry() {
        for text in [
            "INVALID_WAIT_PREDICATE",
            "INVALID_WAIT_SOURCE",
            "EvaluationFailure SyntaxError: invalid regex",
        ] {
            let (code, message, next) = wait_error_detail(text);
            assert!(code.starts_with("INVALID_WAIT_"));
            assert!(!message.contains("EvaluationFailure"));
            assert!(!next.contains("doctor"));
            assert!(!next.starts_with("retry"));
        }
        assert_eq!(wait_error_detail("STALE_REF: replaced").0, "STALE_REF");
        assert_eq!(wait_error_detail("timeout: waitForFunction").0, "TIMEOUT");
    }
}
