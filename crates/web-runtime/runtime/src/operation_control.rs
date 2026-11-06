use std::sync::Mutex;

type OperationId = (String, String);

#[derive(Default)]
struct State {
    active: Option<OperationId>,
    cancel_requested: bool,
    // Diagnostic receipt for the most recently finished operation, not a history.
    completed: Option<OperationId>,
}

#[derive(Default)]
pub(crate) struct OperationControl {
    state: Mutex<State>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum CancelResult {
    Requested,
    AlreadyCompleted,
    NoMatch,
}

impl OperationControl {
    pub(crate) fn begin(
        &self,
        session: &str,
        request: &str,
    ) -> Result<OperationLease<'_>, &'static str> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.active.is_some() {
            return Err("another web.run is active");
        }
        state.active = Some((session.to_owned(), request.to_owned()));
        state.cancel_requested = false;
        Ok(OperationLease { control: self })
    }

    pub(crate) fn cancel(&self, session: &str, request: &str) -> CancelResult {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let matches = |pair: &OperationId| pair.0 == session && pair.1 == request;
        // A reused request ID belongs to the current operation, even if the
        // previous completion receipt had exactly the same ID.
        if state.active.as_ref().is_some_and(matches) {
            state.cancel_requested = true;
            CancelResult::Requested
        } else if state.completed.as_ref().is_some_and(matches) {
            CancelResult::AlreadyCompleted
        } else {
            CancelResult::NoMatch
        }
    }

    pub(crate) fn is_cancelled(&self, session: &str, request: &str) -> bool {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.cancel_requested
            && state
                .active
                .as_ref()
                .is_some_and(|pair| pair.0 == session && pair.1 == request)
    }
}

pub(crate) struct OperationLease<'a> {
    control: &'a OperationControl,
}

impl Drop for OperationLease<'_> {
    fn drop(&mut self) {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(pair) = state.active.take() {
            // Replacing the receipt drops both old Strings on every completion.
            state.completed = Some(pair);
        }
        state.cancel_requested = false;
    }
}

#[cfg(test)]
mod tests {
    use super::{CancelResult, OperationControl};

    #[test]
    fn finished_runs_are_not_active_and_old_cancellation_cannot_reach_a_new_run() {
        let control = OperationControl::default();
        {
            let _lease = control.begin("session-a", "request-1").unwrap();
            assert!(control.begin("session-b", "request-2").is_err());
            assert_eq!(
                control.cancel("session-b", "request-1"),
                CancelResult::NoMatch
            );
            assert_eq!(control.cancel("session-a", "wrong"), CancelResult::NoMatch);
            assert!(!control.is_cancelled("session-a", "request-1"));
            assert_eq!(
                control.cancel("session-a", "request-1"),
                CancelResult::Requested
            );
            assert!(control.is_cancelled("session-a", "request-1"));
        }
        assert_eq!(
            control.cancel("session-a", "request-1"),
            CancelResult::AlreadyCompleted,
            "finished operation remained active or cancellable"
        );
        assert!(!control.is_cancelled("session-a", "request-1"));
        {
            let _lease = control.begin("session-a", "request-1").unwrap();
            assert!(!control.is_cancelled("session-a", "request-1"));
            assert_eq!(
                control.cancel("session-a", "request-1"),
                CancelResult::Requested,
                "old completion receipt blocked cancellation of a current reused ID"
            );
        }
        fn fail_after_start(control: &OperationControl) -> Result<(), ()> {
            let _lease = control.begin("session-b", "request-2").unwrap();
            assert_eq!(
                control.cancel("session-a", "request-1"),
                CancelResult::AlreadyCompleted
            );
            assert!(!control.is_cancelled("session-b", "request-2"));
            Err(())
        }
        assert!(fail_after_start(&control).is_err());
        assert_eq!(
            control.cancel("session-b", "request-2"),
            CancelResult::AlreadyCompleted,
            "error return left an active operation behind"
        );
        assert_eq!(
            control.cancel("session-a", "request-1"),
            CancelResult::NoMatch,
            "older receipts must expire without claiming a cancellation"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn completion_receipts_release_private_memory_after_cancelled_runs() {
        const CHILD: &str = "GREPPY_COMPLETION_MEMORY_CONTRACT_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "operation_control::tests::completion_receipts_release_private_memory_after_cancelled_runs",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "fresh-process private-memory contract failed:\n{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            print!("{}", String::from_utf8_lossy(&result.stdout));
            return;
        }

        fn private_bytes() -> u64 {
            let rollup = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap();
            ["Private_Clean:", "Private_Dirty:", "Private_Hugetlb:"]
                .iter()
                .map(|key| {
                    rollup
                        .lines()
                        .find_map(|line| {
                            let mut fields = line.split_whitespace();
                            (fields.next() == Some(*key))
                                .then(|| fields.next().unwrap().parse::<u64>().unwrap() * 1024)
                        })
                        .unwrap_or(0)
                })
                .sum()
        }

        let control = OperationControl::default();
        let suffix = "r".repeat(16 * 1024);
        let complete_cancelled = |index| {
            let request = format!("{index}-{suffix}");
            let lease = control.begin("owned-session", &request).unwrap();
            assert_eq!(
                control.cancel("owned-session", &request),
                CancelResult::Requested
            );
            drop(lease);
        };
        for index in 0..32 {
            complete_cancelled(index);
        }
        let before = private_bytes();
        for index in 32..4128 {
            complete_cancelled(index);
        }
        let growth = private_bytes().saturating_sub(before);
        println!("completion_receipt_private_growth_bytes={growth}");
        assert!(
            growth < 8 * 1024 * 1024,
            "completed cancellation receipts retained private memory: {growth} bytes"
        );
        assert_eq!(
            control.cancel("owned-session", &format!("4127-{suffix}")),
            CancelResult::AlreadyCompleted
        );
    }
}
