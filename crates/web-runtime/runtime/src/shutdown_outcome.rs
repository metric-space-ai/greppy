//! Keep the failing shutdown stage rather than attributing every escalation to
//! an elapsed EOF budget. Clock/poll hooks make the existing ordering testable.
use std::{fmt, io};

#[derive(Debug)]
pub(super) enum Failure {
    Send(io::Error),
    Wait(io::Error),
    EofDeadline,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Send(error) => write!(f, "shutdown send failed: {error}"),
            Self::Wait(error) => write!(f, "shutdown wait failed: {error}"),
            Self::EofDeadline => f.write_str("eof timeout"),
        }
    }
}

pub(super) fn await_exit(
    sent: io::Result<()>,
    mut before_deadline: impl FnMut() -> bool,
    mut poll_exit: impl FnMut() -> io::Result<bool>,
    mut pause: impl FnMut(),
) -> Result<(), Failure> {
    sent.map_err(Failure::Send)?;
    while before_deadline() {
        if poll_exit().map_err(Failure::Wait)? {
            return Ok(());
        }
        pause();
    }
    Err(Failure::EofDeadline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_send_never_checks_the_clock_or_polls_and_preserves_error() {
        let error = await_exit(
            Err(io::Error::from(io::ErrorKind::BrokenPipe)),
            || panic!("a send failure is not an EOF wait"),
            || panic!("must not poll after failed send"),
            || panic!("must not pause after failed send"),
        )
        .unwrap_err();
        assert!(
            matches!(&error, Failure::Send(source) if source.kind() == io::ErrorKind::BrokenPipe)
        );
        assert!(error.to_string().contains("shutdown send failed"));
        assert!(!error.to_string().contains("eof timeout"));
    }

    #[test]
    fn timed_out_send_is_not_mislabelled_as_eof_deadline() {
        let error = await_exit(
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "protocol not writable",
            )),
            || false,
            || panic!("no poll"),
            || panic!("no pause"),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("shutdown send failed: protocol not writable"));
        assert!(!error.to_string().contains("eof timeout"));
    }

    #[test]
    fn poll_failure_preserves_its_stage_without_pausing() {
        let error = await_exit(
            Ok(()),
            || true,
            || Err(io::Error::new(io::ErrorKind::Other, "waitpid failed")),
            || panic!("no pause after wait failure"),
        )
        .unwrap_err();
        assert!(matches!(&error, Failure::Wait(_)));
        assert_eq!(error.to_string(), "shutdown wait failed: waitpid failed");
    }

    #[test]
    fn only_an_elapsed_eof_budget_produces_eof_timeout() {
        let error = await_exit(
            Ok(()),
            || false,
            || panic!("expired budget must not poll"),
            || panic!("expired budget must not pause"),
        )
        .unwrap_err();
        assert!(matches!(error, Failure::EofDeadline));
        assert_eq!(error.to_string(), "eof timeout");
    }

    #[test]
    fn successful_exit_does_not_pause_or_request_another_poll() {
        let mut polls = 0;
        let result = await_exit(
            Ok(()),
            || true,
            || {
                polls += 1;
                Ok(true)
            },
            || panic!("already exited"),
        );
        assert!(result.is_ok());
        assert_eq!(polls, 1);
    }

    #[test]
    fn waiting_retains_the_existing_poll_then_pause_order() {
        use std::cell::RefCell;
        let events = RefCell::new(Vec::new());
        let mut ticks = 0;
        let result = await_exit(
            Ok(()),
            || {
                ticks += 1;
                events.borrow_mut().push("clock");
                ticks < 3
            },
            || {
                events.borrow_mut().push("poll");
                Ok(false)
            },
            || events.borrow_mut().push("pause"),
        );
        assert!(matches!(result, Err(Failure::EofDeadline)));
        assert_eq!(
            *events.borrow(),
            ["clock", "poll", "pause", "clock", "poll", "pause", "clock"]
        );
    }
}
