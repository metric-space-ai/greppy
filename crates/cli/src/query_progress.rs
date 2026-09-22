//! A query must remain observable even while blocked below its handler.
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub(crate) struct QueryProgress {
    stop: Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl QueryProgress {
    fn start(interval: Duration, mut report: impl FnMut(Duration) + Send + 'static) -> Self {
        let (stop, receiver) = mpsc::channel();
        let started = Instant::now();
        let thread = std::thread::Builder::new()
            .name("greppy-query-progress".into())
            .spawn(move || {
                while let Err(RecvTimeoutError::Timeout) = receiver.recv_timeout(interval) {
                    report(started.elapsed());
                }
            })
            .ok();
        Self { stop, thread }
    }
}

impl Drop for QueryProgress {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(crate) fn for_command(command: Option<&crate::Command>) -> Option<QueryProgress> {
    use crate::Command;
    let name = match command? {
        Command::Index { .. } => "index",
        Command::SearchGraph { .. } => "search-graph",
        Command::SearchSymbol { .. } => "search-symbol",
        Command::SearchPattern { .. } => "search-pattern",
        Command::Search { .. } => "search",
        Command::Context { .. } => "context",
        Command::WhoCalls { .. } => "who-calls",
        Command::Callees { .. } => "callees",
        Command::Impact { .. } => "impact",
        Command::Path { .. } => "path",
        Command::Brief { .. } => "brief",
        Command::Trace { .. } => "trace",
        Command::Read { .. } => "read",
        Command::ReadSmart { .. } => "read-smart",
        Command::WhereAmI { .. } => "where-am-i",
        _ => return None,
    };
    Some(QueryProgress::start(
        Duration::from_secs(5),
        move |elapsed| {
            // stderr preserves JSON and grep-compatible stdout contracts. Do not
            // guess the blocking phase, index progress, or a completion estimate.
            eprintln!(
            "greppy: {name} still running ({}s elapsed); completion time unknown; next update in 5s. This is not a completed result.",
            elapsed.as_secs()
        );
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_query_reports_repeatedly_and_stops_before_drop_returns() {
        let (tx, rx) = mpsc::channel();
        let guard = QueryProgress::start(Duration::from_millis(5), move |elapsed| {
            let _ = tx.send(elapsed);
        });
        let first = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let second = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(second >= first);
        drop(guard);
        while rx.try_recv().is_ok() {}
        assert!(rx.recv().is_err());
    }

    #[test]
    fn fast_query_does_not_wait_for_or_emit_a_heartbeat() {
        let (tx, rx) = mpsc::channel();
        let guard = QueryProgress::start(Duration::from_secs(60), move |_| {
            let _ = tx.send(());
        });
        drop(guard);
        assert!(rx.recv().is_err());
    }
}
