//! Compact progress for commands that can wait on index work.
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const INITIAL_DELAY: Duration = Duration::from_secs(2);
const STALL_AFTER: Duration = Duration::from_secs(60);
const MIN_FORECAST_SHIFT: Duration = Duration::from_secs(30);

pub(crate) struct QueryProgress {
    stop: Sender<()>,
    thread: Option<JoinHandle<()>>,
}

pub(crate) struct LocalQueryProgress {
    progress: Arc<Mutex<JobProgress>>,
    _reporter: QueryProgress,
}

impl LocalQueryProgress {
    pub(crate) fn start(command: &'static str, state: &str, unit: &str) -> Self {
        let progress = Arc::new(Mutex::new(JobProgress {
            state: state.to_owned(),
            completed: 0,
            total: 0,
            unit: unit.to_owned(),
            pid: None,
            started_at_unix_secs: None,
            rate_milli_spans_per_second: 0,
            eta_unix_secs: None,
        }));
        let observed = Arc::clone(&progress);
        let mut reporter = ProgressReporter::default();
        let thread = QueryProgress::start(INITIAL_DELAY, move |elapsed| {
            let snapshot = observed.lock().ok().map(|progress| progress.clone());
            if let Some(line) =
                reporter.observe_at(command, snapshot, elapsed, crate::unix_now_secs_cli())
            {
                eprintln!("{line}");
            }
        });
        Self {
            progress,
            _reporter: thread,
        }
    }

    pub(crate) fn phase(&self, state: &str, total: usize, unit: &str) {
        if let Ok(mut progress) = self.progress.lock() {
            progress.state = state.to_owned();
            progress.completed = 0;
            progress.total = total as u64;
            progress.unit = unit.to_owned();
        }
    }

    pub(crate) fn completed(&self, completed: usize) {
        if let Ok(mut progress) = self.progress.lock() {
            progress.completed = completed as u64;
        }
    }
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct JobProgress {
    state: String,
    completed: u64,
    total: u64,
    unit: String,
    pid: Option<u64>,
    started_at_unix_secs: Option<u64>,
    rate_milli_spans_per_second: u64,
    eta_unix_secs: Option<u64>,
}

impl JobProgress {
    fn read(path: &std::path::Path) -> Option<Self> {
        let value = crate::read_background_job(path)?;
        let progress = Self::from_value(&value)?;
        let owner = progress.pid.and_then(|pid| u32::try_from(pid).ok())?;
        crate::process_is_alive(owner).then_some(progress)
    }

    fn from_value(value: &serde_json::Value) -> Option<Self> {
        Some(Self {
            state: value.get("state")?.as_str()?.to_owned(),
            completed: value
                .get("completed_spans")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            total: value
                .get("total_spans")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            unit: value
                .get("progress_unit")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("items")
                .to_owned(),
            pid: value.get("pid").and_then(serde_json::Value::as_u64),
            started_at_unix_secs: value
                .get("started_at_unix_secs")
                .and_then(serde_json::Value::as_u64),
            rate_milli_spans_per_second: value
                .get("rate_milli_spans_per_second")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            eta_unix_secs: value
                .get("eta_unix_secs")
                .and_then(serde_json::Value::as_u64),
        })
    }

    fn published_forecast(&self, now_unix_secs: u64) -> Option<(Prognosis, Duration)> {
        if self.state != "embedding" || self.rate_milli_spans_per_second == 0 {
            return None;
        }
        let deadline = self.eta_unix_secs?;
        let prognosis = if deadline <= now_unix_secs {
            Prognosis::EstimateExceeded
        } else {
            Prognosis::Remaining(Duration::from_secs(deadline - now_unix_secs))
        };
        Some((prognosis, Duration::from_secs(deadline)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prognosis {
    Remaining(Duration),
    Complete,
    EstimateExceeded,
}

impl Prognosis {
    fn from_measurement(remaining: u64, completed: u64, elapsed: Duration) -> Option<Self> {
        if remaining == 0 {
            return Some(Self::Complete);
        }
        if completed == 0 || elapsed.is_zero() {
            return None;
        }
        let seconds = remaining
            .saturating_mul(elapsed.as_secs().max(1))
            .div_ceil(completed);
        Some(Self::Remaining(Duration::from_secs(seconds)))
    }

    fn label(self) -> String {
        match self {
            Self::Remaining(remaining) if remaining.as_secs() <= 120 => {
                format!("about {}s", round_up(remaining.as_secs(), 10))
            }
            Self::Remaining(remaining) => {
                format!("about {}m", remaining.as_secs().div_ceil(60))
            }
            Self::Complete => "complete".into(),
            Self::EstimateExceeded => "estimate exceeded".into(),
        }
    }
}

fn round_up(value: u64, quantum: u64) -> u64 {
    value.div_ceil(quantum) * quantum
}

#[derive(Default)]
struct ProgressReporter {
    state: Option<String>,
    pid: Option<u64>,
    started_at_unix_secs: Option<u64>,
    total: u64,
    phase_started_at: Duration,
    phase_started_completed: u64,
    last_completed: u64,
    last_progress_at: Duration,
    prognosis: Option<Prognosis>,
    forecast_deadline: Option<Duration>,
    reported_deadline: Option<Duration>,
    reported_remaining: Option<Duration>,
    stalled: bool,
    missing_reported: bool,
}

impl ProgressReporter {
    fn observe_at(
        &mut self,
        command: &str,
        job: Option<JobProgress>,
        elapsed: Duration,
        now_unix_secs: u64,
    ) -> Option<String> {
        let Some(job) = job else {
            if self.missing_reported {
                return None;
            }
            self.missing_reported = true;
            return Some(format!(
                "greppy: {command} still running; no detailed progress is available"
            ));
        };
        self.missing_reported = false;

        let reset = self.state.as_deref() != Some(job.state.as_str())
            || self.pid != job.pid
            || self.started_at_unix_secs != job.started_at_unix_secs
            || self.total != job.total
            || job.completed < self.last_completed;
        let progress_changed = !reset && job.completed != self.last_completed;
        let mut should_report = reset;

        if reset {
            self.state = Some(job.state.clone());
            self.pid = job.pid;
            self.started_at_unix_secs = job.started_at_unix_secs;
            self.total = job.total;
            self.phase_started_at = elapsed;
            self.phase_started_completed = job.completed;
            self.last_completed = job.completed;
            self.last_progress_at = elapsed;
            let published = job.published_forecast(now_unix_secs);
            self.prognosis = if job.total > 0 && job.completed >= job.total {
                Some(Prognosis::Complete)
            } else {
                published.map(|(prognosis, _)| prognosis)
            };
            self.forecast_deadline = published.map(|(_, deadline)| deadline);
            self.reported_deadline = None;
            self.reported_remaining = None;
            self.stalled = false;
        } else {
            if progress_changed {
                self.last_completed = job.completed;
                self.last_progress_at = elapsed;
            }
            let previous_prognosis = self.prognosis;
            let forecast = if job.state == "embedding" {
                job.published_forecast(now_unix_secs)
            } else if progress_changed {
                job.total.checked_sub(job.completed).and_then(|remaining| {
                    Prognosis::from_measurement(
                        remaining,
                        job.completed.saturating_sub(self.phase_started_completed),
                        elapsed.saturating_sub(self.phase_started_at),
                    )
                    .map(|prognosis| {
                        let deadline = match prognosis {
                            Prognosis::Remaining(remaining) => elapsed.saturating_add(remaining),
                            Prognosis::Complete | Prognosis::EstimateExceeded => elapsed,
                        };
                        (prognosis, deadline)
                    })
                })
            } else {
                None
            };
            if job.state == "embedding" || forecast.is_some() {
                self.prognosis = forecast.map(|(prognosis, _)| prognosis);
                self.forecast_deadline = forecast.map(|(_, deadline)| deadline);
            }
            if let Some((Prognosis::Remaining(_), deadline)) = forecast {
                should_report |= previous_prognosis == Some(Prognosis::EstimateExceeded)
                    || self.forecast_changed_substantially(deadline);
            } else if let Some((prognosis, _)) = forecast {
                should_report |= previous_prognosis != Some(prognosis);
            } else if previous_prognosis.is_some() && self.prognosis.is_none() {
                should_report = true;
            }
            if job.state != "embedding"
                && !progress_changed
                && matches!(self.prognosis, Some(Prognosis::Remaining(_)))
                && self
                    .forecast_deadline
                    .is_some_and(|deadline| elapsed >= deadline)
            {
                self.prognosis = Some(Prognosis::EstimateExceeded);
                should_report = true;
            }
        }

        let active =
            !is_terminal_state(&job.state) && (job.total == 0 || job.completed < job.total);
        let stalled = active && elapsed.saturating_sub(self.last_progress_at) >= STALL_AFTER;
        if stalled != self.stalled {
            self.stalled = stalled;
            should_report = true;
        }
        if !should_report {
            return None;
        }

        if let Some(Prognosis::Remaining(remaining)) = self.prognosis {
            self.reported_deadline = self.forecast_deadline;
            self.reported_remaining = Some(remaining);
        }

        let progress = if job.total > 0 {
            format!("{}/{} {}", job.completed, job.total, job.unit)
        } else if job.completed == 0 {
            format!("{} total unknown", job.unit)
        } else {
            format!("{} {}", job.completed, job.unit)
        };
        let prognosis = if stalled {
            format!("no progress reported for {}s", STALL_AFTER.as_secs())
        } else if job.total == 0 {
            "phase ETA unavailable until total is known".into()
        } else if let Some(prognosis) = self.prognosis {
            format!("phase ETA {}", prognosis.label())
        } else {
            "measuring phase ETA".into()
        };
        let pid = job
            .pid
            .map(|pid| format!("; pid {pid}"))
            .unwrap_or_default();
        Some(format!(
            "greppy: {command} — {}: {progress}; {prognosis}{pid}",
            job.state
        ))
    }

    #[cfg(test)]
    fn observe(
        &mut self,
        command: &str,
        job: Option<JobProgress>,
        elapsed: Duration,
    ) -> Option<String> {
        self.observe_at(command, job, elapsed, 0)
    }

    fn forecast_changed_substantially(&self, deadline: Duration) -> bool {
        let (Some(previous_deadline), Some(previous_remaining)) =
            (self.reported_deadline, self.reported_remaining)
        else {
            return true;
        };
        let shift = deadline.abs_diff(previous_deadline);
        let threshold = MIN_FORECAST_SHIFT.max(previous_remaining / 4);
        shift >= threshold
    }
}

fn is_terminal_state(state: &str) -> bool {
    matches!(state, "complete" | "completed" | "failed" | "degraded")
}

pub(crate) fn for_command(
    command: Option<&crate::Command>,
    configured_root: Option<&str>,
) -> Option<QueryProgress> {
    use crate::Command;
    let (name, root_hint) = match command? {
        Command::Index {
            path,
            recovery_path,
            ..
        } => {
            let operand = match path.as_deref() {
                Some("status") => None,
                Some("recover") => recovery_path.as_deref(),
                path => path,
            };
            ("index", configured_root.or(operand))
        }
        Command::SearchGraph { .. } => ("search-graph", configured_root),
        Command::SearchSymbol { .. } => ("search-symbol", configured_root),
        Command::Search { .. } => ("search", configured_root),
        Command::Context { .. } => ("context", configured_root),
        Command::WhoCalls { .. } => ("who-calls", configured_root),
        Command::Callees { .. } => ("callees", configured_root),
        Command::Impact { .. } => ("impact", configured_root),
        Command::Path { .. } => ("path", configured_root),
        Command::Brief { .. } => ("brief", configured_root),
        Command::Trace { .. } => ("trace", configured_root),
        Command::Read { .. } => ("read", configured_root),
        Command::ReadSmart { .. } => ("read-smart", configured_root),
        Command::WhereAmI { .. } => ("where-am-i", configured_root),
        _ => return None,
    };
    let job_path = crate::resolve_root(root_hint)
        .ok()
        .map(|root| crate::background_job_path(&root));
    let mut reporter = ProgressReporter::default();
    Some(QueryProgress::start(INITIAL_DELAY, move |elapsed| {
        let job = job_path.as_deref().and_then(JobProgress::read);
        if let Some(line) = reporter.observe_at(name, job, elapsed, crate::unix_now_secs_cli()) {
            eprintln!("{line}");
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(state: &str, completed: u64, total: u64) -> JobProgress {
        JobProgress {
            state: state.into(),
            completed,
            total,
            unit: "spans".into(),
            pid: Some(42),
            started_at_unix_secs: Some(1_000),
            rate_milli_spans_per_second: 0,
            eta_unix_secs: None,
        }
    }

    #[test]
    fn phase_eta_uses_only_progress_measured_in_that_phase() {
        let mut reporter = ProgressReporter::default();
        assert!(reporter
            .observe(
                "search",
                Some(job("indexing", 400, 1000)),
                Duration::from_secs(2)
            )
            .unwrap()
            .contains("measuring phase ETA"));

        let update = reporter
            .observe(
                "search",
                Some(job("indexing", 500, 1000)),
                Duration::from_secs(12),
            )
            .unwrap();
        assert!(update.contains("phase ETA about 50s"), "{update}");

        let new_phase = reporter
            .observe(
                "search",
                Some(job("writing", 500, 1000)),
                Duration::from_secs(13),
            )
            .unwrap();
        assert!(new_phase.contains("measuring phase ETA"), "{new_phase}");
    }

    #[test]
    fn published_measured_eta_is_parsed_and_aged_from_its_deadline() {
        let value = serde_json::json!({
            "state": "embedding",
            "completed_spans": 400,
            "total_spans": 1000,
            "progress_unit": "spans",
            "pid": 42,
            "started_at_unix_secs": 900,
            "rate_milli_spans_per_second": 1_000,
            "eta_seconds": 100,
            "eta_unix_secs": 1_100
        });
        let parsed = JobProgress::from_value(&value).unwrap();
        assert_eq!(parsed.started_at_unix_secs, Some(900));
        assert_eq!(
            parsed.published_forecast(1_040),
            Some((
                Prognosis::Remaining(Duration::from_secs(60)),
                Duration::from_secs(1_100)
            ))
        );

        let mut reporter = ProgressReporter::default();
        let line = reporter
            .observe_at(
                "search",
                Some(parsed.clone()),
                Duration::from_secs(2),
                1_040,
            )
            .unwrap();
        assert!(line.contains("phase ETA about 60s"), "{line}");
        assert!(reporter
            .observe_at("search", Some(parsed), Duration::from_secs(12), 1_050)
            .is_none());

        let mut unmeasured = JobProgress::from_value(&value).unwrap();
        unmeasured.rate_milli_spans_per_second = 0;
        let mut reporter = ProgressReporter::default();
        let line = reporter
            .observe_at("search", Some(unmeasured), Duration::from_secs(2), 1_040)
            .unwrap();
        assert!(line.contains("measuring phase ETA"), "{line}");
    }

    #[test]
    fn completed_phase_is_reported_once() {
        let mut reporter = ProgressReporter::default();
        let completed = job("complete", 100, 100);
        let first = reporter
            .observe("search", Some(completed.clone()), Duration::from_secs(2))
            .unwrap();
        assert!(first.contains("phase ETA complete"), "{first}");
        assert!(reporter
            .observe("search", Some(completed), Duration::from_secs(4))
            .is_none());
    }

    #[test]
    fn expired_published_eta_reports_exceeded_once_until_forecast_changes() {
        let mut expired = job("embedding", 400, 1000);
        expired.rate_milli_spans_per_second = 1_000;
        expired.eta_unix_secs = Some(1_000);

        let mut reporter = ProgressReporter::default();
        let first = reporter
            .observe_at(
                "search",
                Some(expired.clone()),
                Duration::from_secs(2),
                1_010,
            )
            .unwrap();
        assert!(first.contains("phase ETA estimate exceeded"), "{first}");
        assert!(reporter
            .observe_at(
                "search",
                Some(expired.clone()),
                Duration::from_secs(4),
                1_012,
            )
            .is_none());

        expired.completed = 450;
        expired.eta_unix_secs = Some(1_100);
        let renewed = reporter
            .observe_at("search", Some(expired), Duration::from_secs(6), 1_020)
            .unwrap();
        assert!(renewed.contains("phase ETA about 80s"), "{renewed}");
    }

    #[test]
    fn steady_deadline_is_suppressed_but_material_forecast_shift_is_reported() {
        let mut reporter = ProgressReporter::default();
        assert!(reporter
            .observe(
                "search",
                Some(job("indexing", 0, 100)),
                Duration::from_secs(0)
            )
            .is_some());
        assert!(reporter
            .observe(
                "search",
                Some(job("indexing", 10, 100)),
                Duration::from_secs(10)
            )
            .is_some());
        assert!(reporter
            .observe(
                "search",
                Some(job("indexing", 20, 100)),
                Duration::from_secs(20)
            )
            .is_none());

        let shifted = reporter
            .observe(
                "search",
                Some(job("indexing", 21, 100)),
                Duration::from_secs(50),
            )
            .unwrap();
        assert!(shifted.contains("phase ETA"), "{shifted}");
    }

    #[test]
    fn zero_total_stalls_once_and_resume_or_identity_change_resets_measurement() {
        let mut reporter = ProgressReporter::default();
        let initial = reporter
            .observe(
                "search",
                Some(job("counting", 0, 0)),
                Duration::from_secs(2),
            )
            .unwrap();
        assert!(initial.contains("spans total unknown"), "{initial}");
        assert!(
            initial.contains("phase ETA unavailable until total is known"),
            "{initial}"
        );
        let stalled = reporter
            .observe(
                "search",
                Some(job("counting", 0, 0)),
                Duration::from_secs(62),
            )
            .unwrap();
        assert!(
            stalled.contains("no progress reported for 60s"),
            "{stalled}"
        );
        assert!(reporter
            .observe(
                "search",
                Some(job("counting", 0, 0)),
                Duration::from_secs(122)
            )
            .is_none());

        let mut replacement = job("counting", 0, 0);
        replacement.started_at_unix_secs = Some(1_001);
        let reset = reporter
            .observe("search", Some(replacement), Duration::from_secs(123))
            .unwrap();
        assert!(reset.contains("spans total unknown"), "{reset}");
        assert!(
            reset.contains("phase ETA unavailable until total is known"),
            "{reset}"
        );
    }

    #[test]
    fn missing_job_status_is_honest_and_emitted_once() {
        let mut reporter = ProgressReporter::default();
        let first = reporter.observe("search", None, Duration::ZERO).unwrap();
        assert!(
            first.contains("no detailed progress is available"),
            "{first}"
        );
        assert!(reporter
            .observe("search", None, Duration::from_secs(30))
            .is_none());
    }

    #[test]
    fn fast_query_drop_joins_reporter_without_emitting() {
        let (tx, rx) = mpsc::channel();
        let guard = QueryProgress::start(Duration::from_secs(60), move |_| {
            let _ = tx.send(());
        });
        drop(guard);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn unknown_local_total_is_explicit() {
        let mut reporter = ProgressReporter::default();
        let mut discovering = job("discovering_files", 0, 0);
        discovering.unit = "files".into();
        let line = reporter
            .observe("search-pattern", Some(discovering), Duration::from_secs(2))
            .unwrap();
        assert!(line.contains("files total unknown"), "{line}");
        assert!(
            line.contains("phase ETA unavailable until total is known"),
            "{line}"
        );
    }

    #[test]
    fn measured_local_forecast_expires_once_and_recovers_on_progress() {
        let mut reporter = ProgressReporter::default();
        assert!(reporter
            .observe(
                "search-pattern",
                Some(job("scanning_files", 0, 256)),
                Duration::from_secs(2),
            )
            .is_some());

        let measured = reporter
            .observe(
                "search-pattern",
                Some(job("scanning_files", 128, 256)),
                Duration::from_secs(4),
            )
            .unwrap();
        assert!(measured.contains("phase ETA about 10s"), "{measured}");

        let exceeded = reporter
            .observe(
                "search-pattern",
                Some(job("scanning_files", 128, 256)),
                Duration::from_secs(8),
            )
            .unwrap();
        assert!(
            exceeded.contains("phase ETA estimate exceeded"),
            "{exceeded}"
        );
        assert!(reporter
            .observe(
                "search-pattern",
                Some(job("scanning_files", 128, 256)),
                Duration::from_secs(10),
            )
            .is_none());

        let resumed = reporter
            .observe(
                "search-pattern",
                Some(job("scanning_files", 192, 256)),
                Duration::from_secs(12),
            )
            .unwrap();
        assert!(resumed.contains("phase ETA about 10s"), "{resumed}");
    }

    #[test]
    fn local_progress_tracks_real_phase_counters() {
        let progress = LocalQueryProgress::start("search-pattern", "discovering_files", "files");
        progress.phase("scanning_files", 257, "files");
        progress.completed(128);

        let snapshot = progress.progress.lock().unwrap().clone();
        assert_eq!(snapshot.state, "scanning_files");
        assert_eq!(snapshot.completed, 128);
        assert_eq!(snapshot.total, 257);
        assert_eq!(snapshot.unit, "files");
    }

    #[test]
    fn live_pattern_search_does_not_subscribe_to_graph_index_progress() {
        let cli =
            <crate::Cli as clap::Parser>::try_parse_from(["greppy", "search-pattern", "needle"])
                .unwrap();
        assert!(for_command(cli.command.as_ref(), None).is_none());
    }

    #[test]
    fn job_progress_requires_a_live_owner() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("index.job");
        let value = |pid| {
            serde_json::json!({
                "state": "classifying_files",
                "completed_spans": 1,
                "total_spans": 2,
                "progress_unit": "files",
                "pid": pid,
                "started_at_unix_secs": 1
            })
        };

        crate::write_background_job(&path, &value(std::process::id())).unwrap();
        assert!(JobProgress::read(&path).is_some());

        // u32::MAX is outside the process-id range supported by our target
        // platforms, so this record cannot identify a live job owner.
        crate::write_background_job(&path, &value(u32::MAX)).unwrap();
        assert!(JobProgress::read(&path).is_none());
    }
}
