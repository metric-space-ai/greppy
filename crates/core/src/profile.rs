//! Activatable scoped phase/sub-phase timing.
//!
//! Ported from `.vendor/codebase-memory-mcp.git/src/foundation/profile.c`
//! (`cbm_profile_init`, `cbm_profile_enable`, `cbm_profile_now`,
//! `cbm_profile_log_elapsed`, and the `CBM_PROF_*` macros).
//!
//! Profiling is gated by an env flag. When disabled (the default), creating
//! and dropping spans costs a single atomic load plus a branch — effectively
//! zero overhead and, crucially, **no log output**. Enable it by setting
//! `GREPPLUS_PROFILE` to any non-empty value other than `"0"`, mirroring the
//! upstream `CBM_PROFILE` semantics.
//!
//! Two ways to use it:
//! - [`Profiler`] — an explicit recorder that captures named spans and their
//!   elapsed durations into a buffer you can inspect (`Profiler::spans`,
//!   `Profiler::report`). Always records, independent of the env flag; useful
//!   for tests and structured aggregation.
//! - [`span`] / [`Span`] — a scoped RAII timer that, on drop, emits a single
//!   structured `tracing` line **only when profiling is active**. This is the
//!   direct analogue of the `CBM_PROF_START` / `CBM_PROF_END` macros.
//!
//! Everything here is `std::time` only — no external crates.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Runtime-active flag. Set once by [`init`] from the env, or forced on by
/// [`enable`]. Defaults to `false` (profiling off).
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Name of the environment variable that activates env-gated profiling.
pub const PROFILE_ENV: &str = "GREPPLUS_PROFILE";

/// Initialise env-gated profiling by reading [`PROFILE_ENV`].
///
/// Profiling is activated when the variable is present and its value is
/// neither empty nor `"0"` (matching the upstream `CBM_PROFILE` rule). Safe
/// to call multiple times; the flag is only ever turned on here, never off.
///
/// Returns the resulting active state.
pub fn init() -> bool {
    let on = match std::env::var(PROFILE_ENV) {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    };
    if on {
        ACTIVE.store(true, Ordering::Relaxed);
    }
    is_active()
}

/// Force-enable profiling at runtime (e.g. behind a CLI `--profile` flag),
/// independent of the environment. Mirrors `cbm_profile_enable`.
pub fn enable() {
    ACTIVE.store(true, Ordering::Relaxed);
}

/// Whether env-gated profiling output is currently active.
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// A single recorded span: a phase, an optional sub-phase, the elapsed
/// duration, and an optional item count used to compute a throughput rate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSpan {
    /// Coarse phase name (e.g. `"index"`).
    pub phase: String,
    /// Optional finer sub-phase name (e.g. `"parse"`). Empty when unused.
    pub sub: String,
    /// Wall-clock time elapsed for this span.
    pub elapsed: Duration,
    /// Optional number of items processed; `0` means "not provided".
    pub items: u64,
}

impl ProfileSpan {
    /// Items processed per second, or `None` when no items were recorded or
    /// the elapsed time was zero (rate undefined).
    pub fn rate_per_s(&self) -> Option<f64> {
        let secs = self.elapsed.as_secs_f64();
        if self.items == 0 || secs <= 0.0 {
            None
        } else {
            Some(self.items as f64 / secs)
        }
    }
}

/// An explicit profiler that records named spans into an in-memory buffer.
///
/// Unlike the env-gated [`span`] helper, a `Profiler` **always** records — it
/// is the structured, inspectable side of the API. Callers drive it with
/// [`Profiler::start`] (returning a handle) and [`Profiler::record`] (paired
/// with the handle) or the convenience [`Profiler::time`] wrapper.
#[derive(Debug, Default)]
pub struct Profiler {
    spans: Vec<ProfileSpan>,
}

/// Opaque handle returned by [`Profiler::start`], capturing the start instant.
#[derive(Debug, Clone, Copy)]
pub struct Started {
    at: Instant,
}

impl Profiler {
    /// Create an empty profiler.
    pub fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// Begin timing. Pair the returned handle with [`Profiler::record`] or
    /// [`Profiler::record_n`].
    pub fn start(&self) -> Started {
        Started { at: Instant::now() }
    }

    /// Record a completed span (no item count) given a handle from
    /// [`Profiler::start`]. Returns the elapsed duration.
    pub fn record(&mut self, phase: &str, sub: &str, started: Started) -> Duration {
        self.record_n(phase, sub, started, 0)
    }

    /// Record a completed span with an item count for rate computation.
    /// Returns the elapsed duration.
    pub fn record_n(&mut self, phase: &str, sub: &str, started: Started, items: u64) -> Duration {
        let elapsed = started.at.elapsed();
        self.spans.push(ProfileSpan {
            phase: phase.to_string(),
            sub: sub.to_string(),
            elapsed,
            items,
        });
        elapsed
    }

    /// Time a closure, recording one span around it. Returns the closure's
    /// value. Convenience over `start`/`record`.
    pub fn time<T>(&mut self, phase: &str, sub: &str, f: impl FnOnce() -> T) -> T {
        let started = self.start();
        let out = f();
        self.record(phase, sub, started);
        out
    }

    /// All spans recorded so far, in record order.
    pub fn spans(&self) -> &[ProfileSpan] {
        &self.spans
    }

    /// Number of spans recorded.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether no spans have been recorded.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Discard all recorded spans.
    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// Total elapsed across all recorded spans.
    pub fn total(&self) -> Duration {
        self.spans.iter().map(|s| s.elapsed).sum()
    }

    /// Render the recorded spans as upstream-style structured lines, one per
    /// span, e.g. `prof phase=index sub=parse ms=12 us=12345 items=900 rate_per_s=72900`.
    /// Returns the lines without emitting them, so callers control output.
    pub fn report(&self) -> Vec<String> {
        self.spans.iter().map(format_span).collect()
    }

    /// Emit every recorded span through `tracing` at info level, but only when
    /// env-gated profiling [`is_active`]. A no-op otherwise.
    pub fn emit(&self) {
        if !is_active() {
            return;
        }
        for s in &self.spans {
            emit_span(s);
        }
    }
}

/// A scoped RAII timer. On drop it emits a single structured `tracing` line
/// **iff** profiling [`is_active`]; otherwise the drop is a no-op beyond the
/// atomic-load check. Direct analogue of `CBM_PROF_START` + `CBM_PROF_END`.
#[derive(Debug)]
pub struct Span {
    phase: String,
    sub: String,
    items: u64,
    start: Instant,
}

/// Begin a scoped span for `phase`/`sub`. The returned [`Span`] emits on drop
/// when profiling is active. Cheap when inactive (no allocation is avoided,
/// but no output occurs).
pub fn span(phase: &str, sub: &str) -> Span {
    Span {
        phase: phase.to_string(),
        sub: sub.to_string(),
        items: 0,
        start: Instant::now(),
    }
}

impl Span {
    /// Attach an item count so the emitted line includes a throughput rate.
    pub fn with_items(mut self, items: u64) -> Self {
        self.items = items;
        self
    }

    /// Set or update the item count in place.
    pub fn set_items(&mut self, items: u64) {
        self.items = items;
    }

    /// Elapsed time since this span began, without consuming it.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if !is_active() {
            return;
        }
        let s = ProfileSpan {
            phase: std::mem::take(&mut self.phase),
            sub: std::mem::take(&mut self.sub),
            elapsed: self.start.elapsed(),
            items: self.items,
        };
        emit_span(&s);
    }
}

fn format_span(s: &ProfileSpan) -> String {
    let us = s.elapsed.as_micros();
    let ms = us / 1000;
    match (s.items, s.rate_per_s()) {
        (items, Some(rate)) if items > 0 => format!(
            "prof phase={} sub={} ms={} us={} items={} rate_per_s={}",
            s.phase, s.sub, ms, us, items, rate as u64
        ),
        (items, _) if items > 0 => format!(
            "prof phase={} sub={} ms={} us={} items={}",
            s.phase, s.sub, ms, us, items
        ),
        _ => format!("prof phase={} sub={} ms={} us={}", s.phase, s.sub, ms, us),
    }
}

fn emit_span(s: &ProfileSpan) {
    let us = s.elapsed.as_micros() as u64;
    let ms = us / 1000;
    match s.rate_per_s() {
        Some(rate) if s.items > 0 => tracing::info!(
            target: "prof",
            phase = %s.phase,
            sub = %s.sub,
            ms,
            us,
            items = s.items,
            rate_per_s = rate as u64,
            "prof"
        ),
        _ if s.items > 0 => tracing::info!(
            target: "prof",
            phase = %s.phase,
            sub = %s.sub,
            ms,
            us,
            items = s.items,
            "prof"
        ),
        _ => tracing::info!(
            target: "prof",
            phase = %s.phase,
            sub = %s.sub,
            ms,
            us,
            "prof"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialise tests that mutate the process-global ACTIVE flag / env var so
    // they do not race each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn profiler_records_spans_in_order() {
        let mut p = Profiler::new();
        assert!(p.is_empty());
        let s1 = p.start();
        p.record("a", "x", s1);
        let s2 = p.start();
        p.record_n("b", "y", s2, 10);

        assert_eq!(p.len(), 2);
        assert_eq!(p.spans()[0].phase, "a");
        assert_eq!(p.spans()[0].sub, "x");
        assert_eq!(p.spans()[1].phase, "b");
        assert_eq!(p.spans()[1].items, 10);
    }

    #[test]
    fn profiler_time_records_nonzero_elapsed() {
        let mut p = Profiler::new();
        let out = p.time("phase", "sub", || {
            // Busy-spin until the monotonic clock advances so elapsed > 0
            // without relying on a (forbidden) foreground sleep.
            let start = Instant::now();
            while start.elapsed().is_zero() {
                std::hint::spin_loop();
            }
            42
        });
        assert_eq!(out, 42);
        assert_eq!(p.len(), 1);
        assert!(
            p.spans()[0].elapsed > Duration::ZERO,
            "expected a positive elapsed duration"
        );
    }

    #[test]
    fn total_sums_all_spans() {
        let mut p = Profiler::new();
        // Inject deterministic spans directly to assert on the arithmetic.
        p.spans.push(ProfileSpan {
            phase: "a".into(),
            sub: String::new(),
            elapsed: Duration::from_micros(100),
            items: 0,
        });
        p.spans.push(ProfileSpan {
            phase: "b".into(),
            sub: String::new(),
            elapsed: Duration::from_micros(250),
            items: 0,
        });
        assert_eq!(p.total(), Duration::from_micros(350));
    }

    #[test]
    fn rate_per_s_is_none_without_items_or_time() {
        let no_items = ProfileSpan {
            phase: "p".into(),
            sub: String::new(),
            elapsed: Duration::from_secs(1),
            items: 0,
        };
        assert_eq!(no_items.rate_per_s(), None);

        let zero_time = ProfileSpan {
            phase: "p".into(),
            sub: String::new(),
            elapsed: Duration::ZERO,
            items: 100,
        };
        assert_eq!(zero_time.rate_per_s(), None);
    }

    #[test]
    fn rate_per_s_computes_throughput() {
        let s = ProfileSpan {
            phase: "p".into(),
            sub: String::new(),
            elapsed: Duration::from_secs(2),
            items: 1000,
        };
        // 1000 items over 2s == 500/s.
        assert_eq!(s.rate_per_s(), Some(500.0));
    }

    #[test]
    fn report_lines_match_expected_shapes() {
        let mut p = Profiler::new();
        p.spans.push(ProfileSpan {
            phase: "index".into(),
            sub: "parse".into(),
            elapsed: Duration::from_micros(12_345),
            items: 0,
        });
        p.spans.push(ProfileSpan {
            phase: "index".into(),
            sub: "write".into(),
            elapsed: Duration::from_secs(2),
            items: 1000,
        });
        let lines = p.report();
        assert_eq!(lines[0], "prof phase=index sub=parse ms=12 us=12345");
        assert_eq!(
            lines[1],
            "prof phase=index sub=write ms=2000 us=2000000 items=1000 rate_per_s=500"
        );
    }

    #[test]
    fn env_gating_off_by_default_then_init_activates() {
        let _g = ENV_LOCK.lock().unwrap();
        ACTIVE.store(false, Ordering::Relaxed);

        // Empty / "0" / unset must NOT activate.
        std::env::remove_var(PROFILE_ENV);
        assert!(!init());
        std::env::set_var(PROFILE_ENV, "");
        assert!(!init());
        std::env::set_var(PROFILE_ENV, "0");
        assert!(!init());
        assert!(!is_active());

        // Any other non-empty value activates.
        std::env::set_var(PROFILE_ENV, "1");
        assert!(init());
        assert!(is_active());

        // Reset shared state for other tests.
        std::env::remove_var(PROFILE_ENV);
        ACTIVE.store(false, Ordering::Relaxed);
    }

    #[test]
    fn enable_forces_active() {
        let _g = ENV_LOCK.lock().unwrap();
        ACTIVE.store(false, Ordering::Relaxed);
        assert!(!is_active());
        enable();
        assert!(is_active());
        ACTIVE.store(false, Ordering::Relaxed);
    }

    #[test]
    fn span_drop_is_noop_when_inactive() {
        let _g = ENV_LOCK.lock().unwrap();
        ACTIVE.store(false, Ordering::Relaxed);
        // Should neither panic nor emit. We can only assert it does not panic
        // and that elapsed is observable before drop.
        let s = span("phase", "sub").with_items(5);
        assert!(s.elapsed() < Duration::from_secs(60));
        drop(s);
        // Still inactive afterwards.
        assert!(!is_active());
    }

    #[test]
    fn span_set_items_updates_in_place() {
        let _g = ENV_LOCK.lock().unwrap();
        ACTIVE.store(false, Ordering::Relaxed);
        let mut s = span("p", "s");
        assert_eq!(s.items, 0);
        s.set_items(7);
        assert_eq!(s.items, 7);
    }

    #[test]
    fn clear_empties_the_buffer() {
        let mut p = Profiler::new();
        p.time("a", "b", || {});
        assert!(!p.is_empty());
        p.clear();
        assert!(p.is_empty());
    }
}
