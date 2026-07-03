//! Query-latency and error-rate counters.
//!
//! Ported from `.vendor/codebase-memory-mcp.git/src/foundation/diagnostics.c`
//! (`cbm_query_stats_t`, `cbm_diag_record_query`). Upstream tracks a running
//! count, an error count, cumulative and max latency; this port keeps that
//! exact accounting and additionally retains a bounded ring of recent samples
//! so a [`Snapshot`] can report p50/p95-ish percentiles alongside the mean —
//! the headline numbers a soak/diagnostics reader cares about.
//!
//! The periodic JSON-file writer thread from upstream is **not** ported here:
//! it is an application/IO concern, not a core type, and would pull in process
//! and filesystem coupling that `grepplus-core` deliberately avoids. The
//! counters themselves — the part other crates would record into — live here.
//! See `residualGaps` in the track report.
//!
//! `std`-only: percentiles are computed by sorting a copy of the retained
//! samples; no histogram crate, no external deps.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Maximum number of recent latency samples retained for percentile
/// estimation. Bounded so memory stays flat under sustained load; older
/// samples are overwritten ring-buffer style.
pub const SAMPLE_CAPACITY: usize = 4096;

/// Thread-safe accumulator for query latency and error rate.
///
/// Counts and cumulative/max latency are plain atomics (lock-free, matching
/// the upstream `atomic_fetch_add` / CAS-max). The recent-sample ring is
/// guarded by a small `Mutex`; it is only touched on record and on snapshot,
/// and never held across user code.
#[derive(Debug)]
pub struct Diagnostics {
    count: AtomicU64,
    errors: AtomicU64,
    total_us: AtomicU64,
    max_us: AtomicU64,
    samples: Mutex<Samples>,
}

#[derive(Debug)]
struct Samples {
    buf: Vec<u64>,
    /// Next write position (ring index).
    next: usize,
    /// True once `buf` has wrapped at least once (i.e. it is full).
    wrapped: bool,
}

impl Samples {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            next: 0,
            wrapped: false,
        }
    }

    fn push(&mut self, us: u64) {
        if self.buf.len() < SAMPLE_CAPACITY {
            self.buf.push(us);
            self.next = self.buf.len() % SAMPLE_CAPACITY;
            if self.buf.len() == SAMPLE_CAPACITY {
                self.wrapped = true;
            }
        } else {
            self.buf[self.next] = us;
            self.next = (self.next + 1) % SAMPLE_CAPACITY;
            self.wrapped = true;
        }
    }
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl Diagnostics {
    /// Create an empty diagnostics accumulator.
    pub const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            total_us: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
            samples: Mutex::new(Samples {
                buf: Vec::new(),
                next: 0,
                wrapped: false,
            }),
        }
    }

    /// Record one completed query: its wall-clock `duration` and whether it
    /// succeeded (`ok == false` counts as an error). Mirrors
    /// `cbm_diag_record_query`. Saturating microsecond conversion guards
    /// against pathological durations.
    pub fn record_query(&self, duration: Duration, ok: bool) {
        let us = duration.as_micros().min(u64::MAX as u128) as u64;

        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(us, Ordering::Relaxed);
        if !ok {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }

        // Lock-free max via CAS loop (upstream parity).
        let mut old = self.max_us.load(Ordering::Relaxed);
        while us > old {
            match self
                .max_us
                .compare_exchange_weak(old, us, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(actual) => old = actual,
            }
        }

        if let Ok(mut s) = self.samples.lock() {
            s.push(us);
        }
    }

    /// Total queries recorded so far.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Total queries that were recorded as errors.
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Take a consistent-enough point-in-time snapshot of the counters and
    /// derived latency statistics.
    pub fn snapshot(&self) -> Snapshot {
        let count = self.count.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let total_us = self.total_us.load(Ordering::Relaxed);
        let max_us = self.max_us.load(Ordering::Relaxed);

        let mean_us = total_us.checked_div(count).unwrap_or(0);

        // Copy + sort the retained samples for percentiles.
        let mut sorted: Vec<u64> = match self.samples.lock() {
            Ok(s) => s.buf.clone(),
            Err(poisoned) => poisoned.into_inner().buf.clone(),
        };
        sorted.sort_unstable();

        let p50_us = percentile(&sorted, 50);
        let p95_us = percentile(&sorted, 95);

        Snapshot {
            count,
            errors,
            total_us,
            max_us,
            mean_us,
            p50_us,
            p95_us,
            sampled: sorted.len() as u64,
        }
    }

    /// Reset every counter and discard retained samples. Useful between soak
    /// windows or test cases.
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
        self.total_us.store(0, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
        if let Ok(mut s) = self.samples.lock() {
            *s = Samples::new();
        }
    }
}

/// Nearest-rank percentile (1..=100) over a pre-sorted slice, in the slice's
/// units. Returns `0` for an empty slice.
fn percentile(sorted: &[u64], pct: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let pct = pct.clamp(1, 100) as usize;
    // Nearest-rank: rank = ceil(pct/100 * n), 1-based.
    let n = sorted.len();
    let rank = (pct * n).div_ceil(100); // 1..=n
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}

/// Immutable view of the accumulated query statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    /// Total queries recorded.
    pub count: u64,
    /// Queries recorded as errors.
    pub errors: u64,
    /// Cumulative latency across all queries, in microseconds.
    pub total_us: u64,
    /// Largest single-query latency observed, in microseconds.
    pub max_us: u64,
    /// Mean latency (`total_us / count`), in microseconds; `0` when no queries.
    pub mean_us: u64,
    /// ~50th-percentile latency over retained samples, in microseconds.
    pub p50_us: u64,
    /// ~95th-percentile latency over retained samples, in microseconds.
    pub p95_us: u64,
    /// Number of samples backing the percentile estimates (≤ [`SAMPLE_CAPACITY`]).
    pub sampled: u64,
}

impl Snapshot {
    /// Error rate as a fraction in `[0.0, 1.0]`; `0.0` when no queries.
    pub fn error_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.errors as f64 / self.count as f64
        }
    }

    /// Mean latency as a [`Duration`].
    pub fn mean(&self) -> Duration {
        Duration::from_micros(self.mean_us)
    }

    /// Max latency as a [`Duration`].
    pub fn max(&self) -> Duration {
        Duration::from_micros(self.max_us)
    }

    /// ~p50 latency as a [`Duration`].
    pub fn p50(&self) -> Duration {
        Duration::from_micros(self.p50_us)
    }

    /// ~p95 latency as a [`Duration`].
    pub fn p95(&self) -> Duration {
        Duration::from_micros(self.p95_us)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_is_all_zero() {
        let d = Diagnostics::new();
        let s = d.snapshot();
        assert_eq!(s.count, 0);
        assert_eq!(s.errors, 0);
        assert_eq!(s.total_us, 0);
        assert_eq!(s.max_us, 0);
        assert_eq!(s.mean_us, 0);
        assert_eq!(s.p50_us, 0);
        assert_eq!(s.p95_us, 0);
        assert_eq!(s.sampled, 0);
        assert_eq!(s.error_rate(), 0.0);
    }

    #[test]
    fn counters_accumulate() {
        let d = Diagnostics::new();
        d.record_query(Duration::from_micros(100), true);
        d.record_query(Duration::from_micros(300), true);
        d.record_query(Duration::from_micros(200), false); // an error

        let s = d.snapshot();
        assert_eq!(s.count, 3);
        assert_eq!(s.errors, 1);
        assert_eq!(s.total_us, 600);
        assert_eq!(s.max_us, 300);
        assert_eq!(s.mean_us, 200);
        assert_eq!(d.count(), 3);
        assert_eq!(d.errors(), 1);
    }

    #[test]
    fn error_rate_is_fraction() {
        let d = Diagnostics::new();
        for _ in 0..4 {
            d.record_query(Duration::from_micros(10), true);
        }
        d.record_query(Duration::from_micros(10), false);
        let s = d.snapshot();
        assert_eq!(s.count, 5);
        assert_eq!(s.errors, 1);
        assert!((s.error_rate() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn max_tracks_the_largest() {
        let d = Diagnostics::new();
        d.record_query(Duration::from_micros(50), true);
        d.record_query(Duration::from_micros(5000), true);
        d.record_query(Duration::from_micros(500), true);
        assert_eq!(d.snapshot().max_us, 5000);
    }

    #[test]
    fn percentiles_over_known_distribution() {
        let d = Diagnostics::new();
        // 1..=100 microseconds, one sample each.
        for i in 1..=100u64 {
            d.record_query(Duration::from_micros(i), true);
        }
        let s = d.snapshot();
        assert_eq!(s.sampled, 100);
        // Nearest-rank: p50 -> rank 50 -> value 50; p95 -> rank 95 -> value 95.
        assert_eq!(s.p50_us, 50);
        assert_eq!(s.p95_us, 95);
        assert_eq!(s.max_us, 100);
        assert_eq!(s.mean_us, 50); // (1+..+100)/100 = 5050/100 = 50 (integer)
    }

    #[test]
    fn percentile_helper_edges() {
        assert_eq!(percentile(&[], 50), 0);
        assert_eq!(percentile(&[42], 50), 42);
        assert_eq!(percentile(&[42], 95), 42);
        let v: Vec<u64> = (1..=10).collect();
        assert_eq!(percentile(&v, 100), 10);
        assert_eq!(percentile(&v, 1), 1);
    }

    #[test]
    fn samples_are_bounded_and_ring_overwrites() {
        let d = Diagnostics::new();
        let total = SAMPLE_CAPACITY + 500;
        for i in 0..total {
            d.record_query(Duration::from_micros(i as u64), true);
        }
        let s = d.snapshot();
        // Count keeps growing unbounded...
        assert_eq!(s.count, total as u64);
        // ...but retained samples are capped.
        assert_eq!(s.sampled, SAMPLE_CAPACITY as u64);
        // Oldest (small) samples were overwritten; the smallest retained
        // sample should be at least the number we dropped.
        let dropped = (total - SAMPLE_CAPACITY) as u64;
        assert!(
            s.p50_us >= dropped,
            "p50={} should reflect overwritten low samples (dropped {})",
            s.p50_us,
            dropped
        );
    }

    #[test]
    fn reset_clears_everything() {
        let d = Diagnostics::new();
        d.record_query(Duration::from_micros(123), false);
        assert_eq!(d.count(), 1);
        d.reset();
        let s = d.snapshot();
        assert_eq!(s.count, 0);
        assert_eq!(s.errors, 0);
        assert_eq!(s.max_us, 0);
        assert_eq!(s.sampled, 0);
    }

    #[test]
    fn concurrent_records_accumulate_exactly() {
        use std::sync::Arc;
        use std::thread;

        let d = Arc::new(Diagnostics::new());
        let threads = 8;
        let per = 1000;
        let mut handles = Vec::new();
        for _ in 0..threads {
            let d = Arc::clone(&d);
            handles.push(thread::spawn(move || {
                for _ in 0..per {
                    d.record_query(Duration::from_micros(10), true);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let s = d.snapshot();
        assert_eq!(s.count, (threads * per) as u64);
        assert_eq!(s.total_us, (threads * per * 10) as u64);
        assert_eq!(s.errors, 0);
    }

    #[test]
    fn snapshot_is_copy_and_comparable() {
        let d = Diagnostics::new();
        d.record_query(Duration::from_micros(7), true);
        let a = d.snapshot();
        let b = a; // Copy
        assert_eq!(a, b);
    }
}
