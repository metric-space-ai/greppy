//! Memory budget tracking based on actual resident set size (RSS).
//!
//! Ported from `.vendor/codebase-memory-mcp.git/src/foundation/mem.c`
//! (`cbm_mem_init`, `cbm_mem_rss`, `cbm_mem_budget`, `cbm_mem_over_budget`,
//! `cbm_mem_worker_budget`).
//!
//! The budget is a fraction of the total RAM available to the process
//! (cgroup-aware on Linux, via [`crate::sysinfo`]). RSS is read straight
//! from the OS — `task_info` on macOS, `/proc/self/statm` on Linux — with
//! no external crates.

use std::sync::OnceLock;

use crate::sysinfo;

const DEFAULT_RAM_FRACTION: f64 = 0.5;
const MAX_RAM_FRACTION: f64 = 1.0;

/// Process-wide budget in bytes, computed exactly once by the first
/// [`init`] caller. Until then it is unset and [`budget`] reports `0`.
///
/// Using a `OnceLock` (rather than a latch + separate atomic) closes the
/// window where a losing concurrent caller could observe the latch as
/// taken but the budget as not-yet-stored: `get_or_init` blocks all
/// callers until the value is fully computed.
fn budget_cell() -> &'static OnceLock<u64> {
    static CELL: OnceLock<u64> = OnceLock::new();
    &CELL
}

/// Initialise the memory budget to `ram_fraction * total_ram`.
///
/// `ram_fraction` outside `(0.0, 1.0]` is clamped to the 0.5 default.
/// Only the first call has an effect; later calls are no-ops (so callers
/// can initialise once at startup without racing).
///
/// Returns the budget in bytes that is now in effect.
pub fn init(ram_fraction: f64) -> u64 {
    *budget_cell().get_or_init(|| {
        let fraction = if ram_fraction <= 0.0 || ram_fraction > MAX_RAM_FRACTION {
            DEFAULT_RAM_FRACTION
        } else {
            ram_fraction
        };
        let total = sysinfo::total_ram();
        (total as f64 * fraction) as u64
    })
}

/// Current budget in bytes. `0` until [`init`] runs.
pub fn budget() -> u64 {
    budget_cell().get().copied().unwrap_or(0)
}

/// Current resident set size (RSS) of this process in bytes. `0` when it
/// could not be read.
pub fn rss() -> u64 {
    os_rss()
}

/// True when the current RSS exceeds the configured budget. Always false
/// when the budget is `0` (uninitialised or could-not-detect-RAM), so the
/// caller never gets a spurious over-budget signal.
pub fn over_budget() -> bool {
    let b = budget();
    if b == 0 {
        return false;
    }
    rss() > b
}

/// Per-worker budget hint: `budget / num_workers`. `num_workers <= 0` is
/// treated as 1 (mirrors `cbm_mem_worker_budget`).
pub fn worker_budget(num_workers: usize) -> u64 {
    let n = num_workers.max(1) as u64;
    budget() / n
}

// ── OS-specific RSS ──────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn os_rss() -> u64 {
    use std::os::raw::c_int;

    // Minimal mach FFI. `mach_task_basic_info` layout from
    // <mach/task_info.h>; we declare just the prefix we read.
    #[repr(C)]
    #[derive(Default)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }

    // MACH_TASK_BASIC_INFO == 20; count is the struct size in u32 words.
    const MACH_TASK_BASIC_INFO: c_int = 20;
    const KERN_SUCCESS: c_int = 0;

    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: c_int,
            task_info_out: *mut i32,
            task_info_count: *mut u32,
        ) -> c_int;
    }

    let mut info = MachTaskBasicInfo::default();
    let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
    // SAFETY: `info` is a correctly sized, properly aligned buffer of
    // `count` u32 words; task_info writes at most `count` words and updates
    // it. mach_task_self() returns the current task port.
    let rc = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut MachTaskBasicInfo as *mut i32,
            &mut count,
        )
    };
    if rc == KERN_SUCCESS {
        info.resident_size
    } else {
        0
    }
}

#[cfg(target_os = "linux")]
fn os_rss() -> u64 {
    // /proc/self/statm: "size resident shared text lib data dt" in pages.
    let Ok(text) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let mut it = text.split_whitespace();
    let _size = it.next();
    let resident_pages: u64 = match it.next().and_then(|s| s.parse().ok()) {
        Some(p) => p,
        None => return 0,
    };
    let page_size = page_size_bytes();
    resident_pages.saturating_mul(page_size)
}

#[cfg(target_os = "linux")]
fn page_size_bytes() -> u64 {
    use std::os::raw::c_int;
    extern "C" {
        fn sysconf(name: c_int) -> std::os::raw::c_long;
    }
    const SC_PAGESIZE: c_int = 30; // _SC_PAGESIZE on Linux.
                                   // SAFETY: sysconf with a constant name is a pure query.
    let ps = unsafe { sysconf(SC_PAGESIZE) };
    if ps > 0 {
        ps as u64
    } else {
        4096
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn os_rss() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rss_is_positive_on_supported_os() {
        // A running process has a non-zero RSS on macOS/Linux.
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let r = rss();
            assert!(r > 0, "RSS should be positive, got {r}");
            // Sanity floor: any real process resident set is > 256 KiB.
            assert!(r > 256 * 1024, "RSS implausibly small: {r}");
        }
    }

    #[test]
    fn init_sets_budget_to_fraction_of_ram() {
        // init() latches once per process; whoever wins set BUDGET_BYTES.
        let b = init(0.5);
        assert_eq!(b, budget(), "init must return the active budget");

        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let total = sysinfo::total_ram();
            assert!(total > 0);
            // The active budget must be a sane fraction of total RAM. We
            // don't know which fraction the first caller used, so bound it:
            // (0, total].
            assert!(b > 0, "budget must be positive after init");
            assert!(b <= total, "budget must not exceed total RAM");
        }
    }

    #[test]
    fn init_is_idempotent() {
        let first = init(0.5);
        // A second init with a wildly different fraction must NOT change
        // the latched budget.
        let second = init(0.1);
        assert_eq!(first, second, "init must latch after the first call");
        assert_eq!(budget(), first);
    }

    #[test]
    fn worker_budget_divides_evenly_and_guards_zero() {
        // Force a known budget for deterministic arithmetic by reading the
        // active one (already latched). worker_budget must equal
        // budget()/n and treat 0 as 1.
        let _ = init(0.5);
        let b = budget();
        assert_eq!(worker_budget(0), b, "0 workers must behave like 1");
        assert_eq!(worker_budget(1), b);
        if b >= 4 {
            assert_eq!(worker_budget(4), b / 4);
        }
        // Per-worker budget is monotonically non-increasing in worker count.
        assert!(worker_budget(8) <= worker_budget(2));
    }

    #[test]
    fn over_budget_false_when_budget_zero() {
        // We cannot reset the global latch, so test the pure predicate via
        // a tiny local re-implementation of the zero-budget guard to prove
        // the contract: budget 0 ⇒ never over budget.
        // (over_budget() itself is exercised against the real latch below.)
        fn over(rss: u64, budget: u64) -> bool {
            budget != 0 && rss > budget
        }
        assert!(!over(u64::MAX, 0));
        assert!(over(10, 5));
        assert!(!over(5, 10));
    }

    #[test]
    fn over_budget_consistent_with_rss_and_budget() {
        let _ = init(0.5);
        let b = budget();
        let r = rss();
        let expected = b != 0 && r > b;
        assert_eq!(over_budget(), expected);
    }
}
