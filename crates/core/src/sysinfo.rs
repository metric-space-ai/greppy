//! System information: CPU core counts and total RAM, cgroup-aware on
//! Linux so the values reflect a container's effective limits rather than
//! the host's totals.
//!
//! Ported from `.vendor/codebase-memory-mcp.git/src/foundation/system_info.c`
//! (`cbm_system_info`, `cbm_detect_cgroup_cpus`, `cbm_default_worker_count`).
//!
//! No external dependencies: macOS uses `sysctl` via a tiny FFI shim,
//! Linux reads `/sys/fs/cgroup` and `/proc` directly plus `sysconf`, and
//! other platforms fall back to [`std::thread::available_parallelism`].

use std::sync::OnceLock;

/// Hardware/container facts that do not change over the process lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemInfo {
    /// Logical CPUs available to this process (cgroup-aware on Linux).
    pub logical_cores: usize,
    /// Physical / performance cores. On Linux this equals
    /// `logical_cores` (the kernel does not distinguish P/E here); on
    /// macOS it is the performance-core count when available.
    pub physical_cores: usize,
    /// Total RAM in bytes available to this process (cgroup-aware on
    /// Linux). `0` when it could not be determined.
    pub total_ram: u64,
}

const DEFAULT_CORES: usize = 1;
const MIN_WORKERS: usize = 1;
/// Upper clamp for the `GREPPLUS_WORKERS` / `CBM_WORKERS` override.
const WORKERS_MAX: usize = 256;

fn detect() -> SystemInfo {
    #[cfg(target_os = "macos")]
    {
        detect_macos()
    }
    #[cfg(target_os = "linux")]
    {
        detect_linux()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        detect_fallback()
    }
}

/// Generic fallback used on platforms without a dedicated probe.
#[allow(dead_code)]
fn detect_fallback() -> SystemInfo {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(DEFAULT_CORES);
    SystemInfo {
        logical_cores: logical,
        physical_cores: logical,
        total_ram: 0,
    }
}

// ── macOS ────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    fn sysctl_u64(name: &str) -> Option<u64> {
        let cname = CString::new(name).ok()?;
        let mut val: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `cname` is a valid NUL-terminated string; `val`/`len`
        // describe a correctly sized output buffer. sysctlbyname writes at
        // most `len` bytes and updates `len`.
        let rc = unsafe {
            sysctlbyname(
                cname.as_ptr(),
                &mut val as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return None;
        }
        // Some sysctls are 32-bit; handle the narrow case too.
        if len == std::mem::size_of::<u32>() {
            return Some((val as u32) as u64);
        }
        Some(val)
    }

    pub(super) fn detect() -> super::SystemInfo {
        let logical = sysctl_u64("hw.ncpu")
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .unwrap_or(super::DEFAULT_CORES);

        // Performance cores when the perflevel sysctls exist (Apple
        // Silicon); on Intel they are absent so physical == logical.
        let physical = sysctl_u64("hw.perflevel0.physicalcpu")
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .unwrap_or(logical)
            .min(logical);

        let total_ram = sysctl_u64("hw.memsize").unwrap_or(0);

        super::SystemInfo {
            logical_cores: logical,
            physical_cores: physical,
            total_ram,
        }
    }
}

#[cfg(target_os = "macos")]
fn detect_macos() -> SystemInfo {
    macos::detect()
}

// ── Linux ────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn detect_linux() -> SystemInfo {
    let host_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(DEFAULT_CORES);

    let host_ram = read_meminfo_total().unwrap_or(0);

    // Cgroup-aware overrides. min(cgroup, host) defends against
    // mis-mounted cgroups reporting more than the host actually has.
    let cg_cpus = detect_cgroup_cpus("/sys/fs/cgroup");
    let logical = match cg_cpus {
        Some(c) if c > 0 && c < host_cpus => c,
        _ => host_cpus,
    };

    let cg_ram = detect_cgroup_mem("/sys/fs/cgroup");
    let total_ram = match cg_ram {
        Some(r) if r > 0 && (host_ram == 0 || r < host_ram) => r,
        _ => host_ram,
    };

    SystemInfo {
        logical_cores: logical,
        physical_cores: logical, // Linux does not distinguish P/E here.
        total_ram,
    }
}

#[cfg(target_os = "linux")]
fn read_meminfo_total() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            // Format: "MemTotal:       16384256 kB"
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Effective CPU count from a cgroup file tree, or `None` when there is no
/// quota (caller falls back to the host count). Mirrors
/// `cbm_detect_cgroup_cpus`.
#[cfg(target_os = "linux")]
fn detect_cgroup_cpus(cgroup_root: &str) -> Option<usize> {
    // cgroup v2: "<root>/cpu.max" — "<quota> <period>" or "max <period>".
    if let Ok(buf) = std::fs::read_to_string(format!("{cgroup_root}/cpu.max")) {
        let buf = buf.trim();
        if buf.starts_with("max") {
            return None;
        }
        let mut it = buf.split_whitespace();
        let quota: i64 = it.next().and_then(|s| s.parse().ok())?;
        let period: i64 = it.next().and_then(|s| s.parse().ok())?;
        if quota > 0 && period > 0 {
            let n = (quota + period - 1) / period; // ceil
            return Some((n as usize).max(MIN_WORKERS));
        }
        return None;
    }

    // cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us. quota -1 = unlimited.
    let quota: i64 = std::fs::read_to_string(format!("{cgroup_root}/cpu/cpu.cfs_quota_us"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if quota <= 0 {
        return None;
    }
    let period: i64 = std::fs::read_to_string(format!("{cgroup_root}/cpu/cpu.cfs_period_us"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if period <= 0 {
        return None;
    }
    let n = (quota + period - 1) / period;
    Some((n as usize).max(MIN_WORKERS))
}

/// Effective memory limit in bytes from a cgroup file tree, or `None` when
/// unlimited. Mirrors `cbm_detect_cgroup_mem`.
#[cfg(target_os = "linux")]
fn detect_cgroup_mem(cgroup_root: &str) -> Option<u64> {
    // cgroup v2: "<root>/memory.max" — "max" or integer bytes.
    if let Ok(buf) = std::fs::read_to_string(format!("{cgroup_root}/memory.max")) {
        let buf = buf.trim();
        if buf.starts_with("max") {
            return None;
        }
        let n: u64 = buf.parse().ok()?;
        return if n == 0 { None } else { Some(n) };
    }

    // cgroup v1: memory.limit_in_bytes. The "unlimited" sentinel is a very
    // large value (~PAGE_COUNTER_MAX); treat anything past half of u64::MAX
    // as effectively unlimited.
    let n: u64 = std::fs::read_to_string(format!("{cgroup_root}/memory/memory.limit_in_bytes"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if n == 0 || n >= u64::MAX / 2 {
        None
    } else {
        Some(n)
    }
}

// ── Public API ───────────────────────────────────────────────────────

/// Cached, cgroup-aware system information. Hardware facts are immutable
/// for the life of the process, so the first call detects and the rest
/// return the cached value.
pub fn system_info() -> &'static SystemInfo {
    static CACHE: OnceLock<SystemInfo> = OnceLock::new();
    CACHE.get_or_init(detect)
}

/// Logical CPU count available to this process (cgroup-aware on Linux).
pub fn logical_cpu_count() -> usize {
    system_info().logical_cores.max(MIN_WORKERS)
}

/// Physical / performance CPU count (P-cores on Apple Silicon; equals the
/// logical count on Linux and Intel macOS).
pub fn physical_cpu_count() -> usize {
    system_info().physical_cores.max(MIN_WORKERS)
}

/// Total RAM in bytes available to this process (cgroup-aware on Linux).
/// `0` when undetermined.
pub fn total_ram() -> u64 {
    system_info().total_ram
}

/// Default worker count, mirroring `cbm_default_worker_count`.
///
/// Precedence: an explicit `GREPPLUS_WORKERS` (or legacy `CBM_WORKERS`)
/// env override clamped to `[1, 256]` wins; otherwise:
///   * `initial == true`  → all logical cores (the user is waiting on the
///     first index, so use everything),
///   * `initial == false` → `physical_cores - 1`, leaving headroom for the
///     user's other apps during incremental work (never below 1).
pub fn default_worker_count(initial: bool) -> usize {
    if let Some(n) = worker_env_override() {
        return n;
    }

    let info = system_info();
    if initial {
        return info.logical_cores.max(MIN_WORKERS);
    }
    info.physical_cores.saturating_sub(1).max(MIN_WORKERS)
}

fn worker_env_override() -> Option<usize> {
    for key in ["GREPPLUS_WORKERS", "CBM_WORKERS"] {
        if let Ok(val) = std::env::var(key) {
            if let Ok(n) = val.trim().parse::<usize>() {
                if (MIN_WORKERS..=WORKERS_MAX).contains(&n) {
                    return Some(n);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_sane() {
        let info = system_info();
        assert!(info.logical_cores >= 1, "must report at least one core");
        assert!(info.physical_cores >= 1);
        assert!(
            info.physical_cores <= info.logical_cores,
            "physical ({}) must not exceed logical ({})",
            info.physical_cores,
            info.logical_cores
        );
    }

    #[test]
    fn accessors_match_cached_info() {
        let info = system_info();
        assert_eq!(logical_cpu_count(), info.logical_cores.max(1));
        assert_eq!(physical_cpu_count(), info.physical_cores.max(1));
        assert_eq!(total_ram(), info.total_ram);
    }

    #[test]
    fn system_info_is_cached_stable() {
        // Two calls must return the exact same cached struct.
        assert_eq!(system_info(), system_info());
    }

    #[test]
    fn total_ram_is_plausible_on_supported_os() {
        // On macOS and Linux we should be able to read total RAM; assert a
        // floor of 64 MiB so a bogus near-zero reading fails the test.
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(
                total_ram() >= 64 * 1024 * 1024,
                "total_ram looked implausibly small: {}",
                total_ram()
            );
        }
    }

    #[test]
    fn default_worker_count_initial_uses_all_cores() {
        // Clear any ambient override for a deterministic check.
        // (Tests run in-process; we only read, never set globally here.)
        if worker_env_override().is_none() {
            let expected = system_info().logical_cores.max(1);
            assert_eq!(default_worker_count(true), expected);
        }
    }

    #[test]
    fn default_worker_count_incremental_leaves_headroom() {
        if worker_env_override().is_none() {
            let info = system_info();
            let expected = info.physical_cores.saturating_sub(1).max(1);
            assert_eq!(default_worker_count(false), expected);
            assert!(default_worker_count(false) >= 1);
        }
    }

    #[test]
    fn worker_env_override_is_clamped() {
        // Exercise the override path via a scoped env var. Serialized by
        // a process-wide mutex to avoid races with other env tests.
        let _guard = ENV_LOCK.lock().unwrap();

        std::env::set_var("GREPPLUS_WORKERS", "7");
        assert_eq!(default_worker_count(true), 7);
        assert_eq!(default_worker_count(false), 7);

        // Out-of-range values are ignored (fall back to detection).
        std::env::set_var("GREPPLUS_WORKERS", "0");
        assert!(worker_env_override().is_none());
        std::env::set_var("GREPPLUS_WORKERS", "99999");
        assert!(worker_env_override().is_none());
        std::env::set_var("GREPPLUS_WORKERS", "not-a-number");
        assert!(worker_env_override().is_none());

        std::env::remove_var("GREPPLUS_WORKERS");
    }

    #[test]
    fn legacy_cbm_workers_override_honored() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("GREPPLUS_WORKERS");
        std::env::set_var("CBM_WORKERS", "3");
        assert_eq!(worker_env_override(), Some(3));
        std::env::remove_var("CBM_WORKERS");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_v2_cpu_max_parsing() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!(
            "grepplus-cg-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // 150000/100000 → ceil = 2 cores.
        let mut f = std::fs::File::create(dir.join("cpu.max")).unwrap();
        write!(f, "150000 100000").unwrap();
        drop(f);
        assert_eq!(detect_cgroup_cpus(dir.to_str().unwrap()), Some(2));

        // "max" → unlimited → None.
        std::fs::write(dir.join("cpu.max"), b"max 100000").unwrap();
        assert_eq!(detect_cgroup_cpus(dir.to_str().unwrap()), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_v2_memory_max_parsing() {
        let dir = std::env::temp_dir().join(format!(
            "grepplus-cgm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(dir.join("memory.max"), b"536870912").unwrap(); // 512 MiB
        assert_eq!(detect_cgroup_mem(dir.to_str().unwrap()), Some(536870912));

        std::fs::write(dir.join("memory.max"), b"max").unwrap();
        assert_eq!(detect_cgroup_mem(dir.to_str().unwrap()), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // Serialize env-mutating tests (set_var/remove_var are process-global).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
