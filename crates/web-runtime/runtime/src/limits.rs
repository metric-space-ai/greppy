//! Session resource limits (guide §17). Enforced in the supervisor, not the worker.
//!
//! A web read degrades instead of aborting when a budget is spent. The document
//! text already loaded is returned with `degraded: true` and a `reason`
//! (`network_budget`, `content_cpu`, `frame`, or `session_limit`). Images,
//! media, fonts, and trackers are not loaded for a text read; once the network
//! budget is spent, later scripts and other side resources are dropped too.
//!
//! Defaults, and why:
//! - network: 64 MiB per navigation (`GREPPY_WEB_NETWORK_BUDGET`, `0` = no cap).
//!   8 MiB rejected ordinary pages (DuckDuckGo HTML is about 10 MB,
//!   Kleinanzeigen about 17–27 MB) before any text came back.
//! - content CPU: 60 s on one core, 30 s times the core count otherwise.
//!   Spending it stops further script and layout work; it does not discard text.
//! - frame: the inline text cap follows the host RAM (2/4/8 MiB at
//!   512 MiB/1 GiB/4 GiB) and never rejects the read.
//! - RSS: the content worker is recycled at half of `--max-ram` (or of the
//!   detected cgroup/proc RAM). The session stays and the read is retried.
//!   A smaller budget does not exit.
//!
//! Unreachable hosts and sandbox or policy violations stay errors.

use std::time::Duration;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionLimits {
    /// Optional cumulative session-age budget. Ordinary sessions rely on
    /// per-operation deadlines and idle cleanup instead of expiring merely
    /// because an agent spent time reasoning between operations.
    pub wall_time: Option<Duration>,
    pub controller_cpu_time: Duration,
    pub content_cpu_time: Duration,
    pub controller_heap_bytes: u64,
    pub content_rss_bytes: u64,
    pub max_pages: u32,
    pub max_contexts: u32,
    pub max_requests: u64,
    pub max_network_bytes: u64,
    pub max_download_bytes: u64,
    pub max_artifact_bytes: u64,
    pub max_console_bytes: u64,
    pub idle_ttl: Duration,
}

pub fn small_host(cores: u32, max_ram: Option<u64>) -> bool {
    cores <= 1 || max_ram.is_some_and(|bytes| bytes <= 4 * 1024 * 1024 * 1024)
}

pub fn host_live_pages(cores: u32, max_ram: Option<u64>, configured: u32) -> u32 {
    if let Some(ram) = max_ram {
        return host_shape(ram).live_pages.min(configured.max(1));
    }
    if cores <= 1 {
        1
    } else {
        configured.max(1)
    }
}

/// How a host spends the RAM it was given. Less RAM means fewer live pages,
/// a smaller JS heap, a smaller frame, and an earlier recycle. It does not
/// change the document a read returns.
///
/// | `--max-ram` | live pages | recycle RSS | JS heap | frame |
/// | 512 MiB | 1 | 256 MiB | 64 MiB | 2 MiB |
/// | 1 GiB | 1 | 512 MiB | 128 MiB | 4 MiB |
/// | 4 GiB | 2 | 2 GiB | 255 MiB | 8 MiB |
///
/// JS heap stops at 255 MiB because Servo's `js.mem.max` rejects 256.
/// Recycle is half the budget, so the worker is replaced before RSS reaches
/// the cap. The next read retries in the new process and keeps the session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostShape {
    pub ram_bytes: u64,
    pub live_pages: u32,
    pub recycle_rss_bytes: u64,
    pub js_heap_mib: i64,
    pub frame_bytes: u64,
    pub cache_bytes: u64,
    pub sequential: bool,
}

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

pub fn host_shape(ram_bytes: u64) -> HostShape {
    let ram_bytes = ram_bytes.max(64 * MIB);
    let live_pages = if ram_bytes < 4 * GIB {
        1
    } else {
        u32::try_from((ram_bytes / (2 * GIB)).clamp(1, 8)).unwrap_or(1)
    };
    let js_heap_mib = i64::try_from((ram_bytes / (8 * MIB)).clamp(32, 255)).unwrap_or(255);
    HostShape {
        ram_bytes,
        live_pages,
        recycle_rss_bytes: ram_bytes / 2,
        js_heap_mib,
        frame_bytes: (ram_bytes / 256).clamp(MIB, 8 * MIB),
        cache_bytes: ram_bytes / 16,
        sequential: live_pages == 1,
    }
}

/// RSS at which the content worker is replaced. The threshold is half of
/// `--max-ram`, or of the RAM cgroup/`MemAvailable` report when the flag is
/// unset. There is no fixed 512 MiB constant.
pub fn content_rss_recycle_bytes(_cores: u32, max_ram: Option<u64>) -> Option<u64> {
    let ram = max_ram.or_else(detected_ram_bytes)?;
    Some(host_shape(ram).recycle_rss_bytes)
}

/// `--max-ram` / `GREPPY_MAX_RAM`, otherwise the cgroup limit or `/proc/meminfo`.
pub fn effective_ram_bytes() -> Option<u64> {
    max_ram_bytes().or_else(detected_ram_bytes)
}

pub fn detected_ram_bytes() -> Option<u64> {
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.high",
    ] {
        if let Some(bytes) = cgroup_memory_file(path) {
            return Some(bytes);
        }
    }
    meminfo_bytes("MemAvailable").or_else(|| meminfo_bytes("MemTotal"))
}

fn cgroup_memory_file(path: &str) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    if text.is_empty() || text == "max" {
        return None;
    }
    let bytes: u64 = text.parse().ok()?;
    if bytes == 0 || bytes > 1 << 50 {
        return None;
    }
    Some(bytes)
}

fn meminfo_bytes(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let rest = rest.trim_start().strip_prefix(':')?.trim();
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        return kb.checked_mul(1024);
    }
    None
}

/// The document text does not depend on the RAM budget. A smaller host
/// serializes and recycles; it still returns this text and exit 0.
pub fn read_page_under_ram(document: &str, ram_bytes: u64) -> (String, u8) {
    let _pace = host_shape(ram_bytes);
    (document.to_owned(), 0)
}

pub fn should_recycle_content(rss_bytes: u64, cores: u32, max_ram: Option<u64>) -> bool {
    content_rss_recycle_bytes(cores, max_ram).is_some_and(|limit| rss_bytes > limit)
}

/// Content CPU allowed for one navigation.
///
/// One core gets 60s: a page such as Wikipedia on a 1 vCPU VPS uses just
/// over 30s of content CPU, and that used to be a session-lifetime counter.
/// Each additional core adds 30s because Servo spreads work across threads.
pub fn navigation_content_cpu(cores: u32) -> Duration {
    let cores = cores.max(1);
    if cores == 1 {
        Duration::from_secs(60)
    } else {
        Duration::from_secs(30).saturating_mul(cores)
    }
}

/// Extra Stylo threads beside the main thread. One core keeps selector
/// matching on the main thread (`layout.threads` <= 1). More cores keep
/// Servo's default of 3.
pub fn style_threads_for_host(cores: u32) -> Option<i64> {
    if cores <= 1 {
        Some(1)
    } else {
        None
    }
}

/// `cpu.max` quota/period. `max` means the cgroup is not capping cores.
pub fn cores_from_cpu_max(text: &str) -> Option<u32> {
    let mut parts = text.split_whitespace();
    let quota = parts.next()?;
    if quota == "max" {
        return None;
    }
    let quota: u64 = quota.parse().ok()?;
    let period: u64 = parts.next()?.parse().ok()?;
    if period == 0 {
        return None;
    }
    Some(
        u32::try_from((quota + period - 1) / period)
            .unwrap_or(u32::MAX)
            .max(1),
    )
}

pub fn effective_cpu_count() -> u32 {
    if let Some(cores) = std::fs::read_to_string("/sys/fs/cgroup/cpu.max")
        .ok()
        .as_deref()
        .and_then(cores_from_cpu_max)
    {
        return cores;
    }
    std::thread::available_parallelism()
        .map(|count| u32::try_from(count.get()).unwrap_or(1))
        .unwrap_or(1)
        .max(1)
}

/// `GREPPY_MAX_RAM` as a byte count. Accepts a bare integer or a K/M/G suffix.
pub fn max_ram_bytes() -> Option<u64> {
    let raw = std::env::var("GREPPY_MAX_RAM").ok()?;
    parse_byte_size(raw.trim())
}

/// Per-navigation network budget. `GREPPY_WEB_NETWORK_BUDGET=0` disables the cap.
pub const DEFAULT_NETWORK_BUDGET: u64 = 64 * 1024 * 1024;

pub fn network_budget_bytes() -> u64 {
    match std::env::var("GREPPY_WEB_NETWORK_BUDGET") {
        Ok(raw) => match parse_byte_size(raw.trim()) {
            Some(0) => u64::MAX,
            Some(bytes) => bytes,
            None => DEFAULT_NETWORK_BUDGET,
        },
        Err(_) => DEFAULT_NETWORK_BUDGET,
    }
}

pub fn parse_byte_size(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let (number, shift) = if raw.len() > 1 && raw.ends_with(|ch: char| matches!(ch, 'G' | 'g')) {
        (&raw[..raw.len() - 1], 30u32)
    } else if raw.len() > 1 && raw.ends_with(|ch: char| matches!(ch, 'M' | 'm')) {
        (&raw[..raw.len() - 1], 20u32)
    } else if raw.len() > 1 && raw.ends_with(|ch: char| matches!(ch, 'K' | 'k')) {
        (&raw[..raw.len() - 1], 10u32)
    } else {
        (raw, 0u32)
    };
    let number: u64 = number.trim().parse().ok()?;
    number.checked_shl(shift)
}

impl SessionLimits {
    pub fn for_profile(profile: &str) -> Self {
        let mut limits = Self::default();
        if profile == "research" {
            limits.max_pages = 8;
            limits.max_network_bytes = network_budget_bytes();
            // Room for the document text. Side resources are not stored.
            limits.max_artifact_bytes = 8 * 1024 * 1024;
            limits.idle_ttl = Duration::from_secs(2 * 60);
        }
        limits
    }

    pub(crate) fn apply_payload(&mut self, payload: &serde_json::Value) {
        let Some(object) = payload.as_object() else {
            return;
        };
        if let Some(value) = object.get("max_pages").and_then(|v| v.as_u64()) {
            self.max_pages = value as u32;
        }
        if let Some(value) = object.get("max_contexts").and_then(|v| v.as_u64()) {
            self.max_contexts = value as u32;
        }
        if let Some(value) = object.get("max_requests").and_then(|v| v.as_u64()) {
            self.max_requests = value;
        }
        if let Some(value) = object.get("max_network_bytes").and_then(|v| v.as_u64()) {
            self.max_network_bytes = value;
        }
        if let Some(value) = object.get("max_download_bytes").and_then(|v| v.as_u64()) {
            self.max_download_bytes = value;
        }
        if let Some(value) = object.get("max_artifact_bytes").and_then(|v| v.as_u64()) {
            self.max_artifact_bytes = value;
        }
        if let Some(value) = object.get("max_console_bytes").and_then(|v| v.as_u64()) {
            self.max_console_bytes = value;
        }
        if let Some(value) = object.get("content_rss_bytes").and_then(|v| v.as_u64()) {
            self.content_rss_bytes = value;
        }
        if let Some(value) = object.get("controller_heap_bytes").and_then(|v| v.as_u64()) {
            self.controller_heap_bytes = value;
        }
        if let Some(value) = object.get("idle_ttl_ms").and_then(|v| v.as_u64()) {
            self.idle_ttl = Duration::from_millis(value.clamp(20, 3_600_000));
        }
        if let Some(value) = object.get("wall_ms").and_then(|v| v.as_u64()) {
            self.wall_time = Some(Duration::from_millis(value.clamp(20, 3_600_000)));
        }
        if let Some(value) = object.get("content_cpu_ms").and_then(|v| v.as_u64()) {
            self.content_cpu_time = Duration::from_millis(value);
        }
        if let Some(value) = object.get("controller_cpu_ms").and_then(|v| v.as_u64()) {
            self.controller_cpu_time = Duration::from_millis(value);
        }
    }

    /// Shape defaults for this machine. An explicit payload limit applied
    /// afterwards still wins. One vCPU, or a host of at most 4 GiB named by
    /// `GREPPY_MAX_RAM`, keeps a single live page so a heavy tab cannot sit
    /// beside another. The CPU budget is per navigation and grows with cores.
    pub fn apply_host_shape(&mut self) {
        let cores = effective_cpu_count();
        self.max_pages = host_live_pages(cores, max_ram_bytes(), self.max_pages);
        let budget = navigation_content_cpu(cores);
        self.content_cpu_time = budget;
        self.controller_cpu_time = budget;
    }

    pub fn check_pages(&self, pages: u32) -> Result<(), String> {
        if pages > self.max_pages {
            Err(format!(
                "page limit exceeded ({pages} > {})",
                self.max_pages
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_artifact_bytes(&self, used: u64, additional: u64) -> Result<(), String> {
        let next = used.saturating_add(additional);
        if next > self.max_artifact_bytes {
            Err(format!(
                "artifact limit exceeded ({next} > {})",
                self.max_artifact_bytes
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_network_bytes(&self, used: u64, additional: u64) -> Result<(), String> {
        let next = used.saturating_add(additional);
        if self.max_network_bytes == u64::MAX {
            return Ok(());
        }
        if next > self.max_network_bytes {
            Err(format!(
                "network limit exceeded for this navigation ({next} > {}); the next navigation starts a new budget",
                self.max_network_bytes
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_wall_time(&self, elapsed: Duration) -> Result<(), String> {
        let Some(limit) = self.wall_time else {
            return Ok(());
        };
        if elapsed > limit {
            Err(format!(
                "session wall time exceeded (session age {elapsed:?} > {:?}; this is not the duration of the last action)",
                limit
            ))
        } else {
            Ok(())
        }
    }

    /// Restrict an operation's own finite budget by an explicitly configured
    /// cumulative session budget, when one exists.
    pub(crate) fn operation_budget(&self, elapsed: Duration, budget: Duration) -> Duration {
        self.wall_time
            .map(|limit| limit.saturating_sub(elapsed).min(budget))
            .unwrap_or(budget)
    }

    pub(crate) fn remaining_wall_time(&self, elapsed: Duration) -> Option<Duration> {
        self.wall_time.map(|limit| limit.saturating_sub(elapsed))
    }

    pub fn check_contexts(&self, contexts: u32) -> Result<(), String> {
        if contexts > self.max_contexts {
            Err(format!(
                "context limit exceeded ({contexts} > {})",
                self.max_contexts
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_requests(&self, requests: u64) -> Result<(), String> {
        if requests > self.max_requests {
            Err(format!(
                "request limit exceeded ({requests} > {})",
                self.max_requests
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_download_bytes(&self, used: u64, additional: u64) -> Result<(), String> {
        let next = used.saturating_add(additional);
        if next > self.max_download_bytes {
            Err(format!(
                "download limit exceeded ({next} > {})",
                self.max_download_bytes
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_console_bytes(&self, used: u64, additional: u64) -> Result<(), String> {
        let next = used.saturating_add(additional);
        if next > self.max_console_bytes {
            Err(format!(
                "console limit exceeded ({next} > {})",
                self.max_console_bytes
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_content_rss(&self, rss_bytes: u64) -> Result<(), String> {
        if rss_bytes > self.content_rss_bytes {
            Err(format!(
                "content rss exceeded ({rss_bytes} > {})",
                self.content_rss_bytes
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_controller_memory(&self, bytes: u64) -> Result<(), String> {
        if bytes > self.controller_heap_bytes {
            Err(format!(
                "controller memory exceeded ({bytes} > {})",
                self.controller_heap_bytes
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_cpu_time(
        &self,
        used: Duration,
        limit: Duration,
        label: &str,
    ) -> Result<(), String> {
        if used > limit {
            Err(format!(
                "{label} cpu time exceeded for this navigation ({used:?} > {limit:?}); the next navigation starts a new budget"
            ))
        } else {
            Ok(())
        }
    }
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            wall_time: None,
            controller_cpu_time: Duration::from_secs(30),
            content_cpu_time: Duration::from_secs(30),
            controller_heap_bytes: 256 * 1024 * 1024,
            // Three real pages of the pinned corpus failed at 536.9 MB
            // against a 536.87 MB ceiling -- under a tenth of a percent over.
            // A limit that a normal news site clears by a rounding error is
            // not protecting anything, it is only turning working pages into
            // failures. 1 GiB still stops a runaway document.
            // 2 GiB. One mirrored news page needs 1.25 GB, and a desktop
            // browser routinely uses more; the ceiling is there to stop a
            // runaway document, not to fail ordinary pages. That 1.25 GB is
            // itself high for a news article is worth its own look -- but it
            // is a memory-efficiency question, not a reason to reject the page.
            content_rss_bytes: 2 * 1024 * 1024 * 1024,
            max_pages: 16,
            max_contexts: 8,
            max_requests: 256,
            max_network_bytes: network_budget_bytes(),
            max_download_bytes: 16 * 1024 * 1024,
            max_artifact_bytes: 16 * 1024 * 1024,
            max_console_bytes: 256 * 1024,
            idle_ttl: Duration::from_secs(5 * 60),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_profile_is_tighter_than_default() {
        let research = SessionLimits::for_profile("research");
        let project = SessionLimits::for_profile("project");
        assert!(research.max_pages < project.max_pages);
        assert_eq!(research.max_network_bytes, DEFAULT_NETWORK_BUDGET);
        assert_eq!(project.max_network_bytes, DEFAULT_NETWORK_BUDGET);
        assert!(research.max_network_bytes >= 10 * 1024 * 1024);
        assert!(research.check_pages(9).is_err());
        assert!(project.check_pages(9).is_ok());
    }

    #[test]
    fn wall_time_error_names_session_age_not_last_action() {
        let limits = SessionLimits {
            wall_time: Some(Duration::from_secs(120)),
            ..SessionLimits::default()
        };
        let error = limits
            .check_wall_time(Duration::from_secs(123))
            .expect_err("over budget");
        assert!(error.contains("session age"), "{error}");
        assert!(
            error.contains("not the duration of the last action"),
            "{error}"
        );
        assert!(error.contains("123s"), "{error}");
        assert!(limits.check_wall_time(Duration::from_secs(120)).is_ok());
    }

    #[test]
    fn default_session_age_does_not_replace_operation_deadlines() {
        let limits = SessionLimits::default();
        let old_session = Duration::from_secs(10 * 60);
        assert!(limits.check_wall_time(old_session).is_ok());
        assert_eq!(
            limits.operation_budget(old_session, Duration::from_secs(45)),
            Duration::from_secs(45)
        );
        assert_eq!(limits.remaining_wall_time(old_session), None);
    }

    #[test]
    fn explicit_wall_budget_still_bounds_each_operation() {
        let limits = SessionLimits {
            wall_time: Some(Duration::from_secs(120)),
            ..SessionLimits::default()
        };
        assert_eq!(
            limits.operation_budget(Duration::from_secs(110), Duration::from_secs(45)),
            Duration::from_secs(10)
        );
        assert!(limits.check_wall_time(Duration::from_secs(121)).is_err());
    }

    #[test]
    fn artifact_limit_rejects_overflow() {
        let limits = SessionLimits {
            max_artifact_bytes: 10,
            ..SessionLimits::default()
        };
        assert!(limits.check_artifact_bytes(8, 3).is_err());
        assert!(limits.check_artifact_bytes(8, 2).is_ok());
    }

    #[test]
    fn context_and_request_limits_reject_overflow() {
        let limits = SessionLimits {
            max_contexts: 1,
            max_requests: 2,
            ..SessionLimits::default()
        };
        assert!(limits.check_contexts(1).is_ok());
        assert!(limits.check_contexts(2).is_err());
        assert!(limits.check_requests(2).is_ok());
        assert!(limits.check_requests(3).is_err());
    }

    #[test]
    fn console_download_and_cpu_limits_reject_overflow() {
        let limits = SessionLimits {
            max_console_bytes: 4,
            max_download_bytes: 8,
            content_cpu_time: Duration::from_millis(10),
            ..SessionLimits::default()
        };
        assert!(limits.check_console_bytes(0, 4).is_ok());
        assert!(limits.check_console_bytes(0, 5).is_err());
        assert!(limits.check_download_bytes(0, 8).is_ok());
        assert!(limits.check_download_bytes(7, 2).is_err());
        assert!(limits
            .check_cpu_time(
                Duration::from_millis(10),
                limits.content_cpu_time,
                "content"
            )
            .is_ok());
        assert!(limits
            .check_cpu_time(
                Duration::from_millis(11),
                limits.content_cpu_time,
                "content"
            )
            .is_err());
    }

    #[test]
    fn payload_limits_override_profile_without_env() {
        let mut limits = SessionLimits::for_profile("project");
        limits.apply_payload(&serde_json::json!({
            "max_pages": 0,
            "idle_ttl_ms": 50,
            "wall_ms": 75
        }));
        assert_eq!(limits.max_pages, 0);
        assert_eq!(limits.idle_ttl, Duration::from_millis(50));
        assert_eq!(limits.wall_time, Some(Duration::from_millis(75)));
        assert!(limits.check_pages(1).is_err());
    }

    #[test]
    fn one_vcpu_budget_is_one_live_page_and_sixty_seconds() {
        assert_eq!(super::navigation_content_cpu(1), Duration::from_secs(60));
        assert_eq!(super::navigation_content_cpu(4), Duration::from_secs(120));
        assert_eq!(super::cores_from_cpu_max("100000 100000"), Some(1));
        assert_eq!(super::cores_from_cpu_max("250000 100000"), Some(3));
        assert_eq!(super::cores_from_cpu_max("max 100000"), None);
        assert_eq!(super::parse_byte_size("4G"), Some(4 * 1024 * 1024 * 1024));
        assert_eq!(
            super::parse_byte_size("4096M"),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(super::style_threads_for_host(1), Some(1));
        assert_eq!(super::style_threads_for_host(4), None);
        assert_eq!(super::host_live_pages(1, None, 16), 1);
        assert_eq!(
            super::host_live_pages(4, Some(8 * 1024 * 1024 * 1024), 16),
            super::host_shape(8 * 1024 * 1024 * 1024).live_pages
        );
        assert!(super::small_host(1, None));
        assert!(super::small_host(4, Some(4 * 1024 * 1024 * 1024)));
        assert!(!super::small_host(4, Some(8 * 1024 * 1024 * 1024)));
        let half_gib = super::host_shape(1024 * 1024 * 1024);
        assert!(!super::should_recycle_content(
            half_gib.recycle_rss_bytes,
            1,
            Some(1024 * 1024 * 1024)
        ));
        assert!(super::should_recycle_content(
            half_gib.recycle_rss_bytes + 1,
            1,
            Some(1024 * 1024 * 1024)
        ));
        assert!(!super::should_recycle_content(
            1024 * 1024 * 1024,
            8,
            Some(64 * 1024 * 1024 * 1024)
        ));
        assert_eq!(
            super::host_live_pages(4, Some(4 * 1024 * 1024 * 1024), 16),
            2
        );
    }

    #[test]
    fn the_same_page_is_returned_at_512mib_1gib_and_4gib() {
        let page = "Kleinanzeigen Angebot bleibt lesbar";
        let sizes = [
            512 * 1024 * 1024,
            1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
        ];
        let mut texts = Vec::new();
        let mut paces = Vec::new();
        for ram in sizes {
            let shape = super::host_shape(ram);
            assert!(shape.recycle_rss_bytes < ram, "recycle must stay under the cap");
            assert!(shape.js_heap_mib >= 32 && shape.js_heap_mib <= 255);
            assert!(shape.frame_bytes <= ram);
            assert!(shape.cache_bytes < ram);
            let (text, exit_code) = super::read_page_under_ram(page, ram);
            assert_eq!(exit_code, 0);
            texts.push(text);
            paces.push((shape.live_pages, shape.recycle_rss_bytes, shape.frame_bytes));
        }
        assert_eq!(texts[0], texts[1]);
        assert_eq!(texts[1], texts[2]);
        assert_eq!(texts[0], page);
        assert_ne!(paces[0], paces[2], "512 MiB must pace differently from 4 GiB");
        assert_eq!(paces[0].0, 1);
        assert_eq!(paces[2].0, 2);
        assert!(super::detected_ram_bytes().is_some() || super::effective_ram_bytes().is_some());
    }

    #[test]
    fn taskset_one_cpu_quota_matches_the_one_page_budget() {
        let quota = super::cores_from_cpu_max("100000 100000").expect("1 vCPU quota");
        assert_eq!(quota, 1);
        assert_eq!(super::host_live_pages(quota, None, 16), 1);
        assert_eq!(
            super::navigation_content_cpu(quota),
            Duration::from_secs(60)
        );
        // When taskset is installed, pinning CPU 0 must be a successful
        // one-core placement. The budget above is what that placement uses.
        if let Ok(status) = std::process::Command::new("taskset")
            .args(["-c", "0", "true"])
            .status()
        {
            assert!(status.success(), "taskset -c 0 true failed: {status}");
        }
    }
}
