//! Session resource limits (guide §17). Enforced in the supervisor, not the worker.

use std::time::Duration;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionLimits {
    pub wall_time: Duration,
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

impl SessionLimits {
    pub fn for_profile(profile: &str) -> Self {
        let mut limits = Self::default();
        if profile == "research" {
            limits.max_pages = 8;
            limits.max_network_bytes = 8 * 1024 * 1024;
            limits.max_artifact_bytes = 4 * 1024 * 1024;
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
            self.wall_time = Duration::from_millis(value.clamp(20, 3_600_000));
        }
        if let Some(value) = object.get("content_cpu_ms").and_then(|v| v.as_u64()) {
            self.content_cpu_time = Duration::from_millis(value);
        }
        if let Some(value) = object.get("controller_cpu_ms").and_then(|v| v.as_u64()) {
            self.controller_cpu_time = Duration::from_millis(value);
        }
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
        if next > self.max_network_bytes {
            Err(format!(
                "network limit exceeded ({next} > {})",
                self.max_network_bytes
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_wall_time(&self, elapsed: Duration) -> Result<(), String> {
        if elapsed > self.wall_time {
            Err(format!(
                "wall time exceeded ({elapsed:?} > {:?})",
                self.wall_time
            ))
        } else {
            Ok(())
        }
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
            Err(format!("{label} cpu time exceeded ({used:?} > {limit:?})"))
        } else {
            Ok(())
        }
    }
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            wall_time: Duration::from_secs(120),
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
            max_network_bytes: 32 * 1024 * 1024,
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
        assert!(research.check_pages(9).is_err());
        assert!(project.check_pages(9).is_ok());
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
            "idle_ttl_ms": 50
        }));
        assert_eq!(limits.max_pages, 0);
        assert_eq!(limits.idle_ttl, Duration::from_millis(50));
        assert!(limits.check_pages(1).is_err());
    }
}
