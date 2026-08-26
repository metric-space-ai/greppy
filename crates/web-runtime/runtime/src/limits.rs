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
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            wall_time: Duration::from_secs(120),
            controller_cpu_time: Duration::from_secs(30),
            content_cpu_time: Duration::from_secs(30),
            controller_heap_bytes: 256 * 1024 * 1024,
            content_rss_bytes: 512 * 1024 * 1024,
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
}
