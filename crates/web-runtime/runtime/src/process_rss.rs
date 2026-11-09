//! Linux RSS observation without creating a process for each sample.
//!
//! /proc/PID/status reports VmRSS in kB (1024 bytes). This remains an
//! instantaneous, kernel-accounted RSS sample; it is not a PSS or leak check.

pub(crate) fn sample_rss_bytes(pid: u32) -> u64 {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()
        .and_then(|status| parse_rss_bytes(&status))
        .unwrap_or(0)
}

fn parse_rss_bytes(status: &str) -> Option<u64> {
    let value = status.lines().find_map(|line| line.strip_prefix("VmRSS:"))?;
    let mut fields = value.split_whitespace();
    let kibibytes = fields.next()?.parse::<u64>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    Some(kibibytes.saturating_mul(1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rss_is_resident_kibibytes_not_virtual_size_or_another_unit() {
        let status = "Name:\tworker\nVmSize:\t999999 kB\nVmHWM:\t9000 kB\nVmRSS:\t 12345 kB\nRssAnon:\t123 kB\n";
        assert_eq!(parse_rss_bytes(status), Some(12_641_280));
        assert_eq!(parse_rss_bytes("VmRSS:\t0 kB\n"), Some(0));
        assert_eq!(
            parse_rss_bytes(&format!("VmRSS:\t{} kB\n", u64::MAX)),
            Some(u64::MAX)
        );
        for status in [
            "VmSize:\t123 kB\n",
            "VmRSS:\t123 MB\n",
            "VmRSS:\t123\n",
            "VmRSS:\t-123 kB\n",
            "VmRSS:\t18446744073709551616 kB\n",
            "VmRSS:\t123 kB extra\n",
        ] {
            assert_eq!(parse_rss_bytes(status), None, "{status:?}");
        }
    }

    #[test]
    fn samples_the_real_kernel_status_and_preserves_missing_process_zero() {
        assert!(sample_rss_bytes(std::process::id()) > 0);
        // Linux PID_MAX_LIMIT is below u32::MAX; this cannot identify a process.
        assert_eq!(sample_rss_bytes(u32::MAX), 0);
    }
}
