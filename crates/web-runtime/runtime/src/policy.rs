//! Page network policy (guide §16.2).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetworkProfile {
    Research,
    Project,
}

impl NetworkProfile {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "research" => Some(Self::Research),
            "project" => Some(Self::Project),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Research => "research",
            Self::Project => "project",
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum UrlDecision {
    Allow,
    Deny { reason: &'static str },
}

pub fn decide_url(profile: NetworkProfile, url: &str) -> UrlDecision {
    let Ok(parsed) = url::Url::parse(url) else {
        return UrlDecision::Deny {
            reason: "unparseable URL",
        };
    };
    match parsed.scheme() {
        "http" | "https" => {}
        "data" | "about" => return UrlDecision::Allow,
        _ => {
            return UrlDecision::Deny {
                reason: "scheme is not http(s)",
            };
        }
    }
    let Some(host) = parsed.host_str() else {
        return UrlDecision::Deny {
            reason: "URL has no host",
        };
    };
    if is_metadata_host(host) {
        return UrlDecision::Deny {
            reason: "cloud metadata endpoint denied",
        };
    }
    if host.eq_ignore_ascii_case("localhost") || is_blocked_ip(host) {
        return match profile {
            NetworkProfile::Project => UrlDecision::Allow,
            NetworkProfile::Research => UrlDecision::Deny {
                reason: "research profile denies loopback and private networks",
            },
        };
    }
    UrlDecision::Allow
}

fn is_metadata_host(host: &str) -> bool {
    let trimmed = host.trim_matches(|ch| ch == '[' || ch == ']');
    if trimmed.eq_ignore_ascii_case("metadata.google.internal")
        || trimmed.ends_with(".metadata.google.internal")
        || trimmed == "169.254.169.254"
        || trimmed.eq_ignore_ascii_case("::ffff:169.254.169.254")
    {
        return true;
    }
    match trimmed.parse::<IpAddr>() {
        Ok(IpAddr::V4(addr)) => addr == Ipv4Addr::new(169, 254, 169, 254),
        Ok(IpAddr::V6(addr)) => addr.to_ipv4_mapped() == Some(Ipv4Addr::new(169, 254, 169, 254)),
        Err(_) => false,
    }
}

fn is_blocked_ip(host: &str) -> bool {
    let trimmed = host.trim_matches(|ch| ch == '[' || ch == ']');
    let Ok(ip) = trimmed.parse::<IpAddr>() else {
        return false;
    };
    match ip {
        IpAddr::V4(addr) => ipv4_blocked(addr),
        IpAddr::V6(addr) => {
            if let Some(mapped) = addr.to_ipv4_mapped() {
                return ipv4_blocked(mapped);
            }
            addr.is_loopback()
                || addr.is_multicast()
                || addr.is_unspecified()
                || ipv6_unique_local(addr)
                || ipv6_link_local(addr)
        }
    }
}

fn ipv4_blocked(addr: Ipv4Addr) -> bool {
    addr.is_loopback()
        || addr.is_private()
        || addr.is_link_local()
        || addr.is_multicast()
        || addr.is_unspecified()
        || is_cgnat(addr)
}

fn ipv6_unique_local(addr: Ipv6Addr) -> bool {
    addr.segments()[0] & 0xfe00 == 0xfc00
}

fn ipv6_link_local(addr: Ipv6Addr) -> bool {
    addr.segments()[0] & 0xffc0 == 0xfe80
}

fn is_cgnat(addr: Ipv4Addr) -> bool {
    let octets = addr.octets();
    octets[0] == 100 && octets[1] >= 64 && octets[1] <= 127
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_denies_loopback_project_allows() {
        assert!(matches!(
            decide_url(NetworkProfile::Research, "http://127.0.0.1/x"),
            UrlDecision::Deny { .. }
        ));
        assert_eq!(
            decide_url(NetworkProfile::Project, "http://127.0.0.1/x"),
            UrlDecision::Allow
        );
    }

    #[test]
    fn both_profiles_deny_metadata() {
        assert!(matches!(
            decide_url(NetworkProfile::Project, "http://169.254.169.254/latest"),
            UrlDecision::Deny { .. }
        ));
        assert!(matches!(
            decide_url(NetworkProfile::Research, "http://metadata.google.internal/"),
            UrlDecision::Deny { .. }
        ));
        assert!(matches!(
            decide_url(
                NetworkProfile::Project,
                "http://[::ffff:169.254.169.254]/latest"
            ),
            UrlDecision::Deny { .. }
        ));
    }

    #[test]
    fn research_denies_mapped_loopback_and_link_local() {
        assert!(matches!(
            decide_url(NetworkProfile::Research, "http://[::ffff:127.0.0.1]/"),
            UrlDecision::Deny { .. }
        ));
        assert!(matches!(
            decide_url(NetworkProfile::Research, "http://[fe80::1]/"),
            UrlDecision::Deny { .. }
        ));
        assert!(matches!(
            decide_url(NetworkProfile::Research, "http://[fd00::1]/"),
            UrlDecision::Deny { .. }
        ));
    }

    #[test]
    fn public_https_is_allowed() {
        assert_eq!(
            decide_url(NetworkProfile::Research, "https://example.com/path"),
            UrlDecision::Allow
        );
    }
}