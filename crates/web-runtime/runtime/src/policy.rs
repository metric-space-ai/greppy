//! Page network policy (guide §16.2).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};

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
    let literal = decide_host_literal(profile, host);
    if !matches!(literal, UrlDecision::Allow) {
        return literal;
    }
    if is_ip_literal(host) || host.eq_ignore_ascii_case("localhost") {
        return UrlDecision::Allow;
    }
    decide_resolved_hostname(profile, host)
}

pub fn decide_ip(profile: NetworkProfile, ip: IpAddr) -> UrlDecision {
    if is_metadata_ip(ip) {
        return UrlDecision::Deny {
            reason: "cloud metadata endpoint denied",
        };
    }
    if ip_is_loopback(ip) {
        return match profile {
            NetworkProfile::Project => UrlDecision::Allow,
            NetworkProfile::Research => UrlDecision::Deny {
                reason: "research profile denies loopback and private networks",
            },
        };
    }
    if ip_is_non_public(ip) {
        return UrlDecision::Deny {
            reason: "LAN and non-public endpoints denied",
        };
    }
    UrlDecision::Allow
}

fn decide_host_literal(profile: NetworkProfile, host: &str) -> UrlDecision {
    if is_metadata_host(host) {
        return UrlDecision::Deny {
            reason: "cloud metadata endpoint denied",
        };
    }
    if is_loopback_host(host) {
        return match profile {
            NetworkProfile::Project => UrlDecision::Allow,
            NetworkProfile::Research => UrlDecision::Deny {
                reason: "research profile denies loopback and private networks",
            },
        };
    }
    if is_blocked_ip(host) {
        return UrlDecision::Deny {
            reason: "LAN and non-public endpoints denied",
        };
    }
    UrlDecision::Allow
}

fn decide_resolved_hostname(profile: NetworkProfile, host: &str) -> UrlDecision {
    let Ok(addrs) = (host, 80).to_socket_addrs() else {
        return UrlDecision::Allow;
    };
    for addr in addrs {
        if let UrlDecision::Deny { reason } = decide_ip(profile, addr.ip()) {
            return UrlDecision::Deny { reason };
        }
    }
    UrlDecision::Allow
}

fn is_metadata_host(host: &str) -> bool {
    let trimmed = host.trim_matches(|ch| ch == '[' || ch == ']');
    if trimmed.eq_ignore_ascii_case("metadata.google.internal")
        || trimmed.ends_with(".metadata.google.internal")
        || trimmed.eq_ignore_ascii_case("instance-data")
        || trimmed == "169.254.169.254"
        || trimmed.eq_ignore_ascii_case("::ffff:169.254.169.254")
        || trimmed == "100.100.100.200"
    {
        return true;
    }
    match trimmed.parse::<IpAddr>() {
        Ok(ip) => is_metadata_ip(ip),
        Err(_) => false,
    }
}

fn is_metadata_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(addr) => {
            addr == Ipv4Addr::new(169, 254, 169, 254) || addr == Ipv4Addr::new(100, 100, 100, 200)
        }
        IpAddr::V6(addr) => {
            if let Some(mapped) = addr.to_ipv4_mapped() {
                return is_metadata_ip(IpAddr::V4(mapped));
            }
            addr == Ipv6Addr::new(0xfd00, 0xec2, 0, 0, 0, 0, 0, 0x254)
        }
    }
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let trimmed = host.trim_matches(|ch| ch == '[' || ch == ']');
    match trimmed.parse::<IpAddr>() {
        Ok(ip) => ip_is_loopback(ip),
        Err(_) => false,
    }
}

fn is_ip_literal(host: &str) -> bool {
    let trimmed = host.trim_matches(|ch| ch == '[' || ch == ']');
    trimmed.parse::<IpAddr>().is_ok()
}

fn is_blocked_ip(host: &str) -> bool {
    let trimmed = host.trim_matches(|ch| ch == '[' || ch == ']');
    match trimmed.parse::<IpAddr>() {
        Ok(ip) => ip_is_non_public(ip),
        Err(_) => false,
    }
}

fn ip_is_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(addr) => addr.is_loopback(),
        IpAddr::V6(addr) => {
            if let Some(mapped) = addr.to_ipv4_mapped() {
                return mapped.is_loopback();
            }
            addr.is_loopback()
        }
    }
}

fn ip_is_non_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(addr) => ipv4_non_public(addr),
        IpAddr::V6(addr) => {
            if let Some(mapped) = addr.to_ipv4_mapped() {
                return ipv4_non_public(mapped);
            }
            addr.is_multicast()
                || addr.is_unspecified()
                || ipv6_unique_local(addr)
                || ipv6_link_local(addr)
        }
    }
}

fn ipv4_non_public(addr: Ipv4Addr) -> bool {
    addr.is_private()
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
        assert_eq!(
            decide_url(NetworkProfile::Project, "http://localhost/x"),
            UrlDecision::Allow
        );
    }

    #[test]
    fn both_profiles_deny_lan() {
        for url in [
            "http://192.168.1.1/",
            "http://10.0.0.5/",
            "http://172.16.0.1/",
            "http://100.64.0.1/",
            "http://[fd00::1]/",
            "http://[fe80::1]/",
        ] {
            assert!(
                matches!(
                    decide_url(NetworkProfile::Project, url),
                    UrlDecision::Deny {
                        reason: "LAN and non-public endpoints denied"
                    }
                ),
                "project should deny LAN {url}"
            );
            assert!(
                matches!(decide_url(NetworkProfile::Research, url), UrlDecision::Deny { .. }),
                "research should deny LAN {url}"
            );
        }
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
        assert!(matches!(
            decide_url(NetworkProfile::Project, "http://100.100.100.200/latest"),
            UrlDecision::Deny { .. }
        ));
        assert!(matches!(
            decide_url(NetworkProfile::Project, "http://[fd00:ec2::254]/latest"),
            UrlDecision::Deny { .. }
        ));
        assert!(matches!(
            decide_url(NetworkProfile::Research, "http://instance-data/latest"),
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
    fn resolved_loopback_is_denied_for_research_allowed_for_project() {
        assert!(matches!(
            decide_ip(NetworkProfile::Research, IpAddr::V4(Ipv4Addr::LOCALHOST)),
            UrlDecision::Deny { .. }
        ));
        assert_eq!(
            decide_ip(NetworkProfile::Project, IpAddr::V4(Ipv4Addr::LOCALHOST)),
            UrlDecision::Allow
        );
    }

    #[test]
    fn resolved_metadata_and_lan_are_denied_for_both_profiles() {
        let metadata = IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254));
        let lan = IpAddr::V4(Ipv4Addr::new(10, 1, 2, 3));
        assert!(matches!(
            decide_ip(NetworkProfile::Project, metadata),
            UrlDecision::Deny {
                reason: "cloud metadata endpoint denied"
            }
        ));
        assert!(matches!(
            decide_ip(NetworkProfile::Research, metadata),
            UrlDecision::Deny { .. }
        ));
        assert!(matches!(
            decide_ip(NetworkProfile::Project, lan),
            UrlDecision::Deny {
                reason: "LAN and non-public endpoints denied"
            }
        ));
        assert!(matches!(
            decide_ip(NetworkProfile::Research, lan),
            UrlDecision::Deny { .. }
        ));
    }

    #[test]
    fn public_https_is_allowed() {
        assert_eq!(
            decide_host_literal(NetworkProfile::Research, "example.com"),
            UrlDecision::Allow
        );
        assert_eq!(
            decide_url(NetworkProfile::Research, "https://example.com/path"),
            UrlDecision::Allow
        );
    }
}
