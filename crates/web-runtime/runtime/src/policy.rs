//! Page network policy (guide §16.2).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

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

    fn as_code(self) -> u8 {
        match self {
            Self::Research => 0,
            Self::Project => 1,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Project,
            _ => Self::Research,
        }
    }
}

/// Process-wide network profile shared with the connect-time policy proxy.
#[derive(Clone)]
pub struct SharedProfile(Arc<AtomicU8>);

impl SharedProfile {
    pub fn new(profile: NetworkProfile) -> Self {
        Self(Arc::new(AtomicU8::new(profile.as_code())))
    }

    pub fn get(&self) -> NetworkProfile {
        NetworkProfile::from_code(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, profile: NetworkProfile) {
        self.0.store(profile.as_code(), Ordering::Relaxed);
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
    match pin_connect_addr(profile, host, 80) {
        Ok(_) => UrlDecision::Allow,
        Err(reason) => UrlDecision::Deny { reason },
    }
}

/// Resolve `host` and pick a SocketAddr that is allowed *and* is the address
/// the caller must dial. Metadata in the answer denies the whole name.
pub fn pin_connect_addr(
    profile: NetworkProfile,
    host: &str,
    port: u16,
) -> Result<SocketAddr, &'static str> {
    pin_connect_addr_with(profile, host, port, default_resolve)
}

fn default_resolve(host: &str, port: u16) -> Result<Vec<SocketAddr>, &'static str> {
    (host, port)
        .to_socket_addrs()
        .map(|iter| iter.collect())
        .map_err(|_| "DNS resolution failed")
}

pub fn pin_connect_addr_with(
    profile: NetworkProfile,
    host: &str,
    port: u16,
    resolve: impl Fn(&str, u16) -> Result<Vec<SocketAddr>, &'static str>,
) -> Result<SocketAddr, &'static str> {
    let host = host.trim_matches(|ch| ch == '[' || ch == ']');
    if let Ok(ip) = host.parse::<IpAddr>() {
        return match decide_ip(profile, ip) {
            UrlDecision::Allow => Ok(SocketAddr::new(ip, port)),
            UrlDecision::Deny { reason } => Err(reason),
        };
    }
    if host.eq_ignore_ascii_case("localhost") {
        return pin_connect_addr_with(profile, "127.0.0.1", port, resolve);
    }
    let addrs = resolve(host, port)?;
    if addrs.is_empty() {
        return Err("DNS resolution failed");
    }
    let mut first_allowed = None;
    for addr in addrs {
        match decide_ip(profile, addr.ip()) {
            UrlDecision::Deny {
                reason: "cloud metadata endpoint denied",
            } => return Err("cloud metadata endpoint denied"),
            UrlDecision::Deny { .. } => {}
            UrlDecision::Allow => {
                if first_allowed.is_none() {
                    first_allowed = Some(addr);
                }
            }
        }
    }
    first_allowed.ok_or("LAN and non-public endpoints denied")
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

    #[test]
    fn pin_connect_addr_denies_metadata_and_lan_literals() {
        assert!(pin_connect_addr(
            NetworkProfile::Project,
            "169.254.169.254",
            80
        )
        .is_err());
        assert!(pin_connect_addr(NetworkProfile::Project, "10.1.2.3", 80).is_err());
        assert!(pin_connect_addr(NetworkProfile::Research, "127.0.0.1", 9).is_err());
        assert!(pin_connect_addr(NetworkProfile::Project, "127.0.0.1", 9).is_ok());
    }

    #[test]
    fn pin_connect_addr_skips_lan_but_denies_metadata_in_the_answer() {
        let public = "8.8.8.8:80".parse().unwrap();
        let lan = "10.0.0.1:80".parse().unwrap();
        let meta = "169.254.169.254:80".parse().unwrap();
        let pinned = pin_connect_addr_with(
            NetworkProfile::Research,
            "mixed.test",
            80,
            |_, _| Ok(vec![lan, public]),
        )
        .expect("public A record remains usable");
        assert_eq!(pinned, public);
        assert!(pin_connect_addr_with(
            NetworkProfile::Research,
            "rebind.test",
            80,
            |_, _| Ok(vec![public, meta]),
        )
        .is_err());
    }
}
