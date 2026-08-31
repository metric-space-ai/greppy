//! Connect-time HTTP/HTTPS proxy that dials only policy-pinned SocketAddrs.

use crate::policy::{allowed_connect_addrs, decide_host_literal, SharedProfile, UrlDecision};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use url::Url;

const HEADER_LIMIT: usize = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

fn connect_allowed(
    profile: crate::policy::NetworkProfile,
    host: &str,
    port: u16,
    resolve: &ConnectResolve,
) -> io::Result<TcpStream> {
    let mut addrs = allowed_connect_addrs(profile, host, port, |h, p| resolve(h, p))
        .map_err(|reason| io::Error::new(io::ErrorKind::PermissionDenied, reason))?;
    addrs.sort_by_key(|addr| if addr.is_ipv4() { 0 } else { 1 });
    let mut last_error = io::Error::new(io::ErrorKind::NotFound, "no allowed address");
    for addr in addrs {
        if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: policy-proxy dial {host}:{port} -> {addr}"); }
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => {
                if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: policy-proxy connected {addr}"); }
                return Ok(stream);
            }
            Err(error) => {
                if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: policy-proxy dial {addr} failed: {error}"); }
                last_error = error;
            }
        }
    }
    Err(last_error)
}

pub(crate) type ConnectResolve =
    Arc<dyn Fn(&str, u16) -> Result<Vec<SocketAddr>, &'static str> + Send + Sync>;

pub struct PolicyProxy {
    addr: SocketAddr,
    _profile: SharedProfile,
    transferred: Arc<AtomicU64>,
}

impl PolicyProxy {
    pub fn spawn(profile: SharedProfile) -> io::Result<Self> {
        Self::spawn_with_resolve(profile, Arc::new(crate::policy::default_resolve))
    }

    pub(crate) fn spawn_with_resolve(
        profile: SharedProfile,
        resolve: ConnectResolve,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let thread_profile = profile.clone();
        let transferred = Arc::new(AtomicU64::new(0));
        let thread_transferred = Arc::clone(&transferred);
        thread::Builder::new()
            .name("greppy-policy-proxy".into())
            .spawn(move || accept_loop(listener, thread_profile, resolve, thread_transferred))?;
        Ok(Self {
            addr,
            _profile: profile,
            transferred,
        })
    }

    pub fn uri(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Total bytes relayed through the proxy in both directions since spawn.
    ///
    /// Every engine request is forced through this proxy, so this is the
    /// engine's real network volume — unlike the fixed 4096-byte accounting
    /// stub the session budget uses.
    pub fn bytes_transferred(&self) -> u64 {
        self.transferred.load(Ordering::Relaxed)
    }
}

/// Copy `reader` to `writer`, adding every relayed byte onto `counter`.
fn copy_counted(
    reader: &mut impl Read,
    writer: &mut impl Write,
    counter: &AtomicU64,
) -> io::Result<u64> {
    let mut buf = [0_u8; 16 * 1024];
    let mut total = 0_u64;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => return Ok(total),
            Ok(n) => n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        writer.write_all(&buf[..n])?;
        counter.fetch_add(n as u64, Ordering::Relaxed);
        total += n as u64;
    }
}

fn accept_loop(
    listener: TcpListener,
    profile: SharedProfile,
    resolve: ConnectResolve,
    transferred: Arc<AtomicU64>,
) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let profile = profile.clone();
        let resolve = Arc::clone(&resolve);
        let transferred = Arc::clone(&transferred);
        let _ = thread::Builder::new()
            .name("greppy-policy-proxy-conn".into())
            .spawn(move || {
                let _ = handle_client(stream, profile, resolve, transferred);
            });
    }
}

fn handle_client(
    mut client: TcpStream,
    profile: SharedProfile,
    resolve: ConnectResolve,
    transferred: Arc<AtomicU64>,
) -> io::Result<()> {
    let _ = client.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = client.set_write_timeout(Some(Duration::from_secs(30)));
    loop {
        let (head, rest) = match read_headers(&mut client) {
            Ok(pair) => pair,
            Err(_) => return Ok(()),
        };
        if head.trim().is_empty() {
            return Ok(());
        }
        let first = head.lines().next().unwrap_or("");
        let mut parts = first.split_whitespace();
        let method = parts.next().unwrap_or("").to_ascii_uppercase();
        let target = parts.next().unwrap_or("").to_owned();
        if method == "CONNECT" {
            return handle_connect(client, profile, resolve, &target, &transferred);
        }
        let keep =
            handle_forward(&mut client, &profile, &resolve, &head, rest, &target, &transferred)?;
        if !keep {
            return Ok(());
        }
    }
}

fn handle_connect(
    mut client: TcpStream,
    profile: SharedProfile,
    resolve: ConnectResolve,
    target: &str,
    transferred: &Arc<AtomicU64>,
) -> io::Result<()> {
    if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: policy-proxy CONNECT target={target}"); }
    let (host, port) = split_host_port(target, 80);
    match connect_allowed(profile.get(), &host, port, &resolve) {
        Ok(server) => {
            client.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
            splice(client, server, Arc::clone(transferred))
        }
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
            client.write_all(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )?;
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn handle_forward(
    client: &mut TcpStream,
    profile: &SharedProfile,
    resolve: &ConnectResolve,
    head: &str,
    rest: Vec<u8>,
    target: &str,
    transferred: &AtomicU64,
) -> io::Result<bool> {
    let (host, port, path) = match parse_http_target(target, head) {
        Some(parsed) => parsed,
        None => {
            client.write_all(
                b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )?;
            return Ok(false);
        }
    };
    if host_header_forbidden(profile.get(), head) {
        client.write_all(
            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )?;
        return Ok(false);
    }
    let mut server = match connect_allowed(profile.get(), &host, port, resolve) {
        Ok(server) => server,
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
            client.write_all(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )?;
            return Ok(false);
        }
        Err(error) => return Err(error),
    };
    let mut rewritten = rewrite_request_line(head, &path);
    if !rewritten.contains("\r\nHost:") && !rewritten.to_ascii_lowercase().contains("\r\nhost:") {
        let host_header = if port == 80 {
            format!("Host: {host}\r\n")
        } else {
            format!("Host: {host}:{port}\r\n")
        };
        if let Some(idx) = rewritten.find("\r\n") {
            rewritten.insert_str(idx + 2, &host_header);
        }
    }
    server.write_all(rewritten.as_bytes())?;
    server.write_all(&rest)?;
    transferred.fetch_add((rewritten.len() + rest.len()) as u64, Ordering::Relaxed);
    let extra = remaining_body_length(head).saturating_sub(rest.len());
    if extra > 0 {
        let mut buf = vec![0_u8; extra.min(HEADER_LIMIT)];
        let mut left = extra;
        while left > 0 {
            let take = left.min(buf.len());
            let n = client.read(&mut buf[..take])?;
            if n == 0 {
                break;
            }
            server.write_all(&buf[..n])?;
            transferred.fetch_add(n as u64, Ordering::Relaxed);
            left -= n;
        }
    }
    let _ = server.shutdown(std::net::Shutdown::Write);
    copy_counted(&mut server, client, transferred).map(|_| ())?;
    Ok(!connection_close(head))
}

fn connection_close(head: &str) -> bool {
    head.lines().any(|line| {
        let lower = line.to_ascii_lowercase();
        lower.starts_with("connection:") && lower.contains("close")
    })
}

fn host_header_value(head: &str) -> Option<String> {
    head.lines().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        lower
            .strip_prefix("host:")
            .map(|value| value.trim().to_owned())
    })
}

fn host_header_forbidden(profile: crate::policy::NetworkProfile, head: &str) -> bool {
    let Some(hostport) = host_header_value(head) else {
        return false;
    };
    let (host, _) = split_host_port(&hostport, 80);
    matches!(
        decide_host_literal(profile, &host),
        UrlDecision::Deny { .. }
    )
}

fn parse_http_target(target: &str, head: &str) -> Option<(String, u16, String)> {
    if let Ok(url) = Url::parse(target) {
        let host = url.host_str()?.to_owned();
        let port = url.port_or_known_default()?;
        let mut path = String::new();
        path.push_str(url.path());
        if let Some(query) = url.query() {
            path.push('?');
            path.push_str(query);
        }
        if path.is_empty() {
            path.push('/');
        }
        return Some((host, port, path));
    }
    if target.starts_with('/') {
        let host_line = head
            .lines()
            .find(|line| line.to_ascii_lowercase().starts_with("host:"))?;
        let hostport = host_line.split_once(':')?.1.trim();
        let (host, port) = split_host_port(hostport, 80);
        return Some((host, port, target.to_owned()));
    }
    None
}

fn split_host_port(hostport: &str, default_port: u16) -> (String, u16) {
    let hostport = hostport.trim();
    if let Some(rest) = hostport.strip_prefix('[') {
        if let Some((host, tail)) = rest.split_once(']') {
            let port = tail
                .strip_prefix(':')
                .and_then(|p| p.parse().ok())
                .unwrap_or(default_port);
            return (host.to_owned(), port);
        }
    }
    if let Some((host, port)) = hostport.rsplit_once(':') {
        if let Ok(port) = port.parse() {
            return (host.to_owned(), port);
        }
    }
    (hostport.to_owned(), default_port)
}

fn rewrite_request_line(head: &str, path: &str) -> String {
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut parts = first.splitn(3, ' ');
    let method = parts.next().unwrap_or("GET");
    let _target = parts.next();
    let version = parts.next().unwrap_or("HTTP/1.1");
    let mut out = format!("{method} {path} {version}\r\n");
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("proxy-connection:") {
            continue;
        }
        out.push_str(line);
        out.push_str("\r\n");
    }
    if !out.ends_with("\r\n\r\n") {
        out.push_str("\r\n");
    }
    out
}

fn remaining_body_length(head: &str) -> usize {
    for line in head.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            return value.trim().parse().unwrap_or(0);
        }
    }
    0
}

fn read_headers(stream: &mut TcpStream) -> io::Result<(String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut byte = [0_u8; 1];
    while buf.len() < HEADER_LIMIT {
        let n = stream.read(&mut byte)?;
        if n == 0 {
            break;
        }
        buf.push(byte[0]);
        if buf.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    if let Some(idx) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
        let head = String::from_utf8_lossy(&buf[..idx + 4]).into_owned();
        let rest = buf[idx + 4..].to_vec();
        return Ok((head, rest));
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "proxy request headers truncated",
    ))
}

fn splice(mut a: TcpStream, mut b: TcpStream, transferred: Arc<AtomicU64>) -> io::Result<()> {
    let _ = a.set_nodelay(true);
    let _ = b.set_nodelay(true);
    let mut a_read = a.try_clone()?;
    let mut b_write = b.try_clone()?;
    let up_counter = Arc::clone(&transferred);
    let up = thread::spawn(move || copy_counted(&mut a_read, &mut b_write, &up_counter));
    let down = copy_counted(&mut b, &mut a, &transferred);
    let _ = up.join();
    down.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::NetworkProfile;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn serve_once(body: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0_u8; 2048];
            let _ = stream.read(&mut buf);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
        });
        format!("http://{addr}/pin")
    }

    fn proxy_get(proxy: &PolicyProxy, url: &str) -> (u16, String) {
        let mut stream = TcpStream::connect(proxy.addr()).unwrap();
        let req = format!("GET {url} HTTP/1.1\r\nHost: ignored\r\nConnection: close\r\n\r\n");
        stream.write_all(req.as_bytes()).unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).unwrap();
        let text = String::from_utf8_lossy(&buf).into_owned();
        let status = text
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        (status, text)
    }

    fn proxy_connect(proxy: &PolicyProxy, hostport: &str) -> u16 {
        let mut stream = TcpStream::connect(proxy.addr()).unwrap();
        let req = format!("CONNECT {hostport} HTTP/1.1\r\nHost: {hostport}\r\n\r\n");
        stream.write_all(req.as_bytes()).unwrap();
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let mut buf = Vec::new();
        let mut tmp = [0_u8; 256];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        String::from_utf8_lossy(&buf)
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    fn read_one_http_response(stream: &mut TcpStream) -> (u16, String) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
        let mut buf = Vec::new();
        let mut tmp = [0_u8; 512];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(idx) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..idx + 4]).into_owned();
                        let mut need = remaining_body_length(&head);
                        let have = buf.len().saturating_sub(idx + 4);
                        need = need.saturating_sub(have);
                        while need > 0 {
                            match stream.read(&mut tmp) {
                                Ok(0) => break,
                                Ok(n) => {
                                    buf.extend_from_slice(&tmp[..n]);
                                    need = need.saturating_sub(n);
                                }
                                Err(_) => break,
                            }
                        }
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let text = String::from_utf8_lossy(&buf).into_owned();
        let status = text
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        (status, text)
    }

    #[test]
    fn project_proxy_pins_loopback_http() {
        let origin = serve_once(b"pinned-ok");
        let proxy = PolicyProxy::spawn(SharedProfile::new(NetworkProfile::Project)).unwrap();
        let (status, body) = proxy_get(&proxy, &origin);
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("pinned-ok"), "{body}");
    }

    #[test]
    fn project_proxy_forwards_redirect_location() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0_u8; 2048];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let (status, extra, body) = if req.contains("GET /start") {
                    ("302 Found", format!("Location: http://{addr}/end\r\n"), "")
                } else {
                    ("200 OK", String::new(), "at-end")
                };
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(body.as_bytes());
            }
        });
        let proxy = PolicyProxy::spawn(SharedProfile::new(NetworkProfile::Project)).unwrap();
        let (status, body) = proxy_get(&proxy, &format!("http://{addr}/start"));
        assert_eq!(status, 302, "{body}");
        assert!(
            body.to_ascii_lowercase().contains("location:") && body.contains("/end"),
            "missing Location /end: {body:?}"
        );
    }

    #[test]
    fn research_proxy_denies_loopback_without_dialing_metadata() {
        let origin = serve_once(b"should-not-see");
        let proxy = PolicyProxy::spawn(SharedProfile::new(NetworkProfile::Research)).unwrap();
        let (status, body) = proxy_get(&proxy, &origin);
        assert_eq!(status, 403, "{body}");
        assert!(!body.contains("should-not-see"), "{body}");
        let (status, _) = proxy_get(&proxy, "http://169.254.169.254/latest");
        assert_eq!(status, 403);
    }

    #[test]
    fn proxy_does_not_follow_redirect_to_metadata_and_denies_the_next_hop() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0_u8; 2048];
                let _ = stream.read(&mut buf);
                let extra = "Location: http://169.254.169.254/latest/meta-data/\r\n";
                let header = format!(
                    "HTTP/1.1 302 Found\r\nContent-Length: 0\r\n{extra}Connection: close\r\n\r\n"
                );
                let _ = stream.write_all(header.as_bytes());
            }
        });
        let proxy = PolicyProxy::spawn(SharedProfile::new(NetworkProfile::Project)).unwrap();
        let (status, body) = proxy_get(&proxy, &format!("http://{addr}/jump"));
        assert_eq!(status, 302, "{body}");
        assert!(
            body.to_ascii_lowercase().contains("location:")
                && body.contains("169.254.169.254"),
            "proxy must surface the hop, not fetch it: {body:?}"
        );
        let (status, hop) = proxy_get(&proxy, "http://169.254.169.254/latest/meta-data/");
        assert_eq!(status, 403, "{hop}");
        assert!(!hop.to_ascii_lowercase().contains("ami-id"), "{hop}");
    }

    #[test]
    fn proxy_denies_later_lookup_that_rebinds_to_metadata() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let origin = serve_once(b"first-hop");
        let fixture: std::net::SocketAddr = origin
            .trim_start_matches("http://")
            .trim_end_matches("/pin")
            .parse()
            .unwrap();
        let lookups = Arc::new(AtomicUsize::new(0));
        let resolve = {
            let lookups = Arc::clone(&lookups);
            Arc::new(move |host: &str, port: u16| {
                if host != "flip.test" {
                    return crate::policy::default_resolve(host, port);
                }
                let n = lookups.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Ok(vec![std::net::SocketAddr::new(fixture.ip(), fixture.port())])
                } else {
                    Ok(vec!["169.254.169.254:80".parse().unwrap()])
                }
            }) as ConnectResolve
        };
        let proxy = PolicyProxy::spawn_with_resolve(
            SharedProfile::new(NetworkProfile::Project),
            resolve,
        )
        .unwrap();
        let (status, body) = proxy_get(&proxy, "http://flip.test/pin");
        assert_eq!(status, 200, "first lookup must pin the fixture: {body}");
        assert!(body.contains("first-hop"), "{body}");
        let (status, hop) = proxy_get(&proxy, "http://flip.test/pin");
        assert_eq!(status, 403, "rebound metadata lookup must be denied: {hop}");
        assert!(!hop.contains("first-hop"), "{hop}");
        assert!(lookups.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn proxy_denies_metadata_hostname_connect_without_dns() {
        let resolve: ConnectResolve = Arc::new(|host, _| {
            panic!("must not resolve metadata hostname {host}");
        });
        let proxy = PolicyProxy::spawn_with_resolve(
            SharedProfile::new(NetworkProfile::Project),
            resolve,
        )
        .unwrap();
        let status = proxy_connect(&proxy, "metadata.google.internal:80");
        assert_eq!(status, 403, "metadata hostname CONNECT must fail closed");
    }

    #[test]
    fn proxy_denies_https_connect_after_rebind_to_metadata() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let fixture = listener.local_addr().unwrap();
        thread::spawn(move || {
            let _ = listener.accept();
        });
        let lookups = Arc::new(AtomicUsize::new(0));
        let resolve = {
            let lookups = Arc::clone(&lookups);
            Arc::new(move |host: &str, port: u16| {
                if host != "flip.test" {
                    return crate::policy::default_resolve(host, port);
                }
                let n = lookups.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Ok(vec![fixture])
                } else {
                    Ok(vec!["169.254.169.254:443".parse().unwrap()])
                }
            }) as ConnectResolve
        };
        let proxy = PolicyProxy::spawn_with_resolve(
            SharedProfile::new(NetworkProfile::Project),
            resolve,
        )
        .unwrap();
        let first = proxy_connect(&proxy, "flip.test:443");
        assert_eq!(first, 200, "first CONNECT must pin the fixture");
        let second = proxy_connect(&proxy, "flip.test:443");
        assert_eq!(second, 403, "rebound CONNECT must be denied");
        assert!(lookups.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn proxy_denies_metadata_host_header_on_allowed_url() {
        let resolve: ConnectResolve = Arc::new(|host, _| {
            panic!("must not resolve when Host header is metadata: {host}");
        });
        let proxy = PolicyProxy::spawn_with_resolve(
            SharedProfile::new(NetworkProfile::Project),
            resolve,
        )
        .unwrap();
        let mut stream = TcpStream::connect(proxy.addr()).unwrap();
        stream
            .write_all(
                b"GET http://example.test/pin HTTP/1.1\r\nHost: 169.254.169.254\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).unwrap();
        let text = String::from_utf8_lossy(&buf);
        let status = text
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        assert_eq!(status, 403, "{text}");
    }

    #[test]
    fn proxy_repins_keep_alive_http_and_denies_rebind() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let origin = serve_once(b"first-hop");
        let fixture: std::net::SocketAddr = origin
            .trim_start_matches("http://")
            .trim_end_matches("/pin")
            .parse()
            .unwrap();
        let lookups = Arc::new(AtomicUsize::new(0));
        let resolve = {
            let lookups = Arc::clone(&lookups);
            Arc::new(move |host: &str, port: u16| {
                if host != "flip.test" {
                    return crate::policy::default_resolve(host, port);
                }
                let n = lookups.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Ok(vec![std::net::SocketAddr::new(fixture.ip(), fixture.port())])
                } else {
                    Ok(vec!["169.254.169.254:80".parse().unwrap()])
                }
            }) as ConnectResolve
        };
        let proxy = PolicyProxy::spawn_with_resolve(
            SharedProfile::new(NetworkProfile::Project),
            resolve,
        )
        .unwrap();
        let mut stream = TcpStream::connect(proxy.addr()).unwrap();
        stream
            .write_all(b"GET http://flip.test/pin HTTP/1.1\r\nHost: flip.test\r\nConnection: keep-alive\r\n\r\n")
            .unwrap();
        let (status, body) = read_one_http_response(&mut stream);
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("first-hop"), "{body}");
        stream
            .write_all(b"GET http://flip.test/pin HTTP/1.1\r\nHost: flip.test\r\nConnection: close\r\n\r\n")
            .unwrap();
        let (status, hop) = read_one_http_response(&mut stream);
        assert_eq!(status, 403, "keep-alive rebind must re-pin and deny: {hop}");
        assert!(!hop.contains("first-hop"), "{hop}");
        assert!(lookups.load(Ordering::SeqCst) >= 2);
    }
}
