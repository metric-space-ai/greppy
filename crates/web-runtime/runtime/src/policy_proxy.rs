//! Connect-time HTTP/HTTPS proxy that dials only policy-pinned SocketAddrs.

use crate::policy::{pin_connect_addr, SharedProfile};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::Duration;
use url::Url;

const HEADER_LIMIT: usize = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct PolicyProxy {
    addr: SocketAddr,
    _profile: SharedProfile,
}

impl PolicyProxy {
    pub fn spawn(profile: SharedProfile) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let thread_profile = profile.clone();
        thread::Builder::new()
            .name("greppy-policy-proxy".into())
            .spawn(move || accept_loop(listener, thread_profile))?;
        Ok(Self {
            addr,
            _profile: profile,
        })
    }

    pub fn uri(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

fn accept_loop(listener: TcpListener, profile: SharedProfile) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let profile = profile.clone();
        let _ = thread::Builder::new()
            .name("greppy-policy-proxy-conn".into())
            .spawn(move || {
                let _ = handle_client(stream, profile);
            });
    }
}

fn handle_client(mut client: TcpStream, profile: SharedProfile) -> io::Result<()> {
    let _ = client.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = client.set_write_timeout(Some(Duration::from_secs(30)));
    let (head, rest) = read_headers(&mut client)?;
    let first = head.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_ascii_uppercase();
    let target = parts.next().unwrap_or("");
    if method == "CONNECT" {
        return handle_connect(client, profile, target);
    }
    handle_forward(client, profile, &head, rest, target)
}

fn handle_connect(mut client: TcpStream, profile: SharedProfile, target: &str) -> io::Result<()> {
    let (host, port) = split_host_port(target, 443);
    match pin_connect_addr(profile.get(), &host, port) {
        Ok(addr) => {
            let server = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)?;
            client.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
            splice(client, server)
        }
        Err(_) => {
            client.write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
            Ok(())
        }
    }
}

fn handle_forward(
    mut client: TcpStream,
    profile: SharedProfile,
    head: &str,
    rest: Vec<u8>,
    target: &str,
) -> io::Result<()> {
    let (host, port, path) = match parse_http_target(target, head) {
        Some(parsed) => parsed,
        None => {
            client.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
            return Ok(());
        }
    };
    let addr = match pin_connect_addr(profile.get(), &host, port) {
        Ok(addr) => addr,
        Err(_) => {
            client.write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
            return Ok(());
        }
    };
    let mut server = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)?;
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
            left -= n;
        }
    }
    let _ = server.shutdown(std::net::Shutdown::Write);
    io::copy(&mut server, &mut client).map(|_| ())
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
        let host_line = head.lines().find(|line| line.to_ascii_lowercase().starts_with("host:"))?;
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

fn splice(mut a: TcpStream, mut b: TcpStream) -> io::Result<()> {
    let _ = a.set_nodelay(true);
    let _ = b.set_nodelay(true);
    let mut a_read = a.try_clone()?;
    let mut b_write = b.try_clone()?;
    let up = thread::spawn(move || io::copy(&mut a_read, &mut b_write));
    let down = io::copy(&mut b, &mut a);
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
            let Ok((mut stream, _)) = listener.accept() else { return };
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
        let req = format!(
            "GET {url} HTTP/1.1\r\nHost: ignored\r\nConnection: close\r\n\r\n"
        );
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
                    (
                        "302 Found",
                        format!("Location: http://{addr}/end\r\n"),
                        "",
                    )
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
}
