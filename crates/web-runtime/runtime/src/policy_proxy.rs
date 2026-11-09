//! Connect-time HTTP/HTTPS proxy that dials only policy-pinned SocketAddrs.

#[cfg(test)]
#[test]
fn dropping_proxy_terminates_an_active_tunnel() {
    check_proxy_tunnel_drop(false);
    check_proxy_tunnel_drop(true);
}

#[cfg(test)]
fn check_proxy_tunnel_drop(half_closed: bool) {
    let origin = TcpListener::bind("127.0.0.1:0").unwrap();
    origin.set_nonblocking(true).unwrap();
    let proxy =
        PolicyProxy::spawn(SharedProfile::new(crate::policy::NetworkProfile::Project)).unwrap();
    let active = Arc::clone(&proxy.active_connections);
    let mut client = TcpStream::connect(proxy.addr()).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    write!(
        client,
        "CONNECT {} HTTP/1.1\r\nHost: {}\r\n\r\n",
        origin.local_addr().unwrap(),
        origin.local_addr().unwrap()
    )
    .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let (mut server, _) = loop {
        match origin.accept() {
            Ok(pair) => break pair,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "proxy did not connect"
                );
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => panic!("{error}"),
        }
    };
    let mut response = Vec::new();
    while !response.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        client.read_exact(&mut byte).unwrap();
        response.push(byte[0]);
    }
    assert!(response.starts_with(b"HTTP/1.1 200"));
    if half_closed {
        client.shutdown(std::net::Shutdown::Write).unwrap();
        server.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut request = Vec::new();
        server.read_to_end(&mut request).unwrap();
        assert!(request.is_empty());
        // Origin keeps its write side open with no response/EOF. Only that
        // descriptor is now readable-interest eligible in the relay poll.
    }
    assert_eq!(active.load(Ordering::Acquire), 1);
    drop(proxy);
    while active.load(Ordering::Acquire) != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "closed context retained an active tunnel"
        );
        thread::sleep(Duration::from_millis(2));
    }
    server
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    assert_eq!(server.read(&mut [0]).unwrap(), 0);
}

#[cfg(test)]
#[test]
fn dropping_proxy_releases_its_listener() {
    let proxy =
        PolicyProxy::spawn(SharedProfile::new(crate::policy::NetworkProfile::Project)).unwrap();
    let address = proxy.addr();
    drop(proxy);
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while std::net::TcpStream::connect_timeout(&address, Duration::from_millis(20)).is_ok() {
        assert!(
            std::time::Instant::now() < deadline,
            "closed context retained its policy listener"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}
use crate::policy::{allowed_connect_addrs, decide_host_literal, SharedProfile, UrlDecision};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use url::Url;

const HEADER_LIMIT: usize = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const PER_PROXY_CONNECTION_LIMIT: usize = 32;
const GLOBAL_CONNECTION_LIMIT: usize = 128;
static GLOBAL_ACTIVE_CONNECTIONS: AtomicUsize = AtomicUsize::new(0);

struct ConnectionPermit {
    local: Arc<AtomicUsize>,
}

impl ConnectionPermit {
    fn acquire(local: &Arc<AtomicUsize>) -> Option<Self> {
        if !increment_below(local, PER_PROXY_CONNECTION_LIMIT) {
            return None;
        }
        if !increment_below(&GLOBAL_ACTIVE_CONNECTIONS, GLOBAL_CONNECTION_LIMIT) {
            local.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(Self {
            local: Arc::clone(local),
        })
    }
}

impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        self.local.fetch_sub(1, Ordering::AcqRel);
        GLOBAL_ACTIVE_CONNECTIONS.fetch_sub(1, Ordering::AcqRel);
    }
}

fn increment_below(counter: &AtomicUsize, limit: usize) -> bool {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        if current >= limit {
            return false;
        }
        match counter.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(observed) => current = observed,
        }
    }
}

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
        if crate::supervisor::phase_trace_enabled() {
            if crate::supervisor::phase_trace_enabled() {
                eprintln!("web-runtime: policy-proxy dial {host}:{port} -> {addr}");
            }
        }
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => {
                if crate::supervisor::phase_trace_enabled() {
                    if crate::supervisor::phase_trace_enabled() {
                        eprintln!("web-runtime: policy-proxy connected {addr}");
                    }
                }
                return Ok(stream);
            }
            Err(error) => {
                if crate::supervisor::phase_trace_enabled() {
                    if crate::supervisor::phase_trace_enabled() {
                        eprintln!("web-runtime: policy-proxy dial {addr} failed: {error}");
                    }
                }
                last_error = error;
            }
        }
    }
    Err(last_error)
}

pub(crate) type ConnectResolve =
    Arc<dyn Fn(&str, u16) -> Result<Vec<SocketAddr>, &'static str> + Send + Sync>;

struct ProxyLifetime {
    closing: AtomicBool,
    next_connection: AtomicU64,
    clients: Mutex<HashMap<u64, TcpStream>>,
}

struct LiveConnection {
    lifetime: Arc<ProxyLifetime>,
    id: u64,
}

impl ProxyLifetime {
    fn register(self: &Arc<Self>, client: &TcpStream) -> io::Result<Option<LiveConnection>> {
        let mut clients = self
            .clients
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.closing.load(Ordering::Acquire) {
            return Ok(None);
        }
        let duplicate = client.try_clone()?;
        let id = self.next_connection.fetch_add(1, Ordering::Relaxed);
        clients.insert(id, duplicate);
        Ok(Some(LiveConnection {
            lifetime: Arc::clone(self),
            id,
        }))
    }

    fn close(&self) {
        self.closing.store(true, Ordering::Release);
        let clients = self
            .clients
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for client in clients.values() {
            // Wake blocked header reads and the tunnel poll without a timer or
            // another thread. The relay then closes both directions.
            let _ = client.shutdown(std::net::Shutdown::Both);
        }
    }
}

impl Drop for LiveConnection {
    fn drop(&mut self) {
        self.lifetime
            .clients
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

pub struct PolicyProxy {
    addr: SocketAddr,
    _profile: SharedProfile,
    transferred: Arc<AtomicU64>,
    active_connections: Arc<AtomicUsize>,
    lifetime: Arc<ProxyLifetime>,
    listener_thread: Option<thread::JoinHandle<()>>,
}

impl Drop for PolicyProxy {
    fn drop(&mut self) {
        self.lifetime.close();
        // Wake the listener's poll. No request/connection worker is started
        // after the closing flag, including this local wake connection.
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(20));
        if let Some(worker) = self.listener_thread.take() {
            let _ = worker.join();
        }
    }
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
        listener.set_nonblocking(true)?;
        let lifetime = Arc::new(ProxyLifetime {
            closing: AtomicBool::new(false),
            next_connection: AtomicU64::new(0),
            clients: Mutex::new(HashMap::new()),
        });
        let thread_lifetime = Arc::clone(&lifetime);
        let thread_profile = profile.clone();
        let transferred = Arc::new(AtomicU64::new(0));
        let active_connections = Arc::new(AtomicUsize::new(0));
        let thread_transferred = Arc::clone(&transferred);
        let thread_active_connections = Arc::clone(&active_connections);
        let listener_thread = thread::Builder::new()
            .name("greppy-policy-proxy".into())
            .spawn(move || {
                accept_loop(
                    listener,
                    thread_profile,
                    resolve,
                    thread_transferred,
                    thread_active_connections,
                    thread_lifetime,
                )
            })?;
        Ok(Self {
            addr,
            _profile: profile,
            transferred,
            active_connections,
            lifetime,
            listener_thread: Some(listener_thread),
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

    /// Connections currently holding a bounded proxy worker slot.
    pub fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Acquire)
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
    active_connections: Arc<AtomicUsize>,
    lifetime: Arc<ProxyLifetime>,
) {
    while !lifetime.closing.load(Ordering::Acquire) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let mut ready = libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut ready, 1, 50) };
            if result < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                break;
            }
            if result <= 0 {
                continue;
            }
        }
        if lifetime.closing.load(Ordering::Acquire) {
            break;
        }
        let mut stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                #[cfg(not(unix))]
                thread::sleep(Duration::from_millis(5));
                continue;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        let Some(permit) = ConnectionPermit::acquire(&active_connections) else {
            // Reject before allocating an OS thread. A hostile page can open
            // arbitrary subresource/WebSocket connections, so queueing here
            // merely turns memory pressure into unbounded latency.
            let _ = stream.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
            );
            continue;
        };
        let connection = match lifetime.register(&stream) {
            Ok(Some(connection)) => connection,
            _ => continue,
        };
        let profile = profile.clone();
        let resolve = Arc::clone(&resolve);
        let transferred = Arc::clone(&transferred);
        let lifetime = Arc::clone(&lifetime);
        let spawned = thread::Builder::new()
            .name("greppy-policy-proxy-conn".into())
            .spawn(move || {
                let _permit = permit;
                let _connection = connection;
                let _ = handle_client(stream, profile, resolve, transferred, lifetime);
            });
        if let Err(error) = spawned {
            #[cfg(unix)]
            crate::worker_resource::report_thread_spawn_failure("policy proxy connection", &error);
            #[cfg(not(unix))]
            eprintln!("web-runtime: failed to spawn policy proxy connection: {error}");
        }
    }
}

fn handle_client(
    mut client: TcpStream,
    profile: SharedProfile,
    resolve: ConnectResolve,
    transferred: Arc<AtomicU64>,
    lifetime: Arc<ProxyLifetime>,
) -> io::Result<()> {
    let _ = client.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = client.set_write_timeout(Some(Duration::from_secs(30)));
    loop {
        if lifetime.closing.load(Ordering::Acquire) {
            return Ok(());
        }
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
            return handle_connect(client, profile, resolve, &target, &transferred, &lifetime);
        }
        let keep = handle_forward(
            &mut client,
            &profile,
            &resolve,
            &head,
            rest,
            &target,
            &transferred,
            &lifetime,
        )?;
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
    lifetime: &Arc<ProxyLifetime>,
) -> io::Result<()> {
    if crate::supervisor::phase_trace_enabled() {
        if crate::supervisor::phase_trace_enabled() {
            eprintln!("web-runtime: policy-proxy CONNECT target={target}");
        }
    }
    let (host, port) = split_host_port(target, 80);
    match connect_allowed(profile.get(), &host, port, &resolve) {
        Ok(mut server) => {
            let Some(_origin_connection) = lifetime.register(&server)? else { return Ok(()); };
            if lifetime.closing.load(Ordering::Acquire) {
                return Ok(());
            }
            client.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
            #[cfg(unix)]
            {
                let _ = client.set_nodelay(true);
                let _ = server.set_nodelay(true);
                crate::proxy_tunnel::copy_bidirectional_cancellable(
                    &mut client,
                    &mut server,
                    transferred,
                    Some(&lifetime.closing),
                )
            }
            #[cfg(not(unix))]
            {
                splice(client, server, Arc::clone(transferred))
            }
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
    lifetime: &Arc<ProxyLifetime>,
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
    let Some(_origin_connection) = lifetime.register(&server)? else { return Ok(false); };
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
    for line in lines.take_while(|line| !line.is_empty()) {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("proxy-connection:") {
            continue;
        }
        out.push_str(line);
        out.push_str("\r\n");
    }
    // One empty line ends the headers. Copying split()'s trailing empty
    // entries inserted another CRLF before every forwarded request body.
    out.push_str("\r\n");
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
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 4096];
    while buf.len() < HEADER_LIMIT {
        // Peek in batches, then consume only header bytes. Reading beyond the
        // terminator here would swallow a pipelined request or the first TLS
        // bytes after CONNECT; those belong to the next reader.
        let take = chunk.len().min(HEADER_LIMIT - buf.len());
        let n = stream.peek(&mut chunk[..take])?;
        if n == 0 {
            break;
        }
        let previous = buf.len();
        buf.extend_from_slice(&chunk[..n]);
        // Only a delimiter crossing the old/new boundary or in the new chunk
        // can be new. Rescanning the entire prefix per byte was quadratic.
        let scan_from = previous.saturating_sub(3);
        if let Some(idx) = buf[scan_from..]
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
        {
            let end = scan_from + idx + 4;
            stream.read_exact(&mut chunk[..end - previous])?;
            buf.truncate(end);
            return Ok((String::from_utf8_lossy(&buf).into_owned(), Vec::new()));
        }
        stream.read_exact(&mut chunk[..n])?;
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "proxy request headers truncated",
    ))
}

#[cfg(all(unix, test))]
fn splice(mut a: TcpStream, mut b: TcpStream, transferred: Arc<AtomicU64>) -> io::Result<()> {
    let _ = a.set_nodelay(true);
    let _ = b.set_nodelay(true);
    crate::proxy_tunnel::copy_bidirectional(&mut a, &mut b, &transferred)
}

#[cfg(not(unix))]
fn splice(mut a: TcpStream, mut b: TcpStream, transferred: Arc<AtomicU64>) -> io::Result<()> {
    let _ = a.set_nodelay(true);
    let _ = b.set_nodelay(true);
    let mut a_read = a.try_clone()?;
    let mut b_write = b.try_clone()?;
    let up_counter = Arc::clone(&transferred);
    let up = thread::spawn(move || {
        let result = copy_counted(&mut a_read, &mut b_write, &up_counter);
        if result.is_err() {
            // Wake the other copy before joining it: a reset must not leave
            // the peer waiting for bytes on a connection that already failed.
            let _ = a_read.shutdown(std::net::Shutdown::Both);
            let _ = b_write.shutdown(std::net::Shutdown::Both);
        } else {
            // Forward EOF without discarding a response still coming back.
            let _ = b_write.shutdown(std::net::Shutdown::Write);
        }
        result
    });
    let down = copy_counted(&mut b, &mut a, &transferred);
    if down.is_err() {
        let _ = a.shutdown(std::net::Shutdown::Both);
        let _ = b.shutdown(std::net::Shutdown::Both);
    } else {
        let _ = a.shutdown(std::net::Shutdown::Write);
    }
    let up = up
        .join()
        .map_err(|_| io::Error::other("proxy upload worker panicked"))?;
    down.and(up).map(|_| ())
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
    fn header_batches_preserve_body_and_connect_bytes() {
        for method in [
            "POST http://example.test/ HTTP/1.1",
            "CONNECT example.test:443 HTTP/1.1",
        ] {
            for end in [4093, 4094, 4095, 4096, 4097, HEADER_LIMIT] {
                let prefix = format!("{method}\r\nX-Pad: ");
                let header = format!("{prefix}{}\r\n\r\n", "x".repeat(end - prefix.len() - 4));
                let trailing = b"\x16\x03\x01body-or-next-request\r\n\r\n";
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
                let (mut server, _) = listener.accept().unwrap();
                server
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let sent = header.clone();
                let writer = thread::spawn(move || {
                    client.write_all(sent.as_bytes()).unwrap();
                    client.write_all(trailing).unwrap();
                });
                let (parsed, rest) = read_headers(&mut server).unwrap();
                assert_eq!(parsed, header);
                assert!(rest.is_empty());
                let mut remaining = Vec::new();
                server.read_to_end(&mut remaining).unwrap();
                assert_eq!(remaining, trailing, "{method}, end={end}");
                writer.join().unwrap();
            }
        }
    }

    #[test]
    fn header_batches_reject_oversize_and_truncated_headers() {
        for input in [b"GET / HTTP/1.1\r\n".to_vec(), vec![b'x'; HEADER_LIMIT + 1]] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let (mut server, _) = listener.accept().unwrap();
            server
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let writer = thread::spawn(move || {
                client.write_all(&input).unwrap();
            });
            let error = read_headers(&mut server).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            writer.join().unwrap();
        }
    }

    #[test]
    fn proxy_preserves_post_body_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body = b"abc123xyz";
        let origin = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let (header, _) = read_headers(&mut stream).unwrap();
            assert!(header.starts_with("POST /submit HTTP/1.1\r\n"));
            assert!(!header.to_ascii_lowercase().contains("proxy-connection:"));
            let mut received = vec![0; body.len()];
            stream.read_exact(&mut received).unwrap();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        received.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
            stream.write_all(&received).unwrap();
            received
        });
        let proxy = PolicyProxy::spawn(SharedProfile::new(NetworkProfile::Project)).unwrap();
        let mut client = TcpStream::connect(proxy.addr()).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        client
            .write_all(format!(
                "POST http://{address}/submit HTTP/1.1\r\nHost: {address}\r\nContent-Length: {}\r\nProxy-Connection: close\r\nConnection: close\r\n\r\n", body.len()
            ).as_bytes())
            .unwrap();
        client.write_all(body).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        assert_eq!(origin.join().unwrap(), body);
        assert_eq!(response.split_once("\r\n\r\n").unwrap().1.as_bytes(), body);
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
            body.to_ascii_lowercase().contains("location:") && body.contains("169.254.169.254"),
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
                    Ok(vec![std::net::SocketAddr::new(
                        fixture.ip(),
                        fixture.port(),
                    )])
                } else {
                    Ok(vec!["169.254.169.254:80".parse().unwrap()])
                }
            }) as ConnectResolve
        };
        let proxy =
            PolicyProxy::spawn_with_resolve(SharedProfile::new(NetworkProfile::Project), resolve)
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
        let proxy =
            PolicyProxy::spawn_with_resolve(SharedProfile::new(NetworkProfile::Project), resolve)
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
        let proxy =
            PolicyProxy::spawn_with_resolve(SharedProfile::new(NetworkProfile::Project), resolve)
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
        let proxy =
            PolicyProxy::spawn_with_resolve(SharedProfile::new(NetworkProfile::Project), resolve)
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
                    Ok(vec![std::net::SocketAddr::new(
                        fixture.ip(),
                        fixture.port(),
                    )])
                } else {
                    Ok(vec!["169.254.169.254:80".parse().unwrap()])
                }
            }) as ConnectResolve
        };
        let proxy =
            PolicyProxy::spawn_with_resolve(SharedProfile::new(NetworkProfile::Project), resolve)
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

    #[test]
    fn proxy_caps_idle_connections_before_allocating_more_threads() {
        let proxy = PolicyProxy::spawn(SharedProfile::new(NetworkProfile::Project)).unwrap();
        let clients = (0..PER_PROXY_CONNECTION_LIMIT)
            .map(|_| TcpStream::connect(proxy.addr()).unwrap())
            .collect::<Vec<_>>();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while proxy.active_connections() != PER_PROXY_CONNECTION_LIMIT
            && std::time::Instant::now() < deadline
        {
            thread::yield_now();
        }
        assert_eq!(proxy.active_connections(), PER_PROXY_CONNECTION_LIMIT);

        let mut rejected = TcpStream::connect(proxy.addr()).unwrap();
        rejected
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut response = String::new();
        rejected.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503"), "{response:?}");
        assert_eq!(proxy.active_connections(), PER_PROXY_CONNECTION_LIMIT);

        drop(clients);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while proxy.active_connections() != 0 && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(proxy.active_connections(), 0);
    }

    fn connected_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        for stream in [&client, &server] {
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
        }
        (client, server)
    }

    #[test]
    fn splice_reset_wakes_the_other_direction_before_join() {
        use std::os::fd::AsRawFd;

        let (mut client, proxy_client) = connected_pair();
        let (proxy_server, mut origin) = connected_pair();
        proxy_client
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let (done, completed) = std::sync::mpsc::channel();
        let worker = thread::spawn(move || {
            let result = splice(proxy_client, proxy_server, Arc::new(AtomicU64::new(0)));
            let _ = done.send(result);
        });

        client.write_all(b"request").unwrap();
        let mut request = [0_u8; 7];
        origin.read_exact(&mut request).unwrap();
        assert_eq!(&request, b"request");
        let linger = libc::linger {
            l_onoff: 1,
            l_linger: 0,
        };
        let result = unsafe {
            libc::setsockopt(
                origin.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_LINGER,
                std::ptr::addr_of!(linger).cast(),
                std::mem::size_of_val(&linger) as libc::socklen_t,
            )
        };
        assert_eq!(result, 0, "set abortive origin close");
        drop(origin);

        let result = completed.recv_timeout(Duration::from_secs(2));
        // Unblock and join the worker even if this assertion would fail on
        // a regression, so the test never leaves a blocked proxy behind.
        let _ = client.shutdown(std::net::Shutdown::Both);
        worker.join().expect("splice worker");
        let result = result.expect("splice must finish promptly after reset");
        assert!(result.is_err(), "reset must remain an error: {result:?}");
    }

    #[test]
    fn splice_client_half_close_preserves_complete_server_reply() {
        let (mut client, proxy_client) = connected_pair();
        let (proxy_server, mut origin) = connected_pair();
        let worker =
            thread::spawn(move || splice(proxy_client, proxy_server, Arc::new(AtomicU64::new(0))));

        client.write_all(b"request").unwrap();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        let mut request = Vec::new();
        origin.read_to_end(&mut request).unwrap();
        assert_eq!(request, b"request");
        origin.write_all(b"complete reply").unwrap();
        origin.shutdown(std::net::Shutdown::Write).unwrap();
        let mut reply = Vec::new();
        client.read_to_end(&mut reply).unwrap();
        assert_eq!(reply, b"complete reply");
        worker.join().expect("splice worker").expect("clean splice");
    }
}
