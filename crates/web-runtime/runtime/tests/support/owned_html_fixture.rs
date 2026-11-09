//! Scope-owned HTML fixtures. Never retain a listener, thread, or owned body past Drop.
use std::borrow::Cow;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const MAX_CONNECTIONS: usize = 32;
const MAX_HEADER: usize = 16 * 1024;
const WRITE_CHUNK: usize = 64 * 1024;

pub struct HtmlFixture {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    #[cfg(test)]
    body: Weak<Cow<'static, str>>,
}

impl HtmlFixture {
    pub fn start(html: impl Into<Cow<'static, str>>) -> (Self, String) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind HTML fixture");
        listener.set_nonblocking(true).expect("nonblocking fixture");
        let address = listener.local_addr().expect("fixture address");
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let html = Arc::new(html.into());
        #[cfg(test)]
        let body = Arc::downgrade(&html);
        let worker = thread::spawn(move || serve(listener, worker_stop, html));
        (
            Self {
                stop,
                worker: Some(worker),
                #[cfg(test)]
                body,
            },
            format!("http://{address}/"),
        )
    }
}

impl Drop for HtmlFixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            // All socket operations are nonblocking; even a silent/slow client
            // cannot keep the fixture alive after its owning test has finished.
            let _ = worker.join();
        }
    }
}

struct Connection {
    stream: TcpStream,
    request: Vec<u8>,
    response: Option<Vec<u8>>,
    header_sent: usize,
    body_sent: usize,
    head_only: bool,
}

impl Connection {
    fn new(stream: TcpStream) -> io::Result<Self> {
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream,
            request: Vec::new(),
            response: None,
            header_sent: 0,
            body_sent: 0,
            head_only: false,
        })
    }

    // false closes this connection; true keeps it for the next bounded poll.
    fn poll(&mut self, body: &[u8]) -> bool {
        if self.response.is_none() {
            let mut buffer = [0_u8; 2048];
            match self.stream.read(&mut buffer) {
                Ok(0) => return false,
                Ok(size) => {
                    if self.request.len() + size > MAX_HEADER {
                        return false;
                    }
                    self.request.extend_from_slice(&buffer[..size]);
                    if !self.request.windows(4).any(|part| part == b"\r\n\r\n") {
                        return true;
                    }
                    self.head_only = self.request.starts_with(b"HEAD ");
                    self.response = Some(format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    ).into_bytes());
                    self.request.clear();
                }
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock
                        || error.kind() == io::ErrorKind::Interrupted =>
                {
                    return true
                }
                Err(_) => return false,
            }
        }
        let header = self.response.as_ref().expect("complete request");
        if self.header_sent < header.len() {
            return advance(&mut self.stream, header, &mut self.header_sent);
        }
        if self.head_only || self.body_sent == body.len() {
            return false;
        }
        // Stream the shared body directly. A large fixture must not allocate
        // another copy for every favicon, image, or page request.
        advance(&mut self.stream, body, &mut self.body_sent)
    }
}

fn advance(stream: &mut TcpStream, bytes: &[u8], offset: &mut usize) -> bool {
    let end = (*offset + WRITE_CHUNK).min(bytes.len());
    match stream.write(&bytes[*offset..end]) {
        Ok(0) => false,
        Ok(size) => {
            *offset += size;
            true
        }
        Err(error) => {
            error.kind() == io::ErrorKind::WouldBlock || error.kind() == io::ErrorKind::Interrupted
        }
    }
}

fn serve(listener: TcpListener, stop: Arc<AtomicBool>, html: Arc<Cow<'static, str>>) {
    let mut connections = Vec::new();
    while !stop.load(Ordering::Acquire) {
        for _ in connections.len()..MAX_CONNECTIONS {
            match listener.accept() {
                Ok((stream, _)) => {
                    if let Ok(connection) = Connection::new(stream) {
                        connections.push(connection);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        let mut index = 0;
        while index < connections.len() {
            if connections[index].poll(html.as_bytes()) {
                index += 1;
            } else {
                connections.swap_remove(index);
            }
        }
        thread::sleep(Duration::from_millis(2));
    }
    drop(connections);
    drop(listener);
    drop(html);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Shutdown;

    fn connect(url: &str) -> TcpStream {
        let stream = TcpStream::connect(url.trim_start_matches("http://").trim_end_matches('/'))
            .expect("connect fixture");
        stream
            .set_read_timeout(Some(Duration::from_secs(4)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(4)))
            .unwrap();
        stream
    }

    fn request(url: &str, method: &str) -> Vec<u8> {
        let mut stream = connect(url);
        write!(stream, "{method} / HTTP/1.1\r\nHost: fixture\r\n\r\n").unwrap();
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .expect("read fixture response");
        response
    }

    #[test]
    fn owned_html_fixture_preserves_large_body_and_handles_idle_and_eof_clients() {
        let body = "ä-fixture-data".repeat(32 * 1024);
        let (owner, url) = HtmlFixture::start(body.clone());
        let _silent = connect(&url);
        let mut eof = connect(&url);
        eof.shutdown(Shutdown::Write).unwrap();
        let mut unexpected = Vec::new();
        eof.read_to_end(&mut unexpected).unwrap();
        assert!(unexpected.is_empty(), "EOF fabricated an HTML response");

        let response = request(&url, "GET");
        let split = response
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
            .unwrap()
            + 4;
        assert_eq!(
            &response[split..],
            body.as_bytes(),
            "large HTML fixture body changed"
        );
        let header = std::str::from_utf8(&response[..split]).unwrap();
        assert!(header.contains(&format!("Content-Length: {}\r\n", body.len())));
        let head = request(&url, "HEAD");
        assert!(head.ends_with(b"\r\n\r\n"), "HEAD returned an HTML body");
        drop(owner);
    }

    #[test]
    fn owned_html_fixture_drop_releases_listener_connections_and_body() {
        let (owner, url) = HtmlFixture::start(String::from("<p>owned allocation</p>"));
        let body = owner.body.clone();
        let mut silent = connect(&url);
        assert!(!request(&url, "GET").is_empty(), "fixture did not start");
        drop(owner);
        assert!(
            TcpStream::connect(url.trim_start_matches("http://").trim_end_matches('/')).is_err(),
            "fixture listener survived guard drop"
        );
        assert!(body.upgrade().is_none(), "fixture body survived guard drop");
        let mut byte = [0_u8; 1];
        match silent.read(&mut byte) {
            Ok(0) => (),
            Err(error)
                if error.kind() == io::ErrorKind::ConnectionReset
                    || error.kind() == io::ErrorKind::ConnectionAborted =>
            {
                ()
            }
            result => panic!("fixture idle connection survived guard drop: {result:?}"),
        }
    }
}
