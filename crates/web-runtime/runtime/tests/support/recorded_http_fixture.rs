//! Bounded HTTP fixture: an idle speculative connection must not block a request.
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const MAX_CONNECTIONS: usize = 32;
const MAX_HEADER: usize = 8192;
const MAX_RESPONSE: usize = 128 * 1024;

pub(crate) struct HttpFixture {
    pub(crate) origin: String,
    pub(crate) requests: Arc<Mutex<Vec<String>>>,
    accepted: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

struct Connection {
    stream: TcpStream,
    header: Vec<u8>,
    response: Option<(Vec<u8>, usize)>,
}

impl Connection {
    // A single nonblocking read/write per poll keeps every connection eligible.
    fn poll(&mut self, requests: &Mutex<Vec<String>>, respond: &impl Fn(&str) -> Vec<u8>) -> bool {
        if self.response.is_none() {
            let mut chunk = [0; 1024];
            let remaining = (MAX_HEADER - self.header.len()).min(chunk.len());
            if remaining == 0 {
                return false;
            }
            match self.stream.read(&mut chunk[..remaining]) {
                Ok(0) => return false,
                Ok(n) => self.header.extend_from_slice(&chunk[..n]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return true,
                Err(_) => return false,
            }
            if !self.header.windows(4).any(|part| part == b"\r\n\r\n") {
                return true;
            }
            let header = String::from_utf8_lossy(&self.header);
            let mut line = header.lines().next().unwrap_or("").split_whitespace();
            let (Some("GET"), Some(path), Some(version), None) =
                (line.next(), line.next(), line.next(), line.next())
            else {
                return false;
            };
            if !path.starts_with('/') || !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
                return false;
            }
            let mut observed = requests.lock().unwrap();
            assert!(observed.len() < 256, "HTTP fixture request bound exceeded");
            observed.push(path.to_owned());
            drop(observed);
            let response = respond(path);
            assert!(
                response.len() <= MAX_RESPONSE,
                "HTTP fixture response bound exceeded"
            );
            self.response = Some((response, 0));
        }
        let (response, offset) = self.response.as_mut().unwrap();
        match self.stream.write(&response[*offset..]) {
            Ok(0) => false,
            Ok(n) => {
                *offset += n;
                *offset < response.len()
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => true,
            Err(_) => false,
        }
    }
}

impl HttpFixture {
    pub(crate) fn start(respond: impl Fn(&str) -> Vec<u8> + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}/", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&requests);
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_by_worker = Arc::clone(&accepted);
        let worker = thread::spawn(move || {
            let mut clients = Vec::<Connection>::new();
            while !stopped.load(Ordering::Relaxed) {
                while clients.len() < MAX_CONNECTIONS {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            stream.set_nonblocking(true).unwrap();
                            clients.push(Connection {
                                stream,
                                header: Vec::new(),
                                response: None,
                            });
                            accepted_by_worker.fetch_add(1, Ordering::Release);
                        }
                        Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                        Err(error) => panic!("HTTP fixture accept: {error}"),
                    }
                }
                let mut index = 0;
                while index < clients.len() {
                    if clients[index].poll(&observed, &respond) {
                        index += 1;
                    } else {
                        clients.swap_remove(index);
                    }
                }
                thread::sleep(Duration::from_millis(2));
            }
        });
        Self {
            origin,
            requests,
            accepted,
            stop,
            worker: Some(worker),
        }
    }
}

impl Drop for HttpFixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}

pub(crate) fn http_response(body: &[u8], content_type: &str, disposition: &str) -> Vec<u8> {
    assert!(
        body.len() <= MAX_RESPONSE - 1024,
        "HTTP fixture body bound exceeded"
    );
    let mut response = format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n{disposition}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes();
    response.extend_from_slice(body);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn silent_connection_does_not_block_a_complete_request_or_fabricate_a_root_request() {
        let fixture = HttpFixture::start(|path| http_response(path.as_bytes(), "text/plain", ""));
        let address = fixture
            .origin
            .strip_prefix("http://")
            .unwrap()
            .trim_end_matches('/');
        let silent = TcpStream::connect(address).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while fixture.accepted.load(Ordering::Acquire) < 1 {
            assert!(
                Instant::now() < deadline,
                "silent connection was not accepted"
            );
            thread::sleep(Duration::from_millis(2));
        }
        let request = |path: &str| {
            let mut client = TcpStream::connect(address).unwrap();
            client
                .set_read_timeout(Some(Duration::from_millis(500)))
                .unwrap();
            client
                .set_write_timeout(Some(Duration::from_millis(500)))
                .unwrap();
            write!(
                client,
                "GET {path} HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            let mut response = String::new();
            client
                .read_to_string(&mut response)
                .expect("independent HTTP request was blocked by silent connection");
            assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response:?}");
            assert!(
                response.ends_with(&format!("\r\n\r\n{path}")),
                "{response:?}"
            );
        };
        // The first socket is still open and has sent no bytes.
        request("/ready");
        assert_eq!(*fixture.requests.lock().unwrap(), ["/ready"]);
        drop(silent);
        // Closing an empty connection must not synthesize GET /.
        request("/after-eof");
        assert_eq!(*fixture.requests.lock().unwrap(), ["/ready", "/after-eof"]);
    }
}
