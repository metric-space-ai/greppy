//! Bounded HTTP fixture: an idle speculative connection must not block a request.
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const MAX_CONNECTIONS: usize = 32;
const MAX_HEADER: usize = 8192;
const MAX_REQUEST_BODY: usize = 128 * 1024;
const MAX_RESPONSE: usize = 128 * 1024;

pub(crate) struct HttpFixture {
    pub(crate) origin: String,
    pub(crate) requests: Arc<Mutex<Vec<String>>>,
    accepted: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

pub(crate) struct HttpRequest<'a> {
    pub(crate) method: &'a str,
    pub(crate) path: &'a str,
    headers: &'a str,
    pub(crate) body: std::borrow::Cow<'a, [u8]>,
}

impl HttpRequest<'_> {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }
}

fn visit_chunked(mut encoded: &[u8], mut visit: impl FnMut(&[u8])) -> Option<usize> {
    let mut total: usize = 0;
    loop {
        let line_end = encoded.windows(2).position(|part| part == b"\r\n")?;
        let size_text = std::str::from_utf8(&encoded[..line_end]).ok()?;
        let size = usize::from_str_radix(size_text.split(';').next()?.trim(), 16).ok()?;
        encoded = &encoded[line_end + 2..];
        if size == 0 {
            return (encoded.starts_with(b"\r\n")
                || encoded.windows(4).any(|part| part == b"\r\n\r\n"))
            .then_some(total);
        }
        let end = size.checked_add(2)?;
        if encoded.len() < end || &encoded[size..end] != b"\r\n" {
            return None;
        }
        total = total.checked_add(size)?;
        if total > MAX_REQUEST_BODY {
            return None;
        }
        visit(&encoded[..size]);
        encoded = &encoded[end..];
    }
}

fn parse_request(bytes: &[u8]) -> Result<Option<HttpRequest<'_>>, ()> {
    let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") else {
        return Ok(None);
    };
    let header_end = header_end + 4;
    if header_end > MAX_HEADER {
        return Err(());
    }
    let headers = std::str::from_utf8(&bytes[..header_end]).map_err(|_| ())?;
    let mut line = headers.lines().next().unwrap_or("").split_whitespace();
    let (Some(method), Some(path), Some(version), None) =
        (line.next(), line.next(), line.next(), line.next())
    else {
        return Err(());
    };
    if !path.starts_with('/') || !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err(());
    }
    let mut request = HttpRequest {
        method,
        path,
        headers,
        body: std::borrow::Cow::Borrowed(&[]),
    };
    let body = &bytes[header_end..];
    request.body = if request
        .header("transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        if request.header("content-length").is_some() {
            return Err(());
        }
        // Incomplete fragments allocate no decoded body; copy only once the frame is complete.
        let Some(size) = visit_chunked(body, |_| {}) else {
            return Ok(None);
        };
        let mut decoded = Vec::with_capacity(size);
        visit_chunked(body, |chunk| decoded.extend_from_slice(chunk)).ok_or(())?;
        std::borrow::Cow::Owned(decoded)
    } else {
        let content_length = request
            .header("content-length")
            .map(|value| value.parse::<usize>().map_err(|_| ()))
            .transpose()?
            .unwrap_or(0);
        if content_length > MAX_REQUEST_BODY {
            return Err(());
        }
        if body.len() < content_length {
            return Ok(None);
        }
        std::borrow::Cow::Borrowed(&body[..content_length])
    };
    if request.body.len() > MAX_REQUEST_BODY {
        return Err(());
    }
    Ok(Some(request))
}

struct Connection {
    stream: TcpStream,
    header: Vec<u8>,
    response: Option<(Vec<u8>, usize)>,
}

impl Connection {
    // A single nonblocking read/write per poll keeps every connection eligible.
    fn poll(
        &mut self,
        requests: &Mutex<Vec<String>>,
        respond: &impl Fn(&HttpRequest<'_>) -> Vec<u8>,
        get_only: bool,
    ) -> bool {
        if self.response.is_none() {
            let mut chunk = [0; 1024];
            let remaining = (MAX_HEADER + MAX_REQUEST_BODY - self.header.len()).min(chunk.len());
            if remaining == 0 {
                return false;
            }
            match self.stream.read(&mut chunk[..remaining]) {
                Ok(0) => return false,
                Ok(n) => self.header.extend_from_slice(&chunk[..n]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return true,
                Err(_) => return false,
            }
            let request = match parse_request(&self.header) {
                Ok(Some(request)) => request,
                Ok(None) => {
                    return self.header.windows(4).any(|part| part == b"\r\n\r\n")
                        || self.header.len() < MAX_HEADER
                }
                Err(()) => return false,
            };
            if get_only && request.method != "GET" {
                return false;
            }
            let mut observed = requests.lock().unwrap();
            assert!(observed.len() < 256, "HTTP fixture request bound exceeded");
            observed.push(request.path.to_owned());
            drop(observed);
            let response = respond(&request);
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
        Self::start_with_request(move |request| respond(request.path), true)
    }

    pub(crate) fn start_with_request(
        respond: impl Fn(&HttpRequest<'_>) -> Vec<u8> + Send + 'static,
        get_only: bool,
    ) -> Self {
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
                    if clients[index].poll(&observed, &respond, get_only) {
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
