//! Scope-owned status, redirect and research pages using the shared bounded server.
use super::recorded_http_fixture::{http_response, HttpFixture};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

pub(crate) fn serve_status_fixture() -> (HttpFixture, String) {
    let origin = Arc::new(OnceLock::<String>::new());
    let response_origin = Arc::clone(&origin);
    let repeated = AtomicUsize::new(0);
    let fixture = HttpFixture::start(move |path| {
        let origin = response_origin.get().expect("status fixture origin");
        if let Some(target) = match path {
            "/jump" => Some("landed"),
            "/jump-empty" => Some("empty"),
            _ => None,
        } {
            return format!("HTTP/1.1 302 Found\r\nLocation: {origin}/{target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").into_bytes();
        }
        if path == "/chunked" {
            return b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n".to_vec();
        }
        let (status, body) = match path {
            "/missing" => ("404 Not Found", b"missing".as_slice()),
            "/repeat" => {
                if repeated.fetch_add(1, Ordering::SeqCst) == 0 {
                    ("200 OK", b"repeated ok".as_slice())
                } else {
                    ("404 Not Found", b"repeated missing".as_slice())
                }
            }
            "/empty" => ("204 No Content", b"".as_slice()),
            "/landed" => ("200 OK", b"landed".as_slice()),
            _ => ("200 OK", b"ok".as_slice()),
        };
        let mut response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        ).into_bytes();
        response.extend_from_slice(body);
        response
    });
    origin
        .set(fixture.origin.trim_end_matches('/').to_owned())
        .unwrap();
    let url = fixture.origin.clone();
    (fixture, url)
}

pub(crate) fn serve_site() -> (HttpFixture, String) {
    let origin = Arc::new(OnceLock::<String>::new());
    let response_origin = Arc::clone(&origin);
    let article = include_str!("../../fixtures/article.html");
    let fixture = HttpFixture::start(move |path| {
        let origin = response_origin.get().expect("research fixture origin");
        let body = if path.starts_with("/search") {
            format!(
                "<!DOCTYPE html><html><body><h1>Results</h1><ul><li><a href=\"{origin}/article\">Greppy Article</a></li></ul></body></html>"
            )
        } else {
            article.to_owned()
        };
        http_response(body.as_bytes(), "text/html; charset=utf-8", "")
    });
    let url = fixture.origin.trim_end_matches('/').to_owned();
    origin.set(url.clone()).unwrap();
    (fixture, url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    fn connect(url: &str) -> TcpStream {
        let stream =
            TcpStream::connect(url.trim_start_matches("http://").trim_end_matches('/')).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
    }

    fn request(url: &str, path: &str) -> String {
        let mut stream = connect(url);
        write!(stream, "GET {path} HTTP/1.1\r\nHost: fixture\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    fn assert_released(fixture: HttpFixture, url: &str, mut idle: TcpStream) {
        let requests = Arc::downgrade(&fixture.requests);
        drop(fixture);
        assert!(
            TcpStream::connect(url.trim_start_matches("http://").trim_end_matches('/')).is_err(),
            "scenario fixture listener survived guard drop"
        );
        assert!(
            requests.upgrade().is_none(),
            "scenario fixture retained its request log"
        );
        let mut byte = [0; 1];
        match idle.read(&mut byte) {
            Ok(0) => (),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted
                ) =>
            {
                ()
            }
            result => panic!("scenario fixture idle connection survived guard drop: {result:?}"),
        }
    }

    #[test]
    fn owned_http_scenarios_keep_status_redirect_chunked_and_repeat_contracts() {
        let (fixture, url) = serve_status_fixture();
        let idle = connect(&url);
        let origin = url.trim_end_matches('/');
        assert_eq!(request(&url, "/jump"), format!("HTTP/1.1 302 Found\r\nLocation: {origin}/landed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"));
        assert_eq!(request(&url, "/jump-empty"), format!("HTTP/1.1 302 Found\r\nLocation: {origin}/empty\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"));
        assert_eq!(request(&url, "/chunked"), "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n");
        assert!(request(&url, "/repeat").ends_with("\r\n\r\nrepeated ok"));
        assert!(request(&url, "/repeat").starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(request(&url, "/missing").ends_with("\r\n\r\nmissing"));
        assert!(request(&url, "/empty").starts_with("HTTP/1.1 204 No Content\r\n"));
        assert!(request(&url, "/empty").ends_with("Content-Length: 0\r\nConnection: close\r\n\r\n"));
        assert_released(fixture, &url, idle);
    }

    #[test]
    fn owned_http_scenarios_keep_search_links_article_bytes_and_release_state() {
        let (fixture, url) = serve_site();
        let idle = connect(&url);
        let search = request(&url, "/search?q=greppy");
        assert!(search.ends_with(&format!("<!DOCTYPE html><html><body><h1>Results</h1><ul><li><a href=\"{url}/article\">Greppy Article</a></li></ul></body></html>")));
        let article = request(&url, "/article");
        assert!(article.ends_with(include_str!("../../fixtures/article.html")));
        assert_released(fixture, &url, idle);
    }
}
