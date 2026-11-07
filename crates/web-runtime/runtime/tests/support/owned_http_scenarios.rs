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

pub(crate) fn serve_form_post_fixture(
    redirect_post: bool,
) -> (HttpFixture, String, std::sync::mpsc::Receiver<Vec<u8>>) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let fixture = HttpFixture::start_with_request(
        move |request| {
            let posted = request.method == "POST" && matches!(request.path, "/login" | "/complete");
            let body = if posted {
                sender.send(request.body.to_vec()).unwrap();
                "<!doctype html><title>submitted</title><body>submitted</body>"
            } else {
                "<!doctype html><title>login</title><form method='post' action='/login'><input id='username' name='username'><input id='password' name='password'><button id='submit' type='submit'>Login</button></form>"
            };
            if redirect_post && request.method == "POST" && request.path == "/login" {
                b"HTTP/1.1 307 Temporary Redirect\r\nLocation: /complete\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
            } else {
                http_response(body.as_bytes(), "text/html; charset=utf-8", "")
            }
        },
        false,
    );
    let url = fixture.origin.clone();
    (fixture, url, receiver)
}

pub(crate) fn serve_cookie_isolation_fixture() -> (HttpFixture, String) {
    let fixture = HttpFixture::start_with_request(
        |request| {
            let cookie = request.header("cookie").unwrap_or("");
            let body = format!("<!doctype html><title>cookie</title><body>{cookie}</body>");
            let set_cookie = if request.path == "/set" {
                "Set-Cookie: isolation=A; Path=/; Max-Age=3600; HttpOnly\r\n"
            } else {
                ""
            };
            http_response(body.as_bytes(), "text/html; charset=utf-8", set_cookie)
        },
        false,
    );
    let url = fixture.origin.trim_end_matches('/').to_owned();
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
        request_headers(url, path, "")
    }

    fn request_headers(url: &str, path: &str, headers: &str) -> String {
        let mut stream = connect(url);
        write!(
            stream,
            "GET {path} HTTP/1.1\r\nHost: fixture\r\n{headers}\r\n"
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    fn assert_no_response_yet(stream: &mut TcpStream) {
        stream
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();
        let mut byte = [0_u8; 1];
        let error = stream
            .read(&mut byte)
            .expect_err("incomplete POST fabricated a response");
        assert!(
            matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ),
            "{error}"
        );
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
    }

    #[test]
    fn owned_http_scenarios_preserve_fragmented_post_bodies_and_307_replay() {
        let (fixture, url, posted) = serve_form_post_fixture(true);
        let idle = connect(&url);
        let body = b"username=alice&password=fixture";
        let mut login = connect(&url);
        write!(
            login,
            "POST /login HTTP/1.1\r\nHost: fixture\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .unwrap();
        login.write_all(&body[..5]).unwrap();
        assert_no_response_yet(&mut login);
        assert!(matches!(
            posted.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        login.write_all(&body[5..]).unwrap();
        let mut response = String::new();
        login.read_to_string(&mut response).unwrap();
        assert_eq!(response, "HTTP/1.1 307 Temporary Redirect\r\nLocation: /complete\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        assert_eq!(
            posted
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .as_slice(),
            body.as_slice(),
            "POST fixture discarded or changed the real body"
        );

        let mut complete = connect(&url);
        complete.write_all(b"POST /complete HTTP/1.1\r\nHost: fixture\r\nTransfer-Encoding: chunked\r\n\r\n5;proof=fragment\r\n").unwrap();
        complete.write_all(&body[..5]).unwrap();
        write!(complete, "\r\n{:x}\r\n", body.len() - 5).unwrap();
        complete.write_all(&body[5..]).unwrap();
        complete.write_all(b"\r\n0\r\n").unwrap();
        assert_no_response_yet(&mut complete);
        assert!(matches!(
            posted.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        complete.write_all(b"\r\n").unwrap();
        let mut response = String::new();
        complete.read_to_string(&mut response).unwrap();
        assert!(response
            .ends_with("\r\n\r\n<!doctype html><title>submitted</title><body>submitted</body>"));
        assert_eq!(
            posted
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .as_slice(),
            body.as_slice(),
            "chunked POST fixture discarded or changed the real body"
        );
        assert!(matches!(
            posted.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert_released(fixture, &url, idle);
    }

    #[test]
    fn owned_http_scenarios_preserve_cookie_headers_echo_and_release_state() {
        let (fixture, url) = serve_cookie_isolation_fixture();
        let idle = connect(&url);
        let set = request(&url, "/set");
        assert!(set.contains("\r\nSet-Cookie: isolation=A; Path=/; Max-Age=3600; HttpOnly\r\n"));
        let echo = request_headers(&url, "/echo", "cookie: isolation=A\r\n");
        assert!(
            echo.ends_with("<!doctype html><title>cookie</title><body>isolation=A</body>"),
            "cookie fixture discarded or changed the real header"
        );
        let clean = request(&url, "/echo");
        assert!(clean.ends_with("<!doctype html><title>cookie</title><body></body>"));
        assert_released(fixture, &url, idle);
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
