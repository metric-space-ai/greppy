use super::*;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::net::TcpListener;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread::JoinHandle;
use ureq::rustls::{
    self,
    pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer},
};

const BODY: &[u8] = b"model fixture over real TLS";
const CERT: &[u8] = include_bytes!("../tests/fixtures/local-model-tls/cert.der");
const KEY: &[u8] = include_bytes!("../tests/fixtures/local-model-tls/key.der");

fn spec() -> ArtifactSpec {
    ArtifactSpec {
        sha256: Sha256::digest(BODY).into(),
        size_bytes: BODY.len() as u64,
    }
}

fn response(status: &str, headers: &str, body: &[u8]) -> Vec<u8> {
    let mut bytes = format!("HTTP/1.1 {status}\r\nConnection: close\r\n{headers}\r\n").into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

fn ok() -> Vec<u8> {
    response(
        "200 OK",
        &format!("Content-Length: {}\r\n", BODY.len()),
        BODY,
    )
}

struct TlsFixture {
    url: String,
    hits: Arc<AtomicUsize>,
    stop: Arc<AtomicUsize>,
    worker: Option<JoinHandle<()>>,
}

impl TlsFixture {
    fn new(replies: Vec<(Vec<u8>, Duration)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![CertificateDer::from(CERT.to_vec())],
                PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(KEY.to_vec())),
            )
            .unwrap();
        let config = Arc::new(config);
        let hits = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicUsize::new(0));
        let worker_hits = hits.clone();
        let worker_stop = stop.clone();
        let worker = std::thread::spawn(move || {
            let started = std::time::Instant::now();
            while worker_stop.load(Ordering::Acquire) == 0
                && started.elapsed() < Duration::from_secs(15)
            {
                let (socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let connection = rustls::ServerConnection::new(config.clone()).unwrap();
                let mut stream = rustls::StreamOwned::new(connection, socket);
                let mut request = Vec::new();
                let mut byte = [0; 1];
                while request.len() < 16384 && !request.ends_with(b"\r\n\r\n") {
                    if stream.read_exact(&mut byte).is_err() {
                        break;
                    }
                    request.push(byte[0]);
                }
                // Untrusted-certificate tests must fail before an HTTP request.
                if !request.ends_with(b"\r\n\r\n") {
                    continue;
                }
                let hit = worker_hits.fetch_add(1, Ordering::SeqCst);
                let (reply, delay) = &replies[hit.min(replies.len() - 1)];
                std::thread::sleep(*delay);
                let _ = stream.write_all(reply);
                let _ = stream.flush();
            }
        });
        Self {
            url: format!("https://localhost:{port}/artifact"),
            hits,
            stop,
            worker: Some(worker),
        }
    }

    fn trusted_downloader(&self, options: DownloadOptions) -> ModelDownloader {
        let mut roots = rustls::RootCertStore::empty();
        roots.add(CertificateDer::from(CERT.to_vec())).unwrap();
        let config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let agent = ModelDownloader::agent_builder(&options)
            .tls_config(Arc::new(config))
            .build();
        ModelDownloader { agent, options }
    }
}

impl Drop for TlsFixture {
    fn drop(&mut self) {
        self.stop.store(1, Ordering::Release);
        self.worker.take().unwrap().join().unwrap();
    }
}

fn options() -> DownloadOptions {
    DownloadOptions {
        attempts: 2,
        retry_delay: Duration::from_millis(5),
        connect_timeout: Duration::from_secs(1),
        read_timeout: Duration::from_millis(200),
    }
}

#[test]
fn https_first_use_streams_verified_bytes_and_cache_hit_never_reconnects() {
    let server = TlsFixture::new(vec![(ok(), Duration::ZERO)]);
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
    let client = server.trusted_downloader(options());
    assert_eq!(server.hits.load(Ordering::SeqCst), 0);
    assert!(!dir.path().join("models").exists());
    let mut events = Vec::new();
    let path = client
        .ensure(
            &cache,
            &spec(),
            &server.url,
            &ProvisionCancel::default(),
            |e| events.push(e),
        )
        .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), BODY);
    assert!(events.contains(&ProvisionEvent::Verifying));
    assert_eq!(
        events.last(),
        Some(&ProvisionEvent::ArtifactReady { cached: false })
    );
    client
        .ensure(
            &cache,
            &spec(),
            &server.url,
            &ProvisionCancel::default(),
            |_| {},
        )
        .unwrap();
    assert_eq!(server.hits.load(Ordering::SeqCst), 1);
}

#[test]
fn transient_http_failure_retries_but_404_and_bad_hash_do_not() {
    for (first, retries) in [
        (
            response("503 Unavailable", "Content-Length: 0\r\n", b""),
            true,
        ),
        (response("404 Missing", "Content-Length: 0\r\n", b""), false),
        (
            response(
                "200 OK",
                &format!("Content-Length: {}\r\n", BODY.len()),
                &vec![b'x'; BODY.len()],
            ),
            false,
        ),
    ] {
        let server = TlsFixture::new(vec![(first, Duration::ZERO), (ok(), Duration::ZERO)]);
        let dir = tempfile::tempdir().unwrap();
        let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
        let mut events = Vec::new();
        let result = server.trusted_downloader(options()).ensure(
            &cache,
            &spec(),
            &server.url,
            &ProvisionCancel::default(),
            |e| events.push(e),
        );
        assert_eq!(result.is_ok(), retries);
        assert_eq!(
            server.hits.load(Ordering::SeqCst),
            if retries { 2 } else { 1 }
        );
        assert_eq!(
            events.contains(&ProvisionEvent::Retrying { next_attempt: 2 }),
            retries
        );
    }
}

#[test]
fn short_body_restarts_in_clean_staging() {
    let first = response(
        "200 OK",
        &format!("Content-Length: {}\r\n", BODY.len()),
        b"partial",
    );
    let server = TlsFixture::new(vec![(first, Duration::ZERO), (ok(), Duration::ZERO)]);
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
    let path = server
        .trusted_downloader(options())
        .ensure(
            &cache,
            &spec(),
            &server.url,
            &ProvisionCancel::default(),
            |_| {},
        )
        .unwrap();
    assert_eq!(std::fs::read(path).unwrap(), BODY);
    assert_eq!(server.hits.load(Ordering::SeqCst), 2);
    assert_eq!(
        std::fs::read_dir(dir.path().join("models/objects"))
            .unwrap()
            .count(),
        1
    );
}

#[test]
fn untrusted_tls_is_rejected_without_http_or_cache_publication() {
    let server = TlsFixture::new(vec![(ok(), Duration::ZERO)]);
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
    let client = ModelDownloader::new(DownloadOptions {
        attempts: 1,
        ..options()
    })
    .unwrap();
    assert!(client
        .ensure(
            &cache,
            &spec(),
            &server.url,
            &ProvisionCancel::default(),
            |_| {}
        )
        .is_err());
    assert_eq!(server.hits.load(Ordering::SeqCst), 0);
    assert!(cache.lookup_verified(&spec()).unwrap().is_none());
}

#[test]
fn http_and_https_downgrade_redirects_are_refused() {
    let redirect = response(
        "302 Found",
        "Location: http://127.0.0.1:1/weights\r\nContent-Length: 0\r\n",
        b"",
    );
    let server = TlsFixture::new(vec![(redirect, Duration::ZERO)]);
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
    let client = server.trusted_downloader(DownloadOptions {
        attempts: 1,
        ..options()
    });
    for url in ["http://127.0.0.1:1/weights", &server.url] {
        assert!(client
            .ensure(&cache, &spec(), url, &ProvisionCancel::default(), |_| {})
            .is_err());
    }
    assert!(cache.lookup_verified(&spec()).unwrap().is_none());
}

#[test]
fn declared_length_mismatch_and_encoded_response_do_not_retry() {
    for reply in [
        response("200 OK", "Content-Length: 999\r\n", BODY),
        response("200 OK", "Content-Encoding: gzip\r\n", BODY),
        response("206 Partial Content", "", BODY),
    ] {
        let server = TlsFixture::new(vec![(reply, Duration::ZERO)]);
        let dir = tempfile::tempdir().unwrap();
        let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
        assert!(server
            .trusted_downloader(options())
            .ensure(
                &cache,
                &spec(),
                &server.url,
                &ProvisionCancel::default(),
                |_| {}
            )
            .is_err());
        assert_eq!(server.hits.load(Ordering::SeqCst), 1);
        assert!(cache.lookup_verified(&spec()).unwrap().is_none());
    }
}

#[test]
fn cancellation_during_retry_delay_stops_without_another_connection() {
    let server = TlsFixture::new(vec![(
        response("503 Busy", "Content-Length: 0\r\n", b""),
        Duration::ZERO,
    )]);
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
    let cancel = ProvisionCancel::default();
    let result = server.trusted_downloader(options()).ensure(
        &cache,
        &spec(),
        &server.url,
        &cancel,
        |event| {
            if matches!(event, ProvisionEvent::Retrying { .. }) {
                cancel.cancel();
            }
        },
    );
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
    assert_eq!(server.hits.load(Ordering::SeqCst), 1);
}

#[test]
fn stalled_peer_is_bounded_and_never_publishes() {
    let server = TlsFixture::new(vec![(ok(), Duration::from_millis(600))]);
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("models")).unwrap();
    let start = std::time::Instant::now();
    assert!(server
        .trusted_downloader(DownloadOptions {
            attempts: 1,
            ..options()
        })
        .ensure(
            &cache,
            &spec(),
            &server.url,
            &ProvisionCancel::default(),
            |_| {}
        )
        .is_err());
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(cache.lookup_verified(&spec()).unwrap().is_none());
}
