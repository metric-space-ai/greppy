use std::io::{self, Read};
use std::net::TcpStream;
use std::time::Duration;

pub(crate) fn normalize_accepted_stream(stream: &TcpStream) -> io::Result<()> {
    stream.set_nonblocking(false)?; // normalize accepted fixture socket
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    stream.set_write_timeout(Some(Duration::from_secs(1)))
}

pub(crate) fn read_headers(stream: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut request = Vec::with_capacity(2048);
    let mut buffer = [0_u8; 2048];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return if request.is_empty() {
                Ok(None)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete fixture HTTP headers",
                ))
            };
        }
        if request.len() + count > 65536 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "fixture HTTP headers exceed64KiB",
            ));
        }
        request.extend_from_slice(&buffer[..count]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            return Ok(Some(request));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::os::fd::AsRawFd;
    use std::sync::mpsc;
    use std::thread;

    struct ObservedSocket {
        stream: TcpStream,
        first_read: Option<mpsc::Sender<()>>,
    }

    impl Read for ObservedSocket {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let count = self.stream.read(buffer)?;
            if count > 0 {
                if let Some(sender) = self.first_read.take() {
                    sender.send(()).expect("report actual first socket read");
                }
            }
            Ok(count)
        }
    }

    #[test]
    fn accepted_navigation_fixture_stream_reads_complete_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (accepted, _) = listener.accept().unwrap();
        accepted.set_nonblocking(true).unwrap();
        normalize_accepted_stream(&accepted).unwrap();
        // SAFETY: accepted owns this live socket for the complete flags query.
        let flags = unsafe { libc::fcntl(accepted.as_raw_fd(), libc::F_GETFL) };
        assert!(flags >= 0);
        assert_eq!(
            flags & libc::O_NONBLOCK,
            0,
            "accepted lifecycle socket retained nonblocking mode"
        );
        assert_eq!(
            accepted.read_timeout().unwrap(),
            Some(Duration::from_secs(1))
        );
        let first = b"GET /landed HTTP/1.1\r\nHost: local";
        let last = b"host\r\n\r\n";
        client.write_all(first).unwrap();
        let (observed, ready) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut socket = ObservedSocket {
                stream: accepted,
                first_read: Some(observed),
            };
            let result = read_headers(&mut socket);
            // Retain the actual socket even if a faulty reader returns early.
            (result, socket)
        });
        ready
            .recv_timeout(Duration::from_secs(2))
            .expect("observe real partial header read");
        client.write_all(last).unwrap();
        let (headers, _socket_owner) = reader.join().unwrap();
        assert_eq!(
            headers.unwrap().unwrap(),
            [first.as_slice(), last.as_slice()].concat(),
            "lifecycle fixture accepted incomplete HTTP headers"
        );
    }
}
