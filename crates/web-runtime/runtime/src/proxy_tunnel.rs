//! Bounded full-duplex TCP relay without a second OS thread per tunnel.
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const BUFFER_BYTES: usize = 16 * 1024;

struct Direction {
    bytes: [u8; BUFFER_BYTES],
    start: usize,
    end: usize,
    eof: bool,
    sent_eof: bool,
    read_timeout: Option<Duration>,
    write_timeout: Option<Duration>,
    read_wait: Option<Instant>,
    write_wait: Option<Instant>,
}

impl Direction {
    fn new(source: &TcpStream, destination: &TcpStream) -> io::Result<Self> {
        Ok(Self {
            bytes: [0; BUFFER_BYTES], start: 0, end: 0, eof: false, sent_eof: false,
            read_timeout: source.read_timeout()?, write_timeout: destination.write_timeout()?,
            read_wait: None, write_wait: None,
        })
    }

    fn pending(&self) -> bool { self.start < self.end }
    fn complete(&self) -> bool { self.eof && !self.pending() }
    fn can_read(&self) -> bool { !self.eof && !self.pending() }

    fn transfer(&mut self, source: &mut TcpStream, destination: &mut TcpStream,
        counter: &AtomicU64) -> io::Result<bool>
    {
        let mut progress = false;
        if self.pending() {
            match destination.write(&self.bytes[self.start..self.end]) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(written) => {
                    self.start += written;
                    counter.fetch_add(written as u64, Ordering::Relaxed);
                    self.write_wait = None;
                    progress = true;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    arm_timeout(&mut self.write_wait, self.write_timeout)?;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(true),
                Err(error) => return Err(error),
            }
        } else {
            self.write_wait = None;
        }
        if self.can_read() {
            match source.read(&mut self.bytes) {
                Ok(0) => {
                    self.eof = true;
                    self.read_wait = None;
                    progress = true;
                }
                Ok(read) => {
                    self.start = 0; self.end = read;
                    self.read_wait = None;
                    progress = true;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    arm_timeout(&mut self.read_wait, self.read_timeout)?;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(true),
                Err(error) => return Err(error),
            }
        } else {
            self.read_wait = None;
        }
        if self.complete() && !self.sent_eof {
            // Forward only this direction's EOF. The other endpoint may still
            // owe a complete response after the request's half-close.
            let _ = destination.shutdown(Shutdown::Write);
            self.sent_eof = true;
            progress = true;
        }
        Ok(progress)
    }
}

fn arm_timeout(wait: &mut Option<Instant>, timeout: Option<Duration>) -> io::Result<()> {
    if let Some(timeout) = timeout {
        let deadline = wait.get_or_insert_with(|| Instant::now() + timeout);
        if Instant::now() >= *deadline { return Err(io::ErrorKind::TimedOut.into()); }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn copy_bidirectional(a: &mut TcpStream, b: &mut TcpStream,
    transferred: &AtomicU64) -> io::Result<()>
{
    copy_bidirectional_cancellable(a, b, transferred, None)
}

pub(crate) fn copy_bidirectional_cancellable(a: &mut TcpStream, b: &mut TcpStream,
    transferred: &AtomicU64, closing: Option<&AtomicBool>) -> io::Result<()>
{
    let mut up = Direction::new(a, b)?;
    let mut down = Direction::new(b, a)?;
    a.set_nonblocking(true)?;
    b.set_nonblocking(true)?;
    let result = (|| {
        loop {
            if closing.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let progress_up = up.transfer(a, b, transferred)?;
            let progress_down = down.transfer(b, a, transferred)?;
            if up.complete() && down.complete() { return Ok(()); }
            if progress_up || progress_down { continue; }
            let events = |read: bool, write: bool| {
                (if read { libc::POLLIN } else { 0 }) |
                (if write { libc::POLLOUT } else { 0 })
            };
            let a_events = events(up.can_read(), down.pending());
            let b_events = events(down.can_read(), up.pending());
            let mut descriptors = [
                libc::pollfd { fd: if a_events == 0 { -1 } else { a.as_raw_fd() },
                    events: a_events, revents: 0 },
                libc::pollfd { fd: if b_events == 0 { -1 } else { b.as_raw_fd() },
                    events: b_events, revents: 0 },
            ];
            let remaining = [up.read_wait, up.write_wait, down.read_wait, down.write_wait]
                .into_iter().flatten().map(|deadline|
                    deadline.saturating_duration_since(Instant::now())).min();
            let timeout = remaining.map_or(-1, |value|
                value.as_millis().saturating_add(1).min(i32::MAX as u128) as i32);
            let ready = unsafe { libc::poll(descriptors.as_mut_ptr(), 2, timeout) };
            if ready < 0 {
                let error = io::Error::last_os_error();
                if error.kind() != io::ErrorKind::Interrupted { return Err(error); }
            }
            if descriptors.iter().any(|p| p.revents & libc::POLLNVAL != 0) {
                return Err(io::ErrorKind::InvalidInput.into());
            }
        }
    })();
    if result.is_err() {
        // A reset/timeout in either direction immediately wakes its peer.
        let _ = a.shutdown(Shutdown::Both);
        let _ = b.shutdown(Shutdown::Both);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        for socket in [&client, &server] {
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            socket.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
        }
        (client, server)
    }

    #[test]
    fn full_duplex_backpressure_preserves_every_byte_and_accounting() {
        let (mut client, mut proxy_client) = pair();
        let (mut proxy_server, mut origin) = pair();
        let total = ArcCounter::default();
        let actual = total.0.clone();
        let relay = thread::spawn(move ||
            copy_bidirectional(&mut proxy_client, &mut proxy_server, &actual));
        let request: Vec<u8> = (0..2 * 1024 * 1024).map(|n| (n % 251) as u8).collect();
        let reply: Vec<u8> = request.iter().map(|v| 255 - v).collect();
        let upload = request.clone();
        let expected_reply = reply.clone();
        let mut client_write = client.try_clone().unwrap();
        let writer = thread::spawn(move || {
            client_write.write_all(&upload).unwrap();
            client_write.shutdown(Shutdown::Write).unwrap();
        });
        let mut origin_write = origin.try_clone().unwrap();
        let responder = thread::spawn(move || {
            origin_write.write_all(&reply).unwrap();
            origin_write.shutdown(Shutdown::Write).unwrap();
        });
        let receiver = thread::spawn(move || {
            let mut read = Vec::new();
            origin.read_to_end(&mut read).unwrap();
            read
        });
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        writer.join().unwrap(); responder.join().unwrap();
        let received = receiver.join().unwrap();
        relay.join().unwrap().unwrap();
        assert_eq!(received, request);
        assert_eq!(response, expected_reply);
        assert_eq!(total.0.load(Ordering::Relaxed), (request.len() + response.len()) as u64);
    }

    #[derive(Default)]
    struct ArcCounter(std::sync::Arc<AtomicU64>);
}
