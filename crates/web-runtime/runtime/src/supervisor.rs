use crate::protocol::{read_message, write_message, Message, WorkerKind};
use std::ffi::OsString;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const REAP_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Eq, PartialEq)]
pub struct Config {
    pub controller_worker: PathBuf,
    pub content_worker: PathBuf,
}

impl Config {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, String> {
        let mut controller_worker = None;
        let mut content_worker = None;
        let mut args = args.into_iter();

        while let Some(argument) = args.next() {
            match argument.to_str() {
                Some("--controller-worker") => {
                    set_path(&mut controller_worker, "--controller-worker", args.next())?;
                }
                Some("--content-worker") => {
                    set_path(&mut content_worker, "--content-worker", args.next())?;
                }
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }

        Ok(Self {
            controller_worker: controller_worker
                .ok_or_else(|| "missing --controller-worker PATH".to_owned())?,
            content_worker: content_worker
                .ok_or_else(|| "missing --content-worker PATH".to_owned())?,
        })
    }
}

fn set_path(
    destination: &mut Option<PathBuf>,
    option: &str,
    value: Option<OsString>,
) -> Result<(), String> {
    if destination.is_some() {
        return Err(format!("duplicate {option}"));
    }
    let value = value.ok_or_else(|| format!("missing path after {option}"))?;
    if value.is_empty() {
        return Err(format!("empty path after {option}"));
    }
    *destination = Some(value.into());
    Ok(())
}

pub fn run(config: Config) -> io::Result<()> {
    let mut controller = WorkerProcess::spawn(&config.controller_worker, WorkerKind::Controller)?;
    controller.handshake()?;
    println!("web_runtime.controller=ready");

    let mut content = WorkerProcess::spawn(&config.content_worker, WorkerKind::Content)?;
    content.handshake()?;
    println!("web_runtime.content=ready");

    content.shutdown()?;
    println!("web_runtime.content=stopped");
    controller.shutdown()?;
    println!("web_runtime.controller=stopped");
    println!("web_runtime.supervisor=stopped");
    Ok(())
}

struct WorkerProcess {
    worker: WorkerKind,
    child: Child,
    input: Option<BufWriter<ChildStdin>>,
    messages: Receiver<io::Result<Message>>,
    reader_thread: Option<JoinHandle<()>>,
    reaped: bool,
}

impl WorkerProcess {
    fn spawn(path: &Path, worker: WorkerKind) -> io::Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "failed to spawn {worker:?} worker at {}: {error}",
                        path.display()
                    ),
                )
            })?;

        let input = child.stdin.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "worker stdin was not piped")
        })?;
        let output = child.stdout.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "worker stdout was not piped")
        })?;
        let (message_sender, messages) = mpsc::channel();
        let reader_thread = match thread::Builder::new()
            .name(format!("web-runtime-{worker:?}-protocol-reader"))
            .spawn(move || {
                let mut output = BufReader::new(output);
                loop {
                    match read_message(&mut output) {
                        Ok(message) => {
                            if message_sender.send(Ok(message)).is_err() {
                                return;
                            }
                        }
                        Err(error) => {
                            let _ = message_sender.send(Err(error));
                            return;
                        }
                    }
                }
            }) {
            Ok(reader_thread) => reader_thread,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::new(
                    error.kind(),
                    format!("failed to spawn {worker:?} protocol reader: {error}"),
                ));
            }
        };

        Ok(Self {
            worker,
            child,
            input: Some(BufWriter::new(input)),
            messages,
            reader_thread: Some(reader_thread),
            reaped: false,
        })
    }

    fn handshake(&mut self) -> io::Result<()> {
        self.send(&Message::hello(self.worker))?;
        self.expect(Message::ready(self.worker))
    }

    fn shutdown(&mut self) -> io::Result<()> {
        self.send(&Message::shutdown())?;
        self.expect(Message::shutdown_ack(self.worker))?;
        self.input.take();

        let status = self.wait_for_exit()?;
        self.reaped = true;
        self.join_reader();
        if !status.success() {
            return Err(io::Error::other(format!(
                "{:?} worker exited with {status}",
                self.worker
            )));
        }
        Ok(())
    }

    fn send(&mut self, message: &Message) -> io::Result<()> {
        let input = self.input.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "worker stdin is already closed")
        })?;
        write_message(input, message)
    }

    fn expect(&mut self, expected: Message) -> io::Result<()> {
        let actual = match self.messages.recv_timeout(PROCESS_TIMEOUT) {
            Ok(result) => result?,
            Err(RecvTimeoutError::Timeout) => {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "timed out after {PROCESS_TIMEOUT:?} waiting for {:?} worker message",
                        self.worker
                    ),
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    format!("{:?} worker protocol reader stopped", self.worker),
                ));
            }
        };
        if actual != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{:?} worker sent {actual:?}, expected {expected:?}",
                    self.worker
                ),
            ));
        }
        Ok(())
    }

    fn wait_for_exit(&mut self) -> io::Result<std::process::ExitStatus> {
        let deadline = Instant::now() + PROCESS_TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "timed out after {PROCESS_TIMEOUT:?} reaping {:?} worker",
                        self.worker
                    ),
                ));
            }
            thread::sleep(REAP_POLL_INTERVAL);
        }
    }

    fn join_reader(&mut self) {
        if let Some(reader_thread) = self.reader_thread.take() {
            let _ = reader_thread.join();
        }
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        self.input.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.reaped = true;
        self.join_reader();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_worker_paths_in_either_order() {
        let config = Config::parse([
            OsString::from("--content-worker"),
            OsString::from("content"),
            OsString::from("--controller-worker"),
            OsString::from("controller"),
        ])
        .unwrap();

        assert_eq!(config.controller_worker, PathBuf::from("controller"));
        assert_eq!(config.content_worker, PathBuf::from("content"));
    }

    #[test]
    fn requires_both_worker_paths() {
        let error = Config::parse([
            OsString::from("--controller-worker"),
            OsString::from("controller"),
        ])
        .unwrap_err();
        assert!(error.contains("--content-worker"));
    }
}
