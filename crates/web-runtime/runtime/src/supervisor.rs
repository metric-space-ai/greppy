use crate::protocol::{read_message, write_message, Message, WorkerKind};
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const SCRIPT_TIMEOUT: Duration = Duration::from_secs(120);
const REAP_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Eq, PartialEq)]
pub struct Config {
    pub controller_worker: PathBuf,
    pub content_worker: PathBuf,
    pub scripts: Vec<PathBuf>,
    pub fixture_url: Option<String>,
    pub socket: Option<PathBuf>,
    pub run_id: Option<String>,
}

impl Config {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, String> {
        let mut controller_worker = None;
        let mut content_worker = None;
        let mut scripts = Vec::new();
        let mut fixture_url = None;
        let mut socket = None;
        let mut run_id = None;
        let mut args = args.into_iter();

        while let Some(argument) = args.next() {
            match argument.to_str() {
                Some("--controller-worker") => {
                    set_path(&mut controller_worker, "--controller-worker", args.next())?;
                }
                Some("--content-worker") => {
                    set_path(&mut content_worker, "--content-worker", args.next())?;
                }
                Some("--script") => {
                    let mut script = None;
                    set_path(&mut script, "--script", args.next())?;
                    scripts.push(script.expect("script path"));
                }
                Some("--fixture-url") => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value after --fixture-url".to_owned())?;
                    let value = value
                        .into_string()
                        .map_err(|value| format!("invalid --fixture-url {value:?}"))?;
                    if value.is_empty() {
                        return Err("empty value after --fixture-url".to_owned());
                    }
                    if fixture_url.replace(value).is_some() {
                        return Err("duplicate --fixture-url".to_owned());
                    }
                }
                Some("--socket") => {
                    set_path(&mut socket, "--socket", args.next())?;
                }
                Some("--run-id") => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value after --run-id".to_owned())?;
                    let value = value
                        .into_string()
                        .map_err(|value| format!("invalid --run-id {value:?}"))?;
                    if value.is_empty() {
                        return Err("empty value after --run-id".to_owned());
                    }
                    if run_id.replace(value).is_some() {
                        return Err("duplicate --run-id".to_owned());
                    }
                }
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }

        Ok(Self {
            controller_worker: controller_worker
                .ok_or_else(|| "missing --controller-worker PATH".to_owned())?,
            content_worker: content_worker
                .ok_or_else(|| "missing --content-worker PATH".to_owned())?,
            scripts,
            fixture_url,
            socket,
            run_id,
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
    if let Some(socket) = config.socket.clone() {
        let run_id = config
            .run_id
            .clone()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing --run-id"))?;
        #[cfg(unix)]
        {
            return crate::daemon::serve(crate::daemon::DaemonConfig {
                socket,
                run_id,
                controller_worker: config.controller_worker,
                content_worker: config.content_worker,
                fixture_url: config.fixture_url,
            });
        }
        #[cfg(not(unix))]
        {
            let _ = (socket, run_id);
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "web-runtime supervisor socket mode requires Unix domain sockets",
            ));
        }
    }

    let mut controller = WorkerProcess::spawn(
        &config.controller_worker,
        WorkerKind::Controller,
        random_capability()?,
    )?;
    controller.handshake()?;
    println!("web_runtime.controller=ready");

    let mut content = WorkerProcess::spawn(
        &config.content_worker,
        WorkerKind::Content,
        random_capability()?,
    )?;
    content.handshake()?;
    println!("web_runtime.content=ready");

    for script in &config.scripts {
        let fixture_url = config.fixture_url.clone().unwrap_or_default();
        run_script(&mut controller, &mut content, script, fixture_url)?;
        println!("web_runtime.script=ok");
    }

    content.shutdown()?;
    println!("web_runtime.content=stopped");
    controller.shutdown()?;
    println!("web_runtime.controller=stopped");
    println!("web_runtime.supervisor=stopped");
    Ok(())
}

fn random_capability() -> io::Result<String> {
    let mut bytes = [0_u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn run_script(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    script: &Path,
    fixture_url: String,
) -> io::Result<()> {
    let source = std::fs::read_to_string(script).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to read script {}: {error}", script.display()),
        )
    })?;
    let specifier = script
        .canonicalize()
        .unwrap_or_else(|_| script.to_path_buf())
        .to_string_lossy()
        .into_owned();
    controller.send(&Message::run_script(specifier, source, fixture_url))?;
    route_until_script_complete(controller, content, SCRIPT_TIMEOUT)
}

pub(crate) fn route_until_script_complete(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    timeout: Duration,
) -> io::Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("timed out after {timeout:?} running controller script"),
            ));
        }
        match recv_any(controller, content, remaining)? {
            Incoming::Controller(Message::EngineCall {
                request_id,
                method,
                params,
                ..
            }) => {
                content.send(&Message::engine_call(request_id, method, params))?;
            }
            Incoming::Content(Message::EngineResult {
                request_id,
                ok,
                result,
                error,
                ..
            }) => {
                controller.send(&Message::engine_result(request_id, ok, result, error))?;
            }
            Incoming::Controller(Message::ScriptComplete {
                ok, result, error, ..
            }) => {
                if !ok {
                    return Err(io::Error::other(format!(
                        "controller script failed: {}",
                        error.unwrap_or_else(|| result.to_string())
                    )));
                }
                return Ok(());
            }
            Incoming::Controller(message) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected controller message during script: {message:?}"),
                ));
            }
            Incoming::Content(message) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected content message during script: {message:?}"),
                ));
            }
        }
    }
}

enum Incoming {
    Controller(Message),
    Content(Message),
}

fn recv_any(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    timeout: Duration,
) -> io::Result<Incoming> {
    let deadline = Instant::now() + timeout;
    loop {
        match controller.messages.try_recv() {
            Ok(message) => return Ok(Incoming::Controller(message?)),
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "controller worker protocol reader stopped",
                ));
            }
        }
        match content.messages.try_recv() {
            Ok(message) => return Ok(Incoming::Content(message?)),
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "content worker protocol reader stopped",
                ));
            }
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for worker protocol traffic",
            ));
        }
        thread::sleep(REAP_POLL_INTERVAL);
    }
}

pub(crate) struct WorkerProcess {
    worker: WorkerKind,
    child: Child,
    input: Option<BufWriter<ChildStdin>>,
    messages: Receiver<io::Result<Message>>,
    reader_thread: Option<JoinHandle<()>>,
    reaped: bool,
}

fn inherited_worker_env() -> Vec<(OsString, OsString)> {
    const ALLOW: &[&str] = &[
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "LD_LIBRARY_PATH",
        "FONTCONFIG_PATH",
        "FONTCONFIG_FILE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
    ];
    std::env::vars_os()
        .filter(|(key, _)| key.to_str().is_some_and(|name| ALLOW.contains(&name)))
        .collect()
}

impl WorkerProcess {
    pub(crate) fn spawn(path: &Path, worker: WorkerKind, capability: String) -> io::Result<Self> {
        let mut child = Command::new(path)
            .arg("--capability")
            .arg(capability)
            .env_clear()
            .envs(inherited_worker_env())
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

    pub(crate) fn handshake(&mut self) -> io::Result<()> {
        self.send(&Message::hello(self.worker))?;
        self.expect(Message::ready(self.worker))
    }

    pub(crate) fn shutdown(&mut self) -> io::Result<()> {
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

    pub(crate) fn recv(&mut self, timeout: Duration) -> io::Result<Message> {
        match self.messages.recv_timeout(timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "timed out after {timeout:?} waiting for {:?} worker message",
                    self.worker
                ),
            )),
            Err(RecvTimeoutError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("{:?} worker protocol reader stopped", self.worker),
            )),
        }
    }

    pub(crate) fn is_running(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) => {
                self.reaped = true;
                false
            }
            Err(_) => false,
        }
    }

    pub(crate) fn send(&mut self, message: &Message) -> io::Result<()> {
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
        assert_eq!(config.scripts, Vec::<PathBuf>::new());
        assert_eq!(config.socket, None);
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
