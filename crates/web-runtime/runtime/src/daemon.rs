//! Unix-socket client/supervisor daemon (guide §6.3, §9).

use crate::protocol::{Message, WorkerKind};
use crate::session::{Session, SessionState};
use crate::supervisor::WorkerProcess;
use greppy_web_client::{
    new_session_id, serve_connection, ErrorObject, Handshake, Request, Response, SCHEMA,
};
use std::collections::HashMap;
use std::io;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct DaemonConfig {
    pub socket: PathBuf,
    pub run_id: String,
    pub controller_worker: PathBuf,
    pub content_worker: PathBuf,
    pub fixture_url: Option<String>,
}

pub fn serve(config: DaemonConfig) -> io::Result<()> {
    if let Some(parent) = config.socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(&config.socket);
    let listener = UnixListener::bind(&config.socket)?;
    let mut daemon = Daemon::start(config)?;
    for connection in listener.incoming() {
        let stream = connection?;
        serve_connection(stream, |request| daemon.handle(request)).map_err(io::Error::other)?;
    }
    Ok(())
}

struct Daemon {
    run_id: String,
    fixture_url: String,
    controller: WorkerProcess,
    content: WorkerProcess,
    sessions: HashMap<String, Session>,
}

impl Daemon {
    fn start(config: DaemonConfig) -> io::Result<Self> {
        let mut controller = WorkerProcess::spawn(
            &config.controller_worker,
            WorkerKind::Controller,
            random_token()?,
        )?;
        controller.handshake()?;
        let mut content =
            WorkerProcess::spawn(&config.content_worker, WorkerKind::Content, random_token()?)?;
        content.handshake()?;
        Ok(Self {
            run_id: config.run_id,
            fixture_url: config.fixture_url.unwrap_or_default(),
            controller,
            content,
            sessions: HashMap::new(),
        })
    }

    fn handle(&mut self, request: Request) -> Response {
        if request.schema != SCHEMA {
            return Response::error(
                &request,
                ErrorObject::new(
                    "protocol_violation",
                    format!("unsupported schema {}", request.schema),
                    request.request_id.clone(),
                    30,
                    "send schema greppy.web-runtime.v1",
                ),
            );
        }
        if request.run_id != self.run_id {
            let mut error = ErrorObject::new(
                "session_not_owned",
                "run_id does not match this supervisor",
                request.request_id.clone(),
                32,
                "create a session under this Greppy run",
            );
            error.session_id = request.session_id.clone();
            return Response::error(&request, error);
        }
        match request.operation.as_str() {
            "handshake" => self.handshake(&request),
            "web.status" | "web.doctor" => self.status(&request),
            "web.session.create" => self.session_create(&request),
            "web.session.list" => self.session_list(&request),
            "web.session.close" => self.session_close(&request),
            "web.run" => self.web_run(&request),
            other => Response::error(
                &request,
                ErrorObject::new(
                    "unsupported_playwright_operation",
                    format!("{other} is not implemented in this runtime build"),
                    request.request_id.clone(),
                    31,
                    "use web.status, web.session.*, or web.run",
                ),
            ),
        }
    }

    fn handshake(&self, request: &Request) -> Response {
        let mut response = Response::ok(
            request,
            serde_json::json!({
                "label": "experimental web-runtime spike",
            }),
        );
        response.handshake = Some(Handshake {
            protocol_version: SCHEMA.to_owned(),
            runtime_build_id: "web-runtime-0.1.0".to_owned(),
            playwright_compatibility_version: "1.62.1".to_owned(),
            servo_revision: "77fccacc1f1fdce10498d50173aafaa09d02879e".to_owned(),
            v8_revision: "deno_core-0.410.0".to_owned(),
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            supported_capabilities: vec![
                "chromium.launch".into(),
                "session".into(),
                "web.run".into(),
            ],
            compatibility_coverage_level: "unverified".to_owned(),
            max_message_bytes: greppy_web_client::MAX_FRAME_BYTES as u64,
            max_artifact_bytes: greppy_web_client::MAX_FRAME_BYTES as u64,
        });
        response
    }

    fn status(&self, request: &Request) -> Response {
        Response::ok(
            request,
            serde_json::json!({
                "label": "experimental web-runtime spike",
                "playwright_compatibility_version": "1.62.1",
                "compatibility_coverage_level": "unverified",
                "sessions": self.sessions.len(),
                "workers": 2,
                "engines_linked_into_greppy_parent": false,
            }),
        )
    }

    fn session_create(&mut self, request: &Request) -> Response {
        let profile = request
            .payload
            .get("profile")
            .and_then(|v| v.as_str())
            .unwrap_or("research");
        if profile != "research" && profile != "project" {
            return Response::error(
                request,
                ErrorObject::new(
                    "protocol_violation",
                    "profile must be research or project",
                    request.request_id.clone(),
                    30,
                    "pass --profile research|project",
                ),
            );
        }
        let id = new_session_id();
        let mut session = Session::new(&id, &self.run_id);
        if session.transition(SessionState::Ready).is_err() {
            return Response::error(
                request,
                ErrorObject::new(
                    "engine_error",
                    "failed to create session",
                    request.request_id.clone(),
                    38,
                    "retry web.session.create",
                ),
            );
        }
        self.sessions.insert(id.clone(), session);
        Response::ok(
            request,
            serde_json::json!({
                "session_id": id,
                "profile": profile,
                "state": "ready",
            }),
        )
    }

    fn session_list(&self, request: &Request) -> Response {
        let sessions: Vec<_> = self
            .sessions
            .values()
            .map(|session| {
                serde_json::json!({
                    "session_id": session.id,
                    "state": format!("{:?}", session.state).to_lowercase(),
                    "run_id": session.run_id,
                })
            })
            .collect();
        Response::ok(request, serde_json::json!({ "sessions": sessions }))
    }

    fn session_close(&mut self, request: &Request) -> Response {
        let Some(session_id) = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone())
        else {
            return Response::error(
                request,
                ErrorObject::new(
                    "protocol_violation",
                    "session_id is required",
                    request.request_id.clone(),
                    30,
                    "pass the session id",
                ),
            );
        };
        match self.sessions.remove(&session_id) {
            Some(mut session) => {
                let _ = session.transition(SessionState::Closing);
                let _ = session.transition(SessionState::Closed);
                Response::ok(
                    request,
                    serde_json::json!({ "session_id": session_id, "state": "closed" }),
                )
            }
            None => {
                let mut error = ErrorObject::new(
                    "session_not_found",
                    format!("session {session_id} was not found"),
                    request.request_id.clone(),
                    32,
                    "create a session first",
                );
                error.session_id = Some(session_id);
                Response::error(request, error)
            }
        }
    }

    fn web_run(&mut self, request: &Request) -> Response {
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return Response::error(
                request,
                ErrorObject::new(
                    "protocol_violation",
                    "web.run requires session_id",
                    request.request_id.clone(),
                    30,
                    "create a session and pass --session",
                ),
            );
        };
        if !self.sessions.contains_key(&session_id) {
            let mut error = ErrorObject::new(
                "session_not_found",
                format!("session {session_id} was not found"),
                request.request_id.clone(),
                32,
                "create a session first",
            );
            error.session_id = Some(session_id);
            return Response::error(request, error);
        }
        if let Some(session) = self.sessions.get_mut(&session_id) {
            if let Err(message) = session.begin_operation(&request.request_id) {
                return Response::error(
                    request,
                    ErrorObject::new(
                        "engine_error",
                        message,
                        request.request_id.clone(),
                        38,
                        "wait for the session to become ready",
                    ),
                );
            }
        }
        let source = request
            .payload
            .get("script_text")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let file = request
            .payload
            .get("script_file")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let (specifier, source) = match (source, file) {
            (Some(text), _) => ("greppy:stdin".to_owned(), text),
            (None, Some(path)) => match std::fs::read_to_string(&path) {
                Ok(text) => (path, text),
                Err(error) => {
                    self.finish_session(&session_id);
                    return Response::error(
                        request,
                        ErrorObject::new(
                            "protocol_violation",
                            format!("cannot read script file: {error}"),
                            request.request_id.clone(),
                            30,
                            "pass a readable --script-file",
                        ),
                    );
                }
            },
            (None, None) => {
                self.finish_session(&session_id);
                return Response::error(
                    request,
                    ErrorObject::new(
                        "protocol_violation",
                        "web.run requires script_text or script_file",
                        request.request_id.clone(),
                        30,
                        "use --script-file or --script-stdin",
                    ),
                );
            }
        };
        let started = Instant::now();
        let outcome = run_script_on_workers(
            &mut self.controller,
            &mut self.content,
            &specifier,
            source,
            self.fixture_url.clone(),
            Duration::from_millis(request.deadline_ms.max(1_000)),
        );
        self.finish_session(&session_id);
        match outcome {
            Ok(()) => {
                let mut response = Response::ok(
                    request,
                    serde_json::json!({ "session_id": session_id, "completed": true }),
                );
                response.metrics.wall_ms = started.elapsed().as_millis() as u64;
                response
            }
            Err(error) => {
                let mut object = ErrorObject::new(
                    "controller_exception",
                    error.to_string(),
                    request.request_id.clone(),
                    33,
                    "inspect the controller script and retry",
                );
                object.session_id = Some(session_id);
                Response::error(request, object)
            }
        }
    }

    fn finish_session(&mut self, session_id: &str) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            let _ = session.transition(SessionState::Ready);
        }
    }
}

fn run_script_on_workers(
    controller: &mut WorkerProcess,
    content: &mut WorkerProcess,
    specifier: &str,
    source: String,
    fixture_url: String,
    timeout: Duration,
) -> io::Result<()> {
    controller.send(&Message::run_script(
        specifier.to_owned(),
        source,
        fixture_url,
    ))?;
    crate::supervisor::route_until_script_complete(controller, content, timeout)
}

fn random_token() -> io::Result<String> {
    use std::io::Read;
    let mut bytes = [0_u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub fn socket_exists(path: &Path) -> bool {
    path.exists()
}
