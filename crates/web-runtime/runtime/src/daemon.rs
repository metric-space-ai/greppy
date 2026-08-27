//! Unix-socket client/supervisor daemon (guide §6.3, §9).

use crate::artifacts::ArtifactStore;
use crate::policy::{decide_url, NetworkProfile, UrlDecision};
use crate::protocol::{Message, WorkerKind};
use crate::session::{Session, SessionState};
use crate::supervisor::WorkerProcess;
use greppy_web_client::{
    new_session_id, serve_connection, ErrorObject, Handshake, Request, Response, SCHEMA,
};
use serde_json::json;
use std::collections::HashMap;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    let mut daemon = Daemon::start(config)?;
    let listener = UnixListener::bind(&daemon.socket)?;
    let mut permissions = std::fs::metadata(&daemon.socket)?.permissions();
    permissions.set_mode(0o600);
    std::fs::set_permissions(&daemon.socket, permissions)?;
    // A closed probe connection or a single malformed client must not take
    // down the supervisor; leftover-worker flakes showed up as "socket never
    // created" when this loop exited.
    for connection in listener.incoming() {
        let stream = match connection {
            Ok(stream) => stream,
            Err(_) => continue,
        };
        let _ = serve_connection(stream, |request| daemon.handle(request));
    }
    Ok(())
}

struct Daemon {
    socket: PathBuf,
    run_id: String,
    fixture_url: String,
    search_endpoint: Option<String>,
    store: ArtifactStore,
    controller_worker: PathBuf,
    content_worker: PathBuf,
    controller: WorkerProcess,
    content: WorkerProcess,
    sessions: HashMap<String, Session>,
    next_engine_id: AtomicU64,
    last_crash: Option<String>,
    crash_receipts: Vec<serde_json::Value>,
    last_request: Instant,
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
        let data_root = data_root(&config.run_id);
        Ok(Self {
            socket: config.socket,
            run_id: config.run_id.clone(),
            fixture_url: config.fixture_url.unwrap_or_default(),
            search_endpoint: std::env::var("GREPPY_WEB_SEARCH_ENDPOINT").ok(),
            store: ArtifactStore::new(data_root)?,
            controller_worker: config.controller_worker,
            content_worker: config.content_worker,
            controller,
            content,
            sessions: HashMap::new(),
            next_engine_id: AtomicU64::new(1),
            last_crash: None,
            crash_receipts: Vec::new(),
            last_request: Instant::now(),
        })
    }

    fn handle(&mut self, request: Request) -> Response {
        self.last_request = Instant::now();
        self.reap_idle_sessions();
        self.ensure_workers();
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
            "web.observe" => self.web_observe(&request),
            "web.screenshot" => self.web_screenshot(&request),
            "web.read" => self.web_read(&request),
            "web.search" => self.web_search(&request),
            "web.research" => self.web_research(&request),
            "web.artifacts" => self.web_artifacts(&request),
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
                "web.observe".into(),
                "web.screenshot".into(),
                "web.read".into(),
                "web.search".into(),
                "web.research".into(),
                "web.artifacts".into(),
                "page.route".into(),
                "page.frames".into(),
                "page.setInputFiles".into(),
            ],
            compatibility_coverage_level: "unverified".to_owned(),
            max_message_bytes: greppy_web_client::MAX_FRAME_BYTES as u64,
            max_artifact_bytes: greppy_web_client::MAX_FRAME_BYTES as u64,
        });
        response
    }

    fn status(&mut self, request: &Request) -> Response {
        let idle = self
            .sessions
            .values()
            .filter(|session| session.state == SessionState::Ready)
            .count();
        let busy = self
            .sessions
            .values()
            .filter(|session| session.state == SessionState::Busy)
            .count();
        let failed = self
            .sessions
            .values()
            .filter(|session| session.state == SessionState::Failed)
            .count();
        let controller_alive = self.controller.is_running();
        let content_alive = self.content.is_running();
        Response::ok(
            request,
            serde_json::json!({
                "label": "experimental web-runtime spike",
                "runtime_version": "0.1.0",
                "runtime_build_id": "web-runtime-0.1.0",
                "playwright_compatibility_version": "1.62.1",
                "compatibility_coverage_level": "unverified",
                "process_health": {
                    "controller_alive": controller_alive,
                    "content_alive": content_alive,
                    "healthy": controller_alive && content_alive,
                },
                "sessions": self.sessions.len(),
                "session_counts": {
                    "total": self.sessions.len(),
                    "idle": idle,
                    "active": busy,
                    "failed": failed,
                },
                "ready": idle,
                "busy": busy,
                "failed": failed,
                "workers": 2,
                "controller_alive": controller_alive,
                "content_alive": content_alive,
                "resource_totals": {
                    "sessions": self.sessions.len(),
                    "workers": 2,
                    "crash_receipts": self.crash_receipts.len(),
                },
                "last_crash": self.last_crash.clone(),
                "crash_receipts": self.crash_receipts.clone(),
                "unsupported_capability_count": 501,
                "conformance_receipt_id": "contracts/web-runtime/receipts/oracle-setcontent.json",
                "engines_linked_into_greppy_parent": false,
                "signed_distributable": false,
                "oracle_receipt": "contracts/web-runtime/receipts/oracle-setcontent.json",
                "oracle_receipts": [
                    "contracts/web-runtime/receipts/oracle-setcontent.json",
                    "contracts/web-runtime/receipts/oracle-dialog.json",
                    "contracts/web-runtime/receipts/oracle-fill.json",
                    "contracts/web-runtime/receipts/oracle-console.json",
                    "contracts/web-runtime/receipts/oracle-content.json"
                ],
                "inventory_entries": 1354,
                "compatibility_coverage_level_note": "schema implemented is not Chromium oracle behavior; oracle receipts are scoped cases only",
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
        let parsed = NetworkProfile::parse(profile).expect("validated");
        let id = new_session_id();
        let mut session = Session::new(&id, &self.run_id, parsed);
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
        self.journal(&id, "session.ready", json!({ "profile": profile }));
        Response::ok(
            request,
            json!({
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
                if let Some(page) = session.page_id.take() {
                    let _ = self.engine_call("page.close", json!({ "page": page }));
                }
                let _ = session.transition(SessionState::Closed);
                self.journal(&session_id, "session.closed", json!({}));
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
        if !self.controller.is_running() {
            if let Err(error) = self.recover_controller("controller worker exited") {
                self.finish_session(&session_id);
                return engine_error(request, error, 33);
            }
        }
        let profile = self
            .sessions
            .get(&session_id)
            .map(|session| session.profile)
            .unwrap_or(NetworkProfile::Research);
        if let Err(error) =
            self.engine_call("session.setProfile", json!({ "profile": profile.as_str() }))
        {
            self.finish_session(&session_id);
            return engine_error(request, error, 34);
        }
        let started = Instant::now();
        let content_pid = self.content.pid();
        let controller_pid = self.controller.pid();
        let outcome = {
            let controller = &mut self.controller;
            let content = &mut self.content;
            let sessions = &mut self.sessions;
            let session_key = session_id.clone();
            if let Err(error) = controller.send(&crate::protocol::Message::run_script(
                specifier.clone(),
                source,
                self.fixture_url.clone(),
            )) {
                Err(error)
            } else {
                crate::supervisor::route_until_script_complete_gated(
                    controller,
                    content,
                    Duration::from_millis(request.deadline_ms.max(1_000)),
                    |method, params| {
                        gate_session_engine(
                            sessions,
                            &session_key,
                            content_pid,
                            controller_pid,
                            method,
                            params,
                        )
                    },
                )
            }
        };
        let (network_bytes, peak_rss) = self
            .sessions
            .get(&session_id)
            .map(|session| (session.network_bytes, session.peak_rss_bytes))
            .unwrap_or((0, 0));
        self.finish_session(&session_id);
        match outcome {
            Ok(()) => {
                let mut response = Response::ok(
                    request,
                    serde_json::json!({ "session_id": session_id, "completed": true }),
                );
                response.metrics.wall_ms = started.elapsed().as_millis() as u64;
                response.metrics.network_bytes = network_bytes;
                response.metrics.peak_rss_bytes = peak_rss.max(sample_rss_bytes(content_pid));
                response.metrics.content_cpu_ms = sample_cpu_ms(content_pid);
                response.metrics.controller_cpu_ms = sample_cpu_ms(controller_pid);
                response
            }
            Err(error) => {
                let message = error.to_string();
                if let Some(limit) = message.strip_prefix("resource_limit: ") {
                    let mut response = limit_error(request, limit);
                    if let Some(error) = response.error.as_mut() {
                        error.session_id = Some(session_id);
                    }
                    response.metrics.wall_ms = started.elapsed().as_millis() as u64;
                    response.metrics.network_bytes = network_bytes;
                    response.metrics.peak_rss_bytes = peak_rss;
                    return response;
                }
                let mut object = ErrorObject::new(
                    "controller_exception",
                    message,
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

    fn web_observe(&mut self, request: &Request) -> Response {
        match self.with_session_page(request, "web.observe") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.engine_call("page.observe", json!({ "page": page })) {
                    Ok(mut tree) => {
                        self.finish_session(&session_id);
                        if let Some(object) = tree.as_object_mut() {
                            object.insert(
                                "untrusted_content_boundary".into(),
                                json!("UNTRUSTED_PAGE_CONTENT"),
                            );
                        }
                        Response::ok(request, tree)
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        engine_error(request, error, 34)
                    }
                }
            }
        }
    }

    fn web_screenshot(&mut self, request: &Request) -> Response {
        match self.with_session_page(request, "web.screenshot") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.engine_call("page.screenshot", json!({ "page": page })) {
                    Ok(result) => {
                        let Some(b64) = result.get("png_base64").and_then(|v| v.as_str()) else {
                            self.finish_session(&session_id);
                            return engine_error(request, "screenshot missing png", 34);
                        };
                        let bytes = match decode_base64(b64) {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                self.finish_session(&session_id);
                                return engine_error(request, error, 34);
                            }
                        };
                        let stored = self.store_bytes(
                            request,
                            &session_id,
                            &bytes,
                            "image/png",
                            "web.screenshot",
                            true,
                        );
                        self.finish_session(&session_id);
                        match stored {
                            Ok(manifest) => {
                                let mut response = Response::ok(
                                    request,
                                    json!({
                                        "session_id": session_id,
                                        "digest": manifest.digest.hex,
                                        "byte_count": manifest.byte_count,
                                        "object_path": manifest.object_path,
                                    }),
                                );
                                response
                                    .artifacts
                                    .push(serde_json::to_value(manifest).unwrap_or(json!({})));
                                response
                            }
                            Err(response) => response,
                        }
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        engine_error(request, error, 34)
                    }
                }
            }
        }
    }

    fn web_read(&mut self, request: &Request) -> Response {
        let Some(url) = request.payload.get("url").and_then(|v| v.as_str()) else {
            return protocol_error(request, "web.read requires url");
        };
        match self.with_session_page(request, "web.read") {
            Err(response) => response,
            Ok((session_id, page)) => {
                match self.navigate_and_extract(&session_id, &page, url, request) {
                    Ok(source) => {
                        let mut response = Response::ok(
                            request,
                            json!({
                                "session_id": session_id,
                                "source": source,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        );
                        self.attach_session_metrics(&session_id, &mut response);
                        self.finish_session(&session_id);
                        response
                    }
                    Err(response) => {
                        self.finish_session(&session_id);
                        response
                    }
                }
            }
        }
    }

    fn web_search(&mut self, request: &Request) -> Response {
        let Some(query) = request.payload.get("query").and_then(|v| v.as_str()) else {
            return protocol_error(request, "web.search requires query");
        };
        let limit = request
            .payload
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .max(1) as usize;
        match self.with_session_page(request, "web.search") {
            Err(response) => response,
            Ok((session_id, page)) => {
                let search_url = self.search_url(query);
                match self.navigate_and_extract(&session_id, &page, &search_url, request) {
                    Ok(mut source) => {
                        let links = self
                            .engine_call("page.observe", json!({ "page": page }))
                            .ok()
                            .and_then(|tree| tree.get("links").cloned())
                            .and_then(|value| value.as_array().cloned())
                            .unwrap_or_default();
                        let results: Vec<_> = links.into_iter().take(limit).collect();
                        source["classification"] = json!("aggregator");
                        self.finish_session(&session_id);
                        Response::ok(
                            request,
                            json!({
                                "query": query,
                                "results": results,
                                "source": source,
                                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                            }),
                        )
                    }
                    Err(response) => {
                        self.finish_session(&session_id);
                        response
                    }
                }
            }
        }
    }

    fn web_research(&mut self, request: &Request) -> Response {
        let Some(query) = request.payload.get("query").and_then(|v| v.as_str()) else {
            return protocol_error(request, "web.research requires query");
        };
        let max_sources = request
            .payload
            .get("max_sources")
            .and_then(|v| v.as_u64())
            .unwrap_or(3)
            .clamp(1, 8) as usize;
        let search = self.web_search(request);
        if search.status != "ok" {
            return search;
        }
        let results = search
            .result
            .as_ref()
            .and_then(|value| value.get("results"))
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        let mut admitted = Vec::new();
        let mut omitted = 0u32;
        let mut omitted_reasons = Vec::new();
        for result in results.into_iter().take(max_sources) {
            let Some(href) = result.get("href").and_then(|v| v.as_str()) else {
                omitted += 1;
                omitted_reasons.push(json!({"reason": "missing href"}));
                continue;
            };
            let mut read_req = request.clone();
            read_req.payload =
                json!({ "url": href, "session_id": request.payload.get("session_id") });
            match self.web_read(&read_req) {
                response if response.status == "ok" => {
                    if let Some(source) = response
                        .result
                        .and_then(|value| value.get("source").cloned())
                    {
                        admitted.push(source);
                    } else {
                        omitted += 1;
                        omitted_reasons.push(json!({
                            "url": href,
                            "reason": "read returned no source",
                        }));
                    }
                }
                response => {
                    omitted += 1;
                    omitted_reasons.push(json!({
                        "url": href,
                        "status": response.status,
                        "error": response.error,
                    }));
                }
            }
        }
        let snippets: Vec<_> = admitted
            .iter()
            .map(|source| {
                json!({
                    "url": source.get("final_url"),
                    "title": source.get("title"),
                    "snippet": source.get("text").and_then(|v| v.as_str()).unwrap_or("").chars().take(280).collect::<String>(),
                    "digest": source.get("digest"),
                })
            })
            .collect();
        let continuation = if omitted > 0 {
            json!(format!("offset={}", admitted.len()))
        } else {
            serde_json::Value::Null
        };
        Response::ok(
            request,
            json!({
                "query_summary": query,
                "admitted_sources": admitted.len(),
                "omitted": omitted,
                "omitted_reasons": omitted_reasons,
                "evidence": snippets,
                "sources": admitted.into_iter().map(model_facing_source).collect::<Vec<_>>(),
                "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
                "continuation_token": continuation,
            }),
        )
    }

    fn web_artifacts(&mut self, request: &Request) -> Response {
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return protocol_error(request, "web.artifacts requires session_id");
        };
        if !self.sessions.contains_key(&session_id) {
            return missing_session(request, &session_id);
        }
        match self.store.list_session(&session_id) {
            Ok(list) => Response::ok(
                request,
                json!({ "session_id": session_id, "artifacts": list }),
            ),
            Err(error) => engine_error(request, error.to_string(), 39),
        }
    }

    fn with_session_page(
        &mut self,
        request: &Request,
        operation: &str,
    ) -> Result<(String, String), Response> {
        let session_id = request
            .payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.session_id.clone());
        let Some(session_id) = session_id else {
            return Err(protocol_error(
                request,
                &format!("{operation} requires session_id"),
            ));
        };
        if !self.sessions.contains_key(&session_id) {
            return Err(missing_session(request, &session_id));
        }
        let content_rss = sample_rss_bytes(self.content.pid());
        let controller_rss = sample_rss_bytes(self.controller.pid());
        let wall_time_error = self.sessions.get_mut(&session_id).and_then(|session| {
            session.peak_rss_bytes = session.peak_rss_bytes.max(content_rss);
            if let Err(message) = session.begin_operation(&request.request_id) {
                Some(("engine", message))
            } else if let Err(message) = session.limits.check_wall_time(session.started.elapsed()) {
                let _ = session.transition(SessionState::Failed);
                Some(("limit", message))
            } else if let Err(message) = session.limits.check_content_rss(content_rss) {
                let _ = session.transition(SessionState::Failed);
                Some(("limit", message))
            } else if let Err(message) = session.limits.check_controller_memory(controller_rss) {
                let _ = session.transition(SessionState::Failed);
                Some(("limit", message))
            } else {
                None
            }
        });
        match wall_time_error {
            Some(("engine", message)) => return Err(engine_error(request, message, 38)),
            Some(("limit", message)) => {
                let _ = self.recover_content(&format!("wall time exceeded: {message}"));
                return Err(limit_error(request, message));
            }
            Some(_) => unreachable!(),
            None => {}
        }
        let page = self
            .sessions
            .get(&session_id)
            .and_then(|session| session.page_id.clone());
        let page = match page {
            Some(page) => page,
            None => {
                if let Some(session) = self.sessions.get(&session_id) {
                    if let Err(message) =
                        session.limits.check_pages(session.pages.saturating_add(1))
                    {
                        return Err(limit_error(request, message));
                    }
                }
                match self.engine_call("session.ensurePage", json!({})) {
                    Ok(result) => {
                        let page = result
                            .get("page")
                            .and_then(|value| value.as_str())
                            .map(str::to_owned);
                        let Some(page) = page else {
                            self.finish_session(&session_id);
                            return Err(engine_error(request, "session has no page", 34));
                        };
                        if let Some(session) = self.sessions.get_mut(&session_id) {
                            session.page_id = Some(page.clone());
                            session.pages = 1;
                        }
                        page
                    }
                    Err(error) => {
                        self.finish_session(&session_id);
                        return Err(engine_error(request, error, 34));
                    }
                }
            }
        };
        let profile = self
            .sessions
            .get(&session_id)
            .map(|session| session.profile)
            .unwrap_or(NetworkProfile::Research);
        if let Err(error) =
            self.engine_call("session.setProfile", json!({ "profile": profile.as_str() }))
        {
            self.finish_session(&session_id);
            return Err(engine_error(request, error, 34));
        }
        Ok((session_id, page))
    }

    fn navigate_and_extract(
        &mut self,
        session_id: &str,
        page: &str,
        url: &str,
        request: &Request,
    ) -> Result<serde_json::Value, Response> {
        let profile = self
            .sessions
            .get(session_id)
            .map(|session| session.profile)
            .unwrap_or(NetworkProfile::Research);
        if let UrlDecision::Deny { reason } = decide_url(profile, url) {
            return Err({
                let mut error = ErrorObject::new(
                    "policy_denied",
                    format!("{reason}: {}", redact_secrets(url)),
                    request.request_id.clone(),
                    36,
                    "use the project profile for loopback fixtures",
                );
                error.session_id = Some(session_id.to_owned());
                Response::error(request, error)
            });
        }
        if let Some(session) = self.sessions.get_mut(session_id) {
            if let Err(message) = session
                .limits
                .check_requests(session.requests.saturating_add(1))
            {
                return Err(limit_error(request, message));
            }
            if let Err(message) = session
                .limits
                .check_network_bytes(session.network_bytes, 4096)
            {
                return Err(limit_error(request, message));
            }
            session.requests = session.requests.saturating_add(1);
            session.network_bytes = session.network_bytes.saturating_add(4096);
        }
        self.engine_call("page.goto", json!({ "page": page, "url": url }))
            .map_err(|error| engine_error(request, error, 34))?;
        let tree = self
            .engine_call("page.observe", json!({ "page": page }))
            .map_err(|error| engine_error(request, error, 34))?;
        let recorded = self
            .engine_call("page.requests", json!({ "page": page }))
            .unwrap_or_else(|_| json!({ "requests": [] }));
        let responses = self
            .engine_call("page.responses", json!({ "page": page }))
            .ok()
            .and_then(|value| value.get("responses").cloned())
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default();
        let http_status = responses
            .iter()
            .rev()
            .find_map(|row| row.get("status").and_then(|value| value.as_u64()));
        let text = tree
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let title = tree
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let stored = self.store_bytes(
            request,
            session_id,
            text.as_bytes(),
            "text/plain",
            &request.operation,
            false,
        )?;
        Ok(model_facing_source(json!({
            "requested_url": url,
            "final_url": tree.get("url"),
            "redirect_chain": redirect_chain(url, tree.get("url"), recorded.get("requests")),
            "retrieved_at": stored.timestamp,
            "title": title,
            "media_type": "text/html",
            "text": text,
            "digest": stored.digest.hex,
            "artifact_digest": stored.digest.hex,
            "http_status": http_status,
            "classification": "original",
            "session_id": session_id,
            "operation_id": request.request_id,
            "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
        })))
    }

    fn store_bytes(
        &mut self,
        request: &Request,
        session_id: &str,
        bytes: &[u8],
        media_type: &str,
        operation: &str,
        sensitive: bool,
    ) -> Result<crate::artifacts::ArtifactManifest, Response> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if let Err(message) = session
                .limits
                .check_artifact_bytes(session.artifact_bytes, bytes.len() as u64)
            {
                return Err(limit_error(request, message));
            }
            session.artifact_bytes = session.artifact_bytes.saturating_add(bytes.len() as u64);
        }
        self.store
            .put(
                bytes,
                media_type,
                session_id,
                &self.run_id,
                &format!("{operation}:{}", request.request_id),
                sensitive,
            )
            .map_err(|error| engine_error(request, error.to_string(), 39))
    }

    fn search_url(&self, query: &str) -> String {
        if let Some(endpoint) = &self.search_endpoint {
            if endpoint.contains('?') {
                format!("{endpoint}&q={}", urlencoding(query))
            } else {
                format!("{endpoint}?q={}", urlencoding(query))
            }
        } else {
            format!("https://html.duckduckgo.com/html/?q={}", urlencoding(query))
        }
    }

    fn engine_call(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        if !self.content.is_running() {
            self.recover_content("content worker exited")?;
            return Err(
                "content worker crashed and was restarted; session pages were reset".into(),
            );
        }
        let request_id = self.next_engine_id.fetch_add(1, Ordering::Relaxed);
        if let Err(error) =
            self.content
                .send(&Message::engine_call(request_id, method.to_owned(), params))
        {
            let _ = self.recover_content(&format!("content send failed: {error}"));
            return Err(error.to_string());
        }
        match self.content.recv(Duration::from_secs(60)) {
            Ok(Message::EngineResult {
                request_id: got,
                ok,
                result,
                error,
                ..
            }) if got == request_id => {
                if ok {
                    Ok(result)
                } else {
                    Err(error.unwrap_or_else(|| "engine call failed".to_owned()))
                }
            }
            Ok(other) => Err(format!("unexpected content message {other:?}")),
            Err(error) => {
                let message = error.to_string();
                let _ = self.recover_content(&format!("content worker: {message}"));
                Err(message)
            }
        }
    }

    fn ensure_workers(&mut self) {
        if !self.content.is_running() {
            let _ = self.recover_content("content worker exited");
        }
        if !self.controller.is_running() {
            let _ = self.recover_controller("controller worker exited");
        }
    }

    fn record_crash(&mut self, worker: &str, reason: &str, recovered: bool) {
        self.last_crash = Some(reason.to_owned());
        let recovered_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        self.crash_receipts.push(json!({
            "kind": "worker_crash",
            "worker": worker,
            "reason": reason,
            "recovered": recovered,
            "recovered_at_unix_ms": recovered_at_unix_ms,
        }));
    }

    fn recover_controller(&mut self, reason: &str) -> Result<(), String> {
        let token = match random_token() {
            Ok(token) => token,
            Err(error) => {
                self.record_crash("controller", reason, false);
                return Err(error.to_string());
            }
        };
        let mut controller =
            match WorkerProcess::spawn(&self.controller_worker, WorkerKind::Controller, token) {
                Ok(controller) => controller,
                Err(error) => {
                    self.record_crash("controller", reason, false);
                    return Err(error.to_string());
                }
            };
        if let Err(error) = controller.handshake() {
            self.record_crash("controller", reason, false);
            return Err(error.to_string());
        }
        self.controller = controller;
        self.record_crash("controller", reason, true);
        Ok(())
    }

    fn recover_content(&mut self, reason: &str) -> Result<(), String> {
        let token = match random_token() {
            Ok(token) => token,
            Err(error) => {
                self.record_crash("content", reason, false);
                return Err(error.to_string());
            }
        };
        let mut content =
            match WorkerProcess::spawn(&self.content_worker, WorkerKind::Content, token) {
                Ok(content) => content,
                Err(error) => {
                    self.record_crash("content", reason, false);
                    return Err(error.to_string());
                }
            };
        if let Err(error) = content.handshake() {
            self.record_crash("content", reason, false);
            return Err(error.to_string());
        }
        self.content = content;
        self.record_crash("content", reason, true);
        let mut failed = Vec::new();
        for session in self.sessions.values_mut() {
            session.page_id = None;
            session.pages = 0;
            if session.state != SessionState::Failed
                && session.state != SessionState::Closing
                && session.state != SessionState::Closed
            {
                let _ = session.transition(SessionState::Failed);
                failed.push(session.id.clone());
            }
        }
        for session_id in failed {
            self.journal(&session_id, "session.failed", json!({ "reason": reason }));
        }
        self.journal("runtime", "content.recovered", json!({ "reason": reason }));
        Ok(())
    }

    fn journal(&self, session_id: &str, event: &str, extra: serde_json::Value) {
        let path = self
            .store
            .root()
            .join("sessions")
            .join(session_id)
            .join("journal.jsonl");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let line = json!({
            "event": event,
            "session_id": session_id,
            "run_id": self.run_id,
            "extra": extra,
        });
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }

    fn reap_idle_sessions(&mut self) {
        let now = Instant::now();
        let stale: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, session)| {
                session.state != SessionState::Busy
                    && now.duration_since(session.last_heartbeat) > session.limits.idle_ttl
            })
            .map(|(id, _)| id.clone())
            .collect();
        for session_id in stale {
            if let Some(mut session) = self.sessions.remove(&session_id) {
                if let Some(page) = session.page_id.take() {
                    let _ = self.engine_call("page.close", json!({ "page": page }));
                }
            }
        }
    }
}

fn data_root(run_id: &str) -> PathBuf {
    let base = std::env::var("GREPPY_STORE_DIR")
        .or_else(|_| std::env::var("GREPPY_RUNTIME_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("greppy-web-runtime"));
    base.join("web-runtime").join(run_id)
}

fn urlencoding(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err("invalid base64".into()),
        }
    }
    let filtered: Vec<u8> = input
        .bytes()
        .filter(|c| *c != b'=' && !c.is_ascii_whitespace())
        .collect();
    let mut out = Vec::new();
    for chunk in filtered.chunks(4) {
        let a = val(chunk[0])?;
        let b = val(*chunk.get(1).unwrap_or(&b'A'))?;
        let c = val(*chunk.get(2).unwrap_or(&b'A'))?;
        let d = val(*chunk.get(3).unwrap_or(&b'A'))?;
        let triple = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | d as u32;
        out.push(((triple >> 16) & 255) as u8);
        if chunk.len() > 2 {
            out.push(((triple >> 8) & 255) as u8);
        }
        if chunk.len() > 3 {
            out.push((triple & 255) as u8);
        }
    }
    Ok(out)
}

fn protocol_error(request: &Request, message: &str) -> Response {
    Response::error(
        request,
        ErrorObject::new(
            "protocol_violation",
            message,
            request.request_id.clone(),
            30,
            "see greppy web --help",
        ),
    )
}

fn missing_session(request: &Request, session_id: &str) -> Response {
    let mut error = ErrorObject::new(
        "session_not_found",
        format!("session {session_id} was not found"),
        request.request_id.clone(),
        32,
        "create a session first",
    );
    error.session_id = Some(session_id.to_owned());
    Response::error(request, error)
}

fn engine_error(request: &Request, message: impl Into<String>, exit_code: i32) -> Response {
    Response::error(
        request,
        ErrorObject::new(
            "engine_error",
            redact_secrets(&message.into()),
            request.request_id.clone(),
            exit_code,
            "retry the operation or inspect web.doctor",
        ),
    )
}

fn limit_error(request: &Request, message: impl Into<String>) -> Response {
    Response::error(
        request,
        ErrorObject::new(
            "resource_limit",
            redact_secrets(&message.into()),
            request.request_id.clone(),
            37,
            "close the session or raise the documented limit",
        ),
    )
}

const MODEL_TEXT_CHARS: usize = 4096;

fn model_facing_source(mut source: serde_json::Value) -> serde_json::Value {
    if let Some(text) = source
        .get("text")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
    {
        let truncated = text.chars().count() > MODEL_TEXT_CHARS;
        let snippet: String = text.chars().take(MODEL_TEXT_CHARS).collect();
        if let Some(object) = source.as_object_mut() {
            object.insert("text".into(), json!(snippet));
            object.insert("text_truncated".into(), json!(truncated));
            if truncated {
                object.insert("full_text".into(), json!("artifact"));
            }
        }
    }
    source
}

fn redact_secrets(input: &str) -> String {
    let mut out = input.to_owned();
    if let Some(scheme) = out.find("://") {
        let rest_at = scheme + 3;
        if let Some(at_rel) = out[rest_at..].find('@') {
            let creds = &out[rest_at..rest_at + at_rel];
            if let Some(colon) = creds.find(':') {
                let user = creds[..colon].to_owned();
                out.replace_range(rest_at..rest_at + at_rel, &format!("{user}:****"));
            }
        }
    }
    for key in ["password", "token", "secret", "authorization"] {
        let needle = format!("{key}=");
        let mut search_from = 0;
        let lower = out.to_ascii_lowercase();
        while let Some(rel) = lower[search_from..].find(&needle) {
            let start = search_from + rel + needle.len();
            let end = out[start..]
                .find(|ch: char| matches!(ch, '&' | ' ' | '"' | '\'' | '\n' | '\r'))
                .map(|idx| start + idx)
                .unwrap_or(out.len());
            out.replace_range(start..end, "****");
            search_from = start + 4;
        }
    }
    out
}

fn redirect_chain(
    requested: &str,
    final_url: Option<&serde_json::Value>,
    requests: Option<&serde_json::Value>,
) -> Vec<String> {
    let mut chain = vec![requested.to_owned()];
    if let Some(serde_json::Value::Array(rows)) = requests {
        for row in rows {
            let Some(url) = row.get("url").and_then(|value| value.as_str()) else {
                continue;
            };
            let main = row
                .get("main_frame")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !main {
                continue;
            }
            if chain.last().map(String::as_str) != Some(url) {
                chain.push(url.to_owned());
            }
        }
    }
    if let Some(final_url) = final_url.and_then(|value| value.as_str()) {
        if chain.last().map(String::as_str) != Some(final_url) {
            chain.push(final_url.to_owned());
        }
    }
    chain
}

fn sample_rss_bytes(pid: u32) -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "rss="])
        .output();
    let Ok(output) = output else {
        return 0;
    };
    let kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0);
    kb.saturating_mul(1024)
}

fn sample_cpu_ms(pid: u32) -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "time="])
        .output();
    let Ok(output) = output else {
        return 0;
    };
    parse_ps_time(String::from_utf8_lossy(&output.stdout).trim())
}

fn parse_ps_time(text: &str) -> u64 {
    let text = text.trim();
    if text.is_empty() {
        return 0;
    }
    let mut parts = text.split(':');
    let mut values = [0_f64; 3];
    let mut count = 0;
    for part in parts.by_ref() {
        if count >= 3 {
            break;
        }
        values[count] = part.parse().unwrap_or(0.0);
        count += 1;
    }
    let seconds = match count {
        1 => values[0],
        2 => values[0] * 60.0 + values[1],
        3 => values[0] * 3600.0 + values[1] * 60.0 + values[2],
        _ => 0.0,
    };
    (seconds * 1000.0) as u64
}

fn gate_session_engine(
    sessions: &mut HashMap<String, Session>,
    session_id: &str,
    content_pid: u32,
    controller_pid: u32,
    method: &str,
    params: &serde_json::Value,
) -> Result<(), String> {
    let Some(session) = sessions.get_mut(session_id) else {
        return Err("session was closed".to_owned());
    };
    session.limits.check_wall_time(session.started.elapsed())?;
    let content_rss = sample_rss_bytes(content_pid);
    session.peak_rss_bytes = session.peak_rss_bytes.max(content_rss);
    session.limits.check_content_rss(content_rss)?;
    session
        .limits
        .check_controller_memory(sample_rss_bytes(controller_pid))?;
    match method {
        "browser.newContext" => {
            session
                .limits
                .check_contexts(session.contexts.saturating_add(1))?;
            session.contexts = session.contexts.saturating_add(1);
        }
        "context.newPage" | "session.ensurePage" => {
            session
                .limits
                .check_pages(session.pages.saturating_add(1))?;
            session.pages = session.pages.saturating_add(1);
        }
        "page.goto" | "page.reload" | "page.goBack" | "page.goForward" | "page.frameGoto" => {
            session
                .limits
                .check_requests(session.requests.saturating_add(1))?;
            session
                .limits
                .check_network_bytes(session.network_bytes, 4096)?;
            session.requests = session.requests.saturating_add(1);
            session.network_bytes = session.network_bytes.saturating_add(4096);
        }
        "page.saveDownload" => {
            let extra = params
                .get("byteLength")
                .and_then(|value| value.as_u64())
                .unwrap_or(1);
            session
                .limits
                .check_download_bytes(session.download_bytes, extra)?;
            session.download_bytes = session.download_bytes.saturating_add(extra);
        }
        _ => {}
    }
    Ok(())
}

impl Daemon {
    fn attach_session_metrics(&self, session_id: &str, response: &mut Response) {
        if let Some(session) = self.sessions.get(session_id) {
            response.metrics.network_bytes = session.network_bytes;
            response.metrics.peak_rss_bytes = session.peak_rss_bytes;
        }
        response.metrics.content_cpu_ms = sample_cpu_ms(self.content.pid());
        response.metrics.controller_cpu_ms = sample_cpu_ms(self.controller.pid());
        if response.metrics.peak_rss_bytes == 0 {
            response.metrics.peak_rss_bytes = sample_rss_bytes(self.content.pid());
        }
    }
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

#[cfg(test)]
mod redirect_chain_tests {
    use super::redirect_chain;
    use serde_json::json;

    #[test]
    fn parse_ps_time_minutes_and_seconds() {
        assert_eq!(super::parse_ps_time("0:01.50"), 1500);
        assert_eq!(super::parse_ps_time("1:00.00"), 60_000);
        assert_eq!(super::parse_ps_time(""), 0);
    }

    #[test]
    fn recorded_main_frame_hops_are_kept() {
        let requests = json!([
            {"url": "http://example.test/start", "main_frame": true},
            {"url": "http://example.test/asset.css", "main_frame": false},
            {"url": "http://example.test/end", "main_frame": true, "redirect": true}
        ]);
        let chain = redirect_chain(
            "http://example.test/start",
            Some(&json!("http://example.test/end")),
            Some(&requests),
        );
        assert_eq!(
            chain,
            vec![
                "http://example.test/start".to_owned(),
                "http://example.test/end".to_owned()
            ]
        );
    }

    #[test]
    fn final_url_is_appended_when_requests_are_missing() {
        let chain = redirect_chain(
            "http://example.test/start",
            Some(&json!("http://example.test/end")),
            None,
        );
        assert_eq!(
            chain,
            vec![
                "http://example.test/start".to_owned(),
                "http://example.test/end".to_owned()
            ]
        );
    }

    #[test]
    fn redact_secrets_masks_userinfo_and_password_query() {
        assert_eq!(
            super::redact_secrets("https://alice:s3cret@example.test/x"),
            "https://alice:****@example.test/x"
        );
        let masked = super::redact_secrets("http://example.test/?password=s3cret&q=1");
        assert!(!masked.contains("s3cret"), "{masked}");
        assert!(masked.contains("password=****"), "{masked}");
    }

    #[test]
    fn model_facing_source_truncates_long_text() {
        let long = "x".repeat(5000);
        let compact = super::model_facing_source(json!({
            "text": long,
            "digest": "abc"
        }));
        assert_eq!(compact["text_truncated"], true);
        assert_eq!(compact["text"].as_str().unwrap().chars().count(), 4096);
        assert_eq!(compact["full_text"], "artifact");
        assert_eq!(compact["digest"], "abc");
    }
}
