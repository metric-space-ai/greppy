use crate::policy::{decide_url, NetworkProfile, SharedProfile, UrlDecision};
use crate::policy_proxy::PolicyProxy;
use crate::protocol::{
    read_message, timeout_ms_from_json, write_message, Message, WorkerKind, MAX_FRAME_BYTES,
};
use crate::web_api_shims::shim_source;
use crate::worker::require_worker_auth;
use dpi::PhysicalSize;
use serde_json::json;
use servo::{
    ConsoleLogLevel, CreateNewWebViewRequest, DevicePoint, EmbedderControl, EventLoopWaker,
    InputEvent, InputEventId, InputEventResult, JSValue, LoadStatus, MouseButton,
    MouseButtonAction, MouseButtonEvent, MouseMoveEvent, Preferences, RenderingContext, RgbaImage,
    Servo, ServoBuilder, SimpleDialog, SoftwareRenderingContext, TouchEvent, TouchEventType,
    TouchId, TouchPointerType,
    UserContentManager, UserScript, WebResourceLoad, WebResourceResponse, WebView, WebViewBuilder,
    WebViewDelegate, WebViewPoint, WheelDelta, WheelEvent, WheelMode,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use url::Url;

const ACTION_TIMEOUT: Duration = Duration::from_secs(30);
const KEYBOARD_RUNTIME: &str = include_str!("../js/keyboard-runtime.js");
const WAIT_FOR_FUNCTION_RUNTIME: &str = include_str!("../js/wait-for-function-runtime.js");

struct SlowOp<'a> {
    method: &'a str,
    started: Instant,
}

impl Drop for SlowOp<'_> {
    fn drop(&mut self) {
        let ms = self.started.elapsed().as_millis();
        if ms >= 200 {
            if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: slow-op {} {ms}ms", self.method); } }
        }
    }
}

fn confine_worker_path(path: &Path) -> io::Result<PathBuf> {
    let root = std::env::temp_dir()
        .canonicalize()
        .unwrap_or_else(|_| std::env::temp_dir());
    let requested = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::temp_dir().join(path)
    };
    let mut prefix = requested.as_path();
    let mut suffix = Vec::new();
    while !prefix.exists() {
        match prefix.file_name() {
            Some(name) => {
                suffix.push(name.to_os_string());
                prefix = prefix.parent().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("path outside worker temp: {}", requested.display()),
                    )
                })?;
            }
            None => break,
        }
    }
    let mut out = prefix
        .canonicalize()
        .unwrap_or_else(|_| prefix.to_path_buf());
    for name in suffix.into_iter().rev() {
        if name == Component::ParentDir.as_os_str() || name == ".." {
            out.pop();
            continue;
        }
        if name == Component::CurDir.as_os_str() || name == "." {
            continue;
        }
        out.push(name);
    }
    if !out.starts_with(&root) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("path outside worker temp: {}", out.display()),
        ));
    }
    Ok(out)
}

fn call_timeout(params: &serde_json::Value) -> Duration {
    Duration::from_millis(timeout_ms_from_json(
        params.get("timeout"),
        ACTION_TIMEOUT.as_millis() as u64,
        20,
        120_000,
    ))
}

#[derive(Clone)]
struct WakeFlag {
    state: Arc<(Mutex<WakeInner>, Condvar)>,
}

struct WakeInner {
    generation: u64,
    consumed: u64,
}

impl WakeFlag {
    fn new() -> Self {
        Self {
            state: Arc::new((
                Mutex::new(WakeInner {
                    generation: 1,
                    consumed: 0,
                }),
                Condvar::new(),
            )),
        }
    }

    fn generation(&self) -> u64 {
        self.lock().generation
    }

    fn wait_for_generation(&self, last: u64, timeout: Duration) -> bool {
        let (lock, cvar) = &*self.state;
        let mut inner = lock.lock().unwrap_or_else(|error| error.into_inner());
        if inner.generation != last {
            return true;
        }
        if timeout.is_zero() {
            return false;
        }
        let started = Instant::now();
        while inner.generation == last {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return false;
            }
            let (guard, timed_out) = match cvar.wait_timeout(inner, remaining) {
                Ok(pair) => pair,
                Err(error) => error.into_inner(),
            };
            inner = guard;
            if timed_out.timed_out() && inner.generation == last {
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    fn notify_without_generation(&self) {
        self.state.1.notify_all();
    }

    fn take_pending(&self) -> bool {
        let mut inner = self.lock();
        if inner.generation == inner.consumed {
            return false;
        }
        inner.consumed = inner.generation;
        true
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, WakeInner> {
        self.state
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

impl EventLoopWaker for WakeFlag {
    fn clone_box(&self) -> Box<dyn EventLoopWaker> {
        Box::new(self.clone())
    }

    fn wake(&self) {
        {
            let mut inner = self.lock();
            inner.generation = inner.generation.wrapping_add(1);
        }
        self.state.1.notify_all();
    }
}

fn parse_wait_done_signal(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("__greppyWaitDone:")?;
    let (nonce, status) = rest.split_once(':')?;
    if nonce.len() != 32
        || !nonce
            .as_bytes()
            .iter()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return None;
    }
    if !matches!(status, "ok" | "timeout" | "error") {
        return None;
    }
    if rest.len() != nonce.len() + 1 + status.len() {
        return None;
    }
    Some((nonce, status))
}

fn alloc_wait_nonce() -> io::Result<String> {
    let mut rnd = [0_u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut rnd)?;
    if rnd.iter().all(|byte| *byte == 0) {
        return Err(io::Error::other("wait nonce entropy was all zeros"));
    }
    Ok(rnd.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WakePoll {
    Ready,
    NeedSpin { from: u64, to: u64 },
    TimedOut,
}
const ANIMATION_FRAME_BUDGET: Duration = Duration::from_millis(16);

fn animation_frame_budget(remaining: Duration) -> Duration {
    remaining.min(ANIMATION_FRAME_BUDGET)
}

fn poll_wake_step(
    wake: &WakeFlag,
    mut predicate: impl FnMut() -> bool,
    timeout: Duration,
) -> WakePoll {
    let observed = wake.generation();
    if predicate() {
        return WakePoll::Ready;
    }
    if wake.wait_for_generation(observed, timeout) {
        return WakePoll::NeedSpin {
            from: observed,
            to: wake.generation(),
        };
    }
    if predicate() {
        WakePoll::Ready
    } else {
        WakePoll::TimedOut
    }
}

fn recorded_url_contains(rec: &serde_json::Value, needle: &str) -> bool {
    rec.get("url")
        .and_then(|url| url.as_str())
        .map(|url| url.contains(needle))
        .unwrap_or(false)
}

/// Event-driven recorded-wait. `ready` must not consume; `take` consumes at
/// most once per successful return. Putting `take()` inside `poll_wake_step`
/// drops the value when the predicate is `take().is_some()`.
fn wait_for_recorded_loop<T>(
    wake: &WakeFlag,
    timeout: Duration,
    timeout_label: &str,
    mut before_wait: impl FnMut() -> io::Result<()>,
    mut pump_animating: impl FnMut(Duration) -> bool,
    mut on_need_spin: impl FnMut(),
    mut ready: impl FnMut() -> bool,
    mut take: impl FnMut() -> Option<T>,
) -> io::Result<T> {
    let deadline = Instant::now() + timeout.max(Duration::from_millis(20));
    loop {
        before_wait()?;
        if let Some(value) = take() {
            return Ok(value);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                timeout_label.to_owned(),
            ));
        }
        if pump_animating(remaining) {
            continue;
        }
        match poll_wake_step(wake, &mut ready, remaining) {
            WakePoll::Ready => {}
            WakePoll::TimedOut => {
                if let Some(value) = take() {
                    return Ok(value);
                }
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    timeout_label.to_owned(),
                ));
            }
            WakePoll::NeedSpin { .. } => on_need_spin(),
        }
    }
}

struct RouteRule {
    pattern: String,
    action: String,
    body: Vec<u8>,
    status: u16,
    content_type: String,
    continue_headers: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DialogAction {
    Accept,
    Dismiss,
}

struct Delegate {
    new_frame_ready: RefCell<bool>,
    routes: RefCell<Vec<RouteRule>>,
    file_paths: RefCell<Vec<std::path::PathBuf>>,
    requests: RefCell<Vec<serde_json::Value>>,
    downloads: RefCell<Vec<serde_json::Value>>,
    popups: RefCell<Vec<(WebView, WebView)>>,
    opener_id: RefCell<Option<String>>,
    last_dialogs: RefCell<Vec<serde_json::Value>>,
    dialog_action: RefCell<DialogAction>,
    prompt_text: RefCell<Option<String>>,
    init_scripts: RefCell<Vec<String>>,
    viewport: RefCell<(u32, u32)>,
    extra_headers: RefCell<Vec<(String, String)>>,
    last_console: RefCell<Vec<serde_json::Value>>,
    profile: SharedProfile,
    denied_navigation: RefCell<Option<String>>,
    last_file_choosers: RefCell<Vec<serde_json::Value>>,
    last_responses: RefCell<Vec<serde_json::Value>>,
    rendering_context: Rc<dyn RenderingContext>,
    wait_notices: RefCell<HashMap<String, String>>,
    /// Receipts for synthetic input: Servo acknowledges every input event,
    /// and a painter-side drop (no display list yet -> empty hit test)
    /// arrives as InputEventResult::DispatchFailed. Confirmed delivery
    /// (finding 034) retries on that instead of losing clicks silently.
    input_receipts: RefCell<HashMap<InputEventId, InputEventResult>>,
    wake: WakeFlag,
    /// Auxiliary webviews (popups) do not inherit the parent's user content
    /// manager; without threading it through here a popup would miss the Web
    /// API shims and die on the very ReferenceError they exist to prevent.
    user_content: Rc<UserContentManager>,
}

impl Delegate {
    fn new(
        rendering_context: Rc<dyn RenderingContext>,
        profile: SharedProfile,
        wake: WakeFlag,
        user_content: Rc<UserContentManager>,
    ) -> Self {
        Self {
            new_frame_ready: RefCell::new(false),
            routes: RefCell::new(Vec::new()),
            file_paths: RefCell::new(Vec::new()),
            requests: RefCell::new(Vec::new()),
            downloads: RefCell::new(Vec::new()),
            popups: RefCell::new(Vec::new()),
            last_dialogs: RefCell::new(Vec::new()),
            dialog_action: RefCell::new(DialogAction::Accept),
            prompt_text: RefCell::new(None),
            init_scripts: RefCell::new(Vec::new()),
            viewport: RefCell::new((1280, 720)),
            extra_headers: RefCell::new(Vec::new()),
            last_console: RefCell::new(Vec::new()),
            profile,
            denied_navigation: RefCell::new(None),
            last_file_choosers: RefCell::new(Vec::new()),
            last_responses: RefCell::new(Vec::new()),
            opener_id: RefCell::new(None),
            rendering_context,
            wait_notices: RefCell::new(HashMap::new()),
            input_receipts: RefCell::new(HashMap::new()),
            wake,
            user_content,
        }
    }

    fn mark_request_failure(&self, url: &str, error_text: &str) {
        if let Some(row) = self
            .requests
            .borrow_mut()
            .iter_mut()
            .rev()
            .find(|row| row.get("url").and_then(|value| value.as_str()) == Some(url))
        {
            row["failure"] = json!({ "errorText": error_text });
            self.wake.wake();
        }
    }

    fn note_wait_signal(&self, text: &str) {
        let Some((token, status)) = parse_wait_done_signal(text) else {
            return;
        };
        self.wait_notices
            .borrow_mut()
            .entry(token.to_owned())
            .or_insert_with(|| status.to_owned());
    }

    fn wait_notice(&self, token: &str) -> Option<String> {
        self.wait_notices.borrow().get(token).cloned()
    }

    fn clear_wait_notice(&self, token: &str) {
        self.wait_notices.borrow_mut().remove(token);
    }
}

impl WebViewDelegate for Delegate {
    fn notify_input_event_handled(
        &self,
        _webview: WebView,
        event_id: InputEventId,
        result: InputEventResult,
    ) {
        self.input_receipts
            .borrow_mut()
            .insert(event_id, result);
        self.wake.wake();
    }

    fn notify_new_frame_ready(&self, webview: WebView) {
        *self.new_frame_ready.borrow_mut() = true;
        webview.paint();
        self.wake.wake();
    }

    fn notify_animating_changed(&self, _webview: WebView, animating: bool) {
        if animating {
            self.wake.wake();
        }
    }

    fn show_console_message(&self, _webview: WebView, level: ConsoleLogLevel, message: String) {
        self.note_wait_signal(&message);
        if message.starts_with("__greppyWaitDone:") {
            self.wake.wake();
            return;
        }
        let kind = match level {
            ConsoleLogLevel::Log => "log",
            ConsoleLogLevel::Debug => "debug",
            ConsoleLogLevel::Info => "info",
            ConsoleLogLevel::Warn => "warning",
            ConsoleLogLevel::Error => "error",
            ConsoleLogLevel::Trace => "trace",
            ConsoleLogLevel::Dir => "dir",
        };
        self.last_console.borrow_mut().push(json!({
            "type": kind,
            "text": message,
        }));
        self.wake.wake();
    }

    fn request_create_new(&self, parent: WebView, request: CreateNewWebViewRequest) {
        let child = request
            .builder(Rc::clone(&self.rendering_context))
            .delegate(Rc::new(Delegate::new(
                Rc::clone(&self.rendering_context),
                self.profile.clone(),
                self.wake.clone(),
                Rc::clone(&self.user_content),
            )))
            .user_content_manager(Rc::clone(&self.user_content))
            .build();
        child.show();
        self.popups.borrow_mut().push((child, parent));
    }

    fn show_embedder_control(&self, _webview: WebView, embedder_control: EmbedderControl) {
        match embedder_control {
            EmbedderControl::FilePicker(mut picker) => {
                self.last_file_choosers.borrow_mut().push(json!({
                    "multiple": picker.allow_select_multiple(),
                }));
                let paths = self.file_paths.borrow().clone();
                self.wake.wake();
                if paths.is_empty() {
                    picker.dismiss();
                } else {
                    picker.select(&paths);
                    picker.submit();
                }
            }
            EmbedderControl::SimpleDialog(dialog) => {
                let kind = match &dialog {
                    SimpleDialog::Alert(_) => "alert",
                    SimpleDialog::Confirm(_) => "confirm",
                    SimpleDialog::Prompt(_) => "prompt",
                };
                let default_value = match &dialog {
                    SimpleDialog::Prompt(prompt_dialog) => prompt_dialog.current_value().to_owned(),
                    _ => String::new(),
                };
                let action = *self.dialog_action.borrow();
                let prompt = self.prompt_text.borrow().clone();
                self.last_dialogs.borrow_mut().push(json!({
                    "type": kind,
                    "message": dialog.message(),
                    "defaultValue": default_value,
                    "action": match action {
                        DialogAction::Accept => "accept",
                        DialogAction::Dismiss => "dismiss",
                    },
                }));
                match (action, dialog) {
                    (DialogAction::Accept, SimpleDialog::Prompt(mut prompt_dialog)) => {
                        if let Some(text) = prompt {
                            prompt_dialog.set_current_value(&text);
                        }
                        prompt_dialog.confirm();
                    }
                    (DialogAction::Accept, other) => other.confirm(),
                    (DialogAction::Dismiss, dialog) => dialog.dismiss(),
                }
            }
            other => drop(other),
        }
    }

    fn load_web_resource(&self, _webview: WebView, load: WebResourceLoad) {
        let url = load.request.url.to_string();
        let mut headers: Vec<serde_json::Value> = load
            .request
            .headers
            .iter()
            .filter_map(|(name, value)| {
                value.to_str().ok().map(|value| {
                    json!({
                        "name": name.as_str(),
                        "value": value,
                    })
                })
            })
            .collect();
        for (name, value) in self.extra_headers.borrow().iter() {
            let extras = extra_request_headers(&[(name.clone(), value.clone())]);
            if extras.is_empty() {
                continue;
            }
            headers.retain(|entry| {
                entry
                    .get("name")
                    .and_then(|n| n.as_str())
                    .is_none_or(|n| !n.eq_ignore_ascii_case(name))
            });
            headers.push(json!({ "name": name, "value": value }));
        }
        let abort_match = self
            .routes
            .borrow()
            .iter()
            .any(|rule| rule.action == "abort" && pattern_matches(&rule.pattern, &url));
        let policy = decide_url(self.profile.get(), &url);
        let failure = match &policy {
            UrlDecision::Deny { reason } => Some(format!("policy_denied: {reason}")),
            UrlDecision::Allow if abort_match => Some("net::ERR_FAILED".to_owned()),
            UrlDecision::Allow => None,
        };
        self.requests.borrow_mut().push(json!({
            "url": url,
            "method": load.request.method.to_string(),
            "main_frame": load.request.is_for_main_frame,
            "redirect": load.request.is_redirect,
            "headers": headers,
            "failure": failure.as_ref().map(|error_text| json!({ "errorText": error_text })),
        }));
        self.wake.wake();
        if let UrlDecision::Deny { reason } = policy {
            if load.request.is_for_main_frame {
                *self.denied_navigation.borrow_mut() = Some(reason.to_owned());
            }
            let denied_url = load.request.url.clone();
            load.intercept(WebResourceResponse::new(denied_url))
                .cancel();
            return;
        }
        let matched = self
            .routes
            .borrow()
            .iter()
            .find(|rule| pattern_matches(&rule.pattern, &url))
            .map(|rule| {
                (
                    rule.action.clone(),
                    rule.body.clone(),
                    rule.status,
                    rule.content_type.clone(),
                    rule.continue_headers.clone(),
                )
            });
        let Some((action, body, status, content_type, continue_headers)) = matched else {
            let extras = extra_request_headers(&self.extra_headers.borrow());
            if !extras.is_empty() {
                load.continue_with_headers(extras);
            }
            return;
        };
        let request_url = load.request.url.clone();
        match action.as_str() {
            "abort" => {
                self.mark_request_failure(&url, "net::ERR_FAILED");
                if load.request.is_for_main_frame {
                    *self.denied_navigation.borrow_mut() = Some("net::ERR_FAILED".to_owned());
                }
                load.intercept(WebResourceResponse::new(request_url))
                    .cancel();
            }
            "fulfill" => {
                let status_code =
                    http::StatusCode::from_u16(status).unwrap_or(http::StatusCode::OK);
                let mut header_map = http::HeaderMap::new();
                if let Ok(value) = http::HeaderValue::from_str(&content_type) {
                    header_map.insert(http::header::CONTENT_TYPE, value);
                }
                let mut intercepted = load.intercept(
                    WebResourceResponse::new(request_url.clone())
                        .status_code(status_code)
                        .headers(header_map),
                );
                if !body.is_empty() {
                    intercepted.send_body_data(body.clone());
                }
                intercepted.finish();
                let status_text = status_code.canonical_reason().unwrap_or("").to_owned();
                let body_b64 = base64_encode(&body);
                self.last_responses.borrow_mut().push(json!({
                    "url": request_url.to_string(),
                    "status": status,
                    "statusText": status_text,
                    "ok": status < 400,
                    "bodyBase64": body_b64,
                    "byteLength": body.len(),
                    "headers": {
                        "content-type": content_type,
                    },
                }));
                self.wake.wake();
                let lower = content_type.to_ascii_lowercase();
                let is_download = lower.contains("octet-stream") || lower.contains("attachment");
                if is_download {
                    let suggested = request_url
                        .path_segments()
                        .and_then(|segments| segments.last())
                        .filter(|name| !name.is_empty())
                        .unwrap_or("download")
                        .to_owned();
                    self.downloads.borrow_mut().push(json!({
                        "url": request_url.to_string(),
                        "byteLength": body.len(),
                        "bodyBase64": body_b64,
                        "suggestedFilename": suggested,
                        "contentType": content_type,
                    }));
                    self.wake.wake();
                }
            }
            "continue" => {
                let mut merged = self.extra_headers.borrow().clone();
                merged.extend(continue_headers);
                let extras = extra_request_headers(&merged);
                if extras.is_empty() {
                    return;
                }
                load.continue_with_headers(extras);
            }
            _ => {}
        }
    }
}

fn extra_request_headers(extras: &[(String, String)]) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    for (name, value) in extras {
        let lower = name.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "host"
                | "content-length"
                | "transfer-encoding"
                | "content-encoding"
                | "connection"
                | "proxy-connection"
                | "proxy-authorization"
                | "keep-alive"
                | "te"
                | "trailer"
                | "upgrade"
        ) {
            continue;
        }
        let Ok(name) = http::HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let Ok(value) = http::HeaderValue::from_str(value) else {
            continue;
        };
        headers.insert(name, value);
    }
    headers
}

fn pattern_matches(pattern: &str, url: &str) -> bool {
    if pattern == "**/*" || pattern == "*" {
        return true;
    }
    if let Some(rest) = pattern.strip_prefix("**/") {
        return url.contains(rest.trim_end_matches('*'));
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return url.starts_with(prefix);
    }
    url == pattern || url.contains(pattern)
}

enum ObjectLife {
    Live {
        generation: u64,
        parent: Option<String>,
    },
    Disposed {
        generation: u64,
    },
}

impl ObjectLife {
    fn disposed(generation: u64) -> Self {
        Self::Disposed { generation }
    }
}

enum PageSlot {
    Live {
        pair: (WebView, Rc<Delegate>),
        generation: u64,
        context_id: Option<String>,
        browser_id: Option<String>,
    },
    Disposed {
        generation: u64,
    },
}

impl PageSlot {
    fn live(
        webview: WebView,
        delegate: Rc<Delegate>,
        context_id: Option<String>,
        browser_id: Option<String>,
    ) -> Self {
        Self::Live {
            pair: (webview, delegate),
            generation: 1,
            context_id,
            browser_id,
        }
    }
}

fn object_disposed(kind: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("object_disposed: {kind} has been closed"),
    )
}

fn reject_generation(kind: &str, stored: u64, wanted: Option<u64>, live: bool) -> io::Result<()> {
    if wanted.is_some_and(|value| value != stored) || !live {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("object_disposed: {kind} has been closed (generation {stored})"),
        ));
    }
    Ok(())
}

fn alloc_pump_token(nonce: &Cell<u64>) -> String {
    let n = nonce.get();
    nonce.set(n.wrapping_add(1));
    format!("__greppyPump{n}")
}

fn track_pump_token(pending: &RefCell<Vec<String>>, nonce: &Cell<u64>) -> String {
    let token = alloc_pump_token(nonce);
    pending.borrow_mut().push(token.clone());
    token
}

#[allow(dead_code)]
fn reclaim_tracked_token(pending: &RefCell<Vec<String>>, token: &str) {
    pending.borrow_mut().retain(|name| name != token);
}

fn reclaim_all_tracked_tokens(pending: &RefCell<Vec<String>>) {
    pending.borrow_mut().clear();
}

enum WaitOutcome {
    Ok(JSValue),
    Timeout,
    Error(String),
}

fn wait_for_function_waiter_script(
    source: &str,
    token: &str,
    timeout_ms: u128,
    frame_index: Option<u64>,
) -> io::Result<String> {
    let source_js = serde_json::to_string(source).map_err(io::Error::other)?;
    let token_js = serde_json::to_string(token).map_err(io::Error::other)?;
    let mut waiter = String::from("(function(source, token, timeoutMs) { ");
    waiter.push_str(WAIT_FOR_FUNCTION_RUNTIME);
    waiter.push_str("\nreturn greppyWaitForFunction(source, token, timeoutMs); })(");
    waiter.push_str(&source_js);
    waiter.push_str(", ");
    waiter.push_str(&token_js);
    waiter.push_str(", ");
    waiter.push_str(&timeout_ms.to_string());
    waiter.push_str(")");
    if let Some(index) = frame_index {
        let waiter_js = serde_json::to_string(&waiter).map_err(io::Error::other)?;
        let mut wrapped = String::from(
            "(function(index, waiter) { var frame = document.querySelectorAll('iframe')[index]; if (!frame) throw new Error('no frame'); return frame.contentWindow.eval(waiter); })(",
        );
        wrapped.push_str(&index.to_string());
        wrapped.push_str(", ");
        wrapped.push_str(&waiter_js);
        wrapped.push_str(")");
        Ok(wrapped)
    } else {
        Ok(waiter)
    }
}

struct ContentEngine {
    servo: Servo,
    rendering_context: Rc<dyn RenderingContext>,
    pages: HashMap<String, PageSlot>,
    browsers: HashMap<String, ObjectLife>,
    contexts: HashMap<String, ObjectLife>,
    next_id: u64,
    pump_nonce: Cell<u64>,
    pump_pending: RefCell<Vec<String>>,
    parent_alive: Arc<AtomicBool>,
    wake: WakeFlag,
    profile: SharedProfile,
    /// Carries the Web API shims. Shared by every page, so a shim reaches
    /// frames and popups too, not just the tab the agent drove.
    user_content: Rc<UserContentManager>,
    _proxy: PolicyProxy,
}

impl ContentEngine {
    fn new(parent_alive: Arc<AtomicBool>) -> io::Result<Self> {
        let rendering_context = Rc::new(
            SoftwareRenderingContext::new(PhysicalSize {
                width: 1280,
                height: 720,
            })
            .map_err(|error| io::Error::other(format!("software renderer failed: {error:?}")))?,
        );
        rendering_context.make_current().map_err(|error| {
            io::Error::other(format!("renderer make_current failed: {error:?}"))
        })?;

        let profile = SharedProfile::new(NetworkProfile::Research);
        let proxy = PolicyProxy::spawn(profile.clone())?;
        let preferences = engine_preferences(&proxy.uri());
        let wake = WakeFlag::new();
        let servo = ServoBuilder::default()
            .preferences(preferences)
            .event_loop_waker(Box::new(wake.clone()))
            .build();
        let user_content = Rc::new(UserContentManager::new(&servo));
        user_content.add_script(Rc::new(UserScript::new(shim_source().to_owned(), None)));
        Ok(Self {
            servo,
            rendering_context,
            pages: HashMap::new(),
            browsers: HashMap::new(),
            contexts: HashMap::new(),
            next_id: 1,
            pump_nonce: Cell::new(1),
            pump_pending: RefCell::new(Vec::new()),
            parent_alive,
            wake,
            profile,
            user_content,
            _proxy: proxy,
        })
    }

    fn parent_dead(&self) -> bool {
        !self.parent_alive.load(Ordering::Relaxed)
    }

    fn parent_gone() -> io::Error {
        io::Error::new(
            io::ErrorKind::BrokenPipe,
            "supervisor closed content worker stdin",
        )
    }

    fn alloc_id(&mut self, prefix: &str) -> String {
        let id = format!("{prefix}-{}-{}", std::process::id(), self.next_id);
        self.next_id += 1;
        id
    }

    fn spin_until(
        &self,
        timeout: Duration,
        mut predicate: impl FnMut() -> bool,
    ) -> io::Result<bool> {
        let deadline = Instant::now() + timeout;
        loop {
            if self.parent_dead() {
                return Err(Self::parent_gone());
            }
            if predicate() {
                return Ok(true);
            }
            // Load-status and WebResourceRequested are event-loop messages.
            // Continue-with-headers waits on that same loop; a missed waker
            // must not sit on the Condvar until ACTION_TIMEOUT with status=Started.
            self.servo.spin_event_loop();
            if predicate() {
                return Ok(true);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            match poll_wake_step(
                &self.wake,
                &mut predicate,
                remaining.min(Duration::from_millis(10)),
            ) {
                WakePoll::Ready => return Ok(true),
                WakePoll::TimedOut => {
                    if Instant::now() >= deadline {
                        return Ok(false);
                    }
                }
                WakePoll::NeedSpin { .. } => {
                    self.servo.spin_event_loop();
                }
            }
        }
    }

    /// How far a navigation must get before `goto` returns.
    ///
    /// Playwright lets the caller choose; the runtime used to wait for the
    /// full load in every case. On a page whose sub-resource never finishes,
    /// `readyState` stays `loading` forever, so `goto` timed out on a document
    /// that was parsed, titled and fully readable -- three pages of the pinned
    /// corpus fail exactly this way.
    fn load_committed_for(
        &self,
        webview: &WebView,
        last_js: &mut Instant,
        until: WaitUntil,
    ) -> bool {
        match (webview.load_status(), until) {
            (LoadStatus::Complete, _) => true,
            // The document is parsed: the DOM is there and can be read, which
            // is exactly what `domcontentloaded` promises.
            (LoadStatus::HeadParsed, WaitUntil::DomContentLoaded) => true,
            _ => self.load_committed(webview, last_js),
        }
    }

    fn load_committed(&self, webview: &WebView, last_js: &mut Instant) -> bool {
        match webview.load_status() {
            LoadStatus::Complete => true,
            // Poll readyState at 25ms, not 200ms. Large documents sit in
            // HeadParsed for their whole parse; on the release build the
            // 200ms cadence alone cost ~1.3s of a 2.1s navigation commit
            // (nav-trace, page 044) while each evaluate costs well under a
            // millisecond of CPU.
            LoadStatus::HeadParsed if last_js.elapsed() >= Duration::from_millis(25) => {
                *last_js = Instant::now();
                match self.evaluate_until(
                    webview.clone(),
                    "document.readyState",
                    Duration::from_millis(150),
                ) {
                    Ok(JSValue::String(state)) => {
                        load_status_allows_navigation(LoadStatus::HeadParsed, Some(&state))
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn spin_until_loaded(
        &self,
        webview: &WebView,
        timeout: Duration,
        url_settled: impl FnMut() -> bool,
    ) -> io::Result<bool> {
        self.spin_until_loaded_until(webview, timeout, WaitUntil::Load, url_settled)
    }

    fn spin_until_loaded_until(
        &self,
        webview: &WebView,
        timeout: Duration,
        until: WaitUntil,
        mut url_settled: impl FnMut() -> bool,
    ) -> io::Result<bool> {
        let deadline = Instant::now() + timeout;
        let mut last_js = Instant::now() - Duration::from_millis(200);
        let mut trace = NavTrace::begin();
        loop {
            if self.parent_dead() {
                return Err(Self::parent_gone());
            }
            trace.note(webview, &mut url_settled);
            if url_settled() && self.load_committed_for(webview, &mut last_js, until) {
                trace.finish(webview);
                return Ok(true);
            }
            self.servo.spin_event_loop();
            trace.note(webview, &mut url_settled);
            if url_settled() && self.load_committed_for(webview, &mut last_js, until) {
                trace.finish(webview);
                return Ok(true);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            match poll_wake_step(
                &self.wake,
                &mut || false,
                remaining.min(Duration::from_millis(10)),
            ) {
                WakePoll::Ready | WakePoll::TimedOut => {
                    if Instant::now() >= deadline {
                        return Ok(false);
                    }
                }
                WakePoll::NeedSpin { .. } => {
                    self.servo.spin_event_loop();
                }
            }
        }
    }

    fn next_pump_token(&self) -> String {
        track_pump_token(&self.pump_pending, &self.pump_nonce)
    }

    fn live_webviews(&self) -> Vec<WebView> {
        self.pages
            .values()
            .filter_map(|slot| match slot {
                PageSlot::Live { pair, .. } => Some(pair.0.clone()),
                PageSlot::Disposed { .. } => None,
            })
            .collect()
    }

    fn sweep_pump_namespace(&self, webview: &WebView) -> bool {
        let script = r#"(function() {
          var names = Object.getOwnPropertyNames(window);
          for (var i = 0; i < names.length; i++) {
            var k = names[i];
            if (k.indexOf('__greppyPump') === 0) {
              try { delete window[k]; } catch (_err) {}
            }
          }
          var left = 0;
          names = Object.getOwnPropertyNames(window);
          for (var j = 0; j < names.length; j++) {
            if (names[j].indexOf('__greppyPump') === 0) left++;
          }
          return left;
        })()"#;
        matches!(
            self.evaluate_until(webview.clone(), script, Duration::from_millis(80)),
            Ok(JSValue::Number(value)) if value == 0.0
        )
    }

    fn reclaim_pump_tokens(&self) {
        if self.pump_pending.borrow().is_empty() {
            return;
        }
        let views = self.live_webviews();
        if views.is_empty() {
            reclaim_all_tracked_tokens(&self.pump_pending);
            return;
        }
        let mut all_clear = true;
        for webview in views {
            if !self.sweep_pump_namespace(&webview) {
                all_clear = false;
            }
        }
        if all_clear {
            reclaim_all_tracked_tokens(&self.pump_pending);
        }
    }

    fn settle_pump_tokens(&self, webview: &WebView) {
        if self.sweep_pump_namespace(webview) {
            reclaim_all_tracked_tokens(&self.pump_pending);
        }
    }

    fn pump_servo(&self, webview: &WebView, min: Duration, deadline: Instant) -> bool {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining < Duration::from_millis(4) {
            return false;
        }
        let wait = min.min(remaining);
        let ms = wait.as_millis().max(1);
        self.pump_servo_with_token(webview, &self.next_pump_token(), ms, wait, deadline)
    }

    fn pump_servo_with_token(
        &self,
        webview: &WebView,
        token: &str,
        ms: u128,
        wait: Duration,
        deadline: Instant,
    ) -> bool {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let schedule = format!(
            "(function() {{ window['{token}'] = 0; setTimeout(function() {{ window['{token}'] = 1; }}, {ms}); return 0; }})()"
        );
        let schedule_budget = remaining.min(Duration::from_millis(200));
        if self
            .evaluate_until(webview.clone(), &schedule, schedule_budget)
            .is_err()
        {
            return false;
        }
        let poll = format!("window['{token}']");
        let min_until = Instant::now() + wait;
        let mut next_poll = min_until;
        while Instant::now() < deadline {
            if self.parent_dead() {
                return false;
            }
            webview.paint();
            self.servo.spin_event_loop();
            if Instant::now() >= next_poll {
                let poll_budget = deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(80));
                if poll_budget.is_zero() {
                    return false;
                }
                match self.evaluate_until(webview.clone(), &poll, poll_budget) {
                    Ok(JSValue::Number(value)) if value >= 1.0 => return true,
                    Err(_) => return false,
                    _ => {}
                }
                next_poll = Instant::now() + Duration::from_millis(8);
            }
            thread::sleep(Duration::from_millis(2));
        }
        false
    }

    fn page(&self, page_id: &str) -> io::Result<&(WebView, Rc<Delegate>)> {
        match self.pages.get(page_id) {
            Some(PageSlot::Live { pair, .. }) => Ok(pair),
            Some(PageSlot::Disposed { generation }) => Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("object_disposed: Page has been closed (generation {generation})"),
            )),
            None => Err(object_disposed("Page")),
        }
    }

    fn dispose_page(&mut self, page_id: &str) {
        if let Some(PageSlot::Live { generation, .. }) = self.pages.get(page_id) {
            let generation = *generation;
            self.pages
                .insert(page_id.to_owned(), PageSlot::Disposed { generation });
        }
    }

    fn dispose_all_pages(&mut self) {
        let live: Vec<(String, u64)> = self
            .pages
            .iter()
            .filter_map(|(id, slot)| match slot {
                PageSlot::Live { generation, .. } => Some((id.clone(), *generation)),
                PageSlot::Disposed { .. } => None,
            })
            .collect();
        for (id, generation) in live {
            self.pages.insert(id, PageSlot::Disposed { generation });
        }
    }

    fn dispose_pages_owned_by_context(&mut self, context_id: &str) {
        let live: Vec<(String, u64)> = self
            .pages
            .iter()
            .filter_map(|(id, slot)| match slot {
                PageSlot::Live {
                    generation,
                    context_id: Some(owner),
                    ..
                } if owner == context_id => Some((id.clone(), *generation)),
                _ => None,
            })
            .collect();
        for (id, generation) in live {
            self.pages.insert(id, PageSlot::Disposed { generation });
        }
    }

    fn dispose_pages_owned_by_browser(&mut self, browser_id: &str) {
        let live: Vec<(String, u64)> = self
            .pages
            .iter()
            .filter_map(|(id, slot)| match slot {
                PageSlot::Live {
                    generation,
                    browser_id: Some(owner),
                    ..
                } if owner == browser_id => Some((id.clone(), *generation)),
                _ => None,
            })
            .collect();
        for (id, generation) in live {
            self.pages.insert(id, PageSlot::Disposed { generation });
        }
    }

    fn reject_stale_objects(&self, method: &str, params: &serde_json::Value) -> io::Result<()> {
        if matches!(
            method,
            "chromium.launch" | "page.close" | "page.isClosed" | "context.close" | "browser.close"
        ) {
            return Ok(());
        }
        if let Some(page_id) = params.get("page").and_then(|value| value.as_str()) {
            let wanted = params.get("generation").and_then(|value| value.as_u64());
            match self.pages.get(page_id) {
                Some(PageSlot::Live { generation, .. }) => {
                    reject_generation("Page", *generation, wanted, true)?;
                }
                Some(PageSlot::Disposed { generation }) => {
                    reject_generation("Page", *generation, wanted, false)?;
                }
                None => return Err(object_disposed("Page")),
            }
        }
        if let Some(context_id) = params.get("context").and_then(|value| value.as_str()) {
            let wanted = params.get("generation").and_then(|value| value.as_u64());
            match self.contexts.get(context_id) {
                Some(ObjectLife::Live { generation, .. }) => {
                    reject_generation("BrowserContext", *generation, wanted, true)?;
                }
                Some(ObjectLife::Disposed { generation }) => {
                    reject_generation("BrowserContext", *generation, wanted, false)?;
                }
                None if method.starts_with("context.") => {
                    return Err(object_disposed("BrowserContext"));
                }
                None => {}
            }
        }
        if let Some(browser_id) = params.get("browser").and_then(|value| value.as_str()) {
            let wanted = params.get("generation").and_then(|value| value.as_u64());
            match self.browsers.get(browser_id) {
                Some(ObjectLife::Live { generation, .. }) => {
                    reject_generation("Browser", *generation, wanted, true)?;
                }
                Some(ObjectLife::Disposed { generation }) => {
                    reject_generation("Browser", *generation, wanted, false)?;
                }
                None if method.starts_with("browser.") => {
                    return Err(object_disposed("Browser"));
                }
                None => {}
            }
        }
        Ok(())
    }

    fn evaluate(&self, webview: WebView, script: &str) -> io::Result<JSValue> {
        self.evaluate_until(webview, script, ACTION_TIMEOUT)
    }

    fn evaluate_until(
        &self,
        webview: WebView,
        script: &str,
        timeout: Duration,
    ) -> io::Result<JSValue> {
        let saved = Rc::new(RefCell::new(None));
        let callback_slot = Rc::clone(&saved);
        webview.evaluate_javascript(script, move |result| {
            *callback_slot.borrow_mut() = Some(result);
        });
        let ready = Rc::clone(&saved);
        if !self.spin_until(timeout.max(Duration::from_millis(1)), move || {
            ready.borrow().is_some()
        })? {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out evaluating page JavaScript",
            ));
        }
        let result = saved.borrow_mut().take().expect("evaluation completed");
        result.map_err(|error| io::Error::other(format!("page JavaScript failed: {error:?}")))
    }

    fn wait_for_recorded<T>(
        &self,
        webview: &WebView,
        timeout: Duration,
        timeout_label: &str,
        ready: impl FnMut() -> bool,
        take: impl FnMut() -> Option<T>,
    ) -> io::Result<T> {
        wait_for_recorded_loop(
            &self.wake,
            timeout,
            timeout_label,
            || {
                if self.parent_dead() {
                    Err(Self::parent_gone())
                } else {
                    Ok(())
                }
            },
            |remaining| {
                if !webview.animating() {
                    return false;
                }
                let observed = self.wake.generation();
                let _ = self
                    .wake
                    .wait_for_generation(observed, animation_frame_budget(remaining));
                webview.paint();
                self.servo.spin_event_loop();
                true
            },
            || {
                webview.paint();
                self.servo.spin_event_loop();
            },
            ready,
            take,
        )
    }

    fn wait_for_function_truthy(
        &self,
        webview: WebView,
        delegate: &Delegate,
        source: &str,
        timeout: Duration,
        frame_index: Option<u64>,
    ) -> io::Result<serde_json::Value> {
        let timeout = timeout.max(Duration::from_millis(20));
        let deadline = Instant::now() + timeout;
        let token = alloc_wait_nonce()?;
        delegate.clear_wait_notice(&token);
        let waiter =
            wait_for_function_waiter_script(source, &token, timeout.as_millis(), frame_index)?;
        let install_budget = deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(1));
        let first = match self.evaluate_until(webview.clone(), &waiter, install_budget) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                self.drop_wait_slot(&webview, &token);
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "timeout: waitForFunction",
                ));
            }
            Err(error) => {
                self.drop_wait_slot(&webview, &token);
                return Err(error);
            }
        };
        if let Some(result) = self.finish_if_expected_nonce(&webview, delegate, &token)? {
            return result;
        }
        if jsvalue_is_truthy(&first) {
            self.drop_wait_slot(&webview, &token);
            self.settle_pump_tokens(&webview);
            return serialize_wait_value(first);
        }
        loop {
            if self.parent_dead() {
                self.drop_wait_slot(&webview, &token);
                return Err(Self::parent_gone());
            }
            if let Some(result) = self.finish_if_expected_nonce(&webview, delegate, &token)? {
                return result;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                self.drop_wait_slot(&webview, &token);
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "timeout: waitForFunction",
                ));
            }
            // Servo requires regular event-loop spins while requestAnimationFrame
            // callbacks are pending (`WebView::animating`). That is the rAF clock,
            // not a Rust predicate sample.
            if webview.animating() {
                let observed = self.wake.generation();
                let _ = self
                    .wake
                    .wait_for_generation(observed, animation_frame_budget(remaining));
                webview.paint();
                self.servo.spin_event_loop();
                continue;
            }
            match poll_wake_step(
                &self.wake,
                || delegate.wait_notice(&token).is_some(),
                remaining,
            ) {
                WakePoll::Ready => {}
                WakePoll::TimedOut => {
                    if let Some(result) =
                        self.finish_if_expected_nonce(&webview, delegate, &token)?
                    {
                        return result;
                    }
                    self.drop_wait_slot(&webview, &token);
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "timeout: waitForFunction",
                    ));
                }
                WakePoll::NeedSpin { .. } => {
                    webview.paint();
                    self.servo.spin_event_loop();
                }
            }
        }
    }

    fn wait_slot_key(token: &str) -> String {
        format!("__greppyWait_{token}")
    }

    fn drop_wait_slot(&self, webview: &WebView, token: &str) {
        let Ok(key_js) = serde_json::to_string(&Self::wait_slot_key(token)) else {
            return;
        };
        let mut script = String::from("(function(key) { var slot = window[key]; if (slot && typeof slot.cleanup === 'function') { try { slot.cleanup(); } catch (_e) {} } try { delete window[key]; } catch (_e) {} return 0; })(");
        script.push_str(&key_js);
        script.push_str(")");
        let _ = self.evaluate_until(webview.clone(), &script, Duration::from_millis(80));
    }

    fn take_completed_wait_slot(
        &self,
        webview: &WebView,
        token: &str,
    ) -> io::Result<Option<(String, JSValue)>> {
        let key_js =
            serde_json::to_string(&Self::wait_slot_key(token)).map_err(io::Error::other)?;
        let mut script = String::from("(function(key) { var slot = window[key]; if (!slot || !slot.done) return [0, '', null]; var status = String(slot.status || ''); var value = slot.value; if (typeof slot.cleanup === 'function') { try { slot.cleanup(); } catch (_e) {} } try { delete window[key]; } catch (_e) {} return [1, status, value]; })(");
        script.push_str(&key_js);
        script.push_str(")");
        match self.evaluate_until(webview.clone(), &script, Duration::from_millis(80))? {
            JSValue::Array(mut items) if items.len() >= 3 => {
                let value = items.remove(2);
                let status_value = items.remove(1);
                let done_value = items.remove(0);
                let done = match done_value {
                    JSValue::Number(value) => value != 0.0,
                    JSValue::Boolean(value) => value,
                    _ => false,
                };
                if !done {
                    return Ok(None);
                }
                let status = match status_value {
                    JSValue::String(value) => value,
                    _ => String::new(),
                };
                Ok(Some((status, value)))
            }
            _ => Ok(None),
        }
    }

    fn finish_if_expected_nonce(
        &self,
        webview: &WebView,
        delegate: &Delegate,
        token: &str,
    ) -> io::Result<Option<io::Result<serde_json::Value>>> {
        let Some(notice) = delegate.wait_notice(token) else {
            return Ok(None);
        };
        let Some((status, value)) = self.take_completed_wait_slot(webview, token)? else {
            delegate.clear_wait_notice(token);
            return Ok(None);
        };
        let status = if status.is_empty() { notice } else { status };
        Ok(Some(self.finish_wait_outcome(
            webview,
            match status.as_str() {
                "ok" => WaitOutcome::Ok(value),
                "timeout" => WaitOutcome::Timeout,
                "error" => WaitOutcome::Error(match value {
                    JSValue::String(message) => message,
                    other => format!("{other:?}"),
                }),
                other => WaitOutcome::Error(other.to_owned()),
            },
        )))
    }

    fn finish_wait_outcome(
        &self,
        webview: &WebView,
        outcome: WaitOutcome,
    ) -> io::Result<serde_json::Value> {
        match outcome {
            WaitOutcome::Ok(value) => {
                self.settle_pump_tokens(webview);
                serialize_wait_value(value)
            }
            WaitOutcome::Timeout => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timeout: waitForFunction",
            )),
            WaitOutcome::Error(message) => Err(io::Error::other(message)),
        }
    }

    fn run_init_scripts(&self, page_id: &str) -> io::Result<()> {
        let scripts = self.page(page_id)?.1.init_scripts.borrow().clone();
        if scripts.is_empty() {
            return Ok(());
        }
        let (webview, _) = self.page(page_id)?.clone();
        for source in scripts {
            let _ = self.evaluate(webview.clone(), &source);
        }
        Ok(())
    }

    fn assign_pending_files(&self, page_id: &str, selector: &str) -> io::Result<serde_json::Value> {
        let paths = self.page(page_id)?.1.file_paths.borrow().clone();
        if paths.is_empty() {
            return Ok(json!({ "dom_files": 0, "changed": 0, "skipped": true }));
        }
        let mut payloads = Vec::new();
        for path in &paths {
            let path = confine_worker_path(path)?;
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(json!({
                        "dom_files": 0,
                        "changed": 0,
                        "error": format!("cannot read {}: {error}", path.display()),
                    }));
                }
            };
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("upload.bin")
                .to_owned();
            payloads.push(json!({
                "name": name,
                "type": "application/octet-stream",
                "b64": base64_encode(&bytes),
            }));
        }
        let (webview, _) = self.page(page_id)?.clone();
        let selector_json = serde_json::to_string(&selector).map_err(io::Error::other)?;
        let files_json = serde_json::to_string(&payloads).map_err(io::Error::other)?;
        let script = format!(
            r#"(function(selector, files) {{
  var input = document.querySelector(selector);
  if (!input) return {{ dom_files: 0, changed: 0, error: "no input" }};
  try {{
    if (typeof DataTransfer === "undefined" || typeof File === "undefined") {{
      return {{
        dom_files: 0,
        changed: 0,
        error: "Servo 0.5.0 page JS has no DataTransfer/File constructors; HTMLInputElement.files cannot be assigned without FilePicker"
      }};
    }}
    var dt = new DataTransfer();
    files.forEach(function(file) {{
      var raw = atob(file.b64);
      var buf = new Uint8Array(raw.length);
      for (var i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
      dt.items.add(new File([buf], file.name, {{ type: file.type || "application/octet-stream" }}));
    }});
    input.files = dt.files;
    var changed = 0;
    var onChange = function() {{ changed += 1; }};
    input.addEventListener("input", onChange);
    input.addEventListener("change", onChange);
    input.dispatchEvent(new Event("input", {{ bubbles: true }}));
    input.dispatchEvent(new Event("change", {{ bubbles: true }}));
    input.removeEventListener("input", onChange);
    input.removeEventListener("change", onChange);
    return {{
      dom_files: input.files ? input.files.length : 0,
      changed: changed,
      name: input.files && input.files[0] ? input.files[0].name : ""
    }};
  }} catch (error) {{
    return {{
      dom_files: 0,
      changed: 0,
      error: String(error)
    }};
  }}
}})({selector_json}, {files_json})"#
        );
        match self.evaluate(webview, &script) {
            Ok(value) => Ok(jsvalue_to_json(value)),
            Err(error) => Ok(json!({
                "dom_files": 0,
                "changed": 0,
                "error": error.to_string(),
            })),
        }
    }

    fn locator_eval(&self, params: &serde_json::Value, expr: &str) -> io::Result<JSValue> {
        let page_id = required_str(params, "page")?;
        let selector = params
            .get("selector")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let (webview, _) = self.page(&page_id)?.clone();
        let source = format!(
            "(function(selector) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); if (nodes.length !== 1) throw new Error('strict mode: expected 1 node, got ' + nodes.length); {expr} }})({selector})"
        );
        self.evaluate(webview, &source)
    }

    fn handle(&mut self, method: &str, params: serde_json::Value) -> io::Result<serde_json::Value> {
        let _slow = SlowOp {
            method,
            started: Instant::now(),
        };
        self.reject_stale_objects(method, &params)?;
        self.reclaim_pump_tokens();
        match method {
            "chromium.launch" => {
                let browser = self.alloc_id("browser");
                self.browsers.insert(
                    browser.clone(),
                    ObjectLife::Live {
                        generation: 1,
                        parent: None,
                    },
                );
                Ok(json!({ "browser": browser, "generation": 1 }))
            }
            "browser.newContext" => {
                if let Some(browser_id) = params.get("browser").and_then(|value| value.as_str()) {
                    match self.browsers.get(browser_id) {
                        Some(ObjectLife::Live { .. }) => {}
                        _ => return Err(object_disposed("Browser")),
                    }
                }
                let context = self.alloc_id("context");
                let browser_id = params
                    .get("browser")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned);
                self.contexts.insert(
                    context.clone(),
                    ObjectLife::Live {
                        generation: 1,
                        parent: browser_id,
                    },
                );
                Ok(json!({ "context": context, "generation": 1 }))
            }
            "context.newPage" => {
                let context_id = params
                    .get("context")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned);
                let browser_id = context_id
                    .as_deref()
                    .and_then(|id| match self.contexts.get(id) {
                        Some(ObjectLife::Live { parent, .. }) => parent.clone(),
                        _ => None,
                    });
                if let Some(context_id) = context_id.as_deref() {
                    match self.contexts.get(context_id) {
                        Some(ObjectLife::Live { .. }) => {}
                        _ => return Err(object_disposed("BrowserContext")),
                    }
                }
                let page = self.alloc_id("page");
                let delegate = Rc::new(Delegate::new(
                    Rc::clone(&self.rendering_context),
                    self.profile.clone(),
                    self.wake.clone(),
                    Rc::clone(&self.user_content),
                ));
                let webview = WebViewBuilder::new(&self.servo, Rc::clone(&self.rendering_context))
                    .delegate(delegate.clone())
                    .user_content_manager(Rc::clone(&self.user_content))
                    .build();
                webview.show();
                webview.focus();
                let created = webview.clone();
                if !self.spin_until(ACTION_TIMEOUT, move || created.url().is_some())? {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "timed out creating page",
                    ));
                }
                self.pages.insert(
                    page.clone(),
                    PageSlot::live(webview, delegate, context_id, browser_id),
                );
                Ok(json!({ "page": page, "generation": 1 }))
            }
            "session.setProfile" => {
                let name = required_str(&params, "profile")?;
                let parsed = NetworkProfile::parse(&name).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "profile must be research or project",
                    )
                })?;
                self.profile.set(parsed);
                Ok(json!({ "profile": self.profile.get().as_str() }))
            }
            "session.networkBytes" => {
                // Real bytes relayed through the policy proxy, both
                // directions — the metric behind web.run's network_bytes,
                // which previously reported a fixed 4096-per-navigation
                // accounting stub.
                Ok(json!({ "bytes": self._proxy.bytes_transferred() }))
            }
            "page.goto" => {
                let page_id = required_str(&params, "page")?;
                let url = required_str(&params, "url")?;
                if let UrlDecision::Deny { reason } = decide_url(self.profile.get(), &url) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("policy_denied: {reason}"),
                    ));
                }
                let url = Url::parse(&url)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                delegate.denied_navigation.replace(None);
                let previous = webview.url();
                // Navigating to the URL the page already shows leaves `url`
                // unchanged, so a URL comparison cannot tell the new document
                // from the old one: the settle check below passed on the spot,
                // at the old document's `Complete`, and the query that followed
                // ran against a document being torn down. The same page then
                // answered 2, then 0, then 2 (Fund 028). Stamp the outgoing
                // document instead -- a fresh one never carries the stamp.
                let same_url = previous
                    .as_ref()
                    .is_some_and(|old| urls_match(old, &url));
                let stamped = same_url
                    && self
                        .evaluate_until(
                            webview.clone(),
                            "window.__greppyNavStamp = 1; true",
                            Duration::from_millis(150),
                        )
                        .is_ok();
                let goto_started = NavTrace::enabled().then(Instant::now);
                // Extra headers ride WebResourceLoad::continue_with_headers in
                // the fetch pipeline (above TLS). UrlRequest/load_request is
                // top-level navigation only and has stalled at HeadParsed.
                webview.load(url.clone());
                let loading = webview.clone();
                let expected = url.clone();
                let denied = Rc::clone(&delegate);
                let until = WaitUntil::from_params(&params);
                let engine = &*self;
                let mut last_stamp = Instant::now() - Duration::from_millis(200);
                if !self.spin_until_loaded_until(&loading, call_timeout(&params), until, || {
                    if denied.denied_navigation.borrow().is_some() {
                        return true;
                    }
                    let url_settled = loading.url().is_some_and(|current| {
                        urls_match(&current, &expected)
                            || previous.as_ref().is_some_and(|old| current != *old)
                    });
                    if !url_settled || !stamped {
                        return url_settled;
                    }
                    // Poll at the same 25ms cadence the readyState probe uses.
                    if last_stamp.elapsed() < Duration::from_millis(25) {
                        return false;
                    }
                    last_stamp = Instant::now();
                    matches!(
                        engine.evaluate_until(
                            loading.clone(),
                            "typeof window.__greppyNavStamp === 'undefined'",
                            Duration::from_millis(150),
                        ),
                        Ok(JSValue::Boolean(true))
                    )
                })? {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!(
                            "timed out navigating to {url} (status={:?}, url={:?})",
                            webview.load_status(),
                            webview.url()
                        ),
                    ));
                }
                if let Some(reason) = delegate.denied_navigation.borrow().clone() {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("policy_denied: {reason}"),
                    ));
                }
                if let Some(final_url) = webview.url() {
                    if let UrlDecision::Deny { reason } =
                        decide_url(self.profile.get(), final_url.as_str())
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            format!("policy_denied: {reason}"),
                        ));
                    }
                }
                // Hypothese 11 (Fund 023): die Navigation nimmt dem View den
                // Zustand, den die Eingabezustellung braucht. show() wurde beim
                // Anlegen gerufen, nach load() aber nie wieder.
                // show() lief nur beim Anlegen; ohne Wiederholung nach der
                // Navigation stellt Servo keine synthetische Eingabe mehr zu
                // (Fund 023). Kein zusaetzlicher spin_event_loop hier: ein
                // Extra-Umlauf mitten in der Navigation bringt die
                // Ereignisreihenfolge durcheinander -- Touch verlor dabei sein
                // touchend.
                webview.show();
                webview.focus();
                let loaded_ms = goto_started.map(|t| t.elapsed().as_millis());
                webview.paint();
                self.servo.spin_event_loop();
                // Complete is the load signal. Subresource scripts are waited
                // by waitForFunction / init scripts on the event loop; a
                // wall-clock sleep after every goto burned the 60s script
                // budget without observing script start.
                self.servo.spin_event_loop();
                let painted_ms = goto_started.map(|t| t.elapsed().as_millis());
                self.run_init_scripts(&page_id)?;
                if let (Some(started), Some(loaded), Some(painted)) =
                    (goto_started, loaded_ms, painted_ms)
                {
                    if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: goto-trace loaded_ms={loaded} painted_ms={painted} init_ms={} url={url}",
                        started.elapsed().as_millis(),
                    ); }
                }
                let final_url = webview
                    .url()
                    .map(|u| u.to_string())
                    .unwrap_or_else(|| url.to_string());
                let recorded = delegate.last_responses.borrow();
                let matched = recorded.iter().rev().find(|row| {
                    row.get("url")
                        .and_then(|value| value.as_str())
                        .is_some_and(|recorded_url| recorded_url == final_url)
                });
                let http = url.scheme() == "http" || url.scheme() == "https";
                let recorded_status = matched
                    .and_then(|row| row.get("status"))
                    .and_then(|value| value.as_u64());
                let status_text = matched
                    .and_then(|row| row.get("statusText"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_owned();
                let headers = matched
                    .and_then(|row| row.get("headers"))
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                drop(recorded);
                // A failed probe means UNKNOWN, never "empty": on a page
                // whose script thread is still busy (a 7.6MB spec mid-parse)
                // these evaluates time out, and treating that as an empty
                // document declared a 98s successful navigation failed.
                let text = match self.evaluate(
                    webview.clone(),
                    "(document.body && document.body.innerText) || \"\"",
                ) {
                    Ok(JSValue::String(text)) => Some(text),
                    _ => None,
                };
                if let Some(text) = &text {
                    if text.contains("Could not load the requested page") {
                        return Err(io::Error::other(format!("navigation failed: {text}")));
                    }
                }
                let html =
                    match self.evaluate(webview.clone(), "document.documentElement.outerHTML") {
                        Ok(JSValue::String(html)) => Some(html),
                        _ => None,
                    };
                // CONNECT-tunneled fetches often never match last_responses, so a
                // missing recorded status is not proof of failure. The Servo error
                // shell is already rejected above; accept a rendered document.
                //
                // Empty innerText plus tiny HTML is not proof either: a
                // <frameset> page has no body at all, so its innerText is
                // empty and the markup fits in a couple hundred bytes (bench
                // page 024). Count elements beyond the implied html/head/body
                // shell before declaring the response missing.
                let rendered_elements = match self.evaluate(
                    webview.clone(),
                    "document.querySelectorAll('*:not(html):not(head):not(body)').length",
                ) {
                    Ok(JSValue::Number(count)) => Some(count as u64),
                    _ => None,
                };
                if http
                    && recorded_status.is_none()
                    && text.as_deref().is_some_and(|t| t.trim().is_empty())
                    && html.as_deref().is_some_and(|h| h.len() < 500)
                    && rendered_elements == Some(0)
                {
                    return Err(io::Error::other(format!(
                        "navigation failed: no HTTP response for {url}"
                    )));
                }
                let status = recorded_status.unwrap_or(200);
                Ok(json!({
                    "url": final_url,
                    "status": status,
                    "statusText": if status_text.is_empty() && status < 400 {
                        "OK".to_owned()
                    } else {
                        status_text
                    },
                    "ok": status < 400,
                    "headers": headers,
                }))
            }
            "locator.click" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                let probe = format!(
                    "{}-{}",
                    std::process::id(),
                    CLICK_PROBE_SEQ.fetch_add(1, Ordering::Relaxed)
                );
                let encoded_probe =
                    serde_json::to_string(&probe).expect("click probe serializes");
                self.locator_eval(
                    &params,
                    &format!(
                        "var node = nodes[0]; var token = {encoded_probe}; var attr = 'data-greppy-click-probe'; var pending = 'pending:' + token; node.setAttribute(attr, pending); var mark = function() {{ if (node.getAttribute(attr) === pending) node.setAttribute(attr, 'seen:' + token); node.removeEventListener('click', mark, true); }}; node.addEventListener('click', mark, true); return true"
                    ),
                )?;
                self.present_exclusively(&webview);
                click_at(
                    &webview,
                    &delegate,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                    || self.servo.spin_event_loop(),
                )?;
                self.servo.spin_event_loop();
                let dispatch = match self.locator_eval(
                    &params,
                    &format!(
                        "var node = nodes[0]; var token = {encoded_probe}; var attr = 'data-greppy-click-probe'; var state = node.getAttribute(attr); if (state === 'seen:' + token) {{ node.removeAttribute(attr); return 'native'; }} if (state === 'pending:' + token) {{ node.click(); node.removeAttribute(attr); return 'dom-fallback'; }} return 'document-changed'"
                    ),
                ) {
                    Ok(JSValue::String(dispatch)) => dispatch,
                    // A successful click may navigate or replace its own node,
                    // making the old locator intentionally unresolvable. Never
                    // issue a second click in that case.
                    Err(_) => "document-changed".to_owned(),
                    Ok(_) => "unknown".to_owned(),
                };
                if let Some(selector) = params
                    .get("selector")
                    .and_then(|value| value.get("value"))
                    .and_then(|value| value.as_str())
                {
                    let _ = self.assign_pending_files(&page_id, selector);
                }
                Ok(json!({ "dispatch": dispatch }))
            }
            "locator.tap" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                tap_at(
                    &webview,
                    &delegate,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                    || self.servo.spin_event_loop(),
                )?;
                self.servo.spin_event_loop();
                Ok(json!({}))
            }
            "page.touch.tap" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, delegate) = self.page(&page_id)?.clone();
                tap_at(&webview, &delegate, x, y, 0.0, 0.0, || {
                    self.servo.spin_event_loop()
                })?;
                self.servo.spin_event_loop();
                Ok(json!({}))
            }
            "locator.dblclick" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.present_exclusively(&webview);
                click_at(
                    &webview,
                    &delegate,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                    || self.servo.spin_event_loop(),
                )?;
                self.servo.spin_event_loop();
                click_at(
                    &webview,
                    &delegate,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                    || self.servo.spin_event_loop(),
                )?;
                self.servo.spin_event_loop();
                let _ = self.locator_eval(
                    &params,
                    "nodes[0].dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); return true",
                );
                Ok(json!({}))
            }
            "locator.fill" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let value = required_str(&params, "value")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.present_exclusively(&webview);
                click_at(
                    &webview,
                    &delegate,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                    || self.servo.spin_event_loop(),
                )?;
                self.servo.spin_event_loop();
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let script = fill_script(&selector, &value);
                self.evaluate(webview, &script)?;
                Ok(json!({}))
            }
            "locator.innerText" => {
                let page_id = required_str(&params, "page")?;
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                match self.evaluate(webview, &inner_text_script(&selector))? {
                    JSValue::String(text) => Ok(json!({ "text": text })),
                    other => Err(io::Error::other(format!(
                        "innerText returned non-string: {other:?}"
                    ))),
                }
            }
            "page.evaluate" => {
                let page_id = required_str(&params, "page")?;
                let source = required_str(&params, "source")?;
                let (webview, _) = self.page(&page_id)?.clone();
                evaluate_serialized(self.evaluate(webview, &source)?)
            }
            "page.waitForFunction" => {
                let page_id = required_str(&params, "page")?;
                let source = required_str(&params, "source")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.wait_for_function_truthy(
                    webview,
                    &delegate,
                    &source,
                    call_timeout(&params),
                    None,
                )
            }
            "page.frameWaitForFunction" => {
                let page_id = required_str(&params, "page")?;
                let index = params.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let source = required_str(&params, "source")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.wait_for_function_truthy(
                    webview,
                    &delegate,
                    &source,
                    call_timeout(&params),
                    Some(index),
                )
            }
            "browser.close" => {
                if let Some(browser_id) = params.get("browser").and_then(|value| value.as_str()) {
                    if let Some(ObjectLife::Live { generation, .. }) = self.browsers.get(browser_id)
                    {
                        let generation = *generation;
                        self.browsers
                            .insert(browser_id.to_owned(), ObjectLife::disposed(generation));
                    }
                    let owned: Vec<(String, u64)> = self
                        .contexts
                        .iter()
                        .filter_map(|(id, life)| match life {
                            ObjectLife::Live {
                                generation,
                                parent: Some(parent),
                            } if parent == browser_id => Some((id.clone(), *generation)),
                            _ => None,
                        })
                        .collect();
                    for (id, generation) in owned {
                        self.contexts
                            .insert(id, ObjectLife::Disposed { generation });
                    }
                    self.dispose_pages_owned_by_browser(browser_id);
                } else {
                    self.dispose_all_pages();
                }
                Ok(json!({}))
            }
            "page.close" => {
                let page_id = required_str(&params, "page")?;
                self.dispose_page(&page_id);
                Ok(json!({}))
            }
            "page.isClosed" => {
                let page_id = required_str(&params, "page")?;
                let closed = !matches!(self.pages.get(&page_id), Some(PageSlot::Live { .. }));
                Ok(json!({ "closed": closed }))
            }
            "session.ensurePage" => self.handle("context.newPage", params),
            "page.url" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                Ok(json!({
                    "url": webview.url().map(|url| url.to_string()).unwrap_or_default()
                }))
            }
            "page.title" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                match self.evaluate(webview, "document.title")? {
                    JSValue::String(title) => Ok(json!({ "title": title })),
                    other => Ok(json!({ "title": format!("{other:?}") })),
                }
            }
            "page.content" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                match self.evaluate(webview, "document.documentElement.outerHTML")? {
                    JSValue::String(html) => Ok(json!({ "html": html })),
                    other => Err(io::Error::other(format!("content returned {other:?}"))),
                }
            }
            "page.observe" => {
                let page_id = required_str(&params, "page")?;
                let snapshot = params.get("snapshot").and_then(|value| value.as_str());
                let (webview, _) = self.page(&page_id)?.clone();
                match self.evaluate(webview, &observe_script(snapshot))? {
                    JSValue::String(text) => serde_json::from_str(&text)
                        .map_err(|error| io::Error::other(format!("observe json: {error}"))),
                    JSValue::Object(values) => Ok(jsvalue_to_json(JSValue::Object(values))),
                    other => Err(io::Error::other(format!("observe returned {other:?}"))),
                }
            }
            "page.screenshot" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let clip = params.get("clip").and_then(|value| {
                    if !value.is_object() {
                        return None;
                    }
                    let x = value
                        .get("x")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0)
                        .max(0.0) as u32;
                    let y = value
                        .get("y")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0)
                        .max(0.0) as u32;
                    let width = value
                        .get("width")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0)
                        .max(1.0) as u32;
                    let height = value
                        .get("height")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0)
                        .max(1.0) as u32;
                    Some((x, y, width, height))
                });
                // renderComplete: the agent explicitly asks for the finished
                // rendering — Servo's readiness machine (load fired, every
                // image and web font in) instead of the instant framebuffer.
                // The default answers "what does the page look like NOW".
                let png = if params
                    .get("renderComplete")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false)
                {
                    self.screenshot_png_render_complete(&webview, clip)?
                } else {
                    self.screenshot_png(&webview, clip)?
                };
                screenshot_engine_result(&png)
            }
            "locator.count" => {
                let page_id = required_str(&params, "page")?;
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(selector) {{ {SELECTOR_RUNTIME} return greppyResolveNodes(selector).length; }})({selector})"
                );
                match self.evaluate(webview, &script)? {
                    JSValue::Number(count) => Ok(json!({ "count": count as u64 })),
                    other => Err(io::Error::other(format!("count returned {other:?}"))),
                }
            }
            "locator.isVisible" => {
                let page_id = required_str(&params, "page")?;
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                match self.evaluate(webview, &resolve_script(&selector))? {
                    JSValue::Object(values) => {
                        let count = number_field(&values, "count").unwrap_or(0.0);
                        let visible = bool_field(&values, "visible").unwrap_or_else(|_| {
                            let width = number_field(&values, "width").unwrap_or(0.0);
                            let height = number_field(&values, "height").unwrap_or(0.0);
                            count == 1.0 && width > 0.0 && height > 0.0
                        });
                        Ok(json!({ "visible": count == 1.0 && visible }))
                    }
                    _ => Ok(json!({ "visible": false })),
                }
            }
            "locator.waitFor" => {
                let _ = self.resolve_actionable(&params)?;
                Ok(json!({}))
            }
            "locator.hover" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                hover_at(
                    &webview,
                    &delegate,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                    &mut || self.servo.spin_event_loop(),
                )?;
                self.servo.spin_event_loop();
                Ok(json!({}))
            }
            "locator.check" | "locator.uncheck" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let checked = method == "locator.check";
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, delegate) = self.page(&page_id)?.clone();
                // A property write plus dispatched input/change satisfies
                // plain forms but stays invisible to frameworks that bind
                // checkbox state to native CLICKS (React normalises checkbox
                // onChange over click): finding 033 measured click 5/5 vs
                // check 0/5 on the same app. So check IS a real, confirmed
                // click when the state has to change (Playwright semantics:
                // matching state is a no-op), verified afterwards; only if
                // the click did not toggle (custom widget swallowing the
                // event) fall back to property + events.
                let read_state = format!(
                    "(function(selector) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); if (nodes.length !== 1) throw new Error('strict mode'); return !!nodes[0].checked; }})({selector})"
                );
                let current = matches!(
                    self.evaluate(webview.clone(), &read_state)?,
                    JSValue::Boolean(true)
                );
                if current != checked {
                    self.present_exclusively(&webview);
                    click_at(
                        &webview,
                        &delegate,
                        resolved.x,
                        resolved.y,
                        resolved.width,
                        resolved.height,
                        || self.servo.spin_event_loop(),
                    )?;
                    self.servo.spin_event_loop();
                    let after = matches!(
                        self.evaluate(webview.clone(), &read_state)?,
                        JSValue::Boolean(true)
                    );
                    if after != checked {
                        let source = format!(
                            "(function(selector, checked) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); if (nodes.length !== 1) throw new Error('strict mode'); var el = nodes[0]; if (el.checked !== checked) {{ el.checked = checked; el.dispatchEvent(new Event('input', {{ bubbles: true }})); el.dispatchEvent(new Event('change', {{ bubbles: true }})); }} return true; }})({selector}, {checked})"
                        );
                        self.evaluate(webview, &source)?;
                    }
                }
                Ok(json!({}))
            }
            "locator.selectOption" => {
                let _ = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let value = required_str(&params, "value")?;
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                // Same rule as check: a selection the page never hears about
                // is worse than a failure (finding 019).
                let source = format!(
                    "(function(selector, value) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); if (nodes.length !== 1) throw new Error('strict mode'); var el = nodes[0]; if (el.value !== value) {{ el.value = value; el.dispatchEvent(new Event('input', {{ bubbles: true }})); el.dispatchEvent(new Event('change', {{ bubbles: true }})); }} return true; }})({selector}, {})",
                    serde_json::to_string(&value).map_err(io::Error::other)?
                );
                self.evaluate(webview, &source)?;
                Ok(json!({}))
            }
            "locator.inputValue" => {
                match self.locator_eval(&params, "return String(nodes[0].value || '')")? {
                    JSValue::String(value) => Ok(json!({ "value": value })),
                    other => Ok(json!({ "value": format!("{other:?}") })),
                }
            }
            "locator.getAttribute" => {
                let name = required_str(&params, "name")?;
                let name_json = serde_json::to_string(&name).map_err(io::Error::other)?;
                match self.locator_eval(
                    &params,
                    &format!("return nodes[0].getAttribute({name_json})"),
                )? {
                    JSValue::String(value) => Ok(json!({ "value": value })),
                    JSValue::Null => Ok(json!({ "value": serde_json::Value::Null })),
                    other => Ok(json!({ "value": format!("{other:?}") })),
                }
            }
            "locator.isChecked" => match self.locator_eval(&params, "return !!nodes[0].checked")? {
                JSValue::Boolean(value) => Ok(json!({ "checked": value })),
                _ => Ok(json!({ "checked": false })),
            },
            "locator.isEnabled" => match self.locator_eval(
                &params,
                "return !(nodes[0].disabled || nodes[0].getAttribute('aria-disabled') === 'true')",
            )? {
                JSValue::Boolean(value) => Ok(json!({ "enabled": value })),
                _ => Ok(json!({ "enabled": false })),
            },
            "locator.isDisabled" => match self.locator_eval(
                &params,
                "return !!(nodes[0].disabled || nodes[0].getAttribute('aria-disabled') === 'true')",
            )? {
                JSValue::Boolean(value) => Ok(json!({ "disabled": value })),
                _ => Ok(json!({ "disabled": false })),
            },
            "locator.isHidden" => {
                let visible = self.handle("locator.isVisible", params)?;
                Ok(
                    json!({ "hidden": !visible.get("visible").and_then(|v| v.as_bool()).unwrap_or(false) }),
                )
            }
            "locator.innerHTML" => {
                match self.locator_eval(&params, "return String(nodes[0].innerHTML || '')")? {
                    JSValue::String(html) => Ok(json!({ "html": html })),
                    other => Ok(json!({ "html": format!("{other:?}") })),
                }
            }
            "locator.focus" => {
                self.locator_eval(&params, "nodes[0].focus(); return true")?;
                Ok(json!({}))
            }
            "locator.blur" => {
                self.locator_eval(&params, "nodes[0].blur(); return true")?;
                Ok(json!({}))
            }
            "locator.boundingBox" => {
                let resolved = self.resolve_actionable(&params)?;
                Ok(json!({
                    "x": resolved.x,
                    "y": resolved.y,
                    "width": resolved.width,
                    "height": resolved.height,
                }))
            }
            "locator.screenshot" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let clip = Some((
                    resolved.x.max(0.0) as u32,
                    resolved.y.max(0.0) as u32,
                    resolved.width.max(1.0) as u32,
                    resolved.height.max(1.0) as u32,
                ));
                let png = self.screenshot_png(&webview, clip)?;
                screenshot_engine_result(&png)
            }
            "locator.allTextContents" => {
                let page_id = required_str(&params, "page")?;
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(selector) {{ {SELECTOR_RUNTIME} return greppyResolveNodes(selector).map(function(el) {{ return ((el.innerText || el.textContent || '') + '').trim(); }}); }})({selector})"
                );
                Ok(json!({ "values": jsvalue_to_json(self.evaluate(webview, &source)?) }))
            }
            "locator.evaluate" => {
                let source = required_str(&params, "source")?;
                let page_id = required_str(&params, "page")?;
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(selector, source) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); if (nodes.length !== 1) throw new Error('strict mode'); return (0, eval)('(' + source + ')')(nodes[0]); }})({selector}, {})",
                    serde_json::to_string(&source).map_err(io::Error::other)?
                );
                evaluate_serialized(self.evaluate(webview, &script)?)
            }
            "locator.evaluateAll" => {
                let source = required_str(&params, "source")?;
                let page_id = required_str(&params, "page")?;
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(selector, source) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); return (0, eval)('(' + source + ')')(nodes); }})({selector}, {})",
                    serde_json::to_string(&source).map_err(io::Error::other)?
                );
                evaluate_serialized(self.evaluate(webview, &script)?)
            }
            "locator.dispatchEvent" => {
                let event = required_str(&params, "event")?;
                let event_json = serde_json::to_string(&event).map_err(io::Error::other)?;
                self.locator_eval(
                    &params,
                    &format!(
                        "nodes[0].dispatchEvent(new Event({event_json}, {{ bubbles: true }})); return true"
                    ),
                )?;
                Ok(json!({}))
            }
            "locator.isEditable" => match self.locator_eval(
                &params,
                "return !nodes[0].disabled && (nodes[0].isContentEditable || 'value' in nodes[0])",
            )? {
                JSValue::Boolean(value) => Ok(json!({ "editable": value })),
                _ => Ok(json!({ "editable": false })),
            },
            "locator.scrollIntoViewIfNeeded" => {
                self.locator_eval(&params, "nodes[0].scrollIntoView(); return true")?;
                Ok(json!({}))
            }
            "locator.selectText" => {
                self.locator_eval(
                    &params,
                    "if (nodes[0].select) { nodes[0].select(); } else if (nodes[0].setSelectionRange && typeof nodes[0].value === 'string') { nodes[0].setSelectionRange(0, nodes[0].value.length); } return true",
                )?;
                Ok(json!({}))
            }
            "page.setExtraHTTPHeaders" => {
                let page_id = required_str(&params, "page")?;
                let headers = params
                    .get("headers")
                    .and_then(|value| value.as_object())
                    .cloned()
                    .unwrap_or_default();
                let stored: Vec<(String, String)> = headers
                    .into_iter()
                    .filter_map(|(name, value)| {
                        value.as_str().map(|value| (name, value.to_owned()))
                    })
                    .collect();
                self.page(&page_id)?.1.extra_headers.replace(stored);
                Ok(json!({}))
            }
            "page.addScriptTag" => {
                let page_id = required_str(&params, "page")?;
                let content = params
                    .get("content")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let src = params
                    .get("url")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(content, src) {{ var el = document.createElement('script'); if (src) el.src = src; else el.text = content; (document.head || document.documentElement).appendChild(el); return true; }})({}, {})",
                    serde_json::to_string(&content).map_err(io::Error::other)?,
                    serde_json::to_string(&src).map_err(io::Error::other)?
                );
                self.evaluate(webview, &script)?;
                Ok(json!({}))
            }
            "page.addStyleTag" => {
                let page_id = required_str(&params, "page")?;
                let content = params
                    .get("content")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(css) {{ var el = document.createElement('style'); el.textContent = css; (document.head || document.documentElement).appendChild(el); return true; }})({})",
                    serde_json::to_string(&content).map_err(io::Error::other)?
                );
                self.evaluate(webview, &script)?;
                Ok(json!({}))
            }
            "page.setContent" => {
                let page_id = required_str(&params, "page")?;
                let html = required_str(&params, "html")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(html) {{ document.open(); document.write(html); document.close(); return document.documentElement.outerHTML.length; }})({})",
                    serde_json::to_string(&html).map_err(io::Error::other)?
                );
                self.evaluate(webview.clone(), &source)?;
                webview.paint();
                self.servo.spin_event_loop();
                self.run_init_scripts(&page_id)?;
                Ok(json!({}))
            }
            "page.reload" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let url = webview
                    .url()
                    .ok_or_else(|| io::Error::other("page has no url to reload"))?;
                let mut goto_params = json!({ "page": page_id, "url": url.as_str() });
                if let Some(timeout) = params.get("timeout") {
                    goto_params["timeout"] = timeout.clone();
                }
                self.handle("page.goto", goto_params)
            }
            "page.waitForLoadState" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let loading = webview.clone();
                if !self.spin_until_loaded(&loading, call_timeout(&params), || true)? {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "timed out waiting for load state",
                    ));
                }
                Ok(json!({}))
            }
            "page.keyboard.type" => {
                let page_id = required_str(&params, "page")?;
                let text = required_str(&params, "text")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(text) {{ {KEYBOARD_RUNTIME} greppyType(greppyActive(), text); return true; }})({})",
                    serde_json::to_string(&text).map_err(io::Error::other)?
                );
                self.evaluate(webview, &source)?;
                Ok(json!({}))
            }
            "page.keyboard.insertText" => {
                let page_id = required_str(&params, "page")?;
                let text = required_str(&params, "text")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(text) {{ {KEYBOARD_RUNTIME} greppyInsertText(greppyActive(), text); return true; }})({})",
                    serde_json::to_string(&text).map_err(io::Error::other)?
                );
                self.evaluate(webview, &source)?;
                Ok(json!({}))
            }
            "page.keyboard.press" => {
                let page_id = required_str(&params, "page")?;
                let key = required_str(&params, "key")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(key) {{ {KEYBOARD_RUNTIME} greppyPress(greppyActive(), key); return true; }})({})",
                    serde_json::to_string(&key).map_err(io::Error::other)?
                );
                self.evaluate(webview.clone(), &source)?;
                Ok(json!({}))
            }
            "page.keyboard.down" => {
                let page_id = required_str(&params, "page")?;
                let key = required_str(&params, "key")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(key) {{ {KEYBOARD_RUNTIME} greppyDown(greppyActive(), key); return true; }})({})",
                    serde_json::to_string(&key).map_err(io::Error::other)?
                );
                self.evaluate(webview, &source)?;
                Ok(json!({}))
            }
            "page.keyboard.up" => {
                let page_id = required_str(&params, "page")?;
                let key = required_str(&params, "key")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(key) {{ {KEYBOARD_RUNTIME} greppyUp(greppyActive(), key); return true; }})({})",
                    serde_json::to_string(&key).map_err(io::Error::other)?
                );
                self.evaluate(webview, &source)?;
                Ok(json!({}))
            }
            "page.setDialogPolicy" => {
                let page_id = required_str(&params, "page")?;
                let action = params
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("accept");
                let delegate = &self.page(&page_id)?.1;
                *delegate.dialog_action.borrow_mut() = if action == "dismiss" {
                    DialogAction::Dismiss
                } else {
                    DialogAction::Accept
                };
                delegate.prompt_text.replace(
                    params
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned),
                );
                Ok(json!({}))
            }
            "page.dialogs" => {
                let page_id = required_str(&params, "page")?;
                let consume = params
                    .get("consume")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let dialogs = if consume {
                    self.page(&page_id)?
                        .1
                        .last_dialogs
                        .borrow_mut()
                        .drain(..)
                        .collect::<Vec<_>>()
                } else {
                    self.page(&page_id)?.1.last_dialogs.borrow().clone()
                };
                Ok(json!({ "dialogs": dialogs }))
            }
            "page.consoleMessages" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({
                    "messages": self.page(&page_id)?.1.last_console.borrow().clone()
                }))
            }
            "page.clearConsoleMessages" => {
                let page_id = required_str(&params, "page")?;
                self.page(&page_id)?.1.last_console.borrow_mut().clear();
                Ok(json!({}))
            }
            "page.clearPageErrors" => {
                let page_id = required_str(&params, "page")?;
                self.page(&page_id)?
                    .1
                    .last_console
                    .borrow_mut()
                    .retain(|row| {
                        row.get("type").and_then(|value| value.as_str()) != Some("error")
                    });
                Ok(json!({}))
            }
            "page.fileChoosers" => {
                let page_id = required_str(&params, "page")?;
                let consume = params
                    .get("consume")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let choosers = if consume {
                    self.page(&page_id)?
                        .1
                        .last_file_choosers
                        .borrow_mut()
                        .drain(..)
                        .collect::<Vec<_>>()
                } else {
                    self.page(&page_id)?.1.last_file_choosers.borrow().clone()
                };
                Ok(json!({ "choosers": choosers }))
            }
            "context.close" => {
                if let Some(context_id) = params.get("context").and_then(|value| value.as_str()) {
                    if let Some(ObjectLife::Live { generation, .. }) = self.contexts.get(context_id)
                    {
                        let generation = *generation;
                        self.contexts
                            .insert(context_id.to_owned(), ObjectLife::Disposed { generation });
                    }
                    self.dispose_pages_owned_by_context(context_id);
                }
                Ok(json!({}))
            }
            "page.addRoute" => {
                let page_id = required_str(&params, "page")?;
                let pattern = required_str(&params, "pattern")?;
                let action = params
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("continue")
                    .to_owned();
                let body = match params.get("bodyBase64").and_then(|value| value.as_str()) {
                    Some(b64) if !b64.is_empty() => base64_decode(b64)?,
                    Some(_) => Vec::new(),
                    None => {
                        if params.get("body").is_some() {
                            return Err(io::Error::other(
                                "fulfill body must be bodyBase64; UTF-8 strings are not a canonical body",
                            ));
                        }
                        Vec::new()
                    }
                };
                let status = params.get("status").and_then(|v| v.as_u64()).unwrap_or(200) as u16;
                let content_type = params
                    .get("contentType")
                    .and_then(|value| value.as_str())
                    .unwrap_or("text/html")
                    .to_owned();
                let continue_headers = params
                    .get("headers")
                    .and_then(|value| value.as_object())
                    .map(|object| {
                        object
                            .iter()
                            .filter_map(|(name, value)| {
                                value.as_str().map(|value| (name.clone(), value.to_owned()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                self.page(&page_id)?.1.routes.borrow_mut().push(RouteRule {
                    pattern,
                    action,
                    body,
                    status,
                    content_type,
                    continue_headers,
                });
                Ok(json!({}))
            }
            "page.unroute" => {
                let page_id = required_str(&params, "page")?;
                let pattern = required_str(&params, "pattern")?;
                self.page(&page_id)?
                    .1
                    .routes
                    .borrow_mut()
                    .retain(|rule| rule.pattern != pattern);
                Ok(json!({}))
            }
            "page.unrouteAll" => {
                let page_id = required_str(&params, "page")?;
                self.page(&page_id)?.1.routes.borrow_mut().clear();
                Ok(json!({}))
            }
            "page.setInputFiles" => {
                let page_id = required_str(&params, "page")?;
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("#file")
                    .to_owned();
                let files = params
                    .get("files")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let mut paths: Vec<std::path::PathBuf> = Vec::new();
                for value in &files {
                    let raw = value.as_str().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "setInputFiles requires path strings",
                        )
                    })?;
                    let path = confine_worker_path(Path::new(raw))?;
                    std::fs::read(&path).map_err(|error| {
                        io::Error::new(
                            error.kind(),
                            format!("setInputFiles cannot read {}: {error}", path.display()),
                        )
                    })?;
                    paths.push(path);
                }
                self.page(&page_id)?.1.file_paths.replace(paths);
                self.assign_pending_files(&page_id, &selector)
            }
            "page.waitForRequest" => {
                let page_id = required_str(&params, "page")?;
                let needle = params
                    .get("needle")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_owned();
                let (webview, delegate) = self.page(&page_id)?.clone();
                let hit = self.wait_for_recorded(
                    &webview,
                    call_timeout(&params),
                    "timeout: waitForRequest",
                    || {
                        delegate
                            .requests
                            .borrow()
                            .iter()
                            .any(|rec| recorded_url_contains(rec, &needle))
                    },
                    || {
                        delegate
                            .requests
                            .borrow()
                            .iter()
                            .find(|rec| recorded_url_contains(rec, &needle))
                            .cloned()
                    },
                )?;
                Ok(json!({ "request": crate::daemon::redact_json(hit) }))
            }
            "page.waitForResponse" => {
                let page_id = required_str(&params, "page")?;
                let needle = params
                    .get("needle")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_owned();
                let (webview, delegate) = self.page(&page_id)?.clone();
                let hit = self.wait_for_recorded(
                    &webview,
                    call_timeout(&params),
                    "timeout: waitForResponse",
                    || {
                        delegate
                            .last_responses
                            .borrow()
                            .iter()
                            .any(|rec| recorded_url_contains(rec, &needle))
                    },
                    || {
                        delegate
                            .last_responses
                            .borrow()
                            .iter()
                            .find(|rec| recorded_url_contains(rec, &needle))
                            .cloned()
                    },
                )?;
                Ok(json!({ "response": hit }))
            }
            "page.waitForDownload" => {
                let page_id = required_str(&params, "page")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                let hit = self.wait_for_recorded(
                    &webview,
                    call_timeout(&params),
                    "timeout: waitForEvent download",
                    || !delegate.downloads.borrow().is_empty(),
                    || delegate.downloads.borrow().first().cloned(),
                )?;
                Ok(json!({ "download": hit }))
            }
            "page.waitForFileChooser" => {
                let page_id = required_str(&params, "page")?;
                let (webview, delegate) = self.page(&page_id)?.clone();
                let hit = self.wait_for_recorded(
                    &webview,
                    call_timeout(&params),
                    "timeout: waitForEvent filechooser",
                    || !delegate.last_file_choosers.borrow().is_empty(),
                    || {
                        let mut choosers = delegate.last_file_choosers.borrow_mut();
                        if choosers.is_empty() {
                            None
                        } else {
                            Some(choosers.remove(0))
                        }
                    },
                )?;
                Ok(json!({ "chooser": hit }))
            }
            "page.requests" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({
                    "requests": crate::daemon::redact_json(json!(self.page(&page_id)?.1.requests.borrow().clone()))
                }))
            }
            "page.responses" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({ "responses": self.page(&page_id)?.1.last_responses.borrow().clone() }))
            }
            "page.downloads" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({ "downloads": self.page(&page_id)?.1.downloads.borrow().clone() }))
            }
            "page.saveDownload" => {
                let page_id = required_str(&params, "page")?;
                let url = required_str(&params, "url")?;
                let path = required_str(&params, "path")?;
                let body_b64 = {
                    let downloads = self.page(&page_id)?.1.downloads.borrow();
                    downloads
                        .iter()
                        .rev()
                        .find(|row| {
                            row.get("url").and_then(|value| value.as_str()) == Some(url.as_str())
                        })
                        .and_then(|row| {
                            row.get("bodyBase64")
                                .and_then(|value| value.as_str())
                                .map(str::to_owned)
                        })
                        .ok_or_else(|| io::Error::other("no matching download body"))?
                };
                let bytes = base64_decode(&body_b64)?;
                let path = confine_worker_path(Path::new(&path))?;
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                std::fs::write(&path, &bytes)?;
                let written = std::fs::read(&path)?;
                if written != bytes {
                    return Err(io::Error::other("saveDownload readback mismatch"));
                }
                Ok(json!({
                    "ok": true,
                    "bytes": written.len(),
                    "hex": hex_encode(&written),
                }))
            }
            "page.popups" => {
                let page_id = required_str(&params, "page")?;
                let taken: Vec<(WebView, WebView)> = {
                    let delegate = &self.page(&page_id)?.1;
                    delegate.popups.borrow_mut().drain(..).collect()
                };
                let mut pages = Vec::new();
                for (webview, parent) in taken {
                    let opener = self
                        .pages
                        .iter()
                        .find_map(|(id, slot)| match slot {
                            PageSlot::Live {
                                pair: (existing, _),
                                ..
                            } if existing == &parent => Some(id.clone()),
                            _ => None,
                        })
                        .unwrap_or_else(|| page_id.clone());
                    let (context_id, browser_id) = match self.pages.get(&opener) {
                        Some(PageSlot::Live {
                            context_id,
                            browser_id,
                            ..
                        }) => (context_id.clone(), browser_id.clone()),
                        _ => match self.pages.get(&page_id) {
                            Some(PageSlot::Live {
                                context_id,
                                browser_id,
                                ..
                            }) => (context_id.clone(), browser_id.clone()),
                            _ => (None, None),
                        },
                    };
                    let id = self.alloc_id("page");
                    let delegate = Rc::new(Delegate::new(
                        Rc::clone(&self.rendering_context),
                        self.profile.clone(),
                        self.wake.clone(),
                        Rc::clone(&self.user_content),
                    ));
                    delegate.opener_id.replace(Some(opener.clone()));
                    self.pages.insert(
                        id.clone(),
                        PageSlot::live(webview, delegate, context_id, browser_id),
                    );
                    pages.push(json!({ "page": id, "opener": opener, "generation": 1 }));
                }
                Ok(json!({ "pages": pages }))
            }
            "page.opener" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({
                    "page": self.page(&page_id)?.1.opener_id.borrow().clone()
                }))
            }
            "page.frames" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let value = self.evaluate(
                    webview,
                    r#"JSON.stringify(Array.from(document.querySelectorAll("iframe")).map(function(frame, index) {
  return { id: String(index), name: frame.name || "", url: frame.src || "" };
}))"#,
                )?;
                let frames = match value {
                    JSValue::String(text) => {
                        serde_json::from_str::<serde_json::Value>(&text).unwrap_or(json!([]))
                    }
                    other => jsvalue_to_json(other),
                };
                Ok(json!({ "frames": frames }))
            }
            "page.frameIsDetached" => {
                let page_id = required_str(&params, "page")?;
                let index = params.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(index) {{ return !document.querySelectorAll('iframe')[index]; }})({index})"
                );
                let detached = match self.evaluate(webview, &source)? {
                    JSValue::Boolean(value) => value,
                    other => matches!(other, JSValue::String(text) if text == "true"),
                };
                Ok(json!({ "detached": detached }))
            }
            "page.frameGoto" => {
                let page_id = required_str(&params, "page")?;
                let index = params.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let url = required_str(&params, "url")?;
                if let UrlDecision::Deny { reason } = decide_url(self.profile.get(), &url) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("policy_denied: {reason}"),
                    ));
                }
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(index, url) {{ var frame = document.querySelectorAll('iframe')[index]; if (!frame) throw new Error('no frame'); try {{ frame.contentWindow.location.replace(url); }} catch (e) {{ frame.src = url; }} return String(url); }})({index}, {})",
                    serde_json::to_string(&url).map_err(io::Error::other)?
                );
                let assigned = match self.evaluate(webview.clone(), &source)? {
                    JSValue::String(text) => text,
                    _ => url.clone(),
                };
                let ready_script = format!(
                    "(function(index, want) {{ var frame = document.querySelectorAll('iframe')[index]; if (!frame) return false; try {{ var loc = String(frame.contentWindow.location.href || ''); var doc = frame.contentDocument; return loc.indexOf(want) !== -1 && doc && (doc.readyState === 'complete' || doc.readyState === 'interactive'); }} catch (e) {{ return false; }} }})({index}, {})",
                    serde_json::to_string(&url).map_err(io::Error::other)?
                );
                let deadline = Instant::now() + call_timeout(&params);
                loop {
                    if self.parent_dead() {
                        return Err(Self::parent_gone());
                    }
                    let ready = match self.evaluate(webview.clone(), &ready_script) {
                        Ok(JSValue::Boolean(value)) => value,
                        Ok(JSValue::String(text)) => text == "true",
                        Ok(_) => false,
                        Err(_) => false,
                    };
                    if ready {
                        break;
                    }
                    if Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "timed out waiting for frame navigation",
                        ));
                    }
                    self.servo.spin_event_loop();
                    thread::sleep(Duration::from_millis(1));
                }
                Ok(json!({ "url": assigned }))
            }
            "page.frameEvaluate" => {
                let page_id = required_str(&params, "page")?;
                let index = params.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let source = required_str(&params, "source")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(index, source) {{ var frame = document.querySelectorAll('iframe')[index]; if (!frame) throw new Error('no frame'); return frame.contentWindow.eval(source); }})({index}, {})",
                    serde_json::to_string(&source).map_err(io::Error::other)?
                );
                evaluate_serialized(self.evaluate(webview, &script)?)
            }
            "page.goBack" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let ok = webview.can_go_back();
                if ok {
                    webview.go_back(1);
                    let loading = webview.clone();
                    let _ = self.spin_until_loaded(&loading, call_timeout(&params), || true)?;
                    webview.paint();
                    self.servo.spin_event_loop();
                }
                Ok(json!({
                    "ok": ok,
                    "url": webview.url().map(|url| url.to_string()).unwrap_or_default()
                }))
            }
            "page.goForward" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let ok = webview.can_go_forward();
                if ok {
                    webview.go_forward(1);
                    let loading = webview.clone();
                    let _ = self.spin_until_loaded(&loading, call_timeout(&params), || true)?;
                    webview.paint();
                    self.servo.spin_event_loop();
                }
                Ok(json!({
                    "ok": ok,
                    "url": webview.url().map(|url| url.to_string()).unwrap_or_default()
                }))
            }
            "page.addCookies" => {
                let page_id = required_str(&params, "page")?;
                let cookies = params.get("cookies").cloned().unwrap_or(json!([]));
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(cookies) {{ cookies.forEach(function(c) {{ var parts = [c.name + '=' + c.value]; if (c.path) parts.push('path=' + c.path); if (c.domain) parts.push('domain=' + c.domain); if (c.secure) parts.push('Secure'); if (c.sameSite) parts.push('SameSite=' + c.sameSite); document.cookie = parts.join(';'); }}); return true; }})({})",
                    cookies
                );
                let _ = self.evaluate(webview, &script);
                Ok(json!({}))
            }
            "page.cookies" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                match self.evaluate(webview, "document.cookie")? {
                    JSValue::String(cookie) => Ok(json!({ "cookie": cookie })),
                    _ => Ok(json!({ "cookie": "" })),
                }
            }
            "page.clearCookies" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let _ = self.evaluate(
                    webview,
                    r#"(function() { document.cookie.split(';').forEach(function(part) { var name = part.split('=')[0].trim(); if (name) document.cookie = name + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/'; }); return document.cookie; })()"#,
                );
                Ok(json!({}))
            }
            "page.tracing" => {
                let page_id = required_str(&params, "page")?;
                let delegate = &self.page(&page_id)?.1;
                Ok(crate::daemon::redact_json(json!({
                    "requests": delegate.requests.borrow().clone(),
                    "downloads": delegate.downloads.borrow().clone(),
                    "file_paths": delegate
                        .file_paths
                        .borrow()
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>(),
                })))
            }
            "page.addInitScript" => {
                let page_id = required_str(&params, "page")?;
                let source = required_str(&params, "source")?;
                self.page(&page_id)?
                    .1
                    .init_scripts
                    .borrow_mut()
                    .push(source);
                Ok(json!({}))
            }
            "page.setViewportSize" => {
                let page_id = required_str(&params, "page")?;
                let width = params
                    .get("width")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0)
                    .max(1) as u32;
                let height = params
                    .get("height")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0)
                    .max(1) as u32;
                let (webview, delegate) = self.page(&page_id)?.clone();
                webview.resize(PhysicalSize { width, height });
                *delegate.viewport.borrow_mut() = (width, height);
                self.servo.spin_event_loop();
                Ok(json!({ "width": width, "height": height }))
            }
            "page.viewportSize" => {
                let page_id = required_str(&params, "page")?;
                let (width, height) = *self.page(&page_id)?.1.viewport.borrow();
                Ok(json!({ "width": width, "height": height }))
            }
            "page.mouse.click" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.present_exclusively(&webview);
                click_at(&webview, &delegate, x, y, 0.0, 0.0, || {
                    self.servo.spin_event_loop()
                })?;
                self.servo.spin_event_loop();
                Ok(json!({}))
            }
            "page.mouse.move" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.present_exclusively(&webview);
                let point = WebViewPoint::Device(DevicePoint::new(x as f32, y as f32));
                dispatch_input_and_wait(
                    &webview,
                    &delegate,
                    || InputEvent::MouseMove(MouseMoveEvent::new(point)),
                    &mut || self.servo.spin_event_loop(),
                )?;
                Ok(json!({}))
            }
            "page.mouse.down" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.present_exclusively(&webview);
                let point = WebViewPoint::Device(DevicePoint::new(x as f32, y as f32));
                dispatch_input_and_wait(
                    &webview,
                    &delegate,
                    || InputEvent::MouseButton(MouseButtonEvent::new(
                        MouseButtonAction::Down,
                        MouseButton::Left,
                        point,
                    )),
                    &mut || self.servo.spin_event_loop(),
                )?;
                Ok(json!({}))
            }
            "page.mouse.wheel" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let delta_x = params.get("deltaX").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let delta_y = params.get("deltaY").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.present_exclusively(&webview);
                let probe = format!(
                    "{}-{}",
                    std::process::id(),
                    WHEEL_PROBE_SEQ.fetch_add(1, Ordering::Relaxed)
                );
                let encoded_probe =
                    serde_json::to_string(&probe).expect("wheel probe serializes");
                self.evaluate(
                    webview.clone(),
                    &format!(
                        "(function() {{ var node = document.elementFromPoint({x}, {y}); if (!node) return false; var token = {encoded_probe}; var attr = 'data-greppy-wheel-probe'; var pending = 'pending:' + token; node.setAttribute(attr, pending); var mark = function() {{ if (node.getAttribute(attr) === pending) node.setAttribute(attr, 'seen:' + token); node.removeEventListener('wheel', mark, true); }}; node.addEventListener('wheel', mark, true); return true; }})()"
                    ),
                )?;
                let point = WebViewPoint::Device(DevicePoint::new(x as f32, y as f32));
                dispatch_input_and_wait(
                    &webview,
                    &delegate,
                    || InputEvent::Wheel(WheelEvent::new(
                        WheelDelta {
                            x: delta_x,
                            y: delta_y,
                            z: 0.0,
                            mode: WheelMode::DeltaPixel,
                        },
                        point,
                    )),
                    &mut || self.servo.spin_event_loop(),
                )?;
                let dispatch = match self.evaluate(
                    webview,
                    &format!(
                        "(function() {{ var token = {encoded_probe}; var attr = 'data-greppy-wheel-probe'; var node = document.querySelector('[' + attr + '=\"pending:' + token + '\"],[' + attr + '=\"seen:' + token + '\"]'); if (!node) return 'document-changed'; var state = node.getAttribute(attr); if (state === 'seen:' + token) {{ node.removeAttribute(attr); return 'native'; }} node.dispatchEvent(new WheelEvent('wheel', {{ bubbles: true, cancelable: true, deltaX: {delta_x}, deltaY: {delta_y}, deltaMode: 0, clientX: {x}, clientY: {y} }})); node.removeAttribute(attr); return 'dom-fallback'; }})()"
                    ),
                ) {
                    Ok(JSValue::String(dispatch)) => dispatch,
                    Err(_) => "document-changed".to_owned(),
                    Ok(_) => "unknown".to_owned(),
                };
                Ok(json!({ "dispatch": dispatch }))
            }
            "page.mouse.up" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, delegate) = self.page(&page_id)?.clone();
                self.present_exclusively(&webview);
                let point = WebViewPoint::Device(DevicePoint::new(x as f32, y as f32));
                dispatch_input_and_wait(
                    &webview,
                    &delegate,
                    || InputEvent::MouseButton(MouseButtonEvent::new(
                        MouseButtonAction::Up,
                        MouseButton::Left,
                        point,
                    )),
                    &mut || self.servo.spin_event_loop(),
                )?;
                Ok(json!({}))
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported_playwright_operation: {other}"),
            )),
        }
    }

    /// Make `target` the only visible webview before delivering synthetic
    /// input or reading the framebuffer. All live webviews share one
    /// rendering context and every page ever opened stays shown; input
    /// delivery then hits whichever stale webview the internal order offers
    /// and the event dies silently in a dead document (finding 034: the CLI
    /// verb loop lost 8 of 12 clicks to orphaned sessions, non-monotonically;
    /// closing the orphans made it 12/12).
    fn present_exclusively(&self, target: &WebView) {
        for other in self.live_webviews() {
            if other.id() != target.id() {
                other.hide();
            }
        }
        target.show();
        target.focus();
        // hide/show travel through the constellation asynchronously; without
        // a spin the hit test can still see the old visibility and route the
        // very next input into a hidden webview (2 of 12 clicks still died).
        self.servo.spin_event_loop();
    }
    fn resolve_actionable(&self, params: &serde_json::Value) -> io::Result<ResolvedNode> {
        let page_id = required_str(params, "page")?;
        let selector = params
            .get("selector")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let (webview, _) = self.page(&page_id)?.clone();
        self.present_exclusively(&webview);
        let script = resolve_script(&selector);
        let deadline = Instant::now() + call_timeout(params);
        let mut last = String::from("failed_check=stable");
        let mut stable: Option<ResolvedNode> = None;
        let mut stable_since: Option<Instant> = None;
        let timeout_err = |last: &str| {
            let detail = if last.contains("failed_check=") {
                last.to_owned()
            } else {
                format!("failed_check=stable; {last}")
            };
            io::Error::new(
                io::ErrorKind::TimedOut,
                format!("timed out waiting for actionable locator target ({detail})"),
            )
        };
        loop {
            if self.parent_dead() {
                return Err(Self::parent_gone());
            }
            if Instant::now() >= deadline {
                return Err(timeout_err(&last));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let sampled = match self.evaluate_until(webview.clone(), &script, remaining) {
                Ok(value) => value,
                Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                    return Err(timeout_err(&last));
                }
                Err(error) => return Err(error),
            };
            match sampled {
                JSValue::Object(values) => {
                    if bool_field(&values, "staleRef").unwrap_or(false) {
                        return Err(io::Error::other(
                            "STALE_REF: observed node no longer belongs to the active document",
                        ));
                    }
                    let count = number_field(&values, "count")? as usize;
                    let width = number_field(&values, "width").unwrap_or(0.0);
                    let height = number_field(&values, "height").unwrap_or(0.0);
                    let disabled = bool_field(&values, "disabled").unwrap_or(false);
                    let readonly = bool_field(&values, "readonly").unwrap_or(false);
                    let visible =
                        bool_field(&values, "visible").unwrap_or(width > 0.0 && height > 0.0);
                    let hit = bool_field(&values, "hit").unwrap_or(visible);
                    let need_editable = params
                        .get("editable")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false);
                    let offset_left = number_field(&values, "offsetLeft").unwrap_or(0.0);
                    let offset_top = number_field(&values, "offsetTop").unwrap_or(0.0);
                    last = format!(
                        "count={count} width={width} height={height} offsetLeft={offset_left} offsetTop={offset_top} disabled={disabled} readonly={readonly} visible={visible} hit={hit}"
                    );
                    if count == 1 && visible && hit && !disabled && (!need_editable || !readonly) {
                        let node = ResolvedNode {
                            x: number_field(&values, "x")?,
                            y: number_field(&values, "y")?,
                            width,
                            height,
                            offset_left,
                            offset_top,
                        };
                        if let Some(prev) = &stable {
                            if (prev.x - node.x).abs() < 1.0
                                && (prev.y - node.y).abs() < 1.0
                                && (prev.width - node.width).abs() < 1.0
                                && (prev.height - node.height).abs() < 1.0
                                && (prev.offset_left - node.offset_left).abs() < 1.0
                                && (prev.offset_top - node.offset_top).abs() < 1.0
                            {
                                if stable_since.is_some_and(|since| {
                                    since.elapsed() >= Duration::from_millis(32)
                                }) {
                                    self.settle_pump_tokens(&webview);
                                    return Ok(node);
                                }
                            } else {
                                stable_since = Some(Instant::now());
                                last = format!("failed_check=stable; {last}");
                            }
                        } else {
                            stable_since = Some(Instant::now());
                        }
                        stable = Some(node);
                    } else {
                        stable = None;
                        stable_since = None;
                        if count > 1 {
                            return Err(io::Error::other(format!(
                                "strict mode: selector matched {count} nodes"
                            )));
                        }
                        last = if count == 0 {
                            format!("failed_check=attached; {last}")
                        } else if !visible {
                            format!("failed_check=visible; {last}")
                        } else if !hit {
                            format!("failed_check=hit; {last}")
                        } else {
                            format!("failed_check=enabled; {last}")
                        };
                    }
                }
                other => {
                    stable = None;
                    stable_since = None;
                    last = format!("failed_check=attached; {other:?}");
                }
            }
            if Instant::now() >= deadline {
                return Err(timeout_err(&last));
            }
            if !self.pump_servo(&webview, Duration::from_millis(16), deadline) {
                if Instant::now() >= deadline {
                    return Err(timeout_err(&last));
                }
                // One missed pump is not a dead event loop: an 80ms evaluate
                // window can close while the script thread is merely busy,
                // and the very next attempt succeeds (the first fill after a
                // fresh page failed exactly this way while a retry took
                // 295ms). Keep trying until the caller's deadline; only note
                // the stall for the eventual timeout message.
                last = format!("event_loop_slow; {last}");
                continue;
            }
        }
    }

    /// Screenshot after the page's rendering is COMPLETE — the opt-in behind
    /// `renderComplete`. Drives Servo's readiness machine, which waits for
    /// the load event, every image, and every web font before capturing.
    /// This is the right tool when the agent wants "how does it look
    /// finished"; the default [`Self::screenshot_png`] answers "how does it
    /// look now" and never blocks on a page that keeps streaming assets.
    fn screenshot_png_render_complete(
        &self,
        webview: &WebView,
        clip: Option<(u32, u32, u32, u32)>,
    ) -> io::Result<Vec<u8>> {
        webview.paint();
        self.rendering_context.present();
        let saved = Rc::new(RefCell::new(None));
        let callback = Rc::clone(&saved);
        webview.take_screenshot(None, move |result| {
            *callback.borrow_mut() = Some(result);
        });
        let pending = Rc::clone(&saved);
        if !self.spin_until(ACTION_TIMEOUT, move || pending.borrow().is_some())? {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for complete rendering; retry without renderComplete for an instant screenshot",
            ));
        }
        let image = saved
            .borrow_mut()
            .take()
            .expect("screenshot completed")
            .map_err(|error| io::Error::other(format!("screenshot failed: {error:?}")))?;
        encode_screenshot_png(image, clip)
    }

    fn screenshot_png(
        &self,
        webview: &WebView,
        clip: Option<(u32, u32, u32, u32)>,
    ) -> io::Result<Vec<u8>> {
        // An agent's screenshot means "what does the page look like NOW".
        // Servo's WebView::take_screenshot answers a different question -
        // "what does the page look like once it is finished" - because its
        // readiness machine waits for load, every image, and every web font
        // (reftest semantics). On a page that keeps streaming assets that
        // moment never comes and the agent gets a timeout instead of a
        // picture; Chromium happily screenshots mid-load. So paint and read
        // the framebuffer directly.
        //
        // Read BEFORE present: read_to_image reads the back buffer, which
        // holds the freshly painted frame until present swaps it away. The
        // present afterwards keeps the swap chain producing frames so
        // locator actionability never sees `stable` (event_loop_stalled).
        webview.paint();
        let size = self.rendering_context.size2d();
        let rect = servo::DeviceIntRect::from_size(servo::DeviceIntSize::new(
            size.width as i32,
            size.height as i32,
        ));
        // `read_to_image` returns None only when nothing has rendered yet
        // (a page that has not produced its first frame). Give that first
        // frame a short window instead of the old 30s readiness wait.
        let mut image = self.rendering_context.read_to_image(rect);
        self.rendering_context.present();
        if image.is_none() {
            let deadline = Instant::now() + Duration::from_secs(2);
            while image.is_none() && Instant::now() < deadline {
                self.servo.spin_event_loop();
                webview.paint();
                image = self.rendering_context.read_to_image(rect);
                self.rendering_context.present();
            }
        }
        let image = image.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out capturing screenshot: no frame rendered",
            )
        })?;
        encode_screenshot_png(image, clip)
    }
}

fn encode_screenshot_png(
    image: RgbaImage,
    clip: Option<(u32, u32, u32, u32)>,
) -> io::Result<Vec<u8>> {
    let image = match clip {
        Some((x, y, w, h)) => crop_rgba(&image, x, y, w, h),
        None => image,
    };
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, image.width(), image.height());
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|error| io::Error::other(format!("png header: {error}")))?;
        writer
            .write_image_data(image.as_raw())
            .map_err(|error| io::Error::other(format!("png data: {error}")))?;
    }
    Ok(out)
}

fn crop_rgba(image: &RgbaImage, x: u32, y: u32, width: u32, height: u32) -> RgbaImage {
    let src_w = image.width();
    let src_h = image.height();
    let x = x.min(src_w.saturating_sub(1));
    let y = y.min(src_h.saturating_sub(1));
    let width = width.min(src_w.saturating_sub(x)).max(1);
    let height = height.min(src_h.saturating_sub(y)).max(1);
    let src = image.as_raw();
    let mut out = vec![0_u8; (width * height * 4) as usize];
    for row in 0..height {
        let src_off = (((y + row) * src_w + x) * 4) as usize;
        let dst_off = (row * width * 4) as usize;
        let span = (width * 4) as usize;
        out[dst_off..dst_off + span].copy_from_slice(&src[src_off..src_off + span]);
    }
    RgbaImage::from_raw(width, height, out).unwrap_or_else(|| image.clone())
}

struct ResolvedNode {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    offset_left: f64,
    offset_top: f64,
}

const SELECTOR_RUNTIME: &str = r#"
function greppyAccessibleName(el) {
  const labelled = el.getAttribute('aria-label');
  if (labelled) return labelled.trim();
  if (el.id) {
    const by = document.querySelector('label[for="' + el.id + '"]');
    if (by) return (by.textContent || '').trim();
  }
  return ((el.innerText || el.textContent || el.value || '') + '').trim();
}
function greppyRoleOf(el) {
  const explicit = el.getAttribute('role');
  if (explicit) return explicit;
  const tag = el.tagName.toLowerCase();
  if (tag === 'button') return 'button';
  if (tag === 'input' && (el.type === 'button' || el.type === 'submit' || el.type === 'reset')) return 'button';
  if (tag === 'a' && el.hasAttribute('href')) return 'link';
  if (tag === 'input' || tag === 'textarea') return 'textbox';
  return tag;
}
function greppyIsDisplayed(el) {
  var n = el;
  while (n && n.nodeType === 1) {
    var style = getComputedStyle(n);
    if (style && (style.display === "none" || style.visibility === "hidden" || style.visibility === "collapse")) {
      return false;
    }
    n = n.parentElement;
  }
  var rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}
function greppyQueryAll(root, sel) {
  var visible = null;
  var css = String(sel);
  if (css.indexOf(":visible") !== -1) {
    visible = true;
    css = css.split(":visible").join("");
  }
  if (css.indexOf(":hidden") !== -1) {
    visible = false;
    css = css.split(":hidden").join("");
  }
  css = css.replace(/\s{2,}/g, " ").trim();
  try {
    var ctx = root === document ? document : root;
    var nodes = css ? Array.from(ctx.querySelectorAll(css)) : [];
    if (visible === null) return nodes;
    return nodes.filter(function (el) {
      var shown = greppyIsDisplayed(el);
      return visible ? shown : !shown;
    });
  } catch (error) { return []; }
}
function greppyCandidates(root) {
  return greppyQueryAll(root === document ? document : root, '*');
}
function greppyResolveIn(root, selector) {
  if (selector.type === 'css') {
    return greppyQueryAll(root === document ? document : root, selector.value);
  }
  if (selector.type === 'xpath') {
    try {
      const ctx = root === document ? document : root;
      const result = document.evaluate(
        selector.value,
        ctx,
        null,
        XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
        null
      );
      const nodes = [];
      for (let i = 0; i < result.snapshotLength; i++) {
        nodes.push(result.snapshotItem(i));
      }
      return nodes;
    } catch (error) {
      return [];
    }
  }
  if (selector.type === 'label') {
    const labels = greppyQueryAll(root === document ? document : root, 'label');
    const match = labels.find((label) => (label.textContent || '').trim() === selector.name);
    if (!match) return [];
    if (match.control) return [match.control];
    if (match.htmlFor) {
      const el = document.getElementById(match.htmlFor);
      return el ? [el] : [];
    }
    const nested = match.querySelector("input, textarea, select, button");
    return nested ? [nested] : [];
  }
  const pool = greppyCandidates(root);
  if (selector.type === 'role') {
    return pool.filter((el) => {
      if (greppyRoleOf(el) !== selector.role) return false;
      if (selector.name == null) return true;
      return greppyAccessibleName(el) === selector.name;
    });
  }
  if (selector.type === 'text') {
    const wanted = selector.value;
    return pool.filter((el) => ((el.innerText || el.textContent || '') + '').trim() === wanted);
  }
  if (selector.type === 'placeholder') {
    return pool.filter((el) => (el.getAttribute('placeholder') || '') === selector.name);
  }
  if (selector.type === 'alt') {
    return pool.filter((el) => (el.getAttribute('alt') || '') === selector.name);
  }
  if (selector.type === 'title') {
    return pool.filter((el) => (el.getAttribute('title') || '') === selector.name);
  }
  if (selector.type === 'testid') {
    const attr = selector.attr || 'data-testid';
    return pool.filter((el) => (el.getAttribute(attr) || '') === selector.name);
  }
  if (selector.type === 'filter') {
    return pool;
  }
  if (selector.type === 'framecss' || selector.type === 'frametext' || selector.type === 'framerole') {
    const framesAll = greppyQueryAll(root === document ? document : root, selector.frame || 'iframe');
    let frames = framesAll;
    if (selector.frameIndex != null) {
      const idx = selector.frameIndex < 0 ? framesAll.length + selector.frameIndex : selector.frameIndex;
      const frame = framesAll[idx];
      frames = frame ? [frame] : [];
    }
    let nodes = [];
    for (let i = 0; i < frames.length; i++) {
      try {
        const doc = frames[i].contentDocument;
        if (!doc) continue;
        if (selector.type === 'framecss') {
          nodes = nodes.concat(Array.from(doc.querySelectorAll(selector.value)));
        } else if (selector.type === 'frametext') {
          const wanted = selector.value;
          nodes = nodes.concat(Array.from(doc.querySelectorAll('body *')).filter((el) => {
            return ((el.innerText || el.textContent || '') + '').trim() === wanted;
          }));
        } else {
          const role = selector.role;
          const name = selector.name;
          nodes = nodes.concat(Array.from(doc.querySelectorAll('body *')).filter((el) => {
            if (greppyRoleOf(el) !== role) return false;
            if (name == null) return true;
            return greppyAccessibleName(el) === name;
          }));
        }
      } catch (error) {}
    }
    return nodes;
  }
  return [];
}
function greppyResolveNodes(selector) {
  if (selector.type === 'filter') {
    let nodes = greppyResolveNodes(selector.scope);
    if (selector.hasText) {
      const wanted = String(selector.hasText);
      nodes = nodes.filter((el) => ((el.innerText || el.textContent || '') + '').indexOf(wanted) !== -1);
    }
    if (selector.has) {
      nodes = nodes.filter((el) => greppyResolveIn(el, selector.has).length > 0);
    }
    if (selector.hasNot) {
      nodes = nodes.filter((el) => greppyResolveIn(el, selector.hasNot).length === 0);
    }
    if (selector.nth != null) {
      const idx = selector.nth < 0 ? nodes.length + selector.nth : selector.nth;
      const el = nodes[idx];
      return el ? [el] : [];
    }
    return nodes;
  }
  let roots = [document];
  if (selector.scope) {
    roots = greppyResolveNodes(selector.scope);
    if (!roots.length) return [];
  }
  let nodes = [];
  for (let i = 0; i < roots.length; i++) {
    nodes = nodes.concat(greppyResolveIn(roots[i], selector));
  }
  if (selector.nth != null) {
    const idx = selector.nth < 0 ? nodes.length + selector.nth : selector.nth;
    const el = nodes[idx];
    return el ? [el] : [];
  }
  return nodes;
}
"#;

fn resolve_script(selector: &serde_json::Value) -> String {
    format!(
        "(function(selector) {{ {SELECTOR_RUNTIME}
          function greppyStyleHidden(el) {{
            var n = el;
            while (n && n.nodeType === 1) {{
              var style = getComputedStyle(n);
              if (style && (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse')) {{
                return true;
              }}
              n = n.parentElement;
            }}
            return false;
          }}
          function greppyHitTarget(el, rect) {{
            if (rect.width <= 0 || rect.height <= 0) return false;
            var doc = el.ownerDocument || document;
            var top = doc.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
            if (!top) return false;
            return el === top || el.contains(top);
          }}
          if (selector.snapshot != null &&
              (!document.documentElement ||
               document.documentElement.getAttribute('data-greppy-ref-snapshot') !== selector.snapshot)) {{
            return {{ staleRef: true, count: 0 }};
          }}
          const nodes = greppyResolveNodes(selector);
          if (selector.snapshot != null && nodes.length !== 1) {{
            return {{ staleRef: true, count: nodes.length }};
          }}
          if (nodes.length !== 1) {{
            return {{ count: nodes.length, x: 0, y: 0, width: 0, height: 0, disabled: false, readonly: false, visible: false, hit: false }};
          }}
          const el = nodes[0];
          void document.body.offsetHeight;
          var rect = el.getBoundingClientRect();
          var hidden = greppyStyleHidden(el);
          var hit = greppyHitTarget(el, rect);
          if (!hidden && (!hit || rect.width <= 0 || rect.height <= 0) && el.scrollIntoView) {{
            el.scrollIntoView({{ block: 'nearest', inline: 'nearest' }});
            void document.body.offsetHeight;
            rect = el.getBoundingClientRect();
            hidden = greppyStyleHidden(el);
            hit = greppyHitTarget(el, rect);
          }}
          var visible = !hidden && rect.width > 0 && rect.height > 0;
          return {{
            count: 1,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            offsetLeft: el.offsetLeft,
            offsetTop: el.offsetTop,
            disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
            readonly: !!(el.readOnly || el.getAttribute('readonly') != null || el.getAttribute('aria-readonly') === 'true'),
            visible: visible,
            hit: !!(hit && visible)
          }};
        }})({selector})"
    )
}

fn inner_text_script(selector: &serde_json::Value) -> String {
    format!(
        "(function(selector) {{ {SELECTOR_RUNTIME}
          const nodes = greppyResolveNodes(selector);
          if (nodes.length !== 1) throw new Error('strict mode: expected 1 node, got ' + nodes.length);
          return nodes[0].innerText;
        }})({selector})"
    )
}

fn fill_script(selector: &serde_json::Value, value: &str) -> String {
    let value = serde_json::to_string(value).expect("string JSON");
    format!(
        "(function(selector, value) {{ {SELECTOR_RUNTIME}
          const nodes = greppyResolveNodes(selector);
          if (nodes.length !== 1) throw new Error('strict mode: expected 1 node, got ' + nodes.length);
          const el = nodes[0];
          el.focus();
          el.value = value;
          el.dispatchEvent(new Event('input', {{ bubbles: true }}));
          el.dispatchEvent(new Event('change', {{ bubbles: true }}));
          return true;
        }})({selector}, {value})"
    )
}

/// Per-navigation phase timing, printed to stderr when `GREPPY_WEB_TRACE_NAV`
/// is set. Off by default: this exists to attribute the fixed per-navigation
/// overhead (finding 020) to a phase — url-settled, HeadParsed, Complete —
/// not to run in production.
struct NavTrace {
    started: Option<Instant>,
    settled_ms: Option<u128>,
    head_parsed_ms: Option<u128>,
    complete_ms: Option<u128>,
}

impl NavTrace {
    fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("GREPPY_WEB_TRACE_NAV").is_some())
    }

    fn begin() -> Self {
        Self {
            started: Self::enabled().then(Instant::now),
            settled_ms: None,
            head_parsed_ms: None,
            complete_ms: None,
        }
    }

    fn note(&mut self, webview: &WebView, url_settled: &mut impl FnMut() -> bool) {
        let Some(started) = self.started else { return };
        let elapsed = started.elapsed().as_millis();
        if self.settled_ms.is_none() && url_settled() {
            self.settled_ms = Some(elapsed);
        }
        match webview.load_status() {
            LoadStatus::HeadParsed if self.head_parsed_ms.is_none() => {
                self.head_parsed_ms = Some(elapsed);
            }
            LoadStatus::Complete if self.complete_ms.is_none() => {
                self.complete_ms = Some(elapsed);
            }
            _ => {}
        }
    }

    fn finish(&mut self, webview: &WebView) {
        let Some(started) = self.started else { return };
        if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: nav-trace settled_ms={:?} head_parsed_ms={:?} complete_ms={:?} commit_ms={} url={:?}",
            self.settled_ms,
            self.head_parsed_ms,
            self.complete_ms,
            started.elapsed().as_millis(),
            webview.url().map(|u| u.to_string()),
        ); }
    }
}

/// Engine preferences for a content worker reachable only through `proxy_uri`.
///
/// Servo ships several DOM features switched off by preference. Leaving
/// `IntersectionObserver` off is not a neutral default here: every modern
/// framework touches it during hydration, and the resulting `ReferenceError`
/// takes the whole page down rather than degrading one feature.
fn engine_preferences(proxy_uri: &str) -> Preferences {
    let mut preferences = Preferences::default();
    preferences.network_http_proxy_uri = proxy_uri.to_owned();
    preferences.network_https_proxy_uri = proxy_uri.to_owned();
    preferences.network_http_no_proxy = String::new();
    preferences.network_enforce_tls_enabled = false;
    preferences.dom_intersection_observer_enabled = true;
    preferences
}

fn hover_at(
    webview: &WebView,
    delegate: &Delegate,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    spin: &mut impl FnMut(),
) -> io::Result<()> {
    let point = WebViewPoint::Device(DevicePoint::new(
        (x + width / 2.0) as f32,
        (y + height / 2.0) as f32,
    ));
    dispatch_input_and_wait(
        webview,
        delegate,
        || InputEvent::MouseMove(MouseMoveEvent::new(point)),
        spin,
    )
}

fn tap_at(
    webview: &WebView,
    delegate: &Delegate,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    mut spin: impl FnMut(),
) -> io::Result<()> {
    let point = WebViewPoint::Device(DevicePoint::new(
        (x + width / 2.0) as f32,
        (y + height / 2.0) as f32,
    ));
    let id = TouchId(1);
    dispatch_input_and_wait(
        webview,
        delegate,
        || InputEvent::Touch(TouchEvent::new(
            TouchEventType::Down,
            id,
            point,
            TouchPointerType::Touch,
        )),
        &mut spin,
    )?;
    dispatch_input_and_wait(
        webview,
        delegate,
        || InputEvent::Touch(TouchEvent::new(
            TouchEventType::Up,
            id,
            point,
            TouchPointerType::Touch,
        )),
        &mut spin,
    )
}

fn dispatch_input_and_wait(
    webview: &WebView,
    delegate: &Delegate,
    make_event: impl Fn() -> InputEvent,
    spin: &mut impl FnMut(),
) -> io::Result<()> {
    for _attempt in 0..8 {
        let event_id = webview.notify_input_event(make_event());
        let deadline = Instant::now() + Duration::from_millis(500);
        loop {
            spin();
            if let Some(result) = delegate.input_receipts.borrow_mut().remove(&event_id) {
                if !result.contains(InputEventResult::DispatchFailed) {
                    return Ok(());
                }
                break;
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::yield_now();
        }
        webview.paint();
        spin();
    }
    Err(io::Error::other(
        "input delivery failed: painter dropped or did not acknowledge the event after 8 retries",
    ))
}

/// Synthesize a left click at the centre of a box.
///
/// `spin` must drive the engine's event loop; it is called between the move,
/// the press and the release. Delivering all three in one batch leaves the
/// hit test unresolved and no `click` event reaches the DOM at all — a
/// document-level capture listener sees nothing, while `hover` alone works.
/// The drag path already spins between press and release for the same reason.
fn click_at(
    webview: &WebView,
    delegate: &Delegate,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    mut spin: impl FnMut(),
) -> io::Result<()> {
    hover_at(webview, delegate, x, y, width, height, &mut spin)?;
    let point = WebViewPoint::Device(DevicePoint::new(
        (x + width / 2.0) as f32,
        (y + height / 2.0) as f32,
    ));
    dispatch_input_and_wait(
        webview,
        delegate,
        || InputEvent::MouseButton(MouseButtonEvent::new(
            MouseButtonAction::Down,
            MouseButton::Left,
            point,
        )),
        &mut spin,
    )?;
    dispatch_input_and_wait(
        webview,
        delegate,
        || InputEvent::MouseButton(MouseButtonEvent::new(
            MouseButtonAction::Up,
            MouseButton::Left,
            point,
        )),
        &mut spin,
    )
}

fn load_status_allows_navigation(status: LoadStatus, ready_state: Option<&str>) -> bool {
    match status {
        LoadStatus::Complete => true,
        LoadStatus::HeadParsed => matches!(ready_state, Some("complete") | Some("interactive")),
        LoadStatus::Started => false,
    }
}

fn urls_match(current: &Url, expected: &Url) -> bool {
    current.scheme() == expected.scheme()
        && current.host() == expected.host()
        && current.port_or_known_default() == expected.port_or_known_default()
        && current.path().trim_end_matches('/') == expected.path().trim_end_matches('/')
}

fn required_str(params: &serde_json::Value, key: &str) -> io::Result<String> {
    params
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("engine call missing string field {key}"),
            )
        })
}

fn number_field(values: &HashMap<String, JSValue>, key: &str) -> io::Result<f64> {
    match values.get(key) {
        Some(JSValue::Number(value)) => Ok(*value),
        other => Err(io::Error::other(format!(
            "expected number {key}, got {other:?}"
        ))),
    }
}

fn bool_field(values: &HashMap<String, JSValue>, key: &str) -> io::Result<bool> {
    match values.get(key) {
        Some(JSValue::Boolean(value)) => Ok(*value),
        other => Err(io::Error::other(format!(
            "expected bool {key}, got {other:?}"
        ))),
    }
}

fn jsvalue_to_json(value: JSValue) -> serde_json::Value {
    match value {
        JSValue::Undefined | JSValue::Null => serde_json::Value::Null,
        JSValue::Boolean(value) => serde_json::Value::Bool(value),
        JSValue::Number(value) => serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        JSValue::String(value) => serde_json::Value::String(value),
        JSValue::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(jsvalue_to_json).collect())
        }
        JSValue::Object(values) => serde_json::Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, jsvalue_to_json(value)))
                .collect(),
        ),
        JSValue::Element(value)
        | JSValue::ShadowRoot(value)
        | JSValue::Frame(value)
        | JSValue::Window(value) => serde_json::Value::String(value),
    }
}

fn jsvalue_is_truthy(value: &JSValue) -> bool {
    match value {
        JSValue::Undefined | JSValue::Null => false,
        JSValue::Boolean(value) => *value,
        JSValue::Number(value) => *value != 0.0 && !value.is_nan(),
        JSValue::String(value) => !value.is_empty(),
        JSValue::Array(_)
        | JSValue::Object(_)
        | JSValue::Element(_)
        | JSValue::ShadowRoot(_)
        | JSValue::Frame(_)
        | JSValue::Window(_) => true,
    }
}
fn evaluate_serialized(value: JSValue) -> io::Result<serde_json::Value> {
    Ok(json!({ "serialized": serialize_jsvalue(value)? }))
}

/// waitForFunction must keep JSON values (including `{answer:42}`) and still
/// return a truthy stand-in for host objects. `page.evaluate` of an Element
/// stays undefined; a wait predicate that returned a node is not `undefined`.
fn serialize_wait_value(value: JSValue) -> io::Result<serde_json::Value> {
    match value {
        JSValue::Element(_) | JSValue::ShadowRoot(_) | JSValue::Frame(_) | JSValue::Window(_) => {
            Ok(json!({ "serialized": { "o": [] } }))
        }
        other => evaluate_serialized(other),
    }
}

fn serialize_jsvalue(value: JSValue) -> io::Result<serde_json::Value> {
    match value {
        JSValue::Undefined => Ok(json!({ "v": "undefined" })),
        JSValue::Null => Ok(json!({ "v": "null" })),
        JSValue::Boolean(value) => Ok(json!({ "b": value })),
        JSValue::Number(value) => {
            if value.is_nan() {
                Ok(json!({ "v": "NaN" }))
            } else if value.is_infinite() {
                Ok(json!({
                    "v": if value.is_sign_positive() {
                        "Infinity"
                    } else {
                        "-Infinity"
                    }
                }))
            } else if value == 0.0 && value.is_sign_negative() {
                Ok(json!({ "v": "-0" }))
            } else {
                Ok(json!({ "n": value }))
            }
        }
        JSValue::String(value) => Ok(json!({ "s": value })),
        JSValue::Array(values) => {
            let mut encoded = Vec::with_capacity(values.len());
            for item in values {
                encoded.push(serialize_jsvalue(item)?);
            }
            Ok(json!({ "a": encoded }))
        }
        JSValue::Object(values) => {
            let mut encoded = Vec::with_capacity(values.len());
            for (key, item) in values {
                encoded.push(json!({ "k": key, "v": serialize_jsvalue(item)? }));
            }
            Ok(json!({ "o": encoded }))
        }
        JSValue::Element(_) | JSValue::ShadowRoot(_) | JSValue::Frame(_) | JSValue::Window(_) => {
            Ok(json!({ "v": "undefined" }))
        }
    }
}

#[cfg(test)]
mod serialize_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn serialize_jsvalue_keeps_undefined_and_non_finite_distinct() {
        assert_eq!(
            serialize_jsvalue(JSValue::Undefined).unwrap(),
            json!({ "v": "undefined" })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Null).unwrap(),
            json!({ "v": "null" })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Number(f64::NAN)).unwrap(),
            json!({ "v": "NaN" })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Number(f64::INFINITY)).unwrap(),
            json!({ "v": "Infinity" })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Number(f64::NEG_INFINITY)).unwrap(),
            json!({ "v": "-Infinity" })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Number(-0.0)).unwrap(),
            json!({ "v": "-0" })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Boolean(true)).unwrap(),
            json!({ "b": true })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Number(42.0)).unwrap(),
            json!({ "n": 42.0 })
        );
        assert_eq!(
            serialize_jsvalue(JSValue::Element("node".into())).unwrap(),
            json!({ "v": "undefined" })
        );
        assert_eq!(
            serialize_wait_value(JSValue::Element("node".into())).unwrap(),
            json!({ "serialized": { "o": [] } })
        );
        assert_eq!(
            serialize_wait_value(JSValue::Object(
                [("answer".into(), JSValue::Number(42.0))]
                    .into_iter()
                    .collect()
            ))
            .unwrap()["serialized"]["o"][0]["k"],
            json!("answer")
        );
    }

    #[test]
    fn engine_enables_intersection_observer_and_pins_the_proxy() {
        let preferences = engine_preferences("http://127.0.0.1:4242");
        // Off by default in Servo; every modern framework dies without it.
        assert!(preferences.dom_intersection_observer_enabled);
        assert_eq!(preferences.network_http_proxy_uri, "http://127.0.0.1:4242");
        assert_eq!(preferences.network_https_proxy_uri, "http://127.0.0.1:4242");
        assert!(preferences.network_http_no_proxy.is_empty());
    }

    #[test]
    fn headparsed_with_interactive_ready_state_commits_navigation() {
        assert!(load_status_allows_navigation(
            LoadStatus::Complete,
            None
        ));
        assert!(load_status_allows_navigation(
            LoadStatus::HeadParsed,
            Some("interactive")
        ));
        assert!(load_status_allows_navigation(
            LoadStatus::HeadParsed,
            Some("complete")
        ));
        assert!(!load_status_allows_navigation(
            LoadStatus::HeadParsed,
            Some("loading")
        ));
        assert!(!load_status_allows_navigation(LoadStatus::HeadParsed, None));
        assert!(!load_status_allows_navigation(
            LoadStatus::Started,
            Some("complete")
        ));
    }

    #[test]
    fn object_disposed_error_names_the_kind() {
        let err = object_disposed("Page");
        assert!(
            err.to_string()
                .contains("object_disposed: Page has been closed"),
            "{err}"
        );
    }

    #[test]
    fn reject_generation_compares_wanted_against_stored() {
        assert!(reject_generation("Page", 1, Some(1), true).is_ok());
        assert!(reject_generation("Page", 1, None, true).is_ok());
        let stale = reject_generation("Page", 1, Some(9), true).unwrap_err();
        assert!(
            stale
                .to_string()
                .contains("object_disposed: Page has been closed (generation 1)"),
            "{stale}"
        );
        let disposed = reject_generation("BrowserContext", 4, Some(4), false).unwrap_err();
        assert!(
            disposed
                .to_string()
                .contains("object_disposed: BrowserContext has been closed (generation 4)"),
            "{disposed}"
        );
        let disposed_mismatch = reject_generation("Browser", 2, Some(8), false).unwrap_err();
        assert!(
            disposed_mismatch
                .to_string()
                .contains("object_disposed: Browser has been closed (generation 2)"),
            "{disposed_mismatch}"
        );
    }

    #[test]
    fn pump_tokens_are_distinct_monotonic_nonces() {
        let nonce = Cell::new(1);
        let first = alloc_pump_token(&nonce);
        let second = alloc_pump_token(&nonce);
        let third = alloc_pump_token(&nonce);
        assert_eq!(first, "__greppyPump1");
        assert_eq!(second, "__greppyPump2");
        assert_eq!(third, "__greppyPump3");
        assert_ne!(first, second);
        assert_ne!(second, third);
        let nonce = Cell::new(1);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10_000 {
            assert!(seen.insert(alloc_pump_token(&nonce)), "pump token repeated");
        }
    }

    #[test]
    fn pump_token_ledger_reclaims_after_repeated_timeout_cycles() {
        let nonce = Cell::new(1);
        let pending = RefCell::new(Vec::new());
        for cycle in 0..50 {
            let token = track_pump_token(&pending, &nonce);
            assert!(
                token.starts_with("__greppyPump"),
                "cycle {cycle} token {token}"
            );
        }
        assert_eq!(
            pending.borrow().len(),
            50,
            "timeouts must stay tracked until reclaim"
        );
        reclaim_all_tracked_tokens(&pending);
        assert!(
            pending.borrow().is_empty(),
            "recovery reclaim must drop the entire pending set"
        );
        let recovered = track_pump_token(&pending, &nonce);
        assert_eq!(
            pending.borrow().as_slice(),
            std::slice::from_ref(&recovered)
        );
        reclaim_tracked_token(&pending, &recovered);
        assert!(pending.borrow().is_empty());
        let later = track_pump_token(&pending, &nonce);
        assert_ne!(later, recovered);
    }

    #[test]
    fn jsvalue_is_truthy_matches_wait_for_function_contract() {
        assert!(!jsvalue_is_truthy(&JSValue::Undefined));
        assert!(!jsvalue_is_truthy(&JSValue::Null));
        assert!(!jsvalue_is_truthy(&JSValue::Boolean(false)));
        assert!(jsvalue_is_truthy(&JSValue::Boolean(true)));
        assert!(!jsvalue_is_truthy(&JSValue::Number(0.0)));
        assert!(jsvalue_is_truthy(&JSValue::Number(1.0)));
        assert!(!jsvalue_is_truthy(&JSValue::String(String::new())));
        assert!(jsvalue_is_truthy(&JSValue::String("ok".into())));
        assert!(jsvalue_is_truthy(&JSValue::Object(Default::default())));
    }

    #[test]
    fn parse_wait_done_signal_requires_crypto_nonce_and_status() {
        let nonce = "0123456789abcdef0123456789abcdef";
        assert_eq!(
            parse_wait_done_signal(&format!("__greppyWaitDone:{nonce}:ok")),
            Some((nonce, "ok"))
        );
        assert_eq!(
            parse_wait_done_signal(&format!("__greppyWaitDone:{nonce}:timeout")),
            Some((nonce, "timeout"))
        );
        assert_eq!(
            parse_wait_done_signal(&format!("__greppyWaitDone:{nonce}:error")),
            Some((nonce, "error"))
        );
        assert_eq!(parse_wait_done_signal("__greppyWait:__greppyWait3"), None);
        assert_eq!(parse_wait_done_signal("__greppyWaitDone:"), None);
        assert_eq!(parse_wait_done_signal("__greppyWaitDone:token:ok"), None);
        assert_eq!(
            parse_wait_done_signal(&format!("__greppyWaitDone:{nonce}:ok:extra")),
            None
        );
        assert_eq!(
            parse_wait_done_signal("__greppyWaitDone:0123456789ABCDEF0123456789ABCDEF:ok"),
            None
        );
        assert_eq!(
            parse_wait_done_signal(&format!("__greppyWaitDone:{nonce}:ready")),
            None
        );
    }

    #[test]
    fn wait_notices_keep_the_first_signal_for_a_nonce() {
        let a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let notices = RefCell::new(HashMap::<String, String>::new());
        for text in [
            format!("__greppyWaitDone:{a}:ok"),
            format!("__greppyWaitDone:{a}:timeout"),
            format!("__greppyWaitDone:{b}:timeout"),
        ] {
            if let Some((token, status)) = parse_wait_done_signal(&text) {
                notices
                    .borrow_mut()
                    .entry(token.to_owned())
                    .or_insert_with(|| status.to_owned());
            }
        }
        assert_eq!(notices.borrow().get(a).map(String::as_str), Some("ok"));
        assert_eq!(notices.borrow().get(b).map(String::as_str), Some("timeout"));
    }

    #[test]
    fn wait_for_recorded_concurrent_producer_does_not_drop_consumed_chooser() {
        // Load-bearing: a producer records+wakes after the waiter's first empty
        // take() and before poll_wake_step. Using take() as the wake predicate
        // (`take().is_some()`) would pop the chooser and drop it on Ready.
        let wake = WakeFlag::new();
        let queue = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let (arm_producer, wait_arm) = mpsc::channel::<()>();
        let (inserted, wait_inserted) = mpsc::channel::<()>();
        let producer_queue = Arc::clone(&queue);
        let producer_wake = wake.clone();
        let producer = std::thread::spawn(move || {
            wait_arm.recv().expect("arm");
            producer_queue
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(json!({ "multiple": false, "id": "chooser-1" }));
            producer_wake.wake();
            inserted.send(()).expect("inserted");
        });
        let takes = AtomicUsize::new(0);
        let wait_queue = Arc::clone(&queue);
        let result = wait_for_recorded_loop(
            &wake,
            Duration::from_secs(2),
            "timeout: waitForEvent filechooser",
            || Ok(()),
            |_| false,
            || {},
            || {
                !wait_queue
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .is_empty()
            },
            || {
                let n = takes.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    arm_producer.send(()).expect("arm producer");
                    wait_inserted.recv().expect("chooser recorded");
                    return None;
                }
                let mut choosers = wait_queue.lock().unwrap_or_else(|error| error.into_inner());
                if choosers.is_empty() {
                    None
                } else {
                    Some(choosers.remove(0))
                }
            },
        );
        producer.join().expect("producer");
        let hit = result.expect("chooser must be delivered, not dropped in the wake predicate");
        assert_eq!(hit["id"], json!("chooser-1"));
        assert!(
            queue
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty(),
            "chooser must be consumed once"
        );
        assert!(
            takes.load(Ordering::SeqCst) >= 2,
            "first take is the empty miss; later take consumes"
        );
    }

    #[test]
    fn alloc_wait_nonce_is_32_lowercase_hex_and_unique() {
        let first = alloc_wait_nonce().unwrap();
        let second = alloc_wait_nonce().unwrap();
        assert_eq!(first.len(), 32);
        assert_eq!(second.len(), 32);
        assert!(first
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f')));
        assert_ne!(first, second);
    }

    #[test]
    fn animation_frame_budget_caps_condvar_wait_at_16ms() {
        assert_eq!(
            animation_frame_budget(Duration::from_secs(30)),
            Duration::from_millis(16)
        );
        assert_eq!(
            animation_frame_budget(Duration::from_millis(5)),
            Duration::from_millis(5)
        );
        assert_eq!(animation_frame_budget(Duration::ZERO), Duration::ZERO);
        let wake = WakeFlag::new();
        let last = wake.generation();
        let t0 = Instant::now();
        assert!(!wake.wait_for_generation(last, animation_frame_budget(Duration::from_secs(5))));
        let elapsed = t0.elapsed();
        assert!(elapsed >= Duration::from_millis(10), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(80), "{elapsed:?}");
        assert_eq!(wake.generation(), last);
        assert_eq!(
            poll_wake_step(
                &wake,
                || false,
                animation_frame_budget(Duration::from_secs(1))
            ),
            WakePoll::TimedOut
        );
    }

    #[test]
    fn call_timeout_accepts_v8_f64_and_integer() {
        assert_eq!(
            call_timeout(&serde_json::json!({ "timeout": 250 })),
            Duration::from_millis(250)
        );
        let timeout =
            serde_json::Value::Number(serde_json::Number::from_f64(250.0).expect("finite"));
        assert_eq!(
            timeout.as_u64(),
            None,
            "precondition: V8 f64 250.0 is not as_u64"
        );
        assert_eq!(
            call_timeout(&serde_json::json!({ "timeout": timeout })),
            Duration::from_millis(250)
        );
        assert_eq!(
            call_timeout(&serde_json::json!({ "timeout": 0 })),
            Duration::from_millis(20)
        );
        assert_eq!(
            call_timeout(&serde_json::json!({ "timeout": "nope" })),
            ACTION_TIMEOUT
        );
    }

    #[test]
    fn wake_flag_blocks_on_condvar_until_generation_changes() {
        let wake = WakeFlag::new();
        let last = wake.generation();
        let waker = wake.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            waker.wake();
        });
        let started = Instant::now();
        assert!(wake.wait_for_generation(last, Duration::from_secs(1)));
        thread.join().unwrap();
        assert!(wake.generation() != last);
        assert!(started.elapsed() < Duration::from_millis(400));
        let current = wake.generation();
        let t0 = Instant::now();
        assert!(!wake.wait_for_generation(current, Duration::from_millis(40)));
        assert!(t0.elapsed() >= Duration::from_millis(20));
        assert!(wake.take_pending());
        assert!(!wake.take_pending());
    }

    #[test]
    fn poll_wake_does_not_lose_a_wake_that_arrives_during_predicate() {
        let wake = WakeFlag::new();
        let start = wake.generation();
        match poll_wake_step(
            &wake,
            || {
                wake.wake();
                false
            },
            Duration::from_millis(80),
        ) {
            WakePoll::NeedSpin { from, to } => {
                assert_eq!(from, start, "baseline must be the pre-predicate generation");
                assert_ne!(to, start, "wake during predicate must be visible");
            }
            other => panic!("lost wakeup: {other:?}"),
        }
    }

    #[test]
    fn poll_wake_timeout_without_wake_is_not_need_spin() {
        let wake = WakeFlag::new();
        let start = wake.generation();
        let t0 = Instant::now();
        assert_eq!(
            poll_wake_step(&wake, || false, Duration::from_millis(40)),
            WakePoll::TimedOut
        );
        assert_eq!(
            wake.generation(),
            start,
            "timeout without wake must not bump generation (no spin)"
        );
        assert!(t0.elapsed() >= Duration::from_millis(20));
        assert!(t0.elapsed() < Duration::from_millis(400));
    }

    #[test]
    fn wait_for_generation_sees_wakes_that_already_happened() {
        let wake = WakeFlag::new();
        let last = wake.generation();
        wake.wake();
        wake.wake();
        assert!(
            wake.wait_for_generation(last, Duration::ZERO),
            "already-advanced generation must not wait"
        );
        assert_eq!(wake.generation(), last.wrapping_add(2));
    }

    #[test]
    fn poll_wake_sees_multiple_wakes_during_predicate_as_one_need_spin() {
        let wake = WakeFlag::new();
        let start = wake.generation();
        match poll_wake_step(
            &wake,
            || {
                wake.wake();
                wake.wake();
                false
            },
            Duration::ZERO,
        ) {
            WakePoll::NeedSpin { from, to } => {
                assert_eq!(from, start);
                assert_eq!(to, start.wrapping_add(2));
            }
            other => panic!("multiple wakes must be NeedSpin, got {other:?}"),
        }
    }

    #[test]
    fn poll_wake_spurious_notify_without_generation_times_out() {
        let wake = WakeFlag::new();
        let start = wake.generation();
        let waker = wake.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            waker.notify_without_generation();
        });
        assert_eq!(
            poll_wake_step(&wake, || false, Duration::from_millis(50)),
            WakePoll::TimedOut
        );
        thread.join().unwrap();
        assert_eq!(wake.generation(), start);
    }

    #[test]
    fn poll_wake_ready_predicate_does_not_wait() {
        let wake = WakeFlag::new();
        let start = Instant::now();
        assert_eq!(
            poll_wake_step(&wake, || true, Duration::from_secs(5)),
            WakePoll::Ready
        );
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[test]
    fn poll_wake_deadline_rechecks_predicate_without_treating_it_as_spin() {
        let wake = WakeFlag::new();
        let start = wake.generation();
        let mut calls = 0;
        assert_eq!(
            poll_wake_step(
                &wake,
                || {
                    calls += 1;
                    calls > 1
                },
                Duration::from_millis(25),
            ),
            WakePoll::Ready
        );
        assert_eq!(calls, 2, "deadline must recheck predicate without a wake");
        assert_eq!(wake.generation(), start);
    }

    #[test]
    fn oversized_screenshot_engine_result_uses_sidecar_file() {
        let small = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
        let small_value = screenshot_engine_result(&small).expect("small screenshot");
        assert!(
            small_value.get("png_base64").and_then(|v| v.as_str()).is_some(),
            "small PNG must stay inline: {small_value:?}"
        );
        assert!(small_value.get("png_path").is_none());
        let mut small_frame = Vec::new();
        write_message(
            &mut small_frame,
            &Message::engine_result(1, true, small_value, None),
        )
        .expect("small screenshot must fit the worker frame");

        let large = vec![0xA5; 900_000];
        let large_inline = json!({
            "png_base64": base64_encode(&large),
            "byte_count": large.len(),
        });
        assert!(
            engine_result_frame_len(&large_inline) > MAX_FRAME_BYTES,
            "900 KiB PNG base64 must exceed the 1 MiB worker frame"
        );
        let mut too_big = Vec::new();
        let inline_err = write_message(
            &mut too_big,
            &Message::engine_result(1, true, large_inline, None),
        )
        .expect_err("inline oversized PNG must be refused");
        assert!(
            inline_err.to_string().contains("exceeds"),
            "{inline_err}"
        );

        let large_value = screenshot_engine_result(&large).expect("large screenshot");
        let path = large_value
            .get("png_path")
            .and_then(|v| v.as_str())
            .expect("sidecar path")
            .to_owned();
        assert!(
            large_value.get("png_base64").is_none(),
            "oversized PNG must not ride the frame as base64: {large_value:?}"
        );
        assert_eq!(
            std::fs::read(&path).expect("sidecar bytes"),
            large
        );
        let mut large_frame = Vec::new();
        write_message(
            &mut large_frame,
            &Message::engine_result(1, true, large_value, None),
        )
        .expect("sidecar screenshot must fit the worker frame");
        let _ = std::fs::remove_file(&path);
    }
}

const OBSERVE_JS: &str = r#"(function(snapshot) {
  const refAttr = 'data-greppy-ref';
  const snapshotAttr = 'data-greppy-ref-snapshot';
  if (snapshot != null) {
    Array.from(document.querySelectorAll('[' + refAttr + ']')).forEach(function(node) {
      node.removeAttribute(refAttr);
    });
    if (document.documentElement) document.documentElement.setAttribute(snapshotAttr, snapshot);
  }
  const candidates = snapshot == null ? [] : Array.from(document.querySelectorAll(
    'a[href],button,input,select,textarea,summary,[role="button"],[role="link"],[contenteditable="true"]'
  )).filter(function(node) {
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  });
  const capped = candidates.slice(0, 200);
  const actionables = capped.map(function(node, index) {
    const ref = index + 1;
    node.setAttribute(refAttr, snapshot + ':' + ref);
    const text = ((node.innerText || node.textContent || node.value || '') + '').trim().slice(0, 160);
    return {
      ref: '@' + ref,
      tag: node.tagName.toLowerCase(),
      role: node.getAttribute('role') || null,
      name: (node.getAttribute('aria-label') || '').trim() || null,
      text: text,
      href: node.href || null,
      disabled: !!(node.disabled || node.getAttribute('aria-disabled') === 'true')
    };
  });
  return JSON.stringify({
    url: location.href,
    title: document.title,
    text: ((document.body && document.body.innerText) || '').slice(0, 8000),
    headings: Array.from(document.querySelectorAll('h1,h2,h3,h4')).map(function(h) {
      return (h.innerText || '').trim();
    }).filter(Boolean).slice(0, 20),
    links: Array.from(document.querySelectorAll('a[href]')).slice(0, 20).map(function(a) {
      return { href: a.href, text: ((a.innerText || '').trim()).slice(0, 80) };
    }),
    actionables: actionables,
    ref_count: actionables.length,
    refs_truncated: candidates.length > capped.length
  });
})(__GREPPY_SNAPSHOT__)"#;

fn observe_script(snapshot: Option<&str>) -> String {
    let encoded = serde_json::to_string(&snapshot).expect("snapshot token serializes");
    OBSERVE_JS.replace("__GREPPY_SNAPSHOT__", &encoded)
}

static CLICK_PROBE_SEQ: AtomicU64 = AtomicU64::new(1);
static WHEEL_PROBE_SEQ: AtomicU64 = AtomicU64::new(1);
static SCREENSHOT_SIDECAR_SEQ: AtomicU64 = AtomicU64::new(1);

fn engine_result_frame_len(result: &serde_json::Value) -> usize {
    serde_json::to_vec(&Message::engine_result(0, true, result.clone(), None))
        .map(|bytes| bytes.len())
        .unwrap_or(usize::MAX)
}

fn write_screenshot_sidecar(png: &[u8]) -> io::Result<PathBuf> {
    let seq = SCREENSHOT_SIDECAR_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "greppy-web-shot-{}-{seq}.png",
        std::process::id()
    ));
    let confined = confine_worker_path(&path)?;
    std::fs::write(&confined, png)?;
    Ok(confined)
}

fn screenshot_engine_result(png: &[u8]) -> io::Result<serde_json::Value> {
    let inline = json!({
        "png_base64": base64_encode(png),
        "byte_count": png.len(),
    });
    if engine_result_frame_len(&inline) <= MAX_FRAME_BYTES {
        return Ok(inline);
    }
    let path = write_screenshot_sidecar(png)?;
    Ok(json!({
        "png_path": path.to_string_lossy(),
        "byte_count": png.len(),
    }))
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let a = chunk[0] as u32;
        let b = chunk.get(1).copied().unwrap_or(0) as u32;
        let c = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (a << 16) | (b << 8) | c;
        out.push(TABLE[((triple >> 18) & 63) as usize] as char);
        out.push(TABLE[((triple >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(input: &str) -> io::Result<Vec<u8>> {
    fn sextet(byte: u8) -> io::Result<u32> {
        match byte {
            b'A'..=b'Z' => Ok(u32::from(byte - b'A')),
            b'a'..=b'z' => Ok(u32::from(byte - b'a') + 26),
            b'0'..=b'9' => Ok(u32::from(byte - b'0') + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid base64 body",
            )),
        }
    }
    let chars: Vec<u8> = input
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    if chars.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid base64 length",
        ));
    }
    let mut out = Vec::with_capacity(chars.len() / 4 * 3);
    for chunk in chars.chunks(4) {
        let pad = chunk.iter().filter(|byte| **byte == b'=').count();
        let a = sextet(chunk[0])?;
        let b = sextet(chunk[1])?;
        let c = if chunk[2] == b'=' {
            0
        } else {
            sextet(chunk[2])?
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            sextet(chunk[3])?
        };
        let n = (a << 18) | (b << 12) | (c << 6) | d;
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn is_parent_eof(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::UnexpectedEof | io::ErrorKind::BrokenPipe
    )
}

pub fn run() -> io::Result<()> {
    let capability = require_worker_auth(std::env::args_os().skip(1))?;
    crate::supervisor::apply_worker_sandbox(
        &std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("/")),
        &std::env::temp_dir(),
    )?;
    let (mut protocol_in, mut protocol_out) = crate::worker::take_protocol_channel()?;
    let parent_alive = Arc::new(AtomicBool::new(true));
    let mut engine = ContentEngine::new(Arc::clone(&parent_alive))?;
    match read_message(&mut protocol_in)? {
        Message::Hello {
            worker: WorkerKind::Content,
            capability: hello_capability,
            ..
        } if hello_capability == capability => {}
        unexpected => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("content worker expected Hello, received {unexpected:?}"),
            ));
        }
    }
    write_message(&mut protocol_out, &Message::ready(WorkerKind::Content))?;

    let (tx, rx) = mpsc::channel();
    let parent_for_reader = Arc::clone(&parent_alive);
    thread::Builder::new()
        .name("web-content-protocol-reader".to_owned())
        .spawn(move || {
            let mut stdin = protocol_in;
            loop {
                match read_message(&mut stdin) {
                    Ok(message) => {
                        if tx.send(Ok(message)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        parent_for_reader.store(false, Ordering::Relaxed);
                        let _ = tx.send(Err(error));
                        return;
                    }
                }
            }
        })
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to spawn content protocol reader: {error}"),
            )
        })?;

    loop {
        if engine.parent_dead() {
            return Ok(());
        }
        // Protocol first. Spinning Servo before recv starved engine calls when
        // browser.close left a wake bit with no pages and spin_event_loop
        // blocked; the supervisor then sat in session.setProfile until the
        // client Unix read deadline expired as EAGAIN.
        let wait = if engine.pages.is_empty() {
            Duration::from_millis(200)
        } else if engine.wake.take_pending() {
            Duration::ZERO
        } else {
            Duration::from_millis(10)
        };
        match rx.recv_timeout(wait) {
            Ok(Ok(Message::EngineCall {
                request_id,
                method,
                params,
                ..
            })) => {
                let call_started = NavTrace::enabled().then(|| {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis();
                    if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: call-trace recv method={method} at_ms={now}"); } }
                    Instant::now()
                });
                let reply = match engine.handle(&method, params) {
                    Ok(result) => Message::engine_result(request_id, true, result, None),
                    Err(error) => Message::engine_result(
                        request_id,
                        false,
                        serde_json::Value::Null,
                        Some(error.to_string()),
                    ),
                };
                if let Some(started) = call_started {
                    if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: call-trace method={method} handle_ms={}",
                        started.elapsed().as_millis()
                    ); }
                }
                engine.wake.wake();
                if let Err(error) = write_message(&mut protocol_out, &reply) {
                    if crate::supervisor::phase_trace_enabled() { if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: engine result write failed: {error}"); } }
                    let fallback = Message::engine_result(
                        request_id,
                        false,
                        serde_json::Value::Null,
                        Some(error.to_string()),
                    );
                    write_message(&mut protocol_out, &fallback)?;
                }
            }
            Ok(Ok(Message::Shutdown { .. })) => {
                drop(engine);
                write_message(
                    &mut protocol_out,
                    &Message::shutdown_ack(WorkerKind::Content),
                )?;
                return Ok(());
            }
            Ok(Ok(unexpected)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("content worker received {unexpected:?}"),
                ));
            }
            Ok(Err(error)) if is_parent_eof(&error) => return Ok(()),
            Ok(Err(error)) => return Err(error),
            Err(RecvTimeoutError::Timeout) => {
                if !engine.pages.is_empty() {
                    let started = Instant::now();
                    engine.servo.spin_event_loop();
                    let elapsed = started.elapsed();
                    if elapsed >= Duration::from_millis(200) {
                        if crate::supervisor::phase_trace_enabled() { eprintln!("web-runtime: phase content-spin elapsed_ms={} pages={}",
                            elapsed.as_millis(),
                            engine.pages.len()
                        ); }
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
    }
}

/// The navigation milestone a caller waits for. `networkidle` and `commit`
/// map onto the two we can actually observe: idle behaves like a full load,
/// commit like a parsed document.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaitUntil {
    Load,
    DomContentLoaded,
}

impl WaitUntil {
    fn from_params(params: &serde_json::Value) -> Self {
        match params.get("waitUntil").and_then(|value| value.as_str()) {
            Some("domcontentloaded") | Some("commit") => Self::DomContentLoaded,
            _ => Self::Load,
        }
    }
}
