use dpi::PhysicalSize;
use serde_json::json;
use servo::{
    ConsoleLogLevel, CreateNewWebViewRequest, DevicePoint, EmbedderControl, EventLoopWaker,
    InputEvent, JSValue, LoadStatus, MouseButton, MouseButtonAction, MouseButtonEvent,
    MouseMoveEvent, Preferences, RenderingContext, Servo, ServoBuilder, SimpleDialog,
    SoftwareRenderingContext, UrlRequest, WebResourceLoad, WebResourceResponse, WebView,
    WebViewBuilder, WebViewDelegate, WebViewPoint,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use url::Url;
use web_runtime::protocol::{read_message, write_message, Message, WorkerKind};
use web_runtime::worker::require_capability;

const ACTION_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct WakeFlag(Arc<AtomicBool>);

impl EventLoopWaker for WakeFlag {
    fn clone_box(&self) -> Box<dyn EventLoopWaker> {
        Box::new(self.clone())
    }

    fn wake(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

struct RouteRule {
    pattern: String,
    action: String,
    body: Vec<u8>,
    status: u16,
    content_type: String,
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
    last_file_choosers: RefCell<Vec<serde_json::Value>>,
    last_responses: RefCell<Vec<serde_json::Value>>,
    rendering_context: Rc<dyn RenderingContext>,
}

impl Delegate {
    fn new(rendering_context: Rc<dyn RenderingContext>) -> Self {
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
            viewport: RefCell::new((800, 600)),
            extra_headers: RefCell::new(Vec::new()),
            last_console: RefCell::new(Vec::new()),
            last_file_choosers: RefCell::new(Vec::new()),
            last_responses: RefCell::new(Vec::new()),
            opener_id: RefCell::new(None),
            rendering_context,
        }
    }
}

impl WebViewDelegate for Delegate {
    fn notify_new_frame_ready(&self, webview: WebView) {
        *self.new_frame_ready.borrow_mut() = true;
        webview.paint();
    }

    fn show_console_message(&self, _webview: WebView, level: ConsoleLogLevel, message: String) {
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
    }

    fn request_create_new(&self, parent: WebView, request: CreateNewWebViewRequest) {
        let child = request
            .builder(Rc::clone(&self.rendering_context))
            .delegate(Rc::new(Delegate::new(Rc::clone(&self.rendering_context))))
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
        let headers: Vec<serde_json::Value> = load
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
        self.requests.borrow_mut().push(json!({
            "url": url,
            "method": load.request.method.to_string(),
            "main_frame": load.request.is_for_main_frame,
            "redirect": load.request.is_redirect,
            "headers": headers,
        }));
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
                )
            });
        let Some((action, body, status, content_type)) = matched else {
            return;
        };
        let request_url = load.request.url.clone();
        match action.as_str() {
            "abort" => {
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
                let lower = content_type.to_ascii_lowercase();
                let is_download =
                    lower.contains("octet-stream") || lower.contains("attachment");
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
                }
            }
            _ => {}
        }
    }
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

struct ContentEngine {
    servo: Servo,
    rendering_context: Rc<dyn RenderingContext>,
    pages: HashMap<String, (WebView, Rc<Delegate>)>,
    next_id: u64,
    parent_alive: Arc<AtomicBool>,
    wake: Arc<AtomicBool>,
}

impl ContentEngine {
    fn new(parent_alive: Arc<AtomicBool>) -> io::Result<Self> {
        let rendering_context = Rc::new(
            SoftwareRenderingContext::new(PhysicalSize {
                width: 800,
                height: 600,
            })
            .map_err(|error| io::Error::other(format!("software renderer failed: {error:?}")))?,
        );
        rendering_context.make_current().map_err(|error| {
            io::Error::other(format!("renderer make_current failed: {error:?}"))
        })?;

        let mut preferences = Preferences::default();
        preferences.network_http_proxy_uri = String::new();
        preferences.network_https_proxy_uri = String::new();
        let wake = Arc::new(AtomicBool::new(true));
        let servo = ServoBuilder::default()
            .preferences(preferences)
            .event_loop_waker(Box::new(WakeFlag(Arc::clone(&wake))))
            .build();
        Ok(Self {
            servo,
            rendering_context,
            pages: HashMap::new(),
            next_id: 1,
            parent_alive,
            wake,
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
        let id = format!("{prefix}-{}", self.next_id);
        self.next_id += 1;
        id
    }

    fn spin_until(
        &self,
        timeout: Duration,
        mut predicate: impl FnMut() -> bool,
    ) -> io::Result<bool> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if self.parent_dead() {
                return Err(Self::parent_gone());
            }
            if predicate() {
                return Ok(true);
            }
            self.servo.spin_event_loop();
            thread::sleep(Duration::from_millis(1));
        }
        Ok(predicate())
    }

    fn page(&self, page_id: &str) -> io::Result<&(WebView, Rc<Delegate>)> {
        self.pages.get(page_id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("unknown page {page_id}"))
        })
    }

    fn evaluate(&self, webview: WebView, script: &str) -> io::Result<JSValue> {
        let saved = Rc::new(RefCell::new(None));
        let callback_slot = Rc::clone(&saved);
        webview.evaluate_javascript(script, move |result| {
            *callback_slot.borrow_mut() = Some(result);
        });
        let ready = Rc::clone(&saved);
        if !self.spin_until(ACTION_TIMEOUT, move || ready.borrow().is_some())? {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out evaluating page JavaScript",
            ));
        }
        let result = saved.borrow_mut().take().expect("evaluation completed");
        result.map_err(|error| io::Error::other(format!("page JavaScript failed: {error:?}")))
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
            let bytes = match std::fs::read(path) {
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
        match method {
            "chromium.launch" => {
                let browser = self.alloc_id("browser");
                Ok(json!({ "browser": browser }))
            }
            "browser.newContext" => {
                let context = self.alloc_id("context");
                Ok(json!({ "context": context }))
            }
            "context.newPage" => {
                let page = self.alloc_id("page");
                let delegate = Rc::new(Delegate::new(Rc::clone(&self.rendering_context)));
                let webview = WebViewBuilder::new(&self.servo, Rc::clone(&self.rendering_context))
                    .delegate(delegate.clone())
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
                self.pages.insert(page.clone(), (webview, delegate));
                Ok(json!({ "page": page }))
            }
            "page.goto" => {
                let page_id = required_str(&params, "page")?;
                let url = required_str(&params, "url")?;
                let url = Url::parse(&url)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
                let (webview, _) = self.page(&page_id)?.clone();
                let extra = self.page(&page_id)?.1.extra_headers.borrow().clone();
                if extra.is_empty() {
                    webview.load(url.clone());
                } else {
                    let mut headers = http::HeaderMap::new();
                    for (name, value) in extra {
                        let Ok(name) = http::HeaderName::from_bytes(name.as_bytes()) else {
                            continue;
                        };
                        let Ok(value) = http::HeaderValue::from_str(&value) else {
                            continue;
                        };
                        headers.append(name, value);
                    }
                    webview.load_request(UrlRequest::new(url.clone()).headers(headers));
                }
                let loading = webview.clone();
                let expected = url.clone();
                if !self.spin_until(ACTION_TIMEOUT, move || {
                    loading.load_status() == LoadStatus::Complete
                        && loading
                            .url()
                            .is_some_and(|current| urls_match(&current, &expected))
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
                webview.paint();
                self.servo.spin_event_loop();
                thread::sleep(Duration::from_millis(20));
                self.servo.spin_event_loop();
                self.run_init_scripts(&page_id)?;
                Ok(
                    json!({ "url": webview.url().map(|u| u.to_string()).unwrap_or_else(|| url.to_string()) }),
                )
            }
            "locator.click" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                click_at(
                    &webview,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                );
                self.servo.spin_event_loop();
                if let Some(selector) = params
                    .get("selector")
                    .and_then(|value| value.get("value"))
                    .and_then(|value| value.as_str())
                {
                    let _ = self.assign_pending_files(&page_id, selector);
                }
                Ok(json!({}))
            }
            "locator.dblclick" => {
                let resolved = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                click_at(
                    &webview,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                );
                self.servo.spin_event_loop();
                click_at(
                    &webview,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                );
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
                let (webview, _) = self.page(&page_id)?.clone();
                click_at(
                    &webview,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                );
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
                let value = jsvalue_to_json(self.evaluate(webview, &source)?);
                Ok(json!({ "value": value }))
            }
            "browser.close" => {
                self.pages.clear();
                Ok(json!({}))
            }
            "page.close" => {
                let page_id = required_str(&params, "page")?;
                self.pages.remove(&page_id);
                Ok(json!({}))
            }
            "page.isClosed" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({ "closed": !self.pages.contains_key(&page_id) }))
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
                let (webview, _) = self.page(&page_id)?.clone();
                match self.evaluate(webview, OBSERVE_JS)? {
                    JSValue::String(text) => serde_json::from_str(&text)
                        .map_err(|error| io::Error::other(format!("observe json: {error}"))),
                    JSValue::Object(values) => Ok(jsvalue_to_json(JSValue::Object(values))),
                    other => Err(io::Error::other(format!("observe returned {other:?}"))),
                }
            }
            "page.screenshot" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let png = self.screenshot_png(&webview)?;
                Ok(json!({ "png_base64": base64_encode(&png) }))
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
                        let width = number_field(&values, "width").unwrap_or(0.0);
                        let height = number_field(&values, "height").unwrap_or(0.0);
                        Ok(json!({ "visible": count == 1.0 && width > 0.0 && height > 0.0 }))
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
                let (webview, _) = self.page(&page_id)?.clone();
                hover_at(
                    &webview,
                    resolved.x,
                    resolved.y,
                    resolved.width,
                    resolved.height,
                );
                self.servo.spin_event_loop();
                Ok(json!({}))
            }
            "locator.check" | "locator.uncheck" => {
                let _ = self.resolve_actionable(&params)?;
                let page_id = required_str(&params, "page")?;
                let checked = method == "locator.check";
                let selector = params
                    .get("selector")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let (webview, _) = self.page(&page_id)?.clone();
                let source = format!(
                    "(function(selector, checked) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); if (nodes.length !== 1) throw new Error('strict mode'); nodes[0].checked = checked; return true; }})({selector}, {checked})"
                );
                self.evaluate(webview, &source)?;
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
                let source = format!(
                    "(function(selector, value) {{ {SELECTOR_RUNTIME} var nodes = greppyResolveNodes(selector); if (nodes.length !== 1) throw new Error('strict mode'); nodes[0].value = value; return true; }})({selector}, {})",
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
            "locator.boundingBox" => {
                let resolved = self.resolve_actionable(&params)?;
                Ok(json!({
                    "x": resolved.x,
                    "y": resolved.y,
                    "width": resolved.width,
                    "height": resolved.height,
                }))
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
                Ok(json!({ "value": jsvalue_to_json(self.evaluate(webview, &script)?) }))
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
                Ok(json!({ "value": jsvalue_to_json(self.evaluate(webview, &script)?) }))
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
                self.handle("page.goto", json!({ "page": page_id, "url": url.as_str() }))
            }
            "page.waitForLoadState" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let loading = webview.clone();
                if !self.spin_until(ACTION_TIMEOUT, move || {
                    loading.load_status() == LoadStatus::Complete
                })? {
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
                    "(function(text) {{ var el = document.activeElement || document.body; if (el && 'value' in el) {{ el.value = String(el.value || '') + text; }} return true; }})({})",
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
                    "(function(key) {{ const el = document.activeElement || document.body; el.dispatchEvent(new KeyboardEvent('keydown', {{ key: key, bubbles: true }})); el.dispatchEvent(new KeyboardEvent('keyup', {{ key: key, bubbles: true }})); return true; }})({})",
                    serde_json::to_string(&key).map_err(io::Error::other)?
                );
                self.evaluate(webview.clone(), &source)?;
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
            "context.close" => Ok(json!({})),
            "page.addRoute" => {
                let page_id = required_str(&params, "page")?;
                let pattern = required_str(&params, "pattern")?;
                let action = params
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("continue")
                    .to_owned();
                let body = if let Some(b64) = params
                    .get("bodyBase64")
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.is_empty())
                {
                    base64_decode(b64)?
                } else {
                    params
                        .get("body")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .as_bytes()
                        .to_vec()
                };
                let status = params.get("status").and_then(|v| v.as_u64()).unwrap_or(200) as u16;
                let content_type = params
                    .get("contentType")
                    .and_then(|value| value.as_str())
                    .unwrap_or("text/html")
                    .to_owned();
                self.page(&page_id)?.1.routes.borrow_mut().push(RouteRule {
                    pattern,
                    action,
                    body,
                    status,
                    content_type,
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
                let paths: Vec<std::path::PathBuf> = files
                    .iter()
                    .filter_map(|v| v.as_str().map(std::path::PathBuf::from))
                    .collect();
                for path in &paths {
                    std::fs::read(path).map_err(|error| {
                        io::Error::new(
                            error.kind(),
                            format!("setInputFiles cannot read {}: {error}", path.display()),
                        )
                    })?;
                }
                self.page(&page_id)?.1.file_paths.replace(paths);
                self.assign_pending_files(&page_id, &selector)
            }
            "page.requests" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({ "requests": self.page(&page_id)?.1.requests.borrow().clone() }))
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
                        .find(|row| row.get("url").and_then(|value| value.as_str()) == Some(url.as_str()))
                        .and_then(|row| {
                            row.get("bodyBase64")
                                .and_then(|value| value.as_str())
                                .map(str::to_owned)
                        })
                        .ok_or_else(|| io::Error::other("no matching download body"))?
                };
                let bytes = base64_decode(&body_b64)?;
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                std::fs::write(&path, &bytes)?;
                let written = std::fs::read(&path)?;
                if written != bytes {
                    return Err(io::Error::other("saveDownload readback mismatch"));
                }
                Ok(json!({ "ok": true, "bytes": written.len() }))
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
                        .find_map(|(id, (existing, _))| {
                            if existing == &parent {
                                Some(id.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| page_id.clone());
                    let id = self.alloc_id("page");
                    let delegate = Rc::new(Delegate::new(Rc::clone(&self.rendering_context)));
                    delegate.opener_id.replace(Some(opener.clone()));
                    self.pages.insert(id.clone(), (webview, delegate));
                    pages.push(json!({ "page": id, "opener": opener }));
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
            "page.frameEvaluate" => {
                let page_id = required_str(&params, "page")?;
                let index = params.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let source = required_str(&params, "source")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let script = format!(
                    "(function(index, source) {{ var frame = document.querySelectorAll('iframe')[index]; if (!frame) throw new Error('no frame'); return frame.contentWindow.eval(source); }})({index}, {})",
                    serde_json::to_string(&source).map_err(io::Error::other)?
                );
                let value = jsvalue_to_json(self.evaluate(webview, &script)?);
                Ok(json!({ "value": value }))
            }
            "page.goBack" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let ok = webview.can_go_back();
                if ok {
                    webview.go_back(1);
                    let loading = webview.clone();
                    let _ = self.spin_until(ACTION_TIMEOUT, move || {
                        loading.load_status() == LoadStatus::Complete
                    })?;
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
                    let _ = self.spin_until(ACTION_TIMEOUT, move || {
                        loading.load_status() == LoadStatus::Complete
                    })?;
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
                    "(function(cookies) {{ cookies.forEach(function(c) {{ document.cookie = c.name + '=' + c.value; }}); return true; }})({})",
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
                Ok(json!({
                    "requests": delegate.requests.borrow().clone(),
                    "downloads": delegate.downloads.borrow().clone(),
                    "file_paths": delegate
                        .file_paths
                        .borrow()
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>(),
                }))
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
                let width = params.get("width").and_then(|v| v.as_u64()).unwrap_or(800) as u32;
                let height = params.get("height").and_then(|v| v.as_u64()).unwrap_or(600) as u32;
                *self.page(&page_id)?.1.viewport.borrow_mut() = (width, height);
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
                let (webview, _) = self.page(&page_id)?.clone();
                click_at(&webview, x, y, 0.0, 0.0);
                self.servo.spin_event_loop();
                Ok(json!({}))
            }
            "page.mouse.move" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, _) = self.page(&page_id)?.clone();
                hover_at(&webview, x, y, 0.0, 0.0);
                self.servo.spin_event_loop();
                Ok(json!({}))
            }
            "page.mouse.down" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, _) = self.page(&page_id)?.clone();
                let point = WebViewPoint::Device(DevicePoint::new(x as f32, y as f32));
                webview.notify_input_event(InputEvent::MouseButton(MouseButtonEvent::new(
                    MouseButtonAction::Down,
                    MouseButton::Left,
                    point,
                )));
                Ok(json!({}))
            }
            "page.mouse.up" => {
                let page_id = required_str(&params, "page")?;
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let (webview, _) = self.page(&page_id)?.clone();
                let point = WebViewPoint::Device(DevicePoint::new(x as f32, y as f32));
                webview.notify_input_event(InputEvent::MouseButton(MouseButtonEvent::new(
                    MouseButtonAction::Up,
                    MouseButton::Left,
                    point,
                )));
                Ok(json!({}))
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported_playwright_operation: {other}"),
            )),
        }
    }

    fn resolve_actionable(&self, params: &serde_json::Value) -> io::Result<ResolvedNode> {
        let page_id = required_str(params, "page")?;
        let selector = params
            .get("selector")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let (webview, _) = self.page(&page_id)?.clone();
        let script = resolve_script(&selector);
        let deadline = Instant::now() + ACTION_TIMEOUT;
        let mut last;
        loop {
            if self.parent_dead() {
                return Err(Self::parent_gone());
            }
            match self.evaluate(webview.clone(), &script)? {
                JSValue::Object(values) => {
                    let count = number_field(&values, "count")? as usize;
                    let width = number_field(&values, "width").unwrap_or(0.0);
                    let height = number_field(&values, "height").unwrap_or(0.0);
                    let disabled = bool_field(&values, "disabled").unwrap_or(false);
                    last =
                        format!("count={count} width={width} height={height} disabled={disabled}");
                    if count == 1 && width > 0.0 && height > 0.0 && !disabled {
                        return Ok(ResolvedNode {
                            x: number_field(&values, "x")?,
                            y: number_field(&values, "y")?,
                            width,
                            height,
                        });
                    }
                    if count > 1 {
                        return Err(io::Error::other(format!(
                            "strict mode: selector matched {count} nodes"
                        )));
                    }
                }
                other => {
                    last = format!("{other:?}");
                }
            }
            if Instant::now() >= deadline {
                let html = self
                    .evaluate(
                        webview,
                        "document.documentElement ? document.documentElement.outerHTML.slice(0, 500) : ''",
                    )
                    .ok();
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "timed out waiting for actionable locator target ({last}; html={html:?})"
                    ),
                ));
            }
            self.servo.spin_event_loop();
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn screenshot_png(&self, webview: &WebView) -> io::Result<Vec<u8>> {
        let saved = Rc::new(RefCell::new(None));
        let callback = Rc::clone(&saved);
        webview.take_screenshot(None, move |result| {
            *callback.borrow_mut() = Some(result);
        });
        let deadline = Instant::now() + ACTION_TIMEOUT;
        while saved.borrow().is_none() {
            if self.parent_dead() {
                return Err(Self::parent_gone());
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "timed out capturing screenshot",
                ));
            }
            self.servo.spin_event_loop();
            thread::sleep(Duration::from_millis(1));
        }
        let image = saved
            .borrow_mut()
            .take()
            .expect("screenshot completed")
            .map_err(|error| io::Error::other(format!("screenshot failed: {error:?}")))?;
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
}

struct ResolvedNode {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
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
function greppyQueryAll(root, sel) {
  try { return Array.from(root.querySelectorAll(sel)); } catch (error) { return []; }
}
function greppyCandidates(root) {
  return greppyQueryAll(root === document ? document : root, '*');
}
function greppyResolveIn(root, selector) {
  if (selector.type === 'css') {
    return greppyQueryAll(root === document ? document : root, selector.value);
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
    const frames = greppyQueryAll(root === document ? document : root, selector.frame || 'iframe');
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
          const nodes = greppyResolveNodes(selector);
          if (nodes.length !== 1) {{
            return {{ count: nodes.length, x: 0, y: 0, width: 0, height: 0, disabled: false }};
          }}
          const el = nodes[0];
          void document.body.offsetHeight;
          const rect = el.getBoundingClientRect();
          return {{
            count: 1,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true')
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

fn hover_at(webview: &WebView, x: f64, y: f64, width: f64, height: f64) {
    let point = WebViewPoint::Device(DevicePoint::new(
        (x + width / 2.0) as f32,
        (y + height / 2.0) as f32,
    ));
    webview.notify_input_event(InputEvent::MouseMove(MouseMoveEvent::new(point)));
}

fn click_at(webview: &WebView, x: f64, y: f64, width: f64, height: f64) {
    hover_at(webview, x, y, width, height);
    let point = WebViewPoint::Device(DevicePoint::new(
        (x + width / 2.0) as f32,
        (y + height / 2.0) as f32,
    ));
    webview.notify_input_event(InputEvent::MouseButton(MouseButtonEvent::new(
        MouseButtonAction::Down,
        MouseButton::Left,
        point,
    )));
    webview.notify_input_event(InputEvent::MouseButton(MouseButtonEvent::new(
        MouseButtonAction::Up,
        MouseButton::Left,
        point,
    )));
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

const OBSERVE_JS: &str = r#"JSON.stringify({
  url: location.href,
  title: document.title,
  text: ((document.body && document.body.innerText) || '').slice(0, 8000),
  headings: Array.from(document.querySelectorAll('h1,h2,h3,h4')).map(function(h) {
    return (h.innerText || '').trim();
  }).filter(Boolean).slice(0, 20),
  links: Array.from(document.querySelectorAll('a[href]')).slice(0, 20).map(function(a) {
    return { href: a.href, text: ((a.innerText || '').trim()).slice(0, 80) };
  })
})"#;

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
        let c = if chunk[2] == b'=' { 0 } else { sextet(chunk[2])? };
        let d = if chunk[3] == b'=' { 0 } else { sextet(chunk[3])? };
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

fn is_parent_eof(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::UnexpectedEof | io::ErrorKind::BrokenPipe
    )
}

#[cfg(unix)]
fn steal_protocol_stdout() -> io::Result<std::fs::File> {
    use std::os::fd::{AsFd, AsRawFd};
    let protocol = std::io::stdout().as_fd().try_clone_to_owned()?;
    let err = std::io::stderr().as_raw_fd();
    let rc = unsafe { libc::dup2(err, libc::STDOUT_FILENO) };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(std::fs::File::from(protocol))
}

#[cfg(not(unix))]
fn steal_protocol_stdout() -> io::Result<std::fs::File> {
    // Best-effort: keep stdout as the protocol stream on non-Unix.
    use std::os::fd::AsFd;
    Ok(std::fs::File::from(
        std::io::stdout().as_fd().try_clone_to_owned()?,
    ))
}

fn main() -> io::Result<()> {
    let _capability = require_capability(std::env::args_os().skip(1))?;
    let mut protocol_out = steal_protocol_stdout()?;
    let parent_alive = Arc::new(AtomicBool::new(true));
    let mut engine = ContentEngine::new(Arc::clone(&parent_alive))?;
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    match read_message(&mut stdin)? {
        Message::Hello {
            worker: WorkerKind::Content,
            ..
        } => {}
        unexpected => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("content worker expected Hello, received {unexpected:?}"),
            ));
        }
    }
    drop(stdin);
    write_message(&mut protocol_out, &Message::ready(WorkerKind::Content))?;

    let (tx, rx) = mpsc::channel();
    let parent_for_reader = Arc::clone(&parent_alive);
    thread::Builder::new()
        .name("web-content-protocol-reader".to_owned())
        .spawn(move || {
            let mut stdin = io::stdin();
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
        // Servo asked to be pumped, or a live document needs a modest idle
        // tick. Never spin the compositor while the protocol is idle and no
        // pages exist — that kept CPU at 100% after the supervisor died.
        if engine.wake.swap(false, Ordering::Relaxed) {
            engine.servo.spin_event_loop();
        }
        let wait = if engine.pages.is_empty() {
            Duration::from_millis(200)
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
                let reply = match engine.handle(&method, params) {
                    Ok(result) => Message::engine_result(request_id, true, result, None),
                    Err(error) => Message::engine_result(
                        request_id,
                        false,
                        serde_json::Value::Null,
                        Some(error.to_string()),
                    ),
                };
                engine.wake.store(true, Ordering::Relaxed);
                write_message(&mut protocol_out, &reply)?;
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
                    engine.servo.spin_event_loop();
                }
            }
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
    }
}
