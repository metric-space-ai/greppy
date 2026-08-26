use dpi::PhysicalSize;
use serde_json::json;
use servo::{
    CreateNewWebViewRequest, DevicePoint, EmbedderControl, EventLoopWaker, InputEvent, JSValue,
    LoadStatus, MouseButton, MouseButtonAction, MouseButtonEvent, MouseMoveEvent, Preferences,
    RenderingContext, Servo, ServoBuilder, SoftwareRenderingContext, WebResourceLoad,
    WebResourceResponse, WebView, WebViewBuilder, WebViewDelegate, WebViewPoint,
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
}

struct Delegate {
    new_frame_ready: RefCell<bool>,
    routes: RefCell<Vec<RouteRule>>,
    file_paths: RefCell<Vec<std::path::PathBuf>>,
    requests: RefCell<Vec<serde_json::Value>>,
    downloads: RefCell<Vec<serde_json::Value>>,
    popups: RefCell<Vec<WebView>>,
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
            rendering_context,
        }
    }
}

impl WebViewDelegate for Delegate {
    fn notify_new_frame_ready(&self, webview: WebView) {
        *self.new_frame_ready.borrow_mut() = true;
        webview.paint();
    }

    fn request_create_new(&self, _parent: WebView, request: CreateNewWebViewRequest) {
        let child = request
            .builder(Rc::clone(&self.rendering_context))
            .delegate(Rc::new(Delegate::new(Rc::clone(&self.rendering_context))))
            .build();
        child.show();
        self.popups.borrow_mut().push(child);
    }

    fn show_embedder_control(&self, _webview: WebView, embedder_control: EmbedderControl) {
        if let EmbedderControl::FilePicker(mut picker) = embedder_control {
            let paths = self.file_paths.borrow().clone();
            if paths.is_empty() {
                picker.dismiss();
            } else {
                picker.select(&paths);
                picker.submit();
            }
        }
    }

    fn load_web_resource(&self, _webview: WebView, load: WebResourceLoad) {
        let url = load.request.url.to_string();
        self.requests.borrow_mut().push(json!({
            "url": url,
            "main_frame": load.request.is_for_main_frame,
            "redirect": load.request.is_redirect,
        }));
        let matched = self
            .routes
            .borrow()
            .iter()
            .find(|rule| pattern_matches(&rule.pattern, &url))
            .map(|rule| (rule.action.clone(), rule.body.clone()));
        let Some((action, body)) = matched else {
            return;
        };
        let request_url = load.request.url.clone();
        match action.as_str() {
            "abort" => {
                load.intercept(WebResourceResponse::new(request_url))
                    .cancel();
            }
            "fulfill" => {
                let mut intercepted = load.intercept(WebResourceResponse::new(request_url.clone()));
                if !body.is_empty() {
                    intercepted.send_body_data(body.clone());
                }
                intercepted.finish();
                self.downloads.borrow_mut().push(json!({
                    "url": request_url.to_string(),
                    "bytes": body.len(),
                }));
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
                webview.load(url.clone());
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
                    "(function(text) {{ var el = document.activeElement || document.body; if (el && el.value !== undefined) {{ el.value = (el.value || ) + text; }} return true; }})({})",
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
                    "(function(key) {{ const el = document.activeElement || document.body; el.dispatchEvent(new KeyboardEvent(keydown, {{ key, bubbles: true }})); el.dispatchEvent(new KeyboardEvent(keyup, {{ key, bubbles: true }})); return true; }})({})",
                    serde_json::to_string(&key).map_err(io::Error::other)?
                );
                self.evaluate(webview.clone(), &source)?;
                Ok(json!({}))
            }
            "page.setDialogPolicy" => Ok(json!({})),
            "page.addRoute" => {
                let page_id = required_str(&params, "page")?;
                let pattern = required_str(&params, "pattern")?;
                let action = params
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("continue")
                    .to_owned();
                let body = params
                    .get("body")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .as_bytes()
                    .to_vec();
                self.page(&page_id)?.1.routes.borrow_mut().push(RouteRule {
                    pattern,
                    action,
                    body,
                });
                Ok(json!({}))
            }
            "page.setInputFiles" => {
                let page_id = required_str(&params, "page")?;
                let files = params
                    .get("files")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let paths: Vec<std::path::PathBuf> = files
                    .iter()
                    .filter_map(|v| v.as_str().map(std::path::PathBuf::from))
                    .collect();
                self.page(&page_id)?.1.file_paths.replace(paths);
                Ok(json!({}))
            }
            "page.requests" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({ "requests": self.page(&page_id)?.1.requests.borrow().clone() }))
            }
            "page.downloads" => {
                let page_id = required_str(&params, "page")?;
                Ok(json!({ "downloads": self.page(&page_id)?.1.downloads.borrow().clone() }))
            }
            "page.popups" => {
                let page_id = required_str(&params, "page")?;
                let taken: Vec<WebView> = {
                    let delegate = &self.page(&page_id)?.1;
                    delegate.popups.borrow_mut().drain(..).collect()
                };
                let mut ids = Vec::new();
                for webview in taken {
                    let id = self.alloc_id("page");
                    let delegate = Rc::new(Delegate::new(Rc::clone(&self.rendering_context)));
                    self.pages.insert(id.clone(), (webview, delegate));
                    ids.push(id);
                }
                Ok(json!({ "pages": ids }))
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
                }
                Ok(json!({ "ok": ok }))
            }
            "page.goForward" => {
                let page_id = required_str(&params, "page")?;
                let (webview, _) = self.page(&page_id)?.clone();
                let ok = webview.can_go_forward();
                if ok {
                    webview.go_forward(1);
                }
                Ok(json!({ "ok": ok }))
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
function greppyResolveNodes(selector) {
  let nodes = [];
  if (selector.type === 'css') {
    nodes = Array.from(document.querySelectorAll(selector.value));
  } else if (selector.type === 'label') {
    const labels = Array.from(document.querySelectorAll('label'));
    const match = labels.find((label) => (label.textContent || '').trim() === selector.name);
    if (match) {
      if (match.control) nodes = [match.control];
      else if (match.htmlFor) {
        const el = document.getElementById(match.htmlFor);
        nodes = el ? [el] : [];
      }
    }
  } else if (selector.type === 'role') {
    nodes = Array.from(document.querySelectorAll('body *')).filter((el) => {
      if (greppyRoleOf(el) !== selector.role) return false;
      if (selector.name == null) return true;
      return greppyAccessibleName(el) === selector.name;
    });
  } else if (selector.type === 'text') {
    const wanted = selector.value;
    nodes = Array.from(document.querySelectorAll('body *')).filter((el) => {
      const text = ((el.innerText || el.textContent || '') + '').trim();
      return text === wanted;
    });
  }
  if (selector.nth != null) {
    const el = nodes[selector.nth];
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

fn is_parent_eof(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::UnexpectedEof | io::ErrorKind::BrokenPipe
    )
}

fn main() -> io::Result<()> {
    let _capability = require_capability(std::env::args_os().skip(1))?;
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
    write_message(&mut io::stdout(), &Message::ready(WorkerKind::Content))?;

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
                write_message(&mut io::stdout(), &reply)?;
            }
            Ok(Ok(Message::Shutdown { .. })) => {
                drop(engine);
                write_message(
                    &mut io::stdout(),
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
