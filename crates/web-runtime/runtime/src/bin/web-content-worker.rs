use dpi::PhysicalSize;
use serde_json::json;
use servo::{
    DevicePoint, EventLoopWaker, InputEvent, JSValue, LoadStatus, MouseButton, MouseButtonAction,
    MouseButtonEvent, MouseMoveEvent, Preferences, RenderingContext, Servo, ServoBuilder,
    SoftwareRenderingContext, WebView, WebViewBuilder, WebViewDelegate, WebViewPoint,
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

struct Delegate {
    new_frame_ready: RefCell<bool>,
}

impl Default for Delegate {
    fn default() -> Self {
        Self {
            new_frame_ready: RefCell::new(false),
        }
    }
}

impl WebViewDelegate for Delegate {
    fn notify_new_frame_ready(&self, webview: WebView) {
        *self.new_frame_ready.borrow_mut() = true;
        webview.paint();
    }
}

struct ContentEngine {
    servo: Servo,
    rendering_context: Rc<dyn RenderingContext>,
    pages: HashMap<String, (WebView, Rc<Delegate>)>,
    next_id: u64,
}

impl ContentEngine {
    fn new() -> io::Result<Self> {
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
        let servo = ServoBuilder::default()
            .preferences(preferences)
            .event_loop_waker(Box::new(WakeFlag(Arc::new(AtomicBool::new(false)))))
            .build();
        Ok(Self {
            servo,
            rendering_context,
            pages: HashMap::new(),
            next_id: 1,
        })
    }

    fn alloc_id(&mut self, prefix: &str) -> String {
        let id = format!("{prefix}-{}", self.next_id);
        self.next_id += 1;
        id
    }

    fn spin_until(&self, timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate() {
                return true;
            }
            self.servo.spin_event_loop();
            thread::sleep(Duration::from_millis(1));
        }
        predicate()
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
        if !self.spin_until(ACTION_TIMEOUT, move || ready.borrow().is_some()) {
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
                let delegate = Rc::new(Delegate::default());
                let webview = WebViewBuilder::new(&self.servo, Rc::clone(&self.rendering_context))
                    .delegate(delegate.clone())
                    .build();
                webview.show();
                webview.focus();
                let created = webview.clone();
                self.spin_until(ACTION_TIMEOUT, move || created.url().is_some());
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
                            .is_some_and(|current| current.as_str() == expected.as_str())
                }) {
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
                Ok(json!({ "url": url.as_str() }))
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
  if (selector.type === 'css') {
    return Array.from(document.querySelectorAll(selector.value));
  }
  if (selector.type === 'label') {
    const labels = Array.from(document.querySelectorAll('label'));
    const match = labels.find((label) => (label.textContent || '').trim() === selector.name);
    if (!match) return [];
    if (match.control) return [match.control];
    if (match.htmlFor) {
      const el = document.getElementById(match.htmlFor);
      return el ? [el] : [];
    }
    return [];
  }
  if (selector.type === 'role') {
    return Array.from(document.querySelectorAll('body *')).filter((el) => {
      if (greppyRoleOf(el) !== selector.role) return false;
      if (selector.name == null) return true;
      return greppyAccessibleName(el) === selector.name;
    });
  }
  return [];
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

fn click_at(webview: &WebView, x: f64, y: f64, width: f64, height: f64) {
    let point = WebViewPoint::Device(DevicePoint::new(
        (x + width / 2.0) as f32,
        (y + height / 2.0) as f32,
    ));
    webview.notify_input_event(InputEvent::MouseMove(MouseMoveEvent::new(point)));
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

fn main() -> io::Result<()> {
    let _capability = require_capability(std::env::args_os().skip(1))?;
    let mut engine = ContentEngine::new()?;
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
        engine.servo.spin_event_loop();
        match rx.recv_timeout(Duration::from_millis(5)) {
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
            Ok(Err(error)) => return Err(error),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "content protocol reader stopped",
                ));
            }
        }
    }
}
