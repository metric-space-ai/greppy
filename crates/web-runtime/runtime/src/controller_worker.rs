use crate::protocol::{read_message, timeout_ms_from_json, write_message, Message, WorkerKind};
use crate::worker::require_worker_auth;
use std::fs::File;
use deno_core::error::CoreError;
use deno_core::url::Url;
use deno_core::{
    extension, op2, FastString, JsRuntime, ModuleLoadResponse, ModuleLoader, ModuleSource,
    ModuleSourceCode, ModuleSpecifier, ModuleType, OpState, RequestedModuleType, ResolutionKind,
    RuntimeOptions,
};
use deno_error::JsErrorBox;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const PLAYWRIGHT_JS: &str = include_str!("../js/playwright.mjs");
/// RPC/transport headroom beyond the requested action timeout so the engine
/// can return a named actionability error instead of racing the same deadline.
const ENGINE_RPC_HEADROOM_MS: u64 = 250;

#[derive(Clone)]
struct EngineBridge {
    next_id: Arc<AtomicU64>,
    stdout: Arc<Mutex<File>>,
    script_stdout: Arc<Mutex<Vec<String>>>,
    pending: Arc<
        Mutex<
            HashMap<
                u64,
                tokio::sync::oneshot::Sender<Result<serde_json::Value, String>>,
            >,
        >,
    >,
}

struct PlaywrightLoader {
    script_root: Option<PathBuf>,
}

fn granted_script_root(specifier: &str) -> Option<PathBuf> {
    if specifier == "greppy:stdin" || specifier.starts_with("greppy:") {
        return None;
    }
    let path = Path::new(specifier);
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())?;
    parent.canonicalize().ok()
}

fn path_is_within(root: &Path, candidate: &Path) -> bool {
    candidate.starts_with(root) && candidate.components().count() >= root.components().count()
}

fn wrap_cjs_source(source: &str) -> String {
    let mut wrapped = String::from(
        r#"import * as __greppyPlaywright from "playwright";
const module = { exports: {} };
function require(id) {
  if (id === "playwright") {
    return __greppyPlaywright;
  }
  throw new Error("controller module policy denied require(" + String(id) + ")");
}
(function (module, exports, require) {
"#,
    );
    wrapped.push_str(source);
    wrapped.push_str(
        r#"
})(module, module.exports, require);
export default module.exports;
if (module.exports && typeof module.exports.then === "function") {
  await module.exports;
}
"#,
    );
    wrapped
}

fn denied_module(specifier: &ModuleSpecifier) -> ModuleLoadResponse {
    ModuleLoadResponse::Sync(Err(JsErrorBox::generic(format!(
        "controller module policy denied {specifier}"
    ))))
}

impl ModuleLoader for PlaywrightLoader {
    fn resolve(
        &self,
        specifier: &str,
        referrer: &str,
        _kind: ResolutionKind,
    ) -> Result<ModuleSpecifier, deno_core::error::ModuleLoaderError> {
        if specifier == "playwright" {
            return Ok(ModuleSpecifier::parse("greppy:playwright").expect("static specifier"));
        }
        if specifier.starts_with("greppy:") {
            return ModuleSpecifier::parse(specifier).map_err(JsErrorBox::from_err);
        }
        deno_core::resolve_import(specifier, referrer).map_err(JsErrorBox::from_err)
    }

    fn load(
        &self,
        module_specifier: &ModuleSpecifier,
        _maybe_referrer: Option<&deno_core::ModuleLoadReferrer>,
        options: deno_core::ModuleLoadOptions,
    ) -> ModuleLoadResponse {
        if module_specifier.as_str() == "greppy:playwright" {
            return ModuleLoadResponse::Sync(Ok(ModuleSource::new(
                ModuleType::JavaScript,
                ModuleSourceCode::String(FastString::from_static(PLAYWRIGHT_JS)),
                module_specifier,
                None,
            )));
        }
        if module_specifier.scheme() != "file" {
            return denied_module(module_specifier);
        }
        let Some(root) = self.script_root.as_deref() else {
            return denied_module(module_specifier);
        };
        let Ok(path) = module_specifier.to_file_path() else {
            return denied_module(module_specifier);
        };
        let Ok(canonical) = path.canonicalize() else {
            return denied_module(module_specifier);
        };
        if !path_is_within(root, &canonical) {
            return denied_module(module_specifier);
        }
        let ext = canonical
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        let Ok(source) = std::fs::read_to_string(&canonical) else {
            return denied_module(module_specifier);
        };
        let (module_type, code) = match ext {
            "mjs" | "js" => (ModuleType::JavaScript, source),
            "cjs" => (ModuleType::JavaScript, wrap_cjs_source(&source)),
            "json" => {
                if options.requested_module_type != RequestedModuleType::Json {
                    return denied_module(module_specifier);
                }
                (ModuleType::Json, source)
            }
            _ => return denied_module(module_specifier),
        };
        ModuleLoadResponse::Sync(Ok(ModuleSource::new(
            module_type,
            ModuleSourceCode::String(FastString::from(code)),
            module_specifier,
            None,
        )))
    }
}

/// Non-blocking timer. deno_core 0.410: `#[op2] async fn` + `tokio::time::sleep`
/// (see deno_core runtime/ops.rs `op_async_sleep`). `nofast` keeps this off the
/// blocking fast-op path so `await ops.op_sleep_ms(n)` actually yields.
#[op2(async(deferred), nofast)]
async fn op_sleep_ms(ms: u32) {
    tokio::time::sleep(Duration::from_millis(u64::from(ms.min(60_000)))).await;
}

#[op2(async(deferred))]
#[serde]
async fn op_engine_call(
    state: Rc<RefCell<OpState>>,
    #[string] method: String,
    #[serde] params: serde_json::Value,
) -> Result<serde_json::Value, JsErrorBox> {
    let timeout_ms = timeout_ms_from_json(params.get("timeout"), 30_000, 1, 120_000);
    let (request_id, receiver, stdout, pending) = {
        let state = state.borrow();
        let bridge = state.borrow::<EngineBridge>();
        let request_id = bridge.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        bridge
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(request_id, sender);
        (
            request_id,
            receiver,
            Arc::clone(&bridge.stdout),
            Arc::clone(&bridge.pending),
        )
    };
    let trace = std::env::var_os("GREPPY_WEB_TRACE_NAV").is_some();
    let sys_ms = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    };
    if trace {
        eprintln!(
            "web-runtime: call-trace send method={method} at_ms={}",
            sys_ms()
        );
    }
    {
        let mut stdout = stdout.lock().unwrap_or_else(|error| error.into_inner());
        write_message(
            &mut *stdout,
            &Message::engine_call(request_id, method.clone(), params),
        )
        .map_err(|error| JsErrorBox::generic(error.to_string()))?;
    }
    let watchdog_ms = timeout_ms
        .saturating_add(ENGINE_RPC_HEADROOM_MS)
        .min(180_000);
    let outcome = tokio::time::timeout(Duration::from_millis(watchdog_ms), receiver).await;
    if trace {
        eprintln!(
            "web-runtime: call-trace done method={method} at_ms={}",
            sys_ms()
        );
    }
    match outcome {
        Ok(Ok(Ok(value))) => Ok(value),
        Ok(Ok(Err(error))) => Err(JsErrorBox::generic(error)),
        Ok(Err(_)) => Err(JsErrorBox::generic("engine call was cancelled")),
        Err(_) => {
            pending
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(&request_id);
            Err(JsErrorBox::generic(format!(
                "timed out after {timeout_ms}ms"
            )))
        }
    }
}

#[op2(fast)]
fn op_capture_stdout(state: Rc<RefCell<OpState>>, #[string] line: String) {
    let state = state.borrow();
    let bridge = state.borrow::<EngineBridge>();
    bridge
        .script_stdout
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(line);
}

fn confine_screenshot_sidecar(path: &str) -> Result<PathBuf, JsErrorBox> {
    let requested = PathBuf::from(path);
    let root = std::env::temp_dir()
        .canonicalize()
        .unwrap_or_else(|_| std::env::temp_dir());
    let canon = requested
        .canonicalize()
        .map_err(|error| JsErrorBox::generic(error.to_string()))?;
    if !canon.starts_with(&root) {
        return Err(JsErrorBox::generic(format!(
            "path outside worker temp: {}",
            canon.display()
        )));
    }
    let name = canon
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if !name.starts_with("greppy-web-shot-") || !name.ends_with(".png") {
        return Err(JsErrorBox::generic("refusing non-screenshot sidecar"));
    }
    Ok(canon)
}

fn base64_encode_png(bytes: &[u8]) -> String {
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

#[op2]
#[string]
fn op_read_temp_png(#[string] path: String) -> Result<String, JsErrorBox> {
    let confined = confine_screenshot_sidecar(&path)?;
    let bytes = std::fs::read(&confined).map_err(|error| JsErrorBox::generic(error.to_string()))?;
    let _ = std::fs::remove_file(&confined);
    Ok(base64_encode_png(&bytes))
}

extension!(
    greppy_playwright,
    ops = [op_engine_call, op_sleep_ms, op_capture_stdout, op_read_temp_png],
    options = { bridge: EngineBridge },
    state = |state, options| {
        state.put(options.bridge);
    },
);

pub fn run() -> io::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(io::Error::other)?;
    let _enter = runtime.enter();
    run_with_tokio(runtime)
}

fn run_with_tokio(tokio_runtime: tokio::runtime::Runtime) -> io::Result<()> {
    let capability = require_worker_auth(std::env::args_os().skip(1))?;
    crate::supervisor::apply_worker_sandbox(
        &std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("/")),
        &std::env::temp_dir(),
    )?;
    let (mut protocol_in, protocol_out) = crate::worker::take_protocol_channel()?;
    let stdout = Arc::new(Mutex::new(protocol_out));
    let pending = Arc::new(Mutex::new(HashMap::new()));
    let script_stdout = Arc::new(Mutex::new(Vec::new()));
    let bridge = EngineBridge {
        next_id: Arc::new(AtomicU64::new(1)),
        stdout: Arc::clone(&stdout),
        script_stdout: Arc::clone(&script_stdout),
        pending: Arc::clone(&pending),
    };
    let mut runtime = new_js_runtime(bridge.clone(), None);
    runtime
        .execute_script("<web-controller-worker>", "1 + 1")
        .map_err(|error| io::Error::other(format!("JavaScript startup probe failed: {error}")))?;
    install_process_env_allow_list(&mut runtime)?;
    install_console_capture(&mut runtime)?;

    {
        match read_message(&mut protocol_in)? {
            Message::Hello {
                worker: WorkerKind::Controller,
                capability: hello_capability,
                ..
            } if hello_capability == capability => {}
            unexpected => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("controller worker expected Hello, received {unexpected:?}"),
                ));
            }
        }
        let mut stdout = stdout.lock().unwrap_or_else(|error| error.into_inner());
        write_message(&mut *stdout, &Message::ready(WorkerKind::Controller))?;
    }

    let (control_tx, control_rx) = mpsc::channel();
    let pending_for_reader = Arc::clone(&pending);
    thread::Builder::new()
        .name("web-controller-protocol-reader".to_owned())
        .spawn(move || {
            let mut stdin = protocol_in;
            loop {
                match read_message(&mut stdin) {
                    Ok(Message::EngineResult {
                        request_id,
                        ok,
                        result,
                        error,
                        ..
                    }) => {
                        if let Some(sender) = pending_for_reader
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .remove(&request_id)
                        {
                            let _ = sender.send(if ok {
                                Ok(result)
                            } else {
                                Err(error.unwrap_or_else(|| "engine call failed".to_owned()))
                            });
                        }
                    }
                    Ok(message) => {
                        if control_tx.send(Ok(message)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = control_tx.send(Err(error));
                        return;
                    }
                }
            }
        })
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to spawn controller protocol reader: {error}"),
            )
        })?;

    loop {
        let message = match recv_control(&control_rx) {
            Ok(message) => message,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::UnexpectedEof | io::ErrorKind::BrokenPipe
                ) =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        match message {
            Message::Shutdown { .. } => {
                drop(runtime);
                let mut stdout = stdout.lock().unwrap_or_else(|error| error.into_inner());
                return write_message(&mut *stdout, &Message::shutdown_ack(WorkerKind::Controller));
            }
            Message::RunScript {
                specifier,
                source,
                fixture_url,
                ..
            } => {
                drop(runtime);
                runtime = new_js_runtime(bridge.clone(), granted_script_root(&specifier));
                install_process_env_allow_list(&mut runtime)?;
                install_console_capture(&mut runtime)?;
                bridge
                    .script_stdout
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .clear();
                let result = run_script(
                    &tokio_runtime,
                    &mut runtime,
                    &specifier,
                    source,
                    fixture_url,
                );
                let captured = bridge
                    .script_stdout
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .join("\n");
                let payload = serde_json::json!({ "stdout": captured });
                let mut stdout = stdout.lock().unwrap_or_else(|error| error.into_inner());
                match result {
                    Ok(()) => write_message(
                        &mut *stdout,
                        &Message::script_complete(true, payload, None),
                    )?,
                    Err(error) => write_message(
                        &mut *stdout,
                        &Message::script_complete(false, payload, Some(error)),
                    )?,
                }
            }
            unexpected => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "controller worker expected RunScript or Shutdown, received {unexpected:?}"
                    ),
                ));
            }
        }
    }
}

fn new_js_runtime(bridge: EngineBridge, script_root: Option<PathBuf>) -> JsRuntime {
    JsRuntime::new(RuntimeOptions {
        module_loader: Some(Rc::new(PlaywrightLoader { script_root })),
        extensions: vec![greppy_playwright::init(bridge)],
        ..Default::default()
    })
}

fn install_process_env_allow_list(runtime: &mut JsRuntime) -> io::Result<()> {
    runtime
        .execute_script(
            "<greppy-process-env>",
            "globalThis.process = Object.freeze({ env: Object.freeze({ NODE_ENV: \"production\" }) });",
        )
        .map_err(|error| io::Error::other(format!("process.env allow-list failed: {error}")))?;
    Ok(())
}

fn install_console_capture(runtime: &mut JsRuntime) -> io::Result<()> {
    runtime
        .execute_script(
            "<greppy-console-capture>",
            r#"
(function () {
  const ops = Deno.core.ops;
  const capture = (args) => {
    const line = Array.prototype.map.call(args, (value) => {
      if (typeof value === "string") return value;
      try { return JSON.stringify(value); } catch (_error) { return String(value); }
    }).join(" ");
    try { ops.op_capture_stdout(line); } catch (_error) {}
  };
  const wrap = (method) => {
    const original = console[method].bind(console);
    console[method] = function () {
      capture(arguments);
      return original.apply(console, arguments);
    };
  };
  wrap("log");
  wrap("info");
  wrap("warn");
  wrap("error");
  wrap("debug");
})();
"#,
        )
        .map_err(|error| io::Error::other(format!("console capture hook failed: {error}")))?;
    Ok(())
}

fn recv_control(control_rx: &mpsc::Receiver<io::Result<Message>>) -> io::Result<Message> {
    match control_rx.recv() {
        Ok(message) => message,
        Err(_) => Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "controller protocol reader stopped",
        )),
    }
}

fn run_script(
    tokio_runtime: &tokio::runtime::Runtime,
    runtime: &mut JsRuntime,
    specifier: &str,
    source: String,
    fixture_url: String,
) -> Result<(), String> {
    let fixture = serde_json::to_string(&fixture_url).map_err(|error| error.to_string())?;
    runtime
        .execute_script(
            "<greppy-fixture-url>",
            format!("var fixtureUrl = {fixture};"),
        )
        .map_err(|error| error.to_string())?;

    let module_url = Url::from_file_path(specifier)
        .or_else(|_| Url::parse(specifier))
        .map_err(|_| format!("script specifier is not a file path or URL: {specifier}"))?;
    let source = if specifier.ends_with(".cjs") {
        wrap_cjs_source(&source)
    } else {
        source
    };

    let future = async {
        let module_id = runtime
            .load_main_es_module_from_code(&module_url, source)
            .await?;
        let evaluation = runtime.mod_evaluate(module_id);
        runtime.run_event_loop(Default::default()).await?;
        evaluation.await?;
        Ok::<(), CoreError>(())
    };
    tokio_runtime
        .block_on(future)
        .map_err(|error| error.to_string())
}
