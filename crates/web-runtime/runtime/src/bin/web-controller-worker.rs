use deno_core::error::CoreError;
use deno_core::url::Url;
use deno_core::{
    extension, op2, FastString, JsRuntime, ModuleLoadResponse, ModuleLoader, ModuleSource,
    ModuleSourceCode, ModuleSpecifier, ModuleType, OpState, ResolutionKind, RuntimeOptions,
};
use deno_error::JsErrorBox;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use web_runtime::protocol::{read_message, write_message, Message, WorkerKind};
use web_runtime::worker::require_capability;

const PLAYWRIGHT_JS: &str = include_str!("../../js/playwright.mjs");
const MESSAGE_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone)]
struct EngineBridge {
    next_id: Arc<AtomicU64>,
    stdout: Arc<Mutex<io::Stdout>>,
    pending: Arc<
        Mutex<
            HashMap<
                u64,
                deno_core::futures::channel::oneshot::Sender<Result<serde_json::Value, String>>,
            >,
        >,
    >,
}

struct PlaywrightLoader;

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
        _options: deno_core::ModuleLoadOptions,
    ) -> ModuleLoadResponse {
        if module_specifier.as_str() == "greppy:playwright" {
            return ModuleLoadResponse::Sync(Ok(ModuleSource::new(
                ModuleType::JavaScript,
                ModuleSourceCode::String(FastString::from_static(PLAYWRIGHT_JS)),
                module_specifier,
                None,
            )));
        }
        ModuleLoadResponse::Sync(Err(JsErrorBox::generic(format!(
            "controller module policy denied {module_specifier}"
        ))))
    }
}

#[op2(async(deferred))]
#[serde]
async fn op_engine_call(
    state: Rc<RefCell<OpState>>,
    #[string] method: String,
    #[serde] params: serde_json::Value,
) -> Result<serde_json::Value, JsErrorBox> {
    let (request_id, receiver, stdout) = {
        let state = state.borrow();
        let bridge = state.borrow::<EngineBridge>();
        let request_id = bridge.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = deno_core::futures::channel::oneshot::channel();
        bridge
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(request_id, sender);
        (request_id, receiver, Arc::clone(&bridge.stdout))
    };
    {
        let mut stdout = stdout.lock().unwrap_or_else(|error| error.into_inner());
        write_message(
            &mut *stdout,
            &Message::engine_call(request_id, method, params),
        )
        .map_err(|error| JsErrorBox::generic(error.to_string()))?;
    }
    match receiver.await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(JsErrorBox::generic(error)),
        Err(_) => Err(JsErrorBox::generic("engine call was cancelled")),
    }
}

extension!(
    greppy_playwright,
    ops = [op_engine_call],
    options = { bridge: EngineBridge },
    state = |state, options| {
        state.put(options.bridge);
    },
);

fn main() -> io::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(io::Error::other)?;
    let _enter = runtime.enter();
    run(runtime)
}

fn run(tokio_runtime: tokio::runtime::Runtime) -> io::Result<()> {
    let _capability = require_capability(std::env::args_os().skip(1))?;
    let stdout = Arc::new(Mutex::new(io::stdout()));
    let pending = Arc::new(Mutex::new(HashMap::new()));
    let bridge = EngineBridge {
        next_id: Arc::new(AtomicU64::new(1)),
        stdout: Arc::clone(&stdout),
        pending: Arc::clone(&pending),
    };
    let mut runtime = new_js_runtime(bridge.clone());
    runtime
        .execute_script("<web-controller-worker>", "1 + 1")
        .map_err(|error| io::Error::other(format!("JavaScript startup probe failed: {error}")))?;

    {
        let stdin = io::stdin();
        let mut stdin = stdin.lock();
        match read_message(&mut stdin)? {
            Message::Hello {
                worker: WorkerKind::Controller,
                ..
            } => {}
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
            let mut stdin = io::stdin();
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
        match recv_control(&control_rx)? {
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
                runtime = new_js_runtime(bridge.clone());
                let result = run_script(
                    &tokio_runtime,
                    &mut runtime,
                    &specifier,
                    source,
                    fixture_url,
                );
                let mut stdout = stdout.lock().unwrap_or_else(|error| error.into_inner());
                match result {
                    Ok(()) => write_message(
                        &mut *stdout,
                        &Message::script_complete(true, serde_json::Value::Null, None),
                    )?,
                    Err(error) => write_message(
                        &mut *stdout,
                        &Message::script_complete(false, serde_json::Value::Null, Some(error)),
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

fn new_js_runtime(bridge: EngineBridge) -> JsRuntime {
    JsRuntime::new(RuntimeOptions {
        module_loader: Some(Rc::new(PlaywrightLoader)),
        extensions: vec![greppy_playwright::init(bridge)],
        ..Default::default()
    })
}

fn recv_control(control_rx: &mpsc::Receiver<io::Result<Message>>) -> io::Result<Message> {
    match control_rx.recv_timeout(MESSAGE_TIMEOUT) {
        Ok(message) => message,
        Err(RecvTimeoutError::Timeout) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "timed out waiting for supervisor message",
        )),
        Err(RecvTimeoutError::Disconnected) => Err(io::Error::new(
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
