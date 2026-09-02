//! Worker bridge events and bounded stream coalescing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use super::session::PersistedMessage;
use crate::agent_control::ConnId;

#[derive(Debug, Clone)]
pub struct RemoteRequest {
    pub conn: ConnId,
    pub id: Value,
    pub method: String,
    pub params: Value,
}

const STREAM_CAP_BYTES: usize = 256 * 1024;
const DISCRETE_CAPACITY: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionCommand {
    Prompt(String),
    RemotePrompt { text: String, source: String },
    Cancel,
    SetModel(String),
    SetEndpoint(String),
    Resume(String),
    Compact,
    Quit,
}

#[derive(Debug, Clone)]
pub enum SessionEvent {
    SetupProgress {
        phase: String,
        detail: Option<String>,
        unit: String,
        completed: usize,
        total: usize,
        rate_milli_per_second: Option<u64>,
        eta_seconds: Option<u64>,
        elapsed_seconds: u64,
    },
    BackgroundProgress {
        phase: String,
        detail: Option<String>,
        unit: String,
        completed: usize,
        total: usize,
        rate_milli_per_second: Option<u64>,
        eta_seconds: Option<u64>,
    },
    BackgroundReady,
    SetupReady,
    #[allow(dead_code)]
    SetupBlocked(String),
    GatewayRequired(String),
    EndpointRejected {
        endpoint: String,
        message: String,
    },
    Configuration {
        endpoint: String,
        model: String,
        models: Vec<String>,
    },
    #[allow(dead_code)]
    Text(String),
    #[allow(dead_code)]
    Thinking(String),
    ToolStart {
        id: String,
        summary: String,
    },
    ToolFinish {
        id: String,
        failed: bool,
        elapsed_ms: u64,
        preview: String,
    },
    Done {
        input_tokens: u64,
        output_tokens: u64,
        cache_read: u64,
        cache_write: u64,
        turns: u64,
        stop: String,
        #[allow(dead_code)]
        messages: Vec<PersistedMessage>,
    },
    Compacted {
        #[allow(dead_code)]
        messages: Vec<PersistedMessage>,
    },
    Error(String),
    Warning(String),
}

#[derive(Debug, Default)]
struct StreamBuf {
    text: String,
    thinking: String,
    dropped: u64,
}

#[derive(Debug, Clone)]
pub struct EventBridge {
    tx: SyncSender<SessionEvent>,
    stream: Arc<Mutex<StreamBuf>>,
    setup: Arc<Mutex<Option<SessionEvent>>>,
    saturated: Arc<AtomicBool>,
    dropped: Arc<AtomicU64>,
}

pub struct EventIntake {
    rx: Receiver<SessionEvent>,
    stream: Arc<Mutex<StreamBuf>>,
    setup: Arc<Mutex<Option<SessionEvent>>>,
    saturated: Arc<AtomicBool>,
}

pub fn bounded_pair() -> (EventBridge, EventIntake) {
    let (tx, rx) = mpsc::sync_channel(DISCRETE_CAPACITY);
    let stream = Arc::new(Mutex::new(StreamBuf::default()));
    let setup = Arc::new(Mutex::new(None));
    let saturated = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicU64::new(0));
    (
        EventBridge {
            tx,
            stream: Arc::clone(&stream),
            setup: Arc::clone(&setup),
            saturated: Arc::clone(&saturated),
            dropped: Arc::clone(&dropped),
        },
        EventIntake {
            rx,
            stream,
            setup,
            saturated,
        },
    )
}

impl EventBridge {
    pub fn send_setup_progress(&self, event: SessionEvent) {
        debug_assert!(matches!(
            event,
            SessionEvent::SetupProgress { .. } | SessionEvent::BackgroundProgress { .. }
        ));
        if let Ok(mut latest) = self.setup.lock() {
            *latest = Some(event);
        }
    }

    pub fn send_text(&self, delta: &str) {
        append_capped(&self.stream, true, delta, &self.dropped);
    }

    pub fn send_thinking(&self, delta: &str) {
        append_capped(&self.stream, false, delta, &self.dropped);
    }

    pub fn send_discrete(&self, event: SessionEvent) {
        match self.tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_event)) => {
                // Lifecycle events are intentionally ordered while admitted to
                // the bounded queue. Once the queue is full, dropping the new
                // event is the only bounded, non-blocking fallback; the UI is
                // told about the loss through `Intake::saturated` on its next
                // poll. Never switch to `send` here: the worker can be the
                // only producer and a busy UI must not stall the agent loop.
                self.saturated.store(true, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {}
        }
    }

    #[allow(dead_code)]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl EventIntake {
    pub fn poll(&self, timeout: Duration) -> Intake {
        let mut discrete = Vec::new();
        match self.rx.recv_timeout(timeout) {
            Ok(event) => discrete.push(event),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                append_setup(&self.setup, &mut discrete);
                return Intake {
                    discrete,
                    text: take_field(&self.stream, true),
                    thinking: take_field(&self.stream, false),
                    disconnected: true,
                    saturated: self.saturated.swap(false, Ordering::Relaxed),
                };
            }
        }
        loop {
            match self.rx.try_recv() {
                Ok(event) => discrete.push(event),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    append_setup(&self.setup, &mut discrete);
                    return Intake {
                        discrete,
                        text: take_field(&self.stream, true),
                        thinking: take_field(&self.stream, false),
                        disconnected: true,
                        saturated: self.saturated.swap(false, Ordering::Relaxed),
                    };
                }
            }
        }
        append_setup(&self.setup, &mut discrete);
        Intake {
            discrete,
            text: take_field(&self.stream, true),
            thinking: take_field(&self.stream, false),
            disconnected: false,
            saturated: self.saturated.swap(false, Ordering::Relaxed),
        }
    }

    pub fn try_poll(&self) -> Intake {
        self.poll(Duration::from_millis(0))
    }
}

fn append_setup(setup: &Arc<Mutex<Option<SessionEvent>>>, discrete: &mut Vec<SessionEvent>) {
    if let Ok(mut latest) = setup.lock() {
        if let Some(event) = latest.take() {
            // Progress describes work which happened before any subsequently
            // admitted lifecycle event (most importantly SetupReady). Apply it
            // first so a late coalesced snapshot cannot put the UI back into
            // Setup after readiness was announced.
            discrete.insert(0, event);
        }
    }
}

#[derive(Debug, Default)]
pub struct Intake {
    pub discrete: Vec<SessionEvent>,
    pub text: String,
    pub thinking: String,
    pub disconnected: bool,
    pub saturated: bool,
}

fn append_capped(buf: &Arc<Mutex<StreamBuf>>, text: bool, delta: &str, dropped: &Arc<AtomicU64>) {
    let Ok(mut guard) = buf.lock() else {
        return;
    };
    let target = if text {
        &mut guard.text
    } else {
        &mut guard.thinking
    };
    target.push_str(delta);
    if target.len() > STREAM_CAP_BYTES {
        let overflow = target.len() - STREAM_CAP_BYTES;
        target.drain(..overflow);
        while !target.is_empty() && !target.is_char_boundary(0) {
            target.remove(0);
        }
        guard.dropped = guard.dropped.saturating_add(1);
        dropped.fetch_add(1, Ordering::Relaxed);
    }
}

fn take_field(buf: &Arc<Mutex<StreamBuf>>, text: bool) -> String {
    let Ok(mut guard) = buf.lock() else {
        return String::new();
    };
    if text {
        std::mem::take(&mut guard.text)
    } else {
        std::mem::take(&mut guard.thinking)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_deltas_coalesce_instead_of_unbounded_queue() {
        let (bridge, intake) = bounded_pair();
        for _ in 0..10_000 {
            bridge.send_text("x");
        }
        let intake = intake.try_poll();
        assert!(intake.discrete.is_empty());
        assert_eq!(intake.text.len(), 10_000);
        assert!(!intake.disconnected);
    }

    #[test]
    fn stream_buffer_is_capped() {
        let (bridge, intake) = bounded_pair();
        bridge.send_text(&"a".repeat(STREAM_CAP_BYTES + 32));
        let intake = intake.try_poll();
        assert_eq!(intake.text.len(), STREAM_CAP_BYTES);
        assert!(bridge.dropped() >= 1);
    }

    #[test]
    fn discrete_saturation_is_reported() {
        let (bridge, intake) = bounded_pair();
        for i in 0..(DISCRETE_CAPACITY + 4) {
            bridge.send_discrete(SessionEvent::Warning(format!("{i}")));
        }
        let snapshot = intake.try_poll();
        assert!(snapshot.saturated);
        assert_eq!(snapshot.discrete.len(), DISCRETE_CAPACITY);
        let warnings: Vec<_> = snapshot
            .discrete
            .iter()
            .map(|event| match event {
                SessionEvent::Warning(message) => message.clone(),
                _ => panic!("expected warning event"),
            })
            .collect();
        assert_eq!(
            warnings,
            (0..DISCRETE_CAPACITY)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
        );
        assert!(!intake.try_poll().saturated);
    }

    #[test]
    fn discrete_events_stay_ordered_while_stream_deltas_coalesce() {
        let (bridge, intake) = bounded_pair();
        bridge.send_text("hel");
        bridge.send_discrete(SessionEvent::Warning("first".to_string()));
        bridge.send_text("lo");
        bridge.send_thinking("think-");
        bridge.send_discrete(SessionEvent::Warning("second".to_string()));
        bridge.send_thinking("ing");

        let snapshot = intake.try_poll();
        assert_eq!(snapshot.text, "hello");
        assert_eq!(snapshot.thinking, "think-ing");
        assert_eq!(snapshot.discrete.len(), 2);
        assert!(
            matches!(&snapshot.discrete[0], SessionEvent::Warning(message) if message == "first")
        );
        assert!(
            matches!(&snapshot.discrete[1], SessionEvent::Warning(message) if message == "second")
        );
        assert!(!snapshot.saturated);
    }

    #[test]
    fn receiver_surfaces_sender_disconnect_without_waiting() {
        let (bridge, intake) = bounded_pair();
        drop(bridge);

        let snapshot = intake.try_poll();
        assert!(snapshot.disconnected);
        assert!(snapshot.discrete.is_empty());
    }

    #[test]
    fn setup_progress_coalesces_before_readiness_without_saturating() {
        let (bridge, intake) = bounded_pair();
        for completed in 0..10_000 {
            bridge.send_setup_progress(SessionEvent::SetupProgress {
                phase: "indexing".into(),
                detail: None,
                unit: "Dateien".into(),
                completed,
                total: 10_000,
                rate_milli_per_second: None,
                eta_seconds: None,
                elapsed_seconds: 1,
            });
        }
        bridge.send_discrete(SessionEvent::SetupReady);

        let snapshot = intake.try_poll();
        assert!(!snapshot.saturated);
        assert_eq!(snapshot.discrete.len(), 2);
        assert!(matches!(
            &snapshot.discrete[0],
            SessionEvent::SetupProgress {
                completed: 9_999,
                ..
            }
        ));
        assert!(matches!(&snapshot.discrete[1], SessionEvent::SetupReady));
    }
}
