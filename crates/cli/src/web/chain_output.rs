//! Opt-in reduction of automatic action observations. Explicit replies pass through.
//! Earlier observations remain in a private, immutable view history.
use super::view::Scope;
use greppy_core::error::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;

const HISTORY_LIMIT: usize = 8 * 1024 * 1024;
thread_local! {
    static ACTIVE: RefCell<Option<Buffer>> = const { RefCell::new(None) };
    static MACHINE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

pub(super) struct MachineMode(bool, PhantomData<Rc<()>>);

pub(super) fn machine_mode(enabled: bool) -> MachineMode {
    if enabled {
        MACHINE_DEPTH.with(|depth| depth.set(depth.get() + 1));
    }
    MachineMode(enabled, PhantomData)
}

pub(super) fn machine_active() -> bool {
    MACHINE_DEPTH.with(|depth| depth.get() != 0)
}

impl Drop for MachineMode {
    fn drop(&mut self) {
        if self.0 {
            MACHINE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        }
    }
}

#[derive(Clone, Serialize)]
struct Reply {
    step: usize,
    verb: String,
    scope: Scope,
    payload: Value,
}

#[derive(Default)]
struct Batch {
    earlier: Vec<Reply>,
    last: Option<Reply>,
}

enum Emission {
    Reply(Reply),
    Note(String),
}

impl Batch {
    fn plan(self, dir: &Path) -> Vec<Emission> {
        let Some(last) = self.last else {
            return Vec::new();
        };
        if self.earlier.is_empty() {
            return vec![Emission::Reply(last)];
        }
        let records = serde_json::to_value(&self.earlier).expect("JSON replies serialize");
        match super::view::archive_chain(&records, last.scope.clone(), dir) {
            Ok(command) => vec![
                Emission::Reply(last),
                Emission::Note(format!(
                    "{} earlier automatic observations archived (not current state); {command}",
                    self.earlier.len()
                )),
            ],
            Err(error) => {
                let mut output = vec![Emission::Note(format!(
                    "Chain history unavailable: {}; showing all observations; no action repeated.",
                    json!(error)
                ))];
                output.extend(self.earlier.into_iter().map(Emission::Reply));
                output.push(Emission::Reply(last));
                output
            }
        }
    }
}

struct Buffer {
    step: usize,
    verb: String,
    deferred: bool,
    batch: Batch,
    bytes: usize,
    dir: PathBuf,
}

impl Buffer {
    fn new(dir: PathBuf) -> Self {
        Self {
            step: 0,
            verb: String::new(),
            deferred: false,
            batch: Batch::default(),
            bytes: 0,
            dir,
        }
    }

    fn take(&mut self) -> Batch {
        self.bytes = 0;
        std::mem::take(&mut self.batch)
    }

    fn flush(&mut self) -> Result<()> {
        let batch = self.take();
        for item in batch.plan(&self.dir) {
            match item {
                Emission::Reply(reply) => {
                    println!("observation after step {} ({})", reply.step, reply.verb);
                    super::common::emit_web_uncaptured(false, &reply.payload, reply.scope)?;
                }
                Emission::Note(note) => println!("{note}"),
            }
        }
        Ok(())
    }

    fn push(&mut self, reply: Reply, bytes: usize) {
        if let Some(previous) = self.batch.last.replace(reply) {
            self.batch.earlier.push(previous);
        }
        self.bytes += bytes;
        self.deferred = true;
    }
}

/// Only one owner per thread. Nested chains keep their explicit output.
pub(super) struct Guard {
    active: bool,
    _not_send: PhantomData<Rc<()>>,
}

pub(super) fn start(enabled: bool, dir: PathBuf) -> Option<Guard> {
    if !enabled {
        return None;
    }
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return None;
        }
        *slot = Some(Buffer::new(dir));
        Some(Guard {
            active: true,
            _not_send: PhantomData,
        })
    })
}

impl Guard {
    pub(super) fn step(&self, step: usize, verb: &str) {
        ACTIVE.with(|slot| {
            if let Some(buffer) = slot.borrow_mut().as_mut() {
                buffer.step = step;
                buffer.verb = verb.to_owned();
                buffer.deferred = false;
            }
        });
    }

    pub(super) fn deferred(&self) -> bool {
        ACTIVE.with(|slot| slot.borrow().as_ref().is_some_and(|buffer| buffer.deferred))
    }

    pub(super) fn finish(mut self) -> Result<()> {
        self.active = false;
        ACTIVE.with(|slot| {
            if let Some(mut buffer) = slot.borrow_mut().take() {
                buffer.flush()?;
            }
            Ok(())
        })
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.active {
            // Preserve observations even when dispatch exits through a Rust error.
            let result: Result<()> = ACTIVE.with(|slot| {
                if let Some(mut buffer) = slot.borrow_mut().take() {
                    buffer.flush()?;
                }
                Ok(())
            });
            if let Err(error) = result {
                eprintln!("chain output could not be completed: {error}; no action repeated");
            }
        }
    }
}

fn only_keys(value: &Value, keys: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.keys().all(|key| keys.contains(&key.as_str())))
}

fn automatic(verb: &str, payload: &Value) -> bool {
    if ![
        "open", "goto", "back", "forward", "reload", "click", "fill", "type", "clear", "select",
        "check", "uncheck", "press", "hover", "scroll", "upload", "wait",
    ]
    .contains(&verb)
    {
        return false;
    }
    let operation = if verb == "open" {
        "web.goto".to_owned()
    } else {
        format!("web.{verb}")
    };
    if payload["schema"] != "greppy.web-runtime.v1"
        || payload["operation"] != operation
        || payload["status"] != "ok"
        || !payload["error"].is_null()
        || !payload["handshake"].is_null()
        || !(payload["artifacts"].is_null() || payload["artifacts"] == json!([]))
        || !only_keys(
            payload,
            &[
                "schema",
                "request_id",
                "operation",
                "status",
                "result",
                "error",
                "metrics",
                "artifacts",
                "handshake",
            ],
        )
    {
        return false;
    }
    let result = &payload["result"];
    let valid_result = if verb == "wait" {
        result["held"] == true
            && result["wait_backend"] == "native_v1"
            && ["session_id", "tab_id", "document_id"]
                .iter()
                .all(|key| result[*key].as_str().is_some_and(|id| !id.is_empty()))
            && result["waited_ms"].as_u64().is_some()
            && only_keys(
                result,
                &[
                    "session_id",
                    "tab_id",
                    "document_id",
                    "held",
                    "waited_ms",
                    "detail",
                    "wait_backend",
                    "page_state",
                    "untrusted_content_boundary",
                ],
            )
    } else {
        result["ok"] == true
            && only_keys(
                result,
                &[
                    "ok",
                    "dispatch",
                    "session_id",
                    "tab_id",
                    "url",
                    "status",
                    "untrusted_content_boundary",
                    "page_state",
                ],
            )
    };
    if !valid_result {
        return false;
    }
    if !result["status"].is_null()
        && !result["status"]
            .as_u64()
            .is_some_and(|status| (200..400).contains(&status))
    {
        return false;
    }
    let state = &result["page_state"];
    if state["schema"] != "greppy.web.page-state.v1"
        || state["status"] != "available"
        || !only_keys(state, &["schema", "status", "snapshot"])
    {
        return false;
    }
    let snapshot = &state["snapshot"];
    if snapshot["actionable_schema"] != "greppy.web.actionable.v2"
        || snapshot["refs_truncated"] != false
        || !only_keys(
            snapshot,
            &[
                "actionable_schema",
                "actionables",
                "headings",
                "links",
                "ref_count",
                "refs_truncated",
                "text",
                "title",
                "untrusted_content_boundary",
                "url",
            ],
        )
    {
        return false;
    }
    let Some(controls) = snapshot["actionables"].as_array() else {
        return false;
    };
    if snapshot["ref_count"].as_u64() != Some(controls.len() as u64) {
        return false;
    }
    controls.iter().all(|control| {
        only_keys(
            control,
            &[
                "ref",
                "role",
                "tag",
                "name",
                "text",
                "value",
                "type",
                "checked",
                "selected",
                "disabled",
                "expanded",
                "invalid",
                "href",
                "name_source",
                "name_truncated",
                "selected_options",
                "selected_options_truncated",
                "value_redacted",
                "value_truncated",
                "select_choices",
            ],
        ) && [
            "invalid",
            "name_truncated",
            "selected_options_truncated",
            "value_truncated",
        ]
        .iter()
        .all(|key| control[*key].is_null() || control[*key] == false)
            && (control["select_choices"].is_null()
                || (super::view::compact_select_choices(&control["select_choices"]).is_some()
                    && control["select_choices"]["choices_truncated"] == false
                    && control["select_choices"]["choices"]
                        .as_array()
                        .is_some_and(|choices| {
                            choices.iter().all(|choice| {
                                choice["value_truncated"] == false
                                    && choice["label_truncated"] == false
                            })
                        })))
    })
}

pub(super) fn capture(json_out: bool, payload: &Value, scope: &Scope) -> Result<bool> {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(buffer) = slot.as_mut() else {
            return Ok(false);
        };
        if !json_out && automatic(&buffer.verb, payload) {
            let scope = Scope {
                session: scope
                    .session
                    .clone()
                    .or_else(|| payload["result"]["session_id"].as_str().map(str::to_owned)),
                tab: scope
                    .tab
                    .clone()
                    .or_else(|| payload["result"]["tab_id"].as_str().map(str::to_owned)),
            };
            let bytes = payload.to_string().len();
            let receipt_session = payload["result"]["session_id"].as_str();
            let receipt_tab = payload["result"]["tab_id"].as_str();
            let consistent = receipt_session
                .is_none_or(|session| scope.session.as_deref() == Some(session))
                && receipt_tab.is_none_or(|tab| scope.tab.as_deref() == Some(tab));
            if consistent
                && scope.session.as_deref().is_some_and(|id| !id.is_empty())
                && scope.tab.as_deref().is_some_and(|id| !id.is_empty())
                && bytes <= HISTORY_LIMIT
            {
                if buffer.bytes + bytes > HISTORY_LIMIT
                    || buffer
                        .batch
                        .last
                        .as_ref()
                        .is_some_and(|last| last.scope != scope)
                {
                    buffer.flush()?;
                }
                let reply = Reply {
                    step: buffer.step,
                    verb: buffer.verb.clone(),
                    scope,
                    payload: payload.clone(),
                };
                buffer.push(reply, bytes);
                if buffer.verb == "wait" {
                    // The native wait observed this exact session/tab after its
                    // predicate held. Archive the preceding automatic states,
                    // but emit the full wait receipt now: a later action must
                    // never hide the explicit condition result in history.
                    buffer.flush()?;
                    buffer.deferred = false;
                }
                return Ok(true);
            }
        }
        buffer.deferred = false;
        // Legacy waits without a fresh native page-state receipt cannot replace
        // an observation. Their condition result remains explicit.
        let held_wait = !json_out
            && buffer.verb == "wait"
            && payload["operation"] == "web.wait"
            && payload["status"] == "ok"
            && payload["result"]["held"] == true
            && payload["result"]["page_state"].is_null()
            && payload["error"].is_null();
        if !held_wait {
            buffer.flush()?;
        }
        Ok(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> Scope {
        Scope {
            session: Some("session-a".into()),
            tab: Some("tab-a".into()),
        }
    }

    fn action(step: usize) -> Value {
        json!({"schema":"greppy.web-runtime.v1", "operation":"web.click", "status":"ok",
            "request_id":format!("request-{step}"), "artifacts":[], "metrics":{},
            "result":{"ok":true,"dispatch":"native","session_id":"session-a",
                "page_state":{"schema":"greppy.web.page-state.v1","status":"available",
                    "snapshot":{"actionable_schema":"greppy.web.actionable.v2", "title":"Checkout",
                        "text":format!("state {step}"),"ref_count":1,"refs_truncated":false,
                        "actionables":[{"ref":"@1","role":"checkbox","checked":step % 2 == 1,"disabled":false}]}}}})
    }

    fn take(mut guard: Guard) -> Buffer {
        guard.active = false;
        ACTIVE.with(|slot| slot.borrow_mut().take().unwrap())
    }

    fn native_wait() -> Value {
        let mut payload = action(2);
        payload["operation"] = json!("web.wait");
        let state = payload["result"]["page_state"].clone();
        payload["result"] = json!({"session_id":"session-a", "tab_id":"tab-a",
            "document_id":"document-current", "held":true, "waited_ms":7,
            "wait_backend":"native_v1", "detail":null, "page_state":state});
        payload
    }

    #[test]
    fn native_wait_emits_current_state_and_archives_only_earlier_action() {
        let dir = tempfile::tempdir().unwrap();
        let guard = start(true, dir.path().into()).unwrap();
        guard.step(1, "click");
        let earlier = action(1);
        assert!(capture(false, &earlier, &scope()).unwrap());
        guard.step(2, "wait");
        let wait = native_wait();
        assert!(automatic("wait", &wait));
        assert!(capture(false, &wait, &scope()).unwrap());
        assert!(!guard.deferred());
        assert!(
            take(guard).batch.last.is_none(),
            "wait must not linger behind later actions"
        );
        let paths: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(paths.len(), 1);
        let saved: Value = serde_json::from_slice(&std::fs::read(&paths[0]).unwrap()).unwrap();
        let records: Vec<Value> = serde_json::from_str(saved["body"].as_str().unwrap()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["payload"], earlier);
        let shown = super::super::view::render(&wait, scope(), dir.path()).unwrap();
        assert!(shown.contains("state 2") && !shown.contains("state 1"));
        assert!(shown.contains("\"held\":true") && shown.contains("\"waited_ms\":7"));
    }

    #[test]
    fn native_wait_errors_unknown_data_and_scope_mismatch_do_not_replace_history() {
        let base = native_wait();
        for (path, replacement) in [
            (vec!["status"], json!("error")),
            (vec!["result", "held"], json!(false)),
            (vec!["result", "wait_backend"], json!("future_v2")),
            (vec!["result", "future_detail"], json!(false)),
            (vec!["result", "page_state", "status"], json!("unavailable")),
            (
                vec!["result", "page_state", "snapshot", "refs_truncated"],
                json!(true),
            ),
        ] {
            let mut changed = base.clone();
            let mut field = &mut changed;
            for key in path {
                field = &mut field[key];
            }
            *field = replacement;
            assert!(!automatic("wait", &changed));
        }
        for key in ["session_id", "tab_id"] {
            let dir = tempfile::tempdir().unwrap();
            let guard = start(true, dir.path().into()).unwrap();
            guard.step(1, "click");
            assert!(capture(false, &action(1), &scope()).unwrap());
            guard.step(2, "wait");
            let mut changed = base.clone();
            changed["result"][key] = json!("other");
            assert!(!capture(false, &changed, &scope()).unwrap());
            assert!(take(guard).batch.last.is_none());
            assert_eq!(
                std::fs::read_dir(dir.path()).unwrap().count(),
                0,
                "scope mismatch must show, not archive, the previous state"
            );
        }
        for key in ["session_id", "tab_id", "document_id"] {
            let mut changed = base.clone();
            changed["result"][key] = json!("");
            assert!(
                !automatic("wait", &changed),
                "empty {key} is not an identity"
            );
        }
    }

    #[test]
    fn unknown_action_tab_is_never_treated_as_a_shared_identity() {
        let dir = tempfile::tempdir().unwrap();
        let guard = start(true, dir.path().into()).unwrap();
        let unknown_tab = Scope {
            session: scope().session,
            tab: None,
        };
        for step in 1..=2 {
            guard.step(step, "click");
            assert!(!capture(false, &action(step), &unknown_tab).unwrap());
            assert!(!guard.deferred());
        }
        assert!(take(guard).batch.last.is_none());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn known_complete_select_choices_coalesce_but_truncated_or_unknown_do_not() {
        let mut payload = action(1);
        payload["result"]["page_state"]["snapshot"]["actionables"][0]["select_choices"] = json!({
            "schema":"greppy.web.select-choices.v1", "choices_total":1, "choices_truncated":false,
            "choices":[{"value":"", "label":"Empty", "disabled":true,
                "value_truncated":false, "label_truncated":false}]});
        assert!(automatic("click", &payload));
        for (path, replacement) in [
            (vec!["schema"], json!("greppy.web.select-choices.v99")),
            (vec!["future_flag"], json!(false)),
            (vec!["choices_truncated"], json!(true)),
        ] {
            let mut changed = payload.clone();
            let mut value = &mut changed["result"]["page_state"]["snapshot"]["actionables"][0]
                ["select_choices"];
            for key in path {
                value = &mut value[key];
            }
            *value = replacement;
            assert!(!automatic("click", &changed));
        }
        payload["result"]["page_state"]["snapshot"]["actionables"][0]["select_choices"]
            ["choices"][0]["label_truncated"] = json!(true);
        assert!(!automatic("click", &payload));
    }

    #[test]
    fn eight_actions_emit_last_state_and_preserve_exact_private_history() {
        let dir = tempfile::tempdir().unwrap();
        let guard = start(true, dir.path().into()).unwrap();
        for step in 1..=8 {
            guard.step(step, "click");
            assert!(capture(false, &action(step), &scope()).unwrap());
            assert!(guard.deferred());
        }
        let mut buffer = take(guard);
        let plan = buffer.take().plan(dir.path());
        assert_eq!(plan.len(), 2);
        let Emission::Reply(last) = &plan[0] else {
            panic!("final state missing")
        };
        assert_eq!(last.step, 8);
        assert_eq!(last.payload, action(8));
        let Emission::Note(note) = &plan[1] else {
            panic!("history pointer missing")
        };
        assert!(note.contains("7 earlier automatic observations"));
        assert!(note.contains("greppy web result next"));
        let files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(files.len(), 1);
        let saved: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
        assert!(saved["header"]
            .as_str()
            .unwrap()
            .contains("not current page state"));
        assert_eq!(saved["scope"]["session"], "session-a");
        let history: Vec<Value> = serde_json::from_str(saved["body"].as_str().unwrap()).unwrap();
        assert_eq!(history.len(), 7);
        for (index, record) in history.iter().enumerate() {
            assert_eq!(record["step"], index + 1);
            assert_eq!(record["payload"], action(index + 1));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&files[0]).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn unknown_fields_errors_and_truncation_never_coalesce() {
        let base = action(1);
        assert!(automatic("click", &base));
        let paths = [
            vec!["new_warning"],
            vec!["result", "unknown_result"],
            vec!["result", "page_state", "new_revision"],
            vec!["result", "page_state", "snapshot", "new_relation"],
        ];
        for path in paths {
            let mut changed = base.clone();
            let mut value = &mut changed;
            for key in path {
                value = &mut value[key];
            }
            *value = json!(false); // Even false unknown data must be retained.
            assert!(!automatic("click", &changed));
        }
        for key in [
            "invalid",
            "name_truncated",
            "value_truncated",
            "selected_options_truncated",
            "future_control_flag",
        ] {
            let mut changed = base.clone();
            changed["result"]["page_state"]["snapshot"]["actionables"][0][key] = json!(true);
            assert!(!automatic("click", &changed), "{key}");
        }
        let mut changed = base.clone();
        changed["status"] = json!("error");
        assert!(!automatic("click", &changed));
        let mut changed = base.clone();
        changed["result"]["page_state"]["status"] = json!("unavailable");
        assert!(!automatic("click", &changed));
        let mut changed = base.clone();
        changed["result"]["page_state"]["snapshot"]["ref_count"] = json!(2);
        assert!(!automatic("click", &changed));
        assert!(!automatic("observe", &base));
        assert!(!automatic("extract", &base));
    }

    #[test]
    fn failed_history_storage_shows_every_observation_without_replay() {
        let blocked = tempfile::NamedTempFile::new().unwrap();
        let mut buffer = Buffer::new(blocked.path().into());
        for step in 1..=3 {
            buffer.push(
                Reply {
                    step,
                    verb: "click".into(),
                    scope: scope(),
                    payload: action(step),
                },
                1,
            );
        }
        let plan = buffer.take().plan(blocked.path());
        let Emission::Note(note) = &plan[0] else {
            panic!("storage warning missing")
        };
        assert!(note.contains("no action repeated"));
        let shown: Vec<_> = plan
            .into_iter()
            .filter_map(|entry| match entry {
                Emission::Reply(reply) => Some(reply.payload),
                _ => None,
            })
            .collect();
        assert_eq!(shown, vec![action(1), action(2), action(3)]);
    }

    #[test]
    fn wait_preserves_last_observation_but_explicit_query_flushes_it() {
        let dir = tempfile::tempdir().unwrap();
        let guard = start(true, dir.path().into()).unwrap();
        guard.step(1, "click");
        assert!(capture(false, &action(1), &scope()).unwrap());
        guard.step(2, "wait");
        let held = json!({"operation":"web.wait","status":"ok","result":{"held":true}});
        assert!(!capture(false, &held, &scope()).unwrap());
        ACTIVE.with(|slot| {
            assert_eq!(
                slot.borrow()
                    .as_ref()
                    .unwrap()
                    .batch
                    .last
                    .as_ref()
                    .unwrap()
                    .step,
                1
            )
        });
        guard.step(3, "observe");
        let query =
            json!({"operation":"web.observe","status":"ok","result":{"text":"Requested scope"}});
        assert!(!capture(false, &query, &scope()).unwrap());
        assert!(take(guard).batch.last.is_none());
    }

    #[test]
    fn session_or_tab_switch_separates_observation_histories() {
        for next_scope in [
            Scope {
                session: Some("other".into()),
                tab: Some("tab-a".into()),
            },
            Scope {
                session: Some("session-a".into()),
                tab: Some("tab-b".into()),
            },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let guard = start(true, dir.path().into()).unwrap();
            for step in 1..=2 {
                guard.step(step, "click");
                assert!(capture(false, &action(step), &scope()).unwrap());
            }
            guard.step(3, "click");
            let mut next = action(3);
            next["result"]["session_id"] = json!(next_scope.session);
            next["result"]["tab_id"] = json!(next_scope.tab);
            assert!(capture(false, &next, &next_scope).unwrap());
            let buffer = take(guard);
            assert!(buffer.batch.earlier.is_empty());
            assert_eq!(buffer.batch.last.unwrap().scope, next_scope);
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }

    #[test]
    fn contradictory_scope_and_oversized_payload_remain_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let guard = start(true, dir.path().into()).unwrap();
        guard.step(1, "click");
        let contradictory = Scope {
            session: Some("other-session".into()),
            tab: None,
        };
        assert!(!capture(false, &action(1), &contradictory).unwrap());
        let mut huge = action(1);
        huge["result"]["page_state"]["snapshot"]["text"] = json!("x".repeat(HISTORY_LIMIT));
        assert!(!capture(false, &huge, &scope()).unwrap());
        assert!(take(guard).batch.last.is_none());
    }

    #[test]
    fn machine_mode_is_scoped_and_nesting_restores_the_parent() {
        assert!(!machine_active());
        let outer = machine_mode(true);
        assert!(machine_active());
        {
            let _human_nested = machine_mode(false);
            let _machine_nested = machine_mode(true);
            assert!(machine_active());
        }
        assert!(machine_active());
        drop(outer);
        assert!(!machine_active());
    }

    #[test]
    fn machine_reply_is_not_captured_and_nested_owner_cannot_replace_outer() {
        let dir = tempfile::tempdir().unwrap();
        assert!(start(false, dir.path().into()).is_none());
        let guard = start(true, dir.path().into()).unwrap();
        assert!(start(true, dir.path().into()).is_none());
        guard.step(1, "click");
        assert!(!capture(true, &action(1), &scope()).unwrap());
        assert!(take(guard).batch.last.is_none());
    }

    #[test]
    fn history_budget_flushes_before_accumulating_an_unbounded_chain() {
        let dir = tempfile::tempdir().unwrap();
        let guard = start(true, dir.path().into()).unwrap();
        guard.step(1, "click");
        assert!(capture(false, &action(1), &scope()).unwrap());
        ACTIVE.with(|slot| slot.borrow_mut().as_mut().unwrap().bytes = HISTORY_LIMIT);
        guard.step(2, "click");
        assert!(capture(false, &action(2), &scope()).unwrap());
        let buffer = take(guard);
        assert!(buffer.batch.earlier.is_empty());
        assert_eq!(buffer.batch.last.unwrap().step, 2);
        assert!(buffer.bytes < HISTORY_LIMIT);
    }
}
