use super::*;

// Test-only relocation of identical, separately hash-verified runtime bytes.
// Default behavior continues to test the Cargo-produced runtime in place.
fn workflow_supervisor(socket: &Path, run_id: &str, fixture: &str) -> Supervisor {
    let Some(runtime) = std::env::var_os("GREPPY_WORKFLOW_TEST_RUNTIME") else {
        return Supervisor::spawn(socket, run_id, |command| {
            command.arg("--fixture-url").arg(fixture);
        });
    };
    let mut command = Command::new(runtime);
    command
        .arg("--socket")
        .arg(socket)
        .arg("--run-id")
        .arg(run_id)
        .arg("--fixture-url")
        .arg(fixture)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .process_group(0);
    Supervisor::finish_spawn(socket, run_id, command, TEST_DEADLINE)
}

const PAGE: &str = r#"<!doctype html><html><body>
<label>Value <input id="value" value="1"></label>
<button id="save" onclick="window.saves++; const v=document.getElementById('value').value; setTimeout(() => { document.getElementById('done').textContent='Saved '+v; }, 120)">Save</button>
<button id="increment" onclick="window.count++; document.getElementById('count').textContent='Count '+window.count">Increment</button>
<p id="done">Not saved</p><p id="count">Count 0</p>
<script>window.count=0;window.saves=0;</script>
</body></html>"#;

#[test]
fn native_workflow_preflight_diagnoses_the_exact_field_before_any_mutation() {
    let fixture = serve_fixture(
        r#"<!doctype html><html><body>
      <select id="region"><option>All</option><option>EU</option></select>
      <input id="enabled" type="checkbox">
      <select id="sort"><option>default</option><option>ascending</option></select>
      <p>3 matching items</p>
      <script>window.changes=0;document.addEventListener('change',()=>window.changes++);</script>
    </body></html>"#,
    );
    let socket = std::env::temp_dir().join(format!(
        "greppy-workflow-diagnostic-{}.sock",
        std::process::id()
    ));
    let _guard = workflow_supervisor(&socket, "run_workflow_diagnostic", &fixture);
    wait_for_socket(&socket, Duration::from_secs(30));
    let call = |operation: &str, payload| {
        unix_request(
            &socket,
            &Request::new("run_workflow_diagnostic", operation, payload),
            Duration::from_secs(35),
        )
        .expect("workflow diagnostic request")
    };
    let created = call("web.session.create", json!({"profile":"project"}));
    assert_eq!(created.status, "ok", "{created:?}");
    let session = created.result.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let went = call("web.goto", json!({"session_id":session,"url":fixture}));
    assert_eq!(went.status, "ok", "{went:?}");
    let state = || {
        let result = call(
            "web.evaluate",
            json!({"session_id":session,"source":
            "JSON.stringify({region:document.getElementById('region').value,enabled:document.getElementById('enabled').checked,sort:document.getElementById('sort').value,changes:window.changes})"}),
        );
        assert_eq!(result.status, "ok", "{result:?}");
        serde_json::from_str::<serde_json::Value>(result.result.unwrap()["value"].as_str().unwrap())
            .unwrap()
    };
    let initial = state();
    let mut steps = json!([
        {"action":{"operation":"select","selector":{"type":"css","value":"#region"},"value":"EU"}},
        {"action":{"operation":"check","selector":{"type":"css","value":"#enabled"}}},
        {"action":{"operation":"select","selector":{"type":"css","value":"#sort"},"value":"ascending"},
         "expect":{"condition":{"query":"3 matching items"},"timeout_ms":3000}}
    ]);
    for _ in 0..2 {
        let response = call(
            "web.workflow",
            json!({"version":1,"session_id":session,"steps":steps}),
        );
        assert_eq!(response.status, "error", "{response:?}");
        let error = response.error.unwrap();
        assert_eq!(error.exit_code, 30);
        assert!(
            error.message.contains("step 3 expectation query"),
            "{error:?}"
        );
        assert!(error.message.contains("CSS"));
        assert!(error.next_action.contains("--expect 'text=EXPECTED TEXT'"));
        let detail = response.result.unwrap();
        assert_eq!(detail["phase"], "preflight");
        assert_eq!(detail["actions_attempted"], 0);
        assert_eq!(detail["completed_steps"], 0);
        assert_eq!(detail["preflight"]["field"], "expectation.query");
        assert_eq!(detail["preflight"]["syntax"], "css");
        assert_eq!(state(), initial, "failed preflight mutated a control");
    }
    steps[2]["expect"]["condition"]["query"] = json!("text=3 matching items");
    steps[2]["action"]["selector"]["value"] = json!("[");
    let response = call(
        "web.workflow",
        json!({"version":1,"session_id":session,"steps":steps}),
    );
    assert_eq!(response.status, "error", "{response:?}");
    let error = response.error.unwrap();
    assert!(
        error.message.contains("step 3 action selector"),
        "{error:?}"
    );
    assert!(!error.next_action.contains("--expect"));
    assert_eq!(response.result.unwrap()["actions_attempted"], 0);
    assert_eq!(state(), initial);
    steps[2]["action"]["selector"]["value"] = json!("#sort");
    let success = call(
        "web.workflow",
        json!({"version":1,"session_id":session,"steps":steps}),
    );
    assert_eq!(success.status, "ok", "{success:?}");
    assert_eq!(success.result.unwrap()["completed_steps"], 3);
    let changed = state();
    assert_eq!(changed["region"], "EU");
    assert_eq!(changed["enabled"], true);
    assert_eq!(changed["sort"], "ascending");
    assert!(changed["changes"].as_f64().unwrap() > 0.0);
}

#[test]
fn native_workflow_preflights_all_steps_and_preserves_partial_effects() {
    let fixture = serve_fixture(PAGE);
    let socket = std::env::temp_dir().join(format!(
        "greppy-workflow-effects-{}.sock",
        std::process::id()
    ));
    let _guard = workflow_supervisor(&socket, "run_workflow_effects", &fixture);
    wait_for_socket(&socket, Duration::from_secs(30));
    let call = |operation: &str, payload| {
        unix_request(
            &socket,
            &Request::new("run_workflow_effects", operation, payload),
            Duration::from_secs(35),
        )
        .expect("workflow request")
    };
    let created = call("web.session.create", json!({"profile":"project"}));
    assert_eq!(created.status, "ok", "{created:?}");
    let session = created.result.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let went = call("web.goto", json!({"session_id":session,"url":fixture}));
    assert_eq!(went.status, "ok", "{went:?}");
    let page = went.result.unwrap()["tab_id"].as_str().unwrap().to_owned();
    let evaluate = |source: &str| {
        let response = call(
            "web.evaluate",
            json!({"session_id":session,"tab_id":page,"source":source}),
        );
        assert_eq!(response.status, "ok", "{response:?}");
        response.result.unwrap()["value"].clone()
    };
    let workflow = |steps| {
        call(
            "web.workflow",
            json!({"version":1,"session_id":session,"tab_id":page,"steps":steps}),
        )
    };
    let increment =
        json!({"action":{"operation":"click","selector":{"type":"css","value":"#increment"}}});

    let malformed = workflow(
        json!([increment, {"action":{"operation":"click","selector":{"type":"css","value":"["}}}]),
    );
    assert_eq!(malformed.status, "error", "{malformed:?}");
    let detail = malformed.result.unwrap();
    assert_eq!(detail["phase"], "preflight");
    assert_eq!(detail["actions_attempted"], 0);
    assert_eq!(detail["preflight"]["step"].as_f64(), Some(2.0));
    assert_eq!(evaluate("window.count").as_f64(), Some(0.0));

    let invalid_regex = workflow(
        json!([increment, {"expect":{"condition":{"query":"text~/[/","absent":true},"timeout_ms":100}}]),
    );
    assert_eq!(invalid_regex.status, "error", "{invalid_regex:?}");
    assert_eq!(invalid_regex.result.unwrap()["phase"], "preflight");
    assert_eq!(evaluate("window.count").as_f64(), Some(0.0));
    let arbitrary = workflow(
        json!([increment, {"expect":{"condition":{"source":"window.count=999;true"},"timeout_ms":100}}]),
    );
    assert_eq!(arbitrary.status, "error", "{arbitrary:?}");
    assert_eq!(evaluate("window.count").as_f64(), Some(0.0));

    let success = workflow(json!([
        {"action":{"operation":"fill","selector":{"type":"css","value":"#value"},"value":"3"}},
        {"action":{"operation":"click","selector":{"type":"css","value":"#save"}},"expect":{"condition":{"query":"text=Saved 3"},"timeout_ms":3000}},
        {"action":{"operation":"click","selector":{"type":"css","value":"#increment"}},"expect":{"condition":{"query":"text=Count 1"},"timeout_ms":1000}}
    ]));
    assert_eq!(success.status, "ok", "{success:?}");
    let detail = success.result.unwrap();
    assert_eq!(detail["completed_steps"], 3);
    assert_eq!(detail["actions_attempted"], 3);
    assert_eq!(detail["steps"][1]["expectation"]["result"]["held"], true);
    assert!(detail["steps"][0]["action"]["receipt"]
        .get("page_state")
        .is_none());
    assert!(detail["steps"][1]["expectation"]["result"]
        .get("page_state")
        .is_none());
    assert_eq!(detail["page_state"]["status"], "available");
    assert_eq!(
        evaluate("JSON.stringify({count:window.count,saves:window.saves,value:document.getElementById('value').value})"),
        r#"{"count":1,"saves":1,"value":"3"}"#
    );

    let timed_out = workflow(
        json!([increment, {"expect":{"condition":{"query":"css=#never"},"timeout_ms":75}}, increment]),
    );
    assert_eq!(timed_out.status, "error", "{timed_out:?}");
    assert_eq!(timed_out.error.unwrap().code, "TIMEOUT");
    let detail = timed_out.result.unwrap();
    assert_eq!(detail["failed_step"], 2);
    assert_eq!(detail["phase"], "expectation");
    assert_eq!(detail["completed_steps"], 1);
    assert_eq!(detail["actions_attempted"], 1);
    assert_eq!(detail["rolled_back"], false);
    assert_eq!(evaluate("window.count").as_f64(), Some(2.0));

    let observed = call("web.observe", json!({"session_id":session,"tab_id":page}));
    assert_eq!(observed.status, "ok", "{observed:?}");
    let observation = observed.result.unwrap();
    let old = observation["actionables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["name"] == "Increment")
        .unwrap()["ref"]
        .as_str()
        .unwrap()
        .to_owned();
    evaluate("document.getElementById('increment').replaceWith(document.getElementById('increment').cloneNode(true)); true");
    let stale = workflow(
        json!([{"action":{"operation":"click","selector":{"type":"ref","value":old[1..].parse::<u64>().unwrap()}}}]),
    );
    assert_eq!(stale.status, "error", "{stale:?}");
    assert_eq!(stale.error.unwrap().code, "STALE_REF");
    assert_eq!(evaluate("window.count").as_f64(), Some(2.0));
    let stale_expect = workflow(
        json!([{"action":{"operation":"click","selector":{"type":"css","value":"#increment"}},"expect":{"condition":{"query":old,"absent":true},"timeout_ms":500}}]),
    );
    assert_eq!(stale_expect.status, "error", "{stale_expect:?}");
    assert_eq!(stale_expect.error.unwrap().code, "STALE_REF");
    assert_eq!(stale_expect.result.unwrap()["phase"], "expectation");
    assert_eq!(evaluate("window.count").as_f64(), Some(3.0));
}

#[test]
fn native_workflow_keeps_explicit_tab_through_navigation_and_request_timeout() {
    let fixture = serve_fixture(PAGE);
    let socket =
        std::env::temp_dir().join(format!("greppy-workflow-scope-{}.sock", std::process::id()));
    let _guard = workflow_supervisor(&socket, "run_workflow_scope", &fixture);
    wait_for_socket(&socket, Duration::from_secs(30));
    let call = |operation: &str, payload| {
        unix_request(
            &socket,
            &Request::new("run_workflow_scope", operation, payload),
            Duration::from_secs(35),
        )
        .expect("workflow request")
    };
    let created = call("web.session.create", json!({"profile":"project"}));
    assert_eq!(created.status, "ok", "{created:?}");
    let session = created.result.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let went = call("web.goto", json!({"session_id":session,"url":fixture}));
    assert_eq!(went.status, "ok", "{went:?}");
    let first = went.result.unwrap()["tab_id"].as_str().unwrap().to_owned();
    let other = call("web.tab.new", json!({"session_id":session}));
    assert_eq!(other.status, "ok", "{other:?}");
    let second = other.result.unwrap()["tab"].as_str().unwrap().to_owned();
    assert_ne!(first, second);
    let other_went = call(
        "web.goto",
        json!({"session_id":session,"tab_id":second,"url":fixture}),
    );
    assert_eq!(other_went.status, "ok", "{other_went:?}");
    let increment =
        json!({"action":{"operation":"click","selector":{"type":"css","value":"#increment"}}});
    let explicit = call(
        "web.workflow",
        json!({"version":1,"session_id":session,"tab_id":first,"steps":[increment]}),
    );
    assert_eq!(explicit.status, "ok", "{explicit:?}");
    assert_eq!(explicit.result.unwrap()["tab_id"], first);
    for (tab, count) in [(&first, 1), (&second, 0)] {
        let state = call(
            "web.evaluate",
            json!({"session_id":session,"tab_id":tab,"source":"window.count"}),
        );
        assert_eq!(state.status, "ok", "{state:?}");
        assert_eq!(
            state.result.unwrap()["value"].as_f64(),
            Some(f64::from(count))
        );
    }
    let destination = format!("{fixture}?next=1");
    let navigation = call(
        "web.workflow",
        json!({"version":1,"session_id":session,"tab_id":first,"steps":[
            {"action":{"operation":"goto","url":destination},"expect":{"condition":{"url":destination},"timeout_ms":2000}}
        ]}),
    );
    assert_eq!(navigation.status, "ok", "{navigation:?}");
    assert_eq!(
        navigation.result.unwrap()["page_state"]["snapshot"]["url"],
        destination
    );

    let mut bounded = Request::new(
        "run_workflow_scope",
        "web.workflow",
        json!({"version":1,"session_id":session,"tab_id":first,"steps":[
            {"expect":{"condition":{"query":"css=#never"},"timeout_ms":10000}}, increment
        ]}),
    );
    bounded.deadline_ms = 400;
    let started = Instant::now();
    let timeout =
        unix_request(&socket, &bounded, Duration::from_secs(5)).expect("bounded workflow response");
    assert_eq!(timeout.status, "error", "{timeout:?}");
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "workflow granted a fresh wait budget"
    );
    let state = call(
        "web.evaluate",
        json!({"session_id":session,"tab_id":first,"source":"window.count"}),
    );
    assert_eq!(state.status, "ok", "{state:?}");
    assert_eq!(state.result.unwrap()["value"].as_f64(), Some(0.0));
}
