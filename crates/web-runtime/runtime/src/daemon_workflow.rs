//! One external request, declarative preflight, sequential native operations.
use super::*;
use greppy_web_client::workflow::Workflow;

fn step_returned(response: &Response) -> bool {
    response.status == "ok"
        && response.error.is_none()
        && !response
            .result
            .as_ref()
            .is_some_and(|value| value.get("ok") == Some(&json!(false)))
}

impl Daemon {
    pub(super) fn web_workflow(&mut self, request: &Request) -> Response {
        let workflow: Workflow = match serde_json::from_value(request.payload.clone()) {
            Ok(workflow) => workflow,
            Err(error) => return protocol_error(request, &format!("invalid workflow: {error}")),
        };
        if let Err(error) = workflow.validate() {
            return protocol_error(request, &format!("invalid workflow: {error}"));
        }
        if request
            .session_id
            .as_ref()
            .is_some_and(|session| session != &workflow.session_id)
        {
            return protocol_error(
                request,
                "workflow session_id conflicts with request session_id",
            );
        }
        let Some(deadline) = Instant::now().checked_add(Duration::from_millis(request.deadline_ms))
        else {
            return protocol_error(request, "workflow deadline exceeds monotonic clock range");
        };
        let deadline = self
            .workflow_deadline
            .map_or(deadline, |active| active.min(deadline));
        let previous_deadline = self.workflow_deadline.replace(deadline);
        let previous_defer = self.workflow_defer_observation;
        let response = self.execute_workflow(request, workflow, deadline);
        self.workflow_deadline = previous_deadline;
        self.workflow_defer_observation = previous_defer;
        response
    }

    fn execute_workflow(
        &mut self,
        request: &Request,
        workflow: Workflow,
        deadline: Instant,
    ) -> Response {
        let mut scope = request.clone();
        // Null tab is an omitted target, not an invalid explicit tab.
        if workflow.tab_id.is_none() {
            scope.payload.as_object_mut().unwrap().remove("tab_id");
        }
        let (session, page) =
            match self.with_session_page_until(&scope, "web.workflow", Some(deadline)) {
                Ok(scope) => scope,
                Err(response) => return response,
            };
        let preflight = self.engine_call_timed_with_recovery(
            "page.evaluate",
            json!({
                "page": page, "source": workflow.preflight_source(),
            }),
            deadline.saturating_duration_since(Instant::now()),
            false,
        );
        self.finish_session(&session);
        let preflight = match preflight {
            Ok(value) => Self::plain_value(value.get("serialized").unwrap_or(&json!(null))),
            Err(error) => {
                let mut response = engine_error(request, error, 34);
                response.result = Some(
                    json!({"workflow_version":1,"phase":"preflight","completed_steps":0,"actions_attempted":0,"session_id":session,"tab_id":page}),
                );
                return response;
            }
        };
        if preflight.get("valid").and_then(|value| value.as_bool()) != Some(true) {
            let mut response = Response::error(
                request,
                workflow.preflight_error(&preflight, &request.request_id),
            );
            response.result = Some(
                json!({"workflow_version":1,"phase":"preflight","completed_steps":0,"actions_attempted":0,"preflight":preflight,"session_id":session,"tab_id":page}),
            );
            return response;
        }
        let mut receipts = Vec::new();
        let mut completed = 0;
        let mut attempted = 0;
        let mut final_state = None;
        let mut failure = None;
        let mut artifacts = Vec::new();
        for (index, step) in workflow.steps.iter().enumerate() {
            let mut receipt = json!({"step":index + 1});
            let mut phase = "action";
            let mut response = None;
            if let Some(action) = &step.action {
                self.workflow_defer_observation =
                    index + 1 < workflow.steps.len() || step.expect.is_some();
                let mut child = request.clone();
                child.request_id = greppy_web_client::new_request_id();
                child.operation = action.operation().into();
                child.payload = action.payload();
                child.payload["session_id"] = json!(session);
                child.payload["tab_id"] = json!(page);
                child.session_id = Some(session.clone());
                child.deadline_ms = deadline
                    .saturating_duration_since(Instant::now())
                    .as_millis()
                    .min(u64::MAX as u128) as u64;
                let result = if child.deadline_ms == 0 {
                    Response::error(&child, ErrorObject::new("TIMEOUT", "workflow deadline exhausted before this action", child.request_id.clone(), 34, "inspect completed_steps; already executed actions were not rolled back"))
                } else {
                    attempted += 1;
                    self.dispatch_operation(child)
                };
                let mut detail = result.result.clone().unwrap_or(json!({}));
                if let Some(object) = detail.as_object_mut() {
                    final_state = object.remove("page_state").filter(|state| {
                        state.get("status").and_then(|value| value.as_str()) != Some("deferred")
                    });
                }
                receipt["action"] =
                    json!({"operation":action.operation(),"status":result.status,"receipt":detail});
                artifacts.extend(result.artifacts.clone());
                response = Some(result);
            }
            let action_ok = response.as_ref().is_none_or(step_returned);
            if action_ok {
                if let Some(expect) = &step.expect {
                    phase = "expectation";
                    self.workflow_defer_observation = index + 1 < workflow.steps.len();
                    let mut child = request.clone();
                    child.request_id = greppy_web_client::new_request_id();
                    child.operation = "web.wait".into();
                    child.session_id = Some(session.clone());
                    child.deadline_ms = deadline
                        .saturating_duration_since(Instant::now())
                        .as_millis()
                        .min(u64::MAX as u128) as u64;
                    child.payload = json!({"session_id":session,"tab_id":page,"source":expect.condition.source(),"timeout_ms":expect.timeout_ms});
                    if let Some(reference) =
                        expect.condition.reference().expect("validated condition")
                    {
                        child.payload["condition_ref"] = json!({"type":"ref","value":reference});
                    }
                    let result = self.web_wait(&child);
                    let mut detail = result.result.clone().unwrap_or(json!({}));
                    if let Some(object) = detail.as_object_mut() {
                        final_state = object.remove("page_state").filter(|state| {
                            state.get("status").and_then(|value| value.as_str()) != Some("deferred")
                        });
                    }
                    receipt["expectation"] = json!({"status":result.status,"result":detail});
                    artifacts.extend(result.artifacts.clone());
                    response = Some(result);
                }
            }
            let result = response.expect("validated step contains an action or expectation");
            if !step_returned(&result) {
                receipt["failed_phase"] = json!(phase);
                failure = Some((
                    index + 1,
                    phase,
                    result.error.unwrap_or_else(|| {
                        ErrorObject::new(
                            "INVALID_WORKFLOW_STEP_RESULT",
                            "native step did not confirm a valid response",
                            request.request_id.clone(),
                            34,
                            "inspect step receipt; do not repeat an already executed mutation",
                        )
                    }),
                ));
                receipts.push(receipt);
                break;
            }
            // A missing held=true can never turn a wait into a workflow success.
            if step.expect.is_some()
                && result
                    .result
                    .as_ref()
                    .and_then(|value| value.get("held"))
                    .and_then(|value| value.as_bool())
                    != Some(true)
            {
                failure = Some((
                    index + 1,
                    "expectation",
                    ErrorObject::new(
                        "INVALID_WAIT_RESULT",
                        "expectation did not return held=true",
                        request.request_id.clone(),
                        34,
                        "inspect action receipt; no outcome was confirmed",
                    ),
                ));
                receipts.push(receipt);
                break;
            }
            completed += 1;
            receipts.push(receipt);
        }
        self.workflow_defer_observation = false;
        let state = final_state.unwrap_or_else(|| {
            page_state_envelope(
                self.observe_page_bounded(
                    &session,
                    &page,
                    deadline
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_secs(2)),
                    false,
                ),
            )
        });
        let mut result = json!({
            "workflow_version":1, "session_id":session, "tab_id":page,
            "completed_steps":completed, "total_steps":workflow.steps.len(), "actions_attempted":attempted,
            "steps":receipts, "page_state":state, "rolled_back":false,
            "untrusted_content_boundary":"UNTRUSTED_PAGE_CONTENT",
        });
        let mut response = if let Some((step, phase, error)) = failure {
            result["failed_step"] = json!(step);
            result["phase"] = json!(phase);
            let mut response = Response::error(request, error);
            response.result = Some(result);
            response
        } else {
            Response::ok(request, result)
        };
        response.artifacts = artifacts;
        response
    }
}
