//! Compile supported commands into one declarative runtime request.
use super::act::{ActCommand, TargetOpts};
use super::common::*;
use super::expect::ExpectCommand;
use super::nav::NavCommand;
use super::WebCommand;
use greppy_core::error::Result;
use greppy_web_client::workflow::{
    Workflow, WorkflowAction as Action, WorkflowCondition as Condition,
    WorkflowExpectation as Expectation, WorkflowSelector as Selector, WorkflowStep as Step,
    WORKFLOW_VERSION,
};

struct Compiled {
    step: Step,
    session: Option<String>,
    tab: Option<String>,
    open: bool,
}

fn rejected(json_out: bool, message: &str) -> i32 {
    emit_error(json_out, invalid(message)).unwrap_or(EXIT_WEB_INVALID)
}

fn selector(target: &str, opts: &TargetOpts, json_out: bool) -> std::result::Result<Selector, i32> {
    let parsed = parse_target(target, opts.first, opts.last, opts.nth)
        .map_err(|error| emit_error(json_out, error).unwrap_or(EXIT_WEB_INVALID))?;
    serde_json::from_value(parsed.selector).map_err(|_| {
        rejected(
            json_out,
            "target is not supported by native workflow version 1",
        )
    })
}

fn compile_action(command: ActCommand, json_out: bool) -> std::result::Result<Compiled, i32> {
    let (action, opts) = match command {
        ActCommand::Click { target, opts } => (
            Action::Click {
                selector: selector(&target, &opts, json_out)?,
            },
            opts,
        ),
        ActCommand::Fill {
            target,
            value,
            from_env,
            value_stdin,
            opts,
        } => {
            let target = selector(&target, &opts, json_out)?;
            let value = super::act::fill_value(value, from_env, value_stdin, json_out)?;
            (
                Action::Fill {
                    selector: target,
                    value,
                },
                opts,
            )
        }
        ActCommand::Clear { target, opts } => (
            Action::Fill {
                selector: selector(&target, &opts, json_out)?,
                value: String::new(),
            },
            opts,
        ),
        ActCommand::Type { target, text, opts } => (
            Action::Type {
                selector: selector(&target, &opts, json_out)?,
                text,
            },
            opts,
        ),
        ActCommand::Select {
            target,
            value,
            opts,
        } => (
            Action::Select {
                selector: selector(&target, &opts, json_out)?,
                value,
            },
            opts,
        ),
        ActCommand::Check { target, opts } => (
            Action::Check {
                selector: selector(&target, &opts, json_out)?,
            },
            opts,
        ),
        ActCommand::Uncheck { target, opts } => (
            Action::Uncheck {
                selector: selector(&target, &opts, json_out)?,
            },
            opts,
        ),
        ActCommand::Hover { target, opts } => (
            Action::Hover {
                selector: selector(&target, &opts, json_out)?,
            },
            opts,
        ),
        ActCommand::Press {
            target_or_key,
            key,
            opts,
        } => {
            let (target, key) = match key {
                Some(key) => (Some(selector(&target_or_key, &opts, json_out)?), key),
                None => (None, target_or_key),
            };
            (
                Action::Press {
                    selector: target,
                    key,
                },
                opts,
            )
        }
        ActCommand::Scroll { to, by, opts } => {
            let target = to
                .as_ref()
                .map(|target| selector(target, &opts, json_out))
                .transpose()?;
            (
                Action::Scroll {
                    selector: target,
                    delta_y: by,
                },
                opts,
            )
        }
        ActCommand::Upload {
            target,
            paths,
            opts,
        } => {
            let target = selector(&target, &opts, json_out)?;
            let files =
                super::act::stage_uploads(&paths).map_err(|error| rejected(json_out, &error))?;
            (
                Action::Upload {
                    selector: target,
                    files,
                },
                opts,
            )
        }
    };
    let expect = opts.expect.map(|query| Expectation {
        condition: Condition {
            query: Some(query),
            absent: opts.expect_absent,
            ..Condition::default()
        },
        timeout_ms: opts.expect_timeout,
    });
    Ok(Compiled {
        step: Step {
            action: Some(action),
            expect,
        },
        session: opts.session,
        tab: opts.tab,
        open: false,
    })
}

fn compile(command: WebCommand, json_out: bool) -> std::result::Result<Compiled, i32> {
    match command {
        WebCommand::Act(command) => compile_action(command, json_out),
        WebCommand::Nav(command) => {
            let (action, session, tab, open) = match command {
                NavCommand::Open { url, session, tab, .. } => (Action::Goto { url }, session, tab, true),
                NavCommand::Goto { url, session, tab, .. } => (Action::Goto { url }, session, tab, false),
                NavCommand::Back { session, tab, .. } => (Action::Back, session, tab, false),
                NavCommand::Forward { session, tab, .. } => (Action::Forward, session, tab, false),
                NavCommand::Reload { session, tab, .. } => (Action::Reload, session, tab, false),
            };
            Ok(Compiled { step: Step { action: Some(action), expect: None }, session, tab, open })
        }
        WebCommand::Expect(ExpectCommand::Wait { condition, timeout, .. }) => Ok(Compiled {
            step: Step { action: None, expect: Some(Expectation {
                condition: Condition { query: condition.query, url: condition.url, title: condition.title, absent: condition.absent },
                timeout_ms: timeout,
            }) }, session: condition.session, tab: condition.tab, open: false,
        }),
        _ => Err(rejected(json_out, "native workflow supports navigation, actions and wait; no steps executed. Use ordinary web do for other commands.")),
    }
}

fn merge_scope(
    target: &mut Option<String>,
    supplied: Option<String>,
    name: &str,
    json_out: bool,
) -> std::result::Result<(), i32> {
    if let Some(supplied) = supplied {
        if target.as_ref().is_some_and(|target| target != &supplied) {
            return Err(rejected(
                json_out,
                &format!("native workflow cannot switch {name} between steps; no steps executed"),
            ));
        }
        *target = Some(supplied);
    }
    Ok(())
}

pub(super) fn run(commands: Vec<WebCommand>, root: Option<&str>, json_out: bool) -> Result<i32> {
    let mut compiled = Vec::new();
    let mut session = None;
    let mut tab = None;
    let mut create = false;
    for command in commands {
        let value = match compile(command, json_out) {
            Ok(value) => value,
            Err(code) => return Ok(code),
        };
        if let Err(code) = merge_scope(&mut session, value.session, "session", json_out) {
            return Ok(code);
        }
        if let Err(code) = merge_scope(&mut tab, value.tab, "tab", json_out) {
            return Ok(code);
        }
        create |= value.open;
        compiled.push(value.step);
    }
    // Validate all declarative shapes before creating a session or sending any
    // operation. Runtime preflight additionally checks the engine syntax.
    let mut workflow = Workflow {
        version: WORKFLOW_VERSION,
        session_id: session.clone().unwrap_or_else(|| "pending-session".into()),
        tab_id: tab.clone(),
        steps: compiled,
    };
    if let Err(error) = workflow.validate() {
        return emit_error(json_out, invalid(&error));
    }
    let session = match super::nav::resolve_or_create_session(root, session, json_out, create) {
        Ok(session) => session,
        Err(code) => return Ok(code),
    };
    workflow.session_id = session.clone();
    workflow.tab_id = resolve_tab(root, tab);
    // Unsupported runtimes reject this operation before mutation. Never replay
    // a failed/partly completed workflow as individual CLI operations.
    match rpc_response(
        root,
        "web.workflow",
        serde_json::to_value(workflow).expect("validated workflow"),
        Some(session.clone()),
    ) {
        Ok(response) => {
            if let Some(result) = response.result.as_ref() {
                if result.get("session_id").and_then(|value| value.as_str()) == Some(&session) {
                    if let Some(tab) = result.get("tab_id").and_then(|value| value.as_str()) {
                        let _ = write_current_scope(root, &session, Some(tab));
                    }
                }
            }
            emit_response(json_out, response)
        }
        Err(error) => emit_error(json_out, error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: WebCommand,
    }

    #[test]
    fn action_expectation_is_declarative_and_scope_bound() {
        let cli = Cli::try_parse_from([
            "test",
            "click",
            "@7",
            "--expect",
            "css=#done",
            "--expect-absent",
            "--expect-timeout",
            "2500",
            "--session",
            "s",
            "--tab",
            "p",
        ])
        .unwrap();
        let result = compile(cli.command, true).unwrap();
        assert_eq!(result.session.as_deref(), Some("s"));
        assert_eq!(result.tab.as_deref(), Some("p"));
        assert_eq!(
            result.step.action,
            Some(Action::Click {
                selector: Selector::Ref { value: 7 }
            })
        );
        let expect = result.step.expect.unwrap();
        assert!(expect.condition.absent);
        assert_eq!(expect.condition.query.as_deref(), Some("css=#done"));
        assert_eq!(expect.timeout_ms, 2500);
    }

    #[test]
    fn missing_expectation_cannot_silently_ignore_absence_or_timeout() {
        for flag in [vec!["--expect-absent"], vec!["--expect-timeout", "100"]] {
            let mut args = vec!["test", "click", "@1"];
            args.extend(flag);
            assert!(Cli::try_parse_from(args).is_err());
        }
        assert!(Cli::try_parse_from(["test", "click", "@1"]).is_ok());
    }
}
