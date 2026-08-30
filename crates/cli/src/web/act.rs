//! Action verbs: click fill type clear select check uncheck press hover scroll upload.

use super::common::*;
use clap::{Args, Subcommand};
use greppy_core::error::Result;
use serde_json::json;
use std::io::{self, Read};

#[derive(Debug, Args)]
pub struct TargetOpts {
    #[arg(long)]
    pub session: Option<String>,
    #[arg(long)]
    pub first: bool,
    #[arg(long)]
    pub last: bool,
    #[arg(long)]
    pub nth: Option<i64>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Subcommand)]
pub enum ActCommand {
    /// Click TARGET.
    Click {
        target: String,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Set TARGET's value.
    Fill {
        target: String,
        value: Option<String>,
        #[arg(long = "from-env")]
        from_env: Option<String>,
        #[arg(long = "value-stdin")]
        value_stdin: bool,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Type TEXT into TARGET character by character.
    Type {
        target: String,
        text: String,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Clear TARGET.
    Clear {
        target: String,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Select VALUE on TARGET.
    Select {
        target: String,
        value: String,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Check TARGET.
    Check {
        target: String,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Uncheck TARGET.
    Uncheck {
        target: String,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Press KEY, optionally after focusing TARGET.
    Press {
        #[arg(value_name = "TARGET_OR_KEY")]
        target_or_key: String,
        #[arg(value_name = "KEY")]
        key: Option<String>,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Hover TARGET.
    Hover {
        target: String,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Scroll to TARGET or by N pixels. Exactly one of --to / --by.
    Scroll {
        #[arg(long = "to")]
        to: Option<String>,
        #[arg(long = "by")]
        by: Option<i64>,
        #[command(flatten)]
        opts: TargetOpts,
    },
    /// Set files on a file input TARGET.
    Upload {
        target: String,
        paths: Vec<String>,
        #[command(flatten)]
        opts: TargetOpts,
    },
}

pub(super) fn dispatch(command: ActCommand, root: Option<&str>) -> Result<i32> {
    match command {
        ActCommand::Click { target, opts } => locator_rpc(
            root,
            opts.json,
            opts.session,
            "web.click",
            &target,
            opts.first,
            opts.last,
            opts.nth,
            json!({}),
        ),
        ActCommand::Fill {
            target,
            value,
            from_env,
            value_stdin,
            opts,
        } => match fill_value(value, from_env, value_stdin, opts.json) {
            Err(code) => Ok(code),
            Ok(value) => locator_rpc(
                root,
                opts.json,
                opts.session,
                "web.fill",
                &target,
                opts.first,
                opts.last,
                opts.nth,
                json!({ "value": value }),
            ),
        },
        ActCommand::Type { target, text, opts } => locator_rpc(
            root,
            opts.json,
            opts.session,
            "web.type",
            &target,
            opts.first,
            opts.last,
            opts.nth,
            json!({ "text": text }),
        ),
        ActCommand::Clear { target, opts } => locator_rpc(
            root,
            opts.json,
            opts.session,
            "web.fill",
            &target,
            opts.first,
            opts.last,
            opts.nth,
            json!({ "value": "" }),
        ),
        ActCommand::Select {
            target,
            value,
            opts,
        } => locator_rpc(
            root,
            opts.json,
            opts.session,
            "web.select",
            &target,
            opts.first,
            opts.last,
            opts.nth,
            json!({ "value": value }),
        ),
        ActCommand::Check { target, opts } => locator_rpc(
            root,
            opts.json,
            opts.session,
            "web.check",
            &target,
            opts.first,
            opts.last,
            opts.nth,
            json!({}),
        ),
        ActCommand::Uncheck { target, opts } => locator_rpc(
            root,
            opts.json,
            opts.session,
            "web.uncheck",
            &target,
            opts.first,
            opts.last,
            opts.nth,
            json!({}),
        ),
        ActCommand::Press {
            target_or_key,
            key,
            opts,
        } => {
            let Some(session) = opts.session else {
                return emit_error(opts.json, invalid("web press requires --session SESSION"));
            };
            let (target, key) = match key {
                Some(key) => (Some(target_or_key), key),
                None => (None, target_or_key),
            };
            let mut payload = json!({ "session_id": session, "key": key });
            if let Some(target) = target {
                match parse_target(&target, opts.first, opts.last, opts.nth) {
                    Ok(parsed) => payload["selector"] = parsed.selector,
                    Err(error) => return emit_error(opts.json, error),
                }
            }
            rpc(root, opts.json, "web.press", payload, Some(session))
        }
        ActCommand::Hover { target, opts } => locator_rpc(
            root,
            opts.json,
            opts.session,
            "web.hover",
            &target,
            opts.first,
            opts.last,
            opts.nth,
            json!({}),
        ),
        ActCommand::Scroll { to, by, opts } => match (to, by) {
            (Some(target), None) => locator_rpc(
                root,
                opts.json,
                opts.session,
                "web.scroll",
                &target,
                opts.first,
                opts.last,
                opts.nth,
                json!({}),
            ),
            (None, Some(delta)) => {
                let Some(session) = opts.session else {
                    return emit_error(opts.json, invalid("web scroll requires --session SESSION"));
                };
                rpc(
                    root,
                    opts.json,
                    "web.scroll",
                    json!({ "session_id": session, "delta_y": delta }),
                    Some(session),
                )
            }
            _ => emit_error(
                opts.json,
                invalid("web scroll requires exactly one of --to or --by"),
            ),
        },
        ActCommand::Upload {
            target,
            paths,
            opts,
        } => {
            if paths.is_empty() {
                return emit_error(opts.json, invalid("web upload requires PATH..."));
            }
            locator_rpc(
                root,
                opts.json,
                opts.session,
                "web.upload",
                &target,
                opts.first,
                opts.last,
                opts.nth,
                json!({ "files": paths }),
            )
        }
    }
}

fn locator_rpc(
    root: Option<&str>,
    json_out: bool,
    session: Option<String>,
    operation: &str,
    target: &str,
    first: bool,
    last: bool,
    nth: Option<i64>,
    extra: serde_json::Value,
) -> Result<i32> {
    let Some(session) = session else {
        return emit_error(
            json_out,
            invalid(&format!("{operation} requires --session SESSION")),
        );
    };
    let parsed = match parse_target(target, first, last, nth) {
        Ok(parsed) => parsed,
        Err(error) => return emit_error(json_out, error),
    };
    let mut payload = json!({
        "session_id": session,
        "selector": parsed.selector,
    });
    if let Some(object) = extra.as_object() {
        if let Some(dst) = payload.as_object_mut() {
            for (key, value) in object {
                dst.insert(key.clone(), value.clone());
            }
        }
    }
    rpc(root, json_out, operation, payload, Some(session))
}

fn fill_value(
    value: Option<String>,
    from_env: Option<String>,
    value_stdin: bool,
    json: bool,
) -> std::result::Result<String, i32> {
    let sources = [value.is_some(), from_env.is_some(), value_stdin]
        .into_iter()
        .filter(|flag| *flag)
        .count();
    if sources != 1 {
        let _ = emit_error(
            json,
            invalid("web fill requires VALUE, --from-env NAME, or --value-stdin"),
        );
        return Err(EXIT_WEB_INVALID);
    }
    if let Some(value) = value {
        return Ok(value);
    }
    if let Some(name) = from_env {
        match std::env::var(&name) {
            Ok(value) => Ok(value),
            Err(_) => {
                let _ = emit_error(
                    json,
                    invalid(&format!("environment variable {name} is not set")),
                );
                Err(EXIT_WEB_INVALID)
            }
        }
    } else {
        let mut text = String::new();
        if let Err(error) = io::stdin().read_to_string(&mut text) {
            let _ = emit_error(
                json,
                invalid(&format!("failed to read --value-stdin: {error}")),
            );
            return Err(EXIT_WEB_INVALID);
        }
        Ok(text)
    }
}
