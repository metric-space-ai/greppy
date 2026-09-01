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
    pub tab: Option<String>,
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
            opts.tab,
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
                opts.tab,
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
            opts.tab,
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
            opts.tab,
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
            opts.tab,
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
            opts.tab,
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
            opts.tab,
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
            let session = match resolve_session(root, opts.session) {
                Ok(session) => session,
                Err(error) => return emit_error(opts.json, error),
            };
            let (target, key) = match key {
                Some(key) => (Some(target_or_key), key),
                None => (None, target_or_key),
            };
            let mut payload = json!({ "session_id": session, "key": key });
            if let Some(tab) = resolve_tab(root, opts.tab) {
                payload["tab_id"] = json!(tab);
            }
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
            opts.tab,
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
                opts.tab,
                "web.scroll",
                &target,
                opts.first,
                opts.last,
                opts.nth,
                json!({}),
            ),
            (None, Some(delta)) => {
                let session = match resolve_session(root, opts.session) {
                    Ok(session) => session,
                    Err(error) => return emit_error(opts.json, error),
                };
                let mut payload = json!({ "session_id": session, "delta_y": delta });
                if let Some(tab) = resolve_tab(root, opts.tab) {
                    payload["tab_id"] = json!(tab);
                }
                rpc(root, opts.json, "web.scroll", payload, Some(session))
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
            // The worker may only read its own temp directory, so a caller's
            // file is rejected wherever it actually lives. The CLI runs with
            // the caller's permissions and the worker does not, so staging
            // belongs here: copy the file into the allowed area and hand over
            // the copy. The sandbox rule stays exactly as strict.
            let paths = match stage_uploads(&paths) {
                Ok(staged) => staged,
                Err(message) => {
                    return emit_error(opts.json, invalid(&format!("web upload: {message}")))
                }
            };
            locator_rpc(
                root,
                opts.json,
                opts.session,
                opts.tab,
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
    tab: Option<String>,
    operation: &str,
    target: &str,
    first: bool,
    last: bool,
    nth: Option<i64>,
    extra: serde_json::Value,
) -> Result<i32> {
    let session = match resolve_session(root, session) {
        Ok(session) => session,
        Err(error) => return emit_error(json_out, error),
    };
    let parsed = match parse_target(target, first, last, nth) {
        Ok(parsed) => parsed,
        Err(error) => return emit_error(json_out, error),
    };
    let mut payload = json!({
        "session_id": session,
        "selector": parsed.selector,
    });
    if let Some(tab) = resolve_tab(root, tab) {
        payload["tab_id"] = json!(tab);
    }
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

/// Copy files into the directory the content worker is allowed to read.
///
/// Returns the staged paths. A file that is already inside the area is passed
/// through untouched, so repeated uploads do not pile up copies.
fn stage_uploads(paths: &[String]) -> std::result::Result<Vec<String>, String> {
    /// Refuse anything unreasonably large before it is copied twice.
    const MAX_BYTES: u64 = 64 * 1024 * 1024;
    let area = std::env::temp_dir().join("greppy-web-runtime").join("uploads");
    let mut staged = Vec::with_capacity(paths.len());
    for path in paths {
        let source = std::path::Path::new(path);
        let meta = source
            .metadata()
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        if !meta.is_file() {
            return Err(format!("{path} is not a regular file"));
        }
        if meta.len() > MAX_BYTES {
            return Err(format!(
                "{path} is {} bytes; the limit is {MAX_BYTES}",
                meta.len()
            ));
        }
        let canonical = source
            .canonicalize()
            .map_err(|error| format!("cannot resolve {path}: {error}"))?;
        if canonical.starts_with(&area) {
            staged.push(canonical.display().to_string());
            continue;
        }
        std::fs::create_dir_all(&area)
            .map_err(|error| format!("cannot create {}: {error}", area.display()))?;
        let name = canonical
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "upload".to_owned());
        // Keep the original name so the page sees what the caller meant, but
        // scope the directory per process so two uploads of different files
        // with the same name do not collide.
        let dir = area.join(format!("p{}", std::process::id()));
        std::fs::create_dir_all(&dir)
            .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
        let target = dir.join(&name);
        std::fs::copy(&canonical, &target)
            .map_err(|error| format!("cannot stage {path}: {error}"))?;
        staged.push(target.display().to_string());
    }
    Ok(staged)
}
