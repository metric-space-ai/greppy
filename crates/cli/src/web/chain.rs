//! Composition verbs. `script` manages reusable Playwright programs entirely
//! on the client side and needs no runtime operation of its own; it hands the
//! finished source to the existing `web.run`. `do` follows once the action
//! verbs and the session scope are in place.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Saved scripts live beside the other per-workspace web state.
const STORE: &str = ".greppy/web/scripts";

#[derive(Debug, Subcommand)]
pub enum ChainCommand {
    /// Manage reusable Playwright scripts.
    Script {
        #[command(subcommand)]
        command: ScriptCommand,
    },
    /// Run several web commands in one process, separated by `::`.
    ///
    ///   greppy web do --json open URL :: click css=#btn :: observe
    ///
    /// Own flags come BEFORE the first step: everything after it belongs to
    /// the steps, so a trailing `--json` would be read as an argument of the
    /// last command.
    ///
    /// One session, one runtime, no teardown between steps. Stops at the
    /// first failing step so a chain cannot walk on through an unknown page
    /// state.
    Do {
        /// Steps, separated by a literal `::` argument.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        steps: Vec<String>,
        /// Keep going after a failing step instead of stopping.
        #[arg(long)]
        continue_on_error: bool,
        /// Print the parsed steps and exit without running them.
        #[arg(long)]
        explain: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum ScriptCommand {
    /// Store a script under NAME so it can be re-run without rewriting it.
    Save {
        name: String,
        /// Source file to store.
        #[arg(long)]
        file: String,
        /// Replace an existing script of the same name.
        #[arg(long)]
        force: bool,
        #[arg(long)]
        json: bool,
    },
    /// List stored scripts.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Print a stored script.
    Show {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Delete a stored script.
    Rm {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Run a stored script, substituting `{{key}}` placeholders.
    Run {
        name: String,
        /// Placeholder binding, repeatable: `--arg url=https://example.com`.
        #[arg(long = "arg", value_name = "KEY=VALUE")]
        args: Vec<String>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        timeout: Option<u64>,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch(command: ChainCommand, root: Option<&str>) -> Result<i32> {
    let command = match command {
        ChainCommand::Do {
            steps,
            continue_on_error,
            explain,
            json,
        } => return run_chain(root, &steps, continue_on_error, explain, json),
        ChainCommand::Script { command } => command,
    };
    match command {
        ScriptCommand::Save {
            name,
            file,
            force,
            json,
        } => save(&name, &file, force, json),
        ScriptCommand::List { json } => list(json),
        ScriptCommand::Show { name, json } => show(&name, json),
        ScriptCommand::Rm { name, json } => remove(&name, json),
        ScriptCommand::Run {
            name,
            args,
            session,
            timeout,
            json,
        } => run_saved(root, &name, &args, session, timeout, json),
    }
}

/// Script names become file names, so they must not be able to escape the
/// store. Only a conservative character set is accepted.
fn validate_name(name: &str) -> std::result::Result<(), String> {
    if name.is_empty() || name.len() > 64 {
        return Err("name must be 1 to 64 characters".into());
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.')
    {
        return Err("name may contain only letters, digits, '-', '_' and '.'".into());
    }
    if name.starts_with('.') || name.contains("..") {
        return Err("name must not start with '.' or contain '..'".into());
    }
    Ok(())
}

fn store_dir() -> PathBuf {
    PathBuf::from(STORE)
}

fn script_path(name: &str) -> PathBuf {
    store_dir().join(format!("{name}.mjs"))
}

fn ok(json_out: bool, operation: &str, result: serde_json::Value) -> Result<i32> {
    emit_web(
        json_out,
        &json!({
            "schema": "greppy.web-runtime.v1",
            "status": "ok",
            "operation": operation,
            "result": result,
        }),
    )?;
    Ok(0)
}

fn save(name: &str, file: &str, force: bool, json_out: bool) -> Result<i32> {
    if let Err(message) = validate_name(name) {
        return emit_error(json_out, invalid(&format!("web script save: {message}")));
    }
    let source = match std::fs::read_to_string(Path::new(file)) {
        Ok(source) => source,
        Err(error) => {
            return emit_error(
                json_out,
                invalid(&format!("web script save: cannot read {file}: {error}")),
            )
        }
    };
    let target = script_path(name);
    if target.exists() && !force {
        return emit_error(
            json_out,
            invalid(&format!(
                "web script save: `{name}` already exists; pass --force to replace it"
            )),
        );
    }
    if let Err(error) = std::fs::create_dir_all(store_dir()) {
        return emit_error(
            json_out,
            invalid(&format!("web script save: cannot create {STORE}: {error}")),
        );
    }
    if let Err(error) = std::fs::write(&target, source.as_bytes()) {
        return emit_error(
            json_out,
            invalid(&format!("web script save: cannot write script: {error}")),
        );
    }
    ok(
        json_out,
        "web.script.save",
        json!({
            "name": name,
            "path": target.display().to_string(),
            "bytes": source.len(),
            "placeholders": placeholders(&source),
        }),
    )
}

fn list(json_out: bool) -> Result<i32> {
    let mut names: Vec<serde_json::Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(store_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("mjs") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let source = std::fs::read_to_string(&path).unwrap_or_default();
            names.push(json!({
                "name": stem,
                "bytes": source.len(),
                "placeholders": placeholders(&source),
            }));
        }
    }
    names.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    ok(json_out, "web.script.list", json!({ "scripts": names }))
}

fn show(name: &str, json_out: bool) -> Result<i32> {
    if let Err(message) = validate_name(name) {
        return emit_error(json_out, invalid(&format!("web script show: {message}")));
    }
    match std::fs::read_to_string(script_path(name)) {
        Ok(source) => ok(
            json_out,
            "web.script.show",
            json!({ "name": name, "source": source }),
        ),
        Err(_) => emit_error(
            json_out,
            invalid(&format!("web script show: no script named `{name}`")),
        ),
    }
}

fn remove(name: &str, json_out: bool) -> Result<i32> {
    if let Err(message) = validate_name(name) {
        return emit_error(json_out, invalid(&format!("web script rm: {message}")));
    }
    match std::fs::remove_file(script_path(name)) {
        Ok(()) => ok(json_out, "web.script.rm", json!({ "name": name })),
        Err(_) => emit_error(
            json_out,
            invalid(&format!("web script rm: no script named `{name}`")),
        ),
    }
}

/// Every `{{key}}` occurrence in a script, deduplicated and in first-seen order.
fn placeholders(source: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == b'{' && bytes[i + 1] == b'{' {
            if let Some(end) = source[i + 2..].find("}}") {
                let key = source[i + 2..i + 2 + end].trim().to_owned();
                if !key.is_empty()
                    && key
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
                    && !found.contains(&key)
                {
                    found.push(key);
                }
                i += end + 4;
                continue;
            }
        }
        i += 1;
    }
    found
}

fn substitute(source: &str, bindings: &[(String, String)]) -> String {
    let mut out = source.to_owned();
    for (key, value) in bindings {
        out = out.replace(&format!("{{{{{key}}}}}"), value);
    }
    out
}

fn run_saved(
    root: Option<&str>,
    name: &str,
    args: &[String],
    session: Option<String>,
    timeout: Option<u64>,
    json_out: bool,
) -> Result<i32> {
    if let Err(message) = validate_name(name) {
        return emit_error(json_out, invalid(&format!("web script run: {message}")));
    }
    let Ok(source) = std::fs::read_to_string(script_path(name)) else {
        return emit_error(
            json_out,
            invalid(&format!("web script run: no script named `{name}`")),
        );
    };
    let mut bindings = Vec::new();
    for arg in args {
        let Some((key, value)) = arg.split_once('=') else {
            return emit_error(
                json_out,
                invalid(&format!("web script run: --arg `{arg}` is not KEY=VALUE")),
            );
        };
        bindings.push((key.to_owned(), value.to_owned()));
    }
    // An unbound placeholder is a mistake, not an empty string: a script that
    // silently navigates to "" would look like a runtime defect.
    let bound: Vec<&String> = bindings.iter().map(|(key, _)| key).collect();
    let missing: Vec<String> = placeholders(&source)
        .into_iter()
        .filter(|key| !bound.contains(&key))
        .collect();
    if !missing.is_empty() {
        return emit_error(
            json_out,
            invalid(&format!(
                "web script run: unbound placeholder(s) {}; pass --arg KEY=VALUE",
                missing.join(", ")
            )),
        );
    }
    let session = match resolve_session(root, session) {
        Ok(session) => session,
        Err(error) => return emit_error(json_out, error),
    };
    let mut payload = json!({
        "session_id": session,
        "script_source": "stdin",
        "script_text": substitute(&source, &bindings),
    });
    if let Some(timeout) = timeout {
        payload["timeout_seconds"] = json!(timeout);
    }
    rpc(root, json_out, "web.run", payload, Some(session))
}

/// Split the trailing argument list on the literal `::` separator.
fn split_steps(steps: &[String]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    for token in steps {
        if token == "::" {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(token.clone());
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn compact_chain_enabled() -> bool {
    std::env::var("GREPPY_WEB_CHAIN_VIEW").as_deref() == Ok("compact")
}

fn emit_chain_summary(
    json_out: bool,
    total: usize,
    ran: usize,
    failed: usize,
    stopped_at: Option<usize>,
    argv: Option<&Vec<String>>,
) -> Result<()> {
    let mut result = json!({"steps_total": total, "steps_ran": ran, "steps_failed": failed});
    if let Some(step) = stopped_at {
        result["stopped_at"] = json!(step);
        result["argv"] = json!(argv);
    }
    if json_out || !compact_chain_enabled() {
        emit_web(
            json_out,
            &json!({
                "schema": "greppy.web-runtime.v1",
                "status": if failed == 0 { "ok" } else { "error" },
                "operation": "web.do", "result": result,
            }),
        )
    } else {
        println!(
            "chain: {ran}/{total} steps executed, {failed} failed{}",
            stopped_at
                .map(|step| format!("; stopped at {step}; no rollback attempted"))
                .unwrap_or_default()
        );
        Ok(())
    }
}

fn run_chain(
    root: Option<&str>,
    steps: &[String],
    continue_on_error: bool,
    explain: bool,
    json_out: bool,
) -> Result<i32> {
    use clap::Parser;

    let parsed = split_steps(steps);
    if parsed.is_empty() {
        return emit_error(
            json_out,
            invalid("web do requires at least one step, e.g. `web do open URL :: observe`"),
        );
    }

    // Parse every step before running any of them: a typo in step four must
    // not surface after step three has already changed the page.
    #[derive(Parser)]
    #[command(name = "greppy web", no_binary_name = true)]
    struct StepParser {
        #[command(subcommand)]
        command: super::WebCommand,
    }

    let mut commands = Vec::new();
    for (index, step) in parsed.iter().enumerate() {
        match StepParser::try_parse_from(step) {
            Ok(parsed) => commands.push(parsed.command),
            Err(error) => {
                let first = error.to_string();
                let first = first.lines().next().unwrap_or("unparsable step");
                let mut error = invalid(&format!(
                    "web do: step {} (`{}`) is not a valid command: {first}",
                    index + 1,
                    step.join(" ")
                ));
                if step
                    .first()
                    .is_some_and(|arg| arg == "--session" || arg.starts_with("--session="))
                {
                    let hint = "place --session SID after each step's command, not before it: `greppy web do --explain click @1 --session SID :: observe --session SID`";
                    error.message = format!("{}; {hint}", error.message).into();
                    error.next_action = hint.into();
                }
                return emit_error(json_out, error);
            }
        }
    }

    if explain {
        let plan: Vec<serde_json::Value> = parsed
            .iter()
            .enumerate()
            .map(|(index, step)| json!({ "step": index + 1, "argv": step }))
            .collect();
        return ok(json_out, "web.do.explain", json!({ "steps": plan }));
    }

    let mut ran = 0usize;
    let mut failed = 0usize;
    let mut last_code = 0i32;
    // Mixed machine output is deliberately left untouched, including when a
    // step requests JSON explicitly. A literal --json argument may conservatively
    // disable aggregation; it must never accidentally enable it.
    let _machine_mode = super::chain_output::machine_mode(json_out);
    let mut output = super::chain_output::start(
        !json_out
            && compact_chain_enabled()
            && super::view::enabled()
            && !parsed
                .iter()
                .flatten()
                .any(|arg| arg == "--json" || arg == "--jsonl"),
        super::view::cache_dir(),
    );
    for (index, command) in commands.into_iter().enumerate() {
        if let Some(output) = &output {
            output.step(
                index + 1,
                parsed[index].first().map(String::as_str).unwrap_or("step"),
            );
        }
        // `dispatch_inner`, not `dispatch`: the outer guard already owns the
        // runtime lifetime for the whole chain.
        let code = super::dispatch_inner(command, root)?;
        ran += 1;
        // Keep each payload attributable without repeating the protocol and
        // complete command (including filled text) in the human-readable view.
        // Machine consumers still receive the complete typed step record.
        if json_out || !compact_chain_enabled() {
            emit_web(
                json_out,
                &json!({
                    "schema": "greppy.web-runtime.v1",
                    "kind": "step",
                    "operation": "web.do.step",
                    "status": if code == 0 { "ok" } else { "error" },
                    "step": index + 1,
                    "steps_total": parsed.len(),
                    "argv": parsed[index],
                    "exit_code": code,
                }),
            )?;
        } else if code != 0 || !output.as_ref().is_some_and(|output| output.deferred()) {
            let verb = parsed[index].first().map(String::as_str).unwrap_or("step");
            if code == 0 {
                println!("step {}/{} {verb}: ok", index + 1, parsed.len());
            } else {
                println!(
                    "step {}/{} {verb}: FAILED (exit {code})",
                    index + 1,
                    parsed.len()
                );
            }
        }
        if code != 0 {
            failed += 1;
            last_code = code;
            if !continue_on_error {
                if let Some(output) = output.take() {
                    output.finish()?;
                }
                emit_chain_summary(
                    json_out,
                    parsed.len(),
                    ran,
                    failed,
                    Some(index + 1),
                    Some(&parsed[index]),
                )?;
                return Ok(code);
            }
        }
    }
    if let Some(output) = output.take() {
        output.finish()?;
    }
    emit_chain_summary(json_out, parsed.len(), ran, failed, None, None)?;
    Ok(if failed == 0 { 0 } else { last_code })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_split_on_the_double_colon() {
        let argv: Vec<String> = ["open", "URL", "::", "click", "css=#b", "::", "observe"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let steps = split_steps(&argv);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], vec!["open", "URL"]);
        assert_eq!(steps[2], vec!["observe"]);
    }

    #[test]
    fn empty_segments_do_not_become_steps() {
        let argv: Vec<String> = ["::", "observe", "::", "::"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(split_steps(&argv), vec![vec!["observe".to_string()]]);
    }

    #[test]
    fn no_steps_at_all_yields_nothing() {
        assert!(split_steps(&[]).is_empty());
    }

    #[test]
    fn names_may_not_escape_the_store() {
        assert!(validate_name("login").is_ok());
        assert!(validate_name("login-flow_2.v1").is_ok());
        assert!(validate_name("../etc/passwd").is_err());
        assert!(validate_name("..").is_err());
        assert!(validate_name(".hidden").is_err());
        assert!(validate_name("has/slash").is_err());
        assert!(validate_name("").is_err());
    }

    #[test]
    fn placeholders_are_found_once_and_in_order() {
        let source = "await page.goto('{{url}}'); await page.fill('#u', '{{user}}'); // {{url}}";
        assert_eq!(placeholders(source), vec!["url", "user"]);
    }

    #[test]
    fn malformed_placeholders_are_ignored() {
        assert!(placeholders("{{ not closed").is_empty());
        assert!(placeholders("{{has space}}").is_empty());
        assert!(placeholders("{{}}").is_empty());
    }

    #[test]
    fn substitution_replaces_every_occurrence() {
        let source = "a {{x}} b {{x}} c {{y}}";
        let out = substitute(
            source,
            &[("x".into(), "1".into()), ("y".into(), "2".into())],
        );
        assert_eq!(out, "a 1 b 1 c 2");
    }

    #[test]
    fn substitution_leaves_unknown_placeholders_alone() {
        // run_saved rejects these before substitution; the helper itself must
        // not invent a value.
        let out = substitute("go {{url}}", &[("other".into(), "x".into())]);
        assert_eq!(out, "go {{url}}");
    }
}
