//! Experimental `greppy web` client. Does not link V8 or Servo.

use super::*;
use serde_json::json;
use std::path::PathBuf;
use std::process::Command;

pub const EXIT_WEB_UNAVAILABLE: i32 = 31;
pub const EXIT_WEB_SCRIPT: i32 = 33;
pub const EXIT_WEB_WORKER: i32 = 38;

pub fn dispatch(command: WebCommand) -> Result<i32> {
    match command {
        WebCommand::Status { json } => status(json),
        WebCommand::Doctor { json } => doctor(json),
        WebCommand::Run {
            script_file,
            script_stdin,
            json,
        } => run(script_file, script_stdin, json),
    }
}

fn status(json: bool) -> Result<i32> {
    let supervisor = find_binary("web-runtime-supervisor");
    let available = supervisor.as_ref().is_some_and(|path| path.exists());
    let payload = json!({
        "schema": "greppy.web-runtime.v1",
        "status": if available { "experimental" } else { "runtime_unavailable" },
        "compatibility_coverage_level": "unverified",
        "playwright_compatibility_version": "1.62.1",
        "label": "experimental web-runtime spike",
        "supervisor": supervisor.as_ref().map(|path| path.display().to_string()),
    });
    emit_web(json, &payload, available)?;
    Ok(if available { 0 } else { EXIT_WEB_UNAVAILABLE })
}

fn doctor(json: bool) -> Result<i32> {
    let supervisor = find_binary("web-runtime-supervisor");
    let controller = find_binary("web-controller-worker");
    let content = find_binary("web-content-worker");
    let available = [&supervisor, &controller, &content]
        .into_iter()
        .all(|path| path.as_ref().is_some_and(|path| path.exists()));
    let payload = json!({
        "schema": "greppy.web-runtime.v1",
        "status": if available { "experimental" } else { "runtime_unavailable" },
        "supervisor": supervisor.as_ref().map(|path| path.display().to_string()),
        "controller_worker": controller.as_ref().map(|path| path.display().to_string()),
        "content_worker": content.as_ref().map(|path| path.display().to_string()),
        "engines_linked_into_greppy_parent": false,
        "label": "experimental web-runtime spike",
    });
    emit_web(json, &payload, available)?;
    Ok(if available { 0 } else { EXIT_WEB_UNAVAILABLE })
}

fn run(script_file: Option<String>, script_stdin: bool, json: bool) -> Result<i32> {
    if script_stdin {
        let payload = json!({
            "schema": "greppy.web-runtime.v1",
            "status": "error",
            "error": {
                "code": "runtime_unavailable",
                "message": "web run --script-stdin requires the session daemon, which is not in this spike",
                "retryable": false,
                "next_action": "use --script-file for the experimental one-shot supervisor",
                "exit_code": EXIT_WEB_UNAVAILABLE,
            }
        });
        emit_web(json, &payload, false)?;
        return Ok(EXIT_WEB_UNAVAILABLE);
    }
    let Some(script_file) = script_file else {
        return Err(Error::Invalid(
            "web run requires --script-file FILE or --script-stdin".into(),
        ));
    };
    let Some(supervisor) = find_binary("web-runtime-supervisor") else {
        return unavailable(json, "web-runtime-supervisor is not installed");
    };
    let Some(controller) = find_binary("web-controller-worker") else {
        return unavailable(json, "web-controller-worker is not installed");
    };
    let Some(content) = find_binary("web-content-worker") else {
        return unavailable(json, "web-content-worker is not installed");
    };
    let output = Command::new(supervisor)
        .arg("--controller-worker")
        .arg(controller)
        .arg("--content-worker")
        .arg(content)
        .arg("--script")
        .arg(&script_file)
        .output()
        .map_err(|error| Error::Io {
            context: "web-runtime-supervisor".into(),
            source: error,
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let ok = output.status.success();
    let payload = json!({
        "schema": "greppy.web-runtime.v1",
        "status": if ok { "ok" } else { "error" },
        "stdout": stdout,
        "stderr": stderr,
    });
    emit_web(json, &payload, ok)?;
    Ok(if ok {
        0
    } else if output.status.code() == Some(1) {
        EXIT_WEB_SCRIPT
    } else {
        EXIT_WEB_WORKER
    })
}

fn unavailable(json: bool, message: &str) -> Result<i32> {
    let payload = json!({
        "schema": "greppy.web-runtime.v1",
        "status": "error",
        "error": {
            "code": "runtime_unavailable",
            "message": message,
            "retryable": false,
            "next_action": "build crates/web-runtime with controller-runtime and content-runtime features",
            "exit_code": EXIT_WEB_UNAVAILABLE,
        }
    });
    emit_web(json, &payload, false)?;
    Ok(EXIT_WEB_UNAVAILABLE)
}

fn emit_web(json: bool, payload: &serde_json::Value, _ok: bool) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string(payload)
                .map_err(|error| { Error::Invalid(format!("web json encode failed: {error}")) })?
        );
    } else {
        println!("{}", payload);
    }
    Ok(())
}

fn find_binary(name: &str) -> Option<PathBuf> {
    let env_name = format!("GREPPY_{}", name.to_uppercase().replace('-', "_"));
    if let Ok(path) = std::env::var(&env_name) {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_supervisor_is_runtime_unavailable() {
        let _guard = crate::TEST_ENV_LOCK.lock().unwrap();
        std::env::remove_var("GREPPY_WEB_RUNTIME_SUPERVISOR");
        assert!(find_binary("web-runtime-supervisor-missing-name").is_none());
    }
}
