//! Desktop computer use for greppy.
//!
//! [`upstream`] is the only module that calls agent-desktop. Callers see
//! [`command::Command`] and the `greppy.desktop.v1` document.

mod command;
mod schema;
mod upstream;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub use command::{Command, FakeApp, FakeDesktop, FakeWindow, COMMANDS, ONE_PIXEL_PNG};
pub use schema::{Limits, Outcome, SCHEMA};

/// Built-in agent runs set `GREPPY_AGENT_RUN`. Desktop stays off until the
/// parent also sets `GREPPY_CAPABILITY_DESKTOP` (start flag or enable command).
pub fn builtin_agent_blocks_desktop() -> bool {
    std::env::var_os("GREPPY_AGENT_RUN").is_some()
        && std::env::var_os("GREPPY_CAPABILITY_DESKTOP").is_none()
}

pub fn run(command: &Command, limits: &Limits) -> Outcome {
    if builtin_agent_blocks_desktop() {
        return schema::agent_disabled(command, limits);
    }
    upstream::execute(command, None, limits)
}

pub fn run_fake(command: &Command, fake: &FakeDesktop, limits: &Limits) -> Outcome {
    let dir = std::env::temp_dir().join(format!(
        "greppy-desktop-fake-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::create_dir_all(&dir);
    let outcome = with_state_dir(Some(&dir), || {
        upstream::execute(command, Some(fake), limits)
    });
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

static STATE_LOCK: Mutex<()> = Mutex::new(());

fn with_state_dir<T>(dir: Option<&Path>, body: impl FnOnce() -> T) -> T {
    let _guard = STATE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let previous = std::env::var_os("AGENT_DESKTOP_HOME");
    if let Some(dir) = dir {
        std::env::set_var("AGENT_DESKTOP_HOME", dir);
    }
    let result = body();
    match previous {
        Some(value) => std::env::set_var("AGENT_DESKTOP_HOME", value),
        None => std::env::remove_var("AGENT_DESKTOP_HOME"),
    }
    result
}

/// Absolute state directory the sandboxed agent should grant. Exposed so the
/// write-sandbox check can stay next to the adapter without importing it.
pub fn preferred_state_dir(scratch: &Path) -> PathBuf {
    scratch.join("agent-desktop")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn limits() -> Limits {
        Limits {
            json: true,
            max_bytes: Some(64 * 1024),
        }
    }

    fn doc(command: &Command) -> Value {
        run_fake(command, &FakeDesktop::fixture(), &limits()).document
    }

    fn assert_envelope(value: &Value, command: &str, ok: bool) {
        assert_eq!(value["schema"], "greppy.desktop.v1");
        assert_eq!(value["command"], command);
        assert_eq!(value["ok"], ok, "{value}");
        assert_eq!(value["untrusted"], true);
        assert!(value
            .get("retry")
            .and_then(|v| v.get("disposition"))
            .is_some());
        assert!(value.get("data").is_some());
        assert!(value.get("error").is_some());
        assert!(value.get("truncated").is_some());
        for key in [
            "schema",
            "ok",
            "command",
            "untrusted",
            "retry",
            "data",
            "error",
            "truncated",
        ] {
            assert!(value.get(key).is_some(), "missing {key}");
        }
    }

    #[test]
    fn upstream_pin_is_single_tag() {
        let manifest = include_str!("../Cargo.toml");
        let tag = metadata(manifest, "agent-desktop-tag");
        let commit = metadata(manifest, "agent-desktop-commit");
        assert_eq!(tag, "v1.1.0");
        assert_eq!(commit.len(), 40);
        let mut tags = Vec::new();
        for line in manifest.lines() {
            if !line.contains("github.com/vercel-labs/agent-desktop") {
                continue;
            }
            if let Some(start) = line.find("tag = \"") {
                let rest = &line[start + "tag = \"".len()..];
                let tag_value = rest.split('"').next().unwrap_or("");
                if !tag_value.is_empty() {
                    tags.push(tag_value.to_owned());
                }
            }
        }
        assert_eq!(
            tags.len(),
            4,
            "core plus three platform crates, got {tags:?}"
        );
        assert!(tags.iter().all(|item| item == &tag), "{tags:?}");
        let lock = include_str!("../../../Cargo.lock");
        assert!(
            lock.contains(commit.as_str()),
            "Cargo.lock does not record the pinned agent-desktop commit"
        );
    }

    #[test]
    fn platform_adapters_are_target_cfg_dependencies_only() {
        let manifest = include_str!("../Cargo.toml");
        let unconditional = target_table_after(manifest, "[dependencies]");
        let code = unconditional
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        for foreign in [
            "agent-desktop-linux",
            "agent-desktop-macos",
            "agent-desktop-windows",
            "zbus",
            "x11rb",
            "uiautomation",
        ] {
            assert!(
                !code.contains(foreign),
                "{foreign} must not be an unconditional dependency"
            );
        }
        assert!(code.contains("agent-desktop-core"));
        let linux = target_table(manifest, "linux");
        let macos = target_table(manifest, "macos");
        let windows = target_table(manifest, "windows");
        assert!(linux.contains("agent-desktop-linux"));
        assert!(!linux.contains("agent-desktop-macos"));
        assert!(!linux.contains("agent-desktop-windows"));
        assert!(macos.contains("agent-desktop-macos"));
        assert!(!macos.contains("agent-desktop-linux"));
        assert!(windows.contains("agent-desktop-windows"));
        assert!(!windows.contains("agent-desktop-linux"));
    }

    fn target_table(manifest: &str, os: &str) -> String {
        let header = format!("[target.'cfg(target_os = \"{os}\")'.dependencies]");
        target_table_after(manifest, &header)
    }

    fn target_table_after(manifest: &str, header: &str) -> String {
        let rest = manifest
            .split_once(header)
            .unwrap_or_else(|| panic!("missing {header}"))
            .1;
        rest.split("\n[").next().unwrap_or(rest).to_owned()
    }

    fn metadata(toml: &str, key: &str) -> String {
        let prefix = format!("{key} = ");
        toml.lines()
            .find_map(|line| line.trim().strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("missing {key}"))
            .trim_matches('"')
            .to_owned()
    }

    #[test]
    fn capability_manual_names_only_shipped_commands() {
        let manual = include_str!("../../../assets/prompts/capabilities/desktop.md");
        let mut named = Vec::new();
        for line in manual.lines() {
            let trimmed = line.trim_start();
            if line == trimmed || !trimmed.starts_with("greppy desktop ") {
                continue;
            }
            let word: String = trimmed["greppy desktop ".len()..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                .collect();
            if !word.is_empty() {
                named.push(word);
            }
        }
        assert!(!named.is_empty());
        for word in &named {
            assert!(
                COMMANDS.contains(&word.as_str()),
                "manual names unknown desktop command {word}"
            );
        }
        for word in COMMANDS {
            assert!(named.iter().any(|item| item == word), "manual omits {word}");
        }
    }

    #[test]
    fn list_apps_projects_a_stable_document() {
        let value = doc(&Command::ListApps { app: None });
        assert_envelope(&value, "list-apps", true);
        assert_eq!(value["data"]["apps"][0]["name"], "Editor");
        assert_eq!(value["data"]["apps"][0]["pid"], 4242);
        assert_eq!(value["data"]["apps"][0]["bundle_id"], "app.editor");
        assert!(value["data"]["apps"][0].get("presentation").is_none());
        assert_eq!(value["retry"]["disposition"], "not_applicable");
    }

    #[test]
    fn list_windows_permissions_and_screenshot_keep_greppy_fields() {
        let windows = doc(&Command::ListWindows { app: None });
        assert_envelope(&windows, "list-windows", true);
        assert_eq!(windows["data"]["windows"][0]["id"], "w1");
        assert_eq!(windows["data"]["windows"][0]["title"], "Notes");
        assert_eq!(windows["data"]["windows"][0]["app"], "Editor");

        let permissions = doc(&Command::Permissions { request: false });
        assert_envelope(&permissions, "permissions", true);
        assert_eq!(permissions["data"]["accessibility"]["state"], "granted");
        assert_eq!(
            permissions["data"]["screen_recording"]["state"],
            "not_required"
        );

        let shot = doc(&Command::Screenshot {
            app: None,
            window_id: None,
        });
        assert_envelope(&shot, "screenshot", true);
        assert_eq!(shot["data"]["format"], "png");
        assert_eq!(shot["data"]["width"], 1);
        assert_eq!(shot["data"]["height"], 1);
        assert!(shot["data"]["png_base64"].as_str().unwrap().len() > 20);
    }

    #[test]
    fn status_launch_focus_and_wait_survive_the_fake_adapter() {
        let status = doc(&Command::Status);
        assert_envelope(&status, "status", true);
        assert_eq!(
            status["data"]["permissions"]["accessibility"]["state"],
            "granted"
        );
        assert!(status["data"].get("version").is_none());

        let launch = doc(&Command::Launch {
            app: "Editor".into(),
        });
        assert_envelope(&launch, "launch", true);
        assert_eq!(launch["data"]["app"], "Editor");
        assert_eq!(launch["data"]["pid"], 4242);

        let focus = doc(&Command::FocusWindow {
            window_id: Some("w1".into()),
            app: None,
            title: None,
        });
        assert_envelope(&focus, "focus-window", true);
        assert_eq!(focus["data"]["focused"]["id"], "w1");

        let wait = doc(&Command::Wait { milliseconds: 1 });
        assert_envelope(&wait, "wait", true);
        assert_eq!(wait["data"]["waited_ms"], 1);
    }

    #[test]
    fn ref_commands_without_a_snapshot_stay_inside_the_error_envelope() {
        for command in [
            Command::Click {
                ref_id: "e1".into(),
            },
            Command::Type {
                ref_id: "e1".into(),
                text: "hi".into(),
            },
            Command::SetValue {
                ref_id: "e1".into(),
                value: "hi".into(),
            },
            Command::Scroll {
                ref_id: "e1".into(),
                direction: "down".into(),
                amount: 1,
            },
            Command::Get {
                ref_id: "e1".into(),
                property: "role".into(),
            },
            Command::Press {
                combo: "Return".into(),
            },
            Command::Find {
                query: "Notes".into(),
            },
            Command::Snapshot {
                app: None,
                interactive_only: true,
            },
        ] {
            let value = doc(&command);
            assert_envelope(&value, command.name(), false);
            let code = value["error"]["code"].as_str().unwrap();
            assert!(
                matches!(
                    code,
                    "snapshot_not_found"
                        | "element_not_found"
                        | "stale_ref"
                        | "invalid_args"
                        | "platform_not_supported"
                        | "action_not_supported"
                        | "action_failed"
                        | "timeout"
                        | "internal"
                        | "schema_mismatch"
                        | "desktop_error"
                ),
                "{} returned unexpected code {code}: {value}",
                command.name()
            );
            let disposition = value["retry"]["disposition"].as_str().unwrap();
            assert!(matches!(disposition, "safe" | "unsafe" | "unknown"));
        }
    }

    #[test]
    fn bad_property_and_direction_are_usage_errors() {
        let get = run_fake(
            &Command::Get {
                ref_id: "e1".into(),
                property: "nope".into(),
            },
            &FakeDesktop::fixture(),
            &limits(),
        );
        assert_eq!(get.exit_code, schema::EXIT_USAGE);
        assert_eq!(get.document["error"]["code"], "invalid_args");
        let scroll = run_fake(
            &Command::Scroll {
                ref_id: "e1".into(),
                direction: "sideways".into(),
                amount: 1,
            },
            &FakeDesktop::fixture(),
            &limits(),
        );
        assert_eq!(scroll.exit_code, schema::EXIT_USAGE);
    }

    #[test]
    fn builtin_agent_without_the_capability_is_refused() {
        let _guard = STATE_LOCK.lock().unwrap();
        let prev_run = std::env::var_os("GREPPY_AGENT_RUN");
        let prev_cap = std::env::var_os("GREPPY_CAPABILITY_DESKTOP");
        std::env::set_var("GREPPY_AGENT_RUN", "1");
        std::env::remove_var("GREPPY_CAPABILITY_DESKTOP");
        let blocked = run(&Command::Status, &limits());
        std::env::set_var("GREPPY_CAPABILITY_DESKTOP", "1");
        assert!(!builtin_agent_blocks_desktop());
        match prev_run {
            Some(value) => std::env::set_var("GREPPY_AGENT_RUN", value),
            None => std::env::remove_var("GREPPY_AGENT_RUN"),
        }
        match prev_cap {
            Some(value) => std::env::set_var("GREPPY_CAPABILITY_DESKTOP", value),
            None => std::env::remove_var("GREPPY_CAPABILITY_DESKTOP"),
        }
        assert_eq!(blocked.exit_code, schema::EXIT_UNAVAILABLE);
        assert_eq!(blocked.document["error"]["code"], "desktop_disabled");
        assert_eq!(blocked.document["ok"], false);
    }

    #[test]
    fn max_bytes_keeps_the_envelope_and_marks_truncation() {
        let outcome = run_fake(
            &Command::ListApps { app: None },
            &FakeDesktop::fixture(),
            &Limits {
                json: true,
                max_bytes: Some(80),
            },
        );
        assert_eq!(outcome.document["truncated"], true);
        assert_eq!(outcome.document["schema"], SCHEMA);
        assert!(outcome.body.len() <= 400);
    }

    #[test]
    #[ignore = "needs a Linux desktop session (Xvfb and at-spi2-core); not run in unit tests"]
    fn linux_status_against_the_real_adapter_is_still_greppy_json() {
        let outcome = run(&Command::Status, &limits());
        assert_eq!(outcome.document["schema"], SCHEMA);
        assert_eq!(outcome.document["command"], "status");
        assert!(outcome.document.get("ok").is_some());
    }
}
