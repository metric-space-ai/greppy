//! The only module that names agent-desktop types or calls `commands::*::execute`.
//!
//! Everything that leaves this file is a greppy [`crate::schema`] value.

use agent_desktop_core::commands::{
    click, find, focus_window, get, launch, list_apps, list_windows, permissions, press,
    screenshot, scroll, set_value, snapshot, status, type_text, wait,
};
use agent_desktop_core::system::launch_options::LaunchOptions;
use agent_desktop_core::system::launch_result::LaunchResult;
use agent_desktop_core::{
    response::output::ErrorPayload, AppError, AppInfo, AppPresentation, CommandContext, Deadline,
    Direction, ImageBuffer, ImageFormat, InteractionLease, PermissionReport, PermissionState,
    PlatformAdapter, ProcessId, SnapshotSurface, WindowInfo, WindowState,
};
use serde_json::{json, Value};

use crate::command::{Command, FakeApp, FakeDesktop, FakeWindow};
use crate::schema::{self, Failure, Limits, Outcome, Retry};

pub fn execute(command: &Command, fake: Option<&FakeDesktop>, limits: &Limits) -> Outcome {
    let name = command.name();
    let raw = if let Some(fake) = fake {
        dispatch(command, &FakeAdapter { fake })
    } else {
        match platform_adapter() {
            Ok(adapter) => dispatch(command, adapter.as_ref()),
            Err(message) => return schema::unavailable(command, message, limits),
        }
    };
    match raw {
        Ok(value) => match project(command, &value) {
            Ok(data) => schema::success(name, data, limits),
            Err(detail) => schema::failure(
                name,
                &Failure {
                    code: "schema_mismatch".into(),
                    upstream_code: None,
                    message: format!("desktop output no longer matches greppy.desktop.v1: {detail}"),
                    suggestion: Some(
                        "The pinned agent-desktop command returned a shape this adapter does not map. See THIRD_PARTY.md before bumping the pin.".into(),
                    ),
                    retry: Retry {
                        disposition: "unsafe",
                        after_ms: None,
                        requires_fresh_snapshot: false,
                    },
                    exit_code: schema::EXIT_FAILED,
                },
                limits,
            ),
        },
        Err(error) => schema::failure(name, &map_error(&error), limits),
    }
}

fn dispatch(command: &Command, adapter: &dyn PlatformAdapter) -> Result<Value, AppError> {
    let context = CommandContext::new(None, None, false)?;
    let report = permission_report(adapter);
    match command {
        Command::Status => status::execute_with_report_with_context(adapter, &report, &context),
        Command::Permissions { request } => permissions::execute_with_report(
            permissions::PermissionsArgs {
                request: *request,
                guide: false,
            },
            adapter,
            &report,
        ),
        Command::ListApps { app } => list_apps::execute(
            list_apps::ListAppsArgs {
                app: app.clone(),
                timeout_ms: Some(3_000),
            },
            adapter,
        ),
        Command::ListWindows { app } => list_windows::execute(
            list_windows::ListWindowsArgs {
                app: app.clone(),
                timeout_ms: Some(3_000),
            },
            adapter,
        ),
        Command::Launch { app } => launch::execute(
            launch::LaunchArgs {
                app: app.clone(),
                options: LaunchOptions {
                    timeout_ms: 3_000,
                    activate: true,
                    ..LaunchOptions::default()
                },
            },
            adapter,
        ),
        Command::FocusWindow {
            window_id,
            app,
            title,
        } => focus_window::execute(
            focus_window::FocusWindowArgs {
                window_id: window_id.clone(),
                app: app.clone(),
                title: title.clone(),
            },
            adapter,
        ),
        Command::Snapshot {
            app,
            interactive_only,
        } => snapshot::execute(
            snapshot::SnapshotArgs {
                app: app.clone(),
                window_id: None,
                max_depth: 8,
                include_bounds: false,
                interactive_only: *interactive_only,
                compact: true,
                surface: SnapshotSurface::Window,
                skeleton: false,
                root_ref: None,
                timeout_ms: Some(3_000),
                force_electron_a11y: false,
            },
            adapter,
            &context,
        ),
        Command::Find { query } => find::execute(
            find::FindArgs {
                app: None,
                window_id: None,
                root: None,
                surface: SnapshotSurface::Window,
                filter: find::FindFilterArgs {
                    role: None,
                    name: None,
                    description: None,
                    native_id: None,
                    value: None,
                    text: Some(query.clone()),
                    exact: false,
                },
                states: Vec::new(),
                selection: find::FindSelectionArgs {
                    count: false,
                    first: false,
                    last: false,
                    nth: None,
                    limit: Some(20),
                },
                timeout_ms: Some(3_000),
            },
            adapter,
            &context,
        ),
        Command::Get { ref_id, property } => get::execute(
            get::GetArgs {
                ref_id: ref_id.clone(),
                property: match property.as_str() {
                    "text" => get::GetProperty::Text,
                    "value" => get::GetProperty::Value,
                    "title" => get::GetProperty::Title,
                    "bounds" => get::GetProperty::Bounds,
                    "role" => get::GetProperty::Role,
                    "states" => get::GetProperty::States,
                    _ => {
                        return Err(AppError::invalid_input(
                            "property must be text, value, title, bounds, role, or states",
                        ));
                    }
                },
            },
            adapter,
            &context,
        ),
        Command::Click { ref_id } => click::execute(
            agent_desktop_core::commands::helpers::RefArgs {
                ref_id: ref_id.clone(),
                timeout_ms: None,
            },
            adapter,
            &context,
        ),
        Command::Type { ref_id, text } => type_text::execute(
            type_text::TypeArgs {
                ref_id: ref_id.clone(),
                text: text.clone(),
                timeout_ms: None,
                interval_ms: None,
            },
            adapter,
            &context,
        ),
        Command::SetValue { ref_id, value } => set_value::execute(
            set_value::SetValueArgs {
                ref_id: ref_id.clone(),
                value: value.clone(),
                timeout_ms: None,
            },
            adapter,
            &context,
        ),
        Command::Press { combo } => press::execute(
            press::PressArgs {
                combo: combo.clone(),
                app: None,
                window_id: None,
                force: false,
            },
            adapter,
            &context,
        ),
        Command::Scroll {
            ref_id,
            direction,
            amount,
        } => {
            let direction = match direction.as_str() {
                "up" => Direction::Up,
                "down" => Direction::Down,
                "left" => Direction::Left,
                "right" => Direction::Right,
                _ => {
                    return Err(AppError::invalid_input(
                        "direction must be up, down, left, or right",
                    ));
                }
            };
            scroll::execute(
                scroll::ScrollArgs {
                    ref_id: ref_id.clone(),
                    direction,
                    amount: *amount,
                    timeout_ms: None,
                },
                adapter,
                &context,
            )
        }
        Command::Wait { milliseconds } => wait::execute(
            wait::WaitArgs {
                mode: wait::WaitModeArgs {
                    ms: Some(*milliseconds),
                    ..wait::WaitModeArgs::default()
                },
                predicate: wait::WaitPredicateArgs {
                    predicate: None,
                    value: None,
                    action: None,
                    count: None,
                },
                timeout_ms: *milliseconds,
                app: None,
            },
            adapter,
            &context,
        ),
        Command::Screenshot { app, window_id } => screenshot::execute(
            screenshot::ScreenshotArgs {
                app: app.clone(),
                window_id: window_id.clone(),
                screen: None,
                output_path: None,
            },
            adapter,
        ),
    }
}

fn permission_report(adapter: &dyn PlatformAdapter) -> PermissionReport {
    let Ok(deadline) = Deadline::standard() else {
        return unknown_report();
    };
    adapter
        .permission_report(deadline)
        .unwrap_or_else(|_| unknown_report())
}

fn unknown_report() -> PermissionReport {
    PermissionReport {
        accessibility: PermissionState::Unknown,
        screen_recording: PermissionState::Unknown,
        automation: PermissionState::NotRequired,
    }
}

fn map_error(error: &AppError) -> Failure {
    let payload = ErrorPayload::from_app_error(error);
    let (code, upstream_code) = match payload.code.as_str() {
        "PERM_DENIED" => ("permission_denied", None),
        "ELEMENT_NOT_FOUND" => ("element_not_found", None),
        "APP_NOT_FOUND" => ("app_not_found", None),
        "ACTION_FAILED" => ("action_failed", None),
        "ACTION_NOT_SUPPORTED" => ("action_not_supported", None),
        "STALE_REF" => ("stale_ref", None),
        "AMBIGUOUS_TARGET" => ("ambiguous_target", None),
        "WINDOW_NOT_FOUND" => ("window_not_found", None),
        "PLATFORM_NOT_SUPPORTED" => ("platform_not_supported", None),
        "TIMEOUT" => ("timeout", None),
        "INVALID_ARGS" => ("invalid_args", None),
        "NOTIFICATION_NOT_FOUND" => ("notification_not_found", None),
        "SNAPSHOT_NOT_FOUND" => ("snapshot_not_found", None),
        "POLICY_DENIED" => ("policy_denied", None),
        "APP_UNRESPONSIVE" => ("app_unresponsive", None),
        "INTERNAL" => ("internal", None),
        other => ("desktop_error", Some(other.to_owned())),
    };
    let disposition = match serde_json::to_value(payload.disposition)
        .ok()
        .as_ref()
        .and_then(|value| value.get("retry"))
        .and_then(|retry| retry.as_str())
    {
        Some("safe") => "safe",
        Some("unsafe") => "unsafe",
        _ => "unknown",
    };
    let after_ms = payload
        .recovery
        .as_ref()
        .and_then(|hint| hint.retry_after_ms);
    let requires_fresh_snapshot = payload
        .recovery
        .as_ref()
        .is_some_and(|hint| hint.requires_fresh_snapshot);
    Failure {
        code: code.to_owned(),
        upstream_code,
        message: payload.message,
        suggestion: payload.suggestion,
        retry: Retry {
            disposition,
            after_ms,
            requires_fresh_snapshot,
        },
        exit_code: schema::exit_for_code(code, disposition),
    }
}

fn project(command: &Command, value: &Value) -> Result<Value, String> {
    match command {
        Command::Status => {
            let permissions = value.get("permissions").ok_or("permissions")?;
            Ok(json!({
                "platform": value.get("platform").and_then(|v| v.as_str()).ok_or("platform")?,
                "permissions": map_permissions(permissions)?,
                "snapshot_id": value.get("snapshot_id").cloned().unwrap_or(Value::Null),
                "ref_count": value.get("ref_count").cloned().unwrap_or(Value::Null),
                "session_id": value.get("session_id").cloned().unwrap_or(Value::Null),
            }))
        }
        Command::Permissions { .. } => map_permissions(value),
        Command::ListApps { .. } => {
            let apps = value.get("apps").and_then(|v| v.as_array()).ok_or("apps")?;
            let mut out = Vec::new();
            for app in apps {
                out.push(map_app(app)?);
            }
            Ok(json!({"apps": out}))
        }
        Command::ListWindows { .. } => {
            let windows = value.as_array().ok_or("windows")?;
            let mut out = Vec::new();
            for window in windows {
                out.push(map_window(window)?);
            }
            Ok(json!({"windows": out}))
        }
        Command::Launch { .. } => Ok(json!({
            "app": value.get("app").and_then(|v| v.as_str()).ok_or("app")?,
            "pid": value.get("pid").and_then(|v| v.as_u64()).ok_or("pid")?,
            "window_id": value.pointer("/window/id").cloned().unwrap_or(Value::Null),
        })),
        Command::FocusWindow { .. } => {
            let focused = value.get("focused").ok_or("focused")?;
            Ok(json!({"focused": map_window(focused)?}))
        }
        Command::Snapshot { .. } => Ok(json!({
            "app": value.get("app").cloned().unwrap_or(Value::Null),
            "snapshot_id": value.get("snapshot_id").cloned().unwrap_or(Value::Null),
            "ref_count": value.get("ref_count").cloned().unwrap_or(Value::Null),
            "window": value.get("window").cloned().unwrap_or(Value::Null),
            "tree": value.get("tree").cloned().unwrap_or(Value::Null),
        })),
        Command::Find { .. } => Ok(json!({
            "matches": value.get("matches").or_else(|| value.get("elements")).cloned().unwrap_or(Value::Null),
            "count": value.get("count").cloned().unwrap_or(Value::Null),
        })),
        Command::Get { .. } => Ok(json!({
            "property": value.get("property").cloned().unwrap_or(Value::Null),
            "value": value.get("value").cloned().unwrap_or(value.clone()),
        })),
        Command::Screenshot { .. } => {
            let format = value
                .get("format")
                .and_then(|v| v.as_str())
                .ok_or("format")?;
            let mut data = json!({
                "format": format,
                "width": value.get("width").and_then(|v| v.as_u64()).ok_or("width")?,
                "height": value.get("height").and_then(|v| v.as_u64()).ok_or("height")?,
            });
            if format == "png" {
                if let Some(png) = value.get("data").and_then(|v| v.as_str()) {
                    data["png_base64"] = json!(png);
                }
            }
            Ok(data)
        }
        Command::Wait { .. } => Ok(json!({
            "waited_ms": value.get("waited_ms").and_then(|v| v.as_u64()).ok_or("waited_ms")?,
        })),
        Command::Click { .. }
        | Command::Type { .. }
        | Command::SetValue { .. }
        | Command::Press { .. }
        | Command::Scroll { .. } => Ok(json!({
            "delivered": !value.is_null(),
        })),
    }
}

fn map_permissions(value: &Value) -> Result<Value, String> {
    Ok(json!({
        "accessibility": permission_state(value.get("accessibility").ok_or("accessibility")?)?,
        "screen_recording": permission_state(value.get("screen_recording").ok_or("screen_recording")?)?,
        "automation": permission_state(value.get("automation").ok_or("automation")?)?,
    }))
}

fn permission_state(value: &Value) -> Result<Value, String> {
    let state = value.get("state").and_then(|v| v.as_str()).ok_or("state")?;
    let mut out = json!({"state": state});
    if let Some(suggestion) = value.get("suggestion").and_then(|v| v.as_str()) {
        out["suggestion"] = json!(suggestion);
    }
    Ok(out)
}

fn map_app(value: &Value) -> Result<Value, String> {
    let mut out = json!({
        "name": value.get("name").and_then(|v| v.as_str()).ok_or("name")?,
        "pid": value.get("pid").and_then(|v| v.as_u64()).ok_or("pid")?,
    });
    if let Some(bundle) = value.get("bundle_id").and_then(|v| v.as_str()) {
        out["bundle_id"] = json!(bundle);
    }
    Ok(out)
}

fn map_window(value: &Value) -> Result<Value, String> {
    Ok(json!({
        "id": value.get("id").and_then(|v| v.as_str()).ok_or("id")?,
        "title": value.get("title").and_then(|v| v.as_str()).ok_or("title")?,
        "app": value
            .get("app")
            .or_else(|| value.get("app_name"))
            .and_then(|v| v.as_str())
            .ok_or("app")?,
        "pid": value.get("pid").and_then(|v| v.as_u64()).ok_or("pid")?,
    }))
}

fn platform_adapter() -> Result<Box<dyn PlatformAdapter>, String> {
    #[cfg(target_os = "linux")]
    {
        Ok(Box::new(agent_desktop_linux::LinuxAdapter::new()))
    }
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(agent_desktop_macos::MacOSAdapter::new()))
    }
    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(agent_desktop_windows::WindowsAdapter::new()))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err("greppy desktop has no accessibility adapter for this operating system".into())
    }
}

struct FakeAdapter<'a> {
    fake: &'a FakeDesktop,
}

impl agent_desktop_core::ObservationOps for FakeAdapter<'_> {
    fn list_windows(
        &self,
        filter: &agent_desktop_core::WindowFilter,
        _deadline: Deadline,
    ) -> Result<Vec<WindowInfo>, agent_desktop_core::AdapterError> {
        Ok(self
            .fake
            .windows
            .iter()
            .filter(|window| filter.app.as_ref().is_none_or(|app| app == &window.app))
            .map(window_info)
            .collect())
    }

    fn list_apps(
        &self,
        _deadline: Deadline,
    ) -> Result<Vec<AppInfo>, agent_desktop_core::AdapterError> {
        Ok(self.fake.apps.iter().map(app_info).collect())
    }
}

impl agent_desktop_core::ActionOps for FakeAdapter<'_> {}

impl agent_desktop_core::InputOps for FakeAdapter<'_> {}

impl agent_desktop_core::SystemOps for FakeAdapter<'_> {
    fn permission_report(
        &self,
        _deadline: Deadline,
    ) -> Result<PermissionReport, agent_desktop_core::AdapterError> {
        let accessibility = if self.fake.accessibility_granted {
            PermissionState::Granted
        } else {
            PermissionState::Denied {
                suggestion: "grant accessibility".into(),
            }
        };
        Ok(PermissionReport {
            accessibility,
            screen_recording: PermissionState::NotRequired,
            automation: PermissionState::NotRequired,
        })
    }

    fn unknown_accessibility_means_unsupported(&self) -> bool {
        false
    }

    fn supported_surfaces(&self) -> Vec<SnapshotSurface> {
        vec![SnapshotSurface::Window]
    }

    fn acquire_interaction_lease(
        &self,
        deadline: Deadline,
    ) -> Result<InteractionLease, agent_desktop_core::AdapterError> {
        InteractionLease::guarded(deadline, ())
    }

    fn focus_window(
        &self,
        _win: &WindowInfo,
        _lease: &InteractionLease,
    ) -> Result<(), agent_desktop_core::AdapterError> {
        Ok(())
    }

    fn focused_window(
        &self,
        _deadline: Deadline,
    ) -> Result<Option<WindowInfo>, agent_desktop_core::AdapterError> {
        Ok(self.fake.windows.first().map(window_info))
    }

    fn launch_app(
        &self,
        id: &str,
        _options: &LaunchOptions,
        _lease: &InteractionLease,
    ) -> Result<LaunchResult, agent_desktop_core::AdapterError> {
        let window = self
            .fake
            .windows
            .iter()
            .find(|window| window.app == id)
            .or_else(|| self.fake.windows.first())
            .map(window_info);
        Ok(LaunchResult {
            app: id.to_owned(),
            pid: ProcessId::new(self.fake.apps.first().map(|app| app.pid).unwrap_or(1)),
            process_instance: None,
            window,
            cdp: None,
            renderer: None,
            suggestion: None,
        })
    }

    fn screenshot(
        &self,
        _target: agent_desktop_core::ScreenshotTarget,
        _deadline: Deadline,
    ) -> Result<ImageBuffer, agent_desktop_core::AdapterError> {
        Ok(ImageBuffer {
            data: self.fake.png.clone(),
            format: ImageFormat::Png,
            width: 1,
            height: 1,
            scale_factor: 1.0,
        })
    }
}

fn app_info(app: &FakeApp) -> AppInfo {
    AppInfo {
        name: app.name.clone(),
        pid: ProcessId::new(app.pid),
        bundle_id: app.bundle_id.clone(),
        process_instance: None,
        presentation: Some(AppPresentation::Foreground),
    }
}

fn window_info(window: &FakeWindow) -> WindowInfo {
    WindowInfo {
        id: window.id.clone(),
        title: window.title.clone(),
        app: window.app.clone(),
        pid: ProcessId::new(window.pid),
        process_instance: None,
        bounds: None,
        state: WindowState {
            is_focused: true,
            accessible: true,
            minimized: Some(false),
            visible: Some(true),
        },
    }
}
