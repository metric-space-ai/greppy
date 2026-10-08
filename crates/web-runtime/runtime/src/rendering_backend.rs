//! Chooses a GL connection for the Servo content worker.
//!
//! `servo::SoftwareRenderingContext::new` always starts at surfman's default
//! `Connection::new`. On Linux that connection is Wayland unless this crate
//! enables surfman's `sm-x11` feature (see `Cargo.toml`). Wayland's
//! `Connection::new` calls `wayland_sys::client::wayland_client_handle()`,
//! which panics with `Library libwayland-client.so could not be loaded.`
//! when the shared library is absent — before `wl_display_connect` can
//! return an error and before the Unix multi backend can fall through to
//! X11 or Mesa surfaceless.
//!
//! Callers therefore probe the library (the same check as
//! `wayland_sys::client::is_lib_available`, which does not panic) and skip
//! the surfman constructor when the probe is false.

use std::fmt;

/// Install hint surfaced instead of the libwayland dlopen panic.
pub const LIBWAYLAND_INSTALL_HINT: &str =
    "install libwayland-client0 (Debian/Ubuntu) to enable the browser";

/// Backend chosen for `SoftwareRenderingContext::new`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderingBackend {
    /// `WAYLAND_DISPLAY` is set and libwayland-client loads. Surfman connects
    /// to that display.
    Wayland,
    /// No Wayland display. Surfman's Unix multi connection (enabled via
    /// `sm-x11`) tries Wayland, gets `ConnectionFailed` without panicking
    /// because the library is present, then uses X11 or Mesa surfaceless.
    HeadlessSoftware,
}

/// Why the content worker must not open a surfman connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderingStartupError {
    /// libwayland-client.so cannot be dlopened. Opening the default surfman
    /// connection would panic inside wayland-sys.
    MissingWaylandClient,
    /// The surfman constructor returned an error.
    Renderer(String),
}

impl fmt::Display for RenderingStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingWaylandClient => formatter.write_str(LIBWAYLAND_INSTALL_HINT),
            Self::Renderer(detail) => write!(formatter, "software renderer failed: {detail}"),
        }
    }
}

impl std::error::Error for RenderingStartupError {}

/// Pick a backend from an injected availability probe.
///
/// `libwayland_available` matches `wayland_sys::client::is_lib_available`:
/// false means a later `wayland_client_handle()` would panic. When it is
/// false this function returns [`RenderingStartupError::MissingWaylandClient`]
/// and the caller must not construct a surfman connection.
///
/// `wayland_display_set` is true when `WAYLAND_DISPLAY` is non-empty. An
/// unset display selects [`RenderingBackend::HeadlessSoftware`] so the
/// content worker does not require a compositor.
pub fn select_rendering_backend(
    libwayland_available: bool,
    wayland_display_set: bool,
) -> Result<RenderingBackend, RenderingStartupError> {
    if !libwayland_available {
        return Err(RenderingStartupError::MissingWaylandClient);
    }
    if wayland_display_set {
        Ok(RenderingBackend::Wayland)
    } else {
        Ok(RenderingBackend::HeadlessSoftware)
    }
}

/// Probe `libwayland-client.so` the way wayland-sys 0.31 does, without
/// panicking when it is missing.
///
/// Non-Linux targets do not dlopen this library from surfman's default
/// connection, so the probe reports the library as available.
pub fn libwayland_client_available() -> bool {
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
    #[cfg(target_os = "linux")]
    unsafe {
        // SAFETY: dlopen/dlclose are called with a static C string and the
        // returned handle is closed before this function returns. A null
        // handle means the library is absent; that is not an error.
        for name in [c"libwayland-client.so.0", c"libwayland-client.so"] {
            let handle = libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
            if !handle.is_null() {
                libc::dlclose(handle);
                return true;
            }
        }
        false
    }
}

/// Open the surfman-backed renderer selected by [`select_rendering_backend`].
///
/// `open_default` is `SoftwareRenderingContext::new`. It is not called when
/// the library probe is false. A panic whose message names
/// `libwayland-client` is converted into [`RenderingStartupError::MissingWaylandClient`]
/// so a stale probe cannot take down the process.
pub fn open_rendering_backend<T, F>(
    libwayland_available: bool,
    wayland_display_set: bool,
    open_default: F,
) -> Result<T, RenderingStartupError>
where
    F: FnOnce() -> Result<T, String> + std::panic::UnwindSafe,
{
    select_rendering_backend(libwayland_available, wayland_display_set)?;
    match std::panic::catch_unwind(open_default) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(detail)) => Err(RenderingStartupError::Renderer(detail)),
        Err(payload) => {
            let message = panic_message(&payload);
            if message.contains("libwayland-client") {
                Err(RenderingStartupError::MissingWaylandClient)
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_selects_install_error_for_either_display_state() {
        for display_set in [false, true] {
            let error = select_rendering_backend(false, display_set).unwrap_err();
            assert_eq!(
                error,
                RenderingStartupError::MissingWaylandClient,
                "display_set={display_set}"
            );
            assert_eq!(error.to_string(), LIBWAYLAND_INSTALL_HINT);
        }
    }

    #[test]
    fn unset_display_selects_headless_software_when_library_loads() {
        assert_eq!(
            select_rendering_backend(true, false).unwrap(),
            RenderingBackend::HeadlessSoftware
        );
    }

    #[test]
    fn set_display_selects_wayland_when_library_loads() {
        assert_eq!(
            select_rendering_backend(true, true).unwrap(),
            RenderingBackend::Wayland
        );
    }

    #[test]
    fn missing_library_does_not_call_the_surfman_constructor() {
        let error = open_rendering_backend(false, false, || -> Result<(), String> {
            panic!("surfman constructor must not run without libwayland");
        })
        .unwrap_err();
        assert_eq!(error.to_string(), LIBWAYLAND_INSTALL_HINT);
    }

    #[test]
    fn wayland_dlopen_panic_becomes_the_install_error() {
        let error = open_rendering_backend(true, true, || -> Result<(), String> {
            panic!("Library libwayland-client.so could not be loaded.");
        })
        .unwrap_err();
        assert_eq!(error.to_string(), LIBWAYLAND_INSTALL_HINT);
    }

    #[test]
    fn headless_selection_still_opens_the_default_connection() {
        let backend =
            open_rendering_backend(true, false, || Ok(RenderingBackend::HeadlessSoftware))
                .expect("headless open");
        assert_eq!(backend, RenderingBackend::HeadlessSoftware);
    }

    #[test]
    fn renderer_errors_keep_their_detail() {
        let error =
            open_rendering_backend(true, false, || Err("ConnectionFailed".to_owned())).unwrap_err();
        assert_eq!(
            error.to_string(),
            "software renderer failed: ConnectionFailed"
        );
    }
}
