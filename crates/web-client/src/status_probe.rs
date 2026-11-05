//! Bounded status probes: a failed query is not an ownership denial.
use crate::{new_request_id, ErrorObject};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStatus {
    pub runtime_image_id: String,
    pub sessions: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeStatusProbe {
    Ready(RuntimeStatus),
    Starting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusProbeError {
    NotRunning,
    NotOwned,
    Transport,
    Malformed,
    Rejected,
    Starting,
}

impl StatusProbeError {
    pub fn into_error(self) -> ErrorObject {
        let (code, exit, message, recovery) = match self {
            Self::NotOwned => (
                "session_not_owned", 32, "web-runtime explicitly rejected this run or attach capability",
                "resume through the issuing Greppy parent; standalone shells must use the same workspace, runtime installation and saved owner route",
            ),
            other => (
                "runtime_unavailable", 31,
                match other {
                    Self::NotRunning => "saved web-runtime socket is unavailable",
                    Self::Transport => "web-runtime status transport failed; ownership is unverified",
                    Self::Malformed => "web-runtime status response is invalid; ownership is unverified",
                    Self::Rejected => "web-runtime rejected status without an ownership denial",
                    Self::Starting => "web-runtime is still starting; ownership is unverified",
                    Self::NotOwned => unreachable!(),
                },
                "inspect greppy web doctor and the original owner context; preserve the saved session route and do not automatically repeat the failed operation",
            ),
        };
        ErrorObject::new(code, message, new_request_id(), exit, recovery)
    }
}

#[cfg(unix)]
pub fn probe_runtime_status(
    socket: &std::path::Path,
    run_id: &str,
    capability: &str,
) -> Result<RuntimeStatusProbe, StatusProbeError> {
    use crate::{FrameError, Request, UnixClientError, SCHEMA};
    use std::io::ErrorKind;
    let mut request = Request::new(run_id, "web.status", serde_json::json!({}));
    request.capability = capability.to_owned();
    let response = crate::unix_request(socket, &request, std::time::Duration::from_millis(400))
        .map_err(|error| match error {
            UnixClientError::Connect(error)
                if matches!(
                    error.kind(),
                    ErrorKind::NotFound | ErrorKind::ConnectionRefused
                ) =>
            {
                StatusProbeError::NotRunning
            }
            UnixClientError::Frame(
                FrameError::Empty
                | FrameError::Oversize { .. }
                | FrameError::Utf8(_)
                | FrameError::Json(_),
            ) => StatusProbeError::Malformed,
            _ => StatusProbeError::Transport,
        })?;
    if response.schema != SCHEMA
        || response.request_id != request.request_id
        || response.operation != request.operation
    {
        return Err(StatusProbeError::Malformed);
    }
    if let Some(error) = response.error {
        if response.status != "error" || response.result.is_some() {
            return Err(StatusProbeError::Malformed);
        }
        return match error.code.as_str() {
            "session_not_owned" if error.exit_code == 32 => Err(StatusProbeError::NotOwned),
            "starting" => Ok(RuntimeStatusProbe::Starting),
            _ => Err(StatusProbeError::Rejected),
        };
    }
    if response.status != "ok" {
        return Err(StatusProbeError::Malformed);
    }
    let result = response.result.ok_or(StatusProbeError::Malformed)?;
    if result.get("state").and_then(|value| value.as_str()) == Some("starting") {
        return Ok(RuntimeStatusProbe::Starting);
    }
    let image = result
        .get("runtime_build_id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or(StatusProbeError::Malformed)?;
    Ok(RuntimeStatusProbe::Ready(RuntimeStatus {
        runtime_image_id: image.to_owned(),
        sessions: result.get("sessions").and_then(|value| value.as_u64()),
    }))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{read_frame, write_frame, Request, Response};
    use serde_json::json;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::Duration;

    fn exchange(
        handle: impl FnOnce(std::os::unix::net::UnixStream, Request) + Send + 'static,
    ) -> Result<RuntimeStatusProbe, StatusProbeError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "gp-status-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let socket = directory.join("s");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let request: Request = read_frame(&mut stream).unwrap();
            assert_eq!(request.operation, "web.status");
            handle(stream, request);
        });
        let result = probe_runtime_status(
            &socket,
            "private-contract-run",
            "synthetic-contract-capability",
        );
        server.join().unwrap();
        std::fs::remove_file(socket).unwrap();
        std::fs::remove_dir(directory).unwrap();
        result
    }

    #[test]
    fn status_probe_private_socket_distinguishes_denial_transport_and_malformed() {
        let denied = exchange(|mut stream, request| {
            let response = Response::error(
                &request,
                ErrorObject::new(
                    "session_not_owned",
                    "denied",
                    request.request_id.clone(),
                    32,
                    "original owner",
                ),
            );
            write_frame(&mut stream, &response).unwrap();
        })
        .unwrap_err();
        assert_eq!(denied, StatusProbeError::NotOwned);
        assert_eq!(denied.into_error().exit_code, 32);
        let transport = exchange(|_stream, _request| {}).unwrap_err();
        assert_eq!(transport, StatusProbeError::Transport);
        assert_eq!(
            transport.into_error().exit_code,
            31,
            "transport is not proof of another owner"
        );
        for value in [
            json!({}),
            json!({"runtime_build_id":42}),
            json!({"runtime_build_id":""}),
        ] {
            let malformed = exchange(move |mut stream, request| {
                write_frame(&mut stream, &Response::ok(&request, value)).unwrap();
            })
            .unwrap_err();
            assert_eq!(malformed, StatusProbeError::Malformed);
            assert_eq!(malformed.into_error().exit_code, 31);
        }
        let uncorrelated = exchange(|mut stream, request| {
            let mut response = Response::error(
                &request,
                ErrorObject::new(
                    "session_not_owned",
                    "denied",
                    request.request_id.clone(),
                    32,
                    "original owner",
                ),
            );
            response.request_id = "unrelated".into();
            write_frame(&mut stream, &response).unwrap();
        })
        .unwrap_err();
        assert_eq!(uncorrelated, StatusProbeError::Malformed);
        let invalid_frame = exchange(|mut stream, _request| {
            write_frame(&mut stream, &json!({"not":"a response"})).unwrap();
        })
        .unwrap_err();
        assert_eq!(invalid_frame, StatusProbeError::Malformed);
        let rejected = exchange(|mut stream, request| {
            write_frame(
                &mut stream,
                &Response::error(
                    &request,
                    ErrorObject::new(
                        "unavailable",
                        "not ready",
                        request.request_id.clone(),
                        31,
                        "inspect",
                    ),
                ),
            )
            .unwrap();
        })
        .unwrap_err();
        assert_eq!(rejected, StatusProbeError::Rejected);
        let ready = exchange(|mut stream, request| {
            write_frame(
                &mut stream,
                &Response::ok(
                    &request,
                    json!({"runtime_build_id":"private-image", "sessions":2}),
                ),
            )
            .unwrap();
        })
        .unwrap();
        assert_eq!(
            ready,
            RuntimeStatusProbe::Ready(RuntimeStatus {
                runtime_image_id: "private-image".into(),
                sessions: Some(2)
            })
        );
        let starting = exchange(|mut stream, request| {
            write_frame(
                &mut stream,
                &Response::ok(&request, json!({"state":"starting"})),
            )
            .unwrap();
        })
        .unwrap();
        assert_eq!(starting, RuntimeStatusProbe::Starting);
        let starting_error = exchange(|mut stream, request| {
            write_frame(
                &mut stream,
                &Response::error(
                    &request,
                    ErrorObject::new(
                        "starting",
                        "starting",
                        request.request_id.clone(),
                        31,
                        "inspect",
                    ),
                ),
            )
            .unwrap();
        })
        .unwrap();
        assert_eq!(starting_error, RuntimeStatusProbe::Starting);
    }

    #[test]
    fn status_probe_silent_connected_peer_is_not_an_ownership_denial() {
        let error =
            exchange(|_stream, _request| thread::sleep(Duration::from_millis(650))).unwrap_err();
        assert_eq!(error, StatusProbeError::Transport);
        let error = error.into_error();
        assert_eq!(error.code, "runtime_unavailable");
        assert_eq!(error.exit_code, 31);
        assert!(!error.retryable);
        assert!(!error.next_action.contains("inherit"));
        assert!(error
            .next_action
            .contains("preserve the saved session route"));
        let denied = StatusProbeError::NotOwned.into_error();
        assert!(denied.next_action.contains("standalone shells"));
        assert!(!denied.next_action.contains("fd 4"));
    }
}
