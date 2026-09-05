//! Client/supervisor protocol for the Greppy web runtime.
//!
//! This crate must not depend on `deno_core` or `servo`.

mod frame;
mod protocol;

/// Shared DOM description for native ref inspection and CLI query inspection.
pub const DESCRIBE_NODE_JS: &str = include_str!("describe-node.js");

pub use frame::{read_frame, write_frame, FrameError, MAX_FRAME_BYTES};
pub use protocol::{
    new_request_id, new_session_id, ErrorObject, Handshake, Metrics, Request, Response, SCHEMA,
};

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::{request as unix_request, serve_connection, UnixClientError};
