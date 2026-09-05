//! Client/supervisor protocol for the Greppy web runtime.
//!
//! This crate must not depend on `deno_core` or `servo`.

mod frame;
pub mod observation_context;
mod protocol;

pub use observation_context::{
    ActionOperation, ActionOutcome, ActionReceipt, ActionTicket, ObservationContext,
    ObservationContextError, ObservationContextSchema, ObservationContextState, SetGoalRequest,
};

pub use frame::{read_frame, write_frame, FrameError, MAX_FRAME_BYTES};
pub use protocol::{
    new_request_id, new_session_id, ErrorObject, Handshake, Metrics, Request, Response, SCHEMA,
};

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::{request as unix_request, serve_connection, UnixClientError};
