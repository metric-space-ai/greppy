//! Model stream abstraction for the agent loop.
//!
//! Separates the turn-streaming contract from the concrete HTTP client so the
//! loop can be unit-tested with a scripted fake.

use crate::client::{Client, ClientError, TurnResult};
use crate::protocol::{ModelRequest, StreamEvent};

/// Streaming model interface used by the agent loop.
///
/// Implementations must invoke `on_event` for every incremental [`StreamEvent`]
/// observed while assembling the turn, then return the completed
/// [`TurnResult`]. Transport and protocol failures surface as [`ClientError`].
pub trait ModelStream {
    /// Stream one model turn against `req`, forwarding events to `on_event`.
    fn stream_turn(
        &mut self,
        req: &ModelRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError>;
}

impl ModelStream for Client {
    fn stream_turn(
        &mut self,
        req: &ModelRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        // Client::stream_turn takes &self; &mut self reborrows cleanly.
        Client::stream_turn(self, req, on_event)
    }
}
