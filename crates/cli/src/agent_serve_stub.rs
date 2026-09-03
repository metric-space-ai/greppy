//! Windows placeholder for the socket-hosted headless session.
//!
//! `greppy agent serve` needs the Unix control socket; the invocation path
//! already refuses early on other platforms, and this keeps the shared run
//! path compiling.

use greppy_agent::{AgentConfig, Client, GreppyEnv};

use crate::agent::SessionSummary;
use crate::agent_json::{JsonEmitter, JsonSession};
use crate::agent_tui::{SessionRecord, SessionStore};

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct ServeLaunch<'a> {
    pub(crate) task: &'a str,
    pub(crate) endpoint: &'a str,
    pub(crate) model: &'a str,
    pub(crate) sandbox: &'a str,
    pub(crate) idle_timeout_secs: Option<u64>,
    pub(crate) json_session: &'a JsonSession,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    _client: Client,
    _env: GreppyEnv,
    _config: AgentConfig,
    _store: &SessionStore,
    _record: SessionRecord,
    _resumed: bool,
    _emitter: &mut JsonEmitter,
    _launch: ServeLaunch<'_>,
) -> Result<SessionSummary, String> {
    Err("greppy agent serve: Unix domain sockets are unsupported on this platform".to_string())
}
