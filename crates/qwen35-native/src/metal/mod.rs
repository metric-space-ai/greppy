/// Metal backend hook for the Qwen3.5 summarizer.
///
/// The previous local Qwen3.5 Metal probe is AGPL and is intentionally not
/// vendored into this MIT workspace. The production backend must be a
/// clean-room port or come from compatibly licensed kernels.
pub const BACKEND_STATUS: &str = "pending-clean-room-metal-forward";

pub mod model;
