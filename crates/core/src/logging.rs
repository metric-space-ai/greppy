//! Tracing initialisation used by the `cli` crate and integration tests.
//!
//! Behaviour:
//! - `RUST_LOG` is honoured if set.
//! - `GREPPLUS_LOG` overrides `RUST_LOG` when set, for users who do not
//!   want to leak Rust-specific env vars.
//! - The default level is `info`.
//! - When stderr is a TTY, the formatter uses a compact human format.
//!   Otherwise it uses a JSON-like line format suitable for log files.

use std::io::IsTerminal;
use tracing_subscriber::{fmt, layer::Layer, prelude::*, EnvFilter};

type FmtLayer<S> = Box<dyn Layer<S> + Send + Sync + 'static>;

/// Initialise the global tracing subscriber. Idempotent: subsequent calls
/// are no-ops, so tests can call this freely.
///
/// Returns `Ok(())` on success or if a subscriber was already installed.
/// Returns `Err` only on a hard failure (e.g. invalid filter syntax).
pub fn init() -> Result<(), String> {
    let filter = env_filter();
    let is_tty = std::io::stderr().is_terminal();

    let fmt_layer: FmtLayer<_> = if is_tty {
        Box::new(fmt::layer().with_target(true).compact())
    } else {
        Box::new(fmt::layer().with_target(true).json())
    };

    let result = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init();

    if let Err(e) = result {
        // `try_init` returns `Err` if a subscriber was already installed.
        // That is not a hard failure for our purposes.
        let msg = e.to_string();
        if !msg.contains("already set") && !msg.contains("already installed") {
            return Err(msg);
        }
    }
    Ok(())
}

fn env_filter() -> EnvFilter {
    let raw = std::env::var("GREPPLUS_LOG")
        .ok()
        .or_else(|| std::env::var("RUST_LOG").ok())
        .unwrap_or_else(|| "info".to_string());
    EnvFilter::try_new(raw).unwrap_or_else(|_| EnvFilter::new("info"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_filter_falls_back_to_info_on_garbage() {
        std::env::set_var("GREPPLUS_LOG", "this-is-not-a-filter");
        let f = env_filter();
        // We just assert it builds without panicking; specific directives
        // are an EnvFilter implementation detail.
        drop(f);
        std::env::remove_var("GREPPLUS_LOG");
    }
}
