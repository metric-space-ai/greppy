//! Turn web or desktop manuals on for the built-in agent.
//!
//! `enable` prints greppy's own manual. The parent agent keeps that text as
//! the tool result and does not rewrite the system prompt. A successful
//! `enable desktop` also unlocks `greppy desktop` for the rest of the session.
//! Calling enable again prints the same manual.

use clap::Subcommand;
use greppy_agent::{capability_manual, capability_source, AGENT_RUN_ENV};
use greppy_core::error::{Error, Result};

#[derive(Debug, Subcommand)]
pub enum CapabilitiesVerb {
    /// Return the web or desktop manual. Inside the built-in agent this also
    /// unlocks desktop for the session. It does not change the system prompt.
    Enable {
        /// `web` or `desktop`.
        name: String,
    },
    /// Print the signed manual. This does not enable it.
    Show { name: String },
}

pub fn dispatch(command: CapabilitiesVerb) -> Result<i32> {
    match command {
        CapabilitiesVerb::Enable { name } => {
            let builtin = std::env::var_os(AGENT_RUN_ENV).is_some();
            let Some(text) = capability_manual(&name, builtin) else {
                return Err(Error::Invalid(format!(
                    "unknown capability '{name}'; expected web or desktop"
                )));
            };
            print_manual(text);
            Ok(0)
        }
        CapabilitiesVerb::Show { name } => {
            let Some(text) = capability_source(&name) else {
                return Err(Error::Invalid(format!(
                    "unknown capability '{name}'; expected web or desktop"
                )));
            };
            print_manual(text);
            Ok(0)
        }
    }
}

fn print_manual(text: &str) {
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;

    #[test]
    fn parse_capabilities_enable() {
        let cli = Cli::try_parse_from(["greppy", "capabilities", "enable", "desktop"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Capabilities {
                command: CapabilitiesVerb::Enable { ref name },
            }) if name == "desktop"
        ));
    }
}
