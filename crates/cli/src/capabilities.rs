//! Turn web or desktop manuals on for the built-in agent.
//!
//! Outside the agent this command only prints a receipt or the manual. Inside
//! the agent the parent records the capability and appends the manual to the
//! system prompt before the next turn.

use clap::Subcommand;
use greppy_agent::capability_source;
use greppy_core::error::{Error, Result};

#[derive(Debug, Subcommand)]
pub enum CapabilitiesVerb {
    /// Enable web or desktop. The built-in agent inserts the manual next turn.
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
            if capability_source(&name).is_none() {
                return Err(Error::Invalid(format!(
                    "unknown capability '{name}'; expected web or desktop"
                )));
            }
            println!(
                "enabled {name}; a built-in agent inserts the manual into the system prompt on the next turn"
            );
            Ok(0)
        }
        CapabilitiesVerb::Show { name } => {
            let Some(text) = capability_source(&name) else {
                return Err(Error::Invalid(format!(
                    "unknown capability '{name}'; expected web or desktop"
                )));
            };
            print!("{text}");
            if !text.ends_with('\n') {
                println!();
            }
            Ok(0)
        }
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
