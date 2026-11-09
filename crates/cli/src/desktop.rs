//! `greppy desktop` — accessibility-tree computer use.
//!
//! The command surface is greppy's. Upstream types stay in `greppy-desktop`.

use clap::Subcommand;
use greppy_core::error::{Error, Result};
use greppy_desktop::{run, Command as DesktopCommand, Limits};

#[derive(Debug, Subcommand)]
pub enum DesktopVerb {
    /// Adapter, permissions, and session state.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Accessibility and screen-recording permission state.
    Permissions {
        /// Ask the platform to prompt for missing permissions.
        #[arg(long)]
        request: bool,
        #[arg(long)]
        json: bool,
    },
    /// Applications the adapter can see.
    ListApps {
        /// Case-insensitive name or bundle id.
        app: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Windows the adapter can see.
    ListWindows {
        app: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Start an application, or attach to one already running.
    Launch {
        app: String,
        #[arg(long)]
        json: bool,
    },
    /// Raise one window and wait until it is focused.
    FocusWindow {
        #[arg(long)]
        window_id: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Accessibility outline of the front window.
    Snapshot {
        #[arg(long)]
        app: Option<String>,
        /// Interactive controls only.
        #[arg(short = 'i', long)]
        interactive: bool,
        #[arg(long)]
        json: bool,
    },
    /// Find elements whose text matches QUERY.
    Find {
        query: String,
        #[arg(long)]
        json: bool,
    },
    /// Read one property of a snapshot ref.
    Get {
        #[arg(value_name = "REF")]
        ref_id: String,
        /// text, value, title, bounds, role, or states.
        #[arg(long, default_value = "text")]
        property: String,
        #[arg(long)]
        json: bool,
    },
    /// Activate a snapshot ref.
    Click {
        #[arg(value_name = "REF")]
        ref_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Type into a snapshot ref. Secrets use --from-env or --value-stdin.
    Type {
        #[arg(value_name = "REF")]
        ref_id: String,
        text: Option<String>,
        /// Read the text from an environment variable. The value is not echoed.
        #[arg(long, value_name = "NAME")]
        from_env: Option<String>,
        /// Read the text from stdin. The value is not echoed.
        #[arg(long)]
        value_stdin: bool,
        #[arg(long)]
        json: bool,
    },
    /// Replace the value of a snapshot ref.
    SetValue {
        #[arg(value_name = "REF")]
        ref_id: String,
        value: String,
        #[arg(long)]
        json: bool,
    },
    /// Press a key combo such as Return or ctrl+l.
    Press {
        combo: String,
        #[arg(long)]
        json: bool,
    },
    /// Scroll a snapshot ref.
    Scroll {
        #[arg(value_name = "REF")]
        ref_id: String,
        #[arg(long, default_value = "down")]
        direction: String,
        #[arg(long, default_value_t = 3)]
        amount: u32,
        #[arg(long)]
        json: bool,
    },
    /// Wait for a number of milliseconds.
    Wait {
        #[arg(long)]
        ms: u64,
        #[arg(long)]
        json: bool,
    },
    /// Capture a PNG. With --json the document carries png_base64; the built-in
    /// agent attaches that image and does not keep the bytes in the session log.
    Screenshot {
        #[arg(long)]
        app: Option<String>,
        #[arg(long)]
        window_id: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

impl DesktopVerb {
    pub(crate) fn requests_json(&self) -> bool {
        match self {
            Self::Status { json }
            | Self::Permissions { json, .. }
            | Self::ListApps { json, .. }
            | Self::ListWindows { json, .. }
            | Self::Launch { json, .. }
            | Self::FocusWindow { json, .. }
            | Self::Snapshot { json, .. }
            | Self::Find { json, .. }
            | Self::Get { json, .. }
            | Self::Click { json, .. }
            | Self::Type { json, .. }
            | Self::SetValue { json, .. }
            | Self::Press { json, .. }
            | Self::Scroll { json, .. }
            | Self::Wait { json, .. }
            | Self::Screenshot { json, .. } => *json,
        }
    }
}

pub fn dispatch(json: bool, verb: DesktopVerb, max_bytes: Option<usize>) -> Result<i32> {
    let command = to_command(verb)?;
    let outcome = run(&command, &Limits { json, max_bytes });
    println!("{}", outcome.body);
    Ok(outcome.exit_code)
}

fn to_command(verb: DesktopVerb) -> Result<DesktopCommand> {
    Ok(match verb {
        DesktopVerb::Status { json: _ } => DesktopCommand::Status,
        DesktopVerb::Permissions { request, json: _ } => DesktopCommand::Permissions { request },
        DesktopVerb::ListApps { app, json: _ } => DesktopCommand::ListApps { app },
        DesktopVerb::ListWindows { app, json: _ } => DesktopCommand::ListWindows { app },
        DesktopVerb::Launch { app, json: _ } => DesktopCommand::Launch { app },
        DesktopVerb::FocusWindow {
            window_id,
            app,
            title,
            json: _,
        } => DesktopCommand::FocusWindow {
            window_id,
            app,
            title,
        },
        DesktopVerb::Snapshot {
            app,
            interactive,
            json: _,
        } => DesktopCommand::Snapshot {
            app,
            interactive_only: interactive,
        },
        DesktopVerb::Find { query, json: _ } => DesktopCommand::Find { query },
        DesktopVerb::Get {
            ref_id,
            property,
            json: _,
        } => DesktopCommand::Get { ref_id, property },
        DesktopVerb::Click { ref_id, json: _ } => DesktopCommand::Click { ref_id },
        DesktopVerb::Type {
            ref_id,
            text,
            from_env,
            value_stdin,
            json: _,
        } => DesktopCommand::Type {
            ref_id,
            text: secret_text(text, from_env, value_stdin)?,
        },
        DesktopVerb::SetValue {
            ref_id,
            value,
            json: _,
        } => DesktopCommand::SetValue { ref_id, value },
        DesktopVerb::Press { combo, json: _ } => DesktopCommand::Press { combo },
        DesktopVerb::Scroll {
            ref_id,
            direction,
            amount,
            json: _,
        } => DesktopCommand::Scroll {
            ref_id,
            direction,
            amount,
        },
        DesktopVerb::Wait { ms, json: _ } => DesktopCommand::Wait { milliseconds: ms },
        DesktopVerb::Screenshot {
            app,
            window_id,
            json: _,
        } => DesktopCommand::Screenshot { app, window_id },
    })
}

fn secret_text(
    text: Option<String>,
    from_env: Option<String>,
    value_stdin: bool,
) -> Result<String> {
    let sources =
        usize::from(text.is_some()) + usize::from(from_env.is_some()) + usize::from(value_stdin);
    if sources != 1 {
        return Err(Error::Invalid(
            "type takes exactly one of TEXT, --from-env NAME, or --value-stdin".into(),
        ));
    }
    if let Some(name) = from_env {
        return std::env::var(&name).map_err(|_| {
            Error::Invalid(format!(
                "environment variable {name} is not set; the value was not read"
            ))
        });
    }
    if value_stdin {
        let mut body = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut body)
            .map_err(|error| Error::Invalid(format!("read --value-stdin: {error}")))?;
        return Ok(body.trim_end_matches(['\n', '\r']).to_owned());
    }
    Ok(text.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;

    #[test]
    fn parse_desktop_status_json_and_type_from_env() {
        let cli = Cli::try_parse_from(["greppy", "desktop", "status", "--json"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Desktop {
                command: DesktopVerb::Status { json: true },
            })
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "desktop",
            "type",
            "e1",
            "--from-env",
            "TOKEN",
            "--json",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Desktop {
                command:
                    DesktopVerb::Type {
                        ref_id,
                        from_env,
                        text,
                        value_stdin,
                        json: true,
                    },
            }) => {
                assert_eq!(ref_id, "e1");
                assert_eq!(from_env.as_deref(), Some("TOKEN"));
                assert!(text.is_none());
                assert!(!value_stdin);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
