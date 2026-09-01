//! Slash-command parsing and the command catalog.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    Help,
    Setup,
    Clear,
    Model { query: String },
    Endpoint { url: String },
    Usage,
    Tools,
    Copy,
    Exit,
    Sessions { query: String },
    Name { title: String },
    Compact,
    Unknown(String),
}

#[derive(Debug, Clone, Copy)]
pub struct CommandSpec {
    pub name: &'static str,
    pub summary: &'static str,
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/help",
        summary: "commands and key bindings",
    },
    CommandSpec {
        name: "/setup",
        summary: "configure all agent settings",
    },
    CommandSpec {
        name: "/clear",
        summary: "clear the visible transcript",
    },
    CommandSpec {
        name: "/model",
        summary: "select a gateway model",
    },
    CommandSpec {
        name: "/endpoint",
        summary: "set and verify the model gateway URL",
    },
    CommandSpec {
        name: "/usage",
        summary: "session token usage",
    },
    CommandSpec {
        name: "/tools",
        summary: "tool executions for this session",
    },
    CommandSpec {
        name: "/copy",
        summary: "copy the last assistant reply",
    },
    CommandSpec {
        name: "/sessions",
        summary: "resume a saved session",
    },
    CommandSpec {
        name: "/name",
        summary: "rename the current session",
    },
    CommandSpec {
        name: "/compact",
        summary: "summarize older messages, keep recent ones",
    },
    CommandSpec {
        name: "/exit",
        summary: "finish and publish the proposal",
    },
];

pub fn parse_slash(input: &str) -> Option<SlashCommand> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let (cmd, rest) = trimmed
        .split_once(char::is_whitespace)
        .map(|(a, b)| (a, b.trim().to_string()))
        .unwrap_or((trimmed, String::new()));
    Some(match cmd {
        "/help" => SlashCommand::Help,
        "/setup" => SlashCommand::Setup,
        "/clear" => SlashCommand::Clear,
        "/model" => SlashCommand::Model { query: rest },
        "/endpoint" => SlashCommand::Endpoint { url: rest },
        "/usage" => SlashCommand::Usage,
        "/tools" => SlashCommand::Tools,
        "/copy" => SlashCommand::Copy,
        "/exit" | "/quit" | "/q" => SlashCommand::Exit,
        "/sessions" => SlashCommand::Sessions { query: rest },
        "/name" => SlashCommand::Name { title: rest },
        "/compact" => SlashCommand::Compact,
        other => SlashCommand::Unknown(other.to_string()),
    })
}

pub fn completions(prefix: &str) -> Vec<CommandSpec> {
    let needle = prefix.trim().to_ascii_lowercase();
    COMMANDS
        .iter()
        .copied()
        .filter(|spec| spec.name.starts_with(&needle) || needle.is_empty())
        .collect()
}

pub const KEYBINDINGS: &[&str] = &[
    "Enter                 submit prompt",
    "Shift/Alt+Enter       newline",
    "PageUp/PageDown       scroll viewport",
    "Mouse wheel           scroll transcript",
    "End                   follow stream",
    "Tab / Shift+Tab       next/previous completion",
    "Ctrl+C                cancel run; idle exit; twice force-exit",
    "Esc                   close overlay, then completions",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_commands_and_aliases() {
        assert_eq!(parse_slash("/help"), Some(SlashCommand::Help));
        assert_eq!(parse_slash("/setup"), Some(SlashCommand::Setup));
        assert_eq!(parse_slash(" /q "), Some(SlashCommand::Exit));
        assert_eq!(
            parse_slash("/name Review parser"),
            Some(SlashCommand::Name {
                title: "Review parser".into()
            })
        );
        assert_eq!(
            parse_slash("/endpoint http://127.0.0.1:8317"),
            Some(SlashCommand::Endpoint {
                url: "http://127.0.0.1:8317".into()
            })
        );
        assert!(matches!(
            parse_slash("/nope"),
            Some(SlashCommand::Unknown(_))
        ));
        assert_eq!(parse_slash("hello"), None);
    }

    #[test]
    fn completion_filters_prefix() {
        let hits = completions("/mo");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "/model");
    }
}
