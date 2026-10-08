//! Mode rendering of the two owner-signed prompt sources. No copied manual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinPromptMode {
    OneShot,
    Interactive,
    Serve,
    Acp,
}
impl BuiltinPromptMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::OneShot => "one-shot",
            Self::Interactive => "interactive",
            Self::Serve => "serve",
            Self::Acp => "acp",
        }
    }
    pub fn file_name(self) -> &'static str {
        match self {
            Self::OneShot => "one-shot-prompt.md",
            Self::Interactive => "interactive-prompt.md",
            Self::Serve => "serve-prompt.md",
            Self::Acp => "acp-prompt.md",
        }
    }
}
pub const BUILTIN_MODES: [BuiltinPromptMode; 4] = [
    BuiltinPromptMode::OneShot,
    BuiltinPromptMode::Interactive,
    BuiltinPromptMode::Serve,
    BuiltinPromptMode::Acp,
];
fn without_section(text: &str, heading: &str, next: &str) -> String {
    let start = text
        .find(&format!("\n{heading}:\n"))
        .expect("signed section")
        + 1;
    let end = start
        + text[start..]
            .find(&format!("\n{next}:\n"))
            .expect("next signed section")
        + 1;
    format!("{}{}", &text[..start], &text[end..])
}
pub fn render_builtin(public: &str, adapter: &str, mode: BuiltinPromptMode) -> String {
    let (intro, modes) = adapter
        .split_once("[the line for the current mode is inserted here]\n")
        .expect("owner-approved mode marker");
    let mut tools = without_section(public, "EXECUTION", "ROUTING");
    tools = without_section(&tools, "AGENT", "CHAIN");
    tools = without_section(&tools, "CHAIN", "OUTPUT AND SCOPE");
    tools = tools.replace("Omitted NEW or DIFF is read from stdin. ", "");
    tools = tools.replace("--root DIR selects another repository. ", "");
    tools = tools.replace(" or --value-stdin", "");
    // The harness prepares and tracks its own worktree index. Keep the shared
    // preparation/recovery guidance, but do not advertise a duplicate rebuild.
    tools = tools.replace("INDEX:\n  greppy index [PATH]                rebuild graph and meaning index when reported stale\n  greppy index PATH --agent-worktree\n                                     index the agent worktree belonging to PATH\n\n", "");
    tools = tools
        .lines()
        .filter(|line| !line.starts_with("  greppy web match "))
        .collect::<Vec<_>>()
        .join("\n");
    tools.push('\n');
    if mode == BuiltinPromptMode::Acp {
        // ACP isolation/proposal policy is explicitly still an owner decision.
        // Reuse only the approved role sentence; never make the worktree/PR
        // promise or expose browser commands unavailable in ACP.
        let role = intro.split_once(". ").expect("approved role sentence").0;
        let browser = tools.find("\nBROWSER:\n").expect("signed browser section");
        return format!("{role}.\n\nACP (agent stdio)\n\n{}", &tools[..browser]);
    }
    let prefix = match mode {
        BuiltinPromptMode::OneShot => "-p:",
        BuiltinPromptMode::Interactive => "TUI:",
        BuiltinPromptMode::Serve => "serve:",
        BuiltinPromptMode::Acp => unreachable!(),
    };
    let line = modes
        .lines()
        .find(|line| line.starts_with(prefix))
        .expect("signed mode line");
    format!("{}{line}\n\n{tools}", intro)
}
