//! Greppy-owned desktop commands. No upstream types.

/// Stable command names advertised by `greppy desktop` and the desktop manual.
pub const COMMANDS: &[&str] = &[
    "status",
    "permissions",
    "list-apps",
    "list-windows",
    "launch",
    "focus-window",
    "snapshot",
    "find",
    "get",
    "click",
    "type",
    "set-value",
    "press",
    "scroll",
    "wait",
    "screenshot",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Status,
    Permissions {
        request: bool,
    },
    ListApps {
        app: Option<String>,
    },
    ListWindows {
        app: Option<String>,
    },
    Launch {
        app: String,
    },
    FocusWindow {
        window_id: Option<String>,
        app: Option<String>,
        title: Option<String>,
    },
    Snapshot {
        app: Option<String>,
        interactive_only: bool,
    },
    Find {
        query: String,
    },
    Get {
        ref_id: String,
        property: String,
    },
    Click {
        ref_id: String,
    },
    Type {
        ref_id: String,
        text: String,
    },
    SetValue {
        ref_id: String,
        value: String,
    },
    Press {
        combo: String,
    },
    Scroll {
        ref_id: String,
        direction: String,
        amount: u32,
    },
    Wait {
        milliseconds: u64,
    },
    Screenshot {
        app: Option<String>,
        window_id: Option<String>,
    },
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Permissions { .. } => "permissions",
            Self::ListApps { .. } => "list-apps",
            Self::ListWindows { .. } => "list-windows",
            Self::Launch { .. } => "launch",
            Self::FocusWindow { .. } => "focus-window",
            Self::Snapshot { .. } => "snapshot",
            Self::Find { .. } => "find",
            Self::Get { .. } => "get",
            Self::Click { .. } => "click",
            Self::Type { .. } => "type",
            Self::SetValue { .. } => "set-value",
            Self::Press { .. } => "press",
            Self::Scroll { .. } => "scroll",
            Self::Wait { .. } => "wait",
            Self::Screenshot { .. } => "screenshot",
        }
    }
}

/// Canned desktop session for tests. The real adapter is not contacted.
#[derive(Clone, Debug)]
pub struct FakeDesktop {
    pub apps: Vec<FakeApp>,
    pub windows: Vec<FakeWindow>,
    pub accessibility_granted: bool,
    pub png: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct FakeApp {
    pub name: String,
    pub pid: u32,
    pub bundle_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FakeWindow {
    pub id: String,
    pub title: String,
    pub app: String,
    pub pid: u32,
}

impl Default for FakeDesktop {
    fn default() -> Self {
        Self::fixture()
    }
}

impl FakeDesktop {
    pub fn fixture() -> Self {
        Self {
            apps: vec![FakeApp {
                name: "Editor".into(),
                pid: 4242,
                bundle_id: Some("app.editor".into()),
            }],
            windows: vec![FakeWindow {
                id: "w1".into(),
                title: "Notes".into(),
                app: "Editor".into(),
                pid: 4242,
            }],
            accessibility_granted: true,
            png: ONE_PIXEL_PNG.to_vec(),
        }
    }
}

pub const ONE_PIXEL_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 218, 99, 252, 207, 192, 80, 15, 0, 4,
    133, 1, 128, 132, 169, 140, 33, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
