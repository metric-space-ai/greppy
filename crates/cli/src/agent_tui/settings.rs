//! Persistent interactive-agent preferences.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentSettings {
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub language: String,
    pub private_store: bool,
    pub no_sandbox: bool,
    pub skip_selfcheck: bool,
    pub acceleration: String,
    pub workspace_backend: String,
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            endpoint: None,
            model: None,
            language: "English".into(),
            private_store: false,
            no_sandbox: false,
            skip_selfcheck: false,
            acceleration: "auto".into(),
            workspace_backend: "auto".into(),
        }
    }
}

impl AgentSettings {
    pub fn load() -> Self {
        let path = settings_path();
        std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> io::Result<()> {
        save_to(&settings_path(), self)
    }
}

pub fn settings_path() -> PathBuf {
    if let Some(path) = std::env::var_os("GREPPY_CONFIG_DIR") {
        return PathBuf::from(path).join("agent-settings.json");
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("greppy")
            .join("agent-settings.json");
    }
    #[cfg(all(target_os = "linux", not(target_os = "android")))]
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    {
        return base.join("greppy").join("agent-settings.json");
    }
    std::env::temp_dir()
        .join("greppy")
        .join("agent-settings.json")
}

fn save_to(path: &Path, settings: &AgentSettings) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let bytes = serde_json::to_vec_pretty(settings)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    std::fs::write(&temp, bytes)?;
    std::fs::rename(temp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("agent-settings.json");
        let settings = AgentSettings {
            endpoint: Some("http://127.0.0.1:18318".into()),
            private_store: true,
            acceleration: "cpu".into(),
            ..AgentSettings::default()
        };
        save_to(&path, &settings).unwrap();
        let restored: AgentSettings =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(restored, settings);
    }
}
