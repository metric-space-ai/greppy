//! Persistent interactive-agent preferences.

use std::collections::BTreeMap;
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
    /// Context limits per exact model id. Read by `greppy -p` as well.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelLimits>,
}

/// Context limits of one model, in tokens of that model's tokenizer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelLimits {
    /// The model's real context window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Prompt size that triggers a context checkpoint; defaults to 80 % of
    /// `context_window` and never exceeds it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_at_tokens: Option<u64>,
}

impl ModelLimits {
    pub fn compaction_threshold(&self) -> Option<u64> {
        let at = self
            .compact_at_tokens
            .or_else(|| self.context_window.map(|window| window / 5 * 4))?;
        let at = self.context_window.map_or(at, |window| at.min(window));
        (at > 0).then_some(at)
    }
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
            models: BTreeMap::new(),
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

    /// Compaction thresholds in tokens for every model with configured limits.
    pub fn compaction_thresholds(&self) -> BTreeMap<String, u64> {
        self.models
            .iter()
            .filter_map(|(model, limits)| Some((model.clone(), limits.compaction_threshold()?)))
            .collect()
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

    #[test]
    fn model_limits_parse_and_resolve_thresholds() {
        let settings: AgentSettings = serde_json::from_str(
            r#"{"model":"m","models":{
                "DeepSeek-V4.1-Flash\uff20gpu4-a4500":{"context_window":1000000,"compact_at_tokens":800000},
                "window-only":{"context_window":200000},
                "above-window":{"context_window":100000,"compact_at_tokens":150000},
                "threshold-only":{"compact_at_tokens":64000},
                "empty":{}}}"#,
        )
        .unwrap();
        let thresholds = settings.compaction_thresholds();
        assert_eq!(thresholds["DeepSeek-V4.1-Flash\u{ff20}gpu4-a4500"], 800_000);
        assert_eq!(thresholds["window-only"], 160_000);
        assert_eq!(thresholds["above-window"], 100_000);
        assert_eq!(thresholds["threshold-only"], 64_000);
        assert!(!thresholds.contains_key("empty"));
        // Settings written before model limits existed still load.
        let legacy: AgentSettings = serde_json::from_str(r#"{"model":"m"}"#).unwrap();
        assert!(legacy.models.is_empty());
    }
}
