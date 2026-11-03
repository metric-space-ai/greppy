//! Host RAM budget for headless machines.
//!
//! The browser is reserved first. Embedding and summary models are loaded
//! only from what remains, or not at all. Unset means the product default:
//! GPU `auto`, both models, no global web cap. See `docs/max-ram.md`.

use std::fmt::Write as _;

pub const ENV_MAX_RAM: &str = "GREPPY_MAX_RAM";

/// One content-process page, from the comment on
/// `SessionLimits::content_rss_bytes` (a mirrored news page at 1.25 GB).
/// That figure is the process RSS, so the ~400 MiB runtime image is not added
/// again. It is a planning reserve, not a new measurement.
pub const BROWSER_RESERVE_BYTES: u64 = 1280 * 1024 * 1024;

/// Room for the greppy process and the OS page cache. Not measured.
pub const PROCESS_SLACK_BYTES: u64 = 256 * 1024 * 1024;

/// `estimated_gpu_memory` workspace for EmbeddingGemma. The Linux CUDA
/// footprint file recorded 702_910_464 host RSS with weights on the GPU;
/// the CPU loader copies quantized weights into the process, so the plan
/// uses this larger constant.
pub const EMBED_HOST_BUDGET_BYTES: u64 = 768 * 1024 * 1024;

/// `estimated_gpu_memory` workspace for Qwen3.5. The same Linux file recorded
/// 541_036_544 host RSS on CUDA. CPU residency is unmeasured; the plan uses
/// the workspace constant.
pub const SUMMARY_HOST_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSlot {
    Off,
    Lazy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Below 2 GiB, including a 1 GiB machine.
    Below2GiB,
    /// At least 2 GiB and below 3 GiB.
    At2GiB,
    /// At least 3 GiB and below 4 GiB.
    At3GiB,
    /// At least 4 GiB.
    AtLeast4GiB,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebLimits {
    pub max_live_pages: u32,
    pub max_sessions: u32,
    /// Inactive tabs are closed in the content process and restored later.
    pub swap_inactive_tabs: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamBudget {
    pub bytes: u64,
    pub stage: Stage,
    pub embeddings: ModelSlot,
    pub summaries: ModelSlot,
    pub embed_model_ttl_s: u64,
    pub embed_exit_ttl_s: u64,
    pub summary_model_ttl_s: u64,
    pub summary_exit_ttl_s: u64,
    /// Do not start the summary daemon until a command actually needs a sentence.
    pub suppress_summary_prewarm: bool,
    /// Select CPU when the caller did not name a device.
    pub cpu_if_unspecified: bool,
    pub web: WebLimits,
}

impl RamBudget {
    /// `--max-ram` wins over the environment. `Ok(None)` is the product default.
    pub fn resolve(cli: Option<&str>) -> Result<Option<Self>, String> {
        if let Some(raw) = cli.map(str::trim).filter(|raw| !raw.is_empty()) {
            return Ok(Some(Self::parse(raw)?));
        }
        match std::env::var(ENV_MAX_RAM) {
            Err(_) => Ok(None),
            Ok(raw) if raw.trim().is_empty() => Ok(None),
            Ok(raw) => Ok(Some(Self::parse(&raw)?)),
        }
    }

    /// Daemon and library paths. A broken value is reported and ignored so a
    /// bad variable cannot take the web supervisor down.
    pub fn from_env() -> Option<Self> {
        let Ok(raw) = std::env::var(ENV_MAX_RAM) else {
            return None;
        };
        if raw.trim().is_empty() {
            return None;
        }
        match Self::parse(&raw) {
            Ok(budget) => Some(budget),
            Err(error) => {
                eprintln!("greppy: ignoring {ENV_MAX_RAM} ({error})");
                None
            }
        }
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        let bytes = parse_quantity(raw)?;
        Ok(Self::from_bytes(bytes))
    }

    pub fn from_bytes(bytes: u64) -> Self {
        let stage = if bytes < 2 * GIB {
            Stage::Below2GiB
        } else if bytes < 3 * GIB {
            Stage::At2GiB
        } else if bytes < 4 * GIB {
            Stage::At3GiB
        } else {
            Stage::AtLeast4GiB
        };
        let remainder = bytes
            .saturating_sub(BROWSER_RESERVE_BYTES)
            .saturating_sub(PROCESS_SLACK_BYTES);
        let embeddings = if remainder >= EMBED_HOST_BUDGET_BYTES {
            ModelSlot::Lazy
        } else {
            ModelSlot::Off
        };
        let summaries = if embeddings == ModelSlot::Lazy
            && remainder >= EMBED_HOST_BUDGET_BYTES.saturating_add(SUMMARY_HOST_BUDGET_BYTES)
        {
            ModelSlot::Lazy
        } else {
            ModelSlot::Off
        };
        // A second live page costs another browser reserve. It is allowed only
        // when that page still leaves the enabled models inside the machine.
        let model_budget = match (embeddings, summaries) {
            (ModelSlot::Lazy, ModelSlot::Lazy) => {
                EMBED_HOST_BUDGET_BYTES.saturating_add(SUMMARY_HOST_BUDGET_BYTES)
            }
            (ModelSlot::Lazy, ModelSlot::Off) => EMBED_HOST_BUDGET_BYTES,
            _ => 0,
        };
        let max_live_pages = if remainder.saturating_sub(model_budget) >= BROWSER_RESERVE_BYTES {
            2
        } else {
            1
        };
        // Each session builds its own Servo bundle inside the one content
        // process. Keep that at one until the 4 GiB stage, which has spare
        // after both model budgets.
        let max_sessions = if matches!(stage, Stage::AtLeast4GiB) {
            2
        } else {
            1
        };
        let (embed_model_ttl_s, embed_exit_ttl_s, summary_model_ttl_s, summary_exit_ttl_s) =
            if summaries == ModelSlot::Lazy {
                (30, 120, 30, 120)
            } else {
                (60, 180, 60, 180)
            };
        Self {
            bytes,
            stage,
            embeddings,
            summaries,
            embed_model_ttl_s,
            embed_exit_ttl_s,
            summary_model_ttl_s,
            summary_exit_ttl_s,
            suppress_summary_prewarm: true,
            cpu_if_unspecified: embeddings == ModelSlot::Lazy || summaries == ModelSlot::Lazy,
            web: WebLimits {
                max_live_pages,
                max_sessions,
                swap_inactive_tabs: true,
            },
        }
    }

    pub fn embeddings_off(&self) -> bool {
        self.embeddings == ModelSlot::Off
    }

    pub fn summaries_off(&self) -> bool {
        self.summaries == ModelSlot::Off
    }

    /// Publish the canonical byte count and the daemon idle windows.
    pub fn install_process_env(&self) {
        std::env::set_var(ENV_MAX_RAM, self.bytes.to_string());
        if self.embeddings == ModelSlot::Lazy {
            std::env::set_var(
                "GREPPY_EMBED_DAEMON_MODEL_TTL_S",
                self.embed_model_ttl_s.to_string(),
            );
            std::env::set_var(
                "GREPPY_EMBED_DAEMON_EXIT_TTL_S",
                self.embed_exit_ttl_s.to_string(),
            );
        }
        if self.summaries == ModelSlot::Lazy {
            std::env::set_var(
                "GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S",
                self.summary_model_ttl_s.to_string(),
            );
            std::env::set_var(
                "GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S",
                self.summary_exit_ttl_s.to_string(),
            );
        }
    }

    pub fn announce_line(&self) -> String {
        format!(
            "greppy: RAM budget {} — embeddings {}, summaries {}, web live pages {}, sessions {}{}",
            format_bytes(self.bytes),
            slot_word(self.embeddings),
            slot_word(self.summaries),
            self.web.max_live_pages,
            self.web.max_sessions,
            if self.cpu_if_unspecified {
                ", inference device cpu unless one was named"
            } else {
                ""
            }
        )
    }
}

fn slot_word(slot: ModelSlot) -> &'static str {
    match slot {
        ModelSlot::Off => "off",
        ModelSlot::Lazy => "lazy",
    }
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes % GIB == 0 {
        format!("{}GiB", bytes / GIB)
    } else {
        format!("{bytes}B")
    }
}

pub fn parse_quantity(raw: &str) -> Result<u64, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("RAM budget is empty; use for example 2g or 2048m".into());
    }
    let split = raw
        .char_indices()
        .find(|(_, ch)| !ch.is_ascii_digit())
        .map(|(index, _)| index)
        .unwrap_or(raw.len());
    let (number, suffix) = raw.split_at(split);
    if number.is_empty() {
        return Err(format!(
            "RAM budget `{raw}` has no number; use 1g, 2g, 3g, 4g, 1536m, or a byte count"
        ));
    }
    let quantity: u64 = number
        .parse()
        .map_err(|_| format!("RAM budget `{raw}` is not an integer quantity"))?;
    if quantity == 0 {
        return Err("RAM budget must be at least 1 byte".into());
    }
    let suffix = suffix.trim().to_ascii_lowercase();
    let multiplier: u64 = match suffix.as_str() {
        "" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => GIB,
        "t" | "tb" | "tib" => 1024 * GIB,
        _ => {
            return Err(format!(
                "RAM budget `{raw}` has an unknown unit; use k, m, or g (1024-based)"
            ));
        }
    };
    quantity
        .checked_mul(multiplier)
        .ok_or_else(|| format!("RAM budget `{raw}` overflows"))
}

pub fn remainder_bytes(bytes: u64) -> u64 {
    bytes
        .saturating_sub(BROWSER_RESERVE_BYTES)
        .saturating_sub(PROCESS_SLACK_BYTES)
}

/// One-line description used by tests and the design table.
pub fn decision_summary(bytes: u64) -> String {
    let budget = RamBudget::from_bytes(bytes);
    let mut line = String::new();
    let _ = write!(
        line,
        "remainder={} embed={:?} summary={:?} pages={} sessions={}",
        remainder_bytes(bytes),
        budget.embeddings,
        budget.summaries,
        budget.web.max_live_pages,
        budget.web.max_sessions
    );
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantities_are_1024_based() {
        assert_eq!(parse_quantity("2g").unwrap(), 2 * GIB);
        assert_eq!(parse_quantity("2G").unwrap(), 2 * GIB);
        assert_eq!(parse_quantity("2gb").unwrap(), 2 * GIB);
        assert_eq!(parse_quantity("1536m").unwrap(), 1536 * 1024 * 1024);
        assert_eq!(parse_quantity("2048").unwrap(), 2048);
    }

    #[test]
    fn quantities_reject_zero_and_junk() {
        assert!(parse_quantity("0").is_err());
        assert!(parse_quantity("0g").is_err());
        assert!(parse_quantity("1.5g").is_err());
        assert!(parse_quantity("many").is_err());
        assert!(parse_quantity("").is_err());
    }

    #[test]
    fn one_and_two_gib_keep_the_browser_and_drop_models() {
        for label in ["1g", "2g", "1536m"] {
            let budget = RamBudget::parse(label).unwrap();
            assert_eq!(budget.embeddings, ModelSlot::Off, "{label}");
            assert_eq!(budget.summaries, ModelSlot::Off, "{label}");
            assert!(!budget.cpu_if_unspecified, "{label}");
            assert_eq!(budget.web.max_live_pages, 1, "{label}");
            assert_eq!(budget.web.max_sessions, 1, "{label}");
            assert!(budget.web.swap_inactive_tabs, "{label}");
            assert!(budget.suppress_summary_prewarm, "{label}");
        }
        assert_eq!(RamBudget::parse("1g").unwrap().stage, Stage::Below2GiB);
        assert_eq!(RamBudget::parse("2g").unwrap().stage, Stage::At2GiB);
        assert_eq!(RamBudget::parse("1536m").unwrap().stage, Stage::Below2GiB);
    }

    #[test]
    fn three_gib_keeps_lazy_embeddings_only() {
        let budget = RamBudget::parse("3g").unwrap();
        assert_eq!(budget.stage, Stage::At3GiB);
        assert_eq!(budget.embeddings, ModelSlot::Lazy);
        assert_eq!(budget.summaries, ModelSlot::Off);
        assert!(budget.cpu_if_unspecified);
        assert_eq!(budget.embed_model_ttl_s, 60);
        assert_eq!(budget.embed_exit_ttl_s, 180);
        assert_eq!(budget.web.max_live_pages, 1);
        assert_eq!(budget.web.max_sessions, 1);
    }

    #[test]
    fn four_gib_keeps_both_models_lazy_and_one_live_page() {
        let budget = RamBudget::parse("4g").unwrap();
        assert_eq!(budget.stage, Stage::AtLeast4GiB);
        assert_eq!(budget.embeddings, ModelSlot::Lazy);
        assert_eq!(budget.summaries, ModelSlot::Lazy);
        assert!(budget.cpu_if_unspecified);
        assert_eq!(budget.embed_model_ttl_s, 30);
        assert_eq!(budget.summary_model_ttl_s, 30);
        assert_eq!(budget.embed_exit_ttl_s, 120);
        assert_eq!(budget.web.max_live_pages, 1);
        assert_eq!(budget.web.max_sessions, 2);
        assert!(budget.suppress_summary_prewarm);
    }

    #[test]
    fn five_gib_stays_on_the_four_gib_model_policy() {
        let budget = RamBudget::parse("5g").unwrap();
        assert_eq!(budget.stage, Stage::AtLeast4GiB);
        assert_eq!(budget.embeddings, ModelSlot::Lazy);
        assert_eq!(budget.summaries, ModelSlot::Lazy);
    }

    #[test]
    fn flag_wins_over_the_environment() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var(ENV_MAX_RAM, "4g");
        let budget = RamBudget::resolve(Some("1g")).unwrap().unwrap();
        std::env::remove_var(ENV_MAX_RAM);
        assert_eq!(budget.bytes, GIB);
        assert!(budget.embeddings_off());
    }

    #[test]
    fn install_writes_canonical_bytes_and_ttls() {
        let _lock = ENV_LOCK.lock().unwrap();
        let budget = RamBudget::parse("4g").unwrap();
        budget.install_process_env();
        assert_eq!(std::env::var(ENV_MAX_RAM).unwrap(), (4 * GIB).to_string());
        assert_eq!(
            std::env::var("GREPPY_EMBED_DAEMON_MODEL_TTL_S").unwrap(),
            "30"
        );
        assert_eq!(
            std::env::var("GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S").unwrap(),
            "120"
        );
        std::env::remove_var(ENV_MAX_RAM);
        std::env::remove_var("GREPPY_EMBED_DAEMON_MODEL_TTL_S");
        std::env::remove_var("GREPPY_EMBED_DAEMON_EXIT_TTL_S");
        std::env::remove_var("GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S");
        std::env::remove_var("GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S");
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
