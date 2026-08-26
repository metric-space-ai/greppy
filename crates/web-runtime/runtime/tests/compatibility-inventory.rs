//! Completeness gate for `contracts/web-runtime/compatibility.v1.json`.
//! Entries may gain evidence in place; they must never be deleted.

use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;

const MIN_ENTRIES: usize = 1354;
const SYMBOL_SHA256: &str = "bf21f67dcffc86d0a5f97a559ecdad2dec5d1220fd02e94f791cd88c4b5e71c9";

fn inventory_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/web-runtime/compatibility.v1.json")
}

fn load() -> Value {
    let bytes = std::fs::read(inventory_path()).expect("compatibility inventory");
    serde_json::from_slice(&bytes).expect("inventory json")
}

fn sha256_hex(bytes: &[u8]) -> String {
    web_runtime::artifacts::hex_sha256(bytes)
}

#[test]
fn inventory_keeps_every_phase0_symbol() {
    let data = load();
    let entries = data["entries"].as_array().expect("entries");
    assert_eq!(
        data["entry_count"].as_u64().unwrap() as usize,
        entries.len()
    );
    assert!(
        entries.len() >= MIN_ENTRIES,
        "inventory shrank: {} < {MIN_ENTRIES}",
        entries.len()
    );

    let mut symbols = Vec::new();
    let mut seen = HashSet::new();
    for entry in entries {
        let symbol = entry["symbol"].as_str().expect("symbol");
        assert!(
            seen.insert(symbol.to_owned()),
            "duplicate inventory symbol {symbol}"
        );
        symbols.push(symbol.to_owned());
        for field in ["schema", "source", "behavior"] {
            let value = entry[field].as_str().unwrap_or("");
            assert_ne!(value, "unknown", "{symbol} {field} is unknown");
            assert!(
                matches!(
                    value,
                    "unverified" | "unsupported" | "implemented" | "passing"
                ),
                "{symbol} {field} has illegal status {value}"
            );
        }
    }
    symbols.sort();
    let digest = sha256_hex(symbols.join("\n").as_bytes());
    assert_eq!(
        digest, SYMBOL_SHA256,
        "inventory symbol set changed; entries must be updated in place, not replaced"
    );
}

#[test]
fn implemented_playwright_surface_is_inventoried() {
    let data = load();
    let entries = data["entries"].as_array().expect("entries");
    let by_symbol: HashSet<_> = entries
        .iter()
        .filter_map(|e| e["symbol"].as_str())
        .collect();
    let required = [
        "chromium.launch",
        "Browser.newContext",
        "Browser.close",
        "BrowserContext.newPage",
        "Page.goto",
        "Page.evaluate",
        "Page.setContent",
        "Page.getByRole",
        "Page.getByLabel",
        "Page.getByText",
        "Page.locator",
        "Locator.click",
        "Locator.fill",
        "Locator.hover",
        "Locator.innerText",
        "Locator.count",
        "Locator.isVisible",
        "Locator.waitFor",
        "Locator.first",
        "firefox.launch",
        "webkit.launch",
    ];
    for symbol in required {
        assert!(by_symbol.contains(symbol), "missing inventory row {symbol}");
    }

    let implemented: Vec<_> = entries
        .iter()
        .filter(|e| e["schema"] == "implemented")
        .map(|e| e["symbol"].as_str().unwrap())
        .collect();
    assert!(
        implemented.contains(&"chromium.launch"),
        "chromium.launch should be marked implemented: {implemented:?}"
    );

    let firefox = entries
        .iter()
        .find(|e| e["symbol"] == "firefox.launch")
        .unwrap();
    assert_eq!(firefox["schema"], "unsupported");
    let webkit = entries
        .iter()
        .find(|e| e["symbol"] == "webkit.launch")
        .unwrap();
    assert_eq!(webkit["schema"], "unsupported");
}

#[test]
fn playwright_module_methods_are_not_silent_omissions() {
    let source = include_str!("../js/playwright.mjs");
    let data = load();
    let by_symbol: HashSet<_> = data["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e["symbol"].as_str().map(str::to_owned))
        .collect();

    let mut claimed = Vec::new();
    for (prefix, class) in [("Page", "class Page"), ("Locator", "class Locator")] {
        let start = source.find(class).expect(class);
        let rest = &source[start..];
        let end = rest[1..]
            .find("\nclass ")
            .map(|i| i + 1)
            .unwrap_or(rest.len());
        let body = &rest[..end];
        for line in body.lines() {
            let line = line.trim();
            let name = if let Some(rest) = line.strip_prefix("async ") {
                rest.split('(').next()
            } else if line.starts_with("getBy") || line.starts_with("locator(") {
                line.split('(').next()
            } else if line.ends_with("() {") && !line.contains(' ') {
                line.strip_suffix("() {")
            } else {
                None
            };
            if let Some(name) = name {
                if name.starts_with('_') || name == "constructor" {
                    continue;
                }
                claimed.push(format!("{prefix}.{name}"));
            }
        }
    }
    claimed.push("chromium.launch".into());
    claimed.push("firefox.launch".into());
    claimed.push("webkit.launch".into());
    claimed.push("Browser.newContext".into());
    claimed.push("Browser.close".into());
    claimed.push("BrowserContext.newPage".into());
    claimed.push("Locator.first".into());
    claimed.push("Locator.nth".into());
    claimed.push("Page.getByRole".into());
    claimed.push("Page.getByLabel".into());
    claimed.push("Page.getByText".into());
    claimed.push("Page.locator".into());

    let missing: Vec<_> = claimed
        .iter()
        .filter(|symbol| !by_symbol.contains(symbol.as_str()))
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "playwright.mjs claims APIs with no inventory row: {missing:?}"
    );
}
