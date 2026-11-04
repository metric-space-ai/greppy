//! Disk snapshot of one inactive tab.
//!
//! What the content worker can actually return is documented on
//! [`PageSnapshot`]. Restoring a tab replays those fields; it does not
//! resume a Servo document, its JavaScript heap, or its history stack.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Cap the outerHTML copy. A larger document is omitted and the URL is the
/// restore path. Holding the HTML in the supervisor would defeat the swap.
pub const MAX_HTML_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageSnapshot {
    pub schema: u32,
    pub url: String,
    pub title: String,
    pub scroll_x: i64,
    pub scroll_y: i64,
    /// `document.cookie` only. HttpOnly cookies are not in this string.
    pub document_cookie: String,
    pub local_storage: Vec<(String, String)>,
    pub session_storage: Vec<(String, String)>,
    /// `document.documentElement.outerHTML`, capped. `None` when omitted.
    pub html: Option<String>,
    pub html_truncated: bool,
    /// Human-readable limits of this capture. Not a machine contract.
    pub notes: Vec<String>,
}

impl PageSnapshot {
    pub fn blank() -> Self {
        Self {
            schema: 1,
            url: String::new(),
            title: String::new(),
            scroll_x: 0,
            scroll_y: 0,
            document_cookie: String::new(),
            local_storage: Vec::new(),
            session_storage: Vec::new(),
            html: None,
            html_truncated: false,
            notes: Vec::new(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(raw: &str) -> Result<Self, String> {
        serde_json::from_str(raw).map_err(|error| format!("tab snapshot: {error}"))
    }
}

pub fn cap_html(html: String) -> (Option<String>, bool) {
    if html.len() <= MAX_HTML_BYTES {
        (Some(html), false)
    } else {
        (None, true)
    }
}

/// `page.cookies` returns the `document.cookie` string. Split it into the
/// name/value pairs `page.addCookies` accepts. Path defaults to `/` because
/// the cookie string does not carry path or domain.
pub fn cookie_pairs(document_cookie: &str) -> Vec<(String, String)> {
    document_cookie
        .split(';')
        .filter_map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return None;
            }
            let (name, value) = part.split_once('=').unwrap_or((part, ""));
            let name = name.trim();
            if name.is_empty() {
                None
            } else {
                Some((name.to_string(), value.trim().to_string()))
            }
        })
        .collect()
}

/// `page.evaluate` serializes a string result as `{"serialized":{"s":"..."}}`.
pub fn parse_evaluate_state(value: &Value) -> Option<EvalState> {
    let serialized = value.get("serialized")?;
    let raw = serialized.get("s").and_then(Value::as_str)?;
    let parsed: Value = serde_json::from_str(raw).ok()?;
    let scroll_x = parsed.get("x").and_then(Value::as_i64).unwrap_or(0);
    let scroll_y = parsed.get("y").and_then(Value::as_i64).unwrap_or(0);
    Some(EvalState {
        scroll_x,
        scroll_y,
        local_storage: string_map(parsed.get("local")),
        session_storage: string_map(parsed.get("session")),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalState {
    pub scroll_x: i64,
    pub scroll_y: i64,
    pub local_storage: Vec<(String, String)>,
    pub session_storage: Vec<(String, String)>,
}

fn string_map(value: Option<&Value>) -> Vec<(String, String)> {
    let Some(object) = value.and_then(Value::as_object) else {
        return Vec::new();
    };
    let mut pairs = Vec::with_capacity(object.len());
    for (key, item) in object {
        if let Some(text) = item.as_str() {
            pairs.push((key.clone(), text.to_string()));
        }
    }
    pairs
}

pub fn capture_script() -> &'static str {
    r#"(function(){
  function dump(store){
    var out = {};
    try {
      for (var i = 0; i < store.length; i++) {
        var key = store.key(i);
        out[key] = String(store.getItem(key));
      }
    } catch (error) {}
    return out;
  }
  return JSON.stringify({
    x: Math.round(window.scrollX || window.pageXOffset || 0),
    y: Math.round(window.scrollY || window.pageYOffset || 0),
    local: dump(window.localStorage),
    session: dump(window.sessionStorage)
  });
})()"#
}

pub fn restore_script(state: &PageSnapshot) -> String {
    let payload = serde_json::json!({
        "x": state.scroll_x,
        "y": state.scroll_y,
        "local": map_from_pairs(&state.local_storage),
        "session": map_from_pairs(&state.session_storage),
    });
    format!(
        r#"(function(state){{
  function fill(store, values){{
    try {{
      Object.keys(values).forEach(function(key){{ store.setItem(key, values[key]); }});
    }} catch (error) {{}}
  }}
  fill(window.localStorage, state.local || {{}});
  fill(window.sessionStorage, state.session || {{}});
  window.scrollTo(state.x || 0, state.y || 0);
  return true;
}})({})"#,
        payload
    )
}

fn map_from_pairs(pairs: &[(String, String)]) -> serde_json::Map<String, Value> {
    pairs
        .iter()
        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
        .collect()
}

/// Navigating is the faithful restore. `setContent` only covers a document
/// that has no http(s) URL, because writing outerHTML drops scripts' runtime
/// state and is not the page the network would have produced.
pub fn restore_via_navigation(url: &str) -> bool {
    let url = url.trim();
    url.starts_with("http://") || url.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookie_string_splits_without_inventing_a_domain() {
        let pairs = cookie_pairs("a=b; c=d e; =missing; bare");
        assert_eq!(
            pairs,
            vec![
                ("a".into(), "b".into()),
                ("c".into(), "d e".into()),
                ("bare".into(), "".into()),
            ]
        );
    }

    #[test]
    fn evaluate_payload_reads_scroll_and_storage() {
        let value = serde_json::json!({
            "serialized": {
                "s": "{\"x\":12,\"y\":40,\"local\":{\"k\":\"v\"},\"session\":{}}"
            }
        });
        let state = parse_evaluate_state(&value).unwrap();
        assert_eq!(state.scroll_x, 12);
        assert_eq!(state.scroll_y, 40);
        assert_eq!(state.local_storage, vec![("k".into(), "v".into())]);
        assert!(state.session_storage.is_empty());
    }

    #[test]
    fn html_over_the_cap_is_omitted() {
        let html = "x".repeat(MAX_HTML_BYTES + 1);
        let (kept, truncated) = cap_html(html);
        assert!(kept.is_none());
        assert!(truncated);
    }

    #[test]
    fn snapshot_roundtrip_keeps_the_url() {
        let mut snapshot = PageSnapshot::blank();
        snapshot.url = "https://example.test/path".into();
        snapshot.document_cookie = "a=b".into();
        let restored = PageSnapshot::from_json(&snapshot.to_json()).unwrap();
        assert_eq!(restored.url, snapshot.url);
        assert!(restore_via_navigation(&restored.url));
        assert!(!restore_via_navigation("about:blank"));
    }
}
