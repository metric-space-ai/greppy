//! Text extraction for `web read`, `web search`, and `web research`.
//!
//! Those commands want the document text. Images, fonts, and media are what
//! pushed ordinary pages over the per-navigation network budget and what kept
//! llvmpipe busy on a single core.

/// Subresource destinations a text read does not need.
pub fn blocks_subresource(kind: &str) -> bool {
    matches!(
        kind,
        "image" | "font" | "audio" | "video" | "track" | "embed" | "object" | "manifest"
    )
}

/// Hosts that are ads or trackers, not the article.
pub fn blocks_tracker_host(host: &str) -> bool {
    const MARKERS: &[&str] = &[
        "doubleclick.net",
        "googlesyndication.com",
        "google-analytics.com",
        "googletagmanager.com",
        "googleadservices.com",
        "facebook.net",
        "scorecardresearch.com",
        "adservice.google.",
        "ads.twitter.com",
        "adnxs.com",
        "criteo.com",
        "taboola.com",
        "outbrain.com",
    ];
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    MARKERS
        .iter()
        .any(|marker| host == *marker || host.ends_with(&format!(".{marker}")))
}

pub fn is_text_read(operation: &str) -> bool {
    matches!(operation, "web.read" | "web.search" | "web.research")
}

/// A short read of the worker pipe. `read_exact` reports this when the
/// content process is killed mid-frame.
pub fn incomplete_worker_frame(message: &str) -> bool {
    message.contains("failed to fill whole buffer")
        || message.contains("unexpected end of file")
        || message.contains("UnexpectedEof")
}

pub fn worker_frame_message() -> &'static str {
    "content worker died while reading the page (incomplete protocol frame: failed to fill whole buffer). The read was retried once"
}

/// A navigation that timed out or ran out of budget may still have text.
pub fn navigation_can_salvage(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("timed out")
        || message.contains("timeout")
        || message.contains("network limit")
        || message.contains("cpu time exceeded")
}

pub fn result_has_text(result: &serde_json::Value) -> bool {
    let text = result
        .pointer("/source/text")
        .or_else(|| result.get("text"))
        .and_then(|value| value.as_str())
        .unwrap_or("");
    !text.trim().is_empty()
}

pub fn mark_partial(result: &mut serde_json::Value, reason: &str) {
    let has_source = result.get("source").is_some();
    let target = if has_source {
        result.get_mut("source")
    } else {
        Some(result)
    };
    if let Some(object) = target.and_then(|value| value.as_object_mut()) {
        object.insert("partial".into(), serde_json::json!(true));
        object.insert("partial_reason".into(), serde_json::json!(reason));
    }
}

/// DuckDuckGo HTML wraps results as `/l/?uddg=...`. Keep the target document
/// and drop the search engine's own links.
pub fn research_target_url(href: &str) -> Option<String> {
    let href = href.trim();
    if let Some((_, rest)) = href.split_once("uddg=") {
        let encoded = rest.split(['&', '#']).next().unwrap_or("");
        return http_document(&percent_decode(encoded));
    }
    http_document(href)
}

fn http_document(url: &str) -> Option<String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return None;
    }
    let host = host_of(url);
    if host.is_empty() || is_search_host(host) {
        return None;
    }
    Some(url.to_owned())
}

fn host_of(url: &str) -> &str {
    let rest = url.split("://").nth(1).unwrap_or(url);
    rest.split(['/', '?', '#']).next().unwrap_or("").split('@').next_back().unwrap_or("").split(':').next().unwrap_or("")
}

fn is_search_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "duckduckgo.com"
        || host.ends_with(".duckduckgo.com")
        || host == "html.duckduckgo.com"
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = &input[index + 1..index + 3];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    index += 3;
                } else {
                    out.push(b'%');
                    index += 1;
                }
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(out).unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned())
}

/// A live runtime with no cookie yet belongs to a sibling that is still
/// starting. Minting a second token, or failing immediately, is the race that
/// made one of four parallel session creates return exit 32.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachChoice {
    UseInherited,
    UseCookie,
    WaitForOwner,
    Mint,
}

pub fn choose_attach(inherited: bool, cookie: bool, socket_up: bool) -> AttachChoice {
    if inherited {
        AttachChoice::UseInherited
    } else if cookie {
        AttachChoice::UseCookie
    } else if socket_up {
        AttachChoice::WaitForOwner
    } else {
        AttachChoice::Mint
    }
}

/// Only the process that wins the spawn lock may replace the socket file.
pub fn may_replace_runtime_socket(will_spawn: bool) -> bool {
    will_spawn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_reads_skip_images_fonts_and_trackers() {
        assert!(blocks_subresource("image"));
        assert!(blocks_subresource("font"));
        assert!(blocks_subresource("video"));
        assert!(!blocks_subresource("script"));
        assert!(!blocks_subresource("document"));
        assert!(blocks_tracker_host("www.google-analytics.com"));
        assert!(blocks_tracker_host("ads.twitter.com"));
        assert!(!blocks_tracker_host("en.wikipedia.org"));
    }

    #[test]
    fn research_unwraps_duckduckgo_and_drops_the_serp() {
        let target = research_target_url(
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FBerlin&rut=1",
        );
        assert_eq!(target.as_deref(), Some("https://en.wikipedia.org/wiki/Berlin"));
        assert!(research_target_url("https://duckduckgo.com/html/?q=berlin").is_none());
        assert_eq!(
            research_target_url("https://www.tagesschau.de/inland/x.html").as_deref(),
            Some("https://www.tagesschau.de/inland/x.html")
        );
    }

    #[test]
    fn a_killed_worker_frame_is_recognized_and_a_timeout_can_keep_text() {
        assert!(incomplete_worker_frame(
            "failed to fill whole buffer"
        ));
        assert!(navigation_can_salvage("timed out after 15s waiting for page.goto"));
        assert!(navigation_can_salvage(
            "network limit exceeded for this navigation (10900000 > 8388608)"
        ));
        assert!(!navigation_can_salvage("policy_denied: blocked"));
        let mut result = serde_json::json!({"source": {"text": "Berlin"}});
        assert!(result_has_text(&result));
        mark_partial(&mut result, "timeout");
        assert_eq!(result["source"]["partial"], true);
        assert_eq!(result["source"]["text"], "Berlin");
    }

    #[test]
    fn parallel_session_create_waits_for_the_owner_cookie() {
        assert_eq!(
            choose_attach(false, false, true),
            AttachChoice::WaitForOwner
        );
        assert_eq!(choose_attach(false, true, true), AttachChoice::UseCookie);
        assert_eq!(choose_attach(false, false, false), AttachChoice::Mint);
        assert!(!may_replace_runtime_socket(false));
        assert!(may_replace_runtime_socket(true));
    }
}
