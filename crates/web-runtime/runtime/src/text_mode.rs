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
    degrade_reason(message).is_some()
}

/// Unreachable hosts and sandbox or policy violations stay errors.
pub fn hard_failure(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("policy_denied")
        || message.contains("sandbox")
        || message.contains("not permitted")
        || message.contains("unreachable")
        || message.contains("connection refused")
        || message.contains("failed to connect")
        || message.contains("no route to host")
        || message.contains("name or service not known")
        || message.contains("certificate")
}

/// Budget and transport caps become a successful, degraded read.
pub fn degrade_reason(message: &str) -> Option<&'static str> {
    if hard_failure(message) {
        return None;
    }
    let message = message.to_ascii_lowercase();
    if message.contains("network limit") || message.contains("network budget") {
        Some("network_budget")
    } else if message.contains("cpu time") || message.contains("timed out") || message.contains("timeout") {
        Some("content_cpu")
    } else if message.contains("frame") || message.contains("too large") || message.contains("max_frame") {
        Some("frame")
    } else if message.contains("limit exceeded") || message.contains("resource_limit") {
        Some("session_limit")
    } else {
        None
    }
}

/// Drop a subresource. The main document is never dropped.
///
/// Text reads skip images, media, fonts, and trackers immediately. After the
/// network budget is spent, every later side resource (including scripts) is
/// dropped so the document text can still be returned.
pub fn cancel_subresource(
    main_frame: bool,
    text_only: bool,
    budget_exhausted: bool,
    kind: &str,
    host: &str,
) -> bool {
    if main_frame {
        return false;
    }
    if budget_exhausted {
        return true;
    }
    text_only && (blocks_subresource(kind) || blocks_tracker_host(host))
}

/// Keep response text inside the 8 MiB IPC frame.
pub const RESPONSE_TEXT_CAP: usize = 6 * 1024 * 1024;

pub fn cap_text_for_frame(text: &str) -> (&str, bool) {
    if text.len() <= RESPONSE_TEXT_CAP {
        return (text, false);
    }
    let mut end = RESPONSE_TEXT_CAP;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (&text[..end], true)
}

/// What a read returns after a navigation. Exit 0 unless the failure is real.
pub struct DocumentRead {
    pub text: String,
    pub degraded: bool,
    pub reason: &'static str,
    pub exit_code: u8,
}

pub fn finish_document_read(
    text: &str,
    side_bytes: u64,
    budget: u64,
    engine_error: Option<&str>,
) -> DocumentRead {
    if let Some(message) = engine_error {
        if hard_failure(message) {
            return DocumentRead {
                text: String::new(),
                degraded: false,
                reason: "",
                exit_code: 34,
            };
        }
    }
    let (fitted, framed) = cap_text_for_frame(text);
    let mut reason = "";
    let mut degraded = false;
    if side_bytes > budget && budget != u64::MAX {
        degraded = true;
        reason = "network_budget";
    }
    if let Some(message) = engine_error {
        if let Some(found) = degrade_reason(message) {
            degraded = true;
            reason = found;
        }
    }
    if framed {
        degraded = true;
        if reason.is_empty() {
            reason = "frame";
        }
    }
    DocumentRead {
        text: fitted.to_owned(),
        degraded,
        reason,
        exit_code: 0,
    }
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
        let code = degrade_reason(reason).unwrap_or("session_limit");
        object.insert("degraded".into(), serde_json::json!(true));
        object.insert("reason".into(), serde_json::json!(code));
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
        assert_eq!(result["source"]["degraded"], true);
        assert_eq!(result["source"]["reason"], "content_cpu");
        assert_eq!(result["source"]["text"], "Berlin");
    }

    #[test]
    fn twenty_megabytes_of_side_resources_keep_the_document() {
        let document = "Kleinanzeigen Angebot bleibt lesbar";
        let side = 21 * 1024 * 1024;
        assert!(side > 20 * 1024 * 1024);
        assert!(cancel_subresource(false, true, false, "image", "cdn.example"));
        assert!(cancel_subresource(false, true, false, "font", "fonts.example"));
        assert!(cancel_subresource(false, true, false, "video", "media.example"));
        assert!(cancel_subresource(false, false, true, "script", "example.com"));
        assert!(!cancel_subresource(true, true, true, "document", "example.com"));
        let tight = finish_document_read(document, side, 8 * 1024 * 1024, None);
        let wide = finish_document_read(document, side, 64 * 1024 * 1024, None);
        assert_eq!(tight.exit_code, 0);
        assert_eq!(wide.exit_code, 0);
        assert_eq!(tight.text, document);
        assert_eq!(wide.text, tight.text);
        let refused = finish_document_read(document, side, 64 * 1024 * 1024, Some("connection refused"));
        assert_eq!(refused.exit_code, 34);
        assert!(!refused.degraded);
        let timed = finish_document_read(document, 0, 64 * 1024 * 1024, Some("timed out after 30s"));
        assert_eq!(timed.exit_code, 0);
        assert_eq!(timed.reason, "content_cpu");
        assert_eq!(timed.text, document);
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
