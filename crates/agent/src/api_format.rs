//! Which chat protocol a gateway speaks.
//!
//! `auto` keeps today's Anthropic Messages default. A path selects a format
//! only when it is exactly the Anthropic or OpenAI completion route; every
//! other URL, including the historical `http://127.0.0.1:8317` gateway, stays
//! on Anthropic so an existing endpoint does not change behavior.

use crate::client::normalize_base_url;

/// Wire protocol used for one model turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApiFormat {
    /// Anthropic Messages: `POST /v1/messages`.
    #[default]
    Anthropic,
    /// OpenAI Chat Completions: `POST /v1/chat/completions`.
    OpenAi,
}

/// Operator choice. `Auto` applies the path heuristic; `Explicit` wins even
/// when the URL already names the other protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApiFormatSetting {
    #[default]
    Auto,
    Explicit(ApiFormat),
}

impl ApiFormatSetting {
    /// Parse `GREPPY_API_FORMAT` / `--api-format`.
    ///
    /// Accepts `auto`, `anthropic`, and `openai` only. Unknown values are an
    /// error so a typo cannot silently fall back to the other protocol.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "anthropic" => Ok(Self::Explicit(ApiFormat::Anthropic)),
            "openai" => Ok(Self::Explicit(ApiFormat::OpenAi)),
            other => Err(format!(
                "unknown API format {other:?}; expected auto, anthropic, or openai"
            )),
        }
    }
}

/// URLs for one gateway base and format choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEndpoint {
    pub format: ApiFormat,
    /// Scheme, host, port, and any prefix that is not a known API suffix.
    pub origin: String,
    pub completion_url: String,
    pub models_url: String,
}

/// Resolve the completion and model-list URLs.
///
/// A trailing `/v1`, `/v1/messages`, or `/v1/chat/completions` is the API
/// suffix, not part of the origin. `http://127.0.0.1:11434/v1` therefore
/// lists models at `/v1/models` rather than `/v1/v1/models`.
pub fn resolve_endpoint(base: &str, setting: ApiFormatSetting) -> ResolvedEndpoint {
    let base = normalize_base_url(base);
    let (without_query, _query) = split_query(&base);
    let hint = path_format_hint(without_query);
    let format = match setting {
        ApiFormatSetting::Explicit(format) => format,
        ApiFormatSetting::Auto => hint.unwrap_or(ApiFormat::Anthropic),
    };
    let origin = strip_api_suffix(without_query);
    let completion_path = match format {
        ApiFormat::Anthropic => "/v1/messages",
        ApiFormat::OpenAi => "/v1/chat/completions",
    };
    ResolvedEndpoint {
        format,
        completion_url: format!("{origin}{completion_path}"),
        models_url: format!("{origin}/v1/models"),
        origin,
    }
}

/// Plain-HTTP refusal, when the URL is not `http://`.
///
/// The agent client is built without a TLS stack. Callers surface this before
/// opening a socket so an `https://` provider fails once, with a reason, and
/// does not look like a connect timeout.
pub fn plain_http_refusal(url: &str) -> Option<String> {
    let url = url.trim();
    if url.starts_with("http://") {
        None
    } else if url.starts_with("https://") {
        Some(
            "plain HTTP only (no TLS stack): https endpoints are not reachable. \
             Point GREPPY_ENDPOINT at an http:// gateway, or put the provider behind \
             a local TLS-terminating proxy such as CLIProxyAPI."
                .to_string(),
        )
    } else {
        Some(format!(
            "unsupported endpoint scheme; expected http:// (got {})",
            redact_userinfo(url)
        ))
    }
}

fn redact_userinfo(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    match rest.split_once('@') {
        Some((_, host)) => format!("{scheme}://{host}"),
        None => url.to_string(),
    }
}

fn split_query(url: &str) -> (&str, &str) {
    match url.split_once('?') {
        Some((path, query)) => (path, query),
        None => (url, ""),
    }
}

/// Unambiguous completion routes only. `/v1` alone is both protocols.
fn path_format_hint(url: &str) -> Option<ApiFormat> {
    let path = url_path(url).trim_end_matches('/');
    if path.ends_with("/chat/completions") {
        Some(ApiFormat::OpenAi)
    } else if path.ends_with("/v1/messages") {
        Some(ApiFormat::Anthropic)
    } else {
        None
    }
}

fn strip_api_suffix(url: &str) -> String {
    let path = url_path(url).trim_end_matches('/');
    let stripped = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/messages",
        "/v1",
    ]
    .iter()
    .find_map(|suffix| path.strip_suffix(suffix))
    .unwrap_or(path);
    let host = url_host(url);
    format!("{host}{stripped}")
}

fn url_host(url: &str) -> &str {
    let Some(rest) = url.split_once("://").map(|(_, rest)| rest) else {
        return url;
    };
    match rest.find('/') {
        Some(index) => &url[..url.len() - rest.len() + index],
        None => url,
    }
}

fn url_path(url: &str) -> &str {
    let Some(rest) = url.split_once("://").map(|(_, rest)| rest) else {
        return url;
    };
    match rest.find('/') {
        Some(index) => &rest[index..],
        None => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(base: &str, setting: ApiFormatSetting) -> ResolvedEndpoint {
        resolve_endpoint(base, setting)
    }

    #[test]
    fn default_gateway_stays_anthropic() {
        let got = resolved("http://127.0.0.1:8317/", ApiFormatSetting::Auto);
        assert_eq!(got.format, ApiFormat::Anthropic);
        assert_eq!(got.origin, "http://127.0.0.1:8317");
        assert_eq!(got.completion_url, "http://127.0.0.1:8317/v1/messages");
        assert_eq!(got.models_url, "http://127.0.0.1:8317/v1/models");
    }

    #[test]
    fn bare_v1_prefix_is_not_a_format_hint() {
        let got = resolved("http://127.0.0.1:11434/v1", ApiFormatSetting::Auto);
        assert_eq!(got.format, ApiFormat::Anthropic);
        assert_eq!(got.completion_url, "http://127.0.0.1:11434/v1/messages");
        assert_eq!(got.models_url, "http://127.0.0.1:11434/v1/models");
    }

    #[test]
    fn chat_completions_path_selects_openai_only_when_auto() {
        let got = resolved(
            "http://127.0.0.1:8080/v1/chat/completions",
            ApiFormatSetting::Auto,
        );
        assert_eq!(got.format, ApiFormat::OpenAi);
        assert_eq!(
            got.completion_url,
            "http://127.0.0.1:8080/v1/chat/completions"
        );
        assert_eq!(got.models_url, "http://127.0.0.1:8080/v1/models");
    }

    #[test]
    fn messages_path_selects_anthropic() {
        let got = resolved(
            "http://127.0.0.1:8317/v1/messages?beta=1",
            ApiFormatSetting::Auto,
        );
        assert_eq!(got.format, ApiFormat::Anthropic);
        assert_eq!(got.completion_url, "http://127.0.0.1:8317/v1/messages");
    }

    #[test]
    fn explicit_format_wins_over_the_other_path() {
        let got = resolved(
            "http://127.0.0.1:8317/v1/messages",
            ApiFormatSetting::Explicit(ApiFormat::OpenAi),
        );
        assert_eq!(got.format, ApiFormat::OpenAi);
        assert_eq!(
            got.completion_url,
            "http://127.0.0.1:8317/v1/chat/completions"
        );
    }

    #[test]
    fn prefixed_v1_keeps_the_proxy_prefix() {
        let got = resolved(
            "http://127.0.0.1:8317/openai/v1",
            ApiFormatSetting::Explicit(ApiFormat::OpenAi),
        );
        assert_eq!(got.origin, "http://127.0.0.1:8317/openai");
        assert_eq!(
            got.completion_url,
            "http://127.0.0.1:8317/openai/v1/chat/completions"
        );
        assert_eq!(got.models_url, "http://127.0.0.1:8317/openai/v1/models");
    }

    #[test]
    fn parse_rejects_unknown_format() {
        assert_eq!(
            ApiFormatSetting::parse("openai").unwrap(),
            ApiFormatSetting::Explicit(ApiFormat::OpenAi)
        );
        assert_eq!(
            ApiFormatSetting::parse(" Anthropic ").unwrap(),
            ApiFormatSetting::Explicit(ApiFormat::Anthropic)
        );
        assert!(ApiFormatSetting::parse("responses").is_err());
    }

    #[test]
    fn https_is_refused_without_embedding_userinfo() {
        let message = plain_http_refusal("https://user:sekrit@api.openai.com/v1").unwrap();
        assert!(message.contains("no TLS"));
        assert!(!message.contains("sekrit"));
        assert!(plain_http_refusal("http://127.0.0.1:11434/v1").is_none());
    }
}
