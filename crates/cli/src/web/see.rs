//! Inspection verbs. `match` filters record streams entirely client-side and
//! needs no runtime operation; `find extract inspect dom` follow once the
//! runtime exposes a node query.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use regex::Regex;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

#[derive(Debug, Subcommand)]
pub enum SeeCommand {
    /// Filter a JSONL record stream on stdin. Records that satisfy every
    /// predicate are written to stdout unchanged.
    ///
    /// Predicates are space separated and combined with AND:
    ///   field=value  field!=value  field~/regex/i
    ///   field>N  field>=N  field<N  field<=N
    /// Field paths may be dotted (`data.status`, `context.session`).
    Match {
        /// Query, e.g. `data.status>=400` or `role=button name~/save/i`.
        query: String,
        /// Emit the count of matching records instead of the records.
        #[arg(long)]
        count: bool,
        #[arg(long)]
        json: bool,
    },
    /// Locate nodes in the current page.
    ///
    ///   greppy web find 'css=#btn'
    ///   greppy web find 'role=button'
    ///   greppy web find 'text~/save|apply/i'
    Find {
        /// Node query: css=, xpath=, text=, text~/re/, role=, id=, tag=.
        /// A bare argument is read as a CSS selector.
        query: String,
        /// Return only the first match.
        #[arg(long)]
        first: bool,
        /// Cap the number of returned nodes.
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Pull values out of matching nodes.
    ///
    ///   greppy web extract 'css=a.lnk' --fields text,href
    Extract {
        /// Node query, same grammar as `find`.
        query: String,
        /// Comma-separated fields: text, href, value, id, tag, attr:NAME.
        #[arg(long, default_value = "text")]
        fields: String,
        #[arg(long, default_value_t = 200)]
        limit: usize,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Describe one node in detail.
    Inspect {
        /// Node query, same grammar as `find`.
        /// Also accepts @N from the current page's observe snapshot.
        query: String,
        /// Include every attribute.
        #[arg(long)]
        attrs: bool,
        /// Include the node's outer HTML.
        #[arg(long)]
        html: bool,
        #[arg(long)]
        session: Option<String>,
        /// Inspect this tab within the selected session.
        #[arg(long)]
        tab: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Read document markup.
    Dom {
        #[command(subcommand)]
        command: DomCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum DomCommand {
    /// Outer HTML of the document, or of the nodes a query matches.
    Html {
        /// Optional node query; omit for the whole document.
        query: Option<String>,
        /// Cap the returned markup per node.
        #[arg(long, default_value_t = 20000)]
        limit: usize,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Node, element and text counts for the document.
    Stats {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn dispatch(command: SeeCommand, root: Option<&str>) -> Result<i32> {
    match command {
        SeeCommand::Match { query, count, json } => run_match(&query, count, json),
        SeeCommand::Find {
            query,
            first,
            limit,
            session,
            json,
        } => {
            if let Err(message) = validate_query(&query) {
                return emit_error(json, invalid(&format!("web find: {message}")));
            }
            let take = if first { 1 } else { limit };
            let body = format!(
                "return {{ count: nodes.length, nodes: nodes.slice(0, {take}).map(function(e) \
                 {{ return describe(e, false); }}) }};"
            );
            super::runtimes::evaluate(root, json, session, &query_expression(&query, &body))
        }
        SeeCommand::Extract {
            query,
            fields,
            limit,
            session,
            json,
        } => {
            if let Err(message) = validate_query(&query) {
                return emit_error(json, invalid(&format!("web extract: {message}")));
            }
            if let Err(message) = validate_query(&query) {
                return emit_error(json, invalid(&format!("web extract: {message}")));
            }
            let wanted: Vec<&str> = fields
                .split(',')
                .map(str::trim)
                .filter(|f| !f.is_empty())
                .collect();
            if wanted.is_empty() {
                return emit_error(
                    json,
                    invalid("web extract: --fields must name at least one field"),
                );
            }
            // `attr:` without a name silently produced a column of nulls; an
            // unknown field did the same. Both read as "the page has nothing
            // there", which is a different statement from "you asked wrong".
            const FIELDS: [&str; 6] = ["text", "href", "value", "id", "tag", "checked"];
            for field in &wanted {
                if let Some(name) = field.strip_prefix("attr:") {
                    if name.trim().is_empty() {
                        return emit_error(
                            json,
                            invalid("web extract: `attr:` needs an attribute name"),
                        );
                    }
                } else if !FIELDS.contains(field) {
                    return emit_error(
                        json,
                        invalid(&format!(
                            "web extract: unknown field `{field}`; expected one of {} or attr:NAME",
                            FIELDS.join(", ")
                        )),
                    );
                }
            }
            let list = serde_json::Value::String(wanted.join(",")).to_string();
            let body = format!(
                "var want = {list}.split(','); \
                 return {{ count: nodes.length, rows: nodes.slice(0, {limit}).map(function(e) {{ \
                   var row = {{}}; \
                   want.forEach(function(f) {{ \
                     if (f.indexOf('attr:') === 0) row[f] = e.getAttribute(f.slice(5)); \
                     else if (f === 'text') row.text = String(e.textContent == null ? '' : e.textContent).replace(/\\s+/g, ' ').trim(); \
                     else if (f === 'tag') row.tag = e.tagName.toLowerCase(); \
                     else if (f === 'id') row.id = e.id || null; \
                     else row[f] = e[f] === undefined ? null : e[f]; \
                   }}); \
                   return row; \
                 }}) }};"
            );
            super::runtimes::evaluate(root, json, session, &query_expression(&query, &body))
        }
        SeeCommand::Inspect {
            query,
            attrs,
            html,
            session,
            tab,
            json,
        } => {
            if query.trim().starts_with('@') {
                let parsed = match parse_target(&query, false, false, None) {
                    Ok(parsed) => parsed,
                    Err(error) => return emit_error(json, error),
                };
                let session = match resolve_session(root, session) {
                    Ok(session) => session,
                    Err(error) => return emit_error(json, error),
                };
                let mut payload = json!({
                    "session_id": session, "selector": parsed.selector,
                    "attrs": attrs, "html": html,
                });
                if let Some(tab) = resolve_tab(root, tab) {
                    payload["tab_id"] = json!(tab);
                }
                return rpc(root, json, "web.inspect", payload, Some(session));
            }
            if let Err(message) = validate_query(&query) {
                return emit_error(json, invalid(&format!("web inspect: {message}")));
            }
            let with_html = if html {
                "if (nodes.length) out.html = nodes[0].outerHTML.slice(0, 20000);"
            } else {
                ""
            };
            let body = format!(
                "if (!nodes.length) return {{ count: 0, node: null }}; \
                 var out = {{ count: nodes.length, node: describe(nodes[0], {attrs}) }}; \
                 {with_html} return out;"
            );
            super::runtimes::evaluate_on_tab(
                root,
                json,
                session,
                resolve_tab(root, tab),
                &query_expression(&query, &body),
            )
        }
        SeeCommand::Dom { command } => match command {
            DomCommand::Html {
                query,
                limit,
                session,
                json,
            } => match query {
                None => {
                    let source = format!(
                        "(function(){{ var h = document.documentElement.outerHTML; \
                         return {{ bytes: h.length, truncated: h.length > {limit}, \
                         html: h.slice(0, {limit}) }}; }})()"
                    );
                    super::runtimes::evaluate(root, json, session, &source)
                }
                Some(query) => {
                    if let Err(message) = validate_query(&query) {
                        return emit_error(json, invalid(&format!("web dom html: {message}")));
                    }
                    let body = format!(
                        "return {{ count: nodes.length, html: nodes.map(function(e) \
                         {{ return e.outerHTML.slice(0, {limit}); }}) }};"
                    );
                    super::runtimes::evaluate(root, json, session, &query_expression(&query, &body))
                }
            },
            DomCommand::Stats { session, json } => {
                let source = "(function(){ return { \
                     elements: document.querySelectorAll('*').length, \
                     links: document.querySelectorAll('a[href]').length, \
                     images: document.querySelectorAll('img').length, \
                     inputs: document.querySelectorAll('input,select,textarea').length, \
                     scripts: document.querySelectorAll('script').length, \
                     bytes: document.documentElement.outerHTML.length, \
                     title: document.title, readyState: document.readyState }; })()";
                super::runtimes::evaluate(root, json, session, source)
            }
        },
    }
}

/// One predicate of a query.
#[derive(Debug)]
struct Predicate {
    path: Vec<String>,
    op: Op,
}

#[derive(Debug)]
enum Op {
    Eq(String),
    Ne(String),
    Re(Regex),
    Cmp(Ordering, f64),
}

#[derive(Debug, Clone, Copy)]
enum Ordering {
    Gt,
    Ge,
    Lt,
    Le,
}

/// Split on spaces, but keep spaces inside `/regex/` and `"quoted"` runs.
fn split_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut cur = String::new();
    let mut in_re = false;
    let mut in_quote = false;
    let mut prev_was_tilde = false;
    for ch in query.chars() {
        match ch {
            '"' => {
                in_quote = !in_quote;
                cur.push(ch);
            }
            '/' if !in_quote => {
                // A slash opens a regex only right after `~`; the next
                // unescaped slash closes it.
                if in_re {
                    in_re = false;
                } else if prev_was_tilde {
                    in_re = true;
                }
                cur.push(ch);
            }
            ' ' if !in_re && !in_quote => {
                if !cur.is_empty() {
                    terms.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(ch),
        }
        prev_was_tilde = ch == '~';
    }
    if !cur.is_empty() {
        terms.push(cur);
    }
    terms
}

fn parse_query(query: &str) -> std::result::Result<Vec<Predicate>, String> {
    let mut out = Vec::new();
    for term in split_terms(query) {
        out.push(parse_term(&term)?);
    }
    if out.is_empty() {
        return Err("empty query".into());
    }
    Ok(out)
}

fn parse_term(term: &str) -> std::result::Result<Predicate, String> {
    // Longest operators first so `>=` is not read as `>`.
    for (marker, make) in [
        ("!=", 0u8),
        (">=", 1),
        ("<=", 2),
        ("~", 3),
        ("=", 4),
        (">", 5),
        ("<", 6),
    ] {
        if let Some(at) = term.find(marker) {
            if at == 0 {
                continue;
            }
            let (field, rest) = term.split_at(at);
            let rest = &rest[marker.len()..];
            let path: Vec<String> = field.split('.').map(str::to_owned).collect();
            let op = match make {
                0 => Op::Ne(unquote(rest)),
                1 => Op::Cmp(Ordering::Ge, number(rest)?),
                2 => Op::Cmp(Ordering::Le, number(rest)?),
                3 => Op::Re(regex(rest)?),
                4 => Op::Eq(unquote(rest)),
                5 => Op::Cmp(Ordering::Gt, number(rest)?),
                _ => Op::Cmp(Ordering::Lt, number(rest)?),
            };
            return Ok(Predicate { path, op });
        }
    }
    Err(format!("term `{term}` has no operator"))
}

fn unquote(value: &str) -> String {
    let trimmed = value.trim();
    trimmed
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(trimmed)
        .to_owned()
}

fn number(value: &str) -> std::result::Result<f64, String> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("`{value}` is not a number"))
}

/// `~/pattern/flags` — Rust `regex` syntax. No lookaround, no backreferences.
fn regex(value: &str) -> std::result::Result<Regex, String> {
    let value = value.trim();
    let body = value.strip_prefix('/').ok_or("regex must start with /")?;
    let close = body.rfind('/').ok_or("regex must end with /")?;
    let (pattern, flags) = body.split_at(close);
    let flags = &flags[1..];
    let mut prefix = String::new();
    for flag in flags.chars() {
        match flag {
            'i' | 'm' | 's' | 'x' | 'u' => prefix.push(flag),
            other => return Err(format!("unsupported regex flag `{other}`")),
        }
    }
    let full = if prefix.is_empty() {
        pattern.to_owned()
    } else {
        format!("(?{prefix}){pattern}")
    };
    Regex::new(&full).map_err(|error| format!("invalid regex: {error}"))
}

fn lookup<'a>(record: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut cur = record;
    for key in path {
        cur = cur.get(key)?;
    }
    Some(cur)
}

/// Scalar rendering used for `=`, `!=` and regex comparisons. Strings compare
/// as themselves so `role=button` does not need quotes around `button`.
fn as_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn as_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn matches(record: &Value, predicates: &[Predicate]) -> bool {
    predicates.iter().all(|predicate| {
        let Some(value) = lookup(record, &predicate.path) else {
            // An absent field never satisfies a predicate. Absence is not
            // equality with the empty string.
            return false;
        };
        match &predicate.op {
            Op::Eq(want) => as_text(value) == *want,
            Op::Ne(want) => as_text(value) != *want,
            Op::Re(re) => re.is_match(&as_text(value)),
            Op::Cmp(ordering, want) => match as_number(value) {
                Some(got) => match ordering {
                    Ordering::Gt => got > *want,
                    Ordering::Ge => got >= *want,
                    Ordering::Lt => got < *want,
                    Ordering::Le => got <= *want,
                },
                None => false,
            },
        }
    })
}

fn run_match(query: &str, count_only: bool, json_out: bool) -> Result<i32> {
    let predicates = match parse_query(query) {
        Ok(predicates) => predicates,
        Err(message) => return emit_error(json_out, invalid(&format!("web match: {message}"))),
    };
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut seen = 0usize;
    let mut hit = 0usize;
    let mut malformed = 0usize;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        seen += 1;
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            malformed += 1;
            continue;
        };
        if matches(&record, &predicates) {
            hit += 1;
            if !count_only {
                let _ = writeln!(stdout, "{line}");
            }
        }
    }
    if count_only {
        let payload = serde_json::json!({
            "schema": "greppy.web-runtime.v1",
            "status": "ok",
            "operation": "web.match",
            "result": { "seen": seen, "matched": hit, "malformed": malformed },
        });
        emit_web(json_out, &payload)?;
    }
    // No match is a real answer, not an error: exit 0 with an empty stream.
    Ok(0)
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use serde_json::json;

    fn matched(query: &str, record: serde_json::Value) -> bool {
        let predicates = parse_query(query).expect("query parses");
        matches(&record, &predicates)
    }

    #[test]
    fn equality_compares_strings_without_quotes() {
        assert!(matched("role=button", json!({ "role": "button" })));
        assert!(!matched("role=button", json!({ "role": "link" })));
    }

    #[test]
    fn dotted_paths_reach_into_nested_records() {
        assert!(matched(
            "context.session=s_7",
            json!({ "context": { "session": "s_7" } })
        ));
    }

    #[test]
    fn absent_field_never_matches() {
        assert!(!matched("role=button", json!({ "name": "x" })));
        assert!(!matched("role!=button", json!({ "name": "x" })));
    }

    #[test]
    fn numeric_comparisons_use_numbers_not_text() {
        assert!(matched("status>=400", json!({ "status": 404 })));
        assert!(!matched("status>=400", json!({ "status": 200 })));
        // "90" must not sort above "400" as text.
        assert!(!matched("status>=400", json!({ "status": 90 })));
    }

    #[test]
    fn regex_honours_the_case_insensitive_flag() {
        assert!(matched("name~/save/i", json!({ "name": "Save draft" })));
        assert!(!matched("name~/save/", json!({ "name": "Save draft" })));
    }

    #[test]
    fn spaces_inside_a_regex_do_not_split_the_query() {
        let predicates = parse_query("name~/save draft/i visible=true").expect("parses");
        assert_eq!(predicates.len(), 2);
        assert!(matched(
            "name~/save draft/i visible=true",
            json!({ "name": "Save Draft", "visible": true })
        ));
    }

    #[test]
    fn every_predicate_must_hold() {
        assert!(!matched(
            "role=button visible=true",
            json!({ "role": "button", "visible": false })
        ));
    }

    #[test]
    fn unsupported_regex_flag_is_a_query_error() {
        assert!(parse_query("name~/x/z").is_err());
    }

    #[test]
    fn term_without_operator_is_a_query_error() {
        assert!(parse_query("button").is_err());
    }

    #[test]
    fn booleans_compare_as_text_and_as_number() {
        assert!(matched("visible=true", json!({ "visible": true })));
        assert!(matched("visible>=1", json!({ "visible": true })));
    }

    #[test]
    fn node_query_quotes_are_only_cli_grouping() {
        assert_eq!(normalize_node_query("css=a b"), "css=a b");
        assert_eq!(normalize_node_query(r#"css="a b""#), "css=a b");
        assert_eq!(
            normalize_node_query(r#"text="Save draft""#),
            "text=Save draft"
        );
    }

    #[test]
    fn node_queries_preserve_bare_css_operators() {
        for query in [
            "input[name=quantity]",
            "[data-x=y]",
            "input[class~=quantity]",
            "div~span",
            "div ~ span",
            "input:has(+ input[value='3'])",
            "css=div~span",
        ] {
            assert!(validate_query(query).is_ok(), "{query}");
        }
    }

    #[test]
    fn node_queries_reject_unknown_conditions_and_malformed_regexes() {
        for query in [
            "time=500ms",
            "unknown=value",
            "text~/[bad/",
            "text~/x/z",
            "css~div",
            "",
        ] {
            assert!(validate_query(query).is_err(), "{query}");
        }
        for query in [
            "text=Quantity:",
            "text~/Quantity:/",
            "role=checkbox",
            "css=#absent",
        ] {
            assert!(validate_query(query).is_ok(), "{query}");
        }
    }
}

// ---------------------------------------------------------------------------
// Node queries: find / extract / inspect / dom
//
// All four are questions about the live document, so all four are one
// `web.evaluate` with a generated snippet instead of four engine operations.
// The resolver below is shared by every one of them.
// ---------------------------------------------------------------------------

/// JavaScript that turns one query term into a node list.
///
/// Supported forms, matching the CLI query grammar:
///   css=SELECTOR   xpath=EXPR   text=EXACT   text~/re/flags
///   role=NAME      id=NAME      tag=NAME
/// A bare argument is treated as a CSS selector, which is what a caller who
/// types `#btn` means.
pub(super) const RESOLVER_JS: &str = greppy_web_client::NODE_QUERY_RESOLVER_JS;

/// Serialize one node the way `find` and `inspect` report it.
pub(super) const DESCRIBE_JS: &str = greppy_web_client::DESCRIBE_NODE_JS;

/// Check a node query before it reaches the page.
///
/// Three failures used to surface as engine noise or, worse, as an empty
/// result: an empty query, a malformed regex, and an unknown kind. The last
/// is the dangerous one — a caller reads `count: 0` as "the page has no such
/// element" when in truth the query was never understood.
pub(super) fn validate_query(query: &str) -> std::result::Result<(), String> {
    validate_query_impl(query, true)
}

/// Conditions already use the JavaScript regex dialect in the page. Validate
/// the query kind here without narrowing that dialect to Rust regex syntax.
/// A malformed native regex is returned as an evaluation error, not polled.
pub(super) fn validate_condition_query(query: &str) -> std::result::Result<(), String> {
    validate_query_impl(query, false)
}

fn validate_query_impl(query: &str, validate_regex: bool) -> std::result::Result<(), String> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Err("empty query; expected css=, xpath=, text=, role=, id= or tag=".into());
    }
    const KINDS: [&str; 6] = ["css", "xpath", "text", "role", "id", "tag"];
    let Some(split) = trimmed.find(['=', '~']) else {
        // No operator at all: treated as a CSS selector, which is what a
        // caller who typed `#btn` means.
        return Ok(());
    };
    let (kind, rest) = trimmed.split_at(split);
    // Match the resolver's anchored lowercase prefix grammar. Operators
    // inside CSS attributes or after a combinator are not query kinds.
    if kind.is_empty() || !kind.bytes().all(|byte| byte.is_ascii_lowercase()) {
        return Ok(());
    }
    let op = &rest[..1];
    let value = &rest[1..];
    if op == "~" && !KINDS.contains(&kind) {
        // div~span remains a bare CSS general-sibling selector.
        return Ok(());
    }
    if !KINDS.contains(&kind) {
        return Err(format!(
            "unknown query kind `{kind}`; expected one of {}",
            KINDS.join(", ")
        ));
    }
    if op == "~" {
        if kind != "text" {
            return Err(format!("`~` needs a text query, not `{kind}`"));
        }
        if !validate_regex {
            return Ok(());
        }
        let body = value
            .strip_prefix('/')
            .ok_or("regex must be written ~/pattern/flags")?;
        let close = body
            .rfind('/')
            .ok_or("regex must be written ~/pattern/flags")?;
        let (pattern, flags) = body.split_at(close);
        for flag in flags[1..].chars() {
            if !"imsu".contains(flag) {
                return Err(format!("unsupported regex flag `{flag}`"));
            }
        }
        // Compile here so a malformed pattern is a usage error rather than a
        // JavaScript exception from inside the page.
        Regex::new(pattern).map_err(|error| format!("invalid regex: {error}"))?;
    }
    Ok(())
}

/// Build the expression for a query-based command.
pub(super) fn query_expression_pub(query: &str, body: &str) -> String {
    query_expression(query, body)
}

fn query_expression(query: &str, body: &str) -> String {
    // The query is embedded as a JSON string so quotes and backslashes in a
    // regex survive intact.
    let literal = serde_json::Value::String(normalize_node_query(query)).to_string();
    format!(
        "(function(){{ var resolve = {RESOLVER_JS}; var describe = {DESCRIBE_JS}; \
         var nodes = resolve({literal}); {body} }})()"
    )
}

pub(super) fn normalize_node_query(query: &str) -> String {
    let trimmed = query.trim();
    let Some((kind, value)) = trimmed.split_once('=') else {
        return trimmed.to_owned();
    };
    if !matches!(kind, "css" | "xpath" | "text" | "role" | "id" | "tag") {
        return trimmed.to_owned();
    }
    let value = value.trim();
    match serde_json::from_str::<String>(value) {
        Ok(value) => format!("{kind}={value}"),
        Err(_) => trimmed.to_owned(),
    }
}
