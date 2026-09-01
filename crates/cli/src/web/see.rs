//! Inspection verbs. `match` filters record streams entirely client-side and
//! needs no runtime operation; `find extract inspect dom` follow once the
//! runtime exposes a node query.

use super::common::*;
use clap::Subcommand;
use greppy_core::error::Result;
use regex::Regex;
use serde_json::Value;
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
        query: String,
        /// Include every attribute.
        #[arg(long)]
        attrs: bool,
        /// Include the node's outer HTML.
        #[arg(long)]
        html: bool,
        #[arg(long)]
        session: Option<String>,
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
            json,
        } => {
            if let Err(message) = validate_query(&query) {
                return emit_error(json, invalid(&format!("web inspect: {message}")));
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
            super::runtimes::evaluate(root, json, session, &query_expression(&query, &body))
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
pub(super) const RESOLVER_JS: &str = r#"
(function(q) {
  function esc(s) { return String(s).replace(/"/g, '\\"'); }
  var m = /^([a-z]+)(=|~)([\s\S]*)$/.exec(q);
  var kind = m ? m[1] : "css";
  var op = m ? m[2] : "=";
  var val = m ? m[3] : q;
  var all = Array.prototype.slice.call(document.querySelectorAll("*"));
  function norm(s) { return String(s == null ? "" : s).replace(/\s+/g, " ").trim(); }
  function reOf(v) {
    var r = /^\/([\s\S]*)\/([imsu]*)$/.exec(v);
    return r ? new RegExp(r[1], r[2]) : new RegExp(v);
  }
  if (kind === "css") return Array.prototype.slice.call(document.querySelectorAll(val));
  if (kind === "xpath") {
    var out = [], it = document.evaluate(val, document, null, 5, null), n;
    while ((n = it.iterateNext())) out.push(n);
    return out;
  }
  if (kind === "id") return Array.prototype.slice.call(document.querySelectorAll(String.fromCharCode(35) + val));
  if (kind === "tag") return Array.prototype.slice.call(document.getElementsByTagName(val));
  if (kind === "role") {
    return all.filter(function (e) {
      var r = e.getAttribute("role");
      if (r) return r === val;
      var t = e.tagName.toLowerCase();
      if (val === "button") return t === "button" || (t === "input" && /^(button|submit|reset)$/.test(e.type || ""));
      if (val === "link") return t === "a" && e.hasAttribute("href");
      if (val === "textbox") return t === "textarea" || (t === "input" && !/^(button|submit|reset|checkbox|radio|file)$/.test(e.type || ""));
      if (val === "checkbox") return t === "input" && e.type === "checkbox";
      if (val === "heading") return /^h[1-6]$/.test(t);
      return false;
    });
  }
  if (kind === "text") {
    if (op === "~") { var re = reOf(val); return all.filter(function (e) { return re.test(norm(e.textContent)); }); }
    return all.filter(function (e) { return norm(e.textContent) === norm(val); });
  }
  return [];
})
"#;

/// Serialize one node the way `find` and `inspect` report it.
pub(super) const DESCRIBE_JS: &str = r#"
(function(e, withAttrs) {
  var r = e.getBoundingClientRect();
  var out = {
    tag: e.tagName.toLowerCase(),
    id: e.id || null,
    text: String(e.textContent == null ? "" : e.textContent).replace(/\s+/g, " ").trim().slice(0, 120),
    visible: !!(r.width || r.height) && getComputedStyle(e).visibility !== "hidden" && getComputedStyle(e).display !== "none",
    box: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }
  };
  if (e.value !== undefined) out.value = e.value;
  if (e.checked !== undefined) out.checked = e.checked;
  if (e.disabled !== undefined) out.disabled = e.disabled;
  if (e.href) out.href = e.href;
  if (withAttrs) {
    out.attrs = {};
    for (var i = 0; i < e.attributes.length; i++) out.attrs[e.attributes[i].name] = e.attributes[i].value;
  }
  return out;
})
"#;

/// Check a node query before it reaches the page.
///
/// Three failures used to surface as engine noise or, worse, as an empty
/// result: an empty query, a malformed regex, and an unknown kind. The last
/// is the dangerous one — a caller reads `count: 0` as "the page has no such
/// element" when in truth the query was never understood.
pub(super) fn validate_query(query: &str) -> std::result::Result<(), String> {
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
    let op = &rest[..1];
    let value = &rest[1..];
    if !kind.is_empty() && !KINDS.contains(&kind) {
        return Err(format!(
            "unknown query kind `{kind}`; expected one of {}",
            KINDS.join(", ")
        ));
    }
    if op == "~" {
        if kind != "text" {
            return Err(format!("`~` needs a text query, not `{kind}`"));
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
    let literal = serde_json::Value::String(query.to_owned()).to_string();
    format!(
        "(function(){{ var resolve = {RESOLVER_JS}; var describe = {DESCRIBE_JS}; \
         var nodes = resolve({literal}); {body} }})()"
    )
}
