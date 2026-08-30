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
}

pub(super) fn dispatch(command: SeeCommand, _root: Option<&str>) -> Result<i32> {
    match command {
        SeeCommand::Match { query, count, json } => run_match(&query, count, json),
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
