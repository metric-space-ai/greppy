//! Shared predicate grammar for JSON records emitted by `web match` and
//! runtime-backed diagnostic streams such as `web network`.

use regex::Regex;
use serde_json::Value;

#[derive(Debug)]
pub struct Predicate {
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

/// Parse the record-query language documented by `greppy web match`.
pub fn parse(query: &str) -> Result<Vec<Predicate>, String> {
    let mut predicates = Vec::new();
    for term in split_terms(query) {
        predicates.push(parse_term(&term)?);
    }
    if predicates.is_empty() {
        return Err("empty query".into());
    }
    Ok(predicates)
}

/// Return true when every predicate matches the supplied JSON record.
pub fn matches(record: &Value, predicates: &[Predicate]) -> bool {
    predicates.iter().all(|predicate| {
        let Some(value) = lookup(record, &predicate.path) else {
            return false;
        };
        match &predicate.op {
            Op::Eq(want) => as_text(value) == *want,
            Op::Ne(want) => as_text(value) != *want,
            Op::Re(regex) => regex.is_match(&as_text(value)),
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

fn split_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut in_regex = false;
    let mut in_quote = false;
    let mut previous_was_tilde = false;
    for character in query.chars() {
        match character {
            '"' => {
                in_quote = !in_quote;
                current.push(character);
            }
            '/' if !in_quote => {
                if in_regex {
                    in_regex = false;
                } else if previous_was_tilde {
                    in_regex = true;
                }
                current.push(character);
            }
            ' ' if !in_regex && !in_quote => {
                if !current.is_empty() {
                    terms.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(character),
        }
        previous_was_tilde = character == '~';
    }
    if !current.is_empty() {
        terms.push(current);
    }
    terms
}

fn parse_term(term: &str) -> Result<Predicate, String> {
    for (marker, kind) in [
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
            let path = field.split('.').map(str::to_owned).collect();
            let op = match kind {
                0 => Op::Ne(unquote(rest)),
                1 => Op::Cmp(Ordering::Ge, number(rest)?),
                2 => Op::Cmp(Ordering::Le, number(rest)?),
                3 => Op::Re(parse_regex(rest)?),
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

fn number(value: &str) -> Result<f64, String> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("`{value}` is not a number"))
}

fn parse_regex(value: &str) -> Result<Regex, String> {
    let value = value.trim();
    let body = value.strip_prefix('/').ok_or("regex must start with /")?;
    let close = body.rfind('/').ok_or("regex must end with /")?;
    let (pattern, flags) = body.split_at(close);
    let mut prefix = String::new();
    for flag in flags[1..].chars() {
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
    let mut current = record;
    for key in path {
        current = current.get(key)?;
    }
    Some(current)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn status_query_filters_network_records() {
        let predicates = parse("status>=400 url~/missing/").unwrap();
        assert!(matches(
            &json!({"url":"https://example.test/missing","status":404}),
            &predicates
        ));
        assert!(!matches(
            &json!({"url":"https://example.test/ok","status":200}),
            &predicates
        ));
    }
}
