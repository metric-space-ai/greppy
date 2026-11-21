//! `greppy isolate PLACE…` — everything around a place in one call.
//!
//! A fix starts from a place: the line a report names, or a symbol. What an
//! agent needs next is always the same five answers — the symbol at that
//! place, its direct callers, everything that reaches it, what it calls and
//! what it reaches. Asked one command at a time that is five processes, each
//! comparing the whole checkout with the index before it answers. Here the
//! same five answers come from one process and one comparison.

use greppy_core::error::Result;

/// The answers per place, in the order they are asked.
const ANSWERS: [(&str, &str); 4] = [
    ("callers", "direct callers and usages"),
    ("reach", "everything that reaches it"),
    ("callees", "what it calls"),
    ("forward", "everything it reaches"),
];

pub(crate) fn dispatch_isolate(places: &[String], json: bool, root: Option<&str>) -> Result<i32> {
    crate::reuse_freshness_verdicts_for_this_command();
    // Brought up to date once (or left as it is, and said so) for every answer.
    let mut store = crate::freshness::open_default_store_query_writer(root)?;
    crate::freshness::maybe_reindex_stale(&mut store, root)?;
    drop(store);
    let mut isolated = Vec::with_capacity(places.len());
    let mut complete = true;
    for place in places {
        let (seed, mut gaps) = seed_at(place, root)?;
        let mut answers = serde_json::Map::new();
        if let Some(symbol) = seed
            .get("qualified_name")
            .and_then(serde_json::Value::as_str)
        {
            for (key, _) in ANSWERS {
                let said = match crate::captured_json(|| ask(key, symbol, root)) {
                    Ok((0, said)) if said.get("symbol_found") != Some(&false.into()) => said,
                    Ok((_, said)) => {
                        gaps.push(format!("{key}: {}", unanswered(&said)));
                        said
                    }
                    Err(error) => {
                        gaps.push(format!("{key}: {error}"));
                        serde_json::Value::Null
                    }
                };
                answers.insert(
                    key.into(),
                    said.get("hits")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!([])),
                );
            }
        }
        complete &= gaps.is_empty();
        let mut entry = serde_json::json!({ "place": place, "seed": seed, "gaps": gaps });
        if let Some(object) = entry.as_object_mut() {
            object.extend(answers);
        }
        isolated.push(entry);
    }
    if json {
        let said = serde_json::json!({
            "command": "isolate",
            "status": if complete { "ok" } else { "incomplete" },
            "places": isolated,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&said)
                .map_err(|error| greppy_core::Error::Invalid(format!("encode isolate: {error}")))?
        );
    } else {
        print_text(&isolated);
    }
    Ok(if complete { 0 } else { 1 })
}

/// The symbol a place names: the innermost one at `file:line`, or the place
/// itself when it is a symbol.
fn seed_at(place: &str, root: Option<&str>) -> Result<(serde_json::Value, Vec<String>)> {
    let is_line = place
        .rsplit_once(':')
        .is_some_and(|(file, line)| !file.ends_with(':') && line.parse::<u32>().is_ok());
    if !is_line {
        return Ok((serde_json::json!({ "qualified_name": place }), Vec::new()));
    }
    let (_, said) =
        crate::captured_json(|| crate::dispatch_graph_locate(Some(place), None, None, true, root))?;
    match said.get("hits").and_then(|hits| hits.get(0)) {
        Some(hit) => {
            let mut seed = hit.clone();
            seed["match_kind"] = said.get("match_kind").cloned().unwrap_or_default();
            Ok((seed, Vec::new()))
        }
        None => Ok((
            serde_json::Value::Null,
            vec![format!(
                "no symbol at {place}: {}",
                said.get("warning")
                    .or_else(|| said.get("status"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("not in the index")
            )],
        )),
    }
}

fn unanswered(said: &serde_json::Value) -> String {
    if said.get("symbol_found") == Some(&false.into()) {
        return "no such symbol in the index".into();
    }
    ["warning", "message", "status"]
        .iter()
        .find_map(|key| said.get(*key).and_then(serde_json::Value::as_str))
        .unwrap_or("no answer")
        .to_string()
}

fn ask(answer: &str, symbol: &str, root: Option<&str>) -> Result<i32> {
    match answer {
        "callers" => crate::nav::dispatch_who_calls(Some(symbol), &[], false, true, true, root),
        "callees" => crate::nav::dispatch_callees(Some(symbol), &[], false, true, true, root),
        direction => crate::nav::dispatch_impact(
            Some(symbol),
            &[],
            if direction == "reach" {
                "incoming"
            } else {
                "outgoing"
            },
            None,
            crate::cli_surface::IMPACT_DEFAULT_DEPTH,
            true,
            true,
            root,
        ),
    }
}

fn print_text(isolated: &[serde_json::Value]) {
    for entry in isolated {
        let place = entry["place"].as_str().unwrap_or_default();
        match entry["seed"]["qualified_name"].as_str() {
            Some(symbol) => println!("{place}  {symbol}"),
            None => println!("{place}  (no symbol)"),
        }
        for (key, title) in ANSWERS {
            let hits = entry[key].as_array().map(Vec::as_slice).unwrap_or_default();
            println!("  {title}: {}", hits.len());
            for hit in hits {
                let file = hit["file"].as_str().or(hit["file_path"].as_str());
                let line = hit["line"].as_i64().or(hit["start_line"].as_i64());
                let name = hit["qualified_name"].as_str().or(hit["name"].as_str());
                if let (Some(file), Some(line), Some(name)) = (file, line, name) {
                    println!("    {file}:{line}  {name}");
                }
            }
        }
        for gap in entry["gaps"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            println!("  gap: {}", gap.as_str().unwrap_or_default());
        }
    }
}
