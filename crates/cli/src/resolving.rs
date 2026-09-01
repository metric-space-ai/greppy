//! Turning a name the agent typed into nodes in the graph.
//!
//! Split out of `lib.rs`; `use super::*` keeps every private helper there
//! reachable, and no behaviour changes.

use super::*;

/// How an unknown flag relates to the flags the subcommand really has.
///
/// The three cases need different answers, and collapsing them is what let a
/// typo through: dropping `--jsoon` and answering in text serves a caller who
/// asked for JSON the wrong shape of answer, which is worse than a refusal.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum UnknownFlagKind {
    /// The subcommand lists this flag, yet clap rejected it: retired here.
    RetiredHere,
    /// Within a typo's distance of a real flag, so the intent is known.
    TypoOf,
    /// Nothing close. The caller invented it; dropping it is safe.
    Invented,
}

pub(crate) fn classify_unknown_flag(
    argv: &[std::ffi::OsString],
    subcommand: &str,
    clap_message: &str,
) -> Option<UnknownFlagKind> {
    let unknown = unknown_flag_name(clap_message)?;
    let candidates = subcommand_flag_candidates(argv, subcommand);
    let closest = candidates
        .into_iter()
        .min_by_key(|candidate| levenshtein(&unknown, candidate))?;
    Some(match levenshtein(&unknown, closest) {
        0 => UnknownFlagKind::RetiredHere,
        1..=2 => UnknownFlagKind::TypoOf,
        _ => UnknownFlagKind::Invented,
    })
}

pub(crate) fn closest_valid_invocation(
    argv: &[std::ffi::OsString],
    subcommand: &str,
    clap_message: &str,
) -> Option<String> {
    let unknown = unknown_flag_name(clap_message)?;
    let unknown = unknown.as_str();
    let candidates = subcommand_flag_candidates(argv, subcommand);
    let replacement = candidates
        .into_iter()
        .min_by_key(|candidate| levenshtein(unknown, candidate))?;
    let distance = levenshtein(unknown, replacement);
    if distance > 4 || distance == 0 {
        return None;
    }
    Some(
        argv.iter()
            .map(|argument| {
                let argument = argument.to_string_lossy();
                let value = if argument == unknown {
                    replacement
                } else {
                    argument.as_ref()
                };
                shell_quote_cli(value)
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// The flags `subcommand` actually accepts, for near-miss comparison.
fn subcommand_flag_candidates(argv: &[std::ffi::OsString], subcommand: &str) -> Vec<&'static str> {
    let mut candidates = vec![
        "--root",
        "--device",
        "--json",
        "--code",
        "--all",
        "--limit",
        "--max",
        "--max-bytes",
        "--offset",
    ];
    let invocation = grep_passthrough_args(argv);
    let nested_edit = if subcommand == "edit" {
        invocation.get(1).and_then(|argument| argument.to_str())
    } else {
        None
    };
    candidates.extend(match subcommand {
        "read" => vec!["--symbol", "--path", "--handle", "--lines", "--line"],
        "who-calls" | "callees" | "brief" | "semantic-search" | "semantic" => {
            vec!["--path"]
        }
        "search-code" => vec![
            "--path",
            "--changed",
            "--staged",
            "--since",
            "--base",
            "--no-code",
            "--fixed",
        ],
        "changes" => vec!["--base"],
        "search-symbols" => vec!["--path", "--kind"],
        "impact" => vec!["--direction", "--edge", "--depth", "--since", "--base"],
        "trace" => vec!["--symbol", "--direction", "--edge", "--depth"],
        "path" => vec!["--from", "--to", "--edge"],
        "graph-locate" => vec!["--file", "--line"],
        "plus" => vec!["--k", "--explain"],
        "context" => vec!["--k", "--lines"],
        "edit" => match nested_edit {
            Some("replace") | Some("delete") => vec![
                "--file",
                "--old",
                "--old-file",
                "--pattern",
                "--lines",
                "--symbol",
                "--body",
                "--target",
                "--content",
                "--content-file",
                "--expect",
                "--dry-run",
                "--report",
            ],
            Some("insert") => vec![
                "--symbol",
                "--lines",
                "--target",
                "--before",
                "--after",
                "--content",
                "--content-file",
                "--dry-run",
                "--report",
            ],
            Some("patch") => vec![
                "--symbol",
                "--lines",
                "--target",
                "--patch-file",
                "--dry-run",
                "--report",
            ],
            _ => Vec::new(),
        },
        _ => Vec::new(),
    });
    candidates
}

/// If `symbol` is a qualified `Owner.member` / `Owner::member` query,
/// return the node ids of every primary-labelled node that genuinely
/// matches it: `name == member` AND the node's [`qname_owner_segment`]
/// equals the **last** segment of `owner`. Returns `None` when `symbol` is
/// bare (no separator) so the caller keeps its existing bare-name path.
///
/// Owner matching compares against the last `::`/`.`-segment of the query
/// owner, so both the natural `JsonReader.peekNumber` and a
/// fully-qualified `com.google.gson.JsonReader.peekNumber` resolve to the
/// `JsonReader` owner segment the qname carries.
///
/// Never-guess is preserved end to end: this only ever *narrows* the set
/// of same-named primary nodes to those whose owner matches. It returns
/// the genuine matching set — one id when the owner is unique, several
/// when the same `Owner.member` legitimately exists in multiple files —
/// and never picks one arbitrarily. An empty set (owner matches nothing)
/// is returned as `Some(vec![])`, which the callers surface as
/// "not found" without silently falling back to a bare-name guess that
/// would ignore the owner the agent supplied.
pub(crate) fn resolve_qualified_ids(
    rows: &[greppy_search::graph::SearchGraphRow],
    symbol: &str,
) -> Option<Vec<i64>> {
    let (owner, member) = split_qualified(symbol)?;
    // Compare on the last segment of the query owner so both the natural
    // `Owner.method` and a fully-qualified `pkg.Owner.method` match.
    let owner_tail = owner
        .rsplit("::")
        .next()
        .and_then(|s| s.rsplit('.').next())
        .unwrap_or(owner);
    let mut ids: Vec<i64> = rows
        .iter()
        .filter(|r| {
            is_primary_label(&r.label)
                && (r.name.eq_ignore_ascii_case(symbol)
                    || (r.name.eq_ignore_ascii_case(member)
                        && qname_owner_segment(&r.qualified_name)
                            .is_some_and(|owner| owner.eq_ignore_ascii_case(owner_tail))))
        })
        .map(|r| r.id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    Some(ids)
}

/// Resolve a `--symbol`/positional symbol argument to a node id within
/// the open store. Ranks candidates by [`label_rank`] (preferring
/// type/def-like labels), breaking ties deterministically by node id, so
/// a shared name resolves to the real definition rather than an `Impl`,
/// `EnumVariant`, or `Call` site. Falls back to an exact
/// `qualified_name` match. When `symbol` is `None` the first node in the
/// graph is used (preserves the historical no-arg `greppy trace`
/// behaviour).
pub(crate) fn resolve_symbol_id(
    store: &greppy_store::Store,
    symbol: Option<&str>,
) -> Result<Option<i64>> {
    // Push the name filter into SQL. The old form loaded the first 10k
    // nodes of the project (ordered by qualified_name) and filtered in
    // memory — on a repo bigger than the cap (django: 56k nodes) every
    // symbol outside that window silently resolved as "not found".
    let rows = symbol_candidate_rows(store, symbol)?;
    let id = match symbol {
        // Qualified query (`Owner.method` / `Owner::method`): resolve within
        // the owner-matched set only. `trace`/`impact`/`path` need a single
        // start node, so among the owner-matched candidates we pick the
        // best-ranked (then lowest id) — the same deterministic discipline
        // as the bare-name path, but confined to the nodes the owner
        // actually disambiguates to. An empty owner match yields `None`
        // (not found) instead of falling back to a bare-name guess that
        // ignores the owner the agent supplied.
        Some(s) if split_qualified(s).is_some() => {
            resolve_qualified_ids(&rows, s).and_then(|ids| {
                rows.iter()
                    .filter(|r| ids.contains(&r.id))
                    .min_by(|a, b| {
                        label_rank(&a.label)
                            .cmp(&label_rank(&b.label))
                            .then(a.id.cmp(&b.id))
                    })
                    .map(|r| r.id)
            })
        }
        Some(s) => {
            let best = rows
                .iter()
                .filter(|r| bare_symbol_name_matches(&r.name, s))
                .min_by(|a, b| {
                    label_rank(&a.label)
                        .cmp(&label_rank(&b.label))
                        .then(a.id.cmp(&b.id))
                })
                .map(|r| r.id);
            if best.is_some() {
                best
            } else {
                // Fall back to an exact qualified_name match — its own
                // indexed lookup now that `rows` only holds name matches.
                let q = greppy_search::GraphQuery::any()
                    .with_qualified_name(s)
                    .with_limit(1);
                greppy_search::search_graph(store, &q)?
                    .first()
                    .map(|r| r.id)
            }
        }
        None => rows.first().map(|r| r.id),
    };
    Ok(id)
}

/// Split `file/path.ext::REST` into `(path, rest)` when the head segment is a
/// file path. `search-symbols`/`read` PRINT qualified names as
/// `path::Owner::name` or `path::Kind::name`, so agents naturally feed those
/// forms — and their simplifications (`path::name`) — straight back in.
/// Postel's law: the tool must accept every form it emits. The path only
/// NARROWS; the last `::`/`.` segment is always the symbol name.
pub(crate) fn split_path_qualified(query: &str) -> Option<(&str, &str)> {
    let idx = query.find("::")?;
    let head = &query[..idx];
    let looks_like_path = head.contains('/')
        || std::path::Path::new(head)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| !e.is_empty() && e.chars().all(|c| c.is_ascii_alphanumeric()));
    looks_like_path.then(|| (head, &query[idx + 2..]))
}

pub(crate) fn resolve_symbol_nodes(
    store: &greppy_store::Store,
    symbol: Option<&str>,
) -> Result<Vec<i64>> {
    let Some(s) = symbol else {
        // No symbol: mirror resolve_symbol_id's "first node" behaviour.
        return Ok(resolve_symbol_id(store, None)?.into_iter().collect());
    };
    // Exact qualified names are the lossless locator emitted by search and
    // navigation commands. Resolve that spelling before simplifying a
    // path-qualified query to its leaf name: providers may expose an
    // enclosing/synthetic node whose qualified name is stable even when its
    // short leaf is not independently addressable.
    if s.contains("::") {
        let mut exact = greppy_search::search_graph(
            store,
            &greppy_search::GraphQuery::any()
                .with_qualified_name(s)
                .with_limit(10_000),
        )?
        .into_iter()
        .filter(|row| is_primary_label(&row.label))
        .map(|row| row.id)
        .collect::<Vec<_>>();
        exact.sort_unstable();
        exact.dedup();
        if !exact.is_empty() {
            return Ok(exact);
        }
    }
    // Path-qualified query (`path::name`, `path::Kind::name`, `path::Owner::name`
    // — exactly what search-symbols/read print): the last segment is the name,
    // the file path narrows, any middle Kind/Owner segment narrows further but
    // is not required. This accepts every emitted form and its natural
    // simplifications instead of only the one exact string.
    if let Some((path, spec)) = split_path_qualified(s) {
        let name = spec
            .rsplit("::")
            .next()
            .and_then(|x| x.rsplit('.').next())
            .unwrap_or(spec);
        let middle = split_qualified(spec).map(|(owner, _)| {
            owner
                .rsplit("::")
                .next()
                .and_then(|o| o.rsplit('.').next())
                .unwrap_or(owner)
        });
        let rows = symbol_candidate_rows(store, Some(name))?;
        let in_path: Vec<&greppy_search::graph::SearchGraphRow> = rows
            .iter()
            .filter(|r| {
                r.name.eq_ignore_ascii_case(name)
                    && is_primary_label(&r.label)
                    && indexed_path_matches_query(&r.file_path, path)
            })
            .collect();
        // If a middle segment was given (Kind label or owner), prefer the
        // subset it disambiguates to; otherwise take all name+path matches.
        let mut ids: Vec<i64> = match middle {
            Some(m) => {
                let narrowed: Vec<i64> = in_path
                    .iter()
                    .filter(|r| {
                        qname_owner_segment(&r.qualified_name) == Some(m)
                            || r.label.eq_ignore_ascii_case(m)
                    })
                    .map(|r| r.id)
                    .collect();
                if narrowed.is_empty() {
                    in_path.iter().map(|r| r.id).collect()
                } else {
                    narrowed
                }
            }
            None => in_path.iter().map(|r| r.id).collect(),
        };
        ids.sort_unstable();
        ids.dedup();
        if !ids.is_empty() {
            return Ok(ids);
        }
    }
    // Name filter pushed into SQL — see resolve_symbol_id for why the old
    // capped whole-project scan was wrong on large repos.
    let rows = symbol_candidate_rows(store, Some(s))?;
    // Qualified query (`Owner.method` / `Owner::method`): narrow the
    // same-named primary nodes to those the owner disambiguates to. This is
    // the natural form a coding agent types; without it the whole query
    // (name == "Owner.method") matches nothing and the command reports
    // "symbol not found". We return the owner-matched set as-is — one node
    // when unique, several when `Owner.method` legitimately exists in more
    // than one file (aggregated downstream, same as a bare name) — never a
    // guess. An empty owner match returns an empty set so the caller
    // reports "not found" rather than ignoring the owner and guessing.
    if let Some(ids) = resolve_qualified_ids(&rows, s) {
        return Ok(ids);
    }
    let mut ids: Vec<i64> = rows
        .iter()
        // Case-insensitive equality: symbol_candidate_rows only returns a
        // case-variant when it is UNAMBIGUOUS, so this never guesses.
        .filter(|r| bare_symbol_name_matches(&r.name, s) && is_primary_label(&r.label))
        .map(|r| r.id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        // No primary-labelled node — fall back to the single best match
        // (e.g. a name that only exists as a Call/Import pseudo-node).
        if let Some(id) = resolve_symbol_id(store, Some(s))? {
            ids.push(id);
        }
    }
    Ok(ids)
}

pub(crate) fn is_synthetic_file_anchor(label: &str, name: &str, qualified_name: &str) -> bool {
    name == "__file__"
        || qualified_name.ends_with("::__file__")
        || qualified_name.ends_with(".__file__")
        || (label == "File" && qualified_name.ends_with("__file__"))
}

pub(crate) fn display_symbol_name(
    label: &str,
    name: &str,
    qualified_name: &str,
    file_path: &str,
) -> String {
    if is_synthetic_file_anchor(label, name, qualified_name) {
        if file_path.is_empty() {
            "Module <unknown>".to_string()
        } else {
            format!("Module {file_path}")
        }
    } else {
        qualified_name.to_string()
    }
}

pub(crate) fn display_node_name(node: &greppy_store::Node) -> String {
    display_symbol_name(
        &node.label,
        &node.name,
        &node.qualified_name,
        &node.file_path,
    )
}

pub(crate) fn display_row_name(row: &greppy_search::graph::SearchGraphRow) -> String {
    display_symbol_name(&row.label, &row.name, &row.qualified_name, &row.file_path)
}

pub(crate) fn resolve_expand_alias(root: Option<&str>, alias: &str) -> Option<String> {
    let path = expand_alias_path(root, alias)?;
    std::fs::read_to_string(path)
        .ok()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
}

pub(crate) fn display_context_def_name(store: &greppy_store::Store, def: &ContextDef) -> String {
    if let Some(id) = def.node_id {
        if let Ok(Some(node)) = store.get_node(id) {
            return display_node_name(&node);
        }
    }
    let name = def
        .qualified_name
        .rsplit("::")
        .next()
        .unwrap_or(&def.qualified_name);
    display_symbol_name("", name, &def.qualified_name, &def.file_path)
}

/// Markers that identify a repository / project root when walking up
/// from the current directory. Kept in sync with the markers
/// `greppy_core::workspace::project_identity` recognises so that the
/// store path (hashed from the resolved root) and the project name
/// (derived from the same root) always agree (RV-006 / RV-011).
/// Resolve the effective workspace root for a command.
///
/// * If `--root <PATH>` was given, canonicalize it and resolve its enclosing
///   Git worktree/project root through the shared core resolver.
/// * Otherwise start at the current directory and walk **up** until a
///   repo marker (`.git`, `Cargo.toml`, `pyproject.toml`) is found,
///   returning that directory.
/// * If no marker is found anywhere in the chain, fall back to the
///   current directory.
///
/// This is the single source of truth every command routes through, so
/// `greppy index .` from the repo root and `greppy search-code Q`
/// from a subdirectory resolve to the **same** store path and the
/// **same** project identity (RV-006 closes the subdir/exit-73 gap;
/// RV-011 closes the index/search project-name mismatch).
pub(crate) fn resolve_root(root: Option<&str>) -> Result<std::path::PathBuf> {
    if let Some(r) = root {
        // Defect D9: `--root` used to be taken verbatim, so a relative
        // (`--root .`) or non-canonical (`/tmp/...` vs `/private/tmp/...`
        // on macOS, trailing slash) argument keyed the store/workspace
        // state differently than the indexer, which records the
        // canonicalized root — later lookups then failed with "no
        // workspace_state". Normalize to the canonical absolute path so
        // every spelling of the same directory is one workspace.
        let explicit = absolutize_path(std::path::Path::new(r));
        return Ok(workspace_locator::resolve_workspace_root(&explicit));
    }
    let cwd = std::env::current_dir()
        .map_err(|e| Error::io("read current_dir for root resolution", e))?;
    Ok(workspace_locator::resolve_workspace_root(&cwd))
}

/// The flag clap rejected, e.g. `--regex` from `unexpected argument '--regex' found`.
pub(crate) fn unknown_flag_name(clap_message: &str) -> Option<String> {
    let unknown = clap_message
        .strip_prefix("error: unexpected argument '")?
        .split('\'')
        .next()?;
    unknown.starts_with("--").then(|| unknown.to_string())
}

/// The same invocation with the rejected flag removed, so the call can proceed.
///
/// A flag the agent guessed is not a reason to answer nothing: dropping it
/// yields the query the caller actually meant. Its value is dropped with it
/// when it is a separate argument, and `None` means there was nothing to drop
/// (so the caller keeps its usage error).
/// Flags that were retired because they RESTRICTED a query.
///
/// Dropping a decorative flag costs nothing. Dropping a scope filter answers a
/// different question than the caller asked -- `--changed` means "only what I
/// touched", and silently searching the whole repository instead is a wrong
/// answer wearing the shape of a right one. These stay refusals.
const RETIRED_SCOPE_FLAGS: &[&str] = &["--changed", "--staged", "--since", "--base"];

/// Guessed flags that ask for LESS, paired with the spelling that delivers it.
pub(crate) const NARROWING_GUESSES: &[(&str, &str)] = &[
    ("--lines", "--offset N --limit M"),
    ("--line", "--offset N --limit M"),
    ("--first", "--head M"),
    ("--last", "--tail N"),
];

pub(crate) fn argv_without_unknown_flag(
    argv: &[std::ffi::OsString],
    clap_message: &str,
) -> Option<Vec<std::ffi::OsString>> {
    let unknown = unknown_flag_name(clap_message)?;
    if RETIRED_SCOPE_FLAGS.contains(&unknown.as_str()) {
        return None;
    }
    // A guessed flag that NARROWS has a real spelling here. Dropping it would
    // answer a wider question than the caller asked -- `--lines 1:20` becoming
    // the whole file is the same wrong-answer-shaped-right as a dropped scope
    // filter. Refusing while naming the spelling is still a way forward.
    if NARROWING_GUESSES.iter().any(|(guess, _)| *guess == unknown) {
        return None;
    }
    let mut out = Vec::with_capacity(argv.len());
    let mut dropped = false;
    let mut options_ended = false;
    for argument in argv {
        let text = argument.to_string_lossy();
        if text == "--" {
            options_ended = true;
            out.push(argument.clone());
            continue;
        }
        if !options_ended && text == unknown {
            // Drop the flag alone, never the token behind it: in
            // `brief --lines parse_path` that token is the symbol being asked
            // about, and eating it turns a recoverable call into "required
            // arguments were not provided". If it really was the flag's value,
            // the retry sees it as a stray and this same recovery removes it.
            dropped = true;
            continue;
        }
        if !options_ended && text.starts_with(&format!("{unknown}=")) {
            dropped = true;
            continue;
        }
        out.push(argument.clone());
    }
    dropped.then_some(out)
}

/// A stray positional that is plainly a path, rewritten as the path filter.
///
/// `grep PATTERN PATH` is muscle memory, and agents write
/// `greppy search-pattern foo modules/logging/` constantly. greppy spells that
/// `--path`, and refusing the habit costs a turn and teaches nothing. If the
/// rejected argument looks like a path, say so and use it as the filter.
pub(crate) fn argv_with_stray_path_as_filter(
    argv: &[std::ffi::OsString],
    clap_message: &str,
) -> Option<(Vec<std::ffi::OsString>, String)> {
    let stray = clap_message
        .strip_prefix("error: unexpected argument '")?
        .split('\'')
        .next()?;
    if stray.starts_with('-') || stray.is_empty() {
        return None;
    }
    let looks_like_path = stray.contains('/') || std::path::Path::new(stray).exists();
    if !looks_like_path {
        return None;
    }
    let mut out = Vec::with_capacity(argv.len() + 1);
    let mut replaced = false;
    let mut options_ended = false;
    for argument in argv {
        if argument == "--" {
            options_ended = true;
            out.push(argument.clone());
            continue;
        }
        if !options_ended && !replaced && argument.to_string_lossy() == stray {
            out.push(std::ffi::OsString::from("--path"));
            out.push(argument.clone());
            replaced = true;
            continue;
        }
        out.push(argument.clone());
    }
    replaced.then(|| (out, stray.to_string()))
}

/// A stray positional on a command that takes none, dropped so the call runs.
///
/// `greppy doctor parse_path` is a natural guess -- every other verb takes a
/// symbol -- and refusing it teaches nothing that running doctor would not have
/// shown. Only commands whose usage line carries no placeholder qualify; where
/// a different SHAPE is required (`path --from A --to B`), the usage line is
/// the answer and the refusal stands.
pub(crate) fn argv_without_stray_positional(
    argv: &[std::ffi::OsString],
    clap_message: &str,
    usage: Option<&str>,
) -> Option<(Vec<std::ffi::OsString>, String)> {
    let stray = clap_message
        .strip_prefix("error: unexpected argument '")?
        .split('\'')
        .next()?;
    if stray.starts_with('-') || stray.is_empty() {
        return None;
    }
    // A placeholder OUTSIDE the optional brackets means the command wants an
    // argument of its own; then the stray is a shape problem, not a surplus.
    // `[--root DIR]` is uppercase too, so the brackets have to go first.
    let bare = {
        let mut kept = String::new();
        let mut depth = 0usize;
        for ch in usage?.chars() {
            match ch {
                '[' | '<' => depth += 1,
                ']' | '>' => depth = depth.saturating_sub(1),
                _ if depth == 0 => kept.push(ch),
                _ => {}
            }
        }
        kept
    };
    if bare
        .split_whitespace()
        .skip(2)
        .any(|word| word.chars().any(|c| c.is_ascii_uppercase()))
    {
        return None;
    }
    let mut out = Vec::with_capacity(argv.len());
    let mut dropped = false;
    let mut options_ended = false;
    for argument in argv {
        if argument == "--" {
            options_ended = true;
            out.push(argument.clone());
            continue;
        }
        if !options_ended && !dropped && argument.to_string_lossy() == stray {
            dropped = true;
            continue;
        }
        out.push(argument.clone());
    }
    dropped.then(|| (out, stray.to_string()))
}
