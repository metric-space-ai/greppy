//! AGENTS.md is the system prompt. It is prose, so nothing stops it from
//! drifting back to describing a tool that no longer exists — and a prompt that
//! promises more than the code delivers is how `find-usages` came to advertise
//! "calls, uses, imports" while returning 15 of 43 references and no imports at
//! all. These assertions are the guard rail: they do not judge the wording,
//! they hold the few statements that must stay true.

fn prompt() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../AGENTS.md")
        .canonicalize()
        .expect("AGENTS.md must exist next to the crates");
    std::fs::read_to_string(path).expect("AGENTS.md must be readable")
}

/// Everything from `NAVIGATE:` up to the next section heading. Headings are the
/// only lines that start at column 0 and end in a colon, so that is the boundary
/// — the blank line inside the section before the flags is not one.
fn navigate_section(text: &str) -> String {
    let mut lines = text.lines().skip_while(|line| *line != "NAVIGATE:");
    let heading = lines.next().expect("AGENTS.md must have a NAVIGATE section");
    let body = lines.take_while(|line| {
        !(line.ends_with(':') && line.starts_with(|c: char| c.is_ascii_uppercase()))
    });
    std::iter::once(heading)
        .chain(body)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_removed_command_is_not_advertised() {
    let text = prompt();
    assert!(
        !text.contains("find-usages"),
        "find-usages was removed from the CLI; the prompt must not name it"
    );
}

#[test]
fn navigate_lists_exactly_the_five_commands() {
    let section = navigate_section(&prompt());
    for verb in [
        "where-am-i",
        "who-calls S",
        "callees S",
        "brief S",
        "impact S",
        "path --from A --to B",
    ] {
        assert!(
            section.contains(verb),
            "NAVIGATE must describe `{verb}`; section was:\n{section}"
        );
    }
    let described = section
        .lines()
        .filter(|line| line.starts_with("  ") && !line.starts_with("    ") && !line.starts_with("  --"))
        .filter(|line| !line.trim().is_empty())
        .count();
    assert_eq!(
        described, 7,
        "NAVIGATE has six commands plus the multi-symbol note; got {described} entries:\n{section}"
    );
}

#[test]
fn the_result_shape_is_the_one_the_commands_print() {
    let text = prompt();
    assert!(
        text.contains("`file:line  name`"),
        "the Throughout paragraph must state the shape a result is printed in"
    );
    assert!(
        !text.contains("qualified_name file:line"),
        "the qualified name is no longer printed beside the address; that shape \
         put the path on the line twice"
    );
    assert!(
        text.contains("trailing `test`"),
        "the test marker appears in output, so it belongs in the prompt once"
    );
    assert!(
        text.contains("generated hint, not source"),
        "sentences after an em dash are generated; the one trust boundary the \
         agent cannot discover belongs in the prompt"
    );
}

#[test]
fn code_does_not_promise_a_handle() {
    let section = navigate_section(&prompt());
    let code_flag = section
        .lines()
        .find(|line| line.trim_start().starts_with("--code"))
        .expect("NAVIGATE must document --code");
    assert!(
        !code_flag.contains("handle"),
        "--code prints the source at the reported location and hides nothing a \
         handle could point at; got: {code_flag}"
    );
}

#[test]
fn multi_symbol_is_promised_only_where_it_holds() {
    let section = navigate_section(&prompt());
    let note = section
        .lines()
        .find(|line| line.contains("several symbols at once"))
        .expect("NAVIGATE must say which commands take several symbols");
    assert!(
        note.contains("who-calls") && note.contains("callees") && !note.contains("path"),
        "path takes exactly one --from and one --to; got: {note}"
    );
}

/// Everything from `SEARCH:` up to the next section heading, same boundary rule
/// as `navigate_section`.
fn search_section(text: &str) -> String {
    let mut lines = text.lines().skip_while(|line| *line != "SEARCH:");
    let heading = lines.next().expect("AGENTS.md must have a SEARCH section");
    let body = lines.take_while(|line| {
        !(line.ends_with(':') && line.starts_with(|c: char| c.is_ascii_uppercase()))
    });
    std::iter::once(heading)
        .chain(body)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn search_is_one_family_on_one_axis() {
    let text = prompt();
    for old in ["search-code", "search-symbols", "semantic-search"] {
        assert!(
            !text.contains(old),
            "`{old}` was renamed in 0.3.0; the prompt must not resurrect it"
        );
    }
    let section = search_section(&text);
    for verb in ["search \"", "search-symbol NAME", "search-pattern REGEX"] {
        assert!(
            section.contains(verb),
            "SEARCH must describe `{verb}`; section was:\n{section}"
        );
    }
    // grep compatibility has a spelling already: grep's own, via the
    // passthrough. A --grep flag would be a second spelling of it.
    assert!(
        !section.contains("--grep"),
        "grep-compatible output is the passthrough, not a search flag"
    );
}

#[test]
fn footer_flags_hold_for_every_command_in_their_section() {
    // A flag in a section footer applies to every command of the section; an
    // option one command needs is part of that command's syntax, spelled in
    // the command column like `path --from A --to B`. A footer line naming a
    // command with a colon is the scope-prefix notation coming back.
    let text = prompt();
    for section in [
        navigate_section(&text),
        search_section(&text),
        read_section(&text),
    ] {
        for line in section.lines().filter(|l| l.starts_with("  --")) {
            for verb in [
                "search:", "search-symbol:", "search-pattern:", "who-calls:",
                "callees:", "brief:", "impact:", "path:", "read:", "read-smart:",
                "read-file:",
            ] {
                assert!(
                    !line.contains(verb),
                    "footer flag scoped to one command: {line}"
                );
            }
        }
    }
}

/// Everything from `READ:` up to the next section heading.
fn read_section(text: &str) -> String {
    // The EDIT heading carries prose on the same line ("EDIT: an edit applies
    // completely…"), so the boundary is: column 0, and everything before the
    // first colon is upper case.
    fn is_heading(line: &str) -> bool {
        line.split(':').next().is_some_and(|head| {
            !head.is_empty()
                && line.contains(':')
                && head.chars().all(|c| c.is_ascii_uppercase() || c == ' ')
        })
    }
    let mut lines = text.lines().skip_while(|line| *line != "READ:");
    let heading = lines.next().expect("AGENTS.md must have a READ section");
    let body = lines.take_while(|line| !is_heading(line));
    std::iter::once(heading)
        .chain(body)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn read_is_bytes_and_the_lossy_view_is_its_own_verb() {
    let text = prompt();
    let section = read_section(&text);
    for verb in ["read S", "read-smart S", "read-file PATH"] {
        assert!(
            section.contains(verb),
            "READ must describe `{verb}`; section was:\n{section}"
        );
    }
    // The bare read is unconditional. Words that would announce truncation,
    // folding or summarizing on it belong to read-smart and read-file only.
    let read_line_block: String = section
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("read S"))
        .take_while(|l| !l.trim_start().starts_with("read-smart"))
        .collect::<Vec<_>>()
        .join("\n");
    for word in ["fold", "paginat", "summar", "semantic"] {
        assert!(
            !read_line_block.contains(word),
            "the bare read must not advertise lossy behaviour ({word}): {read_line_block}"
        );
    }
    // The guess heuristic and its repair flag are dead: read takes symbols,
    // read-file takes paths, nothing is resolved by luck.
    assert!(
        !text.contains("also a path on disk") && !section.contains("--symbol"),
        "the name-vs-path guess heuristic must not come back"
    );
    // --context died with automatic documentation inclusion.
    assert!(
        !section.contains("--context"),
        "--context's documented purpose is default behaviour now"
    );
}

#[test]
fn orient_is_dissolved() {
    let text = prompt();
    assert!(
        !text.contains("ORIENT:"),
        "ORIENT dissolved into NAVIGATE; the section must not return"
    );
    for dead in ["map [PATH]", "outline PATH", "verify -- CMD", "\n  changes "] {
        assert!(
            !text.contains(dead),
            "`{dead}` died with ORIENT and must not be advertised"
        );
    }
}

/// Everything from `EDIT:` up to the next section heading.
fn edit_section(text: &str) -> String {
    fn is_heading(line: &str) -> bool {
        line.split(':').next().is_some_and(|head| {
            !head.is_empty()
                && line.contains(':')
                && head.chars().all(|c| c.is_ascii_uppercase() || c == ' ')
        })
    }
    let mut lines = text.lines().skip_while(|line| *line != "EDIT:");
    let heading = lines.next().expect("AGENTS.md must have an EDIT section");
    let body = lines.take_while(|line| !is_heading(line));
    std::iter::once(heading)
        .chain(body)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn edit_is_eleven_verbs_with_visible_signatures() {
    let text = prompt();
    let section = edit_section(&text);
    for verb in [
        "replace S [NEW]",
        "replace-text F OLD [NEW]",
        "replace-lines F A:B [NEW]",
        "replace-span H [NEW]",
        "insert-lines F N [NEW]",
        "delete S",
        "delete-lines F A:B",
        "patch [DIFF]",
        "write PATH [NEW]",
        "rename S NAME",
        "undo [ID]",
    ] {
        assert!(
            section.contains(verb),
            "EDIT must carry `{verb}` with its signature visible; section:\n{section}"
        );
    }
    // The zoo stays dead, the trained vocabulary stays: no invented flags, no
    // nested `edit` prefix, no JSON plans, and no prose addressed at the agent.
    for dead in [
        "ensure-", "change-signature", "apply --plan", "recover", "--content",
        "--target", "--old-file", "greppy edit ", "WHERE", "must occur",
        "data set", "move --file", "remove --file",
    ] {
        assert!(
            !section.contains(dead),
            "`{dead}` must not return to the EDIT section:\n{section}"
        );
    }
}

#[test]
fn the_header_leads_with_the_product() {
    let text = prompt();
    let header: String = text.lines().take(4).collect::<Vec<_>>().join(" ");
    assert!(
        header.contains("holds this repository as a graph"),
        "the identity line names the product, not the fallback; header: {header}"
    );
    assert!(
        header.contains("byte-identical output") && header.contains("grep's exit codes"),
        "the grep promise is stated exactly once, in the header: {header}"
    );
}
