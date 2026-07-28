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
    for verb in ["who-calls S", "callees S", "brief S", "impact S", "path --from A --to B"] {
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
        described, 6,
        "NAVIGATE has five commands plus the multi-symbol note; got {described} entries:\n{section}"
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
