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
    let heading = lines
        .next()
        .expect("AGENTS.md must have a NAVIGATE section");
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
        .filter(|line| {
            line.starts_with("  ") && !line.starts_with("    ") && !line.starts_with("  --")
        })
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
                "search:",
                "search-symbol:",
                "search-pattern:",
                "who-calls:",
                "callees:",
                "brief:",
                "impact:",
                "path:",
                "read:",
                "read-smart:",
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
    for dead in [
        "map [PATH]",
        "outline PATH",
        "verify -- CMD",
        "\n  changes ",
    ] {
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
        "ensure-",
        "change-signature",
        "apply --plan",
        "recover",
        "--content",
        "--target",
        "--old-file",
        "greppy edit ",
        "WHERE",
        "must occur",
        "data set",
        "move --file",
        "remove --file",
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

/// The eleven assertions above hold CONCEPTS: they catch a return to a retired
/// idea, not an edit that stays inside the ideas. That is how a CHAIN example
/// was silently changed while every guard stayed green. This one holds the
/// FILE: any byte that moves breaks it, and updating the constant below is the
/// owner's signature on the change.
#[test]
fn the_prompt_is_frozen_byte_for_byte() {
    use std::fmt::Write as _;

    const APPROVED_SHA256: &str =
        "9972c50e5fdc82cb40a6a62db92a7be763a128f14f7c71b97d8c32462b33dbb8";

    let text = prompt();
    let digest = {
        // Tiny SHA-256 so the guard needs no dependency the test tree lacks.
        let mut state: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut message = text.as_bytes().to_vec();
        let bit_len = (message.len() as u64) * 8;
        message.push(0x80);
        while message.len() % 64 != 56 {
            message.push(0);
        }
        message.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in message.chunks(64) {
            let mut w = [0u32; 64];
            for (index, word) in chunk.chunks(4).enumerate() {
                w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for index in 16..64 {
                let s0 = w[index - 15].rotate_right(7)
                    ^ w[index - 15].rotate_right(18)
                    ^ (w[index - 15] >> 3);
                let s1 = w[index - 2].rotate_right(17)
                    ^ w[index - 2].rotate_right(19)
                    ^ (w[index - 2] >> 10);
                w[index] = w[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[index - 7])
                    .wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
            for index in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[index])
                    .wrapping_add(w[index]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
                *slot = slot.wrapping_add(value);
            }
        }
        let mut hex = String::new();
        for word in state {
            let _ = write!(hex, "{word:08x}");
        }
        hex
    };

    assert_eq!(
        digest, APPROVED_SHA256,
        "AGENTS.md changed. The system prompt is the product's contract and is \
         frozen: if this change is intended, the owner approves it by updating \
         APPROVED_SHA256 in this test. If it is not, revert the file."
    );
}
