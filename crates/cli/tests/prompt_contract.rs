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
    // The product and one-shot routing lead: the first thing read is which
    // greppy command answers the question, not how to fall back to grep.
    assert!(
        header.contains("greppy") && !header.starts_with("grep "),
        "the header leads with the product, not the fallback; header: {header}"
    );
    assert!(
        header.contains("Default to ONE compact"),
        "one-shot routing is the first thing the reader meets; header: {header}"
    );
    assert!(
        text.contains("Do NOT run grep/find/read loops"),
        "the fallback prohibition remains part of the prompt contract"
    );
    assert!(
        text.contains("holds this repository as a graph"),
        "the identity line names the product, not the fallback"
    );
    assert!(
        text.contains("byte-identical output") && text.contains("grep's exit codes"),
        "the grep promise is stated exactly once"
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

    // 05.08.2026: owner approved the AGENT: section (greppy -p delegation
    // block) for 0.3.1 — wording frozen in the same review.
    // 08.08.2026: owner approved three further changes in one review — the
    // navigation directive moved ahead of the identity line (880f0eb), the
    // merge that carried the AGENT block into this line (7cc8bb5), and the
    // INDEX block, which names the one command an answer can now ask for
    // ("run `greppy index .` first") and would otherwise be unexplained.
    // 22.08.2026: the owner approved the 0.3.2 one-shot routing contract:
    // named definition/caller/impact/path questions go directly to one
    // command and stop when its answer is sufficient.
    // 22.08.2026: release-gate forensics approved compact graph-first routes;
    // source text is requested only when the question needs body evidence.
    // 23.08.2026: repeated exact-commit release evidence approved one ranked
    // semantic candidate set and forbids paraphrased semantic retry loops.
    // 24.08.2026: the 0.3.2 coding gate approved one precise Greppy edit with
    // verification and no redundant reread after a successful edit.
    // 30.08.2026: the BROWSER section was added, and it names the WHOLE
    // browser surface, not only what ships today. That is deliberate: the
    // prompt is the contract, and
    // browser_section_names_only_existing_web_subcommands is the work list
    // that fails until the last of those subcommands exists. Owner decision.
    // Owner-approved 2026-08-31: the browser block moved out of the shipped
    // prompt into assets/prompts/web-beta.md while its command surface was
    // still moving. AGENTS.md kept only a pointer during that stabilization.
    // 01.09.2026 (merge of 0.3.4 main and the web branch): the frozen text is
    // now the plain union of two individually owner-approved states - main's
    // 26.08 bash-smart execution contract wording plus the branch's 31.08
    // web-beta pointer. No new sentence was written for the merge; the hash
    // moves mechanically to the union.
    // 01.09.2026: a reproduced usability report showed that NAVIGATE's
    // generic --code footer incorrectly included `path`, whose stable surface
    // is deliberately a bounded call-site tree. The owner-approved correction
    // names the exception and the exact `read` recovery instead of adding a
    // new path output mode to the 0.3.x stability line.
    // 03.09.2026: the owner approved the stabilized browser block for the
    // 0.4.0 public prompt. AGENTS.md now enables it for external agents while
    // the built-in agent includes the byte-identical canonical asset.
    const APPROVED_SHA256: &str =
        "f9957e033efbd4ec40a038e150a52d2252ca1969648e97dc96d1e8ebd62ba0c1";

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

// ---------------------------------------------------------------------------
// BROWSER section guard.
//
// The BROWSER section is written first and is the contract: it names the whole
// browser surface, and the CLI is finished when every line in it resolves to a
// real subcommand. This test is therefore a WORK LIST, not a scolding — it
// fails until the last verb ships, and it names exactly what is left.
//
// The direction is deliberate. `find-usages` once advertised behaviour the code
// did not have and nobody noticed, because nothing compared prose to the binary.
// Here the comparison is a test, so the gap is visible on every run instead of
// being discovered by an agent in the field.

/// Every `greppy web WORD` the prompt shows as a COMMAND, as the bare WORD.
///
/// Only indented command lines count — the house style writes commands
/// indented under a heading. Prose like "use greppy web for every web step"
/// must not be read as a subcommand named `for`.
fn web_subcommands_named_in(text: &str) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if line == trimmed || !trimmed.starts_with("greppy web ") {
            continue;
        }
        let rest = &trimmed["greppy web ".len()..];
        let word: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
            .collect();
        if !word.is_empty() {
            out.insert(word);
        }
    }
    out
}

/// The subcommand table the binary really has, read from its own help.
fn web_subcommands_from_help() -> std::collections::BTreeSet<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["web", "--help"])
        .output()
        .expect("greppy web --help must run");
    let text = String::from_utf8_lossy(&output.stdout);
    let mut out = std::collections::BTreeSet::new();
    let mut in_commands = false;
    for line in text.lines() {
        if line.starts_with("Commands:") {
            in_commands = true;
            continue;
        }
        if in_commands {
            if line.trim().is_empty() || !line.starts_with("  ") {
                break;
            }
            if let Some(word) = line.split_whitespace().next() {
                out.insert(word.to_owned());
            }
        }
    }
    assert!(
        !out.is_empty(),
        "could not read the web subcommand table from --help:\n{text}"
    );
    out
}

/// The canonical browser portion of both shipped prompt surfaces.
fn beta_web_prompt() -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/prompts/web-beta.md")
        .canonicalize()
        .ok()?;
    std::fs::read_to_string(path).ok()
}

#[test]
fn browser_section_names_only_existing_web_subcommands() {
    // The canonical browser block names only commands the binary ships.
    let Some(text) = beta_web_prompt() else {
        return; // beta prompt not present; nothing to guard
    };
    if !text.contains("BROWSER") {
        return; // section not written yet; nothing to guard
    }
    let named = web_subcommands_named_in(&text);
    assert!(
        !named.is_empty(),
        "a BROWSER section that names no `greppy web` command is not a prompt"
    );
    let real = web_subcommands_from_help();
    let invented: Vec<_> = named.difference(&real).cloned().collect();
    assert!(
        invented.is_empty(),
        "BROWSER beta work list — {} of {} advertised commands are still missing.\n\
         \n\
         missing: {invented:?}\n\
         \n\
         shipped: {real:?}\n\
         \n\
         The prompt is the contract and was written first on purpose. This test\n\
         goes green when the last of these subcommands exists in `greppy web\n\
         --help`. Do not shorten the prompt to make it pass.",
        invented.len(),
        named.len()
    );
}

fn delimited_browser_block(text: &str) -> Option<&str> {
    let start = if text.starts_with("BROWSER:") {
        0
    } else {
        text.find("\nBROWSER:")? + 1
    };
    let tail = &text[start..];
    let end = tail.find("END BROWSER")? + "END BROWSER".len();
    Some(&tail[..end])
}

#[test]
fn public_and_builtin_browser_prompts_are_byte_identical() {
    let public = prompt();
    let beta = beta_web_prompt().expect("canonical browser prompt must ship");
    assert_eq!(
        delimited_browser_block(&public),
        delimited_browser_block(&beta),
        "AGENTS.md and assets/prompts/web-beta.md must expose one browser contract"
    );
}

#[test]
fn browser_section_is_delimited_so_edits_are_visible() {
    let text = prompt();
    if !text.contains("BROWSER:") {
        return;
    }
    assert!(
        text.contains("BROWSER:") && text.contains("END BROWSER"),
        "the BROWSER section must stay between its markers, so an accidental \
         edit is visible in the diff instead of silently reshaping the prompt"
    );
}

#[test]
fn browser_section_marks_page_text_as_untrusted() {
    let text = prompt();
    if !text.contains("BROWSER:") {
        return;
    }
    let lowered = text.to_lowercase();
    assert!(
        lowered.contains("untrusted"),
        "the BROWSER section must tell the agent that page text is untrusted \
         input; a browser prompt without that line is a prompt-injection hole"
    );
}
