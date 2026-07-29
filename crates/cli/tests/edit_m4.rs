//! The M4 edit grammar is retired. Its three suites pinned `greppy edit
//! change-signature`, `greppy edit data set/ensure` and the LSP backend spec —
//! all deliberately killed by the 0.3.0 EDIT redesign (dead-listed in the
//! prompt contract; trace counts showed zero real use). What replaced them is
//! pinned by edit_family.rs. This file pins the retirement: dead greppy
//! vocabulary is REFUSED before grep passthrough (unknown_verb_refusal) — an
//! agent with a stale habit learns immediately instead of getting garbage
//! grep matches for `edit` as a pattern.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

#[test]
fn the_retired_edit_grammar_is_refused_not_grepped() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("edit-m4-retired");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("f.txt"), "edit me\n").unwrap();

    for tail in [
        &["edit", "change-signature", "foo", "--spec", "x"][..],
        &["edit", "data", "set", "$.a", "1"][..],
        &["edit", "text", "f.txt"][..],
    ] {
        let out = Command::new(bin())
            .current_dir(&dir)
            .args(tail)
            .output()
            .expect("run greppy");
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(64),
            "`greppy {}` must refuse as invalid vocabulary; got: {text}",
            tail.join(" ")
        );
        assert!(
            text.contains("unrecognized subcommand 'edit'"),
            "the refusal names the dead verb; got: {text}"
        );
        assert!(
            !text.contains("applied") && !text.contains("f.txt:"),
            "the refusal neither edits nor greps; got: {text}"
        );
    }
}
