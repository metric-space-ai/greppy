//! Rust callers are complete: every call site a fix agent must visit is in
//! `who-calls`, and no site is attributed to a target it does not reach --
//! calls in `tokio::select!` and other known macros, receivers behind
//! `Arc<T>`, `T::open(..).await?` or a method's declared return type, trait
//! objects, field reads, several calls in one caller, tests in the same file.

#[path = "support/nav_fixture.rs"]
mod nav_fixture;

use nav_fixture::*;

#[test]
fn who_calls_lists_every_call_site_of_one_caller() {
    let (repo, store) = indexed("sites");
    let out = who_calls(&repo, &store, "src/node.rs::answer_one");
    for marker in ["site-1", "site-2", "site-3"] {
        let row = format!("src/node.rs:{}  dispatch", line_of(NODE_RS, marker));
        assert!(out.contains(&row), "missing {row}; got:\n{out}");
    }

    let (code, out, err) = run(
        &["who-calls", "src/node.rs::answer_one", "--json"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    let value: serde_json::Value = serde_json::from_str(&out).expect("json");
    let hits = value["hits"].as_array().expect("hits");
    assert_eq!(hits.len(), 1, "one caller; got {value}");
    let lines: Vec<u64> = hits[0]["lines"]
        .as_array()
        .expect("lines")
        .iter()
        .filter_map(|line| line.as_u64())
        .collect();
    let expected: Vec<u64> = ["site-1", "site-2", "site-3"]
        .iter()
        .map(|marker| line_of(NODE_RS, marker) as u64)
        .collect();
    assert_eq!(lines, expected, "{value}");
}

#[test]
fn who_calls_sees_calls_and_references_inside_tokio_select() {
    let (repo, store) = indexed("select");
    let out = who_calls(&repo, &store, "src/dock.rs::publish");
    let row = format!("src/node.rs:{}  run", line_of(NODE_RS, "select-publish"));
    assert!(out.contains(&row), "missing {row}; got:\n{out}");

    // `dock` is `Dock::open(..).await?`: the receiver's type is known, and
    // the other package's `Dock::serve` is not the one meant.
    let out = who_calls(&repo, &store, "src/dock.rs::Dock::serve");
    let row = format!("src/node.rs:{}  run", line_of(NODE_RS, "select-serve"));
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
    assert!(!out.contains("other/src"), "{out}");
    // `use wj_dock::Dock;` names the workspace package `wj-dock`.
    let out = who_calls(&repo, &store, "engine/dock/src/lib.rs::Dock::stand_by");
    let row = format!(
        "app/src/lib.rs:{}  wait",
        line_of(APP_LIB_RS, "workspace-import")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
    let out = who_calls(&repo, &store, "other/src/lib.rs::Dock::stand_by");
    assert!(!out.contains("app/src"), "{out}");
    let out = who_calls(&repo, &store, "other/src/lib.rs::Dock::serve");
    let row = format!(
        "other/src/lib.rs:{}  unrelated",
        line_of(OTHER_LIB_RS, "other-serve")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
    assert!(!out.contains("src/node.rs"), "{out}");

    // A function handed to a call inside the macro is referenced there.
    let out = who_calls(&repo, &store, "src/dock.rs::carry_out");
    let row = format!("src/node.rs:{}  run", line_of(NODE_RS, "select-serve"));
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
}

#[test]
fn who_calls_follows_receivers_behind_arc_clones_and_unwrapped_constructors() {
    let (repo, store) = indexed("receivers");
    let out = who_calls(&repo, &store, "src/node.rs::Shown::say");
    for marker in ["say-arc", "say-closure"] {
        let row = format!("src/node.rs:{}  keep", line_of(NODE_RS, marker));
        assert!(out.contains(&row), "missing {row}; got:\n{out}");
    }
    let out = who_calls(&repo, &store, "src/dock.rs::Dock::stand");
    let row = format!("src/node.rs:{}  checked", line_of(NODE_RS, "stand-unwrap"));
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
}

#[test]
fn who_calls_reaches_trait_methods_through_trait_objects() {
    let (repo, store) = indexed("traits");
    // A required trait method is a definition of its own.
    let out = who_calls(&repo, &store, "src/calls.rs::admits");
    let row = format!("src/calls.rs:{}  ask", line_of(CALLS_RS, "admits-dyn"));
    assert!(out.contains(&row), "missing {row}; got:\n{out}");

    // The implementation is called wherever the trait method is.
    let out = who_calls(&repo, &store, "src/blobs.rs::Blobs::admits");
    assert!(out.contains(&row), "missing {row} on the impl; got:\n{out}");

    // A receiver whose type the source does not spell out is a possible
    // caller, listed rather than silently dropped.
    let out = who_calls(&repo, &store, "src/blobs.rs::Blobs::answer");
    let site = format!(
        "src/calls.rs:{} route",
        line_of(CALLS_RS, "answer-unproven")
    );
    assert!(
        out.contains(&site) && out.contains("receiver type not proven"),
        "missing possible caller {site}; got:\n{out}"
    );
    // A type that only inherits the trait's default method, and a generic
    // bounded by the trait, can only reach the trait's declaration.
    let out = who_calls(&repo, &store, "src/calls.rs::Greets::greet");
    for (marker, caller) in [
        ("default-method-call", "welcome"),
        ("generic-bound-call", "welcome_any"),
    ] {
        let site = format!(
            "possible caller: src/calls.rs:{} {caller}",
            line_of(CALLS_RS, marker)
        );
        assert!(out.contains(&site), "missing {site}; got:\n{out}");
    }
    // `impact` follows proven edges only, and says that more may exist.
    let (code, out, err) = run(&["impact", "src/blobs.rs::Blobs::answer"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("possible call sites whose receiver type is not proven"),
        "{out}"
    );
}

#[test]
fn field_reads_are_found_across_crates_in_closures_and_in_macros() {
    let (repo, store) = indexed("fields");
    // `rights` is what `Dock::rights_of` returns: its declared type names
    // the owner; the free function `may_lead` is not the field.
    let out = who_calls(&repo, &store, "src/dock.rs::Rights::may_lead");
    let row = format!(
        "src/node.rs:{}  admitted",
        line_of(NODE_RS, "may-lead-read")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
    let out = who_calls(&repo, &store, "src/node.rs::may_lead");
    assert!(
        !out.contains("admitted"),
        "a field read is not a call; got:\n{out}"
    );

    // A closure parameter has no spelled type: the read is listed as a
    // possible use, never as a proven one.
    let out = who_calls(&repo, &store, "src/calls.rs::SourceState::next_page");
    let row = format!(
        "possible use: src/calls.rs:{} resume",
        line_of(CALLS_RS, "next-page-closure")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");

    let out = who_calls(&repo, &store, "src/calls.rs::Source::article_type");
    let row = format!(
        "src/calls.rs:{}  article",
        line_of(CALLS_RS, "article-type-macro")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
    // `let source = match check_source() { Ok(source) => source, .. }`.
    let row = format!(
        "src/calls.rs:{}  listed",
        line_of(CALLS_RS, "article-type-match")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
}

#[test]
fn closure_and_loop_bindings_are_not_callers() {
    let (repo, store) = indexed("bindings");
    let out = who_calls(&repo, &store, "src/calls.rs::Folder::secret");
    let row = format!(
        "src/calls.rs:{}  Folder::keep",
        line_of(CALLS_RS, "secret-call")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");
    let closure = line_of(CALLS_RS, "|(name, secret)|");
    assert!(
        !out.contains(&format!("src/calls.rs:{closure} ")),
        "a closure parameter is not a call; got:\n{out}"
    );
    let loop_pattern = line_of(CALLS_RS, "for (name, secret)");
    assert!(
        !out.contains(&format!("src/calls.rs:{loop_pattern} ")),
        "a loop pattern is not a call; got:\n{out}"
    );
}

#[test]
fn who_calls_includes_tests_in_the_same_file() {
    let (repo, store) = indexed("same-file-test");
    let out = who_calls(&repo, &store, "src/calls.rs::loopback");
    let row = format!("src/calls.rs:{}  ", line_of(CALLS_RS, "loopback-test"));
    assert!(
        out.lines()
            .any(|line| line.starts_with(&row) && line.contains("loops")),
        "missing {row}loops; got:\n{out}"
    );
}

#[test]
fn standard_library_receivers_do_not_bind_to_project_types_of_that_name() {
    let (repo, store) = indexed("std-path");
    let out = who_calls(&repo, &store, "src/store_path.rs::Path::join");
    let own = format!(
        "src/store_path.rs:{}  own",
        line_of(STORE_PATH_RS, "own-join")
    );
    assert!(out.contains(&own), "missing {own}; got:\n{out}");
    assert!(
        !out.contains("src/fs.rs"),
        "`folder: &std::path::Path` does not call the project's Path::join; got:\n{out}"
    );

    // `use rusqlite::Connection;` names another crate's type, not the
    // project's `Connection`.
    let out = who_calls(&repo, &store, "src/store_path.rs::Connection::transaction");
    let own = format!(
        "src/store_path.rs:{}  begin",
        line_of(STORE_PATH_RS, "own-transaction")
    );
    assert!(out.contains(&own), "missing {own}; got:\n{out}");
    assert!(!out.contains("src/fs.rs"), "{out}");
}

#[test]
fn who_calls_finds_associated_function_calls_imported_from_another_package() {
    let (repo, store) = indexed("workspace-assoc");
    // `app/src/lib.rs` imports `use wj_dock::{Dock, Stream};` and calls
    // `Dock::open(..)`: only `wj-dock`'s `Dock::open` gets the caller, not
    // the same-named `Dock::open` in `other` or the root package.
    let out = who_calls(&repo, &store, "engine/dock/src/lib.rs::Dock::open");
    let row = format!(
        "app/src/lib.rs:{}  start",
        line_of(APP_LIB_RS, "workspace-open")
    );
    assert!(out.contains(&row), "missing {row}; got:\n{out}");

    let out = who_calls(&repo, &store, "other/src/lib.rs::Dock::open");
    assert!(
        !out.contains("app/src"),
        "unimported package's Dock::open must not get the call; got:\n{out}"
    );
    let out = who_calls(&repo, &store, "src/dock.rs::Dock::open");
    assert!(
        !out.contains("app/src"),
        "root package's Dock::open must not get the call; got:\n{out}"
    );
}
