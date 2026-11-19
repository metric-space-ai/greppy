//! Navigation answers say what they mean: an ambiguous name names each
//! definition with its qualified spelling, `brief` keeps doc comments as
//! written, kinds are not read from a return type, `trace` refuses a
//! positional start, and `plus --path` scopes its hits.

#[path = "support/nav_fixture.rs"]
mod nav_fixture;

use nav_fixture::*;

#[test]
fn a_file_qualified_name_with_two_owners_is_ambiguous() {
    let (repo, store) = indexed("ambiguous-new");
    let (code, out, err) = run(&["who-calls", "src/calls.rs::new"], &repo, &store);
    assert_eq!(code, 1, "must refuse; stderr={err}\nstdout={out}");
    assert!(
        out.contains("is 2 definitions")
            && out.contains("src/calls.rs::Calls::new")
            && out.contains("src/calls.rs::Call::new"),
        "must name each definition; got:\n{out}"
    );
    let out = who_calls(&repo, &store, "src/calls.rs::Calls::new");
    assert!(out.contains("make"), "{out}");

    let (code, out, err) = run(&["who-calls", "src/calls.rs::new", "--json"], &repo, &store);
    assert_ne!(code, 0, "json must refuse too; stderr={err}\nstdout={out}");
    assert!(
        format!("{out}{err}").contains("src/calls.rs::Calls::new"),
        "stdout={out}\nstderr={err}"
    );
}

#[test]
fn an_ambiguous_bare_name_prints_the_qualified_spelling() {
    let (repo, store) = indexed("ambiguous-text");
    let (code, out, err) = run(&["who-calls", "dup"], &repo, &store);
    assert_eq!(code, 1, "must refuse; stderr={err}\nstdout={out}");
    assert!(
        out.contains("src/brief.rs::dup") && out.contains("src/fs.rs::dup"),
        "text mode must print the <path>::<name> spelling; got:\n{out}"
    );
}

#[test]
fn brief_keeps_multi_line_doc_comments_and_jsdoc_as_written() {
    let (repo, store) = indexed("brief-docs");
    let (code, out, err) = run(&["brief", "run_trigger"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("(aborted when it no longer does or the app changes)."),
        "the whole doc comment is the purpose; got:\n{out}"
    );
    let fn_line = line_of(BRIEF_RS, "pub async fn run_trigger");
    assert!(
        out.contains(&format!("src/brief.rs:{fn_line}")),
        "the head starts at the definition, not inside its doc; got:\n{out}"
    );

    let (code, out, err) = run(&["brief", "engineModelWanted"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("Throws, in words, if it cannot be asked with that model."),
        "the JSDoc is the purpose, as written; got:\n{out}"
    );
}

#[test]
fn an_async_fn_returning_a_static_str_is_not_a_static() {
    let (repo, store) = indexed("kind-label");
    let (code, out, err) = run(&["search-symbol", "audit"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    let row = out
        .lines()
        .find(|line| line.contains("src/brief.rs"))
        .unwrap_or_else(|| panic!("audit missing; got:\n{out}"));
    assert!(!row.contains("static"), "{row}");
}

#[test]
fn trace_without_symbol_flag_is_an_error() {
    let (repo, store) = indexed("trace");
    let (code, out, err) = run(&["trace", "src/node.rs::dispatch"], &repo, &store);
    assert_eq!(code, 64, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("--symbol") && !out.contains("ignoring"),
        "must refuse and name --symbol; got:\n{out}"
    );
    let (code, out, err) = run(&["trace"], &repo, &store);
    assert_eq!(code, 64, "stderr={err}\nstdout={out}");
}

#[test]
fn plus_honours_path_filters() {
    let (repo, store) = indexed("plus-path");
    let (code, out, err) = run(
        &["plus", "admits", "--path", "src/blobs.rs", "--k", "20"],
        &repo,
        &store,
    );
    assert!(code == 0 || code == 1, "stderr={err}\nstdout={out}");
    assert!(out.contains("src/blobs.rs"), "{out}");
    assert!(
        out.lines()
            .filter(|line| line.contains(".rs:") || line.contains(".ts:"))
            .all(|line| line.starts_with("src/blobs.rs")),
        "only hits under --path; got:\n{out}"
    );
}
