//! TypeScript members are definitions (object-literal arrow members,
//! interface and object-type methods) and JavaScript library calls
//! (`rows.filter(..)`) do not bind to project functions of the same name.

#[path = "support/nav_fixture.rs"]
mod nav_fixture;

use nav_fixture::*;

#[test]
fn typescript_object_literal_members_and_interface_methods_are_definitions() {
    let (repo, store) = indexed("ts-members");
    let (code, out, err) = run(&["search-symbol", "agentNow"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    let interface = format!("web/window.ts:{}", line_of(WINDOW_TS, "agentNow(): string"));
    let literal = format!("web/window.ts:{}", line_of(WINDOW_TS, "agentNow: () =>"));
    let alias = format!("web/window.ts:{}", line_of(WINDOW_TS, "client-member"));
    assert!(
        out.contains(&interface) && out.contains(&literal) && out.contains(&alias),
        "interface, object-type and object-literal members must be found; got:\n{out}"
    );
}

#[test]
fn array_prototype_calls_do_not_bind_to_project_functions_of_that_name() {
    let (repo, store) = indexed("js-builtin");
    let out = who_calls(&repo, &store, "web/window.ts::filter");
    assert!(
        !out.contains("positive"),
        "`rows.filter(..)` is Array.prototype.filter; got:\n{out}"
    );
}
