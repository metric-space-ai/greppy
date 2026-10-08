//! One minimal two-function fixture per language that previously emitted no
//! call edges (or attributed them to the file module). Each case must define
//! both functions, emit one CALLS edge from the enclosing caller, and give
//! that caller a multi-line span that covers the call.

use std::path::Path;

use greppy_parser::{extract, language_for_path, ExtractionResult};

struct Case {
    /// Path under `tests/fixtures/`, also used as the extraction file path so
    /// qualified names stay stable.
    file: &'static str,
    caller: &'static str,
    callee: &'static str,
    /// When set, the caller must also USAGE-reference this name (Haskell
    /// `map f`: the argument is a use, not only the callee).
    arg_use: Option<&'static str>,
}

#[test]
fn minimal_two_function_calls() {
    let cases = [
        Case {
            file: "calls/powershell/calls.ps1",
            caller: "Caller",
            callee: "Helper",
            arg_use: None,
        },
        Case {
            file: "calls/cmake/calls.cmake",
            caller: "caller",
            callee: "helper",
            arg_use: None,
        },
        Case {
            file: "calls/erlang/calls.erl",
            caller: "caller",
            callee: "helper",
            arg_use: None,
        },
        Case {
            file: "calls/elm/Calls.elm",
            caller: "caller",
            callee: "helper",
            arg_use: None,
        },
        Case {
            file: "calls/pascal/calls.pas",
            caller: "Caller",
            callee: "Helper",
            arg_use: None,
        },
        Case {
            file: "calls/commonlisp/calls.lisp",
            caller: "caller",
            callee: "helper",
            arg_use: None,
        },
        Case {
            file: "calls/ada/calls.adb",
            caller: "Caller",
            callee: "Helper",
            arg_use: None,
        },
        Case {
            file: "calls/haskell/Calls.hs",
            caller: "caller",
            callee: "helper",
            arg_use: Some("helper"),
        },
        Case {
            file: "calls/fortran/calls.f90",
            caller: "caller",
            callee: "helper",
            arg_use: None,
        },
    ];

    for case in cases {
        let disk = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(case.file);
        let source =
            std::fs::read(&disk).unwrap_or_else(|err| panic!("read {}: {err}", disk.display()));
        let language = language_for_path(Path::new(case.file));
        let result = extract(language, &source, case.file)
            .unwrap_or_else(|err| panic!("{}: extract failed: {err}", case.file));
        assert_function(&result, case.file, case.caller);
        assert_function(&result, case.file, case.callee);
        assert_call(&result, case.file, case.caller, case.callee);
        assert_caller_spans_call(&result, case.file, case.caller);
        if let Some(arg) = case.arg_use {
            assert_arg_use(&result, case.file, case.caller, arg);
        }
    }
}

fn assert_function(result: &ExtractionResult, file: &str, name: &str) {
    assert!(
        result
            .nodes
            .iter()
            .any(|node| node.label == "Function" && node.name == name),
        "{file}: missing Function {name}\nnodes: {:#?}",
        result.nodes
    );
}

fn assert_call(result: &ExtractionResult, file: &str, caller: &str, callee: &str) {
    let source = format!("{file}::Function::{caller}");
    let function_suffix = format!("::Function::{callee}");
    let callee_suffix = format!("::__callee__::{callee}");
    let found = result.edges.iter().any(|edge| {
        if edge.edge_type != "CALLS" || edge.source_qualified_name != source {
            return false;
        }
        let by_name = edge.properties.get("callee_name").and_then(|v| v.as_str()) == Some(callee);
        let by_target = edge.target_qualified_name.ends_with(&function_suffix)
            || edge.target_qualified_name.ends_with(&callee_suffix);
        by_name || by_target
    });
    assert!(
        found,
        "{file}: missing CALLS {caller} -> {callee}\nedges: {:#?}",
        result.edges
    );
}

fn assert_caller_spans_call(result: &ExtractionResult, file: &str, caller: &str) {
    let source = format!("{file}::Function::{caller}");
    let node = result
        .nodes
        .iter()
        .find(|node| node.label == "Function" && node.name == caller)
        .unwrap_or_else(|| panic!("{file}: missing caller node"));
    assert!(
        node.end_line > node.start_line,
        "{file}: {caller} span is signature-only ({}, {})",
        node.start_line,
        node.end_line
    );
    let call = result
        .edges
        .iter()
        .find(|edge| edge.edge_type == "CALLS" && edge.source_qualified_name == source);
    let Some(call) = call else {
        return;
    };
    assert!(
        call.line >= node.start_line && call.line <= node.end_line,
        "{file}: call line {} outside {caller} span {}-{}",
        call.line,
        node.start_line,
        node.end_line
    );
}

fn assert_arg_use(result: &ExtractionResult, file: &str, caller: &str, name: &str) {
    let source = format!("{file}::Function::{caller}");
    let found = result.edges.iter().any(|edge| {
        edge.edge_type == "USAGE"
            && edge.source_qualified_name == source
            && edge.properties.get("ref_name").and_then(|v| v.as_str()) == Some(name)
    });
    assert!(
        found,
        "{file}: missing USAGE of argument {name} from {caller}\nedges: {:#?}",
        result.edges
    );
}
