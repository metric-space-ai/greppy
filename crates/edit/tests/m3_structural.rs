//! M3 structural/scoped edit contract tests.

use std::path::{Path, PathBuf};

use greppy_edit::certificate::{SelectorClass, SelectorEngine};
use greppy_edit::ensure::{
    ensure_annotation, ensure_argument, ensure_import, ensure_method, remove_if_present,
};
use greppy_edit::verbs::{regex_cas, rename_in_span, VerbOptions};
use greppy_edit::{Certificate, EditHandle, Language, Status};

fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn write(root: &Path, name: &str, content: &[u8]) -> PathBuf {
    let path = root.join(name);
    std::fs::write(&path, content).unwrap();
    path
}

fn planned_options(root: &Path, name: &str, content: &[u8], range: (usize, usize)) -> VerbOptions {
    let handle = EditHandle::for_range(root, Path::new(name), content, range.0, range.1).unwrap();
    VerbOptions {
        planned_file_sha256: Some(handle.file_sha256),
        planned_target_sha256: Some(handle.target_sha256),
        planned_target_range: Some(range),
        ..Default::default()
    }
}

fn compact(certificate: &Certificate) -> serde_json::Value {
    let value: serde_json::Value =
        serde_json::from_str(&certificate.to_compact_json_pretty().unwrap()).unwrap();
    assert_eq!(value["exit_code"], certificate.exit_code());
    assert!(value.get("validators").is_none());
    for operation in value["operations"].as_array().unwrap() {
        assert!(operation.get("node_before").is_none());
        assert!(operation.get("node_after").is_none());
        for postcondition in operation["postconditions"].as_array().unwrap() {
            assert!(postcondition.get("detail").is_none());
        }
    }
    if certificate.status == Status::Applied {
        assert!(value["operations"][0]["unified_diff"].is_string());
    }
    value
}

fn python_def_range(content: &[u8], name: &str) -> (usize, usize) {
    let text = std::str::from_utf8(content).unwrap();
    let start = text.find(&format!("def {name}(")).unwrap();
    let end = text[start + 1..]
        .find("\ndef ")
        .map(|offset| start + 1 + offset)
        .unwrap_or(content.len());
    (start, end)
}

#[test]
fn rename_call_is_scoped_to_calls_and_idempotent() {
    let ws = workspace();
    let original = b"def run():\n    old()\n    value = old\n\ndef other():\n    old()\n";
    let file = write(ws.path(), "m.py", original);
    let range = python_def_range(original, "run");

    let first = rename_in_span(
        ws.path(),
        &file,
        range,
        "old",
        "new",
        Some(1),
        Language::Python,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(first.status, Status::Applied);
    assert_eq!(first.operations[0].target_matches, 1);
    assert_eq!(
        first.operations[0].selector_engine,
        SelectorEngine::TreeSitter
    );
    assert_eq!(
        first.operations[0].selector_class,
        SelectorClass::Structural
    );
    compact(&first);

    let after_first = std::fs::read(&file).unwrap();
    let text = std::str::from_utf8(&after_first).unwrap();
    assert!(text.contains("def run():\n    new()"), "{text}");
    assert!(text.contains("value = old"), "{text}");
    assert!(text.contains("def other():\n    old()"), "{text}");

    let second = rename_in_span(
        ws.path(),
        &file,
        range,
        "old",
        "new",
        Some(1),
        Language::Python,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(second.status, Status::AlreadySatisfied);
    assert_eq!(second.exit_code(), 0);
    assert_eq!(std::fs::read(&file).unwrap(), after_first);
    compact(&second);
}

#[test]
fn rename_call_expect_mismatch_is_atomic() {
    let ws = workspace();
    let original = b"def run():\n    old()\n    old()\n";
    let file = write(ws.path(), "m.py", original);
    let certificate = rename_in_span(
        ws.path(),
        &file,
        (0, original.len()),
        "old",
        "new",
        Some(1),
        Language::Python,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(certificate.status, Status::Ambiguous);
    assert_eq!(certificate.exit_code(), 11);
    assert_eq!(certificate.operations[0].target_matches, 2);
    assert_eq!(std::fs::read(&file).unwrap(), original);
    compact(&certificate);
}

#[test]
fn rename_call_honors_planned_hashes() {
    let ws = workspace();
    let original = b"def run():\n    old()\n";
    let file = write(ws.path(), "m.py", original);
    let range = (0, original.len());
    let options = planned_options(ws.path(), "m.py", original, range);
    let live = b"# concurrent edit\ndef run():\n    old()\n";
    std::fs::write(&file, live).unwrap();
    let certificate = rename_in_span(
        ws.path(),
        &file,
        range,
        "old",
        "new",
        Some(1),
        Language::Python,
        &options,
    )
    .unwrap();
    assert_eq!(certificate.status, Status::Stale);
    assert_eq!(certificate.exit_code(), 12);
    assert_eq!(
        certificate.operations[0].selector_engine,
        SelectorEngine::TreeSitter
    );
    assert_eq!(
        certificate.operations[0].selector_class,
        SelectorClass::Structural
    );
    assert_eq!(std::fs::read(&file).unwrap(), live);
    compact(&certificate);
}

#[test]
fn regex_cas_is_repeat_safe_and_weakly_classified() {
    let ws = workspace();
    let original = b"port = 9000\nhost = local\n";
    let file = write(ws.path(), "app.ini", original);
    let first = regex_cas(
        ws.path(),
        &file,
        r"(?m)^port\s*=\s*\d+$",
        "port = 8080",
        1,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(first.status, Status::Applied);
    assert_eq!(first.operations[0].selector_engine, SelectorEngine::Regex);
    assert_eq!(first.operations[0].selector_class, SelectorClass::RegexWeak);
    compact(&first);

    let after_first = std::fs::read(&file).unwrap();
    let second = regex_cas(
        ws.path(),
        &file,
        r"(?m)^port\s*=\s*\d+$",
        "port = 8080",
        1,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(second.status, Status::AlreadySatisfied);
    assert_eq!(second.operations[0].target_matches, 1);
    assert_eq!(std::fs::read(&file).unwrap(), after_first);
    compact(&second);
}

#[test]
fn regex_cas_expect_mismatch_reports_actual_count() {
    let ws = workspace();
    let original = b"port = 9000\nport = 9001\n";
    let file = write(ws.path(), "app.ini", original);
    let certificate = regex_cas(
        ws.path(),
        &file,
        r"(?m)^port\s*=\s*\d+$",
        "port = 8080",
        1,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(certificate.status, Status::Ambiguous);
    assert_eq!(certificate.exit_code(), 11);
    assert_eq!(certificate.operations[0].target_matches, 2);
    assert_eq!(
        certificate.operations[0].selector_class,
        SelectorClass::RegexWeak
    );
    assert_eq!(std::fs::read(&file).unwrap(), original);
    compact(&certificate);
}

#[test]
fn regex_cas_honors_planned_hashes() {
    let ws = workspace();
    let original = b"port = 9000\n";
    let file = write(ws.path(), "app.ini", original);
    let options = planned_options(ws.path(), "app.ini", original, (0, original.len()));
    let live = b"port = 9001\n";
    std::fs::write(&file, live).unwrap();
    let certificate = regex_cas(
        ws.path(),
        &file,
        r"(?m)^port\s*=\s*\d+$",
        "port = 8080",
        1,
        &options,
    )
    .unwrap();
    assert_eq!(certificate.status, Status::Stale);
    assert_eq!(
        certificate.operations[0].selector_engine,
        SelectorEngine::Regex
    );
    assert_eq!(
        certificate.operations[0].selector_class,
        SelectorClass::RegexWeak
    );
    assert_eq!(std::fs::read(&file).unwrap(), live);
    compact(&certificate);
}

#[test]
fn ensure_import_is_idempotent_with_compact_counts() {
    let ws = workspace();
    let original = b"def run():\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let first = ensure_import(
        ws.path(),
        &file,
        "auth.validators",
        Some("validate"),
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(first.status, Status::Applied);
    compact(&first);

    let after_first = std::fs::read(&file).unwrap();
    let second = ensure_import(
        ws.path(),
        &file,
        "auth.validators",
        Some("validate"),
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(second.status, Status::AlreadySatisfied);
    assert_eq!(second.operations[0].target_matches, 1);
    assert_eq!(std::fs::read(&file).unwrap(), after_first);
    compact(&second);
}

#[test]
fn ensure_import_conflict_and_stale_are_atomic() {
    let ws = workspace();
    let conflict = b"from other.module import validate\n";
    let conflict_file = write(ws.path(), "conflict.py", conflict);
    let certificate = ensure_import(
        ws.path(),
        &conflict_file,
        "auth.validators",
        Some("validate"),
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(certificate.status, Status::InvalidResult);
    assert_eq!(std::fs::read(&conflict_file).unwrap(), conflict);
    compact(&certificate);

    let original = b"def run():\n    pass\n";
    let stale_file = write(ws.path(), "stale.py", original);
    let options = planned_options(ws.path(), "stale.py", original, (0, original.len()));
    let live = b"# concurrent\ndef run():\n    pass\n";
    std::fs::write(&stale_file, live).unwrap();
    let stale = ensure_import(ws.path(), &stale_file, "os", None, &options).unwrap();
    assert_eq!(stale.status, Status::Stale);
    assert_eq!(
        stale.operations[0].selector_class,
        SelectorClass::Structural
    );
    assert_eq!(std::fs::read(&stale_file).unwrap(), live);
    compact(&stale);
}

#[test]
fn ensure_import_inserts_go_module_inside_existing_group() {
    let ws = workspace();
    let original =
        b"package main\n\nimport (\n\t\"fmt\"\n)\n\nfunc main() { fmt.Println(\"ok\") }\n";
    let file = write(ws.path(), "m.go", original);

    let certificate = ensure_import(
        ws.path(),
        &file,
        "time",
        Some("time"),
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(certificate.status, Status::Applied);
    assert_eq!(certificate.exit_code(), 0);
    assert!(certificate.published);
    assert_eq!(certificate.operations[0].syntax.new_errors, 0);
    assert_eq!(certificate.operations[0].syntax.new_missing_nodes, 0);
    let changed = std::fs::read_to_string(&file).unwrap();
    assert!(
        changed.contains("import (\n\t\"fmt\"\n\t\"time\"\n)"),
        "{changed}"
    );
    assert!(!changed.contains(")\nimport \"time\""), "{changed}");
}

#[test]
fn ensure_import_inserts_names_inside_grouped_imports() {
    let ws = workspace();
    let cases: &[(&str, &[u8], &str, &str, &str)] = &[
        (
            "m.py",
            b"from auth import (\n    check,\n)\n\ndef run():\n    pass\n",
            "auth",
            "validate",
            "from auth import (\n    check,\n    validate,\n)",
        ),
        (
            "m.rs",
            b"use crate::auth::{\n    check,\n};\n\nfn main() {}\n",
            "crate::auth",
            "validate",
            "use crate::auth::{\n    check,\n    validate,\n};",
        ),
        (
            "m.ts",
            b"import {\n  check,\n} from \"auth\";\n\ncheck();\n",
            "auth",
            "validate",
            "import {\n  check,\n  validate,\n} from \"auth\";",
        ),
    ];

    for &(path, original, module, name, expected) in cases {
        let file = write(ws.path(), path, original);
        let certificate = ensure_import(
            ws.path(),
            &file,
            module,
            Some(name),
            &VerbOptions::default(),
        )
        .unwrap();

        assert_eq!(certificate.status, Status::Applied, "{path}");
        assert_eq!(certificate.exit_code(), 0, "{path}");
        assert_eq!(certificate.operations[0].syntax.new_errors, 0, "{path}");
        assert_eq!(
            certificate.operations[0].syntax.new_missing_nodes, 0,
            "{path}"
        );
        let changed = std::fs::read_to_string(file).unwrap();
        assert!(changed.contains(expected), "{path}: {changed}");
    }
}

#[test]
fn ensure_import_rejects_syntax_breaking_projection() {
    let ws = workspace();
    let original =
        b"package main\n\nimport (\n\t\"fmt\"\n)\n\nfunc main() { fmt.Println(\"ok\") }\n";
    let file = write(ws.path(), "m.go", original);

    let certificate = ensure_import(
        ws.path(),
        &file,
        "bad\npath",
        None,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(certificate.status, Status::InvalidResult);
    assert_eq!(certificate.exit_code(), 13);
    assert!(!certificate.published);
    assert!(
        certificate.operations[0].syntax.new_errors > 0
            || certificate.operations[0].syntax.new_missing_nodes > 0
    );
    assert_eq!(std::fs::read(&file).unwrap(), original);
}

#[test]
fn ensure_annotation_is_idempotent() {
    let ws = workspace();
    let original = b"def run():\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let first = ensure_annotation(
        ws.path(),
        &file,
        (0, original.len() - 1),
        "@retry",
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(first.status, Status::Applied);
    compact(&first);

    let after_first = std::fs::read(&file).unwrap();
    let start = std::str::from_utf8(&after_first)
        .unwrap()
        .find("def run")
        .unwrap();
    let second = ensure_annotation(
        ws.path(),
        &file,
        (start, after_first.len() - 1),
        "@retry",
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(second.status, Status::AlreadySatisfied);
    assert_eq!(second.operations[0].target_matches, 1);
    assert_eq!(std::fs::read(&file).unwrap(), after_first);
    compact(&second);
}

#[test]
fn ensure_annotation_invalid_scope_and_stale_are_atomic() {
    let ws = workspace();
    let original = b"def run():\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let invalid = ensure_annotation(
        ws.path(),
        &file,
        (0, original.len() + 1),
        "@retry",
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(invalid.status, Status::NotFound);
    assert_eq!(std::fs::read(&file).unwrap(), original);

    let options = planned_options(ws.path(), "m.py", original, (0, original.len() - 1));
    let live = b"# concurrent\ndef run():\n    pass\n";
    std::fs::write(&file, live).unwrap();
    let stale = ensure_annotation(
        ws.path(),
        &file,
        (0, original.len() - 1),
        "@retry",
        &options,
    )
    .unwrap();
    assert_eq!(stale.status, Status::Stale);
    assert_eq!(std::fs::read(&file).unwrap(), live);
    compact(&stale);
}

#[test]
fn ensure_argument_is_idempotent() {
    let ws = workspace();
    let original = b"def run():\n    fetch(url)\n";
    let file = write(ws.path(), "m.py", original);
    let first = ensure_argument(
        ws.path(),
        &file,
        (0, original.len()),
        "fetch",
        "timeout=30",
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(first.status, Status::Applied);
    compact(&first);

    let after_first = std::fs::read(&file).unwrap();
    let second = ensure_argument(
        ws.path(),
        &file,
        (0, after_first.len()),
        "fetch",
        "timeout=30",
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(second.status, Status::AlreadySatisfied);
    assert_eq!(second.operations[0].target_matches, 1);
    assert_eq!(std::fs::read(&file).unwrap(), after_first);
    compact(&second);
}

#[test]
fn ensure_argument_missing_target_and_stale_are_atomic() {
    let ws = workspace();
    let original = b"def run():\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let missing = ensure_argument(
        ws.path(),
        &file,
        (0, original.len()),
        "fetch",
        "timeout=30",
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(missing.status, Status::NotFound);
    assert_eq!(std::fs::read(&file).unwrap(), original);

    let planned = b"def run():\n    fetch(url)\n";
    std::fs::write(&file, planned).unwrap();
    let options = planned_options(ws.path(), "m.py", planned, (0, planned.len()));
    let live = b"# concurrent\ndef run():\n    fetch(url)\n";
    std::fs::write(&file, live).unwrap();
    let stale = ensure_argument(
        ws.path(),
        &file,
        (0, planned.len()),
        "fetch",
        "timeout=30",
        &options,
    )
    .unwrap();
    assert_eq!(stale.status, Status::Stale);
    assert_eq!(
        stale.operations[0].selector_class,
        SelectorClass::Structural
    );
    assert_eq!(std::fs::read(&file).unwrap(), live);
    compact(&stale);
}

#[test]
fn ensure_method_inserts_inside_class_and_is_idempotent() {
    let ws = workspace();
    let original = b"class Worker:\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let source = "    def ping(self):\n        return 1";
    let first = ensure_method(
        ws.path(),
        &file,
        (0, original.len()),
        "ping",
        source,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(first.status, Status::Applied, "{first:?}");
    compact(&first);
    let after_first = std::fs::read(&file).unwrap();
    let text = std::str::from_utf8(&after_first).unwrap();
    assert!(
        text.contains("class Worker:\n    pass\n    def ping"),
        "{text}"
    );

    let second = ensure_method(
        ws.path(),
        &file,
        (0, after_first.len()),
        "ping",
        source,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(second.status, Status::AlreadySatisfied, "{second:?}");
    assert_eq!(second.operations[0].target_matches, 1);
    assert_eq!(std::fs::read(&file).unwrap(), after_first);
    compact(&second);
}

#[test]
fn ensure_method_ignores_nested_same_name() {
    let ws = workspace();
    let original = b"class Worker:\n    def outer(self):\n        def ping():\n            return 0\n        return ping()\n";
    let file = write(ws.path(), "m.py", original);
    let certificate = ensure_method(
        ws.path(),
        &file,
        (0, original.len()),
        "ping",
        "    def ping(self):\n        return 1",
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(certificate.status, Status::Applied, "{certificate:?}");
    let output = std::fs::read_to_string(&file).unwrap();
    assert_eq!(output.matches("def ping").count(), 2, "{output}");
    compact(&certificate);
}

#[test]
fn ensure_method_invalid_scope_and_stale_are_atomic() {
    let ws = workspace();
    let original = b"class Worker:\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let source = "    def ping(self):\n        return 1";
    let invalid = ensure_method(
        ws.path(),
        &file,
        (0, original.len() + 1),
        "ping",
        source,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(invalid.status, Status::NotFound);
    assert_eq!(std::fs::read(&file).unwrap(), original);

    let options = planned_options(ws.path(), "m.py", original, (0, original.len()));
    let live = b"# concurrent\nclass Worker:\n    pass\n";
    std::fs::write(&file, live).unwrap();
    let stale = ensure_method(
        ws.path(),
        &file,
        (0, original.len()),
        "ping",
        source,
        &options,
    )
    .unwrap();
    assert_eq!(stale.status, Status::Stale);
    assert_eq!(std::fs::read(&file).unwrap(), live);
    compact(&stale);
}

#[test]
fn remove_if_present_is_idempotent() {
    let ws = workspace();
    let original = b"def gone():\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let first = remove_if_present(
        ws.path(),
        Some((file.clone(), (0, original.len() - 1))),
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(first.status, Status::Applied);
    compact(&first);

    let after_first = std::fs::read(&file).unwrap();
    let second = remove_if_present(ws.path(), None, &VerbOptions::default()).unwrap();
    assert_eq!(second.status, Status::AlreadySatisfied);
    assert_eq!(second.exit_code(), 0);
    assert_eq!(std::fs::read(&file).unwrap(), after_first);
    compact(&second);
}

#[test]
fn remove_if_present_honors_planned_hashes() {
    let ws = workspace();
    let original = b"def gone():\n    pass\n";
    let file = write(ws.path(), "m.py", original);
    let range = (0, original.len() - 1);
    let options = planned_options(ws.path(), "m.py", original, range);
    let live = b"# concurrent\ndef gone():\n    pass\n";
    std::fs::write(&file, live).unwrap();
    let certificate = remove_if_present(ws.path(), Some((file.clone(), range)), &options).unwrap();
    assert_eq!(certificate.status, Status::Stale);
    assert_eq!(std::fs::read(&file).unwrap(), live);
    compact(&certificate);
}
