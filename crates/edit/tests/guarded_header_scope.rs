use greppy_edit::txn::syntax_counts;
use greppy_parser::Language;

#[test]
fn guarded_linkage_does_not_hide_wrong_scope_or_lookalikes() {
    let pair =
        "#ifdef __cplusplus\nextern \"C\" {\n#endif\nint value;\n#ifdef __cplusplus\n}\n#endif\n";
    for source in [
        format!("void f(void) {{\n{pair}}}\n"),
        format!("struct S {{\n{pair}}};\n"),
        pair.replacen("int value;", "#endif\nint value;\n#ifdef OTHER", 1),
    ] {
        let counts = syntax_counts(Language::C, source.as_bytes()).unwrap();
        assert!(counts.errors > 0 || counts.missing > 0, "{source}");
    }
    let commented = format!("/*\n{pair}*/\nint broken( ;\n");
    let counts = syntax_counts(Language::C, commented.as_bytes()).unwrap();
    assert!(counts.errors > 0 || counts.missing > 0);
}
