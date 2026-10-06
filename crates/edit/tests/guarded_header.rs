use greppy_edit::txn::{
    first_syntax_diagnostic, syntax_counts, syntax_language_for_edit, syntax_language_for_path,
};
use std::path::Path;

#[test]
fn guarded_protocol_header_preserves_syntax_gate() {
    let valid = include_bytes!("fixtures/guarded-protocol.h");
    let language = syntax_language_for_path(Path::new("protocol.h"), b"");
    let baseline = syntax_counts(language, b"").unwrap();
    assert_eq!(syntax_counts(language, valid).unwrap(), baseline);
    assert!(first_syntax_diagnostic(language, valid).is_none());
    let text = std::str::from_utf8(valid).unwrap();
    for malformed in [
        text.replacen(
            "fma_codec_name(uint32_t codec);",
            "fma_codec_name(uint32_t codec;",
            1,
        ),
        text.replacen("extern \"C\" {", "extern \"C\"", 1),
        text.replacen("\n}\n#endif", "\n#endif", 1),
        text.trim_end().strip_suffix("#endif").unwrap().to_string(),
        text.replacen("extern \"C\" {\n#endif", "extern \"C\" {", 1),
    ] {
        let counts = syntax_counts(language, malformed.as_bytes()).unwrap();
        assert!(
            counts.errors > baseline.errors || counts.missing > baseline.missing,
            "{malformed}"
        );
        assert!(first_syntax_diagnostic(language, malformed.as_bytes()).is_some());
    }
}

#[test]
fn compatible_header_can_add_cpp_default_member_without_weakening_c() {
    let path = Path::new("member.h");
    let before = b"struct Example { int value; };\n";
    let after = b"struct Example { int value{}; };\n";
    for baseline in [b"".as_slice(), before.as_slice()] {
        let language = syntax_language_for_edit(path, baseline, after);
        assert_eq!(language, greppy_parser::Language::Cpp);
        for bytes in [baseline, after.as_slice()] {
            let counts = syntax_counts(language, bytes).unwrap();
            assert_eq!((counts.errors, counts.missing), (0, 0));
        }
    }
    let c = b"struct AtomicState { _Atomic(int) value; };\n";
    assert_eq!(
        syntax_language_for_edit(path, before, c),
        greppy_parser::Language::C
    );
    assert_eq!(
        syntax_language_for_edit(Path::new("member.c"), before, after),
        greppy_parser::Language::C
    );
    let broken = b"struct Example { int value{; };\n";
    let language = syntax_language_for_edit(path, before, broken);
    let counts = syntax_counts(language, broken).unwrap();
    assert!(counts.errors > 0 || counts.missing > 0);
}

#[test]
fn c11_atomic_type_specifiers_preserve_real_errors_and_expression_rules() {
    let c = greppy_parser::Language::C;
    for source in [
        "_Atomic(int) value;",
        "struct AtomicState { _Atomic(int) value; };",
        "_Atomic(unsigned long *) value;",
        "typedef int MyType; _Atomic(MyType) value;",
        "void f(void) { _Atomic(int) local; }",
        "_Atomic int value;",
        "_Atomic /* type comment */ (int) value;",
        "_Atomic int first; _Atomic(int) second;",
        "const char *s = \"_Atomic(int)\"; /* _Atomic() */",
    ] {
        let counts = syntax_counts(c, source.as_bytes()).unwrap();
        assert_eq!((counts.errors, counts.missing), (0, 0), "{source}");
        assert!(first_syntax_diagnostic(c, source.as_bytes()).is_none());
    }
    for source in [
        "_Atomic() value;",
        "_Atomic(int value;",
        "_Atomic(int named) value;",
        "_Atomic(1 + 2) value;",
        "struct AtomicState { _Atomic(int) value; }; int broken( ;",
    ] {
        let counts = syntax_counts(c, source.as_bytes()).unwrap();
        assert!(counts.errors > 0 || counts.missing > 0, "{source}");
        assert!(
            first_syntax_diagnostic(c, source.as_bytes()).is_some(),
            "{source}"
        );
    }
}
