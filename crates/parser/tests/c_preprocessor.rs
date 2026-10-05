use greppy_parser::{
    c_preprocessor::c_preprocessor_validation_view, parse, parse_for_syntax_validation, Language,
};

#[test]
fn local_jni_qualifiers_and_xmacro_fields_validate_without_changing_source() {
    for source in [
        "#define JNIEXPORT __attribute__((visibility(\"default\")))\n#define JNICALL\nJNIEXPORT int JNICALL probe(void) { return 0; }\n",
        "#define FIELDS(X) X(int, count) X(float, ratio)\n#define DECL(type, name) type name;\nstruct record { FIELDS(DECL) };\n",
        "#define FIELDS(X) \\\n X(int, count) \\\n X(float, ratio)\n#define DECL(type, name) type name;\nstruct record { FIELDS(DECL) };\n",
    ] {
        let original = source.as_bytes().to_vec();
        let view = c_preprocessor_validation_view(&original).unwrap();
        let tree = parse_for_syntax_validation(Language::C, &view.bytes).unwrap();
        assert!(!tree.root_node().has_error(), "{source}\n{}", tree.root_node().to_sexp());
        assert_eq!(original, source.as_bytes());
        // Extraction API receives the original source, never the expanded view.
        let mut raw = tree_sitter::Parser::new();
        raw.set_language(&Language::C.grammar()).unwrap();
        let raw_tree = raw.parse(&original, None).unwrap();
        assert_eq!(parse(Language::C, &original).unwrap().root_node().to_sexp(), raw_tree.root_node().to_sexp());
    }
}

#[test]
fn unsupported_used_macros_fail_at_the_invocation() {
    for source in [
        "#define JOIN(a,b) a ## b\nint JOIN(a,b);\n",
        "#define STR(a) #a\nconst char *x = STR(a);\n",
        "#define MANY(...) int x;\nMANY(x)\n",
        "#ifdef SOME_BUILD\n#define EXPORT\n#endif\nEXPORT int x;\n",
    ] {
        let error = match c_preprocessor_validation_view(source.as_bytes()) {
            Ok(_) => panic!("unsupported used macro accepted: {source}"),
            Err(error) => error,
        };
        let last_line = source.lines().last().unwrap();
        let line_at = source.rfind(last_line).unwrap();
        assert!(error.offset >= line_at && error.offset < line_at + last_line.len());
        assert!(error.reason.contains("compiler preprocessing"));
    }
}

#[test]
fn expansion_keeps_literals_comments_numbers_and_source_mapping() {
    let source =
        b"#define NAME int\n#define u8 bad\nconst char *s = u8\"NAME\"; /* NAME */\nNAME value;\n";
    let view = c_preprocessor_validation_view(source).unwrap();
    let text = std::str::from_utf8(&view.bytes).unwrap();
    assert!(text.contains("u8\"NAME\"; /* NAME */"));
    assert!(text.contains("int  value;"));
    let source_at = source.windows(5).position(|s| s == b"value").unwrap();
    let view_at = view.bytes.windows(5).position(|s| s == b"value").unwrap();
    assert_eq!(view.source_offset(view_at), source_at);
    assert_eq!(view.expanded_offset(source_at), view_at);
    let number = b"#define ABC 1\nint x = 123ABC;";
    assert!(c_preprocessor_validation_view(number)
        .unwrap()
        .bytes
        .ends_with(b"123ABC;"));
}

#[test]
fn broken_expansions_and_wrong_arity_do_not_become_valid() {
    let source = b"#define DECL(t,n) t n\nstruct item { DECL(int,value) };";
    let view = c_preprocessor_validation_view(source).unwrap();
    assert!(parse_for_syntax_validation(Language::C, &view.bytes)
        .unwrap()
        .root_node()
        .has_error());
    let source = b"#define DECL(t,n) t n;\nstruct item { DECL(int,value,extra) };";
    let error = c_preprocessor_validation_view(source).err().unwrap();
    assert!(error.reason.contains("argument count"));
}

#[test]
fn nested_arguments_and_source_ordered_redefinitions_are_respected() {
    let source =
        b"#define ID(x) x\nID(ID(int)) first;\n#undef ID\n#define ID(x) float\nID(int) second;\n";
    let view = c_preprocessor_validation_view(source).unwrap();
    assert!(!parse_for_syntax_validation(Language::C, &view.bytes)
        .unwrap()
        .root_node()
        .has_error());
}

#[test]
fn macro_substitution_does_not_paste_distinct_operator_tokens() {
    let source = b"#define PLUS(x) +x\nint f(int n){return PLUS(+n);}";
    let view = c_preprocessor_validation_view(source).unwrap();
    assert!(!std::str::from_utf8(&view.bytes).unwrap().contains("++"));
    assert!(!parse_for_syntax_validation(Language::C, &view.bytes)
        .unwrap()
        .root_node()
        .has_error());
}
