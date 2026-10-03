use greppy_edit::txn::{first_syntax_diagnostic, syntax_counts, syntax_language_for_path};
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
